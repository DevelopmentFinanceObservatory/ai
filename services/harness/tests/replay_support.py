"""What the `harness.recipe_replay` and `harness.live_refresh` tests share:
the apis service's publish door, replay lookup and versions route as a fake behind
`httpx2.MockTransport`, its mint and snapshot of one dataset as a canned upstream,
and frames of that dataset at two versions, each version named by the token its
canonical frame mints.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import re
from collections.abc import Callable, Mapping
from typing import Any

import httpx2
import pyarrow as pa
import pyarrow.parquet as pq
from common.live_apis import PRINCIPAL_HEADER, LiveApisClient
from common.materialisation import FrameRef, frame_to_parquet_bytes
from data_mcp.client import DATASETS_PREFIX, GATED_PREFIXES, UpstreamClient
from data_mcp.settings import Settings as DataMcpSettings
from dfo_frame_grammar import Frame, frame_sha256, mint_content_version
from presentation_mcp.client import PlatformAssembler
from presentation_mcp.settings import Settings as PresentationSettings

PRINCIPAL = "datasets-principal-7f3a"
SPEC = "spec_" + "a" * 32
OPTIONS: dict[str, Any] = {"format": "parquet", "bundle": False, "layout": "frame"}
REPORTERS = ("dcid:FRA", "dcid:GBR", "dcid:USA")
STAT_VAR = "dcid:OfficialDevelopmentAssistanceNet_Disbursement"

Pair = tuple[str, str]


def pair(ref: FrameRef) -> Pair:
    return ref.spec_id, ref.content_version


def reporter_rows(
    values: Mapping[str, Mapping[str, float]], *, stat_var: str = STAT_VAR
) -> list[dict[str, Any]]:
    """One row for each reporter and year of `values`, `{reporter: {year: value}}`."""
    return [
        {
            "stat_var": stat_var,
            "reporter": reporter,
            "time_period": year,
            "value": value,
            "unit": "dcid:USDollar",
        }
        for reporter, by_year in values.items()
        for year, value in by_year.items()
    ]


def reporter_frame(values: Mapping[str, Mapping[str, float]]) -> Frame:
    return Frame.build(role_keys=["reporter"], rows=reporter_rows(values))


def yearly(years: range, *, start: float) -> dict[str, float]:
    """A value for each year of `years`, rising by 10 from `start`."""
    return {str(year): start + 10.0 * index for index, year in enumerate(years)}


# Three reporters, 2019 to 2022, and the same data a year on with other values.
FIRST_VALUES = {
    reporter: yearly(range(2019, 2023), start=100.0 * (index + 1))
    for index, reporter in enumerate(REPORTERS)
}
SECOND_VALUES = {
    reporter: yearly(range(2019, 2024), start=150.0 * (index + 1))
    for index, reporter in enumerate(REPORTERS)
}


def version_of(values: Mapping[str, Mapping[str, float]]) -> str:
    """The `content_version` apis mints for the frame of `values`."""
    return mint_content_version(frame_sha256=frame_sha256(reporter_frame(values)))


FIRST = version_of(FIRST_VALUES)
SECOND = version_of(SECOND_VALUES)


_DESCRIPTION_PATH = re.compile(r"/api/datasets/[^/]+/[^/]+/description")
DATASETS_CAPABILITIES = GATED_PREFIXES[DATASETS_PREFIX]


def problem(status: int, code: str, **members: object) -> httpx2.Response:
    return httpx2.Response(
        status,
        json={"code": code, "retryable": False, "detail": "refused", **members},
        headers={"content-type": "application/problem+json"},
    )


def data_mcp_client(handler: Callable[[httpx2.Request], httpx2.Response]) -> UpstreamClient:
    """data-mcp's upstream client over `handler`.

    A version's description (`GET /api/datasets/{id}/{content_version}/description`)
    answers 404 `not_found`, as apis does for a version stored without a record, so
    `get_data` renders its digest without a description. The Datasets capabilities route
    answers 200, so the client's probe after that 404 finds the surface mounted. Every
    other request reaches `handler`.
    """

    def answer(request: httpx2.Request) -> httpx2.Response:
        path = request.url.path
        if path == DATASETS_CAPABILITIES:
            return httpx2.Response(200, json={})
        if request.method == "GET" and _DESCRIPTION_PATH.fullmatch(path):
            return problem(404, "not_found")
        return handler(request)

    return UpstreamClient(settings=DataMcpSettings(), transport=httpx2.MockTransport(answer))


class FakeApis:
    """The apis service's publish door, replay lookup and versions route over plain dicts.

    `frames` holds each published pair's rows, `recipes` its recipe, `versions`
    each dataset's listed versions, and `enrichment_stamp` the stamp the versions
    answer lists, none when `None`. A pair keeps its first recipe and refuses one
    with another normal form or replay key, and a replay key names one frame, as
    apis' recipe store has it.
    """

    def __init__(self) -> None:
        self.frames: dict[Pair, list[dict[str, Any]]] = {}
        self.recipes: dict[Pair, dict[str, Any]] = {}
        self.versions: dict[str, list[dict[str, Any]]] = {}
        self.enrichment_stamp: str | None = None
        self.requests: list[httpx2.Request] = []
        self.publish_answer: httpx2.Response | None = None

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        path = request.url.path
        # The versions route is open to every caller and answers in one page here.
        if path.endswith("/versions"):
            spec_id = path.split("/")[-2]
            return httpx2.Response(
                200,
                json={
                    "schema_version": 1,
                    "spec_id": spec_id,
                    "versions": self.versions.get(spec_id, []),
                    "next_cursor": None,
                    "complete": True,
                }
                | (
                    {}
                    if self.enrichment_stamp is None
                    else {"enrichment_stamp": self.enrichment_stamp}
                ),
            )
        if request.headers.get(PRINCIPAL_HEADER) != PRINCIPAL:
            return problem(403, "publish_not_permitted")
        if path == "/api/datasets/frames/publish":
            return self.publish_answer or self._publish(
                json.loads(gzip.decompress(request.content))
            )
        if path.startswith("/api/datasets/recipes/replays/"):
            key = path.rsplit("/", 1)[1]
            held = [pair for pair, recipe in self.recipes.items() if recipe["replay_key"] == key]
            if not held:
                return problem(404, "recipe_not_found")
            return httpx2.Response(200, json={"spec_id": held[0][0], "content_version": held[0][1]})
        if path.startswith("/api/datasets/frames/") and path.endswith("/recipe"):
            spec_id, content_version = path.split("/")[-3:-1]
            return self._recipe((spec_id, content_version))
        raise AssertionError(f"unexpected call to {path}")

    def _recipe(self, key: Pair) -> httpx2.Response:
        held = self.recipes.get(key)
        if held is None:
            return problem(404, "recipe_not_found")
        return httpx2.Response(200, json=held)

    def _publish(self, body: dict[str, Any]) -> httpx2.Response:
        key = (body["spec_id"], body["content_version"])
        created = key not in self.frames
        self.frames.setdefault(key, body["rows"])
        recipe = body.get("recipe")
        if recipe is not None and key in self.recipes:
            for field in ("normal_form_hash", "replay_key"):
                if self.recipes[key][field] != recipe[field]:
                    return problem(
                        409, "publish_conflict", constraint_id=f"recipe_requires_matching_{field}"
                    )
        elif recipe is not None:
            if any(held["replay_key"] == recipe["replay_key"] for held in self.recipes.values()):
                return problem(
                    409, "publish_conflict", constraint_id="replay_key_requires_one_frame"
                )
            self.recipes[key] = recipe
        return httpx2.Response(
            200,
            json={
                "schema_version": 1,
                "published": {
                    "spec_id": key[0],
                    "content_version": key[1],
                    "artifact_digest": "sha256:fake",
                    "created": created,
                },
            },
        )

    def published(self) -> list[httpx2.Request]:
        """Every request the publish door received."""
        return [r for r in self.requests if r.url.path == "/api/datasets/frames/publish"]

    def client(self) -> LiveApisClient:
        async def no_sleep(_: float) -> None:
            return None

        return LiveApisClient(
            base_url="http://apis.test",
            datasets_principal=PRINCIPAL,
            transport=httpx2.MockTransport(self),
            sleep=no_sleep,
        )

    def assembler(self) -> PlatformAssembler:
        """presentation-mcp's client over this fake, whose `publish_frame` is the
        replayer's publisher."""
        return PlatformAssembler(
            settings=PresentationSettings(
                datasets_first_party_principal=PRINCIPAL, upstream_backoff_base_s=0.01
            ),
            transport=httpx2.MockTransport(self),
        )


class Upstream:
    """apis' mint for one dataset: the current version's manifest, its frame
    file by artifact digest, and the version's snapshot, which lists its canonical
    frame beside the frame file. `move` serves other rows as the next version."""

    def __init__(self) -> None:
        self.content_version = FIRST
        self._files: dict[str, bytes] = {}
        self._digest, self._canonical_digest = self._stored(FIRST_VALUES)
        self.mints = 0

    def _stored(self, values: Mapping[str, Mapping[str, float]]) -> tuple[str, str]:
        buf = io.BytesIO()
        pq.write_table(pa.Table.from_pylist(reporter_rows(values)), buf)
        canonical = frame_to_parquet_bytes(reporter_frame(values))
        digests = []
        for data in (buf.getvalue(), canonical):
            digest = f"sha256:{hashlib.sha256(data).hexdigest()}"
            self._files[digest] = data
            digests.append(digest)
        return digests[0], digests[1]

    def move(self, values: Mapping[str, Mapping[str, float]]) -> str:
        """Serve `values` as the next version, and return its `content_version`."""
        self.content_version = version_of(values)
        self._digest, self._canonical_digest = self._stored(values)
        return self.content_version

    def handler(self, request: httpx2.Request) -> httpx2.Response:
        if request.url.path == f"/api/datasets/{SPEC}/{self.content_version}":
            return httpx2.Response(200, json=self._snapshot())
        assert request.url.path == "/api/datasets", request.url.path
        self.mints += 1
        return httpx2.Response(200, json=self._manifest())

    def _snapshot(self) -> dict[str, Any]:
        manifest = self._manifest()
        listed = manifest["artifacts"][0]
        manifest["artifacts"].append(
            listed
            | {
                "artifact_digest": self._canonical_digest,
                "rendering": {"format": "frame"},
                "bytes": len(self._files[self._canonical_digest]),
                "url": f"/api/datasets/{SPEC}/{self.content_version}/files/canonical",
            }
        )
        return manifest

    def _manifest(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "definition": {
                "schema_version": 1,
                "spec_id": SPEC,
                "identity": {
                    "identity_version": "spec1",
                    "shape": {"id": "OdaNetDisbursement", "kind": "stat_var"},
                    "entities": {"reporter": list(REPORTERS)},
                    "facets": {"unit": {"ref": "USD", "state": "value"}},
                    "requested_window": {"kind": "all"},
                    "slots": [],
                },
            },
            "content_version": self.content_version,
            "coverage": {"rows": 12, "dimensions": [{"label": "years", "count": 4}]},
            "citation": {"text": "OECD DAC", "sources": []},
            "trace": {"graph_revision": "g1", "facets": []},
            "warnings": [],
            "artifacts": [
                {
                    "artifact_digest": self._digest,
                    "rendering": {"bundle": False, "format": "parquet", "layout": "frame"},
                    "bytes": len(self._files[self._digest]),
                    "media_type": "application/vnd.apache.parquet",
                    "created_at": "2026-08-11T00:00:00+00:00",
                    "tier": "working",
                    "url": "/links/should-not-be-fetched",
                }
            ],
        }

    async def fetch(self, *, artifact_digest: str, **_: Any) -> bytes:
        return self._files[artifact_digest]
