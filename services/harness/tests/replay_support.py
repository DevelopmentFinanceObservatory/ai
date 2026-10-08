"""What the `harness.recipe_replay` and `harness.live_refresh` tests share:
dfo-apis' publish door, replay lookup and versions route as a fake behind
`httpx2.MockTransport`, its mint of one dataset as a canned upstream, and frames
of that dataset at two versions.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
from collections.abc import Mapping
from typing import Any

import httpx2
import pyarrow as pa
import pyarrow.parquet as pq
from common.live_apis import PRINCIPAL_HEADER, LiveApisClient
from common.materialisation import FrameRef
from dfo_frame_grammar import Frame
from presentation_mcp.client import PlatformAssembler
from presentation_mcp.settings import Settings as PresentationSettings

PRINCIPAL = "datasets-principal-7f3a"
SPEC = "spec_" + "a" * 32
FIRST = "cv_" + "1" * 32
SECOND = "cv_" + "2" * 32
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


def problem(status: int, code: str, **members: object) -> httpx2.Response:
    return httpx2.Response(
        status,
        json={"code": code, "retryable": False, "detail": "refused", **members},
        headers={"content-type": "application/problem+json"},
    )


class FakeApis:
    """dfo-apis' publish door, replay lookup and versions route over plain dicts.

    `frames` holds each published pair's rows, `recipes` its recipe, and `versions`
    each dataset's listed versions. A pair keeps its first recipe, and a replay key
    names one frame, as the publish door has it.
    """

    def __init__(self) -> None:
        self.frames: dict[Pair, list[dict[str, Any]]] = {}
        self.recipes: dict[Pair, dict[str, Any]] = {}
        self.versions: dict[str, list[dict[str, Any]]] = {}
        self.requests: list[httpx2.Request] = []
        self.publish_answer: httpx2.Response | None = None

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        if request.headers.get(PRINCIPAL_HEADER) != PRINCIPAL:
            return problem(403, "publish_not_permitted")
        path = request.url.path
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
        if path.endswith("/versions"):
            spec_id = path.split("/")[-2]
            return httpx2.Response(
                200,
                json={
                    "schema_version": 1,
                    "spec_id": spec_id,
                    "versions": self.versions.get(spec_id, []),
                },
            )
        raise AssertionError(f"unexpected call to {path}")

    def _publish(self, body: dict[str, Any]) -> httpx2.Response:
        key = (body["spec_id"], body["content_version"])
        created = key not in self.frames
        self.frames.setdefault(key, body["rows"])
        recipe = body.get("recipe")
        if recipe is not None and key not in self.recipes:
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
    """dfo-apis' mint for one dataset: the current version's manifest, and its
    frame file by artifact digest. `move` serves other rows as the next version."""

    def __init__(self) -> None:
        self.content_version = FIRST
        self._files: dict[str, bytes] = {}
        self._digest = self._stored(FIRST_VALUES)
        self.mints = 0

    def _stored(self, values: Mapping[str, Mapping[str, float]]) -> str:
        buf = io.BytesIO()
        pq.write_table(pa.Table.from_pylist(reporter_rows(values)), buf)
        digest = f"sha256:{hashlib.sha256(buf.getvalue()).hexdigest()}"
        self._files[digest] = buf.getvalue()
        return digest

    def move(self, values: Mapping[str, Mapping[str, float]], *, to: str) -> None:
        self.content_version = to
        self._digest = self._stored(values)

    def handler(self, request: httpx2.Request) -> httpx2.Response:
        assert request.url.path == "/api/datasets", request.url.path
        self.mints += 1
        return httpx2.Response(
            200,
            json={
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
                        "url": "https://signed.example/should-not-be-fetched",
                        "url_expires_at": "2026-08-11T00:15:00+00:00",
                    }
                ],
            },
        )

    async def fetch(self, *, artifact_digest: str, **_: Any) -> bytes:
        return self._files[artifact_digest]
