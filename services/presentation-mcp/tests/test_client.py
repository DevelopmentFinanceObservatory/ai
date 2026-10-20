"""`PlatformAssembler` tests: the wire shape of each of the five calls, the
`WriteReply`/`Refusal` translation, and the retry loop. `httpx2.MockTransport`
throughout, so `.net_guard`'s `no_network` fixture is what proves no test
here reaches a real socket rather than merely relying on the mock.
"""

from __future__ import annotations

import asyncio
import contextlib
import copy
import dataclasses
import gzip
import hashlib
import json
import logging
import os
import re
import threading
from collections.abc import AsyncIterator, Awaitable, Callable, Coroutine, Iterator, Mapping
from typing import Any

import httpx2
import pytest
import rfc8785
from common.apis_auth import ApisAuthError
from common.materialisation import (
    FrameIntegrityError,
    FrameRef,
    GcsMaterialisationStore,
    HeldFrame,
    InMemoryMaterialisationStore,
    MintRecord,
    StoredFrameTrace,
    gcs_store,
    get_store,
    register_trace_codec,
    set_store,
)
from common.recipe import build_recipe
from common.row_budget import ROW_BUDGET_RETRY_AFTER_S, RowBudget, row_budget, set_row_budget
from dfo_frame_grammar import Frame
from presentation_mcp import client as client_module
from presentation_mcp import frame_publish
from presentation_mcp.client import PRINCIPAL_HEADER, WRITE_TOKEN_HEADER, PlatformAssembler
from presentation_mcp.errors import PlatformError, PlatformTransportError
from presentation_mcp.frame_publish import PublishBody
from presentation_mcp.platform_contract import (
    Caller,
    NormalFormConflict,
    Refusal,
    RetryMode,
    WriteReply,
)
from presentation_mcp.settings import Settings

from .net_guard import no_network  # noqa: F401 -- imported for its autouse fixture

# A well-formed `ops_`/`ocv_` pair, fitting `FrameRef`'s own strict
# `[0-9a-f]{32}` id pattern -- unlike `FRAME_REF`/`op_frame_ref` elsewhere in
# this module, which use short ids that only need to satisfy this client's
# own looser path-segment pattern.
_OPS_SPEC_ID = "ops_" + "a" * 32
_OCV_CONTENT_VERSION = "ocv_" + "b" * 32


@dataclasses.dataclass(frozen=True, kw_only=True)
class _StubTrace:
    """A minimal stand-in for `analysis_mcp.trace.Trace`: satisfies
    `common.materialisation.trace.StoredFrameTrace` structurally, plus the
    `input_content_versions` field `frame_publish._input_content_versions`
    reads duck-typed off the real thing, and the `steps` `common.steps
    .assemble_steps` reads: one `_StubStep`, or none for a trace stored before
    steps were.
    """

    upstream_trace: dict[str, object]
    frame_content_version: str
    input_content_versions: tuple[str, ...]
    steps: tuple[_StubStep, ...] = ()


@dataclasses.dataclass(frozen=True)
class _StubStep:
    """A stand-in for `analysis_mcp.steps.Step`: the JSON form of one step."""

    record: dict[str, Any]

    def model_dump(self, *, mode: str) -> dict[str, Any]:
        return copy.deepcopy(self.record)


@dataclasses.dataclass(frozen=True)
class _StubIdentity:
    def canonical_string(self) -> str:
        return "0.3.0|1.0.0|0.2.0"


@dataclasses.dataclass(frozen=True)
class _StubExecutionRecord:
    content: str


@dataclasses.dataclass(frozen=True, kw_only=True)
class _DerivedTrace(_StubTrace):
    """`_StubTrace` with what `common.recipe.build_recipe` reads off a real trace."""

    execution_record: _StubExecutionRecord
    execution_identity: _StubIdentity = _StubIdentity()


@pytest.fixture
def materialisation_store() -> Iterator[InMemoryMaterialisationStore]:
    """A fresh in-process frame store for one test, torn down after --
    `common.materialisation`'s own holder is a process-wide singleton, so a
    test that installs one must clear it, or a later test in this same
    module would inherit it.
    """
    store = InMemoryMaterialisationStore()
    set_store(store)
    try:
        yield store
    finally:
        set_store(None)


async def _put_ops_frame(
    store: InMemoryMaterialisationStore,
    *,
    spec_id: str = _OPS_SPEC_ID,
    content_version: str = _OCV_CONTENT_VERSION,
    input_content_versions: tuple[str, ...] = ("cv_" + "c" * 32,),
    execution_record: _StubExecutionRecord | None = None,
    step: dict[str, Any] | None = None,
) -> None:
    """Stores a one-row frame under an `ops_`/`ocv_` pair. With `execution_record`
    its trace also carries what a recipe is built from, and with `step` the
    record of its own operation."""
    frame = Frame.build(
        role_keys=["recipientCountry"],
        rows=[
            {
                "stat_var": "dcid:health_oda",
                "recipientCountry": "country/ETH",
                "time_period": "2024",
                "value": 12.5,
            }
        ],
    )
    trace: _StubTrace = _StubTrace(
        upstream_trace={},
        frame_content_version=content_version,
        input_content_versions=input_content_versions,
        steps=() if step is None else (_StubStep(step),),
    )
    if execution_record is not None:
        trace = _DerivedTrace(**vars(trace), execution_record=execution_record)
    await store.put(
        FrameRef(spec_id=spec_id, content_version=content_version),
        frame=frame,
        trace=trace,
        writer="test",
    )


Handler = (
    Callable[[httpx2.Request], httpx2.Response]
    | Callable[[httpx2.Request], Coroutine[None, None, httpx2.Response]]
)

FRAME_REF = {"spec_id": "spec_7c2f91", "content_version": "cv_4a91e2"}

_INVALID_COMPONENT_IDS = (
    "",
    "   ",
    "component_1",
    "../../datasets/spec_1/cv_1",
    "cmp_bad/segment",
    "cmp_bad?revision=1",
    "cmp_bad#fragment",
    "cmp_%2Fdatasets",
    f"cmp_{'a' * 65}",
)
_INVALID_SPEC_IDS = (
    "",
    "spec_../components/cmp_1",
    "spec_bad/segment",
    "spec_bad?query=1",
    "spec_bad#fragment",
    "spec_%2Fcomponents",
    f"spec_{'a' * 65}",
)
_INVALID_CONTENT_VERSIONS = (
    "",
    "cv_../components/cmp_1",
    "cv_bad/segment",
    "cv_bad?query=1",
    "cv_bad#fragment",
    "cv_%2Fcomponents",
    f"cv_{'a' * 65}",
)

INTENT = {
    "type": "line_chart",
    "frame_ref": FRAME_REF,
    "select": {"statvar": "health_oda", "recipient": "country/ETH"},
}

# The platform's stamped envelope, abridged. The client forwards it unread.
ENVELOPE = {
    "schema_version": 1,
    "catalog_version": "2026.8.0",
    "component_id": "cmp_1",
    "revision": 1,
    "provenance": {"inputs": [FRAME_REF]},
    "component": {"type": "line_chart", "series": [{"points": [["2015", 1.5]]}]},
}

WRITE_TOKEN = "cmw_Zm9vYmFyYmF6cXV4cXV1"

# What `create_component` mints: `cmw_` and 22 url-safe characters.
_CREATE_TOKEN_PATTERN = re.compile(r"^cmw_[A-Za-z0-9_-]{22}$")

WRITE_BODY = {
    "component_id": "cmp_1",
    "revision": 1,
    "catalog_version": "2026.8.0",
    "composition_id": "comp_1",
    "composition_revision": 3,
    "changed": [],
    "fact_summary": {"recipient": "country/ETH"},
    "resolved_payload": ENVELOPE,
    "affordance": (
        "Revise this component with PATCH /api/presentation/components/cmp_1, "
        "sending only the fields to change, or read it back with GET "
        "/api/presentation/components/cmp_1."
    ),
}


_PUBLISH_PRINCIPAL = "presentation-mcp"


def _settings(
    *,
    upstream_connect_timeout_s: float = 1.0,
    upstream_read_timeout_s: float = 1.0,
    upstream_retry_budget_s: float = 10.0,
    upstream_max_attempts: int = 3,
    upstream_backoff_base_s: float = 0.01,
    datasets_first_party_principal: str | None = _PUBLISH_PRINCIPAL,
) -> Settings:
    """Configured with a publish principal by default, so every existing
    publish-mechanics test here (path, body, memoisation, ...) exercises
    those mechanics rather than the `publish_not_configured` local refusal --
    the dedicated `--- x-dfo-principal on publish ---` tests below cover that
    refusal, and the unconfigured default, on their own.
    """
    return Settings(
        upstream_connect_timeout_s=upstream_connect_timeout_s,
        upstream_read_timeout_s=upstream_read_timeout_s,
        upstream_retry_budget_s=upstream_retry_budget_s,
        upstream_max_attempts=upstream_max_attempts,
        upstream_backoff_base_s=upstream_backoff_base_s,
        datasets_first_party_principal=datasets_first_party_principal,
    )


def _recording_sleep() -> tuple[list[float], Callable[[float], Awaitable[None]]]:
    delays: list[float] = []

    async def sleep(seconds: float) -> None:
        delays.append(seconds)

    return delays, sleep


def _assembler(
    handler: Handler,
    *,
    settings: Settings | None = None,
    sleep: Callable[[float], Awaitable[None]] | None = None,
) -> PlatformAssembler:
    settings = _settings() if settings is None else settings
    transport = httpx2.MockTransport(handler)
    if sleep is None:
        return PlatformAssembler(settings=settings, transport=transport)
    return PlatformAssembler(
        settings=settings,
        sleep=sleep,
        transport=transport,
    )


def _counting_handler(response: httpx2.Response) -> tuple[list[httpx2.Request], Handler]:
    requests: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        return response

    return requests, handler


def _problem_body(*, code: str, retryable: bool, **extra: object) -> bytes:
    payload: dict[str, object] = {
        "type": f"/codes/{code}",
        "title": "refused",
        "status": 400,
        "detail": "because",
        "code": code,
        "retryable": retryable,
        "trace_id": "trace-1",
        "capability_version": "v1",
        **extra,
    }
    return json.dumps(payload).encode()


# --- wire shape, one test per call ------------------------------------------


async def test_create_component_posts_intent_and_write_token_no_composition_id() -> None:
    requests, handler = _counting_handler(httpx2.Response(201, json=WRITE_BODY))

    async with _assembler(handler) as assembler:
        reply = await assembler.create_component(intent=INTENT, caller=Caller.EXTERNAL)

    assert isinstance(reply, WriteReply)
    request = requests[0]
    assert request.method == "POST"
    assert request.url.path == "/api/presentation/components"
    body = json.loads(request.content)
    assert set(body) == {"intent", "write_token"}
    assert body["intent"] == INTENT
    assert _CREATE_TOKEN_PATTERN.fullmatch(body["write_token"])


async def test_revise_component_patches_the_component_id() -> None:
    requests, handler = _counting_handler(httpx2.Response(200, json=WRITE_BODY | {"revision": 2}))
    patch = {"title": "New"}

    async with _assembler(handler) as assembler:
        reply = await assembler.revise_component(
            component_id="cmp_1", patch=patch, write_token=WRITE_TOKEN, caller=Caller.FIRST_PARTY
        )

    assert isinstance(reply, WriteReply)
    assert reply.revision == 2
    request = requests[0]
    assert request.method == "PATCH"
    assert request.url.path == "/api/presentation/components/cmp_1"
    assert json.loads(request.content) == {"intent": patch}
    assert request.headers[WRITE_TOKEN_HEADER] == WRITE_TOKEN


# --- the write token --------------------------------------------------------


async def test_a_create_reply_carries_the_write_token_and_sends_no_token_header() -> None:
    requests, handler = _counting_handler(
        httpx2.Response(201, json=WRITE_BODY | {"write_token": WRITE_TOKEN})
    )

    async with _assembler(handler) as assembler:
        reply = await assembler.create_component(intent=INTENT, caller=Caller.EXTERNAL)

    assert isinstance(reply, WriteReply)
    assert reply.write_token == WRITE_TOKEN
    assert WRITE_TOKEN_HEADER not in requests[0].headers


async def test_a_revise_reply_carries_no_write_token() -> None:
    _, handler = _counting_handler(httpx2.Response(200, json=WRITE_BODY | {"revision": 2}))

    async with _assembler(handler) as assembler:
        reply = await assembler.revise_component(
            component_id="cmp_1", patch={}, write_token=WRITE_TOKEN, caller=Caller.EXTERNAL
        )

    assert isinstance(reply, WriteReply)
    assert reply.write_token is None


# The apis service's write_refused problem body: its detail names the write-token
# header and folds in a fix that names its REST routes.
WRITE_REFUSED_PROBLEM = _problem_body(
    code="write_refused",
    retryable=False,
    status=403,
    detail=(
        "the x-dfo-write-token sent does not grant writing component 'cmp_1' fix: create a "
        "new component with POST /api/presentation/components; read this one first with "
        "GET /api/presentation/components/cmp_1 to reuse its intent"
    ),
)


async def test_write_refused_maps_to_a_refusal_whose_fix_is_a_new_component() -> None:
    problem = WRITE_REFUSED_PROBLEM
    requests, handler = _counting_handler(httpx2.Response(403, content=problem))

    async with _assembler(handler) as assembler:
        reply = await assembler.revise_component(
            component_id="cmp_1", patch={}, write_token="cmw_wrong", caller=Caller.EXTERNAL
        )

    assert isinstance(reply, Refusal)
    assert reply.error == "write_refused"
    assert reply.retry == RetryMode.DO_NOT_RETRY
    assert reply.fix is not None
    assert "Create a new component" in reply.fix
    assert "describe_component" in reply.fix
    assert "write_token" in reply.detail
    assert "write_component" in reply.detail
    assert "describe_component" in reply.detail
    for text in (reply.detail, reply.fix):
        assert "/api/" not in text
        assert "x-dfo-" not in text
    assert len(requests) == 1


@pytest.mark.parametrize("write_token", ["", "cmw_", "cmp_1", "cmw_bad\r\nx-evil: 1"])
async def test_revise_component_refuses_a_malformed_write_token_locally(write_token: str) -> None:
    calls, handler = _counting_handler(httpx2.Response(200, json=WRITE_BODY))

    async with _assembler(handler) as assembler:
        reply = await assembler.revise_component(
            component_id="cmp_1", patch={}, write_token=write_token, caller=Caller.EXTERNAL
        )

    assert isinstance(reply, Refusal)
    assert reply.pointer == "/write_token"
    assert reply.retry == RetryMode.DO_NOT_RETRY
    assert calls == []


async def test_get_frame_view_builds_the_path_and_forwards_narrowing_params() -> None:
    requests, handler = _counting_handler(
        httpx2.Response(200, json={"spec_id": FRAME_REF["spec_id"], "dimensions": {}})
    )

    async with _assembler(handler) as assembler:
        view = await assembler.get_frame_view(
            frame_ref=FRAME_REF, dimension="recipient", query="ken"
        )

    assert not isinstance(view, Refusal)
    request = requests[0]
    assert request.method == "GET"
    assert request.url.path == "/api/presentation/frames/spec_7c2f91/cv_4a91e2"
    assert dict(request.url.params) == {"dimension": "recipient", "query": "ken"}


async def test_get_frame_view_omits_absent_narrowing_params() -> None:
    requests, handler = _counting_handler(httpx2.Response(200, json={"spec_id": "x"}))

    async with _assembler(handler) as assembler:
        await assembler.get_frame_view(frame_ref=FRAME_REF)

    assert dict(requests[0].url.params) == {}


async def test_get_frame_view_reaches_the_platform_for_an_analysis_minted_ref() -> None:
    """An `ops_`/`ocv_` pair, the shape an analysis-mcp verb mints, is not
    refused locally: it builds the same path shape as a `spec_`/`cv_` ref and
    reaches the platform, which is the only party that decides whether the
    frame exists.
    """
    op_frame_ref = {"spec_id": "ops_7c2f91", "content_version": "ocv_4a91e2"}
    requests, handler = _counting_handler(
        httpx2.Response(200, json={"spec_id": op_frame_ref["spec_id"], "dimensions": {}})
    )

    async with _assembler(handler) as assembler:
        view = await assembler.get_frame_view(frame_ref=op_frame_ref)

    assert not isinstance(view, Refusal)
    assert requests[0].url.path == "/api/presentation/frames/ops_7c2f91/ocv_4a91e2"


def _publish_then_frame_handler(
    frame_response: httpx2.Response,
) -> tuple[list[httpx2.Request], Handler]:
    """Routes by path: a publish call gets a bare `created` reply, anything
    else gets `frame_response`. Every request is recorded in call order, so
    a test can assert the publish call happened, and happened first.
    """
    requests: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        if request.url.path == "/api/datasets/frames/publish":
            return httpx2.Response(
                200,
                json={
                    "schema_version": 1,
                    "published": {
                        "spec_id": _OPS_SPEC_ID,
                        "content_version": _OCV_CONTENT_VERSION,
                        "artifact_digest": "sha256:exemplar",
                        "created": True,
                    },
                },
            )
        return frame_response

    return requests, handler


async def test_get_frame_view_publishes_an_ops_ref_before_reaching_the_platform(
    materialisation_store: InMemoryMaterialisationStore, caplog: pytest.LogCaptureFixture
) -> None:
    """An `ops_`/`ocv_` ref this process holds gets published to apis'
    `datasets_api` surface before the `presentation_api` call that reads it
    -- the whole point of the publish hook. Its trace carries no op-spec and
    no step, so it is published without a recipe or steps, and the pair is
    logged for each.
    """
    caplog.set_level(logging.WARNING, logger=frame_publish.__name__)
    await _put_ops_frame(materialisation_store)
    op_frame_ref = {"spec_id": _OPS_SPEC_ID, "content_version": _OCV_CONTENT_VERSION}
    requests, handler = _publish_then_frame_handler(
        httpx2.Response(200, json={"spec_id": _OPS_SPEC_ID, "dimensions": {}})
    )

    async with _assembler(handler) as assembler:
        view = await assembler.get_frame_view(frame_ref=op_frame_ref)

    assert not isinstance(view, Refusal)
    assert [request.url.path for request in requests] == [
        "/api/datasets/frames/publish",
        f"/api/presentation/frames/{_OPS_SPEC_ID}/{_OCV_CONTENT_VERSION}",
    ]
    assert requests[0].headers["content-type"] == "application/json"
    publish_body = json.loads(gzip.decompress(requests[0].content))
    assert publish_body["spec_id"] == _OPS_SPEC_ID
    assert publish_body["content_version"] == _OCV_CONTENT_VERSION
    assert publish_body["role_keys"] == ["recipientCountry"]
    assert publish_body["input_content_versions"] == ["cv_" + "c" * 32]
    assert "recipe" not in publish_body
    assert "steps" not in publish_body
    warnings = [record for record in caplog.records if record.levelno == logging.WARNING]
    assert [record.args[:2] for record in warnings if isinstance(record.args, tuple)] == [
        (_OPS_SPEC_ID, _OCV_CONTENT_VERSION)
    ] * 2


_ROOT_REF = FrameRef(spec_id="spec_" + "1" * 32, content_version="cv_" + "c" * 32)


async def _put_derived_frame(
    store: InMemoryMaterialisationStore,
    *,
    of: FrameRef = _ROOT_REF,
    content_version: str = _OCV_CONTENT_VERSION,
) -> FrameRef:
    """Stores a `rank` of `of` under `content_version` with the trace a recipe is
    built from and its step, and `_ROOT_REF`'s mint record, and returns the derived
    frame's pair."""
    await store.put_mint_record(
        MintRecord(frame=_ROOT_REF, dataset=_ROOT_REF, options={"layout": "frame"})
    )
    content = json.dumps(
        {
            "frame_ref": of.model_dump(),
            "identity_version": "ops1",
            "parameters": {"rank_by": ["recipientCountry"]},
            "verb": "rank",
        },
        separators=(",", ":"),
    )
    spec_id = f"ops_{hashlib.sha256(content.encode()).hexdigest()[:32]}"
    await _put_ops_frame(
        store,
        spec_id=spec_id,
        content_version=content_version,
        input_content_versions=(of.content_version,),
        execution_record=_StubExecutionRecord(content),
        step={**_RANK_STEP, "id": spec_id, "input": _version_ref(of)},
    )
    return FrameRef(spec_id=spec_id, content_version=content_version)


def _version_ref(ref: FrameRef) -> dict[str, str]:
    return {"id": ref.spec_id, "content_version": ref.content_version}


_RANK_STEP: dict[str, Any] = {
    "op": "rank",
    "policy": "rank_by_latest_value",
    "params": {"limit": 1},
    "kept": {"entities": 1},
}

# The step list of `_put_derived_frame`'s rank: the root's retrieval, then the rank.
_RANK_STEPS = [
    {
        "id": "s1",
        "op": "retrieve",
        "input": _version_ref(_ROOT_REF),
        "options": {"layout": "frame"},
    },
    {**_RANK_STEP, "id": "s2", "input": "s1"},
]


async def test_a_derived_frame_is_published_with_its_recipe_and_steps(
    materialisation_store: InMemoryMaterialisationStore,
) -> None:
    ref = await _put_derived_frame(materialisation_store)
    spec_id, root = ref.spec_id, _ROOT_REF
    requests, handler = _publish_then_frame_handler(
        httpx2.Response(200, json={"spec_id": spec_id, "dimensions": {}})
    )

    async with _assembler(handler) as assembler:
        view = await assembler.get_frame_view(frame_ref=ref.model_dump())

    assert not isinstance(view, Refusal)
    publish_body = json.loads(gzip.decompress(requests[0].content))
    assert publish_body["recipe"] == await build_recipe(ref, store=materialisation_store)
    assert publish_body["recipe"]["tree"]["inputs"] == {
        "frame_ref": {"frame": root.model_dump(), "root": 0}
    }
    assert publish_body["recipe"]["roots"][0]["dataset"] == root.model_dump()
    assert publish_body["steps"] == _RANK_STEPS
    assert set(publish_body) == {
        "spec_id",
        "content_version",
        "role_keys",
        "rows",
        "input_content_versions",
        "recipe",
        "steps",
    }


async def test_publishing_reads_each_ancestor_s_trace_and_mint_record_once(
    materialisation_store: InMemoryMaterialisationStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A rank of a rank of a root: the recipe and the steps of the leaf read the
    derived ancestor's trace and the root's mint record once between them."""
    ancestor = await _put_derived_frame(materialisation_store)
    leaf = await _put_derived_frame(
        materialisation_store, of=ancestor, content_version="ocv_" + "9" * 32
    )
    reads: list[tuple[str, str]] = []
    for name in ("get_trace", "get_mint_record"):
        read = getattr(materialisation_store, name)

        async def counted(ref: FrameRef, *, name: str = name, read: Any = read) -> Any:
            reads.append((name, ref.spec_id))
            return await read(ref)

        monkeypatch.setattr(materialisation_store, name, counted)

    async with client_module.publish_body_for(
        spec_id=leaf.spec_id, content_version=leaf.content_version, max_frame_rows=10
    ) as published:
        body = json.loads(published.data)

    assert [step["op"] for step in body["steps"]] == ["retrieve", "rank", "rank"]
    assert body["recipe"]["tree"]["inputs"]["frame_ref"]["frame"] == ancestor.model_dump()
    assert sorted(reads) == [
        ("get_mint_record", _ROOT_REF.spec_id),
        ("get_trace", ancestor.spec_id),
    ]


async def test_a_recipe_over_the_platforms_size_ceiling_is_left_out_of_the_body(
    materialisation_store: InMemoryMaterialisationStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    ref = await _put_derived_frame(materialisation_store)
    recipe = await build_recipe(ref, store=materialisation_store)
    recipe_bytes = len(json.dumps(recipe, separators=(",", ":")))
    bodies: list[dict[str, Any]] = []

    for ceiling in (recipe_bytes, recipe_bytes - 1):
        monkeypatch.setattr(frame_publish, "_RECIPE_CEILING_BYTES", ceiling)
        async with client_module.publish_body_for(
            spec_id=ref.spec_id, content_version=ref.content_version, max_frame_rows=10
        ) as published:
            bodies.append(json.loads(published.data))

    assert bodies[0]["recipe"] == recipe
    assert "recipe" not in bodies[1]


async def test_publish_frame_sends_the_callers_recipe_and_says_whether_it_was_new(
    materialisation_store: InMemoryMaterialisationStore,
) -> None:
    """A frame is published for no component, with the recipe the caller built, and
    every call posts: an earlier publish in this process may have carried none."""
    ref = await _put_derived_frame(materialisation_store)
    recipe = await build_recipe(ref, store=materialisation_store) | {"anchor": {"period": "2023"}}
    requests: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        assert request.url.path == "/api/datasets/frames/publish"
        requests.append(request)
        return httpx2.Response(
            200, json={"schema_version": 1, "published": {"created": len(requests) == 1}}
        )

    async with _assembler(handler) as assembler:
        answers = [
            await assembler.publish_frame(
                spec_id=ref.spec_id, content_version=ref.content_version, recipe=recipe
            )
            for _ in range(2)
        ]

    assert answers == [True, False]
    bodies = [json.loads(gzip.decompress(request.content)) for request in requests]
    assert [body["recipe"] for body in bodies] == [recipe, recipe]
    assert [body["steps"] for body in bodies] == [_RANK_STEPS, _RANK_STEPS]
    assert requests[0].headers[PRINCIPAL_HEADER]


async def test_publish_frame_refuses_a_frame_the_store_does_not_hold_and_sends_nothing(
    materialisation_store: InMemoryMaterialisationStore,
) -> None:
    requests, handler = _counting_handler(httpx2.Response(200, json={}))

    async with _assembler(handler) as assembler:
        refused = await assembler.publish_frame(
            spec_id="ops_" + "a" * 32, content_version="ocv_" + "b" * 32, recipe={}
        )

    assert isinstance(refused, Refusal)
    assert refused.error == "request_invalid"
    assert requests == []


async def test_publish_frame_refuses_a_recipe_over_the_ceiling_and_sends_nothing(
    materialisation_store: InMemoryMaterialisationStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The caller's recipe is never dropped: a frame published without it would be
    of no use to a caller that built one."""
    ref = await _put_derived_frame(materialisation_store)
    recipe = await build_recipe(ref, store=materialisation_store)
    monkeypatch.setattr(
        frame_publish, "_RECIPE_CEILING_BYTES", len(json.dumps(recipe, separators=(",", ":"))) - 1
    )
    requests, handler = _counting_handler(httpx2.Response(200, json={}))

    async with _assembler(handler) as assembler:
        refused = await assembler.publish_frame(
            spec_id=ref.spec_id, content_version=ref.content_version, recipe=recipe
        )

    assert isinstance(refused, Refusal)
    assert refused.error == "request_invalid"
    assert requests == []


@pytest.mark.parametrize(
    ("constraint_id", "published", "answered"),
    [
        ("recipe_requires_matching_normal_form_hash", True, NormalFormConflict),
        ("recipe_requires_matching_replay_key", True, bool),
        ("replay_key_requires_one_frame", True, bool),
        ("lineage_requires_matching_inputs", False, Refusal),
        (None, False, Refusal),
    ],
)
async def test_a_recipe_conflict_leaves_the_frame_published_and_any_other_conflict_refuses(
    materialisation_store: InMemoryMaterialisationStore,
    constraint_id: str | None,
    published: bool,
    answered: type,
) -> None:
    """apis records the frame before it compares recipes, so a recipe conflict
    means the pair is published under the recipe apis already held. A caller
    publishing its own recipe is told when the held one has another normal form."""
    ref = await _put_derived_frame(materialisation_store)
    recipe = await build_recipe(ref, store=materialisation_store)
    paths: list[str] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        paths.append(request.url.path)
        if request.url.path == "/api/datasets/frames/publish":
            extra = {} if constraint_id is None else {"constraint_id": constraint_id}
            return httpx2.Response(
                409, content=_problem_body(code="publish_conflict", retryable=False, **extra)
            )
        return httpx2.Response(200, json={"spec_id": ref.spec_id, "dimensions": {}})

    async with _assembler(handler) as assembler:
        replies = [await assembler.get_frame_view(frame_ref=ref.model_dump()) for _ in range(2)]
        own = await assembler.publish_frame(
            spec_id=ref.spec_id, content_version=ref.content_version, recipe=recipe
        )

    assert own is False if answered is bool else isinstance(own, answered)
    assert paths.pop() == "/api/datasets/frames/publish"
    frame_path = f"/api/presentation/frames/{ref.spec_id}/{ref.content_version}"
    if published:
        assert not any(isinstance(reply, Refusal) for reply in replies)
        assert paths == ["/api/datasets/frames/publish", frame_path, frame_path]
    else:
        assert [getattr(reply, "error", None) for reply in replies] == ["publish_conflict"] * 2
        assert paths == ["/api/datasets/frames/publish"] * 2


@pytest.mark.parametrize("ancestor", ["stores_no_step", "fails_a_store_check"])
async def test_a_chain_through_an_unreadable_ancestor_step_publishes_without_steps(
    materialisation_store: InMemoryMaterialisationStore,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
    ancestor: str,
) -> None:
    """The frame's own trace holds a step, and the trace of the frame it read was
    stored before steps were or fails a store check, so the chain is unknown and
    the frame publishes without it."""
    caplog.set_level(logging.WARNING, logger=frame_publish.__name__)
    older = FrameRef(spec_id="ops_" + "d" * 32, content_version="ocv_" + "e" * 32)
    await _put_ops_frame(
        materialisation_store, spec_id=older.spec_id, content_version=older.content_version
    )
    if ancestor == "fails_a_store_check":
        get_trace = materialisation_store.get_trace

        async def failing_get_trace(ref: FrameRef) -> StoredFrameTrace | None:
            if ref == older:
                raise FrameIntegrityError(ref, reason="sha256_mismatch")
            return await get_trace(ref)

        monkeypatch.setattr(materialisation_store, "get_trace", failing_get_trace)
    await _put_ops_frame(
        materialisation_store,
        input_content_versions=("cv_" + "f" * 32,),
        step={**_RANK_STEP, "id": _OPS_SPEC_ID, "input": _version_ref(older)},
    )
    requests, handler = _publish_then_frame_handler(
        httpx2.Response(200, json={"spec_id": _OPS_SPEC_ID, "dimensions": {}})
    )

    async with _assembler(handler) as assembler:
        view = await assembler.get_frame_view(
            frame_ref={"spec_id": _OPS_SPEC_ID, "content_version": _OCV_CONTENT_VERSION}
        )

    assert not isinstance(view, Refusal)
    publish_body = json.loads(gzip.decompress(requests[0].content))
    assert "steps" not in publish_body
    assert "composed_stat_vars" not in publish_body
    assert any(
        record.getMessage().startswith(
            f"publishing {_OPS_SPEC_ID}/{_OCV_CONTENT_VERSION} without steps"
        )
        for record in caplog.records
    )


@pytest.mark.parametrize("bound", ["_MAX_STEPS", "_MAX_STEPS_BYTES"])
async def test_a_step_list_over_a_door_bound_refuses_the_publish_and_sends_nothing(
    materialisation_store: InMemoryMaterialisationStore,
    monkeypatch: pytest.MonkeyPatch,
    bound: str,
) -> None:
    """The steps are never dropped to make a body fit."""
    ref = await _put_derived_frame(materialisation_store)
    at_bound = {
        "_MAX_STEPS": len(_RANK_STEPS),
        "_MAX_STEPS_BYTES": len(rfc8785.dumps(_RANK_STEPS)),
    }[bound]
    monkeypatch.setattr(frame_publish, bound, at_bound - 1)
    requests, handler = _counting_handler(httpx2.Response(200, json={}))

    async with _assembler(handler) as assembler:
        reply = await assembler.get_frame_view(frame_ref=ref.model_dump())

    assert isinstance(reply, Refusal)
    assert reply.error == "request_invalid"
    assert reply.retry == RetryMode.DO_NOT_RETRY
    assert requests == []


async def test_a_stored_step_naming_no_frame_pair_refuses_the_publish_and_sends_nothing(
    materialisation_store: InMemoryMaterialisationStore,
) -> None:
    await _put_ops_frame(
        materialisation_store, step={**_RANK_STEP, "id": _OPS_SPEC_ID, "input": "s1"}
    )
    requests, handler = _counting_handler(httpx2.Response(200, json={}))

    async with _assembler(handler) as assembler:
        reply = await assembler.get_frame_view(
            frame_ref={"spec_id": _OPS_SPEC_ID, "content_version": _OCV_CONTENT_VERSION}
        )

    assert isinstance(reply, Refusal)
    assert reply.error == "request_invalid"
    assert requests == []


async def test_a_door_refusal_of_the_steps_fails_the_publish(
    materialisation_store: InMemoryMaterialisationStore,
) -> None:
    ref = await _put_derived_frame(materialisation_store)
    requests, handler = _counting_handler(
        httpx2.Response(
            400,
            content=_problem_body(code="request_invalid", retryable=False, pointer="/steps/1"),
        )
    )

    async with _assembler(handler) as assembler:
        reply = await assembler.get_frame_view(frame_ref=ref.model_dump())

    assert isinstance(reply, Refusal)
    assert (reply.error, reply.pointer) == ("request_invalid", "/steps/1")
    assert "Publishing this frame to the platform failed" in reply.detail
    assert [request.url.path for request in requests] == ["/api/datasets/frames/publish"]


@pytest.mark.parametrize(
    ("role", "described", "cells"),
    [
        ("observationAbout", "observationAbout_type", ("Continent",)),
        ("financingRecipient", "financingRecipient_party_kind", ("total",)),
        ("financingProvider", "financingProvider_party_sector", ("official", "private")),
        ("financingProvider", "financingProvider_party_whole", ("true", "false")),
    ],
    ids=["place-type", "party-kind", "party-sector", "party-whole"],
)
async def test_publish_leaves_a_place_type_or_party_kind_column_out_of_the_published_frame(
    materialisation_store: InMemoryMaterialisationStore,
    role: str,
    described: str,
    cells: tuple[str, ...],
) -> None:
    """A level filter's output carries `observationAbout_type` as a role, and a
    sum keyed by a flow party carries `financingRecipient_party_kind` and the
    party's sector and whole columns. apis joins every role's value into a
    series label, so the published frame holds the place column alone, and the
    door's own `Frame.build` accepts it with every row kept.
    """
    rows = [
        {
            "stat_var": "dcid:LifeExpectancy_Person",
            role: place,
            described: cells[index % len(cells)],
            "time_period": year,
            "value": value,
        }
        for index, (place, year, value) in enumerate(
            (
                ("africa", "2020", 63.8),
                ("africa", "2021", 63.2),
                ("asia", "2020", 74.2),
            )
        )
    ]
    frame = Frame.build(role_keys=[role, described], rows=rows)
    await materialisation_store.put(
        FrameRef(spec_id=_OPS_SPEC_ID, content_version=_OCV_CONTENT_VERSION),
        frame=frame,
        trace=_StubTrace(
            upstream_trace={},
            frame_content_version=_OCV_CONTENT_VERSION,
            input_content_versions=("cv_" + "c" * 32,),
        ),
        writer="test",
    )
    requests, handler = _publish_then_frame_handler(
        httpx2.Response(200, json={"spec_id": _OPS_SPEC_ID, "dimensions": {}})
    )

    async with _assembler(handler) as assembler:
        await assembler.get_frame_view(
            frame_ref={"spec_id": _OPS_SPEC_ID, "content_version": _OCV_CONTENT_VERSION}
        )

    publish_body = json.loads(gzip.decompress(requests[0].content))
    assert publish_body["role_keys"] == [role]
    assert all(described not in row for row in publish_body["rows"])
    published = Frame.build(role_keys=publish_body["role_keys"], rows=publish_body["rows"])
    assert published.columns == tuple(c for c in frame.columns if c != described)
    assert published.table.num_rows == 3


async def test_get_frame_view_does_not_publish_for_a_spec_ref(
    materialisation_store: InMemoryMaterialisationStore,
) -> None:
    """A `spec_`/`cv_` ref names a Datasets snapshot, never the publish
    door's own `ops_`-only shape -- no publish call, even though this same
    store can (and, for a real `get_data` mint, does) hold a `spec_`/`cv_`
    entry too.
    """
    requests, handler = _counting_handler(
        httpx2.Response(200, json={"spec_id": FRAME_REF["spec_id"], "dimensions": {}})
    )

    async with _assembler(handler) as assembler:
        view = await assembler.get_frame_view(frame_ref=FRAME_REF)

    assert not isinstance(view, Refusal)
    assert [request.url.path for request in requests] == [
        f"/api/presentation/frames/{FRAME_REF['spec_id']}/{FRAME_REF['content_version']}"
    ]


_DERIVED_REF = FrameRef(spec_id=_ROOT_REF.spec_id, content_version="cv_" + "d" * 32)


async def _frame_ref_on_the_wire(call: str, frame_ref: FrameRef) -> dict[str, str]:
    """The pair the platform receives when `call` is made with `frame_ref`."""
    ref = frame_ref.model_dump()
    requests, handler = _counting_handler(
        httpx2.Response(200, json=WRITE_BODY | {"dimensions": {}})
    )
    async with _assembler(handler) as assembler:
        match call:
            case "get_frame_view":
                reply = await assembler.get_frame_view(frame_ref=ref)
                assert not isinstance(reply, Refusal)
                _, _, spec_id, content_version = requests[0].url.path.rsplit("/", 3)
                return {"spec_id": spec_id, "content_version": content_version}
            case "create_component":
                reply = await assembler.create_component(
                    intent={"frame_ref": ref}, caller=Caller.EXTERNAL
                )
            case _:
                reply = await assembler.revise_component(
                    component_id="cmp_1",
                    patch={"frame_ref": ref},
                    write_token=WRITE_TOKEN,
                    caller=Caller.EXTERNAL,
                )
    assert not isinstance(reply, Refusal)
    return json.loads(requests[0].content)["intent"]["frame_ref"]


@pytest.mark.parametrize("call", ["get_frame_view", "create_component", "revise_component"])
@pytest.mark.parametrize(
    ("record", "called", "sent"),
    [
        pytest.param(
            MintRecord(frame=_DERIVED_REF, dataset=_ROOT_REF, options={"columns": ["x"]}),
            _DERIVED_REF,
            _ROOT_REF,
            id="derived pair goes as its dataset pair",
        ),
        pytest.param(
            MintRecord(frame=_ROOT_REF, dataset=_ROOT_REF, options={"layout": "frame"}),
            _ROOT_REF,
            _ROOT_REF,
            id="pair equal to its dataset pair goes unchanged",
        ),
        pytest.param(
            None, _DERIVED_REF, _DERIVED_REF, id="pair with no mint record goes unchanged"
        ),
    ],
)
async def test_a_spec_pair_goes_to_the_platform_as_the_dataset_pair_its_mint_record_names(
    materialisation_store: InMemoryMaterialisationStore,
    call: str,
    record: MintRecord | None,
    called: FrameRef,
    sent: FrameRef,
) -> None:
    if record is not None:
        await materialisation_store.put_mint_record(record)

    assert await _frame_ref_on_the_wire(call, called) == sent.model_dump()


@pytest.mark.parametrize("call", ["get_frame_view", "create_component", "revise_component"])
@pytest.mark.parametrize(
    "failure",
    [
        pytest.param(FrameIntegrityError(_DERIVED_REF, reason="sha256_mismatch"), id="integrity"),
        pytest.param(OSError("storage unreachable"), id="storage"),
    ],
)
async def test_a_spec_pair_whose_mint_record_lookup_fails_goes_to_the_platform_unchanged(
    materialisation_store: InMemoryMaterialisationStore,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
    call: str,
    failure: Exception,
) -> None:
    async def failing_get_mint_record(ref: FrameRef) -> MintRecord | None:
        raise failure

    monkeypatch.setattr(materialisation_store, "get_mint_record", failing_get_mint_record)
    caplog.set_level(logging.WARNING, logger=client_module.__name__)

    assert await _frame_ref_on_the_wire(call, _DERIVED_REF) == _DERIVED_REF.model_dump()

    [record] = [r for r in caplog.records if r.name == "presentation_mcp.client"]
    assert record.levelno == logging.WARNING
    assert _DERIVED_REF.spec_id in record.getMessage()
    assert record.exc_info is not None
    assert record.exc_info[1] is failure


async def test_a_cancelled_mint_record_lookup_propagates(
    materialisation_store: InMemoryMaterialisationStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def cancelled_get_mint_record(ref: FrameRef) -> MintRecord | None:
        raise asyncio.CancelledError

    monkeypatch.setattr(materialisation_store, "get_mint_record", cancelled_get_mint_record)

    with pytest.raises(asyncio.CancelledError):
        await _frame_ref_on_the_wire("get_frame_view", _DERIVED_REF)


async def test_create_component_publishes_an_ops_frame_ref_before_creating(
    materialisation_store: InMemoryMaterialisationStore,
) -> None:
    """`create_component` never validates `intent["frame_ref"]` locally
    (unlike `get_frame_view`), so this is the one place proving the write
    path publishes too.
    """
    await _put_ops_frame(materialisation_store)
    intent = INTENT | {
        "frame_ref": {"spec_id": _OPS_SPEC_ID, "content_version": _OCV_CONTENT_VERSION}
    }
    requests, handler = _publish_then_frame_handler(httpx2.Response(201, json=WRITE_BODY))

    async with _assembler(handler) as assembler:
        reply = await assembler.create_component(intent=intent, caller=Caller.EXTERNAL)

    assert isinstance(reply, WriteReply)
    assert [request.url.path for request in requests] == [
        "/api/datasets/frames/publish",
        "/api/presentation/components",
    ]


async def test_create_component_does_not_publish_twice_for_the_same_pair(
    materialisation_store: InMemoryMaterialisationStore,
) -> None:
    """A ref already published earlier this process's lifetime is a
    same-process no-op on a second call -- the per-process memo
    (`PlatformAssembler._published`) this hook relies on instead of a
    network round trip every time.
    """
    await _put_ops_frame(materialisation_store)
    frame_ref = {"spec_id": _OPS_SPEC_ID, "content_version": _OCV_CONTENT_VERSION}
    requests, handler = _publish_then_frame_handler(
        httpx2.Response(200, json={"spec_id": _OPS_SPEC_ID})
    )

    async with _assembler(handler) as assembler:
        first = await assembler.get_frame_view(frame_ref=frame_ref)
        second = await assembler.get_frame_view(frame_ref=frame_ref)

    assert not isinstance(first, Refusal)
    assert not isinstance(second, Refusal)
    assert [request.url.path for request in requests] == [
        "/api/datasets/frames/publish",
        f"/api/presentation/frames/{_OPS_SPEC_ID}/{_OCV_CONTENT_VERSION}",
        f"/api/presentation/frames/{_OPS_SPEC_ID}/{_OCV_CONTENT_VERSION}",
    ]


# --- x-dfo-principal on publish ------------------------------------------


def _publish_not_permitted_body(*, principal: str | None) -> bytes:
    """apis' `datasets_api.doors.publish.PublishNotPermitted` wire shape:
    `publish_not_permitted`, 403, not retryable.
    """
    detail = "the publish permission is not granted to this caller"
    if principal is None:
        detail = f"no principal header was presented; {detail}"
    return _problem_body(code="publish_not_permitted", retryable=False, detail=detail)


async def test_publish_call_carries_the_configured_principal_header(
    materialisation_store: InMemoryMaterialisationStore,
) -> None:
    """The one call this client makes that the apis service's publish door actually
    checks (`datasets_api.doors.publish`) carries the configured principal.
    No other call carries it -- that door is the only route on this surface
    that reads the header at all.
    """
    await _put_ops_frame(materialisation_store)
    frame_ref = {"spec_id": _OPS_SPEC_ID, "content_version": _OCV_CONTENT_VERSION}
    requests, handler = _publish_then_frame_handler(
        httpx2.Response(200, json={"spec_id": _OPS_SPEC_ID, "dimensions": {}})
    )

    async with _assembler(handler) as assembler:
        view = await assembler.get_frame_view(frame_ref=frame_ref)

    assert not isinstance(view, Refusal)
    publish_request, frame_request = requests
    assert publish_request.url.path == "/api/datasets/frames/publish"
    assert publish_request.headers[PRINCIPAL_HEADER] == _PUBLISH_PRINCIPAL
    assert PRINCIPAL_HEADER not in frame_request.headers


async def test_publish_refuses_locally_when_principal_unconfigured(
    materialisation_store: InMemoryMaterialisationStore,
) -> None:
    """An unset `datasets_first_party_principal` refuses before any network
    call at all -- the platform would refuse this identically (403
    `publish_not_permitted`) on every attempt, so there is nothing to gain by
    reaching it first.
    """
    await _put_ops_frame(materialisation_store)
    frame_ref = {"spec_id": _OPS_SPEC_ID, "content_version": _OCV_CONTENT_VERSION}
    calls, handler = _counting_handler(httpx2.Response(200, json={"spec_id": _OPS_SPEC_ID}))
    settings = _settings(datasets_first_party_principal=None)

    async with _assembler(handler, settings=settings) as assembler:
        reply = await assembler.get_frame_view(frame_ref=frame_ref)

    assert isinstance(reply, Refusal)
    assert reply.error == "publish_not_configured"
    assert reply.retry == RetryMode.DO_NOT_RETRY
    assert calls == []


async def test_publish_wraps_the_platforms_real_publish_not_permitted_refusal(
    materialisation_store: InMemoryMaterialisationStore,
) -> None:
    """A principal this process sends but the platform doesn't accept -- the
    misconfiguration a bare-200 mock can never catch -- surfaces the
    platform's own 403 `publish_not_permitted` refusal, reworded to name publish
    as the call that actually failed.
    """
    await _put_ops_frame(materialisation_store)
    frame_ref = {"spec_id": _OPS_SPEC_ID, "content_version": _OCV_CONTENT_VERSION}

    def handler(request: httpx2.Request) -> httpx2.Response:
        assert request.url.path == "/api/datasets/frames/publish"
        return httpx2.Response(
            403, content=_publish_not_permitted_body(principal=_PUBLISH_PRINCIPAL)
        )

    async with _assembler(handler) as assembler:
        reply = await assembler.get_frame_view(frame_ref=frame_ref)

    assert isinstance(reply, Refusal)
    assert reply.error == "publish_not_permitted"
    assert reply.retry == RetryMode.DO_NOT_RETRY
    assert "trace-1" not in json.dumps(dataclasses.asdict(reply), default=str)
    assert "Publishing this frame to the platform failed" in reply.detail
    assert "publish permission is not granted" in reply.detail


async def _publish_with_ceiling(
    store: InMemoryMaterialisationStore,
    monkeypatch: pytest.MonkeyPatch,
    *,
    ceiling_delta: int,
) -> tuple[Refusal | dict[str, object], list[httpx2.Request], int]:
    """Publish the stored frame with the ceiling set `ceiling_delta` bytes
    under its real compressed body size; returns the reply, the requests sent,
    and that size."""
    await _put_ops_frame(store)
    frame_ref = {"spec_id": _OPS_SPEC_ID, "content_version": _OCV_CONTENT_VERSION}
    async with client_module.publish_body_for(
        spec_id=_OPS_SPEC_ID,
        content_version=_OCV_CONTENT_VERSION,
        max_frame_rows=_settings().max_frame_rows,
    ) as published:
        body_bytes = len(gzip.compress(published.data, 1, mtime=0))
    monkeypatch.setattr(client_module, "_PUBLISH_BODY_CEILING_BYTES", body_bytes + ceiling_delta)
    requests, handler = _publish_then_frame_handler(
        httpx2.Response(200, json={"spec_id": _OPS_SPEC_ID, "dimensions": {}})
    )
    async with _assembler(handler) as assembler:
        reply = await assembler.get_frame_view(frame_ref=frame_ref)
    return reply, requests, body_bytes


async def test_publish_body_over_the_ceiling_refuses_before_any_request(
    materialisation_store: InMemoryMaterialisationStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    reply, requests, body_bytes = await _publish_with_ceiling(
        materialisation_store, monkeypatch, ceiling_delta=-1
    )

    assert isinstance(reply, Refusal)
    assert reply.error == "over_ceiling"
    assert reply.retry == RetryMode.CORRECT_AND_RETRY
    assert reply.fix == {
        "ceiling_kind": "publish_body_bytes",
        "body_bytes": body_bytes,
        "ceiling": body_bytes - 1,
    }
    assert requests == []


async def test_publish_body_at_the_ceiling_is_sent(
    materialisation_store: InMemoryMaterialisationStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    reply, requests, _ = await _publish_with_ceiling(
        materialisation_store, monkeypatch, ceiling_delta=0
    )

    assert not isinstance(reply, Refusal)
    assert requests[0].url.path == "/api/datasets/frames/publish"


async def test_publish_body_refusal_names_the_real_ceiling_and_this_frames_row_width(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Uses the real 32 MiB constant: 40,000,000 compressed bytes over 200,000
    rows is 200 bytes a row, so 167,772 rows of that shape fit."""

    @contextlib.asynccontextmanager
    async def oversized(**_: object) -> AsyncIterator[PublishBody]:
        yield PublishBody(data=b"{}", rows=200_000, hold_through=_passthrough)

    monkeypatch.setattr(client_module, "publish_body_for", oversized)
    monkeypatch.setattr(client_module.gzip, "compress", lambda *_, **__: b"x" * 40_000_000)
    requests, handler = _publish_then_frame_handler(httpx2.Response(200, json={}))

    async with _assembler(handler) as assembler:
        reply = await assembler.get_frame_view(
            frame_ref={"spec_id": _OPS_SPEC_ID, "content_version": _OCV_CONTENT_VERSION}
        )

    assert isinstance(reply, Refusal)
    assert reply.fix == {
        "ceiling_kind": "publish_body_bytes",
        "body_bytes": 40_000_000,
        "ceiling": 33_554_432,
    }
    assert "40,000,000 compressed bytes" in reply.detail
    assert "33,554,432" in reply.detail
    assert "200 bytes a row" in reply.detail
    assert "167,772 rows" in reply.detail
    assert requests == []


def _publish_posts(requests: list[httpx2.Request]) -> list[httpx2.Request]:
    return [r for r in requests if r.url.path == "/api/datasets/frames/publish"]


async def _passthrough(work: Awaitable[bytes]) -> bytes:
    return await work


@contextlib.contextmanager
def _stub_published_body(
    monkeypatch: pytest.MonkeyPatch, *, data: bytes, rows: int = 1_000
) -> Iterator[None]:
    @contextlib.asynccontextmanager
    async def stub(**_: str) -> AsyncIterator[PublishBody]:
        yield PublishBody(data=data, rows=rows, hold_through=_passthrough)

    monkeypatch.setattr(client_module, "publish_body_for", stub)
    yield


async def _publish_stub(handler: Handler) -> Refusal | dict[str, object]:
    async with _assembler(handler) as assembler:
        return await assembler.get_frame_view(
            frame_ref={"spec_id": _OPS_SPEC_ID, "content_version": _OCV_CONTENT_VERSION}
        )


async def test_publish_post_is_gzipped_json_the_door_can_decode(
    materialisation_store: InMemoryMaterialisationStore,
) -> None:
    await _put_ops_frame(materialisation_store)
    async with client_module.publish_body_for(
        spec_id=_OPS_SPEC_ID,
        content_version=_OCV_CONTENT_VERSION,
        max_frame_rows=_settings().max_frame_rows,
    ) as published:
        expected = published.data
    requests, handler = _publish_then_frame_handler(
        httpx2.Response(200, json={"spec_id": _OPS_SPEC_ID, "dimensions": {}})
    )

    await _publish_stub(handler)

    (post,) = _publish_posts(requests)
    assert post.headers["Content-Encoding"] == "gzip"
    assert post.headers["Content-Type"] == "application/json"
    assert gzip.decompress(post.content) == expected
    assert "Content-Encoding" not in requests[-1].headers


async def test_publish_compression_runs_off_the_event_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_compress = gzip.compress
    threads: list[int] = []

    def recording(*args: Any, **kwargs: Any) -> bytes:
        threads.append(threading.get_ident())
        return real_compress(*args, **kwargs)

    monkeypatch.setattr(client_module.gzip, "compress", recording)
    requests, handler = _publish_then_frame_handler(httpx2.Response(200, json={}))

    with _stub_published_body(monkeypatch, data=b'{"a": 1}'):
        await _publish_stub(handler)

    assert threads
    assert threading.get_ident() not in threads
    assert len(_publish_posts(requests)) == 1


async def test_publish_ceiling_bounds_the_compressed_bytes_not_the_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Compressible JSON well over 32 MiB goes through; incompressible bytes
    whose gzip form is over 32 MiB are refused with the compressed size."""
    requests, handler = _publish_then_frame_handler(httpx2.Response(200, json={}))
    with _stub_published_body(monkeypatch, data=b'{"a":"' + b"x" * 40_000_000 + b'"}'):
        sent = await _publish_stub(handler)
    assert not isinstance(sent, Refusal)
    assert len(_publish_posts(requests)) == 1

    noise = os.urandom(33 * 1024 * 1024)
    requests, handler = _publish_then_frame_handler(httpx2.Response(200, json={}))
    with _stub_published_body(monkeypatch, data=noise):
        refused = await _publish_stub(handler)
    assert isinstance(refused, Refusal)
    assert refused.error == "over_ceiling"
    assert refused.fix["ceiling_kind"] == "publish_body_bytes"
    assert refused.fix["ceiling"] == 32 * 1024 * 1024
    assert refused.fix["body_bytes"] == len(gzip.compress(noise, 1, mtime=0))
    assert refused.fix["body_bytes"] > 32 * 1024 * 1024
    assert "compressed bytes" in refused.detail
    assert requests == []


async def test_publish_decoded_bound_refuses_before_compressing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*_: Any, **__: Any) -> bytes:
        raise AssertionError("compressed a body over the decoded bound")

    monkeypatch.setattr(client_module.gzip, "compress", forbidden)
    requests, handler = _publish_then_frame_handler(httpx2.Response(200, json={}))
    over = client_module._PUBLISH_DECODED_BODY_CEILING_BYTES + 1

    with _stub_published_body(monkeypatch, data=bytes(over), rows=300_000):
        refused = await _publish_stub(handler)

    assert isinstance(refused, Refusal)
    assert refused.error == "over_ceiling"
    assert refused.retry == RetryMode.CORRECT_AND_RETRY
    assert refused.fix == {
        "ceiling_kind": "publish_body_decoded_bytes",
        "body_bytes": over,
        "ceiling": 129_000_000,
    }
    assert "uncompressed bytes" in refused.detail
    assert requests == []


async def test_publish_decoded_bound_admits_a_body_at_the_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests, handler = _publish_then_frame_handler(httpx2.Response(200, json={}))

    with _stub_published_body(monkeypatch, data=bytes(129_000_000)):
        reply = await _publish_stub(handler)

    assert not isinstance(reply, Refusal)
    assert len(_publish_posts(requests)) == 1


async def test_publish_retry_resends_the_same_compressed_bytes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    compressions = 0
    real_compress = gzip.compress

    def counting(*args: Any, **kwargs: Any) -> bytes:
        nonlocal compressions
        compressions += 1
        return real_compress(*args, **kwargs)

    monkeypatch.setattr(client_module.gzip, "compress", counting)
    posts: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        if request.url.path != "/api/datasets/frames/publish":
            return httpx2.Response(200, json={})
        posts.append(request)
        if len(posts) == 1:
            return httpx2.Response(503, content=b"unavailable")
        return httpx2.Response(200, json={})

    _, sleep = _recording_sleep()
    with _stub_published_body(monkeypatch, data=b'{"rows": [1, 2, 3]}'):
        async with _assembler(handler, sleep=sleep) as assembler:
            reply = await assembler.get_frame_view(
                frame_ref={"spec_id": _OPS_SPEC_ID, "content_version": _OCV_CONTENT_VERSION}
            )

    assert not isinstance(reply, Refusal)
    assert len(posts) == 2
    assert posts[0].content == posts[1].content
    assert all(p.headers["Content-Encoding"] == "gzip" for p in posts)
    assert compressions == 1


async def test_publish_is_refused_while_the_row_budget_is_full_and_sent_once_it_frees(
    materialisation_store: InMemoryMaterialisationStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The stored frame's one row is claimed before it is read and held while its
    body is encoded and posted. While another call holds the whole budget the publish
    refuses `service_unavailable` before any read or request, and once that call
    releases it publishes."""
    await _put_ops_frame(materialisation_store)
    frame_ref = {"spec_id": _OPS_SPEC_ID, "content_version": _OCV_CONTENT_VERSION}
    in_flight_at_decode: list[int] = []
    in_flight_at_encode: list[int] = []
    encode = frame_publish.publish_body
    open_frame = materialisation_store.open

    async def recording_open(ref: FrameRef) -> HeldFrame | None:
        held = await open_frame(ref)
        assert held is not None

        async def read(*, max_rows: int) -> tuple[Frame, StoredFrameTrace]:
            in_flight_at_decode.append(row_budget().in_flight)
            return await held.read(max_rows=max_rows)

        return HeldFrame(num_rows=held.num_rows, read=read)

    def recording_encode(frame: Frame, **kwargs: Any) -> bytes:
        in_flight_at_encode.append(row_budget().in_flight)
        return encode(frame, **kwargs)

    monkeypatch.setattr(materialisation_store, "open", recording_open)
    monkeypatch.setattr(frame_publish, "publish_body", recording_encode)
    set_row_budget(RowBudget(max_rows=1))
    requests, route = _publish_then_frame_handler(
        httpx2.Response(200, json={"spec_id": _OPS_SPEC_ID, "dimensions": {}})
    )
    in_flight_at_post: list[int] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        if request.url.path == "/api/datasets/frames/publish":
            in_flight_at_post.append(row_budget().in_flight)
        return route(request)

    async with _assembler(handler) as assembler:
        with row_budget().claim(1):
            refused = await assembler.get_frame_view(frame_ref=frame_ref)
        assert requests == []
        admitted = await assembler.get_frame_view(frame_ref=frame_ref)

    assert isinstance(refused, Refusal)
    assert refused.error == "service_unavailable"
    assert refused.retry == RetryMode.RETRY_AFTER_DELAY
    assert refused.retry_after_s == ROW_BUDGET_RETRY_AFTER_S
    assert refused.fix == {
        "service": "presentation-mcp",
        "rows_requested": 1,
        "rows_free": 0,
        "rows_in_flight_max": 1,
    }
    assert not isinstance(admitted, Refusal)
    assert requests[0].url.path == "/api/datasets/frames/publish"
    assert in_flight_at_decode == [1]
    assert in_flight_at_encode == [1]
    assert in_flight_at_post == [1]
    assert row_budget().in_flight == 0


async def test_cancelled_publish_keeps_its_rows_claimed_until_the_compression_thread_ends(
    materialisation_store: InMemoryMaterialisationStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cancelling the publish stops the awaiting coroutine, but the compression thread
    keeps running and holding the body, so the frame's rows stay claimed until it ends."""
    await _put_ops_frame(materialisation_store)
    real_compress = gzip.compress
    compressing = threading.Event()
    gate = threading.Event()

    def gated(*args: Any, **kwargs: Any) -> bytes:
        compressing.set()
        assert gate.wait(timeout=10)
        return real_compress(*args, **kwargs)

    monkeypatch.setattr(client_module.gzip, "compress", gated)
    set_row_budget(RowBudget(max_rows=10))
    requests, handler = _publish_then_frame_handler(httpx2.Response(200, json={}))

    async with _assembler(handler) as assembler:
        publish = asyncio.create_task(
            assembler.get_frame_view(
                frame_ref={"spec_id": _OPS_SPEC_ID, "content_version": _OCV_CONTENT_VERSION}
            )
        )
        while not compressing.is_set():
            await asyncio.sleep(0.01)
        publish.cancel()
        with pytest.raises(asyncio.CancelledError):
            await publish

        assert row_budget().in_flight == 1

        gate.set()
        for _ in range(500):
            if row_budget().in_flight == 0:
                break
            await asyncio.sleep(0.01)

    assert row_budget().in_flight == 0
    assert requests == []


# --- a frame stored without a recorded row count, under a lowered ceiling ---

# A lowered `MAX_FRAME_ROWS`, under the store's own 250,000-row ceiling, with the
# smallest frame row budget the startup check allows beside it.
_LOWERED_MAX_FRAME_ROWS = 100_000


def _encode_stub_trace(trace: _StubTrace) -> dict[str, Any]:
    return {
        "upstream_trace": trace.upstream_trace,
        "frame_content_version": trace.frame_content_version,
        "input_content_versions": list(trace.input_content_versions),
    }


def _decode_stub_trace(record: dict[str, Any]) -> _StubTrace:
    return _StubTrace(
        upstream_trace=record["upstream_trace"],
        frame_content_version=record["frame_content_version"],
        input_content_versions=tuple(record["input_content_versions"]),
    )


register_trace_codec(
    _StubTrace,
    kind="presentation_mcp_test_stub_trace",
    version=1,
    encode=_encode_stub_trace,
    decode=_decode_stub_trace,
)


class _FakeGcsBlob:
    """`GcsMaterialisationStore`'s narrow blob seam over plain dicts. Duplicates
    `packages/common/tests/test_gcs_store.py`'s own fake, since a service's tests don't
    import another package's own test module."""

    def __init__(self, client: _FakeGcsClient, name: str) -> None:
        self._client = client
        self._name = name
        self.metadata: dict[str, str] | None = None
        self.size: int | None = None

    def upload_from_string(
        self, data: bytes, *, content_type: str, if_generation_match: int
    ) -> None:
        assert if_generation_match == 0
        self._client.objects[self._name] = data
        self._client.metadata[self._name] = dict(self.metadata or {})

    def download_as_bytes(self) -> bytes:
        self._client.downloads.append(self._name)
        return self._client.objects[self._name]

    def reload(self) -> None:
        self.metadata = dict(self._client.metadata[self._name])
        self.size = len(self._client.objects[self._name])


class _FakeGcsClient:
    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}
        self.metadata: dict[str, dict[str, str]] = {}
        self.downloads: list[str] = []

    def bucket(self, name: str) -> _FakeGcsClient:
        del name
        return self

    def blob(self, name: str) -> _FakeGcsBlob:
        return _FakeGcsBlob(self, name)

    def close(self) -> None:
        pass


@pytest.fixture
def lowered_max_frame_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MAX_FRAME_ROWS", str(_LOWERED_MAX_FRAME_ROWS))
    monkeypatch.setenv("FRAME_ROWS_IN_FLIGHT_MAX", str(2 * _LOWERED_MAX_FRAME_ROWS))
    set_row_budget(RowBudget(max_rows=2 * _LOWERED_MAX_FRAME_ROWS))


@pytest.fixture
async def gcs_frame_store() -> AsyncIterator[_FakeGcsClient]:
    client = _FakeGcsClient()
    set_store(GcsMaterialisationStore(bucket_name="frames", client=client))
    try:
        yield client
    finally:
        set_store(None)


async def _put_ops_rows(client: _FakeGcsClient, rows: int, *, record_rows: bool) -> None:
    """`rows` copies of `_put_ops_frame`'s one row, stored under the `ops_`/`ocv_` pair
    with or without the row count `put` records."""
    staging = InMemoryMaterialisationStore()
    await _put_ops_frame(staging)
    one_row, trace = await staging.get(
        FrameRef(spec_id=_OPS_SPEC_ID, content_version=_OCV_CONTENT_VERSION)
    ) or pytest.fail("the staged frame is held")
    frame = Frame.from_table(columns=one_row.columns, table=one_row.table.take([0] * rows))
    await get_store().put(
        FrameRef(spec_id=_OPS_SPEC_ID, content_version=_OCV_CONTENT_VERSION),
        frame=frame,
        trace=trace,
        writer="test",
    )
    if not record_rows:
        (metadata,) = client.metadata.values()
        del metadata["frame_rows"]


async def _publish_ops_frame() -> tuple[Refusal | dict[str, object], list[httpx2.Request]]:
    requests, handler = _publish_then_frame_handler(
        httpx2.Response(200, json={"spec_id": _OPS_SPEC_ID, "dimensions": {}})
    )
    async with _assembler(handler, settings=_settings()) as assembler:
        reply = await assembler.get_frame_view(
            frame_ref={"spec_id": _OPS_SPEC_ID, "content_version": _OCV_CONTENT_VERSION}
        )
    return reply, requests


@pytest.mark.usefixtures("lowered_max_frame_rows")
async def test_a_frame_with_no_recorded_row_count_publishes_under_a_lowered_ceiling(
    gcs_frame_store: _FakeGcsClient,
) -> None:
    await _put_ops_rows(gcs_frame_store, 1, record_rows=False)

    reply, requests = await _publish_ops_frame()

    assert not isinstance(reply, Refusal)
    assert requests[0].url.path == "/api/datasets/frames/publish"
    assert row_budget().high_water == _LOWERED_MAX_FRAME_ROWS
    assert row_budget().in_flight == 0


@pytest.mark.usefixtures("lowered_max_frame_rows")
async def test_a_frame_with_no_recorded_row_count_over_the_ceiling_refuses_without_decoding(
    gcs_frame_store: _FakeGcsClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    await _put_ops_rows(gcs_frame_store, _LOWERED_MAX_FRAME_ROWS + 1, record_rows=False)

    def _fail_if_called(data: bytes) -> Frame:
        raise AssertionError("the decode must not run past the claim")

    monkeypatch.setattr(gcs_store, "frame_from_verified_parquet_bytes", _fail_if_called)

    reply, requests = await _publish_ops_frame()

    assert isinstance(reply, Refusal)
    assert reply.error == "request_invalid"
    assert (
        f"stored frame has {_LOWERED_MAX_FRAME_ROWS + 1} rows, over this read's "
        f"{_LOWERED_MAX_FRAME_ROWS}-row limit"
    ) in reply.detail
    assert requests == []
    assert row_budget().in_flight == 0


@pytest.mark.usefixtures("lowered_max_frame_rows")
async def test_a_frame_with_a_recorded_row_count_over_the_ceiling_refuses_before_downloading(
    gcs_frame_store: _FakeGcsClient,
) -> None:
    await _put_ops_rows(gcs_frame_store, _LOWERED_MAX_FRAME_ROWS + 1, record_rows=True)

    reply, requests = await _publish_ops_frame()

    assert isinstance(reply, Refusal)
    assert reply.error == "request_invalid"
    assert (
        f"has {_LOWERED_MAX_FRAME_ROWS + 1} rows, over this service's "
        f"{_LOWERED_MAX_FRAME_ROWS}-row ceiling"
    ) in reply.detail
    assert requests == []
    assert gcs_frame_store.downloads == []
    assert row_budget().high_water == 0


async def test_concurrent_first_publishes_of_the_same_pair_single_flight(
    materialisation_store: InMemoryMaterialisationStore,
) -> None:
    """Two concurrent first calls for the same never-yet-published pair
    publish it exactly once -- `self._publish_locks`' single flight, not
    each racing its own POST before either records the memo.
    """
    await _put_ops_frame(materialisation_store)
    frame_ref = {"spec_id": _OPS_SPEC_ID, "content_version": _OCV_CONTENT_VERSION}
    requests: list[httpx2.Request] = []

    async def handler(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        if request.url.path == "/api/datasets/frames/publish":
            # Cedes control here so the second concurrent call has a chance
            # to run while the first is mid-publish -- without the lock it
            # would find no memo yet and post its own second publish.
            await asyncio.sleep(0)
            return httpx2.Response(
                200,
                json={
                    "schema_version": 1,
                    "published": {
                        "spec_id": _OPS_SPEC_ID,
                        "content_version": _OCV_CONTENT_VERSION,
                        "artifact_digest": "sha256:exemplar",
                        "created": True,
                    },
                },
            )
        return httpx2.Response(200, json={"spec_id": _OPS_SPEC_ID, "dimensions": {}})

    async with _assembler(handler) as assembler:
        first, second = await asyncio.gather(
            assembler.get_frame_view(frame_ref=frame_ref),
            assembler.get_frame_view(frame_ref=frame_ref),
        )

    assert not isinstance(first, Refusal)
    assert not isinstance(second, Refusal)
    publish_calls = [r for r in requests if r.url.path == "/api/datasets/frames/publish"]
    assert len(publish_calls) == 1


async def test_get_component_builds_the_path_and_forwards_revision() -> None:
    requests, handler = _counting_handler(
        httpx2.Response(200, json={"component_id": "cmp_1", "revision": 3, "intent": {}})
    )

    async with _assembler(handler) as assembler:
        document = await assembler.get_component(component_id="cmp_1", revision=3)

    assert not isinstance(document, Refusal)
    request = requests[0]
    assert request.method == "GET"
    assert request.url.path == "/api/presentation/components/cmp_1"
    assert dict(request.url.params) == {"revision": "3"}


async def test_get_component_forwards_the_resolved_payload_unread() -> None:
    body = {"component_id": "cmp_1", "revision": 1, "resolved_payload": ENVELOPE}
    _requests, handler = _counting_handler(httpx2.Response(200, json=body))

    async with _assembler(handler) as assembler:
        document = await assembler.get_component(component_id="cmp_1")

    assert document == body


async def test_get_component_omits_revision_param_when_absent() -> None:
    requests, handler = _counting_handler(
        httpx2.Response(200, json={"component_id": "cmp_1", "revision": 1, "intent": {}})
    )

    async with _assembler(handler) as assembler:
        await assembler.get_component(component_id="cmp_1")

    assert dict(requests[0].url.params) == {}


async def test_release_returns_the_whole_document() -> None:
    requests, handler = _counting_handler(httpx2.Response(200, json=WRITE_BODY))

    async with _assembler(handler) as assembler:
        document = await assembler.release()

    assert document == WRITE_BODY
    assert requests[0].method == "GET"
    assert requests[0].url.path == "/api/presentation/release"


# --- identity headers ------------------------------------------------------


async def test_no_session_header_goes_to_the_platform() -> None:
    """The component id and its write token are the grants, so no call
    carries a session, and only the revise carries the token."""
    requests, handler = _counting_handler(httpx2.Response(200, json=WRITE_BODY))

    async with _assembler(handler) as assembler:
        await assembler.create_component(intent=INTENT, caller=Caller.EXTERNAL)
        await assembler.revise_component(
            component_id="cmp_1", patch={}, write_token=WRITE_TOKEN, caller=Caller.EXTERNAL
        )
        await assembler.get_frame_view(frame_ref=FRAME_REF)
        await assembler.get_component(component_id="cmp_1")
        await assembler.release()

    assert len(requests) == 5
    for request in requests:
        assert not any("session" in key.lower() for key in request.headers)
    assert [WRITE_TOKEN_HEADER in request.headers for request in requests] == [
        False,
        True,
        False,
        False,
        False,
    ]


async def test_caller_never_reaches_the_wire() -> None:
    requests, handler = _counting_handler(httpx2.Response(201, json=WRITE_BODY))

    async with _assembler(handler) as assembler:
        await assembler.create_component(intent=INTENT, caller=Caller.FIRST_PARTY)

    request = requests[0]
    assert "caller" not in json.loads(request.content)
    assert not any("caller" in key.lower() for key in request.headers)


# --- WriteReply mapping ------------------------------------------------------


async def test_write_reply_maps_and_drops_composition_and_affordance_fields() -> None:
    _requests, handler = _counting_handler(
        httpx2.Response(
            201,
            json=WRITE_BODY
            | {"changed": [{"field": "title", "prior": "Old", "new": "New"}, {"field": "select"}]},
        )
    )

    async with _assembler(handler) as assembler:
        reply = await assembler.create_component(intent=INTENT, caller=Caller.EXTERNAL)

    assert isinstance(reply, WriteReply)
    assert reply.component_id == "cmp_1"
    assert reply.revision == 1
    assert reply.fact_summary == {"recipient": "country/ETH"}
    assert reply.changed == (
        {"field": "title", "prior": "Old", "new": "New"},
        {"field": "select"},
    )
    assert "affordance" not in dataclasses.asdict(reply)
    field_names = {f.name for f in dataclasses.fields(WriteReply)}
    assert "catalog_version" not in field_names
    assert "composition_id" not in field_names
    assert "composition_revision" not in field_names
    assert "affordance" not in field_names


async def test_write_reply_keeps_the_resolved_payload_verbatim() -> None:
    _requests, handler = _counting_handler(httpx2.Response(201, json=WRITE_BODY))

    async with _assembler(handler) as assembler:
        created = await assembler.create_component(intent=INTENT, caller=Caller.EXTERNAL)
        revised = await assembler.revise_component(
            component_id="cmp_1",
            patch={"title": "New"},
            write_token=WRITE_TOKEN,
            caller=Caller.EXTERNAL,
        )

    assert isinstance(created, WriteReply)
    assert isinstance(revised, WriteReply)
    assert created.resolved_payload == ENVELOPE
    assert revised.resolved_payload == ENVELOPE


async def test_write_reply_without_a_resolved_payload_carries_none() -> None:
    body = {k: v for k, v in WRITE_BODY.items() if k != "resolved_payload"}
    _requests, handler = _counting_handler(httpx2.Response(201, json=body))

    async with _assembler(handler) as assembler:
        reply = await assembler.create_component(intent=INTENT, caller=Caller.EXTERNAL)

    assert isinstance(reply, WriteReply)
    assert reply.resolved_payload is None


async def test_create_reply_carries_an_empty_changed_tuple() -> None:
    _requests, handler = _counting_handler(httpx2.Response(201, json=WRITE_BODY))

    async with _assembler(handler) as assembler:
        reply = await assembler.create_component(intent=INTENT, caller=Caller.EXTERNAL)

    assert isinstance(reply, WriteReply)
    assert reply.changed == ()


# --- Refusal translation -----------------------------------------------------


async def test_retryable_problem_composes_a_retry_mode_from_the_cached_release_table() -> None:
    release_body = {"retry_modes": {"binding_invalid": "correct_and_retry"}}
    responses = iter(
        [
            httpx2.Response(200, json=release_body),
            httpx2.Response(400, content=_problem_body(code="binding_invalid", retryable=True)),
        ]
    )

    def handler(_request: httpx2.Request) -> httpx2.Response:
        return next(responses)

    async with _assembler(handler) as assembler:
        await assembler.release()
        reply = await assembler.get_component(component_id="cmp_1")

    assert isinstance(reply, Refusal)
    assert reply.error == "binding_invalid"
    assert reply.retry == RetryMode.CORRECT_AND_RETRY
    assert reply.detail == "because"


async def test_a_platform_refusal_leaves_the_platforms_trace_id_out_of_the_reply_and_logs_it(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The platform's `trace_id` stays out of the refusal on the wire and lands
    in a warning with the error code. `asdict` with `None` dropped is the shape
    `server._reply` puts on the wire."""
    body = json.dumps(
        {
            "code": "internal_error",
            "detail": "Internal error",
            "trace_id": "41d94ed2",
            "retryable": False,
        }
    ).encode()
    _requests, handler = _counting_handler(httpx2.Response(500, content=body))
    caplog.set_level(logging.WARNING, logger=client_module.__name__)

    async with _assembler(handler) as assembler:
        reply = await assembler.create_component(intent=INTENT, caller=Caller.EXTERNAL)

    assert isinstance(reply, Refusal)
    assert reply.error == "internal_error"
    assert reply.retry == RetryMode.DO_NOT_RETRY
    wire = {key: value for key, value in dataclasses.asdict(reply).items() if value is not None}
    assert wire == {
        "error": "internal_error",
        "retry": RetryMode.DO_NOT_RETRY,
        "detail": "Internal error",
    }
    assert "41d94ed2" not in json.dumps(wire)
    [record] = [r for r in caplog.records if r.name == "presentation_mcp.client"]
    assert "41d94ed2" in record.getMessage()
    assert "internal_error" in record.getMessage()


@pytest.mark.parametrize(
    "trace_id",
    [
        pytest.param("", id="empty"),
        pytest.param(41, id="not-a-string"),
        pytest.param(None, id="null"),
    ],
)
async def test_a_platform_refusal_without_a_usable_trace_id_logs_nothing(
    trace_id: object, caplog: pytest.LogCaptureFixture
) -> None:
    body = _problem_body(code="binding_invalid", retryable=False, trace_id=trace_id)
    _requests, handler = _counting_handler(httpx2.Response(400, content=body))
    caplog.set_level(logging.WARNING, logger=client_module.__name__)

    async with _assembler(handler) as assembler:
        reply = await assembler.get_component(component_id="cmp_1")

    assert isinstance(reply, Refusal)
    assert [r for r in caplog.records if r.name == "presentation_mcp.client"] == []


async def test_retryable_false_forces_do_not_retry_even_when_the_table_says_otherwise() -> None:
    release_body = {"retry_modes": {"not_found": "correct_and_retry"}}
    responses = iter(
        [
            httpx2.Response(200, json=release_body),
            httpx2.Response(404, content=_problem_body(code="not_found", retryable=False)),
        ]
    )

    def handler(_request: httpx2.Request) -> httpx2.Response:
        return next(responses)

    async with _assembler(handler) as assembler:
        await assembler.release()
        reply = await assembler.get_component(component_id="cmp_1")

    assert isinstance(reply, Refusal)
    assert reply.retry == RetryMode.DO_NOT_RETRY


async def test_unknown_code_falls_back_to_retry_after_delay() -> None:
    _requests, handler = _counting_handler(
        httpx2.Response(503, content=_problem_body(code="service_unavailable", retryable=True))
    )

    async with _assembler(handler) as assembler:
        # No release() call: the retry-mode table stays empty.
        reply = await assembler.get_component(component_id="cmp_1")

    assert isinstance(reply, Refusal)
    assert reply.error == "service_unavailable"
    assert reply.retry == RetryMode.RETRY_AFTER_DELAY


async def test_unparseable_5xx_refuses_transient_unavailable() -> None:
    _requests, handler = _counting_handler(httpx2.Response(500, content=b"<html>oops</html>"))
    settings = _settings(upstream_max_attempts=1)

    async with _assembler(handler, settings=settings) as assembler:
        reply = await assembler.get_component(component_id="cmp_1")

    assert isinstance(reply, Refusal)
    assert reply.error == "transient_unavailable"
    assert reply.retry == RetryMode.RETRY_AFTER_DELAY


async def test_unparseable_4xx_refuses_request_invalid() -> None:
    _requests, handler = _counting_handler(httpx2.Response(400, content=b"plain text"))

    async with _assembler(handler) as assembler:
        reply = await assembler.get_component(component_id="cmp_1")

    assert isinstance(reply, Refusal)
    assert reply.error == "request_invalid"
    assert reply.retry == RetryMode.DO_NOT_RETRY


async def test_unclassified_sub_500_status_detail_carries_no_status_code() -> None:
    """`_refusal`'s no-problem-body branch keeps `detail` free of the
    numeric status code. `detail` carries prose for an integrator to read.
    """
    _requests, handler = _counting_handler(httpx2.Response(307))

    async with _assembler(handler) as assembler:
        reply = await assembler.get_component(component_id="cmp_1")

    assert isinstance(reply, Refusal)
    assert reply.error == "request_invalid"
    assert not any(char.isdigit() for char in reply.detail)


async def test_unclassified_500_or_more_status_detail_carries_no_status_code() -> None:
    settings = _settings(upstream_max_attempts=1)
    _requests, handler = _counting_handler(httpx2.Response(503))

    async with _assembler(handler, settings=settings) as assembler:
        reply = await assembler.get_component(component_id="cmp_1")

    assert isinstance(reply, Refusal)
    assert reply.error == "transient_unavailable"
    assert not any(char.isdigit() for char in reply.detail)


async def test_transport_error_refuses_transient_unavailable() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ConnectError("boom", request=request)

    _delays, sleep = _recording_sleep()

    async with _assembler(handler, sleep=sleep) as assembler:
        reply = await assembler.get_component(component_id="cmp_1")

    assert isinstance(reply, Refusal)
    assert reply.error == "transient_unavailable"
    assert reply.retry == RetryMode.RETRY_AFTER_DELAY


async def test_fix_renders_alternatives_issues_constraint_and_truncation() -> None:
    problem = _problem_body(
        code="binding_invalid",
        retryable=True,
        constraint_id="log_scale_requires_positive_values",
        alternatives=[
            {"ref": "country/USA", "label": "United States"},
            {"ref": "country/COD", "label": "country/COD"},
        ],
        issues=[
            {
                "axis": "recipient",
                "alternatives": [{"ref": "country/KEN", "label": "Kenya"}],
            }
        ],
        truncated=True,
    )
    _requests, handler = _counting_handler(httpx2.Response(400, content=problem))

    async with _assembler(handler) as assembler:
        reply = await assembler.get_component(component_id="cmp_1")

    assert isinstance(reply, Refusal)
    assert reply.fix is not None
    assert "log_scale_requires_positive_values" in reply.fix
    # A retry's select names refs, so each value carries its ref beside a
    # label that differs from it.
    assert "United States (country/USA), country/COD" in reply.fix
    assert "recipient: Kenya (country/KEN)" in reply.fix
    assert "more values exist" in reply.fix


async def test_issues_render_their_own_truncation_independent_of_the_top_level_flag() -> None:
    problem = _problem_body(
        code="binding_invalid",
        retryable=True,
        truncated=False,
        issues=[
            {
                "axis": "recipient",
                "alternatives": [{"ref": "country/KEN", "label": "country/KEN"}],
                "truncated": True,
            },
            {
                "axis": "prices",
                "alternatives": [{"ref": "constant", "label": "constant"}],
                "truncated": False,
            },
        ],
    )
    _requests, handler = _counting_handler(httpx2.Response(400, content=problem))

    async with _assembler(handler) as assembler:
        reply = await assembler.get_component(component_id="cmp_1")

    assert isinstance(reply, Refusal)
    assert reply.fix is not None
    recipient_part, prices_part = (part.strip() for part in reply.fix.split(";"))
    assert "recipient" in recipient_part and "more values exist" in recipient_part
    assert "prices" in prices_part and "more values exist" not in prices_part


async def test_pointer_passes_through_unrepaired() -> None:
    problem = _problem_body(code="binding_invalid", retryable=True, pointer="/select/recipient")
    _requests, handler = _counting_handler(httpx2.Response(400, content=problem))

    async with _assembler(handler) as assembler:
        reply = await assembler.get_component(component_id="cmp_1")

    assert isinstance(reply, Refusal)
    assert reply.pointer == "/select/recipient"


# --- local refusal: malformed route segments ---------------------------------


async def test_frame_ref_missing_content_version_refuses_locally_without_a_call() -> None:
    calls, handler = _counting_handler(httpx2.Response(200, json={}))

    async with _assembler(handler) as assembler:
        reply = await assembler.get_frame_view(frame_ref={"spec_id": "spec_1"})

    assert isinstance(reply, Refusal)
    assert reply.error == "request_invalid"
    assert reply.retry == RetryMode.DO_NOT_RETRY
    assert calls == []


@pytest.mark.parametrize("component_id", _INVALID_COMPONENT_IDS)
async def test_get_component_refuses_invalid_component_id_locally_without_a_call(
    component_id: str,
) -> None:
    calls, handler = _counting_handler(httpx2.Response(200, json={}))

    async with _assembler(handler) as assembler:
        reply = await assembler.get_component(component_id=component_id)

    assert isinstance(reply, Refusal)
    assert reply.error == "request_invalid"
    assert reply.retry == RetryMode.DO_NOT_RETRY
    assert reply.pointer == "/component_id"
    assert calls == []


@pytest.mark.parametrize("component_id", _INVALID_COMPONENT_IDS)
async def test_revise_component_refuses_invalid_component_id_locally_without_a_call(
    component_id: str,
) -> None:
    calls, handler = _counting_handler(httpx2.Response(200, json=WRITE_BODY))

    async with _assembler(handler) as assembler:
        reply = await assembler.revise_component(
            component_id=component_id,
            patch={"title": "New"},
            write_token=WRITE_TOKEN,
            caller=Caller.EXTERNAL,
        )

    assert isinstance(reply, Refusal)
    assert reply.error == "request_invalid"
    assert reply.retry == RetryMode.DO_NOT_RETRY
    assert reply.pointer == "/component_id"
    assert calls == []


async def test_get_component_with_a_well_formed_unknown_id_still_calls_the_platform() -> None:
    """A well-formed but unrecognized id still reaches the platform."""
    problem = _problem_body(code="not_found", retryable=False)
    calls, handler = _counting_handler(httpx2.Response(404, content=problem))

    async with _assembler(handler) as assembler:
        reply = await assembler.get_component(component_id="cmp_missing")

    assert isinstance(reply, Refusal)
    assert reply.error == "not_found"
    assert len(calls) == 1


async def test_frame_ref_missing_spec_id_refuses_locally_without_a_call() -> None:
    calls, handler = _counting_handler(httpx2.Response(200, json={}))

    async with _assembler(handler) as assembler:
        reply = await assembler.get_frame_view(frame_ref={"content_version": "cv_1"})

    assert isinstance(reply, Refusal)
    assert reply.error == "request_invalid"
    assert calls == []


@pytest.mark.parametrize(
    ("field", "value"),
    [("spec_id", value) for value in _INVALID_SPEC_IDS]
    + [("content_version", value) for value in _INVALID_CONTENT_VERSIONS],
)
async def test_frame_ref_with_an_invalid_path_segment_refuses_locally_without_a_call(
    field: str,
    value: str,
) -> None:
    calls, handler = _counting_handler(httpx2.Response(200, json={}))
    frame_ref = FRAME_REF | {field: value}

    async with _assembler(handler) as assembler:
        reply = await assembler.get_frame_view(frame_ref=frame_ref)

    assert isinstance(reply, Refusal)
    assert reply.error == "request_invalid"
    assert reply.retry == RetryMode.DO_NOT_RETRY
    assert reply.pointer == f"/frame_ref/{field}"
    assert calls == []


# --- retry behaviour ----------------------------------------------------------


async def test_transport_errors_retry_up_to_max_attempts() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ConnectError("boom", request=request)

    delays, sleep = _recording_sleep()
    settings = _settings(upstream_max_attempts=3)

    async with _assembler(handler, settings=settings, sleep=sleep) as assembler:
        reply = await assembler.get_component(component_id="cmp_1")

    assert isinstance(reply, Refusal)
    assert len(delays) == settings.upstream_max_attempts - 1


async def test_retryable_5xx_retries_up_to_max_attempts() -> None:
    calls, handler = _counting_handler(
        httpx2.Response(503, content=_problem_body(code="service_unavailable", retryable=True))
    )
    _delays, sleep = _recording_sleep()
    settings = _settings(upstream_max_attempts=3)

    async with _assembler(handler, settings=settings, sleep=sleep) as assembler:
        reply = await assembler.get_component(component_id="cmp_1")

    assert isinstance(reply, Refusal)
    assert len(calls) == settings.upstream_max_attempts


async def test_non_retryable_5xx_is_attempted_once() -> None:
    calls, handler = _counting_handler(
        httpx2.Response(503, content=_problem_body(code="release_mismatch", retryable=False))
    )

    async with _assembler(handler) as assembler:
        reply = await assembler.get_component(component_id="cmp_1")

    assert isinstance(reply, Refusal)
    assert len(calls) == 1


async def test_4xx_is_never_retried() -> None:
    calls, handler = _counting_handler(
        httpx2.Response(400, content=_problem_body(code="schema_invalid", retryable=True))
    )
    _delays, sleep = _recording_sleep()

    async with _assembler(handler, sleep=sleep) as assembler:
        reply = await assembler.get_component(component_id="cmp_1")

    assert isinstance(reply, Refusal)
    assert len(calls) == 1


async def test_backoff_delays_use_full_jitter_up_to_the_exponential_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("presentation_mcp.client.random.uniform", lambda _low, high: high)
    calls, handler = _counting_handler(
        httpx2.Response(503, content=_problem_body(code="service_unavailable", retryable=True))
    )
    delays, sleep = _recording_sleep()
    settings = _settings(
        upstream_max_attempts=3, upstream_backoff_base_s=1.0, upstream_retry_budget_s=100.0
    )

    async with _assembler(handler, settings=settings, sleep=sleep) as assembler:
        await assembler.get_component(component_id="cmp_1")

    assert len(calls) == 3
    assert delays == [1.0, 2.0]


# --- create's retry policy ----------------------------------------------------


def _raising_handler(
    exc_factory: Callable[[httpx2.Request], Exception],
) -> tuple[list[httpx2.Request], Handler]:
    """A handler that always raises, recording every request it saw first."""
    requests: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        raise exc_factory(request)

    return requests, handler


def _sequence_handler(
    *outcomes: httpx2.Response | Exception,
) -> tuple[list[httpx2.Request], Handler]:
    """A handler that answers each request with the next outcome, raising exceptions."""
    requests: list[httpx2.Request] = []
    remaining = iter(outcomes)

    def handler(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        outcome = next(remaining)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    return requests, handler


def _write_tokens(requests: list[httpx2.Request]) -> list[str]:
    return [json.loads(request.content)["write_token"] for request in requests]


async def test_create_retries_a_connect_phase_transport_error() -> None:
    calls, handler = _raising_handler(lambda request: httpx2.ConnectError("boom", request=request))
    _delays, sleep = _recording_sleep()
    settings = _settings(upstream_max_attempts=3)

    async with _assembler(handler, settings=settings, sleep=sleep) as assembler:
        reply = await assembler.create_component(intent=INTENT, caller=Caller.EXTERNAL)

    assert isinstance(reply, Refusal)
    assert len(calls) == settings.upstream_max_attempts


async def test_a_failed_token_mint_refuses_transient_unavailable_and_sends_nothing() -> None:
    calls, handler = _raising_handler(lambda request: httpx2.ConnectError("boom", request=request))

    async def no_token() -> Mapping[str, str]:
        raise ApisAuthError("metadata server answered 500")

    async with PlatformAssembler(
        settings=_settings(), auth=no_token, transport=httpx2.MockTransport(handler)
    ) as assembler:
        reply = await assembler.create_component(intent=INTENT, caller=Caller.EXTERNAL)

    assert isinstance(reply, Refusal)
    assert (reply.error, reply.retry) == ("transient_unavailable", RetryMode.RETRY_AFTER_DELAY)
    assert calls == []


async def test_create_retries_a_read_phase_transport_error_with_the_same_token() -> None:
    requests, handler = _sequence_handler(
        httpx2.ReadTimeout("boom", request=httpx2.Request("POST", "http://test")),
        httpx2.Response(201, json=WRITE_BODY),
    )
    _delays, sleep = _recording_sleep()

    async with _assembler(handler, sleep=sleep) as assembler:
        reply = await assembler.create_component(intent=INTENT, caller=Caller.EXTERNAL)

    assert isinstance(reply, WriteReply)
    tokens = _write_tokens(requests)
    assert len(tokens) == 2
    assert tokens[0] == tokens[1]
    assert _CREATE_TOKEN_PATTERN.fullmatch(tokens[0])


async def test_create_retries_a_timeout_504_with_the_same_token_and_returns_the_201() -> None:
    requests, handler = _sequence_handler(
        httpx2.Response(504, content=_problem_body(code="timeout", retryable=True)),
        httpx2.Response(201, json=WRITE_BODY | {"write_token": WRITE_TOKEN}),
    )
    _delays, sleep = _recording_sleep()

    async with _assembler(handler, sleep=sleep) as assembler:
        reply = await assembler.create_component(intent=INTENT, caller=Caller.EXTERNAL)

    assert isinstance(reply, WriteReply)
    assert reply.component_id == "cmp_1"
    assert reply.write_token == WRITE_TOKEN
    tokens = _write_tokens(requests)
    assert len(tokens) == 2
    assert tokens[0] == tokens[1]
    assert _CREATE_TOKEN_PATTERN.fullmatch(tokens[0])


async def test_separate_creates_send_different_tokens() -> None:
    requests, handler = _counting_handler(httpx2.Response(201, json=WRITE_BODY))

    async with _assembler(handler) as assembler:
        await assembler.create_component(intent=INTENT, caller=Caller.EXTERNAL)
        await assembler.create_component(intent=INTENT, caller=Caller.EXTERNAL)

    first, second = _write_tokens(requests)
    assert first != second


async def test_revise_component_retries_a_read_phase_transport_error() -> None:
    calls, handler = _raising_handler(lambda request: httpx2.ReadTimeout("boom", request=request))
    _delays, sleep = _recording_sleep()
    settings = _settings(upstream_max_attempts=3)

    async with _assembler(handler, settings=settings, sleep=sleep) as assembler:
        reply = await assembler.revise_component(
            component_id="cmp_1",
            patch={"title": "New"},
            write_token=WRITE_TOKEN,
            caller=Caller.EXTERNAL,
        )

    assert isinstance(reply, Refusal)
    assert len(calls) == settings.upstream_max_attempts


async def test_get_component_retries_a_read_phase_transport_error() -> None:
    calls, handler = _raising_handler(lambda request: httpx2.ReadTimeout("boom", request=request))
    _delays, sleep = _recording_sleep()
    settings = _settings(upstream_max_attempts=3)

    async with _assembler(handler, settings=settings, sleep=sleep) as assembler:
        reply = await assembler.get_component(component_id="cmp_1")

    assert isinstance(reply, Refusal)
    assert len(calls) == settings.upstream_max_attempts


async def test_create_retries_a_retryable_5xx_problem_response() -> None:
    calls, handler = _counting_handler(
        httpx2.Response(503, content=_problem_body(code="service_unavailable", retryable=True))
    )
    _delays, sleep = _recording_sleep()
    settings = _settings(upstream_max_attempts=3)

    async with _assembler(handler, settings=settings, sleep=sleep) as assembler:
        reply = await assembler.create_component(intent=INTENT, caller=Caller.EXTERNAL)

    assert isinstance(reply, Refusal)
    assert reply.error == "service_unavailable"
    assert len(calls) == settings.upstream_max_attempts


# --- request()'s own exceptions (used directly by callers other than the five) --


async def test_request_raises_platform_error_on_a_terminal_failure() -> None:
    _calls, handler = _counting_handler(
        httpx2.Response(422, content=_problem_body(code="over_ceiling", retryable=False))
    )

    async with _assembler(handler) as assembler:
        with pytest.raises(PlatformError) as exc_info:
            await assembler.request(
                "GET",
                "/api/presentation/release",
                route="/api/presentation/release",
            )

    assert exc_info.value.status == 422
    assert exc_info.value.code == "over_ceiling"


async def test_request_raises_platform_transport_error_when_every_attempt_fails() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ConnectError("boom", request=request)

    _delays, sleep = _recording_sleep()

    async with _assembler(handler, sleep=sleep) as assembler:
        with pytest.raises(PlatformTransportError):
            await assembler.request(
                "GET",
                "/api/presentation/release",
                route="/api/presentation/release",
            )
