"""Tests for the `get_data` tool: the mint call, scope surgery, and refusals."""

from __future__ import annotations

import asyncio
import functools
import hashlib
import io
import json
import logging
from collections.abc import Callable, Coroutine, Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any, ClassVar

import httpx2
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from common.apis_auth import AuthProvider
from common.caller import conversation_caller, supplied_caller
from common.materialisation import (
    FrameRef,
    InMemoryMaterialisationStore,
    MintRecord,
    frame_from_verified_parquet_bytes,
    frame_to_parquet_bytes,
)
from common.materialisation import set_store as set_frame_store
from common.place_type import PARTY_KINDS
from data_mcp import client_holder, get_data_tool, mint_admission
from data_mcp.client import UpstreamClient
from data_mcp.deadline import EXTERNAL_CALL_BUDGET, CallBudget, CallDeadline, use_call_budget
from data_mcp.definition_store import (
    DefinitionStoreUnavailable,
    IdentityMismatch,
    StoredDefinition,
    canonicalize,
    set_store,
    spec_id_for,
)
from data_mcp.errors import UpstreamError
from data_mcp.frame_materialisation import (
    CANONICAL_FRAME_RENDERING,
    KIND_COLUMNS,
    MaterialisationOutcome,
    frame_ref_for,
)
from data_mcp.get_data_tool import RootMint
from data_mcp.manifest import Manifest, SeriesStatisticsModel, strip_manifest
from data_mcp.mint_admission import (
    BACKGROUND_MINT_SPAN_ATTRIBUTE,
    BACKGROUND_OUTCOME_KEY,
    MINT_ADMISSION_HEADER,
    MINT_ADMISSION_META_KEY,
    background_outcome,
)
from data_mcp.operations import QueuedOperation, ResolvedOperation
from data_mcp.series_statistics import SeriesStatisticsSchemaError
from data_mcp.server import mcp
from data_mcp.settings import Settings
from data_mcp.stat_var_list import (
    COMPONENT_STAT_VARS_OMITTED_KEY,
    MAX_LISTED_STAT_VARS,
    STAT_VARS_OMITTED_KEY,
    with_stat_var_list,
)
from dfo_frame_grammar import Frame, frame_sha256, mint_content_version
from fastmcp import Client
from fastmcp.client.client import CallToolResult
from fastmcp.exceptions import ResourceError
from fastmcp.tools import ToolResult
from mcp.types import ResourceLink, TextContent
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from .net_guard import no_network  # noqa: F401 -- imported for its autouse fixture
from .node_fixtures import load_node_fixture
from .resolve_results import members_result, resolve_result, resolved_result
from .test_client import _client, _problem_body, _settings, id_token_auth, text_of
from .test_operations import _FakeClock

Handler = (
    Callable[[httpx2.Request], httpx2.Response]
    | Callable[[httpx2.Request], Coroutine[None, None, httpx2.Response]]
)


def mint_pair(manifest: Manifest) -> FrameRef:
    """The manifest's own pair, which a frame minted with the default columns is
    held under. Built unvalidated, since most manifests here carry a short
    `content_version` like `"cv_1"`."""
    return FrameRef.model_construct(
        spec_id=manifest.definition.spec_id, content_version=manifest.content_version
    )


@pytest.fixture(autouse=True)
def _stub_materialise_and_register(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every test in this module drives the mint path only, most with a
    manifest carrying a plain `content_version` like `"cv_1"` that doesn't
    match `FrameRef`'s own pair pattern -- the real `materialise_and_register`
    would refuse those outright. Stubbed to a no-op success for every test
    here except the ones below that override it again to exercise the real
    fetch-parse-register path end to end.
    """

    async def _stub(*, manifest: Manifest, **_: Any) -> MaterialisationOutcome:
        return MaterialisationOutcome(
            status="registered", attempts=0, frame_ref=mint_pair(manifest)
        )

    monkeypatch.setattr(get_data_tool, "materialise_and_register", _stub)


@pytest.fixture(autouse=True)
def _stub_description_read() -> Iterator[None]:
    """Every test here renders without a version description, so its handler need not
    answer the description route. `test_version_description.py` drives the read.

    Patched on a context of its own, so a test's `monkeypatch.undo()`, which removes
    the materialisation stub, keeps this one.
    """

    async def _no_description(**_: Any) -> None:
        return None

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(get_data_tool, "fetch_version_description", _no_description)
        yield


# Store doubles


class _StubStore:
    """A store that always answers `stored` for `.get()`, whatever the id."""

    def __init__(self, stored: StoredDefinition | None) -> None:
        self._stored = stored
        self.get_calls: list[tuple[str, int]] = []

    async def put_many(self, definitions: Any) -> None:
        raise AssertionError("no re-mint is expected in this test")

    async def get(self, spec_id: str, *, schema_version: int) -> StoredDefinition | None:
        self.get_calls.append((spec_id, schema_version))
        return self._stored

    async def aclose(self) -> None:
        pass


class _FailingGetStore:
    """A store whose backend is unreachable: every `.get()` refuses."""

    async def put_many(self, definitions: Any) -> None:
        raise AssertionError("no re-mint is expected in this test")

    async def get(self, spec_id: str, *, schema_version: int) -> StoredDefinition | None:
        raise DefinitionStoreUnavailable("backend unreachable")

    async def aclose(self) -> None:
        pass


class _MismatchStore:
    """A store whose held bytes no longer reproduce the id they're keyed under."""

    async def put_many(self, definitions: Any) -> None:
        raise AssertionError("no re-mint is expected in this test")

    async def get(self, spec_id: str, *, schema_version: int) -> StoredDefinition | None:
        raise IdentityMismatch(expected_spec_id=spec_id, actual_spec_id="spec_" + "0" * 32)

    async def aclose(self) -> None:
        pass


class _RecordingStore:
    """A store seeded with definitions and answering `.get()` from them, that also records
    and applies every `.put_many()` write -- so a definition written by one `get_data` call is
    visible to a later `.get()` against the same instance, the way the real store is.
    """

    def __init__(self, *seeded: StoredDefinition) -> None:
        self._by_id: dict[tuple[str, int], StoredDefinition] = {
            (d.spec_id, d.schema_version): d for d in seeded
        }
        self.put_calls: list[StoredDefinition] = []

    async def put_many(self, definitions: Any) -> None:
        for definition in definitions:
            self.put_calls.append(definition)
            self._by_id[(definition.spec_id, definition.schema_version)] = definition

    async def get(self, spec_id: str, *, schema_version: int) -> StoredDefinition | None:
        return self._by_id.get((spec_id, schema_version))

    async def aclose(self) -> None:
        pass


class _PutFailsStore:
    """A store whose `.get()` answers normally but whose backend refuses every write."""

    def __init__(self, stored: StoredDefinition) -> None:
        self._stored = stored

    async def put_many(self, definitions: Any) -> None:
        raise DefinitionStoreUnavailable("backend unreachable")

    async def get(self, spec_id: str, *, schema_version: int) -> StoredDefinition | None:
        return self._stored

    async def aclose(self) -> None:
        pass


class _SlowPutStore:
    """A store whose `.get()` answers normally but whose write outruns any real budget
    it's given, so a call bounded by the deadline times out under it -- distinct from
    `_PutFailsStore`'s backend-refuses-outright case."""

    def __init__(self, stored: StoredDefinition) -> None:
        self._stored = stored

    async def put_many(self, definitions: Any) -> None:
        await asyncio.sleep(1.0)

    async def get(self, spec_id: str, *, schema_version: int) -> StoredDefinition | None:
        return self._stored

    async def aclose(self) -> None:
        pass


# Fixtures: a real identity document that reproduces its own spec_id


def _identity_doc(
    *,
    entities: dict[str, list[str]] | None = None,
    shape_id: str = "MalariaGrantDisbursements",
    open_roles: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "identity_version": "spec1",
        "shape": {"id": shape_id, "kind": "stat_var"},
        "entities": entities if entities is not None else {"recipient": ["country/KEN"]},
        "facets": {"provenance": {"ref": "OECD_DAC2A", "state": "value"}},
        "requested_window": {"kind": "all"},
        "slots": [],
        **({"open_roles": open_roles} if open_roles is not None else {}),
    }


def _stored_definition(**identity_kwargs: Any) -> StoredDefinition:
    identity = _identity_doc(**identity_kwargs)
    canonical_bytes = canonicalize(identity)
    return StoredDefinition(
        spec_id=spec_id_for(canonical_bytes), schema_version=1, canonical_bytes=canonical_bytes
    )


def _manifest_response(
    *,
    spec_id: str,
    content_version: str = "cv_1",
    entities: dict[str, list[str]] | None = None,
    shape_id: str = "MalariaGrantDisbursements",
    artifact_digest: str = "sha256:" + "1" * 64,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "definition": {
            "schema_version": 1,
            "spec_id": spec_id,
            "identity": _identity_doc(entities=entities, shape_id=shape_id),
        },
        "content_version": content_version,
        "coverage": {
            "rows": 1,
            "dimensions": [{"count": 1, "label": "recipient"}],
            "window": {"start_year": 2015, "end_year": 2015},
        },
        "citation": {
            "text": "OECD DAC2a",
            "sources": [{"ref": "OECD_DAC2A", "label": "OECD DAC2a"}],
        },
        "trace": {},
        "warnings": [],
        "artifacts": [
            {
                "artifact_digest": artifact_digest,
                "rendering": {"bundle": False, "format": "parquet", "layout": "frame"},
                "bytes": 100,
                "media_type": "application/octet-stream",
                "created_at": "2026-08-11T00:00:00+00:00",
                "tier": "working",
                "url": "/links/should-not-leak",
            }
        ],
    }


def _manifest_response_with_two_artifacts(
    *, spec_id: str, content_version: str = "cv_1"
) -> dict[str, Any]:
    """`_manifest_response`, but with a second, distinct artifact: a csv rendering
    another caller stored under the same pair, listed after the requested one."""
    response = _manifest_response(spec_id=spec_id, content_version=content_version)
    second = dict(response["artifacts"][0])
    second["artifact_digest"] = "sha256:" + "2" * 64
    second["rendering"] = {"bundle": False, "format": "csv"}
    second["url"] = "/links/should-not-leak-either"
    response["artifacts"].append(second)
    return response


def _manifest_response_with_statistics(
    *, spec_id: str, content_version: str = "cv_1", series_count: int = 1
) -> dict[str, Any]:
    """`_manifest_response`, plus `series_count` and a matching statistics artifact entry."""
    response = _manifest_response(spec_id=spec_id, content_version=content_version)
    response["series_count"] = series_count
    response["artifacts"].append(
        {
            "artifact_digest": "sha256:" + "5" * 64,
            "rendering": {"statistics": "series"},
            "bytes": 200,
            "media_type": "application/vnd.apache.parquet",
            "created_at": "2026-08-11T00:00:00+00:00",
            "tier": "working",
            "url": "/links/should-not-leak-stats",
        }
    )
    return response


def _mint_only_handler(status: int, payload: dict[str, Any]) -> Handler:
    def handler(request: httpx2.Request) -> httpx2.Response:
        if request.url.path == "/api/datasets":
            return httpx2.Response(status, json=payload)
        raise AssertionError(f"unexpected call to {request.url.path}")

    return handler


async def _call_get_data(
    handler: Handler,
    store: Any,
    arguments: dict[str, object],
    *,
    auth: AuthProvider | None = None,
    meta: dict[str, Any] | None = None,
) -> CallToolResult:
    async with _client(handler, auth=auth) as upstream:
        client_holder.set_client(upstream)
        async with Client(mcp) as client:
            # server.py's lifespan already built its own store on entering
            # `Client(mcp)`; this override lands before the tool call itself
            # runs, since `get_data` only reads the store via `get_store()`
            # once the request is dispatched below.
            set_store(store)
            return await client.call_tool("get_data", arguments, raise_on_error=False, meta=meta)


# Successful mint


async def test_successful_mint_returns_a_frame_reference_and_stripped_manifest() -> None:
    stored = _stored_definition()
    requests: list[dict[str, Any]] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        if request.url.path == "/api/datasets":
            requests.append(json.loads(request.content))
            return httpx2.Response(200, json=_manifest_response(spec_id=stored.spec_id))
        raise AssertionError(f"unexpected call to {request.url.path}")

    result = await _call_get_data(handler, _StubStore(stored), {"spec_id": stored.spec_id})

    assert result.is_error is False
    assert stored.spec_id in text_of(result)
    assert "cv_1" in text_of(result)


async def test_mint_request_body_matches_exactly() -> None:
    stored = _stored_definition()
    requests: list[dict[str, Any]] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        if request.url.path == "/api/datasets":
            requests.append(json.loads(request.content))
            return httpx2.Response(200, json=_manifest_response(spec_id=stored.spec_id))
        raise AssertionError(f"unexpected call to {request.url.path}")

    await _call_get_data(handler, _StubStore(stored), {"spec_id": stored.spec_id})

    identity = json.loads(stored.canonical_bytes)
    assert requests == [
        {
            "schema_version": 1,
            "input": {
                "kind": "spec_id",
                "spec_id": stored.spec_id,
                "definition": {
                    "schema_version": 1,
                    "spec_id": stored.spec_id,
                    "identity": identity,
                },
                "scope": {},
            },
            "options": {"format": "parquet", "bundle": False, "layout": "frame"},
        }
    ]


async def test_mint_request_body_matches_exactly_with_a_scope_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`test_mint_request_body_matches_exactly` only proves the whole-body shape on the
    no-scope path, where `options` is a hardcoded constant the test setup never touches.
    This proves the same on a scope-bearing request, where nothing about the body is fixed
    by test setup alone -- in particular that `options` still requests Parquet.
    """
    stored = _stored_definition(entities={"recipient": ["country/KEN"]})
    requests: list[dict[str, Any]] = []
    registered: list[dict[str, Any]] = []

    async def materialise(
        *, manifest: Manifest, options: Any, scope: Any, **_: Any
    ) -> MaterialisationOutcome:
        registered.append({"options": dict(options), "scope": dict(scope)})
        return MaterialisationOutcome(
            status="registered", attempts=0, frame_ref=mint_pair(manifest)
        )

    monkeypatch.setattr(get_data_tool, "materialise_and_register", materialise)

    def handler(request: httpx2.Request) -> httpx2.Response:
        if request.url.path == "/api/graph/resolve":
            return httpx2.Response(
                200,
                json={"results": [resolved_result("Uganda", "country/UGA")]},
            )
        if request.url.path == "/api/datasets":
            requests.append(json.loads(request.content))
            return httpx2.Response(
                200,
                json=_manifest_response(
                    spec_id=stored.spec_id, entities={"recipient": ["country/UGA"]}
                ),
            )
        raise AssertionError(f"unexpected call to {request.url.path}")

    await _call_get_data(
        handler,
        _StubStore(stored),
        {"spec_id": stored.spec_id, "scope": {"entities": {"recipient": ["Uganda"]}}},
    )

    # The frame's mint record is written from the options and scope the mint carried.
    assert registered == [
        {
            "options": {"format": "parquet", "bundle": False, "layout": "frame"},
            "scope": {"entities": {"recipient": ["country/UGA"]}},
        }
    ]
    identity = json.loads(stored.canonical_bytes)
    assert requests == [
        {
            "schema_version": 1,
            "input": {
                "kind": "spec_id",
                "spec_id": stored.spec_id,
                "definition": {
                    "schema_version": 1,
                    "spec_id": stored.spec_id,
                    "identity": identity,
                },
                "scope": {"entities": {"recipient": ["country/UGA"]}},
            },
            "options": {"format": "parquet", "bundle": False, "layout": "frame"},
        }
    ]


async def test_mint_call_uses_retry_false_and_a_125s_timeout() -> None:
    stored = _stored_definition()
    timeouts: list[float] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        if request.url.path == "/api/datasets":
            timeouts.append(request.extensions["timeout"]["read"])
            return httpx2.Response(200, json=_manifest_response(spec_id=stored.spec_id))
        raise AssertionError(f"unexpected call to {request.url.path}")

    await _call_get_data(handler, _StubStore(stored), {"spec_id": stored.spec_id})

    assert timeouts == [pytest.approx(125.0)]


@pytest.mark.parametrize(
    "arguments_extra",
    [
        {},
        {"scope": {"entities": {"recipient": ["Uganda"]}}},
        {"scope": {"window": {"kind": "range", "start_year": 2020, "end_year": 2024}}},
    ],
    ids=["no_scope", "entities_override", "window_override"],
)
async def test_the_definition_sidecar_is_always_sent(arguments_extra: dict[str, Any]) -> None:
    """On a store hit, the mint request always carries the `definition` sidecar.

    `mint_spec_id` only raises `spec_not_found` when its own registry misses AND the request
    carried no `definition` sidecar. A store hit always attaches the sidecar here, whatever
    the call's scope override looks like, so `spec_not_found` can never come back from
    `/api/datasets` on this path. A store miss is different: see
    `test_registry_fallback_serves_a_definition_store_never_held` and
    `test_registry_fallback_miss_still_refuses_definition_not_held` for the sidecar-free
    fallback path this tool takes then.
    """
    stored = _stored_definition(entities={"recipient": ["country/KEN"]})
    requests: list[dict[str, Any]] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        if request.url.path == "/api/graph/resolve":
            texts = json.loads(request.content)["texts"]
            return httpx2.Response(
                200,
                json={"results": [resolved_result(text, "country/UGA") for text in texts]},
            )
        if request.url.path == "/api/datasets":
            requests.append(json.loads(request.content))
            return httpx2.Response(200, json=_manifest_response(spec_id=stored.spec_id))
        raise AssertionError(f"unexpected call to {request.url.path}")

    result = await _call_get_data(
        handler, _StubStore(stored), {"spec_id": stored.spec_id, **arguments_extra}
    )

    assert result.is_error is False
    assert len(requests) == 1
    definition = requests[0]["input"]["definition"]
    assert definition == {
        "schema_version": 1,
        "spec_id": stored.spec_id,
        "identity": json.loads(stored.canonical_bytes),
    }


# A mint that crosses apis' own sync budget: 202, then polled to completion

# Shaped like apis' own `op_` + sha256-hex ids (`OperationEnvelope.operation_id`'s
# own pattern constraint) so these tests exercise the real validated shape rather than
# a placeholder a stricter pattern would itself reject.
_OPERATION_ID = "op_" + hashlib.sha256(b"test-operation").hexdigest()
_OPERATION_POLL_PATH = f"/api/datasets/operations/{_OPERATION_ID}"


async def test_202_operation_polled_to_done_returns_the_same_result_as_a_sync_mint() -> None:
    stored = _stored_definition()
    manifest = _manifest_response(spec_id=stored.spec_id)
    calls: list[str] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        calls.append(request.url.path)
        if request.url.path == "/api/datasets":
            return httpx2.Response(
                202,
                json={
                    "operation": {
                        "operation_id": _OPERATION_ID,
                        "status": "pending",
                        "poll": _OPERATION_POLL_PATH,
                        "retry_after_ms": 1,
                    }
                },
            )
        if request.url.path == _OPERATION_POLL_PATH:
            return httpx2.Response(
                200,
                json={
                    "operation": {"operation_id": _OPERATION_ID, "status": "done"},
                    "manifest": manifest,
                },
            )
        raise AssertionError(f"unexpected call to {request.url.path}")

    result = await _call_get_data(handler, _StubStore(stored), {"spec_id": stored.spec_id})

    assert result.is_error is False
    assert stored.spec_id in text_of(result)
    assert "cv_1" in text_of(result)
    assert calls == ["/api/datasets", _OPERATION_POLL_PATH]
    assert _OPERATION_ID in text_of(result)
    assert "queued operation" in text_of(result)


_BACKGROUND_META = {MINT_ADMISSION_META_KEY: "background"}
_HARNESS_CALLER = conversation_caller("principal/conversation")


@contextmanager
def _over_loopback(*, loopback: bool = True) -> Iterator[None]:
    """Make the in-memory client's calls read as arriving over the harness's
    loopback mount (`common.mcp_telemetry.is_loopback_request`), or as
    arriving off it. The in-memory transport has no HTTP request to mark."""

    def marked() -> bool:
        return loopback

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(mint_admission, "is_loopback_request", marked)
        yield


@pytest.mark.parametrize(
    ("meta", "loopback", "admissions", "background"),
    [
        (_BACKGROUND_META, True, ["background"], "queued"),
        ({MINT_ADMISSION_META_KEY: "foreground"}, True, [None, None], None),
        (None, True, [None, None], None),
        (_BACKGROUND_META, False, [None, None], None),
    ],
    ids=["background", "other_value", "unmarked", "off_loopback"],
)
async def test_a_background_call_marks_its_mint_post_and_ends_once_it_is_answered(
    meta: dict[str, Any] | None,
    loopback: bool,
    admissions: list[str | None],
    background: str | None,
    exported_spans: InMemorySpanExporter,
) -> None:
    """A harness call over loopback whose `_meta` asks for a background mint sends apis
    the admission header on its mint POST and ends on apis' 202 without polling, its span
    marked. Any other call, the marked call off loopback included, sends the header
    nowhere and polls the operation to its frame."""
    stored = _stored_definition()
    manifest = _manifest_response(spec_id=stored.spec_id)
    sent: list[str | None] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        sent.append(request.headers.get(MINT_ADMISSION_HEADER))
        if request.url.path == "/api/datasets":
            operation = {
                "operation_id": _OPERATION_ID,
                "status": "pending",
                "poll": _OPERATION_POLL_PATH,
                "retry_after_ms": 1,
            }
            return httpx2.Response(202, json={"operation": operation})
        if request.url.path == _OPERATION_POLL_PATH:
            return httpx2.Response(
                200,
                json={
                    "operation": {"operation_id": _OPERATION_ID, "status": "done"},
                    "manifest": manifest,
                },
            )
        raise AssertionError(f"unexpected call to {request.url.path}")

    with _over_loopback(loopback=loopback):
        result = await _call_get_data(
            handler, _StubStore(stored), {"spec_id": stored.spec_id}, meta=meta
        )

    assert result.is_error is False
    assert sent == admissions
    assert (result.structured_content or {}).get(BACKGROUND_OUTCOME_KEY) == background
    marked = [
        span
        for span in exported_spans.get_finished_spans()
        if (span.attributes or {}).get(BACKGROUND_MINT_SPAN_ATTRIBUTE) is True
    ]
    assert len(marked) == (1 if background else 0)


class _ApisStillMinting(httpx2.AsyncBaseTransport):
    """apis receiving the mint POST, as httpcore reports a written request body to
    the request's trace extension, then answering only once `answer` is set: 200,
    or `declined` apis' 503 decline of a background mint."""

    def __init__(self, spec_id: str, *, declined: bool = False) -> None:
        self.spec_id = spec_id
        self.declined = declined
        self.answer = asyncio.Event()
        self.posts = 0
        self.answered = False

    async def handle_async_request(self, request: httpx2.Request) -> httpx2.Response:
        assert request.url.path == "/api/datasets", request.url.path
        self.posts += 1
        trace = request.extensions.get("trace")
        if trace is not None:
            await trace("http11.send_request_body.complete", {"request": request})
        await self.answer.wait()
        self.answered = True
        if self.declined:
            return httpx2.Response(
                503,
                content=_problem_body(
                    code="service_unavailable",
                    retryable=True,
                    constraint_id="mint_background_declined",
                ),
                headers={"Retry-After": "1"},
            )
        return httpx2.Response(200, json=_manifest_response(spec_id=self.spec_id))


async def _background_call(apis: _ApisStillMinting, store: Any, *, answer: bool) -> CallToolResult:
    """One background `get_data` call from the harness against `apis`, answered by
    apis before the server shuts down when `answer`, and left unanswered at shutdown
    otherwise."""
    async with UpstreamClient(settings=_settings(), transport=apis) as upstream:
        client_holder.set_client(upstream)
        with supplied_caller(_HARNESS_CALLER), _over_loopback():
            async with Client(mcp) as client:
                set_store(store)
                result = await asyncio.wait_for(
                    client.call_tool(
                        "get_data",
                        {"spec_id": apis.spec_id},
                        raise_on_error=False,
                        meta=_BACKGROUND_META,
                    ),
                    timeout=5,
                )
                assert not apis.answered
                (posting,) = get_data_tool._background_mints
                assert not posting.done()
                if answer:
                    apis.answer.set()
                    await asyncio.wait_for(posting, timeout=5)
        if not answer:
            assert posting.cancelled()
    return result


async def test_a_background_call_answers_posted_once_its_post_is_out_and_keeps_it_open() -> None:
    """The call answers "posted" while apis is still minting; the POST stays open
    until apis answers it, sent once."""
    stored = _stored_definition()
    apis = _ApisStillMinting(stored.spec_id)

    result = await _background_call(apis, _StubStore(stored), answer=True)

    assert result.is_error is False
    assert (result.structured_content or {})[BACKGROUND_OUTCOME_KEY] == "posted"
    assert apis.answered
    assert apis.posts == 1
    assert not get_data_tool._background_mints


async def test_apis_declining_a_posted_background_mint_is_logged_as_declined(
    caplog: pytest.LogCaptureFixture,
) -> None:
    stored = _stored_definition()
    apis = _ApisStillMinting(stored.spec_id, declined=True)

    with caplog.at_level(logging.INFO, logger="data_mcp.get_data_tool"):
        result = await _background_call(apis, _StubStore(stored), answer=True)

    assert (result.structured_content or {})[BACKGROUND_OUTCOME_KEY] == "posted"
    (line,) = [
        r.getMessage() for r in caplog.records if "background mint answered" in r.getMessage()
    ]
    assert f"spec_id={stored.spec_id}" in line
    assert "caller=" in line
    assert "outcome=declined" in line
    assert "elapsed_s=" in line


async def test_a_background_mint_holds_no_progress_reporter_past_its_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The background mint's task outlives the request, so it is given no `on_poll`,
    which would hold the request's `Context`."""
    given: list[object] = []
    attempt_mint = get_data_tool._attempt_mint

    async def recording(**kwargs: Any) -> ToolResult:
        given.append(kwargs["on_poll"])
        return await attempt_mint(**kwargs)

    monkeypatch.setattr(get_data_tool, "_attempt_mint", recording)
    stored = _stored_definition()

    await _background_call(_ApisStillMinting(stored.spec_id), _StubStore(stored), answer=True)

    assert given == [None]


@pytest.mark.parametrize(
    ("structured", "word"),
    [
        ({"spec_id": "s", BACKGROUND_OUTCOME_KEY: "posted"}, "posted"),
        (
            {"code": "service_unavailable", "fix": {"constraint_id": "mint_background_declined"}},
            "declined",
        ),
        ({"code": "service_unavailable", "fix": {}}, "service_unavailable"),
        ({"spec_id": "s"}, "answered"),
        (None, "answered"),
    ],
    ids=["outcome", "declined", "refusal", "frame", "nothing"],
)
def test_a_background_calls_answer_reads_as_one_word(
    structured: dict[str, Any] | None, word: str
) -> None:
    assert background_outcome(structured) == word


async def test_a_foreground_mint_of_an_open_background_post_takes_its_answer() -> None:
    """apis would run the same mint twice, so the run's own call for the frame a
    background POST is still minting posts nothing and answers from that POST."""
    stored = _stored_definition()
    apis = _ApisStillMinting(stored.spec_id)

    async with UpstreamClient(settings=_settings(), transport=apis) as upstream:
        client_holder.set_client(upstream)
        with supplied_caller(_HARNESS_CALLER), _over_loopback():
            async with Client(mcp) as client:
                set_store(_StubStore(stored))
                arguments = {"spec_id": stored.spec_id}
                await client.call_tool(
                    "get_data", arguments, raise_on_error=False, meta=_BACKGROUND_META
                )
                foreground = asyncio.ensure_future(
                    client.call_tool("get_data", arguments, raise_on_error=False)
                )
                for _ in range(20):
                    await asyncio.sleep(0.01)
                assert apis.posts == 1
                assert not foreground.done()

                apis.answer.set()
                result = await asyncio.wait_for(foreground, timeout=5)

    assert result.is_error is False
    assert apis.posts == 1
    assert not get_data_tool._open_mint_posts


async def test_a_background_call_for_a_request_already_being_posted_answers_joined_unsent() -> None:
    """A background call for the frame a foreground POST is still minting sends
    nothing, since apis would run the same mint twice, and answers `joined` at once."""
    stored = _stored_definition()
    apis = _ApisStillMinting(stored.spec_id)

    async with UpstreamClient(settings=_settings(), transport=apis) as upstream:
        client_holder.set_client(upstream)
        with _over_loopback():
            async with Client(mcp) as client:
                set_store(_StubStore(stored))
                arguments = {"spec_id": stored.spec_id}
                foreground = asyncio.ensure_future(
                    client.call_tool("get_data", arguments, raise_on_error=False)
                )
                for _ in range(20):
                    await asyncio.sleep(0.01)
                assert apis.posts == 1

                joined = await asyncio.wait_for(
                    client.call_tool(
                        "get_data", arguments, raise_on_error=False, meta=_BACKGROUND_META
                    ),
                    timeout=5,
                )
                assert apis.posts == 1
                apis.answer.set()
                await asyncio.wait_for(foreground, timeout=5)

    assert (joined.structured_content or {})[BACKGROUND_OUTCOME_KEY] == "joined"
    assert apis.posts == 1
    assert not get_data_tool._open_mint_posts


class _ApisFirstPostOpen(httpx2.AsyncBaseTransport):
    """apis holding the first mint POST open until `answer` is set, then answering it
    `first`, and answering every later POST 200 after a short hold, long enough for
    every call waiting on it to wake before it ends. Each POST is reported written out
    to the request's trace extension on arrival, as httpcore reports it, except that a
    `first` exception fails the first POST's connection before its body is written."""

    def __init__(self, first: httpx2.Response | Exception) -> None:
        self.first = first
        self.answer = asyncio.Event()
        self.posts = 0

    async def handle_async_request(self, request: httpx2.Request) -> httpx2.Response:
        self.posts += 1
        first = self.posts == 1
        trace = request.extensions.get("trace")
        if trace is not None and not (first and isinstance(self.first, Exception)):
            await trace("http11.send_request_body.complete", {"request": request})
        if not first:
            await asyncio.sleep(0.05)
            return httpx2.Response(200, json={})
        await self.answer.wait()
        if isinstance(self.first, Exception):
            raise self.first
        return self.first


_REQUEST_BODY = {"input": {"spec_id": "spec"}}

# A deadline under which `DIRECT_CALL_BUDGET`, the default, sends a mint POST.
_DIRECT_DEADLINE_S = 120.0

# A budget whose deadline runs out long before apis answers the open POST.
_SHORT_BUDGET = CallBudget(
    deadline_s=None,
    pre_mint_reserve_s=0.2,
    mint_timeout_s=0.1,
    min_mint_timeout_s=0.1,
    post_mint_reserve_s=0.05,
)


async def _open_post(
    upstream: UpstreamClient, apis: _ApisFirstPostOpen, *, admission: Any = None
) -> asyncio.Task[httpx2.Response | ToolResult]:
    """A `_post_mint` of `_REQUEST_BODY` under the default budget, once apis has its POST."""
    task = asyncio.create_task(
        get_data_tool._post_mint(
            upstream,
            _REQUEST_BODY,
            spec_id="spec",
            settings=_settings(),
            deadline=CallDeadline.after(_DIRECT_DEADLINE_S),
            admission=admission,
        )
    )
    while apis.posts == 0 and not task.done():
        await asyncio.sleep(0.01)
    assert apis.posts == 1, await task
    return task


async def test_a_foreground_mint_out_of_time_on_an_open_post_refuses_without_posting() -> None:
    """A foreground call joined to an open POST of the same request waits up to its own
    deadline. When that runs out first it refuses `mint_unanswered`, retryable, and
    sends nothing, so apis never runs the mint twice. The open POST answers its caller."""
    apis = _ApisFirstPostOpen(httpx2.Response(200, json={}))

    async with UpstreamClient(settings=_settings(), transport=apis) as upstream:
        async with asyncio.timeout(5):
            open_post = await _open_post(upstream, apis)
            with use_call_budget(_SHORT_BUDGET):
                joined = await get_data_tool._post_mint(
                    upstream,
                    _REQUEST_BODY,
                    spec_id="spec",
                    settings=_settings(),
                    deadline=CallDeadline.after(0.3),
                )
            assert apis.posts == 1
            apis.answer.set()
            answered = await open_post

    assert isinstance(joined, ToolResult)
    refusal = joined.structured_content or {}
    assert refusal["code"] == "service_unavailable"
    assert refusal["retry_mode"] == "retry_after_delay"
    assert refusal["fix"]["spec_id"] == "spec"
    assert refusal["fix"]["waited_s"] >= 0.0
    assert "elapsed_s" not in refusal["fix"]
    assert isinstance(answered, httpx2.Response)
    assert apis.posts == 1
    assert not get_data_tool._open_mint_posts


async def test_a_foreground_mint_joined_to_an_open_post_raises_its_failure() -> None:
    """A failure of the open POST reaches the call that joined it, which posts nothing."""
    apis = _ApisFirstPostOpen(
        httpx2.Response(404, content=_problem_body(code="spec_not_found", retryable=False))
    )

    async with UpstreamClient(settings=_settings(), transport=apis) as upstream:
        async with asyncio.timeout(5):
            open_post = await _open_post(upstream, apis)
            joined = asyncio.ensure_future(
                get_data_tool._post_mint(
                    upstream,
                    _REQUEST_BODY,
                    spec_id="spec",
                    settings=_settings(),
                    deadline=CallDeadline.after(_DIRECT_DEADLINE_S),
                )
            )
            await asyncio.sleep(0.05)
            apis.answer.set()
            with pytest.raises(UpstreamError) as from_open:
                await open_post
            with pytest.raises(UpstreamError) as from_joined:
                await joined

    assert from_joined.value.code == from_open.value.code == "spec_not_found"
    assert apis.posts == 1
    assert not get_data_tool._open_mint_posts


async def test_a_foreground_mint_joined_to_a_cancelled_post_refuses_without_posting() -> None:
    """apis may still run a POST whose caller went away, so the call that joined it
    refuses `mint_unanswered`, retryable, and posts nothing."""
    apis = _ApisFirstPostOpen(httpx2.Response(200, json={}))

    async with UpstreamClient(settings=_settings(), transport=apis) as upstream:
        async with asyncio.timeout(5):
            open_post = await _open_post(upstream, apis)
            joined = asyncio.ensure_future(
                get_data_tool._post_mint(
                    upstream,
                    _REQUEST_BODY,
                    spec_id="spec",
                    settings=_settings(),
                    deadline=CallDeadline.after(_DIRECT_DEADLINE_S),
                )
            )
            await asyncio.sleep(0.05)
            open_post.cancel()
            refused = await joined

    assert isinstance(refused, ToolResult)
    refusal = refused.structured_content or {}
    assert refusal["code"] == "service_unavailable"
    assert refusal["retry_mode"] == "retry_after_delay"
    assert "waited_s" in refusal["fix"]
    assert apis.posts == 1
    assert not get_data_tool._open_mint_posts


@pytest.mark.parametrize("joiners", [1, 3])
async def test_foreground_mints_joined_to_a_background_post_never_written_out_post_once(
    joiners: int,
) -> None:
    """A background POST whose connection failed before its body went out started
    nothing in apis. The foreground calls that joined it wake together, and exactly one
    posts its own request: the others join that POST and take apis' answer to it."""
    apis = _ApisFirstPostOpen(httpx2.ConnectError("refused"))

    async with UpstreamClient(settings=_settings(), transport=apis) as upstream:
        async with asyncio.timeout(5):
            open_post = await _open_post(upstream, apis, admission="background")
            foreground = [
                asyncio.ensure_future(
                    get_data_tool._post_mint(
                        upstream,
                        _REQUEST_BODY,
                        spec_id="spec",
                        settings=_settings(),
                        deadline=CallDeadline.after(_DIRECT_DEADLINE_S),
                    )
                )
                for _ in range(joiners)
            ]
            await asyncio.sleep(0.05)
            assert apis.posts == 1
            apis.answer.set()
            unsent = await open_post
            posted = await asyncio.gather(*foreground)

    assert isinstance(unsent, ToolResult)
    assert (unsent.structured_content or {})["code"] == "service_unavailable"
    assert len(posted) == joiners
    assert all(isinstance(each, httpx2.Response) and each.status_code == 200 for each in posted)
    assert apis.posts == 2
    assert not get_data_tool._open_mint_posts


async def test_a_foreground_mint_joined_to_a_declined_background_post_posts_its_own() -> None:
    """apis ran nothing for a background POST it declined, so the foreground call that
    joined it posts its own request once that POST has ended."""
    apis = _ApisFirstPostOpen(
        httpx2.Response(
            503,
            content=_problem_body(
                code="service_unavailable",
                retryable=True,
                constraint_id="mint_background_declined",
            ),
            headers={"Retry-After": "1"},
        )
    )

    async with UpstreamClient(settings=_settings(), transport=apis) as upstream:
        async with asyncio.timeout(5):
            open_post = await _open_post(upstream, apis, admission="background")
            joined = asyncio.ensure_future(
                get_data_tool._post_mint(
                    upstream,
                    _REQUEST_BODY,
                    spec_id="spec",
                    settings=_settings(),
                    deadline=CallDeadline.after(_DIRECT_DEADLINE_S),
                )
            )
            await asyncio.sleep(0.05)
            assert apis.posts == 1
            apis.answer.set()
            with pytest.raises(UpstreamError):
                await open_post
            posted = await joined

    assert isinstance(posted, httpx2.Response)
    assert posted.status_code == 200
    assert apis.posts == 2
    assert not get_data_tool._open_mint_posts


async def test_shutdown_closes_a_background_post_apis_has_not_answered() -> None:
    stored = _stored_definition()
    apis = _ApisStillMinting(stored.spec_id)

    await _background_call(apis, _StubStore(stored), answer=False)

    assert not apis.answered
    assert not get_data_tool._background_mints


async def test_get_data_reports_progress_during_a_polled_mint() -> None:
    """A real FastMCP in-memory client wired with a `progress_handler` sees at least
    one progress notification while `get_data` polls a queued operation to done --
    proof the callback reaches `Context.report_progress` end to end, not just that
    `resolve_operation_response`'s own `on_poll` fires."""
    stored = _stored_definition()
    manifest = _manifest_response(spec_id=stored.spec_id)

    def handler(request: httpx2.Request) -> httpx2.Response:
        if request.url.path == "/api/datasets":
            return httpx2.Response(
                202,
                json={
                    "operation": {
                        "operation_id": _OPERATION_ID,
                        "status": "pending",
                        "poll": _OPERATION_POLL_PATH,
                        "retry_after_ms": 1,
                    }
                },
            )
        if request.url.path == _OPERATION_POLL_PATH:
            return httpx2.Response(
                200,
                json={
                    "operation": {"operation_id": _OPERATION_ID, "status": "done"},
                    "manifest": manifest,
                },
            )
        raise AssertionError(f"unexpected call to {request.url.path}")

    progress_reports: list[tuple[float, float | None, str | None]] = []

    async def progress_handler(progress: float, total: float | None, message: str | None) -> None:
        progress_reports.append((progress, total, message))

    async with _client(handler) as upstream:
        client_holder.set_client(upstream)
        async with Client(mcp) as client:
            set_store(_StubStore(stored))
            result = await client.call_tool(
                "get_data",
                {"spec_id": stored.spec_id},
                raise_on_error=False,
                progress_handler=progress_handler,
            )

    assert result.is_error is False
    assert progress_reports, "expected at least one progress notification during the polled mint"
    assert all(total is None for _, total, _ in progress_reports)
    messages = [message for _, _, message in progress_reports]
    assert messages[0] == "Preparing the dataset (queued, 0s)"
    assert all(
        message is not None and message.startswith("Preparing the dataset (")
        for message in messages
    )


async def test_202_operation_that_fails_refuses_with_the_upstream_code() -> None:
    stored = _stored_definition()

    def handler(request: httpx2.Request) -> httpx2.Response:
        if request.url.path == "/api/datasets":
            return httpx2.Response(
                202,
                json={
                    "operation": {
                        "operation_id": _OPERATION_ID,
                        "status": "pending",
                        "poll": _OPERATION_POLL_PATH,
                        "retry_after_ms": 1,
                    }
                },
            )
        if request.url.path == _OPERATION_POLL_PATH:
            return httpx2.Response(
                500,
                content=_problem_body(
                    code="operation_failed", retryable=False, detail="operation did not complete"
                ),
            )
        raise AssertionError(f"unexpected call to {request.url.path}")

    result = await _call_get_data(handler, _StubStore(stored), {"spec_id": stored.spec_id})

    assert result.is_error is True
    assert result.structured_content is not None
    assert result.structured_content["code"] == "upstream_error"
    assert result.structured_content["upstream_code"] == "operation_failed"


async def test_202_operation_not_found_on_poll_refuses_with_the_upstream_code() -> None:
    """The id this call itself computed and just submitted comes back unknown to the
    poll route -- distinct from `operation_failed` (the operation ran and did not
    complete): the operation store itself has no record of it at all.
    """
    stored = _stored_definition()

    def handler(request: httpx2.Request) -> httpx2.Response:
        if request.url.path == "/api/datasets":
            return httpx2.Response(
                202,
                json={
                    "operation": {
                        "operation_id": _OPERATION_ID,
                        "status": "pending",
                        "poll": _OPERATION_POLL_PATH,
                        "retry_after_ms": 1,
                    }
                },
            )
        if request.url.path == _OPERATION_POLL_PATH:
            return httpx2.Response(
                404,
                content=_problem_body(
                    code="operation_not_found", retryable=False, detail="no such operation"
                ),
            )
        raise AssertionError(f"unexpected call to {request.url.path}")

    result = await _call_get_data(handler, _StubStore(stored), {"spec_id": stored.spec_id})

    assert result.is_error is True
    assert result.structured_content is not None
    assert result.structured_content["code"] == "upstream_error"
    assert result.structured_content["upstream_code"] == "operation_not_found"


async def test_202_operation_still_running_at_the_poll_reserve_refuses_retryably(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A mint POST that returns 202 with only the poll reserve of the call deadline
    left refuses on the first check, without polling: deterministic, and the same
    refusal a real, much longer-running operation eventually hits.
    """
    clock = _FakeClock()
    monkeypatch.setattr(get_data_tool, "_deadline_clock", clock)
    stored = _stored_definition()

    def handler(request: httpx2.Request) -> httpx2.Response:
        if request.url.path == "/api/datasets":
            clock.now += 165.0 - 30.0
            return httpx2.Response(
                202,
                json={
                    "operation": {
                        "operation_id": _OPERATION_ID,
                        "status": "pending",
                        "poll": _OPERATION_POLL_PATH,
                        "retry_after_ms": 1500,
                    }
                },
            )
        raise AssertionError(f"unexpected call to {request.url.path}")

    result = await _call_get_data(handler, _StubStore(stored), {"spec_id": stored.spec_id})

    assert result.is_error is True
    assert result.structured_content is not None
    assert result.structured_content["code"] == "operation_still_processing"
    assert result.structured_content["retry_mode"] == "retry_after_delay"
    assert result.structured_content["fix"]["operation_id"] == _OPERATION_ID


async def test_a_202_hands_the_poll_the_calls_own_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The poll reads the same deadline the mint POST ran under, so a POST that ran
    long leaves the poll correspondingly less, never a fresh budget."""
    captured: dict[str, object] = {}

    async def _fake_resolve_operation_response(
        *,
        client: object,
        submit_payload: object,
        deadline: CallDeadline,
        poll_interval_s: float,
        on_poll: object = None,
        raise_upstream_error: bool = False,
    ) -> object:
        captured["deadline"] = deadline
        return ResolvedOperation(
            manifest_payload={"content_version": "cv_1", "definition": {"spec_id": "spec_x"}},
            operation_id=_OPERATION_ID,
            poll_attempts=0,
            elapsed_s=0.0,
        )

    monkeypatch.setattr(
        get_data_tool, "resolve_operation_response", _fake_resolve_operation_response
    )
    response = httpx2.Response(
        202,
        json={"operation": {"operation_id": _OPERATION_ID, "status": "pending"}},
    )
    deadline = CallDeadline.after(40.0)

    result = await get_data_tool._mint_payload_or_refusal(
        response, client=object(), settings=Settings(), deadline=deadline
    )

    assert not isinstance(result, ToolResult)
    assert captured["deadline"] is deadline


# Definition store refusals


async def test_store_miss_refuses_definition_not_held() -> None:
    """A store miss attempts a sidecar-free registry fallback (see the tests in the
    "Registry fallback on a store miss" section below) before refusing. This pins the
    end-to-end outcome when the registry misses too: `definition_not_held`, not
    `upstream_error`, reached only after that one upstream call.
    """
    spec_id = "spec_" + "a" * 32

    def handler(request: httpx2.Request) -> httpx2.Response:
        if request.url.path == "/api/datasets":
            return httpx2.Response(
                404, content=_problem_body(code="spec_not_found", retryable=False)
            )
        raise AssertionError(f"unexpected call to {request.url.path}")

    result = await _call_get_data(handler, _StubStore(None), {"spec_id": spec_id})

    assert result.is_error is True
    assert result.structured_content is not None
    assert result.structured_content["code"] == "definition_not_held"


async def test_unreachable_store_refuses_service_unavailable_distinct_from_not_held() -> None:
    def handler(_request: httpx2.Request) -> httpx2.Response:
        raise AssertionError("no upstream call expected when the store is unreachable")

    result = await _call_get_data(handler, _FailingGetStore(), {"spec_id": "spec_" + "a" * 32})

    assert result.is_error is True
    assert result.structured_content is not None
    assert result.structured_content["code"] == "service_unavailable"
    assert result.structured_content["code"] != "definition_not_held"
    assert result.structured_content["retry_mode"] == "retry_after_delay"


async def test_identity_mismatch_refuses_capability_unavailable() -> None:
    def handler(_request: httpx2.Request) -> httpx2.Response:
        raise AssertionError("no upstream call expected on an identity mismatch")

    result = await _call_get_data(handler, _MismatchStore(), {"spec_id": "spec_" + "a" * 32})

    assert result.is_error is True
    assert result.structured_content is not None
    assert result.structured_content["code"] == "capability_unavailable"
    assert result.structured_content["retry_mode"] == "do_not_retry"


# Registry fallback on a store miss


async def test_registry_fallback_serves_a_definition_store_never_held() -> None:
    """A `spec_id` this process's store never saw can still be served straight from
    apis' own permanent registry: the platform contract's `definition_not_held` means
    "held by neither store nor registry", not "held by neither store nor a sidecar this
    call carried" -- so a store miss alone must not refuse before trying the registry.
    """
    spec_id = "spec_" + "c" * 32
    requests: list[dict[str, Any]] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        if request.url.path == "/api/datasets":
            requests.append(json.loads(request.content))
            return httpx2.Response(200, json=_manifest_response(spec_id=spec_id))
        raise AssertionError(f"unexpected call to {request.url.path}")

    result = await _call_get_data(handler, _StubStore(None), {"spec_id": spec_id})

    assert result.is_error is False
    assert len(requests) == 1
    # No `definition` sidecar -- that's what lets `mint_spec_id` serve this id straight
    # from its own registry rather than raising `spec_not_found` for a request it can't
    # trust to reconstruct the identity itself.
    assert "definition" not in requests[0]["input"]


async def test_registry_fallback_miss_still_refuses_definition_not_held() -> None:
    """A store miss followed by a registry miss (`spec_not_found`) is the case the
    platform contract's `definition_not_held` actually names -- not `upstream_error`,
    which is where every other upstream failure on this call still routes.
    """
    spec_id = "spec_" + "d" * 32

    def handler(request: httpx2.Request) -> httpx2.Response:
        if request.url.path == "/api/datasets":
            return httpx2.Response(
                404, content=_problem_body(code="spec_not_found", retryable=False)
            )
        raise AssertionError(f"unexpected call to {request.url.path}")

    result = await _call_get_data(handler, _StubStore(None), {"spec_id": spec_id})

    assert result.is_error is True
    assert result.structured_content is not None
    assert result.structured_content["code"] == "definition_not_held"
    assert result.structured_content["code"] != "upstream_error"


async def test_registry_fallback_reports_no_dropped_roles() -> None:
    """The registry-fallback path has no base identity to diff a scope override against
    (see `get_data`'s own comment), so it never claims a role was dropped -- even when the
    served manifest's own identity would, under the normal stored-definition path, have
    shown one.
    """
    spec_id = "spec_" + "e" * 32

    def handler(request: httpx2.Request) -> httpx2.Response:
        if request.url.path == "/api/graph/resolve":
            return httpx2.Response(
                200,
                json={"results": [resolved_result("Togo", "country/TGO")]},
            )
        if request.url.path == "/api/datasets":
            return httpx2.Response(
                200,
                json=_manifest_response(
                    spec_id=spec_id,
                    entities={"provider": ["country/FRA"], "recipient": ["country/TGO"]},
                ),
            )
        raise AssertionError(f"unexpected call to {request.url.path}")

    result = await _call_get_data(
        handler,
        _StubStore(None),
        {"spec_id": spec_id, "scope": {"entities": {"recipient": ["Togo"]}}},
    )

    assert result.is_error is False
    assert "no longer bound" not in text_of(result)


# Mint upstream failure


async def test_mint_transport_failure_becomes_a_refusal() -> None:
    stored = _stored_definition()

    def handler(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ConnectError("boom", request=request)

    result = await _call_get_data(handler, _StubStore(stored), {"spec_id": stored.spec_id})

    assert result.is_error is True
    assert result.structured_content is not None
    assert result.structured_content["retry_mode"] == "retry_after_delay"


async def test_upstream_invalid_role_ref_pointer_matches_the_local_entities_convention() -> None:
    """A locally-rejected name and an upstream-rejected role both point at
    `/scope/entities/{role}` -- this tool's own argument path -- rather than
    leaking apis' `/input`-wrapped wire envelope. The rendered message agrees
    with the structured `fix.pointer`: `request_invalid`'s factory embeds the
    pointer inline in `message`, so both come from the same rebase call rather
    than naming two different paths for the same mistake.
    """
    stored = _stored_definition()

    def handler(request: httpx2.Request) -> httpx2.Response:
        if request.url.path == "/api/datasets":
            return httpx2.Response(
                422,
                content=_problem_body(
                    code="invalid_role_ref",
                    retryable=False,
                    issues=[
                        {
                            "pointer": "/input/scope/entities/recipient",
                            "member": "recipient",
                            "axis": "recipient",
                        }
                    ],
                ),
            )
        raise AssertionError(f"unexpected call to {request.url.path}")

    result = await _call_get_data(handler, _StubStore(stored), {"spec_id": stored.spec_id})

    assert result.is_error is True
    assert result.structured_content is not None
    assert result.structured_content["code"] == "request_invalid"
    assert result.structured_content["fix"]["problems"][0].startswith("scope.entities.recipient: ")
    assert "/scope/entities/recipient" in text_of(result)
    assert "/input/scope/entities/" not in text_of(result)


_UNIT_ISSUE_DETAIL = (
    "facet axis 'unit' is unbound, no default policy is declared, and no "
    "probed candidate has any observations"
)
# The platform sends each issue's own code and detail.
_FACET_REFUSAL: dict[str, Any] = {
    "retryable": False,
    "detail": "one or more facet axes could not be resolved",
    "issues": [
        {
            "pointer": "/facets/unit",
            "axis": "unit",
            "code": "default_facet_undefined",
            "detail": _UNIT_ISSUE_DETAIL,
        }
    ],
}


@pytest.mark.parametrize("path", ["sync", "polled"])
async def test_a_refused_mint_names_the_spec_and_places_it_was_minting(path: str) -> None:
    """`default_facet_undefined` relays apis' detail and retryable flag as sent, and
    adds only what get_data was asking for: the spec_id and the places it bound. The same
    holds when the mint ran as a 202 operation and its poll reports the refusal."""
    stored = _stored_definition(entities={"observationAbout": ["africa"]})

    def handler(request: httpx2.Request) -> httpx2.Response:
        if request.url.path == "/api/datasets" and path == "sync":
            return httpx2.Response(
                422, content=_problem_body(code="default_facet_undefined", **_FACET_REFUSAL)
            )
        if request.url.path == "/api/datasets":
            return httpx2.Response(
                202,
                json={
                    "operation": {
                        "operation_id": _OPERATION_ID,
                        "status": "pending",
                        "poll": _OPERATION_POLL_PATH,
                        "retry_after_ms": 1,
                    }
                },
            )
        if request.url.path == _OPERATION_POLL_PATH:
            return httpx2.Response(
                500,
                content=_problem_body(
                    code="operation_failed",
                    cause={"code": "default_facet_undefined"},
                    **_FACET_REFUSAL,
                ),
            )
        raise AssertionError(f"unexpected call to {request.url.path}")

    result = await _call_get_data(handler, _StubStore(stored), {"spec_id": stored.spec_id})

    assert result.is_error is True
    assert result.structured_content is not None
    assert result.structured_content["code"] == "default_facet_undefined"
    assert result.structured_content["retry_mode"] == "do_not_retry"
    assert result.structured_content["fix"] == {
        "issues": [
            {
                "axis": "unit",
                "alternatives": [],
                "truncated": False,
                "code": "default_facet_undefined",
                "detail": _UNIT_ISSUE_DETAIL,
            }
        ],
        "detail": "one or more facet axes could not be resolved",
        "retryable": False,
        "spec_id": stored.spec_id,
        "entities": {"observationAbout": ["africa"]},
    }
    assert result.structured_content["message"] == (
        "The platform refused the requested data for 1 place."
    )


# Resolve upstream failure -- `resolve_entities` deliberately lets `UpstreamError`
# propagate uncaught (see its own docstring); these prove `get_data` itself catches it
# rather than letting it escape as a bare exception with none of this module's refusal
# fields.


async def test_resolve_service_unavailable_refuses_rather_than_raising() -> None:
    stored = _stored_definition()

    def handler(request: httpx2.Request) -> httpx2.Response:
        if request.url.path == "/api/graph/resolve":
            return httpx2.Response(
                503,
                content=_problem_body(
                    code="service_unavailable", retryable=True, detail="graph is degraded"
                ),
            )
        raise AssertionError("mint should not be reached when resolve fails upstream")

    result = await _call_get_data(
        handler,
        _StubStore(stored),
        {"spec_id": stored.spec_id, "scope": {"entities": {"recipient": ["Uganda"]}}},
    )

    assert result.is_error is True
    assert result.structured_content is not None
    assert result.structured_content["code"] == "service_unavailable"


async def test_resolve_unrecognised_status_refuses_rather_than_raising() -> None:
    stored = _stored_definition()

    def handler(request: httpx2.Request) -> httpx2.Response:
        if request.url.path == "/api/graph/resolve":
            return httpx2.Response(200, json={"results": [{"status": "something_new"}]})
        raise AssertionError(
            "mint should not be reached when resolve returns an unrecognised status"
        )

    result = await _call_get_data(
        handler,
        _StubStore(stored),
        {"spec_id": stored.spec_id, "scope": {"entities": {"recipient": ["Uganda"]}}},
    )

    assert result.is_error is True
    assert result.structured_content is not None
    assert result.structured_content["code"] == "upstream_error"
    assert result.structured_content["retry_mode"] == "do_not_retry"


async def test_resolve_response_missing_results_key_refuses_rather_than_raising() -> None:
    """A 2xx body missing `results` fails `ResolveResponse` validation, which the resolve
    guard must catch rather than let escape as a raw exception with none of this module's
    refusal fields.
    """
    stored = _stored_definition()

    def handler(request: httpx2.Request) -> httpx2.Response:
        if request.url.path == "/api/graph/resolve":
            return httpx2.Response(200, json={})
        raise AssertionError("mint should not be reached when resolve's body is unusable")

    result = await _call_get_data(
        handler,
        _StubStore(stored),
        {"spec_id": stored.spec_id, "scope": {"entities": {"recipient": ["Uganda"]}}},
    )

    assert result.is_error is True
    assert result.structured_content is not None
    assert result.structured_content["code"] == "upstream_error"
    assert result.structured_content["retry_mode"] == "do_not_retry"


async def test_resolve_response_wrongly_typed_nested_value_refuses_rather_than_raising() -> None:
    """A 2xx body whose `entity` is a string rather than a mapping fails `ResolveResponse`
    validation, which the resolve guard must also catch.
    """
    stored = _stored_definition()

    def handler(request: httpx2.Request) -> httpx2.Response:
        if request.url.path == "/api/graph/resolve":
            return httpx2.Response(
                200, json={"results": [{"status": "resolved", "entity": "not-a-mapping"}]}
            )
        raise AssertionError("mint should not be reached when resolve's body is unusable")

    result = await _call_get_data(
        handler,
        _StubStore(stored),
        {"spec_id": stored.spec_id, "scope": {"entities": {"recipient": ["Uganda"]}}},
    )

    assert result.is_error is True
    assert result.structured_content is not None
    assert result.structured_content["code"] == "upstream_error"
    assert result.structured_content["retry_mode"] == "do_not_retry"


# Entity resolution outcomes, each with role present


async def test_ambiguous_entity_refuses_with_the_role_present() -> None:
    stored = _stored_definition()

    def handler(request: httpx2.Request) -> httpx2.Response:
        if request.url.path == "/api/graph/resolve":
            return httpx2.Response(
                200,
                json={
                    "results": [
                        resolve_result(
                            "Georgia",
                            "ambiguous",
                            candidates=[
                                {"ref": "geoId/13", "label": "Georgia", "kind": "place_other"}
                            ],
                        )
                    ]
                },
            )
        raise AssertionError("mint should not be reached on an ambiguous entity")

    result = await _call_get_data(
        handler,
        _StubStore(stored),
        {"spec_id": stored.spec_id, "scope": {"entities": {"recipient": ["Georgia"]}}},
    )

    assert result.is_error is True
    assert result.structured_content is not None
    assert result.structured_content["code"] == "entity_ambiguous"
    assert result.structured_content["fix"]["role"] == "recipient"


async def test_ambiguous_entity_candidates_each_carry_their_own_type() -> None:
    """A candidate list of bare dcids -- no `type` per entry -- fails this test: the model
    can't tell a US state (`geoId/13`) from the country (`country/GEO`) by dcid alone, so
    `entity_ambiguous` names every candidate's type, not just its dcid and label.
    """
    stored = _stored_definition()

    def handler(request: httpx2.Request) -> httpx2.Response:
        if request.url.path == "/api/graph/resolve":
            return httpx2.Response(
                200,
                json={
                    "results": [
                        resolve_result(
                            "Georgia",
                            "ambiguous",
                            candidates=[
                                {"ref": "geoId/13", "label": "Georgia", "kind": "place_other"},
                                {"ref": "country/GEO", "label": "Georgia", "kind": "country"},
                            ],
                        )
                    ]
                },
            )
        raise AssertionError("mint should not be reached on an ambiguous entity")

    result = await _call_get_data(
        handler,
        _StubStore(stored),
        {"spec_id": stored.spec_id, "scope": {"entities": {"recipient": ["Georgia"]}}},
    )

    assert result.is_error is True
    assert result.structured_content is not None
    candidates = result.structured_content["fix"]["candidates"]
    assert {c["dcid"]: c["type"] for c in candidates} == {
        "geoId/13": "place_other",
        "country/GEO": "country",
    }


async def test_not_found_entity_refuses_with_the_role_present() -> None:
    stored = _stored_definition()

    def handler(request: httpx2.Request) -> httpx2.Response:
        if request.url.path == "/api/graph/resolve":
            return httpx2.Response(
                200, json={"results": [resolve_result("Nowhereland", "not_found", near_matches=[])]}
            )
        raise AssertionError("mint should not be reached on a not_found entity")

    result = await _call_get_data(
        handler,
        _StubStore(stored),
        {"spec_id": stored.spec_id, "scope": {"entities": {"recipient": ["Nowhereland"]}}},
    )

    assert result.is_error is True
    assert result.structured_content is not None
    assert result.structured_content["code"] == "entity_not_found"
    assert result.structured_content["fix"]["role"] == "recipient"
    assert result.structured_content["fix"]["had_near_matches"] is False


async def test_not_held_entity_refuses_entity_not_found_with_a_distinct_detail() -> None:
    stored = _stored_definition()

    def handler(request: httpx2.Request) -> httpx2.Response:
        if request.url.path == "/api/graph/resolve":
            return httpx2.Response(
                200,
                json={
                    "results": [
                        resolve_result(
                            "Andorra",
                            "not_held",
                            entity={"ref": "country/AND", "label": "Andorra", "kind": None},
                        )
                    ]
                },
            )
        raise AssertionError("mint should not be reached on a not_held entity")

    result = await _call_get_data(
        handler,
        _StubStore(stored),
        {"spec_id": stored.spec_id, "scope": {"entities": {"recipient": ["Andorra"]}}},
    )

    assert result.is_error is True
    assert result.structured_content is not None
    assert result.structured_content["code"] == "entity_not_found"
    assert result.structured_content["fix"]["role"] == "recipient"
    # Distinct from a genuine miss: `near_matches` reads the same as a
    # not_found refusal, but `not_held` is the machine-readable
    # discriminator. The door names no kind for it, so none is stated.
    assert result.structured_content["fix"]["not_held"] is True
    assert "entity_type" not in result.structured_content["fix"]


async def test_an_organisation_resolves_into_scope() -> None:
    stored = _stored_definition()
    requests: list[dict[str, Any]] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        if request.url.path == "/api/graph/resolve":
            return httpx2.Response(
                200,
                json={"results": [resolved_result("World Bank", "WorldBank", "organisation")]},
            )
        if request.url.path == "/api/datasets":
            requests.append(json.loads(request.content))
            return httpx2.Response(
                200,
                json=_manifest_response(spec_id=stored.spec_id, entities={"donor": ["WorldBank"]}),
            )
        raise AssertionError(f"unexpected call to {request.url.path}")

    result = await _call_get_data(
        handler,
        _StubStore(stored),
        {"spec_id": stored.spec_id, "scope": {"entities": {"donor": ["World Bank"]}}},
    )

    assert result.is_error is False
    assert requests[0]["input"]["scope"]["entities"] == {"donor": ["WorldBank"]}


async def test_invalid_name_refuses_request_invalid_with_the_role_in_the_pointer() -> None:
    stored = _stored_definition()

    def handler(_request: httpx2.Request) -> httpx2.Response:
        raise AssertionError("a blank name needs no upstream call at all")

    result = await _call_get_data(
        handler,
        _StubStore(stored),
        {"spec_id": stored.spec_id, "scope": {"entities": {"recipient": ["   "]}}},
    )

    assert result.is_error is True
    assert result.structured_content is not None
    assert result.structured_content["code"] == "request_invalid"
    assert result.structured_content["fix"]["problems"][0].startswith("scope.entities.recipient: ")


# Partial region membership


async def test_region_expansion_reaches_the_mint_call_as_member_dcids_only() -> None:
    """Nothing else asserts what `input.scope.entities` carries on the wire for a resolved
    region: this proves member dcids reach the mint call, and neither the region's own name
    nor any m49/wikidataId container id ever does.
    """
    stored = _stored_definition(entities={"recipient": ["country/KEN"]})
    requests: list[dict[str, Any]] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        if request.url.path == "/api/graph/resolve":
            return httpx2.Response(
                200,
                json={
                    "results": [
                        members_result(
                            "Sub-Saharan Africa",
                            "undata-geo/G00134000",
                            "subregion",
                            [("country/KEN", "Kenya"), ("country/NGA", "Nigeria")],
                        )
                    ]
                },
            )
        if request.url.path == "/api/datasets":
            requests.append(json.loads(request.content))
            return httpx2.Response(
                200,
                json=_manifest_response(
                    spec_id=stored.spec_id, entities={"recipient": ["country/KEN", "country/NGA"]}
                ),
            )
        raise AssertionError(f"unexpected call to {request.url.path}")

    result = await _call_get_data(
        handler,
        _StubStore(stored),
        {"spec_id": stored.spec_id, "scope": {"entities": {"recipient": ["Sub-Saharan Africa"]}}},
    )

    assert result.is_error is False
    wire_entities = requests[0]["input"]["scope"]["entities"]
    assert wire_entities == {"recipient": ["country/KEN", "country/NGA"]}
    wire_text = json.dumps(wire_entities)
    assert "Sub-Saharan Africa" not in wire_text
    assert "m49/" not in wire_text


# dcid passthrough: how a caller answers this tool's own `entity_ambiguous` refusal


@pytest.mark.parametrize(
    ("entity", "dcid"),
    [
        ("dcid:country/GEO", "country/GEO"),
        ("dcid:geoId/13", "geoId/13"),
        # An unmarked namespaced dcid, as digests and recall entries show one.
        ("country/KEN", "country/KEN"),
        (" geoId/13 ", "geoId/13"),
        ("dcid:wikidataId/Q30", "wikidataId/Q30"),
        # Bare dcids carry no `/`: a group, a region and an organisation.
        ("dcid:GroupOfSeven", "GroupOfSeven"),
        ("dcid:africa", "africa"),
        ("dcid:WorldBank", "WorldBank"),
        (" dcid: undata-geo/G00406000 ", "undata-geo/G00406000"),
    ],
)
async def test_a_dcid_skips_resolution_and_reaches_mint_bare(entity: str, dcid: str) -> None:
    stored = _stored_definition()
    requests: list[dict[str, Any]] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        if request.url.path == "/api/graph/resolve":
            raise AssertionError("a dcid needs no resolve call")
        if request.url.path == "/api/datasets":
            requests.append(json.loads(request.content))
            return httpx2.Response(
                200,
                json=_manifest_response(spec_id=stored.spec_id, entities={"recipient": [dcid]}),
            )
        raise AssertionError(f"unexpected call to {request.url.path}")

    result = await _call_get_data(
        handler,
        _StubStore(stored),
        {"spec_id": stored.spec_id, "scope": {"entities": {"recipient": [entity]}}},
    )

    assert result.is_error is False
    assert requests[0]["input"]["scope"]["entities"] == {"recipient": [dcid]}


async def test_marked_dcid_mixes_with_a_resolved_name_in_the_same_role() -> None:
    stored = _stored_definition()
    resolve_texts: list[str] = []
    requests: list[dict[str, Any]] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        if request.url.path == "/api/graph/resolve":
            texts = json.loads(request.content)["texts"]
            resolve_texts.extend(texts)
            return httpx2.Response(
                200,
                json={"results": [resolved_result("Togo", "country/TGO") for _ in texts]},
            )
        if request.url.path == "/api/datasets":
            requests.append(json.loads(request.content))
            return httpx2.Response(
                200,
                json=_manifest_response(
                    spec_id=stored.spec_id, entities={"recipient": ["GroupOfSeven", "country/TGO"]}
                ),
            )
        raise AssertionError(f"unexpected call to {request.url.path}")

    result = await _call_get_data(
        handler,
        _StubStore(stored),
        {
            "spec_id": stored.spec_id,
            "scope": {"entities": {"recipient": ["dcid:GroupOfSeven", "Togo"]}},
        },
    )

    assert result.is_error is False
    # Only the name reaches the resolve door -- the dcid never does.
    assert resolve_texts == ["Togo"]
    assert requests[0]["input"]["scope"]["entities"]["recipient"] == ["GroupOfSeven", "country/TGO"]


async def test_an_ambiguous_candidate_passed_back_as_offered_reaches_mint_as_its_dcid() -> None:
    """The round trip the marker exists for: the refusal's `entity` for a bare-dcid candidate,
    passed back unchanged, mints that dcid and is never resolved again."""
    stored = _stored_definition()
    resolve_texts: list[str] = []
    requests: list[dict[str, Any]] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        if request.url.path == "/api/graph/resolve":
            texts = json.loads(request.content)["texts"]
            resolve_texts.extend(texts)
            candidates = [
                {"ref": "GroupOfSeven", "label": "Group of Seven", "kind": "group"},
                {"ref": "country/GMB", "label": "Gambia", "kind": "country"},
            ]
            return httpx2.Response(
                200, json={"results": [resolve_result("G", "ambiguous", candidates=candidates)]}
            )
        if request.url.path == "/api/datasets":
            requests.append(json.loads(request.content))
            return httpx2.Response(
                200,
                json=_manifest_response(
                    spec_id=stored.spec_id, entities={"recipient": ["GroupOfSeven"]}
                ),
            )
        raise AssertionError(f"unexpected call to {request.url.path}")

    refused = await _call_get_data(
        handler,
        _StubStore(stored),
        {"spec_id": stored.spec_id, "scope": {"entities": {"recipient": ["G"]}}},
    )
    assert refused.structured_content is not None
    offered = refused.structured_content["fix"]["candidates"][0]["entity"]
    assert offered in text_of(refused)

    result = await _call_get_data(
        handler,
        _StubStore(stored),
        {"spec_id": stored.spec_id, "scope": {"entities": {"recipient": [offered]}}},
    )

    assert result.is_error is False
    assert resolve_texts == ["G"]
    assert requests[0]["input"]["scope"]["entities"] == {"recipient": ["GroupOfSeven"]}


@pytest.mark.parametrize("entity", ["dcid:", " dcid:  "])
async def test_a_marker_with_no_dcid_is_refused_before_any_call(entity: str) -> None:
    stored = _stored_definition()
    calls: list[str] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        calls.append(request.url.path)
        return httpx2.Response(500)

    result = await _call_get_data(
        handler,
        _StubStore(stored),
        {"spec_id": stored.spec_id, "scope": {"entities": {"recipient": [entity]}}},
    )

    assert result.is_error is True
    assert calls == []


async def test_slash_containing_spoken_name_is_resolved_not_treated_as_dcid_passthrough() -> None:
    """`Congo/Brazzaville` and `Bosnia/Herzegovina` are the standard spoken disambiguation for
    those countries -- exactly what a caller types, not a dcid the resolve door handed back.
    A capitalized segment before the slash means the value is a name to resolve, never a
    passthrough dcid.
    """
    stored = _stored_definition()
    resolve_texts: list[str] = []
    requests: list[dict[str, Any]] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        if request.url.path == "/api/graph/resolve":
            texts = json.loads(request.content)["texts"]
            resolve_texts.extend(texts)
            return httpx2.Response(
                200,
                json={"results": [resolved_result("Congo", "country/COG") for _ in texts]},
            )
        if request.url.path == "/api/datasets":
            requests.append(json.loads(request.content))
            return httpx2.Response(
                200,
                json=_manifest_response(
                    spec_id=stored.spec_id, entities={"recipient": ["country/COG", "country/BIH"]}
                ),
            )
        raise AssertionError(f"unexpected call to {request.url.path}")

    result = await _call_get_data(
        handler,
        _StubStore(stored),
        {
            "spec_id": stored.spec_id,
            "scope": {"entities": {"recipient": ["Congo/Brazzaville", "Bosnia/Herzegovina"]}},
        },
    )

    assert result.is_error is False
    # Both slash-containing names reach the resolve door as names -- neither
    # skips straight to mint as a passthrough dcid.
    assert resolve_texts == ["Congo/Brazzaville", "Bosnia/Herzegovina"]


# Duplicate dcid dedup


async def test_duplicate_dcids_across_a_region_and_its_member_are_deduped_before_the_wire() -> None:
    """`{"recipient": ["Sub-Saharan Africa", "Kenya"]}` appends `country/KEN` once from the
    region expansion and again from the individual resolve; upstream preserves this list
    verbatim into the hashed identity document, so an undeduped list would mint two
    different spec_ids for the same logical question depending on phrasing.
    """
    stored = _stored_definition()
    requests: list[dict[str, Any]] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        if request.url.path == "/api/graph/resolve":
            texts = json.loads(request.content)["texts"]
            results = []
            for text in texts:
                if text == "Sub-Saharan Africa":
                    results.append(
                        members_result(
                            "Sub-Saharan Africa",
                            "undata-geo/G00134000",
                            "subregion",
                            [("country/KEN", "Kenya"), ("country/NGA", "Nigeria")],
                        )
                    )
                elif text == "Kenya":
                    results.append(resolved_result("Kenya", "country/KEN"))
                else:
                    raise AssertionError(f"unexpected resolve text {text!r}")
            return httpx2.Response(200, json={"results": results})
        if request.url.path == "/api/datasets":
            requests.append(json.loads(request.content))
            return httpx2.Response(
                200,
                json=_manifest_response(
                    spec_id=stored.spec_id, entities={"recipient": ["country/KEN", "country/NGA"]}
                ),
            )
        raise AssertionError(f"unexpected call to {request.url.path}")

    result = await _call_get_data(
        handler,
        _StubStore(stored),
        {
            "spec_id": stored.spec_id,
            "scope": {"entities": {"recipient": ["Sub-Saharan Africa", "Kenya"]}},
        },
    )

    assert result.is_error is False
    assert requests[0]["input"]["scope"]["entities"]["recipient"] == ["country/KEN", "country/NGA"]


# Dropped-role notes


async def test_dropping_a_role_is_named_in_the_acknowledgement() -> None:
    stored = _stored_definition(
        entities={"provider": ["country/FRA"], "recipient": ["country/TGO"]}
    )

    def handler(request: httpx2.Request) -> httpx2.Response:
        if request.url.path == "/api/graph/resolve":
            return httpx2.Response(
                200,
                json={"results": [resolved_result("Togo", "country/TGO")]},
            )
        if request.url.path == "/api/datasets":
            return httpx2.Response(
                200,
                json=_manifest_response(
                    spec_id=stored.spec_id, entities={"recipient": ["country/TGO"]}
                ),
            )
        raise AssertionError(f"unexpected call to {request.url.path}")

    result = await _call_get_data(
        handler,
        _StubStore(stored),
        {"spec_id": stored.spec_id, "scope": {"entities": {"recipient": ["Togo"]}}},
    )

    assert result.is_error is False
    assert "provider" in text_of(result)
    assert "no longer bound" in text_of(result)


async def test_an_open_role_the_scope_leaves_out_is_not_named_dropped() -> None:
    # The open origin stays open under a scope that binds only the destination.
    stored = _stored_definition(entities={"destination": ["country/NGA"]}, open_roles=["origin"])

    def handler(request: httpx2.Request) -> httpx2.Response:
        if request.url.path == "/api/datasets":
            return httpx2.Response(
                200,
                json=_manifest_response(
                    spec_id=stored.spec_id, entities={"destination": ["country/GHA"]}
                ),
            )
        raise AssertionError(f"unexpected call to {request.url.path}")

    result = await _call_get_data(
        handler,
        _StubStore(stored),
        {"spec_id": stored.spec_id, "scope": {"entities": {"destination": ["dcid:country/GHA"]}}},
    )

    assert result.is_error is False
    assert "no longer bound" not in text_of(result)


async def test_explicit_empty_entities_map_does_not_warn_and_clears_entities() -> None:
    stored = _stored_definition(
        entities={"provider": ["country/FRA"], "recipient": ["country/TGO"]}
    )
    requests: list[dict[str, Any]] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        if request.url.path == "/api/datasets":
            requests.append(json.loads(request.content))
            return httpx2.Response(
                200, json=_manifest_response(spec_id=stored.spec_id, entities={})
            )
        raise AssertionError("an empty entities map needs no resolve call")

    result = await _call_get_data(
        handler, _StubStore(stored), {"spec_id": stored.spec_id, "scope": {"entities": {}}}
    )

    assert result.is_error is False
    assert "no longer bound" not in text_of(result)
    assert requests[0]["input"]["scope"]["entities"] == {}


# Re-mint identity


async def test_remint_returns_the_manifests_own_spec_id_never_the_argument() -> None:
    stored = _stored_definition()
    new_identity = _identity_doc(entities={"recipient": ["country/UGA"]})
    new_spec_id = spec_id_for(canonicalize(new_identity))

    def handler(request: httpx2.Request) -> httpx2.Response:
        if request.url.path == "/api/datasets":
            return httpx2.Response(
                200,
                json=_manifest_response(
                    spec_id=new_spec_id, entities={"recipient": ["country/UGA"]}
                ),
            )
        raise AssertionError(f"unexpected call to {request.url.path}")

    result = await _call_get_data(handler, _RecordingStore(stored), {"spec_id": stored.spec_id})

    assert result.is_error is False
    assert new_spec_id in text_of(result)
    assert result.structured_content is not None
    assert result.structured_content["definition"]["spec_id"] == new_spec_id


# Re-mint persistence: the new definition has to be readable by a later `get_data` call


async def test_remint_persists_the_new_definition_and_a_second_call_reads_it() -> None:
    """The digest hands the model a spec_id it just told it to use for downstream tools --
    including a later `get_data` call. Without persisting the re-minted definition, that
    later call would refuse `definition_not_held` even though this deployment's own upstream
    registry now holds the id, which is exactly what this test proves doesn't happen.
    """
    stored = _stored_definition()
    new_identity = _identity_doc(entities={"recipient": ["country/UGA"]})
    new_spec_id = spec_id_for(canonicalize(new_identity))
    store = _RecordingStore(stored)

    def handler(request: httpx2.Request) -> httpx2.Response:
        if request.url.path == "/api/graph/resolve":
            return httpx2.Response(
                200,
                json={"results": [resolved_result("Uganda", "country/UGA")]},
            )
        if request.url.path == "/api/datasets":
            return httpx2.Response(
                200,
                json=_manifest_response(
                    spec_id=new_spec_id, entities={"recipient": ["country/UGA"]}
                ),
            )
        raise AssertionError(f"unexpected call to {request.url.path}")

    result = await _call_get_data(
        handler,
        store,
        {"spec_id": stored.spec_id, "scope": {"entities": {"recipient": ["Uganda"]}}},
    )

    assert result.is_error is False
    assert len(store.put_calls) == 1
    persisted = store.put_calls[0]
    assert persisted.spec_id == new_spec_id
    assert json.loads(persisted.canonical_bytes) == new_identity

    def second_handler(request: httpx2.Request) -> httpx2.Response:
        if request.url.path == "/api/datasets":
            return httpx2.Response(
                200,
                json=_manifest_response(
                    spec_id=new_spec_id, entities={"recipient": ["country/UGA"]}
                ),
            )
        raise AssertionError(f"unexpected call to {request.url.path}")

    second_result = await _call_get_data(second_handler, store, {"spec_id": new_spec_id})

    # A store miss on `new_spec_id` would refuse `definition_not_held` instead of reaching
    # the mint call at all -- `is_error is False` here is only possible because the store
    # `.get()` above found what the first call persisted.
    assert second_result.is_error is False


async def test_remint_with_no_scope_change_never_writes_to_the_store() -> None:
    """When the mint route hands back the same spec_id the call was made with, nothing
    re-minted -- the base definition is already registered, so a write here would be pure
    overhead and would trip `_StubStore`'s own guard in every other test in this module.
    """
    stored = _stored_definition()

    result = await _call_get_data(
        _mint_only_handler(200, _manifest_response(spec_id=stored.spec_id)),
        _StubStore(stored),
        {"spec_id": stored.spec_id},
    )

    assert result.is_error is False


async def test_remint_persist_failure_still_returns_the_result_with_a_warning() -> None:
    """The mint already ran (and can take up to 125 seconds); a store outage on the write
    that follows shouldn't throw that away. The call still succeeds, but the digest must not
    claim the returned id is safely reusable when it might not be -- that would tell the model
    something untrue about a later `get_data` call against this same id.
    """
    stored = _stored_definition()
    new_identity = _identity_doc(entities={"recipient": ["country/UGA"]})
    new_spec_id = spec_id_for(canonicalize(new_identity))

    def handler(request: httpx2.Request) -> httpx2.Response:
        if request.url.path == "/api/datasets":
            return httpx2.Response(
                200,
                json=_manifest_response(
                    spec_id=new_spec_id, entities={"recipient": ["country/UGA"]}
                ),
            )
        raise AssertionError(f"unexpected call to {request.url.path}")

    result = await _call_get_data(handler, _PutFailsStore(stored), {"spec_id": stored.spec_id})

    assert result.is_error is False
    assert new_spec_id in text_of(result)
    assert "could not confirm that the definition store holds it" in text_of(result)


async def test_remint_persist_past_its_deadline_reserve_degrades_the_same_way(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The write above must finish with `CallBudget.post_mint_reserve_s` (30s) of the call deadline
    left. A write that outruns that bound degrades like a store outage does, and the
    frame fetch after it still gets the 30s."""
    clock = _FakeClock()
    monkeypatch.setattr(get_data_tool, "_deadline_clock", clock)
    stored = _stored_definition()
    new_identity = _identity_doc(entities={"recipient": ["country/UGA"]})
    new_spec_id = spec_id_for(canonicalize(new_identity))
    frame_remaining: list[float] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        if request.url.path == "/api/datasets":
            # Leaves 30.05s of the 165s deadline: 0.05s past the 30s the write must
            # finish with.
            clock.now += 134.95
            return httpx2.Response(
                200,
                json=_manifest_response(
                    spec_id=new_spec_id, entities={"recipient": ["country/UGA"]}
                ),
            )
        raise AssertionError(f"unexpected call to {request.url.path}")

    async def materialise(
        *, manifest: Manifest, deadline: CallDeadline, **_: Any
    ) -> MaterialisationOutcome:
        frame_remaining.append(deadline.remaining())
        return MaterialisationOutcome(
            status="registered", attempts=1, frame_ref=mint_pair(manifest)
        )

    monkeypatch.setattr(get_data_tool, "materialise_and_register", materialise)

    result = await _call_get_data(handler, _SlowPutStore(stored), {"spec_id": stored.spec_id})

    assert result.is_error is False
    assert new_spec_id in text_of(result)
    assert "could not confirm that the definition store holds it" in text_of(result)
    assert frame_remaining == [pytest.approx(30.05, abs=0.01)]


async def test_an_unconfirmed_frame_store_write_returns_the_digest_with_a_note(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stored = _stored_definition()

    def handler(request: httpx2.Request) -> httpx2.Response:
        if request.url.path == "/api/datasets":
            return httpx2.Response(200, json=_manifest_response(spec_id=stored.spec_id))
        raise AssertionError(f"unexpected call to {request.url.path}")

    async def materialise(*, manifest: Manifest, **_: Any) -> MaterialisationOutcome:
        return MaterialisationOutcome(
            status="degraded",
            attempts=1,
            degraded_reason="store_write",
            frame_ref=mint_pair(manifest),
        )

    monkeypatch.setattr(get_data_tool, "materialise_and_register", materialise)

    result = await _call_get_data(handler, _StubStore(stored), {"spec_id": stored.spec_id})

    assert result.is_error is False
    assert "the frame store did not confirm the write" in text_of(result)
    assert "could not be retrieved" not in text_of(result)


async def test_the_result_reports_the_frames_role_keys_and_stat_var_count_beside_the_manifest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stored = _stored_definition()
    role_keys = ("financingProvider", "financingRecipient", "stat_var_crsPurpose")

    def handler(request: httpx2.Request) -> httpx2.Response:
        if request.url.path == "/api/datasets":
            response = _manifest_response(spec_id=stored.spec_id)
            response["coverage"]["dimensions"] = [
                {"count": 1, "label": "financingProvider", "axis": "origin"},
                {"count": 2, "label": "financingRecipient", "axis": "destination"},
                {"count": 1, "label": "years"},
            ]
            return httpx2.Response(200, json=response)
        raise AssertionError(f"unexpected call to {request.url.path}")

    async def materialise(*, manifest: Manifest, **_: Any) -> MaterialisationOutcome:
        return MaterialisationOutcome(
            status="registered",
            attempts=1,
            role_keys=role_keys,
            stat_var_count=461,
            frame_ref=mint_pair(manifest),
        )

    monkeypatch.setattr(get_data_tool, "materialise_and_register", materialise)

    result = await _call_get_data(handler, _StubStore(stored), {"spec_id": stored.spec_id})

    assert result.is_error is False
    assert result.structured_content is not None
    assert result.structured_content["role_keys"] == list(role_keys)
    assert result.structured_content["stat_var_count"] == 461
    manifest = strip_manifest(result.structured_content)
    assert [(d.label, d.axis) for d in manifest.coverage.dimensions] == [
        ("financingProvider", "origin"),
        ("financingRecipient", "destination"),
        ("years", None),
    ]


async def test_the_result_states_no_frame_shape_when_the_frame_was_not_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stored = _stored_definition()

    def handler(request: httpx2.Request) -> httpx2.Response:
        if request.url.path == "/api/datasets":
            return httpx2.Response(200, json=_manifest_response(spec_id=stored.spec_id))
        raise AssertionError(f"unexpected call to {request.url.path}")

    async def materialise(*, manifest: Manifest, **_: Any) -> MaterialisationOutcome:
        return MaterialisationOutcome(
            status="degraded", attempts=3, degraded_reason="fetch", frame_ref=mint_pair(manifest)
        )

    monkeypatch.setattr(get_data_tool, "materialise_and_register", materialise)

    result = await _call_get_data(handler, _StubStore(stored), {"spec_id": stored.spec_id})

    assert result.is_error is False
    assert result.structured_content is not None
    assert "role_keys" not in result.structured_content
    assert "stat_var_count" not in result.structured_content


def _stat_var_entries(count: int) -> list[dict[str, Any]]:
    """`count` `trace.resolved_stat_vars` entries shaped as apis writes them."""
    prefix = "dcid:Amount_FinancialFlow_Gross_Disbursement_CRSTypeOfFlow10_CRSPurposeCode"
    return [
        {
            "ref": f"{prefix}{index:05d}_Nominal",
            "roles": ["financingProvider", "financingRecipient"],
            "observation_count": index % 2,
        }
        for index in range(count)
    ]


@pytest.mark.parametrize(
    ("member_count", "omitted"),
    [
        (MAX_LISTED_STAT_VARS, False),
        (MAX_LISTED_STAT_VARS + 1, True),
        # The Kenya development finance family whose whole list overran one result.
        (6_310, True),
    ],
)
async def test_a_long_stat_var_list_is_left_out_of_the_result_and_read_back_whole(
    member_count: int, omitted: bool
) -> None:
    stored = _stored_definition()
    response = _manifest_response(spec_id=stored.spec_id)
    response["trace"] = {"family_member_count": member_count}
    response["trace"]["resolved_stat_vars"] = _stat_var_entries(member_count)

    def handler(request: httpx2.Request) -> httpx2.Response:
        if request.url.path == "/api/datasets":
            return httpx2.Response(200, json=response)
        if request.url.path == f"/api/datasets/{stored.spec_id}/cv_1":
            return httpx2.Response(200, json=response)
        raise AssertionError(f"unexpected call to {request.url.path}")

    result = await _call_get_data(handler, _StubStore(stored), {"spec_id": stored.spec_id})

    assert result.is_error is False
    structured = result.structured_content
    assert structured is not None
    assert structured["trace"]["family_member_count"] == member_count
    assert ("resolved_stat_vars" in structured["trace"]) is not omitted
    assert structured.get(STAT_VARS_OMITTED_KEY) == (member_count if omitted else None)
    async with _client(handler) as upstream:
        client_holder.set_client(upstream)
        restored = await with_stat_var_list(structured)
    assert restored["trace"]["resolved_stat_vars"] == response["trace"]["resolved_stat_vars"]
    assert STAT_VARS_OMITTED_KEY not in restored


@pytest.mark.parametrize(
    ("bilateral", "omitted"), [(MAX_LISTED_STAT_VARS - 16, False), (6_310, True)]
)
async def test_a_composite_s_long_member_lists_are_left_out_of_the_result_and_read_back_whole(
    bilateral: int, omitted: bool
) -> None:
    stored = _stored_definition()
    response = _manifest_response(spec_id=stored.spec_id)
    members = [entry["ref"] for entry in _stat_var_entries(bilateral + 16)]
    response["trace"] = {
        "components": [
            {"role": "bilateral", "periods": [[2011, 2024]], "stat_vars": members[:bilateral]},
            {"role": "imputed", "periods": [[2011, 2024]], "stat_vars": members[bilateral:]},
        ]
    }

    def handler(request: httpx2.Request) -> httpx2.Response:
        if request.url.path in ("/api/datasets", f"/api/datasets/{stored.spec_id}/cv_1"):
            return httpx2.Response(200, json=response)
        raise AssertionError(f"unexpected call to {request.url.path}")

    result = await _call_get_data(handler, _StubStore(stored), {"spec_id": stored.spec_id})

    structured = result.structured_content
    assert structured is not None
    listed = [len(c["stat_vars"]) for c in structured["trace"]["components"]]
    assert listed == ([0, 0] if omitted else [bilateral, 16])
    assert structured.get(COMPONENT_STAT_VARS_OMITTED_KEY) == (bilateral + 16 if omitted else None)
    async with _client(handler) as upstream:
        client_holder.set_client(upstream)
        restored = await with_stat_var_list(structured)
    assert restored["trace"]["components"] == response["trace"]["components"]
    assert COMPONENT_STAT_VARS_OMITTED_KEY not in restored


async def test_a_snapshot_listing_other_stat_vars_than_the_result_left_out_is_refused() -> None:
    result = {
        "definition": {"spec_id": "spec_" + "a" * 32},
        "content_version": "cv_1",
        "trace": {},
        STAT_VARS_OMITTED_KEY: 300,
    }

    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, json={"trace": {"resolved_stat_vars": _stat_var_entries(299)}})

    async with _client(handler) as upstream:
        client_holder.set_client(upstream)
        with pytest.raises(ResourceError):
            await with_stat_var_list(result)


async def test_remint_identity_mismatch_refuses_rather_than_serving_an_unverified_definition() -> (
    None
):
    """The mint route's own manifest carries an identity that doesn't hash to the spec_id it
    claims -- the same corruption/spoofing check `search_tool.py` already applies on its own
    write path, applied here before this call ever persists or serves the re-minted result.
    """
    stored = _stored_definition()
    tampered_spec_id = "spec_" + "7" * 32

    def handler(request: httpx2.Request) -> httpx2.Response:
        if request.url.path == "/api/datasets":
            return httpx2.Response(
                200,
                json=_manifest_response(
                    spec_id=tampered_spec_id, entities={"recipient": ["country/UGA"]}
                ),
            )
        raise AssertionError(f"unexpected call to {request.url.path}")

    result = await _call_get_data(handler, _RecordingStore(stored), {"spec_id": stored.spec_id})

    assert result.is_error is True
    assert result.structured_content is not None
    assert result.structured_content["code"] == "capability_unavailable"


# Resource links on a successful mint


async def test_successful_mint_carries_a_text_block_first_then_one_resource_link() -> None:
    stored = _stored_definition()

    result = await _call_get_data(
        _mint_only_handler(200, _manifest_response(spec_id=stored.spec_id)),
        _StubStore(stored),
        {"spec_id": stored.spec_id},
    )

    assert result.is_error is False
    assert isinstance(result.content[0], TextContent)
    resource_links = [b for b in result.content if isinstance(b, ResourceLink)]
    assert len(resource_links) == 1
    assert str(resource_links[0].uri) == (
        f"dfo://artifact/{stored.spec_id}/cv_1/sha256:" + "1" * 64
    )


@pytest.mark.parametrize(
    ("rendering", "title"),
    [
        ({"format": "parquet", "layout": "frame"}, "parquet artifact"),
        ({"statistics": "series"}, "series statistics artifact"),
    ],
)
def test_a_resource_link_title_names_its_artifact_once(
    rendering: dict[str, Any], title: str
) -> None:
    response = _manifest_response(spec_id="spec_" + "a" * 32)
    response["artifacts"][0].update(rendering=rendering)

    (link,) = get_data_tool._artifact_resource_links(Manifest.model_validate(response))

    assert link.title == title
    # The link resolves to the JSON document holding the download URL.
    assert link.mime_type == "application/json"


async def test_a_sibling_rendering_gets_no_resource_link_and_leaves_the_manifest() -> None:
    stored = _stored_definition()

    result = await _call_get_data(
        _mint_only_handler(200, _manifest_response_with_two_artifacts(spec_id=stored.spec_id)),
        _StubStore(stored),
        {"spec_id": stored.spec_id},
    )

    assert result.is_error is False
    requested = "sha256:" + "1" * 64
    resource_links = [b for b in result.content if isinstance(b, ResourceLink)]
    assert [str(link.uri) for link in resource_links] == [
        f"dfo://artifact/{stored.spec_id}/cv_1/{requested}"
    ]
    assert result.structured_content is not None
    assert [a["artifact_digest"] for a in result.structured_content["artifacts"]] == [requested]


async def test_no_resource_link_carries_a_signed_url() -> None:
    stored = _stored_definition()

    result = await _call_get_data(
        _mint_only_handler(200, _manifest_response_with_two_artifacts(spec_id=stored.spec_id)),
        _StubStore(stored),
        {"spec_id": stored.spec_id},
    )

    assert result.is_error is False
    for block in result.content:
        dumped = json.dumps(block.model_dump(mode="json"))
        assert "/links/should-not-leak" not in dumped
        assert "/links/should-not-leak-either" not in dumped


async def test_a_refusal_result_carries_no_resource_links() -> None:
    async with Client(mcp) as client:
        result = await client.call_tool(
            "get_data",
            {"spec_id": "spec_" + "a" * 32, "scope": {"entities": {"recipient": ["x" * 201]}}},
            raise_on_error=False,
        )

    assert result.is_error is True
    assert not any(isinstance(block, ResourceLink) for block in result.content)


# No artifact URL leaks


async def test_no_artifact_url_leaks_anywhere_in_the_result() -> None:
    stored = _stored_definition()

    result = await _call_get_data(
        _mint_only_handler(200, _manifest_response(spec_id=stored.spec_id)),
        _StubStore(stored),
        {"spec_id": stored.spec_id},
    )

    assert result.is_error is False
    dumped_structured = json.dumps(result.structured_content)
    assert "/links/should-not-leak" not in dumped_structured
    for block in result.content:
        assert "/links/should-not-leak" not in json.dumps(block.model_dump(mode="json"))


# Unusable mint responses


async def test_non_json_mint_response_refuses_rather_than_raising() -> None:
    stored = _stored_definition()

    def handler(request: httpx2.Request) -> httpx2.Response:
        if request.url.path == "/api/datasets":
            return httpx2.Response(200, content=b"not json")
        raise AssertionError(f"unexpected call to {request.url.path}")

    result = await _call_get_data(handler, _StubStore(stored), {"spec_id": stored.spec_id})

    assert result.is_error is True
    assert result.structured_content is not None
    assert result.structured_content["code"] == "upstream_error"
    assert result.structured_content["retry_mode"] == "do_not_retry"
    assert result.structured_content["upstream_code"] is None


async def test_invalid_utf8_mint_response_refuses_rather_than_raising() -> None:
    """`httpx2.Response.json()` raises `UnicodeDecodeError` on invalid UTF-8, a `ValueError`
    sibling of `json.JSONDecodeError` -- not a subclass -- so a bare `except
    json.JSONDecodeError` lets this escape uncaught.
    """
    stored = _stored_definition()

    def handler(request: httpx2.Request) -> httpx2.Response:
        if request.url.path == "/api/datasets":
            return httpx2.Response(200, content=b"\xff\xfe\x00invalid utf8 \xff")
        raise AssertionError(f"unexpected call to {request.url.path}")

    result = await _call_get_data(handler, _StubStore(stored), {"spec_id": stored.spec_id})

    assert result.is_error is True
    assert result.structured_content is not None
    assert result.structured_content["code"] == "upstream_error"
    assert result.structured_content["retry_mode"] == "do_not_retry"
    assert result.structured_content["upstream_code"] is None


async def test_malformed_artifact_refuses_without_leaking_its_signed_url() -> None:
    """A malformed artifact fails validation before `strip_manifest` ever gets to drop its
    signed `url` -- pydantic-core's default error rendering embeds the raw, un-stripped input
    for the failing field, so a naive `str(exc)` would put that URL into both the refusal
    message (`content`) and the structured `fix.detail` (`structured_content`). This is the
    same signed URL the success path already proves never leaks, in
    `test_no_artifact_url_leaks_anywhere_in_the_result`.

    The artifact below carries only its signed `url`, missing every other required field:
    with more keys present, pydantic-core's own truncation of the `input_value` repr in
    `str(exc)` would cut the URL short before it reaches this test's full-string match,
    understating the leak rather than reproducing it in full.
    """
    stored = _stored_definition()
    malformed_response = _manifest_response(spec_id=stored.spec_id)
    malformed_response["artifacts"] = [{"url": "/links/should-not-leak"}]

    result = await _call_get_data(
        _mint_only_handler(200, malformed_response),
        _StubStore(stored),
        {"spec_id": stored.spec_id},
    )

    assert result.is_error is True
    assert result.structured_content is not None
    dumped_structured = json.dumps(result.structured_content)
    assert "/links/should-not-leak" not in text_of(result)
    assert "/links/should-not-leak" not in dumped_structured


# Region resolved with no confirmed members


async def test_region_with_no_confirmed_members_refuses_entity_not_found() -> None:
    stored = _stored_definition()

    def handler(request: httpx2.Request) -> httpx2.Response:
        if request.url.path == "/api/graph/resolve":
            return httpx2.Response(
                200,
                json={"results": [members_result("Micronesia", "micronesia", "subregion", [])]},
            )
        raise AssertionError("mint should not be reached when no member is confirmed")

    result = await _call_get_data(
        handler,
        _StubStore(stored),
        {"spec_id": stored.spec_id, "scope": {"entities": {"recipient": ["Micronesia"]}}},
    )

    assert result.is_error is True
    assert result.structured_content is not None
    assert result.structured_content["code"] == "entity_not_found"
    assert result.structured_content["fix"]["role"] == "recipient"
    assert result.structured_content["fix"]["not_held"] is True
    assert result.structured_content["fix"]["entity_type"] == "subregion"


# Local argument validation: every input cap


async def test_too_many_roles_is_rejected_locally() -> None:
    entities = {f"role{i}": ["x"] for i in range(9)}
    async with Client(mcp) as client:
        result = await client.call_tool(
            "get_data",
            {"spec_id": "spec_" + "a" * 32, "scope": {"entities": entities}},
            raise_on_error=False,
        )

    assert result.is_error is True


async def test_too_many_entity_names_summed_is_rejected_locally() -> None:
    entities = {"recipient": [f"Country{i}" for i in range(51)]}
    async with Client(mcp) as client:
        result = await client.call_tool(
            "get_data",
            {"spec_id": "spec_" + "a" * 32, "scope": {"entities": entities}},
            raise_on_error=False,
        )

    assert result.is_error is True


async def test_overlength_entity_name_is_rejected_locally() -> None:
    async with Client(mcp) as client:
        result = await client.call_tool(
            "get_data",
            {"spec_id": "spec_" + "a" * 32, "scope": {"entities": {"recipient": ["x" * 201]}}},
            raise_on_error=False,
        )

    assert result.is_error is True


async def test_empty_entity_name_is_rejected_locally() -> None:
    async with Client(mcp) as client:
        result = await client.call_tool(
            "get_data",
            {"spec_id": "spec_" + "a" * 32, "scope": {"entities": {"recipient": [""]}}},
            raise_on_error=False,
        )

    assert result.is_error is True


async def test_malformed_spec_id_is_rejected_locally() -> None:
    async with Client(mcp) as client:
        result = await client.call_tool(
            "get_data", {"spec_id": "not-a-spec-id"}, raise_on_error=False
        )

    assert result.is_error is True


async def test_short_spec_id_is_rejected_locally() -> None:
    async with Client(mcp) as client:
        result = await client.call_tool("get_data", {"spec_id": "spec_short"}, raise_on_error=False)

    assert result.is_error is True


async def test_unforeseen_scope_field_is_rejected_locally() -> None:
    async with Client(mcp) as client:
        result = await client.call_tool(
            "get_data",
            {"spec_id": "spec_" + "a" * 32, "scope": {"groupby": ["recipient"]}},
            raise_on_error=False,
        )

    assert result.is_error is True


# The mint input: the spec_id and its stored definition.


def _described_identity_doc(
    *,
    entities: dict[str, list[str]] | None = None,
    window: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "identity_version": "spec1",
        "shape": {
            "kind": "statvar_schema",
            "population_type": "Person",
            "measured_property": "count",
            "stat_type": "measuredValue",
        },
        "entities": entities if entities is not None else {"observationAbout": ["country/GTM"]},
        "requested_window": window if window is not None else {"kind": "range", "start_year": 2010},
        "slots": [{"property": "gender", "binding": {"state": "unbound"}}],
    }


def _described_stored_definition(**identity_kwargs: Any) -> StoredDefinition:
    canonical_bytes = canonicalize(_described_identity_doc(**identity_kwargs))
    return StoredDefinition(
        spec_id=spec_id_for(canonical_bytes), schema_version=1, canonical_bytes=canonical_bytes
    )


async def test_the_mint_posts_the_spec_id_and_its_definition_alone() -> None:
    stored = _described_stored_definition()
    requests: list[dict[str, Any]] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        assert request.url.path == "/api/datasets"
        requests.append(json.loads(request.content))
        return httpx2.Response(200, json=_manifest_response(spec_id=stored.spec_id))

    result = await _call_get_data(
        handler,
        _StubStore(stored),
        {"spec_id": stored.spec_id},
    )

    assert result.is_error is False
    assert requests[0]["input"] == {
        "kind": "spec_id",
        "spec_id": stored.spec_id,
        "scope": {},
        "definition": {
            "schema_version": 1,
            "spec_id": stored.spec_id,
            "identity": _described_identity_doc(),
        },
    }
    assert result.structured_content is not None
    assert result.structured_content["definition"]["spec_id"] == stored.spec_id


async def test_the_registry_fallback_mint_posts_the_spec_id_alone() -> None:
    spec_id = "spec_" + "a" * 32
    requests: list[dict[str, Any]] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        requests.append(json.loads(request.content))
        return httpx2.Response(200, json=_manifest_response(spec_id=spec_id))

    result = await _call_get_data(
        handler,
        _StubStore(None),
        {"spec_id": spec_id},
    )

    assert result.is_error is False
    assert requests[0]["input"] == {"kind": "spec_id", "spec_id": spec_id, "scope": {}}


async def test_a_member_list_is_refused_before_any_call() -> None:
    """apis resolves a family's members itself and takes no list."""

    def handler(request: httpx2.Request) -> httpx2.Response:
        raise AssertionError("a refused argument must not reach dfo-apis")

    result = await _call_get_data(
        handler,
        _StubStore(_described_stored_definition()),
        {
            "spec_id": "spec_" + "a" * 32,
            "resolved_stat_var_dcids": ["Count_Person"],
        },
    )

    assert result.is_error is True


async def test_a_scoped_remint_stores_the_new_identity_alone() -> None:
    stored = _described_stored_definition()
    window = {"kind": "range", "start_year": 2020, "end_year": 2020}
    windowed = _described_identity_doc(window=window)
    windowed_spec_id = spec_id_for(canonicalize(windowed))
    response = _manifest_response(spec_id=windowed_spec_id)
    response["definition"]["identity"] = windowed
    requests: list[dict[str, Any]] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        requests.append(json.loads(request.content))
        return httpx2.Response(200, json=response)

    store = _RecordingStore(stored)
    result = await _call_get_data(
        handler,
        store,
        {
            "spec_id": stored.spec_id,
            "scope": {"window": window},
        },
    )

    assert result.is_error is False
    assert len(requests) == 1
    assert store.put_calls == [
        StoredDefinition(
            spec_id=windowed_spec_id, schema_version=1, canonical_bytes=canonicalize(windowed)
        )
    ]


# End-to-end materialisation: a real `get_data` call, through the real
# `materialise_and_register`, registers a real frame in the shared store.


def _parquet_bytes() -> bytes:
    """A one-row frame in the canonical layout a version's canonical frame is in."""
    return _frame_parquet(["dcid:Test"])


def _content_version_of(data: bytes) -> str:
    """The `content_version` apis mints for the canonical frame `data` holds."""
    return mint_content_version(frame_sha256=frame_sha256(frame_from_verified_parquet_bytes(data)))


def _digest(data: bytes) -> str:
    return f"sha256:{hashlib.sha256(data).hexdigest()}"


def _serve_canonical_frames(
    monkeypatch: pytest.MonkeyPatch, bytes_by_cv: Mapping[str, bytes]
) -> None:
    """Answer the canonical-frame lookup of each `content_version` in `bytes_by_cv`
    as the snapshot route would list those bytes."""

    async def locate(ref: FrameRef, **_: Any) -> tuple[str, str]:
        return _digest(bytes_by_cv[ref.content_version]), f"https://apis.test/{ref.content_version}"

    monkeypatch.setattr("data_mcp.frame_materialisation._locate_canonical_frame", locate)


async def test_successful_get_data_call_registers_a_frame_end_to_end(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The real `materialise_and_register` (not the module's own autouse stub above)
    fetches, parses, and registers a frame for a real `get_data` call. The bare
    signed-URL GET is the one hop `get_data_tool.py` never threads a test transport
    into (see `frame_materialisation.materialise_and_register`'s own `transport`
    parameter, a seam `test_frame_materialisation.py` exercises directly); patched here
    at `_fetch_artifact_bytes` so this test proves the tool-to-store wiring without
    opening a real socket.
    """
    monkeypatch.undo()  # remove this module's own autouse stub for this one test
    stored = _stored_definition()
    data = _parquet_bytes()
    content_version = _content_version_of(data)
    rendering_digest = "sha256:" + "4" * 64
    fetched: list[tuple[str, str | None]] = []

    async def fake_fetch(*, artifact_digest: str, link: str | None, **_kwargs: Any) -> bytes:
        fetched.append((artifact_digest, link))
        return data

    monkeypatch.setattr("data_mcp.frame_materialisation._fetch_artifact_bytes", fake_fetch)
    minted = _manifest_response(
        spec_id=stored.spec_id, content_version=content_version, artifact_digest=rendering_digest
    )
    # The snapshot of the version lists its canonical frame beside the rendering.
    snapshot = json.loads(json.dumps(minted))
    snapshot["artifacts"].append(
        snapshot["artifacts"][0]
        | {
            "artifact_digest": _digest(data),
            "rendering": dict(CANONICAL_FRAME_RENDERING),
            "url": f"/api/datasets/{stored.spec_id}/{content_version}/files/canonical",
        }
    )

    def handler(request: httpx2.Request) -> httpx2.Response:
        if request.url.path == "/api/datasets":
            return httpx2.Response(200, json=minted)
        if request.url.path == f"/api/datasets/{stored.spec_id}/{content_version}":
            return httpx2.Response(200, json=snapshot)
        raise AssertionError(f"unexpected call to {request.url.path}")

    frame_store = InMemoryMaterialisationStore()
    set_frame_store(frame_store)
    result = await _call_get_data(handler, _StubStore(stored), {"spec_id": stored.spec_id})

    assert result.is_error is False
    # Checked on `frame_store` directly, not via `get_frame_store()`: `Client(mcp)`'s own
    # shutdown lifespan already ran and cleared the process-wide holder by the time
    # `_call_get_data` returns (its `return` sits inside the `async with Client(mcp)`
    # block, so the block's exit -- and `aclose_frame_store()` with it -- happens before
    # control reaches here). `InMemoryMaterialisationStore.aclose()` is a no-op, so the
    # object itself still holds what was registered.
    stored_frame = await frame_store.get(
        FrameRef(spec_id=stored.spec_id, content_version=content_version)
    )
    assert stored_frame is not None
    frame, _trace = stored_frame
    assert len(frame.rows) == 1
    assert [(digest, link and link.rsplit("/", 1)[-1]) for digest, link in fetched] == [
        (_digest(data), "canonical")
    ]


async def test_a_frame_request_reads_the_frame_artifact_past_a_long_layout_one_listed_first(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The mint asks for `layout: "frame"`, and of a manifest listing the long-layout
    file under the same pair first, the frame file is the one linked. The frame
    registered is the version's canonical frame."""
    monkeypatch.undo()  # remove this module's own autouse stub for this one test
    stored = _stored_definition()
    data = _parquet_bytes()
    content_version = _content_version_of(data)
    frame_digest = "sha256:" + "5" * 64
    _serve_canonical_frames(monkeypatch, {content_version: data})
    manifest = _manifest_response(
        spec_id=stored.spec_id, content_version=content_version, artifact_digest=frame_digest
    )
    long_layout = {
        **manifest["artifacts"][0],
        "artifact_digest": "sha256:" + "9" * 64,
        "rendering": {"bundle": False, "format": "parquet", "layout": 2},
    }
    manifest["artifacts"].insert(0, long_layout)
    posted: list[dict[str, Any]] = []
    fetched_digests: list[str] = []

    async def fake_fetch(*, artifact_digest: str, **_kwargs: Any) -> bytes:
        fetched_digests.append(artifact_digest)
        return data

    monkeypatch.setattr("data_mcp.frame_materialisation._fetch_artifact_bytes", fake_fetch)

    def handler(request: httpx2.Request) -> httpx2.Response:
        if request.url.path == "/api/datasets":
            posted.append(json.loads(request.content))
            return httpx2.Response(200, json=manifest)
        raise AssertionError(f"unexpected call to {request.url.path}")

    frame_store = InMemoryMaterialisationStore()
    set_frame_store(frame_store)
    result = await _call_get_data(handler, _StubStore(stored), {"spec_id": stored.spec_id})

    assert result.is_error is False
    assert [body["options"] for body in posted] == [
        {"format": "parquet", "bundle": False, "layout": "frame"}
    ]
    assert fetched_digests == [_digest(data)]
    resource_links = [b for b in result.content if isinstance(b, ResourceLink)]
    assert [str(link.uri) for link in resource_links] == [
        f"dfo://artifact/{stored.spec_id}/{content_version}/{frame_digest}"
    ]
    assert (
        await frame_store.get(FrameRef(spec_id=stored.spec_id, content_version=content_version))
        is not None
    )


# Per-series statistics: fetched, rendered, and degraded


async def test_statistics_are_fetched_and_rendered_in_the_digest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stored = _stored_definition()

    async def fake_fetch(**_kwargs: Any) -> list[SeriesStatisticsModel]:
        return [
            SeriesStatisticsModel.model_validate(
                {
                    "stat_var": "dcid:MalariaGrantDisbursements",
                    "entities": [{"role": "recipient", "ref": "country/KEN"}],
                    "facet": {"unit": "USD"},
                    "observations": 1,
                    "first": {"value": 1000.0, "period": "2015"},
                    "latest": {"value": 1000.0, "period": "2015"},
                    "minimum": {"value": 1000.0, "period": "2015"},
                    "maximum": {"value": 1000.0, "period": "2015"},
                }
            )
        ]

    monkeypatch.setattr(get_data_tool, "fetch_series_statistics", fake_fetch)

    handler = _mint_only_handler(200, _manifest_response_with_statistics(spec_id=stored.spec_id))
    result = await _call_get_data(handler, _StubStore(stored), {"spec_id": stored.spec_id})

    assert result.is_error is False
    assert "MalariaGrantDisbursements" in text_of(result)
    assert "first 1.0k (2015)" in text_of(result)


async def test_a_statistics_fetch_failure_degrades_to_a_count_only_line(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stored = _stored_definition()

    async def failing_fetch(**_kwargs: Any) -> list[SeriesStatisticsModel]:
        raise SeriesStatisticsSchemaError("expected dfo.schema=... got None")

    monkeypatch.setattr(get_data_tool, "fetch_series_statistics", failing_fetch)

    handler = _mint_only_handler(200, _manifest_response_with_statistics(spec_id=stored.spec_id))
    result = await _call_get_data(handler, _StubStore(stored), {"spec_id": stored.spec_id})

    assert result.is_error is False
    assert "could not fetch or decode" in text_of(result)
    assert "MalariaGrantDisbursements" not in text_of(result).split("Statistics:")[-1]


async def test_statistics_past_the_cap_never_fetches(monkeypatch: pytest.MonkeyPatch) -> None:
    stored = _stored_definition()
    calls: list[object] = []

    async def fake_fetch(**kwargs: Any) -> list[SeriesStatisticsModel]:
        calls.append(kwargs)
        return []

    monkeypatch.setattr(get_data_tool, "fetch_series_statistics", fake_fetch)

    handler = _mint_only_handler(
        200,
        _manifest_response_with_statistics(spec_id=stored.spec_id, series_count=9),
    )
    result = await _call_get_data(handler, _StubStore(stored), {"spec_id": stored.spec_id})

    assert result.is_error is False
    assert calls == []
    assert "9 series retrieved" in text_of(result)


# The call deadline: every step takes its bound from what remains of one deadline


def _resolved_uganda() -> httpx2.Response:
    return httpx2.Response(
        200,
        json={"results": [resolved_result("Uganda", "country/UGA")]},
    )


async def test_a_direct_call_takes_its_deadline_from_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An override between the 165s default and the 170s cap takes effect."""
    clock = _FakeClock()
    monkeypatch.setattr(get_data_tool, "_deadline_clock", clock)
    monkeypatch.setattr(get_data_tool, "Settings", lambda: Settings(get_data_deadline_s=168.0))
    stored = _stored_definition()
    expires_at: list[float] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        assert request.url.path == "/api/datasets"
        return httpx2.Response(200, json=_manifest_response(spec_id=stored.spec_id))

    async def materialise(
        *, manifest: Manifest, deadline: CallDeadline, **_: Any
    ) -> MaterialisationOutcome:
        expires_at.append(deadline.expires_at)
        return MaterialisationOutcome(
            status="registered", attempts=1, frame_ref=mint_pair(manifest)
        )

    monkeypatch.setattr(get_data_tool, "materialise_and_register", materialise)

    result = await _call_get_data(handler, _StubStore(stored), {"spec_id": stored.spec_id})

    assert result.is_error is False
    assert expires_at == [168.0]


async def test_entity_resolution_past_its_share_of_the_deadline_refuses_before_minting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A 50.05s deadline leaves pre-mint work 0.05s before the 50s it must keep."""
    monkeypatch.setattr(get_data_tool, "Settings", lambda: Settings(get_data_deadline_s=50.05))
    stored = _stored_definition()
    paths: list[str] = []

    async def handler(request: httpx2.Request) -> httpx2.Response:
        paths.append(request.url.path)
        await asyncio.sleep(1.0)
        return _resolved_uganda()

    result = await _call_get_data(
        handler,
        _StubStore(stored),
        {"spec_id": stored.spec_id, "scope": {"entities": {"recipient": ["Uganda"]}}},
    )

    assert result.is_error is True
    assert result.structured_content is not None
    assert result.structured_content["code"] == "service_unavailable"
    assert result.structured_content["retry_mode"] == "retry_after_delay"
    assert result.structured_content["fix"]["service"] == "dfo-apis"
    assert paths == ["/api/graph/resolve"]


async def test_the_mint_post_keeps_the_reserve_for_what_follows_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Resolution spends 20s of the 165s deadline; the mint POST then gets
    165 - 20 - 30 = 115s, not its 125s cap."""
    clock = _FakeClock()
    monkeypatch.setattr(get_data_tool, "_deadline_clock", clock)
    stored = _stored_definition()
    mint_read_timeouts: list[float] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        if request.url.path == "/api/graph/resolve":
            clock.now += 20.0
            return _resolved_uganda()
        if request.url.path == "/api/datasets":
            mint_read_timeouts.append(request.extensions["timeout"]["read"])
            return httpx2.Response(200, json=_manifest_response(spec_id=stored.spec_id))
        raise AssertionError(f"unexpected call to {request.url.path}")

    result = await _call_get_data(
        handler,
        _StubStore(stored),
        {"spec_id": stored.spec_id, "scope": {"entities": {"recipient": ["Uganda"]}}},
    )

    assert result.is_error is False
    assert mint_read_timeouts == [pytest.approx(115.0, abs=0.01)]


async def test_a_mint_post_cut_by_the_deadline_refuses_as_unanswered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = _FakeClock()
    monkeypatch.setattr(get_data_tool, "_deadline_clock", clock)
    stored = _stored_definition()

    def handler(request: httpx2.Request) -> httpx2.Response:
        assert request.url.path == "/api/datasets"
        clock.now += 140.0
        raise httpx2.ReadTimeout("no answer", request=request)

    result = await _call_get_data(handler, _StubStore(stored), {"spec_id": stored.spec_id})

    assert result.is_error is True
    assert result.structured_content is not None
    assert result.structured_content["code"] == "service_unavailable"
    assert result.structured_content["retry_mode"] == "retry_after_delay"
    assert result.structured_content["fix"]["spec_id"] == stored.spec_id
    assert result.structured_content["fix"]["elapsed_s"] == 140.0
    assert "may still be running it" in result.structured_content["message"]
    assert "is safe" in result.structured_content["message"]


async def test_a_mint_post_cut_by_its_own_cap_with_deadline_left_refuses_as_unanswered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The mint POST's own `CallBudget.mint_timeout_s` (125s) cap ends it first, well inside the
    165s call deadline (135s remained after the pre-mint reserve). apis does not
    cancel a mint whose client went away, so this refuses with `mint_unanswered`, like
    a deadline-cut POST, whose message says the mint may still be running."""
    clock = _FakeClock()
    monkeypatch.setattr(get_data_tool, "_deadline_clock", clock)
    stored = _stored_definition()

    def handler(request: httpx2.Request) -> httpx2.Response:
        clock.now += 125.0
        raise httpx2.ReadTimeout("no answer", request=request)

    result = await _call_get_data(handler, _StubStore(stored), {"spec_id": stored.spec_id})

    assert result.is_error is True
    assert result.structured_content is not None
    assert result.structured_content["code"] == "service_unavailable"
    assert result.structured_content["retry_mode"] == "retry_after_delay"
    assert result.structured_content["fix"]["spec_id"] == stored.spec_id
    assert "may still be running it" in result.structured_content["message"]


async def test_a_near_deadline_async_mint_finishes_inside_the_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A 202 after 100s, then 20s of polling to `done`, leaves 45s: the frame fetch
    and the statistics fetch each see what is left, and the call returns its digest
    with the deadline unspent."""
    clock = _FakeClock()
    monkeypatch.setattr(get_data_tool, "_deadline_clock", clock)
    stored = _stored_definition()
    seen_remaining: dict[str, float] = {}

    def handler(request: httpx2.Request) -> httpx2.Response:
        if request.url.path == "/api/datasets":
            clock.now += 100.0
            return httpx2.Response(
                202,
                json={
                    "operation": {
                        "operation_id": _OPERATION_ID,
                        "status": "pending",
                        "retry_after_ms": 1,
                    }
                },
            )
        if request.url.path == _OPERATION_POLL_PATH:
            clock.now += 20.0
            return httpx2.Response(
                200,
                json={
                    "operation": {"operation_id": _OPERATION_ID, "status": "done"},
                    "manifest": _manifest_response_with_statistics(spec_id=stored.spec_id),
                },
            )
        raise AssertionError(f"unexpected call to {request.url.path}")

    async def materialise(
        *, manifest: Manifest, deadline: CallDeadline, **_: Any
    ) -> MaterialisationOutcome:
        seen_remaining["frame"] = deadline.remaining()
        return MaterialisationOutcome(
            status="registered", attempts=1, frame_ref=mint_pair(manifest)
        )

    async def fetch_statistics(*, deadline: CallDeadline, **_: Any) -> list[SeriesStatisticsModel]:
        seen_remaining["statistics"] = deadline.remaining()
        return []

    monkeypatch.setattr(get_data_tool, "materialise_and_register", materialise)
    monkeypatch.setattr(get_data_tool, "fetch_series_statistics", fetch_statistics)

    result = await _call_get_data(handler, _StubStore(stored), {"spec_id": stored.spec_id})

    assert result.is_error is False
    assert seen_remaining == {"frame": 45.0, "statistics": 42.0}
    assert clock.now < 165.0


# The external call budget


def _poll_waits_move(clock: _FakeClock, monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the operation poll's waits move `clock` instead of real time."""

    async def sleep(seconds: float) -> None:
        clock.now += seconds

    monkeypatch.setattr(
        get_data_tool,
        "resolve_operation_response",
        functools.partial(get_data_tool.resolve_operation_response, sleep=sleep),
    )


def _operation_done_at(clock: _FakeClock, *, done_at: float, spec_id: str) -> Handler:
    """apis for one mint over its sync budget, done once `clock` reaches `done_at`.

    Every POST answers 202 with the same operation id, as a repeat submit rejoins
    the operation while coverage is unchanged.
    """

    def status() -> str:
        return "done" if clock.now >= done_at else "running"

    def handler(request: httpx2.Request) -> httpx2.Response:
        operation = {"operation_id": _OPERATION_ID, "status": status()}
        if request.url.path == "/api/datasets":
            return httpx2.Response(202, json={"operation": operation})
        if request.url.path == _OPERATION_POLL_PATH:
            body: dict[str, Any] = {"operation": operation}
            if operation["status"] == "done":
                body["manifest"] = _manifest_response(spec_id=spec_id)
            return httpx2.Response(200, json=body)
        raise AssertionError(f"unexpected call to {request.url.path}")

    return handler


async def test_an_external_call_refuses_a_202_inside_42s_and_a_repeat_returns_the_frame(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The poll stops with the external budget's 10s post-mint reserve in hand, and
    the repeat call the refusal asks for rejoins the operation and returns its frame."""
    clock = _FakeClock()
    monkeypatch.setattr(get_data_tool, "_deadline_clock", clock)
    _poll_waits_move(clock, monkeypatch)
    stored = _stored_definition()
    handler = _operation_done_at(clock, done_at=50.0, spec_id=stored.spec_id)

    with use_call_budget(EXTERNAL_CALL_BUDGET):
        first = await _call_get_data(handler, _StubStore(stored), {"spec_id": stored.spec_id})
        first_ended_at = clock.now
        assert first.structured_content is not None
        clock.now += first.structured_content["retry_after_s"]
        second_started_at = clock.now
        second = await _call_get_data(handler, _StubStore(stored), {"spec_id": stored.spec_id})

    assert first.is_error is True
    assert first.structured_content["code"] == "operation_still_processing"
    assert first.structured_content["retry_after_s"] == 5.0
    assert EXTERNAL_CALL_BUDGET.deadline_s is not None
    assert (
        first_ended_at <= EXTERNAL_CALL_BUDGET.deadline_s - EXTERNAL_CALL_BUDGET.post_mint_reserve_s
    )
    assert second.is_error is False
    assert stored.spec_id in text_of(second)
    assert clock.now - second_started_at < 45.0


async def test_an_external_call_sees_a_125s_operation_finish_within_four_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = _FakeClock()
    monkeypatch.setattr(get_data_tool, "_deadline_clock", clock)
    _poll_waits_move(clock, monkeypatch)
    stored = _stored_definition()
    handler = _operation_done_at(clock, done_at=125.0, spec_id=stored.spec_id)
    outcomes: list[str] = []
    durations: list[float] = []

    with use_call_budget(EXTERNAL_CALL_BUDGET):
        for _ in range(4):
            started_at = clock.now
            result = await _call_get_data(handler, _StubStore(stored), {"spec_id": stored.spec_id})
            durations.append(clock.now - started_at)
            if not result.is_error:
                outcomes.append("frame")
                break
            assert result.structured_content is not None
            outcomes.append(result.structured_content["code"])
            clock.now += result.structured_content["retry_after_s"]

    assert outcomes == ["operation_still_processing"] * 3 + ["frame"]
    assert all(duration < 45.0 for duration in durations)


async def test_an_external_mint_post_gets_the_32s_before_the_reserve_and_promises_no_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A synchronous mint slower than the external budget allows gets no answer; the
    refusal says a repeat may run it again rather than that a repeat is safe."""
    clock = _FakeClock()
    monkeypatch.setattr(get_data_tool, "_deadline_clock", clock)
    stored = _stored_definition()
    mint_read_timeouts: list[float] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        assert request.url.path == "/api/datasets"
        mint_read_timeouts.append(request.extensions["timeout"]["read"])
        clock.now += 32.0
        raise httpx2.ReadTimeout("no answer", request=request)

    with use_call_budget(EXTERNAL_CALL_BUDGET):
        result = await _call_get_data(handler, _StubStore(stored), {"spec_id": stored.spec_id})

    assert mint_read_timeouts == [pytest.approx(32.0, abs=0.01)]
    assert result.is_error is True
    assert result.structured_content is not None
    assert result.structured_content["code"] == "service_unavailable"
    assert result.structured_content["retry_mode"] == "retry_after_delay"
    assert result.structured_content["retry_after_s"] == pytest.approx(93.0)
    message = result.structured_content["message"]
    assert "may still be running it upstream" in message
    assert "runs the request again from the start" in message
    assert "is safe" not in message


def _mint_never_answers(clock: _FakeClock, posted: list[str]) -> Handler:
    """apis running every mint synchronously for longer than the external
    budget's 32s: each POST records its spec_id and gets no answer."""

    def handler(request: httpx2.Request) -> httpx2.Response:
        assert request.url.path == "/api/datasets"
        posted.append(json.loads(request.content)["input"]["spec_id"])
        clock.now += 35.0
        raise httpx2.ReadTimeout("no answer", request=request)

    return handler


async def test_an_abandoned_external_mint_holds_its_slot_until_its_window_ends(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """apis runs an abandoned synchronous mint for up to 120s after its POST, and the
    ledger holds its slot a further margin past that. Until then a mint for another
    definition finds the one external slot held and refuses without sending, with the
    time left as its wait."""
    clock = _FakeClock()
    monkeypatch.setattr(get_data_tool, "_deadline_clock", clock)
    first, second = _stored_definition(), _stored_definition(shape_id="OtherShape")
    posted: list[str] = []
    handler = _mint_never_answers(clock, posted)

    with use_call_budget(EXTERNAL_CALL_BUDGET):
        abandoned = await _call_get_data(handler, _StubStore(first), {"spec_id": first.spec_id})
        clock.now = 124.5
        refused = await _call_get_data(handler, _StubStore(second), {"spec_id": second.spec_id})
        clock.now = 125.0
        admitted = await _call_get_data(handler, _StubStore(second), {"spec_id": second.spec_id})

    assert abandoned.structured_content is not None
    assert abandoned.structured_content["retry_after_s"] == pytest.approx(90.0)
    assert refused.structured_content is not None
    assert refused.structured_content["code"] == "service_unavailable"
    assert refused.structured_content["fix"] == {"service": "dfo-apis", "limit": 1}
    assert refused.structured_content["retry_after_s"] == pytest.approx(0.5)
    assert admitted.structured_content is not None
    assert admitted.structured_content["fix"]["spec_id"] == second.spec_id
    assert posted == [first.spec_id, second.spec_id]


async def test_a_repeat_of_an_abandoned_external_mint_refuses_until_its_window_ends(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A repeat of the same call inside the window would start the same mint twice; it
    refuses with the time left, and a repeat once that has passed mints again."""
    clock = _FakeClock()
    monkeypatch.setattr(get_data_tool, "_deadline_clock", clock)
    stored = _stored_definition()
    posted: list[str] = []
    handler = _mint_never_answers(clock, posted)

    with use_call_budget(EXTERNAL_CALL_BUDGET):
        await _call_get_data(handler, _StubStore(stored), {"spec_id": stored.spec_id})
        clock.now = 40.0
        refused = await _call_get_data(handler, _StubStore(stored), {"spec_id": stored.spec_id})
        assert refused.structured_content is not None
        clock.now += refused.structured_content["retry_after_s"]
        await _call_get_data(handler, _StubStore(stored), {"spec_id": stored.spec_id})

    assert refused.structured_content["code"] == "service_unavailable"
    assert refused.structured_content["retry_after_s"] == pytest.approx(85.0)
    assert refused.structured_content["fix"] == {
        "service": "dfo-apis",
        "spec_id": stored.spec_id,
        "started_s_ago": 40.0,
    }
    assert posted == [stored.spec_id, stored.spec_id]


async def test_a_direct_call_is_neither_limited_nor_counted_by_the_mint_ledger(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With the external slot held, direct calls still mint, and a direct mint that got
    no answer holds no slot: an external call for its definition mints once the
    external mint's window has ended."""
    clock = _FakeClock()
    monkeypatch.setattr(get_data_tool, "_deadline_clock", clock)
    held, other = _stored_definition(), _stored_definition(shape_id="OtherShape")
    posted: list[str] = []
    answers = iter(["none", "manifest", "none", "manifest"])

    def handler(request: httpx2.Request) -> httpx2.Response:
        assert request.url.path == "/api/datasets"
        spec_id = json.loads(request.content)["input"]["spec_id"]
        posted.append(spec_id)
        if next(answers) == "none":
            clock.now += 35.0
            raise httpx2.ReadTimeout("no answer", request=request)
        return httpx2.Response(200, json=_manifest_response(spec_id=spec_id))

    with use_call_budget(EXTERNAL_CALL_BUDGET):
        await _call_get_data(handler, _StubStore(held), {"spec_id": held.spec_id})
    served = await _call_get_data(handler, _StubStore(other), {"spec_id": other.spec_id})
    unanswered = await _call_get_data(handler, _StubStore(other), {"spec_id": other.spec_id})
    clock.now = 125.0
    with use_call_budget(EXTERNAL_CALL_BUDGET):
        external = await _call_get_data(handler, _StubStore(other), {"spec_id": other.spec_id})

    assert served.is_error is False
    assert unanswered.structured_content is not None
    assert unanswered.structured_content["retry_after_s"] == 5.0
    assert external.is_error is False
    assert posted == [held.spec_id, other.spec_id, other.spec_id, other.spec_id]


async def test_a_mint_with_no_id_token_refuses_unavailable_and_frees_its_slot() -> None:
    """A token that cannot be minted means the POST was never sent, so the external
    slot is freed at once and the next external mint is admitted."""
    failed, other = _stored_definition(), _stored_definition(shape_id="OtherShape")
    posted: list[str] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        spec_id = json.loads(request.content)["input"]["spec_id"]
        posted.append(spec_id)
        return httpx2.Response(200, json=_manifest_response(spec_id=spec_id))

    with use_call_budget(EXTERNAL_CALL_BUDGET):
        refused = await _call_get_data(
            handler, _StubStore(failed), {"spec_id": failed.spec_id}, auth=id_token_auth(mints=0)
        )
        admitted = await _call_get_data(handler, _StubStore(other), {"spec_id": other.spec_id})

    assert refused.structured_content is not None
    assert refused.structured_content["code"] == "service_unavailable"
    assert refused.structured_content["retry_after_s"] == 5.0
    assert admitted.is_error is False
    assert posted == [other.spec_id]


def _statistics_manifest() -> Manifest:
    return strip_manifest(_manifest_response_with_statistics(spec_id="spec_" + "a" * 32))


async def test_statistics_below_the_floor_are_skipped_and_degraded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[object] = []

    async def fetch_statistics(**kwargs: Any) -> list[SeriesStatisticsModel]:
        calls.append(kwargs)
        return []

    monkeypatch.setattr(get_data_tool, "fetch_series_statistics", fetch_statistics)
    clock = _FakeClock()

    result = await get_data_tool._fetch_statistics_for_digest(
        _statistics_manifest(), deadline=CallDeadline.after(7.9, clock=clock)
    )

    assert result == (None, True)
    assert calls == []


async def test_statistics_run_with_the_reserve_held_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen_remaining: list[float] = []

    async def fetch_statistics(*, deadline: CallDeadline, **_: Any) -> list[SeriesStatisticsModel]:
        seen_remaining.append(deadline.remaining())
        return []

    monkeypatch.setattr(get_data_tool, "fetch_series_statistics", fetch_statistics)
    clock = _FakeClock()

    result = await get_data_tool._fetch_statistics_for_digest(
        _statistics_manifest(), deadline=CallDeadline.after(8.0, clock=clock)
    )

    assert result == ([], False)
    assert seen_remaining == [5.0]


async def test_a_statistics_fetch_that_outruns_its_deadline_degrades(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fetch_statistics(**_: Any) -> list[SeriesStatisticsModel]:
        raise TimeoutError

    monkeypatch.setattr(get_data_tool, "fetch_series_statistics", fetch_statistics)

    result = await get_data_tool._fetch_statistics_for_digest(
        _statistics_manifest(), deadline=CallDeadline.after(60.0)
    )

    assert result == (None, True)


@pytest.mark.parametrize(
    ("continuations", "continuing"),
    [
        pytest.param([], None, id="nothing-continued"),
        pytest.param(
            [{"axis": "measurementMethod", "method": "Estimated", "periods": ["2025", "2026"]}],
            "Estimated",
            id="continued",
        ),
    ],
)
async def test_statistics_fetch_passes_the_manifests_series_count_and_continuation(
    monkeypatch: pytest.MonkeyPatch, continuations: list[dict[str, Any]], continuing: str | None
) -> None:
    seen: list[tuple[int, str | None]] = []

    async def fetch_statistics(
        *, expected_series: int, continuing: str | None, **_: Any
    ) -> list[SeriesStatisticsModel]:
        seen.append((expected_series, continuing))
        return []

    monkeypatch.setattr(get_data_tool, "fetch_series_statistics", fetch_statistics)
    manifest = _statistics_manifest()
    manifest = manifest.model_copy(
        update={"trace": {**manifest.trace, "continuations": continuations}}
    )

    result = await get_data_tool._fetch_statistics_for_digest(
        manifest, deadline=CallDeadline.after(60.0)
    )

    assert result == ([], False)
    assert seen == [(manifest.series_count, continuing)]


async def test_a_row_count_mismatch_degrades_the_statistics_line(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fetch_statistics(**_: Any) -> list[SeriesStatisticsModel]:
        raise SeriesStatisticsSchemaError("decoded 0 row(s), but the manifest named 1")

    monkeypatch.setattr(get_data_tool, "fetch_series_statistics", fetch_statistics)

    result = await get_data_tool._fetch_statistics_for_digest(
        _statistics_manifest(), deadline=CallDeadline.after(60.0)
    )

    assert result == (None, True)


# Summed retrieval: `stat_vars="sum"` forwards a groupby, and the summed frame keys apart


_SUM_GROUPBY = {"by": ["time_period"], "op": "sum"}


def _summed_manifest_response(
    *,
    spec_id: str,
    content_version: str,
    artifact_digest: str = "sha256:" + "1" * 64,
    contributing_member_count: int | None = None,
    stat_var_label: str | None = None,
) -> dict[str, Any]:
    """A summed mint's manifest: the groupby rendering, a trace listing two StatVars with
    rows and one without, and, for a version stored before descriptions
    (`contributing_member_count` set), apis' `stat_vars_summed` warning."""
    response = _manifest_response(
        spec_id=spec_id, content_version=content_version, artifact_digest=artifact_digest
    )
    response["trace"] = {
        "resolved_stat_vars": [
            {"ref": "dcid:Grants_Primary", "observation_count": 4},
            {"ref": "dcid:Grants_Secondary", "observation_count": 3},
            {"ref": "dcid:Grants_Tertiary", "observation_count": 0},
        ]
    }
    response["artifacts"][0]["rendering"] = {
        "bundle": False,
        "format": "parquet",
        "layout": "frame",
        "groupby": _SUM_GROUPBY,
    }
    if contributing_member_count is not None:
        response["warnings"] = [
            {
                "code": "stat_vars_summed",
                "severity": "info",
                "message": (
                    "The series sums observations from more than one StatVar in the requested "
                    f"family. {contributing_member_count} StatVars were summed into "
                    f"dcid:{spec_id}."
                ),
                "contributing_member_count": contributing_member_count,
                "stat_var_label": stat_var_label,
            }
        ]
    return response


async def test_a_summed_call_forwards_the_groupby_in_the_mint_body() -> None:
    stored = _stored_definition()
    requests: list[dict[str, Any]] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        requests.append(json.loads(request.content))
        return httpx2.Response(
            200, json=_summed_manifest_response(spec_id=stored.spec_id, content_version="cv_2")
        )

    result = await _call_get_data(
        handler, _StubStore(stored), {"spec_id": stored.spec_id, "stat_vars": "sum"}
    )

    assert result.is_error is False
    assert requests == [
        {
            "schema_version": 1,
            "input": {
                "kind": "spec_id",
                "spec_id": stored.spec_id,
                "definition": {
                    "schema_version": 1,
                    "spec_id": stored.spec_id,
                    "identity": json.loads(stored.canonical_bytes),
                },
                "scope": {},
            },
            "options": {
                "format": "parquet",
                "bundle": False,
                "layout": "frame",
                "groupby": _SUM_GROUPBY,
            },
        }
    ]


async def test_a_separate_call_sends_no_groupby() -> None:
    stored = _stored_definition()
    options: list[dict[str, Any]] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        options.append(json.loads(request.content)["options"])
        return httpx2.Response(200, json=_manifest_response(spec_id=stored.spec_id))

    result = await _call_get_data(
        handler, _StubStore(stored), {"spec_id": stored.spec_id, "stat_vars": "separate"}
    )

    assert result.is_error is False
    assert options == [{"format": "parquet", "bundle": False, "layout": "frame"}]


async def test_one_mint_posts_the_sum_option() -> None:
    """A described identity is minted under its own spec_id, with the summing option on
    the same request."""
    stored = _described_stored_definition()
    requests: list[dict[str, Any]] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        assert request.url.path == "/api/datasets"
        requests.append(json.loads(request.content))
        return httpx2.Response(200, json=_manifest_response(spec_id=stored.spec_id))

    result = await _call_get_data(
        handler,
        _RecordingStore(stored),
        {
            "spec_id": stored.spec_id,
            "stat_vars": "sum",
        },
    )

    assert result.is_error is False
    assert len(requests) == 1
    assert requests[0]["input"]["kind"] == "spec_id"
    assert requests[0]["input"]["spec_id"] == stored.spec_id
    assert "resolved_stat_var_dcids" not in requests[0]["input"]
    assert requests[0]["options"] == {
        "format": "parquet",
        "bundle": False,
        "layout": "frame",
        "groupby": {"by": ["time_period"], "op": "sum"},
    }
    assert result.structured_content is not None
    assert result.structured_content["definition"]["spec_id"] == stored.spec_id
    assert "substitution" not in result.structured_content


async def test_an_unsummed_result_reports_no_summed_count() -> None:
    stored = _stored_definition()
    handler = _mint_only_handler(200, _manifest_response(spec_id=stored.spec_id))

    result = await _call_get_data(handler, _StubStore(stored), {"spec_id": stored.spec_id})

    assert result.is_error is False
    assert result.structured_content is not None
    assert result.structured_content["summed_stat_var_count"] is None
    assert result.structured_content["summed_stat_var_label"] is None


async def test_a_summed_call_dfo_apis_refuses_returns_rollup_not_summable() -> None:
    stored = _stored_definition()
    detail = (
        "groupby axis 'stat_var' refused: StatVar(s) Amount_Gross, Amount_Net decode to the "
        "same values across ('crsPurpose',) but are distinct measures"
    )

    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(
            422,
            content=_problem_body(
                code="rollup_not_summable",
                retryable=False,
                detail=detail,
                pointer="/options/groupby",
                issues=[{"pointer": "/options/groupby", "axis": "stat_var"}],
            ),
        )

    result = await _call_get_data(
        handler, _StubStore(stored), {"spec_id": stored.spec_id, "stat_vars": "sum"}
    )

    assert result.is_error is True
    assert result.structured_content is not None
    assert result.structured_content["code"] == "rollup_not_summable"
    assert result.structured_content["retry_mode"] == "do_not_retry"
    remedy = "Ask about one of the series it combines instead of the total."
    assert result.structured_content["fix"] == {
        "axes": ["stat_var"],
        "detail": detail,
        "remedy": remedy,
    }
    assert text_of(result).split("\n\n")[0] == (
        "The series in this dataset can't be added into one total: a total could count "
        f"the same amount twice or merge different measures. {remedy} Repeating the call "
        'refuses the same way. A get_data call with stat_vars "separate" returns the series '
        "one by one."
    )


async def test_a_summed_operation_dfo_apis_refuses_returns_rollup_not_summable() -> None:
    stored = _stored_definition()

    def handler(request: httpx2.Request) -> httpx2.Response:
        if request.url.path == "/api/datasets":
            return httpx2.Response(
                202,
                json={
                    "operation": {
                        "operation_id": _OPERATION_ID,
                        "status": "pending",
                        "poll": _OPERATION_POLL_PATH,
                        "retry_after_ms": 1,
                    }
                },
            )
        if request.url.path == _OPERATION_POLL_PATH:
            return httpx2.Response(
                500,
                content=_problem_body(
                    code="operation_failed",
                    retryable=False,
                    detail=f"operation {_OPERATION_ID!r} did not complete successfully: "
                    "rollup_not_summable",
                    cause={"code": "rollup_not_summable"},
                ),
            )
        raise AssertionError(f"unexpected call to {request.url.path}")

    result = await _call_get_data(
        handler, _StubStore(stored), {"spec_id": stored.spec_id, "stat_vars": "sum"}
    )

    assert result.is_error is True
    assert result.structured_content is not None
    assert result.structured_content["code"] == "rollup_not_summable"
    assert result.structured_content["retry_mode"] == "do_not_retry"
    assert result.structured_content["fix"]["remedy"] == (
        "Ask about one of the series it combines instead of the total."
    )


# Breakdown rolls a code axis up to one level, gated on the capabilities document


_SECTOR_BREAKDOWN = {"code_system": "crsPurpose", "level": 0}
# The `options.groupby` `_SECTOR_BREAKDOWN` mints with.
_SECTOR_BREAKDOWN_GROUPBY = {
    "by": ["crsPurpose", "time_period"],
    "op": "sum",
    "levels": {"crsPurpose": 0},
}


def _breakdown_handler(
    capabilities: dict[str, Any], posted: list[dict[str, Any]], *, spec_id: str
) -> Handler:
    """Answer the capabilities read with `capabilities` and record each mint body."""

    def handler(request: httpx2.Request) -> httpx2.Response:
        if request.url.path == "/api/datasets/capabilities":
            return httpx2.Response(200, json=capabilities)
        if request.url.path == "/api/datasets":
            posted.append(json.loads(request.content))
            return httpx2.Response(200, json=_manifest_response(spec_id=spec_id))
        raise AssertionError(f"unexpected call to {request.url.path}")

    return handler


async def test_an_advertised_breakdown_posts_a_levels_groupby_and_describes_the_frame() -> None:
    stored = _stored_definition()
    posted: list[dict[str, Any]] = []
    capabilities = {"capability_version": "v1", "groupby_level_axes": ["crsPurpose"]}

    result = await _call_get_data(
        _breakdown_handler(capabilities, posted, spec_id=stored.spec_id),
        _StubStore(stored),
        {"spec_id": stored.spec_id, "breakdown": _SECTOR_BREAKDOWN},
    )

    assert result.is_error is False
    assert [body["options"] for body in posted] == [
        {
            "format": "parquet",
            "bundle": False,
            "layout": "frame",
            "groupby": {
                "by": ["crsPurpose", "time_period"],
                "op": "sum",
                "levels": {"crsPurpose": 0},
            },
        }
    ]
    assert "stat_var_crsPurpose" in text_of(result)


@pytest.mark.parametrize(
    ("capabilities", "arguments", "code", "fix"),
    [
        pytest.param(
            {"capability_version": "v1", "summable_axes": ["crsPurpose"]},
            {"breakdown": _SECTOR_BREAKDOWN},
            "capability_unavailable",
            {"option": "breakdown"},
            id="not-advertised",
        ),
        pytest.param(
            {"capability_version": "v1", "groupby_level_axes": ["crsPurpose"]},
            {"breakdown": {"code_system": "crsTypeOfFlow", "level": 0}},
            "request_invalid",
            {
                "problems": [
                    "breakdown.code_system: 'crsTypeOfFlow' cannot be rolled up to a level; this deployment rolls up crsPurpose"
                ]
            },
            id="code-system-not-rolled-up",
        ),
        pytest.param(
            {"capability_version": "v1", "groupby_level_axes": ["crsPurpose"]},
            {"breakdown": _SECTOR_BREAKDOWN, "stat_vars": "sum"},
            "request_invalid",
            {
                "problems": [
                    'stat_vars: "sum" totals the whole family into one series, which leaves no code to break it down by; use "separate" with breakdown'
                ]
            },
            id="beside-sum",
        ),
    ],
)
async def test_a_breakdown_the_deployment_cannot_serve_refuses_before_any_mint(
    capabilities: dict[str, Any],
    arguments: dict[str, object],
    code: str,
    fix: dict[str, str],
) -> None:
    stored = _stored_definition()
    posted: list[dict[str, Any]] = []

    result = await _call_get_data(
        _breakdown_handler(capabilities, posted, spec_id=stored.spec_id),
        _StubStore(stored),
        {"spec_id": stored.spec_id, **arguments},
    )

    assert result.is_error is True
    assert result.structured_content is not None
    assert result.structured_content["code"] == code
    assert fix.items() <= result.structured_content["fix"].items()
    assert posted == []


def _frame_parquet(stat_vars: list[str]) -> bytes:
    """One row per StatVar, so a frame's distinct `stat_var` values tell it apart, in
    the canonical layout a version's canonical frame is in."""
    frame = Frame.build(
        role_keys=["observationAbout"],
        rows=[
            {
                "stat_var": stat_var,
                "observationAbout": "country/KEN",
                "time_period": "2020",
                "value": 1.5,
            }
            for stat_var in stat_vars
        ],
    )
    return frame_to_parquet_bytes(frame)


async def _call_get_data_in_one_process(
    handler: Handler, store: Any, calls: list[dict[str, object]]
) -> list[CallToolResult]:
    """Every call in `calls`, in order, against one data-mcp server session."""
    async with _client(handler) as upstream:
        client_holder.set_client(upstream)
        async with Client(mcp) as client:
            set_store(store)
            return [
                await client.call_tool("get_data", arguments, raise_on_error=False)
                for arguments in calls
            ]


class _RawAndSummedMint:
    """apis for one spec: the raw frame under one `content_version`, and the summed
    frame under its own unless a single StatVar contributed (`sole_contributor`), when
    apis serves the raw frame's own manifest. Each version's canonical frame is its
    bytes. Records every artifact fetch."""

    def __init__(self, *, spec_id: str, sole_contributor: bool = False) -> None:
        self.spec_id = spec_id
        self.sole_contributor = sole_contributor
        raw = (
            ["dcid:Grants_Primary"]
            if sole_contributor
            else ["dcid:Grants_Primary", "dcid:Grants_Secondary"]
        )
        raw_bytes = _frame_parquet(raw)
        summed_bytes = _frame_parquet([f"dcid:{spec_id}"])
        self.raw_cv = _content_version_of(raw_bytes)
        self.summed_cv = _content_version_of(summed_bytes)
        self.bytes_by_cv = {self.raw_cv: raw_bytes, self.summed_cv: summed_bytes}
        self.fetched: list[str] = []

    def _digest(self, content_version: str) -> str:
        return f"sha256:{hashlib.sha256(self.bytes_by_cv[content_version]).hexdigest()}"

    def handler(self, request: httpx2.Request) -> httpx2.Response:
        assert request.url.path == "/api/datasets"
        summed = "groupby" in json.loads(request.content)["options"]
        if summed and not self.sole_contributor:
            return httpx2.Response(
                200,
                json=_summed_manifest_response(
                    spec_id=self.spec_id,
                    content_version=self.summed_cv,
                    artifact_digest=self._digest(self.summed_cv),
                ),
            )
        response = _manifest_response(
            spec_id=self.spec_id,
            content_version=self.raw_cv,
            artifact_digest=self._digest(self.raw_cv),
        )
        # A sum one StatVar contributes to keeps the raw version, under its own rendering.
        response["artifacts"][0]["rendering"] = json.loads(request.content)["options"]
        return httpx2.Response(200, json=response)

    async def fetch(self, *, content_version: str, **_kwargs: Any) -> bytes:
        self.fetched.append(content_version)
        return self.bytes_by_cv[content_version]


async def _distinct_stat_vars(frame_store: InMemoryMaterialisationStore, ref: FrameRef) -> set[str]:
    stored_frame = await frame_store.get(ref)
    assert stored_frame is not None
    frame, _trace = stored_frame
    return set(frame.table.column("stat_var").to_pylist())


@pytest.mark.parametrize(
    "order", [("separate", "sum"), ("sum", "separate")], ids=["raw-then-summed", "summed-then-raw"]
)
async def test_raw_and_summed_calls_for_one_spec_register_two_distinct_frames(
    monkeypatch: pytest.MonkeyPatch, order: tuple[str, str]
) -> None:
    """The frame store is insert-only and keyed on `{spec_id, content_version}`: the
    raw frame is the raw version's canonical frame under its pair, and the summed
    frame, which apis minted as a version of its own, is held under the pair
    naming its groupby, whichever call comes first. The summed frame's family ref
    marks it summed, with no warning or description."""
    monkeypatch.undo()  # the real `materialise_and_register`
    stored = _stored_definition()
    upstream = _RawAndSummedMint(spec_id=stored.spec_id)
    monkeypatch.setattr("data_mcp.frame_materialisation._fetch_artifact_bytes", upstream.fetch)
    _serve_canonical_frames(monkeypatch, upstream.bytes_by_cv)
    frame_store = InMemoryMaterialisationStore()
    set_frame_store(frame_store)

    results = await _call_get_data_in_one_process(
        upstream.handler,
        _StubStore(stored),
        [{"spec_id": stored.spec_id, "stat_vars": reading} for reading in order],
    )

    assert [r.is_error for r in results] == [False, False]
    cv_by_reading = {
        reading: result.structured_content["content_version"]
        for reading, result in zip(order, results, strict=True)
        if result.structured_content is not None
    }
    assert cv_by_reading == {"separate": upstream.raw_cv, "sum": upstream.summed_cv}
    summed_counts = {
        reading: result.structured_content["summed_stat_var_count"]
        for reading, result in zip(order, results, strict=True)
        if result.structured_content is not None
    }
    assert summed_counts == {"separate": None, "sum": 2}
    assert sorted(upstream.fetched) == sorted([upstream.raw_cv, upstream.summed_cv])
    raw_ref = FrameRef(spec_id=stored.spec_id, content_version=upstream.raw_cv)
    summed_ref = frame_ref_for(
        spec_id=stored.spec_id, content_version=upstream.summed_cv, groupby=_SUM_GROUPBY
    )
    summed_content = results[order.index("sum")].structured_content
    assert summed_content is not None
    assert summed_content["frame_ref"] == summed_ref.model_dump(mode="json")
    assert (
        await frame_store.get(FrameRef(spec_id=stored.spec_id, content_version=upstream.summed_cv))
        is None
    )
    assert await _distinct_stat_vars(frame_store, raw_ref) == {
        "dcid:Grants_Primary",
        "dcid:Grants_Secondary",
    }
    assert await _distinct_stat_vars(frame_store, summed_ref) == {f"dcid:{stored.spec_id}"}


def _breakdown_frame_parquet(*, spec_id: str, sectors: list[str]) -> bytes:
    """apis' frame for a `groupby.levels` roll-up: one row per sector, every row's
    `stat_var` the family ref, and `stat_var_crsPurpose` the sector's label."""
    table = pa.table(
        {
            "stat_var": pa.array([f"dcid:{spec_id}"] * len(sectors), type=pa.string()),
            "observationAbout": pa.array(["country/KEN"] * len(sectors), type=pa.string()),
            "stat_var_crsPurpose": pa.array(sectors, type=pa.string()),
            "time_period": pa.array(["2020"] * len(sectors), type=pa.string()),
            "value": pa.array([1.5] * len(sectors), type=pa.float64()),
        }
    )
    buf = io.BytesIO()
    pq.write_table(table, buf)
    return buf.getvalue()


async def test_a_breakdown_frame_registers_under_its_own_pair_with_the_code_column(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A breakdown frame reaches the store under the pair naming the roll-up's groupby
    and the `content_version` apis minted for it, with `stat_var_crsPurpose` as a
    role column and one StatVar."""
    monkeypatch.undo()  # the real `materialise_and_register`
    stored = _stored_definition()
    raw = _RawAndSummedMint(spec_id=stored.spec_id)
    breakdown_cv = "cv_" + "c" * 32
    breakdown_bytes = _breakdown_frame_parquet(
        spec_id=stored.spec_id,
        sectors=["Social infrastructure and services", "Humanitarian aid"],
    )
    breakdown_digest = f"sha256:{hashlib.sha256(breakdown_bytes).hexdigest()}"

    def handler(request: httpx2.Request) -> httpx2.Response:
        if request.url.path == "/api/datasets/capabilities":
            return httpx2.Response(200, json={"groupby_level_axes": ["crsPurpose"]})
        options = json.loads(request.content)["options"]
        if "groupby" not in options:
            return raw.handler(request)
        response = _manifest_response(
            spec_id=stored.spec_id, content_version=breakdown_cv, artifact_digest=breakdown_digest
        )
        response["artifacts"][0]["rendering"] = options
        return httpx2.Response(200, json=response)

    async def fetch(*, content_version: str, **_kwargs: Any) -> bytes:
        if content_version == breakdown_cv:
            return breakdown_bytes
        return await raw.fetch(content_version=content_version)

    monkeypatch.setattr("data_mcp.frame_materialisation._fetch_artifact_bytes", fetch)
    _serve_canonical_frames(monkeypatch, raw.bytes_by_cv)
    frame_store = InMemoryMaterialisationStore()
    set_frame_store(frame_store)

    separate, broken_down = await _call_get_data_in_one_process(
        handler,
        _StubStore(stored),
        [
            {"spec_id": stored.spec_id},
            {"spec_id": stored.spec_id, "breakdown": _SECTOR_BREAKDOWN},
        ],
    )

    assert separate.is_error is False
    assert broken_down.is_error is False
    content = broken_down.structured_content
    assert content is not None
    ref = frame_ref_for(
        spec_id=stored.spec_id,
        content_version=breakdown_cv,
        groupby=_SECTOR_BREAKDOWN_GROUPBY,
    )
    assert content["frame_ref"] == ref.model_dump(mode="json")
    assert content["role_keys"] == ["observationAbout", "stat_var_crsPurpose"]
    assert content["stat_var_count"] == 1
    assert await _distinct_stat_vars(frame_store, ref) == {f"dcid:{stored.spec_id}"}


async def test_a_sole_contributor_summed_call_returns_the_raw_frame_without_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With one StatVar holding rows, apis answers a summed request with the raw
    frame's own `content_version` and no `stat_vars_summed` warning. The sum keeps the
    raw version, so its rendering is held under the pair naming its groupby, apart
    from the raw version's canonical frame."""
    monkeypatch.undo()  # the real `materialise_and_register`
    stored = _stored_definition()
    upstream = _RawAndSummedMint(spec_id=stored.spec_id, sole_contributor=True)
    monkeypatch.setattr("data_mcp.frame_materialisation._fetch_artifact_bytes", upstream.fetch)
    _serve_canonical_frames(monkeypatch, upstream.bytes_by_cv)
    frame_store = InMemoryMaterialisationStore()
    set_frame_store(frame_store)

    raw, summed = await _call_get_data_in_one_process(
        upstream.handler,
        _StubStore(stored),
        [{"spec_id": stored.spec_id}, {"spec_id": stored.spec_id, "stat_vars": "sum"}],
    )

    assert raw.is_error is False
    assert summed.is_error is False
    assert summed.structured_content is not None
    assert summed.structured_content["content_version"] == upstream.raw_cv
    assert summed.structured_content["summed_stat_var_count"] is None
    assert upstream.fetched == [upstream.raw_cv, upstream.raw_cv]
    raw_ref = FrameRef(spec_id=stored.spec_id, content_version=upstream.raw_cv)
    summed_ref = frame_ref_for(
        spec_id=stored.spec_id, content_version=upstream.raw_cv, groupby=_SUM_GROUPBY
    )
    assert summed.structured_content["frame_ref"] == summed_ref.model_dump(mode="json")
    assert summed_ref != raw_ref
    assert await _distinct_stat_vars(frame_store, raw_ref) == {"dcid:Grants_Primary"}
    assert await _distinct_stat_vars(frame_store, summed_ref) == {"dcid:Grants_Primary"}


# Place types: a spec that names no place asks for each place's types.


@pytest.mark.parametrize(
    ("stat_vars", "groupby"),
    [("separate", None), ("sum", {"by": ["time_period"], "op": "sum"})],
)
async def test_place_types_asks_for_the_type_column_alone(
    stat_vars: str, groupby: dict[str, Any] | None
) -> None:
    # A list naming only an enrichment adds it to every default column, so the
    # decoded `stat_var_<code system>` columns stay.
    stored = _stored_definition()
    options: list[dict[str, Any]] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        options.append(json.loads(request.content)["options"])
        return httpx2.Response(200, json=_manifest_response(spec_id=stored.spec_id))

    result = await _call_get_data(
        handler,
        _StubStore(stored),
        {"spec_id": stored.spec_id, "stat_vars": stat_vars, "place_types": True},
    )

    assert result.is_error is False
    assert options == [
        {
            "format": "parquet",
            "bundle": False,
            "layout": "frame",
            **({} if groupby is None else {"groupby": groupby}),
            "columns": ["observationAbout_type"],
        }
    ]


async def test_place_types_names_every_party_kind_a_flow_frame_can_hold() -> None:
    """A kind the description leaves out is one an outside model meets as a bare token."""
    async with Client(mcp) as client:
        (tool,) = [tool for tool in await client.list_tools() if tool.name == "get_data"]

    description = tool.input_schema["properties"]["place_types"]["description"]
    assert {kind for kind in PARTY_KINDS if kind not in description} == set()


async def test_a_call_without_place_types_sends_no_columns() -> None:
    stored = _stored_definition()
    options: list[dict[str, Any]] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        options.append(json.loads(request.content)["options"])
        return httpx2.Response(200, json=_manifest_response(spec_id=stored.spec_id))

    result = await _call_get_data(handler, _StubStore(stored), {"spec_id": stored.spec_id})

    assert result.is_error is False
    assert "columns" not in options[0]


def _column_unknown(*, served: list[str], truncated: bool = False) -> httpx2.Response:
    """apis' 422 for an `options.columns` token the table lacks, listing what it serves."""
    return httpx2.Response(
        422,
        json={
            "type": "/codes/column_unknown",
            "title": "Column Unknown",
            "status": 422,
            "detail": "column 'financingRecipient_type' is not present on this table",
            "code": "column_unknown",
            "pointer": "/options/columns",
            "alternatives": [{"ref": ref, "label": ref} for ref in served],
            "truncated": truncated,
            "retryable": False,
        },
    )


def _failed_operation_column_unknown(
    *,
    served: list[str],
    pointer: str = "/options/columns",
    truncated: bool = False,
) -> httpx2.Response:
    """An operation failure that records apis' column refusal as its cause."""
    return httpx2.Response(
        500,
        json={
            "type": "/codes/operation_failed",
            "status": 500,
            "code": "operation_failed",
            "pointer": pointer,
            "alternatives": [{"ref": ref, "label": ref} for ref in served],
            "truncated": truncated,
            "retryable": False,
            "cause": {"code": "column_unknown"},
        },
    )


async def test_place_types_a_table_without_the_place_column_mints_the_default_columns(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="data_mcp.get_data_tool")
    stored = _stored_definition(entities={"recipient": ["country/KEN"]})
    options: list[dict[str, Any]] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        options.append(json.loads(request.content)["options"])
        if len(options) == 1:
            return _column_unknown(served=["stat_var", "entity", "entity_type", "value"])
        return httpx2.Response(200, json=_manifest_response(spec_id=stored.spec_id))

    result = await _call_get_data(
        handler, _StubStore(stored), {"spec_id": stored.spec_id, "place_types": True}
    )

    assert result.is_error is False, result.structured_content
    assert [sent.get("columns") for sent in options] == [["observationAbout_type"], None]


@pytest.mark.parametrize("queued", [False, True], ids=["direct", "terminal-operation"])
@pytest.mark.parametrize(
    ("pointer", "alternatives"),
    [
        ("/input/spec_id", [{"ref": "stat_var", "label": "stat_var"}]),
        ("/options/columns", [{"label": "stat_var"}]),
    ],
    ids=["wrong-pointer", "malformed-alternatives"],
)
async def test_an_ineligible_nonflow_place_type_refusal_is_relayed_without_a_retry(
    queued: bool, pointer: str, alternatives: list[dict[str, str]]
) -> None:
    stored = _stored_definition(entities={"recipient": ["country/KEN"]})
    posts: list[httpx2.Request] = []

    def refusal() -> httpx2.Response:
        if queued:
            return httpx2.Response(
                500,
                json={
                    "type": "/codes/operation_failed",
                    "status": 500,
                    "code": "operation_failed",
                    "pointer": pointer,
                    "alternatives": alternatives,
                    "truncated": False,
                    "retryable": False,
                    "cause": {"code": "column_unknown"},
                },
            )
        return httpx2.Response(
            422,
            json={
                "type": "/codes/column_unknown",
                "status": 422,
                "code": "column_unknown",
                "pointer": pointer,
                "alternatives": alternatives,
                "truncated": False,
                "retryable": False,
            },
        )

    def handler(request: httpx2.Request) -> httpx2.Response:
        if request.url.path == "/api/datasets":
            posts.append(request)
            if not queued:
                return refusal()
            return httpx2.Response(
                202,
                json={
                    "operation": {
                        "operation_id": _OPERATION_ID,
                        "status": "failed",
                        "poll": _OPERATION_POLL_PATH,
                    }
                },
            )
        assert request.url.path == _OPERATION_POLL_PATH
        return refusal()

    result = await _call_get_data(
        handler, _StubStore(stored), {"spec_id": stored.spec_id, "place_types": True}
    )

    assert result.is_error is True
    assert result.structured_content is not None
    assert result.structured_content["upstream_code"] == "column_unknown"
    assert len(posts) == 1


async def test_a_polled_column_refusal_retries_with_the_flow_metadata_the_table_serves() -> None:
    stored = _stored_definition(entities={"origin": ["country/FRA"]})
    options: list[dict[str, Any]] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        if request.url.path == "/api/datasets":
            options.append(json.loads(request.content)["options"])
            if len(options) == 1:
                return httpx2.Response(
                    202,
                    json={
                        "operation": {
                            "operation_id": _OPERATION_ID,
                            "status": "failed",
                            "poll": _OPERATION_POLL_PATH,
                        }
                    },
                )
            return httpx2.Response(200, json=_manifest_response(spec_id=stored.spec_id))
        assert request.url.path == _OPERATION_POLL_PATH
        return _failed_operation_column_unknown(
            served=["financingProvider_type", "financingProvider_party_kind"]
        )

    result = await _call_get_data(
        handler, _StubStore(stored), {"spec_id": stored.spec_id, "place_types": True}
    )

    assert result.is_error is False, result.structured_content
    assert [sent["columns"] for sent in options] == [
        ["financingProvider_type", "financingRecipient_type", *KIND_COLUMNS],
        ["financingProvider_type", *KIND_COLUMNS],
    ]


async def test_a_second_mixed_column_refusal_does_not_submit_a_third_mint() -> None:
    stored = _stored_definition(entities={"origin": ["country/FRA"]})
    posts: list[dict[str, Any]] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        if request.url.path == "/api/datasets":
            posts.append(json.loads(request.content)["options"])
            if len(posts) == 1:
                return _column_unknown(
                    served=["financingProvider_type", "financingProvider_party_kind"]
                )
            return httpx2.Response(
                202,
                json={
                    "operation": {
                        "operation_id": _OPERATION_ID,
                        "status": "failed",
                        "poll": _OPERATION_POLL_PATH,
                    }
                },
            )
        assert request.url.path == _OPERATION_POLL_PATH
        return _failed_operation_column_unknown(
            served=["financingProvider_type", "financingProvider_party_kind"]
        )

    result = await _call_get_data(
        handler, _StubStore(stored), {"spec_id": stored.spec_id, "place_types": True}
    )

    assert result.is_error is True
    assert result.structured_content["upstream_code"] == "column_unknown"
    assert len(posts) == 2


@pytest.mark.parametrize(
    ("pointer", "truncated"),
    [("/input/spec_id", False), ("/options/columns", True)],
    ids=["pointer-ineligible", "truncated"],
)
async def test_an_ineligible_polled_column_refusal_is_relayed_without_a_retry(
    pointer: str, truncated: bool
) -> None:
    stored = _stored_definition(entities={"origin": ["country/FRA"]})
    posts: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        if request.url.path == "/api/datasets":
            posts.append(request)
            return httpx2.Response(
                202,
                json={
                    "operation": {
                        "operation_id": _OPERATION_ID,
                        "status": "failed",
                        "poll": _OPERATION_POLL_PATH,
                    }
                },
            )
        assert request.url.path == _OPERATION_POLL_PATH
        return _failed_operation_column_unknown(
            served=["financingProvider_type", "financingProvider_party_kind"],
            pointer=pointer,
            truncated=truncated,
        )

    result = await _call_get_data(
        handler, _StubStore(stored), {"spec_id": stored.spec_id, "place_types": True}
    )

    assert result.is_error is True
    assert result.structured_content["upstream_code"] == "column_unknown"
    assert len(posts) == 1


async def test_a_polled_column_refusal_with_malformed_alternatives_is_relayed() -> None:
    stored = _stored_definition(entities={"origin": ["country/FRA"]})
    posts: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        if request.url.path == "/api/datasets":
            posts.append(request)
            return httpx2.Response(
                202,
                json={
                    "operation": {
                        "operation_id": _OPERATION_ID,
                        "status": "failed",
                        "poll": _OPERATION_POLL_PATH,
                    }
                },
            )
        assert request.url.path == _OPERATION_POLL_PATH
        return httpx2.Response(
            500,
            json={
                "type": "/codes/operation_failed",
                "status": 500,
                "code": "operation_failed",
                "pointer": "/options/columns",
                "alternatives": [{"label": "financingProvider_type"}],
                "truncated": False,
                "retryable": False,
                "cause": {"code": "column_unknown"},
            },
        )

    result = await _call_get_data(
        handler, _StubStore(stored), {"spec_id": stored.spec_id, "place_types": True}
    )

    assert result.is_error is True
    assert result.structured_content["upstream_code"] == "column_unknown"
    assert len(posts) == 1


@pytest.mark.parametrize(
    "served",
    [
        ["stat_var", "financingProvider", "financingProvider_type", "value"],
        ["stat_var", "observationAbout_type", "value"],
    ],
    ids=["party-types-only", "place-type-only"],
)
async def test_place_types_a_flow_table_serving_no_kind_is_refused(served: list[str]) -> None:
    # A server without kinds answers `kind` with 422 `column_unknown`. A flow frame
    # without kinds would let a sum count party totals beside their parties.
    stored = _stored_definition(entities={"origin": ["country/FRA"]})
    calls: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        calls.append(request)
        return _column_unknown(served=served)

    result = await _call_get_data(
        handler, _StubStore(stored), {"spec_id": stored.spec_id, "place_types": True}
    )

    assert result.is_error is True
    assert len(calls) == 1


@pytest.mark.parametrize(
    "identity",
    [
        {"entities": {"destination": ["country/KEN"]}, "open_roles": ["origin"]},
        {"entities": {}, "open_roles": ["origin"]},
    ],
    ids=["named-and-open-axes", "dac1-open-donor-axis"],
)
async def test_place_types_a_spec_naming_a_flow_axis_asks_for_kinds(
    identity: dict[str, Any],
) -> None:
    # A bare `kind` covers every entity column the table has, including a side the
    # identity leaves unnamed, and cannot name a side the table lacks.
    stored = _stored_definition(**identity)
    options: list[dict[str, Any]] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        options.append(json.loads(request.content)["options"])
        return httpx2.Response(200, json=_manifest_response(spec_id=stored.spec_id))

    result = await _call_get_data(
        handler, _StubStore(stored), {"spec_id": stored.spec_id, "place_types": True}
    )

    assert result.is_error is False, result.structured_content
    assert [sent["columns"] for sent in options] == [
        ["financingProvider_type", "financingRecipient_type", *KIND_COLUMNS]
    ]


@pytest.mark.parametrize(
    ("entities", "served", "expected"),
    [
        (
            {"origin": ["country/FRA"]},
            ["financingProvider_type", "financingProvider_party_kind"],
            ["financingProvider_type", *KIND_COLUMNS],
        ),
        (
            {"destination": ["country/KEN"]},
            ["financingRecipient_type", "financingRecipient_party_kind"],
            ["financingRecipient_type", *KIND_COLUMNS],
        ),
    ],
    ids=["provider-only", "destination-only"],
)
async def test_place_types_retries_with_the_flow_metadata_the_table_serves(
    entities: dict[str, list[str]], served: list[str], expected: list[str]
) -> None:
    stored = _stored_definition(entities=entities)
    options: list[dict[str, Any]] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        options.append(json.loads(request.content)["options"])
        return (
            _column_unknown(served=served)
            if len(options) == 1
            else httpx2.Response(200, json=_manifest_response(spec_id=stored.spec_id))
        )

    result = await _call_get_data(
        handler, _StubStore(stored), {"spec_id": stored.spec_id, "place_types": True}
    )

    assert result.is_error is False, result.structured_content
    assert [sent["columns"] for sent in options] == [
        ["financingProvider_type", "financingRecipient_type", *KIND_COLUMNS],
        expected,
    ]


async def test_a_second_flow_metadata_refusal_is_relayed_after_one_retry() -> None:
    stored = _stored_definition(entities={"origin": ["country/FRA"]})
    calls: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        calls.append(request)
        return _column_unknown(served=["financingProvider_type", "financingProvider_party_kind"])

    result = await _call_get_data(
        handler, _StubStore(stored), {"spec_id": stored.spec_id, "place_types": True}
    )

    assert result.is_error is True
    assert result.structured_content["upstream_code"] == "column_unknown"
    assert len(calls) == 2


@pytest.mark.parametrize(
    "served",
    [
        ["financingProvider_party_kind"],
        ["financingProvider_type"],
    ],
    ids=["present-role-without-type", "missing-qualified-kind"],
)
async def test_incomplete_remaining_flow_metadata_is_relayed_without_a_retry(
    served: list[str],
) -> None:
    stored = _stored_definition(entities={"origin": ["country/FRA"]})
    calls: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        calls.append(request)
        return _column_unknown(served=served)

    result = await _call_get_data(
        handler, _StubStore(stored), {"spec_id": stored.spec_id, "place_types": True}
    )

    assert result.is_error is True
    assert len(calls) == 1


async def test_a_column_unknown_with_malformed_alternatives_is_relayed_without_a_retry() -> None:
    stored = _stored_definition(entities={"origin": ["country/FRA"]})
    calls: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        calls.append(request)
        return httpx2.Response(
            422,
            json={
                "type": "/codes/column_unknown",
                "status": 422,
                "code": "column_unknown",
                "pointer": "/options/columns",
                "alternatives": [{"label": "financingProvider_type"}],
                "truncated": False,
            },
        )

    result = await _call_get_data(
        handler, _StubStore(stored), {"spec_id": stored.spec_id, "place_types": True}
    )

    assert result.is_error is True
    assert len(calls) == 1


async def test_a_column_unknown_listing_a_truncated_set_is_refused_without_a_retry() -> None:
    stored = _stored_definition(entities={"origin": ["country/FRA"]})
    calls: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        calls.append(request)
        return _column_unknown(served=["financingProvider_type"], truncated=True)

    result = await _call_get_data(
        handler, _StubStore(stored), {"spec_id": stored.spec_id, "place_types": True}
    )

    assert result.is_error is True
    assert len(calls) == 1


class _PlainAndTypedMint:
    """apis for one place-subject spec: the default columns, or the place-type
    column added when `options.columns` names it, both under one `content_version`
    (apis keeps it across renderings), each its own artifact. Every manifest lists
    both renderings, the one not requested first, as a done operation lists every
    rendering stored under its pair. Records every artifact fetch as
    `(content_version, artifact_digest)`."""

    def __init__(self, *, spec_id: str) -> None:
        self.spec_id = spec_id
        places = ["Earth", "country/USA", "geoId/06"]
        plain = {
            "stat_var": pa.array(["dcid:Test"] * 3, type=pa.string()),
            "observationAbout": pa.array(places, type=pa.string()),
            "time_period": pa.array(["2023"] * 3, type=pa.string()),
            "value": pa.array([105.0, 27.0, 3.9], type=pa.float64()),
        }
        typed = plain | {
            "observationAbout_type": pa.array(
                ["Place", "Country", "AdministrativeArea1,State"], type=pa.string()
            )
        }
        self.bytes_by_digest: dict[str, bytes] = {}
        self.digest_by_rendering: dict[bool, str] = {}
        for is_typed, columns in ((False, plain), (True, typed)):
            buf = io.BytesIO()
            pq.write_table(pa.table(columns), buf)
            digest = f"sha256:{hashlib.sha256(buf.getvalue()).hexdigest()}"
            self.bytes_by_digest[digest] = buf.getvalue()
            self.digest_by_rendering[is_typed] = digest
        # The version's canonical frame: the plain rows in the canonical layout.
        self.canonical = frame_to_parquet_bytes(
            Frame.build(
                role_keys=["observationAbout"],
                rows=[
                    {
                        "stat_var": "dcid:Test",
                        "observationAbout": place,
                        "time_period": "2023",
                        "value": value,
                    }
                    for place, value in zip(places, [105.0, 27.0, 3.9], strict=True)
                ],
            )
        )
        self.bytes_by_digest[_digest(self.canonical)] = self.canonical
        self.content_version = _content_version_of(self.canonical)
        self.fetched: list[tuple[str, str]] = []

    def handler(self, request: httpx2.Request) -> httpx2.Response:
        assert request.url.path == "/api/datasets"
        is_typed = "columns" in json.loads(request.content)["options"]
        response = _manifest_response(spec_id=self.spec_id, content_version=self.content_version)
        plain = response["artifacts"][0]
        renderings = {
            False: plain["rendering"],
            True: plain["rendering"] | {"columns": ["observationAbout_type"]},
        }
        response["artifacts"] = [
            plain
            | {"artifact_digest": self.digest_by_rendering[typed], "rendering": renderings[typed]}
            for typed in (not is_typed, is_typed)
        ]
        return httpx2.Response(200, json=response)

    async def fetch(self, *, content_version: str, artifact_digest: str, **_: Any) -> bytes:
        self.fetched.append((content_version, artifact_digest))
        return self.bytes_by_digest[artifact_digest]


async def _mint_recorded_root(
    handler: Handler, stored: StoredDefinition, record: MintRecord
) -> RootMint | QueuedOperation | ToolResult:
    """`mint_recorded_root` for `record` against `handler` as apis."""
    async with _client(handler) as upstream:
        client_holder.set_client(upstream)
        set_store(_StubStore(stored))
        return await get_data_tool.mint_recorded_root(record, deadline=CallDeadline.after(165.0))


async def test_a_recorded_root_is_minted_again_with_its_options_and_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.undo()  # the real `materialise_and_register`
    stored = _stored_definition()
    upstream = _PlainAndTypedMint(spec_id=stored.spec_id)
    monkeypatch.setattr("data_mcp.frame_materialisation._fetch_artifact_bytes", upstream.fetch)
    frame_store = InMemoryMaterialisationStore()
    set_frame_store(frame_store)
    sent: list[dict[str, Any]] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        sent.append(json.loads(request.content))
        return upstream.handler(request)

    options = {
        "format": "parquet",
        "bundle": False,
        "layout": "frame",
        "columns": ["observationAbout_type"],
    }
    scope = {"window": {"kind": "all"}}
    older = FrameRef(spec_id=stored.spec_id, content_version="cv_" + "1" * 32)
    record = MintRecord(frame=older, dataset=older, options=options, scope=scope)

    minted = await _mint_recorded_root(handler, stored, record)

    assert isinstance(minted, RootMint)
    (body,) = sent
    assert body["options"] == options
    assert body["input"]["spec_id"] == stored.spec_id
    assert body["input"]["scope"] == scope
    dataset = FrameRef(spec_id=stored.spec_id, content_version=upstream.content_version)
    assert minted.dataset == dataset
    assert minted.frame == frame_ref_for(
        spec_id=stored.spec_id,
        content_version=upstream.content_version,
        columns=("observationAbout_type",),
    )
    assert minted.options == options
    assert minted.stat_var_count == 1
    assert await frame_store.get(minted.frame) is not None
    assert await frame_store.get_mint_record(minted.frame) == MintRecord(
        frame=minted.frame, dataset=dataset, options=options, scope=scope, stat_var_count=1
    )


async def test_a_queued_root_mint_returns_at_once_and_a_repeat_rejoins_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.undo()  # the real `materialise_and_register`
    stored = _stored_definition()
    upstream = _PlainAndTypedMint(spec_id=stored.spec_id)
    monkeypatch.setattr("data_mcp.frame_materialisation._fetch_artifact_bytes", upstream.fetch)
    _serve_canonical_frames(monkeypatch, {upstream.content_version: upstream.canonical})
    set_frame_store(InMemoryMaterialisationStore())
    submitted: list[bytes] = []
    answers = [
        httpx2.Response(
            202,
            json={
                "operation": {
                    "operation_id": _OPERATION_ID,
                    "status": "pending",
                    "retry_after_ms": 1500,
                }
            },
        )
    ]

    def handler(request: httpx2.Request) -> httpx2.Response:
        # A poll of the operation would be a wait, and is refused here.
        assert request.url.path == "/api/datasets"
        submitted.append(request.content)
        return answers.pop(0) if answers else upstream.handler(request)

    root = FrameRef(spec_id=stored.spec_id, content_version="cv_" + "1" * 32)
    record = MintRecord(
        frame=root,
        dataset=root,
        options={"format": "parquet", "bundle": False, "layout": "frame"},
    )

    queued = await _mint_recorded_root(handler, stored, record)
    again = await _mint_recorded_root(handler, stored, record)

    assert queued == QueuedOperation(
        operation_id=_OPERATION_ID, status="pending", retry_after_s=1.5
    )
    assert isinstance(again, RootMint)
    # The repeat is the same request, which apis answers from the operation it holds.
    assert submitted[0] == submitted[1]
    assert len(submitted) == 2


@pytest.mark.parametrize(
    ("entities", "recorded", "minted"),
    [
        (
            {"origin": ["country/FRA"]},
            {"format": "parquet", "columns": ["financingProvider_type", "financingRecipient_type"]},
            {
                "format": "parquet",
                "columns": ["financingProvider_type", "financingRecipient_type", *KIND_COLUMNS],
            },
        ),
        (
            {"origin": ["country/FRA"]},
            {"format": "parquet", "columns": ["kind"]},
            {
                "format": "parquet",
                "columns": ["financingProvider_type", "financingRecipient_type", *KIND_COLUMNS],
            },
        ),
        (
            {"origin": ["country/FRA"]},
            {"format": "parquet", "columns": ["party_kind"]},
            {
                "format": "parquet",
                "columns": ["financingProvider_type", "financingRecipient_type", *KIND_COLUMNS],
            },
        ),
        (
            {"origin": ["country/FRA"]},
            {"format": "parquet"},
            {
                "format": "parquet",
                "columns": ["financingProvider_type", "financingRecipient_type", *KIND_COLUMNS],
            },
        ),
        ({"recipient": ["country/KEN"]}, {"format": "parquet"}, {"format": "parquet"}),
    ],
    ids=[
        "flow-party-types",
        "flow-retired-kind",
        "flow-kind-without-class-and-whole",
        "flow-no-columns",
        "place-no-columns",
    ],
)
async def test_a_recorded_flow_root_minted_without_kinds_is_minted_again_with_them(
    monkeypatch: pytest.MonkeyPatch,
    entities: dict[str, list[str]],
    recorded: dict[str, Any],
    minted: dict[str, Any],
) -> None:
    # A flow root recorded before kinds were served holds no kinds, and a sum across its
    # parties would refuse; the replay mints and registers it with `KIND_COLUMNS`.
    stored = _stored_definition(entities=entities)
    registered: list[Mapping[str, Any]] = []

    async def degraded(*, manifest: Manifest, options: Mapping[str, Any], **_: Any):
        registered.append(options)
        return MaterialisationOutcome(
            status="degraded", attempts=1, degraded_reason="fetch", frame_ref=mint_pair(manifest)
        )

    monkeypatch.setattr(get_data_tool, "materialise_and_register", degraded)
    sent: list[dict[str, Any]] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        sent.append(json.loads(request.content)["options"])
        return httpx2.Response(200, json=_manifest_response(spec_id=stored.spec_id))

    root = FrameRef(spec_id=stored.spec_id, content_version="cv_" + "1" * 32)

    await _mint_recorded_root(
        handler, stored, MintRecord(frame=root, dataset=root, options=recorded)
    )

    assert sent == [minted]
    assert registered == [minted]


@pytest.mark.parametrize(
    "recorded_columns", [list(KIND_COLUMNS), ["party_kind"]], ids=["kinds", "kind-alone"]
)
async def test_a_recorded_root_without_party_totals_is_minted_again_with_the_option(
    monkeypatch: pytest.MonkeyPatch, recorded_columns: list[str]
) -> None:
    # The option is apis': the replay sends it as recorded, with the kinds alone it asks
    # for now, and the record of the new pair keeps it.
    stored = _stored_definition(**_FLOW_OPEN_PROVIDER)
    registered: list[Mapping[str, Any]] = []

    async def registered_as_asked(
        *, manifest: Manifest, options: Mapping[str, Any], **_: Any
    ) -> MaterialisationOutcome:
        registered.append(options)
        return MaterialisationOutcome(
            status="registered",
            attempts=1,
            frame_ref=mint_pair(manifest),
            stat_var_count=1,
        )

    monkeypatch.setattr(get_data_tool, "materialise_and_register", registered_as_asked)
    sent: list[dict[str, Any]] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        sent.append(json.loads(request.content)["options"])
        return httpx2.Response(
            200, json=_manifest_response(spec_id=stored.spec_id, content_version="cv_" + "2" * 32)
        )

    recorded = {"format": "parquet", "columns": recorded_columns, "without_party_totals": True}
    expected = recorded | {"columns": list(KIND_COLUMNS)}
    root = FrameRef(spec_id=stored.spec_id, content_version="cv_" + "1" * 32)

    minted = await _mint_recorded_root(
        handler, stored, MintRecord(frame=root, dataset=root, options=recorded)
    )

    assert sent == [expected]
    assert registered == [expected]
    assert isinstance(minted, RootMint)
    assert minted.options == expected


@pytest.mark.parametrize(
    ("entities", "served", "expected"),
    [
        (
            {"origin": ["country/FRA"]},
            ["financingProvider_type", "financingProvider_party_kind"],
            ["financingProvider_type", *KIND_COLUMNS],
        ),
        (
            {"destination": ["country/KEN"]},
            ["financingRecipient_type", "financingRecipient_party_kind"],
            ["financingRecipient_type", *KIND_COLUMNS],
        ),
    ],
    ids=["provider-only", "destination-only"],
)
async def test_a_replayed_flow_root_records_the_metadata_options_that_succeeded(
    monkeypatch: pytest.MonkeyPatch,
    entities: dict[str, list[str]],
    served: list[str],
    expected: list[str],
) -> None:
    stored = _stored_definition(entities=entities)
    root = FrameRef(spec_id=stored.spec_id, content_version="cv_" + "1" * 32)
    sent: list[dict[str, Any]] = []

    async def registered(*, manifest: Manifest, **_: Any) -> MaterialisationOutcome:
        return MaterialisationOutcome(
            status="registered",
            attempts=0,
            frame_ref=root,
            stat_var_count=1,
        )

    monkeypatch.setattr(get_data_tool, "materialise_and_register", registered)

    def handler(request: httpx2.Request) -> httpx2.Response:
        sent.append(json.loads(request.content)["options"])
        return (
            _column_unknown(served=served)
            if len(sent) == 1
            else httpx2.Response(
                200,
                json=_manifest_response(
                    spec_id=stored.spec_id, content_version=root.content_version
                ),
            )
        )

    minted = await _mint_recorded_root(
        handler,
        stored,
        MintRecord(frame=root, dataset=root, options={"format": "parquet"}),
    )

    assert isinstance(minted, RootMint)
    assert minted.options == {"format": "parquet", "columns": expected}
    assert [options["columns"] for options in sent] == [
        ["financingProvider_type", "financingRecipient_type", *KIND_COLUMNS],
        expected,
    ]


async def test_a_replayed_terminal_column_refusal_retries_without_waiting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stored = _stored_definition(entities={"origin": ["country/FRA"]})
    root = FrameRef(spec_id=stored.spec_id, content_version="cv_" + "1" * 32)
    posts: list[dict[str, Any]] = []

    async def should_not_materialise(**_: Any) -> MaterialisationOutcome:
        raise AssertionError("the retried operation is still queued")

    monkeypatch.setattr(get_data_tool, "materialise_and_register", should_not_materialise)

    def handler(request: httpx2.Request) -> httpx2.Response:
        if request.url.path == "/api/datasets":
            posts.append(json.loads(request.content)["options"])
            if len(posts) == 1:
                return httpx2.Response(
                    202,
                    json={
                        "operation": {
                            "operation_id": _OPERATION_ID,
                            "status": "failed",
                            "poll": _OPERATION_POLL_PATH,
                        }
                    },
                )
            return httpx2.Response(
                202,
                json={
                    "operation": {
                        "operation_id": _OPERATION_ID,
                        "status": "pending",
                        "poll": _OPERATION_POLL_PATH,
                    }
                },
            )
        assert request.url.path == _OPERATION_POLL_PATH
        return _failed_operation_column_unknown(
            served=["financingProvider_type", "financingProvider_party_kind"]
        )

    result = await _mint_recorded_root(
        handler, stored, MintRecord(frame=root, dataset=root, options={"format": "parquet"})
    )

    assert result == QueuedOperation(
        operation_id=_OPERATION_ID, status="pending", retry_after_s=2.0
    )
    assert [options["columns"] for options in posts] == [
        ["financingProvider_type", "financingRecipient_type", *KIND_COLUMNS],
        ["financingProvider_type", *KIND_COLUMNS],
    ]


async def test_a_replayed_terminal_column_refusal_records_the_corrected_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stored = _stored_definition(entities={"origin": ["country/FRA"]})
    root = FrameRef(spec_id=stored.spec_id, content_version="cv_" + "1" * 32)

    async def registered(*, manifest: Manifest, **_: Any) -> MaterialisationOutcome:
        return MaterialisationOutcome(
            status="registered", attempts=0, frame_ref=root, stat_var_count=1
        )

    monkeypatch.setattr(get_data_tool, "materialise_and_register", registered)
    posts: list[dict[str, Any]] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        if request.url.path == "/api/datasets":
            posts.append(json.loads(request.content)["options"])
            if len(posts) == 1:
                return httpx2.Response(
                    202,
                    json={
                        "operation": {
                            "operation_id": _OPERATION_ID,
                            "status": "failed",
                            "poll": _OPERATION_POLL_PATH,
                        }
                    },
                )
            return httpx2.Response(
                200,
                json=_manifest_response(
                    spec_id=stored.spec_id, content_version=root.content_version
                ),
            )
        assert request.url.path == _OPERATION_POLL_PATH
        return _failed_operation_column_unknown(
            served=["financingProvider_type", "financingProvider_party_kind"]
        )

    result = await _mint_recorded_root(
        handler, stored, MintRecord(frame=root, dataset=root, options={"format": "parquet"})
    )

    assert isinstance(result, RootMint)
    assert result.options == {
        "format": "parquet",
        "columns": ["financingProvider_type", *KIND_COLUMNS],
    }


@pytest.mark.parametrize("queued", [False, True], ids=["direct", "terminal-operation"])
async def test_a_replayed_explicit_place_type_is_not_removed_by_negotiation(
    queued: bool,
) -> None:
    stored = _stored_definition(entities={"recipient": ["country/KEN"]})
    root = FrameRef(spec_id=stored.spec_id, content_version="cv_" + "1" * 32)
    posts: list[dict[str, Any]] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        if request.url.path == "/api/datasets":
            posts.append(json.loads(request.content)["options"])
            if not queued:
                return _column_unknown(served=["stat_var", "entity", "value"])
            return httpx2.Response(
                202,
                json={
                    "operation": {
                        "operation_id": _OPERATION_ID,
                        "status": "failed",
                        "poll": _OPERATION_POLL_PATH,
                    }
                },
            )
        assert request.url.path == _OPERATION_POLL_PATH
        return _failed_operation_column_unknown(served=["stat_var", "entity", "value"])

    result = await _mint_recorded_root(
        handler,
        stored,
        MintRecord(
            frame=root,
            dataset=root,
            options={"format": "parquet", "columns": ["observationAbout_type"]},
        ),
    )

    assert isinstance(result, ToolResult)
    assert result.structured_content["upstream_code"] == "column_unknown"
    assert posts == [{"format": "parquet", "columns": ["observationAbout_type"]}]


async def test_a_replayed_explicit_flow_type_is_not_removed_by_negotiation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stored = _stored_definition(entities={"origin": ["country/FRA"]})
    root = FrameRef(spec_id=stored.spec_id, content_version="cv_" + "1" * 32)
    calls: list[httpx2.Request] = []

    async def should_not_materialise(**_: Any) -> MaterialisationOutcome:
        raise AssertionError("the explicit unsupported type must be relayed")

    monkeypatch.setattr(get_data_tool, "materialise_and_register", should_not_materialise)

    def handler(request: httpx2.Request) -> httpx2.Response:
        calls.append(request)
        return _column_unknown(served=["financingProvider_type", "financingProvider_party_kind"])

    result = await _mint_recorded_root(
        handler,
        stored,
        MintRecord(
            frame=root,
            dataset=root,
            options={"format": "parquet", "columns": ["financingRecipient_type", "custom"]},
        ),
    )

    assert isinstance(result, ToolResult)
    assert result.structured_content["upstream_code"] == "column_unknown"
    assert len(calls) == 1
    assert json.loads(calls[0].content)["options"]["columns"] == [
        "financingRecipient_type",
        "custom",
        "financingProvider_type",
        *KIND_COLUMNS,
    ]


async def test_a_root_whose_frame_could_not_be_fetched_refuses_with_a_wait(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stored = _stored_definition()

    async def degraded(*, manifest: Manifest, **_: Any) -> MaterialisationOutcome:
        return MaterialisationOutcome(
            status="degraded",
            attempts=3,
            degraded_reason="fetch",
            frame_ref=mint_pair(manifest),
        )

    monkeypatch.setattr(get_data_tool, "materialise_and_register", degraded)
    root = FrameRef(spec_id=stored.spec_id, content_version="cv_" + "1" * 32)
    record = MintRecord(frame=root, dataset=root, options={"format": "parquet"})

    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, json=_manifest_response(spec_id=stored.spec_id))

    refused = await _mint_recorded_root(handler, stored, record)

    assert isinstance(refused, ToolResult)
    assert refused.structured_content is not None
    assert refused.structured_content["code"] == "service_unavailable"
    assert refused.structured_content["retry_mode"] == "retry_after_delay"


@pytest.mark.parametrize(
    "order", [(False, True), (True, False)], ids=["plain-then-typed", "typed-then-plain"]
)
async def test_a_typed_and_a_plain_read_of_one_mint_are_two_frames(
    monkeypatch: pytest.MonkeyPatch, order: tuple[bool, bool]
) -> None:
    """apis answers both renderings under one `{spec_id, content_version}`. The
    store is insert-only, so under one key whichever was stored first would be served
    to the other. The plain frame keeps the mint's pair, so a frame stored before
    renderings had their own identity still resolves, and the typed frame is held
    under a `content_version` of its own. Each is fetched once, under the mint's pair."""
    monkeypatch.undo()  # the real `materialise_and_register`
    stored = _stored_definition()
    upstream = _PlainAndTypedMint(spec_id=stored.spec_id)
    monkeypatch.setattr("data_mcp.frame_materialisation._fetch_artifact_bytes", upstream.fetch)
    _serve_canonical_frames(monkeypatch, {upstream.content_version: upstream.canonical})
    frame_store = InMemoryMaterialisationStore()
    set_frame_store(frame_store)

    results = await _call_get_data_in_one_process(
        upstream.handler,
        _StubStore(stored),
        [{"spec_id": stored.spec_id, "place_types": is_typed} for is_typed in order],
    )

    assert [r.is_error for r in results] == [False, False]
    content = {
        is_typed: result.structured_content
        for is_typed, result in zip(order, results, strict=True)
        if result.structured_content is not None
    }
    plain_ref = FrameRef.model_validate(content[False]["frame_ref"])
    typed_ref = FrameRef.model_validate(content[True]["frame_ref"])
    assert plain_ref == FrameRef(spec_id=stored.spec_id, content_version=upstream.content_version)
    assert typed_ref.spec_id == stored.spec_id
    assert typed_ref.content_version != upstream.content_version
    # The manifest stays apis' document, naming the materialisation its artifacts
    # are read under.
    assert {c["content_version"] for c in content.values()} == {upstream.content_version}
    assert content[False]["role_keys"] == ["observationAbout"]
    assert content[True]["role_keys"] == ["observationAbout", "observationAbout_type"]
    assert sorted(upstream.fetched) == sorted(
        [
            (upstream.content_version, _digest(upstream.canonical)),
            (upstream.content_version, upstream.digest_by_rendering[True]),
        ]
    )
    typed_text = text_of(results[order.index(True)])
    assert f"content_version={typed_ref.content_version}" in typed_text
    plain_frame = await frame_store.get(plain_ref)
    typed_frame = await frame_store.get(typed_ref)
    assert plain_frame is not None
    assert typed_frame is not None
    # Each frame's mint record sits under the frame's own pair and names apis'.
    dataset = FrameRef(spec_id=stored.spec_id, content_version=upstream.content_version)
    plain_options = {"format": "parquet", "bundle": False, "layout": "frame"}
    assert await frame_store.get_mint_record(plain_ref) == MintRecord(
        frame=plain_ref, dataset=dataset, options=plain_options, stat_var_count=1
    )
    assert await frame_store.get_mint_record(typed_ref) == MintRecord(
        frame=typed_ref,
        dataset=dataset,
        options=plain_options | {"columns": ["observationAbout_type"]},
        stat_var_count=1,
    )
    assert "observationAbout_type" not in plain_frame[0].columns
    assert typed_frame[0].table.column("observationAbout_type").to_pylist() == [
        "Place",
        "Country",
        "AdministrativeArea1,State",
    ]


_FLOW_OPEN_PROVIDER = {"entities": {"destination": ["country/KEN"]}, "open_roles": ["origin"]}
_PLACE_SPEC = {"entities": {"recipient": ["country/KEN"]}}


@pytest.mark.parametrize(
    ("identity", "arguments", "columns", "cut"),
    [
        (_FLOW_OPEN_PROVIDER, {"without_party_totals": True}, list(KIND_COLUMNS), True),
        (
            _FLOW_OPEN_PROVIDER,
            {"place_types": True},
            ["financingProvider_type", "financingRecipient_type", *KIND_COLUMNS],
            False,
        ),
        (
            _FLOW_OPEN_PROVIDER,
            {"without_party_totals": True, "place_types": True},
            ["financingProvider_type", "financingRecipient_type", *KIND_COLUMNS],
            True,
        ),
        (_FLOW_OPEN_PROVIDER, {}, None, False),
        (_PLACE_SPEC, {"without_party_totals": True}, None, False),
        (
            _PLACE_SPEC,
            {"without_party_totals": True, "place_types": True},
            ["observationAbout_type"],
            False,
        ),
    ],
    ids=[
        "flow-totals-only",
        "flow-types-only",
        "flow-both",
        "flow-neither",
        "place-totals-only",
        "place-both",
    ],
)
async def test_without_party_totals_sends_a_flow_spec_the_option_with_its_kinds(
    monkeypatch: pytest.MonkeyPatch,
    identity: dict[str, Any],
    arguments: dict[str, Any],
    columns: list[str] | None,
    cut: bool,
) -> None:
    stored = _stored_definition(**identity)
    registered: list[Mapping[str, Any]] = []

    async def record(
        *, manifest: Manifest, options: Mapping[str, Any], **_: Any
    ) -> MaterialisationOutcome:
        registered.append(options)
        return MaterialisationOutcome(
            status="registered", attempts=0, frame_ref=mint_pair(manifest)
        )

    monkeypatch.setattr(get_data_tool, "materialise_and_register", record)
    sent: list[dict[str, Any]] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        sent.append(json.loads(request.content)["options"])
        return httpx2.Response(200, json=_manifest_response(spec_id=stored.spec_id))

    result = await _call_get_data(
        handler, _StubStore(stored), {"spec_id": stored.spec_id, **arguments}
    )

    assert result.is_error is False, result.structured_content
    assert [sent_options.get("columns") for sent_options in sent] == [columns]
    # Sent only when true, as apis records it in a rendering only then.
    assert [sent_options.get("without_party_totals") for sent_options in sent] == [
        True if cut else None
    ]
    assert registered == sent


class _FlowMint:
    """apis for a flow spec with an open provider side and a named recipient. Its frame
    holds single parties, group totals, a residual and a party of no kind. A mint with
    `options.without_party_totals` answers as apis cuts it: the rows of the providers of
    kind `total` left out, under a version of its own (`CUT_VERSION`) whose coverage counts
    the rows kept and whose trace lists the totals left out. The artifact of each mint is the
    rendering it asked for. Records every artifact fetch."""

    # provider, provider kind
    PROVIDERS: ClassVar[list[tuple[str, str | None]]] = [
        ("country/FRA", "party"),
        ("DACTotal", "total"),
        ("G7", "total"),
        ("DACUnspecified", "residual"),
        ("country/XYZ", None),
    ]
    CUT_VERSION = "cv_" + "c" * 32

    def __init__(self, *, spec_id: str) -> None:
        self.spec_id = spec_id
        self.content_version = _content_version_of(_parquet_bytes())
        self.artifacts = {cut: self._artifact(cut=cut) for cut in (False, True)}
        self.fetched: list[str] = []

    def _providers(self, *, cut: bool) -> list[tuple[str, str | None]]:
        return [(name, kind) for name, kind in self.PROVIDERS if not (cut and kind == "total")]

    def _artifact(self, *, cut: bool) -> bytes:
        providers = self._providers(cut=cut)
        count = len(providers)
        table = pa.table(
            {
                "stat_var": pa.array(["dcid:Test"] * count, type=pa.string()),
                "financingProvider": pa.array([name for name, _ in providers]),
                "financingRecipient": pa.array(["country/KEN"] * count),
                "time_period": pa.array(["2021"] * count, type=pa.string()),
                "value": pa.array([float(index + 1) for index in range(count)]),
                "financingProvider_party_kind": pa.array(
                    [kind for _, kind in providers], pa.string()
                ),
                "financingRecipient_party_kind": pa.array(["party"] * count),
            }
        )
        buf = io.BytesIO()
        pq.write_table(table, buf)
        return buf.getvalue()

    def handler(self, request: httpx2.Request) -> httpx2.Response:
        assert request.url.path == "/api/datasets"
        options = json.loads(request.content)["options"]
        cut = options.get("without_party_totals") is True
        providers = self._providers(cut=cut)
        response = _manifest_response(
            spec_id=self.spec_id, content_version=self.CUT_VERSION if cut else self.content_version
        )
        response["definition"]["identity"] = _identity_doc(**_FLOW_OPEN_PROVIDER)
        response["coverage"]["rows"] = len(providers)
        response["coverage"]["dimensions"] = [
            {"count": len(providers), "label": "financingProvider", "axis": "origin"},
            {"count": 1, "label": "financingRecipient", "axis": "destination"},
        ]
        if cut:
            response["trace"] = {
                "party_totals_left_out": [
                    {"role": "financingProvider", "entities": ["DACTotal", "G7"]}
                ]
            }
        response["artifacts"][0] |= {
            "artifact_digest": _digest(self.artifacts[cut]),
            "rendering": options,
        }
        return httpx2.Response(200, json=response)

    async def fetch(self, *, artifact_digest: str, **_: Any) -> bytes:
        self.fetched.append(artifact_digest)
        return next(data for data in self.artifacts.values() if _digest(data) == artifact_digest)


async def test_a_frame_without_party_totals_is_dfo_apis_cut_version_held_and_served_again(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.undo()  # the real `materialise_and_register`
    stored = _stored_definition(**_FLOW_OPEN_PROVIDER)
    upstream = _FlowMint(spec_id=stored.spec_id)
    monkeypatch.setattr("data_mcp.frame_materialisation._fetch_artifact_bytes", upstream.fetch)
    frame_store = InMemoryMaterialisationStore()
    set_frame_store(frame_store)

    with_totals, left_out, left_out_again = await _call_get_data_in_one_process(
        upstream.handler,
        _StubStore(stored),
        [
            {"spec_id": stored.spec_id, "place_types": True},
            {"spec_id": stored.spec_id, "without_party_totals": True},
            {"spec_id": stored.spec_id, "without_party_totals": True},
        ],
    )

    assert [r.is_error for r in (with_totals, left_out, left_out_again)] == [False] * 3
    assert with_totals.structured_content is not None
    assert left_out.structured_content is not None
    assert left_out_again.structured_content is not None
    kept_ref = FrameRef.model_validate(with_totals.structured_content["frame_ref"])
    cut_ref = FrameRef.model_validate(left_out.structured_content["frame_ref"])
    # The manifest names apis' cut version, and the frame is held under the pair its
    # kinds and the cut derive from that version.
    assert left_out.structured_content["content_version"] == _FlowMint.CUT_VERSION
    assert cut_ref == frame_ref_for(
        spec_id=stored.spec_id,
        content_version=_FlowMint.CUT_VERSION,
        columns=KIND_COLUMNS,
        without_party_totals=True,
    )
    assert cut_ref != kept_ref
    # The repeat is a store hit: it fetches nothing and names the same frame.
    assert left_out_again.structured_content["frame_ref"] == cut_ref.model_dump()
    assert upstream.fetched == [
        _digest(upstream.artifacts[False]),
        _digest(upstream.artifacts[True]),
    ]
    kept_held = await frame_store.get(kept_ref)
    cut_held = await frame_store.get(cut_ref)
    assert kept_held is not None
    assert cut_held is not None
    assert sorted(kept_held[0].table.column("financingProvider").to_pylist()) == sorted(
        name for name, _ in upstream.PROVIDERS
    )
    assert sorted(cut_held[0].table.column("financingProvider").to_pylist()) == [
        "DACUnspecified",
        "country/FRA",
        "country/XYZ",
    ]
    # The record keeps the option, so a replay asks apis for the cut again.
    record = await frame_store.get_mint_record(cut_ref)
    assert record is not None
    assert record.dataset == FrameRef(spec_id=stored.spec_id, content_version=_FlowMint.CUT_VERSION)
    assert record.options == {
        "format": "parquet",
        "bundle": False,
        "layout": "frame",
        "columns": list(KIND_COLUMNS),
        "without_party_totals": True,
    }
    # The digest names the totals the manifest's trace lists, on the fresh mint and the hit.
    for result in (left_out, left_out_again):
        assert (
            "Left out of the frame: 2 group totals on financingProvider (DACTotal, G7)"
            in text_of(result)
        )
    assert "Left out of the frame" not in text_of(with_totals)


# `GET /api/datasets/{id}/options` as apis answered it for the DAC2a spec "official
# development assistance to Kenya from every provider, 2015-2022".
_REGISTRY_OPTIONS = json.loads(
    (Path(__file__).parent / "fixtures" / "options_dac2a_kenya_open_provider.json").read_text()
)
_REGISTRY_SPEC_ID = _REGISTRY_OPTIONS["definition"]["spec_id"]


class _RegistryFlowMint:
    """apis for the spec `_REGISTRY_OPTIONS` records, which the definition store does not
    hold: its registry answers the options route, and a flow table refuses a place type
    column and serves the artifact of the rendering that asks for party kinds, cut of the
    providers of kind `total` under a version of its own (`CUT_VERSION`) when the mint asks
    for `without_party_totals`. Every manifest carries the recorded definition, coverage and
    trace. Records each request."""

    CUT_VERSION = "cv_" + "d" * 32

    PROVIDERS: ClassVar[list[tuple[str, str | None]]] = [
        ("country/FRA", "party"),
        ("DACWorldBank", "total"),
        ("WorldBankGroup", "total"),
        ("AfricanDevelopmentBankGroup", "total"),
        ("country/XYZ", None),
    ]

    def __init__(self, *, options_response: dict[str, Any] | None = None) -> None:
        self.options_response = _REGISTRY_OPTIONS if options_response is None else options_response
        self.requests: list[tuple[str, dict[str, Any] | None]] = []
        self.artifacts = {cut: self._artifact(cut=cut) for cut in (False, True)}
        self.fetched: list[str] = []

    def _artifact(self, *, cut: bool) -> bytes:
        """The frame of the rendering that asks for party kinds, less the providers of kind
        `total` when `cut`."""
        providers = [(name, kind) for name, kind in self.PROVIDERS if not (cut and kind == "total")]
        count = len(providers)
        buf = io.BytesIO()
        pq.write_table(
            pa.table(
                {
                    "stat_var": pa.array(["dcid:Test"] * count),
                    "financingProvider": pa.array([name for name, _ in providers]),
                    "financingRecipient": pa.array(["country/KEN"] * count),
                    "time_period": pa.array(["2021"] * count),
                    "value": pa.array([float(index + 1) for index in range(count)]),
                    "financingProvider_party_kind": pa.array(
                        [kind for _, kind in providers], pa.string()
                    ),
                    "financingRecipient_party_kind": pa.array(["party"] * count),
                }
            ),
            buf,
        )
        return buf.getvalue()

    def handler(self, request: httpx2.Request) -> httpx2.Response:
        if request.method == "GET":
            self.requests.append((request.url.path, None))
            assert request.url.path.endswith("/options")
            return httpx2.Response(200, json=self.options_response)
        body = json.loads(request.content)
        self.requests.append((request.url.path, body))
        columns = body["options"].get("columns")
        if columns == ["observationAbout_type"]:
            return _column_unknown(served=["financingProvider_type", "financingRecipient_type"])
        cut = body["options"].get("without_party_totals") is True
        response = _manifest_response(
            spec_id=body["input"]["spec_id"],
            content_version=self.CUT_VERSION if cut else _REGISTRY_OPTIONS["content_version"],
        )
        response["definition"] = _REGISTRY_OPTIONS["definition"]
        response["coverage"] = _REGISTRY_OPTIONS["coverage"]
        response["trace"] = _REGISTRY_OPTIONS["trace"]
        if cut:
            response["trace"] = response["trace"] | {
                "party_totals_left_out": [
                    {
                        "role": "financingProvider",
                        "entities": sorted(
                            name for name, kind in self.PROVIDERS if kind == "total"
                        ),
                    }
                ]
            }
        response["artifacts"][0] |= {
            "artifact_digest": _digest(self.artifacts[cut]),
            "rendering": body["options"],
        }
        return httpx2.Response(200, json=response)

    async def fetch(self, *, artifact_digest: str, **_: Any) -> bytes:
        self.fetched.append(artifact_digest)
        return next(data for data in self.artifacts.values() if _digest(data) == artifact_digest)


@pytest.mark.parametrize(
    ("argument", "columns", "totals_served"),
    [
        ("place_types", ["financingProvider_type", "financingRecipient_type", *KIND_COLUMNS], True),
        ("without_party_totals", list(KIND_COLUMNS), False),
    ],
)
async def test_a_flow_spec_the_definition_store_lacks_is_minted_with_kinds(
    monkeypatch: pytest.MonkeyPatch, argument: str, columns: list[str], totals_served: bool
) -> None:
    monkeypatch.undo()  # the real `materialise_and_register`
    upstream = _RegistryFlowMint()
    monkeypatch.setattr("data_mcp.frame_materialisation._fetch_artifact_bytes", upstream.fetch)
    frame_store = InMemoryMaterialisationStore()
    set_frame_store(frame_store)
    definitions = _RecordingStore()

    result = await _call_get_data(
        upstream.handler, definitions, {"spec_id": _REGISTRY_SPEC_ID, argument: True}
    )

    assert result.is_error is False, result.structured_content
    assert result.structured_content is not None
    # The registry is read first, the mint asks for kinds with the definition beside it, and
    # the definition is kept for the next call.
    assert [
        (path, body and body["options"].get("columns")) for path, body in upstream.requests
    ] == [
        (f"/api/datasets/{_REGISTRY_SPEC_ID}/options", None),
        ("/api/datasets", columns),
    ]
    mint = upstream.requests[1][1]
    assert mint is not None
    assert ("without_party_totals" in mint["options"]) is not totals_served
    assert mint["input"]["definition"]["identity"] == _REGISTRY_OPTIONS["definition"]["identity"]
    assert [put.spec_id for put in definitions.put_calls] == [_REGISTRY_SPEC_ID]
    held = await frame_store.get(FrameRef.model_validate(result.structured_content["frame_ref"]))
    assert held is not None
    table = held[0].table
    assert set(table.column("financingProvider_party_kind").to_pylist()) >= {"party"}
    providers = set(table.column("financingProvider").to_pylist())
    assert ("DACWorldBank" in providers) is totals_served
    assert ("country/XYZ" in providers) and ("country/FRA" in providers)


@pytest.mark.parametrize(
    ("argument", "columns"),
    [
        ("place_types", ["financingProvider_type", "financingRecipient_type", *KIND_COLUMNS]),
        ("without_party_totals", list(KIND_COLUMNS)),
    ],
)
async def test_a_spec_the_registry_serves_under_another_id_is_minted_with_kinds_and_no_sidecar(
    monkeypatch: pytest.MonkeyPatch, argument: str, columns: list[str]
) -> None:
    # The registry answers the definition under its own id, which the requested id is not,
    # so the mint cannot carry it. Its flow sides still choose the columns.
    monkeypatch.undo()  # the real `materialise_and_register`
    upstream = _RegistryFlowMint()
    monkeypatch.setattr("data_mcp.frame_materialisation._fetch_artifact_bytes", upstream.fetch)
    set_frame_store(InMemoryMaterialisationStore())
    definitions = _RecordingStore()

    result = await _call_get_data(
        upstream.handler, definitions, {"spec_id": "spec_" + "e" * 32, argument: True}
    )

    assert result.is_error is False, result.structured_content
    mint = upstream.requests[1][1]
    assert mint is not None
    assert mint["options"]["columns"] == columns
    assert "definition" not in mint["input"]
    # Only the manifest's own definition is kept, as for any re-mint under another id.
    assert [put.spec_id for put in definitions.put_calls] == [_REGISTRY_SPEC_ID]


@pytest.mark.parametrize("argument", ["place_types", "without_party_totals"])
async def test_a_flow_frame_that_cannot_carry_the_asked_kinds_is_refused(
    monkeypatch: pytest.MonkeyPatch, argument: str
) -> None:
    # The registry reads the spec as a place spec, and apis then serves a flow manifest
    # for the mint without kinds: that frame would carry every group total.
    monkeypatch.undo()  # the real `materialise_and_register`
    place = _stored_definition(entities={"recipient": ["country/KEN"]})
    upstream = _RegistryFlowMint(
        options_response={
            "definition": {
                "schema_version": 1,
                "spec_id": place.spec_id,
                "identity": json.loads(place.canonical_bytes),
            }
        }
    )
    monkeypatch.setattr("data_mcp.frame_materialisation._fetch_artifact_bytes", upstream.fetch)
    set_frame_store(InMemoryMaterialisationStore())

    result = await _call_get_data(
        upstream.handler, _RecordingStore(), {"spec_id": place.spec_id, argument: True}
    )

    assert result.is_error is True
    assert result.structured_content is not None
    assert result.structured_content["code"] == "upstream_error"
    assert upstream.fetched == []


async def test_a_registry_definition_the_store_cannot_keep_is_logged_and_the_call_goes_on(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.WARNING, logger="data_mcp.get_data_tool")
    monkeypatch.undo()  # the real `materialise_and_register`
    upstream = _RegistryFlowMint()
    monkeypatch.setattr("data_mcp.frame_materialisation._fetch_artifact_bytes", upstream.fetch)
    set_frame_store(InMemoryMaterialisationStore())

    result = await _call_get_data(
        upstream.handler,
        _PutFailsStore(None),  # holds nothing and refuses writes
        {"spec_id": _REGISTRY_SPEC_ID, "without_party_totals": True},
    )

    assert result.is_error is False, result.structured_content
    assert [
        record.getMessage() for record in caplog.records if _REGISTRY_SPEC_ID in record.getMessage()
    ]


async def test_a_spec_the_registry_does_not_hold_either_refuses_before_minting() -> None:
    spec_id = "spec_" + "d" * 32
    paths: list[str] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        paths.append(request.url.path)
        return httpx2.Response(404, content=_problem_body(code="spec_not_found", retryable=False))

    result = await _call_get_data(
        handler, _StubStore(None), {"spec_id": spec_id, "without_party_totals": True}
    )

    assert result.is_error is True
    assert result.structured_content is not None
    assert result.structured_content["code"] == "definition_not_held"
    assert paths == [f"/api/datasets/{spec_id}/options"]


def test_a_refusal_names_a_place_by_its_label_and_only_counts_one_without() -> None:
    """A model repeats a refusal's message to the user, so a place get_data holds no
    label for (one the spec's identity binds) is counted, never shown as its dcid."""
    labels = {"country/KEN": "Kenya"}
    assert get_data_tool._places_named(["country/KEN"], labels) == "Kenya"
    assert (
        get_data_tool._places_named(["country/KEN", "country/GHA"], labels)
        == "Kenya and 1 more place"
    )
    assert get_data_tool._places_named(["country/GHA", "country/NGA"], labels) == "2 places"


# Warning refs: every warning's refs labelled by one node lookup

_NODES_PATH = "/api/graph/nodes"

# Two warnings naming three distinct refs between them, `country/KEN` in both, and one
# naming none.
_REF_WARNINGS: list[dict[str, Any]] = [
    {
        "code": "coverage_below_scope",
        "severity": "warn",
        "message": "2 of 14 series report at a different frequency.",
        "member_count": 14,
        "off_cadence_members": {"count": 2, "refs": ["dcid:country/KEN", "dcid:country/UGA"]},
    },
    {
        "code": "deflator_missing_omitted",
        "severity": "warn",
        "message": "It left out 3 entities.",
        "omitted_entity_years": {
            "count": 3,
            "entities": [
                {"ref": "dcid:country/KEN", "years": ["2004"]},
                {"ref": "dcid:country/TCD", "years": ["2010-2012"]},
            ],
        },
    },
    {"code": "facet_fallback", "severity": "info", "message": "Fell back."},
]


def _returned_warnings(result: CallToolResult) -> list[dict[str, Any]]:
    assert result.structured_content is not None
    return result.structured_content["warnings"]


def _with_messages(*messages: str) -> list[dict[str, Any]]:
    """`_REF_WARNINGS` with the given messages, in order, every field kept, and the two
    `stat_vars_summed` fields the stripped manifest dumps as null."""
    return [
        {"contributing_member_count": None, "stat_var_label": None, **warning, "message": message}
        for warning, message in zip(_REF_WARNINGS, messages, strict=True)
    ]


def _ref_warnings_handler(
    spec_id: str,
    nodes: Handler,
    node_requests: list[list[str]],
    *,
    on_mint: Callable[[], None] = lambda: None,
) -> Handler:
    """Answer the mint with `_REF_WARNINGS`, calling `on_mint` first, and the node batch
    route with `nodes`, recording each request's dcids in `node_requests`."""
    payload = _manifest_response(spec_id=spec_id)
    payload["warnings"] = _REF_WARNINGS

    async def handler(request: httpx2.Request) -> httpx2.Response:
        if request.url.path == "/api/datasets":
            on_mint()
            return httpx2.Response(200, json=payload)
        if request.url.path == _NODES_PATH:
            node_requests.append(json.loads(request.content)["dcids"])
            response = nodes(request)
            return response if isinstance(response, httpx2.Response) else await response
        raise AssertionError(f"unexpected call to {request.url.path}")

    return handler


async def _never_answers(_request: httpx2.Request) -> httpx2.Response:
    await asyncio.Event().wait()
    raise AssertionError("the wait above never ends")


def _labelled_nodes(request: httpx2.Request) -> httpx2.Response:
    """Kenya and Uganda by name; `country/TCD` not held."""
    country = load_node_fixture("node_country")
    labels = {"country/KEN": "Kenya", "country/UGA": "Uganda"}
    entries = [
        {**country, "ref": dcid, "label": labels[dcid]}
        if dcid in labels
        else {"ref": dcid, "status": "not_held"}
        for dcid in json.loads(request.content)["dcids"]
    ]
    return httpx2.Response(200, json={"nodes": entries})


async def test_every_warnings_refs_are_labelled_by_one_node_lookup() -> None:
    stored = _stored_definition()
    node_requests: list[list[str]] = []

    result = await _call_get_data(
        _ref_warnings_handler(stored.spec_id, _labelled_nodes, node_requests),
        _StubStore(stored),
        {"spec_id": stored.spec_id},
    )

    assert result.is_error is False
    assert node_requests == [["country/KEN", "country/UGA", "country/TCD"]]
    text = text_of(result)
    assert "Series at another frequency: Kenya, Uganda." in text
    assert "Entities left out: Kenya (2004), country/TCD (2010-2012), and 1 more." in text
    # The harness copies the returned messages into card warnings.
    assert _returned_warnings(result) == _with_messages(
        "2 of 14 series report at a different frequency. "
        "Series at another frequency: Kenya, Uganda.",
        "It left out 3 entities. "
        "Entities left out: Kenya (2004), country/TCD (2010-2012), and 1 more.",
        "Fell back.",
    )


@pytest.mark.parametrize(
    ("nodes", "seconds_left"),
    [
        (
            lambda _request: httpx2.Response(
                500, content=_problem_body(code="internal", retryable=False)
            ),
            None,
        ),
        (lambda _request: httpx2.Response(200, json={"nodes": "not a list"}), None),
        # The mint answers with 3.05s of the call deadline left, so the label read, which
        # must leave 3s unspent, gets 0.05s: its request is sent and cut unanswered.
        (_never_answers, get_data_tool._STATISTICS_RESERVE_S + 0.05),
    ],
    ids=["refused", "unparsable", "past_its_reserve"],
)
async def test_a_failed_label_lookup_names_the_warnings_refs_bare(
    monkeypatch: pytest.MonkeyPatch, nodes: Handler, seconds_left: float | None
) -> None:
    clock = _FakeClock()
    monkeypatch.setattr(get_data_tool, "_deadline_clock", clock)
    stored = _stored_definition()
    node_requests: list[list[str]] = []

    def on_mint() -> None:
        if seconds_left is not None:
            clock.now = Settings().get_data_deadline_s - seconds_left

    # Without the read's own bound, the unanswered route would hold the call open; this
    # guard turns that into a failure instead of a hang.
    async with asyncio.timeout(5.0):
        result = await _call_get_data(
            _ref_warnings_handler(stored.spec_id, nodes, node_requests, on_mint=on_mint),
            _StubStore(stored),
            {"spec_id": stored.spec_id},
        )

    assert result.is_error is False
    assert len(node_requests) == 1
    text = text_of(result)
    assert "Series at another frequency: country/KEN, country/UGA." in text
    assert "Entities left out: country/KEN (2004), country/TCD (2010-2012), and 1 more." in text
    assert _returned_warnings(result) == _with_messages(
        "2 of 14 series report at a different frequency. "
        "Series at another frequency: country/KEN, country/UGA.",
        "It left out 3 entities. "
        "Entities left out: country/KEN (2004), country/TCD (2010-2012), and 1 more.",
        "Fell back.",
    )
