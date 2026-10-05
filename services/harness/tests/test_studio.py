"""The studio routes end-to-end over real HTTP, with scripted seams.

Everything runs against a FastAPI app through httpx2's ASGI transport -- real
routing, real SSE streaming, real ADK run/resume mechanics -- with the graph's
resolver and executor scripted (no model, no MCP server), following
`test_graph.py`'s fakes. Every frame read off the wire is also validated
against `harness.wire.STUDIO_FRAME_ADAPTER`, so a test that passes here has
also proven its stream contract-valid.

The scenarios pin the protocol claims `docs/studio-contract-v1.md` makes:
the pausing frames end their stream and `/api/studio/interrupt` answers with
a new stream (delta 1); `seq` stays monotonic across those segments (delta
2); bearer auth is required (delta 5); interrupt wire ids are opaque and
single-use (delta 7); the composition manifest is server-owned with
`edit_conflict` on stale revisions (delta 9).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import anyio
import harness.studio as studio_module
import harness.studio_state as studio_state_module
import httpx2
import pytest
from analysis_mcp.server import mcp as analysis_mcp
from common import telemetry
from common.telemetry import record_event
from compiler.surface import (
    FixtureSurface,
    McpSurface,
    ProgressCallback,
    ProgressReport,
    ToolCallResult,
    ToolSchema,
    ToolSurface,
)
from data_mcp.definition_store import (
    DefinitionStoreUnavailable,
    InMemoryDefinitionStore,
    canonicalize,
    spec_id_for,
)
from data_mcp.definition_store import set_store as set_definition_store
from data_mcp.server import mcp as data_mcp
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from google.adk.sessions import BaseSessionService, InMemorySessionService
from harness.conversation import current_conversation_id
from harness.graph import AGENT_NAME, GenerativeNodes, create_root_workflow
from harness.mcp_client import McpExecutor
from harness.mcp_deadlines import McpTimeoutPolicy
from harness.provenance import ProvenanceStore
from harness.scripted import (
    ScriptedCompose,
    ScriptedConverse,
    ScriptedPlannerModel,
    ScriptedSurface,
)
from harness.scripted import ScriptedExecutor as MessagePathScriptedExecutor
from harness.scripted import ScriptedResolver as MessagePathScriptedResolver
from harness.scripted_plan import scripted_fetch_series_stats
from harness.settings import Settings
from harness.studio import (
    StudioRunner,
    _acquire_or_conflict,
    _end_stream,
    _sse_stream,
    _unwind,
    create_studio_router,
)
from harness.studio_state import (
    _KEY_COMPOSITION,
    ConversationState,
    StudioStore,
    _PendingInterrupt,
    land_component,
    make_recall,
)
from harness.studio_translate import _RecordingExecutor
from harness.tool_contracts import ResolverError
from harness.wire import (
    MAX_SEED_SPEC_BYTES,
    MAX_TURN_REQUEST_BYTES,
    STUDIO_FRAME_ADAPTER,
    ActivityStep,
    ComponentType,
    CompositionActor,
    CompositionBlock,
    ErrorCode,
    ErrorFrame,
    MessageDelta,
    RetryMode,
    StepStatus,
    ToolError,
)
from mcp import MCPError
from mcp.types import REQUEST_TIMEOUT
from opentelemetry import trace as otel_trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from planner.operations import GrowthOperation, RetrieveOperation, Translation
from planner.plan import (
    SEEDS_STATE_KEY,
    DatasetRequirement,
    Decision,
    Entity,
    GrowthRequirement,
    Period,
    ResolvedBy,
    plan_revision,
)
from planner.plan import Plan as PlannerPlan
from planner.workflow import SelectedSource, ToolId, Workflow, WorkflowStep
from presentation_mcp.platform_contract import RESOLVED_PAYLOAD_META_KEY

from .data_mcp_definitions import DataMcpDefinitions
from .resolved_revision import resolved_revision
from .sqlite_session import sqlite_session_service

SPEC_A = "spec_" + "a" * 32
SPEC_B = "spec_" + "b" * 32
REF_A = {"spec_id": SPEC_A, "content_version": "cv_" + "1" * 32}
AUTH = {"Authorization": "Bearer test-token"}
CONVERSATION = "conv_01test0000"
# A write token as the platform issues one on a create: the grant a later
# revise of that component must present.
WRITE_TOKEN = "cmw_" + "t" * 22

# A platform-shaped component reply, the wire fields of component.created
# minus event/seq, as write_component's executor returns it.
COMPONENT = {
    "component_id": "cmp_01test0001",
    "revision": 1,
    "type": "stat_card",
    "catalog_version": "2026.8.0",
    "min_catalog_version": "2026.8.0",
    "payload": resolved_revision(
        "stat_card",
        component_id="cmp_01test0001",
        revision=1,
        frame_ref=REF_A,
        title="Aid to Kenya",
    ),
    "fact_summary": {
        "component_id": "cmp_01test0001",
        "revision": 1,
        "series": [{"id": "ken", "label": "Kenya", "point_count": 9}],
        "indicators": [{"id": "IND.1", "name": "Aid", "description": "Aid flows."}],
        "attribution": {
            "label": "Test source · 1 indicator",
            "source_count": 1,
            "indicator_count": 1,
        },
    },
    "provenance": {
        "inputs": [REF_A],
        "sources": [{"id": "src_test", "name": "Test source", "licence": "CC BY 4.0"}],
        "freshness": {
            "data_through": "2024",
            "last_refreshed": "2026-06-30",
            "partial_period": False,
        },
        "warnings": [],
        "generated_at": "2026-08-28T00:00:00Z",
        "citation": "Test source (2026).",
    },
}


def _get_data_manifest(*, spec_id: str = SPEC_A) -> dict[str, Any]:
    """A minimal valid `get_data` manifest. The hand-built workflows below
    carry no `evidence_bindings`, so only its shape (a definition manifest and
    the `frame_ref` beside it, which `extract_frame_ref` reads) matters."""
    return {
        "schema_version": 1,
        "definition": {"schema_version": 1, "spec_id": spec_id, "identity": {}},
        "content_version": "cv_" + "1" * 32,
        "coverage": {"rows": 10, "dimensions": []},
        "citation": {"text": "Test source", "sources": []},
        "trace": {},
        "artifacts": [],
        "frame_ref": {"spec_id": spec_id, "content_version": "cv_" + "1" * 32},
    }


def _analysis_result(*, spec_id: str = "ops_" + "2" * 32) -> dict[str, Any]:
    """A minimal analysis-mcp result: one row of one StatVar, matching a
    `WorkflowStep`'s default `expected_role_keys=()`."""
    return {
        "spec_id": spec_id,
        "content_version": "ocv_" + "3" * 32,
        "rows": 1,
        "role_keys": [],
        "stat_var_count": 1,
    }


@pytest.fixture
async def real_schemas() -> list[ToolSchema]:
    """The real servers' tool schemas. A `WorkflowStep` hashed against one of
    these passes `execute_approved_plan`'s own surface check."""
    return await McpSurface(
        sources={"data-mcp": data_mcp, "analysis-mcp": analysis_mcp}
    ).list_tools()


class _TimedSurface:
    """Wraps a `ToolSurface`, adding a small delay before each call.

    A `FixtureSurface`'s canned answer does no real work, so its elapsed time
    can round to 0ms; `_RecordingSurface`'s `duration_ms` needs to be
    reliably non-zero for a test to assert on it meaningfully.
    """

    def __init__(self, surface: ToolSurface) -> None:
        self._surface = surface

    async def list_tools(self) -> list[ToolSchema]:
        return await self._surface.list_tools()

    async def call_tool(
        self,
        *,
        tool: ToolId,
        arguments: Mapping[str, Any],
        on_progress: ProgressCallback | None = None,
    ) -> ToolCallResult:
        await asyncio.sleep(0.002)
        return await self._surface.call_tool(
            tool=tool, arguments=arguments, on_progress=on_progress
        )


def _fixture_surface(schemas: list[ToolSchema], *, analysis_tool: str) -> ToolSurface:
    """A `ToolSurface` over the real schemas, answering `get_data` and one
    analysis tool with a canned success."""
    named = {schema.name: schema for schema in schemas}
    return _TimedSurface(
        FixtureSurface(
            schemas=[named["get_data"], named[analysis_tool]],
            responses={
                "get_data": ToolCallResult(
                    is_error=False, structured_content=_get_data_manifest(), text="fetched"
                ),
                analysis_tool: ToolCallResult(
                    is_error=False, structured_content=_analysis_result(), text="computed"
                ),
            },
        )
    )


def _envelope(**overrides: Any) -> dict[str, Any]:
    envelope: dict[str, Any] = {
        "status": "definite",
        "spec_id": SPEC_A,
        "choice_points": 0,
        "parts": 1,
        "warnings": [],
        "candidates": [],
    }
    envelope.update(overrides)
    return envelope


class ScriptedResolver:
    def __init__(self, envelopes: list[dict[str, Any]]) -> None:
        self.envelopes = envelopes
        self.calls: list[dict[str, Any]] = []

    async def __call__(
        self,
        *,
        question: str,
        spec_id: str | None = None,
        wide: bool = False,
        parts: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        self.calls.append({"question": question, "spec_id": spec_id, "wide": wide, "parts": parts})
        index = min(len(self.calls) - 1, len(self.envelopes) - 1)
        return self.envelopes[index]


class ScriptedExecutor:
    """`strict=True` fails the test on an unexpected extra call. The permissive
    default answers every call, so a route making calls the script never
    anticipated passes unnoticed."""

    def __init__(
        self, results: list[dict[str, Any]] | None = None, *, strict: bool = False
    ) -> None:
        self.results = results or []
        self.strict = strict
        self.calls: list[dict[str, Any]] = []

    async def __call__(
        self, *, tool: str, args: dict[str, Any], on_progress: ProgressCallback | None = None
    ) -> dict[str, Any]:
        self.calls.append({"tool": tool, "args": args})
        if len(self.calls) <= len(self.results):
            return self.results[len(self.calls) - 1]
        if self.strict:
            raise AssertionError(f"unscripted executor call #{len(self.calls)}: {tool}({args})")
        return {"outcome": "ok", "frame_refs": [REF_A]}


# The session service backend every route-exercising test in this module
# builds its store and runner over: `_new_session_service` is the one seam
# `session_service_backend` (below) switches, so the whole suite -- `_app`'s
# default and each inline `StudioRunner(...)` a test builds directly -- runs
# through `InMemorySessionService` and a sqlite-backed `DatabaseSessionService`
# without any call site naming which one it is under.
class _SessionServiceBackend:
    factory: Callable[[], BaseSessionService] = staticmethod(InMemorySessionService)


def _new_session_service() -> BaseSessionService:
    return _SessionServiceBackend.factory()


@pytest.fixture(autouse=True, params=["memory", "database"])
def session_service_backend(
    request: pytest.FixtureRequest, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> str:
    """Every studio-door test in this file runs twice: once building its
    session service through `InMemorySessionService`, once through a
    sqlite-backed `DatabaseSessionService`, proving the studio door's
    durable state survives the JSON round trip a real deployment's Postgres
    also forces it through. `db_path` is per-test (`tmp_path`), so two tests
    running the database backend never share a file."""
    if request.param == "database":
        db_path = tmp_path / "studio.db"
        monkeypatch.setattr(
            _SessionServiceBackend,
            "factory",
            staticmethod(lambda: sqlite_session_service(db_path)),
        )
    return request.param


def _app(
    resolver: ScriptedResolver,
    executor: ScriptedExecutor,
    *,
    settings: Settings | None = None,
    edit_model: Any | None = None,
    generative: GenerativeNodes | None = None,
    tool_surface: ToolSurface | None = None,
    session_service: BaseSessionService | None = None,
    store: StudioStore | None = None,
    served: Sequence[Mapping[str, Any]] = (),
) -> FastAPI:
    # Mirrors server.py's wiring: the store exists first so the recall seam
    # can resolve the current conversation per request. A caller naming no
    # `generative` still gets a working `planner` node -- the node has no
    # scripted stand-in of its own, unlike compose/narrate/converse/refusal --
    # so any test whose plan reaches choice_points > 0 can pause on a real
    # plan.proposed frame without caring about its content. `session_service`
    # defaults to this test's own backend (`session_service_backend`); a
    # cross-instance test passes one explicitly, shared with a second `_app`,
    # to build two `StudioRunner`s over one database service. `store`
    # defaults to a fresh one over `session_service`; a test that needs its
    # own clock (a segment TTL scenario) builds one and passes it instead.
    # `served` are the seeds or Spec blocks the Search API answers each
    # seed's identity with (`DataMcpDefinitions`).
    session_service = session_service or _new_session_service()
    store = store or StudioStore(session_service, app_name=AGENT_NAME)
    resolved_settings = settings or Settings()
    studio = StudioRunner(
        create_workflow=lambda recorder, surface: create_root_workflow(
            resolver=resolver,
            executor=recorder,
            generative=generative or GenerativeNodes(planner=ScriptedPlannerModel()),
            recall=make_recall(store),
            tool_surface=surface,
            definition_surface=DataMcpDefinitions(served=served),
            fetch_series_stats=scripted_fetch_series_stats,
        ),
        executor=executor,
        session_service=session_service,
        tool_surface=tool_surface,
        store=store,
        edit_model=edit_model,
        settings=resolved_settings,
    )
    app = FastAPI()
    app.state.studio = studio
    app.include_router(create_studio_router(studio, settings=resolved_settings))
    return app


def _plan_generative() -> GenerativeNodes:
    """The scripted planner + compose pair `server.py` wires for
    `STUDIO_SCRIPTED=true`: a test that needs an approved-plan turn to
    produce a real component pairs this with `tool_surface=ScriptedSurface()`.
    `executor`'s own scripting then answers `compose`'s `describe_frame` and
    `write_component` calls, the plan's own steps running through the
    surface invisibly to it."""
    return GenerativeNodes(planner=ScriptedPlannerModel(), compose=ScriptedCompose())


def _client(app: FastAPI) -> httpx2.AsyncClient:
    return httpx2.AsyncClient(
        transport=httpx2.ASGITransport(app=app), base_url="http://studio.test"
    )


def _turn_body(message: str = "kenya oda 2023") -> dict[str, Any]:
    return {
        "kind": "turn.request",
        "schema_version": 1,
        "conversation_id": CONVERSATION,
        "message": message,
    }


def _interrupt_body(interrupt_id: str, decision: str, payload: Any = None) -> dict[str, Any]:
    body: dict[str, Any] = {
        "kind": "interrupt.response",
        "schema_version": 1,
        "conversation_id": CONVERSATION,
        "interrupt_id": interrupt_id,
        "decision": decision,
    }
    if payload is not None:
        body["payload"] = payload
    return body


async def _stream(
    client: httpx2.AsyncClient, path: str, body: dict[str, Any]
) -> list[dict[str, Any]]:
    """POST and read the SSE stream to its end; every frame must be
    contract-valid."""
    frames: list[dict[str, Any]] = []
    async with client.stream("POST", path, json=body, headers=AUTH) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        async for line in response.aiter_lines():
            if line.startswith("data: "):
                frame = json.loads(line[len("data: ") :])
                STUDIO_FRAME_ADAPTER.validate_python(frame)
                frames.append(frame)
    return frames


def _events(frames: list[dict[str, Any]]) -> list[str]:
    return [frame["event"] for frame in frames]


def _frame(frames: list[dict[str, Any]], event: str) -> dict[str, Any]:
    matches = [frame for frame in frames if frame["event"] == event]
    assert matches, f"no {event} frame in {_events(frames)}"
    return matches[-1]


# --- context cache -------------------------------------------------------------


def _cache_studio(settings: Settings | None) -> StudioRunner:
    return StudioRunner(
        create_workflow=lambda recorder, surface: create_root_workflow(
            resolver=ScriptedResolver([_envelope()]),
            executor=recorder,
            generative=GenerativeNodes(planner=ScriptedPlannerModel()),
            tool_surface=surface,
            fetch_series_stats=scripted_fetch_series_stats,
        ),
        executor=ScriptedExecutor(),
        session_service=_new_session_service(),
        settings=settings,
    )


def test_context_cache_defaults_on_with_no_settings() -> None:
    studio = _cache_studio(None)
    assert studio.runner.context_cache_config is not None
    assert studio.runner.context_cache_config.cache_intervals == 10


def test_context_cache_follows_the_settings_toggle_and_interval() -> None:
    studio = _cache_studio(Settings(studio_context_cache=True, studio_cache_intervals=25))
    assert studio.runner.context_cache_config.cache_intervals == 25

    studio = _cache_studio(Settings(studio_context_cache=False))
    assert studio.runner.context_cache_config is None


def test_context_cache_interval_rejects_out_of_range_values() -> None:
    with pytest.raises(ValueError, match="studio_cache_intervals must be between 1 and 100"):
        Settings(studio_cache_intervals=0)
    with pytest.raises(ValueError, match="studio_cache_intervals must be between 1 and 100"):
        Settings(studio_cache_intervals=101)


# --- auth ---------------------------------------------------------------------


async def test_a_request_without_a_bearer_token_is_401() -> None:
    async with _client(_app(ScriptedResolver([_envelope()]), ScriptedExecutor())) as client:
        response = await client.post("/api/studio/turn", json=_turn_body())
    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"


async def test_a_wrong_shared_secret_is_401() -> None:
    app = _app(
        ScriptedResolver([_envelope()]),
        ScriptedExecutor(),
        settings=Settings(studio_api_token="right"),
    )
    async with _client(app) as client:
        response = await client.post(
            "/api/studio/turn",
            json=_turn_body(),
            headers={"Authorization": "Bearer wrong"},
        )
    assert response.status_code == 401


# --- the direct-fetch turn ------------------------------------------------------


async def test_the_resolve_step_names_every_search_part() -> None:
    parts = [
        {"variable": "life expectancy", "status": "definite"},
        {"variable": "health spending", "status": "candidates"},
    ]
    envelope = _envelope(parts=2, part_digests=parts)
    executor = ScriptedExecutor([{"outcome": "ok", "frame_refs": [REF_A]}])
    async with _client(_app(ScriptedResolver([envelope]), executor)) as client:
        frames = await _stream(client, "/api/studio/turn", _turn_body())

    resolve = next(f for f in frames if f["event"] == "activity.step" and f["step_id"] == "resolve")
    assert resolve["detail"] == (
        "resolved: definite — 2 parts: life expectancy: definite; health spending: candidates"
    )


async def test_a_definite_turn_streams_steps_text_and_done() -> None:
    executor = ScriptedExecutor([{"outcome": "ok", "frame_refs": [REF_A]}], strict=True)
    async with _client(_app(ScriptedResolver([_envelope()]), executor)) as client:
        frames = await _stream(client, "/api/studio/turn", _turn_body())

    # A definite resolve fetches once and asks the executor for nothing else.
    assert [call["tool"] for call in executor.calls] == ["get_data"]
    events = _events(frames)
    assert events[-1] == "done"
    assert "suggestions" in events
    resolve = next(f for f in frames if f["event"] == "activity.step" and f["step_id"] == "resolve")
    assert resolve["status"] == "complete"
    fetch_step = next(
        f for f in frames if f["event"] == "activity.step" and f.get("tool") == "get_data"
    )
    assert fetch_step["status"] == "complete"
    complete = _frame(frames, "message.complete")
    assert "1 frame(s)" in complete["text"]
    done = _frame(frames, "done")
    assert done["trace_id"].startswith("tr_")
    assert "composition" not in done  # nothing moved; null is elided
    seqs = [frame["seq"] for frame in frames]
    assert seqs == sorted(seqs) and len(set(seqs)) == len(seqs)


async def test_a_compose_seam_describes_each_frame_and_writes_the_chosen_intents() -> None:
    """With a compose model injected, the card comes from `write_component`,
    not from the fetch: each frame is described first, every intent is
    written, and a refused write fails its step while the turn completes."""
    seen: dict[str, Any] = {}
    kept = {"type": "stat_card", "title": "Aid to Kenya", "frame_ref": REF_A}
    dropped = {"type": "line_chart", "title": "Aid over time", "frame_ref": REF_A}

    async def compose(
        *,
        question: str,
        frame_refs: list[dict[str, Any]],
        frame_views: list[dict[str, Any]],
        frame_provenance: list[dict[str, Any]],
    ) -> dict[str, Any]:
        seen.update(
            question=question,
            frame_refs=frame_refs,
            frame_views=frame_views,
            frame_provenance=frame_provenance,
        )
        return {"intents": [kept, dropped]}

    executor = ScriptedExecutor(
        [
            {"outcome": "ok", "frame_refs": [REF_A]},
            _VIEW,
            {"outcome": "ok", "frame_refs": [], "component": COMPONENT, "summary": "created"},
            {
                "outcome": "refusal",
                "refusal": {
                    "code": "schema_invalid",
                    "retry": "do_not_retry",
                    "message": "line_chart needs a period dimension",
                },
            },
        ],
        strict=True,
    )
    app = _app(
        ScriptedResolver([_envelope()]), executor, generative=GenerativeNodes(compose=compose)
    )
    async with _client(app) as client:
        frames = await _stream(client, "/api/studio/turn", _turn_body())

    assert [c["tool"] for c in executor.calls] == [
        "get_data",
        "describe_frame",
        "write_component",
        "write_component",
    ]
    assert executor.calls[1]["args"] == {"frame_ref": REF_A}
    assert executor.calls[2]["args"] == {"intent": kept}
    assert executor.calls[3]["args"] == {"intent": dropped}
    assert seen == {
        "question": "kenya oda 2023",
        "frame_refs": [REF_A],
        "frame_views": [{"available": True, "view": _VIEW["view"]}],
        "frame_provenance": [{"step": 0, "tool": "get_data", "is_final_step": True}],
    }

    writes = [
        f for f in frames if f["event"] == "activity.step" and f.get("tool") == "write_component"
    ]
    assert [w["status"] for w in writes] == ["complete", "failed"]
    assert writes[0]["produced"] == {"component_id": "cmp_01test0001", "revision": 1}
    assert "period dimension" in writes[1]["detail"]
    (created,) = [f for f in frames if f["event"] == "component.created"]
    assert created["component_id"] == "cmp_01test0001"
    assert created["created_from"]["step_id"] == writes[0]["step_id"]
    done = _frame(frames, "done")
    assert [b["component_id"] for b in done["composition"]["blocks"]] == ["cmp_01test0001"]


async def test_a_planner_run_shows_working_on_a_plan_and_not_the_seams_late_reports(
    real_schemas: list[ToolSchema],
) -> None:
    """A question routed to the planner opens "Working on a plan" as the
    draft starts and completes it before the plan card. The seam's
    `on_progress` reports, relayed only once the draft is done, never
    reach the wire (see `harness.studio_translate`'s module docstring)."""

    class ProgressPlanner(ScriptedPlanner):
        async def __call__(self, *, on_progress: Any = None, **kwargs: Any) -> dict[str, Any]:
            if on_progress is not None:
                on_progress({"agent": "dfo_planner", "tools": ["propose_plan"]})
                on_progress({"agent": "dfo_translator", "tools": ["discover_source"]})
            return await super().__call__(**kwargs)

    resolver = ScriptedResolver([_envelope(choice_points=2)])
    planner = ProgressPlanner(real_schemas)
    app = _app(resolver, ScriptedExecutor(), generative=GenerativeNodes(planner=planner))
    async with _client(app) as client:
        frames = await _stream(client, "/api/studio/turn", _turn_body("q with forks"))

    lines = [
        (index, frame["label"], frame["status"])
        for index, frame in enumerate(frames)
        if frame["event"] == "activity.step" and frame["step_id"] != "resolve"
    ]
    assert [(label, status) for _, label, status in lines] == [
        ("Working on a plan", "running"),
        ("Working on a plan", "complete"),
    ]
    assert lines[-1][0] < _events(frames).index("plan.proposed")


async def test_a_crash_mid_execute_reports_the_finished_steps_on_the_failing_turn(
    real_schemas: list[ToolSchema],
) -> None:
    """The step that completed before the crash streams its own
    `activity.step` frame before the stream-fatal error frame, and the next
    turn starts clean. The plan's own steps run through `tool_surface`, so
    the surface -- not `executor` -- is what dies on its second call."""

    class DiesOnTheSecondCall:
        def __init__(self) -> None:
            self.calls = 0

        async def list_tools(self) -> list[ToolSchema]:
            return real_schemas

        async def call_tool(
            self,
            *,
            tool: ToolId,
            arguments: Mapping[str, Any],
            on_progress: ProgressCallback | None = None,
        ) -> ToolCallResult:
            self.calls += 1
            if self.calls == 1:
                return ToolCallResult(
                    is_error=False, structured_content=_get_data_manifest(), text="fetched"
                )
            raise RuntimeError("analysis mount went away")

    resolver = ScriptedResolver([_envelope(choice_points=2), _envelope()])
    planner = ScriptedPlanner(real_schemas)
    app = _app(
        resolver,
        ScriptedExecutor(),
        generative=GenerativeNodes(planner=planner),
        tool_surface=DiesOnTheSecondCall(),
    )
    async with _client(app) as client:
        first = await _stream(client, "/api/studio/turn", _turn_body("q with forks"))
        plan = _frame(first, "plan.proposed")
        failed = await _stream(
            client, "/api/studio/interrupt", _interrupt_body(plan["interrupt_id"], "approve")
        )
        second = await _stream(client, "/api/studio/turn", _turn_body("kenya oda 2023"))

    finished = [frame for frame in failed if frame["event"] == "activity.step"]
    assert [frame["status"] for frame in finished] == ["running", "complete"]
    assert finished[0]["step_id"] == finished[1]["step_id"]
    assert _events(failed)[-1] == "error"
    assert "component.created" not in _events(second)
    assert _events(second)[-1] == "done"


async def test_a_cancelled_turn_keeps_its_recorded_steps_out_of_the_next_turn(
    real_schemas: list[ToolSchema],
) -> None:
    """A step that finished while a later one hung was recorded and never
    translated, because no graph event follows a cancel. It dies with its
    turn: the next turn's stream carries only its own steps and cards."""

    class HangsOnTheSecondCall:
        def __init__(self) -> None:
            self.calls = 0

        async def list_tools(self) -> list[ToolSchema]:
            return real_schemas

        async def call_tool(
            self,
            *,
            tool: ToolId,
            arguments: Mapping[str, Any],
            on_progress: ProgressCallback | None = None,
        ) -> ToolCallResult:
            self.calls += 1
            if self.calls == 1:
                return ToolCallResult(
                    is_error=False, structured_content=_get_data_manifest(), text="fetched"
                )
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

    surface = HangsOnTheSecondCall()
    resolver = ScriptedResolver([_envelope(choice_points=2), _envelope()])
    planner = ScriptedPlanner(real_schemas)
    app = _app(
        resolver,
        ScriptedExecutor(),
        generative=GenerativeNodes(planner=planner),
        tool_surface=surface,
    )
    async with _client(app) as client:
        first = await _stream(client, "/api/studio/turn", _turn_body("q with forks"))
        plan = _frame(first, "plan.proposed")
        resume = asyncio.ensure_future(
            _stream(
                client, "/api/studio/interrupt", _interrupt_body(plan["interrupt_id"], "approve")
            )
        )

        async def _wait_for_two_calls() -> None:
            while surface.calls < 2:  # step one recorded, step two hanging
                await asyncio.sleep(0.01)

        await asyncio.wait_for(_wait_for_two_calls(), timeout=5)
        cancel = await client.post(
            "/api/studio/cancel",
            json={"kind": "turn.cancel", "schema_version": 1, "conversation_id": CONVERSATION},
            headers=AUTH,
        )
        cancelled = await asyncio.wait_for(resume, timeout=5)
        second = await _stream(client, "/api/studio/turn", _turn_body("kenya oda 2023"))

    assert cancel.status_code == 204
    assert _frame(cancelled, "error")["error"]["error"] == "cancelled"
    assert "component.created" not in _events(second)
    assert _events(second)[-1] == "done"


async def test_a_malformed_drafted_plan_ends_the_turn_as_an_error_not_a_silent_success() -> None:
    """A draft the seam could not itself validate must not reach the approve
    gate or the executor. `graph.py`'s `planner` node validates the seam's
    record as a `planner.plan.Plan` before writing it to state; the
    compiler's capability table restricts the tool vocabulary structurally
    once `execute` runs through it, so this test covers Pydantic validation
    of the drafted plan's shape itself, proven at the graph level by
    `test_graph.py::test_the_planner_node_raises_on_a_malformed_plan_rather_than_constructing_one`.
    """

    async def drafts_badly(**kwargs: Any) -> dict[str, Any]:
        return {"plan": {"decisions": "not-a-list"}, "summary": {}}

    executor = ScriptedExecutor(strict=True)
    app = _app(
        ScriptedResolver([_envelope(choice_points=2)]),
        executor,
        generative=GenerativeNodes(planner=drafts_badly),
    )
    async with _client(app) as client:
        frames = await _stream(client, "/api/studio/turn", _turn_body("q with forks"))

    assert _events(frames)[-1] == "error"
    assert executor.calls == []


# --- the plan flow: pause, approve, execute --------------------------------------


async def test_plan_proposed_pauses_and_approve_resumes_on_a_new_stream() -> None:
    """The scripted planner/surface/compose trio -- `harness.scripted`'s
    production seams, the same ones `server.py` wires for
    `STUDIO_SCRIPTED=true` -- draft the plan and run it; `executor` serves
    only `compose`'s `describe_frame`/`write_component` calls, since
    the plan's own steps run through `tool_surface`."""
    resolver = ScriptedResolver([_envelope(choice_points=2)])
    executor = ScriptedExecutor(
        [
            {"outcome": "ok", "frame_refs": [REF_A]},
            {"outcome": "ok", "frame_refs": [REF_A], "component": COMPONENT},
        ]
    )
    app = _app(
        resolver,
        executor,
        generative=GenerativeNodes(planner=ScriptedPlannerModel(), compose=ScriptedCompose()),
        tool_surface=ScriptedSurface(),
    )
    async with _client(app) as client:
        first = await _stream(client, "/api/studio/turn", _turn_body("q with forks"))

        # The pausing frame is terminal for its stream: no done, no steps yet.
        assert _events(first)[-1] == "plan.proposed"
        assert executor.calls == []
        plan = _frame(first, "plan.proposed")
        assert plan["version"] == 1
        assert plan["interrupt_id"].startswith("int_")
        assert plan["actions"] == ["approve", "amend", "reject"]
        assert [step["label"] for step in plan["steps"]] == [
            "Fetch life expectancy at birth for Kenya",
            "Average life expectancy at birth by year",
        ]
        decision = plan["decisions"][0]
        assert decision["resolved_by"] == "default"
        assert decision["chosen"] in decision["alternatives"]

        second = await _stream(
            client, "/api/studio/interrupt", _interrupt_body(plan["interrupt_id"], "approve")
        )

    events = _events(second)
    assert events[-1] == "done"
    # The plan's own two steps ran through `tool_surface`, invisibly to this
    # `executor`; its two calls are `compose`'s describe-then-write.
    assert [c["tool"] for c in executor.calls] == ["describe_frame", "write_component"]
    created = _frame(second, "component.created")
    assert created["component_id"] == "cmp_01test0001"
    updated = _frame(second, "composition.updated")
    assert updated["manifest"]["counts"] == {
        "all": 1,
        "stats": 1,
        "charts": 0,
        "tables": 0,
        "notes": 0,
    }
    done = _frame(second, "done")
    assert done["composition"]["blocks"][0]["component_id"] == "cmp_01test0001"
    # seq is monotonic ACROSS the two segments (delta 2).
    seqs = [f["seq"] for f in first] + [f["seq"] for f in second]
    assert seqs == sorted(seqs) and len(set(seqs)) == len(seqs)


@pytest.fixture
def otlp_spans(monkeypatch: pytest.MonkeyPatch) -> InMemorySpanExporter:
    """The OTLP backend of `common.telemetry`, exporting to memory: an
    in-memory provider installed as the process's tracer provider for the
    test, and restored after it."""
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(otel_trace, "_TRACER_PROVIDER", provider)
    monkeypatch.setattr(telemetry, "_backend", "otel")
    return exporter


async def test_a_turn_paused_for_approval_exports_as_one_trace(
    otlp_spans: InMemorySpanExporter,
) -> None:
    """`/turn` and the `/interrupt` that resumes it are separate requests on
    separate tasks. Spans and events made in the resolve node of the first
    segment and in the executor of the resumed one all nest under the one
    `turn` span, whose `dfo.external_id` is the turn's wire `trace_id`."""

    class ProbingResolver(ScriptedResolver):
        async def __call__(self, **kwargs: Any) -> dict[str, Any]:
            with otel_trace.get_tracer("test.probe").start_as_current_span("resolve_probe"):
                return await super().__call__(**kwargs)

    class ProbingExecutor(ScriptedExecutor):
        async def __call__(
            self, *, tool: str, args: dict[str, Any], on_progress: ProgressCallback | None = None
        ) -> dict[str, Any]:
            with otel_trace.get_tracer("test.probe").start_as_current_span("executor_probe"):
                record_event("compose.write_refused", metadata={"tool": tool})
                return await super().__call__(tool=tool, args=args)

    app = _app(
        ProbingResolver([_envelope(choice_points=2)]),
        ProbingExecutor(
            [
                {"outcome": "ok", "frame_refs": [REF_A]},
                {"outcome": "ok", "frame_refs": [REF_A], "component": COMPONENT},
            ]
        ),
        generative=_plan_generative(),
        tool_surface=ScriptedSurface(),
    )
    async with _client(app) as client:
        first = await _stream(client, "/api/studio/turn", _turn_body("q with forks"))
        plan = _frame(first, "plan.proposed")
        second = await _stream(
            client, "/api/studio/interrupt", _interrupt_body(plan["interrupt_id"], "approve")
        )

    spans = list(otlp_spans.get_finished_spans())
    (turn,) = [span for span in spans if span.name == "turn"]
    assert (turn.attributes or {})["dfo.external_id"] == _frame(second, "done")["trace_id"]
    assert {span.context.trace_id for span in spans} == {turn.context.trace_id}
    assert len([span for span in spans if span.name == "resolve_probe"]) == 1
    executor_probes = [span for span in spans if span.name == "executor_probe"]
    assert len(executor_probes) == 2  # describe_frame, then write_component
    # `turn` is the only root: the probes of both segments sit below it.
    assert [span.name for span in spans if span.parent is None] == ["turn"]
    assert [event.name for event in executor_probes[-1].events] == ["compose.write_refused"]


async def test_every_stream_flushes_telemetry_before_the_response_ends(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    flushes: list[str] = []

    async def flush() -> None:
        flushes.append("flush")

    monkeypatch.setattr(studio_module, "flush_telemetry", flush)
    app = _app(ScriptedResolver([_envelope(choice_points=2)]), ScriptedExecutor())
    async with _client(app) as client:
        first = await _stream(client, "/api/studio/turn", _turn_body("q with forks"))
        assert flushes == ["flush"]  # the stream that paused for approval
        plan = _frame(first, "plan.proposed")
        await _stream(
            client, "/api/studio/interrupt", _interrupt_body(plan["interrupt_id"], "reject")
        )

    assert flushes == ["flush", "flush"]  # the rejecting stream


async def test_stopping_a_paused_turn_ends_its_span_and_flushes_after_it(
    otlp_spans: InMemorySpanExporter, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A paused turn's `turn` span stays open, and `/cancel` ends it with no
    stream left to flush it, so the route flushes once the span has ended."""
    finished_at_flush: list[list[str]] = []

    async def flush() -> None:
        finished_at_flush.append([span.name for span in otlp_spans.get_finished_spans()])

    monkeypatch.setattr(studio_module, "flush_telemetry", flush)
    app = _app(ScriptedResolver([_envelope(choice_points=2)]), ScriptedExecutor())
    async with _client(app) as client:
        await _stream(client, "/api/studio/turn", _turn_body("q with forks"))
        assert len(finished_at_flush) == 1  # the stream that paused
        assert "turn" not in finished_at_flush[0]

        cancel = await client.post(
            "/api/studio/cancel",
            json={"kind": "turn.cancel", "schema_version": 1, "conversation_id": CONVERSATION},
            headers=AUTH,
        )

    assert cancel.status_code == 204
    assert len(finished_at_flush) == 2
    assert "turn" in finished_at_flush[1]


async def test_cancelling_an_idle_conversation_flushes_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    flushes: list[str] = []

    async def flush() -> None:
        flushes.append("flush")

    monkeypatch.setattr(studio_module, "flush_telemetry", flush)
    async with _client(_app(ScriptedResolver([_envelope()]), ScriptedExecutor())) as client:
        cancel = await client.post(
            "/api/studio/cancel",
            json={"kind": "turn.cancel", "schema_version": 1, "conversation_id": CONVERSATION},
            headers=AUTH,
        )

    assert cancel.status_code == 204
    assert flushes == []


class ScriptedPlanner:
    """A `PlannerModel` seam, standing in for `harness.planner_seam`.

    Builds a real, translated `planner.plan.Plan` directly rather than
    running the planner agent and its translator (covered model-free in
    `test_planner_seam.py`). It proves the wire from a `planner.plan.Plan`-
    shaped payload to the plan card: any `PlannerModel`, including the real
    `harness.planner_seam.PlannerSeam`, returns that same shape, so a
    planner that settles a decision with the user reaches the plan card as
    written here.
    """

    def __init__(self, schemas: list[ToolSchema]) -> None:
        self.calls: list[dict[str, Any]] = []
        self._schemas = {schema.name: schema for schema in schemas}

    async def __call__(
        self, *, question: str, prior_plan: dict[str, Any] | None = None, **kwargs: Any
    ) -> dict[str, Any]:
        self.calls.append({"question": question, "prior_plan": prior_plan, **kwargs})
        decision = Decision(
            slot="window",
            description="Which years?",
            options=["2015-2024", "since 2018"],
            chosen="since 2018",
            resolved_by=ResolvedBy.USER,
            rationale="the user's explicit preference",
        )
        dataset = DatasetRequirement(
            id="spending",
            decision_ids=["window"],
            measure="health spending",
            entities=[Entity(name="Kenya", role="recipient")],
            period=Period(start=2018, end=2024),
        )
        growth = GrowthRequirement(
            id="growth",
            decision_ids=["window"],
            input_id="spending",
            window_start=2018,
            window_end=2024,
        )
        draft = (
            PlannerPlan.model_validate(prior_plan).amend(
                decisions=[decision], evidence_requirements=[dataset, growth]
            )
            if prior_plan is not None
            else PlannerPlan(
                question=question, decisions=[decision], evidence_requirements=[dataset, growth]
            )
        )
        translation = Translation(
            operations=[
                RetrieveOperation(id="get_data", requirement_id=dataset.id, spec_id=SPEC_A),
                GrowthOperation(
                    id="growth",
                    requirement_id=growth.id,
                    input="get_data",
                    window_start=2018,
                    window_end=2024,
                ),
            ],
            terminal=["growth"],
        )
        get_data_schema = self._schemas["get_data"]
        growth_schema = self._schemas["growth"]
        workflow = Workflow(
            steps=[
                WorkflowStep(
                    id="get_data",
                    operation_id="get_data",
                    tool=ToolId(server="data-mcp", name="get_data"),
                    args={"spec_id": SPEC_A},
                    input_schema_hash=get_data_schema.input_schema_hash,
                    output_schema_hash=get_data_schema.output_schema_hash,
                    implementation_version=get_data_schema.version,
                ),
                WorkflowStep(
                    id="growth",
                    operation_id="growth",
                    tool=ToolId(server="analysis-mcp", name="growth"),
                    args={"window_start": "2018", "window_end": "2024"},
                    bindings={"frame_ref": {"from_step": "get_data", "output": "frame_ref"}},
                    input_schema_hash=growth_schema.input_schema_hash,
                    output_schema_hash=growth_schema.output_schema_hash,
                    implementation_version=growth_schema.version,
                ),
            ],
            sources=[
                SelectedSource(spec_id=SPEC_A, sentence="health spending in Kenya", has_data=True)
            ],
            # No evidence bindings: this test is about the wire, not the
            # evidence check, and a hand-built plan carries no real source
            # discovery to check a manifest against.
            evidence_bindings=[],
            terminal_steps=["growth"],
        )
        plan = draft.with_translation(translation=translation, workflow=workflow)
        return {
            "plan": plan.model_dump(mode="json"),
            "summary": {"version": plan.version, "revision": plan_revision(plan)},
        }


async def test_a_planner_seams_own_labels_and_decisions_reach_the_plan_card(
    real_schemas: list[ToolSchema],
) -> None:
    """`plan.proposed`'s steps, decisions, requirements and operations are
    composed by `harness.plan_display` from the seam's real `planner.plan.Plan`,
    not the scripted record's bare tool names. Approving still ends the turn
    cleanly by the revision key, and runs the workflow's steps through
    `execute_approved_plan` against a real `ToolSurface`: every step produces
    an `activity.step` frame with a real `duration_ms`, not merely a subset
    of them.
    """
    resolver = ScriptedResolver([_envelope(choice_points=2)])
    executor = ScriptedExecutor()
    planner = ScriptedPlanner(real_schemas)
    app = _app(
        resolver,
        executor,
        generative=GenerativeNodes(planner=planner),
        tool_surface=_fixture_surface(real_schemas, analysis_tool="growth"),
    )
    async with _client(app) as client:
        first = await _stream(client, "/api/studio/turn", _turn_body("q with forks"))
        plan = _frame(first, "plan.proposed")

        assert [step["label"] for step in plan["steps"]] == [
            "Fetch health spending for Kenya",
            "Growth in health spending, 2018-2024",
        ]
        assert plan["steps"][0]["description"].startswith("Retrieves health spending")
        assert len(plan["revision"]) == 64  # a sha256 hex digest
        assert plan["requirements"][0]["kind"] == "dataset"
        assert plan["operations"][0]["kind"] == "retrieve"
        # `user` was unreachable while the planner hardcoded planner_default.
        assert plan["decisions"][0]["resolved_by"] == "user"

        second = await _stream(
            client, "/api/studio/interrupt", _interrupt_body(plan["interrupt_id"], "approve")
        )

    # The revision key matched, so `approve` routed "approved" and the turn
    # ran to completion with no error -- proof the gate itself works.
    assert _events(second)[-1] == "done"
    # The seam saw the resolve envelope, which is what makes its plan runnable.
    assert planner.calls[0]["envelope"]["status"] == "definite"
    # Every step of the two-step workflow produced its own activity frame,
    # each timed by `_RecordingSurface` -- not merely "some frames exist".
    steps = [
        f
        for f in second
        if f["event"] == "activity.step" and f.get("tool") and f["status"] != "running"
    ]
    assert [step["tool"] for step in steps] == ["get_data", "growth"]
    assert all(step["status"] == "complete" for step in steps)
    assert all(step["duration_ms"] > 0 for step in steps)


_POLLING = "Preparing the dataset (in progress, 12s)"
_RECEIVED = "Result received; checking it"


def _tool_steps(frames: list[dict[str, Any]]) -> list[tuple[str, str, str, str | None]]:
    """Every tool step's `activity.step` frame, as (step_id, tool, status, detail)."""
    return [
        (frame["step_id"], frame["tool"], frame["status"], frame.get("detail"))
        for frame in frames
        if frame["event"] == "activity.step" and frame.get("tool")
    ]


class _PollsOnceSurface(ScriptedSurface):
    """`ScriptedSurface`, whose `get_data` reports once and yields to the
    event loop before it answers, as data-mcp does between two polls."""

    async def call_tool(
        self,
        *,
        tool: ToolId,
        arguments: Mapping[str, Any],
        on_progress: ProgressCallback | None = None,
    ) -> ToolCallResult:
        if tool.name == "get_data" and on_progress is not None:
            on_progress(ProgressReport(progress=12, message=_POLLING))
            await asyncio.sleep(0)
        return await super().call_tool(tool=tool, arguments=arguments)


async def test_a_polled_get_data_step_updates_one_activity_line_until_it_completes() -> None:
    """While `get_data` waits on a background operation, its step streams a
    `running` frame carrying data-mcp's report; once the call answers, the
    line says so, then completes on the same `step_id`. Studio keeps one
    Activity line per step, so the plan's two steps still read as two
    lines."""
    app = _app(
        ScriptedResolver([_envelope(choice_points=2)]),
        ScriptedExecutor(),
        generative=_plan_generative(),
        tool_surface=_PollsOnceSurface(),
    )
    async with _client(app) as client:
        first = await _stream(client, "/api/studio/turn", _turn_body("q with forks"))
        plan = _frame(first, "plan.proposed")
        second = await _stream(
            client, "/api/studio/interrupt", _interrupt_body(plan["interrupt_id"], "approve")
        )

    steps = _tool_steps(second)
    get_data = [step for step in steps if step[1] == "get_data"]
    assert [(status, detail) for _, _, status, detail in get_data] == [
        ("running", _POLLING),
        ("running", _RECEIVED),
        ("complete", "scripted get_data result"),
    ]
    assert len({step_id for step_id, *_ in get_data}) == 1
    plan_lines = {step_id for step_id, tool, *_ in steps if tool in ("get_data", "aggregate")}
    assert len(plan_lines) == len(plan["steps"]) == 2
    assert _events(second)[-1] == "done"


class _ReportingScriptedExecutor(ScriptedExecutor):
    """`ScriptedExecutor`, whose `get_data` sends one progress report first."""

    async def __call__(
        self, *, tool: str, args: dict[str, Any], on_progress: ProgressCallback | None = None
    ) -> dict[str, Any]:
        if tool == "get_data" and on_progress is not None:
            on_progress(ProgressReport(progress=12, message=_POLLING))
        return await super().__call__(tool=tool, args=args, on_progress=on_progress)


async def test_a_direct_fetch_streams_its_report_on_the_line_it_completes() -> None:
    """The no-plan path: `fetch`'s own `get_data` call reports through the
    executor, and its final frame reuses the running frame's `step_id`."""
    app = _app(ScriptedResolver([_envelope()]), _ReportingScriptedExecutor())
    async with _client(app) as client:
        frames = await _stream(client, "/api/studio/turn", _turn_body())

    get_data = [step for step in _tool_steps(frames) if step[1] == "get_data"]
    assert [(status, detail) for _, _, status, detail in get_data] == [
        ("running", _POLLING),
        ("complete", None),
    ]
    assert get_data[0][0] == get_data[1][0]


class _PollsThenHangsSurface(ScriptedSurface):
    """`ScriptedSurface`, whose `get_data` reports once and then never answers."""

    async def call_tool(
        self,
        *,
        tool: ToolId,
        arguments: Mapping[str, Any],
        on_progress: ProgressCallback | None = None,
    ) -> ToolCallResult:
        if tool.name == "get_data":
            if on_progress is not None:
                on_progress(ProgressReport(progress=12, message=_POLLING))
            await asyncio.Event().wait()
        return await super().call_tool(tool=tool, arguments=arguments)


class _BusySurface(ScriptedSurface):
    """`ScriptedSurface`, whose `get_data` is always busy and names a 30s wait."""

    async def call_tool(
        self,
        *,
        tool: ToolId,
        arguments: Mapping[str, Any],
        on_progress: ProgressCallback | None = None,
    ) -> ToolCallResult:
        if tool.name == "get_data":
            refusal = {
                "code": "service_unavailable",
                "message": "the dataset service is busy",
                "retry_mode": "retry_after_delay",
                "retry_after_s": 30,
            }
            return ToolCallResult(is_error=True, structured_content=refusal, text="busy")
        return await super().call_tool(tool=tool, arguments=arguments)


@pytest.mark.parametrize(
    ("surface", "failed_detail"),
    [
        (_PollsThenHangsSurface(), "transport_failed: step 'retrieve' ran past its 0.2s budget"),
        (_BusySurface(), "service_unavailable: the dataset service is busy"),
    ],
    ids=["mid-call", "retry-delay"],
)
async def test_a_step_its_budget_cuts_off_closes_its_activity_line_as_failed(
    monkeypatch: pytest.MonkeyPatch, surface: ToolSurface, failed_detail: str
) -> None:
    """A step whose budget runs out mid-call, or during a retry's delay,
    streamed `running` frames and never returned a result. Its line still
    ends on a `failed` frame before the turn moves on, so Studio shows no
    step waiting after the run is over."""
    monkeypatch.setattr("executor.run.step_budget_s", lambda *_args, **_kwargs: 0.2)
    app = _app(
        ScriptedResolver([_envelope(choice_points=2)]),
        ScriptedExecutor(),
        generative=_plan_generative(),
        tool_surface=surface,
    )
    async with _client(app) as client:
        first = await _stream(client, "/api/studio/turn", _turn_body("q with forks"))
        plan = _frame(first, "plan.proposed")
        second = await _stream(
            client, "/api/studio/interrupt", _interrupt_body(plan["interrupt_id"], "approve")
        )

    get_data = [frame for frame in second if frame.get("tool") == "get_data"]
    assert len({frame["step_id"] for frame in get_data}) == 1
    assert {frame["status"] for frame in get_data[:-1]} == {"running"}
    assert get_data[-1]["status"] == "failed"
    assert get_data[-1]["detail"].startswith(failed_detail)
    assert get_data[-1]["duration_ms"] is not None
    assert second.index(get_data[-1]) < _events(second).index("plan.proposed")


async def test_planner_lines_and_plan_step_lines_never_share_a_step_id_across_an_approval() -> None:
    """Studio resumes an approved turn into the Activity entry the plan was
    drafted in and matches lines by `step_id`, so a plan step's id must
    differ from every planner line's id in the segment before it."""
    app = _app(
        ScriptedResolver([_envelope(choice_points=2)]),
        ScriptedExecutor(),
        generative=GenerativeNodes(planner=ScriptedPlannerModel(), compose=ScriptedCompose()),
        tool_surface=ScriptedSurface(),
    )
    async with _client(app) as client:
        first = await _stream(client, "/api/studio/turn", _turn_body("q with forks"))
        plan = _frame(first, "plan.proposed")
        second = await _stream(
            client, "/api/studio/interrupt", _interrupt_body(plan["interrupt_id"], "approve")
        )

    planner_lines = {
        frame["step_id"]
        for frame in first
        if frame["event"] == "activity.step" and frame["label"] == "Working on a plan"
    }
    step_lines = {
        frame["step_id"]
        for frame in second
        if frame["event"] == "activity.step" and frame.get("tool") in ("get_data", "aggregate")
    }
    assert len(planner_lines) == 1
    assert len(step_lines) == len(plan["steps"]) == 2
    assert planner_lines.isdisjoint(step_lines)


def _running_line(seq: int) -> ActivityStep:
    return ActivityStep(
        event="activity.step",
        seq=seq,
        step_id=f"s{seq}",
        label="Fetch the data",
        status=StepStatus.RUNNING,
        tool="get_data",
        detail=_POLLING,
    )


def _data_frames(chunks: list[str]) -> list[dict[str, Any]]:
    return [json.loads(chunk[len("data: ") :]) for chunk in chunks if chunk.startswith("data: ")]


async def test_a_stop_closes_the_activity_lines_its_stream_left_running() -> None:
    """A Stop ends the stream mid-step. The step's line would otherwise stay
    pending in Studio, so the stream closes it as failed before the
    `cancelled` error frame. A planner line, already timed, stays as sent."""
    planner_line = ActivityStep(
        event="activity.step",
        seq=1,
        step_id="s1",
        label="planner: submit_plan",
        status=StepStatus.RUNNING,
        tool="submit_plan",
        duration_ms=5,
    )

    async def segment():
        yield _running_line(0)
        yield planner_line
        await asyncio.Event().wait()  # the step's call is still waiting

    conversation = ConversationState(conversation_id=CONVERSATION)
    conversation.begin_turn()
    conversation.seq = 2
    stream = _sse_stream(segment(), conversation=conversation, heartbeat_seconds=0.02)
    sent = [await anext(stream), await anext(stream)]
    _end_stream(conversation, "cancelled")
    rest = [chunk async for chunk in stream]

    frames = _data_frames(sent + rest)
    assert [(f["event"], f.get("step_id"), f.get("status")) for f in frames] == [
        ("activity.step", "s0", "running"),
        ("activity.step", "s1", "running"),
        ("activity.step", "s0", "failed"),
        ("error", None, None),
    ]
    assert frames[2]["detail"] == "Stopped before this step finished"
    assert frames[2]["duration_ms"] >= 0
    assert frames[3]["error"]["error"] == "cancelled"
    assert [f["seq"] for f in frames] == sorted({f["seq"] for f in frames})


async def test_a_stream_fatal_error_closes_the_activity_lines_left_running_first() -> None:
    """A turn that fails mid-step (a fetch's MCP timeout, say) ends on an
    `error` frame with its step's line still open; the stream closes the
    line first, and the error frame keeps `seq` rising."""

    async def segment():
        yield _running_line(0)
        yield ErrorFrame(
            event="error",
            seq=1,
            error=ToolError(
                error=ErrorCode.TRANSIENT_UNAVAILABLE,
                retry=RetryMode.RETRY_AFTER_DELAY,
                detail="get_data timed out",
                retry_after_seconds=30,
            ),
        )

    conversation = ConversationState(conversation_id=CONVERSATION)
    conversation.begin_turn()
    conversation.seq = 2
    frames = _data_frames(
        [chunk async for chunk in _sse_stream(segment(), conversation=conversation)]
    )

    assert [(f["event"], f.get("step_id"), f.get("status")) for f in frames] == [
        ("activity.step", "s0", "running"),
        ("activity.step", "s0", "failed"),
        ("error", None, None),
    ]
    assert frames[1]["detail"] == "The turn failed before this step finished"
    assert [f["seq"] for f in frames] == [0, 2, 3]


async def test_a_segment_lost_closes_the_activity_lines_its_stream_left_running() -> None:
    """A takeover ends the stream mid-step, same as a Stop: the step's line
    would otherwise stay pending forever in Studio, since dfo-website
    matches a later frame to an already-open line by `step_id` and patches
    it in place -- nothing ever resends this step's frame on THIS stream.
    The stream closes the line as failed before the stream-fatal
    `transient_unavailable` frame, and leaves the turn itself open, since
    the instance that took the segment over may still finish it."""

    async def segment():
        yield _running_line(0)
        await asyncio.Event().wait()  # the step's call is still waiting

    conversation = ConversationState(conversation_id=CONVERSATION)
    conversation.begin_turn()
    conversation.seq = 1
    stream = _sse_stream(segment(), conversation=conversation, heartbeat_seconds=0.02)
    sent = [await anext(stream)]
    _end_stream(conversation, "segment_lost")
    rest = [chunk async for chunk in stream]

    frames = _data_frames(sent + rest)
    assert [(f["event"], f.get("step_id"), f.get("status")) for f in frames] == [
        ("activity.step", "s0", "running"),
        ("activity.step", "s0", "failed"),
        ("error", None, None),
    ]
    assert (
        frames[1]["detail"] == "the connection to the turn was lost; the step may not have finished"
    )
    assert frames[1]["duration_ms"] >= 0
    assert frames[2]["error"]["error"] == "transient_unavailable"
    assert [f["seq"] for f in frames] == sorted({f["seq"] for f in frames})
    assert conversation.turn_recorded is False  # the turn stays open for the takeover to finish


async def test_amend_yields_plan_versioned_then_a_fresh_proposal() -> None:
    resolver = ScriptedResolver([_envelope(choice_points=2)])
    executor = ScriptedExecutor(
        [
            {"outcome": "ok", "frame_refs": [REF_A]},
            {"outcome": "ok", "frame_refs": [REF_A], "component": COMPONENT},
        ]
    )
    app = _app(resolver, executor, generative=_plan_generative(), tool_surface=ScriptedSurface())
    async with _client(app) as client:
        first = await _stream(client, "/api/studio/turn", _turn_body("q with forks"))
        plan_v1 = _frame(first, "plan.proposed")

        second = await _stream(
            client,
            "/api/studio/interrupt",
            _interrupt_body(plan_v1["interrupt_id"], "amend", "use commitments"),
        )

        events = _events(second)
        assert events[-1] == "plan.proposed"
        versioned = _frame(second, "plan.versioned")
        assert versioned["version"] == 2
        assert versioned["supersedes"] == plan_v1["plan_version_id"]
        plan_v2 = _frame(second, "plan.proposed")
        assert plan_v2["version"] == 2
        assert plan_v2["interrupt_id"] != plan_v1["interrupt_id"]
        assert executor.calls == []

        third = await _stream(
            client, "/api/studio/interrupt", _interrupt_body(plan_v2["interrupt_id"], "approve")
        )

    assert _events(third)[-1] == "done"
    # The plan's own steps ran through `tool_surface`; `executor` only saw
    # `compose`'s describe-then-write.
    assert [c["tool"] for c in executor.calls] == ["describe_frame", "write_component"]


async def test_reject_ends_the_turn_without_computing() -> None:
    resolver = ScriptedResolver([_envelope(choice_points=2)])
    executor = ScriptedExecutor()
    async with _client(_app(resolver, executor)) as client:
        first = await _stream(client, "/api/studio/turn", _turn_body("q with forks"))
        plan = _frame(first, "plan.proposed")

        second = await _stream(
            client, "/api/studio/interrupt", _interrupt_body(plan["interrupt_id"], "reject")
        )

    assert _events(second) == ["message.complete", "done"]
    assert executor.calls == []


class _UnrelatedPlansPlanner:
    """A `PlannerModel` that ignores `prior_plan`: every call starts a fresh
    `planner.plan.Plan` at version 1, for a different measure each time --
    standing in for a seam that reads an amendment as a new question rather
    than a revision. Two calls in one conversation then both draft a
    version-1 plan with nothing else in common, the exact shape that makes
    the bare integer ambiguous across them."""

    def __init__(self, measures: list[str], schemas: list[ToolSchema]) -> None:
        self._measures = iter(measures)
        self.calls: list[dict[str, Any]] = []
        self._schemas = {schema.name: schema for schema in schemas}

    async def __call__(self, *, question: str, **kwargs: Any) -> dict[str, Any]:
        self.calls.append({"question": question, **kwargs})
        measure = next(self._measures)
        decision = Decision(
            slot="window",
            description="Which years?",
            options=["2015-2024", "2018-2024"],
            chosen="2015-2024",
            resolved_by=ResolvedBy.PLANNER_DEFAULT,
            rationale="a full decade covers the series",
        )
        dataset = DatasetRequirement(
            id="spending",
            decision_ids=["window"],
            measure=measure,
            entities=[Entity(name="Kenya", role="recipient")],
            period=Period(start=2015, end=2024),
        )
        growth = GrowthRequirement(
            id="growth",
            decision_ids=["window"],
            input_id="spending",
            window_start=2015,
            window_end=2024,
        )
        draft = PlannerPlan(
            question=question, decisions=[decision], evidence_requirements=[dataset, growth]
        )
        translation = Translation(
            operations=[
                RetrieveOperation(id="get_data", requirement_id=dataset.id, spec_id=SPEC_A),
                GrowthOperation(
                    id="growth",
                    requirement_id=growth.id,
                    input="get_data",
                    window_start=2015,
                    window_end=2024,
                ),
            ],
            terminal=["growth"],
        )
        get_data_schema = self._schemas["get_data"]
        growth_schema = self._schemas["growth"]
        workflow = Workflow(
            steps=[
                WorkflowStep(
                    id="get_data",
                    operation_id="get_data",
                    tool=ToolId(server="data-mcp", name="get_data"),
                    args={"spec_id": SPEC_A},
                    input_schema_hash=get_data_schema.input_schema_hash,
                    output_schema_hash=get_data_schema.output_schema_hash,
                    implementation_version=get_data_schema.version,
                ),
                WorkflowStep(
                    id="growth",
                    operation_id="growth",
                    tool=ToolId(server="analysis-mcp", name="growth"),
                    args={"window_start": "2015", "window_end": "2024"},
                    bindings={"frame_ref": {"from_step": "get_data", "output": "frame_ref"}},
                    input_schema_hash=growth_schema.input_schema_hash,
                    output_schema_hash=growth_schema.output_schema_hash,
                    implementation_version=growth_schema.version,
                ),
            ],
            sources=[SelectedSource(spec_id=SPEC_A, sentence=measure, has_data=True)],
            # No evidence bindings, per `ScriptedPlanner`'s note above.
            evidence_bindings=[],
            terminal_steps=["growth"],
        )
        plan = draft.with_translation(translation=translation, workflow=workflow)
        return {"plan": plan.model_dump(mode="json"), "summary": {"version": plan.version}}


async def test_a_stale_plan_card_cannot_approve_a_later_plan(
    real_schemas: list[ToolSchema],
) -> None:
    """Two unrelated plans can both land at version 1 (`_UnrelatedPlansPlanner`),
    so approval must key off the plan card's own revision, not its version
    number, through the served path. Card 1's interrupt id is consumed by
    the amend that requested card 2, so
    replaying it 404s rather than reaching for whatever plan is now pending,
    and only card 2's own id approves card 2."""
    resolver = ScriptedResolver([_envelope(choice_points=2)])
    executor = ScriptedExecutor()
    planner = _UnrelatedPlansPlanner(["first measure", "second measure"], real_schemas)
    app = _app(
        resolver,
        executor,
        generative=GenerativeNodes(planner=planner),
        tool_surface=_fixture_surface(real_schemas, analysis_tool="growth"),
    )
    async with _client(app) as client:
        first = await _stream(client, "/api/studio/turn", _turn_body("q with forks"))
        card1 = _frame(first, "plan.proposed")
        assert card1["version"] == 1

        second = await _stream(
            client,
            "/api/studio/interrupt",
            _interrupt_body(card1["interrupt_id"], "amend", "a different question entirely"),
        )
        card2 = _frame(second, "plan.proposed")
        assert card2["version"] == 1  # unrelated to card 1, also fresh at version 1
        assert card2["revision"] != card1["revision"]
        assert card2["interrupt_id"] != card1["interrupt_id"]

        # card 1's id was consumed by the amend request; it cannot be replayed
        # to approve whatever plan is pending now (card 2).
        replay = await client.post(
            "/api/studio/interrupt",
            json=_interrupt_body(card1["interrupt_id"], "approve"),
            headers=AUTH,
        )
        assert replay.status_code == 404

        third = await _stream(
            client, "/api/studio/interrupt", _interrupt_body(card2["interrupt_id"], "approve")
        )

    assert _events(third)[-1] == "done"


_BUDGET_EXHAUSTED_TEXT = (
    "This request has used its budget of 4 plan drafts, so nothing was "
    "recorded. Your next message starts with a fresh budget."
)
# `harness.graph`'s `planner` node appends this when it routes a refusal
# back to `approve` with a plan still `PENDING_APPROVAL`: the seam's own
# sentence states facts it can verify from inside one turn; the node adds
# the approvability claim, since only it holds `ctx.state["plan"]`.
_STILL_APPROVABLE_TEXT = (
    f"{_BUDGET_EXHAUSTED_TEXT} Version 1 is still there to approve; anything "
    "past that needs a new question."
)


class _RefusingAmendPlanner:
    """A `PlannerModel` that drafts once for real, then answers every later
    call with `PlannerSeam`'s own `BUDGET_EXHAUSTED` shape -- standing in
    for a real seam whose amendment round hits `MAX_PLAN_ROUNDS`, without
    needing a real model to drive it there."""

    def __init__(self, schemas: list[ToolSchema]) -> None:
        self._schemas = {schema.name: schema for schema in schemas}
        self.calls = 0

    async def __call__(self, *, question: str, **kwargs: Any) -> dict[str, Any]:
        self.calls += 1
        if self.calls > 1:
            return {"refused": {"text": _BUDGET_EXHAUSTED_TEXT}}
        decision = Decision(
            slot="window",
            description="Which years?",
            options=["2015-2024", "2018-2024"],
            chosen="2015-2024",
            resolved_by=ResolvedBy.PLANNER_DEFAULT,
            rationale="a full decade covers the series",
        )
        dataset = DatasetRequirement(
            id="spending",
            decision_ids=["window"],
            measure="health spending",
            entities=[Entity(name="Kenya", role="recipient")],
            period=Period(start=2015, end=2024),
        )
        growth = GrowthRequirement(
            id="growth",
            decision_ids=["window"],
            input_id="spending",
            window_start=2015,
            window_end=2024,
        )
        draft = PlannerPlan(
            question=question, decisions=[decision], evidence_requirements=[dataset, growth]
        )
        translation = Translation(
            operations=[
                RetrieveOperation(id="get_data", requirement_id=dataset.id, spec_id=SPEC_A),
                GrowthOperation(
                    id="growth",
                    requirement_id=growth.id,
                    input="get_data",
                    window_start=2015,
                    window_end=2024,
                ),
            ],
            terminal=["growth"],
        )
        get_data_schema = self._schemas["get_data"]
        growth_schema = self._schemas["growth"]
        workflow = Workflow(
            steps=[
                WorkflowStep(
                    id="get_data",
                    operation_id="get_data",
                    tool=ToolId(server="data-mcp", name="get_data"),
                    args={"spec_id": SPEC_A},
                    input_schema_hash=get_data_schema.input_schema_hash,
                    output_schema_hash=get_data_schema.output_schema_hash,
                    implementation_version=get_data_schema.version,
                ),
                WorkflowStep(
                    id="growth",
                    operation_id="growth",
                    tool=ToolId(server="analysis-mcp", name="growth"),
                    args={"window_start": "2015", "window_end": "2024"},
                    bindings={"frame_ref": {"from_step": "get_data", "output": "frame_ref"}},
                    input_schema_hash=growth_schema.input_schema_hash,
                    output_schema_hash=growth_schema.output_schema_hash,
                    implementation_version=growth_schema.version,
                ),
            ],
            sources=[SelectedSource(spec_id=SPEC_A, sentence="health spending", has_data=True)],
            evidence_bindings=[],
            terminal_steps=["growth"],
        )
        plan = draft.with_translation(translation=translation, workflow=workflow)
        return {"plan": plan.model_dump(mode="json"), "summary": {}}


async def test_a_refused_amendment_still_offers_the_pending_version_for_approval(
    real_schemas: list[ToolSchema],
) -> None:
    """`PlannerSeam`'s own `MAX_PLAN_ROUNDS` cap can refuse an amendment
    outright (`{"refused": {"text": ...}}`). The plan the user was already
    looking at is still `PENDING_APPROVAL` and untouched by that refusal,
    so the served turn re-offers it for approval instead of ending the
    turn on it: the refusal's own sentence streams as a message and a
    fresh `plan.proposed` for the very same version follows, answerable
    with the same revision the first card carried -- not merely a refusal
    sentence that says so.
    """
    resolver = ScriptedResolver([_envelope(choice_points=2)])
    executor = ScriptedExecutor()
    planner = _RefusingAmendPlanner(schemas=real_schemas)
    app = _app(
        resolver,
        executor,
        generative=GenerativeNodes(planner=planner),
        tool_surface=_fixture_surface(real_schemas, analysis_tool="growth"),
    )
    async with _client(app) as client:
        first = await _stream(client, "/api/studio/turn", _turn_body("kenya debt"))
        plan_v1 = _frame(first, "plan.proposed")

        second = await _stream(
            client,
            "/api/studio/interrupt",
            _interrupt_body(plan_v1["interrupt_id"], "amend", "use commitments"),
        )

        # No new version: the refused redraft recorded nothing, so there is
        # nothing to supersede the first card with.
        assert "plan.versioned" not in _events(second)
        message = _frame(second, "message.complete")
        assert message["text"] == _STILL_APPROVABLE_TEXT
        plan_again = _frame(second, "plan.proposed")
        assert plan_again["version"] == plan_v1["version"]
        assert plan_again["revision"] == plan_v1["revision"]
        # A fresh interrupt, even though the plan under it did not change --
        # the one the user was first shown was consumed by the amend answer.
        assert plan_again["interrupt_id"] != plan_v1["interrupt_id"]

        third = await _stream(
            client,
            "/api/studio/interrupt",
            _interrupt_body(plan_again["interrupt_id"], "approve"),
        )

    assert _events(third)[-1] == "done"
    assert executor.calls == []


_QUESTION_FIXED_TEXT = (
    "The plan already on record answers this turn's question, so this "
    "was not recorded as a change to it. Ask the other question as its "
    "own message."
)
_QUESTION_FIXED_STILL_APPROVABLE_TEXT = (
    f"{_QUESTION_FIXED_TEXT} Version 1 is still there to approve; anything "
    "past that needs a new question."
)


class _RestartRefusingPlanner(_RefusingAmendPlanner):
    """`_RefusingAmendPlanner`, refusing every later call the way a real
    `PlannerSeam` -- its pair always built with `question_fixed=True`
    (`harness.planner_seam.PlannerSeam.__init__`) -- refuses an amendment
    that tries to restart for a different question: `QUESTION_FIXED`, not
    `BUDGET_EXHAUSTED`. Records every call's own `question`, so a test can
    prove `ctx.state["question"]` reached the second call unchanged from
    the first."""

    def __init__(self, schemas: list[ToolSchema]) -> None:
        super().__init__(schemas)
        self.questions_seen: list[str] = []

    async def __call__(self, *, question: str, **kwargs: Any) -> dict[str, Any]:
        self.questions_seen.append(question)
        if self.calls > 0:
            self.calls += 1
            return {"refused": {"text": _QUESTION_FIXED_TEXT}}
        return await super().__call__(question=question, **kwargs)


async def test_an_amend_that_restarts_is_refused_with_no_version_regression_or_duplicate_id(
    real_schemas: list[ToolSchema],
) -> None:
    """A restart the graph's `question_fixed` pair refuses (`QUESTION_FIXED`)
    reaches the wire exactly like a `BUDGET_EXHAUSTED` refusal (see
    `test_a_refused_amendment_still_offers_the_pending_version_for_approval`,
    just above): the same pending plan is re-offered, under the same
    `plan_version_id` and `revision` -- proving `_interrupt_frames`
    (`harness.studio_translate`) neither regresses `latest_plan_version` nor
    mints a second wire id for a version it already gave one to -- and no
    `plan.versioned` frame fires, since nothing was recorded to supersede
    the first card with. `ctx.state["question"]` is what the graph derives
    and narrates before the planner ever runs; the refused restart records
    nothing, so the second call sees the same `question` the first one did.
    """
    resolver = ScriptedResolver([_envelope(choice_points=2)])
    executor = ScriptedExecutor()
    planner = _RestartRefusingPlanner(schemas=real_schemas)
    app = _app(
        resolver,
        executor,
        generative=GenerativeNodes(planner=planner),
        tool_surface=_fixture_surface(real_schemas, analysis_tool="growth"),
    )
    async with _client(app) as client:
        first = await _stream(client, "/api/studio/turn", _turn_body("kenya debt"))
        plan_v1 = _frame(first, "plan.proposed")

        second = await _stream(
            client,
            "/api/studio/interrupt",
            _interrupt_body(plan_v1["interrupt_id"], "amend", "what was Kenya's ODA in 2023?"),
        )

        assert "plan.versioned" not in _events(second)
        message = _frame(second, "message.complete")
        assert message["text"] == _QUESTION_FIXED_STILL_APPROVABLE_TEXT
        plan_again = _frame(second, "plan.proposed")
        assert plan_again["version"] == plan_v1["version"]
        assert plan_again["revision"] == plan_v1["revision"]
        assert plan_again["plan_version_id"] == plan_v1["plan_version_id"]
        assert plan_again["interrupt_id"] != plan_v1["interrupt_id"]

        third = await _stream(
            client,
            "/api/studio/interrupt",
            _interrupt_body(plan_again["interrupt_id"], "approve"),
        )

    assert _events(third)[-1] == "done"
    assert executor.calls == []
    assert planner.questions_seen == ["kenya debt", "kenya debt"]


async def test_an_approval_is_recorded_with_its_approver_and_its_revision() -> None:
    """`/interrupt` appends a typed `ApprovalRecord` for every plan decision,
    not only for an approval. `auth_enforced` is `False` because `Settings()`
    here leaves `studio_api_token` unset (the auth-open case); the field
    lets a record from this case be told apart from one written with the
    token gate closed, which `test_an_interrupt_under_a_closed_token_gate_records_auth_enforced`
    below covers."""
    session_service = _new_session_service()
    store = StudioStore(session_service, app_name=AGENT_NAME)
    studio = StudioRunner(
        create_workflow=lambda recorder, surface: create_root_workflow(
            resolver=ScriptedResolver([_envelope(choice_points=2)]),
            executor=recorder,
            generative=GenerativeNodes(planner=ScriptedPlannerModel()),
            tool_surface=surface,
            fetch_series_stats=scripted_fetch_series_stats,
        ),
        executor=ScriptedExecutor(),
        session_service=session_service,
        store=store,
    )
    app = FastAPI()
    app.include_router(create_studio_router(studio, settings=Settings()))

    async with _client(app) as client:
        first = await _stream(client, "/api/studio/turn", _turn_body())
        plan = _frame(first, "plan.proposed")
        await _stream(
            client, "/api/studio/interrupt", _interrupt_body(plan["interrupt_id"], "approve")
        )

    principal = "u_" + hashlib.sha256(b"test-token").hexdigest()[:16]
    conversation = store.current(principal, CONVERSATION)
    assert len(conversation.approvals) == 1
    record = conversation.approvals[0]
    assert record.decision == "approved"
    assert record.approver == principal
    assert record.plan_version == 1
    assert record.revision == plan["revision"]
    assert len(record.revision) == 64  # a sha256 hex digest
    assert record.auth_enforced is False
    assert record.interrupt_id == plan["interrupt_id"]
    assert record.conversation_id == CONVERSATION


async def test_an_interrupt_under_a_closed_token_gate_records_auth_enforced() -> None:
    """The companion to the auth-open case above, driven through the real
    `/api/studio/interrupt` route with `STUDIO_API_TOKEN` set: the `/interrupt`
    handler writes `auth_enforced=settings.studio_api_token is not None` onto
    every `ApprovalRecord` it appends, so a closed gate must show up as
    `True` on the record, not merely as the 401 a wrong token would already
    draw."""
    session_service = _new_session_service()
    store = StudioStore(session_service, app_name=AGENT_NAME)
    studio = StudioRunner(
        create_workflow=lambda recorder, surface: create_root_workflow(
            resolver=ScriptedResolver([_envelope(choice_points=2)]),
            executor=recorder,
            generative=GenerativeNodes(planner=ScriptedPlannerModel()),
            tool_surface=surface,
            fetch_series_stats=scripted_fetch_series_stats,
        ),
        executor=ScriptedExecutor(),
        session_service=session_service,
        store=store,
    )
    app = FastAPI()
    settings = Settings(studio_api_token="test-token")
    app.include_router(create_studio_router(studio, settings=settings))

    async with _client(app) as client:
        first = await _stream(client, "/api/studio/turn", _turn_body())
        plan = _frame(first, "plan.proposed")
        await _stream(
            client, "/api/studio/interrupt", _interrupt_body(plan["interrupt_id"], "approve")
        )

    principal = "u_" + hashlib.sha256(b"test-token").hexdigest()[:16]
    conversation = store.current(principal, CONVERSATION)
    assert len(conversation.approvals) == 1
    record = conversation.approvals[0]
    assert record.auth_enforced is True
    assert record.approver == principal


async def test_a_raise_building_the_approval_record_leaves_the_pending_entry_for_a_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`ApprovalRecord` is `extra="forbid"`; a future field added to it that
    a caller forgets to supply raises where today's fields cannot. If that
    raise happened after `/interrupt` deleted the pending entry, the user's
    decision would be lost with no pending interrupt left to retry against.
    This breaks construction on purpose and checks the entry survives, the
    conversation lock is released rather than left held, and a second,
    unpatched attempt still succeeds."""
    resolver = ScriptedResolver([_envelope(choice_points=2)])
    executor = ScriptedExecutor()
    session_service = _new_session_service()
    store = StudioStore(session_service, app_name=AGENT_NAME)
    studio = StudioRunner(
        create_workflow=lambda recorder, surface: create_root_workflow(
            resolver=resolver,
            executor=recorder,
            generative=GenerativeNodes(planner=ScriptedPlannerModel()),
            tool_surface=surface,
            fetch_series_stats=scripted_fetch_series_stats,
        ),
        executor=executor,
        session_service=session_service,
        store=store,
    )
    app = FastAPI()
    app.include_router(create_studio_router(studio, settings=Settings()))

    def _boom(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("approval record construction failed")

    async with _client(app) as client:
        first = await _stream(client, "/api/studio/turn", _turn_body())
        plan = _frame(first, "plan.proposed")

        with monkeypatch.context() as patch:
            patch.setattr("harness.studio.ApprovalRecord", _boom)
            with pytest.raises(RuntimeError, match="approval record construction failed"):
                await _stream(
                    client,
                    "/api/studio/interrupt",
                    _interrupt_body(plan["interrupt_id"], "approve"),
                )

        principal = "u_" + hashlib.sha256(b"test-token").hexdigest()[:16]
        conversation = store.current(principal, CONVERSATION)
        assert plan["interrupt_id"] in conversation.pending
        assert conversation.approvals == []
        assert not conversation.lock.locked()

        await _stream(
            client, "/api/studio/interrupt", _interrupt_body(plan["interrupt_id"], "approve")
        )

    # The retry reaches `ApprovalRecord` unpatched: the record is appended
    # and the pending entry it was keyed on is finally consumed. (This
    # setup has no `tool_surface`, so the resumed turn's own execution
    # refuses downstream -- irrelevant to what this test is proving.)
    assert len(conversation.approvals) == 1
    assert plan["interrupt_id"] not in conversation.pending


# --- the clarify flow -------------------------------------------------------------


async def test_candidates_elicit_and_the_answer_resumes() -> None:
    resolver = ScriptedResolver(
        [
            _envelope(
                status="candidates",
                spec_id=None,
                candidates=[
                    {"spec_id": SPEC_A, "label": "reading A"},
                    {"spec_id": SPEC_B, "label": "reading B"},
                ],
            ),
            _envelope(spec_id=SPEC_B),
        ]
    )
    executor = ScriptedExecutor()
    async with _client(_app(resolver, executor)) as client:
        first = await _stream(client, "/api/studio/turn", _turn_body("ambiguous q"))

        assert _events(first)[-1] == "elicitation.request"
        elicitation = _frame(first, "elicitation.request")
        assert elicitation["input"] == "single_select"
        assert [option["id"] for option in elicitation["options"]] == [SPEC_A, SPEC_B]

        second = await _stream(
            client,
            "/api/studio/interrupt",
            _interrupt_body(elicitation["interrupt_id"], "answer", SPEC_B),
        )

    assert _events(second)[-1] == "done"
    assert resolver.calls[-1]["spec_id"] == SPEC_B


async def test_stop_during_a_pause_invalidates_the_ask_it_left_behind() -> None:
    """POST /cancel while the turn is paused on an elicitation: by then the
    stream that emitted the pause has already ended, so `cancel_event` is
    back to `None` and cannot be the signal cancel acts on. The fix must
    still refuse a later `/interrupt` on that id, and must not resume graph
    execution -- proven here by the resolver never seeing the second call
    `spec_id=SPEC_B` would have produced."""
    resolver = ScriptedResolver(
        [
            _envelope(
                status="candidates",
                spec_id=None,
                candidates=[
                    {"spec_id": SPEC_A, "label": "reading A"},
                    {"spec_id": SPEC_B, "label": "reading B"},
                ],
            ),
            _envelope(spec_id=SPEC_B),
        ]
    )
    executor = ScriptedExecutor()
    async with _client(_app(resolver, executor)) as client:
        first = await _stream(client, "/api/studio/turn", _turn_body("ambiguous q"))
        elicitation = _frame(first, "elicitation.request")

        # The pause already ended the stream that armed cancel_event, same
        # as production: a Stop press only ever reaches this route after
        # the SSE connection that showed the ask is long closed.
        cancel = await client.post(
            "/api/studio/cancel",
            json={"kind": "turn.cancel", "schema_version": 1, "conversation_id": CONVERSATION},
            headers=AUTH,
        )
        assert cancel.status_code == 204

        replay = await client.post(
            "/api/studio/interrupt",
            json=_interrupt_body(elicitation["interrupt_id"], "answer", SPEC_B),
            headers=AUTH,
        )

    assert replay.status_code == 404
    assert replay.json()["error"] == "not_found"
    assert len(resolver.calls) == 1  # the resumed call never happened


async def test_a_raise_building_the_elicitation_frame_leaves_no_orphaned_pending_entry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The elicitation branch of `_interrupt_frames` writes
    `conversation.pending[wire_id]` only after `ElicitationRequest` has been
    built. Written any earlier, a raise there would register an interrupt
    the client was never shown: an orphan, not a loss -- the turn ends on
    an `error` frame with no `interrupt_id` for the client to ever answer
    with, so the entry would sit in `pending` unresolvable. This breaks
    construction on purpose and checks nothing is left behind, and that a
    later, unpatched turn still reaches a real elicitation clean."""
    resolver = ScriptedResolver(
        [
            _envelope(
                status="candidates",
                spec_id=None,
                candidates=[
                    {"spec_id": SPEC_A, "label": "reading A"},
                    {"spec_id": SPEC_B, "label": "reading B"},
                ],
            ),
        ]
    )
    executor = ScriptedExecutor()
    session_service = _new_session_service()
    store = StudioStore(session_service, app_name=AGENT_NAME)
    studio = StudioRunner(
        create_workflow=lambda recorder, surface: create_root_workflow(
            resolver=resolver,
            executor=recorder,
            tool_surface=surface,
            fetch_series_stats=scripted_fetch_series_stats,
        ),
        executor=executor,
        session_service=session_service,
        store=store,
    )
    app = FastAPI()
    app.include_router(create_studio_router(studio, settings=Settings()))

    def _boom(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("elicitation frame construction failed")

    async with _client(app) as client:
        with monkeypatch.context() as patch:
            patch.setattr("harness.studio_translate.ElicitationRequest", _boom)
            first = await _stream(client, "/api/studio/turn", _turn_body("ambiguous q"))

        # Caught by `StudioRunner._frames`'s `except Exception` clause and
        # serialised into the stream rather than raised through the
        # transport (SSE headers are already on the wire by then).
        assert _events(first)[-1] == "error"

        principal = "u_" + hashlib.sha256(b"test-token").hexdigest()[:16]
        conversation = store.current(principal, CONVERSATION)
        assert conversation.pending == {}
        assert not conversation.lock.locked()

        second = await _stream(client, "/api/studio/turn", _turn_body("ambiguous q"))

    assert _events(second)[-1] == "elicitation.request"
    elicitation = _frame(second, "elicitation.request")
    assert elicitation["interrupt_id"] in conversation.pending


# --- interrupt bookkeeping ---------------------------------------------------------


async def test_an_unknown_interrupt_id_is_a_404_tool_error() -> None:
    """A known conversation with no such pending interrupt -- distinct from
    `test_an_unknown_conversation_is_a_404_on_interrupt` below, which the
    store never held at all."""
    async with _client(_app(ScriptedResolver([_envelope()]), ScriptedExecutor())) as client:
        await _stream(client, "/api/studio/turn", _turn_body())
        response = await client.post(
            "/api/studio/interrupt",
            json=_interrupt_body("int_nonexistent", "approve"),
            headers=AUTH,
        )
    assert response.status_code == 404
    assert response.json()["error"] == "not_found"


async def test_an_unknown_conversation_is_a_404_on_interrupt() -> None:
    async with _client(_app(ScriptedResolver([_envelope()]), ScriptedExecutor())) as client:
        response = await client.post(
            "/api/studio/interrupt",
            json=_interrupt_body("int_nonexistent", "approve"),
            headers=AUTH,
        )
    assert response.status_code == 404
    body = response.json()
    assert body["error"] == "not_found"
    assert body["pointer"] == "/conversation_id"


async def test_an_unknown_conversation_is_a_404_on_turn_with_a_component_ref() -> None:
    """A `/turn` naming a `component_ref` on a conversation the store never
    held refers to state the server never had -- refused rather than
    silently starting a new, empty conversation under it."""
    async with _client(_app(ScriptedResolver([_envelope()]), ScriptedExecutor())) as client:
        body = _turn_body()
        body["component_ref"] = {"intent": "follow_up", "component_id": "cmp_ghost", "revision": 1}
        response = await client.post("/api/studio/turn", json=body, headers=AUTH)
    assert response.status_code == 404
    payload = response.json()
    assert payload["error"] == "not_found"
    assert payload["pointer"] == "/conversation_id"


async def test_a_wire_interrupt_id_is_single_use() -> None:
    resolver = ScriptedResolver([_envelope(choice_points=2)])
    async with _client(_app(resolver, ScriptedExecutor())) as client:
        first = await _stream(client, "/api/studio/turn", _turn_body("q with forks"))
        plan = _frame(first, "plan.proposed")
        await _stream(
            client, "/api/studio/interrupt", _interrupt_body(plan["interrupt_id"], "approve")
        )

        replay = await client.post(
            "/api/studio/interrupt",
            json=_interrupt_body(plan["interrupt_id"], "approve"),
            headers=AUTH,
        )
    assert replay.status_code == 404


async def test_answering_a_plan_interrupt_is_a_422_binding_error() -> None:
    resolver = ScriptedResolver([_envelope(choice_points=2)])
    async with _client(_app(resolver, ScriptedExecutor())) as client:
        first = await _stream(client, "/api/studio/turn", _turn_body("q with forks"))
        plan = _frame(first, "plan.proposed")
        response = await client.post(
            "/api/studio/interrupt",
            json=_interrupt_body(plan["interrupt_id"], "answer", SPEC_A),
            headers=AUTH,
        )
    assert response.status_code == 422
    body = response.json()
    assert body["error"] == "binding_invalid"
    assert "approve" in body["fix"]


class _HangingSurface(ScriptedSurface):
    """`ScriptedSurface`, but its first `call_tool` hangs forever -- the
    resumed segment's crash point every test below abandons A's resume at:
    after `/interrupt`'s own save (the segment claimed, the entry marked
    resuming, the answer recorded) has already landed, but before the
    resumed segment's first plan step -- let alone its own checkpoint --
    completes."""

    def __init__(self) -> None:
        self.entered = asyncio.Event()

    async def call_tool(
        self,
        *,
        tool: ToolId,
        arguments: Mapping[str, Any],
        on_progress: ProgressCallback | None = None,
    ) -> ToolCallResult:
        self.entered.set()
        await asyncio.Event().wait()
        raise AssertionError("unreachable")


async def _noop_release(*args: Any, **kwargs: Any) -> None:
    """Stands in for `StudioStore.release`: a crashed instance's segment
    lease and resuming pending entries are never cleared or dropped --
    exactly what a real crash leaves behind, since nothing runs a
    process's `finally` blocks when it dies."""


async def _approve_plan_then_hang(
    client: httpx2.AsyncClient, app: FastAPI, plan: dict[str, Any]
) -> _HangingSurface:
    """Start `/interrupt approve`'s resume stream, let it reach its first
    (scripted) plan step, and abandon it there with `store.release` patched
    to a no-op -- `POST /cancel` ends the stream (through the same local
    fast path a Stop press uses) without ever persisting the segment's
    release, standing in for a crash between `/interrupt`'s save and the
    resumed segment's own checkpoint. Returns once the abandoned stream's
    `error` frame has been read off the wire."""
    hanging = _HangingSurface()
    app.state.studio.surface_recorder._surface = hanging
    interrupt_task = asyncio.ensure_future(
        _stream(client, "/api/studio/interrupt", _interrupt_body(plan["interrupt_id"], "approve"))
    )
    await asyncio.wait_for(hanging.entered.wait(), timeout=5)
    app.state.studio.store.release = _noop_release  # type: ignore[method-assign]
    cancel = await client.post(
        "/api/studio/cancel",
        json={"kind": "turn.cancel", "schema_version": 1, "conversation_id": CONVERSATION},
        headers=AUTH,
    )
    assert cancel.status_code == 204
    crashed = await asyncio.wait_for(interrupt_task, timeout=5)
    assert _events(crashed)[-1] == "error"
    return hanging


async def test_a_crash_between_interrupts_save_and_the_resume_leaves_a_retryable_answer(
    tmp_path: Path,
) -> None:
    """A's resume of a plan approval crashes (`_approve_plan_then_hang`)
    strictly after `/interrupt`'s own save landed -- the segment claimed,
    the entry marked resuming, the one `ApprovalRecord` appended -- but
    before the resumed segment's first plan step, let alone its own
    checkpoint, completes. B's same answer is refused while A's segment
    lease is still live, and resumes the turn once it expires (an injected
    clock), without a second `ApprovalRecord` and past every `seq` A's dead
    attempt may already have sent."""
    clock = [1_000.0]
    principal = "u_" + hashlib.sha256(b"test-token").hexdigest()[:16]
    session_service = sqlite_session_service(tmp_path / "shared.db")
    store_a = StudioStore(session_service, app_name=AGENT_NAME, clock=lambda: clock[0])
    store_b = StudioStore(session_service, app_name=AGENT_NAME, clock=lambda: clock[0])
    app_a = _app(
        ScriptedResolver([_envelope(choice_points=2)]),
        ScriptedExecutor(),
        generative=_plan_generative(),
        tool_surface=ScriptedSurface(),
        session_service=session_service,
        store=store_a,
    )

    async with _client(app_a) as client_a:
        first = await _stream(client_a, "/api/studio/turn", _turn_body("q with forks"))
        plan = _frame(first, "plan.proposed")

        probe = StudioStore(session_service, app_name=AGENT_NAME, clock=lambda: clock[0])
        saved_seq = (await probe.load(principal, CONVERSATION)).seq

        await _approve_plan_then_hang(client_a, app_a, plan)

    app_b = _app(
        ScriptedResolver([_envelope()]),
        ScriptedExecutor(
            [
                {"outcome": "ok", "frame_refs": [REF_A]},
                {"outcome": "ok", "frame_refs": [REF_A], "component": COMPONENT},
            ]
        ),
        generative=_plan_generative(),
        tool_surface=ScriptedSurface(),
        session_service=session_service,
        store=store_b,
    )
    async with _client(app_b) as client_b:
        busy = await client_b.post(
            "/api/studio/interrupt",
            json=_interrupt_body(plan["interrupt_id"], "approve"),
            headers=AUTH,
        )
        assert busy.status_code == 409
        busy_body = busy.json()
        assert busy_body["error"] == "transient_unavailable"
        assert 0 < busy_body["retry_after_seconds"] <= studio_state_module.SEGMENT_TTL_S

        clock[0] += studio_state_module.SEGMENT_TTL_S + 1  # A's segment has expired

        resumed = await _stream(
            client_b, "/api/studio/interrupt", _interrupt_body(plan["interrupt_id"], "approve")
        )

    assert _events(resumed)[-1] == "done"
    assert resumed[0]["seq"] >= saved_seq + 1000

    reloaded = await StudioStore(session_service, app_name=AGENT_NAME).load(principal, CONVERSATION)
    assert reloaded is not None
    assert len(reloaded.approvals) == 1
    assert plan["interrupt_id"] not in reloaded.pending


async def test_a_different_answer_after_a_crashed_resume_expires_is_refused(
    tmp_path: Path,
) -> None:
    """The same crash as above, but B's retry disagrees with the answer
    that actually resumed the turn: refused for good, even once A's
    segment has expired and a matching retry would have succeeded."""
    clock = [1_000.0]
    session_service = sqlite_session_service(tmp_path / "shared.db")
    store_a = StudioStore(session_service, app_name=AGENT_NAME, clock=lambda: clock[0])
    store_b = StudioStore(session_service, app_name=AGENT_NAME, clock=lambda: clock[0])
    app_a = _app(
        ScriptedResolver([_envelope(choice_points=2)]),
        ScriptedExecutor(),
        generative=_plan_generative(),
        tool_surface=ScriptedSurface(),
        session_service=session_service,
        store=store_a,
    )

    async with _client(app_a) as client_a:
        first = await _stream(client_a, "/api/studio/turn", _turn_body("q with forks"))
        plan = _frame(first, "plan.proposed")
        await _approve_plan_then_hang(client_a, app_a, plan)

    app_b = _app(
        ScriptedResolver([_envelope()]),
        ScriptedExecutor(),
        generative=_plan_generative(),
        tool_surface=ScriptedSurface(),
        session_service=session_service,
        store=store_b,
    )
    async with _client(app_b) as client_b:
        clock[0] += studio_state_module.SEGMENT_TTL_S + 1  # A's segment has expired

        mismatched = await client_b.post(
            "/api/studio/interrupt",
            json=_interrupt_body(plan["interrupt_id"], "reject"),
            headers=AUTH,
        )

    assert mismatched.status_code == 409
    body = mismatched.json()
    assert body["error"] == "edit_conflict"

    principal = "u_" + hashlib.sha256(b"test-token").hexdigest()[:16]
    reloaded = await StudioStore(session_service, app_name=AGENT_NAME).load(principal, CONVERSATION)
    assert reloaded is not None
    assert plan["interrupt_id"] in reloaded.pending
    assert reloaded.pending[plan["interrupt_id"]].answer == {"decision": "approve", "payload": None}


async def test_two_concurrent_first_answers_to_one_interrupt_give_one_stream_and_one_409(
    tmp_path: Path,
) -> None:
    """A and B both answer the SAME fresh interrupt at once: whichever
    instance's claim save lands first resumes the turn; the other's save
    finds the ledger already moved and answers the existing 409, rather
    than each resuming its own segment of the turn."""
    session_service = sqlite_session_service(tmp_path / "shared.db")
    app_a = _app(
        ScriptedResolver([_envelope(choice_points=2)]),
        ScriptedExecutor(),
        generative=_plan_generative(),
        tool_surface=ScriptedSurface(),
        session_service=session_service,
    )
    app_b = _app(
        ScriptedResolver([_envelope()]),
        ScriptedExecutor(),
        generative=_plan_generative(),
        tool_surface=ScriptedSurface(),
        session_service=session_service,
    )

    async with _client(app_a) as client_a:
        first = await _stream(client_a, "/api/studio/turn", _turn_body("q with forks"))
        plan = _frame(first, "plan.proposed")

    async with _client(app_a) as client_a, _client(app_b) as client_b:
        task_a = asyncio.ensure_future(
            client_a.post(
                "/api/studio/interrupt",
                json=_interrupt_body(plan["interrupt_id"], "approve"),
                headers=AUTH,
            )
        )
        task_b = asyncio.ensure_future(
            client_b.post(
                "/api/studio/interrupt",
                json=_interrupt_body(plan["interrupt_id"], "approve"),
                headers=AUTH,
            )
        )
        responses = await asyncio.gather(task_a, task_b)

    statuses = sorted(response.status_code for response in responses)
    assert statuses == [200, 409]
    conflict = next(response for response in responses if response.status_code == 409)
    assert conflict.json()["error"] == "transient_unavailable"


async def test_a_resumed_segments_stream_fatal_error_leaves_no_resuming_entry(
    tmp_path: Path,
) -> None:
    """A resumed segment that ends in a stream-fatal error (not a crash)
    drops its resuming entry through `release`'s own `drop_resuming`, the
    same as a clean `done`, so a later retry of the same answer is refused
    the same way an unknown interrupt id is."""
    resolver = ScriptedResolver([_envelope(choice_points=2)])

    class _BoomExecutor(ScriptedExecutor):
        async def __call__(
            self, *, tool: str, args: dict[str, Any], on_progress: ProgressCallback | None = None
        ) -> dict[str, Any]:
            raise RuntimeError("compose call failed")

    app = _app(
        resolver,
        _BoomExecutor(),
        generative=_plan_generative(),
        tool_surface=ScriptedSurface(),
    )
    async with _client(app) as client:
        first = await _stream(client, "/api/studio/turn", _turn_body("q with forks"))
        plan = _frame(first, "plan.proposed")

        resumed = await _stream(
            client, "/api/studio/interrupt", _interrupt_body(plan["interrupt_id"], "approve")
        )
        assert _events(resumed)[-1] == "error"

        retry = await client.post(
            "/api/studio/interrupt",
            json=_interrupt_body(plan["interrupt_id"], "approve"),
            headers=AUTH,
        )

    assert retry.status_code == 404
    assert retry.json()["error"] == "not_found"


# --- the composition routes ---------------------------------------------------------


async def _conversation_with_component(client: httpx2.AsyncClient) -> dict[str, Any]:
    """Run one plan turn that creates COMPONENT; return the final manifest."""
    frames = await _stream(client, "/api/studio/turn", _turn_body("q with forks"))
    plan = _frame(frames, "plan.proposed")
    resumed = await _stream(
        client, "/api/studio/interrupt", _interrupt_body(plan["interrupt_id"], "approve")
    )
    return _frame(resumed, "done")["composition"]


def _mutation_body(manifest: dict[str, Any], operations: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "kind": "composition.mutate",
        "schema_version": 1,
        "conversation_id": CONVERSATION,
        "composition_id": manifest["composition_id"],
        "expected_revision": manifest["revision"],
        "operations": operations,
    }


async def test_pin_mints_a_user_revision() -> None:
    resolver = ScriptedResolver([_envelope(choice_points=2)])
    executor = ScriptedExecutor(
        [
            {"outcome": "ok", "frame_refs": [REF_A]},
            {"outcome": "ok", "frame_refs": [REF_A], "component": COMPONENT},
        ]
    )
    app = _app(resolver, executor, generative=_plan_generative(), tool_surface=ScriptedSurface())
    async with _client(app) as client:
        manifest = await _conversation_with_component(client)
        response = await client.post(
            "/api/studio/composition",
            json=_mutation_body(manifest, [{"op": "pin", "component_id": "cmp_01test0001"}]),
            headers=AUTH,
        )

    assert response.status_code == 200
    updated = response.json()
    assert updated["revision"] == manifest["revision"] + 1
    assert updated["actor"] == "user"
    assert updated["blocks"][0]["pinned"] is True


async def test_a_stale_expected_revision_is_an_edit_conflict_and_applies_nothing() -> None:
    resolver = ScriptedResolver([_envelope(choice_points=2)])
    executor = ScriptedExecutor(
        [
            {"outcome": "ok", "frame_refs": [REF_A]},
            {"outcome": "ok", "frame_refs": [REF_A], "component": COMPONENT},
        ]
    )
    app = _app(resolver, executor, generative=_plan_generative(), tool_surface=ScriptedSurface())
    async with _client(app) as client:
        manifest = await _conversation_with_component(client)
        body = _mutation_body(manifest, [{"op": "remove", "component_id": "cmp_01test0001"}])
        body["expected_revision"] = manifest["revision"] + 41

        conflict = await client.post("/api/studio/composition", json=body, headers=AUTH)

        assert conflict.status_code == 409
        assert conflict.json()["error"] == "edit_conflict"
        # Nothing applied: the block is still there at the same revision.
        retry = await client.post(
            "/api/studio/composition",
            json=_mutation_body(manifest, [{"op": "pin", "component_id": "cmp_01test0001"}]),
            headers=AUTH,
        )
    assert retry.status_code == 200


async def test_two_same_instance_compositions_give_one_success_and_one_conflict() -> None:
    """Two `/composition` calls target the SAME conversation and instance,
    both against the current revision. A's `mutate_composition` call is
    delayed (`gated_mutate_composition`) until after B's whole request has
    run, and the composition lock serializes them so the two never
    interleave at `load`. One call succeeds; the other's `expected_revision`
    is stale, so it returns `edit_conflict` rather than a second silent 200
    that drops one edit."""
    resolver = ScriptedResolver([_envelope(choice_points=2)])
    executor = ScriptedExecutor(
        [
            {"outcome": "ok", "frame_refs": [REF_A]},
            {"outcome": "ok", "frame_refs": [REF_A], "component": COMPONENT},
        ]
    )
    app = _app(resolver, executor, generative=_plan_generative(), tool_surface=ScriptedSurface())
    async with _client(app) as client:
        manifest = await _conversation_with_component(client)

        real_mutate_composition = app.state.studio.store.mutate_composition
        a_paused = asyncio.Event()
        release_a = asyncio.Event()
        calls = 0

        async def gated_mutate_composition(*args: Any, **kwargs: Any) -> Any:
            nonlocal calls
            calls += 1
            if calls == 1:
                a_paused.set()
                await release_a.wait()
            return await real_mutate_composition(*args, **kwargs)

        app.state.studio.store.mutate_composition = gated_mutate_composition

        task_a = asyncio.ensure_future(
            client.post(
                "/api/studio/composition",
                json=_mutation_body(manifest, [{"op": "pin", "component_id": "cmp_01test0001"}]),
                headers=AUTH,
            )
        )
        await asyncio.wait_for(a_paused.wait(), timeout=5)

        task_b = asyncio.ensure_future(
            client.post(
                "/api/studio/composition",
                json=_mutation_body(manifest, [{"op": "remove", "component_id": "cmp_01test0001"}]),
                headers=AUTH,
            )
        )
        await asyncio.sleep(0.05)  # let B run as far as it can without A's release

        release_a.set()
        response_a = await asyncio.wait_for(task_a, timeout=5)
        response_b = await asyncio.wait_for(task_b, timeout=5)

    statuses = sorted([response_a.status_code, response_b.status_code])
    assert statuses == [200, 409]
    conflict = response_a if response_a.status_code == 409 else response_b
    assert conflict.json()["error"] == "edit_conflict"


async def test_a_persistently_stale_composition_is_an_edit_conflict_and_applies_nothing(
    tmp_path: Path,
) -> None:
    """A second store's own write lands first on every one of
    `mutate_composition`'s bounded retries, so all of them go stale: the
    route answers 409 rather than retrying forever, and the live
    conversation is left holding the last manifest it fetched, not the
    caller's rejected pin."""
    principal = "u_" + hashlib.sha256(b"test-token").hexdigest()[:16]
    session_service = sqlite_session_service(tmp_path / "shared.db")
    resolver = ScriptedResolver([_envelope(choice_points=2)])
    executor = ScriptedExecutor(
        [
            {"outcome": "ok", "frame_refs": [REF_A]},
            {"outcome": "ok", "frame_refs": [REF_A], "component": COMPONENT},
        ]
    )
    app = _app(
        resolver,
        executor,
        generative=_plan_generative(),
        tool_surface=ScriptedSurface(),
        session_service=session_service,
    )
    store_b = StudioStore(session_service, app_name=AGENT_NAME)

    async with _client(app) as client:
        manifest = await _conversation_with_component(client)

        real_append_event = session_service.append_event
        attempts = 0

        async def append_event_then_interfere(session: Any, event: Any) -> Any:
            # An unrelated write from a second store lands between every
            # fetch this route's retry makes and its own append, so the
            # append itself always finds the session moved on -- exactly
            # what a genuinely losing race looks like, whichever key the
            # other writer happened to touch.
            nonlocal attempts
            attempts += 1
            session_service.append_event = real_append_event
            try:
                await store_b.load(principal, CONVERSATION)
                store_b.current(principal, CONVERSATION).seq = attempts
                await store_b.save(principal, CONVERSATION)
            finally:
                session_service.append_event = append_event_then_interfere
            return await real_append_event(session, event)

        session_service.append_event = append_event_then_interfere

        response = await client.post(
            "/api/studio/composition",
            json=_mutation_body(manifest, [{"op": "pin", "component_id": "cmp_01test0001"}]),
            headers=AUTH,
        )

    assert response.status_code == 409
    assert response.json()["error"] == "edit_conflict"
    assert attempts == studio_state_module._COMPOSITION_MUTATE_RETRY_ATTEMPTS
    live = app.state.studio.store.current(principal, CONVERSATION)
    assert live.composition.blocks[0].pinned is False
    reloaded = await StudioStore(session_service, app_name=AGENT_NAME).load(principal, CONVERSATION)
    assert reloaded.composition.blocks[0].pinned is False


async def test_an_unknown_composition_id_is_a_404_tool_error() -> None:
    """A known conversation with no such composition -- distinct from
    `test_an_unknown_conversation_is_a_404_on_composition` below, which the
    store never held at all."""
    async with _client(_app(ScriptedResolver([_envelope()]), ScriptedExecutor())) as client:
        await _stream(client, "/api/studio/turn", _turn_body())
        response = await client.post(
            "/api/studio/composition",
            json={
                "kind": "composition.mutate",
                "schema_version": 1,
                "conversation_id": CONVERSATION,
                "composition_id": "cps_nonexistent",
                "operations": [{"op": "pin", "component_id": "cmp_x"}],
            },
            headers=AUTH,
        )
    assert response.status_code == 404
    assert response.json()["error"] == "not_found"


async def test_an_unknown_conversation_is_a_404_on_composition() -> None:
    async with _client(_app(ScriptedResolver([_envelope()]), ScriptedExecutor())) as client:
        response = await client.post(
            "/api/studio/composition",
            json={
                "kind": "composition.mutate",
                "schema_version": 1,
                "conversation_id": CONVERSATION,
                "composition_id": "cps_nonexistent",
                "operations": [{"op": "pin", "component_id": "cmp_x"}],
            },
            headers=AUTH,
        )
    assert response.status_code == 404
    body = response.json()
    assert body["error"] == "not_found"
    assert body["pointer"] == "/conversation_id"


# --- failures reach the stream as error frames ---------------------------------------


async def test_a_crashing_resolver_becomes_a_stream_fatal_error_frame() -> None:
    class BrokenResolver:
        async def __call__(
            self, *, question: str, spec_id: str | None = None, wide: bool = False
        ) -> dict[str, Any]:
            raise RuntimeError("resolver exploded")

    async with _client(_app(BrokenResolver(), ScriptedExecutor())) as client:  # type: ignore[arg-type]
        frames = await _stream(client, "/api/studio/turn", _turn_body())

    assert _events(frames)[-1] == "error"
    error = _frame(frames, "error")["error"]
    assert error["error"] == "transient_unavailable"
    assert "done" not in _events(frames)


def _timeout_error(tool: str, arguments: dict[str, Any] | None = None) -> MCPError:
    """What `mcp_client._call_tool` raises when a per-tool deadline elapses:
    the same `MCPError` the transport raises, tagged with which tool hit it
    and a key of the call itself (tool + its own arguments) -- the same
    shape `mcp_client._McpTimeoutError`/`_canonical_arguments` build, so
    these tests exercise `_stream_fatal`'s real repeat logic."""
    error = MCPError(code=REQUEST_TIMEOUT, message=f"{tool} timed out")
    error.timeout_tool = tool  # type: ignore[attr-defined]
    canonical = json.dumps(arguments or {}, sort_keys=True, separators=(",", ":"), default=str)
    error.timeout_key = f"{tool}:{canonical}"  # type: ignore[attr-defined]
    return error


async def test_a_first_search_timeout_is_transient_and_names_the_tool() -> None:
    class TimeoutResolver:
        async def __call__(
            self, *, question: str, spec_id: str | None = None, wide: bool = False
        ) -> dict[str, Any]:
            raise _timeout_error("search", {"question": question})

    async with _client(_app(TimeoutResolver(), ScriptedExecutor())) as client:  # type: ignore[arg-type]
        frames = await _stream(client, "/api/studio/turn", _turn_body())

    error = _frame(frames, "error")["error"]
    assert error["error"] == "transient_unavailable"
    assert error["retry"] == "retry_after_delay"
    assert error["retry_after_seconds"] == 30
    assert "search" in error["detail"]
    assert "timed out" in error["detail"]


async def test_a_repeat_search_timeout_in_the_same_conversation_stops_advising_the_wait() -> None:
    """The first timeout on a call advises the pinned wait; the SAME call
    (same tool, same question) timing out again, later in the same
    conversation, says that wait already bought nothing instead of
    repeating the advice."""

    class TimeoutResolver:
        async def __call__(
            self, *, question: str, spec_id: str | None = None, wide: bool = False
        ) -> dict[str, Any]:
            raise _timeout_error("search", {"question": question})

    app = _app(TimeoutResolver(), ScriptedExecutor())  # type: ignore[arg-type]
    async with _client(app) as client:
        first = await _stream(client, "/api/studio/turn", _turn_body())
        second = await _stream(
            client, "/api/studio/turn", _turn_body()
        )  # same question -> a repeat

    first_error = _frame(first, "error")["error"]
    assert first_error["retry"] == "retry_after_delay"

    second_error = _frame(second, "error")["error"]
    assert second_error["error"] == "transient_unavailable"
    assert second_error["retry"] == "do_not_retry"
    assert second_error.get("retry_after_seconds") is None
    assert "did not help" in second_error["detail"]


async def test_a_repeat_tool_with_different_arguments_is_not_a_repeat() -> None:
    """The repeat memory is keyed by the call's own arguments too: a
    DIFFERENT search timing out after an earlier one, in the same
    conversation, is evidence the service itself is slow -- still worth
    advising the pinned wait -- not proof that waiting doesn't help THIS
    query."""

    class TimeoutResolver:
        async def __call__(
            self, *, question: str, spec_id: str | None = None, wide: bool = False
        ) -> dict[str, Any]:
            raise _timeout_error("search", {"question": question})

    app = _app(TimeoutResolver(), ScriptedExecutor())  # type: ignore[arg-type]
    async with _client(app) as client:
        await _stream(client, "/api/studio/turn", _turn_body("kenya oda 2023"))
        second = await _stream(client, "/api/studio/turn", _turn_body("mali oda 2024"))

    second_error = _frame(second, "error")["error"]
    assert second_error["retry"] == "retry_after_delay"
    assert second_error["retry_after_seconds"] == 30


async def test_a_timeout_on_a_different_tool_is_not_a_repeat() -> None:
    """The per-conversation timeout memory is keyed by tool (and
    arguments): a `get_data` timeout after a `search` timeout is still a
    first occurrence for `get_data`."""

    class SearchTimeoutOnceResolver:
        def __init__(self) -> None:
            self.calls = 0

        async def __call__(
            self, *, question: str, spec_id: str | None = None, wide: bool = False
        ) -> dict[str, Any]:
            self.calls += 1
            if self.calls == 1:
                raise _timeout_error("search", {"question": question})
            return _envelope()

    class GetDataTimeoutExecutor:
        async def __call__(
            self, *, tool: str, args: dict[str, Any], on_progress: ProgressCallback | None = None
        ) -> dict[str, Any]:
            raise _timeout_error("get_data", args)

    app = _app(SearchTimeoutOnceResolver(), GetDataTimeoutExecutor())  # type: ignore[arg-type]
    async with _client(app) as client:
        await _stream(client, "/api/studio/turn", _turn_body())  # search times out first
        second = await _stream(client, "/api/studio/turn", _turn_body("again"))

    second_error = _frame(second, "error")["error"]
    assert "get_data" in second_error["detail"]
    assert second_error["retry"] == "retry_after_delay"
    assert second_error["retry_after_seconds"] == 30


async def test_a_write_component_timeout_is_never_advised_a_retry() -> None:
    """The write was sent, so the platform may have stored the component, and a
    retry can store a second draft."""

    async def compose(**_: Any) -> dict[str, Any]:
        return {"intents": [{"type": "stat_card", "title": "Aid to Kenya", "frame_ref": REF_A}]}

    class WriteTimeoutExecutor:
        async def __call__(
            self, *, tool: str, args: dict[str, Any], on_progress: ProgressCallback | None = None
        ) -> dict[str, Any]:
            if tool == "get_data":
                return {"outcome": "ok", "frame_refs": [REF_A]}
            if tool == "describe_frame":
                return _VIEW
            raise _timeout_error(tool, args)

    app = _app(
        ScriptedResolver([_envelope()]),
        WriteTimeoutExecutor(),  # type: ignore[arg-type]
        generative=GenerativeNodes(compose=compose),
    )
    async with _client(app) as client:
        frames = await _stream(client, "/api/studio/turn", _turn_body())

    error = _frame(frames, "error")["error"]
    assert error["error"] == "transient_unavailable"
    assert error["retry"] == "do_not_retry"
    assert error.get("retry_after_seconds") is None
    assert "may already exist" in error["detail"]


async def test_a_surface_changed_refusal_reaches_the_wire_as_do_not_retry() -> None:
    """`executor.outcomes.RefusalKind.SURFACE_CHANGED` -- a plan step whose
    tool has changed since the plan was compiled -- has no member of its own
    in `harness.wire.ErrorCode`. `_stream_fatal` must not fall back to
    `transient_unavailable`/`retry_after_delay` for it: retrying a stale
    compiled plan can never clear the mismatch, so the wire frame has to
    carry `capability_unavailable`/`do_not_retry` and no `retry_after_seconds`,
    honouring the refusal's own declared `retry_mode` instead of discarding
    it."""

    class SurfaceChangedResolver:
        async def __call__(
            self, *, question: str, spec_id: str | None = None, wide: bool = False
        ) -> dict[str, Any]:
            raise ResolverError(
                "surface changed",
                refusal={
                    "code": "surface_changed",
                    "retry_mode": "do_not_retry",
                    "message": "a plan step's tool has changed since the plan was compiled",
                },
            )

    async with _client(
        _app(SurfaceChangedResolver(), ScriptedExecutor())  # type: ignore[arg-type]
    ) as client:
        frames = await _stream(client, "/api/studio/turn", _turn_body())

    assert _events(frames)[-1] == "error"
    error = _frame(frames, "error")["error"]
    assert error["error"] == "capability_unavailable"
    assert error["retry"] == "do_not_retry"
    # `exclude_none=True` on the wire frame drops an absent field entirely.
    assert "retry_after_seconds" not in error
    assert "done" not in _events(frames)


# --- heartbeats (delta 3) -------------------------------------------------------


async def test_heartbeat_comments_fill_gaps_between_frames() -> None:
    """`_sse_stream` emits `: ka` comment lines while the next frame is slow,
    and the comments carry no seq: they are SSE comments outside the frame
    sequence."""

    async def slow_frames():
        yield MessageDelta(event="message.delta", seq=0, text="one")
        await asyncio.sleep(0.12)
        yield MessageDelta(event="message.delta", seq=1, text="two")

    lines = [line async for line in _sse_stream(slow_frames(), heartbeat_seconds=0.05)]

    data_lines = [line for line in lines if line.startswith("data: ")]
    heartbeats = [line for line in lines if line.startswith(": ka")]
    assert len(data_lines) == 2
    assert len(heartbeats) >= 1
    assert lines[0].startswith("data: ")  # the first frame was not delayed


# --- cancel (delta 14) ----------------------------------------------------------


async def test_cancel_aborts_a_streaming_turn_with_a_cancelled_error_frame() -> None:
    """POST /api/studio/cancel while a turn streams: the stream ends with a
    stream-fatal `cancelled` error frame and no `done` follows."""

    class HangingResolver:
        async def __call__(
            self, *, question: str, spec_id: str | None = None, wide: bool = False
        ) -> dict[str, Any]:
            await asyncio.Event().wait()  # never resolves
            raise AssertionError("unreachable")

    studio = StudioRunner(
        create_workflow=lambda recorder, surface: create_root_workflow(
            resolver=HangingResolver(),  # type: ignore[arg-type]
            executor=recorder,
            tool_surface=surface,
            fetch_series_stats=scripted_fetch_series_stats,
        ),
        executor=ScriptedExecutor(),
        session_service=_new_session_service(),
    )
    app = FastAPI()
    app.include_router(create_studio_router(studio, settings=Settings()))

    async def _armed() -> None:
        while not any(
            entry.state is not None and entry.state.cancel_event is not None
            for entry in studio.store._runtime.values()
        ):
            await asyncio.sleep(0.01)

    async with _client(app) as client:
        stream_task = asyncio.ensure_future(_stream(client, "/api/studio/turn", _turn_body()))
        await asyncio.wait_for(_armed(), timeout=5)
        cancel = await client.post(
            "/api/studio/cancel",
            json={"kind": "turn.cancel", "schema_version": 1, "conversation_id": CONVERSATION},
            headers=AUTH,
        )
        frames = await asyncio.wait_for(stream_task, timeout=5)

    assert cancel.status_code == 204
    assert _events(frames)[-1] == "error"
    assert _frame(frames, "error")["error"]["error"] == "cancelled"
    assert "done" not in _events(frames)


async def test_cancelling_an_idle_conversation_is_a_no_op_204() -> None:
    """The conversation id here was never created by a `/turn`, so this also
    pins the unknown-conversation case: `/cancel`'s contract is an idempotent
    no-op, unlike `/interrupt` and `/composition`'s 404."""
    async with _client(_app(ScriptedResolver([_envelope()]), ScriptedExecutor())) as client:
        response = await client.post(
            "/api/studio/cancel",
            json={"kind": "turn.cancel", "schema_version": 1, "conversation_id": CONVERSATION},
            headers=AUTH,
        )
    assert response.status_code == 204


async def test_closing_the_sse_stream_unwinds_the_segment_generator() -> None:
    """Unit level: closing the transport generator unwinds the segment
    generator and disarms the cancel event. (Transport-level disconnect is
    pinned separately below.)"""
    closed = asyncio.Event()

    async def segment():
        try:
            yield MessageDelta(event="message.delta", seq=0, text="one")
            await asyncio.Event().wait()  # hang like a slow tool call
        finally:
            closed.set()

    conversation = ConversationState(conversation_id=CONVERSATION)
    stream = _sse_stream(segment(), conversation=conversation)
    first = await anext(stream)
    assert first.startswith("data: ")
    await stream.aclose()

    await asyncio.wait_for(closed.wait(), timeout=2)
    assert conversation.cancel_event is None  # disarmed on unwind


async def test_a_cancelled_response_task_records_the_turn_and_unwinds_the_segment() -> None:
    """What a client disconnect becomes inside the process: uvicorn cancels
    the response task. The stream must close the segment generator, record
    the turn as cancelled, and disarm the cancel event on its way out."""
    closed = asyncio.Event()

    async def segment():
        try:
            yield MessageDelta(event="message.delta", seq=0, text="one")
            await asyncio.Event().wait()  # hang like a slow tool call
        finally:
            closed.set()

    conversation = ConversationState(conversation_id=CONVERSATION)
    conversation.begin_turn()
    conversation.turn_user_message = "kenya oda 2023"
    conversation.turn_final_text = "one"
    received: list[str] = []

    async def consume() -> None:
        async for chunk in _sse_stream(
            segment(), conversation=conversation, heartbeat_seconds=0.02
        ):
            received.append(chunk)  # noqa: PERF401 -- observed one chunk at a time

    task = asyncio.ensure_future(consume())
    while not received:  # the first frame is out; the stream is parked at a heartbeat wait
        await asyncio.sleep(0.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    await asyncio.wait_for(closed.wait(), timeout=2)
    assert conversation.turns == [
        {"user": "kenya oda 2023", "assistant": "one", "error": "cancelled"}
    ]
    assert conversation.cancel_event is None


async def test_unwinding_under_a_cancelled_response_cancels_the_segment_once() -> None:
    """A response task whose anyio scope is cancelled (Starlette's task group
    once the response is cancelled) is re-cancelled on every loop iteration.
    `_unwind` cancels the segment task once and keeps those re-cancellations
    away from it, so the segment's own cleanup runs to the end."""
    cleanup_cancels = 0
    cleaned_up = False

    async def segment() -> None:
        nonlocal cleanup_cancels, cleaned_up
        try:
            await asyncio.Event().wait()  # a slow tool call
        finally:
            try:
                await asyncio.sleep(0.05)  # the ledger release and connection close
            except asyncio.CancelledError:
                cleanup_cancels += 1
                raise
            cleaned_up = True

    pending = asyncio.ensure_future(segment())
    await asyncio.sleep(0)  # the segment is parked in its slow call
    with anyio.CancelScope() as scope:
        scope.cancel()
        await _unwind(pending)

    assert cleanup_cancels == 0
    assert cleaned_up


async def _lose_the_client_while_parked(
    conversation: ConversationState, *, how: str, paused: bool
) -> None:
    """Park `_sse_stream` after one frame with the segment idle, then take the
    client away: `how="cancel"` cancels the consumer while the stream awaits its
    next frame; `how="close"` closes the stream at a heartbeat `yield`. When
    `paused`, the conversation holds a pending plan interrupt, as it does once
    `plan.proposed` has been emitted."""
    if paused:
        conversation.pending["int_01"] = _PendingInterrupt(
            adk_id="adk_1", kind="plan", plan_version=1, plan_revision="rev_1"
        )

    async def segment():
        yield MessageDelta(event="message.delta", seq=0, text="one")
        await asyncio.Event().wait()  # the run has not drained yet

    stream = _sse_stream(segment(), conversation=conversation, heartbeat_seconds=0.02)
    assert (await anext(stream)).startswith("data: ")
    if how == "close":
        assert (await anext(stream)).startswith(": ka")
        await stream.aclose()
        return

    async def consume() -> None:
        async for _ in stream:
            pass

    task = asyncio.ensure_future(consume())
    await asyncio.sleep(0.01)  # parked at the heartbeat wait
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.parametrize("how", ["cancel", "close"])
async def test_a_client_that_leaves_a_paused_turn_keeps_it_open_and_resumable(how: str) -> None:
    """The turn is waiting on `/interrupt`: a disconnect records nothing and
    leaves the pending interrupt and the telemetry span in place; the next
    `begin_turn` abandons it exactly as it does for a client that read the
    stream to the end."""
    conversation = ConversationState(conversation_id=CONVERSATION)
    conversation.begin_turn()
    conversation.turn_user_message = "kenya oda 2023"
    span = conversation._telemetry_span

    await _lose_the_client_while_parked(conversation, how=how, paused=True)

    assert conversation.awaiting_answer
    assert not conversation.turn_recorded
    assert conversation.turns == []
    assert conversation._telemetry_span is span
    assert conversation.cancel_event is None

    conversation.begin_turn()
    assert conversation.turns == [{"user": "kenya oda 2023", "assistant": "", "error": "abandoned"}]
    assert not conversation.awaiting_answer


@pytest.mark.parametrize("how", ["cancel", "close"])
async def test_a_client_that_leaves_a_running_turn_records_it_as_cancelled(how: str) -> None:
    conversation = ConversationState(conversation_id=CONVERSATION)
    conversation.begin_turn()
    conversation.turn_user_message = "kenya oda 2023"

    await _lose_the_client_while_parked(conversation, how=how, paused=False)

    assert not conversation.awaiting_answer
    assert conversation.turn_recorded
    assert conversation.turns == [{"user": "kenya oda 2023", "assistant": "", "error": "cancelled"}]
    assert conversation._telemetry_span is None
    assert conversation.cancel_event is None


# NOTE: the transport half of a disconnect cannot be driven in-process --
# httpx2's ASGITransport runs the app to completion and never delivers
# http.disconnect, so a route-level test would hang (verified). The process
# half -- uvicorn cancelling the response task -- is the test above. The
# RELIABLE cancellation path either way is POST /api/studio/cancel (delta 14).


# --- the Edit Card route (component_ref intent=edit) ----------------------------


STORED_INTENT = {
    "type": "stat_card",
    "title": "Aid to Kenya",
    "select": {"stat_var": "dcid:Aid"},
    "frame_ref": REF_A,
}


def _edit_body(message: str) -> dict[str, Any]:
    return {
        "kind": "turn.request",
        "schema_version": 1,
        "conversation_id": CONVERSATION,
        "message": message,
        "component_ref": {
            "component_id": "cmp_01test0001",
            "revision": 1,
            "intent": "edit",
        },
    }


def _described(revision: int = 1, *, stored_intent: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "outcome": "ok",
        "frame_refs": [],
        "component_body": {
            "component_id": "cmp_01test0001",
            "revision": revision,
            "stored_intent": stored_intent or STORED_INTENT,
            "fact_summary": COMPONENT["fact_summary"],
        },
    }


_VIEW = {"outcome": "ok", "frame_refs": [], "view": {"dimensions": {"stat_var": ["dcid:Aid"]}}}


class ScriptedEditModel:
    def __init__(self, result: dict[str, Any]) -> None:
        self.result = result
        self.calls: list[dict[str, Any]] = []

    async def __call__(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        return self.result


async def test_component_ref_edit_patch_streams_component_revised() -> None:
    revised_component = {
        **COMPONENT,
        "revision": 2,
        "fact_summary": {**COMPONENT["fact_summary"], "revision": 2},
    }
    executor = ScriptedExecutor(
        [
            {"outcome": "ok", "frame_refs": [REF_A]},
            {
                "outcome": "ok",
                "frame_refs": [REF_A],
                "component": COMPONENT,
                "write_token": WRITE_TOKEN,
            },
            _described(),
            _VIEW,
            {"outcome": "ok", "frame_refs": [], "component": revised_component},
        ],
        strict=True,
    )
    edit_model = ScriptedEditModel(
        {
            "text": "Renamed the card.",
            "patch": {"title": "Aid to Kenya, 2018-2024"},
            "recreate": None,
        }
    )
    app = _app(
        ScriptedResolver([_envelope(choice_points=2)]),
        executor,
        edit_model=edit_model,
        generative=_plan_generative(),
        tool_surface=ScriptedSurface(),
    )
    async with _client(app) as client:
        await _conversation_with_component(client)
        frames = await _stream(client, "/api/studio/turn", _edit_body("rename it"))

    assert _events(frames)[-1] == "done"
    revised = _frame(frames, "component.revised")
    assert revised["component_id"] == "cmp_01test0001"
    assert revised["revision"] == 2
    assert revised["changed"] == ["title"]
    manifest = _frame(frames, "done")["composition"]
    assert manifest["blocks"][0]["revision"] == 2
    assert manifest["blocks"][0]["title"] == "Aid to Kenya, 2018-2024"
    assert edit_model.calls[0]["stored_intent"] == STORED_INTENT
    # the edit turn's exact call sequence and the write's exact arguments
    assert [c["tool"] for c in executor.calls[2:]] == [
        "describe_component",
        "describe_frame",
        "write_component",
    ]
    assert executor.calls[-1]["args"] == {
        "intent": {"title": "Aid to Kenya, 2018-2024"},
        "component_id": "cmp_01test0001",
        "write_token": WRITE_TOKEN,
    }


async def _edit_without_a_write_token(
    patch: dict[str, Any], *, stored_intent: dict[str, Any] = STORED_INTENT
) -> tuple[list[dict[str, Any]], ScriptedExecutor]:
    """An Edit Card patch on a card this conversation holds no write
    token for: its create returned none, as a fork, a gallery card or a
    card made before tokens would."""
    recreated = {
        **COMPONENT,
        "component_id": "cmp_01test0002",
        "payload": resolved_revision(
            "stat_card",
            component_id="cmp_01test0002",
            frame_ref=REF_A,
            title="Aid to Kenya, 2018-2024",
        ),
    }
    executor = ScriptedExecutor(
        [
            {"outcome": "ok", "frame_refs": [REF_A]},
            {"outcome": "ok", "frame_refs": [REF_A], "component": COMPONENT},
            _described(stored_intent=stored_intent),
            _VIEW,
            {"outcome": "ok", "frame_refs": [], "component": recreated},
        ],
        strict=True,
    )
    edit_model = ScriptedEditModel({"text": "Renamed the card.", "patch": patch, "recreate": None})
    app = _app(
        ScriptedResolver([_envelope(choice_points=2)]),
        executor,
        edit_model=edit_model,
        generative=_plan_generative(),
        tool_surface=ScriptedSurface(),
    )
    async with _client(app) as client:
        await _conversation_with_component(client)
        frames = await _stream(client, "/api/studio/turn", _edit_body("rename it"))
    return frames, executor


async def test_an_edit_patch_on_a_card_holding_no_write_token_recreates_it_in_its_slot() -> None:
    """A card this conversation holds no write token for cannot be
    revised, so the patch is merged into the stored intent and written as
    a create, and the new card takes the old block's slot."""
    frames, executor = await _edit_without_a_write_token(
        {"title": "Aid to Kenya, 2018-2024", "select": {"unit": "USD"}}
    )

    assert "component.revised" not in _events(frames)
    assert _frame(frames, "component.created")["component_id"] == "cmp_01test0002"
    manifest = _frame(frames, "done")["composition"]
    assert [b["component_id"] for b in manifest["blocks"]] == ["cmp_01test0002"]
    # A create: the stored intent with the patch merged in at the top
    # level, as dfo-apis `merge_intent` merges a revise, so `select` is
    # replaced whole, and no component_id or token.
    assert executor.calls[-1] == {
        "tool": "write_component",
        "args": {
            "intent": {
                **STORED_INTENT,
                "title": "Aid to Kenya, 2018-2024",
                "select": {"unit": "USD"},
            }
        },
    }


async def test_a_rebuilt_card_is_reported_as_a_new_card() -> None:
    """The model drafted its text for an in-place revise; the user reads
    that the card became a new one with a fresh history."""
    frames, _ = await _edit_without_a_write_token({"title": "Aid to Kenya, 2018-2024"})

    text = _frame(frames, "message.complete")["text"]
    assert text != "Renamed the card."
    assert "new card" in text
    assert "new id" in text
    assert "fresh revision history" in text
    step = next(f for f in frames if f["event"] == "activity.step" and f["step_id"] == "e2")
    assert step["label"] == "Rebuild as a new card"


async def test_a_patch_dropping_a_select_key_drops_it_from_the_rebuilt_card() -> None:
    """The widened select replaces the stored one whole, so the dropped
    series filter does not survive into the new card."""
    stored = {**STORED_INTENT, "select": {"stat_var": "dcid:Aid", "series": ["dac"]}}

    _, executor = await _edit_without_a_write_token(
        {"select": {"stat_var": "dcid:Aid"}}, stored_intent=stored
    )

    assert executor.calls[-1]["args"]["intent"] == {**stored, "select": {"stat_var": "dcid:Aid"}}


async def test_a_patch_naming_a_create_only_field_is_refused_before_any_write() -> None:
    edit_model = ScriptedEditModel(
        {"text": "Changed it.", "patch": {"type": "line_chart"}, "recreate": None}
    )
    executor = ScriptedExecutor(
        [
            {"outcome": "ok", "frame_refs": [REF_A]},
            {"outcome": "ok", "frame_refs": [REF_A], "component": COMPONENT},
            _described(),
            _VIEW,
        ],
        strict=True,
    )
    app = _app(
        ScriptedResolver([_envelope(choice_points=2)]),
        executor,
        edit_model=edit_model,
        generative=_plan_generative(),
        tool_surface=ScriptedSurface(),
    )
    async with _client(app) as client:
        await _conversation_with_component(client)
        frames = await _stream(client, "/api/studio/turn", _edit_body("make it a line"))

    assert _frame(frames, "message.complete")["text"] == (
        "The edit was refused: 'type' is create-only and cannot be revised."
    )
    assert "component.created" not in _events(frames)
    assert [c["tool"] for c in executor.calls].count("write_component") == 1


class _TokenIssuingPlatform:
    """A presentation-mcp client stand-in that behaves like the platform on
    write tokens: a create returns a fresh `cmw_` token top-level in its
    structured content (and on the last line of its text, as the real tool
    does), and a revise is refused unless it presents that component's
    token."""

    def __init__(self) -> None:
        self.tokens: dict[str, str] = {}
        self.revisions: dict[str, int] = {}
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.results: list[SimpleNamespace] = []

    async def __aenter__(self) -> _TokenIssuingPlatform:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        return None

    def _result(
        self,
        body: dict[str, Any],
        *,
        text: list[str] | None = None,
        meta: dict[str, Any] | None = None,
    ) -> SimpleNamespace:
        content = [SimpleNamespace(type="text", text=block) for block in text or []]
        result = SimpleNamespace(
            is_error=False, structured_content=body, content=content, meta=meta
        )
        self.results.append(result)
        return result

    def _written(self, component_id: str, *, token: str | None) -> SimpleNamespace:
        revision = self.revisions[component_id]
        body: dict[str, Any] = {
            "component_id": component_id,
            "revision": revision,
            "fact_summary": {},
        }
        text = [json.dumps(body)]
        if token is not None:
            body["write_token"] = token
            text.append(f"Keep this write_token with the component_id.\nwrite_token: {token}")
        envelope = resolved_revision(
            "stat_card",
            component_id=component_id,
            revision=revision,
            frame_ref=REF_A,
            title="Aid to Kenya",
        )
        return self._result(body, text=text, meta={RESOLVED_PAYLOAD_META_KEY: envelope})

    async def call_tool(
        self,
        tool: str,
        arguments: dict[str, Any],
        *,
        meta: dict[str, str],
        progress_handler: Any,
        raise_on_error: bool,
    ) -> SimpleNamespace:
        self.calls.append((tool, dict(arguments)))
        if tool == "describe_frame":
            return self._result({})
        if tool == "describe_component":
            component_id = arguments["component_id"]
            return self._result(
                {
                    "component_id": component_id,
                    "revision": self.revisions[component_id],
                    "stored_intent": STORED_INTENT,
                    "fact_summary": {},
                }
            )
        assert tool == "write_component", tool
        component_id = arguments.get("component_id")
        if component_id is None:
            component_id = f"cmp_01test{len(self.tokens) + 1:04d}"
            self.tokens[component_id] = f"cmw_{component_id}_secret"
            self.revisions[component_id] = 1
            return self._written(component_id, token=self.tokens[component_id])
        if arguments.get("write_token") != self.tokens.get(component_id):
            return self._result(
                {"error": "write_refused", "retry": "do_not_retry", "detail": "wrong token"}
            )
        self.revisions[component_id] += 1
        return self._written(component_id, token=None)


async def test_a_write_token_reaches_the_revise_and_nothing_the_client_sees(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A studio create then an Edit Card revise, through the real
    `McpExecutor` over a platform that issues `cmw_` tokens: the revise
    presents the create's token, and no SSE frame, digest, recall answer,
    turn record, executor record or state repr carries it. The ledger
    holds it apart from the component digest entries."""
    platform = _TokenIssuingPlatform()
    monkeypatch.setattr(
        McpTimeoutPolicy, "client", lambda self, url, *, auth, caller, tool=None: platform
    )
    records: list[Any] = []
    drain = _RecordingExecutor.drain

    def _drain_and_keep(self: _RecordingExecutor, conversation: str) -> list[Any]:
        drained = drain(self, conversation)
        records.extend(drained)
        return drained

    monkeypatch.setattr(_RecordingExecutor, "drain", _drain_and_keep)
    principal = "u_" + hashlib.sha256(b"test-token").hexdigest()[:16]
    provenance = ProvenanceStore()
    provenance.put(conversation=f"{principal}/{CONVERSATION}", manifest={}, **REF_A)
    session_service = _new_session_service()
    edit_model = ScriptedEditModel(
        {
            "text": "Renamed the card.",
            "patch": {"title": "Aid to Kenya, 2018-2024"},
            "recreate": None,
        }
    )
    app = _app(
        ScriptedResolver([_envelope(choice_points=2)]),
        McpExecutor(settings=Settings(), provenance=provenance),
        edit_model=edit_model,
        generative=_plan_generative(),
        tool_surface=ScriptedSurface(),
        session_service=session_service,
    )
    async with _client(app) as client:
        created = await _stream(client, "/api/studio/turn", _turn_body("q with forks"))
        plan = _frame(created, "plan.proposed")
        resumed = await _stream(
            client, "/api/studio/interrupt", _interrupt_body(plan["interrupt_id"], "approve")
        )
        edited = await _stream(client, "/api/studio/turn", _edit_body("rename it"))

    component_id = _frame(resumed, "component.created")["component_id"]
    token = platform.tokens[component_id]
    assert _frame(edited, "component.revised")["revision"] == 2
    revise = platform.calls[-1]
    assert revise[0] == "write_component"
    assert revise[1]["component_id"] == component_id
    assert revise[1]["write_token"] == token

    conversation = app.state.studio.store.current(principal, CONVERSATION)
    assert conversation.write_tokens == {component_id: token}
    client_visible = [
        *created,
        *resumed,
        *edited,
        conversation.canvas_digest_state(focus=component_id),
        conversation.conversation_digest_state(),
        conversation.recall_state(scope="canvas", query="aid kenya"),
        conversation.recall_state(scope="conversation", query="rename"),
        conversation.components,
        conversation.turns,
    ]
    assert "cmw_" not in json.dumps(client_visible, default=str)
    assert "cmw_" not in repr(conversation)
    assert any(
        record.tool == "write_component" and "component_id" in record.args for record in records
    )
    assert "cmw_" not in json.dumps(
        [(record.args, record.result) for record in records], default=str
    )
    # The seam took the token off every result, content text included,
    # before any reader downstream of `McpExecutor` saw it.
    assert "cmw_" not in json.dumps(
        [(r.structured_content, [block.text for block in r.content]) for r in platform.results]
    )

    reloaded = await StudioStore(session_service, app_name=AGENT_NAME).load(principal, CONVERSATION)
    assert reloaded.write_tokens == {component_id: token}
    assert "cmw_" not in json.dumps(reloaded.components)


async def test_component_ref_edit_recreate_replaces_the_block_in_place() -> None:
    recreated = {
        **COMPONENT,
        "component_id": "cmp_01test0002",
        "type": "line_chart",
        "payload": resolved_revision(
            "line_chart",
            component_id="cmp_01test0002",
            frame_ref=REF_A,
            title="Aid to Kenya over time",
        ),
    }
    third = {**COMPONENT, "component_id": "cmp_01test0003"}
    executor = ScriptedExecutor(
        [
            {"outcome": "ok", "frame_refs": [REF_A]},
            {"outcome": "ok", "frame_refs": [REF_A], "component": COMPONENT},
            {"outcome": "ok", "frame_refs": [REF_A]},
            {"outcome": "ok", "frame_refs": [REF_A], "component": third},
            _described(),
            _VIEW,
            {"outcome": "ok", "frame_refs": [], "component": recreated},
        ],
        strict=True,
    )
    edit_model = ScriptedEditModel(
        {
            "text": "Rebuilt as a line chart.",
            "patch": None,
            "recreate": {
                "type": "line_chart",
                "title": "Aid to Kenya over time",
                "select": {"stat_var": "dcid:Aid"},
            },
        }
    )
    app = _app(
        ScriptedResolver([_envelope(choice_points=2)]),
        executor,
        edit_model=edit_model,
        generative=_plan_generative(),
        tool_surface=ScriptedSurface(),
    )
    async with _client(app) as client:
        await _conversation_with_component(client)
        # A second card, pinned by the user, so the replacement has a
        # neighbour whose slot and pin it must leave alone.
        second = await _stream(client, "/api/studio/turn", _turn_body("another with forks"))
        plan = _frame(second, "plan.proposed")
        resumed = await _stream(
            client, "/api/studio/interrupt", _interrupt_body(plan["interrupt_id"], "approve")
        )
        before = _frame(resumed, "done")["composition"]
        pinned = await client.post(
            "/api/studio/composition",
            json=_mutation_body(before, [{"op": "pin", "component_id": "cmp_01test0003"}]),
            headers=AUTH,
        )
        assert pinned.status_code == 200
        frames = await _stream(client, "/api/studio/turn", _edit_body("make it a line chart"))

    created = _frame(frames, "component.created")
    assert created["component_id"] == "cmp_01test0002"
    manifest = _frame(frames, "done")["composition"]
    assert [b["component_id"] for b in before["blocks"]] == ["cmp_01test0001", "cmp_01test0003"]
    # Replaced in its own slot; the pinned neighbour is untouched.
    assert [b["component_id"] for b in manifest["blocks"]] == ["cmp_01test0002", "cmp_01test0003"]
    assert manifest["blocks"][0]["type"] == "line_chart"
    assert manifest["blocks"][1]["pinned"] is True
    # the recreate intent was pinned to the stored frame_ref by the route,
    # carrying the model's full intent and no component_id, which makes it a create
    write_call = executor.calls[-1]
    assert write_call["tool"] == "write_component"
    assert write_call["args"] == {
        "intent": {
            "type": "line_chart",
            "title": "Aid to Kenya over time",
            "select": {"stat_var": "dcid:Aid"},
            "frame_ref": REF_A,
        }
    }


async def test_component_ref_edit_refusal_is_an_honest_message() -> None:
    executor = ScriptedExecutor(
        [
            {"outcome": "ok", "frame_refs": [REF_A]},
            {"outcome": "ok", "frame_refs": [REF_A], "component": COMPONENT},
            _described(),
            _VIEW,
            {
                "outcome": "refusal",
                "refusal": {
                    "code": "schema_invalid",
                    "retry_mode": "do_not_retry",
                    "fix": None,
                    "message": "'observationAbout' does not hold value(s) ['country/KEN']",
                },
            },
        ],
        strict=True,
    )
    edit_model = ScriptedEditModel(
        {
            "text": "Narrowing to Kenya.",
            "patch": {"select": {"observationAbout": "country/KEN"}},
            "recreate": None,
        }
    )
    app = _app(
        ScriptedResolver([_envelope(choice_points=2)]),
        executor,
        edit_model=edit_model,
        generative=_plan_generative(),
        tool_surface=ScriptedSurface(),
    )
    async with _client(app) as client:
        await _conversation_with_component(client)
        frames = await _stream(client, "/api/studio/turn", _edit_body("only Kenya"))

    assert "component.revised" not in _events(frames)
    assert _events(frames)[-1] == "done"
    message = _frame(frames, "message.complete")["text"]
    assert "does not hold value" in message
    failed = [f for f in frames if f["event"] == "activity.step" and f["status"] == "failed"]
    assert failed, "the refused write should surface as a failed step"
    assert executor.calls[-1]["tool"] == "write_component"  # the write was what refused


# --- canvas digest and canvas questions (WS-C) -----------------------------------


class RecordingConverse:
    """A converse seam that records the digest it was given and classifies
    canvas questions the way the real model does -- by the question text."""

    def __init__(self) -> None:
        self.digests: list[dict[str, Any] | None] = []
        self.histories: list[dict[str, Any] | None] = []
        self.recall_results: list[dict[str, Any] | None] = []

    async def __call__(
        self,
        *,
        question: str,
        canvas_digest: dict[str, Any] | None = None,
        conversation_digest: dict[str, Any] | None = None,
        recall_result: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self.digests.append(canvas_digest)
        self.histories.append(conversation_digest)
        self.recall_results.append(recall_result)
        about = "canvas" if "canvas" in question.lower() else "data"
        # A trivial but real decomposition -- these tests exercise digest
        # passing and recall, not converse's own decomposition, so a bare
        # single part is enough to keep the "data" route reaching probe.
        search_parts = None if about == "canvas" else [{"variable": question}]
        return {"text": f"Looking into {question}.", "search_parts": search_parts, "about": about}


class DigestNarrate:
    async def __call__(self, *, question: str, facts: dict[str, Any]) -> dict[str, Any]:
        titles = [b.get("title") for b in (facts.get("canvas") or {}).get("blocks", [])]
        return {"text": f"Canvas holds: {', '.join(filter(None, titles))}."}


async def test_turn_two_receives_the_canvas_digest() -> None:
    converse = RecordingConverse()
    executor = ScriptedExecutor(
        [
            {"outcome": "ok", "frame_refs": [REF_A]},
            {"outcome": "ok", "frame_refs": [REF_A], "component": COMPONENT},
        ]
    )
    app = _app(
        ScriptedResolver([_envelope(choice_points=2), _envelope()]),
        executor,
        generative=GenerativeNodes(
            converse=converse, planner=ScriptedPlannerModel(), compose=ScriptedCompose()
        ),
        tool_surface=ScriptedSurface(),
    )
    async with _client(app) as client:
        await _conversation_with_component(client)
        await _stream(client, "/api/studio/turn", _turn_body("now the same for Uganda"))

    assert converse.digests[0] is None  # turn 1: empty canvas, no digest
    turn2 = converse.digests[-1]
    assert turn2 is not None
    assert [b["component_id"] for b in turn2["blocks"]] == ["cmp_01test0001"]
    assert turn2["blocks"][0]["title"] == "Aid to Kenya"
    assert turn2["blocks"][0]["fact_summary"]["series"][0]["label"] == "Kenya"


async def test_turn_two_receives_the_anaphor_window() -> None:
    """`studio.py` seeds `anaphor_window` from `ConversationState.
    anaphor_window` the same way it seeds `canvas_digest`: turn one lands a
    card with a resolved window, and turn two's "the same years" reaches
    the resolver with that window already substituted."""
    windowed_component = {
        **COMPONENT,
        "fact_summary": {**COMPONENT["fact_summary"], "window": {"start": "2015", "end": "2020"}},
    }
    converse = RecordingConverse()
    resolver = ScriptedResolver([_envelope(choice_points=2), _envelope()])
    executor = ScriptedExecutor(
        [
            {"outcome": "ok", "frame_refs": [REF_A]},
            {"outcome": "ok", "frame_refs": [REF_A], "component": windowed_component},
        ]
    )
    app = _app(
        resolver,
        executor,
        generative=GenerativeNodes(
            converse=converse, planner=ScriptedPlannerModel(), compose=ScriptedCompose()
        ),
        tool_surface=ScriptedSurface(),
    )
    async with _client(app) as client:
        await _conversation_with_component(client)
        await _stream(client, "/api/studio/turn", _turn_body("compare it for the same years"))

    assert "the same years" not in resolver.calls[-1]["question"]
    assert "2015 to 2020" in resolver.calls[-1]["question"]


async def test_a_canvas_question_answers_from_the_digest_without_search() -> None:
    converse = RecordingConverse()
    resolver = ScriptedResolver([_envelope(choice_points=2), _envelope()])
    executor = ScriptedExecutor(
        [
            {"outcome": "ok", "frame_refs": [REF_A]},
            {"outcome": "ok", "frame_refs": [REF_A], "component": COMPONENT},
        ]
    )
    app = _app(
        resolver,
        executor,
        generative=GenerativeNodes(
            converse=converse,
            narrate=DigestNarrate(),
            planner=ScriptedPlannerModel(),
            compose=ScriptedCompose(),
        ),
        tool_surface=ScriptedSurface(),
    )
    async with _client(app) as client:
        await _conversation_with_component(client)
        resolves_before = len(resolver.calls)
        frames = await _stream(client, "/api/studio/turn", _turn_body("what is on my canvas?"))

    assert len(resolver.calls) == resolves_before  # no search on the canvas turn
    assert _events(frames)[-1] == "done"
    assert "Aid to Kenya" in _frame(frames, "message.complete")["text"]


# --- elicitation skip and free text (deltas 12) ----------------------------------


def _candidates_envelope() -> dict[str, Any]:
    return _envelope(
        status="candidates",
        spec_id=None,
        candidates=[
            {"spec_id": SPEC_A, "label": "reading A"},
            {"spec_id": SPEC_B, "label": "reading B"},
        ],
    )


async def test_skip_proceeds_on_the_top_ranked_candidate() -> None:
    resolver = ScriptedResolver([_candidates_envelope(), _envelope(spec_id=SPEC_A)])
    async with _client(_app(resolver, ScriptedExecutor())) as client:
        first = await _stream(client, "/api/studio/turn", _turn_body("ambiguous q"))
        elicitation = _frame(first, "elicitation.request")
        assert elicitation["skippable"] is True
        assert elicitation["allow_free_text"] is True
        assert elicitation["progress"] == {"completed": 1, "total": 2}

        second = await _stream(
            client,
            "/api/studio/interrupt",
            _interrupt_body(elicitation["interrupt_id"], "skip"),
        )

    assert _events(second)[-1] == "done"
    assert resolver.calls[-1]["spec_id"] == SPEC_A  # the top-ranked candidate


async def test_a_free_text_answer_re_resolves_with_the_refined_question() -> None:
    resolver = ScriptedResolver([_candidates_envelope(), _envelope(spec_id=SPEC_B)])
    async with _client(_app(resolver, ScriptedExecutor())) as client:
        first = await _stream(client, "/api/studio/turn", _turn_body("ambiguous q"))
        elicitation = _frame(first, "elicitation.request")

        second = await _stream(
            client,
            "/api/studio/interrupt",
            _interrupt_body(elicitation["interrupt_id"], "answer", "the OECD reading please"),
        )

    assert _events(second)[-1] == "done"
    refined = resolver.calls[-1]
    assert refined["spec_id"] is None  # not mistaken for an option id
    assert "the OECD reading please" in refined["question"]


# --- auto_add and the add / set_card_meta ops (deltas 11, 13) --------------------


async def test_auto_add_false_streams_the_component_without_composing_it() -> None:
    executor = ScriptedExecutor(
        [
            {"outcome": "ok", "frame_refs": [REF_A]},
            {"outcome": "ok", "frame_refs": [REF_A], "component": COMPONENT},
        ]
    )
    app = _app(
        ScriptedResolver([_envelope(choice_points=2)]),
        executor,
        generative=_plan_generative(),
        tool_surface=ScriptedSurface(),
    )
    async with _client(app) as client:
        frames = await _stream(
            client, "/api/studio/turn", {**_turn_body("q with forks"), "auto_add": False}
        )
        plan = _frame(frames, "plan.proposed")
        resumed = await _stream(
            client, "/api/studio/interrupt", _interrupt_body(plan["interrupt_id"], "approve")
        )

        assert "component.created" in _events(resumed)
        assert "composition.updated" not in _events(resumed)
        assert _frame(resumed, "done").get("composition") is None

        # the client adds it later (the "Pin to Canvas" menu / Added chip);
        # with no composition yet and no composition_id named, the add
        # bootstraps an empty canvas (delta 11).
        response = await client.post(
            "/api/studio/composition",
            json={
                "kind": "composition.mutate",
                "schema_version": 1,
                "conversation_id": CONVERSATION,
                "operations": [{"op": "add", "component_id": "cmp_01test0001"}],
            },
            headers=AUTH,
        )

    assert response.status_code == 200
    bootstrapped = response.json()
    assert [b["component_id"] for b in bootstrapped["blocks"]] == ["cmp_01test0001"]
    assert bootstrapped["blocks"][0]["title"] == "Aid to Kenya"


async def test_add_appends_a_block_rebuilt_from_the_streamed_component() -> None:
    executor = ScriptedExecutor(
        [
            {"outcome": "ok", "frame_refs": [REF_A]},
            {"outcome": "ok", "frame_refs": [REF_A], "component": COMPONENT},
            {"outcome": "ok", "frame_refs": [REF_A]},
            {
                "outcome": "ok",
                "frame_refs": [REF_A],
                "component": {**COMPONENT, "component_id": "cmp_01test0002"},
            },
        ]
    )
    resolver = ScriptedResolver([_envelope(choice_points=2), _envelope(choice_points=2)])
    app = _app(resolver, executor, generative=_plan_generative(), tool_surface=ScriptedSurface())
    async with _client(app) as client:
        manifest = await _conversation_with_component(client)
        # a second component that stays off the canvas (auto_add=false)
        frames = await _stream(
            client, "/api/studio/turn", {**_turn_body("q with forks"), "auto_add": False}
        )
        plan = _frame(frames, "plan.proposed")
        resumed = await _stream(
            client, "/api/studio/interrupt", _interrupt_body(plan["interrupt_id"], "approve")
        )
        assert _frame(resumed, "component.created")["component_id"] == "cmp_01test0002"

        response = await client.post(
            "/api/studio/composition",
            json=_mutation_body(manifest, [{"op": "add", "component_id": "cmp_01test0002"}]),
            headers=AUTH,
        )

    assert response.status_code == 200
    updated = response.json()
    assert [b["component_id"] for b in updated["blocks"]] == ["cmp_01test0001", "cmp_01test0002"]
    assert updated["blocks"][1]["title"] == "Aid to Kenya"
    assert updated["actor"] == "user"


async def test_set_card_meta_writes_harness_owned_card_fields() -> None:
    executor = ScriptedExecutor(
        [
            {"outcome": "ok", "frame_refs": [REF_A]},
            {"outcome": "ok", "frame_refs": [REF_A], "component": COMPONENT},
        ]
    )
    app = _app(
        ScriptedResolver([_envelope(choice_points=2)]),
        executor,
        generative=_plan_generative(),
        tool_surface=ScriptedSurface(),
    )
    async with _client(app) as client:
        manifest = await _conversation_with_component(client)
        response = await client.post(
            "/api/studio/composition",
            json=_mutation_body(
                manifest,
                [
                    {
                        "op": "set_card_meta",
                        "component_id": "cmp_01test0001",
                        "meta": {"subtitle": "2018-2024", "y_label": "USD ($)"},
                    }
                ],
            ),
            headers=AUTH,
        )

    assert response.status_code == 200
    block = response.json()["blocks"][0]
    assert block["meta"]["subtitle"] == "2018-2024"
    assert block["meta"]["y_label"] == "USD ($)"


async def test_a_canvas_question_on_an_empty_canvas_says_so() -> None:
    """The empty-canvas edge: no digest exists, but the question is still a
    canvas question -- it must not fall through to a data search."""
    converse = RecordingConverse()
    resolver = ScriptedResolver([_envelope()])
    app = _app(resolver, ScriptedExecutor(), generative=GenerativeNodes(converse=converse))
    async with _client(app) as client:
        frames = await _stream(client, "/api/studio/turn", _turn_body("what's on my canvas?"))

    assert resolver.calls == []  # never searched
    assert _events(frames)[-1] == "done"
    assert "nothing on your canvas" in _frame(frames, "message.complete")["text"]


# --- regression pins --------------------------------------------------------


async def test_edit_frames_keep_seq_monotonic_across_the_recreate() -> None:
    """Regression guard: the recreated component's frame must not be
    constructed before the step that precedes it, which would give it a
    LOWER seq and break delta 2's monotonic guarantee."""
    recreated = {
        **COMPONENT,
        "component_id": "cmp_01test0002",
        "type": "line_chart",
        "payload": resolved_revision("line_chart", component_id="cmp_01test0002", frame_ref=REF_A),
    }
    executor = ScriptedExecutor(
        [
            {"outcome": "ok", "frame_refs": [REF_A]},
            {"outcome": "ok", "frame_refs": [REF_A], "component": COMPONENT},
            _described(),
            _VIEW,
            {"outcome": "ok", "frame_refs": [], "component": recreated},
        ],
        strict=True,
    )
    edit_model = ScriptedEditModel(
        {
            "text": "Rebuilt.",
            "patch": None,
            "recreate": {
                "type": "line_chart",
                "title": "T",
                "select": {"stat_var": "dcid:Aid"},
            },
        }
    )
    app = _app(
        ScriptedResolver([_envelope(choice_points=2)]),
        executor,
        edit_model=edit_model,
        generative=_plan_generative(),
        tool_surface=ScriptedSurface(),
    )
    async with _client(app) as client:
        await _conversation_with_component(client)
        frames = await _stream(client, "/api/studio/turn", _edit_body("make it a line chart"))

    seqs = [frame["seq"] for frame in frames]
    assert seqs == sorted(seqs) and len(set(seqs)) == len(seqs), f"seq not monotonic: {seqs}"


async def test_a_crashing_edit_model_becomes_a_stream_fatal_error_frame() -> None:
    class ExplodingEditModel:
        async def __call__(self, **kwargs: Any) -> dict[str, Any]:
            raise ValueError("edit drafted BOTH a patch and a recreate; exactly one is legal")

    executor = ScriptedExecutor(
        [
            {"outcome": "ok", "frame_refs": [REF_A]},
            {"outcome": "ok", "frame_refs": [REF_A], "component": COMPONENT},
            _described(),
            _VIEW,
        ],
        strict=True,
    )
    app = _app(
        ScriptedResolver([_envelope(choice_points=2)]),
        executor,
        edit_model=ExplodingEditModel(),
        generative=_plan_generative(),
        tool_surface=ScriptedSurface(),
    )
    async with _client(app) as client:
        await _conversation_with_component(client)
        frames = await _stream(client, "/api/studio/turn", _edit_body("break please"))

    assert _events(frames)[-1] == "error"
    assert "done" not in _events(frames)


async def test_answer_without_a_payload_is_refused() -> None:
    resolver = ScriptedResolver([_candidates_envelope(), _envelope(spec_id=SPEC_A)])
    async with _client(_app(resolver, ScriptedExecutor())) as client:
        first = await _stream(client, "/api/studio/turn", _turn_body("ambiguous q"))
        elicitation = _frame(first, "elicitation.request")
        response = await client.post(
            "/api/studio/interrupt",
            json=_interrupt_body(elicitation["interrupt_id"], "answer"),
            headers=AUTH,
        )
        assert response.status_code == 422
        assert "requires a text payload" in response.json()["detail"]
        # the interrupt is still pending after the refusal; skipping works
        second = await _stream(
            client,
            "/api/studio/interrupt",
            _interrupt_body(elicitation["interrupt_id"], "skip"),
        )
    assert _events(second)[-1] == "done"


async def test_approve_with_a_payload_is_refused() -> None:
    executor = ScriptedExecutor()
    async with _client(_app(ScriptedResolver([_envelope(choice_points=2)]), executor)) as client:
        first = await _stream(client, "/api/studio/turn", _turn_body("q with forks"))
        plan = _frame(first, "plan.proposed")
        response = await client.post(
            "/api/studio/interrupt",
            json=_interrupt_body(plan["interrupt_id"], "approve", "stray"),
            headers=AUTH,
        )
    assert response.status_code == 422
    assert "does not take a payload" in response.json()["detail"]


async def test_adding_an_already_composed_component_mints_no_revision() -> None:
    executor = ScriptedExecutor(
        [
            {"outcome": "ok", "frame_refs": [REF_A]},
            {"outcome": "ok", "frame_refs": [REF_A], "component": COMPONENT},
        ]
    )
    app = _app(
        ScriptedResolver([_envelope(choice_points=2)]),
        executor,
        generative=_plan_generative(),
        tool_surface=ScriptedSurface(),
    )
    async with _client(app) as client:
        manifest = await _conversation_with_component(client)
        response = await client.post(
            "/api/studio/composition",
            json=_mutation_body(manifest, [{"op": "add", "component_id": "cmp_01test0001"}]),
            headers=AUTH,
        )
    assert response.status_code == 200
    assert response.json()["revision"] == manifest["revision"]  # no spurious edit conflict


async def test_a_failing_operation_applies_nothing() -> None:
    """Transactional mutate: a valid pin followed by a missing remove leaves
    the manifest untouched (revision AND pinned state)."""
    executor = ScriptedExecutor(
        [
            {"outcome": "ok", "frame_refs": [REF_A]},
            {"outcome": "ok", "frame_refs": [REF_A], "component": COMPONENT},
        ]
    )
    app = _app(
        ScriptedResolver([_envelope(choice_points=2)]),
        executor,
        generative=_plan_generative(),
        tool_surface=ScriptedSurface(),
    )
    async with _client(app) as client:
        manifest = await _conversation_with_component(client)
        response = await client.post(
            "/api/studio/composition",
            json=_mutation_body(
                manifest,
                [
                    {"op": "pin", "component_id": "cmp_01test0001"},
                    {"op": "remove", "component_id": "cmp_missing"},
                ],
            ),
            headers=AUTH,
        )
        assert response.status_code == 404
        # nothing applied: pin again from the SAME revision succeeds
        retry = await client.post(
            "/api/studio/composition",
            json=_mutation_body(manifest, [{"op": "pin", "component_id": "cmp_01test0001"}]),
            headers=AUTH,
        )
    assert retry.status_code == 200
    assert retry.json()["revision"] == manifest["revision"] + 1


async def test_a_concurrent_turn_on_a_streaming_conversation_is_409() -> None:
    """The lock is taken eagerly in the route: a second request cannot slip
    past the guard while the first is still streaming."""
    entered = asyncio.Event()

    class HangingResolver:
        async def __call__(
            self, *, question: str, spec_id: str | None = None, wide: bool = False
        ) -> dict[str, Any]:
            entered.set()
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

    studio = StudioRunner(
        create_workflow=lambda recorder, surface: create_root_workflow(
            resolver=HangingResolver(),  # type: ignore[arg-type]
            executor=recorder,
            tool_surface=surface,
            fetch_series_stats=scripted_fetch_series_stats,
        ),
        executor=ScriptedExecutor(),
        session_service=_new_session_service(),
    )
    app = FastAPI()
    app.include_router(create_studio_router(studio, settings=Settings()))
    async with _client(app) as client:
        stream_task = asyncio.ensure_future(_stream(client, "/api/studio/turn", _turn_body()))
        await asyncio.wait_for(entered.wait(), timeout=5)
        second = await client.post("/api/studio/turn", json=_turn_body("again"), headers=AUTH)
        assert second.status_code == 409
        cancel = await client.post(
            "/api/studio/cancel",
            json={"kind": "turn.cancel", "schema_version": 1, "conversation_id": CONVERSATION},
            headers=AUTH,
        )
        assert cancel.status_code == 204
        await asyncio.wait_for(stream_task, timeout=5)


async def test_the_lock_is_held_from_the_route_not_from_the_first_frame() -> None:
    """Eagerness itself, which the route test above cannot see: a second
    acquisition is refused before any stream has started, and only for the
    conversation that holds the lock."""
    store = StudioStore(_new_session_service())

    first = await _acquire_or_conflict(store, "u_a", CONVERSATION)
    assert isinstance(first, asyncio.Lock)
    assert first.locked()

    second = await _acquire_or_conflict(store, "u_a", CONVERSATION)
    assert isinstance(second, JSONResponse)
    assert second.status_code == 409
    assert json.loads(second.body)["error"] == "transient_unavailable"

    other = await _acquire_or_conflict(store, "u_a", "conv_01test0009")
    assert isinstance(other, asyncio.Lock)
    assert other is not first

    first.release()
    third = await _acquire_or_conflict(store, "u_a", CONVERSATION)
    assert third is first


async def test_closing_at_a_heartbeat_unwinds_cleanly() -> None:
    """Regression: aclose() during a heartbeat yield raised GeneratorExit at
    a point that skipped the unwind, leaving the source 'already running'."""
    closed = asyncio.Event()

    async def segment():
        try:
            await asyncio.Event().wait()  # hang BEFORE any frame
            yield None
        finally:
            closed.set()

    conversation = ConversationState(conversation_id=CONVERSATION)
    stream = _sse_stream(segment(), conversation=conversation, heartbeat_seconds=0.02)
    first = await anext(stream)
    assert first.startswith(": ka")
    await stream.aclose()
    await asyncio.wait_for(closed.wait(), timeout=2)
    assert conversation.cancel_event is None


# --- conversation history: turn records, digest, recall, plain refusals -------


class RecallingConverse(RecordingConverse):
    """Asks for recall on EVERY call it can -- proving the round cap is
    structural (the node grants exactly one round, whatever the model says)."""

    def __init__(self, scope: str = "conversation") -> None:
        super().__init__()
        self.scope = scope
        self.calls = 0

    async def __call__(self, **kwargs: Any) -> dict[str, Any]:
        self.calls += 1
        drafted = await super().__call__(**kwargs)
        drafted["recall"] = {"scope": self.scope, "query": "forks"}
        return drafted


class ScriptedRefusal:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def __call__(self, *, question: str, status: str, warnings: list[str]) -> dict[str, Any]:
        self.calls.append({"question": question, "status": status, "warnings": warnings})
        return {"text": "I couldn't answer that -- did you mean military spending, or debt?"}


async def test_turn_two_receives_the_conversation_digest() -> None:
    converse = RecordingConverse()
    executor = ScriptedExecutor(
        [
            {"outcome": "ok", "frame_refs": [REF_A]},
            {"outcome": "ok", "frame_refs": [REF_A], "component": COMPONENT},
        ]
    )
    app = _app(
        ScriptedResolver([_envelope(choice_points=2), _envelope()]),
        executor,
        generative=GenerativeNodes(
            converse=converse, planner=ScriptedPlannerModel(), compose=ScriptedCompose()
        ),
        tool_surface=ScriptedSurface(),
    )
    async with _client(app) as client:
        await _conversation_with_component(client)
        await _stream(client, "/api/studio/turn", _turn_body("now the same for Uganda"))

    assert converse.histories[0] is None  # turn 1: no history yet
    history = converse.histories[-1]
    assert history is not None
    assert history["total_turns"] == 1
    (record,) = history["turns"]
    assert record["user"] == "q with forks"
    assert record["cards"] == [
        {
            "component_id": "cmp_01test0001",
            "type": "stat_card",
            "title": "Aid to Kenya",
            "action": "created",
        }
    ]


class GreetingConverse(RecordingConverse):
    """Answers "hi" itself (`about: "reply"`); anything else is a data turn."""

    async def __call__(self, *, question: str, **kwargs: Any) -> dict[str, Any]:
        drafted = await super().__call__(question=question, **kwargs)
        if question != "hi":
            return drafted
        return {"text": "Hello. Ask me about aid.", "search_parts": [], "about": "reply"}


async def test_a_reply_closes_the_turn_without_search_and_is_kept_in_history() -> None:
    converse = GreetingConverse()
    resolver = ScriptedResolver([_envelope()])
    app = _app(
        resolver,
        ScriptedExecutor([{"outcome": "ok", "frame_refs": [REF_A]}]),
        generative=GenerativeNodes(converse=converse),
    )
    async with _client(app) as client:
        frames = await _stream(client, "/api/studio/turn", _turn_body("hi"))
        await _stream(client, "/api/studio/turn", _turn_body("aid to Kenya"))

    assert "activity.step" not in _events(frames)
    assert _events(frames)[-1] == "done"
    texts = [(f["event"], f["text"]) for f in frames if f["event"].startswith("message.")]
    assert texts == [
        ("message.delta", "Hello. Ask me about aid."),
        ("message.complete", "Hello. Ask me about aid."),
    ]
    assert len(resolver.calls) == 1  # the second turn's search only
    (record,) = converse.histories[-1]["turns"]
    assert record == {"user": "hi", "assistant": "Hello. Ask me about aid."}


async def test_a_failed_turn_is_recorded_with_its_error() -> None:
    class ExplodingResolver:
        async def __call__(self, **kwargs: Any) -> dict[str, Any]:
            raise RuntimeError("boom")

    session_service = _new_session_service()
    store = StudioStore(session_service, app_name=AGENT_NAME)
    studio = StudioRunner(
        create_workflow=lambda recorder, surface: create_root_workflow(
            resolver=ExplodingResolver(),
            executor=recorder,
            tool_surface=surface,
            fetch_series_stats=scripted_fetch_series_stats,
        ),
        executor=ScriptedExecutor([]),
        session_service=session_service,
        store=store,
    )
    app = FastAPI()
    app.include_router(create_studio_router(studio, settings=Settings()))
    async with _client(app) as client:
        frames = await _stream(client, "/api/studio/turn", _turn_body("q that dies"))
        assert frames[-1]["event"] == "error"

    principal = "u_" + hashlib.sha256(b"test-token").hexdigest()[:16]
    conversation = store.current(principal, CONVERSATION)
    (record,) = conversation.turns
    assert record["user"] == "q that dies"
    assert record["error"] == "transient_unavailable"


async def test_converse_recall_is_served_once_and_bounded() -> None:
    converse = RecallingConverse()
    executor = ScriptedExecutor(
        [
            {"outcome": "ok", "frame_refs": [REF_A]},
            {"outcome": "ok", "frame_refs": [REF_A], "component": COMPONENT},
            {"outcome": "ok", "frame_refs": [REF_A]},
            {"outcome": "ok", "frame_refs": [REF_A], "component": COMPONENT},
        ]
    )
    app = _app(
        ScriptedResolver([_envelope(choice_points=2), _envelope()]),
        executor,
        generative=GenerativeNodes(
            converse=converse, planner=ScriptedPlannerModel(), compose=ScriptedCompose()
        ),
        tool_surface=ScriptedSurface(),
    )
    async with _client(app) as client:
        await _conversation_with_component(client)
        converse.calls = 0
        await _stream(client, "/api/studio/turn", _turn_body("about the kenya aid question"))

    # Turn 2: initial call + exactly ONE recall round, though the scripted
    # converse asks every time.
    assert converse.calls == 2
    served = converse.recall_results[-1]
    assert served is not None and served["scope"] == "conversation"
    assert served["total_turns"] == 1
    assert served["turns"][0]["user"] == "q with forks"


async def test_recall_canvas_scope_serves_full_fact_summaries() -> None:
    state = ConversationState(conversation_id=CONVERSATION)
    state.components["cmp_01test0001"] = {
        "type": "line_chart",
        "title": "Aid to Kenya",
        "revision": 1,
        "fact_summary": COMPONENT["fact_summary"],
    }
    served = state.recall_state(scope="canvas", query="kenya")

    assert served["total_cards"] == 1
    assert served["cards"][0]["fact_summary"]["series"][0]["label"] == "Kenya"

    missed = state.recall_state(scope="canvas", query="zanzibar")
    assert missed["cards"] == []


def test_the_conversation_digest_is_bounded() -> None:
    state = ConversationState(conversation_id=CONVERSATION)
    for index in range(12):
        state.turns.append({"user": f"question {index}", "assistant": "a" * 400})

    digest = state.conversation_digest_state()

    assert digest is not None
    assert digest["total_turns"] == 12
    assert len(digest["turns"]) == 10  # newest window
    assert digest["turns"][0]["user"] == "question 2"
    assert all(len(turn["assistant"]) <= 280 for turn in digest["turns"])
    assert len(json.dumps(digest["turns"])) <= 5_000


async def test_a_deadend_close_is_plain_language_with_diagnostics_on_the_step() -> None:
    refusal = ScriptedRefusal()
    diagnostic = "chosen shape cosine 0.66 is weak (< 0.75)"
    resolver = ScriptedResolver(
        [
            _envelope(
                status="unsupported",
                spec_id=None,
                warnings=[{"code": "weak_match", "severity": "warn", "message": diagnostic}],
            )
        ]
    )
    app = _app(resolver, ScriptedExecutor([]), generative=GenerativeNodes(refusal=refusal))
    async with _client(app) as client:
        frames = await _stream(client, "/api/studio/turn", _turn_body("spending in Kenya"))

    message = _frame(frames, "message.complete")
    assert message["text"] == "I couldn't answer that -- did you mean military spending, or debt?"
    assert "cosine" not in message["text"]
    resolve = _frame(frames, "activity.step")
    assert diagnostic in (resolve["detail"] or "")
    assert refusal.calls == [
        {"question": "spending in Kenya", "status": "unsupported", "warnings": [diagnostic]}
    ]
    assert frames[-1]["event"] == "done"


async def test_a_deadend_without_a_refusal_seam_stays_plain() -> None:
    resolver = ScriptedResolver(
        [
            _envelope(
                status="unsupported",
                spec_id=None,
                warnings=[{"code": "weak_match", "severity": "warn", "message": "cosine 0.66"}],
            )
        ]
    )
    app = _app(resolver, ScriptedExecutor([]))
    async with _client(app) as client:
        frames = await _stream(client, "/api/studio/turn", _turn_body("spending in Kenya"))

    message = _frame(frames, "message.complete")
    assert "cannot be expressed" in message["text"]
    assert "cosine" not in message["text"]


# --- codex-round fixes: recorder isolation, terminal-path records -------------


async def test_two_streaming_conversations_keep_their_steps_apart() -> None:
    """Through the routes: both conversations sit inside their fetch at the
    same moment, and each stream still reports only its own call."""
    both_inside = asyncio.Barrier(2)

    class MeetingExecutor:
        def __init__(self) -> None:
            self.calls: list[str] = []

        async def __call__(
            self, *, tool: str, args: dict[str, Any], on_progress: ProgressCallback | None = None
        ) -> dict[str, Any]:
            self.calls.append(current_conversation_id.get())
            await both_inside.wait()
            spec = args["spec_id"]
            return {
                "outcome": "ok",
                "frame_refs": [{"spec_id": spec, "content_version": "cv_" + "1" * 32}],
                "summary": f"fetched {spec}",
            }

    class ByQuestionResolver:
        async def __call__(
            self, *, question: str, spec_id: str | None = None, wide: bool = False
        ) -> dict[str, Any]:
            return _envelope(spec_id=SPEC_B if "uganda" in question else SPEC_A)

    executor = MeetingExecutor()
    app = _app(ByQuestionResolver(), executor)  # type: ignore[arg-type]
    other = "conv_01test0002"
    async with _client(app) as client:
        frames_a, frames_b = await asyncio.wait_for(
            asyncio.gather(
                _stream(client, "/api/studio/turn", _turn_body("kenya oda 2023")),
                _stream(
                    client,
                    "/api/studio/turn",
                    {**_turn_body("uganda oda 2023"), "conversation_id": other},
                ),
            ),
            timeout=10,
        )

    principal = "u_" + hashlib.sha256(b"test-token").hexdigest()[:16]
    assert sorted(executor.calls) == [f"{principal}/{CONVERSATION}", f"{principal}/{other}"]

    def fetch_details(frames: list[dict[str, Any]]) -> list[str]:
        return [
            f["detail"]
            for f in frames
            if f["event"] == "activity.step" and f.get("tool") == "get_data"
        ]

    assert fetch_details(frames_a) == [f"fetched {SPEC_A}"]
    assert fetch_details(frames_b) == [f"fetched {SPEC_B}"]
    assert _events(frames_a)[-1] == "done"
    assert _events(frames_b)[-1] == "done"


async def test_the_recorder_keeps_concurrent_conversations_apart() -> None:
    async def executor(
        *, tool: str, args: dict[str, Any], on_progress: ProgressCallback | None = None
    ) -> dict[str, Any]:
        return {"outcome": "ok", "frame_refs": []}

    recorder = _RecordingExecutor(executor, store=StudioStore(_new_session_service()))
    current_conversation_id.set("u_a/conv_a")
    await recorder(tool="get_data", args={"n": 1})
    current_conversation_id.set("u_b/conv_b")
    await recorder(tool="get_data", args={"n": 2})
    current_conversation_id.set("u_a/conv_a")
    await recorder(tool="get_data", args={"n": 3})

    a_records = recorder.drain("u_a/conv_a")
    assert [record.args["n"] for record in a_records] == [1, 3]
    # B's record survives A's drain and comes out only for B.
    b_records = recorder.drain("u_b/conv_b")
    assert [record.args["n"] for record in b_records] == [2]
    assert recorder.drain("u_a/conv_a") == []


async def test_the_recorder_closes_a_call_that_raises_or_is_cancelled_as_a_failed_step() -> None:
    started = asyncio.Event()

    async def executor(
        *, tool: str, args: dict[str, Any], on_progress: ProgressCallback | None = None
    ) -> dict[str, Any]:
        if args["n"] == 1:
            raise RuntimeError("transport closed")
        started.set()
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    recorder = _RecordingExecutor(executor, store=StudioStore(_new_session_service()))
    current_conversation_id.set("u_a/conv_a")
    with pytest.raises(RuntimeError):
        await recorder(tool="get_data", args={"n": 1})
    task = asyncio.create_task(recorder(tool="get_data", args={"n": 2}))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    records = recorder.drain("u_a/conv_a")
    assert [(record.result["outcome"], record.result["refusal"]["code"]) for record in records] == [
        ("refusal", "RuntimeError"),
        ("refusal", "cancelled"),
    ]


async def test_a_rejected_plan_turn_is_recorded() -> None:
    session_service = _new_session_service()
    store = StudioStore(session_service, app_name=AGENT_NAME)
    studio = StudioRunner(
        create_workflow=lambda recorder, surface: create_root_workflow(
            resolver=ScriptedResolver([_envelope(choice_points=2)]),
            executor=recorder,
            generative=GenerativeNodes(planner=ScriptedPlannerModel()),
            tool_surface=surface,
            fetch_series_stats=scripted_fetch_series_stats,
        ),
        executor=ScriptedExecutor([]),
        session_service=session_service,
        store=store,
    )
    app = FastAPI()
    app.include_router(create_studio_router(studio, settings=Settings()))
    async with _client(app) as client:
        frames = await _stream(client, "/api/studio/turn", _turn_body("q with forks"))
        plan = _frame(frames, "plan.proposed")
        resumed = await _stream(
            client, "/api/studio/interrupt", _interrupt_body(plan["interrupt_id"], "reject")
        )
        assert resumed[-1]["event"] == "done"

    principal = "u_" + hashlib.sha256(b"test-token").hexdigest()[:16]
    (record,) = store.current(principal, CONVERSATION).turns
    assert record["user"] == "q with forks"
    assert "rejected" in record["assistant"].lower()
    assert "error" not in record


async def test_an_amendment_reaches_the_turn_record() -> None:
    session_service = _new_session_service()
    store = StudioStore(session_service, app_name=AGENT_NAME)
    studio = StudioRunner(
        create_workflow=lambda recorder, surface: create_root_workflow(
            resolver=ScriptedResolver([_envelope(choice_points=2)]),
            executor=recorder,
            generative=_plan_generative(),
            tool_surface=surface,
            fetch_series_stats=scripted_fetch_series_stats,
        ),
        executor=ScriptedExecutor(
            [
                {"outcome": "ok", "frame_refs": [REF_A]},
                {"outcome": "ok", "frame_refs": [REF_A], "component": COMPONENT},
            ]
        ),
        session_service=session_service,
        store=store,
        tool_surface=ScriptedSurface(),
    )
    app = FastAPI()
    app.include_router(create_studio_router(studio, settings=Settings()))
    async with _client(app) as client:
        frames = await _stream(client, "/api/studio/turn", _turn_body("q with forks"))
        plan = _frame(frames, "plan.proposed")
        amended = await _stream(
            client,
            "/api/studio/interrupt",
            _interrupt_body(plan["interrupt_id"], "amend", "only since 2015"),
        )
        replan = _frame(amended, "plan.proposed")
        final = await _stream(
            client, "/api/studio/interrupt", _interrupt_body(replan["interrupt_id"], "approve")
        )
        assert final[-1]["event"] == "done"

    principal = "u_" + hashlib.sha256(b"test-token").hexdigest()[:16]
    (record,) = store.current(principal, CONVERSATION).turns
    assert "only since 2015" in record["user"]


async def test_a_superseded_pause_is_recorded_as_abandoned() -> None:
    session_service = _new_session_service()
    store = StudioStore(session_service, app_name=AGENT_NAME)
    studio = StudioRunner(
        create_workflow=lambda recorder, surface: create_root_workflow(
            resolver=ScriptedResolver([_envelope(choice_points=2), _envelope()]),
            executor=recorder,
            generative=GenerativeNodes(planner=ScriptedPlannerModel()),
            tool_surface=surface,
            fetch_series_stats=scripted_fetch_series_stats,
        ),
        executor=ScriptedExecutor(
            [
                {"outcome": "ok", "frame_refs": [REF_A]},
                {"outcome": "ok", "frame_refs": [REF_A]},
                {"outcome": "ok", "frame_refs": [REF_A], "component": COMPONENT},
            ]
        ),
        session_service=session_service,
        store=store,
    )
    app = FastAPI()
    app.include_router(create_studio_router(studio, settings=Settings()))
    async with _client(app) as client:
        paused = await _stream(client, "/api/studio/turn", _turn_body("q with forks"))
        assert paused[-1]["event"] == "plan.proposed"
        # A fresh turn supersedes the pause; the abandoned turn must not
        # vanish from history.
        second = await _stream(client, "/api/studio/turn", _turn_body("aid to kenya, quickly"))
        assert second[-1]["event"] == "done"

    principal = "u_" + hashlib.sha256(b"test-token").hexdigest()[:16]
    records = store.current(principal, CONVERSATION).turns
    assert [record["user"] for record in records] == ["q with forks", "aid to kenya, quickly"]
    assert records[0]["error"] == "abandoned"
    assert "error" not in records[1]


async def test_a_recreated_edit_card_reaches_the_turn_record() -> None:
    class TypeSwitchEdit:
        async def __call__(self, **kwargs: Any) -> dict[str, Any]:
            return {
                "text": "Changed it to a bar chart.",
                "patch": None,
                "recreate": {"type": "bar_chart", "title": "Aid to Kenya", "select": {"s": "v"}},
            }

    rebuilt = {**COMPONENT, "component_id": "cmp_01test0002", "type": "bar_chart"}
    session_service = _new_session_service()
    store = StudioStore(session_service, app_name=AGENT_NAME)
    studio = StudioRunner(
        create_workflow=lambda recorder, surface: create_root_workflow(
            resolver=ScriptedResolver([_envelope(choice_points=2)]),
            executor=recorder,
            generative=_plan_generative(),
            tool_surface=surface,
            fetch_series_stats=scripted_fetch_series_stats,
        ),
        executor=ScriptedExecutor(
            [
                {"outcome": "ok", "frame_refs": [REF_A]},
                {"outcome": "ok", "frame_refs": [REF_A], "component": COMPONENT},
                {
                    "outcome": "ok",
                    "frame_refs": [],
                    "component_body": {"stored_intent": {"frame_ref": REF_A}},
                },
                {"outcome": "ok", "frame_refs": []},
                {"outcome": "ok", "frame_refs": [], "component": rebuilt},
            ]
        ),
        session_service=session_service,
        store=store,
        edit_model=TypeSwitchEdit(),
        tool_surface=ScriptedSurface(),
    )
    app = FastAPI()
    app.include_router(create_studio_router(studio, settings=Settings()))
    async with _client(app) as client:
        await _conversation_with_component(client)
        body = _turn_body("make it a bar chart")
        body["component_ref"] = {
            "component_id": "cmp_01test0001",
            "revision": 1,
            "intent": "edit",
        }
        frames = await _stream(client, "/api/studio/turn", body)
        assert _frame(frames, "component.created")["component_id"] == "cmp_01test0002"

    principal = "u_" + hashlib.sha256(b"test-token").hexdigest()[:16]
    records = store.current(principal, CONVERSATION).turns
    assert records[-1]["cards"] == [
        {
            "component_id": "cmp_01test0002",
            "type": "bar_chart",
            "title": "Aid to Kenya",
            "action": "recreated",
        }
    ]


async def test_a_recreated_edit_cards_resolved_window_reaches_the_turn_record() -> None:
    """The rebuilt card's own resolved span (`fact_summary.window`) rides
    its `turn_components` entry into `conversation.turns`, the same way
    `land_component` attaches `window` off the digest entry's `time`, so a
    card last touched through the Edit Card route still feeds
    `ConversationState.anaphor_window`."""

    class TypeSwitchEdit:
        async def __call__(self, **kwargs: Any) -> dict[str, Any]:
            return {
                "text": "Changed it to a bar chart.",
                "patch": None,
                "recreate": {"type": "bar_chart", "title": "Aid to Kenya", "select": {"s": "v"}},
            }

    rebuilt = {
        **COMPONENT,
        "component_id": "cmp_01test0002",
        "type": "bar_chart",
        "fact_summary": {
            **COMPONENT["fact_summary"],
            "window": {"start": "2018", "end": "2024"},
        },
    }
    session_service = _new_session_service()
    store = StudioStore(session_service, app_name=AGENT_NAME)
    studio = StudioRunner(
        create_workflow=lambda recorder, surface: create_root_workflow(
            resolver=ScriptedResolver([_envelope(choice_points=2)]),
            executor=recorder,
            generative=_plan_generative(),
            tool_surface=surface,
            fetch_series_stats=scripted_fetch_series_stats,
        ),
        executor=ScriptedExecutor(
            [
                {"outcome": "ok", "frame_refs": [REF_A]},
                {"outcome": "ok", "frame_refs": [REF_A], "component": COMPONENT},
                {
                    "outcome": "ok",
                    "frame_refs": [],
                    "component_body": {"stored_intent": {"frame_ref": REF_A}},
                },
                {"outcome": "ok", "frame_refs": []},
                {"outcome": "ok", "frame_refs": [], "component": rebuilt},
            ]
        ),
        session_service=session_service,
        store=store,
        edit_model=TypeSwitchEdit(),
        tool_surface=ScriptedSurface(),
    )
    app = FastAPI()
    app.include_router(create_studio_router(studio, settings=Settings()))
    async with _client(app) as client:
        await _conversation_with_component(client)
        body = _turn_body("make it a bar chart")
        body["component_ref"] = {
            "component_id": "cmp_01test0001",
            "revision": 1,
            "intent": "edit",
        }
        frames = await _stream(client, "/api/studio/turn", body)
        assert _frame(frames, "component.created")["component_id"] == "cmp_01test0002"

    principal = "u_" + hashlib.sha256(b"test-token").hexdigest()[:16]
    records = store.current(principal, CONVERSATION).turns
    assert records[-1]["cards"] == [
        {
            "component_id": "cmp_01test0002",
            "type": "bar_chart",
            "title": "Aid to Kenya",
            "action": "recreated",
            "window": "2018-2024",
        }
    ]


# --- "add X to the chart": the message path swaps the block, end to end -----


async def test_add_to_the_chart_replaces_the_existing_card_with_a_two_series_one() -> None:
    """The full scripted wiring runs with `STUDIO_SCRIPTED=true`
    (`harness.scripted`'s `ScriptedConverse`, `ScriptedCompose`,
    `ScriptedResolver`, and `ScriptedExecutor`, no local test fakes). A
    direct fetch draws one card, then "add Guatemala to the chart", a
    question `ScriptedResolver` resolves to the Kenya+Guatemala frame and
    `ScriptedConverse` reads as naming the newest canvas block. The canvas
    stays at one block, now the two-series card, in the first card's
    place."""
    session_service = _new_session_service()
    store = StudioStore(session_service, app_name=AGENT_NAME)
    studio = StudioRunner(
        create_workflow=lambda recorder, surface: create_root_workflow(
            resolver=MessagePathScriptedResolver(),
            executor=recorder,
            generative=GenerativeNodes(converse=ScriptedConverse(), compose=ScriptedCompose()),
            tool_surface=surface,
            fetch_series_stats=scripted_fetch_series_stats,
        ),
        executor=MessagePathScriptedExecutor(),
        session_service=session_service,
        store=store,
    )
    app = FastAPI()
    app.include_router(create_studio_router(studio, settings=Settings()))
    async with _client(app) as client:
        first = await _stream(client, "/api/studio/turn", _turn_body("Life expectancy in Kenya"))
        assert _events(first)[-1] == "done"
        first_manifest = _frame(first, "done")["composition"]
        assert len(first_manifest["blocks"]) == 1

        second = await _stream(client, "/api/studio/turn", _turn_body("add Guatemala to the chart"))

    assert _events(second)[-1] == "done"
    second_created = _frame(second, "component.created")
    assert {series["label"] for series in second_created["fact_summary"]["series"]} == {
        "Guatemala",
        "Kenya",
    }
    second_manifest = _frame(second, "done")["composition"]
    # One block. The two-series card replaced the first card in its slot.
    assert len(second_manifest["blocks"]) == 1
    assert second_manifest["blocks"][0]["component_id"] == second_created["component_id"]

    principal = "u_" + hashlib.sha256(b"test-token").hexdigest()[:16]
    turns = store.current(principal, CONVERSATION).turns
    assert turns[-1]["cards"][0]["action"] == "replaced"


class _DistinctIdExecutor:
    """Wraps `MessagePathScriptedExecutor`. The real stand-in hardcodes
    `"cmp_scripted0002"` for every fresh `write_component` call, so two
    turns' cards would otherwise land under the same id and a set of
    component ids could not tell "one block survived" from "two blocks
    happen to share a name". Each fresh write (never a revise patch, which
    already names its own `component_id` in `args`) gets the next
    `cmp_distinctNNNN` id instead, on both the top-level component and its
    nested `fact_summary`."""

    def __init__(self) -> None:
        self._inner = MessagePathScriptedExecutor()
        self._writes = 0

    async def __call__(
        self, *, tool: str, args: dict[str, Any], on_progress: ProgressCallback | None = None
    ) -> dict[str, Any]:
        result = await self._inner(tool=tool, args=args)
        component = result.get("component")
        if tool == "write_component" and "component_id" not in args and isinstance(component, dict):
            self._writes += 1
            new_id = f"cmp_distinct{self._writes:04d}"
            component = {
                **component,
                "component_id": new_id,
                "fact_summary": {**component["fact_summary"], "component_id": new_id},
            }
            result = {**result, "component": component}
        return result


class _CardVanishesMidTurnCompose:
    """Wraps `ScriptedCompose`. Right when compose runs, after converse
    already validated `revises` against the turn-start canvas but before
    the write, it removes the named target's own block -- not the whole
    canvas -- off the conversation's composition, and PERSISTS that removal
    (a real `/composition` call would). This stands in for a concurrent
    `/composition` call removing just that card, since that route runs
    without the turn lock so the canvas can change mid-turn. The checkpoint
    `StudioRunner._checkpointed` runs before `done` and reconciles against
    the LEDGER's own composition, not this instance's in-process copy, so
    the removal must actually be saved for the reconcile to see it."""

    def __init__(
        self, store: StudioStore, principal: str, conversation_id: str, *, target: str
    ) -> None:
        self._inner = ScriptedCompose()
        self._store = store
        self._principal = principal
        self._conversation_id = conversation_id
        self._target = target

    async def __call__(self, **kwargs: Any) -> dict[str, Any]:
        conversation = self._store.current(self._principal, self._conversation_id)
        if conversation.composition is not None:
            blocks = [
                block
                for block in conversation.composition.blocks
                if block.component_id != self._target
            ]
            conversation.composition = conversation.composition.model_copy(
                update={"blocks": blocks}
            )
            await self._store.save(self._principal, self._conversation_id, keys={_KEY_COMPOSITION})
        return await self._inner(**kwargs)


async def test_add_to_the_chart_appends_instead_when_the_target_vanishes_mid_turn() -> None:
    """The target block is removed (`/composition`) between converse's
    turn-start validation and compose's write landing. The canvas ends
    with only the new card: the old one, the sole block, is the one that
    vanished, and a target already gone cannot be swapped. The turn record
    says `created`, not `replaced`, because `land_component`'s own check,
    made synchronously right after the write returns, catches the removal
    in time for narrate's `components_replaced` fact to say so too."""
    session_service = _new_session_service()
    store = StudioStore(session_service, app_name=AGENT_NAME)
    principal = "u_" + hashlib.sha256(b"test-token").hexdigest()[:16]
    executor = _DistinctIdExecutor()
    studio = StudioRunner(
        create_workflow=lambda recorder, surface: create_root_workflow(
            resolver=MessagePathScriptedResolver(),
            executor=recorder,
            generative=GenerativeNodes(
                converse=ScriptedConverse(),
                compose=_CardVanishesMidTurnCompose(
                    store, principal, CONVERSATION, target="cmp_distinct0001"
                ),
            ),
            tool_surface=surface,
            fetch_series_stats=scripted_fetch_series_stats,
        ),
        executor=executor,
        session_service=session_service,
        store=store,
    )
    app = FastAPI()
    app.include_router(create_studio_router(studio, settings=Settings()))
    async with _client(app) as client:
        first = await _stream(client, "/api/studio/turn", _turn_body("Life expectancy in Nigeria"))
        assert _events(first)[-1] == "done"
        first_created = _frame(first, "component.created")
        assert first_created["component_id"] == "cmp_distinct0001"

        second = await _stream(client, "/api/studio/turn", _turn_body("add Guatemala to the chart"))

    assert _events(second)[-1] == "done"
    second_created = _frame(second, "component.created")
    assert second_created["component_id"] == "cmp_distinct0002"
    second_manifest = _frame(second, "done")["composition"]
    # One block: the new card only. The old one, removed mid-turn, does
    # not survive alongside it.
    assert len(second_manifest["blocks"]) == 1
    assert second_manifest["blocks"][0]["component_id"] == "cmp_distinct0002"
    assert first_created["component_id"] not in {
        block["component_id"] for block in second_manifest["blocks"]
    }

    turns = store.current(principal, CONVERSATION).turns
    assert turns[-1]["cards"][0]["action"] == "created"


async def test_add_to_the_chart_replace_action_survives_removal_between_landing_and_drain() -> None:
    """`land_component` decides and mutates atomically, right after the
    write returns, inside `compose` -- so a removal that lands strictly
    AFTER that call (here: at the start of `narrate`, which only runs once
    `compose` has already finished) does not desync what narrate says
    from what the recorder reports. Both come from the one stamped action,
    made before the removal, not re-derived afterward: narrate's
    `components_replaced` and the turn record's `action` say
    `replaced`, even though the manifest the client actually receives at
    `done` already reflects the later removal."""
    session_service = _new_session_service()
    store = StudioStore(session_service, app_name=AGENT_NAME)
    principal = "u_" + hashlib.sha256(b"test-token").hexdigest()[:16]
    executor = _DistinctIdExecutor()
    captured_facts: list[dict[str, Any]] = []

    class _RemoveNewCardNarrate:
        """Runs strictly after `compose` returns -- `land_component` has
        already landed the replace by the time this call happens -- so
        removing the just-landed card here simulates `/composition`
        interleaving in the gap between the atomic landing call and the
        translator's later drain. Persists the removal (a real
        `/composition` call would) so the checkpoint that runs before
        `done` -- reconciling against the LEDGER's own composition, not
        this instance's in-process copy -- sees it too."""

        async def __call__(self, *, question: str, facts: dict[str, Any]) -> dict[str, Any]:
            del question
            captured_facts.append(facts)
            conversation = store.current(principal, CONVERSATION)
            if conversation.composition is not None:
                blocks = [
                    block
                    for block in conversation.composition.blocks
                    if block.component_id != "cmp_distinct0002"
                ]
                conversation.composition = conversation.composition.model_copy(
                    update={"blocks": blocks}
                )
                await store.save(principal, CONVERSATION, keys={_KEY_COMPOSITION})
            return {"text": "Updated the chart."}

    studio = StudioRunner(
        create_workflow=lambda recorder, surface: create_root_workflow(
            resolver=MessagePathScriptedResolver(),
            executor=recorder,
            generative=GenerativeNodes(
                converse=ScriptedConverse(),
                compose=ScriptedCompose(),
                narrate=_RemoveNewCardNarrate(),
            ),
            tool_surface=surface,
            fetch_series_stats=scripted_fetch_series_stats,
        ),
        executor=executor,
        session_service=session_service,
        store=store,
    )
    app = FastAPI()
    app.include_router(create_studio_router(studio, settings=Settings()))
    async with _client(app) as client:
        first = await _stream(client, "/api/studio/turn", _turn_body("Life expectancy in Nigeria"))
        assert _events(first)[-1] == "done"
        assert _frame(first, "component.created")["component_id"] == "cmp_distinct0001"

        second = await _stream(client, "/api/studio/turn", _turn_body("add Guatemala to the chart"))

    assert _events(second)[-1] == "done"
    assert captured_facts[-1]["components_replaced"] is True
    second_manifest = _frame(second, "done")["composition"]
    # The removal narrate performed already landed by the time the stream
    # emits `done`: the manifest the client sees has no blocks left.
    assert second_manifest["blocks"] == []

    turns = store.current(principal, CONVERSATION).turns
    # Agrees with narrate's fact: the recorder's own action is `replaced`,
    # the true outcome of the one atomic call, not re-derived at drain
    # time against a canvas the removal has since changed.
    assert turns[-1]["cards"][0]["action"] == "replaced"


class _TwoIntentsCompose:
    """Wraps `ScriptedCompose`. For a "to the chart" turn only, drafts two
    `line_chart` intents off the turn's one final-step frame instead of
    one, both naming the same `frame_ref` -- `_admissible_intents` keeps
    every intent naming a step's produced frame eligible, not just the
    first, so both survive. Stands in for a model drawing two cards from
    data it fetched once. Any other turn (the first, seeding the canvas)
    passes through unchanged, one intent as usual."""

    def __init__(self) -> None:
        self._inner = ScriptedCompose()

    async def __call__(self, **kwargs: Any) -> dict[str, Any]:
        drafted = await self._inner(**kwargs)
        intents = drafted["intents"]
        if not intents or "to the chart" not in str(kwargs.get("question", "")):
            return drafted
        first = intents[0]
        second = {**first, "title": f"{first['title']} (second)"}
        return {"intents": [first, second]}


async def test_two_components_in_one_turn_the_first_replaces_the_second_appends() -> None:
    """`revises_target` (`harness.graph`'s `compose` node) is consumed by
    the first component that actually lands, then cleared, so a turn that
    lands two components can only ever swap the first: `land_component`
    (`harness.studio_state`) replaces the canvas's existing target in
    place for the first write, then appends for the second, whatever
    `revises` named. The manifest ends with exactly the two new cards, in
    that order, and the old one is gone; `components_replaced` and the
    turn's own recorded actions agree with it."""
    session_service = _new_session_service()
    store = StudioStore(session_service, app_name=AGENT_NAME)
    principal = "u_" + hashlib.sha256(b"test-token").hexdigest()[:16]
    executor = _DistinctIdExecutor()

    class _CapturingNarrate:
        def __init__(self) -> None:
            self.captured: list[dict[str, Any]] = []

        async def __call__(self, *, question: str, facts: dict[str, Any]) -> dict[str, Any]:
            del question
            self.captured.append(facts)
            return {"text": "Updated the chart."}

    narrate = _CapturingNarrate()
    studio = StudioRunner(
        create_workflow=lambda recorder, surface: create_root_workflow(
            resolver=MessagePathScriptedResolver(),
            executor=recorder,
            generative=GenerativeNodes(
                converse=ScriptedConverse(), compose=_TwoIntentsCompose(), narrate=narrate
            ),
            tool_surface=surface,
            fetch_series_stats=scripted_fetch_series_stats,
        ),
        executor=executor,
        session_service=session_service,
        store=store,
    )
    app = FastAPI()
    app.include_router(create_studio_router(studio, settings=Settings()))
    async with _client(app) as client:
        first = await _stream(client, "/api/studio/turn", _turn_body("Life expectancy in Nigeria"))
        assert _events(first)[-1] == "done"
        assert _frame(first, "component.created")["component_id"] == "cmp_distinct0001"

        second = await _stream(client, "/api/studio/turn", _turn_body("add Guatemala to the chart"))

    assert _events(second)[-1] == "done"
    created_ids = [
        frame["component_id"] for frame in second if frame["event"] == "component.created"
    ]
    assert created_ids == ["cmp_distinct0002", "cmp_distinct0003"]

    assert narrate.captured[-1]["components_replaced"] is True

    second_manifest = _frame(second, "done")["composition"]
    # Exactly the two new cards, in landing order; the pre-turn card the
    # first one swapped out does not survive alongside them.
    assert [block["component_id"] for block in second_manifest["blocks"]] == [
        "cmp_distinct0002",
        "cmp_distinct0003",
    ]

    turns = store.current(principal, CONVERSATION).turns
    assert [card["action"] for card in turns[-1]["cards"]] == ["replaced", "created"]


class _FirstWriteMissingFactSummaryExecutor:
    """Wraps `_DistinctIdExecutor`. The turn's first `write_component` call
    answers `ok` with a component that carries no `fact_summary` -- a
    malformed-but-`ok` response -- and every other write, this turn's
    second included, answers normally."""

    def __init__(self, *, trigger: int) -> None:
        self._inner = _DistinctIdExecutor()
        self.writes = 0
        self._trigger = trigger

    async def __call__(
        self, *, tool: str, args: dict[str, Any], on_progress: ProgressCallback | None = None
    ) -> dict[str, Any]:
        if tool != "write_component":
            return await self._inner(tool=tool, args=args)
        self.writes += 1
        result = await self._inner(tool=tool, args=args)
        if self.writes != self._trigger or not isinstance(result.get("component"), dict):
            return result
        component = dict(result["component"])
        del component["fact_summary"]
        return {**result, "component": component}


async def test_a_component_missing_fact_summary_never_lands_and_the_revise_target_survives() -> (
    None
):
    """An `ok` write whose component fails the `component.created` wire
    contract -- missing `fact_summary`, here -- must not consume the
    turn's revise target or replace the existing card. It is refused
    the same as any other unusable write, the existing card stays
    exactly as it was, and the target is still there for the next
    write this same turn that actually validates."""
    session_service = _new_session_service()
    store = StudioStore(session_service, app_name=AGENT_NAME)
    principal = "u_" + hashlib.sha256(b"test-token").hexdigest()[:16]
    executor = _FirstWriteMissingFactSummaryExecutor(trigger=2)
    studio = StudioRunner(
        create_workflow=lambda recorder, surface: create_root_workflow(
            resolver=MessagePathScriptedResolver(),
            executor=recorder,
            generative=GenerativeNodes(converse=ScriptedConverse(), compose=_TwoIntentsCompose()),
            tool_surface=surface,
            fetch_series_stats=scripted_fetch_series_stats,
        ),
        executor=executor,
        session_service=session_service,
        store=store,
    )
    app = FastAPI()
    app.include_router(create_studio_router(studio, settings=Settings()))
    async with _client(app) as client:
        first = await _stream(client, "/api/studio/turn", _turn_body("Life expectancy in Nigeria"))
        assert _events(first)[-1] == "done"
        assert _frame(first, "component.created")["component_id"] == "cmp_distinct0001"

        second = await _stream(client, "/api/studio/turn", _turn_body("add Guatemala to the chart"))

    assert _events(second)[-1] == "done"
    created_ids = [
        frame["component_id"] for frame in second if frame["event"] == "component.created"
    ]
    # Only the second write's component reaches the wire: the first,
    # missing fact_summary, produces no component.created frame at all.
    assert created_ids == ["cmp_distinct0003"]

    second_manifest = _frame(second, "done")["composition"]
    # One block: the second write's card, in the first (target) card's own
    # place -- the revise target the invalid first write never consumed.
    assert [block["component_id"] for block in second_manifest["blocks"]] == ["cmp_distinct0003"]

    principal_turns = store.current(principal, CONVERSATION).turns
    assert [card["action"] for card in principal_turns[-1]["cards"]] == ["replaced"]


class _HangsOnTheThirdWriteExecutor:
    """Wraps `_DistinctIdExecutor`. The third `write_component` call across
    the conversation -- the second write of a two-intent turn -- hangs
    forever instead of returning, standing in for a write whose response
    never lands before the client disconnects or POST /api/studio/cancel
    fires. The first write of that turn has already returned by then, so
    `harness.graph`'s `compose` has already stamped its record `land=True`
    (and, being the first usable write, consumed the revise target); the
    hang keeps `compose`'s node from ever returning, so no ADK event
    follows for either write."""

    def __init__(self) -> None:
        self._inner = _DistinctIdExecutor()
        self.writes = 0

    async def __call__(
        self, *, tool: str, args: dict[str, Any], on_progress: ProgressCallback | None = None
    ) -> dict[str, Any]:
        if tool == "write_component":
            self.writes += 1
            if self.writes == 3:
                await asyncio.Event().wait()
        return await self._inner(tool=tool, args=args)


async def test_first_write_ok_then_cancellation_before_drain_leaves_the_canvas_unchanged() -> None:
    """A cancel that lands while `compose`'s second write hangs kills the
    turn before `compose`'s node ever yields an event -- no event means
    `harness.studio_translate`'s `_component_frames` never runs for
    either write, so neither the first write's already-returned component
    nor the second lands: `harness.studio.StudioRunner._frames`'s
    `finally` drops both undrained records with the turn. The canvas is
    exactly what the first turn left it, not one card landed partway
    through the second."""
    session_service = _new_session_service()
    store = StudioStore(session_service, app_name=AGENT_NAME)
    principal = "u_" + hashlib.sha256(b"test-token").hexdigest()[:16]
    executor = _HangsOnTheThirdWriteExecutor()
    studio = StudioRunner(
        create_workflow=lambda recorder, surface: create_root_workflow(
            resolver=MessagePathScriptedResolver(),
            executor=recorder,
            generative=GenerativeNodes(converse=ScriptedConverse(), compose=_TwoIntentsCompose()),
            tool_surface=surface,
            fetch_series_stats=scripted_fetch_series_stats,
        ),
        executor=executor,
        session_service=session_service,
        store=store,
    )
    app = FastAPI()
    app.include_router(create_studio_router(studio, settings=Settings()))
    async with _client(app) as client:
        first = await _stream(client, "/api/studio/turn", _turn_body("Life expectancy in Nigeria"))
        assert _events(first)[-1] == "done"
        before = [b.component_id for b in store.current(principal, CONVERSATION).composition.blocks]

        resume = asyncio.ensure_future(
            _stream(client, "/api/studio/turn", _turn_body("add Guatemala to the chart"))
        )

        async def _wait_for_the_hang() -> None:
            while executor.writes < 3:  # first write landed, second hanging
                await asyncio.sleep(0.01)

        await asyncio.wait_for(_wait_for_the_hang(), timeout=5)
        cancel = await client.post(
            "/api/studio/cancel",
            json={"kind": "turn.cancel", "schema_version": 1, "conversation_id": CONVERSATION},
            headers=AUTH,
        )
        cancelled = await asyncio.wait_for(resume, timeout=5)

    assert cancel.status_code == 204
    assert _frame(cancelled, "error")["error"]["error"] == "cancelled"
    assert "component.created" not in _events(cancelled)
    after = [b.component_id for b in store.current(principal, CONVERSATION).composition.blocks]
    assert after == before


class _RaisesOnTheThirdWriteExecutor:
    """Wraps `_DistinctIdExecutor`. Same trigger as
    `_HangsOnTheThirdWriteExecutor`, except the second write of the
    two-intent turn raises instead of hanging -- stands in for a write
    whose call itself failed after the platform may already have
    committed it."""

    def __init__(self) -> None:
        self._inner = _DistinctIdExecutor()
        self.writes = 0

    async def __call__(
        self, *, tool: str, args: dict[str, Any], on_progress: ProgressCallback | None = None
    ) -> dict[str, Any]:
        if tool == "write_component":
            self.writes += 1
            if self.writes == 3:
                raise RuntimeError("presentation-mcp connection reset")
        return await self._inner(tool=tool, args=args)


async def test_first_write_ok_then_second_write_raises_still_announces_what_actually_landed() -> (
    None
):
    """Unlike a cancellation, ADK still surfaces the node's already-tracked
    events before it re-raises a plain exception out of `compose`, so the
    translator's drain (and the landing it performs) runs for the first
    write before the second one's failure ends the stream. The first
    write's card is announced (`component.created` plus
    `composition.updated`) and lands for real -- narrate never runs, so no
    text closes the turn, but the server's own composition is exactly what
    the client was told. The second write, which never returned, produces
    no frame and never reaches the canvas. The stream still ends with a
    stream-fatal error."""
    session_service = _new_session_service()
    store = StudioStore(session_service, app_name=AGENT_NAME)
    principal = "u_" + hashlib.sha256(b"test-token").hexdigest()[:16]
    executor = _RaisesOnTheThirdWriteExecutor()
    studio = StudioRunner(
        create_workflow=lambda recorder, surface: create_root_workflow(
            resolver=MessagePathScriptedResolver(),
            executor=recorder,
            generative=GenerativeNodes(converse=ScriptedConverse(), compose=_TwoIntentsCompose()),
            tool_surface=surface,
            fetch_series_stats=scripted_fetch_series_stats,
        ),
        executor=executor,
        session_service=session_service,
        store=store,
    )
    app = FastAPI()
    app.include_router(create_studio_router(studio, settings=Settings()))
    async with _client(app) as client:
        first = await _stream(client, "/api/studio/turn", _turn_body("Life expectancy in Nigeria"))
        assert _events(first)[-1] == "done"

        second = await _stream(client, "/api/studio/turn", _turn_body("add Guatemala to the chart"))

    assert _events(second)[-1] == "error"
    created = [frame for frame in second if frame["event"] == "component.created"]
    # Exactly the first write's card -- the second, which raised, never
    # produced a component to announce.
    assert [frame["component_id"] for frame in created] == ["cmp_distinct0002"]
    announced_manifest = _frame(second, "composition.updated")["manifest"]
    server_blocks = store.current(principal, CONVERSATION).composition.blocks
    assert [b["component_id"] for b in announced_manifest["blocks"]] == [
        b.component_id for b in server_blocks
    ]


# --- cross-instance: two StudioRunners over one shared database service ----


async def test_a_plan_paused_on_one_instance_is_approved_and_continued_on_another(
    tmp_path: Path,
) -> None:
    """A pauses on the plan. B, a separate `StudioRunner` and `StudioStore`
    that never built this plan, answers `/interrupt` and completes the
    turn, restoring the plan's step order into its own `_RecordingSurface`
    from the ledger's `_PendingInterrupt.step_order` before resuming. A
    second `/turn` on B then continues the same conversation: the turn
    index advances past what A claimed, and the digest carries A's turn."""
    principal = "u_" + hashlib.sha256(b"test-token").hexdigest()[:16]
    session_service = sqlite_session_service(tmp_path / "shared.db")
    app_a = _app(
        ScriptedResolver([_envelope(choice_points=2)]),
        ScriptedExecutor(),
        generative=_plan_generative(),
        tool_surface=ScriptedSurface(),
        session_service=session_service,
    )
    executor_b = ScriptedExecutor(
        [
            {"outcome": "ok", "frame_refs": [REF_A]},
            {"outcome": "ok", "frame_refs": [REF_A], "component": COMPONENT},
        ]
    )
    app_b = _app(
        # B's own resolver is called exactly once: for the SECOND `/turn`
        # below. B never resolves the first turn's question at all --
        # `/interrupt` resumes the ADK invocation A's own `/turn` already
        # started, in the per-turn session A created.
        ScriptedResolver([_envelope()]),
        executor_b,
        generative=_plan_generative(),
        tool_surface=ScriptedSurface(),
        session_service=session_service,
    )

    async with _client(app_a) as client_a, _client(app_b) as client_b:
        first = await _stream(client_a, "/api/studio/turn", _turn_body("q with forks"))
        plan = _frame(first, "plan.proposed")
        assert app_b.state.studio.surface_recorder.step_order(f"{principal}/{CONVERSATION}") == []

        resumed = await _stream(
            client_b, "/api/studio/interrupt", _interrupt_body(plan["interrupt_id"], "approve")
        )
        assert _events(resumed)[-1] == "done"
        assert app_b.state.studio.surface_recorder.step_order(f"{principal}/{CONVERSATION}") == [
            step["id"] for step in plan["steps"]
        ]

        second = await _stream(client_b, "/api/studio/turn", _turn_body("a follow-up question"))

    assert _events(second)[-1] == "done"
    state = await StudioStore(session_service, app_name=AGENT_NAME).load(principal, CONVERSATION)
    assert state is not None
    assert state.turn_index == 2
    digest = state.conversation_digest_state()
    assert digest is not None
    assert digest["total_turns"] == 2


async def test_a_double_approve_across_two_instances_gives_one_success_and_one_409(
    tmp_path: Path,
) -> None:
    """A and B both hold the same plan pause and both answer `/interrupt`
    approve. B's own save is held back (`delayed_save`) until after A's has
    already committed, reproducing the actual race: both loaded the pending
    entry before either wrote. B's save then finds the ledger has moved and
    loses cleanly with a 409, rather than resuming its own segment of a
    turn A has already finished."""
    session_service = sqlite_session_service(tmp_path / "shared.db")
    app_a = _app(
        ScriptedResolver([_envelope(choice_points=2)]),
        ScriptedExecutor(),
        generative=_plan_generative(),
        tool_surface=ScriptedSurface(),
        session_service=session_service,
    )
    app_b = _app(
        ScriptedResolver([_envelope(choice_points=2)]),
        ScriptedExecutor(),
        generative=_plan_generative(),
        tool_surface=ScriptedSurface(),
        session_service=session_service,
    )

    b_ready = asyncio.Event()
    a_committed = asyncio.Event()
    real_save = app_b.state.studio.store.save

    async def delayed_save(*args: Any, **kwargs: Any) -> None:
        b_ready.set()
        await a_committed.wait()
        await real_save(*args, **kwargs)

    async with _client(app_a) as client_a, _client(app_b) as client_b:
        first = await _stream(client_a, "/api/studio/turn", _turn_body("q with forks"))
        plan = _frame(first, "plan.proposed")

        app_b.state.studio.store.save = delayed_save
        task_b = asyncio.ensure_future(
            client_b.post(
                "/api/studio/interrupt",
                json=_interrupt_body(plan["interrupt_id"], "approve"),
                headers=AUTH,
            )
        )
        await asyncio.wait_for(b_ready.wait(), timeout=5)

        resumed_a = await _stream(
            client_a, "/api/studio/interrupt", _interrupt_body(plan["interrupt_id"], "approve")
        )
        assert _events(resumed_a)[-1] == "done"

        a_committed.set()
        response_b = await asyncio.wait_for(task_b, timeout=5)

    assert response_b.status_code == 409
    assert response_b.json()["error"] == "transient_unavailable"


async def test_two_instances_racing_the_first_turn_on_a_brand_new_conversation_both_complete(
    tmp_path: Path,
) -> None:
    """A and B both get the first-ever `/turn` for a never-before-seen
    conversation id at nearly the same time. B's own `store.create` is
    delayed until after A's create-and-claim has already committed, so B's
    `create` raises `AlreadyExistsError` -- B must load what A created and
    continue to its own claim, which advances the turn index it just loaded,
    rather than answering a bare 500."""
    session_service = sqlite_session_service(tmp_path / "shared.db")
    app_a = _app(
        ScriptedResolver([_envelope()]), ScriptedExecutor(), session_service=session_service
    )
    app_b = _app(
        ScriptedResolver([_envelope()]), ScriptedExecutor(), session_service=session_service
    )

    a_committed = asyncio.Event()
    real_create = app_b.state.studio.store.create

    async def delayed_create(*args: Any, **kwargs: Any) -> Any:
        await a_committed.wait()
        return await real_create(*args, **kwargs)

    app_b.state.studio.store.create = delayed_create

    async with _client(app_a) as client_a, _client(app_b) as client_b:
        task_b = asyncio.ensure_future(_stream(client_b, "/api/studio/turn", _turn_body("b")))
        await asyncio.sleep(0)  # let B's request reach the delayed create and start waiting

        first_a = await _stream(client_a, "/api/studio/turn", _turn_body("a"))
        assert _events(first_a)[-1] == "done"

        a_committed.set()
        second_b = await asyncio.wait_for(task_b, timeout=5)

    assert _events(second_b)[-1] == "done"
    principal = "u_" + hashlib.sha256(b"test-token").hexdigest()[:16]
    state = await StudioStore(session_service, app_name=AGENT_NAME).load(principal, CONVERSATION)
    assert state is not None
    assert state.turn_index == 2


async def test_a_composition_edit_on_one_instance_survives_a_concurrent_turns_release(
    tmp_path: Path,
) -> None:
    """B's `/composition` pins A's first turn's card while A's SECOND turn
    is still landing its own new one, mid-stream, before A's release. A's
    release finds the ledger has moved (B's pin) and replays A's own
    recorded append onto B's freshly pinned manifest instead of overwriting
    it with A's own stale copy -- B's edit survives."""
    principal = "u_" + hashlib.sha256(b"test-token").hexdigest()[:16]
    session_service = sqlite_session_service(tmp_path / "shared.db")
    app_b = _app(
        ScriptedResolver([_envelope()]), ScriptedExecutor(), session_service=session_service
    )

    first_manifest: dict[str, Any] = {}

    class _PinFirstCardFromB:
        """`narrate`'s scripted stand-in, wrapped: on the SECOND turn only
        (the first turn's own narrate call must not race its own card),
        pins the first turn's card through B before answering."""

        def __init__(self, client_b: httpx2.AsyncClient) -> None:
            self._client_b = client_b
            self.calls = 0

        async def __call__(self, *, question: str, facts: dict[str, Any]) -> dict[str, Any]:
            del question, facts
            self.calls += 1
            if self.calls == 2:
                response = await self._client_b.post(
                    "/api/studio/composition",
                    json={
                        "kind": "composition.mutate",
                        "schema_version": 1,
                        "conversation_id": CONVERSATION,
                        "composition_id": first_manifest["composition_id"],
                        "expected_revision": first_manifest["revision"],
                        "operations": [
                            {
                                "op": "pin",
                                "component_id": first_manifest["blocks"][0]["component_id"],
                            }
                        ],
                    },
                    headers=AUTH,
                )
                assert response.status_code == 200, response.text
            return {"text": f"Answer {self.calls}."}

    async with _client(app_b) as client_b:
        narrate = _PinFirstCardFromB(client_b)
        app_a = _app(
            MessagePathScriptedResolver(),
            _DistinctIdExecutor(),
            generative=GenerativeNodes(
                converse=ScriptedConverse(), compose=ScriptedCompose(), narrate=narrate
            ),
            session_service=session_service,
        )
        async with _client(app_a) as client_a:
            first = await _stream(
                client_a, "/api/studio/turn", _turn_body("Life expectancy in Kenya")
            )
            assert _events(first)[-1] == "done"
            first_manifest.update(_frame(first, "done")["composition"])

            second = await _stream(
                client_a, "/api/studio/turn", _turn_body("compare Guatemala and Nigeria")
            )
            assert _events(second)[-1] == "done"

    reloaded = await StudioStore(session_service, app_name=AGENT_NAME).load(principal, CONVERSATION)
    assert reloaded is not None
    blocks = reloaded.composition.blocks
    assert len(blocks) == 2
    first_block = next(
        b for b in blocks if b.component_id == first_manifest["blocks"][0]["component_id"]
    )
    assert first_block.pinned is True


async def test_ledger_at_turn_n_with_no_stream_running_lets_the_next_turn_succeed(
    tmp_path: Path,
) -> None:
    """A conversation whose ledger holds a completed turn n, with its
    per-turn session `#t{n}` still existing, is idle rather than mid-claim:
    the next `/turn` claims n+1 and succeeds, on a SECOND instance that
    never saw the first."""
    principal = "u_" + hashlib.sha256(b"test-token").hexdigest()[:16]
    session_service = sqlite_session_service(tmp_path / "shared.db")
    app_a = _app(
        ScriptedResolver([_envelope()]), ScriptedExecutor(), session_service=session_service
    )
    app_b = _app(
        ScriptedResolver([_envelope()]), ScriptedExecutor(), session_service=session_service
    )

    async with _client(app_a) as client_a:
        first = await _stream(client_a, "/api/studio/turn", _turn_body("first"))
    assert _events(first)[-1] == "done"

    async with _client(app_b) as client_b:
        second = await _stream(client_b, "/api/studio/turn", _turn_body("second"))
    assert _events(second)[-1] == "done"

    state = await StudioStore(session_service, app_name=AGENT_NAME).load(principal, CONVERSATION)
    assert state is not None
    assert state.turn_index == 2


# --- the segment lease wired into the routes --------------------------------


class _HangingResolver:
    """Never resolves, so a `/turn` stream stays open indefinitely -- the
    gated runner the cross-instance tests below hold A's stream open with,
    instead of racing a request against A's own completion."""

    def __init__(self) -> None:
        self.entered = asyncio.Event()

    async def __call__(
        self, *, question: str, spec_id: str | None = None, wide: bool = False, parts=None
    ) -> dict[str, Any]:
        self.entered.set()
        await asyncio.Event().wait()
        raise AssertionError("unreachable")


async def test_a_turn_from_another_instance_while_a_streams_is_409_with_retry_after(
    tmp_path: Path,
) -> None:
    """B's `/turn` arrives on a separate instance while A's segment is still
    live: the ledger's lease, not A's process-local lock, answers with the
    existing 409, carrying the segment's own remaining TTL as
    `retry_after_seconds`."""
    resolver = _HangingResolver()
    session_service = sqlite_session_service(tmp_path / "shared.db")
    app_a = _app(resolver, ScriptedExecutor(), session_service=session_service)
    app_b = _app(
        ScriptedResolver([_envelope()]), ScriptedExecutor(), session_service=session_service
    )

    async with _client(app_a) as client_a, _client(app_b) as client_b:
        stream_task = asyncio.ensure_future(_stream(client_a, "/api/studio/turn", _turn_body()))
        await asyncio.wait_for(resolver.entered.wait(), timeout=5)

        second = await client_b.post("/api/studio/turn", json=_turn_body("again"), headers=AUTH)

        cancel = await client_a.post(
            "/api/studio/cancel",
            json={"kind": "turn.cancel", "schema_version": 1, "conversation_id": CONVERSATION},
            headers=AUTH,
        )
        assert cancel.status_code == 204
        await asyncio.wait_for(stream_task, timeout=5)

    assert second.status_code == 409
    body = second.json()
    assert body["error"] == "transient_unavailable"
    assert 0 < body["retry_after_seconds"] <= studio_state_module.SEGMENT_TTL_S


async def test_a_stalled_segments_expiry_lets_another_instance_take_over(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A's segment expires (an injected clock, so the test does not wait
    out the real TTL) while A's stream is still open: B's `/turn` claims
    the conversation and completes, and A's own next heartbeat renewal
    finds itself fenced out, ending A's stream with a stream-fatal
    `transient_unavailable` error rather than `done`. A's renewals stall
    until B's turn is done: a heartbeat that renews first would keep A's
    segment alive, which is not the stalled instance this test is about."""
    monkeypatch.setattr(studio_module, "HEARTBEAT_INTERVAL_S", 0.05)
    clock = [1_000.0]
    session_service = sqlite_session_service(tmp_path / "shared.db")
    store_a = StudioStore(session_service, app_name=AGENT_NAME, clock=lambda: clock[0])
    stalled = asyncio.Event()
    renew_a = store_a.renew

    async def stalled_renew(*args: Any) -> Any:
        await stalled.wait()
        return await renew_a(*args)

    monkeypatch.setattr(store_a, "renew", stalled_renew)
    store_b = StudioStore(session_service, app_name=AGENT_NAME, clock=lambda: clock[0])
    resolver = _HangingResolver()
    app_a = _app(resolver, ScriptedExecutor(), session_service=session_service, store=store_a)
    app_b = _app(
        ScriptedResolver([_envelope()]),
        ScriptedExecutor(),
        session_service=session_service,
        store=store_b,
    )

    async with _client(app_a) as client_a, _client(app_b) as client_b:
        stream_task_a = asyncio.ensure_future(_stream(client_a, "/api/studio/turn", _turn_body()))
        await asyncio.wait_for(resolver.entered.wait(), timeout=5)

        clock[0] += studio_state_module.SEGMENT_TTL_S + 1  # A's segment has expired

        second_b = await _stream(client_b, "/api/studio/turn", _turn_body("from b"))
        assert _events(second_b)[-1] == "done"
        stalled.set()

        frames_a = await asyncio.wait_for(stream_task_a, timeout=5)

    assert _events(frames_a)[-1] == "error"
    assert _frame(frames_a, "error")["error"]["error"] == "transient_unavailable"
    assert "done" not in _events(frames_a)


async def test_a_heartbeat_that_cannot_renew_ends_the_stream_once_its_clock_passes_the_deadline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`renew` raises on every call: the stream stays open, retried every
    heartbeat interval, for as long as the store's own injectable clock
    (the same seam every other TTL comparison in this store uses, not the
    wall clock) reads before `segment.expires_at - 10`. Moving that clock
    past the deadline, with no real time elapsed, is what ends the stream
    with a stream-fatal `segment_lost` error."""
    monkeypatch.setattr(studio_module, "HEARTBEAT_INTERVAL_S", 0.05)
    clock = [1_000.0]
    session_service = sqlite_session_service(tmp_path / "shared.db")
    store = StudioStore(session_service, app_name=AGENT_NAME, clock=lambda: clock[0])

    async def _renew_always_fails(*_args: Any, **_kwargs: Any) -> str:
        raise RuntimeError("renewal unavailable")

    monkeypatch.setattr(store, "renew", _renew_always_fails)
    resolver = _HangingResolver()
    app = _app(resolver, ScriptedExecutor(), session_service=session_service, store=store)

    async with _client(app) as client:
        stream_task = asyncio.ensure_future(_stream(client, "/api/studio/turn", _turn_body()))
        await asyncio.wait_for(resolver.entered.wait(), timeout=5)

        principal = "u_" + hashlib.sha256(b"test-token").hexdigest()[:16]
        state = store.current(principal, CONVERSATION)
        assert state is not None and state.segment is not None
        deadline = state.segment.expires_at - 10

        clock[0] = deadline - 1
        await asyncio.sleep(0.2)  # several heartbeat ticks, clock still short of the deadline
        assert not stream_task.done()

        clock[0] = deadline + 1
        frames = await asyncio.wait_for(stream_task, timeout=5)

    assert _events(frames)[-1] == "error"
    assert _frame(frames, "error")["error"]["error"] == "transient_unavailable"
    assert "done" not in _events(frames)


async def test_cancel_from_another_instance_stops_the_live_stream_within_a_heartbeat(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """B, which has never loaded this conversation, POSTs `/cancel`: with no
    local armed event to signal, it falls to `request_cancel`, which finds
    A's live segment and flags it durably. A's next heartbeat renewal (the
    interval shrunk here so the test does not wait out the real one) picks
    the flag up and ends A's stream the same way a same-instance Stop
    does."""
    monkeypatch.setattr(studio_module, "HEARTBEAT_INTERVAL_S", 0.05)
    resolver = _HangingResolver()
    session_service = sqlite_session_service(tmp_path / "shared.db")
    app_a = _app(resolver, ScriptedExecutor(), session_service=session_service)
    app_b = _app(
        ScriptedResolver([_envelope()]), ScriptedExecutor(), session_service=session_service
    )

    async with _client(app_a) as client_a, _client(app_b) as client_b:
        stream_task = asyncio.ensure_future(_stream(client_a, "/api/studio/turn", _turn_body()))
        await asyncio.wait_for(resolver.entered.wait(), timeout=5)

        cancel = await client_b.post(
            "/api/studio/cancel",
            json={"kind": "turn.cancel", "schema_version": 1, "conversation_id": CONVERSATION},
            headers=AUTH,
        )
        assert cancel.status_code == 204

        frames = await asyncio.wait_for(stream_task, timeout=5)

    assert _events(frames)[-1] == "error"
    assert _frame(frames, "error")["error"]["error"] == "cancelled"
    assert "done" not in _events(frames)


async def test_composition_after_a_heartbeat_renewal_still_sees_a_remote_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A's heartbeat renews its live segment -- refreshing `entry.session`
    without moving `entry.state.composition` or its baseline -- before B
    writes the composition directly to the ledger. A's
    own `/composition` call afterwards must still see B's write and build
    on it: `mutate_composition` always fetches the ledger fresh, so A's own
    stale `entry.session` (fresh only because of the unrelated renewal)
    never stands in for what the ledger actually holds."""
    monkeypatch.setattr(studio_module, "HEARTBEAT_INTERVAL_S", 0.05)
    principal = "u_" + hashlib.sha256(b"test-token").hexdigest()[:16]
    session_service = sqlite_session_service(tmp_path / "shared.db")
    resolver = _HangingResolver()
    app_a = _app(resolver, ScriptedExecutor(), session_service=session_service)

    async with _client(app_a) as client_a:
        stream_task = asyncio.ensure_future(_stream(client_a, "/api/studio/turn", _turn_body()))
        await asyncio.wait_for(resolver.entered.wait(), timeout=5)

        state_a = app_a.state.studio.store.current(principal, CONVERSATION)
        segment_id = state_a.segment.id
        first_expires_at = state_a.segment.expires_at

        async def _renewed() -> None:
            while state_a.segment is None or state_a.segment.expires_at == first_expires_at:
                await asyncio.sleep(0.01)

        await asyncio.wait_for(_renewed(), timeout=5)
        assert state_a.segment.id == segment_id  # renewed, not taken over

        store_b = StudioStore(session_service, app_name=AGENT_NAME)
        state_b = await store_b.load(principal, CONVERSATION)
        block = CompositionBlock(
            component_id="cmp_from_b", revision=1, type=ComponentType.LINE_CHART, title="B"
        )
        state_b.composition = studio_state_module._next_manifest(
            state_b.composition, [block], actor=CompositionActor.USER
        )
        await store_b.save(principal, CONVERSATION)

        response = await client_a.post(
            "/api/studio/composition",
            json={
                "kind": "composition.mutate",
                "schema_version": 1,
                "conversation_id": CONVERSATION,
                "operations": [{"op": "pin", "component_id": "cmp_from_b"}],
            },
            headers=AUTH,
        )

        cancel = await client_a.post(
            "/api/studio/cancel",
            json={"kind": "turn.cancel", "schema_version": 1, "conversation_id": CONVERSATION},
            headers=AUTH,
        )
        assert cancel.status_code == 204
        await asyncio.wait_for(stream_task, timeout=5)

    assert response.status_code == 200, response.text
    pinned = response.json()["blocks"][0]
    assert pinned["component_id"] == "cmp_from_b"
    assert pinned["pinned"] is True

    reloaded = await StudioStore(session_service, app_name=AGENT_NAME).load(principal, CONVERSATION)
    assert reloaded.composition.blocks[0].pinned is True


async def test_a_release_failure_does_not_strand_the_local_lock() -> None:
    """`StudioRunner._locked`'s nested finally releases the conversation
    lock even when the store's own release call raises (`StudioStore.release`
    never does in practice; this pins the guarantee independently of that
    contract) -- the exception propagates, but the lock still releases."""
    studio = StudioRunner(
        create_workflow=lambda recorder, surface: create_root_workflow(
            resolver=ScriptedResolver([_envelope()]),
            executor=recorder,
            tool_surface=surface,
            fetch_series_stats=scripted_fetch_series_stats,
        ),
        executor=ScriptedExecutor(),
        session_service=_new_session_service(),
    )
    conversation = await studio.store.create("u_a", CONVERSATION)
    await conversation.lock.acquire()
    segment_id = studio.store.claim_segment("u_a", CONVERSATION)
    await studio.store.save("u_a", CONVERSATION)

    async def broken_release(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("db is down")

    studio.store.release = broken_release  # type: ignore[method-assign]

    async def one_frame() -> Any:
        yield MessageDelta(event="message.delta", seq=0, text="one")

    with pytest.raises(RuntimeError):
        async for _ in studio._locked("u_a", CONVERSATION, conversation, segment_id, one_frame()):
            pass

    assert not conversation.lock.locked()


async def test_a_heartbeat_exception_does_not_strand_the_local_lock() -> None:
    """`StudioRunner._locked`'s nested finally releases the conversation
    lock even when `_unwind(heartbeat)` re-raises a non-cancel exception
    from the heartbeat task (`_heartbeat` catches only `renew`'s own
    exceptions; one from `_end_stream` or `store.now()` reaches `_unwind`
    unhandled) -- the exception propagates, but both the segment and the
    lock still release."""
    studio = StudioRunner(
        create_workflow=lambda recorder, surface: create_root_workflow(
            resolver=ScriptedResolver([_envelope()]),
            executor=recorder,
            tool_surface=surface,
            fetch_series_stats=scripted_fetch_series_stats,
        ),
        executor=ScriptedExecutor(),
        session_service=_new_session_service(),
    )
    conversation = await studio.store.create("u_a", CONVERSATION)
    await conversation.lock.acquire()
    segment_id = studio.store.claim_segment("u_a", CONVERSATION)
    await studio.store.save("u_a", CONVERSATION)

    started = asyncio.Event()

    async def broken_heartbeat(*args: Any, **kwargs: Any) -> None:
        started.set()
        raise RuntimeError("heartbeat exploded")

    studio._heartbeat = broken_heartbeat  # type: ignore[method-assign]

    async def one_frame() -> Any:
        yield MessageDelta(event="message.delta", seq=0, text="one")
        # Let the heartbeat task actually run and raise before `_locked`'s
        # finally reaches `_unwind`: `started.set()` and the raise below it
        # are the same uninterrupted step, so once this wait returns the
        # task is already done with `RuntimeError`, not merely scheduled.
        await started.wait()

    with pytest.raises(RuntimeError, match="heartbeat exploded"):
        async for _ in studio._locked("u_a", CONVERSATION, conversation, segment_id, one_frame()):
            pass

    assert not conversation.lock.locked()
    assert studio.store.current("u_a", CONVERSATION).segment is None


async def test_a_pause_frame_is_not_sent_when_its_checkpoint_fails() -> None:
    """When the save that runs before `plan.proposed` fails, the stream ends
    with a stream-fatal error frame and the client never receives
    `plan.proposed`."""
    resolver = ScriptedResolver([_envelope(choice_points=2)])
    app = _app(
        resolver,
        ScriptedExecutor(),
        generative=GenerativeNodes(planner=ScriptedPlannerModel(), compose=ScriptedCompose()),
        tool_surface=ScriptedSurface(),
    )
    studio = app.state.studio

    async def broken_checkpoint(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("db is down")

    studio.store.checkpoint = broken_checkpoint  # type: ignore[method-assign]

    async with _client(app) as client:
        frames = await _stream(client, "/api/studio/turn", _turn_body("q with forks"))

    assert _events(frames)[-1] == "error"
    assert "plan.proposed" not in _events(frames)
    assert _frame(frames, "error")["error"]["error"] == "transient_unavailable"


async def test_a_remote_composition_write_during_a_stream_reaches_its_done_frame_and_the_ledger(
    tmp_path: Path,
) -> None:
    """A remote instance's `/composition` write, landing mid-stream before
    A's second turn finishes, is merged in by the checkpoint that runs
    before `done`: A's own Done frame carries it, not just a later reload
    from the ledger, pinned on the composition manifest specifically."""
    principal = "u_" + hashlib.sha256(b"test-token").hexdigest()[:16]
    session_service = sqlite_session_service(tmp_path / "shared.db")
    app_b = _app(
        ScriptedResolver([_envelope()]), ScriptedExecutor(), session_service=session_service
    )

    first_manifest: dict[str, Any] = {}

    class _PinFirstCardFromB:
        """`narrate`'s scripted stand-in, wrapped: on the SECOND turn only
        (the first turn's own narrate call must not race its own card),
        pins the first turn's card through B before answering."""

        def __init__(self, client_b: httpx2.AsyncClient) -> None:
            self._client_b = client_b
            self.calls = 0

        async def __call__(self, *, question: str, facts: dict[str, Any]) -> dict[str, Any]:
            del question, facts
            self.calls += 1
            if self.calls == 2:
                response = await self._client_b.post(
                    "/api/studio/composition",
                    json={
                        "kind": "composition.mutate",
                        "schema_version": 1,
                        "conversation_id": CONVERSATION,
                        "composition_id": first_manifest["composition_id"],
                        "expected_revision": first_manifest["revision"],
                        "operations": [
                            {
                                "op": "pin",
                                "component_id": first_manifest["blocks"][0]["component_id"],
                            }
                        ],
                    },
                    headers=AUTH,
                )
                assert response.status_code == 200, response.text
            return {"text": f"Answer {self.calls}."}

    async with _client(app_b) as client_b:
        narrate = _PinFirstCardFromB(client_b)
        app_a = _app(
            MessagePathScriptedResolver(),
            _DistinctIdExecutor(),
            generative=GenerativeNodes(
                converse=ScriptedConverse(), compose=ScriptedCompose(), narrate=narrate
            ),
            session_service=session_service,
        )
        async with _client(app_a) as client_a:
            first = await _stream(
                client_a, "/api/studio/turn", _turn_body("Life expectancy in Kenya")
            )
            assert _events(first)[-1] == "done"
            first_manifest.update(_frame(first, "done")["composition"])

            second = await _stream(
                client_a, "/api/studio/turn", _turn_body("compare Guatemala and Nigeria")
            )
            assert _events(second)[-1] == "done"

    pinned_component_id = first_manifest["blocks"][0]["component_id"]
    done_block = next(
        b
        for b in _frame(second, "done")["composition"]["blocks"]
        if b["component_id"] == pinned_component_id
    )
    assert done_block["pinned"] is True

    reloaded = await StudioStore(session_service, app_name=AGENT_NAME).load(principal, CONVERSATION)
    assert reloaded is not None
    assert reloaded.composition is not None
    ledger_block = next(
        b for b in reloaded.composition.blocks if b.component_id == pinned_component_id
    )
    assert ledger_block.pinned is True


async def test_a_card_landed_during_compositions_own_fetch_survives_live_and_the_ledger() -> None:
    """A card the local stream lands (`land_component`, still unsaved)
    exactly while `/composition`'s own `mutate_composition` fetch is in
    flight is folded into the manifest `mutate` is handed -- the pin this
    request applies to an EARLIER card and the just-landed card both reach
    the response, the live conversation, and the ledger."""
    resolver = ScriptedResolver([_envelope(choice_points=2)])
    executor = ScriptedExecutor(
        [
            {"outcome": "ok", "frame_refs": [REF_A]},
            {"outcome": "ok", "frame_refs": [REF_A], "component": COMPONENT},
        ]
    )
    app = _app(resolver, executor, generative=_plan_generative(), tool_surface=ScriptedSurface())
    session_service = app.state.studio.store._session_service
    principal = "u_" + hashlib.sha256(b"test-token").hexdigest()[:16]

    async with _client(app) as client:
        await _conversation_with_component(client)
        store = app.state.studio.store
        state = store.current(principal, CONVERSATION)

        turn_lock = store.lock_for(principal, CONVERSATION)
        await turn_lock.acquire()  # stands in for a stream still owning the turn

        real_get_session = session_service.get_session
        landed: list[bool] = []

        async def get_session_then_land(*args: Any, **kwargs: Any) -> Any:
            if not landed:
                landed.append(True)
                action = land_component(
                    state,
                    {"component_id": "cmp_landed", "type": "line_chart", "revision": 1},
                    revises=None,
                )
                assert action == "created"
            return await real_get_session(*args, **kwargs)

        session_service.get_session = get_session_then_land
        try:
            response = await client.post(
                "/api/studio/composition",
                json={
                    "kind": "composition.mutate",
                    "schema_version": 1,
                    "conversation_id": CONVERSATION,
                    "operations": [{"op": "pin", "component_id": "cmp_01test0001"}],
                },
                headers=AUTH,
            )
        finally:
            session_service.get_session = real_get_session
            turn_lock.release()

    assert response.status_code == 200, response.text
    updated = response.json()
    assert {b["component_id"] for b in updated["blocks"]} == {"cmp_01test0001", "cmp_landed"}
    pinned = next(b for b in updated["blocks"] if b["component_id"] == "cmp_01test0001")
    assert pinned["pinned"] is True

    live = store.current(principal, CONVERSATION)
    assert {b.component_id for b in live.composition.blocks} == {"cmp_01test0001", "cmp_landed"}

    reloaded = await StudioStore(session_service, app_name=AGENT_NAME).load(principal, CONVERSATION)
    assert {b.component_id for b in reloaded.composition.blocks} == {"cmp_01test0001", "cmp_landed"}


def test_digest_user_text_is_truncated_and_oldest_dropped_under_byte_cap() -> None:
    state = ConversationState(conversation_id=CONVERSATION)
    for index in range(10):
        state.turns.append({"user": f"q{index} " + "u" * 900, "assistant": "a" * 900})

    digest = state.conversation_digest_state()

    assert digest is not None
    assert all(len(turn["user"]) <= 280 for turn in digest["turns"])
    assert len(json.dumps(digest["turns"])) <= 5_000
    # Truncated turns are ~590 bytes each, so the 5,000-byte cap keeps the
    # newest eight and drops q0 and q1 -- oldest first, newest kept.
    assert [turn["user"].split()[0] for turn in digest["turns"]] == [f"q{i}" for i in range(2, 10)]
    assert digest["total_turns"] == 10


# --- seeded turns: Search results handed to Studio -----------------------------


@pytest.fixture
def definition_store() -> Any:
    """The definition store the process holds, fresh per test: data-mcp's, when it runs in the
    harness's process (`DataMcpDefinitions()`)."""
    store = InMemoryDefinitionStore()
    set_definition_store(store)
    yield store
    set_definition_store(None)


_SEED_IDENTITY = {
    "identity_version": "spec1",
    "seed": "education aid",
    "requested_window": {"kind": "all"},
}
_SEED_SPEC = spec_id_for(canonicalize(_SEED_IDENTITY))


def _seed_spec(**resolved: Any) -> dict[str, Any]:
    """A Spec block whose identity hashes to its spec_id."""
    return {
        "spec_id": _SEED_SPEC,
        "definition": {"schema_version": 1, "spec_id": _SEED_SPEC, "identity": _SEED_IDENTITY},
        "resolved": {"rendered_sentence": "Education aid from every donor", **resolved},
    }


def _seeded_turn_body(
    message: str, *, operations: list[dict[str, Any]] | None = None, **resolved: Any
) -> dict[str, Any]:
    if operations is None:
        operations = [{"verb": "rank", "phrase": "top 5 donors", "over": "origin"}]
    seed = {"spec": _seed_spec(**resolved), "operations": operations}
    return {**_turn_body(message), "seed_spec": [seed]}


async def test_a_seeded_turn_stores_its_seed_and_plans_without_searching_or_minting(
    definition_store: InMemoryDefinitionStore,
) -> None:
    resolver = ScriptedResolver([_envelope()])
    executor = ScriptedExecutor(strict=True)
    app = _app(resolver, executor, served=[_seed_spec()])
    async with _client(app) as client:
        frames = await _stream(
            client, "/api/studio/turn", _seeded_turn_body("top 5 donors for education aid")
        )

    assert resolver.calls == []
    assert executor.calls == []
    assert await definition_store.get(_SEED_SPEC, schema_version=1) is not None
    steps = [frame for frame in frames if frame["event"] == "activity.step"]
    assert (steps[0]["tool"], steps[0]["status"], steps[0]["label"]) == (
        "receive_seeds",
        "complete",
        "Receive the datasets from Search",
    )
    assert "resolve" not in {step["step_id"] for step in steps}
    assert _events(frames)[-1] == "plan.proposed"


async def test_a_seeded_turn_whose_definition_store_is_unreachable_searches_the_question() -> None:
    class _Unreachable:
        async def put_many(self, definitions: Any) -> None:
            raise DefinitionStoreUnavailable("backend unreachable")

        async def get(self, spec_id: str, *, schema_version: int) -> None:
            raise DefinitionStoreUnavailable("backend unreachable")

        async def aclose(self) -> None:
            pass

    set_definition_store(_Unreachable())
    try:
        resolver = ScriptedResolver([_envelope()])
        async with _client(_app(resolver, ScriptedExecutor(), served=[_seed_spec()])) as client:
            frames = await _stream(
                client, "/api/studio/turn", _seeded_turn_body("top 5 donors for education aid")
            )
    finally:
        set_definition_store(None)

    steps = [frame for frame in frames if frame["event"] == "activity.step"]
    failed = [step for step in steps if step["status"] == "failed"]
    assert [step["label"] for step in failed] == ["Receive the datasets from Search"] * 2
    assert failed[0]["detail"] == (
        "service_unavailable: data-mcp could not resolve seed 0: the definition store is "
        "unreachable right now. Retry after 5.0s."
    )
    assert failed[1]["detail"].startswith(
        "service_unavailable: data-mcp could not resolve the datasets"
    )
    assert len(resolver.calls) == 1
    assert resolver.calls[0]["question"] == "top 5 donors for education aid"
    assert "resolve" in {step["step_id"] for step in steps}
    assert _events(frames)[-1] == "done"


async def test_a_no_year_seed_with_no_operation_fetches_the_year_search_showed(
    definition_store: InMemoryDefinitionStore,
) -> None:
    del definition_store
    headline = {"year": 2024, "year_named": False, "observations": 13356, "series": 13356}
    resolver = ScriptedResolver([_envelope()])
    executor = ScriptedExecutor()
    async with _client(_app(resolver, executor, served=[_seed_spec(headline=headline)])) as client:
        frames = await _stream(
            client,
            "/api/studio/turn",
            _seeded_turn_body("education aid", operations=[], headline=headline),
        )

    assert "elicitation.request" not in _events(frames)
    assert _events(frames)[-1] == "done"
    assert resolver.calls == []
    assert executor.calls[0] == {
        "tool": "get_data",
        "args": {
            "spec_id": _SEED_SPEC,
            "scope": {"window": {"kind": "range", "start_year": 2024, "end_year": 2024}},
        },
    }


async def test_a_single_reference_seed_spec_runs_the_question_and_shows_it_was_not_used() -> None:
    """The shape `seed_spec` was reserved with is accepted, as it was: the turn
    answers the question, and a failed step says the reference seeded nothing."""
    resolver = ScriptedResolver([_envelope()])
    executor = ScriptedExecutor()
    body = {
        **_turn_body("life expectancy in Kenya"),
        "seed_spec": {"spec_id": _SEED_SPEC, "content_version": "cv_" + "0" * 32},
    }
    async with _client(_app(resolver, executor)) as client:
        frames = await _stream(client, "/api/studio/turn", body)

    assert _events(frames)[-1] == "done"
    assert len(resolver.calls) == 1
    (failed,) = [
        frame
        for frame in frames
        if frame["event"] == "activity.step" and frame["status"] == "failed"
    ]
    assert failed["label"] == "Receive the datasets from Search"
    assert failed["detail"].startswith("seed_reference_unused: seed_spec holds one")


async def test_a_seed_past_the_size_cap_fails_its_step_and_the_turn_searches_the_question(
    definition_store: InMemoryDefinitionStore,
) -> None:
    resolver = ScriptedResolver([_envelope()])
    executor = ScriptedExecutor()
    body = _seeded_turn_body("top 5 donors for education aid")
    body["seed_spec"][0]["spec"]["trace"] = {"padding": "x" * MAX_SEED_SPEC_BYTES}
    async with _client(_app(resolver, executor)) as client:
        frames = await _stream(client, "/api/studio/turn", body)

    steps = [frame for frame in frames if frame["event"] == "activity.step"]
    (failed,) = [step for step in steps if step["status"] == "failed"]
    assert failed["label"] == "Receive the datasets from Search"
    assert failed["detail"].startswith("seed_invalid: seed 0 cannot be used")
    assert "-byte cap on one seed" in failed["detail"]
    assert await definition_store.get(_SEED_SPEC, schema_version=1) is None
    assert len(resolver.calls) == 1
    assert _events(frames)[-1] == "done"


# The Spec block the dev Search API served for "top 5 donors of education aid
# to Kenya": a CRS flow with Kenya bound under `destination`.
_KENYA_SPEC = json.loads(
    (Path(__file__).parent / "fixtures" / "seed_education_aid_kenya.json").read_text()
)["spec"]


def _member_paths(value: Any, path: str = "") -> set[str]:
    """The path of every leaf member under `value`, a list's items as `[]`.

    An identity counts as one member: data-mcp stores it whole.
    """
    if isinstance(value, dict) and not path.endswith("identity"):
        return {
            leaf
            for key, member in value.items()
            for leaf in _member_paths(member, f"{path}.{key}" if path else key)
        }
    if isinstance(value, list):
        return {leaf for item in value for leaf in _member_paths(item, f"{path}[]")}
    return {path}


async def test_a_seeded_turn_s_session_keeps_only_what_the_seed_node_reads_of_each_seed(
    definition_store: InMemoryDefinitionStore,
) -> None:
    """The state a seeded turn's session is created with holds each seed's
    spec_id, definition, StatVar dcids and operations, and for a seed the
    harness cannot take, the refusal its step fails with at its own index."""
    del definition_store
    session_service = _new_session_service()
    created: list[dict[str, Any]] = []
    create_session = session_service.create_session

    async def recording_create_session(**kwargs: Any) -> Any:
        if SEEDS_STATE_KEY in (kwargs.get("state") or {}):
            created.append(kwargs["state"])
        return await create_session(**kwargs)

    session_service.create_session = recording_create_session  # type: ignore[method-assign]
    oversized = _seed_spec()
    oversized["trace"] = {"padding": "x" * MAX_SEED_SPEC_BYTES}
    body = {
        **_turn_body("top 5 donors of education aid to Kenya"),
        "seed_spec": [
            {
                "spec": _KENYA_SPEC,
                "operations": [{"verb": "rank", "phrase": "top 5 donors", "over": "origin"}],
            },
            {"spec": oversized, "operations": []},
        ],
    }
    app = _app(
        ScriptedResolver([_envelope()]),
        ScriptedExecutor(),
        session_service=session_service,
        served=[_KENYA_SPEC],
    )
    async with _client(app) as client:
        frames = await _stream(client, "/api/studio/turn", body)

    (state,) = created
    kept, refused = json.loads(state[SEEDS_STATE_KEY])
    assert _member_paths(kept) == {
        "spec.spec_id",
        "spec.definition.schema_version",
        "spec.definition.spec_id",
        "spec.definition.identity",
        "spec.resolved.stat_vars.items[].ref",
        "operations[].verb",
        "operations[].phrase",
        "operations[].over",
        "operations[].target_year",
    }
    assert kept["spec"]["definition"] == _KENYA_SPEC["definition"]
    assert [item["ref"] for item in kept["spec"]["resolved"]["stat_vars"]["items"]] == [
        item["ref"] for item in _KENYA_SPEC["resolved"]["stat_vars"]["items"]
    ]
    assert set(refused) == {"refused"}
    assert refused["refused"]["code"] == "seed_invalid"
    assert refused["refused"]["detail"].startswith("seed 1 cannot be used")
    (failed,) = [
        frame
        for frame in frames
        if frame["event"] == "activity.step" and frame["status"] == "failed"
    ]
    assert failed["detail"] == f"seed_invalid: {refused['refused']['detail']}"
    assert _events(frames)[-1] == "plan.proposed"


def _malformed_seed(
    *, identity: Mapping[str, Any] | None = None, spec_id: str | None = None, **resolved: Any
) -> dict[str, Any]:
    """A seed that parses as `harness.wire.Seed`, whose `identity` (by default
    `_SEED_IDENTITY` for another reading) or `resolved` holds a member of a
    type the seed's readers cannot take. Its identity hashes to its spec_id
    unless `spec_id` is given."""
    identity = dict(identity or {**_SEED_IDENTITY, "seed": "malformed"})
    spec_id = spec_id or spec_id_for(canonicalize(identity))
    return {
        "spec": {
            "spec_id": spec_id,
            "definition": {"schema_version": 1, "spec_id": spec_id, "identity": identity},
            "resolved": {"rendered_sentence": "A malformed seed", **resolved},
        },
        "operations": [],
    }


@pytest.mark.parametrize(
    ("malformed", "refused"),
    [
        (
            _malformed_seed(identity={**_SEED_IDENTITY, "requested_window": "x"}),
            "seed_dropped: seed 0 is left out: Search no longer resolves it (request_invalid)",
        ),
        (
            _malformed_seed(
                identity={**_SEED_IDENTITY, "count": 2**70}, spec_id="spec_" + "0" * 32
            ),
            "seed_invalid: seed 0 cannot be used",
        ),
        (
            _malformed_seed(identity={**_SEED_IDENTITY, "entities": ["country/KEN"]}),
            "seed_dropped: seed 0 is left out: Search no longer resolves it (request_invalid)",
        ),
        (
            _malformed_seed(identity={**_SEED_IDENTITY, "shape": "statvar_schema"}),
            "seed_dropped: seed 0 is left out: Search no longer resolves it (request_invalid)",
        ),
        (_malformed_seed(stat_vars=["Count_Person"]), "seed_invalid: seed 0 cannot be used"),
        (
            _malformed_seed(stat_vars={"items": ["Count_Person"]}),
            "seed_invalid: seed 0 cannot be used",
        ),
        (
            _malformed_seed(stat_vars={"items": [{"ref": 7, "slot_values": []}]}),
            "seed_invalid: seed 0 cannot be used",
        ),
    ],
    ids=[
        "requested-window-a-string",
        "identity-integer-past-2-53",
        "identity-entities-a-list",
        "identity-shape-a-string",
        "resolved-stat-vars-a-list",
        "resolved-stat-var-items-strings",
        "resolved-stat-var-ref-a-number",
    ],
)
async def test_a_seed_the_harness_or_search_cannot_take_fails_its_step_and_the_other_seed_plans(
    definition_store: InMemoryDefinitionStore, malformed: dict[str, Any], refused: str
) -> None:
    """A seed the harness cannot read is refused as it is taken; one whose
    identity Search refuses is dropped with Search's reason."""
    resolver = ScriptedResolver([_envelope()])
    executor = ScriptedExecutor()
    body = _seeded_turn_body("top 5 donors for education aid")
    body["seed_spec"].insert(0, malformed)
    async with _client(_app(resolver, executor, served=[_seed_spec()])) as client:
        frames = await _stream(client, "/api/studio/turn", body)

    assert "error" not in _events(frames)
    steps = [frame for frame in frames if frame["event"] == "activity.step"]
    (failed,) = [step for step in steps if step["status"] == "failed"]
    assert failed["label"] == "Receive the datasets from Search"
    assert failed["detail"].startswith(refused)
    malformed_spec_id = malformed["spec"]["spec_id"]
    assert await definition_store.get(malformed_spec_id, schema_version=1) is None
    assert await definition_store.get(_SEED_SPEC, schema_version=1) is not None
    assert resolver.calls == []
    assert _events(frames)[-1] == "plan.proposed"


async def test_a_request_past_the_body_cap_is_refused_before_it_is_parsed() -> None:
    resolver = ScriptedResolver([_envelope()])
    executor = ScriptedExecutor()
    body = _seeded_turn_body("top 5 donors for education aid")
    body["seed_spec"][0]["spec"]["trace"] = {"padding": "x" * MAX_TURN_REQUEST_BYTES}
    async with _client(_app(resolver, executor)) as client:
        response = await client.post("/api/studio/turn", json=body, headers=AUTH)

    assert response.status_code == 413
    assert response.json()["error"] == "resource_exhausted"
    assert "6,356,992-byte cap" in response.json()["detail"]
    assert executor.calls == []
    assert resolver.calls == []
