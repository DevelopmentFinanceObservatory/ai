"""The studio routes end-to-end over real HTTP, with scripted fakes.

Everything runs against a FastAPI app through httpx2's ASGI transport, with
real routing, real SSE streaming and the real turn engine's run and resume,
on the turn graph `tests.studio_routes.orchestrator_app` wires as
`harness.server` does. The agent's model answers from a script, and search,
the executor, the planner and the plan's tool surface are scripted (no
model, no MCP server). Every frame read off the wire is also validated
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
import copy
import hashlib
import itertools
import json
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import asynccontextmanager, suppress
from dataclasses import replace
from datetime import UTC, datetime, timedelta, tzinfo
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import harness.graph.approve as approve_module
import harness.graph.execute as execute_module
import harness.graph.switch as switch_module
import harness.studio as studio_module
import harness.studio_state as studio_state_module
import harness.studio_store as studio_store_module
import harness.studio_stream as studio_stream_module
import harness.turn.loop as turn_loop_module
import harness.turn.store as turn_store_module
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
    InMemoryDefinitionStore,
    canonicalize,
    spec_id_for,
)
from data_mcp.definition_store import set_store as set_definition_store
from data_mcp.server import mcp as data_mcp
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from google.genai import types
from harness.conversation import current_conversation_id
from harness.early_mint import EarlyMints
from harness.graph.state import (
    ORCHESTRATOR_PLAN_RECORDS_KEY,
    ORCHESTRATOR_PRIOR_LEDGER_KEY,
    ORCHESTRATOR_SEARCHES_KEY,
    ORCHESTRATOR_SPECS_KEY,
    frame_id_for,
)
from harness.mcp_client import McpExecutor
from harness.mcp_deadlines import McpTimeoutPolicy
from harness.plan_display import decision_option_id
from harness.planner_adapter import PlannerAdapter
from harness.provenance import CARD_FRAME_KEY, ProvenanceStore
from harness.scripted import SCRIPTED_SPEC, ScriptedPlannerModel, ScriptedSurface
from harness.sentences import FAILURE_CLASS, StopCode, refusal_sentence
from harness.settings import Settings
from harness.studio import (
    _acquire_or_conflict,
    create_studio_router,
)
from harness.studio_edit import _canvas_is_behind
from harness.studio_state import (
    ConversationState,
    DatabaseUnavailable,
    SegmentLost,
    _PendingChoice,
    _PendingOption,
)
from harness.studio_store import (
    SEQ_BLOCK,
    ConversationStale,
    Renewal,
    SegmentClose,
    StudioStore,
    _deserialize_pending,
)
from harness.studio_stream import Finish, LockHold, WorkEnd, ending
from harness.studio_translate import (
    STREAM_RETRY_AFTER_S,
    _RecordingExecutor,
    _stream_fatal,
    keep_turn_records,
)
from harness.tool_contracts import HarnessCode, ResolverError, harness_refusal
from harness.turn.store import CursorStale, TurnStatus
from harness.turn_events import PLANNER_ASK
from harness.turn_records import SearchReading
from harness.wire import (
    MAX_SEED_SPEC_BYTES,
    MAX_TURN_REQUEST_BYTES,
    ComponentType,
    CompositionActor,
    CompositionBlock,
    InterruptResponse,
    PlanDecision,
    PlanDecisionOption,
    TurnRequest,
)
from mcp import MCPError
from mcp.types import REQUEST_TIMEOUT
from opentelemetry import trace as otel_trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from planner.operations import GrowthOperation, RetrieveOperation, Translation
from planner.plan import (
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
from sqlalchemy import text
from sqlalchemy.exc import OperationalError
from sqlalchemy.ext.asyncio import AsyncEngine
from starlette.requests import ClientDisconnect

from .data_mcp_definitions import DataMcpDefinitions, SearchStub
from .failure_text import classed_sentence, is_user_sentence
from .graph_turns import agent_read
from .resolved_revision import resolved_revision
from .scope_scenarios import _hashed_scope, _switched_answer
from .scripted_genai import ScriptedGenai, call_part, reply, text_part
from .scripted_tools import (
    CannedSearch,
    chart,
    definite_envelope,
    fetch_call,
    make_charts_call,
    plan_call,
    search_call,
)
from .store_clock import StoreClock
from .studio_routes import AUTH, ScriptLlm, orchestrator_app, stream, studio_client, turn_body
from .test_turn_records import _top_donors
from .turn_driver import CONVERSATION

SPEC_A = "spec_" + "a" * 32
REF_A = {"spec_id": SPEC_A, "content_version": "cv_" + "1" * 32}
# The frame a `fetch` of SPEC_A registers when `get_data` answers with REF_A.
FRAME_A = frame_id_for(REF_A)
# The frame `ScriptedSurface`'s plan hands back to the agent once it has run.
PLAN_FRAME = frame_id_for({"spec_id": "ops_" + "9" * 32, "content_version": "ocv_" + "9" * 32})
# The PRINCIPAL `AUTH`'s bearer token resolves to.
PRINCIPAL = "u_" + hashlib.sha256(b"test-token").hexdigest()[:16]
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

_VIEW = {
    "outcome": "ok",
    "frame_refs": [],
    "view": {"dimensions": [{"name": "stat_var", "values": ["dcid:Aid"], "plural": False}]},
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


class ScriptedExecutor:
    """Answers each call with the next of `results`, then by tool: `get_data`
    with REF_A, `describe_frame` with `_VIEW`, a create with a copy of
    COMPONENT, anything else with REF_A. `strict=True` fails the test on a
    call past `results`, so a route making calls the script never
    anticipated cannot pass unnoticed."""

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
        if tool == "describe_frame":
            return _VIEW
        if tool == "write_component":
            return {"outcome": "ok", "frame_refs": [], "component": copy.deepcopy(COMPONENT)}
        return {"outcome": "ok", "frame_refs": [REF_A], "summary": "fetched"}


def _card_results(component: Mapping[str, Any] = COMPONENT, **write: Any) -> list[dict[str, Any]]:
    """The executor's answers to one `_card` turn: the fetch, the chart's
    frame description, and the write landing `component` (with `write`'s
    extra keys, e.g. a `write_token`)."""
    return [
        {"outcome": "ok", "frame_refs": [REF_A], "summary": "fetched"},
        _VIEW,
        {"outcome": "ok", "frame_refs": [], "component": copy.deepcopy(dict(component)), **write},
    ]


def _text(text: str) -> types.Part:
    return types.Part.from_text(text=text)


def _card(
    *, title: str = "Aid to Kenya", type_: str = "stat_card", charts: int = 1
) -> tuple[types.Part, ...]:
    """A turn that searches, fetches SPEC_A and draws `charts` cards on its
    frame. Its search needs a `_search()` envelope."""
    drawn = [
        chart(FRAME_A, type_=type_, title=title if index == 0 else f"{title} ({index + 1})")
        for index in range(charts)
    ]
    return (
        search_call("aid to Kenya"),
        fetch_call(SPEC_A),
        make_charts_call(*drawn),
        _text("Here is the card."),
    )


def _plan(brief: str = "q with forks", *after: types.Part) -> tuple[types.Part, ...]:
    """A turn that hands `brief` to the planner and pauses on its plan;
    `after` is what the agent does once an approved plan is handed back."""
    return (plan_call(brief), *after)


def _charts_the_plan() -> tuple[types.Part, ...]:
    """What the agent does with an approved plan handed back: one card on
    its frame, then its answer."""
    return (
        make_charts_call(chart(PLAN_FRAME, type_="stat_card", title="Aid to Kenya")),
        _text("Charted."),
    )


def _ask(*options: str) -> tuple[types.Part, ...]:
    """A turn that asks the user which reading they meant."""
    return _ask_questions(("Which reading?", options or ("A", "B")))


def _ask_questions(*questions: tuple[str, tuple[str, ...]]) -> tuple[types.Part, ...]:
    """A turn that asks the user each question, offering its options."""
    return (
        call_part(
            "ask_user",
            {
                "questions": [
                    {"question": question, "options": [{"label": option} for option in options]}
                    for question, options in questions
                ]
            },
        ),
    )


def _pick(option_id: str) -> dict[str, Any]:
    return {"kind": "option", "option_id": option_id}


def _search(turns: int = 1) -> CannedSearch:
    """Search answering `turns` searches with one definite part reading SPEC_A."""
    return CannedSearch(*[definite_envelope(spec_id=SPEC_A) for _ in range(turns)])


class _HangingResolver:
    """Never answers, so a `/turn` stream that searches stays open
    indefinitely. The cancel and cross-instance tests hold A's stream open
    with it, so no request races A's own completion."""

    def __init__(self) -> None:
        self.entered = asyncio.Event()

    async def __call__(self, **kwargs: Any) -> dict[str, Any]:
        self.entered.set()
        await asyncio.Event().wait()
        raise AssertionError("unreachable")


def _engine() -> AsyncEngine:
    """The engine every store in the running test is built on, so two stores
    stand for two instances sharing one database."""
    assert _Database.engine is not None
    return _Database.engine


class _Database:
    """The running test's Postgres engine (`database`)."""

    engine: AsyncEngine | None = None


@pytest.fixture(autouse=True)
def database(monkeypatch: pytest.MonkeyPatch, pg_engine: AsyncEngine) -> AsyncEngine:
    """Every studio-door test in this file stores its conversations and
    turns on the test's own Postgres database."""
    monkeypatch.setattr(_Database, "engine", pg_engine)
    return pg_engine


def _app(
    *script: types.Part,
    resolver: Any = None,
    executor: Any = None,
    planner: Any = None,
    tool_surface: ToolSurface | None = None,
    settings: Settings | None = None,
    edit_model: Any | None = None,
    engine: AsyncEngine | None = None,
    store: StudioStore | None = None,
    served: Sequence[Mapping[str, Any]] = (),
    definition_surface: Any = None,
    early_mints: EarlyMints | None = None,
    heartbeat_seconds: float | None = None,
) -> FastAPI:
    """`orchestrator_app` on this test's database, its agent's model
    answering from `script` and kept as `app.state.model`. The executor
    defaults to this module's `ScriptedExecutor`; the planner to
    `ScriptedPlannerModel`. A cross-instance test passes a second `store`
    on the same engine; a segment TTL scenario passes its own `store` with
    its own clock. A module outside this one, where `database` does not
    run, passes the `engine` of its own `pg_engine`."""
    app, model = orchestrator_app(
        *script,
        resolver=resolver,
        executor=executor or ScriptedExecutor(),
        planner=planner,
        tool_surface=tool_surface,
        engine=engine or _engine(),
        store=store,
        settings=settings,
        edit_model=edit_model,
        served=served,
        definition_surface=definition_surface,
        early_mints=early_mints,
        heartbeat_seconds=heartbeat_seconds,
    )
    app.state.model = model
    return app


def _model(app: FastAPI) -> ScriptLlm:
    return app.state.model


def _client(app: FastAPI) -> httpx2.AsyncClient:
    return studio_client(app)


def _turn_body(message: str = "kenya oda 2023") -> dict[str, Any]:
    return turn_body(message)


def _interrupt_body(
    interrupt_id: str,
    decision: str,
    payload: Any = None,
    *,
    answers: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "kind": "interrupt.response",
        "schema_version": 1,
        "conversation_id": CONVERSATION,
        "interrupt_id": interrupt_id,
        "decision": decision,
    }
    if payload is not None:
        body["payload"] = payload
    if answers is not None:
        body["answers"] = answers
    return body


async def _stream(
    client: httpx2.AsyncClient, path: str, body: dict[str, Any]
) -> list[dict[str, Any]]:
    """POST and read the SSE stream to its end; every frame must be
    contract-valid."""
    return await stream(client, path, body)


def _events(frames: list[dict[str, Any]]) -> list[str]:
    return [frame["event"] for frame in frames]


def _frame(frames: list[dict[str, Any]], event: str) -> dict[str, Any]:
    matches = [frame for frame in frames if frame["event"] == event]
    assert matches, f"no {event} frame in {_events(frames)}"
    return matches[-1]


async def _conversation_with_component(client: httpx2.AsyncClient) -> dict[str, Any]:
    """Run one `_card` turn; return the final manifest."""
    frames = await _stream(client, "/api/studio/turn", _turn_body())
    return _frame(frames, "done")["composition"]


# --- auth ---------------------------------------------------------------------


async def test_a_request_without_a_bearer_token_is_401() -> None:
    async with _client(_app()) as client:
        response = await client.post("/api/studio/turn", json=_turn_body())
    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"


async def test_a_wrong_shared_secret_is_401() -> None:
    app = _app(settings=Settings(studio_api_token="right"))
    async with _client(app) as client:
        response = await client.post(
            "/api/studio/turn",
            json=_turn_body(),
            headers={"Authorization": "Bearer wrong"},
        )
    assert response.status_code == 401


# --- a search, fetch and chart turn ----------------------------------------------


async def test_a_definite_turn_streams_steps_text_and_done() -> None:
    executor = ScriptedExecutor(_card_results()[:2], strict=True)
    app = _app(
        search_call("aid to Kenya"),
        fetch_call(SPEC_A),
        _text("Kenya received aid."),
        resolver=_search(),
        executor=executor,
    )
    async with _client(app) as client:
        frames = await _stream(client, "/api/studio/turn", _turn_body())

    # One fetch, which describes its frame, and nothing else asked of the executor.
    assert [call["tool"] for call in executor.calls] == ["get_data", "describe_frame"]
    events = _events(frames)
    assert events[-1] == "done"
    assert "suggestions" in events
    fetch_step = [f for f in frames if f["event"] == "activity.step" and f.get("tool") == "fetch"]
    assert fetch_step[-1]["status"] == "complete"
    assert _frame(frames, "message.complete")["text"] == "Kenya received aid."
    done = _frame(frames, "done")
    assert done["trace_id"].startswith("tr_")
    assert "composition" not in done  # nothing moved; null is elided
    seqs = [frame["seq"] for frame in frames]
    assert seqs == sorted(seqs) and len(set(seqs)) == len(seqs)


async def test_a_turn_past_its_call_cap_ends_on_the_cap_sentence_with_no_further_model_call() -> (
    None
):
    """The route makes the request's budget current, and a model call past
    its cap answers the cap's sentence (`harness.turn_budget.bounded_by`)
    without asking the model: the sentence is the turn's final message."""
    app = _app(
        fetch_call(SPEC_A),
        _text("never sent"),
        executor=ScriptedExecutor(_card_results()[:2], strict=True),
        settings=Settings(turn_model_calls_max=1),
    )
    async with _client(app) as client:
        frames = await _stream(client, "/api/studio/turn", _turn_body())

    assert len(_model(app).requests) == 1
    assert _frame(frames, "message.complete")["text"] == refusal_sentence(
        {"code": StopCode.TURN_MODEL_CALLS_SPENT}
    )
    assert _events(frames)[-1] == "done"


async def test_a_chart_call_writes_each_chart_and_a_refused_write_fails_its_line() -> None:
    """Each chart of one `make_charts` call is its own write and its own
    Activity line: a refused write fails its line while the turn completes,
    and the card that landed names the line it came from."""
    executor = ScriptedExecutor(
        [
            *_card_results(),
            {
                "outcome": "refusal",
                "refusal": {
                    "code": "schema_invalid",
                    "retry": "do_not_retry",
                    "message": "line_chart needs a period dimension",
                },
            },
        ]
    )
    app = _app(*_card(charts=2), resolver=_search(), executor=executor)
    async with _client(app) as client:
        frames = await _stream(client, "/api/studio/turn", _turn_body())

    writes = [c for c in executor.calls if c["tool"] == "write_component"]
    assert [w["args"]["intent"]["title"] for w in writes] == ["Aid to Kenya", "Aid to Kenya (2)"]
    assert all(w["args"]["intent"]["frame_ref"] == REF_A for w in writes)
    lines = [
        f for f in frames if f["event"] == "activity.step" and f.get("tool") == "write_component"
    ]
    assert [line["status"] for line in lines] == ["complete", "failed"]
    assert lines[0]["produced"] == {"component_id": "cmp_01test0001", "revision": 1}
    assert lines[1]["detail"] == classed_sentence("schema_invalid")
    assert is_user_sentence(lines[1]["detail"])
    # The call's own line names the card it could not draw, after the reason.
    (call,) = [
        f
        for f in frames
        if f["event"] == "activity.step"
        and f.get("tool") == "make_charts"
        and f["status"] == "failed"
    ]
    assert call["detail"].startswith(classed_sentence("schema_invalid").removesuffix("."))
    assert call["detail"].endswith("Aid to Kenya (2)")
    assert is_user_sentence(call["detail"])
    (created,) = [f for f in frames if f["event"] == "component.created"]
    assert created["component_id"] == "cmp_01test0001"
    assert created["created_from"]["step_id"] == lines[0]["step_id"]
    done = _frame(frames, "done")
    assert [b["component_id"] for b in done["composition"]["blocks"]] == ["cmp_01test0001"]


async def test_a_planner_run_shows_working_on_a_plan_and_not_the_planners_late_reports(
    real_schemas: list[ToolSchema],
) -> None:
    """A plan call opens "Working on a plan" as the draft starts and
    completes it before the plan card. The planner's `on_progress` reports,
    relayed only once the draft is done, never reach the wire (see
    `harness.studio_translate`'s module docstring)."""

    class ProgressPlanner(ScriptedPlanner):
        async def __call__(self, *, on_progress: Any = None, **kwargs: Any) -> dict[str, Any]:
            if on_progress is not None:
                on_progress({"agent": "dfo_planner", "tools": ["get_skill"]})
                on_progress({"agent": "dfo_planner", "tools": ["propose_plan"]})
            return await super().__call__(**kwargs)

    app = _app(*_plan(), planner=ProgressPlanner(real_schemas))
    async with _client(app) as client:
        frames = await _stream(client, "/api/studio/turn", _turn_body("q with forks"))

    lines = [
        (index, frame["label"], frame["status"])
        for index, frame in enumerate(frames)
        if frame["event"] == "activity.step" and frame["label"] != "Reading your question"
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

    app = _app(
        *_plan(),
        planner=ScriptedPlanner(real_schemas),
        tool_surface=DiesOnTheSecondCall(),
    )
    async with _client(app) as client:
        first = await _stream(client, "/api/studio/turn", _turn_body("q with forks"))
        plan = _frame(first, "plan.proposed")
        failed = await _stream(
            client, "/api/studio/interrupt", _interrupt_body(plan["interrupt_id"], "approve")
        )
        second = await _stream(client, "/api/studio/turn", _turn_body("kenya oda 2023"))

    # The plan's own steps; "Starting the plan" is a status line with no tool.
    # Each step's line opens as it starts, and the step the crash cut off
    # closes as failed.
    lines: dict[str, list[str]] = {}
    for frame in failed:
        if frame["event"] == "activity.step" and frame.get("tool"):
            lines.setdefault(frame["step_id"], []).append(frame["status"])
    assert list(lines.values()) == [["running", "complete"], ["running", "failed"]]
    assert _events(failed)[-1] == "error"
    assert "component.created" not in _events(second)
    assert _events(second)[-1] == "done"


async def test_a_cancelled_turn_keeps_its_recorded_steps_out_of_the_next_turn(
    real_schemas: list[ToolSchema],
) -> None:
    """A step that finished while a later one hung was recorded and never
    translated, because no workflow event follows a cancel. It dies with its
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
    app = _app(*_plan(), planner=ScriptedPlanner(real_schemas), tool_surface=surface)
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
    step_tools = {f.get("tool") for f in second if f["event"] == "activity.step"}
    assert not step_tools & {"get_data", "growth"}
    assert _events(second)[-1] == "done"


async def test_a_malformed_drafted_plan_ends_the_turn_as_an_error_not_a_silent_success() -> None:
    """A draft the planner could not itself validate must not reach the approve
    gate or the executor. `harness.graph.planning`'s `planner` node validates the planner's
    record as a `planner.plan.Plan` before writing it to state, outside
    `handing_back_planner_failures`, which hands back only a planner's own
    `PlannerAdapterError`."""

    async def drafts_badly(**kwargs: Any) -> dict[str, Any]:
        return {"plan": {"decisions": "not-a-list"}, "summary": {}}

    executor = ScriptedExecutor(strict=True)
    app = _app(*_plan(), executor=executor, planner=drafts_badly)
    async with _client(app) as client:
        frames = await _stream(client, "/api/studio/turn", _turn_body("q with forks"))

    assert _events(frames)[-1] == "error"
    assert "plan.proposed" not in _events(frames)
    assert executor.calls == []


# --- the plan flow: pause, approve, execute --------------------------------------


class _HeldSends:
    """`EarlyMints`' send: records each call, then waits for `release`."""

    def __init__(self) -> None:
        self.release = asyncio.Event()
        self.sent: list[dict[str, Any]] = []

    async def __call__(self, args: dict[str, Any]) -> str:
        self.sent.append(args)
        await self.release.wait()
        return "posted"


async def test_the_stream_that_offers_a_card_closes_once_its_early_mints_are_posted() -> None:
    """The card's frames all reach the stream, which stays open until the
    card's early call is done and closes right after."""
    sends = _HeldSends()
    early = EarlyMints(send=sends, max_concurrent=4, settle_s=30.0)
    app = _app(*_plan("q with forks"), tool_surface=ScriptedSurface(), early_mints=early)
    async with _client(app) as client:
        streaming = asyncio.ensure_future(
            _stream(client, "/api/studio/turn", _turn_body("q with forks"))
        )
        for _ in range(500):
            if sends.sent:
                break
            await asyncio.sleep(0.01)
        await asyncio.sleep(0.2)
        assert sends.sent and not streaming.done()
        sends.release.set()
        frames = await asyncio.wait_for(streaming, timeout=10)
    assert _events(frames)[-1] == "plan.proposed"


async def test_the_stream_that_waits_on_early_mints_flushes_telemetry_after_they_end(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The last flush of the card's response comes after its early call ended,
    so the call's spans are exported while the request still has CPU."""
    sends = _HeldSends()
    ended: list[bool] = []

    async def flush() -> None:
        ended.append(sends.release.is_set())

    monkeypatch.setattr(studio_module, "flush_telemetry", flush)
    monkeypatch.setattr(studio_stream_module, "flush_telemetry", flush)
    early = EarlyMints(send=sends, max_concurrent=4, settle_s=30.0)
    app = _app(*_plan("q with forks"), tool_surface=ScriptedSurface(), early_mints=early)
    async with _client(app) as client:
        streaming = asyncio.ensure_future(
            _stream(client, "/api/studio/turn", _turn_body("q with forks"))
        )
        for _ in range(500):
            if ended:
                break
            await asyncio.sleep(0.01)
        sends.release.set()
        await asyncio.wait_for(streaming, timeout=10)

    assert ended[-1] is True


async def test_an_approval_posted_while_the_cards_stream_waits_on_early_mints_is_taken() -> None:
    """The card's stream releases the conversation before it waits on the early
    calls, so an approval posted while it still waits runs at once."""
    sends = _HeldSends()
    early = EarlyMints(send=sends, max_concurrent=4, settle_s=30.0)
    app = _app(
        *_plan("q with forks", *_charts_the_plan()),
        tool_surface=ScriptedSurface(),
        early_mints=early,
    )
    store = app.state.studio.store
    try:
        async with _client(app) as client:
            first = asyncio.ensure_future(
                _stream(client, "/api/studio/turn", _turn_body("q with forks"))
            )
            # The in-process transport hands the stream over only once it has
            # ended, so the card's pause is read off the conversation instead.
            for _ in range(500):
                conversation = store.current(PRINCIPAL, CONVERSATION)
                if sends.sent and conversation.pending and not conversation.lock.locked():
                    break
                await asyncio.sleep(0.01)
            (interrupt_id,) = conversation.pending
            assert not first.done()

            second = await asyncio.wait_for(
                _stream(client, "/api/studio/interrupt", _interrupt_body(interrupt_id, "approve")),
                timeout=10,
            )

            assert "plan.approved" in _events(second)
            assert not first.done()
            sends.release.set()
            assert _events(await asyncio.wait_for(first, timeout=10))[-1] == "plan.proposed"
    finally:
        await early.aclose()


async def test_the_stream_waits_no_longer_than_the_bound_for_early_mints(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A call still pending at the bound is logged and left running. The
    stream carries the same frames as a stream with no early mints."""
    sends = _HeldSends()
    early = EarlyMints(send=sends, max_concurrent=4, settle_s=0.2)
    app = _app(*_plan("q with forks"), tool_surface=ScriptedSurface(), early_mints=early)
    try:
        with caplog.at_level("INFO", logger="harness.early_mint"):
            async with _client(app) as client:
                started = time.monotonic()
                frames = await asyncio.wait_for(
                    _stream(client, "/api/studio/turn", _turn_body("q with forks")), timeout=10
                )
                waited = time.monotonic() - started
        assert len(sends.sent) == 1
        assert any("outcome=unsent_at_stream_end" in r.getMessage() for r in caplog.records)
        assert waited < 5
        plain = _app(*_plan("q with forks"), tool_surface=ScriptedSurface())
        async with _client(plain) as client:
            unwaited = await _stream(client, "/api/studio/turn", _turn_body("q with forks"))
        assert _events(frames) == _events(unwaited)
        assert (
            _frame(frames, "plan.proposed")["steps"] == _frame(unwaited, "plan.proposed")["steps"]
        )
    finally:
        await early.aclose()


async def test_plan_proposed_pauses_and_approve_resumes_on_a_new_stream() -> None:
    """The scripted planner and surface (`harness.scripted`'s fakes)
    draft the plan and run it, and once it is handed back the agent charts
    its frame. `executor` serves only the hand-back's `describe_frame` and the
    chart's `write_component`, since the plan's own steps run through
    `tool_surface`."""
    executor = ScriptedExecutor()
    app = _app(
        *_plan("q with forks", *_charts_the_plan()),
        executor=executor,
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
    # The approved plan leads the resumed stream, once, as the card showed it.
    assert events[0] == "plan.approved"
    assert events.count("plan.approved") == 1
    approved = _frame(second, "plan.approved")
    assert approved["plan_version_id"] == plan["plan_version_id"]
    assert approved["version"] == plan["version"]
    assert approved["title"] == plan["title"]
    assert approved["caveats"] == []
    assert [step["label"] for step in approved["steps"]] == [
        step["label"] for step in plan["steps"]
    ]
    # The plan's own two steps ran through `tool_surface`, invisibly to this
    # `executor`: it saw the hand-back describe the plan's frame, then the chart.
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


def _turn_spans(exporter: InMemorySpanExporter) -> int:
    """How many `turn` spans `exporter` holds finished; the model and agent
    spans a turn also exports are left out."""
    return sum(span.name == "turn" for span in exporter.get_finished_spans())


def _recording_span_ends(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str | None, str]]:
    """The status and level of each turn span that ends, in order."""
    ended: list[tuple[str | None, str]] = []
    real_end = studio_state_module.end_turn_span

    def recording_end(handle: Any, **kwargs: Any) -> None:
        if handle is not None:
            ended.append((kwargs.get("error"), kwargs.get("level", "ERROR")))
        real_end(handle, **kwargs)

    monkeypatch.setattr(studio_state_module, "end_turn_span", recording_end)
    monkeypatch.setattr(studio_stream_module, "end_turn_span", recording_end)
    return ended


async def test_a_turn_paused_for_approval_exports_a_span_per_segment(
    otlp_spans: InMemorySpanExporter,
) -> None:
    """`/turn` and the `/interrupt` that resumes it are separate requests on
    separate tasks, each a segment with a `turn` span of its own whose
    `dfo.external_id` is the turn's wire `trace_id`. Spans and events made in
    the search of the first segment nest under the first, and those made in
    the executor of the resumed one under the second."""

    class ProbingSearch(CannedSearch):
        async def __call__(self, **kwargs: Any) -> dict[str, Any]:
            with otel_trace.get_tracer("test.probe").start_as_current_span("resolve_probe"):
                return await super().__call__(**kwargs)

    class ProbingExecutor(ScriptedExecutor):
        async def __call__(
            self, *, tool: str, args: dict[str, Any], on_progress: ProgressCallback | None = None
        ) -> dict[str, Any]:
            with otel_trace.get_tracer("test.probe").start_as_current_span("executor_probe"):
                record_event("executor.probed", metadata={"tool": tool})
                return await super().__call__(tool=tool, args=args)

    app = _app(
        search_call("aid to Kenya"),
        *_plan("q with forks", *_charts_the_plan()),
        resolver=ProbingSearch(definite_envelope(spec_id=SPEC_A)),
        executor=ProbingExecutor(),
        tool_surface=ScriptedSurface(),
    )
    async with _client(app) as client:
        first = await _stream(client, "/api/studio/turn", _turn_body("q with forks"))
        plan = _frame(first, "plan.proposed")
        second = await _stream(
            client, "/api/studio/interrupt", _interrupt_body(plan["interrupt_id"], "approve")
        )

    spans = list(otlp_spans.get_finished_spans())
    paused, resumed = [span for span in spans if span.name == "turn"]
    trace_id = _frame(second, "done")["trace_id"]
    assert [(turn.attributes or {})["dfo.external_id"] for turn in (paused, resumed)] == [
        trace_id,
        trace_id,
    ]
    (resolve_probe,) = [span for span in spans if span.name == "resolve_probe"]
    assert resolve_probe.context.trace_id == paused.context.trace_id
    executor_probes = [span for span in spans if span.name == "executor_probe"]
    assert len(executor_probes) == 2  # describe_frame, then write_component
    assert {span.context.trace_id for span in executor_probes} == {resumed.context.trace_id}
    # Each segment's `turn` is a root, with that segment's probes below it.
    assert [span.name for span in spans if span.parent is None] == ["turn", "turn"]
    assert [event.name for event in executor_probes[-1].events] == ["executor.probed"]


async def test_every_stream_flushes_telemetry_before_the_response_ends(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    flushes: list[str] = []

    async def flush() -> None:
        flushes.append("flush")

    monkeypatch.setattr(studio_module, "flush_telemetry", flush)
    monkeypatch.setattr(studio_stream_module, "flush_telemetry", flush)
    app = _app(*_plan())
    async with _client(app) as client:
        first = await _stream(client, "/api/studio/turn", _turn_body("q with forks"))
        assert flushes == ["flush"]  # the stream that paused for approval
        plan = _frame(first, "plan.proposed")
        await _stream(
            client, "/api/studio/interrupt", _interrupt_body(plan["interrupt_id"], "reject")
        )

    assert flushes == ["flush", "flush"]  # the rejecting stream


async def test_stopping_a_paused_turn_ends_its_trace_as_cancelled(
    otlp_spans: InMemorySpanExporter, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The segment that paused ended and exported its span as its stream
    ended. The `/cancel` that closes the paused turn records a zero-length
    span of its own on the turn's trace, `cancelled` at warning level, and
    exports it."""
    ended = _recording_span_ends(monkeypatch)
    finished_at_flush: list[list[str]] = []

    async def flush() -> None:
        finished_at_flush.append(
            [
                (span.attributes or {})["dfo.external_id"]
                for span in otlp_spans.get_finished_spans()
                if span.name == "turn"
            ]
        )

    monkeypatch.setattr(studio_module, "flush_telemetry", flush)
    monkeypatch.setattr(studio_stream_module, "flush_telemetry", flush)
    app = _app(*_plan())
    async with _client(app) as client:
        await _stream(client, "/api/studio/turn", _turn_body("q with forks"))
        live = app.state.studio.store.current(PRINCIPAL, CONVERSATION)
        assert live is not None
        trace_id = live.trace_id

        cancel = await client.post(
            "/api/studio/cancel",
            json={"kind": "turn.cancel", "schema_version": 1, "conversation_id": CONVERSATION},
            headers=AUTH,
        )

    assert cancel.status_code == 204
    assert ended == [("paused", "DEFAULT"), ("cancelled", "WARNING")]
    assert finished_at_flush == [[trace_id], [trace_id, trace_id]]


async def test_cancelling_an_idle_conversation_flushes_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    flushes: list[str] = []

    async def flush() -> None:
        flushes.append("flush")

    monkeypatch.setattr(studio_module, "flush_telemetry", flush)
    monkeypatch.setattr(studio_stream_module, "flush_telemetry", flush)
    async with _client(_app()) as client:
        cancel = await client.post(
            "/api/studio/cancel",
            json={"kind": "turn.cancel", "schema_version": 1, "conversation_id": CONVERSATION},
            headers=AUTH,
        )

    assert cancel.status_code == 204
    assert flushes == []


async def test_a_paused_segment_s_span_ends_paused_and_the_answer_opens_its_own(
    otlp_spans: InMemorySpanExporter, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The segment that pauses for approval ends its span as `paused`, and
    exports it before its stream ends. The answer's segment opens a span of
    its own on the same trace id and ends it `done`."""
    app = _app(*_plan())
    ended = _recording_span_ends(monkeypatch)
    exported_at_flush: list[list[str]] = []

    async def flush() -> None:
        exported_at_flush.append(
            [
                (span.attributes or {})["dfo.external_id"]
                for span in otlp_spans.get_finished_spans()
                if span.name == "turn"
            ]
        )

    monkeypatch.setattr(studio_module, "flush_telemetry", flush)
    monkeypatch.setattr(studio_stream_module, "flush_telemetry", flush)

    async with _client(app) as client:
        first = await _stream(client, "/api/studio/turn", _turn_body("q with forks"))
        plan = _frame(first, "plan.proposed")
        assert ended == [("paused", "DEFAULT")]
        second = await _stream(
            client, "/api/studio/interrupt", _interrupt_body(plan["interrupt_id"], "reject")
        )

    trace_id = _frame(second, "done")["trace_id"]
    assert ended == [("paused", "DEFAULT"), ("done", "DEFAULT")]
    assert exported_at_flush == [[trace_id], [trace_id, trace_id]]


@pytest.mark.parametrize("stopped", [False, True], ids=["rejected", "stopped"])
async def test_a_rejected_plan_s_stream_ends_its_span_and_reads_a_stop(
    otlp_spans: InMemorySpanExporter, monkeypatch: pytest.MonkeyPatch, stopped: bool
) -> None:
    """The stream that answers a plan's rejection runs as every segment's
    stream does: its span ends `done` and is exported. The rejection closes
    the turn before its first frame, which ends any Stop for it, so a Stop
    before the stream's closing write changes nothing, and the stream still
    ends on `done`, as the span says."""
    app = _app(*_plan(), tool_surface=ScriptedSurface())
    store = app.state.studio.store
    ended = _recording_span_ends(monkeypatch)
    async with _client(app) as client:
        plan = _frame(await _stream(client, "/api/studio/turn", _turn_body("q")), "plan.proposed")
        closing, proceed = asyncio.Event(), asyncio.Event()
        real_close = store.close_segment

        async def close(*args: Any, **kwargs: Any) -> SegmentClose:
            if stopped and not closing.is_set():
                closing.set()
                await proceed.wait()
            return await real_close(*args, **kwargs)

        store.close_segment = close  # type: ignore[method-assign]
        rejecting = asyncio.ensure_future(
            _stream(
                client, "/api/studio/interrupt", _interrupt_body(plan["interrupt_id"], "reject")
            )
        )
        if stopped:
            await asyncio.wait_for(closing.wait(), timeout=5)
            await _post_stop(client)
            proceed.set()
        frames = await asyncio.wait_for(rejecting, timeout=5)

    assert _ending_frame(frames) == "done"
    assert ended == [("paused", "DEFAULT"), ("done", "DEFAULT")]
    assert _turn_spans(otlp_spans) == 2


async def test_a_pause_whose_closing_write_fails_leaves_no_plan_to_answer() -> None:
    """The closing write after `plan.proposed` was held fails, so the pause
    frame never goes out and nothing of the pause is stored: the answer to
    the plan the client never saw is a 404, the lease lapses by its TTL, and
    the next `/turn` after that streams."""
    clock = await StoreClock.start(_engine())
    store = StudioStore(_engine(), **clock.store_kwargs)
    app = _app(*_plan(), _text("Next."), tool_surface=ScriptedSurface(), store=store)
    real_close = store.close_segment
    interrupt_ids: list[str] = []

    async def failing_once(*args: Any, **kwargs: Any) -> SegmentClose:
        if not interrupt_ids:
            live = store.current(PRINCIPAL, CONVERSATION)
            assert live is not None
            interrupt_ids.extend(live.pending)
            return SegmentClose.FAILED
        return await real_close(*args, **kwargs)

    store.close_segment = failing_once  # type: ignore[method-assign]
    async with _client(app) as client:
        first = await _stream(client, "/api/studio/turn", _turn_body("q with forks"))
        assert "plan.proposed" not in _events(first)
        (interrupt_id,) = interrupt_ids
        answer = await client.post(
            "/api/studio/interrupt", json=_interrupt_body(interrupt_id, "approve"), headers=AUTH
        )
        assert answer.status_code == 404
        await clock.advance(studio_state_module.SEGMENT_TTL_S + 1)
        second = await _stream(client, "/api/studio/turn", _turn_body("next"))

    assert _events(second)[-1] == "done"


async def test_a_refusal_whose_draft_cannot_be_dropped_still_releases_the_lock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Giving up the working copy raises on a `/turn` refused as busy; the
    conversation's lock is released all the same."""
    app = _app(_text("Hello."))
    store = app.state.studio.store
    other = StudioStore(_engine())

    def broken_drop(*args: Any) -> bool:
        raise RuntimeError("drop failed")

    async with _client(app) as client:
        await _stream(client, "/api/studio/turn", _turn_body("hi"))
        await other.load(PRINCIPAL, CONVERSATION)
        other.claim_segment(PRINCIPAL, CONVERSATION)
        await other.save(PRINCIPAL, CONVERSATION)
        monkeypatch.setattr(store, "drop_draft", broken_drop)
        with pytest.raises(RuntimeError, match="drop failed"):
            await client.post("/api/studio/turn", json=_turn_body("again"), headers=AUTH)

    assert not store.lock_for(PRINCIPAL, CONVERSATION).locked()


@pytest.mark.parametrize(
    ("client_leaves", "span"),
    [
        pytest.param(True, ("cancelled", "WARNING"), id="client-gone"),
        pytest.param(False, ("failed", "ERROR"), id="inner-cancellation"),
    ],
)
async def test_a_request_cancelled_after_its_claim_ends_the_span_as_cancelled(
    otlp_spans: InMemorySpanExporter,
    monkeypatch: pytest.MonkeyPatch,
    client_leaves: bool,
    span: tuple[str, str],
) -> None:
    """`/turn` claims, and a `CancelledError` ends the request while the
    turn is being opened. When the client left, which cancels the request's
    own task, the claimed segment's span ends `cancelled` at warning level:
    a client gone is no failure. A `CancelledError` an inner task let out,
    with the request's task never cancelled, is a failure."""
    app = _app(_text("Hello."))
    studio = app.state.studio
    ended = _recording_span_ends(monkeypatch)
    opening = asyncio.Event()

    async def hanging_open(*args: Any, **kwargs: Any) -> None:
        opening.set()
        if not client_leaves:
            raise asyncio.CancelledError
        await asyncio.Event().wait()

    monkeypatch.setattr(studio.turns, "open_turn", hanging_open)
    request = asyncio.ensure_future(
        _turn_endpoint(studio)(
            TurnRequest.model_validate(_turn_body("hi")), authorization=AUTH["Authorization"]
        )
    )
    await asyncio.wait_for(opening.wait(), timeout=5)
    if client_leaves:
        request.cancel()
    with pytest.raises(asyncio.CancelledError):
        await request

    assert ended == [span]
    assert not studio.store.lock_for(PRINCIPAL, CONVERSATION).locked()


# --- the span table: every way a segment ends and a turn closes ------------------------
#
# A stream row drives one way a `/turn` segment's stream ends through the
# route, whose owner ends the segment's span (`harness.studio_stream`), with
# a store double answering the closing write as named, or a Stop, a lost
# lease or a client leaving at a named point, and returns the frames that
# stream sent. A closing row drives the requests that close a turn without
# a stream of its own, and names every span they end.


def _turn_endpoint(studio: Any) -> Callable[..., Any]:
    router = create_studio_router(studio, settings=Settings())
    return next(
        route.endpoint
        for route in router.routes
        if getattr(route, "path", "") == "/api/studio/turn"
    )


def _close_as(store: StudioStore, result: SegmentClose) -> None:
    """Make the segment's closing write answer `result` without writing:
    `lost` as when another instance took the lease, `failed` as when the
    write did not commit."""

    async def close_segment(*args: Any, **kwargs: Any) -> SegmentClose:
        return result

    store.close_segment = close_segment  # type: ignore[method-assign]


def _stop_before_the_close(store: StudioStore) -> None:
    """Record a Stop on the segment's live lease as its closing write starts,
    after its work ended, as `/cancel` from any instance records one."""
    real = store.close_segment

    async def close_segment(principal: str, conversation_id: str, *args: Any) -> SegmentClose:
        await store.request_cancel(principal, conversation_id)
        return await real(principal, conversation_id, *args)

    store.close_segment = close_segment  # type: ignore[method-assign]


async def _sse_frames(chunks: Any) -> list[dict[str, Any]]:
    """The frames of a raw SSE body read so far."""
    frames = []
    for chunk in chunks:
        text = chunk.decode() if isinstance(chunk, bytes) else chunk
        frames += [
            json.loads(line[len("data: ") :])
            for line in text.splitlines()
            if line.startswith("data: ")
        ]
    return frames


async def _stream_done(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    async with _client(_app(_text("Hello."))) as client:
        return await _stream(client, "/api/studio/turn", _turn_body("hi"))


async def _stream_paused(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    async with _client(_app(*_plan())) as client:
        return await _stream(client, "/api/studio/turn", _turn_body("q with forks"))


async def _stream_paused_close_fails(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    app = _app(*_plan())
    _close_as(app.state.studio.store, SegmentClose.FAILED)
    async with _client(app) as client:
        return await _stream(client, "/api/studio/turn", _turn_body("q with forks"))


async def _stream_done_close_fails(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    app = _app(_text("Hello."))
    _close_as(app.state.studio.store, SegmentClose.FAILED)
    async with _client(app) as client:
        return await _stream(client, "/api/studio/turn", _turn_body("hi"))


async def _stream_done_close_lost(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    app = _app(_text("Hello."))
    _close_as(app.state.studio.store, SegmentClose.LOST)
    async with _client(app) as client:
        return await _stream(client, "/api/studio/turn", _turn_body("hi"))


async def _stream_lost_at_the_heartbeat(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    app = _app(*_card(), resolver=_HangingResolver(), heartbeat_seconds=0.05)

    async def taken(*args: Any, **kwargs: Any) -> Renewal:
        return Renewal(held=False, stop_id=None)

    app.state.studio.store.renew = taken  # type: ignore[method-assign]
    async with _client(app) as client:
        return await asyncio.wait_for(_stream(client, "/api/studio/turn", _turn_body()), timeout=5)


async def _stream_unavailable_at_the_heartbeat(
    monkeypatch: pytest.MonkeyPatch,
) -> list[dict[str, Any]]:
    app = _app(*_card(), resolver=_HangingResolver(), heartbeat_seconds=0.05)
    store = app.state.studio.store

    async def unreachable(*args: Any, **kwargs: Any) -> Renewal:
        raise DatabaseUnavailable("unreachable")

    store.renew = unreachable  # type: ignore[method-assign]
    store.lease_deadline = lambda *args: 0.0  # type: ignore[method-assign]
    async with _client(app) as client:
        return await asyncio.wait_for(_stream(client, "/api/studio/turn", _turn_body()), timeout=5)


async def _stream_unavailable_at_a_card_write(
    monkeypatch: pytest.MonkeyPatch,
) -> list[dict[str, Any]]:
    app = _app(*_card(), resolver=_search(), executor=ScriptedExecutor(_card_results()))

    async def unreachable(*args: Any, **kwargs: Any) -> Any:
        raise DatabaseUnavailable("unreachable")

    app.state.studio.store.change_canvas = unreachable  # type: ignore[method-assign]
    async with _client(app) as client:
        return await _stream(client, "/api/studio/turn", _turn_body())


async def _stop_mid_run(app: FastAPI, resolver: _HangingResolver) -> list[dict[str, Any]]:
    async with _client(app) as client:
        streaming = asyncio.ensure_future(_stream(client, "/api/studio/turn", _turn_body()))
        await asyncio.wait_for(resolver.entered.wait(), timeout=5)
        await _post_stop(client)
        return await asyncio.wait_for(streaming, timeout=5)


async def _stream_stopped_mid_run(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    resolver = _HangingResolver()
    return await _stop_mid_run(_app(*_card(), resolver=resolver), resolver)


async def _stream_stopped_mid_run_close_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> list[dict[str, Any]]:
    resolver = _HangingResolver()
    app = _app(*_card(), resolver=resolver)
    _close_as(app.state.studio.store, SegmentClose.FAILED)
    return await _stop_mid_run(app, resolver)


async def _stream_stopped_after_the_pause(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    app = _app(*_plan())
    _stop_before_the_close(app.state.studio.store)
    async with _client(app) as client:
        return await _stream(client, "/api/studio/turn", _turn_body("q with forks"))


async def _stream_stopped_after_done(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    app = _app(_text("Hello."))
    _stop_before_the_close(app.state.studio.store)
    async with _client(app) as client:
        return await _stream(client, "/api/studio/turn", _turn_body("hi"))


async def _stream_failed(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    app = _app(_text("Hello."))

    def broken(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("the segment failed")

    monkeypatch.setattr(studio_module, "keep_turn_records", broken)
    async with _client(app) as client:
        return await _stream(client, "/api/studio/turn", _turn_body("hi"))


async def _stream_node_failed(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    app = _app(_text("Hello."))
    studio = app.state.studio
    graph = studio.graph

    async def broken(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("the node failed")

    start = replace(graph.nodes[graph.start], run=broken)
    monkeypatch.setattr(studio, "graph", replace(graph, nodes={**graph.nodes, graph.start: start}))
    async with _client(app) as client:
        return await _stream(client, "/api/studio/turn", _turn_body("hi"))


async def _leave(
    response: Any,
    until: Callable[[list[dict[str, Any]]], bool],
    enough: asyncio.Event | None = None,
) -> list[dict[str, Any]]:
    """Serve `response` as the server does, and leave once `until` holds for
    the frames read, or `enough` is set: the client's disconnect reaches
    the response as `http.disconnect`. Returns the frames read once the
    response has returned."""
    chunks: list[bytes] = []
    enough = enough or asyncio.Event()

    async def receive() -> dict[str, Any]:
        await enough.wait()
        return {"type": "http.disconnect"}

    async def send(message: dict[str, Any]) -> None:
        if message["type"] == "http.response.body":
            chunks.append(message.get("body", b""))
            if until(await _sse_frames(chunks)):
                enough.set()

    scope = {"type": "http", "asgi": {"spec_version": "2.3"}, "method": "POST", "headers": []}
    await asyncio.wait_for(response(scope, receive, send), timeout=10)
    return await _sse_frames(chunks)


async def _read_until(
    app: FastAPI,
    until: Callable[[list[dict[str, Any]]], bool],
    enough: asyncio.Event | None = None,
) -> Any:
    """Serve `/turn`'s response until `until` holds for the frames read, or
    `enough` is set, then leave (`_leave`). Returns the frames read."""
    studio = app.state.studio
    response = await _turn_endpoint(studio)(
        TurnRequest.model_validate(_turn_body()), authorization=AUTH["Authorization"]
    )
    return await _leave(response, until, enough)


async def _stream_client_gone_mid_run(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    resolver = _HangingResolver()
    app = _app(*_card(), resolver=resolver)
    return await _read_until(app, lambda frames: False, resolver.entered)


async def _stream_client_gone_at_a_frame(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """The client takes the stream's first frame and leaves."""
    app = _app(*_card(), resolver=_HangingResolver())
    return await _read_until(app, bool)


async def _stream_client_gone_after_the_pause_frame(
    monkeypatch: pytest.MonkeyPatch,
) -> list[dict[str, Any]]:
    app = _app(*_plan())
    return await _read_until(
        app, lambda frames: any(frame["event"] == "plan.proposed" for frame in frames)
    )


async def _stream_without_its_ending_frame(
    monkeypatch: pytest.MonkeyPatch,
) -> list[dict[str, Any]]:
    app = _app(_text("Hello."))

    async def no_frames(self: Any) -> list[Any]:
        return []

    monkeypatch.setattr(studio_module._Translator, "finish", no_frames)
    async with _client(app) as client:
        return await _stream(client, "/api/studio/turn", _turn_body("hi"))


def _ending_frame(frames: list[dict[str, Any]]) -> str | None:
    """The last ending frame on the wire: `done`, `pause`, or a stream-fatal
    error frame's code; `None` when the stream sent none."""
    ending = None
    for frame in frames:
        if frame["event"] == "done":
            ending = "done"
        elif frame["event"] in ("plan.proposed", "elicitation.request"):
            ending = "pause"
        elif frame["event"] == "error":
            ending = frame["error"]["error"]
    return ending


@pytest.mark.parametrize(
    ("drive", "wire", "span", "saved", "gone"),
    [
        pytest.param(_stream_done, "done", ("done", "DEFAULT"), True, False, id="turn-ends"),
        pytest.param(_stream_paused, "pause", ("paused", "DEFAULT"), True, False, id="turn-pauses"),
        pytest.param(
            _stream_paused_close_fails,
            "transient_unavailable",
            ("unsaved", "ERROR"),
            False,
            False,
            id="turn-pauses-close-fails",
        ),
        pytest.param(
            _stream_done_close_fails,
            "transient_unavailable",
            ("unsaved", "ERROR"),
            False,
            False,
            id="turn-ends-close-fails",
        ),
        pytest.param(
            _stream_done_close_lost,
            "transient_unavailable",
            ("lost", "WARNING"),
            False,
            False,
            id="turn-ends-close-lost",
        ),
        pytest.param(
            _stream_lost_at_the_heartbeat,
            "transient_unavailable",
            ("lost", "WARNING"),
            False,
            False,
            id="heartbeat-loss",
        ),
        pytest.param(
            _stream_unavailable_at_the_heartbeat,
            "transient_unavailable",
            ("lost", "WARNING"),
            False,
            False,
            id="heartbeat-unavailable",
        ),
        pytest.param(
            _stream_unavailable_at_a_card_write,
            "transient_unavailable",
            ("lost", "WARNING"),
            False,
            False,
            id="card-write-unavailable",
        ),
        pytest.param(
            _stream_stopped_mid_run,
            "cancelled",
            ("cancelled", "WARNING"),
            True,
            False,
            id="stop-mid-run",
        ),
        pytest.param(
            _stream_stopped_mid_run_close_fails,
            "cancelled",
            ("cancelled", "WARNING"),
            False,
            False,
            id="stop-mid-run-close-fails",
        ),
        pytest.param(
            _stream_stopped_after_the_pause,
            "cancelled",
            ("cancelled", "WARNING"),
            True,
            False,
            id="stop-after-the-pause",
        ),
        pytest.param(
            _stream_stopped_after_done,
            "done",
            ("done", "DEFAULT"),
            True,
            False,
            id="stop-after-done",
        ),
        pytest.param(
            _stream_node_failed,
            "capability_unavailable",
            ("failed", "ERROR"),
            True,
            False,
            id="turn-fails",
        ),
        pytest.param(
            _stream_failed,
            "capability_unavailable",
            ("unclosed", "ERROR"),
            True,
            False,
            id="segment-fails",
        ),
        pytest.param(
            _stream_without_its_ending_frame,
            "capability_unavailable",
            ("unclosed", "ERROR"),
            True,
            False,
            id="segment-ends-without-its-frame",
        ),
        pytest.param(
            _stream_client_gone_mid_run,
            None,
            ("left", "WARNING"),
            True,
            True,
            id="client-gone-mid-run",
        ),
        pytest.param(
            _stream_client_gone_at_a_frame,
            None,
            ("left", "WARNING"),
            True,
            True,
            id="client-gone-at-a-frame",
        ),
        pytest.param(
            _stream_client_gone_after_the_pause_frame,
            "pause",
            ("paused", "DEFAULT"),
            True,
            False,
            id="client-gone-after-the-pause-frame",
        ),
    ],
)
async def test_a_segment_s_span_names_how_its_stream_ended(
    otlp_spans: InMemorySpanExporter,
    monkeypatch: pytest.MonkeyPatch,
    drive: Callable[[pytest.MonkeyPatch], Any],
    wire: str | None,
    span: tuple[str, str],
    saved: bool,
    gone: bool,
) -> None:
    """The row's stream sends `wire` as its last ending frame. Its segment's
    span ends once with the status and level `harness.studio_stream.ending`
    gives that row, with `dfo.saved` saying whether the closing write
    committed and `dfo.client_gone` whether the client had left."""
    ended = _recording_span_ends(monkeypatch)
    frames = await drive(monkeypatch)
    assert _ending_frame(frames) == wire
    assert ended[-1] == span
    turns = [exported for exported in otlp_spans.get_finished_spans() if exported.name == "turn"]
    assert len(turns) == len(ended)
    attributes = turns[-1].attributes or {}
    assert attributes.get("dfo.saved") == str(saved).lower()
    assert attributes.get("dfo.client_gone") == str(gone).lower()
    assert "dfo.stored" not in attributes


# --- the ending table against the real store --------------------------------------
#
# Each row ends a segment's work one way through a route and its real work
# adapter: `/turn`'s first segment, `/interrupt`'s resume of the plan it
# paused on, or `/interrupt`'s rejection of that plan. It makes the closing
# write on the real store end one way by acting on the database first: a
# Stop recorded on the lease, the lease handed to another segment, or the
# conversation's row held locked by another connection past a short
# timeout. The row asserts what the real store returned, the wire and span
# `harness.studio_stream.ending` gives, and what a fresh instance reads
# back.


async def _work_ends_done(
    app: FastAPI, resolver: Any, monkeypatch: pytest.MonkeyPatch
) -> list[dict[str, Any]]:
    async with _client(app) as client:
        return await _stream(client, "/api/studio/turn", _turn_body("hi"))


async def _work_pauses(
    app: FastAPI, resolver: Any, monkeypatch: pytest.MonkeyPatch
) -> list[dict[str, Any]]:
    async with _client(app) as client:
        return await _stream(client, "/api/studio/turn", _turn_body("q with forks"))


async def _work_fails_at_a_node(
    app: FastAPI, resolver: Any, monkeypatch: pytest.MonkeyPatch
) -> list[dict[str, Any]]:
    studio = app.state.studio
    graph = studio.graph

    async def broken(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("the node failed")

    start = replace(graph.nodes[graph.start], run=broken)
    monkeypatch.setattr(studio, "graph", replace(graph, nodes={**graph.nodes, graph.start: start}))
    return await _work_ends_done(app, resolver, monkeypatch)


async def _work_stopped(
    app: FastAPI, resolver: Any, monkeypatch: pytest.MonkeyPatch
) -> list[dict[str, Any]]:
    return await _stop_mid_run(app, resolver)


async def _work_raises(
    app: FastAPI, resolver: Any, monkeypatch: pytest.MonkeyPatch
) -> list[dict[str, Any]]:
    def broken(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("the segment failed")

    monkeypatch.setattr(studio_module, "keep_turn_records", broken)
    return await _work_ends_done(app, resolver, monkeypatch)


async def _work_left(
    app: FastAPI, resolver: Any, monkeypatch: pytest.MonkeyPatch
) -> list[dict[str, Any]]:
    return await _read_until(app, lambda frames: False, resolver.entered)


def _interrupt_endpoint(studio: Any) -> Callable[..., Any]:
    router = create_studio_router(studio, settings=Settings())
    return next(
        route.endpoint
        for route in router.routes
        if getattr(route, "path", "") == "/api/studio/interrupt"
    )


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


class _BrokenSurface(ScriptedSurface):
    """A plan step's tool call fails, so the node running it fails."""

    async def call_tool(self, **kwargs: Any) -> ToolCallResult:
        raise RuntimeError("analysis mount went away")


async def _paused_plan(client: httpx2.AsyncClient) -> dict[str, Any]:
    """`/turn`'s first segment, paused on its plan; the plan.proposed frame."""
    return _frame(
        await _stream(client, "/api/studio/turn", _turn_body("q with forks")), "plan.proposed"
    )


async def _plan_approved(
    app: FastAPI, resolver: Any, monkeypatch: pytest.MonkeyPatch
) -> list[dict[str, Any]]:
    async with _client(app) as client:
        plan = await _paused_plan(client)
        return await _stream(
            client, "/api/studio/interrupt", _interrupt_body(plan["interrupt_id"], "approve")
        )


async def _plan_switched(
    app: FastAPI, resolver: Any, monkeypatch: pytest.MonkeyPatch
) -> list[dict[str, Any]]:
    async with _client(app) as client:
        plan = await _plan_with_a_choice(app, client)
        return await _stream(client, "/api/studio/interrupt", _switch_body(plan))


async def _plan_approved_then_stopped(
    app: FastAPI, resolver: Any, monkeypatch: pytest.MonkeyPatch
) -> list[dict[str, Any]]:
    hanging = app.state.studio.surface_recorder._surface
    async with _client(app) as client:
        plan = await _paused_plan(client)
        streaming = asyncio.ensure_future(
            _stream(
                client, "/api/studio/interrupt", _interrupt_body(plan["interrupt_id"], "approve")
            )
        )
        await asyncio.wait_for(hanging.entered.wait(), timeout=5)
        await _post_stop(client)
        return await asyncio.wait_for(streaming, timeout=5)


async def _plan_approved_then_raises(
    app: FastAPI, resolver: Any, monkeypatch: pytest.MonkeyPatch
) -> list[dict[str, Any]]:
    def broken(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("the segment failed")

    async with _client(app) as client:
        plan = await _paused_plan(client)
        monkeypatch.setattr(studio_module, "keep_turn_records", broken)
        return await _stream(
            client, "/api/studio/interrupt", _interrupt_body(plan["interrupt_id"], "approve")
        )


async def _plan_approved_then_left(
    app: FastAPI, resolver: Any, monkeypatch: pytest.MonkeyPatch
) -> list[dict[str, Any]]:
    studio = app.state.studio
    async with _client(app) as client:
        plan = await _paused_plan(client)
    response = await _interrupt_endpoint(studio)(
        InterruptResponse.model_validate(_interrupt_body(plan["interrupt_id"], "approve")),
        authorization=AUTH["Authorization"],
    )
    return await _leave(response, lambda frames: False, studio.surface_recorder._surface.entered)


async def _plan_rejected(
    app: FastAPI, resolver: Any, monkeypatch: pytest.MonkeyPatch
) -> list[dict[str, Any]]:
    async with _client(app) as client:
        plan = await _paused_plan(client)
        return await _stream(
            client, "/api/studio/interrupt", _interrupt_body(plan["interrupt_id"], "reject")
        )


async def _edit_ends(
    app: FastAPI, resolver: Any, monkeypatch: pytest.MonkeyPatch
) -> list[dict[str, Any]]:
    async with _client(app) as client:
        await _conversation_with_component(client)
        return await _stream(client, "/api/studio/turn", _edit_body("rename it"))


async def _edit_stopped(
    app: FastAPI, resolver: Any, monkeypatch: pytest.MonkeyPatch
) -> list[dict[str, Any]]:
    async with _client(app) as client:
        await _conversation_with_component(client)
        streaming = asyncio.ensure_future(
            _stream(client, "/api/studio/turn", _edit_body("rename it"))
        )
        await asyncio.wait_for(app.state.studio.edit_model.entered.wait(), timeout=5)
        await _post_stop(client)
        return await asyncio.wait_for(streaming, timeout=5)


async def _edit_heartbeat_raises(
    app: FastAPI, resolver: Any, monkeypatch: pytest.MonkeyPatch
) -> list[dict[str, Any]]:
    studio = app.state.studio

    async def broken(*args: Any, **kwargs: Any) -> None:
        await studio.edit_model.entered.wait()
        raise RuntimeError("the heartbeat failed")

    monkeypatch.setattr(studio, "_heartbeat", broken)
    return await _edit_ends(app, resolver, monkeypatch)


async def _edit_left(
    app: FastAPI, resolver: Any, monkeypatch: pytest.MonkeyPatch
) -> list[dict[str, Any]]:
    studio = app.state.studio
    async with _client(app) as client:
        await _conversation_with_component(client)
    response = await _turn_endpoint(studio)(
        TurnRequest.model_validate(_edit_body("rename it")), authorization=AUTH["Authorization"]
    )
    return await _leave(response, lambda frames: False, studio.edit_model.entered)


class _FailingEditModel:
    """An edit model whose draft breaks the edit's rules."""

    async def __call__(self, **kwargs: Any) -> dict[str, Any]:
        raise ValueError("the edit drafted both a patch and a recreate")


def _edit_app(finish: Finish) -> dict[str, Any]:
    """The search, executor and edit model of an Edit Card driver's app: a
    card turn draws the card the edit then reads, and an edit that ends
    `done` writes it."""
    results = [*_card_results(write_token=WRITE_TOKEN), _described(), _VIEW]
    edit_model: Any = _HangingEditModel()
    if finish is Finish.DONE:
        revised = {
            **COMPONENT,
            "revision": 2,
            "fact_summary": {**COMPONENT["fact_summary"], "revision": 2},
        }
        results.append({"outcome": "ok", "frame_refs": [], "component": revised})
        edit_model = ScriptedEditModel(
            {"text": "Renamed the card.", "patch": {"title": "Aid to Kenya"}, "recreate": None}
        )
    elif finish is Finish.FAILED:
        edit_model = _FailingEditModel()
    return {"resolver": _search(), "executor": ScriptedExecutor(results), "edit_model": edit_model}


# Each way a route's work ends: its driver, the app's script, the resolver
# its search waits on (`None` for the default), the tool surface its plan
# steps call, and how many closing writes its driver lets through before
# the one under test (the `/turn` that pauses before an `/interrupt`).
_PLAN_SCRIPT = (*_plan(), _text("Hello."))
_WORK_ENDS: dict[tuple[str, Finish], tuple[Any, tuple[types.Part, ...], bool, Any, int]] = {
    ("turn", Finish.DONE): (_work_ends_done, (_text("Hello."),), False, None, 0),
    ("turn", Finish.PAUSED): (_work_pauses, tuple(_plan()), False, None, 0),
    ("turn", Finish.FAILED): (_work_fails_at_a_node, (_text("Hello."),), False, None, 0),
    ("turn", Finish.CANCELLED): (_work_stopped, tuple(_card()), True, None, 0),
    ("turn", Finish.UNCLOSED): (_work_raises, (_text("Hello."),), False, None, 0),
    ("turn", Finish.LEFT): (_work_left, tuple(_card()), True, None, 0),
    ("resume", Finish.DONE): (_plan_approved, _PLAN_SCRIPT, False, ScriptedSurface, 1),
    ("resume", Finish.PAUSED): (_plan_switched, tuple(_plan()), False, ScriptedSurface, 1),
    ("resume", Finish.FAILED): (_plan_approved, _PLAN_SCRIPT, False, _BrokenSurface, 1),
    ("resume", Finish.CANCELLED): (
        _plan_approved_then_stopped,
        _PLAN_SCRIPT,
        False,
        _HangingSurface,
        1,
    ),
    ("resume", Finish.UNCLOSED): (
        _plan_approved_then_raises,
        _PLAN_SCRIPT,
        False,
        ScriptedSurface,
        1,
    ),
    ("resume", Finish.LEFT): (_plan_approved_then_left, _PLAN_SCRIPT, False, _HangingSurface, 1),
    ("reject", Finish.DONE): (_plan_rejected, tuple(_plan()), False, ScriptedSurface, 1),
    ("edit", Finish.DONE): (_edit_ends, tuple(_card()), False, None, 1),
    ("edit", Finish.FAILED): (_edit_ends, tuple(_card()), False, None, 1),
    ("edit", Finish.CANCELLED): (_edit_stopped, tuple(_card()), False, None, 1),
    ("edit", Finish.UNCLOSED): (_edit_heartbeat_raises, tuple(_card()), False, None, 1),
    ("edit", Finish.LEFT): (_edit_left, tuple(_card()), False, None, 1),
}


def _close_on_the_database(
    store: StudioStore, close: SegmentClose, *, after: int = 0
) -> list[SegmentClose]:
    """Make the segment's closing write on the real store end `close` by
    acting on the database before it runs, once `after` closing writes have
    gone through untouched; returns the results the write under test gave."""
    real = store.close_segment
    results: list[SegmentClose] = []
    passed = 0

    async def close_segment(principal: str, conversation_id: str, segment_id: str) -> SegmentClose:
        nonlocal passed
        if passed < after:
            passed += 1
            return await real(principal, conversation_id, segment_id)
        if close is SegmentClose.STOPPED:
            await store.request_cancel(principal, conversation_id)
        if close is SegmentClose.LOST:
            async with _engine().begin() as conn:
                await conn.execute(text("UPDATE conversations SET lease_segment_id = 'seg_other'"))
        if close is SegmentClose.FAILED:
            async with _engine().begin() as conn:
                await conn.execute(text("SELECT id FROM conversations FOR UPDATE"))
                results.append(await real(principal, conversation_id, segment_id, timeout=0.3))
                return results[-1]
        results.append(await real(principal, conversation_id, segment_id))
        return results[-1]

    store.close_segment = close_segment  # type: ignore[method-assign]
    return results


_SAVED, _STOPPED, _LOST, _FAILED = (
    SegmentClose.SAVED,
    SegmentClose.STOPPED,
    SegmentClose.LOST,
    SegmentClose.FAILED,
)


_CAP = "capability_unavailable"
_TRANSIENT = "transient_unavailable"


_ENDINGS_BY_FINISH = {
    Finish.DONE: [
        (_SAVED, "done", "done", "done"),
        (_LOST, _TRANSIENT, "done", None),
        (_FAILED, _TRANSIENT, "done", None),
    ],
    Finish.PAUSED: [
        (_SAVED, "pause", "paused", None),
        (_STOPPED, "cancelled", "cancelled", "cancelled"),
        (_LOST, _TRANSIENT, "paused", None),
        (_FAILED, _TRANSIENT, "paused", None),
    ],
    Finish.FAILED: [
        (_SAVED, _CAP, "failed", _CAP),
        (_LOST, _CAP, "failed", None),
        (_FAILED, _CAP, "failed", None),
    ],
    Finish.CANCELLED: [
        (_SAVED, "cancelled", "cancelled", "cancelled"),
        (_LOST, "cancelled", "cancelled", None),
        (_FAILED, "cancelled", "cancelled", None),
    ],
    Finish.UNCLOSED: [
        (_SAVED, _CAP, "running", _CAP),
        (_STOPPED, "cancelled", "cancelled", "cancelled"),
        (_LOST, _TRANSIENT, "running", None),
        (_FAILED, _CAP, "running", None),
    ],
    Finish.LEFT: [
        (_SAVED, None, "running", None),
        (_STOPPED, None, "cancelled", "cancelled"),
        (_LOST, None, "running", None),
        (_FAILED, None, "running", None),
    ],
}


def _endings(route: str, finish: Finish) -> list[Any]:
    """The cells of `finish`'s row of the ending table `route` reaches, as
    `(route, finish, close, wire, cursor, recorded)`. A resume whose work
    raises does so before its segment runs, so its turn stays paused where
    `/turn`'s stays running. An Edit Card stream opens no turn on the loop,
    so the only cursor is the card turn's before it, `done`, and its closing
    write never finds a Stop live for a turn with no cursor."""
    paused = route == "resume" and finish is Finish.UNCLOSED
    if route == "edit":
        return [
            (route, finish, close, wire, "done", recorded)
            for close, wire, _, recorded in _ENDINGS_BY_FINISH[finish]
            if close is not _STOPPED
        ]
    return [
        (
            route,
            finish,
            close,
            wire,
            "paused" if paused and cursor == "running" else cursor,
            recorded,
        )
        for close, wire, cursor, recorded in _ENDINGS_BY_FINISH[finish]
    ]


@pytest.mark.parametrize(
    ("route", "finish", "close", "wire", "cursor", "recorded"),
    [
        *(
            row
            for finish in Finish
            if finish in _ENDINGS_BY_FINISH
            for row in _endings("turn", finish)
        ),
        *(
            row
            for finish in Finish
            if finish in _ENDINGS_BY_FINISH
            for row in _endings("resume", finish)
        ),
        *_endings("reject", Finish.DONE),
        *(
            row
            for finish in Finish
            if ("edit", finish) in _WORK_ENDS
            for row in _endings("edit", finish)
        ),
    ],
    ids=str,
)
async def test_every_ending_this_route_reaches_holds_on_the_real_store(
    otlp_spans: InMemorySpanExporter,
    monkeypatch: pytest.MonkeyPatch,
    route: str,
    finish: Finish,
    close: SegmentClose,
    wire: str | None,
    cursor: str,
    recorded: str | None,
) -> None:
    """The work `route` streams ends `finish` and the real store's closing
    write ends `close`. The stream's last ending frame is `wire`, the span
    ends on the status `ending` gives that cell with `dfo.saved` saying
    whether the write committed, and a fresh instance reads the turn's
    cursor as `cursor`, the record closed on `recorded` (open for `None`),
    and the lease cleared exactly when the write committed."""
    drive, script, hangs, surface, after = _WORK_ENDS[route, finish]
    resolver = _HangingResolver() if hangs else None
    kwargs: dict[str, Any] = {
        "resolver": resolver,
        "tool_surface": surface() if surface is not None else None,
    }
    if route == "edit":
        kwargs |= _edit_app(finish)
    app = _app(*script, **kwargs)
    results = _close_on_the_database(app.state.studio.store, close, after=after)
    ended = _recording_span_ends(monkeypatch)

    frames = await drive(app, resolver, monkeypatch)

    assert results == [close]
    end = ending(WorkEnd(finish), close)
    assert _ending_frame(frames) == wire
    assert ended[-1][0] == end.status.value
    spans = [span for span in otlp_spans.get_finished_spans() if span.name == "turn"]
    assert len(spans) == len(ended) == after + 1
    assert (spans[-1].attributes or {}).get("dfo.saved") == str(end.saved).lower()
    assert await _turn_statuses() == [cursor]
    async with _engine().connect() as conn:
        lease = (
            await conn.execute(text("SELECT lease_segment_id FROM conversations"))
        ).scalar_one()
    assert (lease is None) is end.saved
    stored = await StudioStore(_engine()).load(PRINCIPAL, CONVERSATION)
    assert stored is not None
    assert stored.turn_recorded is (recorded is not None)
    if recorded is not None:
        assert stored.turns[-1].get("error", "done") == recorded


async def test_a_declined_plan_whose_client_left_before_the_first_chunk_is_recorded() -> None:
    """The client of an `/interrupt` that declines the plan is gone before
    the response's first chunk: the server's send raises at once, so the
    response sets the client-gone flag before the owner first runs. The
    owner records the rejection before anything else, so the turn its
    cursor closed is recorded on its reply and the lease is cleared."""
    app = _app(*_plan(), tool_surface=ScriptedSurface())
    studio = app.state.studio
    async with _client(app) as client:
        plan = await _paused_plan(client)
    response = await _interrupt_endpoint(studio)(
        InterruptResponse.model_validate(_interrupt_body(plan["interrupt_id"], "reject")),
        authorization=AUTH["Authorization"],
    )

    async def receive() -> dict[str, Any]:
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    async def send(message: dict[str, Any]) -> None:
        raise OSError("the client is gone")

    scope = {"type": "http", "asgi": {"spec_version": "2.4"}, "method": "POST", "headers": []}
    with pytest.raises(ClientDisconnect):
        await asyncio.wait_for(response(scope, receive, send), timeout=10)

    stored = await StudioStore(_engine()).load(PRINCIPAL, CONVERSATION)
    assert stored is not None
    assert stored.turn_recorded
    assert "error" not in stored.turns[-1]
    assert stored.segment is None


async def test_an_edit_card_stopped_and_left_before_its_next_renewal_is_recorded_cancelled() -> (
    None
):
    """A Stop from another instance lands on an Edit Card stream's lease,
    and the client leaves before the stream's heartbeat next renews. The
    owner cuts the edit off as left, and its closing write acts on the Stop
    its own lease carries: the turn, which has no cursor, is recorded
    `cancelled`, the lease is cleared and no Stop stays live."""
    app = _app(*_card(), **_edit_app(Finish.LEFT))
    studio = app.state.studio
    async with _client(app) as client:
        await _conversation_with_component(client)
    response = await _turn_endpoint(studio)(
        TurnRequest.model_validate(_edit_body("rename it")), authorization=AUTH["Authorization"]
    )
    stopped = asyncio.Event()

    async def stop_elsewhere() -> None:
        await studio.edit_model.entered.wait()
        app_b = _app(tool_surface=ScriptedSurface(), store=StudioStore(_engine()))
        async with _client(app_b) as client_b:
            assert (await _post_stop(client_b)).status_code == 204
        stopped.set()

    stopping = asyncio.create_task(stop_elsewhere())
    frames = await _leave(response, lambda frames: False, stopped)
    await stopping

    assert _ending_frame(frames) is None
    stored = await StudioStore(_engine()).load(PRINCIPAL, CONVERSATION)
    assert stored is not None
    assert stored.turn_recorded
    assert stored.turns[-1]["error"] == "cancelled"
    assert stored.segment is None
    assert not await StudioStore(_engine()).stop_live(PRINCIPAL, CONVERSATION)


async def _close_claim_refused_before_its_save(monkeypatch: pytest.MonkeyPatch) -> None:
    app = _app(_text("Hello."))

    async def claimed_elsewhere(
        principal: str, conversation_id: str, *args: Any, **kwargs: Any
    ) -> None:
        raise ConversationStale(conversation_id)

    app.state.studio.store.claim_turn_lease = claimed_elsewhere  # type: ignore[method-assign]
    async with _client(app) as client:
        refused = await client.post("/api/studio/turn", json=_turn_body("hi"), headers=AUTH)
    assert refused.status_code == 409


async def _close_failure_after_the_claim(monkeypatch: pytest.MonkeyPatch) -> None:
    app = _app(_text("Hello."))

    async def unavailable(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("turn store unavailable")

    monkeypatch.setattr(app.state.studio.turns, "open_turn", unavailable)
    async with _client(app) as client:
        with pytest.raises(RuntimeError):
            await _stream(client, "/api/studio/turn", _turn_body("hi"))


async def _close_cancel_after_the_claim(monkeypatch: pytest.MonkeyPatch) -> None:
    app = _app(_text("Hello."))
    studio = app.state.studio
    opening = asyncio.Event()

    async def hanging(*args: Any, **kwargs: Any) -> None:
        opening.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(studio.turns, "open_turn", hanging)
    request = asyncio.ensure_future(
        _turn_endpoint(studio)(
            TurnRequest.model_validate(_turn_body("hi")), authorization=AUTH["Authorization"]
        )
    )
    await asyncio.wait_for(opening.wait(), timeout=5)
    request.cancel()
    with pytest.raises(asyncio.CancelledError):
        await request


async def _close_answer_refused_after_its_claim(monkeypatch: pytest.MonkeyPatch) -> None:
    app = _app(*_plan(), tool_surface=ScriptedSurface())
    turns = app.state.studio.turns
    real_open_cursor = turns.open_cursor
    reads: list[int] = []

    async def moved_after_the_claim(*args: Any, **kwargs: Any) -> Any:
        # The read after the claim finds the turn no longer paused there.
        reads.append(1)
        return None if len(reads) == 2 else await real_open_cursor(*args, **kwargs)

    async with _client(app) as client:
        plan = _frame(
            await _stream(client, "/api/studio/turn", _turn_body("q with forks")), "plan.proposed"
        )
        reads.clear()
        monkeypatch.setattr(turns, "open_cursor", moved_after_the_claim)
        moved = await client.post(
            "/api/studio/interrupt",
            json=_interrupt_body(plan["interrupt_id"], "approve"),
            headers=AUTH,
        )
    assert moved.status_code == 409


async def _close_cancel_of_a_paused_turn(monkeypatch: pytest.MonkeyPatch) -> None:
    async with _client(_app(*_plan())) as client:
        await _stream(client, "/api/studio/turn", _turn_body("q with forks"))
        await _post_stop(client)


async def _close_cancel_of_a_turn_its_client_left(monkeypatch: pytest.MonkeyPatch) -> None:
    resolver = _HangingResolver()
    app = _app(*_card(), resolver=resolver)
    await _read_until(app, lambda frames: False, resolver.entered)
    async with _client(app) as client:
        assert (await _post_stop(client)).status_code == 204


async def _crashed_first_segment(app: FastAPI) -> None:
    """The first segment's instance claimed the turn and died before it
    opened: the turn is stored begun, its lease dropped, and its span never
    reached the trace."""
    store = app.state.studio.store
    live = await store.create(PRINCIPAL, CONVERSATION)
    live.begin_turn()
    live.turn_user_message = "q"
    segment_id = store.claim_segment(PRINCIPAL, CONVERSATION)
    await store.save(PRINCIPAL, CONVERSATION)
    await store.drop_lease(PRINCIPAL, CONVERSATION, segment_id)


async def _close_cancel_of_a_turn_never_opened(monkeypatch: pytest.MonkeyPatch) -> None:
    app = _app()
    await _crashed_first_segment(app)
    async with _client(app) as client:
        assert (await _post_stop(client)).status_code == 204


async def _close_turn_over_a_stopped_turn_left_open(monkeypatch: pytest.MonkeyPatch) -> None:
    resolver = _HangingResolver()
    app = _app(search_call("aid to Kenya"), _text("Hello."), resolver=resolver)
    await _read_until(app, lambda frames: False, resolver.entered)
    # Another instance claims, takes a Stop on its lease and dies: the Stop
    # stays live for the turn left open, with no holder to act on it.
    other = StudioStore(_engine())
    await other.load(PRINCIPAL, CONVERSATION)
    segment_id = other.claim_segment(PRINCIPAL, CONVERSATION)
    await other.save(PRINCIPAL, CONVERSATION)
    assert await other.request_cancel(PRINCIPAL, CONVERSATION)
    await other.drop_lease(PRINCIPAL, CONVERSATION, segment_id)
    assert await _stop_live()
    async with _client(app) as client:
        await _stream(client, "/api/studio/turn", _turn_body("hi"))


async def _close_turn_over_a_paused_turn(monkeypatch: pytest.MonkeyPatch) -> None:
    async with _client(_app(*_plan(), _text("Hello."))) as client:
        await _stream(client, "/api/studio/turn", _turn_body("q with forks"))
        await _stream(client, "/api/studio/turn", _turn_body("hi"))


async def _close_turn_over_a_crashed_resume(monkeypatch: pytest.MonkeyPatch) -> None:
    app = _app(*_plan(), _text("Hello."))
    store = app.state.studio.store
    async with _client(app) as client:
        await _stream(client, "/api/studio/turn", _turn_body("q with forks"))
        # The answer's segment claimed the pause on an instance that then
        # died: its entry is resuming, and its span never reached the trace.
        live = store.current(PRINCIPAL, CONVERSATION)
        assert live is not None
        for pending in live.pending.values():
            pending.resuming_segment = "seg_crashed"
        await store.save(PRINCIPAL, CONVERSATION)
        await _stream(client, "/api/studio/turn", _turn_body("hi"))


async def _close_turn_over_a_crashed_first_segment(monkeypatch: pytest.MonkeyPatch) -> None:
    app = _app(_text("Hello."))
    await _crashed_first_segment(app)
    async with _client(app) as client:
        await _stream(client, "/api/studio/turn", _turn_body("hi"))


async def _close_turn_over_one_that_failed_before_streaming(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = _app(_text("Hello."))
    turns = app.state.studio.turns
    open_turn = turns.open_turn

    async def unavailable(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("turn store unavailable")

    turns.open_turn = unavailable
    async with _client(app) as client:
        with pytest.raises(RuntimeError):
            await _stream(client, "/api/studio/turn", _turn_body("hi"))
        turns.open_turn = open_turn
        await _stream(client, "/api/studio/turn", _turn_body("hi again"))


_DONE = ("done", "DEFAULT")
_PAUSED = ("paused", "DEFAULT")
_ABANDONED = ("abandoned", "WARNING")
_CANCELLED = ("cancelled", "WARNING")
_LEFT = ("left", "WARNING")


@pytest.mark.parametrize(
    ("drive", "spans"),
    [
        pytest.param(
            _close_claim_refused_before_its_save,
            [("not_started", "WARNING")],
            id="claim-refused-before-its-save",
        ),
        pytest.param(
            _close_failure_after_the_claim, [("failed", "ERROR")], id="failure-after-the-claim"
        ),
        pytest.param(_close_cancel_after_the_claim, [_CANCELLED], id="cancel-after-the-claim"),
        pytest.param(
            _close_answer_refused_after_its_claim,
            [_PAUSED, ("not_started", "WARNING")],
            id="answer-refused-after-its-claim",
        ),
        pytest.param(
            _close_cancel_of_a_paused_turn, [_PAUSED, _CANCELLED], id="cancel-of-a-paused-turn"
        ),
        pytest.param(
            _close_cancel_of_a_turn_its_client_left,
            [_LEFT, _CANCELLED],
            id="cancel-of-a-turn-its-client-left",
        ),
        pytest.param(
            _close_cancel_of_a_turn_never_opened,
            [_CANCELLED],
            id="cancel-of-a-turn-never-opened",
        ),
        pytest.param(
            _close_turn_over_a_stopped_turn_left_open,
            [_LEFT, _CANCELLED, _DONE],
            id="turn-over-a-stopped-turn-left-open",
        ),
        pytest.param(
            _close_turn_over_a_paused_turn, [_PAUSED, _ABANDONED, _DONE], id="turn-over-a-pause"
        ),
        pytest.param(
            _close_turn_over_a_crashed_resume,
            [_PAUSED, _ABANDONED, _DONE],
            id="turn-over-a-crashed-resume",
        ),
        pytest.param(
            _close_turn_over_a_crashed_first_segment,
            [_ABANDONED, _DONE],
            id="turn-over-a-crashed-first-segment",
        ),
        pytest.param(
            _close_turn_over_one_that_failed_before_streaming,
            [("failed", "ERROR"), _ABANDONED, _DONE],
            id="turn-over-one-that-failed-before-streaming",
        ),
    ],
)
async def test_a_turn_closed_without_a_stream_of_its_own_ends_its_spans_as_the_table_says(
    otlp_spans: InMemorySpanExporter,
    monkeypatch: pytest.MonkeyPatch,
    drive: Callable[[pytest.MonkeyPatch], Any],
    spans: list[tuple[str, str]],
) -> None:
    """Every span a row's requests end has the status and level of that
    way of closing (`record_unstreamed_span`, `record_closed_turn_span`, or
    the stream's owner), in the order they end, and each is exported
    once."""
    ended = _recording_span_ends(monkeypatch)
    await drive(monkeypatch)
    assert ended == spans
    assert _turn_spans(otlp_spans) == len(spans)


class ScriptedPlanner:
    """A `PlannerModel`, standing in for `harness.planner_adapter`.

    Builds a real, translated `planner.plan.Plan` directly rather than
    running the planner agent and its translator (covered model-free in
    `test_planner_adapter.py`). It proves the wire from a `planner.plan.Plan`-
    shaped payload to the plan card: any `PlannerModel`, including the real
    `harness.planner_adapter.PlannerAdapter`, returns that same shape, so a
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


async def test_a_planners_own_labels_and_decisions_reach_the_plan_card(
    real_schemas: list[ToolSchema],
) -> None:
    """`plan.proposed`'s steps, decisions, requirements and operations are
    composed by `harness.plan_display` from the planner's real `planner.plan.Plan`,
    not the scripted record's bare tool names. Approving still ends the turn
    cleanly by the revision key, and runs the workflow's steps through
    `execute_approved_plan` against a real `ToolSurface`: every step produces
    an `activity.step` frame with a real `duration_ms`, not merely a subset
    of them.
    """
    planner = ScriptedPlanner(real_schemas)
    app = _app(
        *_plan(),
        planner=planner,
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
    # The planner drafted for the agent's brief.
    assert planner.calls[0]["question"] == "q with forks"
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


def _tool_steps(frames: list[dict[str, Any]]) -> list[tuple[str, str, str, str | None]]:
    """Every tool step's `activity.step` frame, as (step_id, tool, status, detail)."""
    return [
        (frame["step_id"], frame["tool"], frame["status"], frame.get("detail"))
        for frame in frames
        if frame["event"] == "activity.step" and frame.get("tool")
    ]


class _PollsOnceSurface(ScriptedSurface):
    """`ScriptedSurface`, whose `get_data` reports once and waits a moment
    before it answers, as data-mcp waits between two polls."""

    async def call_tool(
        self,
        *,
        tool: ToolId,
        arguments: Mapping[str, Any],
        on_progress: ProgressCallback | None = None,
    ) -> ToolCallResult:
        if tool.name == "get_data" and on_progress is not None:
            on_progress(ProgressReport(progress=12, message=_POLLING))
            await asyncio.sleep(0.05)
        return await super().call_tool(tool=tool, arguments=arguments)


async def test_a_polled_get_data_step_updates_one_activity_line_until_it_completes() -> None:
    """While `get_data` waits on a background operation, its step streams a
    `running` frame carrying data-mcp's report; once the call answers, the
    line drops the report and shows the step's description again, then completes
    on the same `step_id`. Studio keeps one
    Activity line per step, so the plan's two steps still read as two
    lines. A plan step's surface call is the only place progress reaches
    the wire: the agent's `fetch` calls `get_data` with none."""
    app = _app(*_plan(), tool_surface=_PollsOnceSurface())
    async with _client(app) as client:
        first = await _stream(client, "/api/studio/turn", _turn_body("q with forks"))
        plan = _frame(first, "plan.proposed")
        second = await _stream(
            client, "/api/studio/interrupt", _interrupt_body(plan["interrupt_id"], "approve")
        )

    steps = _tool_steps(second)
    get_data = [step for step in steps if step[1] == "get_data"]
    source = "life expectancy at birth in Kenya, 1960 to 2023"
    assert [(status, detail) for _, _, status, detail in get_data] == [
        ("running", f"{_POLLING} \u00b7 step 1 of 2"),
        ("running", f"step 1 of 2 \u00b7 {source}"),
        ("complete", "64 observations \u00b7 World Development Indicators"),
    ]
    assert len({step_id for step_id, *_ in get_data}) == 1
    plan_lines = {step_id for step_id, tool, *_ in steps if tool in ("get_data", "aggregate")}
    assert len(plan_lines) == len(plan["steps"]) == 2
    assert _events(second)[-1] == "done"


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
    "surface", [_PollsThenHangsSurface(), _BusySurface()], ids=["mid-call", "retry-delay"]
)
async def test_a_step_its_budget_cuts_off_closes_its_activity_line_as_failed(
    monkeypatch: pytest.MonkeyPatch, surface: ToolSurface
) -> None:
    """A step whose budget runs out mid-call, or during a retry's delay,
    streamed `running` frames and never returned a result. Its line still
    ends on a `failed` frame before the turn moves on, so Studio shows no
    step waiting after the run is over."""
    monkeypatch.setattr("executor.run.step_budget_s", lambda *_args, **_kwargs: 0.2)
    app = _app(*_plan(), tool_surface=surface)
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
    assert is_user_sentence(get_data[-1]["detail"])
    assert get_data[-1]["duration_ms"] is not None
    assert second.index(get_data[-1]) < _events(second).index("plan.proposed")


async def test_planner_lines_and_plan_step_lines_never_share_a_step_id_across_an_approval() -> None:
    """Studio resumes an approved turn into the Activity entry the plan was
    drafted in and matches lines by `step_id`, so a plan step's id must
    differ from every planner line's id in the segment before it."""
    app = _app(*_plan(), tool_surface=ScriptedSurface())
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


async def test_an_approval_the_approve_node_refuses_sends_no_approved_plan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The approve node takes no approval whose time it cannot read, and routes the
    answer to the planner as an amendment: the stream offers the plan again and
    names no approved plan."""
    monkeypatch.setattr(approve_module, "_answered_at", lambda answer: None)
    app = _app(*_plan(), tool_surface=ScriptedSurface())
    async with _client(app) as client:
        first = await _stream(client, "/api/studio/turn", _turn_body("q with forks"))
        plan = _frame(first, "plan.proposed")

        second = await _stream(
            client, "/api/studio/interrupt", _interrupt_body(plan["interrupt_id"], "approve")
        )

    events = _events(second)
    assert "plan.approved" not in events
    assert events[-1] == "plan.proposed"


async def test_amend_yields_plan_versioned_then_a_fresh_proposal() -> None:
    executor = ScriptedExecutor()
    app = _app(
        *_plan("q with forks", *_charts_the_plan()),
        executor=executor,
        tool_surface=ScriptedSurface(),
    )
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
        assert "plan.approved" not in events
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
    assert _events(third).count("plan.approved") == 1
    assert _frame(third, "plan.approved")["version"] == 2
    # The plan's own steps ran through `tool_surface`; `executor` only saw
    # the hand-back describe the plan's frame, then the chart.
    assert [c["tool"] for c in executor.calls] == ["describe_frame", "write_component"]


async def test_reject_ends_the_turn_without_computing() -> None:
    executor = ScriptedExecutor()
    async with _client(_app(*_plan(), executor=executor)) as client:
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
    standing in for a planner that reads an amendment as a new question rather
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
    and only card 2's own id approves card 2. A replay naming card 1's
    revision is refused 409 with card 2 to show in its place."""
    planner = _UnrelatedPlansPlanner(["first measure", "second measure"], real_schemas)
    app = _app(
        *_plan(),
        planner=planner,
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

        stale = await client.post(
            "/api/studio/interrupt",
            json={
                **_interrupt_body(card1["interrupt_id"], "approve"),
                "plan_revision": card1["revision"],
            },
            headers=AUTH,
        )
        assert stale.status_code == 409
        assert stale.json()["constraint_id"] == "plan_revision_stale"
        assert stale.json()["current_plan"] == card2

        third = await _stream(
            client, "/api/studio/interrupt", _interrupt_body(card2["interrupt_id"], "approve")
        )

    assert _events(third)[-1] == "done"


_BUDGET_EXHAUSTED_TEXT = (
    "This request has used its budget of 4 plan drafts, so nothing was "
    "recorded. Your next message starts with a fresh budget."
)
# `harness.graph`'s `planner` node appends this when it routes a refusal
# back to `approve` with a plan still `PENDING_APPROVAL`: the planner's own
# sentence states facts it can verify from inside one turn; the node adds
# the approvability claim, since only it holds `ctx.state["plan"]`.
_STILL_APPROVABLE_TEXT = (
    f"{_BUDGET_EXHAUSTED_TEXT} Version 1 is still there to approve; anything "
    "past that needs a new question."
)


class _RefusingAmendPlanner:
    """A `PlannerModel` that drafts once for real, then answers every later
    call with `PlannerAdapter`'s own `BUDGET_EXHAUSTED` shape -- standing in
    for a real adapter whose amendment round hits `MAX_PLAN_ROUNDS`, without
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
    """`PlannerAdapter`'s own `MAX_PLAN_ROUNDS` cap can refuse an amendment
    outright (`{"refused": {"text": ...}}`). The plan the user was already
    looking at is still `PENDING_APPROVAL` and untouched by that refusal,
    so the served turn re-offers it for approval instead of ending the
    turn on it: the refusal's own sentence streams as a message and a
    fresh `plan.proposed` for the very same version follows, answerable
    with the same revision the first card carried -- not merely a refusal
    sentence that says so.
    """
    executor = ScriptedExecutor()
    app = _app(
        *_plan("kenya debt"),
        executor=executor,
        planner=_RefusingAmendPlanner(schemas=real_schemas),
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
    assert "write_component" not in [c["tool"] for c in executor.calls]


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
    `PlannerAdapter`, which holds the turn's question fixed
    (`submission.service._resolve_intent`), refuses an amendment
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
    """A restart the planner refuses (`QUESTION_FIXED`)
    reaches the wire exactly like a `BUDGET_EXHAUSTED` refusal (see
    `test_a_refused_amendment_still_offers_the_pending_version_for_approval`,
    just above): the same pending plan is re-offered, under the same
    `plan_version_id` and `revision` -- proving `_interrupt_frames`
    (`harness.studio_translate`) neither regresses `latest_plan_version` nor
    mints a second wire id for a version it already gave one to -- and no
    `plan.versioned` frame fires, since nothing was recorded to supersede
    the first card with. The planner drafts for the agent's brief; the
    refused restart records nothing, so the second call sees the same brief
    the first one did.
    """
    executor = ScriptedExecutor()
    planner = _RestartRefusingPlanner(schemas=real_schemas)
    app = _app(
        *_plan("kenya debt"),
        executor=executor,
        planner=planner,
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
    assert "write_component" not in [c["tool"] for c in executor.calls]
    assert planner.questions_seen == ["kenya debt", "kenya debt"]


async def test_an_approval_is_recorded_with_its_approver_and_its_revision() -> None:
    """`/interrupt` appends a typed `ApprovalRecord` for every plan decision,
    not only for an approval. `auth_enforced` is `False` because `Settings()`
    here leaves `studio_api_token` unset (the auth-open case); the field
    lets a record from this case be told apart from one written with the
    token gate closed, which `test_an_interrupt_under_a_closed_token_gate_records_auth_enforced`
    below covers."""
    store = StudioStore(
        _engine(),
    )
    app = _app(*_plan(), store=store)

    async with _client(app) as client:
        first = await _stream(client, "/api/studio/turn", _turn_body())
        plan = _frame(first, "plan.proposed")
        await _stream(
            client, "/api/studio/interrupt", _interrupt_body(plan["interrupt_id"], "approve")
        )

    conversation = store.current(PRINCIPAL, CONVERSATION)
    assert len(conversation.approvals) == 1
    record = conversation.approvals[0]
    assert record.decision == "approved"
    assert record.approver == PRINCIPAL
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
    store = StudioStore(
        _engine(),
    )
    app = _app(
        *_plan(),
        store=store,
        settings=Settings(studio_api_token="test-token"),
    )

    async with _client(app) as client:
        first = await _stream(client, "/api/studio/turn", _turn_body())
        plan = _frame(first, "plan.proposed")
        await _stream(
            client, "/api/studio/interrupt", _interrupt_body(plan["interrupt_id"], "approve")
        )

    conversation = store.current(PRINCIPAL, CONVERSATION)
    assert len(conversation.approvals) == 1
    record = conversation.approvals[0]
    assert record.auth_enforced is True
    assert record.approver == PRINCIPAL


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
    store = StudioStore(
        _engine(),
    )
    app = _app(*_plan(), store=store)

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

        conversation = store.current(PRINCIPAL, CONVERSATION)
        assert plan["interrupt_id"] in conversation.pending
        assert conversation.approvals == []
        assert not conversation.lock.locked()
        stored = await StudioStore(
            _engine(),
        ).load(PRINCIPAL, CONVERSATION)
        assert stored is not None and stored.approvals == []
        assert stored.pending[plan["interrupt_id"]].answer is None

        await _stream(
            client, "/api/studio/interrupt", _interrupt_body(plan["interrupt_id"], "approve")
        )

    # The retry reaches `ApprovalRecord` unpatched: the record is appended
    # and the pending entry it was keyed on is finally consumed. (This
    # setup has no `tool_surface`, so the resumed turn's own execution
    # refuses downstream -- irrelevant to what this test is proving.)
    assert len(conversation.approvals) == 1
    assert plan["interrupt_id"] not in conversation.pending


async def test_a_failure_before_the_turn_streams_releases_the_lock_and_the_claimed_lease(
    otlp_spans: InMemorySpanExporter, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The turn cannot be opened after the claim save landed, so `/turn`
    fails before its stream starts. The lock and the segment it claimed are
    both released, the claimed segment's span ends `failed` and is
    exported, and the next `/turn` is accepted. That turn closes the first
    as `abandoned`, and records a zero-length span on the first turn's
    trace that says so."""
    ended = _recording_span_ends(monkeypatch)
    store = StudioStore(
        _engine(),
    )
    app = _app(_text("Hello."), store=store)
    turns = app.state.studio.turns
    open_turn = turns.open_turn

    async def _unavailable(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("turn store unavailable")

    turns.open_turn = _unavailable
    async with _client(app) as client:
        with pytest.raises(RuntimeError, match="turn store unavailable"):
            await _stream(client, "/api/studio/turn", _turn_body("hi"))

        assert not store.lock_for(PRINCIPAL, CONVERSATION).locked()
        live = store.current(PRINCIPAL, CONVERSATION)
        assert (live.turn_index, live.turn_recorded, live.segment) == (1, False, None)
        store.claim_segment(PRINCIPAL, CONVERSATION)  # not refused as busy
        assert _turn_spans(otlp_spans) == 1
        first_trace = live.trace_id
        stored = await store.load(PRINCIPAL, CONVERSATION)
        assert stored is not None and stored.segment is None
        turns.open_turn = open_turn
        frames = await _stream(client, "/api/studio/turn", _turn_body("hi again"))

    assert _events(frames)[-1] == "done"
    assert ended == [("failed", "ERROR"), ("abandoned", "WARNING"), ("done", "DEFAULT")]
    assert [
        (span.attributes or {})["dfo.external_id"]
        for span in otlp_spans.get_finished_spans()
        if span.name == "turn"
    ] == [first_trace, first_trace, _frame(frames, "done")["trace_id"]]


async def test_a_turn_whose_claim_loses_forgets_the_turn_it_began() -> None:
    """Another instance's `/turn` claimed the conversation since the load, so
    this request's claim save loses and it answers the retryable 409. The
    turn it began and the segment it claimed in memory were never saved, so
    the instance forgets them, and keeps the conversation's lock."""
    store = StudioStore(
        _engine(),
    )
    lock = store.lock_for(PRINCIPAL, CONVERSATION)
    app = _app(_text("Hello."), store=store)
    claim = store.claim_turn_lease

    async def claimed_elsewhere(
        principal: str, conversation_id: str, *args: Any, **kwargs: Any
    ) -> None:
        raise ConversationStale(conversation_id)

    store.claim_turn_lease = claimed_elsewhere  # type: ignore[method-assign]

    async with _client(app) as client:
        response = await client.post("/api/studio/turn", json=_turn_body("hi"), headers=AUTH)

    assert (response.status_code, response.json()["error"]) == (409, "transient_unavailable")
    live = store.current(PRINCIPAL, CONVERSATION)
    assert (live.turn_index, live.segment) == (0, None)
    assert store.lock_for(PRINCIPAL, CONVERSATION) is lock
    assert not lock.locked()
    store.claim_turn_lease = claim  # type: ignore[method-assign]
    stored = await store.load(PRINCIPAL, CONVERSATION)
    assert stored is not None and (stored.turn_index, stored.segment) == (0, None)


async def test_a_failure_before_a_resume_streams_keeps_the_answer_for_a_same_answer_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`/interrupt` fails after its claim save, before its stream starts:
    only the lease is dropped, so the same answer retried at once claims
    and resumes the turn, and the pause survives. The instance forgets the
    state it changed and never saved."""
    store = StudioStore(
        _engine(),
    )
    app = _app(*_ask("reading A", "reading B"), _text("You meant reading B."), store=store)
    async with _client(app) as client:
        first = await _stream(client, "/api/studio/turn", _turn_body("ambiguous q"))
        elicitation = _frame(first, "elicitation.request")
        (question,) = elicitation["questions"]
        body = _interrupt_body(
            elicitation["interrupt_id"],
            "answer",
            answers={"q1": _pick(question["options"][1]["id"])},
        )

        with monkeypatch.context() as patch:

            def _boom(*args: Any, **kwargs: Any) -> Any:
                raise RuntimeError("answer lookup failed")

            patch.setattr("harness.studio._given_answers", _boom)
            with pytest.raises(RuntimeError, match="answer lookup failed"):
                await _stream(client, "/api/studio/interrupt", body)

        assert not store.lock_for(PRINCIPAL, CONVERSATION).locked()
        assert store.current(PRINCIPAL, CONVERSATION).segment is None
        stored = await store.load(PRINCIPAL, CONVERSATION)
        assert stored is not None and stored.segment is None
        assert stored.pending[elicitation["interrupt_id"]].resuming_segment is not None
        frames = await _stream(client, "/api/studio/interrupt", body)

    assert _events(frames)[-1] == "done"
    assert _frame(frames, "message.complete")["text"] == "You meant reading B."


# --- asks the agent puts to the user ---------------------------------------------


async def test_an_ask_elicits_and_the_answer_resumes_on_a_new_stream() -> None:
    app = _app(*_ask("reading A", "reading B"), _text("You meant reading B."))
    async with _client(app) as client:
        first = await _stream(client, "/api/studio/turn", _turn_body("ambiguous q"))

        assert _events(first)[-1] == "elicitation.request"
        elicitation = _frame(first, "elicitation.request")
        (question,) = elicitation["questions"]
        assert question["input"] == "single_select"
        assert [option["label"] for option in question["options"]] == ["reading A", "reading B"]

        second = await _stream(
            client,
            "/api/studio/interrupt",
            _interrupt_body(
                elicitation["interrupt_id"],
                "answer",
                answers={"q1": _pick(question["options"][1]["id"])},
            ),
        )

    assert _events(second)[-1] == "done"
    assert _frame(second, "message.complete")["text"] == "You meant reading B."
    assert "The user chose: reading B" in agent_read(_model(app).requests[-1])
    (record,) = app.state.studio.store.current(PRINCIPAL, CONVERSATION).turns
    assert record["user"] == "ambiguous q\n(then: reading B)"


async def test_the_planner_s_ask_elicits_and_its_answer_resumes_the_planner() -> None:
    """The planner's own ask, through a real `PlannerAdapter`, pauses the turn
    on its question. Studio's answer runs the planner again with the pick in
    its brief, the resumed stream asks nothing again, and the turn ends."""
    asked = {
        "message": "Which source did you mean?",
        "candidates": [
            {"spec_id": "spec_a", "label": "reading A"},
            {"spec_id": "spec_b", "label": "reading B"},
        ],
    }
    llm = ScriptedGenai(
        [reply(call_part(PLANNER_ASK, asked, id="call-1")), reply(text_part("done"))]
    )
    adapter = PlannerAdapter(
        settings=Settings(), surface=FixtureSurface(schemas=[]), client=llm.client
    )
    app = _app(*_plan(), _text("No plan came of it."), planner=adapter)
    async with _client(app) as client:
        first = await _stream(client, "/api/studio/turn", _turn_body("q with forks"))

        assert _events(first)[-1] == "elicitation.request"
        elicitation = _frame(first, "elicitation.request")
        (question,) = elicitation["questions"]
        assert question["question"] == asked["message"]
        assert [option["id"] for option in question["options"]] == ["spec_a", "spec_b"]

        second = await _stream(
            client,
            "/api/studio/interrupt",
            _interrupt_body(elicitation["interrupt_id"], "answer", answers={"q1": _pick("spec_b")}),
        )

    assert "elicitation.request" not in _events(second)
    assert _events(second)[-1] == "done"
    assert len(llm.requests) == 2
    brief = "".join(
        part.text or "" for content in llm.requests[-1].content_list for part in content.parts or []
    )
    assert "They picked 'reading B', id `spec_b`." in brief


async def test_a_search_after_the_answer_reads_the_message_the_answer_refined() -> None:
    """Each segment's searches send the turn's message as it then stands: the
    question first, and after the resume the question with the answer."""
    search = CannedSearch(definite_envelope(spec_id=SPEC_A), definite_envelope(spec_id=SPEC_A))
    app = _app(
        search_call("aid"),
        *_ask("reading A", "reading B"),
        search_call("aid"),
        _text("You meant reading B."),
        resolver=search,
    )
    async with _client(app) as client:
        first = await _stream(client, "/api/studio/turn", _turn_body("ambiguous q"))
        elicitation = _frame(first, "elicitation.request")
        (question,) = elicitation["questions"]
        second = await _stream(
            client,
            "/api/studio/interrupt",
            _interrupt_body(
                elicitation["interrupt_id"],
                "answer",
                answers={"q1": _pick(question["options"][1]["id"])},
            ),
        )

    assert _events(second)[-1] == "done"
    assert [call["context"].user_message for call in search.calls] == [
        "ambiguous q",
        "ambiguous q\n(then: reading B)",
    ]


async def test_several_questions_resume_with_every_answer_in_one_reply() -> None:
    """Two questions share one pause. Each may be left to the agent, the pause
    as a whole may not be dismissed, and one reply answers both. The turn's
    record keeps each answer under its question, in words."""
    app = _app(
        *_ask_questions(
            ("How should the aid be counted?", ("Net", "Gross")),
            ("Which prices?", ("Constant", "Current")),
        ),
        _text("Gross it is."),
    )
    async with _client(app) as client:
        first = await _stream(client, "/api/studio/turn", _turn_body("ambiguous q"))
        elicitation = _frame(first, "elicitation.request")
        assert elicitation.get("skippable", False) is False
        assert [
            (q["id"], q["question"], q["skippable"], [o["label"] for o in q["options"]])
            for q in elicitation["questions"]
        ] == [
            ("q1", "How should the aid be counted?", True, ["Net", "Gross"]),
            ("q2", "Which prices?", True, ["Constant", "Current"]),
        ]
        gross = elicitation["questions"][0]["options"][1]["id"]

        second = await _stream(
            client,
            "/api/studio/interrupt",
            _interrupt_body(
                elicitation["interrupt_id"],
                "answer",
                answers={"q1": _pick(gross), "q2": {"kind": "skipped"}},
            ),
        )

    assert _events(second)[-1] == "done"
    asked = agent_read(_model(app).requests[-1])
    assert "1. How should the aid be counted? The user chose: Gross" in asked
    assert "2. Which prices? The user left this to you. Go on with Constant." in asked
    (record,) = app.state.studio.store.current(PRINCIPAL, CONVERSATION).turns
    assert record["user"] == "ambiguous q\n(then: How should the aid be counted? Gross)"


async def test_the_turn_record_keeps_a_picked_count_under_its_spec_id() -> None:
    """A pick whose option carries a spec_id is recorded as `label (spec_id)`,
    so a later turn can tell which series the user settled."""
    app = _app(
        search_call("aid to Kenya"),
        call_part(
            "ask_user",
            {
                "questions": [
                    {
                        "question": "How should the aid be counted?",
                        "options": [{"label": "Net", "spec_id": SPEC_A}, {"label": "Gross"}],
                    }
                ]
            },
        ),
        _text("Net it is."),
        resolver=_search(),
    )
    async with _client(app) as client:
        first = await _stream(client, "/api/studio/turn", _turn_body("ambiguous q"))
        elicitation = _frame(first, "elicitation.request")
        (question,) = elicitation["questions"]
        assert question["options"][0]["id"] == SPEC_A

        await _stream(
            client,
            "/api/studio/interrupt",
            _interrupt_body(elicitation["interrupt_id"], "answer", answers={"q1": _pick(SPEC_A)}),
        )

    (record,) = app.state.studio.store.current(PRINCIPAL, CONVERSATION).turns
    assert record["user"] == f"ambiguous q\n(then: Net ({SPEC_A}))"


async def test_stop_during_a_pause_invalidates_the_ask_it_left_behind() -> None:
    """POST /cancel while the turn is paused on an elicitation: by then the
    stream that emitted the pause has already ended, so no live segment can
    take the Stop. The route must still refuse a later `/interrupt` on that
    id, and must not resume the turn. The test proves it by checking the agent's model is not asked again."""
    app = _app(*_ask("reading A", "reading B"))
    async with _client(app) as client:
        first = await _stream(client, "/api/studio/turn", _turn_body("ambiguous q"))
        elicitation = _frame(first, "elicitation.request")
        asked = len(_model(app).requests)

        # The pause already ended the stream, same as production: a Stop
        # press only ever reaches this route after the SSE connection that
        # showed the ask is long closed.
        cancel = await client.post(
            "/api/studio/cancel",
            json={"kind": "turn.cancel", "schema_version": 1, "conversation_id": CONVERSATION},
            headers=AUTH,
        )
        assert cancel.status_code == 204

        replay = await client.post(
            "/api/studio/interrupt",
            json=_interrupt_body(
                elicitation["interrupt_id"],
                "answer",
                answers={"q1": _pick(elicitation["questions"][0]["options"][1]["id"])},
            ),
            headers=AUTH,
        )

    assert replay.status_code == 404
    assert replay.json()["error"] == "not_found"
    assert len(_model(app).requests) == asked  # the resumed turn never ran


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
    store = StudioStore(
        _engine(),
    )
    app = _app(*_ask(), *_ask(), store=store)

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

        conversation = store.current(PRINCIPAL, CONVERSATION)
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
    async with _client(_app()) as client:
        await _stream(client, "/api/studio/turn", _turn_body())
        response = await client.post(
            "/api/studio/interrupt",
            json=_interrupt_body("int_nonexistent", "approve"),
            headers=AUTH,
        )
    assert response.status_code == 404
    assert response.json()["error"] == "not_found"


async def test_an_unknown_conversation_is_a_404_on_interrupt() -> None:
    async with _client(_app()) as client:
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
    async with _client(_app()) as client:
        body = _turn_body()
        body["component_ref"] = {"intent": "follow_up", "component_id": "cmp_ghost", "revision": 1}
        response = await client.post("/api/studio/turn", json=body, headers=AUTH)
    assert response.status_code == 404
    payload = response.json()
    assert payload["error"] == "not_found"
    assert payload["pointer"] == "/conversation_id"


async def test_a_follow_up_on_a_card_the_conversation_never_held_is_a_404() -> None:
    """A `follow_up` naming a card this conversation never stored would run as
    a plain turn with no card in view; refused, so the client sees it."""
    async with _client(_app()) as client:
        await _stream(client, "/api/studio/turn", _turn_body())
        body = _turn_body("tell me more about this")
        body["component_ref"] = {"intent": "follow_up", "component_id": "cmp_ghost", "revision": 1}
        response = await client.post("/api/studio/turn", json=body, headers=AUTH)
    assert response.status_code == 404
    payload = response.json()
    assert payload["error"] == "not_found"
    assert payload["pointer"] == "/component_ref/component_id"


async def test_a_follow_up_on_a_stored_card_runs_the_turn() -> None:
    app = _app(*_card(), resolver=_search())
    async with _client(app) as client:
        await _conversation_with_component(client)
        body = _turn_body("tell me more about this")
        body["component_ref"] = {
            "intent": "follow_up",
            "component_id": COMPONENT["component_id"],
            "revision": 1,
        }
        frames = await _stream(client, "/api/studio/turn", body)
    assert _events(frames)[-1] == "done"


async def test_a_wire_interrupt_id_is_single_use() -> None:
    async with _client(_app(*_plan())) as client:
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
    async with _client(_app(*_plan())) as client:
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


async def _never_closed(*args: Any, **kwargs: Any) -> SegmentClose:
    """Stands in for `StudioStore.close_segment`: a crashed instance's
    segment lease and resuming pending entries are never cleared or dropped
    -- exactly what a real crash leaves behind, since nothing runs a
    process's `finally` blocks when it dies."""
    return SegmentClose.FAILED


async def _never_written(*args: Any, **kwargs: Any) -> Any:
    """Stands in for `TurnStore.finish`: a crashed instance never closes
    its turn, so the turn stays as its last write left it. Raised as the
    `SegmentLost` a write under a lost lease is refused with."""
    raise SegmentLost("the instance died before this write")


async def _crash_leaves_no_stop() -> None:
    """The Stop `_dies_on_a_stop` dies on only triggers the crash. Cleared,
    it leaves the crash alone for a retry to meet; left live, the next
    claimant would close the turn on it."""
    async with _engine().begin() as conn:
        await conn.execute(
            text("UPDATE conversations SET stop_id = NULL, stop_turn = NULL, lease_cancel = false")
        )


def _dies_on_a_stop(app: FastAPI) -> None:
    """Stand `app`'s instance in for one that dies as a Stop reaches it:
    neither its segment's closing write nor its turn's close is written
    (`_never_closed`, `_never_written`)."""
    app.state.studio.store.close_segment = _never_closed
    app.state.studio.turns.finish = _never_written


async def _approve_plan_then_hang(
    client: httpx2.AsyncClient, app: FastAPI, plan: dict[str, Any]
) -> _HangingSurface:
    """Start `/interrupt approve`'s resume stream, let it reach its first
    (scripted) plan step, and abandon it there: `POST /cancel` ends the
    stream without ever persisting the segment's closing write or the turn's
    close (`_dies_on_a_stop`), standing in for a crash between
    `/interrupt`'s save and the resumed segment's own checkpoint. Returns once the abandoned stream's
    `error` frame has been read off the wire."""
    hanging = _HangingSurface()
    app.state.studio.surface_recorder._surface = hanging
    interrupt_task = asyncio.ensure_future(
        _stream(client, "/api/studio/interrupt", _interrupt_body(plan["interrupt_id"], "approve"))
    )
    await asyncio.wait_for(hanging.entered.wait(), timeout=5)
    _dies_on_a_stop(app)
    cancel = await client.post(
        "/api/studio/cancel",
        json={"kind": "turn.cancel", "schema_version": 1, "conversation_id": CONVERSATION},
        headers=AUTH,
    )
    assert cancel.status_code == 204
    crashed = await asyncio.wait_for(interrupt_task, timeout=5)
    assert _events(crashed)[-1] == "error"
    await _crash_leaves_no_stop()
    return hanging


@pytest.mark.parametrize(
    ("moved", "streams"),
    [
        pytest.param("version = version + 1", True, id="same-pause"),
        pytest.param("interrupt_id = 'int_elsewhere'", False, id="pause-gone"),
    ],
)
async def test_an_answer_runs_on_the_cursor_read_after_its_claim(moved: str, streams: bool) -> None:
    """A write lands on the turn between `/interrupt`'s first read of the
    cursor and its claim save. When the turn still holds the same open
    pause, the answer resumes it from the cursor read after the claim;
    when it does not, the answer is refused with a 409 to reload and answer
    again, and nothing of it is kept: no claim, no approval, no entry marked
    resuming."""
    app = _app(*_plan(), tool_surface=ScriptedSurface())
    store = app.state.studio.store
    real_save = store.save
    moved_once = False

    async def save_after_a_write(*args: Any, **kwargs: Any) -> Any:
        nonlocal moved_once
        if not moved_once:
            moved_once = True
            async with _engine().begin() as conn:
                await conn.execute(text(f"UPDATE turn_cursors SET {moved}"))
        return await real_save(*args, **kwargs)

    async with _client(app) as client:
        plan = _frame(
            await _stream(client, "/api/studio/turn", _turn_body("q with forks")), "plan.proposed"
        )
        store.save = save_after_a_write
        body = _interrupt_body(plan["interrupt_id"], "approve")
        if streams:
            assert _events(await _stream(client, "/api/studio/interrupt", body))[-1] == "done"
            return
        refused = await client.post("/api/studio/interrupt", json=body, headers=AUTH)

    assert refused.status_code == 409
    assert refused.json()["retry"] == "refresh_then_retry"
    stored = await StudioStore(_engine()).load(PRINCIPAL, CONVERSATION)
    assert stored is not None and stored.segment is None
    assert stored.approvals == []
    assert stored.pending[plan["interrupt_id"]].resuming_segment is None


async def test_an_answer_whose_save_finds_the_conversation_moved_is_refused_to_reload() -> None:
    """The answer's own save, after the claim's, finds the conversation
    moved (only a takeover after the claim lapsed can do that): the answer
    is refused with a 409 to reload, and nothing of it is kept."""
    app = _app(*_plan(), tool_surface=ScriptedSurface())
    store = app.state.studio.store
    real_save = store.save
    saves = 0

    async def second_save_stale(*args: Any, **kwargs: Any) -> Any:
        nonlocal saves
        saves += 1
        if saves == 2:
            raise ConversationStale("moved")
        return await real_save(*args, **kwargs)

    async with _client(app) as client:
        plan = _frame(
            await _stream(client, "/api/studio/turn", _turn_body("q with forks")), "plan.proposed"
        )
        store.save = second_save_stale
        refused = await client.post(
            "/api/studio/interrupt",
            json=_interrupt_body(plan["interrupt_id"], "approve"),
            headers=AUTH,
        )

    assert refused.status_code == 409
    assert (refused.json()["error"], refused.json()["retry"]) == (
        "edit_conflict",
        "refresh_then_retry",
    )
    stored = await StudioStore(_engine()).load(PRINCIPAL, CONVERSATION)
    assert stored is not None and stored.segment is None
    assert stored.approvals == []
    assert stored.pending[plan["interrupt_id"]].resuming_segment is None


async def test_a_crash_between_interrupts_save_and_the_resume_leaves_a_retryable_answer() -> None:
    """A's resume of a plan approval crashes (`_approve_plan_then_hang`)
    strictly after `/interrupt`'s own save landed -- the segment claimed,
    the entry marked resuming, the one `ApprovalRecord` appended -- but
    before the resumed segment's first plan step, let alone its own
    checkpoint, completes. B's same answer is refused while A's segment
    lease is still live, and resumes the turn once it expires (an injected
    clock), without a second `ApprovalRecord` and past every `seq` A's dead
    attempt may already have sent."""
    clock = await StoreClock.start(_engine())
    store_a = StudioStore(
        _engine(),
        **clock.store_kwargs,
    )
    store_b = StudioStore(
        _engine(),
        **clock.store_kwargs,
    )
    app_a = _app(
        *_plan(),
        tool_surface=ScriptedSurface(),
        store=store_a,
    )

    async with _client(app_a) as client_a:
        first = await _stream(client_a, "/api/studio/turn", _turn_body("q with forks"))
        plan = _frame(first, "plan.proposed")

        await _approve_plan_then_hang(client_a, app_a, plan)

    app_b = _app(tool_surface=ScriptedSurface(), store=store_b)
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

        await clock.advance(studio_state_module.SEGMENT_TTL_S + 1)  # A's segment has expired

        resumed = await _stream(
            client_b, "/api/studio/interrupt", _interrupt_body(plan["interrupt_id"], "approve")
        )

    assert _events(resumed)[-1] == "done"
    # The turn's first segment numbered from 0 and A's dead resume from the
    # block after it, so B's resume starts past both blocks.
    assert resumed[0]["seq"] >= 2 * SEQ_BLOCK
    # A's dead attempt sent the approved plan once the approve node took it.
    assert "plan.approved" not in _events(resumed)

    reloaded = await StudioStore(
        _engine(),
    ).load(PRINCIPAL, CONVERSATION)
    assert reloaded is not None
    assert len(reloaded.approvals) == 1
    assert plan["interrupt_id"] not in reloaded.pending


async def test_a_retry_after_an_attempt_that_died_before_the_approval_sends_it() -> None:
    """A's resume of a plan approval dies after `/interrupt`'s own save but before
    the graph takes the approval, so it never sent `plan.approved`. B's retry of the
    same answer, once A's segment expires, sends the approved plan once."""
    clock = await StoreClock.start(_engine())
    store_a = StudioStore(
        _engine(),
        **clock.store_kwargs,
    )
    store_b = StudioStore(
        _engine(),
        **clock.store_kwargs,
    )
    app_a = _app(*_plan(), store=store_a)
    entered = asyncio.Event()

    async def _dead(*_args: Any, **_kwargs: Any) -> Any:
        entered.set()
        await asyncio.Event().wait()

    async with _client(app_a) as client_a:
        first = await _stream(client_a, "/api/studio/turn", _turn_body("q with forks"))
        plan = _frame(first, "plan.proposed")
        # The turn never takes the answer: its write hangs.
        app_a.state.studio.turns.record_answer = _dead
        attempt = asyncio.ensure_future(
            _stream(
                client_a, "/api/studio/interrupt", _interrupt_body(plan["interrupt_id"], "approve")
            )
        )
        await asyncio.wait_for(entered.wait(), timeout=5)
        _dies_on_a_stop(app_a)
        cancel = await client_a.post(
            "/api/studio/cancel",
            json={"kind": "turn.cancel", "schema_version": 1, "conversation_id": CONVERSATION},
            headers=AUTH,
        )
        assert cancel.status_code == 204
        dead = await asyncio.wait_for(attempt, timeout=5)
        assert "plan.approved" not in _events(dead)
        await _crash_leaves_no_stop()

    await clock.advance(studio_state_module.SEGMENT_TTL_S + 1)  # A's segment has expired
    app_b = _app(tool_surface=ScriptedSurface(), store=store_b)
    async with _client(app_b) as client_b:
        resumed = await _stream(
            client_b, "/api/studio/interrupt", _interrupt_body(plan["interrupt_id"], "approve")
        )

    assert _events(resumed)[-1] == "done"
    assert _events(resumed).count("plan.approved") == 1
    assert _frame(resumed, "plan.approved")["revision"] == plan["revision"]


async def test_a_cancel_on_a_lapsed_lease_fences_out_the_stream_still_running_under_it() -> None:
    """A's stream outlives its lease without renewing it. A Stop on B finds no
    live segment, so it closes the conversation itself. A's next checkpoint
    is refused, so it cannot write over what the Stop saved."""
    clock = await StoreClock.start(_engine())
    store_a = StudioStore(
        _engine(),
        **clock.store_kwargs,
    )
    store_b = StudioStore(
        _engine(),
        **clock.store_kwargs,
    )
    state_a = await store_a.create(PRINCIPAL, CONVERSATION)
    segment_id = store_a.claim_segment(PRINCIPAL, CONVERSATION)
    state_a.begin_turn()
    await store_a.save(PRINCIPAL, CONVERSATION)
    await clock.advance(studio_state_module.SEGMENT_TTL_S + 1)  # A's lease has lapsed

    app_b = _app(tool_surface=ScriptedSurface(), store=store_b)
    async with _client(app_b) as client_b:
        cancel = await client_b.post(
            "/api/studio/cancel",
            json={"kind": "turn.cancel", "schema_version": 1, "conversation_id": CONVERSATION},
            headers=AUTH,
        )
    assert cancel.status_code == 204

    state_a.turn_final_text = "written after the Stop"
    with pytest.raises(SegmentLost):
        await store_a.checkpoint(
            PRINCIPAL, CONVERSATION, segment_id, store_a.snapshot(PRINCIPAL, CONVERSATION)
        )
    reloaded = await StudioStore(
        _engine(),
    ).load(PRINCIPAL, CONVERSATION)
    assert reloaded is not None
    assert reloaded.segment is None
    assert reloaded.turn_final_text == ""


def _reading(checkpoint: int) -> dict[str, Any]:
    """A search reading the turn's checkpoint `checkpoint` holds."""
    return SearchReading(text=f"aid, checkpoint {checkpoint}", status="definite").model_dump(
        mode="json"
    )


@pytest.mark.parametrize("lapsed", ["finishes_the_turn", "writes_continuously"])
async def test_a_turn_claimed_over_a_lapsed_segment_records_its_last_write(
    monkeypatch: pytest.MonkeyPatch, lapsed: str
) -> None:
    """A's turn is left running under a lease that lapsed, and A goes on
    writing while B's `/turn` runs. Either A finishes the turn `done` after
    B's load, its record unwritten, or A writes the turn's cursor and the
    conversation's record over and over until B's claim fences it out. B
    streams, and the turn A left is recorded from A's last write: the
    finished turn with the readings, plans and specs its cursor holds, the
    turn left running abandoned on A's last cursor and record."""
    _, record = _top_donors()
    plan = record.model_dump(mode="json")
    used = record.sources[0].spec_id

    def turn_state(checkpoint: int) -> dict[str, Any]:
        return {
            ORCHESTRATOR_SEARCHES_KEY: [_reading(checkpoint)],
            ORCHESTRATOR_PLAN_RECORDS_KEY: [plan],
            ORCHESTRATOR_SPECS_KEY: {used: {"spec_id": used}},
        }

    clock = await StoreClock.start(_engine())
    store_a = StudioStore(_engine(), **clock.store_kwargs)
    turns = turn_store_module.TurnStore(_engine())
    state_a = await store_a.create(PRINCIPAL, CONVERSATION)
    segment_a = store_a.claim_segment(PRINCIPAL, CONVERSATION)
    state_a.begin_turn()
    state_a.turn_user_message = "aid to Kenya"
    await store_a.save(PRINCIPAL, CONVERSATION)
    left = await turns.open_turn(
        PRINCIPAL,
        CONVERSATION,
        segment_id=segment_a,
        turn_index=1,
        node="plan",
        inputs={},
        state=turn_state(1),
    )
    await clock.advance(studio_state_module.SEGMENT_TTL_S + 1)  # A's lease has lapsed

    store_b = StudioStore(_engine(), **clock.store_kwargs)
    app_b = _app(_text("Hello."), store=store_b)
    # A's last committed cursor checkpoint and record text.
    last = {"checkpoint": 1, "text": ""}
    writing: asyncio.Task[None] | None = None

    async def write_until_fenced() -> None:
        cursor = left
        for checkpoint in itertools.count(2):
            try:
                cursor = await turns.advance(
                    cursor,
                    segment_id=segment_a,
                    node="plan",
                    node_input=None,
                    state=turn_state(checkpoint),
                )
                last["checkpoint"] = checkpoint
                state_a.turn_final_text = f"written by A at {checkpoint}"
                await store_a.save(PRINCIPAL, CONVERSATION)
                last["text"] = state_a.turn_final_text
            except (SegmentLost, ConversationStale, CursorStale):
                return

    load = store_b.load

    async def load_then_write(principal: str, conversation_id: str) -> Any:
        nonlocal writing
        loaded = await load(principal, conversation_id)
        if writing is None:
            if lapsed == "finishes_the_turn":
                await turns.finish(
                    left, segment_id=segment_a, status=TurnStatus.DONE, state=turn_state(1)
                )
                writing = asyncio.get_running_loop().create_task(asyncio.sleep(0))
            else:
                writing = asyncio.get_running_loop().create_task(write_until_fenced())
                while last["checkpoint"] < 3:
                    await asyncio.sleep(0)
        return loaded

    monkeypatch.setattr(store_b, "load", load_then_write)
    async with _client(app_b) as client:
        frames = await _stream(client, "/api/studio/turn", _turn_body("hi"))
    assert writing is not None
    await asyncio.wait_for(writing, 5)

    assert _events(frames)[-1] == "done"
    stored = await turns.turn(left.id)
    conversation = await StudioStore(_engine()).load(PRINCIPAL, CONVERSATION)
    assert conversation is not None
    first = conversation.turns[0]
    assert first["readings"][0] == _reading(last["checkpoint"])
    assert [p["revision"] for p in first["plans"]] == [plan["revision"]]
    assert first["specs"] == {used: {"spec_id": used}}
    if lapsed == "finishes_the_turn":
        assert (stored.status, first["error"]) == (TurnStatus.DONE, "state_not_saved")
    else:
        assert (stored.status, first["error"]) == (TurnStatus.ABANDONED, "abandoned")
        assert first["assistant"] == last["text"]


class _CallingSurface(ScriptedSurface):
    """`ScriptedSurface`, recording the name of every tool it is called with."""

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[str] = []

    async def call_tool(
        self,
        *,
        tool: ToolId,
        arguments: Mapping[str, Any],
        on_progress: ProgressCallback | None = None,
    ) -> ToolCallResult:
        self.calls.append(tool.name)
        return await super().call_tool(tool=tool, arguments=arguments, on_progress=on_progress)


async def test_a_retried_approval_is_dated_by_its_first_answer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A's resume of a plan approval answered at T dies before its run is
    admitted. B's retry of the same answer a day and an hour later carries
    the plan approved as of T on, so its run is refused as expired and
    nothing is dispatched."""
    answered = datetime.now(UTC) - timedelta(hours=25)

    class _AnsweredThen(datetime):
        @classmethod
        def now(cls, tz: tzinfo | None = None) -> datetime:
            return answered.astimezone(tz)

    entered = asyncio.Event()

    async def _dies_before_admission(plan: PlannerPlan, **kwargs: Any) -> Any:
        entered.set()
        await asyncio.Event().wait()

    approved_at: list[datetime] = []
    approve = PlannerPlan.approve

    def _recording_approve(self: PlannerPlan, *, at: datetime) -> PlannerPlan:
        approved_at.append(at)
        return approve(self, at=at)

    monkeypatch.setattr(PlannerPlan, "approve", _recording_approve)
    clock = await StoreClock.start(_engine())
    store_a = StudioStore(
        _engine(),
        **clock.store_kwargs,
    )
    store_b = StudioStore(
        _engine(),
        **clock.store_kwargs,
    )
    app_a = _app(*_plan(), tool_surface=ScriptedSurface(), store=store_a)

    async with _client(app_a) as client_a:
        first = await _stream(client_a, "/api/studio/turn", _turn_body("q with forks"))
        plan = _frame(first, "plan.proposed")
        with monkeypatch.context() as patched:
            patched.setattr(studio_module, "datetime", _AnsweredThen)
            patched.setattr(execute_module, "execute_approved_plan", _dies_before_admission)
            resume = asyncio.ensure_future(
                _stream(
                    client_a,
                    "/api/studio/interrupt",
                    _interrupt_body(plan["interrupt_id"], "approve"),
                )
            )
            await asyncio.wait_for(entered.wait(), timeout=5)
            _dies_on_a_stop(app_a)
            cancel = await client_a.post(
                "/api/studio/cancel",
                json={"kind": "turn.cancel", "schema_version": 1, "conversation_id": CONVERSATION},
                headers=AUTH,
            )
            assert cancel.status_code == 204
            assert _events(await asyncio.wait_for(resume, timeout=5))[-1] == "error"
            await _crash_leaves_no_stop()

    surface = _CallingSurface()
    app_b = _app(tool_surface=surface, store=store_b)
    async with _client(app_b) as client_b:
        await clock.advance(studio_state_module.SEGMENT_TTL_S + 1)  # A's segment has expired
        await _stream(
            client_b, "/api/studio/interrupt", _interrupt_body(plan["interrupt_id"], "approve")
        )

    assert approved_at == [answered]
    assert surface.calls == []
    reloaded = await StudioStore(
        _engine(),
    ).load(PRINCIPAL, CONVERSATION)
    assert reloaded is not None
    assert [record.at for record in reloaded.approvals] == [answered]


async def test_a_different_answer_after_a_crashed_resume_expires_is_refused() -> None:
    """The same crash as above, but B's retry disagrees with the answer
    that actually resumed the turn: refused for good, even once A's
    segment has expired and a matching retry would have succeeded."""
    clock = await StoreClock.start(_engine())
    store_a = StudioStore(
        _engine(),
        **clock.store_kwargs,
    )
    store_b = StudioStore(
        _engine(),
        **clock.store_kwargs,
    )
    app_a = _app(
        *_plan(),
        tool_surface=ScriptedSurface(),
        store=store_a,
    )

    async with _client(app_a) as client_a:
        first = await _stream(client_a, "/api/studio/turn", _turn_body("q with forks"))
        plan = _frame(first, "plan.proposed")
        await _approve_plan_then_hang(client_a, app_a, plan)

    app_b = _app(
        tool_surface=ScriptedSurface(),
        store=store_b,
    )
    async with _client(app_b) as client_b:
        await clock.advance(studio_state_module.SEGMENT_TTL_S + 1)  # A's segment has expired

        mismatched = await client_b.post(
            "/api/studio/interrupt",
            json=_interrupt_body(plan["interrupt_id"], "reject"),
            headers=AUTH,
        )

    assert mismatched.status_code == 409
    body = mismatched.json()
    assert body["error"] == "edit_conflict"
    reloaded = await StudioStore(
        _engine(),
    ).load(PRINCIPAL, CONVERSATION)
    assert reloaded is not None
    assert plan["interrupt_id"] in reloaded.pending
    assert reloaded.pending[plan["interrupt_id"]].answer == {"decision": "approve", "payload": None}
    # B's claim, never saved, is gone from its live state.
    assert store_b.current(PRINCIPAL, CONVERSATION).segment == reloaded.segment


@pytest.mark.parametrize(
    ("retried", "status"),
    [pytest.param(0, 200, id="same-answers"), pytest.param(1, 409, id="other-answers")],
)
async def test_a_retried_resume_of_an_ask_must_carry_the_answers_that_resumed_it(
    retried: int, status: int
) -> None:
    """A's resume of an answered ask dies on its search, its segment lease
    left behind. Once the lease expires, B's retry resumes only with the
    answers A resumed with; other answers to the same question are refused
    for good."""
    clock = await StoreClock.start(_engine())
    store_a = StudioStore(
        _engine(),
        **clock.store_kwargs,
    )
    store_b = StudioStore(
        _engine(),
        **clock.store_kwargs,
    )
    resolver = _HangingResolver()
    app_a = _app(
        *_ask("reading A", "reading B"),
        *_card(),
        resolver=resolver,
        store=store_a,
    )

    async with _client(app_a) as client_a:
        first = await _stream(client_a, "/api/studio/turn", _turn_body("ambiguous q"))
        elicitation = _frame(first, "elicitation.request")
        option_ids = [option["id"] for option in elicitation["questions"][0]["options"]]
        resume = asyncio.ensure_future(
            _stream(
                client_a,
                "/api/studio/interrupt",
                _interrupt_body(
                    elicitation["interrupt_id"], "answer", answers={"q1": _pick(option_ids[0])}
                ),
            )
        )
        await asyncio.wait_for(resolver.entered.wait(), timeout=5)
        _dies_on_a_stop(app_a)
        cancel = await client_a.post(
            "/api/studio/cancel",
            json={"kind": "turn.cancel", "schema_version": 1, "conversation_id": CONVERSATION},
            headers=AUTH,
        )
        assert cancel.status_code == 204
        await asyncio.wait_for(resume, timeout=5)
        await _crash_leaves_no_stop()

    app_b = _app(_text("Reading A."), store=store_b)
    async with _client(app_b) as client_b:
        await clock.advance(studio_state_module.SEGMENT_TTL_S + 1)  # A's segment has expired
        retry = await client_b.post(
            "/api/studio/interrupt",
            json=_interrupt_body(
                elicitation["interrupt_id"], "answer", answers={"q1": _pick(option_ids[retried])}
            ),
            headers=AUTH,
        )

    assert retry.status_code == status


class _GatedSurface(ScriptedSurface):
    """`ScriptedSurface`, but each `call_tool` waits for `release` first, so a
    resumed segment holds its lease until the test lets it go on."""

    def __init__(self) -> None:
        super().__init__()
        self.release = asyncio.Event()

    async def call_tool(self, **kwargs: Any) -> ToolCallResult:
        await self.release.wait()
        return await super().call_tool(**kwargs)


async def test_two_concurrent_first_answers_to_one_interrupt_give_one_stream_and_one_409() -> None:
    """A and B both answer the SAME fresh interrupt at once: whichever
    instance's claim save lands first resumes the turn; the other's save
    finds the ledger already moved and answers the existing 409, rather
    than each resuming its own segment of the turn. The winner's plan step
    waits on a gate, so its segment still holds the lease when the loser's
    answer comes back."""
    surface_a, surface_b = _GatedSurface(), _GatedSurface()
    app_a = _app(*_plan(), tool_surface=surface_a)
    app_b = _app(tool_surface=surface_b)

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
        done, _ = await asyncio.wait(
            {task_a, task_b}, timeout=10, return_when=asyncio.FIRST_COMPLETED
        )
        assert [task.result().status_code for task in done] == [409]
        surface_a.release.set()
        surface_b.release.set()
        responses = await asyncio.gather(task_a, task_b)

    statuses = sorted(response.status_code for response in responses)
    assert statuses == [200, 409]
    conflict, loser = next(
        (response, app)
        for response, app in zip(responses, (app_a, app_b), strict=True)
        if response.status_code == 409
    )
    assert conflict.json()["error"] == "transient_unavailable"
    # The losing instance forgets the answer it marked and never saved.
    lost = loser.state.studio.store.current(PRINCIPAL, CONVERSATION)
    assert lost.pending[plan["interrupt_id"]].answer is None


async def test_an_answer_refused_as_busy_leaves_no_unsaved_approval_behind() -> None:
    """Another instance streams on the conversation, so an approval's claim
    is refused as busy. The approval it recorded and never saved is
    dropped from the instance's live state."""
    app = _app(*_plan(), tool_surface=ScriptedSurface())
    store = app.state.studio.store
    other = StudioStore(
        _engine(),
    )

    async with _client(app) as client:
        first = await _stream(client, "/api/studio/turn", _turn_body("q with forks"))
        plan = _frame(first, "plan.proposed")
        await other.load(PRINCIPAL, CONVERSATION)
        other.claim_segment(PRINCIPAL, CONVERSATION)
        await other.save(PRINCIPAL, CONVERSATION)

        busy = await client.post(
            "/api/studio/interrupt",
            json=_interrupt_body(plan["interrupt_id"], "approve"),
            headers=AUTH,
        )

    assert busy.status_code == 409
    assert store.current(PRINCIPAL, CONVERSATION).approvals == []
    reloaded = await StudioStore(
        _engine(),
    ).load(PRINCIPAL, CONVERSATION)
    assert reloaded is not None and reloaded.approvals == []
    assert reloaded.pending[plan["interrupt_id"]].answer is None


async def test_a_turn_refused_as_busy_leaves_the_live_state_as_loaded(
    otlp_spans: InMemorySpanExporter, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Another instance streams on a conversation with a card, so a `/turn`
    is refused as busy. The live state keeps the canvas, trace id and turn
    the load read, and the span the refused turn opened ends as
    `not_started` at warning level and is exported."""
    app = _app(*_card(), resolver=_search())
    store = app.state.studio.store
    other = StudioStore(
        _engine(),
    )
    ended: list[tuple[str | None, str]] = []
    real_end = studio_state_module.end_turn_span

    def recording_end(handle: Any, **kwargs: Any) -> None:
        if handle is not None:
            ended.append((kwargs.get("error"), kwargs.get("level", "ERROR")))
        real_end(handle, **kwargs)

    flushes: list[int] = []

    async def flush() -> None:
        flushes.append(_turn_spans(otlp_spans))

    async with _client(app) as client:
        await _conversation_with_component(client)
        loaded = store.current(PRINCIPAL, CONVERSATION)
        before = (loaded.trace_id, loaded.turn_index, loaded.composition)
        await other.load(PRINCIPAL, CONVERSATION)
        other.claim_segment(PRINCIPAL, CONVERSATION)
        await other.save(PRINCIPAL, CONVERSATION)
        monkeypatch.setattr(studio_state_module, "end_turn_span", recording_end)
        monkeypatch.setattr(studio_stream_module, "end_turn_span", recording_end)
        monkeypatch.setattr(studio_module, "flush_telemetry", flush)
        monkeypatch.setattr(studio_stream_module, "flush_telemetry", flush)

        busy = await client.post("/api/studio/turn", json=_turn_body("again"), headers=AUTH)

    assert busy.status_code == 409
    live = store.current(PRINCIPAL, CONVERSATION)
    assert live.composition is not None
    assert (live.trace_id, live.turn_index, live.composition) == before
    reloaded = await other.load(PRINCIPAL, CONVERSATION)
    assert reloaded is not None
    assert (reloaded.trace_id, reloaded.turn_index) == before[:2]
    assert ended == [("not_started", "WARNING")]
    assert flushes == [2]  # the first turn's span, then the refused one's


async def test_an_answer_after_a_refused_turn_resumes_on_the_paused_turn_s_span(
    otlp_spans: InMemorySpanExporter,
) -> None:
    """A `/turn` on a paused conversation loses its claim save, so nothing
    is claimed. The answer that follows resumes the paused turn on its own
    span: the span the resumed segment ends is the one whose external id is
    the paused turn's trace id."""
    app = _app(*_plan(), tool_surface=ScriptedSurface())
    store = app.state.studio.store
    real_claim = store.claim_turn_lease
    refused: list[str] = []

    async def claimed_elsewhere(
        principal: str, conversation_id: str, *args: Any, **kwargs: Any
    ) -> None:
        refused.append(conversation_id)
        raise ConversationStale(conversation_id)

    async with _client(app) as client:
        first = await _stream(client, "/api/studio/turn", _turn_body("q with forks"))
        plan = _frame(first, "plan.proposed")
        store.claim_turn_lease = claimed_elsewhere  # type: ignore[method-assign]
        stale = await client.post("/api/studio/turn", json=_turn_body("other"), headers=AUTH)
        store.claim_turn_lease = real_claim  # type: ignore[method-assign]
        second = await _stream(
            client, "/api/studio/interrupt", _interrupt_body(plan["interrupt_id"], "reject")
        )

    assert (stale.status_code, refused) == (409, [CONVERSATION])
    trace_id = _frame(second, "done")["trace_id"]
    turns = [span for span in otlp_spans.get_finished_spans() if span.name == "turn"]
    assert (turns[-1].attributes or {})["dfo.external_id"] == trace_id


async def test_an_answer_whose_turn_holds_no_pause_is_refused_before_anything_is_saved(
    otlp_spans: InMemorySpanExporter,
) -> None:
    """No open turn is paused on the answer's interrupt, so `/interrupt`
    answers 404 before its claim save: no approval of the plan is recorded
    and the pending entry is not marked resuming, live or stored, so an
    answer with another decision is refused the same way, never as a
    conflict with the first. The lock is released, and the refused answers
    open no span: the one span ended is the paused segment's."""
    app = _app(*_plan(), tool_surface=ScriptedSurface())
    store = app.state.studio.store

    async with _client(app) as client:
        first = await _stream(client, "/api/studio/turn", _turn_body("q with forks"))
        plan = _frame(first, "plan.proposed")
        async with _engine().begin() as conn:
            await conn.execute(text("DELETE FROM turn_cursors"))

        responses = [
            await client.post(
                "/api/studio/interrupt",
                json=_interrupt_body(plan["interrupt_id"], *decision),
                headers=AUTH,
            )
            for decision in (("approve",), ("amend", "use commitments"))
        ]

    assert [(r.status_code, r.json()["error"]) for r in responses] == [(404, "not_found")] * 2
    assert not store.lock_for(PRINCIPAL, CONVERSATION).locked()
    stored = await StudioStore(_engine()).load(PRINCIPAL, CONVERSATION)
    for state in (store.current(PRINCIPAL, CONVERSATION), stored):
        assert state is not None and state.segment is None
        assert state.approvals == []
        pending = state.pending[plan["interrupt_id"]]
        assert (pending.answer, pending.resuming_segment) == (None, None)
    assert _turn_spans(otlp_spans) == 1


async def test_a_resumed_segments_stream_fatal_error_leaves_no_resuming_entry() -> None:
    """A resumed segment that ends in a stream-fatal error (not a crash)
    drops its resuming entry through `release`'s own `drop_resuming`, the
    same as a clean `done`, so a later retry of the same answer is refused
    the same way an unknown interrupt id is."""

    class _BoomExecutor(ScriptedExecutor):
        async def __call__(
            self, *, tool: str, args: dict[str, Any], on_progress: ProgressCallback | None = None
        ) -> dict[str, Any]:
            raise RuntimeError("describe_frame call failed")

    # The resumed segment's hand-back describes the plan's frame through the
    # executor, which raises.
    app = _app(*_plan(), executor=_BoomExecutor(), tool_surface=ScriptedSurface())
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
    app = _app(*_card(), resolver=_search())
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
    app = _app(*_card(), resolver=_search())
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
    interleave. One call succeeds; the other's `expected_revision`
    is stale, so it returns `edit_conflict` rather than a second silent 200
    that drops one edit."""
    app = _app(*_card(), resolver=_search())
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


async def test_a_composition_edit_leaves_the_instance_s_live_state_alone() -> None:
    """`/composition` reads the stored canvas and digest entries in its own
    transaction. It never reloads the live state this instance keeps, which
    a `/turn` on the same instance may be claiming over, so a field the live
    state holds and has not saved survives the edit."""
    app = _app(*_card(), resolver=_search())
    store = app.state.studio.store
    async with _client(app) as client:
        manifest = await _conversation_with_component(client)
        live = store.current(PRINCIPAL, CONVERSATION)
        live.turn_user_message = "claimed, not yet saved"
        response = await client.post(
            "/api/studio/composition",
            json=_mutation_body(manifest, [{"op": "pin", "component_id": "cmp_01test0001"}]),
            headers=AUTH,
        )

    assert response.status_code == 200, response.text
    assert store.current(PRINCIPAL, CONVERSATION) is live
    assert live.turn_user_message == "claimed, not yet saved"
    assert live.composition.blocks[0].pinned is True


async def test_an_unknown_composition_id_is_a_404_tool_error() -> None:
    """A known conversation with no such composition -- distinct from
    `test_an_unknown_conversation_is_a_404_on_composition` below, which the
    store never held at all."""
    async with _client(_app()) as client:
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
    async with _client(_app()) as client:
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


def _fatal(conversation: ConversationState, exc: Exception) -> dict[str, Any]:
    """The wire `error` of the stream-fatal frame `exc` ends a fresh turn of
    `conversation` with."""
    conversation.begin_turn()
    frame = _stream_fatal(conversation, exc)
    return frame.model_dump(mode="json", exclude_none=True)["error"]


@pytest.mark.parametrize(
    ("tool", "arguments", "retry"),
    [
        ("search", {"question": "kenya oda 2023"}, "do_not_retry"),
        ("search", {"question": "mali oda 2024"}, "retry_after_delay"),
        ("get_data", {"spec_id": SPEC_A}, "retry_after_delay"),
    ],
    ids=["same-call", "different-arguments", "different-tool"],
)
def test_a_timeout_is_advised_the_wait_unless_the_same_call_timed_out_before(
    tool: str, arguments: dict[str, Any], retry: str
) -> None:
    """The first timeout on a call is transient and advises the pinned wait.
    The conversation remembers each timed-out call by its tool and its own
    arguments: the same call timing out again says the wait bought nothing,
    while a different search, or another tool, is a first occurrence: evidence
    the service is slow, and still worth the wait."""
    conversation = ConversationState(conversation_id=CONVERSATION)

    first = _fatal(conversation, _timeout_error("search", {"question": "kenya oda 2023"}))
    again = _fatal(conversation, _timeout_error(tool, arguments))

    assert first["error"] == "transient_unavailable"
    assert first["retry"] == "retry_after_delay"
    assert first["retry_after_seconds"] == 30
    assert again["error"] == "transient_unavailable"
    assert again["retry"] == retry
    assert again.get("retry_after_seconds") == (30 if retry == "retry_after_delay" else None)
    # The reply is the transient sentence for the frame's retry mode. It names
    # no tool and no trace id.
    for error in (first, again):
        assert error["detail"] == classed_sentence("transient_unavailable", error["retry"])
        assert is_user_sentence(error["detail"])
        assert conversation.trace_id not in error["detail"]
    assert (again["detail"] == first["detail"]) is (retry == "retry_after_delay")


def test_a_surface_changed_refusal_reaches_the_wire_as_do_not_retry() -> None:
    """`executor.outcomes.RefusalKind.SURFACE_CHANGED` -- a plan step whose
    tool has changed since the plan was compiled -- has no member of its own
    in `harness.wire.ErrorCode`. `_stream_fatal` must not fall back to
    `transient_unavailable`/`retry_after_delay` for it: retrying a stale
    compiled plan can never clear the mismatch, so the wire frame has to
    carry `capability_unavailable`/`do_not_retry` and no `retry_after_seconds`,
    honouring the refusal's own declared `retry_mode` instead of discarding
    it."""
    refused = ResolverError(
        "surface changed",
        refusal={
            "code": "surface_changed",
            "retry_mode": "do_not_retry",
            "message": "a plan step's tool has changed since the plan was compiled",
        },
    )

    error = _fatal(ConversationState(conversation_id=CONVERSATION), refused)

    assert error["error"] == "capability_unavailable"
    assert error["retry"] == "do_not_retry"
    assert "retry_after_seconds" not in error


async def test_a_write_component_timeout_is_never_advised_a_retry() -> None:
    """The write was sent, so the platform may have stored the component, and a
    retry can store a second draft. An Edit Card write is a call no agent
    tool answers, so its timeout ends the stream."""

    class WriteTimeoutExecutor(ScriptedExecutor):
        async def __call__(
            self, *, tool: str, args: dict[str, Any], on_progress: ProgressCallback | None = None
        ) -> dict[str, Any]:
            if tool == "write_component" and "component_id" in args:
                self.calls.append({"tool": tool, "args": args})
                raise _timeout_error(tool, args)
            return await super().__call__(tool=tool, args=args)

    executor = WriteTimeoutExecutor([*_card_results(write_token=WRITE_TOKEN), _described(), _VIEW])
    edit_model = ScriptedEditModel(
        {"text": "Renamed the card.", "patch": {"title": "Aid to Kenya, 2018"}, "recreate": None}
    )
    app = _app(*_card(), resolver=_search(), executor=executor, edit_model=edit_model)
    async with _client(app) as client:
        await _conversation_with_component(client)
        frames = await _stream(client, "/api/studio/turn", _edit_body("rename it"))

    assert executor.calls[-1]["tool"] == "write_component"
    error = _frame(frames, "error")["error"]
    assert error["error"] == "transient_unavailable"
    assert error["retry"] == "do_not_retry"
    assert error.get("retry_after_seconds") is None
    assert error["detail"] == classed_sentence(HarnessCode.WRITE_UNCONFIRMED, "do_not_retry")
    assert error["detail"] != classed_sentence("transient_unavailable", "do_not_retry")
    assert is_user_sentence(error["detail"])


# --- cancel (delta 14) ----------------------------------------------------------


async def test_cancel_aborts_a_streaming_turn_with_a_cancelled_error_frame() -> None:
    """POST /api/studio/cancel while a turn streams: the stream ends with a
    stream-fatal `cancelled` error frame and no `done` follows."""

    app = _app(*_card(), resolver=_HangingResolver())
    studio = app.state.studio

    async def _armed() -> None:
        while not any(
            entry.state is not None and entry.state.renew_now is not None
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
    async with _client(_app()) as client:
        response = await client.post(
            "/api/studio/cancel",
            json={"kind": "turn.cancel", "schema_version": 1, "conversation_id": CONVERSATION},
            headers=AUTH,
        )
    assert response.status_code == 204


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
            *_card_results(write_token=WRITE_TOKEN),
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
        *_card(),
        resolver=_search(),
        executor=executor,
        edit_model=edit_model,
    )
    async with _client(app) as client:
        await _conversation_with_component(client)
        conversation = app.state.studio.store.current(PRINCIPAL, CONVERSATION)
        drawn = [card["component_id"] for card in conversation.canvas_cards()]
        frames = await _stream(client, "/api/studio/turn", _edit_body("rename it"))

    assert _events(frames)[-1] == "done"
    # A revised card no longer draws the intent make_charts wrote it from.
    assert (drawn, conversation.canvas_cards()) == (["cmp_01test0001"], [])
    revised = _frame(frames, "component.revised")
    assert revised["component_id"] == "cmp_01test0001"
    assert revised["revision"] == 2
    assert revised["changed"] == ["title"]
    manifest = _frame(frames, "done")["composition"]
    assert manifest["blocks"][0]["revision"] == 2
    assert manifest["blocks"][0]["title"] == "Aid to Kenya, 2018-2024"
    assert edit_model.calls[0]["stored_intent"] == STORED_INTENT
    # the edit turn's exact call sequence and the write's exact arguments
    assert [c["tool"] for c in executor.calls[3:]] == [
        "describe_component",
        "describe_frame",
        "write_component",
    ]
    assert executor.calls[-1]["args"] == {
        "intent": {"title": "Aid to Kenya, 2018-2024"},
        "component_id": "cmp_01test0001",
        "write_token": WRITE_TOKEN,
        CARD_FRAME_KEY: STORED_INTENT["frame_ref"],
    }


async def _edit_without_a_write_token(
    patch: dict[str, Any],
    *,
    stored_intent: dict[str, Any] = STORED_INTENT,
    removed: bool = False,
) -> tuple[list[dict[str, Any]], ScriptedExecutor]:
    """An Edit Card patch on a card this conversation holds no write
    token for: its create returned none, as a fork, a gallery card or a
    card made before tokens would. `removed` takes the card off the canvas
    before the edit."""
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
            *_card_results(),
            _described(stored_intent=stored_intent),
            _VIEW,
            {"outcome": "ok", "frame_refs": [], "component": recreated},
        ],
        strict=True,
    )
    edit_model = ScriptedEditModel({"text": "Renamed the card.", "patch": patch, "recreate": None})
    app = _app(
        *_card(),
        resolver=_search(),
        executor=executor,
        edit_model=edit_model,
    )
    async with _client(app) as client:
        manifest = await _conversation_with_component(client)
        if removed:
            response = await client.post(
                "/api/studio/composition",
                json=_mutation_body(manifest, [{"op": "remove", "component_id": "cmp_01test0001"}]),
                headers=AUTH,
            )
            assert response.json()["blocks"] == []
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
    # level, as apis `merge_intent` merges a revise, so `select` is
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


async def test_a_card_rebuilt_after_it_left_the_canvas_is_not_put_back_on_it() -> None:
    frames, _ = await _edit_without_a_write_token(
        {"title": "Aid to Kenya, 2018-2024"}, removed=True
    )

    assert _frame(frames, "component.created")["component_id"] == "cmp_01test0002"
    assert "composition.updated" not in _events(frames)
    assert _frame(frames, "done")["composition"]["blocks"] == []


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
            *_card_results(),
            _described(),
            _VIEW,
        ],
        strict=True,
    )
    app = _app(
        *_card(),
        resolver=_search(),
        executor=executor,
        edit_model=edit_model,
    )
    async with _client(app) as client:
        await _conversation_with_component(client)
        frames = await _stream(client, "/api/studio/turn", _edit_body("make it a line"))

    # The closing message and the failed step give the sentence of the code.
    sentence = classed_sentence(HarnessCode.FIELD_CREATE_ONLY, "correct_and_retry")
    assert _frame(frames, "message.complete")["text"] == f"The edit was refused. {sentence}"
    step = next(f for f in frames if f["event"] == "activity.step" and f["step_id"] == "e2")
    assert (step["status"], step["detail"]) == ("failed", sentence)
    assert is_user_sentence(sentence)
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
        is_error: bool = False,
    ) -> SimpleNamespace:
        content = [SimpleNamespace(type="text", text=block) for block in text or []]
        result = SimpleNamespace(
            is_error=is_error, structured_content=body, content=content, meta=meta
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
                {
                    "code": "write_refused",
                    "retry_mode": "do_not_retry",
                    "message": "wrong token",
                    "fix": {},
                },
                is_error=True,
            )
        self.revisions[component_id] += 1
        return self._written(component_id, token=None)


async def test_a_write_token_reaches_the_revise_and_nothing_the_client_sees(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A studio create then an Edit Card revise, through the real
    `McpExecutor` over a platform that issues `cmw_` tokens: the revise
    presents the create's token, and no SSE frame, digest,
    turn record, executor record or state repr carries it. The ledger
    holds it apart from the component digest entries."""
    platform = _TokenIssuingPlatform()
    monkeypatch.setattr(
        McpTimeoutPolicy, "client", lambda self, url, *, auth, caller, tool=None: platform
    )
    records: list[Any] = []
    drain = _RecordingExecutor.drain

    def _drain_and_keep(
        self: _RecordingExecutor, conversation: str, upto: int | None = None
    ) -> list[Any]:
        drained = drain(self, conversation, upto)
        records.extend(drained)
        return drained

    release = _RecordingExecutor.release

    def _keep_and_release(self: _RecordingExecutor, conversation: str) -> None:
        # The records the stream did not translate, which its owner's end
        # drops, taken through the unpatched drain so none is kept twice.
        records.extend(drain(self, conversation))
        release(self, conversation)

    monkeypatch.setattr(_RecordingExecutor, "drain", _drain_and_keep)
    monkeypatch.setattr(_RecordingExecutor, "release", _keep_and_release)
    provenance = ProvenanceStore()
    provenance.put(conversation=f"{PRINCIPAL}/{CONVERSATION}", manifest={}, **REF_A)
    edit_model = ScriptedEditModel(
        {
            "text": "Renamed the card.",
            "patch": {"title": "Aid to Kenya, 2018-2024"},
            "recreate": None,
        }
    )
    # The plan's steps run through `tool_surface`, so the platform sees only
    # the hand-back's frame description, the chart's create and the revise.
    app = _app(
        *_plan("q with forks", *_charts_the_plan()),
        executor=McpExecutor(settings=Settings(), provenance=provenance),
        edit_model=edit_model,
        tool_surface=ScriptedSurface(),
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
    assert CARD_FRAME_KEY not in revise[1]

    conversation = app.state.studio.store.current(PRINCIPAL, CONVERSATION)
    assert conversation.write_tokens == {component_id: token}
    client_visible = [
        *created,
        *resumed,
        *edited,
        conversation.canvas_digest_state(focus=component_id),
        conversation.conversation_digest_state(),
        conversation.components,
        conversation.turns,
    ]
    assert "cmw_" not in json.dumps(client_visible, default=str)
    assert "cmw_" not in repr(conversation)
    # The edit names the card's frame, read from its stored intent, so the
    # executor reads the card's lineage without describing the card again.
    assert [
        record.args.get(CARD_FRAME_KEY)
        for record in records
        if record.tool == "write_component" and "component_id" in record.args
    ] == [REF_A]
    assert "cmw_" not in json.dumps(
        [(record.args, record.result) for record in records], default=str
    )
    # `McpExecutor` took the token off every result, content text included,
    # before any reader downstream of `McpExecutor` saw it.
    assert "cmw_" not in json.dumps(
        [(r.structured_content, [block.text for block in r.content]) for r in platform.results]
    )

    reloaded = await StudioStore(
        _engine(),
    ).load(PRINCIPAL, CONVERSATION)
    assert reloaded.write_tokens == {component_id: token}
    assert "cmw_" not in json.dumps(reloaded.components)


@pytest.mark.parametrize(
    ("edit", "event"),
    [
        pytest.param(
            {"text": "Renamed.", "patch": {"title": "Aid to Kenya, 2018-2024"}, "recreate": None},
            "component.revised",
            id="revise",
        ),
        pytest.param(
            {
                "text": "Rebuilt.",
                "patch": None,
                "recreate": {"type": "stat_card", "title": "Aid", "select": {}},
            },
            "component.created",
            id="recreate",
        ),
    ],
)
async def test_an_edit_on_another_instance_states_the_sources_and_count_of_the_first_draw(
    monkeypatch: pytest.MonkeyPatch, edit: dict[str, Any], event: str
) -> None:
    """A card is drawn on one instance from a frame resting on two
    retrievals whose step declared one StatVar, and edited on a second
    instance whose provenance store is empty. The edit reads the lineage the
    card's write stored with it, so the edited card states the same sources
    and indicator count as the first draw."""
    platform = _TokenIssuingPlatform()
    monkeypatch.setattr(
        McpTimeoutPolicy, "client", lambda self, url, *, auth, caller, tool=None: platform
    )
    conversation = f"{PRINCIPAL}/{CONVERSATION}"
    drawn = ProvenanceStore()
    retrievals = [
        ({"spec_id": "spec_" + str(n) * 32, "content_version": "cv_" + str(n) * 32}, source)
        for n, source in ((2, "src:a"), (3, "src:b"))
    ]
    for ref, source in retrievals:
        manifest = {
            "citation": {"text": source, "sources": [{"ref": source, "label": source}]},
            "trace": {"resolved_stat_vars": [{"ref": f"dcid:{source}"}]},
        }
        drawn.put(conversation=conversation, manifest=manifest, **ref)
    # The plan's chart names the plan's terminal frame, and the platform
    # draws it from REF_A.
    for frame in (REF_A, {"spec_id": "ops_" + "9" * 32, "content_version": "ocv_" + "9" * 32}):
        assert drawn.alias(
            conversation=conversation,
            **frame,
            sources=[(ref["spec_id"], ref["content_version"]) for ref, _ in retrievals],
            stat_var_count=1,
        )
    first_store = StudioStore(_engine())
    first = _app(
        *_plan("q with forks", *_charts_the_plan()),
        executor=McpExecutor(settings=Settings(), provenance=drawn, card_lineages=first_store),
        store=first_store,
        tool_surface=ScriptedSurface(),
    )
    async with _client(first) as client:
        created = await _stream(client, "/api/studio/turn", _turn_body("q with forks"))
        plan = _frame(created, "plan.proposed")
        resumed = await _stream(
            client, "/api/studio/interrupt", _interrupt_body(plan["interrupt_id"], "approve")
        )
    second_store = StudioStore(_engine())
    second = _app(
        executor=McpExecutor(
            settings=Settings(),
            provenance=ProvenanceStore(),
            card_lineages=second_store,
        ),
        store=second_store,
        edit_model=ScriptedEditModel(edit),
    )
    async with _client(second) as client:
        edited = await _stream(client, "/api/studio/turn", _edit_body("change it"))

    drawn_card = _frame(resumed, "component.created")
    edited_card = _frame(edited, event)
    assert [source["id"] for source in drawn_card["provenance"]["sources"]] == ["src:a", "src:b"]
    assert drawn_card["fact_summary"]["attribution"]["indicator_count"] == 1
    # The attribution names the sources and counts them and the indicators.
    assert edited_card["fact_summary"]["attribution"] == drawn_card["fact_summary"]["attribution"]
    if event == "component.created":
        assert edited_card["provenance"]["sources"] == drawn_card["provenance"]["sources"]


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
            *_card_results(),
            *_card_results(third),
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
        *_card(),
        *_card(),
        resolver=_search(2),
        executor=executor,
        edit_model=edit_model,
    )
    async with _client(app) as client:
        await _conversation_with_component(client)
        # A second card, pinned by the user, so the replacement has a
        # neighbour whose slot and pin it must leave alone.
        before = await _conversation_with_component(client)
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
            *_card_results(),
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
        *_card(),
        resolver=_search(),
        executor=executor,
        edit_model=edit_model,
    )
    async with _client(app) as client:
        await _conversation_with_component(client)
        frames = await _stream(client, "/api/studio/turn", _edit_body("only Kenya"))

    assert "component.revised" not in _events(frames)
    assert _events(frames)[-1] == "done"
    # The platform's message names a dimension and a dcid for the model. The
    # closing message and the failed step give the sentence of its code.
    sentence = classed_sentence("schema_invalid", "do_not_retry")
    message = _frame(frames, "message.complete")["text"]
    assert message == f"The edit was refused. {sentence}"
    assert "observationAbout" not in message
    (failed,) = [f for f in frames if f["event"] == "activity.step" and f["status"] == "failed"]
    assert failed["detail"] == sentence
    assert is_user_sentence(sentence)
    assert executor.calls[-1]["tool"] == "write_component"  # the write was what refused


def _unreadable_refusal() -> dict[str, Any]:
    """What the executor answers a write whose stored card the harness cannot
    read (`harness.mcp_client._withheld_card`)."""
    return {
        "outcome": "refusal",
        "refusal": harness_refusal(
            code=HarnessCode.STORED_CARD_UNREADABLE,
            message="the platform stored card cmp_01test0001 but it fails the component contract",
        )
        | {"component_id": "cmp_01test0001"},
    }


@pytest.mark.parametrize("path", ["revise", "rebuild"])
async def test_an_edit_stored_as_a_card_the_harness_cannot_read_is_a_reported_failure(
    path: str,
) -> None:
    """The platform stored the card, so nothing refused the edit, and the harness
    cannot read what was stored. The turn ends on the failure sentence for
    `stored_card_unreadable` with no component frame and does not go fatal. A
    revise says the change was saved and the canvas shows the earlier version,
    so the next edit builds on it. A rebuild says the canvas is unchanged."""
    revise = path == "revise"
    edit_model = ScriptedEditModel(
        {
            "text": "Edited.",
            "patch": {"title": "Aid to Kenya, 2018-2024"} if revise else None,
            "recreate": None
            if revise
            else {"type": "line_chart", "title": "t", "select": {"stat_var": "dcid:Aid"}},
        }
    )
    executor = ScriptedExecutor(
        [
            *_card_results(**({"write_token": WRITE_TOKEN} if revise else {})),
            _described(),
            _VIEW,
            _unreadable_refusal(),
        ],
        strict=True,
    )
    app = _app(*_card(), resolver=_search(), executor=executor, edit_model=edit_model)
    async with _client(app) as client:
        await _conversation_with_component(client)
        frames = await _stream(client, "/api/studio/turn", _edit_body("change it"))

    events = _events(frames)
    assert events[-1] == "done"
    assert "error" not in events
    assert "component.revised" not in events
    assert "component.created" not in events
    sentence = classed_sentence("stored_card_unreadable", "do_not_retry")
    message = _frame(frames, "message.complete")["text"]
    lead = (
        "Your change was saved, but the card cannot be shown with it, so the canvas still "
        "shows the earlier version. Your next edit builds on the saved change."
        if revise
        else "The card on the canvas is unchanged."
    )
    assert message == f"{lead} {sentence}"
    (failed,) = [f for f in frames if f["event"] == "activity.step" and f["status"] == "failed"]
    assert failed["detail"] == sentence
    assert failed.get("produced") is None


@pytest.mark.parametrize(
    ("card", "revised"),
    [
        pytest.param(
            {**COMPONENT, "revision": 1, "type": "no_such_component_type"},
            False,
            id="no-op-is-not-judged-on-readability",
        ),
        pytest.param(
            {
                **{k: v for k, v in COMPONENT.items() if k != "provenance"},
                "revision": 2,
                "fact_summary": {**COMPONENT["fact_summary"], "revision": 2},
            },
            True,
            id="revise-needs-only-what-the-revised-frame-carries",
        ),
    ],
)
async def test_a_revise_is_judged_on_what_its_own_frame_needs(
    card: dict[str, Any], revised: bool
) -> None:
    """A no-op revise ends on the no-op sentence whatever the stored card
    holds, and a revised card streams `component.revised` when it carries the
    members that frame needs, though `component.created` would refuse it."""
    executor = ScriptedExecutor(
        [
            *_card_results(write_token=WRITE_TOKEN),
            _described(),
            _VIEW,
            {"outcome": "ok", "frame_refs": [], "component": card},
        ],
        strict=True,
    )
    edit_model = ScriptedEditModel(
        {"text": "Renamed.", "patch": {"title": "Aid to Kenya, 2018-2024"}, "recreate": None}
    )
    app = _app(*_card(), resolver=_search(), executor=executor, edit_model=edit_model)
    async with _client(app) as client:
        await _conversation_with_component(client)
        frames = await _stream(client, "/api/studio/turn", _edit_body("rename it"))

    events = _events(frames)
    assert events[-1] == "done"
    assert ("component.revised" in events) is revised
    assert "component.created" not in events
    assert (_frame(frames, "message.complete")["text"] == "Renamed.") is revised
    assert not [f for f in frames if f["event"] == "activity.step" and f["status"] == "failed"]


async def test_a_no_op_revise_over_a_readable_head_above_the_canvas_catches_the_canvas_up() -> None:
    """A revise timed out after the platform stored it, so the canvas shows revision 1
    while the head is 2. The same edit again is a no-op write that returns the readable
    head at revision 2: it is applied as a revise, with `component.revised` for that
    revision, the canvas catching up, and the usual success text."""
    head = {
        **COMPONENT,
        "revision": 2,
        "fact_summary": {**COMPONENT["fact_summary"], "revision": 2},
        "payload": resolved_revision(
            "stat_card",
            component_id="cmp_01test0001",
            revision=2,
            frame_ref=REF_A,
            title="Aid to Kenya, stored by the timed-out edit",
        ),
    }
    executor = ScriptedExecutor(
        [
            *_card_results(write_token=WRITE_TOKEN),
            _described(2),
            _VIEW,
            {"outcome": "ok", "frame_refs": [], "component": head},
        ],
        strict=True,
    )
    edit_model = ScriptedEditModel(
        {"text": "Renamed.", "patch": {"note": "same again"}, "recreate": None}
    )
    app = _app(*_card(), resolver=_search(), executor=executor, edit_model=edit_model)
    async with _client(app) as client:
        await _conversation_with_component(client)
        frames = await _stream(client, "/api/studio/turn", _edit_body("rename it"))

    assert _frame(frames, "component.revised")["revision"] == 2
    assert _frame(frames, "message.complete")["text"] == "Renamed."
    assert not [f for f in frames if f["event"] == "activity.step" and f["status"] == "failed"]
    block = _frame(frames, "done")["composition"]["blocks"][0]
    assert block["revision"] == 2
    assert block["title"] == "Aid to Kenya, stored by the timed-out edit"
    assert _frame(frames, "composition.updated")["manifest"]["blocks"][0]["title"] == block["title"]
    entry = app.state.studio.store.current(PRINCIPAL, CONVERSATION).components["cmp_01test0001"]
    assert entry["title"] == block["title"]


async def test_a_revise_whose_card_write_lost_its_acknowledgement_reports_one_canvas_change(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The card's write commits and its acknowledgement is lost. The retry
    finds the block already revised and mints no second manifest revision,
    and the stream still sends the `composition.updated` the commit made."""
    monkeypatch.setattr(studio_store_module, "_RETRY_FIRST_DELAY_S", 0)
    revised_component = {
        **COMPONENT,
        "revision": 2,
        "fact_summary": {**COMPONENT["fact_summary"], "revision": 2},
    }
    executor = ScriptedExecutor(
        [
            *_card_results(write_token=WRITE_TOKEN),
            _described(),
            _VIEW,
            {"outcome": "ok", "frame_refs": [], "component": revised_component},
        ],
        strict=True,
    )
    edit_model = ScriptedEditModel(
        {"text": "Renamed the card.", "patch": {"title": "Aid to Kenya"}, "recreate": None}
    )
    app = _app(*_card(), resolver=_search(), executor=executor, edit_model=edit_model)
    store = app.state.studio.store
    async with _client(app) as client:
        landed = await _conversation_with_component(client)
        real_transaction, real_write = store._transaction, store._write_composition
        faults: list[int] = []
        armed: list[int] = []

        async def arming_write(*args: Any, **kwargs: Any) -> None:
            armed.append(1)
            await real_write(*args, **kwargs)

        @asynccontextmanager
        async def losing_the_ack() -> Any:
            async with real_transaction() as conn:
                yield conn
            if armed and not faults:
                faults.append(1)
                reset = ConnectionResetError("connection reset")
                raise OperationalError("COMMIT", {}, reset) from reset

        monkeypatch.setattr(store, "_write_composition", arming_write)
        monkeypatch.setattr(store, "_transaction", losing_the_ack)
        frames = await _stream(client, "/api/studio/turn", _edit_body("rename it"))

    assert faults == [1]
    (updated,) = [f for f in frames if f["event"] == "composition.updated"]
    reloaded = await StudioStore(_engine()).load(PRINCIPAL, CONVERSATION)
    assert reloaded.composition.revision == landed["revision"] + 1
    assert updated["manifest"]["revision"] == reloaded.composition.revision
    assert reloaded.composition.blocks[0].revision == 2


async def test_an_edit_whose_canvas_write_failed_lands_when_repeated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A revise lands on the platform (revision 1 to 2) and the card's write
    fails, so the card's digest entry and its block stay at revision 1 and the
    turn ends in an error. The same edit again is a no-op write over the head
    at revision 2, which the conversation is behind, so it applies as a revise
    and the canvas reaches revision 2."""
    head = copy.deepcopy(COMPONENT)
    head["revision"] = 2
    head["fact_summary"]["revision"] = 2
    head["payload"] = resolved_revision(
        "stat_card", component_id="cmp_01test0001", revision=2, frame_ref=REF_A, title="Renamed"
    )
    written = {"outcome": "ok", "frame_refs": [], "component": head}
    executor = ScriptedExecutor(
        [
            *_card_results(write_token=WRITE_TOKEN),
            _described(),
            _VIEW,
            written,
            _described(2),
            _VIEW,
            written,
        ],
        strict=True,
    )
    edit_model = ScriptedEditModel(
        {"text": "Renamed.", "patch": {"title": "Renamed"}, "recreate": None}
    )
    app = _app(*_card(), resolver=_search(), executor=executor, edit_model=edit_model)
    store = app.state.studio.store
    async with _client(app) as client:
        await _conversation_with_component(client)
        real_write = store._write_composition
        failures = 0

        async def fail_once(*args: Any, **kwargs: Any) -> None:
            nonlocal failures
            if not failures:
                failures += 1
                raise RuntimeError("the canvas write failed")
            await real_write(*args, **kwargs)

        monkeypatch.setattr(store, "_write_composition", fail_once)
        failed = await _stream(client, "/api/studio/turn", _edit_body("rename it"))
        stored = await StudioStore(_engine()).load(PRINCIPAL, CONVERSATION)
        repeated = await _stream(client, "/api/studio/turn", _edit_body("rename it"))

    assert _events(failed)[-1] == "error"
    assert stored.components["cmp_01test0001"]["revision"] == 1
    assert stored.composition.blocks[0].revision == 1
    assert _frame(repeated, "message.complete")["text"] == "Renamed."
    reloaded = await StudioStore(_engine()).load(PRINCIPAL, CONVERSATION)
    assert reloaded.composition.blocks[0].revision == 2
    assert reloaded.components["cmp_01test0001"]["revision"] == 2


@pytest.mark.parametrize(
    ("recorded", "shown", "behind"),
    [
        (1, 1, True),
        (2, 1, True),
        (1, None, True),
        (2, 2, False),
        (2, None, False),
    ],
    ids=[
        "both-behind",
        "block-behind",
        "held-card-behind",
        "caught-up",
        "held-card-caught-up",
    ],
)
def test_the_conversation_is_behind_a_head_its_entry_or_block_is_below(
    recorded: int, shown: int | None, behind: bool
) -> None:
    conversation = ConversationState(conversation_id=CONVERSATION)
    conversation.components["cmp_01test0001"] = {"type": "stat_card", "revision": recorded}
    if shown is not None:
        block = CompositionBlock(
            component_id="cmp_01test0001",
            revision=shown,
            type=ComponentType.STAT_CARD,
            title="Aid",
        )
        conversation.composition = studio_state_module._next_manifest(
            None, [block], actor=CompositionActor.MODEL
        )

    assert _canvas_is_behind(conversation, "cmp_01test0001", {"revision": 2}) is behind


async def test_a_repeat_of_a_rebuild_stored_unreadable_writes_again() -> None:
    """The first rebuild stored a card the harness could not read. The same
    rebuild again is written again, since nothing records the first: each
    repeat leaves one more orphan, which the platform's refusal log names."""
    recreate = {"type": "line_chart", "title": "t", "select": {"stat_var": "dcid:Aid"}}
    executor = ScriptedExecutor(
        [
            *_card_results(),
            _described(),
            _VIEW,
            _unreadable_refusal(),
            _described(),
            _VIEW,
            _unreadable_refusal(),
        ],
        strict=True,
    )
    edit_model = ScriptedEditModel({"text": "Rebuilt.", "patch": None, "recreate": recreate})
    app = _app(*_card(), resolver=_search(), executor=executor, edit_model=edit_model)
    async with _client(app) as client:
        await _conversation_with_component(client)
        await _stream(client, "/api/studio/turn", _edit_body("make it a line"))
        frames = await _stream(client, "/api/studio/turn", _edit_body("make it a line"))

    sentence = classed_sentence("stored_card_unreadable", "do_not_retry")
    assert _frame(frames, "message.complete")["text"] == (
        f"The card on the canvas is unchanged. {sentence}"
    )
    rebuilds = [
        call
        for call in executor.calls
        if call["tool"] == "write_component" and call["args"]["intent"]["type"] == "line_chart"
    ]
    assert len(rebuilds) == 2


async def test_a_revise_that_needs_another_frame_is_written_as_a_new_card_in_its_slot() -> None:
    """The executor refuses a revise whose select counts other series than the
    card's rolling frame did. The edit is then written as a create of the stored
    intent with the patch merged in, which the executor moves to the right frame."""
    patch = {"select": {"stat_var": "dcid:Aid", "donor": "country/FRA"}}
    recreated = {**COMPONENT, "component_id": "cmp_01test0002"}
    executor = ScriptedExecutor(
        [
            *_card_results(write_token=WRITE_TOKEN),
            _described(),
            _VIEW,
            {
                "outcome": "refusal",
                "refusal": {
                    "code": "rolling_select_needs_new_frame",
                    "origin": "harness",
                    "retry_mode": "do_not_retry",
                    "message": "The card cannot be changed in place.",
                },
            },
            {"outcome": "ok", "frame_refs": [], "component": recreated},
        ],
        strict=True,
    )
    app = _app(
        *_card(),
        resolver=_search(),
        executor=executor,
        edit_model=ScriptedEditModel({"text": "Now France.", "patch": patch, "recreate": None}),
    )
    async with _client(app) as client:
        await _conversation_with_component(client)
        frames = await _stream(client, "/api/studio/turn", _edit_body("only France"))

    revise, create = executor.calls[-2:]
    assert revise["args"] == {
        "intent": patch,
        "component_id": "cmp_01test0001",
        "write_token": WRITE_TOKEN,
        CARD_FRAME_KEY: STORED_INTENT["frame_ref"],
    }
    assert create == {
        "tool": "write_component",
        "args": {"intent": {**STORED_INTENT, **patch}},
    }
    assert _frame(frames, "component.created")["component_id"] == "cmp_01test0002"
    assert _frame(frames, "message.complete")["text"] == "Now France."
    manifest = _frame(frames, "done")["composition"]
    assert [block["component_id"] for block in manifest["blocks"]] == ["cmp_01test0002"]


@pytest.mark.parametrize(
    "period",
    [
        pytest.param(
            {
                "window": {
                    "start": {"anchor": "latest", "offset": -2},
                    "end": {"anchor": "latest", "offset": 0},
                }
            },
            id="relative-window",
        ),
        pytest.param({"window": {"start": "2015", "end": "2020"}}, id="fixed-window"),
        pytest.param(
            {
                "change": {
                    "basis": [
                        {"anchor": "latest", "offset": -1},
                        {"anchor": "latest", "offset": 0},
                    ]
                }
            },
            id="relative-change",
        ),
    ],
)
async def test_an_edit_writes_the_period_as_the_model_drafted_it(period: dict[str, Any]) -> None:
    """A person may ask for a period counted from the latest data or for named
    years; the route sends either form to the write untouched."""
    recreate = {
        "type": "stat_card",
        "title": "Aid to Kenya",
        "select": STORED_INTENT["select"],
        **period,
    }
    executor = ScriptedExecutor(
        [
            *_card_results(),
            _described(),
            _VIEW,
            {"outcome": "ok", "frame_refs": [], "component": COMPONENT},
        ],
        strict=True,
    )
    app = _app(
        *_card(),
        resolver=_search(),
        executor=executor,
        edit_model=ScriptedEditModel({"text": "Done.", "patch": None, "recreate": recreate}),
    )
    async with _client(app) as client:
        await _conversation_with_component(client)
        await _stream(client, "/api/studio/turn", _edit_body("change the period"))

    assert executor.calls[-1] == {
        "tool": "write_component",
        "args": {"intent": {**recreate, "frame_ref": REF_A}},
    }


_ORDERED_BAR = {
    "type": "bar_chart",
    "title": "Aid to Kenya",
    "select": {"stat_var": "dcid:Aid"},
    "frame_ref": REF_A,
    "order": {"by": "value", "direction": "descending"},
}
_ORDERED_LINE = {
    **_ORDERED_BAR,
    "type": "line_chart",
    "order": {"by": "value", "direction": "descending", "period": "2023"},
}
_LARGEST_FIRST = {"by": "value", "direction": "descending"}
_OTHER_SELECT = {"stat_var": "dcid:Loans"}


@pytest.mark.parametrize(
    ("stored", "recreate", "written_order"),
    [
        # The model copied the stored order onto a type that takes none.
        (_ORDERED_BAR, {"type": "stat_card", "order": _LARGEST_FIRST}, None),
        # The model left the order out, so the rebuilt table is unordered.
        (_ORDERED_BAR, {"type": "table"}, None),
        # A bar chart takes no period.
        (_ORDERED_LINE, {"type": "bar_chart", "order": _ORDERED_LINE["order"]}, _LARGEST_FIRST),
        # The same series rank at the stored period.
        (_ORDERED_LINE, {"type": "table", "order": _ORDERED_LINE["order"]}, _ORDERED_LINE["order"]),
        # Other series may hold no value at the stored period.
        (
            _ORDERED_LINE,
            {"type": "line_chart", "select": _OTHER_SELECT, "order": _ORDERED_LINE["order"]},
            _LARGEST_FIRST,
        ),
        # A period the model names afresh stands.
        (
            _ORDERED_LINE,
            {
                "type": "line_chart",
                "select": _OTHER_SELECT,
                "order": {**_LARGEST_FIRST, "period": "2021"},
            },
            {**_LARGEST_FIRST, "period": "2021"},
        ),
    ],
)
async def test_an_edit_recreate_writes_the_drafted_order_where_the_type_takes_it(
    stored: dict[str, Any], recreate: dict[str, Any], written_order: dict[str, str] | None
) -> None:
    """A recreate writes the order the model drafted, made legal for its type:
    none on a stat card, no period on a bar chart, and the stored card's period
    only over its series and window."""
    recreate = {"title": "Aid to Kenya", "select": stored["select"], **recreate}
    executor = ScriptedExecutor(
        [
            *_card_results(),
            _described(stored_intent=stored),
            _VIEW,
            {"outcome": "ok", "frame_refs": [], "component": COMPONENT},
        ],
        strict=True,
    )
    app = _app(
        *_card(),
        resolver=_search(),
        executor=executor,
        edit_model=ScriptedEditModel({"text": "Done.", "patch": None, "recreate": recreate}),
    )
    async with _client(app) as client:
        await _conversation_with_component(client)
        await _stream(client, "/api/studio/turn", _edit_body("rebuild it"))

    write = executor.calls[-1]
    assert write["tool"] == "write_component"
    assert write["args"]["intent"].get("order") == written_order


@pytest.mark.parametrize(
    ("patch", "written_order"),
    [
        # A new select ranks at the platform's own period.
        ({"select": _OTHER_SELECT}, _LARGEST_FIRST),
        ({"order": None}, None),
        (
            {"order": {"by": "value", "direction": "ascending"}},
            {"by": "value", "direction": "ascending"},
        ),
    ],
)
async def test_an_edit_patch_writes_the_order_it_names(
    patch: dict[str, Any], written_order: dict[str, str] | None
) -> None:
    """A patch sets or clears the order, and one that changes the select and
    leaves the order alone drops the stored order's period."""
    _, executor = await _edit_without_a_write_token(patch, stored_intent=_ORDERED_LINE)

    write = executor.calls[-1]
    assert write["tool"] == "write_component"
    assert write["args"]["intent"].get("order") == written_order


# --- the canvas digest -----------------------------------------------------------


def _digest(request: Any, label: str) -> dict[str, Any] | None:
    """The digest the agent's `request` carries under `label`, the heading
    `orchestrator.agent` prepends it with, or `None` when it carries none."""
    opening = f"{label}: "
    for line in agent_read(request).split("\n"):
        if line.startswith(opening):
            return json.loads(line.removeprefix(opening))
    return None


async def test_turn_two_receives_the_canvas_digest() -> None:
    app = _app(*_card(), resolver=_search())
    async with _client(app) as client:
        await _conversation_with_component(client)
        await _stream(client, "/api/studio/turn", _turn_body("now the same for Uganda"))

    requests = _model(app).requests
    assert _digest(requests[0], "The user's canvas") is None  # turn 1: empty canvas
    turn2 = _digest(requests[-1], "The user's canvas")
    assert turn2 is not None
    assert turn2 == {"cards_on_canvas": 1, "cards_hidden": 0}


# --- elicitation skip and free text (deltas 12) ----------------------------------


async def test_a_skipped_ask_resumes_and_hands_the_skip_back() -> None:
    """An ask is skippable and takes free text, and a skip resumes the turn
    with the agent told the user skipped it."""
    app = _app(*_ask(), _text("Ask me something else."))
    async with _client(app) as client:
        first = await _stream(client, "/api/studio/turn", _turn_body("ambiguous q"))
        elicitation = _frame(first, "elicitation.request")
        assert elicitation["skippable"] is True
        assert elicitation["questions"][0]["allow_free_text"] is True

        second = await _stream(
            client,
            "/api/studio/interrupt",
            _interrupt_body(elicitation["interrupt_id"], "skip"),
        )

    assert _events(second)[-1] == "done"
    assert "The user skipped the question." in agent_read(_model(app).requests[-1])


async def test_a_free_text_answer_resumes_in_the_users_own_words() -> None:
    app = _app(*_ask(), _text("The OECD reading it is."))
    async with _client(app) as client:
        first = await _stream(client, "/api/studio/turn", _turn_body("ambiguous q"))
        elicitation = _frame(first, "elicitation.request")

        second = await _stream(
            client,
            "/api/studio/interrupt",
            _interrupt_body(
                elicitation["interrupt_id"],
                "answer",
                answers={"q1": {"kind": "text", "text": "the OECD reading please"}},
            ),
        )

    assert _events(second)[-1] == "done"
    assert "The user answered in their own words: the OECD reading please" in agent_read(
        _model(app).requests[-1]
    )


async def test_a_pause_opened_before_questions_takes_its_answer_as_a_payload() -> None:
    """An elicitation stored with flat `option_ids` reads back as one question,
    and `/interrupt` takes its answer as a single `payload`, a pick of an
    offered id."""
    app = _app(*_ask("reading A", "reading B"), _text("You meant reading B."))
    async with _client(app) as client:
        first = await _stream(client, "/api/studio/turn", _turn_body("ambiguous q"))
        elicitation = _frame(first, "elicitation.request")
        option_id = elicitation["questions"][0]["options"][1]["id"]
        conversation = app.state.studio.store.current(PRINCIPAL, CONVERSATION)
        stored = conversation.pending[elicitation["interrupt_id"]]
        conversation.pending = _deserialize_pending(
            {
                elicitation["interrupt_id"]: {
                    "adk_id": stored.adk_id,
                    "kind": "elicitation",
                    "option_ids": [
                        option["id"] for option in elicitation["questions"][0]["options"]
                    ],
                }
            }
        )
        await app.state.studio.store.save(PRINCIPAL, CONVERSATION)

        second = await _stream(
            client,
            "/api/studio/interrupt",
            _interrupt_body(elicitation["interrupt_id"], "answer", option_id),
        )

    assert _events(second)[-1] == "done"
    assert "The user chose: reading B" in agent_read(_model(app).requests[-1])


# Each answer `/interrupt` refuses to an elicitation of two questions, both
# skippable on their own, offering options "A"/"B" and "C"/"D".
_REFUSED_ANSWERS: dict[str, tuple[dict[str, Any], str]] = {
    "missing-question": ({"answers": {"q1": {"kind": "skipped"}}}, "missing ['q2']"),
    "extra-question": (
        {
            "answers": {
                "q1": {"kind": "skipped"},
                "q2": {"kind": "skipped"},
                "q3": {"kind": "skipped"},
            }
        },
        "extra ['q3']",
    ),
    "unknown-option": (
        {"answers": {"q1": _pick("o9"), "q2": {"kind": "skipped"}}},
        "option 'o9' is not offered by question 'q1'",
    ),
    "payload-instead-of-answers": ({"payload": "o1"}, "carries answers, not a payload"),
    "no-answers": ({}, "requires answers"),
    "skip-of-the-whole-pause": ({"decision": "skip"}, "cannot be skipped as a whole"),
}


@pytest.mark.parametrize("case", sorted(_REFUSED_ANSWERS))
async def test_an_answer_that_does_not_fit_the_questions_is_refused(case: str) -> None:
    change, detail = _REFUSED_ANSWERS[case]
    app = _app(*_ask_questions(("Count?", ("A", "B")), ("Prices?", ("C", "D"))))
    async with _client(app) as client:
        first = await _stream(client, "/api/studio/turn", _turn_body("ambiguous q"))
        elicitation = _frame(first, "elicitation.request")
        body = {**_interrupt_body(elicitation["interrupt_id"], "answer"), **change}
        response = await client.post("/api/studio/interrupt", json=body, headers=AUTH)

    assert response.status_code == 422
    assert response.json()["error"] == "binding_invalid"
    assert detail in response.json()["detail"]
    assert (
        elicitation["interrupt_id"]
        in app.state.studio.store.current(PRINCIPAL, CONVERSATION).pending
    )


async def test_a_question_that_is_the_only_one_cannot_be_skipped_alone() -> None:
    """A lone question is dismissed with the whole pause, so its own
    `skipped` answer is refused."""
    async with _client(_app(*_ask())) as client:
        first = await _stream(client, "/api/studio/turn", _turn_body("ambiguous q"))
        elicitation = _frame(first, "elicitation.request")
        response = await client.post(
            "/api/studio/interrupt",
            json=_interrupt_body(
                elicitation["interrupt_id"], "answer", answers={"q1": {"kind": "skipped"}}
            ),
            headers=AUTH,
        )
    assert response.status_code == 422
    assert "question 'q1' cannot be skipped" in response.json()["detail"]


# --- auto_add and the add / set_card_meta ops (deltas 11, 13) --------------------


async def test_auto_add_false_streams_the_component_without_composing_it() -> None:
    app = _app(*_card(), resolver=_search())
    async with _client(app) as client:
        frames = await _stream(client, "/api/studio/turn", {**_turn_body(), "auto_add": False})

        assert "component.created" in _events(frames)
        assert "composition.updated" not in _events(frames)
        assert _frame(frames, "done").get("composition") is None

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
        [*_card_results(), *_card_results({**COMPONENT, "component_id": "cmp_01test0002"})]
    )
    app = _app(*_card(), *_card(), resolver=_search(2), executor=executor)
    async with _client(app) as client:
        manifest = await _conversation_with_component(client)
        # a second component that stays off the canvas (auto_add=false)
        frames = await _stream(client, "/api/studio/turn", {**_turn_body(), "auto_add": False})
        assert _frame(frames, "component.created")["component_id"] == "cmp_01test0002"

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
    executor = ScriptedExecutor()
    app = _app(
        *_card(),
        resolver=_search(),
        executor=executor,
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
            *_card_results(),
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
        *_card(),
        resolver=_search(),
        executor=executor,
        edit_model=edit_model,
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
            *_card_results(),
            _described(),
            _VIEW,
        ],
        strict=True,
    )
    app = _app(
        *_card(),
        resolver=_search(),
        executor=executor,
        edit_model=ExplodingEditModel(),
    )
    async with _client(app) as client:
        await _conversation_with_component(client)
        frames = await _stream(client, "/api/studio/turn", _edit_body("break please"))

    assert _events(frames)[-1] == "error"
    assert "done" not in _events(frames)


async def test_answer_without_answers_is_refused() -> None:
    async with _client(_app(*_ask())) as client:
        first = await _stream(client, "/api/studio/turn", _turn_body("ambiguous q"))
        elicitation = _frame(first, "elicitation.request")
        response = await client.post(
            "/api/studio/interrupt",
            json=_interrupt_body(elicitation["interrupt_id"], "answer"),
            headers=AUTH,
        )
        assert response.status_code == 422
        assert "requires answers" in response.json()["detail"]
        # the interrupt is still pending after the refusal; skipping works
        second = await _stream(
            client,
            "/api/studio/interrupt",
            _interrupt_body(elicitation["interrupt_id"], "skip"),
        )
    assert _events(second)[-1] == "done"


async def test_approve_with_a_payload_is_refused() -> None:
    async with _client(_app(*_plan())) as client:
        first = await _stream(client, "/api/studio/turn", _turn_body("q with forks"))
        plan = _frame(first, "plan.proposed")
        response = await client.post(
            "/api/studio/interrupt",
            json=_interrupt_body(plan["interrupt_id"], "approve", "stray"),
            headers=AUTH,
        )
    assert response.status_code == 422
    assert "does not take a payload" in response.json()["detail"]


# A scope choice a plan card shows: health counted as CRS 120 alone, with
# CRS 120 and 130 to switch to and a third option out of reach.
_CHOICE_ID = "spec_" + "b" * 32 + ":sector"
_SHOWN_CHOICE = _PendingChoice(
    active="sector:120",
    options={
        "sector:120": _PendingOption(label="health (CRS 120)", status="available"),
        "sector:120+130": _PendingOption(label="health and population", status="available"),
        "sector:130": _PendingOption(label="population (CRS 130)", status="unavailable"),
    },
)


async def _plan_with_a_choice(
    app: FastAPI,
    client: httpx2.AsyncClient,
    *,
    choices: dict[str, _PendingChoice] | None = None,
) -> dict[str, Any]:
    """Open a turn pausing on its plan card, and keep `choices`
    (`_SHOWN_CHOICE` by default) on the pause as a card showing them would;
    the plan.proposed frame."""
    first = await _stream(client, "/api/studio/turn", _turn_body())
    plan = _frame(first, "plan.proposed")
    conversation = app.state.studio.store.current(PRINCIPAL, CONVERSATION)
    conversation.pending[plan["interrupt_id"]].choices = choices or {_CHOICE_ID: _SHOWN_CHOICE}
    await app.state.studio.store.save(PRINCIPAL, CONVERSATION)
    return plan


def _switch_body(plan: dict[str, Any], **change: Any) -> dict[str, Any]:
    return {
        **_interrupt_body(plan["interrupt_id"], "switch"),
        "choice_id": _CHOICE_ID,
        "option_id": "sector:120+130",
        "plan_revision": plan["revision"],
        **change,
    }


async def test_a_switch_the_card_cannot_take_is_refused_with_its_constraint() -> None:
    """A stale revision is refused 409, an option the card does not offer
    as switchable 422, each naming its constraint and leaving the pause to
    answer; a switch carrying a payload breaks the payload discipline."""
    app = _app(*_plan())
    async with _client(app) as client:
        plan = await _plan_with_a_choice(app, client)
        cases = [
            ({"plan_revision": "0" * 64}, 409, "plan_revision_stale"),
            ({"choice_id": "spec_other:sector"}, 422, "switch_option_unavailable"),
            ({"option_id": "sector:999"}, 422, "switch_option_unavailable"),
            ({"option_id": "sector:130"}, 422, "switch_option_unavailable"),
            ({"option_id": "sector:120"}, 422, "switch_option_unavailable"),
            ({"payload": "stray"}, 422, None),
        ]
        answers = [
            await client.post(
                "/api/studio/interrupt", json=_switch_body(plan, **change), headers=AUTH
            )
            for change, _, _ in cases
        ]
        conversation = app.state.studio.store.current(PRINCIPAL, CONVERSATION)
        assert plan["interrupt_id"] in conversation.pending

    assert [(r.status_code, r.json().get("constraint_id")) for r in answers] == [
        (status, constraint) for _, status, constraint in cases
    ]
    assert all(r.json()["detail"] for r in answers)
    assert answers[0].json()["current_plan"] == plan
    assert answers[-1].json()["error"] == "binding_invalid"
    assert conversation.approvals == []


# A planner decision a plan card shows as a switchable row
# (`harness.plan_display.switchable_decision`): the DAC total, with the top
# 10 donors to switch to.
_DECISION_ID = "decision/donor_scope"
_DAC_TOTAL = decision_option_id("the DAC total")
_TOP_TEN = decision_option_id("the top 10 donors")
_SHOWN_DECISION = _PendingChoice(
    active=_DAC_TOTAL,
    options={
        _DAC_TOTAL: _PendingOption(label="the DAC total", status="available"),
        _TOP_TEN: _PendingOption(label="the top 10 donors", status="available"),
    },
)


async def test_a_planner_decision_switch_is_checked_as_a_scope_switch_and_resumes_the_plan() -> (
    None
):
    """A switch naming a planner decision's row is refused 409 on a stale
    revision and 422 on an option the row does not offer or already holds;
    one to another option resumes the paused plan, approving nothing, and the
    turn's record says which option the user chose."""
    app = _app(*_plan())
    async with _client(app) as client:
        plan = await _plan_with_a_choice(app, client, choices={_DECISION_ID: _SHOWN_DECISION})
        cases = [
            ({"plan_revision": "0" * 64}, 409, "plan_revision_stale"),
            (
                {"option_id": decision_option_id("the top 20 donors")},
                422,
                "switch_option_unavailable",
            ),
            ({"option_id": _DAC_TOTAL}, 422, "switch_option_unavailable"),
        ]
        refused = [
            await client.post(
                "/api/studio/interrupt",
                json=_switch_body(plan, choice_id=_DECISION_ID, **change),
                headers=AUTH,
            )
            for change, _, _ in cases
        ]
        await _stream(
            client,
            "/api/studio/interrupt",
            _switch_body(plan, choice_id=_DECISION_ID, option_id=_TOP_TEN),
        )

    assert [(r.status_code, r.json().get("constraint_id")) for r in refused] == [
        (status, constraint) for _, status, constraint in cases
    ]
    conversation = app.state.studio.store.current(PRINCIPAL, CONVERSATION)
    assert conversation.approvals == []
    assert conversation.turn_user_message.endswith("(then: use the top 10 donors)")


class _AskingPickPlanner(ScriptedPlannerModel):
    """`ScriptedPlannerModel` whose measure decision fixes the measure, so the
    card offers it to switch, and whose replan for a pick asks the user a
    question and then drafts until it is cancelled."""

    def __init__(self) -> None:
        super().__init__()
        self.replanning = asyncio.Event()

    async def __call__(self, *, ctx: Any, **kwargs: Any) -> dict[str, Any]:
        if kwargs.get("pick") is None:
            drafted = await super().__call__(ctx=ctx, **kwargs)
            (measure,) = drafted["plan"]["decisions"]
            measure["fixes"] = {"measure": measure["chosen"]}
            return drafted
        if kwargs.get("answer") is None:
            return {"asked": {"call_id": "replan_ask", "args": {"message": "Which years?"}}}
        self.replanning.set()
        await asyncio.sleep(5)
        raise AssertionError("a stopped replan is cancelled")


async def test_a_stop_after_a_question_the_planner_asked_while_replanning_reoffers_the_plan() -> (
    None
):
    """The user picks another option for a planner decision, and the planner
    asks a question while it replans. The stream that answers it runs the
    replan, so a Stop there is a switch's Stop: the stream ends on the same
    plan re-offered with `switch_cancelled`, and the turn stays paused on it.
    The planner node holds the turn's `StopScope` open over the replan, so
    the route arms nothing."""
    planner = _AskingPickPlanner()
    app = _app(*_plan(), planner=planner)
    stop = {"kind": "turn.cancel", "schema_version": 1, "conversation_id": CONVERSATION}
    async with _client(app) as client:
        first = await _stream(client, "/api/studio/turn", _turn_body())
        plan = _frame(first, "plan.proposed")
        (decision,) = [d for d in plan["decisions"] if d.get("choice_id")]
        option = next(
            o["option_id"]
            for o in decision["options"]
            if o["option_id"] != decision["active_option_id"]
        )
        asked = await _stream(
            client,
            "/api/studio/interrupt",
            _switch_body(plan, choice_id=decision["choice_id"], option_id=option),
        )
        elicitation = _frame(asked, "elicitation.request")
        conversation = app.state.studio.store.current(PRINCIPAL, CONVERSATION)
        resumed = asyncio.ensure_future(
            _stream(
                client,
                "/api/studio/interrupt",
                _interrupt_body(elicitation["interrupt_id"], "skip"),
            )
        )
        await asyncio.wait_for(planner.replanning.wait(), timeout=5)
        cancelled = await client.post("/api/studio/cancel", json=stop, headers=AUTH)
        assert cancelled.status_code == 204
        frames = await asyncio.wait_for(resumed, timeout=5)

    assert _events(frames)[-1] == "plan.proposed"
    again = _frame(frames, "plan.proposed")
    assert again["revision"] == plan["revision"]
    assert again["switch_failed"]["code"] == "switch_cancelled"
    assert list(conversation.pending) == [again["interrupt_id"]]
    assert conversation.awaiting_answer


async def test_a_stale_approve_gets_the_current_card_and_the_shown_revision_approves() -> None:
    """An approve naming a revision other than the paused plan's is refused
    409 with the card the stream sent, approves nothing, and leaves the
    pause; one naming the card's own revision approves it."""
    app = _app(*_plan(), tool_surface=ScriptedSurface())
    async with _client(app) as client:
        first = await _stream(client, "/api/studio/turn", _turn_body("q with forks"))
        plan = _frame(first, "plan.proposed")
        approve = _interrupt_body(plan["interrupt_id"], "approve")
        stale = await client.post(
            "/api/studio/interrupt", json={**approve, "plan_revision": "0" * 64}, headers=AUTH
        )
        conversation = app.state.studio.store.current(PRINCIPAL, CONVERSATION)
        assert conversation.approvals == []
        approved = await _stream(
            client, "/api/studio/interrupt", {**approve, "plan_revision": plan["revision"]}
        )

    assert stale.status_code == 409
    assert stale.json()["constraint_id"] == "plan_revision_stale"
    assert stale.json()["detail"]
    assert stale.json()["current_plan"] == plan
    assert _events(approved)[-1] == "done"
    assert [record.revision for record in conversation.approvals] == [plan["revision"]]


async def test_a_switch_approves_nothing_and_the_plan_is_offered_again() -> None:
    """A switch the card offers resumes the turn with no approval record;
    the graph revises the plan and offers it on a fresh card. Here the plan
    states no scope choice, so the same plan comes back saying why on the
    card, with no chat message."""
    app = _app(*_plan())
    async with _client(app) as client:
        plan = await _plan_with_a_choice(app, client)
        second = await _stream(client, "/api/studio/interrupt", _switch_body(plan))

    conversation = app.state.studio.store.current(PRINCIPAL, CONVERSATION)
    assert conversation.approvals == []
    assert "message.complete" not in _events(second)
    assert _events(second)[-1] == "plan.proposed"
    again = _frame(second, "plan.proposed")
    assert again["interrupt_id"] != plan["interrupt_id"]
    assert again["revision"] == plan["revision"]
    assert again["switch_failed"]["reason"] == "That choice is no longer on the plan."
    assert conversation.turn_user_message.endswith("(then: count health and population)")


class _StopsTaken:
    """Counts the Stop requests `store`'s heartbeat renewals read, each once,
    so a test sends its next Stop only once the one before it was taken."""

    def __init__(self, store: StudioStore) -> None:
        self.taken = 0
        self._changed = asyncio.Condition()
        self._seen: set[str] = set()
        renew = store.renew

        async def counting(*args: Any, **kwargs: Any) -> Any:
            renewal = await renew(*args, **kwargs)
            if renewal.stop_id is not None and renewal.stop_id not in self._seen:
                self._seen.add(renewal.stop_id)
                async with self._changed:
                    self.taken += 1
                    self._changed.notify_all()
            return renewal

        store.renew = counting  # type: ignore[method-assign]

    async def reach(self, taken: int) -> None:
        async with self._changed:
            await asyncio.wait_for(self._changed.wait_for(lambda: self.taken >= taken), timeout=5)


async def test_a_lock_just_handed_to_a_waiter_bounds_the_cancels_wait() -> None:
    """The lock was just handed to a waiter that has not run yet, so it reads
    as free: `/cancel`'s wait for it queues behind that waiter and still
    gives up at its deadline."""
    lock = asyncio.Lock()
    await lock.acquire()
    waiter = asyncio.ensure_future(lock.acquire())
    await asyncio.sleep(0)
    lock.release()
    assert not lock.locked()

    took = await asyncio.wait_for(
        studio_module._lock_or_claim(lock, asyncio.Event(), timeout=0.05), timeout=1
    )

    assert not took
    await waiter
    lock.release()


async def test_a_claim_ends_the_cancels_wait_for_a_lock_just_handed_to_a_waiter() -> None:
    """The lock was just handed to a waiting request, so it reads as free;
    that request's claim commits while `/cancel` waits behind it, and the
    wait ends on the claim, well before its deadline."""
    lock = asyncio.Lock()
    claimed = asyncio.Event()
    await lock.acquire()
    waiter = asyncio.ensure_future(lock.acquire())
    await asyncio.sleep(0)
    lock.release()
    assert not lock.locked()

    asyncio.get_running_loop().call_later(0.05, claimed.set)

    took = await asyncio.wait_for(studio_module._lock_or_claim(lock, claimed, timeout=5), timeout=1)

    assert not took
    await waiter
    lock.release()


async def test_a_cancel_whose_claim_keeps_losing_backs_off_until_its_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`/cancel` finds no live lease and its own claim loses each time: it
    decides again after a backoff, as the store's retries do, and answers
    the retryable 503 at `CANCEL_LOCK_WAIT_S`."""
    monkeypatch.setattr(studio_module, "CANCEL_LOCK_WAIT_S", 1.0)
    attempts: list[float] = []

    async def loses(*_args: Any, **_kwargs: Any) -> Any:
        attempts.append(time.monotonic())
        raise studio_module._ClaimLost

    monkeypatch.setattr(studio_module, "_stop_under_lock", loses)
    async with _client(_app()) as client:
        cancel = await _post_stop(client)

    assert cancel.status_code == 503
    assert cancel.json()["retry"] == "retry_after_delay"
    assert 2 <= len(attempts) <= 4


async def test_a_cancel_cancelled_after_it_took_the_lock_lets_the_lock_go() -> None:
    """`/cancel` waits for the conversation lock, the lock passes to it, and
    the request is cancelled before it runs again: the lock is released, so
    the conversation is not locked out."""
    lock = asyncio.Lock()
    await lock.acquire()
    waiting = asyncio.ensure_future(studio_module._lock_or_claim(lock, asyncio.Event(), timeout=5))
    await asyncio.sleep(0)
    lock.release()
    await asyncio.sleep(0)  # the lock passes to the wait, which has not run since
    waiting.cancel()

    with pytest.raises(asyncio.CancelledError):
        await waiting
    assert not lock.locked()


@pytest.mark.parametrize(
    ("stops", "deaf"),
    [(1, False), (2, True)],
    ids=["one-stop", "a-second-stop"],
)
async def test_a_stop_while_a_switch_runs_offers_the_plan_unchanged_on_a_live_pause(
    monkeypatch: pytest.MonkeyPatch, stops: int, deaf: bool
) -> None:
    """POST /cancel while the switch's revision runs: the stream reads on and
    ends on the same plan re-offered under a fresh interrupt, carrying
    `switch_failed` `switch_cancelled`, which asking again can get past; the
    turn stays paused on it. A switch that does not hear the Stop ends as any
    Stop ends a stream, on a second Stop."""
    entered = asyncio.Event()

    async def _revising(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        entered.set()
        await asyncio.sleep(5)
        raise AssertionError("a stopped revision is cancelled")

    async def _deaf(_ctx: Any, revision: Any, **_kwargs: Any) -> dict[str, Any]:
        # A switch that does not hear the first Stop.
        async with _ctx.stop.open():
            return await revision

    monkeypatch.setattr(switch_module, "_switched_plan", _revising)
    if deaf:
        monkeypatch.setattr(switch_module, "_unless_stopped", _deaf)
    app = _app(*_plan())
    taken = _StopsTaken(app.state.studio.store)
    stop = {"kind": "turn.cancel", "schema_version": 1, "conversation_id": CONVERSATION}
    async with _client(app) as client:
        plan = await _plan_with_a_choice(app, client)
        conversation = app.state.studio.store.current(PRINCIPAL, CONVERSATION)
        resumed = asyncio.ensure_future(
            _stream(client, "/api/studio/interrupt", _switch_body(plan))
        )
        await asyncio.wait_for(entered.wait(), timeout=5)
        for sent in range(1, stops + 1):
            cancelled = await client.post("/api/studio/cancel", json=stop, headers=AUTH)
            assert cancelled.status_code == 204
            await taken.reach(sent)
        frames = await asyncio.wait_for(resumed, timeout=5)

    if deaf:
        assert _events(frames)[-1] == "error"
        assert _frame(frames, "error")["error"]["error"] == "cancelled"
        assert not conversation.pending
        return
    assert _events(frames)[-1] == "plan.proposed"
    again = _frame(frames, "plan.proposed")
    assert again["interrupt_id"] != plan["interrupt_id"]
    assert again["revision"] == plan["revision"]
    assert again["switch_failed"] == {
        "choice_id": _CHOICE_ID,
        "option_id": "sector:120+130",
        "code": "switch_cancelled",
        "reason": "You stopped this change, so the plan is as it was.",
        "retryable": True,
    }
    assert list(conversation.pending) == [again["interrupt_id"]]
    assert conversation.awaiting_answer


_STOP = {"kind": "turn.cancel", "schema_version": 1, "conversation_id": CONVERSATION}


async def _stop_live() -> bool:
    """Whether a Stop is live for the conversation's turn, as a fresh
    instance reads it (`StudioStore.stop_live`)."""
    return await StudioStore(_engine()).stop_live(PRINCIPAL, CONVERSATION)


class _StopReaders:
    """Who acted on each Stop: the heartbeat (renewals reporting a new live
    stop id, `_StopsTaken`), whether it ended the turn or a node's
    `StopScope` took it; the end of the segment, counted as `release` (the
    stream's closing write closing the turn, `StudioStore.close_segment`);
    a claimant that found the Stop live and closed its turn
    (`_act_on_stop`); or `/cancel` under the lock (`_stop_open_turn`)."""

    def __init__(self, app: FastAPI, monkeypatch: pytest.MonkeyPatch) -> None:
        store = app.state.studio.store
        self.heartbeat = _StopsTaken(store)
        self.counts = {"release": 0, "claim": 0, "cancel": 0}
        close_segment = store.close_segment
        act_on_stop = studio_module._act_on_stop
        stop_open = studio_module._stop_open_turn

        async def counted_close_segment(*args: Any, **kwargs: Any) -> SegmentClose:
            result = await close_segment(*args, **kwargs)
            self.counts["release"] += result is SegmentClose.STOPPED
            return result

        async def counted_claim(*args: Any, **kwargs: Any) -> Any:
            refused = await act_on_stop(*args, **kwargs)
            if not await _stop_live():
                self.counts["claim"] += 1
            return refused

        async def counted_cancel(*args: Any, **kwargs: Any) -> Any:
            self.counts["cancel"] += 1
            return await stop_open(*args, **kwargs)

        monkeypatch.setattr(store, "close_segment", counted_close_segment)
        monkeypatch.setattr(studio_module, "_act_on_stop", counted_claim)
        monkeypatch.setattr(studio_module, "_stop_open_turn", counted_cancel)

    def read(self) -> dict[str, int]:
        counts = {"heartbeat": self.heartbeat.taken, **self.counts}
        return {party: count for party, count in counts.items() if count}


class _Row:
    """One row's instance: its app, the readers on it, and its clock."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.monkeypatch = monkeypatch
        self.app: FastAPI | None = None
        self.readers: _StopReaders | None = None
        self.clock: StoreClock | None = None

    async def build(self, *script: types.Part, clock: bool = False, **kwargs: Any) -> FastAPI:
        if clock:
            self.clock = await StoreClock.start(_engine())
            kwargs["store"] = StudioStore(_engine(), **self.clock.store_kwargs)
        self.app = _app(*script, **kwargs)
        self.readers = _StopReaders(self.app, self.monkeypatch)
        return self.app

    async def lapse(self) -> None:
        assert self.clock is not None
        await self.clock.advance(studio_state_module.SEGMENT_TTL_S + 1)


async def _wait_for_stream(app: FastAPI) -> None:
    studio = app.state.studio
    while not any(
        entry.state is not None and entry.state.renew_now is not None
        for entry in studio.store._runtime.values()
    ):
        await asyncio.sleep(0.01)


async def _turn_statuses() -> list[str]:
    async with _engine().connect() as conn:
        return list(
            (await conn.execute(text("SELECT status FROM turn_cursors ORDER BY id"))).scalars()
        )


async def _post_stop(client: httpx2.AsyncClient) -> httpx2.Response:
    return await client.post("/api/studio/cancel", json=_STOP, headers=AUTH)


async def _stop_while_a_node_runs(row: _Row):
    app = await row.build(*_card(), resolver=_HangingResolver(), heartbeat_seconds=0.1)
    async with _client(app) as client:
        stream = asyncio.ensure_future(_stream(client, "/api/studio/turn", _turn_body()))
        await asyncio.wait_for(_wait_for_stream(app), timeout=5)
        cancel = await _post_stop(client)
        return await asyncio.wait_for(stream, timeout=5), cancel


def _switch_held_by_a_stop(row: _Row, *, deaf: bool, ends: asyncio.Event | None = None) -> None:
    """The switch's revision hangs until cancelled, or, given `ends`, fails
    once it is set; a deaf switch holds the scope open without hearing its
    event, and sets `row.scope_closed` once the scope has closed."""
    entered = asyncio.Event()
    row.switch_entered = entered  # type: ignore[attr-defined]

    async def revising(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        entered.set()
        if ends is not None:
            await ends.wait()
            raise switch_module.SwitchFailedError("r", code="c")
        await asyncio.sleep(5)
        raise AssertionError("a stopped revision is cancelled")

    scopes: list[asyncio.Event] = []
    row.scopes = scopes  # type: ignore[attr-defined]
    closed = asyncio.Event()
    row.scope_closed = closed  # type: ignore[attr-defined]

    async def deaf_scope(_ctx: Any, revision: Any, **_kwargs: Any) -> dict[str, Any]:
        try:
            async with _ctx.stop.open() as stopped:
                scopes.append(stopped)
                return await revision
        finally:
            closed.set()

    row.monkeypatch.setattr(switch_module, "_switched_plan", revising)
    if deaf:
        row.monkeypatch.setattr(switch_module, "_unless_stopped", deaf_scope)


def _stop_while_a_switch_runs(stops: int):
    async def scenario(row: _Row):
        _switch_held_by_a_stop(row, deaf=stops == 2)
        app = await row.build(*_plan(), tool_surface=ScriptedSurface())
        async with _client(app) as client:
            plan = await _plan_with_a_choice(app, client)
            stream = asyncio.ensure_future(
                _stream(client, "/api/studio/interrupt", _switch_body(plan))
            )
            await asyncio.wait_for(row.switch_entered.wait(), timeout=5)  # type: ignore[attr-defined]
            for sent in range(1, stops + 1):
                cancel = await _post_stop(client)
                assert cancel.status_code == 204
                assert row.readers is not None
                await row.readers.heartbeat.reach(sent)
            return await asyncio.wait_for(stream, timeout=5), cancel

    return scenario


async def _stop_taken_by_the_scope_then_the_client_leaves(row: _Row):
    """The scope takes the first Stop and the client leaves before the
    turn's pause write records it taken: the Stop is still live, so the
    release closes the turn `cancelled` under the lease it keeps."""
    _switch_held_by_a_stop(row, deaf=True)
    app = await row.build(*_plan(), _text("Hello."))
    async with _client(app) as client:
        plan = await _plan_with_a_choice(app, client)
        stream = asyncio.ensure_future(_stream(client, "/api/studio/interrupt", _switch_body(plan)))
        await asyncio.wait_for(row.switch_entered.wait(), timeout=5)  # type: ignore[attr-defined]
        cancel = await _post_stop(client)
        assert row.readers is not None
        await row.readers.heartbeat.reach(1)
        await asyncio.wait_for(row.scopes[0].wait(), timeout=5)  # type: ignore[attr-defined]
        stream.cancel()
        with suppress(asyncio.CancelledError):
            await stream
        while app.state.studio.store.lock_for(PRINCIPAL, CONVERSATION).locked():
            await asyncio.sleep(0.01)
        return await _stream(client, "/api/studio/turn", _turn_body("hi")), cancel


async def _second_stop_after_the_pause_write(row: _Row):
    """The scope takes the first Stop, the turn pauses on the re-offered
    plan, recording it taken, and a second Stop commits before the closing
    write: that write closes the turn `cancelled`, and the stream ends on
    `cancelled` alone, the re-offered plan never sent."""
    _switch_held_by_a_stop(row, deaf=False)
    app = await row.build(*_plan())
    store = app.state.studio.store
    sent: list[httpx2.Response] = []
    async with _client(app) as client:
        plan = await _plan_with_a_choice(app, client)
        real = store.close_segment

        async def second_stop_first(*args: Any, **kwargs: Any) -> Any:
            if not sent:
                sent.append(await _post_stop(client))
            return await real(*args, **kwargs)

        stream = asyncio.ensure_future(_stream(client, "/api/studio/interrupt", _switch_body(plan)))
        await asyncio.wait_for(row.switch_entered.wait(), timeout=5)  # type: ignore[attr-defined]
        first = await _post_stop(client)
        assert first.status_code == 204
        assert row.readers is not None
        await row.readers.heartbeat.reach(1)
        store.close_segment = second_stop_first
        frames = await asyncio.wait_for(stream, timeout=5)
    assert "plan.proposed" not in _events(frames)
    return frames, sent[0]


async def _stop_whose_pause_write_fails(row: _Row):
    """The scope takes the first Stop, and the pause write that would
    record it taken fails: the drive closes the turn `failed`, which ends
    the Stop."""
    _switch_held_by_a_stop(row, deaf=False)
    app = await row.build(*_plan())
    turns = app.state.studio.turns
    real_pause = turns.pause

    async def pause(cursor: Any, **kwargs: Any) -> Any:
        if kwargs.get("stop_taken_id") is not None:
            raise ConnectionError("the database is unreachable")
        return await real_pause(cursor, **kwargs)

    async with _client(app) as client:
        plan = await _plan_with_a_choice(app, client)
        turns.pause = pause
        stream = asyncio.ensure_future(_stream(client, "/api/studio/interrupt", _switch_body(plan)))
        await asyncio.wait_for(row.switch_entered.wait(), timeout=5)  # type: ignore[attr-defined]
        cancel = await _post_stop(client)
        return await asyncio.wait_for(stream, timeout=5), cancel


async def _stop_after_the_claim_before_the_abandon(row: _Row):
    """`/turn` commits its claim over a turn left paused, and a Stop lands on
    the new lease before the claim abandons the old turn: the old turn ends
    `abandoned`, and the Stop, which names the new turn, ends it
    `cancelled`."""
    app = await row.build(*_plan(), *_card(), resolver=_HangingResolver(), heartbeat_seconds=0.1)
    sent: list[httpx2.Response] = []
    async with _client(app) as client:
        await _stream(client, "/api/studio/turn", _turn_body("q with forks"))
        real = studio_module.abandon

        async def stopped_first(*args: Any, **kwargs: Any) -> Any:
            sent.append(await _post_stop(client))
            return await real(*args, **kwargs)

        row.monkeypatch.setattr(studio_module, "abandon", stopped_first)
        frames = await asyncio.wait_for(
            _stream(client, "/api/studio/turn", _turn_body()), timeout=5
        )
    return frames, sent[0]


async def _stop_whose_claimant_cannot_close_the_turn(row: _Row):
    """A Stop left live on a paused turn whose lease lapsed. The next
    `/turn` acts on it first, but its close is refused (another writer moved
    the cursor), so that `/turn` answers a retryable 409 and starts nothing;
    its retry acts on the Stop and runs the new turn."""
    app = await row.build(*_plan(), _text("Hello."), clock=True)
    store = app.state.studio.store
    sent: list[httpx2.Response] = []
    async with _client(app) as client:
        real_close = store.close_segment

        async def stop_then_fail(*args: Any, **kwargs: Any) -> Any:
            if not sent:
                sent.append(await _post_stop(client))
                return SegmentClose.FAILED
            return await real_close(*args, **kwargs)

        store.close_segment = stop_then_fail
        await _stream(client, "/api/studio/turn", _turn_body("q with forks"))
        assert await _stop_live()
        await row.lapse()

        async def moved(*_args: Any, **_kwargs: Any) -> Any:
            raise CursorStale("another writer moved the cursor")

        real_stop_turn = studio_module.stop_turn
        row.monkeypatch.setattr(studio_module, "stop_turn", moved)
        busy = await client.post("/api/studio/turn", json=_turn_body("hi"), headers=AUTH)
        assert busy.status_code == 409
        assert busy.json()["retry"] == "retry_after_delay"
        assert await _turn_statuses() == ["paused"]
        assert await _stop_live()
        row.monkeypatch.setattr(studio_module, "stop_turn", real_stop_turn)
        return await _stream(client, "/api/studio/turn", _turn_body("hi")), sent[0]


async def _stop_whose_claim_commits_as_the_lease_is_read(row: _Row):
    """`/cancel` finds no live lease while a `/turn` on this instance holds
    the lock, and that `/turn`'s claim commits before `/cancel` starts to
    wait: the wait sees the claim, and the Stop is recorded on it."""
    row.monkeypatch.setattr(studio_module, "CANCEL_LOCK_WAIT_S", 1.0)
    app = await row.build(*_card(), resolver=_HangingResolver(), heartbeat_seconds=0.1)
    store = app.state.studio.store
    real_save, real_request = store.save, store.request_cancel
    entered, go, saved = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def held(*args: Any, **kwargs: Any) -> Any:
        if entered.is_set():
            return await real_save(*args, **kwargs)
        entered.set()
        await go.wait()
        result = await real_save(*args, **kwargs)
        saved.set()
        return result

    async def claimed_after_the_read(*args: Any, **kwargs: Any) -> bool:
        found = await real_request(*args, **kwargs)
        if not go.is_set():
            go.set()
            await saved.wait()
        return found

    store.save = held
    store.request_cancel = claimed_after_the_read
    async with _client(app) as client:
        stream = asyncio.ensure_future(_stream(client, "/api/studio/turn", _turn_body()))
        await asyncio.wait_for(entered.wait(), timeout=5)
        cancel = await _post_stop(client)
        return await asyncio.wait_for(stream, timeout=5), cancel


async def _stop_in_after_commit_work(row: _Row):
    entered = asyncio.Event()
    real = turn_loop_module._after_commit

    async def held(ctx: Any, node: str) -> None:
        if (await _turn_statuses())[-1] == "paused":
            entered.set()
            await asyncio.sleep(5)
        await real(ctx, node)

    row.monkeypatch.setattr(turn_loop_module, "_after_commit", held)
    app = await row.build(*_plan())
    async with _client(app) as client:
        stream = asyncio.ensure_future(
            _stream(client, "/api/studio/turn", _turn_body("q with forks"))
        )
        await asyncio.wait_for(entered.wait(), timeout=5)
        cancel = await _post_stop(client)
        return await asyncio.wait_for(stream, timeout=5), cancel


def _stop_before_the_release(message: str, *, contested: bool = False):
    """The Stop commits after the stream's last renewal and before its
    closing write. `contested`: before that write closes the turn under the
    lease, another instance's answer to the card is refused as busy."""

    async def scenario(row: _Row):
        script = (_text("Hello."),) if message == "hi" else _plan()
        app = await row.build(*script)
        store = app.state.studio.store
        sent: list[httpx2.Response] = []
        async with _client(app) as client:
            real = store.close_segment
            other = _app()

            async def stopped_first(*args: Any, **kwargs: Any) -> Any:
                if not sent:
                    sent.append(await _post_stop(client))
                if contested:
                    plan_id = store.current(PRINCIPAL, CONVERSATION).pending
                    async with _client(other) as other_client:
                        for interrupt_id in list(plan_id):
                            busy = await other_client.post(
                                "/api/studio/interrupt",
                                json=_interrupt_body(interrupt_id, "approve"),
                                headers=AUTH,
                            )
                            assert busy.status_code == 409
                return await real(*args, **kwargs)

            store.close_segment = stopped_first
            frames = await _stream(client, "/api/studio/turn", _turn_body(message))
        return frames, sent[0]

    return scenario


async def _stop_whose_release_cannot_close_the_running_turn(row: _Row):
    """A node fails and its close is not written, so the turn is left
    running; the Stop commits before the closing write, and ending the
    turn's runs inside that write fails. The stream ends on the node's
    error frame alone, the lease lapses with the Stop live, and the next
    turn acts on it first."""

    class BrokenSurface(ScriptedSurface):
        async def call_tool(self, **kwargs: Any) -> ToolCallResult:
            raise RuntimeError("analysis mount went away")

    app = await row.build(*_plan(), _text("Hello."), tool_surface=BrokenSurface(), clock=True)
    studio = app.state.studio
    real_finish = studio.turns.finish

    async def finish_unwritten(cursor: Any, **kwargs: Any) -> Any:
        if kwargs.get("status") is TurnStatus.FAILED:
            raise ConnectionError("the database is unreachable")
        return await real_finish(cursor, **kwargs)

    end_unfinished = turn_store_module.end_unfinished
    failures = [RuntimeError("the run ledger is unreachable")]

    async def ends_once(*args: Any, **kwargs: Any) -> None:
        if failures:
            raise failures.pop()
        await end_unfinished(*args, **kwargs)

    row.monkeypatch.setattr(turn_store_module, "end_unfinished", ends_once)
    async with _client(app) as client:
        plan = _frame(
            await _stream(client, "/api/studio/turn", _turn_body("q with forks")), "plan.proposed"
        )
        studio.turns.finish = finish_unwritten
        real_close = studio.store.close_segment
        sent: list[httpx2.Response] = []

        async def stopped_first(*args: Any, **kwargs: Any) -> Any:
            if not sent:
                sent.append(await _post_stop(client))
            return await real_close(*args, **kwargs)

        studio.store.close_segment = stopped_first
        stopped = await _stream(
            client, "/api/studio/interrupt", _interrupt_body(plan["interrupt_id"], "approve")
        )
        # The closing write did not commit, so the turn stays open with its
        # Stop live, and the stream ends on the node's own error frame.
        assert stopped[-1]["error"]["error"] == "capability_unavailable"
        assert await _stop_live()
        await row.lapse()
        return await _stream(client, "/api/studio/turn", _turn_body("hi")), sent[0]


async def _stop_whose_release_write_fails(row: _Row):
    """The Stop commits before the closing write, and that write never
    lands. The lease lapses with the Stop live, and the next turn acts on
    it first."""
    app = await row.build(*_plan(), _text("Hello."), clock=True)
    store = app.state.studio.store
    sent: list[httpx2.Response] = []
    async with _client(app) as client:
        real_close = store.close_segment

        async def stop_then_fail(*args: Any, **kwargs: Any) -> Any:
            if not sent:
                sent.append(await _post_stop(client))
                return SegmentClose.FAILED
            return await real_close(*args, **kwargs)

        store.close_segment = stop_then_fail
        await _stream(client, "/api/studio/turn", _turn_body("q with forks"))
        assert await _stop_live()
        await row.lapse()
        return await _stream(client, "/api/studio/turn", _turn_body("hi")), sent[0]


def _stop_with_no_lease(running: bool):
    async def scenario(row: _Row):
        app = await row.build(*_plan())
        async with _client(app) as client:
            frames = await _stream(client, "/api/studio/turn", _turn_body("q with forks"))
            if running:
                async with _engine().begin() as conn:
                    await conn.execute(
                        text(
                            "UPDATE turn_cursors SET status = 'running', pause_kind = NULL, "
                            "interrupt_id = NULL, pause_payload = NULL"
                        )
                    )
            return frames, await _post_stop(client)

    return scenario


async def _stop_after_the_turn_ended(row: _Row):
    app = await row.build(_text("Hello."))
    async with _client(app) as client:
        frames = await _stream(client, "/api/studio/turn", _turn_body("hi"))
        return frames, await _post_stop(client)


async def _stop_before_the_claim_save(row: _Row):
    app = await row.build(*_card(), resolver=_HangingResolver(), heartbeat_seconds=0.1)
    store = app.state.studio.store
    real = store.claim_turn_lease
    entered, go = asyncio.Event(), asyncio.Event()

    async def held(*args: Any, **kwargs: Any) -> Any:
        if not entered.is_set():
            entered.set()
            await go.wait()
        return await real(*args, **kwargs)

    store.claim_turn_lease = held  # type: ignore[method-assign]
    async with _client(app) as client:
        stream = asyncio.ensure_future(_stream(client, "/api/studio/turn", _turn_body()))
        await asyncio.wait_for(entered.wait(), timeout=5)
        cancel_task = asyncio.ensure_future(_post_stop(client))
        await asyncio.sleep(0.2)
        assert not cancel_task.done()  # it waits on the claim
        go.set()
        cancel = await asyncio.wait_for(cancel_task, timeout=5)
        return await asyncio.wait_for(stream, timeout=5), cancel


async def _stop_whose_claim_is_abandoned(row: _Row):
    """`/cancel` waits on an `/interrupt` that holds the lock, the Stop lands
    on the claim, and the `/interrupt` then abandons its claim (the pause
    moved). The Stop stays live, and the next turn acts on it first."""
    app = await row.build(*_plan(), _text("Hello."))
    studio = app.state.studio
    store = studio.store
    async with _client(app) as client:
        plan = _frame(
            await _stream(client, "/api/studio/turn", _turn_body("q with forks")), "plan.proposed"
        )
        real_save, real_open = store.save, studio.turns.open_cursor
        entered, go, stopped = asyncio.Event(), asyncio.Event(), asyncio.Event()
        opened = 0

        async def held_save(*args: Any, **kwargs: Any) -> Any:
            if not entered.is_set():
                entered.set()
                await go.wait()
                async with _engine().begin() as conn:
                    await conn.execute(text("UPDATE turn_cursors SET interrupt_id = 'int_moved'"))
            return await real_save(*args, **kwargs)

        async def open_after_the_stop(*args: Any, **kwargs: Any) -> Any:
            nonlocal opened
            opened += 1
            if opened == 2:
                await stopped.wait()
            return await real_open(*args, **kwargs)

        store.save = held_save
        studio.turns.open_cursor = open_after_the_stop
        answer = asyncio.ensure_future(
            client.post(
                "/api/studio/interrupt",
                json=_interrupt_body(plan["interrupt_id"], "approve"),
                headers=AUTH,
            )
        )
        await asyncio.wait_for(entered.wait(), timeout=5)
        cancel_task = asyncio.ensure_future(_post_stop(client))
        await asyncio.sleep(0.2)
        assert not cancel_task.done()  # it waits on the claim
        go.set()
        cancel = await asyncio.wait_for(cancel_task, timeout=5)
        stopped.set()
        assert (await asyncio.wait_for(answer, timeout=5)).status_code == 409
        store.save, studio.turns.open_cursor = real_save, real_open
        assert await _stop_live()
        return await _stream(client, "/api/studio/turn", _turn_body("hi")), cancel


@pytest.mark.parametrize(
    ("scenario", "turns_after", "last_frame", "readers"),
    [
        pytest.param(
            _stop_while_a_node_runs,
            ["cancelled"],
            "cancelled",
            {"heartbeat": 1},
            id="lease-live-node-running",
        ),
        pytest.param(
            _stop_while_a_switch_runs(1),
            ["paused"],
            "plan.proposed",
            {"heartbeat": 1},
            id="lease-live-scope-armed-first-stop",
        ),
        pytest.param(
            _stop_while_a_switch_runs(2),
            ["cancelled"],
            "cancelled",
            {"heartbeat": 2},
            id="lease-live-scope-armed-second-stop",
        ),
        pytest.param(
            _second_stop_after_the_pause_write,
            ["cancelled"],
            "cancelled",
            {"heartbeat": 1, "release": 1},
            id="lease-live-scope-armed-second-stop-after-the-pause-write",
        ),
        pytest.param(
            _stop_taken_by_the_scope_then_the_client_leaves,
            ["cancelled", "done"],
            "done",
            {"heartbeat": 1, "release": 1},
            id="lease-live-scope-took-it-then-the-client-left",
        ),
        pytest.param(
            _stop_whose_pause_write_fails,
            ["failed"],
            "transient_unavailable",
            {"heartbeat": 1},
            id="lease-live-scope-took-it-pause-write-fails",
        ),
        pytest.param(
            _stop_in_after_commit_work,
            ["cancelled"],
            "cancelled",
            {"heartbeat": 1},
            id="lease-live-pause-committed-after-commit-work",
        ),
        pytest.param(
            _stop_before_the_release("q with forks"),
            ["cancelled"],
            "cancelled",
            {"release": 1},
            id="lease-live-segment-returned-paused",
        ),
        pytest.param(
            _stop_before_the_release("q with forks", contested=True),
            ["cancelled"],
            "cancelled",
            {"release": 1},
            id="lease-live-segment-returned-paused-another-claim-refused",
        ),
        pytest.param(
            _stop_before_the_release("hi"),
            ["done"],
            "done",
            {},
            id="lease-live-turn-closed",
        ),
        pytest.param(
            _stop_after_the_claim_before_the_abandon,
            ["abandoned", "cancelled"],
            "cancelled",
            {"heartbeat": 1},
            id="claim-committed-before-the-abandon",
        ),
        pytest.param(
            _stop_whose_release_cannot_close_the_running_turn,
            ["cancelled", "done"],
            "done",
            {"claim": 1},
            id="lease-kept-closing-fails-next-claim-acts",
        ),
        pytest.param(
            _stop_whose_release_write_fails,
            ["cancelled", "done"],
            "done",
            {"claim": 1},
            id="release-write-fails-next-claim-acts",
        ),
        pytest.param(
            _stop_whose_claimant_cannot_close_the_turn,
            ["cancelled", "done"],
            "done",
            {"claim": 1},
            id="claimant-close-refused-turn-refused-retry-acts",
        ),
        pytest.param(
            _stop_with_no_lease(running=False),
            ["cancelled"],
            "plan.proposed",
            {"cancel": 1},
            id="no-lease-turn-paused",
        ),
        pytest.param(
            _stop_with_no_lease(running=True),
            ["cancelled"],
            "plan.proposed",
            {"cancel": 1},
            id="no-lease-turn-left-running",
        ),
        pytest.param(
            _stop_after_the_turn_ended,
            ["done"],
            "done",
            {"cancel": 1},
            id="no-lease-no-open-turn",
        ),
        pytest.param(
            _stop_before_the_claim_save,
            ["cancelled"],
            "cancelled",
            {"heartbeat": 1},
            id="lock-held-before-the-claim-save",
        ),
        pytest.param(
            _stop_whose_claim_commits_as_the_lease_is_read,
            ["cancelled"],
            "cancelled",
            {"heartbeat": 1},
            id="lock-held-claim-commits-as-the-lease-is-read",
        ),
        pytest.param(
            _stop_whose_claim_is_abandoned,
            ["cancelled", "done"],
            "done",
            {"claim": 1},
            id="lock-held-claim-abandoned-next-claim-acts",
        ),
    ],
)
async def test_each_stop_is_read_by_exactly_one_party(
    monkeypatch: pytest.MonkeyPatch,
    scenario: Any,
    turns_after: list[str],
    last_frame: str,
    readers: dict[str, int],
) -> None:
    """One row per row of `/cancel`'s table: the Stop, posted through the
    route at the named point, is acted on by the party the row names, every
    turn ends as the row says, the last stream's last frame is the row's,
    and no Stop is left live."""
    row = _Row(monkeypatch)
    frames, cancel = await scenario(row)

    assert cancel.status_code == 204
    assert await _turn_statuses() == turns_after
    last = frames[-1]
    assert (last["error"]["error"] if last["event"] == "error" else last["event"]) == last_frame
    assert row.readers is not None
    assert row.readers.read() == readers
    assert not await _stop_live()
    if turns_after[0] == "cancelled" and "plan.proposed" in _events(frames):
        assert row.app is not None
        async with _client(row.app) as client:
            plan = _frame(frames, "plan.proposed")
            refused = await client.post(
                "/api/studio/interrupt",
                json=_interrupt_body(plan["interrupt_id"], "approve"),
                headers=AUTH,
            )
        assert refused.status_code == 404
    if turns_after == ["paused"]:
        # The re-offered card is live: an approve of it runs the plan as it
        # stood before the switch.
        assert row.app is not None
        reoffered = _frame(frames, "plan.proposed")
        async with _client(row.app) as client:
            approved = await _stream(
                client,
                "/api/studio/interrupt",
                _interrupt_body(reoffered["interrupt_id"], "approve"),
            )
        assert _events(approved)[-1] == "done"
        assert await _turn_statuses() == ["done"]


async def test_a_stop_that_lands_at_the_claim_save_closes_the_turn_and_the_retry_streams() -> None:
    """A Stop lands on the turn left paused after `/turn` found none live and
    before its claim save: the save refuses, `/turn` closes the paused turn
    `cancelled` under a claim of its own and answers a retryable 409, the
    stopped card's answer is refused, and the retried `/turn` streams."""
    app = _app(*_plan(), _text("Hello."))
    store = app.state.studio.store
    real_claim = store.claim_turn_lease
    landed: list[bool] = []

    async def stop_lands_first(*args: Any, **kwargs: Any) -> Any:
        if not landed:
            # The claim of the second `/turn`.
            landed.append(True)
            async with _engine().begin() as conn:
                await conn.execute(text("UPDATE conversations SET lease_cancel = true"))
        return await real_claim(*args, **kwargs)

    async with _client(app) as client:
        plan = _frame(
            await _stream(client, "/api/studio/turn", _turn_body("q with forks")), "plan.proposed"
        )
        store.claim_turn_lease = stop_lands_first  # type: ignore[method-assign]
        refused = await client.post("/api/studio/turn", json=_turn_body("hi"), headers=AUTH)
        assert landed
        stopped = await _turn_statuses()
        answered = await client.post(
            "/api/studio/interrupt",
            json=_interrupt_body(plan["interrupt_id"], "approve"),
            headers=AUTH,
        )
        retried = await _stream(client, "/api/studio/turn", _turn_body("hi"))

    assert refused.status_code == 409
    assert refused.json()["retry"] == "retry_after_delay"
    assert stopped == ["cancelled"]
    assert answered.status_code == 404
    assert _events(retried)[-1] == "done"
    assert await _turn_statuses() == ["cancelled", "done"]
    assert not await _stop_live()


async def test_a_cancel_whose_claim_loses_to_another_instance_records_the_stop_on_its_lease() -> (
    None
):
    """`/cancel` finds no live lease and claims to close the paused turn
    itself, but another instance claims the conversation first: `/cancel`
    decides again, finds that instance's live lease and records the Stop on
    it, so the 204 it answers names a Stop the lease's holder acts on."""
    app = _app(*_plan())
    store = app.state.studio.store
    real_save = store.save
    other = StudioStore(_engine())

    async def another_claims_first(*args: Any, **kwargs: Any) -> Any:
        if kwargs.get("stopping"):
            await other.load(PRINCIPAL, CONVERSATION)
            other.claim_segment(PRINCIPAL, CONVERSATION)
            await other.save(PRINCIPAL, CONVERSATION)
        return await real_save(*args, **kwargs)

    async with _client(app) as client:
        await _stream(client, "/api/studio/turn", _turn_body("q with forks"))
        store.save = another_claims_first
        started = time.monotonic()
        cancel = await _post_stop(client)
        elapsed = time.monotonic() - started

    assert cancel.status_code == 204
    # The winner's lease was there to record the Stop on, so no backoff ran.
    assert elapsed < studio_store_module.retry_delays().__next__()
    async with _engine().connect() as conn:
        flagged = (await conn.execute(text("SELECT lease_cancel FROM conversations"))).scalar_one()
    assert flagged
    assert await _turn_statuses() == ["paused"]


@pytest.mark.parametrize("fails", ["write", "turn"], ids=["write-times-out", "turn-close-fails"])
async def test_a_closing_write_that_fails_under_a_stop_leaves_the_stop_for_the_next_claim(
    monkeypatch: pytest.MonkeyPatch, fails: str
) -> None:
    """A Stop commits before the closing write of a stream paused on a plan,
    and that write fails: it times out on a row another connection holds,
    or closing the turn's cursor inside it fails. Nothing of it commits, so
    the stream ends on the transient frame, the turn's cursor stays paused
    with its Stop live, the record stays open with no pause stored to
    answer, and once the lease lapses
    the next `/turn` closes it `cancelled` before it streams."""
    clock = await StoreClock.start(_engine())
    store = StudioStore(_engine(), **clock.store_kwargs)
    app = _app(*_plan(), _text("Hello."), store=store)
    real_close = store.close_segment
    real_finish_on = studio_store_module.finish_on
    sent: list[httpx2.Response] = []

    async def finish_on(*args: Any, **kwargs: Any) -> Any:
        if kwargs.get("status") is TurnStatus.CANCELLED and len(sent) == 1:
            raise ConnectionError("the database is unreachable")
        return await real_finish_on(*args, **kwargs)

    monkeypatch.setattr(studio_store_module, "finish_on", finish_on)
    async with _client(app) as client:

        async def stopped_first(*args: Any, **kwargs: Any) -> SegmentClose:
            if sent:
                return await real_close(*args, **kwargs)
            sent.append(await _post_stop(client))
            if fails == "turn":
                return await real_close(*args, **kwargs)
            async with _engine().begin() as conn:
                await conn.execute(text("SELECT id FROM conversations FOR UPDATE"))
                return await real_close(*args, timeout=0.3)

        store.close_segment = stopped_first  # type: ignore[method-assign]
        frames = await _stream(client, "/api/studio/turn", _turn_body("q with forks"))
        sent.append(sent[0])
        live = await _stop_live()
        statuses = await _turn_statuses()
        stored = await StudioStore(_engine()).load(PRINCIPAL, CONVERSATION)
        assert stored is not None
        await clock.advance(studio_state_module.SEGMENT_TTL_S + 1)
        after = await _stream(client, "/api/studio/turn", _turn_body("hi"))

    assert sent[0].status_code == 204
    assert "plan.proposed" not in _events(frames)
    assert _frame(frames, "error")["error"]["error"] == "transient_unavailable"
    assert live
    assert statuses == ["paused"]
    assert not stored.turn_recorded
    assert not stored.pending
    assert _events(after)[-1] == "done"
    assert await _turn_statuses() == ["cancelled", "done"]
    assert store.current(PRINCIPAL, CONVERSATION).turns[0]["error"] == "cancelled"
    assert not await _stop_live()


@pytest.mark.parametrize("fails", ["record", "turn"], ids=["record-save-fails", "turn-close-fails"])
async def test_a_cancel_whose_close_fails_answers_503_and_leaves_the_stop_live(
    fails: str,
) -> None:
    """`/cancel` finds no live lease on a paused turn and one of the two
    writes that close it fails: it answers a retryable 503 and leaves a Stop
    live, so the next `/turn` closes the turn `cancelled` before it streams.
    The instance's live state changes only with a save that commits."""
    app = _app(*_plan(), _text("Hello."))
    store, turns = app.state.studio.store, app.state.studio.turns
    real_save, real_finish = store.save, turns.finish
    claimed = [False]

    async def save(*args: Any, **kwargs: Any) -> Any:
        if kwargs.get("stopping"):
            claimed[0] = True
        elif fails == "record" and claimed[0]:
            claimed[0] = False
            raise ConnectionError("the database is unreachable")
        return await real_save(*args, **kwargs)

    async def finish(cursor: Any, **kwargs: Any) -> Any:
        if fails == "turn" and claimed[0] and kwargs.get("status") is TurnStatus.CANCELLED:
            claimed[0] = False
            raise ConnectionError("the database is unreachable")
        return await real_finish(cursor, **kwargs)

    async with _client(app) as client:
        await _stream(client, "/api/studio/turn", _turn_body("q with forks"))
        store.save, turns.finish = save, finish
        refused = await _post_stop(client)
        store.save, turns.finish = real_save, real_finish
        held = bool(store.current(PRINCIPAL, CONVERSATION).pending)
        live = await _stop_live()
        statuses = await _turn_statuses()
        after = await _stream(client, "/api/studio/turn", _turn_body("hi"))

    assert refused.status_code == 503
    assert refused.json()["retry"] == "retry_after_delay"
    assert held is (fails == "record")
    assert live
    assert statuses == ["paused"]
    assert _events(after)[-1] == "done"
    assert await _turn_statuses() == ["cancelled", "done"]
    assert store.current(PRINCIPAL, CONVERSATION).turns[0]["error"] == "cancelled"


async def test_a_segment_left_unclosed_on_a_taken_stop_ends_its_stream_cancelled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A segment whose close on the Stop its `StopScope` took could not be
    written ends `unclosed` with no `TurnFailed`, the turn open with that
    Stop live: the stream still ends on a terminal `cancelled` frame."""

    async def unclosed(_store: Any, _lease: Any, cursor: Any, **_kwargs: Any) -> Any:
        return turn_loop_module.Segment(cursor=cursor, outcome=turn_loop_module.Outcome.UNCLOSED)

    monkeypatch.setattr(studio_module, "run_segment", unclosed)
    async with _client(_app()) as client:
        frames = await _stream(client, "/api/studio/turn", _turn_body("hi"))

    assert frames[-1]["event"] == "error"
    assert frames[-1]["error"]["error"] == "cancelled"


class _HangingEditModel:
    """An Edit Card model whose call waits until it is cancelled."""

    def __init__(self) -> None:
        self.entered = asyncio.Event()
        self.cancelled = False

    async def __call__(self, **kwargs: Any) -> dict[str, Any]:
        self.entered.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.cancelled = True
            raise
        raise AssertionError("unreachable")


async def test_a_stop_during_an_edit_card_stream_ends_it_cancelled_with_nothing_left_running() -> (
    None
):
    """The Edit Card stream opens no turn on the loop, so its claim leaves the
    conversation's turn index with no cursor. A Stop during it is live for
    that turn while the lease is: the stream's heartbeat ends it on a
    `cancelled` frame, the edit's model call is cancelled, the lease is
    released, no Stop is left live, and the next turn streams."""
    executor = ScriptedExecutor([*_card_results(write_token=WRITE_TOKEN), _described(), _VIEW])
    edit_model = _HangingEditModel()
    app = _app(
        *_card(),
        _text("Hello."),
        resolver=_search(),
        executor=executor,
        edit_model=edit_model,
        heartbeat_seconds=0.1,
    )
    async with _client(app) as client:
        await _conversation_with_component(client)
        stream = asyncio.ensure_future(_stream(client, "/api/studio/turn", _edit_body("rename it")))
        await asyncio.wait_for(edit_model.entered.wait(), timeout=5)
        cancel = await _post_stop(client)
        frames = await asyncio.wait_for(stream, timeout=5)
        lock = app.state.studio.store.lock_for(PRINCIPAL, CONVERSATION)
        async with _engine().connect() as conn:
            lease = (
                await conn.execute(text("SELECT lease_segment_id FROM conversations"))
            ).scalar_one()
        live = await _stop_live()
        after = await _stream(client, "/api/studio/turn", _turn_body("hi"))

    assert cancel.status_code == 204
    assert _events(frames)[-1] == "error"
    assert _frame(frames, "error")["error"]["error"] == "cancelled"
    assert edit_model.cancelled
    assert not lock.locked()
    assert lease is None
    assert not live
    assert _events(after)[-1] == "done"
    assert await _turn_statuses() == ["done", "done"]


@pytest.mark.parametrize("stops", [1, 2], ids=["one-stop", "a-second-stop"])
async def test_a_stop_from_another_instance_stops_a_switch_and_a_second_ends_its_stream(
    monkeypatch: pytest.MonkeyPatch, stops: int
) -> None:
    """B POSTs `/cancel` while A's switch runs and the switch does not hear it:
    A's heartbeat reads B's durable flag as the Stop of the switch and keeps
    watching, so a second Stop from B ends A's stream with `cancelled`. One
    Stop alone lets the stream read on."""
    entered = asyncio.Event()
    finish = asyncio.Event()

    async def _revising(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        entered.set()
        await finish.wait()
        raise switch_module.SwitchFailedError("r", code="c")

    scopes: list[asyncio.Event] = []

    async def _deaf(_ctx: Any, revision: Any, **_kwargs: Any) -> dict[str, Any]:
        async with _ctx.stop.open() as stopped:
            scopes.append(stopped)
            return await revision

    monkeypatch.setattr(switch_module, "_switched_plan", _revising)
    monkeypatch.setattr(switch_module, "_unless_stopped", _deaf)
    app_a = _app(*_plan(), heartbeat_seconds=0.05)
    app_b = _app()
    taken = _StopsTaken(app_a.state.studio.store)
    stop = {"kind": "turn.cancel", "schema_version": 1, "conversation_id": CONVERSATION}
    async with _client(app_a) as client_a, _client(app_b) as client_b:
        plan = await _plan_with_a_choice(app_a, client_a)
        resumed = asyncio.ensure_future(
            _stream(client_a, "/api/studio/interrupt", _switch_body(plan))
        )
        await asyncio.wait_for(entered.wait(), timeout=5)
        for sent in range(1, stops + 1):
            cancelled = await client_b.post("/api/studio/cancel", json=stop, headers=AUTH)
            assert cancelled.status_code == 204
            await taken.reach(sent)
        if stops == 1:
            await asyncio.wait_for(scopes[0].wait(), timeout=5)  # the scope took it
            finish.set()
        frames = await asyncio.wait_for(resumed, timeout=5)

    if stops == 1:
        assert _events(frames)[-1] == "plan.proposed"
        return
    assert _events(frames)[-1] == "error"
    assert _frame(frames, "error")["error"]["error"] == "cancelled"


_SECTOR_LABELS = ("health (CRS 120)", "health and population", "population (CRS 130)")


@pytest.mark.parametrize(
    ("choice_id", "option_ids"),
    [
        pytest.param(_CHOICE_ID, ["sector:120", "sector:120+130", "sector:130"], id="scope-choice"),
        pytest.param(
            "decision/sector",
            [decision_option_id(label) for label in _SECTOR_LABELS],
            id="planner-decision",
        ),
    ],
)
async def test_an_approved_plan_names_what_a_choice_held_before_the_turns_first_switch(
    choice_id: str, option_ids: list[str]
) -> None:
    """Health counted as CRS 120, switched to CRS 120 and 130, then to CRS
    130, then approved: `plan.approved` names the choice by its id, as the
    card did (`decision/<slot>` for a planner decision), and CRS 120, the
    label on the card the first switch was made from, as what it changed
    from."""
    labels = dict(zip(option_ids, _SECTOR_LABELS, strict=True))

    async def _shows(app: FastAPI, interrupt_id: str, active: str) -> None:
        # The card the pause holds shows the sector choice holding `active`.
        conversation = app.state.studio.store.current(PRINCIPAL, CONVERSATION)
        pending = conversation.pending[interrupt_id]
        decision = PlanDecision(
            id="scope_sector",
            question="Sector",
            chosen=labels[active],
            reason="",
            resolved_by="user",
            choice_id=choice_id,
            active_option_id=active,
            options=[
                PlanDecisionOption(option_id=option, label=label, status="available")
                for option, label in labels.items()
            ],
        )
        assert pending.card is not None
        pending.card = pending.card.model_copy(update={"decisions": [decision]})
        pending.choices = {
            choice_id: _PendingChoice(
                active=active,
                options={
                    o: _PendingOption(label=lb, status="available") for o, lb in labels.items()
                },
            )
        }
        await app.state.studio.store.save(PRINCIPAL, CONVERSATION)

    app = _app(*_plan(), tool_surface=ScriptedSurface())
    async with _client(app) as client:
        card = _frame(await _stream(client, "/api/studio/turn", _turn_body()), "plan.proposed")
        await _shows(app, card["interrupt_id"], option_ids[0])
        for option in option_ids[1:]:
            frames = await _stream(
                client,
                "/api/studio/interrupt",
                _switch_body(card, choice_id=choice_id, option_id=option),
            )
            card = _frame(frames, "plan.proposed")
            await _shows(app, card["interrupt_id"], option)
        approved = await _stream(
            client,
            "/api/studio/interrupt",
            {
                **_interrupt_body(card["interrupt_id"], "approve"),
                "plan_revision": card["revision"],
            },
        )

    assert _frame(approved, "plan.approved")["choices"] == [
        {
            "choice_id": choice_id,
            "question": "Sector",
            "chosen": "population (CRS 130)",
            "changed_from": "health (CRS 120)",
        }
    ]


async def test_an_amendment_drops_the_labels_the_turns_switches_kept() -> None:
    """An amendment drafts the plan again, possibly under the same
    requirement ids, so the next `plan.approved` names no `changed_from`
    the turn's switches kept before it."""
    app = _app(*_plan(), tool_surface=ScriptedSurface())
    async with _client(app) as client:
        plan = await _plan_with_a_choice(app, client)
        conversation = app.state.studio.store.current(PRINCIPAL, CONVERSATION)
        conversation.turn_choice_baseline = {_CHOICE_ID: "health (CRS 120)"}
        await app.state.studio.store.save(PRINCIPAL, CONVERSATION)
        await _stream(
            client,
            "/api/studio/interrupt",
            _interrupt_body(plan["interrupt_id"], "amend", "use commitments"),
        )

    assert conversation.turn_choice_baseline == {}


async def test_a_switch_leaves_its_label_out_of_the_words_later_searches_send() -> None:
    """A switch to grant equivalents that changes nothing (the plan states
    no scope choice) stays in the turn's record, and a search after the plan
    runs sends the user's own words alone, so Search reads no basis the user
    did not state."""
    basis_id = "spec_" + "b" * 32 + ":measure_basis"
    basis = _PendingChoice(
        active="measure_basis:gross",
        options={
            "measure_basis:gross": _PendingOption(label="gross disbursements", status="available"),
            "measure_basis:grant_equivalent": _PendingOption(
                label="grant equivalents", status="available"
            ),
        },
    )
    search = _search()
    app = _app(
        *_plan("q with forks", search_call("aid"), _text("Done.")),
        resolver=search,
        tool_surface=ScriptedSurface(),
    )
    async with _client(app) as client:
        plan = await _plan_with_a_choice(app, client, choices={basis_id: basis})
        switched = await _stream(
            client,
            "/api/studio/interrupt",
            _switch_body(plan, choice_id=basis_id, option_id="measure_basis:grant_equivalent"),
        )
        again = _frame(switched, "plan.proposed")
        final = await _stream(
            client, "/api/studio/interrupt", _interrupt_body(again["interrupt_id"], "approve")
        )

    assert _events(final)[-1] == "done"
    assert [call["context"].user_message for call in search.calls] == ["kenya oda 2023"]
    (record,) = app.state.studio.store.current(PRINCIPAL, CONVERSATION).turns
    assert record["user"] == "kenya oda 2023\n(then: count grant equivalents)"


def _scoped_envelope(scope: dict[str, Any]) -> dict[str, Any]:
    """A search envelope with one definite part, "health aid", compiled as `scope` states."""
    pin = scope["compiled"]["spec_id"]
    block = {
        "spec_id": pin,
        "sentence": "US health aid to Africa",
        "definition": {
            "schema_version": 1,
            "spec_id": pin,
            "identity": scope["compiled"]["identity"],
        },
        "resolved": {"scope_choices": scope},
    }
    part = {"status": "definite", "variable_text": "health aid", "spec": block}
    return {"schema_version": 1, "status": "definite", "parts": [part]}


def _chip_body(chip: Mapping[str, Any]) -> dict[str, Any]:
    """The request a click on `chip`, an `apply_choice` suggestion, sends."""
    return {
        "kind": "turn.request",
        "schema_version": 1,
        "conversation_id": CONVERSATION,
        "apply_choice": {key: chip[key] for key in ("choice_id", "option_id", "turn", "spec_id")},
    }


def _first_answer_chip(scope: dict[str, Any], choice_id: str, option_id: str) -> dict[str, Any]:
    """A chip on the conversation's first answer, moving `scope`'s reading."""
    return {
        "choice_id": choice_id,
        "option_id": option_id,
        "turn": 1,
        "spec_id": scope["compiled"]["spec_id"],
    }


_OPTION_REFUSED = {
    "code": "requested_option_unavailable",
    "title": "Requested Option Unavailable",
    "detail": "Scope entry 1 cannot apply to this part: option_unknown.",
    "pointer": "/input/entries/0/scope_entries/1",
    "retryable": False,
}


def _chip_app(
    *script: types.Part,
    executor: Any = None,
    fetches: Sequence[Mapping[str, Any]] = ({},),
    searches: int = 1,
    refuse: bool = False,
) -> tuple[FastAPI, dict[str, Any], SearchStub]:
    """An app whose first turn searches US health aid to Africa, fetches it
    once per entry of `fetches` (each the fetch's `scope`, `stat_vars` or
    `breakdown`) and answers, then runs `script`; Search answers a chip moving
    its sector to CRS 120 and 130. The search finds health aid each of
    `searches` times it is called. With `refuse`, Search refuses the chip.
    Returns the app, the scope and the Search stub."""
    scope = _hashed_scope()
    search = (
        SearchStub(refuse={scope["id"]: _OPTION_REFUSED})
        if refuse
        else SearchStub(answer={scope["id"]: _switched_answer(scope)[1]})
    )
    app = _app(
        search_call("health aid"),
        *(fetch_call(scope["compiled"]["spec_id"], **fetched) for fetched in fetches),
        _text("Health aid to Africa was this much."),
        *script,
        resolver=CannedSearch(*[_scoped_envelope(scope)] * searches),
        executor=executor,
        definition_surface=DataMcpDefinitions(InMemoryDefinitionStore(), search=search),
    )
    return app, scope, search


_TWO_YEARS = {"kind": "range", "start_year": 2015, "end_year": 2020}


@pytest.mark.parametrize(
    "fetches",
    [
        pytest.param(
            [
                {
                    "scope": {
                        "entities": {"destination": ["Ghana", "Kenya"]},
                        "window": _TWO_YEARS,
                    },
                    "stat_vars": "sum",
                }
            ],
            id="places-and-years",
        ),
        pytest.param(
            [
                {"scope": {"entities": {"destination": ["Ghana"]}}},
                {"scope": {"entities": {"destination": ["Kenya"]}}},
            ],
            id="ghana-then-kenya",
        ),
        pytest.param(
            [{"scope": {"window": _TWO_YEARS}}, {"scope": {"window": {"kind": "latest"}}}],
            id="years-then-latest",
        ),
    ],
)
async def test_a_chip_fetches_the_switched_reading_narrowed_as_the_answer_fetched_it(
    fetches: list[dict[str, Any]],
) -> None:
    """The chip fetches the switched reading once for each way the answer
    fetched it, with that fetch's places, years and reading, so it changes the
    sector alone."""
    executor = ScriptedExecutor()
    app, scope, _ = _chip_app(_text("Now with population."), executor=executor, fetches=fetches)
    sector = f"{scope['id']}:sector"
    async with _client(app) as client:
        await _stream(client, "/api/studio/turn", _turn_body("health aid to Ghana and Kenya"))
        await _stream(
            client,
            "/api/studio/turn",
            _chip_body(_first_answer_chip(scope, sector, "sector:120+130")),
        )

    calls = [call["args"] for call in executor.calls if call["tool"] == "get_data"]
    answer, chip = calls[: len(fetches)], calls[len(fetches) :]
    assert [c["spec_id"] for c in answer] == [scope["compiled"]["spec_id"]] * len(fetches)
    assert [c["spec_id"] for c in chip] == [_switched_answer(scope)[0]] * len(fetches)
    for asked, made, refetched in zip(fetches, answer, chip, strict=True):
        assert {key: made[key] for key in asked} == asked
        assert {key: refetched[key] for key in asked} == asked


async def test_a_chip_that_no_longer_applies_is_refused_with_its_constraint() -> None:
    """A chip whose choice no stored answer holds, or that is already on its
    option, is stale; an option the choice does not offer or Search cannot
    serve is unavailable. Each is a 422 before any stream, and a chip on a
    conversation the store never had is unknown."""
    app, scope, search = _chip_app()
    sector, basis = f"{scope['id']}:sector", f"{scope['id']}:measure_basis"
    cases = [
        ((f"{scope['id']}:provider", "provider:inclusive"), "scope_choice_stale"),
        ((sector, "sector:120"), "scope_choice_stale"),
        ((sector, "sector:999"), "switch_option_unavailable"),
        ((basis, "measure_basis:net"), "switch_option_unavailable"),
    ]
    async with _client(app) as client:
        await _stream(client, "/api/studio/turn", _turn_body("health aid to Africa"))
        answers = [
            await client.post(
                "/api/studio/turn", json=_chip_body(_first_answer_chip(scope, *pick)), headers=AUTH
            )
            for pick, _ in cases
        ]
        unknown = await client.post(
            "/api/studio/turn",
            json={
                **_chip_body(_first_answer_chip(scope, sector, "sector:120+130")),
                "conversation_id": "conv_other",
            },
            headers=AUTH,
        )
        turns = len(app.state.studio.store.current(PRINCIPAL, CONVERSATION).turns)

    assert [(r.status_code, r.json()["constraint_id"]) for r in answers] == [
        (422, constraint) for _, constraint in cases
    ]
    assert all(r.json()["pointer"] == "/apply_choice" and r.json()["detail"] for r in answers)
    assert unknown.status_code == 404
    assert search.posted == []
    assert turns == 1


async def test_a_chip_is_shown_applied_and_kept_and_each_step_is_recorded(
    otlp_spans: InMemorySpanExporter,
) -> None:
    """The first answer offers a chip per other option of its reading's
    choices. Clicking one applies it with no search and answers from the data
    it fetched, with the basis line; the turn after it, which moves nothing
    back, keeps it. The telemetry holds each step."""
    app, scope, search = _chip_app(_text("Now with population."), _text("Glad to help."))
    async with _client(app) as client:
        first = await _stream(client, "/api/studio/turn", _turn_body("health aid to Africa"))
        chips = _frame(first, "suggestions")["items"]
        (chip,) = [c for c in chips if c["option_id"] == "sector:120+130"]
        applied = await _stream(client, "/api/studio/turn", _chip_body(chip))
        await _stream(client, "/api/studio/turn", _turn_body("thanks"))
        turns = app.state.studio.store.current(PRINCIPAL, CONVERSATION).turns

    new_pin = _switched_answer(scope)[0]
    assert [(c["kind"], c["option_id"]) for c in chips] == [
        ("apply_choice", "sector:120+130"),
        ("apply_choice", "region:members_and_residuals"),
        ("apply_choice", "measure_basis:grant_equivalent"),
    ]
    assert chip == {
        "kind": "apply_choice",
        "label": "health (CRS 120) and population and reproductive health (CRS 130)",
        "choice_id": f"{scope['id']}:sector",
        "option_id": "sector:120+130",
        "turn": 1,
        "spec_id": scope["compiled"]["spec_id"],
    }
    assert len(search.posted) == 1
    assert len(app.state.model.requests) == 5  # search, fetch, answer; the chip; thanks
    text = _frame(applied, "message.complete")["text"]
    assert text.startswith("Now with population.\n\nCounted as: ")
    assert turns[1]["user"] == "count " + chip["label"]
    assert turns[1]["applied_choice"]["spec_id"] == new_pin
    events = [event.name for span in otlp_spans.get_finished_spans() for event in span.events]
    assert [name for name in events if name.startswith("scope_choice.")] == [
        "scope_choice.chips_shown",
        "scope_choice.applied",
        "scope_choice.chips_shown",
        "scope_choice.kept",
    ]


async def test_a_chip_search_refuses_is_recorded_as_not_applied(
    otlp_spans: InMemorySpanExporter,
) -> None:
    """The turn keeps the chip it was asked to apply, and with no reading
    Search compiled for it, the chip is recorded as not applied."""
    app, _, search = _chip_app(_text("That could not be counted."), refuse=True)
    async with _client(app) as client:
        first = await _stream(client, "/api/studio/turn", _turn_body("health aid to Africa"))
        (chip,) = [
            c for c in _frame(first, "suggestions")["items"] if c["option_id"] == "sector:120+130"
        ]
        await _stream(client, "/api/studio/turn", _chip_body(chip))
        turns = app.state.studio.store.current(PRINCIPAL, CONVERSATION).turns

    events = [event.name for span in otlp_spans.get_finished_spans() for event in span.events]
    assert len(search.posted) == 1
    assert "spec_id" not in turns[1]["applied_choice"]
    assert "scope_choice.not_applied" in events
    assert "scope_choice.applied" not in events


async def test_a_chip_applies_only_on_the_answer_it_was_offered_with() -> None:
    """A chip on the Africa answer is stale once a newer answer, here the same
    question asked for Ghana, stands: it is refused 422 before any stream, and
    Search is asked nothing. The newer answer's own chip still applies."""
    pin = _hashed_scope()["compiled"]["spec_id"]
    app, scope, search = _chip_app(
        search_call("health aid"),
        fetch_call(pin),
        _text("Health aid to Ghana was this much."),
        _text("Now with population."),
        searches=2,
    )
    async with _client(app) as client:
        africa = await _stream(client, "/api/studio/turn", _turn_body("health aid to Africa"))
        ghana = await _stream(client, "/api/studio/turn", _turn_body("health aid to Ghana"))
        old, new = (
            next(
                c
                for c in _frame(frames, "suggestions")["items"]
                if c["option_id"] == "sector:120+130"
            )
            for frames in (africa, ghana)
        )
        refused = await client.post("/api/studio/turn", json=_chip_body(old), headers=AUTH)
        applied = await _stream(client, "/api/studio/turn", _chip_body(new))
        turns = app.state.studio.store.current(PRINCIPAL, CONVERSATION).turns

    assert (old["turn"], new["turn"]) == (1, 2)
    assert old["spec_id"] == new["spec_id"] == pin
    assert refused.status_code == 422
    assert refused.json()["constraint_id"] == "scope_choice_stale"
    assert refused.json()["pointer"] == "/apply_choice"
    assert len(search.posted) == 1
    assert _events(applied)[-1] == "done"
    assert turns[2]["applied_choice"]["spec_id"] == _switched_answer(scope)[0]


async def test_a_chip_whose_reading_a_plan_card_switch_moved_is_refused() -> None:
    """The Africa answer stands as the latest answered turn, but a later turn
    paused on its plan and a switch on the card moved the chip's choice to a
    new reading: the chip would switch a reading the answer did not use, so it
    is refused 422 as stale. The switch moved the sector and the chip moves
    the region, an option the new reading still offers, so the reading alone
    marks the chip stale."""
    app, scope, search = _chip_app(*_plan())
    async with _client(app) as client:
        first = await _stream(client, "/api/studio/turn", _turn_body("health aid to Africa"))
        (chip,) = [
            c
            for c in _frame(first, "suggestions")["items"]
            if c["option_id"] == "region:members_and_residuals"
        ]
        paused = await _stream(client, "/api/studio/turn", _turn_body("q with forks"))
        # The switch node stores the reading Search compiled for the switch on
        # the paused turn, as `store_search` writes it.
        conversation = app.state.studio.store.current(PRINCIPAL, CONVERSATION)
        moved = copy.deepcopy(conversation.turns[0]["readings"][-1])
        pin, envelope = _switched_answer(scope)
        moved["spec_id"] = pin
        moved["scope"] = envelope["parts"][0]["spec"]["resolved"]["scope_choices"]
        conversation.turn_readings = [*conversation.turn_readings, moved]
        await app.state.studio.store.save(PRINCIPAL, CONVERSATION)
        refused = await client.post("/api/studio/turn", json=_chip_body(chip), headers=AUTH)

    assert "plan.proposed" in _events(paused)
    assert refused.status_code == 422
    assert refused.json()["constraint_id"] == "scope_choice_stale"
    assert search.posted == []


async def test_answers_to_a_plan_are_refused() -> None:
    async with _client(_app(*_plan())) as client:
        first = await _stream(client, "/api/studio/turn", _turn_body("q with forks"))
        plan = _frame(first, "plan.proposed")
        response = await client.post(
            "/api/studio/interrupt",
            json=_interrupt_body(
                plan["interrupt_id"], "approve", answers={"q1": {"kind": "skipped"}}
            ),
            headers=AUTH,
        )
    assert response.status_code == 422
    assert "a plan interrupt takes no answers" in response.json()["detail"]


async def test_adding_an_already_composed_component_mints_no_revision() -> None:
    executor = ScriptedExecutor()
    app = _app(
        *_card(),
        resolver=_search(),
        executor=executor,
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
    executor = ScriptedExecutor()
    app = _app(
        *_card(),
        resolver=_search(),
        executor=executor,
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
    resolver = _HangingResolver()
    app = _app(*_card(), resolver=resolver)
    async with _client(app) as client:
        stream_task = asyncio.ensure_future(_stream(client, "/api/studio/turn", _turn_body()))
        await asyncio.wait_for(resolver.entered.wait(), timeout=5)
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
    store = StudioStore(_engine())

    first = await _acquire_or_conflict(store, "u_a", CONVERSATION)
    assert isinstance(first, LockHold)
    assert first.lock.locked()

    second = await _acquire_or_conflict(store, "u_a", CONVERSATION)
    assert isinstance(second, JSONResponse)
    assert second.status_code == 409
    assert json.loads(second.body)["error"] == "transient_unavailable"

    other = await _acquire_or_conflict(store, "u_a", "conv_01test0009")
    assert isinstance(other, LockHold)
    assert other.lock is not first.lock

    first.release()
    third = await _acquire_or_conflict(store, "u_a", CONVERSATION)
    assert isinstance(third, LockHold) and third.lock is first.lock


# --- conversation history: turn records and digest ----------------------------


async def test_turn_two_receives_the_conversation_digest() -> None:
    app = _app(*_card(), resolver=_search())
    async with _client(app) as client:
        await _conversation_with_component(client)
        await _stream(client, "/api/studio/turn", _turn_body("now the same for Uganda"))

    requests = _model(app).requests
    label = "Earlier turns in this conversation"
    assert _digest(requests[0], label) is None  # turn 1: no history yet
    history = _digest(requests[-1], label)
    assert history is not None
    assert history["total_turns"] == 1
    (record,) = history["turns"]
    assert record["user"] == "kenya oda 2023"
    assert record["cards"] == [
        {
            "component_id": "cmp_01test0001",
            "type": "stat_card",
            "title": "Aid to Kenya",
            "action": "created",
        }
    ]


async def test_a_plain_answer_is_kept_in_history() -> None:
    app = _app(_text("Hello. Ask me about aid."))
    async with _client(app) as client:
        frames = await _stream(client, "/api/studio/turn", _turn_body("hi"))
        await _stream(client, "/api/studio/turn", _turn_body("aid to Kenya"))

    assert _events(frames)[-1] == "done"
    texts = [(f["event"], f["text"]) for f in frames if f["event"].startswith("message.")]
    assert texts == [
        ("message.delta", "Hello. Ask me about aid."),
        ("message.complete", "Hello. Ask me about aid."),
    ]
    history = _digest(_model(app).requests[-1], "Earlier turns in this conversation")
    assert history is not None
    (record,) = history["turns"]
    assert record == {"user": "hi", "assistant": "Hello. Ask me about aid."}


async def test_a_failed_turn_is_recorded_with_its_error() -> None:
    class BrokenSurface(ScriptedSurface):
        async def call_tool(self, **kwargs: Any) -> ToolCallResult:
            raise RuntimeError("analysis mount went away")

    store = StudioStore(
        _engine(),
    )
    app = _app(*_plan(), tool_surface=BrokenSurface(), store=store)
    async with _client(app) as client:
        first = await _stream(client, "/api/studio/turn", _turn_body("q that dies"))
        plan = _frame(first, "plan.proposed")
        frames = await _stream(
            client, "/api/studio/interrupt", _interrupt_body(plan["interrupt_id"], "approve")
        )
        assert frames[-1]["event"] == "error"

    conversation = store.current(PRINCIPAL, CONVERSATION)
    (record,) = conversation.turns
    assert record["user"] == "q that dies"
    # A surface that raises is a defect, which the same turn meets again.
    assert record["error"] == "capability_unavailable"


@pytest.mark.parametrize(("then", "closed_as"), [("cancel", "cancelled"), ("turn", "abandoned")])
async def test_a_failed_turn_whose_close_was_not_written_is_closed_by_what_comes_next(
    then: str, closed_as: str
) -> None:
    """A node fails and its close cannot be written: the client gets the
    node's error frame, and the conversation's record of the turn holds
    that error. The turn stays running until a later Stop closes it
    `cancelled` or a later turn `abandoned`, which leave the record as it
    is."""

    class BrokenSurface(ScriptedSurface):
        async def call_tool(self, **kwargs: Any) -> ToolCallResult:
            raise RuntimeError("analysis mount went away")

    store = StudioStore(_engine())
    app = _app(*_plan(), _text("Hello."), tool_surface=BrokenSurface(), store=store)
    turns = app.state.studio.turns
    real_finish = turns.finish

    async def finish_unwritten(cursor: Any, **kwargs: Any) -> Any:
        if kwargs.get("status") is TurnStatus.FAILED:
            raise ConnectionError("the database is unreachable")
        return await real_finish(cursor, **kwargs)

    async with _client(app) as client:
        first = await _stream(client, "/api/studio/turn", _turn_body("q that dies"))
        plan = _frame(first, "plan.proposed")
        turns.finish = finish_unwritten
        frames = await _stream(
            client, "/api/studio/interrupt", _interrupt_body(plan["interrupt_id"], "approve")
        )
        assert frames[-1]["event"] == "error"
        assert frames[-1]["error"]["error"] == "capability_unavailable"
        (shown,) = store.current(PRINCIPAL, CONVERSATION).turns
        left = await turns.open_cursor(PRINCIPAL, CONVERSATION)
        assert left is not None and left.status is TurnStatus.RUNNING

        if then == "cancel":
            cancel = await client.post(
                "/api/studio/cancel",
                json={"kind": "turn.cancel", "schema_version": 1, "conversation_id": CONVERSATION},
                headers=AUTH,
            )
            assert cancel.status_code == 204
        else:
            assert (
                _events(await _stream(client, "/api/studio/turn", _turn_body("hi")))[-1] == "done"
            )

    (record, *_) = (await StudioStore(_engine()).load(PRINCIPAL, CONVERSATION)).turns
    assert record == shown
    assert record["user"] == "q that dies"
    assert record["error"] == "capability_unavailable"
    assert record["assistant"] == frames[-1]["error"]["detail"]
    async with _engine().connect() as conn:
        status = (
            await conn.execute(
                text("SELECT status FROM turn_cursors WHERE id = :id"), {"id": left.id}
            )
        ).scalar_one()
    assert status == closed_as


def test_the_conversation_digest_is_bounded() -> None:
    state = ConversationState(conversation_id=CONVERSATION)
    for index in range(12):
        state.turns.append({"user": f"question {index}", "assistant": "a" * 400})

    digest = state.conversation_digest_state()

    assert digest is not None
    assert digest["total_turns"] == 12
    assert len(digest["turns"]) == 10  # newest window
    assert digest["turns"][0]["user"] == "question 2"
    assert all(len(turn["assistant"]) <= 280 for turn in digest["turns"][:-1])
    assert digest["turns"][-1]["assistant"] == "a" * 400  # the previous answer, whole
    assert len(json.dumps(digest["turns"])) <= 5_000


# --- recorder isolation and terminal-path records ------------------------------


async def test_two_streaming_conversations_keep_their_steps_apart() -> None:
    """Through the routes: both conversations sit inside their fetch at the
    same moment, and each stream still reports only its own call. Each
    conversation's agent fetches without a search, so the shared script
    serves both whichever asks first."""
    both_inside = asyncio.Barrier(2)

    class MeetingExecutor(ScriptedExecutor):
        async def __call__(
            self, *, tool: str, args: dict[str, Any], on_progress: ProgressCallback | None = None
        ) -> dict[str, Any]:
            if tool != "get_data":
                return await super().__call__(tool=tool, args=args)
            conversation = current_conversation_id.get()
            self.calls.append({"tool": tool, "conversation": conversation})
            await both_inside.wait()
            return {
                "outcome": "ok",
                "frame_refs": [REF_A],
                "summary": f"fetched for {conversation}",
            }

    executor = MeetingExecutor()
    app = _app(
        fetch_call(SPEC_A),
        fetch_call(SPEC_A),
        _text("Kenya."),
        _text("Uganda."),
        executor=executor,
    )
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

    mine, theirs = f"{PRINCIPAL}/{CONVERSATION}", f"{PRINCIPAL}/{other}"
    fetched = [call["conversation"] for call in executor.calls if call["tool"] == "get_data"]
    assert sorted(fetched) == [mine, theirs]

    def fetch_details(frames: list[dict[str, Any]]) -> list[str]:
        return [
            f["detail"]
            for f in frames
            if f["event"] == "activity.step"
            and f.get("tool") == "fetch"
            and f["status"] == "complete"
        ]

    assert fetch_details(frames_a) == [f"fetched for {mine}"]
    assert fetch_details(frames_b) == [f"fetched for {theirs}"]
    assert _events(frames_a)[-1] == "done"
    assert _events(frames_b)[-1] == "done"


async def test_the_recorder_keeps_concurrent_conversations_apart() -> None:
    async def executor(
        *, tool: str, args: dict[str, Any], on_progress: ProgressCallback | None = None
    ) -> dict[str, Any]:
        return {"outcome": "ok", "frame_refs": []}

    recorder = _RecordingExecutor(
        executor,
        store=StudioStore(
            _engine(),
        ),
    )
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

    recorder = _RecordingExecutor(
        executor,
        store=StudioStore(
            _engine(),
        ),
    )
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
        ("refusal", HarnessCode.RAISED),
        ("refusal", "cancelled"),
    ]
    # Both are codes the failure table classes, and the exception stays in
    # the message.
    assert all(record.result["refusal"]["code"] in FAILURE_CLASS for record in records)
    assert "RuntimeError" in records[0].result["refusal"]["message"]


async def test_a_rejected_plan_turn_is_recorded() -> None:
    store = StudioStore(
        _engine(),
    )
    app = _app(*_plan(), store=store)
    async with _client(app) as client:
        frames = await _stream(client, "/api/studio/turn", _turn_body("q with forks"))
        plan = _frame(frames, "plan.proposed")
        resumed = await _stream(
            client, "/api/studio/interrupt", _interrupt_body(plan["interrupt_id"], "reject")
        )
        assert resumed[-1]["event"] == "done"

    (record,) = store.current(PRINCIPAL, CONVERSATION).turns
    assert record["user"] == "q with forks"
    assert "rejected" in record["assistant"].lower()
    assert "error" not in record


async def test_an_amendment_reaches_the_turn_record() -> None:
    store = StudioStore(
        _engine(),
    )
    app = _app(
        *_plan(),
        tool_surface=ScriptedSurface(),
        store=store,
    )
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

    (record,) = store.current(PRINCIPAL, CONVERSATION).turns
    assert "only since 2015" in record["user"]


async def test_a_superseded_pause_is_recorded_as_abandoned() -> None:
    store = StudioStore(
        _engine(),
    )
    app = _app(*_plan(), store=store)
    async with _client(app) as client:
        paused = await _stream(client, "/api/studio/turn", _turn_body("q with forks"))
        assert paused[-1]["event"] == "plan.proposed"
        # The paused turn's cursor holds a plan record, which the record
        # that closes the turn keeps.
        async with _engine().begin() as conn:
            await conn.execute(
                text(
                    "UPDATE turn_cursors SET state = CAST("
                    "CAST(state AS jsonb) || CAST(:records AS jsonb) AS json)"
                ),
                {"records": json.dumps({ORCHESTRATOR_PLAN_RECORDS_KEY: [{"plan": 1}]})},
            )
        # A fresh turn supersedes the pause; the abandoned turn must not
        # vanish from history.
        second = await _stream(client, "/api/studio/turn", _turn_body("aid to kenya, quickly"))
        assert second[-1]["event"] == "done"

    records = store.current(PRINCIPAL, CONVERSATION).turns
    assert [record["user"] for record in records] == ["q with forks", "aid to kenya, quickly"]
    assert records[0]["error"] == "abandoned"
    assert records[0]["plans"] == [{"plan": 1}]
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
    store = StudioStore(
        _engine(),
    )
    app = _app(
        *_card(),
        resolver=_search(),
        executor=ScriptedExecutor(
            [
                *_card_results(),
                {
                    "outcome": "ok",
                    "frame_refs": [],
                    "component_body": {"stored_intent": {"frame_ref": REF_A}},
                },
                {"outcome": "ok", "frame_refs": []},
                {"outcome": "ok", "frame_refs": [], "component": rebuilt},
            ]
        ),
        store=store,
        edit_model=TypeSwitchEdit(),
    )
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
    records = store.current(PRINCIPAL, CONVERSATION).turns
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
    store = StudioStore(
        _engine(),
    )
    app = _app(
        *_card(),
        resolver=_search(),
        executor=ScriptedExecutor(
            [
                *_card_results(),
                {
                    "outcome": "ok",
                    "frame_refs": [],
                    "component_body": {"stored_intent": {"frame_ref": REF_A}},
                },
                {"outcome": "ok", "frame_refs": []},
                {"outcome": "ok", "frame_refs": [], "component": rebuilt},
            ]
        ),
        store=store,
        edit_model=TypeSwitchEdit(),
    )
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
    records = store.current(PRINCIPAL, CONVERSATION).turns
    assert records[-1]["cards"] == [
        {
            "component_id": "cmp_01test0002",
            "type": "bar_chart",
            "title": "Aid to Kenya",
            "action": "recreated",
            "window": "2018-2024",
        }
    ]


# --- the cards one chart call lands, end to end -------------------------------


async def test_a_component_missing_fact_summary_never_lands(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """An `ok` write whose component fails the `component.created` wire
    contract (here, a missing `fact_summary`) is refused like any other
    unusable write. No frame announces it and the canvas is unchanged."""
    missing = {key: value for key, value in COMPONENT.items() if key != "fact_summary"}
    store = StudioStore(
        _engine(),
    )
    app = _app(
        *_card(),
        resolver=_search(),
        executor=ScriptedExecutor(_card_results(missing)),
        store=store,
    )
    with caplog.at_level("ERROR", logger="harness.studio_translate"):
        async with _client(app) as client:
            frames = await _stream(client, "/api/studio/turn", _turn_body())

    assert _events(frames)[-1] == "done"
    assert "component.created" not in _events(frames)
    assert "composition" not in _frame(frames, "done")
    assert store.current(PRINCIPAL, CONVERSATION).composition is None
    unreadable = [r for r in caplog.records if "fails the component contract" in r.getMessage()]
    assert len(unreadable) == 1
    assert "fact_summary" in unreadable[0].getMessage()


class _WritesUntil(ScriptedExecutor):
    """`ScriptedExecutor` over two `_card` turns, the second drawing two
    charts: its first write lands `cmp_01test0002`, and the conversation's
    third write -- the second turn's second -- runs `third_write` instead
    of answering."""

    def __init__(self, third_write: Callable[[], Any]) -> None:
        super().__init__(
            [
                *_card_results(),
                *_card_results({**COMPONENT, "component_id": "cmp_01test0002"}),
            ]
        )
        self._third_write = third_write
        self.writes = 0

    async def __call__(
        self, *, tool: str, args: dict[str, Any], on_progress: ProgressCallback | None = None
    ) -> dict[str, Any]:
        if tool == "write_component":
            self.writes += 1
            if self.writes == 3:
                await self._third_write()
        return await super().__call__(tool=tool, args=args)


async def _hang() -> None:
    await asyncio.Event().wait()


async def _reset() -> None:
    raise RuntimeError("presentation-mcp connection reset")


async def test_first_write_ok_then_cancellation_before_drain_leaves_the_canvas_unchanged() -> None:
    """A cancel that lands while a chart call's second write hangs kills the
    turn before the call returns. No event follows, so
    `harness.studio_translate`'s `_component_frames` does not run for either
    write, and neither the first write's already-returned component nor the
    second lands. `harness.studio.StudioRunner._frames`'s `finally` drops
    both undrained records with the turn. The canvas is exactly what the
    first turn left it, with no card landed partway through the second."""
    store = StudioStore(
        _engine(),
    )
    executor = _WritesUntil(_hang)
    app = _app(
        *_card(),
        *_card(charts=2),
        resolver=_search(2),
        executor=executor,
        store=store,
    )
    async with _client(app) as client:
        await _conversation_with_component(client)
        before = [b.component_id for b in store.current(PRINCIPAL, CONVERSATION).composition.blocks]

        resume = asyncio.ensure_future(
            _stream(client, "/api/studio/turn", _turn_body("add Guatemala to the chart"))
        )

        async def _wait_for_the_hang() -> None:
            while executor.writes < 3:  # first write returned, second hanging
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
    after = [b.component_id for b in store.current(PRINCIPAL, CONVERSATION).composition.blocks]
    assert after == before == ["cmp_01test0001"]


async def test_a_chart_stored_as_a_card_the_harness_cannot_read_fails_its_line() -> None:
    """`make_charts` draws nothing for a card the executor refused as unreadable,
    and the recorded write that stored it ends the turn's line as a failed step
    with the `stored_card_unreadable` sentence: no `produced`, no component
    frame, and no canvas update."""
    executor = ScriptedExecutor(
        [*_card_results()[:2], _unreadable_refusal()],
        strict=True,
    )
    app = _app(*_card(), resolver=_search(), executor=executor)
    async with _client(app) as client:
        frames = await _stream(client, "/api/studio/turn", _turn_body())

    events = _events(frames)
    assert events[-1] == "done"
    assert "error" not in events
    assert "component.created" not in events
    (line,) = [
        f for f in frames if f["event"] == "activity.step" and f.get("tool") == "write_component"
    ]
    assert line["status"] == "failed"
    assert line.get("produced") is None
    assert line["detail"] == classed_sentence("stored_card_unreadable", "do_not_retry")
    assert "composition.updated" not in events


async def test_first_write_ok_then_second_write_raises_still_announces_what_actually_landed() -> (
    None
):
    """`make_charts` answers a write that raised as a chart it could not draw
    and draws nothing after it, so the turn goes on. The first write's card
    is announced (`component.created` plus `composition.updated`) and lands
    for real: the server's own composition is exactly what the client was
    told. The second write, which never returned, produces no frame and
    never reaches the canvas."""
    store = StudioStore(
        _engine(),
    )
    app = _app(
        *_card(),
        *_card(charts=2),
        resolver=_search(2),
        executor=_WritesUntil(_reset),
        store=store,
    )
    async with _client(app) as client:
        await _conversation_with_component(client)
        second = await _stream(client, "/api/studio/turn", _turn_body("add Guatemala to the chart"))

    assert _events(second)[-1] == "done"
    created = [frame for frame in second if frame["event"] == "component.created"]
    # Exactly the first write's card -- the second, which raised, never
    # produced a component to announce.
    assert [frame["component_id"] for frame in created] == ["cmp_01test0002"]
    announced_manifest = _frame(second, "composition.updated")["manifest"]
    server_blocks = store.current(PRINCIPAL, CONVERSATION).composition.blocks
    assert [b["component_id"] for b in announced_manifest["blocks"]] == [
        b.component_id for b in server_blocks
    ]
    assert [b.component_id for b in server_blocks] == ["cmp_01test0001", "cmp_01test0002"]


# --- cross-instance: two StudioRunners over one shared database service ----


async def test_a_plan_paused_on_one_instance_is_approved_and_continued_on_another() -> None:
    """A pauses on the plan. B, a separate `StudioRunner` and `StudioStore`
    that never built this plan, answers `/interrupt` and completes the
    turn, restoring the plan's step order into its own `_RecordingSurface`
    from the ledger's `_PendingInterrupt.step_order` before resuming. A
    stale approve on B first gets back the card A sent, read from the
    ledger. A second `/turn` on B then continues the same conversation: the turn
    index advances past what A claimed, and the digest carries A's turn."""
    app_a = _app(
        *_plan(),
        tool_surface=ScriptedSurface(),
    )
    # B's own model answers only after the hand-back: `/interrupt` resumes
    # the turn A's own `/turn` already started, on the conversation
    # A created, so B never sees the first turn's question.
    app_b = _app(tool_surface=ScriptedSurface())

    async with _client(app_a) as client_a, _client(app_b) as client_b:
        first = await _stream(client_a, "/api/studio/turn", _turn_body("q with forks"))
        plan = _frame(first, "plan.proposed")
        assert app_b.state.studio.surface_recorder.step_order(f"{PRINCIPAL}/{CONVERSATION}") == []
        stale = await client_b.post(
            "/api/studio/interrupt",
            json={**_interrupt_body(plan["interrupt_id"], "approve"), "plan_revision": "0" * 64},
            headers=AUTH,
        )
        assert stale.status_code == 409
        assert stale.json()["current_plan"] == plan

        resumed = await _stream(
            client_b, "/api/studio/interrupt", _interrupt_body(plan["interrupt_id"], "approve")
        )
        assert _events(resumed)[-1] == "done"
        assert app_b.state.studio.surface_recorder.step_order(f"{PRINCIPAL}/{CONVERSATION}") == [
            step["id"] for step in plan["steps"]
        ]
        # B names the plan steps' Activity lines with the plan card's labels.
        assert app_b.state.studio.surface_recorder.step_labels(f"{PRINCIPAL}/{CONVERSATION}") == {
            step["id"]: step["label"] for step in plan["steps"]
        }
        lines = [frame["label"] for frame in resumed if frame["event"] == "activity.step"]
        assert plan["steps"][0]["label"] in lines

        second = await _stream(client_b, "/api/studio/turn", _turn_body("a follow-up question"))

    assert _events(second)[-1] == "done"
    state = await StudioStore(
        _engine(),
    ).load(PRINCIPAL, CONVERSATION)
    assert state is not None
    assert state.turn_index == 2
    digest = state.conversation_digest_state()
    assert digest is not None
    assert digest["total_turns"] == 2


async def test_a_double_approve_across_two_instances_gives_one_success_and_one_409() -> None:
    """A and B both hold the same plan pause and both answer `/interrupt`
    approve. B's own save is held back (`delayed_save`) until after A's has
    already committed, reproducing the actual race: both loaded the pending
    entry before either wrote. B's save then finds the ledger has moved and
    loses cleanly with a 409, rather than resuming its own segment of a
    turn A has already finished."""
    app_a = _app(
        *_plan(),
        tool_surface=ScriptedSurface(),
    )
    app_b = _app(
        *_plan(),
        tool_surface=ScriptedSurface(),
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


async def test_two_instances_racing_the_first_turn_on_a_brand_new_conversation_both_complete() -> (
    None
):
    """A and B both get the first-ever `/turn` for a never-before-seen
    conversation id at nearly the same time. B's own `store.create` is
    delayed until after A's create-and-claim has already committed, so B's
    `create` raises `ConversationExists`. B must load what A created and
    continue to its own claim, which advances the turn index it just loaded,
    rather than answering a bare 500."""
    app_a = _app()
    app_b = _app()

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
    state = await StudioStore(
        _engine(),
    ).load(PRINCIPAL, CONVERSATION)
    assert state is not None
    assert state.turn_index == 2


async def test_a_composition_edit_on_one_instance_survives_a_concurrent_turns_release() -> None:
    """B's `/composition` pins A's first turn's card while A's SECOND turn
    is still writing its own new one, mid-stream, before A's release. A's
    release finds the ledger has moved (B's pin) and replays A's own
    recorded append onto B's freshly pinned manifest instead of overwriting
    it with A's own stale copy -- B's edit survives."""
    app_b = _app()

    async with _client(app_b) as client_b:
        executor = _PinsTheFirstCardFromB(client_b)
        app_a = _app(
            *_card(),
            *_card(),
            resolver=_search(2),
            executor=executor,
        )
        async with _client(app_a) as client_a:
            executor.first_manifest = await _conversation_with_component(client_a)
            second = await _stream(client_a, "/api/studio/turn", _turn_body("now for Uganda"))
            assert _events(second)[-1] == "done"

    reloaded = await StudioStore(
        _engine(),
    ).load(PRINCIPAL, CONVERSATION)
    assert reloaded is not None
    blocks = reloaded.composition.blocks
    assert [b.component_id for b in blocks] == ["cmp_01test0001", "cmp_01test0002"]
    assert blocks[0].pinned is True


class _PinsTheFirstCardFromB(ScriptedExecutor):
    """`ScriptedExecutor` over two `_card` turns, the second landing
    `cmp_01test0002`. Its second write, the second turn's, first pins
    the first turn's card (`first_manifest`) through `client_b`, another
    instance, as a concurrent `/composition` call would."""

    def __init__(self, client_b: httpx2.AsyncClient) -> None:
        super().__init__(
            [
                *_card_results(),
                *_card_results({**COMPONENT, "component_id": "cmp_01test0002"}),
            ]
        )
        self._client_b = client_b
        self.first_manifest: dict[str, Any] = {}
        self.writes = 0

    async def __call__(
        self, *, tool: str, args: dict[str, Any], on_progress: ProgressCallback | None = None
    ) -> dict[str, Any]:
        if tool == "write_component":
            self.writes += 1
            if self.writes == 2:
                response = await self._client_b.post(
                    "/api/studio/composition",
                    json=_mutation_body(
                        self.first_manifest,
                        [
                            {
                                "op": "pin",
                                "component_id": self.first_manifest["blocks"][0]["component_id"],
                            }
                        ],
                    ),
                    headers=AUTH,
                )
                assert response.status_code == 200, response.text
        return await super().__call__(tool=tool, args=args)


async def test_ledger_at_turn_n_with_no_stream_running_lets_the_next_turn_succeed() -> None:
    """A conversation whose ledger holds a completed turn n, with its
    per-turn session `#t{n}` still existing, is idle rather than mid-claim:
    the next `/turn` claims n+1 and succeeds, on a SECOND instance that
    never saw the first."""
    app_a = _app()
    app_b = _app()

    async with _client(app_a) as client_a:
        first = await _stream(client_a, "/api/studio/turn", _turn_body("first"))
    assert _events(first)[-1] == "done"

    async with _client(app_b) as client_b:
        second = await _stream(client_b, "/api/studio/turn", _turn_body("second"))
    assert _events(second)[-1] == "done"

    state = await StudioStore(
        _engine(),
    ).load(PRINCIPAL, CONVERSATION)
    assert state is not None
    assert state.turn_index == 2


# --- the segment lease wired into the routes --------------------------------


async def test_a_turn_from_another_instance_while_a_streams_is_409_with_retry_after() -> None:
    """B's `/turn` arrives on a separate instance while A's segment is still
    live: the ledger's lease, not A's process-local lock, answers with the
    existing 409, carrying the segment's own remaining TTL as
    `retry_after_seconds`."""
    resolver = _HangingResolver()
    app_a = _app(*_card(), resolver=resolver)
    app_b = _app()

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
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A's segment expires (an injected clock, so the test does not wait
    out the real TTL) while A's stream is still open: B's `/turn` claims
    the conversation and completes, and A's own next heartbeat renewal
    finds itself fenced out, ending A's stream with a stream-fatal
    `transient_unavailable` error rather than `done`. A's renewals stall
    until B's turn is done: a heartbeat that renews first would keep A's
    segment alive, which is not the stalled instance this test is about."""
    clock = await StoreClock.start(_engine())
    store_a = StudioStore(
        _engine(),
        **clock.store_kwargs,
    )
    stalled = asyncio.Event()
    renew_a = store_a.renew

    async def stalled_renew(*args: Any, **kwargs: Any) -> Any:
        await stalled.wait()
        return await renew_a(*args, **kwargs)

    monkeypatch.setattr(store_a, "renew", stalled_renew)
    store_b = StudioStore(
        _engine(),
        **clock.store_kwargs,
    )
    resolver = _HangingResolver()
    app_a = _app(*_card(), resolver=resolver, store=store_a, heartbeat_seconds=0.05)
    app_b = _app(
        store=store_b,
    )

    async with _client(app_a) as client_a, _client(app_b) as client_b:
        stream_task_a = asyncio.ensure_future(_stream(client_a, "/api/studio/turn", _turn_body()))
        await asyncio.wait_for(resolver.entered.wait(), timeout=5)

        await clock.advance(studio_state_module.SEGMENT_TTL_S + 1)  # A's segment has expired

        second_b = await _stream(client_b, "/api/studio/turn", _turn_body("from b"))
        assert _events(second_b)[-1] == "done"
        stalled.set()

        frames_a = await asyncio.wait_for(stream_task_a, timeout=5)

    assert _events(frames_a)[-1] == "error"
    assert _frame(frames_a, "error")["error"]["error"] == "transient_unavailable"
    assert "done" not in _events(frames_a)


@pytest.mark.parametrize("claim_takes", [0, 13], ids=["prompt-claim", "slow-claim"])
async def test_a_heartbeat_that_cannot_renew_ends_the_stream_once_its_clock_passes_the_deadline(
    monkeypatch: pytest.MonkeyPatch, claim_takes: float
) -> None:
    """`renew` raises on every call: the stream stays open, retried every
    heartbeat interval, for as long as the store's instance clock reads
    before 10 seconds short of a TTL past the load the claim's lease expiry
    was dated from. Moving that clock past the deadline, with no real time
    elapsed, is what ends the stream with a stream-fatal `segment_lost`
    error. A claim whose turn takes `claim_takes` seconds to open
    starts its heartbeat later and keeps the same deadline, since the lease
    expires on the same date."""
    clock = await StoreClock.start(_engine())
    loaded_at = clock.at
    store = StudioStore(
        _engine(),
        **clock.store_kwargs,
    )

    async def _renew_always_fails(*_args: Any, **_kwargs: Any) -> str:
        raise RuntimeError("renewal unavailable")

    monkeypatch.setattr(store, "renew", _renew_always_fails)
    resolver = _HangingResolver()
    app = _app(*_card(), resolver=resolver, store=store, heartbeat_seconds=0.05)
    turns = app.state.studio.turns
    real_open_turn = turns.open_turn

    async def slow_open_turn(*args: Any, **kwargs: Any) -> Any:
        await clock.advance(claim_takes)
        return await real_open_turn(*args, **kwargs)

    monkeypatch.setattr(turns, "open_turn", slow_open_turn)

    async with _client(app) as client:
        stream_task = asyncio.ensure_future(_stream(client, "/api/studio/turn", _turn_body()))
        await asyncio.wait_for(resolver.entered.wait(), timeout=5)
        state = store.current(PRINCIPAL, CONVERSATION)
        assert state is not None and state.segment is not None
        deadline = loaded_at + studio_state_module.SEGMENT_TTL_S - 10

        await clock.set(deadline - 1)
        await asyncio.sleep(0.2)  # several heartbeat ticks, clock still short of the deadline
        assert not stream_task.done()

        await clock.set(deadline + 1)
        frames = await asyncio.wait_for(stream_task, timeout=5)

    assert _events(frames)[-1] == "error"
    assert _frame(frames, "error")["error"]["error"] == "transient_unavailable"
    assert "done" not in _events(frames)
    assert state.turn_recorded is False  # the turn stays as stored for a retry


async def test_cancel_from_another_instance_stops_the_live_stream_within_a_heartbeat(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """B, which has never loaded this conversation, POSTs `/cancel`: with no
    local armed event to signal, it falls to `request_cancel`, which finds
    A's live segment and flags it durably. A's next heartbeat renewal (the
    interval shrunk here so the test does not wait out the real one) picks
    the flag up and ends A's stream the same way a same-instance Stop
    does."""
    resolver = _HangingResolver()
    app_a = _app(*_card(), resolver=resolver, heartbeat_seconds=0.05)
    app_b = _app()

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
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A's heartbeat renews its live segment, which leaves
    `entry.state.composition` alone, before B writes the canvas on another
    instance. A's own `/composition` call afterwards must still see B's write
    and build on it: `mutate_composition` always reads the stored canvas, so
    A's live one never stands in for what the store holds."""
    resolver = _HangingResolver()
    app_a = _app(*_card(), resolver=resolver, heartbeat_seconds=0.05)

    async with _client(app_a) as client_a:
        stream_task = asyncio.ensure_future(_stream(client_a, "/api/studio/turn", _turn_body()))
        await asyncio.wait_for(resolver.entered.wait(), timeout=5)

        state_a = app_a.state.studio.store.current(PRINCIPAL, CONVERSATION)
        segment_id = state_a.segment.id
        first_expires_at = state_a.segment.expires_at

        async def _renewed() -> None:
            while state_a.segment is None or state_a.segment.expires_at == first_expires_at:
                await asyncio.sleep(0.01)

        await asyncio.wait_for(_renewed(), timeout=5)
        assert state_a.segment.id == segment_id  # renewed, not taken over

        store_b = StudioStore(
            _engine(),
        )
        await store_b.load(PRINCIPAL, CONVERSATION)
        block = CompositionBlock(
            component_id="cmp_from_b", revision=1, type=ComponentType.LINE_CHART, title="B"
        )
        await store_b.mutate_composition(
            PRINCIPAL,
            CONVERSATION,
            lambda manifest, _components: studio_state_module._next_manifest(
                manifest, [block], actor=CompositionActor.USER
            ),
        )

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

    reloaded = await StudioStore(
        _engine(),
    ).load(PRINCIPAL, CONVERSATION)
    assert reloaded.composition.blocks[0].pinned is True


async def test_a_request_cancelled_while_it_drops_its_claim_still_releases_the_lock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`/turn` claims, then its turn cannot be opened, and the request is
    cancelled while it drops the lease it claimed. The drop runs to the end,
    and the conversation's lock is released, so the next request on this
    instance is not refused."""
    app = _app(_text("Hello."))
    studio = app.state.studio
    store = studio.store

    async def unavailable(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("the turn cannot be opened")

    started, proceed = asyncio.Event(), asyncio.Event()
    real_drop = store._drop_lease

    async def slow_drop(*args: Any, **kwargs: Any) -> None:
        started.set()
        await proceed.wait()
        await real_drop(*args, **kwargs)

    monkeypatch.setattr(studio.turns, "open_turn", unavailable)
    monkeypatch.setattr(store, "_drop_lease", slow_drop)
    router = create_studio_router(studio, settings=Settings())
    endpoint = next(
        route.endpoint
        for route in router.routes
        if getattr(route, "path", "") == "/api/studio/turn"
    )
    request = asyncio.ensure_future(
        endpoint(TurnRequest.model_validate(_turn_body("hi")), authorization=AUTH["Authorization"])
    )
    await asyncio.wait_for(started.wait(), timeout=5)
    request.cancel()
    proceed.set()
    with pytest.raises(asyncio.CancelledError):
        await request

    assert not store.lock_for(PRINCIPAL, CONVERSATION).locked()
    stored = await store.load(PRINCIPAL, CONVERSATION)
    assert stored is not None and stored.segment is None


async def test_a_claim_whose_span_end_raises_still_releases_the_lock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`/turn` claims, then its turn cannot be opened. Recording the span of
    the claim it lets go of raises; the lock is released all the same."""
    app = _app(_text("Hello."))
    studio = app.state.studio

    async def unavailable(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("the turn cannot be opened")

    def broken_record(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("telemetry is down")

    monkeypatch.setattr(studio.turns, "open_turn", unavailable)
    monkeypatch.setattr(studio_module, "record_unstreamed_span", broken_record)

    with pytest.raises(RuntimeError, match="telemetry is down"):
        await _turn_endpoint(studio)(
            TurnRequest.model_validate(_turn_body("hi")), authorization=AUTH["Authorization"]
        )

    assert not studio.store.lock_for(PRINCIPAL, CONVERSATION).locked()


async def test_a_request_cancelled_after_it_let_go_leaves_the_next_request_s_claim_alone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A `/turn` refused as busy lets go of the lock, then waits on the
    export of the span it recorded. Request B takes the lock and a working
    copy, and A is cancelled during that export: B keeps both."""
    app = _app(_text("Hello."))
    studio = app.state.studio
    store = studio.store
    async with _client(app) as client:
        await _stream(client, "/api/studio/turn", _turn_body("hi"))
    other = StudioStore(_engine())
    await other.load(PRINCIPAL, CONVERSATION)
    await other.claim_turn_lease(PRINCIPAL, CONVERSATION)
    flushing = asyncio.Event()

    async def held_flush() -> None:
        # Only A's first export waits; a second release by A returns.
        if not flushing.is_set():
            flushing.set()
            await asyncio.Event().wait()

    monkeypatch.setattr(studio_module, "flush_telemetry", held_flush)
    request_a = asyncio.ensure_future(
        _turn_endpoint(studio)(
            TurnRequest.model_validate(_turn_body("again")), authorization=AUTH["Authorization"]
        )
    )
    await asyncio.wait_for(flushing.wait(), timeout=5)
    lock = store.lock_for(PRINCIPAL, CONVERSATION)
    assert not lock.locked()
    await lock.acquire()
    work_b = store.draft(PRINCIPAL, CONVERSATION)

    request_a.cancel()
    with pytest.raises(asyncio.CancelledError):
        await request_a

    assert lock.locked()
    draft = store._runtime[(PRINCIPAL, CONVERSATION)].draft
    assert draft is not None and draft[0] is work_b


async def test_a_turn_whose_body_never_starts_leaves_no_lock_or_lease() -> None:
    """Under ASGI spec 2.3 Starlette reads the client's disconnect beside
    the body, so a client already gone ends the response before the body's
    first frame is pulled. The segment's owner runs as a task of its own
    and still ends the segment: the lock is released, the lease cleared and
    the turn left open, and the next `/turn`, on another instance, claims
    and records it `abandoned`."""
    app = _app(search_call("aid to Kenya"), resolver=_HangingResolver())
    store = app.state.studio.store
    body = json.dumps(_turn_body("hi")).encode()
    received = 0

    async def receive() -> dict[str, Any]:
        nonlocal received
        received += 1
        if received == 1:
            return {"type": "http.request", "body": body, "more_body": False}
        return {"type": "http.disconnect"}

    sent: list[dict[str, Any]] = []

    async def send(message: dict[str, Any]) -> None:
        # A server's send yields to the loop, where the disconnect's
        # cancellation lands before the body is pulled.
        await asyncio.sleep(0)
        sent.append(message)

    path = "/api/studio/turn"
    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "root_path": "",
        "headers": [
            (b"authorization", AUTH["Authorization"].encode()),
            (b"content-type", b"application/json"),
            (b"content-length", str(len(body)).encode()),
        ],
        "client": ("127.0.0.1", 1),
        "server": ("test", 80),
    }
    await asyncio.wait_for(app(scope, receive, send), timeout=10)

    assert not any(message.get("body") for message in sent)
    assert not store.lock_for(PRINCIPAL, CONVERSATION).locked()
    stored = await StudioStore(_engine()).load(PRINCIPAL, CONVERSATION)
    assert stored is not None and stored.segment is None and not stored.turn_recorded
    async with _client(_app(_text("Again."))) as client:
        again = await _stream(client, "/api/studio/turn", _turn_body("again"))
    assert _frame(again, "done")
    closed = await StudioStore(_engine()).load(PRINCIPAL, CONVERSATION)
    assert closed is not None and closed.turns[0]["error"] == "abandoned"


async def test_a_pause_frame_is_not_sent_when_its_closing_write_fails() -> None:
    """When the closing write after the pause fails, the stream ends with
    the transient frame and the client never receives `plan.proposed`."""
    app = _app(*_plan(), tool_surface=ScriptedSurface())
    _close_as(app.state.studio.store, SegmentClose.FAILED)

    async with _client(app) as client:
        frames = await _stream(client, "/api/studio/turn", _turn_body("q with forks"))

    assert _events(frames)[-1] == "error"
    assert "plan.proposed" not in _events(frames)
    error = _frame(frames, "error")["error"]
    assert error["error"] == "transient_unavailable"
    assert error["detail"] == classed_sentence("transient_unavailable", "retry_after_delay")
    assert is_user_sentence(error["detail"])


async def test_a_stream_that_lost_its_segment_ends_its_span_as_lost(
    otlp_spans: InMemorySpanExporter, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The closing write finds the lease another instance's, so nothing more
    of the turn is stored. The segment's span ends once, `lost` at warning
    level, and is exported before the stream ends."""
    app = _app(_text("Hello."))
    ended = _recording_span_ends(monkeypatch)
    exported_at_flush: list[int] = []

    async def flush() -> None:
        exported_at_flush.append(_turn_spans(otlp_spans))

    monkeypatch.setattr(studio_stream_module, "flush_telemetry", flush)
    _close_as(app.state.studio.store, SegmentClose.LOST)

    async with _client(app) as client:
        frames = await _stream(client, "/api/studio/turn", _turn_body("hi"))

    error = _frame(frames, "error")["error"]
    assert (error["error"], error["retry"]) == ("transient_unavailable", "retry_after_delay")
    assert ended == [("lost", "WARNING")]
    assert exported_at_flush == [1]


@pytest.mark.parametrize(
    "failure",
    [
        pytest.param(SegmentLost("taken over"), id="taken-over"),
        pytest.param(
            DatabaseUnavailable("unreachable until the lease's deadline"),
            id="database-unavailable",
        ),
    ],
)
async def test_a_card_fenced_out_by_a_takeover_ends_the_turn_as_transient(
    caplog: pytest.LogCaptureFixture, failure: Exception
) -> None:
    """Another instance took the conversation over while this stream drew a
    card, so the card's canvas write is fenced out. The stream ends as at a
    fenced-out checkpoint: a transient error, the turn left open for the
    instance that owns it, and no internal fault in the log. A write that
    could not reach the database before the lease's deadline ends it the
    same way, leaving the turn as stored, since no close could be saved."""
    app = _app(*_card(), resolver=_search())
    store = app.state.studio.store

    async def taken_over(*args: Any, **kwargs: Any) -> Any:
        raise failure

    store.change_canvas = taken_over  # type: ignore[method-assign]

    with caplog.at_level("WARNING"):
        async with _client(app) as client:
            frames = await _stream(client, "/api/studio/turn", _turn_body())

    error = _frame(frames, "error")["error"]
    assert (error["error"], error["retry"], error["retry_after_seconds"]) == (
        "transient_unavailable",
        "retry_after_delay",
        STREAM_RETRY_AFTER_S,
    )
    stored = await store.load(PRINCIPAL, CONVERSATION)
    assert stored is not None
    assert not stored.turn_recorded
    assert not [r for r in caplog.records if "studio turn failed" in r.getMessage()]


async def test_an_edited_card_fenced_out_by_a_takeover_ends_the_turn_as_transient(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The Edit Card route's own canvas write, fenced out by a takeover, ends
    its stream as the turn route's does."""
    revised_component = {
        **COMPONENT,
        "revision": 2,
        "fact_summary": {**COMPONENT["fact_summary"], "revision": 2},
    }
    executor = ScriptedExecutor(
        [
            *_card_results(write_token=WRITE_TOKEN),
            _described(),
            _VIEW,
            {"outcome": "ok", "frame_refs": [], "component": revised_component},
        ],
        strict=True,
    )
    edit_model = ScriptedEditModel(
        {"text": "Renamed the card.", "patch": {"title": "Aid to Kenya"}, "recreate": None}
    )
    app = _app(*_card(), resolver=_search(), executor=executor, edit_model=edit_model)
    store = app.state.studio.store

    async def taken_over(*args: Any, **kwargs: Any) -> Any:
        raise SegmentLost("taken over")

    async with _client(app) as client:
        await _conversation_with_component(client)
        store.change_canvas = taken_over  # type: ignore[method-assign]
        with caplog.at_level("WARNING"):
            frames = await _stream(client, "/api/studio/turn", _edit_body("rename it"))

    error = _frame(frames, "error")["error"]
    assert (error["error"], error["retry"]) == ("transient_unavailable", "retry_after_delay")
    stored = await store.load(PRINCIPAL, CONVERSATION)
    assert stored is not None
    assert not stored.turn_recorded
    assert not [r for r in caplog.records if "studio edit turn failed" in r.getMessage()]


async def test_a_remote_composition_write_during_a_stream_reaches_its_done_frame_and_the_ledger() -> (
    None
):
    """A remote instance's `/composition` write, landing mid-stream before
    A's second turn finishes, is merged in by the checkpoint that runs
    before `done`: A's own Done frame carries it, not just a later reload
    from the ledger, pinned on the composition manifest specifically."""
    app_b = _app()

    async with _client(app_b) as client_b:
        executor = _PinsTheFirstCardFromB(client_b)
        app_a = _app(
            *_card(),
            *_card(),
            resolver=_search(2),
            executor=executor,
        )
        async with _client(app_a) as client_a:
            executor.first_manifest = await _conversation_with_component(client_a)
            second = await _stream(client_a, "/api/studio/turn", _turn_body("now for Uganda"))
            assert _events(second)[-1] == "done"

    pinned_component_id = executor.first_manifest["blocks"][0]["component_id"]
    done_block = next(
        b
        for b in _frame(second, "done")["composition"]["blocks"]
        if b["component_id"] == pinned_component_id
    )
    assert done_block["pinned"] is True

    reloaded = await StudioStore(
        _engine(),
    ).load(PRINCIPAL, CONVERSATION)
    assert reloaded is not None
    assert reloaded.composition is not None
    ledger_block = next(
        b for b in reloaded.composition.blocks if b.component_id == pinned_component_id
    )
    assert ledger_block.pinned is True


def test_digest_asks_stay_whole_and_older_answers_go_before_the_oldest_turn() -> None:
    state = ConversationState(conversation_id=CONVERSATION)
    for index in range(10):
        state.turns.append({"user": f"q{index} " + "u" * 900, "assistant": "a" * 900})

    digest = state.conversation_digest_state()

    assert digest is not None
    assert len(json.dumps(digest)) <= 5_000
    # Whole asks run about 900 bytes, so once the older answers are gone the
    # 5,000-byte cap keeps the newest four turns, dropping the oldest first.
    assert [turn["user"] for turn in digest["turns"]] == [
        f"q{i} " + "u" * 900 for i in range(6, 10)
    ]
    assert [turn["assistant"] for turn in digest["turns"]] == ["", "", "", "a" * 900]
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


async def test_a_seeded_turn_stores_its_seed_and_answers_without_searching_or_minting(
    definition_store: InMemoryDefinitionStore,
) -> None:
    search = CannedSearch()
    executor = ScriptedExecutor(strict=True)
    app = _app(resolver=search, executor=executor, served=[_seed_spec()])
    async with _client(app) as client:
        frames = await _stream(
            client, "/api/studio/turn", _seeded_turn_body("top 5 donors for education aid")
        )

    assert search.calls == []
    assert executor.calls == []
    assert await definition_store.get(_SEED_SPEC, schema_version=1) is not None
    steps = [frame for frame in frames if frame["event"] == "activity.step" and frame.get("tool")]
    assert (steps[0]["tool"], steps[0]["status"], steps[0]["label"]) == (
        "receive_seeds",
        "complete",
        "Receive the datasets from Search",
    )
    assert _events(frames)[-1] == "done"


async def test_a_single_reference_seed_spec_runs_the_question_and_shows_it_was_not_used() -> None:
    """The shape `seed_spec` was reserved with is accepted, as it was: the turn
    answers the question, and a failed step says the reference seeded nothing."""
    body = {
        **_turn_body("life expectancy in Kenya"),
        "seed_spec": {"spec_id": _SEED_SPEC, "content_version": "cv_" + "0" * 32},
    }
    async with _client(_app()) as client:
        frames = await _stream(client, "/api/studio/turn", body)

    assert _events(frames)[-1] == "done"
    (failed,) = [
        frame
        for frame in frames
        if frame["event"] == "activity.step" and frame["status"] == "failed"
    ]
    assert failed["label"] == "Receive the datasets from Search"
    assert failed["detail"] == (
        "The datasets from Search could not be used, so the answer starts from your question"
    )


async def test_a_seed_past_the_size_cap_fails_its_step_and_the_turn_answers_the_question(
    definition_store: InMemoryDefinitionStore,
) -> None:
    body = _seeded_turn_body("top 5 donors for education aid")
    body["seed_spec"][0]["spec"]["trace"] = {"padding": "x" * MAX_SEED_SPEC_BYTES}
    async with _client(_app()) as client:
        frames = await _stream(client, "/api/studio/turn", body)

    steps = [frame for frame in frames if frame["event"] == "activity.step"]
    (failed,) = [step for step in steps if step["status"] == "failed"]
    assert failed["label"] == "Receive the datasets from Search"
    assert failed["detail"] == "This Search result was left out"
    assert await definition_store.get(_SEED_SPEC, schema_version=1) is None
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


class _OpenedTurns:
    """Records the inputs and opening state of each turn `app` opens."""

    def __init__(self, app: FastAPI) -> None:
        self.turns: list[tuple[dict[str, Any], dict[str, Any]]] = []
        turns = app.state.studio.turns
        open_turn = turns.open_turn

        async def recording(*args: Any, **kwargs: Any) -> Any:
            self.turns.append((kwargs["inputs"], kwargs["state"]))
            return await open_turn(*args, **kwargs)

        turns.open_turn = recording


async def test_a_seeded_turn_s_session_keeps_only_what_the_seed_node_reads_of_each_seed(
    definition_store: InMemoryDefinitionStore,
) -> None:
    """The inputs a seeded turn opens with hold each seed's spec_id,
    definition, StatVar dcids and operations, and for a seed the harness
    cannot take, the refusal its step fails with at its own index."""
    del definition_store
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
        served=[_KENYA_SPEC],
    )
    opened = _OpenedTurns(app)
    async with _client(app) as client:
        frames = await _stream(client, "/api/studio/turn", body)

    ((inputs, _),) = opened.turns
    kept, refused = inputs["seeds"]
    assert _member_paths(kept) == {
        "spec.spec_id",
        "spec.definition.schema_version",
        "spec.definition.spec_id",
        "spec.definition.identity",
        "operations[].verb",
        "operations[].phrase",
        "operations[].over",
        "operations[].target_year",
    }
    assert kept["spec"]["definition"] == _KENYA_SPEC["definition"]
    assert set(refused) == {"refused"}
    assert refused["refused"]["code"] == "seed_invalid"
    assert refused["refused"]["detail"].startswith("seed 1 cannot be used")
    (failed,) = [
        frame
        for frame in frames
        if frame["event"] == "activity.step" and frame["status"] == "failed"
    ]
    assert failed["detail"] == "This Search result was left out"
    assert _events(frames)[-1] == "done"


def _malformed_seed(
    *, identity: Mapping[str, Any] | None = None, spec_id: str | None = None
) -> dict[str, Any]:
    """A seed that parses as `harness.wire.Seed`, whose `identity` (by default
    `_SEED_IDENTITY` for another reading) holds a member of a type the seed's
    readers cannot take. Its identity hashes to its spec_id unless `spec_id`
    is given."""
    identity = dict(identity or {**_SEED_IDENTITY, "seed": "malformed"})
    spec_id = spec_id or spec_id_for(canonicalize(identity))
    return {
        "spec": {
            "spec_id": spec_id,
            "definition": {"schema_version": 1, "spec_id": spec_id, "identity": identity},
            "resolved": {"rendered_sentence": "A malformed seed"},
        },
        "operations": [],
    }


@pytest.mark.parametrize(
    "malformed",
    [
        _malformed_seed(identity={**_SEED_IDENTITY, "requested_window": "x"}),
        _malformed_seed(identity={**_SEED_IDENTITY, "count": 2**70}, spec_id="spec_" + "0" * 32),
        _malformed_seed(identity={**_SEED_IDENTITY, "entities": ["country/KEN"]}),
        _malformed_seed(identity={**_SEED_IDENTITY, "shape": "statvar_schema"}),
    ],
    ids=[
        "requested-window-a-string",
        "identity-integer-past-2-53",
        "identity-entities-a-list",
        "identity-shape-a-string",
    ],
)
async def test_a_seed_the_harness_or_search_cannot_take_fails_its_step_and_the_other_is_kept(
    definition_store: InMemoryDefinitionStore, malformed: dict[str, Any]
) -> None:
    """A seed the harness cannot read is refused as it is taken; one whose
    identity Search refuses is dropped."""
    search = CannedSearch()
    body = _seeded_turn_body("top 5 donors for education aid")
    body["seed_spec"].insert(0, malformed)
    async with _client(_app(resolver=search, served=[_seed_spec()])) as client:
        frames = await _stream(client, "/api/studio/turn", body)

    assert "error" not in _events(frames)
    steps = [frame for frame in frames if frame["event"] == "activity.step"]
    (failed,) = [step for step in steps if step["status"] == "failed"]
    assert failed["label"] == "Receive the datasets from Search"
    assert failed["detail"] == "This Search result was left out"
    malformed_spec_id = malformed["spec"]["spec_id"]
    assert await definition_store.get(malformed_spec_id, schema_version=1) is None
    assert await definition_store.get(_SEED_SPEC, schema_version=1) is not None
    assert search.calls == []
    assert _events(frames)[-1] == "done"


async def test_a_request_past_the_body_cap_is_refused_before_it_is_parsed() -> None:
    search = CannedSearch()
    executor = ScriptedExecutor()
    body = _seeded_turn_body("top 5 donors for education aid")
    body["seed_spec"][0]["spec"]["trace"] = {"padding": "x" * MAX_TURN_REQUEST_BYTES}
    async with _client(_app(resolver=search, executor=executor)) as client:
        response = await client.post("/api/studio/turn", json=body, headers=AUTH)

    assert response.status_code == 413
    assert response.json()["error"] == "resource_exhausted"
    assert "6,356,992-byte cap" in response.json()["detail"]
    assert executor.calls == []
    assert search.calls == []


async def test_a_turn_session_is_seeded_with_earlier_readings_and_no_block() -> None:
    """The turn opens with the earlier turns' readings, warnings and the
    spec_ids it can pin from a stored block, without the block."""
    store = StudioStore(
        _engine(),
    )
    app = _app(store=store)
    opened = _OpenedTurns(app)
    spec_id = "spec_" + "a" * 32
    reading = SearchReading(text="aid", status="definite", spec_id=spec_id).model_dump(mode="json")
    conversation = await store.create(PRINCIPAL, CONVERSATION)
    conversation.turn_index = 1
    conversation.turn_recorded = True
    conversation.turns = [
        {
            "user": "q1",
            "assistant": "a1",
            "readings": [reading],
            "specs": {spec_id: {"spec_id": spec_id, "note": "stored block"}},
        }
    ]
    await store.save(PRINCIPAL, CONVERSATION)

    async with _client(app) as client:
        await _stream(client, "/api/studio/turn", _turn_body())

    ((inputs, state),) = opened.turns
    assert "stored block" not in json.dumps([inputs, state], default=str)
    assert state.get(ORCHESTRATOR_PRIOR_LEDGER_KEY) == {
        "readings": [reading],
        "warnings": {spec_id: []},
        "spec_ids": [spec_id],
    }


def test_the_turn_keeps_the_plan_records_readings_and_spec_blocks_its_events_write() -> None:
    conversation = ConversationState(conversation_id="conv_keep")
    delta = {
        ORCHESTRATOR_PLAN_RECORDS_KEY: [{"plan": 1}],
        ORCHESTRATOR_SEARCHES_KEY: [{"reading": 1}],
        ORCHESTRATOR_SPECS_KEY: {"spec_a": {"spec_id": "spec_a"}},
    }

    keep_turn_records(conversation, delta)

    assert conversation.turn_plans == [{"plan": 1}]
    assert conversation.turn_readings == [{"reading": 1}]
    assert conversation.turn_specs == {"spec_a": {"spec_id": "spec_a"}}


@pytest.mark.parametrize("closer", ["cancel", "next_turn"])
async def test_a_paused_turn_closed_without_a_stream_keeps_its_stored_spec_blocks(
    closer: str,
) -> None:
    """The record a turn paused on a question closes with, by `/cancel` or by
    the next `/turn` abandoning it, holds the spec blocks the turn's rows
    hold, as a turn closed any other way does."""
    app = _app(
        search_call("aid to Kenya"),
        *_plan("q with forks", *_ask()),
        resolver=CannedSearch(definite_envelope(spec_id=SCRIPTED_SPEC)),
        executor=ScriptedExecutor(),
        tool_surface=ScriptedSurface(),
    )
    async with _client(app) as client:
        first = await _stream(client, "/api/studio/turn", _turn_body("q with forks"))
        plan = _frame(first, "plan.proposed")
        paused = await _stream(
            client, "/api/studio/interrupt", _interrupt_body(plan["interrupt_id"], "approve")
        )
        assert _events(paused)[-1] == "elicitation.request"
        cursor = await app.state.studio.turns.open_cursor(PRINCIPAL, CONVERSATION)
        assert cursor is not None
        stored = await app.state.studio.turns.specs(cursor)
        assert stored

        if closer == "cancel":
            cancel = await client.post(
                "/api/studio/cancel",
                json={"kind": "turn.cancel", "schema_version": 1, "conversation_id": CONVERSATION},
                headers=AUTH,
            )
            assert cancel.status_code == 204
        else:
            await _stream(client, "/api/studio/turn", _turn_body("another question"))

    record = app.state.studio.store.current(PRINCIPAL, CONVERSATION).turns[0]
    assert record["error"] == ("cancelled" if closer == "cancel" else "abandoned")
    assert record["specs"] == stored
