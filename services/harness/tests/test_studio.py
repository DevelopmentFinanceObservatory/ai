"""The studio routes end-to-end over real HTTP, with scripted seams.

Everything runs against a FastAPI app through httpx2's ASGI transport, with
real routing, real SSE streaming and real ADK run/resume mechanics, on the
orchestrator topology `orchestrator_fakes.orchestrator_app` wires as
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
import json
from collections.abc import AsyncGenerator, Callable, Mapping, Sequence
from datetime import UTC, datetime, timedelta, tzinfo
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import anyio
import harness.graph.execute as execute_module
import harness.graph.planning as planning_module
import harness.graph.switch as switch_module
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
    InMemoryDefinitionStore,
    canonicalize,
    spec_id_for,
)
from data_mcp.definition_store import set_store as set_definition_store
from data_mcp.server import mcp as data_mcp
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from google.adk.events import Event, EventActions
from google.adk.models.llm_response import LlmResponse
from google.adk.sessions import BaseSessionService, InMemorySessionService
from google.genai import types
from harness.conversation import current_conversation_id
from harness.graph.state import (
    AGENT_NAME,
    ORCHESTRATOR_PLAN_RECORDS_KEY,
    ORCHESTRATOR_PRIOR_LEDGER_KEY,
    ORCHESTRATOR_SEARCHES_KEY,
    ORCHESTRATOR_SPECS_KEY,
    frame_id_for,
)
from harness.mcp_client import McpExecutor
from harness.mcp_deadlines import McpTimeoutPolicy
from harness.provenance import ProvenanceStore
from harness.scripted import ScriptedSurface
from harness.sentences import FAILURE_CLASS, SeamCode, refusal_sentence
from harness.settings import Settings
from harness.studio import (
    _acquire_or_conflict,
    _end_stream,
    _keep_turn_records,
    _sse_stream,
    _unwind,
)
from harness.studio_state import (
    ConversationState,
    StudioStore,
    _deserialize_pending,
    _PendingChoice,
    _PendingInterrupt,
    _PendingOption,
    land_component,
)
from harness.studio_translate import _RecordingExecutor, _stream_fatal
from harness.tool_contracts import HarnessCode, ResolverError
from harness.turn_budget import bounded
from harness.turn_records import SearchReading
from harness.wire import (
    MAX_SEED_SPEC_BYTES,
    MAX_TURN_REQUEST_BYTES,
    ActivityStep,
    ComponentType,
    CompositionActor,
    CompositionBlock,
    ErrorCode,
    ErrorFrame,
    MessageDelta,
    PlanDecision,
    PlanDecisionOption,
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

from .data_mcp_definitions import DataMcpDefinitions, SearchStub
from .failure_text import classed_sentence, is_user_sentence
from .orchestrator_fakes import (
    AUTH,
    CONVERSATION,
    CannedSearch,
    ScriptLlm,
    call,
    chart,
    definite_envelope,
    fetch_call,
    make_charts_call,
    orchestrator_app,
    plan_call,
    request_text,
    search_call,
    stream,
    studio_client,
    turn_body,
)
from .resolved_revision import resolved_revision
from .sqlite_session import sqlite_session_service
from .test_graph import _hashed_scope, _switched_answer

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
        search_call("kenya oda 2023", "aid to Kenya"),
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
        call(
            "ask_user",
            questions=[
                {"question": question, "options": [{"label": option} for option in options]}
                for question, options in questions
            ],
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


# The session service backend every route-exercising test in this module
# builds its store and runner over: `_new_session_service` is the one seam
# `session_service_backend` (below) switches, so the whole suite runs
# through `InMemorySessionService` and a sqlite-backed
# `DatabaseSessionService` without any call site naming which one it is
# under.
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
    *script: types.Part,
    resolver: Any = None,
    executor: Any = None,
    planner: Any = None,
    tool_surface: ToolSurface | None = None,
    settings: Settings | None = None,
    edit_model: Any | None = None,
    session_service: BaseSessionService | None = None,
    store: StudioStore | None = None,
    served: Sequence[Mapping[str, Any]] = (),
    definition_surface: Any = None,
) -> FastAPI:
    """`orchestrator_app` over this test's session backend, its agent's
    model answering from `script` and kept as `app.state.model`. The
    executor defaults to this module's `ScriptedExecutor`; the planner to
    `ScriptedPlannerModel`. A cross-instance test passes one
    `session_service`, shared with a second `_app`; a segment TTL scenario
    passes its own `store` with its own clock."""
    app, model = orchestrator_app(
        *script,
        resolver=resolver,
        executor=executor or ScriptedExecutor(),
        planner=planner,
        tool_surface=tool_surface,
        session_service=session_service or _new_session_service(),
        store=store,
        settings=settings,
        edit_model=edit_model,
        served=served,
        definition_surface=definition_surface,
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
        search_call("kenya oda 2023", "aid to Kenya"),
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


async def test_a_turn_past_its_call_cap_ends_on_the_cap_sentence_with_no_further_model_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The route makes the request's budget current, and a model call past
    its cap answers the cap's sentence (`harness.turn_budget.bounded`)
    without asking the model: the sentence is the turn's final message."""
    original = ScriptLlm.generate_content_async

    async def bounded_script(
        self: ScriptLlm, llm_request: Any, stream: bool = False
    ) -> AsyncGenerator[LlmResponse]:
        async for response in bounded(lambda: original(self, llm_request, stream), stream=stream):
            yield response

    monkeypatch.setattr(ScriptLlm, "generate_content_async", bounded_script)
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
        {"code": SeamCode.TURN_MODEL_CALLS_SPENT}
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


async def test_a_planner_run_shows_working_on_a_plan_and_not_the_seams_late_reports(
    real_schemas: list[ToolSchema],
) -> None:
    """A plan call opens "Working on a plan" as the draft starts and
    completes it before the plan card. The seam's `on_progress` reports,
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

    # The plan's own step; "Starting the plan" is a status line with no tool.
    finished = [
        frame for frame in failed if frame["event"] == "activity.step" and frame.get("tool")
    ]
    assert [frame["status"] for frame in finished] == ["running", "complete"]
    assert finished[0]["step_id"] == finished[1]["step_id"]
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
    """A draft the seam could not itself validate must not reach the approve
    gate or the executor. `harness.graph.planning`'s `planner` node validates the seam's
    record as a `planner.plan.Plan` before writing it to state, outside
    `handing_back_planner_failures`, which hands back only a seam's own
    `PlannerSeamError`."""

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


async def test_plan_proposed_pauses_and_approve_resumes_on_a_new_stream() -> None:
    """The scripted planner and surface (`harness.scripted`'s seams)
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


async def test_a_turn_paused_for_approval_exports_as_one_trace(
    otlp_spans: InMemorySpanExporter,
) -> None:
    """`/turn` and the `/interrupt` that resumes it are separate requests on
    separate tasks. Spans and events made in the search of the first segment
    and in the executor of the resumed one all nest under the one `turn`
    span, whose `dfo.external_id` is the turn's wire `trace_id`."""

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
        search_call("q with forks", "aid to Kenya"),
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
    (turn,) = [span for span in spans if span.name == "turn"]
    assert (turn.attributes or {})["dfo.external_id"] == _frame(second, "done")["trace_id"]
    assert {span.context.trace_id for span in spans} == {turn.context.trace_id}
    assert len([span for span in spans if span.name == "resolve_probe"]) == 1
    executor_probes = [span for span in spans if span.name == "executor_probe"]
    assert len(executor_probes) == 2  # describe_frame, then write_component
    # `turn` is the only root: the probes of both segments sit below it.
    assert [span.name for span in spans if span.parent is None] == ["turn"]
    assert [event.name for event in executor_probes[-1].events] == ["executor.probed"]


async def test_every_stream_flushes_telemetry_before_the_response_ends(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    flushes: list[str] = []

    async def flush() -> None:
        flushes.append("flush")

    monkeypatch.setattr(studio_module, "flush_telemetry", flush)
    app = _app(*_plan())
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
    app = _app(*_plan())
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
    async with _client(_app()) as client:
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
    # The seam drafted for the agent's brief.
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
    assert [(status, detail) for _, _, status, detail in get_data] == [
        ("running", _POLLING),
        ("running", _RECEIVED),
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
        label="planner: propose_plan",
        status=StepStatus.RUNNING,
        tool="propose_plan",
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
    assert frames[2]["detail"] == frames[3]["error"]["detail"]
    assert frames[2]["detail"] == classed_sentence("cancelled", "do_not_retry")
    assert is_user_sentence(frames[2]["detail"])
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
                detail="The fetch did not finish.",
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
    # The line closes on the reason the turn ended for.
    assert frames[1]["detail"] == frames[2]["error"]["detail"]
    assert [f["seq"] for f in frames] == [0, 2, 3]


async def test_a_segment_lost_closes_the_activity_lines_its_stream_left_running() -> None:
    """A takeover ends the stream mid-step, same as a Stop: the step's line
    would otherwise stay pending forever in Studio, since website
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
    assert frames[1]["detail"] == frames[2]["error"]["detail"]
    assert frames[1]["detail"] == classed_sentence("transient_unavailable", "retry_after_delay")
    assert is_user_sentence(frames[1]["detail"])
    assert frames[1]["duration_ms"] >= 0
    assert frames[2]["error"]["error"] == "transient_unavailable"
    assert [f["seq"] for f in frames] == sorted({f["seq"] for f in frames})
    assert conversation.turn_recorded is False  # the turn stays open for the takeover to finish


async def test_an_approval_the_approve_node_refuses_sends_no_approved_plan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The approve node takes no approval whose time it cannot read, and routes the
    answer to the planner as an amendment: the stream offers the plan again and
    names no approved plan."""
    monkeypatch.setattr(planning_module, "_answered_at", lambda answer: None)
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
    """A restart the planner's `question_fixed` pair refuses (`QUESTION_FIXED`)
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
    session_service = _new_session_service()
    store = StudioStore(session_service, app_name=AGENT_NAME)
    app = _app(*_plan(), session_service=session_service, store=store)

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
    session_service = _new_session_service()
    store = StudioStore(session_service, app_name=AGENT_NAME)
    app = _app(
        *_plan(),
        session_service=session_service,
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
    session_service = _new_session_service()
    store = StudioStore(session_service, app_name=AGENT_NAME)
    app = _app(*_plan(), session_service=session_service, store=store)

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

        await _stream(
            client, "/api/studio/interrupt", _interrupt_body(plan["interrupt_id"], "approve")
        )

    # The retry reaches `ApprovalRecord` unpatched: the record is appended
    # and the pending entry it was keyed on is finally consumed. (This
    # setup has no `tool_surface`, so the resumed turn's own execution
    # refuses downstream -- irrelevant to what this test is proving.)
    assert len(conversation.approvals) == 1
    assert plan["interrupt_id"] not in conversation.pending


async def test_a_raise_before_the_turn_streams_releases_the_conversation_lock() -> None:
    """The per-turn session cannot be created, so `/turn` raises before its
    stream starts and the lock the claim took is released."""
    store = StudioStore(_new_session_service(), app_name=AGENT_NAME)
    app = _app(_text("Hello."), store=store)

    async def _unavailable(**kwargs: Any) -> None:
        raise RuntimeError("session backend unavailable")

    app.state.studio._ensure_session = _unavailable
    async with _client(app) as client:
        with pytest.raises(RuntimeError, match="session backend unavailable"):
            await _stream(client, "/api/studio/turn", _turn_body("hi"))

    assert not store.current(PRINCIPAL, CONVERSATION).lock.locked()


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
    assert "The user chose: reading B" in request_text(_model(app).requests[-1])
    (record,) = app.state.studio.store.current(PRINCIPAL, CONVERSATION).turns
    assert record["user"] == "ambiguous q\n(then: reading B)"


async def test_a_search_after_the_answer_reads_the_message_the_answer_refined() -> None:
    """Each segment's searches send the turn's message as it then stands: the
    question first, and after the resume the question with the answer."""
    search = CannedSearch(definite_envelope(spec_id=SPEC_A), definite_envelope(spec_id=SPEC_A))
    app = _app(
        search_call("ambiguous q", "aid"),
        *_ask("reading A", "reading B"),
        search_call("aid, reading B", "aid"),
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
    asked = request_text(_model(app).requests[-1])
    assert "1. How should the aid be counted? The user chose: Gross" in asked
    assert "2. Which prices? The user left this to you. Go on with Constant." in asked
    (record,) = app.state.studio.store.current(PRINCIPAL, CONVERSATION).turns
    assert record["user"] == "ambiguous q\n(then: How should the aid be counted? Gross)"


async def test_the_turn_record_keeps_a_picked_count_under_its_spec_id() -> None:
    """A pick whose option carries a spec_id is recorded as `label (spec_id)`,
    so a later turn can tell which series the user settled."""
    app = _app(
        search_call("kenya oda 2023", "aid to Kenya"),
        call(
            "ask_user",
            questions=[
                {
                    "question": "How should the aid be counted?",
                    "options": [{"label": "Net", "spec_id": SPEC_A}, {"label": "Gross"}],
                }
            ],
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
    stream that emitted the pause has already ended, so `cancel_event` is
    back to `None` and cannot be the signal cancel acts on. The fix must
    still refuse a later `/interrupt` on that id, and must not resume the
    workflow. The test proves it by checking the agent's model is not asked again."""
    app = _app(*_ask("reading A", "reading B"))
    async with _client(app) as client:
        first = await _stream(client, "/api/studio/turn", _turn_body("ambiguous q"))
        elicitation = _frame(first, "elicitation.request")
        asked = len(_model(app).requests)

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
    session_service = _new_session_service()
    store = StudioStore(session_service, app_name=AGENT_NAME)
    app = _app(*_ask(), *_ask(), session_service=session_service, store=store)

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
    session_service = sqlite_session_service(tmp_path / "shared.db")
    store_a = StudioStore(session_service, app_name=AGENT_NAME, clock=lambda: clock[0])
    store_b = StudioStore(session_service, app_name=AGENT_NAME, clock=lambda: clock[0])
    app_a = _app(
        *_plan(),
        tool_surface=ScriptedSurface(),
        session_service=session_service,
        store=store_a,
    )

    async with _client(app_a) as client_a:
        first = await _stream(client_a, "/api/studio/turn", _turn_body("q with forks"))
        plan = _frame(first, "plan.proposed")

        probe = StudioStore(session_service, app_name=AGENT_NAME, clock=lambda: clock[0])
        saved_seq = (await probe.load(PRINCIPAL, CONVERSATION)).seq

        await _approve_plan_then_hang(client_a, app_a, plan)

    app_b = _app(tool_surface=ScriptedSurface(), session_service=session_service, store=store_b)
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
    # A's dead attempt sent the approved plan once the approve node took it.
    assert "plan.approved" not in _events(resumed)

    reloaded = await StudioStore(session_service, app_name=AGENT_NAME).load(PRINCIPAL, CONVERSATION)
    assert reloaded is not None
    assert len(reloaded.approvals) == 1
    assert plan["interrupt_id"] not in reloaded.pending


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
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
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
    clock = [1_000.0]
    session_service = sqlite_session_service(tmp_path / "shared.db")
    store_a = StudioStore(session_service, app_name=AGENT_NAME, clock=lambda: clock[0])
    store_b = StudioStore(session_service, app_name=AGENT_NAME, clock=lambda: clock[0])
    app_a = _app(
        *_plan(), tool_surface=ScriptedSurface(), session_service=session_service, store=store_a
    )

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
            app_a.state.studio.store.release = _noop_release  # type: ignore[method-assign]
            cancel = await client_a.post(
                "/api/studio/cancel",
                json={"kind": "turn.cancel", "schema_version": 1, "conversation_id": CONVERSATION},
                headers=AUTH,
            )
            assert cancel.status_code == 204
            assert _events(await asyncio.wait_for(resume, timeout=5))[-1] == "error"

    surface = _CallingSurface()
    app_b = _app(tool_surface=surface, session_service=session_service, store=store_b)
    async with _client(app_b) as client_b:
        clock[0] += studio_state_module.SEGMENT_TTL_S + 1  # A's segment has expired
        await _stream(
            client_b, "/api/studio/interrupt", _interrupt_body(plan["interrupt_id"], "approve")
        )

    assert approved_at == [answered]
    assert surface.calls == []
    reloaded = await StudioStore(session_service, app_name=AGENT_NAME).load(PRINCIPAL, CONVERSATION)
    assert reloaded is not None
    assert [record.at for record in reloaded.approvals] == [answered]


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
        *_plan(),
        tool_surface=ScriptedSurface(),
        session_service=session_service,
        store=store_a,
    )

    async with _client(app_a) as client_a:
        first = await _stream(client_a, "/api/studio/turn", _turn_body("q with forks"))
        plan = _frame(first, "plan.proposed")
        await _approve_plan_then_hang(client_a, app_a, plan)

    app_b = _app(
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
    reloaded = await StudioStore(session_service, app_name=AGENT_NAME).load(PRINCIPAL, CONVERSATION)
    assert reloaded is not None
    assert plan["interrupt_id"] in reloaded.pending
    assert reloaded.pending[plan["interrupt_id"]].answer == {"decision": "approve", "payload": None}


@pytest.mark.parametrize(
    ("retried", "status"),
    [pytest.param(0, 200, id="same-answers"), pytest.param(1, 409, id="other-answers")],
)
async def test_a_retried_resume_of_an_ask_must_carry_the_answers_that_resumed_it(
    tmp_path: Path, retried: int, status: int
) -> None:
    """A's resume of an answered ask dies on its search, its segment lease
    left behind. Once the lease expires, B's retry resumes only with the
    answers A resumed with; other answers to the same question are refused
    for good."""
    clock = [1_000.0]
    session_service = sqlite_session_service(tmp_path / "shared.db")
    store_a = StudioStore(session_service, app_name=AGENT_NAME, clock=lambda: clock[0])
    store_b = StudioStore(session_service, app_name=AGENT_NAME, clock=lambda: clock[0])
    resolver = _HangingResolver()
    app_a = _app(
        *_ask("reading A", "reading B"),
        *_card(),
        resolver=resolver,
        session_service=session_service,
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
        app_a.state.studio.store.release = _noop_release  # type: ignore[method-assign]
        cancel = await client_a.post(
            "/api/studio/cancel",
            json={"kind": "turn.cancel", "schema_version": 1, "conversation_id": CONVERSATION},
            headers=AUTH,
        )
        assert cancel.status_code == 204
        await asyncio.wait_for(resume, timeout=5)

    app_b = _app(_text("Reading A."), session_service=session_service, store=store_b)
    async with _client(app_b) as client_b:
        clock[0] += studio_state_module.SEGMENT_TTL_S + 1  # A's segment has expired
        retry = await client_b.post(
            "/api/studio/interrupt",
            json=_interrupt_body(
                elicitation["interrupt_id"], "answer", answers={"q1": _pick(option_ids[retried])}
            ),
            headers=AUTH,
        )

    assert retry.status_code == status


async def test_two_concurrent_first_answers_to_one_interrupt_give_one_stream_and_one_409(
    tmp_path: Path,
) -> None:
    """A and B both answer the SAME fresh interrupt at once: whichever
    instance's claim save lands first resumes the turn; the other's save
    finds the ledger already moved and answers the existing 409, rather
    than each resuming its own segment of the turn."""
    session_service = sqlite_session_service(tmp_path / "shared.db")
    app_a = _app(
        *_plan(),
        tool_surface=ScriptedSurface(),
        session_service=session_service,
    )
    app_b = _app(
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
    interleave at `load`. One call succeeds; the other's `expected_revision`
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


async def test_a_persistently_stale_composition_is_an_edit_conflict_and_applies_nothing(
    tmp_path: Path,
) -> None:
    """A second store's own write lands first on every one of
    `mutate_composition`'s bounded retries, so all of them go stale: the
    route answers 409 rather than retrying forever, and the live
    conversation is left holding the last manifest it fetched, not the
    caller's rejected pin."""
    session_service = sqlite_session_service(tmp_path / "shared.db")
    app = _app(
        *_card(),
        resolver=_search(),
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
                await store_b.load(PRINCIPAL, CONVERSATION)
                store_b.current(PRINCIPAL, CONVERSATION).seq = attempts
                await store_b.save(PRINCIPAL, CONVERSATION)
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
    live = app.state.studio.store.current(PRINCIPAL, CONVERSATION)
    assert live.composition.blocks[0].pinned is False
    reloaded = await StudioStore(session_service, app_name=AGENT_NAME).load(PRINCIPAL, CONVERSATION)
    assert reloaded.composition.blocks[0].pinned is False


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

    app = _app(*_card(), resolver=_HangingResolver())
    studio = app.state.studio

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
    async with _client(_app()) as client:
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


async def test_a_switch_stream_arms_its_own_stop_and_clears_only_that_one() -> None:
    """A stream running a switch arms `switch_stop` and the `SWITCH_STOP` its
    frames read. Ending, it leaves alone a stop a later switch stream armed
    meanwhile, so an earlier stream's unwind cannot disarm the next switch."""
    seen: list[asyncio.Event | None] = []

    async def segment():
        seen.append(switch_module.SWITCH_STOP.get())
        yield MessageDelta(event="message.delta", seq=0, text="one")

    conversation = ConversationState(conversation_id=CONVERSATION)
    stream = _sse_stream(segment(), conversation=conversation, switch=True)
    await anext(stream)
    armed = conversation.switch_stop
    later = conversation.switch_stop = asyncio.Event()
    await stream.aclose()

    assert armed is not None
    assert seen == [armed]
    assert conversation.switch_stop is later


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


async def test_a_client_that_leaves_at_the_pause_frame_still_releases_the_segment() -> None:
    """The client closes its `/turn` stream right after `plan.proposed`, while
    the segment's end-of-segment save (`StudioStore.release`) is still on its
    read. Starlette, as uvicorn runs it, cancels the response on
    `http.disconnect`, and the stream cancels the segment task under that
    save. The save still lands, so the ledger holds no live segment and the
    `/interrupt` that answers the plan streams at once."""
    session_service = _new_session_service()
    store = StudioStore(session_service, app_name=AGENT_NAME)
    app = _app(*_plan(), session_service=session_service, store=store)
    releasing = asyncio.Event()
    real_release = store.release
    real_get_session = session_service.get_session

    async def release(*args: Any, **kwargs: Any) -> None:
        releasing.set()
        await real_release(*args, **kwargs)

    async def get_session(*args: Any, **kwargs: Any) -> Any:
        if releasing.is_set():
            await asyncio.sleep(0.05)  # the save's read is in flight at the disconnect
        return await real_get_session(*args, **kwargs)

    store.release = release  # type: ignore[method-assign]
    session_service.get_session = get_session  # type: ignore[method-assign]

    body = json.dumps(_turn_body()).encode()
    request_sent = False
    chunks: list[bytes] = []

    async def receive() -> dict[str, Any]:
        nonlocal request_sent
        if not request_sent:
            request_sent = True
            return {"type": "http.request", "body": body, "more_body": False}
        await releasing.wait()
        return {"type": "http.disconnect"}

    async def send(message: dict[str, Any]) -> None:
        if message["type"] == "http.response.body":
            chunks.append(message.get("body", b""))

    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/api/studio/turn",
        "raw_path": b"/api/studio/turn",
        "query_string": b"",
        "root_path": "",
        "headers": [
            (b"content-type", b"application/json"),
            (b"content-length", str(len(body)).encode()),
            (b"authorization", AUTH["Authorization"].encode()),
        ],
        "client": ("127.0.0.1", 50000),
        "server": ("studio.test", 80),
    }
    await asyncio.wait_for(app(scope, receive, send), timeout=10)

    first = [
        json.loads(line[len("data: ") :])
        for line in b"".join(chunks).decode().splitlines()
        if line.startswith("data: ")
    ]
    plan = _frame(first, "plan.proposed")
    ledger = await StudioStore(session_service, app_name=AGENT_NAME).load(PRINCIPAL, CONVERSATION)
    assert ledger is not None
    assert ledger.segment is None

    async with _client(app) as client:
        second = await _stream(
            client, "/api/studio/interrupt", _interrupt_body(plan["interrupt_id"], "reject")
        )
    assert _events(second)[-1] == "done"


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
    presents the create's token, and no SSE frame, digest,
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
    provenance = ProvenanceStore()
    provenance.put(conversation=f"{PRINCIPAL}/{CONVERSATION}", manifest={}, **REF_A)
    session_service = _new_session_service()
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

    reloaded = await StudioStore(session_service, app_name=AGENT_NAME).load(PRINCIPAL, CONVERSATION)
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


# --- the canvas digest -----------------------------------------------------------


def _digest(request: Any, label: str) -> dict[str, Any] | None:
    """The digest the agent's `request` carries under `label`, the heading
    `orchestrator.agent` prepends it with, or `None` when it carries none."""
    opening = f"{label}: "
    for line in request_text(request).split("\n"):
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
    assert "The user skipped the question." in request_text(_model(app).requests[-1])


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
    assert "The user answered in their own words: the OECD reading please" in request_text(
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
    assert "The user chose: reading B" in request_text(_model(app).requests[-1])


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


@pytest.mark.parametrize(
    ("stops", "deaf"),
    [(1, False), (2, True), (1, True)],
    ids=["one-stop", "a-second-stop", "no-offer-in-time"],
)
async def test_a_stop_while_a_switch_runs_offers_the_plan_unchanged_on_a_live_pause(
    monkeypatch: pytest.MonkeyPatch, stops: int, deaf: bool
) -> None:
    """POST /cancel while the switch's revision runs: the stream reads on and
    ends on the same plan re-offered under a fresh interrupt, carrying
    `switch_failed` `switch_cancelled`, which asking again can get past; the
    turn stays paused on it. A switch that does not hear the Stop ends as any
    Stop ends a stream, on a second Stop or once `SWITCH_REOFFER_SECONDS`
    pass with no offer."""
    entered = asyncio.Event()

    async def _revising(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        entered.set()
        await asyncio.sleep(5)
        raise AssertionError("a stopped revision is cancelled")

    async def _deaf(_ctx: Any, revision: Any, **_kwargs: Any) -> dict[str, Any]:
        # A switch that does not hear the first Stop.
        return await revision

    monkeypatch.setattr(switch_module, "_switched_plan", _revising)
    monkeypatch.setattr(studio_module, "SWITCH_REOFFER_SECONDS", 0.2 if stops == 1 else 4.0)
    if deaf:
        monkeypatch.setattr(switch_module, "_unless_stopped", _deaf)
    app = _app(*_plan())
    stop = {"kind": "turn.cancel", "schema_version": 1, "conversation_id": CONVERSATION}
    async with _client(app) as client:
        plan = await _plan_with_a_choice(app, client)
        conversation = app.state.studio.store.current(PRINCIPAL, CONVERSATION)
        resumed = asyncio.ensure_future(
            _stream(client, "/api/studio/interrupt", _switch_body(plan))
        )
        await asyncio.wait_for(entered.wait(), timeout=5)
        for _ in range(stops):
            armed = conversation.cancel_event
            assert armed is not None and not armed.is_set()
            cancelled = await client.post("/api/studio/cancel", json=stop, headers=AUTH)
            assert cancelled.status_code == 204
            while conversation.cancel_event is armed:
                await asyncio.sleep(0.01)
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
    assert conversation.switch_stop is None


@pytest.mark.parametrize("stops", [1, 2], ids=["one-stop", "a-second-stop"])
async def test_a_stop_from_another_instance_stops_a_switch_and_a_second_ends_its_stream(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stops: int
) -> None:
    """B POSTs `/cancel` while A's switch runs and the switch does not hear it:
    A's heartbeat reads B's durable flag as the Stop of the switch and keeps
    watching, so a second Stop from B ends A's stream with `cancelled` well
    before `SWITCH_REOFFER_SECONDS`. One Stop alone lets the stream read on."""
    monkeypatch.setattr(studio_module, "HEARTBEAT_INTERVAL_S", 0.05)
    monkeypatch.setattr(studio_module, "SWITCH_REOFFER_SECONDS", 30.0)
    entered = asyncio.Event()
    finish = asyncio.Event()

    async def _revising(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        entered.set()
        await finish.wait()
        raise switch_module.SwitchFailedError(
            "r", choice_id=_CHOICE_ID, option_id="sector:120+130", code="c"
        )

    async def _deaf(_ctx: Any, revision: Any, **_kwargs: Any) -> dict[str, Any]:
        return await revision

    monkeypatch.setattr(switch_module, "_switched_plan", _revising)
    monkeypatch.setattr(switch_module, "_unless_stopped", _deaf)
    session_service = sqlite_session_service(tmp_path / "shared.db")
    app_a = _app(*_plan(), session_service=session_service)
    app_b = _app(session_service=session_service)
    stop = {"kind": "turn.cancel", "schema_version": 1, "conversation_id": CONVERSATION}
    async with _client(app_a) as client_a, _client(app_b) as client_b:
        plan = await _plan_with_a_choice(app_a, client_a)
        conversation = app_a.state.studio.store.current(PRINCIPAL, CONVERSATION)
        resumed = asyncio.ensure_future(
            _stream(client_a, "/api/studio/interrupt", _switch_body(plan))
        )
        await asyncio.wait_for(entered.wait(), timeout=5)
        switch_stop = conversation.switch_stop
        assert switch_stop is not None
        for _ in range(stops):
            cancelled = await client_b.post("/api/studio/cancel", json=stop, headers=AUTH)
            assert cancelled.status_code == 204
            await asyncio.wait_for(switch_stop.wait(), timeout=5)
        if stops == 1:
            finish.set()
        frames = await asyncio.wait_for(resumed, timeout=5)

    if stops == 1:
        assert _events(frames)[-1] == "plan.proposed"
        return
    assert _events(frames)[-1] == "error"
    assert _frame(frames, "error")["error"]["error"] == "cancelled"


async def test_an_approved_plan_names_what_a_choice_held_before_the_turns_first_switch() -> None:
    """Health counted as CRS 120, switched to CRS 120 and 130, then to CRS
    130, then approved: `plan.approved` names CRS 120, the label on the card
    the first switch was made from, as what the choice changed from."""
    labels = {
        "sector:120": "health (CRS 120)",
        "sector:120+130": "health and population",
        "sector:130": "population (CRS 130)",
    }

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
            choice_id=_CHOICE_ID,
            active_option_id=active,
            options=[
                PlanDecisionOption(option_id=option, label=label, status="available")
                for option, label in labels.items()
            ],
        )
        assert pending.card is not None
        pending.card = pending.card.model_copy(update={"decisions": [decision]})
        pending.choices = {
            _CHOICE_ID: _PendingChoice(
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
        await _shows(app, card["interrupt_id"], "sector:120")
        for option, active in [("sector:120+130", "sector:120+130"), ("sector:130", "sector:130")]:
            frames = await _stream(
                client,
                "/api/studio/interrupt",
                _switch_body(card, option_id=option),
            )
            card = _frame(frames, "plan.proposed")
            await _shows(app, card["interrupt_id"], active)
        approved = await _stream(
            client,
            "/api/studio/interrupt",
            {
                **_interrupt_body(card["interrupt_id"], "approve"),
                "plan_revision": card["revision"],
            },
        )

    assert _frame(approved, "plan.approved")["choices"] == [
        {"question": "Sector", "chosen": "population (CRS 130)", "changed_from": "health (CRS 120)"}
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
        *_plan("q with forks", search_call("aid", "aid"), _text("Done.")),
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
        search_call("health aid to Africa", "health aid"),
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
        search_call("health aid to Africa", "health aid"),
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

    session_service = _new_session_service()
    store = StudioStore(session_service, app_name=AGENT_NAME)
    app = _app(*_plan(), tool_surface=BrokenSurface(), session_service=session_service, store=store)
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
    assert record["error"] == "transient_unavailable"


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
        ("refusal", HarnessCode.RAISED),
        ("refusal", "cancelled"),
    ]
    # Both are codes the failure table classes, and the exception stays in
    # the message.
    assert all(record.result["refusal"]["code"] in FAILURE_CLASS for record in records)
    assert "RuntimeError" in records[0].result["refusal"]["message"]


async def test_a_rejected_plan_turn_is_recorded() -> None:
    session_service = _new_session_service()
    store = StudioStore(session_service, app_name=AGENT_NAME)
    app = _app(*_plan(), session_service=session_service, store=store)
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
    session_service = _new_session_service()
    store = StudioStore(session_service, app_name=AGENT_NAME)
    app = _app(
        *_plan(),
        tool_surface=ScriptedSurface(),
        session_service=session_service,
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
    session_service = _new_session_service()
    store = StudioStore(session_service, app_name=AGENT_NAME)
    app = _app(*_plan(), session_service=session_service, store=store)
    async with _client(app) as client:
        paused = await _stream(client, "/api/studio/turn", _turn_body("q with forks"))
        assert paused[-1]["event"] == "plan.proposed"
        # A fresh turn supersedes the pause; the abandoned turn must not
        # vanish from history.
        second = await _stream(client, "/api/studio/turn", _turn_body("aid to kenya, quickly"))
        assert second[-1]["event"] == "done"

    records = store.current(PRINCIPAL, CONVERSATION).turns
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
        session_service=session_service,
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
    session_service = _new_session_service()
    store = StudioStore(session_service, app_name=AGENT_NAME)
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
        session_service=session_service,
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


async def test_a_component_missing_fact_summary_never_lands() -> None:
    """An `ok` write whose component fails the `component.created` wire
    contract (here, a missing `fact_summary`) is refused like any other
    unusable write. No frame announces it and the canvas is unchanged."""
    missing = {key: value for key, value in COMPONENT.items() if key != "fact_summary"}
    session_service = _new_session_service()
    store = StudioStore(session_service, app_name=AGENT_NAME)
    app = _app(
        *_card(),
        resolver=_search(),
        executor=ScriptedExecutor(_card_results(missing)),
        session_service=session_service,
        store=store,
    )
    async with _client(app) as client:
        frames = await _stream(client, "/api/studio/turn", _turn_body())

    assert _events(frames)[-1] == "done"
    assert "component.created" not in _events(frames)
    assert "composition" not in _frame(frames, "done")
    assert store.current(PRINCIPAL, CONVERSATION).composition is None


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
    session_service = _new_session_service()
    store = StudioStore(session_service, app_name=AGENT_NAME)
    executor = _WritesUntil(_hang)
    app = _app(
        *_card(),
        *_card(charts=2),
        resolver=_search(2),
        executor=executor,
        session_service=session_service,
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


async def test_first_write_ok_then_second_write_raises_still_announces_what_actually_landed() -> (
    None
):
    """`make_charts` answers a write that raised as a chart it could not draw
    and draws nothing after it, so the turn goes on. The first write's card
    is announced (`component.created` plus `composition.updated`) and lands
    for real: the server's own composition is exactly what the client was
    told. The second write, which never returned, produces no frame and
    never reaches the canvas."""
    session_service = _new_session_service()
    store = StudioStore(session_service, app_name=AGENT_NAME)
    app = _app(
        *_card(),
        *_card(charts=2),
        resolver=_search(2),
        executor=_WritesUntil(_reset),
        session_service=session_service,
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


async def test_a_plan_paused_on_one_instance_is_approved_and_continued_on_another(
    tmp_path: Path,
) -> None:
    """A pauses on the plan. B, a separate `StudioRunner` and `StudioStore`
    that never built this plan, answers `/interrupt` and completes the
    turn, restoring the plan's step order into its own `_RecordingSurface`
    from the ledger's `_PendingInterrupt.step_order` before resuming. A
    stale approve on B first gets back the card A sent, read from the
    ledger. A second `/turn` on B then continues the same conversation: the turn
    index advances past what A claimed, and the digest carries A's turn."""
    session_service = sqlite_session_service(tmp_path / "shared.db")
    app_a = _app(
        *_plan(),
        tool_surface=ScriptedSurface(),
        session_service=session_service,
    )
    # B's own model answers only after the hand-back: `/interrupt` resumes
    # the ADK invocation A's own `/turn` already started, in the per-turn
    # session A created, so B never sees the first turn's question.
    app_b = _app(tool_surface=ScriptedSurface(), session_service=session_service)

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
    state = await StudioStore(session_service, app_name=AGENT_NAME).load(PRINCIPAL, CONVERSATION)
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
        *_plan(),
        tool_surface=ScriptedSurface(),
        session_service=session_service,
    )
    app_b = _app(
        *_plan(),
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
    app_a = _app(session_service=session_service)
    app_b = _app(session_service=session_service)

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
    state = await StudioStore(session_service, app_name=AGENT_NAME).load(PRINCIPAL, CONVERSATION)
    assert state is not None
    assert state.turn_index == 2


async def test_a_composition_edit_on_one_instance_survives_a_concurrent_turns_release(
    tmp_path: Path,
) -> None:
    """B's `/composition` pins A's first turn's card while A's SECOND turn
    is still writing its own new one, mid-stream, before A's release. A's
    release finds the ledger has moved (B's pin) and replays A's own
    recorded append onto B's freshly pinned manifest instead of overwriting
    it with A's own stale copy -- B's edit survives."""
    session_service = sqlite_session_service(tmp_path / "shared.db")
    app_b = _app(session_service=session_service)

    async with _client(app_b) as client_b:
        executor = _PinsTheFirstCardFromB(client_b)
        app_a = _app(
            *_card(),
            *_card(),
            resolver=_search(2),
            executor=executor,
            session_service=session_service,
        )
        async with _client(app_a) as client_a:
            executor.first_manifest = await _conversation_with_component(client_a)
            second = await _stream(client_a, "/api/studio/turn", _turn_body("now for Uganda"))
            assert _events(second)[-1] == "done"

    reloaded = await StudioStore(session_service, app_name=AGENT_NAME).load(PRINCIPAL, CONVERSATION)
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


async def test_ledger_at_turn_n_with_no_stream_running_lets_the_next_turn_succeed(
    tmp_path: Path,
) -> None:
    """A conversation whose ledger holds a completed turn n, with its
    per-turn session `#t{n}` still existing, is idle rather than mid-claim:
    the next `/turn` claims n+1 and succeeds, on a SECOND instance that
    never saw the first."""
    session_service = sqlite_session_service(tmp_path / "shared.db")
    app_a = _app(session_service=session_service)
    app_b = _app(session_service=session_service)

    async with _client(app_a) as client_a:
        first = await _stream(client_a, "/api/studio/turn", _turn_body("first"))
    assert _events(first)[-1] == "done"

    async with _client(app_b) as client_b:
        second = await _stream(client_b, "/api/studio/turn", _turn_body("second"))
    assert _events(second)[-1] == "done"

    state = await StudioStore(session_service, app_name=AGENT_NAME).load(PRINCIPAL, CONVERSATION)
    assert state is not None
    assert state.turn_index == 2


# --- the segment lease wired into the routes --------------------------------


async def test_a_turn_from_another_instance_while_a_streams_is_409_with_retry_after(
    tmp_path: Path,
) -> None:
    """B's `/turn` arrives on a separate instance while A's segment is still
    live: the ledger's lease, not A's process-local lock, answers with the
    existing 409, carrying the segment's own remaining TTL as
    `retry_after_seconds`."""
    resolver = _HangingResolver()
    session_service = sqlite_session_service(tmp_path / "shared.db")
    app_a = _app(*_card(), resolver=resolver, session_service=session_service)
    app_b = _app(session_service=session_service)

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

    async def stalled_renew(*args: Any, **kwargs: Any) -> Any:
        await stalled.wait()
        return await renew_a(*args, **kwargs)

    monkeypatch.setattr(store_a, "renew", stalled_renew)
    store_b = StudioStore(session_service, app_name=AGENT_NAME, clock=lambda: clock[0])
    resolver = _HangingResolver()
    app_a = _app(*_card(), resolver=resolver, session_service=session_service, store=store_a)
    app_b = _app(
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
    app = _app(*_card(), resolver=resolver, session_service=session_service, store=store)

    async with _client(app) as client:
        stream_task = asyncio.ensure_future(_stream(client, "/api/studio/turn", _turn_body()))
        await asyncio.wait_for(resolver.entered.wait(), timeout=5)
        state = store.current(PRINCIPAL, CONVERSATION)
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
    app_a = _app(*_card(), resolver=resolver, session_service=session_service)
    app_b = _app(session_service=session_service)

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
    session_service = sqlite_session_service(tmp_path / "shared.db")
    resolver = _HangingResolver()
    app_a = _app(*_card(), resolver=resolver, session_service=session_service)

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

        store_b = StudioStore(session_service, app_name=AGENT_NAME)
        state_b = await store_b.load(PRINCIPAL, CONVERSATION)
        block = CompositionBlock(
            component_id="cmp_from_b", revision=1, type=ComponentType.LINE_CHART, title="B"
        )
        state_b.composition = studio_state_module._next_manifest(
            state_b.composition, [block], actor=CompositionActor.USER
        )
        await store_b.save(PRINCIPAL, CONVERSATION)

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

    reloaded = await StudioStore(session_service, app_name=AGENT_NAME).load(PRINCIPAL, CONVERSATION)
    assert reloaded.composition.blocks[0].pinned is True


async def test_a_release_failure_does_not_strand_the_local_lock() -> None:
    """`StudioRunner._locked`'s nested finally releases the conversation
    lock even when the store's own release call raises (`StudioStore.release`
    never does in practice; this pins the guarantee independently of that
    contract) -- the exception propagates, but the lock still releases."""
    studio = _app().state.studio
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
    studio = _app().state.studio
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
    app = _app(*_plan(), tool_surface=ScriptedSurface())
    studio = app.state.studio

    async def broken_checkpoint(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("db is down")

    studio.store.checkpoint = broken_checkpoint  # type: ignore[method-assign]

    async with _client(app) as client:
        frames = await _stream(client, "/api/studio/turn", _turn_body("q with forks"))

    assert _events(frames)[-1] == "error"
    assert "plan.proposed" not in _events(frames)
    error = _frame(frames, "error")["error"]
    assert error["error"] == "transient_unavailable"
    assert error["detail"] == classed_sentence(HarnessCode.STATE_NOT_SAVED, "retry_after_delay")
    assert is_user_sentence(error["detail"])


async def test_a_remote_composition_write_during_a_stream_reaches_its_done_frame_and_the_ledger(
    tmp_path: Path,
) -> None:
    """A remote instance's `/composition` write, landing mid-stream before
    A's second turn finishes, is merged in by the checkpoint that runs
    before `done`: A's own Done frame carries it, not just a later reload
    from the ledger, pinned on the composition manifest specifically."""
    session_service = sqlite_session_service(tmp_path / "shared.db")
    app_b = _app(session_service=session_service)

    async with _client(app_b) as client_b:
        executor = _PinsTheFirstCardFromB(client_b)
        app_a = _app(
            *_card(),
            *_card(),
            resolver=_search(2),
            executor=executor,
            session_service=session_service,
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

    reloaded = await StudioStore(session_service, app_name=AGENT_NAME).load(PRINCIPAL, CONVERSATION)
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
    app = _app(*_card(), resolver=_search())
    session_service = app.state.studio.store._session_service

    async with _client(app) as client:
        await _conversation_with_component(client)
        store = app.state.studio.store
        state = store.current(PRINCIPAL, CONVERSATION)

        turn_lock = store.lock_for(PRINCIPAL, CONVERSATION)
        await turn_lock.acquire()  # stands in for a stream still owning the turn

        real_get_session = session_service.get_session
        landed: list[bool] = []

        async def get_session_then_land(*args: Any, **kwargs: Any) -> Any:
            if not landed:
                landed.append(True)
                action = land_component(
                    state,
                    {"component_id": "cmp_landed", "type": "line_chart", "revision": 1},
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

    live = store.current(PRINCIPAL, CONVERSATION)
    assert {b.component_id for b in live.composition.blocks} == {"cmp_01test0001", "cmp_landed"}

    reloaded = await StudioStore(session_service, app_name=AGENT_NAME).load(PRINCIPAL, CONVERSATION)
    assert {b.component_id for b in reloaded.composition.blocks} == {"cmp_01test0001", "cmp_landed"}


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
    """The planner is seeded with the earlier turns' readings, warnings and
    the spec_ids it can pin from a stored block, without the block."""
    session_service = _new_session_service()
    store = StudioStore(session_service, app_name=AGENT_NAME)
    app = _app(session_service=session_service, store=store)
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

    session = await session_service.get_session(
        app_name=AGENT_NAME, user_id=PRINCIPAL, session_id=f"{CONVERSATION}#t2"
    )
    assert session is not None
    assert "stored block" not in json.dumps(session.state, default=str)
    assert session.state.get(ORCHESTRATOR_PRIOR_LEDGER_KEY) == {
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

    _keep_turn_records(conversation, Event(actions=EventActions(state_delta=delta)))

    assert conversation.turn_plans == [{"plan": 1}]
    assert conversation.turn_readings == [{"reading": 1}]
    assert conversation.turn_specs == {"spec_a": {"spec_id": "spec_a"}}
