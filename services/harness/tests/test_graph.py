"""Behavioural tests for the plan path of the root workflow: the `planner`,
`approve` and `execute` nodes the orchestrator's `plan` tool reaches, and
the helpers they share.

The workflow tests run the orchestrator topology (`orchestrator_runner`):
the agent hands a brief to a planner seam, the turn pauses on the plan card,
and the answer resumes it into `execute` and `plan_done`. The rest are unit
tests of the node helpers. Everything runs offline against scripted fakes:
no model, no MCP server.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Iterator, Mapping
from contextlib import suppress
from pathlib import Path
from typing import Any, cast

import pytest
from analysis_mcp.server import mcp as analysis_mcp
from compiler.surface import (
    FixtureSurface,
    McpSurface,
    ProgressCallback,
    ProgressReport,
    ToolCallResult,
    ToolSchema,
    ToolSurface,
)
from data_mcp.server import mcp as data_mcp
from executor.classify import ToolFailure
from executor.outcomes import (
    Accepted,
    MethodChoiceNeeded,
    RediscoveryNeeded,
    RefusalKind,
    Refused,
)
from executor.run import StepProgressCallback, execute_approved_plan
from google.adk import Context
from google.adk.events import Event, RequestInput
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.workflow._errors import NodeInterruptedError
from google.adk.workflow.utils._workflow_hitl_utils import create_request_input_event
from google.genai import types
from harness.chart_binding import LINE_CHART_MAX_VALUES, select_binding_ok
from harness.graph import answer as answer_module
from harness.graph.execute import (
    _failed_step,
    _mark_masked_step_failed,
    _replan_refusal,
    _start_relayed,
)
from harness.graph.plan_done import _frame_unavailable_reason
from harness.graph.state import (
    APPROVE_INTERRUPT,
    PLANNER_STARTED_KEY,
    STEP_PROGRESS_KEY,
    UNREADABLE_PLAN_NOTE,
    UNREADABLE_PLAN_UNPLANNED_NOTE,
    GraphState,
    frame_id_for,
)
from harness.scripted import ScriptedPlannerModel, ScriptedSurface
from harness.scripted_plan import scripted_fetch_series_stats
from harness.seeds import stale_reason
from harness.studio_translate import _node_name
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from planner.operations import AggregateOperation, GrowthOperation, RetrieveOperation, Translation
from planner.plan import (
    AggregateRequirement,
    DatasetRequirement,
    Decision,
    Entity,
    GrowthRequirement,
    Period,
    PlanStatus,
    ResolvedBy,
    plan_revision,
)
from planner.plan import Plan as PlannerPlan
from planner.records import EvidenceIssue, EvidenceIssueCode, GroupBy, Reduction, Severity
from planner.workflow import EvidenceBinding, SelectedSource, ToolId, Workflow, WorkflowStep
from pydantic import ValidationError
from translator.discovery import PartFault, PartNotResolvedError

from .orchestrator_fakes import (
    ScriptLlm,
    create_request_input_response,
    handed_back,
    open_session,
    orchestrator_runner,
    plan_call,
    request_text,
    run_turn,
    visited_nodes,
)

SPEC_A = "spec_" + "a" * 32
SPEC_B = "spec_" + "b" * 32
REF_A = {"spec_id": SPEC_A, "content_version": "cv_" + "1" * 32}
REF_B = {"spec_id": SPEC_B, "content_version": "cv_" + "2" * 32}


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


class _CallRecordingSurface:
    """Wraps a `ToolSurface`, recording every call -- `FixtureSurface` itself does not."""

    def __init__(self, surface: ToolSurface) -> None:
        self._surface = surface
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def list_tools(self) -> list[ToolSchema]:
        return await self._surface.list_tools()

    async def call_tool(
        self,
        *,
        tool: ToolId,
        arguments: Mapping[str, Any],
        on_progress: ProgressCallback | None = None,
    ) -> ToolCallResult:
        self.calls.append((tool.name, dict(arguments)))
        return await self._surface.call_tool(
            tool=tool, arguments=arguments, on_progress=on_progress
        )


def _fixture_surface(
    schemas: list[ToolSchema],
    *,
    analysis_tool: str = "aggregate",
    analysis_result: dict[str, Any] | None = None,
    get_data_result: ToolCallResult | None = None,
    analysis_call_result: ToolCallResult | None = None,
) -> FixtureSurface:
    """A `ToolSurface` over the real schemas, answering `get_data` and one
    analysis tool. A caller wanting a refusal from either passes its own
    `ToolCallResult`."""
    named = {schema.name: schema for schema in schemas}
    return FixtureSurface(
        schemas=[named["get_data"], named[analysis_tool]],
        responses={
            "get_data": get_data_result
            or ToolCallResult(
                is_error=False, structured_content=_get_data_manifest(), text="fetched"
            ),
            analysis_tool: analysis_call_result
            or ToolCallResult(
                is_error=False,
                structured_content=analysis_result or _analysis_result(),
                text="computed",
            ),
        },
    )


class ScriptedExecutor:
    """The executor `plan_done` describes the plan's final frames through:
    the scripted results in order, then a described frame with an empty view."""

    def __init__(self, results: list[dict[str, Any]] | None = None) -> None:
        self.results = results or []
        self.calls: list[dict[str, Any]] = []

    async def __call__(
        self, *, tool: str, args: dict[str, Any], on_progress: ProgressCallback | None = None
    ) -> dict[str, Any]:
        self.calls.append({"tool": tool, "args": args})
        if len(self.calls) <= len(self.results):
            return self.results[len(self.calls) - 1]
        return {"outcome": "ok", "view": {}}


_BRIEF = "kenya debt"


def _plan_runner(
    planner: Any, *, tool_surface: Any = None, executor: Any = None
) -> tuple[Runner, ScriptLlm]:
    """The orchestrator topology whose agent hands `_BRIEF` to `planner` and
    then answers in plain text; `tool_surface` runs the approved plan."""
    return orchestrator_runner(
        plan_call(_BRIEF),
        types.Part.from_text(text="Handed to the planner."),
        planner=planner,
        tool_surface=tool_surface,
        executor=executor or ScriptedExecutor(),
    )


async def _start(runner: Runner) -> list[Event]:
    """Open `runner`'s session and run the turn that asks for a plan."""
    await open_session(runner)
    return await run_turn(runner, types.UserContent(_BRIEF))


async def _resume(runner: Runner, interrupt_id: str, response: dict[str, Any]) -> list[Event]:
    """Answer the pause `interrupt_id` with `response`."""
    part = create_request_input_response(interrupt_id, {"result": response})
    return await run_turn(runner, types.Content(role="user", parts=[part]))


def _session_state(runner: Runner) -> dict[str, Any]:
    """The live state of the session `_start` opened."""
    service = cast(InMemorySessionService, runner.session_service)
    return service.sessions["t"]["u"]["s"].state


def _pending_interrupt(events: list[Event]) -> str | None:
    """The pause the turn ended on, if it ended on one."""
    ids = [c.id for e in events for c in e.get_function_calls() if c.name == "adk_request_input"]
    return ids[-1] if ids else None


def _output_of(events: list[Event], node: str) -> Any:
    outputs = [e.output for e in events if _node_name(e) == node and e.output is not None]
    return outputs[-1] if outputs else None


# --- execute's run: progress relay and cancellation ---------------------------


_PREPARING = "Preparing the dataset (in progress, 3s)"
_RECEIVED = "Result received; checking it"


def _step_progress(event: Event) -> dict[str, Any] | None:
    return (event.custom_metadata or {}).get(STEP_PROGRESS_KEY)


class _ReportsThenWaits(ScriptedSurface):
    """`ScriptedSurface`, whose `get_data` sends one progress report and then
    waits for `release` before answering, standing in for data-mcp polling
    a background dfo-apis operation."""

    def __init__(self) -> None:
        super().__init__()
        self.release = asyncio.Event()
        self.waiting: asyncio.Task[Any] | None = None
        self.cancelled = False

    async def call_tool(
        self,
        *,
        tool: ToolId,
        arguments: Mapping[str, Any],
        on_progress: ProgressCallback | None = None,
    ) -> ToolCallResult:
        if tool.name == "get_data":
            assert on_progress is not None
            on_progress(ProgressReport(progress=3, message=_PREPARING))
            self.waiting = asyncio.current_task()
            try:
                await self.release.wait()
            except asyncio.CancelledError:
                self.cancelled = True
                raise
        return await super().call_tool(tool=tool, arguments=arguments, on_progress=on_progress)


async def _approval_stream(surface: ToolSurface) -> Any:
    """The event stream of an approved scripted plan's run against `surface`."""
    runner, _ = _plan_runner(ScriptedPlannerModel(), tool_surface=surface)
    events = await _start(runner)
    part = create_request_input_response(
        APPROVE_INTERRUPT, {"result": {"approved": _output_of(events, "planner")["revision"]}}
    )
    return runner.run_async(
        user_id="u", session_id="s", new_message=types.Content(role="user", parts=[part])
    )


async def test_execute_relays_a_report_while_its_step_is_still_running() -> None:
    """The surface answers only once the test has seen the report, so the
    report must reach the event stream while the call is still waiting. Once
    a call answers, its step reports that, so the step's line stops showing
    the server's wait while the run goes on."""
    surface = _ReportsThenWaits()
    stream = await _approval_stream(surface)
    seen: list[Event] = []

    async with asyncio.timeout(10):
        async for event in stream:
            seen.append(event)
            if _step_progress(event) is not None and not surface.release.is_set():
                assert surface.waiting is not None and not surface.waiting.done()
                surface.release.set()

    progress = [_step_progress(event) for event in seen if _step_progress(event)]
    assert progress == [
        {"step_id": "retrieve", "tool": "get_data", "message": _PREPARING},
        {"step_id": "retrieve", "tool": "get_data", "message": _RECEIVED},
        {"step_id": "totals", "tool": "aggregate", "message": _RECEIVED},
    ]
    assert _node_name(next(e for e in seen if _step_progress(e))) == "execute"
    assert "plan_done" in visited_nodes(seen)


async def test_stopping_a_turn_mid_step_cancels_the_run_and_leaves_no_task_behind() -> None:
    """Studio stops a turn by cancelling the pending `anext` on its event
    stream (`harness.studio._unwind`). The plan's run is a task of its own,
    so the `execute` node must cancel it rather than leave it running."""
    surface = _ReportsThenWaits()
    stream = await _approval_stream(surface)

    async with asyncio.timeout(10):
        while _step_progress(await anext(stream)) is None:
            pass
        pending = asyncio.ensure_future(anext(stream))
        await asyncio.sleep(0)
        pending.cancel()
        with suppress(asyncio.CancelledError):
            await pending
        await stream.aclose()

    assert surface.waiting is not None and surface.waiting.done()
    assert surface.cancelled


async def test_relay_collapses_many_reports_from_one_step_while_blocked() -> None:
    """`_start_relayed` coalesces at enqueue time, not after: a step that
    reports many times before the reader gets a chance to drain leaves only
    its latest report pending, so a fast producer never grows the relay
    past one entry per distinct step."""

    async def call(on_progress: StepProgressCallback) -> str:
        for i in range(50):
            on_progress("only-step", ProgressReport(progress=i, message=f"tick {i}"))
        return "done"

    task, relay = _start_relayed(call)
    result = await task  # every report above queued before anything drained the relay

    reports = [report async for report in relay.latest()]

    assert len(reports) == 1
    assert reports[0].step_id == "only-step"
    assert reports[0].report == ProgressReport(progress=49, message="tick 49")
    assert result == "done"
    assert {task for task in asyncio.all_tasks() if not task.done()} == {asyncio.current_task()}


# --- the session state schema ---------------------------------------------------


def test_a_session_saved_with_a_key_the_schema_no_longer_declares_still_validates() -> None:
    """A session saved at an interrupt can hold a key the state schema does not
    declare, such as `spec_id_stat_vars`. The schema validates it, keeps the
    fields it declares, and leaves the key unread."""
    saved = {
        "surface": "download",
        "spec_id_stat_vars": json.dumps({"spec_id": SPEC_A, "stat_vars": ["Count_Person"]}),
    }

    state = GraphState.model_validate(saved)

    assert state.surface == "download"
    assert "spec_id_stat_vars" not in state.model_dump()


def test_every_graph_state_field_has_a_default() -> None:
    """A session saved at an interrupt resumes under the current schema, so a
    field it does not hold has to validate from its default."""
    required = [name for name, field in GraphState.model_fields.items() if field.is_required()]

    assert required == []
    assert GraphState().model_dump()["seeds"] is None


# --- version-bound approval -----------------------------------------------------


async def test_stale_version_approval_routes_to_amend() -> None:
    """A wrong answer routes to amend; `ScriptedPlannerModel` redrafts the
    same canned steps as the next version, bumping `version` and
    `previous_version` the way any `PlannerModel` seam must."""
    surface = _CallRecordingSurface(ScriptedSurface())
    runner, _ = _plan_runner(ScriptedPlannerModel(), tool_surface=surface)
    events = await _start(runner)
    assert _pending_interrupt(events) == APPROVE_INTERRUPT
    first = _output_of(events, "planner")

    events = await _resume(runner, APPROVE_INTERRUPT, {"approved": "not-a-real-revision"})

    # Wrong answer: nothing executed; the planner redrafted to version 2 and
    # a fresh approval interrupt is pending.
    assert surface.calls == []
    assert _pending_interrupt(events) == APPROVE_INTERRUPT
    amended = _output_of(events, "planner")
    assert amended["version"] == 2
    assert amended["previous_version"] == 1
    assert amended["status"] == "pending_approval"
    assert amended["revision"] != first["revision"]

    events = await _resume(runner, APPROVE_INTERRUPT, {"approved": amended["revision"]})
    assert [call[0] for call in surface.calls] == ["get_data", "aggregate"]
    assert _output_of(events, "execute")["plan_version"] == 2


async def test_a_plan_approved_unchanged_opens_a_span_and_logs_the_facts(
    exported_spans: InMemorySpanExporter, caplog: pytest.LogCaptureFixture
) -> None:
    """Version 1 approved unchanged: `approve`'s `plan.approved` span and log
    line record the version and that no amendment preceded it."""
    surface = _CallRecordingSurface(ScriptedSurface())
    runner, _ = _plan_runner(ScriptedPlannerModel(), tool_surface=surface)
    events = await _start(runner)

    assert _pending_interrupt(events) == APPROVE_INTERRUPT
    plan_payload = _output_of(events, "planner")
    assert plan_payload["version"] == 1

    with caplog.at_level("INFO", logger="harness.graph.planning"):
        events = await _resume(runner, APPROVE_INTERRUPT, {"approved": plan_payload["revision"]})

    assert _output_of(events, "execute")["plan_version"] == 1

    spans = [s for s in exported_spans.get_finished_spans() if s.name == "plan.approved"]
    assert len(spans) == 1
    attributes = spans[0].attributes
    assert attributes is not None
    assert attributes["dfo.plan_version"] == 1
    assert attributes["dfo.amended"] is False

    record = next(r for r in caplog.records if "plan approved" in r.getMessage())
    assert "plan_version=1" in record.getMessage()
    assert "amended=False" in record.getMessage()


# --- final frames: what `plan_done` describes ------------------------------------


_FRAME_DO_NOT_RETRY = "this step's data view could not be prepared"


@pytest.mark.parametrize(
    ("retry_mode", "reason"),
    [
        pytest.param(["retry_after_delay"], _FRAME_DO_NOT_RETRY, id="list"),
        pytest.param({"mode": 1}, _FRAME_DO_NOT_RETRY, id="dict"),
        pytest.param(None, _FRAME_DO_NOT_RETRY, id="none"),
        pytest.param(
            "retry_after_delay", "this step's data view is temporarily unavailable", id="string"
        ),
    ],
)
def test_a_frame_unavailable_reason_reads_a_retry_mode_of_any_type(
    retry_mode: Any, reason: str
) -> None:
    assert _frame_unavailable_reason({"retry_mode": retry_mode}) == reason


def _seam_plan_with_intermediate_step(*, schemas: list[ToolSchema]) -> PlannerPlan:
    """A real, three-step plan: get_data -> totals (aggregate, an
    intermediate step) -> change (growth, the one terminal step).

    `_seam_plan`'s two-step plan has one non-terminal step, so a failure
    after it leaves one frame. A third step lets two steps, one of them
    genuinely intermediate, complete before the terminal one fails.
    """
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
    totals = AggregateRequirement(
        id="totals",
        decision_ids=["window"],
        input_id="spending",
        reduction=Reduction.SUM,
        group_by=(GroupBy.PERIOD,),
    )
    change = GrowthRequirement(
        id="change",
        decision_ids=["window"],
        input_id="totals",
        window_start=2015,
        window_end=2024,
    )
    draft = PlannerPlan(
        question="kenya debt growth",
        decisions=[decision],
        evidence_requirements=[dataset, totals, change],
        version=1,
    )
    translation = Translation(
        operations=[
            RetrieveOperation(id="get_data", requirement_id=dataset.id, spec_id=SPEC_A),
            AggregateOperation(
                id="totals",
                requirement_id=totals.id,
                input="get_data",
                reduction=Reduction.SUM,
                group_by=(GroupBy.PERIOD,),
            ),
            GrowthOperation(
                id="change",
                requirement_id=change.id,
                input="totals",
                window_start=2015,
                window_end=2024,
            ),
        ],
        terminal=["change"],
    )
    named = {schema.name: schema for schema in schemas}
    get_data_schema = named["get_data"]
    aggregate_schema = named["aggregate"]
    growth_schema = named["growth"]
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
                id="totals",
                operation_id="totals",
                tool=ToolId(server="analysis-mcp", name="aggregate"),
                args={"aggregation": "sum", "group_by": ["time_period"]},
                bindings={"frame_ref": {"from_step": "get_data", "output": "frame_ref"}},
                input_schema_hash=aggregate_schema.input_schema_hash,
                output_schema_hash=aggregate_schema.output_schema_hash,
                implementation_version=aggregate_schema.version,
            ),
            WorkflowStep(
                id="change",
                operation_id="change",
                tool=ToolId(server="analysis-mcp", name="growth"),
                args={"window_start": "2015", "window_end": "2024", "entity_by": []},
                bindings={"frame_ref": {"from_step": "totals", "output": "frame_ref"}},
                input_schema_hash=growth_schema.input_schema_hash,
                output_schema_hash=growth_schema.output_schema_hash,
                implementation_version=growth_schema.version,
            ),
        ],
        sources=[
            SelectedSource(spec_id=SPEC_A, sentence="health spending in Kenya", has_data=True)
        ],
        evidence_bindings=[],
        terminal_steps=["change"],
    )
    return draft.with_translation(translation=translation, workflow=workflow)


_CHANGE_REFUSAL = ToolCallResult(
    is_error=True,
    structured_content={
        "code": "entity_ambiguous",
        "retry_mode": "correct_and_retry",
        "message": "more than one entity matches 'Kenya'",
        "fix": {"candidates": ["country/KEN", "country/KEN-old"]},
    },
    text="refused",
)


def _intermediate_step_surface(real_schemas: list[ToolSchema]) -> FixtureSurface:
    """get_data -> totals succeed; change (growth), the plan's one terminal
    step, always refuses. A run against this surface completes two steps,
    one of them intermediate, before exhausting the replan budget."""
    named = {schema.name: schema for schema in real_schemas}
    return FixtureSurface(
        schemas=[named["get_data"], named["aggregate"], named["growth"]],
        responses={
            "get_data": ToolCallResult(
                is_error=False, structured_content=_get_data_manifest(), text="fetched"
            ),
            "aggregate": ToolCallResult(
                is_error=False, structured_content=_analysis_result(), text="computed"
            ),
            "growth": _CHANGE_REFUSAL,
        },
    )


async def test_a_stopped_plan_hands_back_no_intermediate_steps_frame(
    real_schemas: list[ToolSchema],
) -> None:
    """growth, the plan's one terminal step, fails past the replan budget,
    leaving get_data's and totals' frames in `completed`. Neither is a final
    step, so `plan_done` describes neither and hands the agent no frame to
    chart: an intermediate frame never stands in for the plan's answer."""
    plan = _seam_plan_with_intermediate_step(schemas=real_schemas)
    seam = _Seam([{"plan": plan.model_dump(mode="json"), "summary": {}}])
    executor = ScriptedExecutor()
    runner, model = _plan_runner(
        seam, tool_surface=_intermediate_step_surface(real_schemas), executor=executor
    )
    await _start(runner)

    events = await _resume(runner, APPROVE_INTERRUPT, {"approved": plan_revision(plan)})
    assert _pending_interrupt(events) == APPROVE_INTERRUPT
    replanned = _output_of(events, "planner")
    events = await _resume(runner, APPROVE_INTERRUPT, {"approved": replanned["revision"]})

    executed = _output_of(events, "execute")
    assert executed["failed_step"]["step_id"] == "change"
    assert len(executed["frame_refs"]) == 2
    assert executor.calls == []
    _, body = handed_back(model)
    assert body["frames"] == []


def _seam_plan_with_two_terminal_steps(*, schemas: list[ToolSchema]) -> PlannerPlan:
    """A real plan with two independent terminal steps reading the same
    retrieval directly: `totals` (aggregate) and `change` (growth), at
    different positions in `outcome.steps` (`workflow.steps` is
    `[get_data, totals, change]`, so `totals` sits at index 1 and `change`
    at index 2). `Translation.terminal` is a tuple with `min_length=1`, not
    `max_length=1` (`planner.operations.py`), so a plan answering through
    more than one terminal step is a real shape, not a hypothetical one --
    and `is_final_step` must mark both `totals` and `change` final, since
    neither is any more the question's answer than the other."""
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
    totals = AggregateRequirement(
        id="totals",
        decision_ids=["window"],
        input_id="spending",
        reduction=Reduction.SUM,
        group_by=(GroupBy.PERIOD,),
    )
    change = GrowthRequirement(
        id="change",
        decision_ids=["window"],
        input_id="spending",
        window_start=2015,
        window_end=2024,
    )
    draft = PlannerPlan(
        question="kenya debt totals and growth",
        decisions=[decision],
        evidence_requirements=[dataset, totals, change],
        version=1,
    )
    translation = Translation(
        operations=[
            RetrieveOperation(id="get_data", requirement_id=dataset.id, spec_id=SPEC_A),
            AggregateOperation(
                id="totals",
                requirement_id=totals.id,
                input="get_data",
                reduction=Reduction.SUM,
                group_by=(GroupBy.PERIOD,),
            ),
            GrowthOperation(
                id="change",
                requirement_id=change.id,
                input="get_data",
                window_start=2015,
                window_end=2024,
            ),
        ],
        terminal=["totals", "change"],
    )
    named = {schema.name: schema for schema in schemas}
    get_data_schema = named["get_data"]
    aggregate_schema = named["aggregate"]
    growth_schema = named["growth"]
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
                id="totals",
                operation_id="totals",
                tool=ToolId(server="analysis-mcp", name="aggregate"),
                args={"aggregation": "sum", "group_by": ["time_period"]},
                bindings={"frame_ref": {"from_step": "get_data", "output": "frame_ref"}},
                input_schema_hash=aggregate_schema.input_schema_hash,
                output_schema_hash=aggregate_schema.output_schema_hash,
                implementation_version=aggregate_schema.version,
            ),
            WorkflowStep(
                id="change",
                operation_id="change",
                tool=ToolId(server="analysis-mcp", name="growth"),
                args={"window_start": "2015", "window_end": "2024", "entity_by": []},
                bindings={"frame_ref": {"from_step": "get_data", "output": "frame_ref"}},
                input_schema_hash=growth_schema.input_schema_hash,
                output_schema_hash=growth_schema.output_schema_hash,
                implementation_version=growth_schema.version,
            ),
        ],
        sources=[
            SelectedSource(spec_id=SPEC_A, sentence="health spending in Kenya", has_data=True)
        ],
        evidence_bindings=[],
        terminal_steps=["totals", "change"],
    )
    return draft.with_translation(translation=translation, workflow=workflow)


async def test_two_terminal_steps_both_come_back_to_the_agent(
    real_schemas: list[ToolSchema],
) -> None:
    """`totals` (step index 1) and `change` (step index 2) are both
    terminal. Finality inferred from `max(step)` would mark only `change`
    final and drop `totals`; `execute`'s Accepted branch states
    `is_final_step` on each entry, so `plan_done` describes and lists both."""
    plan = _seam_plan_with_two_terminal_steps(schemas=real_schemas)
    seam = _Seam([{"plan": plan.model_dump(mode="json"), "summary": {}}])
    named = {schema.name: schema for schema in real_schemas}
    surface = FixtureSurface(
        schemas=[named["get_data"], named["aggregate"], named["growth"]],
        responses={
            "get_data": ToolCallResult(
                is_error=False, structured_content=_get_data_manifest(), text="fetched"
            ),
            "aggregate": ToolCallResult(
                is_error=False, structured_content=_analysis_result(), text="computed"
            ),
            "growth": ToolCallResult(
                is_error=False,
                # A distinct spec_id: totals and change must be different frames.
                structured_content=_analysis_result(spec_id="ops_" + "4" * 32),
                text="computed",
            ),
        },
    )
    executor = ScriptedExecutor()
    runner, model = _plan_runner(seam, tool_surface=surface, executor=executor)
    await _start(runner)

    events = await _resume(runner, APPROVE_INTERRUPT, {"approved": plan_revision(plan)})

    assert len(_output_of(events, "execute")["frame_refs"]) == 2
    assert [c["tool"] for c in executor.calls] == ["describe_frame", "describe_frame"]
    _, body = handed_back(model)
    assert len(body["frames"]) == 2


# --- select_binding_ok: the select checks `make_charts` runs before a write ---


_MULTISERIES_VIEW: dict[str, Any] = {
    "dimensions": [
        {"name": "stat_var", "plural": False, "values": ["dcid:Count_Person"]},
        {
            "name": "observationAbout",
            "plural": True,
            "values": [
                f"country/{code}" for code in ("BLZ", "CRI", "GTM", "HND", "NIC", "PAN", "SLV")
            ],
        },
    ],
    "series_count": 7,
}


def _population_view(*, plural: str) -> dict[str, Any]:
    """The ONEcampaign/dfo-apis#953 population frame's view: one `stat_var`, seven countries, and
    a `unit` column holding `USDollar` on some rows and null on others, which
    dfo-apis reports as plural with the one non-null value listed. `plural`
    names a further dimension marked plural with two listed values."""
    return {
        "dimensions": [
            {"name": "stat_var", "plural": False, "values": ["dcid:Count_Person"]},
            {
                "name": "observationAbout",
                "plural": True,
                "values": ["country/GTM", "country/HND"],
                "total_count": 7,
            },
            {"name": "unit", "plural": True, "values": ["USDollar"], "total_count": 1},
            {"name": plural, "plural": True, "values": ["a", "b"], "total_count": 2},
        ]
    }


@pytest.mark.parametrize("facet", ["measurementMethod", "provenance"])
def test_stat_card_may_leave_plural_facets_unpinned(facet: str) -> None:
    """dfo-apis refuses an unbound facet only when it splits a series, so a
    population card pinning the country passes with `unit` = `{USDollar,
    null}` and a second plural facet column left unbound."""
    select = {"stat_var": "dcid:Count_Person", "observationAbout": "country/GTM"}
    view = _population_view(plural=facet)
    assert select_binding_ok(type_="stat_card", select=select, view=view) is None


def test_stat_card_still_refuses_an_unpinned_non_facet_plural_dimension() -> None:
    select = {"stat_var": "dcid:Count_Person", "observationAbout": "country/GTM"}
    reason = select_binding_ok(
        type_="stat_card", select=select, view=_population_view(plural="sex")
    )
    assert reason is not None
    assert reason["code"] == "select_invalid"


# --- the select_binding_ok array-dimension cap, per component type -----------


def test_select_binding_ok_permits_a_table_with_two_array_dimensions() -> None:
    """The platform's catalogue grammar allows `table` to carry arrays on
    more than one dimension (their cross-product, at most 24 series); only
    `line_chart` and `bar_chart` are limited to at most one array-valued
    dimension."""
    select = {"observationAbout": ["country/KEN", "country/UGA"], "stat_var": ["a", "b"]}
    assert select_binding_ok(type_="table", select=select, view={}) is None


def test_select_binding_ok_still_rejects_a_line_chart_with_two_array_dimensions() -> None:
    select = {"observationAbout": ["country/KEN", "country/UGA"], "stat_var": ["a", "b"]}
    assert select_binding_ok(type_="line_chart", select=select, view={}) is not None


def _countries(count: int) -> list[str]:
    return [f"country/C{index:02d}" for index in range(count)]


# The platform's release document as presentation-mcp's tests record it.
_RELEASE = Path(__file__).parents[2] / "presentation-mcp/tests/fixtures/release.json"


def test_a_line_chart_lists_no_more_values_than_the_catalogue_allows() -> None:
    select_schema = json.loads(_RELEASE.read_text())["intent_schemas_tolerant"]["line_chart"][
        "properties"
    ]["select"]
    (array,) = [
        branch
        for branch in select_schema["additionalProperties"]["anyOf"]
        if branch["type"] == "array"
    ]
    cap = array["maxItems"]

    allowed = select_binding_ok(
        type_="line_chart", select={"observationAbout": _countries(cap)}, view={}
    )
    refused = select_binding_ok(
        type_="line_chart", select={"observationAbout": _countries(cap + 1)}, view={}
    )

    assert cap == LINE_CHART_MAX_VALUES
    assert allowed is None
    assert refused is not None
    assert refused["code"] == "series_limit_exceeded"


@pytest.mark.parametrize("type_", ["bar_chart", "table"])
def test_select_binding_ok_permits_24_series(type_: str) -> None:
    select = {"observationAbout": _countries(24), "stat_var": "a"}
    assert select_binding_ok(type_=type_, select=select, view={}) is None


@pytest.mark.parametrize("type_", ["bar_chart", "table"])
def test_select_binding_ok_rejects_a_25th_series(type_: str) -> None:
    """dfo-apis refuses a component past 24 series even though the catalogue
    grammar's `maxItems` allows an array of 30."""
    select = {"observationAbout": _countries(25), "stat_var": "a"}
    reason = select_binding_ok(type_=type_, select=select, view={})
    assert reason is not None
    assert reason["message"] == (
        f"{type_} select expands to an estimated 25 series, over the platform's maximum of 24"
    )


def test_select_binding_ok_counts_a_tables_cross_product_against_the_cap() -> None:
    select = {"observationAbout": _countries(5), "stat_var": ["a", "b", "c", "d", "e"]}
    reason = select_binding_ok(type_="table", select=select, view={})
    assert reason is not None
    assert reason["message"] == (
        "table select expands to an estimated 25 series, over the platform's maximum of 24"
    )


def _counted_view(**total_counts: int) -> dict[str, Any]:
    """A `describe_frame` view whose dimensions carry dfo-apis's `total_count`."""
    return {
        "dimensions": [
            {"name": name, "plural": count > 1, "values": [], "total_count": count}
            for name, count in total_counts.items()
        ]
    }


@pytest.mark.parametrize(
    ("type_", "select", "view", "estimate"),
    [
        ("bar_chart", {"stat_var": "a"}, _counted_view(stat_var=1, donor=30), 30),
        ("table", {"stat_var": "a"}, _counted_view(stat_var=1, donor=30), 30),
        (
            "line_chart",
            {"donor": [f"donor/{index}" for index in range(5)]},
            _counted_view(stat_var=5, donor=30),
            25,
        ),
        (
            "table",
            {"donor": [f"donor/{index}" for index in range(20)]},
            _counted_view(stat_var=2, donor=30),
            40,
        ),
        ("table", {"stat_var": "a"}, _counted_view(stat_var=1, donor=30, unit=30), 30),
        (
            "table",
            {"stat_var": "a", "unit": ["USDollar", "EUR"]},
            _counted_view(stat_var=1, donor=30, unit=3, measurementMethod=5),
            60,
        ),
    ],
)
def test_select_binding_ok_matches_dfo_apis_series_estimate(
    type_: str, select: dict[str, Any], view: dict[str, Any], estimate: int
) -> None:
    """The harness computes dfo-apis's own estimate (`_check_series_estimate`
    in dfo-data-tools' `assembler/resolve.py`): over every coordinate
    dimension, the bound array's length, 1 for a single bound value, and
    `max(total_count, 1)` for an unbound dimension, on every component type
    the assembler resolves. Each case here is one dfo-apis refuses with
    `MAX_SERIES_PER_COMPONENT`: an unbound plural dimension, a `line_chart`
    array past 24, and an unbound dimension multiplying a table's array.
    An unbound facet column counts 1 and a bound one counts its values."""
    reason = select_binding_ok(type_=type_, select=select, view=view)
    assert reason is not None
    assert reason["message"] == (
        f"{type_} select expands to an estimated {estimate} series, "
        "over the platform's maximum of 24"
    )


def test_select_binding_ok_counts_an_unbound_dimension_at_its_total_count() -> None:
    """An unbound dimension holding 24 values is within the cap, and binding
    it to a single value brings 30 down to 1."""
    assert (
        select_binding_ok(
            type_="bar_chart", select={"stat_var": "a"}, view=_counted_view(stat_var=1, donor=24)
        )
        is None
    )
    assert (
        select_binding_ok(
            type_="bar_chart",
            select={"stat_var": "a", "donor": "donor/1"},
            view=_counted_view(stat_var=1, donor=30),
        )
        is None
    )
    assert (
        select_binding_ok(
            type_="bar_chart",
            select={"stat_var": "a"},
            view=_counted_view(stat_var=1, donor=24, unit=30, provenance=30),
        )
        is None
    )


def test_select_binding_ok_counts_a_dimension_with_no_total_count_as_one() -> None:
    """Scripted fixtures list bare dimension names, or entries without
    `total_count`; neither refuses for want of the figure."""
    bare = {"dimensions": ["stat_var", "observationAbout"]}
    assert select_binding_ok(type_="bar_chart", select={"stat_var": "a"}, view=bare) is None
    assert (
        select_binding_ok(type_="table", select={"stat_var": "a"}, view=_MULTISERIES_VIEW) is None
    )


def test_select_binding_ok_refuses_a_dimension_the_view_does_not_declare() -> None:
    """dfo-apis estimates series over its own dimensions only, and refuses a
    select binding any other name as an invalid binding; the harness gives
    the same reason."""
    select = {"stat_var": "a", "donor": [f"donor/{index}" for index in range(30)]}
    reason = select_binding_ok(type_="bar_chart", select=select, view=_counted_view(stat_var=1))
    assert reason is not None
    assert reason["code"] == "select_invalid"
    assert reason["message"] == "bar_chart select binds ['donor'], which the frame does not declare"
    bare = {"dimensions": ["stat_var", "observationAbout"]}
    reason = select_binding_ok(type_="table", select={"sector": "x"}, view=bare)
    assert reason is not None
    assert reason["message"] == "table select binds ['sector'], which the frame does not declare"


def test_select_binding_ok_rejects_a_non_mapping_select() -> None:
    """A `select_json` that parsed to a JSON array or scalar is a rejection
    reason here, not an `AttributeError` on `.items()`: `make_charts` hands
    it whatever `json.loads` made of the model's `select_json`."""
    assert select_binding_ok(type_="table", select=["not", "a", "mapping"], view={}) is not None


# --- the Plan objects stay real ------------------------------------------------


async def test_plan_amendment_lineage_is_real() -> None:
    """The plan carried through amend cycles is a valid `planner.plan.Plan`."""
    runner, _ = _plan_runner(ScriptedPlannerModel())
    await _start(runner)
    events = await _resume(runner, APPROVE_INTERRUPT, {"approved": 99})

    output = _output_of(events, "planner")
    plan_fields = {k: v for k, v in output.items() if k not in ("summary", "revision")}
    plan = PlannerPlan.model_validate(plan_fields)
    assert plan.version == 2
    assert plan.previous_version == 1
    assert plan.decisions[0].chosen in plan.decisions[0].options


# --- the planner node against a model seam --------------------------------


def _seam_plan(
    *,
    question: str = "kenya debt",
    version: int = 1,
    schemas: list[ToolSchema],
    bind_evidence: bool = False,
    stat_vars: str | None = None,
    terminal: tuple[str, ...] = ("totals",),
) -> PlannerPlan:
    """A real, translated `planner.plan.Plan` -- the shape a `PlannerModel` seam hands back.

    `stat_vars` is the reading its `get_data` step sends, for a plan whose
    source is a family of StatVars; `None` sends none. `terminal` names the
    steps that answer: `totals` by default, and `get_data` with it for a plan
    that shows the series as well as its total.
    """
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
    totals = AggregateRequirement(
        id="totals",
        decision_ids=["window"],
        input_id="spending",
        reduction=Reduction.SUM,
        group_by=(GroupBy.PERIOD,),
    )
    draft = PlannerPlan(
        question=question,
        decisions=[decision],
        evidence_requirements=[dataset, totals],
        version=version,
    )
    translation = Translation(
        operations=[
            RetrieveOperation(id="get_data", requirement_id=dataset.id, spec_id=SPEC_A),
            AggregateOperation(
                id="totals",
                requirement_id=totals.id,
                input="get_data",
                reduction=Reduction.SUM,
                group_by=(GroupBy.PERIOD,),
            ),
        ],
        terminal=list(terminal),
    )
    named = {schema.name: schema for schema in schemas}
    get_data_schema = named["get_data"]
    aggregate_schema = named["aggregate"]
    workflow = Workflow(
        steps=[
            WorkflowStep(
                id="get_data",
                operation_id="get_data",
                tool=ToolId(server="data-mcp", name="get_data"),
                args={"spec_id": SPEC_A} | ({"stat_vars": stat_vars} if stat_vars else {}),
                input_schema_hash=get_data_schema.input_schema_hash,
                output_schema_hash=get_data_schema.output_schema_hash,
                implementation_version=get_data_schema.version,
            ),
            WorkflowStep(
                id="totals",
                operation_id="totals",
                tool=ToolId(server="analysis-mcp", name="aggregate"),
                args={"aggregation": "sum", "group_by": ["time_period"]},
                bindings={"frame_ref": {"from_step": "get_data", "output": "frame_ref"}},
                input_schema_hash=aggregate_schema.input_schema_hash,
                output_schema_hash=aggregate_schema.output_schema_hash,
                implementation_version=aggregate_schema.version,
            ),
        ],
        sources=[
            SelectedSource(spec_id=SPEC_A, sentence="health spending in Kenya", has_data=True)
        ],
        # No evidence bindings by default: most of these tests are about
        # approval/execution wiring, not the evidence check, and carry no
        # real source discovery to check a manifest against. A caller
        # testing a refusal on the bound step (which never reaches the
        # check, since it only runs on success) opts in.
        evidence_bindings=(
            [EvidenceBinding(requirement_id=dataset.id, step_id="get_data", spec_id=SPEC_A)]
            if bind_evidence
            else []
        ),
        terminal_steps=list(terminal),
    )
    return draft.with_translation(translation=translation, workflow=workflow)


class _Seam:
    """A `PlannerModel` returning `{"plan": ..., "summary": ...}` verbatim."""

    def __init__(self, plans: list[dict[str, Any]]) -> None:
        self._plans = plans
        self.calls: list[dict[str, Any]] = []

    async def __call__(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        index = min(len(self.calls) - 1, len(self._plans) - 1)
        return self._plans[index]


async def test_the_planner_node_validates_the_seams_plan_and_writes_it_to_state(
    real_schemas: list[ToolSchema],
) -> None:
    """The seam returns `{"plan": <planner.plan.Plan JSON>, "summary": ...}`;
    the node validates it as a `planner.plan.Plan`, and emits `summary` and
    `revision` beside it -- both the record and the wire keys, not one
    constructed from parts the way the frozen adapter's protocol was.
    """
    plan = _seam_plan(schemas=real_schemas)
    seam = _Seam([{"plan": plan.model_dump(mode="json"), "summary": {"version": plan.version}}])
    runner, _ = _plan_runner(seam)

    events = await _start(runner)

    output = _output_of(events, "planner")
    assert output["summary"] == {"version": 1}
    assert output["revision"] == plan_revision(plan)
    plan_fields = {k: v for k, v in output.items() if k not in ("summary", "revision")}
    validated = PlannerPlan.model_validate(plan_fields)
    assert validated == plan


async def test_the_planner_node_raises_on_a_malformed_plan_rather_than_constructing_one() -> None:
    """A draft the seam could not itself validate must not reach the approve gate."""
    seam = _Seam([{"plan": {"decisions": "not-a-list"}, "summary": {}}])
    runner, _ = _plan_runner(seam)

    with pytest.raises(ValidationError):
        await _start(runner)


class _StallingSeam:
    """A `PlannerModel` that blocks mid-draft until released, recording
    whether its own task was cancelled and when it wrote to the graph's
    session state -- the two things a turn closed mid-draft must leave
    true: the draft is gone, and it wrote nothing after that close."""

    def __init__(self, *, ready: asyncio.Event, release: asyncio.Event) -> None:
        self._ready = ready
        self._release = release
        self.task: asyncio.Task[Any] | None = None
        self.state: dict[str, Any] | None = None

    async def __call__(self, *, ctx: Any, **_kwargs: Any) -> dict[str, Any]:
        self.task = asyncio.current_task()
        self.state = ctx.session.state
        self.state["seam_writes"] = [*self.state.get("seam_writes", []), "before"]
        self._ready.set()
        await self._release.wait()
        self.state["seam_writes"] = [*self.state.get("seam_writes", []), "after"]
        return {"plan": {}, "summary": {}}


async def test_a_turn_closed_mid_draft_cancels_the_seam_and_writes_no_state_after() -> None:
    """`studio._sse_stream`'s stop-and-disconnect path closes the ADK event
    stream through `_unwind`: cancel the in-flight next-frame task and
    await that cancellation, which throws into whatever node is suspended.
    The `planner` node is suspended in exactly that way whenever a draft is
    in flight: it awaits `models.planner(...)` directly (`harness.graph.planning`'s
    `planner`), so a cancellation there throws straight into the seam's own
    suspended `ctx.run_node` call. The draft must not survive: it is a live
    model and tool round trip that would otherwise keep running, and keep
    writing session state, after the turn has closed."""
    ready = asyncio.Event()
    release = asyncio.Event()
    seam = _StallingSeam(ready=ready, release=release)
    runner, _ = _plan_runner(seam)
    await open_session(runner)
    stream = runner.run_async(user_id="u", session_id="s", new_message=types.UserContent(_BRIEF))

    # Drive the stream one frame at a time, the way `studio._sse_stream`
    # does, until the seam signals a draft is in flight. `next_frame` is
    # then the in-flight `anext()` task, suspended inside the planner
    # node's own `await models.planner(...)` call -- exactly what
    # `_unwind`, below, closes over in the real stop-and-disconnect path.
    next_frame = asyncio.ensure_future(stream.__anext__())
    ready_wait = asyncio.ensure_future(ready.wait())
    async with asyncio.timeout(5):
        while not ready_wait.done():
            done, _ = await asyncio.wait(
                {next_frame, ready_wait}, return_when=asyncio.FIRST_COMPLETED
            )
            if next_frame in done:
                next_frame.result()
                next_frame = asyncio.ensure_future(stream.__anext__())
    assert seam.task is not None and not seam.task.done(), "the draft must still be in flight"

    # Cancel the in-flight next-frame task and await the cancellation,
    # rather than firing it and moving on, so whatever the closed generator
    # does to clean up actually runs before this assertion. Unlike
    # `studio._unwind`, this wait is unshielded: nothing here cancels the
    # awaiting task, so a shield would change nothing.
    next_frame.cancel()
    with suppress(asyncio.CancelledError):
        await next_frame

    # If the draft survived the close it is still blocked on `release`;
    # release it and let it finish, so a late write shows up if one lands.
    release.set()
    if not seam.task.done():
        with suppress(asyncio.CancelledError):
            await asyncio.wait_for(seam.task, timeout=5)

    assert seam.task.cancelled(), "a turn closed mid-draft must cancel the draft, not orphan it"
    assert seam.state is not None
    assert seam.state.get("seam_writes") == ["before"], "no draft write may land after the close"


class _AskingSeam:
    """A `PlannerModel` that reports two progress rounds, then pauses the
    same way a real `PlannerSeam` does once `planner.agent`'s
    `adk_request_input` tool call is what it's awaiting: `ctx.run_node`
    raises `NodeInterruptedError` straight out of the awaited call
    (`google.adk.workflow._errors`, a `BaseException` used exclusively for
    this). Raised directly here rather than through a real nested
    `ctx.run_node` call -- so the interrupt this synthesises carries none of
    ADK's own bookkeeping for resuming it later (`test_planner_seam.py`'s
    `test_a_planner_ask_pauses_the_run_and_elects_nothing_until_answered` proves
    that half, through a real one) -- this is only for the one property
    this node itself owns: drop the progress queued before the pause, then
    let it propagate."""

    async def __call__(self, *, on_progress: Any = None, **_kwargs: Any) -> dict[str, Any]:
        if on_progress is not None:
            on_progress({"agent": "dfo_planner", "tools": ["propose_plan"]})
            on_progress({"agent": "dfo_planner", "tools": ["adk_request_input"]})
        raise NodeInterruptedError()


async def test_a_planner_side_pause_relays_no_progress_after_it() -> None:
    """The `planner` node awaits the seam directly, so a pause's
    `NodeInterruptedError` passes straight through it. A real pause has
    already streamed the ask by then, so the node relays none of the
    progress queued before it (a frame after the ask breaks the stream's
    pause), and the pause reaches the workflow's own handling untouched. No
    `plan.proposed` or `refused` output follows: the turn is suspended,
    not settled."""
    seam = _AskingSeam()
    runner, _ = _plan_runner(seam)

    events = await _start(runner)

    progress = [
        event.custom_metadata["progress"]
        for event in events
        if event.custom_metadata and "progress" in event.custom_metadata
    ]
    assert progress == []
    assert _output_of(events, "planner") is None
    assert _output_of(events, "approve") is None
    assert _output_of(events, "plan_done") is None


class _GatedSeam:
    """A `PlannerModel` that drafts only once its caller has seen the
    planner node's start event, then pauses like `_AskingSeam`."""

    def __init__(self) -> None:
        self.seen = asyncio.Event()
        self.drafted = False

    async def __call__(self, **_kwargs: Any) -> dict[str, Any]:
        await asyncio.wait_for(self.seen.wait(), timeout=5)
        self.drafted = True
        raise NodeInterruptedError()


async def test_the_planner_node_announces_a_draft_before_the_seam_returns() -> None:
    """The node's first event reaches the runner stream while the seam is
    still drafting, so Studio can say a plan is under way."""
    seam = _GatedSeam()
    runner, _ = _plan_runner(seam)
    await open_session(runner)
    started = []
    async for event in runner.run_async(
        user_id="u", session_id="s", new_message=types.UserContent(_BRIEF)
    ):
        if event.custom_metadata and PLANNER_STARTED_KEY in event.custom_metadata:
            started.append(event.custom_metadata[PLANNER_STARTED_KEY])
            seam.seen.set()

    assert started == [{"revising": False}]
    assert seam.drafted


async def test_a_refused_redraft_over_an_unreadable_plan_says_the_plan_was_discarded(
    real_schemas: list[ToolSchema],
) -> None:
    """The user approved a plan the release cannot read, and the planner
    refuses to draft it again. No card follows to say what became of the plan
    they approved, so the planner's answer the agent reads opens with it."""
    plan = _seam_plan(schemas=real_schemas)
    seam = _Seam(
        [
            {"plan": plan.model_dump(mode="json"), "summary": {}},
            {"refused": {"text": "The draft budget is spent.", "final": True}},
        ]
    )
    runner, model = _plan_runner(seam, tool_surface=_fixture_surface(real_schemas))
    await _start(runner)
    _session_state(runner)["plan"] = json.dumps({"question": "kenya debt", "tool_calls": []})

    events = await _resume(runner, APPROVE_INTERRUPT, {"approved": 1})

    refused = _output_of(events, "planner")
    assert refused == {
        "text": f"{UNREADABLE_PLAN_UNPLANNED_NOTE} The draft budget is spent.",
        "final": True,
    }
    assert _pending_interrupt(events) is None
    assert (
        f"The planner could not make a plan: {UNREADABLE_PLAN_UNPLANNED_NOTE} The draft "
        "budget is spent." in request_text(model.requests[-1])
    )


_PLANNER_ASK = "planner_ask"


def _ask_the_user(ctx: Context) -> Iterator[Any]:
    """A node that pauses on a question and outputs the answer it is resumed with."""
    answer = (ctx.resume_inputs or {}).get(_PLANNER_ASK)
    if answer is None:
        yield create_request_input_event(
            RequestInput(interrupt_id=_PLANNER_ASK, message="Which years?")
        )
        return
    yield Event(output=answer)


class _AskingOnAmendSeam(_Seam):
    """A `PlannerModel` that drafts its first plan outright and asks the user
    a question, through a real `ctx.run_node` pause, on every later call."""

    async def __call__(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        if len(self.calls) > 1:
            await kwargs["ctx"].run_node(_ask_the_user)
        return self._plans[0]


async def test_a_planner_ask_resumed_over_an_unreadable_plan_drafts_from_the_start(
    real_schemas: list[ToolSchema],
) -> None:
    """The planner asks the user a question while amending a plan, and the
    plan on record is unreadable when the answer resumes it. The `planner`
    node finds that itself: nothing is left to amend, so the seam is handed
    no amendment and no prior plan, and the card says why."""
    plan = _seam_plan(schemas=real_schemas)
    seam = _AskingOnAmendSeam([{"plan": plan.model_dump(mode="json"), "summary": {}}])
    runner, _ = _plan_runner(seam, tool_surface=_fixture_surface(real_schemas))
    await _start(runner)
    events = await _resume(runner, APPROVE_INTERRUPT, {"approved": None, "amendment": "only 2020"})
    assert _pending_interrupt(events) == _PLANNER_ASK
    assert seam.calls[1]["amendment"] == "only 2020"
    _session_state(runner)["plan"] = json.dumps({"question": "kenya debt", "tool_calls": []})

    events = await _resume(runner, _PLANNER_ASK, {"answer": "2020"})

    assert _pending_interrupt(events) == APPROVE_INTERRUPT
    assert seam.calls[2]["amendment"] is None
    assert seam.calls[2]["prior_plan"] is None
    redrafted = _output_of(events, "planner")
    assert redrafted["refusal_note"] == UNREADABLE_PLAN_NOTE
    assert redrafted["restarted"] is True


async def test_an_approval_for_a_superseded_revision_routes_to_amend(
    real_schemas: list[ToolSchema],
) -> None:
    """The `approve` node keys a model-drafted plan on
    `plan_revision(plan)`, not the version integer. An answer naming a
    revision that does not match `ctx.state["plan"]` -- because an
    amendment already replaced it -- routes to "amend" the same way a stale
    version does, even when the superseding plan happens to share the
    same version number (`propose_plan` can start a fresh plan at version 1
    for what it reads as a different question, so two unrelated plans can
    both be version 1 in one conversation).
    """
    plan_v1 = _seam_plan(question="kenya debt", schemas=real_schemas)
    plan_v2 = _seam_plan(question="kenya debt, revised", schemas=real_schemas)
    seam = _Seam(
        [
            {"plan": plan_v1.model_dump(mode="json"), "summary": {}},
            {"plan": plan_v2.model_dump(mode="json"), "summary": {}},
        ]
    )
    executor = ScriptedExecutor()
    runner, _ = _plan_runner(seam, tool_surface=_fixture_surface(real_schemas), executor=executor)
    events = await _start(runner)
    assert _pending_interrupt(events) == APPROVE_INTERRUPT
    revision_1 = plan_revision(plan_v1)
    assert _output_of(events, "planner")["revision"] == revision_1

    # A wrong answer routes to amend, which redrafts -- an unrelated plan
    # that also happens to be version 1.
    events = await _resume(runner, APPROVE_INTERRUPT, {"approved": "not-a-real-revision"})
    assert _pending_interrupt(events) == APPROVE_INTERRUPT
    amended = _output_of(events, "planner")
    assert amended["version"] == 1
    revision_2 = amended["revision"]
    assert revision_2 != revision_1

    # The first card's revision no longer names what's pending: superseded,
    # not approved -- nothing executes and a fresh approval is still pending.
    events = await _resume(runner, APPROVE_INTERRUPT, {"approved": revision_1})
    assert executor.calls == []
    assert _pending_interrupt(events) == APPROVE_INTERRUPT
    assert _output_of(events, "planner")["revision"] == revision_2

    # The correct, current revision approves cleanly: `approve()` succeeds,
    # the gate is passed (no fresh approval interrupt), and the approved
    # workflow's steps ran through `execute_approved_plan` against the
    # tool surface: the executor only describes the final frame.
    events = await _resume(runner, APPROVE_INTERRUPT, {"approved": revision_2})
    assert _pending_interrupt(events) is None
    assert [call["tool"] for call in executor.calls] == ["describe_frame"]
    assert _output_of(events, "execute")["plan_version"] == 1


# The scripted plan as the code at commit 2563638 serialised it, beside the
# revision that code computed (`tests/planner/test_revision_format.py`). Its
# workflow is pinned to the schemas `ScriptedSurface` served at that commit,
# which it still serves.
_BASE_FORMAT_PLAN: dict[str, Any] = json.loads(
    (Path(__file__).parent / "fixtures" / "revision_base_plans.json").read_text()
)["aggregate"]


async def test_the_scripted_surface_declares_the_schemas_the_saved_plan_names() -> None:
    declared = {schema.name: schema for schema in await ScriptedSurface().list_tools()}

    for step in _BASE_FORMAT_PLAN["plan"]["workflow"]["steps"]:
        assert declared[step["tool"]["name"]].input_schema_hash == step["input_schema_hash"]


async def test_a_pending_plan_saved_at_the_base_revision_format_is_approved_unamended() -> None:
    """The card a user holds names the revision computed when the plan was
    saved. A plan saved without the fields its models have since gained
    still hashes to that revision, so the answer approves it and it runs."""
    seam = _Seam([{"plan": _BASE_FORMAT_PLAN["plan"], "summary": {}}])
    surface = _CallRecordingSurface(ScriptedSurface())
    runner, _ = _plan_runner(seam, tool_surface=surface)
    events = await _start(runner)
    assert _pending_interrupt(events) == APPROVE_INTERRUPT
    assert _output_of(events, "planner")["revision"] == _BASE_FORMAT_PLAN["revision"]
    # The session holds the plan exactly as it was saved.
    _session_state(runner)["plan"] = json.dumps(_BASE_FORMAT_PLAN["plan"])

    events = await _resume(runner, APPROVE_INTERRUPT, {"approved": _BASE_FORMAT_PLAN["revision"]})

    assert _pending_interrupt(events) is None
    assert len(seam.calls) == 1
    assert [call[0] for call in surface.calls] == ["get_data", "aggregate"]
    assert _output_of(events, "execute")["plan_version"] == 1
    finished = PlannerPlan.model_validate_json(_session_state(runner)["plan"])
    assert finished.status is PlanStatus.COMPLETED
    assert finished.approved_revision == _BASE_FORMAT_PLAN["revision"]


async def test_an_approved_plan_saved_at_the_base_revision_format_executes() -> None:
    """`approved_revision` was computed when the plan was approved;
    `Plan.start_execution` compares it with the revision computed now."""
    approved = PlannerPlan.model_validate(
        _BASE_FORMAT_PLAN["plan"]
        | {"status": "approved", "approved_revision": _BASE_FORMAT_PLAN["revision"]}
    )

    finished, outcome = await execute_approved_plan(
        approved, surface=ScriptedSurface(), fetch_series_stats=scripted_fetch_series_stats
    )

    assert isinstance(outcome, Accepted), outcome
    assert finished.status is PlanStatus.COMPLETED


async def test_a_saved_plan_whose_retrieval_posts_its_spec_id_alone_runs_as_saved() -> None:
    """The saved retrieval step carries `spec_id` alone, as `get_data` takes
    it, so the call is made as saved."""
    approved = PlannerPlan.model_validate(
        _BASE_FORMAT_PLAN["plan"]
        | {"status": "approved", "approved_revision": _BASE_FORMAT_PLAN["revision"]}
    )
    surface = _CallRecordingSurface(ScriptedSurface())

    _, outcome = await execute_approved_plan(
        approved, surface=surface, fetch_series_stats=scripted_fetch_series_stats
    )

    assert isinstance(outcome, Accepted), outcome
    assert surface.calls[0] == ("get_data", {"spec_id": "spec_" + "f" * 32})


# A population rank as the code at commit 2563638 compiled it against the tool
# schemas the servers declared then.
_BASE_FORMAT_LIVE_PLAN: dict[str, Any] = json.loads(
    (Path(__file__).parent / "fixtures" / "revision_base_plans.json").read_text()
)["rank"]


async def test_an_approved_plan_compiled_against_an_older_get_data_is_refused_for_reapproval(
    real_schemas: list[ToolSchema],
) -> None:
    """Its retrieval step names the argument schema `get_data` declared when it
    was compiled. The run stops before any call, with the step and the remedy
    named, and the plan keeps its revision."""
    saved = _BASE_FORMAT_LIVE_PLAN["plan"]
    (retrieval,) = (s for s in saved["workflow"]["steps"] if s["tool"]["name"] == "get_data")
    approved = PlannerPlan.model_validate(
        saved | {"status": "approved", "approved_revision": _BASE_FORMAT_LIVE_PLAN["revision"]}
    )
    surface = _CallRecordingSurface(_fixture_surface(real_schemas, analysis_tool="rank"))

    _, outcome = await execute_approved_plan(
        approved, surface=surface, fetch_series_stats=scripted_fetch_series_stats
    )

    assert isinstance(outcome, Refused), outcome
    assert outcome.refusal is RefusalKind.SURFACE_CHANGED
    assert outcome.step_id == retrieval["id"]
    assert outcome.detail == (
        "'get_data' takes different arguments than the plan was compiled against; "
        "the plan needs compiling and approving again"
    )
    assert outcome.revision == _BASE_FORMAT_LIVE_PLAN["revision"]
    assert surface.calls == []


async def test_a_refused_amendment_still_offers_the_pending_version_for_approval(
    real_schemas: list[ToolSchema],
) -> None:
    """A `PlannerModel` seam can refuse to draft at all
    (`harness.planner_seam.PlannerSeam`'s own `MAX_PLAN_ROUNDS` cap,
    surfaced as `{"refused": {"text": ...}}` rather than a raised
    `PlannerSeamError`). The planner node never writes `ctx.state["plan"]`
    on that path, so the plan the user already had -- still
    `PENDING_APPROVAL` -- stays exactly as it was. Since that plan is
    still real and still pending, the refusal is not a dead end: the node
    routes back to `approve`, which raises a fresh interrupt for the same
    plan, and answering it with the plan's own original revision approves
    it -- proving the refusal's claim that the version is "still there to
    approve" by actually approving it, not only by reading the sentence
    that says so.
    """
    plan_v1 = _seam_plan(question="kenya debt", schemas=real_schemas)
    seam = _Seam(
        [
            {"plan": plan_v1.model_dump(mode="json"), "summary": {}},
            {
                "refused": {
                    "text": "This request has used its budget of 4 plan drafts, so nothing "
                    "was recorded. Your next message starts with a fresh budget."
                }
            },
        ]
    )
    executor = ScriptedExecutor()
    runner, _ = _plan_runner(seam, tool_surface=_fixture_surface(real_schemas), executor=executor)
    events = await _start(runner)
    assert _pending_interrupt(events) == APPROVE_INTERRUPT
    revision_1 = plan_revision(plan_v1)
    assert _output_of(events, "planner")["revision"] == revision_1

    events = await _resume(runner, APPROVE_INTERRUPT, {"approved": "not-a-real-revision"})

    # Not a dead end: a fresh approval interrupt for the same, unmodified
    # plan comes back instead, carrying why the amendment did not land.
    assert "plan_done" not in visited_nodes(events)
    assert _pending_interrupt(events) == APPROVE_INTERRUPT
    refused_output = _output_of(events, "planner")
    assert refused_output["revision"] == revision_1
    assert "budget of 4 plan drafts" in refused_output["refusal_note"]
    assert "still there to approve" in refused_output["refusal_note"]

    # `ctx.state["plan"]` was only ever written by the first, successful
    # planner call -- the refused second call left it untouched, so the
    # plan the user already saw is still exactly what's on record.
    assert PlannerPlan.model_validate_json(_session_state(runner)["plan"]) == plan_v1

    # Answering the fresh interrupt with the version's own original
    # revision approves it -- the round-cap refusal cost the user their
    # amendment, not their already-drafted plan. The plan's own compiled
    # workflow runs it through to completion against the tool surface.
    events = await _resume(runner, APPROVE_INTERRUPT, {"approved": revision_1})
    assert _pending_interrupt(events) is None
    approved = PlannerPlan.model_validate_json(_session_state(runner)["plan"])
    assert approved.status is PlanStatus.COMPLETED


# --- the execute node: the compiled workflow, the tool surface, the outcome map --


async def test_the_execute_node_runs_the_approved_workflow_through_the_surface(
    real_schemas: list[ToolSchema],
) -> None:
    """`execute` calls `execute_approved_plan` against the injected
    `ToolSurface`; the executor sees no plan step, only `plan_done`'s
    description of the final frame."""
    plan = _seam_plan(schemas=real_schemas)
    seam = _Seam([{"plan": plan.model_dump(mode="json"), "summary": {}}])
    executor = ScriptedExecutor()
    surface = _CallRecordingSurface(_fixture_surface(real_schemas))
    runner, _ = _plan_runner(seam, tool_surface=surface, executor=executor)
    events = await _start(runner)
    revision = plan_revision(plan)

    events = await _resume(runner, APPROVE_INTERRUPT, {"approved": revision})

    assert [call[0] for call in surface.calls] == ["get_data", "aggregate"]
    assert [call["tool"] for call in executor.calls] == ["describe_frame"]
    executed = _output_of(events, "execute")
    # `Accepted.results` is terminal-steps-only: one frame ref
    # for "totals", not one per step.
    assert executed["frame_refs"] == [
        {"spec_id": "ops_" + "2" * 32, "content_version": "ocv_" + "3" * 32}
    ]
    assert executed["step_summaries"] == ["fetched", "computed"]
    assert executed["plan_version"] == 1
    assert _output_of(events, "execute")["plan_version"] == 1


class _ServedPairsPlatform:
    """presentation-mcp over dfo-apis, which serves a frame under the pair it
    minted: `describe_frame` refuses `frame_unavailable` on any other pair and
    answers ok on one it serves."""

    def __init__(self, served: list[dict[str, str]]) -> None:
        self.served = served
        self.calls: list[dict[str, Any]] = []

    async def __call__(
        self, *, tool: str, args: dict[str, Any], on_progress: ProgressCallback | None = None
    ) -> dict[str, Any]:
        del on_progress
        self.calls.append({"tool": tool, "args": args})
        if args["frame_ref"] not in self.served:
            return {
                "outcome": "refusal",
                "refusal": {
                    "code": "frame_unavailable",
                    "retry_mode": "do_not_retry",
                    "origin": "service",
                    "fix": None,
                    "message": "no stored version",
                },
            }
        return {"outcome": "ok", "view": {"dimensions": ["stat_var"]}}


async def test_a_final_retrieval_read_with_extra_columns_is_shown_by_its_manifests_pair(
    real_schemas: list[ToolSchema],
) -> None:
    """A retrieval the plan answers with is held by data-mcp under a pair of its
    own once read with extra columns, and dfo-apis serves only its manifest's pair.
    `execute` hands on the manifest's pair, so `plan_done` describes the series
    and registers it for charting, while the total reading it was given the
    frame's own pair."""
    plan = _seam_plan(schemas=real_schemas, terminal=("get_data", "totals"))
    seam = _Seam([{"plan": plan.model_dump(mode="json"), "summary": {}}])
    served = {"spec_id": SPEC_A, "content_version": "cv_" + "1" * 32}
    typed = {"spec_id": SPEC_A, "content_version": "cv_" + "9" * 32}
    total = {"spec_id": "ops_" + "2" * 32, "content_version": "ocv_" + "3" * 32}
    surface = _CallRecordingSurface(
        _fixture_surface(
            real_schemas,
            get_data_result=ToolCallResult(
                is_error=False,
                structured_content=_get_data_manifest() | {"frame_ref": typed},
                text="fetched",
            ),
        )
    )
    platform = _ServedPairsPlatform([served, total])
    runner, model = _plan_runner(seam, tool_surface=surface, executor=platform)
    await _start(runner)

    events = await _resume(runner, APPROVE_INTERRUPT, {"approved": plan_revision(plan)})

    (aggregate,) = [args for name, args in surface.calls if name == "aggregate"]
    assert aggregate["frame_ref"] == typed
    assert _output_of(events, "execute")["frame_refs"] == [served, total]
    assert [(call["tool"], call["args"]["frame_ref"]) for call in platform.calls] == [
        ("describe_frame", served),
        ("describe_frame", total),
    ]
    _, body = handed_back(model)
    assert [entry["frame"] for entry in body["frames"]] == [
        frame_id_for(served),
        frame_id_for(total),
    ]
    assert "frames_not_chartable" not in body


async def test_a_summed_familys_total_reaches_the_agent_in_the_hand_back(
    real_schemas: list[ToolSchema], monkeypatch: pytest.MonkeyPatch
) -> None:
    """`execute` hands `plan_done` the sentences `_stat_var_totals` writes,
    and `plan_done` gives each to the agent under `totals`."""
    sentence = (
        'By default, the figures for "OECD Creditor Reporting System (CRS)" sum the '
        "family's 74 indicators."
    )
    monkeypatch.setattr(answer_module, "_stat_var_totals", lambda *a, **k: [sentence])
    plan = _seam_plan(schemas=real_schemas)
    seam = _Seam([{"plan": plan.model_dump(mode="json"), "summary": {}}])
    runner, model = _plan_runner(seam, tool_surface=_fixture_surface(real_schemas))
    await _start(runner)

    events = await _resume(runner, APPROVE_INTERRUPT, {"approved": plan_revision(plan)})

    assert _output_of(events, "execute")["stat_var_totals"] == [sentence]
    _, body = handed_back(model)
    assert body["totals"] == [sentence]


async def test_a_none_tool_surface_refuses_loudly_rather_than_executing_nothing(
    real_schemas: list[ToolSchema],
) -> None:
    """`create_root_workflow(tool_surface=None)`, stated explicitly, must not
    let an approved plan reach execute silently: reading a plan the node
    cannot run has to fail loudly."""
    plan = _seam_plan(schemas=real_schemas)
    seam = _Seam([{"plan": plan.model_dump(mode="json"), "summary": {}}])
    runner, _ = _plan_runner(seam)
    await _start(runner)
    revision = plan_revision(plan)

    with pytest.raises(RuntimeError, match="no tool surface"):
        await _resume(runner, APPROVE_INTERRUPT, {"approved": revision})


async def test_a_rediscovery_outcome_replans_naming_the_requirement(
    real_schemas: list[ToolSchema],
) -> None:
    """A retrieval step's own evidence binding names the requirement and the
    source a `RediscoveryNeeded` refusal reports -- a bare retry would only
    reproduce it, so `execute` routes back to the planner naming what to
    rediscover, never retrying the call itself."""
    plan = _seam_plan(schemas=real_schemas, bind_evidence=True)
    seam = _Seam([{"plan": plan.model_dump(mode="json"), "summary": {}}])
    refusal = ToolCallResult(
        is_error=True,
        structured_content={
            "code": "upstream_version_changed",
            "retry_mode": "refresh_then_retry",
            "message": "the source's upstream version changed since this plan was approved",
        },
        text="refused",
    )
    surface = _fixture_surface(real_schemas, get_data_result=refusal)
    runner, _ = _plan_runner(seam, tool_surface=surface)
    events = await _start(runner)
    revision = plan_revision(plan)

    events = await _resume(runner, APPROVE_INTERRUPT, {"approved": revision})

    # Replanned (MAX_REPLANS == 1 covers one round): a fresh approval is
    # pending rather than the turn closing on the first refusal.
    assert _pending_interrupt(events) == APPROVE_INTERRUPT
    assert seam.calls[-1]["refusal"] == {
        "kind": "rediscovery_needed",
        "step_id": "get_data",
        "tool": "get_data",
        "requirement_id": "spending",
        "spec_id": SPEC_A,
        "candidates": None,
        "refusal": {
            "code": "upstream_version_changed",
            "retry_mode": "refresh_then_retry",
            "message": "the source's upstream version changed since this plan was approved",
            "fix": {},
        },
        "also_stopped": [],
    }


async def test_a_method_choice_refusal_replans_once_then_closes_with_failed_step(
    real_schemas: list[ToolSchema],
) -> None:
    """`MAX_REPLANS == 1`: a second identical refusal closes the turn with
    what ran and the failed step named, rather than a third plan."""
    plan = _seam_plan(schemas=real_schemas)
    seam = _Seam([{"plan": plan.model_dump(mode="json"), "summary": {}}])
    refusal = ToolCallResult(
        is_error=True,
        structured_content={
            "code": "entity_ambiguous",
            "retry_mode": "correct_and_retry",
            "message": "more than one entity matches 'Kenya'",
            "fix": {"candidates": ["country/KEN", "country/KEN-old"]},
        },
        text="refused",
    )
    surface = _fixture_surface(real_schemas, analysis_call_result=refusal)
    executor = ScriptedExecutor()
    runner, model = _plan_runner(seam, tool_surface=surface, executor=executor)
    events = await _start(runner)
    revision = plan_revision(plan)

    events = await _resume(runner, APPROVE_INTERRUPT, {"approved": revision})
    assert _pending_interrupt(events) == APPROVE_INTERRUPT
    replanned = _output_of(events, "planner")
    events = await _resume(runner, APPROVE_INTERRUPT, {"approved": replanned["revision"]})

    assert _pending_interrupt(events) is None
    executed = _output_of(events, "execute")
    assert executed["failed_step"] == {
        "step_id": "totals",
        "tool": "aggregate",
        "code": "entity_ambiguous",
        "message": "more than one entity matches 'Kenya'",
    }
    # get_data ran and completed before the refusal; its frame is kept.
    assert executed["frame_refs"] == [REF_A]
    assert executed["step_summaries"] == ["fetched"]
    _, body = handed_back(model)
    assert body["failed_step"] == executed["failed_step"]


async def test_a_stop_classified_refusal_closes_the_turn_without_spending_a_replan(
    real_schemas: list[ToolSchema],
) -> None:
    """An `over_ceiling` refusal naming no `ceiling_kind` at all -- an older
    upstream, or a raw payload assembled by hand -- is one
    `executor.classify.classify_refusal` cannot place as narrowing with scope
    or as naming a different spec, so it stays a `STOP`: `execute` closes the
    turn on this first refusal instead of spending the turn's one replan on a
    guess. Unlike `test_a_method_choice_refusal_
    replans_once_then_closes_with_failed_step`'s `entity_ambiguous`, no
    second approval is ever pending."""
    plan = _seam_plan(schemas=real_schemas)
    seam = _Seam([{"plan": plan.model_dump(mode="json"), "summary": {}}])
    refusal = ToolCallResult(
        is_error=True,
        structured_content={
            "code": "over_ceiling",
            "retry_mode": "correct_and_retry",
            "message": "over the coverage ceiling: too many series.",
            "fix": {"detail": "too many series"},
        },
        text="refused",
    )
    surface = _fixture_surface(real_schemas, analysis_call_result=refusal)
    runner, model = _plan_runner(seam, tool_surface=surface)
    events = await _start(runner)
    revision = plan_revision(plan)

    events = await _resume(runner, APPROVE_INTERRUPT, {"approved": revision})

    # No replan, no second approval: the turn closes on the first refusal.
    assert _pending_interrupt(events) is None
    executed = _output_of(events, "execute")
    # A plain `Refused` (the `_stopped_by_refusal` catch-all `STEP_REFUSED`,
    # not a `MethodChoiceNeeded`/`RediscoveryNeeded`) carries `tool=None`,
    # but `_failed_step` reports the tool's own `code`/`message` off
    # `outcome.payload` -- the `ToolFailure` the raw call actually returned
    # -- rather than the internal `RefusalKind` that closed the turn.
    assert executed["failed_step"] == {
        "step_id": "totals",
        "tool": None,
        "code": "over_ceiling",
        "message": "over the coverage ceiling: too many series.",
    }
    assert executed["frame_refs"] == [REF_A]
    _, body = handed_back(model)
    assert body["failed_step"] == {
        "step_id": "totals",
        "code": "over_ceiling",
        "message": "over the coverage ceiling: too many series.",
    }


async def test_a_member_cardinality_ceiling_closes_the_turn_without_replanning(
    real_schemas: list[ToolSchema],
) -> None:
    """A `member_cardinality` `over_ceiling` refusal is fixed by the resolved
    spec_id's own five-tuple and the index version behind it: no scope edit
    narrows it, and rediscovering the same spec by paraphrase search
    reproduces the identical refusal. `classify_refusal` classifies it
    `STOP`, so `execute` closes the turn on this first refusal instead of
    spending the turn's one replan on a rediscovery that cannot help."""
    plan = _seam_plan(schemas=real_schemas, bind_evidence=True)
    seam = _Seam([{"plan": plan.model_dump(mode="json"), "summary": {}}])
    refusal = ToolCallResult(
        is_error=True,
        structured_content={
            "code": "over_ceiling",
            "retry_mode": "correct_and_retry",
            "message": "over the coverage ceiling: 2933 members. A narrower question "
            "resolves to a smaller definition.",
            "fix": {
                "detail": "2933 members",
                "ceiling_kind": "member_cardinality",
                "count": 2933,
                "cap": 700,
            },
        },
        text="refused",
    )
    surface = _fixture_surface(real_schemas, get_data_result=refusal)
    runner, model = _plan_runner(seam, tool_surface=surface)
    events = await _start(runner)
    revision = plan_revision(plan)

    events = await _resume(runner, APPROVE_INTERRUPT, {"approved": revision})

    # No replan, no second approval: the turn closes on the first refusal.
    assert _pending_interrupt(events) is None
    executed = _output_of(events, "execute")
    assert executed["failed_step"] == {
        "step_id": "get_data",
        "tool": None,
        "code": "over_ceiling",
        "message": "over the coverage ceiling: 2933 members. A narrower question "
        "resolves to a smaller definition.",
    }
    _, body = handed_back(model)
    assert (body["failed_step"]["step_id"], body["failed_step"]["code"]) == (
        "get_data",
        "over_ceiling",
    )


async def test_a_member_cardinality_ceilings_remedy_reaches_the_agent_in_the_hand_back(
    real_schemas: list[ToolSchema],
) -> None:
    """`fix.remedy`, the one user-facing sentence naming what the user can do
    after a `member_cardinality` ceiling's dead end, reaches
    the agent verbatim in the hand-back's `failed_step`, beside the code and
    message."""
    plan = _seam_plan(schemas=real_schemas, bind_evidence=True)
    seam = _Seam([{"plan": plan.model_dump(mode="json"), "summary": {}}])
    remedy = (
        "This spec_id's definition has more member series than one request can "
        "serve. scope.entities and scope.window do not change that. A narrower "
        "question, such as one naming a sector for aid flows, resolves to a "
        "smaller definition."
    )
    refusal = ToolCallResult(
        is_error=True,
        structured_content={
            "code": "over_ceiling",
            "retry_mode": "correct_and_retry",
            "message": f"over the coverage ceiling: 2933 members. {remedy}",
            "fix": {
                "detail": "2933 members",
                "ceiling_kind": "member_cardinality",
                "count": 2933,
                "cap": 700,
                "remedy": remedy,
            },
        },
        text="refused",
    )
    surface = _fixture_surface(real_schemas, get_data_result=refusal)
    runner, model = _plan_runner(seam, tool_surface=surface)
    await _start(runner)

    events = await _resume(runner, APPROVE_INTERRUPT, {"approved": plan_revision(plan)})

    assert _pending_interrupt(events) is None
    assert _output_of(events, "execute")["failed_step"]["remedy"] == remedy
    _, body = handed_back(model)
    assert body["failed_step"]["remedy"] == remedy


async def test_a_family_returned_unsummed_closes_the_turn_on_a_message_the_user_can_act_on(
    real_schemas: list[ToolSchema],
) -> None:
    """A `get_data` asked to sum its family declares three StatVars, so the
    sum did not happen. The same plan draws the same frame, so `execute`
    closes the turn on this first refusal: no replan, no second approval of
    an identical plan. The agent reads the executor's own message and
    remedy, both written for the user, in place of the count of `stat_vars`
    the refusal's `detail` states."""
    plan = _seam_plan(schemas=real_schemas, stat_vars="sum")
    seam = _Seam([{"plan": plan.model_dump(mode="json"), "summary": {}}])
    unsummed = ToolCallResult(
        is_error=False,
        structured_content=_get_data_manifest() | {"stat_var_count": 3},
        text="fetched",
    )
    surface = _fixture_surface(real_schemas, get_data_result=unsummed)
    runner, model = _plan_runner(seam, tool_surface=surface)
    events = await _start(runner)

    events = await _resume(runner, APPROVE_INTERRUPT, {"approved": plan_revision(plan)})

    assert _pending_interrupt(events) is None
    assert len(seam.calls) == 1
    message = (
        "The data service returned this measure's component series side by side and did "
        "not add them up, so there is no total to report."
    )
    remedy = "Ask for the component series separately, or ask for the total again later."
    executed = _output_of(events, "execute")
    assert executed["failed_step"] == {
        "step_id": "get_data",
        "tool": None,
        "code": "family_not_summed",
        "message": message,
        "remedy": remedy,
    }
    assert executed["frame_refs"] == []
    text, body = handed_back(model)
    assert body["failed_step"] == {
        "step_id": "get_data",
        "code": "family_not_summed",
        "message": message,
        "remedy": remedy,
    }
    assert "distinct stat_vars" not in text


async def test_an_over_ceiling_rows_refusal_still_replans(
    real_schemas: list[ToolSchema],
) -> None:
    """A `rows`/`bytes` ceiling narrows with `scope.entities`/`scope.window`,
    so -- unlike `member_cardinality` above -- it is a method choice, not a
    dead end, and replans once like any other correctable refusal."""
    plan = _seam_plan(schemas=real_schemas)
    seam = _Seam([{"plan": plan.model_dump(mode="json"), "summary": {}}])
    refusal = ToolCallResult(
        is_error=True,
        structured_content={
            "code": "over_ceiling",
            "retry_mode": "correct_and_retry",
            "message": "over the coverage ceiling: too many rows. Narrow the scope and retry.",
            "fix": {"detail": "too many rows", "ceiling_kind": "rows"},
        },
        text="refused",
    )
    surface = _fixture_surface(real_schemas, analysis_call_result=refusal)
    runner, _ = _plan_runner(seam, tool_surface=surface)
    events = await _start(runner)
    revision = plan_revision(plan)

    events = await _resume(runner, APPROVE_INTERRUPT, {"approved": revision})

    # A method choice, not a dead end: the planner is asked again (the
    # scripted seam redrafts the same plan, so the version does not bump),
    # and a fresh approval is pending rather than the turn closing.
    assert _pending_interrupt(events) == APPROVE_INTERRUPT
    assert len(seam.calls) == 2
    assert seam.calls[-1]["refusal"]["code"] == "over_ceiling"


async def test_a_not_approved_outcome_raises_instead_of_replanning(
    real_schemas: list[ToolSchema], monkeypatch: pytest.MonkeyPatch
) -> None:
    """`Refused(NOT_APPROVED)` means the workflow's own `approve` node let an
    unapproved plan through -- replanning would loop on the same defect, so
    `execute` raises instead. The `approve` node cannot itself
    produce this in normal operation (it calls `plan.approve()` immediately
    before routing here), so the outcome is stubbed to prove this node's own
    branch rather than trying to coax `approve` into a defect it structurally
    prevents."""
    plan = _seam_plan(schemas=real_schemas)
    seam = _Seam([{"plan": plan.model_dump(mode="json"), "summary": {}}])

    async def _stub_not_approved(
        drafted_plan: Any, *, surface: Any, max_concurrent_steps: int = 4, **_kwargs: Any
    ) -> Any:
        del surface, max_concurrent_steps
        return drafted_plan, Refused(
            revision=plan_revision(drafted_plan),
            refusal=RefusalKind.NOT_APPROVED,
            detail="this plan is approved, not approved; nothing runs",
        )

    monkeypatch.setattr("harness.graph.execute.execute_approved_plan", _stub_not_approved)
    runner, _ = _plan_runner(seam, tool_surface=_fixture_surface(real_schemas))
    await _start(runner)
    revision = plan_revision(plan)

    with pytest.raises(RuntimeError, match="not_approved"):
        await _resume(runner, APPROVE_INTERRUPT, {"approved": revision})


def test_the_replan_refusal_and_failed_step_maps_cover_every_outcome_kind() -> None:
    """`_replan_refusal`'s payload shapes, pinned directly against each outcome
    kind -- not merely reachable through one graph run apiece."""
    method_choice = MethodChoiceNeeded(
        revision="rev1",
        step_id="totals",
        tool=ToolId(server="analysis-mcp", name="aggregate"),
        code="entity_ambiguous",
        candidates=["country/KEN", "country/KEN-old"],
        refusal=ToolFailure(
            code="entity_ambiguous", message="ambiguous", retry_mode="correct_and_retry"
        ),
        completed=(),
    )
    assert _replan_refusal(method_choice) == {
        "kind": "method_choice_needed",
        "step_id": "totals",
        "tool": "aggregate",
        "code": "entity_ambiguous",
        "candidates": ["country/KEN", "country/KEN-old"],
        "refusal": {
            "code": "entity_ambiguous",
            "message": "ambiguous",
            "retry_mode": "correct_and_retry",
            "fix": {},
        },
        "also_stopped": [],
    }
    assert _failed_step(method_choice) == {
        "step_id": "totals",
        "tool": "aggregate",
        "code": "entity_ambiguous",
        "message": "ambiguous",
    }

    rediscovery = RediscoveryNeeded(
        revision="rev1",
        step_id="get_data",
        tool=ToolId(server="data-mcp", name="get_data"),
        requirement_id="spending",
        spec_id=SPEC_A,
        refusal=ToolFailure(
            code="upstream_version_changed",
            message="the source moved",
            retry_mode="refresh_then_retry",
        ),
        completed=(),
    )
    assert _replan_refusal(rediscovery) == {
        "kind": "rediscovery_needed",
        "step_id": "get_data",
        "tool": "get_data",
        "requirement_id": "spending",
        "spec_id": SPEC_A,
        # An empty `fix` carries nothing to choose from -- `candidates_of`
        # reads `None`, not an empty dict standing in for "no candidates".
        "candidates": None,
        "refusal": {
            "code": "upstream_version_changed",
            "message": "the source moved",
            "retry_mode": "refresh_then_retry",
            "fix": {},
        },
        "also_stopped": [],
    }
    assert _failed_step(rediscovery) == {
        "step_id": "get_data",
        "tool": "get_data",
        "code": "rediscovery_needed",
        "message": "the source moved",
    }

    refused = Refused(
        revision="rev1",
        refusal=RefusalKind.EMPTY_ANSWER,
        detail="no rows matched",
        step_id="totals",
    )
    assert _replan_refusal(refused) == {
        "kind": "refused",
        "refusal": "empty_answer",
        "detail": "no rows matched",
        "step_id": "totals",
        "requirement_id": None,
        "issues": [],
        "also_stopped": [],
    }
    assert _failed_step(refused) == {
        "step_id": "totals",
        "tool": None,
        "code": "empty_answer",
        "message": "no rows matched",
    }

    # A `STEP_REFUSED` stop carries the tool's own `ToolFailure` as
    # `payload` -- `_failed_step` reports its `code`/`message`, not
    # `refusal.value`/`detail`, so the agent is told what the
    # tool actually said ("over_ceiling") rather than the internal kind
    # that closed the turn ("step_refused").
    refused_with_payload = Refused(
        revision="rev1",
        refusal=RefusalKind.STEP_REFUSED,
        detail="'aggregate' refused: refused",
        step_id="totals",
        payload=ToolFailure(
            code="over_ceiling",
            message="over the coverage ceiling: too many series",
            retry_mode="correct_and_retry",
        ),
    )
    assert _failed_step(refused_with_payload) == {
        "step_id": "totals",
        "tool": None,
        "code": "over_ceiling",
        "message": "over the coverage ceiling: too many series",
    }

    # A `REQUIREMENT_UNMET` stop carries `check_evidence`'s own findings as
    # `issues` -- `_failed_step` reports the specific `EvidenceIssueCode`
    # and detail of whichever ERROR(s) fired, not `refusal.value`/`detail`'s
    # generic "requirement X is not met by what Y returned", which names
    # neither which of the ten sub-checks failed nor what it found. A
    # `WARNING`-severity issue on the same outcome (a real shape:
    # `check_evidence` can return caveats alongside the errors that stopped
    # the run) is left out of the reported code/message; only errors decide
    # what the run reports as its reason.
    refused_with_issues = Refused(
        revision="rev1",
        refusal=RefusalKind.REQUIREMENT_UNMET,
        detail="requirement 'req_gdp' is not met by what 'get_data' returned",
        step_id="fetch_gdp",
        requirement_id="req_gdp",
        issues=(
            EvidenceIssue(
                code=EvidenceIssueCode.SOURCE_UNVERIFIED,
                severity=Severity.WARNING,
                requirement_id="req_gdp",
                detail="the source states no provenance",
            ),
            EvidenceIssue(
                code=EvidenceIssueCode.SOURCE_PIN_MISMATCH,
                severity=Severity.ERROR,
                requirement_id="req_gdp",
                detail="the requirement pins 'spec_a', the source found is 'spec_b'",
                found="spec_b",
                required="spec_a",
            ),
        ),
    )
    assert _failed_step(refused_with_issues) == {
        "step_id": "fetch_gdp",
        "tool": None,
        "code": "source_pin_mismatch",
        "message": "the requirement pins 'spec_a', the source found is 'spec_b'",
    }

    # No ERROR-severity issue at all (every finding a caveat) falls back to
    # the first issue given, not nothing. `check_evidence` does not produce
    # this case (it only raises when `errors(findings)` is non-empty), but
    # `_failed_step` owes a code/message pair for whatever `issues` it is
    # handed.
    refused_with_only_warnings = Refused(
        revision="rev1",
        refusal=RefusalKind.REQUIREMENT_UNMET,
        detail="requirement 'req_gdp' is not met by what 'get_data' returned",
        step_id="fetch_gdp",
        requirement_id="req_gdp",
        issues=(
            EvidenceIssue(
                code=EvidenceIssueCode.SOURCE_UNVERIFIED,
                severity=Severity.WARNING,
                requirement_id="req_gdp",
                detail="the source states no provenance",
            ),
        ),
    )
    assert _failed_step(refused_with_only_warnings) == {
        "step_id": "fetch_gdp",
        "tool": None,
        "code": "source_unverified",
        "message": "the source states no provenance",
    }


def test_replan_refusal_names_every_sibling_a_wave_surfaced() -> None:
    """`also_stopped` carries the rest of a wave's stops in the primary's own
    per-kind shape, empty when the wave surfaced only the one outcome
    (`executor.run._Run.execute`'s siblings, folded into the outcome
    `_run_workflow` returns)."""
    sibling = RediscoveryNeeded(
        revision="rev1",
        step_id="fetch_population",
        tool=ToolId(server="data-mcp", name="get_data"),
        requirement_id="population",
        spec_id=SPEC_A,
        refusal=ToolFailure(
            code="upstream_version_changed",
            message="the population source moved too",
            retry_mode="refresh_then_retry",
        ),
        completed=(),
    )
    primary = RediscoveryNeeded(
        revision="rev1",
        step_id="get_data",
        tool=ToolId(server="data-mcp", name="get_data"),
        requirement_id="spending",
        spec_id=SPEC_A,
        refusal=ToolFailure(
            code="upstream_version_changed",
            message="the source moved",
            retry_mode="refresh_then_retry",
        ),
        completed=(),
        also_stopped=(sibling,),
    )

    payload = _replan_refusal(primary)

    assert payload["also_stopped"] == [
        {
            "kind": "rediscovery_needed",
            "step_id": "fetch_population",
            "tool": "get_data",
            "requirement_id": "population",
            "spec_id": SPEC_A,
            "candidates": None,
            "refusal": {
                "code": "upstream_version_changed",
                "message": "the population source moved too",
                "retry_mode": "refresh_then_retry",
                "fix": {},
            },
        }
    ]
    assert _replan_refusal(sibling)["also_stopped"] == []


def test_mark_masked_step_failed_corrects_every_sibling_a_wave_surfaced() -> None:
    """A wave whose primary and one `also_stopped` sibling both passed their
    raw call but were refused by a post-call check (`_check_evidence`) each
    carry a distinct `step_id`. `_mark_masked_step_failed` corrects both
    Activity feed records, not just the primary's."""

    class _RecorderStub:
        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []

        def mark_step_failed(self, *, step_id: str, tool: str, code: str, detail: str) -> None:
            self.calls.append({"step_id": step_id, "code": code, "detail": detail})

    sibling = Refused(
        revision="rev1",
        refusal=RefusalKind.EMPTY_ANSWER,
        detail="no rows matched",
        step_id="fetch_population",
    )
    primary = Refused(
        revision="rev1",
        refusal=RefusalKind.REQUIREMENT_UNMET,
        detail="requirement 'req_gdp' is not met by what 'get_data' returned",
        step_id="fetch_gdp",
        requirement_id="req_gdp",
        issues=(
            EvidenceIssue(
                code=EvidenceIssueCode.SOURCE_PIN_MISMATCH,
                severity=Severity.ERROR,
                requirement_id="req_gdp",
                detail="the requirement pins 'spec_a', the source found is 'spec_b'",
                found="spec_b",
                required="spec_a",
            ),
        ),
        also_stopped=(sibling,),
    )

    recorder = _RecorderStub()
    _mark_masked_step_failed(recorder, primary, tools={})

    assert recorder.calls == [
        {
            "step_id": "fetch_gdp",
            "code": "source_pin_mismatch",
            "detail": "the requirement pins 'spec_a', the source found is 'spec_b'",
        },
        {
            "step_id": "fetch_population",
            "code": "empty_answer",
            "detail": "no rows matched",
        },
    ]


# --- the seed intake's wording ---------------------------------------------------


def test_a_stale_reason_for_another_dataset_omits_a_spec_id_search_did_not_name() -> None:
    part = {"status": "definite"}
    unnamed = PartNotResolvedError(PartFault.ANOTHER_DATASET, part=part)
    named = PartNotResolvedError(PartFault.ANOTHER_DATASET, part=part, answered="spec_x")
    assert stale_reason(unnamed) == "Search now resolves it to another dataset"
    assert stale_reason(named) == "Search now resolves it to another dataset (spec_x)"
