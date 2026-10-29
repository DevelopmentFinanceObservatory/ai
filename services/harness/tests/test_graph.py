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
import copy
import json
from collections.abc import Awaitable, Callable, Iterator, Mapping
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar, NamedTuple, cast, override

import pytest
from analysis_mcp.server import mcp as analysis_mcp
from common.materialisation import FrameRef
from compiler.evidence import check_evidence, evidence_from_source
from compiler.surface import (
    FixtureSurface,
    McpSurface,
    ProgressCallback,
    ProgressReport,
    ToolCallResult,
    ToolSchema,
    ToolSurface,
)
from data_mcp.definition_store import InMemoryDefinitionStore
from data_mcp.definition_store.identity import canonicalize, spec_id_for
from data_mcp.definition_writes import windowed_identity
from data_mcp.search_result import SEARCH_ENVELOPE_META_KEY
from data_mcp.server import mcp as data_mcp
from executor.classify import ToolFailure
from executor.completion import FramePublished
from executor.ledger import InMemoryRunLedger
from executor.outcomes import (
    ISSUE_REMEDIES,
    Accepted,
    EvidenceRecord,
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
from harness import scripted
from harness.chart_binding import (
    LINE_CHART_MAX_VALUES,
    covers_every_row,
    every_series_selects,
    over_cap_dimension,
    select_binding_ok,
)
from harness.graph import answer as answer_module
from harness.graph import switch as switch_module
from harness.graph.execute import (
    _failed_step,
    _mark_masked_step_failed,
    _planner_refusal,
    _replan_refusal,
    _start_relayed,
    record_served_scopes,
)
from harness.graph.plan_done import frame_unavailable_reason, plan_done_node
from harness.graph.state import (
    APPROVE_INTERRUPT,
    ORCHESTRATOR_APPLIED_KEY,
    ORCHESTRATOR_FRAMES_KEY,
    ORCHESTRATOR_KEPT_ASKS_KEY,
    ORCHESTRATOR_PLAN_RECORDS_KEY,
    ORCHESTRATOR_PRIOR_LEDGER_KEY,
    ORCHESTRATOR_SEARCHES_KEY,
    ORCHESTRATOR_SPECS_KEY,
    ORCHESTRATOR_UNAVAILABLE_KEY,
    PLANNER_STARTED_KEY,
    STEP_PROGRESS_KEY,
    UNREADABLE_PLAN_NOTE,
    UNREADABLE_PLAN_UNPLANNED_NOTE,
    GraphState,
    frame_id_for,
    frame_store_ref,
)
from harness.graph.switch import (
    SCOPE_CHOICE_NOT_APPLIED,
    SWITCH_CANCELLED,
    SWITCH_STOP,
    SwitchFailedError,
)
from harness.plan_display import choice_id as row_choice_id
from harness.planner_seam import PlannerSeam, store_aligned_readings
from harness.scope_choices import PLANNER_AXES_BLOCK_KEY, PartScope
from harness.scripted import ScriptedPlannerModel, ScriptedSurface
from harness.scripted_plan import scripted_fetch_series_stats
from harness.seeds import stale_reason
from harness.settings import Settings
from harness.studio_translate import _node_name
from harness.tool_contracts import HarnessCode, reading_facts
from harness.turn_records import CHIP_STALE_NOTE, CompiledFrom, SearchReading, basis_line
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from planner.operations import AggregateOperation, GrowthOperation, RetrieveOperation, Translation
from planner.plan import (
    ASKS_STATE_KEY,
    WINDOWED_PINS_STATE_KEY,
    AggregateRequirement,
    AlignedPin,
    Coverage,
    DatasetRequirement,
    Decision,
    Entity,
    GrowthRequirement,
    Period,
    PinSource,
    PlanStatus,
    ResolvedBy,
    SessionState,
    WindowedPin,
    authored,
    open_window,
    plan_revision,
    repinned,
    save_aligned_pin,
    scope_choice_slot,
    windowed_pins,
    with_authored_requirements,
)
from planner.plan import Plan as PlannerPlan
from planner.records import (
    EvidenceIssue,
    EvidenceIssueCode,
    GroupBy,
    Reduction,
    RelativePeriod,
    RollingWindow,
    Severity,
)
from planner.workflow import EvidenceBinding, SelectedSource, ToolId, Workflow, WorkflowStep
from pydantic import ValidationError
from submission.service import (
    scope_choice_decisions,
)
from translator.discovery import RESOLVED_STATE_PREFIX, PartFault, PartNotResolvedError

from .data_mcp_definitions import DataMcpDefinitions, SearchStub
from .orchestrator_fakes import (
    ScriptLlm,
    approve_answer,
    call,
    create_request_input_response,
    frame_for,
    handed_back,
    open_session,
    orchestrator_runner,
    plan_call,
    request_text,
    run_turn,
    visited_nodes,
)
from .share_plan import ALL_DONORS, DAC_DONORS, US_DONOR, us_share_of_kenya_oda
from .split_plan import FIRST_YEAR, SPEC_DEBT_HISTORY, split_draft, stored_split_plan
from .test_turn_records import _example_part, _manifest, _served_identity, _switchable_scope

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


# --- execute's served scopes ---------------------------------------------------


class _ManifestSurface:
    """Answers `retrieval_manifest` with `manifest` for the one frame it holds."""

    def __init__(self, frame: FrameRef, manifest: dict[str, Any]) -> None:
        self.frame = frame
        self.manifest = manifest

    def retrieval_manifest(self, *, frame_ref: Mapping[str, str | None]) -> dict[str, Any] | None:
        return self.manifest if frame_ref == self.frame.model_dump() else None


def test_a_runs_retrieval_keeps_how_it_was_served_on_the_turns_scoped_reading() -> None:
    """A plan step's retrieval is kept on the reading of its spec_id with
    scope choices, as a fetch of it would be; a reading without any, and
    state no retrieval touched, are left as they are."""
    scoped = SearchReading.from_part({**_example_part(), "spec_id": SPEC_A})
    other = SearchReading(text="life", status="definite", spec_id="spec_" + "c" * 32)
    frame = FrameRef(spec_id=SPEC_A, content_version="cv_" + "1" * 32)
    outcome = Accepted(
        revision="rev",
        evidence=(
            EvidenceRecord(
                requirement_id="aid",
                step_id="aid_op",
                frame_ref=frame,
                spec_id=SPEC_A,
                stat_vars=(),
                window=None,
                gaps=(),
                citation="CRS",
            ),
        ),
        steps=(),
        results={},
    )
    surface = _ManifestSurface(frame, _manifest(_served_identity(residuals=False)))
    state: dict[str, Any] = {
        ORCHESTRATOR_SEARCHES_KEY: [r.model_dump(mode="json") for r in (scoped, other)]
    }
    untouched: dict[str, Any] = {ORCHESTRATOR_SEARCHES_KEY: [other.model_dump(mode="json")]}

    record_served_scopes(cast("Any", state), cast("Any", surface), outcome)
    record_served_scopes(cast("Any", untouched), cast("Any", surface), outcome)

    kept, unchanged = (SearchReading.model_validate(r) for r in state[ORCHESTRATOR_SEARCHES_KEY])
    assert kept.fetch is not None and kept.fetch.served is not None
    assert kept.fetch.served.residuals("destination") == frozenset()
    assert unchanged == other
    assert untouched == {ORCHESTRATOR_SEARCHES_KEY: [other.model_dump(mode="json")]}
    assert basis_line(state[ORCHESTRATOR_SEARCHES_KEY], specs={}, plans=[]) == (
        "Counted as: health (CRS 120) and population and reproductive health (CRS 130) · "
        "Africa's countries only · "
        "gross disbursements, current USD."
    )


def test_a_windowed_copys_retrieval_is_kept_on_the_reading_it_was_copied_from() -> None:
    """A search over 2024 found the part; the plan reads 2015 to 2024 from
    its windowed copy. The copy's retrieval is kept on the search's reading,
    and the plan's source names the reading it was copied from, so the
    basis line states how the copy's figures were counted."""
    copy_id = "spec_" + "d" * 32
    searched = SearchReading.from_part({**_example_part(), "spec_id": SPEC_A})
    frame = FrameRef(spec_id=copy_id, content_version="cv_" + "1" * 32)
    outcome = Accepted(
        revision="rev",
        evidence=(
            EvidenceRecord(
                requirement_id="aid",
                step_id="aid_op",
                frame_ref=frame,
                spec_id=copy_id,
                stat_vars=(),
                window=None,
                gaps=(),
                citation="CRS",
            ),
        ),
        steps=(),
        results={},
    )
    surface = _ManifestSurface(frame, _manifest(_served_identity(residuals=False), spec_id=copy_id))
    pin = WindowedPin(spec_id=copy_id, identity={}, windowed_from=SPEC_A)
    state: dict[str, Any] = {
        ORCHESTRATOR_SEARCHES_KEY: [searched.model_dump(mode="json")],
        WINDOWED_PINS_STATE_KEY: [pin.model_dump(mode="json")],
    }

    record_served_scopes(cast("Any", state), cast("Any", surface), outcome)

    plan = {"sources": [{"spec_id": copy_id, "windowed_from": SPEC_A}]}
    assert basis_line(state[ORCHESTRATOR_SEARCHES_KEY], specs={}, plans=[plan]) == (
        "Counted as: health (CRS 120) and population and reproductive health (CRS 130) · "
        "Africa's countries only · "
        "gross disbursements, current USD."
    )


# --- execute's run: progress relay and cancellation ---------------------------


_PREPARING = "Preparing the dataset (in progress, 3s)"
_RECEIVED = "Result received; checking it"


def _step_progress(event: Event) -> dict[str, Any] | None:
    return (event.custom_metadata or {}).get(STEP_PROGRESS_KEY)


class _ReportsThenWaits(ScriptedSurface):
    """`ScriptedSurface`, whose `get_data` sends one progress report and then
    waits for `release` before answering, standing in for data-mcp polling
    a background apis operation."""

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
        APPROVE_INTERRUPT, {"result": approve_answer(_output_of(events, "planner")["revision"])}
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


@pytest.mark.parametrize(
    "wrong",
    [
        lambda revision: approve_answer("not-a-real-revision"),
        lambda revision: {"approved": revision},
        lambda revision: {"approved": revision, "at": "2026-10-01T09:30:00"},
        lambda revision: {"approved": revision, "at": "yesterday"},
    ],
    ids=["stale_revision", "no_time", "time_without_offset", "unreadable_time"],
)
async def test_stale_version_approval_routes_to_amend(wrong: Callable[[str], Any]) -> None:
    """A wrong answer, one naming another revision or stating no
    timezone-aware approval time, routes to amend; `ScriptedPlannerModel`
    redrafts the same canned steps as the next version, bumping `version`
    and `previous_version` the way any `PlannerModel` seam must."""
    surface = _CallRecordingSurface(ScriptedSurface())
    runner, _ = _plan_runner(ScriptedPlannerModel(), tool_surface=surface)
    events = await _start(runner)
    assert _pending_interrupt(events) == APPROVE_INTERRUPT
    first = _output_of(events, "planner")

    events = await _resume(runner, APPROVE_INTERRUPT, wrong(first["revision"]))

    # Wrong answer: nothing executed; the planner redrafted to version 2 and
    # a fresh approval interrupt is pending.
    assert surface.calls == []
    assert _pending_interrupt(events) == APPROVE_INTERRUPT
    amended = _output_of(events, "planner")
    assert amended["version"] == 2
    assert amended["previous_version"] == 1
    assert amended["status"] == "pending_approval"
    assert amended["revision"] != first["revision"]

    events = await _resume(runner, APPROVE_INTERRUPT, approve_answer(amended["revision"]))
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
        events = await _resume(runner, APPROVE_INTERRUPT, approve_answer(plan_payload["revision"]))

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
    assert frame_unavailable_reason({"retry_mode": retry_mode}) == reason


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


def _stopping_at_growth_surface(real_schemas: list[ToolSchema]) -> FixtureSurface:
    """get_data and totals (aggregate) succeed; change (growth) always
    refuses. A run against this surface completes a retrieval and a derived
    step before exhausting the replan budget."""
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


@pytest.mark.parametrize(
    "build_plan",
    [
        # totals feeds change, the one terminal step.
        _seam_plan_with_intermediate_step,
        # totals is a terminal step of its own, finished beside the failed change.
        _seam_plan_with_two_terminal_steps,
    ],
    ids=["intermediate", "sibling_terminal"],
)
async def test_a_stopped_plan_hands_back_no_derived_steps_figures(
    real_schemas: list[ToolSchema], build_plan: Callable[..., PlannerPlan]
) -> None:
    """change fails past the replan budget, leaving get_data's and totals'
    frames in `completed`. totals' derived frame is never published on a
    stopped run, so neither its frame nor its digest reaches the agent,
    whether totals fed change or was a terminal step of its own; the
    retrieval's digest still does, and `plan_done` lists no frame to chart."""
    plan = build_plan(schemas=real_schemas)
    seam = _Seam([{"plan": plan.model_dump(mode="json"), "summary": {}}])
    executor = ScriptedExecutor()
    runner, model = _plan_runner(
        seam, tool_surface=_stopping_at_growth_surface(real_schemas), executor=executor
    )
    await _start(runner)

    events = await _resume(runner, APPROVE_INTERRUPT, approve_answer(plan_revision(plan)))
    assert _pending_interrupt(events) == APPROVE_INTERRUPT
    replanned = _output_of(events, "planner")
    events = await _resume(runner, APPROVE_INTERRUPT, approve_answer(replanned["revision"]))

    executed = _output_of(events, "execute")
    assert executed["failed_step"]["step_id"] == "change"
    assert [step["step_id"] for step in executed["frame_ref_steps"]] == ["get_data"]
    assert [step["digest"] for step in executed["step_summaries"]] == ["fetched"]
    assert executor.calls == []
    _, body = handed_back(model)
    assert body["frames"] == []
    assert body["step_summaries"] == ["fetched"]


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

    events = await _resume(runner, APPROVE_INTERRUPT, approve_answer(plan_revision(plan)))

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
    """The population frame's view: one `stat_var`, seven countries, and
    a `unit` column holding `USDollar` on some rows and null on others, which
    apis reports as plural with the one non-null value listed. `plural`
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
    """apis refuses an unbound facet only when it splits a series, so a
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
    """apis refuses a component past 24 series even though the catalogue
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
    assert reason["code"] == "series_limit_exceeded"
    assert "estimated 25 series" in reason["message"]


def _counted_view(**total_counts: int) -> dict[str, Any]:
    """A `describe_frame` view whose dimensions carry the apis service's `total_count`."""
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
    """The harness computes the apis service's own estimate (`_check_series_estimate`
    in dfo-data-tools' `assembler/resolve.py`): over every coordinate
    dimension, the bound array's length, 1 for a single bound value, and
    `max(total_count, 1)` for an unbound dimension, on every component type
    the assembler resolves. Each case here is one apis refuses with
    `MAX_SERIES_PER_COMPONENT`: an unbound plural dimension, a `line_chart`
    array past 24, and an unbound dimension multiplying a table's array.
    An unbound facet column counts 1 and a bound one counts its values."""
    reason = select_binding_ok(type_=type_, select=select, view=view)
    assert reason is not None
    assert reason["message"].startswith(
        f"{type_} select expands to an estimated {estimate} series, "
        "over the platform's maximum of 24"
    )


def test_over_cap_dimension_names_the_one_dimension_that_multiplies() -> None:
    view = _counted_view(stat_var=1, financingProvider=86, financingRecipient=1)
    wide = _counted_view(stat_var=5, financingProvider=86, financingRecipient=1)

    assert over_cap_dimension(select={}, view=view) == ("financingProvider", 86)
    assert over_cap_dimension(
        select={"financingProvider": ["a", "b"], "stat_var": "x"}, view=wide
    ) == ("financingProvider", 2)
    assert over_cap_dimension(select={}, view=wide) is None


@pytest.mark.parametrize(
    ("select", "view", "retry_mode", "named", "charts"),
    [
        pytest.param(
            {},
            _counted_view(stat_var=1, financingRecipient=6, financingProvider=51),
            "correct_and_retry",
            ["financingRecipient (6 values)", "financingProvider (51 values)"],
            "each financingRecipient value is 6 charts, which fits",
            id="smaller_fits_one_call",
        ),
        pytest.param(
            {"financingRecipient": _countries(12)},
            _counted_view(stat_var=1, financingRecipient=40, financingProvider=51),
            "correct_and_retry",
            ["financingRecipient (12 values)", "financingProvider (51 values)"],
            "each financingRecipient value is 12 charts, more than the 8",
            id="smaller_past_one_call",
        ),
        pytest.param(
            {},
            _counted_view(stat_var=2, financingRecipient=3, financingProvider=51),
            "correct_and_retry",
            ["stat_var (2 values)", "financingRecipient (3 values)", "financingProvider (51"],
            "each combination of stat_var and financingRecipient values is 6 charts, which fits",
            id="three_dimensions",
        ),
        pytest.param(
            {},
            _counted_view(stat_var=1, financingRecipient=1, financingProvider=51),
            "do_not_retry",
            [],
            "",
            id="one_dimension",
        ),
    ],
)
def test_a_select_over_the_cap_on_several_dimensions_is_told_what_draws(
    select: dict[str, Any], view: dict[str, Any], retry_mode: str, named: list[str], charts: str
) -> None:
    """Where two or more dimensions multiply past the cap, the refusal names
    each with the values it contributes, counts the charts that bind all but
    the widest, and says whether one call draws them. One dimension over the
    cap keeps the bare estimate: `make_charts` draws its largest members."""
    refusal = select_binding_ok(type_="table", select=select, view=view)

    assert refusal is not None
    assert refusal["code"] == "series_limit_exceeded"
    assert refusal["retry_mode"] == retry_mode
    assert all(name in refusal["message"] for name in named)
    assert charts in refusal["message"]
    assert ("multiply" in refusal["message"]) == bool(named)


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
    """apis estimates series over its own dimensions only, and refuses a
    select binding any other name as an invalid binding; the harness gives
    the same reason."""
    select = {"stat_var": "a", "donor": [f"donor/{index}" for index in range(30)]}
    reason = select_binding_ok(type_="bar_chart", select=select, view=_counted_view(stat_var=1))
    assert reason is not None
    assert (reason["code"], reason["retry_mode"]) == ("select_invalid", "correct_and_retry")
    assert reason["fix"] == {"dimensions": ["stat_var"]}
    flow = {"dimensions": ["stat_var", "financingProvider", "financingRecipient"]}
    reason = select_binding_ok(type_="table", select={"observationAbout": "country/MWI"}, view=flow)
    assert reason is not None
    assert reason["fix"] == {"dimensions": ["stat_var", "financingProvider", "financingRecipient"]}


@pytest.mark.parametrize(
    ("type_", "view", "selects"),
    [
        ("stat_card", {"dimensions": [{"name": "donor", "values": ["a"]}]}, [{"donor": "a"}]),
        # A dimension of one value before one of several.
        (
            "bar_chart",
            {
                "dimensions": [
                    {"name": "recipient", "values": ["k", "u"], "total_count": 2},
                    {"name": "donor", "values": ["a"], "total_count": 1},
                ]
            },
            [{"donor": "a"}, {"recipient": ["k", "u"]}],
        ),
        # A view that lists fewer values than the dimension holds names no series set.
        (
            "table",
            {"dimensions": [{"name": "donor", "values": ["a", "b"], "total_count": 3}]},
            [],
        ),
        # A facet may hold rows with no value, which no select on it names.
        ("table", {"dimensions": [{"name": "unit", "values": ["USD"], "total_count": 1}]}, []),
        # A select the binding check refuses is not offered.
        (
            "line_chart",
            {"dimensions": [{"name": "donor", "values": list("abcdefghi"), "total_count": 9}]},
            [],
        ),
    ],
)
def test_every_series_selects_bind_one_dimension_the_view_lists_in_full(
    type_: str, view: dict[str, Any], selects: list[dict[str, Any]]
) -> None:
    assert every_series_selects(type_=type_, view=view) == selects


def test_a_select_covers_every_row_only_when_none_is_empty_on_its_dimension() -> None:
    """The view counts a dimension's values without its empty cells, so only
    the rows say whether a select on it names every series."""
    rows = [{"donor": "a", "recipient": "k"}, {"donor": None, "recipient": "total"}, {"donor": "a"}]
    assert covers_every_row({"recipient": ["k", "total"]}, rows[:2])
    assert not covers_every_row({"donor": "a"}, rows[:2])
    assert not covers_every_row({"recipient": ["k", "total"]}, rows)


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
    events = await _resume(runner, APPROVE_INTERRUPT, approve_answer(99))

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

    events = await _resume(runner, APPROVE_INTERRUPT, approve_answer(1))

    refused = _output_of(events, "planner")
    assert refused == {
        "text": f"{UNREADABLE_PLAN_UNPLANNED_NOTE} The draft budget is spent.",
        "final": True,
        # The scripted refusal states no code.
        "code": None,
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
    events = await _resume(runner, APPROVE_INTERRUPT, approve_answer("not-a-real-revision"))
    assert _pending_interrupt(events) == APPROVE_INTERRUPT
    amended = _output_of(events, "planner")
    assert amended["version"] == 1
    revision_2 = amended["revision"]
    assert revision_2 != revision_1

    # The first card's revision no longer names what's pending: superseded,
    # not approved -- nothing executes and a fresh approval is still pending.
    events = await _resume(runner, APPROVE_INTERRUPT, approve_answer(revision_1))
    assert executor.calls == []
    assert _pending_interrupt(events) == APPROVE_INTERRUPT
    assert _output_of(events, "planner")["revision"] == revision_2

    # The correct, current revision approves cleanly: `approve()` succeeds,
    # the gate is passed (no fresh approval interrupt), and the approved
    # workflow's steps ran through `execute_approved_plan` against the
    # tool surface: the executor only describes the final frame.
    events = await _resume(runner, APPROVE_INTERRUPT, approve_answer(revision_2))
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

    events = await _resume(runner, APPROVE_INTERRUPT, approve_answer(_BASE_FORMAT_PLAN["revision"]))

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
        | {
            "status": "approved",
            "approved_revision": _BASE_FORMAT_PLAN["revision"],
            "approved_at": datetime.now(UTC),
        }
    )

    async def published(frame: FrameRef) -> FramePublished:
        return FramePublished(replay_key="rpk_" + "e" * 64)

    finished, outcome = await execute_approved_plan(
        approved,
        surface=ScriptedSurface(),
        publisher=published,
        fetch_series_stats=scripted_fetch_series_stats,
    )

    assert isinstance(outcome, Accepted), outcome
    assert finished.status is PlanStatus.COMPLETED


async def test_a_saved_plan_whose_retrieval_posts_its_spec_id_alone_runs_as_saved() -> None:
    """The saved retrieval step carries `spec_id` alone, as `get_data` takes
    it, so the call is made as saved."""
    approved = PlannerPlan.model_validate(
        _BASE_FORMAT_PLAN["plan"]
        | {
            "status": "approved",
            "approved_revision": _BASE_FORMAT_PLAN["revision"],
            "approved_at": datetime.now(UTC),
        }
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
        saved
        | {
            "status": "approved",
            "approved_revision": _BASE_FORMAT_LIVE_PLAN["revision"],
            "approved_at": datetime.now(UTC),
        }
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

    events = await _resume(runner, APPROVE_INTERRUPT, approve_answer("not-a-real-revision"))

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
    events = await _resume(runner, APPROVE_INTERRUPT, approve_answer(revision_1))
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

    events = await _resume(runner, APPROVE_INTERRUPT, approve_answer(revision))

    assert [call[0] for call in surface.calls] == ["get_data", "aggregate"]
    assert [call["tool"] for call in executor.calls] == ["describe_frame"]
    executed = _output_of(events, "execute")
    # `Accepted.results` is terminal-steps-only: one frame ref
    # for "totals", not one per step.
    assert executed["frame_refs"] == [
        {"spec_id": "ops_" + "2" * 32, "content_version": "ocv_" + "3" * 32}
    ]
    assert [(step["tool"], step["digest"]) for step in executed["step_summaries"]] == [
        ("get_data", "fetched"),
        ("aggregate", "computed"),
    ]
    assert executed["plan_version"] == 1
    assert _output_of(events, "execute")["plan_version"] == 1


class _ServedPairsPlatform:
    """presentation-mcp over apis, which serves a frame under the pair it
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
    own once read with extra columns, and apis serves only its manifest's pair.
    `execute` hands on the manifest's pair, so `plan_done` describes the series
    and registers it for charting, while the total reading it was given the
    frame's own pair. Each frame is registered with the pair the store holds
    it under, which is the typed pair for the retrieval and its own pair for
    the total."""
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

    events = await _resume(runner, APPROVE_INTERRUPT, approve_answer(plan_revision(plan)))

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
    frames = _session_state(runner)[ORCHESTRATOR_FRAMES_KEY]
    assert [(frame["ref"], frame["store_ref"]) for frame in frames.values()] == [
        (served, typed),
        (total, total),
    ]


_FINAL_RETRIEVAL = {"step_id": "get_data", "tool": "get_data"}


@pytest.mark.parametrize(
    ("state", "node_input"),
    [
        (
            {},
            {
                "frame_refs": [REF_A],
                "frame_ref_steps": [_FINAL_RETRIEVAL | {"is_final_step": True}],
            },
        ),
        (
            {
                ASKS_STATE_KEY: ["Kenya's debt service"],
                ORCHESTRATOR_KEPT_ASKS_KEY: {
                    "0": {
                        "steps": [_FINAL_RETRIEVAL | {"frame_ref": REF_A, "digest": "fetched"}],
                        "totals": [],
                    }
                },
            },
            {"text": "no plan avoids the failure", "final": True},
        ),
    ],
    ids=["execute-event", "kept-ask"],
)
async def test_a_hand_back_recorded_without_store_pairs_registers_its_frames(
    state: dict[str, Any], node_input: dict[str, Any]
) -> None:
    """A hand-back event or kept ask recorded without store pairs names each
    frame by the pair apis serves alone. Its frame is registered without
    a store pair, so the store is opened by the frame's own pair."""
    ctx = SimpleNamespace(state=dict(state))
    plan_done = plan_done_node(executor=ScriptedExecutor(), run_ledger=InMemoryRunLedger())

    [_ async for _ in plan_done(cast("Any", ctx), node_input)]

    assert ctx.state[ORCHESTRATOR_FRAMES_KEY] == {frame_id_for(REF_A): {"ref": REF_A, "view": {}}}
    entry = ctx.state[ORCHESTRATOR_FRAMES_KEY][frame_id_for(REF_A)]
    assert frame_store_ref(entry) == REF_A


@pytest.mark.parametrize(
    ("whole", "named"),
    [(DAC_DONORS, "from official donors (DAC)"), (ALL_DONORS, "from every donor")],
    ids=["dac_donors", "all_donors"],
)
async def test_a_share_frame_is_registered_with_the_whole_its_plan_divides_by(
    decision: Callable[..., Decision], whole: dict[str, Any], named: str
) -> None:
    """The final frame of a share step carries the whole the approved plan's
    denominator source holds, which `make_charts` writes into the card's title."""
    plan = us_share_of_kenya_oda(decision, whole=whole, numerator=US_DONOR)
    ctx = SimpleNamespace(state={"plan": plan.model_dump_json()})
    plan_done = plan_done_node(executor=ScriptedExecutor(), run_ledger=InMemoryRunLedger())
    node_input = {
        "frame_refs": [REF_A],
        "frame_ref_steps": [{"step_id": "share", "tool": "share_of", "is_final_step": True}],
    }

    [_ async for _ in plan_done(cast("Any", ctx), node_input)]

    assert ctx.state[ORCHESTRATOR_FRAMES_KEY][frame_id_for(REF_A)]["whole"] == named


async def test_a_summed_familys_total_reaches_the_agent_in_the_hand_back(
    real_schemas: list[ToolSchema], monkeypatch: pytest.MonkeyPatch
) -> None:
    """`execute` hands `plan_done` the sentences `_stat_var_totals` writes,
    and `plan_done` gives each to the agent under `totals`."""
    sentence = (
        'By default, the figures for "OECD Creditor Reporting System (CRS)" are the '
        "sum of 74 indicators."
    )
    monkeypatch.setattr(answer_module, "_stat_var_totals", lambda *a, **k: [sentence])
    plan = _seam_plan(schemas=real_schemas)
    seam = _Seam([{"plan": plan.model_dump(mode="json"), "summary": {}}])
    runner, model = _plan_runner(seam, tool_surface=_fixture_surface(real_schemas))
    await _start(runner)

    events = await _resume(runner, APPROVE_INTERRUPT, approve_answer(plan_revision(plan)))

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
        await _resume(runner, APPROVE_INTERRUPT, approve_answer(revision))


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

    events = await _resume(runner, APPROVE_INTERRUPT, approve_answer(revision))

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

    events = await _resume(runner, APPROVE_INTERRUPT, approve_answer(revision))
    assert _pending_interrupt(events) == APPROVE_INTERRUPT
    replanned = _output_of(events, "planner")
    events = await _resume(runner, APPROVE_INTERRUPT, approve_answer(replanned["revision"]))

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
    assert [step["digest"] for step in executed["step_summaries"]] == ["fetched"]
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

    events = await _resume(runner, APPROVE_INTERRUPT, approve_answer(revision))

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

    events = await _resume(runner, APPROVE_INTERRUPT, approve_answer(revision))

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

    events = await _resume(runner, APPROVE_INTERRUPT, approve_answer(plan_revision(plan)))

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

    events = await _resume(runner, APPROVE_INTERRUPT, approve_answer(plan_revision(plan)))

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

    events = await _resume(runner, APPROVE_INTERRUPT, approve_answer(revision))

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
        await _resume(runner, APPROVE_INTERRUPT, approve_answer(revision))


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
    failed = _failed_step(refused)
    assert (failed["step_id"], failed["tool"], failed["code"]) == ("totals", None, "empty_answer")
    assert "no rows matched" not in failed["message"]

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

    # A stop with every donor named missing. The failed step carries what the
    # user can do about it, since the code alone says only that the names could
    # not be resolved.
    refused_names_missing = Refused(
        revision="rev1",
        refusal=RefusalKind.REQUIREMENT_UNMET,
        detail="requirement 'donor_oda' is not met by what 'get_data' returned",
        step_id="donor_oda",
        requirement_id="donor_oda",
        issues=tuple(
            EvidenceIssue(
                code=EvidenceIssueCode.ENTITY_MISSING,
                severity=Severity.ERROR,
                requirement_id="donor_oda",
                detail=f"the source resolves no entity named {name!r}",
                required=name,
            )
            for name in ("United States", "Germany")
        ),
    )
    names_missing = _failed_step(refused_names_missing)
    assert names_missing.pop("remedy") == ISSUE_REMEDIES[EvidenceIssueCode.ENTITY_MISSING]
    assert names_missing == {
        "step_id": "donor_oda",
        "tool": None,
        "code": "entity_missing",
        "message": "the source resolves no entity named 'United States'; the source "
        "resolves no entity named 'Germany'",
    }


_GET_DATA = ToolId(server="data-mcp", name="get_data")


def _with_a_second_rolling_reader(decision: Callable[..., Decision]) -> PlannerPlan:
    """`stored_split_plan` with `last_three`, a growth over the newest three
    years, reading `debt` too, so the split has it read `debt_history`."""
    plan = stored_split_plan(decision)
    window = RollingWindow(
        window_start=RelativePeriod(offset=-3), window_end=RelativePeriod(offset=0)
    )
    last_three = GrowthRequirement(
        id="last_three", input_id="debt", window_start=2021, window_end=2024, rolling=window
    )
    draft = split_draft(decision)
    split = with_authored_requirements(draft, [*authored(draft).evidence_requirements, last_three])
    assert split.requirement("last_three").input_id == "debt_history"  # type: ignore[union-attr]
    assert plan.translation is not None and plan.workflow is not None
    (growth,) = [step for step in plan.workflow.steps if step.id == "last_five"]
    return split.with_translation(
        translation=plan.translation.model_copy(
            update={
                "operations": (
                    *plan.translation.operations,
                    GrowthOperation(
                        id="last_three",
                        requirement_id="last_three",
                        input="debt_history",
                        window_start=2021,
                        window_end=2024,
                        rolling=window,
                    ),
                ),
                "terminal": (*plan.translation.terminal, "last_three"),
            }
        ),
        workflow=plan.workflow.model_copy(
            update={
                "steps": (
                    *plan.workflow.steps,
                    growth.model_copy(update={"id": "last_three", "operation_id": "last_three"}),
                ),
                "terminal_steps": (*plan.workflow.terminal_steps, "last_three"),
            }
        ),
    )


_ROLLED_START = Refused(
    revision="rev1",
    refusal=RefusalKind.ROLLING_WINDOW_OUTSIDE_DATASET,
    detail=(
        "step 'last_five' would start in 1995, counted from 2023, before 2000, "
        f"the first year {SPEC_DEBT_HISTORY} holds"
    ),
    step_id="last_five",
    requirement_id="debt_history",
    payload=ToolFailure(
        code="rolling_window_outside_dataset",
        message="The data for external debt service paid starts in 2000.",
        retry_mode="do_not_retry",
    ),
)
_REDISCOVERY = RediscoveryNeeded(
    revision="rev1",
    step_id="debt_history",
    tool=_GET_DATA,
    requirement_id="debt_history",
    spec_id=SPEC_DEBT_HISTORY,
    refusal=ToolFailure(
        code="upstream_version_changed", message="", retry_mode="refresh_then_retry"
    ),
    completed=(),
)
_CAPABILITY = Refused(
    revision="rev1",
    refusal=RefusalKind.STEP_REFUSED,
    detail="'get_data' refused: capability_unavailable",
    step_id="debt_history",
    requirement_id="debt_history",
    payload=ToolFailure(
        code="capability_unavailable",
        message="this deployment cannot read debt_history's window",
        retry_mode="do_not_retry",
    ),
)


@pytest.mark.parametrize(
    ("stop", "second_reader", "ids", "failed", "unavailable"),
    [
        pytest.param(
            _ROLLED_START,
            False,
            ("last_five", "debt"),
            {"last_five": "last_five"},
            {},
            id="rolled-start-before-the-copy",
        ),
        pytest.param(
            _ROLLED_START,
            True,
            ("last_five", "debt"),
            {"last_five": "last_five"},
            {},
            id="rolled-start-beside-a-second-reader",
        ),
        pytest.param(
            _REDISCOVERY,
            False,
            ("debt", "debt"),
            {"last_five": "last_five"},
            {},
            id="copy-retrieval-needs-rediscovery",
        ),
        pytest.param(
            _REDISCOVERY,
            True,
            ("debt", "debt"),
            {"last_five": "last_five", "last_three": "last_three"},
            {},
            id="copy-retrieval-read-by-two-rolling-steps",
        ),
        pytest.param(
            _CAPABILITY,
            False,
            ("debt", "debt"),
            {"last_five": "last_five"},
            {SPEC_DEBT_HISTORY: "this deployment cannot read debt's window"},
            id="copy-retrieval-capability-unavailable",
        ),
    ],
)
async def test_a_stop_on_a_copy_names_the_requirement_the_model_wrote(
    decision: Callable[..., Decision],
    stop: Refused | RediscoveryNeeded,
    second_reader: bool,
    ids: tuple[str, str],
    failed: dict[str, str],
    unavailable: dict[str, str],
) -> None:
    """A run stopped on a copy the split added (`debt_history`) names no copy.
    The planner reads the step and requirement the model wrote (`debt`). The
    agent and the plan record read the stop as landing on each step the model
    wrote that reads the copy, `last_five` and, where it is there, `last_three`:
    `debt` ran on its own pin and feeds `total`. A stop on `last_five` itself
    lands on it alone. A capability the copy's retrieval lacks marks the
    copy's own pin unavailable, and leaves `debt`'s."""
    plan = stored_split_plan(decision)
    readers = ["last_five"]
    if second_reader:
        plan = _with_a_second_rolling_reader(decision)
        readers.append("last_three")
    plan = PlannerPlan.model_validate(
        {
            **plan.model_dump(),
            "asks": ["debt service each year", *(f"its change, {step}" for step in readers)],
            "answered": [
                {"ask": 0, "outputs": ["total"]},
                *({"ask": n, "outputs": [step]} for n, step in enumerate(readers, start=1)),
            ],
        }
    )

    refusal = _planner_refusal(stop, plan=plan)
    ctx = SimpleNamespace(state={"plan": plan.model_dump_json()})
    plan_done = plan_done_node(executor=ScriptedExecutor(), run_ledger=InMemoryRunLedger())
    node_input = {"frame_refs": [], "frame_ref_steps": [], "failed_step": _failed_step(stop)}
    (handed,) = [event async for event in plan_done(cast("Any", ctx), node_input)]

    assert (refusal["step_id"], refusal["requirement_id"]) == ids
    assert "debt_history" not in json.dumps(refusal)
    assert "debt_history" not in json.dumps(handed)
    (record,) = ctx.state[ORCHESTRATOR_PLAN_RECORDS_KEY]
    assert [step["id"] for step in record["steps"] if step["failed"]] == list(
        dict.fromkeys(failed.values())
    )
    result = json.loads(handed["text"].split("\n")[1])
    assert result["failed_step"]["step_id"] == "last_five"
    assert [
        (entry["step"], entry["status"], entry.get("failed_at"))
        for ask in result["asks"]
        for entry in ask["answered_by"]
    ] == [
        ("total", "not_run", None),
        *(
            (step, "failed", failed[step]) if step in failed else (step, "not_run", None)
            for step in readers
        ),
    ]
    assert ctx.state[ORCHESTRATOR_UNAVAILABLE_KEY] == unavailable


def test_a_stop_on_a_copy_renames_it_only_where_a_field_names_a_step(
    decision: Callable[..., Decision],
) -> None:
    """The planner reads `debt` for the copy in the step, the requirement and
    the tool's message. A ref the tool's `fix` holds keeps the text it has,
    though it ends in the copy's id."""
    plan = stored_split_plan(decision)
    stop = RediscoveryNeeded(
        revision="rev1",
        step_id="debt_history",
        tool=_GET_DATA,
        requirement_id="debt_history",
        spec_id=SPEC_DEBT_HISTORY,
        refusal=ToolFailure(
            code="upstream_version_changed",
            message="debt_history needs its source found again",
            retry_mode="refresh_then_retry",
            fix={"topic": "debt_history", "path": "dc/topic/debt_history"},
        ),
        completed=(),
    )

    refusal = _planner_refusal(stop, plan=plan)

    assert (refusal["step_id"], refusal["requirement_id"]) == ("debt", "debt")
    assert refusal["refusal"]["message"] == "debt needs its source found again"
    assert refusal["refusal"]["fix"] == {"topic": "debt_history", "path": "dc/topic/debt_history"}


def test_a_window_short_gap_on_a_copy_names_the_origin_and_the_years_its_retrieval_requested(
    decision: Callable[..., Decision], source: Callable[..., SelectedSource]
) -> None:
    """The copy keeps its origin's stated 2015-2024 and is retrieved from the
    dataset's first year. The gap the planner reads names the origin and both."""
    plan = stored_split_plan(decision)
    copy = plan.requirement("debt_history")
    assert isinstance(copy, DatasetRequirement)
    retrieved = source(
        spec_id=SPEC_DEBT_HISTORY,
        coverage_window=Period(start=FIRST_YEAR, end=2023),
        identity={"stat_var": "DebtServicePaid", "requested_window": open_window(FIRST_YEAR)},
    )
    findings = check_evidence(
        copy.model_copy(update={"coverage": Coverage(on_missing="fail")}),
        evidence_from_source(retrieved),
    )
    stop = Refused(
        revision="rev1",
        refusal=RefusalKind.REQUIREMENT_UNMET,
        detail="requirement 'debt_history' is not met by what 'get_data' returned",
        step_id="debt_history",
        requirement_id="debt_history",
        issues=tuple(findings),
    )

    refusal = _planner_refusal(stop, plan=plan)

    (gap,) = [issue for issue in refusal["issues"] if issue["code"] == "window_short"]
    # A finding naming no series reads as it did before findings could name one.
    assert set(gap) == {"code", "severity", "requirement_id", "detail", "found", "required"}
    assert gap["requirement_id"] == "debt"
    assert gap["detail"] == (
        "the source covers 2000-2023, the requirement asks for 2015-2024, "
        "and its retrieval requests every year from 2000"
    )
    assert "debt_history" not in json.dumps(refusal)


@pytest.mark.parametrize("kind", list(RefusalKind))
def test_a_stop_no_tool_refused_reaches_the_orchestrator_by_its_kind_alone(
    kind: RefusalKind,
) -> None:
    """A run's own stop carries the ledger's failure reason or an exception's
    text in `detail`, which can name literal spec ids: the orchestrator is told
    the stop's code and a sentence for its kind, never the detail."""
    detail = f"the run failed before it came to an outcome: store {'ops_' + 'd' * 32} refused"
    refused = Refused(revision="rev1", refusal=kind, detail=detail, run_id="run_1")

    failed = _failed_step(refused)

    assert failed["code"] == kind.value
    assert failed["message"]
    assert "ops_" not in json.dumps(failed)


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


# --- the switch node: a scope choice moved on the plan card, in code ----------


def _hashed_scope() -> dict[str, Any]:
    """US health aid to Africa's scope (`_switchable_scope`), its base under the
    spec_id its identity hashes to, so data-mcp takes it as posted: health as
    CRS 120 alone from the user's words, Africa's countries only carried,
    gross disbursements by default."""
    scope = _switchable_scope()
    base = spec_id_for(canonicalize(scope["base"]["identity"]))
    scope["id"] = scope["base"]["spec_id"] = base
    for choice in scope["choices"]:
        choice["id"] = f"{base}:{choice['axis']}"
    return scope


# The window the plan of `_scoped_plan` reads its dataset over.
_PLAN_WINDOW = {"kind": "range", "start_year": 2015, "end_year": 2024}


def _switched_answer(scope: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """The spec_id and envelope Search answers for `scope` switched to CRS
    120 and 130: grant equivalents become the basis, since gross is out of
    reach beside the switch."""
    switched = copy.deepcopy(scope)
    identity = copy.deepcopy(scope["base"]["identity"])
    identity["slots"] = [s for s in identity["slots"] if s["property"] != "nettingBasis"]
    identity["shape"]["measured_property"] = "grantEquivalent"
    spec_id = spec_id_for(canonicalize(identity))
    sector, _, basis = switched["choices"]
    sector["active"] = "sector:120+130"
    basis["active"] = "measure_basis:grant_equivalent"
    basis["options"][0]["availability"] = {
        "status": "unavailable",
        "code": "conflicts_with",
        "detail": {"entry": "sector:120+130"},
    }
    switched["entries"] = [
        switched["entries"][1],
        {
            "axis": "sector",
            "option_id": "sector:120+130",
            "provenance": "explicit",
            "concept": "health",
            "origin": {"kind": "choice", "choice_id": sector["id"]},
        },
    ]
    switched["compiled"] = {"spec_id": spec_id, "identity": identity, "components": []}
    block = {
        "spec_id": spec_id,
        "sentence": "US health aid to Africa",
        "definition": {"schema_version": 1, "spec_id": spec_id, "identity": identity},
        "resolved": {"scope_choices": switched},
    }
    envelope = {
        "schema_version": 1,
        "status": "definite",
        "parts": [{"status": "definite", "variable_text": scope["id"], "spec": block}],
    }
    return spec_id, envelope


def _scoped_plan(
    pin: str, scope: dict[str, Any], *, schemas: list[ToolSchema], version: int = 1
) -> PlannerPlan:
    """`_seam_plan` with its dataset pinned to `pin`, citing the scope
    decisions a pin of `scope`'s part states."""
    plan = _seam_plan(schemas=schemas)
    block = {"spec_id": pin, "resolved": {"scope_choices": scope}}
    scoped = scope_choice_decisions({pin: PinSource(block=block, settled=True)})[pin]
    return PlannerPlan.model_validate(
        {
            **plan.model_dump(),
            "version": version,
            "previous_version": version - 1 if version > 1 else None,
            "decisions": [*plan.decisions, *scoped],
            "evidence_requirements": [
                r.model_copy(
                    update={
                        "source_spec_id": pin,
                        "decision_ids": (*r.decision_ids, *(d.slot for d in scoped)),
                    }
                )
                if isinstance(r, DatasetRequirement)
                else r
                for r in plan.evidence_requirements
            ],
        }
    )


def _row_id(plan: PlannerPlan, axis: str) -> str:
    """The plan card's id for the row of `plan`'s one dataset's choice on `axis`
    (`harness.plan_display.choice_id`)."""
    (dataset,) = [r for r in plan.evidence_requirements if isinstance(r, DatasetRequirement)]
    return f"{dataset.id}:{axis}"


class _RevisingSeam(_Seam):
    """A `PlannerModel` drafting `plans` and answering each revision with `revised`."""

    def __init__(self, plans: list[dict[str, Any]], revised: dict[str, Any]) -> None:
        super().__init__(plans)
        self._revised = revised
        self.revisions: list[dict[str, Any]] = []

    async def revise(self, **kwargs: Any) -> dict[str, Any]:
        self.revisions.append(kwargs)
        kwargs["ctx"].state["planner:rounds"] = 1
        return self._revised


# The pin of the whole `_StoppedSeam` compiles over every sector of its part.
_WHOLE_PIN = "spec_" + "7" * 32


class _StoppedSeam(_RevisingSeam):
    """A `_RevisingSeam` whose revision the user stops while it runs: it writes
    drafting state, compiles a declared whole over every sector of the switched
    source and stores its reading (`store_aligned_readings`), as a revision of a
    share does, sets `SWITCH_STOP` as a Stop does, and waits to be cancelled."""

    async def revise(self, **kwargs: Any) -> dict[str, Any]:
        self.revisions.append(kwargs)
        ctx = kwargs["ctx"]
        ctx.state["planner:rounds"] = 1
        (origin,) = {
            r.source_spec_id
            for r in authored(
                PlannerPlan.model_validate(kwargs["prior_plan"])
            ).evidence_requirements
            if isinstance(r, DatasetRequirement)
        }
        block = {"spec_id": _WHOLE_PIN, "resolved": {"scope_choices": None}}
        save_aligned_pin(
            cast(SessionState, ctx.state),
            AlignedPin(
                origin=cast(str, origin),
                option_ids=(),
                spec_id=_WHOLE_PIN,
                digest={
                    "part_digests": [
                        {
                            "variable": _WHOLE_PIN,
                            "status": "definite",
                            "spec_id": _WHOLE_PIN,
                            "spec_block": block,
                        }
                    ]
                },
                whole_axis="sector",
            ),
        )
        store_aligned_readings(ctx, cast(SessionState, ctx.state))
        stop = SWITCH_STOP.get()
        assert stop is not None
        stop.set()
        return await self._stopped()

    async def _stopped(self) -> dict[str, Any]:
        """What the revision does once the Stop is set: waits to be cancelled."""
        # Cancelled at once; the bound fails the test, not the suite, if not.
        await asyncio.sleep(5)
        raise AssertionError("a stopped revision is cancelled")


class _FinishingStoppedSeam(_StoppedSeam):
    """A `_StoppedSeam` whose revision finishes in the step the Stop comes in,
    so the switch's wait sees both done."""

    async def _stopped(self) -> dict[str, Any]:
        return self._revised


async def _switch_runner(
    real_schemas: list[ToolSchema],
    search: SearchStub,
    revised: dict[str, Any] | None = None,
    *,
    option_id: str = "sector:120+130",
    aligned: bool = False,
    entity_groups: dict[str, Any] | None = None,
    stored_unreadable: bool = False,
    stored_groups: list[dict[str, Any]] | None = None,
    seam_type: type[_RevisingSeam] = _RevisingSeam,
    earlier_whole: AlignedPin | None = None,
) -> tuple[Runner, _RevisingSeam, PlannerPlan, dict[str, Any], list[Event]]:
    """A turn paused on a plan pinning the compiled US health aid source, its
    reading stored (marked as one the planner compiled to align a basis when
    `aligned`, as one whose sets were not readable when `stored_unreadable`, and
    as one resolved with `stored_groups` when given) beside its block naming `entity_groups` when given, held as the
    whole `earlier_whole` an earlier plan compiled from its part when given,
    then answered with a switch to `option_id`, by default its sector to CRS
    120 and 130, through `search`. Returns the runner, the seam, the plan, the scope and the
    resumed turn's events."""
    scope = _hashed_scope()
    pin = scope["compiled"]["spec_id"]
    plan = _scoped_plan(pin, scope, schemas=real_schemas)
    seam = seam_type(
        [{"plan": plan.model_dump(mode="json"), "summary": {}}],
        revised or {"refused": {"text": "unused"}},
    )
    runner, _ = orchestrator_runner(
        plan_call(_BRIEF),
        planner=seam,
        executor=ScriptedExecutor(),
        definition_surface=DataMcpDefinitions(InMemoryDefinitionStore(), search=search),
    )
    await _start(runner)
    reading = SearchReading.from_part(
        {
            "variable": "health aid",
            "status": "definite",
            "spec_id": pin,
            "scope": scope,
            "planner_axes": ["measure_basis"] if aligned else [],
            "entity_groups": stored_groups or [],
            "entity_groups_unreadable": stored_unreadable,
        }
    )
    _session_state(runner)[ORCHESTRATOR_SEARCHES_KEY] = [reading.model_dump(mode="json")]
    if earlier_whole is not None:
        save_aligned_pin(
            cast(SessionState, _session_state(runner)),
            earlier_whole.model_copy(update={"spec_id": pin}),
        )
    if entity_groups is not None:
        _session_state(runner)[ORCHESTRATOR_SPECS_KEY] = {
            pin: {
                "spec_id": pin,
                "resolved": {"scope_choices": scope, "entity_groups": entity_groups},
            }
        }
    choice_id = _row_id(plan, option_id.split(":", maxsplit=1)[0])
    events = await _resume(
        runner,
        APPROVE_INTERRUPT,
        {"approved": None, "switch": {"choice_id": choice_id, "option_id": option_id}},
    )
    return runner, seam, plan, scope, events


@pytest.mark.parametrize("basis_row", [True, False], ids=["basis-row", "no-basis-row"])
async def test_a_switch_compiles_through_search_repins_the_plan_and_says_what_moved(
    real_schemas: list[ToolSchema], basis_row: bool
) -> None:
    """The switch node posts the scope's base identity with its entries and
    the switch, registers what Search compiles, re-pins the plan to it with
    no planner call, stores the new reading, and offers the revised plan
    with the basis the switch moved, under the id its row has, or would have
    on a plan showing no basis row."""
    scope = _hashed_scope()
    new_pin, envelope = _switched_answer(scope)
    revised = _scoped_plan(
        new_pin,
        envelope["parts"][0]["spec"]["resolved"]["scope_choices"],
        schemas=real_schemas,
        version=2,
    )
    if not basis_row:
        basis = {d.slot for d in revised.decisions if d.scope and d.scope.axis == "measure_basis"}
        revised = PlannerPlan.model_validate(
            {
                **revised.model_dump(),
                "decisions": [d for d in revised.decisions if d.slot not in basis],
                "evidence_requirements": [
                    r.model_copy(
                        update={"decision_ids": tuple(i for i in r.decision_ids if i not in basis)}
                    )
                    for r in revised.evidence_requirements
                ],
            }
        )
    search = SearchStub(answer={scope["id"]: envelope})

    runner, seam, plan, scope, events = await _switch_runner(
        real_schemas, search, {"plan": revised.model_dump(mode="json"), "summary": {}}
    )

    assert len(seam.calls) == 1  # the first draft; the switch drafts nothing
    (posted,) = search.posted
    (entry,) = posted["input"]["entries"]
    assert (entry["variable_text"], entry["identity"]) == (scope["id"], scope["base"]["identity"])
    assert entry["scope_entries"] == [
        scope["entries"][1],
        {
            "axis": "sector",
            "option_id": "sector:120+130",
            "provenance": "explicit",
            "concept": "health",
            "origin": {"kind": "choice", "choice_id": f"{scope['id']}:sector"},
        },
    ]
    (revision,) = seam.revisions
    assert revision["inherits"] == {new_pin: scope["compiled"]["spec_id"]}
    assert [r["source_spec_id"] for r in revision["evidence_requirements"]] == [new_pin]
    assert PlannerPlan.model_validate(revision["prior_plan"]) == plan
    state = _session_state(runner)
    assert new_pin in state["planner:pin_sources"]
    assert PlannerPlan.model_validate_json(state["plan"]) == revised
    stored = SearchReading.model_validate(state[ORCHESTRATOR_SEARCHES_KEY][-1])
    assert (stored.text, stored.spec_id) == ("health aid", new_pin)
    assert stored.scope is not None
    assert stored.scope.choices[0].active == "sector:120+130"
    assert _pending_interrupt(events) == APPROVE_INTERRUPT
    offered = _output_of(events, "switch")
    assert offered["revision"] == plan_revision(revised)
    assert offered["moved"] == [
        {
            "choice_id": _row_id(revised, "measure_basis"),
            "from_label": "gross disbursements, current USD",
            "to_label": "grant equivalents",
            "reason": "It conflicts with another choice.",
        }
    ]


def test_one_move_across_two_switched_sources_with_no_row_is_listed_once(
    real_schemas: list[ToolSchema],
) -> None:
    """Two switched sources, each pinned by a requirement of the plan and each
    moved from gross disbursements to grant equivalents, with no basis row on
    the card: the move is listed once, under the id a row over both would have."""
    scope = _hashed_scope()
    _, envelope = _switched_answer(scope)
    after = envelope["parts"][0]["spec"]["resolved"]["scope_choices"]

    def source(prefix: str) -> Any:
        before, answered = copy.deepcopy(scope), copy.deepcopy(after)
        for raw in (before, answered):
            raw["id"] = f"{prefix}_{raw['id']}"
            for choice in raw["choices"]:
                choice["id"] = f"{raw['id']}:{choice['axis']}"
        return switch_module._Switched(
            pin=f"{prefix}_pin",
            reading=cast(Any, None),
            scope=PartScope.from_search(before),
            envelope={},
            spec_id=f"{prefix}_spec",
            after=PartScope.from_search(answered),
        )

    (dataset,) = [
        r
        for r in _seam_plan(schemas=real_schemas).evidence_requirements
        if isinstance(r, DatasetRequirement)
    ]
    plan = SimpleNamespace(
        decisions=(),
        evidence_requirements=[
            dataset.model_copy(update={"id": "first", "source_spec_id": "a_spec"}),
            dataset.model_copy(update={"id": "second", "source_spec_id": "b_spec"}),
        ],
    )

    moved = switch_module._moved_entries(cast(Any, plan), [source("a"), source("b")], axis="sector")

    assert [entry["choice_id"] for entry in moved] == ["first:measure_basis"]


async def test_a_switch_on_a_pin_an_earlier_plan_declared_a_whole_is_applied(
    real_schemas: list[ToolSchema],
) -> None:
    """The switched source was compiled for an earlier plan as a whole with its
    sector left open, and the plan now pins it as a plain requirement with no
    `whole_of`: it is no whole, so the user's pick of one sector is posted to
    Search and the revised plan offered."""
    scope = _hashed_scope()
    new_pin, envelope = _switched_answer(scope)
    revised = _scoped_plan(
        new_pin,
        envelope["parts"][0]["spec"]["resolved"]["scope_choices"],
        schemas=real_schemas,
        version=2,
    )
    search = SearchStub(answer={scope["id"]: envelope})
    part = "spec_" + "4" * 32
    whole = AlignedPin(origin=part, option_ids=(), spec_id="", digest={}, whole_axis="sector")

    _, seam, _, _, events = await _switch_runner(
        real_schemas,
        search,
        {"plan": revised.model_dump(mode="json"), "summary": {}},
        earlier_whole=whole,
    )

    assert len(search.posted) == 1
    assert len(seam.revisions) == 1
    offered = _output_of(events, "switch")
    assert "switch_failed" not in offered
    assert offered["revision"] == plan_revision(revised)


# The origin alignment compiled `_switch_plan_with_an_aligned_input`'s input from.
_ALIGNED_ORIGIN = "spec_" + "a" * 32


class _RecordingPlanner:
    """A `PlannerModel` whose revision records its arguments and refuses."""

    def __init__(self) -> None:
        self.revisions: list[dict[str, Any]] = []

    async def revise(self, **kwargs: Any) -> dict[str, Any]:
        self.revisions.append(kwargs)
        return {"refused": {"code": "plan_not_ready", "reason": "recorded"}}


@pytest.mark.parametrize("row", ["own", "aligned"])
async def test_a_switch_moves_only_the_requirements_its_row_is_for(
    real_schemas: list[ToolSchema], row: str
) -> None:
    """The plan pins the US health aid source as the user's own and holds an input
    alignment moved onto a compile Search answered with the same spec_id, each with
    its own row. A switch of either row posts that row's reading, the compile's for
    the aligned row, and re-pins the requirement that row is for alone."""
    scope = _hashed_scope()
    pin = scope["compiled"]["spec_id"]
    _, envelope = _switched_answer(scope)
    plan = _scoped_plan(pin, scope, schemas=real_schemas)
    (own,) = [r for r in plan.evidence_requirements if isinstance(r, DatasetRequirement)]
    own_decisions = [d for d in plan.decisions if d.scope is not None]
    aligned_decisions = [
        d.model_copy(
            update={
                "slot": scope_choice_slot(pin, d.scope.axis, aligned_for="aligned"),
                "scope": d.scope.model_copy(update={"aligned_for": "aligned"}),
            }
        )
        for d in own_decisions
        if d.scope is not None
    ]
    aligned = own.model_copy(
        update={
            "id": "aligned",
            "aligned_from": _ALIGNED_ORIGIN,
            "aligned_options": ("measure_basis:gross",),
            "decision_ids": tuple(d.slot for d in aligned_decisions),
        }
    )
    plan = plan.model_copy(
        update={
            "decisions": (*plan.decisions, *aligned_decisions),
            "evidence_requirements": (*plan.evidence_requirements, aligned),
        }
    )
    compiled_scope = copy.deepcopy(scope)
    compiled_scope["entries"] = compiled_scope["entries"][:1]
    readings = [
        SearchReading.from_part(
            {"variable": "health aid", "status": "definite", "spec_id": pin, "scope": scope}
        ),
        SearchReading.from_part(
            {
                "variable": "health aid",
                "status": "definite",
                "spec_id": pin,
                "scope": compiled_scope,
                "compiled_from": CompiledFrom(
                    origin=_ALIGNED_ORIGIN, option_ids=("measure_basis:gross",)
                ).model_dump(mode="json"),
            }
        ),
    ]
    search = SearchStub(answer={scope["id"]: envelope})
    planner = _RecordingPlanner()
    ctx = SimpleNamespace(
        state={ORCHESTRATOR_SEARCHES_KEY: [r.model_dump(mode="json") for r in readings]}
    )
    (clicked,) = [
        d
        for d in plan.decisions
        if d.scope is not None
        and d.scope.axis == "sector"
        and (d.scope.aligned_for is not None) == (row == "aligned")
    ]

    with pytest.raises(SwitchFailedError, match="recorded"):
        await switch_module._switched_plan(
            cast(Any, ctx),
            plan,
            choice_id=row_choice_id(plan, clicked),
            option_id="sector:120+130",
            planner=cast(Any, planner),
            surface=DataMcpDefinitions(InMemoryDefinitionStore(), search=search),
        )

    (posted,) = search.posted
    (entry,) = posted["input"]["entries"]
    read = scope if row == "own" else compiled_scope
    assert entry["scope_entries"][:-1] == read["entries"][1:]
    (revision,) = planner.revisions
    assert [r["id"] for r in revision["evidence_requirements"]] == [
        own.id if row == "own" else "aligned"
    ]


_AFRICA_TOTAL = {"label": "Africa", "ref": "africa", "total": "DACAfricaTotal"}
# Africa as a set naming a class of party outside `common.place_type.PartyClass`: the
# set does not read, but what a resubmit posts of it does.
_AFRICA_UNKNOWN_CLASS = {"label": "Africa", "ref": "africa", "whole_of": ["sovereign_fund"]}
# Africa as a set stating a registry id that is not text, so what a resubmit posts of
# it does not read.
_AFRICA_UNPOSTABLE = {"label": "Africa", "ref": "africa", "group_id": 7}


_AFRICA_POSTED = {"destination": {"label": "Africa", "ref": "africa"}}


@pytest.mark.parametrize(
    ("group", "stored_unreadable", "stored_groups", "posted"),
    [
        pytest.param(_AFRICA_TOTAL, False, None, _AFRICA_POSTED, id="block-reads-stored-readable"),
        pytest.param(_AFRICA_TOTAL, True, None, _AFRICA_POSTED, id="block-reads-stored-unreadable"),
        pytest.param(
            _AFRICA_UNKNOWN_CLASS,
            True,
            None,
            _AFRICA_POSTED,
            id="block-of-an-unknown-class-stored-unreadable",
        ),
        pytest.param(
            _AFRICA_UNPOSTABLE,
            False,
            [{"role": "destination", **_AFRICA_TOTAL}],
            _AFRICA_POSTED,
            id="block-unreadable-stored-readable",
        ),
        # The block names no sets and the reading was stored with the G7 on its
        # donors: the stored set is posted, since the block's none says nothing of it.
        pytest.param(
            None,
            False,
            [{"role": "origin", "label": "G7"}],
            {"origin": {"label": "G7"}},
            id="block-names-none-stored-readable",
        ),
    ],
)
async def test_a_switch_posts_the_parts_groups_as_every_resubmit_posts_them(
    real_schemas: list[ToolSchema],
    group: dict[str, Any] | None,
    stored_unreadable: bool,
    stored_groups: list[dict[str, Any]] | None,
    posted: dict[str, Any],
) -> None:
    """The part's block names Africa as the set its recipients were asked for,
    with the total party Search states for it. The switch posts the set as a
    compile on another basis does (`compiler.sources.wire_groups`): its label
    and ref, and no total, which Search recomputes. It reads the block again,
    so a reading stored when an earlier release could not read the set switches
    once this release can. A set naming a class of party this release does not
    know is posted by its label and ref, all the resubmit sends of it. Where what
    the block's set posts does not read, as a set stating a registry id that is
    not text, or where the block names no sets, it posts the
    readable sets the reading was stored with."""
    scope = _hashed_scope()
    _, envelope = _switched_answer(scope)
    search = SearchStub(answer={scope["id"]: envelope})

    await _switch_runner(
        real_schemas,
        search,
        entity_groups={"destination": group} if group is not None else {},
        stored_unreadable=stored_unreadable,
        stored_groups=stored_groups,
    )

    (sent,) = search.posted
    (entry,) = sent["input"]["entries"]
    assert entry.get("entity_groups") == posted


async def test_a_switch_whose_parts_sets_cannot_be_read_offers_the_plan_unchanged(
    real_schemas: list[ToolSchema],
) -> None:
    """The part's block names a set on its recipients with a registry id that is not
    text, so what the switch would post of it cannot be read. Posting the part without it would let
    Search count another population, so the switch posts nothing and offers the
    plan unchanged under `switch_failed`, which asking again cannot get past."""
    scope = _hashed_scope()
    _, envelope = _switched_answer(scope)
    search = SearchStub(answer={scope["id"]: envelope})

    runner, _, plan, _, events = await _switch_runner(
        real_schemas, search, entity_groups={"destination": _AFRICA_UNPOSTABLE}
    )

    assert search.posted == []
    offered = _output_of(events, "switch")
    assert offered["revision"] == plan_revision(plan)
    failed = offered["switch_failed"]
    assert (failed["code"], failed["retryable"]) == (SCOPE_CHOICE_NOT_APPLIED, False)
    assert PlannerPlan.model_validate_json(_session_state(runner)["plan"]) == plan


@pytest.mark.parametrize(
    ("option_id", "kept"),
    [("sector:120+130", True), ("measure_basis:grant_equivalent", False)],
    ids=["another-axis", "the-basis"],
)
async def test_a_switch_keeps_the_planners_basis_mark_unless_it_switches_the_basis(
    real_schemas: list[ToolSchema], option_id: str, kept: bool
) -> None:
    """A reading the planner compiled to align a basis keeps that mark, on the
    new reading, its stored block and its pin's block, through a switch of its
    sector, since its basis entry is still the planner's, and the revised card
    states that basis as the planner's; a switch of the basis is the user's,
    drops the mark, and the card states that basis as the user's."""
    scope = _hashed_scope()
    new_pin, envelope = _switched_answer(scope)
    answered = envelope["parts"][0]["spec"]["resolved"]["scope_choices"]
    basis = answered["choices"][2]
    basis["provenance"] = "explicit"
    answered["entries"].append(
        {
            "axis": "measure_basis",
            "option_id": basis["active"],
            "provenance": "explicit",
            "origin": {"kind": "choice", "choice_id": basis["id"]},
        }
    )
    revised = _scoped_plan(
        new_pin,
        envelope["parts"][0]["spec"]["resolved"]["scope_choices"],
        schemas=real_schemas,
        version=2,
    )
    search = SearchStub(answer={scope["id"]: envelope})

    runner, *_ = await _switch_runner(
        real_schemas,
        search,
        {"plan": revised.model_dump(mode="json"), "summary": {}},
        option_id=option_id,
        aligned=True,
    )

    state = _session_state(runner)
    stored = SearchReading.model_validate(state[ORCHESTRATOR_SEARCHES_KEY][-1])
    marked = ["measure_basis"] if kept else []
    assert (stored.spec_id, list(stored.planner_axes)) == (new_pin, marked)
    assert state[ORCHESTRATOR_SPECS_KEY][new_pin].get(PLANNER_AXES_BLOCK_KEY, []) == marked
    pinned = state["planner:pin_sources"][new_pin]["block"]
    assert pinned.get(PLANNER_AXES_BLOCK_KEY, []) == marked
    card = scope_choice_decisions({new_pin: PinSource(block=pinned, settled=True)}).get(new_pin, ())
    assert [
        d.resolved_by for d in card if d.scope is not None and d.scope.axis == "measure_basis"
    ] == ([ResolvedBy.PLANNER_DEFAULT] if kept else [ResolvedBy.USER])


@pytest.mark.parametrize(
    ("refused_by", "code", "retryable"),
    [
        ("search", "requested_option_unavailable", False),
        ("search", "transient_unavailable", True),
        ("revision", "operation_unsupported", False),
        ("revision", "discovery_unavailable", True),
        ("revision", "surface_unavailable", True),
    ],
    ids=["search", "search-transient", "revision", "revision-transient", "revision-surface"],
)
async def test_a_switch_that_cannot_apply_offers_the_plan_unchanged_with_why(
    real_schemas: list[ToolSchema], refused_by: str, code: str, retryable: bool
) -> None:
    """Search refusing the switch (`requested_option_unavailable`), or the
    revision failing, offers the same plan again with the failed switch, its
    code, one sentence saying why and whether asking again can succeed, which
    a Search that could not answer "just now" can,
    leaves the readings as they were and the planner's drafting state as it
    found it. A revision's refusal gives the reason its own
    (`harness.sentences.refusal_reason`); only one Search could not answer
    in time can succeed later. The part Search answered stays in the turn's
    resolve store, which holds whatever the switch's outcome."""
    scope = _hashed_scope()
    _, envelope = _switched_answer(scope)
    search = (
        SearchStub(
            refuse={
                scope["id"]: {
                    "code": code,
                    "title": "Requested Option Unavailable",
                    "detail": "Scope entry 1 cannot apply to this part: option_unknown.",
                    "pointer": "/input/entries/0/scope_entries/1",
                    "retryable": retryable,
                }
            }
        )
        if refused_by == "search"
        else SearchStub(answer={scope["id"]: envelope})
    )

    why = "Official development assistance would then count different providers' spending."
    refused = {
        "refused": {
            "text": "This cannot be calculated as planned, so nothing was recorded.",
            "reason": why,
            "final": False,
            "code": code,
        }
    }

    runner, seam, plan, scope, events = await _switch_runner(real_schemas, search, refused)

    assert len(seam.calls) == 1
    assert len(seam.revisions) == (0 if refused_by == "search" else 1)
    offered = _output_of(events, "switch")
    assert offered["revision"] == plan_revision(plan)
    assert "refusal_note" not in offered
    assert offered["switch_failed"] == {
        "choice_id": _row_id(plan, "sector"),
        "option_id": "sector:120+130",
        # A Search that did not answer states no code of its own.
        "code": SCOPE_CHOICE_NOT_APPLIED if refused_by == "search" and retryable else code,
        "reason": f"Search {'could not' if retryable else 'cannot'} count health (CRS 120) "
        "and population and reproductive health (CRS 130) "
        f"{'just now' if retryable else 'for this source'}."
        if refused_by == "search"
        else why,
        "retryable": retryable,
    }
    state = _session_state(runner)
    assert PlannerPlan.model_validate_json(state["plan"]) == plan
    assert [r["spec_id"] for r in state[ORCHESTRATOR_SEARCHES_KEY]] == [
        scope["compiled"]["spec_id"]
    ]
    # The fake seam drafted with no pin sources or rounds of its own.
    assert state.get("planner:rounds") is None
    assert state.get("planner:pin_sources") is None
    resolved = [
        value for key, value in state.items() if key.startswith(RESOLVED_STATE_PREFIX) and value
    ]
    assert len(resolved) == (0 if refused_by == "search" else 1)
    assert _pending_interrupt(events) == APPROVE_INTERRUPT


@pytest.mark.parametrize(
    "seam_type",
    [_StoppedSeam, _FinishingStoppedSeam],
    ids=["revision-cancelled", "revision-finishes-with-the-stop"],
)
async def test_a_switch_the_user_stops_offers_the_plan_unchanged_to_answer_again(
    real_schemas: list[ToolSchema], seam_type: type[_StoppedSeam]
) -> None:
    """A Stop while the revision runs (`SWITCH_STOP`) cancels it and offers
    the plan unchanged under `switch_cancelled`, which asking again can get
    past, on a live pause, with the drafting state, the plan and the readings
    the revision wrote put back, a declared whole's reading among them. A
    revision that finishes in the same step as the Stop ends stopped too."""
    scope = _hashed_scope()
    new_pin, envelope = _switched_answer(scope)
    revised = _scoped_plan(
        new_pin,
        envelope["parts"][0]["spec"]["resolved"]["scope_choices"],
        schemas=real_schemas,
        version=2,
    )
    token = SWITCH_STOP.set(asyncio.Event())
    try:
        runner, seam, plan, scope, events = await _switch_runner(
            real_schemas,
            SearchStub(answer={scope["id"]: envelope}),
            {"plan": revised.model_dump(mode="json"), "summary": {}},
            seam_type=seam_type,
        )
    finally:
        SWITCH_STOP.reset(token)

    assert len(seam.revisions) == 1
    offered = _output_of(events, "switch")
    assert offered["revision"] == plan_revision(plan)
    assert offered["switch_failed"] == {
        "choice_id": _row_id(plan, "sector"),
        "option_id": "sector:120+130",
        "code": SWITCH_CANCELLED,
        "reason": "You stopped this change, so the plan is as it was.",
        "retryable": True,
    }
    state = _session_state(runner)
    assert PlannerPlan.model_validate_json(state["plan"]) == plan
    assert [r["spec_id"] for r in state[ORCHESTRATOR_SEARCHES_KEY]] == [
        scope["compiled"]["spec_id"]
    ]
    assert state.get("planner:rounds") is None
    assert _pending_interrupt(events) == APPROVE_INTERRUPT


class _RaisingSeam(_StoppedSeam):
    """A `_StoppedSeam` whose revision, once it has written its state, raises an
    error of its own with no Stop. `sessions` holds the session each revision ran in."""

    sessions: ClassVar[list[Any]] = []

    async def revise(self, **kwargs: Any) -> dict[str, Any]:
        self.sessions.append(kwargs["ctx"].session)
        token = SWITCH_STOP.set(asyncio.Event())
        try:
            return await super().revise(**kwargs)
        finally:
            SWITCH_STOP.reset(token)

    async def _stopped(self) -> dict[str, Any]:
        raise RuntimeError("revision broke")


async def test_a_switch_whose_revision_raises_puts_the_state_back_and_raises(
    real_schemas: list[ToolSchema],
) -> None:
    """A revision ending on an error other than `SwitchFailedError` ends the switch
    with that error, once the drafting state, the plan and the readings it wrote,
    a declared whole's reading among them, are put back for the next turn."""
    scope = _hashed_scope()
    new_pin, envelope = _switched_answer(scope)
    revised = _scoped_plan(
        new_pin,
        envelope["parts"][0]["spec"]["resolved"]["scope_choices"],
        schemas=real_schemas,
        version=2,
    )
    _RaisingSeam.sessions.clear()

    with pytest.raises(RuntimeError, match="revision broke"):
        await _switch_runner(
            real_schemas,
            SearchStub(answer={scope["id"]: envelope}),
            {"plan": revised.model_dump(mode="json"), "summary": {}},
            seam_type=_RaisingSeam,
        )

    (session,) = _RaisingSeam.sessions
    state = session.state
    assert [r["spec_id"] for r in state[ORCHESTRATOR_SEARCHES_KEY]] == [
        scope["compiled"]["spec_id"]
    ]
    assert state.get("planner:rounds") is None


def _unwinding_revision(
    unwinding: asyncio.Event, *, error: type[BaseException], linger: float = 0
) -> Callable[[], Awaitable[dict[str, Any]]]:
    """A revision that waits until cancelled, then sets `unwinding`, waits
    `linger` seconds and raises `error` as its cleanup ends."""

    async def revision() -> dict[str, Any]:
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            unwinding.set()
            await asyncio.sleep(linger)
            raise error from None
        raise AssertionError("unreachable")

    return revision


_UNWIND_ERRORS = pytest.mark.parametrize(
    "error", [asyncio.CancelledError, RuntimeError], ids=["clean-cancel", "cleanup-raises"]
)


@_UNWIND_ERRORS
async def test_a_stopped_revision_ends_the_switch_as_stopped_however_it_unwinds(
    error: type[BaseException],
) -> None:
    """A Stop cancels the revision, whose cleanup ends cleanly or raises an error of
    its own: either way the user stopped the switch, so it ends under
    `switch_cancelled`, which the `switch` node answers by putting the plan back."""
    stop = asyncio.Event()
    unwinding = asyncio.Event()
    revision = _unwinding_revision(unwinding, error=error)
    token = SWITCH_STOP.set(stop)
    try:
        switch = asyncio.ensure_future(switch_module._unless_stopped(cast(Any, None), revision()))
    finally:
        SWITCH_STOP.reset(token)
    await asyncio.sleep(0)
    stop.set()

    with pytest.raises(SwitchFailedError) as stopped:
        await asyncio.wait_for(switch, timeout=5)
    assert stopped.value.code == SWITCH_CANCELLED


@_UNWIND_ERRORS
async def test_a_switch_cancelled_while_its_stopped_revision_unwinds_stays_cancelled(
    error: type[BaseException],
) -> None:
    """A Stop cancels the revision, and the switch itself is cancelled while the
    revision's own cleanup still runs, which then ends cleanly or raises: the
    switch ends cancelled, and offers no plan as if only the Stop had ended it."""
    stop = asyncio.Event()
    unwinding = asyncio.Event()
    revision = _unwinding_revision(unwinding, error=error, linger=5)

    token = SWITCH_STOP.set(stop)
    try:
        switch = asyncio.ensure_future(switch_module._unless_stopped(cast(Any, None), revision()))
    finally:
        SWITCH_STOP.reset(token)
    await asyncio.sleep(0)
    stop.set()
    await asyncio.wait_for(unwinding.wait(), timeout=5)
    switch.cancel()

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(switch, timeout=5)


# The source the planner wrote on the deployed plan's numerator, in the
# question's words, and the provenance Search states for the composite of
# bilateral and imputed multilateral aid the user switched it to.
_STATED_SOURCE = "OECD Creditor Reporting System (CRS)"
_INCLUSIVE_PROVENANCE = {
    "ref": "provenance/CRSAndONEImputedMultilateralSectors",
    "label": "CRS and ONE Imputed Multilateral Sectors",
}


class _RecordedFixtures(FixtureSurface):
    """A `FixtureSurface` recording each call's tool and arguments in `calls`."""

    def __init__(
        self, *, schemas: list[ToolSchema], responses: Mapping[str, ToolCallResult]
    ) -> None:
        super().__init__(schemas=schemas, responses=responses)
        self.calls: list[tuple[str, Mapping[str, object]]] = []

    @override
    async def call_tool(
        self, *, tool: ToolId, arguments: Mapping[str, object], on_progress: Any = None
    ) -> ToolCallResult:
        self.calls.append((tool.name, arguments))
        return await super().call_tool(tool=tool, arguments=arguments, on_progress=on_progress)


class _TranslatingSeam(_Seam):
    """A `PlannerModel` drafting `plans` and revising through the real
    `harness.planner_seam.PlannerSeam.revise`: the proposal chain, the
    translation's discovery and the compile, over `schemas`, answering
    `responses`, with no model."""

    def __init__(
        self,
        plans: list[dict[str, Any]],
        *,
        schemas: list[ToolSchema],
        responses: Mapping[str, ToolCallResult] | None = None,
    ) -> None:
        super().__init__(plans)
        self.surface = _RecordedFixtures(schemas=schemas, responses=responses or {})
        self._seam = PlannerSeam(settings=Settings(), surface=self.surface)

    async def revise(self, **kwargs: Any) -> dict[str, Any]:
        return await self._seam.revise(**kwargs)


class _SwitchedOff(NamedTuple):
    """A switch off a stated source (`_switch_off_a_stated_source`): the runner, the
    switch's events, the seam that revised, the pin Search answered and the copy
    of it over the plan's years."""

    runner: Any
    events: list[Any]
    seam: _TranslatingSeam
    new_pin: str
    copy_pin: str


async def _switch_off_a_stated_source(real_schemas: list[ToolSchema]) -> _SwitchedOff:
    """The deployed failure: the planner stated the CRS as the health
    numerator's source, and the user switched it to bilateral plus imputed
    multilateral, a dataset Search names otherwise and holds from 1995. The
    switch posts the identity over every year its search answered, which the
    plan pins. Admission copies the pin over the years the plan reads, and the
    translation resolves the copy, which Search answers over those years."""
    scope = _hashed_scope()
    pin = scope["compiled"]["spec_id"]
    new_pin, envelope = _switched_answer(scope)
    block = envelope["parts"][0]["spec"]
    block["resolved"] |= {
        "coverage": {
            "kind": "exact",
            "has_data": True,
            "window": {"start_year": 1995, "end_year": 2024},
        },
        "facets": {"provenance": _INCLUSIVE_PROVENANCE},
        "entities": [{"role": "country", "ref": "country/KEN", "label": "Kenya"}],
    }
    scoped = _scoped_plan(pin, scope, schemas=real_schemas)
    plan = PlannerPlan.model_validate(
        {
            **scoped.model_dump(),
            "asks": [_BRIEF],
            "answered": [{"ask": 0, "outputs": ["spending"]}],
            "translation": None,
            "workflow": None,
            "evidence_requirements": [
                r.model_copy(update={"source": _STATED_SOURCE})
                for r in scoped.evidence_requirements
                if isinstance(r, DatasetRequirement)
            ],
        }
    )
    identity = block["definition"]["identity"]
    copy_pin, copied = windowed_identity(identity, _PLAN_WINDOW)
    over_window = copy.deepcopy(envelope)
    copied_block = over_window["parts"][0]["spec"]
    copied_block["spec_id"] = copied_block["definition"]["spec_id"] = copy_pin
    copied_block["definition"]["identity"] = copied
    copied_block["resolved"]["coverage"]["window"]["start_year"] = 2015
    seam = _TranslatingSeam(
        [{"plan": plan.model_dump(mode="json"), "summary": {}}],
        schemas=real_schemas,
        responses={
            "resolve_definition": ToolCallResult(
                is_error=False, meta={SEARCH_ENVELOPE_META_KEY: over_window}
            )
        },
    )
    search = SearchStub(answer={scope["id"]: envelope})
    runner, _ = orchestrator_runner(
        plan_call(_BRIEF),
        planner=seam,
        executor=ScriptedExecutor(),
        definition_surface=DataMcpDefinitions(InMemoryDefinitionStore(), search=search),
    )
    await _start(runner)
    reading = SearchReading.from_part(
        {"variable": "health aid", "status": "definite", "spec_id": pin, "scope": scope}
    )
    _session_state(runner)[ORCHESTRATOR_SEARCHES_KEY] = [reading.model_dump(mode="json")]

    events = await _resume(
        runner,
        APPROVE_INTERRUPT,
        {
            "approved": None,
            "switch": {"choice_id": _row_id(plan, "sector"), "option_id": "sector:120+130"},
        },
    )
    return _SwitchedOff(runner, events, seam, new_pin, copy_pin)


async def test_a_switch_moves_a_requirement_off_the_source_it_stated_and_the_plan_runs(
    real_schemas: list[ToolSchema],
) -> None:
    """A switch off a stated source (`_switch_off_a_stated_source`) drops the
    stated source with the pin it left, so the translation takes the new pin
    with no search, the plan compiles, and its run accepts what the new
    dataset returns. The translation resolves the copy admission made over the
    plan's years, and searches nothing."""
    runner, events, seam, _, copy_pin = await _switch_off_a_stated_source(real_schemas)

    assert [(tool, arguments.get("spec_id")) for tool, arguments in seam.surface.calls] == [
        ("resolve_definition", copy_pin)
    ]
    offered = _output_of(events, "switch")
    assert "switch_failed" not in offered, offered.get("switch_failed")
    revised = PlannerPlan.model_validate_json(_session_state(runner)["plan"])
    (dataset,) = [r for r in revised.evidence_requirements if isinstance(r, DatasetRequirement)]
    assert (dataset.source_spec_id, dataset.source) == (copy_pin, None)
    assert revised.workflow is not None
    assert [step.args.get("spec_id") for step in revised.workflow.steps] == [copy_pin]
    approved = PlannerPlan.model_validate(
        revised.model_dump()
        | {
            "status": "approved",
            "approved_revision": plan_revision(revised),
            "approved_at": datetime.now(UTC),
        }
    )
    manifest = _get_data_manifest(spec_id=copy_pin)
    manifest["citation"]["sources"] = [_INCLUSIVE_PROVENANCE]
    fetched = ToolCallResult(is_error=False, structured_content=manifest, text="fetched")

    _, outcome = await execute_approved_plan(
        approved,
        surface=_fixture_surface(real_schemas, get_data_result=fetched),
        fetch_series_stats=scripted_fetch_series_stats,
    )

    assert isinstance(outcome, Accepted), outcome


async def test_a_later_period_amendment_reads_the_switched_dataset_over_every_year_it_holds(
    real_schemas: list[ToolSchema],
) -> None:
    """After a switch off a stated source (`_switch_off_a_stated_source`), the
    stored reading is of the switched dataset, the planner's block for it
    states the years Search holds it for, from 1995, and the plan's copy over
    its own years is held as a copy of that dataset. A later amendment of the period
    reads the dataset whole and copies it over the new years."""
    runner, _, _, new_pin, copy_pin = await _switch_off_a_stated_source(real_schemas)

    state = _session_state(runner)
    shown = state["planner:pin_sources"][new_pin]["block"]
    assert shown["resolved"]["coverage"]["window"]["start_year"] == 1995
    stored = SearchReading.model_validate(state[ORCHESTRATOR_SEARCHES_KEY][-1])
    assert stored.spec_id == new_pin
    assert windowed_pins(cast(SessionState, state))[copy_pin].windowed_from == new_pin


@pytest.mark.parametrize(
    ("exact_codes", "kept"), [(frozenset({"DT.ODA.ALLD.CD"}), True), (frozenset(), False)]
)
def test_a_repinned_requirement_keeps_its_series_code_only_where_the_new_dataset_is_it(
    exact_codes: frozenset[str], kept: bool
) -> None:
    stated = DatasetRequirement(
        id="aid",
        decision_ids=["window"],
        measure="aid",
        entities=[Entity(name="Kenya", role="recipient")],
        period=Period(start=2015, end=2024),
        source="World Bank",
        source_ref="provenance/WorldBank",
        source_exclude=True,
        source_code="DT.ODA.ALLD.CD",
        source_spec_id="spec_" + "a" * 32,
    )

    moved = repinned(stated, "spec_" + "b" * 32, exact_codes)

    assert (moved.source_spec_id, moved.source, moved.source_ref) == (
        "spec_" + "b" * 32,
        None,
        None,
    )
    assert moved.source_exclude is False
    assert moved.source_code == ("DT.ODA.ALLD.CD" if kept else None)


# --- a chip under an answer, applied with no search ----------------------------


class _RecordingExecutor(scripted.ScriptedExecutor):
    """`harness.scripted.ScriptedExecutor`, recording each call and each step a node records."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.steps: list[dict[str, Any]] = []

    async def __call__(
        self, *, tool: str, args: dict[str, Any], on_progress: ProgressCallback | None = None
    ) -> dict[str, Any]:
        self.calls.append({"tool": tool, "args": args})
        return await super().__call__(tool=tool, args=args, on_progress=on_progress)

    def record_step(self, **step: Any) -> None:
        self.steps.append(step)


class _BrokenFetchExecutor(_RecordingExecutor):
    """`_RecordingExecutor` whose `get_data` raises, as a fault inside DFO would."""

    async def __call__(
        self, *, tool: str, args: dict[str, Any], on_progress: ProgressCallback | None = None
    ) -> dict[str, Any]:
        if tool == "get_data":
            raise RuntimeError("get_data broke")
        return await super().__call__(tool=tool, args=args, on_progress=on_progress)


async def _chip_turn(
    search: SearchStub,
    *,
    option_id: str = "sector:120+130",
    executor: _RecordingExecutor | None = None,
    spec_id: str | None = None,
    entity_groups: dict[str, Any] | None = None,
) -> tuple[Runner, ScriptLlm, _RecordingExecutor, dict[str, Any]]:
    """A turn applying a chip that moves the US health aid answer's
    `option_id` axis to `option_id`, offered on the reading `spec_id`, by
    default the answer's own, its reading stored by an earlier turn from a
    block naming `entity_groups` when given, resubmitted through `search`.
    Its model tries to search, then answers."""
    scope = _hashed_scope()
    pin = scope["compiled"]["spec_id"]
    block = {"spec_id": pin, "resolved": {"scope_choices": scope}}
    if entity_groups is not None:
        block["resolved"]["entity_groups"] = entity_groups
    reading = SearchReading.from_part(
        {"variable": "health aid", "status": "definite", "spec_id": pin, **reading_facts(block)}
    ).model_dump(mode="json")
    executor = executor or _RecordingExecutor()
    runner, model = orchestrator_runner(
        call("search", query="health aid", parts=[{"variable": "health aid"}]),
        executor=executor,
        definition_surface=DataMcpDefinitions(InMemoryDefinitionStore(), search=search),
    )
    axis = option_id.split(":", maxsplit=1)[0]
    await open_session(
        runner,
        apply_choice=json.dumps(
            {
                "choice_id": f"{scope['id']}:{axis}",
                "option_id": option_id,
                "spec_id": spec_id or pin,
            }
        ),
        **{ORCHESTRATOR_PRIOR_LEDGER_KEY: {"readings": [reading], "warnings": {}, "spec_ids": []}},
    )
    await run_turn(runner, types.UserContent("Count health and population."))
    return runner, model, executor, scope


async def test_a_chip_compiles_through_search_fetches_what_it_compiled_and_never_searches() -> None:
    """The `seed` node posts the stored reading's base identity with the
    switched entries, stores the reading Search compiles, fetches its spec_id
    and hands the agent the result; the agent's own search is refused unsent."""
    scope = _hashed_scope()
    new_pin, envelope = _switched_answer(scope)
    search = SearchStub(answer={scope["id"]: envelope})

    runner, model, executor, scope = await _chip_turn(search)

    (posted,) = search.posted
    (entry,) = posted["input"]["entries"]
    assert entry["identity"] == scope["base"]["identity"]
    assert entry["scope_entries"][-1]["option_id"] == "sector:120+130"
    assert [c["args"]["spec_id"] for c in executor.calls if c["tool"] == "get_data"] == [new_pin]
    state = _session_state(runner)
    stored = SearchReading.model_validate(state[ORCHESTRATOR_SEARCHES_KEY][-1])
    assert (stored.text, stored.spec_id) == ("health aid", new_pin)
    assert stored.fetch is not None and stored.fetch.refusal_code is None
    assert state[ORCHESTRATOR_APPLIED_KEY] == {
        "choice_id": f"{scope['id']}:sector",
        "option_id": "sector:120+130",
        "from_option_id": "sector:120",
        "spec_id": new_pin,
    }
    first = request_text(model.requests[0])
    assert f"counted it again as {new_pin}" in first
    assert frame_for(new_pin) in first
    searched = model.requests[1].contents[-1].parts[0].function_response.response
    assert searched["code"] == "tool_budget_spent"
    assert [step["result"]["outcome"] for step in executor.steps] == ["ok"]


@pytest.mark.parametrize(
    "group", [_AFRICA_TOTAL, _AFRICA_UNKNOWN_CLASS], ids=["total", "unknown-class"]
)
async def test_a_chip_posts_the_groups_its_reading_was_resolved_with(
    group: dict[str, Any],
) -> None:
    """The answer's reading came from a search whose block named Africa as the
    set its recipients were asked for, and no plan kept that block. The chip's
    resubmit posts the set as every resubmit does (`compiler.sources.wire_groups`),
    one naming a class of party this release does not know too, since the reading
    stored what the resubmit posts of it."""
    scope = _hashed_scope()
    _, envelope = _switched_answer(scope)
    search = SearchStub(answer={scope["id"]: envelope})

    await _chip_turn(search, entity_groups={"destination": group})

    (posted,) = search.posted
    (entry,) = posted["input"]["entries"]
    assert entry["entity_groups"] == {"destination": {"label": "Africa", "ref": "africa"}}


async def test_a_chip_whose_readings_sets_could_not_be_read_posts_nothing() -> None:
    """The answer's reading came from a block naming a set on its recipients with a
    registry id that is not text, and no block of it is kept, so nothing names the set. Posting the part
    without it would let Search count another population, so the chip posts nothing
    and its step fails as a plan card's switch over such a block does."""
    scope = _hashed_scope()
    _, envelope = _switched_answer(scope)
    search = SearchStub(answer={scope["id"]: envelope})

    runner, _, executor, _ = await _chip_turn(
        search, entity_groups={"destination": _AFRICA_UNPOSTABLE}
    )

    assert search.posted == []
    (step,) = executor.steps
    refusal = step["result"]["refusal"]
    assert (refusal["code"], refusal["retry_mode"]) == (SCOPE_CHOICE_NOT_APPLIED, "do_not_retry")
    assert not any(c["tool"] == "get_data" for c in executor.calls)
    assert ORCHESTRATOR_APPLIED_KEY not in _session_state(runner)


async def test_a_chip_offered_on_another_reading_is_refused_stale_and_compiles_nothing() -> None:
    """The chip names a reading other than the newest one holding its choice, as
    after a later switch of that choice: it is refused as the studio refuses a
    stale chip, and nothing is posted or fetched."""
    scope = _hashed_scope()
    _, envelope = _switched_answer(scope)
    search = SearchStub(answer={scope["id"]: envelope})

    runner, model, executor, _ = await _chip_turn(search, spec_id=f"spec_{'9' * 32}")

    assert search.posted == []
    assert CHIP_STALE_NOTE in request_text(model.requests[0])
    (step,) = executor.steps
    assert step["result"]["refusal"]["code"] == SCOPE_CHOICE_NOT_APPLIED
    assert not any(c["tool"] == "get_data" for c in executor.calls)
    assert ORCHESTRATOR_APPLIED_KEY not in _session_state(runner)


async def test_a_chip_whose_fetch_raises_answers_as_the_agents_own_fetch_would() -> None:
    """The chip's fetch raises inside DFO: the agent is handed the failure the
    orchestrator's own tool calls answer with, and the turn goes on."""
    scope = _hashed_scope()
    _, envelope = _switched_answer(scope)
    search = SearchStub(answer={scope["id"]: envelope})

    _, model, _, _ = await _chip_turn(search, executor=_BrokenFetchExecutor())

    first = request_text(model.requests[0])
    assert "fetch failed inside DFO (RuntimeError)" in first
    assert HarnessCode.RAISED.value in first


@pytest.mark.parametrize(
    ("code", "retryable", "step_code", "retry_mode"),
    [
        ("requested_option_unavailable", False, "requested_option_unavailable", "do_not_retry"),
        # A Search that did not answer states no code of its own.
        ("transient_unavailable", True, SCOPE_CHOICE_NOT_APPLIED, "retry_after_delay"),
    ],
    ids=["refused", "transient"],
)
async def test_a_chip_search_refuses_reads_as_a_plan_cards_switch_and_fetches_nothing(
    code: str, retryable: bool, step_code: str, retry_mode: str
) -> None:
    """The chip's step fails with the code and retryability the plan card's
    switch reports for the same refusal."""
    scope = _hashed_scope()
    search = SearchStub(
        refuse={
            scope["id"]: {
                "code": code,
                "title": "Requested Option Unavailable",
                "detail": "Scope entry 1 cannot apply to this part: option_unknown.",
                "pointer": "/input/entries/0/scope_entries/1",
                "retryable": retryable,
            }
        }
    )

    runner, model, executor, _ = await _chip_turn(search)

    note = (
        f"Search {'could not' if retryable else 'cannot'} count health (CRS 120) and "
        "population and reproductive health (CRS 130) "
        f"{'just now' if retryable else 'for this source'}."
    )
    assert note in request_text(model.requests[0])
    (step,) = executor.steps
    assert step["result"]["refusal"] == {
        "code": step_code,
        "retry_mode": retry_mode,
        "fix": None,
        "message": note,
    }
    assert not any(c["tool"] == "get_data" for c in executor.calls)
    assert ORCHESTRATOR_APPLIED_KEY not in _session_state(runner)
