"""Tests for the planner agent: plan-lifecycle tools and agent construction.

Most tool functions are exercised directly (`FunctionTool.func`) against a fake
tool context, rather than through a full ADK `Runner` -- what belongs to this
package is the business logic each tool function runs once ADK decides to call
it. `_FakeToolContext` carries no `tool_confirmation`: `declare_ready` gates
nothing of its own, so a body that tried to touch it would raise
`AttributeError` here rather than silently working again -- see the tests
under "Submitting".

The translator is stood in for by `_translate`, which completes a draft --
always a dataset requirement plus the derived one that reads it, the minimal
shape `Plan` accepts -- through the real `Plan.with_translation`, so these
tests stay about the planner's own transitions.
"""

from __future__ import annotations

import ast
import asyncio
import inspect
import json
import logging
from collections.abc import Awaitable, Callable, Iterator, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast, override

import pytest
from analysis_mcp.server import mcp as analysis_mcp
from compiler.compile import CompileIssueCode, compile_plan
from compiler.sources import PROVIDER_ACCOUNT, DiscoveryStatus, source_from_spec_block
from compiler.surface import FixtureSurface, McpSurface, ToolCallResult, ToolSurface
from data_mcp.definition_store import InMemoryDefinitionStore, canonicalize, spec_id_for
from data_mcp.definition_writes import windowed_identity
from data_mcp.search_result import SEARCH_ENVELOPE_META_KEY
from data_mcp.server import mcp as data_mcp
from data_mcp.tool_errors import ToolRefusal, to_tool_result
from google.adk.agents import LlmAgent
from google.adk.events import Event, EventActions, RequestInput
from google.adk.features import FeatureName
from google.adk.features._feature_registry import temporary_feature_override
from google.adk.models.google_llm import Gemini
from google.adk.planners import BuiltInPlanner
from google.adk.tools.function_tool import FunctionTool
from google.adk.tools.long_running_tool import LongRunningFunctionTool
from google.adk.workflow.utils._workflow_hitl_utils import (
    create_request_input_event,
    create_request_input_response,
)
from google.genai import _api_client, errors, types
from harness.graph.state import APPROVE_INTERRUPT, store_search
from harness.orchestrator_tools import _search_context
from harness.plan_display import plan_words, shown_decisions
from harness.planner_seam import _PLANNER_BRANCH, PlannerSeam, _reading_pin_sources
from harness.prefix_cache import PrefixCache
from harness.scope_choices import PLANNER_AXES_BLOCK_KEY
from harness.sentences import PlanWords, refusal_reason, refusal_sentence
from harness.studio_state import _turn_identities
from harness.tool_contracts import reading_facts
from harness.turn_records import RecordedIdentity, _periods
from planner import agent as planner_agent
from planner.agent import (
    ToolContextSession,
    build_planner_tools,
    create_planner_agent,
    declare_ready,
    planner_instruction,
    read_full_history,
    with_source_reads,
)
from planner.declaration import FIELD_DESCRIBED_ENUMS, SHARED_FIELDS, PlanTool, compact_schema
from planner.operations import (
    AggregateOperation,
    RetrieveOperation,
    Translation,
)
from planner.parties import PARTY_ROLES, PartyRole, flow_side, party_role, role_party
from planner.plan import (
    ASK_BREAKDOWNS_STATE_KEY,
    ASKS_ACROSS_YEARS_STATE_KEY,
    ASKS_STATE_KEY,
    MAX_DECISION_DESCRIPTION_CHARS,
    MAX_DECISION_RATIONALE_CHARS,
    MAX_PLAN_DECISIONS,
    MAX_PLAN_RECORD_CHARS,
    MAX_PLAN_REQUIREMENTS,
    MAX_PLAN_ROUNDS,
    MAX_QUESTION_CHARS,
    PLAN_ROUNDS_STATE_KEY,
    PLAN_STATE_KEY,
    WINDOWED_PINS_STATE_KEY,
    AggregateRequirement,
    AnswerMode,
    AskAnswer,
    AskBreakdown,
    ComparisonRequirement,
    DatasetRequirement,
    Decision,
    DecisionFixes,
    Decline,
    Entity,
    EvidenceRequirement,
    FilterRequirement,
    FirstYear,
    FullHistory,
    GrowthRequirement,
    JoinRequirement,
    Period,
    PinSource,
    Plan,
    PlanStatus,
    QuotientRequirement,
    RankRequirement,
    RequirementSide,
    ResolvedBy,
    ShareOfRequirement,
    SumOfRequirement,
    Unit,
    UnsupportedOperation,
    WholeOf,
    WindowedPin,
    clear_unsupported,
    current_discovery_time_left,
    current_recorded_identity,
    fix_disagreements,
    is_dataset,
    is_scope_choice,
    is_search_pin,
    load_plan,
    load_plan_rounds,
    open_window,
    pin_slot_moves,
    pin_sources,
    plan_revision,
    record_optional_decline,
    record_unsupported,
    save_full_history,
    save_pin_sources,
    save_plan,
    save_plan_rounds,
    scope_choice_slot,
    search_pin_slot,
    split_of,
    stored_plan,
    windowed_pins,
)
from planner.records import (
    MAX_ENTITIES,
    AmbiguousEntityReading,
    AmbiguousSourceReading,
    Breakdown,
    BreakdownDimension,
    DenominatorScope,
    EntityCandidate,
    EntityIn,
    EntityKey,
    EvidenceIssue,
    EvidenceIssueCode,
    GroupBy,
    Identifier,
    PeriodThreshold,
    RankAmong,
    RankDirection,
    Reduction,
    RelativePeriod,
    RollingWindow,
    Severity,
    SourceNameCandidate,
    StatVarReading,
)
from planner.tool_results import Outcome, ProblemCode, problem
from planner.workflow import EvidenceBinding, SelectedSource, ToolId, Workflow, WorkflowStep
from pydantic import TypeAdapter, ValidationError
from pydantic_core import to_jsonable_python
from submission import service as submission_service
from submission.service import (
    COMBINATION_OPTIONS,
    COMBINE_OPTION,
    AlignedPin,
    Realignment,
    SourceReadings,
    _choice_not_offered,
    _combination_unasked,
    _entity_dcid_unoffered,
    _entity_dcid_unpicked,
    _repinned,
    _resolve_intent,
    _source_ref_unpicked,
    first_year_reads,
    identity_on_record,
    save_aligned_pin,
    scope_choice_decisions,
    searched_pins,
)
from translator.discovery import (
    DefinitionNotResolvedError,
    DiscoveryLog,
    DiscoveryRecord,
    build_discovery_tools,
    load_log,
    name_pinned_groups,
    save_log,
)
from translator.mechanical import build_translation
from translator.referral import (
    ENTITY_CHOICE_REPAIR,
    OUTCOME_OF,
    REPAIR_OF,
    Referral,
    ReferralKind,
    StalePin,
    load_referral,
    save_referral,
)
from translator.tool import Translate, authored_answer, build_translate

from ..basis_alignment import (
    HEALTH,
    TOTAL,
    TOTAL_BILATERAL,
    TOTAL_GROSS,
    TOTAL_GROSS_BILATERAL,
    BasisCompiles,
    ProviderCompiles,
    WholeCompiles,
    basis_block,
    dac1_multilateral,
    dac1_with_contributions,
    provider_block,
    total_without_bilateral,
    us_2023_block,
)
from ..context_probe import plan_record
from ..data_mcp_definitions import DataMcpDefinitions
from ..decision_drafts import drafts
from ..session_events import response_event
from ..split_plan import split_draft
from ..test_studio_translate import _translator as _studio_translator
from ..test_turn_records import _switchable_scope

# The `decision`, `requirement`, `derived_requirement` and `requirements`
# fixtures, from conftest.py. Annotated structurally rather than imported: a
# test module importing its own conftest by name is ambiguous when the
# workspace root has one too.
DecisionFactory = Callable[..., Decision]
RequirementFactory = Callable[..., EvidenceRequirement]
DerivedRequirementFactory = Callable[..., AggregateRequirement]
RequirementPair = tuple[EvidenceRequirement, ...]

SIMPLE_TOOLS = ("adk_request_input", "propose_plan", "reject_plan")

SPEC = "spec_" + "a" * 32
POPULATION_SPEC = "spec_" + "c" * 32

# What `declare_ready` summarises for the plan `_translate` builds with its defaults.
DEFAULT_STEPS = [
    {
        "id": "debt",
        "tool": "data-mcp.get_data",
        "args": {"spec_id": SPEC},
        "bindings": {},
    },
    {
        "id": "totals",
        "tool": "analysis-mcp.aggregate",
        "args": {"aggregation": "sum", "group_by": ["time_period"]},
        "bindings": {"frame_ref": {"from_step": "debt", "output": "frame_ref"}},
    },
]
DEFAULT_SOURCES = [{"spec_id": SPEC, "sentence": "external debt service"}]

# The referral kinds the user decides (`translator.referral.USER_DECIDES`).
HANDED_BACK_KINDS = (
    ReferralKind.REQUIREMENT_UNMEETABLE,
    ReferralKind.UNSUPPORTED_ASK,
    ReferralKind.SOURCE_NAME_AMBIGUOUS,
)


class _FakeSession:
    """A minimal stand-in for ADK's `Session`: just `.events`, read by
    `planner.agent._elected_spec_ids` to find what the user actually chose
    answering `adk_request_input`, and by `_answered_since_referral` to find
    whether they answered at all. Empty by default -- a test that needs an
    election on record populates it with `_ask_event` and `_answer_event`,
    below, and one that needs an answer or a referral with `_ask_answered_events`
    and `_hand_back`."""

    def __init__(self) -> None:
        self.events: list[Any] = []


class _FakeToolContext:
    """A minimal stand-in for ADK's `ToolContext`.

    `.state` (a plain, mutable mapping), `.session` (`.events` only) and
    `.actions` (where `declare_ready` ends the run) are all `planner.agent`'s
    tool functions use: none of them gates on a confirmation any more, so
    none needs `.tool_confirmation`. A tool body that tried to read it would
    raise `AttributeError` against this stand-in, which is exactly the
    regression the tests under "Submitting" rely on.
    """

    def __init__(self) -> None:
        self.state: dict[str, str] = {}
        self.actions = EventActions()
        self.session = _FakeSession()
        self.invocation_id = "turn-1"


def _ask_event(candidates: list[dict[str, str]], *, call_id: str = "call-1") -> Any:
    """A fake session event for `adk_request_input`'s own function call,
    carrying `candidates` in its `args` and `call_id` in its `id` -- the
    shape `_elected_spec_ids` pairs against a matching `_answer_event`'s
    `id`. A `SimpleNamespace` tree rather than a real
    `google.adk.events.Event`: `_elected_spec_ids` only ever reads
    `.content.parts[*].function_call.{id,name,args}` and
    `.content.parts[*].function_response`, so that is all this (and
    `_answer_event`, below) build. `function_response` is set to `None`
    explicitly -- a real ADK `Part` always carries the field, defaulting to
    `None`, and `_elected_spec_ids` reads it unconditionally off every part.
    """
    call = SimpleNamespace(id=call_id, name="adk_request_input", args={"candidates": candidates})
    part = SimpleNamespace(function_call=call, function_response=None)
    return SimpleNamespace(content=SimpleNamespace(parts=[part]))


def _answer_event(spec_id: str, *, call_id: str = "call-1") -> Any:
    """A fake session event for the `FunctionResponse` answering `_ask_event`'s
    call -- same `call_id`, and `response["result"]["spec_id"]` set to what
    the user chose, the shape `harness.studio`'s `/interrupt` route actually
    sends (`create_request_input_response`).
    """
    response = SimpleNamespace(
        id=call_id, name="adk_request_input", response={"result": {"spec_id": spec_id}}
    )
    part = SimpleNamespace(function_call=None, function_response=response)
    return SimpleNamespace(content=SimpleNamespace(parts=[part]))


def _referred_event(kind: ReferralKind, requirement_id: str | None) -> Any:
    """The `propose_plan` event carrying the answer whose translation hands the plan back."""
    return response_event(
        "propose_plan",
        {
            "outcome": Outcome.OK.value,
            "status": "draft",
            "version": 1,
            "translation": {
                "kind": kind.value,
                "outcome": OUTCOME_OF[kind].value,
                "reason": "r",
                "requirement_id": requirement_id,
                "candidates": [],
            },
        },
    )


def _ask_response(call_id: str, body: dict[str, Any]) -> Any:
    """A `FunctionResponse` named `adk_request_input` under `call_id`, carrying `body`."""
    response = SimpleNamespace(id=call_id, name="adk_request_input", response=body)
    part = SimpleNamespace(function_call=None, function_response=response)
    return SimpleNamespace(content=SimpleNamespace(parts=[part]))


def _ask_answered_events(
    result: Any = "use current prices", *, call_id: str = "ask-1"
) -> list[Any]:
    """An `adk_request_input` call that offered no candidates, and the response that
    answers it with `result`. A user's free-text answer leaves these two events in the
    session."""
    return [_ask_event([], call_id=call_id), _ask_response(call_id, {"result": result})]


def _clarify_ask_event(*questions: list[dict[str, str]]) -> Event:
    """A real ADK event for the graph's own `clarify` interrupt
    (`harness.graph.state.CLARIFY_INTERRUPT`), built through ADK's own
    `create_request_input_event`: this is the shape `_offered_candidates`'s
    `payload` branch reads, distinct from `ask_user`'s flat
    `args["candidates"]` -- `RequestInput.model_dump` nests the caller's own
    `payload` under `args["payload"]`. Each argument is one question's
    candidates, under the ids `q1`, `q2`, ... by position
    (`harness.graph.state.ask_questions`). Every clarify round shares the
    fixed interrupt id `"clarify"` (`CLARIFY_INTERRUPT`), which is easy to
    confuse with `_ask_event`'s per-call id.
    """
    return create_request_input_event(
        RequestInput(
            interrupt_id="clarify",
            payload={
                "questions": [
                    {"id": f"q{number}", "question": f"Question {number}", "candidates": candidates}
                    for number, candidates in enumerate(questions, start=1)
                ]
            },
        )
    )


def _legacy_clarify_ask_event(candidates: list[dict[str, str]]) -> Event:
    """The `clarify` interrupt of a single question, with one flat list at
    `payload["candidates"]`."""
    return create_request_input_event(
        RequestInput(interrupt_id="clarify", payload={"candidates": candidates})
    )


def _option(spec_id: str) -> dict[str, str]:
    """One question's answer: the candidate `spec_id` picked."""
    return {"kind": "option", "spec_id": spec_id}


_SKIPPED = {"kind": "skipped"}


def _clarify_answer_event(*answers: dict[str, str]) -> Event:
    """A real ADK event answering `_clarify_ask_event`, the shape
    `harness.studio`'s `/interrupt` route actually sends: one answer per question,
    by position under `q1`, `q2`, ...
    (`create_request_input_response(CLARIFY_INTERRUPT, {"result": {"answers": ...}})`)."""
    result = {"answers": {f"q{number}": answer for number, answer in enumerate(answers, start=1)}}
    part = create_request_input_response("clarify", {"result": result})
    return Event(content=types.Content(role="user", parts=[part]))


def _legacy_clarify_answer_event(spec_id: str) -> Event:
    """The answer body of a single-question pause: `{"spec_id": ...}`."""
    part = create_request_input_response("clarify", {"result": {"spec_id": spec_id}})
    return Event(content=types.Content(role="user", parts=[part]))


def _show(
    ctx: _FakeToolContext, spec_id: str, *, settled: bool = True, earlier_turn: bool = False
) -> None:
    """Add `spec_id` to the datasets `harness.planner_seam` showed the planner."""
    sources = pin_sources(ctx.state)
    sources[spec_id] = PinSource(settled=settled, earlier_turn=earlier_turn)
    save_pin_sources(ctx.state, sources)


def _tools_by_name(
    *, translate: Translate | None = None, question_fixed: bool = False
) -> dict[str, FunctionTool]:
    return {
        tool.name: tool
        for tool in build_planner_tools(translate=translate, question_fixed=question_fixed)
    }


async def _propose_translated(
    tools: dict[str, FunctionTool],
    ctx: object,
    decision: Decision,
    requirements: Sequence[EvidenceRequirement],
    **intent: Any,
) -> dict[str, object]:
    """`_propose` against tools built with a translator, whose `propose_plan` is a coroutine."""
    return await cast(
        "Awaitable[dict[str, object]]", _propose(tools, ctx, decision, requirements, **intent)
    )


def _propose(
    tools: dict[str, FunctionTool],
    ctx: object,
    decision: Decision,
    requirements: Sequence[EvidenceRequirement],
    *,
    question: str = "q",
    answer_mode: AnswerMode | None = AnswerMode.COMPUTED,
    amends_version: int | None = None,
    starts_new_plan: bool = False,
) -> dict[str, object]:
    """One `propose_plan` call, stating its intent the way a correct model would.

    A plan already recorded in `ctx` means this call patches it, so
    `amends_version` defaults to that recorded version, the reading the tool
    requires the model to state (`planner.agent._resolve_intent`). A first
    call passes nothing. `answer_mode` defaults to `computed`, since
    `requirements` is the read pair every caller passes unless it overrides
    it. A test about a call that states its intent wrongly, or not at all,
    calls `tools["propose_plan"].func` directly.
    """
    recorded = load_plan(ctx.state)  # type: ignore[attr-defined]
    if amends_version is None and recorded is not None and not starts_new_plan:
        amends_version = recorded.version
    return tools["propose_plan"].func(
        question=question,
        answer_mode=answer_mode,
        decisions=drafts([decision]),
        evidence_requirements=list(requirements),
        amends_version=amends_version,
        starts_new_plan=starts_new_plan,
        tool_context=ctx,
    )


def _translate(
    ctx: object,
    *,
    searched_as: str | None = None,
    warnings: Sequence[EvidenceIssue] = (),
    reduction: Reduction = Reduction.SUM,
) -> Plan:
    """Complete the draft in `ctx` the way the translator would, through the real transition.

    Mirrors `tests/planner/test_plan.py`'s `_translation`/`_workflow` helpers:
    a `RetrieveOperation` for the dataset requirement, an `AggregateOperation`
    for the derived one that reads it, and a workflow whose `EvidenceBinding`
    carries the source's `spec_id`. `searched_as` and `warnings` land on the
    workflow as discovery and the compiler would leave them. `reduction` has
    to match the derived requirement's own `reduction` -- `with_translation`
    refuses an operation that disagrees with the requirement it carries out.
    """
    state = ctx.state  # type: ignore[attr-defined]
    draft = load_plan(state)
    assert draft is not None
    dataset = next(r for r in draft.evidence_requirements if is_dataset(r))
    (derived,) = [r for r in draft.evidence_requirements if not is_dataset(r)]
    translation = Translation(
        operations=[
            RetrieveOperation(id="debt", requirement_id=dataset.id, spec_id=SPEC),
            AggregateOperation(
                id="totals",
                requirement_id=derived.id,
                input="debt",
                reduction=reduction,
                group_by=(GroupBy.PERIOD,),
            ),
        ],
        terminal=["totals"],
    )
    workflow = Workflow(
        steps=[
            WorkflowStep(
                id="debt",
                operation_id="debt",
                tool=ToolId(server="data-mcp", name="get_data"),
                args={"spec_id": SPEC},
                input_schema_hash="hash",
            ),
            WorkflowStep(
                id="totals",
                operation_id="totals",
                tool=ToolId(server="analysis-mcp", name="aggregate"),
                args={"aggregation": "sum", "group_by": ["time_period"]},
                bindings={"frame_ref": {"from_step": "debt", "output": "frame_ref"}},
                input_schema_hash="hash",
            ),
        ],
        sources=[
            SelectedSource(
                spec_id=SPEC,
                sentence="external debt service",
                searched_as=searched_as,
                has_data=True,
            )
        ],
        evidence_bindings=[
            EvidenceBinding(requirement_id=dataset.id, step_id="debt", spec_id=SPEC)
        ],
        terminal_steps=["totals"],
        warnings=tuple(warnings),
    )
    translated = draft.with_translation(translation=translation, workflow=workflow)
    save_plan(state, translated)
    return translated


def _declare(ctx: _FakeToolContext) -> dict[str, object]:
    """`declare_ready` over the plan on record, as a proposal whose translation compiled."""
    return declare_ready({"outcome": Outcome.OK.value, "translation": {"outcome": "ok"}}, ctx)  # type: ignore[arg-type]


def _tool_args(
    tool_name: str, decision: Decision, requirements: RequirementPair
) -> dict[str, object]:
    """The minimum arguments each of `SIMPLE_TOOLS` needs to reach its own body."""
    by_tool: dict[str, dict[str, object]] = {
        "adk_request_input": {
            "message": "Which source did you mean?",
            "candidates": [{"spec_id": SPEC, "label": "a candidate"}],
        },
        "propose_plan": {
            "question": "q",
            "decisions": drafts([decision]),
            "evidence_requirements": list(requirements),
            # This tool runs against a context that already holds a plan, so
            # its call states that it patches it.
            "amends_version": 1,
        },
        "reject_plan": {},
    }
    return by_tool[tool_name]


def test_build_planner_tools_registers_the_three_planner_tools() -> None:
    tools = _tools_by_name()

    assert set(tools) == set(SIMPLE_TOOLS)


# --- Submitting ---------------------------------------------------------------------


async def test_a_proposal_that_compiles_declares_the_plan_ready(
    decision: DecisionFactory, requirements: RequirementPair
) -> None:
    # A proposal whose translation compiles declares the plan ready in code,
    # asking for no confirmation: `_FakeToolContext` carries no
    # `tool_confirmation` attribute at all, so a body that tried to read it
    # would raise `AttributeError` here rather than requesting one. It ends
    # the planner's run, so no model call follows. The plan itself is left
    # exactly where translation put it.
    async def compiling(tool_context: Any) -> dict[str, object]:
        _translate(tool_context)
        return {"outcome": Outcome.OK.value}

    tools = _tools_by_name(translate=compiling)
    ctx = _FakeToolContext()

    result = await _propose_translated(tools, ctx, decision(), requirements)

    translated = load_plan(ctx.state)
    assert translated is not None
    assert result["outcome"] == Outcome.OK.value
    assert result["translation"] == {"outcome": Outcome.OK.value}
    assert "submission" not in result
    assert result["summary"] == {
        "version": 1,
        "revision": plan_revision(translated),
        "warnings": [],
    }
    stored = load_plan(ctx.state)
    assert stored is not None
    assert stored.status == PlanStatus.PENDING_APPROVAL
    assert stored.approved_revision is None
    assert ctx.actions.skip_summarization is True


def test_two_plans_differing_only_in_a_reduction_carry_different_revisions(
    decision: DecisionFactory,
    requirement: RequirementFactory,
    derived_requirement: DerivedRequirementFactory,
) -> None:
    # The revision covers each requirement's own fields, so two plans
    # differing only in an `AggregateRequirement.reduction` produce two
    # different revisions, guarding against one approval key covering two
    # different plans.
    summed_ctx = _FakeToolContext()
    averaged_ctx = _FakeToolContext()
    tools = _tools_by_name()
    _propose(
        tools, summed_ctx, decision(), (requirement(), derived_requirement(reduction=Reduction.SUM))
    )
    _translate(summed_ctx, reduction=Reduction.SUM)
    _propose(
        tools,
        averaged_ctx,
        decision(),
        (requirement(), derived_requirement(reduction=Reduction.MEAN)),
    )
    _translate(averaged_ctx, reduction=Reduction.MEAN)

    summed = _declare(summed_ctx)
    averaged = _declare(averaged_ctx)

    assert summed["summary"]["version"] == averaged["summary"]["version"] == 1
    assert summed["summary"]["revision"] != averaged["summary"]["revision"]


def test_a_blocked_plan_is_refused_before_it_reaches_the_summary(
    decision: DecisionFactory, requirements: RequirementPair
) -> None:
    # A draft cannot run, so it is refused before any summary is built --
    # there is no compiled record to show -- and the run goes on.
    tools = _tools_by_name()
    ctx = _FakeToolContext()
    _propose(tools, ctx, decision(), requirements)

    result = _declare(ctx)

    assert result["submission"]["code"] == ProblemCode.WORKFLOW_MISSING.value  # type: ignore[index]
    assert "summary" not in result
    assert not ctx.actions.skip_summarization


def test_the_summary_carries_every_warning_the_compiler_let_through(
    decision: DecisionFactory, requirements: RequirementPair
) -> None:
    # Approving a plan approves its caveats: what the compiler let through
    # reaches the user in the same payload as the calls, not in prose.
    tools = _tools_by_name()
    ctx = _FakeToolContext()
    _propose(tools, ctx, decision(), requirements)
    warning = EvidenceIssue(
        code=EvidenceIssueCode.WINDOW_SHORT,
        severity=Severity.WARNING,
        requirement_id="annual_debt_service",
        detail="the source covers 2015-2022, the requirement asks for 2015-2024",
    )
    _translate(ctx, warnings=[warning])

    result = _declare(ctx)

    assert result["summary"]["warnings"] == [warning.model_dump(mode="json")]


def test_a_rejected_plan_is_never_declared_ready(
    decision: DecisionFactory, requirements: RequirementPair
) -> None:
    tools = _tools_by_name()
    ctx = _FakeToolContext()
    _propose(tools, ctx, decision(), requirements)
    _translate(ctx)
    tools["reject_plan"].func(tool_context=ctx)

    result = _declare(ctx)

    blocked = cast("dict[str, Any]", result["submission"])
    assert blocked["code"] == ProblemCode.PLAN_REJECTED.value
    assert blocked["action"]["call"] == "propose_plan"
    assert "summary" not in result
    # A refusal leaves the run open, so the model reads it and corrects.
    assert not ctx.actions.skip_summarization


@pytest.mark.parametrize(
    "reach",
    [
        lambda plan: plan.approve(at=datetime.now(UTC)),
        lambda plan: plan.approve(at=datetime.now(UTC)).start_execution(),
        lambda plan: plan.approve(at=datetime.now(UTC)).start_execution().complete(),
        lambda plan: plan.approve(at=datetime.now(UTC)).start_execution().fail(),
    ],
    ids=["approved", "executing", "completed", "failed"],
)
def test_a_plan_not_awaiting_approval_is_never_declared_ready(
    reach: Callable[[Plan], Plan], decision: DecisionFactory, requirements: RequirementPair
) -> None:
    # `harness.graph.execute` seeds a `FAILED` plan straight back into the planner after
    # an execution refusal, and that plan still carries a workflow at the
    # version it failed with -- exactly the shape every earlier check in
    # `_execution_blocked` lets through. Declaring it ready unchanged must not
    # succeed: approving what comes back would raise inside `Plan.approve`,
    # which requires `PENDING_APPROVAL`.
    tools = _tools_by_name()
    ctx = _FakeToolContext()
    _propose(tools, ctx, decision(), requirements)
    translated = _translate(ctx)
    advanced = reach(translated)
    save_plan(ctx.state, advanced)

    result = _declare(ctx)

    blocked = cast("dict[str, Any]", result["submission"])
    assert blocked["outcome"] == Outcome.REFUSED.value
    assert blocked["code"] == ProblemCode.TRANSITION_REFUSED.value
    assert blocked["action"]["call"] == "propose_plan"
    assert "summary" not in result
    stored = load_plan(ctx.state)
    assert stored is not None
    assert stored.status == advanced.status


async def test_a_translation_that_did_not_compile_declares_nothing(
    decision: DecisionFactory, requirements: RequirementPair
) -> None:
    # A draft states the methodology; nothing has been compiled to run, so
    # declaring it ready would put nothing real in front of the user. The
    # proposal answers its verdict alone, and the run goes on.
    async def unavailable(tool_context: Any) -> dict[str, object]:
        del tool_context
        return {"outcome": "refused", "code": "surface_unavailable", "message": "m"}

    tools = _tools_by_name(translate=unavailable)
    ctx = _FakeToolContext()

    result = await _propose_translated(tools, ctx, decision(), requirements)

    assert result["translation"] == {
        "outcome": "refused",
        "code": "surface_unavailable",
        "message": "m",
    }
    assert "summary" not in result
    assert "submission" not in result
    assert not ctx.actions.skip_summarization
    stored = load_plan(ctx.state)
    assert stored is not None
    assert stored.status == PlanStatus.DRAFT


def test_a_plan_with_a_pending_referral_is_never_declared_ready(
    decision: DecisionFactory, requirements: RequirementPair
) -> None:
    tools = _tools_by_name()
    ctx = _FakeToolContext()
    _propose(tools, ctx, decision(), requirements)
    _translate(ctx)
    dataset, _derived = requirements
    referral = Referral(
        kind=ReferralKind.NO_DATA, reason="no data found", requirement_id=dataset.id
    )
    save_referral(ctx.state, referral)

    result = _declare(ctx)

    blocked = cast("dict[str, Any]", result["submission"])
    assert blocked["code"] == ProblemCode.REFERRAL_PENDING.value
    assert blocked["action"]["call"] == "propose_plan"
    assert blocked["detail"]["referral"]["kind"] == ReferralKind.NO_DATA.value
    assert "summary" not in result


def _declare_over_a_handed_back_plan(
    decision: DecisionFactory,
    requirements: RequirementPair,
    kind: ReferralKind,
    *,
    answered: bool = False,
) -> dict[str, Any]:
    tools = _tools_by_name()
    ctx = _FakeToolContext()
    _propose(tools, ctx, decision(), requirements)
    _translate(ctx)
    dataset, _derived = requirements
    _hand_back(ctx, kind, dataset.id)
    if answered:
        ctx.session.events.extend(_ask_answered_events())
    return cast("dict[str, Any]", _declare(ctx)["submission"])


@pytest.mark.parametrize("kind", HANDED_BACK_KINDS)
def test_the_gate_sends_the_planner_to_the_user_over_an_unanswered_referral(
    decision: DecisionFactory, requirements: RequirementPair, kind: ReferralKind
) -> None:
    result = _declare_over_a_handed_back_plan(decision, requirements, kind)

    assert result["code"] == ProblemCode.REFERRAL_PENDING.value
    assert result["action"]["call"] == "adk_request_input"


@pytest.mark.parametrize("kind", HANDED_BACK_KINDS)
def test_the_gate_sends_the_planner_to_propose_once_the_user_has_answered(
    decision: DecisionFactory, requirements: RequirementPair, kind: ReferralKind
) -> None:
    result = _declare_over_a_handed_back_plan(decision, requirements, kind, answered=True)

    assert result["code"] == ProblemCode.REFERRAL_PENDING.value
    assert result["action"]["call"] == "propose_plan"


@pytest.mark.parametrize("kind", [kind for kind in ReferralKind if kind not in HANDED_BACK_KINDS])
def test_the_gate_sends_the_planner_to_propose_over_a_referral_it_settles_itself(
    decision: DecisionFactory, requirements: RequirementPair, kind: ReferralKind
) -> None:
    result = _declare_over_a_handed_back_plan(decision, requirements, kind)

    assert result["code"] == ProblemCode.REFERRAL_PENDING.value
    assert result["action"]["call"] == "propose_plan"


@pytest.mark.parametrize(
    ("tool_name", "body_result_key"),
    [
        ("propose_plan", "version"),
        ("reject_plan", "status"),
    ],
)
async def test_the_other_planner_tools_run_through_run_async(
    tool_name: str,
    body_result_key: str,
    decision: DecisionFactory,
    requirements: RequirementPair,
) -> None:
    # No planner tool needs a compiled plan to run its own body, or the
    # planner could never draft a plan on its own.
    # Each tool's own result key is the evidence its body ran.
    tools = _tools_by_name()
    ctx = _FakeToolContext()
    # `reject_plan` needs something to reject before its body can succeed.
    _propose(tools, ctx, decision(), requirements)

    result = await tools[tool_name].run_async(
        args=_tool_args(tool_name, decision(), requirements), tool_context=ctx
    )

    assert body_result_key in result


# --- The status the planner never writes -------------------------------------------


async def test_no_planner_tool_writes_pending_approval(
    decision: DecisionFactory, requirements: RequirementPair
) -> None:
    # Session state has no per-key owner, so this is the invariant a test
    # holds: every planner tool that stores a plan leaves it in a status
    # other than the one that says "ready to approve", which the translator
    # alone writes.
    tools = _tools_by_name()

    ctx = _FakeToolContext()
    _propose(tools, ctx, decision(), requirements)
    proposed = load_plan(ctx.state)
    _translate(ctx)
    _declare(ctx)
    submitted = load_plan(ctx.state)

    rejected_ctx = _FakeToolContext()
    _propose(tools, rejected_ctx, decision(), requirements)
    tools["reject_plan"].func(tool_context=rejected_ctx)
    rejected = load_plan(rejected_ctx.state)

    assert proposed is not None and proposed.status == PlanStatus.DRAFT
    assert submitted is not None and submitted.status == PlanStatus.PENDING_APPROVAL
    assert rejected is not None and rejected.status == PlanStatus.REJECTED


def test_no_planner_code_names_the_transition_that_mints_pending_approval() -> None:
    # The structural fact examples cannot reach: `Plan.with_translation` is
    # the only transition anywhere that produces `PENDING_APPROVAL`, and no
    # code in this module calls it -- an AST walk rather than a substring
    # search, since a comment or docstring may mention it and a
    # `getattr(plan, "with_" + "translation")` may not.
    #
    # `PENDING_APPROVAL` itself is named once, in `_execution_blocked`'s own
    # allowlist (`declare_ready` requires a plan already at that status; see
    # its module docstring) -- a comparison, not a mint, so only a mention
    # outside a `Compare` counts against the invariant.
    tree = ast.parse(inspect.getsource(planner_agent))
    parents: dict[ast.AST, ast.AST] = {
        child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)
    }

    def _names_pending_approval(node: ast.AST) -> bool:
        return (isinstance(node, ast.Attribute) and node.attr == "PENDING_APPROVAL") or (
            isinstance(node, ast.Name) and node.id == "PENDING_APPROVAL"
        )

    called = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    minted = [
        node
        for node in ast.walk(tree)
        if _names_pending_approval(node) and not isinstance(parents.get(node), ast.Compare)
    ]

    assert "with_translation" not in called
    assert minted == []


@pytest.mark.parametrize("tool_name", SIMPLE_TOOLS)
async def test_no_planner_tool_moves_a_draft_to_pending_approval(
    tool_name: str, decision: DecisionFactory, requirements: RequirementPair
) -> None:
    # The behavioural half of the invariant above: every planner tool, run
    # against a draft, leaves the plan anywhere but the status that says
    # "ready to approve".
    tools = _tools_by_name()
    ctx = _FakeToolContext()
    _propose(tools, ctx, decision(), requirements)
    await tools[tool_name].run_async(
        args=_tool_args(tool_name, decision(), requirements), tool_context=ctx
    )

    stored = load_plan(ctx.state)
    assert stored is not None
    assert stored.status is not PlanStatus.PENDING_APPROVAL


# --- Proposing ---------------------------------------------------------------------------


def test_propose_plan_records_a_draft(
    decision: DecisionFactory, requirements: RequirementPair
) -> None:
    tools = _tools_by_name()
    ctx = _FakeToolContext()

    result = _propose(
        tools, ctx, decision(), requirements, question="How dependent is Kenya on Chinese lending?"
    )

    assert result["outcome"] == Outcome.OK.value
    assert result["status"] == PlanStatus.DRAFT.value
    assert result["version"] == 1
    assert PLAN_STATE_KEY in ctx.state
    stored = load_plan(ctx.state)
    assert stored is not None
    assert stored.evidence_requirements[0].id == "annual_debt_service"
    assert stored.workflow is None


def test_propose_plan_answers_with_no_echo_of_the_plan(
    decision: DecisionFactory, requirements: RequirementPair
) -> None:
    """The response carries no copy of what the model just sent or what state holds.

    The plan's `decisions` and `evidence_requirements` are the model's own call
    arguments and are recorded in session state, where translation reads
    them. Returning them here is a second copy that can disagree with the
    first, and a model that reads the function response instead of its own
    call is reading the copy. `status` crosses into a genai FunctionResponse,
    so it is the string, not a `PlanStatus` that only serialises by virtue of
    being a StrEnum.
    """
    tools = _tools_by_name()

    result = _propose(tools, _FakeToolContext(), decision(), requirements)

    assert type(result["status"]) is str
    assert set(result) == {"outcome", "status", "version", "previous_version", "question"}
    assert "decisions" not in json.dumps(result)
    assert "evidence_requirements" not in json.dumps(result)


def test_propose_plan_called_again_amends_rather_than_resets(
    decision: DecisionFactory, requirements: RequirementPair
) -> None:
    tools = _tools_by_name()
    ctx = _FakeToolContext()
    _propose(tools, ctx, decision(), requirements)

    second = _propose(
        tools, ctx, decision(chosen="total debt stock", resolved_by=ResolvedBy.USER), requirements
    )

    assert second["version"] == 2
    assert second["status"] == PlanStatus.DRAFT.value
    assert second["previous_version"] == 1


def test_propose_plan_after_translation_returns_to_a_draft(
    decision: DecisionFactory, requirements: RequirementPair
) -> None:
    # The operations were translated from the previous version's decisions,
    # so the amended plan has to be translated again before anyone approves it.
    tools = _tools_by_name()
    ctx = _FakeToolContext()
    _propose(tools, ctx, decision(), requirements)
    _translate(ctx)

    second = _propose(
        tools, ctx, decision(chosen="total debt stock", resolved_by=ResolvedBy.USER), requirements
    )

    assert second["status"] == PlanStatus.DRAFT.value
    stored = load_plan(ctx.state)
    assert stored is not None
    assert stored.workflow is None
    assert stored.translation is None


def test_propose_plan_for_a_different_question_starts_a_fresh_plan(
    decision: DecisionFactory,
    requirement: RequirementFactory,
    derived_requirement: DerivedRequirementFactory,
    requirements: RequirementPair,
) -> None:
    tools = _tools_by_name()
    ctx = _FakeToolContext()
    _propose(
        tools, ctx, decision(), requirements, question="How dependent is Kenya on Chinese lending?"
    )

    # A genuinely different question, not a revision of the first -- this
    # must not amend the first plan onto the second question's methodology.
    # The call says so with `starts_new_plan`. Its wording alone decides
    # nothing.
    second = _propose(
        tools,
        ctx,
        decision(slot="measure", options=["total ODA", "bilateral ODA"], chosen="total ODA"),
        (
            requirement(id="total_oda", measure="total ODA received"),
            derived_requirement(id="total_oda_totals", input_id="total_oda"),
        ),
        question="What was Kenya's total ODA in 2023?",
        starts_new_plan=True,
    )

    assert second["version"] == 1
    assert second["previous_version"] is None
    assert second["question"] == "What was Kenya's total ODA in 2023?"


def test_a_defaulted_time_window_needs_no_period_decision(
    decision: DecisionFactory,
    requirement: RequirementFactory,
    derived_requirement: DerivedRequirementFactory,
) -> None:
    # A question with no time text resolves across every period with data
    # (the envelope's `time_window_defaulted` warning). When the resolver's
    # identity states a full `range` for that coverage, `graph_envelope`
    # carries it forward as `window` (`test_tool_contracts.py` has the
    # parsing). The planner copies that verbatim into the dataset
    # requirement's own `period` and cites no decision that fixes it, since
    # nothing was chosen between alternative ranges.
    tools = _tools_by_name()
    ctx = _FakeToolContext()
    dataset = requirement(period=Period(start=1960, end=2023))

    result = _propose(tools, ctx, decision(), (dataset, derived_requirement()))

    assert result["outcome"] == Outcome.OK.value
    stored = load_plan(ctx.state)
    assert stored is not None
    (stored_dataset,) = [r for r in stored.evidence_requirements if is_dataset(r)]
    assert stored_dataset.period == Period(start=1960, end=2023)
    assert all(d.fixes is None or d.fixes.period is None for d in stored.decisions)


def test_a_reworded_question_with_the_amendment_stated_still_patches(
    decision: DecisionFactory, requirements: RequirementPair
) -> None:
    """The regression the explicit signal exists for.

    A model that means to amend and rewords the `question` while it does so
    must still patch. The call's own `amends_version` says which it is, and
    the wording decides nothing. Without that, the reworded question read as
    a fresh draft and every item the patch did not restate was dropped. A
    self-consistent partial patch validated, so nothing reported the loss.
    """
    tools = _tools_by_name()
    ctx = _FakeToolContext()
    _propose(
        tools, ctx, decision(), requirements, question="How dependent is Kenya on Chinese lending?"
    )

    patched = _propose(
        tools,
        ctx,
        decision(chosen="total debt stock", resolved_by=ResolvedBy.USER),
        (),
        question="How dependent is Kenya on lending from China?",
    )

    assert patched["outcome"] == Outcome.OK.value, patched
    assert patched["version"] == 2
    assert patched["previous_version"] == 1
    stored = load_plan(ctx.state)
    assert stored is not None
    # The patch restated no requirement, and every one is still there.
    assert [r.id for r in stored.evidence_requirements] == [r.id for r in requirements]
    assert [d.slot for d in stored.decisions] == ["measure"]
    assert stored.decisions[0].chosen == "total debt stock"


def test_an_amendment_records_the_question_it_states(
    decision: DecisionFactory, requirements: RequirementPair
) -> None:
    """The stored question follows the amendment that rewords it.

    The planner reads the recorded question back next to the requirements it
    is about to patch, so a question left at its first wording contradicts an
    amendment that moved the period. The same reworded question with no
    `amends_version` is refused and records nothing.
    """
    tools = _tools_by_name()
    ctx = _FakeToolContext()
    _propose(tools, ctx, decision(), requirements, question="Kenya's debt service since 1990")

    unstated = tools["propose_plan"].func(
        question="Kenya's debt service, 2000 to 2023",
        decisions=drafts([decision(rationale="Reconsidered.")]),
        tool_context=ctx,
    )
    amended = _propose(
        tools,
        ctx,
        decision(rationale="Reconsidered."),
        (),
        question="Kenya's debt service, 2000 to 2023",
    )

    assert unstated["code"] == ProblemCode.INVALID_PLAN.value
    assert amended["outcome"] == Outcome.OK.value, amended
    assert (amended["version"], amended["previous_version"]) == (2, 1)
    assert amended["question"] == "Kenya's debt service, 2000 to 2023"
    stored = load_plan(ctx.state)
    assert stored is not None
    assert stored.question == "Kenya's debt service, 2000 to 2023"
    assert [r.id for r in stored.evidence_requirements] == [r.id for r in requirements]


def test_an_amendment_that_states_no_question_keeps_the_recorded_one(
    decision: DecisionFactory, requirements: RequirementPair
) -> None:
    """`question` follows the amend contract. A patch states only what changes."""
    tools = _tools_by_name()
    ctx = _FakeToolContext()
    _propose(tools, ctx, decision(), requirements, question="Kenya's debt service since 1990")

    amended = tools["propose_plan"].func(
        decisions=drafts([decision(rationale="Reconsidered.")]), amends_version=1, tool_context=ctx
    )

    assert amended["outcome"] == Outcome.OK.value, amended
    assert (amended["version"], amended["previous_version"]) == (2, 1)
    assert amended["question"] == "Kenya's debt service since 1990"
    stored = load_plan(ctx.state)
    assert stored is not None
    assert stored.question == "Kenya's debt service since 1990"


def test_an_amendment_that_states_no_answer_mode_keeps_the_recorded_one(
    decision: DecisionFactory, requirements: RequirementPair
) -> None:
    """`answer_mode` follows the same amend contract as `question`."""
    tools = _tools_by_name()
    ctx = _FakeToolContext()
    _propose(tools, ctx, decision(), requirements, answer_mode=AnswerMode.SERIES)

    amended = tools["propose_plan"].func(
        decisions=drafts([decision(rationale="Reconsidered.")]), amends_version=1, tool_context=ctx
    )

    assert amended["outcome"] == Outcome.OK.value, amended
    stored = load_plan(ctx.state)
    assert stored is not None
    assert stored.answer_mode == AnswerMode.SERIES


def test_a_first_draft_without_a_question_is_refused_and_records_nothing(
    decision: DecisionFactory, requirements: RequirementPair
) -> None:
    tools = _tools_by_name()
    ctx = _FakeToolContext()

    refused = tools["propose_plan"].func(
        decisions=drafts([decision()]), evidence_requirements=list(requirements), tool_context=ctx
    )

    assert refused["code"] == ProblemCode.INVALID_PLAN.value
    assert refused["outcome"] == Outcome.INVALID_INPUT.value
    assert "first draft" in str(refused["message"])
    assert "starts_new_plan" not in str(refused["message"])
    assert PLAN_STATE_KEY not in ctx.state


@pytest.mark.parametrize(
    ("role", "meant"), [("origin", "donor"), ("Lenders", "creditor"), ("country", None)]
)
def test_an_entity_role_outside_the_party_words_is_refused_and_records_nothing(
    decision: DecisionFactory,
    requirement: RequirementFactory,
    derived_requirement: DerivedRequirementFactory,
    role: str,
    meant: str | None,
) -> None:
    tools = _tools_by_name()
    ctx = _FakeToolContext()

    refused = tools["propose_plan"].func(
        question="q",
        answer_mode=AnswerMode.SERIES,
        decisions=drafts([decision()]),
        evidence_requirements=[
            requirement(entities=[Entity(name="Kenya", role=role)]),
            derived_requirement(),
        ],
        tool_context=ctx,
    )

    assert refused["code"] == ProblemCode.INVALID_PLAN.value
    assert (f"write {meant!r}" in str(refused["message"])) is (meant is not None)
    assert PLAN_STATE_KEY not in ctx.state


def test_a_restart_without_a_question_is_refused_and_leaves_the_plan_alone(
    decision: DecisionFactory, requirements: RequirementPair
) -> None:
    tools = _tools_by_name()
    ctx = _FakeToolContext()
    _propose(tools, ctx, decision(), requirements)

    refused = tools["propose_plan"].func(
        decisions=drafts([decision()]),
        evidence_requirements=list(requirements),
        starts_new_plan=True,
        tool_context=ctx,
    )

    assert refused["code"] == ProblemCode.INVALID_PLAN.value
    assert "starts_new_plan" in str(refused["message"])
    stored = load_plan(ctx.state)
    assert stored is not None
    assert (stored.question, stored.version) == ("q", 1)
    assert load_plan_rounds(ctx.state) == 1


def test_a_first_draft_without_an_answer_mode_is_refused_and_records_nothing(
    decision: DecisionFactory, requirements: RequirementPair
) -> None:
    tools = _tools_by_name()
    ctx = _FakeToolContext()

    refused = tools["propose_plan"].func(
        question="q",
        decisions=drafts([decision()]),
        evidence_requirements=list(requirements),
        tool_context=ctx,
    )

    assert refused["code"] == ProblemCode.INVALID_PLAN.value
    assert refused["outcome"] == Outcome.INVALID_INPUT.value
    assert "answer_mode" in str(refused["message"])
    assert PLAN_STATE_KEY not in ctx.state


def test_a_restart_without_an_answer_mode_is_refused_and_leaves_the_plan_alone(
    decision: DecisionFactory, requirements: RequirementPair
) -> None:
    tools = _tools_by_name()
    ctx = _FakeToolContext()
    _propose(tools, ctx, decision(), requirements)

    refused = tools["propose_plan"].func(
        question="q",
        decisions=drafts([decision()]),
        evidence_requirements=list(requirements),
        starts_new_plan=True,
        tool_context=ctx,
    )

    assert refused["code"] == ProblemCode.INVALID_PLAN.value
    assert "starts_new_plan" in str(refused["message"])
    assert "answer_mode" in str(refused["message"])
    stored = load_plan(ctx.state)
    assert stored is not None
    assert (stored.question, stored.version) == ("q", 1)
    assert load_plan_rounds(ctx.state) == 1


def test_propose_plan_counts_the_rounds_a_question_has_been_through(
    decision: DecisionFactory, requirements: RequirementPair
) -> None:
    tools = _tools_by_name()
    ctx = _FakeToolContext()

    _propose(tools, ctx, decision(), requirements)
    first = load_plan_rounds(ctx.state)
    _propose(tools, ctx, decision(), requirements)
    second = load_plan_rounds(ctx.state)

    assert (first, second) == (1, 2)
    assert PLAN_ROUNDS_STATE_KEY in ctx.state


def _propose_up_to_the_cap(
    tools: dict[str, FunctionTool],
    ctx: object,
    decision: Decision,
    requirements: RequirementPair,
) -> dict[str, object]:
    """Propose `MAX_PLAN_ROUNDS` versions of question `q`; return the last result."""
    for _ in range(MAX_PLAN_ROUNDS):
        result = _propose(tools, ctx, decision, requirements)
    return result


def test_propose_plan_refuses_a_round_past_the_cap_and_touches_nothing(
    decision: DecisionFactory, requirements: RequirementPair
) -> None:
    # Every version opens a fresh discovery budget, so a refer-and-repropose
    # loop has to end somewhere. Past the cap nothing is recorded: the stored
    # plan, its round count and the pending referral all stay as they were,
    # and no `action` is named, since any tool named here restarts the loop.
    tools = _tools_by_name()
    ctx = _FakeToolContext()
    last = _propose_up_to_the_cap(tools, ctx, decision(), requirements)
    dataset, _derived = requirements
    save_referral(
        ctx.state,
        Referral(kind=ReferralKind.NO_DATA, reason="no data found", requirement_id=dataset.id),
    )

    result = _propose(tools, ctx, decision(), requirements)

    assert last["version"] == MAX_PLAN_ROUNDS
    assert result["outcome"] == Outcome.REFUSED.value
    assert result["code"] == ProblemCode.BUDGET_EXHAUSTED.value
    assert result["detail"] == {
        "scope": "plan_rounds",
        "rounds": MAX_PLAN_ROUNDS,
        "version": MAX_PLAN_ROUNDS,
    }
    assert "action" not in result
    stored = load_plan(ctx.state)
    assert stored is not None
    assert stored.version == MAX_PLAN_ROUNDS
    assert load_plan_rounds(ctx.state) == MAX_PLAN_ROUNDS
    assert load_referral(ctx.state) is not None


def test_the_last_version_that_compiled_stays_declarable_past_the_cap(
    decision: DecisionFactory, requirements: RequirementPair
) -> None:
    tools = _tools_by_name()
    ctx = _FakeToolContext()
    _propose_up_to_the_cap(tools, ctx, decision(), requirements)
    _translate(ctx)
    refused = _propose(tools, ctx, decision(), requirements)

    result = _declare(ctx)

    assert refused["code"] == ProblemCode.BUDGET_EXHAUSTED.value
    assert cast("dict[str, Any]", result["summary"])["version"] == MAX_PLAN_ROUNDS


def test_a_fresh_entry_into_the_planner_starts_its_rounds_afresh(
    decision: DecisionFactory,
    requirement: RequirementFactory,
    derived_requirement: DerivedRequirementFactory,
    requirements: RequirementPair,
) -> None:
    # The allowance is per entry into the planner, so a spent one does not stop
    # the user asking something else. `PlannerSeam.__call__` zeroes the count
    # on that entry (`save_plan_rounds` below stands in for it), and the new
    # question gets a full allowance.
    tools = _tools_by_name()
    ctx = _FakeToolContext()
    _propose_up_to_the_cap(tools, ctx, decision(), requirements)
    save_plan_rounds(ctx.state, 0)
    oda = (
        requirement(id="total_oda", measure="total ODA received"),
        derived_requirement(id="total_oda_totals", input_id="total_oda"),
    )
    oda_decision = decision(
        slot="measure", options=["total ODA", "bilateral ODA"], chosen="total ODA"
    )

    first = _propose(
        tools,
        ctx,
        oda_decision,
        oda,
        question="What was Kenya's ODA in 2023?",
        starts_new_plan=True,
    )
    second = _propose(tools, ctx, oda_decision, oda, question="What was Kenya's ODA in 2023?")

    assert (first["version"], second["version"]) == (1, 2)
    assert load_plan_rounds(ctx.state) == 2


def _restart(
    tools: dict[str, FunctionTool],
    ctx: object,
    decision: Decision,
    requirements: RequirementPair,
    *,
    question: str,
) -> dict[str, object]:
    """One `propose_plan` call that states `starts_new_plan` and no amendment."""
    return _propose(tools, ctx, decision, requirements, question=question, starts_new_plan=True)


def test_the_fifth_call_is_refused_when_the_first_four_were_a_draft_and_three_restarts(
    decision: DecisionFactory, requirements: RequirementPair
) -> None:
    """`starts_new_plan` spends a proposal like an amendment does.

    A first draft and `MAX_PLAN_ROUNDS - 1` restarts use the whole allowance
    (calls one to four), so the next restart, the fifth call, is refused with
    nothing recorded. Restarts share the allowance with amendments, so a model
    cannot restart without bound inside one entry into the planner.
    """
    tools = _tools_by_name()
    ctx = _FakeToolContext()
    _propose(tools, ctx, decision(), requirements, question="Q0")
    for n in range(1, MAX_PLAN_ROUNDS):
        recorded = _restart(tools, ctx, decision(), requirements, question=f"Q{n}")
        assert recorded["outcome"] == Outcome.OK.value, recorded
        assert recorded["version"] == 1

    refused = _restart(tools, ctx, decision(), requirements, question="Q-refused")

    assert refused["outcome"] == Outcome.REFUSED.value
    assert refused["code"] == ProblemCode.BUDGET_EXHAUSTED.value
    assert refused["detail"] == {"scope": "plan_rounds", "rounds": MAX_PLAN_ROUNDS, "version": 1}
    assert "action" not in refused
    stored = load_plan(ctx.state)
    assert stored is not None
    assert stored.question == f"Q{MAX_PLAN_ROUNDS - 1}"
    assert load_plan_rounds(ctx.state) == MAX_PLAN_ROUNDS


def test_a_refused_restart_says_what_to_do_instead_of_asking_for_a_new_question(
    decision: DecisionFactory, requirements: RequirementPair
) -> None:
    # A call that has just claimed to be a new question is told what remains
    # possible (approve the last version that compiled, or wait for the next
    # message). Telling it to ask a new question reads as an instruction to
    # make the same call again.
    tools = _tools_by_name()
    ctx = _FakeToolContext()
    _propose_up_to_the_cap(tools, ctx, decision(), requirements)
    _translate(ctx)

    refused = _restart(tools, ctx, decision(), requirements, question="Another question")

    message = str(refused["message"])
    assert "restart" in message
    assert "new question" not in message
    assert "last version that compiled can still be approved" in message
    assert "fresh budget" in message


def test_the_refusal_at_the_cap_offers_an_approval_only_where_a_plan_compiled(
    decision: DecisionFactory, requirements: RequirementPair
) -> None:
    """Nothing compiled means no plan card exists, so nothing is approvable."""
    tools = _tools_by_name()
    ctx = _FakeToolContext()
    _propose_up_to_the_cap(tools, ctx, decision(), requirements)

    uncompiled_amendment = _propose(tools, ctx, decision(), requirements)
    uncompiled_restart = _restart(tools, ctx, decision(), requirements, question="Another")
    _translate(ctx)
    compiled_amendment = _propose(tools, ctx, decision(), requirements)
    compiled_restart = _restart(tools, ctx, decision(), requirements, question="Another")

    for refused in (uncompiled_amendment, uncompiled_restart):
        assert refused["code"] == ProblemCode.BUDGET_EXHAUSTED.value
        assert "approved" not in str(refused["message"])
        assert "fresh budget" in str(refused["message"])
    for refused in (compiled_amendment, compiled_restart):
        assert refused["code"] == ProblemCode.BUDGET_EXHAUSTED.value
        assert "last version that compiled can still be approved" in str(refused["message"])
        assert "fresh budget" in str(refused["message"])


def test_the_refusal_at_the_cap_counts_proposals_in_this_run_not_versions_of_the_question(
    decision: DecisionFactory, requirements: RequirementPair
) -> None:
    # A replan or an amendment re-enters the planner with a fresh count while
    # the plan's version keeps climbing, so a plan at version six can be at
    # its cap of four. The text names the cap as per planner run and says a
    # fresh budget follows.
    tools = _tools_by_name()
    ctx = _FakeToolContext()
    _propose_up_to_the_cap(tools, ctx, decision(), requirements)

    refused = _propose(tools, ctx, decision(), requirements)

    message = str(refused["message"])
    assert "planner run" in message
    assert "This question has been through" not in message
    assert "new question" not in message
    assert "fresh budget" in message


def test_an_amendment_between_restarts_still_counts_against_the_cap(
    decision: DecisionFactory, requirements: RequirementPair
) -> None:
    # Draft, restart, amendment, restart: four proposals. The fifth call is
    # refused whichever way it is stated.
    tools = _tools_by_name()
    ctx = _FakeToolContext()
    _propose(tools, ctx, decision(), requirements, question="Q0")
    _restart(tools, ctx, decision(), requirements, question="Q1")
    amended = _propose(tools, ctx, decision(rationale="Reconsidered."), requirements)
    _restart(tools, ctx, decision(), requirements, question="Q2")

    restart = _restart(tools, ctx, decision(), requirements, question="Q3")
    amendment = _propose(tools, ctx, decision(), requirements)

    assert (amended["outcome"], amended["version"]) == (Outcome.OK.value, 2)
    assert restart["code"] == ProblemCode.BUDGET_EXHAUSTED.value
    assert amendment["code"] == ProblemCode.BUDGET_EXHAUSTED.value
    assert load_plan_rounds(ctx.state) == MAX_PLAN_ROUNDS


def test_a_restart_inside_the_cap_still_begins_version_one_under_the_question_it_states(
    decision: DecisionFactory, requirements: RequirementPair
) -> None:
    tools = _tools_by_name()
    ctx = _FakeToolContext()
    _propose(tools, ctx, decision(), requirements, question="How dependent is Kenya on lending?")

    restarted = _restart(
        tools, ctx, decision(), requirements, question="What was Kenya's ODA in 2023?"
    )

    assert restarted["outcome"] == Outcome.OK.value, restarted
    assert restarted["version"] == 1
    assert restarted["previous_version"] is None
    assert restarted["question"] == "What was Kenya's ODA in 2023?"
    assert load_plan_rounds(ctx.state) == 2


def test_a_first_draft_is_never_refused_by_the_round_count() -> None:
    # Nothing recorded means nothing to count against, whatever the count
    # reads and however the call is stated.
    for starts_new_plan in (False, True):
        resolved = _resolve_intent(
            None,
            amends_version=None,
            starts_new_plan=starts_new_plan,
            rounds_so_far=MAX_PLAN_ROUNDS,
        )

        assert resolved == (None, 1)


# The Ghana turn: net ODA to Ghana summed over its providers, added to Ghana's
# net flows on external debt, each side pinned to a source of its own dataset.
# Each source's identity is over every year.
_GHANA_AID_IDENTITY: dict[str, Any] = {
    "identity_version": "spec1",
    "dataset": "net ODA to Ghana",
    "requested_window": {"kind": "all"},
}
_GHANA_DEBT_IDENTITY: dict[str, Any] = {**_GHANA_AID_IDENTITY, "dataset": "Ghana's net debt flows"}
SPEC_GHANA_AID = spec_id_for(canonicalize(_GHANA_AID_IDENTITY))
SPEC_GHANA_DEBT = spec_id_for(canonicalize(_GHANA_DEBT_IDENTITY))
CRS = "provenance/OECDCreditorReportingSystem"
IMPUTED = "provenance/ONEImputedMultilateralSectors"
DAC2A = "provenance/OECDDAC2A"
IDS = "provenance/WorldBankInternationalDebtStatistics"
# A provider's flows table Search may serve beyond today's declared ones.
NEW_FLOWS = "provenance/NewProviderFlows"
# The account Search serves for each table's provenance; IDS states none.
_ACCOUNTS = dict.fromkeys((CRS, IMPUTED, DAC2A, NEW_FLOWS), PROVIDER_ACCOUNT)


def _hold_ghana_sources(ctx: _FakeToolContext, *, datasets: tuple[str, str], held_as: str) -> None:
    """Put the two Ghana sources, of `datasets` in turn, where `held_as` says the turn holds them.

    `shown`: blocks the planner was shown (`pin_sources`). `found`: sources the
    translator's search recorded, as a referral's candidates are.
    """
    held = [
        SelectedSource(
            spec_id=spec_id,
            sentence=spec_id,
            has_data=True,
            identity=identity,
            provenance_ref=dataset,
            provenance_account=_ACCOUNTS.get(dataset),
        )
        for spec_id, identity, dataset in zip(
            (SPEC_GHANA_AID, SPEC_GHANA_DEBT),
            (_GHANA_AID_IDENTITY, _GHANA_DEBT_IDENTITY),
            datasets,
            strict=True,
        )
    ]
    if held_as == "shown":
        save_pin_sources(
            ctx.state,
            {
                source.spec_id: PinSource(
                    block={
                        "spec_id": source.spec_id,
                        "resolved": {
                            "rendered_sentence": source.sentence,
                            "facets": {
                                "provenance": {
                                    "ref": source.provenance_ref,
                                    "account": source.provenance_account,
                                }
                            },
                        },
                    },
                    settled=True,
                )
                for source in held
            },
        )
    else:
        log = load_log(ctx.state)
        for requirement_id, source in zip(("aid", "debt"), held, strict=True):
            found = DiscoveryRecord(
                revision="earlier",
                tool="search",
                requirement_id=requirement_id,
                requirement_key="earlier",
                status=DiscoveryStatus.DEFINITE,
                spec_ids=(source.spec_id,),
            )
            log = log.with_record(found, (source,))
        save_log(ctx.state, log)


def _sum_of_two_sources(
    ctx: _FakeToolContext,
    *,
    datasets: tuple[str, str],
    resolved_by: ResolvedBy,
    decision: DecisionFactory,
    pinning_decision: Callable[..., Decision],
    requirement: RequirementFactory,
    held_as: str = "shown",
) -> Plan:
    """A plan summing two pinned sources, of `datasets` in turn, held in `ctx` as `held_as` says."""
    _hold_ghana_sources(ctx, datasets=datasets, held_as=held_as)
    ghana = [Entity(name="Ghana", role="recipient")]
    return Plan(
        question="Net flows to Ghana combining aid and debt",
        answer_mode=AnswerMode.COMPUTED,
        decisions=[
            pinning_decision(
                SPEC_GHANA_AID, slot="aid_source", resolved_by=ResolvedBy.PLANNER_DEFAULT
            ),
            pinning_decision(
                SPEC_GHANA_DEBT, slot="debt_source", resolved_by=ResolvedBy.PLANNER_DEFAULT
            ),
            decision(
                slot="combine_aid_debt",
                description="How to combine aid and debt flows.",
                options=["add them", "show them separately"],
                chosen="add them",
                resolved_by=resolved_by,
            ),
        ],
        evidence_requirements=[
            requirement(
                id="aid",
                decision_ids=["aid_source"],
                measure="net ODA",
                entities=ghana,
                across=("donor",),
                source_spec_id=SPEC_GHANA_AID,
            ),
            AggregateRequirement(
                id="aid_total",
                input_id="aid",
                reduction=Reduction.SUM,
                group_by=(GroupBy.PERIOD,),
            ),
            requirement(
                id="debt",
                decision_ids=["debt_source"],
                measure="net flows on external debt",
                entities=ghana,
                source_spec_id=SPEC_GHANA_DEBT,
            ),
            SumOfRequirement(
                id="net_flows",
                decision_ids=["combine_aid_debt"],
                left=RequirementSide(input_id="aid_total"),
                right=RequirementSide(input_id="debt"),
            ),
        ],
    )


def _shown_readings(ctx: _FakeToolContext) -> SourceReadings:
    """The readings of the blocks `ctx` shows the planner, as one proposal reads them."""
    return SourceReadings.of_shown(pin_sources(ctx.state))


@pytest.mark.parametrize(
    ("datasets", "resolved_by", "refused"),
    [
        ((DAC2A, IDS), ResolvedBy.PLANNER_DEFAULT, True),
        ((DAC2A, IDS), ResolvedBy.USER, False),
        ((CRS, IMPUTED), ResolvedBy.PLANNER_DEFAULT, False),
        ((CRS, NEW_FLOWS), ResolvedBy.PLANNER_DEFAULT, False),
        ((IDS, IDS), ResolvedBy.PLANNER_DEFAULT, False),
    ],
    ids=[
        "aid-and-debt-by-default",
        "aid-and-debt-by-the-user",
        "two-aid-tables",
        "two-provider-flow-tables",
        "one-dataset",
    ],
)
def test_a_sum_of_two_datasets_needs_the_users_decision(
    decision: DecisionFactory,
    pinning_decision: Callable[..., Decision],
    requirement: RequirementFactory,
    datasets: tuple[str, str],
    resolved_by: ResolvedBy,
    refused: bool,
) -> None:
    # Tables Search serves with the provider account add up as their skill
    # says, so a sum of two of them is the planner's to record; any other pair
    # is the user's to decide. The user picked adding them in every row.
    ctx = _FakeToolContext()
    ctx.session.events.extend(_combination_asked({"spec_id": COMBINE_OPTION}))
    plan = _sum_of_two_sources(
        ctx,
        datasets=datasets,
        resolved_by=resolved_by,
        decision=decision,
        pinning_decision=pinning_decision,
        requirement=requirement,
    )

    result = _combination_unasked(
        plan, ToolContextSession(ctx), previous=None, sources=_shown_readings(ctx)
    )

    if refused:
        assert result is not None
        assert result["code"] == ProblemCode.COMBINATION_UNASKED.value
        assert result["action"]["call"] == "adk_request_input"
        assert result["action"]["arguments"] == {
            "options": [dict(option) for option in COMBINATION_OPTIONS]
        }
    else:
        assert result is None


@pytest.mark.parametrize("held_as", ["shown", "found"])
@pytest.mark.parametrize(
    ("datasets", "refused"),
    [((CRS, IMPUTED), False), ((DAC2A, IDS), True)],
    ids=["two-aid-tables", "aid-and-debt"],
)
def test_propose_plan_reads_the_dataset_of_every_pin_it_sums(
    decision: DecisionFactory,
    pinning_decision: Callable[..., Decision],
    requirement: RequirementFactory,
    datasets: tuple[str, str],
    refused: bool,
    held_as: str,
) -> None:
    # A source the translator found is in no store the planner was shown; the
    # gate reads each pin's dataset all the same.
    tools = _tools_by_name()
    ctx = _FakeToolContext()
    plan = _sum_of_two_sources(
        ctx,
        datasets=datasets,
        resolved_by=ResolvedBy.PLANNER_DEFAULT,
        decision=decision,
        pinning_decision=pinning_decision,
        requirement=requirement,
        held_as=held_as,
    )

    result = tools["propose_plan"].func(
        question=plan.question,
        answer_mode=plan.answer_mode,
        decisions=drafts(plan.decisions),
        evidence_requirements=list(plan.evidence_requirements),
        tool_context=ctx,
    )

    if refused:
        assert result["code"] == ProblemCode.COMBINATION_UNASKED.value, result
    else:
        assert result["outcome"] == Outcome.OK.value, result
        stored = load_plan(ctx.state)
        assert stored is not None
        aid = stored.requirement("aid")
        assert isinstance(aid, DatasetRequirement) and aid.source_spec_id is not None
        if held_as == "shown":
            assert aid.source_spec_id == SPEC_GHANA_AID
        else:
            # A found source's identity is on record, so the pin moves to its copy
            # over the requirement's years.
            copy = windowed_pins(ctx.state).get(aid.source_spec_id)
            assert copy is not None and copy.windowed_from == SPEC_GHANA_AID


# The datasets `_sum_of_two_sources` adds by default, as a `combination_unasked`
# refusal's `detail` lists them.
_AID_AND_DEBT = [DAC2A, IDS]


def _combination_asked(
    *answers: Any,
    options: list[dict[str, str]] | None = None,
    call_id: str = "ask-1",
    about: tuple[str, list[str]] | None = ("net_flows", _AID_AND_DEBT),
) -> list[Any]:
    """A `combination_unasked` refusal naming the sum `about` (its requirement id and
    datasets; `None` for no refusal), then an `adk_request_input` call offering
    `options`, `COMBINATION_OPTIONS` by default, and one response to it per answer in
    `answers`."""
    offered = [dict(option) for option in COMBINATION_OPTIONS] if options is None else options
    refusals = (
        []
        if about is None
        else [
            response_event(
                "propose_plan",
                {
                    "outcome": Outcome.REFUSED.value,
                    "code": ProblemCode.COMBINATION_UNASKED.value,
                    "message": "m",
                    "detail": {"requirement_id": about[0], "datasets": about[1]},
                },
            )
        ]
    )
    call = SimpleNamespace(
        id=call_id,
        name="adk_request_input",
        args={"message": "Add them, or show them separately?", "options": offered},
    )
    asked = SimpleNamespace(
        content=SimpleNamespace(parts=[SimpleNamespace(function_call=call, function_response=None)])
    )
    return [
        *refusals,
        asked,
        *(_ask_response(call_id, {"result": answer}) for answer in answers),
    ]


@pytest.mark.parametrize(
    ("asks", "traced"),
    [
        ([], False),
        (_ask_answered_events("add them"), False),
        (_combination_asked({"text": "separately, please"}), False),
        (_combination_asked({"text": "add them"}), False),
        (_combination_asked({"spec_id": "add them"}), True),
        (_combination_asked({"spec_id": "show them separately"}), False),
        (_combination_asked({"spec_id": "add them"}, {"spec_id": "show them separately"}), False),
        (
            _combination_asked(
                {"spec_id": "add them"},
                options=[{"id": "add them", "label": "x"}, {"id": "no", "label": "y"}],
            ),
            False,
        ),
        (_combination_asked({"spec_id": "add them"}, about=None), False),
        (_combination_asked({"spec_id": "add them"}, about=("other_sum", _AID_AND_DEBT)), False),
        (_combination_asked({"spec_id": "add them"}, about=("net_flows", [DAC2A, CRS])), False),
        (
            [
                *_combination_asked({"spec_id": "show them separately"}),
                *_combination_asked({"spec_id": "add them"}, about=None, call_id="ask-2"),
            ],
            False,
        ),
    ],
    ids=[
        "unasked",
        "free-text-ask-answered-with-its-choice",
        "paraphrase-picking-the-other-option",
        "typed-in-place-of-a-pick",
        "picked-its-choice",
        "picked-the-other-option",
        "picked-then-changed-their-mind",
        "asked-with-other-options",
        "asked-with-no-refusal-before-it",
        "picked-for-another-sum",
        "picked-for-the-sum-over-other-datasets",
        "second-ask-after-one-refusal",
    ],
)
def test_a_users_decision_to_combine_two_datasets_needs_their_pick(
    decision: DecisionFactory,
    pinning_decision: Callable[..., Decision],
    requirement: RequirementFactory,
    asks: list[Any],
    traced: bool,
) -> None:
    # A `user` decision is the model's own claim, so the one a `sum_of` cites
    # holds only once the user has picked adding the two from the options a
    # combination ask offers.
    ctx = _FakeToolContext()
    ctx.session.events.extend(asks)
    plan = _sum_of_two_sources(
        ctx,
        datasets=(DAC2A, IDS),
        resolved_by=ResolvedBy.USER,
        decision=decision,
        pinning_decision=pinning_decision,
        requirement=requirement,
    )

    result = _combination_unasked(
        plan, ToolContextSession(ctx), previous=None, sources=_shown_readings(ctx)
    )

    if traced:
        assert result is None
    else:
        assert result is not None
        assert result["code"] == ProblemCode.COMBINATION_UNASKED.value
        assert result["detail"] == {"requirement_id": "net_flows", "datasets": _AID_AND_DEBT}


def test_a_pick_to_add_two_datasets_settles_nothing_for_a_sum_over_others(
    decision: DecisionFactory,
    pinning_decision: Callable[..., Decision],
    requirement: RequirementFactory,
) -> None:
    # The user agreed to add aid and debt; the same sum over aid and another
    # dataset nobody declared is a combination they were never asked about.
    other = "provenance/IMFWorldEconomicOutlook"
    ctx = _FakeToolContext()
    ctx.session.events.extend(_combination_asked({"spec_id": COMBINE_OPTION}))
    plan = _sum_of_two_sources(
        ctx,
        datasets=(DAC2A, other),
        resolved_by=ResolvedBy.USER,
        decision=decision,
        pinning_decision=pinning_decision,
        requirement=requirement,
    )

    result = _combination_unasked(
        plan, ToolContextSession(ctx), previous=None, sources=_shown_readings(ctx)
    )

    assert result is not None
    assert result["code"] == ProblemCode.COMBINATION_UNASKED.value
    assert result["detail"] == {"requirement_id": "net_flows", "datasets": sorted([DAC2A, other])}


def _with_debt_side(
    plan: Plan, *, requirement_id: str, spec_id: str | None, measure: str | None = None
) -> Plan:
    """`plan` with its `debt` requirement renamed `requirement_id`, pinned to `spec_id`.

    A `measure` replaces what the requirement asks for.
    """

    def moved(r: EvidenceRequirement) -> EvidenceRequirement:
        if r.id == "debt":
            update = {"id": requirement_id, "source_spec_id": spec_id}
            if measure is not None:
                update["measure"] = measure
            return r.model_copy(update=update)
        if isinstance(r, SumOfRequirement):
            return r.model_copy(update={"right": RequirementSide(input_id=requirement_id)})
        return r

    return plan.model_copy(
        update={"evidence_requirements": tuple(moved(r) for r in plan.evidence_requirements)}
    )


@pytest.mark.parametrize(
    ("asked", "stated", "measure", "route", "refused"),
    [
        (("debt", None), ("debt", None), None, "pick", False),
        (("debt", None), ("other_debt", None), None, "pick", True),
        (("debt", None), ("debt", None), "debt service paid", "pick", True),
        (("debt", "spec-unread-a"), ("debt", "spec-unread-a"), None, "pick", False),
        (("debt", "spec-unread-a"), ("debt", "spec-unread-b"), None, "pick", True),
        (("debt", "spec-unread-a"), ("debt", "spec-unread-a"), None, "carry", False),
        (("debt", "spec-unread-a"), ("debt", "spec-unread-b"), None, "carry", True),
    ],
    ids=[
        "picked-for-this-unpinned-side",
        "picked-for-another-unpinned-side",
        "picked-for-this-unpinned-side-asking-for-another-measure",
        "picked-for-this-unreadable-pin",
        "picked-for-another-unreadable-pin",
        "carried-over-this-unreadable-pin",
        "carried-over-another-unreadable-pin",
    ],
)
def test_a_side_no_pin_names_a_dataset_for_is_a_dataset_of_its_own(
    decision: DecisionFactory,
    pinning_decision: Callable[..., Decision],
    requirement: RequirementFactory,
    asked: tuple[str, str | None],
    stated: tuple[str, str | None],
    measure: str | None,
    route: str,
    refused: bool,
) -> None:
    # The user agreed to add aid to one side whose dataset nobody can name. A
    # sum whose unknown side is another unpinned requirement, one asking for
    # something else, or another pin whose dataset this turn cannot read, is a combination they were never
    # asked about, whether the pick is in this session or the sum is carried.
    ctx = _FakeToolContext()
    base = _sum_of_two_sources(
        ctx,
        datasets=(DAC2A, IDS),
        resolved_by=ResolvedBy.USER,
        decision=decision,
        pinning_decision=pinning_decision,
        requirement=requirement,
    )
    first = _with_debt_side(base, requirement_id=asked[0], spec_id=asked[1])
    plan = _with_debt_side(base, requirement_id=stated[0], spec_id=stated[1], measure=measure)
    previous = None
    if route == "pick":
        refusal = _combination_unasked(
            first, ToolContextSession(ctx), previous=None, sources=_shown_readings(ctx)
        )
        assert refusal is not None
        detail = refusal["detail"]
        ctx.session.events.extend(
            _combination_asked(
                {"spec_id": COMBINE_OPTION}, about=(detail["requirement_id"], detail["datasets"])
            )
        )
    else:
        previous = first

    result = _combination_unasked(
        plan, ToolContextSession(ctx), previous=previous, sources=_shown_readings(ctx)
    )

    assert (result is not None) is refused
    if result is not None:
        assert result["code"] == ProblemCode.COMBINATION_UNASKED.value


_SEPARATELY = "show them separately"


@pytest.mark.parametrize(
    ("previous_of", "asks", "chosen", "refused"),
    [
        (lambda plan: plan, [], COMBINE_OPTION, False),
        (
            lambda plan: plan.model_copy(
                update={
                    "evidence_requirements": tuple(
                        r for r in plan.evidence_requirements if r.id != "net_flows"
                    )
                }
            ),
            [],
            COMBINE_OPTION,
            True,
        ),
        (
            lambda plan: plan.model_copy(
                update={
                    "evidence_requirements": tuple(
                        r.model_copy(update={"source_spec_id": SPEC_GHANA_AID})
                        if r.id == "debt"
                        else r
                        for r in plan.evidence_requirements
                    )
                }
            ),
            [],
            COMBINE_OPTION,
            True,
        ),
        (lambda plan: plan, _combination_asked({"spec_id": COMBINE_OPTION}), COMBINE_OPTION, False),
        (lambda plan: plan, _combination_asked({"spec_id": _SEPARATELY}), COMBINE_OPTION, True),
        (lambda plan: plan, _combination_asked({"spec_id": _SEPARATELY}), _SEPARATELY, True),
        (lambda plan: plan, [], _SEPARATELY, True),
        (
            lambda plan: plan,
            _combination_asked({"spec_id": _SEPARATELY}, about=None),
            COMBINE_OPTION,
            True,
        ),
    ],
    ids=[
        "carried-unchanged",
        "added-by-this-call",
        "carried-over-other-datasets",
        "carried-with-its-pick-intact",
        "carried-after-the-user-picked-showing-them-separately",
        "patch-recording-showing-them-separately",
        "patch-changing-its-decision",
        "carried-after-an-unbound-ask-answered-separately",
    ],
)
def test_a_sum_carried_from_an_earlier_plan_keeps_its_sum_without_a_new_pick(
    decision: DecisionFactory,
    pinning_decision: Callable[..., Decision],
    requirement: RequirementFactory,
    previous_of: Callable[[Plan], Plan],
    asks: list[Any],
    chosen: str,
    refused: bool,
) -> None:
    # A later turn's session holds no pick. A sum the amended plan stated over
    # the same datasets, citing the same decision to add them, was checked when
    # it was recorded; one this call adds, one whose sides now read other
    # datasets, one whose decision changed, or one the user has since asked to
    # see separately needs removing or the pick again.
    ctx = _FakeToolContext()
    ctx.session.events.extend(asks)
    recorded = _sum_of_two_sources(
        ctx,
        datasets=(DAC2A, IDS),
        resolved_by=ResolvedBy.USER,
        decision=decision,
        pinning_decision=pinning_decision,
        requirement=requirement,
    )
    plan = recorded.model_copy(
        update={
            "decisions": tuple(
                d.model_copy(update={"chosen": chosen}) if d.slot == "combine_aid_debt" else d
                for d in recorded.decisions
            )
        }
    )

    result = _combination_unasked(
        plan, ToolContextSession(ctx), previous=previous_of(recorded), sources=_shown_readings(ctx)
    )

    assert (result is not None) is refused
    if result is not None:
        assert result["code"] == ProblemCode.COMBINATION_UNASKED.value


def test_propose_plan_refuses_a_patch_summing_two_datasets_on_a_users_source_pick(
    decision: DecisionFactory,
    pinning_decision: Callable[..., Decision],
    requirement: RequirementFactory,
) -> None:
    # The user elected the aid source; a patch that adds a sum citing that
    # election has no pick on adding aid and debt, so it is refused.
    tools = _tools_by_name()
    ctx = _FakeToolContext()
    summed = _sum_of_two_sources(
        ctx,
        datasets=(DAC2A, IDS),
        resolved_by=ResolvedBy.PLANNER_DEFAULT,
        decision=decision,
        pinning_decision=pinning_decision,
        requirement=requirement,
    )
    ctx.session.events = [
        _ask_event([{"spec_id": SPEC_GHANA_AID, "label": "aid"}]),
        _answer_event(SPEC_GHANA_AID),
    ]
    elected = pinning_decision(SPEC_GHANA_AID, slot="aid_source")
    debt_source = next(d for d in summed.decisions if d.slot == "debt_source")
    first = tools["propose_plan"].func(
        question=summed.question,
        answer_mode=AnswerMode.SERIES,
        decisions=drafts([elected, debt_source]),
        evidence_requirements=[r for r in summed.evidence_requirements if r.id != "net_flows"],
        tool_context=ctx,
    )
    assert first["outcome"] == Outcome.OK.value, first
    summing = summed.requirement("net_flows").model_copy(update={"decision_ids": ("aid_source",)})

    result = tools["propose_plan"].func(
        answer_mode=AnswerMode.COMPUTED,
        decisions=[],
        evidence_requirements=[summing],
        amends_version=1,
        tool_context=ctx,
    )

    assert result["code"] == ProblemCode.COMBINATION_UNASKED.value, result
    assert result["detail"] == {"requirement_id": "net_flows", "datasets": _AID_AND_DEBT}
    recorded = load_plan(ctx.state)
    assert recorded is not None
    assert recorded.version == 1


# --- question_fixed ------------------------------------------------------------------


def test_question_fixed_refuses_a_restart_and_records_nothing(
    decision: DecisionFactory, requirements: RequirementPair
) -> None:
    tools = _tools_by_name(question_fixed=True)
    ctx = _FakeToolContext()
    _propose(tools, ctx, decision(), requirements, question="Kenya's debt service since 1990")

    refused = _restart(
        tools, ctx, decision(), requirements, question="What was Kenya's ODA in 2023?"
    )

    assert refused["outcome"] == Outcome.REFUSED.value
    assert refused["code"] == ProblemCode.QUESTION_FIXED.value
    stored = load_plan(ctx.state)
    assert stored is not None
    assert stored.version == 1
    assert stored.question == "Kenya's debt service since 1990"
    assert load_plan_rounds(ctx.state) == 1


def test_question_fixed_restart_refusal_says_what_happened_and_needs_nothing_further(
    decision: DecisionFactory, requirements: RequirementPair
) -> None:
    tools = _tools_by_name(question_fixed=True)
    ctx = _FakeToolContext()
    _propose(tools, ctx, decision(), requirements)

    refused = _restart(tools, ctx, decision(), requirements, question="A different question")

    message = str(refused["message"])
    assert "fixed" in message
    assert "nothing is recorded" in message
    assert "nothing further is needed from you" in message


def test_question_fixed_leaves_starts_new_plan_alone_on_a_first_draft(
    decision: DecisionFactory, requirements: RequirementPair
) -> None:
    # Nothing recorded means there is no restart to refuse: the flag it
    # states is beside the point on a first draft, fixed mode or not.
    tools = _tools_by_name(question_fixed=True)
    ctx = _FakeToolContext()

    first = tools["propose_plan"].func(
        question="q",
        answer_mode=AnswerMode.COMPUTED,
        decisions=drafts([decision()]),
        evidence_requirements=list(requirements),
        starts_new_plan=True,
        tool_context=ctx,
    )

    assert first["outcome"] == Outcome.OK.value
    assert first["version"] == 1


def test_question_fixed_refuses_an_amendment_that_changes_the_question(
    decision: DecisionFactory, requirements: RequirementPair
) -> None:
    tools = _tools_by_name(question_fixed=True)
    ctx = _FakeToolContext()
    _propose(tools, ctx, decision(), requirements, question="Kenya's debt service since 1990")

    refused = _propose(
        tools, ctx, decision(), requirements, question="What was Kenya's ODA in 2023?"
    )

    assert refused["outcome"] == Outcome.INVALID_INPUT.value
    assert refused["code"] == ProblemCode.INVALID_PLAN.value
    assert "fixed for this turn" in str(refused["message"])
    assert "starts_new_plan=true" in str(refused["message"])
    stored = load_plan(ctx.state)
    assert stored is not None
    assert stored.version == 1
    assert stored.question == "Kenya's debt service since 1990"


def test_question_fixed_accepts_an_amendment_that_omits_the_question(
    decision: DecisionFactory, requirements: RequirementPair
) -> None:
    tools = _tools_by_name(question_fixed=True)
    ctx = _FakeToolContext()
    _propose(tools, ctx, decision(), requirements, question="Kenya's debt service since 1990")

    patched = tools["propose_plan"].func(
        decisions=drafts([decision(chosen="total debt stock", resolved_by=ResolvedBy.USER)]),
        evidence_requirements=[],
        amends_version=1,
        tool_context=ctx,
    )

    assert patched["outcome"] == Outcome.OK.value, patched
    assert patched["version"] == 2
    assert patched["question"] == "Kenya's debt service since 1990"


def test_question_fixed_accepts_an_amendment_restating_the_same_question(
    decision: DecisionFactory, requirements: RequirementPair
) -> None:
    tools = _tools_by_name(question_fixed=True)
    ctx = _FakeToolContext()
    _propose(tools, ctx, decision(), requirements, question="Kenya's debt service since 1990")

    patched = _propose(
        tools, ctx, decision(), requirements, question="Kenya's debt service since 1990"
    )

    assert patched["outcome"] == Outcome.OK.value, patched
    assert patched["version"] == 2


def test_question_fixed_restart_takes_precedence_over_budget_exhausted(
    decision: DecisionFactory, requirements: RequirementPair
) -> None:
    # Both refusals record nothing, so nothing is lost by naming the
    # structural reason first: a restart can never succeed under a fixed
    # question, however fresh the next budget is, so leading with a budget
    # message would misleadingly suggest otherwise.
    tools = _tools_by_name(question_fixed=True)
    ctx = _FakeToolContext()
    _propose_up_to_the_cap(tools, ctx, decision(), requirements)

    refused = _restart(tools, ctx, decision(), requirements, question="A different question")

    assert refused["code"] == ProblemCode.QUESTION_FIXED.value
    assert load_plan_rounds(ctx.state) == MAX_PLAN_ROUNDS


def test_question_fixed_amendment_question_check_still_yields_to_budget_exhausted(
    decision: DecisionFactory, requirements: RequirementPair
) -> None:
    # A correctly-shaped amendment (its own `amends_version` names the recorded
    # version) that only lost the race for the last round is exactly what the
    # round cap exists to stop, so budget still wins here, the opposite
    # ordering from the restart case.
    tools = _tools_by_name(question_fixed=True)
    ctx = _FakeToolContext()
    _propose_up_to_the_cap(tools, ctx, decision(), requirements)

    refused = _propose(tools, ctx, decision(), requirements, question="A different question")

    assert refused["code"] == ProblemCode.BUDGET_EXHAUSTED.value


def test_resolve_intent_question_fixed_restart_precedes_the_round_cap() -> None:
    plan = Plan(question="q", decisions=[], evidence_requirements=[])

    resolved = _resolve_intent(
        plan,
        amends_version=None,
        starts_new_plan=True,
        rounds_so_far=MAX_PLAN_ROUNDS,
        question="a different question",
        question_fixed=True,
    )

    assert isinstance(resolved, dict)
    assert resolved["code"] == ProblemCode.QUESTION_FIXED.value


def test_unfixed_mode_restart_is_unaffected_by_question_fixed_defaulting_false() -> None:
    plan = Plan(question="q", decisions=[], evidence_requirements=[])

    resolved = _resolve_intent(
        plan, amends_version=None, starts_new_plan=True, rounds_so_far=0, question="a new one"
    )

    assert resolved == (None, 1)


def test_the_instruction_varies_with_question_fixed() -> None:
    free = planner_instruction(has_translator=True, question_fixed=False)
    fixed = planner_instruction(has_translator=True, question_fixed=True)

    assert "different question that needs a plan of its own" in free
    assert "different question that needs a plan of its own" not in fixed
    assert "`starts_new_plan=true` and stop there" in fixed
    assert "`starts_new_plan=true` and stop there" not in free


def test_planner_instruction_defaults_question_fixed_to_false() -> None:
    assert planner_instruction(has_translator=True) == planner_instruction(
        has_translator=True, question_fixed=False
    )


def test_create_planner_agent_threads_question_fixed_into_the_instruction() -> None:
    fixed = create_planner_agent(model_name="m", question_fixed=True)
    free = create_planner_agent(model_name="m", question_fixed=False)

    assert "`starts_new_plan=true` and stop there" in str(fixed.instruction)
    assert "`starts_new_plan=true` and stop there" not in str(free.instruction)


def test_create_planner_agent_defaults_question_fixed_to_false() -> None:
    default = create_planner_agent(model_name="m")

    assert "`starts_new_plan=true` and stop there" not in str(default.instruction)


@pytest.mark.parametrize("cached", [False, True], ids=["uncached", "prefix-cached"])
async def test_the_planner_model_retries_a_503_before_the_turn_sees_it(
    cached: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The fault goes in under the SDK's retry loop, so the attempts counted
    # are the ones the SDK makes.
    attempts = 0

    async def unavailable(self: Any, http_request: Any, stream: bool = False) -> Any:
        nonlocal attempts
        attempts += 1
        raise errors.ServerError(503, {"error": {"code": 503, "status": "UNAVAILABLE"}})

    async def no_wait(seconds: float) -> None:
        return None

    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    monkeypatch.setattr(_api_client.BaseApiClient, "_async_request_once", unavailable)
    monkeypatch.setattr(asyncio, "sleep", no_wait)
    agent = create_planner_agent(
        model_name="gemini-3.8-flash", prefix_cache=PrefixCache() if cached else None
    )
    assert isinstance(agent.model, Gemini)

    with pytest.raises(errors.ServerError):
        await agent.model.api_client.aio.models.generate_content(
            model=agent.model.model, contents="question"
        )

    assert attempts == 3


def _arguments(tools: dict[str, FunctionTool], name: str = "propose_plan") -> dict[str, Any]:
    declaration = tools[name]._get_declaration()
    assert declaration is not None
    schema = declaration.parameters_json_schema
    assert schema is not None
    return schema["properties"]


def test_question_fixed_changes_propose_plans_declaration_for_question_and_starts_new_plan() -> (
    None
):
    free = _arguments(_tools_by_name())
    fixed = _arguments(_tools_by_name(question_fixed=True))

    changed = {name for name in free if free[name] != fixed[name]}

    assert changed == {"question", "starts_new_plan"}
    assert "has to match the one already recorded" in fixed["question"]["description"]
    assert "Refused, recording nothing" in fixed["starts_new_plan"]["description"]


def test_a_proposal_that_fails_validation_spends_no_round(
    decision: DecisionFactory, requirements: RequirementPair
) -> None:
    # Rounds count the proposals that were recorded. A proposal the plan's own
    # rules reject leaves the allowance where it was.
    tools = _tools_by_name()
    ctx = _FakeToolContext()
    for _ in range(MAX_PLAN_ROUNDS - 1):
        _propose(tools, ctx, decision(), requirements)

    invalid = tools["propose_plan"].func(
        question="q",
        decisions=drafts([decision()]),
        removed_evidence_requirements=[r.id for r in requirements],
        amends_version=load_plan(ctx.state).version,
        tool_context=ctx,
    )
    last = _propose(tools, ctx, decision(), requirements)

    assert invalid["code"] == ProblemCode.INVALID_PLAN.value
    assert load_plan_rounds(ctx.state) == MAX_PLAN_ROUNDS
    assert last["version"] == MAX_PLAN_ROUNDS


def _ratio(ratio_id: str, numerator: str, denominator: str) -> QuotientRequirement:
    """A quotient `ratio_id` of `numerator` over `denominator`."""
    return QuotientRequirement(
        id=ratio_id,
        decision_ids=["measure"],
        numerator=RequirementSide(input_id=numerator),
        denominator=RequirementSide(input_id=denominator),
    )


@pytest.mark.parametrize(
    "ratios",
    [
        pytest.param([_ratio("ratio", "ratio", "annual_debt_service")], id="reads-itself"),
        pytest.param(
            [
                _ratio("first", "second", "annual_debt_service"),
                _ratio("second", "first", "annual_debt_service"),
            ],
            id="two-step-cycle",
        ),
    ],
)
def test_a_proposal_whose_requirements_read_in_a_cycle_is_refused_and_records_nothing(
    decision: DecisionFactory,
    requirement: Callable[..., DatasetRequirement],
    ratios: list[QuotientRequirement],
) -> None:
    """A ratio reading itself, directly or through another ratio, is refused as a plan
    whose requirements do not form a DAG, before the chain walks what each side
    derives from."""
    tools = _tools_by_name()
    ctx = _FakeToolContext()

    result = _propose(tools, ctx, decision(), [requirement(), *ratios])

    assert result["code"] == ProblemCode.INVALID_PLAN.value, result
    assert "do not form a DAG" in str(result["message"])
    assert load_plan(ctx.state) is None


def test_propose_plan_after_rejection_revives_the_plan(
    decision: DecisionFactory, requirements: RequirementPair
) -> None:
    # Rejection blocks submission, not re-planning: a user who declines and
    # then asks for a change gets a new version of the same lineage, and
    # that version is declared ready once translated.
    tools = _tools_by_name()
    ctx = _FakeToolContext()
    _propose(tools, ctx, decision(), requirements)
    tools["reject_plan"].func(tool_context=ctx)

    revived = _propose(
        tools, ctx, decision(chosen="total debt stock", resolved_by=ResolvedBy.USER), requirements
    )
    _translate(ctx)

    assert revived["status"] == PlanStatus.DRAFT.value
    assert revived["version"] == 2
    assert cast("dict[str, Any]", _declare(ctx)["summary"])["version"] == 2


def test_propose_plan_clears_a_pending_referral(
    decision: DecisionFactory, requirements: RequirementPair
) -> None:
    tools = _tools_by_name()
    ctx = _FakeToolContext()
    _propose(tools, ctx, decision(), requirements)
    dataset, _derived = requirements
    save_referral(
        ctx.state,
        Referral(kind=ReferralKind.NO_DATA, reason="no data found", requirement_id=dataset.id),
    )
    assert load_referral(ctx.state) is not None

    _propose(
        tools, ctx, decision(chosen="total debt stock", resolved_by=ResolvedBy.USER), requirements
    )

    assert load_referral(ctx.state) is None


# The choices a `source_name_ambiguous` and an `entity_name_ambiguous` referral
# each have to carry.
_CHOICES_OF: dict[ReferralKind, dict[str, Any]] = {
    ReferralKind.SOURCE_NAME_AMBIGUOUS: {
        "ambiguous_sources": (
            AmbiguousSourceReading(
                text="shared",
                candidates=tuple(
                    SourceNameCandidate(ref=f"provenance/{name}", label=name, kind="provenance")
                    for name in ("a", "b")
                ),
            ),
        )
    },
    ReferralKind.ENTITY_NAME_AMBIGUOUS: {
        "ambiguous_entities": (
            AmbiguousEntityReading(
                text="Congo",
                candidates=tuple(
                    EntityCandidate(ref=f"country/{code}", label=code) for code in ("COD", "COG")
                ),
                entity=EntityKey(name="Congo", role="donor"),
            ),
        ),
        "repair": ENTITY_CHOICE_REPAIR,
    },
}


def _referral_of(kind: ReferralKind, requirement_id: str) -> Referral:
    """A valid `Referral` of `kind`, naming what that kind has to name."""
    if kind in _CHOICES_OF:
        return Referral(kind=kind, reason="r", requirement_id=requirement_id, **_CHOICES_OF[kind])
    if kind is ReferralKind.SOURCE_AMBIGUOUS:
        candidates = tuple(
            SelectedSource(
                spec_id="spec_" + letter * 32, sentence=f"source {letter}", has_data=True
            )
            for letter in "ab"
        )
        return Referral(kind=kind, reason="r", requirement_id=requirement_id, candidates=candidates)
    if kind is ReferralKind.REQUIREMENT_NEEDS_REPAIR:
        return Referral(
            kind=kind,
            reason="r",
            requirement_id=requirement_id,
            constraint=CompileIssueCode.ENTITY_REQUIRED,
            repair="name the entity",
        )
    if kind is ReferralKind.INTERNAL_FAILURE:
        return Referral(kind=kind, reason="r", reference="0123456789ab")
    if kind in (ReferralKind.DISCOVERY_BUDGET_EXHAUSTED, ReferralKind.SURFACE_UNAVAILABLE):
        return Referral(kind=kind, reason="r")
    return Referral(kind=kind, reason="r", requirement_id=requirement_id)


def _hand_back(ctx: _FakeToolContext, kind: ReferralKind, requirement_id: str) -> None:
    """Record a referral the way the translator does: in session state, and as the
    event the planner receives it on."""
    save_referral(ctx.state, _referral_of(kind, requirement_id))
    ctx.session.events.append(_referred_event(kind, requirement_id))


def _recorded_after_a_handed_back_plan(
    decision: DecisionFactory, requirements: RequirementPair, kind: ReferralKind
) -> tuple[dict[str, FunctionTool], _FakeToolContext, str]:
    """A recorded plan, and a referral of `kind` on its dataset requirement."""
    tools = _tools_by_name()
    ctx = _FakeToolContext()
    _propose(tools, ctx, decision(), requirements)
    dataset, _derived = requirements
    _hand_back(ctx, kind, dataset.id)
    return tools, ctx, dataset.id


@pytest.mark.parametrize("kind", HANDED_BACK_KINDS)
def test_propose_plan_refuses_an_amendment_while_the_user_has_not_answered_the_referral(
    decision: DecisionFactory, requirements: RequirementPair, kind: ReferralKind
) -> None:
    """The unmeetable referral goes to the user before anything is proposed.

    A by-key amendment replaces a requirement whole, so a plan proposed over the
    referral can drop the constraint the referral names and still validate. The
    refusal names the requirement and points at the tool that asks the user. It
    records nothing and spends no round. The plan, its round count and the
    pending referral all stay as they were.
    """
    tools, ctx, requirement_id = _recorded_after_a_handed_back_plan(decision, requirements, kind)
    rounds_before = load_plan_rounds(ctx.state)

    refused = _propose(tools, ctx, decision(rationale="Reconsidered."), ())

    assert refused["outcome"] == Outcome.REFUSED.value
    assert refused["code"] == ProblemCode.REFERRAL_PENDING.value
    assert refused["action"]["call"] == "adk_request_input"
    assert requirement_id in str(refused["message"])
    assert kind.value in str(refused["message"])
    stored = load_plan(ctx.state)
    assert stored is not None
    assert stored.version == 1
    assert load_plan_rounds(ctx.state) == rounds_before
    assert load_referral(ctx.state) is not None


@pytest.mark.parametrize("kind", HANDED_BACK_KINDS)
def test_propose_plan_is_allowed_once_the_user_has_answered(
    decision: DecisionFactory, requirements: RequirementPair, kind: ReferralKind
) -> None:
    tools, ctx, _requirement_id = _recorded_after_a_handed_back_plan(decision, requirements, kind)
    ctx.session.events.extend(_ask_answered_events())

    amended = _propose(tools, ctx, decision(rationale="Reconsidered."), ())

    assert amended["outcome"] == Outcome.OK.value, amended
    assert amended["version"] == 2
    assert load_referral(ctx.state) is None


def test_propose_plan_refuses_a_restart_over_an_unanswered_referral(
    decision: DecisionFactory, requirements: RequirementPair
) -> None:
    tools, ctx, _requirement_id = _recorded_after_a_handed_back_plan(
        decision, requirements, ReferralKind.REQUIREMENT_UNMEETABLE
    )

    refused = _restart(tools, ctx, decision(), requirements, question="Another question")

    assert refused["code"] == ProblemCode.REFERRAL_PENDING.value
    stored = load_plan(ctx.state)
    assert stored is not None
    assert (stored.question, stored.version) == ("q", 1)
    assert load_plan_rounds(ctx.state) == 1


def test_a_pending_referral_on_a_copy_names_its_origin_to_the_model(
    decision: DecisionFactory,
) -> None:
    """A referral on `debt_history`, the split's copy of `debt`, reaches the model as `debt`.

    Both the refusal of a proposal over it and the declaration gate's
    `detail` name the requirement the model wrote, while the referral in
    session state keeps the copy's id.
    """
    tools = _tools_by_name()
    ctx = _FakeToolContext()
    save_plan(ctx.state, split_draft(decision))
    _hand_back(ctx, ReferralKind.REQUIREMENT_UNMEETABLE, "debt_history")

    refused = _propose(tools, ctx, decision(rationale="Reconsidered."), ())
    gated = cast("dict[str, Any]", _declare(ctx)["submission"])

    assert refused["code"] == ProblemCode.REFERRAL_PENDING.value
    assert "`debt`" in str(refused["message"])
    assert gated["code"] == ProblemCode.REFERRAL_PENDING.value
    assert gated["detail"]["referral"]["requirement_id"] == "debt"
    assert "debt_history" not in json.dumps([refused, gated])
    stored = load_referral(ctx.state)
    assert stored is not None
    assert stored.requirement_id == "debt_history"


def test_a_first_draft_is_not_held_by_a_referral(
    decision: DecisionFactory, requirements: RequirementPair
) -> None:
    tools = _tools_by_name()
    ctx = _FakeToolContext()
    dataset, _derived = requirements
    _hand_back(ctx, ReferralKind.REQUIREMENT_UNMEETABLE, dataset.id)

    first = _propose(tools, ctx, decision(), requirements)

    assert first["outcome"] == Outcome.OK.value, first
    assert first["version"] == 1


@pytest.mark.parametrize(
    "kind",
    [kind for kind in ReferralKind if kind not in HANDED_BACK_KINDS],
)
def test_every_other_referral_kind_still_lets_the_planner_propose_unasked(
    decision: DecisionFactory, requirements: RequirementPair, kind: ReferralKind
) -> None:
    tools, ctx, _requirement_id = _recorded_after_a_handed_back_plan(decision, requirements, kind)

    amended = _propose(tools, ctx, decision(rationale="Reconsidered."), ())

    assert amended["outcome"] == Outcome.OK.value, amended


@pytest.mark.parametrize(
    ("kind", "pin", "refused"),
    [
        (ReferralKind.SOURCE_AMBIGUOUS, SPEC, False),
        (ReferralKind.SOURCE_AMBIGUOUS, POPULATION_SPEC, True),
        (ReferralKind.SOURCE_AMBIGUOUS, None, False),
        (ReferralKind.PINNED_SOURCE_NOT_FOUND, POPULATION_SPEC, True),
        (ReferralKind.NO_DATA, POPULATION_SPEC, False),
    ],
    ids=["offered", "not_offered", "unpinned", "none_offered", "not_a_choice"],
)
def test_a_pending_source_choice_admits_only_a_pin_it_offered(
    decision: DecisionFactory,
    requirements: RequirementPair,
    kind: ReferralKind,
    pin: str | None,
    refused: bool,
) -> None:
    """A `choice_required` referral's candidates are the only sources its requirement may pin.

    `_referral_of` offers `SPEC` and another candidate for `source_ambiguous`,
    and nothing for `pinned_source_not_found`. A plan pinning anything else
    there is refused and records nothing; a referral that is no choice holds
    no pin.
    """
    tools, ctx, _requirement_id = _recorded_after_a_handed_back_plan(decision, requirements, kind)
    dataset, derived = requirements
    choice = decision(fixes=DecisionFixes(source_spec_id=pin)) if pin else decision()

    result = _propose(
        tools, ctx, choice, [dataset.model_copy(update={"source_spec_id": pin}), derived]
    )

    if refused:
        assert (result["outcome"], result["code"]) == (
            Outcome.REFUSED.value,
            ProblemCode.SOURCE_NOT_OFFERED.value,
        )
        stored = load_plan(ctx.state)
        assert stored is not None
        assert stored.version == 1
        assert load_referral(ctx.state) is not None
    else:
        assert result["outcome"] == Outcome.OK.value, result


def test_an_answer_given_before_the_referral_does_not_settle_it(
    decision: DecisionFactory, requirements: RequirementPair
) -> None:
    tools = _tools_by_name()
    ctx = _FakeToolContext()
    _propose(tools, ctx, decision(), requirements)
    ctx.session.events.extend(_ask_answered_events())
    dataset, _derived = requirements
    _hand_back(ctx, ReferralKind.REQUIREMENT_UNMEETABLE, dataset.id)

    refused = _propose(tools, ctx, decision(rationale="Reconsidered."), ())

    assert refused["code"] == ProblemCode.REFERRAL_PENDING.value


def _unsupported_ask_handed_back(
    decision: DecisionFactory, requirements: RequirementPair
) -> tuple[dict[str, FunctionTool], _FakeToolContext]:
    tools, ctx, _requirement_id = _recorded_after_a_handed_back_plan(
        decision, requirements, ReferralKind.UNSUPPORTED_ASK
    )
    return tools, ctx


@pytest.mark.parametrize(
    "body",
    [
        {
            "error": "Invoking `adk_request_input()` failed as the following mandatory "
            "input parameters are not present:\ncandidates"
        },
        {
            "outcome": Outcome.INVALID_INPUT.value,
            "code": ProblemCode.INVALID_ARGUMENTS.value,
            "message": "recommended_spec_id 'spec_x' names no candidate in []",
        },
    ],
    ids=["adk_missing_argument", "ask_user_invalid_arguments"],
)
def test_an_ask_the_tool_rejected_does_not_settle_the_referral(
    decision: DecisionFactory, requirements: RequirementPair, body: dict[str, Any]
) -> None:
    """A rejected `adk_request_input` call comes back as a `FunctionResponse` at once, with
    no user involved and no `result` in its body."""
    tools, ctx = _unsupported_ask_handed_back(decision, requirements)
    ctx.session.events.extend([_ask_event([], call_id="ask-1"), _ask_response("ask-1", body)])

    refused = _propose(tools, ctx, decision(rationale="Reconsidered."), ())

    assert refused["code"] == ProblemCode.REFERRAL_PENDING.value


def test_an_answer_to_an_ask_raised_before_the_referral_does_not_settle_it(
    decision: DecisionFactory, requirements: RequirementPair
) -> None:
    """The model asks something and calls the translator in one turn. The answer belongs to
    an ask issued before the model saw the referral, so the user has not seen the referral."""
    tools = _tools_by_name()
    ctx = _FakeToolContext()
    _propose(tools, ctx, decision(), requirements)
    dataset, _derived = requirements
    ctx.session.events.append(_ask_event([], call_id="ask-1"))
    _hand_back(ctx, ReferralKind.UNSUPPORTED_ASK, dataset.id)
    ctx.session.events.append(_ask_response("ask-1", {"result": "the second measure"}))

    refused = _propose(tools, ctx, decision(rationale="Reconsidered."), ())

    assert refused["code"] == ProblemCode.REFERRAL_PENDING.value


def test_an_answer_with_no_call_on_record_does_not_settle_the_referral(
    decision: DecisionFactory, requirements: RequirementPair
) -> None:
    tools, ctx = _unsupported_ask_handed_back(decision, requirements)
    ctx.session.events.append(_ask_response("ask-1", {"result": "use current prices"}))

    refused = _propose(tools, ctx, decision(rationale="Reconsidered."), ())

    assert refused["code"] == ProblemCode.REFERRAL_PENDING.value


@pytest.mark.parametrize(
    "result",
    [{"skip": True}, "", "   ", {"text": ""}, {"text": "  "}, {}, None, {"approved": None}],
    ids=[
        "skip",
        "empty",
        "blank",
        "empty_text",
        "blank_text",
        "empty_dict",
        "null",
        "no_answer_key",
    ],
)
def test_a_result_that_carries_no_answer_does_not_settle_the_referral(
    decision: DecisionFactory, requirements: RequirementPair, result: Any
) -> None:
    """A skip is the user declining to choose. The planner may not choose in their place, so
    it ends the turn by telling them what cannot be done."""
    tools, ctx = _unsupported_ask_handed_back(decision, requirements)
    ctx.session.events.extend(_ask_answered_events(result))

    refused = _propose(tools, ctx, decision(rationale="Reconsidered."), ())

    assert refused["code"] == ProblemCode.REFERRAL_PENDING.value


@pytest.mark.parametrize(
    "result",
    ["use current prices", {"text": "use current prices"}, {"spec_id": SPEC}],
    ids=["text_string", "text", "spec_id"],
)
def test_a_real_answer_to_an_ask_raised_after_the_referral_settles_it(
    decision: DecisionFactory, requirements: RequirementPair, result: Any
) -> None:
    tools, ctx = _unsupported_ask_handed_back(decision, requirements)
    ctx.session.events.extend(_ask_answered_events(result))

    amended = _propose(tools, ctx, decision(rationale="Reconsidered."), ())

    assert amended["outcome"] == Outcome.OK.value, amended


def test_an_answer_after_a_referral_counts() -> None:
    """The referral is read off the recording answer's `translation`."""
    ctx = _FakeToolContext()
    ctx.session.events.append(_referred_event(ReferralKind.UNSUPPORTED_ASK, "gdp"))
    ctx.session.events.extend(_ask_answered_events())

    assert ToolContextSession(ctx).answers_since_referral().answered


def test_a_refused_proposal_after_the_answer_does_not_take_the_answer_back(
    decision: DecisionFactory, requirements: RequirementPair
) -> None:
    """A proposal the plan rules refused leaves the user's answer standing, so the
    retry is not sent back to ask the same question twice."""
    tools, ctx, _requirement_id = _recorded_after_a_handed_back_plan(
        decision, requirements, ReferralKind.REQUIREMENT_UNMEETABLE
    )
    ctx.session.events.extend(_ask_answered_events())
    ctx.session.events.append(
        response_event("propose_plan", {"outcome": "invalid_input", "code": "invalid_plan"})
    )

    amended = _propose(tools, ctx, decision(rationale="Reconsidered."), ())

    assert amended["outcome"] == Outcome.OK.value, amended


def test_a_second_referral_after_a_recorded_proposal_needs_its_own_answer(
    decision: DecisionFactory, requirements: RequirementPair
) -> None:
    tools, ctx, requirement_id = _recorded_after_a_handed_back_plan(
        decision, requirements, ReferralKind.REQUIREMENT_UNMEETABLE
    )
    ctx.session.events.extend(_ask_answered_events())
    settled = _propose(tools, ctx, decision(rationale="Reconsidered."), ())
    ctx.session.events.append(response_event("propose_plan", settled))
    _hand_back(ctx, ReferralKind.REQUIREMENT_UNMEETABLE, requirement_id)

    refused = _propose(tools, ctx, decision(rationale="Reconsidered again."), ())

    assert settled["outcome"] == Outcome.OK.value, settled
    assert refused["code"] == ProblemCode.REFERRAL_PENDING.value


def test_a_referral_does_not_reach_the_next_turn(
    decision: DecisionFactory, requirements: RequirementPair
) -> None:
    """`_persist_durable` mirrors the plan, the round count and the briefed flag, so the
    session a new turn opens carries no referral for the guard to hold."""
    tools, ctx, _requirement_id = _recorded_after_a_handed_back_plan(
        decision, requirements, ReferralKind.REQUIREMENT_UNMEETABLE
    )
    next_turn = SimpleNamespace(session=SimpleNamespace(state=ctx.state), state={})

    PlannerSeam._persist_durable(next_turn)  # type: ignore[arg-type]

    fresh = _FakeToolContext()
    fresh.state.update(next_turn.state)
    assert load_referral(fresh.state) is None
    assert load_plan(fresh.state) is not None
    amended = _propose(tools, fresh, decision(rationale="Reconsidered."), ())
    assert amended["outcome"] == Outcome.OK.value, amended


def test_propose_plan_rejects_a_decision_whose_chosen_is_not_in_options(
    requirement: RequirementFactory,
) -> None:
    tools = _tools_by_name()
    ctx = _FakeToolContext()

    # A plain dict, not a `Decision`: this is what ADK's own arg
    # preprocessing hands `propose_plan` when a single item in the list
    # fails validation -- it silently falls back to raw dicts for the whole
    # list rather than raising. `propose_plan` has to catch this itself.
    # Fails validating `decisions`, so `evidence_requirements` is never
    # reached -- a single dataset requirement, which would otherwise be
    # invalid on its own, is fine here.
    bad_decision = {
        "slot": "measure",
        "description": "What to measure Kenya's dependence by.",
        "options": ["annual debt service", "total debt stock"],
        "chosen": "total debt service",  # typo: not one of the two options
        "resolved_by": ResolvedBy.PLANNER_DEFAULT,
        "rationale": "Standard measure for near-term dependence.",
    }

    result = tools["propose_plan"].func(
        question="q",
        decisions=drafts([bad_decision]),
        evidence_requirements=[requirement()],
        tool_context=ctx,
    )

    assert result["code"] == ProblemCode.INVALID_PLAN.value
    assert result["outcome"] == Outcome.INVALID_INPUT.value
    assert "not one of the stated options" in result["message"]
    assert "errors" in result["detail"]
    assert PLAN_STATE_KEY not in ctx.state


def test_propose_plan_rejects_a_malformed_requirement(decision: DecisionFactory) -> None:
    tools = _tools_by_name()
    ctx = _FakeToolContext()

    # Missing the required `period` -- same ADK raw-dict-fallback scenario as
    # the Decision case above, for a requirement.
    bad_requirement = {
        "kind": "dataset",
        "id": "annual_debt_service",
        "decision_ids": ["measure"],
        "measure": "external debt service paid",
        "entities": [{"name": "Kenya"}],
    }

    result = tools["propose_plan"].func(
        question="q",
        decisions=drafts([decision()]),
        evidence_requirements=[bad_requirement],
        tool_context=ctx,
    )

    assert result["code"] == ProblemCode.INVALID_PLAN.value
    assert "errors" in result["detail"]
    assert PLAN_STATE_KEY not in ctx.state


def test_propose_plan_accepts_a_mixed_list_of_dataset_and_derived_dicts(
    decision: DecisionFactory,
) -> None:
    # The shape ADK actually hands a union-typed list parameter: plain dicts,
    # one per kind, never pre-built model instances. `propose_plan` has to
    # rebuild the models itself through `_REQUIREMENTS_ADAPTER`.
    tools = _tools_by_name()
    ctx = _FakeToolContext()
    dataset_dict = {
        "kind": "dataset",
        "id": "annual_debt_service",
        "decision_ids": ["measure"],
        "measure": "external debt service paid",
        "entities": [{"name": "Kenya", "role": "debtor"}],
        "period": {"start": 2015, "end": 2024},
        "unit": {"currency": "USD", "prices": "current"},
    }
    derived_dict = {
        "kind": "aggregate",
        "id": "annual_totals",
        "decision_ids": ["measure"],
        "input_id": "annual_debt_service",
        "reduction": "sum",
        "group_by": ["period"],
    }

    result = tools["propose_plan"].func(
        question="q",
        answer_mode=AnswerMode.COMPUTED,
        decisions=drafts([decision()]),
        evidence_requirements=[dataset_dict, derived_dict],
        tool_context=ctx,
    )

    assert result["outcome"] == Outcome.OK.value
    assert result["status"] == PlanStatus.DRAFT.value
    stored = load_plan(ctx.state)
    assert stored is not None
    assert [r.kind for r in stored.evidence_requirements] == ["dataset", "aggregate"]


def test_propose_plan_rejects_a_requirement_citing_no_decision_the_plan_makes(
    decision: DecisionFactory, requirement: RequirementFactory
) -> None:
    # The plan-level rule, reached through the tool: a citation of a decision
    # the plan does not make is a problem the agent can act on, not an
    # exception that ends the turn. Fails before the citation check's own
    # sibling rules would.
    tools = _tools_by_name()
    ctx = _FakeToolContext()

    result = _propose(tools, ctx, decision(), (requirement(decision_ids=["time_grain"]),))

    assert result["code"] == ProblemCode.INVALID_PLAN.value
    assert "does not make" in result["message"]
    assert PLAN_STATE_KEY not in ctx.state


def test_propose_plan_rejects_an_uncited_decision(
    decision: DecisionFactory, requirement: RequirementFactory
) -> None:
    tools = _tools_by_name()
    ctx = _FakeToolContext()
    uncited = decision(
        slot="time_grain", options=["annual totals", "quarterly totals"], chosen="annual totals"
    )

    result = tools["propose_plan"].func(
        question="q",
        answer_mode=AnswerMode.COMPUTED,
        decisions=drafts([decision(), uncited]),
        evidence_requirements=[requirement()],
        tool_context=ctx,
    )

    assert result["code"] == ProblemCode.INVALID_PLAN.value
    assert "no evidence requirement cites" in result["message"]
    assert PLAN_STATE_KEY not in ctx.state


def test_propose_plan_accepts_a_dataset_requirement_that_is_the_answer(
    decision: DecisionFactory, requirement: RequirementFactory
) -> None:
    # A lone dataset requirement is the answer when the question asks to see a
    # series. The retrieval's frame carries one series per entity, which a
    # chart draws with no calculation over it.
    tools = _tools_by_name()
    ctx = _FakeToolContext()

    result = _propose(tools, ctx, decision(), (requirement(),), answer_mode=AnswerMode.SERIES)

    assert "code" not in result, result
    assert PLAN_STATE_KEY in ctx.state


def test_propose_plan_accepts_a_set_larger_than_the_entity_limit_named_once(
    decision: DecisionFactory, requirement: RequirementFactory
) -> None:
    # Africa holds more members than a requirement may list, so a question about
    # African countries names the set once and asks for its members.
    fixture = Path(__file__).parent.parent / "fixtures" / "search_spec_region_members.json"
    africa = json.loads(fixture.read_text())["spec"]["resolved"]["entities"]
    assert len(africa) > MAX_ENTITIES
    tools = _tools_by_name()
    ctx = _FakeToolContext()
    countries = requirement(
        measure="government health expenditure",
        entities=[Entity(name="Africa", members=True)],
        unit=None,
    )

    result = _propose(tools, ctx, decision(), (countries,), answer_mode=AnswerMode.SERIES)

    assert "code" not in result, result
    plan = load_plan(ctx.state)
    assert plan is not None
    assert plan.evidence_requirements[0].entities == (Entity(name="Africa", members=True),)  # type: ignore[union-attr]


def test_propose_plan_needs_a_derived_requirement_for_a_growth_question(
    decision: DecisionFactory,
    requirement: RequirementFactory,
    derived_requirement: DerivedRequirementFactory,
) -> None:
    # A question about how a value changed asks for a figure the frame does
    # not carry, so the dataset requirement still feeds a calculation.
    tools = _tools_by_name()
    ctx = _FakeToolContext()

    result = _propose(
        tools,
        ctx,
        decision(),
        (requirement(), derived_requirement()),
        question="How much did Kenya's debt service grow between 2016 and 2020?",
    )

    assert "code" not in result, result
    assert PLAN_STATE_KEY in ctx.state


def test_propose_plan_needs_at_least_one_requirement(decision: DecisionFactory) -> None:
    tools = _tools_by_name()
    ctx = _FakeToolContext()

    result = tools["propose_plan"].func(
        question="q",
        answer_mode=AnswerMode.COMPUTED,
        decisions=drafts([decision()]),
        evidence_requirements=[],
        tool_context=ctx,
    )

    assert result["code"] == ProblemCode.INVALID_PLAN.value
    assert "at least one evidence requirement" in result["message"]
    assert PLAN_STATE_KEY not in ctx.state


# --- Stating which a call is ---------------------------------------------------------


def test_a_call_with_a_plan_recorded_must_state_which_it_is(
    decision: DecisionFactory, requirements: RequirementPair
) -> None:
    """A recorded plan and no stated intent is refused. The reading is never guessed from wording.

    A question reworded while the model meant to amend would otherwise read
    as a fresh draft and drop everything the call did not restate. The call
    states which it is, and the refusal names the version to pass.
    """
    tools = _tools_by_name()
    ctx = _FakeToolContext()
    _propose(tools, ctx, decision(), requirements)

    result = tools["propose_plan"].func(
        question="Kenya's dependence on lending from China",
        decisions=drafts([decision(chosen="total debt stock")]),
        evidence_requirements=list(requirements),
        tool_context=ctx,
    )

    assert result["code"] == ProblemCode.INVALID_PLAN.value
    assert "A plan is already recorded (version 1)" in result["message"]
    assert "amends_version=1" in result["message"]
    assert "starts_new_plan=true" in result["message"]
    stored = load_plan(ctx.state)
    assert stored is not None
    assert stored.version == 1
    assert stored.decisions[0].chosen == "annual debt service"


def test_an_amendment_naming_the_wrong_version_is_refused(
    decision: DecisionFactory, requirements: RequirementPair
) -> None:
    tools = _tools_by_name()
    ctx = _FakeToolContext()
    _propose(tools, ctx, decision(), requirements)
    _propose(tools, ctx, decision(chosen="total debt stock"), requirements)

    result = tools["propose_plan"].func(
        question="q",
        decisions=drafts([decision(rationale="Reconsidered.")]),
        amends_version=1,
        tool_context=ctx,
    )

    assert result["code"] == ProblemCode.INVALID_PLAN.value
    assert "amends version 1, but version 2 is the one recorded" in result["message"]
    assert "amends_version=2" in result["message"]
    stored = load_plan(ctx.state)
    assert stored is not None
    assert stored.version == 2


def test_a_call_with_no_plan_recorded_needs_no_statement_of_intent(
    decision: DecisionFactory, requirements: RequirementPair
) -> None:
    # Nothing recorded means a first draft. There is no second reading to
    # tell apart.
    tools = _tools_by_name()
    ctx = _FakeToolContext()

    result = tools["propose_plan"].func(
        question="q",
        answer_mode=AnswerMode.COMPUTED,
        decisions=drafts([decision()]),
        evidence_requirements=list(requirements),
        tool_context=ctx,
    )

    assert result["outcome"] == Outcome.OK.value, result
    assert result["version"] == 1
    assert result["previous_version"] is None


def test_starts_new_plan_begins_a_plan_of_its_own(
    decision: DecisionFactory, requirements: RequirementPair
) -> None:
    # A different question under a spent plan gets version 1 of a new lineage,
    # and the recorded plan's items do not ride along.
    tools = _tools_by_name()
    ctx = _FakeToolContext()
    _propose(tools, ctx, decision(), requirements)

    result = tools["propose_plan"].func(
        question="What was Kenya's total ODA in 2023?",
        answer_mode=AnswerMode.COMPUTED,
        decisions=drafts(
            [decision(slot="oda", options=["total ODA", "net ODA"], chosen="total ODA")]
        ),
        evidence_requirements=[
            DatasetRequirement(
                id="oda",
                decision_ids=["oda"],
                measure="total ODA received",
                entities=[Entity(name="Kenya")],
                period=Period(start=2023, end=2023),
            ),
            AggregateRequirement(
                id="oda_totals",
                decision_ids=["oda"],
                input_id="oda",
                reduction=Reduction.SUM,
                group_by=(GroupBy.PERIOD,),
            ),
        ],
        starts_new_plan=True,
        tool_context=ctx,
    )

    assert result["outcome"] == Outcome.OK.value, result
    assert result["version"] == 1
    assert result["previous_version"] is None
    stored = load_plan(ctx.state)
    assert stored is not None
    assert [d.slot for d in stored.decisions] == ["oda"]


def test_an_amendment_stating_both_reads_as_the_amendment(
    decision: DecisionFactory, requirements: RequirementPair
) -> None:
    # Both stated is an amendment. Patching the plan the version names is the
    # narrower reading, so it wins.
    tools = _tools_by_name()
    ctx = _FakeToolContext()
    _propose(tools, ctx, decision(), requirements)

    result = tools["propose_plan"].func(
        question="q",
        decisions=drafts([decision(rationale="Reconsidered.")]),
        amends_version=1,
        starts_new_plan=True,
        tool_context=ctx,
    )

    assert result["outcome"] == Outcome.OK.value, result
    assert result["version"] == 2


# --- Patching an existing plan -----------------------------------------------------


def test_propose_plan_patched_by_slot_carries_the_rest_forward(
    decision: DecisionFactory, requirements: RequirementPair
) -> None:
    """A second call for the same question patches the plan.

    The decision is matched by `slot` and replaced in place, and the
    requirements the patch does not mention are carried forward rather than
    deleted. This is the difference a model that retypes only what changed
    relies on. Its unmentioned plan survives.
    """
    tools = _tools_by_name()
    ctx = _FakeToolContext()
    _propose(tools, ctx, decision(), requirements)

    patched = tools["propose_plan"].func(
        question="q",
        decisions=drafts([decision(chosen="total debt stock", resolved_by=ResolvedBy.USER)]),
        amends_version=1,
        tool_context=ctx,
    )

    assert patched["version"] == 2
    stored = load_plan(ctx.state)
    assert stored is not None
    assert [d.slot for d in stored.decisions] == ["measure"]
    assert stored.decisions[0].chosen == "total debt stock"
    # Nothing was restated, and nothing was lost.
    assert [r.id for r in stored.evidence_requirements] == [r.id for r in requirements]


def test_propose_plan_patched_adds_a_decision_the_plan_does_not_state(
    decision: DecisionFactory,
    requirement: RequirementFactory,
    derived_requirement: DerivedRequirementFactory,
) -> None:
    tools = _tools_by_name()
    ctx = _FakeToolContext()
    _propose(tools, ctx, decision(), (requirement(), derived_requirement()))
    added = decision(
        slot="time_grain",
        options=["annual totals", "quarterly totals"],
        chosen="annual totals",
    )

    patched = tools["propose_plan"].func(
        question="q",
        decisions=drafts([added]),
        evidence_requirements=[
            requirement(decision_ids=["measure", "time_grain"]),
            derived_requirement(decision_ids=["time_grain"]),
        ],
        amends_version=1,
        tool_context=ctx,
    )

    assert patched["outcome"] == Outcome.OK.value
    stored = load_plan(ctx.state)
    assert stored is not None
    assert [d.slot for d in stored.decisions] == ["measure", "time_grain"]


def test_propose_plan_patched_removes_only_what_it_names(
    decision: DecisionFactory,
    requirement: RequirementFactory,
    derived_requirement: DerivedRequirementFactory,
) -> None:
    """Absence from a patch means unchanged. Deletion has its own channel.

    Two independent chains in the plan, and a patch that restates one of them
    and names the other's two requirement ids for removal. The requirement the
    patch does not mention is carried forward, and the ones it names are gone,
    so the merged plan validates.
    """
    tools = _tools_by_name()
    ctx = _FakeToolContext()
    second = decision(
        slot="aid_measure",
        options=["gross ODA", "net ODA"],
        chosen="gross ODA",
    )
    _propose(
        tools,
        ctx,
        decision(),
        (requirement(), derived_requirement()),
    )
    tools["propose_plan"].func(  # version 1 -> 2: adds the second chain
        question="q",
        decisions=drafts([second]),
        evidence_requirements=[
            requirement(id="aid", decision_ids=["aid_measure"], measure="aid disbursements"),
            derived_requirement(id="aid_total", decision_ids=["aid_measure"], input_id="aid"),
        ],
        amends_version=1,
        tool_context=ctx,
    )

    patched = tools["propose_plan"].func(
        question="q",
        removed_evidence_requirements=["aid", "aid_total"],
        removed_decisions=["aid_measure"],
        amends_version=2,
        tool_context=ctx,
    )

    assert patched["outcome"] == Outcome.OK.value, patched
    stored = load_plan(ctx.state)
    assert stored is not None
    assert [r.id for r in stored.evidence_requirements] == ["annual_debt_service", "annual_totals"]
    assert [d.slot for d in stored.decisions] == ["measure"]


def test_propose_plan_refuses_a_removal_naming_no_requirement(
    decision: DecisionFactory, requirements: RequirementPair
) -> None:
    tools = _tools_by_name()
    ctx = _FakeToolContext()
    _propose(tools, ctx, decision(), requirements)

    result = tools["propose_plan"].func(
        question="q",
        removed_evidence_requirements=["no_such_requirement"],
        amends_version=1,
        tool_context=ctx,
    )

    assert result["code"] == ProblemCode.INVALID_PLAN.value
    assert "every removed id must name one the plan states" in result["message"]
    stored = load_plan(ctx.state)
    assert stored is not None
    assert stored.version == 1


def test_a_patch_that_leaves_a_decision_uncited_is_an_invalid_plan(
    decision: DecisionFactory, requirements: RequirementPair
) -> None:
    # The merged plan is validated in full. Replacing the requirements with
    # ones citing a decision the plan does not make is caught the same way a
    # first draft's would be, and nothing is recorded.
    tools = _tools_by_name()
    ctx = _FakeToolContext()
    _propose(tools, ctx, decision(), requirements)

    result = tools["propose_plan"].func(
        question="q",
        decisions=drafts([decision(slot="window")]),
        amends_version=1,
        tool_context=ctx,
    )

    assert result["code"] == ProblemCode.INVALID_PLAN.value
    assert "no evidence requirement cites" in result["message"]
    stored = load_plan(ctx.state)
    assert stored is not None
    assert stored.version == 1


def test_a_patch_removing_a_requirement_another_still_reads_is_an_invalid_plan(
    decision: DecisionFactory, requirements: RequirementPair
) -> None:
    """The merged plan is validated in full, so a removal can fail the rule the
    plan states about its own graph (the derived requirement still reads the
    dataset one the patch removed)."""
    tools = _tools_by_name()
    ctx = _FakeToolContext()
    _propose(tools, ctx, decision(), requirements)
    dataset, _derived = requirements

    result = tools["propose_plan"].func(
        question="q",
        removed_evidence_requirements=[dataset.id],
        amends_version=1,
        tool_context=ctx,
    )

    assert result["code"] == ProblemCode.INVALID_PLAN.value
    assert "read requirements this plan does not state" in result["message"]
    stored = load_plan(ctx.state)
    assert stored is not None
    assert stored.version == 1


# --- The cardinality cap ------------------------------------------------------------


def _plan_at_the_cap(decisions: int) -> tuple[list[Decision], list[EvidenceRequirement]]:
    """A plan stating `decisions` decisions and as many requirements as they need.

    Half as many dataset requirements as decisions, each cited by one dataset
    requirement and the aggregate that reads it, so every decision is cited
    and every dataset requirement feeds a calculation. Requirements come to
    the same count as decisions, which is what the requirement cap allows.
    """
    made = [
        Decision(
            slot=f"slot_{index}",
            description=f"Choice {index}.",
            options=["a", "b"],
            chosen="a",
            resolved_by=ResolvedBy.PLANNER_DEFAULT,
            rationale=f"Choice {index} is the standard reading.",
        )
        for index in range(decisions)
    ]
    half = decisions // 2
    requirements: list[EvidenceRequirement] = [
        DatasetRequirement(
            id=f"req_{index}",
            decision_ids=[f"slot_{index}", f"slot_{half + index}"],
            measure=f"measure {index}",
            entities=[Entity(name="Kenya")],
            period=Period(start=2015, end=2024),
        )
        for index in range(half)
    ]
    requirements += [
        AggregateRequirement(
            id=f"agg_{index}",
            decision_ids=[f"slot_{index}", f"slot_{half + index}"],
            input_id=f"req_{index}",
            reduction=Reduction.SUM,
            group_by=(GroupBy.PERIOD,),
        )
        for index in range(half)
    ]
    return made, requirements


def test_a_plan_at_the_decision_cap_records() -> None:
    """The cap is a bound a real plan reaches without being refused."""
    tools = _tools_by_name()
    ctx = _FakeToolContext()
    decisions, requirements = _plan_at_the_cap(MAX_PLAN_DECISIONS)

    result = tools["propose_plan"].func(
        question="q",
        answer_mode=AnswerMode.COMPUTED,
        decisions=drafts(decisions),
        evidence_requirements=requirements,
        tool_context=ctx,
    )

    assert result["outcome"] == Outcome.OK.value, result
    stored = load_plan(ctx.state)
    assert stored is not None
    assert len(stored.decisions) == MAX_PLAN_DECISIONS
    assert len(stored.evidence_requirements) == MAX_PLAN_DECISIONS


def test_a_plan_one_requirement_past_the_cap_records_nothing() -> None:
    """One requirement over the cap fails, so the cap is exactly at the number
    the passing case above measured."""
    tools = _tools_by_name()
    ctx = _FakeToolContext()
    decisions, requirements = _plan_at_the_cap(MAX_PLAN_DECISIONS)
    over = [*requirements, requirements[-1].model_copy(update={"id": "one_too_many"})]

    result = tools["propose_plan"].func(
        question="q",
        decisions=drafts(decisions),
        evidence_requirements=over,
        tool_context=ctx,
    )

    assert result["code"] == ProblemCode.INVALID_PLAN.value
    assert "evidence_requirements" in result["message"]
    assert PLAN_STATE_KEY not in ctx.state


def test_an_oversized_plan_is_refused_naming_the_field() -> None:
    """The cap reaches the model as `INVALID_PLAN`, never as a shortened plan."""
    tools = _tools_by_name()
    ctx = _FakeToolContext()
    decisions, requirements = _plan_at_the_cap(MAX_PLAN_DECISIONS)
    decisions = [*decisions, decisions[-1].model_copy(update={"slot": "one_too_many"})]

    result = tools["propose_plan"].func(
        question="q",
        decisions=drafts(decisions),
        evidence_requirements=requirements,
        tool_context=ctx,
    )

    assert result["code"] == ProblemCode.INVALID_PLAN.value
    assert f"at most {MAX_PLAN_DECISIONS} items" in result["message"]
    locations = [error["loc"] for error in result["detail"]["errors"]]
    assert ["decisions"] in locations
    assert PLAN_STATE_KEY not in ctx.state


def test_a_field_name_the_model_supplies_reaches_the_refusal_cut_short(
    decision: DecisionFactory, requirement: RequirementFactory
) -> None:
    """An unknown key on a requirement appears in the error location, so the refusal
    cuts each string in a location at a fixed length."""
    tools = _tools_by_name()
    ctx = _FakeToolContext()
    long_key = "K" * 20_000
    dataset = requirement().model_dump(mode="json") | {long_key: 1}

    result = tools["propose_plan"].func(
        question="q",
        decisions=drafts([decision()]),
        evidence_requirements=[dataset],
        tool_context=ctx,
    )

    assert result["code"] == ProblemCode.INVALID_PLAN.value
    rendered = json.dumps(result)
    assert long_key[:200] not in rendered
    assert len(rendered) < 3_000
    assert PLAN_STATE_KEY not in ctx.state


@pytest.mark.parametrize(
    "missing", ["decisions", "evidence_requirements"], ids=["decisions", "requirements"]
)
def test_propose_plan_rejects_a_null_list(
    missing: str, decision: DecisionFactory, requirements: RequirementPair
) -> None:
    # ADK converts a `list[BaseModel]` argument only when the incoming value is
    # already a list, and a JSON `null` is not a missing mandatory argument --
    # the key is present -- so `None` reaches the function body untouched.
    tools = _tools_by_name()
    ctx = _FakeToolContext()
    args: dict[str, object] = {
        "decisions": [decision()],
        "evidence_requirements": list(requirements),
    }
    args[missing] = None

    result = tools["propose_plan"].func(question="q", tool_context=ctx, **args)

    assert result["code"] == ProblemCode.INVALID_PLAN.value
    assert PLAN_STATE_KEY not in ctx.state


# --- Asking the user (`adk_request_input`) ---------------------------------------------


def test_the_ask_tool_is_registered_as_a_long_running_call() -> None:
    tools = _tools_by_name()

    tool = tools["adk_request_input"]

    assert isinstance(tool, LongRunningFunctionTool)
    assert tool.is_long_running is True


def test_asking_the_user_needs_no_candidates() -> None:
    """A referral that carries no candidates (`unsupported_ask`, `requirement_unmeetable`)
    is put to the user with a message alone."""
    tools = _tools_by_name()
    tool = tools["adk_request_input"]

    declaration = tool._get_declaration()

    assert declaration is not None
    schema = declaration.parameters_json_schema
    assert schema is not None
    assert schema.get("required") == ["message"]
    assert tool.func(message="What should change?") is None


def test_asking_the_user_with_valid_candidates_pauses() -> None:
    # A `LongRunningFunctionTool` pauses the turn by returning `None`
    # (`ask_user`'s own docstring, and `google.adk.tools.request_input`'s).
    tools = _tools_by_name()

    result = tools["adk_request_input"].func(
        message="Which source did you mean?",
        candidates=[{"spec_id": SPEC, "label": "a candidate"}],
        recommended_spec_id=SPEC,
        rationale="the only one with full coverage",
    )

    assert result is None


def test_asking_the_user_rejects_a_recommendation_naming_no_candidate() -> None:
    tools = _tools_by_name()

    result = tools["adk_request_input"].func(
        message="Which source did you mean?",
        candidates=[{"spec_id": SPEC, "label": "a candidate"}],
        recommended_spec_id=POPULATION_SPEC,
        rationale="not one of the candidates",
    )

    assert result is not None
    assert result["code"] == ProblemCode.INVALID_ARGUMENTS.value


@pytest.mark.parametrize(
    ("candidates", "options", "refused"),
    [
        ([], [dict(option) for option in COMBINATION_OPTIONS], False),
        ([], [{"id": "add them", "label": "Add them"}, {"id": "no", "label": "No"}], True),
        (
            [{"spec_id": SPEC, "label": "a candidate"}],
            [dict(option) for option in COMBINATION_OPTIONS],
            True,
        ),
    ],
    ids=["refusals-options", "other-options", "options-with-candidates"],
)
def test_asking_the_user_takes_only_the_options_a_refusal_carries(
    candidates: list[dict[str, str]], options: list[dict[str, str]], refused: bool
) -> None:
    tools = _tools_by_name()

    result = tools["adk_request_input"].func(
        message="Add them?", candidates=candidates, options=options
    )

    if refused:
        assert result is not None
        assert result["code"] == ProblemCode.INVALID_ARGUMENTS.value
    else:
        assert result is None


def test_propose_plan_accepts_a_user_election_ask_user_actually_chose(
    decision: DecisionFactory, requirements: RequirementPair
) -> None:
    tools = _tools_by_name()
    ctx = _FakeToolContext()
    ctx.session.events = [
        _ask_event([{"spec_id": SPEC, "label": "a"}, {"spec_id": POPULATION_SPEC, "label": "b"}]),
        _answer_event(SPEC),
    ]
    dataset, derived = requirements
    elected = decision(
        chosen="total debt stock",
        resolved_by=ResolvedBy.USER,
        fixes=DecisionFixes(source_spec_id=SPEC),
    )

    result = tools["propose_plan"].func(
        question="q",
        answer_mode=AnswerMode.COMPUTED,
        decisions=drafts([elected]),
        evidence_requirements=[dataset.model_copy(update={"source_spec_id": SPEC}), derived],
        tool_context=ctx,
    )

    assert result["outcome"] == Outcome.OK.value
    assert PLAN_STATE_KEY in ctx.state


@pytest.mark.parametrize(
    ("source_mismatch", "picked_before", "answer", "pin", "resolved_by", "stated_after"),
    [
        (True, False, SPEC, SPEC, ResolvedBy.USER, None),
        (
            True,
            False,
            SPEC,
            POPULATION_SPEC,
            ResolvedBy.PLANNER_DEFAULT,
            "World Development Indicators",
        ),
        (True, False, None, SPEC, ResolvedBy.PLANNER_DEFAULT, "World Development Indicators"),
        (True, True, None, SPEC, ResolvedBy.PLANNER_DEFAULT, "World Development Indicators"),
        (False, False, SPEC, SPEC, ResolvedBy.USER, "World Development Indicators"),
    ],
    ids=[
        "user-picks-the-pin",
        "user-picks-another",
        "free-text-answer",
        "picked-before-the-referral",
        "other-unmeetable-cause",
    ],
)
def test_a_pin_the_user_elects_over_a_stated_source_not_found_replaces_it(
    decision: DecisionFactory,
    requirement: Callable[..., DatasetRequirement],
    derived_requirement: Callable[..., EvidenceRequirement],
    source_mismatch: bool,
    picked_before: bool,
    answer: str | None,
    pin: str,
    resolved_by: ResolvedBy,
    stated_after: str | None,
) -> None:
    """The source the requirement states names none of the sources discovery found,
    so the referral offers those, and the one the user picks replaces the stated
    source; left in place, the stated source would refuse the pick on the next
    translate. A pin the user did not pick, after a pick of another or a free-text
    answer, keeps the stated source, so the next translate refuses it again; so
    does a pick the user made answering an ask raised before the referral. A
    referral for any other cause offers nothing, so nothing gives way."""
    tools = _tools_by_name()
    ctx = _FakeToolContext()
    stated = requirement(source="World Development Indicators")
    _propose(tools, ctx, decision(), (stated, derived_requirement()))
    if picked_before:
        ctx.session.events[:0] = [
            _ask_event([{"spec_id": SPEC, "label": "earlier"}], call_id="earlier"),
            _answer_event(SPEC, call_id="earlier"),
        ]
    found = tuple(
        SelectedSource(spec_id=spec_id, sentence=f"debt service {spec_id}", has_data=True)
        for spec_id in (SPEC, POPULATION_SPEC)
    )
    save_referral(
        ctx.state,
        Referral(
            kind=ReferralKind.REQUIREMENT_UNMEETABLE,
            reason="r",
            requirement_id=stated.id,
            candidates=found if source_mismatch else (),
            source_mismatch=source_mismatch,
        ),
    )
    ctx.session.events.append(_referred_event(ReferralKind.REQUIREMENT_UNMEETABLE, stated.id))
    ctx.session.events.extend(
        [
            _ask_event([{"spec_id": s.spec_id, "label": s.sentence} for s in found]),
            _answer_event(answer),
        ]
        if answer is not None
        else _ask_answered_events("why can't you use WDI?")
    )
    elected = decision(
        chosen="total debt stock",
        resolved_by=resolved_by,
        fixes=DecisionFixes(source_spec_id=pin),
    )

    result = _propose(tools, ctx, elected, (stated.model_copy(update={"source_spec_id": pin}),))

    assert result["outcome"] == Outcome.OK.value, result
    stored = load_plan(ctx.state)
    assert stored is not None
    pinned = stored.requirement(stated.id)
    assert isinstance(pinned, DatasetRequirement)
    assert (pinned.source_spec_id, pinned.source) == (pin, stated_after)


# A dataset of education aid as a search in the conversation recorded it, over
# every year with data, and the same dataset as Search found it for "the last
# five years": a closed range.
_AID_IDENTITY: dict[str, Any] = {
    "identity_version": "spec1",
    "dataset": "education aid",
    "requested_window": {"kind": "all"},
}
_AID_SPEC = spec_id_for(canonicalize(_AID_IDENTITY))
_AID_2020_2024 = {
    **_AID_IDENTITY,
    "requested_window": {"kind": "range", "start_year": 2020, "end_year": 2024},
}
_AID_2020_2024_SPEC = spec_id_for(canonicalize(_AID_2020_2024))


@pytest.fixture
def aid_on_record() -> Iterator[None]:
    """Both education-aid identities on record, as `harness.studio` sets the lookup for a segment."""
    recorded = {
        spec_id_for(canonicalize(identity)): RecordedIdentity(identity=identity)
        for identity in (_AID_IDENTITY, _AID_2020_2024)
    }
    token = current_recorded_identity.set(recorded.get)
    yield
    current_recorded_identity.reset(token)


def _propose_pinned(
    tools: dict[str, FunctionTool],
    ctx: object,
    decision: Decision,
    requirements: Sequence[EvidenceRequirement],
    *,
    amends_version: int | None = None,
) -> dict[str, object]:
    return tools["propose_plan"].func(
        question="q",
        answer_mode=AnswerMode.SERIES,
        decisions=drafts([decision]),
        evidence_requirements=list(requirements),
        amends_version=amends_version,
        tool_context=ctx,
    )


@pytest.mark.usefixtures("aid_on_record")
def test_a_recorded_pin_over_two_periods_is_copied_over_each(
    pinning_decision: Callable[..., Decision],
    requirement: Callable[..., DatasetRequirement],
) -> None:
    """Two periods of a recorded pin retrieve two spec_ids: the decision the
    model writes fixes no pin, so each requirement takes its own copy, and
    the translator retrieves each with no search."""
    tools = _tools_by_name()
    ctx = _FakeToolContext()

    result = _propose_pinned(
        tools,
        ctx,
        pinning_decision(_AID_SPEC, resolved_by=ResolvedBy.PLANNER_DEFAULT),
        [
            requirement(source_spec_id=_AID_SPEC, period=Period(start=2024, end=2024)),
            requirement(
                id="earlier", source_spec_id=_AID_SPEC, period=Period(start=2015, end=2015)
            ),
        ],
    )

    assert result["outcome"] == Outcome.OK.value, result
    plan = load_plan(ctx.state)
    assert plan is not None
    pins = [
        item.source_spec_id
        for item in plan.evidence_requirements
        if isinstance(item, DatasetRequirement)
    ]
    copies = windowed_pins(ctx.state)
    assert len(set(pins)) == 2
    assert {copies[pin].windowed_from for pin in pins if pin is not None} == {_AID_SPEC}


@pytest.mark.usefixtures("aid_on_record")
@pytest.mark.parametrize(
    ("pin", "copies"),
    [(_AID_2020_2024_SPEC, {}), (_AID_SPEC, {_AID_2020_2024_SPEC: _AID_SPEC})],
    ids=["recorded-over-the-period", "recorded-over-every-year"],
)
def test_a_recorded_pin_is_copied_over_the_requirements_period_only_when_its_window_differs(
    pinning_decision: Callable[..., Decision],
    requirement: Callable[..., DatasetRequirement],
    pin: str,
    copies: dict[str, str],
) -> None:
    """A pin recorded over the requirement's own period is retrieved as it
    stands; one recorded over other years is copied over that period."""
    tools = _tools_by_name()
    ctx = _FakeToolContext()

    result = _propose_pinned(
        tools,
        ctx,
        pinning_decision(pin, resolved_by=ResolvedBy.PLANNER_DEFAULT),
        [requirement(source_spec_id=pin, period=Period(start=2020, end=2024))],
    )

    assert result["outcome"] == Outcome.OK.value, result
    plan = load_plan(ctx.state)
    assert plan is not None
    (pinned,) = plan.evidence_requirements
    assert isinstance(pinned, DatasetRequirement)
    assert pinned.source_spec_id == _AID_2020_2024_SPEC
    assert {
        spec_id: copy.windowed_from for spec_id, copy in windowed_pins(ctx.state).items()
    } == copies


# --- the envelope's own spec_id, synthesized into a Decision ---------------------------


@pytest.mark.parametrize(
    "earlier_turn", [False, True], ids=["this_turns_block", "an_earlier_turns_block"]
)
def test_a_verbatim_pin_of_a_settled_source_records_as_written(
    decision: DecisionFactory, requirements: RequirementPair, earlier_turn: bool
) -> None:
    """A pin of a spec_id a search settled on, offering no other measure,
    needs no decision: the plan records the model's decisions and the pin
    as written."""
    tools = _tools_by_name()
    ctx = _FakeToolContext()
    _show(ctx, SPEC, earlier_turn=earlier_turn)
    dataset, derived = requirements
    pinned = dataset.model_copy(update={"source_spec_id": SPEC})

    result = tools["propose_plan"].func(
        question="q",
        answer_mode=AnswerMode.COMPUTED,
        decisions=drafts([decision()]),
        evidence_requirements=[pinned, derived],
        tool_context=ctx,
    )

    assert result["outcome"] == Outcome.OK.value, result
    plan = load_plan(ctx.state)
    assert plan is not None
    assert plan.decisions == (decision(),)
    assert plan.requirement(dataset.id) == pinned


# Search's answer to "Bilateral plus imputed multilateral aid from France to
# Kenya": gross disbursements elected, net disbursements offered beside it.
_FRANCE_KENYA = json.loads(
    (
        Path(__file__).parents[3]
        / "data-mcp"
        / "tests"
        / "fixtures"
        / "resolve_aid_france_kenya.json"
    ).read_text()
)["parts"][0]["spec"]
_GROSS = _FRANCE_KENYA["spec_id"]
_NET = _FRANCE_KENYA["resolved"]["alternative_measures"][0]["definition"]["spec_id"]


def _measure_ask_events(picked: str, *, offered: tuple[str, str] = (_GROSS, _NET)) -> list[Any]:
    """The orchestrator's `ask_user` on how to count the aid, through the clarify
    interrupt, each option offered under `offered`, and the user's pick of `picked`."""
    return [
        _clarify_ask_event(
            [
                {"spec_id": offered[0], "label": "Gross disbursements"},
                {"spec_id": offered[1], "label": "Net disbursements"},
            ]
        ),
        _clarify_answer_event(_option(picked)),
    ]


def _count_and_prices_events(count: dict[str, str], prices: dict[str, str]) -> list[Any]:
    """One `ask_user` call with two questions, how to count the aid and in which
    prices, and the answers `count` and `prices` to them. The prices options carry
    no spec_id, so their ids are the positional `o1` and `o2`."""
    return [
        _clarify_ask_event(
            [
                {"spec_id": _GROSS, "label": "Gross disbursements"},
                {"spec_id": _NET, "label": "Net disbursements"},
            ],
            [
                {"spec_id": "o1", "label": "Current prices"},
                {"spec_id": "o2", "label": "Constant prices"},
            ],
        ),
        _clarify_answer_event(count, prices),
    ]


def _legacy_measure_ask_events(picked: str) -> list[Any]:
    """The ask of `_measure_ask_events` in the single-question pause shape."""
    return [
        _legacy_clarify_ask_event(
            [
                {"spec_id": _GROSS, "label": "Gross disbursements"},
                {"spec_id": _NET, "label": "Net disbursements"},
            ]
        ),
        _legacy_clarify_answer_event(picked),
    ]


@pytest.mark.parametrize(
    ("pin", "events", "chosen", "resolved_by"),
    [
        (_GROSS, [], "gross disbursements", ResolvedBy.PLANNER_DEFAULT),
        (_GROSS, _measure_ask_events(_GROSS), "gross disbursements", ResolvedBy.USER),
        (_GROSS, _measure_ask_events(_NET), "gross disbursements", ResolvedBy.PLANNER_DEFAULT),
        (_NET, [], "net disbursements", ResolvedBy.PLANNER_DEFAULT),
        (_NET, _measure_ask_events(_NET), "net disbursements", ResolvedBy.USER),
        (
            _NET,
            _measure_ask_events("o2", offered=("o1", "o2")),
            "net disbursements",
            ResolvedBy.PLANNER_DEFAULT,
        ),
        (
            _GROSS,
            _count_and_prices_events(_option(_GROSS), _SKIPPED),
            "gross disbursements",
            ResolvedBy.USER,
        ),
        (
            _GROSS,
            _count_and_prices_events(_SKIPPED, _option("o1")),
            "gross disbursements",
            ResolvedBy.PLANNER_DEFAULT,
        ),
        (
            _GROSS,
            _count_and_prices_events(_SKIPPED, _option(_GROSS)),
            "gross disbursements",
            ResolvedBy.PLANNER_DEFAULT,
        ),
        (
            _GROSS,
            _legacy_measure_ask_events(_GROSS),
            "gross disbursements",
            ResolvedBy.USER,
        ),
    ],
    ids=[
        "elected",
        "elected-asked",
        "elected-other-picked",
        "sibling",
        "sibling-asked",
        "same-label-other-ask",
        "two-questions-count-answered",
        "two-questions-count-skipped",
        "two-questions-id-from-other-question",
        "legacy-single-candidate-pick",
    ],
)
def test_a_pin_of_an_aid_measure_states_how_the_flow_is_counted(
    decision: DecisionFactory,
    requirements: RequirementPair,
    pin: str,
    events: list[Any],
    chosen: str,
    resolved_by: ResolvedBy,
) -> None:
    """A pin of the measure search elected, or of one it offers beside it,
    gets the measure decision, settled by the user when they picked that
    measure's own spec_id answering the orchestrator's `ask_user`. A pick
    under any other id, the same label included, settles nothing here."""
    tools = _tools_by_name()
    ctx = _FakeToolContext()
    _show_block(ctx, _FRANCE_KENYA)
    ctx.session.events = list(events)
    dataset, derived = requirements

    result = tools["propose_plan"].func(
        question="q",
        answer_mode=AnswerMode.COMPUTED,
        decisions=drafts([decision()]),
        evidence_requirements=[dataset.model_copy(update={"source_spec_id": pin}), derived],
        tool_context=ctx,
    )

    assert result["outcome"] == Outcome.OK.value, result
    plan = load_plan(ctx.state)
    assert plan is not None
    (measure,) = [d for d in plan.decisions if d.fixes and d.fixes.source_spec_id == pin]
    assert measure.options == ("gross disbursements", "net disbursements")
    assert (measure.chosen, measure.resolved_by) == (chosen, resolved_by)
    assert measure in shown_decisions(plan)


# Search's answers to "gross ODA disbursements" and "ODA disbursements" from
# Japan to Kenya: the same gross disbursements source offering net, the
# netting basis named by the first query and elected by the engine for the second.
_MEASURE_NAMED = json.loads(
    (Path(__file__).parents[1] / "fixtures" / "search_measure_named.json").read_text()
)["blocks"]


@pytest.mark.parametrize(
    ("variable", "resolved_by", "rationale"),
    [
        (
            "gross ODA disbursements",
            ResolvedBy.USER,
            "The user's question named gross disbursements.",
        ),
        (
            "ODA disbursements",
            ResolvedBy.PLANNER_DEFAULT,
            "Search elected gross disbursements for this source.",
        ),
    ],
    ids=["named", "elected"],
)
def test_a_measure_the_question_named_is_the_users_decision(
    decision: DecisionFactory,
    requirements: RequirementPair,
    variable: str,
    resolved_by: ResolvedBy,
    rationale: str,
) -> None:
    """A pin of the measure the words of the user's question named, as
    search's trace records them, is the user's measure decision."""
    tools = _tools_by_name()
    ctx = _FakeToolContext()
    block = _MEASURE_NAMED[variable]
    _show_block(ctx, block)
    dataset, derived = requirements

    result = tools["propose_plan"].func(
        question="q",
        answer_mode=AnswerMode.COMPUTED,
        decisions=drafts([decision()]),
        evidence_requirements=[
            dataset.model_copy(update={"source_spec_id": block["spec_id"]}),
            derived,
        ],
        tool_context=ctx,
    )

    assert result["outcome"] == Outcome.OK.value, result
    plan = load_plan(ctx.state)
    assert plan is not None
    (measure,) = [
        d for d in plan.decisions if d.fixes and d.fixes.source_spec_id == block["spec_id"]
    ]
    assert (measure.chosen, measure.resolved_by, measure.rationale) == (
        "gross disbursements",
        resolved_by,
        rationale,
    )


def test_a_pick_after_the_plan_pinned_the_measure_makes_its_decision_the_users(
    decision: DecisionFactory, requirements: RequirementPair
) -> None:
    """The plan already pins gross on the planner's say when the user picks it
    answering a later ask: the next proposal rewrites that decision as theirs."""
    tools = _tools_by_name()
    ctx = _FakeToolContext()
    _show_block(ctx, _FRANCE_KENYA)
    dataset, derived = requirements
    pinned = [dataset.model_copy(update={"source_spec_id": _GROSS}), derived]
    first = tools["propose_plan"].func(
        question="q",
        answer_mode=AnswerMode.COMPUTED,
        decisions=drafts([decision()]),
        evidence_requirements=pinned,
        tool_context=ctx,
    )
    assert first["outcome"] == Outcome.OK.value, first
    ctx.session.events = _measure_ask_events(_GROSS)

    second = tools["propose_plan"].func(
        evidence_requirements=pinned, amends_version=first["version"], tool_context=ctx
    )

    assert second["outcome"] == Outcome.OK.value, second
    plan = load_plan(ctx.state)
    assert plan is not None
    (measure,) = [d for d in plan.decisions if d.fixes and d.fixes.source_spec_id == _GROSS]
    assert (measure.chosen, measure.resolved_by) == ("gross disbursements", ResolvedBy.USER)


def _amendment_events(text: str) -> list[Any]:
    """The approval card (`harness.graph.APPROVE_INTERRUPT`) and the user's amendment of it."""
    card = create_request_input_event(RequestInput(interrupt_id="approve", payload={}))
    answer = create_request_input_response(
        "approve", {"result": {"approved": None, "amendment": text}}
    )
    return [card, Event(content=types.Content(role="user", parts=[answer]))]


@pytest.mark.parametrize("amended", [True, False], ids=["amended", "not-amended"])
def test_a_re_pin_to_another_measure_is_the_planners_even_after_an_amendment(
    decision: DecisionFactory, requirements: RequirementPair, amended: bool
) -> None:
    """An amendment at approval names no spec_id, so a re-pin from gross to
    net made while answering one is the planner's, as it is with no
    amendment on record: the pin moves to net, and the measure decision says
    search elected gross."""
    tools = _tools_by_name()
    ctx = _FakeToolContext()
    _show_block(ctx, _FRANCE_KENYA)
    dataset, derived = requirements
    first = tools["propose_plan"].func(
        question="q",
        answer_mode=AnswerMode.COMPUTED,
        decisions=drafts([decision()]),
        evidence_requirements=[dataset.model_copy(update={"source_spec_id": _GROSS}), derived],
        tool_context=ctx,
    )
    assert first["outcome"] == Outcome.OK.value, first
    if amended:
        ctx.session.events = _amendment_events("make it a bar chart")
    re_pinned = dataset.model_copy(update={"source_spec_id": _NET})

    second = tools["propose_plan"].func(
        evidence_requirements=[re_pinned, derived],
        amends_version=first["version"],
        tool_context=ctx,
    )

    assert second["outcome"] == Outcome.OK.value, second
    plan = load_plan(ctx.state)
    assert plan is not None
    (pinned,) = [r for r in plan.evidence_requirements if r.id == dataset.id]
    assert isinstance(pinned, DatasetRequirement)
    assert pinned.source_spec_id == _NET
    (measure,) = [d for d in plan.decisions if d.fixes and d.fixes.source_spec_id == _NET]
    assert (measure.chosen, measure.resolved_by, measure.rationale) == (
        "net disbursements",
        ResolvedBy.PLANNER_DEFAULT,
        "Chosen in place of gross disbursements, the measure search elected.",
    )


class _ResolveOnly:
    """A surface answering `resolve_definition` from `resolving` and recording any other call."""

    def __init__(self, resolving: DataMcpDefinitions) -> None:
        self.resolving = resolving
        self.other_calls: list[str] = []

    async def list_tools(self) -> list[Any]:
        return []

    async def call_tool(self, *, tool: ToolId, arguments: Any, on_progress: Any = None) -> Any:
        if tool.name != "resolve_definition":
            self.other_calls.append(tool.name)
            raise AssertionError(f"{tool.name} called")
        return await self.resolving.call_tool(
            tool=tool, arguments=arguments, on_progress=on_progress
        )


@pytest.mark.parametrize("from_translator", [False, True], ids=["search", "translator-session"])
async def test_an_amendment_re_pins_an_aid_requirement_to_the_measure_offered_beside_it(
    decision: DecisionFactory, requirement: Callable[..., DatasetRequirement], from_translator: bool
) -> None:
    """ "Use net disbursements": the patch pins the sibling spec_id the plan
    section names, the measure decision follows it with no decision of the
    model's, and the pin is windowed to the requirement's years as the
    elected one was. The translator reads the copy from Search's answer for
    its identity, recorded from the block that offered it, with no search
    by measure words."""
    tools = _tools_by_name()
    ctx = _FakeToolContext()
    _show_block(ctx, _FRANCE_KENYA)
    dataset = requirement(
        id="aid",
        measure="official development assistance",
        entities=[Entity(name="France", role="donor"), Entity(name="Kenya", role="recipient")],
        period=Period(start=2015, end=2024),
    )
    identities = _turn_identities({_GROSS: _FRANCE_KENYA}, ())
    if from_translator:
        dataset = dataset.model_copy(update={"unit": Unit(currency="USD", prices="current")})
        save_log(
            ctx.state,
            DiscoveryLog(sources={"aid": {_GROSS: source_from_spec_block(_FRANCE_KENYA)}}),
        )
        identities = {}
    token = current_recorded_identity.set(identities.get)
    try:
        first = tools["propose_plan"].func(
            question="q",
            answer_mode=AnswerMode.SERIES,
            decisions=drafts([decision()]),
            evidence_requirements=[dataset.model_copy(update={"source_spec_id": _GROSS})],
            tool_context=ctx,
        )
        assert first["outcome"] == Outcome.OK.value, first
        second = tools["propose_plan"].func(
            evidence_requirements=[dataset.model_copy(update={"source_spec_id": _NET})],
            amends_version=first["version"],
            tool_context=ctx,
        )
    finally:
        current_recorded_identity.reset(token)

    assert second["outcome"] == Outcome.OK.value, second
    plan = load_plan(ctx.state)
    assert plan is not None
    (copy,) = windowed_pins(ctx.state).values()
    assert copy.windowed_from == _NET
    (pinned,) = plan.evidence_requirements
    assert isinstance(pinned, DatasetRequirement)
    assert pinned.source_spec_id == copy.spec_id
    (measure,) = [d for d in plan.decisions if d.fixes and d.fixes.source_spec_id]
    assert (measure.slot, measure.chosen) == (
        search_pin_slot(copy.spec_id),
        "net disbursements",
    )

    answered = {
        **_FRANCE_KENYA,
        "spec_id": copy.spec_id,
        "definition": {"schema_version": 1, "spec_id": copy.spec_id, "identity": copy.identity},
    }
    surface = _ResolveOnly(DataMcpDefinitions(InMemoryDefinitionStore(), served=[answered]))
    discover = {t.name: t for t in build_discovery_tools(surface=cast(ToolSurface, surface))}
    result = await discover["discover_source"].func(requirement_id="aid", tool_context=ctx)

    assert result["status"] == "definite", result
    assert surface.other_calls == []
    assert surface.resolving.calls == [{"spec_id": copy.spec_id, "identity": copy.identity}]
    assert [record.tool for record in load_log(ctx.state).records] == ["pin"]


def test_a_measure_decision_moved_to_a_copy_over_another_period_keeps_its_measures(
    decision: DecisionFactory,
) -> None:
    measure = decision(
        slot=search_pin_slot(_GROSS),
        options=("gross disbursements", "net disbursements"),
        chosen="gross disbursements",
        fixes=DecisionFixes(source_spec_id=_GROSS),
    )

    (moved,), _ = _repinned(
        [measure],
        [],
        decision_pins={measure.slot: SPEC},
        requirement_pins={},
        slots={measure.slot: search_pin_slot(SPEC)},
    )

    assert (moved.slot, moved.fixes) == (search_pin_slot(SPEC), DecisionFixes(source_spec_id=SPEC))
    assert (moved.options, moved.chosen) == (measure.options, measure.chosen)


def test_propose_plan_patches_the_citation_on_a_round_that_already_knows_the_measure_decision(
    decision: DecisionFactory, requirements: RequirementPair
) -> None:
    """A correction round resubmits the pinned requirement by id, which
    replaces it wholesale and drops its citation -- `propose_plan`'s OK
    result never told the model the measure decision's slot, so it has
    nothing to cite. `previous.decisions` already carries that decision from
    the first round, so `_measure_decisions` must still add the
    citation to `decision_ids` even though the decision itself is already
    known and only the append to `decisions` is skipped.
    """
    tools = _tools_by_name()
    ctx = _FakeToolContext()
    _show_block(ctx, _FRANCE_KENYA)
    dataset, derived = requirements

    first = tools["propose_plan"].func(
        question="q",
        answer_mode=AnswerMode.COMPUTED,
        decisions=drafts([decision()]),
        evidence_requirements=[dataset.model_copy(update={"source_spec_id": _GROSS}), derived],
        tool_context=ctx,
    )
    assert first["outcome"] == Outcome.OK.value
    plan = load_plan(ctx.state)
    assert plan is not None
    synthesized = next(d for d in plan.decisions if d.fixes and d.fixes.source_spec_id == _GROSS)

    second = tools["propose_plan"].func(
        question="q",
        answer_mode=AnswerMode.COMPUTED,
        decisions=drafts([decision()]),
        evidence_requirements=[dataset.model_copy(update={"source_spec_id": _GROSS}), derived],
        amends_version=plan.version,
        tool_context=ctx,
    )

    assert second["outcome"] == Outcome.OK.value
    amended = load_plan(ctx.state)
    assert amended is not None
    assert [d.slot for d in amended.decisions if d.fixes and d.fixes.source_spec_id == _GROSS] == [
        synthesized.slot
    ]
    assert synthesized.slot in amended.requirement(dataset.id).decision_ids


def _scoped_gross(*, basis_alternative: str = "available") -> dict[str, Any]:
    """`_FRANCE_KENYA` carrying the US health to Africa scope choices
    (`_switchable_scope`), its grant-equivalent basis `basis_alternative`; as
    "unpublished", that basis and net, otherwise blocked by the sector, are both
    unpublished for the part."""
    scope = _switchable_scope()
    basis = next(choice for choice in scope["choices"] if choice["axis"] == "measure_basis")
    grant = next(o for o in basis["options"] if o["option_id"] == "measure_basis:grant_equivalent")
    unpublished = {"status": "unavailable", "code": "basis_not_published", "detail": {}}
    grant["availability"] = {"status": basis_alternative, "detail": {}}
    if basis_alternative == "unavailable":
        grant["availability"] = unpublished
    if basis_alternative == "unpublished":
        for option in basis["options"]:
            if option["option_id"] != basis["active"]:
                option["availability"] = unpublished
    return {**_FRANCE_KENYA, "resolved": {**_FRANCE_KENYA["resolved"], "scope_choices": scope}}


@pytest.mark.parametrize(
    ("basis_alternative", "axes"),
    [
        ("available", ["sector", "region", "measure_basis"]),
        ("partial", ["sector", "region", "measure_basis"]),
        ("unavailable", ["sector", "region", "measure_basis"]),
        ("unpublished", ["sector", "region"]),
    ],
)
def test_a_pin_states_each_scope_choice_with_another_option_to_switch_to(
    decision: DecisionFactory,
    requirements: RequirementPair,
    basis_alternative: str,
    axes: list[str],
) -> None:
    """A pin of a source whose part carries scope choices gets one decision
    per choice with another option that can be served, after the measure
    decision and cited by the requirement pinning it: the user's where their
    words or an earlier choice settled it, the planner's where Search's
    default did. A basis choice whose other options are blocked only by
    another choice, as net beside CRS 120 alone, still states one: the user
    reaches them by changing that choice. One whose other options are out of
    reach for their own reasons states none."""
    tools = _tools_by_name()
    ctx = _FakeToolContext()
    _show_block(ctx, _scoped_gross(basis_alternative=basis_alternative))
    dataset, derived = requirements

    result = tools["propose_plan"].func(
        question="q",
        answer_mode=AnswerMode.COMPUTED,
        decisions=drafts([decision()]),
        evidence_requirements=[dataset.model_copy(update={"source_spec_id": _GROSS}), derived],
        tool_context=ctx,
    )

    assert result["outcome"] == Outcome.OK.value, result
    plan = load_plan(ctx.state)
    assert plan is not None
    scoped = [d for d in plan.decisions if is_scope_choice(d)]
    assert [d.slot for d in plan.decisions] == [
        "measure",
        search_pin_slot(_GROSS),
        *(scope_choice_slot(_GROSS, axis) for axis in axes),
    ]
    assert [(d.chosen, d.resolved_by) for d in scoped] == [
        ("health (CRS 120)", ResolvedBy.USER),
        ("Africa's countries only", ResolvedBy.USER),
        ("gross disbursements, current USD", ResolvedBy.PLANNER_DEFAULT),
    ][: len(axes)]
    base = _switchable_scope()["id"]
    assert [d.scope.choice_id for d in scoped if d.scope] == [f"{base}:{axis}" for axis in axes]
    assert [d.active_option_id for d in scoped] == [
        "sector:120",
        "region:members_only",
        "measure_basis:gross",
    ][: len(axes)]
    assert plan.requirement(dataset.id).decision_ids == (
        "measure",
        search_pin_slot(_GROSS),
        *(d.slot for d in scoped),
    )
    assert set(scoped) <= set(shown_decisions(plan))


def test_a_restated_requirement_citing_only_its_scope_decisions_still_cites_the_measure(
    decision: DecisionFactory, requirements: RequirementPair
) -> None:
    """A correction round restates the pinned requirement citing the scope
    decisions it was shown and not the measure decision: a scope decision
    states no measure, so the measure citation is added back."""
    tools = _tools_by_name()
    ctx = _FakeToolContext()
    _show_block(ctx, _scoped_gross())
    dataset, derived = requirements
    pinned = dataset.model_copy(update={"source_spec_id": _GROSS})
    first = tools["propose_plan"].func(
        question="q",
        answer_mode=AnswerMode.COMPUTED,
        decisions=drafts([decision()]),
        evidence_requirements=[pinned, derived],
        tool_context=ctx,
    )
    assert first["outcome"] == Outcome.OK.value, first
    scope_slots = tuple(scope_choice_slot(_GROSS, axis) for axis in ("sector", "region"))

    second = tools["propose_plan"].func(
        evidence_requirements=[
            pinned.model_copy(update={"decision_ids": ("measure", *scope_slots)}),
        ],
        amends_version=first["version"],
        tool_context=ctx,
    )

    assert second["outcome"] == Outcome.OK.value, second
    plan = load_plan(ctx.state)
    assert plan is not None
    assert search_pin_slot(_GROSS) in plan.requirement(dataset.id).decision_ids


def test_scope_decisions_past_the_decision_cap_are_left_out_with_no_refusal(
    decision: DecisionFactory, requirements: RequirementPair
) -> None:
    """With the model's decisions and the measure decision one short of
    `MAX_PLAN_DECISIONS`, the first scope choice is written and the others
    are neither written nor cited, and the plan is recorded."""
    tools = _tools_by_name()
    ctx = _FakeToolContext()
    _show_block(ctx, _scoped_gross())
    dataset, derived = requirements
    own = [decision(slot=f"d{index}") for index in range(MAX_PLAN_DECISIONS - 2)]
    cited = tuple(d.slot for d in own)

    result = tools["propose_plan"].func(
        question="q",
        answer_mode=AnswerMode.COMPUTED,
        decisions=drafts(own),
        evidence_requirements=[
            dataset.model_copy(update={"source_spec_id": _GROSS, "decision_ids": cited}),
            derived.model_copy(update={"decision_ids": cited[:1]}),
        ],
        tool_context=ctx,
    )

    assert result["outcome"] == Outcome.OK.value, result
    plan = load_plan(ctx.state)
    assert plan is not None
    assert len(plan.decisions) == MAX_PLAN_DECISIONS
    assert [d.slot for d in plan.decisions if is_scope_choice(d)] == [
        scope_choice_slot(_GROSS, "sector")
    ]
    assert plan.requirement(dataset.id).decision_ids[-2:] == (
        search_pin_slot(_GROSS),
        scope_choice_slot(_GROSS, "sector"),
    )


def test_a_repin_at_the_decision_cap_states_the_new_pins_scope_decision(
    decision: DecisionFactory, requirement: Callable[..., DatasetRequirement]
) -> None:
    """A plan at `MAX_PLAN_DECISIONS` re-pins its US total from the grant-equivalent
    count to the gross one. The old pin's decisions lose their last citation and
    go, so the new pin's basis decision is written and cited in their place."""
    tools = _tools_by_name()
    ctx = _FakeToolContext()
    _show_blocks(
        ctx,
        basis_block(TOTAL, narrow=False, active="measure_basis:grant_equivalent"),
        basis_block(TOTAL_GROSS, narrow=False, active="measure_basis:gross"),
    )
    own = [decision(slot=f"d{index}") for index in range(MAX_PLAN_DECISIONS - 2)]
    total = requirement(
        id="total",
        decision_ids=tuple(d.slot for d in own),
        measure="US total ODA",
        entities=[Entity(name="United States", role="donor")],
        source_spec_id=TOTAL,
    )
    first = tools["propose_plan"].func(
        question="q",
        answer_mode=AnswerMode.SERIES,
        decisions=drafts(own),
        evidence_requirements=[total],
        tool_context=ctx,
    )
    assert first["outcome"] == Outcome.OK.value, first
    plan = load_plan(ctx.state)
    assert plan is not None
    assert len(plan.decisions) == MAX_PLAN_DECISIONS

    second = tools["propose_plan"].func(
        evidence_requirements=[total.model_copy(update={"source_spec_id": TOTAL_GROSS})],
        amends_version=first["version"],
        tool_context=ctx,
    )

    assert second["outcome"] == Outcome.OK.value, second
    plan = load_plan(ctx.state)
    assert plan is not None
    basis = scope_choice_slot(TOTAL_GROSS, "measure_basis")
    assert [d.slot for d in plan.decisions if is_scope_choice(d)] == [basis]
    assert basis in plan.requirement("total").decision_ids


def test_a_windowed_pin_takes_its_scope_decisions_to_the_copy(
    decision: DecisionFactory, requirement: Callable[..., DatasetRequirement]
) -> None:
    """A pin over other years than its search's moves to a windowed copy,
    and its scope decisions move with it, under the copy's slots."""
    tools = _tools_by_name()
    ctx = _FakeToolContext()
    block = _scoped_gross()
    _show_block(ctx, block)
    dataset = requirement(
        id="aid",
        measure="official development assistance",
        entities=[Entity(name="France", role="donor"), Entity(name="Kenya", role="recipient")],
        period=Period(start=2015, end=2024),
        source_spec_id=_GROSS,
    )
    token = current_recorded_identity.set(_turn_identities({_GROSS: block}, ()).get)
    try:
        result = tools["propose_plan"].func(
            question="q",
            answer_mode=AnswerMode.SERIES,
            decisions=drafts([decision()]),
            evidence_requirements=[dataset],
            tool_context=ctx,
        )
    finally:
        current_recorded_identity.reset(token)

    assert result["outcome"] == Outcome.OK.value, result
    plan = load_plan(ctx.state)
    assert plan is not None
    (copy,) = windowed_pins(ctx.state).values()
    scoped = [d for d in plan.decisions if is_scope_choice(d)]
    assert [d.slot for d in scoped] == [
        scope_choice_slot(copy.spec_id, axis) for axis in ("sector", "region", "measure_basis")
    ]
    assert {d.slot for d in scoped} <= set(plan.requirement("aid").decision_ids)


def test_a_scope_decision_moved_to_a_copy_over_another_period_keeps_its_choice() -> None:
    """A windowed copy of a pin takes the scope decisions of the pin under
    the copy's slots, cited under them, with the same choice and options."""
    sector = scope_choice_decisions({_GROSS: PinSource(block=_scoped_gross(), settled=True)})[
        _GROSS
    ][0]
    citing = DatasetRequirement(
        id="aid",
        decision_ids=(sector.slot,),
        measure="aid",
        entities=[Entity(name="Africa", role="recipient")],
        period=Period(start=2015, end=2024),
        source_spec_id=_GROSS,
    )

    (moved,), (moved_requirement,) = _repinned(
        [sector],
        [citing],
        decision_pins={sector.slot: SPEC},
        requirement_pins={citing.id: SPEC},
        slots=pin_slot_moves(_GROSS, SPEC),
    )

    assert is_scope_choice(moved)
    assert (moved.slot, moved.fixes) == (
        scope_choice_slot(SPEC, "sector"),
        DecisionFixes(source_spec_id=SPEC),
    )
    assert (moved.scope, moved.options, moved.chosen) == (
        sector.scope,
        sector.options,
        sector.chosen,
    )
    assert moved_requirement.decision_ids == (moved.slot,)
    assert not is_search_pin(moved)


# --- a calculation's inputs on one measure basis ---


def _share_of_health(
    propose: Callable[..., Any],
    ctx: _FakeToolContext,
    decision: Decision,
    *,
    total: str | None = TOTAL,
    source: str | None = None,
    **intent: Any,
) -> Any:
    """Propose US health aid as a share of the US total, pinning `HEALTH` and `total`,
    the total's source written as `source`."""

    def dataset(
        requirement_id: str, measure: str, pin: str | None, named: str | None = None
    ) -> DatasetRequirement:
        return DatasetRequirement(
            id=requirement_id,
            decision_ids=["measure"],
            measure=measure,
            entities=[Entity(name="United States", role="donor")],
            period=Period(start=2015, end=2024),
            source=named,
            source_spec_id=pin,
        )

    return propose(
        question="What share of US aid goes to health?",
        answer_mode=AnswerMode.COMPUTED,
        decisions=drafts([decision]),
        evidence_requirements=[
            dataset("health", "US aid to health", HEALTH),
            dataset("total", "US total net ODA", total, source),
            ShareOfRequirement(
                id="share",
                decision_ids=["measure"],
                numerator=RequirementSide(input_id="health"),
                denominator=RequirementSide(input_id="total"),
            ),
        ],
        tool_context=ctx,
        **intent,
    )


def _show_blocks(ctx: _FakeToolContext, *blocks: dict[str, Any]) -> None:
    """Record each of `blocks` as a settled answer the planner was shown this entry."""
    save_pin_sources(
        ctx.state, {block["spec_id"]: PinSource(block=block, settled=True) for block in blocks}
    )


@pytest.mark.parametrize("set_unreadable", [False, True], ids=["no-set", "set-unreadable"])
async def test_a_share_of_a_donor_total_counts_the_total_on_the_narrower_parts_basis(
    decision: DecisionFactory, set_unreadable: bool
) -> None:
    """US health aid counted gross on the CRS over the US total counted in grant
    equivalents on DAC1: the total is compiled gross through the resubmit a switch
    posts, the share reads both inputs gross, and the planner's decision says the
    inputs were put on one basis. The compiled total drops the source the planner
    wrote, which discovery would otherwise search back to the grant-equivalent
    total. A named set on the total's block that is not readable, as one naming a
    class of party this release does not know, refuses the compile and posts
    nothing, since a resubmit without it can bind another population."""
    surface = BasisCompiles()
    ctx = _FakeToolContext()
    total_block = basis_block(TOTAL, narrow=False, active="measure_basis:grant_equivalent")
    if set_unreadable:
        total_block["resolved"]["entity_groups"] = {
            "origin": {"label": "DAC countries", "whole_of": ["sovereign_fund"]}
        }
    _show_blocks(ctx, basis_block(HEALTH, narrow=True, active="measure_basis:gross"), total_block)

    result = await _share_of_health(
        _tools_reading(surface)["propose_plan"].func, ctx, decision(), source="OECD DAC1"
    )

    if set_unreadable:
        assert result.get("code") == ProblemCode.OPERATION_UNSUPPORTED.value, result
        assert surface.calls == []
        return
    assert result["outcome"] == Outcome.OK.value, result
    (call,) = surface.calls
    total_scope = basis_block(TOTAL, narrow=False, active="measure_basis:grant_equivalent")[
        "resolved"
    ]["scope_choices"]
    assert (call["spec_id"], call["identity"]) == (
        total_scope["base"]["spec_id"],
        total_scope["base"]["identity"],
    )
    assert "entity_groups" not in call
    assert call["scope_entries"] == [
        {
            "axis": "measure_basis",
            "option_id": "measure_basis:gross",
            "provenance": "explicit",
            "origin": {"kind": "choice", "choice_id": f"{TOTAL}_base:measure_basis"},
        }
    ]
    plan = load_plan(ctx.state)
    assert plan is not None
    assert [
        (r.id, r.source_spec_id)
        for r in plan.evidence_requirements
        if isinstance(r, DatasetRequirement)
    ] == [("health", HEALTH), ("total", TOTAL_GROSS)]
    bases = {
        d.fixes.source_spec_id: d
        for d in plan.decisions
        if d.scope is not None and d.scope.axis == "measure_basis" and d.fixes is not None
    }
    assert {pin: d.active_option_id for pin, d in bases.items()} == {
        HEALTH: "measure_basis:gross",
        TOTAL_GROSS: "measure_basis:gross",
    }
    aligned = bases[TOTAL_GROSS]
    assert aligned.resolved_by is ResolvedBy.PLANNER_DEFAULT
    assert "one basis" in aligned.rationale
    assert aligned.slot in plan.requirement("total").decision_ids
    assert plan.requirement("total").source is None  # type: ignore[union-attr]
    assert not any(d.fixes is not None and d.fixes.source_spec_id == TOTAL for d in plan.decisions)

    # A restatement pinning the total as it was shown takes the compiled pin again.
    again = await _share_of_health(
        _tools_reading(surface)["propose_plan"].func,
        ctx,
        decision(),
        amends_version=plan.version,
    )
    assert again["outcome"] == Outcome.OK.value, again
    amended = load_plan(ctx.state)
    assert amended is not None
    assert amended.requirement("total").source_spec_id == TOTAL_GROSS  # type: ignore[union-attr]
    assert len(surface.calls) == 1


@pytest.mark.parametrize("compiled", [True, False], ids=["compiled-pin", "user-stated"])
def test_a_basis_the_planner_compiled_in_an_earlier_turn_is_not_read_as_the_users(
    decision: DecisionFactory, compiled: bool
) -> None:
    """A later turn shares health aid the user now states as net over the total a
    previous turn compiled on gross. The compiled block's explicit basis entry is
    the planner's, so the user's net is the one basis stated and the total is
    wanted on net; the same entry as the user's own contrasts the two."""
    ctx = _FakeToolContext()
    total = basis_block(TOTAL, narrow=False, active="measure_basis:gross", provenance="explicit")
    if compiled:
        total[PLANNER_AXES_BLOCK_KEY] = ["measure_basis"]
    _show_blocks(
        ctx,
        basis_block(HEALTH, narrow=True, active="measure_basis:net", provenance="explicit"),
        total,
    )

    result = _share_of_health(_tools_by_name()["propose_plan"].func, ctx, decision())

    if compiled:
        assert result["code"] == ProblemCode.BASES_UNALIGNED.value, result
        assert [(r["spec_id"], r["option_ids"]) for r in result["detail"]["realign"]] == [
            (TOTAL, ["measure_basis:net"])
        ]
    else:
        assert result["outcome"] == Outcome.OK.value, result


@pytest.mark.parametrize("provenance", ["explicit", "carried"], ids=["asked-now", "chose-earlier"])
def test_a_basis_the_user_stated_on_one_input_is_the_one_every_input_takes(
    decision: DecisionFactory, provenance: str
) -> None:
    """ "Health aid as a share of net ODA": the user stated net on the total alone,
    in this question or by a choice earlier in the conversation, so the health
    part, though narrower, is the input to compile, on net. With no tool surface
    to compile through, the proposal is refused naming it."""
    ctx = _FakeToolContext()
    _show_blocks(
        ctx,
        basis_block(HEALTH, narrow=True, active="measure_basis:gross"),
        basis_block(TOTAL, narrow=False, active="measure_basis:net", provenance=provenance),
    )

    refused = _share_of_health(_tools_by_name()["propose_plan"].func, ctx, decision())

    assert refused["code"] == ProblemCode.BASES_UNALIGNED.value, refused
    assert refused["detail"] == {
        "requirement_id": "health",
        "realign": [
            {
                "spec_id": HEALTH,
                "option_ids": ["measure_basis:net"],
                "requirement_id": "health",
                "step_id": "share",
            }
        ],
    }
    assert load_plan(ctx.state) is None


def test_a_share_whose_inputs_the_user_stated_on_different_bases_keeps_both(
    decision: DecisionFactory,
) -> None:
    """ "Gross health aid as a share of net ODA": the user stated both bases, so the
    calculation contrasts them and neither input moves."""
    ctx = _FakeToolContext()
    _show_blocks(
        ctx,
        basis_block(HEALTH, narrow=True, active="measure_basis:gross", provenance="explicit"),
        basis_block(TOTAL, narrow=False, active="measure_basis:net", provenance="explicit"),
    )

    result = _share_of_health(_tools_by_name()["propose_plan"].func, ctx, decision())

    assert result["outcome"] == Outcome.OK.value, result
    plan = load_plan(ctx.state)
    assert plan is not None
    assert (
        plan.requirement("health").source_spec_id,  # type: ignore[union-attr]
        plan.requirement("total").source_spec_id,  # type: ignore[union-attr]
    ) == (HEALTH, TOTAL)


async def test_two_requirements_on_one_pin_and_basis_compile_it_once() -> None:
    """A refusal asking two requirements pinning the US total to be counted gross
    compiles the total on gross once, and the next proposal is answered."""
    surface = BasisCompiles()
    ctx = _FakeToolContext()
    _show_blocks(ctx, basis_block(TOTAL, narrow=False, active="measure_basis:grant_equivalent"))
    refusal = {
        "code": ProblemCode.BASES_UNALIGNED.value,
        "detail": {
            "realign": [
                {
                    "spec_id": TOTAL,
                    "option_ids": ["measure_basis:gross"],
                    "requirement_id": requirement_id,
                    "step_id": "share",
                }
                for requirement_id in ("total", "total_again")
            ]
        },
    }
    answers = iter([refusal, {"outcome": Outcome.OK.value}])

    result = await with_source_reads(lambda: next(answers), state=ctx.state, surface=surface)

    assert result == {"outcome": Outcome.OK.value}
    assert surface.tools == ["resolve_definition"]


async def test_a_share_whose_total_is_not_published_on_the_shared_basis_is_refused(
    decision: DecisionFactory,
) -> None:
    """The US total offers no gross count, so the share cannot read both inputs on
    one basis: the plan is refused as unsupported, naming the input and the basis,
    and nothing is compiled."""
    surface = BasisCompiles()
    ctx = _FakeToolContext()
    _show_blocks(
        ctx,
        basis_block(HEALTH, narrow=True, active="measure_basis:gross"),
        basis_block(
            TOTAL, narrow=False, active="measure_basis:grant_equivalent", gross="unavailable"
        ),
    )

    refused = await _share_of_health(_tools_reading(surface)["propose_plan"].func, ctx, decision())

    assert refused["code"] == ProblemCode.OPERATION_UNSUPPORTED.value, refused
    assert refused["detail"] == {
        "operation": UnsupportedOperation.UNSUPPORTED_ASK.value,
        "basis": {
            "spec_id": TOTAL,
            "option_ids": ["measure_basis:gross"],
            "requirement_id": "total",
            "step_id": "share",
        },
    }
    assert surface.calls == []
    assert load_plan(ctx.state) is None


# --- a calculation's inputs on one provider scope ---


def _sum_of_health(
    propose: Callable[..., Any], ctx: _FakeToolContext, decision: Decision, *, accept: bool = False
) -> Any:
    """Propose US health aid plus the US total, pinning `HEALTH` and `TOTAL`; with
    `accept`, the sum accepts a provider difference."""

    def dataset(requirement_id: str, measure: str, pin: str) -> DatasetRequirement:
        return DatasetRequirement(
            id=requirement_id,
            decision_ids=["measure"],
            measure=measure,
            entities=[Entity(name="United States", role="donor")],
            period=Period(start=2015, end=2024),
            source_spec_id=pin,
        )

    return propose(
        question="US health aid plus US total aid?",
        answer_mode=AnswerMode.COMPUTED,
        decisions=drafts([decision]),
        evidence_requirements=[
            dataset("health", "US aid to health", HEALTH),
            dataset("total", "US total ODA", TOTAL),
            SumOfRequirement(
                id="sum",
                decision_ids=["measure"],
                left=RequirementSide(input_id="health"),
                right=RequirementSide(input_id="total"),
                accept_provider_difference=accept,
            ),
        ],
        tool_context=ctx,
    )


def _difference_of_health(
    propose: Callable[..., Any], ctx: _FakeToolContext, decision: Decision
) -> Any:
    """Propose the US total minus US health aid, pinning `TOTAL` and `HEALTH`."""

    def dataset(requirement_id: str, measure: str, pin: str) -> DatasetRequirement:
        return DatasetRequirement(
            id=requirement_id,
            decision_ids=["measure"],
            measure=measure,
            entities=[Entity(name="United States", role="donor")],
            period=Period(start=2015, end=2024),
            source_spec_id=pin,
        )

    return propose(
        question="US total aid minus US health aid?",
        answer_mode=AnswerMode.COMPUTED,
        decisions=drafts([decision]),
        evidence_requirements=[
            dataset("health", "US aid to health", HEALTH),
            dataset("total", "US total ODA", TOTAL),
            ComparisonRequirement(
                id="difference",
                decision_ids=["measure"],
                minuend=RequirementSide(input_id="total"),
                subtrahend=RequirementSide(input_id="health"),
            ),
        ],
        tool_context=ctx,
    )


@pytest.mark.parametrize(
    ("health", "total", "realign"),
    [
        pytest.param(
            ("provider:bilateral", "default"),
            ("provider:inclusive", "default"),
            [(TOTAL, ["provider:bilateral"], "total")],
            id="narrowest-wins",
        ),
        pytest.param(
            ("provider:bilateral", "default"),
            ("provider:inclusive", "explicit"),
            [(HEALTH, ["provider:inclusive"], "health")],
            id="stated-wins",
        ),
    ],
)
@pytest.mark.parametrize(
    "calculation",
    [_share_of_health, _sum_of_health, _difference_of_health],
    ids=["share", "sum", "difference"],
)
def test_a_calculation_counts_its_inputs_on_one_provider_scope(
    decision: DecisionFactory,
    health: tuple[str, str],
    total: tuple[str, str],
    realign: list[tuple[str, list[str], str]],
    calculation: Callable[..., Any],
) -> None:
    """US health aid counted bilateral beside the US total counted with imputed
    multilateral flows: adding them counts the bilateral flows twice, and dividing
    or subtracting them mixes providers' spending, so the inputs are put on one
    provider scope, the one the user stated, else the
    narrower part's. With no tool surface to compile through, the proposal is
    refused naming the input to compile."""
    ctx = _FakeToolContext()
    _show_blocks(
        ctx,
        _from_donors(
            provider_block(HEALTH, narrow=True, active=health[0], provenance=health[1]),
            ["country/USA"],
        ),
        _from_donors(
            provider_block(TOTAL, narrow=False, active=total[0], provenance=total[1]),
            ["country/USA"],
        ),
    )

    refused = calculation(_tools_by_name()["propose_plan"].func, ctx, decision())

    assert refused["code"] == ProblemCode.BASES_UNALIGNED.value, refused
    assert [
        (r["spec_id"], r["option_ids"], r["requirement_id"]) for r in refused["detail"]["realign"]
    ] == realign
    _names_the_requirement(refused, realign[0][2])
    assert load_plan(ctx.state) is None


def _names_the_requirement(refused: Mapping[str, object], requirement_id: str) -> None:
    """The planner's sentence for `refused` names `requirement_id` by its id, and
    the user's reason names it by its name on the plan card."""
    assert f"for requirement {requirement_id!r}" in refusal_sentence(refused)
    named = PlanWords(names={requirement_id: "the named figure"})
    assert "the named figure" in refusal_reason(refused, words=named)


@pytest.mark.parametrize(
    ("calculation", "code"),
    [
        (_share_of_health, Outcome.OK.value),
        (_sum_of_health, ProblemCode.OPERATION_UNSUPPORTED.value),
    ],
    ids=["share", "sum"],
)
def test_inputs_the_user_stated_on_different_provider_scopes_are_shared_and_never_summed(
    decision: DecisionFactory, calculation: Callable[..., Any], code: str
) -> None:
    """ "Bilateral health aid as a share of all US aid with multilateral": the share
    contrasts the two scopes as stated, and a decision of the planner's, cited by
    the share, says the inputs count different providers' spending. The same two
    inputs summed would count the bilateral flows twice, so the sum is refused."""
    ctx = _FakeToolContext()
    _show_blocks(
        ctx,
        provider_block(HEALTH, narrow=True, active="provider:bilateral", provenance="explicit"),
        provider_block(TOTAL, narrow=False, active="provider:inclusive", provenance="explicit"),
    )

    result = calculation(_tools_by_name()["propose_plan"].func, ctx, decision())

    assert result.get("outcome") == code or result.get("code") == code, result
    if code == Outcome.OK.value:
        plan = load_plan(ctx.state)
        assert plan is not None
        (stated,) = [d for d in plan.decisions if d.caveat == "providers_differ"]
        assert stated.slot in plan.requirement("share").decision_ids
    if code == ProblemCode.OPERATION_UNSUPPORTED.value:
        assert result["detail"]["basis"] == {
            "spec_id": TOTAL,
            "option_ids": ["provider:bilateral"],
            "requirement_id": "total",
            "step_id": "sum",
        }
        assert load_plan(ctx.state) is None


@pytest.mark.parametrize(
    ("scopes", "donors", "code"),
    [
        pytest.param(
            ("bilateral", "inclusive"),
            (["country/USA"], ["country/USA"]),
            ProblemCode.OPERATION_UNSUPPORTED.value,
            id="one-donor",
        ),
        pytest.param(
            ("bilateral", "inclusive"),
            (["country/FRA"], ["country/DEU"]),
            Outcome.OK.value,
            id="disjoint-donors",
        ),
        pytest.param(
            ("bilateral", "multilateral"),
            (["country/USA"], ["country/USA"]),
            Outcome.OK.value,
            id="bilateral-and-core-contributions",
        ),
        pytest.param(
            ("contributions", "multilateral"),
            (["country/USA"], ["country/USA"]),
            ProblemCode.OPERATION_UNSUPPORTED.value,
            id="contributions-and-core-contributions",
        ),
        pytest.param(
            ("inclusive", "multilateral"),
            (["country/USA"], ["country/USA"]),
            ProblemCode.OPERATION_UNSUPPORTED.value,
            id="imputed-share-and-core-contributions",
        ),
    ],
)
def test_a_sum_accepting_a_provider_difference_is_still_refused_where_it_counts_flows_twice(
    decision: DecisionFactory,
    scopes: tuple[str, str],
    donors: tuple[list[str], list[str]],
    code: str,
) -> None:
    """US health aid plus the US total on other provider scopes, the sum accepting a
    provider difference. The flag lets each input keep its own scope, so the inputs
    are not realigned, and the sum still refuses where one donor's money would be
    counted twice: bilateral flows in both, or its aid through multilaterals counted
    as core contributions in both, or as core contributions in one and as an imputed
    share of the multilaterals' spending in the other. Over disjoint donors, or over
    bilateral flows and core contributions alone, it is served."""
    ctx = _FakeToolContext()
    _show_blocks(
        ctx,
        _from_donors(
            provider_block(HEALTH, narrow=True, active=f"provider:{scopes[0]}"), donors[0]
        ),
        _from_donors(
            provider_block(TOTAL, narrow=False, active=f"provider:{scopes[1]}"), donors[1]
        ),
    )

    result = _sum_of_health(_tools_by_name()["propose_plan"].func, ctx, decision(), accept=True)

    assert result.get("outcome") == code or result.get("code") == code, result
    if code == ProblemCode.OPERATION_UNSUPPORTED.value:
        assert result["detail"]["basis"]["step_id"] == "sum"
        assert load_plan(ctx.state) is None
    else:
        assert _pins(ctx) == [("health", HEALTH), ("total", TOTAL)]


@pytest.mark.parametrize("accept", [False, True], ids=["aligned", "accepted"])
@pytest.mark.parametrize(
    ("scope", "health_donor", "total_donor", "marked", "code"),
    [
        pytest.param(
            "inclusive",
            "country/USA",
            "InternationalDevelopmentAssociation",
            True,
            ProblemCode.OPERATION_UNSUPPORTED.value,
            id="multilateral-reporter",
        ),
        # IDA stating no marker may be a multilateral's outflows.
        pytest.param(
            "inclusive",
            "country/USA",
            "InternationalDevelopmentAssociation",
            None,
            ProblemCode.OPERATION_UNSUPPORTED.value,
            id="unmarked-organisation",
        ),
        # Two organisations stating no marker: either may be a multilateral whose
        # outflows spend the money the other's imputed share counts, so their own flows
        # being equally unknown says nothing of the streams they count it by.
        pytest.param(
            "inclusive",
            "AsianDevelopmentBank",
            "InternationalDevelopmentAssociation",
            None,
            ProblemCode.OPERATION_UNSUPPORTED.value,
            id="two-unmarked-organisations",
        ),
        pytest.param(
            "inclusive",
            "country/USA",
            "country/USA",
            None,
            Outcome.OK.value,
            id="one-reporter-kind",
        ),
        # Bilateral flows alone: each input counts its reporters' own flows by the one
        # stream, whatever money those count, and no multilateral money of one can be
        # the other's outflows.
        pytest.param(
            "bilateral",
            "country/USA",
            "InternationalDevelopmentAssociation",
            None,
            Outcome.OK.value,
            id="bilateral-beside-an-unmarked-organisation",
        ),
    ],
)
def test_a_sum_on_one_provider_scope_is_refused_where_its_reporters_count_one_money(
    decision: DecisionFactory,
    accept: bool,
    scope: str,
    health_donor: str,
    total_donor: str,
    marked: bool | None,
    code: str,
) -> None:
    """US health aid plus a total, both counted on one provider scope, whether the sum
    aligns its inputs or accepts a provider difference. Where the scope counts imputed
    multilateral flows and the total is IDA's, whose own outflows spend the core
    contributions its funders paid it, both inputs count the donors' aid through
    multilaterals, one as an imputed share and one as IDA's spending, so the sum is
    refused although both hold one option, as it is where both are organisations
    Search states nothing of. Where both report as countries, or the scope counts the
    reporters' own flows alone, they count each money by the same streams, and the
    provider scope says nothing of whether their flows overlap."""
    ctx = _FakeToolContext()
    health = _from_donors(
        provider_block(HEALTH, narrow=True, active=f"provider:{scope}"), [health_donor]
    )
    total = _from_donors(
        provider_block(TOTAL, narrow=False, active=f"provider:{scope}"), [total_donor]
    )
    for block in (health, total):
        (donor,) = block["resolved"]["entities"]
        if not donor["ref"].startswith("country/"):
            donor.update(entity_type="Organization")
            if marked is not None:
                donor.update(multilateral=marked)
    _show_blocks(ctx, health, total)

    result = _sum_of_health(_tools_by_name()["propose_plan"].func, ctx, decision(), accept=accept)

    assert result.get("outcome") == code or result.get("code") == code, result
    if code == Outcome.OK.value:
        assert _pins(ctx) == [("health", HEALTH), ("total", TOTAL)]
        return
    assert result["detail"]["basis"] == {
        "spec_id": TOTAL,
        "option_ids": [f"provider:{scope}"],
        "requirement_id": "total",
        "step_id": "sum",
    }
    assert load_plan(ctx.state) is None


def _from_donors(block: dict[str, Any], donors: list[str]) -> dict[str, Any]:
    """`block` with its part's base and compiled identities bound to `donors`,
    countries, as their origin, all providers when empty, each stated as Search
    states a bound country."""
    resolved = block["resolved"]
    scope = resolved["scope_choices"]
    for identity in (scope["base"]["identity"], scope["compiled"]["identity"]):
        identity["entities"] = {"origin": donors} if donors else {}
    resolved["entities"] = [
        {"ref": donor, "label": donor, "entity_type": "Country", "role": "origin"}
        for donor in donors
    ]
    return block


def _by_instrument(block: dict[str, Any], codes: list[str], *, added: list[str]) -> dict[str, Any]:
    """A copy of `block` with a CRS type-of-finance slot bound to `codes`, one a
    value and several a set, widened by the code index to `added` as Search
    states it (`compiler.sources.slot_codes`)."""
    block = json.loads(json.dumps(block))
    values = [{"ref": {"ref": code}, "value_kind": "enum_value"} for code in codes]
    binding = (
        {"kind": "value", "value": values[0]}
        if len(values) == 1
        else {"kind": "set", "values": values}
    )
    block["resolved"]["slots"].append(
        {
            "key": {"axis": "how", "property": {"ref": "crsTypeOfFinance"}},
            "binding": binding,
            **({"expansion": {"added": [{"ref": code} for code in added]}} if added else {}),
            "summable": True,
        }
    )
    return block


@pytest.mark.parametrize("accept", [False, True], ids=["aligned", "accepted"])
@pytest.mark.parametrize(
    ("loans", "added", "code"),
    [
        pytest.param(
            ["CRSTypeOfFinance400"], ["CRSTypeOfFinance421"], Outcome.OK.value, id="loans"
        ),
        pytest.param(
            ["CRSTypeOfFinance400"],
            ["CRSTypeOfFinance110"],
            ProblemCode.OPERATION_UNSUPPORTED.value,
            id="a-code-in-common",
        ),
        # A set Search states no expansion of may hold any descendant of its values.
        pytest.param(
            ["CRSTypeOfFinance400", "CRSTypeOfFinance421"],
            [],
            ProblemCode.OPERATION_UNSUPPORTED.value,
            id="codes-unknown",
        ),
    ],
)
def test_a_sum_on_one_provider_scope_from_all_donors_counts_no_flow_twice_over_apart_slots(
    decision: DecisionFactory, accept: bool, loans: list[str], added: list[str], code: str
) -> None:
    """Grants plus loans to Kenya from all donors, both counted with imputed
    multilateral flows. Donors left open may include multilateral organisations,
    whose outflows the other input's imputed share can count, but no flow is both a
    grant and a loan, so the sum counts nothing twice and is served. Where the two
    bindings share a code, or one's codes are unknown, a flow can lie in both and
    the sum is refused."""
    ctx = _FakeToolContext()
    grants, other = (
        _to_recipients(
            _from_donors(provider_block(spec_id, narrow=False, active="provider:inclusive"), []),
            ["country/KEN"],
        )
        for spec_id in (HEALTH, TOTAL)
    )
    _show_blocks(
        ctx,
        _by_instrument(grants, ["CRSTypeOfFinance100"], added=["CRSTypeOfFinance110"]),
        _by_instrument(other, loans, added=added),
    )

    result = _sum_of_health(_tools_by_name()["propose_plan"].func, ctx, decision(), accept=accept)

    assert result.get("outcome") == code or result.get("code") == code, result
    if code == Outcome.OK.value:
        assert _pins(ctx) == [("health", HEALTH), ("total", TOTAL)]
    else:
        assert result["detail"]["basis"]["step_id"] == "sum"
        assert load_plan(ctx.state) is None


@pytest.mark.parametrize(
    ("left", "right", "code"),
    [
        pytest.param(
            (["country/FRA"], "explicit"),
            (["country/DEU"], "explicit"),
            Outcome.OK.value,
            id="disjoint-donors",
        ),
        pytest.param(
            (["country/FRA"], "explicit"),
            (["country/DEU"], "carried"),
            Outcome.OK.value,
            id="disjoint-donors-one-carried",
        ),
        pytest.param(
            (["country/FRA"], "explicit"),
            (["country/FRA", "country/DEU"], "explicit"),
            ProblemCode.OPERATION_UNSUPPORTED.value,
            id="a-shared-donor",
        ),
        pytest.param(
            (["country/FRA"], "explicit"),
            ([], "carried"),
            ProblemCode.OPERATION_UNSUPPORTED.value,
            id="all-providers-carried",
        ),
    ],
)
def test_a_sum_over_different_stated_provider_scopes_is_refused_only_where_donors_overlap(
    decision: DecisionFactory,
    left: tuple[list[str], str],
    right: tuple[list[str], str],
    code: str,
) -> None:
    """France's bilateral aid plus Germany's with imputed multilateral flows counts no
    flow twice, so the sum is served on the scopes as stated. A donor in both, or an
    all-providers total beside a named donor, would count the bilateral flows twice. A
    scope the user chose in an earlier turn counts as stated."""
    ctx = _FakeToolContext()
    _show_blocks(
        ctx,
        _from_donors(
            provider_block(HEALTH, narrow=False, active="provider:bilateral", provenance=left[1]),
            left[0],
        ),
        _from_donors(
            provider_block(TOTAL, narrow=False, active="provider:inclusive", provenance=right[1]),
            right[0],
        ),
    )

    result = _sum_of_health(_tools_by_name()["propose_plan"].func, ctx, decision())

    assert result.get("outcome") == code or result.get("code") == code, result


async def test_an_input_moving_on_basis_and_provider_scope_is_compiled_once_on_both(
    decision: DecisionFactory,
) -> None:
    """The US total counted in grant equivalents with imputed multilateral flows,
    read by two shares beside health aid counted gross and bilateral. Each total
    moves on both axes, so the total is compiled once through one resubmit
    posting both options, and the second requirement on the same pin and
    options takes the same compiled pin."""
    surface = ProviderCompiles()
    ctx = _FakeToolContext()
    _show_blocks(
        ctx,
        provider_block(HEALTH, narrow=True, active="provider:bilateral"),
        provider_block(
            TOTAL,
            narrow=False,
            active="provider:inclusive",
            basis="measure_basis:grant_equivalent",
        ),
    )

    def dataset(requirement_id: str, measure: str, pin: str) -> DatasetRequirement:
        return DatasetRequirement(
            id=requirement_id,
            decision_ids=["measure"],
            measure=measure,
            entities=[Entity(name="United States", role="donor")],
            period=Period(start=2015, end=2024),
            source_spec_id=pin,
        )

    def share(step_id: str, total_id: str) -> ShareOfRequirement:
        return ShareOfRequirement(
            id=step_id,
            decision_ids=["measure"],
            numerator=RequirementSide(input_id="health"),
            denominator=RequirementSide(input_id=total_id),
        )

    result = await _tools_reading(surface)["propose_plan"].func(
        question="What share of US aid, and of US ODA, goes to health?",
        answer_mode=AnswerMode.COMPUTED,
        decisions=drafts([decision()]),
        evidence_requirements=[
            dataset("health", "US aid to health", HEALTH),
            dataset("total", "US total net ODA", TOTAL),
            dataset("total_again", "US total ODA", TOTAL),
            share("share", "total"),
            share("share_again", "total_again"),
        ],
        tool_context=ctx,
    )

    assert result["outcome"] == Outcome.OK.value, result
    (call,) = surface.calls
    assert call["scope_entries"] == [
        {
            "axis": "measure_basis",
            "option_id": "measure_basis:gross",
            "provenance": "explicit",
            "origin": {"kind": "choice", "choice_id": f"{TOTAL}_base:measure_basis"},
        },
        {
            "axis": "provider",
            "option_id": "provider:bilateral",
            "provenance": "explicit",
            "origin": {"kind": "choice", "choice_id": f"{TOTAL}_base:provider"},
        },
    ]
    plan = load_plan(ctx.state)
    assert plan is not None
    assert [
        (r.id, r.source_spec_id)
        for r in plan.evidence_requirements
        if isinstance(r, DatasetRequirement)
    ] == [
        ("health", HEALTH),
        ("total", TOTAL_GROSS_BILATERAL),
        ("total_again", TOTAL_GROSS_BILATERAL),
    ]


async def test_a_total_compiled_on_the_bilateral_scope_is_pinned_with_the_planners_decision(
    decision: DecisionFactory,
) -> None:
    """The US total is compiled on bilateral flows alone through the resubmit a switch
    posts; the sum reads the compiled pin, and the provider decision on it is the
    planner's, saying why."""
    surface = ProviderCompiles()
    ctx = _FakeToolContext()
    _show_blocks(
        ctx,
        provider_block(HEALTH, narrow=True, active="provider:bilateral"),
        provider_block(TOTAL, narrow=False, active="provider:inclusive"),
    )

    result = await _sum_of_health(_tools_reading(surface)["propose_plan"].func, ctx, decision())

    assert result["outcome"] == Outcome.OK.value, result
    (call,) = surface.calls
    assert call["scope_entries"][-1] == {
        "axis": "provider",
        "option_id": "provider:bilateral",
        "provenance": "explicit",
        "origin": {"kind": "choice", "choice_id": f"{TOTAL}_base:provider"},
    }
    plan = load_plan(ctx.state)
    assert plan is not None
    assert plan.requirement("total").source_spec_id == TOTAL_BILATERAL  # type: ignore[union-attr]
    (aligned,) = [
        d
        for d in plan.decisions
        if d.scope is not None
        and d.scope.axis == "provider"
        and d.fixes is not None
        and d.fixes.source_spec_id == TOTAL_BILATERAL
    ]
    assert aligned.resolved_by is ResolvedBy.PLANNER_DEFAULT
    assert aligned.active_option_id == "provider:bilateral"


async def test_a_basis_search_read_in_the_planners_words_is_never_the_users(
    decision: DecisionFactory,
) -> None:
    """US health aid counted gross over a US total the planner pinned no dataset
    for. Search reads "net" in the planner's words for the total, an explicit
    entry the user never stated: the total is compiled on the health aid's gross,
    the narrower part's basis, and no later search carries net as the user's."""
    surface = BasisCompiles(
        searched={
            "status": "definite",
            "spec": basis_block(
                TOTAL, narrow=False, active="measure_basis:net", provenance="explicit"
            ),
        }
    )
    ctx = _FakeToolContext()
    _show_blocks(ctx, basis_block(HEALTH, narrow=True, active="measure_basis:gross"))

    result = await _share_of_health(
        _tools_reading(surface)["propose_plan"].func, ctx, decision(), total=None
    )

    assert result["outcome"] == Outcome.OK.value, result
    plan = load_plan(ctx.state)
    assert plan is not None
    assert [
        (r.id, r.source_spec_id)
        for r in plan.evidence_requirements
        if isinstance(r, DatasetRequirement)
    ] == [("health", HEALTH), ("total", TOTAL_GROSS)]
    (searched,) = searched_pins(ctx.state)
    assert searched.digest is not None
    store_search(ctx.state, searched.digest, asked=[()], held_open=[()])
    assert _search_context(ctx.state).carried_entries == []


async def test_a_share_whose_unpinned_input_search_does_not_settle_is_refused(
    decision: DecisionFactory,
) -> None:
    """Search answers the unpinned US total as ambiguous, so one basis cannot be
    guaranteed for the share: the plan is refused as unsupported, naming the
    input, and nothing is recorded."""
    surface = BasisCompiles(searched={"status": "ambiguous"})
    ctx = _FakeToolContext()
    _show_blocks(ctx, basis_block(HEALTH, narrow=True, active="measure_basis:gross"))

    refused = await _share_of_health(
        _tools_reading(surface)["propose_plan"].func, ctx, decision(), total=None
    )

    assert refused["code"] == ProblemCode.OPERATION_UNSUPPORTED.value, refused
    assert refused["detail"] == {
        "operation": UnsupportedOperation.UNSUPPORTED_ASK.value,
        "basis": {
            "spec_id": None,
            "option_ids": [],
            "requirement_id": "total",
            "step_id": "share",
        },
    }
    assert surface.tools == ["search"]
    assert load_plan(ctx.state) is None


@pytest.mark.parametrize(
    ("retry_mode", "code", "held"),
    [
        ("retry_after_delay", ProblemCode.SURFACE_UNAVAILABLE, ()),
        ("do_not_retry", ProblemCode.OPERATION_UNSUPPORTED, ("refused",)),
    ],
    ids=["retryable", "settled"],
)
async def test_a_refused_search_for_an_unpinned_input_is_held_only_when_it_settles(
    decision: DecisionFactory, retry_mode: str, code: ProblemCode, held: tuple[str, ...]
) -> None:
    """Search refuses the search for the unpinned US total. A refusal that can
    succeed later answers that the surface is unavailable and holds nothing, so
    the next proposal searches again; one that cannot is held as a search that
    settled on nothing, and the plan is refused as unsupported."""
    surface = BasisCompiles(
        search_refusal={"code": "upstream_unavailable", "retry_mode": retry_mode}
    )
    ctx = _FakeToolContext()
    _show_blocks(ctx, basis_block(HEALTH, narrow=True, active="measure_basis:gross"))

    refused = await _share_of_health(
        _tools_reading(surface)["propose_plan"].func, ctx, decision(), total=None
    )

    assert refused["code"] == code.value, refused
    assert tuple(pin.status for pin in searched_pins(ctx.state)) == held
    assert load_plan(ctx.state) is None


async def test_a_search_for_an_unpinned_input_takes_only_the_time_the_entry_has_left(
    decision: DecisionFactory,
) -> None:
    """The planner entry has a moment left and Search never answers the search
    for the unpinned US total: the search is cut off when the entry's time runs
    out, the refusal names why, and nothing is held."""
    ctx = _FakeToolContext()
    _show_blocks(ctx, basis_block(HEALTH, narrow=True, active="measure_basis:gross"))
    token = current_discovery_time_left.set(lambda: 0.05)
    try:
        async with asyncio.timeout(2.0):
            refused = await _share_of_health(
                _tools_reading(_SilentSurface())["propose_plan"].func, ctx, decision(), total=None
            )
    finally:
        current_discovery_time_left.reset(token)

    assert refused["code"] == ProblemCode.SURFACE_UNAVAILABLE.value, refused
    assert "the planner entry had left" in refused["message"], refused
    assert searched_pins(ctx.state) == ()


@pytest.mark.parametrize(
    ("two_axes", "option_ids"),
    [
        (False, ["measure_basis:gross"]),
        (True, ["measure_basis:gross", "provider:bilateral"]),
    ],
    ids=["basis", "basis-and-provider"],
)
def test_nested_calculations_put_every_input_they_reach_on_one_basis(
    decision: DecisionFactory, two_axes: bool, option_ids: list[str]
) -> None:
    """US health aid as a share of the sum of two US totals counted in grant
    equivalents, and with imputed multilateral flows where `two_axes`. The sum
    and the share read connected inputs, so all three take one option on each
    axis, the narrowest part's: each total is asked for once on every option it
    moves to, with no refusal for the totals the sum and the share both read."""
    other_total = f"spec_{'5' * 32}"

    def block(spec_id: str, *, narrow: bool, basis: str, provider: str) -> dict[str, Any]:
        if two_axes:
            return provider_block(spec_id, narrow=narrow, active=provider, basis=basis)
        return basis_block(spec_id, narrow=narrow, active=basis)

    ctx = _FakeToolContext()
    _show_blocks(
        ctx,
        block(HEALTH, narrow=True, basis="measure_basis:gross", provider="provider:bilateral"),
        block(
            TOTAL,
            narrow=False,
            basis="measure_basis:grant_equivalent",
            provider="provider:inclusive",
        ),
        block(
            other_total,
            narrow=False,
            basis="measure_basis:grant_equivalent",
            provider="provider:inclusive",
        ),
    )

    def dataset(requirement_id: str, measure: str, pin: str) -> DatasetRequirement:
        return DatasetRequirement(
            id=requirement_id,
            decision_ids=["measure"],
            measure=measure,
            entities=[Entity(name="United States", role="donor")],
            period=Period(start=2015, end=2024),
            source_spec_id=pin,
        )

    refused = _tools_by_name()["propose_plan"].func(
        question="What share of US aid goes to health?",
        answer_mode=AnswerMode.COMPUTED,
        decisions=drafts([decision()]),
        evidence_requirements=[
            dataset("health", "US aid to health", HEALTH),
            dataset("total", "US total net ODA", TOTAL),
            dataset("other", "US other official flows", other_total),
            SumOfRequirement(
                id="both",
                decision_ids=["measure"],
                left=RequirementSide(input_id="total"),
                right=RequirementSide(input_id="other"),
            ),
            ShareOfRequirement(
                id="share",
                decision_ids=["measure"],
                numerator=RequirementSide(input_id="health"),
                denominator=RequirementSide(input_id="both"),
            ),
        ],
        tool_context=ctx,
    )

    assert refused["code"] == ProblemCode.BASES_UNALIGNED.value, refused
    assert sorted((r["spec_id"], r["option_ids"]) for r in refused["detail"]["realign"]) == [
        (TOTAL, option_ids),
        (other_total, option_ids),
    ]


_MEASURE_FIXED = DecisionFixes(measure="external debt service paid")


def _measure_fixes(plan: Plan) -> DecisionFixes | None:
    """What the `measure` decision of `plan` fixes."""
    return next(d for d in plan.decisions if d.slot == "measure").fixes


def test_a_decision_over_a_pins_flow_count_is_refused(
    decision: DecisionFactory, requirements: RequirementPair
) -> None:
    """A decision listing `measure` that a pin of a measure-offering source
    cites restates the flow `propose_plan` records for the pin, so the call
    is refused and the detail names the pin's counts to re-pin to."""
    tools = _tools_by_name()
    ctx = _FakeToolContext()
    _show_block(ctx, _FRANCE_KENYA)
    dataset, derived = requirements

    refused = tools["propose_plan"].func(
        question="q",
        answer_mode=AnswerMode.COMPUTED,
        decisions=drafts([decision(fixes=_MEASURE_FIXED)]),
        evidence_requirements=[dataset.model_copy(update={"source_spec_id": _GROSS}), derived],
        tool_context=ctx,
    )

    assert refused["code"] == ProblemCode.MEASURE_FIXED_BY_PIN.value, refused
    assert refused["detail"] == {
        "slot": "measure",
        "requirement_id": dataset.id,
        "spec_id": _GROSS,
        "measure_decision": search_pin_slot(_GROSS),
        "chosen": "gross disbursements",
        "counts": {"gross disbursements": _GROSS, "net disbursements": _NET},
    }
    assert load_plan(ctx.state) is None


@pytest.mark.parametrize(
    ("fixes", "pin", "shown"),
    [
        (_MEASURE_FIXED, None, True),
        (DecisionFixes(period=Period(start=2015, end=2024)), _GROSS, True),
        (_MEASURE_FIXED, _GROSS, False),
    ],
    ids=["unpinned", "fixes-period", "pin-offers-no-measure"],
)
def test_a_decision_on_a_measure_no_pin_decides_is_recorded(
    decision: DecisionFactory,
    requirements: RequirementPair,
    fixes: DecisionFixes,
    pin: str | None,
    shown: bool,
) -> None:
    """A decision fixing the measure of an unpinned requirement, one fixing
    another field of a pinned requirement, and one on a pin whose source
    offers no measure each record as written."""
    tools = _tools_by_name()
    ctx = _FakeToolContext()
    if shown:
        _show_block(ctx, _FRANCE_KENYA)
    else:
        _show(ctx, _GROSS)
    dataset, derived = requirements

    result = tools["propose_plan"].func(
        question="q",
        answer_mode=AnswerMode.COMPUTED,
        decisions=drafts([decision(fixes=fixes)]),
        evidence_requirements=[dataset.model_copy(update={"source_spec_id": pin}), derived],
        tool_context=ctx,
    )

    assert result["outcome"] == Outcome.OK.value, result
    plan = load_plan(ctx.state)
    assert plan is not None
    assert _measure_fixes(plan) == fixes


@pytest.mark.parametrize(
    "route",
    ["code_slot_restated", "pin_settled_after", "pin_settled_after_user_silent"],
)
def test_an_amendment_restating_a_measure_beside_no_second_flow_decision_is_recorded(
    decision: DecisionFactory, requirements: RequirementPair, route: str
) -> None:
    """An amendment whose measure decision sits beside no other flow
    decision of its pin records: one restating `propose_plan`'s own
    `source_<pin>` decision with a `measure` fix, and one restating a
    measure decision on a requirement pinned before its source offered a
    measure, which `_measure_decisions` never revisits. With a silent user
    the restated decision is theirs, and `user_decision_contradicted` would
    refuse it without its `measure` fix, so it has to land as it stands."""
    tools = _tools_by_name()
    ctx = _FakeToolContext()
    dataset, derived = requirements
    pinned = dataset.model_copy(update={"source_spec_id": _GROSS})
    code_slot = route == "code_slot_restated"
    user = route == "pin_settled_after_user_silent"
    if code_slot:
        _show_block(ctx, _FRANCE_KENYA)
        first_decision = decision()
    else:
        _show(ctx, _GROSS, settled=False)
        first_decision = decision(
            fixes=_MEASURE_FIXED,
            resolved_by=ResolvedBy.USER if user else ResolvedBy.PLANNER_DEFAULT,
        )
    first = tools["propose_plan"].func(
        question="q",
        answer_mode=AnswerMode.COMPUTED,
        decisions=drafts([first_decision]),
        evidence_requirements=[pinned, derived],
        tool_context=ctx,
    )
    assert first["outcome"] == Outcome.OK.value, first
    if user:
        _recorded(ctx, first)
    recorded = load_plan(ctx.state)
    assert recorded is not None
    if code_slot:
        restated = next(d for d in recorded.decisions if d.slot == search_pin_slot(_GROSS))
    else:
        _show_block(ctx, _FRANCE_KENYA)
        restated = first_decision

    amended = tools["propose_plan"].func(
        decisions=drafts([restated.model_copy(update={"fixes": _MEASURE_FIXED})]),
        amends_version=recorded.version,
        tool_context=ctx,
    )

    assert amended["outcome"] == Outcome.OK.value, amended
    plan = load_plan(ctx.state)
    assert plan is not None
    assert [d.slot for d in plan.decisions] == [
        "measure",
        *([search_pin_slot(_GROSS)] if code_slot else []),
    ]


def test_a_stored_decision_over_a_pins_flow_count_loads_and_stands_through_an_amendment(
    decision: DecisionFactory, requirements: RequirementPair
) -> None:
    """A plan recorded before the refusal, holding a decision that fixes a
    pinned measure, still loads, and an amendment that leaves the decision
    as it stands is recorded with it."""
    tools = _tools_by_name()
    ctx = _FakeToolContext()
    _show_block(ctx, _FRANCE_KENYA)
    dataset, derived = requirements
    pinned = dataset.model_copy(update={"source_spec_id": _GROSS})
    first = tools["propose_plan"].func(
        question="q",
        answer_mode=AnswerMode.COMPUTED,
        decisions=drafts([decision()]),
        evidence_requirements=[pinned, derived],
        tool_context=ctx,
    )
    assert first["outcome"] == Outcome.OK.value, first
    recorded = load_plan(ctx.state)
    assert recorded is not None
    stored = Plan.model_validate(
        recorded.model_copy(
            update={
                "decisions": tuple(
                    d.model_copy(update={"fixes": _MEASURE_FIXED}) if d.slot == "measure" else d
                    for d in recorded.decisions
                )
            }
        ).model_dump()
    )
    save_plan(ctx.state, stored)
    loaded = load_plan(ctx.state)
    assert loaded is not None
    assert _measure_fixes(loaded) == _MEASURE_FIXED

    amended = tools["propose_plan"].func(
        evidence_requirements=[derived.model_copy(update={"reduction": Reduction.MEAN})],
        amends_version=loaded.version,
        tool_context=ctx,
    )

    assert amended["outcome"] == Outcome.OK.value, amended
    plan = load_plan(ctx.state)
    assert plan is not None
    assert _measure_fixes(plan) == _MEASURE_FIXED


@pytest.mark.parametrize(
    "offered",
    [(POPULATION_SPEC,), (POPULATION_SPEC, SPEC)],
    ids=["one-offered", "several-offered"],
)
def test_a_re_pin_to_a_source_found_in_place_of_a_stale_pin_needs_no_decision(
    decision: DecisionFactory, requirements: RequirementPair, offered: tuple[str, ...]
) -> None:
    """A re-pin answering a `pinned_source_not_found` referral records as
    written, whether the referral offered one source or several."""
    tools = _tools_by_name()
    ctx = _FakeToolContext()
    _show(ctx, SPEC)
    dataset, derived = requirements
    _propose(tools, ctx, decision(), [dataset.model_copy(update={"source_spec_id": SPEC}), derived])
    first = load_plan(ctx.state)
    assert first is not None
    save_referral(
        ctx.state,
        Referral(
            kind=ReferralKind.PINNED_SOURCE_NOT_FOUND,
            reason="r",
            requirement_id=dataset.id,
            candidates=tuple(
                SelectedSource(spec_id=spec_id, sentence="found instead", has_data=True)
                for spec_id in offered
            ),
        ),
    )
    re_pinned = dataset.model_copy(update={"source_spec_id": POPULATION_SPEC})

    result = tools["propose_plan"].func(
        evidence_requirements=[re_pinned, derived],
        amends_version=first.version,
        tool_context=ctx,
    )

    assert result["outcome"] == Outcome.OK.value, result
    amended = load_plan(ctx.state)
    assert amended is not None
    assert amended.decisions == (decision(),)
    assert amended.requirement(dataset.id) == re_pinned


@pytest.mark.parametrize(
    ("events", "resolved_by"),
    [([], ResolvedBy.PLANNER_DEFAULT), (_measure_ask_events(_GROSS), ResolvedBy.USER)],
    ids=["planner-re-pins", "user-picks"],
)
def test_a_re_pin_to_a_source_a_referral_offered_states_how_the_flow_is_counted(
    decision: DecisionFactory,
    requirements: RequirementPair,
    events: list[Any],
    resolved_by: ResolvedBy,
) -> None:
    """A source a referral offered in place of a stale pin is one no search
    the planner was shown settled, and its block elects a measure among
    several, so a re-pin of it gets the measure decision all the same."""
    tools = _tools_by_name()
    ctx = _FakeToolContext()
    _show(ctx, SPEC)
    dataset, derived = requirements
    _propose(tools, ctx, decision(), [dataset.model_copy(update={"source_spec_id": SPEC}), derived])
    first = load_plan(ctx.state)
    assert first is not None
    save_referral(
        ctx.state,
        Referral(
            kind=ReferralKind.PINNED_SOURCE_NOT_FOUND,
            reason="r",
            requirement_id=dataset.id,
            candidates=(
                source_from_spec_block(_FRANCE_KENYA),
                SelectedSource(spec_id=POPULATION_SPEC, sentence="found instead", has_data=True),
            ),
        ),
    )
    ctx.session.events.extend(events)

    result = tools["propose_plan"].func(
        evidence_requirements=[dataset.model_copy(update={"source_spec_id": _GROSS}), derived],
        amends_version=first.version,
        tool_context=ctx,
    )

    assert result["outcome"] == Outcome.OK.value, result
    amended = load_plan(ctx.state)
    assert amended is not None
    (measure,) = [d for d in amended.decisions if d.fixes and d.fixes.source_spec_id == _GROSS]
    assert measure.options == ("gross disbursements", "net disbursements")
    assert (measure.chosen, measure.resolved_by) == ("gross disbursements", resolved_by)
    assert measure.slot in amended.requirement(dataset.id).decision_ids
    assert measure in shown_decisions(amended)


# --- a share's whole counted from its part's own dataset ---

# The requirement ids of the US 2023 plans (`_us_2023`), which no reader text names.
_PART, _WHOLE, _SHARE, _COMPARE, _TOTAL, _SUM = (
    "health_crs",
    "total_oda",
    "share_health",
    "compare_totals",
    "sum_total",
    "health_plus_total",
)


def _us_block(name: str) -> dict[str, Any]:
    """`US_2023`'s block `name`, or, as "dac1_contributions", the DAC1 total with the
    provider choice dfo-apis gives it (`dac1_with_contributions`), as
    "dac1_multilateral", that total read as DAC1 multilateral ODA
    (`dac1_multilateral`), as "no_bilateral", that total with no bilateral option served
    (`total_without_bilateral`), as "malaria", the health part narrowed to CRS
    12262, malaria control (`_malaria_block`), or, as "untitled_<name>", block
    `<name>` with no dataset named (`facets.provenance` unset)."""
    if name.startswith("untitled_"):
        block = json.loads(json.dumps(_us_block(name.removeprefix("untitled_"))))
        del block["resolved"]["facets"]["provenance"]
        return block
    if name == "dac1_contributions":
        return dac1_with_contributions()
    if name == "dac1_multilateral":
        return dac1_multilateral()
    if name == "no_bilateral":
        return total_without_bilateral()
    if name == "malaria":
        return _malaria_block()
    return us_2023_block(name)


def _malaria_block() -> dict[str, Any]:
    """`US_2023`'s health part bound to CRS 12262, malaria control, under a spec_id of its own."""
    block = json.loads(json.dumps(us_2023_block("health")))
    block["spec_id"] = f"spec_{'7' * 32}"
    for slot in block["resolved"]["scope_choices"]["base"]["identity"]["slots"]:
        if slot["property"] == "crsPurpose":
            slot["binding"] = {"state": "value", "value": "dcid:CRS/12262"}
    return block


# What Search sends as `resolved.concept_label` and `resolved.published_tables` for
# each `US_2023` part these tests read, which the recording predates: the CRS health
# part, its composite inclusive form, and the DAC1 total.
_CRS = "OECD Creditor Reporting System (CRS)"
_SEARCH_LABELS = {
    "health": ("health", _CRS),
    "health_inclusive": ("health", f"{_CRS} and ONE Imputed Multilateral Sectors"),
    "whole": ("official development assistance", _CRS),
    "dac1_total_gross": ("official development assistance", "OECD DAC1"),
    "dac1_contributions": ("official development assistance", "OECD DAC1"),
    "dac1_multilateral": ("multilateral official development assistance", "OECD DAC1"),
    "no_bilateral": ("official development assistance", "OECD DAC1"),
}


def _search_labelled(name: str) -> dict[str, Any]:
    """`_us_block(name)` carrying the concept and tables Search sends for it
    (`_SEARCH_LABELS`); an "untitled_" block names no tables, as it names no dataset."""
    block = json.loads(json.dumps(_us_block(name)))
    concept, tables = _SEARCH_LABELS[name.removeprefix("untitled_")]
    block["resolved"]["concept_label"] = concept
    if not name.startswith("untitled_"):
        block["resolved"]["published_tables"] = tables
    return block


def _stated_provider(block: dict[str, Any]) -> dict[str, Any]:
    """`block` with its provider choice settled by the user's words, as an explicit entry."""
    scope = block["resolved"]["scope_choices"]
    choice = next(c for c in scope["choices"] if c["axis"] == "provider")
    choice["provenance"] = "explicit"
    scope["entries"] = [
        *scope["entries"],
        {
            "axis": "provider",
            "option_id": choice["active"],
            "provenance": "explicit",
            "origin": {"kind": "text", "text": "aid", "start": 0, "end": 3},
        },
    ]
    return block


def _us_2023(
    propose: Callable[..., Any],
    ctx: _FakeToolContext,
    decision: Decision,
    *,
    part: str,
    whole: str | None,
    declared: bool = False,
    share: bool = True,
    source: str | None = None,
    quotient: bool = False,
    summed_whole: bool = False,
    accept: bool = False,
    denominator: Decision | None = None,
    extra: Sequence[EvidenceRequirement] = (),
    own: Sequence[Decision] = (),
) -> Any:
    """Propose the US 2023 part `part` as a share of `whole`, or compared with it,
    each pinned to its block (`_us_block`), the whole to none when `None`; with
    `declared`, the whole declares itself the part over every sector (`whole_of`);
    the whole's source written as `source`; with `quotient`, the part summed by year
    is divided by the whole as a `quotient_of`, as the planner wrote the share in a
    recorded Studio run; with `summed_whole`, the whole too is summed by year and
    the part summed by year is a `share_of` it, or a `quotient_of` it with `quotient`; with `accept`, the share accepts a provider difference;
    with `denominator`, a decision of the planner's the whole alone cites; `extra`
    is proposed after the rest; `own` are further decisions the part cites."""

    def dataset(
        requirement_id: str,
        measure: str,
        name: str | None,
        named: str | None = None,
        cited: tuple[str, ...] = (),
        whole_of: WholeOf | None = None,
    ) -> DatasetRequirement:
        return DatasetRequirement(
            id=requirement_id,
            decision_ids=["measure", *cited],
            measure=measure,
            entities=[Entity(name="United States", role="donor")],
            period=Period(start=2023, end=2023),
            source=named,
            source_spec_id=None if name is None else _us_block(name)["spec_id"],
            whole_of=whole_of,
        )

    summed: list[EvidenceRequirement] = []
    if quotient or summed_whole:
        denominator_id = "whole_by_year" if summed_whole else _WHOLE
        summed = [
            AggregateRequirement(
                id="health_by_year",
                input_id=_PART,
                reduction=Reduction.SUM,
                group_by=(GroupBy.PERIOD,),
            ),
            *(
                [
                    AggregateRequirement(
                        id="whole_by_year",
                        input_id=_WHOLE,
                        reduction=Reduction.SUM,
                        group_by=(GroupBy.PERIOD,),
                    )
                ]
                if summed_whole
                else []
            ),
            (QuotientRequirement if quotient else ShareOfRequirement)(
                id=_SHARE,
                numerator=RequirementSide(input_id="health_by_year"),
                denominator=RequirementSide(input_id=denominator_id),
                accept_provider_difference=accept,
            ),
        ]
    step: EvidenceRequirement = (
        ShareOfRequirement(
            id=_SHARE,
            decision_ids=["measure"],
            numerator=RequirementSide(input_id=_PART),
            denominator=RequirementSide(input_id=_WHOLE),
            accept_provider_difference=accept,
        )
        if share
        else ComparisonRequirement(
            id=_COMPARE,
            decision_ids=["measure"],
            minuend=RequirementSide(input_id=_WHOLE),
            subtrahend=RequirementSide(input_id=_PART),
            accept_provider_difference=accept,
        )
    )
    return propose(
        question="What share of US aid went to health in 2023?",
        answer_mode=AnswerMode.COMPUTED,
        decisions=drafts([decision, *([denominator] if denominator else []), *own]),
        evidence_requirements=[
            dataset(_PART, "US aid to health", part, cited=tuple(d.slot for d in own)),
            dataset(
                _WHOLE,
                "US total ODA",
                whole,
                source,
                (denominator.slot,) if denominator else (),
                WholeOf(part=_PART, axis="sector") if declared else None,
            ),
            *(summed or [step]),
            *extra,
        ],
        tool_context=ctx,
    )


def _pins(ctx: _FakeToolContext) -> list[tuple[str, str | None]]:
    """Each dataset requirement of the plan `ctx` holds, with its pin."""
    plan = load_plan(ctx.state)
    assert plan is not None
    return [
        (r.id, r.source_spec_id)
        for r in plan.evidence_requirements
        if isinstance(r, DatasetRequirement)
    ]


@pytest.mark.parametrize(
    ("part", "whole", "compiled"),
    [
        pytest.param("health", "dac1_total_gross", "whole", id="bilateral-over-dac1"),
        pytest.param(
            "health", "dac1_contributions", "whole", id="bilateral-over-dac1-with-provider"
        ),
        pytest.param("health", None, "whole", id="bilateral-over-an-unpinned-total"),
        pytest.param(
            "health_inclusive", "dac1_total_gross", "whole_inclusive", id="inclusive-over-dac1"
        ),
        pytest.param(
            "health_inclusive", "whole", "whole_inclusive", id="inclusive-over-crs-bilateral"
        ),
    ],
)
async def test_a_declared_whole_is_counted_from_its_parts_own_dataset(
    decision: DecisionFactory, part: str, whole: str | None, compiled: str
) -> None:
    """ "What share of US aid went to health in 2023?" with the total declared the
    health part over every sector, whatever the planner pinned it to: DAC1, which
    also counts core contributions to multilaterals, the CRS bilateral total under
    an inclusive part, or nothing. The whole is compiled from the part's base
    identity with its sector slot unbound and every other choice held as an
    explicit entry, with no search and no model call, and the share divides by it.
    The whole takes its part's scope: each held choice carries the part's
    provenance and the planner's axes are the part's, so a later search carries
    from the whole what it carries from the part. The whole states no scope row of
    its own, since each would restate the part's; a decision of the planner's
    states it, cited by the whole. The whole's source keeps the provenance its
    part's source records."""
    surface = WholeCompiles()
    ctx = _FakeToolContext()
    _show_blocks(ctx, us_2023_block(part), *([_us_block(whole)] if whole else []))
    shown = pin_sources(ctx.state)
    part_pin = us_2023_block(part)["spec_id"]
    provenance = {"provenance_ref": "provenance/OECD_CRS", "provenance_account": "OECD CRS"}
    save_pin_sources(ctx.state, {**shown, part_pin: shown[part_pin].model_copy(update=provenance)})

    result = await _us_2023(
        _tools_reading(surface)["propose_plan"].func,
        ctx,
        decision(),
        part=part,
        whole=whole,
        declared=True,
    )

    assert result["outcome"] == Outcome.OK.value, result
    (call,) = surface.calls
    scope = us_2023_block(part)["resolved"]["scope_choices"]
    identity = json.loads(json.dumps(scope["base"]["identity"]))
    for slot in identity["slots"]:
        if slot["property"] == "crsPurpose":
            slot["binding"] = {"state": "unbound"}
    answered = us_2023_block(compiled)["resolved"]["scope_choices"]["base"]
    assert (call["spec_id"], call["identity"]) == (answered["spec_id"], identity)
    assert call["scope_entries"] == [
        {
            "axis": choice["axis"],
            "option_id": choice["active"],
            "provenance": "explicit",
            "origin": {"kind": "choice", "choice_id": choice["id"]},
        }
        for choice in scope["choices"]
        if choice["axis"] != "sector"
    ]
    assert _pins(ctx) == [
        (_PART, us_2023_block(part)["spec_id"]),
        (_WHOLE, us_2023_block(compiled)["spec_id"]),
    ]
    assert searched_pins(ctx.state) == ()
    held_source = pin_sources(ctx.state)[us_2023_block(compiled)["spec_id"]]
    assert (held_source.provenance_ref, held_source.provenance_account) == (
        provenance["provenance_ref"],
        provenance["provenance_account"],
    )
    held = held_source.block
    assert held is not None
    assert PLANNER_AXES_BLOCK_KEY not in held
    assert {
        choice["axis"]: choice["provenance"]
        for choice in held["resolved"]["scope_choices"]["choices"]
    } == {
        choice["axis"]: choice["provenance"]
        for choice in scope["choices"]
        if choice["axis"] != "sector"
    }
    plan = load_plan(ctx.state)
    assert plan is not None
    assert [
        d.slot
        for d in plan.decisions
        if d.scope is not None
        and d.fixes is not None
        and d.fixes.source_spec_id == us_2023_block(compiled)["spec_id"]
    ] == []
    (stated,) = [d for d in plan.decisions if d.caveat == "derived_whole"]
    title = us_2023_block(part)["resolved"]["facets"]["provenance"]["label"]
    assert stated.chosen == f"the {title} total over every sector"
    whole_requirement = plan.requirement(_WHOLE)
    assert isinstance(whole_requirement, DatasetRequirement)
    assert stated.slot in whole_requirement.decision_ids


@pytest.mark.parametrize(
    ("quotient", "divides"),
    [(True, "quotient"), (False, "share")],
    ids=["quotient", "share"],
)
async def test_a_wholes_caveat_names_the_calculation_that_reads_it_through_a_sum(
    decision: DecisionFactory, quotient: bool, divides: str
) -> None:
    """The part summed by year over the declared whole summed by year, as a
    `quotient_of` or a `share_of`: the caveat on the whole says which calculation
    divides by it, not always a share."""
    ctx = _FakeToolContext()
    _show_blocks(ctx, us_2023_block("health"), us_2023_block("dac1_total_gross"))

    result = await _us_2023(
        _tools_reading(WholeCompiles())["propose_plan"].func,
        ctx,
        decision(),
        part="health",
        whole="dac1_total_gross",
        declared=True,
        quotient=quotient,
        summed_whole=True,
    )

    assert result["outcome"] == Outcome.OK.value, result
    plan = load_plan(ctx.state)
    assert plan is not None
    (stated,) = [d for d in plan.decisions if d.caveat == "derived_whole"]
    assert stated.description == f"Which total the {divides} divides by."


@pytest.mark.parametrize(
    ("declared_after", "whole"),
    [((), True), ((False,), False), ((False, True), True)],
    ids=["declared", "undeclared-later", "declared-again"],
)
async def test_a_declared_whole_offers_no_row_on_the_axis_it_leaves_open(
    decision: DecisionFactory, declared_after: tuple[bool, ...], whole: bool
) -> None:
    """The declared whole over every sector comes back from Search with a sector
    choice of its own: the plan states no sector row on the whole, whose
    caveat says it counts every sector, while the part keeps its own. Amended
    to a plain requirement on the same pin, with no `whole_of`, it is no
    whole, and its sector row returns; declared a whole again, it cites that
    row no more."""
    ctx = _FakeToolContext()
    _show_blocks(ctx, us_2023_block("health"))
    propose = _tools_reading(WholeCompiles(opened=True))["propose_plan"].func

    result = await _us_2023(propose, ctx, decision(), part="health", whole=None, declared=True)
    whole_of = load_plan(ctx.state).requirement(_WHOLE).whole_of  # type: ignore[union-attr]
    for declared in declared_after:
        restated = load_plan(ctx.state).requirement(_WHOLE)  # type: ignore[union-attr]
        result = await propose(
            evidence_requirements=[
                restated.model_copy(update={"whole_of": whole_of if declared else None})
            ],
            amends_version=result["version"],
            tool_context=ctx,
        )

    assert result["outcome"] == Outcome.OK.value, result
    plan = load_plan(ctx.state)
    assert plan is not None
    sectors = {
        d.fixes.source_spec_id
        for d in plan.decisions
        if d.scope is not None and d.scope.axis == "sector" and d.fixes is not None
    }
    whole_pin = dict(_pins(ctx))[_WHOLE]
    assert sectors == {us_2023_block("health")["spec_id"], *([] if whole else [whole_pin])}
    assert any(d.caveat == "derived_whole" for d in plan.decisions) is whole


async def test_a_plain_requirement_on_a_declared_wholes_pin_keeps_its_rows(
    decision: DecisionFactory,
) -> None:
    """A share of US health aid with the total declared a whole over every sector,
    then an amendment adding "US total aid" as a plain requirement the planner pins
    to the whole's compiled dataset. The plain requirement is no whole, so it states
    and cites the sector row Search returned on that dataset, while the declared
    whole still cites none."""
    ctx = _FakeToolContext()
    _show_blocks(ctx, us_2023_block("health"))
    propose = _tools_reading(WholeCompiles(opened=True))["propose_plan"].func
    result = await _us_2023(propose, ctx, decision(), part="health", whole=None, declared=True)
    assert result["outcome"] == Outcome.OK.value, result
    pin = dict(_pins(ctx))[_WHOLE]

    result = await propose(
        evidence_requirements=[
            DatasetRequirement(
                id="us_total",
                decision_ids=["measure"],
                measure="US total aid",
                entities=[Entity(name="United States", role="donor")],
                period=Period(start=2023, end=2023),
                source_spec_id=pin,
            ),
            AggregateRequirement(
                id="total_by_year",
                input_id="us_total",
                reduction=Reduction.SUM,
                group_by=(GroupBy.PERIOD,),
            ),
        ],
        amends_version=result["version"],
        tool_context=ctx,
    )

    assert result["outcome"] == Outcome.OK.value, result
    plan = load_plan(ctx.state)
    assert plan is not None
    rows = {
        d.slot
        for d in plan.decisions
        if d.scope is not None and d.fixes is not None and d.fixes.source_spec_id == pin
    }
    assert rows
    assert rows <= set(plan.requirement("us_total").decision_ids)
    assert rows.isdisjoint(plan.requirement(_WHOLE).decision_ids)


@pytest.mark.parametrize(
    ("title", "titled"),
    [("OECD CRS", True), ("OECD CRS " + "t" * 700, False)],
    ids=["titled", "untitled"],
)
def test_a_declared_whole_is_stated_whatever_its_parts_dataset_title(
    title: str, titled: bool
) -> None:
    """The whole of US health aid over every sector, its part's dataset titled
    `title`: the whole is stated naming the title while that fits the limits on
    its choice and rationale, and without it past them, and is never dropped."""
    block = json.loads(json.dumps(us_2023_block("health")))
    block["resolved"]["facets"]["provenance"]["label"] = title
    whole = AlignedPin(
        origin=block["spec_id"],
        option_ids=(),
        spec_id="spec_" + "9" * 32,
        digest={},
        whole_axis="sector",
    )

    (stated,) = submission_service._whole_decisions(
        {_WHOLE: whole}, [], shown={block["spec_id"]: PinSource(block=block, settled=True)}
    )[_WHOLE]

    assert stated.caveat == "derived_whole"
    assert (title in stated.chosen, title in stated.rationale) == (titled, titled)
    assert "total over every sector" in stated.chosen


async def test_two_wholes_compiled_to_one_pin_each_state_their_own_part(
    decision: DecisionFactory,
) -> None:
    """US health aid and US aid to CRS 120 alone, each a share of a declared whole over
    every sector. Both wholes compile to the one CRS total, yet each whole states its
    own decision, cited by it alone, naming the part it was declared over."""
    surface = WholeCompiles()
    ctx = _FakeToolContext()
    narrow = _malaria_block()
    sector = narrow["resolved"]["scope_choices"]["choices"][0]
    sector["active"] = "sector:120"
    _show_blocks(ctx, us_2023_block("health"), narrow)
    united_states = Entity(name="United States", role="donor")

    result = await _us_2023(
        _tools_reading(surface)["propose_plan"].func,
        ctx,
        decision(),
        part="health",
        whole=None,
        declared=True,
        extra=[
            DatasetRequirement(
                id="narrow_part",
                decision_ids=["measure"],
                measure="US aid to health, CRS 120",
                entities=[united_states],
                period=Period(start=2023, end=2023),
                source_spec_id=narrow["spec_id"],
            ),
            DatasetRequirement(
                id="narrow_whole",
                decision_ids=["measure"],
                measure="US total ODA",
                entities=[united_states],
                period=Period(start=2023, end=2023),
                whole_of=WholeOf(part="narrow_part", axis="sector"),
            ),
            ShareOfRequirement(
                id="narrow_share",
                decision_ids=["measure"],
                numerator=RequirementSide(input_id="narrow_part"),
                denominator=RequirementSide(input_id="narrow_whole"),
            ),
        ],
    )

    assert result["outcome"] == Outcome.OK.value, result
    compiled = us_2023_block("whole")["spec_id"]
    assert [pin for rid, pin in _pins(ctx) if rid in (_WHOLE, "narrow_whole")] == [
        compiled,
        compiled,
    ]
    plan = load_plan(ctx.state)
    assert plan is not None
    stated = {d.slot: d for d in plan.decisions if d.caveat == "derived_whole"}
    named = {}
    for whole_id in (_WHOLE, "narrow_whole"):
        (slot,) = [s for s in plan.requirement(whole_id).decision_ids if s in stated]
        named[whole_id] = stated[slot].rationale
    assert len(stated) == 2
    assert "health incl. population and reproductive health" in named[_WHOLE]
    assert "health, CRS 120 alone" in named["narrow_whole"]


def _uk_total(spec_id: str) -> dict[str, Any]:
    """The UK donor total for 2023 under `spec_id`: `US_2023`'s DAC1 total with the
    United Kingdom as donor."""
    block = json.loads(json.dumps(us_2023_block("dac1_total_gross")))
    block["spec_id"] = spec_id
    scope = block["resolved"]["scope_choices"]
    for identity in (scope["base"]["identity"], scope["compiled"]["identity"]):
        identity["entities"] = {"origin": ["country/GBR"]}
    scope["entries"] = []
    return block


_UK = f"spec_{'a' * 32}"


def _whole_reader(kind: str) -> EvidenceRequirement:
    """A calculation reading `_WHOLE` other than as the denominator of a share of its
    part: the whole minus the UK total, the UK total divided by the whole, or the
    whole summed by year. Built at call time, since `_WHOLE` is rebound further
    down this module."""
    if kind == "comparison":
        return ComparisonRequirement(
            id=_COMPARE,
            decision_ids=["measure"],
            minuend=RequirementSide(input_id=_WHOLE),
            subtrahend=RequirementSide(input_id="uk_total"),
        )
    if kind == "quotient-of-another-dataset":
        return QuotientRequirement(
            id="uk_over_us",
            decision_ids=["measure"],
            numerator=RequirementSide(input_id="uk_total"),
            denominator=RequirementSide(input_id=_WHOLE),
        )
    return AggregateRequirement(
        id=_TOTAL,
        decision_ids=["measure"],
        input_id=_WHOLE,
        reduction=Reduction.SUM,
        group_by=[GroupBy.PERIOD],
    )


@pytest.mark.parametrize("kind", ["comparison", "quotient-of-another-dataset", "aggregate"])
async def test_a_declared_whole_read_by_another_calculation_is_refused_for_a_plain_total(
    decision: DecisionFactory, kind: str
) -> None:
    """US health aid as a share of a declared US total, and that total read again by
    another calculation: the US total minus the UK total, the UK total divided by
    it, or it summed by year. A declared whole is read only as the denominator of a
    share or quotient of its part, so the plan is refused before anything is
    compiled, naming the other calculation to read a plain total of its own."""
    surface = WholeCompiles()
    ctx = _FakeToolContext()
    _show_blocks(ctx, us_2023_block("health"), _uk_total(_UK))
    reader = _whole_reader(kind)

    result = await _us_2023(
        _tools_reading(surface)["propose_plan"].func,
        ctx,
        decision(),
        part="health",
        whole=None,
        declared=True,
        extra=[
            DatasetRequirement(
                id="uk_total",
                decision_ids=["measure"],
                measure="UK total ODA",
                entities=[Entity(name="United Kingdom", role="donor")],
                period=Period(start=2023, end=2023),
                source_spec_id=_UK,
            ),
            reader,
        ],
    )

    assert result["code"] == ProblemCode.INVALID_PLAN.value, result
    assert repr(reader.id) in result["message"]
    assert surface.calls == []
    assert load_plan(ctx.state) is None


def _french(block: dict[str, Any]) -> dict[str, Any]:
    """A copy of `block`, a `US_2023` health part, with France as its donor: a part of
    its own (`PartScope.base`), under a spec_id of its own."""
    part = block["resolved"]["scope_choices"]["base"]["spec_id"]
    text = json.dumps(block)
    for us, fr in (
        (part, f"spec_{'7' * 32}"),
        (block["spec_id"], f"spec_{'8' * 32}"),
        ("country/USA", "country/FRA"),
        ("United States of America", "France"),
    ):
        text = text.replace(us, fr)
    return json.loads(text)


async def test_a_declared_whole_follows_its_part_when_another_calculation_realigns_it(
    decision: DecisionFactory,
) -> None:
    """US bilateral health aid as a share of a declared whole, and also compared with
    French health aid the user asked for with imputed multilateral flows. The
    comparison moves the part to the inclusive compile, so the whole is compiled from
    that final pin and the share divides an inclusive part by an inclusive total."""
    surface = WholeCompiles()
    ctx = _FakeToolContext()
    inclusive = us_2023_block("health_inclusive")
    stated = _stated_provider(_french(inclusive))
    _show_blocks(ctx, us_2023_block("health"), inclusive, stated)
    save_aligned_pin(
        ctx.state,
        AlignedPin(
            origin=us_2023_block("health")["spec_id"],
            option_ids=("provider:inclusive",),
            spec_id=inclusive["spec_id"],
            digest={},
        ),
    )

    result = await _us_2023(
        _tools_reading(surface)["propose_plan"].func,
        ctx,
        decision(),
        part="health",
        whole=None,
        declared=True,
        extra=[
            DatasetRequirement(
                id="health_stated",
                decision_ids=["measure"],
                measure="French aid to health with imputed multilateral flows",
                entities=[Entity(name="France", role="donor")],
                period=Period(start=2023, end=2023),
                source_spec_id=stated["spec_id"],
            ),
            ComparisonRequirement(
                id=_COMPARE,
                decision_ids=["measure"],
                minuend=RequirementSide(input_id="health_stated"),
                subtrahend=RequirementSide(input_id=_PART),
            ),
        ],
    )

    assert result["outcome"] == Outcome.OK.value, result
    (call,) = surface.calls
    assert [e["option_id"] for e in call["scope_entries"] if e["axis"] == "provider"] == [
        "provider:inclusive"
    ]
    assert dict(_pins(ctx))[_PART] == inclusive["spec_id"]
    assert dict(_pins(ctx))[_WHOLE] == us_2023_block("whole_inclusive")["spec_id"]


async def test_a_quotient_over_a_declared_whole_divides_by_the_parts_own_total(
    decision: DecisionFactory,
) -> None:
    """The plan a recorded Studio run wrote for "What share of US aid went to health
    in 2023?": CRS health aid summed by year, divided by the DAC1 total as a
    `quotient_of`, the total declared the part over every sector. It is compiled
    from the part's own dataset and the quotient divides by it."""
    surface = WholeCompiles()
    ctx = _FakeToolContext()
    _show_blocks(ctx, us_2023_block("health"), us_2023_block("dac1_total_gross"))

    result = await _us_2023(
        _tools_reading(surface)["propose_plan"].func,
        ctx,
        decision(),
        part="health",
        whole="dac1_total_gross",
        declared=True,
        quotient=True,
    )

    assert result["outcome"] == Outcome.OK.value, result
    assert len(surface.calls) == 1
    assert _pins(ctx) == [
        (_PART, us_2023_block("health")["spec_id"]),
        (_WHOLE, us_2023_block("whole")["spec_id"]),
    ]


async def test_wholes_declaring_one_part_and_axis_share_one_compile(
    decision: DecisionFactory,
) -> None:
    """The share's whole and the whole a second share of the same part divides by
    both declare the health part over every sector: the whole is compiled once and
    both read it, and replacing it changes no calculation, since each total declared
    what it is."""
    surface = WholeCompiles()
    ctx = _FakeToolContext()
    _show_blocks(ctx, us_2023_block("health"), us_2023_block("dac1_total_gross"))
    reported = DatasetRequirement(
        id="total_reported",
        decision_ids=["measure"],
        measure="US total ODA",
        entities=[Entity(name="United States", role="donor")],
        period=Period(start=2023, end=2023),
        whole_of=WholeOf(part=_PART, axis="sector"),
    )
    second_share = ShareOfRequirement(
        id=_TOTAL,
        decision_ids=["measure"],
        numerator=RequirementSide(input_id=_PART),
        denominator=RequirementSide(input_id=reported.id),
    )

    result = await _us_2023(
        _tools_reading(surface)["propose_plan"].func,
        ctx,
        decision(),
        part="health",
        whole="dac1_total_gross",
        declared=True,
        extra=[reported, second_share],
    )

    assert result["outcome"] == Outcome.OK.value, result
    assert len(surface.calls) == 1
    whole = us_2023_block("whole")["spec_id"]
    assert _pins(ctx) == [
        (_PART, us_2023_block("health")["spec_id"]),
        (_WHOLE, whole),
        ("total_reported", whole),
    ]


async def test_a_declared_whole_drops_the_source_the_planner_wrote(
    decision: DecisionFactory,
) -> None:
    """Inclusive US health aid over a declared whole the planner pinned to no dataset
    and wrote as from the "OECD Creditor Reporting System (CRS)". The whole is the
    part's own inclusive total over every sector, compiled with no search, and it
    carries no source: discovery checks a pin against a stated source, and the CRS
    words would search it back to the bilateral CRS total and refuse the plan."""
    surface = WholeCompiles()
    ctx = _FakeToolContext()
    _show_blocks(ctx, us_2023_block("health_inclusive"))

    result = await _us_2023(
        _tools_reading(surface)["propose_plan"].func,
        ctx,
        decision(),
        part="health_inclusive",
        whole=None,
        declared=True,
        source="OECD Creditor Reporting System (CRS)",
    )

    assert result["outcome"] == Outcome.OK.value, result
    plan = load_plan(ctx.state)
    assert plan is not None
    whole = plan.requirement(_WHOLE)
    assert isinstance(whole, DatasetRequirement)
    assert (whole.source_spec_id, whole.source) == (
        us_2023_block("whole_inclusive")["spec_id"],
        None,
    )


@pytest.mark.parametrize(
    ("part", "labelled", "chosen", "side"),
    [
        pytest.param(
            "health",
            False,
            "the OECD Creditor Reporting System (CRS) total over every sector",
            "the health incl. population and reproductive health (CRS 120 + 130) figures",
            id="named-dataset",
        ),
        pytest.param(
            "untitled_health",
            False,
            "the total over every sector",
            "the health incl. population and reproductive health (CRS 120 + 130) figures",
            id="no-dataset-name",
        ),
        pytest.param(
            "health",
            True,
            "the OECD Creditor Reporting System (CRS) total over every sector",
            "the health figures",
            id="search-labels",
        ),
    ],
)
@pytest.mark.parametrize("later_turn", [False, True], ids=["same-turn", "restated-next-turn"])
async def test_a_declared_whole_replaces_the_planners_decision_on_its_source(
    decision: DecisionFactory, later_turn: bool, part: str, labelled: bool, chosen: str, side: str
) -> None:
    """A recorded Studio run: the planner pinned the DAC1 total as the whole and
    cited a decision of its own, "Source for total aid denominator: dac1 total aid".
    Once the declared whole is compiled from the health figures' own dataset, that
    decision says something false of the plan, so it gives way to one stating the
    denominator in reader words: the CRS total, the same dataset as the health
    figures. The same holds when the whole is declared a turn later, restated with
    `whole_of` and the citation the earlier turn wrote. The health figures are
    named by the concept Search states for them, else by their sector option as
    Search labelled it."""
    surface = WholeCompiles()
    ctx = _FakeToolContext()
    _show_blocks(
        ctx,
        _search_labelled(part) if labelled else _us_block(part),
        us_2023_block("dac1_total_gross"),
    )
    source_choice = decision(
        slot="denominator_source",
        description="Source for total aid denominator",
        options=["dac1_total_aid", "crs_total_aid"],
        chosen="dac1_total_aid",
        rationale="The search results identified DAC1 total aid as the measure for total aid.",
        fixes=DecisionFixes(measure="US total ODA"),
    )
    propose = _tools_reading(surface)["propose_plan"].func

    result = await _us_2023(
        propose,
        ctx,
        decision(),
        part=part,
        whole="dac1_total_gross",
        declared=not later_turn,
        denominator=source_choice,
    )
    if later_turn:
        assert result["outcome"] == Outcome.OK.value, result
        first = load_plan(ctx.state)
        assert first is not None
        pinned = first.requirement(_WHOLE)
        assert "denominator_source" in pinned.decision_ids
        result = await propose(
            evidence_requirements=[
                pinned.model_copy(
                    update={
                        "source_spec_id": None,
                        "whole_of": WholeOf(part=_PART, axis="sector"),
                    }
                )
            ],
            amends_version=result["version"],
            tool_context=ctx,
        )

    assert result["outcome"] == Outcome.OK.value, result
    plan = load_plan(ctx.state)
    assert plan is not None
    whole = plan.requirement(_WHOLE)
    assert "denominator_source" not in {d.slot for d in plan.decisions}
    assert "denominator_source" not in whole.decision_ids
    (stated,) = [
        d for d in plan.decisions if d.slot in whole.decision_ids and d.caveat == "derived_whole"
    ]
    assert stated.chosen == chosen
    assert "  " not in stated.rationale
    assert "DAC1" not in stated.rationale and "dac1" not in stated.rationale
    assert side in stated.rationale


@pytest.mark.parametrize(
    "fixes",
    [
        pytest.param(DecisionFixes(period=Period(start=2023, end=2023)), id="fixes-the-period"),
        pytest.param(None, id="rationale-only"),
    ],
)
async def test_a_declared_whole_keeps_a_planners_decision_on_what_it_does_not_replace(
    decision: DecisionFactory, fixes: DecisionFixes | None
) -> None:
    """A decision of the planner's only the declared whole cites, on the year it
    reads: the compiled whole replaces the total's source, not its years, so the
    decision stays, cited by the whole."""
    ctx = _FakeToolContext()
    _show_blocks(ctx, us_2023_block("health"), us_2023_block("dac1_total_gross"))
    latest = decision(
        slot="latest_year",
        description="Which year to read",
        options=["latest complete year", "latest year with any data"],
        chosen="latest complete year",
        rationale="2023 is the latest year every donor has reported in full.",
        fixes=fixes,
    )

    result = await _us_2023(
        _tools_reading(WholeCompiles())["propose_plan"].func,
        ctx,
        decision(),
        part="health",
        whole="dac1_total_gross",
        declared=True,
        denominator=latest,
    )

    assert result["outcome"] == Outcome.OK.value, result
    plan = load_plan(ctx.state)
    assert plan is not None
    assert latest.slot in {d.slot for d in plan.decisions}
    assert latest.slot in plan.requirement(_WHOLE).decision_ids


async def test_a_restated_decision_a_carried_requirement_cites_stays_beside_a_whole(
    decision: DecisionFactory,
) -> None:
    """The health part cites a decision of the planner's on the denominator's source.
    An amendment restates that decision and the declared whole alone, now citing it
    too: the part, carried, still cites it, so it stays, and the whole keeps its
    citation."""
    tools = _tools_reading(WholeCompiles())
    ctx = _FakeToolContext()
    _show_blocks(ctx, us_2023_block("health"), us_2023_block("dac1_total_gross"))
    source_choice = decision(
        slot="denominator_source",
        description="Source for total aid denominator",
        options=["dac1_total_aid", "crs_total_aid"],
        chosen="dac1_total_aid",
        rationale="The search results identified DAC1 total aid as the measure for total aid.",
    )
    first = await _us_2023(
        tools["propose_plan"].func,
        ctx,
        decision(),
        part="health",
        whole="dac1_total_gross",
        declared=True,
        own=[source_choice],
    )
    assert first["outcome"] == Outcome.OK.value, first
    plan = load_plan(ctx.state)
    assert plan is not None
    whole = plan.requirement(_WHOLE)

    second = await tools["propose_plan"].func(
        decisions=drafts([source_choice]),
        evidence_requirements=[
            whole.model_copy(update={"decision_ids": (*whole.decision_ids, source_choice.slot)})
        ],
        amends_version=first["version"],
        tool_context=ctx,
    )

    assert second["outcome"] == Outcome.OK.value, second
    plan = load_plan(ctx.state)
    assert plan is not None
    assert source_choice.slot in {d.slot for d in plan.decisions}
    assert source_choice.slot in plan.requirement(_PART).decision_ids
    assert source_choice.slot in plan.requirement(_WHOLE).decision_ids


@pytest.mark.parametrize(
    ("part", "whole", "stated"),
    [
        pytest.param("malaria", "health", False, id="malaria-over-health-aid"),
        pytest.param("health", "health_inclusive", True, id="bilateral-over-inclusive-stated"),
        pytest.param("health", "dac1_total_gross", False, id="dac1-with-no-provider-record"),
    ],
)
async def test_a_share_declaring_no_whole_is_never_rewritten(
    decision: DecisionFactory, part: str, whole: str, stated: bool
) -> None:
    """A share whose whole declares nothing keeps the whole the planner pinned:
    malaria aid over health aid, a whole binding a sector itself; bilateral health
    aid over inclusive health aid with both scopes in the user's words; and CRS
    health aid over a DAC1 total Search states no provider scope for, which nothing
    can compare. No whole is compiled and the plan holds the pins as written."""
    surface = WholeCompiles()
    ctx = _FakeToolContext()
    blocks = [_us_block(part), _us_block(whole)]
    _show_blocks(ctx, *([_stated_provider(block) for block in blocks] if stated else blocks))

    result = await _us_2023(
        _tools_reading(surface)["propose_plan"].func, ctx, decision(), part=part, whole=whole
    )

    assert result["outcome"] == Outcome.OK.value, result
    assert surface.calls == []
    assert _pins(ctx) == [
        (_PART, _us_block(part)["spec_id"]),
        (_WHOLE, _us_block(whole)["spec_id"]),
    ]


@pytest.mark.parametrize(
    ("share", "quotient", "step_id"),
    [
        pytest.param(True, False, _SHARE, id="share"),
        pytest.param(True, True, _SHARE, id="quotient-of-summed-part"),
        pytest.param(False, False, _COMPARE, id="comparison"),
    ],
)
@pytest.mark.parametrize("accept", [False, True], ids=["undeclared", "user-asked"])
async def test_a_calculation_over_a_total_counting_other_providers_is_served_only_when_the_user_asks(
    decision: DecisionFactory, accept: bool, share: bool, quotient: bool, step_id: str
) -> None:
    """CRS bilateral health aid over, or subtracted from, a published total Search
    records as counting core contributions to multilaterals, serving no bilateral
    option, with no whole declared. Undeclared, the calculation is refused for
    repair, naming the input and the bilateral scope it is not served on, and never
    rewritten. Where the user asked for that total ("use the DAC1 total") and the
    calculation accepts the provider difference, it reads DAC1 as pinned, and a
    decision of the planner's, cited by the calculation, says in reader words what
    each side counts."""
    surface = WholeCompiles()
    ctx = _FakeToolContext()
    _show_blocks(ctx, _search_labelled("health"), _search_labelled("no_bilateral"))

    result = await _us_2023(
        _tools_reading(surface)["propose_plan"].func,
        ctx,
        decision(),
        part="health",
        whole="no_bilateral",
        accept=accept,
        share=share,
        quotient=quotient,
    )

    assert surface.calls == []
    if not accept:
        assert result["code"] == ProblemCode.PROVIDERS_UNALIGNED.value, result
        assert result["outcome"] == Outcome.INVALID_INPUT.value
        assert result["detail"]["provider"] == {
            "step_id": step_id,
            "requirement_id": _WHOLE,
            "spec_id": us_2023_block("dac1_total_gross")["spec_id"],
            "option_id": "provider:bilateral",
        }
        _names_the_requirement(result, _WHOLE)
        assert load_plan(ctx.state) is None
        return
    assert result["outcome"] == Outcome.OK.value, result
    dac1 = us_2023_block("dac1_total_gross")["spec_id"]
    assert _pins(ctx) == [(_PART, us_2023_block("health")["spec_id"]), (_WHOLE, dac1)]
    plan = load_plan(ctx.state)
    assert plan is not None
    (stated,) = [d for d in plan.decisions if d.caveat == "providers_differ"]
    assert stated.slot in plan.requirement(step_id).decision_ids
    assert stated.resolved_by is ResolvedBy.PLANNER_DEFAULT
    assert (
        "Health counts bilateral flows only, from OECD Creditor Reporting System (CRS)."
    ) in stated.rationale
    assert (
        "Official development assistance counts bilateral plus core contributions to "
        "multilaterals, from OECD DAC1."
    ) in stated.rationale
    reader_text = " ".join((stated.rationale, *stated.options))
    assert not [rid for rid in (_PART, _WHOLE, step_id) if rid in reader_text]


@pytest.mark.parametrize("repinned", [True, False], ids=["repinned", "declined"])
async def test_a_providers_unaligned_refusal_is_repaired_or_declined(
    decision: DecisionFactory, repinned: bool
) -> None:
    """CRS bilateral health aid over a total serving no bilateral option is refused
    for repair. The planner may then pin another total, and owes no decline for
    it; or, with no whole to declare and no other dataset, keep the part and
    decline the share, a decline the refusal earned."""
    surface = WholeCompiles()
    ctx = _FakeToolContext()
    _show_blocks(ctx, us_2023_block("health"), total_without_bilateral())
    propose = _tools_reading(surface)["propose_plan"].func

    refused = await _us_2023(propose, ctx, decision(), part="health", whole="no_bilateral")
    assert refused["code"] == ProblemCode.PROVIDERS_UNALIGNED.value, refused

    if repinned:
        # The DAC1 total under its own scope record, which states no provider scope.
        _show_blocks(ctx, us_2023_block("health"), us_2023_block("dac1_total_gross"))
        result = await _us_2023(propose, ctx, decision(), part="health", whole="dac1_total_gross")
    else:
        result = await propose(
            question="What share of US aid went to health in 2023?",
            answer_mode=AnswerMode.SERIES,
            decisions=drafts([decision()]),
            evidence_requirements=[
                DatasetRequirement(
                    id=_PART,
                    decision_ids=["measure"],
                    measure="US aid to health",
                    entities=[Entity(name="United States", role="donor")],
                    period=Period(start=2023, end=2023),
                    source_spec_id=us_2023_block("health")["spec_id"],
                )
            ],
            declined=[
                Decline(
                    ask="health aid against total aid", code=UnsupportedOperation.UNSUPPORTED_ASK
                )
            ],
            tool_context=ctx,
        )
    assert result["outcome"] == Outcome.OK.value, result


@pytest.mark.parametrize(
    ("whole", "declared", "caveat"),
    [
        pytest.param("dac1_total_gross", True, "derived_whole", id="derived-whole"),
        pytest.param("dac1_contributions", False, "providers_differ", id="providers-differ"),
    ],
)
async def test_a_plan_at_the_decision_cap_still_carries_its_caveats(
    decision: DecisionFactory, whole: str, declared: bool, caveat: str
) -> None:
    """A share whose plan already states `MAX_PLAN_DECISIONS` decisions of the
    planner's: the decision saying the total is compiled from the part, or that the
    inputs count different providers, qualifies the answer, so it is written and
    cited past the cap."""
    ctx = _FakeToolContext()
    _show_blocks(ctx, us_2023_block("health"), _us_block(whole))
    own = [decision(slot=f"d{index}") for index in range(MAX_PLAN_DECISIONS - 1)]

    result = await _us_2023(
        _tools_reading(WholeCompiles())["propose_plan"].func,
        ctx,
        decision(),
        part="health",
        whole=whole,
        declared=declared,
        accept=not declared,
        own=own,
    )

    assert result["outcome"] == Outcome.OK.value, result
    plan = load_plan(ctx.state)
    assert plan is not None
    (noted,) = [d for d in plan.decisions if d.caveat == caveat]
    assert len([d for d in plan.decisions if d.caveat is None]) == MAX_PLAN_DECISIONS
    assert any(noted.slot in r.decision_ids for r in plan.evidence_requirements)


@pytest.mark.parametrize(
    ("whole", "declared", "caveat"),
    [
        pytest.param("dac1_total_gross", True, "derived_whole", id="derived-whole"),
        pytest.param("dac1_contributions", False, "providers_differ", id="providers-differ"),
    ],
)
async def test_a_caveat_survives_a_revision_restating_its_slot(
    decision: DecisionFactory, whole: str, declared: bool, caveat: str
) -> None:
    """A share whose plan states that the total is compiled from the part, or that
    the inputs count different providers. A revision restating that decision's slot
    in the planner's words leaves code's decision standing, so the answer keeps its
    qualification."""
    tools = _tools_reading(WholeCompiles())
    ctx = _FakeToolContext()
    _show_blocks(ctx, us_2023_block("health"), _us_block(whole))
    first = await _us_2023(
        tools["propose_plan"].func,
        ctx,
        decision(),
        part="health",
        whole=whole,
        declared=declared,
        accept=not declared,
    )
    assert first["outcome"] == Outcome.OK.value, first
    plan = load_plan(ctx.state)
    assert plan is not None
    (noted,) = [d for d in plan.decisions if d.caveat == caveat]

    second = await tools["propose_plan"].func(
        decisions=drafts(
            [noted.model_copy(update={"rationale": "Planner rewording.", "caveat": None})]
        ),
        amends_version=first["version"],
        tool_context=ctx,
    )

    assert second["outcome"] == Outcome.OK.value, second
    plan = load_plan(ctx.state)
    assert plan is not None
    (kept,) = [d for d in plan.decisions if d.slot == noted.slot]
    assert kept == noted
    assert any(noted.slot in r.decision_ids for r in plan.evidence_requirements)


@pytest.mark.parametrize(
    ("whole", "whole_counts"),
    [
        pytest.param("whole", None, id="one-scope"),
        pytest.param(
            "dac1_total_gross",
            "Official development assistance, from OECD DAC1, does not say which providers "
            "it counts.",
            id="no-provider-record",
        ),
        pytest.param(
            "dac1_contributions",
            "Official development assistance counts bilateral plus core contributions to "
            "multilaterals, from OECD DAC1.",
            id="contributions-total",
        ),
        pytest.param(
            "dac1_multilateral",
            "Multilateral official development assistance counts core contributions to "
            "multilaterals, from OECD DAC1.",
            id="multilateral-total",
        ),
        pytest.param(
            "untitled_dac1_total_gross",
            "Official development assistance does not say which providers it counts.",
            id="no-provider-record-or-dataset-name",
        ),
        pytest.param(
            "untitled_dac1_contributions",
            "Official development assistance counts bilateral plus core contributions to "
            "multilaterals.",
            id="contributions-total-no-dataset-name",
        ),
    ],
)
async def test_a_share_accepting_a_provider_difference_states_one_unless_its_inputs_match(
    decision: DecisionFactory,
    whole: str,
    whole_counts: str | None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CRS bilateral health aid over a total the user asked for as it stands. The
    share states a provider difference unless every input's scope record holds one
    provider scope and they match: over the DAC1 total counting core contributions,
    and over a DAC1 total Search states no provider scope for, since nothing says it
    counts the same providers, but not over the CRS bilateral total. The stated
    difference names what each input counts, or that its dataset does not say, and
    the dataset it is from where Search names one. The inputs are named for the
    card only where a difference is stated."""
    named: list[object] = []
    card_names = submission_service._card_names

    def counted(*args: Any, **kwargs: Any) -> dict[str, str]:
        named.append(args)
        return card_names(*args, **kwargs)

    monkeypatch.setattr(submission_service, "_card_names", counted)
    ctx = _FakeToolContext()
    _show_blocks(ctx, _search_labelled("health"), _search_labelled(whole))

    result = await _us_2023(
        _tools_reading(WholeCompiles())["propose_plan"].func,
        ctx,
        decision(),
        part="health",
        whole=whole,
        accept=True,
    )

    assert result["outcome"] == Outcome.OK.value, result
    plan = load_plan(ctx.state)
    assert plan is not None
    stated = [d for d in plan.decisions if d.caveat == "providers_differ"]
    assert len(stated) == (whole_counts is not None)
    assert bool(named) == (whole_counts is not None)
    for noted in stated:
        assert "counts bilateral flows only, from OECD Creditor Reporting System (CRS)." in (
            noted.rationale
        )
        assert whole_counts is not None and whole_counts in noted.rationale


async def test_a_provider_difference_names_inputs_of_one_concept_as_the_card_does(
    decision: DecisionFactory,
) -> None:
    """The CRS bilateral total compared with the DAC1 total, the comparison accepting
    the difference. Search names both "official development assistance", so the
    card tells the two apart, and the stated difference names each input as the
    card does, so its two sentences read two subjects."""
    ctx = _FakeToolContext()
    blocks = [_search_labelled("whole"), _search_labelled("dac1_contributions")]
    _show_blocks(ctx, *blocks)

    result = await _us_2023(
        _tools_reading(WholeCompiles())["propose_plan"].func,
        ctx,
        decision(),
        part="whole",
        whole="dac1_contributions",
        share=False,
        accept=True,
    )

    assert result["outcome"] == Outcome.OK.value, result
    plan = load_plan(ctx.state)
    assert plan is not None
    (stated,) = [d for d in plan.decisions if d.caveat == "providers_differ"]
    # The card's names before compile, read off the blocks the inputs pin.
    uncompiled = plan.model_copy(update={"translation": None, "workflow": None})
    pinned = {block["spec_id"]: source_from_spec_block(block) for block in blocks}
    names = plan_words(uncompiled, pinned=pinned).names
    subjects = {f"{names[rid][:1].upper()}{names[rid][1:]} counts " for rid in (_PART, _WHOLE)}
    assert len(subjects) == 2, subjects
    for subject in subjects:
        assert subject in stated.rationale, (subject, stated.rationale)


async def test_a_revisions_provider_difference_names_its_inputs_as_the_revisions_card_does(
    decision: DecisionFactory,
) -> None:
    """US health aid in 2023 from the CRS, whose card names it "health". A revision adds the inclusive CRS health figures, which Search also
    names "health", and compares them with the first, accepting the difference. The
    revision's card names both health inputs afresh, so the stated difference names
    each as that card does, and the first keeps no name the earlier card gave it."""
    ctx = _FakeToolContext()
    blocks = [_search_labelled(name) for name in ("health", "health_inclusive")]
    _show_blocks(ctx, *blocks)
    tools = _tools_reading(WholeCompiles())
    health = DatasetRequirement(
        id=_PART,
        decision_ids=["measure"],
        measure="US aid to health",
        entities=[Entity(name="United States", role="donor")],
        period=Period(start=2023, end=2023),
        source_spec_id=us_2023_block("health")["spec_id"],
    )
    first = await tools["propose_plan"].func(
        question="How much US aid went to health in 2023?",
        answer_mode=AnswerMode.SERIES,
        decisions=drafts([decision()]),
        evidence_requirements=[health],
        tool_context=ctx,
    )
    assert first["outcome"] == Outcome.OK.value, first
    # The card the revision amends shows the first plan compiled.
    prior = load_plan(ctx.state)
    assert prior is not None
    by_spec_id = {block["spec_id"]: source_from_spec_block(block) for block in blocks}
    discovered = {
        r.id: by_spec_id[r.source_spec_id]
        for r in prior.evidence_requirements
        if isinstance(r, DatasetRequirement) and r.source_spec_id is not None
    }
    tools_listed = await McpSurface(
        sources={"data-mcp": data_mcp, "analysis-mcp": analysis_mcp}
    ).list_tools()
    translation = build_translation(prior, {rid: s.spec_id for rid, s in discovered.items()})
    compiled = compile_plan(
        plan=prior,
        translation=translation,
        schemas={tool.name: tool for tool in tools_listed},
        discovered=discovered,
    )
    assert compiled.workflow is not None, compiled.issues
    shown = prior.with_translation(translation=translation, workflow=compiled.workflow)
    assert plan_words(shown).names[_PART] == "health"
    save_plan(ctx.state, shown)
    inclusive = "health_inclusive"

    second = await tools["propose_plan"].func(
        evidence_requirements=[
            health.model_copy(
                update={"id": inclusive, "source_spec_id": us_2023_block(inclusive)["spec_id"]}
            ),
            ComparisonRequirement(
                id=_COMPARE,
                decision_ids=["measure"],
                minuend=RequirementSide(input_id=inclusive),
                subtrahend=RequirementSide(input_id=_PART),
                accept_provider_difference=True,
            ),
        ],
        amends_version=first["version"],
        tool_context=ctx,
    )

    assert second["outcome"] == Outcome.OK.value, second
    plan = load_plan(ctx.state)
    assert plan is not None
    (stated,) = [
        d for d in plan.decisions if d.caveat == "providers_differ" and d.slot.endswith(_COMPARE)
    ]
    # The revision's card names, read before compile off the blocks the inputs pin.
    uncompiled = plan.model_copy(update={"translation": None, "workflow": None})
    names = plan_words(uncompiled, pinned=by_spec_id).names
    assert names[_PART] != "health", names
    for requirement_id in (_PART, inclusive):
        name = names[requirement_id]
        subject = f"{name[:1].upper()}{name[1:]} counts "
        assert subject in stated.rationale, (subject, stated.rationale)


async def test_a_provider_difference_numbers_namesake_inputs_in_the_order_the_plan_states(
    decision: DecisionFactory,
) -> None:
    """The comparison of `test_a_provider_difference_names_inputs_of_one_concept_as_
    the_card_does` over two CRS totals named alike, one counting bilateral flows and
    the other, declared first, imputed multilateral ones too. The card numbers
    the two "official development assistance" inputs in the order the plan states
    them, whatever order its workflow runs them in, which the compiler sets by
    requirement id (`compiler.compile._canonical_key`), so the stated difference
    numbers them in that order too."""
    ctx = _FakeToolContext()
    inclusive = _another_part(_search_labelled("whole"))
    provider = next(
        c for c in inclusive["resolved"]["scope_choices"]["choices"] if c["axis"] == "provider"
    )
    provider["active"] = "provider:inclusive"
    blocks = {_WHOLE: inclusive, _PART: _search_labelled("whole")}
    _show_blocks(ctx, *blocks.values())
    datasets = [
        DatasetRequirement(
            id=requirement_id,
            decision_ids=["measure"],
            measure="US total ODA",
            entities=[Entity(name="United States of America", role="donor")],
            period=Period(start=2023, end=2023),
            source_spec_id=block["spec_id"],
        )
        for requirement_id, block in blocks.items()
    ]
    compared = ComparisonRequirement(
        id=_COMPARE,
        decision_ids=["measure"],
        minuend=RequirementSide(input_id=_WHOLE),
        subtrahend=RequirementSide(input_id=_PART),
        accept_provider_difference=True,
    )

    result = await _tools_reading(WholeCompiles())["propose_plan"].func(
        question="How does US ODA with imputed multilateral flows compare with bilateral?",
        answer_mode=AnswerMode.COMPUTED,
        decisions=drafts([decision()]),
        evidence_requirements=[*datasets, compared],
        tool_context=ctx,
    )

    assert result["outcome"] == Outcome.OK.value, result
    plan = load_plan(ctx.state)
    assert plan is not None
    (stated,) = [d for d in plan.decisions if d.caveat == "providers_differ"]
    # The card's names over the plan compiled, whose workflow runs the retrievals by id.
    tools = await McpSurface(
        sources={"data-mcp": data_mcp, "analysis-mcp": analysis_mcp}
    ).list_tools()
    discovered = {rid: source_from_spec_block(block) for rid, block in blocks.items()}
    translation = build_translation(plan, {rid: s.spec_id for rid, s in discovered.items()})
    compiled = compile_plan(
        plan=plan,
        translation=translation,
        schemas={tool.name: tool for tool in tools},
        discovered=discovered,
    )
    assert compiled.workflow is not None, compiled.issues
    plan = plan.with_translation(translation=translation, workflow=compiled.workflow)
    pinned = {block["spec_id"]: source_from_spec_block(block) for block in blocks.values()}
    names = plan_words(plan, pinned=pinned).names
    assert names[_WHOLE].endswith("(1)"), names
    for requirement_id, block in blocks.items():
        (choice,) = [
            c for c in block["resolved"]["scope_choices"]["choices"] if c["axis"] == "provider"
        ]
        (label,) = [o["label"] for o in choice["options"] if o["option_id"] == choice["active"]]
        name = names[requirement_id]
        sentence = f"{name[:1].upper()}{name[1:]} counts {label}"
        assert sentence in stated.rationale, (sentence, stated.rationale)


@pytest.mark.parametrize(
    ("whole", "labelled", "counted"),
    [
        pytest.param(
            "dac1_contributions",
            True,
            "Health counts bilateral plus imputed multilateral, from OECD Creditor Reporting "
            "System (CRS) and ONE Imputed Multilateral Sectors. Official development "
            "assistance counts bilateral plus core contributions to multilaterals, from OECD "
            "DAC1.",
            id="contributions-total",
        ),
        pytest.param(
            "dac1_total_gross",
            True,
            "Health counts bilateral plus imputed multilateral, from OECD Creditor Reporting "
            "System (CRS) and ONE Imputed Multilateral Sectors. Official development "
            "assistance, from OECD DAC1, does not say which providers it counts.",
            id="no-provider-record",
        ),
        pytest.param(
            "dac1_contributions",
            False,
            "Health incl. population and reproductive health (CRS 120 + 130) counts "
            "bilateral plus imputed multilateral, from bilateral plus imputed multilateral "
            "(OECD CRS and ONE Imputed Multilateral Sectors). Gross disbursements (official "
            "development assistance) counts bilateral plus core contributions to "
            "multilaterals, from OECD DAC1.",
            id="before-search-labels",
        ),
    ],
)
async def test_a_carried_share_restates_its_provider_difference_when_an_input_is_repinned(
    decision: DecisionFactory, whole: str, labelled: bool, counted: str
) -> None:
    """CRS bilateral health aid over the DAC1 total, the share accepting the
    difference. A revision re-pins the part alone to the inclusive CRS form, as a
    plan-card switch does, and carries the share: the share's stated difference now
    names what the re-pinned part counts, one sentence per input naming its concept,
    the provider scope it counts and the tables it is published in, the composite's
    both. From a Search that sends no concept or tables, each input is named by the
    labels Search composed, as written."""
    block = _search_labelled if labelled else _us_block
    tools = _tools_reading(WholeCompiles())
    ctx = _FakeToolContext()
    _show_blocks(ctx, block("health"), block("health_inclusive"), block(whole))
    first = await _us_2023(
        tools["propose_plan"].func,
        ctx,
        decision(),
        part="health",
        whole=whole,
        accept=True,
    )
    assert first["outcome"] == Outcome.OK.value, first
    plan = load_plan(ctx.state)
    assert plan is not None
    part = plan.requirement(_PART)
    assert isinstance(part, DatasetRequirement)

    second = await tools["propose_plan"].func(
        evidence_requirements=[
            part.model_copy(update={"source_spec_id": us_2023_block("health_inclusive")["spec_id"]})
        ],
        amends_version=first["version"],
        tool_context=ctx,
    )

    assert second["outcome"] == Outcome.OK.value, second
    plan = load_plan(ctx.state)
    assert plan is not None
    (stated,) = [d for d in plan.decisions if d.caveat == "providers_differ"]
    assert stated.slot in plan.requirement(_SHARE).decision_ids
    assert stated.rationale == (
        "The question asks for the inputs as their datasets count them, so they may count "
        f"different providers' spending. {counted}"
    )


@pytest.mark.parametrize(
    ("title", "measure", "form"),
    [
        pytest.param("OECD CRS", "US aid", "titled", id="titled"),
        pytest.param("OECD CRS " + "t" * 200, "US aid", "untitled", id="untitled"),
        pytest.param("OECD CRS " + "t" * 200, "US aid " + "m" * 190, "counted", id="counted"),
    ],
)
def test_a_provider_difference_over_four_inputs_is_stated_within_the_rationale_limit(
    title: str, measure: str, form: str
) -> None:
    """Four US inputs summed, two counting bilateral aid and two imputed
    multilateral aid, each from a dataset titled `title` and named by the
    concept Search states for it, `measure`: the difference is stated in full while it fits the rationale's
    limit, then with each input named without its dataset title, then as its
    first sentence and how many inputs the sum reads, and is never dropped."""
    ids = ("a", "b", "c", "d")
    blocks = {}
    for index, input_id in enumerate(ids):
        spec_id = f"spec_{str(index + 5) * 32}"
        block = provider_block(
            spec_id,
            narrow=False,
            active="provider:bilateral" if index < 2 else "provider:inclusive",
        )
        block["resolved"]["facets"] = {"provenance": {"ref": "dataset", "label": title}}
        block["resolved"]["concept_label"] = f"{measure} {input_id}"
        blocks[input_id] = block
    requirements: list[EvidenceRequirement] = [
        DatasetRequirement(
            id=input_id,
            measure=f"{measure} {input_id}",
            entities=[Entity(name="United States", role="donor")],
            period=Period(start=2023, end=2023),
            source_spec_id=block["spec_id"],
        )
        for input_id, block in blocks.items()
    ]
    requirements += [
        SumOfRequirement(
            id="ab", left=RequirementSide(input_id="a"), right=RequirementSide(input_id="b")
        ),
        SumOfRequirement(
            id="cd", left=RequirementSide(input_id="c"), right=RequirementSide(input_id="d")
        ),
        SumOfRequirement(
            id="all", left=RequirementSide(input_id="ab"), right=RequirementSide(input_id="cd")
        ),
    ]
    shown = {block["spec_id"]: PinSource(block=block, settled=True) for block in blocks.values()}

    (stated,) = submission_service._difference_decisions(
        requirements, sources=SourceReadings.of_shown(shown)
    )["all"]

    lead = (
        "The inputs are set to different provider scopes, so they count different "
        "providers' spending."
    )
    assert stated.caveat == "providers_differ"
    assert len(stated.rationale) <= MAX_DECISION_RATIONALE_CHARS
    assert stated.rationale.startswith(lead)
    assert (title in stated.rationale) is (form == "titled")
    if form == "counted":
        assert stated.rationale == f"{lead} The calculation reads 4 inputs."
    else:
        assert stated.rationale.count(" counts ") == 4


@pytest.mark.parametrize(
    ("b_counts", "accept", "differs"),
    [
        pytest.param("provider:inclusive", False, True, id="set-apart"),
        pytest.param("provider:inclusive", True, True, id="accepted"),
        pytest.param("provider:bilateral", False, False, id="one-scope"),
    ],
)
def test_a_sum_states_the_provider_difference_of_its_inputs_beside_one_with_no_record(
    b_counts: str, accept: bool, differs: bool
) -> None:
    """US aid a, set to bilateral flows, plus b, set to `b_counts`, plus the DAC1
    total c, whose dataset states no provider scope: (a + b) + c. Where a and b
    count different providers, the outer sum states so as the inner one does,
    whether or not it accepts the difference; c's missing record does not hide
    it. Where a and b match, c alone states nothing on a sum not accepting it."""
    a = provider_block(
        f"spec_{'a' * 32}", narrow=False, active="provider:bilateral", provenance="explicit"
    )
    b = provider_block(f"spec_{'b' * 32}", narrow=False, active=b_counts, provenance="explicit")
    c = us_2023_block("dac1_total_gross")
    requirements: list[EvidenceRequirement] = [
        DatasetRequirement(
            id=input_id,
            measure=f"US aid {input_id}",
            entities=[Entity(name="United States", role="donor")],
            period=Period(start=2023, end=2023),
            source_spec_id=block["spec_id"],
        )
        for input_id, block in (("a", a), ("b", b), ("c", c))
    ]
    requirements += [
        SumOfRequirement(
            id="ab", left=RequirementSide(input_id="a"), right=RequirementSide(input_id="b")
        ),
        SumOfRequirement(
            id="abc",
            left=RequirementSide(input_id="ab"),
            right=RequirementSide(input_id="c"),
            accept_provider_difference=accept,
        ),
    ]
    shown = {block["spec_id"]: PinSource(block=block, settled=True) for block in (a, b, c)}

    stated = submission_service._difference_decisions(
        requirements, sources=SourceReadings.of_shown(shown)
    )

    assert [d.caveat for d in stated["abc"]] == ["providers_differ"] * differs
    assert [d.caveat for d in stated["ab"]] == ["providers_differ"] * differs


def test_a_calculations_difference_reads_its_input_groups_without_their_sides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """US aid a, set to bilateral flows, plus b, set to the inclusive scope: the sum
    states the provider difference off the members of its input group alone, and
    derives none of its sides (`_side_derivations`), which only alignment reads."""

    def derived(*_: object) -> None:
        raise AssertionError("a difference decision derived a calculation's sides")

    monkeypatch.setattr(submission_service, "_side_derivations", derived)
    a = provider_block(
        f"spec_{'a' * 32}", narrow=False, active="provider:bilateral", provenance="explicit"
    )
    b = provider_block(
        f"spec_{'b' * 32}", narrow=False, active="provider:inclusive", provenance="explicit"
    )
    requirements: list[EvidenceRequirement] = [
        DatasetRequirement(
            id=input_id,
            measure=f"US aid {input_id}",
            entities=[Entity(name="United States", role="donor")],
            period=Period(start=2023, end=2023),
            source_spec_id=block["spec_id"],
        )
        for input_id, block in (("a", a), ("b", b))
    ]
    requirements.append(
        SumOfRequirement(
            id="ab", left=RequirementSide(input_id="a"), right=RequirementSide(input_id="b")
        )
    )
    shown = {block["spec_id"]: PinSource(block=block, settled=True) for block in (a, b)}

    stated = submission_service._difference_decisions(
        requirements, sources=SourceReadings.of_shown(shown)
    )

    assert [d.caveat for d in stated["ab"]] == ["providers_differ"]


def test_a_share_over_the_dac1_total_moves_it_to_bilateral_oda(decision: DecisionFactory) -> None:
    """CRS bilateral health aid over the DAC1 total, which counts core contributions to
    multilaterals and offers bilateral flows as DAC1 bilateral ODA, an option binding
    its aid-type slot in place of a provider scope. That option counts bilateral
    flows, so the share puts the total on it: the total is named for compiling on
    `provider:bilateral`."""
    ctx = _FakeToolContext()
    _show_blocks(ctx, us_2023_block("health"), dac1_with_contributions())

    refused = _us_2023(
        _tools_by_name()["propose_plan"].func,
        ctx,
        decision(),
        part="health",
        whole="dac1_contributions",
    )

    assert refused["code"] == ProblemCode.BASES_UNALIGNED.value, refused
    assert [
        (r["spec_id"], r["option_ids"], r["requirement_id"]) for r in refused["detail"]["realign"]
    ] == [(us_2023_block("dac1_total_gross")["spec_id"], ["provider:bilateral"], _WHOLE)]


async def test_a_comparison_with_a_total_stating_no_provider_scope_keeps_both_datasets(
    decision: DecisionFactory,
) -> None:
    """The DAC1 total, whose scope record states no provider scope, minus bilateral
    CRS health aid: nothing says the sides count different providers, so neither is
    compiled again and each keeps its own dataset."""
    surface = WholeCompiles()
    ctx = _FakeToolContext()
    _show_blocks(ctx, us_2023_block("health"), us_2023_block("dac1_total_gross"))

    result = await _us_2023(
        _tools_reading(surface)["propose_plan"].func,
        ctx,
        decision(),
        part="health",
        whole="dac1_total_gross",
        share=False,
    )

    assert result["outcome"] == Outcome.OK.value, result
    assert surface.calls == []
    assert _pins(ctx) == [
        (_PART, us_2023_block("health")["spec_id"]),
        (_WHOLE, us_2023_block("dac1_total_gross")["spec_id"]),
    ]


@pytest.mark.parametrize(
    "flagged_share", [False, True], ids=["sum-alone", "beside-a-flagged-share"]
)
async def test_a_provider_difference_accepted_on_one_share_leaves_every_other_calculation_aligned(
    decision: DecisionFactory, flagged_share: bool
) -> None:
    """CRS bilateral health aid plus a total counting core contributions to
    multilaterals, serving no bilateral option, in a sum mixes providers, so the sum
    is refused, whether or not a share reading the same total accepts the
    provider difference: the user's request exempts that one share. Beside a
    comparison of the same inputs accepting nothing, the comparison, first in plan
    order, is refused first."""
    ctx = _FakeToolContext()
    _show_blocks(ctx, us_2023_block("health"), total_without_bilateral())
    summed = SumOfRequirement(
        id=_SUM,
        decision_ids=["measure"],
        left=RequirementSide(input_id=_PART),
        right=RequirementSide(input_id=_WHOLE),
    )

    result = await _us_2023(
        _tools_reading(WholeCompiles())["propose_plan"].func,
        ctx,
        decision(),
        part="health",
        whole="no_bilateral",
        share=flagged_share,
        accept=flagged_share,
        extra=[summed],
    )

    assert result["code"] == ProblemCode.PROVIDERS_UNALIGNED.value, result
    assert result["detail"]["provider"]["step_id"] == (_SUM if flagged_share else _COMPARE)
    assert load_plan(ctx.state) is None


@pytest.mark.parametrize(
    ("part", "refusal", "code", "cause", "calls"),
    [
        pytest.param(
            "grants",
            None,
            ProblemCode.SHARE_WHOLE_UNDERIVABLE,
            "axis_not_held",
            0,
            id="part-with-no-sector-choice",
        ),
        pytest.param(
            "health",
            {"code": "requested_option_unavailable", "retry_mode": "do_not_retry"},
            ProblemCode.SHARE_WHOLE_UNDERIVABLE,
            "whole_refused",
            1,
            id="option-unavailable",
        ),
        pytest.param(
            "health",
            {"code": "upstream_unavailable", "retry_mode": "retry_after_delay"},
            ProblemCode.SURFACE_UNAVAILABLE,
            None,
            1,
            id="search-unavailable",
        ),
    ],
)
async def test_a_declared_whole_that_cannot_be_counted_from_its_part_is_refused_for_repair(
    decision: DecisionFactory,
    part: str,
    refusal: dict[str, str] | None,
    code: ProblemCode,
    cause: str | None,
    calls: int,
) -> None:
    """A whole declared over US grants, a CRS part Search offers no sector choice on,
    or one Search will not count the way the part is counted: the plan is refused
    with a code of its own, which asks the planner to repin the part or drop the
    declaration, and records no decline. Search not answering stays a retry."""
    surface = WholeCompiles(refusal=refusal)
    ctx = _FakeToolContext()
    _show_blocks(ctx, us_2023_block(part), us_2023_block("dac1_total_gross"))

    refused = await _us_2023(
        _tools_reading(surface)["propose_plan"].func,
        ctx,
        decision(),
        part=part,
        whole="dac1_total_gross",
        declared=True,
    )

    assert refused["code"] == code.value, refused
    assert len(surface.calls) == calls
    if cause is not None:
        assert refused["outcome"] == Outcome.INVALID_INPUT.value
        assert refused["detail"]["whole"] == {
            "whole_id": _WHOLE,
            "part_spec_id": us_2023_block(part)["spec_id"],
            "cause": cause,
        }
        _names_the_requirement(refused, _WHOLE)
    assert load_plan(ctx.state) is None


def _unreadable_scope(block: dict[str, Any]) -> dict[str, Any]:
    """`block` with scope records this release cannot read."""
    copied = json.loads(json.dumps(block))
    copied["resolved"]["scope_choices"] = {"id": "not a scope"}
    return copied


@pytest.mark.parametrize(
    ("source", "cause", "axis"),
    [
        pytest.param(None, "part_not_shown", "sector", id="pin-source-missing"),
        pytest.param(PinSource(settled=True), "part_block_missing", "sector", id="block-missing"),
        pytest.param(
            PinSource(block=_unreadable_scope(us_2023_block("health")), settled=True),
            "scope_unreadable",
            "sector",
            id="scope-unreadable",
        ),
        pytest.param(
            PinSource(block=us_2023_block("grants"), settled=True),
            "axis_not_held",
            "sector",
            id="no-sector-slot",
        ),
        pytest.param(
            PinSource(block=us_2023_block("health"), settled=True),
            "axis_not_held",
            "recipient",
            id="no-recipient-bound",
        ),
    ],
)
async def test_a_whole_compile_names_why_it_could_not_read_its_part(
    source: PinSource | None, cause: str, axis: str
) -> None:
    """A declared whole whose part's pin is not among the datasets shown, is shown
    with no block, holds scope records this release cannot read, offers no sector
    choice binding a slot, or binds no recipient to leave open: each is refused
    under its own cause, with no call to Search, so the planner repairs the right
    thing."""
    surface = WholeCompiles()
    ctx = _FakeToolContext()
    pin = us_2023_block("health")["spec_id"]
    save_pin_sources(ctx.state, {pin: source} if source is not None else {})

    refused = await planner_agent.compile_whole(
        cast(ToolSurface, surface),
        Realignment(pin, (), _WHOLE, _WHOLE, whole_axis=axis),
        state=ctx.state,
    )

    assert refused is not None
    assert refused["code"] == ProblemCode.SHARE_WHOLE_UNDERIVABLE.value
    assert refused["detail"]["whole"]["cause"] == cause
    assert surface.calls == []


@pytest.mark.parametrize(
    ("source", "cause"),
    [
        pytest.param(PinSource(settled=True), "part_block_missing", id="block-missing"),
        pytest.param(
            PinSource(block=_unreadable_scope(us_2023_block("health")), settled=True),
            "scope_unreadable",
            id="scope-unreadable",
        ),
    ],
)
async def test_a_declared_whole_names_why_its_part_could_not_be_read(
    decision: DecisionFactory, source: PinSource, cause: str
) -> None:
    """A whole declared over the US health part, pinned to a dataset the planner was
    shown with no block or with scope records this release cannot read: the plan is
    refused before any compile, under the cause `compile_whole` gives the same
    failure."""
    surface = WholeCompiles()
    ctx = _FakeToolContext()
    pin = us_2023_block("health")["spec_id"]
    _show_blocks(ctx, us_2023_block("dac1_total_gross"))
    save_pin_sources(ctx.state, {**pin_sources(ctx.state), pin: source})

    refused = await _us_2023(
        _tools_reading(surface)["propose_plan"].func,
        ctx,
        decision(),
        part="health",
        whole="dac1_total_gross",
        declared=True,
    )

    assert refused["code"] == ProblemCode.SHARE_WHOLE_UNDERIVABLE.value, refused
    assert refused["detail"]["whole"]["cause"] == cause
    assert surface.calls == []


_KENYA_RECIPIENT = Entity(name="Kenya", role="recipient")


@pytest.mark.parametrize(
    ("whole_of", "entities", "across"),
    [
        pytest.param(WholeOf(part=_SHARE, axis="sector"), None, (), id="part-is-a-calculation"),
        pytest.param(WholeOf(part=_WHOLE, axis="sector"), None, (), id="part-is-itself"),
        pytest.param(
            WholeOf(part=_PART, axis="sector"),
            [Entity(name="France", role="donor")],
            (),
            id="other-parties",
        ),
        # All aid to Kenya does not hold US aid to every recipient.
        pytest.param(
            WholeOf(part=_PART, axis="donor"),
            [_KENYA_RECIPIENT],
            ("donor",),
            id="donor-whole-names-a-party-the-part-does-not",
        ),
        # A donor whole names one group in place of 'donor' only where it is not the
        # part's own donor, which would divide the part by itself.
        pytest.param(
            WholeOf(part=_PART, axis="donor"), None, (), id="donor-whole-names-the-parts-donor"
        ),
        pytest.param(
            WholeOf(part=_PART, axis="donor"),
            [Entity(name="DAC countries", role="donor"), Entity(name="France", role="donor")],
            (),
            id="donor-whole-names-two-donors",
        ),
        # The part names no recipient, so there is none to leave open.
        pytest.param(
            WholeOf(part=_PART, axis="recipient"),
            None,
            ("recipient",),
            id="recipient-whole-of-a-part-naming-none",
        ),
    ],
)
def test_a_whole_of_naming_no_part_it_can_be_counted_from_is_an_invalid_plan(
    decision: DecisionFactory,
    whole_of: WholeOf,
    entities: list[Entity] | None,
    across: tuple[str, ...],
) -> None:
    """A whole declared over a calculation, over itself, for other parties than its
    part, over every donor while naming a party the part does not, or over every
    recipient of a part naming none, does not hold its part or cannot be compiled
    from the part's pin, so the plan is refused for repair."""
    ctx = _FakeToolContext()
    _show_blocks(ctx, us_2023_block("health"))
    whole = DatasetRequirement(
        id=_WHOLE,
        decision_ids=["measure"],
        measure="total ODA",
        entities=entities if entities is not None else [Entity(name="United States", role="donor")],
        across=across,
        period=Period(start=2023, end=2023),
        whole_of=whole_of,
    )

    refused = _tools_by_name()["propose_plan"].func(
        question="What share of US aid went to health in 2023?",
        answer_mode=AnswerMode.COMPUTED,
        decisions=drafts([decision()]),
        evidence_requirements=[
            DatasetRequirement(
                id=_PART,
                decision_ids=["measure"],
                measure="US aid to health",
                entities=[Entity(name="United States", role="donor")],
                period=Period(start=2023, end=2023),
                source_spec_id=us_2023_block("health")["spec_id"],
            ),
            whole,
            ShareOfRequirement(
                id=_SHARE,
                decision_ids=["measure"],
                numerator=RequirementSide(input_id=_PART),
                denominator=RequirementSide(input_id=_WHOLE),
            ),
        ],
        tool_context=ctx,
    )

    assert refused["outcome"] == Outcome.INVALID_INPUT.value, refused
    assert refused["code"] == ProblemCode.INVALID_PLAN.value, refused
    assert load_plan(ctx.state) is None


# --- a share's whole over every donor or recipient -------------------------------

# Search's answers for ODA to Kenya with no donor named, as the planner-entry eval
# recorded them (`search_spec_dac_total_and_donor_open.json`): `dac_total` binds
# `origin` to the published total of official donors (DAC), `donor_open` leaves 143
# providers open on it.
_KENYA_ODA_NO_DONOR: dict[str, Any] = json.loads(
    (
        Path(__file__).parents[1] / "fixtures" / "search_spec_dac_total_and_donor_open.json"
    ).read_text()
)
# dfo-apis's offline engine answering the resubmit `compile_whole` posts for a DAC1 or
# DAC2a donor's whole over every donor (`share_whole_dac_donor_wholes.json`).
_DAC_DONOR_WHOLES: dict[str, Any] = json.loads(
    (Path(__file__).parents[1] / "fixtures" / "share_whole_dac_donor_wholes.json").read_text()
)
_AFRICA_PART = f"spec_{'a' * 32}"
# The spec_id Search answers each whole with, by the axis it opens.
_OPENED_WHOLE = {"donor": f"spec_{'b' * 32}", "recipient": f"spec_{'e' * 32}"}


def _us_health_to_africa() -> dict[str, Any]:
    """`US_2023`'s CRS health part read as US health aid to Africa: Africa's countries
    bound on `destination`, Search's named set for them, and the region choice dfo-apis
    gives a summed region (`search.scope.region.region_rule` there), its residual
    bucket added to the role, under a spec_id of its own."""
    block = json.loads(json.dumps(us_2023_block("health")))
    block["spec_id"] = _AFRICA_PART
    africa = ["country/KEN", "country/NGA", "country/ETH"]
    scope = block["resolved"]["scope_choices"]
    scope["id"] = scope["base"]["spec_id"] = scope["compiled"]["spec_id"] = _AFRICA_PART
    for identity in (scope["base"]["identity"], block["definition"]["identity"]):
        identity["entities"]["destination"] = africa
    scope["compiled"]["identity"]["entities"]["destination"] = [*africa, "AfricaRegional"]
    block["resolved"]["entity_groups"] = {"destination": {"label": "Africa", "ref": "africa"}}
    available = {"status": "available", "detail": {}}
    scope["choices"].append(
        {
            "id": f"{_AFRICA_PART}:region",
            "axis": "region",
            "rule": "region_residuals",
            "provenance": "default",
            "active": "region:members_and_residuals",
            "options": [
                {
                    "option_id": "region:members_and_residuals",
                    "label": "Africa with aid to Africa not allocated by country",
                    "binding": {
                        "kind": "entities",
                        "role": "destination",
                        "add": ["AfricaRegional"],
                    },
                    "availability": available,
                },
                {
                    "option_id": "region:members_only",
                    "label": "Africa's countries only",
                    "binding": {"kind": "entities", "role": "destination", "add": []},
                    "availability": available,
                },
            ],
        }
    )
    return block


class _OpenedWholeCompiles:
    """A tool surface whose `resolve_definition` answers a whole with a role left open
    as `part`, the part's spec block, with the posted identity and entries, the choices
    it posts entries for and the sector, under `_OPENED_WHOLE`'s spec_id for the role it
    opens. `groups` are the sets Search names on the answer; else it counts the members
    left open on the role, as Search does where no total holds them. With `bound`,
    the answer's identity binds those parties on the opened role. Each call's
    arguments are recorded in `calls`."""

    def __init__(
        self,
        part: dict[str, Any],
        *,
        groups: Mapping[str, Any] | None = None,
        bound: list[str] | None = None,
    ) -> None:
        self.part = part
        self.groups = groups
        self.bound = bound
        self.calls: list[Mapping[str, Any]] = []

    async def list_tools(self) -> list[Any]:
        return []

    async def call_tool(
        self, *, tool: ToolId, arguments: Mapping[str, Any], on_progress: Any = None
    ) -> ToolCallResult:
        del on_progress
        assert tool.name == "resolve_definition", tool
        self.calls.append(arguments)
        identity = arguments["identity"]
        held = self.part["resolved"]["scope_choices"]["base"]["identity"]["entities"]
        (opened,) = set(held) - set(identity["entities"])
        spec_id = _OPENED_WHOLE["donor" if opened == "origin" else "recipient"]
        if self.bound is not None:
            identity = {**identity, "entities": {**identity["entities"], opened: self.bound}}
        block = json.loads(json.dumps(self.part))
        block["spec_id"] = spec_id
        block["definition"] = {"schema_version": 1, "spec_id": spec_id, "identity": identity}
        resolved = block["resolved"]
        resolved["entities"] = [e for e in resolved["entities"] if e.get("role") != opened]
        resolved["entity_groups"] = {
            role: group
            for role, group in (resolved.get("entity_groups") or {}).items()
            if role != opened
        } | dict(self.groups or {})
        if self.groups is None:
            resolved["role_axis_facts"] = [
                {"role": opened, "member_count": 143, "members_available": True}
            ]
        scope = resolved["scope_choices"]
        scope["id"] = scope["base"]["spec_id"] = scope["compiled"]["spec_id"] = spec_id
        scope["base"]["identity"] = scope["compiled"]["identity"] = identity
        scope["entries"] = arguments["scope_entries"]
        posted = {entry["axis"] for entry in arguments["scope_entries"]}
        scope["choices"] = [
            {
                **choice,
                "id": f"{spec_id}:{choice['axis']}",
                "provenance": "explicit" if choice["axis"] in posted else choice["provenance"],
            }
            for choice in scope["choices"]
            if choice["axis"] in posted or choice["axis"] == "sector"
        ]
        part = {"status": "definite", "spec": block}
        return ToolCallResult(is_error=False, meta={SEARCH_ENVELOPE_META_KEY: {"parts": [part]}})


async def _share_over_every(
    surface: _OpenedWholeCompiles,
    ctx: _FakeToolContext,
    block: dict[str, Any],
    axis: str,
    decision: DecisionFactory,
    *,
    year: int = 2023,
) -> dict[str, object]:
    """Propose the part `block` as a share of its whole over every `axis`, a donor or
    recipient, summed by year, over `year`, with `surface` answering the whole; `block`
    is shown.
    The part names its donor by Search's label for it, and Africa as its recipient
    where `block` names Africa's set there, else Search's label for its recipient."""
    _show_blocks(ctx, block)
    resolved = block["resolved"]
    labels = {e["role"]: e["label"] for e in resolved["entities"]}
    held = resolved["scope_choices"]["base"]["identity"]["entities"]
    named = [Entity(name=labels["origin"], role="donor")]
    if "destination" in held:
        group = (resolved.get("entity_groups") or {}).get("destination")
        recipient = group["label"] if group else labels["destination"]
        named.append(Entity(name=recipient, role="recipient"))
    side = "giving" if axis == "donor" else "receiving"
    period = Period(start=year, end=year)
    return await _tools_reading(cast(ToolSurface, surface))["propose_plan"].func(
        question="What share of the aid came from the United States in 2023?",
        answer_mode=AnswerMode.COMPUTED,
        decisions=drafts([decision()]),
        evidence_requirements=[
            DatasetRequirement(
                id=_PART,
                decision_ids=["measure"],
                measure="US aid",
                entities=named,
                period=period,
                source_spec_id=block["spec_id"],
            ),
            DatasetRequirement(
                id=_WHOLE,
                decision_ids=["measure"],
                measure="total aid",
                entities=[e for e in named if role_party(cast(str, e.role)) != side],
                across=(axis,),
                period=period,
                whole_of=WholeOf(part=_PART, axis=axis),
            ),
            AggregateRequirement(
                id=_TOTAL,
                decision_ids=["measure"],
                input_id=_WHOLE,
                reduction=Reduction.SUM,
                group_by=(GroupBy.PERIOD,),
            ),
            ShareOfRequirement(
                id=_SHARE,
                decision_ids=["measure"],
                numerator=RequirementSide(input_id=_PART),
                denominator=RequirementSide(input_id=_TOTAL),
            ),
        ],
        tool_context=ctx,
    )


@pytest.mark.parametrize(
    ("part", "axis", "groups", "total", "entries", "posted_groups", "alike", "refused"),
    [
        # US health aid to Africa over all health aid to Africa: Search leaves every
        # reporting donor open on bilateral flows; Africa, its set and its region stay.
        pytest.param(
            "africa",
            "donor",
            None,
            "total over every donor",
            {"sector", "region", "measure_basis", "provider"},
            ["destination"],
            "recipients, sectors, measure basis and providers",
            None,
            id="donor-crs-bilateral",
        ),
        # US health aid to Africa over all US health aid: the region choice and the
        # named set on the recipient side go with the role.
        pytest.param(
            "africa",
            "recipient",
            None,
            "total over every recipient",
            {"sector", "measure_basis", "provider"},
            [],
            "donors, sectors, measure basis and providers",
            None,
            id="recipient-crs",
        ),
        # The same over every donor, where Search's set for Africa names a class of
        # party this release does not know. The set on the recipient side the whole
        # keeps is not readable, and a whole posted without it can bind another
        # population, so the whole is refused with no search.
        pytest.param(
            "africa_unknown_class",
            "donor",
            None,
            "total over every donor",
            {"sector", "region", "measure_basis", "provider"},
            [],
            "recipients, sectors, measure basis and providers",
            "groups_unreadable",
            id="donor-part-set-unreadable",
        ),
    ],
)
async def test_a_declared_whole_over_every_donor_or_recipient_leaves_that_role_open(
    decision: DecisionFactory,
    part: str,
    axis: str,
    groups: Mapping[str, Any] | None,
    total: str,
    entries: set[str],
    posted_groups: list[str],
    alike: str,
    refused: str | None,
) -> None:
    """A share whose whole declares itself its part over every donor or every
    recipient: the whole is compiled from the part's base identity with that role's
    column dropped, the entity set named on it and the region choice adding to it
    left out, every other choice held as an entry, the provider scope as a carried
    one Search may drop over every donor and an explicit one over every recipient,
    whose total names the part's donor, and no search. Search's answer is the population the whole
    holds, and the caveat names it and what both sides' dataset counts alike. A
    part whose set on a kept role is not readable refuses the whole under the cause
    `refused`, with no search."""
    block = _us_health_to_africa() if part.startswith("africa") else _us_block(part)
    if part == "africa_unknown_class":
        block["resolved"]["entity_groups"]["destination"]["whole_of"] = ["sovereign_fund"]
    surface = _OpenedWholeCompiles(block, groups=groups)
    ctx = _FakeToolContext()

    result = await _share_over_every(surface, ctx, block, axis, decision)

    if refused is not None:
        assert result.get("code") == ProblemCode.SHARE_WHOLE_UNDERIVABLE.value, result
        assert result["detail"]["whole"]["cause"] == refused
        assert surface.calls == []
        return
    assert result["outcome"] == Outcome.OK.value, result
    (call,) = surface.calls
    scope = block["resolved"]["scope_choices"]
    column = "origin" if axis == "donor" else "destination"
    identity = json.loads(json.dumps(scope["base"]["identity"]))
    del identity["entities"][column]
    assert call["identity"] == identity
    assert {entry["axis"] for entry in call["scope_entries"]} == entries
    provider = "carried" if axis == "donor" else "explicit"
    assert {e["axis"]: e["provenance"] for e in call["scope_entries"]}["provider"] == provider
    assert list(call.get("entity_groups") or {}) == posted_groups
    assert _pins(ctx) == [(_PART, block["spec_id"]), (_WHOLE, _OPENED_WHOLE[axis])]
    plan = load_plan(ctx.state)
    assert plan is not None
    (stated,) = [d for d in plan.decisions if d.caveat == "derived_whole"]
    assert stated.chosen.endswith(total), stated.chosen
    assert stated.rationale.endswith(f"count the same {alike}."), stated.rationale
    assert stated.slot in plan.requirement(_WHOLE).decision_ids


def test_a_whole_search_counts_on_another_provider_scope_says_what_each_counts() -> None:
    """US inclusive health aid over every donor's health aid, which Search counts on
    bilateral flows, the population of every reporter that holds the part: the
    caveat says which provider scope each side counts, and claims no shared one."""
    part = us_2023_block("health_inclusive")
    whole = AlignedPin(
        origin=part["spec_id"],
        option_ids=(),
        spec_id=_OPENED_WHOLE["donor"],
        digest={"spec_block": us_2023_block("whole")},
        whole_axis="donor",
    )

    (stated,) = submission_service._whole_decisions(
        {_WHOLE: whole}, [], shown={part["spec_id"]: PinSource(block=part, settled=True)}
    )[_WHOLE]

    assert stated.rationale.endswith(
        "so both sides count the same recipients, sectors and measure basis. The total "
        "counts bilateral flows only, the part bilateral plus imputed multilateral."
    ), stated.rationale


@pytest.mark.parametrize(
    ("unstated", "providers"),
    [
        (
            "whole",
            "The total does not say which providers it counts, and the part counts "
            "bilateral plus imputed multilateral.",
        ),
        (
            "health_inclusive",
            "The total counts bilateral flows only, and the part does not say which "
            "providers it counts.",
        ),
    ],
    ids=["whole-unstated", "part-unstated"],
)
def test_a_whole_one_side_of_which_states_no_provider_claims_no_difference(
    unstated: str, providers: str
) -> None:
    """US inclusive health aid over every donor's, its dataset counting nothing
    else the whole and part could share and one of them stating no provider
    scope: the caveat says what the one side counts and that the other does not
    say, and claims no difference it cannot know."""
    blocks = {
        name: json.loads(json.dumps(us_2023_block(name))) for name in ("health_inclusive", "whole")
    }
    part = blocks["health_inclusive"]
    part["resolved"].pop("role_axis_facts")
    for name, block in blocks.items():
        scope = block["resolved"]["scope_choices"]
        scope["choices"] = [
            choice
            for choice in scope["choices"]
            if choice["axis"] == "provider" and name != unstated
        ]
    whole = AlignedPin(
        origin=part["spec_id"],
        option_ids=(),
        spec_id=_OPENED_WHOLE["donor"],
        digest={"spec_block": blocks["whole"]},
        whole_axis="donor",
    )

    (stated,) = submission_service._whole_decisions(
        {_WHOLE: whole}, [], shown={part["spec_id"]: PinSource(block=part, settled=True)}
    )[_WHOLE]

    assert stated.rationale.endswith(providers), stated.rationale
    assert "different providers" not in stated.rationale, stated.rationale


# Entries Search drops beside the whole's own carried provider entry: one the user's
# words asked for, and one carried from another reading's choice.
_OTHER_DROPPED: list[dict[str, Any]] = [
    {
        "entry": {
            "axis": "measure_basis",
            "option_id": "measure_basis:net",
            "provenance": "explicit",
            "origin": {"kind": "text", "text": "net", "start": 0, "end": 3},
        },
        "code": "stated_choice_not_applied",
        "cause": "option_unavailable",
    },
    {
        "entry": {
            "axis": "provider",
            "option_id": "provider:inclusive",
            "provenance": "carried",
            "origin": {"kind": "choice", "choice_id": f"spec_{'f' * 32}:provider"},
        },
        "code": "carried_choice_not_applied",
        "cause": "option_unavailable",
    },
]


class _NoInclusiveWithoutDonor(_OpenedWholeCompiles):
    """`_OpenedWholeCompiles` holding dfo-apis's provider rule for a total naming no
    donor, which never adds imputed figures (`search.scope.provider.
    _providers_not_offered` there, why `all_providers`): an explicit entry on the
    inclusive scope is refused as Search refuses it (`requested_option_unavailable`,
    data-mcp's `upstream_error` around Search's problem), and a carried one is
    dropped (`carried_choice_not_applied`) and the whole counted on bilateral flows.
    Search also reports `_OTHER_DROPPED`, entries the harness did not post."""

    @override
    async def call_tool(
        self, *, tool: ToolId, arguments: Mapping[str, Any], on_progress: Any = None
    ) -> ToolCallResult:
        entries = arguments["scope_entries"]
        position, provider = next(
            (n, entry) for n, entry in enumerate(entries) if entry["axis"] == "provider"
        )
        if "origin" in arguments["identity"]["entities"] or provider["option_id"] != (
            "provider:inclusive"
        ):
            return await super().call_tool(tool=tool, arguments=arguments)
        if provider["provenance"] == "explicit":
            self.calls.append(arguments)
            refused = to_tool_result(
                ToolRefusal.upstream_error(
                    upstream_code="requested_option_unavailable",
                    detail=f"Scope entry {position} cannot apply to this part: "
                    "provider_not_offered.",
                    retry_after_s=None,
                )
            )
            return ToolCallResult(is_error=True, structured_content=refused.structured_content)
        answer = await super().call_tool(tool=tool, arguments=arguments)
        (part,) = cast(dict[str, Any], answer.meta)[SEARCH_ENVELOPE_META_KEY]["parts"]
        scope = part["spec"]["resolved"]["scope_choices"]
        scope["entries"] = [entry for entry in entries if entry is not provider]
        scope["dropped_entries"] = [
            {
                "entry": provider,
                "code": "carried_choice_not_applied",
                "cause": "option_unavailable",
            },
            *_OTHER_DROPPED,
        ]
        for choice in scope["choices"]:
            if choice["axis"] == "provider":
                choice.update(
                    active="provider:bilateral", provenance="default", rule="bilateral_flows"
                )
        return answer


async def test_a_donor_whole_of_an_inclusive_part_is_counted_on_the_scope_search_offers_it(
    decision: DecisionFactory,
) -> None:
    """US health aid counted with its imputed multilateral share, over every donor's
    health aid: Search offers the inclusive scope on no total naming no donor, so
    the whole posts the part's scope as a carried entry and Search counts it on
    bilateral flows. The part keeps its scope, the whole's caveat says what each
    side counts, and the share states no provider difference beside it. The whole
    counts another provider option than the part, so the entry Search dropped stays
    beside every other dropped entry, and the whole's reading says it did not
    apply."""
    block = us_2023_block("health_inclusive")
    surface = _NoInclusiveWithoutDonor(block)
    ctx = _FakeToolContext()

    result = await _share_over_every(surface, ctx, block, "donor", decision)

    assert result["outcome"] == Outcome.OK.value, result
    (call,) = surface.calls
    (provider,) = [e for e in call["scope_entries"] if e["axis"] == "provider"]
    assert (provider["option_id"], provider["provenance"]) == ("provider:inclusive", "carried")
    whole_pin = _OPENED_WHOLE["donor"]
    assert _pins(ctx) == [(_PART, block["spec_id"]), (_WHOLE, whole_pin)]
    whole_block = pin_sources(ctx.state)[whole_pin].spec_block(whole_pin)
    assert whole_block is not None
    choices = whole_block["resolved"]["scope_choices"]["choices"]
    assert {c["axis"]: (c["active"], c["provenance"]) for c in choices}["provider"] == (
        "provider:bilateral",
        "default",
    )
    # What the whole's reading stores and its basis line states as not applied.
    dropped = reading_facts(whole_block)["scope"]["dropped_entries"]
    assert [(d["entry"]["option_id"], d["entry"]["origin"]) for d in dropped] == [
        (provider["option_id"], provider["origin"]),
        *((d["entry"]["option_id"], d["entry"]["origin"]) for d in _OTHER_DROPPED),
    ]
    plan = load_plan(ctx.state)
    assert plan is not None
    (stated,) = [d for d in plan.decisions if d.caveat is not None]
    assert stated.caveat == "derived_whole"
    assert stated.rationale.endswith(
        "so both sides count the same recipients, sectors and measure basis. The total "
        "counts bilateral flows only, the part bilateral plus imputed multilateral."
    ), stated.rationale


# Search's donor-side group for official donors (DAC), as a donor-less DAC2a answer
# names it: a whole-population total of official donors, beside which DAC2a also
# publishes private donors.
_OFFICIAL_DONORS = _KENYA_ODA_NO_DONOR["dac_total"]["resolved"]["entity_groups"]["origin"]
# The DAC countries' group as Search names it on a donor-less DAC1 answer.
_DAC_COUNTRIES = _DAC_DONOR_WHOLES["dac1"]["whole"]["resolved"]["entity_groups"]["origin"]


@pytest.mark.parametrize(
    ("bound", "group", "cause"),
    [
        # Search binds donors as parties, the US among them.
        pytest.param(["country/USA", "country/FRA"], None, None, id="parties-holding-the-part"),
        # Search binds donors that leave the US out.
        pytest.param(
            ["country/FRA", "country/DEU"], None, "part_outside_whole", id="parties-without-it"
        ),
        # Search binds DAC countries' total and lists its members, the US among them.
        pytest.param(["DACCountriesTotal"], _DAC_COUNTRIES, None, id="members-holding-the-part"),
        # The same total, its members listed without the US.
        pytest.param(
            ["DACCountriesTotal"],
            {**_DAC_COUNTRIES, "members": ["country/FRA", "country/DEU"]},
            "part_outside_whole",
            id="members-without-the-part",
        ),
        # The same total with no member list: nothing shows the US is among them.
        pytest.param(
            ["DACCountriesTotal"],
            {k: v for k, v in _DAC_COUNTRIES.items() if k != "members"},
            "whole_members_unstated",
            id="total-listing-no-members",
        ),
        # Official donors, a whole population of its class, where the dataset also
        # publishes private donors: nothing says which class the US is.
        pytest.param(
            ["DACOfficialDonors"],
            _OFFICIAL_DONORS,
            "whole_members_unstated",
            id="whole-class-beside-another",
        ),
        # The same total stating no classes for its side, so nothing says the side
        # holds official donors alone.
        pytest.param(
            ["DACOfficialDonors"],
            {k: v for k, v in _OFFICIAL_DONORS.items() if k != "side_classes"},
            "whole_members_unstated",
            id="whole-class-side-unstated",
        ),
        # The same total stating residuals alone for its side, which is no class of party.
        pytest.param(
            ["DACOfficialDonors"],
            {**_OFFICIAL_DONORS, "side_classes": ["residual"]},
            "whole_members_unstated",
            id="whole-class-side-residuals-only",
        ),
        # The same total where the side publishes no class besides official donors and
        # residuals: every party there is of its class.
        pytest.param(
            ["DACOfficialDonors"],
            {**_OFFICIAL_DONORS, "side_classes": ["official", "residual"]},
            None,
            id="whole-class-of-the-side",
        ),
        # A class of party this release does not know, on the side or the whole: the
        # group is unreadable, so nothing shows the total holds the US.
        pytest.param(
            ["DACOfficialDonors"],
            {**_OFFICIAL_DONORS, "side_classes": ["official", "sovereign_fund"]},
            "whole_members_unstated",
            id="side-class-unknown",
        ),
        pytest.param(
            ["DACOfficialDonors"],
            {**_OFFICIAL_DONORS, "whole_of": ["sovereign_fund"]},
            "whole_members_unstated",
            id="whole-class-unknown",
        ),
    ],
)
async def test_a_donor_whole_is_served_only_where_search_shows_it_holds_the_parts_donor(
    decision: DecisionFactory, bound: list[str], group: dict[str, Any] | None, cause: str | None
) -> None:
    """US health aid to Africa over every donor's, where Search's answer binds donors
    on the opened role: the share is served where those hold the US, as parties,
    as the members Search lists for the total it binds, or as a whole-population
    total of every class the side publishes. It is refused as a whole that does
    not contain its part where Search shows the US is outside
    (`part_outside_whole`), and as one nothing shows holds it where the bound
    total lists no members and no class of party holding it
    (`whole_members_unstated`). A whole binding nothing there counts every donor
    of the dataset
    (`test_a_declared_whole_over_every_donor_or_recipient_leaves_that_role_open`)."""
    block = _us_health_to_africa()
    groups = {"origin": group} if group is not None else None
    surface = _OpenedWholeCompiles(block, bound=bound, groups=groups)
    ctx = _FakeToolContext()

    result = await _share_over_every(surface, ctx, block, "donor", decision)

    if cause is None:
        assert result["outcome"] == Outcome.OK.value, result
        assert dict(_pins(ctx))[_WHOLE] == _OPENED_WHOLE["donor"]
        return
    assert result["code"] == ProblemCode.SHARE_WHOLE_UNDERIVABLE.value, result
    assert result["detail"]["whole"]["cause"] == cause
    assert load_plan(ctx.state) is None


_G7 = [
    "country/CAN",
    "country/DEU",
    "country/FRA",
    "country/GBR",
    "country/ITA",
    "country/JPN",
    "country/USA",
]


def _g7_health_to_africa(members: list[str] | None) -> dict[str, Any]:
    """`_us_health_to_africa` read from the G7's published total in place of the US:
    `GroupOfSeven` bound on `origin`, under Search's named set for it listing
    `members` (none where `None`)."""
    block = _us_health_to_africa()
    scope = block["resolved"]["scope_choices"]
    for identity in (
        scope["base"]["identity"],
        scope["compiled"]["identity"],
        block["definition"]["identity"],
    ):
        identity["entities"]["origin"] = ["GroupOfSeven"]
    block["resolved"]["entities"] = [
        {"ref": "GroupOfSeven", "label": "G7", "entity_type": "Place", "role": "origin"}
    ]
    group: dict[str, Any] = {"label": "G7", "total": "GroupOfSeven"}
    if members is not None:
        group["members"] = members
    block["resolved"]["entity_groups"]["origin"] = group
    return block


@pytest.mark.parametrize(
    ("members", "cause"),
    [
        # Every G7 member is among the DAC countries Search lists for the whole.
        pytest.param(_G7, None, id="members-inside"),
        # One member outside the DAC countries.
        pytest.param([*_G7, "country/ARE"], "part_outside_whole", id="a-member-outside"),
        # The G7 total with no members listed: nothing shows the whole holds it.
        pytest.param(None, "whole_members_unstated", id="members-unlisted"),
    ],
)
async def test_a_donor_whole_holds_a_group_part_where_it_holds_each_listed_member(
    decision: DecisionFactory, members: list[str] | None, cause: str | None
) -> None:
    """G7 health aid to Africa over DAC countries' total, the part bound to the G7's
    own total: the whole holds the part where it holds every member Search lists for
    the G7, does not where one member is outside (`part_outside_whole`), and nothing
    shows it does where Search lists no G7 members (`whole_members_unstated`)."""
    block = _g7_health_to_africa(members)
    surface = _OpenedWholeCompiles(
        block, bound=["DACCountriesTotal"], groups={"origin": _DAC_COUNTRIES}
    )
    ctx = _FakeToolContext()

    result = await _share_over_every(surface, ctx, block, "donor", decision)

    if cause is None:
        assert result["outcome"] == Outcome.OK.value, result
        assert dict(_pins(ctx))[_WHOLE] == _OPENED_WHOLE["donor"]
        return
    assert result["code"] == ProblemCode.SHARE_WHOLE_UNDERIVABLE.value, result
    assert result["detail"]["whole"]["cause"] == cause
    assert load_plan(ctx.state) is None


class _RecordedWhole:
    """A tool surface whose `resolve_definition` answers the donor whole of
    `_DAC_DONOR_WHOLES[name]` as dfo-apis did, once the post is the one recorded."""

    def __init__(self, name: str) -> None:
        self.recorded = _DAC_DONOR_WHOLES[name]
        self.calls: list[Mapping[str, Any]] = []

    async def list_tools(self) -> list[Any]:
        return []

    async def call_tool(
        self, *, tool: ToolId, arguments: Mapping[str, Any], on_progress: Any = None
    ) -> ToolCallResult:
        del on_progress
        assert tool.name == "resolve_definition", tool
        self.calls.append(arguments)
        posted = self.recorded["posted"]
        assert arguments["identity"] == posted["identity"]
        assert arguments["scope_entries"] == posted["scope_entries"]
        part = {"status": "definite", "spec": json.loads(json.dumps(self.recorded["whole"]))}
        return ToolCallResult(is_error=False, meta={SEARCH_ENVELOPE_META_KEY: {"parts": [part]}})


@pytest.mark.parametrize(
    ("name", "total", "alike", "dropped"),
    [
        # The US DAC1 total over DAC countries' total, on the part's own provider scope.
        # DAC1 has no recipient or sector, so the caveat names neither.
        pytest.param(
            "dac1",
            "total for DAC countries",
            "the same measure basis and providers.",
            [],
            id="dac1",
        ),
        # US bilateral ODA over DAC countries' total, where that total holds no
        # bilateral ODA: Search drops the carried bilateral entry and counts the whole
        # on total ODA. The caveat says what each side counts, and the dropped entry
        # stays, so the whole's reading says the part's scope did not apply.
        pytest.param(
            "dac1_bilateral_uncovered",
            "total for DAC countries",
            "the same measure basis. The total counts bilateral plus core contributions "
            "to multilaterals, the part bilateral flows only.",
            ["provider:bilateral"],
            id="dac1-bilateral-over-total-oda",
        ),
        # US ODA to Kenya on DAC2a over every donor's: Search binds no donor and offers
        # no provider axis there, so it drops the carried entry as `axis_absent`, which
        # changed nothing and is not held. The caveat says the total states no
        # provider scope.
        pytest.param(
            "dac2a",
            "total over every donor",
            "the same recipients and measure basis. The total does not say which "
            "providers it counts, and the part counts bilateral flows only.",
            [],
            id="dac2a",
        ),
    ],
)
async def test_a_dac_donor_whole_reads_searchs_answer_for_it(
    decision: DecisionFactory, name: str, total: str, alike: str, dropped: list[str]
) -> None:
    """A donor's DAC1 or DAC2a part as a share of its whole over every donor, Search
    answering the resubmit `compile_whole` posts as dfo-apis's engine answered it:
    DAC1 binds DAC countries' total and lists its members, the US among them, and
    DAC2a binds no donor. The caveat names the total, what both sides count alike
    and, where the provider options differ or one side states none, what each
    counts. A carried entry Search dropped stays only where the whole counts
    another option (`dropped`)."""
    surface = _RecordedWhole(name)
    ctx = _FakeToolContext()
    block = json.loads(json.dumps(surface.recorded["part"]))

    result = await _share_over_every(cast(Any, surface), ctx, block, "donor", decision)

    assert result["outcome"] == Outcome.OK.value, result
    whole_pin = surface.recorded["whole"]["spec_id"]
    assert _pins(ctx) == [(_PART, block["spec_id"]), (_WHOLE, whole_pin)]
    whole_block = pin_sources(ctx.state)[whole_pin].spec_block(whole_pin)
    assert whole_block is not None
    held = reading_facts(whole_block)["scope"]["dropped_entries"]
    assert [d["entry"]["option_id"] for d in held] == dropped
    plan = load_plan(ctx.state)
    assert plan is not None
    (stated,) = [d for d in plan.decisions if d.caveat is not None]
    assert stated.caveat == "derived_whole"
    assert stated.chosen.endswith(total), stated.chosen
    assert stated.rationale.endswith(f"count {alike}"), stated.rationale


class _SearchRefused:
    """A surface that answers no call, recording the name of each it was asked."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    async def list_tools(self) -> list[Any]:
        return []

    async def call_tool(self, *, tool: ToolId, arguments: Any, on_progress: Any = None) -> Any:
        del arguments, on_progress
        self.calls.append(tool.name)
        raise AssertionError(f"{tool.name} called")


@pytest.mark.parametrize(
    ("name", "group"),
    [
        pytest.param("dac1", "DAC countries", id="dac1"),
        pytest.param("dac1_bilateral_uncovered", "DAC countries", id="dac1-bilateral-uncovered"),
        pytest.param("dac2a", None, id="dac2a"),
    ],
)
async def test_a_dac_donor_whole_searchs_answer_compiles(
    decision: DecisionFactory, name: str, group: str | None
) -> None:
    """US ODA as a share of its whole over every donor, the whole pinned to Search's
    answer to the resubmit `compile_whole` posts: the translator takes the pin with no
    fresh search and the plan compiles, whether Search binds the donor side to one
    published total (DAC1's DAC countries' total) or leaves it open (DAC2a)."""
    surface = _RecordedWhole(name)
    ctx = _FakeToolContext()
    block = json.loads(json.dumps(surface.recorded["part"]))
    proposed = await _share_over_every(cast(Any, surface), ctx, block, "donor", decision, year=2019)
    assert proposed["outcome"] == Outcome.OK.value, proposed

    refused = _SearchRefused()
    named = await name_pinned_groups(ctx, surface=cast(ToolSurface, refused))
    named_in_place = [(g.requirement_id, g.entity.name, g.across) for g in named]
    assert named_in_place == ([(_WHOLE, group, "donor")] if group else [])
    discover = {t.name: t for t in build_discovery_tools(surface=cast(ToolSurface, refused))}
    for requirement_id in (_PART, _WHOLE):
        found = await discover["discover_source"].func(
            requirement_id=requirement_id, tool_context=ctx
        )
        assert found["status"] == "definite", (requirement_id, found)
    assert refused.calls == []

    plan = load_plan(ctx.state)
    assert plan is not None
    pins = dict(_pins(ctx))
    shown = pin_sources(ctx.state)
    discovered = {
        rid: source_from_spec_block(cast(Any, shown[spec_id].spec_block(spec_id)))
        for rid, spec_id in pins.items()
        if spec_id is not None
    }
    tools = await McpSurface(
        sources={"data-mcp": data_mcp, "analysis-mcp": analysis_mcp}
    ).list_tools()
    translation = build_translation(plan, {rid: s.spec_id for rid, s in discovered.items()})
    compiled = compile_plan(
        plan=plan,
        translation=translation,
        schemas={tool.name: tool for tool in tools},
        discovered=discovered,
    )
    assert compiled.workflow is not None, compiled.issues


def test_a_wholes_caveat_reads_each_blocks_scope_record_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The caveat on US bilateral ODA's whole over every donor, DAC countries' total
    ODA, reads the part's and the whole's scope records once each, whatever it
    states from them."""
    recorded = _DAC_DONOR_WHOLES["dac1_bilateral_uncovered"]
    part, whole_block = recorded["part"], recorded["whole"]
    read: list[str] = []
    parse = submission_service.part_scope

    def counted(spec_id: str, block: Mapping[str, Any] | None) -> Any:
        read.append(spec_id)
        return parse(spec_id, block)

    monkeypatch.setattr(submission_service, "part_scope", counted)
    whole = AlignedPin(
        origin=part["spec_id"],
        option_ids=(),
        spec_id=whole_block["spec_id"],
        digest={"spec_block": whole_block},
        whole_axis="donor",
    )

    (stated,) = submission_service._whole_decisions(
        {_WHOLE: whole}, [], shown={part["spec_id"]: PinSource(block=part, settled=True)}
    )[_WHOLE]

    assert sorted(read) == sorted([part["spec_id"], whole_block["spec_id"]])
    assert stated.rationale.endswith("the part bilateral flows only."), stated.rationale


def test_a_wholes_caveat_names_its_total_by_axis_where_searchs_groups_are_unreadable() -> None:
    """DAC countries' total ODA as the whole of US ODA, where Search's group for it
    names a class of party this release does not know: the caveat is stated, its
    total named by the axis the whole opens."""
    recorded = _DAC_DONOR_WHOLES["dac1"]
    part, whole_block = recorded["part"], json.loads(json.dumps(recorded["whole"]))
    whole_block["resolved"]["entity_groups"]["origin"]["whole_of"] = ["sovereign_fund"]
    whole = AlignedPin(
        origin=part["spec_id"],
        option_ids=(),
        spec_id=whole_block["spec_id"],
        digest={"spec_block": whole_block},
        whole_axis="donor",
    )

    (stated,) = submission_service._whole_decisions(
        {_WHOLE: whole}, [], shown={part["spec_id"]: PinSource(block=part, settled=True)}
    )[_WHOLE]

    assert stated.chosen.endswith("total over every donor"), stated.chosen


def test_a_recipient_a_part_binds_through_its_region_choice_alone_is_held_for_a_whole() -> None:
    """US health aid to Africa's residual bucket, which the part binds on `destination`
    only through the option its region choice holds active: the part binds a recipient
    a whole over every recipient can leave open, as the identity it compiles to binds
    one there."""
    block = _us_health_to_africa()
    scope_record = block["resolved"]["scope_choices"]
    del scope_record["base"]["identity"]["entities"]["destination"]
    scope_record["compiled"]["identity"]["entities"]["destination"] = ["AfricaRegional"]
    scope = submission_service.part_scope(block["spec_id"], block)
    assert scope is not None

    assert submission_service.whole_axis_held(scope, "recipient")


def _bound_through_a_choice(block: dict[str, Any], column: str) -> dict[str, Any]:
    """A copy of `block` binding its parties on `column` only through the option a
    choice holds active: in the identity its scope compiles to, not its base identity."""
    copied = json.loads(json.dumps(block))
    del copied["resolved"]["scope_choices"]["base"]["identity"]["entities"][column]
    return copied


def _unreadable_recipient_set(block: dict[str, Any]) -> dict[str, Any]:
    """A copy of `block` stating a recipient set naming a class of party this release
    does not know, beside whatever it states on its donor side."""
    copied = json.loads(json.dumps(block))
    groups = copied["resolved"].setdefault("entity_groups", {})
    groups["destination"] = {"label": "Sovereign funds", "whole_of": ["sovereign_fund"]}
    return copied


@pytest.mark.parametrize(
    ("part_of", "whole_of"),
    [
        pytest.param(None, None, id="bound-in-the-base-identity"),
        # The Emirates bound only through a choice's option, as a region option binds
        # its donors: the whole is still checked against them.
        pytest.param(_bound_through_a_choice, None, id="bound-through-a-choice"),
        # An unreadable set on the recipient side says nothing of the donor total.
        pytest.param(None, _unreadable_recipient_set, id="unreadable-set-on-another-role"),
    ],
)
def test_a_dac1_donor_whole_of_a_donor_outside_the_dac_is_refused(
    part_of: Callable[[dict[str, Any], str], dict[str, Any]] | None,
    whole_of: Callable[[dict[str, Any]], dict[str, Any]] | None,
) -> None:
    """The United Arab Emirates' DAC1 total over DAC countries' total, which Search
    binds for its whole and whose members it lists without the Emirates: the
    whole does not contain the part."""
    recorded = _DAC_DONOR_WHOLES["dac1_non_dac"]
    part = recorded["part"] if part_of is None else part_of(recorded["part"], "origin")
    whole = recorded["whole"] if whole_of is None else whole_of(recorded["whole"])
    scope = submission_service.part_scope(part["spec_id"], part)
    assert scope is not None

    held = planner_agent._parties_outside(scope, whole, "origin", part_block=part)

    assert (held.outside, held.unshown) == (("country/ARE",), ())


# dfo-apis's offline engine answering DAC countries' DAC1 totals on each provider scope,
# and CRS health aid to Kenya from IDA and from the United States
# (`sum_multilateral_outflows.json`).
_SUM_BLOCKS: dict[str, Any] = json.loads(
    (Path(__file__).parents[1] / "fixtures" / "sum_multilateral_outflows.json").read_text()
)


def _named(block: dict[str, Any]) -> list[Entity]:
    """The parties `block` binds, as a plan names them: a set's total by Search's label
    for the set, each other entity by its label."""
    resolved = block["resolved"]
    groups = resolved.get("entity_groups") or {}
    roles = {"origin": "donor", "destination": "recipient"}
    return [
        Entity(
            name=groups[e["role"]]["label"]
            if (groups.get(e["role"]) or {}).get("total") == e["ref"]
            else e["label"],
            role=roles[e["role"]],
        )
        for e in resolved["entities"]
    ]


def _unmarked(block: dict[str, Any]) -> dict[str, Any]:
    """A copy of `block` stating nothing of multilaterals, as a Search answer predating
    the markers, or one for an organisation total with no members, does: no
    `multilateral` on its entities, no `holds_multilaterals` on its groups or role
    facts."""
    copied = json.loads(json.dumps(block))
    resolved = copied["resolved"]
    for held in (
        *(resolved.get("entities") or ()),
        *(resolved.get("entity_groups") or {}).values(),
        *(resolved.get("role_axis_facts") or ()),
    ):
        held.pop("multilateral", None)
        held.pop("holds_multilaterals", None)
    return copied


def _beside_ida(block: dict[str, Any]) -> dict[str, Any]:
    """A copy of `block`, a DAC countries' total, binding IDA beside that total on its
    donor side, under a spec_id of its own, with IDA's entity as Search marks it
    (`ida_health_to_kenya`): a multilateral organisation."""
    copied = json.loads(json.dumps(block))
    copied["spec_id"] = f"spec_{'8' * 32}"
    resolved = copied["resolved"]
    ida = next(
        e
        for e in _SUM_BLOCKS["ida_health_to_kenya"]["resolved"]["entities"]
        if e["role"] == "origin"
    )
    resolved["entities"] = [*resolved["entities"], ida]
    scope = resolved["scope_choices"]
    for identity in (scope["base"]["identity"], scope["compiled"]["identity"]):
        identity["entities"]["origin"] = [*identity["entities"]["origin"], ida["ref"]]
    return copied


_SUM_BLOCKS["dac_countries_and_ida_bilateral"] = _beside_ida(_SUM_BLOCKS["dac_countries_bilateral"])


def _group_unnamed(block: dict[str, Any]) -> dict[str, Any]:
    """A copy of `block` whose donor set names neither its total nor its members, only
    what it holds (`holds_multilaterals`)."""
    copied = json.loads(json.dumps(block))
    group = copied["resolved"]["entity_groups"]["origin"]
    group.pop("total")
    group.pop("members")
    return copied


_SUM_BLOCKS["dac_countries_unnamed_and_ida_bilateral"] = _group_unnamed(
    _SUM_BLOCKS["dac_countries_and_ida_bilateral"]
)
_SUM_BLOCKS["dac_countries_multilateral_again"] = {
    **_SUM_BLOCKS["dac_countries_multilateral"],
    "spec_id": f"spec_{'9' * 32}",
}


@pytest.mark.parametrize(
    ("first", "second", "unmarked", "refused"),
    [
        # Multilateral ODA counts the DAC countries' core contributions, which IDA spends.
        pytest.param(
            "dac_countries_multilateral",
            "ida_health_to_kenya",
            None,
            "provider:multilateral",
            id="multilateral-oda-and-ida",
        ),
        # Total ODA counts them too, beside the DAC countries' bilateral flows.
        pytest.param(
            "dac_countries_contributions",
            "ida_health_to_kenya",
            None,
            "provider:contributions",
            id="total-oda-and-ida",
        ),
        # Bilateral ODA counts the DAC countries' own flows alone, and IDA's spending is
        # not theirs: no money is counted twice.
        pytest.param(
            "dac_countries_bilateral", "ida_health_to_kenya", None, None, id="bilateral-oda"
        ),
        # US bilateral health aid and IDA's share no money either.
        pytest.param(
            "us_health_to_kenya", "ida_health_to_kenya", None, None, id="us-bilateral-and-ida"
        ),
        # IDA stating no marker: nothing says its outflows are not a multilateral's.
        pytest.param(
            "dac_countries_multilateral",
            "ida_health_to_kenya",
            _PART,
            "provider:multilateral",
            id="multilateral-oda-and-unmarked-ida",
        ),
        # DAC countries' total stating nothing of what it holds: its bilateral ODA may
        # be a multilateral's outflows, which spend the contributions multilateral ODA
        # counts. With the marker, the two are served (`bilateral-and-core-contributions`
        # in `test_a_sum_accepting_a_provider_difference_is_still_refused_where_it_
        # counts_flows_twice`).
        pytest.param(
            "dac_countries_bilateral",
            "dac_countries_multilateral",
            _TOTAL,
            "provider:bilateral",
            id="unmarked-bilateral-oda-and-multilateral-oda",
        ),
        # DAC countries' total marked as holding no multilateral, bound beside IDA. The
        # bilateral ODA of the two counts IDA's outflows, which spend the contributions
        # multilateral ODA counts.
        pytest.param(
            "dac_countries_and_ida_bilateral",
            "dac_countries_multilateral",
            None,
            "provider:bilateral",
            id="bilateral-oda-of-dac-countries-and-ida-and-multilateral-oda",
        ),
        # The same, the DAC countries' set naming neither its total nor its members.
        # Its marker speaks for the donors it covers, and IDA, bound beside it with a
        # marker of its own, counts its own outflows.
        pytest.param(
            "dac_countries_unnamed_and_ida_bilateral",
            "dac_countries_multilateral",
            None,
            "provider:bilateral",
            id="bilateral-oda-of-an-unnamed-set-and-ida-and-multilateral-oda",
        ),
        # Multilateral ODA twice on one option, the second's donors stated nothing of, so
        # their own flows may be a multilateral's outflows. The option counts the core
        # contributions alone and no reporter's own flows, so both count one stream.
        pytest.param(
            "dac_countries_multilateral",
            "dac_countries_multilateral_again",
            _PART,
            None,
            id="multilateral-oda-and-unmarked-multilateral-oda",
        ),
    ],
)
async def test_a_sum_with_a_multilaterals_outflows_is_refused_where_the_other_counts_its_funding(
    decision: DecisionFactory,
    first: str,
    second: str,
    unmarked: str | None,
    refused: str | None,
) -> None:
    """Two figures added, the sum accepting a provider difference. IDA's health aid to
    Kenya, which Search marks a multilateral's outflows, spends the core contributions
    its funders paid it, so it counts multilateral money: the sum is refused beside a
    figure counting the DAC countries' core contributions, on multilateral ODA or total
    ODA, since no donor or year keeps the two apart and the contributions would be
    counted twice. Beside the DAC countries' bilateral ODA or US bilateral health aid,
    which count bilateral money alone, it is served, unless IDA is bound beside the
    DAC countries there. A figure whose donors Search states nothing of (`unmarked`)
    may be a multilateral's outflows. The refusal names the second figure and the
    option the first is counted on (`refused`)."""
    blocks = {
        _TOTAL: _SUM_BLOCKS[first],
        _PART: _SUM_BLOCKS[second],
    }
    if unmarked is not None:
        blocks[unmarked] = _unmarked(blocks[unmarked])
    ctx = _FakeToolContext()
    _show_blocks(ctx, *blocks.values())
    period = Period(start=2019, end=2019)

    result = await _tools_reading(WholeCompiles())["propose_plan"].func(
        question="How much did these two give in 2019, together?",
        answer_mode=AnswerMode.COMPUTED,
        decisions=drafts([decision()]),
        evidence_requirements=[
            *(
                DatasetRequirement(
                    id=requirement_id,
                    decision_ids=["measure"],
                    measure=f"figure {requirement_id}",
                    entities=_named(block),
                    period=period,
                    source_spec_id=block["spec_id"],
                )
                for requirement_id, block in blocks.items()
            ),
            SumOfRequirement(
                id=_SUM,
                decision_ids=["measure"],
                left=RequirementSide(input_id=_TOTAL),
                right=RequirementSide(input_id=_PART),
                accept_provider_difference=True,
            ),
        ],
        tool_context=ctx,
    )

    if refused is None:
        assert result["outcome"] == Outcome.OK.value, result
        return
    assert result["code"] == ProblemCode.OPERATION_UNSUPPORTED.value, result
    assert result["detail"]["basis"] == {
        "spec_id": blocks[_PART]["spec_id"],
        "option_ids": [refused],
        "requirement_id": _PART,
        "step_id": _SUM,
    }
    assert load_plan(ctx.state) is None


@pytest.mark.parametrize(
    ("whole_of", "money"),
    [
        pytest.param(None, {"bilateral"}, id="readable-set"),
        pytest.param(["sovereign_fund"], {"bilateral", "multilateral"}, id="unreadable-set"),
        # An unreadable set on the recipient side leaves the donor set readable.
        pytest.param("destination", {"bilateral"}, id="unreadable-set-on-another-role"),
    ],
)
def test_a_donor_sets_marker_speaks_for_its_donors_only_where_the_set_is_readable(
    whole_of: list[str] | str | None, money: set[str]
) -> None:
    """DAC countries' bilateral ODA, whose donor set Search marks as holding no
    multilateral: its reporters' own flows count bilateral money. Where the set names
    a class of party this release does not know, the set is not read, so its marker
    says nothing and the total reads as an unmarked organisation's, which may hold a
    multilateral."""
    block = json.loads(json.dumps(_SUM_BLOCKS["dac_countries_bilateral"]))
    if whole_of == "destination":
        block = _unreadable_recipient_set(block)
    elif whole_of is not None:
        block["resolved"]["entity_groups"]["origin"]["whole_of"] = whole_of
    scope = submission_service.part_scope(block["spec_id"], block)
    assert scope is not None

    assert submission_service._own_flows(scope, block) == money


def test_donors_a_part_binds_through_a_choice_alone_count_their_own_flows() -> None:
    """IDA's health aid to Kenya, IDA bound only through the option a choice holds
    active, as a region option binds its donors: the part's flows lie with IDA, and
    its own flows count multilateral money, as IDA bound in the base identity does."""
    block = _bound_through_a_choice(_SUM_BLOCKS["ida_health_to_kenya"], "origin")
    scope = submission_service.part_scope(block["spec_id"], block)
    assert scope is not None

    assert submission_service._Flows.of(scope, block).donors == {
        "InternationalDevelopmentAssociation"
    }
    assert submission_service._own_flows(scope, block) == {"multilateral"}


def _searched_in_words(block: dict[str, Any], *, stated: bool = True) -> dict[str, Any]:
    """A copy of `block` as the planner holds a search for it (`planner.agent.pin_by_search`):
    every axis Search settled other than by default marked as the planner's. With
    `stated`, Search read its provider row in the searched words ("multilateral ODA"),
    an explicit entry from text in place of the entry from a choice it was recorded
    with."""
    copied = json.loads(json.dumps(block))
    scope = copied["resolved"]["scope_choices"]
    if stated:
        scope["entries"] = [
            {
                **entry,
                "origin": {"kind": "text", "text": copied["resolved"]["concept_label"]}
                | {"start": 0, "end": len(copied["resolved"]["concept_label"])},
            }
            if entry["axis"] == "provider"
            else entry
            for entry in scope["entries"]
        ]
    copied[PLANNER_AXES_BLOCK_KEY] = sorted(
        c["axis"] for c in scope["choices"] if c["provenance"] != "default"
    )
    return copied


def _dac_countries(requirement_id: str, block: dict[str, Any]) -> DatasetRequirement:
    """`requirement_id` pinning `block` over 2019, naming the parties it binds (`_named`)."""
    return DatasetRequirement(
        id=requirement_id,
        decision_ids=["measure"],
        measure=f"figure {requirement_id}",
        entities=_named(block),
        period=Period(start=2019, end=2019),
        source_spec_id=block["spec_id"],
    )


@pytest.mark.parametrize(
    ("second", "stated", "refused"),
    [
        pytest.param("dac_countries_multilateral", True, None, id="rows-read-in-words"),
        # The same pins compiled by the planner on those rows, entries from a choice. They are the
        # planner's, but both rows read one DAC1 part, so counting the multilateral row
        # as bilateral would add the bilateral series to itself. Each keeps its row.
        pytest.param("dac_countries_multilateral", False, None, id="rows-the-planner-compiled"),
        # A total Search counts by default beside the bilateral row the planner compiled.
        # One part, so the total keeps its scope, which already holds the bilateral
        # flows, and the sum is refused (`refused`, the option the first input is on).
        pytest.param(
            "dac_countries_contributions", False, "provider:bilateral", id="total-by-default"
        ),
    ],
)
def test_a_sum_of_provider_rows_searched_in_words_keeps_each_row(
    decision: DecisionFactory, second: str, stated: bool, refused: str | None
) -> None:
    """ "US bilateral plus multilateral ODA": the planner searched each input, and
    Search read DAC1 bilateral ODA and multilateral ODA in the words it searched. The
    sum keeps both pins, and is served, since bilateral flows and core contributions
    count no money twice. Alignment never puts two rows of one part on one row, so
    rows the planner compiled keep theirs too, and a total beside the bilateral row
    keeps its own scope and is refused as counting the bilateral flows twice."""
    blocks = {
        _TOTAL: _searched_in_words(_SUM_BLOCKS["dac_countries_bilateral"], stated=stated),
        _PART: _searched_in_words(_SUM_BLOCKS[second], stated=stated),
    }
    ctx = _FakeToolContext()
    _show_blocks(ctx, *blocks.values())

    result = _tools_by_name()["propose_plan"].func(
        question="How much US bilateral plus multilateral ODA was there in 2019?",
        answer_mode=AnswerMode.COMPUTED,
        decisions=drafts([decision()]),
        evidence_requirements=[
            *(_dac_countries(rid, block) for rid, block in blocks.items()),
            SumOfRequirement(
                id=_SUM,
                decision_ids=["measure"],
                left=RequirementSide(input_id=_TOTAL),
                right=RequirementSide(input_id=_PART),
            ),
        ],
        tool_context=ctx,
    )

    if refused is not None:
        assert result["code"] == ProblemCode.OPERATION_UNSUPPORTED.value, result
        assert result["detail"]["basis"] == {
            "spec_id": blocks[_PART]["spec_id"],
            "option_ids": [refused],
            "requirement_id": _PART,
            "step_id": _SUM,
        }
        assert load_plan(ctx.state) is None
        return
    assert result["outcome"] == Outcome.OK.value, result
    assert _pins(ctx) == [(rid, block["spec_id"]) for rid, block in blocks.items()]


def _another_part(block: dict[str, Any]) -> dict[str, Any]:
    """A copy of `block` as a part of its own (`PartScope.base`), under a spec_id of its
    own, as another donor's DAC1 row reads."""
    part = block["resolved"]["scope_choices"]["base"]["spec_id"]
    text = json.dumps(block)
    for held, other in ((part, f"spec_{'6' * 32}"), (block["spec_id"], f"spec_{'5' * 32}")):
        text = text.replace(held, other)
    return json.loads(text)


@pytest.mark.parametrize(
    ("stated", "realigned"),
    [
        pytest.param(True, None, id="rows-read-in-words"),
        # The same rows compiled by the planner, no one's statement, so the second moves
        # to the first input's scope.
        pytest.param(False, "provider:bilateral", id="rows-the-planner-compiled"),
    ],
)
def test_provider_rows_of_two_parts_read_in_words_keep_their_own(
    decision: DecisionFactory, stated: bool, realigned: str | None
) -> None:
    """ "US bilateral ODA plus France's multilateral ODA": two parts, so neither row
    would read the other's series. Search read each row in the words the planner
    searched, so each is stated and the sum keeps both pins, bilateral flows and core
    contributions counting no money twice."""
    blocks = {
        _TOTAL: _searched_in_words(_SUM_BLOCKS["dac_countries_bilateral"], stated=stated),
        _PART: _another_part(
            _searched_in_words(_SUM_BLOCKS["dac_countries_multilateral"], stated=stated)
        ),
    }
    ctx = _FakeToolContext()
    _show_blocks(ctx, *blocks.values())

    result = _tools_by_name()["propose_plan"].func(
        question="How much US bilateral ODA plus France's multilateral ODA was there in 2019?",
        answer_mode=AnswerMode.COMPUTED,
        decisions=drafts([decision()]),
        evidence_requirements=[
            *(_dac_countries(rid, block) for rid, block in blocks.items()),
            SumOfRequirement(
                id=_SUM,
                decision_ids=["measure"],
                left=RequirementSide(input_id=_TOTAL),
                right=RequirementSide(input_id=_PART),
            ),
        ],
        tool_context=ctx,
    )

    if realigned is not None:
        assert result["code"] == ProblemCode.BASES_UNALIGNED.value, result
        assert [(r["spec_id"], r["option_ids"]) for r in result["detail"]["realign"]] == [
            (blocks[_PART]["spec_id"], [realigned])
        ]
        return
    assert result["outcome"] == Outcome.OK.value, result
    assert _pins(ctx) == [(rid, block["spec_id"]) for rid, block in blocks.items()]


def _share(numerator: str, denominator: str) -> ShareOfRequirement:
    """`numerator` as a share of `denominator`."""
    return ShareOfRequirement(
        id=_SHARE,
        decision_ids=["measure"],
        numerator=RequirementSide(input_id=numerator),
        denominator=RequirementSide(input_id=denominator),
    )


def _difference(minuend: str, subtrahend: str) -> ComparisonRequirement:
    """`minuend` less `subtrahend`."""
    return ComparisonRequirement(
        id=_COMPARE,
        decision_ids=["measure"],
        minuend=RequirementSide(input_id=minuend),
        subtrahend=RequirementSide(input_id=subtrahend),
    )


def _added(left: str, right: str) -> SumOfRequirement:
    """`left` plus `right`."""
    return SumOfRequirement(
        id=_SUM,
        decision_ids=["measure"],
        left=RequirementSide(input_id=left),
        right=RequirementSide(input_id=right),
    )


def _reduced(
    requirement_id: str,
    input_id: str,
    reduction: Reduction,
    group_by: tuple[GroupBy, ...] = (),
) -> AggregateRequirement:
    """`input_id` reduced by `reduction` over `group_by`."""
    return AggregateRequirement(
        id=requirement_id,
        decision_ids=["measure"],
        input_id=input_id,
        reduction=reduction,
        group_by=group_by,
    )


_US_MULTILATERAL = _searched_in_words(_SUM_BLOCKS["dac_countries_multilateral"])
_US_TOTAL = _searched_in_words(_SUM_BLOCKS["dac_countries_contributions"])


@pytest.mark.parametrize(
    ("blocks", "steps", "realigned", "caveated"),
    [
        # Multilateral ODA read in words over the DAC1 total Search counts by default.
        # One part, so the total counted as multilateral ODA would be the numerator's
        # series and the share 100%.
        pytest.param(
            {"first": _US_MULTILATERAL, "second": _US_TOTAL},
            [_share("first", "second")],
            None,
            [_SHARE],
            id="multilateral-over-total",
        ),
        # The total stated too ("US ODA"), so both rows are stated, on different scopes.
        pytest.param(
            {
                "first": _US_MULTILATERAL,
                "second": _searched_in_words(
                    _stated_provider(
                        json.loads(json.dumps(_SUM_BLOCKS["dac_countries_contributions"]))
                    )
                ),
            },
            [_share("first", "second")],
            None,
            [_SHARE],
            id="multilateral-over-stated-total",
        ),
        # The same two figures subtracted would give a difference of 0.
        pytest.param(
            {"first": _US_MULTILATERAL, "second": _US_TOTAL},
            [_difference("first", "second")],
            None,
            [_COMPARE],
            id="multilateral-minus-total",
        ),
        # US health aid the user asked for with imputed multilateral flows, less US
        # bilateral health aid. One CRS part, so counting the bilateral figure as
        # inclusive would subtract a figure from itself.
        pytest.param(
            {
                "first": _stated_provider(
                    json.loads(json.dumps(us_2023_block("health_inclusive")))
                ),
                "second": us_2023_block("health"),
            },
            [_difference("first", "second")],
            None,
            [_COMPARE],
            id="inclusive-minus-bilateral-health",
        ),
        # The share beside a comparison of the total with France's multilateral ODA.
        # The comparison alone would move the total onto the multilateral row, and the
        # share would read it twice. Neither moves it.
        pytest.param(
            {
                "first": _US_MULTILATERAL,
                "second": _US_TOTAL,
                "other": _another_part(_US_MULTILATERAL),
            },
            [_share("first", "second"), _difference("second", "other")],
            None,
            [_SHARE, _COMPARE],
            id="a-second-calculation-moves-the-total",
        ),
        # The share beside a comparison of the total with France's bilateral ODA, each
        # step order. The share wants the total on the multilateral row, the comparison
        # on the bilateral one, and the first would read one series twice. Neither
        # moves it, whichever step comes first.
        *(
            pytest.param(
                {
                    "first": _US_MULTILATERAL,
                    "second": _US_TOTAL,
                    "other": _another_part(
                        _searched_in_words(_SUM_BLOCKS["dac_countries_bilateral"])
                    ),
                },
                steps,
                None,
                [_SHARE, _COMPARE],
                id=f"two-calculations-want-the-total-on-two-rows-{order}",
            )
            for order, steps in (
                ("share-first", [_share("first", "second"), _difference("second", "other")]),
                ("comparison-first", [_difference("second", "other"), _share("first", "second")]),
            )
        ),
        # US bilateral plus multilateral ODA, rows the planner compiled, as a share of
        # France's multilateral ODA. The share alone would move the bilateral row onto
        # the multilateral one, and the sum would add one series to itself.
        pytest.param(
            {
                "first": _searched_in_words(_SUM_BLOCKS["dac_countries_bilateral"], stated=False),
                "second": _searched_in_words(
                    _SUM_BLOCKS["dac_countries_multilateral"], stated=False
                ),
                "other": _another_part(_US_MULTILATERAL),
            },
            [_added("first", "second"), _share(_SUM, "other")],
            None,
            [_SUM, _SHARE],
            id="a-share-of-a-sum-moves-a-sum-input",
        ),
        # Multilateral ODA summed by year over the DAC1 total. The aggregate only
        # reduces rows, so counting the total as multilateral ODA would still divide
        # one series by itself.
        pytest.param(
            {"first": _US_MULTILATERAL, "second": _US_TOTAL},
            [
                _reduced("by_year", "first", Reduction.SUM, (GroupBy.PERIOD,)),
                _share("by_year", "second"),
            ],
            None,
            [_SHARE],
            id="multilateral-by-year-over-total",
        ),
        # US bilateral health aid over the DAC1 total. Different parts, so the total
        # moves to the narrower part's scope.
        pytest.param(
            {
                "first": _searched_in_words(_SUM_BLOCKS["us_health_to_kenya"]),
                "second": _US_TOTAL,
            },
            [_share("first", "second")],
            [("second", "provider:bilateral")],
            [],
            id="health-over-total",
        ),
    ],
)
def test_two_sides_never_read_one_series(
    decision: DecisionFactory,
    blocks: dict[str, dict[str, Any]],
    steps: list[EvidenceRequirement],
    realigned: list[tuple[str, str]] | None,
    caveated: list[str],
) -> None:
    """Calculations over a figure on a provider row and one the planner left on
    another. Where the two read one part, no calculation counts one on the other's
    row, which would read one series on both sides of a share, comparison or sum, so
    every pin is kept and each calculation `caveated` states what its inputs count.
    Where they read different parts, the input is aligned on the other's scope
    (`realigned`, each input and the option it moves to)."""
    ctx = _FakeToolContext()
    _show_blocks(ctx, *blocks.values())

    result = _tools_by_name()["propose_plan"].func(
        question="What share of US ODA goes to multilaterals?",
        answer_mode=AnswerMode.COMPUTED,
        decisions=drafts([decision()]),
        evidence_requirements=[
            *(_dac_countries(rid, block) for rid, block in blocks.items()),
            *steps,
        ],
        tool_context=ctx,
    )

    if realigned is not None:
        assert result["code"] == ProblemCode.BASES_UNALIGNED.value, result
        assert [(r["spec_id"], r["option_ids"]) for r in result["detail"]["realign"]] == [
            (blocks[rid]["spec_id"], [option]) for rid, option in realigned
        ]
        return
    assert result["outcome"] == Outcome.OK.value, result
    assert _pins(ctx) == [(rid, block["spec_id"]) for rid, block in blocks.items()]
    plan = load_plan(ctx.state)
    assert plan is not None
    differ = {d.slot for d in plan.decisions if d.caveat == "providers_differ"}
    assert [
        step_id for step_id in caveated if differ & set(plan.requirement(step_id).decision_ids)
    ] == caveated


# The US donor total under a second spec_id, on `TOTAL`'s part, counted net by default.
_TOTAL_NET_DEFAULT = f"spec_{'7' * 32}"


def _recipients(requirement_id: str, input_id: str, *names: str) -> FilterRequirement:
    """`input_id`'s rows to the recipients `names`."""
    return FilterRequirement(
        id=requirement_id,
        decision_ids=["measure"],
        input_id=input_id,
        predicates=(EntityIn(names=names, role="recipient"),),
    )


_BY_PERIOD_AND_ENTITY = (GroupBy.PERIOD, GroupBy.ENTITY)


@pytest.mark.parametrize(
    ("steps", "realigned"),
    [
        # A total of gross disbursements less an average of net ones: the two sides
        # differ by their operation, so aligning the net input on gross reads no one
        # series twice, and the comparison counts both on gross.
        pytest.param(
            [
                _reduced("one", "first", Reduction.SUM),
                _reduced("two", "second", Reduction.MEAN),
                _difference("one", "two"),
            ],
            "measure_basis:gross",
            id="total-less-average",
        ),
        # A total less a total. Aligned, the sides would read one series and the
        # difference would be 0, so each keeps its own basis.
        pytest.param(
            [
                _reduced("one", "first", Reduction.SUM),
                _reduced("two", "second", Reduction.SUM),
                _difference("one", "two"),
            ],
            None,
            id="total-less-total",
        ),
        # The peak year's share of the total. The peak is a figure of its own, so
        # the net input is counted on gross.
        pytest.param(
            [
                _reduced("one", "first", Reduction.MAX),
                _reduced("two", "second", Reduction.SUM),
                _share("one", "two"),
            ],
            "measure_basis:gross",
            id="peak-year-over-total",
        ),
        # Aid to each recipient summed by year over all aid summed by year: the
        # denominator's sum folds Ghana and Kenya into one figure a year, so the two
        # sides read different data and the net input is counted on gross.
        pytest.param(
            [
                _reduced("one", "first", Reduction.SUM, _BY_PERIOD_AND_ENTITY),
                _reduced("two", "second", Reduction.SUM, (GroupBy.PERIOD,)),
                _share("one", "two"),
            ],
            "measure_basis:gross",
            id="each-recipient-over-recipients-summed",
        ),
        # Each recipient's share of the yearly sum over both, keyed by recipient.
        pytest.param(
            [
                _reduced("two", "second", Reduction.SUM, (GroupBy.PERIOD,)),
                ShareOfRequirement(
                    id=_SHARE,
                    decision_ids=["measure"],
                    numerator=RequirementSide(input_id="first"),
                    denominator=RequirementSide(input_id="two"),
                    axis="recipient",
                    denominator_scope=DenominatorScope.POOLED,
                ),
            ],
            "measure_basis:gross",
            id="keyed-share-of-recipients-summed",
        ),
        # The same totals grouped by their columns in another order.
        pytest.param(
            [
                _reduced("one", "first", Reduction.SUM, _BY_PERIOD_AND_ENTITY),
                _reduced("two", "second", Reduction.SUM, _BY_PERIOD_AND_ENTITY[::-1]),
                _difference("one", "two"),
            ],
            None,
            id="group-by-in-another-order",
        ),
        # The same recipients kept on each side, named in another order.
        pytest.param(
            [
                _recipients("one", "first", "Ghana", "Kenya"),
                _recipients("two", "second", "Kenya", "Ghana"),
                _difference("one", "two"),
            ],
            None,
            id="recipients-in-another-order",
        ),
        # A filter keeping every recipient its input names keeps every row, so the
        # filtered side reads what the bare side reads.
        pytest.param(
            [_recipients("two", "second", "Ghana", "Kenya"), _difference("first", "two")],
            None,
            id="filter-keeping-every-row",
        ),
        # A filter keeping one of the two recipients reads another series.
        pytest.param(
            [_recipients("two", "second", "Ghana"), _difference("first", "two")],
            "measure_basis:gross",
            id="filter-keeping-one-recipient",
        ),
    ],
)
def test_sides_read_one_series_only_through_one_derivation(
    decision: DecisionFactory, steps: list[EvidenceRequirement], realigned: str | None
) -> None:
    """Two inputs on one part, US aid to Ghana and Kenya, the first stated gross and
    the second left on net, each read through `steps` and compared. Alignment keeps
    two sides apart only where their whole derivations, from source to comparison,
    would read one series: every operation, parameter and entity selector alike, in
    any order that computes the same, and a filter keeping every row read as its
    input."""
    gross = basis_block(TOTAL, narrow=False, active="measure_basis:gross", provenance="explicit")
    net = json.loads(
        json.dumps(
            basis_block(_TOTAL_NET_DEFAULT, narrow=False, active="measure_basis:net")
        ).replace(f"{_TOTAL_NET_DEFAULT}_base", f"{TOTAL}_base")
    )
    ctx = _FakeToolContext()
    _show_blocks(ctx, gross, net)
    question = "How does US gross ODA over 2015-2023 compare with its yearly average?"
    ctx.state[ASKS_STATE_KEY] = [question]
    ctx.state[ASKS_ACROSS_YEARS_STATE_KEY] = [0]

    def dataset(requirement_id: str, pin: str) -> DatasetRequirement:
        return DatasetRequirement(
            id=requirement_id,
            decision_ids=["measure"],
            measure="US total ODA",
            entities=[
                Entity(name="United States", role="donor"),
                Entity(name="Ghana", role="recipient"),
                Entity(name="Kenya", role="recipient"),
            ],
            period=Period(start=2015, end=2023),
            source_spec_id=pin,
        )

    result = _tools_by_name()["propose_plan"].func(
        question=question,
        answer_mode=AnswerMode.COMPUTED,
        decisions=drafts([decision()]),
        answered=[AskAnswer(ask=0, outputs=(steps[-1].id,))],
        evidence_requirements=[
            dataset("first", TOTAL),
            dataset("second", _TOTAL_NET_DEFAULT),
            *steps,
        ],
        tool_context=ctx,
    )

    if realigned is not None:
        assert result.get("code") == ProblemCode.BASES_UNALIGNED.value, result
        assert [(r["spec_id"], r["option_ids"]) for r in result["detail"]["realign"]] == [
            (_TOTAL_NET_DEFAULT, [realigned])
        ]
        return
    assert result["outcome"] == Outcome.OK.value, result
    assert _pins(ctx) == [("first", TOTAL), ("second", _TOTAL_NET_DEFAULT)]
    # The comparison states that its inputs are counted on different measure bases.
    plan = load_plan(ctx.state)
    assert plan is not None
    differ = {d.slot for d in plan.decisions if d.caveat == "bases_differ"}
    assert differ & set(plan.requirement(steps[-1].id).decision_ids)


_US_DONOR = Entity(name="United States", role="donor")


@pytest.mark.parametrize(
    ("both_reads", "second_reads", "served_count"),
    [
        # 2015-2023 against 2019-2023 on one part reads two series.
        pytest.param({}, {"period": Period(start=2019, end=2023)}, 1, id="another-period"),
        # Aid to Ghana alone, one recipient of the two, reads another series.
        pytest.param(
            {},
            {"entities": [_US_DONOR, Entity(name="Ghana", role="recipient")]},
            1,
            id="another-recipient",
        ),
        # US aid to each recipient reads another frame than its total.
        pytest.param(
            {"entities": [_US_DONOR]}, {"across": ("recipient",)}, 1, id="across-recipients"
        ),
        # Aid by sector group reads another frame than the total.
        pytest.param(
            {},
            {"breakdown": Breakdown(dimension=BreakdownDimension.SECTOR, level=0)},
            1,
            id="by-sector",
        ),
        # A family read one series per member StatVar reads another frame than its
        # sum, the reading a family whose axes are not summable takes by default.
        pytest.param(
            {"stat_vars": StatVarReading.SUM},
            {"stat_vars": StatVarReading.SEPARATE},
            2,
            id="separate-stat-vars",
        ),
    ],
)
def test_one_part_over_other_years_or_entities_is_aligned(
    decision: DecisionFactory,
    both_reads: dict[str, Any],
    second_reads: dict[str, Any],
    served_count: int,
) -> None:
    """US aid to Ghana and Kenya pinned twice on one part, the first stated gross and
    the second left on net, compared, each read as `both_reads` states and the
    second as `second_reads` too, from a part serving `served_count` StatVars. Two
    pins of one part read one series only over one period and one set of entities,
    into one frame (`across`, `breakdown`, `stat_vars`), so a second input read over
    other years or other recipients, or into another frame, is aligned on gross.
    Read alike, the two keep their own bases
    (`test_sides_read_one_series_only_through_one_derivation`, total less total)."""
    gross, net = (
        json.loads(
            json.dumps(basis_block(spec_id, narrow=False, active=active, provenance=provenance))
            .replace(f"{spec_id}_base", f"{TOTAL}_base")
            .replace('"served_count": 1', f'"served_count": {served_count}')
        )
        for spec_id, active, provenance in (
            (TOTAL, "measure_basis:gross", "explicit"),
            (_TOTAL_NET_DEFAULT, "measure_basis:net", "default"),
        )
    )
    ctx = _FakeToolContext()
    _show_blocks(ctx, gross, net)
    question = "How does US gross ODA to Ghana and Kenya compare with net ODA?"
    ctx.state[ASKS_STATE_KEY] = [question]
    reads: dict[str, Any] = {
        "entities": [
            Entity(name="United States", role="donor"),
            Entity(name="Ghana", role="recipient"),
            Entity(name="Kenya", role="recipient"),
        ],
        "period": Period(start=2015, end=2023),
        **both_reads,
    }

    result = _tools_by_name()["propose_plan"].func(
        question=question,
        answer_mode=AnswerMode.COMPUTED,
        decisions=drafts([decision()]),
        answered=[AskAnswer(ask=0, outputs=(_COMPARE,))],
        evidence_requirements=[
            DatasetRequirement(
                id="first",
                decision_ids=["measure"],
                measure="US total ODA",
                source_spec_id=TOTAL,
                **reads,
            ),
            DatasetRequirement(
                id="second",
                decision_ids=["measure"],
                measure="US total ODA",
                source_spec_id=_TOTAL_NET_DEFAULT,
                **(reads | second_reads),
            ),
            _difference("first", "second"),
        ],
        tool_context=ctx,
    )

    assert result.get("code") == ProblemCode.BASES_UNALIGNED.value, result
    assert [(r["spec_id"], r["option_ids"]) for r in result["detail"]["realign"]] == [
        (_TOTAL_NET_DEFAULT, ["measure_basis:gross"])
    ]


@pytest.mark.parametrize(
    ("other_across", "realigned"),
    [
        # Both sums take their second leaf over one frame of one pin, so
        # aligning the net input on gross puts the two sides on one series.
        pytest.param((), False, id="one-frame"),
        # One takes US aid to each recipient, the other its total: the sides read
        # different series whatever the first leaves count, so the net input is
        # aligned on gross.
        pytest.param(("recipient",), True, id="another-frame"),
    ],
)
def test_leaves_with_no_choice_read_one_series_only_in_one_frame(
    decision: DecisionFactory, other_across: tuple[str, ...], realigned: bool
) -> None:
    """Two sums compared, each a US total on one part, the first stated gross and the
    second left on net, plus a figure from a dataset offering no measure basis choice,
    pinned to one spec_id on both. Two such leaves read one series only where
    they read their pin alike, so a second leaf read into another frame (`across`)
    lets the net input be aligned; read alike, each keeps its own basis."""
    gross, net = (
        json.loads(
            json.dumps(
                basis_block(spec_id, narrow=False, active=active, provenance=provenance)
            ).replace(f"{spec_id}_base", f"{TOTAL}_base")
        )
        for spec_id, active, provenance in (
            (TOTAL, "measure_basis:gross", "explicit"),
            (_TOTAL_NET_DEFAULT, "measure_basis:net", "default"),
        )
    )
    unchosen = f"spec_{'8' * 32}"
    plain = basis_block(unchosen, narrow=False, active="measure_basis:net")
    plain["resolved"] = {
        key: value for key, value in plain["resolved"].items() if key != "scope_choices"
    }
    ctx = _FakeToolContext()
    _show_blocks(ctx, gross, net, plain)
    question = "How does US gross ODA plus its loans compare with net ODA plus its loans?"
    ctx.state[ASKS_STATE_KEY] = [question]

    def dataset(requirement_id: str, pin: str, across: tuple[str, ...] = ()) -> DatasetRequirement:
        return DatasetRequirement(
            id=requirement_id,
            decision_ids=["measure"],
            measure="US total ODA",
            entities=[_US_DONOR],
            across=across,
            period=Period(start=2015, end=2023),
            source_spec_id=pin,
        )

    def plus(requirement_id: str, left: str, right: str) -> SumOfRequirement:
        return SumOfRequirement(
            id=requirement_id,
            decision_ids=["measure"],
            left=RequirementSide(input_id=left),
            right=RequirementSide(input_id=right),
        )

    result = _tools_by_name()["propose_plan"].func(
        question=question,
        answer_mode=AnswerMode.COMPUTED,
        decisions=drafts([decision()]),
        answered=[AskAnswer(ask=0, outputs=(_COMPARE,))],
        evidence_requirements=[
            dataset("first", TOTAL),
            dataset("second", _TOTAL_NET_DEFAULT),
            dataset("first_plus", unchosen),
            dataset("second_plus", unchosen, other_across),
            plus("first_sum", "first", "first_plus"),
            plus("second_sum", "second", "second_plus"),
            _difference("first_sum", "second_sum"),
        ],
        tool_context=ctx,
    )

    if realigned:
        assert result.get("code") == ProblemCode.BASES_UNALIGNED.value, result
        assert [(r["spec_id"], r["option_ids"]) for r in result["detail"]["realign"]] == [
            (_TOTAL_NET_DEFAULT, ["measure_basis:gross"])
        ]
        return
    assert result.get("outcome") == Outcome.OK.value, result
    assert _pins(ctx) == [
        ("first", TOTAL),
        ("second", _TOTAL_NET_DEFAULT),
        ("first_plus", unchosen),
        ("second_plus", unchosen),
    ]


def test_sides_on_one_series_before_alignment_move_together(decision: DecisionFactory) -> None:
    """US aid to Ghana and Kenya pinned twice on one part, both left on net, compared,
    and the second taken as a share of France's aid stated gross. The two sides of the
    comparison read one series before alignment, so moving both onto gross, as the
    share's group asks, puts them on no series they were not on, and both are moved."""
    first_net = f"spec_{'8' * 32}"
    france = f"spec_{'9' * 32}"
    blocks = [
        json.loads(
            json.dumps(basis_block(spec_id, narrow=False, active="measure_basis:net")).replace(
                f"{spec_id}_base", f"{TOTAL}_base"
            )
        )
        for spec_id in (first_net, _TOTAL_NET_DEFAULT)
    ]
    blocks.append(
        basis_block(france, narrow=False, active="measure_basis:gross", provenance="explicit")
    )
    ctx = _FakeToolContext()
    _show_blocks(ctx, *blocks)

    def dataset(requirement_id: str, pin: str) -> DatasetRequirement:
        return DatasetRequirement(
            id=requirement_id,
            decision_ids=["measure"],
            measure="total ODA",
            entities=[Entity(name="United States", role="donor")],
            period=Period(start=2015, end=2023),
            source_spec_id=pin,
        )

    result = _tools_by_name()["propose_plan"].func(
        question="How does US ODA compare with itself, and as a share of France's gross ODA?",
        answer_mode=AnswerMode.COMPUTED,
        decisions=drafts([decision()]),
        evidence_requirements=[
            dataset("first", first_net),
            dataset("second", _TOTAL_NET_DEFAULT),
            dataset("france", france),
            _difference("first", "second"),
            _share("second", "france"),
        ],
        tool_context=ctx,
    )

    assert result.get("code") == ProblemCode.BASES_UNALIGNED.value, result
    assert sorted((r["spec_id"], r["option_ids"]) for r in result["detail"]["realign"]) == [
        (_TOTAL_NET_DEFAULT, ["measure_basis:gross"]),
        (first_net, ["measure_basis:gross"]),
    ]


# --- a pin whose identity a search recorded, over the requirement's period ----------

# Two definitions of Guatemala's GDP, as data-mcp stored them. The first is a
# search over every year with data, the second the spec_id a later search
# minted for 1960-2024. They differ only in `requested_window`.
_GDP_ALL, _GDP_1960_2024 = json.loads(
    (Path(__file__).parents[1] / "fixtures" / "gdp_guatemala_definitions.json").read_text()
)


@pytest.fixture
def gdp_on_record() -> Iterator[None]:
    """`_GDP_ALL`'s identity on record, as `harness.studio` sets the lookup for a segment."""
    recorded = RecordedIdentity(identity=_GDP_ALL["identity"])
    token = current_recorded_identity.set({_GDP_ALL["spec_id"]: recorded}.get)
    yield
    current_recorded_identity.reset(token)


def _propose_gdp(
    tools: dict[str, FunctionTool],
    ctx: _FakeToolContext,
    decision: Decision,
    requirements: RequirementPair,
    period: Period,
    *,
    amends_version: int | None = None,
) -> dict[str, object]:
    """Propose `requirements` with the dataset requirement pinning `_GDP_ALL` over `period`."""
    dataset, derived = requirements
    pinned = dataset.model_copy(update={"source_spec_id": _GDP_ALL["spec_id"], "period": period})
    return tools["propose_plan"].func(
        question="q",
        answer_mode=AnswerMode.COMPUTED,
        decisions=drafts([decision]),
        evidence_requirements=[pinned, derived],
        amends_version=amends_version,
        tool_context=ctx,
    )


def _source_pin(plan: Plan | None) -> str | None:
    """The dataset requirement's pin."""
    assert plan is not None
    (dataset,) = [
        item for item in plan.evidence_requirements if isinstance(item, DatasetRequirement)
    ]
    return dataset.source_spec_id


@pytest.mark.usefixtures("gdp_on_record")
def test_propose_plan_pins_a_recorded_identity_over_the_requirements_period(
    decision: DecisionFactory, requirements: RequirementPair
) -> None:
    """An earlier turn's search over every year with data, pinned for 1960-2024, is
    pinned instead to its identity over 1960-2024. That is the spec_id a fresh
    search would mint for it, so no search is needed to find it."""
    tools = _tools_by_name()
    ctx = _FakeToolContext()
    _show(ctx, _GDP_ALL["spec_id"], earlier_turn=True)

    result = _propose_gdp(tools, ctx, decision(), requirements, Period(start=1960, end=2024))

    assert result["outcome"] == Outcome.OK.value, result
    copied = _GDP_1960_2024["spec_id"]
    assert copied == "spec_25c18e1a4d45b9f8455974bf3f055da5"
    plan = load_plan(ctx.state)
    assert _source_pin(plan) == copied
    assert plan is not None
    assert plan.decisions == (decision(),)
    assert windowed_pins(ctx.state) == {
        copied: WindowedPin(
            spec_id=copied,
            identity=_GDP_1960_2024["identity"],
            windowed_from=_GDP_ALL["spec_id"],
        )
    }


@pytest.mark.usefixtures("gdp_on_record")
def test_an_amendment_reasons_about_a_recorded_pin_by_the_spec_id_shown(
    decision: DecisionFactory, requirements: RequirementPair
) -> None:
    """A patch that restates the pin of the spec_id the planner was shown moves it
    over the patch's period, and the copy over the old period is let go."""
    tools = _tools_by_name()
    ctx = _FakeToolContext()
    _show(ctx, _GDP_ALL["spec_id"], earlier_turn=True)
    first = _propose_gdp(tools, ctx, decision(), requirements, Period(start=1960, end=2024))
    assert first["outcome"] == Outcome.OK.value, first

    second = _propose_gdp(
        tools,
        ctx,
        decision(),
        requirements,
        Period(start=2000, end=2024),
        amends_version=int(first["version"]),
    )

    assert second["outcome"] == Outcome.OK.value, second
    copied, _ = windowed_identity(
        _GDP_ALL["identity"], {"kind": "range", "start_year": 2000, "end_year": 2024}
    )
    assert _source_pin(load_plan(ctx.state)) == copied
    assert list(windowed_pins(ctx.state)) == [copied]


def test_propose_plan_keeps_a_pin_with_no_identity_on_record(
    decision: DecisionFactory, requirements: RequirementPair
) -> None:
    """With nothing on record the pin stays as written, for the translator to search."""
    tools = _tools_by_name()
    ctx = _FakeToolContext()
    _show(ctx, _GDP_ALL["spec_id"], earlier_turn=True)

    result = _propose_gdp(tools, ctx, decision(), requirements, Period(start=1960, end=2024))

    assert result["outcome"] == Outcome.OK.value, result
    assert _source_pin(load_plan(ctx.state)) == _GDP_ALL["spec_id"]
    assert windowed_pins(ctx.state) == {}


def test_a_windowed_pin_stored_with_member_lists_is_left_out(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A session that crosses the release holds a copy with `stat_vars`, which
    `WindowedPin` no longer has. It is dropped, and the copy stored in the
    current shape beside it is kept."""
    kept = WindowedPin(
        spec_id=_GDP_1960_2024["spec_id"],
        identity=_GDP_1960_2024["identity"],
        windowed_from=_GDP_ALL["spec_id"],
    )
    stale = kept.model_dump(mode="json") | {"spec_id": "spec_" + "9" * 32, "stat_vars": ["GDP"]}
    state: dict[str, Any] = {WINDOWED_PINS_STATE_KEY: [stale, kept.model_dump(mode="json")]}

    with caplog.at_level(logging.WARNING, logger="planner.plan"):
        held = windowed_pins(state)

    assert held == {kept.spec_id: kept}
    (logged,) = caplog.records
    assert logged.levelno == logging.WARNING
    assert logged.args == (WINDOWED_PINS_STATE_KEY, 1, "stat_vars")


# --- Decisions as the model writes them ---------------------------------------------


def _period_decision(**fixes: object) -> dict[str, object]:
    """A `period` decision as a model writes it, with `fixes` as given."""
    return {
        "slot": "window",
        "description": "Which years to cover.",
        "options": ["2015 to 2024", "2010 to 2024"],
        "chosen": "2015 to 2024",
        "resolved_by": "planner_default",
        "rationale": "The years every source reports.",
        **fixes,
    }


def test_a_decision_naming_its_fields_records_the_plan_its_values_would(
    decision: DecisionFactory, requirements: RequirementPair
) -> None:
    """`fixes` names the fields; the values come off the requirements citing it.

    The recorded plan, its decisions' values and its revision hash are those
    of the plan stating the values on the decision.
    """
    dataset, derived = requirements
    assert isinstance(dataset, DatasetRequirement)
    cited = [dataset.model_copy(update={"decision_ids": ("measure", "window")}), derived]
    ctx = _FakeToolContext()

    result = _tools_by_name()["propose_plan"].func(
        question="q",
        answer_mode=AnswerMode.COMPUTED,
        decisions=[*drafts([decision()]), _period_decision(fixes=["period", "unit"])],
        evidence_requirements=cited,
        tool_context=ctx,
    )

    assert result["outcome"] == Outcome.OK.value
    recorded = load_plan(ctx.state)
    assert recorded is not None
    stated = Plan(
        question="q",
        answer_mode=AnswerMode.COMPUTED,
        decisions=[
            decision(),
            Decision.model_validate(
                _period_decision(fixes=DecisionFixes(period=dataset.period, unit=dataset.unit))
            ),
        ],
        evidence_requirements=cited,
    )
    assert recorded.decisions == stated.decisions
    assert plan_revision(recorded) == plan_revision(stated)
    # The plan record reads the period decision off `fixes`.
    (period,) = _periods(recorded)
    assert (period.decision, period.assumed) == ("window", True)


@pytest.mark.parametrize(
    ("fixes", "derived_cites", "unpriced_cites"),
    [
        (["reduction"], (), ()),
        (["reduction"], ("window",), ()),
        (["unit"], (), ("window",)),
    ],
    ids=["field_the_kind_lacks", "values_disagree", "partly_stated"],
)
def test_a_field_its_requirements_do_not_settle_is_left_off_the_decision(
    decision: DecisionFactory,
    requirement: RequirementFactory,
    requirements: RequirementPair,
    derived_requirement: DerivedRequirementFactory,
    fixes: list[str],
    derived_cites: tuple[str, ...],
    unpriced_cites: tuple[str, ...],
) -> None:
    """A named field no citing requirement states, or two state differently,
    has no one value, so the decision records without it and the plan lands."""
    dataset, derived = requirements
    ctx = _FakeToolContext()

    result = _tools_by_name()["propose_plan"].func(
        question="q",
        answer_mode=AnswerMode.SERIES,
        decisions=[*drafts([decision()]), _period_decision(fixes=fixes)],
        evidence_requirements=[
            dataset.model_copy(update={"decision_ids": ("measure", "window")}),
            derived.model_copy(update={"decision_ids": ("measure", *derived_cites)}),
            derived_requirement(
                id="second_totals",
                reduction=Reduction.MEAN,
                decision_ids=("measure", *derived_cites),
            ),
            requirement(id="unpriced", unit=None, decision_ids=("measure", *unpriced_cites)),
        ],
        tool_context=ctx,
    )

    assert result["outcome"] == Outcome.OK.value, result
    plan = load_plan(ctx.state)
    assert plan is not None
    assert next(d for d in plan.decisions if d.slot == "window").fixes is None


def test_a_decision_naming_the_pin_is_refused_as_invalid_data(
    decision: DecisionFactory, requirements: RequirementPair
) -> None:
    """`source_spec_id` is code's to write, so a draft cannot name it."""
    dataset, derived = requirements
    ctx = _FakeToolContext()

    result = _tools_by_name()["propose_plan"].func(
        question="q",
        answer_mode=AnswerMode.COMPUTED,
        decisions=[*drafts([decision()]), _period_decision(fixes=["source_spec_id"])],
        evidence_requirements=[dataset, derived],
        tool_context=ctx,
    )

    assert result["outcome"] == Outcome.INVALID_INPUT.value
    assert PLAN_STATE_KEY not in ctx.state


def test_an_amendment_moving_a_value_a_planner_decision_fixes_drops_it_and_rebuilds_a_restated_one(
    decision: DecisionFactory, requirements: RequirementPair
) -> None:
    """A patch moving a carried planner decision's period drops the decision
    with its citations; one the patch restates is rebuilt from the patched
    requirements."""
    dataset, derived = requirements
    cited = dataset.model_copy(update={"decision_ids": ("measure", "window")})
    moved = cited.model_copy(update={"period": Period(start=2010, end=2024)})
    tools = _tools_by_name()
    ctx = _FakeToolContext()
    drafted = _period_decision(fixes=["period"])
    first = tools["propose_plan"].func(
        question="q",
        answer_mode=AnswerMode.COMPUTED,
        decisions=[*drafts([decision()]), drafted],
        evidence_requirements=[cited, derived],
        tool_context=ctx,
    )
    assert first["outcome"] == Outcome.OK.value

    carried = tools["propose_plan"].func(
        evidence_requirements=[moved],
        amends_version=first["version"],
        tool_context=ctx,
    )
    assert carried["outcome"] == Outcome.OK.value, carried
    pruned = load_plan(ctx.state)
    assert pruned is not None
    assert [d.slot for d in pruned.decisions] == ["measure"]
    assert pruned.requirement(dataset.id).decision_ids == ("measure",)
    restated = tools["propose_plan"].func(
        decisions=[drafted],
        evidence_requirements=[moved],
        amends_version=carried["version"],
        tool_context=ctx,
    )

    assert restated["outcome"] == Outcome.OK.value, restated
    plan = load_plan(ctx.state)
    assert plan is not None
    window = next(d for d in plan.decisions if d.slot == "window")
    assert window.fixes == DecisionFixes(period=Period(start=2010, end=2024))


_DECIDED = Period(start=2015, end=2024)
_MOVED = Period(start=2010, end=2024)
_DROPPED = {"slot": "window", "chosen": "2015 to 2024", "removed": False}


def _period_moved(
    requirement_id: str | None = "annual_debt_service", stated: object = _MOVED
) -> dict[str, object]:
    return {
        "slot": "window",
        "field": "period",
        "decided": _DECIDED.model_dump(mode="json"),
        "stated": to_jsonable_python(stated),
        "requirement_id": requirement_id,
    }


def _recorded(ctx: _FakeToolContext, answer: dict[str, object]) -> None:
    """Append the response that recorded a plan, as the session holds it."""
    ctx.session.events = [*ctx.session.events, response_event("propose_plan", answer)]


@pytest.mark.parametrize(
    ("route", "code", "detail"),
    [
        ("still_cited", ProblemCode.USER_DECISION_CONTRADICTED, _period_moved()),
        ("uncited", ProblemCode.USER_DECISION_DROPPED, _DROPPED),
        ("renamed", ProblemCode.USER_DECISION_DROPPED, _DROPPED),
        ("removed", ProblemCode.USER_DECISION_DROPPED, {**_DROPPED, "removed": True}),
        ("restated_same_text", ProblemCode.USER_DECISION_CONTRADICTED, _period_moved()),
        (
            "restated_without_fixes",
            ProblemCode.USER_DECISION_CONTRADICTED,
            _period_moved(stated=None),
        ),
        (
            "restated_as_planner_same_text",
            ProblemCode.USER_DECISION_CONTRADICTED,
            {
                "slot": "window",
                "field": "resolved_by",
                "decided": "user",
                "stated": "planner_default",
                "requirement_id": None,
            },
        ),
        (
            "restated_as_planner",
            ProblemCode.USER_DECISION_CONTRADICTED,
            {
                "slot": "window",
                "field": "chosen",
                "decided": "2015 to 2024",
                "stated": "2010 to 2024",
                "requirement_id": None,
            },
        ),
    ],
)
def test_a_user_decision_stands_until_the_user_speaks_again(
    decision: DecisionFactory,
    requirements: RequirementPair,
    route: str,
    code: ProblemCode,
    detail: dict[str, object],
) -> None:
    """With no word from the user since the plan was recorded, a patch may not
    drop, uncite, remove or change their decision, however it goes about it.
    Once they answer, the same patch lands."""
    dataset, derived = requirements
    cited = dataset.model_copy(update={"decision_ids": ("measure", "window")})
    moved = cited.model_copy(update={"period": _MOVED})
    uncited = moved.model_copy(update={"decision_ids": ("measure",)})
    patches: dict[str, dict[str, object]] = {
        "still_cited": {"evidence_requirements": [moved]},
        "uncited": {"evidence_requirements": [uncited]},
        "renamed": {
            "evidence_requirements": [
                uncited.model_copy(update={"id": "debt_service"}),
                derived.model_copy(update={"input_id": "debt_service"}),
            ],
            "removed_evidence_requirements": [dataset.id],
        },
        "removed": {"evidence_requirements": [uncited], "removed_decisions": ["window"]},
        "restated_same_text": {
            "decisions": [_period_decision(fixes=["period"], resolved_by="user")],
            "evidence_requirements": [moved],
        },
        "restated_without_fixes": {
            "decisions": [_period_decision(resolved_by="user")],
            "evidence_requirements": [moved],
        },
        "restated_as_planner": {
            "decisions": [_period_decision(fixes=["period"], chosen="2010 to 2024")],
            "evidence_requirements": [moved],
        },
        "restated_as_planner_same_text": {
            "decisions": [_period_decision(fixes=["period"])],
            "evidence_requirements": [cited],
        },
    }
    tools = _tools_by_name()
    ctx = _FakeToolContext()
    first = tools["propose_plan"].func(
        question="q",
        answer_mode=AnswerMode.COMPUTED,
        decisions=[*drafts([decision()]), _period_decision(fixes=["period"], resolved_by="user")],
        evidence_requirements=[cited, derived],
        tool_context=ctx,
    )
    assert first["outcome"] == Outcome.OK.value
    _recorded(ctx, first)

    refused = tools["propose_plan"].func(
        **patches[route], amends_version=first["version"], tool_context=ctx
    )

    assert refused["outcome"] == Outcome.REFUSED.value, refused
    assert refused["code"] == code.value
    assert refused["detail"] == detail
    assert refused["action"] == {"call": "adk_request_input"}
    kept = load_plan(ctx.state)
    assert kept is not None
    assert kept.version == first["version"]

    ctx.session.events = [*ctx.session.events, *_amendment_events("cover 2010 to 2024")]
    landed = tools["propose_plan"].func(
        **patches[route], amends_version=first["version"], tool_context=ctx
    )
    assert landed["outcome"] == Outcome.OK.value, landed


def _typed(text: str, *, branch: str | None = None) -> Event:
    """A user-role text event: the user's own message, or the planner seam's under `branch`."""
    return Event(
        author="user",
        branch=branch,
        content=types.Content(role="user", parts=[types.Part.from_text(text=text)]),
    )


def test_a_users_choice_of_one_figure_over_every_row_stands_against_a_regrouping(
    decision: DecisionFactory, requirements: RequirementPair
) -> None:
    """The user chose one total over every row of the year read. The decision
    records the empty group, so a patch grouping the total by entity
    contradicts it while the user has said nothing since."""
    dataset, derived = requirements
    one_year = dataset.model_copy(update={"period": Period(start=2024, end=2024)})
    one_figure = derived.model_copy(
        update={"decision_ids": ("measure", "grouping"), "group_by": ()}
    )
    grouping = {
        "slot": "grouping",
        "description": "Whether the total is one figure or one per year.",
        "options": ["one figure", "one per year"],
        "chosen": "one figure",
        "resolved_by": "user",
        "rationale": "The user asked for one figure.",
        "fixes": ["group_by"],
    }
    tools = _tools_by_name()
    ctx = _FakeToolContext()
    first = tools["propose_plan"].func(
        question="q",
        answer_mode=AnswerMode.COMPUTED,
        decisions=[*drafts([decision()]), grouping],
        evidence_requirements=[one_year, one_figure],
        tool_context=ctx,
    )
    assert first["outcome"] == Outcome.OK.value, first
    recorded = load_plan(ctx.state)
    assert recorded is not None
    (made,) = [d for d in recorded.decisions if d.slot == "grouping"]
    assert made.fixes == DecisionFixes(group_by=())
    _recorded(ctx, first)

    regrouped = tools["propose_plan"].func(
        evidence_requirements=[one_figure.model_copy(update={"group_by": (GroupBy.ENTITY,)})],
        amends_version=first["version"],
        tool_context=ctx,
    )

    assert regrouped["code"] == ProblemCode.USER_DECISION_CONTRADICTED.value, regrouped
    assert {key: regrouped["detail"][key] for key in ("field", "decided", "stated")} == {
        "field": "group_by",
        "decided": [],
        "stated": ["entity"],
    }


@pytest.mark.parametrize(
    ("recorded", "after", "lands"),
    [
        (True, lambda version: [_typed("Actually, cover 2010 to 2024.")], True),
        (True, lambda version: [_typed("Call propose_plan again.", branch=_PLANNER_BRANCH)], False),
        (
            True,
            lambda version: [
                *_amendment_events("cover 2010 to 2024"),
                response_event("propose_plan", {"outcome": "ok", "version": version}),
            ],
            True,
        ),
        (False, lambda version: [], True),
    ],
    ids=["user_message", "seam_message", "amendment_then_retry_echo", "not_recorded_here"],
)
def test_what_counts_as_the_user_speaking_after_the_plan_was_recorded(
    decision: DecisionFactory,
    requirements: RequirementPair,
    recorded: bool,
    after: Callable[[int], list[Any]],
    lands: bool,
) -> None:
    """The user's own message counts and the planner seam's does not; a bare
    retry's answer repeating the version moves no mark; and a plan this
    session never recorded was recorded before the turn's message."""
    dataset, derived = requirements
    cited = dataset.model_copy(update={"decision_ids": ("measure", "window")})
    tools = _tools_by_name()
    ctx = _FakeToolContext()
    first = tools["propose_plan"].func(
        question="q",
        answer_mode=AnswerMode.COMPUTED,
        decisions=[*drafts([decision()]), _period_decision(fixes=["period"], resolved_by="user")],
        evidence_requirements=[cited, derived],
        tool_context=ctx,
    )
    assert first["outcome"] == Outcome.OK.value
    if recorded:
        _recorded(ctx, first)
    ctx.session.events = [*ctx.session.events, *after(first["version"])]

    result = tools["propose_plan"].func(
        evidence_requirements=[cited.model_copy(update={"period": _MOVED})],
        amends_version=first["version"],
        tool_context=ctx,
    )

    expected = Outcome.OK.value if lands else ProblemCode.USER_DECISION_CONTRADICTED.value
    assert result.get("code", result["outcome"]) == expected, result


def test_a_user_decision_the_user_s_answer_supersedes_goes_with_it(
    decision: DecisionFactory, requirements: RequirementPair
) -> None:
    """The user picks gross, then answers a later ask with net: the re-pin
    drops the user's gross decision with no edit of `decision_ids`."""
    tools = _tools_by_name()
    ctx = _FakeToolContext()
    _show_block(ctx, _FRANCE_KENYA)
    dataset, derived = requirements
    ctx.session.events = _measure_ask_events(_GROSS)
    first = tools["propose_plan"].func(
        question="q",
        answer_mode=AnswerMode.COMPUTED,
        decisions=drafts([decision()]),
        evidence_requirements=[dataset.model_copy(update={"source_spec_id": _GROSS}), derived],
        tool_context=ctx,
    )
    assert first["outcome"] == Outcome.OK.value, first
    _recorded(ctx, first)
    ctx.session.events = [*ctx.session.events, *_measure_ask_events(_NET)]

    second = tools["propose_plan"].func(
        evidence_requirements=[dataset.model_copy(update={"source_spec_id": _NET}), derived],
        amends_version=first["version"],
        tool_context=ctx,
    )

    assert second["outcome"] == Outcome.OK.value, second
    plan = load_plan(ctx.state)
    assert plan is not None
    assert [(d.slot, d.resolved_by) for d in plan.decisions] == [
        ("measure", ResolvedBy.PLANNER_DEFAULT),
        (search_pin_slot(_NET), ResolvedBy.USER),
    ]


@pytest.mark.parametrize("amended", [True, False], ids=["amended", "silent"])
def test_an_amendment_at_approval_may_re_pin_a_measure_the_user_picked(
    decision: DecisionFactory, requirements: RequirementPair, amended: bool
) -> None:
    """ "Use net disbursements" at approval re-pins the measure the user picked
    as gross, and the user's gross decision goes; with no word from the user,
    the same re-pin is refused."""
    tools = _tools_by_name()
    ctx = _FakeToolContext()
    _show_block(ctx, _FRANCE_KENYA)
    dataset, derived = requirements
    ctx.session.events = _measure_ask_events(_GROSS)
    first = tools["propose_plan"].func(
        question="q",
        answer_mode=AnswerMode.COMPUTED,
        decisions=drafts([decision()]),
        evidence_requirements=[dataset.model_copy(update={"source_spec_id": _GROSS}), derived],
        tool_context=ctx,
    )
    assert first["outcome"] == Outcome.OK.value, first
    _recorded(ctx, first)
    if amended:
        ctx.session.events = [*ctx.session.events, *_amendment_events("use net disbursements")]

    second = tools["propose_plan"].func(
        evidence_requirements=[dataset.model_copy(update={"source_spec_id": _NET}), derived],
        amends_version=first["version"],
        tool_context=ctx,
    )

    if not amended:
        assert second["code"] == ProblemCode.USER_DECISION_DROPPED.value, second
        return
    assert second["outcome"] == Outcome.OK.value, second
    plan = load_plan(ctx.state)
    assert plan is not None
    assert search_pin_slot(_GROSS) not in {d.slot for d in plan.decisions}


@pytest.mark.parametrize("answered", [True, False], ids=["answered", "silent"])
def test_the_user_may_drop_the_series_their_decision_fixed(
    decision: DecisionFactory, requirement: RequirementFactory, answered: bool
) -> None:
    """The user asks to drop the series their period decision fixed: removing
    it takes the decision with it. With no word from the user, it is refused."""
    debt = requirement(decision_ids=("measure", "window"))
    stock = requirement(id="debt_stock", measure="external debt stock")
    tools = _tools_by_name()
    ctx = _FakeToolContext()
    first = tools["propose_plan"].func(
        question="q",
        answer_mode=AnswerMode.SERIES,
        decisions=[*drafts([decision()]), _period_decision(fixes=["period"], resolved_by="user")],
        evidence_requirements=[debt, stock],
        tool_context=ctx,
    )
    assert first["outcome"] == Outcome.OK.value, first
    _recorded(ctx, first)
    if answered:
        ctx.session.events = [*ctx.session.events, *_amendment_events("drop debt service")]

    second = tools["propose_plan"].func(
        removed_evidence_requirements=[debt.id], amends_version=first["version"], tool_context=ctx
    )

    if not answered:
        assert second["code"] == ProblemCode.USER_DECISION_DROPPED.value, second
        return
    assert second["outcome"] == Outcome.OK.value, second
    plan = load_plan(ctx.state)
    assert plan is not None
    assert [d.slot for d in plan.decisions] == ["measure"]


def test_a_patch_moving_a_shared_planner_decision_s_value_is_refused_until_restated() -> None:
    """A planner decision fixing a period and a rank's `among` still backs the
    rank once a patch moves only the dataset's period, so dropping it is
    refused, naming what still rests on it; restating it lands."""
    flows = DatasetRequirement(
        id="flows",
        decision_ids=["scope"],
        measure="aid received",
        entities=[Entity(name="Kenya", role="recipient")],
        period=Period(start=2024, end=2024),
    )
    ranked = RankRequirement(
        id="ranked",
        decision_ids=["scope"],
        input_id="flows",
        direction=RankDirection.HIGHEST_FIRST,
        among=RankAmong.POPULATION,
    )
    scope = {
        "slot": "scope",
        "description": "Which year, and among whom.",
        "options": ["2024 among all", "2023 among all"],
        "chosen": "2024 among all",
        "resolved_by": "planner_default",
        "rationale": "The latest year, among every recipient.",
        "fixes": ["period", "among"],
    }
    moved = flows.model_copy(update={"period": Period(start=2023, end=2023)})
    tools = _tools_by_name()
    ctx = _FakeToolContext()
    first = tools["propose_plan"].func(
        question="q",
        answer_mode=AnswerMode.COMPUTED,
        decisions=[scope],
        evidence_requirements=[flows, ranked],
        tool_context=ctx,
    )
    assert first["outcome"] == Outcome.OK.value, first

    refused = tools["propose_plan"].func(
        evidence_requirements=[moved], amends_version=first["version"], tool_context=ctx
    )

    assert refused["code"] == ProblemCode.SHARED_DECISION_CONTRADICTED.value, refused
    assert refused["detail"] == {
        "slot": "scope",
        "field": "period",
        "decided": Period(start=2024, end=2024).model_dump(mode="json"),
        "stated": Period(start=2023, end=2023).model_dump(mode="json"),
        "requirement_id": "flows",
        "still_cited_by": ["ranked"],
    }
    assert refused["action"] == {"call": "propose_plan", "needs": ["decisions"]}
    kept = load_plan(ctx.state)
    assert kept is not None
    assert kept.version == first["version"]
    restated = tools["propose_plan"].func(
        decisions=[{**scope, "chosen": "2023 among all"}],
        evidence_requirements=[moved],
        amends_version=first["version"],
        tool_context=ctx,
    )
    assert restated["outcome"] == Outcome.OK.value, restated
    plan = load_plan(ctx.state)
    assert plan is not None
    assert plan.decisions[0].fixes == DecisionFixes(
        period=Period(start=2023, end=2023), among=RankAmong.POPULATION
    )


def test_the_card_after_an_amendment_shows_no_decision_the_plan_contradicts(
    decision: DecisionFactory, requirements: RequirementPair
) -> None:
    """The reproduction: the user chose 2015-2024 and the planner chose the
    unit. An amendment to 2010-2024 that resends no decision is refused, and
    one moving only the unit drops the planner's decision. After each, the
    decisions the card lists (`shown_decisions`) and the approval payload's
    `plan.proposed` frame agree with the recorded requirements."""
    dataset, derived = requirements
    assert isinstance(dataset, DatasetRequirement)
    cited = dataset.model_copy(update={"decision_ids": ("measure", "window", "currency")})
    currency = {
        "slot": "currency",
        "description": "Which unit to report in.",
        "options": ["current US dollars", "constant 2020 US dollars"],
        "chosen": "current US dollars",
        "resolved_by": "planner_default",
        "rationale": "The unit every source reports.",
        "fixes": ["unit"],
    }
    tools = _tools_by_name()
    ctx = _FakeToolContext()
    first = tools["propose_plan"].func(
        question="q",
        answer_mode=AnswerMode.COMPUTED,
        decisions=[
            *drafts([decision()]),
            _period_decision(fixes=["period"], resolved_by="user"),
            currency,
        ],
        evidence_requirements=[cited, derived],
        tool_context=ctx,
    )
    assert first["outcome"] == Outcome.OK.value
    _recorded(ctx, first)

    def shown() -> list[tuple[str, str, ResolvedBy]]:
        # The card is offered for a compiled plan, so the recorded one is
        # compiled first, as the translator would.
        plan = _translate(ctx)
        for requirement in plan.evidence_requirements:
            for slot in requirement.decision_ids:
                cited_decision = next(d for d in plan.decisions if d.slot == slot)
                assert fix_disagreements(requirement, cited_decision) == []
        card = [(d.slot, d.chosen, d.resolved_by) for d in shown_decisions(plan)]
        payload = {**plan.model_dump(mode="json"), "summary": "", "revision": plan_revision(plan)}
        (frame,) = _studio_translator()._interrupt_frames(APPROVE_INTERRUPT, {"payload": payload})
        wire = {"user": ResolvedBy.USER, "default": ResolvedBy.PLANNER_DEFAULT}
        assert [(d.id, d.chosen, wire[d.resolved_by]) for d in frame.decisions] == card
        return card

    refused = tools["propose_plan"].func(
        evidence_requirements=[cited.model_copy(update={"period": Period(start=2010, end=2024)})],
        amends_version=first["version"],
        tool_context=ctx,
    )
    assert refused["code"] == ProblemCode.USER_DECISION_CONTRADICTED.value, refused
    assert shown() == [
        ("measure", "annual debt service", ResolvedBy.PLANNER_DEFAULT),
        ("window", "2015 to 2024", ResolvedBy.USER),
        ("currency", "current US dollars", ResolvedBy.PLANNER_DEFAULT),
    ]

    repriced = tools["propose_plan"].func(
        evidence_requirements=[
            cited.model_copy(
                update={"unit": Unit(currency="USD", prices="constant", base_year=2020)}
            )
        ],
        amends_version=first["version"],
        tool_context=ctx,
    )

    assert repriced["outcome"] == Outcome.OK.value, repriced
    assert shown() == [
        ("measure", "annual debt service", ResolvedBy.PLANNER_DEFAULT),
        ("window", "2015 to 2024", ResolvedBy.USER),
    ]


def test_an_amendment_unpinning_a_settled_pin_drops_the_decision_written_for_it(
    decision: DecisionFactory, requirements: RequirementPair
) -> None:
    """The decision `propose_plan` wrote for a settled pin goes with the pin.

    A referral that finds no data for the pin has the model restate the
    requirement unpinned. The model never wrote the pin's decision and has
    no reason to remove it, so `propose_plan` drops it once nothing cites it.
    """
    tools = _tools_by_name()
    ctx = _FakeToolContext()
    _show(ctx, SPEC)
    dataset, derived = requirements
    first = _propose(
        tools, ctx, decision(), [dataset.model_copy(update={"source_spec_id": SPEC}), derived]
    )
    assert first["outcome"] == Outcome.OK.value

    second = tools["propose_plan"].func(
        evidence_requirements=[dataset],
        amends_version=first["version"],
        tool_context=ctx,
    )

    assert second["outcome"] == Outcome.OK.value, second
    plan = load_plan(ctx.state)
    assert plan is not None
    assert [d.slot for d in plan.decisions] == ["measure"]


@pytest.mark.parametrize(
    "shown",
    [{}, {POPULATION_SPEC: True}, {SPEC: False}],
    ids=["nothing_shown", "another_id_shown", "shown_as_a_candidate"],
)
def test_a_pin_no_search_settled_on_records_as_written(
    decision: DecisionFactory, requirements: RequirementPair, shown: dict[str, bool]
) -> None:
    """A pin of an id the planner was not shown, or was shown only as a
    candidate, needs no decision either; the translator confirms it."""
    tools = _tools_by_name()
    ctx = _FakeToolContext()
    for spec_id, settled in shown.items():
        _show(ctx, spec_id, settled=settled)
    dataset, derived = requirements
    pinned = dataset.model_copy(update={"source_spec_id": SPEC})

    result = tools["propose_plan"].func(
        question="q",
        answer_mode=AnswerMode.COMPUTED,
        decisions=drafts([decision()]),
        evidence_requirements=[pinned, derived],
        tool_context=ctx,
    )

    assert result["outcome"] == Outcome.OK.value, result
    plan = load_plan(ctx.state)
    assert plan is not None
    assert plan.decisions == (decision(),)
    assert plan.requirement(dataset.id) == pinned


# --- an election answering the graph's own `clarify` interrupt -------------------------


def test_propose_plan_accepts_a_user_election_from_the_graphs_clarify_interrupt(
    decision: DecisionFactory, requirements: RequirementPair
) -> None:
    """`harness.graph`'s `clarify` interrupt carries its candidates at
    `args["payload"]["candidates"]`, which is easy to confuse with
    `ask_user`'s own flat `args["candidates"]` -- `_offered_candidates`
    reads both."""
    tools = _tools_by_name()
    ctx = _FakeToolContext()
    ctx.session.events = [
        _clarify_ask_event(
            [{"spec_id": SPEC, "label": "a"}, {"spec_id": POPULATION_SPEC, "label": "b"}]
        ),
        _clarify_answer_event(_option(SPEC)),
    ]
    dataset, derived = requirements
    elected = decision(
        chosen="total debt stock",
        resolved_by=ResolvedBy.USER,
        fixes=DecisionFixes(source_spec_id=SPEC),
    )

    result = tools["propose_plan"].func(
        question="q",
        answer_mode=AnswerMode.COMPUTED,
        decisions=drafts([elected]),
        evidence_requirements=[dataset.model_copy(update={"source_spec_id": SPEC}), derived],
        tool_context=ctx,
    )

    assert result["outcome"] == Outcome.OK.value
    assert PLAN_STATE_KEY in ctx.state


def test_elected_spec_ids_pairs_a_response_with_the_call_that_precedes_it() -> None:
    """Every clarify round shares the fixed interrupt id `"clarify"`, so
    pairing a response with its offer means matching it to the call that
    immediately precedes it. Round one offers {SPEC, POPULATION_SPEC} and
    the user picks SPEC; round two, sharing the same id, re-offers a
    disjoint set that no longer includes SPEC. The round-one election
    resolves against round one's own offer regardless of round two's later,
    unrelated one.
    """
    other_spec = "spec_" + "d" * 32
    session = _FakeSession()
    session.events = [
        _clarify_ask_event(
            [{"spec_id": SPEC, "label": "a"}, {"spec_id": POPULATION_SPEC, "label": "b"}]
        ),
        _clarify_answer_event(_option(SPEC)),
        _clarify_ask_event([{"spec_id": other_spec, "label": "c"}]),
    ]
    tool_context = _FakeToolContext()
    tool_context.session = session

    assert planner_agent._elected_spec_ids(tool_context) == frozenset({SPEC})


# --- The declaration ADK builds from a union-typed parameter --------------------------


def test_propose_plans_declaration_describes_every_requirement_kind() -> None:
    # `evidence_requirements: list[EvidenceRequirement]` is a discriminated
    # union; ADK has to be able to build a `FunctionDeclaration` from it, or
    # the model never sees a schema to draft against. ADK keeps the union's
    # `Annotated` metadata, so the declaration is pydantic's own `oneOf` with
    # the `kind` discriminator (Gemini reads `oneOf` as `anyOf`).
    tools = _tools_by_name()

    declaration = tools["propose_plan"]._get_declaration()

    assert declaration is not None
    schema = declaration.parameters_json_schema
    assert schema is not None
    requirement_items = schema["properties"]["evidence_requirements"]["items"]
    assert requirement_items["discriminator"]["propertyName"] == "kind"
    kinds = {ref["$ref"].rsplit("/", 1)[-1] for ref in requirement_items["oneOf"]}
    assert kinds == {
        "DatasetRequirement",
        "AggregateRequirement",
        "GrowthRequirement",
        "ComparisonRequirement",
        "SumOfRequirement",
        "RankRequirement",
        "ShareOfRequirement",
        "QuotientRequirement",
        "JoinRequirement",
        "FilterRequirement",
    }


def _undescribed_properties(schema: dict[str, Any]) -> list[str]:
    """Return `"Model.field"` for every field of every `$defs` model with no `description`.

    Reads `$defs` rather than `model_fields`, since `$defs` is what a model
    the tool actually sees is built from -- ADK's legacy
    `_parse_schema_from_parameter` path drops a nested model's description
    even when `model_fields` carries one. A `$defs` entry with no
    `properties` is an enum, which carries nothing to describe here.
    """
    undescribed: list[str] = []
    for name, model_schema in schema.get("$defs", {}).items():
        properties = model_schema.get("properties")
        if not properties:
            continue
        for field_name, field_schema in properties.items():
            if not field_schema.get("description"):
                undescribed.append(f"{name}.{field_name}")
    return undescribed


def test_propose_plans_declaration_describes_every_field() -> None:
    # Every argument and every property of every model `propose_plan`'s
    # declaration carries has a description, so the model sees what each
    # field means without it having to live in `_PLANNING` prose.
    # `_undescribed_properties` walks `$defs`, the shape ADK actually sends
    # the model. The fields every requirement kind shares are described once,
    # on the `evidence_requirements` argument.
    tools = _tools_by_name()

    declaration = tools["propose_plan"]._get_declaration()

    assert declaration is not None
    schema = declaration.parameters_json_schema
    assert schema is not None
    assert [
        name for name, field in schema["properties"].items() if not field.get("description")
    ] == []
    assert sorted({entry.rsplit(".", 1)[1] for entry in _undescribed_properties(schema)}) == sorted(
        SHARED_FIELDS
    )
    described = schema["properties"]["evidence_requirements"]["description"]
    assert all(f"`{field}`" in described for field in SHARED_FIELDS)


def test_a_decline_no_answer_earns_says_how_it_is_earned() -> None:
    # An `ask_role_dropped` problem earns `breakdown_not_kept`, and no `unsupported`
    # answer does, so both places a model reads what earns a decline say so.
    declaration = _tools_by_name()["propose_plan"]._get_declaration()
    assert declaration is not None and declaration.parameters_json_schema is not None
    declined = declaration.parameters_json_schema["properties"]["declined"]["description"]
    code = Decline.model_json_schema()["properties"]["code"]["description"]

    for described in (declined, code):
        assert "`breakdown_not_kept`" in described
        assert "`ask_role_dropped`" in described


def _schema_nodes(node: Any) -> list[dict[str, Any]]:
    """Every schema object in `node`, a property map's names never read as keywords."""
    if isinstance(node, list):
        return [found for item in node for found in _schema_nodes(item)]
    if not isinstance(node, dict):
        return []
    found = [node]
    for key, value in node.items():
        named = key in ("properties", "$defs") and isinstance(value, dict)
        for child in value.values() if named else [value]:
            found.extend(_schema_nodes(child))
    return found


def test_propose_plans_declaration_states_nothing_its_keys_and_required_do_not() -> None:
    # A title repeats the key it sits under, and a `null` branch or a `null`
    # or `[]` default says what absence from `required` already says.
    schema = _arguments(_tools_by_name())
    declaration = _tools_by_name()["propose_plan"]._get_declaration()
    assert declaration is not None
    nodes = _schema_nodes(declaration.parameters_json_schema)

    assert [node for node in nodes if "title" in node] == []
    assert [node for node in nodes if {"type": "null"} in node.get("anyOf", [])] == []
    assert [node for node in nodes if "default" in node and node["default"] in (None, [])] == []
    assert [
        node for node in nodes if "const" in node and node.get("default") == node["const"]
    ] == []
    assert schema["question"]["type"] == "string"
    assert schema["amends_version"]["type"] == "integer"


def test_propose_plans_declaration_states_identifiers_and_decision_ids_once() -> None:
    # Every requirement kind carries an id and `decision_ids`, and a dozen
    # fields name a requirement or a decision. Each refers to one definition,
    # except a field with a description of its own, which holds a copy of
    # `Identifier` because Gemini reads nothing beside a `$ref`.
    declaration = _tools_by_name()["propose_plan"]._get_declaration()
    assert declaration is not None
    schema = declaration.parameters_json_schema
    assert schema is not None
    defs = schema["$defs"]
    identifier = TypeAdapter(Identifier).json_schema()

    patterned = [
        node for node in _schema_nodes(schema) if node.get("pattern") == identifier["pattern"]
    ]
    decision_ids = [
        model["properties"]["decision_ids"]
        for model in defs.values()
        if "decision_ids" in model.get("properties", {})
    ]

    copies = [node for node in patterned if node is not defs["Identifier"]]
    assert all(
        node.get("description")
        and {key: value for key, value in node.items() if key != "description"} == identifier
        for node in copies
    )
    assert defs["Identifier"] == identifier
    assert defs["DecisionIds"]["items"] == {"$ref": "#/$defs/Identifier"}
    assert len(decision_ids) == 10
    assert all(field == {"$ref": "#/$defs/DecisionIds"} for field in decision_ids)


def test_an_enum_the_field_describes_carries_no_description_of_its_own() -> None:
    # The field using each such enum names every value, so the model reads
    # each value's meaning once, where it writes it.
    declaration = _tools_by_name()["propose_plan"]._get_declaration()
    assert declaration is not None
    schema = declaration.parameters_json_schema
    assert schema is not None
    enums = (StatVarReading, RankAmong, DenominatorScope)
    assert {enum.__name__ for enum in enums} == FIELD_DESCRIBED_ENUMS

    for enum in enums:
        own = TypeAdapter(enum).json_schema()
        using = [node for node in _schema_nodes(schema) if node.get("enum") == own["enum"]]
        assert using, enum.__name__
        for field in using:
            assert field["description"] != own["description"]
            assert all(f"'{value}'" in field["description"] for value in own["enum"])


def test_compacting_copies_a_definition_wherever_a_reference_has_keys_beside_it() -> None:
    # Gemini reads nothing beside a `$ref`, so the field's own keys go on a
    # copy of the definition, and a definition nothing refers to is dropped.
    mode = {"$ref": "#/$defs/Mode", "description": "The mode.", "default": "a"}
    schema = {
        "$defs": {
            "Mode": {"enum": ["a", "b"], "type": "string", "description": "A mode."},
            "Kept": {"type": "integer"},
        },
        "properties": {"mode": mode, "kept": {"$ref": "#/$defs/Kept"}},
    }
    looped = {
        "$defs": {
            "Node": {"properties": {"parent": {"$ref": "#/$defs/Node", "description": "Up."}}}
        },
        "properties": {"root": {"$ref": "#/$defs/Node"}},
    }

    assert compact_schema(schema) == {
        "$defs": {"Kept": {"type": "integer"}},
        "properties": {
            "mode": {
                "enum": ["a", "b"],
                "type": "string",
                "description": "The mode.",
                "default": "a",
            },
            "kept": {"$ref": "#/$defs/Kept"},
        },
    }
    with pytest.raises(ValueError, match="Node"):
        compact_schema(looped)


def test_compacting_refuses_requirement_kinds_that_declare_decision_ids_differently() -> None:
    ids = {"type": "array", "items": {"type": "string"}}
    schema = {
        "$defs": {
            "One": {"properties": {"decision_ids": ids}},
            "Other": {"properties": {"decision_ids": ids | {"maxItems": 3}}},
        }
    }

    with pytest.raises(ValueError, match="decision_ids"):
        compact_schema(schema)


def test_propose_plan_still_accepts_an_explicit_null_its_declaration_omits(
    decision: DecisionFactory, requirements: RequirementPair
) -> None:
    # The declaration drops the `null` branch; the validator is the models',
    # so a call sending `null` for an optional field records as before.
    tools = _tools_by_name()
    ctx = _FakeToolContext()
    dataset, derived = (
        requirement.model_dump(mode="json", exclude_none=True) for requirement in requirements
    )
    dataset |= {"unit": None, "source": None, "stat_vars": None}

    result = tools["propose_plan"].func(
        tool_context=ctx,
        question="q",
        answer_mode=AnswerMode.COMPUTED,
        decisions=drafts([decision().model_dump(mode="json", exclude_none=True)]),
        evidence_requirements=[dataset, derived],
        amends_version=None,
    )

    assert result["outcome"] == Outcome.OK.value


def test_a_plan_tool_refuses_an_argument_it_does_not_describe() -> None:
    def tool(first: str, second: int = 0) -> None:
        """A tool."""
        del first, second

    with pytest.raises(ValueError, match="describe each argument exactly once"):
        PlanTool(tool, arguments={"first": "The first."})._get_declaration()


def test_a_plan_tool_refuses_to_reach_the_model_without_its_argument_descriptions() -> None:
    """With ADK's JSON schema declarations off, the descriptions have nowhere to go.

    Building the planner then raises, and so does a declaration built after
    the feature is turned off, so the planner never runs without the patch
    and version rules its arguments carry.
    """
    built = _tools_by_name()["propose_plan"]

    with temporary_feature_override(FeatureName.JSON_SCHEMA_FOR_FUNC_DECL, False):
        with pytest.raises(RuntimeError, match="JSON_SCHEMA_FOR_FUNC_DECL"):
            create_planner_agent(model_name="gemini-test")
        with pytest.raises(RuntimeError, match="JSON_SCHEMA_FOR_FUNC_DECL"):
            built._get_declaration()


# --- Rejecting and submitting ----------------------------------------------------------


def test_reject_plan_without_a_plan_returns_a_problem() -> None:
    tools = _tools_by_name()

    result = tools["reject_plan"].func(tool_context=_FakeToolContext())

    assert result["code"] == ProblemCode.PLAN_MISSING.value
    assert result["outcome"] == Outcome.STATE_MISSING.value


def test_reject_plan_marks_current_plan_rejected(
    decision: DecisionFactory, requirements: RequirementPair
) -> None:
    tools = _tools_by_name()
    ctx = _FakeToolContext()
    _propose(tools, ctx, decision(), requirements)

    result = tools["reject_plan"].func(tool_context=ctx)

    assert result["outcome"] == Outcome.OK.value
    assert result["status"] == PlanStatus.REJECTED.value


def test_declaring_a_plan_ready_twice_changes_nothing(
    decision: DecisionFactory, requirements: RequirementPair
) -> None:
    # Approval is the graph's business, not the planner's: `declare_ready`
    # calls no transition of its own, so declaring the same plan twice
    # answers the same summary and leaves the plan exactly where it was,
    # rather than refusing a second call the way an approval would.
    tools = _tools_by_name()
    ctx = _FakeToolContext()
    _propose(tools, ctx, decision(), requirements)
    _translate(ctx)

    first = _declare(ctx)
    second = _declare(ctx)

    assert first == second
    stored = load_plan(ctx.state)
    assert stored is not None
    assert stored.status == PlanStatus.PENDING_APPROVAL


# --- Instruction prose ---------------------------------------------------------------------


# The instruction the served planner carries: the translator attached, the
# skill tools built and the question held by the harness graph's seam.
_SERVED_INSTRUCTION = planner_agent.planner_instruction(
    has_translator=True, has_skills=True, question_fixed=True
)


def test_the_instruction_names_every_referral_kind_the_translator_can_send() -> None:
    """Every `ReferralKind` is named, by its own value, in the planner's instruction,
    unless every referral of it carries its own guidance as `repair` (`REPAIR_OF`).

    The planner reads a referral's `kind` verbatim off the function response,
    so a kind the instruction never names is one the model has to guess at.
    Describing a kind in prose alone does not satisfy this: the model
    matches on the string it is sent. A kind in `REPAIR_OF` is told in the
    referral itself, which `Referral` refuses to send without that guidance.
    """
    unnamed = [
        kind.value
        for kind in ReferralKind
        if kind.value not in _SERVED_INSTRUCTION and kind not in REPAIR_OF
    ]
    assert unnamed == [], (
        f"referral kinds the planner is never told how to read: {unnamed}. "
        "Name each one in `_TRANSLATING` and say what the planner does about it."
    )
    for kind in REPAIR_OF:
        valid = _referral_of(kind, "a")
        assert valid.repair == REPAIR_OF[kind]
        with pytest.raises(ValidationError, match="carries its own guidance as repair"):
            Referral.model_validate(valid.model_dump() | {"repair": None})


def test_the_instruction_has_the_planner_ask_before_proposing_over_an_unmeetable_referral() -> None:
    """`requirement_unmeetable` and `unsupported_ask` name a choice that is the user's, so
    the planner is told to say what cannot be done and ask before it proposes anything."""
    instruction = _SERVED_INSTRUCTION

    assert (
        "`unsupported_ask` or `requirement_unmeetable` referral means the requirement cannot be "
        "met as stated. Tell the user what cannot be done and ask, through `adk_request_input`, "
        "before you propose again"
    ) in instruction
    # The options an ask offers are ones discovery found, never ones the model
    # writes into the message itself, on the instruction and on the tool alike.
    grounded = (
        "come only from the referral's `candidates` or a refusal's `options`. With neither, "
        "the ask says what cannot be done and asks what to change"
    )
    assert grounded in instruction
    declaration = _tools_by_name()["adk_request_input"]._get_declaration()
    assert declaration is not None
    schema = declaration.parameters_json_schema
    assert schema is not None
    assert {"candidates", "options"} <= schema["properties"].keys()
    assert {"candidates", "options"}.isdisjoint(schema.get("required", []))


def test_across_states_a_party_nobody_named() -> None:
    # A stand-in such as "World" binds the identity to that one entity, so a
    # rank over it ranks one row; a share of the world total still needs one.
    # `across`'s own description carries the rule, and the instruction points
    # at it. Each column a result's `open_entity_axes` can carry is tied to
    # the party word the plan states.
    instruction = _SERVED_INSTRUCTION
    across = DatasetRequirement.model_fields["across"].description or ""

    assert "goes in `across` (see `across`)" in instruction
    assert "No stand-in entity such as 'World'" in across
    assert "A world figure set beside a named entity names 'World'" in across
    # The party word alone: a qualified population is not one an open axis holds.
    assert "each as the party word alone, such as 'donor', 'recipient' or 'country'" in across
    assert "is named itself in `entities` instead (see `members`)" in across
    # A set named alone is its own figure; its members are asked for by
    # `members`, with the set named once however many members it holds.
    members = Entity.model_fields["members"].description or ""
    assert "'African countries' is 'Africa' with `members` set" in members
    assert "A set is named once, never listed member by member" in members
    assert 'such as "Africa", is one entity with its own figure' in instruction
    assert "set once with `members` set, however many members it has" in instruction
    assert (
        "A result's open axes (`open_entity_axes`) are the columns left open: `origin` is the "
        "giving party (an entity `role` of donor, creditor or exporter), `destination` the "
        "receiving party (recipient, debtor or importer), `observationAbout` the place."
    ) in instruction


def test_breakdown_states_a_total_by_sector_as_well_as_a_ranking() -> None:
    # The instruction's breakdown case is a ranking; the field itself says a
    # plain total by sector splits by it too, so that question does not fall
    # back to one series per StatVar.
    breakdown = DatasetRequirement.model_fields["breakdown"].description or ""

    assert "a total by sector" in breakdown
    assert "a ranking of which sectors received the most" in breakdown


def test_the_summary_carries_nothing_the_plan_record_already_states(
    decision: DecisionFactory, requirements: RequirementPair
) -> None:
    """`declare_ready`'s summary is what a caller cannot read off the record.

    `harness.studio_translate` builds the user's plan card from the plan record
    through `harness.plan_display`, and translation reads the record out
    of session state, so a copy of the decisions, requirements, sources or
    steps here is a second description of one plan that can disagree with the
    first. Nothing on the served path reads a restated copy of the plan back
    to the user.
    """
    tools = _tools_by_name()
    ctx = _FakeToolContext()
    _propose(tools, ctx, decision(), requirements)
    _translate(ctx)

    result = _declare(ctx)

    assert set(cast("dict[str, Any]", result["summary"])) == {"version", "revision", "warnings"}


def test_the_instruction_asks_for_no_closing_text_after_submitting() -> None:
    # The plan card is on the user's screen and a compiled proposal ends the run, so
    # the instruction asks for no closing turn to present the plan.
    instruction = planner_instruction(has_translator=True)

    assert "none left out" not in instruction
    assert "each decision by its slot" not in instruction
    assert "approval in one" not in instruction


# --- Construction -------------------------------------------------------------------------


def test_create_planner_agent_wires_name_model_and_tools() -> None:
    agent = create_planner_agent(model_name="gemini-3.8-flash")

    assert isinstance(agent, LlmAgent)
    assert agent.name == "dfo_planner"
    assert isinstance(agent.model, Gemini)
    assert agent.model.model == "gemini-3.8-flash"
    assert {tool.name for tool in agent.tools} == {  # type: ignore[union-attr]
        "adk_request_input",
        "propose_plan",
        "reject_plan",
    }
    assert agent.sub_agents == []


def test_planner_instruction_treats_a_defaulted_time_window_as_settled() -> None:
    # `time_window_defaulted` is the resolve envelope's own warning code
    # (`harness.tool_contracts._wire_warnings`), and `window` is the
    # envelope's own field carrying the resolver's coverage
    # (`harness.tool_contracts.graph_envelope`). The instruction names
    # both verbatim, so the model can recognise them in the rendered
    # envelope.
    instruction = planner_instruction(has_translator=True)
    assert "time_window_defaulted" in instruction
    assert "`window`" in instruction


def test_the_entity_role_field_declares_the_party_words() -> None:
    # An entity's `role` reaches `search` as its role text, and `search`
    # drops a word outside its lexicon. The declared enum is the closed set
    # `propose_plan` holds a model to, each word on one side of the flow and
    # standing for itself.
    schema = Entity.model_json_schema()["properties"]["role"]
    assert schema["enum"] == list(PARTY_ROLES)
    for party in PARTY_ROLES:
        assert flow_side(party) is not None
        assert party_role(party) == party


def test_planner_instruction_carries_no_sentinel_range() -> None:
    # With no `window` on the envelope, the period is a decision like any
    # other, so "time window" stays in the opening decision list and the
    # instruction states no sentinel fallback range for it. "source" is not
    # in that same list any more (`_PLANNING`'s own opening enumeration):
    # unlike the window, a verbatim source pin needs no `Decision`, so the
    # instruction states its own carve-out for it separately, not grouped
    # with the window's.
    instruction = planner_instruction(has_translator=True)
    assert "1900 through 2100" not in instruction
    assert "1900" not in instruction
    assert "2100" not in instruction
    assert "scope, time window" in instruction
    assert "source, time window" not in instruction


def test_the_planner_thinks_at_low_and_carries_thought_summaries() -> None:
    """The planner runs every generation at LOW, with thoughts on for the trace.

    Measured directly against the real planner brief: LOW reproduces the
    unset default's validity outcome on both a well-formed and a malformed
    draft, at a fraction of the wall time and near-zero reasoning tokens --
    see `planner.settings.AGENT_THINKING_CONFIG`'s own comment.
    """
    agent = create_planner_agent(model_name="gemini-3.8-flash")

    assert isinstance(agent.planner, BuiltInPlanner)
    assert agent.planner.thinking_config.thinking_level is types.ThinkingLevel.LOW
    assert agent.planner.thinking_config.include_thoughts is True


def test_only_a_planner_with_a_translator_is_told_to_read_a_translation() -> None:
    # The standalone planner is what this package's own eval set scores, and
    # its `propose_plan` answers with no `translation`. Neither instruction
    # names a `translate` tool: translation is no tool of its own.
    standalone = create_planner_agent(model_name="m")
    paired = create_planner_agent(
        model_name="m", translate=build_translate(surface=FixtureSurface(schemas=[]))
    )

    assert "`translation`" not in str(standalone.instruction)
    assert "`translation`" in str(paired.instruction)
    assert "`translate`" not in str(standalone.instruction) + str(paired.instruction)


async def test_an_accepted_proposal_answers_with_the_translation_and_no_translate_tool_is_declared(
    decision: DecisionFactory, requirements: RequirementPair
) -> None:
    """A planner built with a translator declares no tool for it: `propose_plan`
    translates each plan it records, once, and answers with the verdict under
    `translation`. A proposal that records nothing translates nothing."""
    translated: list[int] = []

    async def translate(tool_context: Any) -> dict[str, object]:
        stored = load_plan(tool_context.state)
        assert stored is not None
        translated.append(stored.version)
        return {"outcome": Outcome.OK.value}

    agent = create_planner_agent(model_name="gemini-flash-latest", translate=translate)
    tools = {tool.name: tool for tool in agent.tools if isinstance(tool, FunctionTool)}
    ctx = _FakeToolContext()

    accepted = await _propose_translated(tools, ctx, decision(), requirements)
    refused = await _propose_translated(tools, ctx, decision(), requirements, amends_version=7)

    assert set(tools) == {"adk_request_input", "propose_plan", "reject_plan"}
    assert accepted["outcome"] == Outcome.OK.value
    assert accepted["translation"] == {"outcome": Outcome.OK.value}
    assert refused["outcome"] != Outcome.OK.value
    assert "translation" not in refused
    assert translated == [1]


async def test_a_referred_translation_leaves_the_proposed_version_recorded(
    decision: DecisionFactory, requirements: RequirementPair
) -> None:
    """A referral is the translation's verdict, not a refused proposal: the version
    is recorded and stays a draft, and the referral is pending on it."""
    dataset, _derived = requirements
    referral = _referral_of(ReferralKind.NO_DATA, dataset.id)

    async def translate(tool_context: Any) -> dict[str, object]:
        save_referral(tool_context.state, referral)
        return {"kind": referral.kind.value, "outcome": referral.outcome.value}

    tools = _tools_by_name(translate=translate)
    ctx = _FakeToolContext()

    result = await _propose_translated(tools, ctx, decision(), requirements)

    assert (result["outcome"], result["version"]) == (Outcome.OK.value, 1)
    assert result["translation"] == {"kind": "no_data", "outcome": "invalid_input"}
    stored = load_plan(ctx.state)
    assert stored is not None
    assert (stored.version, stored.status) == (1, PlanStatus.DRAFT)
    assert load_referral(ctx.state) == referral


@pytest.mark.parametrize(
    ("kind", "restated", "version"),
    [
        pytest.param(ReferralKind.SURFACE_UNAVAILABLE, {}, 1, id="transient-retried"),
        pytest.param(
            ReferralKind.DISCOVERY_UNAVAILABLE,
            {"question": "q", "answer_mode": AnswerMode.COMPUTED},
            1,
            id="transient-retried-restating-the-question",
        ),
        pytest.param(ReferralKind.NO_DATA, {}, 2, id="not-transient"),
        pytest.param(
            ReferralKind.SURFACE_UNAVAILABLE, {"question": "q2"}, 2, id="transient-with-a-change"
        ),
    ],
)
async def test_a_bare_reproposal_after_a_transient_verdict_spends_no_round(
    decision: DecisionFactory,
    requirements: RequirementPair,
    kind: ReferralKind,
    restated: dict[str, Any],
    version: int,
) -> None:
    """After a transient verdict, `propose_plan` with `amends_version` and no change
    translates the recorded version again: no new version, no plan round. After any
    other verdict, or with a change, the same call is an amendment like any other."""
    dataset, _derived = requirements
    verdicts = [_referral_of(kind, dataset.id), None]
    translated: list[int] = []

    async def translate(tool_context: Any) -> dict[str, object]:
        stored = load_plan(tool_context.state)
        assert stored is not None
        translated.append(stored.version)
        referral = verdicts.pop(0)
        if referral is None:
            return {"outcome": Outcome.OK.value}
        save_referral(tool_context.state, referral)
        return {"kind": referral.kind.value, "outcome": referral.outcome.value}

    tools = _tools_by_name(translate=translate)
    ctx = _FakeToolContext()
    await _propose_translated(tools, ctx, decision(), requirements)

    retried = await cast(
        "Awaitable[dict[str, object]]",
        tools["propose_plan"].func(amends_version=1, tool_context=ctx, **restated),
    )

    assert (retried["outcome"], retried["version"]) == (Outcome.OK.value, version)
    assert retried["translation"] == {"outcome": Outcome.OK.value}
    assert translated == [1, version]
    assert load_plan_rounds(ctx.state) == version
    assert load_referral(ctx.state) is None


def test_an_amendment_past_the_plan_size_limit_is_an_invalid_plan_that_costs_no_round() -> None:
    # The plan is one character under `MAX_PLAN_RECORD_CHARS` when it is first
    # proposed. The amendment lengthens one rationale past it. The model reads
    # the size, the limit and the largest parts, and its allowance is untouched.
    base = Plan.model_validate(plan_record(MAX_PLAN_RECORD_CHARS - 1))
    tools = _tools_by_name()
    ctx = _FakeToolContext()
    first = tools["propose_plan"].func(
        question=base.question,
        answer_mode=base.answer_mode,
        decisions=drafts(list(base.decisions)),
        evidence_requirements=list(base.evidence_requirements),
        tool_context=ctx,
    )
    assert first["outcome"] == Outcome.OK.value
    rounds = load_plan_rounds(ctx.state)

    longer = base.decisions[0].model_copy(update={"rationale": "r" * 10})
    refused = tools["propose_plan"].func(
        question=base.question,
        decisions=drafts([longer]),
        amends_version=1,
        tool_context=ctx,
    )

    assert refused["code"] == ProblemCode.INVALID_PLAN.value
    message = str(refused["message"])
    assert f"over the {MAX_PLAN_RECORD_CHARS:,} a plan may take" in message
    assert "requirement 'dataset_0'" in message
    assert load_plan_rounds(ctx.state) == rounds
    recorded = load_plan(ctx.state)
    assert recorded is not None
    assert recorded.version == 1


# --- A refusal names the field and omits its value ----------------------------------------

# One refused field is reported by its path, type, message and limit, which is a few
# hundred characters in the message and again in the detail.
ONE_REFUSED_FIELD_CHARS = 1_000
MAX_REPORTED_ERRORS = submission_service._MAX_REPORTED_ERRORS


def _wire_chars(problem: dict[str, object]) -> int:
    return len(json.dumps(problem))


def test_a_refused_field_is_named_without_repeating_its_value(
    decision: DecisionFactory, requirements: RequirementPair
) -> None:
    tools = _tools_by_name()
    ctx = _FakeToolContext()
    oversized = decision().model_dump(mode="json") | {"description": "d" * 300_000}

    refused = tools["propose_plan"].func(
        question="q",
        decisions=drafts([oversized]),
        evidence_requirements=list(requirements),
        tool_context=ctx,
    )

    assert refused["code"] == ProblemCode.INVALID_PLAN.value
    assert _wire_chars(refused) < ONE_REFUSED_FIELD_CHARS
    assert "d" * 20 not in json.dumps(refused)
    message = str(refused["message"])
    assert "decisions.0.description" in message
    assert f"at most {MAX_DECISION_DESCRIPTION_CHARS} characters" in message
    assert PLAN_STATE_KEY not in ctx.state


def test_a_refused_question_is_named_without_repeating_it(
    decision: DecisionFactory, requirements: RequirementPair
) -> None:
    tools = _tools_by_name()
    ctx = _FakeToolContext()

    refused = tools["propose_plan"].func(
        question="q" * 300_000,
        answer_mode=AnswerMode.COMPUTED,
        decisions=drafts([decision()]),
        evidence_requirements=list(requirements),
        tool_context=ctx,
    )

    assert refused["code"] == ProblemCode.INVALID_PLAN.value
    assert _wire_chars(refused) < ONE_REFUSED_FIELD_CHARS
    assert "q" * 20 not in json.dumps(refused)
    assert "question" in str(refused["message"])
    assert f"at most {MAX_QUESTION_CHARS} characters" in str(refused["message"])


def test_a_refusal_reports_the_first_failing_fields_and_counts_the_rest(
    decision: DecisionFactory, requirements: RequirementPair
) -> None:
    tools = _tools_by_name()
    ctx = _FakeToolContext()
    failing = [
        decision().model_dump(mode="json", exclude={"fixes"})
        | {"slot": f"slot_{index}", "description": "d" * 600}
        for index in range(MAX_PLAN_DECISIONS)
    ]

    refused = tools["propose_plan"].func(
        question="q",
        decisions=drafts(failing),
        evidence_requirements=list(requirements),
        tool_context=ctx,
    )

    reported = refused["detail"]["errors"]  # type: ignore[index]
    assert len(reported) == MAX_REPORTED_ERRORS
    message = str(refused["message"])
    assert f"decisions.{MAX_REPORTED_ERRORS - 1}.description" in message
    assert f"decisions.{MAX_REPORTED_ERRORS}.description" not in message
    assert f"and {MAX_PLAN_DECISIONS - MAX_REPORTED_ERRORS} more" in message
    assert _wire_chars(refused) < MAX_REPORTED_ERRORS * ONE_REFUSED_FIELD_CHARS


@pytest.mark.parametrize(
    ("argument", "cap"),
    [
        ("removed_decisions", MAX_PLAN_DECISIONS),
        ("removed_evidence_requirements", MAX_PLAN_REQUIREMENTS),
    ],
)
def test_a_removal_list_past_the_plan_cap_is_refused_naming_the_argument(
    decision: DecisionFactory, requirements: RequirementPair, argument: str, cap: int
) -> None:
    tools = _tools_by_name()
    ctx = _FakeToolContext()
    _propose(tools, ctx, decision(), requirements)

    refused = tools["propose_plan"].func(
        question="q",
        amends_version=1,
        tool_context=ctx,
        **{argument: [f"item_{index}" for index in range(cap + 1)]},
    )

    assert refused["code"] == ProblemCode.INVALID_PLAN.value
    assert argument in str(refused["message"])
    assert f"at most {cap} items" in str(refused["message"])
    assert _wire_chars(refused) < ONE_REFUSED_FIELD_CHARS


@pytest.mark.parametrize("argument", ["removed_decisions", "removed_evidence_requirements"])
@pytest.mark.parametrize("length", [65, 200_000])
def test_a_removal_that_cannot_be_an_identifier_is_refused_without_repeating_it(
    decision: DecisionFactory, requirements: RequirementPair, argument: str, length: int
) -> None:
    tools = _tools_by_name()
    ctx = _FakeToolContext()
    _propose(tools, ctx, decision(), requirements)

    refused = tools["propose_plan"].func(
        question="q",
        amends_version=1,
        tool_context=ctx,
        **{argument: ["x" * length]},
    )

    assert refused["code"] == ProblemCode.INVALID_PLAN.value
    assert f"{argument}.0" in str(refused["message"])
    assert _wire_chars(refused) < ONE_REFUSED_FIELD_CHARS
    assert "x" * 20 not in json.dumps(refused)
    recorded = load_plan(ctx.state)
    assert recorded is not None
    assert recorded.version == 1


# --- A calculation the system cannot do ------------------------------------


def _summed_join(requirement: RequirementFactory) -> list[EvidenceRequirement]:
    """Two series joined and then totalled: a sum over a join, which no plan can state."""
    return [
        requirement(id="bilateral"),
        requirement(id="multilateral", measure="imputed multilateral aid"),
        JoinRequirement(
            id="joined", decision_ids=["measure"], left_id="bilateral", right_id="multilateral"
        ),
        AggregateRequirement(
            id="total",
            decision_ids=["measure"],
            input_id="joined",
            reduction=Reduction.SUM,
            group_by=(GroupBy.PERIOD,),
        ),
    ]


def test_a_calculation_over_a_join_answers_unsupported(
    decision: DecisionFactory, requirement: RequirementFactory
) -> None:
    tools = _tools_by_name()
    ctx = _FakeToolContext()

    result = _propose(tools, ctx, decision(), _summed_join(requirement))

    assert result["outcome"] == Outcome.UNSUPPORTED.value
    assert result["code"] == ProblemCode.OPERATION_UNSUPPORTED.value
    assert result["detail"] == {"operation": UnsupportedOperation.CALCULATION_OVER_COMBINED.value}
    assert PLAN_STATE_KEY not in ctx.state


@pytest.mark.parametrize(
    ("declines", "after", "code"),
    [
        pytest.param(False, None, ProblemCode.DECLINE_MISSING, id="dropped-silently"),
        pytest.param(True, None, None, id="declined"),
        pytest.param(False, "resume", ProblemCode.DECLINE_MISSING, id="after-a-resume"),
        pytest.param(False, "new-entry", None, id="next-entry"),
        pytest.param(True, "new-entry", ProblemCode.DECLINE_UNEARNED, id="unearned-next-entry"),
    ],
)
def test_a_plan_after_an_unsupported_answer_has_to_decline_the_ask(
    decision: DecisionFactory,
    requirement: RequirementFactory,
    declines: bool,
    after: str | None,
    code: ProblemCode | None,
) -> None:
    """Once `propose_plan` answered `unsupported` in this planner entry, a plan that drops
    the ask without a `Decline` naming its operation is refused, on a resumed invocation
    too; a new entry starts with nothing owed, and a decline it adds is unearned."""
    tools = _tools_by_name()
    ctx = _FakeToolContext()
    _propose(tools, ctx, decision(), _summed_join(requirement))
    if after == "resume":
        ctx.invocation_id = "turn-1-resumed"
    elif after == "new-entry":
        clear_unsupported(ctx.state)
    declined = (
        [Decline(ask="the combined total", code=UnsupportedOperation.CALCULATION_OVER_COMBINED)]
        if declines
        else []
    )

    result = tools["propose_plan"].func(
        question="q",
        answer_mode=AnswerMode.SERIES,
        decisions=drafts([decision()]),
        evidence_requirements=_summed_join(requirement)[:2],
        declined=declined,
        tool_context=ctx,
    )

    if code is None:
        assert result["outcome"] == Outcome.OK.value, result
        recorded = load_plan(ctx.state)
        assert recorded is not None
        assert [d.model_dump(mode="json") for d in recorded.declined] == [
            d.model_dump(mode="json") for d in declined
        ]
    else:
        assert result["outcome"] == Outcome.INVALID_INPUT.value
        assert result["code"] == code.value
        assert result["detail"] == {
            "operations": [UnsupportedOperation.CALCULATION_OVER_COMBINED.value]
        }
        assert PLAN_STATE_KEY not in ctx.state


@pytest.mark.parametrize(
    "operation",
    [UnsupportedOperation.NO_OPERATION_COVERS, UnsupportedOperation.UNSUPPORTED_ASK],
)
@pytest.mark.parametrize("declines", [None, "whole-question", "naming-an-ask"])
def test_a_translate_unsupported_answer_has_to_be_declined_too(
    decision: DecisionFactory,
    requirements: RequirementPair,
    operation: UnsupportedOperation,
    declines: str | None,
) -> None:
    """A translation's `unsupported` referral kinds bind the next plan as `propose_plan`'s
    own `unsupported` answer does, and a `Decline` with the kind as its code settles it,
    whether or not it names an ask."""
    tools = _tools_by_name()
    ctx = _FakeToolContext()
    record_unsupported(ctx.state, operation)
    if declines == "naming-an-ask":
        ctx.state[ASKS_STATE_KEY] = ["the part asked"]
    declined = (
        [
            Decline(
                ask="the part asked",
                code=operation,
                ask_ref=0 if declines == "naming-an-ask" else None,
            )
        ]
        if declines
        else []
    )

    result = tools["propose_plan"].func(
        question="q",
        answer_mode=AnswerMode.COMPUTED,
        decisions=drafts([decision()]),
        evidence_requirements=list(requirements),
        declined=declined,
        tool_context=ctx,
    )

    if declines:
        assert result["outcome"] == Outcome.OK.value, result
    else:
        assert result["code"] == ProblemCode.DECLINE_MISSING.value
        assert result["detail"] == {"operations": [operation.value]}


def _drafted_arguments(decision: DecisionFactory, requirements: RequirementPair) -> dict[str, Any]:
    """A first draft's `propose_plan` arguments, as JSON a model sends."""
    return {
        "question": "q",
        "answer_mode": AnswerMode.COMPUTED.value,
        "decisions": [draft.model_dump(mode="json") for draft in drafts([decision()])],
        "evidence_requirements": [r.model_dump(mode="json") for r in requirements],
    }


async def _compiled_then_rejected(tool_context: Any) -> dict[str, object]:
    """An `ok` translation after which the plan on record is rejected, so
    `planner.agent.declare_ready`'s gate refuses it."""
    _translate(tool_context)
    plan = load_plan(tool_context.state)
    assert plan is not None
    save_plan(tool_context.state, plan.reject())
    return {"outcome": "ok"}


async def _never_translated(tool_context: Any) -> dict[str, object]:
    raise AssertionError("a refused proposal records nothing to translate")


async def test_a_refusal_action_states_the_decline_it_needs(
    decision: DecisionFactory, requirements: RequirementPair
) -> None:
    operation = next(iter(UnsupportedOperation))
    ctx = _FakeToolContext()
    record_unsupported(ctx.state, operation)
    tools = _tools_by_name(translate=_never_translated)

    answer = await tools["propose_plan"].run_async(
        args=_drafted_arguments(decision, requirements),
        tool_context=ctx,  # type: ignore[arg-type]
    )

    assert answer["code"] == ProblemCode.DECLINE_MISSING.value
    assert answer["action"] == {
        "call": "propose_plan",
        "arguments": {"declined": [{"code": operation.value}]},
        "needs": ["declined.ask"],
    }


async def test_a_gate_refusal_action_calls_propose_plan_with_nothing_stated(
    decision: DecisionFactory, requirements: RequirementPair
) -> None:
    tools = _tools_by_name(translate=_compiled_then_rejected)

    answer = await tools["propose_plan"].run_async(
        args=_drafted_arguments(decision, requirements),
        tool_context=_FakeToolContext(),  # type: ignore[arg-type]
    )

    assert "summary" not in answer
    assert answer["submission"]["code"] == ProblemCode.PLAN_REJECTED.value
    assert answer["submission"]["action"] == {"call": "propose_plan"}


@pytest.mark.parametrize(
    ("declines", "after", "code"),
    [
        pytest.param(True, None, None, id="declined"),
        pytest.param(False, None, None, id="pin-changed"),
        pytest.param(True, "resume", None, id="declined-after-a-resume"),
        pytest.param(True, "new-entry", ProblemCode.DECLINE_UNEARNED, id="unearned-next-entry"),
        pytest.param(True, "declined-alone", ProblemCode.INVALID_PLAN, id="declined-alone"),
    ],
)
def test_an_optional_decline_may_be_made_and_need_not_be(
    decision: DecisionFactory,
    requirements: RequirementPair,
    declines: bool,
    after: str | None,
    code: ProblemCode | None,
) -> None:
    """After a translation refers an irreconcilable pinned pair back, the next plan may
    decline `no_operation_covers` (the user wants the pair side by side) or leave it out
    (the user chose another source, so the pin changed); a plan declining only that, with
    no requirements, gets a refusal the planner can correct, because the pair it declines
    can still be shown; a new entry starts with nothing earned."""
    tools = _tools_by_name()
    ctx = _FakeToolContext()
    record_optional_decline(ctx.state, UnsupportedOperation.NO_OPERATION_COVERS)
    if after == "resume":
        ctx.invocation_id = "turn-1-resumed"
    elif after == "new-entry":
        clear_unsupported(ctx.state)
    declined = (
        [Decline(ask="the comparison", code=UnsupportedOperation.NO_OPERATION_COVERS)]
        if declines
        else []
    )

    result = tools["propose_plan"].func(
        question="q",
        answer_mode=AnswerMode.COMPUTED,
        decisions=drafts([decision()]) if after != "declined-alone" else [],
        evidence_requirements=list(requirements) if after != "declined-alone" else [],
        declined=declined,
        tool_context=ctx,
    )

    if code is None:
        assert result["outcome"] == Outcome.OK.value, result
    else:
        assert result["code"] == code.value
    if code is ProblemCode.DECLINE_UNEARNED:
        assert result["detail"] == {"operations": [UnsupportedOperation.NO_OPERATION_COVERS.value]}
    if code is ProblemCode.INVALID_PLAN:
        assert result["outcome"] == Outcome.INVALID_INPUT.value


_COMBINED_DECLINE = Decline(
    ask="the combined total", code=UnsupportedOperation.CALCULATION_OVER_COMBINED
)


@pytest.mark.parametrize(
    "restated",
    [
        pytest.param(None, id="left-out"),
        pytest.param(
            _COMBINED_DECLINE.model_copy(update={"ask": "the combined total since 2010"}),
            id="reworded",
        ),
    ],
)
def test_a_decline_is_carried_through_an_amendment(
    decision: DecisionFactory, requirement: RequirementFactory, restated: Decline | None
) -> None:
    """A decline earned in one planner entry is carried through an amendment in the next,
    which starts with no `unsupported` answer of its own, whether the amendment leaves
    it out or restates it in other words."""
    tools = _tools_by_name()
    ctx = _FakeToolContext()
    record_unsupported(ctx.state, _COMBINED_DECLINE.code)
    tools["propose_plan"].func(
        question="q",
        answer_mode=AnswerMode.SERIES,
        decisions=drafts([decision()]),
        evidence_requirements=_summed_join(requirement)[:2],
        declined=[_COMBINED_DECLINE],
        tool_context=ctx,
    )
    first = load_plan(ctx.state)
    assert first is not None

    clear_unsupported(ctx.state)

    amended = tools["propose_plan"].func(
        question="q",
        answer_mode=AnswerMode.SERIES,
        decisions=drafts([decision(rationale="Reconsidered.")]),
        evidence_requirements=[],
        declined=[restated] if restated is not None else [],
        amends_version=first.version,
        tool_context=ctx,
    )

    assert amended["outcome"] == Outcome.OK.value, amended
    recorded = load_plan(ctx.state)
    assert recorded is not None
    assert recorded.declined == (restated or _COMBINED_DECLINE,)


# --- The asks the brief lists ----------------------------------------------

_ASKS = ["aid from France to Guatemala", "the total of bilateral and multilateral aid"]
_ASK_DECLINE = Decline(ask=_ASKS[1], code=UnsupportedOperation.CALCULATION_OVER_COMBINED, ask_ref=1)


@pytest.mark.parametrize(
    ("asks", "answered", "declined", "code", "detail"),
    [
        pytest.param(
            _ASKS,
            [AskAnswer(ask=0, outputs=("joined",))],
            [],
            ProblemCode.ASKS_UNACCOUNTED,
            {"uncovered": [1], "repeated": []},
            id="uncovered",
        ),
        pytest.param(
            _ASKS,
            [
                AskAnswer(ask=0, outputs=("joined",)),
                AskAnswer(ask=1, outputs=("joined",)),
                AskAnswer(ask=1, outputs=("joined",)),
            ],
            [],
            ProblemCode.ASKS_UNACCOUNTED,
            {"uncovered": [], "repeated": [1]},
            id="answered-twice",
        ),
        pytest.param(
            _ASKS,
            [AskAnswer(ask=0, outputs=("joined",))],
            [
                _ASK_DECLINE,
                _ASK_DECLINE.model_copy(update={"code": UnsupportedOperation.NO_OPERATION_COVERS}),
            ],
            ProblemCode.ASKS_UNACCOUNTED,
            {"uncovered": [], "repeated": [1]},
            id="declined-twice",
        ),
        pytest.param(
            _ASKS,
            [AskAnswer(ask=0, outputs=("joined",)), AskAnswer(ask=1, outputs=("joined",))],
            [_ASK_DECLINE.model_copy(update={"ask": "a total over the two series combined"})],
            None,
            None,
            id="answered-in-part",
        ),
        pytest.param(
            _ASKS,
            [AskAnswer(ask=0, outputs=("joined",))],
            [_ASK_DECLINE],
            None,
            None,
            id="declined",
        ),
        pytest.param(
            _ASKS,
            [AskAnswer(ask=0, outputs=("joined",)), AskAnswer(ask=1, outputs=("total",))],
            [],
            ProblemCode.INVALID_PLAN,
            None,
            id="unknown-output",
        ),
        pytest.param(
            _ASKS,
            [AskAnswer(ask=0, outputs=("bilateral",)), AskAnswer(ask=1, outputs=("joined",))],
            [],
            None,
            None,
            id="output-another-reads",
        ),
        pytest.param(None, [], [], None, None, id="no-asks"),
        pytest.param(
            None,
            [AskAnswer(ask=0, outputs=("joined",))],
            [],
            ProblemCode.INVALID_PLAN,
            None,
            id="answer-without-asks",
        ),
    ],
)
def test_a_plan_accounts_for_each_ask_once(
    decision: DecisionFactory,
    requirement: RequirementFactory,
    asks: list[str] | None,
    answered: list[AskAnswer],
    declined: list[Decline],
    code: ProblemCode | None,
    detail: dict[str, list[int]] | None,
) -> None:
    """Each ask the state lists is answered once, declined once, or both when the plan
    computes part of it; with none listed, a plan records as it always has."""
    tools = _tools_by_name()
    ctx = _FakeToolContext()
    if asks is not None:
        ctx.state[ASKS_STATE_KEY] = asks
    for decline in declined:
        record_unsupported(ctx.state, decline.code)

    result = tools["propose_plan"].func(
        question="q",
        answer_mode=AnswerMode.SERIES,
        decisions=drafts([decision()]),
        evidence_requirements=_summed_join(requirement)[:3],
        answered=answered,
        declined=declined,
        tool_context=ctx,
    )

    if code is None:
        assert result["outcome"] == Outcome.OK.value, result
        recorded = load_plan(ctx.state)
        assert recorded is not None
        assert recorded.asks == tuple(asks or ())
        assert recorded.answered == tuple(answered)
        assert recorded.declined == tuple(declined)
        return
    assert result["outcome"] == Outcome.INVALID_INPUT.value
    assert result["code"] == code.value
    if detail is not None:
        assert result["detail"] == detail
        action = result.get("action")
        assert (action or {}).get("arguments") == (
            {"answered": [{"ask": number} for number in detail["uncovered"]]}
            if detail["uncovered"]
            else None
        )
    assert PLAN_STATE_KEY not in ctx.state


# One ask that breaks general government debt service down by creditor.
_CREDITOR_ASKS = ["Debt service by creditor for general government in 2031"]
_BY_CREDITOR = Decline(ask="by creditor", code=UnsupportedOperation.BREAKDOWN_NOT_KEPT, ask_ref=0)
_SPEC_BY_CREDITOR = "spec_" + "9" * 32
_SPEC_ODA_BY_DONOR = "spec_" + "8" * 32


_SPEC_DEBT_STOCK = "spec_" + "7" * 32
# A by-creditor debt source's open axes: its 134 debtors and its 97 creditors.
_DEBTORS_AND_CREDITORS: dict[str, int | None] = {"destination": 134, "origin": 97}


def _creditor_requirements(
    across: list[str], *, ranked: bool, pin: str | None = None
) -> list[EvidenceRequirement]:
    """Debt service across `across`, pinned to `pin`, ranked by creditor when `ranked`."""
    dataset = DatasetRequirement(
        id="debt_service",
        decision_ids=["measure"],
        measure="general government debt service",
        across=tuple(across),
        period=Period(start=2031, end=2031),
        source_spec_id=pin,
    )
    if not ranked:
        return [dataset]
    return [
        dataset,
        RankRequirement(
            id="by_creditor",
            decision_ids=["measure"],
            input_id="debt_service",
            direction=RankDirection.HIGHEST_FIRST,
            among=RankAmong.POPULATION,
            axis="creditor",
            reduction=Reduction.SUM,
        ),
    ]


# A settled source shown to the planner: its spec_id, the member count of each
# column it leaves open, and the parties the search that found it asked for.
type _Shown = tuple[str, Mapping[str, int | None], tuple[str, ...]]


def _creditor_context(
    *shown: _Shown,
    asks: Sequence[str] = _CREDITOR_ASKS,
    by: tuple[PartyRole, ...] = ("creditor",),
    **source_fields: Any,
) -> _FakeToolContext:
    """A planner entry under `asks`, the first broken down by `by` when it names a party,
    shown each source in `shown`: settled, with its block, unless `source_fields` says
    otherwise."""
    ctx = _FakeToolContext()
    ctx.state[ASKS_STATE_KEY] = list(asks)
    if by:
        ctx.state[ASK_BREAKDOWNS_STATE_KEY] = [AskBreakdown(ask=0, by=by).model_dump(mode="json")]
    save_pin_sources(
        ctx.state,
        {
            spec_id: PinSource(
                **{
                    "block": {
                        "resolved": {
                            "role_axis_facts": [
                                {"role": column, "member_count": count}
                                for column, count in columns.items()
                            ]
                        }
                    },
                    "settled": True,
                    "asked_roles": asked,
                    **source_fields,
                }
            )
            for spec_id, columns, asked in shown
        },
    )
    return ctx


def _propose_for_creditors(
    ctx: _FakeToolContext,
    decision: DecisionFactory,
    *,
    across: list[str],
    ranked: bool = False,
    pin: str | None = None,
    stock_pin: str | None = None,
    declined: Sequence[Decline] = (),
) -> dict[str, Any]:
    """Propose the debt-service plan answering ask 0; with `stock_pin`, a debt-stock
    dataset over the same `across`, pinned to it, answers ask 1."""
    requirements = _creditor_requirements(across, ranked=ranked, pin=pin)
    answered = [AskAnswer(ask=0, outputs=(requirements[-1].id,))]
    if stock_pin is not None:
        stock = requirements[0].model_copy(
            update={
                "id": "debt_stock",
                "measure": "external debt stock",
                "source_spec_id": stock_pin,
            }
        )
        requirements.append(stock)
        answered.append(AskAnswer(ask=1, outputs=(stock.id,)))
    return _tools_by_name()["propose_plan"].func(
        question="general government debt service by creditor for 2031",
        answer_mode=AnswerMode.SERIES,
        decisions=drafts(
            [decision(fixes=DecisionFixes(among=RankAmong.POPULATION, across=tuple(across)))]
        ),
        evidence_requirements=requirements,
        answered=answered,
        declined=list(declined),
        tool_context=ctx,
    )


def test_a_share_whose_whole_names_its_own_donor_is_refused_until_the_whole_spans_donors(
    decision: DecisionFactory,
) -> None:
    # Run B: the United States' ODA to Kenya over a total naming the United
    # States and Kenya alone divided the United States by itself, 99.7%.
    ctx = _FakeToolContext()
    ctx.state[ASKS_STATE_KEY] = ["The United States' share of ODA to Kenya, each year"]
    ctx.state[ASK_BREAKDOWNS_STATE_KEY] = [
        AskBreakdown(ask=0, over=("donor",)).model_dump(mode="json")
    ]
    kenya = Entity(name="Kenya", role="recipient")
    united_states = Entity(name="United States", role="donor")

    def propose(whole: DatasetRequirement) -> dict[str, Any]:
        period = Period(start=2015, end=2024)
        return _tools_by_name()["propose_plan"].func(
            question="What share of ODA to Kenya came from the United States each year?",
            answer_mode=AnswerMode.COMPUTED,
            decisions=drafts([decision()]),
            evidence_requirements=[
                DatasetRequirement(
                    id="part",
                    decision_ids=["measure"],
                    measure="ODA gross disbursements",
                    entities=(united_states, kenya),
                    period=period,
                ),
                whole.model_copy(update={"period": period}),
                AggregateRequirement(
                    id="whole_by_year",
                    decision_ids=["measure"],
                    input_id="whole",
                    reduction=Reduction.SUM,
                    group_by=(GroupBy.PERIOD,),
                ),
                ShareOfRequirement(
                    id="share",
                    decision_ids=["measure"],
                    numerator=RequirementSide(input_id="part"),
                    denominator=RequirementSide(input_id="whole_by_year"),
                ),
            ],
            answered=[AskAnswer(ask=0, outputs=("share",))],
            tool_context=ctx,
        )

    def whole(**fields: Any) -> DatasetRequirement:
        return DatasetRequirement(
            id="whole",
            decision_ids=["measure"],
            measure="total ODA gross disbursements",
            period=Period(start=2015, end=2024),
            **fields,
        )

    refused = propose(whole(entities=(kenya, united_states)))
    spanning = propose(whole(entities=(kenya,), across=("donor",)))

    assert refused["code"] == ProblemCode.WHOLE_IS_ITS_MEMBER.value, refused
    assert refused["detail"] == {
        "requirement_id": "share",
        "ask": 0,
        "party": "donor",
        "denominator_id": "whole",
        "named": ["United States"],
    }
    assert spanning["outcome"] == Outcome.OK.value, spanning


# The CRS and DAC2a pins a share's part and whole read, by spec_id.
_PART_PIN = "spec_" + "1" * 32
_WHOLE_PIN = "spec_" + "2" * 32

# A donor a side names, with the graph ref its pin's block states for it, or
# `None` when the block resolves no entity by that name.
type _Donor = tuple[str, str | None]


def _labelled_block(spec_id: str, donors: Sequence[_Donor]) -> dict[str, Any]:
    """A search block for `spec_id` resolving Kenya and each donor of `donors` with a ref."""
    entities = [{"role": "destination", "ref": "country/KEN", "label": "Kenya"}]
    entities += [
        {"role": "origin", "ref": ref, "label": name} for name, ref in donors if ref is not None
    ]
    return {"spec_id": spec_id, "resolved": {"entities": entities}}


@pytest.mark.parametrize(
    ("part", "whole", "pinned", "summed", "refused"),
    [
        # "United States" on CRS and "United States of America" on DAC2a are
        # one graph ref, so the whole names the part's member alone.
        pytest.param(
            [("United States", "country/USA")],
            [("United States of America", "country/USA")],
            True,
            False,
            True,
            id="alias-one-ref",
        ),
        # Two names alike in their words, two refs: the whole names another donor.
        pytest.param(
            [("Congo", "country/COG")],
            [("Congo, Dem. Rep.", "country/COD")],
            True,
            False,
            False,
            id="similar-names-two-refs",
        ),
        # Japan resolves on the part's pin alone, so it is matched by name,
        # and the alias beside it by ref.
        pytest.param(
            [("United States", "country/USA"), ("Japan", "country/JPN")],
            [("United States of America", "country/USA"), ("Japan", None)],
            True,
            False,
            True,
            id="mixed-pins",
        ),
        # The whole is read through an aggregate, whose dataset's pin resolves it.
        pytest.param(
            [("United States", "country/USA"), ("Japan", "country/JPN")],
            [("United States of America", "country/USA"), ("Japan", None)],
            True,
            True,
            True,
            id="mixed-pins-summed-whole",
        ),
        # No pin resolves a ref, so names decide: equal names are one member.
        pytest.param(
            [("United States", "country/USA")],
            [("United States", "country/USA")],
            False,
            False,
            True,
            id="no-refs-equal-names",
        ),
        pytest.param(
            [("United States", "country/USA")],
            [("United States of America", "country/USA")],
            False,
            False,
            False,
            id="no-refs-alias",
        ),
    ],
)
def test_a_share_compares_its_whole_to_its_part_by_ref_where_the_pins_resolve_one(
    decision: DecisionFactory,
    part: list[_Donor],
    whole: list[_Donor],
    pinned: bool,
    summed: bool,
    refused: bool,
) -> None:
    ctx = _FakeToolContext()
    ctx.state[ASKS_STATE_KEY] = ["The donors' share of ODA to Kenya, each year"]
    ctx.state[ASK_BREAKDOWNS_STATE_KEY] = [
        AskBreakdown(ask=0, over=("donor",)).model_dump(mode="json")
    ]
    if pinned:
        save_pin_sources(
            ctx.state,
            {
                spec_id: PinSource(block=_labelled_block(spec_id, donors))
                for spec_id, donors in ((_PART_PIN, part), (_WHOLE_PIN, whole))
            },
        )
    period = Period(start=2015, end=2024)

    def dataset(id: str, measure: str, donors: list[_Donor], pin: str) -> DatasetRequirement:
        return DatasetRequirement(
            id=id,
            decision_ids=["measure"],
            measure=measure,
            entities=(
                *(Entity(name=name, role="donor") for name, _ in donors),
                Entity(name="Kenya", role="recipient"),
            ),
            period=period,
            source_spec_id=pin if pinned else None,
        )

    by_year = AggregateRequirement(
        id="whole_by_year",
        decision_ids=["measure"],
        input_id="whole",
        reduction=Reduction.SUM,
        group_by=(GroupBy.PERIOD,),
    )
    result = _tools_by_name()["propose_plan"].func(
        question="What share of ODA to Kenya came from the donors each year?",
        answer_mode=AnswerMode.COMPUTED,
        decisions=drafts([decision()]),
        evidence_requirements=[
            dataset("part", "ODA gross disbursements", part, _PART_PIN),
            dataset("whole", "total ODA gross disbursements", whole, _WHOLE_PIN),
            *((by_year,) if summed else ()),
            ShareOfRequirement(
                id="share",
                decision_ids=["measure"],
                numerator=RequirementSide(input_id="part"),
                denominator=RequirementSide(input_id=by_year.id if summed else "whole"),
            ),
        ],
        answered=[AskAnswer(ask=0, outputs=("share",))],
        tool_context=ctx,
    )

    if refused:
        assert result["code"] == ProblemCode.WHOLE_IS_ITS_MEMBER.value, result
        assert result["detail"]["named"] == sorted(name for name, _ in whole)
    else:
        assert result["outcome"] == Outcome.OK.value, result


_G7_REFS = tuple(f"country/{code}" for code in ("CAN", "DEU", "FRA", "GBR", "ITA", "JPN", "USA"))


@pytest.mark.parametrize(
    ("members", "refused"),
    [
        # "G7" on CRS and "Group of Seven" on DAC2a bind the same seven donors,
        # so each donor is divided by the seven's total.
        pytest.param(_G7_REFS, False, id="pool-of-seven"),
        pytest.param(_G7_REFS[-1:], True, id="pool-of-one"),
    ],
)
def test_a_pooled_share_counts_the_members_its_whole_resolves_its_group_to(
    decision: DecisionFactory, members: tuple[str, ...], refused: bool
) -> None:
    ctx = _FakeToolContext()
    ctx.state[ASKS_STATE_KEY] = ["Each G7 donor's share of G7 ODA to Kenya, each year"]
    ctx.state[ASK_BREAKDOWNS_STATE_KEY] = [
        AskBreakdown(ask=0, over=("donor",)).model_dump(mode="json")
    ]

    def block(spec_id: str, label: str) -> dict[str, Any]:
        entities = [{"role": "destination", "ref": "country/KEN", "label": "Kenya"}]
        entities += [{"role": "origin", "ref": ref, "label": ref} for ref in members]
        return {
            "spec_id": spec_id,
            "resolved": {"entities": entities, "entity_groups": {"origin": {"label": label}}},
        }

    save_pin_sources(
        ctx.state,
        {
            _PART_PIN: PinSource(block=block(_PART_PIN, "G7")),
            _WHOLE_PIN: PinSource(block=block(_WHOLE_PIN, "Group of Seven")),
        },
    )
    period = Period(start=2015, end=2024)
    kenya = Entity(name="Kenya", role="recipient")

    def dataset(id: str, measure: str, group: str, pin: str) -> DatasetRequirement:
        return DatasetRequirement(
            id=id,
            decision_ids=["measure"],
            measure=measure,
            entities=(Entity(name=group, role="donor", members=True), kenya),
            period=period,
            source_spec_id=pin,
        )

    result = _tools_by_name()["propose_plan"].func(
        question="What share of G7 ODA to Kenya came from each G7 donor each year?",
        answer_mode=AnswerMode.COMPUTED,
        decisions=drafts(
            [decision(fixes=DecisionFixes(denominator_scope=DenominatorScope.POOLED))]
        ),
        evidence_requirements=[
            dataset("part", "ODA gross disbursements", "G7", _PART_PIN),
            dataset("whole", "total ODA gross disbursements", "Group of Seven", _WHOLE_PIN),
            AggregateRequirement(
                id="whole_by_year",
                decision_ids=["measure"],
                input_id="whole",
                reduction=Reduction.SUM,
                group_by=(GroupBy.PERIOD,),
            ),
            ShareOfRequirement(
                id="share",
                decision_ids=["measure"],
                numerator=RequirementSide(input_id="part"),
                denominator=RequirementSide(input_id="whole_by_year"),
                axis="donor",
                denominator_scope=DenominatorScope.POOLED,
            ),
        ],
        answered=[AskAnswer(ask=0, outputs=("share",))],
        tool_context=ctx,
    )

    if refused:
        assert result["code"] == ProblemCode.WHOLE_IS_ITS_MEMBER.value, result
    else:
        assert result["outcome"] == Outcome.OK.value, result


def test_a_plan_whose_answer_drops_the_asked_party_is_refused_until_it_keeps_it(
    decision: DecisionFactory,
) -> None:
    ctx = _creditor_context((_SPEC_BY_CREDITOR, {"origin": 97}, ("creditor",)))
    ctx.state[ASKS_ACROSS_YEARS_STATE_KEY] = [0]

    dropped = _propose_for_creditors(ctx, decision, across=["recipient"])
    kept = _propose_for_creditors(ctx, decision, across=["recipient", "creditor"], ranked=True)

    assert dropped["code"] == ProblemCode.ASK_ROLE_DROPPED.value
    assert dropped["detail"] == {"dropped": [{"ask": 0, "by": ["creditor"]}]}
    assert kept["outcome"] == Outcome.OK.value, kept
    recorded = load_plan(ctx.state)
    assert recorded is not None
    assert recorded.ask_breakdowns == (AskBreakdown(ask=0, by=("creditor",)),)
    assert recorded.asks_across_years == (0,)

    amended = _tools_by_name()["propose_plan"].func(
        answered=list(recorded.answered),
        amends_version=1,
        tool_context=ctx,
    )

    assert amended["outcome"] == Outcome.OK.value, amended
    amended_plan = load_plan(ctx.state)
    assert amended_plan is not None
    assert amended_plan.version == 2
    assert amended_plan.asks_across_years == (0,)


_OWN_PIN = {"pin": _SPEC_BY_CREDITOR}


@pytest.mark.parametrize(
    ("shown", "plan", "declined", "held_open"),
    [
        pytest.param(
            [(_SPEC_BY_CREDITOR, _DEBTORS_AND_CREDITORS, ("creditor",))],
            _OWN_PIN,
            [_BY_CREDITOR],
            True,
            id="its-own-pin-holds-debtors-and-creditors",
        ),
        pytest.param(
            [(_SPEC_BY_CREDITOR, _DEBTORS_AND_CREDITORS, ())],
            _OWN_PIN,
            [_BY_CREDITOR],
            True,
            id="its-own-pin-from-a-search-naming-no-party",
        ),
        pytest.param(
            [(_SPEC_BY_CREDITOR, {"destination": 134}, ("creditor",))],
            _OWN_PIN,
            [_BY_CREDITOR],
            None,
            id="its-own-pin-holds-no-creditor",
        ),
        pytest.param(
            [(_SPEC_BY_CREDITOR, {"destination": 134, "origin": 1}, ("creditor",))],
            _OWN_PIN,
            [_BY_CREDITOR],
            None,
            id="its-own-pin-holds-one-creditor",
        ),
        pytest.param(
            [(_SPEC_BY_CREDITOR, {"destination": 134, "origin": None}, ("creditor",))],
            _OWN_PIN,
            [_BY_CREDITOR],
            None,
            id="its-own-pin-counts-no-creditors",
        ),
        pytest.param(
            [
                (_SPEC_ODA_BY_DONOR, _DEBTORS_AND_CREDITORS, ("donor", "recipient")),
                (_SPEC_BY_CREDITOR, {"destination": 134}, ("creditor",)),
            ],
            _OWN_PIN,
            [_BY_CREDITOR],
            None,
            id="aid-open-on-donors-beside-debt",
        ),
        pytest.param(
            [
                (_SPEC_BY_CREDITOR, {"destination": 134}, ("creditor",)),
                (_SPEC_DEBT_STOCK, _DEBTORS_AND_CREDITORS, ("creditor",)),
            ],
            _OWN_PIN,
            [_BY_CREDITOR],
            None,
            id="an-unpinned-debt-stock-source-holds-creditors",
        ),
        pytest.param(
            [
                (_SPEC_BY_CREDITOR, {"destination": 134}, ("creditor",)),
                (_SPEC_DEBT_STOCK, _DEBTORS_AND_CREDITORS, ("creditor",)),
            ],
            {**_OWN_PIN, "stock_pin": _SPEC_DEBT_STOCK},
            [_BY_CREDITOR],
            None,
            id="another-asks-pin-holds-creditors",
        ),
        pytest.param(
            [(_SPEC_BY_CREDITOR, _DEBTORS_AND_CREDITORS, ("creditor",))],
            {},
            [_BY_CREDITOR],
            None,
            id="its-root-pins-no-source",
        ),
        pytest.param([], {}, [_BY_CREDITOR], None, id="no-source-shown"),
        pytest.param([], {"across": ["creditor"]}, [_BY_CREDITOR], False, id="nothing-dropped"),
        pytest.param(
            [],
            {"across": ["creditor"]},
            [_BY_CREDITOR.model_copy(update={"ask_ref": None})],
            False,
            id="no-ask-named",
        ),
    ],
)
def test_a_breakdown_decline_is_earned_unless_the_asks_own_pin_holds_the_party(
    decision: DecisionFactory,
    shown: list[_Shown],
    plan: dict[str, Any],
    declined: list[Decline],
    held_open: bool | None,
) -> None:
    """`held_open` is None when the decline is earned, True when a source the ask's own
    answer pins holds the party open, and False when the decline drops no party at all."""
    asks = [*_CREDITOR_ASKS, "Debt stock by creditor"] if "stock_pin" in plan else _CREDITOR_ASKS
    ctx = _creditor_context(*shown, asks=asks)

    result = _propose_for_creditors(
        ctx, decision, **{"across": ["recipient"], **plan}, declined=declined
    )

    if held_open is None:
        assert result["outcome"] == Outcome.OK.value, result
        recorded = load_plan(ctx.state)
        assert recorded is not None
        assert recorded.declined == tuple(declined)
        return
    assert result["code"] == ProblemCode.DECLINE_UNEARNED.value
    assert result["detail"] == {
        "operations": [UnsupportedOperation.BREAKDOWN_NOT_KEPT.value],
        "held_open": (
            [{"ask": 0, "role": "creditor", "spec_id": _SPEC_BY_CREDITOR}] if held_open else []
        ),
    }
    assert PLAN_STATE_KEY not in ctx.state


@pytest.mark.parametrize(
    "source_fields",
    [
        pytest.param({"settled": False}, id="a-source-no-search-settled-on"),
        pytest.param({"block": None, "earlier_turn": True}, id="an-earlier-turns-pin"),
    ],
)
def test_a_breakdown_decline_stands_on_a_pin_whose_axes_this_entry_does_not_hold(
    decision: DecisionFactory, source_fields: dict[str, Any]
) -> None:
    """The source that refuses the decline when settled with its block earns it otherwise."""
    ctx = _creditor_context(
        (_SPEC_BY_CREDITOR, _DEBTORS_AND_CREDITORS, ("creditor",)), **source_fields
    )

    result = _propose_for_creditors(
        ctx, decision, across=["recipient"], pin=_SPEC_BY_CREDITOR, declined=[_BY_CREDITOR]
    )

    assert result["outcome"] == Outcome.OK.value, result


# A source open on its countries, its creditors and its debtors.
_COUNTRIES_CREDITORS_AND_DEBTORS: dict[str, int | None] = {
    "observationAbout": 134,
    "origin": 97,
    "destination": 120,
}
_CREDITORS_AND_ONE_DEBTOR = {**_COUNTRIES_CREDITORS_AND_DEBTORS, "destination": 1}


@pytest.mark.parametrize(
    ("columns", "asked", "across", "held"),
    [
        pytest.param(
            _COUNTRIES_CREDITORS_AND_DEBTORS,
            (),
            ["country"],
            ["creditor", "debtor"],
            id="holds-both-parties",
        ),
        pytest.param(
            _CREDITORS_AND_ONE_DEBTOR,
            (),
            ["country"],
            ["creditor"],
            id="holds-creditors-and-the-answer-keeps-neither",
        ),
        pytest.param(
            _CREDITORS_AND_ONE_DEBTOR,
            (),
            ["country", "creditor"],
            None,
            id="holds-creditors-and-the-answer-keeps-them",
        ),
        pytest.param(
            _COUNTRIES_CREDITORS_AND_DEBTORS,
            ("creditor",),
            ["country"],
            None,
            id="its-search-asked-for-creditors-alone",
        ),
        pytest.param(
            {**_COUNTRIES_CREDITORS_AND_DEBTORS, "destination": None},
            (),
            ["country"],
            None,
            id="counts-no-debtors",
        ),
    ],
)
def test_a_decline_of_two_parties_is_tested_against_the_pin_holding_them_together(
    decision: DecisionFactory,
    columns: Mapping[str, int | None],
    asked: tuple[str, ...],
    across: list[str],
    held: list[str] | None,
) -> None:
    """The decline is refused when the pin holds more of the ask's parties together than
    the answer keeps, and names those parties. `held` is None when the decline is earned."""
    ctx = _creditor_context((_SPEC_BY_CREDITOR, columns, asked), by=("creditor", "debtor"))

    result = _propose_for_creditors(
        ctx, decision, across=across, pin=_SPEC_BY_CREDITOR, declined=[_BY_CREDITOR]
    )

    if held is None:
        assert result["outcome"] == Outcome.OK.value, result
        return
    assert result["code"] == ProblemCode.DECLINE_UNEARNED.value, result
    assert result["detail"]["held_open"] == [
        {"ask": 0, "role": role, "spec_id": _SPEC_BY_CREDITOR} for role in held
    ]


def test_a_decline_of_two_parties_stands_when_each_pin_holds_one_of_them(
    decision: DecisionFactory,
) -> None:
    """Two outputs, one per party, on two sources: neither source states a pair."""
    ctx = _creditor_context(
        (_SPEC_BY_CREDITOR, {"origin": 97}, ()),
        (_SPEC_DEBT_STOCK, {"destination": 120}, ()),
        by=("creditor", "debtor"),
    )
    by_creditor, by_debtor = (
        DatasetRequirement(
            id=f"by_{party}",
            decision_ids=[party],
            measure="general government debt service",
            across=(party,),
            period=Period(start=2031, end=2031),
            source_spec_id=pin,
        )
        for party, pin in (("creditor", _SPEC_BY_CREDITOR), ("debtor", _SPEC_DEBT_STOCK))
    )

    def propose(declined: list[Decline]) -> dict[str, Any]:
        return _tools_by_name()["propose_plan"].func(
            question="general government debt service by creditor and debtor for 2031",
            answer_mode=AnswerMode.SERIES,
            decisions=drafts(
                [
                    decision(
                        slot=party,
                        fixes=DecisionFixes(among=RankAmong.POPULATION, across=(party,)),
                    )
                    for party in ("creditor", "debtor")
                ]
            ),
            evidence_requirements=[by_creditor, by_debtor],
            answered=[AskAnswer(ask=0, outputs=("by_creditor", "by_debtor"))],
            declined=declined,
            tool_context=ctx,
        )

    undeclined = propose([])
    declined = propose([_BY_CREDITOR])

    assert undeclined["code"] == ProblemCode.ASK_ROLE_DROPPED.value, undeclined
    assert undeclined["detail"] == {
        "dropped": [{"ask": 0, "by": ["debtor"], "together": ["creditor", "debtor"]}]
    }
    assert declined["outcome"] == Outcome.OK.value, declined


# The recorded first proposal for "external debt service in 2022 for each East
# African Community country by creditor": six debtors named, creditors open.
_EAC_SPEC = "spec_4edb0cc0ba2656490d2f852daef3890a"
_EAC_ASK = "external debt service in 2022 for each East African Community country by creditor"
_EAC_DEBTORS = [
    {"name": name, "role": "debtor"}
    for name in (
        "Burundi",
        "Democratic Republic of the Congo",
        "Kenya",
        "Rwanda",
        "Tanzania",
        "Uganda",
    )
]
_EAC_REQUIREMENT: dict[str, Any] = {
    "source_spec_id": _EAC_SPEC,
    "entities": _EAC_DEBTORS,
    "across": ["creditor"],
    "source": "World Bank International Debt Statistics (IDS)",
    "kind": "dataset",
    "period": {"start": 2022, "end": 2022},
    "id": "req_debt_service",
    "measure": "external debt service",
}
_EAC_SUMMED: dict[str, Any] = {
    "kind": "aggregate",
    "id": "summed",
    "input_id": "req_debt_service",
    "reduction": "sum",
    "group_by": ["entity"],
}


@pytest.mark.parametrize(
    ("requirements", "output", "declined", "code", "detail"),
    [
        pytest.param([_EAC_REQUIREMENT], "req_debt_service", [], None, None, id="as-proposed"),
        pytest.param(
            [_EAC_REQUIREMENT],
            "req_debt_service",
            [_BY_CREDITOR],
            ProblemCode.DECLINE_UNEARNED,
            {"operations": [UnsupportedOperation.BREAKDOWN_NOT_KEPT.value], "held_open": []},
            id="declined-with-nothing-to-decline",
        ),
        pytest.param(
            [{**_EAC_REQUIREMENT, "across": []}],
            "req_debt_service",
            [],
            ProblemCode.ASK_ROLE_DROPPED,
            {"dropped": [{"ask": 0, "by": ["creditor"], "together": ["debtor", "creditor"]}]},
            id="no-creditors",
        ),
        pytest.param(
            [_EAC_REQUIREMENT, _EAC_SUMMED],
            "summed",
            [],
            ProblemCode.ASK_ROLE_DROPPED,
            {"dropped": [{"ask": 0, "by": ["creditor"], "together": ["debtor", "creditor"]}]},
            id="creditors-summed-away",
        ),
    ],
)
def test_an_answer_keeps_a_party_whose_members_it_names(
    requirements: list[dict[str, Any]],
    output: str,
    declined: list[Decline],
    code: ProblemCode | None,
    detail: dict[str, Any] | None,
) -> None:
    """Six debtors named beside every creditor is one row per debtor and creditor."""
    ctx = _creditor_context(
        (_EAC_SPEC, {"origin": 97}, ("creditor",)), asks=[_EAC_ASK], by=("debtor", "creditor")
    )

    result = _tools_by_name()["propose_plan"].func(
        question="External debt service in 2022 for each East African Community country by creditor",
        answer_mode=AnswerMode.SERIES,
        answered=[{"ask": 0, "outputs": [output]}],
        evidence_requirements=requirements,
        declined=declined,
        tool_context=ctx,
    )

    if code is None:
        assert result["outcome"] == Outcome.OK.value, result
        return
    assert result["code"] == code.value, result
    assert result["detail"] == detail


_TWO_DEBTORS = _EAC_DEBTORS[:2]
_SUMMED_OVER_PARTIES = {**_EAC_SUMMED, "group_by": ["period"]}


@pytest.mark.parametrize(
    ("by", "columns", "asked", "requirement", "summed", "held_open", "held_named"),
    [
        pytest.param(
            ("debtor",),
            {"origin": 1},
            (),
            {"entities": _TWO_DEBTORS, "across": []},
            True,
            [],
            ["debtor"],
            id="named-debtors-summed-away",
        ),
        pytest.param(
            ("debtor", "creditor"),
            {"origin": 97},
            ("creditor",),
            {"entities": _TWO_DEBTORS, "across": ["creditor"]},
            True,
            ["creditor"],
            ["debtor"],
            id="named-debtors-and-open-creditors-summed-away",
        ),
        pytest.param(
            ("debtor", "creditor"),
            _DEBTORS_AND_CREDITORS,
            (),
            {"entities": [], "across": ["country", "creditor"]},
            False,
            ["debtor", "creditor"],
            None,
            id="country-stands-for-debtors-beside-creditors",
        ),
        pytest.param(
            ("debtor",),
            {"destination": 120},
            (),
            {"entities": [], "across": ["country"]},
            False,
            ["debtor"],
            None,
            id="country-stands-for-debtors",
        ),
        pytest.param(
            ("creditor", "debtor"),
            _DEBTORS_AND_CREDITORS,
            ("debtor",),
            {"entities": [], "across": ["creditor"]},
            False,
            ["creditor", "debtor"],
            None,
            id="its-across-holds-creditors-its-search-did-not-ask-for",
        ),
        pytest.param(
            ("creditor", "debtor"),
            _DEBTORS_AND_CREDITORS,
            ("creditor",),
            {"entities": [], "across": ["creditor"]},
            False,
            None,
            None,
            id="its-search-asked-for-the-party-it-keeps-alone",
        ),
        pytest.param(
            ("debtor",),
            {"observationAbout": 134},
            (),
            {"entities": [], "across": ["country"]},
            False,
            None,
            None,
            id="country-is-the-place-and-no-axis-holds-debtors",
        ),
        pytest.param(
            ("creditor", "debtor"),
            _DEBTORS_AND_CREDITORS,
            ("donor",),
            {"entities": [], "across": ["debtor"]},
            False,
            ["creditor", "debtor"],
            None,
            id="its-search-asked-for-the-giving-side-in-another-word",
        ),
        pytest.param(
            ("creditor",),
            {"origin": 97},
            ("creditor",),
            {"entities": _TWO_DEBTORS, "across": []},
            False,
            ["creditor"],
            None,
            id="a-bare-total-whose-search-asked-for-creditors",
        ),
        pytest.param(
            ("creditor",),
            {"origin": 97},
            ("debtor",),
            {"entities": _TWO_DEBTORS, "across": []},
            False,
            None,
            None,
            id="a-bare-total-whose-search-asked-for-debtors-only",
        ),
    ],
)
def test_a_breakdown_decline_counts_every_party_the_root_can_keep_on_its_pin(
    decision: DecisionFactory,
    by: tuple[PartyRole, ...],
    columns: Mapping[str, int | None],
    asked: tuple[str, ...],
    requirement: dict[str, Any],
    summed: bool,
    held_open: list[str] | None,
    held_named: list[str] | None,
) -> None:
    """A party is kept by naming its members or on an open axis, whatever word `across`
    covers that axis with. `held_open` is None when the decline is earned, and
    `held_named` when the refusal names no party held by naming."""
    ctx = _creditor_context((_EAC_SPEC, columns, asked), asks=[_EAC_ASK], by=by)
    dataset = {**_EAC_REQUIREMENT, **requirement, "decision_ids": ["measure"]}
    fixes = DecisionFixes(among=RankAmong.POPULATION, across=tuple(dataset["across"]) or None)

    result = _tools_by_name()["propose_plan"].func(
        question="External debt service in 2022 for each East African Community country by creditor",
        answer_mode=AnswerMode.SERIES,
        decisions=drafts([decision(fixes=fixes)]),
        answered=[{"ask": 0, "outputs": ["summed" if summed else "req_debt_service"]}],
        evidence_requirements=[dataset, _SUMMED_OVER_PARTIES] if summed else [dataset],
        declined=[_BY_CREDITOR],
        tool_context=ctx,
    )

    if held_open is None:
        assert result["outcome"] == Outcome.OK.value, result
        return
    assert result["code"] == ProblemCode.DECLINE_UNEARNED.value, result
    entries = {
        key: [{"ask": 0, "role": role, "spec_id": _EAC_SPEC} for role in roles]
        for key, roles in (("held_open", held_open), ("held_named", held_named))
        if roles is not None
    }
    assert result["detail"] == {
        "operations": [UnsupportedOperation.BREAKDOWN_NOT_KEPT.value],
        **entries,
    }


@pytest.mark.parametrize(
    "by", [("debtor",), ()], ids=["the-ask-lists-another-party", "the-ask-lists-no-party"]
)
def test_an_open_party_its_ask_does_not_list_still_needs_its_scope_stated(
    by: tuple[PartyRole, ...],
) -> None:
    """A party an ask totals across (`over`) reaches the plan in no breakdown, as here."""
    ctx = _creditor_context((_EAC_SPEC, {"origin": 97}, ("creditor",)), asks=[_EAC_ASK], by=by)

    result = _tools_by_name()["propose_plan"].func(
        question="External debt service in 2022 for each East African Community country by creditor",
        answer_mode=AnswerMode.SERIES,
        answered=[{"ask": 0, "outputs": ["req_debt_service"]}],
        evidence_requirements=[_EAC_REQUIREMENT],
        tool_context=ctx,
    )

    assert result["code"] == ProblemCode.RESULT_SCOPE_UNSTATED.value, result
    assert result["detail"]["gaps"] == [
        {
            "requirement_id": "req_debt_service",
            "ask": 0,
            "role": "creditor",
            "dataset_requirement_id": "req_debt_service",
        }
    ]


# A by-creditor debt dataset as a search recorded it over 2030 to 2031, and
# over 2031 alone, the period `_creditor_requirements` states.
_DEBT_2030_2031: dict[str, Any] = {
    "identity_version": "spec1",
    "dataset": "debt service by creditor",
    "requested_window": {"kind": "range", "start_year": 2030, "end_year": 2031},
}
_DEBT_2031 = {
    **_DEBT_2030_2031,
    "requested_window": {"kind": "range", "start_year": 2031, "end_year": 2031},
}
_DEBT_2030_2031_SPEC = spec_id_for(canonicalize(_DEBT_2030_2031))
_DEBT_2031_SPEC = spec_id_for(canonicalize(_DEBT_2031))


@pytest.fixture
def debt_on_record() -> Iterator[None]:
    """Both debt identities on record, so a pin of the wider one is copied over 2031."""
    recorded = {
        spec_id_for(canonicalize(identity)): RecordedIdentity(identity=identity)
        for identity in (_DEBT_2030_2031, _DEBT_2031)
    }
    token = current_recorded_identity.set(recorded.get)
    yield
    current_recorded_identity.reset(token)


@pytest.mark.usefixtures("debt_on_record")
@pytest.mark.parametrize(
    "pin", [_DEBT_2031_SPEC, _DEBT_2030_2031_SPEC], ids=["over-the-period", "over-a-wider-one"]
)
def test_a_breakdown_decline_is_refused_on_a_pin_copied_over_the_requirements_period(
    decision: DecisionFactory, pin: str
) -> None:
    """The source holds its creditors over whichever years the requirement reads it."""
    ctx = _creditor_context((pin, _DEBTORS_AND_CREDITORS, ("creditor",)))

    result = _propose_for_creditors(
        ctx, decision, across=["recipient"], pin=pin, declined=[_BY_CREDITOR]
    )

    assert result["code"] == ProblemCode.DECLINE_UNEARNED.value, result
    assert result["detail"]["held_open"] == [{"ask": 0, "role": "creditor", "spec_id": pin}]


@pytest.mark.usefixtures("debt_on_record")
def test_a_breakdown_decline_is_refused_on_a_copied_pin_an_amendment_carries(
    decision: DecisionFactory,
) -> None:
    """The recorded plan pins the copy; the decline is checked against the dataset it copies."""
    ctx = _creditor_context((_DEBT_2030_2031_SPEC, _DEBTORS_AND_CREDITORS, ("creditor",)))
    breakdowns = ctx.state.pop(ASK_BREAKDOWNS_STATE_KEY)
    recorded = _propose_for_creditors(ctx, decision, across=["recipient"], pin=_DEBT_2030_2031_SPEC)
    assert recorded["outcome"] == Outcome.OK.value, recorded
    assert {copy.windowed_from for copy in windowed_pins(ctx.state).values()} == {
        _DEBT_2030_2031_SPEC
    }
    ctx.state[ASK_BREAKDOWNS_STATE_KEY] = breakdowns

    amended = _tools_by_name()["propose_plan"].func(
        answered=[AskAnswer(ask=0, outputs=("debt_service",))],
        declined=[_BY_CREDITOR],
        amends_version=1,
        tool_context=ctx,
    )

    assert amended["code"] == ProblemCode.DECLINE_UNEARNED.value, amended
    assert amended["detail"]["held_open"] == [
        {"ask": 0, "role": "creditor", "spec_id": _DEBT_2030_2031_SPEC}
    ]


@pytest.mark.parametrize(
    ("repinned", "held_open"),
    [(_SPEC_BY_CREDITOR, False), (_SPEC_DEBT_STOCK, True)],
    ids=["to-the-same-source", "to-a-source-holding-creditors"],
)
def test_a_carried_breakdown_decline_is_checked_against_the_amended_answers_pin(
    decision: DecisionFactory, repinned: str, held_open: bool
) -> None:
    ctx = _creditor_context(
        (_SPEC_BY_CREDITOR, {"destination": 134}, ("creditor",)),
        (_SPEC_DEBT_STOCK, _DEBTORS_AND_CREDITORS, ("creditor",)),
    )
    recorded = _propose_for_creditors(
        ctx, decision, across=["recipient"], pin=_SPEC_BY_CREDITOR, declined=[_BY_CREDITOR]
    )
    assert recorded["outcome"] == Outcome.OK.value, recorded

    amended = _tools_by_name()["propose_plan"].func(
        evidence_requirements=_creditor_requirements(["recipient"], ranked=False, pin=repinned),
        amends_version=1,
        tool_context=ctx,
    )

    if not held_open:
        assert amended["outcome"] == Outcome.OK.value, amended
        plan = load_plan(ctx.state)
        assert plan is not None
        assert plan.declined == (_BY_CREDITOR,)
        return
    assert amended["code"] == ProblemCode.DECLINE_UNEARNED.value, amended
    assert amended["detail"]["held_open"] == [
        {"ask": 0, "role": "creditor", "spec_id": _SPEC_DEBT_STOCK}
    ]


def test_an_amendment_that_answers_a_declined_ask_replaces_the_decline(
    decision: DecisionFactory, requirement: RequirementFactory
) -> None:
    tools = _tools_by_name()
    ctx = _FakeToolContext()
    ctx.state[ASKS_STATE_KEY] = _ASKS
    record_unsupported(ctx.state, _ASK_DECLINE.code)
    tools["propose_plan"].func(
        question="q",
        answer_mode=AnswerMode.SERIES,
        decisions=drafts([decision()]),
        evidence_requirements=_summed_join(requirement)[:2],
        answered=[AskAnswer(ask=0, outputs=("bilateral",))],
        declined=[_ASK_DECLINE],
        tool_context=ctx,
    )
    clear_unsupported(ctx.state)

    amended = tools["propose_plan"].func(
        answered=[AskAnswer(ask=1, outputs=("multilateral",))],
        amends_version=1,
        tool_context=ctx,
    )

    assert amended["outcome"] == Outcome.OK.value, amended
    recorded = load_plan(ctx.state)
    assert recorded is not None
    assert recorded.declined == ()
    assert recorded.answered == (
        AskAnswer(ask=0, outputs=("bilateral",)),
        AskAnswer(ask=1, outputs=("multilateral",)),
    )


@pytest.mark.parametrize(
    ("asks", "answered"),
    [
        pytest.param(
            ["imputed multilateral aid to Guatemala"],
            [AskAnswer(ask=0, outputs=("multilateral",))],
            id="changed",
        ),
        pytest.param([], [], id="emptied"),
    ],
)
def test_an_amendment_under_other_asks_accounts_for_them_from_scratch(
    decision: DecisionFactory,
    requirement: RequirementFactory,
    asks: list[str],
    answered: list[AskAnswer],
) -> None:
    """A plan recorded under one asks list and amended under another keeps none of its
    answers or ask declines: they are numbered against the old list."""
    tools = _tools_by_name()
    ctx = _FakeToolContext()
    ctx.state[ASKS_STATE_KEY] = _ASKS
    record_unsupported(ctx.state, _ASK_DECLINE.code)
    tools["propose_plan"].func(
        question="q",
        answer_mode=AnswerMode.SERIES,
        decisions=drafts([decision()]),
        evidence_requirements=_summed_join(requirement)[:2],
        answered=[AskAnswer(ask=0, outputs=("bilateral",))],
        declined=[_ASK_DECLINE],
        tool_context=ctx,
    )
    ctx.state[ASKS_STATE_KEY] = asks
    clear_unsupported(ctx.state)

    amended = tools["propose_plan"].func(
        answered=answered,
        removed_evidence_requirements=["bilateral"],
        amends_version=1,
        tool_context=ctx,
    )

    assert amended["outcome"] == Outcome.OK.value, amended
    recorded = load_plan(ctx.state)
    assert recorded is not None
    assert recorded.asks == tuple(asks)
    assert recorded.answered == tuple(answered)
    assert recorded.declined == ()


# --- a pin retrieved with no end year, for a period that follows the newest data ------

_ROLLS_AT_ITS_END = {"window_end": {"anchor": "latest", "offset": 0}}
_ROLLS_AT_BOTH_ENDS = {"window_start": {"anchor": "latest", "offset": -5}, **_ROLLS_AT_ITS_END}
_LAST_FOUR_BACK = {"window_start": {"anchor": "latest", "offset": -4}, **_ROLLS_AT_ITS_END}
_LAST_THREE_BACK = {"window_start": {"anchor": "latest", "offset": -3}, **_ROLLS_AT_ITS_END}


def _growth_as_written(
    dataset: DatasetRequirement, *, start: int, rolling: dict[str, Any] | None
) -> dict[str, Any]:
    """A growth over `dataset` to 2024, as the JSON a model writes for `propose_plan`."""
    growth: dict[str, Any] = {
        "kind": "growth",
        "id": "growth",
        "input_id": dataset.id,
        "window_start": start,
        "window_end": 2024,
    }
    return growth if rolling is None else {**growth, "rolling": rolling}


def _every_year_block(
    identity: dict[str, Any], *, first_year: int, rows: int | None
) -> dict[str, Any]:
    """Search's answer for `identity`, over no window: the first year it holds and its rows."""
    return {
        "spec_id": spec_id_for(canonicalize(identity)),
        "definition": {"identity": identity},
        "resolved": {
            "download": {"rows": rows, "window": {"start_year": first_year, "end_year": 2024}}
        },
    }


def _show_block(ctx: _FakeToolContext, block: dict[str, Any]) -> None:
    """Record `block` as a settled answer the planner was shown this entry."""
    save_pin_sources(ctx.state, {block["spec_id"]: PinSource(block=block, settled=True)})


def _rolling_issue(result: dict[str, Any]) -> tuple[str, str | None, list[str]]:
    """The issue, fault and requirements a refusal of a rolling period names."""
    assert result["code"] == ProblemCode.INVALID_PLAN.value, result
    detail = result["detail"]
    return detail["issue"], detail.get("fault"), detail["requirement_ids"]


@pytest.mark.usefixtures("aid_on_record")
def test_a_period_rolling_at_its_end_is_pinned_from_its_own_start_with_no_end_year(
    pinning_decision: Callable[..., Decision],
    requirement: Callable[..., DatasetRequirement],
) -> None:
    """ "Since 2015": the plan keeps the years it read, and the pin moves to the
    recorded dataset over a range from 2015 with no end year, stored and read back
    as that."""
    tools = _tools_by_name()
    ctx = _FakeToolContext()
    dataset = requirement(source_spec_id=_AID_SPEC, period_end_open=True)

    result = _propose_pinned(
        tools,
        ctx,
        pinning_decision(_AID_SPEC, resolved_by=ResolvedBy.PLANNER_DEFAULT),
        [
            dataset.model_dump(mode="json"),
            _growth_as_written(dataset, start=2015, rolling=_ROLLS_AT_ITS_END),
        ],
    )

    assert result["outcome"] == Outcome.OK.value, result
    window = {"kind": "range", "start_year": 2015}
    opened, identity = windowed_identity(_AID_IDENTITY, window)
    plan = load_plan(ctx.state)
    assert plan is not None
    pinned, growth = plan.evidence_requirements
    assert isinstance(pinned, DatasetRequirement)
    assert (pinned.source_spec_id, pinned.period) == (opened, Period(start=2015, end=2024))
    assert isinstance(growth, GrowthRequirement)
    assert growth.rolling == RollingWindow(window_end=RelativePeriod(offset=0))
    assert windowed_pins(ctx.state) == {
        opened: WindowedPin(spec_id=opened, identity=identity, windowed_from=_AID_SPEC)
    }


@pytest.mark.usefixtures("gdp_on_record")
def test_a_period_rolling_from_its_start_is_pinned_from_the_first_year_the_dataset_holds(
    requirement: Callable[..., DatasetRequirement],
) -> None:
    """ "The last five years": the newest year the data holds can sit behind the
    one the plan read, so the retrieval starts where the dataset does, 1960, and
    not at the plan's 2019. The copy survives the state round trip."""
    tools = _tools_by_name()
    ctx = _FakeToolContext()
    pin = _GDP_ALL["spec_id"]
    _show_block(ctx, _every_year_block(_GDP_ALL["identity"], first_year=1960, rows=65))
    dataset = requirement(
        decision_ids=[],
        source_spec_id=pin,
        period=Period(start=2019, end=2024),
        period_end_open=True,
    )

    result = tools["propose_plan"].func(
        question="q",
        answer_mode=AnswerMode.COMPUTED,
        evidence_requirements=[
            dataset.model_dump(mode="json"),
            _growth_as_written(dataset, start=2019, rolling=_ROLLS_AT_BOTH_ENDS),
        ],
        tool_context=ctx,
    )

    assert result["outcome"] == Outcome.OK.value, result
    opened, identity = windowed_identity(
        _GDP_ALL["identity"], {"kind": "range", "start_year": 1960}
    )
    plan = load_plan(ctx.state)
    assert plan is not None
    assert plan.evidence_requirements[0].source_spec_id == opened  # type: ignore[union-attr]
    held = WindowedPin(spec_id=opened, identity=identity, windowed_from=pin)
    assert windowed_pins(ctx.state) == {opened: held}
    assert "end_year" not in ctx.state[WINDOWED_PINS_STATE_KEY][0]["identity"]["requested_window"]


@pytest.mark.usefixtures("gdp_on_record")
@pytest.mark.parametrize(
    ("shown", "read", "issue", "fault"),
    [
        # No answer over every year is on record, so the first year is unknown.
        (None, False, "rolling_needs_open_period", "first_year_unknown"),
        # `propose_plan` read the dataset over no window, and the answer stated no first year.
        (None, True, "rolling_needs_open_period", "first_year_read_failed"),
        # A first year no plan can state is none.
        (
            {"first_year": 1850, "rows": 65},
            False,
            "rolling_needs_open_period",
            "first_year_unknown",
        ),
        # 250,000 rows is the most one retrieval returns (`max_frame_rows`).
        ({"first_year": 1960, "rows": 250_001}, False, "rolling_frame_too_large", None),
    ],
)
def test_a_start_that_follows_the_newest_data_is_refused_without_a_whole_history_to_retrieve(
    requirement: Callable[..., DatasetRequirement],
    shown: dict[str, Any] | None,
    read: bool,
    issue: str,
    fault: str | None,
) -> None:
    tools = _tools_by_name()
    ctx = _FakeToolContext()
    pin = _GDP_ALL["spec_id"]
    if shown is None:
        _show(ctx, pin)
    else:
        _show_block(ctx, _every_year_block(_GDP_ALL["identity"], **shown))
    if read:
        save_full_history(ctx.state, pin, None)
    dataset = requirement(
        decision_ids=[],
        source_spec_id=pin,
        period=Period(start=2019, end=2024),
        period_end_open=True,
    )

    result = tools["propose_plan"].func(
        question="q",
        answer_mode=AnswerMode.COMPUTED,
        evidence_requirements=[
            dataset,
            _growth_as_written(dataset, start=2019, rolling=_ROLLS_AT_BOTH_ENDS),
        ],
        tool_context=ctx,
    )

    assert _rolling_issue(result) == (issue, fault, [dataset.id])
    assert load_plan(ctx.state) is None
    assert windowed_pins(ctx.state) == {}


@pytest.mark.parametrize(
    ("end_open", "pin", "rolling", "fault"),
    [
        # Unpinned, the dataset is searched for, and no search states its first year.
        (True, None, _ROLLS_AT_BOTH_ENDS, "no_source_on_record"),
        (False, _AID_SPEC, _ROLLS_AT_ITS_END, "dataset_closed"),
    ],
)
@pytest.mark.usefixtures("aid_on_record")
def test_a_rolling_period_is_refused_over_a_dataset_that_cannot_be_retrieved_open(
    pinning_decision: Callable[..., Decision],
    requirement: Callable[..., DatasetRequirement],
    end_open: bool,
    pin: str | None,
    rolling: dict[str, Any],
    fault: str,
) -> None:
    tools = _tools_by_name()
    ctx = _FakeToolContext()
    dataset = requirement(
        decision_ids=["measure"] if pin else [], source_spec_id=pin, period_end_open=end_open
    )

    result = tools["propose_plan"].func(
        question="q",
        answer_mode=AnswerMode.COMPUTED,
        decisions=(
            drafts([pinning_decision(pin, resolved_by=ResolvedBy.PLANNER_DEFAULT)]) if pin else []
        ),
        evidence_requirements=[
            dataset,
            _growth_as_written(dataset, start=2019, rolling=rolling),
        ],
        tool_context=ctx,
    )

    assert _rolling_issue(result) == ("rolling_needs_open_period", fault, [dataset.id])
    assert load_plan(ctx.state) is None


def test_an_unpinned_period_rolling_at_its_end_is_recorded_for_the_translator_to_search(
    requirement: Callable[..., DatasetRequirement],
) -> None:
    """ "Since 2015" with no dataset to pin: the start is fixed, so the translator's
    search from 2015 on finds a definition with no end year."""
    tools = _tools_by_name()
    ctx = _FakeToolContext()
    dataset = requirement(decision_ids=[], period_end_open=True)

    result = tools["propose_plan"].func(
        question="q",
        answer_mode=AnswerMode.COMPUTED,
        evidence_requirements=[
            dataset,
            _growth_as_written(dataset, start=2015, rolling=_ROLLS_AT_ITS_END),
        ],
        tool_context=ctx,
    )

    assert result["outcome"] == Outcome.OK.value, result
    plan = load_plan(ctx.state)
    assert plan is not None
    recorded = plan.evidence_requirements[0]
    assert isinstance(recorded, DatasetRequirement)
    assert (recorded.source_spec_id, recorded.period_end_open) == (None, True)


@pytest.mark.usefixtures("gdp_on_record")
def test_an_open_pin_and_a_closed_period_of_it_are_pinned_apart(
    pinning_decision: Callable[..., Decision],
    requirement: Callable[..., DatasetRequirement],
) -> None:
    """Two periods of one pin retrieve two spec_ids. The decision the model
    writes fixes no pin, so each requirement takes its own copy of it."""
    tools = _tools_by_name()
    pin = _GDP_ALL["spec_id"]
    opened = requirement(id="recent", source_spec_id=pin, period_end_open=True)
    earlier = requirement(id="earlier", source_spec_id=pin, period=Period(start=2000, end=2010))
    ctx = _FakeToolContext()
    _show(ctx, pin)

    result = _propose_pinned(
        tools,
        ctx,
        pinning_decision(pin, resolved_by=ResolvedBy.PLANNER_DEFAULT),
        [opened, earlier],
    )

    assert result["outcome"] == Outcome.OK.value, result
    plan = load_plan(ctx.state)
    assert plan is not None
    pins = [
        item.source_spec_id
        for item in plan.evidence_requirements
        if isinstance(item, DatasetRequirement)
    ]
    copies = windowed_pins(ctx.state)
    assert len(set(pins)) == 2
    assert {copies[copy].windowed_from for copy in pins if copy is not None} == {pin}


# --- a dataset read from its first year serves only the readers rolling from it ---

_ON_EVERY_YEAR = _FRANCE_KENYA["definition"]["identity"]
_EVERY_YEAR_FROM = 1978  # `_FRANCE_KENYA`'s first year on record
_RANKED_ON = Decision(
    slot="population",
    description="Whether to rank every donor or only the ones named.",
    options=["every donor", "the donors named"],
    chosen="every donor",
    resolved_by=ResolvedBy.PLANNER_DEFAULT,
    rationale="The question asks for the top donor among all of them.",
    fixes=DecisionFixes(among=RankAmong.POPULATION),
)
_BY_DONOR = Decision(
    slot="by_donor",
    description="Whether aid is given per donor or summed over them.",
    options=["per donor", "summed over donors"],
    chosen="per donor",
    resolved_by=ResolvedBy.PLANNER_DEFAULT,
    rationale="The question compares donors.",
    fixes=DecisionFixes(across=("donor",)),
)
_SERIES_AND_TOP = ("aid to Kenya each year since 2015", "Kenya's top donor in the newest year")


def _aid_to_kenya(**overrides: Any) -> DatasetRequirement:
    """Gross aid to Kenya by donor, pinned to `_GROSS` with no end year."""
    fields: dict[str, Any] = {
        "id": "aid",
        "decision_ids": ["by_donor"],
        "measure": "gross aid disbursements",
        "entities": [Entity(name="Kenya", role="recipient")],
        "across": ["donor"],
        "period": Period(start=2015, end=2024),
        "unit": Unit(currency="USD", prices="current"),
        "source_spec_id": _GROSS,
        "period_end_open": True,
    }
    return DatasetRequirement(**(fields | overrides))


def _top_donor(input_id: str = "aid", **overrides: Any) -> RankRequirement:
    """The donor giving the most in the newest year, both ends of the window following it."""
    fields: dict[str, Any] = {
        "id": "top",
        "decision_ids": ["population"],
        "input_id": input_id,
        "direction": RankDirection.HIGHEST_FIRST,
        "among": RankAmong.POPULATION,
        "axis": "donor",
        "reduction": Reduction.SUM,
        "window_start": 2024,
        "window_end": 2024,
        "rolling": RollingWindow(
            window_start=RelativePeriod(offset=0), window_end=RelativePeriod(offset=0)
        ),
    }
    return RankRequirement(**(fields | overrides))


_YEARLY_TOTAL = AggregateRequirement(
    id="yearly", input_id="aid", reduction=Reduction.SUM, group_by=(GroupBy.ENTITY, GroupBy.PERIOD)
)


def _propose_split(
    ctx: _FakeToolContext,
    requirements: Sequence[EvidenceRequirement],
    *,
    asks: Sequence[str],
    answered: Sequence[AskAnswer],
    decisions: Sequence[Decision] = (_BY_DONOR, _RANKED_ON),
    across_years: Sequence[int] = (),
    answer_mode: AnswerMode = AnswerMode.COMPUTED,
    amends_version: int | None = None,
) -> dict[str, object]:
    """Propose `requirements` over `_GROSS`, shown with every year it holds, answering `asks`.

    `across_years` numbers the asks marked as one figure over several years.
    """
    _show_block(ctx, _FRANCE_KENYA)
    ctx.state[ASKS_STATE_KEY] = list(asks)
    ctx.state[ASKS_ACROSS_YEARS_STATE_KEY] = list(across_years)
    return _tools_by_name()["propose_plan"].func(
        question="q",
        answer_mode=answer_mode,
        decisions=drafts(decisions) if amends_version is None else [],
        evidence_requirements=list(requirements),
        answered=list(answered),
        amends_version=amends_version,
        tool_context=ctx,
    )


def _windows(ctx: _FakeToolContext, plan: Plan) -> dict[str, object]:
    """The `requested_window` each dataset requirement of `plan` is retrieved over."""
    copies = windowed_pins(ctx.state)
    return {
        requirement.id: copies[requirement.source_spec_id].identity["requested_window"]
        for requirement in plan.evidence_requirements
        if isinstance(requirement, DatasetRequirement) and requirement.source_spec_id
    }


@pytest.mark.parametrize(
    ("requirements", "asks", "answered", "copies", "starts"),
    [
        pytest.param(
            [_aid_to_kenya(), _top_donor()],
            _SERIES_AND_TOP,
            [AskAnswer(ask=0, outputs=("aid",)), AskAnswer(ask=1, outputs=("top",))],
            [("aid_history", "aid")],
            {"aid": 2015, "aid_history": _EVERY_YEAR_FROM},
            id="a-series-beside-the-rank",
        ),
        pytest.param(
            [_aid_to_kenya(), _YEARLY_TOTAL, _top_donor("yearly", axis="recipient")],
            _SERIES_AND_TOP,
            [AskAnswer(ask=0, outputs=("yearly",)), AskAnswer(ask=1, outputs=("top",))],
            [("aid_history", "aid"), ("yearly_history", "yearly")],
            {"aid": 2015, "aid_history": _EVERY_YEAR_FROM},
            id="a-yearly-total-answering-and-read-by-the-rank",
        ),
        pytest.param(
            [_aid_to_kenya(), _top_donor()],
            _SERIES_AND_TOP[1:],
            [AskAnswer(ask=0, outputs=("top",))],
            [],
            {"aid": _EVERY_YEAR_FROM},
            id="the-rank-alone",
        ),
        pytest.param(
            [_aid_to_kenya(period=Period(start=_EVERY_YEAR_FROM, end=2024)), _top_donor()],
            _SERIES_AND_TOP,
            [AskAnswer(ask=0, outputs=("aid",)), AskAnswer(ask=1, outputs=("top",))],
            [],
            {"aid": _EVERY_YEAR_FROM},
            id="a-first-year-on-record-at-the-stated-start",
        ),
    ],
)
def test_only_a_reader_rolling_from_its_start_retrieves_every_year_the_dataset_holds(
    requirements: list[EvidenceRequirement],
    asks: Sequence[str],
    answered: list[AskAnswer],
    copies: list[tuple[str, str]],
    starts: dict[str, int],
) -> None:
    """`top` ranks the newest year, which can sit behind the one the plan read, so
    it reads aid to Kenya from 1978, the first year `_GROSS` holds. A series answer
    or a total per year beside it keeps the stated 2015-2024: its dataset is
    retrieved over a copy of the pin from 2015, and `top` reads a requirement of
    its own retrieved over one from 1978. The plan reads back from storage as recorded,
    with no session state."""
    ctx = _FakeToolContext()

    result = _propose_split(ctx, requirements, asks=asks, answered=answered)

    assert result["outcome"] == Outcome.OK.value, result
    plan = load_plan(ctx.state)
    assert plan is not None
    windows = _windows(ctx, plan)
    assert windows == {step_id: open_window(start) for step_id, start in starts.items()}
    assert [(copy.copy_id, copy.origin_id) for copy in plan.rolling_copies] == copies
    assert plan.first_years == (FirstYear(requirement_id="aid", year=_EVERY_YEAR_FROM),)
    assert plan.requirement("top").input_id == (copies[-1][0] if copies else requirements[-2].id)  # type: ignore[union-attr]
    assert stored_plan(ctx.state[PLAN_STATE_KEY], state_key=PLAN_STATE_KEY) == plan


def test_a_stored_plan_reading_a_whole_history_with_no_first_year_is_planned_again() -> None:
    """A plan stored before plans recorded their first years holds none, so the
    whole-history read it states has no year to start at and it loads as no plan."""
    ctx = _FakeToolContext()
    _propose_split(
        ctx,
        [_aid_to_kenya(), _top_donor()],
        asks=_SERIES_AND_TOP[1:],
        answered=[AskAnswer(ask=0, outputs=("top",))],
    )
    stored = json.loads(ctx.state[PLAN_STATE_KEY])
    del stored["first_years"], stored["rolling_copies"]

    assert load_plan(ctx.state) is not None
    assert stored_plan(json.dumps(stored), state_key=PLAN_STATE_KEY) is None


def test_a_patch_applies_to_the_plan_the_model_wrote_and_is_split_again() -> None:
    """The planner is shown the plan it wrote, and a patch of 'aid' is split
    again, so the copy carries the change too."""
    ctx = _FakeToolContext()
    answered = [AskAnswer(ask=0, outputs=("aid",)), AskAnswer(ask=1, outputs=("top",))]
    first = _propose_split(
        ctx, [_aid_to_kenya(), _top_donor()], asks=_SERIES_AND_TOP, answered=answered
    )
    widened = _aid_to_kenya(entities=[Entity(name="Uganda", role="recipient")])

    second = _propose_split(
        ctx,
        [widened],
        asks=_SERIES_AND_TOP,
        answered=answered,
        amends_version=cast("int", first["version"]),
    )

    assert second["outcome"] == Outcome.OK.value, second
    plan = load_plan(ctx.state)
    assert plan is not None
    assert [requirement.id for requirement in plan.evidence_requirements] == [
        "aid",
        "aid_history",
        "top",
    ]
    for step_id in ("aid", "aid_history"):
        requirement = plan.requirement(step_id)
        assert isinstance(requirement, DatasetRequirement)
        assert requirement.entities == (Entity(name="Uganda", role="recipient"),)


def test_a_patch_naming_a_copy_s_id_names_a_requirement_of_its_own() -> None:
    """The model is shown no copy, so a patch stating 'aid_history', the id
    of 'aid''s copy, states a new requirement and leaves 'aid' as it was. The
    split copies 'aid' under the next free id. A removal of 'aid_history'
    names no requirement the model wrote, and is refused. 'uganda' totals
    the new requirement, since a computed answer reads every dataset."""
    ctx = _FakeToolContext()
    answered = [AskAnswer(ask=0, outputs=("aid",)), AskAnswer(ask=1, outputs=("top",))]
    first = _propose_split(
        ctx, [_aid_to_kenya(), _top_donor()], asks=_SERIES_AND_TOP, answered=answered
    )
    version = cast("int", first["version"])
    uganda = _aid_to_kenya(id="aid_history", entities=[Entity(name="Uganda", role="recipient")])

    uganda_total = _YEARLY_TOTAL.model_copy(update={"id": "uganda", "input_id": "aid_history"})

    added = _propose_split(
        ctx,
        [uganda, uganda_total],
        asks=_SERIES_AND_TOP,
        answered=[
            AskAnswer(ask=0, outputs=("aid", "uganda")),
            AskAnswer(ask=1, outputs=("top",)),
        ],
        amends_version=version,
    )

    assert added["outcome"] == Outcome.OK.value, added
    plan = load_plan(ctx.state)
    assert plan is not None
    assert [(copy.copy_id, copy.origin_id) for copy in plan.rolling_copies] == [
        ("aid_history_2", "aid")
    ]
    for step_id, place in (("aid", "Kenya"), ("aid_history_2", "Kenya"), ("aid_history", "Uganda")):
        requirement = plan.requirement(step_id)
        assert isinstance(requirement, DatasetRequirement)
        assert requirement.entities == (Entity(name=place, role="recipient"),)
    assert plan.requirement("top").input_id == "aid_history_2"  # type: ignore[union-attr]

    ctx = _FakeToolContext()
    first = _propose_split(
        ctx, [_aid_to_kenya(), _top_donor()], asks=_SERIES_AND_TOP, answered=answered
    )
    removed = _tools_by_name()["propose_plan"].func(
        question="q",
        answer_mode=AnswerMode.COMPUTED,
        decisions=[],
        evidence_requirements=[],
        removed_evidence_requirements=["aid_history"],
        answered=answered,
        amends_version=cast("int", first["version"]),
        tool_context=ctx,
    )

    assert removed["code"] == ProblemCode.INVALID_PLAN.value, removed
    stored = load_plan(ctx.state)
    assert stored is not None
    assert [requirement.id for requirement in stored.evidence_requirements] == [
        "aid",
        "aid_history",
        "top",
    ]


def test_a_plan_reading_no_dataset_from_its_first_year_records_no_first_year() -> None:
    """A rank whose window rolls only at its end reads the stated years, so the
    first year on record is not kept, and the plan's revision is the one a plan
    with no such field has."""
    ctx = _FakeToolContext()
    rolling_end = _top_donor(
        window_start=2015, rolling=RollingWindow(window_end=RelativePeriod(offset=0))
    )

    result = _propose_split(
        ctx,
        [_aid_to_kenya(), rolling_end],
        asks=_SERIES_AND_TOP[1:],
        answered=[AskAnswer(ask=0, outputs=("top",))],
        across_years=[0],
    )

    assert result["outcome"] == Outcome.OK.value, result
    plan = load_plan(ctx.state)
    assert plan is not None
    assert (plan.rolling_copies, plan.first_years) == ((), ())
    assert _windows(ctx, plan) == {"aid": open_window(2015)}


def _over(step_id: str, input_id: str, threshold: PeriodThreshold) -> FilterRequirement:
    return FilterRequirement(id=step_id, input_id=input_id, predicates=(threshold,))


_ON_THE_NEWEST = RelativePeriod(offset=0)


@pytest.mark.parametrize(
    ("requirements", "code"),
    [
        pytest.param(
            [
                _aid_to_kenya(),
                _over("f", "aid", PeriodThreshold(op="lte", year=2023, rolling=_ON_THE_NEWEST)),
                _YEARLY_TOTAL.model_copy(update={"id": "answer", "input_id": "f"}),
                _top_donor("f"),
            ],
            ProblemCode.INVALID_PLAN,
            id="rolling-years-from-two-newest-years",
        ),
        pytest.param(
            [
                _aid_to_kenya(),
                _over("f", "aid", PeriodThreshold(op="ne", year=2020)),
                _YEARLY_TOTAL.model_copy(update={"id": "answer", "input_id": "f"}),
                _top_donor("f"),
            ],
            ProblemCode.INVALID_PLAN,
            id="a-year-removed-from-years-that-move",
        ),
        pytest.param(
            [
                _aid_to_kenya(),
                _aid_to_kenya(id="other"),
                ComparisonRequirement(
                    id="answer",
                    minuend=RequirementSide(input_id="aid"),
                    subtrahend=RequirementSide(input_id="other"),
                ),
                _top_donor("answer"),
            ],
            ProblemCode.OPERATION_UNSUPPORTED,
            id="a-rank-of-a-comparison",
        ),
    ],
)
def test_a_refusal_of_a_split_plan_names_only_requirements_the_model_wrote(
    requirements: list[EvidenceRequirement], code: ProblemCode
) -> None:
    """`top` rolls from its start, and reads a step that `answer`, answering
    ask 0, reads or is, so the rules judge a copy of that step over every
    year the datasets hold. Whatever rule refuses the plan, the answer names
    the requirements the model wrote and no copy."""
    ctx = _FakeToolContext()

    refused = _propose_split(
        ctx,
        requirements,
        asks=_SERIES_AND_TOP,
        answered=[AskAnswer(ask=0, outputs=("answer",)), AskAnswer(ask=1, outputs=("top",))],
    )

    assert refused["code"] == code.value, refused
    assert "_history" not in json.dumps(refused)
    assert load_plan(ctx.state) is None


def test_a_source_the_user_picked_stays_admitted_once_a_copy_reads_its_dataset() -> None:
    """The user picks the source a `source_name_ambiguous` referral on 'aid'
    offers, and the draft naming it is admitted. A later patch pins 'aid' and
    has `top` roll, so it reads a copy of it, 'aid_history', carrying the same
    source. The patch leaves the source as the user picked it, so it is
    admitted, and so is the next one. The first draft ranks a fixed year,
    so it needs no pin."""
    ctx = _FakeToolContext()
    kind = ReferralKind.SOURCE_NAME_AMBIGUOUS
    save_referral(ctx.state, _referral_of(kind, "aid"))
    ctx.session.events.extend(
        [
            _referred_event(kind, "aid"),
            _ask_event([{"spec_id": "provenance/a", "label": "A"}]),
            _answer_event("provenance/a"),
        ]
    )
    named = {"source": "shared", "source_ref": "provenance/a"}
    answered = [AskAnswer(ask=0, outputs=("aid",)), AskAnswer(ask=1, outputs=("top",))]
    first = _propose_split(
        ctx,
        [_aid_to_kenya(source_spec_id=None, **named), _top_donor(rolling=None)],
        asks=_SERIES_AND_TOP,
        answered=answered,
    )
    assert first["outcome"] == Outcome.OK.value, first

    pinned = _propose_split(
        ctx,
        [_aid_to_kenya(**named), _top_donor()],
        asks=_SERIES_AND_TOP,
        answered=answered,
        amends_version=cast("int", first["version"]),
    )
    assert pinned["outcome"] == Outcome.OK.value, pinned
    again = _propose_split(
        ctx,
        [_top_donor()],
        asks=_SERIES_AND_TOP,
        answered=answered,
        amends_version=cast("int", pinned["version"]),
    )

    assert again["outcome"] == Outcome.OK.value, again
    plan = load_plan(ctx.state)
    assert plan is not None
    assert [copy.copy_id for copy in plan.rolling_copies] == ["aid_history"]
    for step_id in ("aid", "aid_history"):
        requirement = plan.requirement(step_id)
        assert isinstance(requirement, DatasetRequirement)
        assert requirement.source_ref == "provenance/a"


# Gross aid since 2015, as discovery finds it for 'aid' searching 2015 on, and
# the pin discovery did not find again, aid from Germany since 2015.
_SINCE_2015, _SINCE_2015_IDENTITY = windowed_identity(_ON_EVERY_YEAR, open_window(2015))
_STALE, _STALE_IDENTITY = windowed_identity(
    _ON_EVERY_YEAR | {"entities": {"destination": ["country/KEN"], "origin": ["country/DEU"]}},
    open_window(2015),
)


@pytest.fixture
def sources_since_on_record() -> Iterator[None]:
    """`_SINCE_2015` and `_STALE` on record, as `harness.studio` sets the lookup for a turn."""
    recorded = {
        _SINCE_2015: RecordedIdentity(identity=_SINCE_2015_IDENTITY),
        _STALE: RecordedIdentity(identity=_STALE_IDENTITY),
    }
    token = current_recorded_identity.set(recorded.get)
    yield
    current_recorded_identity.reset(token)


@pytest.mark.usefixtures("sources_since_on_record")
@pytest.mark.parametrize(
    ("whole", "stale", "pin", "refused"),
    [
        pytest.param("aid_history", ("aid",), _SINCE_2015, False, id="both-stale-offered"),
        pytest.param("aid_history", ("aid",), _STALE, True, id="both-stale-not-offered"),
        pytest.param("aid_history", (), _SINCE_2015, False, id="copy-stale-offered"),
        pytest.param("aid_history", (), _STALE, True, id="copy-stale-not-offered"),
        pytest.param("aid", (), _SINCE_2015, False, id="read-whole-offered"),
    ],
)
def test_a_pin_a_source_choice_offered_is_judged_as_the_model_wrote_it(
    whole: str, stale: tuple[str, ...], pin: str, refused: bool
) -> None:
    """Discovery did not find the pin of `whole`, the requirement reading
    'aid' from its first year, and of 'aid' too where `stale` lists it, and
    offers `_SINCE_2015` for each. `whole` is 'aid''s copy 'aid_history'
    where 'aid' answers ask 0, and 'aid' itself where only `top` reads it.
    The model reads one choice, for 'aid'. Pinned there, it is admitted,
    though `whole` retrieves a windowed copy of it from 1978. The stale pin
    is refused, naming 'aid'."""
    ctx = _FakeToolContext()
    for spec_id in (_SINCE_2015, _STALE):
        save_full_history(ctx.state, spec_id, FullHistory(first_year=_EVERY_YEAR_FROM, rows=35))
    answered = [AskAnswer(ask=0, outputs=("aid",)), AskAnswer(ask=1, outputs=("top",))]
    asks = _SERIES_AND_TOP
    if whole == "aid":
        answered, asks = [AskAnswer(ask=0, outputs=("top",))], _SERIES_AND_TOP[1:]
    first = _propose_split(
        ctx, [_aid_to_kenya(source_spec_id=_STALE), _top_donor()], asks=asks, answered=answered
    )
    assert first["outcome"] == Outcome.OK.value, first
    offered = (SelectedSource(spec_id=_SINCE_2015, sentence="gross aid", has_data=True),)
    save_referral(
        ctx.state,
        Referral(
            kind=ReferralKind.PINNED_SOURCE_NOT_FOUND,
            reason="r",
            requirement_id=whole,
            candidates=offered,
            stale_pins=tuple(
                StalePin(requirement_id=step_id, candidates=offered) for step_id in stale
            ),
        ),
    )

    repinned = _propose_split(
        ctx,
        [_aid_to_kenya(source_spec_id=pin)],
        asks=asks,
        answered=answered,
        amends_version=cast("int", first["version"]),
    )

    if refused:
        assert repinned["code"] == ProblemCode.SOURCE_NOT_OFFERED.value, repinned
        assert repinned["detail"] == {"requirement_id": "aid", "candidates": [_SINCE_2015]}
        assert "_history" not in json.dumps(repinned)
    else:
        assert repinned["outcome"] == Outcome.OK.value, repinned
        plan = load_plan(ctx.state)
        assert plan is not None
        read = plan.requirement(whole)
        assert isinstance(read, DatasetRequirement) and read.source_spec_id is not None
        assert windowed_pins(ctx.state)[read.source_spec_id].windowed_from == _SINCE_2015
        if whole != "aid":
            assert plan.requirement("aid").source_spec_id == _SINCE_2015  # type: ignore[union-attr]


# Gross aid over every year it holds, as discovery finds it for 'aid''s copy,
# which searches from the dataset's first year.
_WHOLE, _WHOLE_IDENTITY = windowed_identity(_ON_EVERY_YEAR, open_window(_EVERY_YEAR_FROM))


@pytest.mark.parametrize(
    "held", ["searched", "discovered", "nowhere"], ids=["on-record", "in-the-log", "not-on-record"]
)
def test_a_pin_offered_for_a_copy_s_whole_history_reads_the_origin_s_stated_years(
    held: str,
) -> None:
    """Discovery did not find the pin of 'aid_history', 'aid''s copy, and offers
    `_WHOLE`, found over every year the dataset holds. The model reads the
    choice for 'aid' and pins it there. With its identity on record, from a
    search in the conversation or as the source discovery logged, 'aid'
    retrieves it from its stated 2015 and the copy from 1978. With none, the
    years it holds are unknown, so the pin is refused for 'aid' and nothing
    reads the whole history in its place."""
    on_record = held != "nowhere"
    lookup = {_STALE: RecordedIdentity(identity=_STALE_IDENTITY)}
    if held == "searched":
        lookup[_WHOLE] = RecordedIdentity(identity=_WHOLE_IDENTITY)
    token = current_recorded_identity.set(lookup.get)
    try:
        ctx = _FakeToolContext()
        if held == "discovered":
            found = DiscoveryRecord(
                revision="earlier",
                tool="search",
                requirement_id="aid_history",
                requirement_key="earlier",
                status=DiscoveryStatus.DEFINITE,
                spec_ids=(_WHOLE,),
            )
            whole = SelectedSource(
                spec_id=_WHOLE, sentence="gross aid", has_data=True, identity=_WHOLE_IDENTITY
            )
            save_log(ctx.state, load_log(ctx.state).with_record(found, (whole,)))
        for spec_id in (_WHOLE, _STALE):
            save_full_history(ctx.state, spec_id, FullHistory(first_year=_EVERY_YEAR_FROM, rows=35))
        answered = [AskAnswer(ask=0, outputs=("aid",)), AskAnswer(ask=1, outputs=("top",))]
        first = _propose_split(
            ctx,
            [_aid_to_kenya(source_spec_id=_STALE), _top_donor()],
            asks=_SERIES_AND_TOP,
            answered=answered,
        )
        assert first["outcome"] == Outcome.OK.value, first
        save_referral(
            ctx.state,
            Referral(
                kind=ReferralKind.PINNED_SOURCE_NOT_FOUND,
                reason="r",
                requirement_id="aid_history",
                candidates=(SelectedSource(spec_id=_WHOLE, sentence="gross aid", has_data=True),),
            ),
        )

        repinned = _propose_split(
            ctx,
            [_aid_to_kenya(source_spec_id=_WHOLE)],
            asks=_SERIES_AND_TOP,
            answered=answered,
            amends_version=cast("int", first["version"]),
        )
    finally:
        current_recorded_identity.reset(token)

    if not on_record:
        assert repinned["code"] == ProblemCode.INVALID_PLAN.value, repinned
        assert repinned["detail"] == {
            "issue": "rolling_needs_open_period",
            "requirement_ids": ["aid"],
            "fault": "no_source_on_record",
        }
        assert load_plan(ctx.state).version == first["version"]  # type: ignore[union-attr]
        return
    assert repinned["outcome"] == Outcome.OK.value, repinned
    plan = load_plan(ctx.state)
    assert plan is not None
    held = windowed_pins(ctx.state)
    windows = {}
    for step_id in ("aid", "aid_history"):
        read = plan.requirement(step_id)
        assert isinstance(read, DatasetRequirement) and read.source_spec_id is not None
        windows[step_id] = (
            held[read.source_spec_id].identity["requested_window"]
            if read.source_spec_id in held
            else _WHOLE_IDENTITY["requested_window"]
        )
    assert windows == {"aid": open_window(2015), "aid_history": open_window(_EVERY_YEAR_FROM)}


@pytest.fixture
def gross_on_record() -> Iterator[None]:
    """`_GROSS`'s identity on record, so a closed requirement over it is copied too."""
    recorded = RecordedIdentity(identity=_ON_EVERY_YEAR)
    token = current_recorded_identity.set({_GROSS: recorded}.get)
    yield
    current_recorded_identity.reset(token)


@pytest.mark.parametrize(
    ("requirements", "starts"),
    [
        pytest.param(
            [_aid_to_kenya(), _top_donor()],
            {"aid": 2015, "aid_history": _EVERY_YEAR_FROM},
            id="a-dataset-and-its-copy",
        ),
        pytest.param(
            [_aid_to_kenya(), _aid_to_kenya(id="earlier", period=Period(start=2000, end=2024))],
            {"aid": 2015, "earlier": 2000},
            id="two-open-periods",
        ),
        pytest.param(
            [
                _aid_to_kenya(period_end_open=False),
                _aid_to_kenya(
                    id="earlier", period=Period(start=2000, end=2010), period_end_open=False
                ),
            ],
            {"aid": 2015, "earlier": 2000},
            id="two-closed-periods",
        ),
    ],
)
@pytest.mark.usefixtures("gross_on_record")
def test_one_measure_decision_over_several_periods_keeps_its_pin_and_each_its_copy(
    requirements: list[EvidenceRequirement], starts: dict[str, int]
) -> None:
    """`propose_plan` writes the measure decision `_GROSS` offers, and every
    requirement over the pin cites it. Over different periods the requirements
    retrieve different spec_ids, so the decision keeps the pin it fixes, and
    each requirement its own copy over its period."""
    ctx = _FakeToolContext()
    outputs = tuple(requirement.id for requirement in requirements)

    result = _propose_split(
        ctx,
        requirements,
        asks=_SERIES_AND_TOP,
        answered=[AskAnswer(ask=0, outputs=outputs[:1]), AskAnswer(ask=1, outputs=outputs[1:])],
        decisions=[_BY_DONOR, _RANKED_ON] if "top" in outputs else [_BY_DONOR],
        answer_mode=AnswerMode.COMPUTED if "top" in outputs else AnswerMode.SERIES,
    )

    assert result["outcome"] == Outcome.OK.value, result
    plan = load_plan(ctx.state)
    assert plan is not None
    (measure,) = [d for d in plan.decisions if d.fixes and d.fixes.source_spec_id]
    assert measure.fixes is not None and measure.fixes.source_spec_id == _GROSS
    citing = {r.id for r in plan.evidence_requirements if measure.slot in r.decision_ids}
    assert citing == set(starts)
    windows = _windows(ctx, plan)
    assert {step_id: windows[step_id]["start_year"] for step_id in starts} == starts  # type: ignore[index]


# --- reading a pinned dataset's first year, for a start that follows the newest data ---

_AID_EVERY_YEAR_ID, _AID_EVERY_YEAR = windowed_identity(_AID_2020_2024, {"kind": "all"})


class _DefinitionReads:
    """A tool surface whose `resolve_definition` answers `_AID_2020_2024` over no window, held from 1995."""

    def __init__(self) -> None:
        self.calls: list[Mapping[str, object]] = []

    async def list_tools(self) -> list[Any]:
        return []

    async def call_tool(
        self, *, tool: ToolId, arguments: Mapping[str, object], on_progress: Any = None
    ) -> ToolCallResult:
        del on_progress
        assert tool.name == "resolve_definition", tool
        self.calls.append(arguments)
        block = {
            "spec_id": _AID_EVERY_YEAR_ID,
            "definition": {"identity": _AID_EVERY_YEAR},
            "resolved": {"download": {"rows": 870, "window": {"start_year": 1995}}},
        }
        part = {"status": "definite", "spec": block}
        return ToolCallResult(is_error=False, meta={SEARCH_ENVELOPE_META_KEY: {"parts": [part]}})


def _tools_reading(surface: ToolSurface) -> dict[str, FunctionTool]:
    return {tool.name: tool for tool in build_planner_tools(surface=surface)}


@pytest.fixture(params=["searched", "discovered"])
def aid_held(request: pytest.FixtureRequest) -> str:
    """Where the 2020-2024 education-aid identity is on record: a search in the
    conversation (`aid_on_record`), or discovery's log alone, which the test writes."""
    if request.param == "searched":
        request.getfixturevalue("aid_on_record")
    return request.param


async def test_a_second_proposal_reads_no_first_year_the_turn_already_holds(
    pinning_decision: Callable[..., Decision],
    requirement: Callable[..., DatasetRequirement],
    aid_held: str,
) -> None:
    """The first proposal reads the recorded dataset over no window once; an
    amendment of the turn that still rolls from its start is pinned from the year
    held. A dataset only discovery found, such as a referral's candidate, is on
    record in its log and read the same way."""
    reads = _DefinitionReads()
    propose = _tools_reading(reads)["propose_plan"].func
    ctx = _FakeToolContext()
    if aid_held == "discovered":
        found = DiscoveryRecord(
            revision="earlier",
            tool="search",
            requirement_id="aid",
            requirement_key="earlier",
            status=DiscoveryStatus.DEFINITE,
            spec_ids=(_AID_2020_2024_SPEC,),
        )
        aid = SelectedSource(
            spec_id=_AID_2020_2024_SPEC, sentence="aid", has_data=True, identity=_AID_2020_2024
        )
        save_log(ctx.state, load_log(ctx.state).with_record(found, (aid,)))
    dataset = requirement(
        source_spec_id=_AID_2020_2024_SPEC,
        period=Period(start=2020, end=2024),
        period_end_open=True,
    )
    arguments: dict[str, Any] = {
        "question": "q",
        "answer_mode": AnswerMode.SERIES,
        "decisions": drafts(
            [pinning_decision(_AID_2020_2024_SPEC, resolved_by=ResolvedBy.PLANNER_DEFAULT)]
        ),
    }

    first = await propose(
        **arguments,
        evidence_requirements=[
            dataset,
            _growth_as_written(dataset, start=2019, rolling=_ROLLS_AT_BOTH_ENDS)
            | {"window_start": 2020, "rolling": _LAST_FOUR_BACK},
        ],
        tool_context=ctx,
    )
    second = await propose(
        evidence_requirements=[
            _growth_as_written(dataset, start=2019, rolling=_ROLLS_AT_BOTH_ENDS)
            | {"window_start": 2021, "rolling": _LAST_THREE_BACK},
        ],
        amends_version=first["version"],
        tool_context=ctx,
    )

    assert first["outcome"] == second["outcome"] == Outcome.OK.value, (first, second)
    assert reads.calls == [{"spec_id": _AID_EVERY_YEAR_ID, "identity": _AID_EVERY_YEAR}]
    opened, _ = windowed_identity(_AID_2020_2024, {"kind": "range", "start_year": 1995})
    plan = load_plan(ctx.state)
    assert plan is not None
    assert plan.evidence_requirements[0].source_spec_id == opened  # type: ignore[union-attr]


@pytest.mark.usefixtures("aid_on_record")
async def test_first_year_reads_name_each_read_propose_plan_can_still_make(
    pinning_decision: Callable[..., Decision],
    requirement: Callable[..., DatasetRequirement],
) -> None:
    """Of two datasets the planner was shown, the one whose block states its first year
    needs no read. The other is the read `propose_plan` then makes, and once made it is
    no longer one."""
    reads = _DefinitionReads()
    ctx = _FakeToolContext()
    stated = _every_year_block(_GDP_ALL["identity"], first_year=1960, rows=65)
    save_pin_sources(
        ctx.state,
        {
            _AID_2020_2024_SPEC: PinSource(settled=True),
            stated["spec_id"]: PinSource(block=stated, settled=True),
        },
    )
    dataset = requirement(
        source_spec_id=_AID_2020_2024_SPEC,
        period=Period(start=2020, end=2024),
        period_end_open=True,
    )

    before = first_year_reads(ctx.state)
    proposed = await _tools_reading(reads)["propose_plan"].func(
        question="q",
        answer_mode=AnswerMode.SERIES,
        decisions=drafts(
            [pinning_decision(_AID_2020_2024_SPEC, resolved_by=ResolvedBy.PLANNER_DEFAULT)]
        ),
        evidence_requirements=[
            dataset,
            _growth_as_written(dataset, start=2020, rolling=_LAST_FOUR_BACK),
        ],
        tool_context=ctx,
    )

    assert proposed["outcome"] == Outcome.OK.value, proposed
    assert before == [_AID_2020_2024]
    assert reads.calls == [{"spec_id": _AID_EVERY_YEAR_ID, "identity": _AID_EVERY_YEAR}]
    assert first_year_reads(ctx.state) == []


class _SilentSurface:
    """A tool surface whose every call starts and never answers."""

    async def list_tools(self) -> list[Any]:
        return []

    async def call_tool(self, **kwargs: Any) -> ToolCallResult:
        del kwargs
        await asyncio.Event().wait()
        raise AssertionError("unreachable")


async def test_a_first_year_read_gives_up_at_its_deadline() -> None:
    # Well inside `FIRST_YEAR_READ_DEADLINE_SECONDS`, so only `deadline_s` ends the read.
    async with asyncio.timeout(1.0):
        with pytest.raises(DefinitionNotResolvedError) as raised:
            await read_full_history(_SilentSurface(), _AID_2020_2024, deadline_s=0.01)

    assert raised.value.retryable


def test_propose_plan_declares_the_same_arguments_with_and_without_a_surface() -> None:
    declared = [
        tools["propose_plan"]._get_declaration().model_dump(mode="json")
        for tools in (_tools_by_name(), _tools_reading(_DefinitionReads()))
    ]

    assert declared[0] == declared[1]


def test_a_pin_read_from_the_block_the_planner_was_shown_keeps_its_set() -> None:
    # A requirement with no end year pins a dataset over Africa's countries from
    # the block it was shown; the copy over its window posts Africa again.
    fixture = Path(__file__).parent.parent / "fixtures" / "search_spec_region_members.json"
    block = json.loads(fixture.read_text())["spec"]
    spec_id = block["spec_id"]

    recorded = identity_on_record(
        spec_id, shown={spec_id: PinSource(block=block, settled=True)}, end_open=True
    )

    assert recorded is not None
    assert recorded.identity == block["definition"]["identity"]
    assert [(group.label, group.ref) for group in recorded.groups] == [("Africa", "africa")]


def test_a_shown_block_with_a_set_of_an_unknown_class_puts_no_pin_on_record() -> None:
    fixture = Path(__file__).parent.parent / "fixtures" / "search_spec_region_members.json"
    block = json.loads(fixture.read_text())["spec"]
    block["resolved"]["entity_groups"]["observationAbout"]["whole_of"] = ["party"]
    spec_id = block["spec_id"]

    recorded = identity_on_record(
        spec_id, shown={spec_id: PinSource(block=block, settled=True)}, end_open=True
    )

    assert recorded is None


def test_amendment_keeps_measure_choice_from_the_compiled_family(
    decision: DecisionFactory, requirements: RequirementPair
):
    tools = _tools_by_name()
    ctx = _FakeToolContext()
    _show_block(ctx, _FRANCE_KENYA)
    dataset, derived = requirements
    first = tools["propose_plan"].func(
        question="q",
        answer_mode=AnswerMode.COMPUTED,
        decisions=drafts([decision()]),
        evidence_requirements=[dataset.model_copy(update={"source_spec_id": _GROSS}), derived],
        tool_context=ctx,
    )
    assert first["outcome"] == Outcome.OK.value, first
    translated = _translate(ctx)
    source = source_from_spec_block(_FRANCE_KENYA)
    facets = {"unit": {"ref": "US_Dollar"}}
    identity = {**source.identity, "facets": facets}
    reminted = spec_id_for(canonicalize(identity))
    sibling_identity = {**source.elected_measure.alternatives[0].identity, "facets": facets}
    sibling = spec_id_for(canonicalize(sibling_identity))
    election = source.elected_measure.model_copy(
        update={
            "alternatives": (
                source.elected_measure.alternatives[0].model_copy(
                    update={"spec_id": sibling, "identity": sibling_identity}
                ),
            )
        }
    )
    source = source.model_copy(
        update={"spec_id": reminted, "identity": identity, "elected_measure": election}
    )
    workflow = translated.workflow.model_copy(
        update={
            "sources": (source,),
            "evidence_bindings": tuple(
                b.model_copy(update={"spec_id": reminted})
                for b in translated.workflow.evidence_bindings
            ),
            "steps": tuple(
                step.model_copy(update={"args": {"spec_id": reminted}})
                if step.tool.name == "get_data"
                else step
                for step in translated.workflow.steps
            ),
        }
    )
    translation = translated.translation.model_copy(
        update={
            "operations": tuple(
                op.model_copy(update={"spec_id": reminted})
                if isinstance(op, RetrieveOperation)
                else op
                for op in translated.translation.operations
            )
        }
    )
    compiled = translated.model_copy(update={"workflow": workflow, "translation": translation})
    save_plan(ctx.state, Plan.model_validate(compiled.model_dump(mode="json")))
    save_log(ctx.state, DiscoveryLog(sources={dataset.id: {reminted: source}}))

    second = tools["propose_plan"].func(
        evidence_requirements=[dataset.model_copy(update={"source_spec_id": sibling}), derived],
        amends_version=first["version"],
        tool_context=ctx,
    )

    assert second["outcome"] == Outcome.OK.value, second
    plan = load_plan(ctx.state)
    pinned = plan.requirement(dataset.id).source_spec_id
    assert pinned != sibling
    assert windowed_pins(ctx.state)[pinned].windowed_from == sibling
    measure = next(d for d in plan.decisions if d.fixes and d.fixes.source_spec_id == pinned)
    assert measure.options == ("gross disbursements", "net disbursements")
    assert measure.chosen == "net disbursements"
    assert measure.resolved_by == ResolvedBy.PLANNER_DEFAULT
    assert search_pin_slot(pinned) in plan.requirement(dataset.id).decision_ids


def test_prior_translator_provider_sources_remain_summable(decision, pinning_decision, requirement):

    ctx = _FakeToolContext()
    plan = _sum_of_two_sources(
        ctx,
        datasets=(CRS, DAC2A),
        resolved_by=ResolvedBy.PLANNER_DEFAULT,
        decision=decision,
        pinning_decision=pinning_decision,
        requirement=requirement,
    )
    readings = [
        {
            "text": "aid to Ghana",
            "status": "definite",
            "spec_id": SPEC_GHANA_AID,
            "provenance_ref": CRS,
            "provenance_account": PROVIDER_ACCOUNT,
        },
        {
            "text": "debt of Ghana",
            "status": "definite",
            "spec_id": SPEC_GHANA_DEBT,
            "provenance_ref": DAC2A,
            "provenance_account": PROVIDER_ACCOUNT,
        },
    ]
    save_pin_sources(
        ctx.state,
        _reading_pin_sources(
            readings,
            [r["spec_id"] for r in readings],
            blocks={},
            earlier={r["spec_id"] for r in readings},
            warnings={},
        ),
    )
    assert (
        _combination_unasked(
            plan, ToolContextSession(ctx), previous=None, sources=_shown_readings(ctx)
        )
        is None
    )


@pytest.mark.parametrize("target", ["annual_debt_service", "another_dataset"])
def test_chosen_source_reference_is_bound_to_its_requirement(decision, requirements, target):
    dataset, derived = requirements
    plan = Plan(
        question="question", decisions=[decision()], evidence_requirements=[dataset, derived]
    )
    ctx = _FakeToolContext()
    save_referral(ctx.state, _referral_of(ReferralKind.SOURCE_NAME_AMBIGUOUS, dataset.id))
    ctx.session.events.extend(
        [
            _referred_event(ReferralKind.SOURCE_NAME_AMBIGUOUS, dataset.id),
            _ask_event([{"spec_id": "provenance/a", "label": "A"}]),
            _answer_event("provenance/a"),
        ]
    )
    modified = dataset.model_copy(
        update={"id": target, "source": "shared", "source_ref": "provenance/a"}
    )
    changed = plan.model_copy(update={"evidence_requirements": (modified,)})
    assert (
        _source_ref_unpicked(changed, load_referral(ctx.state), ToolContextSession(ctx), None)
        is None
    ) == (target == dataset.id)
    changed_wording = changed.model_copy(
        update={"evidence_requirements": (modified.model_copy(update={"source": "unrelated"}),)}
    )
    assert (
        _source_ref_unpicked(
            changed_wording, load_referral(ctx.state), ToolContextSession(ctx), None
        )
        is not None
    )
    ctx.state.pop("translator:referral")
    assert (
        _source_ref_unpicked(changed, load_referral(ctx.state), ToolContextSession(ctx), None)
        is not None
    )


@pytest.mark.parametrize(
    ("code", "picked", "valid"),
    [
        ("DT.TDS.DEGG.CD", "provenance/a", True),
        ("DT.TDS.DEGG.CD", "provenance/b", False),
        ("DT.DOD.DECT.CD", "provenance/a", False),
    ],
    ids=["picked_holder", "holder_not_picked", "another_code"],
)
def test_a_code_held_by_several_publishers_takes_the_picked_one_as_its_source(
    decision, requirements, code, picked, valid
):
    dataset, derived = requirements
    referral = _referral_of(ReferralKind.SOURCE_NAME_AMBIGUOUS, dataset.id)
    (group,) = referral.ambiguous_sources
    ctx = _FakeToolContext()
    save_referral(
        ctx.state,
        referral.model_copy(
            update={"ambiguous_sources": (group.model_copy(update={"text": "DT.TDS.DEGG.CD"}),)}
        ),
    )
    ctx.session.events.extend(
        [
            _referred_event(ReferralKind.SOURCE_NAME_AMBIGUOUS, dataset.id),
            _ask_event([{"spec_id": "provenance/a", "label": "a"}]),
            _answer_event(picked),
        ]
    )
    repaired = DatasetRequirement.model_validate(
        dataset.model_dump() | {"source_code": code, "source": "a", "source_ref": "provenance/a"}
    )
    plan = Plan(
        question="question", decisions=[decision()], evidence_requirements=[repaired, derived]
    )
    assert (
        _source_ref_unpicked(plan, load_referral(ctx.state), ToolContextSession(ctx), None) is None
    ) == valid


@pytest.mark.parametrize(
    ("entity", "referred", "previous", "valid"),
    [
        (Entity(name="Congo", role="donor", dcid="country/COG"), True, None, True),
        (Entity(name="Congo", role="donor", dcid="country/XXX"), True, None, False),
        (Entity(name="Congo", role="recipient", dcid="country/COG"), True, None, False),
        (Entity(name="Congo", role="donor", dcid="country/COG"), False, None, False),
        (
            Entity(name="Congo", role="donor", dcid="country/COG"),
            False,
            Entity(name="Congo", role="donor", dcid="country/COG"),
            True,
        ),
        (
            Entity(name="Congo", role="donor", dcid="country/COG"),
            False,
            Entity(name="Congo", role="donor", dcid="country/COD"),
            False,
        ),
        (Entity(name="Congo", role="donor"), False, None, True),
    ],
    ids=[
        "offered",
        "not_offered",
        "another_entity",
        "no_referral",
        "kept_from_previous",
        "changed_from_previous",
        "unpinned",
    ],
)
def test_an_entity_node_must_be_one_a_referral_offered_for_it(
    decision, requirements, entity, referred, previous, valid
):
    dataset, derived = requirements
    ctx = _FakeToolContext()
    if referred:
        save_referral(ctx.state, _referral_of(ReferralKind.ENTITY_NAME_AMBIGUOUS, dataset.id))

    def plan_naming(named: Entity) -> Plan:
        stated = dataset.model_copy(update={"entities": (named,)})
        return Plan(
            question="question", decisions=[decision()], evidence_requirements=[stated, derived]
        )

    refusal = _entity_dcid_unoffered(
        plan_naming(entity),
        load_referral(ctx.state),
        plan_naming(previous) if previous is not None else None,
    )

    assert (refusal is None) == valid
    if refusal is not None:
        assert refusal["code"] == ProblemCode.INVALID_PLAN.value


def _congo_both_ends(requirement_id: str) -> Referral:
    """An `entity_name_ambiguous` referral reading "aid from Congo to Congo": one
    reading per end, each offering both Congos."""
    return Referral(
        kind=ReferralKind.ENTITY_NAME_AMBIGUOUS,
        reason="r",
        requirement_id=requirement_id,
        ambiguous_entities=tuple(
            AmbiguousEntityReading(
                text="Congo",
                candidates=tuple(
                    EntityCandidate(ref=f"country/{code}", label=code) for code in ("COD", "COG")
                ),
                entity=EntityKey(name="Congo", role=role),
            )
            for role in ("donor", "recipient")
        ),
        repair=ENTITY_CHOICE_REPAIR,
    )


# The `requirements` fixture's dataset requirement, which `_congo_both_ends` is about.
_CONGO_REQUIREMENT = "annual_debt_service"


def _congo_pick(role: str, code: str) -> str:
    """The pick of `code` for `role`'s Congo in `_CONGO_REQUIREMENT` (`entity_pick`)."""
    return f"{_CONGO_REQUIREMENT}|{role}|Congo|country/{code}"


def _congo_ask(role: str, *, call_id: str) -> Any:
    """An ask offering one end's Congos under their picks."""
    return _ask_event(
        [{"spec_id": _congo_pick(role, code), "label": code} for code in ("COD", "COG")],
        call_id=call_id,
    )


def _congo_answer(role: str, code: str, *, call_id: str) -> Any:
    """The user's pick of `code` answering the ask about `role`'s Congo."""
    return _answer_event(_congo_pick(role, code), call_id=call_id)


@pytest.mark.parametrize(
    ("events", "stated", "valid"),
    [
        (
            [
                _congo_ask("donor", call_id="a1"),
                _congo_answer("donor", "COD", call_id="a1"),
                _congo_ask("recipient", call_id="a2"),
                _congo_answer("recipient", "COG", call_id="a2"),
            ],
            {"donor": "COD", "recipient": "COG"},
            True,
        ),
        (
            [
                _congo_ask("donor", call_id="a1"),
                _congo_answer("donor", "COD", call_id="a1"),
                _congo_ask("recipient", call_id="a2"),
                _congo_answer("recipient", "COG", call_id="a2"),
            ],
            {"donor": "COG", "recipient": "COD"},
            False,
        ),
        (
            [_congo_ask("donor", call_id="a1"), _congo_answer("donor", "COD", call_id="a1")],
            {"donor": "COD", "recipient": "COG"},
            True,
        ),
        (
            [
                _congo_ask("donor", call_id="a1"),
                _congo_answer("donor", "COD", call_id="a1"),
                _congo_ask("donor", call_id="a2"),
                _congo_answer("donor", "COG", call_id="a2"),
            ],
            {"donor": "COG"},
            True,
        ),
        (
            [_congo_ask("donor", call_id="a1"), _ask_response("a1", {"result": {"skip": True}})],
            {"donor": "COD"},
            False,
        ),
        (
            [_congo_ask("donor", call_id="a1")],
            {"donor": "COD"},
            False,
        ),
        (
            [
                _congo_ask("donor", call_id="a1"),
                _ask_response("a1", {"result": {"text": "the Republic"}}),
            ],
            {"donor": "COG"},
            True,
        ),
        (
            [
                _congo_ask("donor", call_id="a1"),
                _congo_answer("donor", "COD", call_id="a1"),
                _referred_event(ReferralKind.ENTITY_NAME_AMBIGUOUS, _CONGO_REQUIREMENT),
            ],
            {"donor": "COG"},
            False,
        ),
        (
            [
                _ask_event(
                    [
                        {"spec_id": _congo_pick("donor", "COD"), "label": "COD"},
                        {"spec_id": _congo_pick("recipient", "COG"), "label": "COG"},
                    ],
                    call_id="a1",
                ),
                _ask_response(
                    "a1",
                    problem(ProblemCode.INVALID_ARGUMENTS, "mixed", outcome=Outcome.INVALID_INPUT),
                ),
            ],
            {"recipient": "COG"},
            True,
        ),
        (
            [
                _ask_event(
                    [
                        {"spec_id": _congo_pick("donor", "COD"), "label": "COD"},
                        {"spec_id": _congo_pick("recipient", "COG"), "label": "COG"},
                    ],
                    call_id="a1",
                ),
                _ask_response("a1", {"error": "missing message"}),
            ],
            {"recipient": "COG"},
            True,
        ),
        (
            [
                _congo_ask("donor", call_id="a1"),
                _congo_answer("donor", "COD", call_id="a1"),
                _congo_ask("donor", call_id="a2"),
                _ask_response("a2", {"result": {"text": "the Republic"}}),
            ],
            {"donor": "COG"},
            True,
        ),
        (
            [
                _congo_ask("donor", call_id="a1"),
                _congo_answer("donor", "COD", call_id="a1"),
                _referred_event(ReferralKind.ENTITY_NAME_AMBIGUOUS, _CONGO_REQUIREMENT),
                _congo_ask("donor", call_id="a2"),
                _ask_response("a2", {"result": {"text": "the Republic"}}),
            ],
            {"donor": "COG"},
            True,
        ),
        (
            [
                _congo_ask("donor", call_id="a1"),
                _congo_answer("donor", "COD", call_id="a1"),
                _referred_event(ReferralKind.ENTITY_NAME_AMBIGUOUS, _CONGO_REQUIREMENT),
                _congo_ask("donor", call_id="a2"),
                _congo_answer("donor", "COG", call_id="a2"),
            ],
            {"donor": "COD"},
            False,
        ),
        (
            [
                _congo_ask("donor", call_id="a1"),
                _congo_answer("donor", "COD", call_id="a1"),
                _typed("I meant the Republic"),
            ],
            {"donor": "COG"},
            True,
        ),
        (
            [
                _congo_ask("donor", call_id="a1"),
                _congo_answer("donor", "COD", call_id="a1"),
                *_amendment_events("the Republic, as donor"),
            ],
            {"donor": "COG"},
            True,
        ),
    ],
    ids=[
        "each_end_answered",
        "ends_swapped",
        "unasked_end_settled_by_the_planner",
        "latest_answer_wins",
        "asked_and_skipped",
        "asked_and_pending",
        "answered_in_own_words",
        "pick_binds_after_a_re_referral",
        "refused_ask_holds_nothing",
        "ask_adk_rejected_holds_nothing",
        "free_text_after_a_pick",
        "free_text_after_a_re_referral",
        "re_picked_after_a_re_referral",
        "user_message_releases_a_pick",
        "amendment_releases_a_pick",
    ],
)
def test_an_entity_node_must_be_the_one_the_user_picked_for_it(
    decision, requirements, events, stated, valid
):
    dataset, derived = requirements
    ctx = _FakeToolContext()
    save_referral(ctx.state, _congo_both_ends(dataset.id))
    ctx.session.events.extend(
        [_referred_event(ReferralKind.ENTITY_NAME_AMBIGUOUS, dataset.id), *events]
    )
    entities = tuple(
        Entity(name="Congo", role=role, dcid=f"country/{code}") for role, code in stated.items()
    )
    plan = Plan(
        question="question",
        decisions=[decision()],
        evidence_requirements=[dataset.model_copy(update={"entities": entities}), derived],
    )

    refusal = _entity_dcid_unpicked(plan, load_referral(ctx.state), ToolContextSession(ctx), None)

    assert (refusal is None) == valid
    if refusal is not None:
        assert refusal["code"] == ProblemCode.INVALID_PLAN.value


def _congo_split_draft(decision):
    draft = split_draft(decision)
    entities = (Entity(name="Congo", role="donor"), Entity(name="Congo", role="recipient"))
    return Plan.model_validate(
        draft.model_dump()
        | {
            "evidence_requirements": [
                requirement.model_copy(update={"entities": entities})
                if isinstance(requirement, DatasetRequirement)
                else requirement
                for requirement in draft.evidence_requirements
            ]
        }
    )


@pytest.mark.parametrize(("offered_for", "refused"), [("debt", False), ("debt_history", True)])
async def test_an_entity_ask_on_a_history_copy_offers_the_origin_picks(
    decision, offered_for, refused
):
    ctx: Any = _FakeToolContext()
    save_plan(ctx.state, _congo_split_draft(decision))
    save_referral(ctx.state, _congo_both_ends("debt_history"))
    reading = _congo_both_ends(offered_for).ambiguous_entities[0]

    result = await _tools_by_name()["adk_request_input"].run_async(
        args={
            "message": "Which Congo gave the aid?",
            "candidates": [{"spec_id": c.pick, "label": c.label} for c in reading.candidates],
        },
        tool_context=ctx,
    )

    if refused:
        assert result["code"] == ProblemCode.INVALID_ARGUMENTS.value
    else:
        assert result is None


@pytest.mark.parametrize(
    ("answer", "stated", "referred_again", "valid"),
    [
        ("COD", "COD", False, True),
        ("COD", "COG", False, False),
        (None, "COD", False, False),
        ("skip", "COD", False, False),
        ("COD", "COG", True, False),
    ],
    ids=["as-picked", "other-node", "pending", "skipped", "referred-to-origin"],
)
def test_an_entity_answer_on_a_history_copy_binds_its_origin(
    decision, answer, stated, referred_again, valid
):
    draft = _congo_split_draft(decision)
    ctx: Any = _FakeToolContext()
    save_plan(ctx.state, draft)
    referral = _congo_both_ends("debt_history")
    save_referral(ctx.state, referral)
    shown = authored_answer(referral.model_dump(mode="json"), split_of(draft))
    readings = cast("list[dict[str, Any]]", shown["ambiguous_entities"])
    candidates = [
        {"spec_id": candidate["pick"], "label": candidate["label"]}
        for candidate in readings[0]["candidates"]
    ]
    ctx.session.events.extend(
        [_referred_event(referral.kind, str(shown["requirement_id"])), _ask_event(candidates)]
    )
    if answer == "skip":
        ctx.session.events.append(_ask_response("call-1", {"result": {"skip": True}}))
    elif answer is not None:
        ctx.session.events.append(_answer_event(candidates[0]["spec_id"]))
    if referred_again:
        save_referral(ctx.state, _congo_both_ends("debt"))
        ctx.session.events.append(_referred_event(referral.kind, "debt"))
    entities = (
        Entity(name="Congo", role="donor", dcid=f"country/{stated}"),
        Entity(name="Congo", role="recipient"),
    )
    plan = draft.model_copy(
        update={
            "evidence_requirements": tuple(
                requirement.model_copy(update={"entities": entities})
                if isinstance(requirement, DatasetRequirement)
                else requirement
                for requirement in draft.evidence_requirements
            )
        }
    )
    view = ToolContextSession(ctx)

    refusal = _entity_dcid_unpicked(
        plan, submission_service._authored_referral(view, draft), view, draft
    )

    assert (refusal is None) == valid
    if refusal is not None:
        assert refusal["code"] == ProblemCode.INVALID_PLAN.value


def _central_africa(requirement_id: str) -> Referral:
    """An `entity_name_ambiguous` referral reading "aid to Central African countries":
    the UN's group node, restated by its registered name since the row asks for its
    members, beside the African Union's and the World Bank's regions."""
    return Referral(
        kind=ReferralKind.ENTITY_NAME_AMBIGUOUS,
        reason="r",
        requirement_id=requirement_id,
        ambiguous_entities=(
            AmbiguousEntityReading(
                text="Central Africa",
                candidates=(
                    EntityCandidate(
                        ref="region/UN_CentralAfrica",
                        label="UN",
                        restate_as="Central Africa (UN)",
                    ),
                    EntityCandidate(ref="region/AU_CentralAfrica", label="AU"),
                    EntityCandidate(ref="region/WB_CentralAfrica", label="WB"),
                ),
                entity=EntityKey(name="Central Africa", role="recipient"),
            ),
        ),
        repair=ENTITY_CHOICE_REPAIR,
    )


_PICK_UN = f"{_CONGO_REQUIREMENT}|recipient|Central Africa|Central Africa (UN)"
_PICK_AU = f"{_CONGO_REQUIREMENT}|recipient|Central Africa|region/AU_CentralAfrica"
_RESTATED_UN = Entity(name="Central Africa (UN)", role="recipient", members=True)
_PINNED_UN = Entity(
    name="Central Africa", role="recipient", members=True, dcid="region/UN_CentralAfrica"
)
_PINNED_AU = Entity(
    name="Central Africa", role="recipient", members=True, dcid="region/AU_CentralAfrica"
)


@pytest.mark.parametrize(
    ("answer", "entity", "valid"),
    [
        (_PICK_UN, _RESTATED_UN, True),
        (_PICK_UN, _PINNED_AU, False),
        (_PICK_UN, _PINNED_UN, False),
        (_PICK_AU, _RESTATED_UN, False),
        (_PICK_AU, _PINNED_AU, True),
        (None, _RESTATED_UN, True),
        (None, _PINNED_UN, False),
        ("", _RESTATED_UN, False),
    ],
    ids=[
        "group_picked_and_restated",
        "group_picked_node_stated",
        "group_picked_and_pinned",
        "node_picked_group_restated",
        "node_picked_and_stated",
        "unasked_group_restated",
        "unasked_group_pinned",
        "asked_and_pending",
    ],
)
def test_a_group_restated_by_its_name_counts_as_its_candidate(
    decision, requirements, answer, entity, valid
):
    # The restated entity no longer carries the reading's name, so it is the
    # reading of the candidate whose `restate_as` it names.
    dataset, derived = requirements
    ctx = _FakeToolContext()
    save_referral(ctx.state, _central_africa(dataset.id))
    ctx.session.events.append(_referred_event(ReferralKind.ENTITY_NAME_AMBIGUOUS, dataset.id))
    if answer is not None:
        ctx.session.events.append(
            _ask_event(
                [{"spec_id": pick, "label": pick} for pick in (_PICK_UN, _PICK_AU)],
                call_id="a1",
            )
        )
    if answer:
        ctx.session.events.append(_answer_event(answer, call_id="a1"))
    plan = Plan(
        question="question",
        decisions=[decision()],
        evidence_requirements=[dataset.model_copy(update={"entities": (entity,)}), derived],
    )
    referral = load_referral(ctx.state)

    refusal = _entity_dcid_unoffered(plan, referral, None) or _entity_dcid_unpicked(
        plan, referral, ToolContextSession(ctx), None
    )

    assert (refusal is None) == valid
    if refusal is not None:
        assert refusal["code"] == ProblemCode.INVALID_PLAN.value


@pytest.mark.parametrize(
    ("referred", "spec_ids", "refused"),
    [
        (True, [_congo_pick("donor", "COD"), _congo_pick("donor", "COG")], False),
        (True, [_congo_pick("donor", "COD"), _congo_pick("recipient", "COG")], True),
        (True, [_congo_pick("donor", "COD"), _congo_pick("donor", "CGO")], True),
        (True, [_congo_pick("donor", "COD")], True),
        (True, ["country/COD", "country/COG"], True),
        (True, [], False),
        (False, ["country/COD", "country/COG"], False),
    ],
    ids=[
        "one_reading",
        "readings_mixed",
        "pick_mistyped",
        "one_pick_only",
        "bare_ref",
        "no_candidates",
        "no_referral",
    ],
)
async def test_an_ask_under_an_entity_referral_offers_one_reading_by_its_picks(
    referred, spec_ids, refused
):
    ctx = _FakeToolContext()
    if referred:
        save_referral(ctx.state, _congo_both_ends(_CONGO_REQUIREMENT))

    result = await _tools_by_name()["adk_request_input"].run_async(
        args={
            "message": "Which Congo gave the aid?",
            "candidates": [{"spec_id": spec_id, "label": spec_id} for spec_id in spec_ids],
        },
        tool_context=ctx,
    )

    if refused:
        assert result["code"] == ProblemCode.INVALID_ARGUMENTS.value
    else:
        assert result is None


def test_an_entity_referral_leaves_a_kept_pin_as_it_is(
    decision, pinning_decision, requirement, derived_requirement
):
    # The referral offers graph nodes and no sources, so the pin it was
    # discovered under stays until the entity is settled.
    pinned = requirement(decision_ids=["source"], source_spec_id=SPEC_GHANA_AID)
    ctx = _FakeToolContext()
    save_referral(ctx.state, _referral_of(ReferralKind.ENTITY_NAME_AMBIGUOUS, pinned.id))
    plan = Plan(
        question="question",
        decisions=[decision(), pinning_decision(SPEC_GHANA_AID, slot="source")],
        evidence_requirements=[pinned, derived_requirement()],
    )

    assert _choice_not_offered(plan, load_referral(ctx.state), ToolContextSession(ctx)) is None


@pytest.mark.parametrize("modifier", [{"source_ref": "provenance/a"}, {"source_exclude": True}])
def test_source_modifiers_without_wording_are_refused(requirement, modifier):
    with pytest.raises(ValidationError, match="require source wording"):
        requirement(**modifier)


async def test_a_double_count_refusal_names_the_wider_figure_by_its_label(
    decision: DecisionFactory,
) -> None:
    """Bilateral CRS health aid plus the DAC1 total counting core contributions to
    multilaterals, the sum accepting a provider difference: the refusal names the
    total by the count it carries, whose cause is not imputed multilateral flows."""
    ctx = _FakeToolContext()
    _show_blocks(ctx, us_2023_block("health"), dac1_with_contributions())
    summed = SumOfRequirement(
        id=_SUM,
        decision_ids=["measure"],
        left=RequirementSide(input_id=_PART),
        right=RequirementSide(input_id=_WHOLE),
        accept_provider_difference=True,
    )

    refused = await _us_2023(
        _tools_reading(WholeCompiles())["propose_plan"].func,
        ctx,
        decision(),
        part="health",
        whole="dac1_contributions",
        accept=True,
        extra=[summed],
    )

    assert refused["code"] == ProblemCode.OPERATION_UNSUPPORTED.value, refused
    assert refused["detail"]["basis"]["step_id"] == _SUM
    # Once naming the input, once as the figure that holds the bilateral flows.
    assert refused["message"].count("bilateral plus core contributions to multilaterals") == 2


def _education_block(name: str, *, spec_id: str) -> dict[str, Any]:
    """`US_2023`'s health part `name` bound to CRS 110, education, in place of
    health, under `spec_id`: the sector its identity, its resolved slot with the
    codes the code index widens 110 to, and its sector choice name."""
    block = json.loads(json.dumps(us_2023_block(name)))
    block["spec_id"] = spec_id
    resolved = block["resolved"]
    for slot in resolved["scope_choices"]["base"]["identity"]["slots"]:
        if slot["property"] == "crsPurpose":
            slot["binding"] = {"state": "value", "ref": "CRSPurposeCode110"}
    for slot in resolved["slots"]:
        if slot["key"].get("property", {}).get("ref") == "crsPurpose":
            slot["binding"] = {
                "kind": "value",
                "value": {"ref": {"ref": "CRSPurposeCode110"}, "value_kind": "enum_value"},
            }
            slot["expansion"] = {
                "added": [{"ref": "CRSPurposeCode111"}, {"ref": "CRSPurposeCode11110"}]
            }
    for choice in resolved["scope_choices"]["choices"]:
        if choice["axis"] == "sector":
            option = {
                **choice["options"][0],
                "option_id": "sector:110",
                "binding": {
                    "kind": "slot",
                    "slot": "crsPurpose",
                    "state": "value",
                    "refs": ["CRSPurposeCode110"],
                },
            }
            choice["options"], choice["active"] = [option], "sector:110"
    return block


_EDUCATION, _EDUCATION_INCLUSIVE = (f"spec_{'8' * 32}", f"spec_{'9' * 32}")


def _us_dataset(
    requirement_id: str, measure: str, pin: str | None, whole_of: WholeOf | None = None
) -> DatasetRequirement:
    """US aid of `measure` in 2023, pinned to `pin`, declaring `whole_of` when given."""
    return DatasetRequirement(
        id=requirement_id,
        decision_ids=["measure"],
        measure=measure,
        entities=[Entity(name="United States", role="donor")],
        period=Period(start=2023, end=2023),
        source_spec_id=pin,
        whole_of=whole_of,
    )


_FLAGGED_LEAD = "The question asks for the inputs as their datasets count them"


@pytest.mark.parametrize(
    ("wholes", "accepts", "pinned", "compiled", "compare_lead"),
    [
        pytest.param(
            ("dac1_contributions", "dac1_contributions"),
            (True, True),
            ("dac1_contributions", "dac1_contributions"),
            False,
            None,
            id="both-accepting-over-dac1",
        ),
        pytest.param(
            (None, "dac1_contributions"),
            (False, False),
            ("whole", "whole"),
            True,
            None,
            id="declared-whole-beside-dac1",
        ),
        pytest.param(
            (None, "dac1_contributions"),
            (False, True),
            ("whole", "dac1_contributions"),
            True,
            _FLAGGED_LEAD,
            id="declared-whole-beside-accepted-dac1",
        ),
        pytest.param(
            ("dac1_contributions", "whole"),
            (True, False),
            ("dac1_contributions", "whole"),
            False,
            _FLAGGED_LEAD,
            id="one-accepting-over-dac1",
        ),
    ],
)
async def test_comparing_two_shares_keeps_each_shares_own_provider_contract(
    decision: DecisionFactory,
    wholes: tuple[str | None, str],
    accepts: tuple[bool, bool],
    pinned: tuple[str, str],
    compiled: bool,
    compare_lead: str | None,
) -> None:
    """US health and education aid each as a share of a total, the two shares
    compared. The comparison counts numerators with numerators and denominators
    with denominators, and a share's own contract stands.

    Both shares over the DAC1 total, which counts core contributions to
    multilaterals, each accepting the provider difference: the positions already
    match, so the DAC1 total keeps its pin though Search offers DAC1 bilateral ODA
    beside it, nothing is compiled, and the comparison states no provider
    difference beside each share's own. The health share over a whole declared
    from its part, beside the education share over the DAC1 total: the declared
    whole stands in for its bilateral part, so the DAC1 denominator is compiled
    bilateral, unless the education share accepts the difference, when its DAC1
    denominator stays and the comparison states the denominators' difference as
    the user's request. The health share over the DAC1 total accepting the difference,
    beside the education share over the CRS bilateral total: the accepted
    contract stands, nothing moves, and the comparison states the denominators'
    difference as the user's request."""
    surface = WholeCompiles()
    ctx = _FakeToolContext()
    blocks = {name: _us_block(name) for name in {*wholes, *pinned} - {None}}
    _show_blocks(
        ctx,
        us_2023_block("health"),
        _education_block("health", spec_id=_EDUCATION),
        *blocks.values(),
    )

    def total(requirement_id: str, whole: str | None) -> DatasetRequirement:
        if whole is None:
            return _us_dataset(
                requirement_id, "US total ODA", None, WholeOf(part=_PART, axis="sector")
            )
        return _us_dataset(requirement_id, "US total ODA", blocks[whole]["spec_id"])

    def share(requirement_id: str, part: str, whole: str, accept: bool) -> ShareOfRequirement:
        return ShareOfRequirement(
            id=requirement_id,
            decision_ids=["measure"],
            numerator=RequirementSide(input_id=part),
            denominator=RequirementSide(input_id=whole),
            accept_provider_difference=accept,
        )

    result = await _tools_reading(surface)["propose_plan"].func(
        question="Was a bigger share of US aid spent on health or on education in 2023?",
        answer_mode=AnswerMode.COMPUTED,
        decisions=drafts([decision()]),
        evidence_requirements=[
            _us_dataset(_PART, "US aid to health", us_2023_block("health")["spec_id"]),
            _us_dataset("education_crs", "US aid to education", _EDUCATION),
            total(_WHOLE, wholes[0]),
            total("total_education", wholes[1]),
            share(_SHARE, _PART, _WHOLE, accepts[0]),
            share("share_education", "education_crs", "total_education", accepts[1]),
            ComparisonRequirement(
                id=_COMPARE,
                decision_ids=["measure"],
                minuend=RequirementSide(input_id=_SHARE),
                subtrahend=RequirementSide(input_id="share_education"),
            ),
        ],
        tool_context=ctx,
    )

    assert result["outcome"] == Outcome.OK.value, result
    assert bool(surface.calls) is compiled
    assert _pins(ctx) == [
        (_PART, us_2023_block("health")["spec_id"]),
        ("education_crs", _EDUCATION),
        (_WHOLE, blocks[pinned[0]]["spec_id"]),
        ("total_education", blocks[pinned[1]]["spec_id"]),
    ]
    plan = load_plan(ctx.state)
    assert plan is not None
    stated = [
        d
        for d in plan.decisions
        if d.caveat == "providers_differ" and d.slot in plan.requirement(_COMPARE).decision_ids
    ]
    assert [d.rationale.startswith(compare_lead) for d in stated] == (
        [] if compare_lead is None else [True]
    )


def test_comparing_two_quotients_counts_their_numerators_on_one_basis(
    decision: DecisionFactory,
) -> None:
    """US ODA counted gross per person against UK ODA counted net per person: the
    comparison reads two quotients, so it counts their numerators on one basis,
    the first's on a tie, and asks for UK ODA compiled gross."""
    ctx = _FakeToolContext()
    population = basis_block(f"spec_{'a' * 32}", narrow=False, active="measure_basis:gross")
    del population["resolved"]["scope_choices"]
    _show_blocks(
        ctx,
        basis_block(HEALTH, narrow=False, active="measure_basis:gross"),
        basis_block(TOTAL, narrow=False, active="measure_basis:net"),
        population,
    )

    def per_person(requirement_id: str, aid: str) -> QuotientRequirement:
        return QuotientRequirement(
            id=requirement_id,
            decision_ids=["measure"],
            numerator=RequirementSide(input_id=aid),
            denominator=RequirementSide(input_id="population"),
        )

    refused = _tools_by_name()["propose_plan"].func(
        question="Did the US or the UK give more ODA per person in 2023?",
        answer_mode=AnswerMode.COMPUTED,
        decisions=drafts([decision()]),
        evidence_requirements=[
            _us_dataset("us_oda", "US ODA", HEALTH),
            _us_dataset("uk_oda", "UK ODA", TOTAL),
            _us_dataset("population", "Population", population["spec_id"]),
            per_person("us_per_person", "us_oda"),
            per_person("uk_per_person", "uk_oda"),
            ComparisonRequirement(
                id=_COMPARE,
                decision_ids=["measure"],
                minuend=RequirementSide(input_id="us_per_person"),
                subtrahend=RequirementSide(input_id="uk_per_person"),
            ),
        ],
        tool_context=ctx,
    )

    assert refused["code"] == ProblemCode.BASES_UNALIGNED.value, refused
    assert refused["detail"]["realign"] == [
        {
            "spec_id": TOTAL,
            "option_ids": ["measure_basis:gross"],
            "requirement_id": "uk_oda",
            "step_id": _COMPARE,
        }
    ]


def _windowed(block: dict[str, Any], start: int, end: int) -> dict[str, Any]:
    """`block` with its base identity requested over `start` to `end`."""
    block["resolved"]["scope_choices"]["base"]["identity"]["requested_window"] = {
        "kind": "range",
        "start_year": start,
        "end_year": end,
    }
    return block


def _to_recipients(block: dict[str, Any], recipients: list[str]) -> dict[str, Any]:
    """`block` with its base and compiled identities bound to `recipients` as its destination."""
    scope = block["resolved"]["scope_choices"]
    for identity in (scope["base"]["identity"], scope["compiled"]["identity"]):
        identity["entities"]["destination"] = recipients
    return block


def _inclusive_health(spec_id: str) -> dict[str, Any]:
    """`US_2023`'s inclusive health part under `spec_id`."""
    block = json.loads(json.dumps(us_2023_block("health_inclusive")))
    block["spec_id"] = spec_id
    return block


@pytest.mark.parametrize(
    ("bilateral", "inclusive", "code"),
    [
        pytest.param(
            lambda b: b,
            lambda: _education_block("health_inclusive", spec_id=_EDUCATION_INCLUSIVE),
            Outcome.OK.value,
            id="disjoint-sectors",
        ),
        pytest.param(
            lambda b: _windowed(b, 2015, 2018),
            lambda: _windowed(_inclusive_health(_EDUCATION_INCLUSIVE), 2020, 2023),
            Outcome.OK.value,
            id="disjoint-years",
        ),
        pytest.param(
            lambda b: _to_recipients(b, ["country/KEN"]),
            lambda: _to_recipients(_inclusive_health(_EDUCATION_INCLUSIVE), ["country/GHA"]),
            Outcome.OK.value,
            id="disjoint-recipients",
        ),
        pytest.param(
            lambda b: b,
            lambda: _inclusive_health(_EDUCATION_INCLUSIVE),
            ProblemCode.OPERATION_UNSUPPORTED.value,
            id="one-sector",
        ),
        pytest.param(
            lambda b: _windowed(b, 2015, 2020),
            lambda: _windowed(_inclusive_health(_EDUCATION_INCLUSIVE), 2020, 2023),
            ProblemCode.OPERATION_UNSUPPORTED.value,
            id="overlapping-years",
        ),
        pytest.param(
            lambda b: _to_recipients(b, ["country/KEN"]),
            lambda: _to_recipients(_inclusive_health(_EDUCATION_INCLUSIVE), ["africa"]),
            ProblemCode.OPERATION_UNSUPPORTED.value,
            id="a-region-holding-the-recipient",
        ),
    ],
)
def test_a_sum_accepting_a_provider_difference_counts_flows_twice_only_where_they_meet(
    decision: DecisionFactory,
    bilateral: Callable[[dict[str, Any]], dict[str, Any]],
    inclusive: Callable[[], dict[str, Any]],
    code: str,
) -> None:
    """US bilateral health aid plus US aid with imputed multilateral flows, the sum
    accepting the provider difference. The inclusive figure holds the bilateral
    flows only where the two meet on every axis: one donor, a sector, a recipient
    and a year in common. US bilateral health aid plus US inclusive education aid
    counts no flow twice and is served; so is a sum over years or recipient
    countries apart. Over one sector, overlapping years or a region holding the
    recipient, it is refused."""
    ctx = _FakeToolContext()
    left = bilateral(json.loads(json.dumps(us_2023_block("health"))))
    right = inclusive()
    _show_blocks(ctx, left, right)

    result = _tools_by_name()["propose_plan"].func(
        question="US bilateral health aid plus US inclusive aid in 2023?",
        answer_mode=AnswerMode.COMPUTED,
        decisions=drafts([decision()]),
        evidence_requirements=[
            _us_dataset(_PART, "US bilateral aid to health", left["spec_id"]),
            _us_dataset("other_aid", "US aid with imputed multilateral", right["spec_id"]),
            SumOfRequirement(
                id=_SUM,
                decision_ids=["measure"],
                left=RequirementSide(input_id=_PART),
                right=RequirementSide(input_id="other_aid"),
                accept_provider_difference=True,
            ),
        ],
        tool_context=ctx,
    )

    assert result.get("outcome") == code or result.get("code") == code, result
    if code == Outcome.OK.value:
        assert _pins(ctx) == [(_PART, left["spec_id"]), ("other_aid", right["spec_id"])]
    else:
        assert result["detail"]["basis"]["step_id"] == _SUM
        assert load_plan(ctx.state) is None
