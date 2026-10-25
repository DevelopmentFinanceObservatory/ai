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
from typing import Any, cast

import pytest
from compiler.compile import CompileIssueCode
from compiler.sources import PROVIDER_ACCOUNT, DiscoveryStatus, source_from_spec_block
from compiler.surface import FixtureSurface, ToolCallResult, ToolSurface
from data_mcp.definition_store import InMemoryDefinitionStore, canonicalize, spec_id_for
from data_mcp.definition_writes import windowed_identity
from data_mcp.search_result import SEARCH_ENVELOPE_META_KEY
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
from harness.graph.state import APPROVE_INTERRUPT
from harness.plan_display import shown_decisions
from harness.planner_seam import _PLANNER_BRANCH, PlannerSeam, _reading_pin_sources
from harness.prefix_cache import PrefixCache
from harness.studio_state import _turn_identities
from harness.turn_records import RecordedIdentity, _periods
from planner import agent as planner_agent
from planner.agent import (
    ToolContextSession,
    build_planner_tools,
    create_planner_agent,
    declare_ready,
    planner_instruction,
    read_full_history,
)
from planner.declaration import FIELD_DESCRIBED_ENUMS, SHARED_FIELDS, PlanTool, compact_schema
from planner.operations import (
    AggregateOperation,
    RetrieveOperation,
    Translation,
)
from planner.parties import PARTY_ROLES, PartyRole, flow_side, party_role
from planner.plan import (
    ASK_BREAKDOWNS_STATE_KEY,
    ASKS_ACROSS_YEARS_STATE_KEY,
    ASKS_STATE_KEY,
    MAX_DECISION_DESCRIPTION_CHARS,
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
    RankRequirement,
    RequirementSide,
    ResolvedBy,
    SumOfRequirement,
    Unit,
    UnsupportedOperation,
    WindowedPin,
    clear_unsupported,
    current_recorded_identity,
    fix_disagreements,
    is_dataset,
    load_plan,
    load_plan_rounds,
    open_window,
    pin_sources,
    plan_revision,
    record_optional_decline,
    record_unsupported,
    save_full_history,
    save_pin_sources,
    save_plan,
    save_plan_rounds,
    search_pin_slot,
    stored_plan,
    windowed_pins,
)
from planner.records import (
    MAX_ENTITIES,
    AmbiguousSourceReading,
    DenominatorScope,
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
from planner.tool_results import Outcome, ProblemCode
from planner.workflow import EvidenceBinding, SelectedSource, ToolId, Workflow, WorkflowStep
from pydantic import TypeAdapter, ValidationError
from pydantic_core import to_jsonable_python
from submission import service as submission_service
from submission.service import (
    COMBINATION_OPTIONS,
    COMBINE_OPTION,
    _combination_unasked,
    _repinned,
    _resolve_intent,
    _source_ref_unpicked,
    first_year_reads,
    identity_on_record,
)
from translator.discovery import (
    DefinitionNotResolvedError,
    DiscoveryLog,
    DiscoveryRecord,
    build_discovery_tools,
    load_log,
    save_log,
)
from translator.referral import (
    OUTCOME_OF,
    Referral,
    ReferralKind,
    StalePin,
    load_referral,
    save_referral,
)
from translator.tool import Translate, build_translate

from ..context_probe import plan_record
from ..data_mcp_definitions import DataMcpDefinitions
from ..decision_drafts import drafts
from ..session_events import response_event
from ..split_plan import split_draft
from ..test_studio_translate import _translator as _studio_translator

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

    result = _combination_unasked(plan, ToolContextSession(ctx), previous=None)

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
        assert isinstance(aid, DatasetRequirement)
        assert aid.source_spec_id == SPEC_GHANA_AID


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

    result = _combination_unasked(plan, ToolContextSession(ctx), previous=None)

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

    result = _combination_unasked(plan, ToolContextSession(ctx), previous=None)

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
        refusal = _combination_unasked(first, ToolContextSession(ctx), previous=None)
        assert refusal is not None
        detail = refusal["detail"]
        ctx.session.events.extend(
            _combination_asked(
                {"spec_id": COMBINE_OPTION}, about=(detail["requirement_id"], detail["datasets"])
            )
        )
    else:
        previous = first

    result = _combination_unasked(plan, ToolContextSession(ctx), previous=previous)

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

    result = _combination_unasked(plan, ToolContextSession(ctx), previous=previous_of(recorded))

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


def _referral_of(kind: ReferralKind, requirement_id: str) -> Referral:
    """A valid `Referral` of `kind`, naming what that kind has to name."""
    if kind is ReferralKind.SOURCE_NAME_AMBIGUOUS:
        return Referral(
            kind=kind,
            reason="r",
            requirement_id=requirement_id,
            ambiguous_sources=(
                AmbiguousSourceReading(
                    text="shared",
                    candidates=tuple(
                        SourceNameCandidate(ref=f"provenance/{name}", label=name, kind="provenance")
                        for name in ("a", "b")
                    ),
                ),
            ),
        )
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

    answered, _picked = ToolContextSession(ctx).answers_since_referral()

    assert answered


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
    """Every `ReferralKind` is named, by its own value, in the planner's instruction.

    The planner reads a referral's `kind` verbatim off the function response,
    so a kind the instruction never names is one the model has to guess at.
    Describing a kind in prose alone does not satisfy this: the model
    matches on the string it is sent.
    """
    unnamed = [kind.value for kind in ReferralKind if kind.value not in _SERVED_INSTRUCTION]
    assert unnamed == [], (
        f"referral kinds the planner is never told how to read: {unnamed}. "
        "Name each one in `_TRANSLATING` and say what the planner does about it."
    )


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


@pytest.mark.parametrize("on_record", [True, False], ids=["on-record", "not-on-record"])
def test_a_pin_offered_for_a_copy_s_whole_history_reads_the_origin_s_stated_years(
    on_record: bool,
) -> None:
    """Discovery did not find the pin of 'aid_history', 'aid''s copy, and offers
    `_WHOLE`, found over every year the dataset holds. The model reads the
    choice for 'aid' and pins it there. With its identity on record, 'aid'
    retrieves it from its stated 2015 and the copy from 1978. With none, the
    years it holds are unknown, so the pin is refused for 'aid' and nothing
    reads the whole history in its place."""
    lookup = {_STALE: RecordedIdentity(identity=_STALE_IDENTITY)}
    if on_record:
        lookup[_WHOLE] = RecordedIdentity(identity=_WHOLE_IDENTITY)
    token = current_recorded_identity.set(lookup.get)
    try:
        ctx = _FakeToolContext()
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


@pytest.mark.usefixtures("aid_on_record")
async def test_a_second_proposal_reads_no_first_year_the_turn_already_holds(
    pinning_decision: Callable[..., Decision],
    requirement: Callable[..., DatasetRequirement],
) -> None:
    """The first proposal reads the recorded dataset over no window once; an
    amendment of the turn that still rolls from its start is pinned from the year
    held."""
    reads = _DefinitionReads()
    propose = _tools_reading(reads)["propose_plan"].func
    ctx = _FakeToolContext()
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
    assert _combination_unasked(plan, ToolContextSession(ctx), previous=None) is None


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


@pytest.mark.parametrize("modifier", [{"source_ref": "provenance/a"}, {"source_exclude": True}])
def test_source_modifiers_without_wording_are_refused(requirement, modifier):
    with pytest.raises(ValidationError, match="require source wording"):
        requirement(**modifier)
