"""Tests for the planner agent: plan-lifecycle tools and agent construction.

Most tool functions are exercised directly (`FunctionTool.func`) against a fake
tool context, rather than through a full ADK `Runner` -- what belongs to this
package is the business logic each tool function runs once ADK decides to call
it. `_FakeToolContext` carries only `.state`: `submit_plan` gates nothing of
its own, so a body that tried to touch `tool_context.tool_confirmation` or
`.actions` the way the old `execute_plan` did would raise `AttributeError`
here rather than silently working again -- see the tests under "Submitting".

The translator is stood in for by `_translate`, which completes a draft --
always a dataset requirement plus the derived one that reads it, the minimal
shape `Plan` accepts -- through the real `Plan.with_translation`, so these
tests stay about the planner's own transitions.
"""

from __future__ import annotations

import ast
import inspect
import json
from collections.abc import Callable, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from compiler.sources import source_from_spec_block
from data_mcp.definition_store import canonicalize, spec_id_for
from data_mcp.definition_writes import windowed_identity
from google.adk.agents import LlmAgent
from google.adk.events import Event, RequestInput
from google.adk.planners import BuiltInPlanner
from google.adk.tools.function_tool import FunctionTool
from google.adk.tools.long_running_tool import LongRunningFunctionTool
from google.adk.workflow.utils._workflow_hitl_utils import (
    create_request_input_event,
    create_request_input_response,
)
from google.genai import types
from harness.graph import _drop_seeds_from_turn
from harness.planner_seam import PlannerSeam
from planner import agent as planner_agent
from planner.agent import (
    _resolve_intent,
    build_planner_tools,
    create_planner_agent,
    planner_instruction,
)
from planner.operations import (
    AggregateOperation,
    RetrieveOperation,
    Translation,
)
from planner.parties import flow_side
from planner.plan import (
    DROPPED_SEEDS_STATE_KEY,
    ENVELOPE_SPEC_STATE_KEY,
    MAX_DECISION_DESCRIPTION_CHARS,
    MAX_PLAN_DECISIONS,
    MAX_PLAN_RECORD_CHARS,
    MAX_PLAN_REQUIREMENTS,
    MAX_PLAN_ROUNDS,
    MAX_QUESTION_CHARS,
    PLAN_ROUNDS_STATE_KEY,
    PLAN_STATE_KEY,
    SEEDED_SOURCES_STATE_KEY,
    SEEDS_STATE_KEY,
    AggregateRequirement,
    AnswerMode,
    DatasetRequirement,
    Decision,
    DecisionFixes,
    Entity,
    EvidenceRequirement,
    Period,
    Plan,
    PlanStatus,
    ResolvedBy,
    SeededSource,
    dropped_seeds,
    is_dataset,
    load_plan,
    load_plan_rounds,
    names_dropped_seed,
    plan_revision,
    save_dropped_seeds,
    save_plan,
    save_plan_rounds,
    save_seeded_sources,
    seed_of,
    seeded_source,
    seeded_sources,
    split_dropped,
)
from planner.records import (
    EvidenceIssue,
    EvidenceIssueCode,
    GroupBy,
    Reduction,
    Severity,
)
from planner.tool_results import Outcome, ProblemCode
from planner.workflow import EvidenceBinding, SelectedSource, ToolId, Workflow, WorkflowStep
from translator.referral import Referral, ReferralKind, load_referral, save_referral

from ..context_probe import plan_record

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

# What `submit_plan` summarises for the plan `_translate` builds with its defaults.
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

# The referral kinds the user decides (`planner.agent._USER_DECIDES`).
HANDED_BACK_KINDS = (ReferralKind.REQUIREMENT_UNMEETABLE, ReferralKind.UNSUPPORTED_ASK)


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

    `.state` (a plain, mutable mapping) and `.session` (`.events` only) are
    all `planner.agent`'s tool functions use: none of them gates on a
    confirmation any more, so none needs `.tool_confirmation` or `.actions`.
    A tool body that tried to read either would raise `AttributeError`
    against this stand-in, which is exactly the regression the tests under
    "Submitting" rely on.
    """

    def __init__(self) -> None:
        self.state: dict[str, str] = {}
        self.session = _FakeSession()


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


def _response_event(name: str, response: dict[str, Any]) -> Any:
    """A fake session event for the `FunctionResponse` a tool `name` answered with.

    The shape `planner.agent._answered_since_referral` reads: the planner's own
    tool responses (`dfo_translator`, `propose_plan`) and the user's answers
    (`adk_request_input`) are all `FunctionResponse` parts on session events.
    """
    part = SimpleNamespace(
        function_call=None,
        function_response=SimpleNamespace(id=f"{name}-1", name=name, response=response),
    )
    return SimpleNamespace(content=SimpleNamespace(parts=[part]))


def _referred_event(kind: ReferralKind, requirement_id: str | None) -> Any:
    """The event carrying what `dfo_translator` answers the planner when it hands a plan back."""
    return _response_event(
        "dfo_translator",
        {"kind": kind.value, "reason": "r", "requirement_id": requirement_id, "candidates": []},
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


def _clarify_ask_event(candidates: list[dict[str, str]]) -> Event:
    """A real ADK event for the graph's own `clarify` interrupt
    (`harness.graph.CLARIFY_INTERRUPT`), built through ADK's own
    `create_request_input_event`: this is the shape `_offered_candidates`'s
    `payload` branch reads, distinct from `ask_user`'s flat
    `args["candidates"]` -- `RequestInput.model_dump` nests the caller's own
    `payload` (where `clarify` puts `candidates`) under `args["payload"]`.
    Every clarify round shares the fixed interrupt id `"clarify"`
    (`CLARIFY_INTERRUPT`), which is easy to confuse with `_ask_event`'s
    per-call id.
    """
    return create_request_input_event(
        RequestInput(interrupt_id="clarify", payload={"candidates": candidates})
    )


def _clarify_answer_event(spec_id: str) -> Event:
    """A real ADK event answering `_clarify_ask_event`, the shape
    `harness.studio`'s `/interrupt` route actually sends for a clarify pick
    (`create_request_input_response(CLARIFY_INTERRUPT, {"result": {"spec_id": ...}})`)."""
    part = create_request_input_response("clarify", {"result": {"spec_id": spec_id}})
    return Event(content=types.Content(role="user", parts=[part]))


def _tools_by_name(
    *, has_translator: bool = True, question_fixed: bool = False
) -> dict[str, FunctionTool]:
    return {
        tool.name: tool
        for tool in build_planner_tools(
            has_translator=has_translator, question_fixed=question_fixed
        )
    }


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
        decisions=[decision],
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
            "decisions": [decision],
            "evidence_requirements": list(requirements),
            # This tool runs against a context that already holds a plan, so
            # its call states that it patches it.
            "amends_version": 1,
        },
        "reject_plan": {},
    }
    return by_tool[tool_name]


def test_build_planner_tools_registers_the_four_planner_tools() -> None:
    tools = _tools_by_name()

    assert set(tools) == {*SIMPLE_TOOLS, "submit_plan"}


# --- Submitting ---------------------------------------------------------------------


def test_submitting_a_plan_asks_for_no_confirmation(
    decision: DecisionFactory, requirements: RequirementPair
) -> None:
    # `submit_plan` gates nothing itself: `_FakeToolContext` carries no
    # `tool_confirmation` or `actions` attribute at all, so a body that
    # tried to read either would raise `AttributeError` here rather than
    # requesting one -- there is nothing left to stub out. What it returns
    # is the record a reviewer needs to judge the plan: version, revision,
    # every decision and requirement in full, the sources, each step's
    # tool with its arguments and bindings, the warnings and the notes.
    # The plan itself is left exactly where translation put it.
    tools = _tools_by_name()
    ctx = _FakeToolContext()
    _propose(tools, ctx, decision(), requirements)
    translated = _translate(ctx)

    result = tools["submit_plan"].func(version=1, tool_context=ctx)

    assert result["outcome"] == Outcome.OK.value
    assert result["status"] == PlanStatus.PENDING_APPROVAL.value
    assert result["summary"] == {
        "version": 1,
        "revision": plan_revision(translated),
        "warnings": [],
    }
    stored = load_plan(ctx.state)
    assert stored is not None
    assert stored.status == PlanStatus.PENDING_APPROVAL
    assert stored.approved_revision is None


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

    summed = tools["submit_plan"].func(version=1, tool_context=summed_ctx)
    averaged = tools["submit_plan"].func(version=1, tool_context=averaged_ctx)

    assert summed["summary"]["version"] == averaged["summary"]["version"] == 1
    assert summed["summary"]["revision"] != averaged["summary"]["revision"]


async def test_a_blocked_plan_is_refused_before_it_reaches_the_summary(
    decision: DecisionFactory, requirements: RequirementPair
) -> None:
    # A draft cannot run, so it is refused before any summary is built --
    # there is no compiled record to show.
    tools = _tools_by_name()
    ctx = _FakeToolContext()
    _propose(tools, ctx, decision(), requirements)

    result = await tools["submit_plan"].run_async(args={"version": 1}, tool_context=ctx)

    assert result["code"] == ProblemCode.WORKFLOW_MISSING.value
    assert "summary" not in result


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

    result = tools["submit_plan"].func(version=1, tool_context=ctx)

    assert result["summary"]["warnings"] == [warning.model_dump(mode="json")]


@pytest.mark.parametrize(
    ("kill_the_plan", "submitted_version", "expected_code", "expected_action_call"),
    [
        (True, 1, ProblemCode.PLAN_REJECTED, "propose_plan"),
        (False, 1, ProblemCode.VERSION_STALE, None),
    ],
    ids=["rejected-plan", "stale-version"],
)
async def test_submit_plan_stays_blocked_for_a_rejected_or_amended_plan(
    kill_the_plan: bool,
    submitted_version: int,
    expected_code: ProblemCode,
    expected_action_call: str | None,
    decision: DecisionFactory,
    requirements: RequirementPair,
) -> None:
    # A plan that was rejected or amended after it was reviewed cannot be
    # submitted, whatever version is given.
    tools = _tools_by_name()
    ctx = _FakeToolContext()
    _propose(tools, ctx, decision(), requirements)
    _translate(ctx)
    if kill_the_plan:
        tools["reject_plan"].func(tool_context=ctx)
    else:
        _propose(  # amends to version 2
            tools,
            ctx,
            decision(chosen="total debt stock", resolved_by=ResolvedBy.USER),
            requirements,
        )
        _translate(ctx)

    result = await tools["submit_plan"].run_async(
        args={"version": submitted_version}, tool_context=ctx
    )

    assert result["code"] == expected_code.value
    assert "summary" not in result
    if expected_action_call is not None:
        assert result["action"]["call"] == expected_action_call


@pytest.mark.parametrize(
    "reach",
    [
        lambda plan: plan.approve(),
        lambda plan: plan.approve().start_execution(),
        lambda plan: plan.approve().start_execution().complete(),
        lambda plan: plan.approve().start_execution().fail(),
    ],
    ids=["approved", "executing", "completed", "failed"],
)
async def test_submit_plan_refuses_a_plan_not_awaiting_approval(
    reach: Callable[[Plan], Plan], decision: DecisionFactory, requirements: RequirementPair
) -> None:
    # `graph.py` seeds a `FAILED` plan straight back into the planner after
    # an execution refusal, and that plan still carries a workflow at the
    # version it failed with -- exactly the shape every earlier check in
    # `_execution_blocked` lets through. Submitting it unchanged must not
    # succeed: approving what comes back would raise inside `Plan.approve`,
    # which requires `PENDING_APPROVAL`.
    tools = _tools_by_name()
    ctx = _FakeToolContext()
    _propose(tools, ctx, decision(), requirements)
    translated = _translate(ctx)
    advanced = reach(translated)
    save_plan(ctx.state, advanced)

    result = await tools["submit_plan"].run_async(args={"version": 1}, tool_context=ctx)

    assert result["outcome"] == Outcome.REFUSED.value
    assert result["code"] == ProblemCode.TRANSITION_REFUSED.value
    assert result["action"]["call"] == "propose_plan"
    assert "summary" not in result
    stored = load_plan(ctx.state)
    assert stored is not None
    assert stored.status == advanced.status


async def test_submit_plan_refuses_an_untranslated_draft(
    decision: DecisionFactory, requirements: RequirementPair
) -> None:
    # A draft states the methodology; nothing has been compiled to run, so
    # submitting it would put nothing real in front of the user.
    tools = _tools_by_name()
    ctx = _FakeToolContext()
    _propose(tools, ctx, decision(), requirements)

    result = await tools["submit_plan"].run_async(args={"version": 1}, tool_context=ctx)

    assert result["code"] == ProblemCode.WORKFLOW_MISSING.value
    assert result["action"]["call"] == "dfo_translator"
    assert "summary" not in result
    stored = load_plan(ctx.state)
    assert stored is not None
    assert stored.status == PlanStatus.DRAFT


async def test_submit_plan_with_a_pending_referral_is_refused(
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

    result = await tools["submit_plan"].run_async(args={"version": 1}, tool_context=ctx)

    assert result["code"] == ProblemCode.REFERRAL_PENDING.value
    assert result["action"]["call"] == "propose_plan"
    assert result["detail"]["referral"]["kind"] == ReferralKind.NO_DATA.value
    assert "summary" not in result


async def _submit_over_a_handed_back_plan(
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
    return await tools["submit_plan"].run_async(args={"version": 1}, tool_context=ctx)


@pytest.mark.parametrize("kind", HANDED_BACK_KINDS)
async def test_submit_plan_sends_the_planner_to_the_user_over_an_unanswered_referral(
    decision: DecisionFactory, requirements: RequirementPair, kind: ReferralKind
) -> None:
    result = await _submit_over_a_handed_back_plan(decision, requirements, kind)

    assert result["code"] == ProblemCode.REFERRAL_PENDING.value
    assert result["action"]["call"] == "adk_request_input"


@pytest.mark.parametrize("kind", HANDED_BACK_KINDS)
async def test_submit_plan_sends_the_planner_to_propose_once_the_user_has_answered(
    decision: DecisionFactory, requirements: RequirementPair, kind: ReferralKind
) -> None:
    result = await _submit_over_a_handed_back_plan(decision, requirements, kind, answered=True)

    assert result["code"] == ProblemCode.REFERRAL_PENDING.value
    assert result["action"]["call"] == "propose_plan"


@pytest.mark.parametrize("kind", [kind for kind in ReferralKind if kind not in HANDED_BACK_KINDS])
async def test_submit_plan_sends_the_planner_to_propose_over_a_referral_it_settles_itself(
    decision: DecisionFactory, requirements: RequirementPair, kind: ReferralKind
) -> None:
    result = await _submit_over_a_handed_back_plan(decision, requirements, kind)

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
    # No tool but `submit_plan` needs a compiled plan to run its own body,
    # or the planner could never draft a plan on its own.
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
    await tools["submit_plan"].run_async(args={"version": 1}, tool_context=ctx)
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
    # allowlist (`submit_plan` requires a plan already at that status; see
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


@pytest.mark.parametrize("tool_name", [*SIMPLE_TOOLS, "submit_plan"])
async def test_no_planner_tool_moves_a_draft_to_pending_approval(
    tool_name: str, decision: DecisionFactory, requirements: RequirementPair
) -> None:
    # The behavioural half of the invariant above: every planner tool, run
    # against a draft, leaves the plan anywhere but the status that says
    # "ready to approve".
    tools = _tools_by_name()
    ctx = _FakeToolContext()
    _propose(tools, ctx, decision(), requirements)
    args = (
        {"version": 1}
        if tool_name == "submit_plan"
        else _tool_args(tool_name, decision(), requirements)
    )

    await tools[tool_name].run_async(args=args, tool_context=ctx)

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
    arguments and are recorded in session state, where `dfo_translator` reads
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
        decisions=[decision(rationale="Reconsidered.")],
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
        decisions=[decision(rationale="Reconsidered.")], amends_version=1, tool_context=ctx
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
        decisions=[decision(rationale="Reconsidered.")], amends_version=1, tool_context=ctx
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
        decisions=[decision()], evidence_requirements=list(requirements), tool_context=ctx
    )

    assert refused["code"] == ProblemCode.INVALID_PLAN.value
    assert refused["outcome"] == Outcome.INVALID_INPUT.value
    assert "first draft" in str(refused["message"])
    assert "starts_new_plan" not in str(refused["message"])
    assert PLAN_STATE_KEY not in ctx.state


def test_a_restart_without_a_question_is_refused_and_leaves_the_plan_alone(
    decision: DecisionFactory, requirements: RequirementPair
) -> None:
    tools = _tools_by_name()
    ctx = _FakeToolContext()
    _propose(tools, ctx, decision(), requirements)

    refused = tools["propose_plan"].func(
        decisions=[decision()],
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
        decisions=[decision()],
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
        decisions=[decision()],
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


def test_the_last_version_that_compiled_stays_submittable_past_the_cap(
    decision: DecisionFactory, requirements: RequirementPair
) -> None:
    tools = _tools_by_name()
    ctx = _FakeToolContext()
    _propose_up_to_the_cap(tools, ctx, decision(), requirements)
    _translate(ctx)
    refused = _propose(tools, ctx, decision(), requirements)

    result = tools["submit_plan"].func(version=MAX_PLAN_ROUNDS, tool_context=ctx)

    assert refused["code"] == ProblemCode.BUDGET_EXHAUSTED.value
    assert result["status"] == PlanStatus.PENDING_APPROVAL.value
    assert result["version"] == MAX_PLAN_ROUNDS


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
        decisions=[decision()],
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
        decisions=[decision(chosen="total debt stock", resolved_by=ResolvedBy.USER)],
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
    assert "nothing further is needed from you" in fixed
    assert "nothing further is needed from you" not in free
    assert "has to match the one already recorded" in fixed
    assert "has to match the one already recorded" not in free


def test_planner_instruction_defaults_question_fixed_to_false() -> None:
    assert planner_instruction(has_translator=True) == planner_instruction(
        has_translator=True, question_fixed=False
    )


def test_create_planner_agent_threads_question_fixed_into_the_instruction() -> None:
    fixed = create_planner_agent(model_name="m", question_fixed=True)
    free = create_planner_agent(model_name="m", question_fixed=False)

    assert "nothing further is needed from you" in str(fixed.instruction)
    assert "nothing further is needed from you" not in str(free.instruction)


def test_create_planner_agent_defaults_question_fixed_to_false() -> None:
    default = create_planner_agent(model_name="m")

    assert "nothing further is needed from you" not in str(default.instruction)


def test_question_fixed_changes_propose_plans_declaration_for_starts_new_plan() -> None:
    free = _tools_by_name()["propose_plan"]
    fixed = _tools_by_name(question_fixed=True)["propose_plan"]

    assert "that calls for a" in str(free.func.__doc__)
    assert "that calls for a" not in str(fixed.func.__doc__)
    assert "question is fixed for this turn" in str(fixed.func.__doc__)


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
        decisions=[decision()],
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
    # that version is submittable once translated.
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
    assert tools["submit_plan"].func(version=2, tool_context=ctx)["status"] == (
        PlanStatus.PENDING_APPROVAL.value
    )


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
    if kind is ReferralKind.SOURCE_AMBIGUOUS:
        candidates = tuple(
            SelectedSource(
                spec_id="spec_" + letter * 32, sentence=f"source {letter}", has_data=True
            )
            for letter in "ab"
        )
        return Referral(kind=kind, reason="r", requirement_id=requirement_id, candidates=candidates)
    if kind is ReferralKind.NO_OPERATION_COVERS:
        return Referral(kind=kind, reason="r", decision_id="measure")
    if kind is ReferralKind.DISCOVERY_BUDGET_EXHAUSTED:
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


def test_the_translator_repeating_a_referral_does_not_discard_the_answer(
    decision: DecisionFactory, requirements: RequirementPair
) -> None:
    """A translator called again on a plan whose referral is still pending answers with
    that same referral, so the answer the user gave in between still counts."""
    tools, ctx, requirement_id = _recorded_after_a_handed_back_plan(
        decision, requirements, ReferralKind.REQUIREMENT_UNMEETABLE
    )
    ctx.session.events.extend(_ask_answered_events())
    ctx.session.events.append(_referred_event(ReferralKind.REQUIREMENT_UNMEETABLE, requirement_id))

    amended = _propose(tools, ctx, decision(rationale="Reconsidered."), ())

    assert amended["outcome"] == Outcome.OK.value, amended


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
        _response_event("propose_plan", {"outcome": "invalid_input", "code": "invalid_plan"})
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
    ctx.session.events.append(_response_event("propose_plan", settled))
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
        decisions=[bad_decision],
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
        decisions=[decision()],
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
        decisions=[decision()],
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
        decisions=[decision(), uncited],
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
        decisions=[decision()],
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
        decisions=[decision(chosen="total debt stock")],
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
        decisions=[decision(rationale="Reconsidered.")],
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
        decisions=[decision()],
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
        decisions=[decision(slot="oda", options=["total ODA", "net ODA"], chosen="total ODA")],
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
        decisions=[decision(rationale="Reconsidered.")],
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
        decisions=[decision(chosen="total debt stock", resolved_by=ResolvedBy.USER)],
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
        decisions=[added],
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
        decisions=[second],
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
        decisions=[decision(slot="window")],
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
        decisions=decisions,
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
        decisions=decisions,
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
        decisions=decisions,
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
        decisions=[decision()],
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


def test_propose_plan_refuses_a_user_election_ask_user_never_offered(
    decision: DecisionFactory, requirements: RequirementPair
) -> None:
    # A `resolved_by=user` source pin has to trace to what the user actually
    # chose answering `adk_request_input` -- see `planner.plan.
    # DatasetRequirement` and `_check_pin_rests_on_a_fixing_decision`.
    # Nothing was ever offered here, so the pin is untraceable.
    tools = _tools_by_name()
    ctx = _FakeToolContext()
    dataset, derived = requirements
    elected = decision(
        chosen="total debt stock",
        resolved_by=ResolvedBy.USER,
        fixes=DecisionFixes(source_spec_id=SPEC),
    )

    result = tools["propose_plan"].func(
        question="q",
        decisions=[elected],
        evidence_requirements=[dataset.model_copy(update={"source_spec_id": SPEC}), derived],
        tool_context=ctx,
    )

    assert result["outcome"] == Outcome.REFUSED.value
    assert result["code"] == ProblemCode.ELECTION_UNTRACEABLE.value
    assert result["detail"]["spec_ids"] == [SPEC]
    assert PLAN_STATE_KEY not in ctx.state


def test_propose_plan_refuses_a_user_election_ask_user_offered_but_user_chose_differently(
    decision: DecisionFactory, requirements: RequirementPair
) -> None:
    # SPEC was among the candidates `adk_request_input` offered, but the
    # user's own answer -- the `_answer_event` below -- picked
    # POPULATION_SPEC, not SPEC. A pin has to trace to what was actually
    # chosen, not merely offered, so pinning SPEC here is still untraceable.
    tools = _tools_by_name()
    ctx = _FakeToolContext()
    ctx.session.events = [
        _ask_event([{"spec_id": SPEC, "label": "a"}, {"spec_id": POPULATION_SPEC, "label": "b"}]),
        _answer_event(POPULATION_SPEC),
    ]
    dataset, derived = requirements
    elected = decision(
        chosen="total debt stock",
        resolved_by=ResolvedBy.USER,
        fixes=DecisionFixes(source_spec_id=SPEC),
    )

    result = tools["propose_plan"].func(
        question="q",
        decisions=[elected],
        evidence_requirements=[dataset.model_copy(update={"source_spec_id": SPEC}), derived],
        tool_context=ctx,
    )

    assert result["outcome"] == Outcome.REFUSED.value
    assert result["code"] == ProblemCode.ELECTION_UNTRACEABLE.value
    assert result["detail"]["spec_ids"] == [SPEC]
    assert PLAN_STATE_KEY not in ctx.state


def test_propose_plan_refuses_a_user_election_ask_user_asked_but_not_yet_answered(
    decision: DecisionFactory, requirements: RequirementPair
) -> None:
    # `adk_request_input` offered SPEC, but no `FunctionResponse` for that
    # call is on record yet -- the run is still paused there, or the answer
    # hasn't arrived. An offer with no answer is not an election either.
    tools = _tools_by_name()
    ctx = _FakeToolContext()
    ctx.session.events = [_ask_event([{"spec_id": SPEC, "label": "a"}])]
    dataset, derived = requirements
    elected = decision(
        chosen="total debt stock",
        resolved_by=ResolvedBy.USER,
        fixes=DecisionFixes(source_spec_id=SPEC),
    )

    result = tools["propose_plan"].func(
        question="q",
        decisions=[elected],
        evidence_requirements=[dataset.model_copy(update={"source_spec_id": SPEC}), derived],
        tool_context=ctx,
    )

    assert result["outcome"] == Outcome.REFUSED.value
    assert result["code"] == ProblemCode.ELECTION_UNTRACEABLE.value
    assert result["detail"]["spec_ids"] == [SPEC]
    assert PLAN_STATE_KEY not in ctx.state


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
        decisions=[elected],
        evidence_requirements=[dataset.model_copy(update={"source_spec_id": SPEC}), derived],
        tool_context=ctx,
    )

    assert result["outcome"] == Outcome.OK.value
    assert PLAN_STATE_KEY in ctx.state


_SEED_IDENTITY: dict[str, Any] = {
    "identity_version": "spec1",
    "seed": "education aid",
    "requested_window": {"kind": "all"},
}
_SEED_SPEC = spec_id_for(canonicalize(_SEED_IDENTITY))


def _seeded(identity: dict[str, Any] = _SEED_IDENTITY) -> SeededSource:
    """A dataset the user handed over from Search, as `harness.graph`'s `seed` node records it."""
    return SeededSource(
        source=SelectedSource(
            spec_id=spec_id_for(canonicalize(identity)),
            sentence="Education aid from every donor",
            has_data=True,
            identity=identity,
        ),
        seen_year=2024,
    )


def _propose_seeded(
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
        decisions=[decision],
        evidence_requirements=list(requirements),
        amends_version=amends_version,
        tool_context=ctx,
    )


def test_seeded_sources_name_each_spec_id_once() -> None:
    """Every reader looks a seeded dataset up by spec_id, so the state holds one entry for one."""
    state: dict[str, Any] = {}
    other = _seeded({**_SEED_IDENTITY, "seed": "population"})

    with pytest.raises(ValueError, match="more than once"):
        save_seeded_sources(state, [_seeded(), other, _seeded()])

    assert state == {}
    save_seeded_sources(state, [_seeded(), other])
    held = seeded_sources(state)
    assert seeded_source(held, _SEED_SPEC) == _seeded()
    assert seeded_source(held, other.source.spec_id) == other
    assert seeded_source(held, "spec_" + "0" * 32) is None
    assert seeded_source(held, None) is None


def _over(start: int, end: int) -> SeededSource:
    """The seed's dataset over other years: another Search result, under its own spec_id."""
    window = {"kind": "range", "start_year": start, "end_year": end}
    return _seeded({**_SEED_IDENTITY, "requested_window": window})


def test_a_drop_takes_a_seed_and_its_copies_and_leaves_another_seed_over_the_same_dataset() -> None:
    seed, sibling = _seeded(), _over(2015, 2024)
    copy_id, copy_identity = windowed_identity(
        _SEED_IDENTITY, {"kind": "range", "start_year": 2024, "end_year": 2024}
    )
    copied = SeededSource(
        source=seed.source.model_copy(update={"spec_id": copy_id, "identity": copy_identity}),
        windowed_from=_SEED_SPEC,
    )

    kept, taken = split_dropped([seed, sibling, copied], {_SEED_SPEC})

    assert kept == [sibling]
    assert taken == [seed, copied]
    assert [seed_of(item) for item in (seed, sibling, copied)] == [
        _SEED_SPEC,
        sibling.source.spec_id,
        _SEED_SPEC,
    ]


_WINDOW_2010_2023 = {"kind": "range", "start_year": 2010, "end_year": 2023}


def _seed_with_stat_vars() -> SeededSource:
    """`_seeded()` with the StatVars Search elected for it."""
    seed = _seeded()
    return seed.model_copy(
        update={"source": seed.source.model_copy(update={"stat_vars": ("Var_a", "Var_b")})}
    )


def _stored_copy_state(
    seeds: Sequence[SeededSource], window: dict[str, Any] = _WINDOW_2010_2023
) -> dict[str, Any]:
    """Session state holding `seeds` and one copy of `_SEED_SPEC` over `window`, as stored."""
    stored = {
        "seeds": [seed.model_dump(mode="json") for seed in seeds],
        "copies": [{"windowed_from": _SEED_SPEC, "window": window}],
    }
    return {SEEDED_SOURCES_STATE_KEY: json.dumps(stored)}


def test_the_state_stores_a_copy_as_its_seed_and_window(
    pinning_decision: Callable[..., Decision],
    requirement: Callable[..., DatasetRequirement],
) -> None:
    """A copy is its seed over another window: the state holds the seed's spec_id
    and the window, and none of the copy's identity, StatVars or name."""
    tools = _tools_by_name()
    ctx = _FakeToolContext()
    seed = _seed_with_stat_vars()
    save_seeded_sources(ctx.state, [seed])

    result = _propose_seeded(
        tools,
        ctx,
        pinning_decision(_SEED_SPEC),
        [requirement(source_spec_id=_SEED_SPEC, period=Period(start=2010, end=2023))],
    )

    assert result["outcome"] == Outcome.OK.value, result
    copy_id, _ = windowed_identity(_SEED_IDENTITY, _WINDOW_2010_2023)
    assert json.loads(ctx.state[SEEDED_SOURCES_STATE_KEY]) == {
        "seeds": [seed.model_dump(mode="json")],
        "copies": [{"windowed_from": _SEED_SPEC, "window": _WINDOW_2010_2023}],
    }
    assert copy_id not in ctx.state[SEEDED_SOURCES_STATE_KEY]


def test_a_copy_reads_back_as_its_seed_over_its_window() -> None:
    """The copy's spec_id and identity are the seed's over the window, and its
    StatVars and name are the seed's, rebuilt where the state is read."""
    seed = _seed_with_stat_vars()

    (held, copy) = seeded_sources(_stored_copy_state([seed]))

    copy_id, copy_identity = windowed_identity(_SEED_IDENTITY, _WINDOW_2010_2023)
    assert held == seed
    assert copy == SeededSource(
        source=SelectedSource(
            spec_id=copy_id,
            sentence=seed.source.sentence,
            has_data=None,
            stat_vars=("Var_a", "Var_b"),
            identity=copy_identity,
        ),
        windowed_from=_SEED_SPEC,
    )


def test_a_copy_whose_seed_is_missing_is_refused() -> None:
    """A copy is nothing without its seed, so a list holding one without the other
    is refused where it is read and where it is written."""
    seed, other = _seed_with_stat_vars(), _seeded({**_SEED_IDENTITY, "seed": "population"})

    with pytest.raises(ValueError, match=rf"names the seed '{_SEED_SPEC}', which the stored"):
        seeded_sources(_stored_copy_state([other]))
    (_, copy) = seeded_sources(_stored_copy_state([seed]))
    state: dict[str, Any] = {}
    with pytest.raises(ValueError, match=rf"names the seed '{_SEED_SPEC}', which the sources"):
        save_seeded_sources(state, [other, copy])
    with pytest.raises(ValueError, match="names the seed"):
        save_dropped_seeds(state, [copy])
    assert state == {}


def test_a_copy_that_is_not_its_seed_over_its_window_is_refused() -> None:
    """The state rebuilds a copy from its seed, so a copy stating anything else
    would read back changed, and is refused where it is written."""
    seed = _seed_with_stat_vars()
    (_, copy) = seeded_sources(_stored_copy_state([seed]))
    renamed = copy.model_copy(
        update={"source": copy.source.model_copy(update={"sentence": "another name"})}
    )

    with pytest.raises(ValueError, match="is not its seed"):
        save_seeded_sources({}, [seed, renamed])


def test_a_drop_of_a_stored_seed_takes_its_copies_into_the_dropped_state() -> None:
    """A seed and its copy read back from the state go to the dropped datasets
    together, stored the same way, while another seed stays."""
    seed, other = _seed_with_stat_vars(), _seeded({**_SEED_IDENTITY, "seed": "population"})
    state = _stored_copy_state([seed, other])
    held = seeded_sources(state)

    kept, taken = split_dropped(held, {_SEED_SPEC})
    save_seeded_sources(state, kept)
    save_dropped_seeds(state, taken)

    (_, copy) = taken
    assert copy.windowed_from == _SEED_SPEC
    assert seeded_sources(state) == [other]
    assert dropped_seeds(state) == [seed, copy]
    assert json.loads(state[DROPPED_SEEDS_STATE_KEY])["copies"] == [
        {"windowed_from": _SEED_SPEC, "window": _WINDOW_2010_2023}
    ]


def test_one_rule_says_which_dataset_the_turn_dropped() -> None:
    """A pin and a source are asked the same question. The dropped seed and its
    dataset over any other years are dropped; a seed the turn still holds is
    kept, though it is the same dataset over other years."""
    dropped, held = _seeded(), _over(2015, 2024)
    elsewhere = _over(2000, 2005)
    unrelated = _seeded({**_SEED_IDENTITY, "seed": "population"})
    state: dict[str, Any] = {}
    save_seeded_sources(state, [held])
    save_dropped_seeds(state, [dropped])

    def asked(item: SeededSource, *, with_identity: bool) -> bool:
        identity = item.source.identity if with_identity else None
        return names_dropped_seed(state, spec_id=item.source.spec_id, identity=identity)

    assert asked(dropped, with_identity=False) is True
    assert asked(dropped, with_identity=True) is True
    assert asked(held, with_identity=False) is False
    assert asked(held, with_identity=True) is False
    assert asked(elsewhere, with_identity=True) is True
    assert asked(elsewhere, with_identity=False) is False
    assert asked(unrelated, with_identity=True) is False


def test_a_held_seed_cannot_be_pinned_over_the_window_of_its_dropped_sibling(
    pinning_decision: Callable[..., Decision],
    requirement: Callable[..., DatasetRequirement],
) -> None:
    """The seed over 2015 to 2024 is dropped and the seed over every year is held.
    Pinning the held one over 2015 to 2024 moves the pin to the dropped seed's own
    spec_id, which the plan is checked for before any copy is recorded: it is
    refused, and no copy of the held seed takes the dropped spec_id."""
    tools = _tools_by_name()
    ctx = _FakeToolContext()
    held, dropped = _seeded(), _over(2015, 2024)
    save_seeded_sources(ctx.state, [held])
    save_dropped_seeds(ctx.state, [dropped])

    result = _propose_seeded(
        tools,
        ctx,
        pinning_decision(_SEED_SPEC),
        [requirement(source_spec_id=_SEED_SPEC, period=Period(start=2015, end=2024))],
    )

    assert result["code"] == ProblemCode.SEED_DROPPED.value
    assert [item.source.spec_id for item in seeded_sources(ctx.state)] == [_SEED_SPEC]
    assert PLAN_STATE_KEY not in ctx.state


def test_two_seeds_over_one_dataset_pinned_over_one_period_share_the_first_seed_s_copy(
    pinning_decision: Callable[..., Decision],
    requirement: Callable[..., DatasetRequirement],
) -> None:
    """Both pins move to the one spec_id the dataset has over that period, and the
    copy recorded for it is the first seed's, with the first seed's StatVars."""
    tools = _tools_by_name()
    ctx = _FakeToolContext()
    first, second = _over(2000, 2005), _over(2015, 2024)
    first = first.model_copy(
        update={"source": first.source.model_copy(update={"stat_vars": ("Var_first",)})}
    )
    second = second.model_copy(
        update={"source": second.source.model_copy(update={"stat_vars": ("Var_second",)})}
    )
    save_seeded_sources(ctx.state, [first, second])
    period = Period(start=2010, end=2010)

    result = tools["propose_plan"].func(
        question="q",
        answer_mode=AnswerMode.SERIES,
        decisions=[
            pinning_decision(first.source.spec_id, slot="a"),
            pinning_decision(second.source.spec_id, slot="b"),
        ],
        evidence_requirements=[
            requirement(
                id="one", source_spec_id=first.source.spec_id, decision_ids=["a"], period=period
            ),
            requirement(
                id="two", source_spec_id=second.source.spec_id, decision_ids=["b"], period=period
            ),
        ],
        tool_context=ctx,
    )

    assert result["outcome"] == Outcome.OK.value, result
    (copied,) = [item for item in seeded_sources(ctx.state) if item.windowed_from]
    assert copied.windowed_from == first.source.spec_id
    assert copied.source.stat_vars == ("Var_first",)


def test_propose_plan_takes_a_pin_on_a_seed_whose_sibling_over_other_years_was_dropped(
    pinning_decision: Callable[..., Decision],
    requirement: Callable[..., DatasetRequirement],
) -> None:
    tools = _tools_by_name()
    ctx = _FakeToolContext()
    held = _over(2015, 2024)
    save_seeded_sources(ctx.state, [held])
    save_dropped_seeds(ctx.state, [_seeded()])

    result = _propose_seeded(
        tools,
        ctx,
        pinning_decision(held.source.spec_id),
        [requirement(source_spec_id=held.source.spec_id, period=Period(start=2015, end=2024))],
    )

    assert result["outcome"] == Outcome.OK.value, result


@pytest.mark.parametrize("seeded", [True, False], ids=["seeded", "only-in-the-request"])
def test_propose_plan_traces_a_user_election_to_a_seed_the_seed_node_stored(
    seeded: bool,
    pinning_decision: Callable[..., Decision],
    requirement: Callable[..., DatasetRequirement],
) -> None:
    """A dataset the user handed over from Search is the user's election once
    `harness.graph`'s `seed` node has stored it (`SEEDED_SOURCES_STATE_KEY`).
    The request's own seeds (`SEEDS_STATE_KEY`) are not: a turn that fell
    back to the question never stored them."""
    tools = _tools_by_name()
    ctx = _FakeToolContext()
    window = {"kind": "range", "start_year": 2015, "end_year": 2024}
    identity = {**_SEED_IDENTITY, "requested_window": window}
    spec_id = spec_id_for(canonicalize(identity))
    if seeded:
        save_seeded_sources(ctx.state, [_seeded(identity)])
    else:
        ctx.state[SEEDS_STATE_KEY] = json.dumps([{"spec": {"spec_id": spec_id}}])

    result = _propose_seeded(
        tools, ctx, pinning_decision(spec_id), [requirement(source_spec_id=spec_id)]
    )

    if seeded:
        assert result["outcome"] == Outcome.OK.value
        plan = load_plan(ctx.state)
        assert plan is not None
        assert plan.evidence_requirements[0].source_spec_id == spec_id  # type: ignore[union-attr]
    else:
        assert result["code"] == ProblemCode.ELECTION_UNTRACEABLE.value
        assert PLAN_STATE_KEY not in ctx.state


def test_propose_plan_pins_a_seed_over_the_requirements_period(
    pinning_decision: Callable[..., Decision],
    requirement: Callable[..., DatasetRequirement],
) -> None:
    """A seed over every year with data is pinned instead to the seed over the
    requirement's own period, the spec_id `get_data`'s `scope.window` would
    mint, so the plan's bare retrieval of it returns only those years."""
    tools = _tools_by_name()
    ctx = _FakeToolContext()
    save_seeded_sources(ctx.state, [_seeded()])

    result = _propose_seeded(
        tools,
        ctx,
        pinning_decision(_SEED_SPEC),
        [requirement(source_spec_id=_SEED_SPEC, period=Period(start=2024, end=2024))],
    )

    assert result["outcome"] == Outcome.OK.value
    windowed, identity = windowed_identity(
        _SEED_IDENTITY, {"kind": "range", "start_year": 2024, "end_year": 2024}
    )
    plan = load_plan(ctx.state)
    assert plan is not None
    assert plan.evidence_requirements[0].source_spec_id == windowed  # type: ignore[union-attr]
    assert plan.decisions[0].fixes == DecisionFixes(source_spec_id=windowed)
    (seed, copy) = seeded_sources(ctx.state)
    assert seed.source.spec_id == _SEED_SPEC
    assert copy.windowed_from == _SEED_SPEC
    assert copy.source.spec_id == windowed
    assert copy.source.identity == identity


def test_the_copy_of_a_seed_over_another_period_records_nothing_search_stated_for_the_seed(
    pinning_decision: Callable[..., Decision],
    requirement: Callable[..., DatasetRequirement],
) -> None:
    """Search states a seed's coverage, entities, axes and counts for the seed's own
    years. The copy over the requirement's period records what resolves it, the
    spec_id and identity Search answers under and the seed's elected StatVars, with
    the seed's sentence as a name, and none of those facts."""
    answered = json.loads(
        (Path(__file__).parents[1] / "fixtures" / "resubmit_education_oda_kenya.json").read_text()
    )["specs"]
    seed = source_from_spec_block(answered["2015-2023"])
    assert seed.coverage_window == Period(start=2015, end=2023)
    assert seed.coverage_kind == "exact"
    assert seed.has_data is True
    assert seed.entities
    assert seed.open_entity_axes
    assert seed.open_axes
    assert seed.stat_var_member_count == 89
    tools = _tools_by_name()
    ctx = _FakeToolContext()
    save_seeded_sources(ctx.state, [SeededSource(source=seed)])

    result = _propose_seeded(
        tools,
        ctx,
        pinning_decision(seed.spec_id),
        [requirement(source_spec_id=seed.spec_id, period=Period(start=2010, end=2023))],
    )

    assert result["outcome"] == Outcome.OK.value
    (_, copy) = seeded_sources(ctx.state)
    # The spec_id and identity Search itself answers for 2010-2023.
    assert copy == SeededSource(
        source=SelectedSource(
            spec_id=answered["2010-2023"]["spec_id"],
            sentence=seed.sentence,
            has_data=None,
            stat_vars=seed.stat_vars,
            identity=answered["2010-2023"]["definition"]["identity"],
        ),
        windowed_from=seed.spec_id,
    )
    assert copy.source.coverage_window is None
    assert copy.source.coverage_kind is None
    assert copy.source.entities == ()
    assert copy.source.open_entity_axes is None
    assert copy.source.stat_var_member_count is None


def test_an_amendment_naming_the_seed_again_repins_it_over_the_new_period(
    pinning_decision: Callable[..., Decision],
    requirement: Callable[..., DatasetRequirement],
) -> None:
    """The planner reasons about the seed's own spec_id: a patch that pins it
    again, over a narrower period, amends the windowed plan cleanly."""
    tools = _tools_by_name()
    ctx = _FakeToolContext()
    save_seeded_sources(ctx.state, [_seeded()])
    first = _propose_seeded(
        tools,
        ctx,
        pinning_decision(_SEED_SPEC),
        [requirement(source_spec_id=_SEED_SPEC, period=Period(start=2015, end=2024))],
    )

    result = _propose_seeded(
        tools,
        ctx,
        pinning_decision(_SEED_SPEC),
        [requirement(source_spec_id=_SEED_SPEC, period=Period(start=2022, end=2024))],
        amends_version=int(first["version"]),  # type: ignore[call-overload]
    )

    assert result["outcome"] == Outcome.OK.value
    windowed, _ = windowed_identity(
        _SEED_IDENTITY, {"kind": "range", "start_year": 2022, "end_year": 2024}
    )
    plan = load_plan(ctx.state)
    assert plan is not None
    assert plan.evidence_requirements[0].source_spec_id == windowed  # type: ignore[union-attr]
    assert [item.source.spec_id for item in seeded_sources(ctx.state)][-1] == windowed


def test_drafts_and_amendments_across_planner_entries_hold_only_the_copy_the_plan_pins(
    pinning_decision: Callable[..., Decision],
    requirement: Callable[..., DatasetRequirement],
    decision: DecisionFactory,
) -> None:
    """One seed and one requirement, its years changed by a first draft,
    amendments and a new plan over three planner entries, more proposals than
    one entry allows. After each, the state holds the seed and the one copy the
    recorded plan pins; once no requirement pins the seed, the seed alone."""
    tools = _tools_by_name()
    ctx = _FakeToolContext()
    save_seeded_sources(ctx.state, [_seeded()])
    version: int | None = None
    proposals = 0
    for entry in range(3):
        # What `harness.planner_seam` does on each fresh entry into the planner.
        save_plan_rounds(ctx.state, 0)
        for start in range(2010 + 3 * entry, 2013 + 3 * entry):
            result = tools["propose_plan"].func(
                question="q",
                answer_mode=AnswerMode.SERIES,
                decisions=[pinning_decision(_SEED_SPEC)],
                evidence_requirements=[
                    requirement(source_spec_id=_SEED_SPEC, period=Period(start=start, end=2024))
                ],
                amends_version=None if start == 2013 else version,
                starts_new_plan=start == 2013,
                tool_context=ctx,
            )
            assert result["outcome"] == Outcome.OK.value, result
            version = int(result["version"])  # type: ignore[call-overload]
            proposals += 1
            windowed, _ = windowed_identity(
                _SEED_IDENTITY, {"kind": "range", "start_year": start, "end_year": 2024}
            )
            assert [
                (item.source.spec_id, item.windowed_from) for item in seeded_sources(ctx.state)
            ] == [(_SEED_SPEC, None), (windowed, _SEED_SPEC)]
    assert proposals > MAX_PLAN_ROUNDS

    unpinned = tools["propose_plan"].func(
        decisions=[decision(slot="scope")],
        evidence_requirements=[requirement(id="other", decision_ids=["scope"])],
        removed_decisions=["measure"],
        removed_evidence_requirements=["annual_debt_service"],
        amends_version=version,
        tool_context=ctx,
    )

    assert unpinned["outcome"] == Outcome.OK.value, unpinned
    assert seeded_sources(ctx.state) == [_seeded()]


def test_one_decision_cannot_fix_a_seed_over_two_periods(
    pinning_decision: Callable[..., Decision],
    requirement: Callable[..., DatasetRequirement],
) -> None:
    tools = _tools_by_name()
    ctx = _FakeToolContext()
    save_seeded_sources(ctx.state, [_seeded()])

    result = _propose_seeded(
        tools,
        ctx,
        pinning_decision(_SEED_SPEC),
        [
            requirement(source_spec_id=_SEED_SPEC, period=Period(start=2024, end=2024)),
            requirement(
                id="earlier", source_spec_id=_SEED_SPEC, period=Period(start=2015, end=2015)
            ),
        ],
    )

    assert result["code"] == ProblemCode.INVALID_PLAN.value
    assert "its own decision" in str(result["message"])
    assert PLAN_STATE_KEY not in ctx.state


@pytest.mark.parametrize("resolved_by", [ResolvedBy.USER, ResolvedBy.PLANNER_DEFAULT])
def test_propose_plan_refuses_a_pin_on_a_dataset_the_turn_dropped(
    resolved_by: ResolvedBy,
    pinning_decision: Callable[..., Decision],
    requirement: Callable[..., DatasetRequirement],
) -> None:
    """The turn dropped the seed and told the user the plan leaves it out, so a pin on
    it is refused whoever the decision says resolved it, and the refusal does not offer
    `planner_default` as a way round."""
    tools = _tools_by_name()
    ctx = _FakeToolContext()
    save_dropped_seeds(ctx.state, [_seeded()])

    result = _propose_seeded(
        tools,
        ctx,
        pinning_decision(_SEED_SPEC, resolved_by=resolved_by),
        [requirement(source_spec_id=_SEED_SPEC)],
    )

    assert result["code"] == ProblemCode.SEED_DROPPED.value
    assert result["detail"] == {"decisions": ["measure"], "requirements": ["annual_debt_service"]}
    assert "Do not plan that dataset again in this turn" in str(result["message"])
    assert "planner_default" not in str(result["message"])
    assert PLAN_STATE_KEY not in ctx.state


def test_a_patch_that_carries_a_dropped_pin_forward_is_refused(
    pinning_decision: Callable[..., Decision],
    requirement: Callable[..., DatasetRequirement],
    decision: DecisionFactory,
) -> None:
    """The recorded plan pinned the seed over its period before the turn dropped it. A
    patch that leaves that requirement in is refused, and one that removes it lands."""
    tools = _tools_by_name()
    ctx = _FakeToolContext()
    save_seeded_sources(ctx.state, [_seeded()])
    other = requirement(id="other", decision_ids=["scope"])
    first = tools["propose_plan"].func(
        question="q",
        answer_mode=AnswerMode.SERIES,
        decisions=[pinning_decision(_SEED_SPEC), decision(slot="scope")],
        evidence_requirements=[
            requirement(source_spec_id=_SEED_SPEC, period=Period(start=2024, end=2024)),
            other,
        ],
        tool_context=ctx,
    )
    assert first["outcome"] == Outcome.OK.value
    recorded = ctx.state[PLAN_STATE_KEY]
    # What `harness.graph._drop_seeds_from_turn` does: the seed and its copy
    # over the plan's period move from the seeded sources to the dropped ones.
    save_dropped_seeds(ctx.state, seeded_sources(ctx.state))
    save_seeded_sources(ctx.state, [])

    carried = tools["propose_plan"].func(
        evidence_requirements=[other.model_copy(update={"measure": "debt stock"})],
        amends_version=int(first["version"]),
        tool_context=ctx,
    )

    assert carried["code"] == ProblemCode.SEED_DROPPED.value
    assert carried["detail"] == {"decisions": ["measure"], "requirements": ["annual_debt_service"]}
    assert ctx.state[PLAN_STATE_KEY] == recorded

    removed = tools["propose_plan"].func(
        removed_decisions=["measure"],
        removed_evidence_requirements=["annual_debt_service"],
        amends_version=int(first["version"]),
        tool_context=ctx,
    )

    assert removed["outcome"] == Outcome.OK.value
    plan = load_plan(ctx.state)
    assert plan is not None
    assert [item.id for item in plan.evidence_requirements] == ["other"]


# --- the envelope's own spec_id, synthesized into a Decision ---------------------------


class _NodeContext:
    """The part of a graph node's context `_drop_seeds_from_turn` reads: its state."""

    def __init__(self, state: dict[str, Any]) -> None:
        self.state = state


def test_windowed_copy_keeps_statvars_and_a_drop_blocks_its_pin(
    pinning_decision: Callable[..., Decision],
    requirement: Callable[..., DatasetRequirement],
    decision: DecisionFactory,
) -> None:
    """A plan pins a seed through its copy over the plan's period. The copy posts the
    seed's StatVars, and dropping the seed drops the copy with it, so a patch that
    carries the copy's pin forward is refused."""
    tools = _tools_by_name()
    ctx = _FakeToolContext()
    seed = _seeded()
    seed = seed.model_copy(
        update={"source": seed.source.model_copy(update={"stat_vars": ("Var_a", "Var_b")})}
    )
    kept = _seeded({**_SEED_IDENTITY, "seed": "population"})
    save_seeded_sources(ctx.state, [seed, kept])
    other = requirement(id="other", decision_ids=["scope"])
    first = tools["propose_plan"].func(
        question="q",
        answer_mode=AnswerMode.SERIES,
        decisions=[pinning_decision(_SEED_SPEC), decision(slot="scope")],
        evidence_requirements=[
            requirement(source_spec_id=_SEED_SPEC, period=Period(start=2024, end=2024)),
            other,
        ],
        tool_context=ctx,
    )
    assert first["outcome"] == Outcome.OK.value
    (copy,) = [item for item in seeded_sources(ctx.state) if item.windowed_from]
    assert copy.source.spec_id != _SEED_SPEC
    assert copy.source.stat_vars == ("Var_a", "Var_b")

    ctx.state["resolve_envelope"] = json.dumps(
        {
            "seeds": [
                {"spec_id": _SEED_SPEC, "sentence": "education aid", "operations": []},
                {"spec_id": kept.source.spec_id, "sentence": "population", "operations": []},
            ]
        }
    )
    _drop_seeds_from_turn(cast(Any, _NodeContext(ctx.state)), {_SEED_SPEC}, surface="studio")

    assert json.loads(ctx.state["dropped_untold"]) == [{"name": "education aid", "reason": None}]
    assert {item.source.spec_id for item in dropped_seeds(ctx.state)} == {
        _SEED_SPEC,
        copy.source.spec_id,
    }
    assert [item.source.spec_id for item in seeded_sources(ctx.state)] == [kept.source.spec_id]
    carried = tools["propose_plan"].func(
        evidence_requirements=[other.model_copy(update={"measure": "debt stock"})],
        amends_version=int(first["version"]),
        tool_context=ctx,
    )
    assert carried["code"] == ProblemCode.SEED_DROPPED.value
    plan = load_plan(ctx.state)
    assert plan is not None
    assert plan.version == int(first["version"])


def test_propose_plan_synthesizes_the_decision_a_verbatim_envelope_pin_needs(
    decision: DecisionFactory, requirements: RequirementPair
) -> None:
    """A single-part, `status: definite` resolve names one
    spec_id with no candidates to weigh, `harness.planner_seam` renders it as
    an allowed pin, and the model pins it citing only its `measure` decision
    -- no decision fixes the source. Without `ENVELOPE_SPEC_STATE_KEY` telling
    `propose_plan` that this is the exact id the envelope exposed, `Plan`'s
    own `_check_pin_rests_on_a_fixing_decision` refuses this outright (see
    `test_a_pinned_source_needs_a_decision_that_fixes_it`,
    `planner/test_plan.py`); with it set, as `harness.planner_seam.__call__`
    always sets it, `propose_plan` writes the decision itself instead of
    losing a round to the refusal.
    """
    tools = _tools_by_name()
    ctx = _FakeToolContext()
    ctx.state[ENVELOPE_SPEC_STATE_KEY] = SPEC
    dataset, derived = requirements

    result = tools["propose_plan"].func(
        question="q",
        answer_mode=AnswerMode.COMPUTED,
        decisions=[decision()],
        evidence_requirements=[dataset.model_copy(update={"source_spec_id": SPEC}), derived],
        tool_context=ctx,
    )

    assert result["outcome"] == Outcome.OK.value
    plan = load_plan(ctx.state)
    assert plan is not None
    synthesized = [d for d in plan.decisions if d.fixes and d.fixes.source_spec_id == SPEC]
    assert len(synthesized) == 1
    assert synthesized[0].resolved_by is ResolvedBy.PLANNER_DEFAULT
    assert synthesized[0].slot in plan.requirement(dataset.id).decision_ids


def test_propose_plan_patches_the_citation_on_a_round_that_already_knows_the_synthesized_decision(
    decision: DecisionFactory, requirements: RequirementPair
) -> None:
    """A correction round resubmits the pinned requirement by id, which
    replaces it wholesale and drops its citation -- `propose_plan`'s OK
    result never told the model the synthesized decision's slot, so it has
    nothing to cite. `previous.decisions` already carries that decision from
    the first round, so `_synthesize_source_decisions` must still add the
    citation to `decision_ids` even though the decision itself is already
    known and only the append to `decisions` is skipped.
    """
    tools = _tools_by_name()
    ctx = _FakeToolContext()
    ctx.state[ENVELOPE_SPEC_STATE_KEY] = SPEC
    dataset, derived = requirements

    first = tools["propose_plan"].func(
        question="q",
        answer_mode=AnswerMode.COMPUTED,
        decisions=[decision()],
        evidence_requirements=[dataset.model_copy(update={"source_spec_id": SPEC}), derived],
        tool_context=ctx,
    )
    assert first["outcome"] == Outcome.OK.value
    plan = load_plan(ctx.state)
    assert plan is not None
    synthesized = next(d for d in plan.decisions if d.fixes and d.fixes.source_spec_id == SPEC)

    second = tools["propose_plan"].func(
        question="q",
        answer_mode=AnswerMode.COMPUTED,
        decisions=[decision()],
        evidence_requirements=[dataset.model_copy(update={"source_spec_id": SPEC}), derived],
        amends_version=plan.version,
        tool_context=ctx,
    )

    assert second["outcome"] == Outcome.OK.value
    amended = load_plan(ctx.state)
    assert amended is not None
    assert [d.slot for d in amended.decisions if d.fixes and d.fixes.source_spec_id == SPEC] == [
        synthesized.slot
    ]
    assert synthesized.slot in amended.requirement(dataset.id).decision_ids


def test_propose_plan_still_refuses_a_pin_the_envelope_never_exposed(
    decision: DecisionFactory, requirements: RequirementPair
) -> None:
    """The synthesis is scoped to exactly the id `harness.planner_seam` last
    exposed (`ENVELOPE_SPEC_STATE_KEY`) -- never every pin a call happens to
    make. With no state key set at all (an older session, or a caller of
    this seam's pair directly), a pin with no citing decision is refused
    exactly as it always was; this is what proves the fix adds a decision
    rather than loosening `_check_pin_rests_on_a_fixing_decision` itself.
    """
    tools = _tools_by_name()
    ctx = _FakeToolContext()
    dataset, derived = requirements

    result = tools["propose_plan"].func(
        question="q",
        answer_mode=AnswerMode.COMPUTED,
        decisions=[decision()],
        evidence_requirements=[dataset.model_copy(update={"source_spec_id": SPEC}), derived],
        tool_context=ctx,
    )

    assert result["outcome"] == Outcome.INVALID_INPUT.value
    assert "cites no decision that fixes that same source_spec_id" in result["message"]
    assert PLAN_STATE_KEY not in ctx.state


def test_propose_plan_never_overrides_the_models_own_source_decision(
    decision: DecisionFactory, requirements: RequirementPair
) -> None:
    """A model that already cited a decision fixing this exact pin -- its own
    reasoning, `resolved_by` whatever it chose -- gets no synthesized
    stand-in: `_synthesize_source_decisions` only fires where no cited
    decision already fixes the pin.
    """
    tools = _tools_by_name()
    ctx = _FakeToolContext()
    ctx.state[ENVELOPE_SPEC_STATE_KEY] = SPEC
    dataset, derived = requirements
    source = decision(
        slot="source",
        options=["OECD DAC", "World Bank IDS"],
        chosen="OECD DAC",
        resolved_by=ResolvedBy.PLANNER_DEFAULT,
        fixes=DecisionFixes(source_spec_id=SPEC),
    )

    result = tools["propose_plan"].func(
        question="q",
        answer_mode=AnswerMode.COMPUTED,
        decisions=[decision(), source],
        evidence_requirements=[
            dataset.model_copy(
                update={"source_spec_id": SPEC, "decision_ids": ("measure", "source")}
            ),
            derived,
        ],
        tool_context=ctx,
    )

    assert result["outcome"] == Outcome.OK.value
    plan = load_plan(ctx.state)
    assert plan is not None
    assert [d.slot for d in plan.decisions] == ["measure", "source"]


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
        _clarify_answer_event(SPEC),
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
        decisions=[elected],
        evidence_requirements=[dataset.model_copy(update={"source_spec_id": SPEC}), derived],
        tool_context=ctx,
    )

    assert result["outcome"] == Outcome.OK.value
    assert PLAN_STATE_KEY in ctx.state


def test_propose_plan_refuses_a_clarify_candidate_offered_but_not_chosen(
    decision: DecisionFactory, requirements: RequirementPair
) -> None:
    """SPEC was offered in the clarify round, but the user's own answer
    picked POPULATION_SPEC -- membership in what was offered is not itself
    an election."""
    tools = _tools_by_name()
    ctx = _FakeToolContext()
    ctx.session.events = [
        _clarify_ask_event(
            [{"spec_id": SPEC, "label": "a"}, {"spec_id": POPULATION_SPEC, "label": "b"}]
        ),
        _clarify_answer_event(POPULATION_SPEC),
    ]
    dataset, derived = requirements
    elected = decision(
        chosen="total debt stock",
        resolved_by=ResolvedBy.USER,
        fixes=DecisionFixes(source_spec_id=SPEC),
    )

    result = tools["propose_plan"].func(
        question="q",
        decisions=[elected],
        evidence_requirements=[dataset.model_copy(update={"source_spec_id": SPEC}), derived],
        tool_context=ctx,
    )

    assert result["outcome"] == Outcome.REFUSED.value
    assert result["code"] == ProblemCode.ELECTION_UNTRACEABLE.value
    assert result["detail"]["spec_ids"] == [SPEC]


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
        _clarify_answer_event(SPEC),
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
        "RankRequirement",
        "ShareOfRequirement",
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
    # Every property of every model `propose_plan`'s declaration carries has
    # a description, so the model sees what each field means without it
    # having to live in `_PLANNING` prose. `_undescribed_properties` walks
    # `$defs`, the shape ADK actually sends the model.
    tools = _tools_by_name()

    declaration = tools["propose_plan"]._get_declaration()

    assert declaration is not None
    schema = declaration.parameters_json_schema
    assert schema is not None
    assert _undescribed_properties(schema) == []


def test_propose_operations_declaration_describes_every_field() -> None:
    # `propose_operations` is `translator.agent`'s tool, built the same way
    # -- a Pydantic-typed parameter ADK turns into a JSON schema -- so its
    # `Operation` and `Translation` fields need the same coverage.
    from compiler.surface import ProgressCallback, ToolCallResult, ToolSchema  # noqa: PLC0415
    from translator.agent import build_translator_core  # noqa: PLC0415

    class _UnreachableSurface:
        async def list_tools(self) -> list[ToolSchema]:
            raise NotImplementedError

        async def call_tool(
            self,
            *,
            tool: ToolId,
            arguments: dict[str, object],
            on_progress: ProgressCallback | None = None,
        ) -> ToolCallResult:
            raise NotImplementedError

    core = build_translator_core(surface=_UnreachableSurface())
    tool = FunctionTool(core.propose_operations)

    declaration = tool._get_declaration()

    assert declaration is not None
    schema = declaration.parameters_json_schema
    assert schema is not None
    assert _undescribed_properties(schema) == []


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


def test_submit_plan_without_a_plan_returns_a_problem() -> None:
    tools = _tools_by_name()

    result = tools["submit_plan"].func(version=1, tool_context=_FakeToolContext())

    assert result["code"] == ProblemCode.PLAN_MISSING.value
    assert "summary" not in result


def test_submit_plan_after_rejection_is_blocked(
    decision: DecisionFactory, requirements: RequirementPair
) -> None:
    tools = _tools_by_name()
    ctx = _FakeToolContext()
    _propose(tools, ctx, decision(), requirements)
    _translate(ctx)
    tools["reject_plan"].func(tool_context=ctx)

    result = tools["submit_plan"].func(version=1, tool_context=ctx)

    assert result["code"] == ProblemCode.PLAN_REJECTED.value
    assert result["action"]["call"] == "propose_plan"
    assert "summary" not in result


@pytest.mark.parametrize("submitted_version", [1, 99], ids=["older", "newer"])
def test_submit_plan_rejects_any_version_but_the_current_one(
    submitted_version: int, decision: DecisionFactory, requirements: RequirementPair
) -> None:
    tools = _tools_by_name()
    ctx = _FakeToolContext()
    _propose(tools, ctx, decision(), requirements)
    _propose(  # amends to version 2
        tools, ctx, decision(chosen="total debt stock", resolved_by=ResolvedBy.USER), requirements
    )
    _translate(ctx)

    result = tools["submit_plan"].func(version=submitted_version, tool_context=ctx)

    assert result["code"] == ProblemCode.VERSION_STALE.value
    assert result["detail"] == {"current": 2, "given": submitted_version}
    assert "summary" not in result


def test_submit_plan_can_be_called_again_without_changing_anything(
    decision: DecisionFactory, requirements: RequirementPair
) -> None:
    # Approval is the graph's business now, not this tool's: `submit_plan`
    # calls no transition of its own, so calling it twice against the same
    # version answers the same summary and leaves the plan exactly where it
    # was, rather than refusing a second call the way an approval would.
    tools = _tools_by_name()
    ctx = _FakeToolContext()
    _propose(tools, ctx, decision(), requirements)
    _translate(ctx)

    first = tools["submit_plan"].func(version=1, tool_context=ctx)
    second = tools["submit_plan"].func(version=1, tool_context=ctx)

    assert first == second
    stored = load_plan(ctx.state)
    assert stored is not None
    assert stored.status == PlanStatus.PENDING_APPROVAL


def test_submit_plan_with_a_pending_referral_takes_priority_over_version_staleness(
    decision: DecisionFactory, requirements: RequirementPair
) -> None:
    # The order `submit_plan` checks in: a pending referral is reported
    # before a stale version is, since settling the referral is what a fresh
    # proposal would do anyway.
    tools = _tools_by_name()
    ctx = _FakeToolContext()
    _propose(tools, ctx, decision(), requirements)
    _translate(ctx)
    dataset, _derived = requirements
    save_referral(
        ctx.state,
        Referral(kind=ReferralKind.NO_DATA, reason="no data found", requirement_id=dataset.id),
    )

    result = tools["submit_plan"].func(version=99, tool_context=ctx)

    assert result["code"] == ProblemCode.REFERRAL_PENDING.value


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
        "`unsupported_ask` or `requirement_unmeetable` means the requirement cannot be met as "
        "stated: tell the user what cannot be done and ask, through `adk_request_input`, "
        "before proposing anything"
    ) in instruction


def test_the_instruction_tells_the_planner_to_read_the_recommendation() -> None:
    # `Referral.recommendation` rides along in `_execution_blocked`'s detail
    # whenever the translator ranked a candidate; the planner has to be told
    # to read it, or a grounded pick sits unread while the planner invents
    # its own rationale for `resolved_by=planner_default`.
    assert "recommendation" in _SERVED_INSTRUCTION
    assert "names the candidate to prefer" in _SERVED_INSTRUCTION
    assert "rather than one you invent" in _SERVED_INSTRUCTION


def test_the_instruction_still_calls_the_recommendation_advice_not_an_order() -> None:
    assert "It is advice, not an instruction" in _SERVED_INSTRUCTION
    assert "your own reason for departing from what was recommended" in _SERVED_INSTRUCTION


def test_the_instruction_still_requires_a_decision_between_the_candidates() -> None:
    assert (
        "A source is pinned only "
        "through such a decision between the candidates the translator offered, "
        "never picked from the candidates directly"
    ) in _SERVED_INSTRUCTION


def test_the_instruction_still_lists_every_candidate_in_options() -> None:
    assert "not only the one chosen" in _SERVED_INSTRUCTION


def test_the_instruction_still_reserves_resolved_by_user_for_the_question_saying_so() -> None:
    assert (
        "never `resolved_by=user` for a choice the question does not say the user made"
    ) in _SERVED_INSTRUCTION


def test_the_instruction_states_a_party_nobody_named_in_across() -> None:
    # A stand-in such as "World" binds the identity to that one entity, so a
    # rank over it ranks one row; a share of the world total still needs one.
    # Each column the envelope's `open_entity_axes` can carry is tied to the
    # party word the plan states.
    instruction = _SERVED_INSTRUCTION

    assert "state the party in `across` and name no stand-in entity such as `'World'`" in (
        instruction
    )
    assert "A world figure set beside a named entity still names `'World'`." in instruction
    # The party word alone: a qualified population is not one an open axis holds.
    assert (
        "`across` is the party word alone, such as `'donor'`, `'recipient'` or `'country'`, "
        "and covers every member the source holds: for a narrower population, such as a "
        "region, an income group or a donor type, name its members in `entities` instead, "
        "and name no entity in the party `across` covers."
    ) in instruction
    assert (
        "The envelope's `open_entity_axes` lists the columns left open: `origin` is the giving "
        "party, `destination` the receiving party, `observationAbout` the place."
    ) in instruction


def test_the_summary_carries_nothing_the_plan_record_already_states(
    decision: DecisionFactory, requirements: RequirementPair
) -> None:
    """`submit_plan`'s summary is what a caller cannot read off the record.

    `harness.studio_translate` builds the user's plan card from the plan record
    through `harness.plan_display`, and `dfo_translator` reads the record out
    of session state, so a copy of the decisions, requirements, sources or
    steps here is a second description of one plan that can disagree with the
    first. Nothing on the served path reads a restated copy of the plan back
    to the user.
    """
    tools = _tools_by_name()
    ctx = _FakeToolContext()
    _propose(tools, ctx, decision(), requirements)
    _translate(ctx)

    result = tools["submit_plan"].func(version=1, tool_context=ctx)

    assert set(result["summary"]) == {"version", "revision", "warnings"}


def test_the_instruction_asks_for_no_restatement_of_the_plan() -> None:
    # The plan card is on the user's screen. A closing turn that retypes every
    # decision, requirement and step is output tokens spent to produce text
    # nothing reads.
    instruction = planner_instruction(has_translator=True)

    assert "none left out" not in instruction
    assert "each decision by its slot" not in instruction
    assert "do not restate its" in instruction


def test_the_instruction_never_asks_the_planner_to_retype_the_plan() -> None:
    """`dfo_translator` reads the plan from session state, so `request` carries none of it.

    `translate` (`translator.agent`) takes no plan (`requirement_id`,
    `choice`, `rationale`, `decision_id` only), and `run_translate` reaches
    the draft through `current_draft(tool_context)` ->
    `load_plan(tool_context.state)`. An instruction telling the planner to
    paste the plan's `question`, `decisions` and `evidence_requirements` into
    `request` therefore sends a copy the translator never reads, and a model
    retyping structured content can mistype it. This asserts the wording
    directly.
    `test_planner_seam.py::test_a_short_dfo_translator_request_reaches_the_wire_and_no_plan_rides_along`
    covers the wire, confirming a short request actually goes out and the
    translator still reaches the plan.
    """
    instruction = planner_instruction(has_translator=True)

    assert "never retype the plan's" in instruction
    assert "reads the recorded plan itself, from session state" in instruction
    assert "carrying the plan's" not in instruction
    assert "the only way `dfo_translator` sees the plan" not in instruction


# --- Construction -------------------------------------------------------------------------


def test_create_planner_agent_wires_name_model_and_tools() -> None:
    agent = create_planner_agent(model_name="gemini-3.8-flash")

    assert isinstance(agent, LlmAgent)
    assert agent.name == "dfo_planner"
    assert agent.model == "gemini-3.8-flash"
    assert {tool.name for tool in agent.tools} == {  # type: ignore[union-attr]
        "adk_request_input",
        "propose_plan",
        "reject_plan",
        "submit_plan",
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


# The party words the instruction and `Entity.role` name, one pair per flow.
_ROLE_PARTIES = ("recipient", "donor", "debtor", "creditor", "importer", "exporter")


def test_planner_instruction_asks_for_each_entitys_party_role() -> None:
    # An entity with no `role` on a flow measure reaches `search` with no
    # side, and dfo-apis answers `role_not_resolved`, which the translator
    # sends back as a `no_data` referral. The words the instruction names
    # are the plan's own party lexicon (`planner.parties.flow_side`);
    # `from` and `to` stay out of it, since they name the money's movement.
    instruction = planner_instruction(has_translator=True, has_skills=False)
    for party in _ROLE_PARTIES:
        assert f"`'{party}'`" in instruction
        assert flow_side(party) is not None
    assert "never directions such as `'from'` or `'to'`" in instruction


def test_the_entity_role_field_names_parties_not_directions() -> None:
    # The field description is model-facing through `propose_plan`'s schema,
    # so it carries the same party words as the instruction, and no
    # directional example the compiler cannot place.
    description = Entity.model_fields["role"].description or ""
    for party in _ROLE_PARTIES:
        assert f"'{party}'" in description
    assert "'from'" not in description
    assert "'to'" not in description


def test_planner_instruction_carries_no_sentinel_range() -> None:
    # With no `window` on the envelope, the period is a decision like any
    # other, so "time window" stays in the opening decision list and the
    # instruction states no sentinel fallback range for it. "source" is not
    # in that same list any more (`_PLANNING`'s own opening enumeration):
    # unlike the window, a verbatim source pin gets its citing `Decision`
    # synthesized by `planner.agent._synthesize_source_decisions`, so the
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


def test_a_planner_with_no_translator_is_not_told_to_call_one() -> None:
    # The standalone planner is what this package's own eval set scores. ADK
    # raises `Tool 'dfo_translator' not found` rather than answering the model,
    # so an instruction naming a sub-agent that is not attached kills the run
    # mid-turn -- which is exactly what it did, one or two cases per eval run.
    standalone = create_planner_agent(model_name="m")
    paired = create_planner_agent(
        model_name="m",
        translator=LlmAgent(name="dfo_translator", model="m", mode="single_turn"),
    )

    assert "dfo_translator" not in str(standalone.instruction)
    assert "dfo_translator" in str(paired.instruction)


def test_a_planner_with_no_translator_refuses_submission_without_naming_one(
    decision: DecisionFactory, requirements: RequirementPair
) -> None:
    tools = _tools_by_name(has_translator=False)
    ctx = _FakeToolContext()
    _propose(tools, ctx, decision(), requirements)

    result = tools["submit_plan"].func(version=1, tool_context=ctx)

    assert result["code"] == ProblemCode.WORKFLOW_MISSING.value
    assert "dfo_translator" not in result["message"]
    assert "action" not in result


def test_create_planner_agent_attaches_the_translator_it_is_given() -> None:
    translator = LlmAgent(name="dfo_translator", model="m", mode="single_turn")

    agent = create_planner_agent(model_name="gemini-flash-latest", translator=translator)

    assert [sub_agent.name for sub_agent in agent.sub_agents] == ["dfo_translator"]


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
        decisions=list(base.decisions),
        evidence_requirements=list(base.evidence_requirements),
        tool_context=ctx,
    )
    assert first["outcome"] == Outcome.OK.value
    rounds = load_plan_rounds(ctx.state)

    longer = base.decisions[0].model_copy(update={"rationale": "r" * 10})
    refused = tools["propose_plan"].func(
        question=base.question,
        decisions=[longer],
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
MAX_REPORTED_ERRORS = planner_agent._MAX_REPORTED_ERRORS


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
        decisions=[oversized],
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
        decisions=[decision()],
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
        decision().model_dump(mode="json") | {"slot": f"slot_{index}", "description": "d" * 600}
        for index in range(MAX_PLAN_DECISIONS)
    ]

    refused = tools["propose_plan"].func(
        question="q",
        decisions=failing,
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
