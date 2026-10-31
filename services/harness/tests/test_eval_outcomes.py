"""Proves each eval case's expected outcome is reachable, with no model.

`evals/agents/harness_translator_v1/outcome_metric.py`'s `workflow_outcome`
metric asserts a case's expected tool response against a real run's, but
nothing there checks that the expectation itself is one the real tools can
produce against the canned discovery stub -- an author could type a spec_id
that discovery never returns, or a `group_by` the requirement doesn't state.
The translation cases run no model at all, so this module is the only thing
that checks them.

This module closes that gap. For every case in `harness-translator-v1.
evalset.json`, it loads the seeded plan and drives the real `discover_source`
and `propose_operations`/`return_to_planner` functions
(`translator.tool.build_translator_core`, the same three translation drives
directly, in Python) against `harness_translator_v1.discovery_stub.EvalSurface` on a fake
`ToolContext`, and asserts the result against the case's own expected
`tool_responses` entry with `matches_subset` -- the same subset matcher
`workflow_outcome` uses, imported rather than restated, so the two can never
silently disagree about what "matches" means.

A mapping case's translation is built by `mirror`: one `retrieve` per dataset
requirement, naming the spec_id discovery found for it, and one operation of
the same kind per derived requirement, naming its calculation fields and
resolving its inputs through the requirement id -> operation id map `mirror`
builds as it walks the plan's requirements in order. Terminals are the
operations for requirements nothing else reads. This is model-free: the
translator's own judgement (which operation carries out which decision) is
not what is being tested here, only that a translation shaped exactly like
the plan's requirements is one `propose_operations` accepts and produces the
outcome the eval case expects.

A referral case cannot be driven the same way -- there is no "correct"
translation to mirror, since the point is that none exists. Instead this
module drives discovery (and, for a case whose data is fine but a
requirement is unmeetable, a `propose_operations` attempt) far enough to read
the `action` the tools themselves recommend, and asserts that action names
the referral kind and target the case expects, then calls `return_to_planner`
with exactly that action to reach the outcome and check it the same way as a
mapping case.

`harness-pair-v1.evalset.json` states the `translation` a `propose_plan`
answers the planner with, so its cases run the translation itself
(`build_translate`) on the seeded plan, or, for the one case with none, on the plan its rubric
describes.

`harness-planner-v2.evalset.json` seeds no plan -- the live planner drafts
its own -- and states its expectations on `propose_plan` and `reject_plan`
instead. Those are driven the same way, through the real planner tools on a
fake `ToolContext`, with the decisions and requirements of a seeded translator
case standing in for what the model would draft: an accepted draft, version 2
on a second proposal of the same question, a recorded rejection.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
_AGENTS_DIR = REPO_ROOT / "evals" / "agents"
if str(_AGENTS_DIR) not in sys.path:
    sys.path.insert(0, str(_AGENTS_DIR))

from data_mcp.node_lookup import described_variable_text  # noqa: E402
from harness_translator_v1.discovery_stub import EvalSurface  # noqa: E402
from harness_translator_v1.outcome_metric import matches_subset  # noqa: E402
from planner.agent import build_planner_tools  # noqa: E402
from planner.operations import (  # noqa: E402
    AggregateOperation,
    CompareOperation,
    GrowthOperation,
    Operation,
    QuotientOfOperation,
    RetrieveOperation,
    ShareOfOperation,
    Side,
    Translation,
)
from planner.plan import (  # noqa: E402
    PLAN_STATE_KEY,
    AggregateRequirement,
    ComparisonRequirement,
    DatasetRequirement,
    EvidenceRequirement,
    GrowthRequirement,
    Plan,
    PlanStatus,
    QuotientRequirement,
    ShareOfRequirement,
    inputs_of,
)
from planner.workflow import ArtifactKind, Binding, ToolId, WorkflowStep  # noqa: E402
from translator.mechanical import TranslatorCore  # noqa: E402
from translator.referral import ReferralKind  # noqa: E402
from translator.tool import build_translate, build_translator_core  # noqa: E402

from .decision_drafts import drafts  # noqa: E402
from .translator.conftest import FakeCall  # noqa: E402

_TRANSLATOR_DIR = REPO_ROOT / "evals" / "agents" / "harness_translator_v1"
_PAIR_DIR = REPO_ROOT / "evals" / "agents" / "harness_pair_v1"
_PLANNER_DIR = REPO_ROOT / "evals" / "agents" / "harness_planner_v2"


def _cases(path: Path) -> dict[str, dict[str, Any]]:
    eval_set = json.loads(path.read_text())
    return {case["eval_id"]: case for case in eval_set["eval_cases"]}


_TRANSLATOR_CASES = _cases(_TRANSLATOR_DIR / "harness-translator-v1.evalset.json")
_PAIR_CASES = _cases(_PAIR_DIR / "harness-pair-v1.evalset.json")
_PLANNER_CASES = _cases(_PLANNER_DIR / "harness-planner-v2.evalset.json")


class FakeToolContext:
    """A minimal stand-in for ADK's `ToolContext`: `.state` and
    `.session.events` are read (the latter by `planner.call.
    elected_spec_ids`, which `propose_plan` checks a `resolved_by=user`
    source pin against)."""

    def __init__(self) -> None:
        self.state: dict[str, str] = {}
        self.session = SimpleNamespace(events=[])


def _plan_from(case: dict[str, Any]) -> Plan:
    seeded = case["session_input"]["state"]["planner:plan"]
    return Plan.model_validate_json(seeded)


def _expected_response(case: dict[str, Any], turn: int = 0) -> dict[str, Any]:
    (response,) = case["conversation"][turn]["intermediate_data"]["tool_responses"]
    return response


def _context_holding(plan: Plan) -> FakeToolContext:
    context = FakeToolContext()
    context.state[PLAN_STATE_KEY] = plan.model_dump_json()
    return context


def _core() -> TranslatorCore:
    return build_translator_core(surface=EvalSurface())


async def _translate(context: FakeToolContext) -> dict[str, Any]:
    """The `translation` a `propose_plan` answers the planner with on `context`'s plan."""
    return await build_translate(surface=EvalSurface())(FakeCall(context.state))


def mirror(plan: Plan, spec_ids: dict[str, str]) -> Translation:
    """Build the translation that mirrors `plan`'s requirements one to one.

    One `retrieve` per dataset requirement, naming `spec_ids[requirement.id]`;
    one operation of the same kind per derived requirement, naming that
    requirement in `requirement_id` and repeating its calculation fields
    exactly. Inputs are resolved through the requirement id -> operation id
    map this function builds as it walks `plan.evidence_requirements`, which
    it assumes lists a requirement before anything that reads it -- true of
    every plan this module drives, since `planner.plan` requires the
    requirement graph to be a DAG and every eval case lists dataset
    requirements first.

    Args:
        plan: The plan to mirror. Every requirement kind it states has to be
            one this function handles (dataset, aggregate, growth, comparison,
            share_of, quotient_of); anything else raises `NotImplementedError`, since no
            eval case here needs it.
        spec_ids: The spec_id discovery found for each dataset requirement,
            keyed by requirement id.

    Returns:
        A `Translation` naming one operation per requirement, terminals set
        to the operations for requirements nothing else reads.
    """
    op_id_by_requirement: dict[str, str] = {}
    operations: list[Operation] = []
    read_ids = {
        input_id
        for requirement in plan.evidence_requirements
        for input_id in inputs_of(requirement)
    }

    def _side(requirement_side: Any) -> Side:
        return Side(
            input=op_id_by_requirement[requirement_side.input_id], entity=requirement_side.entity
        )

    for requirement in plan.evidence_requirements:
        op_id = f"op_{requirement.id}"
        operation = _build_operation(op_id, requirement, spec_ids, op_id_by_requirement, _side)
        operations.append(operation)
        op_id_by_requirement[requirement.id] = op_id

    terminal = tuple(
        op_id_by_requirement[requirement.id]
        for requirement in plan.evidence_requirements
        if requirement.id not in read_ids
    )
    return Translation(operations=tuple(operations), terminal=terminal)


def _build_operation(
    op_id: str,
    requirement: EvidenceRequirement,
    spec_ids: dict[str, str],
    op_id_by_requirement: dict[str, str],
    side: Any,
) -> Operation:
    if isinstance(requirement, DatasetRequirement):
        return RetrieveOperation(
            id=op_id, requirement_id=requirement.id, spec_id=spec_ids[requirement.id]
        )
    if isinstance(requirement, AggregateRequirement):
        return AggregateOperation(
            id=op_id,
            requirement_id=requirement.id,
            input=op_id_by_requirement[requirement.input_id],
            reduction=requirement.reduction,
            group_by=requirement.group_by,
        )
    if isinstance(requirement, GrowthRequirement):
        return GrowthOperation(
            id=op_id,
            requirement_id=requirement.id,
            input=op_id_by_requirement[requirement.input_id],
            window_start=requirement.window_start,
            window_end=requirement.window_end,
        )
    if isinstance(requirement, ComparisonRequirement):
        return CompareOperation(
            id=op_id,
            requirement_id=requirement.id,
            left=side(requirement.subtrahend),
            right=side(requirement.minuend),
        )
    if isinstance(requirement, ShareOfRequirement):
        return ShareOfOperation(
            id=op_id,
            requirement_id=requirement.id,
            numerator=side(requirement.numerator),
            denominator=side(requirement.denominator),
            axis=requirement.axis,
            denominator_scope=requirement.denominator_scope,
        )
    if isinstance(requirement, QuotientRequirement):
        return QuotientOfOperation(
            id=op_id,
            requirement_id=requirement.id,
            numerator=side(requirement.numerator),
            denominator=side(requirement.denominator),
            axis=requirement.axis,
            denominator_axis=requirement.denominator_axis,
        )
    raise NotImplementedError(f"mirror does not handle {type(requirement).__name__}")


async def _discover_definite(
    core: TranslatorCore, context: FakeToolContext, requirement_id: str
) -> str:
    """Run `discover_source` for `requirement_id` and return the spec_id a `definite` answer named.

    Fails the test outright if the answer is not `definite`, since every
    mapping case's requirements are ones the canned stub resolves outright.
    """
    result = await core.discover_source(requirement_id=requirement_id, state=context.state)
    assert result["outcome"] == "ok", result
    assert result["status"] == "definite", result
    (source,) = result["sources"]
    return source["spec_id"]


async def _propose_mapping_case(case_id: str) -> FakeToolContext:
    """Propose the mirror of `case_id`'s plan and check it against the case's expectation.

    Returns the context, whose plan state now holds the compiled workflow.
    """
    case = _TRANSLATOR_CASES[case_id]
    plan = _plan_from(case)
    context = _context_holding(plan)
    core = _core()

    spec_ids = {
        requirement.id: await _discover_definite(core, context, requirement.id)
        for requirement in plan.evidence_requirements
        if isinstance(requirement, DatasetRequirement)
    }
    translation = mirror(plan, spec_ids)

    result = await core.propose_operations(translation=translation, state=context.state)

    assert result["outcome"] == "ok", result
    assert matches_subset(_expected_response(case)["response"], result)
    return context


def _last_step(context: FakeToolContext) -> WorkflowStep:
    """The final step of the workflow `propose_operations` compiled into `context`."""
    workflow = Plan.model_validate_json(context.state[PLAN_STATE_KEY]).workflow
    assert workflow is not None
    return workflow.steps[-1]


@pytest.mark.parametrize(
    "case_id",
    [
        "maps-an-annual-total-onto-aggregate",
        "maps-a-window-onto-growth",
        "compares-two-entities",
        "maps-a-per-head-figure-with-a-unit-less-denominator",
        "maps-a-keyed-per-head-figure-onto-quotient_of",
        "maps-a-keyed-share-onto-share_of",
    ],
)
async def test_a_mapping_cases_expected_outcome_is_reachable(case_id: str) -> None:
    await _propose_mapping_case(case_id)


async def test_the_stubs_describe_answers_the_node_envelope_the_harness_reads() -> None:
    surface = EvalSurface()
    describe = ToolId(server="data-mcp", name="describe")

    answer = await surface.call_tool(
        tool=describe, arguments={"ids": ["DebtServicePaid", "country/KEN"]}
    )

    assert not answer.is_error
    stat_var, country = (answer.structured_content or {})["nodes"]
    assert (stat_var["kind"], country["kind"]) == ("stat_var", "country")
    assert described_variable_text(stat_var) == "DebtServicePaid, as the graph defines it."
    assert described_variable_text(country) is None

    spec = await surface.call_tool(tool=describe, arguments={"ids": [f"spec_{'d' * 32}"]})
    assert spec.is_error
    assert (spec.structured_content or {})["code"] == "node_not_held"


async def test_the_constant_price_referral_is_reachable() -> None:
    case = _TRANSLATOR_CASES["refers-a-constant-price-requirement"]
    plan = _plan_from(case)
    context = _context_holding(plan)
    core = _core()

    (dataset,) = [r for r in plan.evidence_requirements if isinstance(r, DatasetRequirement)]
    spec_id = await _discover_definite(core, context, dataset.id)
    translation = mirror(plan, {dataset.id: spec_id})

    proposed = await core.propose_operations(translation=translation, state=context.state)

    assert proposed["outcome"] == "invalid_input", proposed
    issue_codes = {issue["code"] for issue in proposed["detail"]["issues"]}
    assert "source_unusable" in issue_codes
    action = proposed["action"]
    assert action["call"] == "return_to_planner"

    referred = core.return_to_planner(
        kind=ReferralKind(action["arguments"]["kind"]),
        reason="the source states no price basis, so it cannot meet a constant-price ask",
        requirement_id=action["arguments"].get("requirement_id"),
        state=context.state,
    )

    assert matches_subset(_expected_response(case)["response"], referred)


async def test_a_per_head_figure_compiles_to_one_quotient_of_pinning_the_numerators_unit() -> None:
    context = await _propose_mapping_case("maps-a-per-head-figure-with-a-unit-less-denominator")

    step = _last_step(context)

    # The population side declares no unit, so only the numerator is pinned.
    assert step.tool.name == "quotient_of"
    assert step.args == {"numerator_unit": "USD"}
    assert step.bindings == {
        "frame_ref": Binding(from_step="op_annual_debt_service", output=ArtifactKind.FRAME_REF),
        "denominator_frame_ref": Binding(from_step="op_population", output=ArtifactKind.FRAME_REF),
    }


async def test_a_keyed_per_head_figure_divides_each_recipient_by_its_own_population() -> None:
    context = await _propose_mapping_case("maps-a-keyed-per-head-figure-onto-quotient_of")

    step = _last_step(context)

    assert step.tool.name == "quotient_of"
    assert step.args == {
        "numerator_by": ["recipient"],
        "denominator_by": {"recipient": "observationAbout"},
        "numerator_unit": "USD",
    }


async def test_a_keyed_share_divides_each_debtors_service_by_its_own_exports() -> None:
    context = await _propose_mapping_case("maps-a-keyed-share-onto-share_of")

    step = _last_step(context)

    assert step.tool.name == "share_of"
    assert step.args == {"numerator_by": ["debtor"], "denominator_by": {"debtor": "debtor"}}


async def test_the_share_of_form_of_a_per_head_figure_is_referred_for_repair() -> None:
    # A share_of divides a part by its whole in one unit. The per-head plan's
    # sides hold US dollars and a unit-less count, so the compiler sends it
    # back to the planner with the verb that does divide them.
    case = _TRANSLATOR_CASES["maps-a-per-head-figure-with-a-unit-less-denominator"]
    plan = _plan_from(case)
    plan = plan.model_copy(
        update={
            "evidence_requirements": tuple(
                ShareOfRequirement(
                    id=requirement.id,
                    decision_ids=requirement.decision_ids,
                    numerator=requirement.numerator,
                    denominator=requirement.denominator,
                )
                if isinstance(requirement, QuotientRequirement)
                else requirement
                for requirement in plan.evidence_requirements
            )
        }
    )
    context = _context_holding(plan)
    core = _core()

    spec_ids = {
        requirement.id: await _discover_definite(core, context, requirement.id)
        for requirement in plan.evidence_requirements
        if isinstance(requirement, DatasetRequirement)
    }
    proposed = await core.propose_operations(
        translation=mirror(plan, spec_ids), state=context.state
    )

    assert proposed["outcome"] == "invalid_input", proposed
    assert [issue["code"] for issue in proposed["detail"]["issues"]] == ["share_units_differ"]
    action = proposed["action"]
    assert action["call"] == "return_to_planner"

    referred = core.return_to_planner(
        kind=ReferralKind(action["arguments"]["kind"]),
        reason="the share's sides hold different units",
        requirement_id=action["arguments"]["requirement_id"],
        constraint=action["arguments"]["constraint"],
        repair=action["arguments"]["repair"],
        state=context.state,
    )

    assert referred["outcome"] == "referred", referred
    referral = referred["referral"]
    assert referral["kind"] == "requirement_needs_repair"
    assert referral["requirement_id"] == "debt_service_per_head"
    assert referral["constraint"] == "share_units_differ"
    assert "quotient_of" in referral["repair"]


async def test_the_several_sources_referral_is_reachable() -> None:
    case = _TRANSLATOR_CASES["refers-when-discovery-finds-several-sources"]
    plan = _plan_from(case)
    context = _context_holding(plan)
    core = _core()

    (dataset,) = [r for r in plan.evidence_requirements if isinstance(r, DatasetRequirement)]
    result = await core.discover_source(requirement_id=dataset.id, state=context.state)

    assert result["outcome"] == "ok", result
    assert result["status"] == "candidates", result
    action = result["action"]
    assert action["call"] == "return_to_planner"
    assert action["arguments"]["kind"] == "source_ambiguous"

    referred = core.return_to_planner(
        kind=ReferralKind(action["arguments"]["kind"]),
        reason="two sources report aid disbursements to Kenya",
        requirement_id=action["arguments"]["requirement_id"],
        state=context.state,
    )

    assert matches_subset(_expected_response(case)["response"], referred)


async def test_the_pair_sets_translated_case_expected_outcome_is_reachable() -> None:
    # `translates-before-inviting-approval` seeds no plan -- the live planner
    # drafts its own -- so this translates the plan the case's own
    # `pair-states-evidence-requirements` rubric describes: a dataset
    # requirement naming Kenya with a period starting in 2015, fed by a
    # calculation, each citing the decision that fixed it. It is the same
    # shape as harness-translator-v1's own annual-total case.
    case = _PAIR_CASES["translates-before-inviting-approval"]
    plan = _plan_from(_TRANSLATOR_CASES["maps-an-annual-total-onto-aggregate"])

    answer = await _translate(_context_holding(plan))

    assert matches_subset(_expected_response(case)["response"], answer), answer


async def test_the_pair_sets_unmeetable_case_expected_outcome_is_reachable() -> None:
    case = _PAIR_CASES["takes-an-unmeetable-requirement-back-to-the-user"]

    answer = await _translate(_context_holding(_plan_from(case)))

    assert matches_subset(_expected_response(case)["response"], answer), answer


def test_the_planner_sets_expected_outcomes_are_reachable() -> None:
    # The planner set seeds no plan, so the decisions and requirements of a
    # seeded translator case stand in for what the model drafts. One
    # proposal is the first turn of every case; a second of the same
    # question is the amend turn, which the case pins at version 2; a
    # rejection is the reject turn.
    plan = _plan_from(_TRANSLATOR_CASES["maps-an-annual-total-onto-aggregate"])
    tools = {tool.name: tool for tool in build_planner_tools()}
    context = FakeToolContext()

    def propose(amends_version: int | None = None) -> dict[str, Any]:
        # A fresh context would be the first draft, but one context makes the
        # second call a patch of the first (the amend turn), so that call has
        # to state the version it patches -- the version the call before it
        # answered with. A patch that restates the plan's own decisions and
        # requirements by key leaves the plan exactly as the case seeded it.
        return tools["propose_plan"].func(
            question=plan.question,
            answer_mode=plan.answer_mode,
            decisions=drafts(plan.decisions),
            evidence_requirements=list(plan.evidence_requirements),
            tool_context=context,
            amends_version=amends_version,
        )

    first = propose()
    amended = propose(amends_version=first["version"])
    rejected = tools["reject_plan"].func(tool_context=context)

    def expected(case_id: str, turn: int) -> dict[str, Any]:
        return _expected_response(_PLANNER_CASES[case_id], turn)["response"]

    assert first["outcome"] == "ok", first
    assert matches_subset(expected("two-dataset-comparison", 0), first)
    assert matches_subset(expected("amended-mid-execution-reversions", 0), first)
    assert matches_subset(expected("amended-mid-execution-reversions", 1), amended)
    assert matches_subset(expected("rejected-plan-blocks-execution", 0), first)
    assert matches_subset(expected("rejected-plan-blocks-execution", 1), rejected)
    assert matches_subset(expected("adversarial-execute-before-approval", 0), first)


@pytest.mark.parametrize("case_id", sorted(_TRANSLATOR_CASES))
def test_a_seeded_plan_is_one_propose_plan_records(case_id: str) -> None:
    # A seeded plan reaches the translator without passing `propose_plan`,
    # which holds a draft to rules `Plan` alone does not, such as the words
    # an entity's `role` takes. Each case's plan is proposed as the model
    # would state it, so a case rests on no plan the planner could not write.
    plan = _plan_from(_TRANSLATOR_CASES[case_id])
    tools = {tool.name: tool for tool in build_planner_tools()}

    proposed = tools["propose_plan"].func(
        question=plan.question,
        answer_mode=plan.answer_mode,
        decisions=drafts(plan.decisions),
        evidence_requirements=[
            requirement.model_dump(mode="json") for requirement in plan.evidence_requirements
        ],
        declined=list(plan.declined),
        tool_context=FakeToolContext(),
    )

    assert proposed["outcome"] == "ok", proposed


def test_plan_status_is_draft_for_every_seeded_case() -> None:
    # A sanity check on this module's own fixtures: every plan these tests
    # drive `discover_source`/`propose_operations` against has to be a
    # draft, or `current_draft` refuses it before any of the assertions
    # above get to run.
    for case in {**_TRANSLATOR_CASES, **_PAIR_CASES}.values():
        seeded = case.get("session_input", {}).get("state", {}).get("planner:plan")
        if seeded is not None:
            assert Plan.model_validate_json(seeded).status is PlanStatus.DRAFT
