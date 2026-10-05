"""Proves each translator eval case's expected outcome is reachable, with no model.

`evals/agents/harness_translator_v1/outcome_metric.py`'s `workflow_outcome`
metric asserts a case's expected `propose_operations`/`return_to_planner`
response against a real run's, but nothing checks that the expectation itself
is one the translator's own core functions can actually produce against the
canned discovery stub -- an author could type a spec_id that discovery never
returns, or a `group_by` the requirement doesn't state, and every model-free
gate would stay green.

This module closes that gap. For every case in `harness-translator-v1.
evalset.json` and `harness-pair-v1.evalset.json`, it loads the seeded plan
(or, for the one pair case with none, builds the plan its rubric describes),
drives the real `discover_source` and `propose_operations`/`return_to_planner`
functions (`translator.agent.build_translator_core`, the same three
`translate` drives directly, in Python, with no model turn of its own)
against `harness_translator_v1.discovery_stub.EvalSurface` on a fake
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

from harness_translator_v1.discovery_stub import EvalSurface  # noqa: E402
from harness_translator_v1.outcome_metric import matches_subset  # noqa: E402
from planner.agent import build_planner_tools  # noqa: E402
from planner.operations import (  # noqa: E402
    AggregateOperation,
    CompareOperation,
    GrowthOperation,
    Operation,
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
    ShareOfRequirement,
    inputs_of,
)
from translator.agent import build_translator_core  # noqa: E402
from translator.mechanical import TranslatorCore  # noqa: E402
from translator.referral import ReferralKind  # noqa: E402

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
    `.session.events` are read (the latter by `planner.agent.
    _elected_spec_ids`, which `propose_plan` checks a `resolved_by=user`
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
            share_of); anything else raises `NotImplementedError`, since no
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
            left=side(requirement.left),
            right=side(requirement.right),
        )
    if isinstance(requirement, ShareOfRequirement):
        return ShareOfOperation(
            id=op_id,
            requirement_id=requirement.id,
            numerator=side(requirement.numerator),
            denominator=side(requirement.denominator),
        )
    raise NotImplementedError(f"mirror does not handle {type(requirement).__name__}")


async def _discover_definite(
    core: TranslatorCore, context: FakeToolContext, requirement_id: str
) -> str:
    """Run `discover_source` for `requirement_id` and return the spec_id a `definite` answer named.

    Fails the test outright if the answer is not `definite`, since every
    mapping case's requirements are ones the canned stub resolves outright.
    """
    result = await core.discover_source(requirement_id=requirement_id, tool_context=context)
    assert result["outcome"] == "ok", result
    assert result["status"] == "definite", result
    (source,) = result["sources"]
    return source["spec_id"]


async def _propose_mapping_case(case_id: str) -> None:
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

    result = await core.propose_operations(translation=translation, tool_context=context)

    assert result["outcome"] == "ok", result
    assert matches_subset(_expected_response(case)["response"], result)


@pytest.mark.parametrize(
    "case_id",
    [
        "maps-an-annual-total-onto-aggregate",
        "maps-a-window-onto-growth",
        "compares-two-entities",
        "maps-a-per-head-figure-onto-share_of",
    ],
)
async def test_a_mapping_cases_expected_outcome_is_reachable(case_id: str) -> None:
    await _propose_mapping_case(case_id)


async def test_the_constant_price_referral_is_reachable() -> None:
    case = _TRANSLATOR_CASES["refers-a-constant-price-requirement"]
    plan = _plan_from(case)
    context = _context_holding(plan)
    core = _core()

    (dataset,) = [r for r in plan.evidence_requirements if isinstance(r, DatasetRequirement)]
    spec_id = await _discover_definite(core, context, dataset.id)
    translation = mirror(plan, {dataset.id: spec_id})

    proposed = await core.propose_operations(translation=translation, tool_context=context)

    assert proposed["outcome"] == "invalid_input", proposed
    issue_codes = {issue["code"] for issue in proposed["detail"]["issues"]}
    assert "requirement_unmeetable" in issue_codes or "source_unusable" in issue_codes
    action = proposed["action"]
    assert action["call"] == "return_to_planner"

    referred = core.return_to_planner(
        kind=ReferralKind(action["arguments"]["kind"]),
        reason="no operation restates a series in another year's prices",
        requirement_id=action["arguments"].get("requirement_id"),
        decision_id=action["arguments"].get("decision_id"),
        tool_context=context,
    )

    assert matches_subset(_expected_response(case)["response"], referred)


async def test_the_several_sources_referral_is_reachable() -> None:
    case = _TRANSLATOR_CASES["refers-when-discovery-finds-several-sources"]
    plan = _plan_from(case)
    context = _context_holding(plan)
    core = _core()

    (dataset,) = [r for r in plan.evidence_requirements if isinstance(r, DatasetRequirement)]
    result = await core.discover_source(requirement_id=dataset.id, tool_context=context)

    assert result["outcome"] == "ok", result
    assert result["status"] == "candidates", result
    action = result["action"]
    assert action["call"] == "return_to_planner"
    assert action["arguments"]["kind"] == "source_ambiguous"

    referred = core.return_to_planner(
        kind=ReferralKind(action["arguments"]["kind"]),
        reason="two sources report aid disbursements to Kenya",
        requirement_id=action["arguments"]["requirement_id"],
        tool_context=context,
    )

    assert matches_subset(_expected_response(case)["response"], referred)


async def test_the_pair_sets_translated_case_expected_outcome_is_reachable() -> None:
    # `translates-before-inviting-approval` seeds no plan -- the live planner
    # drafts its own -- so this builds the plan the case's own
    # `pair-states-evidence-requirements` rubric describes: a dataset
    # requirement naming Kenya with a period starting in 2015, fed by a
    # calculation, each citing the decision that fixed it. It is the same
    # shape as harness-translator-v1's own annual-total case, which is what
    # the case's `note` says the workflow_outcome expectation was written
    # against.
    case = _PAIR_CASES["translates-before-inviting-approval"]
    plan = _plan_from(_TRANSLATOR_CASES["maps-an-annual-total-onto-aggregate"])
    context = _context_holding(plan)
    core = _core()

    spec_ids = {
        requirement.id: await _discover_definite(core, context, requirement.id)
        for requirement in plan.evidence_requirements
        if isinstance(requirement, DatasetRequirement)
    }
    translation = mirror(plan, spec_ids)

    result = await core.propose_operations(translation=translation, tool_context=context)

    assert result["outcome"] == "ok", result
    assert matches_subset(_expected_response(case)["response"], result)


async def test_the_pair_sets_unmeetable_case_expected_outcome_is_reachable() -> None:
    case = _PAIR_CASES["takes-an-unmeetable-requirement-back-to-the-user"]
    plan = _plan_from(case)
    context = _context_holding(plan)
    core = _core()

    (dataset,) = [r for r in plan.evidence_requirements if isinstance(r, DatasetRequirement)]
    spec_id = await _discover_definite(core, context, dataset.id)
    translation = mirror(plan, {dataset.id: spec_id})

    proposed = await core.propose_operations(translation=translation, tool_context=context)

    assert proposed["outcome"] == "invalid_input", proposed
    action = proposed["action"]
    assert action["call"] == "return_to_planner"

    referred = core.return_to_planner(
        kind=ReferralKind(action["arguments"]["kind"]),
        reason="no operation restates a series in another year's prices",
        requirement_id=action["arguments"].get("requirement_id"),
        decision_id=action["arguments"].get("decision_id"),
        tool_context=context,
    )

    assert matches_subset(_expected_response(case)["response"], referred)


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
            decisions=list(plan.decisions),
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


def test_plan_status_is_draft_for_every_seeded_case() -> None:
    # A sanity check on this module's own fixtures: every plan these tests
    # drive `discover_source`/`propose_operations` against has to be a
    # draft, or `current_draft` refuses it before any of the assertions
    # above get to run.
    for case in {**_TRANSLATOR_CASES, **_PAIR_CASES}.values():
        seeded = case.get("session_input", {}).get("state", {}).get("planner:plan")
        if seeded is not None:
            assert Plan.model_validate_json(seeded).status is PlanStatus.DRAFT
