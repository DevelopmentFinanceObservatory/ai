"""Tests for `compiler.compile`: from a translation to the workflow that runs it.

`compile_plan` is pure, so every case states a plan, a translation, a surface
and the sources discovery recorded, and reads the result back. No model, no
server.

A plan's own requirements have to satisfy `planner.plan`'s own rules (every
dataset requirement read under `answer_mode="computed"`, every decision cited)
independently of whatever translation is compiled against them --
`compile_plan` never attaches a
translation to a `Plan`'s own `translation`/`workflow` fields, so
`Plan._translation_and_workflow_agree` never runs here. Where a test's
operation is meant to satisfy one of the plan's own requirements exactly
(the happy paths, and most single-issue negative cases), the requirement
mirrors the operation's calculation fields field for field, so
`planner.proof.mismatches` reports nothing and the test's own issue is the
only one. Where a requirement's own validators forbid mirroring a
deliberately degenerate operation (comparing a side with itself), the
assertion checks the issue is present among others rather than that it is
the only one.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Callable, Mapping, Sequence
from itertools import permutations
from pathlib import Path
from typing import Any

import pytest
from common.unit_refs import PERCENT_UNIT
from compiler.capabilities import CAPABILITIES, ToolCapability
from compiler.compile import CompileIssueCode, CompileResult, compile_plan
from compiler.sources import member_digest, source_from_spec_block
from compiler.surface import ToolSchema
from data_mcp.definition_writes import windowed_identity
from dfo_frame_grammar.frame import Frame
from dfo_frame_grammar.grammar import TIME_PERIOD_COLUMN
from planner.operations import (
    AggregateOperation,
    CompareOperation,
    FilterOperation,
    GroupBy,
    GrowthOperation,
    JoinOperation,
    Operation,
    QuotientOfOperation,
    RankOperation,
    Reduction,
    RetrieveOperation,
    ShareOfOperation,
    Side,
    SumOfOperation,
    Translation,
)
from planner.plan import (
    AggregateRequirement,
    AnswerMode,
    ComparisonRequirement,
    DatasetRequirement,
    Decision,
    DecisionDraft,
    DecisionFixes,
    Entity,
    EvidenceRequirement,
    FilterRequirement,
    GrowthRequirement,
    JoinRequirement,
    Plan,
    PlanStatus,
    QuotientRequirement,
    RankRequirement,
    RequirementSide,
    ShareOfRequirement,
    SumOfRequirement,
    Unit,
    settle_decisions,
)
from planner.records import (
    DenominatorScope,
    EntityIn,
    EvidenceIssueCode,
    GrowthChange,
    Period,
    PeriodThreshold,
    PlaceLevel,
    PlaceType,
    RankAmong,
    RankDirection,
    StatVarReading,
    ValueThreshold,
)
from planner.shapes import pooling_remedy
from planner.workflow import (
    ArtifactKind,
    Binding,
    OpenEntityAxis,
    ReadingSetBy,
    ResolvedEntity,
    ResolvedGroup,
    SelectedSource,
)
from pydantic import TypeAdapter, ValidationError
from translator.mechanical import build_translation

SPEC_KENYA = "spec_" + "a" * 32
SPEC_OTHER = "spec_" + "b" * 32

DecisionFactory = Callable[..., Decision]
RequirementFactory = Callable[..., EvidenceRequirement]
DerivedRequirementFactory = Callable[..., EvidenceRequirement]
SourceFactory = Callable[..., SelectedSource]

Code = CompileIssueCode


def _pinned(stat_var: str, fields: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """`fields` as a source whose identity pins `stat_var`, the one StatVar `search`
    serves for it: two such sources overlap only when they pin one StatVar. The
    identity members `fields` states win over the pin."""
    fields = dict(fields or {})
    identity = {"shape": {"kind": "stat_var", "id": stat_var}} | fields.pop("identity", {})
    return {
        "served_count": 1,
        "shape_member_digest": member_digest([stat_var]),
        **fields,
        "identity": identity,
    }


def _draft(
    decisions: Sequence[Decision],
    requirements: Sequence[EvidenceRequirement],
    *,
    answer_mode: AnswerMode = AnswerMode.COMPUTED,
) -> Plan:
    return Plan(
        question="How much does Kenya pay to service its debt each year?",
        answer_mode=answer_mode,
        decisions=decisions,
        evidence_requirements=requirements,
        status=PlanStatus.DRAFT,
    )


def _retrieve(**overrides: Any) -> RetrieveOperation:
    fields: dict[str, Any] = {
        "id": "debt",
        "requirement_id": "annual_debt_service",
        "spec_id": SPEC_KENYA,
    }
    return RetrieveOperation(**(fields | overrides))


def _aggregate(**overrides: Any) -> AggregateOperation:
    fields: dict[str, Any] = {
        "id": "totals",
        "requirement_id": "annual_totals",
        "input": "debt",
        "reduction": Reduction.SUM,
        "group_by": (GroupBy.PERIOD,),
    }
    return AggregateOperation(**(fields | overrides))


def _translation(*operations: Operation, terminal: Sequence[str] = ("totals",)) -> Translation:
    return Translation(operations=operations, terminal=terminal)


def _codes(result: CompileResult) -> list[Code]:
    return [issue.code for issue in result.issues]


def _plan_refusal(
    decisions: Sequence[Decision], requirements: Sequence[EvidenceRequirement]
) -> str:
    """Why the plan refuses `requirements`, which `propose_plan` reports before any compile."""
    with pytest.raises(ValidationError) as refused:
        _draft(decisions, requirements)
    return str(refused.value)


def _pools(result: CompileResult, *, one_figure: bool) -> bool:
    """Whether every `entity_required` issue in `result` ends on the pooling its reader takes."""
    remedy = pooling_remedy(one_figure=one_figure, each_side=True)
    pooling = [issue for issue in result.issues if issue.code is Code.ENTITY_REQUIRED]
    return bool(pooling) and all(issue.reason.endswith(remedy) for issue in pooling)


@pytest.fixture
def build(
    schemas: dict[str, ToolSchema],
    capabilities: tuple[ToolCapability, ...],
    source: SourceFactory,
) -> Callable[..., CompileResult]:
    """Compile against the fixture surface, with Kenya's source discovered for `annual_debt_service`."""

    def _compile(
        plan: Plan,
        translation: Translation,
        *,
        discovered: dict[str, SelectedSource] | None = None,
        surface: dict[str, ToolSchema] | None = None,
        table: Sequence[ToolCapability] | None = None,
    ) -> CompileResult:
        return compile_plan(
            plan=plan,
            translation=translation,
            schemas=schemas if surface is None else surface,
            discovered={"annual_debt_service": source()} if discovered is None else discovered,
            capabilities=capabilities if table is None else table,
        )

    return _compile


# --- The happy paths ---------------------------------------------------------------


def test_a_retrieval_then_an_aggregate_compiles_to_two_bound_steps(
    build: Callable[..., CompileResult],
    decision: DecisionFactory,
    requirements: tuple[EvidenceRequirement, ...],
) -> None:
    plan = _draft([decision()], list(requirements))

    result = build(plan, _translation(_retrieve(), _aggregate()))

    assert result.ok, result.issues
    workflow = result.workflow
    assert workflow is not None
    retrieve, totals = workflow.steps
    assert (retrieve.tool.server, retrieve.tool.name) == ("data-mcp", "get_data")
    # A flow source's frame holds its open sides' parties without their group totals.
    assert retrieve.args == {"spec_id": SPEC_KENYA, "without_party_totals": True}
    assert retrieve.bindings == {}
    assert (totals.tool.server, totals.tool.name) == ("analysis-mcp", "aggregate")
    # The compiler wrote the frame grammar's period column; the proposal said "period".
    assert totals.args == {"aggregation": "sum", "group_by": ["time_period"]}
    assert totals.bindings == {
        "frame_ref": Binding(from_step="debt", output=ArtifactKind.FRAME_REF)
    }
    assert [(b.requirement_id, b.step_id, b.spec_id) for b in workflow.evidence_bindings] == [
        ("annual_debt_service", "debt", SPEC_KENYA)
    ]
    assert [s.spec_id for s in workflow.sources] == [SPEC_KENYA]
    assert workflow.terminal_steps == ("totals",)
    # The source's informational findings come back as warnings -- which ones a
    # clean source raises is `test_evidence.py`'s to pin -- and they are on the
    # workflow too, so approving the plan approves them.
    assert "unit_matches" in [w.code.value for w in result.warnings]
    assert workflow.warnings == result.warnings


def test_an_unread_dataset_requirement_beside_a_derived_one_compiles_to_two_terminals(
    build: Callable[..., CompileResult],
    decision: DecisionFactory,
    requirement: RequirementFactory,
    derived_requirement: DerivedRequirementFactory,
    source: SourceFactory,
) -> None:
    """Under `answer_mode="series"`, a dataset requirement no calculation
    reads is a terminal answer in its own right. A plan can pair one
    calculation over one retrieval with a second, separate retrieval nothing
    reads, such as asking for Kenya's total debt service and its population
    series in one turn. Each retrieval compiles to its own terminal step,
    with no join or derived requirement tying them together."""
    population = requirement(id="population", measure="population")
    plan = _draft(
        [decision()],
        [requirement(), derived_requirement(), population],
        answer_mode=AnswerMode.SERIES,
    )
    discovered = {"annual_debt_service": source(), "population": source(spec_id=SPEC_OTHER)}

    result = build(
        plan,
        Translation(
            operations=(
                _retrieve(),
                _aggregate(),
                _retrieve(id="pop", requirement_id="population", spec_id=SPEC_OTHER),
            ),
            terminal=("totals", "pop"),
        ),
        discovered=discovered,
    )

    assert result.ok, result.issues
    workflow = result.workflow
    assert workflow is not None
    assert workflow.terminal_steps == ("totals", "pop")
    assert {s.id for s in workflow.steps} == {"debt", "pop", "totals"}


def test_a_compiled_workflow_completes_the_draft_plan(
    build: Callable[..., CompileResult],
    decision: DecisionFactory,
    requirements: tuple[EvidenceRequirement, ...],
) -> None:
    plan = _draft([decision()], list(requirements))
    translation = _translation(_retrieve(), _aggregate())

    result = build(plan, translation)
    assert result.workflow is not None
    translated = plan.with_translation(translation=translation, workflow=result.workflow)

    assert translated.status is PlanStatus.PENDING_APPROVAL
    assert translated.workflow == result.workflow


def _two_chains(
    decision: DecisionFactory,
    requirement: RequirementFactory,
    derived_requirement: DerivedRequirementFactory,
) -> tuple[Plan, tuple[Operation, ...]]:
    """A plan with two independent retrieve-then-sum chains, and the operations carrying it out."""
    gdp = requirement(id="annual_gdp", measure="gross domestic product")
    gdp_totals = derived_requirement(id="gdp_totals", input_id="annual_gdp")
    plan = _draft([decision()], [requirement(), derived_requirement(), gdp, gdp_totals])
    operations = (
        _retrieve(),
        _aggregate(),
        _retrieve(id="gdp", requirement_id="annual_gdp", spec_id=SPEC_OTHER),
        _aggregate(id="gdp_sum", requirement_id="gdp_totals", input="gdp"),
    )
    return plan, operations


def test_a_reversed_listing_compiles_to_byte_identical_workflow_json(
    build: Callable[..., CompileResult],
    decision: DecisionFactory,
    requirement: RequirementFactory,
    derived_requirement: DerivedRequirementFactory,
    source: SourceFactory,
) -> None:
    # Two independent chains: listed one way, the translator's order and a
    # dependency order agree; reversed, they do not. The workflow is hashed
    # into the approved revision, so the same operations have to compile to
    # the same bytes whichever way the translator happened to list them.
    plan, operations = _two_chains(decision, requirement, derived_requirement)
    discovered = {"annual_debt_service": source(), "annual_gdp": source(spec_id=SPEC_OTHER)}

    listed = build(
        plan,
        Translation(operations=operations, terminal=("totals", "gdp_sum")),
        discovered=discovered,
    )
    reversed_ = build(
        plan,
        Translation(operations=operations[::-1], terminal=("gdp_sum", "totals")),
        discovered=discovered,
    )

    assert listed.ok and reversed_.ok, (listed.issues, reversed_.issues)
    assert listed.workflow is not None and reversed_.workflow is not None
    assert listed.workflow.model_dump_json() == reversed_.workflow.model_dump_json()
    # By requirement id, then operation id; the terminals follow the steps.
    assert [s.id for s in listed.workflow.steps] == ["debt", "gdp", "totals", "gdp_sum"]
    assert listed.workflow.terminal_steps == ("totals", "gdp_sum")
    warned_for = [w.requirement_id for w in listed.warnings]
    assert warned_for == sorted(warned_for)
    assert set(warned_for) == {"annual_debt_service", "annual_gdp"}


def test_every_step_precedes_the_steps_reading_it_under_any_listing(
    build: Callable[..., CompileResult],
    decision: DecisionFactory,
    requirement: RequirementFactory,
    derived_requirement: DerivedRequirementFactory,
    source: SourceFactory,
) -> None:
    plan, operations = _two_chains(decision, requirement, derived_requirement)
    discovered = {"annual_debt_service": source(), "annual_gdp": source(spec_id=SPEC_OTHER)}
    workflows = []
    for listing in permutations(operations):
        result = build(
            plan,
            Translation(operations=listing, terminal=("totals", "gdp_sum")),
            discovered=discovered,
        )
        assert result.workflow is not None, result.issues
        workflows.append(result.workflow)

    for workflow in workflows:
        position = {step.id: index for index, step in enumerate(workflow.steps)}
        for step in workflow.steps:
            for binding in step.bindings.values():
                assert position[binding.from_step] < position[step.id], workflow.steps
    assert len({workflow.model_dump_json() for workflow in workflows}) == 1


def test_grouping_by_entity_writes_the_sources_role_columns(
    build: Callable[..., CompileResult],
    decision: DecisionFactory,
    requirement: RequirementFactory,
    derived_requirement: DerivedRequirementFactory,
) -> None:
    plan = _draft(
        [decision()],
        [requirement(), derived_requirement(group_by=(GroupBy.ENTITY, GroupBy.PERIOD))],
    )

    result = build(
        plan, _translation(_retrieve(), _aggregate(group_by=(GroupBy.ENTITY, GroupBy.PERIOD)))
    )

    assert result.ok, result.issues
    assert result.workflow is not None
    assert result.workflow.steps[1].args["group_by"] == ["destination", "time_period"]


def _two_role_source(source: SourceFactory) -> SelectedSource:
    """A source resolving Kenya as the recipient and China as the creditor."""
    return source(
        entities=(
            ResolvedEntity(role="recipient", ref="country/KEN", label="Kenya"),
            ResolvedEntity(role="creditor", ref="country/CHN", label="China"),
        )
    )


def test_grouping_by_entity_writes_only_the_role_columns_the_requirement_resolves_under(
    build: Callable[..., CompileResult],
    decision: DecisionFactory,
    requirement: RequirementFactory,
    derived_requirement: DerivedRequirementFactory,
    source: SourceFactory,
) -> None:
    # The plan names Kenya alone; the source also resolves a creditor. Grouping
    # by the creditor column would split the answer along a line the plan
    # never stated, so only the column Kenya resolved under is a key.
    plan = _draft(
        [decision()],
        [requirement(), derived_requirement(group_by=(GroupBy.ENTITY, GroupBy.PERIOD))],
    )

    result = build(
        plan,
        _translation(_retrieve(), _aggregate(group_by=(GroupBy.ENTITY, GroupBy.PERIOD))),
        discovered={"annual_debt_service": _two_role_source(source)},
    )

    assert result.ok, result.issues
    assert result.workflow is not None
    assert result.workflow.steps[1].args["group_by"] == ["recipient", "time_period"]


def test_a_growth_step_writes_the_window_as_years_and_the_entity_columns(
    build: Callable[..., CompileResult],
    decision: DecisionFactory,
    requirement: RequirementFactory,
) -> None:
    growth_req = GrowthRequirement(
        id="growth_of_debt",
        decision_ids=["measure"],
        input_id="annual_debt_service",
        window_start=2015,
        window_end=2023,
    )
    plan = _draft([decision()], [requirement(), growth_req])
    change = GrowthOperation(
        id="change",
        requirement_id="growth_of_debt",
        input="debt",
        window_start=2015,
        window_end=2023,
    )

    result = build(plan, _translation(_retrieve(), change, terminal=["change"]))

    assert result.ok, result.issues
    assert result.workflow is not None
    step = result.workflow.steps[1]
    assert step.tool.name == "growth"
    assert step.args == {"window_start": "2015", "window_end": "2023", "entity_by": ["destination"]}
    assert step.bindings == {"frame_ref": Binding(from_step="debt", output=ArtifactKind.FRAME_REF)}


def _two_country_plan(
    decision: DecisionFactory,
    requirement: RequirementFactory,
    comparison: EvidenceRequirement,
    *derived: EvidenceRequirement,
) -> Plan:
    """A plan whose dataset requirement names Kenya and Ghana, read by `derived` and `comparison`.

    Compile it against `wide_source()`: the requirement names two entities,
    so a source resolving Kenya alone cannot serve it.
    """
    comparison_decision = decision(
        slot="comparison",
        options=["Kenya against Ghana", "Kenya on its own"],
        chosen="Kenya against Ghana",
    )
    two = requirement(
        decision_ids=["measure", "comparison"],
        entities=[Entity(name="Kenya", role="debtor"), Entity(name="Ghana", role="debtor")],
    )
    return _draft([decision(), comparison_decision], [two, *derived, comparison])


def test_comparing_two_entities_of_one_retrieval_binds_each_side_by_role_and_ref(
    build: Callable[..., CompileResult],
    decision: DecisionFactory,
    requirement: RequirementFactory,
    wide_source: SourceFactory,
) -> None:
    # The proposal names the entities as the requirement does; the compiler
    # writes the source's own column and value, so nothing is guessed.
    gap = CompareOperation(
        id="gap",
        requirement_id="gap",
        left=Side(input="debt", entity="Kenya"),
        right=Side(input="debt", entity="Ghana"),
    )
    comparison = ComparisonRequirement(
        id="gap",
        decision_ids=["measure", "comparison"],
        minuend=RequirementSide(input_id="annual_debt_service", entity="Ghana"),
        subtrahend=RequirementSide(input_id="annual_debt_service", entity="Kenya"),
    )

    result = build(
        _two_country_plan(decision, requirement, comparison),
        _translation(_retrieve(), gap, terminal=["gap"]),
        discovered={"annual_debt_service": wide_source()},
    )

    assert result.ok, result.issues
    assert result.workflow is not None
    step = result.workflow.steps[1]
    assert step.tool.name == "compare"
    assert step.args == {
        "left_entity": {"destination": "country/KEN"},
        "right_entity": {"destination": "country/GHA"},
    }
    assert step.bindings == {"frame_ref": Binding(from_step="debt", output=ArtifactKind.FRAME_REF)}


def test_comparing_two_retrievals_binds_both_frames(
    build: Callable[..., CompileResult],
    decision: DecisionFactory,
    requirement: RequirementFactory,
    source: SourceFactory,
) -> None:
    gdp = requirement(id="annual_gdp", measure="gross domestic product")
    comparison = ComparisonRequirement(
        id="gap",
        decision_ids=["measure"],
        minuend=RequirementSide(input_id="annual_gdp"),
        subtrahend=RequirementSide(input_id="annual_debt_service"),
    )
    plan = _draft([decision()], [requirement(), gdp, comparison])
    gap = CompareOperation(
        id="gap", requirement_id="gap", left=Side(input="debt"), right=Side(input="gdp")
    )
    retrieve_gdp = _retrieve(id="gdp", requirement_id="annual_gdp", spec_id=SPEC_OTHER)

    result = build(
        plan,
        _translation(_retrieve(), retrieve_gdp, gap, terminal=["gap"]),
        discovered={"annual_debt_service": source(), "annual_gdp": source(spec_id=SPEC_OTHER)},
    )

    assert result.ok, result.issues
    assert result.workflow is not None
    step = result.workflow.steps[-1]
    assert step.args == {}
    assert step.bindings == {
        "frame_ref": Binding(from_step="debt", output=ArtifactKind.FRAME_REF),
        "right_frame_ref": Binding(from_step="gdp", output=ArtifactKind.FRAME_REF),
    }
    assert [s.spec_id for s in result.workflow.sources] == [SPEC_KENYA, SPEC_OTHER]


def test_a_difference_subtracts_its_subtrahend_from_its_minuend(
    build: Callable[..., CompileResult],
    decision: DecisionFactory,
    requirement: RequirementFactory,
    source: SourceFactory,
) -> None:
    # "Disbursements minus debt service". analysis-mcp's `compare` takes its
    # right frame minus its left, so the minuend's retrieval binds
    # `right_frame_ref` and the subtrahend's binds `frame_ref`.
    disbursements = requirement(id="disbursements", measure="disbursements")
    debt_service = requirement(id="debt_service", measure="external debt service paid")
    net = ComparisonRequirement(
        id="net",
        decision_ids=["measure"],
        minuend=RequirementSide(input_id="disbursements"),
        subtrahend=RequirementSide(input_id="debt_service"),
    )
    plan = _draft([decision()], [disbursements, debt_service, net])
    sources = {"disbursements": source(spec_id=SPEC_OTHER), "debt_service": source()}
    translation = build_translation(
        plan, {requirement_id: s.spec_id for requirement_id, s in sources.items()}
    )

    result = build(plan, translation, discovered=sources)

    assert result.ok, result.issues
    assert result.workflow is not None
    (step,) = [step for step in result.workflow.steps if step.id == "net"]
    assert step.tool.name == "compare"
    assert step.bindings == {
        "frame_ref": Binding(from_step="debt_service", output=ArtifactKind.FRAME_REF),
        "right_frame_ref": Binding(from_step="disbursements", output=ArtifactKind.FRAME_REF),
    }


def test_a_rank_step_writes_the_sources_role_columns_as_rank_by(
    build: Callable[..., CompileResult],
    decision: DecisionFactory,
    requirement: RequirementFactory,
) -> None:
    rank_req = RankRequirement(
        id="ranked",
        decision_ids=["measure"],
        input_id="annual_debt_service",
        direction=RankDirection.HIGHEST_FIRST,
        among=RankAmong.POPULATION,
        reduction=Reduction.SUM,
    )
    plan = _draft(
        [decision(fixes=DecisionFixes(among=RankAmong.POPULATION))], [requirement(), rank_req]
    )
    ranked = RankOperation(
        id="ranked",
        requirement_id="ranked",
        input="debt",
        direction=RankDirection.HIGHEST_FIRST,
        among=RankAmong.POPULATION,
        reduction=Reduction.SUM,
    )

    result = build(plan, _translation(_retrieve(), ranked, terminal=["ranked"]))

    assert result.ok, result.issues
    assert result.workflow is not None
    step = result.workflow.steps[1]
    assert step.tool.name == "rank"
    assert step.args == {"rank_by": ["destination"], "ascending": False, "aggregation": "sum"}
    assert step.bindings == {"frame_ref": Binding(from_step="debt", output=ArtifactKind.FRAME_REF)}


def test_a_rank_step_with_a_limit_writes_it_as_an_argument(
    build: Callable[..., CompileResult],
    decision: DecisionFactory,
    requirement: RequirementFactory,
) -> None:
    # Without a limit the step carries no `limit` key at all (the test above
    # pins the whole argument dict), so the tool keeps every ranked entity.
    rank_req = RankRequirement(
        id="ranked",
        decision_ids=["measure"],
        input_id="annual_debt_service",
        direction=RankDirection.HIGHEST_FIRST,
        among=RankAmong.POPULATION,
        reduction=Reduction.SUM,
        limit=10,
    )
    plan = _draft(
        [decision(fixes=DecisionFixes(among=RankAmong.POPULATION))], [requirement(), rank_req]
    )
    ranked = RankOperation(
        id="ranked",
        requirement_id="ranked",
        input="debt",
        direction=RankDirection.HIGHEST_FIRST,
        among=RankAmong.POPULATION,
        reduction=Reduction.SUM,
        limit=10,
    )

    result = build(plan, _translation(_retrieve(), ranked, terminal=["ranked"]))

    assert result.ok, result.issues
    assert result.workflow is not None
    assert result.workflow.steps[1].args == {
        "rank_by": ["destination"],
        "ascending": False,
        "aggregation": "sum",
        "limit": 10,
    }


def test_a_rank_step_ranks_by_the_role_column_the_requirement_resolves_under(
    build: Callable[..., CompileResult],
    decision: DecisionFactory,
    requirement: RequirementFactory,
    source: SourceFactory,
) -> None:
    # A rank is per entity by construction, so the creditor the plan never
    # named is not pooled -- but it is not a ranking key either.
    rank_req = RankRequirement(
        id="ranked",
        decision_ids=["measure"],
        input_id="annual_debt_service",
        direction=RankDirection.HIGHEST_FIRST,
        among=RankAmong.POPULATION,
        reduction=Reduction.SUM,
    )
    plan = _draft(
        [decision(fixes=DecisionFixes(among=RankAmong.POPULATION))], [requirement(), rank_req]
    )
    ranked = RankOperation(
        id="ranked",
        requirement_id="ranked",
        input="debt",
        direction=RankDirection.HIGHEST_FIRST,
        among=RankAmong.POPULATION,
        reduction=Reduction.SUM,
    )

    result = build(
        plan,
        _translation(_retrieve(), ranked, terminal=["ranked"]),
        discovered={"annual_debt_service": _two_role_source(source)},
    )

    assert result.ok, result.issues
    assert result.workflow is not None
    assert result.workflow.steps[1].args["rank_by"] == ["recipient"]


def _two_role_requirement(**overrides: Any) -> DatasetRequirement:
    """A requirement naming Kenya as debtor and France and Germany as creditor.

    Two roles, both named, so a rank over it needs an axis to say which one
    it ranks by -- unlike `_two_role_source`, whose surplus role the
    requirement never names at all and which the retrieval shape drops on
    its own.
    """
    fields: dict[str, Any] = {
        "id": "annual_debt_service",
        "decision_ids": ["measure"],
        "measure": "external debt service paid",
        "entities": [
            Entity(name="Kenya", role="debtor"),
            Entity(name="France", role="creditor"),
            Entity(name="Germany", role="creditor"),
        ],
        "period": Period(start=2015, end=2024),
        "unit": Unit(currency="USD", prices="current"),
    }
    return DatasetRequirement(**(fields | overrides))


def _two_role_creditor_source(source: SourceFactory, **overrides: Any) -> SelectedSource:
    """A source resolving Kenya as debtor and France and Germany as creditor -- as named, no surplus."""
    fields: dict[str, Any] = {
        "entities": (
            ResolvedEntity(role="debtor", ref="country/KEN", label="Kenya"),
            ResolvedEntity(role="creditor", ref="country/FRA", label="France"),
            ResolvedEntity(role="creditor", ref="country/DEU", label="Germany"),
        )
    }
    return source(**(fields | overrides))


def test_a_two_role_rank_without_an_axis_is_refused(
    build: Callable[..., CompileResult],
    decision: DecisionFactory,
    requirement: RequirementFactory,
    source: SourceFactory,
) -> None:
    # Without this check, `_role_columns` writes every role column into
    # `rank_by` once none of the requirement's own entities resolve against
    # the discovered source (`_role_columns`' own fallback to "every column
    # the source has"), and the engine would rank debtor-creditor pairs
    # rather than refuse for want of an axis. The plan itself only ever sees
    # the one role its own entities name -- "Kenya" never resolving is a
    # fact only source discovery reveals -- so this is a case the compiler
    # alone can catch.
    rank_req = RankRequirement(
        id="ranked",
        decision_ids=["measure", "population"],
        input_id="annual_debt_service",
        direction=RankDirection.HIGHEST_FIRST,
        among=RankAmong.POPULATION,
        reduction=Reduction.SUM,
    )
    population = decision(slot="population", fixes=DecisionFixes(among=RankAmong.POPULATION))
    plan = _draft([decision(), population], [requirement(), rank_req])
    ranked = RankOperation(
        id="ranked",
        requirement_id="ranked",
        input="debt",
        direction=RankDirection.HIGHEST_FIRST,
        among=RankAmong.POPULATION,
        reduction=Reduction.SUM,
    )
    unmatched_source = source(
        entities=(
            ResolvedEntity(role="debtor", ref="country/UGA", label="Uganda"),
            ResolvedEntity(role="creditor", ref="country/FRA", label="France"),
        )
    )

    result = build(
        plan,
        _translation(_retrieve(), ranked, terminal=["ranked"]),
        discovered={"annual_debt_service": unmatched_source},
    )

    assert Code.RANK_AXIS_REQUIRED in _codes(result)
    assert result.workflow is None


def test_a_rank_writes_only_the_axis_column(
    build: Callable[..., CompileResult],
    decision: DecisionFactory,
    source: SourceFactory,
) -> None:
    # The fix: naming an axis over a two-role input writes that column
    # alone, not both -- `rank_by` is never a debtor-creditor pair.
    rank_req = RankRequirement(
        id="ranked",
        decision_ids=["measure", "population"],
        input_id="annual_debt_service",
        direction=RankDirection.HIGHEST_FIRST,
        among=RankAmong.POPULATION,
        axis="creditor",
        reduction=Reduction.SUM,
    )
    population = decision(slot="population", fixes=DecisionFixes(among=RankAmong.POPULATION))
    plan = _draft([decision(), population], [_two_role_requirement(), rank_req])
    ranked = RankOperation(
        id="ranked",
        requirement_id="ranked",
        input="debt",
        direction=RankDirection.HIGHEST_FIRST,
        among=RankAmong.POPULATION,
        axis="creditor",
        reduction=Reduction.SUM,
    )

    result = build(
        plan,
        _translation(_retrieve(), ranked, terminal=["ranked"]),
        discovered={"annual_debt_service": _two_role_creditor_source(source)},
    )

    assert result.ok, result.issues
    assert result.workflow is not None
    step = result.workflow.steps[-1]
    assert step.args == {"rank_by": ["creditor"], "ascending": False, "aggregation": "sum"}


def test_a_named_rank_over_a_wider_source_is_refused(
    build: Callable[..., CompileResult],
    decision: DecisionFactory,
    source: SourceFactory,
) -> None:
    # The width rule's own new distinction: a `named` rank claims to rank
    # exactly the entities a filter narrowed its axis to (here, France and
    # Germany as creditor); the debtor role is never filtered, so a source
    # resolving a surplus debtor (Uganda, which the requirement never names)
    # leaks into what the rank reduces over, unseen. A `population` rank
    # over the same shape would not be reported -- the wider field is the
    # point there.
    filter_req = FilterRequirement(
        id="named_creditors",
        decision_ids=["measure"],
        input_id="annual_debt_service",
        predicates=(EntityIn(names=("France", "Germany"), role="creditor"),),
    )
    rank_req = RankRequirement(
        id="ranked",
        decision_ids=["measure"],
        input_id="named_creditors",
        direction=RankDirection.HIGHEST_FIRST,
        among=RankAmong.NAMED,
        axis="creditor",
        reduction=Reduction.SUM,
    )
    plan = _draft([decision()], [_two_role_requirement(), filter_req, rank_req])
    filter_op = FilterOperation(
        id="named_creditors",
        requirement_id="named_creditors",
        input="debt",
        predicates=(EntityIn(names=("France", "Germany"), role="creditor"),),
    )
    ranked_op = RankOperation(
        id="ranked",
        requirement_id="ranked",
        input="named_creditors",
        direction=RankDirection.HIGHEST_FIRST,
        among=RankAmong.NAMED,
        axis="creditor",
        reduction=Reduction.SUM,
    )
    wide_creditor_source = _two_role_creditor_source(
        source,
        entities=(
            ResolvedEntity(role="debtor", ref="country/KEN", label="Kenya"),
            ResolvedEntity(role="debtor", ref="country/UGA", label="Uganda"),
            ResolvedEntity(role="creditor", ref="country/FRA", label="France"),
            ResolvedEntity(role="creditor", ref="country/DEU", label="Germany"),
        ),
    )

    result = build(
        plan,
        _translation(_retrieve(), filter_op, ranked_op, terminal=["ranked"]),
        discovered={"annual_debt_service": wide_creditor_source},
    )

    width_issues = [i for i in result.issues if i.code is Code.SOURCE_WIDER_THAN_REQUIREMENT]
    assert len(width_issues) == 1, result.issues
    assert width_issues[0].operation_id == "ranked"


def test_a_population_rank_over_the_same_wide_source_is_not_reported(
    build: Callable[..., CompileResult],
    decision: DecisionFactory,
    source: SourceFactory,
) -> None:
    rank_req = RankRequirement(
        id="ranked",
        decision_ids=["measure", "population"],
        input_id="annual_debt_service",
        direction=RankDirection.HIGHEST_FIRST,
        among=RankAmong.POPULATION,
        axis="creditor",
        reduction=Reduction.SUM,
    )
    population = decision(slot="population", fixes=DecisionFixes(among=RankAmong.POPULATION))
    plan = _draft([decision(), population], [_two_role_requirement(), rank_req])
    ranked_op = RankOperation(
        id="ranked",
        requirement_id="ranked",
        input="debt",
        direction=RankDirection.HIGHEST_FIRST,
        among=RankAmong.POPULATION,
        axis="creditor",
        reduction=Reduction.SUM,
    )
    wide_creditor_source = _two_role_creditor_source(
        source,
        entities=(
            ResolvedEntity(role="debtor", ref="country/KEN", label="Kenya"),
            ResolvedEntity(role="debtor", ref="country/UGA", label="Uganda"),
            ResolvedEntity(role="creditor", ref="country/FRA", label="France"),
            ResolvedEntity(role="creditor", ref="country/DEU", label="Germany"),
        ),
    )

    result = build(
        plan,
        _translation(_retrieve(), ranked_op, terminal=["ranked"]),
        discovered={"annual_debt_service": wide_creditor_source},
    )

    assert Code.SOURCE_WIDER_THAN_REQUIREMENT not in _codes(result)
    assert result.ok, result.issues


_G7_DONORS = (
    ResolvedEntity(role="origin", ref="country/USA", label="United States of America"),
    ResolvedEntity(role="origin", ref="country/FRA", label="France"),
)
_AFRICAN_RECIPIENTS = (
    ResolvedEntity(role="destination", ref="country/KEN", label="Kenya"),
    ResolvedEntity(role="destination", ref="country/GHA", label="Ghana"),
)


def _g7_to_africa(source: SourceFactory) -> SelectedSource:
    """A source binding the G7's donors and Africa's recipients, each set named beside them."""
    return source(
        entities=(*_G7_DONORS, *_AFRICAN_RECIPIENTS),
        groups=(
            ResolvedGroup(role="origin", label="G7"),
            ResolvedGroup(role="destination", label="Africa", ref="africa"),
        ),
    )


def _g7_to_africa_requirement(requirement: RequirementFactory) -> DatasetRequirement:
    return requirement(
        entities=[Entity(name="G7", role="donor"), Entity(name="Africa", role="recipient")]
    )


def test_a_period_sum_over_groups_named_on_both_roles_is_not_wider_than_the_requirement(
    build: Callable[..., CompileResult],
    decision: DecisionFactory,
    requirement: RequirementFactory,
    derived_requirement: DerivedRequirementFactory,
    source: SourceFactory,
) -> None:
    # "G7" and "Africa" each name a set whose members the source binds, so
    # every member is named and none is surplus to the period sum.
    plan = _draft([decision()], [_g7_to_africa_requirement(requirement), derived_requirement()])

    result = build(
        plan,
        _translation(_retrieve(), _aggregate()),
        discovered={"annual_debt_service": _g7_to_africa(source)},
    )

    assert result.ok, result.issues


def test_a_filter_naming_a_group_keeps_its_members(
    build: Callable[..., CompileResult],
    decision: DecisionFactory,
    requirement: RequirementFactory,
    source: SourceFactory,
) -> None:
    kept = FilterRequirement(
        id="kept",
        decision_ids=["measure"],
        input_id="annual_debt_service",
        predicates=(EntityIn(names=("Africa",)),),
    )
    plan = _draft([decision()], [_g7_to_africa_requirement(requirement), kept])
    filter_op = FilterOperation(
        id="kept",
        requirement_id="kept",
        input="debt",
        predicates=(EntityIn(names=("Africa",)),),
    )

    result = build(
        plan,
        _translation(_retrieve(), filter_op, terminal=["kept"]),
        discovered={"annual_debt_service": _g7_to_africa(source)},
    )

    assert result.ok, result.issues
    assert result.workflow is not None
    assert result.workflow.steps[1].args["predicates"] == [
        {
            "kind": "category_match",
            "column": "destination",
            "values": ["country/KEN", "country/GHA"],
        }
    ]


def test_a_side_naming_a_group_of_several_members_is_refused(
    build: Callable[..., CompileResult],
    decision: DecisionFactory,
    requirement: RequirementFactory,
    source: SourceFactory,
) -> None:
    # A side's entity argument picks one series; Africa's members are two.
    # The requirement names one donor and one recipient, so G7's side and
    # Africa's both read the one G7-to-Africa flow, which is degenerate too.
    comparison = ComparisonRequirement(
        id="gap",
        decision_ids=["measure"],
        minuend=RequirementSide(input_id="annual_debt_service", entity="G7"),
        subtrahend=RequirementSide(input_id="annual_debt_service", entity="Africa"),
    )
    plan = _draft([decision()], [_g7_to_africa_requirement(requirement), comparison])
    gap = CompareOperation(
        id="gap",
        requirement_id="gap",
        left=Side(input="debt", entity="Africa"),
        right=Side(input="debt", entity="G7"),
    )

    result = build(
        plan,
        _translation(_retrieve(), gap, terminal=["gap"]),
        discovered={"annual_debt_service": _g7_to_africa(source)},
    )

    assert _codes(result) == [Code.DEGENERATE_SIDES, Code.ENTITY_AMBIGUOUS], result.issues
    assert result.workflow is None


def test_a_share_of_step_binds_both_frames_when_the_sides_differ(
    build: Callable[..., CompileResult],
    decision: DecisionFactory,
    requirement: RequirementFactory,
    source: SourceFactory,
) -> None:
    gdp = requirement(id="annual_gdp", measure="gross domestic product")
    share = ShareOfRequirement(
        id="share",
        decision_ids=["measure"],
        numerator=RequirementSide(input_id="annual_debt_service"),
        denominator=RequirementSide(input_id="annual_gdp"),
    )
    plan = _draft([decision()], [requirement(), gdp, share])
    share_op = ShareOfOperation(
        id="share",
        requirement_id="share",
        numerator=Side(input="debt"),
        denominator=Side(input="gdp"),
    )
    retrieve_gdp = _retrieve(id="gdp", requirement_id="annual_gdp", spec_id=SPEC_OTHER)

    result = build(
        plan,
        _translation(_retrieve(), retrieve_gdp, share_op, terminal=["share"]),
        discovered={"annual_debt_service": source(), "annual_gdp": source(spec_id=SPEC_OTHER)},
    )

    assert result.ok, result.issues
    assert result.workflow is not None
    step = result.workflow.steps[-1]
    assert step.tool.name == "share_of"
    assert step.args == {}
    assert step.bindings == {
        "frame_ref": Binding(from_step="debt", output=ArtifactKind.FRAME_REF),
        "denominator_frame_ref": Binding(from_step="gdp", output=ArtifactKind.FRAME_REF),
    }


def test_a_keyed_share_of_axis_naming_an_unknown_role_is_share_key_unknown(
    build: Callable[..., CompileResult],
    decision: DecisionFactory,
    source: SourceFactory,
) -> None:
    # Both requirements name Kenya as "donor", so the plan itself -- which
    # never sees a discovered source -- validates axis="donor" against both
    # sides happily. Only the denominator's own discovered source reveals
    # the problem: it never resolves Kenya at all, so `_role_columns`' own
    # fallback (every column the source has) hands the denominator's shape
    # a role named for its raw column ("recipient"), not the phrase "donor"
    # the plan stated -- a mismatch source discovery alone can surface.
    health = DatasetRequirement(
        id="req_health",
        decision_ids=["measure"],
        measure="ODA to health",
        entities=[Entity(name="Kenya", role="donor")],
        period=Period(start=2015, end=2024),
    )
    total = DatasetRequirement(
        id="req_total",
        decision_ids=["measure"],
        measure="total ODA",
        entities=[Entity(name="Kenya", role="donor")],
        period=Period(start=2015, end=2024),
    )
    share_req = ShareOfRequirement(
        id="share",
        decision_ids=["measure"],
        numerator=RequirementSide(input_id="req_health"),
        denominator=RequirementSide(input_id="req_total"),
        axis="donor",
        denominator_scope=DenominatorScope.PER_AXIS,
    )
    plan = _draft([decision()], [health, total, share_req])
    health_op = RetrieveOperation(id="health", requirement_id="req_health", spec_id=SPEC_KENYA)
    total_op = RetrieveOperation(id="total", requirement_id="req_total", spec_id=SPEC_OTHER)
    share_op = ShareOfOperation(
        id="share",
        requirement_id="share",
        numerator=Side(input="health"),
        denominator=Side(input="total"),
        axis="donor",
        denominator_scope=DenominatorScope.PER_AXIS,
    )
    donor_source = source(
        entities=(ResolvedEntity(role="donor", ref="country/KEN", label="Kenya"),)
    )
    unmatched_total_source = source(
        spec_id=SPEC_OTHER,
        entities=(ResolvedEntity(role="recipient", ref="country/UGA", label="Uganda"),),
    )

    result = build(
        plan,
        _translation(health_op, total_op, share_op, terminal=["share"]),
        discovered={"req_health": donor_source, "req_total": unmatched_total_source},
    )

    assert Code.SHARE_KEY_UNKNOWN in _codes(result)
    assert result.workflow is None


_THREE_COUNTRIES = (
    ResolvedEntity(role="country", ref="country/KEN", label="Kenya"),
    ResolvedEntity(role="country", ref="country/UGA", label="Uganda"),
    ResolvedEntity(role="country", ref="country/TZA", label="Tanzania"),
)


def test_an_axis_less_share_of_a_pooled_multi_member_numerator_is_share_axis_required(
    build: Callable[..., CompileResult], decision: DecisionFactory, source: SourceFactory
) -> None:
    # Kenya's, Uganda's and Tanzania's combined population divided by
    # itself: with no axis, the numerator pools all three countries just
    # as the denominator does, so the quotient is 1 at every period --
    # the live regression this guards against, reproduced. The two sides
    # read separate retrievals (`ShareOfOperation` itself refuses a
    # numerator and denominator that are otherwise identical), so
    # `DEGENERATE_SIDES`'s own identity check, keyed by retrieve op id,
    # never sees them as the same thing either.
    pop_num = DatasetRequirement(
        id="pop_num",
        decision_ids=["measure"],
        measure="population",
        entities=[Entity(name="Kenya", role="country")],
        period=Period(start=2015, end=2024),
    )
    pop_denom = DatasetRequirement(
        id="pop_denom",
        decision_ids=["measure"],
        measure="population",
        entities=[Entity(name="Kenya", role="country")],
        period=Period(start=2015, end=2024),
    )
    share = ShareOfRequirement(
        id="share",
        decision_ids=["measure"],
        numerator=RequirementSide(input_id="pop_num"),
        denominator=RequirementSide(input_id="pop_denom"),
    )
    plan = _draft([decision()], [pop_num, pop_denom, share])
    fetch_num = RetrieveOperation(id="fetch_num", requirement_id="pop_num", spec_id=SPEC_KENYA)
    fetch_denom = RetrieveOperation(
        id="fetch_denom", requirement_id="pop_denom", spec_id=SPEC_OTHER
    )
    share_op = ShareOfOperation(
        id="share",
        requirement_id="share",
        numerator=Side(input="fetch_num"),
        denominator=Side(input="fetch_denom"),
    )

    result = build(
        plan,
        _translation(fetch_num, fetch_denom, share_op, terminal=["share"]),
        discovered={
            "pop_num": source(spec_id=SPEC_KENYA, entities=_THREE_COUNTRIES),
            "pop_denom": source(spec_id=SPEC_OTHER, entities=_THREE_COUNTRIES),
        },
    )

    assert Code.SHARE_AXIS_REQUIRED in _codes(result)
    assert result.workflow is None
    issue = next(i for i in result.issues if i.code is Code.SHARE_AXIS_REQUIRED)
    assert "country" in issue.reason


def test_a_keyed_share_of_the_same_multi_member_shape_is_not_share_axis_required(
    build: Callable[..., CompileResult], decision: DecisionFactory, source: SourceFactory
) -> None:
    # The shape `SHARE_AXIS_REQUIRED` refuses above, keyed by country: the
    # numerator answers for each of the three members its source resolves,
    # so it raises nothing. The denominator is pooled, a whole side, and its
    # source resolves three countries where its requirement names one, so it
    # alone is refused.
    pop_num = DatasetRequirement(
        id="pop_num",
        decision_ids=["measure"],
        measure="population",
        entities=[Entity(name="Kenya", role="country")],
        period=Period(start=2015, end=2024),
    )
    pop_denom = DatasetRequirement(
        id="pop_denom",
        decision_ids=["measure"],
        measure="population",
        entities=[Entity(name="Kenya", role="country")],
        period=Period(start=2015, end=2024),
    )
    pooled = decision(slot="pooled", fixes=DecisionFixes(denominator_scope=DenominatorScope.POOLED))
    share = ShareOfRequirement(
        id="share",
        decision_ids=["measure", "pooled"],
        numerator=RequirementSide(input_id="pop_num"),
        denominator=RequirementSide(input_id="pop_denom"),
        axis="country",
        denominator_scope=DenominatorScope.POOLED,
    )
    plan = _draft([decision(), pooled], [pop_num, pop_denom, share])
    fetch_num = RetrieveOperation(id="fetch_num", requirement_id="pop_num", spec_id=SPEC_KENYA)
    fetch_denom = RetrieveOperation(
        id="fetch_denom", requirement_id="pop_denom", spec_id=SPEC_OTHER
    )
    share_op = ShareOfOperation(
        id="share",
        requirement_id="share",
        numerator=Side(input="fetch_num"),
        denominator=Side(input="fetch_denom"),
        axis="country",
        denominator_scope=DenominatorScope.POOLED,
    )

    result = build(
        plan,
        _translation(fetch_num, fetch_denom, share_op, terminal=["share"]),
        discovered={
            "pop_num": source(spec_id=SPEC_KENYA, entities=_THREE_COUNTRIES),
            "pop_denom": source(spec_id=SPEC_OTHER, entities=_THREE_COUNTRIES),
        },
    )

    assert [(issue.code, issue.requirement_id) for issue in result.issues] == [
        (Code.ENTITY_REQUIRED, "pop_denom"),
        (Code.SOURCE_WIDER_THAN_REQUIREMENT, "pop_denom"),
    ]


def test_an_axis_less_share_of_single_member_sides_compiles_clean(
    build: Callable[..., CompileResult],
    decision: DecisionFactory,
    requirement: RequirementFactory,
    source: SourceFactory,
) -> None:
    # Kenya's debt service as a share of Kenya's GDP: each side already
    # carries exactly one member, so pooling it loses nothing an axis
    # would have named -- the guard is not blanket.
    gdp = requirement(id="annual_gdp", measure="gross domestic product")
    share = ShareOfRequirement(
        id="share",
        decision_ids=["measure"],
        numerator=RequirementSide(input_id="annual_debt_service"),
        denominator=RequirementSide(input_id="annual_gdp"),
    )
    plan = _draft([decision()], [requirement(), gdp, share])
    share_op = ShareOfOperation(
        id="share",
        requirement_id="share",
        numerator=Side(input="debt"),
        denominator=Side(input="gdp"),
    )
    retrieve_gdp = _retrieve(id="gdp", requirement_id="annual_gdp", spec_id=SPEC_OTHER)

    result = build(
        plan,
        _translation(_retrieve(), retrieve_gdp, share_op, terminal=["share"]),
        discovered={"annual_debt_service": source(), "annual_gdp": source(spec_id=SPEC_OTHER)},
    )

    assert result.ok, result.issues
    assert Code.SHARE_AXIS_REQUIRED not in _codes(result)


def test_a_join_step_writes_the_role_columns_both_sources_share(
    build: Callable[..., CompileResult],
    decision: DecisionFactory,
    requirement: RequirementFactory,
    source: SourceFactory,
) -> None:
    gdp = requirement(id="annual_gdp", measure="gross domestic product")
    joined = JoinRequirement(
        id="joined", decision_ids=["measure"], left_id="annual_debt_service", right_id="annual_gdp"
    )
    plan = _draft([decision()], [requirement(), gdp, joined])
    join_op = JoinOperation(id="joined", requirement_id="joined", left="debt", right="gdp")
    retrieve_gdp = _retrieve(id="gdp", requirement_id="annual_gdp", spec_id=SPEC_OTHER)

    result = build(
        plan,
        _translation(_retrieve(), retrieve_gdp, join_op, terminal=["joined"]),
        discovered={"annual_debt_service": source(), "annual_gdp": source(spec_id=SPEC_OTHER)},
    )

    assert result.ok, result.issues
    assert result.workflow is not None
    step = result.workflow.steps[-1]
    assert step.tool.name == "join"
    assert step.args == {"join_keys": ["destination"]}
    assert step.bindings == {
        "frame_ref": Binding(from_step="debt", output=ArtifactKind.FRAME_REF),
        "right_frame_ref": Binding(from_step="gdp", output=ArtifactKind.FRAME_REF),
    }


def test_a_rank_over_a_join_is_one_typed_issue(
    build: Callable[..., CompileResult],
    decision: DecisionFactory,
    requirement: RequirementFactory,
    source: SourceFactory,
) -> None:
    # Regression for #202: a rank reading a join would otherwise reach no
    # writer and no issue, so `compile_plan` would raise `RuntimeError`
    # rather than reporting why. `ranked_op`'s own requirement cites a harmless, unrelated input --
    # a join is terminal by contract, so no requirement may read one and
    # `Plan` itself refuses that -- and the operation's own `input` is what
    # is deliberately made to diverge, exactly as the sibling comparison
    # regression above does.
    gdp = requirement(id="annual_gdp", measure="gross domestic product")
    joined = JoinRequirement(
        id="joined", decision_ids=["measure"], left_id="annual_debt_service", right_id="annual_gdp"
    )
    rank_req = RankRequirement(
        id="ranked",
        decision_ids=["measure", "population"],
        input_id="annual_debt_service",
        direction=RankDirection.HIGHEST_FIRST,
        among=RankAmong.POPULATION,
        reduction=Reduction.SUM,
    )
    population = decision(slot="population", fixes=DecisionFixes(among=RankAmong.POPULATION))
    plan = _draft([decision(), population], [requirement(), gdp, joined, rank_req])
    join_op = JoinOperation(id="joined", requirement_id="joined", left="debt", right="gdp")
    ranked_op = RankOperation(
        id="ranked_op",
        requirement_id="ranked",
        input="joined",
        direction=RankDirection.HIGHEST_FIRST,
        among=RankAmong.POPULATION,
        reduction=Reduction.SUM,
    )
    retrieve_gdp = _retrieve(id="gdp", requirement_id="annual_gdp", spec_id=SPEC_OTHER)

    result = build(
        plan,
        _translation(_retrieve(), retrieve_gdp, join_op, ranked_op, terminal=["ranked_op"]),
        discovered={"annual_debt_service": source(), "annual_gdp": source(spec_id=SPEC_OTHER)},
    )

    assert Code.INPUT_IS_TERMINAL in _codes(result)
    assert result.workflow is None


@pytest.mark.parametrize("terminal_kind", ["compare", "join"], ids=lambda k: k)
@pytest.mark.parametrize("reader_kind", ["growth", "aggregate", "filter"], ids=lambda k: k)
def test_an_aggregate_a_growth_and_a_filter_reading_a_join_or_a_comparison_are_terminal_reads(
    build: Callable[..., CompileResult],
    decision: DecisionFactory,
    requirement: RequirementFactory,
    derived_requirement: DerivedRequirementFactory,
    source: SourceFactory,
    reader_kind: str,
    terminal_kind: str,
) -> None:
    gdp = requirement(id="annual_gdp", measure="gross domestic product")
    terminal_req: EvidenceRequirement
    terminal_op: Operation
    if terminal_kind == "compare":
        terminal_req = ComparisonRequirement(
            id="term",
            decision_ids=["measure"],
            minuend=RequirementSide(input_id="annual_gdp"),
            subtrahend=RequirementSide(input_id="annual_debt_service"),
        )
        terminal_op = CompareOperation(
            id="term", requirement_id="term", left=Side(input="debt"), right=Side(input="gdp")
        )
    else:
        terminal_req = JoinRequirement(
            id="term",
            decision_ids=["measure"],
            left_id="annual_debt_service",
            right_id="annual_gdp",
        )
        terminal_op = JoinOperation(id="term", requirement_id="term", left="debt", right="gdp")
    reader: Operation
    if reader_kind == "growth":
        reader = GrowthOperation(
            id="reader",
            requirement_id="annual_totals",
            input="term",
            window_start=2015,
            window_end=2023,
        )
    elif reader_kind == "aggregate":
        reader = AggregateOperation(
            id="reader",
            requirement_id="annual_totals",
            input="term",
            reduction=Reduction.SUM,
            group_by=(GroupBy.PERIOD,),
        )
    else:
        reader = FilterOperation(
            id="reader",
            requirement_id="annual_totals",
            input="term",
            predicates=(ValueThreshold(op="gt", value=0),),
        )
    plan = _draft([decision()], [requirement(), gdp, terminal_req, derived_requirement()])
    retrieve_gdp = _retrieve(id="gdp", requirement_id="annual_gdp", spec_id=SPEC_OTHER)

    result = build(
        plan,
        _translation(_retrieve(), retrieve_gdp, terminal_op, reader, terminal=["reader"]),
        discovered={"annual_debt_service": source(), "annual_gdp": source(spec_id=SPEC_OTHER)},
    )

    assert Code.INPUT_IS_TERMINAL in _codes(result)
    assert result.workflow is None


def test_a_rank_over_a_keyed_share_of_now_compiles(
    build: Callable[..., CompileResult],
    decision: DecisionFactory,
    source: SourceFactory,
) -> None:
    # share_of is chainable because it gains an axis, and its shape rests on
    # two retrievals. A walk to a single upstream retrieval returns `None`
    # for anything reading a share_of's shape, the case #202 fixes, so this
    # test compiles a rank over a keyed share_of end to end.
    health = DatasetRequirement(
        id="req_health",
        decision_ids=["measure"],
        measure="ODA to health",
        entities=[Entity(name="Kenya", role="donor")],
        period=Period(start=2015, end=2024),
    )
    total = DatasetRequirement(
        id="req_total",
        decision_ids=["measure"],
        measure="total ODA",
        entities=[Entity(name="Kenya", role="donor")],
        period=Period(start=2015, end=2024),
    )
    share_req = ShareOfRequirement(
        id="share",
        decision_ids=["measure", "pooled"],
        numerator=RequirementSide(input_id="req_health"),
        denominator=RequirementSide(input_id="req_total"),
        axis="donor",
        denominator_scope=DenominatorScope.POOLED,
    )
    rank_req = RankRequirement(
        id="ranked",
        decision_ids=["measure", "population"],
        input_id="share",
        direction=RankDirection.HIGHEST_FIRST,
        among=RankAmong.POPULATION,
        reduction=Reduction.MEAN,
    )
    pooled = decision(slot="pooled", fixes=DecisionFixes(denominator_scope=DenominatorScope.POOLED))
    population = decision(slot="population", fixes=DecisionFixes(among=RankAmong.POPULATION))
    plan = _draft([decision(), pooled, population], [health, total, share_req, rank_req])

    health_op = RetrieveOperation(id="health", requirement_id="req_health", spec_id=SPEC_KENYA)
    total_op = RetrieveOperation(id="total", requirement_id="req_total", spec_id=SPEC_OTHER)
    share_op = ShareOfOperation(
        id="share",
        requirement_id="share",
        numerator=Side(input="health"),
        denominator=Side(input="total"),
        axis="donor",
        denominator_scope=DenominatorScope.POOLED,
    )
    ranked_op = RankOperation(
        id="ranked",
        requirement_id="ranked",
        input="share",
        direction=RankDirection.HIGHEST_FIRST,
        among=RankAmong.POPULATION,
        reduction=Reduction.MEAN,
    )
    donor_entities = (ResolvedEntity(role="donor", ref="country/KEN", label="Kenya"),)
    donor_source = source(entities=donor_entities)
    total_source = source(spec_id=SPEC_OTHER, entities=donor_entities)

    result = build(
        plan,
        _translation(health_op, total_op, share_op, ranked_op, terminal=["ranked"]),
        discovered={"req_health": donor_source, "req_total": total_source},
    )

    assert result.ok, result.issues
    assert result.workflow is not None
    share_step = result.workflow.steps[2]
    assert share_step.tool.name == "share_of"
    assert share_step.args == {"numerator_by": ["donor"]}
    step = result.workflow.steps[-1]
    assert step.tool.name == "rank"
    assert step.args == {"rank_by": ["donor"], "ascending": False, "aggregation": "mean"}


def test_degenerate_sides_are_detected_across_a_filter(
    build: Callable[..., CompileResult],
    decision: DecisionFactory,
    requirement: RequirementFactory,
) -> None:
    # The numerator reads a filter of the same retrieval the denominator
    # reads directly; the requirement names one entity, so both sides are
    # Kenya's own series once the filter's single-input chain is walked back.
    kenya_only = (ValueThreshold(op="gt", value=0),)
    filter_req = FilterRequirement(
        id="positive",
        decision_ids=["measure"],
        input_id="annual_debt_service",
        predicates=kenya_only,
    )
    share = ShareOfRequirement(
        id="share",
        decision_ids=["measure"],
        numerator=RequirementSide(input_id="positive"),
        denominator=RequirementSide(input_id="annual_debt_service"),
    )
    plan = _draft([decision()], [requirement(), filter_req, share])
    filter_op = FilterOperation(
        id="positive", requirement_id="positive", input="debt", predicates=kenya_only
    )
    share_op = ShareOfOperation(
        id="share",
        requirement_id="share",
        numerator=Side(input="positive"),
        denominator=Side(input="debt"),
    )

    result = build(plan, _translation(_retrieve(), filter_op, share_op, terminal=["share"]))

    assert Code.DEGENERATE_SIDES in _codes(result)


def test_a_filter_step_writes_a_value_threshold_predicate(
    build: Callable[..., CompileResult],
    decision: DecisionFactory,
    requirement: RequirementFactory,
) -> None:
    filter_req = FilterRequirement(
        id="filtered",
        decision_ids=["measure"],
        input_id="annual_debt_service",
        predicates=(ValueThreshold(op="gt", value=0),),
    )
    plan = _draft([decision()], [requirement(), filter_req])
    filter_op = FilterOperation(
        id="filtered",
        requirement_id="filtered",
        input="debt",
        predicates=(ValueThreshold(op="gt", value=0),),
    )

    result = build(plan, _translation(_retrieve(), filter_op, terminal=["filtered"]))

    assert result.ok, result.issues
    assert result.workflow is not None
    step = result.workflow.steps[1]
    assert step.tool.name == "filter"
    assert step.args == {
        "predicates": [{"kind": "ordered_threshold", "column": "value", "op": "gt", "value": 0}]
    }
    assert step.bindings == {"frame_ref": Binding(from_step="debt", output=ArtifactKind.FRAME_REF)}


def test_a_filter_step_writes_a_category_match_predicate_from_a_resolved_entity(
    build: Callable[..., CompileResult],
    decision: DecisionFactory,
    requirement: RequirementFactory,
    wide_source: SourceFactory,
) -> None:
    filter_req = FilterRequirement(
        id="filtered",
        decision_ids=["measure"],
        input_id="annual_debt_service",
        predicates=(EntityIn(names=("Kenya",)),),
    )
    plan = _draft(
        [decision()],
        [
            requirement(
                entities=[Entity(name="Kenya", role="debtor"), Entity(name="Ghana", role="debtor")]
            ),
            filter_req,
        ],
    )
    filter_op = FilterOperation(
        id="filtered",
        requirement_id="filtered",
        input="debt",
        predicates=(EntityIn(names=("Kenya",)),),
    )

    result = build(
        plan,
        _translation(_retrieve(), filter_op, terminal=["filtered"]),
        discovered={"annual_debt_service": wide_source()},
    )

    assert result.ok, result.issues
    assert result.workflow is not None
    step = result.workflow.steps[1]
    assert step.args == {
        "predicates": [
            {"kind": "category_match", "column": "destination", "values": ["country/KEN"]}
        ]
    }


# --- Coverage is structural ----------------------------------------------------------


def test_a_requirement_no_operation_carries_out_is_reported(
    build: Callable[..., CompileResult],
    decision: DecisionFactory,
    requirements: tuple[EvidenceRequirement, ...],
) -> None:
    plan = _draft([decision()], list(requirements))

    result = build(plan, _translation(_retrieve()))

    assert Code.REQUIREMENT_UNSATISFIED in _codes(result)
    assert result.issues[_codes(result).index(Code.REQUIREMENT_UNSATISFIED)].requirement_id == (
        "annual_totals"
    )
    assert result.workflow is None


def test_a_requirement_carried_out_twice_is_reported(
    build: Callable[..., CompileResult],
    decision: DecisionFactory,
    requirements: tuple[EvidenceRequirement, ...],
) -> None:
    plan = _draft([decision()], list(requirements))

    result = build(
        plan,
        _translation(
            _retrieve(),
            _retrieve(id="debt_again", requirement_id="annual_debt_service"),
            _aggregate(),
        ),
    )

    assert Code.DUPLICATE_SATISFACTION in _codes(result)


def test_an_operation_naming_an_unknown_requirement_is_reported(
    build: Callable[..., CompileResult],
    decision: DecisionFactory,
    requirements: tuple[EvidenceRequirement, ...],
) -> None:
    plan = _draft([decision()], list(requirements))

    result = build(
        plan, _translation(_retrieve(requirement_id="population"), _aggregate(input="debt"))
    )

    assert Code.REQUIREMENT_UNKNOWN in _codes(result)
    assert Code.REQUIREMENT_UNSATISFIED in _codes(result)


def test_an_operation_of_the_wrong_kind_is_a_kind_mismatch(
    build: Callable[..., CompileResult],
    decision: DecisionFactory,
    requirements: tuple[EvidenceRequirement, ...],
) -> None:
    plan = _draft([decision()], list(requirements))
    # "annual_totals" is an aggregate requirement; a growth operation cannot
    # carry it out, whatever its own fields say.
    wrong_kind = GrowthOperation(
        id="totals",
        requirement_id="annual_totals",
        input="debt",
        window_start=2015,
        window_end=2023,
    )

    result = build(plan, _translation(_retrieve(), wrong_kind, terminal=["totals"]))

    assert Code.KIND_MISMATCH in _codes(result)


def test_an_aggregate_disagreeing_on_reduction_is_an_operation_mismatch(
    build: Callable[..., CompileResult],
    decision: DecisionFactory,
    requirements: tuple[EvidenceRequirement, ...],
) -> None:
    # "annual_totals" asks for `sum`; this operation proposes `mean`.
    plan = _draft([decision()], list(requirements))

    result = build(plan, _translation(_retrieve(), _aggregate(reduction=Reduction.MEAN)))

    assert Code.OPERATION_MISMATCH in _codes(result)
    mismatch = next(i for i in result.issues if i.code is Code.OPERATION_MISMATCH)
    assert "reduction" in mismatch.reason


def test_an_operation_reading_nothing_is_reported(
    build: Callable[..., CompileResult],
    decision: DecisionFactory,
    requirements: tuple[EvidenceRequirement, ...],
) -> None:
    plan = _draft([decision()], list(requirements))

    result = build(plan, _translation(_retrieve(), _aggregate(input="ghost")))

    assert Code.DANGLING_INPUT in _codes(result)


def test_operations_reading_each_other_in_a_cycle_are_reported_once(
    build: Callable[..., CompileResult],
    decision: DecisionFactory,
    requirements: tuple[EvidenceRequirement, ...],
) -> None:
    plan = _draft([decision()], list(requirements))
    a = _aggregate(id="a", requirement_id="annual_totals", input="b")
    b = _aggregate(id="b", requirement_id="annual_totals", input="a")

    result = build(plan, _translation(_retrieve(), a, b, terminal=["a"]))

    assert _codes(result).count(Code.CYCLE) == 1


@pytest.mark.parametrize("reader_kind", ["growth", "aggregate", "compare"], ids=lambda k: k)
def test_an_operation_reading_a_comparison_is_reported(
    build: Callable[..., CompileResult],
    decision: DecisionFactory,
    requirement: RequirementFactory,
    derived_requirement: DerivedRequirementFactory,
    wide_source: SourceFactory,
    reader_kind: str,
) -> None:
    # A comparison rests on two requirements, so nothing single stands behind
    # its result for a reader to be checked against: it is terminal here.
    comparison = ComparisonRequirement(
        id="gap",
        decision_ids=["measure", "comparison"],
        minuend=RequirementSide(input_id="annual_debt_service", entity="Ghana"),
        subtrahend=RequirementSide(input_id="annual_debt_service", entity="Kenya"),
    )
    plan = _two_country_plan(decision, requirement, comparison, derived_requirement())
    gap = CompareOperation(
        id="gap",
        requirement_id="gap",
        left=Side(input="debt", entity="Kenya"),
        right=Side(input="debt", entity="Ghana"),
    )
    reader: Operation
    if reader_kind == "growth":
        reader = GrowthOperation(
            id="change",
            requirement_id="annual_totals",
            input="gap",
            window_start=2015,
            window_end=2023,
        )
    elif reader_kind == "aggregate":
        reader = AggregateOperation(
            id="totals",
            requirement_id="annual_totals",
            input="gap",
            reduction=Reduction.SUM,
            group_by=(GroupBy.PERIOD,),
        )
    else:
        reader = CompareOperation(
            id="again",
            requirement_id="annual_totals",
            left=Side(input="gap"),
            right=Side(input="debt", entity="Kenya"),
        )

    result = build(
        plan,
        _translation(_retrieve(), gap, reader, terminal=[reader.id]),
        discovered={"annual_debt_service": wide_source()},
    )

    assert Code.INPUT_IS_TERMINAL in _codes(result)
    assert result.workflow is None


def test_a_terminal_that_is_not_an_operation_is_reported(
    build: Callable[..., CompileResult],
    decision: DecisionFactory,
    requirements: tuple[EvidenceRequirement, ...],
) -> None:
    plan = _draft([decision()], list(requirements))

    result = build(plan, _translation(_retrieve(), _aggregate(), terminal=["answer"]))

    assert Code.UNKNOWN_TERMINAL in _codes(result)


def test_a_terminal_another_operation_reads_compiles(
    build: Callable[..., CompileResult],
    decision: DecisionFactory,
    requirement: RequirementFactory,
    derived_requirement: DerivedRequirementFactory,
) -> None:
    # `positive` answers an ask and `totals` reads it, so it is both terminal and read.
    positive = (ValueThreshold(op="gt", value=0),)
    filter_req = FilterRequirement(
        id="positive",
        decision_ids=["measure"],
        input_id="annual_debt_service",
        predicates=positive,
    )
    plan = _draft(
        [decision()], [requirement(), filter_req, derived_requirement(input_id="positive")]
    )
    filter_op = FilterOperation(
        id="positive", requirement_id="positive", input="debt", predicates=positive
    )

    result = build(
        plan,
        _translation(
            _retrieve(), filter_op, _aggregate(input="positive"), terminal=["positive", "totals"]
        ),
    )

    assert result.ok, result.issues
    assert result.workflow is not None
    assert result.workflow.terminal_steps == ("positive", "totals")


def test_a_translation_that_stops_at_the_retrieval_compiles(
    build: Callable[..., CompileResult],
    decision: DecisionFactory,
    requirement: RequirementFactory,
) -> None:
    # Under answer_mode=series, the retrieved frame is the answer. A question
    # that asks to see a series is satisfied by the retrieval itself, and the
    # frame carries one series per entity for a chart to draw, so no
    # operation has to read it.
    plan = _draft([decision()], [requirement()], answer_mode=AnswerMode.SERIES)

    result = build(plan, _translation(_retrieve(), terminal=["debt"]))

    assert result.issues == ()
    assert result.workflow is not None
    assert result.workflow.terminal_steps == ("debt",)
    (step,) = result.workflow.steps
    assert step.tool.name == "get_data"
    assert step.args == {"spec_id": SPEC_KENYA, "without_party_totals": True}


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({"served_count": 2}, {"stat_vars": "separate"}),
        ({"served_count": None}, {"stat_vars": "separate"}),
        ({"served_count": 1}, {}),
    ],
    ids=["a-family", "unsettled", "one-served"],
)
def test_a_retrieval_writes_the_spec_id_and_a_family_s_reading_alone(
    build: Callable[..., CompileResult],
    decision: DecisionFactory,
    requirement: RequirementFactory,
    source: SourceFactory,
    overrides: dict[str, object],
    expected: dict[str, object],
) -> None:
    # The mint resolves the family's members itself, so a retrieval posts no
    # StatVar: a family states its reading, and a source serving one StatVar
    # states nothing beside its spec_id and, for a flow source, its party totals.
    plan = _draft([decision()], [requirement()], answer_mode=AnswerMode.SERIES)

    result = build(
        plan,
        _translation(_retrieve(), terminal=["debt"]),
        discovered={"annual_debt_service": source(**overrides)},
    )

    assert result.issues == ()
    assert result.workflow is not None
    (step,) = result.workflow.steps
    assert (
        step.args
        == {
            "spec_id": SPEC_KENYA,
            "without_party_totals": True,
        }
        | expected
    )


def test_an_operation_feeding_no_terminal_is_reported(
    build: Callable[..., CompileResult],
    decision: DecisionFactory,
    requirements: tuple[EvidenceRequirement, ...],
) -> None:
    plan = _draft([decision()], list(requirements))
    stray = _aggregate(id="stray", requirement_id="annual_totals", reduction=Reduction.SUM)

    result = build(plan, _translation(_retrieve(), _aggregate(), stray, terminal=["totals"]))

    assert Code.OPERATION_UNREACHABLE in _codes(result)
    unreachable = next(i for i in result.issues if i.code is Code.OPERATION_UNREACHABLE)
    assert unreachable.operation_id == "stray"


def test_a_repeated_operation_id_is_reported(
    build: Callable[..., CompileResult],
    decision: DecisionFactory,
    requirements: tuple[EvidenceRequirement, ...],
) -> None:
    plan = _draft([decision()], list(requirements))

    result = build(plan, _translation(_retrieve(), _aggregate(), _aggregate()))

    assert Code.DUPLICATE_ID in _codes(result)


# --- Sources ----------------------------------------------------------------------------


def test_a_retrieval_without_a_source_is_reported(
    build: Callable[..., CompileResult],
    decision: DecisionFactory,
    requirements: tuple[EvidenceRequirement, ...],
) -> None:
    plan = _draft([decision()], list(requirements))

    result = build(plan, _translation(_retrieve(spec_id=None), _aggregate()))

    assert Code.SOURCE_UNSELECTED in _codes(result)


def test_a_source_discovery_did_not_record_is_reported(
    build: Callable[..., CompileResult],
    decision: DecisionFactory,
    requirements: tuple[EvidenceRequirement, ...],
) -> None:
    plan = _draft([decision()], list(requirements))

    result = build(plan, _translation(_retrieve(spec_id=SPEC_OTHER), _aggregate()), discovered={})

    assert Code.SOURCE_NOT_DISCOVERED_FOR_REQUIREMENT in _codes(result)


def test_a_spec_id_discovered_for_another_requirement_is_refused(
    build: Callable[..., CompileResult],
    decision: DecisionFactory,
    requirement: RequirementFactory,
    source: SourceFactory,
) -> None:
    # SPEC_KENYA was discovered for "annual_debt_service", not "annual_gdp";
    # naming it for the wrong requirement's retrieval is refused even though
    # the spec_id itself is real.
    gdp = requirement(id="annual_gdp", measure="gross domestic product")
    comparison = ComparisonRequirement(
        id="gap",
        decision_ids=["measure"],
        minuend=RequirementSide(input_id="annual_gdp"),
        subtrahend=RequirementSide(input_id="annual_debt_service"),
    )
    plan = _draft([decision()], [requirement(), gdp, comparison])
    retrieve_gdp = _retrieve(id="gdp", requirement_id="annual_gdp", spec_id=SPEC_KENYA)
    gap = CompareOperation(
        id="gap", requirement_id="gap", left=Side(input="debt"), right=Side(input="gdp")
    )

    result = build(
        plan,
        _translation(_retrieve(), retrieve_gdp, gap, terminal=["gap"]),
        discovered={"annual_debt_service": source(), "annual_gdp": source(spec_id=SPEC_OTHER)},
    )

    assert Code.SOURCE_NOT_DISCOVERED_FOR_REQUIREMENT in _codes(result)
    bad = next(i for i in result.issues if i.code is Code.SOURCE_NOT_DISCOVERED_FOR_REQUIREMENT)
    assert bad.operation_id == "gdp"


def test_a_pinned_source_the_retrieval_does_not_name_is_refused(
    build: Callable[..., CompileResult],
    pinning_decision: Callable[..., Decision],
    requirement: RequirementFactory,
    derived_requirement: DerivedRequirementFactory,
    source: SourceFactory,
) -> None:
    pinned = requirement(source_spec_id=SPEC_KENYA)
    plan = _draft([pinning_decision(SPEC_KENYA)], [pinned, derived_requirement()])

    result = build(
        plan,
        _translation(_retrieve(spec_id=SPEC_OTHER), _aggregate()),
        discovered={"annual_debt_service": source(spec_id=SPEC_OTHER)},
    )

    assert Code.SOURCE_PIN_VIOLATED in _codes(result)


def test_a_source_that_cannot_serve_the_requirement_is_reported_with_the_finding(
    build: Callable[..., CompileResult],
    decision: DecisionFactory,
    requirement: RequirementFactory,
    derived_requirement: DerivedRequirementFactory,
) -> None:
    plan = _draft(
        [decision()], [requirement(entities=[Entity(name="Uganda")]), derived_requirement()]
    )

    result = build(plan, _translation(_retrieve(), _aggregate()))

    assert Code.SOURCE_UNUSABLE in _codes(result)
    unusable = next(i for i in result.issues if i.code is Code.SOURCE_UNUSABLE)
    assert "no entity named 'Uganda'" in unusable.reason


def test_findings_that_do_not_block_come_back_as_warnings(
    build: Callable[..., CompileResult],
    decision: DecisionFactory,
    requirements: tuple[EvidenceRequirement, ...],
    source: SourceFactory,
) -> None:
    plan = _draft([decision()], list(requirements))

    result = build(
        plan,
        _translation(_retrieve(), _aggregate()),
        discovered={"annual_debt_service": source(coverage_window=None)},
    )

    assert result.ok
    assert "period_unverified" in [w.code.value for w in result.warnings]


def test_a_source_wider_than_the_requirement_is_reported_where_an_operation_pools_it(
    build: Callable[..., CompileResult],
    decision: DecisionFactory,
    requirements: tuple[EvidenceRequirement, ...],
    wide_source: SourceFactory,
) -> None:
    # The requirement names Kenya; the source resolves Ghana too, and `get_data`
    # is called with no `scope`, so the frame carries both. A period-only
    # sum then folds the two into one figure the plan never stated.
    plan = _draft([decision()], list(requirements))

    result = build(
        plan,
        _translation(_retrieve(), _aggregate()),
        discovered={"annual_debt_service": wide_source()},
    )

    assert _codes(result) == [Code.SOURCE_WIDER_THAN_REQUIREMENT]
    (issue,) = result.issues
    assert (issue.operation_id, issue.requirement_id) == ("totals", "annual_debt_service")
    assert "['Ghana']" in issue.reason
    assert "'annual_debt_service'" in issue.reason
    assert "'totals'" in issue.reason


def test_a_source_resolving_another_role_is_not_wider_than_the_requirement(
    build: Callable[..., CompileResult],
    decision: DecisionFactory,
    requirements: tuple[EvidenceRequirement, ...],
    source: SourceFactory,
) -> None:
    # The requirement names Kenya, which resolves as the recipient; the source
    # also resolves China as the creditor. A period-only sum over the
    # debtor-creditor pair pools nothing the plan did not state, and China is
    # not an entity the plan could name to narrow it.
    plan = _draft([decision()], list(requirements))

    result = build(
        plan,
        _translation(_retrieve(), _aggregate()),
        discovered={"annual_debt_service": _two_role_source(source)},
    )

    assert result.ok, result.issues


def test_a_whole_side_from_a_wider_source_is_reported(
    build: Callable[..., CompileResult],
    decision: DecisionFactory,
    requirement: RequirementFactory,
    source: SourceFactory,
    wide_source: SourceFactory,
) -> None:
    # By the plan's lights the side is unambiguous -- its requirement names
    # one entity -- so it is the source resolving two that makes a whole side
    # pool them: `entity_required` on the source's count, and the pooling.
    gdp = requirement(id="annual_gdp", measure="gross domestic product")
    comparison = ComparisonRequirement(
        id="gap",
        decision_ids=["measure"],
        minuend=RequirementSide(input_id="annual_gdp"),
        subtrahend=RequirementSide(input_id="annual_debt_service"),
    )
    plan = _draft([decision()], [requirement(), gdp, comparison])
    gap = CompareOperation(
        id="gap", requirement_id="gap", left=Side(input="debt"), right=Side(input="gdp")
    )
    retrieve_gdp = _retrieve(id="gdp", requirement_id="annual_gdp", spec_id=SPEC_OTHER)

    result = build(
        plan,
        _translation(_retrieve(), retrieve_gdp, gap, terminal=["gap"]),
        discovered={
            "annual_debt_service": wide_source(),
            "annual_gdp": source(spec_id=SPEC_OTHER),
        },
    )

    assert _codes(result) == [Code.ENTITY_REQUIRED, Code.SOURCE_WIDER_THAN_REQUIREMENT]
    assert "resolves several entities ['Kenya', 'Ghana']" in result.issues[0].reason
    assert result.issues[1].operation_id == "gap"


def test_a_filter_naming_the_entity_narrows_a_wider_source(
    build: Callable[..., CompileResult],
    decision: DecisionFactory,
    requirement: RequirementFactory,
    derived_requirement: DerivedRequirementFactory,
    wide_source: SourceFactory,
) -> None:
    # The filter keeps Kenya's rows, so the sum downstream pools nothing the
    # plan did not state.
    kenya_only = (EntityIn(names=("Kenya",)),)
    filter_req = FilterRequirement(
        id="kenya_only",
        decision_ids=["measure"],
        input_id="annual_debt_service",
        predicates=kenya_only,
    )
    plan = _draft(
        [decision()], [requirement(), filter_req, derived_requirement(input_id="kenya_only")]
    )
    filter_op = FilterOperation(
        id="kenya_only", requirement_id="kenya_only", input="debt", predicates=kenya_only
    )

    result = build(
        plan,
        _translation(_retrieve(), filter_op, _aggregate(input="kenya_only")),
        discovered={"annual_debt_service": wide_source()},
    )

    assert result.ok, result.issues


def test_an_unqualified_filter_does_not_narrow_a_role_it_never_matched(
    build: Callable[..., CompileResult],
    decision: DecisionFactory,
    derived_requirement: DerivedRequirementFactory,
    source: SourceFactory,
) -> None:
    # The requirement names Kenya as debtor and France and Germany as
    # creditor; the source resolves a third, unnamed creditor, China. The
    # filter keeps Kenya alone, with no role stated -- Kenya resolves under
    # the debtor column only, so that is the one column it narrows. The
    # creditor column stays surplus and un-narrowed, so the aggregate
    # downstream, which pools every role together, still pools China's debt
    # into the total though the plan never named it.
    kenya_only = (EntityIn(names=("Kenya",)),)
    filter_req = FilterRequirement(
        id="kenya_only",
        decision_ids=["measure"],
        input_id="annual_debt_service",
        predicates=kenya_only,
    )
    plan = _draft(
        [decision()],
        [_two_role_requirement(), filter_req, derived_requirement(input_id="kenya_only")],
    )
    filter_op = FilterOperation(
        id="kenya_only", requirement_id="kenya_only", input="debt", predicates=kenya_only
    )
    surplus_creditor_source = _two_role_creditor_source(
        source,
        entities=(
            ResolvedEntity(role="debtor", ref="country/KEN", label="Kenya"),
            ResolvedEntity(role="creditor", ref="country/FRA", label="France"),
            ResolvedEntity(role="creditor", ref="country/DEU", label="Germany"),
            ResolvedEntity(role="creditor", ref="country/CHN", label="China"),
        ),
    )

    result = build(
        plan,
        _translation(_retrieve(), filter_op, _aggregate(input="kenya_only")),
        discovered={"annual_debt_service": surplus_creditor_source},
    )

    width_issues = [i for i in result.issues if i.code is Code.SOURCE_WIDER_THAN_REQUIREMENT]
    assert len(width_issues) == 1, result.issues
    assert width_issues[0].operation_id == "totals"


def test_a_side_naming_its_entity_narrows_a_wider_source(
    build: Callable[..., CompileResult],
    decision: DecisionFactory,
    requirement: RequirementFactory,
    source: SourceFactory,
    wide_source: SourceFactory,
) -> None:
    gdp = requirement(id="annual_gdp", measure="gross domestic product")
    share = ShareOfRequirement(
        id="share",
        decision_ids=["measure"],
        numerator=RequirementSide(input_id="annual_debt_service", entity="Kenya"),
        denominator=RequirementSide(input_id="annual_gdp"),
    )
    plan = _draft([decision()], [requirement(), gdp, share])
    share_op = ShareOfOperation(
        id="share",
        requirement_id="share",
        numerator=Side(input="debt", entity="Kenya"),
        denominator=Side(input="gdp"),
    )
    retrieve_gdp = _retrieve(id="gdp", requirement_id="annual_gdp", spec_id=SPEC_OTHER)

    result = build(
        plan,
        _translation(_retrieve(), retrieve_gdp, share_op, terminal=["share"]),
        discovered={
            "annual_debt_service": wide_source(),
            "annual_gdp": source(spec_id=SPEC_OTHER),
        },
    )

    assert result.ok, result.issues
    assert result.workflow is not None
    assert result.workflow.steps[-1].args == {"numerator_entity": {"destination": "country/KEN"}}


def _whole_side_of(kind: str, filtered: str) -> tuple[EvidenceRequirement, Operation]:
    """A `kind` requirement and its operation reading `filtered` as a whole side against GDP."""
    plan_whole, plan_gdp = (
        RequirementSide(input_id=filtered),
        RequirementSide(input_id="annual_gdp"),
    )
    whole, gdp = Side(input=filtered), Side(input="gdp")
    if kind == "compare":
        return (
            ComparisonRequirement(
                id="gap", decision_ids=["measure"], minuend=plan_gdp, subtrahend=plan_whole
            ),
            CompareOperation(id="gap", requirement_id="gap", left=whole, right=gdp),
        )
    if kind == "sum_of":
        return (
            SumOfRequirement(id="gap", decision_ids=["measure"], left=plan_whole, right=plan_gdp),
            SumOfOperation(id="gap", requirement_id="gap", left=whole, right=gdp),
        )
    return (
        ShareOfRequirement(
            id="gap", decision_ids=["measure"], numerator=plan_gdp, denominator=plan_whole
        ),
        ShareOfOperation(id="gap", requirement_id="gap", numerator=gdp, denominator=whole),
    )


_THREE_DESTINATIONS = (
    ResolvedEntity(role="destination", ref="country/KEN", label="Kenya"),
    ResolvedEntity(role="destination", ref="country/GHA", label="Ghana"),
    ResolvedEntity(role="destination", ref="country/UGA", label="Uganda"),
)


@pytest.mark.parametrize("kind", ["compare", "sum_of", "share_of"])
@pytest.mark.parametrize(
    ("named", "kept", "required"),
    [
        (("Kenya", "Ghana"), ("Kenya",), False),
        (("Kenya", "Ghana"), ("Kenya", "Ghana"), True),
        (("Kenya", "Ghana", "Uganda"), ("Kenya", "Ghana"), True),
    ],
    ids=["one_kept", "two_kept", "two_of_three_kept"],
)
def test_a_whole_side_reading_a_filter_counts_the_entities_it_kept(
    build: Callable[..., CompileResult],
    decision: DecisionFactory,
    requirement: RequirementFactory,
    source: SourceFactory,
    wide_source: SourceFactory,
    kind: str,
    named: tuple[str, ...],
    kept: tuple[str, ...],
    required: bool,
) -> None:
    # The side reads the filtered frame. Keeping Kenya alone leaves one
    # series, keeping two leaves two that the side would pool, and the issue
    # names those two.
    predicates = (EntityIn(names=kept),)
    filtered = FilterRequirement(
        id="kept", decision_ids=["measure"], input_id="annual_debt_service", predicates=predicates
    )
    debt = requirement(
        decision_ids=["measure", "comparison"],
        entities=[Entity(name=name, role="debtor") for name in named],
    )
    gdp = requirement(id="annual_gdp", measure="gross domestic product")
    derived, operation = _whole_side_of(kind, "kept")
    comparison_decision = decision(
        slot="comparison",
        options=["Kenya against the others", "Kenya on its own"],
        chosen="Kenya against the others",
    )
    plan = _draft([decision(), comparison_decision], [debt, filtered, gdp, derived])
    filter_op = FilterOperation(
        id="kept", requirement_id="kept", input="debt", predicates=predicates
    )
    retrieve_gdp = _retrieve(id="gdp", requirement_id="annual_gdp", spec_id=SPEC_OTHER)

    result = build(
        plan,
        _translation(_retrieve(), retrieve_gdp, filter_op, operation, terminal=["gap"]),
        discovered={
            "annual_debt_service": wide_source(
                entities=_THREE_DESTINATIONS, **_pinned("DebtServicePaid")
            ),
            "annual_gdp": source(
                spec_id=SPEC_OTHER,
                **_pinned("GrossDomesticProduct"),
            ),
        },
    )

    pooling = [issue for issue in result.issues if issue.code is Code.ENTITY_REQUIRED]
    assert bool(pooling) is required, result.issues
    assert result.ok is not required, result.issues
    if required:
        assert f"{list(kept)!r}; say which" in pooling[0].reason


def test_an_aggregate_grouping_by_entity_keeps_a_wider_sources_entities_apart(
    build: Callable[..., CompileResult],
    decision: DecisionFactory,
    requirement: RequirementFactory,
    derived_requirement: DerivedRequirementFactory,
    wide_source: SourceFactory,
) -> None:
    plan = _draft(
        [decision()],
        [requirement(), derived_requirement(group_by=(GroupBy.ENTITY, GroupBy.PERIOD))],
    )

    result = build(
        plan,
        _translation(_retrieve(), _aggregate(group_by=(GroupBy.ENTITY, GroupBy.PERIOD))),
        discovered={"annual_debt_service": wide_source()},
    )

    assert result.ok, result.issues
    assert result.workflow is not None
    # The claim is that the entities stay apart, which is the role column
    # reaching `group_by` -- not merely that nothing was reported.
    assert result.workflow.steps[-1].args["group_by"] == ["destination", "time_period"]


def test_a_growth_over_a_wider_source_keeps_its_entities_apart(
    build: Callable[..., CompileResult],
    decision: DecisionFactory,
    requirement: RequirementFactory,
    wide_source: SourceFactory,
) -> None:
    # A growth writes the requirement's role columns as `entity_by`, so a
    # surplus entity stays one row apart from the named one rather than
    # pooled into a single figure.
    growth_req = GrowthRequirement(
        id="growth_of_debt",
        decision_ids=["measure"],
        input_id="annual_debt_service",
        window_start=2015,
        window_end=2023,
    )
    plan = _draft([decision()], [requirement(), growth_req])
    change = GrowthOperation(
        id="change",
        requirement_id="growth_of_debt",
        input="debt",
        window_start=2015,
        window_end=2023,
    )

    result = build(
        plan,
        _translation(_retrieve(), change, terminal=["change"]),
        discovered={"annual_debt_service": wide_source()},
    )

    assert result.ok, result.issues


def test_a_two_branch_graph_reports_source_width_once_not_once_per_path(
    build: Callable[..., CompileResult],
    decision: DecisionFactory,
    requirement: RequirementFactory,
    wide_source: SourceFactory,
) -> None:
    # Two period-only filters both read the one wide retrieval directly,
    # and both feed a compare as its two (whole, unnamed) sides. Reading
    # the compare's own two inputs once reports exactly one issue, whatever
    # else the compare's degenerate self-comparison and its two whole sides
    # report, where a walk restarting from the retrieval for each reader
    # would report it once per branch.
    from_2018 = FilterRequirement(
        id="from_2018",
        decision_ids=["measure"],
        input_id="annual_debt_service",
        predicates=(PeriodThreshold(op="gte", year=2018),),
    )
    filter_req = FilterRequirement(
        id="just_period",
        decision_ids=["measure"],
        input_id="annual_debt_service",
        predicates=(PeriodThreshold(op="lte", year=2022),),
    )
    comparison = ComparisonRequirement(
        id="gap",
        decision_ids=["measure"],
        minuend=RequirementSide(input_id="just_period"),
        subtrahend=RequirementSide(input_id="from_2018"),
    )
    plan = _draft([decision()], [requirement(), from_2018, filter_req, comparison])
    from_2018_op = FilterOperation(
        id="from_2018",
        requirement_id="from_2018",
        input="debt",
        predicates=(PeriodThreshold(op="gte", year=2018),),
    )
    filter_op = FilterOperation(
        id="just_period",
        requirement_id="just_period",
        input="debt",
        predicates=(PeriodThreshold(op="lte", year=2022),),
    )
    gap = CompareOperation(
        id="gap",
        requirement_id="gap",
        left=Side(input="from_2018"),
        right=Side(input="just_period"),
    )

    result = build(
        plan,
        _translation(_retrieve(), from_2018_op, filter_op, gap, terminal=["gap"]),
        discovered={"annual_debt_service": wide_source()},
    )

    width_issues = [i for i in result.issues if i.code is Code.SOURCE_WIDER_THAN_REQUIREMENT]
    assert len(width_issues) == 1, result.issues
    assert width_issues[0].operation_id == "gap"


def _bound_to(**unit: Any) -> dict[str, Any]:
    """Source fields for a definition bound by request to `unit`, as `search` binds OECD
    DAC2a for "constant prices": the frame is converted on fetch, while the source's
    own unit ref stays `CurrencyUSD`."""
    return {
        "unit_ref": "CurrencyUSD",
        "unit_label": "US Dollar",
        "identity": {"facets": {"unit": {"state": "value", "request": unit}}},
    }


@pytest.mark.parametrize(
    ("source_fields", "refused_by"),
    [
        pytest.param(
            _bound_to(currency="USD", prices="constant", base_year=2024),
            None,
            id="bound-by-request-to-constant-2024",
        ),
        pytest.param(
            {"unit_ref": "ConstantUSD_2024", "unit_label": "Constant 2024 US dollars"},
            None,
            id="unit-id-in-constant-2024",
        ),
        pytest.param(
            {
                "unit_ref": "undata/UNIT_MEASURE-CR_USD_K",
                "unit_label": "United States dollars at constant prices",
            },
            EvidenceIssueCode.UNIT_PRICE_BASIS_UNVERIFIED,
            id="constant-prices-of-an-unstated-base-year",
        ),
        pytest.param(
            {"unit_ref": "SDG_CON_USD", "unit_label": "Constant USD"},
            EvidenceIssueCode.UNIT_PRICE_BASIS_UNVERIFIED,
            id="a-unit-id-in-constant-prices-of-an-unstated-base-year",
        ),
        pytest.param(
            {
                "unit_ref": "undata/UNIT_MEASURE-CR_USD_K",
                "unit_label": "United States dollars at constant 2024 prices",
            },
            EvidenceIssueCode.UNIT_PRICE_BASIS_UNVERIFIED,
            id="constant-2024-prices-stated-only-in-the-label",
        ),
        pytest.param(
            _bound_to(currency="USD", prices="current"),
            EvidenceIssueCode.UNIT_PRICE_BASIS_MISMATCH,
            id="bound-by-request-to-current-prices",
        ),
        pytest.param(
            _bound_to(currency="USD", prices="constant", base_year=2015),
            EvidenceIssueCode.UNIT_PRICE_BASIS_MISMATCH,
            id="bound-by-request-to-another-base-year",
        ),
        pytest.param(
            {"unit_ref": "CurrencyUSD", "unit_label": "US Dollar"},
            EvidenceIssueCode.UNIT_PRICE_BASIS_UNVERIFIED,
            id="a-unit-stating-no-price-basis",
        ),
    ],
)
def test_a_constant_price_requirement_is_met_by_a_source_in_those_prices(
    build: Callable[..., CompileResult],
    decision: DecisionFactory,
    requirement: RequirementFactory,
    derived_requirement: DerivedRequirementFactory,
    source: SourceFactory,
    source_fields: dict[str, Any],
    refused_by: EvidenceIssueCode | None,
) -> None:
    # No operation restates a series in other prices, so a requirement in
    # constant 2024 prices compiles over a source whose request or unit id
    # states them. It is refused over one in current prices, in another
    # base year, stating no basis or no base year, or stating constant
    # prices only in its label, which a retrieved series does not carry.
    plan = _draft(
        [decision()],
        [
            requirement(unit=Unit(currency="USD", prices="constant", base_year=2024)),
            derived_requirement(),
        ],
    )

    result = build(
        plan,
        _translation(_retrieve(), _aggregate()),
        discovered={"annual_debt_service": source(**source_fields)},
    )

    assert [(i.code, i.finding) for i in result.issues] == (
        [(Code.SOURCE_UNUSABLE, refused_by)] if refused_by is not None else []
    )


# --- Operations against requirements -----------------------------------------------------


def test_a_growth_window_outside_the_requirements_period_is_reported(
    build: Callable[..., CompileResult],
    decision: DecisionFactory,
    requirement: RequirementFactory,
) -> None:
    # `growth_req`'s own window has to fit the dataset's period -- `Plan`
    # itself refuses otherwise -- so the operation's window is what is
    # deliberately made to fall outside it, which also disagrees with the
    # requirement it carries out (`operation_mismatch`, on `window_start`).
    growth_req = GrowthRequirement(
        id="growth_of_debt",
        decision_ids=["measure"],
        input_id="annual_debt_service",
        window_start=2016,
        window_end=2020,
    )
    plan = _draft([decision()], [requirement(period=Period(start=2015, end=2024)), growth_req])
    change = GrowthOperation(
        id="change",
        requirement_id="growth_of_debt",
        input="debt",
        window_start=2010,
        window_end=2023,
    )

    result = build(plan, _translation(_retrieve(), change, terminal=["change"]))

    assert Code.WINDOW_OUTSIDE_PERIOD in _codes(result)


def test_a_side_naming_an_entity_the_requirement_lacks_is_reported(
    build: Callable[..., CompileResult],
    decision: DecisionFactory,
    requirement: RequirementFactory,
    wide_source: SourceFactory,
) -> None:
    # The requirement's own side has to name an entity its dataset declares
    # -- `Plan` itself refuses otherwise -- so the operation is what
    # deliberately names one the dataset lacks, which also disagrees with
    # the requirement it carries out (`operation_mismatch`, on `right`).
    gap = CompareOperation(
        id="gap",
        requirement_id="gap",
        left=Side(input="debt", entity="Kenya"),
        right=Side(input="debt", entity="Uganda"),
    )
    comparison = ComparisonRequirement(
        id="gap",
        decision_ids=["measure", "comparison"],
        minuend=RequirementSide(input_id="annual_debt_service", entity="Ghana"),
        subtrahend=RequirementSide(input_id="annual_debt_service", entity="Kenya"),
    )

    result = build(
        _two_country_plan(decision, requirement, comparison),
        _translation(_retrieve(), gap, terminal=["gap"]),
        discovered={"annual_debt_service": wide_source()},
    )

    assert Code.UNKNOWN_ENTITY in _codes(result)


def test_a_whole_side_from_a_requirement_with_several_entities_is_reported(
    build: Callable[..., CompileResult],
    decision: DecisionFactory,
    requirement: RequirementFactory,
    wide_source: SourceFactory,
) -> None:
    gap = CompareOperation(
        id="gap",
        requirement_id="gap",
        left=Side(input="debt"),
        right=Side(input="debt", entity="Ghana"),
    )
    comparison = ComparisonRequirement(
        id="gap",
        decision_ids=["measure", "comparison"],
        minuend=RequirementSide(input_id="annual_debt_service", entity="Ghana"),
        subtrahend=RequirementSide(input_id="annual_debt_service"),
    )

    result = build(
        _two_country_plan(decision, requirement, comparison),
        _translation(_retrieve(), gap, terminal=["gap"]),
        discovered={"annual_debt_service": wide_source()},
    )

    assert _codes(result) == [Code.ENTITY_REQUIRED]
    # A compare pairs yearly series, so the side pools per year only.
    assert _pools(result, one_figure=False)


@pytest.mark.parametrize(
    ("denominator", "required"), [("totals", False), ("debt", True)], ids=["total", "both"]
)
def test_a_whole_side_over_a_total_of_several_entities_reads_one_series(
    build: Callable[..., CompileResult],
    decision: DecisionFactory,
    requirement: RequirementFactory,
    derived_requirement: DerivedRequirementFactory,
    wide_source: SourceFactory,
    denominator: str,
    required: bool,
) -> None:
    # Kenya's share of its and Ghana's debt service. The aggregate by period
    # has summed the two debtors into one series, which a whole side reads
    # as it is. Read before the aggregate, the side would pool them.
    share = ShareOfRequirement(
        id="gap",
        decision_ids=["measure"],
        numerator=RequirementSide(input_id="annual_debt_service", entity="Kenya"),
        denominator=RequirementSide(
            input_id="annual_totals" if denominator == "totals" else "annual_debt_service"
        ),
    )
    plan = _two_country_plan(decision, requirement, share, derived_requirement())
    operation = ShareOfOperation(
        id="gap",
        requirement_id="gap",
        numerator=Side(input="debt", entity="Kenya"),
        denominator=Side(input=denominator),
    )

    result = build(
        plan,
        _translation(_retrieve(), _aggregate(), operation, terminal=["gap"]),
        discovered={"annual_debt_service": wide_source()},
    )

    assert (Code.ENTITY_REQUIRED in _codes(result)) is required
    assert result.ok is not required, result.issues


def test_comparing_a_side_with_itself_is_reported(
    build: Callable[..., CompileResult],
    decision: DecisionFactory,
    requirement: RequirementFactory,
    wide_source: SourceFactory,
) -> None:
    gap = CompareOperation(
        id="gap",
        requirement_id="gap",
        left=Side(input="debt", entity="Kenya"),
        right=Side(input="debt", entity="Kenya"),
    )
    # `ComparisonRequirement` itself refuses two identical sides, so the
    # requirement's own right side has to differ from the operation's --
    # `mismatches` then also reports that side, alongside `degenerate_sides`.
    comparison = ComparisonRequirement(
        id="gap",
        decision_ids=["measure", "comparison"],
        minuend=RequirementSide(input_id="annual_debt_service", entity="Ghana"),
        subtrahend=RequirementSide(input_id="annual_debt_service", entity="Kenya"),
    )

    result = build(
        _two_country_plan(decision, requirement, comparison),
        _translation(_retrieve(), gap, terminal=["gap"]),
        discovered={"annual_debt_service": wide_source()},
    )

    assert Code.DEGENERATE_SIDES in _codes(result)


def test_a_share_of_a_series_against_itself_through_two_aggregates_is_reported(
    build: Callable[..., CompileResult],
    decision: DecisionFactory,
    requirement: RequirementFactory,
    derived_requirement: DerivedRequirementFactory,
) -> None:
    # The two sides name different inputs, so `ShareOfOperation`'s own check
    # passes; but both rest on the same retrieval and, the requirement naming
    # one entity, on the same series -- a share of Kenya's debt in itself.
    total_a = derived_requirement(id="total_a")
    total_b = derived_requirement(id="total_b")
    share = ShareOfRequirement(
        id="share",
        decision_ids=["measure"],
        numerator=RequirementSide(input_id="total_a"),
        denominator=RequirementSide(input_id="total_b"),
    )
    plan = _draft([decision()], [requirement(), total_a, total_b, share])
    share_op = ShareOfOperation(
        id="share",
        requirement_id="share",
        numerator=Side(input="total_a"),
        denominator=Side(input="total_b"),
    )

    result = build(
        plan,
        _translation(
            _retrieve(),
            _aggregate(id="total_a", requirement_id="total_a"),
            _aggregate(id="total_b", requirement_id="total_b"),
            share_op,
            terminal=["share"],
        ),
    )

    assert _codes(result) == [Code.DEGENERATE_SIDES]
    assert "'Kenya' of 'debt'" in result.issues[0].reason


def test_two_sides_of_one_retrieval_naming_different_entities_are_not_degenerate(
    build: Callable[..., CompileResult],
    decision: DecisionFactory,
    requirement: RequirementFactory,
    derived_requirement: DerivedRequirementFactory,
    wide_source: SourceFactory,
) -> None:
    by_entity = (GroupBy.ENTITY, GroupBy.PERIOD)
    comparison = ComparisonRequirement(
        id="gap",
        decision_ids=["measure", "comparison"],
        minuend=RequirementSide(input_id="by_entity_b", entity="Ghana"),
        subtrahend=RequirementSide(input_id="by_entity_a", entity="Kenya"),
    )
    plan = _two_country_plan(
        decision,
        requirement,
        comparison,
        derived_requirement(id="by_entity_a", group_by=by_entity),
        derived_requirement(id="by_entity_b", group_by=by_entity),
    )
    gap = CompareOperation(
        id="gap",
        requirement_id="gap",
        left=Side(input="by_entity_a", entity="Kenya"),
        right=Side(input="by_entity_b", entity="Ghana"),
    )

    result = build(
        plan,
        _translation(
            _retrieve(),
            _aggregate(id="by_entity_a", requirement_id="by_entity_a", group_by=by_entity),
            _aggregate(id="by_entity_b", requirement_id="by_entity_b", group_by=by_entity),
            gap,
            terminal=["gap"],
        ),
        discovered={"annual_debt_service": wide_source()},
    )

    assert result.ok, result.issues


_DEBTORS = (Entity(name="Kenya", role="debtor"), Entity(name="Ghana", role="debtor"))
_FLOWS = (
    Entity(name="France", role="donor"),
    Entity(name="Germany", role="donor"),
    Entity(name="Kenya", role="recipient"),
    Entity(name="Uganda", role="recipient"),
)
_FLOW_SOURCE = (
    ResolvedEntity(role="origin", ref="country/FRA", label="France"),
    ResolvedEntity(role="origin", ref="country/DEU", label="Germany"),
    ResolvedEntity(role="destination", ref="country/KEN", label="Kenya"),
    ResolvedEntity(role="destination", ref="country/UGA", label="Uganda"),
)


def _keeps(*names: str, role: str) -> EntityIn:
    return EntityIn(names=names, role=role)


@pytest.mark.parametrize(
    ("entities", "left", "right", "degenerate"),
    [
        pytest.param(
            _DEBTORS,
            ((_keeps("Kenya", role="debtor"),), None),
            ((_keeps("Ghana", role="debtor"),), None),
            False,
            id="kenya-against-ghana",
        ),
        pytest.param(
            _DEBTORS,
            ((_keeps("Kenya", role="debtor"),), None),
            ((_keeps("Kenya", role="debtor"),), None),
            True,
            id="kenya-against-kenya",
        ),
        pytest.param(
            _DEBTORS,
            ((_keeps("Kenya", "Ghana", role="debtor"),), None),
            ((_keeps("Ghana", "Kenya", role="debtor"),), None),
            True,
            id="both-in-either-order",
        ),
        pytest.param(
            _FLOWS,
            ((_keeps("France", role="donor"), _keeps("Kenya", role="recipient")), None),
            ((_keeps("Germany", role="donor"), _keeps("Uganda", role="recipient")), None),
            False,
            id="france-to-kenya-against-germany-to-uganda",
        ),
        pytest.param(
            _FLOWS,
            ((_keeps("Kenya", role="recipient"),), "France"),
            ((_keeps("Uganda", role="recipient"),), "France"),
            False,
            id="france-to-kenya-against-france-to-uganda",
        ),
        pytest.param(
            _FLOWS[::2],
            ((_keeps("Kenya", role="recipient"),), None),
            ((_keeps("France", role="donor"),), None),
            True,
            id="each-filter-keeps-the-one-flow",
        ),
    ],
)
def test_two_filters_of_one_retrieval_are_degenerate_only_when_they_keep_the_same_entities(
    build: Callable[..., CompileResult],
    decision: DecisionFactory,
    requirement: RequirementFactory,
    source: SourceFactory,
    entities: tuple[Entity, ...],
    left: tuple[tuple[EntityIn, ...], str | None],
    right: tuple[tuple[EntityIn, ...], str | None],
    degenerate: bool,
) -> None:
    # Each side reads a filter of the one retrieval, and may name an entity
    # of its own; the side identity reads the names kept per role, so Kenya
    # against Ghana, or France to Kenya against Germany to Uganda, is two
    # series.
    filters = [
        FilterRequirement(
            id=filter_id,
            decision_ids=["measure"],
            input_id="annual_debt_service",
            predicates=predicates,
        )
        for filter_id, (predicates, _) in zip(("left", "right"), (left, right), strict=True)
    ]
    comparison = ComparisonRequirement(
        id="gap",
        decision_ids=["measure"],
        minuend=RequirementSide(input_id="right", entity=right[1]),
        subtrahend=RequirementSide(input_id="left", entity=left[1]),
    )
    gap = CompareOperation(
        id="gap",
        requirement_id="gap",
        left=Side(input="left", entity=left[1]),
        right=Side(input="right", entity=right[1]),
    )
    filter_ops = [
        FilterOperation(id=one.id, requirement_id=one.id, input="debt", predicates=one.predicates)
        for one in filters
    ]
    resolved = (
        _FLOW_SOURCE
        if entities[0].role == "donor"
        else tuple(
            ResolvedEntity(role="debtor", ref=f"country/{one.name[:3].upper()}", label=one.name)
            for one in entities
        )
    )

    result = build(
        _draft([decision()], [requirement(entities=list(entities)), *filters, comparison]),
        _translation(_retrieve(), *filter_ops, gap, terminal=["gap"]),
        discovered={"annual_debt_service": source(entities=resolved)},
    )

    assert (Code.DEGENERATE_SIDES in _codes(result)) is degenerate, result.issues


def test_comparing_requirements_whose_periods_share_no_year_is_reported(
    build: Callable[..., CompileResult],
    decision: DecisionFactory,
    requirement: RequirementFactory,
    source: SourceFactory,
) -> None:
    old = requirement(id="old_gdp", measure="GDP", period=Period(start=1990, end=2000))
    comparison = ComparisonRequirement(
        id="gap",
        decision_ids=["measure"],
        minuend=RequirementSide(input_id="old_gdp"),
        subtrahend=RequirementSide(input_id="annual_debt_service"),
    )
    # The plan refuses sides that share no year (`planner.shapes.rejects`), so
    # the disjoint period is set unchecked, as on a plan the compiler is handed.
    overlapping = old.model_copy(update={"period": Period(start=1990, end=2015)})
    plan = _draft([decision()], [requirement(), overlapping, comparison]).model_copy(
        update={"evidence_requirements": (requirement(), old, comparison)}
    )
    gap = CompareOperation(
        id="gap", requirement_id="gap", left=Side(input="debt"), right=Side(input="gdp")
    )
    retrieve_gdp = _retrieve(id="gdp", requirement_id="old_gdp", spec_id=SPEC_OTHER)

    result = build(
        plan,
        _translation(_retrieve(), retrieve_gdp, gap, terminal=["gap"]),
        discovered={
            "annual_debt_service": source(),
            "old_gdp": source(spec_id=SPEC_OTHER, coverage_window=Period(start=1990, end=2005)),
        },
    )

    assert _codes(result) == [Code.INPUT_SHAPE_UNSUPPORTED, Code.PERIODS_DISJOINT]


@pytest.mark.parametrize(
    "open_entity_axes", [(), (OpenEntityAxis(column="observationAbout", member_count=193),)]
)
def test_grouping_by_entity_needs_a_source_that_resolves_entities(
    build: Callable[..., CompileResult],
    decision: DecisionFactory,
    requirements: tuple[EvidenceRequirement, ...],
    source: SourceFactory,
    open_entity_axes: tuple[OpenEntityAxis, ...],
) -> None:
    # A plan naming Kenya keeps the reason when the source bound nothing,
    # whether or not it defaulted a scope: only `across` reads an open axis.
    dataset, aggregate_req = requirements
    plan = _draft(
        [decision()], [dataset, aggregate_req.model_copy(update={"group_by": (GroupBy.ENTITY,)})]
    )

    result = build(
        plan,
        _translation(_retrieve(), _aggregate(group_by=(GroupBy.ENTITY,))),
        discovered={"annual_debt_service": source(entities=(), open_entity_axes=open_entity_axes)},
    )

    assert Code.ENTITY_UNRESOLVED in _codes(result)


@pytest.mark.parametrize(
    "open_entity_axes", [(), (OpenEntityAxis(column="observationAbout", member_count=193),)]
)
def test_rank_needs_a_source_that_resolves_entities(
    build: Callable[..., CompileResult],
    decision: DecisionFactory,
    requirement: RequirementFactory,
    source: SourceFactory,
    open_entity_axes: tuple[OpenEntityAxis, ...],
) -> None:
    # A plan naming Kenya keeps the reason when the source bound nothing,
    # whether or not it defaulted a scope: only `across` reads an open axis.
    rank_req = RankRequirement(
        id="ranked",
        decision_ids=["measure"],
        input_id="annual_debt_service",
        direction=RankDirection.HIGHEST_FIRST,
        among=RankAmong.POPULATION,
        reduction=Reduction.SUM,
    )
    plan = _draft(
        [decision(fixes=DecisionFixes(among=RankAmong.POPULATION))], [requirement(), rank_req]
    )
    ranked = RankOperation(
        id="ranked",
        requirement_id="ranked",
        input="debt",
        direction=RankDirection.HIGHEST_FIRST,
        among=RankAmong.POPULATION,
        reduction=Reduction.SUM,
    )

    result = build(
        plan,
        _translation(_retrieve(), ranked, terminal=["ranked"]),
        discovered={"annual_debt_service": source(entities=(), open_entity_axes=open_entity_axes)},
    )

    assert Code.ENTITY_UNRESOLVED in _codes(result)


def test_share_of_gets_the_same_side_checks_as_compare(
    build: Callable[..., CompileResult],
    decision: DecisionFactory,
    requirement: RequirementFactory,
    wide_source: SourceFactory,
) -> None:
    # The requirement's own side has to name an entity its dataset declares
    # -- `Plan` itself refuses otherwise -- so the operation is what
    # deliberately names one the dataset lacks, which also disagrees with
    # the requirement it carries out (`operation_mismatch`, on `denominator`).
    share = ShareOfRequirement(
        id="share",
        decision_ids=["measure", "comparison"],
        numerator=RequirementSide(input_id="annual_debt_service", entity="Kenya"),
        denominator=RequirementSide(input_id="annual_debt_service", entity="Ghana"),
    )
    plan = _two_country_plan(decision, requirement, share)
    share_op = ShareOfOperation(
        id="share",
        requirement_id="share",
        numerator=Side(input="debt", entity="Kenya"),
        denominator=Side(input="debt", entity="Uganda"),
    )

    result = build(
        plan,
        _translation(_retrieve(), share_op, terminal=["share"]),
        discovered={"annual_debt_service": wide_source()},
    )

    assert Code.UNKNOWN_ENTITY in _codes(result)


_USD = Unit(currency="USD", prices="current")


def _summed(
    requirement: RequirementFactory, *, right_unit: Unit | None = _USD
) -> tuple[list[EvidenceRequirement], Translation]:
    """Bilateral debt service plus a second source's, added as `total`, read by a growth."""
    multilateral = requirement(
        id="multilateral", measure="imputed multilateral debt service", unit=right_unit
    )
    total = SumOfRequirement(
        id="total",
        decision_ids=["measure"],
        left=RequirementSide(input_id="annual_debt_service"),
        right=RequirementSide(input_id="multilateral"),
    )
    growth = GrowthRequirement(
        id="growth", decision_ids=["measure"], input_id="total", window_start=2016, window_end=2020
    )
    translation = _translation(
        _retrieve(),
        _retrieve(id="multi", requirement_id="multilateral", spec_id=SPEC_OTHER),
        SumOfOperation(
            id="total", requirement_id="total", left=Side(input="debt"), right=Side(input="multi")
        ),
        GrowthOperation(
            id="growth", requirement_id="growth", input="total", window_start=2016, window_end=2020
        ),
        terminal=["growth"],
    )
    return [requirement(), multilateral, total, growth], translation


# The identity facets of the ONE imputed multilateral definition `search` binds
# when the question states its unit: the unit is a request, so the fetched frame
# is converted and declares the request object, while the source's own unit ref
# stays `CurrencyUSD`.
_REQUEST_BOUND_UNIT = {
    "unit": {"state": "value", "request": {"currency": "USD", "prices": "current"}},
    "provenance": {"ref": "provenance/ONEImputedMultilateralSectors", "state": "value"},
}


_BOUND_BY_REQUEST = {"unit_ref": "CurrencyUSD", "identity": {"facets": _REQUEST_BOUND_UNIT}}


@pytest.mark.parametrize(
    ("left_source", "right_source", "pins"),
    [
        pytest.param({}, {}, {"left_unit": "usd", "right_unit": "usd"}, id="unit-ref"),
        pytest.param(
            _BOUND_BY_REQUEST,
            _BOUND_BY_REQUEST,
            {
                "left_unit": {"currency": "USD", "prices": "current"},
                "right_unit": {"currency": "USD", "prices": "current"},
            },
            id="unit-request",
        ),
        pytest.param(
            {},
            {
                "unit_ref": "CurrencyUSD",
                "identity": {
                    "facets": {
                        "unit": {
                            "state": "value",
                            "request": {"currency": {"role": "origin"}, "prices": "current"},
                        }
                    }
                },
            },
            {"left_unit": "usd"},
            id="unit-request-by-role",
        ),
    ],
)
def test_a_sum_binds_both_frames_pins_each_sides_unit_and_feeds_a_growth(
    build: Callable[..., CompileResult],
    decision: DecisionFactory,
    requirement: RequirementFactory,
    source: SourceFactory,
    left_source: dict[str, Any],
    right_source: dict[str, Any],
    pins: dict[str, object],
) -> None:
    requirements, translation = _summed(requirement)

    result = build(
        _draft([decision()], requirements),
        translation,
        discovered={
            "annual_debt_service": source(**_pinned("DebtServicePaid", left_source)),
            "multilateral": source(
                spec_id=SPEC_OTHER,
                **_pinned("MultilateralDebtService", right_source),
            ),
        },
    )

    assert result.ok, result.issues
    assert result.workflow is not None
    steps = {step.id: step for step in result.workflow.steps}
    assert steps["total"].tool.name == "sum_of"
    assert steps["total"].args == pins
    assert steps["total"].bindings == {
        "frame_ref": Binding(from_step="debt", output=ArtifactKind.FRAME_REF),
        "right_frame_ref": Binding(from_step="multi", output=ArtifactKind.FRAME_REF),
    }
    assert steps["total"].expected_role_keys == ()
    assert steps["growth"].bindings == {
        "frame_ref": Binding(from_step="total", output=ArtifactKind.FRAME_REF)
    }


_CHINA = ResolvedEntity(role="origin", ref="country/CHN", label="China")
_KENYA = ResolvedEntity(role="destination", ref="country/KEN", label="Kenya")
_GHANA = ResolvedEntity(role="destination", ref="country/GHA", label="Ghana")


@pytest.mark.parametrize(
    ("entities", "resolved", "pools"),
    [
        pytest.param(
            [Entity(name="China", role="creditor"), Entity(name="Kenya", role="debtor")],
            (_CHINA, _KENYA),
            False,
            id="one-per-side-of-the-flow",
        ),
        pytest.param(
            [Entity(name="Kenya", role="debtor"), Entity(name="Ghana", role="debtor")],
            (_KENYA, _GHANA),
            True,
            id="two-on-one-side",
        ),
    ],
)
def test_a_whole_sum_side_needs_an_entity_only_where_its_requirement_pools_several(
    build: Callable[..., CompileResult],
    decision: DecisionFactory,
    requirement: RequirementFactory,
    source: SourceFactory,
    entities: list[Entity],
    resolved: tuple[ResolvedEntity, ...],
    pools: bool,
) -> None:
    # A creditor and its debtor are one series, which a whole side reads as
    # it is; two debtors are two a whole side would add into one.
    bilateral = requirement(entities=entities)
    multilateral = requirement(
        id="multilateral", measure="imputed multilateral debt service", entities=entities
    )
    total = SumOfRequirement(
        id="total",
        decision_ids=["measure"],
        left=RequirementSide(input_id="annual_debt_service"),
        right=RequirementSide(input_id="multilateral"),
    )
    translation = _translation(
        _retrieve(),
        _retrieve(id="multi", requirement_id="multilateral", spec_id=SPEC_OTHER),
        SumOfOperation(
            id="total", requirement_id="total", left=Side(input="debt"), right=Side(input="multi")
        ),
        terminal=["total"],
    )

    result = build(
        _draft([decision()], [bilateral, multilateral, total]),
        translation,
        discovered={
            "annual_debt_service": source(entities=resolved, **_pinned("DebtServicePaid")),
            "multilateral": source(
                spec_id=SPEC_OTHER,
                **_pinned("MultilateralDebtService"),
                entities=resolved,
            ),
        },
    )

    assert (Code.ENTITY_REQUIRED in _codes(result)) is pools
    assert result.ok is not pools, result.issues
    # A sum adds yearly series, so a pooling side pools per year only.
    assert _pools(result, one_figure=False) is pools


@pytest.mark.parametrize(
    ("right_source", "right_unit", "code"),
    [
        pytest.param(
            {
                "spec_id": SPEC_KENYA,
                "served_count": 1,
                "shape_member_digest": member_digest(["MultilateralDebtService"]),
            },
            _USD,
            Code.SUM_SIDES_OVERLAP,
            id="the-same-definition",
        ),
        pytest.param(
            {"spec_id": SPEC_OTHER, "served_count": None, "shape_member_digest": "d"},
            _USD,
            Code.SUM_SIDES_OVERLAP,
            id="one-family-under-two-definitions",
        ),
        pytest.param(
            {
                "spec_id": SPEC_OTHER,
                "served_count": 1,
                "shape_member_digest": member_digest(["Other"]),
                "unit_ref": "eur",
            },
            Unit(currency="EUR", prices="current"),
            Code.SUM_UNITS_DIFFER,
            id="another-currency",
        ),
        pytest.param(
            {
                "spec_id": SPEC_OTHER,
                "served_count": 1,
                "shape_member_digest": member_digest(["Other"]),
            },
            None,
            Code.SUM_UNITS_DIFFER,
            id="one-side-states-no-unit",
        ),
    ],
)
def test_a_sum_that_could_count_a_flow_twice_or_mix_units_is_refused(
    build: Callable[..., CompileResult],
    decision: DecisionFactory,
    requirement: RequirementFactory,
    source: SourceFactory,
    right_source: dict[str, Any],
    right_unit: Unit | None,
    code: CompileIssueCode,
) -> None:
    requirements, translation = _summed(requirement, right_unit=right_unit)
    translation = translation.model_copy(
        update={
            "operations": tuple(
                op.model_copy(update={"spec_id": right_source["spec_id"]})
                if op.id == "multi"
                else op
                for op in translation.operations
            )
        }
    )

    result = build(
        _draft([decision()], requirements),
        translation,
        discovered={
            # Two families overlap when their digests are equal.
            "annual_debt_service": source(
                **(
                    {"served_count": None, "shape_member_digest": "d"}
                    if right_source.get("shape_member_digest") == "d"
                    else {}
                )
            ),
            "multilateral": source(**right_source),
        },
    )

    assert code in _codes(result)
    assert {Code.SUM_SIDES_OVERLAP, Code.SUM_UNITS_DIFFER} - {code} & set(_codes(result)) == set()


# The StatVar shape of a nominal series, as a definition's identity states it.
_NOMINAL = {"shape": {"kind": "statvar_schema", "measurement_qualifier": "Nominal"}}


# The unit WHO's domestic health spending is served in: constant US dollars
# whose base year neither the id nor the label states.
_CONSTANT_UNSTATED_YEAR = {
    "unit_ref": "undata/UNIT_MEASURE-CR_USD_K",
    "unit_label": "United States dollars at constant prices",
}
# A requirement stating no unit, as a side read in its source's own unit is,
# with or without its source pinned.
_OWN_UNIT = {"unit": None}
_OWN_UNIT_PINNED = {"unit": None, "source_spec_id": SPEC_KENYA, "decision_ids": ["measure", "pin"]}
# A definition bound by request to current euros, whatever its source's own unit.
_REQUEST_BOUND_EUR = {
    "unit": {"state": "value", "request": {"currency": "EUR", "prices": "current"}}
}
# Two units naming no currency the reader recognises, one in each price basis.
_CONSTANT_UNSTATED_YEAR_NO_CURRENCY = {
    "unit_ref": "undata/UNIT_MEASURE-XDC",
    "unit_label": "National currency at constant prices",
}
_CURRENT_NO_CURRENCY = {
    "unit_ref": "undata/UNIT_MEASURE-XDC_CUR",
    "unit_label": "National currency at current prices",
}


@pytest.mark.parametrize(
    ("left_source", "left_requirement", "right_source", "code", "about", "named"),
    [
        pytest.param(
            {"unit_ref": "CurrencyUSD", "identity": _NOMINAL},
            {},
            _BOUND_BY_REQUEST,
            Code.FRAME_UNIT_FORMS_DIFFER,
            "annual_debt_service",
            ["annual_debt_service"],
            id="current-usd-as-a-ref-against-a-request",
        ),
        pytest.param(
            {"unit_ref": "CurrencyUSD", "identity": _NOMINAL},
            {},
            {"unit_ref": "SDG_CU_USD"},
            Code.FRAME_UNIT_FORMS_DIFFER,
            "annual_debt_service",
            ["annual_debt_service", "multilateral"],
            id="current-usd-spelled-two-ways-as-refs-on-both-sides",
        ),
        pytest.param(
            {"unit_ref": "CurrencyEUR", "identity": _NOMINAL},
            {},
            _BOUND_BY_REQUEST,
            Code.FRAME_UNITS_DIFFER,
            None,
            [],
            id="another-currency-as-a-ref",
        ),
        pytest.param(
            {"unit_ref": "CurrencyUSD", "identity": {"shape": {"kind": "statvar_schema"}}},
            {},
            _BOUND_BY_REQUEST,
            Code.FRAME_UNITS_DIFFER,
            None,
            [],
            id="a-ref-stating-no-price-basis",
        ),
        pytest.param(
            {"unit_ref": "CurrencyUSD", "identity": {"shape": {"kind": "statvar_schema"}}},
            _OWN_UNIT_PINNED,
            _BOUND_BY_REQUEST,
            Code.FRAME_UNITS_DIFFER,
            None,
            [],
            id="a-pinned-ref-stating-no-price-basis",
        ),
        pytest.param(
            _CONSTANT_UNSTATED_YEAR,
            _OWN_UNIT_PINNED,
            _BOUND_BY_REQUEST,
            Code.FRAME_UNITS_IRRECONCILABLE,
            "annual_debt_service",
            [],
            id="a-pinned-source-in-constant-prices",
        ),
        pytest.param(
            {"unit_ref": "ConstantUSD_2024"},
            _OWN_UNIT_PINNED,
            _BOUND_BY_REQUEST,
            Code.FRAME_PRICE_BASES_DIFFER,
            None,
            [],
            id="a-pinned-source-in-constant-prices-of-a-stated-base-year",
        ),
        pytest.param(
            _CONSTANT_UNSTATED_YEAR,
            _OWN_UNIT,
            _BOUND_BY_REQUEST,
            Code.FRAME_PRICE_BASES_DIFFER,
            None,
            [],
            id="an-unpinned-source-in-constant-prices",
        ),
        pytest.param(
            _CONSTANT_UNSTATED_YEAR | {"identity": {"facets": _REQUEST_BOUND_EUR}},
            _OWN_UNIT_PINNED,
            _BOUND_BY_REQUEST,
            Code.FRAME_UNITS_DIFFER,
            None,
            [],
            id="a-pinned-source-bound-by-request-to-current-prices",
        ),
        pytest.param(
            _CONSTANT_UNSTATED_YEAR_NO_CURRENCY,
            _OWN_UNIT_PINNED,
            _CURRENT_NO_CURRENCY,
            Code.FRAME_UNITS_IRRECONCILABLE,
            "annual_debt_service",
            [],
            id="a-pinned-source-in-constant-prices-neither-currency-read",
        ),
    ],
)
def test_two_sides_whose_frames_state_their_units_differently_are_refused_before_running(
    build: Callable[..., CompileResult],
    decision: DecisionFactory,
    pinning_decision: Callable[..., Decision],
    requirement: RequirementFactory,
    source: SourceFactory,
    left_source: dict[str, Any],
    left_requirement: dict[str, Any],
    right_source: dict[str, Any],
    code: CompileIssueCode,
    about: str | None,
    named: list[str],
) -> None:
    # analysis-mcp pairs two frames only when their unit cells are equal. A
    # ref and a request denoting one currency and price basis differ in form
    # alone, and the refusal names every side to retrieve by request, so the
    # planner repairs both in one round when both are refs. A requirement
    # may ask for constant prices only at a stated base year, so a pinned
    # source in constant prices of an unstated year never pairs with a side
    # in current prices, and the refusal names the pinned side for the user
    # to decide. A pinned source of a stated year, or an unpinned one, is
    # repaired by stating one unit on both sides.
    requirements, translation = _summed(requirement)
    requirements[0] = requirement(**left_requirement)
    pin = left_requirement.get("source_spec_id")
    pins = [pinning_decision(pin, slot="pin")] if pin else []

    result = build(
        _draft([decision(), *pins], requirements),
        translation,
        discovered={
            "annual_debt_service": source(**_pinned("DebtServicePaid", left_source)),
            "multilateral": source(
                spec_id=SPEC_OTHER,
                **_pinned("MultilateralDebtService", right_source),
            ),
        },
    )

    unit_codes = {
        Code.FRAME_UNIT_FORMS_DIFFER,
        Code.FRAME_UNITS_DIFFER,
        Code.FRAME_PRICE_BASES_DIFFER,
        Code.FRAME_UNITS_IRRECONCILABLE,
    }
    found = [issue for issue in result.issues if issue.code in unit_codes]
    assert [(issue.code, issue.requirement_id) for issue in found] == [(code, about)]
    if code is Code.FRAME_UNIT_FORMS_DIFFER:
        assert [
            r for r in ("annual_debt_service", "multilateral") if repr(r) in found[0].reason
        ] == named


# A definition bound by request to constant 2024 US dollars, as `search` binds
# OECD DAC2a for "constant prices", and a series in current US dollars.
_CONSTANT_2024_BY_REQUEST = {
    "unit_ref": "CurrencyUSD",
    "identity": {
        "facets": {
            "unit": {
                "state": "value",
                "request": {"currency": "USD", "prices": "constant", "base_year": 2024},
            }
        }
    },
}
_CURRENT_USD = {"unit_ref": "CurrencyUSD", "identity": _NOMINAL}


@pytest.mark.parametrize(
    ("left_source", "right_source", "pinned", "code", "about"),
    [
        pytest.param(
            _CONSTANT_2024_BY_REQUEST,
            _CURRENT_USD,
            ("annual_debt_service", "multilateral"),
            Code.FRAME_UNITS_IRRECONCILABLE,
            "annual_debt_service",
            id="both-pinned-the-constant-side-bound-by-request",
        ),
        pytest.param(
            _CURRENT_USD,
            _CONSTANT_2024_BY_REQUEST,
            ("annual_debt_service", "multilateral"),
            Code.FRAME_UNITS_IRRECONCILABLE,
            "multilateral",
            id="both-pinned-the-current-side-first",
        ),
        pytest.param(
            _CONSTANT_2024_BY_REQUEST,
            _CURRENT_USD,
            ("annual_debt_service",),
            Code.FRAME_PRICE_BASES_DIFFER,
            None,
            id="the-other-side-unpinned",
        ),
    ],
)
def test_two_pinned_sides_in_different_prices_go_back_to_the_user(
    build: Callable[..., CompileResult],
    decision: DecisionFactory,
    pinning_decision: Callable[..., Decision],
    requirement: RequirementFactory,
    source: SourceFactory,
    left_source: dict[str, Any],
    right_source: dict[str, Any],
    pinned: tuple[str, ...],
    code: CompileIssueCode,
    about: str | None,
) -> None:
    # A pinned side is served in its source's own unit, a request object
    # included, and no operation restates a series in other prices. With
    # both sides pinned no requirement change pairs them, so the issue names
    # the side in constant prices for the user to decide. With one pinned,
    # stating its unit on the other side repairs the pair.
    requirements, translation = _summed(requirement)
    spec_ids = {"annual_debt_service": SPEC_KENYA, "multilateral": SPEC_OTHER}
    pins: list[Decision] = []
    for index, requirement_id in enumerate(("annual_debt_service", "multilateral")):
        if requirement_id not in pinned:
            continue
        slot = f"pin_{index}"
        pins.append(pinning_decision(spec_ids[requirement_id], slot=slot))
        requirements[index] = DatasetRequirement.model_validate(
            requirements[index].model_dump()
            | {
                "unit": None,
                "source_spec_id": spec_ids[requirement_id],
                "decision_ids": ["measure", slot],
            }
        )

    result = build(
        _draft([decision(), *pins], requirements),
        translation,
        discovered={
            "annual_debt_service": source(**_pinned("DebtServicePaid", left_source)),
            "multilateral": source(
                spec_id=SPEC_OTHER, **_pinned("MultilateralDebtService", right_source)
            ),
        },
    )

    unit_codes = {
        Code.FRAME_UNITS_DIFFER,
        Code.FRAME_PRICE_BASES_DIFFER,
        Code.FRAME_UNITS_IRRECONCILABLE,
    }
    assert [
        (issue.code, issue.requirement_id) for issue in result.issues if issue.code in unit_codes
    ] == [(code, about)]


def _two_sided(
    verb: str, requirement: RequirementFactory
) -> tuple[list[EvidenceRequirement], Translation]:
    """Debt service set against GDP by `verb`, each read by its own retrieval."""
    sides = (
        RequirementSide(input_id="annual_debt_service"),
        RequirementSide(input_id="annual_gdp"),
    )
    debt, gdp = Side(input="debt"), Side(input="gdp")
    derived: EvidenceRequirement
    operation: Operation
    if verb == "share_of":
        derived = ShareOfRequirement(
            id="two", decision_ids=["measure"], numerator=sides[0], denominator=sides[1]
        )
        operation = ShareOfOperation(
            id="two", requirement_id="two", numerator=debt, denominator=gdp
        )
    elif verb == "compare":
        derived = ComparisonRequirement(
            id="two", decision_ids=["measure"], minuend=sides[1], subtrahend=sides[0]
        )
        operation = CompareOperation(id="two", requirement_id="two", left=debt, right=gdp)
    else:
        derived = SumOfRequirement(
            id="two", decision_ids=["measure"], left=sides[0], right=sides[1]
        )
        operation = SumOfOperation(id="two", requirement_id="two", left=debt, right=gdp)
    translation = _translation(
        _retrieve(),
        _retrieve(id="gdp", requirement_id="annual_gdp", spec_id=SPEC_OTHER),
        operation,
        terminal=["two"],
    )
    gdp_requirement = requirement(id="annual_gdp", measure="gross domestic product")
    return [requirement(), gdp_requirement, derived], translation


# A definition whose unit request names its currency by entity role: a
# currency per row, which no unit known ahead covers.
_PER_ROW_UNIT = {
    "unit_ref": "CurrencyUSD",
    "identity": {
        "facets": {
            "unit": {
                "state": "value",
                "request": {"currency": {"role": "origin"}, "prices": "current"},
            }
        }
    },
}


@pytest.mark.parametrize("verb", ["share_of", "compare", "sum_of"])
@pytest.mark.parametrize(
    ("left_source", "right_source", "unbound"),
    [
        pytest.param({}, {"unit_ref": None}, "annual_gdp", id="right-declares-no-unit"),
        pytest.param({"unit_ref": None}, {}, "annual_debt_service", id="left-declares-no-unit"),
        pytest.param({"unit_ref": None}, {"unit_ref": None}, None, id="neither-declares-a-unit"),
        pytest.param({}, {}, None, id="both-in-one-unit"),
        pytest.param({}, _PER_ROW_UNIT, None, id="a-currency-per-row-is-left-to-run-time"),
        pytest.param(
            {},
            {"unit_ref": None, "unit_per_series": True},
            None,
            id="a-unit-per-series-is-left-to-run-time",
        ),
    ],
)
def test_a_side_declaring_no_unit_against_one_in_a_unit_is_refused_before_running(
    build: Callable[..., CompileResult],
    decision: DecisionFactory,
    requirement: RequirementFactory,
    source: SourceFactory,
    verb: str,
    left_source: dict[str, Any],
    right_source: dict[str, Any],
    unbound: str | None,
) -> None:
    # compare, share_of and sum_of each refuse exactly one unit-less side
    # (`cross_frame_unit_unbound`), and pair two unit-less sides. quotient_of
    # divides the pair a share refuses, so the share's issue is its own code,
    # about the operation.
    requirements, translation = _two_sided(verb, requirement)

    result = build(
        _draft([decision()], requirements),
        translation,
        discovered={
            "annual_debt_service": source(**_pinned("DebtServicePaid", left_source)),
            "annual_gdp": source(
                spec_id=SPEC_OTHER,
                **_pinned("Gdp", right_source),
            ),
        },
    )

    if unbound is None:
        assert result.ok, result.issues
        return
    expected = (
        (Code.SHARE_UNITS_DIFFER, "two", None)
        if verb == "share_of"
        else (Code.FRAME_UNIT_UNBOUND, "two", unbound)
    )
    assert [(i.code, i.operation_id, i.requirement_id) for i in result.issues] == [expected]


@pytest.mark.parametrize(
    ("verb", "code"),
    [
        ("share_of", Code.SHARE_UNITS_DIFFER),
        ("compare", Code.FRAME_UNITS_DIFFER),
        ("sum_of", Code.FRAME_UNITS_DIFFER),
    ],
)
def test_two_sides_in_different_units_report_the_code_of_the_verb_that_pairs_them(
    build: Callable[..., CompileResult],
    decision: DecisionFactory,
    requirement: RequirementFactory,
    source: SourceFactory,
    verb: str,
    code: CompileIssueCode,
) -> None:
    requirements, translation = _two_sided(verb, requirement)

    result = build(
        _draft([decision()], requirements),
        translation,
        discovered={
            "annual_debt_service": source(**_pinned("DebtServicePaid", {"unit_ref": "USDollar"})),
            "annual_gdp": source(spec_id=SPEC_OTHER, **_pinned("Gdp", {"unit_ref": "Euro"})),
        },
    )

    assert [(i.code, i.operation_id) for i in result.issues] == [(code, "two")]


def test_a_total_of_two_sources_over_a_span_sums_the_yearly_series_then_the_years(
    build: Callable[..., CompileResult],
    decision: DecisionFactory,
    requirement: RequirementFactory,
    source: SourceFactory,
) -> None:
    # The sum_of adds the two yearly series over the years both report,
    # and an aggregate grouped by entity reduces that one series whole.
    multilateral = requirement(
        id="multilateral",
        measure="imputed multilateral debt service",
        period=Period(start=2016, end=2024),
    )
    total = SumOfRequirement(
        id="total",
        decision_ids=["measure"],
        left=RequirementSide(input_id="annual_debt_service"),
        right=RequirementSide(input_id="multilateral"),
    )
    span = AggregateRequirement(
        id="span",
        decision_ids=["measure"],
        input_id="total",
        reduction=Reduction.SUM,
        group_by=(GroupBy.ENTITY,),
    )
    translation = _translation(
        _retrieve(),
        _retrieve(id="multi", requirement_id="multilateral", spec_id=SPEC_OTHER),
        SumOfOperation(
            id="total", requirement_id="total", left=Side(input="debt"), right=Side(input="multi")
        ),
        _aggregate(id="span", requirement_id="span", input="total", group_by=(GroupBy.ENTITY,)),
        terminal=["span"],
    )

    result = build(
        _draft([decision()], [requirement(), multilateral, total, span]),
        translation,
        discovered={
            "annual_debt_service": source(**_pinned("DebtServicePaid")),
            "multilateral": source(
                spec_id=SPEC_OTHER,
                **_pinned("MultilateralDebtService"),
            ),
        },
    )

    assert result.ok, result.issues
    assert result.workflow is not None
    steps = {step.id: step for step in result.workflow.steps}
    assert steps["span"].args == {"aggregation": "sum", "group_by": []}
    assert steps["span"].bindings == {
        "frame_ref": Binding(from_step="total", output=ArtifactKind.FRAME_REF)
    }


# A described identity of gross ODA disbursements, the `shape` a family's
# StatVars share.
_ODA_SHAPE = {
    "kind": "statvar_schema",
    "population_type": "FinancialFlow",
    "measured_property": "amount",
    "stat_type": "measuredValue",
}
_EDUCATION = ("CRSPurposeCode110", "CRSPurposeCode112", "CRSPurposeCode11220")
_HEALTH = ("CRSPurposeCode120", "CRSPurposeCode121", "CRSPurposeCode12220")


# The properties the education family's identity lists; apis lists the what
# and how slots alone, so a where-classed constraint such as
# `comparisonRegion` is in no identity.
_EDUCATION_SLOTS = ("crsCooperationModality", "crsPurpose", "crsTypeOfFlow")


def _family(
    digest: str,
    *,
    slot_codes: dict[str, tuple[str, ...] | None] | None,
    slots: tuple[str, ...] | None = None,
    **shape: object,
) -> dict:
    """A family `search` states by member digest, its membership not served as one StatVar.

    `slots` is the properties its identity lists, bound or not; `None` lists none.
    """
    identity: dict[str, object] = {"shape": _ODA_SHAPE | shape}
    if slots is not None:
        identity["slots"] = [{"property": prop, "binding": {"state": "unbound"}} for prop in slots]
    return {
        "served_count": None,
        "shape_member_digest": digest,
        "shape_member_count": 40,
        "identity": identity,
        "slot_codes": slot_codes,
    }


@pytest.mark.parametrize(
    ("left_shape", "right_family", "code"),
    [
        pytest.param(
            {},
            {"slot_codes": {"crsTypeOfFlow": ("CRSTypeOfFlow10",)}},
            Code.SUM_SIDES_OVERLAP,
            id="all-sectors-holds-education",
        ),
        pytest.param(
            {},
            {"slot_codes": {"crsTypeOfFlow": ("CRSTypeOfFlow10",), "crsPurpose": _HEALTH}},
            None,
            id="education-and-health-share-no-code",
        ),
        pytest.param(
            {},
            {"slot_codes": {"crsTypeOfFlow": ("CRSTypeOfFlow10",)}, "measured_property": "count"},
            None,
            id="another-measure",
        ),
        pytest.param(
            {},
            {"slot_codes": {"crsCooperationModality": ("CRSModalityC01",)}},
            Code.SUM_OVERLAP_UNDECIDED,
            id="each-binds-a-property-the-other-leaves-open",
        ),
        pytest.param(
            {},
            {
                "slot_codes": {"crsCooperationModality": ("CRSModalityC01",)},
                "slots": ("crsCooperationModality", "crsPurpose", "crsTypeOfFlow"),
            },
            Code.SUM_OVERLAP_UNDECIDED,
            id="the-other-lists-the-bound-properties-open",
        ),
        pytest.param(
            {"slots": _EDUCATION_SLOTS},
            {
                "slot_codes": {"idsCreditorType": ("IDSCreditorTypeBilateral",)},
                "slots": ("idsCreditorType", "financialFlowType"),
            },
            None,
            id="the-other-lists-no-slot-for-a-bound-property",
        ),
        pytest.param(
            {"slots": _EDUCATION_SLOTS},
            {
                "slot_codes": {
                    "crsTypeOfFlow": ("CRSTypeOfFlow10",),
                    "comparisonRegion": ("africa",),
                },
                "slots": _EDUCATION_SLOTS,
            },
            Code.SUM_OVERLAP_UNDECIDED,
            id="a-bound-where-slot-no-identity-lists",
        ),
        pytest.param({}, {"slot_codes": None}, Code.SUM_OVERLAP_UNDECIDED, id="slots-unread"),
        pytest.param(
            {},
            {"slot_codes": {"crsTypeOfFlow": ("CRSTypeOfFlow10",), "crsPurpose": None}},
            Code.SUM_OVERLAP_UNDECIDED,
            id="a-set-whose-codes-are-not-stated",
        ),
        pytest.param(
            {"measurement_denominator": {"ref": "Count_Person"}},
            {
                "slot_codes": {"crsTypeOfFlow": ("CRSTypeOfFlow10",)},
                "measurement_denominator": {"ref": "Count_Household"},
            },
            None,
            id="another-denominator",
        ),
        pytest.param(
            {"measurement_denominator": {"unresolved": True}},
            {
                "slot_codes": {"crsTypeOfFlow": ("CRSTypeOfFlow10",)},
                "measurement_denominator": {"ref": "Count_Person"},
            },
            Code.SUM_OVERLAP_UNDECIDED,
            id="an-unresolved-denominator-against-a-ref",
        ),
        pytest.param(
            {"measurement_denominator": {"unresolved": True}},
            {
                "slot_codes": {"crsTypeOfFlow": ("CRSTypeOfFlow10",)},
                "measurement_denominator": {"unresolved": True},
            },
            Code.SUM_OVERLAP_UNDECIDED,
            id="two-unresolved-denominators-may-differ",
        ),
        pytest.param(
            {"measurement_denominator": {"ref": "Count_Person"}},
            {
                "slot_codes": {"crsTypeOfFlow": ("CRSTypeOfFlow10",)},
                "measurement_denominator": {"ref": "Count_Person"},
            },
            Code.SUM_SIDES_OVERLAP,
            id="one-denominator-on-both-sides",
        ),
        pytest.param(
            {},
            {
                "slot_codes": {"crsTypeOfFlow": ("CRSTypeOfFlow10",)},
                "measurement_denominator": {"ref": "Count_Person"},
            },
            Code.SUM_OVERLAP_UNDECIDED,
            id="a-field-one-side-leaves-out",
        ),
        pytest.param(
            {"measurement_denominator": {"unresolved": True}},
            {
                "slot_codes": {"crsTypeOfFlow": ("CRSTypeOfFlow10",), "crsPurpose": _HEALTH},
                "measurement_denominator": {"unresolved": True},
            },
            None,
            id="unresolved-denominators-and-codes-shared-by-none",
        ),
    ],
)
def test_two_families_listed_by_digest_alone_are_summed_as_their_identities_allow(
    build: Callable[..., CompileResult],
    decision: DecisionFactory,
    requirement: RequirementFactory,
    source: SourceFactory,
    left_shape: dict[str, Any],
    right_family: dict[str, Any],
    code: CompileIssueCode | None,
) -> None:
    requirements, translation = _summed(requirement)
    education = {"crsTypeOfFlow": ("CRSTypeOfFlow10",), "crsPurpose": _EDUCATION}

    result = build(
        _draft([decision()], requirements),
        translation,
        discovered={
            "annual_debt_service": source(
                **_family("d-education", slot_codes=education, **left_shape)
            ),
            "multilateral": source(spec_id=SPEC_OTHER, **_family("d-other", **right_family)),
        },
    )

    sum_codes = {Code.SUM_SIDES_OVERLAP, Code.SUM_OVERLAP_UNDECIDED} & set(_codes(result))
    assert sum_codes == ({code} if code is not None else set())


_UGANDA = ResolvedEntity(role="destination", ref="country/UGA", label="Uganda")
_EAST_AFRICA = ResolvedEntity(role="destination", ref="EastAfrica", label="East Africa")


@pytest.mark.parametrize(
    ("right_entities", "shared", "code"),
    [
        pytest.param((_UGANDA,), {"shape_member_digest": "d"}, None, id="other-recipient"),
        pytest.param(
            (_UGANDA,),
            {"served_count": 1, "shape_member_digest": member_digest(["DebtServicePaid"])},
            None,
            id="other-recipient-one-served",
        ),
        pytest.param(
            None, {"shape_member_digest": "d"}, Code.SUM_SIDES_OVERLAP, id="same-recipient"
        ),
        pytest.param((), {"shape_member_digest": "d"}, Code.SUM_SIDES_OVERLAP, id="unscoped"),
        pytest.param(
            (_EAST_AFRICA,), {"shape_member_digest": "d"}, Code.SUM_SIDES_OVERLAP, id="a-region"
        ),
    ],
)
def test_one_family_summed_over_entities_it_binds_apart_counts_nothing_twice(
    build: Callable[..., CompileResult],
    decision: DecisionFactory,
    requirement: RequirementFactory,
    source: SourceFactory,
    right_entities: tuple[ResolvedEntity, ...] | None,
    shared: dict[str, Any],
    code: CompileIssueCode | None,
) -> None:
    # ODA to Kenya plus ODA to another recipient: one family, its rows kept
    # apart only by the recipient each side binds.
    requirements, translation = _summed(requirement)
    family = {"served_count": None} | shared
    entities = {} if right_entities is None else {"entities": right_entities}

    result = build(
        _draft([decision()], requirements),
        translation,
        discovered={
            "annual_debt_service": source(**family),
            "multilateral": source(spec_id=SPEC_OTHER, **family, **entities),
        },
    )

    overlap = {Code.SUM_SIDES_OVERLAP, Code.SUM_OVERLAP_UNDECIDED} & set(_codes(result))
    assert overlap == ({code} if code is not None else set())


_SPEC_THIRD = "spec_" + "c" * 32


@pytest.mark.parametrize(
    ("third_unit", "code"),
    [
        pytest.param(_USD, None, id="three-sources-in-one-unit"),
        pytest.param(
            Unit(currency="EUR", prices="current"), Code.SUM_UNITS_DIFFER, id="mixed-units"
        ),
        pytest.param(None, Code.SUM_SIDES_OVERLAP, id="a-source-added-twice"),
    ],
)
def test_a_sum_of_a_sum_is_checked_through_to_every_source(
    build: Callable[..., CompileResult],
    decision: DecisionFactory,
    requirement: RequirementFactory,
    source: SourceFactory,
    third_unit: Unit | None,
    code: CompileIssueCode | None,
) -> None:
    # `sum_of(total, third)`, or with no third source `sum_of(total, debt)`,
    # which adds bilateral debt service into the total a second time.
    requirements, translation = _summed(requirement)
    requirements = [r for r in requirements if r.id != "growth"]
    operations = [op for op in translation.operations if op.id != "growth"]
    discovered = {
        "annual_debt_service": source(**_pinned("DebtServicePaid")),
        "multilateral": source(
            spec_id=SPEC_OTHER,
            **_pinned("MultilateralDebtService"),
        ),
    }
    right = RequirementSide(input_id="annual_debt_service")
    right_op = Side(input="debt")
    if code is not Code.SUM_SIDES_OVERLAP:
        requirements.append(
            requirement(id="private", measure="private debt service", unit=third_unit)
        )
        operations.append(_retrieve(id="private", requirement_id="private", spec_id=_SPEC_THIRD))
        discovered["private"] = source(
            spec_id=_SPEC_THIRD,
            **_pinned("PrivateDebtService"),
            unit_ref="usd" if third_unit == _USD else "eur",
        )
        right = RequirementSide(input_id="private")
        right_op = Side(input="private")
    requirements.append(
        SumOfRequirement(
            id="grand_total",
            decision_ids=["measure"],
            left=RequirementSide(input_id="total"),
            right=right,
        )
    )
    operations.append(
        SumOfOperation(
            id="grand_total", requirement_id="grand_total", left=Side(input="total"), right=right_op
        )
    )

    result = build(
        _draft([decision()], requirements),
        _translation(*operations, terminal=["grand_total"]),
        discovered=discovered,
    )

    sum_codes = {Code.SUM_SIDES_OVERLAP, Code.SUM_UNITS_DIFFER} & set(_codes(result))
    assert sum_codes == ({code} if code is not None else set())
    if code is None:
        assert result.ok, result.issues


def test_join_needs_two_sources_that_share_a_role_column(
    build: Callable[..., CompileResult],
    decision: DecisionFactory,
    requirement: RequirementFactory,
    source: SourceFactory,
) -> None:
    gdp = requirement(id="annual_gdp", measure="gross domestic product")
    joined = JoinRequirement(
        id="joined", decision_ids=["measure"], left_id="annual_debt_service", right_id="annual_gdp"
    )
    plan = _draft([decision()], [requirement(), gdp, joined])
    join_op = JoinOperation(id="joined", requirement_id="joined", left="debt", right="gdp")
    retrieve_gdp = _retrieve(id="gdp", requirement_id="annual_gdp", spec_id=SPEC_OTHER)

    result = build(
        plan,
        _translation(_retrieve(), retrieve_gdp, join_op, terminal=["joined"]),
        discovered={
            "annual_debt_service": source(),
            "annual_gdp": source(
                spec_id=SPEC_OTHER,
                entities=(ResolvedEntity(role="creditor", ref="country/CHN", label="China"),),
            ),
        },
    )

    assert Code.JOIN_NO_COMMON_KEY_COLUMNS in _codes(result)


def test_join_keys_fall_back_to_every_role_column_when_the_requirement_resolves_none(
    build: Callable[..., CompileResult],
    decision: DecisionFactory,
    requirement: RequirementFactory,
    source: SourceFactory,
) -> None:
    # Neither source resolves Uganda, which `source_unusable` already says;
    # the join is not also blamed for keys the requirement failed to narrow.
    uganda = [Entity(name="Uganda")]
    gdp = requirement(id="annual_gdp", measure="gross domestic product", entities=uganda)
    joined = JoinRequirement(
        id="joined", decision_ids=["measure"], left_id="annual_debt_service", right_id="annual_gdp"
    )
    plan = _draft([decision()], [requirement(entities=uganda), gdp, joined])
    join_op = JoinOperation(id="joined", requirement_id="joined", left="debt", right="gdp")
    retrieve_gdp = _retrieve(id="gdp", requirement_id="annual_gdp", spec_id=SPEC_OTHER)

    result = build(
        plan,
        _translation(_retrieve(), retrieve_gdp, join_op, terminal=["joined"]),
        discovered={"annual_debt_service": source(), "annual_gdp": source(spec_id=SPEC_OTHER)},
    )

    assert Code.SOURCE_UNUSABLE in _codes(result)
    assert Code.JOIN_NO_COMMON_KEY_COLUMNS not in _codes(result)


def test_filter_entity_names_must_be_an_entity_of_the_upstream_requirement(
    build: Callable[..., CompileResult],
    decision: DecisionFactory,
    requirement: RequirementFactory,
) -> None:
    # The plan refuses a filter naming what its requirement does not
    # (`planner.plan.unknown_names`), and the compiler reads the same rule for
    # a filter operation that departs from its requirement.
    uganda = FilterRequirement(
        id="filtered",
        decision_ids=["measure"],
        input_id="annual_debt_service",
        predicates=(EntityIn(names=("Uganda",)),),
    )
    assert "keeps ['Uganda']" in _plan_refusal([decision()], [requirement(), uganda])
    kenya = uganda.model_copy(update={"predicates": (EntityIn(names=("Kenya",)),)})
    plan = _draft([decision()], [requirement(), kenya])
    filter_op = FilterOperation(
        id="filtered",
        requirement_id="filtered",
        input="debt",
        predicates=(EntityIn(names=("Uganda",)),),
    )

    result = build(plan, _translation(_retrieve(), filter_op, terminal=["filtered"]))

    assert Code.UNKNOWN_ENTITY in _codes(result)


def test_filter_entity_names_must_resolve_on_the_source(
    build: Callable[..., CompileResult],
    decision: DecisionFactory,
    requirement: RequirementFactory,
    source: SourceFactory,
) -> None:
    filter_req = FilterRequirement(
        id="filtered",
        decision_ids=["measure"],
        input_id="annual_debt_service",
        predicates=(EntityIn(names=("Kenya",)),),
    )
    plan = _draft([decision()], [requirement(), filter_req])
    filter_op = FilterOperation(
        id="filtered",
        requirement_id="filtered",
        input="debt",
        predicates=(EntityIn(names=("Kenya",)),),
    )

    result = build(
        plan,
        _translation(_retrieve(), filter_op, terminal=["filtered"]),
        discovered={"annual_debt_service": source(entities=())},
    )

    assert Code.ENTITY_UNRESOLVED in _codes(result)


# --- Tools -----------------------------------------------------------------------------


def test_a_tool_whose_live_schema_drifted_from_its_row_is_reported(
    build: Callable[..., CompileResult],
    decision: DecisionFactory,
    requirements: tuple[EvidenceRequirement, ...],
) -> None:
    # The production table is pinned to the real servers, so against the
    # fixture surface every workflow tool has drifted: the gate holds.
    plan = _draft([decision()], list(requirements))

    result = build(plan, _translation(_retrieve(), _aggregate()), table=CAPABILITIES)

    assert _codes(result) == [Code.CAPABILITY_SCHEMA_DRIFT, Code.CAPABILITY_SCHEMA_DRIFT]
    assert result.issues[0].tool is not None
    assert "re-pinning" in result.issues[0].reason


def test_a_tool_the_surface_does_not_serve_is_reported(
    build: Callable[..., CompileResult],
    decision: DecisionFactory,
    requirements: tuple[EvidenceRequirement, ...],
    schemas: dict[str, ToolSchema],
) -> None:
    plan = _draft([decision()], list(requirements))

    result = build(
        plan,
        _translation(_retrieve(), _aggregate()),
        surface={"get_data": schemas["get_data"]},
    )

    assert _codes(result) == [Code.UNKNOWN_TOOL]
    assert result.issues[0].operation_id == "totals"


def test_a_tool_whose_schema_cannot_be_hashed_is_one_issue(
    build: Callable[..., CompileResult],
    decision: DecisionFactory,
    requirements: tuple[EvidenceRequirement, ...],
    schemas: dict[str, ToolSchema],
    capabilities: tuple[ToolCapability, ...],
) -> None:
    # `aggregate`'s live schema now embeds an integer RFC 8785 cannot encode.
    # The compiler reports one `BAD_TOOL_SCHEMA` issue for that tool and
    # leaves the retrieval, on an unrelated tool, alone -- it does not abort
    # the whole compile the way an unhandled `UnhashableSchemaError` would.
    unhashable = schemas["aggregate"].model_copy(
        update={"parameters": {**schemas["aggregate"].parameters, "maximum": 2**53}}
    )
    surface = {**schemas, "aggregate": unhashable}
    plan = _draft([decision()], list(requirements))

    result = build(
        plan, _translation(_retrieve(), _aggregate()), surface=surface, table=capabilities
    )

    assert _codes(result) == [Code.BAD_TOOL_SCHEMA]
    assert result.issues[0].tool is not None
    assert result.issues[0].tool.name == "aggregate"
    assert "cannot be hashed" in result.issues[0].reason


def test_arguments_the_compiler_wrote_are_still_checked_against_the_live_schema(
    build: Callable[..., CompileResult],
    decision: DecisionFactory,
    requirements: tuple[EvidenceRequirement, ...],
    schemas: dict[str, ToolSchema],
    capabilities: tuple[ToolCapability, ...],
) -> None:
    # A surface whose `aggregate` no longer takes `group_by`, with the row
    # re-pinned to it: the compiler's own output fails the tool's schema and
    # says so, rather than shipping a call the tool would refuse.
    narrowed = schemas["aggregate"].model_copy(
        update={
            "parameters": {
                **schemas["aggregate"].parameters,
                "properties": {
                    k: v
                    for k, v in schemas["aggregate"].parameters["properties"].items()
                    if k != "group_by"  # type: ignore[union-attr]
                },
            }
        }
    )
    surface = {**schemas, "aggregate": narrowed}
    table = tuple(
        c.model_copy(update={"input_schema_hash": narrowed.input_schema_hash})
        if c.tool.name == "aggregate"
        else c
        for c in capabilities
    )
    plan = _draft([decision()], list(requirements))

    result = build(plan, _translation(_retrieve(), _aggregate()), surface=surface, table=table)

    assert _codes(result) == [Code.BAD_ARGS]
    assert "'group_by' was unexpected" in result.issues[0].reason


def test_a_filter_writing_one_predicate_twice_is_refused_as_analysis_mcp_refuses_it(
    build: Callable[..., CompileResult],
    decision: DecisionFactory,
    source: SourceFactory,
) -> None:
    # France named under its role and under none are two predicates to the
    # plan, and one `category_match` once resolved, which analysis-mcp's
    # filter refuses as a predicate stated twice (`op_spec_invalid`).
    predicates = (EntityIn(names=("France",), role="creditor"), EntityIn(names=("France",)))
    filter_req = FilterRequirement(
        id="france", decision_ids=["measure"], input_id="annual_debt_service", predicates=predicates
    )
    plan = _draft([decision()], [_two_role_requirement(), filter_req])
    filter_op = FilterOperation(
        id="france", requirement_id="france", input="debt", predicates=predicates
    )

    result = build(
        plan,
        _translation(_retrieve(), filter_op, terminal=["france"]),
        discovered={"annual_debt_service": _two_role_creditor_source(source)},
    )

    assert _codes(result) == [Code.BAD_ARGS]
    assert result.issues[0].operation_id == "france"


def test_capability_unpinned_when_a_live_schema_declares_an_output_schema(
    build: Callable[..., CompileResult],
    decision: DecisionFactory,
    requirements: tuple[EvidenceRequirement, ...],
    schemas: dict[str, ToolSchema],
    capabilities: tuple[ToolCapability, ...],
) -> None:
    # The fixture surface's `aggregate` now declares an output schema; the
    # capability row still pins none, which is `capability_unpinned`, not
    # silently trusted.
    declaring = schemas["aggregate"].model_copy(
        update={"output_schema": {"type": "object", "properties": {"frame_ref": {}}}}
    )
    surface = {**schemas, "aggregate": declaring}
    plan = _draft([decision()], list(requirements))

    result = build(
        plan, _translation(_retrieve(), _aggregate()), surface=surface, table=capabilities
    )

    assert Code.CAPABILITY_UNPINNED in _codes(result)


def test_output_schema_hash_is_recorded_on_a_step(
    build: Callable[..., CompileResult],
    decision: DecisionFactory,
    requirements: tuple[EvidenceRequirement, ...],
    schemas: dict[str, ToolSchema],
    capabilities: tuple[ToolCapability, ...],
) -> None:
    output_schema = {"type": "object", "properties": {"frame_ref": {}}}
    declaring = schemas["aggregate"].model_copy(update={"output_schema": output_schema})
    surface = {**schemas, "aggregate": declaring}
    table = tuple(
        c.model_copy(update={"output_schema_hash": declaring.output_schema_hash})
        if c.tool.name == "aggregate"
        else c
        for c in capabilities
    )
    plan = _draft([decision()], list(requirements))

    result = build(plan, _translation(_retrieve(), _aggregate()), surface=surface, table=table)

    assert result.ok, result.issues
    assert result.workflow is not None
    assert result.workflow.steps[1].output_schema_hash == declaring.output_schema_hash
    assert result.workflow.steps[0].output_schema_hash is None


def test_every_issue_is_reported_in_one_pass(
    build: Callable[..., CompileResult],
    decision: DecisionFactory,
    requirement: RequirementFactory,
    derived_requirement: DerivedRequirementFactory,
) -> None:
    gdp = requirement(id="annual_gdp", measure="GDP")
    gdp_totals = derived_requirement(id="gdp_totals", input_id="annual_gdp")
    plan = _draft([decision()], [requirement(), derived_requirement(), gdp, gdp_totals])

    result = build(plan, _translation(_retrieve(spec_id=None), _aggregate(input="ghost")))

    # Five distinct findings from one call: the GDP requirements are not
    # carried out, the aggregate reads nothing (both directly, and as a
    # disagreement with the requirement it names, since "ghost" is not the
    # operation that carries out "annual_debt_service"), the retrieval feeds
    # nothing, and it names no source.
    assert set(_codes(result)) == {
        Code.REQUIREMENT_UNSATISFIED,
        Code.DANGLING_INPUT,
        Code.OPERATION_UNREACHABLE,
        Code.SOURCE_UNSELECTED,
        Code.OPERATION_MISMATCH,
    }


# --- Frozen predictions: `expected_role_keys` ----------------------------------------------


def _frame_role_columns(role_keys: Sequence[str]) -> tuple[str, ...]:
    """The role columns `Frame.build(role_keys=role_keys, rows=())` would carry, in its own order.

    The same technique `analysis_mcp.result.observed_shape` uses on a real
    frame, over an empty one built for its columns alone -- what
    `expected_role_keys` is checked against, rather than a hand-written
    literal that could drift from `grammar.normalize_role_keys`' own rules.
    """
    frame = Frame.build(role_keys=role_keys, rows=())
    return frame.columns[1 : frame.columns.index(TIME_PERIOD_COLUMN)]


def test_expected_role_keys_are_frozen_onto_every_step(
    build: Callable[..., CompileResult],
    decision: DecisionFactory,
    requirement: RequirementFactory,
    derived_requirement: DerivedRequirementFactory,
    source: SourceFactory,
) -> None:
    plan = _draft(
        [decision()],
        [requirement(), derived_requirement(group_by=(GroupBy.ENTITY, GroupBy.PERIOD))],
    )

    result = build(
        plan,
        _translation(_retrieve(), _aggregate(group_by=(GroupBy.ENTITY, GroupBy.PERIOD))),
        discovered={"annual_debt_service": _two_role_source(source)},
    )

    assert result.ok, result.issues
    assert result.workflow is not None
    retrieve, totals = result.workflow.steps
    assert retrieve.expected_role_keys == _frame_role_columns(["recipient"])
    assert totals.expected_role_keys == _frame_role_columns(["recipient"])


def test_a_joins_expected_role_keys_are_the_shared_columns(
    build: Callable[..., CompileResult],
    decision: DecisionFactory,
    requirement: RequirementFactory,
    source: SourceFactory,
) -> None:
    gdp = requirement(id="annual_gdp", measure="gross domestic product")
    joined = JoinRequirement(
        id="joined", decision_ids=["measure"], left_id="annual_debt_service", right_id="annual_gdp"
    )
    plan = _draft([decision()], [requirement(), gdp, joined])
    join_op = JoinOperation(id="joined", requirement_id="joined", left="debt", right="gdp")
    retrieve_gdp = _retrieve(id="gdp", requirement_id="annual_gdp", spec_id=SPEC_OTHER)

    result = build(
        plan,
        _translation(_retrieve(), retrieve_gdp, join_op, terminal=["joined"]),
        discovered={"annual_debt_service": source(), "annual_gdp": source(spec_id=SPEC_OTHER)},
    )

    assert result.ok, result.issues
    assert result.workflow is not None
    step = result.workflow.steps[-1]
    assert step.expected_role_keys == _frame_role_columns(["destination"])


def test_rank_donors_by_their_share_of_oda_to_health_compiles_end_to_end(
    build: Callable[..., CompileResult],
    decision: DecisionFactory,
    source: SourceFactory,
) -> None:
    # A keyed share_of feeds a rank. The two sides' literal columns differ
    # ("donor" against "provider"), but the phrase both requirements state
    # ("donor") is what pairs them. This guards against a positional-pairing
    # bug at the compiler's own column-resolution level.
    health = DatasetRequirement(
        id="req_health",
        decision_ids=["measure"],
        measure="ODA to health",
        entities=[Entity(name="Kenya", role="donor")],
        period=Period(start=2015, end=2024),
    )
    total = DatasetRequirement(
        id="req_total",
        decision_ids=["measure"],
        measure="total ODA",
        entities=[Entity(name="Kenya", role="donor")],
        period=Period(start=2015, end=2024),
    )
    share_req = ShareOfRequirement(
        id="share",
        decision_ids=["measure"],
        numerator=RequirementSide(input_id="req_health"),
        denominator=RequirementSide(input_id="req_total"),
        axis="donor",
        denominator_scope=DenominatorScope.PER_AXIS,
    )
    rank_req = RankRequirement(
        id="ranked",
        decision_ids=["measure", "population"],
        input_id="share",
        direction=RankDirection.HIGHEST_FIRST,
        among=RankAmong.POPULATION,
        axis="donor",
        reduction=Reduction.MEAN,
        window_start=2020,
        window_end=2024,
    )
    population = decision(slot="population", fixes=DecisionFixes(among=RankAmong.POPULATION))
    plan = _draft([decision(), population], [health, total, share_req, rank_req])

    health_op = RetrieveOperation(id="health", requirement_id="req_health", spec_id=SPEC_KENYA)
    total_op = RetrieveOperation(id="total", requirement_id="req_total", spec_id=SPEC_OTHER)
    share_op = ShareOfOperation(
        id="share",
        requirement_id="share",
        numerator=Side(input="health"),
        denominator=Side(input="total"),
        axis="donor",
        denominator_scope=DenominatorScope.PER_AXIS,
    )
    ranked_op = RankOperation(
        id="ranked",
        requirement_id="ranked",
        input="share",
        direction=RankDirection.HIGHEST_FIRST,
        among=RankAmong.POPULATION,
        axis="donor",
        reduction=Reduction.MEAN,
        window_start=2020,
        window_end=2024,
    )
    health_source = source(
        entities=(ResolvedEntity(role="donor", ref="country/KEN", label="Kenya"),)
    )
    total_source = source(
        spec_id=SPEC_OTHER,
        entities=(ResolvedEntity(role="provider", ref="country/KEN", label="Kenya"),),
    )

    result = build(
        plan,
        _translation(health_op, total_op, share_op, ranked_op, terminal=["ranked"]),
        discovered={"req_health": health_source, "req_total": total_source},
    )

    assert result.ok, result.issues
    assert result.workflow is not None
    assert len(result.workflow.steps) == 4
    health_step, total_step, share_step, rank_step = result.workflow.steps
    assert share_step.tool.name == "share_of"
    assert share_step.args == {"numerator_by": ["donor"], "denominator_by": {"donor": "provider"}}
    assert share_step.expected_role_keys == _frame_role_columns(["donor"])
    assert rank_step.tool.name == "rank"
    assert rank_step.args == {
        "rank_by": ["donor"],
        "ascending": False,
        "aggregation": "mean",
        "window_start": "2020",
        "window_end": "2024",
    }
    assert rank_step.expected_role_keys == _frame_role_columns(["donor"])
    assert health_step.expected_role_keys == _frame_role_columns(["donor"])
    assert total_step.expected_role_keys == _frame_role_columns(["provider"])


def test_how_has_kenyas_rank_changed_compiles_with_the_filter_between_rank_and_growth(
    build: Callable[..., CompileResult],
    decision: DecisionFactory,
    source: SourceFactory,
) -> None:
    # `retrieve -> rank(per_period) -> filter(EntityIn=[Kenya]) -> growth(absolute)`:
    # the motivating per-period-rank chain. Without the filter, `rejects`
    # refuses the growth over an un-narrowed `ORDINAL` input; this is the
    # translation that names the fix.
    aid = DatasetRequirement(
        id="annual_debt_service",
        decision_ids=["measure"],
        measure="aid received",
        entities=[Entity(name="Kenya", role="recipient")],
        period=Period(start=2010, end=2024),
    )
    filter_req = FilterRequirement(
        id="kenya_only",
        decision_ids=["measure"],
        input_id="ranked",
        predicates=(EntityIn(names=("Kenya",), role="recipient"),),
    )
    growth_req = GrowthRequirement(
        id="growth_of_rank",
        decision_ids=["measure"],
        input_id="kenya_only",
        window_start=2010,
        window_end=2024,
        change=GrowthChange.ABSOLUTE,
    )
    rank_req = RankRequirement(
        id="ranked",
        decision_ids=["measure", "population"],
        input_id="annual_debt_service",
        direction=RankDirection.HIGHEST_FIRST,
        among=RankAmong.POPULATION,
        axis="recipient",
        per_period=True,
        window_start=2010,
        window_end=2024,
    )
    population = decision(slot="population", fixes=DecisionFixes(among=RankAmong.POPULATION))
    plan = _draft([decision(), population], [aid, rank_req, filter_req, growth_req])

    ranked_op = RankOperation(
        id="ranked",
        requirement_id="ranked",
        input="debt",
        direction=RankDirection.HIGHEST_FIRST,
        among=RankAmong.POPULATION,
        axis="recipient",
        per_period=True,
        window_start=2010,
        window_end=2024,
    )
    filter_op = FilterOperation(
        id="kenya_only",
        requirement_id="kenya_only",
        input="ranked",
        predicates=(EntityIn(names=("Kenya",), role="recipient"),),
    )
    growth_op = GrowthOperation(
        id="growth_of_rank",
        requirement_id="growth_of_rank",
        input="kenya_only",
        window_start=2010,
        window_end=2024,
        change=GrowthChange.ABSOLUTE,
    )

    result = build(
        plan,
        _translation(_retrieve(), ranked_op, filter_op, growth_op, terminal=["growth_of_rank"]),
    )

    assert result.ok, result.issues
    assert result.workflow is not None
    steps_by_id = {step.id: step for step in result.workflow.steps}
    assert steps_by_id["ranked"].args["per_period"] is True
    assert steps_by_id["growth_of_rank"].args["change"] == "absolute"


# --- A population left open: `across` over the source's open axis ----------------------


def _population_decision(decision: DecisionFactory) -> Decision:
    return decision(slot="population", fixes=DecisionFixes(among=RankAmong.POPULATION))


def _every_country(**overrides: Any) -> DatasetRequirement:
    """Health spending per person in every country, the population `search` left open."""
    fields: dict[str, Any] = {
        "id": "health",
        "decision_ids": ["measure"],
        "measure": "health spending per person",
        "across": ["country"],
        "period": Period(start=2015, end=2024),
    }
    return DatasetRequirement(**(fields | overrides))


def _rank_over(
    input_id: str, *, axis: str | None, reads: str = "fetch"
) -> tuple[RankRequirement, RankOperation]:
    """A population rank over `input_id`, as a requirement and the operation carrying it out.

    `reads` is the operation producing `input_id`'s frame.
    """
    fields: dict[str, Any] = {
        "id": "ranked",
        "direction": RankDirection.HIGHEST_FIRST,
        "among": RankAmong.POPULATION,
        "axis": axis,
        "reduction": Reduction.MEAN,
    }
    return (
        RankRequirement(decision_ids=["measure", "population"], input_id=input_id, **fields),
        RankOperation(requirement_id="ranked", input=reads, **fields),
    )


def _countries_of(
    input_id: str,
    *,
    reads: str = "fetch",
    filter_id: str = "countries",
    role: str | None = None,
    strict: bool = False,
) -> tuple[FilterRequirement, FilterOperation]:
    """A filter keeping the countries of `input_id`, a requirement that names no place."""
    predicates = (PlaceLevel(level=PlaceType.COUNTRY, role=role, strict=strict),)
    return (
        FilterRequirement(
            id=filter_id, decision_ids=["measure"], input_id=input_id, predicates=predicates
        ),
        FilterOperation(id=filter_id, requirement_id=filter_id, input=reads, predicates=predicates),
    )


def test_top_countries_by_health_spending_ranks_by_the_open_place_axis(
    build: Callable[..., CompileResult],
    decision: DecisionFactory,
    place_scope_block: dict[str, Any],
) -> None:
    # "Top 10 countries by health spending per person": no country named,
    # and dev `search`'s own block leaves its 193 countries open on
    # `observationAbout`. `search` states the mint serves one StatVar, so
    # the retrieval states no reading. The place axis holds every level of
    # place, so a filter keeps the countries first: its retrieval asks for
    # place types, and the filter reads them.
    open_place = source_from_spec_block(place_scope_block)
    countries_req, countries = _countries_of("health")
    rank_req, ranked = _rank_over("countries", axis="country", reads="countries")
    plan = _draft(
        [decision(), _population_decision(decision)],
        [_every_country(), countries_req, rank_req],
    )
    fetch = RetrieveOperation(id="fetch", requirement_id="health", spec_id=open_place.spec_id)

    result = build(
        plan,
        _translation(fetch, countries, ranked, terminal=["ranked"]),
        discovered={"health": open_place},
    )

    assert result.ok, result.issues
    assert result.workflow is not None
    retrieve, keep, rank = result.workflow.steps
    assert open_place.served_count == 1
    assert retrieve.args == {"spec_id": open_place.spec_id, "place_types": True}
    assert retrieve.expected_role_keys == _frame_role_columns(["observationAbout"])
    assert keep.args == {
        "predicates": [
            {"kind": "place_level", "column": "observationAbout_type", "level": "Country"}
        ]
    }
    assert rank.args == {"rank_by": ["observationAbout"], "ascending": False, "aggregation": "mean"}
    assert rank.expected_role_keys == _frame_role_columns(["observationAbout"])


def test_a_rank_over_the_open_place_axis_without_a_level_filter_is_refused(
    build: Callable[..., CompileResult],
    decision: DecisionFactory,
    place_scope_block: dict[str, Any],
) -> None:
    # The plan cannot know which axis "country" covers, so compile, which
    # does, refuses a rank over every level of place.
    open_place = source_from_spec_block(place_scope_block)
    rank_req, ranked = _rank_over("health", axis="country")
    plan = _draft([decision(), _population_decision(decision)], [_every_country(), rank_req])
    fetch = RetrieveOperation(id="fetch", requirement_id="health", spec_id=open_place.spec_id)

    result = build(
        plan,
        _translation(fetch, ranked, terminal=["ranked"]),
        discovered={"health": open_place},
    )

    assert _codes(result) == [Code.INPUT_SHAPE_UNSUPPORTED]
    assert "holds every place with data at every level" in result.issues[0].reason


# IDS leaves its debtors open on `destination` and no `observationAbout`
# (dev, the total series and its by-creditor twin alike).
_IDS_DEBTORS = OpenEntityAxis(column="destination", member_count=134)


def _ids_debt_total(
    build: Callable[..., CompileResult],
    decision: DecisionFactory,
    source: SourceFactory,
    *,
    level_filter: bool,
    windowed: bool = False,
) -> CompileResult:
    """A total over every IDS debtor, `across=('country',)`, with or without a level filter.

    `windowed` retrieves the copy `submission.service._window_pins` pins in
    place of a search over 2024 alone: its identity over the requirement's
    own years.
    """
    identity: dict[str, Any] = {
        "stat_var": "DebtServicePaid",
        "requested_window": {"kind": "range", "start_year": 2024, "end_year": 2024},
    }
    spec_id, identity = (
        windowed_identity(identity, {"kind": "range", "start_year": 2015, "end_year": 2024})
        if windowed
        else (SPEC_KENYA, identity)
    )
    ids = source(spec_id=spec_id, identity=identity, entities=(), open_entity_axes=(_IDS_DEBTORS,))
    countries_req, countries = _countries_of("health", role="country", strict=True)
    reads = "countries" if level_filter else "fetch"
    total_req = AggregateRequirement(
        id="totals",
        decision_ids=["measure"],
        input_id=reads if level_filter else "health",
        reduction=Reduction.SUM,
        group_by=(GroupBy.PERIOD,),
    )
    total = AggregateOperation(
        id="totals",
        requirement_id="totals",
        input=reads,
        reduction=Reduction.SUM,
        group_by=(GroupBy.PERIOD,),
    )
    requirements = [_every_country(), *([countries_req] if level_filter else []), total_req]
    operations = [
        RetrieveOperation(id="fetch", requirement_id="health", spec_id=ids.spec_id),
        *([countries] if level_filter else []),
        total,
    ]
    return build(
        _draft([decision()], requirements),
        _translation(*operations, terminal=["totals"]),
        discovered={"health": ids},
    )


def test_a_total_over_ids_debtors_by_country_needs_no_level_filter(
    build: Callable[..., CompileResult], decision: DecisionFactory, source: SourceFactory
) -> None:
    # "country" covers the debtors' flow axis, which keeps its role: it holds
    # parties, and the retrieval serves them without the World Bank group totals.
    result = _ids_debt_total(build, decision, source, level_filter=False)

    assert result.ok, result.issues
    assert result.workflow is not None
    retrieve, total = result.workflow.steps
    assert retrieve.args["without_party_totals"] is True
    assert "place_types" not in retrieve.args
    assert total.args["group_by"] == ["time_period"]


@pytest.mark.parametrize("windowed", [False, True], ids=["as-searched", "windowed-pin"])
def test_a_level_filter_over_ids_debtors_reads_the_debtors_type_column(
    build: Callable[..., CompileResult],
    decision: DecisionFactory,
    source: SourceFactory,
    windowed: bool,
) -> None:
    result = _ids_debt_total(build, decision, source, level_filter=True, windowed=windowed)

    assert result.ok, result.issues
    assert result.workflow is not None
    retrieve, countries, _ = result.workflow.steps
    assert (retrieve.args["spec_id"] == SPEC_KENYA) is not windowed
    assert retrieve.args["place_types"] is True
    assert retrieve.args["without_party_totals"] is True
    assert countries.args == {
        "predicates": [
            {
                "kind": "place_level",
                "column": "destination_type",
                "level": "Country",
                "strict": True,
            }
        ]
    }


def test_a_rank_requirement_for_country_filtered_debtors_refuses_a_raw_input(
    build: Callable[..., CompileResult], decision: DecisionFactory, source: SourceFactory
) -> None:
    ids = source(entities=(), open_entity_axes=(_IDS_DEBTORS,))
    countries_req, countries = _countries_of("health", role="country", strict=True)
    rank_req, raw_rank = _rank_over("countries", axis="country", reads="fetch")
    plan = _draft(
        [decision(), _population_decision(decision)],
        [_every_country(), countries_req, rank_req],
    )
    fetch = RetrieveOperation(id="fetch", requirement_id="health", spec_id=ids.spec_id)

    result = build(
        plan,
        _translation(fetch, countries, raw_rank, terminal=["ranked"]),
        discovered={"health": ids},
    )

    assert Code.OPERATION_MISMATCH in _codes(result)
    assert result.workflow is None


def test_a_series_of_every_place_asks_for_no_place_types_and_no_party_totals(
    build: Callable[..., CompileResult],
    decision: DecisionFactory,
    place_scope_block: dict[str, Any],
) -> None:
    # No place-level filter reads the retrieval, so its frame carries no type
    # column, and a place source holds no flow party whose totals to leave out.
    open_place = source_from_spec_block(place_scope_block)
    plan = _draft([decision()], [_every_country()], answer_mode=AnswerMode.SERIES)
    fetch = RetrieveOperation(id="fetch", requirement_id="health", spec_id=open_place.spec_id)

    result = build(plan, _translation(fetch, terminal=["fetch"]), discovered={"health": open_place})

    assert result.ok, result.issues
    assert result.workflow is not None
    assert result.workflow.steps[0].args == {"spec_id": open_place.spec_id}


def _kenya_from_every_donor(party: str = "donor") -> DatasetRequirement:
    """Education aid to Kenya from every member of `party`, 2015 to 2023."""
    return DatasetRequirement(
        id="aid",
        decision_ids=["measure"],
        measure="education aid",
        entities=[Entity(name="Kenya", role="recipient")],
        across=(party,),
        period=Period(start=2015, end=2023),
    )


# The constraint properties the recorded education-aid family varies along,
# each a slot the recorded block states summable.
_DONOR_FAMILY_AXES = ("crsCooperationModality", "crsPurpose", "crsTypeOfFinance")


def _donor_family_stating_no_summable(role_open_block: dict[str, Any]) -> SelectedSource:
    """The recorded education-aid block as an apis service that predates `Slot.summable` serves it.

    Every slot loses its `summable`, which `compiler.sources` reads as false,
    so no axis the family varies along may be summed. Nothing else changes.
    """
    block = copy.deepcopy(role_open_block)
    for slot in block["resolved"]["slots"]:
        slot.pop("summable", None)
    return source_from_spec_block(block)


def _rank_donors(
    build: Callable[..., CompileResult],
    decision: DecisionFactory,
    source: SelectedSource,
    party: str = "donor",
    *,
    stat_vars: StatVarReading | None = None,
) -> CompileResult:
    rank_req, ranked = _rank_over("aid", axis=party)
    aid = _kenya_from_every_donor(party)
    decisions = [decision(), _population_decision(decision)]
    if stat_vars is not None:
        decisions.append(decision(slot="reading", fixes=DecisionFixes(stat_vars=stat_vars)))
        aid = aid.model_copy(
            update={"stat_vars": stat_vars, "decision_ids": ("measure", "reading")}
        )
    plan = _draft(decisions, [aid, rank_req])
    return build(
        plan,
        _translation(
            RetrieveOperation(id="fetch", requirement_id="aid", spec_id=source.spec_id),
            ranked,
            terminal=["ranked"],
        ),
        discovered={"aid": source},
    )


@pytest.mark.parametrize("party", ["donor", "donors", "Donors"])
def test_top_donors_to_kenya_ranks_by_the_open_origin_axis(
    build: Callable[..., CompileResult],
    decision: DecisionFactory,
    role_open_block: dict[str, Any],
    party: str,
) -> None:
    # "Top donors for education aid to Kenya": dev `search`'s own block
    # binds Kenya under `destination` and leaves its 55 donors open on
    # `origin`. Its family of StatVars is summable over every axis it varies
    # along, so the retrieval asks for the sum, which keeps both role
    # columns, and the rank reads one measure per donor. The rank keeps the
    # donors and reduces Kenya and the years away. A plural party places as
    # its singular does.
    kenya_from_every_donor = source_from_spec_block(role_open_block)

    result = _rank_donors(build, decision, kenya_from_every_donor, party)

    assert result.ok, result.issues
    assert result.workflow is not None
    retrieve, rank = result.workflow.steps
    assert retrieve.args == {
        "spec_id": role_open_block["spec_id"],
        "stat_vars": "sum",
        "without_party_totals": True,
    }
    assert result.workflow.sources[0].stat_var_reading_set_by is ReadingSetBy.RULE
    assert retrieve.expected_role_keys == _frame_role_columns(["destination", "origin"])
    assert rank.args == {"rank_by": ["origin"], "ascending": False, "aggregation": "mean"}
    assert rank.expected_role_keys == _frame_role_columns(["origin"])
    assert "entity_role_unverified" not in [w.code.value for w in result.warnings]


def _nigeria_ppg_service_to_every_creditor(role_open_block: dict[str, Any]) -> dict[str, Any]:
    """The block `search` answers Nigeria's PPG debt service in 2024 with, every creditor open.

    Asked for an open creditor, it elects the by-creditor twin of the IDS
    total, binds Nigeria on `destination`, and counts 17 creditors on
    `origin`. Built on the recorded education-aid block, its family replaced
    by the one StatVar.
    """
    block = copy.deepcopy(role_open_block)
    identity = block["definition"]["identity"]
    identity["shape"]["population_type"] = "ServicingPayment"
    identity["slots"] = []
    identity["entities"] = {"destination": ["country/NGA"]}
    identity["open_roles"] = ["origin"]
    identity["requested_window"] = {"kind": "range", "start_year": 2024, "end_year": 2024}
    resolved = block["resolved"]
    resolved["shape"].pop("member_digest")
    resolved["shape"]["member_count"] = 1
    resolved["shape"]["slot_keys"] = []
    resolved["slots"] = []
    resolved["inapplicable_slots"] = []
    resolved["stat_vars"] = {
        "member_count": 1,
        "member_count_exact": True,
        "served_count": 1,
        "data_date_range": {"start": "1970", "end": "2024"},
    }
    resolved["entities"] = [
        {"ref": "country/NGA", "label": "Nigeria", "entity_type": "Country", "role": "destination"}
    ]
    resolved["open_axis_facts"] = []
    resolved["role_axis_facts"] = [
        {"role": "origin", "member_count": 17, "members_available": True}
    ]
    return block


def test_top_creditors_nigeria_pays_rank_over_the_open_creditor_axis(
    build: Callable[..., CompileResult],
    decision: DecisionFactory,
    role_open_block: dict[str, Any],
) -> None:
    # "Who does Nigeria pay the most in debt service for PPG debt in 2024
    # (top 5)" is answered by the by-creditor source search elects for an open
    # creditor, which ranks Nigeria's 17 creditors and keeps five.
    source = source_from_spec_block(_nigeria_ppg_service_to_every_creditor(role_open_block))
    debt = DatasetRequirement(
        id="debt",
        decision_ids=["measure"],
        measure="debt service on PPG external debt",
        entities=[Entity(name="Nigeria", role="debtor")],
        across=("creditor",),
        period=Period(start=2024, end=2024),
    )
    fields: dict[str, Any] = {
        "id": "ranked",
        "direction": RankDirection.HIGHEST_FIRST,
        "among": RankAmong.POPULATION,
        "axis": "creditor",
        "reduction": Reduction.SUM,
        "limit": 5,
    }
    plan = _draft(
        [decision(), _population_decision(decision)],
        [debt, RankRequirement(decision_ids=["measure", "population"], input_id="debt", **fields)],
    )
    fetch = RetrieveOperation(id="fetch", requirement_id="debt", spec_id=source.spec_id)
    ranked = RankOperation(requirement_id="ranked", input="fetch", **fields)

    result = build(
        plan, _translation(fetch, ranked, terminal=["ranked"]), discovered={"debt": source}
    )

    assert result.ok, result.issues
    assert result.workflow is not None
    retrieve, rank = result.workflow.steps
    assert retrieve.expected_role_keys == _frame_role_columns(["destination", "origin"])
    assert rank.args == {
        "rank_by": ["origin"],
        "ascending": False,
        "aggregation": "sum",
        "limit": 5,
    }


@pytest.mark.parametrize("party", ["bilateral donor", "DAC donor", "multilateral donor"])
def test_top_donors_of_one_type_are_refused_over_the_open_origin_axis(
    build: Callable[..., CompileResult],
    decision: DecisionFactory,
    role_open_block: dict[str, Any],
    party: str,
) -> None:
    # "Top bilateral donors of education aid to Kenya": the recorded block's
    # open `origin` holds all 55 donors, multilateral agencies among them,
    # so a rank over it would rank every donor under the narrower name.
    result = _rank_donors(build, decision, source_from_spec_block(role_open_block), party)

    assert _codes(result) == [Code.SOURCE_UNUSABLE]
    assert f"{party!r} is not a party a source's open axis holds" in result.issues[0].reason
    assert "needs its members named in `entities`" in result.issues[0].reason
    assert result.workflow is None


@pytest.mark.parametrize(
    ("party", "word"),
    [("donor countries", "donor"), ("Donor countries", "donor"), ("source country", "source")],
)
def test_top_donor_countries_are_refused_over_the_open_origin_axis(
    build: Callable[..., CompileResult],
    decision: DecisionFactory,
    role_open_block: dict[str, Any],
    party: str,
    word: str,
) -> None:
    # "Top donor countries for education aid to Kenya": the place noun
    # narrows the donors to countries, and the recorded block's open `origin`
    # holds all 55 donors, agencies among them. The refusal names the word
    # that covers them all.
    result = _rank_donors(build, decision, source_from_spec_block(role_open_block), party)

    assert _codes(result) == [Code.SOURCE_UNUSABLE]
    (issue,) = result.issues
    assert f"{party!r} narrows a party to its countries" in issue.reason
    assert f"holds every {word}, agencies included" in issue.reason
    assert f"state `across` as '{word}s'" in issue.reason
    assert (
        "direct `{kind:place_level, role:<party>, level:Country, strict:true}` filter bound to "
        "that role and redirect the calculation through that filter" in issue.reason
    )
    assert result.workflow is None


@pytest.mark.parametrize(
    "population", ["African country", "G7 country", "low-income country", "sector", "year"]
)
def test_a_narrower_or_other_population_is_refused_over_the_open_place_axis(
    build: Callable[..., CompileResult],
    decision: DecisionFactory,
    place_scope_block: dict[str, Any],
    population: str,
) -> None:
    # "Top 5 African countries by health spending per person": the recorded
    # block's open `observationAbout` holds all 193 countries, so the rank
    # would answer with the world's top 5.
    open_place = source_from_spec_block(place_scope_block)
    rank_req, ranked = _rank_over("health", axis=population)
    plan = _draft(
        [decision(), _population_decision(decision)],
        [_every_country(across=(population,)), rank_req],
    )

    result = build(
        plan,
        _translation(
            RetrieveOperation(id="fetch", requirement_id="health", spec_id=open_place.spec_id),
            ranked,
            terminal=["ranked"],
        ),
        discovered={"health": open_place},
    )

    assert _codes(result) == [Code.SOURCE_UNUSABLE]
    assert f"{population!r} is not a party a source's open axis holds" in result.issues[0].reason
    assert result.workflow is None


def test_a_rank_over_a_donor_family_read_separately_is_refused(
    build: Callable[..., CompileResult],
    decision: DecisionFactory,
    role_open_block: dict[str, Any],
) -> None:
    # The recorded block lists two of the 89 StatVars Search confirmed for its
    # family. Served with no `summable` on its slots, no axis may be summed,
    # so the rule reads them separately and the frame carries one measure per
    # StatVar, counted from the confirmed members rather than the listing. A
    # rank over the open `origin` axis reads one.
    assert role_open_block["resolved"]["stat_vars"]["member_count"] == 89
    result = _rank_donors(build, decision, _donor_family_stating_no_summable(role_open_block))

    (issue,) = result.issues
    assert issue.code is Code.INPUT_SHAPE_UNSUPPORTED
    assert issue.operation_id == "ranked"
    assert "rank reads a single measure; got 89" in issue.reason
    assert "cannot be summed over " + ", ".join(_DONOR_FAMILY_AXES) in issue.reason
    assert result.workflow is None


def test_a_rank_over_a_donor_family_the_plan_reads_summed_asks_for_the_sum(
    build: Callable[..., CompileResult],
    decision: DecisionFactory,
    role_open_block: dict[str, Any],
) -> None:
    # The plan states `stat_vars='sum'` over the same recorded block, so the
    # retrieval asks apis for the sum and the rank reads one measure over
    # the open `origin` axis.
    result = _rank_donors(
        build,
        decision,
        source_from_spec_block(role_open_block),
        stat_vars=StatVarReading.SUM,
    )

    assert result.ok, result.issues
    assert result.workflow is not None
    retrieve, rank = result.workflow.steps
    assert retrieve.args == {
        "spec_id": role_open_block["spec_id"],
        "stat_vars": "sum",
        "without_party_totals": True,
    }
    assert result.workflow.sources[0].stat_var_reading_set_by is ReadingSetBy.PLAN
    assert rank.args == {"rank_by": ["origin"], "ascending": False, "aggregation": "mean"}
    assert rank.expected_role_keys == _frame_role_columns(["origin"])


@pytest.mark.parametrize("reader", ["nothing", "filter", "growth", "join", "aggregate-by-party"])
def test_a_flow_retrieval_leaves_out_party_totals_whatever_reads_it(
    build: Callable[..., CompileResult],
    decision: DecisionFactory,
    requirement: RequirementFactory,
    source: SourceFactory,
    reader: str,
) -> None:
    # The frame of a flow retrieval holds each open side's parties without their
    # group totals for every verb that reads it, and for none at all.
    debt = requirement()
    discovered = {"annual_debt_service": source()}
    operations: list[Operation] = [_retrieve()]
    requirements: list[EvidenceRequirement] = [debt]
    answer_mode = AnswerMode.COMPUTED
    match reader:
        case "nothing":
            answer_mode, terminal = AnswerMode.SERIES, "debt"
        case "filter":
            predicates = (ValueThreshold(op="gt", value=0),)
            requirements.append(
                FilterRequirement(
                    id="read",
                    decision_ids=["measure"],
                    input_id="annual_debt_service",
                    predicates=predicates,
                )
            )
            operations.append(
                FilterOperation(
                    id="read", requirement_id="read", input="debt", predicates=predicates
                )
            )
            terminal = "read"
        case "growth":
            window = {"window_start": 2015, "window_end": 2023}
            requirements.append(
                GrowthRequirement(
                    id="read", decision_ids=["measure"], input_id="annual_debt_service", **window
                )
            )
            operations.append(
                GrowthOperation(id="read", requirement_id="read", input="debt", **window)
            )
            terminal = "read"
        case "join":
            requirements += [
                requirement(id="annual_gdp", measure="gross domestic product"),
                JoinRequirement(
                    id="read",
                    decision_ids=["measure"],
                    left_id="annual_debt_service",
                    right_id="annual_gdp",
                ),
            ]
            operations += [
                _retrieve(id="gdp", requirement_id="annual_gdp", spec_id=SPEC_OTHER),
                JoinOperation(id="read", requirement_id="read", left="debt", right="gdp"),
            ]
            discovered["annual_gdp"] = source(spec_id=SPEC_OTHER)
            terminal = "read"
        case _:
            group_by = (GroupBy.ENTITY, GroupBy.PERIOD)
            requirements.append(
                AggregateRequirement(
                    id="read",
                    decision_ids=["measure"],
                    input_id="annual_debt_service",
                    reduction=Reduction.SUM,
                    group_by=group_by,
                )
            )
            operations.append(_aggregate(id="read", requirement_id="read", group_by=group_by))
            terminal = "read"

    result = build(
        _draft([decision()], requirements, answer_mode=answer_mode),
        _translation(*operations, terminal=[terminal]),
        discovered=discovered,
    )

    assert result.ok, result.issues
    assert result.workflow is not None
    assert result.workflow.steps[0].args == {"spec_id": SPEC_KENYA, "without_party_totals": True}


def test_grouping_by_entity_over_an_open_axis_naming_no_entity_is_refused(
    build: Callable[..., CompileResult],
    decision: DecisionFactory,
    source: SourceFactory,
) -> None:
    # Each donor's education aid with no donor named. A group by entity keeps
    # the named parties and sums the open `origin`, so it would keep nothing.
    # The plan's own aggregate groups by period, since `planner.plan` refuses
    # the group by entity on the same rule; the operation alone states it.
    every_donor = DatasetRequirement(
        id="aid",
        decision_ids=["measure"],
        measure="education aid",
        across=("donor",),
        period=Period(start=2015, end=2024),
    )
    total = AggregateRequirement(
        id="total",
        decision_ids=["measure"],
        input_id="aid",
        reduction=Reduction.SUM,
        group_by=(GroupBy.PERIOD,),
    )
    plan = _draft([decision()], [every_donor, total])

    result = build(
        plan,
        _translation(
            RetrieveOperation(id="fetch", requirement_id="aid", spec_id=SPEC_KENYA),
            _aggregate(
                id="total", requirement_id="total", input="fetch", group_by=(GroupBy.ENTITY,)
            ),
            terminal=["total"],
        ),
        discovered={
            "aid": source(
                entities=(), open_entity_axes=(OpenEntityAxis(column="origin", member_count=55),)
            )
        },
    )

    refusals = [i for i in result.issues if i.code is Code.INPUT_SHAPE_UNSUPPORTED]
    assert [i.operation_id for i in refusals] == ["total"]
    assert "sums its open ['donor'] away, and this input names no entity" in refusals[0].reason
    assert "group by period alone" in refusals[0].reason
    assert result.workflow is None


def test_a_rank_over_a_one_member_scope_is_refused(
    build: Callable[..., CompileResult],
    decision: DecisionFactory,
    place_scope_block: dict[str, Any],
) -> None:
    # A worldwide total: `search` scopes the place axis to Earth alone, so
    # the ranking would be one row, World ranked first of one.
    place_scope_block["resolved"]["scope"]["members"] = [
        {"ref": "Earth", "label": "World", "entity_type": "Place", "role": "observationAbout"}
    ]
    worldwide = source_from_spec_block(place_scope_block)
    countries_req, countries = _countries_of("health")
    rank_req, ranked = _rank_over("countries", axis="country", reads="countries")
    plan = _draft(
        [decision(), _population_decision(decision)],
        [_every_country(), countries_req, rank_req],
    )

    result = build(
        plan,
        _translation(
            RetrieveOperation(id="fetch", requirement_id="health", spec_id=worldwide.spec_id),
            countries,
            ranked,
            terminal=["ranked"],
        ),
        discovered={"health": worldwide},
    )

    assert _codes(result) == [Code.SOURCE_UNUSABLE]
    (issue,) = result.issues
    assert issue.operation_id == "fetch"
    assert issue.requirement_id == "health"
    assert "holds 1 member on its open axis 'observationAbout'" in issue.reason
    assert "name the entities the answer needs in `entities`" in issue.reason
    assert result.workflow is None


_TWO_PLACES = [
    {"ref": "country/KEN", "label": "Kenya", "entity_type": "Country", "role": "observationAbout"},
    {"ref": "country/GHA", "label": "Ghana", "entity_type": "Country", "role": "observationAbout"},
]


@pytest.mark.parametrize(
    ("members", "refused"), [([], True), (None, True), (7, True), (_TWO_PLACES, False)]
)
def test_an_open_axis_with_an_unknown_count_or_one_below_two_is_source_unusable(
    build: Callable[..., CompileResult],
    decision: DecisionFactory,
    place_scope_block: dict[str, Any],
    members: object,
    refused: bool,
) -> None:
    # An empty list counts no member. A `members` that is no list leaves the
    # count unknown, and a source that cannot show two members on its open
    # axis holds no population, so it is refused before approval. Two listed
    # members rank.
    place_scope_block["resolved"]["scope"]["members"] = members
    scope = source_from_spec_block(place_scope_block)
    countries_req, countries = _countries_of("health")
    rank_req, ranked = _rank_over("countries", axis="country", reads="countries")
    plan = _draft(
        [decision(), _population_decision(decision)],
        [_every_country(), countries_req, rank_req],
    )

    result = build(
        plan,
        _translation(
            RetrieveOperation(id="fetch", requirement_id="health", spec_id=scope.spec_id),
            countries,
            ranked,
            terminal=["ranked"],
        ),
        discovered={"health": scope},
    )

    assert _codes(result) == ([Code.SOURCE_UNUSABLE] if refused else [])
    assert result.ok is not refused
    if refused:
        assert "its open axis 'observationAbout'" in result.issues[0].reason
        assert "instead of stating that party in `across`" in result.issues[0].reason


def test_an_axis_less_share_of_over_an_open_axis_is_offered_its_axis(
    build: Callable[..., CompileResult],
    decision: DecisionFactory,
    place_scope_block: dict[str, Any],
    source: SourceFactory,
) -> None:
    # Every country's health spending over every country's population, with
    # no axis: each side pools 193 members, so the share answers for nobody,
    # and keying it by country gives one figure per country. The plan refuses
    # it offering the axis first; the compiler reads the same rule for a share
    # operation that drops the plan's axis. Each side keeps its countries
    # first, since a share over every level of place is refused before this.
    keyed = "key the step on 'country' with `axis`"
    open_place = source_from_spec_block(place_scope_block)
    population = _every_country(id="population", measure="population")
    spenders_req, spenders = _countries_of("health", filter_id="spenders")
    peoples_req, peoples = _countries_of(
        "population", reads="fetch_population", filter_id="peoples"
    )
    share = ShareOfRequirement(
        id="share",
        decision_ids=["measure"],
        numerator=RequirementSide(input_id="spenders"),
        denominator=RequirementSide(input_id="peoples"),
    )
    steps = [_every_country(), population, spenders_req, peoples_req]

    refusal = _plan_refusal([decision()], [*steps, share])
    assert f"{keyed}, one figure per member" in refusal
    assert pooling_remedy(one_figure=True, each_side=True) in refusal

    per_country = share.model_copy(
        update={
            "axis": "country",
            "denominator_scope": DenominatorScope.PER_AXIS,
            "decision_ids": ("measure", "per_axis"),
        }
    )
    per_axis = decision(
        slot="per_axis", fixes=DecisionFixes(denominator_scope=DenominatorScope.PER_AXIS)
    )
    plan = _draft([decision(), per_axis], [*steps, per_country])
    share_op = ShareOfOperation(
        id="share",
        requirement_id="share",
        numerator=Side(input="spenders"),
        denominator=Side(input="peoples"),
    )

    result = build(
        plan,
        _translation(
            RetrieveOperation(id="fetch", requirement_id="health", spec_id=open_place.spec_id),
            RetrieveOperation(
                id="fetch_population", requirement_id="population", spec_id=SPEC_KENYA
            ),
            spenders,
            peoples,
            share_op,
            terminal=["share"],
        ),
        discovered={
            "health": open_place,
            "population": source(
                spec_id=SPEC_KENYA, entities=(), open_entity_axes=open_place.open_entity_axes
            ),
        },
    )

    whole = {
        issue.requirement_id: issue.reason
        for issue in result.issues
        if issue.code is Code.ENTITY_REQUIRED
    }
    assert keyed in whole["health"]
    assert keyed not in whole["population"]


def test_a_whole_side_over_an_open_axis_is_entity_required(
    build: Callable[..., CompileResult],
    decision: DecisionFactory,
    requirement: RequirementFactory,
    place_scope_block: dict[str, Any],
    source: SourceFactory,
) -> None:
    # Kenya against "every country" as one side: the side names none of the
    # 193 countries, so it cannot say which of them it reads. The plan
    # refuses it (`planner.shapes.open_on_whole_side`), and the compiler reads
    # the same rule for a compare whose side departs from its requirement's.
    open_place = source_from_spec_block(place_scope_block)
    kenya = requirement(
        id="kenya_health",
        measure="health spending per person",
        entities=[Entity(name="Kenya", role="country")],
        unit=None,
    )
    gap = ComparisonRequirement(
        id="gap",
        decision_ids=["measure"],
        minuend=RequirementSide(input_id="health"),
        subtrahend=RequirementSide(input_id="kenya_health"),
    )
    refusal = _plan_refusal([decision()], [kenya, _every_country(), gap])
    assert "covers every 'country' and names none of them" in refusal
    assert pooling_remedy(one_figure=False, each_side=True) in refusal
    pooled = AggregateRequirement(
        id="pooled",
        decision_ids=["measure"],
        input_id="health",
        reduction=Reduction.MEAN,
        group_by=(GroupBy.PERIOD,),
    )
    pooled_gap = gap.model_copy(update={"minuend": RequirementSide(input_id="pooled")})
    plan = _draft([decision()], [kenya, _every_country(), pooled, pooled_gap])
    compare = CompareOperation(
        id="gap", requirement_id="gap", left=Side(input="kenya"), right=Side(input="fetch")
    )

    result = build(
        plan,
        _translation(
            RetrieveOperation(id="kenya", requirement_id="kenya_health", spec_id=SPEC_KENYA),
            RetrieveOperation(id="fetch", requirement_id="health", spec_id=open_place.spec_id),
            _aggregate(
                id="pooled",
                requirement_id="pooled",
                input="fetch",
                reduction=Reduction.MEAN,
            ),
            compare,
            terminal=["gap"],
        ),
        discovered={
            "kenya_health": source(
                entities=(
                    ResolvedEntity(role="observationAbout", ref="country/KEN", label="Kenya"),
                ),
                unit_ref=open_place.unit_ref,
            ),
            "health": open_place,
        },
    )

    (issue,) = [issue for issue in result.issues if issue.code is Code.ENTITY_REQUIRED]
    assert issue.requirement_id == "health"
    assert "covers every 'country' and names none of them" in issue.reason
    assert "name them in the requirement's `entities`" in issue.reason
    assert _pools(result, one_figure=False)


def test_a_donor_across_a_lone_open_destination_is_source_unusable(
    build: Callable[..., CompileResult],
    decision: DecisionFactory,
    source: SourceFactory,
) -> None:
    # The source leaves only the receiving side open, so no column holds
    # every donor; the rank over it builds no step and the source's own
    # finding is the one issue.
    every_donor = DatasetRequirement(
        id="aid",
        decision_ids=["measure"],
        measure="education aid",
        across=("donor",),
        period=Period(start=2015, end=2024),
    )
    rank_req, ranked = _rank_over("aid", axis="donor")
    plan = _draft([decision(), _population_decision(decision)], [every_donor, rank_req])

    result = build(
        plan,
        _translation(
            RetrieveOperation(id="fetch", requirement_id="aid", spec_id=SPEC_KENYA),
            ranked,
            terminal=["ranked"],
        ),
        discovered={
            "aid": source(
                entities=(),
                open_entity_axes=(OpenEntityAxis(column="destination", member_count=30),),
            )
        },
    )

    assert _codes(result) == [Code.SOURCE_UNUSABLE]
    assert "'destination'" in result.issues[0].reason


def _rank_every_donor_with_destination_open(
    build: Callable[..., CompileResult],
    decision: DecisionFactory,
    source: SourceFactory,
    recipients: int | None,
    across: tuple[str, ...] = ("donor",),
) -> CompileResult:
    """A rank over every donor, compiled against a source that also leaves `destination` open."""
    every_donor = DatasetRequirement(
        id="aid",
        decision_ids=["measure"],
        measure="education aid",
        across=across,
        period=Period(start=2015, end=2024),
    )
    rank_req, ranked = _rank_over("aid", axis="donor")
    plan = _draft([decision(), _population_decision(decision)], [every_donor, rank_req])
    both_open = source(
        entities=(),
        open_entity_axes=(
            OpenEntityAxis(column="destination", member_count=recipients),
            OpenEntityAxis(column="origin", member_count=55),
        ),
    )
    return build(
        plan,
        _translation(
            RetrieveOperation(id="fetch", requirement_id="aid", spec_id=SPEC_KENYA),
            ranked,
            terminal=["ranked"],
        ),
        discovered={"aid": both_open},
    )


@pytest.mark.parametrize("recipients", [140, None])
def test_a_source_leaving_both_flow_sides_open_is_source_unusable(
    build: Callable[..., CompileResult],
    decision: DecisionFactory,
    source: SourceFactory,
    recipients: int | None,
) -> None:
    # "Top donors of education aid" with every recipient open too: each donor
    # would carry one row per recipient, which a rank would average silently.
    # An uncounted `destination` may hold several, so it refuses as 140 does.
    result = _rank_every_donor_with_destination_open(build, decision, source, recipients)

    assert _codes(result) == [Code.SOURCE_UNUSABLE]
    assert "also leaves 'destination' open over several members" in result.issues[0].reason
    assert "add its party word to `across`" in result.issues[0].reason


def test_a_source_leaving_both_flow_sides_open_compiles_when_across_states_both(
    build: Callable[..., CompileResult],
    decision: DecisionFactory,
    source: SourceFactory,
) -> None:
    result = _rank_every_donor_with_destination_open(
        build, decision, source, 140, across=("donor", "recipient")
    )

    assert _codes(result) == []


def test_a_second_open_axis_holding_one_member_leaves_the_rank_to_compile(
    build: Callable[..., CompileResult],
    decision: DecisionFactory,
    source: SourceFactory,
) -> None:
    # The source leaves `destination` open with one member on it, so each
    # donor carries one row per period and nothing repeats.
    result = _rank_every_donor_with_destination_open(build, decision, source, 1)

    assert result.ok, result.issues
    assert result.workflow is not None
    rank = result.workflow.steps[-1]
    assert rank.args == {"rank_by": ["origin"], "ascending": False, "aggregation": "mean"}


def test_a_whole_side_over_an_open_axis_beside_a_named_entity_is_refused(
    decision: DecisionFactory, requirement: RequirementFactory
) -> None:
    # Kenya's aid from every donor as one side names none of the donors, as
    # every donor's aid with nobody named does.
    uganda = requirement(
        id="uganda_aid",
        measure="education aid",
        entities=[Entity(name="Uganda", role="recipient")],
        unit=None,
    )
    gap = ComparisonRequirement(
        id="gap",
        decision_ids=["measure"],
        minuend=RequirementSide(input_id="uganda_aid"),
        subtrahend=RequirementSide(input_id="aid"),
    )

    refusal = _plan_refusal([decision()], [_kenya_from_every_donor(), uganda, gap])

    assert "covers every 'donor' and names none of them" in refusal


def test_a_keyed_share_over_an_open_axis_divides_member_by_member(
    build: Callable[..., CompileResult],
    decision: DecisionFactory,
    source: SourceFactory,
) -> None:
    # Each donor's education aid to Kenya over its total aid to Kenya, keyed
    # by donor on both sides: each side reads every donor as its own series,
    # so neither has to name one.
    kenya_destination = ResolvedEntity(role="destination", ref="country/KEN", label="Kenya")
    total = _kenya_from_every_donor().model_copy(update={"id": "total", "measure": "total aid"})
    share = ShareOfRequirement(
        id="share",
        decision_ids=["measure"],
        numerator=RequirementSide(input_id="aid"),
        denominator=RequirementSide(input_id="total"),
        axis="donor",
        denominator_scope=DenominatorScope.PER_AXIS,
    )
    plan = _draft([decision()], [_kenya_from_every_donor(), total, share])
    every_donor = {
        "entities": (kenya_destination,),
        "open_entity_axes": (OpenEntityAxis(column="origin", member_count=55),),
        "coverage_window": Period(start=2015, end=2023),
    }

    result = build(
        plan,
        _translation(
            RetrieveOperation(id="fetch", requirement_id="aid", spec_id=SPEC_KENYA),
            RetrieveOperation(id="fetch_total", requirement_id="total", spec_id=SPEC_OTHER),
            ShareOfOperation(
                id="share",
                requirement_id="share",
                numerator=Side(input="fetch"),
                denominator=Side(input="fetch_total"),
                axis="donor",
                denominator_scope=DenominatorScope.PER_AXIS,
            ),
            terminal=["share"],
        ),
        discovered={
            "aid": source(**every_donor),
            "total": source(spec_id=SPEC_OTHER, **every_donor),
        },
    )

    assert result.ok, result.issues
    assert result.workflow is not None
    share_step = result.workflow.steps[-1]
    assert share_step.tool.name == "share_of"
    assert share_step.args == {"numerator_by": ["origin"], "denominator_by": {"origin": "origin"}}
    assert share_step.expected_role_keys == _frame_role_columns(["origin"])


# --- An aggregate over an open axis -------------------------------------------------

# The two CRS sources the recorded DRC plans pin (`propose_plan_drc_aid.json`),
# aid to the DRC from the United States and from every donor.
SPEC_US_TO_DRC = "spec_459728244c8b0efeaf933157d9dc1ce3"
SPEC_EVERY_DONOR_TO_DRC = "spec_2cdf1c28095aa2918dc7039705ada64b"
DRC = ResolvedEntity(
    role="destination", ref="country/COD", label="Democratic Republic of the Congo"
)
UNITED_STATES = ResolvedEntity(role="origin", ref="country/USA", label="United States")
_DRC_PLANS = json.loads(
    (Path(__file__).parent.parent / "fixtures" / "propose_plan_drc_aid.json").read_text()
)


def _drc_sources(source: SourceFactory) -> dict[str, SelectedSource]:
    """The sources the plans pin: the US flow bound both sides, every donor's open on `origin`."""
    return {
        "us_disbursements": source(
            spec_id=SPEC_US_TO_DRC,
            entities=(DRC, UNITED_STATES),
            **_pinned("UsOdaToDrc"),
        ),
        "all_donors_disbursements": source(
            spec_id=SPEC_EVERY_DONOR_TO_DRC,
            entities=(DRC,),
            open_entity_axes=(OpenEntityAxis(column="origin", member_count=77),),
            **_pinned("OdaToDrc"),
        ),
    }


def _replayed(arguments: dict[str, Any]) -> Plan:
    """The draft `propose_plan` records from `arguments`."""
    requirements = TypeAdapter(list[EvidenceRequirement]).validate_python(
        arguments["evidence_requirements"]
    )
    drafts = TypeAdapter(list[DecisionDraft]).validate_python(arguments["decisions"])
    decisions = settle_decisions(drafts, requirements)
    # An amendment states no question, so the plan keeps the first draft's.
    return Plan(
        question=_DRC_PLANS["plan_v1"]["question"],
        answer_mode=arguments["answer_mode"],
        decisions=decisions,
        evidence_requirements=requirements,
        asks=_DRC_PLANS["asks"],
        answered=arguments["answered"],
        status=PlanStatus.DRAFT,
    )


@pytest.mark.parametrize(
    ("version", "total_group_by", "terminal"),
    [
        (
            "plan_v1",
            ["time_period"],
            (
                "us_disbursements_series",
                "total_disbursements_series",
                "us_growth",
                "total_growth",
                "us_share_window",
            ),
        ),
        (
            "plan_v3",
            ["time_period", "destination"],
            (
                "us_disbursements_series",
                "total_disbursements_series",
                "us_growth",
                "total_growth",
                "us_share_window",
            ),
        ),
        (
            "first_attempt",
            ["time_period"],
            (
                "us_disbursements",
                "total_disbursements",
                "us_growth",
                "total_growth",
                "us_share",
                "us_share_window",
            ),
        ),
    ],
    ids=["plan_v1", "plan_v3", "first_attempt"],
)
def test_the_drc_drafts_compile_and_total_every_donor(
    build: Callable[..., CompileResult],
    source: SourceFactory,
    version: str,
    total_group_by: list[str],
    terminal: tuple[str, ...],
) -> None:
    # The first recorded draft divides the US flow by a total over every
    # donor, which an aggregate by period has already collapsed to one series,
    # so a whole side naming no donor reads that one series. The later draft
    # groups the total by entity too, which keeps the named DRC and still sums
    # the 77 donors. The attempt before both answers asks with requirements
    # another reads, the retrieved US flow among them, and each of those is a
    # terminal too.
    sources = _drc_sources(source)
    plan = _replayed(_DRC_PLANS[version])
    translation = build_translation(
        plan, {requirement_id: s.spec_id for requirement_id, s in sources.items()}
    )

    result = build(plan, translation, discovered=sources)

    assert result.ok, result.issues
    assert result.workflow is not None
    assert set(result.workflow.terminal_steps) == set(terminal)
    (total,) = [step for step in result.workflow.steps if step.id == "total_disbursements"]
    assert total.args == {"aggregation": "sum", "group_by": total_group_by}
    assert total.expected_role_keys == _frame_role_columns(total_group_by[1:])


UNITED_KINGDOM = ResolvedEntity(role="origin", ref="country/GBR", label="United Kingdom")
GERMANY = ResolvedEntity(role="origin", ref="country/DEU", label="Germany")


@pytest.mark.parametrize(
    ("axis", "one_figure"),
    [pytest.param(None, True, id="un-keyed"), pytest.param("donor", False, id="keyed")],
)
def test_a_share_side_pooling_two_donors_is_offered_one_figure_only_un_keyed(
    build: Callable[..., CompileResult],
    decision: DecisionFactory,
    source: SourceFactory,
    axis: str | None,
    one_figure: bool,
) -> None:
    # Germany's ODA to Kenya, or keyed by donor every donor's, over the US's
    # and the UK's together, read whole. Un-keyed, one figure on each side is
    # a share; keyed by donor, the numerator keeps one series per donor and
    # year, which one figure of the denominator would not match.
    years = Period(start=2015, end=2022)
    kenya = Entity(name="Kenya", role="recipient")
    every = DatasetRequirement(
        id="every",
        decision_ids=["measure"],
        measure="ODA",
        entities=[kenya],
        across=("donor",),
        period=years,
    )
    germany = DatasetRequirement(
        id="germany",
        decision_ids=["measure"],
        measure="ODA",
        entities=[Entity(name="Germany", role="donor"), kenya],
        period=years,
    )
    numerator = every if axis is not None else germany
    named = DatasetRequirement(
        id="named",
        decision_ids=["measure"],
        measure="ODA",
        entities=[
            Entity(name="United States", role="donor"),
            Entity(name="United Kingdom", role="donor"),
            kenya,
        ],
        period=years,
    )
    scope = None if axis is None else DenominatorScope.POOLED
    pooled = decision(slot="pooled", fixes=DecisionFixes(denominator_scope=DenominatorScope.POOLED))
    share = ShareOfRequirement(
        id="share",
        decision_ids=["measure"] if axis is None else ["measure", "pooled"],
        numerator=RequirementSide(input_id=numerator.id),
        denominator=RequirementSide(input_id="named"),
        axis=axis,
        denominator_scope=scope,
    )
    numerator_source = (
        source(
            spec_id=SPEC_KENYA,
            entities=(_KENYA,),
            open_entity_axes=(OpenEntityAxis(column="origin", member_count=55),),
            **_pinned("Oda"),
        )
        if axis is not None
        else source(spec_id=SPEC_KENYA, entities=(GERMANY, _KENYA), **_pinned("Oda"))
    )
    sources = {
        numerator.id: numerator_source,
        "named": source(
            spec_id=SPEC_OTHER, entities=(UNITED_STATES, UNITED_KINGDOM, _KENYA), **_pinned("Oda")
        ),
    }
    plan = _draft([decision(), pooled][: 1 + (axis is not None)], [numerator, named, share])
    translation = build_translation(
        plan, {requirement_id: s.spec_id for requirement_id, s in sources.items()}
    )

    result = build(plan, translation, discovered=sources)

    assert _pools(result, one_figure=one_figure), result.issues


@pytest.mark.parametrize(
    ("group_by", "refused"),
    [
        pytest.param((), (), id="one-figure"),
        pytest.param(
            (GroupBy.ENTITY,),
            (Code.ENTITY_REQUIRED, Code.SHARE_AXIS_REQUIRED),
            id="by-entity",
        ),
    ],
)
def test_a_share_two_donors_hold_of_every_donors_total_compiles_from_one_figure_totals(
    build: Callable[..., CompileResult],
    decision: DecisionFactory,
    source: SourceFactory,
    group_by: tuple[GroupBy, ...],
    refused: tuple[Code, ...],
) -> None:
    # "What share of the ODA Kenya received in 2015-2022 came from the
    # United States and the United Kingdom together?" A total grouped by
    # entity keeps the two donors apart, so the unkeyed share takes a whole
    # side naming two donors and pools them, and is refused. A total over
    # every row is one figure on each side, and the share divides one by the
    # other.
    years = Period(start=2015, end=2022)
    kenya = Entity(name="Kenya", role="recipient")
    named = DatasetRequirement(
        id="named",
        decision_ids=["measure"],
        measure="ODA",
        entities=[
            Entity(name="United States", role="donor"),
            Entity(name="United Kingdom", role="donor"),
            kenya,
        ],
        period=years,
    )
    every = DatasetRequirement(
        id="every",
        decision_ids=["measure"],
        measure="ODA",
        entities=[kenya],
        across=("donor",),
        period=years,
    )
    totals = [
        AggregateRequirement(
            id=f"{side.id}_total",
            decision_ids=["measure"],
            input_id=side.id,
            reduction=Reduction.SUM,
            group_by=group_by,
        )
        for side in (named, every)
    ]
    share = ShareOfRequirement(
        id="share",
        decision_ids=["measure"],
        numerator=RequirementSide(input_id="named_total"),
        denominator=RequirementSide(input_id="every_total"),
    )
    plan = _draft([decision()], [named, every, *totals, share])
    sources = {
        "named": source(
            spec_id=SPEC_KENYA,
            entities=(UNITED_STATES, UNITED_KINGDOM, _KENYA),
            **_pinned("Oda"),
        ),
        "every": source(
            spec_id=SPEC_OTHER,
            entities=(_KENYA,),
            open_entity_axes=(OpenEntityAxis(column="origin", member_count=55),),
            **_pinned("Oda"),
        ),
    }
    translation = build_translation(
        plan, {requirement_id: s.spec_id for requirement_id, s in sources.items()}
    )

    result = build(plan, translation, discovered=sources)

    if refused:
        assert _codes(result) == list(refused)
        assert result.workflow is None
        assert _pools(result, one_figure=True)
        return
    assert result.ok, result.issues
    assert result.workflow is not None
    steps = {step.id: step for step in result.workflow.steps}
    for total in totals:
        assert steps[total.id].args == {"aggregation": "sum", "group_by": []}
    (share_op,) = [op for op in translation.operations if isinstance(op, ShareOfOperation)]
    assert share_op.axis is None
    assert steps["share"].args == {}


@pytest.mark.parametrize(
    ("named", "across", "open_column", "kept"),
    [
        (Entity(name="Democratic Republic of the Congo", role="recipient"), "donor", "origin", DRC),
        (Entity(name="United States", role="donor"), "recipient", "destination", UNITED_STATES),
    ],
    ids=["recipient-named-across-donor", "donor-named-across-recipient"],
)
def test_grouping_by_entity_writes_the_named_party_and_sums_the_open_one(
    build: Callable[..., CompileResult],
    decision: DecisionFactory,
    source: SourceFactory,
    named: Entity,
    across: str,
    open_column: str,
    kept: ResolvedEntity,
) -> None:
    flow = DatasetRequirement(
        id="aid",
        decision_ids=["measure"],
        measure="official development assistance",
        entities=[named],
        across=(across,),
        period=Period(start=2015, end=2024),
    )
    total = AggregateRequirement(
        id="total",
        decision_ids=["measure"],
        input_id="aid",
        reduction=Reduction.SUM,
        group_by=(GroupBy.PERIOD, GroupBy.ENTITY),
    )

    result = build(
        _draft([decision()], [flow, total]),
        _translation(
            RetrieveOperation(id="fetch", requirement_id="aid", spec_id=SPEC_KENYA),
            _aggregate(
                id="total",
                requirement_id="total",
                input="fetch",
                group_by=(GroupBy.PERIOD, GroupBy.ENTITY),
            ),
            terminal=["total"],
        ),
        discovered={
            "aid": source(
                entities=(kept,),
                open_entity_axes=(OpenEntityAxis(column=open_column, member_count=77),),
            )
        },
    )

    assert result.ok, result.issues
    assert result.workflow is not None
    total_step = result.workflow.steps[-1]
    assert total_step.args == {"aggregation": "sum", "group_by": ["time_period", kept.role]}
    assert total_step.expected_role_keys == _frame_role_columns([kept.role])


# --- A keyed side of a dividing verb, and quotient_of ------------------------------

_EAST_AFRICA = {"Kenya": "country/KEN", "Uganda": "country/UGA", "Tanzania": "country/TZA"}


def _east_africa(column: str) -> tuple[ResolvedEntity, ...]:
    """Kenya, Uganda and Tanzania as a source resolves them under `column`."""
    return tuple(
        ResolvedEntity(role=column, ref=ref, label=label) for label, ref in _EAST_AFRICA.items()
    )


def _named(names: Sequence[str], role: str) -> list[Entity]:
    return [Entity(name=name, role=role) for name in names]


def _per_recipient(
    verb: str,
    source: SourceFactory,
    *,
    named: Sequence[str] = ("Kenya",),
    whole_named: Sequence[str] | None = None,
    keyed: bool = True,
    whole_keyed: bool = True,
) -> tuple[list[EvidenceRequirement], Translation, dict[str, SelectedSource]]:
    """Aid received set over a second series by `verb`, each source resolving three countries.

    The aid source holds its recipients on `destination`. A `quotient_of`
    divides by population, a unit-less count whose requirement names the
    same countries with no role, held on `observationAbout` and keyed by the
    place word "country". A `share_of` divides by total aid in the aid's own
    unit, held on `recipient`.

    `named` and `whole_named` are the entities each side's requirement
    states. `keyed` keys the numerator by recipient, and `whole_keyed` the
    second side by its own phrase for them.
    """
    whole_named = named if whole_named is None else whole_named
    sides = (RequirementSide(input_id="aid"), RequirementSide(input_id="whole"))
    numerator, denominator = Side(input="fetch_aid"), Side(input="fetch_whole")
    axis = "recipient" if keyed else None
    derived: EvidenceRequirement
    operation: Operation
    if verb == "quotient_of":
        phrase, column, measure, unit_ref = "country", "observationAbout", "population", None
        fields: dict[str, Any] = {
            "axis": axis,
            "denominator_axis": phrase if keyed and whole_keyed else None,
        }
        derived = QuotientRequirement(
            id="divided",
            decision_ids=["measure"],
            numerator=sides[0],
            denominator=sides[1],
            **fields,
        )
        operation = QuotientOfOperation(
            id="divided",
            requirement_id="divided",
            numerator=numerator,
            denominator=denominator,
            **fields,
        )
    else:
        phrase, column, measure, unit_ref = "recipient", "recipient", "total aid", "USDollar"
        fields = {"axis": axis, "denominator_scope": DenominatorScope.PER_AXIS if keyed else None}
        derived = ShareOfRequirement(
            id="divided",
            decision_ids=["measure"],
            numerator=sides[0],
            denominator=sides[1],
            **fields,
        )
        operation = ShareOfOperation(
            id="divided",
            requirement_id="divided",
            numerator=numerator,
            denominator=denominator,
            **fields,
        )
    period = Period(start=2015, end=2024)
    aid = DatasetRequirement(
        id="aid",
        decision_ids=["measure"],
        measure="aid received",
        entities=_named(named, "recipient"),
        period=period,
    )
    whole = DatasetRequirement(
        id="whole",
        decision_ids=["measure"],
        measure=measure,
        entities=[Entity(name=name) for name in whole_named]
        if verb == "quotient_of"
        else _named(whole_named, phrase),
        period=period,
    )
    translation = _translation(
        RetrieveOperation(id="fetch_aid", requirement_id="aid", spec_id=SPEC_KENYA),
        RetrieveOperation(id="fetch_whole", requirement_id="whole", spec_id=SPEC_OTHER),
        operation,
        terminal=["divided"],
    )
    discovered = {
        "aid": source(
            **_pinned(
                "AidReceived", {"entities": _east_africa("destination"), "unit_ref": "USDollar"}
            )
        ),
        "whole": source(
            spec_id=SPEC_OTHER,
            **_pinned("Whole", {"entities": _east_africa(column), "unit_ref": unit_ref}),
        ),
    }
    return [aid, whole, derived], translation, discovered


@pytest.mark.parametrize(
    ("verb", "keys"),
    [
        (
            "share_of",
            {"numerator_by": ["destination"], "denominator_by": {"destination": "recipient"}},
        ),
        (
            "quotient_of",
            {
                "numerator_by": ["destination"],
                "denominator_by": {"destination": "observationAbout"},
                "numerator_unit": "USDollar",
            },
        ),
    ],
)
@pytest.mark.parametrize(
    "named",
    [("Kenya",), ("Kenya", "Uganda", "Tanzania")],
    ids=["one-named-beside-surplus-members", "several-named"],
)
def test_a_side_keyed_by_its_axis_reads_every_member_of_it(
    build: Callable[..., CompileResult],
    decision: DecisionFactory,
    source: SourceFactory,
    verb: str,
    keys: dict[str, Any],
    named: tuple[str, ...],
) -> None:
    # Each recipient's aid over its own counterpart. A keyed side holds one
    # series per member of its key column, so naming several entities there,
    # or one beside others the source resolves, pools nothing.
    requirements, translation, discovered = _per_recipient(verb, source, named=named)

    result = build(_draft([decision()], requirements), translation, discovered=discovered)

    assert result.ok, result.issues
    assert result.workflow is not None
    step = result.workflow.steps[-1]
    assert step.tool.name == verb
    assert step.args == keys
    assert step.bindings == {
        "frame_ref": Binding(from_step="fetch_aid", output=ArtifactKind.FRAME_REF),
        "denominator_frame_ref": Binding(from_step="fetch_whole", output=ArtifactKind.FRAME_REF),
    }
    assert step.expected_role_keys == _frame_role_columns(["destination"])


def test_a_keyed_side_names_an_entity_on_a_column_it_is_not_keyed_by(
    build: Callable[..., CompileResult],
    decision: DecisionFactory,
    source: SourceFactory,
) -> None:
    # The key exempts its own column alone. The aid source also resolves two
    # donors on `origin`, and the numerator, keyed by recipient, still has to
    # say which donor it reads.
    requirements, translation, discovered = _per_recipient("quotient_of", source)
    donors = (
        ResolvedEntity(role="origin", ref="country/USA", label="United States"),
        ResolvedEntity(role="origin", ref="country/DEU", label="Germany"),
    )
    aid = requirements[0]
    assert isinstance(aid, DatasetRequirement)
    requirements[0] = aid.model_copy(
        update={"entities": (*aid.entities, *_named(["United States", "Germany"], "donor"))}
    )
    discovered["aid"] = discovered["aid"].model_copy(
        update={"entities": (*discovered["aid"].entities, *donors)}
    )

    result = build(_draft([decision()], requirements), translation, discovered=discovered)

    assert [(issue.code, issue.requirement_id) for issue in result.issues] == [
        (Code.ENTITY_REQUIRED, "aid")
    ]


def test_a_keyed_quotient_over_an_open_axis_on_both_sides_divides_member_by_member(
    build: Callable[..., CompileResult],
    decision: DecisionFactory,
    place_scope_block: dict[str, Any],
    source: SourceFactory,
) -> None:
    # United States aid to every recipient over every country's population:
    # neither side names a member of its key, and the two sides hold them
    # under different columns. The population keeps its countries first, as
    # any reader of an open place axis does.
    open_place = source_from_spec_block(place_scope_block)
    aid = DatasetRequirement(
        id="aid",
        decision_ids=["measure"],
        measure="aid given",
        entities=[Entity(name="United States", role="donor")],
        across=("recipient",),
        period=Period(start=2015, end=2024),
    )
    population = _every_country(id="population", measure="population")
    peoples_req, peoples = _countries_of(
        "population", reads="fetch_population", filter_id="peoples"
    )
    fields: dict[str, Any] = {"axis": "recipient", "denominator_axis": "country"}
    per_head = QuotientRequirement(
        id="per_head",
        decision_ids=["measure"],
        numerator=RequirementSide(input_id="aid"),
        denominator=RequirementSide(input_id="peoples"),
        **fields,
    )

    result = build(
        _draft([decision()], [aid, population, peoples_req, per_head]),
        _translation(
            RetrieveOperation(id="fetch", requirement_id="aid", spec_id=SPEC_KENYA),
            RetrieveOperation(
                id="fetch_population", requirement_id="population", spec_id=SPEC_OTHER
            ),
            peoples,
            QuotientOfOperation(
                id="per_head",
                requirement_id="per_head",
                numerator=Side(input="fetch"),
                denominator=Side(input="peoples"),
                **fields,
            ),
            terminal=["per_head"],
        ),
        discovered={
            "aid": source(
                entities=(UNITED_STATES,),
                open_entity_axes=(OpenEntityAxis(column="destination", member_count=140),),
                unit_ref="USDollar",
            ),
            "population": source(
                spec_id=SPEC_OTHER,
                entities=(),
                open_entity_axes=open_place.open_entity_axes,
                unit_ref=None,
            ),
        },
    )

    assert result.ok, result.issues
    assert result.workflow is not None
    step = result.workflow.steps[-1]
    assert step.tool.name == "quotient_of"
    assert step.args == {
        "numerator_by": ["destination"],
        "denominator_by": {"destination": "observationAbout"},
        "numerator_unit": "USDollar",
    }
    assert step.expected_role_keys == _frame_role_columns(["destination"])


def test_a_quotient_keyed_on_its_numerator_alone_needs_one_series_to_divide_by(
    build: Callable[..., CompileResult],
    decision: DecisionFactory,
    source: SourceFactory,
) -> None:
    # Each recipient's aid over a population naming three countries, with no
    # `denominator_axis`: every key would divide by one total, and the three
    # series are not summed into one nobody asked for.
    requirements, translation, discovered = _per_recipient(
        "quotient_of", source, named=("Kenya", "Uganda", "Tanzania"), whole_keyed=False
    )

    result = build(_draft([decision()], requirements), translation, discovered=discovered)

    assert [(issue.code, issue.requirement_id) for issue in result.issues] == [
        (Code.ENTITY_REQUIRED, "whole")
    ]
    assert result.workflow is None


def test_a_quotient_keyed_on_its_numerator_alone_divides_every_key_by_one_total(
    build: Callable[..., CompileResult],
    decision: DecisionFactory,
    source: SourceFactory,
) -> None:
    # Each recipient's aid over Kenya's population, one series: the call
    # carries no `denominator_by`.
    requirements, translation, discovered = _per_recipient(
        "quotient_of",
        source,
        named=("Kenya", "Uganda", "Tanzania"),
        whole_named=("Kenya",),
        whole_keyed=False,
    )
    discovered["whole"] = discovered["whole"].model_copy(
        update={"entities": _east_africa("observationAbout")[:1]}
    )

    result = build(_draft([decision()], requirements), translation, discovered=discovered)

    assert result.ok, result.issues
    assert result.workflow is not None
    assert result.workflow.steps[-1].args == {
        "numerator_by": ["destination"],
        "numerator_unit": "USDollar",
    }


def test_an_axis_less_quotient_of_a_multi_member_numerator_is_quotient_axis_required(
    build: Callable[..., CompileResult],
    decision: DecisionFactory,
    source: SourceFactory,
) -> None:
    # Three recipients' aid over Kenya's population with no axis: the
    # numerator would sum the three into a figure for nobody in particular.
    requirements, translation, discovered = _per_recipient(
        "quotient_of", source, named=("Kenya", "Uganda", "Tanzania"), keyed=False
    )

    result = build(_draft([decision()], requirements), translation, discovered=discovered)

    assert (Code.QUOTIENT_AXIS_REQUIRED, "divided") in [
        (issue.code, issue.operation_id) for issue in result.issues
    ]
    assert result.workflow is None


def _per_head(
    requirement: RequirementFactory,
) -> tuple[list[EvidenceRequirement], Translation]:
    """Kenya's debt service divided by its population, each read by its own retrieval."""
    population = requirement(id="population", measure="total population", unit=None)
    per_head = QuotientRequirement(
        id="per_head",
        decision_ids=["measure"],
        numerator=RequirementSide(input_id="annual_debt_service"),
        denominator=RequirementSide(input_id="population"),
    )
    translation = _translation(
        _retrieve(),
        _retrieve(id="people", requirement_id="population", spec_id=SPEC_OTHER),
        QuotientOfOperation(
            id="per_head",
            requirement_id="per_head",
            numerator=Side(input="debt"),
            denominator=Side(input="people"),
        ),
        terminal=["per_head"],
    )
    return [requirement(), population, per_head], translation


@pytest.mark.parametrize(
    ("numerator_source", "denominator_source", "pins"),
    [
        pytest.param(
            {"unit_ref": "USDollar"},
            {"unit_ref": None},
            {"numerator_unit": "USDollar"},
            id="a-unit-over-a-unit-less-count",
        ),
        pytest.param(
            _BOUND_BY_REQUEST,
            {"unit_ref": None},
            {"numerator_unit": {"currency": "USD", "prices": "current"}},
            id="a-converted-unit-pins-the-request-object",
        ),
        pytest.param(
            {"unit_ref": "CurrencyUSD"},
            {"unit_ref": "InternationalDollar"},
            {"numerator_unit": "CurrencyUSD", "denominator_unit": "InternationalDollar"},
            id="two-different-units",
        ),
        pytest.param(
            {"unit_ref": None},
            {"unit_ref": "SquareKilometer"},
            {"denominator_unit": "SquareKilometer"},
            id="a-unit-less-count-over-a-unit",
        ),
        pytest.param(_PER_ROW_UNIT, {"unit_ref": None}, {}, id="a-currency-per-row-is-unpinned"),
    ],
)
def test_a_quotient_takes_any_pair_of_units_and_pins_each_sides_own(
    build: Callable[..., CompileResult],
    decision: DecisionFactory,
    requirement: RequirementFactory,
    source: SourceFactory,
    numerator_source: dict[str, Any],
    denominator_source: dict[str, Any],
    pins: dict[str, Any],
) -> None:
    # Every pair `share_of`, `compare` and `sum_of` refuse before running
    # compiles as a quotient, each side pinned to the unit its frame declares.
    requirements, translation = _per_head(requirement)

    result = build(
        _draft([decision()], requirements),
        translation,
        discovered={
            "annual_debt_service": source(**_pinned("DebtServicePaid", numerator_source)),
            "population": source(spec_id=SPEC_OTHER, **_pinned("Count_Person", denominator_source)),
        },
    )

    assert result.ok, result.issues
    assert result.workflow is not None
    step = result.workflow.steps[-1]
    assert step.tool.name == "quotient_of"
    assert step.args == pins
    assert step.bindings == {
        "frame_ref": Binding(from_step="debt", output=ArtifactKind.FRAME_REF),
        "denominator_frame_ref": Binding(from_step="people", output=ArtifactKind.FRAME_REF),
    }
    assert step.expected_role_keys == ()


def _unit_less_producer(
    kind: str,
) -> tuple[EvidenceRequirement, Operation]:
    """A calculation over Kenya's and Uganda's debt service whose value has no unit of its own."""
    if kind == "share_of":
        fields: dict[str, Any] = {"axis": "debtor", "denominator_scope": DenominatorScope.PER_AXIS}
        return (
            ShareOfRequirement(
                id="made",
                decision_ids=["measure"],
                numerator=RequirementSide(input_id="annual_debt_service"),
                denominator=RequirementSide(input_id="exports"),
                **fields,
            ),
            ShareOfOperation(
                id="made",
                requirement_id="made",
                numerator=Side(input="debt"),
                denominator=Side(input="sold"),
                **fields,
            ),
        )
    if kind == "quotient_of":
        fields = {"axis": "debtor", "denominator_axis": "debtor"}
        return (
            QuotientRequirement(
                id="made",
                decision_ids=["measure"],
                numerator=RequirementSide(input_id="annual_debt_service"),
                denominator=RequirementSide(input_id="exports"),
                **fields,
            ),
            QuotientOfOperation(
                id="made",
                requirement_id="made",
                numerator=Side(input="debt"),
                denominator=Side(input="sold"),
                **fields,
            ),
        )
    fields = {
        "direction": RankDirection.HIGHEST_FIRST,
        "among": RankAmong.POPULATION,
        "per_period": True,
    }
    return (
        RankRequirement(
            id="made",
            decision_ids=["measure", "population"],
            input_id="annual_debt_service",
            **fields,
        ),
        RankOperation(id="made", requirement_id="made", input="debt", **fields),
    )


@pytest.mark.parametrize("kind", ["share_of", "quotient_of", "rank"])
def test_a_quotient_side_holding_a_ratio_a_rate_or_a_rank_position_is_refused(
    build: Callable[..., CompileResult],
    decision: DecisionFactory,
    requirement: RequirementFactory,
    source: SourceFactory,
    kind: str,
) -> None:
    # A share, a quotient and a rank position have no unit to be one part of
    # a composed unit, and analysis-mcp refuses each as a side. The plan's
    # own quotient reads the measured series, since `Plan` refuses the same
    # sides; the operation alone reads the calculation.
    two_debtors = _named(["Kenya", "Uganda"], "debtor")
    debt = requirement(entities=two_debtors)
    exports = requirement(
        id="exports", measure="exports of goods and services", entities=two_debtors
    )
    made_req, made = _unit_less_producer(kind)
    fields: dict[str, Any] = {"axis": "debtor", "denominator_axis": "debtor"}
    over = QuotientRequirement(
        id="over",
        decision_ids=["measure"],
        numerator=RequirementSide(input_id="exports"),
        denominator=RequirementSide(input_id="annual_debt_service"),
        **fields,
    )
    decisions = [decision()]
    if kind == "rank":
        decisions.append(_population_decision(decision))
    plan = _draft(decisions, [debt, exports, made_req, over])
    debtors = tuple(
        ResolvedEntity(role="debtor", ref=_EAST_AFRICA[name], label=name)
        for name in ("Kenya", "Uganda")
    )
    discovered = {
        "annual_debt_service": source(**_pinned("DebtServicePaid", {"entities": debtors})),
        "exports": source(spec_id=SPEC_OTHER, **_pinned("Exports", {"entities": debtors})),
    }
    retrievals = (
        _retrieve(),
        _retrieve(id="sold", requirement_id="exports", spec_id=SPEC_OTHER),
        made,
    )

    def over_reading(numerator: str) -> QuotientOfOperation:
        return QuotientOfOperation(
            id="over",
            requirement_id="over",
            numerator=Side(input=numerator),
            denominator=Side(input="debt"),
            **fields,
        )

    measured = build(
        plan,
        _translation(*retrievals, over_reading("sold"), terminal=["made", "over"]),
        discovered=discovered,
    )
    derived = build(
        plan,
        _translation(*retrievals, over_reading("made"), terminal=["over"]),
        discovered=discovered,
    )

    assert measured.ok, measured.issues
    assert (Code.INPUT_SHAPE_UNSUPPORTED, "over") in [
        (issue.code, issue.operation_id) for issue in derived.issues
    ]
    assert derived.workflow is None


@pytest.mark.parametrize(
    ("unit_ref", "refused"),
    [(None, True), ("USDollar", False)],
    ids=["a-sum-of-unit-less-counts", "a-sum-in-a-unit"],
)
def test_a_quotient_over_a_sum_of_unit_less_sources_is_refused_before_running(
    build: Callable[..., CompileResult],
    decision: DecisionFactory,
    requirement: RequirementFactory,
    source: SourceFactory,
    unit_ref: str | None,
    refused: bool,
) -> None:
    # A sum writes its result under a StatVar of its own, which no reader
    # could name. A side with no unit takes its StatVar as its part of the
    # composed unit, so a sum of unit-less sources has none
    # (`quotient_of_side_unit_unsupported`). A sum in a unit has its unit.
    first = requirement(id="first", measure="urban population", unit=None)
    second = requirement(id="second", measure="rural population", unit=None)
    people = SumOfRequirement(
        id="people",
        decision_ids=["measure"],
        left=RequirementSide(input_id="first"),
        right=RequirementSide(input_id="second"),
    )
    per_head = QuotientRequirement(
        id="per_head",
        decision_ids=["measure"],
        numerator=RequirementSide(input_id="annual_debt_service"),
        denominator=RequirementSide(input_id="people"),
    )
    translation = _translation(
        _retrieve(),
        _retrieve(id="fetch_first", requirement_id="first", spec_id=SPEC_OTHER),
        _retrieve(id="fetch_second", requirement_id="second", spec_id=_SPEC_THIRD),
        SumOfOperation(
            id="people",
            requirement_id="people",
            left=Side(input="fetch_first"),
            right=Side(input="fetch_second"),
        ),
        QuotientOfOperation(
            id="per_head",
            requirement_id="per_head",
            numerator=Side(input="debt"),
            denominator=Side(input="people"),
        ),
        terminal=["per_head"],
    )

    result = build(
        _draft([decision()], [requirement(), first, second, people, per_head]),
        translation,
        discovered={
            "annual_debt_service": source(**_pinned("DebtServicePaid")),
            "first": source(spec_id=SPEC_OTHER, **_pinned("Urban", {"unit_ref": unit_ref})),
            "second": source(spec_id=_SPEC_THIRD, **_pinned("Rural", {"unit_ref": unit_ref})),
        },
    )

    if not refused:
        assert result.ok, result.issues
        return
    assert [(issue.code, issue.operation_id) for issue in result.issues] == [
        (Code.FRAME_UNIT_UNBOUND, "per_head")
    ]


# --- An entity named on a keyed share_of or quotient_of result ---------------------

_KENYA_ON_KEY = {"destination": "country/KEN"}
_UGANDA_ON_KEY = {"destination": "country/UGA"}


def _reading_the_division(kind: str) -> tuple[EvidenceRequirement, Operation]:
    """A calculation that picks Kenya, and for a two-sided one Uganda, out of `divided`."""
    if kind == "filter":
        predicates = (EntityIn(names=("Kenya",), role="recipient"),)
        return (
            FilterRequirement(
                id="picked", decision_ids=["measure"], input_id="divided", predicates=predicates
            ),
            FilterOperation(
                id="picked", requirement_id="picked", input="divided", predicates=predicates
            ),
        )
    sides = (
        RequirementSide(input_id="divided", entity="Kenya"),
        RequirementSide(input_id="divided", entity="Uganda"),
    )
    kenya, uganda = Side(input="divided", entity="Kenya"), Side(input="divided", entity="Uganda")
    if kind == "compare":
        return (
            ComparisonRequirement(
                id="picked", decision_ids=["measure"], minuend=sides[1], subtrahend=sides[0]
            ),
            CompareOperation(id="picked", requirement_id="picked", left=kenya, right=uganda),
        )
    return (
        ShareOfRequirement(
            id="picked", decision_ids=["measure"], numerator=sides[0], denominator=sides[1]
        ),
        ShareOfOperation(id="picked", requirement_id="picked", numerator=kenya, denominator=uganda),
    )


@pytest.mark.parametrize(
    ("kind", "args"),
    [
        ("compare", {"left_entity": _KENYA_ON_KEY, "right_entity": _UGANDA_ON_KEY}),
        ("share_of", {"numerator_entity": _KENYA_ON_KEY, "denominator_entity": _UGANDA_ON_KEY}),
        (
            "filter",
            {
                "predicates": [
                    {"kind": "category_match", "column": "destination", "values": ["country/KEN"]}
                ]
            },
        ),
    ],
)
@pytest.mark.parametrize("verb", ["quotient_of", "share_of"])
def test_an_entity_named_on_a_keyed_division_binds_on_its_key_column(
    build: Callable[..., CompileResult],
    decision: DecisionFactory,
    source: SourceFactory,
    verb: str,
    kind: str,
    args: dict[str, Any],
) -> None:
    # A keyed division rests on two retrievals and holds one series per
    # recipient under its numerator's key column, so a reader naming a
    # recipient resolves it through the numerator's requirement and source.
    requirements, translation, discovered = _per_recipient(
        verb, source, named=("Kenya", "Uganda", "Tanzania")
    )
    picked_req, picked = _reading_the_division(kind)

    result = build(
        _draft([decision()], [*requirements, picked_req]),
        _translation(*translation.operations, picked, terminal=["picked"]),
        discovered=discovered,
    )

    assert result.ok, result.issues
    assert result.workflow is not None
    step = result.workflow.steps[-1]
    assert step.id == "picked"
    assert step.args == args
    assert step.bindings == {
        "frame_ref": Binding(from_step="divided", output=ArtifactKind.FRAME_REF)
    }


def test_an_entity_a_keyed_division_dropped_the_column_of_is_referred(
    build: Callable[..., CompileResult],
    decision: DecisionFactory,
    source: SourceFactory,
) -> None:
    # United States aid per head, keyed by recipient: the result holds one
    # series per recipient and none per donor, so a side cannot pick the
    # donor out of it.
    requirements, translation, discovered = _per_recipient(
        "quotient_of", source, named=("Kenya", "Uganda", "Tanzania")
    )
    aid = requirements[0]
    assert isinstance(aid, DatasetRequirement)
    requirements[0] = aid.model_copy(
        update={"entities": (*aid.entities, Entity(name="United States", role="donor"))}
    )
    discovered["aid"] = discovered["aid"].model_copy(
        update={
            "entities": (
                *discovered["aid"].entities,
                ResolvedEntity(role="origin", ref="country/USA", label="United States"),
            )
        }
    )
    compared = ComparisonRequirement(
        id="picked",
        decision_ids=["measure"],
        minuend=RequirementSide(input_id="divided", entity="Kenya"),
        subtrahend=RequirementSide(input_id="divided", entity="United States"),
    )
    compare = CompareOperation(
        id="picked",
        requirement_id="picked",
        left=Side(input="divided", entity="United States"),
        right=Side(input="divided", entity="Kenya"),
    )

    result = build(
        _draft([decision()], [*requirements, compared]),
        _translation(*translation.operations, compare, terminal=["picked"]),
        discovered=discovered,
    )

    assert [(issue.code, issue.operation_id) for issue in result.issues] == [
        (Code.ENTITY_NOT_ON_FRAME, "picked")
    ]
    assert result.workflow is None


def test_whole_sides_of_two_keyed_quotients_are_entity_required(
    build: Callable[..., CompileResult],
    decision: DecisionFactory,
    source: SourceFactory,
) -> None:
    # Aid per head and people per aid dollar, each keyed over three
    # countries: a comparison naming no entity would read three series a
    # side, and the two are in different composed units besides.
    requirements, translation, discovered = _per_recipient(
        "quotient_of", source, named=("Kenya", "Uganda", "Tanzania")
    )
    fields: dict[str, Any] = {"axis": "country", "denominator_axis": "recipient"}
    inverse_req = QuotientRequirement(
        id="inverse",
        decision_ids=["measure"],
        numerator=RequirementSide(input_id="whole"),
        denominator=RequirementSide(input_id="aid"),
        **fields,
    )
    inverse = QuotientOfOperation(
        id="inverse",
        requirement_id="inverse",
        numerator=Side(input="fetch_whole"),
        denominator=Side(input="fetch_aid"),
        **fields,
    )
    compared = ComparisonRequirement(
        id="picked",
        decision_ids=["measure"],
        minuend=RequirementSide(input_id="inverse"),
        subtrahend=RequirementSide(input_id="divided"),
    )
    compare = CompareOperation(
        id="picked",
        requirement_id="picked",
        left=Side(input="divided"),
        right=Side(input="inverse"),
    )

    result = build(
        _draft([decision()], [*requirements, inverse_req, compared]),
        _translation(*translation.operations, inverse, compare, terminal=["picked"]),
        discovered=discovered,
    )

    assert [(issue.code, issue.operation_id, issue.requirement_id) for issue in result.issues] == [
        (Code.ENTITY_REQUIRED, "picked", "aid"),
        (Code.ENTITY_REQUIRED, "picked", "whole"),
        (Code.FRAME_UNITS_DIFFER, "picked", None),
    ]
    assert result.workflow is None


def test_an_un_keyed_quotient_binds_the_entity_each_side_names(
    build: Callable[..., CompileResult],
    decision: DecisionFactory,
    source: SourceFactory,
) -> None:
    # Kenya's aid over Kenya's population, each source also resolving Uganda
    # and Tanzania: each side's entity is bound on its own source's column.
    requirements, translation, discovered = _per_recipient(
        "quotient_of", source, named=("Kenya",), keyed=False
    )
    per_head = QuotientRequirement(
        id="divided",
        decision_ids=["measure"],
        numerator=RequirementSide(input_id="aid", entity="Kenya"),
        denominator=RequirementSide(input_id="whole", entity="Kenya"),
    )
    divide = QuotientOfOperation(
        id="divided",
        requirement_id="divided",
        numerator=Side(input="fetch_aid", entity="Kenya"),
        denominator=Side(input="fetch_whole", entity="Kenya"),
    )

    result = build(
        _draft([decision()], [*requirements[:2], per_head]),
        _translation(*translation.operations[:2], divide, terminal=["divided"]),
        discovered=discovered,
    )

    assert result.ok, result.issues
    assert result.workflow is not None
    assert result.workflow.steps[-1].args == {
        "numerator_entity": {"destination": "country/KEN"},
        "denominator_entity": {"observationAbout": "country/KEN"},
        "numerator_unit": "USDollar",
    }


# --- A place word keying a side, and what may read a quotient ----------------------


def _over_places(
    verb: str,
    source: SourceFactory,
    *,
    numerator_role: str | None,
    keys: dict[str, Any],
    names: Sequence[str] = ("Kenya", "Uganda", "Tanzania"),
) -> tuple[list[EvidenceRequirement], Translation, dict[str, SelectedSource]]:
    """A measure of `names` divided by `verb` by a second measure of the same places.

    The numerator names its places under `numerator_role`, held on
    `destination`, or with no role, held on `observationAbout`. The
    denominator names them with no role on `observationAbout`: population
    for a `quotient_of`, a total in the numerator's unit for a `share_of`.
    """
    quotient = verb == "quotient_of"
    period = Period(start=2015, end=2024)
    measured = DatasetRequirement(
        id="measured",
        decision_ids=["measure"],
        measure="aid received",
        entities=[Entity(name=name, role=numerator_role) for name in names],
        period=period,
    )
    whole = DatasetRequirement(
        id="whole",
        decision_ids=["measure"],
        measure="population" if quotient else "national income",
        entities=[Entity(name=name) for name in names],
        period=period,
    )
    fields: dict[str, Any] = {"id": "divided", **keys}
    sides = (RequirementSide(input_id="measured"), RequirementSide(input_id="whole"))
    op_sides = (Side(input="fetch_measured"), Side(input="fetch_whole"))
    derived: EvidenceRequirement
    operation: Operation
    if quotient:
        derived = QuotientRequirement(
            decision_ids=["measure"], numerator=sides[0], denominator=sides[1], **fields
        )
        operation = QuotientOfOperation(
            requirement_id="divided", numerator=op_sides[0], denominator=op_sides[1], **fields
        )
    else:
        derived = ShareOfRequirement(
            decision_ids=["measure"], numerator=sides[0], denominator=sides[1], **fields
        )
        operation = ShareOfOperation(
            requirement_id="divided", numerator=op_sides[0], denominator=op_sides[1], **fields
        )
    translation = _translation(
        RetrieveOperation(id="fetch_measured", requirement_id="measured", spec_id=SPEC_KENYA),
        RetrieveOperation(id="fetch_whole", requirement_id="whole", spec_id=SPEC_OTHER),
        operation,
        terminal=["divided"],
    )
    column = "observationAbout" if numerator_role is None else "destination"
    discovered = {
        "measured": source(
            **_pinned("Measured", {"entities": _east_africa(column), "unit_ref": "USDollar"})
        ),
        "whole": source(
            spec_id=SPEC_OTHER,
            **_pinned(
                "Whole",
                {
                    "entities": _east_africa("observationAbout"),
                    "unit_ref": None if quotient else "USDollar",
                },
            ),
        ),
    }
    return [measured, whole, derived], translation, discovered


@pytest.mark.parametrize(
    ("verb", "numerator_role", "names", "keys", "args"),
    [
        pytest.param(
            "quotient_of",
            "recipient",
            ("Kenya", "Uganda", "Tanzania"),
            {"axis": "recipient", "denominator_axis": "country"},
            {
                "numerator_by": ["destination"],
                "denominator_by": {"destination": "observationAbout"},
                "numerator_unit": "USDollar",
            },
            id="quotient-of-a-role-over-places",
        ),
        pytest.param(
            "quotient_of",
            None,
            ("Kenya", "Uganda"),
            {"axis": "country", "denominator_axis": "country"},
            {
                "numerator_by": ["observationAbout"],
                "denominator_by": {"observationAbout": "observationAbout"},
                "numerator_unit": "USDollar",
            },
            id="quotient-of-places-over-places",
        ),
        pytest.param(
            "share_of",
            None,
            ("Kenya", "Uganda"),
            {"axis": "country", "denominator_scope": DenominatorScope.PER_AXIS},
            {
                "numerator_by": ["observationAbout"],
                "denominator_by": {"observationAbout": "observationAbout"},
            },
            id="share-of-places-over-places",
        ),
    ],
)
def test_a_place_word_keys_a_side_by_the_column_its_role_less_places_sit_in(
    build: Callable[..., CompileResult],
    decision: DecisionFactory,
    source: SourceFactory,
    verb: str,
    numerator_role: str | None,
    names: tuple[str, ...],
    keys: dict[str, Any],
    args: dict[str, Any],
) -> None:
    # A place the measure is about states no role, so its column goes by no
    # role phrase. A place word names it on either side.
    requirements, translation, discovered = _over_places(
        verb, source, numerator_role=numerator_role, keys=keys, names=names
    )

    result = build(_draft([decision()], requirements), translation, discovered=discovered)

    assert result.ok, result.issues
    assert result.workflow is not None
    step = result.workflow.steps[-1]
    assert step.tool.name == verb
    assert step.args == args


def test_a_rank_names_the_column_of_role_less_places_by_a_place_word(
    build: Callable[..., CompileResult],
    decision: DecisionFactory,
    source: SourceFactory,
) -> None:
    requirements, translation, discovered = _over_places(
        "quotient_of",
        source,
        numerator_role=None,
        keys={"axis": "country", "denominator_axis": "country"},
    )
    ranked = RankRequirement(
        id="ranked",
        decision_ids=["measure", "population"],
        input_id="divided",
        direction=RankDirection.HIGHEST_FIRST,
        among=RankAmong.POPULATION,
        axis="place",
        reduction=Reduction.MEAN,
    )
    rank = RankOperation(
        id="ranked",
        requirement_id="ranked",
        input="divided",
        direction=RankDirection.HIGHEST_FIRST,
        among=RankAmong.POPULATION,
        axis="place",
        reduction=Reduction.MEAN,
    )

    result = build(
        _draft([decision(), _population_decision(decision)], [*requirements, ranked]),
        _translation(*translation.operations, rank, terminal=["ranked"]),
        discovered=discovered,
    )

    assert result.ok, result.issues
    assert result.workflow is not None
    assert result.workflow.steps[-1].args == {
        "rank_by": ["observationAbout"],
        "ascending": False,
        "aggregation": "mean",
    }


@pytest.mark.parametrize("group_by", [(GroupBy.PERIOD,), (GroupBy.ENTITY,)])
def test_a_sum_over_a_quotient_is_refused_and_a_mean_compiles(
    build: Callable[..., CompileResult],
    decision: DecisionFactory,
    source: SourceFactory,
    group_by: tuple[GroupBy, ...],
) -> None:
    # Dollars per person for three countries add up to no per-person figure.
    # The plan's own sum reads the measured series, since `Plan` refuses one
    # over the quotient; the operation alone reads the quotient.
    requirements, translation, discovered = _over_places(
        "quotient_of",
        source,
        numerator_role="recipient",
        keys={"axis": "recipient", "denominator_axis": "country"},
    )

    def reduced(reduction: Reduction, *, plan_reads: str) -> CompileResult:
        fields: dict[str, Any] = {"id": "reduced", "reduction": reduction, "group_by": group_by}
        requirement = AggregateRequirement(decision_ids=["measure"], input_id=plan_reads, **fields)
        operation = AggregateOperation(requirement_id="reduced", input="divided", **fields)
        return build(
            _draft([decision()], [*requirements, requirement]),
            _translation(*translation.operations, operation, terminal=["reduced"]),
            discovered=discovered,
        )

    summed = reduced(Reduction.SUM, plan_reads="measured")
    mean = reduced(Reduction.MEAN, plan_reads="divided")

    assert (Code.INPUT_SHAPE_UNSUPPORTED, "reduced") in [
        (issue.code, issue.operation_id) for issue in summed.issues
    ]
    assert summed.workflow is None
    assert mean.ok, mean.issues
    assert mean.workflow is not None
    assert mean.workflow.steps[-1].args["aggregation"] == "mean"


@pytest.mark.parametrize("verb", ["compare", "share_of"])
@pytest.mark.parametrize("rates", ["one", "both"])
def test_a_quotient_pairs_with_another_quotient_and_no_measured_side(
    build: Callable[..., CompileResult],
    decision: DecisionFactory,
    requirement: RequirementFactory,
    source: SourceFactory,
    verb: str,
    rates: str,
) -> None:
    # Kenya's debt service per head beside its exports, measured or per head.
    # A composed unit equals no source's own, so the measured pairing is
    # refused on its shape, and never as a share whose units differ, whose
    # repair is the quotient one side already is. The plan's own pairing
    # reads the two measured series, since `Plan` refuses the same sides.
    requirements, translation = _per_head(requirement)
    exports = requirement(id="exports", measure="exports of goods and services")
    exports_per_head = QuotientRequirement(
        id="exports_per_head",
        decision_ids=["measure"],
        numerator=RequirementSide(input_id="exports"),
        denominator=RequirementSide(input_id="population"),
    )
    divide_exports = QuotientOfOperation(
        id="exports_per_head",
        requirement_id="exports_per_head",
        numerator=Side(input="sold"),
        denominator=Side(input="people"),
    )
    plan_sides = (
        ("per_head", "exports_per_head") if rates == "both" else ("annual_debt_service", "exports")
    )
    second = "exports_per_head" if rates == "both" else "sold"
    paired: EvidenceRequirement
    pairing: Operation
    if verb == "compare":
        paired = ComparisonRequirement(
            id="paired",
            decision_ids=["measure"],
            minuend=RequirementSide(input_id=plan_sides[1]),
            subtrahend=RequirementSide(input_id=plan_sides[0]),
        )
        pairing = CompareOperation(
            id="paired",
            requirement_id="paired",
            left=Side(input="per_head"),
            right=Side(input=second),
        )
    else:
        paired = ShareOfRequirement(
            id="paired",
            decision_ids=["measure"],
            numerator=RequirementSide(input_id=plan_sides[0]),
            denominator=RequirementSide(input_id=plan_sides[1]),
        )
        pairing = ShareOfOperation(
            id="paired",
            requirement_id="paired",
            numerator=Side(input="per_head"),
            denominator=Side(input=second),
        )

    result = build(
        _draft([decision()], [*requirements, exports, exports_per_head, paired]),
        _translation(
            *translation.operations,
            _retrieve(id="sold", requirement_id="exports", spec_id=_SPEC_THIRD),
            divide_exports,
            pairing,
            terminal=["paired"] if rates == "both" else ["paired", "exports_per_head"],
        ),
        discovered={
            "annual_debt_service": source(**_pinned("DebtServicePaid")),
            "population": source(spec_id=SPEC_OTHER, **_pinned("Count_Person", {"unit_ref": None})),
            "exports": source(spec_id=_SPEC_THIRD, **_pinned("Exports")),
        },
    )

    if rates == "both":
        assert result.ok, result.issues
    else:
        # The operation also reads other inputs than its requirement states.
        # The unit check reports the pair as two different units for a share
        # as for a comparison.
        assert [(issue.code, issue.operation_id) for issue in result.issues] == [
            (Code.OPERATION_MISMATCH, "paired"),
            (Code.INPUT_SHAPE_UNSUPPORTED, "paired"),
            (Code.FRAME_UNITS_DIFFER, "paired"),
        ]
        assert result.workflow is None


# --- Two keyed sides and the entities they hold in common --------------------------


def _keyed_over(
    verb: str, source: SourceFactory, *, numerator: Sequence[str], denominator: Sequence[str]
) -> tuple[list[EvidenceRequirement], Translation, dict[str, SelectedSource]]:
    """`_per_recipient`, keyed on both sides, each source resolving only the countries its side names."""
    requirements, translation, discovered = _per_recipient(
        verb, source, named=numerator, whole_named=denominator
    )
    for requirement_id, names in (("aid", numerator), ("whole", denominator)):
        held = discovered[requirement_id]
        discovered[requirement_id] = held.model_copy(
            update={"entities": tuple(e for e in held.entities if e.label in names)}
        )
    return requirements, translation, discovered


@pytest.mark.parametrize("verb", ["quotient_of", "share_of"])
def test_two_keyed_sides_holding_no_entity_in_common_are_keys_disjoint(
    build: Callable[..., CompileResult],
    decision: DecisionFactory,
    source: SourceFactory,
    verb: str,
) -> None:
    # Kenya's aid keyed against Uganda's counterpart: no key has an entry on
    # the other side, so the division would pair nothing.
    requirements, translation, discovered = _keyed_over(
        verb, source, numerator=("Kenya",), denominator=("Uganda",)
    )

    result = build(_draft([decision()], requirements), translation, discovered=discovered)

    assert [(issue.code, issue.operation_id) for issue in result.issues] == [
        (Code.KEYS_DISJOINT, "divided")
    ]
    assert result.workflow is None


@pytest.mark.parametrize("verb", ["quotient_of", "share_of"])
def test_two_keyed_sides_holding_one_entity_in_common_compile(
    build: Callable[..., CompileResult],
    decision: DecisionFactory,
    source: SourceFactory,
    verb: str,
) -> None:
    requirements, translation, discovered = _keyed_over(
        verb, source, numerator=("Kenya", "Uganda", "Tanzania"), denominator=("Kenya",)
    )

    result = build(_draft([decision()], requirements), translation, discovered=discovered)

    assert result.ok, result.issues


@pytest.mark.parametrize(
    ("reader", "compiles"),
    [
        pytest.param("sum", False, id="summed"),
        pytest.param("mean", True, id="averaged"),
        pytest.param("quotient_of", False, id="divided-again"),
        pytest.param("compare", False, id="beside-a-measures-change"),
        pytest.param("share_of", False, id="over-a-measures-change"),
        pytest.param("compare_two", True, id="beside-anothers-change"),
    ],
)
def test_an_absolute_change_in_a_quotient_reads_as_the_quotient_does(
    build: Callable[..., CompileResult],
    decision: DecisionFactory,
    source: SourceFactory,
    reader: str,
    compiles: bool,
) -> None:
    # The change in aid per head from 2015 to 2024 is still in dollars per
    # person, so it sums, divides and pairs as the quotient itself does. Where
    # `Plan` refuses the reading, the plan's own requirement reads the change
    # in the plain measure, and the operation alone reads the quotient's.
    requirements, translation, discovered = _over_places(
        "quotient_of",
        source,
        numerator_role="recipient",
        keys={"axis": "recipient", "denominator_axis": "country"},
    )

    def growth(name: str, requirement_input: str, operation_input: str) -> tuple[Any, Any]:
        fields: dict[str, Any] = {
            "id": name,
            "change": GrowthChange.ABSOLUTE,
            "window_start": 2015,
            "window_end": 2024,
        }
        return (
            GrowthRequirement(decision_ids=["measure"], input_id=requirement_input, **fields),
            GrowthOperation(requirement_id=name, input=operation_input, **fields),
        )

    grew_req, grew = growth("grew", "divided", "divided")
    plain_req, plain = growth("plain", "measured", "fetch_measured")
    whole_req, whole = growth("whole_grew", "whole", "fetch_whole")
    plan_reads = "grew" if compiles else "plain"
    kenya, uganda = "Kenya", "Uganda"
    reading_req: EvidenceRequirement
    reading: Operation
    if reader in ("sum", "mean"):
        fields: dict[str, Any] = {
            "id": "reading",
            "reduction": Reduction(reader),
            "group_by": (GroupBy.PERIOD,),
        }
        reading_req = AggregateRequirement(decision_ids=["measure"], input_id=plan_reads, **fields)
        reading = AggregateOperation(requirement_id="reading", input="grew", **fields)
    elif reader == "quotient_of":
        reading_req = QuotientRequirement(
            id="reading",
            decision_ids=["measure"],
            numerator=RequirementSide(input_id=plan_reads, entity=kenya),
            denominator=RequirementSide(input_id="whole_grew", entity=kenya),
        )
        reading = QuotientOfOperation(
            id="reading",
            requirement_id="reading",
            numerator=Side(input="grew", entity=kenya),
            denominator=Side(input="whole_grew", entity=kenya),
        )
    elif reader == "share_of":
        reading_req = ShareOfRequirement(
            id="reading",
            decision_ids=["measure"],
            numerator=RequirementSide(input_id=plan_reads, entity=kenya),
            denominator=RequirementSide(input_id="plain", entity=uganda),
        )
        reading = ShareOfOperation(
            id="reading",
            requirement_id="reading",
            numerator=Side(input="grew", entity=kenya),
            denominator=Side(input="plain", entity=uganda),
        )
    else:
        other = "grew" if reader == "compare_two" else "plain"
        reading_req = ComparisonRequirement(
            id="reading",
            decision_ids=["measure"],
            minuend=RequirementSide(input_id=other, entity=uganda),
            subtrahend=RequirementSide(input_id=plan_reads, entity=kenya),
        )
        reading = CompareOperation(
            id="reading",
            requirement_id="reading",
            left=Side(input="grew", entity=kenya),
            right=Side(input=other, entity=uganda),
        )

    result = build(
        _draft(
            [decision()],
            [*requirements, grew_req, plain_req, whole_req, reading_req],
            answer_mode=AnswerMode.SERIES,
        ),
        _translation(
            *translation.operations,
            grew,
            plain,
            whole,
            reading,
            terminal=["reading", "grew", "plain", "whole_grew"],
        ),
        discovered=discovered,
    )

    if compiles:
        assert result.ok, result.issues
    else:
        assert (Code.INPUT_SHAPE_UNSUPPORTED, "reading") in [
            (issue.code, issue.operation_id) for issue in result.issues
        ]
        assert Code.SHARE_UNITS_DIFFER not in _codes(result)
        assert result.workflow is None


def test_keys_named_apart_still_pair_when_both_sources_hold_every_member(
    build: Callable[..., CompileResult],
    decision: DecisionFactory,
    source: SourceFactory,
) -> None:
    # The numerator's requirement names Kenya and the denominator's Uganda,
    # and each source resolves all three countries: a retrieval writes no
    # scope, so both frames hold all three and every key pairs.
    requirements, translation, discovered = _per_recipient(
        "quotient_of", source, named=("Kenya",), whole_named=("Uganda",)
    )

    result = build(_draft([decision()], requirements), translation, discovered=discovered)

    assert result.ok, result.issues


def test_a_wider_source_on_one_side_still_shares_no_key_with_the_other(
    build: Callable[..., CompileResult],
    decision: DecisionFactory,
    source: SourceFactory,
) -> None:
    # The aid source resolves Tanzania beside the Kenya its requirement
    # names, and the population source Uganda alone: neither of the
    # numerator's two keys has an entry on the other side.
    requirements, translation, discovered = _keyed_over(
        "quotient_of", source, numerator=("Kenya", "Tanzania"), denominator=("Uganda",)
    )
    aid = requirements[0]
    assert isinstance(aid, DatasetRequirement)
    requirements[0] = aid.model_copy(update={"entities": aid.entities[:1]})

    result = build(_draft([decision()], requirements), translation, discovered=discovered)

    assert [(issue.code, issue.operation_id) for issue in result.issues] == [
        (Code.KEYS_DISJOINT, "divided")
    ]


# --- Two sides in composed units ---------------------------------------------------


def _two_per_head_figures(
    decision: DecisionFactory,
    requirement: RequirementFactory,
    source: SourceFactory,
    *,
    verb: str,
    second: tuple[str, str],
    gdp_unit: str,
    grown: bool = False,
) -> tuple[Plan, Translation, dict[str, SelectedSource]]:
    """Kenya's debt service per head paired by `verb` with a second quotient.

    `second` is the numerator and denominator requirement of the second
    quotient, among debt service in `usd`, GDP in `gdp_unit` and a unit-less
    population. `grown` reads each quotient through an absolute growth.
    """
    requirements, translation = _per_head(requirement)
    gdp = requirement(id="gdp", measure="gross domestic product", unit=None)
    reads = {"annual_debt_service": "debt", "population": "people", "gdp": "output"}
    other_req = QuotientRequirement(
        id="other",
        decision_ids=["measure"],
        numerator=RequirementSide(input_id=second[0]),
        denominator=RequirementSide(input_id=second[1]),
    )
    other = QuotientOfOperation(
        id="other",
        requirement_id="other",
        numerator=Side(input=reads[second[0]]),
        denominator=Side(input=reads[second[1]]),
    )
    derived: list[EvidenceRequirement] = [other_req]
    operations: list[Operation] = [
        _retrieve(id="output", requirement_id="gdp", spec_id=_SPEC_THIRD),
        other,
    ]
    sides = ["per_head", "other"]
    if grown:
        change: dict[str, Any] = {
            "window_start": 2015,
            "window_end": 2024,
            "change": GrowthChange.ABSOLUTE,
        }
        for side in sides:
            derived.append(
                GrowthRequirement(
                    id=f"{side}_change", decision_ids=["measure"], input_id=side, **change
                )
            )
            operations.append(
                GrowthOperation(
                    id=f"{side}_change", requirement_id=f"{side}_change", input=side, **change
                )
            )
        sides = [f"{side}_change" for side in sides]
    if verb == "compare":
        derived.append(
            ComparisonRequirement(
                id="paired",
                decision_ids=["measure"],
                minuend=RequirementSide(input_id=sides[1]),
                subtrahend=RequirementSide(input_id=sides[0]),
            )
        )
        operations.append(
            CompareOperation(
                id="paired",
                requirement_id="paired",
                left=Side(input=sides[0]),
                right=Side(input=sides[1]),
            )
        )
    else:
        derived.append(
            ShareOfRequirement(
                id="paired",
                decision_ids=["measure"],
                numerator=RequirementSide(input_id=sides[0]),
                denominator=RequirementSide(input_id=sides[1]),
            )
        )
        operations.append(
            ShareOfOperation(
                id="paired",
                requirement_id="paired",
                numerator=Side(input=sides[0]),
                denominator=Side(input=sides[1]),
            )
        )
    discovered = {
        "annual_debt_service": source(**_pinned("DebtServicePaid")),
        "population": source(spec_id=SPEC_OTHER, **_pinned("Count_Person", {"unit_ref": None})),
        "gdp": source(spec_id=_SPEC_THIRD, **_pinned("GDP", {"unit_ref": gdp_unit})),
    }
    return (
        _draft([decision()], [*requirements, gdp, *derived]),
        _translation(*translation.operations, *operations, terminal=["paired"]),
        discovered,
    )


@pytest.mark.parametrize(
    ("second", "gdp_unit"),
    [
        pytest.param(("gdp", "population"), "Euro", id="another-currency-per-head"),
        pytest.param(("annual_debt_service", "gdp"), "usd", id="another-denominator"),
    ],
)
@pytest.mark.parametrize(
    ("verb", "grown"),
    [("compare", False), ("share_of", False), ("compare", True)],
    ids=["compare", "share_of", "compare-absolute-growths"],
)
def test_two_sides_in_different_composed_units_are_frame_units_differ(
    build: Callable[..., CompileResult],
    decision: DecisionFactory,
    requirement: RequirementFactory,
    source: SourceFactory,
    second: tuple[str, str],
    gdp_unit: str,
    verb: str,
    grown: bool,
) -> None:
    # Dollars per person against euros per person, or against dollars per
    # dollar: analysis-mcp pairs two slices only in one unit. A share over a
    # quotient is never referred to `quotient_of`, which takes no quotient.
    plan, translation, discovered = _two_per_head_figures(
        decision, requirement, source, verb=verb, second=second, gdp_unit=gdp_unit, grown=grown
    )

    result = build(plan, translation, discovered=discovered)

    assert [(issue.code, issue.operation_id) for issue in result.issues] == [
        (Code.FRAME_UNITS_DIFFER, "paired")
    ]
    assert result.workflow is None


@pytest.mark.parametrize("grown", [False, True], ids=["quotients", "absolute-growths"])
def test_two_sides_in_one_composed_unit_compare(
    build: Callable[..., CompileResult],
    decision: DecisionFactory,
    requirement: RequirementFactory,
    source: SourceFactory,
    grown: bool,
) -> None:
    # Debt service per head beside GDP per head, both in dollars per person.
    plan, translation, discovered = _two_per_head_figures(
        decision,
        requirement,
        source,
        verb="compare",
        second=("gdp", "population"),
        gdp_unit="usd",
        grown=grown,
    )

    result = build(plan, translation, discovered=discovered)

    assert result.ok, result.issues
    assert result.workflow is not None
    step = result.workflow.steps[-1]
    assert (step.id, step.tool.name, step.args) == ("paired", "compare", {})


# --- A place dimension whose entities state no role --------------------------------


def _two_places(source: SourceFactory) -> tuple[DatasetRequirement, SelectedSource]:
    """Kenya's and Uganda's population, the two named with no role and held on `observationAbout`."""
    population = DatasetRequirement(
        id="population",
        decision_ids=["measure"],
        measure="population",
        entities=[Entity(name="Kenya"), Entity(name="Uganda")],
        period=Period(start=2015, end=2024),
    )
    held = source(
        **_pinned(
            "Count_Person",
            {"entities": _east_africa("observationAbout")[:2], "unit_ref": None},
        )
    )
    return population, held


def test_an_axis_referral_names_a_role_less_place_column_by_the_place_phrase(
    build: Callable[..., CompileResult],
    decision: DecisionFactory,
    source: SourceFactory,
) -> None:
    # The planner keys on what the referral names, and `Plan` takes
    # "country" for a place dimension and refuses the literal column.
    population, held = _two_places(source)
    gdp = population.model_copy(update={"id": "gdp", "measure": "gross domestic product"})
    per_head = QuotientRequirement(
        id="per_head",
        decision_ids=["measure"],
        numerator=RequirementSide(input_id="gdp"),
        denominator=RequirementSide(input_id="population"),
    )

    result = build(
        _draft([decision()], [gdp, population, per_head]),
        _translation(
            RetrieveOperation(id="output", requirement_id="gdp", spec_id=SPEC_OTHER),
            RetrieveOperation(id="people", requirement_id="population", spec_id=SPEC_KENYA),
            QuotientOfOperation(
                id="per_head",
                requirement_id="per_head",
                numerator=Side(input="output"),
                denominator=Side(input="people"),
            ),
            terminal=["per_head"],
        ),
        discovered={
            "population": held,
            "gdp": held.model_copy(
                update={"spec_id": SPEC_OTHER, "unit_ref": "usd"} | _pinned("GDP", {})
            ),
        },
    )

    (reason,) = [
        issue.reason for issue in result.issues if issue.code is Code.QUOTIENT_AXIS_REQUIRED
    ]
    assert "['country']" in reason
    assert "observationAbout" not in reason


def test_a_filter_names_role_less_places_by_a_place_word(
    build: Callable[..., CompileResult],
    decision: DecisionFactory,
    source: SourceFactory,
) -> None:
    population, held = _two_places(source)
    predicates = (EntityIn(names=("Kenya",), role="country"),)

    result = build(
        _draft(
            [decision()],
            [
                population,
                FilterRequirement(
                    id="kenya",
                    decision_ids=["measure"],
                    input_id="population",
                    predicates=predicates,
                ),
            ],
        ),
        _translation(
            RetrieveOperation(id="people", requirement_id="population", spec_id=SPEC_KENYA),
            FilterOperation(
                id="kenya", requirement_id="kenya", input="people", predicates=predicates
            ),
            terminal=["kenya"],
        ),
        discovered={"population": held},
    )

    assert result.ok, result.issues
    assert result.workflow is not None
    assert result.workflow.steps[-1].args == {
        "predicates": [
            {"kind": "category_match", "column": "observationAbout", "values": ["country/KEN"]}
        ]
    }


def test_a_filter_naming_role_less_places_by_a_place_word_narrows_their_column(
    build: Callable[..., CompileResult],
    decision: DecisionFactory,
    source: SourceFactory,
) -> None:
    # Kenya and Uganda of a source that also resolves Tanzania. The filter
    # names the place column by a place word, and the total over what it
    # kept pools no place the plan left unnamed.
    places = DatasetRequirement(
        id="places",
        decision_ids=["measure"],
        measure="aid received",
        entities=[Entity(name="Kenya"), Entity(name="Uganda")],
        period=Period(start=2015, end=2024),
    )
    predicates = (EntityIn(names=("Kenya", "Uganda"), role="country"),)
    kept_req = FilterRequirement(
        id="kept", decision_ids=["measure"], input_id="places", predicates=predicates
    )
    total_fields: dict[str, Any] = {
        "id": "total",
        "reduction": Reduction.SUM,
        "group_by": (GroupBy.PERIOD,),
    }
    total_req = AggregateRequirement(decision_ids=["measure"], input_id="kept", **total_fields)

    result = build(
        _draft([decision()], [places, kept_req, total_req]),
        _translation(
            RetrieveOperation(id="fetch", requirement_id="places", spec_id=SPEC_KENYA),
            FilterOperation(id="kept", requirement_id="kept", input="fetch", predicates=predicates),
            AggregateOperation(requirement_id="total", input="kept", **total_fields),
            terminal=["total"],
        ),
        discovered={
            "places": source(**_pinned("Measured", {"entities": _east_africa("observationAbout")}))
        },
    )

    assert result.ok, result.issues


# --- A share of a share --------------------------------------------------------------

_SPEC_FOURTH = "spec_" + "d" * 32


@pytest.mark.parametrize(
    ("verb", "left", "right", "published_unit", "issues"),
    [
        pytest.param(
            "share_of",
            "of_gdp",
            "output",
            None,
            [(Code.FRAME_UNIT_UNBOUND, "again", "gdp")],
            id="share-over-a-side-in-dollars",
        ),
        pytest.param("share_of", "of_gdp", "other_share", None, [], id="share-over-another-share"),
        pytest.param(
            "share_of", "of_gdp", "published", PERCENT_UNIT, [], id="share-over-a-published-share"
        ),
        pytest.param(
            "share_of",
            "of_gdp",
            "published",
            None,
            [(Code.FRAME_UNIT_UNBOUND, "again", "published")],
            id="share-over-a-unitless-side",
        ),
        pytest.param(
            "compare",
            "of_gdp",
            "output",
            None,
            [(Code.FRAME_UNIT_UNBOUND, "again", "gdp")],
            id="compared-with-a-side-in-dollars",
        ),
        pytest.param(
            "compare",
            "of_gdp",
            "published",
            PERCENT_UNIT,
            [],
            id="compared-with-a-published-share",
        ),
        pytest.param(
            "compare",
            "share_change",
            "gdp_growth",
            None,
            [(Code.FRAME_UNIT_UNBOUND, "again", None)],
            id="change-in-a-share-compared-with-a-relative-growth",
        ),
        pytest.param(
            "compare", "gdp_growth", "exports_growth", None, [], id="two-relative-growths"
        ),
        pytest.param(
            "compare",
            "exports_growth",
            "gdp_change",
            None,
            [(Code.FRAME_UNIT_UNBOUND, "again", "gdp")],
            id="relative-growth-compared-with-a-change-in-dollars",
        ),
    ],
)
def test_a_share_pairs_only_with_another_side_in_percent(
    build: Callable[..., CompileResult],
    decision: DecisionFactory,
    requirement: RequirementFactory,
    source: SourceFactory,
    verb: str,
    left: str,
    right: str,
    published_unit: str | None,
    issues: list[tuple[Code, str, str | None]],
) -> None:
    # Kenya's debt service as a share of its GDP is in per cent, and so is
    # its change in percentage points. It divides by, or compares with,
    # another share or a published series in per cent as it stands. Against
    # GDP in dollars, a unit-less count or a relative growth analysis-mcp
    # refuses the units, and `quotient_of` takes no ratio, so no operation
    # computes the pair; the issue names the side a source in per cent
    # would clear. A relative growth has no unit: it pairs with another and
    # not with a change in dollars.
    requirement_of = {"output": "gdp"}
    gdp = requirement(id="gdp", measure="gross domestic product")
    exports = requirement(id="exports", measure="exports of goods and services")
    published = requirement(id="published", measure="debt service as a share of exports")
    window = {"window_start": 2015, "window_end": 2023}
    growths = {
        "share_change": ("of_gdp", "of_gdp", GrowthChange.ABSOLUTE),
        "gdp_growth": ("gdp", "output", GrowthChange.RELATIVE),
        "gdp_change": ("gdp", "output", GrowthChange.ABSOLUTE),
        "exports_growth": ("exports", "sold", GrowthChange.RELATIVE),
    }
    used = {left, right}

    def share(share_id: str, numerator: str, denominator: str) -> EvidenceRequirement:
        return ShareOfRequirement(
            id=share_id,
            decision_ids=["measure"],
            numerator=RequirementSide(input_id=numerator),
            denominator=RequirementSide(input_id=denominator),
        )

    def divide(share_id: str, numerator: str, denominator: str) -> Operation:
        return ShareOfOperation(
            id=share_id,
            requirement_id=share_id,
            numerator=Side(input=numerator),
            denominator=Side(input=denominator),
        )

    left_req, right_req = (requirement_of.get(side, side) for side in (left, right))
    result = build(
        _draft(
            [decision()],
            [
                requirement(),
                gdp,
                exports,
                *([published] if "published" in used else []),
                share("of_gdp", "annual_debt_service", "gdp"),
                share("other_share", "exports", "gdp"),
                *(
                    GrowthRequirement(
                        id=growth_id,
                        decision_ids=["measure"],
                        input_id=input_req,
                        change=change,
                        **window,
                    )
                    for growth_id, (input_req, _, change) in growths.items()
                    if growth_id in used
                ),
                (
                    share("again", left_req, right_req)
                    if verb == "share_of"
                    else ComparisonRequirement(
                        id="again",
                        decision_ids=["measure"],
                        minuend=RequirementSide(input_id=right_req),
                        subtrahend=RequirementSide(input_id=left_req),
                    )
                ),
            ],
        ),
        _translation(
            _retrieve(),
            _retrieve(id="output", requirement_id="gdp", spec_id=SPEC_OTHER),
            _retrieve(id="sold", requirement_id="exports", spec_id=_SPEC_THIRD),
            *(
                [_retrieve(id="published", requirement_id="published", spec_id=_SPEC_FOURTH)]
                if "published" in used
                else []
            ),
            divide("of_gdp", "debt", "output"),
            divide("other_share", "sold", "output"),
            *(
                GrowthOperation(
                    id=growth_id,
                    requirement_id=growth_id,
                    input=input_op,
                    change=change,
                    **window,
                )
                for growth_id, (_, input_op, change) in growths.items()
                if growth_id in used
            ),
            (
                divide("again", left, right)
                if verb == "share_of"
                else CompareOperation(
                    id="again",
                    requirement_id="again",
                    left=Side(input=left),
                    right=Side(input=right),
                )
            ),
            terminal=["again", "other_share", "of_gdp"],
        ),
        discovered={
            "annual_debt_service": source(**_pinned("DebtServicePaid")),
            "gdp": source(spec_id=SPEC_OTHER, **_pinned("GDP")),
            "exports": source(spec_id=_SPEC_THIRD, **_pinned("Exports")),
            **(
                {
                    "published": source(
                        spec_id=_SPEC_FOURTH,
                        **_pinned("DebtServiceShareOfExports", {"unit_ref": published_unit}),
                    )
                }
                if "published" in used
                else {}
            ),
        },
    )

    assert [
        (issue.code, issue.operation_id, issue.requirement_id) for issue in result.issues
    ] == issues
    assert result.ok == (not issues)
