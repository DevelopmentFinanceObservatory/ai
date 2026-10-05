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
from collections.abc import Callable, Sequence
from itertools import permutations
from typing import Any

import pytest
from compiler.capabilities import CAPABILITIES, ToolCapability
from compiler.compile import CompileIssueCode, CompileResult, compile_plan
from compiler.sources import source_from_spec_block
from compiler.surface import ToolSchema
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
    RankOperation,
    Reduction,
    RetrieveOperation,
    ShareOfOperation,
    Side,
    Translation,
)
from planner.plan import (
    AggregateRequirement,
    AnswerMode,
    ComparisonRequirement,
    DatasetRequirement,
    Decision,
    DecisionFixes,
    Entity,
    EvidenceRequirement,
    FilterRequirement,
    GrowthRequirement,
    JoinRequirement,
    Plan,
    PlanStatus,
    RankRequirement,
    RequirementSide,
    ShareOfRequirement,
    Unit,
)
from planner.records import (
    DenominatorScope,
    EntityIn,
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
from planner.workflow import (
    ArtifactKind,
    Binding,
    OpenEntityAxis,
    ReadingSetBy,
    ResolvedEntity,
    SelectedSource,
)

SPEC_KENYA = "spec_" + "a" * 32
SPEC_OTHER = "spec_" + "b" * 32

DecisionFactory = Callable[..., Decision]
RequirementFactory = Callable[..., EvidenceRequirement]
DerivedRequirementFactory = Callable[..., EvidenceRequirement]
SourceFactory = Callable[..., SelectedSource]

Code = CompileIssueCode


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
    assert retrieve.args == {
        "spec_id": SPEC_KENYA,
        "resolved_stat_var_dcids": ["DebtServicePaid"],
    }
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
        left=RequirementSide(input_id="annual_debt_service", entity="Kenya"),
        right=RequirementSide(input_id="annual_debt_service", entity="Ghana"),
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
        left=RequirementSide(input_id="annual_debt_service"),
        right=RequirementSide(input_id="annual_gdp"),
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
    # The exact shape `SHARE_AXIS_REQUIRED` refuses above, this time keyed:
    # naming `axis="country"` says which of the three members the
    # numerator answers for, so `_check_share_of_needs_axis` never runs at
    # all. `_check_side`'s own `ENTITY_REQUIRED` reads none of this --
    # it refuses any whole side of a requirement naming several entities
    # whether or not an axis disambiguates them, so it still fires here
    # and the plan does not fully compile; what this test pins is
    # narrower and unaffected by it: this guard's own code never appears
    # among the issues once axis says which member.
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

    assert Code.SHARE_AXIS_REQUIRED not in _codes(result)


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
            left=RequirementSide(input_id="annual_debt_service"),
            right=RequirementSide(input_id="annual_gdp"),
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
        left=RequirementSide(input_id="annual_debt_service", entity="Kenya"),
        right=RequirementSide(input_id="annual_debt_service", entity="Ghana"),
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


def test_a_terminal_another_operation_reads_is_reported(
    build: Callable[..., CompileResult],
    decision: DecisionFactory,
    requirement: RequirementFactory,
    derived_requirement: DerivedRequirementFactory,
) -> None:
    # `positive` is named as an answer and read by `totals`: both the answer
    # and an intermediate, so its frame would leak into the run's results.
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

    assert _codes(result) == [Code.TERMINAL_IS_READ]
    assert result.issues[0].operation_id == "positive"
    assert "'totals'" in result.issues[0].reason


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
    assert step.args == {"spec_id": SPEC_KENYA, "resolved_stat_var_dcids": ["DebtServicePaid"]}


@pytest.mark.parametrize(
    ("stat_vars", "expected"),
    [
        (
            ("Count_Person", "Count_Person_Female"),
            {
                "spec_id": SPEC_KENYA,
                "stat_vars": "separate",
                "resolved_stat_var_dcids": ["Count_Person", "Count_Person_Female"],
            },
        ),
        (("Count_Person",), {"spec_id": SPEC_KENYA, "resolved_stat_var_dcids": ["Count_Person"]}),
        ((), {"spec_id": SPEC_KENYA}),
    ],
    ids=["a-family-elected", "one-elected", "none-elected"],
)
def test_a_retrieval_posts_the_statvars_its_source_elected(
    build: Callable[..., CompileResult],
    decision: DecisionFactory,
    requirement: RequirementFactory,
    source: SourceFactory,
    stat_vars: tuple[str, ...],
    expected: dict[str, object],
) -> None:
    # The source's elected StatVars ride on the retrieval, in the source's
    # order, as `get_data`'s `resolved_stat_var_dcids`, beside the reading a
    # family states. A source that elected none leaves the key out, so
    # `get_data` resolves the family alone.
    plan = _draft([decision()], [requirement()], answer_mode=AnswerMode.SERIES)

    result = build(
        plan,
        _translation(_retrieve(), terminal=["debt"]),
        discovered={"annual_debt_service": source(stat_vars=stat_vars)},
    )

    assert result.issues == ()
    assert result.workflow is not None
    (step,) = result.workflow.steps
    assert step.args == expected


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
        left=RequirementSide(input_id="annual_debt_service"),
        right=RequirementSide(input_id="annual_gdp"),
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
        left=RequirementSide(input_id="annual_debt_service"),
        right=RequirementSide(input_id="annual_gdp"),
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


def test_a_whole_side_reading_a_filter_that_narrowed_the_source_is_not_reported(
    build: Callable[..., CompileResult],
    decision: DecisionFactory,
    requirement: RequirementFactory,
    source: SourceFactory,
    wide_source: SourceFactory,
) -> None:
    # The filter keeps Kenya's rows, so the side pools nothing: what it reads
    # is the filtered frame, not the two-country frame the retrieval produced.
    kenya_only = (EntityIn(names=("Kenya",)),)
    filtered = FilterRequirement(
        id="kenya_only",
        decision_ids=["measure"],
        input_id="annual_debt_service",
        predicates=kenya_only,
    )
    gdp = requirement(id="annual_gdp", measure="gross domestic product")
    share = ShareOfRequirement(
        id="share",
        decision_ids=["measure"],
        numerator=RequirementSide(input_id="kenya_only"),
        denominator=RequirementSide(input_id="annual_gdp"),
    )
    plan = _draft([decision()], [requirement(), filtered, gdp, share])
    filter_op = FilterOperation(
        id="kenya_only", requirement_id="kenya_only", input="debt", predicates=kenya_only
    )
    share_op = ShareOfOperation(
        id="share",
        requirement_id="share",
        numerator=Side(input="kenya_only"),
        denominator=Side(input="gdp"),
    )
    retrieve_gdp = _retrieve(id="gdp", requirement_id="annual_gdp", spec_id=SPEC_OTHER)

    result = build(
        plan,
        _translation(_retrieve(), retrieve_gdp, filter_op, share_op, terminal=["share"]),
        discovered={
            "annual_debt_service": wide_source(),
            "annual_gdp": source(spec_id=SPEC_OTHER),
        },
    )

    assert result.ok, result.issues


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
        left=RequirementSide(input_id="from_2018"),
        right=RequirementSide(input_id="just_period"),
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


def test_constant_prices_are_unmeetable_by_any_operation(
    build: Callable[..., CompileResult],
    decision: DecisionFactory,
    requirement: RequirementFactory,
    derived_requirement: DerivedRequirementFactory,
) -> None:
    # No operation restates a series in another year's prices, so this is
    # refused at compile time with the requirement named, rather than left
    # for a step to claim.
    plan = _draft(
        [decision()],
        [
            requirement(unit=Unit(currency="USD", prices="constant", base_year=2015)),
            derived_requirement(),
        ],
    )

    result = build(plan, _translation(_retrieve(), _aggregate()))

    assert Code.REQUIREMENT_UNMEETABLE in _codes(result)
    unmeetable = next(i for i in result.issues if i.code is Code.REQUIREMENT_UNMEETABLE)
    assert unmeetable.requirement_id == "annual_debt_service"


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
        left=RequirementSide(input_id="annual_debt_service", entity="Kenya"),
        right=RequirementSide(input_id="annual_debt_service", entity="Ghana"),
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
        left=RequirementSide(input_id="annual_debt_service"),
        right=RequirementSide(input_id="annual_debt_service", entity="Ghana"),
    )

    result = build(
        _two_country_plan(decision, requirement, comparison),
        _translation(_retrieve(), gap, terminal=["gap"]),
        discovered={"annual_debt_service": wide_source()},
    )

    assert _codes(result) == [Code.ENTITY_REQUIRED]


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
        left=RequirementSide(input_id="annual_debt_service", entity="Kenya"),
        right=RequirementSide(input_id="annual_debt_service", entity="Ghana"),
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
        left=RequirementSide(input_id="by_entity_a", entity="Kenya"),
        right=RequirementSide(input_id="by_entity_b", entity="Ghana"),
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
        left=RequirementSide(input_id="annual_debt_service"),
        right=RequirementSide(input_id="old_gdp"),
    )
    plan = _draft([decision()], [requirement(), old, comparison])
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

    assert _codes(result) == [Code.PERIODS_DISJOINT]


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
    filter_req = FilterRequirement(
        id="filtered",
        decision_ids=["measure"],
        input_id="annual_debt_service",
        predicates=(EntityIn(names=("Uganda",)),),
    )
    plan = _draft([decision()], [requirement(), filter_req])
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
        "across": "country",
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
    input_id: str, *, reads: str = "fetch", filter_id: str = "countries"
) -> tuple[FilterRequirement, FilterOperation]:
    """A filter keeping the countries of `input_id`, a requirement that names no place."""
    predicates = (PlaceLevel(level=PlaceType.COUNTRY),)
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
    # `observationAbout`. An identity that names no place states no root,
    # and `search` counts the binding as at least one StatVar, so the
    # retrieval reads it as a family: separately, since it varies along no
    # axis to sum over. The place axis holds every level of place, so a
    # filter keeps the countries first: its retrieval asks for place types,
    # and the filter reads them.
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
    assert open_place.pinned is False
    assert retrieve.args == {
        "spec_id": open_place.spec_id,
        "stat_vars": "separate",
        "resolved_stat_var_dcids": list(open_place.stat_vars),
        "place_types": True,
    }
    assert len(open_place.stat_vars) == 1
    assert retrieve.expected_role_keys == _frame_role_columns(["observationAbout"])
    assert keep.args == {
        "predicates": [
            {"kind": "place_level", "column": "observationAbout_type", "level": "Country"}
        ]
    }
    assert rank.args == {"rank_by": ["observationAbout"], "ascending": False, "aggregation": "mean"}
    assert rank.expected_role_keys == _frame_role_columns(["observationAbout"])


def test_a_series_of_every_place_asks_for_no_place_types(
    build: Callable[..., CompileResult],
    decision: DecisionFactory,
    place_scope_block: dict[str, Any],
) -> None:
    # No place-level filter reads the retrieval, so its frame carries no type
    # column.
    open_place = source_from_spec_block(place_scope_block)
    plan = _draft([decision()], [_every_country()], answer_mode=AnswerMode.SERIES)
    fetch = RetrieveOperation(id="fetch", requirement_id="health", spec_id=open_place.spec_id)

    result = build(plan, _translation(fetch, terminal=["fetch"]), discovered={"health": open_place})

    assert result.ok, result.issues
    assert result.workflow is not None
    assert "place_types" not in result.workflow.steps[0].args


def _kenya_from_every_donor(party: str = "donor") -> DatasetRequirement:
    """Education aid to Kenya from every member of `party`, 2015 to 2023."""
    return DatasetRequirement(
        id="aid",
        decision_ids=["measure"],
        measure="education aid",
        entities=[Entity(name="Kenya", role="recipient")],
        across=party,
        period=Period(start=2015, end=2023),
    )


# The constraint properties the recorded education-aid family varies along,
# each a slot the recorded block states summable.
_DONOR_FAMILY_AXES = ("crsCooperationModality", "crsPurpose", "crsTypeOfFinance")


def _donor_family_stating_no_summable(role_open_block: dict[str, Any]) -> SelectedSource:
    """The recorded education-aid block as a dfo-apis that predates `Slot.summable` serves it.

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
        "resolved_stat_var_dcids": list(kenya_from_every_donor.stat_vars),
    }
    assert kenya_from_every_donor.stat_vars
    assert result.workflow.sources[0].stat_var_reading_set_by is ReadingSetBy.RULE
    assert retrieve.expected_role_keys == _frame_role_columns(["destination", "origin"])
    assert rank.args == {"rank_by": ["origin"], "ascending": False, "aggregation": "mean"}
    assert rank.expected_role_keys == _frame_role_columns(["origin"])
    assert "entity_role_unverified" not in [w.code.value for w in result.warnings]


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
        [_every_country(across=population), rank_req],
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
    # The recorded block names two StatVars of its family. Served with no
    # `summable` on its slots, no axis may be summed, so the rule reads them
    # separately and the frame carries one measure per StatVar. A rank over
    # the open `origin` axis reads one.
    result = _rank_donors(build, decision, _donor_family_stating_no_summable(role_open_block))

    (issue,) = result.issues
    assert issue.code is Code.INPUT_SHAPE_UNSUPPORTED
    assert issue.operation_id == "ranked"
    assert "rank reads a single measure; got 2" in issue.reason
    assert "cannot be summed over " + ", ".join(_DONOR_FAMILY_AXES) in issue.reason
    assert result.workflow is None


def test_a_rank_over_a_donor_family_the_plan_reads_summed_asks_for_the_sum(
    build: Callable[..., CompileResult],
    decision: DecisionFactory,
    role_open_block: dict[str, Any],
) -> None:
    # The plan states `stat_vars='sum'` over the same recorded block, so the
    # retrieval asks dfo-apis for the sum and the rank reads one measure over
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
        "resolved_stat_var_dcids": list(source_from_spec_block(role_open_block).stat_vars),
    }
    assert result.workflow.sources[0].stat_var_reading_set_by is ReadingSetBy.PLAN
    assert rank.args == {"rank_by": ["origin"], "ascending": False, "aggregation": "mean"}
    assert rank.expected_role_keys == _frame_role_columns(["origin"])


def test_grouping_by_entity_over_an_open_axis_writes_its_column(
    build: Callable[..., CompileResult],
    decision: DecisionFactory,
    place_scope_block: dict[str, Any],
) -> None:
    open_place = source_from_spec_block(place_scope_block)
    by_country = AggregateRequirement(
        id="by_country",
        decision_ids=["measure"],
        input_id="health",
        reduction=Reduction.MEAN,
        group_by=(GroupBy.ENTITY,),
    )
    plan = _draft([decision()], [_every_country(), by_country])

    result = build(
        plan,
        _translation(
            RetrieveOperation(id="fetch", requirement_id="health", spec_id=open_place.spec_id),
            _aggregate(
                id="by_country",
                requirement_id="by_country",
                input="fetch",
                reduction=Reduction.MEAN,
                group_by=(GroupBy.ENTITY,),
            ),
            terminal=["by_country"],
        ),
        discovered={"health": open_place},
    )

    assert result.ok, result.issues
    assert result.workflow is not None
    assert result.workflow.steps[-1].args == {
        "aggregation": "mean",
        "group_by": ["observationAbout"],
    }


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
        assert "instead of stating `across`" in result.issues[0].reason


def test_an_axis_less_share_of_over_an_open_axis_is_share_axis_required(
    build: Callable[..., CompileResult],
    decision: DecisionFactory,
    place_scope_block: dict[str, Any],
    source: SourceFactory,
) -> None:
    # Every country's health spending over every country's population, with
    # no axis: each side pools 193 members, so the quotient answers for
    # nobody. Each side keeps its countries first, since a share over every
    # level of place is refused before this check.
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
    plan = _draft([decision()], [_every_country(), population, spenders_req, peoples_req, share])
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

    assert Code.SHARE_AXIS_REQUIRED in _codes(result)
    assert result.workflow is None


def test_a_whole_side_over_an_open_axis_is_entity_required(
    build: Callable[..., CompileResult],
    decision: DecisionFactory,
    requirement: RequirementFactory,
    place_scope_block: dict[str, Any],
    source: SourceFactory,
) -> None:
    # Kenya against "every country" as one side: the side names none of the
    # 193 countries, so it cannot say which of them it reads.
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
        left=RequirementSide(input_id="kenya_health"),
        right=RequirementSide(input_id="health"),
    )
    plan = _draft([decision()], [kenya, _every_country(), gap])
    compare = CompareOperation(
        id="gap", requirement_id="gap", left=Side(input="kenya"), right=Side(input="fetch")
    )

    result = build(
        plan,
        _translation(
            RetrieveOperation(id="kenya", requirement_id="kenya_health", spec_id=SPEC_KENYA),
            RetrieveOperation(id="fetch", requirement_id="health", spec_id=open_place.spec_id),
            compare,
            terminal=["gap"],
        ),
        discovered={
            "kenya_health": source(
                entities=(
                    ResolvedEntity(role="observationAbout", ref="country/KEN", label="Kenya"),
                ),
            ),
            "health": open_place,
        },
    )

    assert _codes(result) == [Code.ENTITY_REQUIRED]
    (issue,) = result.issues
    assert issue.requirement_id == "health"
    assert "covers every 'country' and names none of them" in issue.reason
    assert "name them in the requirement's `entities`" in issue.reason


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
        across="donor",
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
) -> CompileResult:
    """A rank over every donor, compiled against a source that also leaves `destination` open."""
    every_donor = DatasetRequirement(
        id="aid",
        decision_ids=["measure"],
        measure="education aid",
        across="donor",
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
    assert "'destination', 'origin' open" in result.issues[0].reason
    assert "name the members of the other" in result.issues[0].reason


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


def test_a_whole_side_over_an_open_axis_beside_a_named_entity_is_entity_required(
    build: Callable[..., CompileResult],
    decision: DecisionFactory,
    requirement: RequirementFactory,
    source: SourceFactory,
) -> None:
    # Kenya's aid from every donor as one side names none of the donors, as
    # every donor's aid with nobody named does.
    kenya_destination = ResolvedEntity(role="destination", ref="country/KEN", label="Kenya")
    uganda_destination = ResolvedEntity(role="destination", ref="country/UGA", label="Uganda")
    uganda = requirement(
        id="uganda_aid",
        measure="education aid",
        entities=[Entity(name="Uganda", role="recipient")],
        unit=None,
    )
    gap = ComparisonRequirement(
        id="gap",
        decision_ids=["measure"],
        left=RequirementSide(input_id="aid"),
        right=RequirementSide(input_id="uganda_aid"),
    )
    plan = _draft([decision()], [_kenya_from_every_donor(), uganda, gap])

    result = build(
        plan,
        _translation(
            RetrieveOperation(id="fetch", requirement_id="aid", spec_id=SPEC_KENYA),
            RetrieveOperation(id="uganda", requirement_id="uganda_aid", spec_id=SPEC_OTHER),
            CompareOperation(
                id="gap", requirement_id="gap", left=Side(input="fetch"), right=Side(input="uganda")
            ),
            terminal=["gap"],
        ),
        discovered={
            "aid": source(
                entities=(kenya_destination,),
                open_entity_axes=(OpenEntityAxis(column="origin", member_count=55),),
                coverage_window=Period(start=2015, end=2023),
            ),
            "uganda_aid": source(spec_id=SPEC_OTHER, entities=(uganda_destination,)),
        },
    )

    assert _codes(result) == [Code.ENTITY_REQUIRED]
    (issue,) = result.issues
    assert issue.requirement_id == "aid"
    assert "covers every 'donor' and names none of them" in issue.reason


def test_a_keyed_share_over_an_open_axis_is_entity_required(
    build: Callable[..., CompileResult],
    decision: DecisionFactory,
    source: SourceFactory,
) -> None:
    # Each donor's education aid to Kenya over its total aid to Kenya, keyed
    # by donor: a keyed share divides member by member and pools nothing, and
    # each side still names none of the donors it reads.
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

    assert _codes(result) == [Code.ENTITY_REQUIRED, Code.ENTITY_REQUIRED]
    for issue in result.issues:
        assert "covers every 'donor' and names none of them" in issue.reason
        assert "so the side cannot say which it reads" in issue.reason
