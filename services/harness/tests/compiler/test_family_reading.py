"""How a source binding a family of StatVars is read, summed or separately.

The spec blocks come from `tests/fixtures/family_spec_blocks.json`: single-part
`search` envelopes recorded from apis, cut to the fields a source reads
and converted to the wire that lists no members. Each slot carries `summable`
as apis serves it. A test that needs a slot the recordings lack edits a
copy of a recorded block and says what it changed.

A source is one StatVar exactly when `search` states it serves one
(`resolved.stat_vars.served_count`). The three blocks that elected their
family's root, keyed on a place, state a served count of one; the CRS
families state their served count, or none where the recorded walk was
capped.

The compile-level tests state the plans that whole questions call for and
check the retrieval's `stat_vars` argument, the aggregate by period that
totals a summed flow across the recipients it leaves open, and the
compile-time refusal of a single-measure verb over a family read separately,
and a breakdown by sector: the retrieval's `breakdown`, and the aggregate and
rank that read its `stat_var_crsPurpose` column.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import pytest
from compiler.capabilities import ToolCapability
from compiler.compile import CompileIssueCode, CompileResult, compile_plan
from compiler.sources import (
    default_reading,
    is_family,
    outcome_from_envelope,
    source_from_spec_block,
    stat_var_reading,
)
from compiler.surface import ToolSchema
from harness.plan_display import _reading_clause
from planner.operations import (
    AggregateOperation,
    CompareOperation,
    FilterOperation,
    GroupBy,
    Operation,
    RankOperation,
    Reduction,
    RetrieveOperation,
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
    Plan,
    PlanStatus,
    RankRequirement,
    RequirementSide,
    ResolvedBy,
)
from planner.records import (
    Breakdown,
    BreakdownDimension,
    Period,
    PeriodThreshold,
    PlaceLevel,
    PlaceType,
    RankAmong,
    RankDirection,
    StatVarReading,
)
from planner.workflow import OpenAxis, ReadingSetBy, SelectedSource

_BLOCKS: dict[str, Any] = json.loads(
    (Path(__file__).parent.parent / "fixtures" / "family_spec_blocks.json").read_text()
)


def _block(name: str) -> dict[str, Any]:
    """A deep copy of one recorded spec block, safe to edit."""
    return copy.deepcopy(_BLOCKS[name]["parts"][0]["spec"])


def _retrieval(source: SelectedSource, stat_vars: str | None = None) -> dict[str, Any]:
    """The `get_data` arguments a retrieval of `source` writes: its spec_id, and the
    reading when it states one."""
    args: dict[str, Any] = {"spec_id": source.spec_id}
    if stat_vars is not None:
        args["stat_vars"] = stat_vars
    return args


def _source(name: str) -> SelectedSource:
    return source_from_spec_block(_block(name))


def _one_confirmed(name: str) -> dict[str, Any]:
    """A recorded block whose `resolved.stat_vars` confirms one member and settles no
    membership, as `search` states it when one member of the binding has data for the
    roles asked and its walk did not finish."""
    block = _block(name)
    block["resolved"]["stat_vars"].update(member_count=1, served_count=None)
    return block


def _slot(block: dict[str, Any], prop: str) -> dict[str, Any]:
    return next(
        slot
        for slot in block["resolved"]["slots"]
        if (slot["key"].get("property") or {}).get("ref") == prop
    )


# --- The rule over recorded blocks ----------------------------------------------


def test_every_recorded_envelope_reads_into_sources() -> None:
    for name, envelope in _BLOCKS.items():
        outcome = outcome_from_envelope(envelope)
        assert outcome.sources, name


def test_french_education_grants_sum_over_three_summable_open_axes() -> None:
    """crsPurpose and crsTypeOfFinance are bound to one hierarchical code each
    (education, grants), which admitted its descendants, so both are open; the
    modality is unbound. The flow type and financial flow type are each bound
    to one flat value, so closed, and the place and time slots name no member."""
    source = _source("france_education_grants")

    assert is_family(source)
    assert source.open_axes == (
        OpenAxis(property="crsCooperationModality", summable=True),
        OpenAxis(property="crsPurpose", summable=True),
        OpenAxis(property="crsTypeOfFinance", summable=True),
    )
    assert default_reading(source) is StatVarReading.SUM


def test_a_family_whose_membership_search_did_not_settle_is_still_a_family() -> None:
    # 1,554 members confirmed, and the walk capped: no served count.
    source = _source("germany_bilateral_oda")

    assert source.served_count is None
    assert source.stat_var_member_count == 1554
    assert is_family(source)
    assert default_reading(source) is StatVarReading.SUM


def test_an_inapplicable_slot_is_no_axis() -> None:
    # dac2aAidType is unbound but listed in `inapplicable_slots`.
    source = _source("germany_oda_to_kenya")

    assert "dac2aAidType" not in {axis.property for axis in source.open_axes}
    assert default_reading(source) is StatVarReading.SUM


def test_male_and_female_population_stays_separate_over_its_set_bound_sex() -> None:
    """A non-CRS family: `gender` is bound to the set {Male, Female} and no code
    system declares it summable. Every other slot is inapplicable."""
    source = _source("male_female_population_nigeria")

    assert is_family(source)
    assert source.open_axes == (OpenAxis(property="gender", summable=False),)
    assert default_reading(source) is StatVarReading.SEPARATE


@pytest.mark.parametrize(
    "name", ["life_expectancy_ghana", "population_kenya", "unemployment_rate_by_sex_spain"]
)
def test_one_served_stat_var_is_no_family_however_many_candidates_search_counted(
    name: str,
) -> None:
    """Each serves one StatVar, the family's root. `member_count` counts the three
    candidates confirmed at the place, and does not make the source a family.
    Every slot of these is inapplicable, so a rule reading the count as a family
    would sum each over no axis at all."""
    for source in outcome_from_envelope(_BLOCKS[name]).sources:
        assert (source.served_count, source.stat_var_member_count) == (1, 3)
        assert not is_family(source)
        assert stat_var_reading(source, None).stat_var_reading is None


def test_every_recorded_source_is_a_family_unless_search_serves_one_stat_var() -> None:
    families = {
        name: [is_family(source) for source in outcome_from_envelope(envelope).sources]
        for name, envelope in _BLOCKS.items()
    }

    assert families == {
        "france_education_grants": [True],
        "germany_bilateral_oda": [True],
        "germany_oda_to_kenya": [True],
        "male_female_population_nigeria": [True],
        "life_expectancy_ghana": [False],
        "population_kenya": [False],
        "unemployment_rate_by_sex_spain": [False, False],
    }


# --- A family `search` confirmed one member of -----------------------------------------


def test_a_family_confirming_one_member_without_a_served_count_is_summed() -> None:
    """France's education grants with one member confirmed and no membership
    settled: the mint may serve more, and the open axes, all summable, read it
    summed."""
    source = source_from_spec_block(_one_confirmed("france_education_grants"))

    assert is_family(source)
    read = stat_var_reading(source, None)
    assert read.stat_var_reading is StatVarReading.SUM
    assert read.stat_var_reading_set_by is ReadingSetBy.RULE


def test_a_family_confirming_one_member_stays_separate_over_an_unsummable_axis() -> None:
    # The recording binds nettingBasis to Gross; unbound, it pairs gross and net.
    block = _one_confirmed("germany_oda_to_kenya")
    _slot(block, "nettingBasis")["binding"] = {"kind": "unbound"}

    source = source_from_spec_block(block)

    assert is_family(source)
    assert stat_var_reading(source, None).stat_var_reading is StatVarReading.SEPARATE


def test_a_source_carrying_no_identity_is_a_family_by_its_served_count() -> None:
    block = _block("france_education_grants")
    del block["definition"]

    source = source_from_spec_block(block)

    assert source.identity is None and is_family(source)
    assert not is_family(source.model_copy(update={"served_count": 1}))


# --- A binding `search` states one served member of -----------------------------------

_PLACE_OPEN_BLOCK: dict[str, Any] = json.loads(
    (Path(__file__).parent.parent / "fixtures" / "search_spec_place_scope.json").read_text()
)["spec"]
_DONOR_OPEN_BLOCK: dict[str, Any] = json.loads(
    (Path(__file__).parent.parent / "fixtures" / "search_spec_role_open.json").read_text()
)["spec"]


def _place_open(**stat_vars: object) -> dict[str, Any]:
    """The recorded place-open block, confirming one member as a floor, with no
    membership settled unless `stat_vars` overrides it."""
    block = copy.deepcopy(_PLACE_OPEN_BLOCK)
    block["resolved"]["stat_vars"].update({"served_count": None} | stat_vars)
    return block


def _donor_open(*, confirmed: int, served: int | None) -> dict[str, Any]:
    block = copy.deepcopy(_DONOR_OPEN_BLOCK)
    block["resolved"]["stat_vars"].update(member_count=confirmed, served_count=served)
    return block


def _stating_no_summable(block: dict[str, Any]) -> dict[str, Any]:
    """`block` as an apis service that predates `Slot.summable` serves it: no slot states one."""
    for slot in block["resolved"]["slots"]:
        slot.pop("summable", None)
    return block


@pytest.mark.parametrize(
    ("block", "family"),
    [
        pytest.param(_place_open(served_count=1), False, id="place-open-one-served"),
        pytest.param(_place_open(), True, id="place-open-floor-of-one"),
        pytest.param(_donor_open(confirmed=1, served=1), False, id="aid-one-served"),
        pytest.param(_donor_open(confirmed=2, served=2), True, id="aid-two-served"),
        pytest.param(_donor_open(confirmed=1, served=None), True, id="aid-one-confirmed"),
        # JSON `true` is no count of one.
        pytest.param(_place_open(served_count=True), True, id="served-count-true"),
    ],
)
def test_a_source_is_a_family_unless_search_states_it_serves_exactly_one_member(
    block: dict[str, Any], family: bool
) -> None:
    assert is_family(source_from_spec_block(block)) is family


def test_a_count_search_states_as_true_is_read_as_no_count() -> None:
    block = _place_open(member_count=True, served_count=True)
    block["resolved"]["shape"]["member_count"] = True

    source = source_from_spec_block(block)

    assert source.shape_member_count is None
    assert source.stat_var_member_count is None
    assert source.served_count is None


def test_a_floor_of_one_with_its_place_left_open_is_a_family_the_card_states_no_reading_for() -> (
    None
):
    """Health spending per person with no country named. `search` confirms
    one member as a floor and settles no membership, so the retrieval reads
    it as a family and states `separate`, and the card, which names one
    indicator varying along no axis, states no reading clause."""
    source = source_from_spec_block(_place_open())

    read = stat_var_reading(source, None)

    assert source.served_count is None
    assert [axis.column for axis in source.open_entity_axes or ()] == ["observationAbout"]
    assert read.stat_var_reading is StatVarReading.SEPARATE
    assert read.stat_var_reading_set_by is ReadingSetBy.RULE
    assert _reading_clause(read) is None


def _summable_expenditure_type(block: dict[str, Any]) -> dict[str, Any]:
    """`block` with its `expenditureType` slot unbound and summable, an open axis of the binding."""
    slot = _slot(block, "expenditureType")
    slot["binding"] = {"kind": "unbound"}
    slot["summable"] = True
    return block


@pytest.mark.parametrize(
    ("requested", "set_by"),
    [(None, ReadingSetBy.RULE), (StatVarReading.SUM, ReadingSetBy.PLAN)],
    ids=["unset", "sum"],
)
def test_a_floor_of_one_over_a_summable_axis_is_summed(
    compile_question: Callable[..., CompileResult],
    requested: StatVarReading | None,
    set_by: ReadingSetBy,
) -> None:
    """One member confirmed over a binding `search` counts as at least one, with
    an open axis the family sums over: the binding may hold more, so the
    retrieval asks for the sum, whether the plan states it or the rule does.
    The requirement names no place, so a filter keeps the countries before
    the total and the retrieval asks for place types."""
    source = source_from_spec_block(_summable_expenditure_type(_place_open()))
    health = _dataset(
        "health",
        [],
        Period(start=2015, end=2022),
        across=("country",),
        **({} if requested is None else {"stat_vars": requested}),
    )

    countries = (PlaceLevel(level=PlaceType.COUNTRY),)

    result = compile_question(
        [
            health,
            FilterRequirement(
                id="countries", decision_ids=["measure"], input_id="health", predicates=countries
            ),
            _yearly_total("countries"),
        ],
        [
            RetrieveOperation(id="health_op", requirement_id="health", spec_id=source.spec_id),
            FilterOperation(
                id="countries", requirement_id="countries", input="health_op", predicates=countries
            ),
            _yearly_total_op("countries"),
        ],
        terminal=["yearly_total"],
        sources={"health": source},
    )

    read = stat_var_reading(source, requested)
    assert is_family(source)
    assert read.stat_var_reading is StatVarReading.SUM
    assert read.stat_var_reading_set_by is set_by
    assert result.issues == ()
    assert _step_args(result, "health_op") == _retrieval(source, "sum") | {"place_types": True}
    assert _reading_clause(read) == (
        "summed over its indicators by default"
        if requested is None
        else "summed over its indicators"
    )


@pytest.mark.parametrize(
    ("block", "clause"),
    [
        pytest.param(_place_open(), None, id="one-confirmed-no-open-axis"),
        pytest.param(
            _place_open(member_count=3, member_count_exact=True),
            "kept separate over whichever of its 3 indicators hold data by default",
            id="three-confirmed",
        ),
        # Search counts the members it confirmed as a floor, so the mint may
        # serve more or fewer of them.
        pytest.param(
            _place_open(member_count=27),
            "kept as separate indicators by default",
            id="twenty-seven-confirmed-as-a-floor",
        ),
        pytest.param(
            _place_open(member_count=None),
            "kept as separate indicators by default",
            id="nothing-counted",
        ),
        pytest.param(
            _stating_no_summable(_donor_open(confirmed=2, served=2)),
            "kept separate over whichever of its 2 indicators hold data by default",
            id="two-served",
        ),
        pytest.param(
            _stating_no_summable(_donor_open(confirmed=1, served=None)),
            "kept as separate indicators by default",
            id="one-confirmed-over-open-axes",
        ),
    ],
)
def test_a_family_read_separately_states_its_clause_unless_it_counts_one_indicator_with_no_axis(
    block: dict[str, Any], clause: str | None
) -> None:
    read = stat_var_reading(source_from_spec_block(block), None)

    assert read.stat_var_reading is StatVarReading.SEPARATE
    assert _reading_clause(read) == clause


# --- The rule over edited copies of recorded blocks ---------------------------------


def test_an_open_slot_listed_unresolved_is_not_summable() -> None:
    block = _block("france_education_grants")
    block["resolved"]["unresolved_slots"] = ["crsPurpose"]

    source = source_from_spec_block(block)

    assert OpenAxis(property="crsPurpose", summable=False) in source.open_axes
    assert default_reading(source) is StatVarReading.SEPARATE


def test_an_open_netting_basis_leaves_a_crs_family_separate() -> None:
    # The recording binds nettingBasis to Gross; unbound, it pairs gross and net.
    block = _block("germany_oda_to_kenya")
    _slot(block, "nettingBasis")["binding"] = {"kind": "unbound"}

    source = source_from_spec_block(block)

    assert OpenAxis(property="nettingBasis", summable=False) in source.open_axes
    assert default_reading(source) is StatVarReading.SEPARATE


def test_a_block_carrying_no_summable_flag_reads_every_axis_unsummable() -> None:
    block = _block("france_education_grants")
    for slot in block["resolved"]["slots"]:
        del slot["summable"]

    source = source_from_spec_block(block)

    assert source.open_axes
    assert not any(axis.summable for axis in source.open_axes)
    assert default_reading(source) is StatVarReading.SEPARATE


def test_a_binding_kind_the_reader_does_not_know_is_an_open_unsummable_axis() -> None:
    block = _block("france_education_grants")
    _slot(block, "crsTypeOfFlow")["binding"] = {"kind": "range"}
    _slot(block, "crsTypeOfFlow")["summable"] = False

    source = source_from_spec_block(block)

    assert OpenAxis(property="crsTypeOfFlow", summable=False) in source.open_axes


def test_a_family_with_no_open_axis_reads_separately() -> None:
    # Nothing to sum over: every slot's own binding is a single unexpanded value.
    block = _block("france_education_grants")
    for slot in block["resolved"]["slots"]:
        slot.pop("expansion", None)
        if slot["binding"]["kind"] != "value" and slot["key"].get("property"):
            slot["binding"] = {"kind": "value", "value": {"ref": {"ref": "X"}}}

    source = source_from_spec_block(block)

    assert is_family(source)
    assert source.open_axes == ()
    assert default_reading(source) is StatVarReading.SEPARATE


# --- Who sets the reading -----------------------------------------------------------


@pytest.mark.parametrize("requested", list(StatVarReading))
def test_the_plan_s_reading_overrides_the_rule(requested: StatVarReading) -> None:
    read = stat_var_reading(_source("germany_bilateral_oda"), requested)

    assert read.stat_var_reading is requested
    assert read.stat_var_reading_set_by is ReadingSetBy.PLAN


def test_the_rule_reads_a_family_the_plan_leaves_unset() -> None:
    read = stat_var_reading(_source("male_female_population_nigeria"), None)

    assert read.stat_var_reading is StatVarReading.SEPARATE
    assert read.stat_var_reading_set_by is ReadingSetBy.RULE


def test_a_single_stat_var_source_has_no_reading_whatever_the_plan_asks() -> None:
    read = stat_var_reading(_source("life_expectancy_ghana"), StatVarReading.SUM)

    assert read.stat_var_reading is None
    assert read.stat_var_reading_set_by is None


# --- Compile-level: whole questions ---------------------------------------------


_WHAT = Decision(
    slot="measure",
    description="Which measure answers the question.",
    options=["the CRS flow", "another source"],
    chosen="the CRS flow",
    resolved_by=ResolvedBy.PLANNER_DEFAULT,
    rationale="The CRS reports the flow by donor and recipient.",
)


def _dataset(
    requirement_id: str,
    entities: Sequence[Entity],
    period: Period,
    **overrides: Any,
) -> DatasetRequirement:
    fields: dict[str, Any] = {
        "id": requirement_id,
        "decision_ids": ["measure"],
        "measure": "official development assistance",
        "entities": entities,
        "period": period,
    }
    return DatasetRequirement(**(fields | overrides))


def _yearly_total(input_id: str) -> AggregateRequirement:
    return AggregateRequirement(
        id="yearly_total",
        decision_ids=["measure"],
        input_id=input_id,
        reduction=Reduction.SUM,
        group_by=(GroupBy.PERIOD,),
    )


def _yearly_total_op(input_id: str) -> AggregateOperation:
    return AggregateOperation(
        id="yearly_total",
        requirement_id="yearly_total",
        input=input_id,
        reduction=Reduction.SUM,
        group_by=(GroupBy.PERIOD,),
    )


@pytest.fixture
def compile_question(
    schemas: dict[str, ToolSchema], capabilities: tuple[ToolCapability, ...]
) -> Callable[..., CompileResult]:
    def _compile(
        requirements: Sequence[EvidenceRequirement],
        operations: Sequence[Operation],
        *,
        terminal: Sequence[str],
        sources: dict[str, SelectedSource],
        decisions: Sequence[Decision] = (_WHAT,),
        answer_mode: AnswerMode = AnswerMode.COMPUTED,
    ) -> CompileResult:
        plan = Plan(
            question="a question about a family of StatVars",
            answer_mode=answer_mode,
            decisions=decisions,
            evidence_requirements=requirements,
            status=PlanStatus.DRAFT,
        )
        return compile_plan(
            plan=plan,
            translation=Translation(operations=operations, terminal=terminal),
            schemas=schemas,
            discovered=sources,
            capabilities=capabilities,
        )

    return _compile


def _step_args(result: CompileResult, step_id: str) -> dict[str, Any]:
    assert result.workflow is not None, result.issues
    return next(step.args for step in result.workflow.steps if step.id == step_id)


def test_chart_germany_s_bilateral_oda_sums_the_family_then_totals_each_year(
    compile_question: Callable[..., CompileResult],
) -> None:
    """ "Chart Germany's bilateral ODA since 2000": the recipient is left open,
    so the summed frame keeps one row per recipient and year, and the chart
    reads the aggregate summing each year."""
    source = _source("germany_bilateral_oda")
    oda = _dataset("oda", [Entity(name="Germany", role="donor")], Period(start=2000, end=2026))

    result = compile_question(
        [oda, _yearly_total("oda")],
        [
            RetrieveOperation(id="oda_op", requirement_id="oda", spec_id=source.spec_id),
            _yearly_total_op("oda_op"),
        ],
        terminal=["yearly_total"],
        sources={"oda": source},
    )

    assert _step_args(result, "oda_op") == _retrieval(source, "sum") | {
        "without_party_totals": True
    }
    assert _step_args(result, "yearly_total")["group_by"] == ["time_period"]
    assert result.workflow is not None
    (selected,) = result.workflow.sources
    assert selected.stat_var_reading is StatVarReading.SUM
    assert selected.stat_var_reading_set_by is ReadingSetBy.RULE


def test_germany_s_bilateral_oda_from_2005_filters_the_summed_frame_then_totals_each_year(
    compile_question: Callable[..., CompileResult],
) -> None:
    """ "Take Germany's bilateral ODA since 2000, keep only the years from 2005
    onwards for every recipient, and chart it"."""
    source = _source("germany_bilateral_oda")
    oda = _dataset("oda", [Entity(name="Germany", role="donor")], Period(start=2000, end=2026))
    from_2005 = FilterRequirement(
        id="from_2005",
        decision_ids=["measure"],
        input_id="oda",
        predicates=(PeriodThreshold(op="gte", year=2005),),
    )

    result = compile_question(
        [oda, from_2005, _yearly_total("from_2005")],
        [
            RetrieveOperation(id="oda_op", requirement_id="oda", spec_id=source.spec_id),
            FilterOperation(
                id="from_2005_op",
                requirement_id="from_2005",
                input="oda_op",
                predicates=(PeriodThreshold(op="gte", year=2005),),
            ),
            _yearly_total_op("from_2005_op"),
        ],
        terminal=["yearly_total"],
        sources={"oda": source},
    )

    assert _step_args(result, "oda_op")["stat_vars"] == "sum"
    assert _step_args(result, "yearly_total")["group_by"] == ["time_period"]


def test_total_education_grants_from_france_sum_the_family_then_total_each_year(
    compile_question: Callable[..., CompileResult],
) -> None:
    """ "total education grants from France in 2020-2024"."""
    source = _source("france_education_grants")
    grants = _dataset(
        "grants",
        [Entity(name="France", role="donor")],
        Period(start=2020, end=2024),
        measure="education grants",
    )

    result = compile_question(
        [grants, _yearly_total("grants")],
        [
            RetrieveOperation(id="grants_op", requirement_id="grants", spec_id=source.spec_id),
            _yearly_total_op("grants_op"),
        ],
        terminal=["yearly_total"],
        sources={"grants": source},
    )

    assert _step_args(result, "grants_op") == _retrieval(source, "sum") | {
        "without_party_totals": True
    }
    assert _step_args(result, "yearly_total") == {
        "aggregation": "sum",
        "group_by": ["time_period"],
    }


def test_a_family_confirming_one_member_asks_for_its_sum_and_feeds_a_single_measure_verb(
    compile_question: Callable[..., CompileResult],
) -> None:
    """France's education grants with one member confirmed and no membership
    settled: the retrieval asks apis for the family's sum, so the frame
    holds one StatVar whichever members hold data, and the yearly total reads it."""
    source = source_from_spec_block(_one_confirmed("france_education_grants"))
    grants = _dataset(
        "grants",
        [Entity(name="France", role="donor")],
        Period(start=2020, end=2024),
        measure="education grants",
    )

    result = compile_question(
        [grants, _yearly_total("grants")],
        [
            RetrieveOperation(id="grants_op", requirement_id="grants", spec_id=source.spec_id),
            _yearly_total_op("grants_op"),
        ],
        terminal=["yearly_total"],
        sources={"grants": source},
    )

    assert _step_args(result, "grants_op") == _retrieval(source, "sum") | {
        "without_party_totals": True
    }
    assert result.issues == ()


def test_chart_germany_s_oda_to_kenya_retrieves_the_sum_as_the_answer(
    compile_question: Callable[..., CompileResult],
) -> None:
    """ "Chart Germany's ODA to Kenya since 2000": both roles named, no verb."""
    source = _source("germany_oda_to_kenya")
    oda = _dataset(
        "oda",
        [Entity(name="Germany", role="donor"), Entity(name="Kenya", role="recipient")],
        Period(start=2000, end=2024),
    )

    result = compile_question(
        [oda],
        [RetrieveOperation(id="oda_op", requirement_id="oda", spec_id=source.spec_id)],
        terminal=["oda_op"],
        sources={"oda": source},
        answer_mode=AnswerMode.SERIES,
    )

    assert _step_args(result, "oda_op") == _retrieval(source, "sum") | {
        "without_party_totals": True
    }


def _gross_and_net_kenya() -> SelectedSource:
    block = _block("germany_oda_to_kenya")
    _slot(block, "nettingBasis")["binding"] = {"kind": "unbound"}
    return source_from_spec_block(block)


def test_a_total_over_a_family_the_rule_leaves_separate_is_refused_at_compile_time(
    compile_question: Callable[..., CompileResult],
) -> None:
    """An open netting basis pairs gross and net flows, so the rule keeps the
    StatVars apart, and an aggregate over them is refused before any call,
    where analysis-mcp's `stat_var_unresolved` would refuse it at run time."""
    source = _gross_and_net_kenya()
    oda = _dataset(
        "oda",
        [Entity(name="Germany", role="donor"), Entity(name="Kenya", role="recipient")],
        Period(start=2000, end=2024),
    )

    result = compile_question(
        [oda, _yearly_total("oda")],
        [
            RetrieveOperation(id="oda_op", requirement_id="oda", spec_id=source.spec_id),
            _yearly_total_op("oda_op"),
        ],
        terminal=["yearly_total"],
        sources={"oda": source},
    )

    (issue,) = result.issues
    assert issue.code is CompileIssueCode.INPUT_SHAPE_UNSUPPORTED
    assert issue.operation_id == "yearly_total"
    # Search confirmed 474 StatVars and settled no membership; the count is the 474.
    assert source.stat_var_member_count == 474
    assert "aggregate reads a single measure; got 474" in issue.reason
    # Summing would refuse `rollup_not_summable`, so the reason omits that advice.
    assert "cannot be summed over nettingBasis" in issue.reason
    assert "stat_vars='sum'" not in issue.reason


def _gross_and_net_kenya_counting(*, confirmed: int | None, served: int | None) -> SelectedSource:
    """Germany's ODA to Kenya over an open netting basis, its `resolved.stat_vars`
    stating `confirmed` members and `served` StatVars."""
    block = _block("germany_oda_to_kenya")
    _slot(block, "nettingBasis")["binding"] = {"kind": "unbound"}
    block["resolved"]["stat_vars"].update(member_count=confirmed, served_count=served)
    return source_from_spec_block(block)


def _compile_yearly_oda_to_kenya(
    compile_question: Callable[..., CompileResult], source: SelectedSource
) -> CompileResult:
    oda = _dataset(
        "oda",
        [Entity(name="Germany", role="donor"), Entity(name="Kenya", role="recipient")],
        Period(start=2000, end=2024),
    )
    return compile_question(
        [oda, _yearly_total("oda")],
        [
            RetrieveOperation(id="oda_op", requirement_id="oda", spec_id=source.spec_id),
            _yearly_total_op("oda_op"),
        ],
        terminal=["yearly_total"],
        sources={"oda": source},
    )


def test_a_flow_search_serves_one_member_of_feeds_a_single_measure_verb(
    compile_question: Callable[..., CompileResult],
) -> None:
    """One member served: the frame holds one StatVar and the yearly total
    reads it, though the open netting basis would keep a family separate. The
    source is no family and states no reading."""
    source = _gross_and_net_kenya_counting(confirmed=1, served=1)

    result = _compile_yearly_oda_to_kenya(compile_question, source)

    assert not is_family(source)
    assert result.issues == ()
    assert _step_args(result, "oda_op") == _retrieval(source) | {"without_party_totals": True}


@pytest.mark.parametrize("confirmed", [1, None], ids=["one-confirmed", "no-count"])
def test_a_family_whose_membership_is_unsettled_read_separately_is_refused_at_compile_time(
    compile_question: Callable[..., CompileResult], confirmed: int | None
) -> None:
    """One member confirmed, or none counted, and no membership settled: more
    members may hold data, so the aggregate is refused as it is for a family
    counted whole, and the reason states no count the source cannot support."""
    source = _gross_and_net_kenya_counting(confirmed=confirmed, served=None)

    result = _compile_yearly_oda_to_kenya(compile_question, source)

    (issue,) = result.issues
    assert issue.code is CompileIssueCode.INPUT_SHAPE_UNSUPPORTED
    assert issue.operation_id == "yearly_total"
    assert issue.reason.endswith(
        "aggregate reads a single measure; got several, one per StatVar of a family read "
        "separately; it cannot be summed over nettingBasis, so the question has to name one "
        "value for it"
    )


def test_a_summable_family_the_plan_reads_separately_is_refused_at_compile_time(
    compile_question: Callable[..., CompileResult],
) -> None:
    """The plan reads France's education grants separately and totals them.
    The plan itself cannot count the family, so the compiler refuses it once
    the source shows 68 StatVars, each open axis summable, and advises
    reading the family summed."""
    source = _source("france_education_grants")
    grants = _dataset(
        "grants",
        [Entity(name="France", role="donor")],
        Period(start=2020, end=2024),
        measure="education grants",
        stat_vars=StatVarReading.SEPARATE,
    )

    result = compile_question(
        [grants, _yearly_total("grants")],
        [
            RetrieveOperation(id="grants_op", requirement_id="grants", spec_id=source.spec_id),
            _yearly_total_op("grants_op"),
        ],
        terminal=["yearly_total"],
        sources={"grants": source},
    )

    (issue,) = result.issues
    assert issue.code is CompileIssueCode.INPUT_SHAPE_UNSUPPORTED
    assert issue.operation_id == "yearly_total"
    assert issue.reason.endswith(
        "aggregate reads a single measure; got 68, one per StatVar of a family read "
        "separately; read the family summed (stat_vars='sum') to give it one"
    )


def test_a_family_of_two_the_plan_reads_separately_is_refused_at_compile_time(
    compile_question: Callable[..., CompileResult],
) -> None:
    """Male and female population, read separately and totalled: apis may
    not sum over sex, so the reason asks for one value of it."""
    source = _source("male_female_population_nigeria")
    people = _dataset(
        "people",
        [Entity(name="Nigeria")],
        Period(start=1991, end=2006),
        measure="population",
        stat_vars=StatVarReading.SEPARATE,
    )

    result = compile_question(
        [people, _yearly_total("people")],
        [
            RetrieveOperation(id="people_op", requirement_id="people", spec_id=source.spec_id),
            _yearly_total_op("people_op"),
        ],
        terminal=["yearly_total"],
        sources={"people": source},
    )

    (issue,) = result.issues
    assert issue.code is CompileIssueCode.INPUT_SHAPE_UNSUPPORTED
    assert issue.operation_id == "yearly_total"
    assert "aggregate reads a single measure; got 2" in issue.reason
    assert "cannot be summed over gender, so the question has to name one value for it" in (
        issue.reason
    )


def test_a_separate_reading_of_a_single_stat_var_source_feeds_an_aggregate(
    compile_question: Callable[..., CompileResult],
) -> None:
    """`stat_vars` has no effect on a source binding one StatVar, so a plan
    reading it separately compiles its total and writes no reading."""
    source = _source("life_expectancy_ghana")
    life = _dataset(
        "life",
        [Entity(name="Ghana")],
        Period(start=2000, end=2020),
        measure="life expectancy",
        stat_vars=StatVarReading.SEPARATE,
    )

    result = compile_question(
        [life, _yearly_total("life")],
        [
            RetrieveOperation(id="life_op", requirement_id="life", spec_id=source.spec_id),
            _yearly_total_op("life_op"),
        ],
        terminal=["yearly_total"],
        sources={"life": source},
    )

    assert result.issues == ()
    assert _step_args(result, "life_op") == _retrieval(source)
    assert _step_args(result, "yearly_total")["group_by"] == ["time_period"]


_READ_AS_TOTAL = Decision(
    slot="read_as_total",
    description="Whether to read the flows as their total or one series each.",
    options=["sum", "separate"],
    chosen="sum",
    resolved_by=ResolvedBy.PLANNER_DEFAULT,
    rationale="The question asks for the total.",
    fixes=DecisionFixes(stat_vars=StatVarReading.SUM),
)


def _compile_explicit_total(
    compile_question: Callable[..., CompileResult],
    source: SelectedSource,
    entities: Sequence[Entity],
) -> CompileResult:
    """A yearly total over `source`'s family, the plan stating `stat_vars='sum'`."""
    oda = _dataset(
        "oda",
        entities,
        Period(start=2000, end=2024),
        decision_ids=["measure", "read_as_total"],
        stat_vars=StatVarReading.SUM,
    )
    return compile_question(
        [oda, _yearly_total("oda")],
        [
            RetrieveOperation(id="oda_op", requirement_id="oda", spec_id=source.spec_id),
            _yearly_total_op("oda_op"),
        ],
        terminal=["yearly_total"],
        sources={"oda": source},
        decisions=(_WHAT, _READ_AS_TOTAL),
    )


def test_an_explicit_total_is_written_as_the_plan_s_reading(
    compile_question: Callable[..., CompileResult],
) -> None:
    """The plan states `sum` over a family summable along every open axis:
    the compiler writes it, and the source records the plan set it."""
    source = _source("germany_bilateral_oda")

    result = _compile_explicit_total(
        compile_question, source, [Entity(name="Germany", role="donor")]
    )

    assert _step_args(result, "oda_op")["stat_vars"] == "sum"
    assert result.workflow is not None
    assert result.workflow.sources[0].stat_var_reading_set_by is ReadingSetBy.PLAN


def test_an_explicit_total_over_an_unsummable_axis_is_refused_at_compile_time(
    compile_question: Callable[..., CompileResult],
) -> None:
    """The question asks for "the total" of a family with an open netting
    basis: the plan states `sum`, which `get_data` would refuse
    `rollup_not_summable`, so compile refuses it before the user approves."""
    source = _gross_and_net_kenya()

    result = _compile_explicit_total(
        compile_question,
        source,
        [Entity(name="Germany", role="donor"), Entity(name="Kenya", role="recipient")],
    )

    (issue,) = result.issues
    assert issue.code is CompileIssueCode.FAMILY_UNSUMMABLE
    assert (issue.operation_id, issue.requirement_id) == ("oda_op", "oda")
    assert "nettingBasis" in issue.reason


def test_male_and_female_population_charted_keeps_one_series_per_sex(
    compile_question: Callable[..., CompileResult],
) -> None:
    source = _source("male_female_population_nigeria")
    people = _dataset(
        "people", [Entity(name="Nigeria")], Period(start=1991, end=2006), measure="population"
    )

    result = compile_question(
        [people],
        [RetrieveOperation(id="people_op", requirement_id="people", spec_id=source.spec_id)],
        terminal=["people_op"],
        sources={"people": source},
        answer_mode=AnswerMode.SERIES,
    )

    assert _step_args(result, "people_op") == _retrieval(source, "separate")


def test_a_comparison_over_a_family_read_separately_is_refused_at_compile_time(
    compile_question: Callable[..., CompileResult],
) -> None:
    people_source = _source("male_female_population_nigeria")
    life_source = _source("life_expectancy_ghana")
    people = _dataset(
        "people", [Entity(name="Nigeria")], Period(start=1991, end=2006), measure="population"
    )
    life = _dataset(
        "life", [Entity(name="Ghana")], Period(start=2000, end=2020), measure="life expectancy"
    )
    difference = ComparisonRequirement(
        id="difference",
        decision_ids=["measure"],
        minuend=RequirementSide(input_id="life"),
        subtrahend=RequirementSide(input_id="people"),
    )

    result = compile_question(
        [people, life, difference],
        [
            RetrieveOperation(
                id="people_op", requirement_id="people", spec_id=people_source.spec_id
            ),
            RetrieveOperation(id="life_op", requirement_id="life", spec_id=life_source.spec_id),
            CompareOperation(
                id="difference",
                requirement_id="difference",
                left=Side(input="people_op"),
                right=Side(input="life_op"),
            ),
        ],
        terminal=["difference"],
        sources={"people": people_source, "life": life_source},
    )

    (issue,) = result.issues
    assert issue.code is CompileIssueCode.INPUT_SHAPE_UNSUPPORTED
    assert "compare reads a single measure; got 2" in issue.reason


def test_a_single_stat_var_source_writes_no_reading(
    compile_question: Callable[..., CompileResult],
) -> None:
    source = _source("life_expectancy_ghana")
    life = _dataset(
        "life", [Entity(name="Ghana")], Period(start=2000, end=2020), measure="life expectancy"
    )

    result = compile_question(
        [life, _yearly_total("life")],
        [
            RetrieveOperation(id="life_op", requirement_id="life", spec_id=source.spec_id),
            _yearly_total_op("life_op"),
        ],
        terminal=["yearly_total"],
        sources={"life": source},
    )

    assert _step_args(result, "life_op") == _retrieval(source)


# --- A breakdown by sector ------------------------------------------------------

_TOP_SECTORS = Breakdown(dimension=BreakdownDimension.SECTOR, level=0)

_EVERY_SECTOR = Decision(
    slot="every_sector",
    description="Whether to rank every sector or named ones.",
    options=["every sector", "named sectors"],
    chosen="every sector",
    resolved_by=ResolvedBy.PLANNER_DEFAULT,
    rationale="The question asks which sectors received the most.",
    fixes=DecisionFixes(among=RankAmong.POPULATION),
)


def _compile_top_sectors(
    compile_question: Callable[..., CompileResult],
    source: SelectedSource,
    breakdown: Breakdown = _TOP_SECTORS,
) -> CompileResult:
    """ "Which sectors received the most of Germany's ODA to Kenya in 2024?"."""
    oda = _dataset(
        "oda",
        [Entity(name="Germany", role="donor"), Entity(name="Kenya", role="recipient")],
        Period(start=2024, end=2024),
        stat_vars=StatVarReading.SEPARATE,
        breakdown=breakdown,
    )
    by_sector: dict[str, Any] = {
        "reduction": Reduction.SUM,
        "group_by": (GroupBy.PERIOD, GroupBy.BREAKDOWN),
    }
    rank: dict[str, Any] = {
        "direction": RankDirection.HIGHEST_FIRST,
        "among": RankAmong.POPULATION,
        "axis": "sector",
        "limit": 10,
    }
    return compile_question(
        [
            oda,
            AggregateRequirement(
                id="by_sector", decision_ids=["measure"], input_id="oda", **by_sector
            ),
            RankRequirement(
                id="top", decision_ids=["measure", "every_sector"], input_id="by_sector", **rank
            ),
        ],
        [
            RetrieveOperation(id="oda_op", requirement_id="oda", spec_id=source.spec_id),
            AggregateOperation(
                id="by_sector_op", requirement_id="by_sector", input="oda_op", **by_sector
            ),
            RankOperation(id="top_op", requirement_id="top", input="by_sector_op", **rank),
        ],
        terminal=["top_op"],
        sources={"oda": source},
        decisions=(_WHAT, _EVERY_SECTOR),
    )


def test_a_breakdown_by_sector_retrieves_sector_totals_and_ranks_them_by_their_column(
    compile_question: Callable[..., CompileResult],
) -> None:
    """The retrieval asks for the family summed within each top-level purpose code,
    and the aggregate and the rank name the `stat_var_crsPurpose` column it lands in."""
    source = _source("germany_oda_to_kenya")

    result = _compile_top_sectors(compile_question, source)

    assert _step_args(result, "oda_op") == _retrieval(source, "separate") | {
        "breakdown": {"code_system": "crsPurpose", "level": 0},
        "without_party_totals": True,
    }
    assert _step_args(result, "by_sector_op")["group_by"] == ["time_period", "stat_var_crsPurpose"]
    assert _step_args(result, "top_op")["rank_by"] == ["stat_var_crsPurpose"]
    assert result.workflow is not None
    keys = {step.id: step.expected_role_keys for step in result.workflow.steps}
    assert keys["by_sector_op"] == keys["top_op"] == ("stat_var_crsPurpose",)


# Paths from the root of the CRS purpose hierarchy, as the graph's
# `specializationOf` parents give them.
_HEALTH_PATH = ("CRSPurposeCode100", "CRSPurposeCode120")
_POPULATION_PATH = ("CRSPurposeCode100", "CRSPurposeCode130")
_BASIC_HEALTH_CARE_PATH = (*_HEALTH_PATH, "CRSPurposeCode122", "CRSPurposeCode12220")


# The CRS purpose codes the graph holds nothing below.
_LEAVES = ("CRSPurposeCode12220",)


def _naming_sectors(
    codes: tuple[str, ...],
    paths: dict[str, tuple[str, ...]],
    *,
    binding: str = "value",
    expanded: bool = True,
) -> SelectedSource:
    """Germany's ODA to Kenya with its purpose slot bound to `codes`, and the paths and
    leaves the hierarchy read found for them.

    Several codes bind as a set, as Search binds "health and population ODA". One code
    binds as `binding`: Search binds health alone as one value its expansion widens to
    general health (121), and basic health care (12220), a leaf, as one value with no
    expansion, which closes the slot. `expanded` false states no expansion."""
    block = _block("germany_oda_to_kenya")
    identity_slot = next(
        slot
        for slot in block["definition"]["identity"]["slots"]
        if slot["property"] == "crsPurpose"
    )
    one_value = len(codes) == 1 and binding == "value"
    identity_slot["binding"] = (
        {"state": "value", "ref": codes[0]} if one_value else {"state": "set", "refs": list(codes)}
    )
    values = [{"ref": {"ref": code}, "value_kind": "enum_value"} for code in codes]
    slot = _slot(block, "crsPurpose")
    slot["binding"] = (
        {"kind": "value", "value": values[0]} if one_value else {"kind": "set", "values": values}
    )
    if expanded:
        slot["expansion"] = {"added": [{"ref": "CRSPurposeCode121"}]}
    else:
        slot.pop("expansion", None)
    return source_from_spec_block(block).model_copy(
        update={"code_paths": paths, "leaf_codes": tuple(c for c in codes if c in _LEAVES)}
    )


_PATHS = {
    "CRSPurposeCode120": _HEALTH_PATH,
    "CRSPurposeCode130": _POPULATION_PATH,
    "CRSPurposeCode12220": _BASIC_HEALTH_CARE_PATH,
}


@pytest.mark.parametrize(
    ("codes", "binding", "expanded", "stated", "level"),
    [
        (("CRSPurposeCode120", "CRSPurposeCode130"), "set", True, 0, 1),
        (("CRSPurposeCode120", "CRSPurposeCode130"), "set", False, 0, 1),
        (("CRSPurposeCode120", "CRSPurposeCode130"), "set", True, 2, 2),
        (("CRSPurposeCode120",), "value", True, 0, 2),
        (("CRSPurposeCode120",), "set", False, 0, 2),
        (("CRSPurposeCode12220",), "set", False, 0, 3),
        (("CRSPurposeCode120", "CRSPurposeCode12220"), "set", True, 0, 2),
        ((), "value", True, 0, 0),
    ],
    ids=[
        "health-and-population-part-below-their-shared-parent",
        "health-and-population-with-no-expansion-stated",
        "a-finer-stated-level-is-kept",
        "one-code-reads-its-children",
        "one-code-with-no-expansion-stated-reads-its-children",
        "a-leaf-in-an-open-slot-reads-its-own-level",
        "a-code-under-another-parts-below-it",
        "no-code-keeps-the-stated-level",
    ],
)
def test_a_breakdown_reads_the_level_its_named_sectors_part_at(
    compile_question: Callable[..., CompileResult],
    codes: tuple[str, ...],
    binding: str,
    expanded: bool,
    stated: int,
    level: int,
) -> None:
    """ "Health and population ODA to Kenya by sector": the two share social
    infrastructure (100) at the top level, so a breakdown there is one total; it reads
    level 1, where they are two sectors. "Health ODA by sector" reads health's
    subsectors, whether or not Search states the expansion, since the graph says health
    has children; a leaf has none and reads itself."""
    source = (
        _naming_sectors(
            codes,
            {code: _PATHS[code] for code in codes},
            binding=binding,
            expanded=expanded,
        )
        if codes
        else _source("germany_oda_to_kenya")
    )

    result = _compile_top_sectors(
        compile_question,
        source,
        Breakdown(dimension=BreakdownDimension.SECTOR, level=stated),
    )

    assert _step_args(result, "oda_op")["breakdown"] == {
        "code_system": "crsPurpose",
        "level": level,
    }


def test_a_breakdown_of_one_leaf_as_search_binds_it_is_unusable(
    compile_question: Callable[..., CompileResult],
) -> None:
    """ "Basic health care ODA by sector": Search binds the leaf as one value with no
    expansion, so the family holds one sector and has nothing to break down by."""
    source = _naming_sectors(
        ("CRSPurposeCode12220",), {"CRSPurposeCode12220": _BASIC_HEALTH_CARE_PATH}, expanded=False
    )

    result = _compile_top_sectors(compile_question, source)

    assert [(issue.code, issue.requirement_id) for issue in result.issues] == [
        (CompileIssueCode.SOURCE_UNUSABLE, "oda")
    ]


def test_a_breakdown_whose_named_sectors_have_no_path_is_unusable(
    compile_question: Callable[..., CompileResult],
) -> None:
    """With the graph holding no path to health, no level of the breakdown is known."""
    source = _naming_sectors(
        ("CRSPurposeCode120", "CRSPurposeCode130"),
        {"CRSPurposeCode130": _POPULATION_PATH},
        binding="set",
    )

    result = _compile_top_sectors(compile_question, source)

    assert [(issue.code, issue.requirement_id) for issue in result.issues] == [
        (CompileIssueCode.SOURCE_UNUSABLE, "oda")
    ]


def test_a_breakdown_over_a_source_that_does_not_vary_by_sector_is_unusable(
    compile_question: Callable[..., CompileResult],
) -> None:
    """Germany's ODA to Kenya with its purpose slot cut from the recorded block: its
    StatVars state no purpose code, so it has no sector to break down by."""
    block = _block("germany_oda_to_kenya")
    block["resolved"]["slots"].remove(_slot(block, "crsPurpose"))
    source = source_from_spec_block(block)

    result = _compile_top_sectors(compile_question, source)

    assert [(issue.code, issue.requirement_id) for issue in result.issues] == [
        (CompileIssueCode.SOURCE_UNUSABLE, "oda")
    ]
