"""How a source binding a family of StatVars is read, summed or separately.

The spec blocks come from `tests/fixtures/family_spec_blocks.json`: single-part
`search` envelopes recorded from dfo-apis, cut to the fields a source reads.
Each slot carries `summable` and each elected item its `constraint_properties`
as dfo-apis serves them. A test that needs a slot the recordings lack edits a
copy of a recorded block and says what it changed.

A role-bearing CRS identity never mints as one StatVar, so it is a family
however many members `search` listed: the recordings list 68 and 200, and the
one-member case is a recorded block cut to its first item. A source restored
from a record that states no pin verdict is a family by the StatVars it lists.

The three blocks that elect one StatVar keyed on a place each state it as
the family's root: its `constraint_properties` are the identity's
`value`-bound slot properties. `_as_root` writes the same two fields onto an
edited block that elects a root.

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
    member_digest,
    outcome_from_envelope,
    posted_stat_vars,
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


def _as_root(block: dict[str, Any]) -> dict[str, Any]:
    """`block` with its elected items stating what Search states of a family's root.

    A root's `constraint_properties` are the identity's `value`-bound slot
    properties, and its `slot_values` those slots' refs
    (`data_mcp.substitution.root_member`).
    """
    bound = {
        slot["property"]: slot["binding"]["ref"]
        for slot in block["definition"]["identity"]["slots"]
        if slot["binding"]["state"] == "value"
    }
    for item in block["resolved"]["stat_vars"]["items"]:
        item["constraint_properties"] = sorted(bound)
        item["slot_values"] = [
            {"key": {"property": {"dcid": prop}}, "value": {"ref": {"dcid": ref}}}
            for prop, ref in bound.items()
        ]
    return block


def _block(name: str) -> dict[str, Any]:
    """A deep copy of one recorded spec block, safe to edit."""
    return copy.deepcopy(_BLOCKS[name]["parts"][0]["spec"])


def _retrieval(source: SelectedSource, stat_vars: str | None = None) -> dict[str, Any]:
    """The `get_data` arguments a retrieval of `source` writes: its spec_id, the reading
    when it states one, and the StatVars it posts (`posted_stat_vars`)."""
    args: dict[str, Any] = {"spec_id": source.spec_id}
    if stat_vars is not None:
        args["stat_vars"] = stat_vars
    if posted := posted_stat_vars(source):
        args["resolved_stat_var_dcids"] = list(posted)
    return args


def _source(name: str) -> SelectedSource:
    return source_from_spec_block(_block(name))


def _one_elected(name: str) -> dict[str, Any]:
    """A recorded block with `resolved.stat_vars` cut to its first item, as `search`
    states it when one member of the binding has data for the roles asked."""
    block = _block(name)
    stat_vars = block["resolved"]["stat_vars"]
    stat_vars.update(items=stat_vars["items"][:1], member_count=1, truncated=False)
    return block


def _slot(block: dict[str, Any], prop: str) -> dict[str, Any]:
    return next(
        slot
        for slot in block["resolved"]["slots"]
        if (slot["key"].get("property") or {}).get("dcid") == prop
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


def test_a_family_capped_by_search_is_still_a_family() -> None:
    # 200 of 1,554 members listed: `items` is capped well above one.
    source = _source("germany_bilateral_oda")

    assert len(source.stat_vars) == 200
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
def test_one_elected_stat_var_is_no_family_however_many_candidates_search_counted(
    name: str,
) -> None:
    """Each binds one StatVar. `member_count` counts the three candidates
    confirmed at the place, so `truncated` reads true, and neither makes the
    source a family. Every slot of these is inapplicable, so a rule reading
    `truncated` as a family would sum each over no axis at all."""
    for source in outcome_from_envelope(_BLOCKS[name]).sources:
        assert len(source.stat_vars) == 1
        assert source.stat_var_member_count == 3
        assert source.stat_var_truncated is True
        assert not is_family(source)
        assert stat_var_reading(source, None).stat_var_reading is None


# --- A family `search` lists one member of -------------------------------------------


def test_every_recorded_role_bearing_identity_is_unpinned_and_every_pinned_one_is_no_family() -> (
    None
):
    """A described identity keyed on `observationAbout` alone whose one
    elected StatVar is its family's root mints as that StatVar, and no other
    does. The pin and the family predicate read the same rule, so a source
    is one or the other."""
    pinned = {
        name: [source.pinned for source in outcome_from_envelope(envelope).sources]
        for name, envelope in _BLOCKS.items()
    }

    assert pinned == {
        "france_education_grants": [False],
        "germany_bilateral_oda": [False],
        "germany_oda_to_kenya": [False],
        "male_female_population_nigeria": [False],
        "life_expectancy_ghana": [True],
        "population_kenya": [True],
        "unemployment_rate_by_sex_spain": [True, True],
    }
    for name, envelope in _BLOCKS.items():
        for source in outcome_from_envelope(envelope).sources:
            assert is_family(source) is (not source.pinned), name


def test_a_role_bearing_family_listing_one_member_is_still_a_family_and_is_summed() -> None:
    """France's education grants with one member listed. An identity keyed
    on a provider states no root, so dfo-apis mints every member of the
    binding, and the open axes, all summable, read it summed."""
    source = source_from_spec_block(_one_elected("france_education_grants"))

    assert len(source.stat_vars) == 1
    assert source.pinned is False
    assert is_family(source)
    read = stat_var_reading(source, None)
    assert read.stat_var_reading is StatVarReading.SUM
    assert read.stat_var_reading_set_by is ReadingSetBy.RULE


def test_a_role_bearing_family_listing_one_member_stays_separate_over_an_unsummable_axis() -> None:
    # The recording binds nettingBasis to Gross; unbound, it pairs gross and net.
    block = _one_elected("germany_oda_to_kenya")
    _slot(block, "nettingBasis")["binding"] = {"kind": "unbound"}

    source = source_from_spec_block(block)

    assert is_family(source)
    assert stat_var_reading(source, None).stat_var_reading is StatVarReading.SEPARATE


def test_an_identity_electing_its_root_beside_another_member_is_a_family() -> None:
    """Total population with the female population elected too. The mint
    reduces the family to its root only when the root is the one StatVar
    `search` elected, so two elected members are read as a family."""
    block = _block("population_kenya")
    block["resolved"]["stat_vars"]["items"].append(
        {"ref": "Count_Person_Female", "slot_values": [], "constraint_properties": ["gender"]}
    )

    source = source_from_spec_block(block)

    assert source.stat_vars == ("Count_Person", "Count_Person_Female")
    assert source.pinned is False
    assert is_family(source)


@pytest.mark.parametrize(
    "constraint_properties",
    [None, ["gender"]],
    ids=["constraints-unknown", "constrained-beyond-the-bound-slots"],
)
def test_one_elected_stat_var_that_is_not_the_root_is_a_family(
    constraint_properties: list[str] | None,
) -> None:
    """Total population in Kenya binds no slot, so its root asserts no
    constraint. An elected StatVar whose constraints Search does not state,
    and one that asserts a constraint, are each one member of the family the
    mint resolves."""
    block = _block("population_kenya")
    (item,) = block["resolved"]["stat_vars"]["items"]
    item["constraint_properties"] = constraint_properties

    source = source_from_spec_block(block)

    assert source.pinned is False
    assert is_family(source)


def test_a_recorded_block_electing_its_root_states_the_roots_constraint_properties() -> None:
    """Total population in Kenya binds no slot, and Search states that its one
    elected StatVar asserts no constraint, so that StatVar is the family's root
    and the source mints as it. An item stating no `constraint_properties` is
    `test_one_elected_stat_var_that_is_not_the_root_is_a_family`."""
    recorded = json.loads(
        (Path(__file__).parent.parent / "fixtures" / "family_spec_blocks.json").read_text()
    )["population_kenya"]["parts"][0]["spec"]
    assert recorded["resolved"]["stat_vars"]["items"] == [
        {"ref": "Count_Person", "constraint_properties": [], "slot_values": []}
    ]

    source = source_from_spec_block(recorded)

    assert source.pinned is True
    assert not is_family(source)


def test_a_source_carrying_no_identity_is_a_family_by_the_stat_vars_it_lists() -> None:
    block = _block("france_education_grants")
    del block["definition"]

    listed = source_from_spec_block(block)
    one = source_from_spec_block(_one_elected("france_education_grants") | {"definition": {}})

    assert listed.identity is None and listed.pinned is None and is_family(listed)
    assert one.identity is None and one.pinned is None and not is_family(one)
    # data-mcp lists none of a family Search states a member digest for; the
    # binding's own count says whether it is a family.
    unlisted = listed.model_copy(update={"stat_vars": ()})
    assert unlisted.shape_member_digest is not None and unlisted.shape_member_count == 183
    assert is_family(unlisted)
    assert not is_family(unlisted.model_copy(update={"shape_member_count": 1}))


@pytest.mark.parametrize("name", ["population_kenya", "life_expectancy_ghana"])
def test_a_restored_source_stating_no_pin_verdict_is_a_family_only_by_the_stat_vars_it_lists(
    name: str,
) -> None:
    """A stored `SelectedSource` whose record carries no `pinned` key: the pin
    rule was not evaluated for it, so its one listed StatVar makes it no
    family and the plan card states no reading."""
    dumped = _source(name).model_dump(mode="json")
    del dumped["pinned"]

    restored = SelectedSource.model_validate(dumped)

    assert restored.pinned is None
    assert not is_family(restored)
    read = stat_var_reading(restored, None)
    assert read.stat_var_reading is None
    assert _reading_clause(read) is None


def test_a_restored_role_bearing_source_stating_no_pin_verdict_counts_the_stat_vars_it_lists() -> (
    None
):
    dumped = _source("france_education_grants").model_dump(mode="json")
    del dumped["pinned"]

    restored = SelectedSource.model_validate(dumped)

    assert len(restored.stat_vars) == 68
    assert is_family(restored)


# --- An unpinned binding `search` states holds one member ----------------------------

_PLACE_OPEN_BLOCK: dict[str, Any] = json.loads(
    (Path(__file__).parent.parent / "fixtures" / "search_spec_place_scope.json").read_text()
)["spec"]
_DONOR_OPEN_BLOCK: dict[str, Any] = json.loads(
    (Path(__file__).parent.parent / "fixtures" / "search_spec_role_open.json").read_text()
)["spec"]


def _place_open(**stat_vars: object) -> dict[str, Any]:
    """The recorded place-open block, its `resolved.stat_vars` fields overridden."""
    block = copy.deepcopy(_PLACE_OPEN_BLOCK)
    block["resolved"]["stat_vars"].update(stat_vars)
    return block


def _place_bound(**stat_vars: object) -> dict[str, Any]:
    """The place-open block with Kenya bound on `observationAbout` and no scope left
    open, its one elected StatVar stated as the family's root (`_as_root`)."""
    block = _as_root(_place_open(**stat_vars))
    block["definition"]["identity"]["entities"] = {"observationAbout": ["country/KEN"]}
    block["resolved"]["entities"] = [
        {"role": "observationAbout", "ref": "country/KEN", "label": "Kenya"}
    ]
    del block["resolved"]["scope"]
    return block


def _without(block: dict[str, Any], *path: str) -> dict[str, Any]:
    """`block` with the key `path` names removed from `resolved`."""
    *parents, key = path
    node = block["resolved"]
    for parent in parents:
        node = node[parent]
    del node[key]
    return block


def _donor_open(*, listed: int, truncated: bool) -> dict[str, Any]:
    block = copy.deepcopy(_DONOR_OPEN_BLOCK)
    stat_vars = block["resolved"]["stat_vars"]
    stat_vars.update(items=stat_vars["items"][:listed], member_count=listed, truncated=truncated)
    return block


def _stating_no_summable(block: dict[str, Any]) -> dict[str, Any]:
    """`block` as a dfo-apis that predates `Slot.summable` serves it: no slot states one."""
    for slot in block["resolved"]["slots"]:
        slot.pop("summable", None)
    return block


def _donor_bound() -> dict[str, Any]:
    block = copy.deepcopy(_DONOR_OPEN_BLOCK)
    block["definition"]["identity"]["entities"]["origin"] = ["country/FRA"]
    block["resolved"]["entities"].append(
        {"role": "origin", "ref": "country/FRA", "label": "France"}
    )
    block["resolved"]["role_axis_facts"] = []
    return block


def _shape_count(block: dict[str, Any], count: object) -> dict[str, Any]:
    block["resolved"]["shape"]["member_count"] = count
    return block


def _other_digest(block: dict[str, Any], dcids: Sequence[str]) -> dict[str, Any]:
    """`block` with the digest of `dcids`, a population other than its one listed StatVar."""
    block["resolved"]["shape"]["member_digest"] = member_digest(dcids)
    return block


def _exact(block: dict[str, Any]) -> dict[str, Any]:
    """`block` with the digest of its one listed StatVar, which states the binding is that member."""
    (item,) = block["resolved"]["stat_vars"]["items"]
    block["resolved"]["shape"]["member_digest"] = member_digest([item["ref"]])
    return block


@pytest.mark.parametrize(
    ("block", "pinned", "family"),
    [
        pytest.param(_place_bound(), True, False, id="place-bound"),
        pytest.param(_exact(_place_open()), False, False, id="place-open-exactly-one-member"),
        pytest.param(
            _without(_exact(_place_open()), "scope"),
            False,
            False,
            id="place-less-exactly-one-member",
        ),
        # The recorded blocks state their binding count with no digest: a floor.
        pytest.param(_place_open(), False, True, id="place-open-floor-of-one"),
        pytest.param(_without(_place_open(), "scope"), False, True, id="place-less-floor-of-one"),
        pytest.param(_donor_open(listed=2, truncated=False), False, True, id="aid-donor-open"),
        pytest.param(_donor_bound(), False, True, id="aid-fully-bound"),
        pytest.param(
            _donor_open(listed=1, truncated=True), False, True, id="aid-one-listed-capped"
        ),
        pytest.param(
            _donor_open(listed=1, truncated=False), False, True, id="aid-one-listed-of-many"
        ),
        pytest.param(
            _place_bound(member_count=4, truncated=True), True, False, id="place-bound-capped"
        ),
        # An exact binding of one still needs the listing to agree, uncapped.
        pytest.param(
            _exact(_place_open(member_count=3, truncated=True)), False, True, id="exact-capped"
        ),
        pytest.param(_exact(_place_open(truncated=True)), False, True, id="exact-list-capped"),
        pytest.param(
            _without(_exact(_place_open()), "stat_vars", "truncated"),
            False,
            True,
            id="exact-truncation-unstated",
        ),
        pytest.param(
            _without(_exact(_place_open()), "stat_vars", "member_count"),
            False,
            True,
            id="exact-confirmed-count-unstated",
        ),
        pytest.param(
            _without(_exact(_place_open()), "shape", "member_count"),
            False,
            True,
            id="digest-with-no-binding-count",
        ),
        pytest.param(
            _shape_count(_exact(_place_open()), 2), False, True, id="exact-binding-of-two"
        ),
        # A digest of some other population says the binding is not the listed member.
        pytest.param(
            _other_digest(_place_open(), ["Count_Person"]),
            False,
            True,
            id="digest-of-another-member",
        ),
        pytest.param(
            _other_digest(_place_open(), ["Count_Person", "Count_Person_Female"]),
            False,
            True,
            id="digest-of-a-larger-population",
        ),
        pytest.param(
            _shape_count(_donor_open(listed=1, truncated=False), 1),
            False,
            True,
            id="aid-count-of-one-under-the-digest-of-257",
        ),
        # JSON `true` is no count of one.
        pytest.param(
            _shape_count(_exact(_place_open()), True), False, True, id="binding-count-true"
        ),
        pytest.param(
            _exact(_place_open(member_count=True)), False, True, id="confirmed-count-true"
        ),
    ],
)
def test_an_unpinned_described_identity_is_a_family_unless_search_states_exactly_one_member(
    block: dict[str, Any], pinned: bool, family: bool
) -> None:
    source = source_from_spec_block(block)

    assert source.pinned is pinned
    assert is_family(source) is family


def test_a_count_search_states_as_true_is_read_as_no_count() -> None:
    source = source_from_spec_block(_shape_count(_place_open(member_count=True), True))

    assert source.shape_member_count is None
    assert source.stat_var_member_count is None


def test_a_floor_of_one_with_its_place_left_open_is_a_family_the_card_states_no_reading_for() -> (
    None
):
    """Health spending per person with no country named. `search` counts its
    binding as at least one, so the retrieval reads it as a family and
    states `separate`, and the card, which names one indicator varying along
    no axis, states no reading clause."""
    source = source_from_spec_block(copy.deepcopy(_PLACE_OPEN_BLOCK))

    read = stat_var_reading(source, None)

    assert source.pinned is False
    assert source.shape_member_digest is None
    assert [axis.column for axis in source.open_entity_axes or ()] == ["observationAbout"]
    assert read.stat_var_reading is StatVarReading.SEPARATE
    assert read.stat_var_reading_set_by is ReadingSetBy.RULE
    assert _reading_clause(read) is None


def test_the_digest_of_one_stat_var_is_read_with_or_without_its_dcid_prefix() -> None:
    """A block listing a prefixed StatVar is refused as a source, so the
    prefix reaches the rule only on a source restored from a record."""
    source = source_from_spec_block(_exact(_place_open()))
    (listed,) = source.stat_vars
    restored = source.model_copy(update={"stat_vars": (f"dcid:{listed}",)})

    assert not is_family(source)
    assert not is_family(restored)


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
    """One member listed over a binding `search` counts as at least one, with
    an open axis the family sums over: the binding may hold more, so the
    retrieval asks for the sum, whether the plan states it or the rule does.
    The requirement names no place, so a filter keeps the countries before
    the total and the retrieval asks for place types."""
    source = source_from_spec_block(_summable_expenditure_type(_place_open()))
    health = _dataset(
        "health",
        [],
        Period(start=2015, end=2022),
        across="country",
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
        "summed over its StatVars by default" if requested is None else "summed over its StatVars"
    )


@pytest.mark.parametrize(
    ("block", "clause"),
    [
        pytest.param(_place_open(), None, id="one-listed-uncapped-no-open-axis"),
        pytest.param(
            _place_open(member_count=3, truncated=True),
            "kept as 3 separate StatVars by default",
            id="one-listed-capped",
        ),
        pytest.param(
            _without(_place_open(), "stat_vars", "truncated"),
            "kept as separate StatVars by default",
            id="one-listed-truncation-unstated",
        ),
        pytest.param(
            _stating_no_summable(_donor_open(listed=2, truncated=False)),
            "kept as 2 separate StatVars by default",
            id="two-listed",
        ),
        pytest.param(
            _stating_no_summable(_donor_open(listed=1, truncated=False)),
            "kept as separate StatVars by default",
            id="one-listed-uncapped-over-open-axes",
        ),
    ],
)
def test_a_family_read_separately_states_its_clause_unless_it_lists_one_indicator_with_no_axis(
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
            slot["binding"] = {"kind": "value", "value": {"ref": {"dcid": "X"}}}

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

    assert _step_args(result, "oda_op") == _retrieval(source, "sum")
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

    assert _step_args(result, "grants_op") == _retrieval(source, "sum")
    assert _step_args(result, "yearly_total") == {
        "aggregation": "sum",
        "group_by": ["time_period"],
    }


def test_a_family_listing_one_member_asks_for_its_sum_and_feeds_a_single_measure_verb(
    compile_question: Callable[..., CompileResult],
) -> None:
    """France's education grants with one member listed: the retrieval asks
    dfo-apis for the family's sum, so the frame holds one StatVar whichever
    members hold data, and the yearly total reads it."""
    source = source_from_spec_block(_one_elected("france_education_grants"))
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

    assert _step_args(result, "grants_op") == _retrieval(source, "sum")
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

    assert _step_args(result, "oda_op") == _retrieval(source, "sum")


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
    # Search lists 200 of the 474 StatVars it confirmed; the count is the 474.
    assert source.stat_var_member_count == 474
    assert "aggregate reads a single measure; got 474" in issue.reason
    # Summing would refuse `rollup_not_summable`, so the reason omits that advice.
    assert "cannot be summed over nettingBasis" in issue.reason
    assert "stat_vars='sum'" not in issue.reason


def _gross_and_net_kenya_listing(items: int, *, truncated: bool) -> SelectedSource:
    """Germany's ODA to Kenya over an open netting basis, with `resolved.stat_vars`
    cut to its first `items` and stating whether `search` capped the list."""
    block = _block("germany_oda_to_kenya")
    _slot(block, "nettingBasis")["binding"] = {"kind": "unbound"}
    stat_vars = block["resolved"]["stat_vars"]
    stat_vars.update(
        items=stat_vars["items"][:items],
        member_count=items + 2 if truncated else items,
        truncated=truncated,
    )
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


def test_a_family_search_confirmed_one_member_of_read_separately_feeds_a_single_measure_verb(
    compile_question: Callable[..., CompileResult],
) -> None:
    """One member listed and the list complete: one member has data for the
    roles asked, so the frame holds one StatVar and the yearly total reads it,
    though the open netting basis keeps the reading separate."""
    source = _gross_and_net_kenya_listing(1, truncated=False)

    result = _compile_yearly_oda_to_kenya(compile_question, source)

    assert len(source.stat_vars) == 1
    assert result.issues == ()
    assert _step_args(result, "oda_op") == _retrieval(source, "separate")


def test_a_fully_bound_flow_search_confirmed_one_member_of_feeds_a_single_measure_verb(
    compile_question: Callable[..., CompileResult],
) -> None:
    """Every member-naming slot of the recorded block bound to one flat value,
    one member listed and a binding whose digest is that member's: the
    source is no family, states no reading, and is one measure."""
    block = _one_elected("germany_oda_to_kenya")
    for slot in block["resolved"]["slots"]:
        if not slot["key"].get("property"):
            continue
        if slot["binding"]["kind"] != "value" or slot.get("expansion"):
            slot["binding"] = {"kind": "value", "value": {"ref": {"dcid": "X"}}}
            slot.pop("expansion", None)
    block["resolved"]["shape"]["member_count"] = 1
    (listed,) = block["resolved"]["stat_vars"]["items"]
    block["resolved"]["shape"]["member_digest"] = member_digest([listed["ref"]])
    source = source_from_spec_block(block)

    result = _compile_yearly_oda_to_kenya(compile_question, source)

    assert source.pinned is False
    assert not is_family(source)
    assert source.open_axes == ()
    assert result.issues == ()
    assert _step_args(result, "oda_op") == _retrieval(source)


def test_a_family_listing_one_member_of_a_capped_list_read_separately_is_refused_at_compile_time(
    compile_question: Callable[..., CompileResult],
) -> None:
    """One member listed of a list `search` capped, with no count of the
    members it confirmed: more members may hold data, so the aggregate is
    refused as it is for a family counted whole, and the reason states no
    count the listing cannot support."""
    source = _gross_and_net_kenya_listing(1, truncated=True).model_copy(
        update={"stat_var_member_count": None}
    )

    result = _compile_yearly_oda_to_kenya(compile_question, source)

    assert len(source.stat_vars) == 1
    (issue,) = result.issues
    assert issue.code is CompileIssueCode.INPUT_SHAPE_UNSUPPORTED
    assert issue.operation_id == "yearly_total"
    assert issue.reason.endswith(
        "aggregate reads a single measure; got several, one per StatVar of a family read "
        "separately; it cannot be summed over nettingBasis, so the question has to name one "
        "value for it"
    )


def test_a_family_listing_no_member_read_separately_is_refused_without_a_count_of_zero(
    compile_question: Callable[..., CompileResult],
) -> None:
    source = _gross_and_net_kenya_listing(0, truncated=False)

    result = _compile_yearly_oda_to_kenya(compile_question, source)

    assert source.stat_vars == ()
    (issue,) = result.issues
    assert issue.code is CompileIssueCode.INPUT_SHAPE_UNSUPPORTED
    assert "aggregate reads a single measure; got several, one per StatVar" in issue.reason


def test_a_summable_family_search_confirmed_one_member_of_feeds_a_verb_when_the_plan_reads_it_separately(
    compile_question: Callable[..., CompileResult],
) -> None:
    """France's education grants with one member listed and the list
    complete, read separately by the plan and totalled."""
    source = source_from_spec_block(_one_elected("france_education_grants"))
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

    assert result.issues == ()
    assert _step_args(result, "grants_op") == _retrieval(source, "separate")


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
    """Male and female population, read separately and totalled: dfo-apis may
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


def test_an_explicit_total_asks_dfo_apis_for_the_sum_the_rule_would_not(
    compile_question: Callable[..., CompileResult],
) -> None:
    """The question asks for "the total" of a family with an open netting
    basis: the plan states `sum`, the compiler writes it, and dfo-apis
    decides whether the family can be summed (`rollup_not_summable`)."""
    source = _gross_and_net_kenya()
    total = Decision(
        slot="read_as_total",
        description="Whether to read the flows as their total or one series each.",
        options=["sum", "separate"],
        chosen="sum",
        resolved_by=ResolvedBy.PLANNER_DEFAULT,
        rationale="The question asks for the total.",
        fixes=DecisionFixes(stat_vars=StatVarReading.SUM),
    )
    oda = _dataset(
        "oda",
        [Entity(name="Germany", role="donor"), Entity(name="Kenya", role="recipient")],
        Period(start=2000, end=2024),
        decision_ids=["measure", "read_as_total"],
        stat_vars=StatVarReading.SUM,
    )

    result = compile_question(
        [oda, _yearly_total("oda")],
        [
            RetrieveOperation(id="oda_op", requirement_id="oda", spec_id=source.spec_id),
            _yearly_total_op("oda_op"),
        ],
        terminal=["yearly_total"],
        sources={"oda": source},
        decisions=(_WHAT, total),
    )

    assert _step_args(result, "oda_op")["stat_vars"] == "sum"
    assert result.workflow is not None
    assert result.workflow.sources[0].stat_var_reading_set_by is ReadingSetBy.PLAN


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
        left=RequirementSide(input_id="people"),
        right=RequirementSide(input_id="life"),
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
    compile_question: Callable[..., CompileResult], source: SelectedSource
) -> CompileResult:
    """ "Which sectors received the most of Germany's ODA to Kenya in 2024?"."""
    oda = _dataset(
        "oda",
        [Entity(name="Germany", role="donor"), Entity(name="Kenya", role="recipient")],
        Period(start=2024, end=2024),
        stat_vars=StatVarReading.SEPARATE,
        breakdown=_TOP_SECTORS,
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
        "breakdown": {"code_system": "crsPurpose", "level": 0}
    }
    assert _step_args(result, "by_sector_op")["group_by"] == ["time_period", "stat_var_crsPurpose"]
    assert _step_args(result, "top_op")["rank_by"] == ["stat_var_crsPurpose"]
    assert result.workflow is not None
    keys = {step.id: step.expected_role_keys for step in result.workflow.steps}
    assert keys["by_sector_op"] == keys["top_op"] == ("stat_var_crsPurpose",)


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
