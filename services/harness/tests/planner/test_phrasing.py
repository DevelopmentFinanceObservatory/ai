"""Tests for `planner.phrasing`: the words a plan card states a reading in.

`test_a_rank_and_a_keyed_share_state_their_population_claims_on_the_card`
is the strongest test of this module's job: a plan's rank and keyed
share_of requirements render `among`, `axis`, `per_period` and
`denominator_scope` in words on the plan card's own sentences, through
`harness.plan_display`, built from this module rather than from a
hand-maintained copy of the same prose.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest
from harness.plan_display import requirement_rows
from planner import phrasing
from planner.plan import (
    AggregateRequirement,
    DatasetRequirement,
    Decision,
    DecisionFixes,
    Entity,
    FilterRequirement,
    Period,
    Plan,
    QuotientRequirement,
    RankRequirement,
    RequirementSide,
    ShareOfRequirement,
)
from planner.records import (
    DenominatorScope,
    GroupBy,
    PlaceType,
    RankAmong,
    RankDirection,
    Reduction,
    ValueThreshold,
)

DecisionFactory = Callable[..., Decision]


@pytest.mark.parametrize(
    ("word", "plural"),
    [
        ("creditor", "creditors"),
        ("creditors", "creditors"),
        ("donor", "donors"),
        ("country", "countries"),
        ("party", "parties"),
        ("economy", "economies"),
        ("beneficiary", "beneficiaries"),
        ("recipient country", "recipient countries"),
        ("key", "keys"),
        ("survey", "surveys"),
        ("entity", "entities"),
    ],
)
def test_pluralize_turns_a_consonant_and_y_into_ies(word: str, plural: str) -> None:
    assert phrasing.pluralize(word) == plural


def test_entities_phrase_joins_names_in_order_deduplicated() -> None:
    entities = [Entity(name="Kenya"), Entity(name="China"), Entity(name="Kenya")]
    assert phrasing.entities_phrase(entities) == "Kenya and China"
    assert phrasing.entities_phrase([Entity(name="Kenya")]) == "Kenya"


def test_entities_phrase_tells_a_sets_members_from_the_set() -> None:
    assert phrasing.entities_phrase([Entity(name="Africa")]) == "Africa"
    assert (
        phrasing.entities_phrase([Entity(name="Africa", members=True), Entity(name="France")])
        == "each member of Africa and France"
    )


def test_entities_phrase_states_the_party_across_covers_last() -> None:
    assert phrasing.entities_phrase([], across="donor") == "every donor"
    assert (
        phrasing.entities_phrase([Entity(name="Kenya", role="recipient")], across="donor")
        == "Kenya and every donor"
    )


@pytest.mark.parametrize(
    ("across", "every"),
    [
        ("countries", "every country"),
        ("donors", "every donor"),
        ("economies", "every economy"),
    ],
)
def test_a_plural_across_reads_in_the_singular(across: str, every: str) -> None:
    assert phrasing.among_phrase(RankAmong.POPULATION, across=across) == (
        f"ranks {every} the source holds"
    )


@pytest.mark.parametrize(
    ("across", "every"),
    [
        ("donors", "every donor"),
        ("recipient", "every recipient"),
        ("country", "every place with data"),
        ("countries", "every place with data"),
        ("economies", "every place with data"),
        ("place", "every place with data"),
    ],
)
def test_a_party_reads_as_every_member_and_the_place_as_every_place_with_data(
    across: str, every: str
) -> None:
    # A retrieval that names no place holds every level of place, so the card
    # does not call it "every country".
    assert phrasing.entities_phrase([], across=across) == every


def test_the_place_reads_as_the_level_a_filter_keeps() -> None:
    assert phrasing.entities_phrase([], across="country", level=PlaceType.CONTINENT) == (
        "every continent"
    )
    assert phrasing.entities_phrase([], across="donors", level=PlaceType.COUNTRY) == "every donor"


def test_every_level_of_place_has_a_noun_the_card_can_pluralise() -> None:
    nouns = {level: phrasing.level_noun(level) for level in PlaceType}

    assert nouns[PlaceType.COUNTRY] == "country"
    assert phrasing.pluralize(nouns[PlaceType.CITY]) == "cities"
    assert phrasing.pluralize(nouns[PlaceType.EUROSTAT_NUTS_2]) == "NUTS 2 regions"
    assert len(set(nouns.values())) == len(PlaceType)


@pytest.mark.parametrize(
    ("across", "member_count", "stated"),
    [
        ("donor", 55, "ranks all 55 donors the source holds"),
        ("donors", 55, "ranks all 55 donors the source holds"),
        ("country", 193, "ranks all 193 countries the source holds"),
        ("countries", 12_345, "ranks all 12,345 countries the source holds"),
    ],
)
def test_among_phrase_states_how_many_members_the_source_holds(
    across: str, member_count: int, stated: str
) -> None:
    assert (
        phrasing.among_phrase(RankAmong.POPULATION, across=across, member_count=member_count)
        == stated
    )
    assert (
        phrasing.among_phrase(RankAmong.NAMED, across=across, member_count=member_count)
        == "ranks exactly the entities named"
    )


def test_among_phrase_states_named_versus_population() -> None:
    assert phrasing.among_phrase(RankAmong.NAMED) == "ranks exactly the entities named"
    assert "every entity the source holds" in phrasing.among_phrase(RankAmong.POPULATION)


def test_among_phrase_states_a_population_rank_over_across_with_no_focus() -> None:
    # Every member of the party `across` covers is unnamed, so none is a focus.
    assert (
        phrasing.among_phrase(RankAmong.POPULATION, across="country")
        == "ranks every country the source holds"
    )
    assert (
        phrasing.among_phrase(RankAmong.NAMED, across="country")
        == "ranks exactly the entities named"
    )


def test_per_period_phrase_states_whether_the_window_is_collapsed() -> None:
    assert "collapsed into one ranking" in phrasing.per_period_phrase(False, limited=False)
    assert "collapsed into one ranking" in phrasing.per_period_phrase(False, limited=True)
    assert "change in rank position" in phrasing.per_period_phrase(True, limited=False)
    limited = phrasing.per_period_phrase(True, limited=True)
    assert "measured values" in limited
    assert "rank position" not in limited


def test_limit_phrase_names_the_end_a_limit_keeps() -> None:
    assert phrasing.limit_phrase(RankDirection.HIGHEST_FIRST, 10) == "the top 10"
    assert phrasing.limit_phrase(RankDirection.LOWEST_FIRST, 2) == "the bottom 2"
    assert phrasing.limit_phrase(RankDirection.HIGHEST_FIRST, 1) == "the top"
    assert phrasing.limit_phrase(RankDirection.LOWEST_FIRST, 1) == "the bottom"


@pytest.mark.parametrize(
    ("limit", "place"),
    [
        (1, "1st"),
        (2, "2nd"),
        (3, "3rd"),
        (4, "4th"),
        (11, "11th"),
        (12, "12th"),
        (13, "13th"),
        (21, "21st"),
        (22, "22nd"),
        (23, "23rd"),
        (100, "100th"),
    ],
)
def test_shared_place_phrase_names_the_last_place_kept(limit: int, place: str) -> None:
    assert phrasing.shared_place_phrase(limit) == f"more when {place} place is shared"


def test_denominator_scope_phrase_states_per_axis_versus_pooled() -> None:
    assert (
        phrasing.denominator_scope_phrase(
            DenominatorScope.PER_AXIS,
            measure="lending",
            step="Total lending by year",
            narrowed_by="Narrow to Kenya",
            entity="Kenya",
        )
        == "each divided by its own total"
    )
    assert (
        phrasing.denominator_scope_phrase(
            DenominatorScope.POOLED,
            measure="GDP",
            step="Fetch GDP for Kenya",
            narrowed_by=None,
            entity=None,
        )
        == "each divided by one total of GDP"
    )


@pytest.mark.parametrize(
    ("narrowed_by", "entity", "phrase"),
    [
        (
            "Rank the top 5 recipients by ODA",
            None,
            'each divided by one total of ODA over the rows "Rank the top 5 recipients by ODA" '
            "keeps",
        ),
        (
            "Rank the top 5 recipients by ODA",
            "Kenya",
            'each divided by the total for Kenya in "Total ODA by year"',
        ),
        (None, "Kenya", 'each divided by the total for Kenya in "Total ODA by year"'),
    ],
)
def test_a_narrowed_pooled_total_is_named_by_the_step_that_narrows_it(
    narrowed_by: str | None, entity: str | None, phrase: str
) -> None:
    assert (
        phrasing.denominator_scope_phrase(
            DenominatorScope.POOLED,
            measure="ODA",
            step="Total ODA by year",
            narrowed_by=narrowed_by,
            entity=entity,
        )
        == phrase
    )


def test_denominator_scope_phrase_refuses_none() -> None:
    """`denominator_scope_phrase` refuses `None` itself, with no caller assert to strip.

    `ShareOfRequirement._denominator_scope_needs_an_axis` should reject a
    plan whose keyed share carries no `denominator_scope` before it ever
    reaches this function, but the field's type is `DenominatorScope |
    None`: nothing in `denominator_scope_phrase`'s own signature stops a
    caller from passing `None` anyway. This is a test of that refusal, not
    of the model validator -- it calls the function directly, past any
    model construction, so the validator upstream is not what is on trial.
    """
    with pytest.raises(ValueError, match="denominator_scope"):
        phrasing.denominator_scope_phrase(
            None, measure="lending", step="Fetch lending for China", narrowed_by=None, entity=None
        )


def test_datasets_of_walks_a_derived_requirement_back_to_its_dataset(
    decision: DecisionFactory,
) -> None:
    dataset = DatasetRequirement(
        id="dataset_a",
        decision_ids=["measure"],
        measure="lending",
        entities=[Entity(name="China", role="creditor")],
        period=Period(start=2024, end=2024),
    )
    rank = RankRequirement(
        id="rank_a",
        decision_ids=["measure", "population_scope"],
        input_id="dataset_a",
        direction=RankDirection.HIGHEST_FIRST,
        among=RankAmong.POPULATION,
    )
    population = dataset.model_copy(update={"id": "population", "measure": "population"})
    per_head = QuotientRequirement(
        id="per_head",
        decision_ids=["measure"],
        numerator=RequirementSide(input_id="dataset_a"),
        denominator=RequirementSide(input_id="population"),
    )
    plan = Plan(
        question="q",
        decisions=[
            decision(),
            decision(slot="population_scope", fixes=DecisionFixes(among=RankAmong.POPULATION)),
        ],
        evidence_requirements=[dataset, rank, population, per_head],
    )

    assert phrasing.datasets_of(rank, plan) == (dataset,)
    # A quotient's entities are its numerator's, as a share's are.
    assert phrasing.datasets_of(per_head, plan) == (dataset,)


def test_narrowing_step_walks_past_an_unlimited_rank_and_through_a_numerator(
    decision: DecisionFactory,
) -> None:
    dataset = DatasetRequirement(
        id="dataset",
        decision_ids=["measure"],
        measure="ODA",
        entities=[Entity(name="UK", role="donor"), Entity(name="Kenya", role="recipient")],
        period=Period(start=2015, end=2024),
    )
    totals = AggregateRequirement(
        id="totals",
        decision_ids=["measure"],
        input_id="dataset",
        reduction=Reduction.SUM,
        group_by=(GroupBy.PERIOD,),
    )
    ranked = RankRequirement(
        id="ranked",
        decision_ids=["measure"],
        input_id="dataset",
        direction=RankDirection.HIGHEST_FIRST,
        among=RankAmong.POPULATION,
        axis="recipient",
        reduction=Reduction.SUM,
        per_period=True,
    )
    whole = ShareOfRequirement(
        id="whole",
        decision_ids=["measure"],
        numerator=RequirementSide(input_id="dataset"),
        denominator=RequirementSide(input_id="totals"),
    )
    limited = ranked.model_copy(update={"id": "limited", "limit": 3})
    kenya = whole.model_copy(
        update={"id": "kenya", "numerator": RequirementSide(input_id="dataset", entity="Kenya")}
    )
    big = FilterRequirement(
        id="big",
        decision_ids=["measure"],
        input_id="dataset",
        predicates=(ValueThreshold(op="gt", value=1e9),),
    )
    per_total = QuotientRequirement(
        id="per_total",
        decision_ids=["measure"],
        numerator=RequirementSide(input_id="dataset"),
        denominator=RequirementSide(input_id="totals"),
        axis="recipient",
    )
    kenya_per_total = per_total.model_copy(
        update={
            "id": "kenya_per_total",
            "numerator": RequirementSide(input_id="dataset", entity="Kenya"),
            "axis": None,
        }
    )
    per_counterpart = per_total.model_copy(
        update={
            "id": "per_counterpart",
            "denominator": RequirementSide(input_id="big"),
            "denominator_axis": "recipient",
        }
    )
    top_of_big = limited.model_copy(update={"id": "top_of_big", "input_id": "big"})
    top_totals = totals.model_copy(update={"id": "top_totals", "input_id": "top_of_big"})
    plan = Plan(
        question="q",
        decisions=[decision(fixes=DecisionFixes(among=RankAmong.POPULATION))],
        evidence_requirements=[
            dataset,
            totals,
            ranked,
            limited,
            whole,
            kenya,
            big,
            per_total,
            kenya_per_total,
            per_counterpart,
            top_of_big,
            top_totals,
        ],
    )

    assert phrasing.narrowing_step(dataset, plan) is None
    assert phrasing.narrowing_step(ranked, plan) is None
    assert phrasing.narrowing_step(limited, plan) is limited
    assert phrasing.narrowing_step(whole, plan) is None
    assert phrasing.narrowing_step(kenya, plan) is kenya
    # A quotient over one total keeps its numerator's rows; one keyed on both
    # sides drops each entity its denominator does not hold.
    assert phrasing.narrowing_step(per_total, plan) is None
    assert phrasing.narrowing_step(kenya_per_total, plan) is kenya_per_total
    assert phrasing.narrowing_step(per_counterpart, plan) is per_counterpart
    # The filter and the rank both drop rows; the rank, nearest the total, is
    # the one whose output the total sums.
    assert phrasing.narrowing_step(top_totals, plan) is top_of_big


def test_a_rank_and_a_keyed_share_state_their_population_claims_on_the_card(
    decision: DecisionFactory,
) -> None:
    dataset_a = DatasetRequirement(
        id="dataset_a",
        decision_ids=["measure"],
        measure="external debt service paid",
        entities=[Entity(name="Kenya", role="debtor")],
        period=Period(start=2015, end=2024),
    )
    rank_req = RankRequirement(
        id="rank_a",
        decision_ids=["measure", "population_scope"],
        input_id="dataset_a",
        direction=RankDirection.HIGHEST_FIRST,
        among=RankAmong.POPULATION,
        axis="debtor",
        per_period=True,
    )
    dataset_b = DatasetRequirement(
        id="dataset_b",
        decision_ids=["measure"],
        measure="lending",
        entities=[Entity(name="China", role="creditor")],
        period=Period(start=2015, end=2024),
    )
    dataset_c = DatasetRequirement(
        id="dataset_c",
        decision_ids=["measure"],
        measure="GDP",
        entities=[Entity(name="Kenya", role="debtor")],
        period=Period(start=2015, end=2024),
    )
    share_req = ShareOfRequirement(
        id="share_a",
        decision_ids=["measure", "share_scope"],
        numerator=RequirementSide(input_id="dataset_b"),
        denominator=RequirementSide(input_id="dataset_c"),
        axis="creditor",
        denominator_scope=DenominatorScope.POOLED,
    )
    plan = Plan(
        question="q",
        decisions=[
            decision(),
            decision(slot="population_scope", fixes=DecisionFixes(among=RankAmong.POPULATION)),
            decision(
                slot="share_scope",
                fixes=DecisionFixes(denominator_scope=DenominatorScope.POOLED),
            ),
        ],
        evidence_requirements=[dataset_a, rank_req, dataset_b, dataset_c, share_req],
    )
    card_sentences = {row["id"]: row["sentence"] for row in requirement_rows(plan)}

    # `among` and `per_period` are the two rank fields a reader cannot infer
    # from a value. The card states both in words.
    for phrase in (
        phrasing.among_phrase(RankAmong.POPULATION),
        phrasing.per_period_phrase(True, limited=False),
    ):
        assert phrase in card_sentences["rank_a"]

    # A keyed share states its denominator scope by the denominator's measure.
    assert (
        phrasing.denominator_scope_phrase(
            DenominatorScope.POOLED,
            measure="GDP",
            step="Fetch GDP for Kenya",
            narrowed_by=None,
            entity=None,
        )
        in card_sentences["share_a"]
    )
