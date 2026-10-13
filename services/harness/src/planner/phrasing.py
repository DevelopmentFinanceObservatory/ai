"""The words a plan card states a rank's or a share_of's reading in.

`harness.plan_display` turns a plan into the card a user approves, and states
what a rank's `among`, `per_period` and `limit` mean and what a keyed
`share_of`'s `denominator_scope` means, population and series-shape claims a
person should read in words, not infer from a field name. This module holds
those words, plus `pluralize` and `entities_phrase`, the general building
blocks the builders reach for.

`datasets_of` walks a requirement back to the dataset requirements its value
ultimately reads, the same walk `plan_display._measure_of` makes for a
requirement's measure. `narrowing_step` walks the same lineage for the last
step that drops rows. `_measure_of` stays in `plan_display` because only the
card states a measure for a share_of's denominator.

`planner` must not import `harness`, so this module, imported from
`harness.plan_display`, lives under `planner/`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from planner.parties import across_party, singular
from planner.plan import (
    AggregateRequirement,
    ComparisonRequirement,
    DatasetRequirement,
    Entity,
    EvidenceRequirement,
    FilterRequirement,
    GrowthRequirement,
    JoinRequirement,
    Plan,
    QuotientRequirement,
    RankRequirement,
    ShareOfRequirement,
    SumOfRequirement,
)
from planner.records import (
    DenominatorScope,
    PeriodThreshold,
    PlaceLevel,
    PlaceType,
    RankAmong,
    RankDirection,
)

# Each level of place in words, in the singular: "Keep countries", "Rank the top 5 continents".
_LEVEL_NOUNS: dict[PlaceType, str] = {
    PlaceType.COUNTRY: "country",
    PlaceType.CONTINENT: "continent",
    PlaceType.ADMINISTRATIVE_AREA_1: "first-level subdivision",
    PlaceType.ADMINISTRATIVE_AREA_2: "second-level subdivision",
    PlaceType.CITY: "city",
    PlaceType.EUROSTAT_NUTS_1: "NUTS 1 region",
    PlaceType.EUROSTAT_NUTS_2: "NUTS 2 region",
    PlaceType.EUROSTAT_NUTS_3: "NUTS 3 region",
}

# What a requirement that names no place retrieves, in words.
EVERY_PLACE = "every place with data"


def pluralize(word: str) -> str:
    """The plural of a role phrase: "donor" gives "donors", "country" gives "countries".

    A phrase already ending in "s" reads as plural and stays as it is. A
    consonant before a final "y" takes "ies"; every other phrase takes "s".
    That covers every role noun the planner and dfo-apis's role vocabulary
    use, none of which is irregular.
    """
    if word.endswith("s"):
        return word
    if len(word) > 1 and word.endswith("y") and word[-2] not in "aeiou":
        return f"{word[:-1]}ies"
    return f"{word}s"


def level_noun(level: PlaceType) -> str:
    """A level of place in words, in the singular: "country", "first-level subdivision"."""
    return _LEVEL_NOUNS[level]


def entities_phrase(
    entities: Sequence[Entity],
    *,
    across: Sequence[str] = (),
    level: PlaceType | None = None,
    member_counts: Mapping[str, int] | None = None,
) -> str:
    """Join entity names in the order given, deduplicated, then "every {word}" for each `across` word.

    The parts read "a", "a and b" or "a, b and c", and an `across` word reads
    in the singular, so "donors" gives "every donor". `member_counts` maps an
    `across` word to how many members the selected source holds on the axis
    it covers, and a word with a count reads "all 50 recipients". A word
    naming the place party reads `EVERY_PLACE`, since such a retrieval holds
    every level of place and its count spans them all, or "every {level}"
    when a filter downstream of it keeps `level` (`kept_level`): "every
    country". A `DatasetRequirement` always names an entity or states
    `across`, so one of the two is set. An entity with `members` set reads
    "each member of {name}", so a card tells the set's members from the set's
    own figure.
    """
    parts: list[str] = []
    for entity in entities:
        name = f"each member of {entity.name}" if entity.members else entity.name
        if name not in parts:
            parts.append(name)
    for word in across:
        count = (member_counts or {}).get(word)
        if across_party(word) == "place":
            parts.append(EVERY_PLACE if level is None else f"every {level_noun(level)}")
        elif count is not None:
            parts.append(f"all {count:,} {pluralize(singular(word))}")
        else:
            parts.append(f"every {singular(word)}")
    if len(parts) == 1:
        return parts[0]
    return ", ".join(parts[:-1]) + f" and {parts[-1]}"


def datasets_of(requirement: EvidenceRequirement, plan: Plan) -> tuple[DatasetRequirement, ...]:
    """The dataset requirements `requirement`'s value ultimately reads.

    A derived requirement names no entities and no `across` of its own, so
    this walks `input_id` (or the relevant side) back to the dataset
    requirement, or requirements, it ultimately reads.
    """
    if isinstance(requirement, DatasetRequirement):
        return (requirement,)
    if isinstance(
        requirement, (AggregateRequirement, GrowthRequirement, RankRequirement, FilterRequirement)
    ):
        return datasets_of(plan.requirement(requirement.input_id), plan)
    if isinstance(requirement, ComparisonRequirement):
        return datasets_of(plan.requirement(requirement.minuend.input_id), plan) + datasets_of(
            plan.requirement(requirement.subtrahend.input_id), plan
        )
    if isinstance(requirement, SumOfRequirement):
        return datasets_of(plan.requirement(requirement.left.input_id), plan) + datasets_of(
            plan.requirement(requirement.right.input_id), plan
        )
    if isinstance(requirement, ShareOfRequirement | QuotientRequirement):
        return datasets_of(plan.requirement(requirement.numerator.input_id), plan)
    if isinstance(requirement, JoinRequirement):
        return datasets_of(plan.requirement(requirement.left_id), plan) + datasets_of(
            plan.requirement(requirement.right_id), plan
        )
    raise AssertionError(f"unhandled requirement kind: {requirement!r}")


def kept_level(requirement: EvidenceRequirement, plan: Plan) -> PlaceType | None:
    """The level of place a filter on `requirement`'s single-input lineage keeps, or `None`.

    Walks from `requirement` itself back through every step that reads one
    input, the way a rank's axis reaches its dataset, and returns the level
    of the first `PlaceLevel` predicate it meets.
    """
    current = requirement
    while isinstance(
        current, (AggregateRequirement, GrowthRequirement, RankRequirement, FilterRequirement)
    ):
        if isinstance(current, FilterRequirement):
            for predicate in current.predicates:
                if isinstance(predicate, PlaceLevel):
                    return predicate.level
        current = plan.requirement(current.input_id)
    return None


def narrowing_step(requirement: EvidenceRequirement, plan: Plan) -> EvidenceRequirement | None:
    """The last step from the dataset to `requirement`'s output that drops rows within a period.

    `requirement` itself counts, and `None` means no step drops rows. The
    step returned is the one nearest `requirement`, so its output is the set
    of rows that remains.

    An aggregate, a growth, a rank with no `limit` and a filter on years
    alone keep every row of each period they keep, and a share_of or a
    quotient_of keeps its numerator's rows unless that side narrows to one
    entity, so the walk goes on through each one's input. Every other step
    drops rows: a filter on entities or values, a rank with a `limit`, a
    comparison, a sum_of, a join, and a quotient_of stating
    `denominator_axis`, which drops each entity its denominator does not
    hold.
    """
    current = requirement
    while not isinstance(current, DatasetRequirement):
        if isinstance(current, ShareOfRequirement | QuotientRequirement):
            if current.numerator.entity is not None or (
                isinstance(current, QuotientRequirement) and current.denominator_axis is not None
            ):
                return current
            current = plan.requirement(current.numerator.input_id)
        elif (
            isinstance(current, (AggregateRequirement, GrowthRequirement))
            or (isinstance(current, RankRequirement) and current.limit is None)
            or (
                isinstance(current, FilterRequirement)
                and all(isinstance(predicate, PeriodThreshold) for predicate in current.predicates)
            )
        ):
            current = plan.requirement(current.input_id)
        else:
            return current
    return None


def among_phrase(
    among: RankAmong, *, across: str | None = None, member_count: int | None = None
) -> str:
    """What a rank's `among` claims about the entities it ranks, in words.

    `across` is the party the rank's axis covers whole, when that axis is one
    of a requirement's `across` words; no entity on it is named, so there is
    no focus.
    `member_count` is how many members the selected source holds on that
    axis, stated when known: "ranks all 55 donors the source holds".
    """
    if among is RankAmong.NAMED:
        return "ranks exactly the entities named"
    if across is not None and member_count is not None:
        return f"ranks all {member_count:,} {pluralize(singular(across))} the source holds"
    if across is not None:
        return f"ranks every {singular(across)} the source holds"
    return "ranks every entity the source holds, with the named entities as the focus"


def per_period_phrase(per_period: bool, *, limited: bool) -> str:
    """What a rank's series holds when `per_period` holds, in words: rank
    positions, or the measured values of each period's kept entities when
    the rank is `limited`."""
    if not per_period:
        return "the window is collapsed into one ranking"
    if limited:
        return "each period is ranked on its own, and its kept entities carry their measured values"
    return (
        "each period is ranked on its own, so a growth over it reads a change in rank "
        "position, not in the measured value"
    )


def limit_phrase(direction: RankDirection, limit: int) -> str:
    """Which end of a ranking a rank's `limit` keeps: "the top 10" or "the bottom 2".

    A limit of one reads "the top" or "the bottom", for a singular noun to follow.
    """
    end = "top" if direction is RankDirection.HIGHEST_FIRST else "bottom"
    return f"the {end}" if limit == 1 else f"the {end} {limit}"


def shared_place_phrase(limit: int) -> str:
    """What a rank's `limit` keeps when the value at its last place is shared, in words."""
    return f"more when {_ordinal(limit)} place is shared"


_ORDINAL_SUFFIXES = {1: "st", 2: "nd", 3: "rd"}  # codespell:ignore nd


def _ordinal(number: int) -> str:
    suffix = "th" if number % 100 in (11, 12, 13) else _ORDINAL_SUFFIXES.get(number % 10, "th")
    return f"{number}{suffix}"


def denominator_scope_phrase(
    scope: DenominatorScope | None,
    *,
    measure: str,
    step: str,
    narrowed_by: str | None,
    entity: str | None,
) -> str:
    """What a keyed share_of's `denominator_scope` claims, in words.

    `measure` is the denominator's measure as the card names it, `step` is
    the card label of the step that produces the denominator, and
    `narrowed_by` is the card label of the last step on its way from the
    dataset that drops rows (`narrowing_step`), or `None`. A pooled total
    sums whatever rows the denominator holds, which need not carry the
    share's axis at all (each donor's ODA over Kenya's GNI), so the phrase
    names the measure and never the axis. When a step drops rows, the phrase
    names that step, whose own card line states the filter or limit. When the
    denominator side narrows to one `entity`, it names the entity and the
    denominator step.

    `ShareOfRequirement._denominator_scope_needs_an_axis` guarantees a keyed
    share carries a `denominator_scope`, but that guarantee lives on the
    model, not the type: the field itself is `DenominatorScope | None`, so a
    caller that skips the model (or runs with assertions stripped, since
    Python drops a bare `assert` under `-O`) can still reach here with
    `None`. Refusing it here, rather than leaving each caller to assert it,
    means the check holds regardless of interpreter flags and a wrong
    reading of `denominator_scope` never reaches a plan card or an approval
    summary as a false "pooled".
    """
    if scope is None:
        raise ValueError(
            "denominator_scope is None for a keyed share_of; "
            "ShareOfRequirement._denominator_scope_needs_an_axis should have rejected this plan"
        )
    if scope is DenominatorScope.PER_AXIS:
        return "each divided by its own total"
    if entity is not None:
        return f'each divided by the total for {entity} in "{step}"'
    if narrowed_by is not None:
        return f'each divided by one total of {measure} over the rows "{narrowed_by}" keeps'
    return f"each divided by one total of {measure}"
