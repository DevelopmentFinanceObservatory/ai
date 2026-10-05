"""Turn a `Plan` into the words the plan card renders.

`step_rows`, `requirement_rows` and `operation_rows` are pure functions from
`planner.plan.Plan` to plain dicts. This module reads `planner.plan`,
`planner.operations`, `planner.records` and `planner.workflow` only, so it
can be written and tested before `harness.wire` carries the fields these
rows fill, and it stays usable from `harness.wire`, `harness.studio_translate`
and `planner.agent` alike without any of them importing each other through
here.

A step's `label` names the work a workflow step does and the entity it does
it to, e.g. "Fetch external debt service paid for Kenya" or "Rank debtors by
external debt service paid", never a bare tool name. `requirement_rows`
and `operation_rows` each carry a longer `sentence`, read off the same
requirement a step's label is read off; a rank or a keyed share_of sentence
states `among`, `axis`, `per_period`, `limit` and `denominator_scope` in
words, using `planner.phrasing` for the words themselves, so the card states
those claims the way the rest of the package does.

`_builder_label` and `_sentence` each dispatch through a `dict` keyed by the
exact requirement class, one builder function per `EvidenceRequirement` union
member; `_measure_of` walks the same eight by `isinstance`, as does
`planner.phrasing.datasets_of`. Every one of the three ends in an
`AssertionError` rather than a silent default: a ninth kind fails loudly
here rather than rendering a blank sentence on the wire. `_ranked_value`
passes every kind it does not transform to `_measure_of`, so it fails the
same way.
"""

from __future__ import annotations

import typing
from collections.abc import Sequence

from compiler.evidence import open_axis_for
from planner.parties import across_party, singular
from planner.phrasing import (
    among_phrase,
    datasets_of,
    denominator_scope_phrase,
    entities_phrase,
    kept_level,
    level_noun,
    limit_phrase,
    narrowing_step,
    per_period_phrase,
    pluralize,
    shared_place_phrase,
)
from planner.plan import (
    AggregateRequirement,
    ComparisonRequirement,
    DatasetRequirement,
    EvidenceRequirement,
    FilterRequirement,
    GrowthRequirement,
    JoinRequirement,
    Plan,
    RankRequirement,
    ShareOfRequirement,
)
from planner.records import (
    EntityIn,
    GroupBy,
    GrowthChange,
    PeriodThreshold,
    PlaceLevel,
    Predicate,
    Reduction,
    StatVarReading,
    ValueThreshold,
)
from planner.workflow import OpenEntityAxis, ReadingSetBy, SelectedSource

_REDUCTION_WORDS: dict[Reduction, str] = {
    Reduction.SUM: "total",
    Reduction.MEAN: "average",
    Reduction.MAX: "highest",
    Reduction.MIN: "lowest",
}

_THRESHOLD_WORDS: dict[str, str] = {
    "gt": "more than",
    "gte": "at least",
    "lt": "less than",
    "lte": "at most",
    "eq": "exactly",
    "ne": "not",
}


def step_rows(plan: Plan) -> list[dict[str, object]]:
    """One row per compiled workflow step, in the order the workflow runs them.

    Each row is the step's `id`, a composed `label` and `description`
    (`_label` and `_sentence`, read off the requirement the step's
    operation carries out), and `args`, the step's own literal arguments,
    unchanged.

    Requires a plan carrying a `translation` and a `workflow`. A step is
    something a compiled plan has, between `with_translation` and `approve`.
    """
    assert plan.translation is not None, "step_rows needs a plan carrying a translation"
    assert plan.workflow is not None, "step_rows needs a plan carrying a workflow"
    operations_by_id = {operation.id: operation for operation in plan.translation.operations}
    rows: list[dict[str, object]] = []
    for step in plan.workflow.steps:
        operation = operations_by_id[step.operation_id]
        requirement = plan.requirement(operation.requirement_id)
        rows.append(
            {
                "id": step.id,
                "label": _label(requirement, plan),
                "description": _sentence(requirement, plan),
                "args": dict(step.args),
            }
        )
    return rows


def requirement_rows(plan: Plan) -> list[dict[str, object]]:
    """One row per evidence requirement the plan states: id, kind, its decisions, and a sentence."""
    return [
        {
            "id": requirement.id,
            "kind": requirement.kind,
            "decision_ids": list(requirement.decision_ids),
            "sentence": _sentence(requirement, plan),
        }
        for requirement in plan.evidence_requirements
    ]


def operation_rows(plan: Plan) -> list[dict[str, object]]:
    """One row per operation the translator proposed: id, kind, its requirement, and a sentence.

    The sentence is read off the requirement the operation carries out, the
    same one `requirement_rows` reads for that requirement. An operation
    states no calculation field a requirement does not already state
    (`planner.proof.mismatches` is what proves the two agree).

    Requires a plan carrying a `translation`.
    """
    assert plan.translation is not None, "operation_rows needs a plan carrying a translation"
    rows: list[dict[str, object]] = []
    for operation in plan.translation.operations:
        requirement = plan.requirement(operation.requirement_id)
        rows.append(
            {
                "id": operation.id,
                "kind": operation.kind,
                "requirement_id": operation.requirement_id,
                "sentence": _sentence(requirement, plan),
            }
        )
    return rows


def _group_by_phrase(group_by: Sequence[GroupBy]) -> str:
    by_period = GroupBy.PERIOD in group_by
    by_entity = GroupBy.ENTITY in group_by
    if by_period and by_entity:
        return "by year and by entity"
    if by_period:
        return "by year"
    return "by entity"


def _predicate_phrase(predicate: Predicate) -> str:
    if isinstance(predicate, EntityIn):
        subject = predicate.role or "the entity"
        return f"{subject} is one of {', '.join(predicate.names)}"
    if isinstance(predicate, PeriodThreshold):
        return f"the year is {_THRESHOLD_WORDS[predicate.op]} {predicate.year}"
    if isinstance(predicate, ValueThreshold):
        return f"the value is {_THRESHOLD_WORDS[predicate.op]} {predicate.value}"
    if isinstance(predicate, PlaceLevel):
        return f"the place is a {level_noun(predicate.level)}"
    raise AssertionError(f"unhandled predicate kind: {predicate!r}")


def _predicates_phrase(predicates: Sequence[Predicate]) -> str:
    return " and ".join(_predicate_phrase(predicate) for predicate in predicates)


def _measure_of(requirement: EvidenceRequirement, plan: Plan) -> str:
    """The measure `requirement`'s value is ultimately a reading of, in words.

    A derived requirement restates no measure of its own, so this walks
    `input_id` (or the relevant side) back to the dataset requirement, or
    requirements, it ultimately reads.
    """
    if isinstance(requirement, DatasetRequirement):
        return requirement.measure
    if isinstance(requirement, (AggregateRequirement, GrowthRequirement, RankRequirement)):
        return _measure_of(plan.requirement(requirement.input_id), plan)
    if isinstance(requirement, FilterRequirement):
        return _measure_of(plan.requirement(requirement.input_id), plan)
    if isinstance(requirement, ComparisonRequirement):
        left = _measure_of(plan.requirement(requirement.left.input_id), plan)
        right = _measure_of(plan.requirement(requirement.right.input_id), plan)
        return f"{right} against {left}"
    if isinstance(requirement, ShareOfRequirement):
        return f"share of {_measure_of(plan.requirement(requirement.numerator.input_id), plan)}"
    if isinstance(requirement, JoinRequirement):
        left = _measure_of(plan.requirement(requirement.left_id), plan)
        right = _measure_of(plan.requirement(requirement.right_id), plan)
        return f"{left} joined with {right}"
    raise AssertionError(f"unhandled requirement kind: {requirement!r}")


def _ranked_value(requirement: EvidenceRequirement, plan: Plan) -> str:
    """What the frame `requirement` produces holds, in words, which a rank orders by.

    `_measure_of` names the measure a value is a reading of; this names the
    value after each transform, in the words that transform's own sentence
    uses: "the total of ODA", "the change in ODA from 2016 to 2020", "share
    of ODA". A filter keeps its input's values. A rank keeps them too, or
    their reduction when it states one, since the planner refuses a rank
    over rank positions; a rank's own label and sentence name the value it
    orders by through this. A comparison and a join feed nothing.
    """
    if isinstance(requirement, FilterRequirement):
        return _ranked_value(plan.requirement(requirement.input_id), plan)
    if isinstance(requirement, (AggregateRequirement, RankRequirement)):
        upstream = _ranked_value(plan.requirement(requirement.input_id), plan)
        if requirement.reduction is None:
            return upstream
        return f"the {_REDUCTION_WORDS[requirement.reduction]} of {upstream}"
    if isinstance(requirement, GrowthRequirement):
        upstream = _ranked_value(plan.requirement(requirement.input_id), plan)
        change = "percentage change" if requirement.change is GrowthChange.RELATIVE else "change"
        return (
            f"the {change} in {upstream} from {requirement.window_start} to "
            f"{requirement.window_end}"
        )
    if isinstance(requirement, ShareOfRequirement):
        return f"share of {_ranked_value(plan.requirement(requirement.numerator.input_id), plan)}"
    return _measure_of(requirement, plan)


def _axis_role(requirement: RankRequirement, plan: Plan) -> str | None:
    """The role a rank's axis names: the stated axis, or the sole role its entities or
    `across` carry when the axis is left to infer from a single-role input."""
    if requirement.axis is not None:
        return requirement.axis
    datasets = datasets_of(requirement, plan)
    roles = sorted(
        {entity.role for dataset in datasets for entity in dataset.entities if entity.role}
        | {dataset.across for dataset in datasets if dataset.across is not None}
    )
    return roles[0] if len(roles) == 1 else None


def _axis_word(requirement: RankRequirement, plan: Plan, *, plural: bool = True) -> str:
    """What a rank's axis names, in words: its role, in the plural when `plural` holds and
    the singular otherwise, or "entities" (or "entity") when it has none. A place axis a
    filter has kept one level of names that level: "countries", "continents"."""
    role = _place_level_word(requirement, plan) or _axis_role(requirement, plan)
    if role is None:
        return "entities" if plural else "entity"
    return pluralize(role) if plural else singular(role)


def _place_level_word(requirement: RankRequirement, plan: Plan) -> str | None:
    """The level a filter keeps on a rank's place axis, in words, or `None` when no filter
    keeps one or the axis is not the place."""
    role = _axis_role(requirement, plan)
    level = kept_level(plan.requirement(requirement.input_id), plan)
    if level is None or role is None or across_party(role) != "place":
        return None
    return level_noun(level)


def _kept_phrase(requirement: RankRequirement, limit: int, plan: Plan) -> str:
    """The entities a rank's `limit` keeps, in words: "the top 10 donors" or "the bottom debtor"."""
    noun = _axis_word(requirement, plan, plural=limit > 1)
    return f"{limit_phrase(requirement.direction, limit)} {noun}"


def _label_dataset(requirement: DatasetRequirement, plan: Plan) -> str:
    parties = entities_phrase(requirement.entities, across=requirement.across)
    return f"Fetch {requirement.measure} for {parties}"


def _label_aggregate(requirement: AggregateRequirement, plan: Plan) -> str:
    upstream = plan.requirement(requirement.input_id)
    word = _REDUCTION_WORDS[requirement.reduction].capitalize()
    return f"{word} {_measure_of(upstream, plan)} {_group_by_phrase(requirement.group_by)}"


def _label_growth(requirement: GrowthRequirement, plan: Plan) -> str:
    upstream = plan.requirement(requirement.input_id)
    return (
        f"Growth in {_measure_of(upstream, plan)}, "
        f"{requirement.window_start}-{requirement.window_end}"
    )


def _label_comparison(requirement: ComparisonRequirement, plan: Plan) -> str:
    left = plan.requirement(requirement.left.input_id)
    right = plan.requirement(requirement.right.input_id)
    return f"Compare {_measure_of(right, plan)} to {_measure_of(left, plan)}"


def _label_rank(requirement: RankRequirement, plan: Plan) -> str:
    ranked = (
        _axis_word(requirement, plan)
        if requirement.limit is None
        else _kept_phrase(requirement, requirement.limit, plan)
    )
    return f"Rank {ranked} by {_ranked_value(requirement, plan)}"


def _label_share_of(requirement: ShareOfRequirement, plan: Plan) -> str:
    numerator = plan.requirement(requirement.numerator.input_id)
    label = f"Share of {_measure_of(numerator, plan)}"
    if requirement.axis is not None:
        label += f" per {requirement.axis}"
    return label


def _label_join(requirement: JoinRequirement, plan: Plan) -> str:
    left = plan.requirement(requirement.left_id)
    right = plan.requirement(requirement.right_id)
    return f"Join {_measure_of(left, plan)} with {_measure_of(right, plan)}"


def _label_filter(requirement: FilterRequirement, plan: Plan) -> str:
    entity_names = sorted(
        {
            name
            for predicate in requirement.predicates
            if isinstance(predicate, EntityIn)
            for name in predicate.names
        }
    )
    if entity_names:
        return f"Narrow to {', '.join(entity_names)}"
    levels = [p.level for p in requirement.predicates if isinstance(p, PlaceLevel)]
    if levels:
        return f"Keep {pluralize(level_noun(levels[0]))}"
    upstream = plan.requirement(requirement.input_id)
    return f"Filter {_measure_of(upstream, plan)}"


# One builder per `EvidenceRequirement` union member, keyed by the exact
# class. The discriminated union never constructs a subclass, so `type()`
# is an exact lookup. A kind added to the union and left out here fails at
# the `KeyError` below rather than rendering a blank label.
_LABEL_BUILDERS: dict[type, typing.Callable[[typing.Any, Plan], str]] = {
    DatasetRequirement: _label_dataset,
    AggregateRequirement: _label_aggregate,
    GrowthRequirement: _label_growth,
    ComparisonRequirement: _label_comparison,
    RankRequirement: _label_rank,
    ShareOfRequirement: _label_share_of,
    JoinRequirement: _label_join,
    FilterRequirement: _label_filter,
}


def _label(requirement: EvidenceRequirement, plan: Plan) -> str:
    """A short label naming the work a step does and the entity it does it to.

    Never a bare tool name: every builder names the requirement's own
    measure and entities, walking a derived requirement back to the
    dataset requirement it ultimately reads. Two steps can read the same,
    e.g. a total over the top 5 recipients and a total over all of them
    are both "Total ODA by year", so a label the plan's other steps share
    ends in its place among them, "(1)" or "(2)", in the order the card
    lists the steps. A share_of sentence that quotes a label then names
    exactly one step.
    """
    label = _builder_label(requirement, plan)
    namesakes = [other.id for other in _card_order(plan) if _builder_label(other, plan) == label]
    if len(namesakes) == 1:
        return label
    return f"{label} ({namesakes.index(requirement.id) + 1})"


def _builder_label(requirement: EvidenceRequirement, plan: Plan) -> str:
    try:
        builder = _LABEL_BUILDERS[type(requirement)]
    except KeyError:
        raise AssertionError(f"unhandled requirement kind: {requirement!r}") from None
    return builder(requirement, plan)


def _card_order(plan: Plan) -> list[EvidenceRequirement]:
    """`plan`'s requirements in the order the card lists their steps.

    A compiled plan lists them in workflow step order, the order `step_rows`
    follows, and any requirement no step carries out after them; before
    compile, in the order the plan states them.
    """
    if plan.translation is None or plan.workflow is None:
        return list(plan.evidence_requirements)
    operations_by_id = {operation.id: operation for operation in plan.translation.operations}
    stepped = [operations_by_id[step.operation_id].requirement_id for step in plan.workflow.steps]
    unstepped = [r.id for r in plan.evidence_requirements if r.id not in stepped]
    return [plan.requirement(requirement_id) for requirement_id in [*stepped, *unstepped]]


def _sentence_dataset(requirement: DatasetRequirement, plan: Plan) -> str:
    """The retrieval in words, with how it reads a family of StatVars. The
    selected source's `describe` definition stays out: for a StatVar it is
    the graph's constraint encoding ("mp=count,pt=Person"), which the
    translator model reads and a user cannot."""
    source = _selected_source(requirement.id, plan)
    reading = _reading_clause(source) if source is not None else None
    parties = entities_phrase(requirement.entities, across=requirement.across)
    return (
        f"Retrieves {requirement.measure} for {parties}, "
        f"{requirement.period.start}-{requirement.period.end}"
        + (f", {reading}." if reading else ".")
    )


def _selected_source(requirement_id: str, plan: Plan) -> SelectedSource | None:
    """The source the compiled plan retrieves `requirement_id` from.

    `None` before compile (`plan.workflow` unset) or when nothing was bound
    for this requirement.
    """
    if plan.workflow is None:
        return None
    step_id = next(
        (
            binding.step_id
            for binding in plan.workflow.evidence_bindings
            if binding.requirement_id == requirement_id
        ),
        None,
    )
    return plan.workflow.source_of(step_id) if step_id is not None else None


def _population_count(requirement: DatasetRequirement, plan: Plan) -> int | None:
    """How many members the selected source holds on the axis `requirement.across` covers.

    `None` before compile, when no source is selected yet, and when
    `compiler.evidence.open_axis_for` matches the word to no open axis.
    """
    source = _selected_source(requirement.id, plan)
    if source is None or requirement.across is None:
        return None
    axis = open_axis_for(requirement.across, source.open_entity_axes)
    return axis.member_count if isinstance(axis, OpenEntityAxis) else None


def _reading_clause(source: SelectedSource) -> str | None:
    """How the retrieval reads `source`'s family of StatVars, or `None` for a source binding one.

    Counts the family as the source does: the members `search` reports,
    which exceed the listed StatVars when the list was capped. A family
    `search` confirmed fewer than two members of is named without a count,
    since its retrieval holds every member with data. Says when the reading
    is the compiler's default (`ReadingSetBy.RULE`), so the user knows the
    plan can be amended to read it the other way.

    A family read separately states no clause when its list holds one
    StatVar, is complete (`stat_var_truncated` is false) and the family
    varies along no open axis: the card names one indicator, and there is
    nothing on it to keep apart. The retrieval's arguments still state the
    reading. A list of one that is capped, or that does not say whether it
    is, a longer list, and a family with an open axis all keep the clause.
    """
    if source.stat_var_reading is None:
        return None
    if (
        source.stat_var_reading is StatVarReading.SEPARATE
        and len(source.stat_vars) == 1
        and source.stat_var_truncated is False
        and not source.open_axes
    ):
        return None
    count = max(len(source.stat_vars), source.stat_var_member_count or 0)
    summed = source.stat_var_reading is StatVarReading.SUM
    if count < 2:
        clause = "summed over its StatVars" if summed else "kept as separate StatVars"
    else:
        clause = (
            f"summed over its {count:,} StatVars"
            if summed
            else f"kept as {count:,} separate StatVars"
        )
    return f"{clause} by default" if source.stat_var_reading_set_by is ReadingSetBy.RULE else clause


def _sentence_aggregate(requirement: AggregateRequirement, plan: Plan) -> str:
    upstream = plan.requirement(requirement.input_id)
    word = _REDUCTION_WORDS[requirement.reduction]
    return (
        f"The {word} of {_measure_of(upstream, plan)}, grouped "
        f"{_group_by_phrase(requirement.group_by)}."
    )


def _sentence_growth(requirement: GrowthRequirement, plan: Plan) -> str:
    upstream = plan.requirement(requirement.input_id)
    change_text = (
        "a percentage change"
        if requirement.change is GrowthChange.RELATIVE
        else "an absolute change, in the input's own unit"
    )
    return (
        f"The change in {_measure_of(upstream, plan)} from {requirement.window_start} to "
        f"{requirement.window_end}, read as {change_text}."
    )


def _sentence_comparison(requirement: ComparisonRequirement, plan: Plan) -> str:
    left = plan.requirement(requirement.left.input_id)
    right = plan.requirement(requirement.right.input_id)
    return (
        f"{_measure_of(right, plan).capitalize()} minus {_measure_of(left, plan)}, at every "
        "period the two share."
    )


def _sentence_rank(requirement: RankRequirement, plan: Plan) -> str:
    role = _axis_role(requirement, plan)
    covering = next(
        (
            dataset
            for dataset in datasets_of(requirement, plan)
            if dataset.across is not None and dataset.across == role
        ),
        None,
    )
    # The source counts every place on a place axis, so a level kept from it
    # is named without a count.
    level = _place_level_word(requirement, plan)
    among_text = among_phrase(
        requirement.among,
        across=(level or role) if covering is not None else None,
        member_count=(
            _population_count(covering, plan) if covering is not None and level is None else None
        ),
    )
    per_period_text = per_period_phrase(
        requirement.per_period, limited=requirement.limit is not None
    )
    sentence = f"Ranks by {_ranked_value(requirement, plan)}: {among_text}; {per_period_text}"
    if requirement.limit is None:
        return f"{sentence}."
    kept = _kept_phrase(requirement, requirement.limit, plan)
    if requirement.per_period:
        kept += " in each period"
    return f"{sentence}; keeps {kept}, {shared_place_phrase(requirement.limit)}."


def _sentence_share_of(requirement: ShareOfRequirement, plan: Plan) -> str:
    numerator = plan.requirement(requirement.numerator.input_id)
    measure = _measure_of(numerator, plan)
    denominator = plan.requirement(requirement.denominator.input_id)
    if requirement.axis is not None:
        narrowing = narrowing_step(denominator, plan)
        scope_text = denominator_scope_phrase(
            requirement.denominator_scope,
            measure=_measure_of(denominator, plan),
            step=_label(denominator, plan),
            narrowed_by=None if narrowing is None else _label(narrowing, plan),
            entity=requirement.denominator.entity,
        )
        return f"Share of {measure} per {requirement.axis}, {scope_text}."
    # A pooled share over a frame that names no place reads one level of it, which
    # the denominator's level filter keeps.
    level = kept_level(denominator, plan)
    parties = "; ".join(
        dict.fromkeys(
            entities_phrase(dataset.entities, across=dataset.across, level=level)
            for dataset in datasets_of(denominator, plan)
        )
    )
    return (
        f"Share of {measure} as one figure overall, divided by "
        f"{_measure_of(denominator, plan)} ({parties})."
    )


def _sentence_join(requirement: JoinRequirement, plan: Plan) -> str:
    left = plan.requirement(requirement.left_id)
    right = plan.requirement(requirement.right_id)
    return (
        f"Joins {_measure_of(left, plan)} with {_measure_of(right, plan)} on the periods "
        "and entities they share."
    )


def _sentence_filter(requirement: FilterRequirement, plan: Plan) -> str:
    return f"Keeps rows where {_predicates_phrase(requirement.predicates)}."


# One builder per `EvidenceRequirement` union member, the same discipline
# `_LABEL_BUILDERS` uses.
_SENTENCE_BUILDERS: dict[type, typing.Callable[[typing.Any, Plan], str]] = {
    DatasetRequirement: _sentence_dataset,
    AggregateRequirement: _sentence_aggregate,
    GrowthRequirement: _sentence_growth,
    ComparisonRequirement: _sentence_comparison,
    RankRequirement: _sentence_rank,
    ShareOfRequirement: _sentence_share_of,
    JoinRequirement: _sentence_join,
    FilterRequirement: _sentence_filter,
}


def _sentence(requirement: EvidenceRequirement, plan: Plan) -> str:
    """One prose sentence stating what approving `requirement` approves.

    A rank sentence states `among`, `per_period` and any `limit`. A keyed share_of
    sentence states `axis` and `denominator_scope`. These are population and
    series-shape claims a person reads in words rather than inferring from a
    field name, worded by `planner.phrasing`.
    """
    try:
        builder = _SENTENCE_BUILDERS[type(requirement)]
    except KeyError:
        raise AssertionError(f"unhandled requirement kind: {requirement!r}") from None
    return builder(requirement, plan)
