"""Turn a `Plan` into the words the plan card renders.

`step_rows`, `requirement_rows`, `operation_rows` and `ask_rows` are pure
functions from `planner.plan.Plan` to plain dicts, `shown_decisions` the
decisions the card and the plan record list, `with_source_names` a
decision's text with each spec_id in it named in words, `shown_plan` the plan
as the model wrote it with the compiled parts folded to match, and
`answer_scopes` how many figures each answering requirement's result gives,
read off its shape. A step whose result answers ends its description with
that count. This module
reads `planner.plan`, `planner.operations`, `planner.records`,
`planner.shapes`, `planner.workflow` and `harness.sentences` only, so it can
be written and tested before `harness.wire` carries the fields these rows
fill, and it stays usable from `harness.wire`, `harness.studio_translate`
and `planner.agent` alike without any of them importing each other through
here.

The plan card reads its own shorter text: `card_rows` gives each step a
short label naming its work and the steps it reads, keeps the detail in its
description, and marks the one step whose result answers (`answer_step`);
`card_title` titles the card by that step's ask; `card_summary` says in one
sentence what the answer counts, for whom, when and from which source;
`card_choice_text`, `card_decision_text` and `CHOICE_NAMES` give a
decision's words; and `choice_id` gives a scope choice's row the id every
switch keeps.

In `step_rows`, which the plan record and the planner read, a step's `label`
names the work a workflow step does and the entity it does
it to, e.g. "Fetch external debt service paid for Kenya" or "Rank debtors by
external debt service paid", never a bare tool name. `requirement_rows`
and `operation_rows` each carry a longer `sentence`, read off the same
requirement a step's label is read off; a rank or a keyed share_of sentence
states `among`, `axis`, `per_period`, `limit` and `denominator_scope` in
words, using `planner.phrasing` for the words themselves, so the card states
those claims the way the rest of the package does. A keyed quotient_of
sentence states `axis` and whether each entity divides by its own
counterpart (`denominator_axis`) or by the denominator's one total.

`_builder_label` and `_sentence` each dispatch through a `dict` keyed by the
exact requirement class, one builder function per `EvidenceRequirement` union
member; `_measure_of` walks the same members by `isinstance`, as does
`planner.phrasing.datasets_of`. Every one of the three ends in an
`AssertionError` rather than a silent default: a kind none of them names
fails loudly here rather than rendering a blank sentence on the wire. `_ranked_value`
passes every kind it does not transform to `_measure_of`, so it fails the
same way.
"""

from __future__ import annotations

import json
import re
import typing
from collections import Counter
from collections.abc import Collection, Iterator, Mapping, Sequence

from compiler.evidence import open_axes_for, resolve_group, resolve_name
from compiler.shapes import breakdown_level
from compiler.sources import stat_var_count
from planner.parties import across_party, flow_side, party_role, role_party, singular
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
    Decision,
    Entity,
    EvidenceRequirement,
    FilterRequirement,
    GrowthRequirement,
    JoinRequirement,
    Plan,
    QuotientRequirement,
    RankRequirement,
    RequirementSide,
    ResolvedBy,
    ShareOfRequirement,
    SumOfRequirement,
    answering_results,
    authored,
    authored_text,
    copy_origins,
    dataset_of,
    frame_shapes,
    inputs_of,
    is_scope_choice,
    is_search_pin,
    side_entities,
)
from planner.records import (
    Breakdown,
    DenominatorScope,
    EntityIn,
    EvidenceIssue,
    GroupBy,
    GrowthChange,
    Period,
    PeriodThreshold,
    PlaceLevel,
    PlaceType,
    Predicate,
    Reduction,
    RollingWindow,
    StatVarReading,
    ValueThreshold,
)
from planner.shapes import FrameShape, RoleDimension, TimeLabel, ValueKind
from planner.workflow import ReadingSetBy, SelectedSource, Workflow, WorkflowStep

from harness.sentences import DECLINE_REASONS, PlanWords

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


def shown_plan(plan: Plan) -> Plan:
    """`plan` as the model wrote it, its compiled parts folded to match.

    The plan as stored runs a dataset read from its first year beside a copy
    of each step between it and its rolling-start readers
    (`Plan.rolling_copies`). The card, the plan record and the planner's
    digest name the model's steps only. This is `planner.plan.authored`
    without the copies' operations, workflow steps, evidence bindings and
    the sources compiled for those bindings, and with a warning that names
    a copy naming its origin, once.

    The result is a view for reading and is not validated. Shapes, which
    need the first years, are read off `plan` itself (`answer_scopes`).
    """
    origins = copy_origins(plan.rolling_copies)
    shown = authored(plan)
    if not origins:
        return shown
    update: dict[str, object] = {}
    copied_operations: set[str] = set()
    if plan.translation is not None:
        copied_operations = {
            operation.id
            for operation in plan.translation.operations
            if operation.requirement_id in origins
        }
        update["translation"] = plan.translation.model_copy(
            update={
                "operations": tuple(
                    operation
                    for operation in plan.translation.operations
                    if operation.id not in copied_operations
                )
            }
        )
    if plan.workflow is not None:
        update["workflow"] = _without_copies(plan.workflow, origins, copied_operations)
    return shown.model_copy(update=update)


def _without_copies(
    workflow: Workflow, origins: Mapping[str, str], copied_operations: Collection[str]
) -> Workflow:
    """`workflow` without the steps carrying out `copied_operations` and the evidence bound to a copy.

    A binding's source is the one at its own position in `sources`
    (`Workflow.source_of`), so a copy's binding takes that source with it.

    A warning names the requirement the model wrote, and its prose names no
    copy (`planner.plan.authored_text`). Where a copy holds warnings of a
    code, they stand in for its origin's of that code: the copy reads the
    origin's stated years over a retrieval from the dataset's first year,
    and its finding says both (`compiler.evidence._asked_for`).
    """
    copied = {
        index
        for index, binding in enumerate(workflow.evidence_bindings)
        if binding.requirement_id in origins
    }
    sources = tuple(
        source
        for index, source in enumerate(workflow.sources)
        if not (index in copied and workflow.evidence_bindings[index].spec_id == source.spec_id)
    )
    on_copies = {
        (origins[issue.requirement_id], issue.code)
        for issue in workflow.warnings
        if issue.requirement_id in origins
    }
    warnings: list[EvidenceIssue] = []
    for issue in workflow.warnings:
        if (issue.requirement_id, issue.code) in on_copies:
            continue
        named = issue.model_copy(
            update={
                "requirement_id": origins.get(issue.requirement_id, issue.requirement_id),
                "detail": authored_text(issue.detail, origins),
            }
        )
        if named not in warnings:
            warnings.append(named)
    return workflow.model_copy(
        update={
            "steps": tuple(
                step for step in workflow.steps if step.operation_id not in copied_operations
            ),
            "sources": sources,
            "evidence_bindings": tuple(
                binding
                for index, binding in enumerate(workflow.evidence_bindings)
                if index not in copied
            ),
            "warnings": tuple(warnings),
        }
    )


def step_rows(plan: Plan) -> list[dict[str, object]]:
    """One row per compiled workflow step, in the order the workflow runs them.

    A step carrying out a copy the split added (`shown_plan`) has no row.
    Each row is the step's `id`, a composed `label` and `description`
    (`_label` and `_sentence`, read off the requirement the step's
    operation carries out), and `args`, the step's own literal arguments,
    unchanged. The description of a step whose result answers
    (`answer_scopes`) ends with how many figures that result gives, so the
    card the user approves states the result's scope.

    Requires a plan carrying a `translation` and a `workflow`. A step is
    something a compiled plan has, between `with_translation` and `approve`.
    """
    assert plan.translation is not None, "step_rows needs a plan carrying a translation"
    assert plan.workflow is not None, "step_rows needs a plan carrying a workflow"
    shown = shown_plan(plan)
    return [
        {
            "id": step.id,
            "label": _label(requirement, shown),
            "description": _with_result(_sentence(requirement, shown), result),
            "args": dict(step.args),
        }
        for step, requirement, result in _stepped(plan, shown)
    ]


def card_rows(plan: Plan) -> list[dict[str, object]]:
    """One row per step of the plan card, the steps `step_rows` lists in its order.

    Each row is the step's `id`, a short `label`, a `description` holding
    the detail, `args` as `step_rows` gives them, and `gives_answer`, true
    on the one step whose result answers the question (`answer_step`).

    A retrieval's label names what it fetches and for whom: "Fetch life
    expectancy at birth for Kenya". What a scope decision settles stays out
    of it, since the card states that once, in the summary (`card_summary`)
    and the choices (`_concepts`). Any other step's label names its work
    and the concepts it reads: "Total health", "Health as a share of
    official development assistance". A switch re-pins a retrieval and
    leaves every label true. A reduction says what its result keeps one
    figure per, read off the result's shape, so a total of one year reads
    without "by year".

    A retrieval's description is Search's account of its source
    (`SelectedSource.sentence`), else the requirement in words; any other
    step's is its `_sentence`. A step whose result answers ends its
    description with how many figures the result gives (`answer_scopes`).

    Requires a compiled plan, as `step_rows` does.
    """
    assert plan.translation is not None, "card_rows needs a plan carrying a translation"
    assert plan.workflow is not None, "card_rows needs a plan carrying a workflow"
    shapes = frame_shapes(plan)
    shown = shown_plan(plan)
    concepts = _concepts(shown).names
    answer = _answer_step(plan, shown)
    rows: list[dict[str, object]] = []
    for step, requirement, result in _stepped(plan, shown):
        source = _selected_source(requirement.id, shown)
        if isinstance(requirement, DatasetRequirement) and source is not None:
            description = _full_stop(source.sentence)
        else:
            description = _sentence(requirement, shown)
        rows.append(
            {
                "id": step.id,
                "label": _card_label(requirement, shown, concepts=concepts, shapes=shapes),
                "description": _with_result(description, result),
                "args": dict(step.args),
                "gives_answer": step.id == answer,
            }
        )
    return rows


def _stepped(
    plan: Plan, shown: Plan
) -> Iterator[tuple[WorkflowStep, EvidenceRequirement, str | None]]:
    """Each workflow step of `shown`, `plan` folded (`shown_plan`), in the order the
    workflow runs them, with the requirement its operation carries out and the
    scope its result gives where that result answers (`answer_scopes`), else `None`.
    The row core `step_rows` and `card_rows` share; requires a compiled plan."""
    assert shown.translation is not None
    assert shown.workflow is not None
    scopes = answer_scopes(plan)
    operations_by_id = {operation.id: operation for operation in shown.translation.operations}
    for step in shown.workflow.steps:
        requirement = shown.requirement(operations_by_id[step.operation_id].requirement_id)
        yield step, requirement, scopes.get(requirement.id)


def _with_result(description: str, result: str | None) -> str:
    """`description` ending with the scope `result` an answering step's result gives."""
    return description if result is None else f"{description} Result: {result}."


def plan_words(
    plan: Plan,
    *,
    shown_as: Plan | None = None,
    pinned: Mapping[str, SelectedSource] | None = None,
) -> PlanWords:
    """How a sentence about `plan` names its requirements
    (`harness.sentences.refusal_sentence`), and the ids each reads (`inputs_of`).

    A retrieval reads by the name the card gives it (`_concepts`), read off
    the source it was compiled from. A plan not compiled, such as a revision
    that failed to compile, reads each retrieval off the source it pins
    (`_pinned_sources`): the one the card's plan `shown_as` reads where the
    card's retrieval reads the same, else the one Search answered for its
    pin (`pinned`, by spec_id). Where `shown_as` is given, a retrieval it
    holds with the same naming facets (`_namings`) reads by its name
    there (`_carried_names`), and any other retrieval is named around those
    names. Any other requirement reads by the retrievals it reads
    (`_concept_of`): "health as a share of official development assistance".
    Each retrieval whose name states the option it holds on a scope axis
    lists that axis in `PlanWords.options`.
    """
    shown = shown_plan(plan)
    prior = shown_plan(shown_as) if shown_as is not None else None
    sources = _pinned_sources(shown, shown_as=prior, pinned=pinned or {})
    namings = _namings(shown, sources=sources)
    carried = _carried_names(namings, prior) if prior is not None else None
    card = _concepts(shown, carried=carried, namings=namings)
    return PlanWords(
        names={
            requirement.id: _concept_of(requirement, shown, card.names)
            for requirement in shown.evidence_requirements
        },
        inputs={
            requirement.id: inputs_of(requirement) for requirement in shown.evidence_requirements
        },
        options=card.options,
    )


def answer_step(plan: Plan) -> str | None:
    """The id of the one step whose result answers the question, or `None` before compile.

    Of the steps whose results answer an ask (`ask_rows`' `answered_by`),
    the last the workflow runs, since a later answering step builds on what
    an earlier one gives (a change over the years it totals, a share of
    the totals it divides). A plan answering no ask by name takes the last
    step whose result answers it (`planner.plan.answering_results`).
    """
    return _answer_step(plan, shown_plan(plan))


def _answer_step(plan: Plan, shown: Plan) -> str | None:
    """`answer_step` of `plan`, folded as `shown` (`shown_plan`)."""
    requirement_id = _answer_requirement(plan, shown)
    if requirement_id is None or plan.translation is None:
        return None
    return next(
        (op.id for op in plan.translation.operations if op.requirement_id == requirement_id),
        None,
    )


def _answer_requirement(plan: Plan, shown: Plan) -> str | None:
    """The requirement `answer_step` carries out, the last answering one in
    `_step_order` of `shown`, `plan` folded (`shown_plan`), or `None` before
    compile."""
    if plan.translation is None or plan.workflow is None:
        return None
    answering = answering_results(plan)
    order = _step_order(shown)
    asked = [rid for rid in order if answering.get(rid) is not None]
    candidates = asked or [rid for rid in order if rid in answering]
    return candidates[-1] if candidates else None


def _step_order(shown: Plan) -> list[str]:
    """The requirement ids of `shown`, compiled, in the order its workflow runs the
    steps carrying them out, the order `step_rows` follows, and any requirement no
    step carries out after them."""
    assert shown.translation is not None and shown.workflow is not None
    operations_by_id = {operation.id: operation for operation in shown.translation.operations}
    stepped = [operations_by_id[step.operation_id].requirement_id for step in shown.workflow.steps]
    return [*stepped, *(r.id for r in shown.evidence_requirements if r.id not in stepped)]


# The plan card's title when no ask of the user's titles it (`card_title`).
_PLAN_TITLE = "Research plan"


def card_title(plan: Plan) -> str:
    """The plan card's title: the ask the answering step answers (`answer_step`), in the
    user's words with its first letter raised; the first ask when that step answers
    none by name; `_PLAN_TITLE` for a plan given no asks, or whose ask holds no
    words."""
    requirement_id = _answer_requirement(plan, shown_plan(plan))
    number = answering_results(plan).get(requirement_id) if requirement_id else None
    if number is None and plan.asks:
        number = 0
    ask = plan.asks[number].strip() if number is not None else ""
    if not ask:
        return _PLAN_TITLE
    return f"{ask[0].upper()}{ask[1:]}"


def card_summary(plan: Plan) -> str | None:
    """One sentence saying what the answer counts, for whom, when and from which source.

    Read off the retrievals the answering requirement reads (`datasets_of`),
    each as `_Counted` states it. A share names its numerator's retrievals,
    then "as a share of" its denominator's, and a quotient "divided by". So does
    a step reading one input over a share or quotient, such as a rank of
    shares or a total of quotients (`_ratio_beneath`). `None` too where the
    answer is a comparison, sum or join over ratios (`_reads_ratio`): the
    retrievals it reads would name each ratio's numerator alone and lose the
    "share of", so the card shows its title only. A fact
    every retrieval shares, the basis, the years or the source, is stated
    once at the head; a retrieval then reads by its sector and its parties,
    or "the total" for its parties when it names no sector. `None` before
    compile.
    """
    shown = shown_plan(plan)
    requirement_id = _answer_requirement(plan, shown)
    if requirement_id is None:
        return None
    requirement = _ratio_beneath(shown.requirement(requirement_id), shown)
    if isinstance(requirement, ShareOfRequirement | QuotientRequirement):
        joiner = (
            ", as a share of " if isinstance(requirement, ShareOfRequirement) else ", divided by "
        )
        sides = [
            datasets_of(shown.requirement(requirement.numerator.input_id), shown),
            datasets_of(shown.requirement(requirement.denominator.input_id), shown),
        ]
    elif _reads_ratio(requirement, shown):
        return None
    else:
        joiner = ""
        sides = [datasets_of(requirement, shown)]
    # Keyed by side, so a share whose two sides read one retrieval keeps both.
    counted = [[_counted(dataset, shown) for dataset in side] for side in sides]
    facts = [fact for side in counted for fact in side]
    if len(facts) == 1:
        (only,) = facts
        # A basis whose label holds a comma ("gross disbursements, current USD")
        # closes it before the sector.
        comma = "," if "," in only.basis else ""
        what = f"{only.basis}{comma} for {only.sector}" if only.sector else only.basis
        return _sentence_case(", ".join(filter(None, [what, only.who, only.years, only.source])))
    shared_basis = len({fact.basis for fact in facts}) == 1
    shared_years = len({fact.years for fact in facts}) == 1
    shared_source = len({fact.source for fact in facts}) == 1
    head = [
        facts[0].basis if shared_basis else None,
        facts[0].years if shared_years else None,
        facts[0].source if shared_source else None,
    ]

    def clause(fact: _Counted) -> str:
        noun = fact.sector or (None if shared_basis else fact.basis) or "the total"
        return ", ".join(
            filter(
                None,
                [
                    f"{noun} {fact.who}".strip(),
                    None if shared_years else fact.years,
                    None if shared_source else fact.source,
                ],
            )
        )

    body = joiner.join(
        " and ".join(dict.fromkeys(clause(fact) for fact in side)) for side in counted
    )
    lead = ", ".join(filter(None, head))
    return _sentence_case(f"{lead}: {body}" if lead else body)


def _ratio_beneath(requirement: EvidenceRequirement, plan: Plan) -> EvidenceRequirement:
    """The share or quotient `requirement` reads through steps reading one input each
    (an aggregate, growth, rank or filter), else `requirement` itself."""
    current = requirement
    while not isinstance(current, ShareOfRequirement | QuotientRequirement):
        inputs = inputs_of(current)
        if len(inputs) != 1:
            return requirement
        current = plan.requirement(inputs[0])
    return current


def _reads_ratio(requirement: EvidenceRequirement, plan: Plan) -> bool:
    """Whether a share or quotient lies anywhere beneath `requirement`."""
    return any(
        isinstance(plan.requirement(input_id), ShareOfRequirement | QuotientRequirement)
        or _reads_ratio(plan.requirement(input_id), plan)
        for input_id in inputs_of(requirement)
    )


class _Counted(typing.NamedTuple):
    """What one retrieval counts, in words: the `basis` it is measured on, the
    `sector` it covers, `who` it reads, the `years` it covers and its `source`."""

    basis: str
    sector: str | None
    who: str
    years: str
    source: str | None


def _counted(requirement: DatasetRequirement, plan: Plan) -> _Counted:
    """What `requirement` counts, as `card_summary` states it.

    The basis is the one its basis decision chose, else its source's
    `measure_label`, else the planner's `measure`; the sector the one its
    sector decision governs (`_sector_concept`). The parties are those its
    source reads (`_who`). The source is named by the tables Search says its
    figures are published in (`SelectedSource.published_tables`), else by its
    provenance label.
    """
    governing = _governing(requirement, plan)
    source = _selected_source(requirement.id, plan)
    basis = governing.get("measure_basis")
    sector = governing.get("sector")
    if basis is not None:
        measure = basis.chosen
    elif source is not None and source.measure_label is not None:
        measure = source.measure_label
    else:
        measure = requirement.measure
    start, end = requirement.period.start, requirement.period.end
    years = f"in {start}" if start == end else f"from {start} to {end}"
    if requirement.period_end_open:
        years += _FOLLOWING
    provenance = (source.published_tables or source.provenance_label) if source else None
    return _Counted(
        basis=measure,
        sector=_sector_concept(sector, source),
        who=_who(requirement, plan),
        years=years,
        source=f"from the {provenance}" if provenance else None,
    )


def _sector_concept(sector: Decision | None, source: SelectedSource | None) -> str | None:
    """The sector a retrieval its `sector` decision governs counts, as `card_summary`
    names it: Search's `concept_label` for its `source`, else the option the
    decision holds, as Search labelled it; `None` where no sector decision governs."""
    if sector is None:
        return None
    if source is not None and source.concept_label is not None:
        return source.concept_label
    return sector.chosen


def _who(
    requirement: DatasetRequirement,
    plan: Plan,
    *,
    sources: Mapping[str, SelectedSource] | None = None,
) -> str:
    """The parties `requirement` reads, by the side of the flow each is on.

    "from United States to 129 recipients": the entities it names; a set's
    members as its region decision counts them, when one governs it; the
    named set its source binds on a side it names nobody on
    (`SelectedSource.groups`); and on a side still empty, how many members
    the source holds on its open axis there. Entities on no side of a flow
    read "for Kenya". A plan not compiled reads its source off `sources`
    (`_source`).
    """
    region = _governing(requirement, plan).get("region")
    source = _source(requirement.id, plan, sources)
    sides: dict[str | None, list[str]] = {"giving": [], "receiving": [], None: []}
    for entity in requirement.entities:
        side = flow_side(entity.role) if entity.role else None
        name = region.chosen if entity.members and region is not None else entities_phrase([entity])
        sides[side].append(name)
    if source is not None:
        _source_parties(source, sides)
    return _sides_phrase(sides)


def _source_parties(source: SelectedSource, sides: dict[str | None, list[str]]) -> None:
    """Name each side of the flow `sides` leaves empty from `source`: the named
    set it binds there (`SelectedSource.groups`), else how many members it
    holds on its open axis there, when two or more."""
    for group in source.groups:
        side = flow_side(group.role)
        if side is not None and not sides[side]:
            sides[side].append(group.label)
    for axis in source.open_entity_axes or ():
        side = flow_side(axis.column)
        if side is None or sides[side] or axis.member_count is None or axis.member_count < 2:
            continue
        word = party_role(axis.column) or singular(axis.column)
        sides[side].append(f"{axis.member_count:,} {pluralize(word)}")


def _sides_phrase(sides: Mapping[str | None, Sequence[str]]) -> str:
    """The parties `sides` names, by the side of the flow each is on: "from
    United States to Kenya", "for Kenya"."""
    joined = {side: _joined(names) for side, names in sides.items() if names}
    return " ".join(
        f"{preposition} {joined[side]}"
        for side, preposition in (("giving", "from"), ("receiving", "to"), (None, "for"))
        if side in joined
    )


def source_whole_name(
    axis: str,
    *,
    source: SelectedSource,
    part: SelectedSource | None,
    sector: str | None,
    measure: str | None,
) -> str | None:
    """What a declared whole leaving the donor or recipient `axis` open
    fetches, named off the source Search answered for it alone, with no plan.

    The concept as the plan card names it (`_entity_whole_concept`). Then the
    parties `source` binds, each side by the named set bound there, else its
    entities, else the members its open axis holds, and the opened side,
    still unnamed, as every party there: "official development assistance
    from DAC countries", "gross disbursements (official development
    assistance) from official donors (DAC) to Kenya". `None` where `axis` is
    on no side of a flow or nothing names the concept."""
    side = flow_side(axis)
    concept = _entity_whole_concept(source=source, part=part, sector=sector, measure=measure or "")
    if side is None or not concept:
        return None
    sides: dict[str | None, list[str]] = {"giving": [], "receiving": [], None: []}
    bound = {flow_side(group.role) for group in source.groups}
    for entity in source.entities:
        if (entity_side := flow_side(entity.role)) is not None and entity_side not in bound:
            sides[entity_side].append(entity.label)
    _source_parties(source, sides)
    if not sides[side]:
        sides[side].append(whole_name(axis))
    return f"{concept} {_sides_phrase(sides)}"


def _entity_whole_concept(
    *,
    source: SelectedSource | None,
    part: SelectedSource | None,
    sector: str | None,
    measure: str,
) -> str:
    """What a declared whole leaving a donor or recipient axis open fetches,
    the one rule the plan card (`_concept`) and its stored reading
    (`source_whole_name`) share, by `concept_name`'s order: the concept
    Search states for the whole's `source`, else for its part's `part`, else
    the `sector` option the whole was counted on, else the label Search
    composed for `source`'s measure, which differs from the part's where
    Search counted the whole on another series, else `part`'s, else
    `measure`."""
    sources = [held for held in (source, part) if held is not None]
    return concept_name(
        measure,
        concept_label=next((held.concept_label for held in sources if held.concept_label), None),
        sector=sector,
        measure_label=next((held.measure_label for held in sources if held.measure_label), None),
    )


def _joined(names: Sequence[str]) -> str:
    """`names` as "a", "a and b" or "a, b and c", each once."""
    unique = list(dict.fromkeys(names))
    if len(unique) == 1:
        return unique[0]
    return ", ".join(unique[:-1]) + f" and {unique[-1]}"


def _sentence_case(text: str) -> str:
    """`text` with its first letter raised and a full stop at its end."""
    return _full_stop(f"{text[0].upper()}{text[1:]}")


def _full_stop(text: str) -> str:
    """`text` stripped, ending in a full stop."""
    stated = text.strip()
    return stated if stated.endswith(".") else f"{stated}."


# Each scope choice's short name on the plan card, by the axis it settles.
CHOICE_NAMES: dict[str, str] = {
    "sector": "Sector",
    "region": "Region",
    "measure_basis": "Basis",
    "provider": "Provider",
}

# A planner slug: lower-case words joined by underscores, with no space.
_SLUG = re.compile(r"[a-z][a-z0-9]*(?:_[a-z0-9]+)+")


def card_choice_text(text: str, plan: Plan, *, decision: Decision | None = None) -> str:
    """An option a decision chose or offers, as the card shows it.

    As `card_decision_text`, except that a text the planner opened with a
    scope option's id ("sector:120+130 health") names only that option, so
    the whole text reads as the option's label, the words Search gives it.
    """
    named = with_source_names(text, plan).strip()
    if (labelled := _option_labelled(named, plan, decision)) is not None:
        return labelled[0]
    return _unslugged(named)


def card_decision_text(text: str, plan: Plan, *, decision: Decision | None = None) -> str:
    """A decision's question, reason or caveat as the card shows it.

    Each spec_id is named (`with_source_names`). A text the planner opened
    with a scope option's id reads with that id as the option's label and
    the rest as the planner wrote it: "provider:inclusive because the user
    asked for it" reads "<label> because the user asked for it". Two sources
    can give one option id different labels, so the label is the one
    `decision`, the decision the text is of, gives the option where it is a
    scope decision offering it, else the one the first scope decision of
    `plan` offering it gives. A slug the planner wrote for an option
    ("crs_total") reads as words ("crs total").
    """
    named = with_source_names(text, plan).strip()
    if (labelled := _option_labelled(named, plan, decision)) is not None:
        label, rest = labelled
        return f"{label}{rest}"
    return _unslugged(named)


def _option_labelled(text: str, plan: Plan, decision: Decision | None) -> tuple[str, str] | None:
    """The label of the scope option whose id `text` opens with, and the text
    after that id (`card_decision_text`); `None` when `text` opens with no
    scope option's id. The label is the one `decision` gives the option,
    where it is a scope decision offering it, else the one the first scope
    decision of `plan` offering it gives."""
    for held in (decision, *plan.decisions):
        if held is None or held.scope is None:
            continue
        for option, label in zip(held.scope.options, held.options, strict=True):
            if text == option.option_id or text.startswith(f"{option.option_id} "):
                return label, text.removeprefix(option.option_id)
    return None


def _unslugged(text: str) -> str:
    """`text` as words when it is a slug the planner wrote ("crs_total")."""
    return text.replace("_", " ") if _SLUG.fullmatch(text) else text


def concept_name(
    measure: str,
    *,
    concept_label: str | None,
    sector: str | None,
    measure_label: str | None,
) -> str:
    """What a retrieval fetches, in the reader's words: the one naming rule the
    plan card (`_concept`) and the caveats `propose_plan` writes
    (`submission.service._card_names`) share.

    Search's `concept_label` for the source ("health", "official development
    assistance"), which states no basis, codes, provider scope or dataset.
    A source stating none, as from a Search that predates the field or one
    recorded before it, is named by the labels Search composed, as Search
    wrote them: the `sector` option its sector choice holds, else its
    `measure_label`. Otherwise the planner's `measure`.
    """
    return concept_label or sector or measure_label or measure


# What a declared whole fetches, by the axis it opens (`WholeOf.axis`), with the
# preposition that sets it after a measure.
_WHOLE_NAMES = {
    "sector": ("in", "all sectors"),
    "donor": ("from", "all donors"),
    "recipient": ("to", "all recipients"),
}


def whole_name(axis: str, *, measure: str | None = None) -> str:
    """What a declared whole leaving `axis` open fetches ("all sectors", "all
    donors"), on `measure` when given ("gross disbursements in all sectors",
    "gross disbursements from all donors")."""
    preposition, name = _WHOLE_NAMES[axis]
    return f"{measure} {preposition} {name}" if measure else name


def _concept(
    requirement: DatasetRequirement,
    plan: Plan,
    *,
    sources: Mapping[str, SelectedSource] | None = None,
) -> str:
    """What `requirement` fetches, in the reader's words, as `_concepts` names it
    (`concept_name`), read off the scope decisions it cites and its source.

    A declared whole (`DatasetRequirement.whole_of`) is its part with the
    part's axis opened, and ignores the planner's `measure`. Over every
    sector it is named by that axis ("all sectors"). Over every donor or
    recipient it is named off its own source and its part's
    (`_entity_whole_concept`), on the sector option the whole cites, else
    its part's, else by its part's `measure`; the parties it reads (`_who`,
    `_parties`) name the population its source holds there. A plan not
    compiled reads its source off `sources` (`_source`)."""
    if requirement.whole_of is not None:
        part = plan.requirement(requirement.whole_of.part)
        if requirement.whole_of.axis == "sector" or not isinstance(part, DatasetRequirement):
            return whole_name(requirement.whole_of.axis)
        sector = _governing(requirement, plan).get("sector") or _governing(part, plan).get("sector")
        return _entity_whole_concept(
            source=_source(requirement.id, plan, sources),
            part=_source(part.id, plan, sources),
            sector=sector.chosen if sector is not None else None,
            measure=part.measure,
        )
    sector = _governing(requirement, plan).get("sector")
    source = _source(requirement.id, plan, sources)
    return concept_name(
        requirement.measure,
        concept_label=source.concept_label if source is not None else None,
        sector=sector.chosen if sector is not None else None,
        measure_label=source.measure_label if source is not None else None,
    )


class _CardNames(typing.NamedTuple):
    """What `_concepts` names each retrieval by, by requirement id (`names`), and
    the axes of `_NAMING_AXES` whose option each name states (`options`)."""

    names: dict[str, str]
    options: dict[str, frozenset[str]]


def _concepts(
    plan: Plan,
    *,
    carried: _CardNames | None = None,
    sources: Mapping[str, SelectedSource] | None = None,
    namings: Mapping[str, _Naming] | None = None,
) -> _CardNames:
    """What each retrieval of `plan` is named by in the card's labels, by requirement id,
    and the axes whose option each name states.

    Its `_concept`. Retrievals named alike add the parties they read, by the
    side of the flow each is on (`_who`): "health from United States to
    Kenya". Any still alike whose periods differ add their period, "health to
    Kenya in 2023". Any still alike that hold different options on the
    provider axis, then on the measure-basis axis (`_NAMING_AXES`), add the
    option each holds, as the source labels it: "health to Kenya, bilateral
    flows only". Any still alike add a number no other name holds, "(1)",
    "(2)", each set of namesakes in the order the plan states them
    (`_card_order`). A retrieval in `carried` keeps the name, and the axes it
    states, given there. Namesakes are found by these facets, carried or not,
    so the others are told apart from it as from each other. The names are
    read off each retrieval's naming facets, `namings` where given, else read
    by `_namings`, a plan not compiled reading its sources off `sources`
    (`_source`).
    """
    carried = carried or _CardNames({}, {})
    namings = namings if namings is not None else _namings(plan, sources=sources)
    datasets = [r for r in _card_order(plan) if isinstance(r, DatasetRequirement)]
    stems = {dataset.id: namings[dataset.id].concept for dataset in datasets}
    for rid in _alike(stems):
        if who := namings[rid].who:
            stems[rid] = f"{stems[rid]} {who}"
    names = {dataset.id: carried.names.get(dataset.id) or stems[dataset.id] for dataset in datasets}
    options = {dataset.id: carried.options.get(dataset.id, frozenset()) for dataset in datasets}
    fresh = [dataset for dataset in datasets if dataset.id not in carried.names]
    for stem in dict.fromkeys(_alike(stems).values()):
        namesakes = [dataset for dataset in datasets if stems[dataset.id] == stem]
        if len({dataset.period for dataset in namesakes}) > 1:
            for dataset in namesakes:
                if dataset.id not in carried.names:
                    names[dataset.id] = f"{stem} {_in_period(dataset)}"
    for index, axis in enumerate(_NAMING_AXES):
        for name in dict.fromkeys(_alike(names).values()):
            namesakes = [dataset for dataset in datasets if names[dataset.id] == name]
            if len({namings[dataset.id].options[index] for dataset in namesakes}) < 2:
                continue
            for dataset in namesakes:
                held = namings[dataset.id].options[index]
                if dataset.id not in carried.names and held is not None:
                    names[dataset.id] = f"{name}, {held}"
                    options[dataset.id] |= {axis}
    for name in dict.fromkeys(_alike(names).values()):
        taken = set(names.values())
        number = 0
        for dataset in fresh:
            if names[dataset.id] != name:
                continue
            number += 1
            while f"{name} ({number})" in taken:
                number += 1
            names[dataset.id] = f"{name} ({number})"
    return _CardNames(names, options)


def _carried_names(namings: Mapping[str, _Naming], shown: Plan) -> _CardNames:
    """The name the card gives each retrieval of `shown` (`_concepts`), and the
    axes it states, that a revision, `namings` its retrievals' naming facets
    (`_namings`), holds with the same facets, by requirement id."""
    earlier = _namings(shown)
    card = _concepts(shown, namings=earlier)
    kept = [rid for rid, naming in namings.items() if earlier.get(rid) == naming]
    return _CardNames(
        {rid: card.names[rid] for rid in kept}, {rid: card.options[rid] for rid in kept}
    )


# The scope axes whose options tell apart retrievals otherwise named alike
# (`_concepts`), in the order their options are added to a name.
_NAMING_AXES = ("provider", "measure_basis")


class _Naming(typing.NamedTuple):
    """What `_concepts` names a retrieval by: its concept, parties and period,
    and the option it holds on each of `_NAMING_AXES`, `None` where no scope
    decision it cites settles that axis."""

    concept: str
    who: str
    period: Period
    options: tuple[str | None, ...]


def _namings(
    plan: Plan, *, sources: Mapping[str, SelectedSource] | None = None
) -> dict[str, _Naming]:
    """The naming facets of each retrieval of `plan`, by requirement id, each
    read once (`_concept`, `_who`, `_held_options`). A plan not compiled reads
    its sources off `sources` (`_source`)."""
    return {
        dataset.id: _Naming(
            _concept(dataset, plan, sources=sources),
            _who(dataset, plan, sources=sources),
            dataset.period,
            _held_options(dataset, plan),
        )
        for dataset in _datasets(plan).values()
    }


def _held_options(requirement: DatasetRequirement, plan: Plan) -> tuple[str | None, ...]:
    """The option `requirement` holds on each of `_NAMING_AXES`: the label the
    scope decision it cites there chose, `None` where it cites none."""
    governing = _governing(requirement, plan)
    return tuple(
        decision.chosen if (decision := governing.get(axis)) is not None else None
        for axis in _NAMING_AXES
    )


def _pinned_sources(
    plan: Plan, *, shown_as: Plan | None, pinned: Mapping[str, SelectedSource]
) -> dict[str, SelectedSource]:
    """The source each retrieval of `plan` reads, by requirement id, where
    `plan` is not compiled. A compiled plan names its own, so the result is
    empty.

    A retrieval the card's plan `shown_as` holds as it stands, or pinned to
    the same spec_id, reads the source the card's retrieval was compiled
    from, so the card's names carry (`_carried_names`). Any other pinned
    retrieval reads the source Search answered for its pin (`pinned`, by
    spec_id). One pinning nothing `pinned` holds reads none."""
    if plan.workflow is not None:
        return {}
    earlier = _datasets(shown_as) if shown_as is not None else {}
    sources: dict[str, SelectedSource] = {}
    for dataset in _datasets(plan).values():
        prior = earlier.get(dataset.id)
        spec_id = dataset.source_spec_id
        same = prior is not None and (
            prior == dataset or (spec_id is not None and prior.source_spec_id == spec_id)
        )
        source = _selected_source(dataset.id, shown_as) if same and shown_as is not None else None
        if source is None and spec_id is not None:
            source = pinned.get(spec_id)
        if source is not None:
            sources[dataset.id] = source
    return sources


def _datasets(plan: Plan) -> dict[str, DatasetRequirement]:
    """The retrievals of `plan`, by requirement id."""
    return {r.id: r for r in plan.evidence_requirements if isinstance(r, DatasetRequirement)}


def _in_period(dataset: DatasetRequirement) -> str:
    """The years `dataset` reads, as a name's tail: "in 2023", "in 2015-2023"."""
    start, end = dataset.period.start, dataset.period.end
    return f"in {start}" if start == end else f"in {start}-{end}"


def _alike(names: Mapping[str, str]) -> dict[str, str]:
    """The entries of `names` whose name another entry shares."""
    counts = Counter(names.values())
    return {rid: name for rid, name in names.items() if counts[name] > 1}


def _sides(requirement: EvidenceRequirement) -> tuple[RequirementSide, RequirementSide] | None:
    """The two sides of a share, quotient, sum or comparison, in the order its
    words name them; `None` for any other requirement."""
    match requirement:
        case ShareOfRequirement() | QuotientRequirement():
            return requirement.numerator, requirement.denominator
        case SumOfRequirement():
            return requirement.left, requirement.right
        case ComparisonRequirement():
            return requirement.minuend, requirement.subtrahend
        case _:
            return None


def _side_named(side: RequirementSide, name: str) -> str:
    """`name`, what `side` reads, with the one entity it narrows to
    (`RequirementSide.entity`) when it names one, so two sides of one
    retrieval read apart: "grant equivalents for United States"."""
    return f"{name} for {side.entity}" if side.entity is not None else name


def _concept_of(requirement: EvidenceRequirement, plan: Plan, concepts: Mapping[str, str]) -> str:
    """The concept a step's value is a reading of, walking a total, a change,
    a rank or a filter back to the retrieval it reads (`_concepts`); a step
    reading two names both, each side with the entity it narrows to
    (`_side_named`)."""
    if isinstance(requirement, DatasetRequirement):
        return concepts[requirement.id]
    if isinstance(requirement, JoinRequirement):
        left, right = (
            _concept_of(plan.requirement(input_id), plan, concepts)
            for input_id in inputs_of(requirement)
        )
        return f"{left} against {right}"
    sides = _sides(requirement)
    if sides is not None:
        left, right = (
            _side_named(side, _concept_of(plan.requirement(side.input_id), plan, concepts))
            for side in sides
        )
        joiner = {
            ShareOfRequirement: "as a share of",
            QuotientRequirement: "divided by",
            SumOfRequirement: "plus",
            ComparisonRequirement: "against",
        }[type(requirement)]
        return f"{left} {joiner} {right}"
    (input_id,) = inputs_of(requirement)
    return _concept_of(plan.requirement(input_id), plan, concepts)


def _card_label(
    requirement: EvidenceRequirement,
    plan: Plan,
    *,
    concepts: Mapping[str, str],
    shapes: Mapping[str, FrameShape],
) -> str:
    """`requirement`'s short card label (`card_rows`). `concepts` names each
    retrieval (`_concepts`) and `shapes` holds each result's shape."""

    def of(requirement_id: str) -> str:
        return _concept_of(plan.requirement(requirement_id), plan, concepts)

    def side(read: RequirementSide) -> str:
        return _side_named(read, of(read.input_id))

    kept = _kept_words(shapes[requirement.id], plan)
    by = f" by {' and '.join(kept)}" if kept else ""
    match requirement:
        case DatasetRequirement():
            label = f"Fetch {_fetched(requirement, plan, concepts)}"
        case AggregateRequirement():
            label = f"{_REDUCTION_WORDS[requirement.reduction]} {of(requirement.input_id)}{by}"
        case GrowthRequirement():
            label = (
                f"Growth in {of(requirement.input_id)}, {requirement.window_start}-"
                f"{requirement.window_end}{_following(requirement.rolling)}"
            )
        case ComparisonRequirement():
            label = f"Compare {side(requirement.minuend)} to {side(requirement.subtrahend)}"
        case SumOfRequirement():
            label = f"Add {side(requirement.left)} and {side(requirement.right)}"
        case RankRequirement():
            ranked = (
                _axis_word(requirement, plan)
                if requirement.limit is None
                else _kept_phrase(requirement, requirement.limit, plan)
            )
            label = f"Rank {ranked} by {of(requirement.input_id)}"
        case ShareOfRequirement() | QuotientRequirement():
            label = _concept_of(requirement, plan, concepts)
            if requirement.axis is not None:
                label += f" per {requirement.axis}"
        case JoinRequirement():
            label = f"Join {of(requirement.left_id)} with {of(requirement.right_id)}"
        case FilterRequirement():
            label = _card_filter_label(requirement, of(requirement.input_id))
        case _:
            raise AssertionError(f"unhandled requirement kind: {requirement!r}")
    return f"{label[0].upper()}{label[1:]}"


def _fetched(requirement: DatasetRequirement, plan: Plan, concepts: Mapping[str, str]) -> str:
    """What a retrieval fetches and for whom: its name in `concepts`
    (`_concepts`), which already names its parties when another retrieval
    shares its concept, else that name with its parties (`_parties`)."""
    name = concepts[requirement.id]
    if name != _concept(requirement, plan):
        return name
    return f"{name} for {_parties(requirement, plan)}"


def _card_filter_label(requirement: FilterRequirement, concept: str) -> str:
    """A filter's short card label: the entities it keeps, the level of place, or neither."""
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
    return f"Filter {concept}"


def shown_decisions(plan: Plan) -> tuple[Decision, ...]:
    """The decisions of `plan` the card and the plan record list.

    Leaves out the decision `propose_plan` writes for a pin of a source a
    search settled on (`planner.plan.is_search_pin`) when its choice names
    the source's spec_id, since the plan names that source in words where
    it lists its sources. One that chooses how the flow is counted, by the
    measures' labels, is listed; pins making the same choice share one
    listed decision (`_listed_slots`). Every scope decision
    (`planner.plan.is_scope_choice`) is listed, and sources whose choices
    are the same share one.
    """
    listed = _listed_slots(plan)
    return tuple(
        decision for decision in plan.decisions if listed.get(decision.slot) == decision.slot
    )


def card_questions(plan: Plan) -> dict[str, str]:
    """What each row of the plan card's choices asks, by the slot of the decision it lists.

    The card lists the decisions `shown_decisions` lists less those that
    qualify the answer (`Decision.caveat`), which reach it only as caveats.
    A scope decision's row asks its axis's short name (`CHOICE_NAMES`).
    Where another row settles the same axis, as for two sources Search
    offers different options on, each adds the inputs it governs, named as
    the card's labels name them (`_concepts` of `shown_plan`, so a rolling
    copy never numbers its origin): "Basis: health". Any other
    row asks the planner's question (`card_decision_text`).
    """
    rows = [decision for decision in shown_decisions(plan) if decision.caveat is None]
    axes = Counter(decision.scope.axis for decision in rows if decision.scope is not None)
    concepts = (
        _concepts(shown_plan(plan)).names if any(count > 1 for count in axes.values()) else {}
    )
    questions: dict[str, str] = {}
    for decision in rows:
        name = CHOICE_NAMES.get(decision.scope.axis) if decision.scope is not None else None
        if name is None:
            questions[decision.slot] = card_decision_text(
                decision.description, plan, decision=decision
            )
            continue
        assert decision.scope is not None
        governed = (
            _governed_concepts(plan, decision, concepts) if axes[decision.scope.axis] > 1 else []
        )
        questions[decision.slot] = f"{name}: {_joined(governed)}" if governed else name
    return questions


def _governed_concepts(plan: Plan, decision: Decision, concepts: Mapping[str, str]) -> list[str]:
    """The names `concepts` gives the retrievals pinning a source the row of scope
    decision `decision` stands for (`listed_together`), in the order the plan states them."""
    pinned = {
        member.fixes.source_spec_id
        for member in listed_together(plan, decision.slot)
        if member.fixes is not None and member.fixes.source_spec_id is not None
    }
    return [
        name
        for requirement_id, name in concepts.items()
        if typing.cast(DatasetRequirement, plan.requirement(requirement_id)).source_spec_id
        in pinned
    ]


def listed_together(plan: Plan, slot: str) -> tuple[Decision, ...]:
    """The decisions of `plan` the card's row for `slot` stands for: the
    decision under `slot` and every one listed with it (`_listed_slots`);
    empty when no decision of `plan` holds `slot`."""
    listed = _listed_slots(plan)
    row = listed.get(slot)
    if row is None:
        return ()
    return tuple(decision for decision in plan.decisions if listed.get(decision.slot) == row)


def choice_id(plan: Plan, decision: Decision) -> str:
    """The id the plan card gives scope decision `decision`'s row: `"<requirement id>:<axis>"`.

    The requirement is the first of `plan` pinning a source the row stands
    for (`listed_together`). A switch re-pins requirements and keeps their
    ids, so the row keeps its id through every switch of the conversation,
    whichever decision of the row is listed. A row whose sources no
    requirement pins keeps Search's own id for the choice.
    """
    assert decision.scope is not None, "only a scope decision has a choice id"
    pinned = {
        member.fixes.source_spec_id
        for member in listed_together(plan, decision.slot) or (decision,)
        if member.fixes is not None and member.fixes.source_spec_id is not None
    }
    return pinned_choice_id(plan, pinned, decision.scope.axis) or decision.scope.choice_id


def pinned_choice_id(plan: Plan, pinned: Collection[str], axis: str) -> str | None:
    """`"<requirement id>:<axis>"` for the first requirement of `plan` pinning one
    of `pinned`, the id `choice_id` gives a row standing for those sources;
    `None` when no requirement pins any."""
    first = next(
        (
            requirement
            for requirement in plan.evidence_requirements
            if isinstance(requirement, DatasetRequirement) and requirement.source_spec_id in pinned
        ),
        None,
    )
    return f"{first.id}:{axis}" if first is not None else None


def chosen_scope_decision(plan: Plan, choice: str) -> Decision | None:
    """The listed scope decision of `plan` whose row has id `choice` (`choice_id`), if any.

    A card sent before rows took `"<requirement id>:<axis>"` ids names a row
    by Search's own id for the choice (`ScopeDecision.choice_id`), so a
    switch from a plan paused then finds its row by that id when no row has
    `choice` as its own.
    """
    scoped = [decision for decision in shown_decisions(plan) if decision.scope is not None]
    return next(
        (decision for decision in scoped if choice_id(plan, decision) == choice),
        next(
            (
                decision
                for decision in scoped
                if decision.scope is not None and decision.scope.choice_id == choice
            ),
            None,
        ),
    )


def _listed_slots(plan: Plan) -> dict[str, str]:
    """Each slot of `plan`'s decisions mapped to the slot of the listed decision stating it.

    Search pins whose decisions state the same description and the same
    chosen label, such as a share's numerator and denominator sources both
    counted as gross disbursements, map to one of them: the first one the
    user settled, else the first. Their options are left out of the
    comparison, since the card and the plan record show no options, and two
    sources offering different other counts make the same choice. A search
    pin whose choice names a spec_id has no entry. Scope decisions of
    several sources map to one of them, chosen the same way, when they
    settle the same axis on the same option with the same options, each as
    available as in the others, so the row reads true for every source it
    stands for. Every other decision maps to its own slot.
    """
    listed: dict[str, str] = {}
    same_choice: dict[tuple[object, ...], list[Decision]] = {}
    for decision in plan.decisions:
        if is_scope_choice(decision):
            assert decision.scope is not None
            key = (decision.scope.axis, decision.chosen, decision.options, decision.scope.options)
            same_choice.setdefault(key, []).append(decision)
        elif not is_search_pin(decision):
            listed[decision.slot] = decision.slot
        elif not _SPEC_ID.search(decision.chosen):
            same_choice.setdefault((decision.description, decision.chosen), []).append(decision)
    for pins in same_choice.values():
        kept = next((pin for pin in pins if pin.resolved_by is ResolvedBy.USER), pins[0])
        listed.update(dict.fromkeys((pin.slot for pin in pins), kept.slot))
    return listed


# A spec_id as `planner.records.SpecId` states it, anywhere in a text.
_SPEC_ID = re.compile(r"spec_[0-9a-f]{32}")


def with_source_names(text: str, plan: Plan) -> str:
    """`text` with each spec_id in it replaced by the name of the source it identifies.

    A source the plan's workflow retrieves is named by its provenance label,
    else its sentence; any other spec_id becomes "another source". A
    decision a model wrote can name a source by its spec_id in `chosen`,
    `options` or `rationale`, and the card and the plan record state no id.
    """
    sources = plan.workflow.sources if plan.workflow is not None else ()
    names = {source.spec_id: source.provenance_label or source.sentence for source in sources}
    return _SPEC_ID.sub(lambda match: names.get(match.group(), "another source"), text)


def requirement_rows(plan: Plan) -> list[dict[str, object]]:
    """One row per evidence requirement the plan states: id, kind, its decisions, and a sentence.

    A requirement's decisions are those `shown_decisions` lists, each cited
    slot replaced by the slot of the listed decision stating its choice.
    """
    listed = _listed_slots(plan)
    shown = shown_plan(plan)
    return [
        {
            "id": requirement.id,
            "kind": requirement.kind,
            "decision_ids": list(
                dict.fromkeys(listed[slot] for slot in requirement.decision_ids if slot in listed)
            ),
            "sentence": _sentence(requirement, shown),
        }
        for requirement in shown.evidence_requirements
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
    shown = shown_plan(plan)
    assert shown.translation is not None
    rows: list[dict[str, object]] = []
    for operation in shown.translation.operations:
        requirement = shown.requirement(operation.requirement_id)
        rows.append(
            {
                "id": operation.id,
                "kind": operation.kind,
                "requirement_id": operation.requirement_id,
                "sentence": _sentence(requirement, shown),
            }
        )
    return rows


def ask_rows(plan: Plan) -> list[dict[str, object]]:
    """One row per ask the plan accounts for, in the brief's order.

    Each row is the `ask`, `answered_by`, the ids of the workflow steps
    whose results answer it, `declined`, the code it was declined under,
    `reason`, that code's `harness.sentences.DECLINE_REASONS` entry, and `not_computed`, the
    part the decline names in the user's words. The last three are `None`
    for an ask the plan answers whole; an ask answered in part carries
    both `answered_by` and a decline. A step's id is that of the operation
    carrying out its requirement (`planner.workflow.WorkflowStep`).

    Requires a plan carrying a `translation`.
    """
    assert plan.translation is not None, "ask_rows needs a plan carrying a translation"
    step_of = {op.requirement_id: op.id for op in plan.translation.operations}
    answered_by = {
        answer.ask: [step_of[output] for output in answer.outputs] for answer in plan.answered
    }
    declined = {d.ask_ref: d for d in plan.declined if d.ask_ref is not None}
    rows: list[dict[str, object]] = []
    for number, ask in enumerate(plan.asks):
        decline = declined.get(number)
        rows.append(
            {
                "ask": ask,
                "answered_by": answered_by.get(number, []),
                "declined": decline.code if decline is not None else None,
                "reason": DECLINE_REASONS[decline.code] if decline is not None else None,
                "not_computed": decline.ask if decline is not None else None,
            }
        )
    return rows


def answer_scopes(plan: Plan) -> dict[str, str]:
    """How many figures each answering result gives, and over which years, in words, by id.

    Those are the requirements `planner.plan.answering_results` names, the
    set the result-scope rule checks.
    """
    shapes = frame_shapes(plan)
    return {output: _scope_phrase(shapes[output], plan) for output in answering_results(plan)}


def share_wholes(plan: Plan) -> dict[str, str]:
    """The whole the shares each step of the compiled `plan` gives divide by, in words, by step id.

    A share_of step's whole is `_divided_whole`'s. A step whose values are
    still its input's shares keeps their whole (`_keeps_shares`). A step of
    anything else, or of shares whose whole cannot be named, has no entry.
    Empty before compile.
    """
    if plan.translation is None or plan.workflow is None:
        return {}
    shapes = frame_shapes(plan)
    operations = {operation.id: operation for operation in plan.translation.operations}
    wholes: dict[str, str] = {}
    for step in plan.workflow.steps:
        requirement = plan.requirement(operations[step.operation_id].requirement_id)
        while _keeps_shares(requirement, shapes):
            requirement = plan.requirement(requirement.input_id)
        if isinstance(requirement, ShareOfRequirement) and (
            whole := _divided_whole(requirement, plan)
        ):
            wholes[step.id] = whole
    return wholes


def _keeps_shares(
    requirement: EvidenceRequirement, shapes: Mapping[str, FrameShape]
) -> typing.TypeGuard[FilterRequirement | RankRequirement | AggregateRequirement]:
    """Whether `requirement`'s values are its input's shares, or their
    average, highest or lowest, so they stay shares of the same whole: a
    filter, or a rank or an aggregate reducing by no total, whose output
    holds a ratio (`ValueKind.RATIO`). A total of shares is no share of it,
    and a rank kept per period with no `limit` writes positions."""
    if isinstance(requirement, FilterRequirement):
        return True
    if isinstance(requirement, RankRequirement | AggregateRequirement):
        return (
            requirement.reduction is not Reduction.SUM
            and shapes[requirement.id].value is ValueKind.RATIO
        )
    return False


def _divided_whole(requirement: ShareOfRequirement, plan: Plan) -> str | None:
    """The whole `requirement` divides by, in words, or `None` when it cannot be named.

    The whole is the population the denominator's selected sources hold on
    each role column (`_populations`): "from official donors (DAC)" for a
    source binding that group's total, "from every donor" for one leaving
    the donors open. A denominator narrowed to fewer members than its
    sources hold has no whole. A column where the numerator holds the same
    population (`_Population.same`) is left out, since a share of health ODA
    from every donor, read from the same dataset, divides by the same donors.
    A side narrowed to one entity of its frame (`RequirementSide.entity`)
    holds that entity on its column (`_narrowed_entity`) and what its
    sources hold on every other; a side narrowed to an entity its sources do
    not resolve to one ref has no population to compare, so the denominator
    then names no whole and the numerator shows nothing to differ. A share
    of one side by itself (`pools_its_own_axis`) divides by every population
    its denominator holds. A keyed share's `axis` column follows its
    `denominator_scope`: `per_axis` divides each member by its own total, so
    that column is no part of the whole; `pooled` divides by the total
    across the column's members, which is part of it whatever the numerator
    holds there.

    Off the axis, a column of named entities (`_Population.named`) is part
    of the whole only where the numerator is known to hold other entities
    there: the United States' share of the United Kingdom's and the United
    States' ODA divides "from United Kingdom and United States". A
    numerator narrowed by a step (`_reads_whole`) shows nothing to differ.
    """
    denominator = _side_populations(requirement.denominator, plan)
    if denominator is None:
        return None
    numerator = (
        {}
        if requirement.pools_its_own_axis
        else _side_populations(requirement.numerator, plan) or {}
    )
    axis = None if requirement.axis is None else role_party(requirement.axis)
    clauses: list[str] = []
    for column, population in denominator.items():
        on_axis = axis is not None and _column_party(column) == axis
        if on_axis and requirement.denominator_scope is DenominatorScope.PER_AXIS:
            continue
        held = numerator.get(column)
        if population.named and not on_axis:
            shown = held is not None and not held.same(population)
        else:
            shown = on_axis or held is None or not held.same(population)
        if shown:
            clauses.append(f"{_preposition(column)} {population.phrase}")
    return " and ".join(clauses) or None


def _side_populations(side: RequirementSide, plan: Plan) -> dict[str, _Population] | None:
    """What one side of a share divides or is divided by: the populations its
    input holds (`_populations`), with the one entity it narrows its frame
    to on that entity's column (`_narrowed_entity`); `None` when either is
    unknown."""
    populations = _populations(plan.requirement(side.input_id), plan)
    if populations is None or side.entity is None:
        return populations
    narrowed = _narrowed_entity(side, plan)
    return None if narrowed is None else populations | narrowed


def _narrowed_entity(side: RequirementSide, plan: Plan) -> dict[str, _Population] | None:
    """The one entity `side` narrows its frame to (`RequirementSide.entity`),
    as a population on the role column its input's source resolves it on
    (`resolve_name`), under the role the input's requirement states for the
    name (`side_entities`); `None` when no source resolves it to one entity."""
    requirements = {requirement.id: requirement for requirement in plan.evidence_requirements}
    named = side_entities(requirements, side)
    found = {
        (resolved.role, resolved.ref)
        for dataset in datasets_of(plan.requirement(side.input_id), plan)
        if (source := _selected_source(dataset.id, plan)) is not None
        for entity in named
        if (resolved := resolve_name(entity, source)) is not None
    }
    if len(found) != 1:
        return None
    ((column, ref),) = found
    return {column: _Population(named[0].name, ("named", ref), named=True)}


def _column_party(column: str) -> str:
    """The party a source's role column holds: a side of a flow, or the place."""
    return flow_side(column) or "place"


def _preposition(column: str) -> str:
    """The preposition a population on role column `column` takes: "from" on the
    giving side of a flow, "to" on the receiving side, "in" for the place."""
    side = flow_side(column)
    if side is None:
        return "in"
    return "from" if side == "giving" else "to"


class _Population(typing.NamedTuple):
    """A population a source holds on one role column: its words, and what identifies it.

    `identity` is the named set's graph ref (its published total's, or its
    label for a set the graph holds no node for); an open axis's column,
    the source's count of its members and the source's dataset over any
    window (`planner.plan.dataset_of`), since the source records no members
    and two sources holding as many are known to hold the same ones only
    when they are one dataset; or the sorted refs of the named entities a
    column holds (`named`). `None` for an axis the source did not count or a
    source stating no identity, which nothing shows equal to another.
    """

    phrase: str
    identity: tuple[str, ...] | None
    named: bool = False

    def same(self, other: _Population) -> bool:
        """Whether `other` is known to be this population."""
        return self.identity is not None and self.identity == other.identity


def _populations(requirement: EvidenceRequirement, plan: Plan) -> dict[str, _Population] | None:
    """The population `requirement`'s selected sources hold on each role column,
    or `None` when a step between narrows them (`_reads_whole`).

    A named set a source binds (`SelectedSource.groups`) is named by its
    label, and an axis a source leaves open for an `across` word reads as
    `entities_phrase` writes it, uncounted: the whole sits in a card title that
    names a period, which an all-years count would read as. A column holding
    only entities the requirement names holds them by the refs its source
    resolves them to (`_named_entities`), `named`.
    """
    datasets = _reads_whole(requirement, plan)
    if datasets is None:
        return None
    populations: dict[str, _Population] = {}
    for dataset in datasets:
        source = _selected_source(dataset.id, plan)
        if source is None:
            continue
        for group in source.groups:
            identity = group.ref or group.total or group.label
            populations.setdefault(group.role, _Population(group.label, ("group", identity)))
        axes = open_axes_for(dataset.across, source.open_entity_axes) if dataset.across else ()
        if not isinstance(axes, str):
            for word, axis in zip(dataset.across, axes, strict=True):
                phrase = entities_phrase((), across=(word,))
                identity = (
                    None
                    if axis.member_count is None or source.identity is None
                    else (
                        axis.column,
                        str(axis.member_count),
                        json.dumps(dataset_of(source.identity), sort_keys=True),
                    )
                )
                populations.setdefault(axis.column, _Population(phrase, identity))
        for column, population in _named_entities(dataset, source).items():
            populations.setdefault(column, population)
    return populations


def _named_entities(dataset: DatasetRequirement, source: SelectedSource) -> dict[str, _Population]:
    """The entities `dataset` names on each role column, by the refs `source`
    resolves them to (`resolve_name`), as `entities_phrase` names them.

    A name a set of the source's answers (`resolve_group`) is the set's
    population, which `_populations` reads from the set. Empty when one name
    resolves to nothing, since the columns are then unknown.
    """
    named: dict[str, list[tuple[Entity, str]]] = {}
    for entity in dataset.entities:
        if resolve_group(entity, source) is not None:
            continue
        resolved = resolve_name(entity, source)
        if resolved is None:
            return {}
        named.setdefault(resolved.role, []).append((entity, resolved.ref))
    return {
        column: _Population(
            entities_phrase([entity for entity, _ in entities]),
            ("named", *sorted({ref for _, ref in entities})),
            named=True,
        )
        for column, entities in named.items()
    }


def _reads_whole(
    requirement: EvidenceRequirement, plan: Plan
) -> tuple[DatasetRequirement, ...] | None:
    """The dataset requirements `requirement` reads every member of, or `None`
    when a step on the way keeps fewer members than they hold.

    A total and a rank keeping every member pass their input's members on,
    as does a filter keeping some years only. A filter on any other field,
    a rank keeping its top members (`limit`), an aggregate or a rank
    reducing by anything but a total (`Reduction.SUM`), whose values are no
    total of the members, and any other step narrow them or read them in a
    way no population names.
    """
    if isinstance(requirement, DatasetRequirement):
        return (requirement,)
    if isinstance(requirement, FilterRequirement):
        if not all(isinstance(predicate, PeriodThreshold) for predicate in requirement.predicates):
            return None
    elif isinstance(requirement, RankRequirement | AggregateRequirement):
        if requirement.reduction not in (None, Reduction.SUM):
            return None
        if isinstance(requirement, RankRequirement) and requirement.limit is not None:
            return None
    else:
        return None
    return _reads_whole(plan.requirement(requirement.input_id), plan)


def _scope_phrase(shape: FrameShape, plan: Plan) -> str:
    """How many figures a result of `shape` gives, and over which years, in words.

    Read off the result's shape (`planner.plan.frame_shapes`): one figure
    per member of each party it keeps apart, per year when it keeps a
    series of years, then the years it covers. "one figure per donor,
    2015-2022" is a growth kept per member of an `across`; "one figure,
    2015-2022" is the same growth over their sum. A party is kept apart
    when its members are open (`across`) or a breakdown's codes, or when
    the requirement names several of them or a set's members.
    """
    phrase = "one figure" + "".join(f" per {word}" for word in _kept_words(shape, plan))
    if shape.period is None:
        return phrase
    years = (
        str(shape.period.start)
        if shape.period.start == shape.period.end
        else f"{shape.period.start}-{shape.period.end}"
    )
    return f"{phrase}, {years}"


def _kept_words(shape: FrameShape, plan: Plan) -> list[str]:
    """What a result of `shape` keeps one figure per, in words: each party it
    keeps apart (`_kept_apart`), then "year" when it keeps a series of
    years. Empty for one figure."""
    per = [
        singular(role.role) if role.role is not None else "place"
        for role in shape.roles
        if _kept_apart(role, plan)
    ]
    if (
        shape.time is TimeLabel.OBSERVATION
        and shape.period is not None
        and (shape.period.start != shape.period.end)
    ):
        per.append("year")
    return per


def _kept_apart(role: RoleDimension, plan: Plan) -> bool:
    """Whether a result keeps one figure per member of `role`'s party."""
    if role.open or role.breakdown:
        return True
    dataset = plan.requirement(role.requirement_id)
    assert isinstance(dataset, DatasetRequirement)
    named = [entity for entity in dataset.entities if entity.role == role.role]
    return len(named) > 1 or any(entity.members for entity in named)


def _group_by_phrase(
    group_by: Sequence[GroupBy], *, breakdown: Breakdown | None = None, grouped: bool = False
) -> str:
    """`group_by` in words, after "grouped" when `grouped`. A group by breakdown names
    the dimension of `breakdown`, the one the aggregate's input states. An empty group
    is one figure over whatever years and parties the input holds, and codes where it
    carries a breakdown, which a filter or a named side may narrow, so it names none
    of them."""
    if not group_by:
        read = "years, parties and codes" if breakdown is not None else "years and parties"
        return f"as one figure over all the {read} read"
    words = {
        GroupBy.PERIOD: "year",
        GroupBy.ENTITY: "entity",
        GroupBy.BREAKDOWN: breakdown.dimension.value if breakdown is not None else "breakdown",
    }
    phrase = " and ".join(f"by {words[key]}" for key in GroupBy if key in group_by)
    return f"grouped {phrase}" if grouped else phrase


def _breakdown_of(requirement: EvidenceRequirement, plan: Plan) -> Breakdown | None:
    """The breakdown the one dataset requirement `requirement` reads states, or `None`."""
    breakdowns = {
        dataset.breakdown
        for dataset in datasets_of(requirement, plan)
        if dataset.breakdown is not None
    }
    return breakdowns.pop() if len(breakdowns) == 1 else None


def _breakdown_clause(breakdown: Breakdown, source: SelectedSource | None) -> str:
    """How a retrieval breaks its total down, in words, at the level it reads.

    With a selected source that is the level the codes it names set
    (`compiler.shapes.breakdown_level`); before compile, or with a named
    code's path unread, the plan's own.
    """
    read = None if source is None else breakdown_level(breakdown, source)
    depth = breakdown.level if read is None else read
    level = (
        "the top level of its hierarchy"
        if depth == 0
        else f"level {depth} below the top of its hierarchy"
    )
    return f"broken down by {breakdown.dimension.value}, each code rolled up to {level}"


# What follows a year, or a span of years, that moves with the newest year the
# data holds. The years beside it are the ones the plan read.
_FOLLOWING = ", following the newest year"
# The same, after a selected source's sentence, which states the years it read.
_FOLLOWING_YEARS = "the years follow the newest year"


def _following(rolling: RollingWindow | None) -> str:
    """How a window's years follow the newest data, as a clause after them; empty for fixed years."""
    if rolling is None:
        return ""
    if rolling.window_start is not None and rolling.window_end is not None:
        return _FOLLOWING
    end = "start" if rolling.window_start is not None else "end"
    return f", the {end} following the newest year"


def _predicate_phrase(predicate: Predicate) -> str:
    if isinstance(predicate, EntityIn):
        subject = predicate.role or "the entity"
        return f"{subject} is one of {', '.join(predicate.names)}"
    if isinstance(predicate, PeriodThreshold):
        following = _FOLLOWING if predicate.rolling is not None else ""
        return f"the year is {_THRESHOLD_WORDS[predicate.op]} {predicate.year}{following}"
    if isinstance(predicate, ValueThreshold):
        return f"the value is {_THRESHOLD_WORDS[predicate.op]} {predicate.value}"
    if isinstance(predicate, PlaceLevel):
        subject = predicate.role or "place"
        return f"the {subject} is a {level_noun(predicate.level)}"
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
        return _dataset_measure(requirement, plan)
    if isinstance(
        requirement, (AggregateRequirement, GrowthRequirement, RankRequirement, FilterRequirement)
    ):
        return _measure_of(plan.requirement(requirement.input_id), plan)
    if isinstance(requirement, ComparisonRequirement):
        minuend = _measure_of(plan.requirement(requirement.minuend.input_id), plan)
        subtrahend = _measure_of(plan.requirement(requirement.subtrahend.input_id), plan)
        return f"{minuend} against {subtrahend}"
    if isinstance(requirement, SumOfRequirement):
        left = _measure_of(plan.requirement(requirement.left.input_id), plan)
        right = _measure_of(plan.requirement(requirement.right.input_id), plan)
        return f"{left} plus {right}"
    if isinstance(requirement, ShareOfRequirement | QuotientRequirement):
        return _divided(requirement, plan, _measure_of)
    if isinstance(requirement, JoinRequirement):
        left = _measure_of(plan.requirement(requirement.left_id), plan)
        right = _measure_of(plan.requirement(requirement.right_id), plan)
        return f"{left} joined with {right}"
    raise AssertionError(f"unhandled requirement kind: {requirement!r}")


# The scope axes whose choice settles what a dataset's measure counts: the
# basis it is counted on and the sector codes it covers.
_MEASURE_AXES = frozenset({"measure_basis", "sector"})


def _dataset_measure(requirement: DatasetRequirement, plan: Plan) -> str:
    """The measure `requirement` fetches, in words.

    The planner's `measure`, unless a scope decision the requirement cites
    settles its basis or its sector and the compiled plan names the source.
    Then the words are the source's `measure_label`, as `_label_dataset`
    names the retrieval, which states the basis the source counts, followed
    by the sector option the sector decision chose. The planner wrote its
    words for the source it pinned, and a switch on the plan card re-pins
    the requirement in code (`harness.graph.switch`), as an input put on its
    calculation's basis is (`submission.service._aligned_bases`), so only
    Search's words follow every pin.
    """
    governing = {
        axis: decision
        for axis, decision in _governing(requirement, plan).items()
        if axis in _MEASURE_AXES
    }
    source = _selected_source(requirement.id, plan)
    if not governing or source is None or source.measure_label is None:
        return requirement.measure
    sector = governing.get("sector")
    return f"{source.measure_label} for {sector.chosen}" if sector else source.measure_label


def _governing(requirement: DatasetRequirement, plan: Plan) -> dict[str, Decision]:
    """The scope decisions `requirement` cites, by the axis each settles."""
    return {
        decision.scope.axis: decision
        for decision in plan.decisions
        if decision.scope is not None and decision.slot in requirement.decision_ids
    }


def _divided(
    requirement: ShareOfRequirement | QuotientRequirement,
    plan: Plan,
    name: typing.Callable[[EvidenceRequirement, Plan], str],
) -> str:
    """A share or a quotient in words, each side named by `name`.

    A share names its numerator alone; a quotient names both measures.
    """
    numerator = name(plan.requirement(requirement.numerator.input_id), plan)
    if isinstance(requirement, ShareOfRequirement):
        return f"share of {numerator}"
    denominator = name(plan.requirement(requirement.denominator.input_id), plan)
    return f"{numerator} divided by {denominator}"


def _ranked_value(requirement: EvidenceRequirement, plan: Plan) -> str:
    """What the frame `requirement` produces holds, in words, which a rank orders by.

    `_measure_of` names the measure a value is a reading of; this names the
    value after each transform, in the words that transform's own sentence
    uses: "the total of ODA", "the change in ODA from 2016 to 2020", "share
    of ODA", "ODA divided by population". A filter keeps its input's values. A rank keeps them too, or
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
            f"{requirement.window_end}{_following(requirement.rolling)}"
        )
    if isinstance(requirement, ShareOfRequirement | QuotientRequirement):
        return _divided(requirement, plan, _ranked_value)
    return _measure_of(requirement, plan)


def _axis_role(requirement: RankRequirement, plan: Plan) -> str | None:
    """The role a rank's axis names: the stated axis, or the sole role its entities or
    `across` carry when the axis is left to infer from a single-role input."""
    if requirement.axis is not None:
        return requirement.axis
    datasets = datasets_of(requirement, plan)
    roles = sorted(
        {entity.role for dataset in datasets for entity in dataset.entities if entity.role}
        | {word for dataset in datasets for word in dataset.across}
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
    """The retrieval named by what it fetches and for whom.

    A compiled plan names its selected source, as `_sentence_dataset` does:
    the source's `measure_label` and the parties it holds (`_parties`),
    since the requirement's `measure` and `across` are the planner's words
    for what it asked: "every recipient" over a source of 50 countries.
    Before compile, or where the source states no label or count, the label
    keeps the requirement's words.
    """
    source = _selected_source(requirement.id, plan)
    measure = requirement.measure
    if source is not None and source.measure_label is not None:
        measure = source.measure_label
    return f"Fetch {measure} for {_parties(requirement, plan)}"


def _side_groups(requirement: DatasetRequirement, plan: Plan) -> list[tuple[str, str]]:
    """Each named set `requirement`'s selected source binds on a side of the
    flow (`SelectedSource.groups`), as that side and the set's label, in the
    source's order; empty before compile. `_parties` names a side by it."""
    source = _selected_source(requirement.id, plan)
    return [
        (side, group.label)
        for group in (source.groups if source is not None else ())
        if (side := flow_side(group.role)) is not None
    ]


def _parties(requirement: DatasetRequirement, plan: Plan, *, level: PlaceType | None = None) -> str:
    """Who `requirement` retrieves, in words, as its selected source holds them.

    The requirement's entities, then each `across` word: one on the side of
    a flow where the source binds a named set (`SelectedSource.groups`)
    reads as the set's label, since a source bound to the total for official
    donors (DAC) holds those donors and not every donor, and any other with
    how many members the source holds on its axis (`_population_count`), as
    `entities_phrase` writes it, a place word with the `level` a filter
    keeps. A set on a side no `across` word covers is left unsaid, as the
    requirement leaves it. Before compile the requirement's own words.
    """
    sides = {across_party(word) for word in requirement.across}
    groups = [(side, label) for side, label in _side_groups(requirement, plan) if side in sides]
    entities = [*requirement.entities, *(Entity(name=label) for _, label in groups)]
    grouped = {side for side, _ in groups}
    across = tuple(word for word in requirement.across if across_party(word) not in grouped)
    counts = {
        word: count
        for word in across
        if (count := _population_count(requirement, word, plan)) is not None
    }
    return entities_phrase(entities, across=across, level=level, member_counts=counts)


def _label_aggregate(requirement: AggregateRequirement, plan: Plan) -> str:
    upstream = plan.requirement(requirement.input_id)
    word = _REDUCTION_WORDS[requirement.reduction].capitalize()
    group = _group_by_phrase(requirement.group_by, breakdown=_breakdown_of(requirement, plan))
    return f"{word} {_measure_of(upstream, plan)} {group}"


def _label_growth(requirement: GrowthRequirement, plan: Plan) -> str:
    upstream = plan.requirement(requirement.input_id)
    return (
        f"Growth in {_measure_of(upstream, plan)}, "
        f"{requirement.window_start}-{requirement.window_end}{_following(requirement.rolling)}"
    )


def _side_measure(side: RequirementSide, plan: Plan) -> str:
    """What `side` reads, by its measure (`_measure_of`), with its entity (`_side_named`)."""
    return _side_named(side, _measure_of(plan.requirement(side.input_id), plan))


def _label_comparison(requirement: ComparisonRequirement, plan: Plan) -> str:
    minuend = _side_measure(requirement.minuend, plan)
    return f"Compare {minuend} to {_side_measure(requirement.subtrahend, plan)}"


def _label_sum_of(requirement: SumOfRequirement, plan: Plan) -> str:
    left = _side_measure(requirement.left, plan)
    return f"Add {left} and {_side_measure(requirement.right, plan)}"


def _label_rank(requirement: RankRequirement, plan: Plan) -> str:
    ranked = (
        _axis_word(requirement, plan)
        if requirement.limit is None
        else _kept_phrase(requirement, requirement.limit, plan)
    )
    return f"Rank {ranked} by {_ranked_value(requirement, plan)}"


def _label_share_of(requirement: ShareOfRequirement, plan: Plan) -> str:
    label = f"Share of {_side_measure(requirement.numerator, plan)}"
    if requirement.axis is not None:
        label += f" per {requirement.axis}"
    return label


def _label_quotient_of(requirement: QuotientRequirement, plan: Plan) -> str:
    numerator = _side_measure(requirement.numerator, plan)
    label = f"Divide {numerator} by {_side_measure(requirement.denominator, plan)}"
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
    SumOfRequirement: _label_sum_of,
    RankRequirement: _label_rank,
    ShareOfRequirement: _label_share_of,
    QuotientRequirement: _label_quotient_of,
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
    ends in its place among them, "(1)" or "(2)", in the order the plan
    states them. A share_of sentence that quotes a label then names
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
    """`plan`'s requirements in the order the plan states them, the order namesakes
    are numbered in (`_concepts`, `_label`).

    One order before and after compile, so a sentence written over the plan
    as proposed (`submission.service._card_names`) numbers namesakes as the
    compiled card does. The card's rows run in workflow order, so "(2)" can
    sit above "(1)". A number tells namesakes apart and says nothing of position.
    """
    return list(plan.evidence_requirements)


def _sentence_dataset(requirement: DatasetRequirement, plan: Plan) -> str:
    """The retrieval in words, with how it reads a family of StatVars.

    A compiled plan states its selected source's own `sentence`, Search's
    account of what the source holds, since the requirement's `measure`
    and `across` are the planner's words for what it asked, which a source
    over fewer parties does not meet: "every recipient" over a source of 50
    countries. Before compile the requirement is stated in those words. The
    selected source's `describe` definition stays out: for a StatVar it is
    the graph's constraint encoding ("mp=count,pt=Person"), which the
    planner model reads and a user cannot."""
    source = _selected_source(requirement.id, plan)
    clauses: list[str] = []
    if requirement.breakdown is not None:
        clauses.append(_breakdown_clause(requirement.breakdown, source))
    # A breakdown sums the family within each code, so no reading of its members applies.
    elif source is not None and (reading := _reading_clause(source)):
        clauses.append(reading)
    if source is not None:
        stated = source.sentence.strip()
        if not stated.endswith("."):
            stated += "."
        if requirement.period_end_open:
            clauses.insert(0, _FOLLOWING_YEARS)
        if not clauses:
            return stated
        read = ", ".join(clauses)
        return f"{stated} {read[0].upper()}{read[1:]}."
    parties = entities_phrase(requirement.entities, across=requirement.across)
    period = f"{requirement.period.start}-{requirement.period.end}" + (
        _FOLLOWING if requirement.period_end_open else ""
    )
    return ", ".join([f"Retrieves {requirement.measure} for {parties}", period, *clauses]) + "."


def _source(
    requirement_id: str, plan: Plan, sources: Mapping[str, SelectedSource] | None
) -> SelectedSource | None:
    """The source the compiled plan retrieves `requirement_id` from
    (`_selected_source`), else, for a plan not compiled, the one `sources`
    holds for it (`_pinned_sources`)."""
    source = _selected_source(requirement_id, plan)
    if source is None and plan.workflow is None:
        source = (sources or {}).get(requirement_id)
    return source


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


def _population_count(requirement: DatasetRequirement, word: str, plan: Plan) -> int | None:
    """How many members the selected source holds on the axis the `across` word `word` covers.

    `None` before compile, when no source is selected yet, and when
    `compiler.evidence.open_axes_for` matches `requirement.across` to no
    set of open axes.
    """
    source = _selected_source(requirement.id, plan)
    if source is None or word not in requirement.across:
        return None
    axes = open_axes_for(requirement.across, source.open_entity_axes)
    if isinstance(axes, str):
        return None
    return axes[requirement.across.index(word)].member_count


def _reading_clause(source: SelectedSource) -> str | None:
    """How the retrieval reads `source`'s family of StatVars, or `None` for a source binding one.

    The user reads a StatVar as an indicator, the noun the card attribution
    counts them by. For a source binding every entity, the count is the one
    the answer shows: the members `search` confirmed to hold data at those
    entities in the plan's years (`stat_var_member_count`), the series the
    retrieval holds, as "summed over its 13 indicators". On a source leaving
    an entity axis open (`open_entity_axes`) `search` confirms members at
    probe places on a place axis, and over group totals as well as parties
    on a role axis, so the retrieval can hold other members than it counts.
    Such a source, and one whose confirmed count `search` states as a floor
    or not at all, is counted by the family `served_count` counts, whatever
    its years, as "summed over whichever of its 288 indicators hold data".
    A family without that count, or counted at fewer than two members, is
    named without a count. Says when the reading is the compiler's default
    (`ReadingSetBy.RULE`), so the user knows the plan can be amended to read
    it the other way.

    A family read separately states no clause when it counts one StatVar and
    varies along no open axis: the card names one indicator, and there is
    nothing on it to keep apart. The retrieval's arguments still state the
    reading.
    """
    if source.stat_var_reading is None:
        return None
    count = stat_var_count(source)
    if source.stat_var_reading is StatVarReading.SEPARATE and count == 1 and not source.open_axes:
        return None
    summed = source.stat_var_reading is StatVarReading.SUM
    retrieved = (
        source.stat_var_member_count
        if source.stat_var_member_count_exact is not False and not source.open_entity_axes
        else None
    )
    if retrieved is not None and retrieved > 1:
        clause = (
            f"summed over its {retrieved:,} indicators"
            if summed
            else f"kept as {retrieved:,} separate indicators"
        )
    elif retrieved is None and (served := source.served_count or 0) > 1:
        clause = (
            f"summed over whichever of its {served:,} indicators hold data"
            if summed
            else f"kept separate over whichever of its {served:,} indicators hold data"
        )
    else:
        clause = "summed over its indicators" if summed else "kept as separate indicators"
    return f"{clause} by default" if source.stat_var_reading_set_by is ReadingSetBy.RULE else clause


def _sentence_aggregate(requirement: AggregateRequirement, plan: Plan) -> str:
    upstream = plan.requirement(requirement.input_id)
    word = _REDUCTION_WORDS[requirement.reduction]
    group = _group_by_phrase(
        requirement.group_by, breakdown=_breakdown_of(requirement, plan), grouped=True
    )
    return f"The {word} of {_measure_of(upstream, plan)}, {group}."


def _sentence_growth(requirement: GrowthRequirement, plan: Plan) -> str:
    upstream = plan.requirement(requirement.input_id)
    change_text = (
        "a percentage change"
        if requirement.change is GrowthChange.RELATIVE
        else "an absolute change, in the input's own unit"
    )
    return (
        f"The change in {_measure_of(upstream, plan)} from {requirement.window_start} to "
        f"{requirement.window_end}{_following(requirement.rolling)}, read as {change_text}."
    )


def _sentence_comparison(requirement: ComparisonRequirement, plan: Plan) -> str:
    minuend = _side_measure(requirement.minuend, plan)
    return (
        f"{minuend[:1].upper()}{minuend[1:]} minus {_side_measure(requirement.subtrahend, plan)}, "
        "at every period the two share."
    )


def _sentence_sum_of(requirement: SumOfRequirement, plan: Plan) -> str:
    left = plan.requirement(requirement.left.input_id)
    right = plan.requirement(requirement.right.input_id)
    return (
        f"Adds {_measure_of(left, plan)} and {_measure_of(right, plan)} for each year both "
        "report; years one side lacks are left out."
    )


def _sentence_rank(requirement: RankRequirement, plan: Plan) -> str:
    role = _axis_role(requirement, plan)
    datasets = datasets_of(requirement, plan)
    covering = next(
        (dataset for dataset in datasets if role is not None and role in dataset.across),
        None,
    )
    # A breakdown's codes are every code the source holds at its level, uncounted.
    by_breakdown = any(
        dataset.breakdown is not None and dataset.breakdown.dimension.value == role
        for dataset in datasets
    )
    # The source counts every place on a place axis, so a level kept from it
    # is named without a count.
    level = _place_level_word(requirement, plan)
    among_text = among_phrase(
        requirement.among,
        across=(level or role) if covering is not None or by_breakdown else None,
        member_count=(
            _population_count(covering, role, plan)
            if covering is not None and role is not None and level is None
            else None
        ),
    )
    # A one-year window holds one row per entity, so it has nothing to collapse.
    one_year = requirement.window_start is not None and (
        requirement.window_start == requirement.window_end
    )
    per_period_text = (
        f"ranks {requirement.window_start}{_following(requirement.rolling)}"
        if one_year and not requirement.per_period
        else per_period_phrase(requirement.per_period, limited=requirement.limit is not None)
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
    divisor = _measure_of(denominator, plan)
    # A denominator narrowed to fewer members than its sources hold names the
    # step that narrows it, whose own line states the filter or limit, in
    # place of the parties its sources hold. A filter keeping a level of place
    # is named by that level.
    narrowing = narrowing_step(denominator, plan)
    if narrowing is not None and not (
        isinstance(narrowing, FilterRequirement)
        and all(isinstance(p, PeriodThreshold | PlaceLevel) for p in narrowing.predicates)
    ):
        return (
            f"Share of {measure} as one figure overall, divided by one total of {divisor} "
            f'over the rows "{_label(narrowing, plan)}" keeps.'
        )
    # A pooled share over a frame that names no place reads one level of it, which
    # the denominator's level filter keeps.
    level = kept_level(denominator, plan)
    parties = "; ".join(
        dict.fromkeys(
            _parties(dataset, plan, level=level) for dataset in datasets_of(denominator, plan)
        )
    )
    return f"Share of {measure} as one figure overall, divided by {divisor} ({parties})."


def _sentence_quotient_of(requirement: QuotientRequirement, plan: Plan) -> str:
    """The quotient in words, naming each side by its measure.

    A keyed quotient says whose denominator each entity divides by: its own
    counterpart when the requirement states `denominator_axis`, else the
    denominator's one total.
    """
    numerator = _measure_of(plan.requirement(requirement.numerator.input_id), plan)
    denominator = _measure_of(plan.requirement(requirement.denominator.input_id), plan)
    if requirement.axis is None:
        sentence = f"{numerator} divided by {denominator}"
    elif requirement.denominator_axis is not None:
        sentence = f"{numerator} per {requirement.axis}, each divided by its own {denominator}"
    else:
        sentence = f"{numerator} per {requirement.axis}, each divided by {denominator}"
    # `str.capitalize` would lower the rest of an acronym such as "GDP".
    return f"{sentence[0].upper()}{sentence[1:]}, at each period both report."


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
    SumOfRequirement: _sentence_sum_of,
    RankRequirement: _sentence_rank,
    ShareOfRequirement: _sentence_share_of,
    QuotientRequirement: _sentence_quotient_of,
    JoinRequirement: _sentence_join,
    FilterRequirement: _sentence_filter,
}


def _sentence(requirement: EvidenceRequirement, plan: Plan) -> str:
    """One prose sentence stating what approving `requirement` approves.

    A rank sentence states `among`, `per_period` and any `limit`. A keyed share_of
    sentence states `axis` and `denominator_scope`, and a keyed quotient_of sentence
    `axis` and whose denominator each entity divides by. These are population and
    series-shape claims a person reads in words rather than inferring from a
    field name, worded by `planner.phrasing`.
    """
    try:
        builder = _SENTENCE_BUILDERS[type(requirement)]
    except KeyError:
        raise AssertionError(f"unhandled requirement kind: {requirement!r}") from None
    return builder(requirement, plan)
