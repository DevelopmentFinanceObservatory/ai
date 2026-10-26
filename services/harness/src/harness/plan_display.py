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

A step's `label` names the work a workflow step does and the entity it does
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

import re
import typing
from collections.abc import Collection, Mapping, Sequence

from compiler.evidence import open_axes_for
from compiler.shapes import breakdown_level
from compiler.sources import stat_var_count, stat_var_count_is_floor
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
    Decision,
    EvidenceRequirement,
    FilterRequirement,
    GrowthRequirement,
    JoinRequirement,
    Plan,
    QuotientRequirement,
    RankRequirement,
    ResolvedBy,
    ShareOfRequirement,
    SumOfRequirement,
    answering_results,
    authored,
    authored_text,
    copy_origins,
    frame_shapes,
    is_scope_choice,
    is_search_pin,
)
from planner.records import (
    Breakdown,
    EntityIn,
    EvidenceIssue,
    GroupBy,
    GrowthChange,
    PeriodThreshold,
    PlaceLevel,
    Predicate,
    Reduction,
    RollingWindow,
    StatVarReading,
    ValueThreshold,
)
from planner.shapes import FrameShape, RoleDimension, TimeLabel
from planner.workflow import ReadingSetBy, SelectedSource, Workflow

from harness.sentences import DECLINE_REASONS

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
    scopes = answer_scopes(plan)
    shown = shown_plan(plan)
    assert shown.translation is not None
    assert shown.workflow is not None
    operations_by_id = {operation.id: operation for operation in shown.translation.operations}
    rows: list[dict[str, object]] = []
    for step in shown.workflow.steps:
        operation = operations_by_id[step.operation_id]
        requirement = shown.requirement(operation.requirement_id)
        description = _sentence(requirement, shown)
        if requirement.id in scopes:
            description = f"{description} Result: {scopes[requirement.id]}."
        rows.append(
            {
                "id": step.id,
                "label": _label(requirement, shown),
                "description": description,
                "args": dict(step.args),
            }
        )
    return rows


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


def listed_together(plan: Plan, slot: str) -> tuple[Decision, ...]:
    """The decisions of `plan` the card's row for `slot` stands for: the
    decision under `slot` and every one listed with it (`_listed_slots`);
    empty when no decision of `plan` holds `slot`."""
    listed = _listed_slots(plan)
    row = listed.get(slot)
    if row is None:
        return ()
    return tuple(decision for decision in plan.decisions if listed.get(decision.slot) == row)


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
    phrase = "one figure" + "".join(f" per {word}" for word in per)
    if shape.period is None:
        return phrase
    years = (
        str(shape.period.start)
        if shape.period.start == shape.period.end
        else f"{shape.period.start}-{shape.period.end}"
    )
    return f"{phrase}, {years}"


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
        return requirement.measure
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
    the source's `measure_label` and how many members the source holds on
    each axis `across` covers (`_population_count`), since the requirement's
    `measure` and `across` are the planner's words for what it asked: "every
    recipient" over a source of 50 countries. Before compile, or where the
    source states no label or count, the label keeps the requirement's words.
    """
    source = _selected_source(requirement.id, plan)
    measure = requirement.measure
    if source is not None and source.measure_label is not None:
        measure = source.measure_label
    counts = {
        word: count
        for word in requirement.across
        if (count := _population_count(requirement, word, plan)) is not None
    }
    parties = entities_phrase(requirement.entities, across=requirement.across, member_counts=counts)
    return f"Fetch {measure} for {parties}"


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


def _label_comparison(requirement: ComparisonRequirement, plan: Plan) -> str:
    minuend = plan.requirement(requirement.minuend.input_id)
    subtrahend = plan.requirement(requirement.subtrahend.input_id)
    return f"Compare {_measure_of(minuend, plan)} to {_measure_of(subtrahend, plan)}"


def _label_sum_of(requirement: SumOfRequirement, plan: Plan) -> str:
    left = plan.requirement(requirement.left.input_id)
    right = plan.requirement(requirement.right.input_id)
    return f"Add {_measure_of(left, plan)} and {_measure_of(right, plan)}"


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


def _label_quotient_of(requirement: QuotientRequirement, plan: Plan) -> str:
    numerator = plan.requirement(requirement.numerator.input_id)
    denominator = plan.requirement(requirement.denominator.input_id)
    label = f"Divide {_measure_of(numerator, plan)} by {_measure_of(denominator, plan)}"
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
    counts them by. Counts the family as the source does (`stat_var_count`).
    A family counted at fewer than two members, or at a floor
    (`stat_var_count_is_floor`), is named without a count: its retrieval
    holds every member with data, which can be more or fewer than the
    members `search` confirmed. Says when the reading is the compiler's
    default (`ReadingSetBy.RULE`), so the user knows the plan can be amended
    to read it the other way.

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
    if count < 2 or stat_var_count_is_floor(source):
        clause = "summed over its indicators" if summed else "kept as separate indicators"
    else:
        clause = (
            f"summed over its {count:,} indicators"
            if summed
            else f"kept as {count:,} separate indicators"
        )
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
    minuend = plan.requirement(requirement.minuend.input_id)
    subtrahend = plan.requirement(requirement.subtrahend.input_id)
    return (
        f"{_measure_of(minuend, plan).capitalize()} minus {_measure_of(subtrahend, plan)}, at every "
        "period the two share."
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
