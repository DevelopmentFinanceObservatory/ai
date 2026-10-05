"""Compiled frame shapes: `planner.shapes`, driven from operations and resolved sources.

`planner.shapes` states the shape algebra over primitives -- `FrameShape`,
`RoleDimension`, `Period`, plain strings and enums -- so it can be driven from
either side without an import cycle. This module is the compiler's driver: it
walks a translation's operations in topological order, the same way
`planner.plan` walks a plan's requirements, and calls the same transforms and
the same `rejects` on the same primitives. What it adds is `ResolvedRole`: a
role dimension carries a plan-vocabulary `role` for `rejects` to match
`axis` against, and this module wraps each one with the literal source column
behind it (`ResolvedRole.column`), which is what the compiler needs to write
into a tool call.

`shapes_for` returns two things: `ResolvedShape` per operation that could be
given one, and one reason string per operation whose own input the algebra
refused. An operation reading a `compare` or a `join` is left out of both --
that refusal is `compiler.compile`'s own `INPUT_IS_TERMINAL`, a graph-level
rule, not a shape one, so this module declines silently rather than
double-reporting it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace

from planner.operations import (
    AggregateOperation,
    CompareOperation,
    FilterOperation,
    GroupBy,
    GrowthOperation,
    JoinOperation,
    Operation,
    RankOperation,
    RetrieveOperation,
    ShareOfOperation,
    SumOfOperation,
)
from planner.plan import (
    DatasetRequirement,
    Entity,
    EvidenceRequirement,
    keeps_one_level,
    states_period_threshold,
)
from planner.records import (
    Breakdown,
    BreakdownDimension,
    EntityIn,
    Identifier,
    Predicate,
    StatVarReading,
)
from planner.shapes import SEVERAL, FrameShape, MeasureCount, RoleDimension, ShapeVerb, year_filter
from planner.shapes import aggregated as _aggregated
from planner.shapes import compared as _compared
from planner.shapes import filtered as _filtered
from planner.shapes import grown as _grown
from planner.shapes import joined as _joined
from planner.shapes import ranked as _ranked
from planner.shapes import rejects as _rejects
from planner.shapes import retrieved as _retrieved
from planner.shapes import shared as _shared
from planner.shapes import summed_sides as _summed_sides
from planner.workflow import Period, SelectedSource

from compiler.evidence import open_axis_for, opens_place_axis, resolve_entity
from compiler.sources import stat_var_count

# The code system behind each breakdown dimension: the `get_data` breakdown's
# `code_system`, the open axis a source's family varies along for it, and the
# `stat_var_<code system>` column its rolled-up codes sit in.
BREAKDOWN_CODE_SYSTEMS: Mapping[BreakdownDimension, str] = {
    BreakdownDimension.SECTOR: "crsPurpose",
}


def breakdown_column(breakdown: Breakdown) -> str:
    """The frame column a retrieval broken down by `breakdown` names each row's code in."""
    return f"stat_var_{BREAKDOWN_CODE_SYSTEMS[breakdown.dimension]}"


@dataclass(frozen=True, slots=True)
class ResolvedRole:
    """One `RoleDimension` of a compiled shape, wrapped with the column behind it.

    `column` is the literal column the frame carries for this role --
    `_role_columns`' own value, now computed once into the shape rather than
    re-walked at every argument site. `role` and `requirement_id` mirror the
    `RoleDimension` this wraps. `spec_id` is the source the column's values
    come from. `narrowed` mirrors the shape's own flag. `surplus` is whether
    the source resolves a label under this column that the requirement does
    not name -- computed once at the retrieval and carried forward unchanged,
    since a column that started out wider than the requirement stays so
    wherever the frame travels, until a filter narrows it. `breakdown`
    mirrors the dimension's own flag: the column holds a breakdown's codes.
    """

    column: str
    role: str | None
    requirement_id: Identifier
    spec_id: str
    narrowed: bool
    surplus: bool
    breakdown: bool = False


@dataclass(frozen=True, slots=True)
class ResolvedShape:
    """A `FrameShape`, plus the literal column behind each of its role dimensions.

    `columns` is positional with `shape.roles`: entry `i` here is the column
    behind role dimension `i` there. Empty whenever `shape.roles` is --
    `compare`'s own shape above all, which carries none since nothing may
    read one.
    """

    shape: FrameShape
    columns: tuple[ResolvedRole, ...]


def shapes_for(
    operations: Mapping[str, Operation],
    *,
    requirements: Mapping[str, EvidenceRequirement],
    sources: Mapping[str, SelectedSource],
    order: Sequence[str],
) -> tuple[dict[str, ResolvedShape], list[str]]:
    """Walk `operations` in `order`, resolving one `ResolvedShape` per operation that has one.

    Mirrors `planner.plan._build_shapes`: each operation's shape is computed
    from its already-built inputs', through the one transform
    `planner.shapes` states for its kind, after `planner.shapes.rejects`
    confirms the input may feed it. An operation is left out of the returned
    dict, with no entry in the issues list either, when its own input has no
    resolved shape yet -- a retrieval with no source, a requirement kind
    mismatch, a dangling input, or a reader of a `compare`/`join` -- since
    each of those is already a `compiler.compile` issue in its own right and
    this module has nothing to add.

    Args:
        operations: Every operation the translation names, by id.
        requirements: Every requirement the plan states, by id.
        sources: The source discovered for each retrieval, by *operation* id.
        order: Operation ids topologically ordered so every input precedes
            its reader.

    Returns:
        A `(shapes_by_id, issues)` pair: the shape resolved for each
        operation that has one, and one `f"{operation_id}: {reason}"` string
        per operation whose own `rejects` check refused its input, for
        `compiler.compile` to report as `INPUT_SHAPE_UNSUPPORTED`.
    """
    retrieve_by_requirement = {
        operation.requirement_id: operation.id
        for operation in operations.values()
        if isinstance(operation, RetrieveOperation)
    }
    shapes: dict[str, ResolvedShape] = {}
    issues: list[str] = []
    for op_id in order:
        operation = operations.get(op_id)
        if operation is None:
            continue
        built = _resolve(
            operation, operations, shapes, requirements, sources, retrieve_by_requirement
        )
        if built is None:
            continue
        if isinstance(built, str):
            issues.append(f"{op_id}: {built}")
            continue
        shapes[op_id] = built
    return shapes, issues


def _input_shape(
    op_id: Identifier, operations: Mapping[str, Operation], shapes: Mapping[str, ResolvedShape]
) -> ResolvedShape | None:
    """`shapes[op_id]`, or `None` when `op_id` has none or names a terminal-by-contract verb.

    A `compare` or a `join` gets its own shape built (for completeness -- see
    each transform's docstring), but nothing may read one; declining here
    lets `compiler.compile`'s own `INPUT_IS_TERMINAL` be the only issue a
    reader of one produces.
    """
    if isinstance(operations.get(op_id), CompareOperation | JoinOperation):
        return None
    return shapes.get(op_id)


def _resolve(  # noqa: PLR0911 -- one isinstance branch per operation kind narrows `ty`'s type
    # for each `_xxx_resolved` call; a dict keyed by `type(operation)` cannot narrow it and
    # every branch would need an `assert isinstance(...)` restoring the same length.
    operation: Operation,
    operations: Mapping[str, Operation],
    shapes: Mapping[str, ResolvedShape],
    requirements: Mapping[str, EvidenceRequirement],
    sources: Mapping[str, SelectedSource],
    retrieve_by_requirement: Mapping[str, str],
) -> ResolvedShape | str | None:
    if isinstance(operation, RetrieveOperation):
        return _retrieve_resolved(operation, requirements, sources)
    if isinstance(operation, AggregateOperation):
        upstream = _input_shape(operation.input, operations, shapes)
        return None if upstream is None else _aggregate_resolved(operation, upstream)
    if isinstance(operation, GrowthOperation):
        upstream = _input_shape(operation.input, operations, shapes)
        return None if upstream is None else _growth_resolved(operation, upstream)
    if isinstance(operation, RankOperation):
        upstream = _input_shape(operation.input, operations, shapes)
        return None if upstream is None else _rank_resolved(operation, upstream)
    if isinstance(operation, FilterOperation):
        upstream = _input_shape(operation.input, operations, shapes)
        return (
            None
            if upstream is None
            else _filter_resolved(operation, upstream, sources, retrieve_by_requirement)
        )
    if isinstance(operation, ShareOfOperation):
        numerator = _input_shape(operation.numerator.input, operations, shapes)
        denominator = _input_shape(operation.denominator.input, operations, shapes)
        if numerator is None or denominator is None:
            return None
        return _share_of_resolved(operation, numerator, denominator)
    if isinstance(operation, CompareOperation):
        left = _input_shape(operation.left.input, operations, shapes)
        right = _input_shape(operation.right.input, operations, shapes)
        if left is None or right is None:
            return None
        return _compare_resolved(left, right)
    if isinstance(operation, SumOfOperation):
        left = _input_shape(operation.left.input, operations, shapes)
        right = _input_shape(operation.right.input, operations, shapes)
        if left is None or right is None:
            return None
        return _sum_of_resolved(left, right)
    assert isinstance(operation, JoinOperation)
    left = _input_shape(operation.left, operations, shapes)
    right = _input_shape(operation.right, operations, shapes)
    if left is None or right is None:
        return None
    return _join_resolved(left, right)


# --- Per-verb resolution -----------------------------------------------------


def _unsummable_axes(source: SelectedSource) -> tuple[str, ...] | None:
    """The open axes of `source`'s family that dfo-apis may not sum over, or `None` when it has none.

    A family varying along no open axis differs in something no slot
    states, so whether dfo-apis can sum it is unknown.
    """
    if not source.open_axes:
        return None
    return tuple(axis.property for axis in source.open_axes if not axis.summable)


def _separate_measures(source: SelectedSource) -> MeasureCount:
    """How many measures a family read separately carries.

    A retrieval of a family holds every member of its binding that has data.
    The StatVars `search` confirmed count them (`stat_var_count`): exactly
    when the listing is complete (`stat_var_truncated` is false), so a
    family `search` confirmed one member of carries one measure, and as a
    lower bound otherwise. The count is `SEVERAL` where it bounds nothing
    above one: no StatVar confirmed, and one of a listing that is capped or
    does not say whether it is.
    """
    count = stat_var_count(source)
    if count > 1 or (count == 1 and source.stat_var_truncated is False):
        return count
    return SEVERAL


def _retrieve_resolved(
    operation: RetrieveOperation,
    requirements: Mapping[str, EvidenceRequirement],
    sources: Mapping[str, SelectedSource],
) -> ResolvedShape | None:
    """The shape a fresh retrieval carries, with each role wrapped by its source column.

    The columns kept are the ones the requirement's own entities resolve
    under, in the source's order, as `compile._role_columns` reads them,
    then the open entity axis `compiler.evidence.open_axis_for` matches to
    its `across`. A requirement with no `across` falls back to every column
    the source resolves when none of its entities do. An `across` no open
    entity axis matches builds no shape: `compile._check_sources` reports it
    as the source's `open_axis_unresolved` finding.

    A requirement stating a breakdown keeps its codes' column last
    (`breakdown_column`), a role nothing in the source resolves and so no
    surplus. A family read separately carries the measures its source counts
    (`_separate_measures`) and the axes it may not be summed over
    (`_unsummable_axes`). A summed family, a breakdown (which sums the family
    within each code), and a source binding one StatVar carry one. A source
    that leaves its place axis open carries every level of place (`compiler.evidence.opens_place_axis`), and
    `compiler.evidence` refuses one under a requirement whose `across` does
    not name the place party, so the plan's own reading
    (`planner.plan.names_no_place`) agrees.
    """
    source = sources.get(operation.id)
    requirement = requirements.get(operation.requirement_id)
    if source is None or not isinstance(requirement, DatasetRequirement):
        return None
    every_column: list[str] = []
    for entity in source.entities:
        if entity.role not in every_column:
            every_column.append(entity.role)
    named = {entity.name.casefold() for entity in requirement.entities}
    wanted: set[str] = set()
    phrase_by_column: dict[str, str] = {}
    for entity in requirement.entities:
        resolved = resolve_entity(entity, source.entities)
        if resolved is None:
            continue
        wanted.add(resolved.role)
        if entity.role is not None:
            phrase_by_column.setdefault(resolved.role, entity.role)
    chosen = [column for column in every_column if column in wanted]
    if requirement.across is None:
        chosen = chosen or every_column
    else:
        axis = open_axis_for(requirement.across, source.open_entity_axes)
        if isinstance(axis, str):
            return None
        chosen.append(axis.column)
        phrase_by_column[axis.column] = requirement.across
    columns: list[ResolvedRole] = []
    dims: list[RoleDimension] = []
    for column in chosen:
        phrase = phrase_by_column.get(column, column)
        surplus = any(
            entity.role == column and entity.label.casefold() not in named
            for entity in source.entities
        )
        dims.append(RoleDimension(role=phrase, requirement_id=operation.requirement_id))
        columns.append(
            ResolvedRole(
                column=column,
                role=phrase,
                requirement_id=operation.requirement_id,
                spec_id=source.spec_id,
                narrowed=False,
                surplus=surplus,
            )
        )
    if requirement.breakdown is not None:
        role = requirement.breakdown.dimension.value
        dims.append(
            RoleDimension(role=role, requirement_id=operation.requirement_id, breakdown=True)
        )
        columns.append(
            ResolvedRole(
                column=breakdown_column(requirement.breakdown),
                role=role,
                requirement_id=operation.requirement_id,
                spec_id=source.spec_id,
                narrowed=False,
                surplus=False,
                breakdown=True,
            )
        )
    separate = source.stat_var_reading is StatVarReading.SEPARATE and requirement.breakdown is None
    shape = _retrieved(
        requirement_id=operation.requirement_id,
        roles=tuple(dims),
        period=requirement.period,
        measures=_separate_measures(source) if separate else 1,
        unsummable_axes=_unsummable_axes(source) if separate else None,
        names_no_place=opens_place_axis(source.open_entity_axes),
    )
    return ResolvedShape(shape=shape, columns=tuple(columns))


def _carry_columns(
    previous: tuple[ResolvedRole, ...], roles: tuple[RoleDimension, ...]
) -> tuple[ResolvedRole, ...]:
    """Rebuild `ResolvedRole`s for `roles`, reusing the column behind each matching entry of `previous`.

    Every transform in `planner.shapes` either carries an input `RoleDimension`
    through unchanged or selects one from a single side, so `(role,
    requirement_id)` always finds its match in `previous` -- this only
    updates `narrowed` to what the new shape says, since `filtered` is the
    one transform that flips it.
    """
    by_key = {(role.role, role.requirement_id): role for role in previous}
    result: list[ResolvedRole] = []
    for dim in roles:
        prior = by_key.get((dim.role, dim.requirement_id))
        if prior is not None:
            result.append(replace(prior, narrowed=dim.narrowed))
    return tuple(result)


def _aggregate_resolved(
    operation: AggregateOperation, upstream: ResolvedShape
) -> ResolvedShape | str:
    by_entity = GroupBy.ENTITY in operation.group_by
    by_period = GroupBy.PERIOD in operation.group_by
    by_breakdown = GroupBy.BREAKDOWN in operation.group_by
    reason = _rejects(
        ShapeVerb.AGGREGATE,
        upstream.shape,
        by_entity=by_entity,
        by_period=by_period,
        by_breakdown=by_breakdown,
    )
    if reason is not None:
        return reason
    shape = _aggregated(
        upstream.shape,
        reduction=operation.reduction,
        by_entity=by_entity,
        by_period=by_period,
        by_breakdown=by_breakdown,
    )
    return ResolvedShape(shape=shape, columns=_carry_columns(upstream.columns, shape.roles))


def _growth_resolved(operation: GrowthOperation, upstream: ResolvedShape) -> ResolvedShape | str:
    reason = _rejects(ShapeVerb.GROWTH, upstream.shape, change=operation.change)
    if reason is not None:
        return reason
    window = Period(start=operation.window_start, end=operation.window_end)
    shape = _grown(upstream.shape, change=operation.change, window=window)
    return ResolvedShape(shape=shape, columns=_carry_columns(upstream.columns, shape.roles))


def _rank_resolved(operation: RankOperation, upstream: ResolvedShape) -> ResolvedShape | str:
    window = (
        Period(start=operation.window_start, end=operation.window_end)
        if operation.window_start is not None and operation.window_end is not None
        else None
    )
    reason = _rejects(
        ShapeVerb.RANK,
        upstream.shape,
        axis=operation.axis,
        among=operation.among,
        window=window,
        per_period=operation.per_period,
    )
    if reason is not None:
        return reason
    shape = _ranked(
        upstream.shape,
        axis=operation.axis,
        window=window,
        per_period=operation.per_period,
        limit=operation.limit,
    )
    return ResolvedShape(shape=shape, columns=_carry_columns(upstream.columns, shape.roles))


def _resolved_narrowed_roles(
    predicates: Sequence[Predicate],
    columns: tuple[ResolvedRole, ...],
    sources: Mapping[str, SelectedSource],
    retrieve_by_requirement: Mapping[str, str],
) -> tuple[str | None, ...]:
    """Which of `columns`' own role phrases an `EntityIn` predicate actually narrows.

    A predicate that states its own `role` narrows that phrase directly --
    the plan already said which dimension it means. An unqualified predicate
    -- `role is None` -- names no dimension, so each of `columns` is checked
    in turn: `resolve_entity` matches the predicate's names against that
    column's own source, the same way a fresh retrieval resolves its own
    entities, and a name landing on that column's literal source column
    narrows that dimension alone. A name the source cannot resolve, or that
    lands on a different column, narrows nothing here -- leaving that
    dimension's `narrowed` flag exactly as `filtered` already carries it in,
    which is no worse than what this can prove.
    """
    narrowed: set[str | None] = set()
    for predicate in predicates:
        if not isinstance(predicate, EntityIn):
            continue
        if predicate.role is not None:
            narrowed.add(predicate.role)
            continue
        for column in columns:
            retrieve_id = retrieve_by_requirement.get(column.requirement_id)
            source = sources.get(retrieve_id) if retrieve_id is not None else None
            if source is None:
                continue
            for name in predicate.names:
                resolved = resolve_entity(Entity(name=name, role=None), source.entities)
                if resolved is not None and resolved.role == column.column:
                    narrowed.add(column.role)
                    break
    return tuple(narrowed)


def _filter_resolved(
    operation: FilterOperation,
    upstream: ResolvedShape,
    sources: Mapping[str, SelectedSource],
    retrieve_by_requirement: Mapping[str, str],
) -> ResolvedShape | str:
    place_level = keeps_one_level(operation.predicates)
    reason = _rejects(
        ShapeVerb.FILTER,
        upstream.shape,
        place_level=place_level,
        period_threshold=states_period_threshold(operation.predicates),
    )
    if reason is not None:
        return reason
    years = year_filter(upstream.shape.period, operation.predicates)
    if isinstance(years, str):
        return years
    narrowed_roles = _resolved_narrowed_roles(
        operation.predicates, upstream.columns, sources, retrieve_by_requirement
    )
    shape = _filtered(
        upstream.shape,
        narrowed_roles=narrowed_roles,
        period=years.period,
        pins_one_year=years.leaves_one_year,
        keeps_one_level=place_level,
    )
    return ResolvedShape(shape=shape, columns=_carry_columns(upstream.columns, shape.roles))


def _share_of_resolved(
    operation: ShareOfOperation, numerator: ResolvedShape, denominator: ResolvedShape
) -> ResolvedShape | str:
    # The denominator's own *in* rule is checked with no axis -- the numerator
    # alone gives the output its role lineage -- with the denominator passed
    # as `other` to the numerator's own call below for the per-axis
    # cross-check.
    reason = _rejects(
        ShapeVerb.SHARE_OF, denominator.shape, axis=None, entity=operation.denominator.entity
    )
    if reason is not None:
        return reason
    reason = _rejects(
        ShapeVerb.SHARE_OF,
        numerator.shape,
        axis=operation.axis,
        scope=operation.denominator_scope,
        other=denominator.shape,
        entity=operation.numerator.entity,
    )
    if reason is not None:
        return reason
    shape = _shared(
        numerator.shape, denominator.shape, axis=operation.axis, scope=operation.denominator_scope
    )
    return ResolvedShape(shape=shape, columns=_carry_columns(numerator.columns, shape.roles))


def _compare_resolved(left: ResolvedShape, right: ResolvedShape) -> ResolvedShape | str:
    for shape, other in ((left.shape, right.shape), (right.shape, left.shape)):
        reason = _rejects(ShapeVerb.COMPARE, shape, other=other)
        if reason is not None:
            return reason
    shape = _compared(left.shape, right.shape)
    return ResolvedShape(shape=shape, columns=())


def _sum_of_resolved(left: ResolvedShape, right: ResolvedShape) -> ResolvedShape | str:
    for shape, other in ((left.shape, right.shape), (right.shape, left.shape)):
        reason = _rejects(ShapeVerb.SUM_OF, shape, other=other)
        if reason is not None:
            return reason
    return ResolvedShape(shape=_summed_sides(left.shape, right.shape), columns=())


def _join_resolved(left: ResolvedShape, right: ResolvedShape) -> ResolvedShape | str:
    for shape, other in ((left.shape, right.shape), (right.shape, left.shape)):
        reason = _rejects(ShapeVerb.JOIN, shape, other=other)
        if reason is not None:
            return reason
    shape = _joined(left.shape, right.shape)
    return ResolvedShape(shape=shape, columns=_carry_columns(left.columns, shape.roles))
