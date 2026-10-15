"""The frame-shape algebra: what a step's output looks like, and what may read it.

A retrieval and every calculation verb downstream of it produces a frame with
a shape -- which role columns it keeps one row per, how many measures it
carries, what kind of number its `value` column holds, whether its rows are
real observations or a collapsed span, whether a rank limit kept only some
entities and cut rows out of it, whether its places mix levels, and which
retrievals it ultimately rests on. `FrameShape`
states that shape as a value; the ten transforms below compute one verb's
output shape from its input's; `rejects` states, for one verb reading one
shape, the reason it may not -- reader-facing prose, or `None` when it may.

This module imports no requirement or operation class. Every transform and
every `rejects` rule takes primitives -- `FrameShape`, `RoleDimension`,
`Period`, plain strings and enums -- which is what lets both `planner.plan`
(driving it from requirements) and `compiler.shapes` (driving it from
operations, through a wrapper that resolves each role to a source column)
share one rule set with no import cycle between them.

`rejects` is the single place a verb's *in* rule is written. A caller that
wants to know whether a step may run calls it before building the step;
nothing here raises -- a `FrameShape` and a `RoleDimension` are themselves
unconditionally constructible, since it is `rejects`, not construction, that
carries every rule a legal plan has to satisfy.

`roles_named` is the single place a role phrase is matched to a role
dimension: an `axis`, a `denominator_axis`, and the compiler's lookup of the
column behind either.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Final, Literal, Protocol

from planner.parties import across_party
from planner.records import (
    DenominatorScope,
    GrowthChange,
    Identifier,
    Period,
    PeriodThreshold,
    Predicate,
    RankAmong,
    Reduction,
)


class ValueKind(StrEnum):
    """What kind of number a frame's `value` column holds.

    `MEASURED` is a value in a unit: the source's own, or for a quotient of
    two measures the unit the two compose (`FrameShape.composed_unit`).
    `RATIO` is a ratio of two values, either a share, in per cent
    (`common.unit_refs.PERCENT_UNIT`), or a relative growth, with no unit.
    `DIFFERENCE` is a change in the input's
    unit -- an absolute growth. `ORDINAL` is a rank position, or a change in
    one counted in places: no unit, no zero, no ratio. It matches
    analysis-mcp's `DerivedMarker.ordinal`, which every verb carries onward,
    so `rank`, `share_of`, `quotient_of` and a relative `growth` refuse it
    here as analysis-mcp does.
    """

    MEASURED = "measured"
    RATIO = "ratio"
    DIFFERENCE = "difference"
    ORDINAL = "ordinal"


class TimeLabel(StrEnum):
    """What a frame's `time_period` names: a series of years, one picked year, or a span.

    `OBSERVATION` is a real period each row was actually observed in, a
    series of years per entity.

    `SPAN` marks a value that collapsed more than one period into one row,
    which analysis-mcp labels with one ISO 8601 interval such as
    `2018/2022` (or the point, when the collapse covered one period).
    `growth` refuses a span; `rank`, `share_of`, `quotient_of`, `compare` and
    `join` read it as one period, matched by its label; `filter` refuses a year
    threshold over a span labelled with an interval.

    `PICKED` marks one observed year per entity, picked by a `max` or `min`
    over several years: analysis-mcp labels each entity's row with the year
    its extreme fell in, the earliest among ties, so entities carry
    different years. Each row's year is real, so a year threshold `filter`
    reads it, and `compare`, `quotient_of` and `join` match it on each
    entity's own year against a series of years whose period covers the
    picked frame's. Every
    reader that sets entities side by side within one year, or follows one
    entity across years, refuses it: `growth`, `rank` in either mode (its
    coverage counts every year of the window, and each entity holds one),
    an `aggregate` by period, `share_of` (`_rejects_picked_share`), and a
    `compare`, `quotient_of` or `join` against another
    picked frame, a span, or a series that misses some of the picked years. A `sum` or
    `mean` over it is a `SPAN` whose years are known only at run time. A
    frame over a single year is never `PICKED`: its one year is every
    entity's.
    """

    OBSERVATION = "observation"
    SPAN = "span"
    PICKED = "picked"


class ShapeVerb(StrEnum):
    """The nine calculation verbs `rejects` states an *in* rule for.

    `get_data` is not among them: it reads nothing upstream, so it has no
    *in* rule to state.
    """

    AGGREGATE = "aggregate"
    GROWTH = "growth"
    RANK = "rank"
    FILTER = "filter"
    SHARE_OF = "share_of"
    QUOTIENT_OF = "quotient_of"
    COMPARE = "compare"
    SUM_OF = "sum_of"
    JOIN = "join"


@dataclass(frozen=True, slots=True)
class RoleDimension:
    """One role a frame keeps one row per value of.

    `role` is the plan's role phrase (`"donor"`, `"recipient"`): a named
    entity's role, or the requirement's `across`. It is `None` when the
    requirement names an entity with no role. `requirement_id` is the dataset
    requirement this dimension's values come from -- always
    exactly one, by construction: every transform below either carries a
    `RoleDimension` through unchanged or selects one from a single side, so
    a role dimension never merges two requirements' entities into one.
    `narrowed` is whether a filter has already kept only named entities on
    this dimension. `breakdown` marks the dimension a requirement's
    breakdown adds, whose values are codes of a classification rather than
    entities, and which an aggregate grouping by breakdown keeps alone.
    `open` marks the dimension a requirement's `across` adds, which holds
    every member of a party nobody named. An aggregate sums it away whatever
    it groups by, and every other verb carries it as it does a named role.
    `place` marks the dimension holding the places the measure is about:
    entities named with no role, or an `across` of the place party
    (`place_role`). A place word names it (`roles_named`).
    """

    role: str | None
    requirement_id: Identifier
    narrowed: bool = False
    breakdown: bool = False
    open: bool = False
    place: bool = False


# The measure count of a family read separately whose source lists none of its
# StatVars, or one of a capped list: more than one, how many unstated.
SEVERAL: Final = "several"

type MeasureCount = int | Literal["several"] | None


@dataclass(frozen=True, slots=True)
class FrameShape:
    """What one step's output frame looks like, in plan vocabulary.

    `roles` is one entry per role the frame keeps one row per value of.
    `measures` is how many measures the frame carries, the count `rejects`
    reads for a verb's single-measure rule. A retrieval carries one, except
    one reading a family of StatVars separately, which carries one per
    member StatVar its source lists, `SEVERAL` where that list is empty or
    is capped at one, or `None` (uncounted) where no source is known to
    count them from: a separate reading of a source binding one StatVar
    carries one, so the single-measure rule refuses only a count it knows to
    be several; a `join` carries two. The executor bounds each
    step's declared StatVar count by the frames it reads
    (`compiler.frame_columns.bound_stat_var_count`). `value` and `time`
    state what kind of number and what kind of period label the frame's own
    columns carry. `period` is the years the
    frame can carry, `None` only when no dataset requirement bounds it.
    `one_row_per_entity` is whether the frame already holds one value per
    entity, so a rank over it needs no reduction. `cut_per_period` is
    whether a per-period rank `limit` removed rows from a frame spanning
    more than one year: an entity keeps only the periods it made the cut
    in, so its series misses years its source holds. A reader that follows
    one entity across periods refuses such a frame. `limited` is whether a
    rank `limit` upstream kept only some entities, in either mode and over
    any period: the frame holds neither the entities a filter named nor the
    source's population, so no rank over it can state a true `among`.
    `cut_per_period` implies `limited`. `roots` is the
    retrievals the frame ultimately rests on, sorted -- used for
    degenerate-side detection. `unsummable_axes` is, for a frame carrying a
    family read separately, the axes the family varies along that apis
    may not sum over (empty when it may sum over every one), or `None` where
    no source is known to say; a single-measure verb's refusal reads it to
    say whether summing the family would give it one measure.

    `mixed_levels` is whether the frame holds places at more than one level:
    a retrieval that names no place holds every place with data, the world,
    regions, countries and their subdivisions alike, until a place-level
    filter keeps one. `place_typed` is whether a place-level filter may read
    the frame: that retrieval, read through filters only, since every other
    verb drops the place-type column the compiler asks `get_data` for.

    `composed_unit` is whether the frame's values are in a unit two measures
    compose, dollars per person: a `quotient_of`'s result, and whatever a
    later step computes in that same unit, such as an average of it or its
    absolute change. A step whose result is a share (in per cent) or has no
    unit (a relative growth or a rank position) clears it. Such a value
    ranks, grows, averages and compares as any value in a unit does. It neither adds up,
    in a `sum_of`, under a `sum` reduction or on a `share_of` side summing
    several rows, nor divides again in a `quotient_of`, and a `compare` or
    a `share_of` pairs it only with another value in a composed unit.

    Time grain is not a field here: every frame this algebra describes is
    annual, and a derived frame inherits its input's grain unchanged.
    """

    roles: tuple[RoleDimension, ...]
    measures: MeasureCount
    value: ValueKind
    time: TimeLabel
    period: Period | None
    one_row_per_entity: bool
    cut_per_period: bool
    limited: bool
    roots: tuple[Identifier, ...]
    unsummable_axes: tuple[str, ...] | None = None
    mixed_levels: bool = False
    place_typed: bool = False
    composed_unit: bool = False


class NamedRole(Protocol):
    """What `roles_named` reads of a role dimension: a `RoleDimension`, or the compiler's wrapper of one."""

    @property
    def role(self) -> str | None: ...

    @property
    def place(self) -> bool: ...


# The phrase a refusal tells the planner to write for a place dimension whose
# entities state no role.
PLACE_PHRASE: Final = "country"


def place_role(role: str | None) -> bool:
    """Whether a dimension under `role` holds the places the measure is about.

    `role` is the role its entities state, `None` for entities stating none,
    or the requirement's `across`. A breakdown's dimension is never one.
    """
    return role is None or across_party(role) == "place"


def roles_named[R: NamedRole](roles: Sequence[R], phrase: str) -> tuple[R, ...]:
    """Every dimension of `roles` that `phrase` names.

    A phrase names the dimension whose role it equals, and a place word
    (`planner.parties.across_party`: "country", "countries", "place") names
    a side's place dimension, which states no role for a phrase to equal.
    One result is a selection. None, or several, is a refusal each caller
    words for its own side (`_names_no_role`).
    """
    place_word = across_party(phrase) == "place"
    return tuple(role for role in roles if role.role == phrase or (place_word and role.place))


def role_named[R: NamedRole](roles: Sequence[R], phrase: str) -> R | None:
    """The one dimension of `roles` that `phrase` names, or `None` when it names none or several."""
    named = roles_named(roles, phrase)
    return named[0] if len(named) == 1 else None


def axis_phrases(roles: Sequence[NamedRole]) -> list[str]:
    """The phrases that name each of `roles`, for a refusal to list: its role, or the place word."""
    return sorted({PLACE_PHRASE if role.role is None else role.role for role in roles})


def _names_no_role(
    field: str, phrase: str, side: str, roles: Sequence[RoleDimension]
) -> str | None:
    """Why `phrase`, stated as `field`, selects no one dimension of `side`'s `roles`, or `None`."""
    named = roles_named(roles, phrase)
    if len(named) == 1:
        return None
    if not named:
        return (
            f"{field} {phrase!r} does not name a role {side} carries; roles are "
            f"{axis_phrases(roles)!r}"
        )
    held = " and ".join(
        "the places named with no role" if role.role is None else repr(role.role) for role in named
    )
    return (
        f"{field} {phrase!r} names more than one dimension {side} carries, {held}; name "
        "that side's places one way, all with no role or all in `across`"
    )


def _select_role(roles: Sequence[RoleDimension], axis: str | None) -> RoleDimension | None:
    """Return the role dimension `axis` names, or the lone one when `axis` is `None` and `roles` has one.

    Returns `None` when `axis` names no one dimension of `roles`
    (`role_named`), or when `axis` is `None` and `roles` holds more than one
    dimension -- either way, the caller's own `rejects` check is what turns
    that into a refusal; this helper only resolves the selection once a
    caller already knows it is legal.
    """
    if axis is None:
        return roles[0] if len(roles) == 1 else None
    return role_named(roles, axis)


def _intersection(left: Period | None, right: Period | None) -> Period | None:
    """Return the years both `left` and `right` cover.

    `None` when either is unbounded, or when they share no year: `Period`
    cannot represent an empty range, and a disjoint pair is exactly what
    `compiler.compile`'s own `PERIODS_DISJOINT` reports from the
    requirements themselves, not from this shape -- this only has to avoid
    raising while that check runs.
    """
    if left is None or right is None:
        return None
    start, end = max(left.start, right.start), min(left.end, right.end)
    return Period(start=start, end=end) if end >= start else None


def windowed_years(period: Period | None, window: Period | None) -> Period | None:
    """Return the years of `period` that `window` keeps; either side unbounded leaves the other.

    `None` only when both are unbounded: `_rejects_rank_window` refuses a
    window that shares no year with a bounded period before any caller
    reads this.
    """
    if window is None:
        return period
    if period is None:
        return window
    return _intersection(period, window)


def _spans_years(period: Period | None) -> bool:
    """Whether `period` covers more than one year; an unbounded one may."""
    return period is None or period.end > period.start


def _merged_time(left: FrameShape, right: FrameShape) -> TimeLabel:
    """The period label a two-sided verb's output carries: a `SPAN`, else `PICKED`, else `OBSERVATION`.

    Both sides match on the period label, so `_rejects_paired_periods` lets a
    span on one side through only against the same span on the other, and a
    picked frame only against a series covering its years, whose rows it matches on
    each entity's picked year. The output keeps the narrower label.
    """
    times = (left.time, right.time)
    if TimeLabel.SPAN in times:
        return TimeLabel.SPAN
    if TimeLabel.PICKED in times:
        return TimeLabel.PICKED
    return TimeLabel.OBSERVATION


def _merged_roots(
    left: tuple[Identifier, ...], right: tuple[Identifier, ...]
) -> tuple[Identifier, ...]:
    """Return the sorted union of two sides' retrieval roots."""
    return tuple(sorted(set(left) | set(right)))


# --- The ten transforms ----------------------------------------------------


def retrieved(
    *,
    requirement_id: Identifier,
    roles: Sequence[RoleDimension],
    period: Period,
    measures: MeasureCount,
    unsummable_axes: tuple[str, ...] | None = None,
    place_typed: bool = False,
    mixed_levels: bool = False,
) -> FrameShape:
    """The shape a fresh `get_data` retrieval carries.

    `roles` names every role column the selected source resolves that
    `requirement_id`'s entities resolve under, the column its `across`
    covers, and the column its breakdown's codes sit in; each is re-stamped
    with this requirement's own id and `narrowed=False`, since nothing has
    filtered the retrieval yet. The caller's own `role` and `breakdown`
    values are kept, and this function states which requirement a role
    belongs to and whether it is narrowed.
    `measures` and `unsummable_axes` are the caller's (see `FrameShape`).
    `place_typed` lets a place-level filter read the retrieval, which
    names no place. `mixed_levels` says it holds places at every level. The
    plan states the first for a requirement whose `across` names the place
    (`planner.plan.names_no_place`), since it cannot know which axis the word
    covers. The compiler states both from the selected source: only a
    source that leaves `observationAbout` open over several places mixes
    levels (`compiler.evidence.opens_place_axis`).
    """
    stamped = tuple(replace(role, requirement_id=requirement_id, narrowed=False) for role in roles)
    return FrameShape(
        roles=stamped,
        measures=measures,
        value=ValueKind.MEASURED,
        time=TimeLabel.OBSERVATION,
        period=period,
        one_row_per_entity=False,
        cut_per_period=False,
        limited=False,
        roots=(requirement_id,),
        unsummable_axes=unsummable_axes,
        mixed_levels=mixed_levels,
        place_typed=place_typed,
    )


def aggregated(
    shape: FrameShape,
    *,
    reduction: Reduction,
    by_entity: bool,
    by_period: bool,
    by_breakdown: bool = False,
) -> FrameShape:
    """The shape `aggregate` produces, reducing `shape` by `reduction` per entity and/or period.

    Role dimensions survive only when the group keeps entity apart, except
    the breakdown's, which a group by breakdown keeps alone. A group by
    entity keeps the parties the requirement names and sums an open
    dimension away, as it sums the periods a group without period. The frame
    becomes one row per entity exactly when the group does not keep
    period apart. `measures`, `value` and `period` carry through unchanged --
    aggregate reduces rows, not what they measure. `cut_per_period` survives
    only a group by both entity and period, which keeps the gaps; a group by
    period alone totals each period's kept rows, and a group by entity alone
    is refused over a cut frame. Mixed levels survive only a group by
    entity that sums no open dimension. A group summing the open place
    dimension is refused over them. A group that keeps period apart keeps
    the input's own period labels, a `SPAN` included. Any other group
    labels each row with the period that supplied its value under a `max`
    or `min`, as analysis-mcp does: over a series
    of years that is one `PICKED` year per row (a single year when the
    period holds one), and over a `SPAN` it is the span itself. A `sum` or
    `mean` over any other group collapses the periods into a `SPAN`. Over a
    `PICKED` input analysis-mcp labels that span with the picked years the
    rows hold, which only the data decides, so its `period` is `None`.
    """
    period = shape.period
    if by_period:
        time = shape.time
    elif reduction in (Reduction.MAX, Reduction.MIN) and shape.time is not TimeLabel.SPAN:
        time = TimeLabel.PICKED if _spans_years(shape.period) else TimeLabel.OBSERVATION
    else:
        time = TimeLabel.SPAN
        if shape.time is TimeLabel.PICKED:
            period = None
    if by_entity:
        roles = tuple(role for role in shape.roles if not role.open)
    elif by_breakdown:
        roles = tuple(role for role in shape.roles if role.breakdown)
    else:
        roles = ()
    return FrameShape(
        roles=roles,
        measures=shape.measures,
        value=shape.value,
        time=time,
        period=period,
        one_row_per_entity=not by_period,
        cut_per_period=shape.cut_per_period and by_entity and by_period,
        limited=shape.limited,
        roots=shape.roots,
        unsummable_axes=shape.unsummable_axes,
        mixed_levels=shape.mixed_levels and by_entity,
        composed_unit=shape.composed_unit,
    )


def grown(shape: FrameShape, *, change: GrowthChange, window: Period) -> FrameShape:
    """The shape `growth` produces over `window`, read as `change` says.

    Every row is labelled with `window` itself, the interval
    `window_start/window_end`, as `run_growth` labels its output, so the
    frame is a `SPAN` over `window`. A relative change reads as a
    dimensionless `RATIO`; an absolute change reads as a `DIFFERENCE` in the input's own unit, or in percentage
    points when the input was a `RATIO`. An absolute change over an
    `ORDINAL` input is a count of places and stays `ORDINAL`.
    An absolute change stays in its input's unit, a composed one included;
    a relative change has none.
    """
    if change is GrowthChange.RELATIVE:
        value = ValueKind.RATIO
    elif shape.value is ValueKind.ORDINAL:
        value = ValueKind.ORDINAL
    else:
        value = ValueKind.DIFFERENCE
    return FrameShape(
        roles=shape.roles,
        measures=shape.measures,
        value=value,
        time=TimeLabel.SPAN,
        period=window,
        one_row_per_entity=True,
        cut_per_period=False,
        limited=shape.limited,
        roots=shape.roots,
        unsummable_axes=shape.unsummable_axes,
        mixed_levels=shape.mixed_levels,
        composed_unit=shape.composed_unit and change is GrowthChange.ABSOLUTE,
    )


def ranked(
    shape: FrameShape,
    *,
    axis: str | None,
    window: Period | None,
    per_period: bool,
    limit: int | None,
) -> FrameShape:
    """The shape `rank` produces, in its collapsing or per-period mode.

    `axis` selects which of `shape`'s role dimensions the output keeps;
    `None` with a single input role picks that lone role. Collapse mode
    (`per_period=False`) carries the entity's own measured `value`, labelled
    a `SPAN`. Per-period mode keeps one row per entity per period of its
    input, with the input's own period labels, and carries the axis role's
    own `narrowed` flag forward unchanged, since that flag is what the
    population/axis rules and the per-period growth rule both read
    afterwards. Its `value` is the rank position itself, an
    `ORDINAL`, unless `limit` is set: a limited ranking keeps each period's
    top entities with their measured values, as `run_rank` writes them, and
    marks the frame `cut_per_period` when it spans more than one real
    period. A `SPAN` input holds one period, so has no gap to leave. A
    per-period rank over a frame already cut keeps its gaps. Either mode
    marks the frame `limited` when `limit` is set.

    The output's `period` is the years of the input's period the window
    keeps: `run_rank` labels a collapsed row with the extent of the periods
    it ranked, so a 2010-2025 window over 2018-2022 labels each row
    `2018/2022`. Over a `SPAN`, `_rejects_rank` admits only a window that
    contains it, so the period stays the span's own.
    """
    axis_role = _select_role(shape.roles, axis)
    roles = (axis_role,) if axis_role is not None else ()
    period = windowed_years(shape.period, window)
    limited = limit is not None or shape.limited
    if per_period:
        return FrameShape(
            roles=roles,
            measures=1,
            value=ValueKind.ORDINAL if limit is None else shape.value,
            time=shape.time,
            period=period,
            one_row_per_entity=False,
            cut_per_period=(limit is not None or shape.cut_per_period)
            and shape.time is TimeLabel.OBSERVATION
            and _spans_years(period),
            limited=limited,
            roots=shape.roots,
            composed_unit=shape.composed_unit and limit is not None,
        )
    return FrameShape(
        roles=roles,
        measures=shape.measures,
        value=shape.value,
        time=TimeLabel.SPAN,
        period=period,
        one_row_per_entity=True,
        cut_per_period=False,
        limited=limited,
        roots=shape.roots,
        unsummable_axes=shape.unsummable_axes,
        composed_unit=shape.composed_unit,
    )


def _year_bounds(predicates: Sequence[Predicate]) -> tuple[int | None, int | None]:
    """The first and last year the `PeriodThreshold` bounds in `predicates` leave standing.

    `gte` and `lte` close on their own year, `gt` on the year after and `lt`
    on the year before. Either end is `None` when no threshold bounds it.
    """
    thresholds = [p for p in predicates if isinstance(p, PeriodThreshold)]
    lower = max(
        (p.year + 1 if p.op == "gt" else p.year for p in thresholds if p.op in ("gt", "gte")),
        default=None,
    )
    upper = min(
        (p.year - 1 if p.op == "lt" else p.year for p in thresholds if p.op in ("lt", "lte")),
        default=None,
    )
    return lower, upper


@dataclass(frozen=True, slots=True)
class YearFilter:
    """What a filter's `PeriodThreshold` predicates leave of a frame's period.

    `period` is the frame's period narrowed to the first and the last year
    left, `None` for an unbounded one. `leaves_one_year` is whether the
    predicates leave exactly one year standing.
    """

    period: Period | None
    leaves_one_year: bool


def year_filter(period: Period | None, predicates: Sequence[Predicate]) -> YearFilter | str:
    """Read the `PeriodThreshold` predicates in `predicates` against `period`.

    `eq` leaves one year, `_year_bounds` closes the others on a bound, and
    `ne` removes its year, which moves an end only when that year is the
    first or the last left. `year != 2020` over 2020-2021 is therefore 2021
    alone, and `year != 2021` over 2020-2022 is still 2020-2022. A filter
    with no year predicate leaves `period` as it is and leaves no one year,
    whatever `period` spans. An unbounded `period` stays unbounded, and one
    year stands over it only where an `eq` or a pair of meeting bounds names
    it. Predicates that leave no year of `period` return the refusal reason:
    bounds that cross, an `eq` year outside `period` or its bounds, two
    different `eq` years, or `ne` years that remove every year left.
    """
    thresholds = [p for p in predicates if isinstance(p, PeriodThreshold)]
    lower, upper = _year_bounds(thresholds)
    if period is None:
        named = any(p.op == "eq" for p in thresholds) or (lower is not None and lower == upper)
        return YearFilter(period=None, leaves_one_year=named)
    start = period.start if lower is None else max(lower, period.start)
    end = period.end if upper is None else min(upper, period.end)
    years = set(range(start, end + 1))
    for threshold in thresholds:
        if threshold.op == "eq":
            years &= {threshold.year}
        elif threshold.op == "ne":
            years.discard(threshold.year)
    if not years:
        return f"the year filter leaves no year of {period.start}-{period.end}"
    return YearFilter(
        period=Period(start=min(years), end=max(years)),
        leaves_one_year=bool(thresholds) and len(years) == 1,
    )


def filtered(
    shape: FrameShape,
    *,
    narrowed_roles: Sequence[str | None],
    period: Period | None,
    pins_one_year: bool,
    keeps_one_level: bool = False,
) -> FrameShape:
    """The shape `filter` produces: everything `shape` carries, plus what got narrowed.

    `narrowed_roles` names the role phrases an `EntityIn` predicate matched.
    An unqualified predicate -- no `role` stated -- carries `None`, and
    since the primitives here cannot tell which single dimension it
    resolved against, it narrows every role dimension the input carries;
    `EntityIn.role` is what disambiguates when a shape carries more than
    one, and stating it narrows the dimension it names (`roles_named`)
    alone, a place word naming the place dimension. Each matching
    dimension is marked `narrowed=True` and every other field is kept.
    `period` and `pins_one_year` are `year_filter`'s reading of the filter's
    predicates against `shape.period`: its `period` and `leaves_one_year`. A filter keeps `cut_per_period` only while `period`
    still spans more than one year, since a single period has no gap
    between periods, and always keeps `limited`. A `PICKED` frame narrowed
    to one year is a plain `OBSERVATION`: every row left is in that year. A filter that
    `keeps_one_level` (a `PlaceLevel` predicate) leaves its places at one
    level; every filter keeps the place-type column its input carries.
    """
    narrows_every_role = None in narrowed_roles
    named = [
        role
        for phrase in narrowed_roles
        if phrase is not None
        for role in roles_named(shape.roles, phrase)
    ]
    roles = tuple(
        replace(role, narrowed=True) if narrows_every_role or role in named else role
        for role in shape.roles
    )
    picked_one_year = shape.time is TimeLabel.PICKED and not _spans_years(period)
    return FrameShape(
        roles=roles,
        measures=shape.measures,
        value=shape.value,
        time=TimeLabel.OBSERVATION if picked_one_year else shape.time,
        period=period,
        one_row_per_entity=shape.one_row_per_entity or pins_one_year,
        cut_per_period=shape.cut_per_period and not pins_one_year and _spans_years(period),
        limited=shape.limited,
        roots=shape.roots,
        unsummable_axes=shape.unsummable_axes,
        mixed_levels=shape.mixed_levels and not keeps_one_level,
        place_typed=shape.place_typed,
        composed_unit=shape.composed_unit,
    )


def shared(
    numerator: FrameShape,
    denominator: FrameShape,
    *,
    axis: str | None,
    scope: DenominatorScope | None,
) -> FrameShape:
    """The shape `share_of` produces, a `RATIO` in per cent with the numerator's own axis role alone.

    Only the numerator's side decides the output's role dimension -- the one
    narrow-lineage rule every non-terminal verb here follows, so which
    requirement stands behind the output column is always answerable. The
    result keeps every numerator entity: a pooled denominator divides each
    by one total per period, and analysis-mcp's `run_share_of` refuses a
    per-axis call outright at the first numerator key the denominator has no
    group for. So the output is `limited` exactly when the numerator is.
    `run_share_of` writes one row per numerator key per period it shares
    with its denominator, and refuses a key sharing none, so the output
    holds one row per entity whenever the numerator does: a share of each
    entity's total over 2018-2022 is one ratio per entity, which a rank
    reads with no reduction. It can also hold one row per entity when only
    the denominator does, such as a yearly series over a denominator
    filtered to one year; `one_row_per_entity` reads the numerator alone,
    and `period`, the intersection of both sides' periods, then names the
    one year. `scope` takes no part in this shape; it is accepted for
    symmetry with `rejects`, which does read it.
    """
    axis_role = _select_role(numerator.roles, axis) if axis is not None else None
    roles = (axis_role,) if axis_role is not None else ()
    return FrameShape(
        roles=roles,
        measures=1,
        value=ValueKind.RATIO,
        time=_merged_time(numerator, denominator),
        period=_intersection(numerator.period, denominator.period),
        one_row_per_entity=numerator.one_row_per_entity,
        cut_per_period=False,
        limited=numerator.limited,
        roots=_merged_roots(numerator.roots, denominator.roots),
    )


def quotiented(
    numerator: FrameShape,
    denominator: FrameShape,
    *,
    axis: str | None,
    denominator_axis: str | None,
) -> FrameShape:
    """The shape `quotient_of` produces: the numerator's own axis role alone, in a composed unit.

    Only the numerator's side decides the output's role dimension, under the
    numerator's own phrase, as for `shared`; the denominator's phrase names
    no column of the result. analysis-mcp's `run_quotient_of` writes one row
    per numerator key per period both sides report, so the output holds one
    row per entity when either side does, and its period is the years both
    cover. It drops a key the denominator has no group for and says so in
    its result, which does not make the output `limited`: only a rank
    `limit` upstream does, on the numerator, or on a denominator keyed by
    `denominator_axis`, whose missing entities take their numerator rows
    with them. A side a per-period limit cut leaves the same gaps in each
    quotient's series.
    """
    axis_role = _select_role(numerator.roles, axis) if axis is not None else None
    roles = (axis_role,) if axis_role is not None else ()
    period = _intersection(numerator.period, denominator.period)
    return FrameShape(
        roles=roles,
        measures=1,
        value=ValueKind.MEASURED,
        time=_merged_time(numerator, denominator),
        period=period,
        one_row_per_entity=numerator.one_row_per_entity or denominator.one_row_per_entity,
        cut_per_period=(numerator.cut_per_period or denominator.cut_per_period)
        and _spans_years(period),
        limited=numerator.limited or (denominator_axis is not None and denominator.limited),
        roots=_merged_roots(numerator.roots, denominator.roots),
        composed_unit=True,
    )


def compared(left: FrameShape, right: FrameShape) -> FrameShape:
    """The shape `compare` produces. Terminal by contract; stated for completeness.

    The difference is in the two sides' one unit, composed when theirs is.
    """
    return FrameShape(
        roles=(),
        measures=1,
        value=ValueKind.DIFFERENCE,
        time=_merged_time(left, right),
        period=_intersection(left.period, right.period),
        one_row_per_entity=True,
        cut_per_period=False,
        limited=left.limited or right.limited,
        roots=_merged_roots(left.roots, right.roots),
        composed_unit=left.composed_unit and right.composed_unit,
    )


def summed_sides(left: FrameShape, right: FrameShape) -> FrameShape:
    """The shape `sum_of` produces: one series, no role dimension, over the years both sides cover.

    analysis-mcp's `run_sum_of` writes one row per period both sides report,
    under its own StatVar, with no role column, so no role dimension carries
    through to name a requirement behind it. `rejects` admits only two sides
    holding the same kind of value, which the sum keeps.
    """
    return FrameShape(
        roles=(),
        measures=1,
        value=left.value,
        time=_merged_time(left, right),
        period=_intersection(left.period, right.period),
        one_row_per_entity=True,
        cut_per_period=False,
        limited=left.limited or right.limited,
        roots=_merged_roots(left.roots, right.roots),
    )


def joined(left: FrameShape, right: FrameShape) -> FrameShape:
    """The shape `join` produces: the two sides' shared role dimensions, two measures.

    Terminal by contract, so nothing downstream reads this shape and the
    one-side lineage rule does not apply to it -- a joined frame's role
    dimensions may, unlike every non-terminal verb's, name more than one
    requirement. `left`'s own `RoleDimension` is kept for each shared role,
    sorted by role name for a deterministic result.
    """
    left_by_role = {role.role: role for role in left.roles if role.role is not None}
    right_role_names = {role.role for role in right.roles if role.role is not None}
    shared_roles = tuple(
        sorted(
            (role for name, role in left_by_role.items() if name in right_role_names),
            key=lambda role: role.role or "",
        )
    )
    return FrameShape(
        roles=shared_roles,
        measures=2,
        value=ValueKind.MEASURED,
        time=_merged_time(left, right),
        period=_intersection(left.period, right.period),
        one_row_per_entity=False,
        cut_per_period=False,
        limited=left.limited or right.limited,
        roots=_merged_roots(left.roots, right.roots),
    )


# --- rejects: the single place a verb's *in* rule is written ---------------

# Why a reader that follows one entity across periods refuses a
# `cut_per_period` frame; each rule appends the order that works.
_CUT_PER_PERIOD = (
    "a per-period rank with a limit keeps an entity only in the periods it made the cut, "
    "so its series misses years its source holds"
)


# What a `PICKED` input holds, for the reason each reader that refuses it gives.
_PICKED = "this input holds each entity's max or min at the year it fell in, one year per entity"


def _mixed_levels(step: str, shape: FrameShape) -> str | None:
    """Why `step`, which sets places side by side, refuses `shape`, or `None` when it does not."""
    if not shape.mixed_levels:
        return None
    return (
        "the input names no place, so it holds every place with data at every level, the "
        f"world, regions, World Bank groups, countries and their subdivisions, and {step} "
        "would set a place beside the places inside it; add a filter with a place_level "
        "predicate, such as level 'Country', reading that requirement before this step"
    )


def _several_measures(verb: str, shape: FrameShape) -> str | None:
    """Why `verb`, which reads one measure, refuses `shape`, or `None` when it does not.

    Only a retrieval reading a family of StatVars separately carries more
    than one measure into a verb (a `join`'s two feed nothing), and only a
    count its source states is refused: an uncounted shape (`None`) may rest
    on a source binding one StatVar, and the compiler counts it once the
    source is known. The reason states the count, or "several" for a family
    whose source lists none of its StatVars or one of a capped list (`SEVERAL`). It advises reading the family summed where
    apis may sum it over every axis it varies along
    (`FrameShape.unsummable_axes`), names the axes it may not sum over
    otherwise, and puts the advice as a condition where neither is known.
    """
    if shape.measures is None or shape.measures == 1:
        return None
    reason = (
        f"{verb} reads a single measure; got {shape.measures}, one per StatVar of a family "
        "read separately"
    )
    axes = shape.unsummable_axes
    if axes is None:
        return (
            f"{reason}; read the family summed (stat_vars='sum') to give it one if it can be "
            "summed over every axis it varies along, otherwise name one value for each axis "
            "it cannot"
        )
    if not axes:
        return f"{reason}; read the family summed (stat_vars='sum') to give it one"
    one_value = "one value for it" if len(axes) == 1 else "one value for each"
    return (
        f"{reason}; it cannot be summed over {', '.join(axes)}, so the question has to name "
        f"{one_value}"
    )


def _missing_group_roles(shape: FrameShape, *, by_entity: bool, by_breakdown: bool) -> str | None:
    """Why `shape` has no role an aggregate's group by entity or breakdown keeps, or `None`.

    A frame with no role dimension that rests on two retrievals, a
    `sum_of`'s or an unkeyed `share_of`'s or `quotient_of`'s, holds one series, so a group by
    entity reduces that series whole: a total over a span of two sources'
    yearly sum. A group by entity keeps only the parties a requirement
    names, so over a frame whose every party is open it would keep none.
    """
    if by_entity and not shape.roles and len(shape.roots) < 2:
        return (
            "aggregate cannot group by entity over a frame with no role dimension; group by "
            "entity in the step that collapsed the roles"
        )
    open_roles = [role.role for role in shape.roles if role.open]
    if by_entity and open_roles and all(role.open or role.breakdown for role in shape.roles):
        return (
            "aggregate grouping by entity keeps the parties the requirement names apart and "
            f"sums its open {open_roles!r} away, and this input names no entity, so the group "
            "keeps nothing; group by period alone to total every member per year, or rank "
            "with a reduction to set each member's total beside the others"
        )
    if by_breakdown and not any(role.breakdown for role in shape.roles):
        return (
            "aggregate cannot group by breakdown over a frame that carries none; state "
            "breakdown on the dataset requirement it reads"
        )
    return None


def _rejects_summed_rate(step: str, shape: FrameShape, reduction: Reduction | None) -> str | None:
    """Why `step` may not reduce `shape` by `reduction`, or `None` when it may.

    A sum of values in a composed unit is no value in it: two countries'
    dollars per person, or two years' of one country, add up to no
    per-person figure, and analysis-mcp would write the total under the
    composed unit all the same. A `mean`, a `max` and a `min` each stay a
    figure in that unit. One figure pooled across entities divides the two
    measures' totals across them; nothing pools across periods, since a
    population summed over years is no population.
    """
    if reduction is not Reduction.SUM or not shape.composed_unit:
        return None
    return (
        f"{step} with reduction 'sum' adds values in one unit, and this input's are per-unit "
        "figures from a quotient_of, which add up to no figure in that unit; take a mean, "
        "max or min, or for one figure pooled across entities total each measure across "
        "them first and divide the two totals"
    )


def _rejects_aggregate(
    shape: FrameShape,
    *,
    by_entity: bool,
    by_period: bool = False,
    by_breakdown: bool = False,
    reduction: Reduction | None = None,
) -> str | None:
    several = _several_measures("aggregate", shape) or _rejects_summed_rate(
        "an aggregate", shape, reduction
    )
    if several is not None:
        return several
    if missing := _missing_group_roles(shape, by_entity=by_entity, by_breakdown=by_breakdown):
        return missing
    sums_open = not by_entity or any(role.open for role in shape.roles)
    if sums_open and (mixed := _mixed_levels("a total or an average across places", shape)):
        return mixed
    if by_period and shape.time is TimeLabel.PICKED:
        return (
            f"an aggregate by period reduces each year's rows, and {_PICKED}, so each year "
            "would hold only the entities whose value fell in it; group by entity alone, or "
            "aggregate the source by period before the max or min"
        )
    if shape.cut_per_period and by_entity and not by_period:
        return (
            f"an aggregate by entity totals each entity's series, and {_CUT_PER_PERIOD}; "
            "rank the source once, with the reduction, the limit and per_period=false, which "
            "reduces each entity's whole series before the limit applies"
        )
    return None


def _rejects_growth_time(shape: FrameShape) -> str | None:
    """Return why `shape`'s periods are no series for growth to read, or `None` when they are."""
    if shape.time is TimeLabel.PICKED:
        return (
            f"growth reads each entity's series across years, and {_PICKED}; compute the "
            "growth over the source's series, then take the max or min of its result"
        )
    if shape.time is not TimeLabel.OBSERVATION:
        return (
            "growth reads a real per-period series, not a span already collapsed by "
            "aggregate, rank or growth"
        )
    return None


def _rejects_growth(shape: FrameShape, *, change: GrowthChange) -> str | None:
    reason = _several_measures("growth", shape) or _rejects_growth_time(shape)
    if reason is not None:
        return reason
    if shape.cut_per_period:
        return (
            f"growth reads each entity's endpoint years, and {_CUT_PER_PERIOD}, so its "
            "endpoints say whether it made the cut, not how it grew; compute the growth "
            "first, then rank its result with the limit"
        )
    unnarrowed = (
        [role.role for role in shape.roles if not role.narrowed]
        if shape.value is ValueKind.ORDINAL
        else []
    )
    if change is GrowthChange.RELATIVE and shape.value in (ValueKind.RATIO, ValueKind.ORDINAL):
        narrowing = (
            f" and add a filter narrowing {unnarrowed!r} before this growth" if unnarrowed else ""
        )
        return (
            f"a relative growth misreads a {shape.value.value} input; state change='absolute'"
            f"{narrowing}"
        )
    if unnarrowed:
        return (
            "an absolute growth over a rank position has to be about one entity, because "
            "each year's rank position depends on which entities reported that year, so a "
            "change in position across a population mixes movement with reporting; add a "
            f"filter narrowing {unnarrowed!r} before this growth"
        )
    return None


def _rejects_rank_axis(shape: FrameShape, axis: str | None) -> str | None:
    """Return why `axis` cannot select a role dimension of `shape`, or `None` when it can."""
    if not shape.roles:
        return "rank needs at least one role dimension to rank entities on"
    if axis is None:
        if len(shape.roles) > 1:
            return (
                "rank over more than one role needs axis naming which one; roles are "
                f"{axis_phrases(shape.roles)!r}"
            )
        return None
    return _names_no_role("axis", axis, "this input", shape.roles)


def _rejects_rank_rows(shape: FrameShape) -> str | None:
    """Return why the rows `shape` holds give a rank nothing true to order, or `None` when they do."""
    if shape.value is ValueKind.ORDINAL:
        return (
            "rank orders entities by a measured value, and this input holds a rank position "
            "or a change in one, which a per-period rank with no limit writes in place of the "
            "measure; rank the source once with the limit the question needs"
        )
    if shape.limited:
        return (
            "a rank with a limit keeps only its top or bottom entities, so a rank over its "
            "result ranks neither the entities named nor every entity the source holds; rank "
            "the source once with the limit the question needs"
        )
    return None


def _rejects_rank_picked(shape: FrameShape) -> str | None:
    """Return why a rank refuses a `PICKED` input, in either mode, or `None` for any other.

    analysis-mcp's `run_rank` measures coverage against every year in the
    window and refuses `coverage_insufficient` below half. A picked frame
    holds one row per entity spread over its picked years, so a collapsing
    rank fails once those fall in three or more years, and a per-period rank
    ranks each year's few entities. Ranking the source with the same
    reduction orders the entities by the same values.
    """
    if shape.time is not TimeLabel.PICKED:
        return None
    return (
        f"a rank counts every year of its window for each entity, and {_PICKED}, so most "
        "years have no row; rank the source with reduction 'max' or 'min', which orders "
        "each entity by the same value"
    )


def _covers(outer: Period | None, inner: Period | None) -> bool:
    """Whether `outer` holds every year of `inner`; an unknown period on either side holds none known."""
    return outer is not None and inner is not None and outer.covers(inner)


def _rejects_rank_window(shape: FrameShape, window: Period | None) -> str | None:
    """Return why `window` leaves a rank no row of `shape`, or `None` when it keeps some.

    A window sharing no year with the input's period keeps no row, and
    `run_rank` refuses `RankNoRowsInWindowError` in either mode.
    analysis-mcp keeps a span row only when the window holds both the
    span's start and end years, so a window that cuts the span leaves no
    row and the rank refuses. A span whose years are unknown here is known
    to fit no window, so any window over it is refused.
    """
    if window is None:
        return None
    if shape.period is not None and _intersection(shape.period, window) is None:
        years = _years_text(shape.period)
        return (
            f"this input holds {years}, which a {window.start}-{window.end} rank window "
            f"misses, so no row is left to rank; set the window inside {years}, or drop it"
        )
    if shape.time is not TimeLabel.SPAN:
        return None
    span = shape.period
    if _covers(window, span):
        return None
    if span is None:
        return (
            "a rank window keeps a row only when it covers every year the row's period does, "
            "and this input's rows carry one span whose years only the data decides; drop "
            "the window"
        )
    return (
        f"a rank window keeps a row only when it covers every year the row's period does, and "
        f"this input's rows carry one span over {span.start}-{span.end}, which a "
        f"{window.start}-{window.end} window cuts; widen the window to cover "
        f"{span.start}-{span.end}, or drop it"
    )


def _rejects_rank(
    shape: FrameShape,
    *,
    axis: str | None,
    among: RankAmong | None = None,
    window: Period | None = None,
    per_period: bool = False,
    reduction: Reduction | None = None,
) -> str | None:
    reason = (
        _several_measures("rank", shape)
        or _rejects_summed_rate("a rank", shape, reduction)
        or _mixed_levels("a rank", shape)
        or _rejects_rank_rows(shape)
        or _rejects_rank_axis(shape, axis)
        or _rejects_rank_picked(shape)
        or _rejects_rank_window(shape, window)
    )
    if reason is not None:
        return reason
    if among is RankAmong.POPULATION:
        axis_role = _select_role(shape.roles, axis)
        if axis_role is not None and axis_role.narrowed:
            return (
                f"a population rank over axis {axis_role.role!r} ranks only the named entities "
                "once a filter has already narrowed that axis; state among='named' to match "
                "what the filter did, or rank before narrowing"
            )
    return None


def _rejects_filter(
    shape: FrameShape, *, place_level: bool = False, period_threshold: bool = False
) -> str | None:
    """Return why a filter refuses `shape`, or `None` when it may read it.

    A year threshold refuses only rows labelled with an interval. A
    `PICKED` frame's rows each carry a real year, so a threshold keeps the
    entities whose value fell in the years it leaves.
    """
    if period_threshold and _labels_a_span(shape):
        return (
            "a year filter compares each row's year, and this input's rows carry one span of "
            "years, such as 2018/2022, that no year orders against; filter the years before "
            "the step that collapsed them, or drop the year predicate"
        )
    if place_level and not shape.place_typed:
        return (
            "a place_level filter reads a requirement that names no place, directly or through "
            "other filters: a named place has no level to choose, and every other step drops "
            "the type each place carries"
        )
    return None


def _rejects_paired_periods(verb: str, shape: FrameShape, other: FrameShape | None) -> str | None:
    """Return why `verb`'s two sides share no period label, or `None` when they may.

    `share_of`, `quotient_of`, `compare` and `join` match rows on the `time_period` string,
    so a span such as `2018/2022` matches only the same span. Against single
    years, or against a span over other years, analysis-mcp refuses
    (`share_of_grain_mismatch`, `comparison_periods_disjoint`) or, for a
    join, matches no row. A span whose years are unknown here (`period` is
    `None`) is known to match no span. A `PICKED` side is read by
    `_rejects_paired_picked`.
    """
    if other is None:
        return None
    if TimeLabel.PICKED in (shape.time, other.time):
        return _rejects_paired_picked(verb, shape, other)
    if _labels_a_span(shape) is not _labels_a_span(other):
        remedy = _paired_spans_remedy(verb, "collapse both sides over the same years, or neither")
        return (
            f"{verb} matches the two sides on their periods, and one side's rows carry one "
            "span of years, such as 2018/2022, while the other's carry single years, so no "
            f"period matches; {remedy}"
        )
    return _rejects_paired_spans(verb, shape, other) if _labels_a_span(shape) else None


def _paired_spans_remedy(verb: str, remedy: str) -> str:
    """What to do about two sides whose spans do not match: `remedy`, or for `sum_of` the sum first.

    `sum_of` adds two yearly series over the years both report and names
    each year one side lacks, so a total of two sources over a span sums
    the series first, over the period the question asked about, and then
    totals the sum.
    """
    if verb != "sum_of":
        return remedy
    return (
        "add the two yearly series with the sum_of first, both over the period the question "
        "asks about, then total the sum with an aggregate grouped by entity; the sum_of "
        "names any year one side lacks"
    )


def _rejects_paired_spans(verb: str, shape: FrameShape, other: FrameShape) -> str | None:
    """Return why two sides labelled with intervals carry different spans, or `None` when equal."""
    if shape.period is None or other.period is None:
        return (
            f"{verb} matches the two sides on their periods, and a side's rows carry a span "
            "whose years only the data decides, such as a total of each entity's max or min, "
            "so no other span is known to match it; "
            f"{_paired_spans_remedy(verb, 'collapse both sides over the same stated years')}"
        )
    if shape.period != other.period:
        return (
            f"{verb} matches the two sides on their periods, and the two sides carry spans "
            f"over different years ({shape.period.start}-{shape.period.end} against "
            f"{other.period.start}-{other.period.end}), so no period matches; "
            f"{_paired_spans_remedy(verb, 'collapse both sides over the same years')}"
        )
    return None


def _rejects_paired_picked(verb: str, shape: FrameShape, other: FrameShape) -> str | None:
    """Return why a `PICKED` side cannot pair with `other`, or `None` when `other` covers it.

    The picked side matches the other side's rows on each entity's own
    year, which may be any year of its period. Another picked side meets it
    only where both values fell in the same year, and a span labelled with
    an interval matches no single year, so both are refused. Any other side,
    a series or a span over one year (labelled with that year's point), has
    to cover every year of the picked period; an unknown period on either
    side covers none known.
    """
    if shape.time is other.time:
        return (
            f"{verb} matches the two sides on their years, and both sides hold each "
            "entity's max or min at the year it fell in, so they meet only where both "
            "fell in the same year; read one side as a yearly series"
        )
    picked, series = (shape, other) if shape.time is TimeLabel.PICKED else (other, shape)
    if _labels_a_span(series):
        return (
            f"{verb} matches the two sides on their periods, and one side holds each "
            "entity's max or min at the year it fell in while the other collapsed its "
            "years into one span, so no row's year matches it; read the other side as a "
            "yearly series"
        )
    if _covers(series.period, picked.period):
        return None
    if picked.period is None:
        return (
            f"{verb} matches the two sides on their years, and one side holds each entity's "
            "max or min at the year it fell in, over years no requirement bounds; bound that "
            "side's requirement to the years the other side covers"
        )
    years = _years_text(picked.period)
    held = "unbounded years" if series.period is None else _years_text(series.period)
    return (
        f"{verb} matches the two sides on their years, and one side holds each entity's max "
        f"or min at the year it fell in, any year of {years}, while the other side holds "
        f"{held}, so an entity whose value fell outside it has no year to match; read the "
        f"other side over {years}"
    )


def _years_text(period: Period) -> str:
    return str(period.start) if period.start == period.end else f"{period.start}-{period.end}"


def _labels_a_span(shape: FrameShape) -> bool:
    """Whether `shape`'s rows carry an interval label. A `SPAN` over one year
    is labelled with that year's point, which matches single years."""
    return shape.time is TimeLabel.SPAN and _spans_years(shape.period)


def _rejects_one_sided_rate(verb: str, shape: FrameShape, other: FrameShape | None) -> str | None:
    """Why `verb` may not pair a side in a composed unit with one that is not, or `None` when it may.

    analysis-mcp pairs two sides only when their units are equal
    (`cross_frame_unit_mismatch`), and a composed unit equals no source's
    own unit, no share's per cent and no unit-less side.
    """
    if other is None or shape.composed_unit is other.composed_unit:
        return None
    return (
        f"{verb} pairs two sides in one unit, and one side's values come from a quotient_of, "
        "whose composed unit pairs only with another quotient in the same unit; read both "
        "sides from the quotient, or neither"
    )


def _rejects_pooled_rate(
    shape: FrameShape, *, axis: str | None, scope: DenominatorScope | None, other: FrameShape
) -> str | None:
    """Why a share_of adds per-unit figures on one of its sides, or `None` when it adds none.

    `shape` is the numerator, keyed by `axis`, and `other` the denominator,
    keyed by `axis` too when `scope` divides per axis. analysis-mcp sums
    each side's rows per key and period and refuses a side in a composed
    unit whose sum reads more than one row
    (`composed_unit_sum_unsupported`). A side reads one row where it is
    keyed by its role dimension or carries none. Left un-keyed, the members
    of an open dimension or the codes of a breakdown are summed, and no
    `entity` narrows either. A dimension of named entities left un-keyed is
    the compiler's to count (`entity_required`, `share_axis_required`).
    """
    sides = (
        ("numerator", shape, axis),
        ("denominator", other, axis if scope is DenominatorScope.PER_AXIS else None),
    )
    for side, side_shape, keyed in sides:
        if not side_shape.composed_unit:
            continue
        key = role_named(side_shape.roles, keyed) if keyed is not None else None
        summed = [
            role for role in side_shape.roles if role is not key and (role.open or role.breakdown)
        ]
        if summed:
            return (
                f"share_of sums each side per period, and its {side} holds one per-unit "
                f"figure from a quotient_of per {axis_phrases(summed)!r}, which add up to no "
                "figure in that unit; key the share on that role with a per-axis denominator, "
                "take their mean, max or min with an aggregate first, or for one figure pooled "
                "across them total each measure first and divide the two totals"
            )
    return None


def _rejects_share_of_axis(
    shape: FrameShape, *, axis: str | None, scope: DenominatorScope | None, other: FrameShape | None
) -> str | None:
    """Return why `axis` cannot key a share_of reading `shape` (paired with `other`), or `None`."""
    if axis is None:
        return None
    reason = _names_no_role("axis", axis, "this side", shape.roles)
    if reason is None and scope is DenominatorScope.PER_AXIS and other is not None:
        reason = _names_no_role("axis", axis, "the other side", other.roles)
        if reason is not None:
            return f"{reason}, so a per-axis share has no counterpart to divide by"
    return reason


def _rejects_picked_share(shape: FrameShape, other: FrameShape | None) -> str | None:
    """Return why `share_of` refuses a `PICKED` side, either one, or `None` when neither is.

    analysis-mcp's `run_share_of` divides each key over the periods both
    sides report, so a picked side would run, and the result would answer
    another question. A side summed across entities (a pooled denominator,
    an un-keyed numerator) adds each entity's value at the year its own max
    or min fell in, so a year's total holds only the entities whose extreme
    fell there. A keyed share gives each entity's share in the year its
    measure peaked, which is not its largest share. The max or min of the
    shares over every year answers both readings, so every picked side is
    refused with that order.
    """
    if TimeLabel.PICKED not in (shape.time, other.time if other is not None else None):
        return None
    return (
        "share_of divides each entity by the other side in the same year, and one side holds "
        "each entity's max or min at the year it fell in, one year per entity, so the share "
        "would be the one in that year, not the largest or smallest share; compute the "
        "share_of over every year, then take the max or min of the shares"
    )


def _rejects_share_of(
    shape: FrameShape,
    *,
    axis: str | None,
    scope: DenominatorScope | None = None,
    other: FrameShape | None = None,
    entity: str | None = None,
) -> str | None:
    reason = (
        _several_measures("share_of", shape)
        or _mixed_levels("a share", shape)
        or _rejects_picked_share(shape, other)
        or _rejects_paired_periods("share_of", shape, other)
        or _rejects_one_sided_rate("share_of", shape, other)
    )
    if reason is not None:
        return reason
    if shape.value is ValueKind.ORDINAL:
        return (
            "share_of reads a measured value or a ratio, not a rank position; "
            "a rank position has no share"
        )
    per_entity_cut = shape.cut_per_period and (axis is not None or entity is not None)
    per_axis_cut = scope is DenominatorScope.PER_AXIS and other is not None and other.cut_per_period
    # A pooled share divides every entity in a period by one total, so ranking
    # the shares keeps the entities the measure ranked first; a per-axis share
    # divides each by its own total, so ranking the shares answers another
    # question, and the advice says so.
    by_share = (
        ", which keeps the top entities by share, not by the measure the limit ranked"
        if scope is DenominatorScope.PER_AXIS
        else ""
    )
    if per_entity_cut or per_axis_cut:
        # A share keyed by an axis keeps that role for a rank to read; one
        # narrowed to an entity keeps none, so it reads its entity from the
        # frame the limited rank reads, which holds every year.
        order = (
            f"compute the share_of first, then rank the shares per period with the limit{by_share}"
            if axis is not None
            else "read the share_of's side from the frame that rank reads, keeping its entity"
        )
        return (
            "a share_of per entity divides each entity over the periods both sides report, and "
            f"{_CUT_PER_PERIOD}, so its shares would cover only the periods it made the cut; "
            f"{order}"
        )
    # Only a numerator descending from the same limited frame is sure to hold
    # nothing the limit dropped, and it would divide each entity's value by
    # itself, so every per-axis share over a limited denominator is refused.
    if scope is DenominatorScope.PER_AXIS and other is not None and other.limited:
        return (
            "a per-axis share_of divides each entity by its own denominator group, and a rank "
            "limit kept only some entities on the denominator side, so analysis-mcp refuses "
            "the call at the first numerator entity the limit dropped; compute the share_of "
            f"over the unlimited denominator, then rank the shares with the limit{by_share}"
        )
    reason = _rejects_share_of_axis(shape, axis=axis, scope=scope, other=other)
    if reason is None and other is not None:
        reason = _rejects_pooled_rate(shape, axis=axis, scope=scope, other=other)
    return reason


# What a side `quotient_of` refuses holds, by its value kind. analysis-mcp
# refuses each (`quotient_of_side_unit_unsupported`, `derived_input_ordinal`),
# and a side in a composed unit with them (`_no_quotient_part`): none is an
# amount in a unit of its own to compose into the result's.
_NO_QUOTIENT_PART: Final = {
    ValueKind.RATIO: "a share or a relative growth, a ratio already",
    ValueKind.ORDINAL: "a rank position, which has no unit",
}


def _no_quotient_part(shape: FrameShape) -> str | None:
    """What `shape` holds that no `quotient_of` divides, or `None` when it has a unit of its own."""
    if shape.composed_unit:
        return "a quotient already, or a figure computed from one, in a unit two measures compose"
    return _NO_QUOTIENT_PART.get(shape.value)


def _rejects_quotient_of_axis(
    shape: FrameShape, *, axis: str | None, denominator_axis: str | None, other: FrameShape | None
) -> str | None:
    """Return why a quotient_of's keys name no role of its sides, or `None` when they do.

    `shape` is the numerator, looked up under `axis`, and `other` the
    denominator, looked up under `denominator_axis`, its own phrase for the
    same entities. With `axis` alone every key divides by the denominator's
    one total, and the denominator's roles are not read.
    """
    if axis is None:
        return None
    reason = _names_no_role("axis", axis, "the numerator", shape.roles)
    if reason is None and denominator_axis is not None and other is not None:
        reason = _names_no_role(
            "denominator_axis", denominator_axis, "the denominator", other.roles
        )
        if reason is not None:
            return f"{reason}, so no key has a counterpart to divide by"
    return reason


def _rejects_quotient_of(
    shape: FrameShape,
    *,
    axis: str | None = None,
    denominator_axis: str | None = None,
    other: FrameShape | None = None,
) -> str | None:
    reason = (
        _several_measures("quotient_of", shape)
        or _mixed_levels("a quotient", shape)
        or _rejects_paired_periods("quotient_of", shape, other)
    )
    if reason is not None:
        return reason
    if (held := _no_quotient_part(shape)) is not None:
        return (
            "quotient_of divides one measured value by another, and this side holds "
            f"{held}; divide the measured series it was computed from"
        )
    return _rejects_quotient_of_axis(
        shape, axis=axis, denominator_axis=denominator_axis, other=other
    )


def _rejects_compare(shape: FrameShape, *, other: FrameShape | None = None) -> str | None:
    return (
        _several_measures("compare", shape)
        or _rejects_paired_periods("compare", shape, other)
        or _rejects_one_sided_rate("compare", shape, other)
    )


def _rejects_sum_of(shape: FrameShape, *, other: FrameShape | None = None) -> str | None:
    reason = _several_measures("sum_of", shape) or _rejects_paired_periods("sum_of", shape, other)
    if reason is not None:
        return reason
    if shape.composed_unit or shape.value in (ValueKind.RATIO, ValueKind.ORDINAL):
        held = "per-unit figure from a quotient_of" if shape.composed_unit else shape.value.value
        return (
            f"sum_of adds two values in one unit, and this side holds a {held}, "
            "which does not add up; sum the measured series first, then compute on the total"
        )
    if other is not None and other.value is not shape.value:
        return (
            f"sum_of adds two values of one kind, and one side holds a {shape.value.value} "
            f"while the other holds a {other.value.value}; read both sides the same way"
        )
    return None


def _rejects_join(shape: FrameShape, *, other: FrameShape | None = None) -> str | None:
    return _several_measures("join", shape) or _rejects_paired_periods("join", shape, other)


_REJECT_RULES: dict[ShapeVerb, Callable[..., str | None]] = {
    ShapeVerb.AGGREGATE: _rejects_aggregate,
    ShapeVerb.GROWTH: _rejects_growth,
    ShapeVerb.RANK: _rejects_rank,
    ShapeVerb.FILTER: _rejects_filter,
    ShapeVerb.SHARE_OF: _rejects_share_of,
    ShapeVerb.QUOTIENT_OF: _rejects_quotient_of,
    ShapeVerb.COMPARE: _rejects_compare,
    ShapeVerb.SUM_OF: _rejects_sum_of,
    ShapeVerb.JOIN: _rejects_join,
}


def rejects(verb: ShapeVerb, shape: FrameShape, **options: object) -> str | None:
    """Return why `verb` may not read `shape`, or `None` when it may.

    The single place every verb's *in* rule is written, so `planner.plan`
    and `compiler.shapes` share one answer to "may this step read that
    shape". A two-sided verb (`share_of`, `quotient_of`, `compare`, `sum_of`,
    `join`) is called once per side, with that side as `shape`; its
    cross-side checks take the counterpart as the `other` keyword. The
    period labels both sides match on read the same regardless of which
    side is passed as which; the axis of a `share_of` or a `quotient_of` is
    stated on the numerator's call, with the denominator as `other`.

    Args:
        verb: Which verb's rule to check.
        shape: The candidate input shape.
        options: The verb's own options -- see each rule function for which
            it reads (`by_entity`/`by_period`/`by_breakdown`/`reduction` for aggregate,
            `change` for growth, `axis`/`among`/`window`/`per_period`/`reduction` for rank,
            `axis`/`scope`/`other`/`entity` for share_of, where `entity` is
            the one entity the side passed as `shape` narrows to,
            `axis`/`denominator_axis`/`other` for quotient_of, and
            `other` for compare, sum_of and join), and `place_level` and
            `period_threshold` for filter, whether it states a
            `PlaceLevel` and a `PeriodThreshold`.

    Returns:
        A reader-facing reason `verb` refuses `shape`, or `None` when it
        does not.
    """
    return _REJECT_RULES[verb](shape, **options)
