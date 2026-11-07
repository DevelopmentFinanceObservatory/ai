"""The select checks `make_charts` (`harness.orchestrator_tools`) runs before
a write: the binding shapes the platform certainly refuses, and the ranking
of one dimension's members that cuts a select over the series cap.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from analysis_mcp.rolling_tool import coordinate_text
from common.place_type import PARTY_ROLES, PLACE_LEVELS, levels_of
from dfo_frame_grammar.grammar import FACET_IDENTITY_COLUMNS, UNIT_COLUMN
from dfo_frame_grammar.rolling import latest_anchor

from harness.tool_contracts import HarnessCode, harness_refusal

# The apis service's `MAX_SERIES_PER_COMPONENT`: the assembler refuses a component
# whose selection's estimated series count (`_series_estimate`) exceeds this.
MAX_SERIES_PER_COMPONENT = 24
# The most rows the apis service's assembler draws in one table (`MAX_TABLE_ROWS` in
# dfo-data-tools' `assembler/limits.py`): series times periods.
MAX_TABLE_ROWS = 2_000
# The catalogue grammar's `maxItems` on a `line_chart` select's one array
# value (the release document's `intent_schemas_tolerant.line_chart`). The
# platform refuses a longer list as `schema_invalid`, and a select resolving to
# more than this many series as `resource_exhausted` (limit
# `MAX_LINE_CHART_SERIES`).
LINE_CHART_MAX_VALUES = 8
# Charts one make_charts call may draw.
MAX_CHARTS_PER_CALL = 8
# The apis service's `facet_dimension_names` (dfo-data-tools' `assembler/frame_view.py`):
# the facet columns a select may leave unbound. The platform refuses one only
# when it splits a series, and its estimate counts an unbound facet as 1.
_FACET_DIMENSIONS = frozenset({UNIT_COLUMN, *FACET_IDENTITY_COLUMNS})


def _plural_dimension_names(view: dict[str, Any]) -> set[str]:
    """The view's non-facet dimension names marked `plural: true` -- the ones
    a `stat_card` select must pin to a single value or the platform resolves
    more than one series. A facet column (`_FACET_DIMENSIONS`) is exempt: the
    platform refuses an unbound facet only when it splits a series, and a
    plural `unit` such as `{USDollar, null}` often names one series with a
    unit on some rows. A dimension entry that is not a `{name, plural,
    values}` mapping (older scripted fixtures list bare dimension names)
    carries no plurality signal, so it is skipped rather than guessed at."""
    return {
        str(dim["name"])
        for dim in view.get("dimensions") or []
        if isinstance(dim, dict)
        and dim.get("plural")
        and dim.get("name")
        and dim["name"] not in _FACET_DIMENSIONS
    }


def periods_covered(view: Mapping[str, Any]) -> int:
    """The periods a table of a frame with `view` draws rows at: every period
    from the first to the last the frame covers, gaps included, as apis
    binds a table's window. A day counts as up to 31 a month, so the count
    errs high, never low. A span frame (`2020/2024`) has one period."""
    coverage = view.get("coverage") or []
    if len(coverage) != 2 or "/" in coverage[0]:
        return 1
    first, last = ([int(part) for part in bound.split("-")] for bound in coverage)
    years = last[0] - first[0]
    if len(first) == len(last) == 1:
        return years + 1
    months = years * 12 + last[1] - first[1]
    if len(first) == len(last) == 2:
        return months + 1
    return months * 31 + last[-1] - first[-1] + 1


def unsplit_facets(view: Mapping[str, Any]) -> frozenset[str]:
    """The facet columns (`_FACET_DIMENSIONS`) `view` marks `plural: false`.
    apis computes `plural` over the series keys under the frame's
    continuation (`frame_view.series_keys` in dfo-data-tools), so such a
    facet splits no series. An estimate's method on the rows that continue
    an observed series names that same series."""
    return frozenset(
        str(dim["name"])
        for dim in view.get("dimensions") or []
        if isinstance(dim, dict)
        and dim.get("name") in _FACET_DIMENSIONS
        and dim.get("plural") is False
    )


def chart_draws(*, type_: str, view: Mapping[str, Any]) -> int:
    """The most members of one dimension a `type_` chart of a frame with
    `view` draws: `LINE_CHART_MAX_VALUES` lines, `MAX_SERIES_PER_COMPONENT`
    bars or table series, and for a table no more than fit `MAX_TABLE_ROWS`
    over the periods it covers (`periods_covered`)."""
    if type_ == "line_chart":
        return LINE_CHART_MAX_VALUES
    if type_ == "table":
        return max(1, min(MAX_SERIES_PER_COMPONENT, MAX_TABLE_ROWS // periods_covered(view)))
    return MAX_SERIES_PER_COMPONENT


def select_binding_ok(
    *,
    type_: str,
    select: dict[str, Any],
    view: dict[str, Any],
) -> dict[str, Any] | None:
    """The binding shapes the platform certainly refuses for their shape
    (measured live against a 7-series frame, and against the catalogue
    grammar): a select binding a dimension the view does not declare; a
    `stat_card` select that leaves a non-facet plural dimension unpinned; and
    for `line_chart` and `bar_chart`, a select that arrays more than one
    dimension. `table` may array more than one dimension; the catalogue
    grammar restricts the single-array-dimension rule to `line_chart` and
    `bar_chart`. A non-mapping `select` (a model that emitted a JSON array or
    scalar for select_json) is a rejection reason here rather than an
    `AttributeError` on `.items()`. Returns `None` when the binding is fine,
    else the harness refusal `select_invalid`, with the reason as its
    `message`. The rejection is caught here, before a `write_component`
    round trip. The refusal of a select binding an undeclared dimension lists
    every dimension name the view declares, in the view's order, as
    `fix.dimensions`. A select over the series a chart draws is the
    platform's to judge (`over_series_limit`)."""
    if not isinstance(select, dict):
        return _select_invalid(f"{type_} select is not a JSON object: {select!r}")
    declared = _declared_dimensions(view)
    undeclared = sorted(select.keys() - declared.keys()) if declared else []
    if undeclared:
        return _select_invalid(
            f"{type_} select binds {undeclared}, which the frame does not declare. "
            f"The frame's dimensions are {list(declared)}",
            fix={"dimensions": list(declared)},
        )
    array_dims = sorted(name for name, value in select.items() if isinstance(value, list))
    if type_ == "stat_card":
        reason = _stat_card_binding_reason(select=select, view=view, array_dims=array_dims)
        return _select_invalid(reason) if reason is not None else None
    if type_ in ("line_chart", "bar_chart") and len(array_dims) > 1:
        return _select_invalid(
            f"{type_} select carries array values for more than one dimension: {array_dims}"
        )
    return None


def over_series_limit(*, type_: str, select: dict[str, Any], view: dict[str, Any]) -> bool:
    """Whether dfo-apis certainly refuses a `type_` select for the series it
    lists or is estimated at, which `make_charts` cuts before a write: a
    `line_chart` array longer than `LINE_CHART_MAX_VALUES` (the catalogue's
    `maxItems`, refused `schema_invalid`), or a `_series_estimate` past
    `MAX_SERIES_PER_COMPONENT` (`_check_series_estimate` in dfo-data-tools'
    `assembler/resolve.py`)."""
    lists = [len(value) for value in select.values() if isinstance(value, list)]
    if type_ == "line_chart" and any(length > LINE_CHART_MAX_VALUES for length in lists):
        return True
    declared = _declared_dimensions(view)
    return _series_estimate(select=select, declared=declared) > MAX_SERIES_PER_COMPONENT


def every_series_selects(*, type_: str, view: dict[str, Any]) -> list[dict[str, Any]]:
    """The selects that may name every series of a frame with `view`, for a
    chart whose select is empty, in the order to try them: each binds one
    non-facet dimension to every value it holds, a dimension of one value
    before one of several. The view lists a dimension's values up to a cap,
    so only a dimension whose listed values number its `total_count` (1 where
    the view states none) is bound, and only a select `select_binding_ok`
    passes is kept. A select over the series the chart draws is cut by
    `make_charts` or judged by the platform.

    The view counts a dimension's values without its empty cells, and a row
    with no value on the bound dimension is not named, so a select here names
    every series only when `covers_every_row` says so of the frame's rows.
    """
    listed = [
        (str(dim["name"]), list(dim["values"]))
        for dim in view.get("dimensions") or []
        if isinstance(dim, dict)
        and dim.get("name")
        and dim["name"] not in _FACET_DIMENSIONS
        and isinstance(dim.get("values"), list)
        and dim["values"]
        and len(dim["values"]) == dim.get("total_count", 1)
    ]
    selects = [
        {name: values[0] if len(values) == 1 else values}
        for name, values in sorted(listed, key=lambda entry: len(entry[1]) > 1)
    ]
    return [
        select
        for select in selects
        if select_binding_ok(type_=type_, select=select, view=view) is None
    ]


def covers_every_row(select: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> bool:
    """Whether every one of `rows` holds a value on each dimension `select`
    binds, so that a select listing every value of those dimensions names
    every row."""
    return all(row.get(name) is not None for row in rows for name in select)


def empty_select_refusal(
    *, type_: str, view: dict[str, Any], unread: bool = False
) -> dict[str, Any]:
    """The refusal of an empty select on a frame no select of
    `every_series_selects` names every series of, or when `unread`, on a
    frame whose rows could not be read to check one, with every dimension
    name the view declares as `fix.dimensions`. Either way a select that
    binds a dimension draws, so the code is the same."""
    declared = list(_declared_dimensions(view))
    reason = (
        "the frame's rows could not be read just now to check that it would show every series"
        if unread
        else "no dimension of the frame lists every series: the view lists only some of its "
        "values, or some series hold no value on it"
    )
    return _select_invalid(
        f"{type_} select is empty, and {reason}. Bind a dimension to a value the view lists "
        f"for it, from {declared}",
        fix={"dimensions": declared},
    )


def over_cap_dimension(*, select: dict[str, Any], view: dict[str, Any]) -> tuple[str, int] | None:
    """The one dimension whose values multiply `_series_estimate` past one,
    unbound or listed, with the factor it contributes, when exactly one does;
    None when none or several do."""
    multiplying = _multiplying_dimensions(select=select, declared=_declared_dimensions(view))
    return next(iter(multiplying.items())) if len(multiplying) == 1 else None


def _multiplying_dimensions(*, select: dict[str, Any], declared: dict[str, int]) -> dict[str, int]:
    """Each dimension whose values multiply `_series_estimate` past one, with
    the factor it contributes: the length of a listed array, or the
    `total_count` of a non-facet dimension `select` leaves unbound. Fewest
    values first, then by name."""
    factors = {
        name: len(select[name]) if isinstance(select.get(name), list) else declared[name]
        for name in declared.keys() | select.keys()
        if isinstance(select.get(name), list)
        or (name not in select and name not in _FACET_DIMENSIONS)
    }
    return dict(
        sorted(
            ((name, count) for name, count in factors.items() if count > 1),
            key=lambda factor: (factor[1], factor[0]),
        )
    )


@dataclass(frozen=True)
class TopMembers:
    """The `members` of one dimension with the largest values at `period`,
    highest first, out of the `holding` compared members that hold a value
    there. `unvalued` are the compared members with no value at `period`, and
    `other_kinds` counts the members left uncompared as group totals, unallocated
    amounts or members of another level."""

    members: list[str]
    period: str
    holding: int
    unvalued: list[str]
    other_kinds: int


# The frame columns that are not coordinates of a series.
_NOT_COORDINATES = frozenset({"time_period", "value", "status"})


def top_members(
    *,
    rows: Sequence[Mapping[str, Any]],
    select: dict[str, Any],
    dimension: str,
    limit: int | None,
    cells: Mapping[str, str] | None = None,
) -> TopMembers | None:
    """The members of `dimension` among `rows` matching `select` with the
    largest values at one period, at most `limit` of them, or all when `limit`
    is None.

    The period is the one period the members compared hold a value at, a
    year, a span or any other label, when they hold only one between them.
    When they hold several it is the frame grammar's anchor (`latest_anchor`)
    of those members: the newest period at least `ANCHOR_SHARE` of them hold
    a value at. The members compared are those `select` names, or with `cells`, each
    member's kind or type cell, only the ones one ranking compares
    (`comparable_members`). A member outside them does not move the period.
    Only a member holding a value at the period is ranked. Cells are compared
    as apis spells them in a view and a select (`coordinate_text`).

    Returns:
        None when there is nothing to rank honestly: no matching rows, rows
        that differ on any other coordinate, a
        member with two rows for one period, or no member to compare. A row
        with no member names nothing a select can list.

    Raises:
        RollingError: The compared members hold several periods and have no
            anchor: no period is held by enough of them, or their periods are
            spans or mix grains.
    """
    matching = _matching_rows(rows=rows, select=select, dimension=dimension)
    if not matching or _differing(matching, dimension=dimension):
        return None
    by_member: dict[str, dict[str, Any]] = {}
    for row in matching:
        periods = by_member.setdefault(str(row[dimension]), {})
        if row["time_period"] in periods:
            return None
        periods[row["time_period"]] = row.get("value")
    compared = set(by_member)
    if cells is not None:
        compared &= comparable_members(
            {member: cells[member] for member in by_member if member in cells},
            dimension=dimension,
        )
    if not compared:
        return None
    observed = {
        member: {period: float(value) for period, value in periods.items() if value is not None}
        for member, periods in by_member.items()
        if member in compared
    }
    held = set().union(*observed.values())
    period = held.pop() if len(held) == 1 else latest_anchor([observed]).period
    values = {member: periods[period] for member, periods in observed.items() if period in periods}
    ranked = sorted(values, key=lambda member: (-values[member], member))
    return TopMembers(
        members=ranked if limit is None else ranked[:limit],
        period=period,
        holding=len(values),
        unvalued=[member for member in observed if member not in values],
        other_kinds=len(by_member) - len(compared),
    )


def _matching_rows(
    *, rows: Sequence[Mapping[str, Any]], select: dict[str, Any], dimension: str
) -> list[dict[str, Any]]:
    """The `rows` that name a `dimension` member and match `select`, their
    cells spelled as a view and a select spell them (`coordinate_text`)."""
    listed = select.get(dimension)
    keyed: list[dict[str, Any]] = [
        {name: coordinate_text(cell) for name, cell in row.items()} for row in rows
    ]
    return [
        row
        for row in keyed
        if row.get(dimension) is not None
        and all(
            row.get(name) in value if isinstance(value, list) else row.get(name) == value
            for name, value in select.items()
            if name != dimension
        )
        and (not isinstance(listed, list) or row.get(dimension) in listed)
    ]


def _differing(matching: Sequence[Mapping[str, Any]], *, dimension: str) -> list[str]:
    """The coordinates other than `dimension` that the `matching` rows hold
    more than one value of, in name order."""
    coordinates = set().union(*matching) - _NOT_COORDINATES - {dimension}
    return sorted(name for name in coordinates if len({row.get(name) for row in matching}) > 1)


def latest_shared_period(
    *, rows: Sequence[Mapping[str, Any]], select: dict[str, Any]
) -> str | None:
    """The latest period at which every series among `rows` matching `select`
    holds a value, the period apis draws a bar chart or a stat card at when
    its intent names no window; None when no rows match or the series share
    no period. A series is the rows sharing every coordinate, and cells are
    compared as apis spells them in a select (`coordinate_text`)."""
    held: dict[tuple[tuple[str, Any], ...], set[str]] = {}
    for row in rows:
        keyed = {
            name: cell if name in _NOT_COORDINATES else coordinate_text(cell)
            for name, cell in row.items()
        }
        if keyed.get("value") is None or not all(
            keyed.get(name) in value if isinstance(value, list) else keyed.get(name) == value
            for name, value in select.items()
        ):
            continue
        series = tuple(sorted((n, c) for n, c in keyed.items() if n not in _NOT_COORDINATES))
        held.setdefault(series, set()).add(str(keyed["time_period"]))
    shared = set.intersection(*held.values()) if held else set()
    return max(shared, default=None)


def comparable_members(cells: Mapping[str, str], *, dimension: str) -> frozenset[str]:
    """The members of `dimension` that one ranking compares, from each member's
    served kind or type cell (`common.place_type`): on a flow party role, the
    members of kind `party`, official and private alike, so a group total such
    as the G7, a whole-population total and an unallocated residual are left out; on a
    place role, the members at the level in `PLACE_LEVELS` most of them sit
    at, so a World Bank aggregate or a region among countries is left out."""
    if dimension in PARTY_ROLES:
        return frozenset(member for member, kind in cells.items() if kind == "party")
    levels = {member: levels_of(cell) & PLACE_LEVELS for member, cell in cells.items()}
    counts = Counter(level for member_levels in levels.values() for level in member_levels)
    if not counts:
        return frozenset()
    level = max(sorted(counts), key=counts.__getitem__)
    return frozenset(member for member, member_levels in levels.items() if level in member_levels)


def _select_invalid(message: str, *, fix: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """The harness refusal for a select the platform would refuse for its
    shape. A select written another way draws, so its retry mode is
    `correct_and_retry`."""
    return harness_refusal(code=HarnessCode.SELECT_INVALID, message=message, fix=fix)


def _stat_card_binding_reason(
    *, select: dict[str, Any], view: dict[str, Any], array_dims: list[str]
) -> str | None:
    """Why a `stat_card` select resolves to more than one series: an array
    value, or a non-facet plural dimension left unpinned. `None` when it
    names one."""
    if array_dims:
        return f"stat_card select carries array values for {array_dims}"
    unpinned = sorted(_plural_dimension_names(view) - set(select))
    if unpinned:
        return f"stat_card select leaves plural dimension(s) {unpinned} unpinned"
    return None


def _declared_dimensions(view: dict[str, Any]) -> dict[str, int]:
    """Each dimension the view lists, with its `total_count`. An entry with
    no `total_count`, or a bare dimension name (older scripted fixtures),
    counts as 1, so a view missing the figure never refuses."""
    declared: dict[str, int] = {}
    for dim in view.get("dimensions") or []:
        if isinstance(dim, str) and dim:
            declared[dim] = 1
        elif isinstance(dim, dict) and dim.get("name"):
            count = dim.get("total_count")
            declared[str(dim["name"])] = count if isinstance(count, int) else 1
    return declared


def _series_estimate(*, select: dict[str, Any], declared: dict[str, int]) -> int:
    """The apis service's own upper bound on the series `select` resolves to
    (`_check_series_estimate` in dfo-data-tools' `assembler/resolve.py`):
    the product, over every declared dimension, of the bound array's
    length, 1 for a single bound value or an unbound facet column, and
    `max(total_count, 1)` for any other dimension `select` leaves unbound.
    `select_binding_ok` has already refused a select binding an undeclared
    dimension, so the dimensions `select` binds count only when the view
    declares none."""
    estimate = 1
    for name in declared.keys() | select.keys():
        if name in select:
            estimate *= len(select[name]) if isinstance(select[name], list) else 1
        elif name not in _FACET_DIMENSIONS:
            estimate *= max(declared[name], 1)
    return estimate
