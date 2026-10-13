"""The select checks `make_charts` (`harness.orchestrator_tools`) runs before
a write: the binding shapes the platform certainly refuses.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from analysis_mcp.rolling_tool import coordinate_text
from common.place_type import PARTY_ROLES, PLACE_LEVELS, levels_of
from dfo_frame_grammar.grammar import FACET_IDENTITY_COLUMNS, UNIT_COLUMN

from harness.tool_contracts import (
    harness_refusal,
)

# dfo-apis's `MAX_SERIES_PER_COMPONENT`: the assembler refuses a component
# whose selection's estimated series count (`_series_estimate`) exceeds this.
MAX_SERIES_PER_COMPONENT = 24
# The most rows dfo-apis's assembler draws in one table (`MAX_TABLE_ROWS` in
# dfo-data-tools' `assembler/limits.py`): series times periods.
MAX_TABLE_ROWS = 2_000
# The catalogue grammar's `maxItems` on a `line_chart` select's one array
# value (the release document's `intent_schemas_tolerant.line_chart`). The
# platform refuses a longer list as `schema_invalid`.
LINE_CHART_MAX_VALUES = 8
# Charts one make_charts call may draw.
MAX_CHARTS_PER_CALL = 8
# dfo-apis's `facet_dimension_names` (dfo-data-tools' `assembler/frame_view.py`):
# the facet columns a select may leave unbound. The platform refuses one only
# when it splits a series, and its estimate counts an unbound facet as 1.
_FACET_DIMENSIONS = frozenset({UNIT_COLUMN, *FACET_IDENTITY_COLUMNS})

# The codes of the harness refusals `select_binding_ok` returns.
SERIES_LIMIT_EXCEEDED = "series_limit_exceeded"
_SELECT_INVALID = "select_invalid"


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


def select_binding_ok(
    *, type_: str, select: dict[str, Any], view: dict[str, Any]
) -> dict[str, Any] | None:
    """The binding shapes the platform certainly refuses (measured live
    against a 7-series frame, and against the catalogue grammar's stated
    per-type caps): a select binding a dimension the view does not
    declare; a `stat_card` select that leaves a non-facet plural dimension
    unpinned; for `line_chart` and `bar_chart`, a select that arrays more
    than one dimension; a `line_chart` array longer than
    `LINE_CHART_MAX_VALUES`; and for every other type, a select whose
    `_series_estimate` exceeds `MAX_SERIES_PER_COMPONENT` (24), the cap
    dfo-apis's assembler enforces although the catalogue grammar's
    `maxItems` allows an array of 30. `table` may array more than one
    dimension; the catalogue grammar restricts the single-array-dimension
    rule to `line_chart` and `bar_chart`. A non-mapping `select`
    (a model that emitted a JSON array or scalar for select_json)
    is a rejection reason here rather than an `AttributeError` on
    `.items()`. Returns `None` when the binding is fine, else
    the harness refusal that says why it is not: `series_limit_exceeded`
    for a select over the series cap and `select_invalid` for every other
    shape, each with the reason as its `message`. It is caught here, before
    a `write_component` round trip."""
    if not isinstance(select, dict):
        return _select_invalid(f"{type_} select is not a JSON object: {select!r}")
    declared = _declared_dimensions(view)
    undeclared = sorted(select.keys() - declared.keys()) if declared else []
    if undeclared:
        return _select_invalid(
            f"{type_} select binds {undeclared}, which the frame does not declare"
        )
    array_dims = sorted(name for name, value in select.items() if isinstance(value, list))
    if type_ == "stat_card":
        reason = _stat_card_binding_reason(select=select, view=view, array_dims=array_dims)
        return _select_invalid(reason) if reason is not None else None
    array_issue = _chart_array_refusal(type_=type_, select=select, array_dims=array_dims)
    if array_issue is not None:
        return array_issue
    return _series_refusal(type_=type_, select=select, declared=declared)


def _series_refusal(
    *, type_: str, select: dict[str, Any], declared: dict[str, int]
) -> dict[str, Any] | None:
    """The refusal of a select whose `_series_estimate` exceeds
    `MAX_SERIES_PER_COMPONENT`; None for a select within it. Where two or
    more dimensions multiply, the message goes on to name the selects that
    draw (`_one_dimension_per_chart`) and the refusal is `correct_and_retry`."""
    series = _series_estimate(select=select, declared=declared)
    if series <= MAX_SERIES_PER_COMPONENT:
        return None
    over = (
        f"{type_} select expands to an estimated {series} series, "
        f"over the platform's maximum of {MAX_SERIES_PER_COMPONENT}"
    )
    multiplying = _multiplying_dimensions(select=select, declared=declared)
    if len(multiplying) < 2:
        return harness_refusal(code=SERIES_LIMIT_EXCEEDED, message=over)
    return harness_refusal(
        code=SERIES_LIMIT_EXCEEDED,
        message=f"{over}: {_one_dimension_per_chart(multiplying)}",
        retry_mode="correct_and_retry",
    )


def over_cap_dimension(*, select: dict[str, Any], view: dict[str, Any]) -> str | None:
    """The one dimension whose values multiply `_series_estimate` past one,
    unbound or listed, when exactly one does; None when none or several do."""
    multiplying = _multiplying_dimensions(select=select, declared=_declared_dimensions(view))
    return next(iter(multiplying)) if len(multiplying) == 1 else None


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


def _one_dimension_per_chart(multiplying: dict[str, int]) -> str:
    """What `make_charts` accepts in place of a select on which the two or
    more `multiplying` dimensions (fewest values first) pass the series cap:
    each chart binds all but one of them to a single value. The dimension
    with the most values is the one left to list, so the charts that cover
    the whole select number the product of the others' counts, and the
    sentence says whether that many fit one call."""
    *bound, (_, most) = multiplying.items()
    named = _and_list([f"{name} ({count} values)" for name, count in multiplying.items()])
    repair = (
        f"{named} multiply. Bind all but one of them to a single value in each chart, "
        "so each chart lists one dimension's members."
    )
    if most > MAX_SERIES_PER_COMPONENT:
        repair += (
            " A chart whose one listed dimension is over the maximum draws its largest "
            "members and says so."
        )
    charts = math.prod(count for _, count in bound)
    each = (
        f"each {bound[0][0]} value"
        if len(bound) == 1
        else f"each combination of {_and_list([name for name, _ in bound])} values"
    )
    if charts <= MAX_CHARTS_PER_CALL:
        return f"{repair} A chart for {each} is {charts} charts, which fits one make_charts call."
    return (
        f"{repair} A chart for {each} is {charts} charts, more than the "
        f"{MAX_CHARTS_PER_CALL} one make_charts call draws: draw the charts for the values "
        "the question names, or ask the user which to show."
    )


def _and_list(items: list[str]) -> str:
    """`items` as an English list: "a and b", "a, b, and c"."""
    if len(items) <= 2:
        return " and ".join(items)
    return f"{', '.join(items[:-1])}, and {items[-1]}"


@dataclass(frozen=True)
class TopMembers:
    """The largest `members` of one dimension, highest first, out of `total`
    members the selection holds, each by its value at `period`, or at its
    own latest period when `period` is None."""

    members: list[str]
    total: int
    period: str | None


# The frame columns that are not coordinates of a series.
_NOT_COORDINATES = frozenset({"time_period", "value", "status"})


def top_members(
    *,
    rows: Sequence[Mapping[str, Any]],
    select: dict[str, Any],
    dimension: str,
    limit: int,
    own_latest: bool,
) -> TopMembers | None:
    """The `limit` largest members of `dimension` among `rows` matching
    `select`, ranked by each member's value at the latest period those rows
    hold, or at its own latest period when `own_latest`. Cells are
    compared as dfo-apis spells them in a view and a select
    (`coordinate_text`). None when there is nothing to rank honestly: no
    matching rows, rows that differ on any other coordinate (a unit, a
    source, a measure), or a member with two rows for one period. A member
    with no value at the period it is ranked by is left out of the ranking
    and counted in `total`; a row with no member names nothing a select can
    list."""
    listed = select.get(dimension)
    keyed: list[dict[str, Any]] = [
        {name: coordinate_text(cell) for name, cell in row.items()} for row in rows
    ]
    matching = [
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
    if not matching:
        return None
    coordinates = set().union(*matching) - _NOT_COORDINATES - {dimension}
    if any(len({row.get(name) for row in matching}) > 1 for name in coordinates):
        return None
    by_member: dict[str, dict[str, float]] = {}
    for row in matching:
        periods = by_member.setdefault(str(row[dimension]), {})
        if row["time_period"] in periods:
            return None
        periods[row["time_period"]] = float(row["value"])
    common = max(period for periods in by_member.values() for period in periods)
    ranked_at = {
        member: max(periods) if own_latest else common for member, periods in by_member.items()
    }
    values = {
        member: periods[ranked_at[member]]
        for member, periods in by_member.items()
        if ranked_at[member] in periods
    }
    ranked = sorted(values, key=lambda member: (-values[member], member))[:limit]
    shared = {ranked_at[member] for member in ranked}
    return TopMembers(
        members=ranked, total=len(by_member), period=shared.pop() if len(shared) == 1 else None
    )


def comparable_members(cells: Mapping[str, str], *, dimension: str) -> frozenset[str]:
    """The members of `dimension` that one ranking compares, from each member's
    served kind or type cell (`common.place_type`): on a flow party role, the
    members of kind `party`, so a group total such as the G7 is left out; on a
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


def _chart_array_refusal(
    *, type_: str, select: dict[str, Any], array_dims: list[str]
) -> dict[str, Any] | None:
    """The refusal of a `line_chart` or `bar_chart` select that arrays more
    than one dimension, or of a `line_chart` array longer than
    `LINE_CHART_MAX_VALUES`; None for any other select."""
    if type_ not in ("line_chart", "bar_chart"):
        return None
    if len(array_dims) > 1:
        return _select_invalid(
            f"{type_} select carries array values for more than one dimension: {array_dims}"
        )
    listed = [len(select[name]) for name in array_dims]
    if type_ == "line_chart" and listed and listed[0] > LINE_CHART_MAX_VALUES:
        return harness_refusal(
            code=SERIES_LIMIT_EXCEEDED,
            message=(
                f"line_chart select lists {listed[0]} values on {array_dims[0]!r}, more than "
                f"the {LINE_CHART_MAX_VALUES} one line_chart draws: list at most "
                f"{LINE_CHART_MAX_VALUES}, or show them all as a bar_chart or a table"
            ),
        )
    return None


def _select_invalid(message: str) -> dict[str, Any]:
    """The harness refusal for a select the platform would refuse for its shape."""
    return harness_refusal(code=_SELECT_INVALID, message=message)


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
    """dfo-apis's own upper bound on the series `select` resolves to
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
