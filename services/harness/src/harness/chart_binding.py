"""The select checks `make_charts` (`harness.orchestrator_tools`) runs before
a write: the binding shapes the platform certainly refuses.
"""

from __future__ import annotations

from typing import Any

from dfo_frame_grammar.grammar import FACET_IDENTITY_COLUMNS, UNIT_COLUMN

from harness.tool_contracts import (
    harness_refusal,
)

# dfo-apis's `MAX_SERIES_PER_COMPONENT`: the assembler refuses a component
# whose selection's estimated series count (`_series_estimate`) exceeds this.
MAX_SERIES_PER_COMPONENT = 24
# The catalogue grammar's `maxItems` on a `line_chart` select's one array
# value (the release document's `intent_schemas_tolerant.line_chart`). The
# platform refuses a longer list as `schema_invalid`.
LINE_CHART_MAX_VALUES = 8
# dfo-apis's `facet_dimension_names` (dfo-data-tools' `assembler/frame_view.py`):
# the facet columns a select may leave unbound. The platform refuses one only
# when it splits a series, and its estimate counts an unbound facet as 1.
_FACET_DIMENSIONS = frozenset({UNIT_COLUMN, *FACET_IDENTITY_COLUMNS})

# The codes of the harness refusals `select_binding_ok` returns.
_SERIES_LIMIT_EXCEEDED = "series_limit_exceeded"
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
    series = _series_estimate(select=select, declared=declared)
    if series > MAX_SERIES_PER_COMPONENT:
        return harness_refusal(
            code=_SERIES_LIMIT_EXCEEDED,
            message=(
                f"{type_} select expands to an estimated {series} series, "
                f"over the platform's maximum of {MAX_SERIES_PER_COMPONENT}"
            ),
        )
    return None


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
            code=_SERIES_LIMIT_EXCEEDED,
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
