"""Renders each verb's result into its own model-facing digest.

The harness reads `structuredContent` in code, so a model only ever sees
this digest. `executor.run` reads a verb's declared shape and row count out
of it, keeps this digest text as its step's `StepRecord.digest`, and
`harness.graph` carries that into `narrate`'s `step_summaries`. Everything
the model needs to act on has to be in this text: the result's own
`{spec_id, content_version}` pair, the computed numbers, units, coverage,
warnings, and trace identity. Each `render_*_digest` function reads only
its own verb's result facts, as dfo-apis' analysis runner states them in a
call's `summary` (`GroupSummary`, `EntityGrowth`, `PeriodComparison`,
`PeriodSum`, `PeriodQuotient`, `EntityRank` and the rest), never
`frame.rows`. The frame itself never appears in a digest.
`render_summary_digest` renders a runner summary (`summary.VerbSummary`) with
its verb's renderer. The in-process tools render their engine result's facts
(`engine_facts.facts_of`), the same JSON the runner sends.

Every list is capped at `LIST_CAP` entries, following `data_mcp.digest`'s
own discipline of bounding the model-facing text while the full result stays
reachable, here by resolving the same `{spec_id, content_version}` pair as a
downstream verb's `frame_ref`. `tests/test_digest_budget.py` catches a leak
of that bound. `render_rank_digest` takes the head of its own
already-caller-ordered `entity_ranks`, since `ascending` states which end of
the ranking the caller wants. Every other renderer's own order is
framework-defined (canonical, chronological, or grain-based) rather than
caller-stated, so those pick the shown entries by magnitude instead
(`summary.capped`).
Each renderer's `_truncation_note` call names which rule it used.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from common.echoed_text import truncate_value
from common.time_grain import span_label
from common.unit_refs import PERCENT_UNIT
from dfo_frame_grammar.grammar import UnitCell

from analysis_mcp.artifact import Exclusion, Interval, Time
from analysis_mcp.execution_identity import ExecutionIdentity
from analysis_mcp.summary import (
    LIST_CAP,
    AggregateFacts,
    CompareFacts,
    EntityGrowth,
    EntityRank,
    FilterFacts,
    FilterRowSummary,
    GroupSummary,
    GrowthFacts,
    JoinFacts,
    KeyPairs,
    MatchedKey,
    PeriodComparison,
    PeriodQuotient,
    PeriodShare,
    PeriodSum,
    PooledParties,
    QuotientOfFacts,
    RankFacts,
    RecomputedInput,
    Rolling,
    ShareOfFacts,
    SumOfFacts,
    VerbSummary,
    capped,
)
from analysis_mcp.units import unit_name

# The prefixes of the two lines in the identity block every digest ends with.
# `harness.graph` drops lines starting with either from a plan step's summary.
FRAME_REFERENCE_PREFIX = "frame reference: spec_id="
EXECUTION_IDENTITY_PREFIX = "execution_identity: "


def _format_number(value: float) -> str:
    """Exact, minimal, reproducible: `repr`'s own round-trip string, with a
    bare trailing `.0` dropped. `-0.0` normalises to `0.0` first --
    IEEE-754 addition of `-0.0 + 0.0` is defined to produce positive zero,
    which is a shorter, more portable normalisation than a separate
    equality check.
    """
    value = value + 0.0
    text = repr(value)
    return text[:-2] if text.endswith(".0") else text


def _signed_number_text(value: float) -> str:
    """`_format_number`, with an explicit leading `+` for a non-negative
    value -- `repr`'s own text already carries the `-` for a negative one,
    so nothing is added there. Used for an absolute growth's difference,
    where the sign is the point (an increase vs. a decrease) and
    `growth_rate:+.1%`'s own percentage formatting does not apply."""
    text = _format_number(value)
    return text if text.startswith("-") else f"+{text}"


def _unit_text(unit: UnitCell | None) -> str:
    """`unit` named as `units.unit_name` names it (a converted unit as
    "converted to <price basis>", a composed unit as "<numerator> per
    <denominator>"), capped the same way `truncate_value` caps
    a key value, or `'(none declared)'` when unbound. `unit` is read straight
    off input rows and carries the same unbounded-length risk as any key
    value."""
    if unit is None:
        return "(none declared)"
    return truncate_value(unit_name(unit))


def _key_text(key: Sequence[tuple[str, str]]) -> str:
    """The `column=value, ...` text for an entity/group key tuple. Each
    value is capped (`truncate_value`); each column name is not, since a
    column name is schema, not a per-row data value."""
    if not key:
        return "(all rows)"
    return ", ".join(f"{column}={truncate_value(value)}" for column, value in key)


# How a flow party role's pooled parties are named in a sum's private-party line, one and
# several: its members, its private members, its whole totals and its private whole totals.
# A pool of members beside whole totals names the totals `whole total` on every role.
_MEMBER_NOUNS: dict[str, tuple[str, str]] = {"financingProvider": ("provider", "providers")}
_OTHER_MEMBER_NOUNS = ("party", "parties")
_PRIVATE_MEMBER_NOUNS: dict[str, tuple[str, str]] = {
    "financingProvider": ("private donor", "private donors")
}
_OTHER_PRIVATE_MEMBER_NOUNS = ("private party", "private parties")
_WHOLE_TOTAL_NOUNS: dict[str, tuple[str, str]] = {
    "financingProvider": ("whole total of providers", "whole totals of providers")
}
_WHOLE_TOTALS = ("whole total", "whole totals")
_PRIVATE_TOTAL_NOUNS: dict[str, tuple[str, str]] = {
    "financingProvider": ("total of private donors", "totals of private donors")
}
_OTHER_PRIVATE_TOTAL_NOUNS = ("private total", "private totals")


def _counted(count: int, nouns: tuple[str, str], *, qualifier: str = "") -> str:
    """`3 providers`, or `3 listed providers` with `qualifier` `listed `."""
    return f"{count} {qualifier}{nouns[count != 1]}"


def _pool_text(pooled: PooledParties, *, qualifier: str = "") -> str:
    """`3 providers`, `2 whole totals of providers`, or `1 whole total and 1 provider`: the
    parties a sum pooled, members and whole totals counted apart."""
    members = pooled.count - pooled.total_count
    member_nouns = _MEMBER_NOUNS.get(pooled.role, _OTHER_MEMBER_NOUNS)
    if not pooled.total_count:
        return _counted(members, member_nouns, qualifier=qualifier)
    if not members:
        nouns = _WHOLE_TOTAL_NOUNS.get(pooled.role, _WHOLE_TOTALS)
        return _counted(pooled.count, nouns, qualifier=qualifier)
    totals = _counted(pooled.total_count, _WHOLE_TOTALS, qualifier=qualifier)
    return f"{totals} and {_counted(members, member_nouns)}"


def _private_count_text(pooled: PooledParties, count: int) -> str:
    """`24 private donors`, `1 total of private donors`, or both joined: the `count` private
    parties a sum pooled, whole totals first."""
    totals = pooled.private_total_count or 0
    parts = []
    if totals:
        parts.append(
            _counted(totals, _PRIVATE_TOTAL_NOUNS.get(pooled.role, _OTHER_PRIVATE_TOTAL_NOUNS))
        )
    if count - totals:
        parts.append(
            _counted(
                count - totals,
                _PRIVATE_MEMBER_NOUNS.get(pooled.role, _OTHER_PRIVATE_MEMBER_NOUNS),
            )
        )
    return " and ".join(parts)


def _private_clause(pooled: PooledParties, count: int) -> str:
    """`24 private donors (311400000)`: the `count` private parties a sum pooled and their
    value, over the periods when the sum spans several. A value past the finite range is left
    out, and the count stands alone."""
    if pooled.private_value is None:
        return _private_count_text(pooled, count)
    value = _format_number(pooled.private_value)
    over = f" over {pooled.periods} periods" if pooled.periods > 1 else ""
    return f"{_private_count_text(pooled, count)} ({value}{over})"


def _private_lines(pooled: Sequence[PooledParties], *, subject: str) -> list[str]:
    """One line per pooled role that held private parties: `subject` is the sum the line
    describes, and the private parties are in it because they sum like any other. A role whose
    frame carries no class column gets no line."""
    return [
        f"{subject} of {_pool_text(entry)} includes {_private_clause(entry, entry.private_count)}."
        for entry in pooled
        if entry.private_count
    ]


def _two_sided_pooled_lines(
    result: QuotientOfFacts | ShareOfFacts, *, keyed_roles: frozenset[str] = frozenset()
) -> list[str]:
    """The private-party lines of a two-input verb's sums. A denominator that pools a role the
    numerator is keyed by ("each donor's share") divides by the sum of the listed parties, and
    says so, whether or not private parties are among them."""
    lines = _private_lines(result.numerator_pooled, subject="The numerator sum")
    listed = [entry for entry in result.denominator_pooled if entry.role in keyed_roles]
    for entry in listed:
        text = f"Shares of the sum of the {_pool_text(entry, qualifier='listed ')}"
        if entry.private_count:
            text += f", which includes {_private_count_text(entry, entry.private_count)}"
        lines.append(f"{text}.")
    lines.extend(
        _private_lines(
            [entry for entry in result.denominator_pooled if entry not in listed],
            subject="The denominator sum",
        )
    )
    return lines


_MAGNITUDE_SELECTION = "kept the largest-magnitude entries"
_RANK_ORDER_SELECTION = f"kept the first {LIST_CAP} in rank order, rank 1 first"


def _truncation_note(shown: Sequence[object], total: int, *, selection: str) -> str:
    """A fixed-shape truncation note, at constant cost regardless of the
    true count. A larger candidate pool still changes *which* entries a
    magnitude-selecting renderer keeps, and a wider-magnitude entry costs
    more to render; `tests/test_digest_budget.py`'s truncation-cost gate
    measures that growth and holds it bounded.

    `selection` names which rule picked `shown` for this renderer; each
    call site passes its own rather than this function guessing from
    `shown` alone, since a single hardcoded string would misstate a
    positional renderer's selection (`render_rank_digest`'s).
    """
    if total <= len(shown):
        return ""
    return (
        f"\n  ... more entries not shown ({selection}); the header's own count is the true total."
    )


_EXCLUSION_REASON_TEXT: dict[str, str] = {
    "unit_unbound": "left unit unbound",
    "unit_not_selected": "declared a unit other than the one selected",
    "outside_window": "outside the requested window",
    "endpoint_missing": "had no row in window_start or window_end",
    "zero_or_sign_crossing_base": "had a zero or sign-crossing base, so a relative change is undefined",
    "non_finite": "computed to a non-finite value",
    "predicate": "removed by the predicate",
    "unmatched_key": "had no matching key on the other side",
    "beyond_limit": "ranked beyond the limit",
    "party_total": "totals of other parties, left out so no figure counts twice",
    "party_residual": "unallocated residuals, left out of a ranking of members",
}

# `entity_not_selected` and `period_unmatched` are excluded here: a
# caller's own `entity` binding asked for those rows to go, and an
# unmatched period already has its own representation on the coverage line
# (`render_compare_digest`, `render_sum_of_digest`).
_DIGEST_HIDDEN_EXCLUSION_REASONS = frozenset({"entity_not_selected", "period_unmatched"})

# Cap on how many distinct entity names an exclusion line lists before
# falling back to the bare row count.
_EXCLUSION_ENTITY_NAME_CAP = 5


def _exclusion_group_label(reason: str, side: str | None) -> str:
    """`reason (reason_text)`, or `reason, side (reason_text)` when `side`
    names one of a two-input verb's two sides."""
    reason_text = _EXCLUSION_REASON_TEXT.get(reason, reason)
    if side is None:
        return f"{reason} ({reason_text})"
    return f"{reason}, {side} ({reason_text})"


def _exclusion_entities_text(group: Sequence[Exclusion]) -> str | None:
    """`role=value, role=value, ...` for up to `_EXCLUSION_ENTITY_NAME_CAP`
    names total across every role column `group`'s exclusions declare
    entities for, sorted by role. `None` when no exclusion in `group` names
    any entity.
    """
    by_role: dict[str, list[str]] = {}
    for exclusion in group:
        for role, values in exclusion.entities.items():
            by_role.setdefault(role, []).extend(values)
    pairs = [
        f"{role}={truncate_value(value)}" for role in sorted(by_role) for value in by_role[role]
    ]
    if not pairs:
        return None
    return ", ".join(pairs[:_EXCLUSION_ENTITY_NAME_CAP])


def _exclusions_block(exclusions: Sequence[Exclusion]) -> str | None:
    """One line per (reason, side) pair: the total row count dropped for
    that reason on that side, summed across every narrowing step that
    raised it, plus up to `_EXCLUSION_ENTITY_NAME_CAP` of the entities
    affected, keyed by role column. `None` when `exclusions` is empty, or
    when every exclusion's own reason is in `_DIGEST_HIDDEN_EXCLUSION_REASONS`.
    """
    shown = [
        exclusion
        for exclusion in exclusions
        if exclusion.reason not in _DIGEST_HIDDEN_EXCLUSION_REASONS
    ]
    if not shown:
        return None
    by_group: dict[tuple[str, str | None], list[Exclusion]] = {}
    for exclusion in shown:
        by_group.setdefault((exclusion.reason, exclusion.side), []).append(exclusion)
    lines = []
    for (reason, side), group in by_group.items():
        total_rows = sum(exclusion.rows for exclusion in group)
        label = _exclusion_group_label(reason, side)
        line = f"  {label}: {total_rows} row(s)"
        entities_text = _exclusion_entities_text(group)
        if entities_text is not None:
            line += f" [{entities_text}]"
        lines.append(line)
    return "\n".join(["exclusions:", *lines])


def _join_and(items: Sequence[str]) -> str:
    """An English list: `a`, `a and b`, or `a, b and c`."""
    items = list(items)
    if len(items) <= 1:
        return items[0] if items else ""
    return ", ".join(items[:-1]) + " and " + items[-1]


def _absent_clause(interval: Interval) -> str:
    """`; every period in the span is present` when `interval.contiguous`,
    `; contiguity is unknown` when it's `None`, else `; {absent, and
    absent} absent` -- with a trailing `(N total)` whenever `absent` is a
    truncated sample of `absent_count`, so a capped list can never read as
    exhaustive. Every `time:` line ends with this clause."""
    if interval.contiguous is None:
        return "; contiguity is unknown"
    if interval.contiguous:
        return "; every period in the span is present"
    clause = f"; {_join_and(interval.absent)} absent"
    if interval.absent_count > len(interval.absent):
        clause += f" ({interval.absent_count} total)"
    return clause


def _time_line(time: Time | None, *, per_item_note: str = "") -> str | None:
    """`time: labels stand for the span {start} to {end}...` when `time` is
    an `Interval`, covering which periods the union spans and, via
    `_absent_clause`, which are missing. `per_item_note` points to a
    per-group window shown elsewhere on the digest, when the renderer has
    one. `None` for a plain `Observation` or an absent `time`.
    """
    if not isinstance(time, Interval):
        return None
    text = f"time: labels stand for the span {time.start} to {time.end}"
    if per_item_note:
        text += f"; {per_item_note}"
    return text + _absent_clause(time)


def _rank_time_line(
    time: Time | None, *, per_period: bool, periods_in_window: Sequence[str]
) -> str | None:
    """`time: rows are labelled {label}; real periods span {start} to
    {end}...` for an `Interval`, where `{label}` is the one `time_period`
    every collapsed row carries (`span_label(periods_in_window)`, as
    `engine.rank.run_rank` stamps it). `None` for a plain `Observation` or an
    absent `time`.

    `per_period=True` states which periods were ranked instead, since each
    row carries the period it was ranked in. `None` when `periods_in_window`
    is empty, which a real call never has but a hand-built `RankFacts` in a
    test may.
    """
    if not periods_in_window:
        return None
    if per_period:
        return (
            "time: each row's time_period is the period it was ranked in; "
            f"periods ranked: {', '.join(periods_in_window)}"
        )
    if not isinstance(time, Interval):
        return None
    text = (
        f"time: rows are labelled {span_label(periods_in_window)}; "
        f"real periods span {time.start} to {time.end}"
    )
    return text + _absent_clause(time)


def _artifact_sections(*, exclusions: Sequence[Exclusion], time_line: str | None) -> list[str]:
    """The `exclusions:`/`time:` sections every renderer appends after its
    own `warnings:` block and before `_derived_note`, in that order.
    `time_line` is pre-rendered by the caller: `_time_line`, or
    `_rank_time_line` for rank."""
    sections: list[str] = []
    exclusions_block = _exclusions_block(exclusions)
    if exclusions_block is not None:
        sections.append(exclusions_block)
    if time_line is not None:
        sections.append(time_line)
    return sections


def _derived_note(derived: Mapping[str, object] | None) -> str | None:
    """A fixed-size statement naming which of this result's own inherited
    fields have become unreliable after transformation, or `None` when the
    result carries no derived marker or none of its flags makes a field
    unreliable.

    `derived` is `applied_policy.detail["derived"]` (see
    `frame_ref_resolution.DerivedMarker`), written to the trace by every
    producer verb. Every flag that's set folds into one `"; "`-joined
    sentence rather than a line per flag. `dimensionless` names no field,
    since a share states its unit (per cent) and a relative growth rate
    states none, and both are true of the result.
    """
    if not derived:
        return None
    dishonest: list[str] = []
    if derived.get("inherited_stat_var"):
        dishonest.append("stat_var (inherited; no longer names what this value measures)")
    if derived.get("ordinal"):
        dishonest.append(
            "value (a rank position, or a difference of positions -- not a measured "
            "quantity; a ratio or a relative growth rate over it is meaningless)"
        )
    if not dishonest:
        return None
    return "derived result -- do not reuse as if genuine: " + "; ".join(dishonest) + "."


def _identity_block(
    *, op_spec_hash: str, content_version: str, execution_identity: ExecutionIdentity
) -> str:
    """The identity block every verb's digest ends with, in the same shape
    and order every other renderer here uses."""
    return "\n".join(
        [
            f"{FRAME_REFERENCE_PREFIX}{op_spec_hash}, content_version={content_version} "
            "-- pass both to downstream tools.",
            f"{EXECUTION_IDENTITY_PREFIX}{execution_identity.canonical_string()}",
        ]
    )


def _time_window_text(summary: GroupSummary) -> str:
    if summary.time_period_min == summary.time_period_max:
        return summary.time_period_min
    return f"{summary.time_period_min}..{summary.time_period_max}"


def _own_window_differs(summary: GroupSummary, period_label: str | None) -> bool:
    """Whether a group's own periods cover less than the call-wide label
    every row carries. Always true when there is no call-wide label."""
    if period_label is None:
        return True
    return span_label((summary.time_period_min, summary.time_period_max)) != period_label


def _rows(count: int) -> str:
    """How many input rows a line's value sums or ranks over, as "1 row" or "3 rows"."""
    return f"{count} row" if count == 1 else f"{count} rows"


def _group_line(summary: GroupSummary, *, period_label: str | None) -> str:
    """One group's line. The group's own window shows unless it equals the
    call-wide `period_label` the header states."""
    key_text = _key_text(summary.group_key)
    window = f"{_time_window_text(summary)}, " if _own_window_differs(summary, period_label) else ""
    value_text = _format_number(summary.value)
    if summary.value_period is not None:
        value_text = f"{value_text} ({summary.value_period})"
    return f"  {key_text} [{window}{_rows(summary.row_count)}]: {value_text}"


def render_aggregate_digest(
    *,
    group_by: Sequence[str],
    aggregation: str,
    result: AggregateFacts,
    op_spec_hash: str,
    content_version: str,
    execution_identity: ExecutionIdentity,
    derived: Mapping[str, object] | None = None,
    exclusions: Sequence[Exclusion] = (),
    time: Time | None = None,
) -> str:
    """Render `result` (the facts of `analysis_mcp.engine.aggregate.run_aggregate`)
    into the digest text a successful `aggregate` tool call returns.

    When every row carries one label (`result.period_label`, set under sum or
    mean with `time_period` outside `group_by`), the header states it once, and a
    group's line carries its own earliest and latest period only when they
    cover less than that label. Otherwise every group's line carries them.
    """
    group_by_text = (
        ", ".join(group_by) if group_by else "(none -- whole frame reduced to one group)"
    )
    header = f"aggregate: {aggregation} grouped by {group_by_text}"
    if result.period_label is not None:
        header += f"; every row's time_period is {result.period_label}"

    # Magnitude, not positional: `group_summaries` is ordered by the
    # group_key's canonical sort, not a criterion the caller stated (see
    # this module's docstring). When `group_by` names `time_period`, the
    # group key is the time axis, so a magnitude cap can drop a scattered
    # subset of years out of an otherwise-contiguous run, and a model
    # scanning the shown years reads a gapped chronology as contiguous.
    shown = capped(result.group_summaries, magnitude=lambda summary: summary.value)
    group_lines = [_group_line(summary, period_label=result.period_label) for summary in shown]
    groups_block = "\n".join([f"{len(result.group_summaries)} group(s):", *group_lines])
    groups_block += _truncation_note(
        shown, len(result.group_summaries), selection=_MAGNITUDE_SELECTION
    )

    unit_line = f"unit: {_unit_text(result.resolved_unit)}"
    coverage_line = (
        f"coverage: {result.realised_coverage_fraction:.1%} "
        f"({result.present_cell_count}/{result.expected_cell_count} expected grid cells present, "
        f"{result.skipped_cell_count} missing)"
    )

    sections = [header, groups_block, unit_line, coverage_line]
    if aggregation == "sum":
        sections.extend(_private_lines(result.pooled_parties, subject="This sum"))

    if result.warnings:
        sections.append(
            "\n".join(
                ["warnings:", *(f"  - {truncate_value(warning)}" for warning in result.warnings)]
            )
        )

    if result.period_label is None:
        per_item_note = "each group's own window is on its line"
    elif any(_own_window_differs(summary, result.period_label) for summary in shown):
        per_item_note = "a group whose own periods cover less shows its window on its line"
    else:
        per_item_note = ""
    time_line = _time_line(time, per_item_note=per_item_note)
    sections.extend(_artifact_sections(exclusions=exclusions, time_line=time_line))
    derived_note = _derived_note(derived)
    if derived_note is not None:
        sections.append(derived_note)

    sections.append(
        _identity_block(
            op_spec_hash=op_spec_hash,
            content_version=content_version,
            execution_identity=execution_identity,
        )
    )
    return "\n\n".join(sections)


def _places_text(value: float) -> str:
    """An ordinal growth's own difference, stated in words rather than left
    as a bare signed number: 12th -> 5th is `-7`, and a negative number for
    an improvement is the single most misreadable value this toolkit
    produces. `"N places better"` for a negative change (the rank number
    went down), `"N places worse"` for a positive one, `"no change"` at
    zero. `value` is always a whole number of places (a difference of two
    integers), so `int()` never truncates a real fraction here.
    """
    places = abs(int(value))
    unit = "place" if places == 1 else "places"
    if value < 0:
        return f"{places} {unit} better"
    if value > 0:
        return f"{places} {unit} worse"
    return "no change"


def _growth_line(growth: EntityGrowth, *, change: str, ordinal: bool) -> str:
    key_text = _key_text(growth.entity_key)
    # `ordinal`: the input carried a rank position, so the difference reads
    # in places, in words -- see `_places_text`. `change == "relative"`: a
    # percentage, the established formatting. `change == "absolute"`: a
    # plain signed difference in the input's own unit (stated separately,
    # on `unit_line`) -- a percentage format here would misstate an
    # absolute USD or places difference as a rate.
    if ordinal:
        value_text = _places_text(growth.growth_rate)
    elif change == "relative":
        value_text = f"{growth.growth_rate:+.1%}"
    else:
        value_text = _signed_number_text(growth.growth_rate)
    return (
        f"  {key_text}: {value_text} "
        f"({_format_number(growth.value_start)} -> {_format_number(growth.value_end)})"
    )


def render_growth_digest(
    *,
    window_start: str,
    window_end: str,
    entity_by: Sequence[str],
    result: GrowthFacts,
    op_spec_hash: str,
    content_version: str,
    execution_identity: ExecutionIdentity,
    derived: Mapping[str, object] | None = None,
    exclusions: Sequence[Exclusion] = (),
    time: Time | None = None,
) -> str:
    """Render `result` (the facts of `analysis_mcp.engine.growth.run_growth`) into
    the digest text a successful `growth` tool call returns.

    When the input carried `DerivedMarker.ordinal` (a per-period rank's own
    output, or a growth reading one), every entity's own line and the
    `unit:` line both read in places, in words, rather than as a signed
    number or a bare `None` unit -- see `_places_text`. A relative change is
    a rate with no unit, which the `unit:` line states before naming the
    input's unit.
    """
    entity_by_text = (
        ", ".join(entity_by) if entity_by else "(none -- the whole frame is one entity)"
    )
    header = f"growth: {window_start}..{window_end}, entity_by {entity_by_text}"

    ordinal = bool(derived and derived.get("ordinal"))
    shown = capped(result.entity_growths, magnitude=lambda growth: growth.growth_rate)
    lines = [_growth_line(growth, change=result.change, ordinal=ordinal) for growth in shown]
    entities_block = "\n".join([f"{len(result.entity_growths)} entit(y/ies):", *lines])
    entities_block += _truncation_note(
        shown, len(result.entity_growths), selection=_MAGNITUDE_SELECTION
    )

    if ordinal:
        unit_line = "unit: absolute change (places)"
    elif result.change == "relative":
        # A rate carries no unit, whatever the input was measured in.
        input_text = (
            f"input in {_unit_text(result.resolved_unit)}"
            if result.resolved_unit is not None
            else "the input declared no unit"
        )
        unit_line = f"unit: none (a relative change; {input_text})"
    elif result.resolved_unit == PERCENT_UNIT:
        # A difference of two per cent figures, such as two years of a
        # share_of result or a published "% of GDP" series.
        unit_line = "unit: absolute change (percentage points)"
    elif result.resolved_unit is None and derived is not None and derived.get("dimensionless"):
        # The input was a relative growth rate, a unit-less fraction, so a
        # difference of two rates is a fraction too.
        unit_line = (
            "unit: absolute change (a difference of rates stated as fractions: "
            "0.02 is 2 percentage points)"
        )
    else:
        unit_line = f"unit: absolute change ({_unit_text(result.resolved_unit)})"

    sections = [header, entities_block, unit_line]
    if result.warnings:
        sections.append(
            "\n".join(
                ["warnings:", *(f"  - {truncate_value(warning)}" for warning in result.warnings)]
            )
        )
    sections.extend(_artifact_sections(exclusions=exclusions, time_line=_time_line(time)))
    derived_note = _derived_note(derived)
    if derived_note is not None:
        sections.append(derived_note)
    sections.append(
        _identity_block(
            op_spec_hash=op_spec_hash,
            content_version=content_version,
            execution_identity=execution_identity,
        )
    )
    return "\n\n".join(sections)


def _comparison_line(comparison: PeriodComparison) -> str:
    return (
        f"  {comparison.time_period}: {_format_number(comparison.left_value)} -> "
        f"{_format_number(comparison.right_value)} (delta {_format_number(comparison.delta)})"
    )


def _difference_label(result: CompareFacts) -> str:
    """What `result`'s StatVar measures, in the order `delta` subtracts.

    Two sides holding different StatVars read "<right> minus <left>". Two
    sides holding one StatVar read "<StatVar>, <right entity> minus <left
    entity>" when exactly one entity column both sides hold
    (`CompareFacts.left_entities`, `right_entities`) holds a different
    value on each, and "difference in <StatVar>" otherwise. A side names its
    first StatVar and how many more it measured (`_side_refs`).
    """
    left = _side_refs(result.left_stat_vars)
    right = _side_refs(result.right_stat_vars)
    if result.left_stat_vars != result.right_stat_vars:
        return f"{right} minus {left}"
    differing = [
        column
        for column, value in result.left_entities.items()
        if result.right_entities.get(column, value) != value
    ]
    if len(differing) == 1:
        (column,) = differing
        return (
            f"{left}, {truncate_value(result.right_entities[column])} minus "
            f"{truncate_value(result.left_entities[column])}"
        )
    return f"difference in {left}"


def render_compare_digest(
    *,
    result: CompareFacts,
    op_spec_hash: str,
    content_version: str,
    execution_identity: ExecutionIdentity,
    derived: Mapping[str, object] | None = None,
    exclusions: Sequence[Exclusion] = (),
    time: Time | None = None,
) -> str:
    """Render `result` (the facts of `analysis_mcp.engine.compare.run_compare`) into
    the digest text a successful `compare` tool call returns. The label line
    names the result's StatVar by what it subtracts (`_difference_label`)."""
    header = (
        f"compare: {len(result.intersected_periods)} intersected period(s) "
        f"(left offered {len(result.left_periods)}, right offered {len(result.right_periods)})"
    )
    label_line = f"stat_var {result.stat_var}: {_difference_label(result)}"

    # Magnitude, not positional: `period_comparisons` is ordered
    # chronologically, not by the caller's own criterion. A magnitude cap
    # drops the smallest-delta periods, which are often the direct answer
    # to "in which years did these two match?"
    shown = capped(result.period_comparisons, magnitude=lambda comparison: comparison.delta)
    lines = [_comparison_line(comparison) for comparison in shown]
    comparisons_block = "\n".join([f"{len(result.period_comparisons)} compared period(s):", *lines])
    comparisons_block += _truncation_note(
        shown, len(result.period_comparisons), selection=_MAGNITUDE_SELECTION
    )

    coverage_line = (
        f"period overlap: left {result.left_coverage_fraction:.1%}, "
        f"right {result.right_coverage_fraction:.1%} (share of each side's own offered "
        "periods that the other side also covers)"
    )
    unit_line = (
        f"left unit: {_unit_text(result.left_resolved_unit)}, "
        f"right unit: {_unit_text(result.right_resolved_unit)}"
    )

    sections = [header, label_line, comparisons_block, coverage_line, unit_line]
    if result.warnings:
        sections.append(
            "\n".join(
                ["warnings:", *(f"  - {truncate_value(warning)}" for warning in result.warnings)]
            )
        )
    sections.extend(_artifact_sections(exclusions=exclusions, time_line=_time_line(time)))
    derived_note = _derived_note(derived)
    if derived_note is not None:
        sections.append(derived_note)
    sections.append(
        _identity_block(
            op_spec_hash=op_spec_hash,
            content_version=content_version,
            execution_identity=execution_identity,
        )
    )
    return "\n\n".join(sections)


def _sum_line(entry: PeriodSum) -> str:
    return (
        f"  {entry.time_period}: {_format_number(entry.left_value)} + "
        f"{_format_number(entry.right_value)} = {_format_number(entry.value)}"
    )


def _left_out_block(result: SumOfFacts) -> str:
    """Each period the sum leaves out, the one side reporting it, named by its
    StatVar, and that side's value there, capped at `LIST_CAP` in period order."""
    left_out = result.left_out
    if not left_out:
        return "left out of the sum: none; both sides report every period"
    measured = {"left": result.left_stat_vars, "right": result.right_stat_vars}
    shown = left_out[:LIST_CAP]
    lines = [
        f"  {entry.time_period}: not in the sum; only the {entry.side} side "
        f"({_side_refs(measured[entry.side])}) reports it: {_format_number(entry.value)}"
        for entry in shown
    ]
    block = "\n".join([f"{len(left_out)} period(s) left out of the sum:", *lines])
    if len(left_out) > len(shown):
        block += f"\n  ... {len(left_out) - len(shown)} more left out, in period order"
    return block


def _side_refs(refs: Sequence[str]) -> str:
    """A side's first StatVar ref, and how many more it measured."""
    if not refs:
        return "no StatVar"
    more = f" and {len(refs) - 1} more" if len(refs) > 1 else ""
    return f"{truncate_value(refs[0])}{more}"


def _period_range_text(periods: Sequence[str]) -> str:
    """`first to last` of sorted `periods`, or the one period alone."""
    return periods[0] if len(periods) == 1 else f"{periods[0]} to {periods[-1]}"


def render_sum_of_digest(
    *,
    result: SumOfFacts,
    op_spec_hash: str,
    content_version: str,
    execution_identity: ExecutionIdentity,
    derived: Mapping[str, object] | None = None,
    exclusions: Sequence[Exclusion] = (),
    time: Time | None = None,
) -> str:
    """Render `result` (the facts of `analysis_mcp.engine.sum_of.run_sum_of`) into
    the digest text a successful `sum_of` tool call returns. Each period one
    side alone reports, which the sum leaves out, is named with that side
    and its value there, straight after the label, so a reader keeping only
    the digest's start still has it. The coverage line states the summed
    span and each side's offered span and coverage."""
    header = (
        f"sum_of: {len(result.intersected_periods)} period(s) both sides report "
        f"(left offered {len(result.left_periods)}, right offered {len(result.right_periods)})"
    )
    label_line = (
        f"stat_var {result.stat_var}: sum of {_side_refs(result.left_stat_vars)} and "
        f"{_side_refs(result.right_stat_vars)}"
    )

    # Magnitude, not positional, as `render_compare_digest` caps its periods.
    shown = capped(result.period_sums, magnitude=lambda entry: entry.value)
    sums_block = "\n".join(
        [f"{len(result.period_sums)} summed period(s):", *(_sum_line(entry) for entry in shown)]
    )
    sums_block += _truncation_note(shown, len(result.period_sums), selection=_MAGNITUDE_SELECTION)

    coverage_line = (
        f"sum over {_period_range_text(result.intersected_periods)}, the periods both sides "
        f"report (left offered {_period_range_text(result.left_periods)}, right "
        f"{_period_range_text(result.right_periods)}); coverage left "
        f"{result.left_coverage_fraction:.1%}, right {result.right_coverage_fraction:.1%}"
    )

    sections = [header, label_line, _left_out_block(result)]
    sections += [sums_block, coverage_line, f"unit: {_unit_text(result.unit)}"]
    if result.warnings:
        sections.append(
            "\n".join(
                ["warnings:", *(f"  - {truncate_value(warning)}" for warning in result.warnings)]
            )
        )
    sections.extend(_artifact_sections(exclusions=exclusions, time_line=_time_line(time)))
    derived_note = _derived_note(derived)
    if derived_note is not None:
        sections.append(derived_note)
    sections.append(
        _identity_block(
            op_spec_hash=op_spec_hash,
            content_version=content_version,
            execution_identity=execution_identity,
        )
    )
    return "\n\n".join(sections)


def _rank_line(rank: EntityRank, *, private: frozenset[tuple[str, str]]) -> str:
    key_text = _key_text(rank.entity_key)
    tag = " (private)" if any(pair in private for pair in rank.entity_key) else ""
    return (
        f"  #{rank.rank} {key_text}{tag}: {_format_number(rank.value)} "
        f"[{_rows(rank.row_count)}, coverage={rank.coverage_fraction:.0%}]"
    )


def _rank_block(ranks: Sequence[EntityRank], *, private: frozenset[tuple[str, str]]) -> str:
    """`ranks`, already in rank order, as a `_rank_line`-per-entity block
    capped at `LIST_CAP` -- shared by the collapse mode's own single block
    and by `_per_period_ranks_block`'s block-per-period call, one period's
    own ranking at a time."""
    shown = ranks[:LIST_CAP]
    lines = [_rank_line(rank, private=private) for rank in shown]
    block = "\n".join([f"{len(ranks)} ranked entit(y/ies):", *lines])
    return block + _truncation_note(shown, len(ranks), selection=_RANK_ORDER_SELECTION)


def _per_period_ranks_block(
    entity_ranks: Sequence[EntityRank], *, private: frozenset[tuple[str, str]]
) -> str:
    """One block per period, each capped and truncation-noted exactly like
    the collapse mode's single block. `entity_ranks` is already grouped by
    period -- the engine (`engine.rank._rank_per_period`) emits one
    period's own ranking, in rank order, before moving to the next -- so
    consecutive runs sharing a `time_period` are the per-period grouping,
    with no second sort needed here.
    """
    periods: list[str] = []
    grouped: dict[str, list[EntityRank]] = {}
    for rank in entity_ranks:
        if rank.time_period not in grouped:
            periods.append(rank.time_period)
        grouped.setdefault(rank.time_period, []).append(rank)
    blocks = [
        f"{period}:\n  " + _rank_block(grouped[period], private=private).replace("\n", "\n  ")
        for period in periods
    ]
    return "\n".join(blocks)


def render_rank_digest(
    *,
    rank_by: Sequence[str],
    result: RankFacts,
    op_spec_hash: str,
    content_version: str,
    execution_identity: ExecutionIdentity,
    derived: Mapping[str, object] | None = None,
    exclusions: Sequence[Exclusion] = (),
    time: Time | None = None,
) -> str:
    """Render `result` (the facts of `analysis_mcp.engine.rank.run_rank`) into the
    digest text a successful `rank` tool call returns.

    `result.per_period=True` renders one block per period instead of one
    ranking overall -- each entity's own `value` here is still the measured
    value (`EntityRank.value`), not the rank position the output *frame*'s
    own column carries with no `limit`; `_derived_note` states that
    distinction when `derived["ordinal"]` is set.

    `result.limit` adds the kept and ranked counts to the header. The rank
    block lists the whole ranking, so the header's count and each line's
    rank number say which entries the output frame holds.
    """
    reduction_text = result.aggregation if result.aggregation is not None else "(raw value)"
    if result.window_start is None:
        window_text = "(no window)"
    elif result.window_start == result.window_end:
        window_text = result.window_start
    else:
        window_text = f"{result.window_start}..{result.window_end}"
    header = f"rank: {', '.join(rank_by)} by {reduction_text}, window {window_text}"
    if result.per_period:
        header += ", per_period=True"
    if result.limit is not None:
        scope, counted = (" per period", "rows") if result.per_period else ("", "entities")
        header += (
            f", limit {result.limit}{scope} (kept {len(result.kept_ranks)} of "
            f"{len(result.entity_ranks)} ranked {counted})"
        )

    # Positional, not `summary.capped`: `entity_ranks` is already ordered by the
    # caller's own criterion (`ascending`, or its default) within each
    # period, so the head of each block is what a caller asking for "the 20
    # lowest/highest" wants. A magnitude selection would show the opposite
    # end of the ranking from rank 1.
    private = frozenset(result.private_parties)
    ranks_block = (
        _per_period_ranks_block(result.entity_ranks, private=private)
        if result.per_period
        else _rank_block(result.entity_ranks, private=private)
    )

    unit_line = f"unit: {_unit_text(result.resolved_unit)}"
    coverage_line = f"overall coverage: {result.overall_coverage_fraction:.1%}"

    sections = [header, ranks_block, unit_line, coverage_line]
    if result.warnings:
        sections.append(
            "\n".join(
                ["warnings:", *(f"  - {truncate_value(warning)}" for warning in result.warnings)]
            )
        )
    time_line = _rank_time_line(
        time, per_period=result.per_period, periods_in_window=result.periods_in_window
    )
    sections.extend(_artifact_sections(exclusions=exclusions, time_line=time_line))
    derived_note = _derived_note(derived)
    if derived_note is not None:
        sections.append(derived_note)
    sections.append(
        _identity_block(
            op_spec_hash=op_spec_hash,
            content_version=content_version,
            execution_identity=execution_identity,
        )
    )
    return "\n\n".join(sections)


def _percent_text(share: PeriodShare) -> str:
    """`share`'s value as its output row holds it, already in per cent."""
    return f"{share.percent:.3f}%"


def _share_line(share: PeriodShare) -> str:
    return (
        f"  {share.time_period}: {_format_number(share.numerator_value)} / "
        f"{_format_number(share.denominator_value)} = {_percent_text(share)}"
    )


def _first_latest_text(shares: Sequence[PeriodShare]) -> str:
    """The earliest and the latest share of one key's chronological
    `shares`, or the one share when there is one."""
    first, latest = shares[0], shares[-1]
    if first is latest:
        return f"only {first.time_period}: {_percent_text(first)}"
    return (
        f"first {first.time_period}: {_percent_text(first)}; "
        f"latest {latest.time_period}: {_percent_text(latest)}"
    )


def _key_share_block(key: tuple[tuple[str, str], ...], shares: Sequence[PeriodShare]) -> str:
    key_text = _key_text(key)
    lines = [_share_line(share) for share in shares]
    heading = f"{key_text} [{len(shares)} period(s)], {_first_latest_text(shares)}:"
    return "\n".join([heading, *lines])


def render_share_of_digest(
    *,
    result: ShareOfFacts,
    op_spec_hash: str,
    content_version: str,
    execution_identity: ExecutionIdentity,
    derived: Mapping[str, object] | None = None,
    exclusions: Sequence[Exclusion] = (),
    time: Time | None = None,
) -> str:
    """Render `result` (the facts of `analysis_mcp.engine.share_of.run_share_of`)
    into the digest text a successful `share_of` tool call returns.

    Every share prints as its output row holds it, in per cent, which the
    header names. An un-keyed call states its first and latest share on the
    line after the header, ahead of everything a reader keeping only the
    digest's start could lose, then renders one flat, magnitude-capped list
    of period shares. A keyed call (`numerator_by` non-empty) instead
    renders one block per key, its heading stating that key's first and
    latest share -- `result.period_shares` is already grouped by key, then
    by period, in the order `run_share_of` built it (sorted keys, sorted
    periods within each), so grouping here needs no re-sort. The key list
    itself is magnitude-capped by each key's own largest-magnitude share,
    the same selection rule every other renderer's own grouped list uses.
    """
    unit_text = _unit_text(PERCENT_UNIT)
    keyed = any(share.key for share in result.period_shares)
    lead: list[str] = []
    if keyed:
        grouped: dict[tuple[tuple[str, str], ...], list[PeriodShare]] = {}
        for share in result.period_shares:
            grouped.setdefault(share.key, []).append(share)
        keys = list(grouped)
        shown_keys = capped(
            keys, magnitude=lambda key: max(abs(share.percent) for share in grouped[key])
        )
        header = (
            f"share_of: {len(keys)} key(s), {len(result.period_shares)} matched "
            f"(key, period) pair(s), in per cent ({unit_text})"
        )
        shares_block = "\n\n".join(_key_share_block(key, grouped[key]) for key in shown_keys)
        shares_block += _truncation_note(shown_keys, len(keys), selection=_MAGNITUDE_SELECTION)
    else:
        header = (
            f"share_of: {len(result.period_shares)} matched period(s), in per cent ({unit_text})"
        )
        lead.append(_first_latest_text(result.period_shares))
        # Magnitude, not positional: `period_shares` is ordered
        # chronologically. A magnitude cap hides periods with the smallest
        # share, often the direct answer to "in which periods was this
        # share near zero?"
        shown = capped(result.period_shares, magnitude=lambda share: share.percent)
        lines = [_share_line(share) for share in shown]
        shares_block = "\n".join([f"{len(result.period_shares)} period share(s):", *lines])
        shares_block += _truncation_note(
            shown, len(result.period_shares), selection=_MAGNITUDE_SELECTION
        )

    label_line = (
        f"stat_var {result.stat_var}: "
        f"{_side_text(result.numerator_stat_var, result.numerator_resolved_unit)} as a share of "
        f"{_side_text(result.denominator_stat_var, result.denominator_resolved_unit)}"
    )
    coverage_line = _coverage_line(result, keyed=keyed)

    # What the share leaves out comes straight after the header, the first
    # and latest share and the label, as `render_quotient_of_digest` places
    # it, so a reader keeping only the digest's start still has it.
    sections = [
        header,
        *lead,
        label_line,
        *_two_sided_pooled_lines(
            result,
            keyed_roles=frozenset(role for share in result.period_shares for role, _ in share.key),
        ),
        *_periods_left_out_lines(result),
        *_dropped_keys_lines(
            (
                (_NO_DENOMINATOR_MATCH, result.unmatched_keys),
                (_NO_SHARED_PERIOD, result.keys_without_shared_period),
            )
        ),
        shares_block,
        coverage_line,
    ]
    if result.warnings:
        sections.append(
            "\n".join(
                ["warnings:", *(f"  - {truncate_value(warning)}" for warning in result.warnings)]
            )
        )
    sections.extend(
        _artifact_sections(
            exclusions=[
                exclusion for exclusion in exclusions if not _dropped_line_covers(exclusion)
            ],
            time_line=_time_line(time),
        )
    )
    derived_note = _derived_note(derived)
    if derived_note is not None:
        sections.append(derived_note)
    sections.append(
        _identity_block(
            op_spec_hash=op_spec_hash,
            content_version=content_version,
            execution_identity=execution_identity,
        )
    )
    return "\n\n".join(sections)


# How many periods or keys one of a quotient's left-out lines names before it
# gives the count alone.
_LEFT_OUT_NAME_CAP = 5


def _dropped_line_covers(exclusion: Exclusion) -> bool:
    """Whether a `share_of` or `quotient_of` digest's own left-out lines
    state all `exclusion` holds, so its exclusions block leaves the record
    out. The dropped-keys line names every unmatched key. The
    zero-denominator line gives cells and periods and no key, so it covers a
    `zero_denominator` record only when that record names no key either; a
    `share_of` call records none, since it refuses a zero denominator."""
    return exclusion.reason == "unmatched_key" or (
        exclusion.reason == "zero_denominator" and not any(exclusion.entities.values())
    )


def _named(items: Sequence[str]) -> str:
    """Up to `_LEFT_OUT_NAME_CAP` of `items`, then how many there are in all."""
    text = ", ".join(items[:_LEFT_OUT_NAME_CAP])
    return f"{text} ({len(items)} total)" if len(items) > _LEFT_OUT_NAME_CAP else text


def _keys_named(keys: Sequence[KeyPairs]) -> str:
    return _named([_key_text(key) for key in keys])


def _quotient_line(entry: PeriodQuotient) -> str:
    label = f"{_key_text(entry.key)}, {entry.time_period}" if entry.key else entry.time_period
    return (
        f"  {label}: {_format_number(entry.numerator_value)} / "
        f"{_format_number(entry.denominator_value)} = {_format_number(entry.quotient)}"
    )


def _periods_left_out_lines(result: QuotientOfFacts | ShareOfFacts) -> list[str]:
    """A line for the periods one side reports and the other reports under
    no key, and one for the (key, period) pairs one side alone reports under
    a key that has rows, when there are any. A key's one-sided period goes
    to the pairs line when the other side reports that period under another
    key, and to the periods line otherwise, so each is named once."""
    lines = []
    sides = (
        ("numerator", result.numerator_periods, result.numerator_only_periods),
        ("denominator", result.denominator_periods, result.denominator_only_periods),
    )
    one_sided = []
    one_sided_pairs = []
    for (side, own, by_key), (_, other, _) in zip(sides, reversed(sides), strict=True):
        other_reports = set(other)
        unmatched = [period for period in own if period not in other_reports]
        if unmatched:
            one_sided.append(f"only the {side} reports {_named(unmatched)}")
        pairs = [
            f"{_key_text(key)} {_period_range_text(shown)} ({len(shown)})"
            for key, periods in by_key
            if (shown := [period for period in periods if period in other_reports])
        ]
        if pairs:
            one_sided_pairs.append(f"only the {side} reports {_named(pairs)}")
    if one_sided:
        lines.append(f"periods left out: {'; '.join(one_sided)}")
    if one_sided_pairs:
        lines.append(f"pairs left out: {'; '.join(one_sided_pairs)}")
    return lines


def _coverage_line(result: QuotientOfFacts | ShareOfFacts, *, keyed: bool) -> str:
    """The periods divided over, each side's offered periods, and each
    side's coverage. A keyed call's coverage counts cells, so a period some
    keys lack lowers it while every period still appears in the ranges."""
    coverage_of = "coverage of (key, period) cells:" if keyed else "coverage"
    return (
        f"divided over {_period_range_text(result.matched_periods)}, the periods both sides "
        f"report (numerator offered {_period_range_text(result.numerator_periods)}, denominator "
        f"{_period_range_text(result.denominator_periods)}); {coverage_of} numerator "
        f"{result.numerator_coverage_fraction:.1%}, denominator "
        f"{result.denominator_coverage_fraction:.1%}"
    )


def _dropped_keys_lines(dropped: Sequence[tuple[str, Sequence[KeyPairs]]]) -> list[str]:
    """A line naming the numerator keys a two-sided result holds no row for,
    each `(why, keys)` of `dropped` that has keys, or no line when none
    has."""
    named = [f"{len(keys)} {why} ({_keys_named(keys)})" for why, keys in dropped if keys]
    return [f"numerator keys left out: {'; '.join(named)}"] if named else []


# Why `_dropped_keys_lines` names a key with no denominator group, and one
# sharing no period with its group.
_NO_DENOMINATOR_MATCH = "with no denominator match"
_NO_SHARED_PERIOD = "sharing no period with the denominator"


def _quotient_left_out_lines(result: QuotientOfFacts) -> list[str]:
    """One line for each kind of thing the quotient leaves out, when there is
    any: the periods and pairs `_periods_left_out_lines` names, the
    numerator keys the result holds no row for, by why, and cells whose
    denominator sums to zero."""
    lines = _periods_left_out_lines(result)
    lines += _dropped_keys_lines(
        (
            (_NO_DENOMINATOR_MATCH, result.unmatched_keys),
            (_NO_SHARED_PERIOD, result.keys_without_shared_period),
            ("with a zero denominator in every matched period", result.keys_with_zero_denominator),
        )
    )
    zero_cells = result.zero_denominator_cells
    if zero_cells:
        periods = sorted({cell.time_period for cell in zero_cells})
        lines.append(
            f"zero denominators: {len(zero_cells)} matched cell(s) left out, the denominator "
            f"summing to zero there, in {_named(periods)}"
        )
    return lines


def _side_text(stat_var: str, unit: UnitCell | None) -> str:
    return f"{truncate_value(stat_var)} (unit: {_unit_text(unit)})"


def render_quotient_of_digest(
    *,
    result: QuotientOfFacts,
    op_spec_hash: str,
    content_version: str,
    execution_identity: ExecutionIdentity,
    derived: Mapping[str, object] | None = None,
    exclusions: Sequence[Exclusion] = (),
    time: Time | None = None,
) -> str:
    """Render `result` (the facts of `analysis_mcp.engine.quotient_of.run_quotient_of`)
    into the digest text a successful `quotient_of` tool call returns.

    Every quotient is a plain number in the composed unit, which the header
    and the quotients' heading both name. What the result leaves out comes
    straight after the label (`_quotient_left_out_lines`), so a reader keeping
    only the digest's start still has it. Quotients are one flat list, each
    line led by its key when the call is keyed, capped by magnitude across
    every (key, period), so a keyed call over many keys and periods stays
    within the same bound as an un-keyed one.
    """
    quotients = result.period_quotients
    unit_text = _unit_text(result.unit)
    keys = {entry.key for entry in quotients}
    counted = (
        f"{len(keys)} key(s), {len(quotients)} matched (key, period) pair(s)"
        if any(keys)
        else f"{len(quotients)} matched period(s)"
    )
    header = f"quotient_of: {counted}, in {unit_text}"
    label_line = (
        f"stat_var {result.stat_var}: "
        f"{_side_text(result.numerator_stat_var, result.numerator_resolved_unit)} divided by "
        f"{_side_text(result.denominator_stat_var, result.denominator_resolved_unit)}"
    )

    # Magnitude, not positional, as `render_share_of_digest` caps its shares.
    shown = capped(quotients, magnitude=lambda entry: entry.quotient)
    quotients_block = "\n".join(
        [
            f"{len(quotients)} quotient(s), numerator / denominator, in {unit_text}:",
            *(_quotient_line(entry) for entry in shown),
        ]
    )
    quotients_block += _truncation_note(shown, len(quotients), selection=_MAGNITUDE_SELECTION)

    coverage_line = _coverage_line(result, keyed=any(keys))

    sections = [header, label_line, *_two_sided_pooled_lines(result)]
    sections += _quotient_left_out_lines(result)
    sections += [quotients_block, coverage_line]
    if result.warnings:
        sections.append(
            "\n".join(
                ["warnings:", *(f"  - {truncate_value(warning)}" for warning in result.warnings)]
            )
        )
    sections.extend(
        _artifact_sections(
            exclusions=[
                exclusion for exclusion in exclusions if not _dropped_line_covers(exclusion)
            ],
            time_line=_time_line(time),
        )
    )
    derived_note = _derived_note(derived)
    if derived_note is not None:
        sections.append(derived_note)
    sections.append(
        _identity_block(
            op_spec_hash=op_spec_hash,
            content_version=content_version,
            execution_identity=execution_identity,
        )
    )
    return "\n\n".join(sections)


def _matched_key_line(matched: MatchedKey) -> str:
    key_text = _key_text(matched.key)
    return (
        f"  {key_text} [{matched.time_period}]: left={_format_number(matched.left_value)}, "
        f"right={_format_number(matched.right_value)}"
    )


def render_join_digest(
    *,
    result: JoinFacts,
    op_spec_hash: str,
    content_version: str,
    execution_identity: ExecutionIdentity,
    derived: Mapping[str, object] | None = None,
    exclusions: Sequence[Exclusion] = (),
    time: Time | None = None,
) -> str:
    """Render `result` (the facts of `analysis_mcp.engine.join.run_join`) into the
    digest text a successful `join` tool call returns."""
    join_keys_text = (
        ", ".join(
            left if left == right else f"{left}={right}"
            for left, right in zip(result.join_keys, result.right_key_columns, strict=True)
        )
        or "(none -- time_period only)"
    )
    header = (
        f"join: on {join_keys_text} + time_period, {result.matched_key_count} matched key(s) "
        f"({result.left_dropped_key_count} left-only, {result.right_dropped_key_count} "
        "right-only, dropped)"
    )

    # `JoinFacts` holds the keys `capped` shows, ties broken to the earlier key.
    shown = result.matched_keys
    lines = [_matched_key_line(matched) for matched in shown]
    matches_block = "\n".join([f"{result.matched_key_count} matched key(s):", *lines])
    matches_block += _truncation_note(
        shown, result.matched_key_count, selection=_MAGNITUDE_SELECTION
    )

    left_unit_text = _unit_text(result.left_resolved_unit)
    right_unit_text = _unit_text(result.right_resolved_unit)
    unit_line = f"left unit: {left_unit_text}, right unit: {right_unit_text}"

    sections = [header, matches_block, unit_line]
    if result.warnings:
        sections.append(
            "\n".join(
                ["warnings:", *(f"  - {truncate_value(warning)}" for warning in result.warnings)]
            )
        )
    sections.extend(_artifact_sections(exclusions=exclusions, time_line=_time_line(time)))
    derived_note = _derived_note(derived)
    if derived_note is not None:
        sections.append(derived_note)
    sections.append(
        _identity_block(
            op_spec_hash=op_spec_hash,
            content_version=content_version,
            execution_identity=execution_identity,
        )
    )
    return "\n\n".join(sections)


def _filter_row_line(summary: FilterRowSummary) -> str:
    key_text = _key_text(summary.row_key)
    unit_text = _unit_text(summary.unit)
    return f"  {key_text}: {_format_number(summary.value)} [{unit_text}]"


def render_filter_digest(
    *,
    predicate_count: int,
    result: FilterFacts,
    op_spec_hash: str,
    content_version: str,
    execution_identity: ExecutionIdentity,
    derived: Mapping[str, object] | None = None,
    exclusions: Sequence[Exclusion] = (),
    time: Time | None = None,
) -> str:
    """Render `result` (the facts of `analysis_mcp.engine.filter.run_filter`) into
    the digest text a successful `filter` tool call returns."""
    header = (
        f"filter: {predicate_count} predicate(s) applied, {result.kept_row_count}/"
        f"{result.total_row_count} row(s) kept ({result.dropped_row_count} dropped)"
    )

    # Magnitude, not positional: `kept_row_summaries` is ordered by the
    # frame grammar's grain order, not a criterion the caller stated. A
    # predicate filters by a threshold, not by magnitude, so there is no
    # caller-stated ordering to preserve here. `FilterFacts` holds the rows
    # `capped` shows.
    shown = result.kept_row_summaries
    lines = [_filter_row_line(summary) for summary in shown]
    rows_block = "\n".join([f"{result.kept_row_summary_count} kept row(s):", *lines])
    rows_block += _truncation_note(
        shown, result.kept_row_summary_count, selection=_MAGNITUDE_SELECTION
    )

    sections = [header, rows_block]
    sections.extend(_artifact_sections(exclusions=exclusions, time_line=_time_line(time)))
    derived_note = _derived_note(derived)
    if derived_note is not None:
        sections.append(derived_note)
    sections.append(
        _identity_block(
            op_spec_hash=op_spec_hash,
            content_version=content_version,
            execution_identity=execution_identity,
        )
    )
    return "\n\n".join(sections)


def _relative_name(offset: int) -> str:
    return "latest" if offset == 0 else f"latest{offset}"


def _recomputed_name(recomputed: RecomputedInput) -> str:
    pair = f"{recomputed.spec_id}, {recomputed.content_version}"
    named = pair if recomputed.verb is None else f"{recomputed.verb} {pair}"
    return f"{named} (alone, latest {recomputed.latest})"


def _rolling_section(rolling: Rolling) -> str:
    """The latest year a call's relative periods were filled from, each period's
    literal, and each derived input computed again under that year."""
    periods = ", ".join(
        f"{_relative_name(period.offset)} is {period.period}" for period in rolling.periods
    )
    text = (
        f"Rolling: the latest year is {rolling.latest}, the newest year at least half of "
        f"the series counted hold; {periods}."
    )
    if rolling.recomputed:
        inputs = "; ".join(_recomputed_name(recomputed) for recomputed in rolling.recomputed)
        text += (
            f" This call computed its rolling inputs again under the latest year "
            f"{rolling.latest}, so their values here can differ from their own results: "
            f"{inputs}."
        )
    return text


def render_summary_digest(summary: VerbSummary) -> str:
    """The digest of a verb call the runner `summary` states, rendered by its
    verb's `render_*_digest` from the summary's facts, canonical parameters,
    identity, derived marker, exclusions and time, with the latest year its
    relative periods were filled from (`_rolling_section`) before the identity."""
    parameters = summary.parameters
    result = summary.result
    render: Callable[..., str]
    own: dict[str, Any] = {}
    match result:
        case AggregateFacts():
            render = render_aggregate_digest
            own = {
                "group_by": parameters.get("group_by", ()),
                "aggregation": parameters["aggregation"],
            }
        case GrowthFacts():
            render = render_growth_digest
            own = {
                "window_start": parameters["window_start"],
                "window_end": parameters["window_end"],
                "entity_by": parameters.get("entity_by", ()),
            }
        case RankFacts():
            render = render_rank_digest
            own = {"rank_by": parameters.get("rank_by", ())}
        case FilterFacts():
            render = render_filter_digest
            own = {"predicate_count": len(parameters.get("predicates", ()))}
        case CompareFacts():
            render = render_compare_digest
        case SumOfFacts():
            render = render_sum_of_digest
        case ShareOfFacts():
            render = render_share_of_digest
        case QuotientOfFacts():
            render = render_quotient_of_digest
        case JoinFacts():
            render = render_join_digest
        case _:
            raise TypeError(f"no digest renders a {type(result).__name__}")
    digest = render(
        result=result,
        **own,
        op_spec_hash=summary.spec_id,
        content_version=summary.content_version,
        execution_identity=summary.execution_identity,
        derived=summary.derived,
        exclusions=summary.exclusions,
        time=summary.time,
    )
    if summary.rolling is None:
        return digest
    # Before the identity block, the one section every digest ends with.
    head, _, identity = digest.rpartition("\n\n")
    return "\n\n".join([head, _rolling_section(summary.rolling), identity])
