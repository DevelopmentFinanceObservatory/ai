"""The signature and docstring of the `rank` tool. `verbs.rank` registers them
and computes each call on dfo-apis' analysis runner, so this body never runs."""

from __future__ import annotations

from typing import Annotated

from common.rolling import ANCHOR_KEY_PATTERN
from fastmcp.tools import ToolResult
from pydantic import Field

from analysis_mcp.op_spec import (
    AGGREGATIONS,
    ALLOW_SPARSE_BREAKDOWNS_DESCRIPTION,
    MAX_RANK_LIMIT,
    FrameRefArgument,
    PinnedUnit,
    WindowRolling,
    WindowYear,
)


async def rank(
    frame_ref: Annotated[
        FrameRefArgument,
        Field(
            description=(
                "The {spec_id, content_version} pair naming a materialised frame or a "
                "prior verb result. Obtained from get_data's or a prior verb's own digest "
                "-- never invented. "
                "A bare spec_id also works: the server fetches the data itself."
            )
        ),
    ],
    rank_by: Annotated[
        tuple[str, ...],
        Field(
            min_length=1,
            description=(
                "Column names identifying each ranked entity. crsPurpose names a summed "
                "frame's stat_var_crsPurpose."
            ),
        ),
    ],
    partition_by: Annotated[
        tuple[str, ...],
        Field(
            default=(),
            description=(
                "Columns splitting the ranking: each partition is ranked on its own and limit "
                "applies within it, so rank_by=['origin'], partition_by=['destination'], "
                "limit=6 keeps each recipient's top 6 donors. Names no rank_by column."
            ),
        ),
    ] = (),
    aggregation: Annotated[
        str | None,
        Field(
            default=None,
            description=(
                "The reduction function to apply to each entity's own rows before ranking. "
                f"Closed vocabulary: {list(AGGREGATIONS)!r}. Left None, each entity must "
                "already carry exactly one row within the window."
            ),
        ),
    ] = None,
    window_start: Annotated[
        WindowYear | None,
        Field(
            default=None,
            description=(
                "The window's first calendar year. Must be set together with window_end. "
                "Or {anchor: 'latest', offset: -9} to follow the data."
            ),
        ),
    ] = None,
    window_end: Annotated[
        WindowYear | None,
        Field(
            default=None,
            description=(
                "The window's last calendar year. Must be set together with window_start. "
                "Or {anchor: 'latest', offset: 0} to follow the data."
            ),
        ),
    ] = None,
    unit: Annotated[
        PinnedUnit | None,
        Field(
            default=None,
            description=(
                "Pins the unit to rank over, when the input frame carries more than "
                "one: a unit ref, a converted unit as its {currency, prices, base_year, deflation_basis} "
                "object, or a composed unit as its {numerator, denominator} object. "
                "Retry with one of a unit_heterogeneous refusal's fix.units values."
            ),
        ),
    ] = None,
    ascending: Annotated[
        bool, Field(default=False, description="Rank lowest-first instead of highest-first.")
    ] = False,
    per_period: Annotated[
        bool,
        Field(
            default=False,
            description=(
                "False (the default) collapses the window into one ranking, one row per "
                "entity, whose value is the measured value and whose time_period is the "
                "extent of the periods the window holds, such as 2018/2022. True keeps "
                "the window's periods apart: each period is ranked on its own, and the "
                "output carries one row per entity per period whose value is that entity's "
                "rank position in that period, not the measured value, so a rank series can "
                "be grown; with limit set, the value stays the measured value. Set this when the question asks how a position changed over time "
                "rather than who is first; name the entity the question is about with a "
                "filter between this call and a growth(change='absolute') -- the series "
                "carries the whole population, and a growth over it needs one entity's own "
                "position, not everyone's."
            ),
        ),
    ] = False,
    limit: Annotated[
        int | None,
        Field(
            default=None,
            ge=1,
            le=MAX_RANK_LIMIT,
            description=(
                "Keep only the entities ranked at this position or better: 10 keeps the "
                "highest 10, or the lowest 10 with ascending=True. Entities that share the "
                "value at the boundary are all kept, so more than limit can remain. Applies "
                "within each partition under partition_by. Applies within each period when "
                "per_period=True, and the output's value is then "
                "the measured value, not the rank position. Left None, every ranked entity "
                "is kept."
            ),
        ),
    ] = None,
    rolling: Annotated[
        WindowRolling | None,
        Field(
            default=None,
            description=(
                "Marks window_start, window_end, or both as following the latest period of "
                "the data: {window_end: {anchor: 'latest', offset: 0, rule: 'half1'}}, with "
                "offset counting whole years back from the latest. The window's years stay "
                "literal and are what is computed; two rolling bounds must be the same latest "
                "year at their offsets. population is required: one {spec_id, bindings} for "
                "each root dataset of the question, bindings narrowing its series as {column: "
                "[values]}; the latest period is taken over those series. Left None, the "
                "window is fixed. The call ranks over the literal years it is given, as a "
                "fixed call does, and the result's {spec_id, content_version} pair differs "
                "from the fixed call's. Such a call runs only when the record anchor_key "
                "names covers it, and refuses rolling_anchor_unresolved otherwise."
            ),
        ),
    ] = None,
    anchor_key: Annotated[
        str | None,
        Field(
            default=None,
            pattern=ANCHOR_KEY_PATTERN,
            description=(
                "The handle of the stored record of the latest period this call's rolling "
                "periods were written from: rak_ and 32 hex digits. Required when the call "
                "is rolling and refused when it is not. It is no part of the result's identity."
            ),
        ),
    ] = None,
    allow_sparse_breakdowns: Annotated[
        bool, Field(default=False, description=ALLOW_SPARSE_BREAKDOWNS_DESCRIPTION)
    ] = False,
) -> ToolResult:
    """Rank entities identified by `rank_by`, optionally reducing each
    entity's own rows with `aggregation` first.

    Use it to order entities by a value, such as the largest donors in a
    year, and to keep the top or bottom few. With `partition_by`, it ranks
    and applies `limit` within each partition, such as the top 6 donors of
    each recipient.

    Ranks what's present. An entity absent from the window simply isn't
    ranked. Competition ranking ("1224"). Entities sharing a value share a
    rank number, and the rank number never breaks a tie. The *order* two
    tied entities are listed in is broken deterministically by the frame
    grammar's own total ordering over `rank_by`'s values. Refuses
    (`coverage_insufficient`, reusing `aggregate`'s own floor) when the
    grid, every ranked entity times every distinct period any entity
    contributes as reflected in the input data rather than the window's
    full calendar range, falls below the configured coverage floor. A
    period where no entity has any data at all is invisible to this check.

    A window year given as {anchor: 'latest', offset: n} follows the data
    as growth's window years do.

    Collapsed, every output row's `time_period` is the extent of the
    periods the window holds, such as `2018/2022`, or that one period when
    the window holds only one. A frame whose rows carry one span, such as
    a growth result, keeps that span; a window keeps a span row only when
    it covers every year of the span.

    `per_period=True` ranks each of the window's periods on its own instead
    of collapsing them: with no `limit`, the output frame's own `value`
    column carries each entity's rank *position* per period, not its
    measured value, and no
    verb here may read that position as a quantity to ratio (`share_of`) or
    grow at a percentage rate (`growth` with `change="relative"`) or rank
    again -- all three refuse (`derived_input_ordinal`) over this call's own
    output, or over anything derived from it. A top or bottom N per period
    is this call with `limit` set.

    `limit` keeps the entities ranked at that position or better in the
    output frame, every one tied at the boundary included, with their
    measured values in either mode: a limited per-period rank writes no
    ordinal marker. Coverage and the floor check cover every ranked entity,
    and the rows left out are reported as a `beyond_limit` exclusion.

    A get_data frame or bare spec_id holding several StatVars is summed over
    them first unless `rank_by` names `stat_var`, keeping the breakdown
    axes `rank_by` and `partition_by` name. The digest gives the count summed.

    Args:
        frame_ref: The {spec_id, content_version} pair naming the input frame.
        rank_by: Column names identifying each ranked entity.
        partition_by: Columns splitting the ranking; `limit` applies per partition.
        aggregation: The reduction function. None expects one row per entity already.
        window_start: The window's first calendar year, paired with window_end.
        window_end: The window's last calendar year, paired with window_start.
        unit: Pins the unit, to answer a prior unit_heterogeneous refusal.
        ascending: Rank lowest-first instead of highest-first.
        per_period: Rank each period on its own; with no limit, the output's value is the
            rank position.
        limit: Keep only the entities ranked at this position or better.
        rolling: Which window years follow the latest period, and the
            populations that period is taken over.
        anchor_key: The handle of the stored latest period, set on a rolling call.
        allow_sparse_breakdowns: Let empty breakdown cells pass the coverage check.

    Returns:
        A digest: each entity's rank, value, and coverage, the resolved
        unit, overall coverage, the StatVars summed, and the result's {spec_id, content_version}
        pair and identity fields. On refusal, a typed error with a
        machine-readable fix.
    """
    raise NotImplementedError(f"{__name__} is a signature; verbs.rank runs the tool")
