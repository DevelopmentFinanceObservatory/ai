"""The signature and docstring of the `aggregate` tool. `verbs.aggregate` registers them
and computes each call on dfo-apis' analysis runner, so this body never runs."""

from __future__ import annotations

from typing import Annotated

from fastmcp.tools import ToolResult
from pydantic import Field

from analysis_mcp.op_spec import (
    AGGREGATIONS,
    ALLOW_SPARSE_BREAKDOWNS_DESCRIPTION,
    FrameRefArgument,
    PinnedUnit,
)


async def aggregate(
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
    aggregation: Annotated[
        str,
        Field(
            min_length=1,
            max_length=64,
            description=(
                f"The reduction function. Closed vocabulary: {list(AGGREGATIONS)!r}. Under "
                "max or min, each group's digest line carries the period that produced the "
                "value in parentheses after it, and if time_period isn't in group_by, the "
                "output row's own time_period is that period. When several periods tie for a "
                "group's max or min, the earliest of them is reported. Under sum or mean with "
                "time_period outside group_by, every output row's time_period is the span "
                "the call covers, such as 2018/2022."
            ),
        ),
    ],
    group_by: Annotated[
        tuple[str, ...],
        Field(
            default=(),
            description=(
                "Column names to group by before reducing. Empty groups the whole input "
                "frame into one row. crsPurpose names a summed frame's stat_var_crsPurpose."
            ),
        ),
    ] = (),
    unit: Annotated[
        PinnedUnit | None,
        Field(
            default=None,
            description=(
                "Pins the unit to aggregate over, when the input frame carries more than "
                "one: a unit ref, a converted unit as its {currency, prices, base_year, deflation_basis} "
                "object, or a composed unit as its {numerator, denominator} object. "
                "Retry with one of a unit_heterogeneous refusal's fix.units values."
            ),
        ),
    ] = None,
    allow_sparse_breakdowns: Annotated[
        bool, Field(default=False, description=ALLOW_SPARSE_BREAKDOWNS_DESCRIPTION)
    ] = False,
) -> ToolResult:
    """Group a stored frame by `group_by` and reduce each group's `value` with `aggregation`.

    Use it to total or average a frame over one of its dimensions.

    `aggregate` skips absent grid cells, not null values. `Frame.build` cannot
    represent an in-frame null observation today (`value` must be finite,
    `status` is reserved), so "coverage" here means cells present versus
    cells expected for the grouped grain. Both figures land in the trace's
    `applied_policy` on every call. Unit and facet heterogeneity are checked
    before any number is computed. A heterogeneous unit refuses with
    candidates to retry against via `unit`, and a heterogeneous facet
    refuses naming a differently-scoped `frame_ref` to retry with instead. A
    result below the configured coverage floor refuses rather than returning
    a number computed from too sparse a grid. Under sum or mean with
    `time_period` outside `group_by`, every output row's `time_period` is
    one span from the earliest to the latest period the call reduced
    (`2018/2022`), the same on every row.

    A get_data frame or bare spec_id holding several StatVars is summed over
    them first unless `group_by` names `stat_var`, keeping the breakdown
    axes it names. The digest gives the count summed.

    Args:
        frame_ref: The {spec_id, content_version} pair naming the input frame.
        aggregation: The reduction function. See the closed vocabulary above.
        group_by: Column names to group by. Empty reduces the whole frame to one row.
        unit: Pins the unit, to answer a prior unit_heterogeneous refusal.
        allow_sparse_breakdowns: Let empty breakdown cells pass the coverage check.

    Returns:
        A digest: the computed numbers per group, the resolved unit, coverage,
        any warnings, the StatVars summed, and the
        result's {spec_id, content_version} pair and
        identity fields. Under max or min, each group's line also carries the
        period that produced the value, in parentheses after it. The full
        result frame is never in the digest or in structured content.
        Resolve it again by that pair. On refusal, a typed error with a
        machine-readable fix.
    """
    raise NotImplementedError(f"{__name__} is a signature; verbs.aggregate runs the tool")
