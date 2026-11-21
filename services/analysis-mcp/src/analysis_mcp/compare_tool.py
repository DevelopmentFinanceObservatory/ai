"""The signature and docstring of the `compare` tool. `verbs.compare` registers them
and computes each call on dfo-apis' analysis runner, so this body never runs."""

from __future__ import annotations

from typing import Annotated, Literal

from fastmcp.tools import ToolResult
from pydantic import Field

from analysis_mcp.op_spec import FrameRefArgument, PinnedUnit


async def compare(
    frame_ref: Annotated[
        FrameRefArgument,
        Field(
            description=(
                "The left side's {spec_id, content_version} pair, naming a materialised "
                "frame or a prior verb result. Obtained from get_data's or a prior verb's "
                "own digest -- never invented. "
                "A bare spec_id also works: the server fetches the data itself."
            )
        ),
    ],
    right_frame_ref: Annotated[
        FrameRefArgument | None,
        Field(
            default=None,
            description=(
                "The right side's frame_ref pair. Omit to compare two slices of the same "
                "frame_ref, narrowed by left_entity/right_entity instead. "
                "A bare spec_id also works: the server fetches the data itself."
            ),
        ),
    ] = None,
    left_entity: Annotated[
        dict[str, str] | None,
        Field(
            default=None,
            description=(
                "{role_column: value} narrowing the left side to one entity's own time "
                "series. Empty when frame_ref already names exactly one entity. Role "
                "columns only -- stat_var and time_period are not accepted selectors."
            ),
        ),
    ] = None,
    right_entity: Annotated[
        dict[str, str] | None,
        Field(default=None, description="Same as left_entity, for the right side."),
    ] = None,
    left_unit: Annotated[
        PinnedUnit | None,
        Field(
            default=None,
            description=(
                "Pins the left side's unit: one of a unit_heterogeneous refusal's fix.units values, a "
                "unit ref, a converted unit's {currency, prices, base_year, deflation_basis} object, or a composed "
                "unit's {numerator, denominator} object."
            ),
        ),
    ] = None,
    right_unit: Annotated[
        PinnedUnit | None,
        Field(
            default=None,
            description=(
                "Pins the right side's unit: one of a unit_heterogeneous refusal's fix.units values, a "
                "unit ref, a converted unit's {currency, prices, base_year, deflation_basis} object, or a composed "
                "unit's {numerator, denominator} object."
            ),
        ),
    ] = None,
    operation: Annotated[
        Literal["difference", "sum"],
        Field(
            description=(
                "difference (the default): right minus left at each period. sum: left plus "
                "right at each period, as one series."
            )
        ),
    ] = "difference",
) -> ToolResult:
    """Subtract or add two slices, period by period, over the periods both report.

    Each side is `frame_ref` (or `right_frame_ref`) narrowed by an `entity`
    binding down to one row per `time_period`. `operation` picks the form:
    `difference` (the default) gives `delta`, `right - left`, at each period;
    `sum` adds the two sides. A period only one side reports is left out,
    never counted as zero; the digest names those periods and each side's
    coverage. Periods with no overlap refuse (`comparison_periods_disjoint`),
    naming each side's own available periods. A sum needs both sides in the
    same unit (`cross_frame_unit_mismatch`, `cross_frame_unit_unbound`);
    differing provenance or measurement method is a warning, and the result
    leaves that facet unbound. The result is one series under its own
    stat_var. Use `join` instead to set two different variables side by side
    for one entity.

    Args:
        frame_ref: The left side's frame_ref.
        right_frame_ref: The right side's frame_ref; omit for two slices of frame_ref.
        left_entity: Narrows the left side to one entity's series.
        right_entity: Narrows the right side to one entity's series.
        left_unit: Pins the left side's unit.
        right_unit: Pins the right side's unit.
        operation: `difference` subtracts, `sum` adds (`sum_of_tool.sum_of`).

    Returns:
        A digest: each intersected period's two values and the delta or sum,
        both sides' coverage, and the result's {spec_id, content_version} pair
        and identity fields. On refusal, a typed error with a
        machine-readable fix.
    """
    raise NotImplementedError(f"{__name__} is a signature; verbs.compare runs the tool")
