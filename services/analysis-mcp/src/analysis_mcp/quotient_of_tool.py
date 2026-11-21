"""The signature and docstring of the `quotient_of` tool. `verbs.quotient_of` registers them
and computes each call on dfo-apis' analysis runner, so this body never runs."""

from __future__ import annotations

from typing import Annotated

from fastmcp.tools import ToolResult
from pydantic import Field

from analysis_mcp.op_spec import FrameRefArgument, PinnedUnit


async def quotient_of(
    frame_ref: Annotated[
        FrameRefArgument,
        Field(
            description=(
                "The numerator's {spec_id, content_version} pair, naming a materialised "
                "frame or a prior verb result. Obtained from get_data's or a prior verb's "
                "own digest -- never invented. "
                "A bare spec_id also works: the server fetches the data itself."
            )
        ),
    ],
    denominator_frame_ref: Annotated[
        FrameRefArgument | None,
        Field(
            default=None,
            description=(
                "The denominator's frame_ref. Omit to divide two slices of the same "
                "frame_ref, narrowed by numerator_entity/denominator_entity instead. "
                "A bare spec_id also works: the server fetches the data itself."
            ),
        ),
    ] = None,
    numerator_entity: Annotated[
        dict[str, str] | None,
        Field(
            default=None,
            description=(
                "{role_column: value} narrowing the numerator's rows before summing each "
                "period. Empty sums every row on this side per period. Role columns only "
                "-- stat_var and time_period are not accepted selectors."
            ),
        ),
    ] = None,
    denominator_entity: Annotated[
        dict[str, str] | None,
        Field(default=None, description="Same as numerator_entity, for the denominator side."),
    ] = None,
    numerator_unit: Annotated[
        PinnedUnit | None,
        Field(
            default=None,
            description=(
                "Pins the numerator's unit: one of a unit_heterogeneous refusal's fix.units values, a "
                "unit ref, a converted unit's {currency, prices, base_year, deflation_basis} object, or a composed "
                "unit's {numerator, denominator} object."
            ),
        ),
    ] = None,
    denominator_unit: Annotated[
        PinnedUnit | None,
        Field(
            default=None,
            description=(
                "Pins the denominator's unit: one of a unit_heterogeneous refusal's fix.units values, a "
                "unit ref, a converted unit's {currency, prices, base_year, deflation_basis} object, or a composed "
                "unit's {numerator, denominator} object."
            ),
        ),
    ] = None,
    numerator_by: Annotated[
        list[str] | None,
        Field(
            default=None,
            description=(
                "Role column(s), on the numerator's frame, to key the output by -- one row "
                "per (key, period) instead of one row per period. These become the output "
                "frame's role columns. Empty (the default) computes a single series."
            ),
        ),
    ] = None,
    denominator_by: Annotated[
        dict[str, str] | None,
        Field(
            default=None,
            description=(
                "{numerator_by column: denominator column holding the same entities}, "
                "pairing the two sides by name, so each key divides by its own counterpart. "
                "Every numerator_by column must be a key here. Omit (the default) to divide "
                "every numerator key by the denominator's one total per period."
            ),
        ),
    ] = None,
) -> ToolResult:
    """Divide one measure by another, per period both report, such as GDP by population.

    Each side is `frame_ref` (or `denominator_frame_ref`) narrowed by an
    `entity` binding and summed per period. The result's unit is composed
    from the two sides, "<numerator's unit> per <denominator's unit>", and is
    never one side's alone; a side that declares no unit is named by its
    stat_var ("USDollar per Count_Person"). The result is one series under
    its own stat_var. `numerator_by` keys the output, one series per key.
    `denominator_by` pairs each `numerator_by` column with the denominator
    column holding the same entities, by name; without it every key divides
    by the denominator's one total. A period only one side reports, a
    numerator key with no denominator match or no shared period, and a cell
    whose denominator sums to zero are left out, never counted as zero; the
    digest names them. The call refuses when nothing is left to divide
    (`comparison_periods_disjoint`, `quotient_of_no_key_matched`,
    `quotient_of_zero_denominator`) and when a side is a share, a growth
    rate or a quotient already (`quotient_of_side_unit_unsupported`). Use
    `share_of` for a part of its whole in one unit.

    Args:
        frame_ref: The numerator's frame_ref.
        denominator_frame_ref: The denominator's frame_ref; omit for a slice of frame_ref.
        numerator_entity: Narrows the numerator's rows before summing.
        denominator_entity: Narrows the denominator's rows before summing.
        numerator_unit: Pins the numerator's unit.
        denominator_unit: Pins the denominator's unit.
        numerator_by: Role column(s) to key the numerator's rows by.
        denominator_by: {numerator_by column: denominator column}, pairing by name.

    Returns:
        A digest: each matched (key, period)'s two values and quotient in
        the composed unit, each side's stat_var and unit, both coverages,
        what was left out, and the result's {spec_id, content_version} pair
        and identity fields. On refusal, a typed error with a
        machine-readable fix.
    """
    raise NotImplementedError(f"{__name__} is a signature; verbs.quotient_of runs the tool")
