"""The signature and docstring of the `share_of` tool. `verbs.share_of` registers them
and computes each call on dfo-apis' analysis runner, so this body never runs."""

from __future__ import annotations

from typing import Annotated, Literal

from fastmcp.tools import ToolResult
from pydantic import Field

from analysis_mcp.op_spec import FrameRefArgument, PinnedUnit


async def share_of(
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
                "The denominator's frame_ref. Omit to compute a share within two slices of "
                "the same frame_ref, narrowed by numerator_entity/denominator_entity instead. "
                "A bare spec_id also works: the server fetches the data itself."
            ),
        ),
    ] = None,
    numerator_entity: Annotated[
        dict[str, str] | None,
        Field(
            default=None,
            description=(
                "{role_column: value} narrowing the numerator's own rows before summing "
                "each period. Empty sums every row on this side per period except group "
                "totals, residuals included; a whole total (party_whole true) of a class no "
                "member holds stays, and one that may hold a member refuses. A filter to member_kinds is for comparing or "
                "ranking members; before a sum it leaves out the residuals. For a total "
                'across a whole side, search naming the population ("all donors") with no '
                "operation over that side. Role columns only -- stat_var and "
                "time_period are not accepted selectors."
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
                "Role column(s), on the numerator's own frame, to key the output by -- "
                "one row per (key, period) instead of one row per period. These become "
                "the output frame's own role columns. Empty (the default) computes "
                "a single figure, with no role columns."
            ),
        ),
    ] = None,
    denominator_by: Annotated[
        dict[str, str] | None,
        Field(
            default=None,
            description=(
                "{numerator_by column: denominator column carrying that same role}, "
                "pairing the two sides by name, never by position. Every numerator_by "
                "column must be a key here (a partial mapping refuses). Omit (the "
                "default) for a pooled denominator: one total per period, shared by "
                "every numerator key."
            ),
        ),
    ] = None,
    operation: Annotated[
        Literal["share", "quotient"],
        Field(
            description=(
                "share (the default): the numerator as a per cent of its whole, in one unit. "
                "quotient: one measure per unit of another in two different units, such as GDP "
                "per person."
            )
        ),
    ] = "share",
) -> ToolResult:
    """Divide a numerator by a denominator, per matched (key, period).

    `operation` picks the form. `share` (the default) gives the numerator's
    share of its whole in per cent, unit `Percent` (5.3 means 5.3%), the unit
    a published "% of GDP" series carries. `quotient` divides one measure by
    another in different units, such as GDP by population; the result's unit
    is "<numerator's unit> per <denominator's unit>", and a side that is a
    share, a growth rate or a quotient already refuses
    (`quotient_of_side_unit_unsupported`). Either way the result is one series
    under its own stat_var.

    Each side is narrowed by its own `entity` binding, then every remaining
    row sharing both `numerator_by`'s key values and a `time_period` is
    summed. `numerator_by` left empty (the default) gives one row per period,
    no role columns. `denominator_by` pairs each `numerator_by` column with
    the denominator's column carrying that role, by name; left empty, the
    denominator is pooled (one total per period, shared by every numerator
    key). A non-empty `denominator_by` must name every `numerator_by`
    column, and a share refuses a key outside `numerator_by`
    (`op_spec_invalid`). A numerator key with no denominator match, a key
    sharing no period, and a period one side holds alone are left out and
    recorded, never counted as zero; the digest names them. The call refuses
    only when nothing is left to divide (`share_of_key_unmatched`,
    `share_of_grain_mismatch`, `quotient_of_no_key_matched`,
    `comparison_periods_disjoint`). A period whose denominator sums to zero
    refuses a share (`share_of_zero_denominator`, retryable with another
    denominator source); a quotient leaves that cell out and refuses only
    when every cell divides by zero (`quotient_of_zero_denominator`). A
    genuine overflow (`non_finite_result`) is not retryable.

    Args:
        frame_ref: The numerator's frame_ref.
        denominator_frame_ref: The denominator's frame_ref; omit for a slice of frame_ref.
        numerator_entity: Narrows the numerator's rows before summing.
        denominator_entity: Narrows the denominator's rows before summing.
        numerator_unit: Pins the numerator's unit.
        denominator_unit: Pins the denominator's unit.
        numerator_by: Role column(s) to key the numerator's own rows by.
        denominator_by: {numerator_by column: denominator column}, pairing by name.
        operation: `share` gives a per cent, `quotient` a composed unit
            (`quotient_of_tool.quotient_of`).

    Returns:
        A digest: the first and latest share or quotient, each matched (key,
        period)'s numerator/denominator values and result, each side's
        stat_var and unit, and the result's {spec_id, content_version} pair
        and identity fields. On refusal, a typed error with a
        machine-readable fix.
    """
    raise NotImplementedError(f"{__name__} is a signature; verbs.share_of runs the tool")
