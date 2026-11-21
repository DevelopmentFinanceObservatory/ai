"""The signature and docstring of the `join` tool. `verbs.join` registers them
and computes each call on dfo-apis' analysis runner, so this body never runs."""

from __future__ import annotations

from typing import Annotated

from fastmcp.tools import ToolResult
from pydantic import Field

from analysis_mcp.op_spec import FrameRefArgument, PinnedUnit


async def join(
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
        FrameRefArgument,
        Field(
            description=(
                "The right side's frame_ref pair -- required; join always needs two frames. "
                "A bare spec_id also works: the server fetches the data itself."
            )
        ),
    ],
    join_keys: Annotated[
        tuple[str, ...],
        Field(
            default=(),
            description=(
                "Role columns to join on, beyond time_period (always matched implicitly). "
                "Empty auto-detects every role column both frames declare in common."
            ),
        ),
    ] = (),
    right_keys: Annotated[
        dict[str, str] | None,
        Field(
            default=None,
            description=(
                "{join_keys column: right frame's column holding the same entities}, "
                "pairing the two sides by name, so Kenya under observationAbout on the left "
                "matches Kenya under destination on the right. Every join_keys column must "
                "be a key here. The output keeps the left names. Omit (the default) to match "
                "each join_keys column under its own name on both sides."
            ),
        ),
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
) -> ToolResult:
    """Inner-join `frame_ref` and `right_frame_ref` on `join_keys` (or every
    shared role column, auto-detected) plus `time_period`.

    `right_keys` pairs a left column with a right column of another name
    holding the same entities, as `share_of`'s `denominator_by` does:
    `join_keys=["observationAbout"]`, `right_keys={"observationAbout":
    "destination"}` joins a WHO series to a CRS series where both name
    Kenya. Values match by equality, so Kenya against Uganda matches
    nothing and both sides' keys are dropped. Auto-detection pairs
    same-name columns only.

    A joined pair of rows never collapses into one row: the result frame is
    long-format, one row per matched key per side, `stat_var` distinguishing
    which side a row's own value came from -- `Frame` allows exactly one
    `value` column, so this is the only representable shape. A key present
    on one side and absent on the other is dropped (inner join); both
    sides' own dropped counts are recorded in the trace and surfaced as a
    digest warning, never a fabricated null for the missing side.

    Auto-detection picks up any entity-identifying role column (a reporting
    country, say) both frames declare, so joining two different entities'
    own series of the same variable auto-detects that column as a join key,
    every row's key value differs between the two sides, and the join
    matches zero rows on both sides. Use `compare` instead to combine one
    variable across two entities; `join` is for combining two different
    variables for one entity.

    Args:
        frame_ref: The left side's frame_ref.
        right_frame_ref: The right side's frame_ref.
        join_keys: Role columns to match on, beyond time_period; empty auto-detects.
        right_keys: {join_keys column: right column}, pairing by name.
        left_unit: Pins the left side's unit.
        right_unit: Pins the right side's unit.

    Returns:
        A digest: each matched key's left/right values, both sides' units,
        the dropped-key counts, and the result's {spec_id, content_version}
        pair and identity fields. On refusal, a typed error with a
        machine-readable fix.
    """
    raise NotImplementedError(f"{__name__} is a signature; verbs.join runs the tool")
