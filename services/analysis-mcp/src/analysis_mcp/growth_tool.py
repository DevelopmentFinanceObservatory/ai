"""The signature and docstring of the `growth` tool. `verbs.growth` registers them
and computes each call on dfo-apis' analysis runner, so this body never runs."""

from __future__ import annotations

from typing import Annotated, Literal

from common.rolling import ANCHOR_KEY_PATTERN
from fastmcp.tools import ToolResult
from pydantic import Field

from analysis_mcp.op_spec import FrameRefArgument, PinnedUnit, WindowRolling, WindowYear


async def growth(
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
    window_start: Annotated[
        WindowYear,
        Field(
            description=(
                "The window's first calendar year, e.g. '2020'. Or {anchor: 'latest', "
                "offset: -9} to follow the data."
            )
        ),
    ],
    window_end: Annotated[
        WindowYear,
        Field(
            description=(
                "The window's last calendar year, e.g. '2023'. Or {anchor: 'latest', "
                "offset: 0} to follow the data."
            )
        ),
    ],
    entity_by: Annotated[
        tuple[str, ...],
        Field(
            default=(),
            description=(
                "Column names identifying each entity whose own growth rate is computed. "
                "Empty treats the whole frame as one entity's series."
            ),
        ),
    ] = (),
    unit: Annotated[
        PinnedUnit | None,
        Field(
            default=None,
            description=(
                "Pins the unit to compute over, when the input frame carries more than "
                "one: a unit ref, a converted unit as its {currency, prices, base_year, deflation_basis} "
                "object, or a composed unit as its {numerator, denominator} object. "
                "Retry with one of a unit_heterogeneous refusal's fix.units values."
            ),
        ),
    ] = None,
    change: Annotated[
        Literal["relative", "absolute"],
        Field(
            default="relative",
            description=(
                "'relative' (the default) computes (value_end - value_start) / value_start; "
                "an entity with a zero or sign-crossing base is excluded, since a percentage "
                "change is undefined there. 'absolute' computes value_end - value_start directly, "
                "keeping the input's own unit; defined over any base, including zero or a "
                "sign crossing."
            ),
        ),
    ] = "relative",
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
                "window is fixed. The call computes over the literal years it is given, as a "
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
) -> ToolResult:
    """Compute each entity's growth across `[window_start, window_end]`.

    Use it for the change between two periods, as a percentage or an amount.

    `change="relative"` (the default) computes
    `growth_rate = (value_end - value_start) / value_start`, the total
    percentage change across the window; a zero or sign-crossing base makes
    that undefined, so such an entity is excluded as
    `zero_or_sign_crossing_base` before any division is attempted.
    `change="absolute"` computes `value_end - value_start` directly, in the
    input's own unit; that arithmetic is defined over any base. Either mode
    excludes an entity whose value overflows as `non_finite`, and the call
    refuses (`non_finite_result`) only when every entity is excluded so.

    Both modes read only each entity's rows in `window_start` and
    `window_end`. Every entity with a row in both is computed, whatever
    years between them it lacks; a warning counts the entities with such
    gaps. An entity missing either endpoint year is left out and listed
    under exclusions as `endpoint_missing`. When no entity has both, the
    call refuses (`growth_gap_year`), counting the entities missing each
    endpoint and suggesting a window at least one entity holds both ends of,
    when one exists.

    A window year given as {anchor: 'latest', offset: n} follows the data:
    the call takes the newest year at least half of the input's series hold
    and moves back -n years. It computes over those literal years, which the
    result names, and a card drawn from the result moves forward when new
    data lands.

    Each output row's `time_period` is the window as a span, such as
    `2020/2023`. A frame whose rows already carry a span, such as an
    `aggregate` or `rank` result over several years, refuses
    (`derived_input_unusable`), because growth reads one row per period.
    Retry over the frame that result was computed from.

    Args:
        frame_ref: The {spec_id, content_version} pair naming the input frame.
        window_start: The window's first calendar year.
        window_end: The window's last calendar year, strictly after window_start.
        entity_by: Column names identifying each entity; empty is one entity.
        unit: Pins the unit, to answer a prior unit_heterogeneous refusal.
        change: 'relative' for a percentage rate, 'absolute' for a difference
            in the input's own unit.
        rolling: Which window years follow the latest period, and the
            populations that period is taken over.
        anchor_key: The handle of the stored latest period, set on a rolling call.

    Returns:
        A digest: each computed entity's growth value and its endpoint
        values, the entities excluded for a missing endpoint, and the
        result's {spec_id, content_version} pair and identity fields. A
        relative growth rate is dimensionless -- `unit` only selected which
        input rows `value_start`/`value_end` were read from, it does not
        denominate the ratio they produced -- so the output frame's own
        `unit` is `None`; an absolute change keeps the input's own unit
        instead. On refusal, a typed error with a machine-readable fix.
    """
    raise NotImplementedError(f"{__name__} is a signature; verbs.growth runs the tool")
