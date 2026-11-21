"""The signature and docstring of the `filter` tool. `verbs.filter_verb` registers them
and computes each call on dfo-apis' analysis runner, so this body never runs."""

from __future__ import annotations

from typing import Annotated

from common.rolling import ANCHOR_KEY_PATTERN
from fastmcp.tools import ToolResult
from pydantic import Field

from analysis_mcp.op_spec import (
    CategoryMatchPredicate,
    FilterRolling,
    FrameRefArgument,
    OrderedThresholdPredicate,
    PlaceLevelPredicate,
)


async def filter_verb(
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
    predicates: Annotated[
        tuple[OrderedThresholdPredicate | CategoryMatchPredicate | PlaceLevelPredicate, ...],
        Field(
            default=(),
            description=(
                "Row-value predicates, applied as a conjunction (a row survives only if it "
                "satisfies every predicate). The closed vocabulary: an ordered_threshold "
                "(column/op/value, value a number against value or a period string against "
                "time_period; refused on time_period when that holds a span such as "
                "2018/2022), a category_match (column/values), or a place_level "
                "(column/level), which keeps a row whose place-type column, such as "
                "observationAbout_type, names a place at that level, such as Country, to keep "
                "one level of place. A role, a window, or a facet binding is out of this "
                "grammar: filter narrows by value, and scope is narrowed in get_data's "
                "scope argument."
            ),
        ),
    ] = (),
    rolling: Annotated[
        FilterRolling | None,
        Field(
            default=None,
            description=(
                "The populations the latest period is taken over, {population: [{spec_id, "
                "bindings}]}: one entry for each root dataset of the question, bindings "
                "narrowing its series as {column: [values]}. Required when a predicate "
                "carries rolling and refused otherwise. A threshold on time_period that "
                "carries rolling compares against the literal value, as a fixed threshold "
                "does, and the result's {spec_id, content_version} pair differs from the "
                "fixed call's. Such a call runs only when the record anchor_key names covers "
                "it, and refuses rolling_anchor_unresolved otherwise."
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
    """Keep only the rows of `frame_ref` satisfying every predicate in `predicates`.

    Use it to keep the rows of a frame that pass a condition.

    Row-value predicates only. An `ordered_threshold` compares a column's
    own ordered cell against a threshold, numerically against `value`, or
    lexically against `time_period`'s own period string at any grain
    (`YYYY`, `YYYY-MM`, `YYYY-MM-DD`). A `category_match` tests a column's
    own string cell against a closed set of values. This includes the
    `unit` column itself. A `place_level` reads a place's types, such as
    a frame's `observationAbout_type` column,
    by `common.place_type.levels_of`: a frame that names no place holds
    every level of place, and `place_level` on it with `"Country"` keeps
    the countries. A `category_match` on `unit` is an ordinary way
    to narrow a heterogeneous-unit frame down to one unit, since filter
    inspects each row in isolation and never combines two rows' own values
    (unlike `aggregate`/`compare`/`growth`/`rank`/`share_of`, filter resolves
    no unit or facet before running). An empty result is valid, and the
    frame grammar accepts a columns-only empty frame. `frame_ref` may
    itself name a prior `aggregate`, `rank` or `growth` result whose
    `time_period` is one span such as `2018/2022`. An `ordered_threshold` on
    `time_period` refuses over it (`filter_predicate_type_mismatch`, `actual_type` `interval`),
    since the span's string says nothing about the periods it covers. A
    `category_match` on the exact span string keeps its rows, and a predicate
    on any other column filters as usual.

    A `time_period` threshold's value may be {anchor: 'latest', offset: n},
    which follows the data as growth's window years do.

    Args:
        frame_ref: The {spec_id, content_version} pair naming the input frame.
        predicates: Row-value predicates, applied as a conjunction. Empty keeps every row.
        rolling: The populations the latest period is taken over, beside a rolling predicate.
        anchor_key: The handle of the stored latest period, set on a rolling call.

    Returns:
        A digest: how many rows were kept versus dropped, a capped preview
        of the kept rows, and the result's {spec_id, content_version} pair
        and identity fields. On refusal, a typed error with a
        machine-readable fix.
    """
    raise NotImplementedError(f"{__name__} is a signature; verbs.filter_verb runs the tool")
