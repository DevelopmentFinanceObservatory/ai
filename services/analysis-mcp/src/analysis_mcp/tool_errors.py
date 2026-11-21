"""Tool-facing refusal grammar, following `data_mcp.tool_errors`'s shape: a
`StrEnum` code, a `StrEnum` retry mode, a frozen kw-only `ToolRefusal` with
one classmethod factory per code, and `to_tool_result` to render one.

The verbs and `resolve_anchor` compute in dfo-apis' analysis runner
(`runner_client`), which refuses with the engine's own codes, retry modes and
fixes, so a runner refusal maps onto this module's fields: `code`, `message`,
`fix`, `runner_retry_mode`, and the `retry_after_s` that `service_unavailable`
and a retryable `upstream_error` set. A runner code this module defines no
factory for passes through as the runner wrote it. `upstream_code` names the
runner's own code under `upstream_error`, which stands for a reply outside the
runner's contract. The paragraphs below say what each code means when a verb's
computation refuses with it, in the runner or, for the store codes, here.

`unit_heterogeneous` and `facet_heterogeneous` carry a disambiguation payload
in `fix["disambiguation"]`: its `candidates`, each an `id` and a `label`.
`unit_heterogeneous` is retryable through whichever unit-pin parameter the
calling verb declares (`fix["parameter"]`); `facet_heterogeneous` has no pin,
so its payload names the axis and candidates for a caller to act on by
choosing a differently-scoped `frame_ref`. A unit refusal's `fix["units"]`
lists each candidate in the form the pin parameter accepts: a unit ref as
its string, a converted unit (a `UnitRequest` cell) as its request object,
a composed unit (a `ComposedUnit` cell) as its `{numerator, denominator}`
object. A converted candidate's label names its price basis and currency, a
composed candidate's label reads "<numerator> per <denominator>", and the
`id` of either is the object to pin. `role_ambiguous` names whichever
column-list parameter the calling verb actually has in `fix["parameter"]`,
never a hardcoded name borrowed from `aggregate`. `place_level_side_unnamed`
is the same code for the `place_level` argument, whose column the call's keys
did not settle. `fix["available"]` lists the frame's `{role}_type` columns, and
the retry goes through `filter` with the argument in `fix["retry_without"]` left
out, since sending it again repeats the refused step.

`comparison_periods_disjoint` fires when compare's two sides share no
overlapping period; `growth_gap_year` fires when no entity has a row in
both of growth's endpoint years. `cross_frame_unit_mismatch`/`cross_frame_unit_unbound`/
`cross_frame_facet_mismatch` fire when two independently-resolved sides
(`compare`'s left/right, `share_of`'s numerator/denominator) disagree on
unit or facet -- distinct from `unit_heterogeneous`/`facet_heterogeneous`,
which fire when *one* frame itself carries more than one value.
A `share_of` call in share mode whose sides resolve to two units refuses
`cross_frame_unit_mismatch` with `fix["operation"]` `quotient`, the form that
divides across units (`share_units_differ`).
`cross_frame_facet_mismatch` fires only when the two sides hold the same
StatVar; two different measures each keep their own facet selection.
`cross_frame_unit_unbound` is the narrower case where one side's rows leave
`unit` unbound entirely rather than resolving to a second, differing unit.

`share_of_grain_mismatch` fires when no numerator key shares a period with
its denominator group; `share_of` leaves out and names each key sharing
none, and `fix["key"]` names the call's key when it has one.
`share_of_zero_denominator` is split out from `non_finite_result` because a
different `frame_ref` or entity binding can produce a nonzero sum, so it is
retryable.

`quotient_of` also leaves out a matched cell whose denominator sums to zero,
which `share_of` refuses, so its three codes name the calls that have
nothing left to return or no unit to state.
`quotient_of_no_key_matched` fires when no numerator key has a denominator
group under `denominator_by`; `fix` samples each side's keys under its own
column names. `quotient_of_zero_denominator` fires when the denominator sums
to zero in every matched cell. `quotient_of_side_unit_unsupported` fires
when a side can be no part of a composed unit; `fix["reason"]` is
`dimensionless` (a share or a relative growth rate), `composed` (a quotient
already) or `operation_stat_var` (rows with no unit under a `sum_of`
result's own stat_var, which nothing names). A quotient whose keys pair but
share no period reuses `comparison_periods_disjoint` under the labels
`numerator` and `denominator`.

`composed_unit_sum_unsupported` fires when a verb is asked to add values in
a composed unit, the unit a `quotient_of` result carries: `aggregate` and
`rank` with `aggregation="sum"`, `sum_of` on either side, and `share_of`
on a side whose (key, period) cell holds more than one row. Two figures
per head added together are not a figure per head, so no total is written
under that unit. `fix` names the verb, the composed unit and, for `sum_of`
and `share_of`, the side. A `share_of` side of one row per cell computes,
as does a mean, max or min over the same frame, and one figure
pooled across entities comes from totalling each measure first and dividing
the totals with `quotient_of`. A frame holding more than one unit refuses
`unit_heterogeneous` first.

`share_of_key_unknown` fires when `denominator_by` is non-empty but leaves
out a `numerator_by` column -- a partial mapping has no defined pairing for
the column(s) it omits (a `denominator_by` key `numerator_by` itself
doesn't carry refuses earlier, at the op-spec, as `op_spec_invalid`).
`share_of_key_unmatched` fires when no numerator key has a denominator
group under `denominator_by`'s mapping. A call where some key has one
divides those keys and records the rest as `unmatched_key`, as
`quotient_of` does.

`filter_predicate_type_mismatch` fires when a predicate's column exists but
its declared cell type doesn't admit that predicate kind, or an
`ordered_threshold` operand carries the wrong type; it is retryable.
`filter_unit_conversion_unsupported` fires when a `category_match` on `unit`
meets a frame whose `unit` cells carry a converted or composed unit, which no
unit string matches. Predicates on every other column still apply to that
frame, so it is retryable: without the unit predicate, or after a `get_data`
call that binds one unit, or, for a composed unit, with the unit pinned in the
verb that reads the frame.

`join_output_collision` fires when both sides of a `join` resolve to the
identical `stat_var`, unit, and facets, so a matched key's two rows would
collide on one grain. `join_no_common_key_columns` fires when `join_keys`
is left empty and the two frames declare no role column in common while
both declare some -- distinct from `join_output_collision`, which fires
after a match is found.

`mixed_time_grain` fires when an input's `time_period` column carries more
than one grain, a span beside a point, or two different spans. It is `do_not_retry`,
since the stored frame is the same on every call. `frame_not_held` fires when a
resolved `frame_ref` names a pair the store cannot return -- never
registered, fabricated, or past the store's own retention. `frame_expired`
fires when a pair aged out past an in-memory store's TTL, with a `get_data`
retry only for a Datasets-family (`spec_...`) pair; a durable, bucket-backed
store never raises it, since a pair its lifecycle rule has deleted comes
back as the same `frame_not_held` a pair that was never registered would.
`frame_ref_incomplete` fires when a frame argument names a derived (`ops_`) or
malformed `spec_id` with no `content_version`. The call is retryable with the
full pair. A bare `spec_` id is fetched (`runner_tree.spec_root`).
A frame the runner lost is fetched or computed again before either refusal
reaches a caller (`tool_pipeline.run_on_runner`). When that fails the refusal
keeps its code and retry mode, and its message gains the sentence saying the
server could not rebuild the frame (`not_rebuilt`).
`frame_integrity_failed` fires when a durable store's read-time check finds
the stored object itself is wrong: its bytes don't match what was recorded
at write time, its row count differs from the count recorded beside it,
its envelope names a different reference than the one requested, or, when the recompute check is enabled, a recomputed
content_version disagrees with the value that check compares against. Every
one of those means the identical reference reaches the identical wrong
object, so the fix is a fresh result under a new reference.
`frame_too_large` fires when a step's result is too large to store, so nothing
is stored and narrowing the call's inputs is the fix.

`derived_input_unusable` refuses a derived input a verb cannot read, with a
retry over a different `frame_ref`. `derived_input_unrecoverable` refuses a
derived (`ops_`) input computed from a dataset version dfo-apis does not
hold (`derived_input_root_lost`), so no retry over the same result fixes it.
Running the steps again from the source data does.

`derived_input_ordinal` fires when an input is a per-period `rank`'s result.
Its `value` column holds a rank position, not a measured value, and no verb here may treat a position
as a quantity to ratio, grow at a percentage rate or rank again. Raised by
`share_of` and `quotient_of` on either side (a ratio has no meaning over a
position), by
`growth` when `change="relative"` (a rank moves by places, not by a
percentage of one) and by `rank` (highest first over positions puts the
last-placed entity first). The message names the retry for `growth`
(`change="absolute"`) and for `rank` (the frame the per-period rank read,
with `limit` set on that call); `share_of` and `quotient_of` have none to
name.

`place_levels_mixed` fires when `rank`, an `aggregate` whose `group_by`
leaves a place column out, or either side of a `share_of` reads a frame
carrying that column's `{role}_type` enrichment whose several places
share no level of `common.place_type.PLACE_LEVELS`. The frame names no place, so it holds
the world, regions, World Bank groups, countries and their subdivisions
side by side. A type outside `PLACE_LEVELS`, such as `Place`,
`ProvisionalNode` or `UNGeoRegion`, names no level, so places sharing only
that type are refused, and one place alone passes whatever its type. It is
retryable with `place_level` set, which `rank`, `aggregate` and `share_of` take,
and `fix["place_level"]` names a level some of the places sit at. On a flow
party role (`financingProvider`, `financingRecipient`) the verbs leave out the
parties whose
`{role}_party_kind` cell is `total` and record them as `party_total`
exclusions; the refusal fires there only when several parties include one
with no kind or a cell no kind names, are all totals on a role the verb pools
(a role of totals alone, each `true` in `{role}_party_whole` on every row, is
summed), hold on a pooled role a whole total beside members it may hold (one
of its `{role}_party_class`, or of none, or any member when the total has no
class; a whole total of a class no member has is summed beside them), or sit
in a frame that carries no kind column for their role. On a role the verb
groups by, a whole total that may hold a member is left out as a
`party_total` and one that holds none keeps its own row. `rank`
ranks a role holding only totals, and leaves the residuals of a ranked role
holding parties out as `party_residual` exclusions.

`service_unavailable` fires when the analysis runner is busy (`runner_client`),
when a root dataset of the call is still building (`analysis_pending`), or when
the frame store fails a read (`frame_store_unavailable`). It is the one
`retry_after_delay` code. The same call succeeds once the wait it names has
passed.

`rolling_anchor_unresolved` fires when the latest period a rolling call stands
on is not established. `resolve_anchor` raises it when its populations and
root frames do not fit together or no period is held by half of the series,
and `growth`, `rank` and `filter` raise it for a rolling call whose
`anchor_key` names no stored record, or a record that does not cover the
call. `fix["detail"]` names the check that failed. It is retryable, since a
`resolve_anchor` call over the question's populations and root frames gives
the key and the period to write the literals from. A call stating a relative
year has no key. The runner raises the code from its own latest year, and
`ToolRefusal.latest_year_unresolved` asks for fixed years. `rolling_grain_unsupported`
fires when a population's periods are not calendar years, which the frames
do not change between calls, so it is `do_not_retry`.

`rendering_not_recorded` fires when a `spec_` frame was fetched with an option
the runner does not read a root with, so no runner call reproduces that frame.
It is retryable over a frame fetched without the option. It also fires when a
lost frame has no mint record to state its rendering, and `get_data` names a
frame that has one.

`upstream_error` is a runner refusal under a code this service defines for its
own failures (`runner_client._LOCAL_CODES`), or an answer outside the runner's
contract. `upstream_code` names the
runner's code. It is `retry_after_delay` when the refusal was retryable and
carries a delay, and `do_not_retry` otherwise.

`internal_error` is the registration-site catch-all's code (see
`server.py`): every other code names something a specific, anticipated
failure produced, and `internal_error` names the absence of one.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Any, Self

from common.echoed_text import truncate_value
from common.materialisation import FrameRef
from common.place_type import (
    PARTY_ROLES,
    PLACE_LEVELS,
    WHOLE_CELL,
    WHOLE_SIDE_POPULATIONS,
    PartyKind,
    class_column,
    enrichment_role,
    levels_of,
    whole_column,
)
from common.refusal import internal_error_message, refusal_meta, refusal_text
from fastmcp.tools import ToolResult


class RetryMode(StrEnum):
    """How a caller should react to a `ToolRefusal`, distinct from the code that caused it."""

    CORRECT_AND_RETRY = "correct_and_retry"
    RETRY_AFTER_DELAY = "retry_after_delay"
    DO_NOT_RETRY = "do_not_retry"


class ToolErrorCode(StrEnum):
    """The refusal codes, each mapped to exactly one `RetryMode`."""

    OP_SPEC_INVALID = "op_spec_invalid"
    STAT_VAR_UNRESOLVED = "stat_var_unresolved"
    ROLE_AMBIGUOUS = "role_ambiguous"
    OPERATION_UNCOVERED = "operation_uncovered"
    FRAME_TOO_LARGE = "frame_too_large"
    UNIT_HETEROGENEOUS = "unit_heterogeneous"
    FACET_HETEROGENEOUS = "facet_heterogeneous"
    COVERAGE_INSUFFICIENT = "coverage_insufficient"
    NON_FINITE_RESULT = "non_finite_result"
    FRAME_NOT_HELD = "frame_not_held"
    FRAME_EXPIRED = "frame_expired"
    FRAME_REF_INCOMPLETE = "frame_ref_incomplete"
    MIXED_TIME_GRAIN = "mixed_time_grain"
    FRAME_INTEGRITY_FAILED = "frame_integrity_failed"
    COMPARISON_PERIODS_DISJOINT = "comparison_periods_disjoint"
    GROWTH_GAP_YEAR = "growth_gap_year"
    CROSS_FRAME_UNIT_MISMATCH = "cross_frame_unit_mismatch"
    CROSS_FRAME_UNIT_UNBOUND = "cross_frame_unit_unbound"
    CROSS_FRAME_FACET_MISMATCH = "cross_frame_facet_mismatch"
    SHARE_OF_GRAIN_MISMATCH = "share_of_grain_mismatch"
    FILTER_PREDICATE_TYPE_MISMATCH = "filter_predicate_type_mismatch"
    FILTER_UNIT_CONVERSION_UNSUPPORTED = "filter_unit_conversion_unsupported"
    SHARE_OF_ZERO_DENOMINATOR = "share_of_zero_denominator"
    SHARE_OF_KEY_UNKNOWN = "share_of_key_unknown"
    SHARE_OF_KEY_UNMATCHED = "share_of_key_unmatched"
    QUOTIENT_OF_ZERO_DENOMINATOR = "quotient_of_zero_denominator"
    QUOTIENT_OF_NO_KEY_MATCHED = "quotient_of_no_key_matched"
    QUOTIENT_OF_SIDE_UNIT_UNSUPPORTED = "quotient_of_side_unit_unsupported"
    COMPOSED_UNIT_SUM_UNSUPPORTED = "composed_unit_sum_unsupported"
    JOIN_OUTPUT_COLLISION = "join_output_collision"
    JOIN_NO_COMMON_KEY_COLUMNS = "join_no_common_key_columns"
    DERIVED_INPUT_UNUSABLE = "derived_input_unusable"
    DERIVED_INPUT_UNRECOVERABLE = "derived_input_unrecoverable"
    DERIVED_INPUT_ORDINAL = "derived_input_ordinal"
    PLACE_LEVELS_MIXED = "place_levels_mixed"
    ROLLING_ANCHOR_UNRESOLVED = "rolling_anchor_unresolved"
    ROLLING_GRAIN_UNSUPPORTED = "rolling_grain_unsupported"
    ROLLING_REANCHOR_UNAVAILABLE = "rolling_reanchor_unavailable"
    SERVICE_UNAVAILABLE = "service_unavailable"
    RENDERING_NOT_RECORDED = "rendering_not_recorded"
    UPSTREAM_ERROR = "upstream_error"
    INTERNAL_ERROR = "internal_error"


# `do_not_retry` codes share one shape: nothing about the call's arguments
# was wrong, so retrying unchanged reaches the identical failure.
# `non_finite_result`/`internal_error`: no specific argument to change.
# `mixed_time_grain`: the frame `frame_ref` names doesn't change between calls.
# `derived_input_unrecoverable`: a derived input whose root dfo-apis does not
# hold, so only a fresh run from the source data fixes it.
# `derived_input_ordinal` stays `correct_and_retry`: `growth` retries with
# `change="absolute"` over the identical frame_ref, `rank` retries over the
# frame_ref the per-period rank read, and `share_of` and `quotient_of` retry
# with a frame_ref that isn't itself a per-period rank's own output -- the
# same "a different input fixes this" shape `derived_input_unusable` has.
# `frame_integrity_failed`: a corrupted or tampered stored object, so the
# identical frame_ref reaches the identical bytes.
# `rolling_grain_unsupported`: the root frames' periods don't change between calls.
# `service_unavailable` is `retry_after_delay`, since the identical call succeeds
# once the wait it names has passed.
# Every other code names something a caller can change (a different
# frame_ref, a supported aggregation, a narrower group_by, a unit pin).
_RETRY_MODE_BY_CODE: dict[ToolErrorCode, RetryMode] = {
    ToolErrorCode.OP_SPEC_INVALID: RetryMode.CORRECT_AND_RETRY,
    ToolErrorCode.STAT_VAR_UNRESOLVED: RetryMode.CORRECT_AND_RETRY,
    ToolErrorCode.ROLE_AMBIGUOUS: RetryMode.CORRECT_AND_RETRY,
    ToolErrorCode.OPERATION_UNCOVERED: RetryMode.CORRECT_AND_RETRY,
    ToolErrorCode.FRAME_TOO_LARGE: RetryMode.CORRECT_AND_RETRY,
    ToolErrorCode.UNIT_HETEROGENEOUS: RetryMode.CORRECT_AND_RETRY,
    ToolErrorCode.FACET_HETEROGENEOUS: RetryMode.CORRECT_AND_RETRY,
    ToolErrorCode.COVERAGE_INSUFFICIENT: RetryMode.CORRECT_AND_RETRY,
    ToolErrorCode.NON_FINITE_RESULT: RetryMode.DO_NOT_RETRY,
    ToolErrorCode.FRAME_NOT_HELD: RetryMode.CORRECT_AND_RETRY,
    ToolErrorCode.FRAME_EXPIRED: RetryMode.CORRECT_AND_RETRY,
    ToolErrorCode.FRAME_REF_INCOMPLETE: RetryMode.CORRECT_AND_RETRY,
    ToolErrorCode.MIXED_TIME_GRAIN: RetryMode.DO_NOT_RETRY,
    ToolErrorCode.FRAME_INTEGRITY_FAILED: RetryMode.DO_NOT_RETRY,
    ToolErrorCode.COMPARISON_PERIODS_DISJOINT: RetryMode.CORRECT_AND_RETRY,
    ToolErrorCode.GROWTH_GAP_YEAR: RetryMode.CORRECT_AND_RETRY,
    ToolErrorCode.CROSS_FRAME_UNIT_MISMATCH: RetryMode.CORRECT_AND_RETRY,
    ToolErrorCode.CROSS_FRAME_UNIT_UNBOUND: RetryMode.CORRECT_AND_RETRY,
    ToolErrorCode.CROSS_FRAME_FACET_MISMATCH: RetryMode.CORRECT_AND_RETRY,
    ToolErrorCode.SHARE_OF_GRAIN_MISMATCH: RetryMode.CORRECT_AND_RETRY,
    ToolErrorCode.FILTER_PREDICATE_TYPE_MISMATCH: RetryMode.CORRECT_AND_RETRY,
    ToolErrorCode.FILTER_UNIT_CONVERSION_UNSUPPORTED: RetryMode.CORRECT_AND_RETRY,
    ToolErrorCode.SHARE_OF_ZERO_DENOMINATOR: RetryMode.CORRECT_AND_RETRY,
    ToolErrorCode.SHARE_OF_KEY_UNKNOWN: RetryMode.CORRECT_AND_RETRY,
    ToolErrorCode.SHARE_OF_KEY_UNMATCHED: RetryMode.CORRECT_AND_RETRY,
    ToolErrorCode.QUOTIENT_OF_ZERO_DENOMINATOR: RetryMode.CORRECT_AND_RETRY,
    ToolErrorCode.QUOTIENT_OF_NO_KEY_MATCHED: RetryMode.CORRECT_AND_RETRY,
    ToolErrorCode.QUOTIENT_OF_SIDE_UNIT_UNSUPPORTED: RetryMode.CORRECT_AND_RETRY,
    ToolErrorCode.COMPOSED_UNIT_SUM_UNSUPPORTED: RetryMode.CORRECT_AND_RETRY,
    ToolErrorCode.JOIN_OUTPUT_COLLISION: RetryMode.CORRECT_AND_RETRY,
    ToolErrorCode.JOIN_NO_COMMON_KEY_COLUMNS: RetryMode.CORRECT_AND_RETRY,
    ToolErrorCode.DERIVED_INPUT_UNUSABLE: RetryMode.CORRECT_AND_RETRY,
    ToolErrorCode.DERIVED_INPUT_UNRECOVERABLE: RetryMode.DO_NOT_RETRY,
    ToolErrorCode.DERIVED_INPUT_ORDINAL: RetryMode.CORRECT_AND_RETRY,
    ToolErrorCode.PLACE_LEVELS_MIXED: RetryMode.CORRECT_AND_RETRY,
    ToolErrorCode.ROLLING_ANCHOR_UNRESOLVED: RetryMode.CORRECT_AND_RETRY,
    ToolErrorCode.ROLLING_GRAIN_UNSUPPORTED: RetryMode.DO_NOT_RETRY,
    ToolErrorCode.ROLLING_REANCHOR_UNAVAILABLE: RetryMode.CORRECT_AND_RETRY,
    ToolErrorCode.SERVICE_UNAVAILABLE: RetryMode.RETRY_AFTER_DELAY,
    ToolErrorCode.RENDERING_NOT_RECORDED: RetryMode.CORRECT_AND_RETRY,
    ToolErrorCode.UPSTREAM_ERROR: RetryMode.RETRY_AFTER_DELAY,
    ToolErrorCode.INTERNAL_ERROR: RetryMode.DO_NOT_RETRY,
}
if set(_RETRY_MODE_BY_CODE) != set(ToolErrorCode):
    # A plain `raise`, not `assert` -- `python -O` strips asserts, and this
    # import-time guard should hold even under `-O`.
    raise AssertionError("every ToolErrorCode needs exactly one _RETRY_MODE_BY_CODE entry")


# The codes of a refusal whose caller frames the runner no longer holds.
_LOST_CODES = frozenset(
    {
        ToolErrorCode.FRAME_NOT_HELD,
        ToolErrorCode.FRAME_EXPIRED,
        ToolErrorCode.DERIVED_INPUT_UNRECOVERABLE,
    }
)

# The sentence a lost-frame refusal gains when the server could not rebuild the frame.
_NOT_REBUILT = (
    "The server could not rebuild this frame by itself. Call get_data with the spec_id "
    "(spec_...) to register it again and pass the frame_ref it returns; for an ops_ result, "
    "repeat the verb calls that made it."
)

# The level a `place_levels_mixed` fix names first when the places sit at it.
_COUNTRY_LEVEL = "Country"

# The party kinds a filter keeps to compare or rank a flow side's members: private parties
# are of kind `party` too, told apart by their class.
_MEMBER_KINDS: tuple[PartyKind, ...] = ("party",)
# The kinds a filter keeps to sum the members of a role alone: its parties and the
# unallocated residuals, which add to a sum though no ranking compares them.
_POOLED_MEMBER_KINDS: tuple[PartyKind, ...] = ("party", "residual")


@dataclass(frozen=True, kw_only=True)
class ToolRefusal:
    """A tool's structured refusal, built by one of its typed factories, or passed
    through from dfo-apis' analysis runner (`runner_client.refusal_for`). A runner
    refusal keeps the runner's `code`, which may be one this module does not define,
    and the `runner_retry_mode` its problem document states."""

    code: ToolErrorCode | str
    message: str
    fix: dict[str, Any]
    retry_after_s: float | None = None
    upstream_code: str | None = None
    runner_retry_mode: RetryMode | None = None

    @property
    def retry_mode(self) -> RetryMode:
        if self.runner_retry_mode is not None:
            return self.runner_retry_mode
        if self.code == ToolErrorCode.UPSTREAM_ERROR and self.retry_after_s is None:
            # A runner refusal that was not retryable names no delay to wait.
            return RetryMode.DO_NOT_RETRY
        if isinstance(self.code, ToolErrorCode):
            return _RETRY_MODE_BY_CODE[self.code]
        # A runner code this module does not define, whose problem document stated
        # no retry mode.
        return RetryMode.DO_NOT_RETRY

    @property
    def lost_frames(self) -> tuple[FrameRef, ...]:
        """The caller's frames this refusal says the runner no longer holds: every
        frame `frames_lost` or `derived_input_root_lost` names, none for any other."""
        if self.code not in _LOST_CODES:
            return ()
        named = self.fix.get("frame_refs")
        if named is None and "spec_id" in self.fix and "content_version" in self.fix:
            named = [self.fix]
        if not isinstance(named, list):
            return ()
        return tuple(
            FrameRef(spec_id=ref["spec_id"], content_version=ref["content_version"])
            for ref in named
        )

    def not_rebuilt(self) -> Self:
        """This lost-frame refusal after the server failed to rebuild its frames."""
        return replace(self, message=f"{self.message} {_NOT_REBUILT}")

    @classmethod
    def op_spec_invalid(cls, *, detail: str) -> Self:
        return cls(
            code=ToolErrorCode.OP_SPEC_INVALID,
            message=f"The op-spec was invalid: {detail}. Correct it and retry.",
            fix={"detail": detail},
        )

    @classmethod
    def place_level_side_unnamed(cls, *, columns: list[str], available: list[str]) -> Self:
        """A `place_level` step read `columns`, which this frame lacks. `available`
        holds the frame's `{role}_type` columns. Choosing a side here could drop every
        row, as on a frame whose provider is a DAC group, so the caller chooses. `filter`
        is the one route for every verb, since a pooled `share_of` denominator has no key to
        name a side with and a key added to `group_by` changes the rows the call returns.
        """
        next_step = (
            "Call filter with a place_level predicate on the column of the side to keep, on "
            "each frame this call reads. Then repeat this call on the filtered frames without "
            "place_level. Or leave place_level out."
            if available
            else "Leave place_level out."
        )
        return cls(
            code=ToolErrorCode.ROLE_AMBIGUOUS,
            message=(
                f"place_level reads {columns}, which this frame does not have. Its place "
                f"type columns: {available}. {next_step}"
            ),
            fix={
                "parameter": "place_level",
                "columns": columns,
                "available": available,
                "retry_without": "place_level",
            },
        )

    @classmethod
    def frame_not_held(cls, *, spec_id: str, content_version: str) -> Self:
        """No result held for this pair. Three causes, not conflated: never
        registered (call get_data, or produce a fresh verb result), fabricated
        (invented rather than copied verbatim from a prior tool response --
        a fabricated pair parses cleanly, since it need only match the
        {spec_id, content_version} shape), or aged out of this store's own
        retention policy (a durable store's own bucket lifecycle rule
        deleted it; an in-memory store's own TTL sweep surfaces the same
        pair as `frame_expired` instead, since it can still tell "was held"
        from "never held")."""
        return cls(
            code=ToolErrorCode.FRAME_NOT_HELD,
            message=(
                f"No result held for spec_id={spec_id!r}, content_version={content_version!r}. "
                "This store is up -- the pair just isn't reachable from here: never registered "
                "(call get_data with this spec_id, or produce a fresh verb result and use its "
                "own reference), fabricated (invented rather than copied verbatim from a prior "
                "tool response), or aged out of this store's own retention policy. Produce a "
                "fresh result and retry with its reference."
            ),
            fix={"spec_id": spec_id, "content_version": content_version},
        )

    @classmethod
    def frame_expired(cls, *, spec_id: str, content_version: str) -> Self:
        if spec_id.startswith("spec_"):
            fix_text = "Call get_data with this spec_id to re-register the identical reference."
        else:
            fix_text = (
                "This was a toolkit result, not a Datasets snapshot -- there is no get_data "
                "retry for it. Retry the verb call that produced it (its own inputs may have "
                "aged out too, in which case resolve those first)."
            )
        return cls(
            code=ToolErrorCode.FRAME_EXPIRED,
            message=(
                f"The frame held for spec_id={spec_id!r}, content_version={content_version!r} "
                f"aged out. {fix_text}"
            ),
            fix={"spec_id": spec_id, "content_version": content_version},
        )

    @classmethod
    def frames_lost(cls, *, frame_refs: Sequence[Mapping[str, str]], expired: bool) -> Self:
        """`frame_expired` (`expired`) or `frame_not_held` for the caller's frames
        `frame_refs`, which the runner read as one dataset version it no longer holds.
        Two sides fetched from one version with different options are two frames over
        one root, so the refusal names both and one round of fetches mends the call."""
        first, *others = frame_refs
        lost = cls.frame_expired if expired else cls.frame_not_held
        refusal = lost(spec_id=first["spec_id"], content_version=first["content_version"])
        if not others:
            return refusal
        named = ", ".join(f"{ref['spec_id']}/{ref['content_version']}" for ref in others)
        return cls(
            code=refusal.code,
            message=(
                f"{refusal.message} The frame(s) {named} of this call were read from the "
                "same data and are lost with it: fetch each again the same way and retry "
                "with the new frame_refs."
            ),
            fix={**refusal.fix, "frame_refs": [dict(ref) for ref in frame_refs]},
        )

    @classmethod
    def bare_frames_lost(cls, *, spec_ids: Sequence[str], expired: bool) -> Self:
        """`frame_expired` (`expired`) or `frame_not_held` for a call whose frames
        `spec_ids`, passed without a `content_version`, the runner could not read at
        their newest version. Its refusal names no root, so each is named, and it names
        no version, so `lost_frames` holds none of them. Reading the newest version is
        all a rebuild would do."""
        named = ", ".join(spec_ids)
        return cls(
            code=ToolErrorCode.FRAME_EXPIRED if expired else ToolErrorCode.FRAME_NOT_HELD,
            message=(
                f"The newest version of {named} cannot be read, and retrying with the same "
                "spec_id reaches the same refusal. Call get_data with the spec_id, or search "
                "for the data again, and retry with the frame_ref it returns."
            ),
            fix={"spec_ids": list(spec_ids)},
        )

    @classmethod
    def frame_ref_incomplete(cls, *, argument: str, spec_id: str) -> Self:
        """The frame argument `argument` names `spec_id`, which is no `spec_` id, with
        no `content_version`. The caller's `spec_id` is echoed capped
        (`echoed_text.truncate_value`)."""
        shown = truncate_value(spec_id)
        return cls(
            code=ToolErrorCode.FRAME_REF_INCOMPLETE,
            message=(
                f"{argument} names {shown!r} without its content_version, and only a "
                "spec_ id is fetched by itself. Pass the full {spec_id, content_version} "
                "pair from the result that made the frame."
            ),
            fix={"argument": argument, "spec_id": shown},
        )

    @classmethod
    def frame_integrity_failed(cls, *, spec_id: str, content_version: str, reason: str) -> Self:
        """The durable store found the stored bytes for this pair don't
        match what was recorded at write time: {reason}. Retrying the
        identical frame_ref reaches the identical corrupted object -- the
        fix is a fresh result under a new reference."""
        return cls(
            code=ToolErrorCode.FRAME_INTEGRITY_FAILED,
            message=(
                f"The stored frame for spec_id={spec_id!r}, content_version={content_version!r} "
                f"failed an integrity check: {reason}. Retrying this reference will not help -- "
                "produce a fresh result and retry with its own reference instead: call get_data "
                "with this spec_id, or re-run the verb call that produced it."
            ),
            fix={"spec_id": spec_id, "content_version": content_version, "reason": reason},
        )

    @classmethod
    def share_units_differ(cls, *, numerator_unit: object, denominator_unit: object) -> Self:
        """A share whose numerator and denominator resolved to two units, each a
        unit as the runner's refusal states it: a unit ref, or a converted or
        composed unit's object. Two units make a quotient, so `fix` names that
        form beside the runner's two units."""
        units = (json.dumps(numerator_unit), json.dumps(denominator_unit))
        return cls(
            code=ToolErrorCode.CROSS_FRAME_UNIT_MISMATCH,
            message=(
                f"The numerator is in {units[0]} and the denominator in {units[1]}, so the "
                "numerator is no share of the denominator. Repeat the call with "
                'operation="quotient" and the same arguments to divide one by the other.'
            ),
            fix={
                "operation": "quotient",
                "numerator": numerator_unit,
                "denominator": denominator_unit,
            },
        )

    @classmethod
    def derived_input_root_lost(
        cls, *, frame_refs: Sequence[Mapping[str, str]], root: Mapping[str, Any], upstream_code: str
    ) -> Self:
        """`frame_refs` are the derived (`ops_`) inputs of the call computed from
        `root`, the dataset version dfo-apis no longer holds
        (`upstream_code`: `snapshot_not_found`, `recipe_not_found` or
        `artifact_expired`). dfo-apis computes a derived input again from its
        stored recipe, so the input is lost with its root."""
        named = ", ".join(f"{ref['spec_id']}/{ref['content_version']}" for ref in frame_refs)
        return cls(
            code=ToolErrorCode.DERIVED_INPUT_UNRECOVERABLE,
            message=(
                f"The derived frame {named} was computed from data the analysis service no "
                f"longer holds ({upstream_code}), so it cannot be computed again, and "
                "retrying with the same frame_ref reaches the same refusal. Recompute it "
                "from its sources: call get_data for the data it came from, repeat the "
                "steps that produced it, and retry with the new frame_ref."
            ),
            fix={
                "frame_refs": [dict(ref) for ref in frame_refs],
                "root": {key: root[key] for key in ("spec_id", "content_version") if key in root},
            },
        )

    @classmethod
    def place_levels_mixed(
        cls,
        *,
        column: str,
        types: Sequence[str],
        whole_total: str | None = None,
        whole_class: str | None = None,
        member_count: int = 0,
        members: Sequence[str] = (),
        takes_place_level: bool = True,
    ) -> Self:
        """`types` are the distinct cells of `column`, sorted and capped by
        the engine that found them, and empty when the frame carries no such
        column. A flow party role's type or kind column (`PARTY_ROLES`) gets a
        message naming the role and the group-total exclusion, since a
        `place_level` filter is not what its refusal asks for. It names a
        search naming the population for a total across the whole side, a
        filter keeping party and residual for a sum of members, and for a
        comparison of members a filter keeping the kinds a kind column's
        `fix.member_kinds` lists. `whole_total` names a whole total of
        `whole_class`, or of no class when None, pooled beside `member_count`
        members it may hold, of which `members` are examples. The message asks
        to keep the whole totals alone, by a filter on the role's whole column,
        or the members alone, by a filter on `column`, and `fix.filters` holds
        both. Any other column's message asks for `place_level` where the
        tool takes it (`takes_place_level`), and otherwise for a `place_level`
        predicate through `filter`."""
        role = enrichment_role(column, roles=PARTY_ROLES)
        if role is not None and whole_total is not None:
            counted = f"{member_count} member{'' if member_count == 1 else 's'}"
            held = (
                f"a whole total of the {whole_class} {role} parties, and {counted} beside "
                "it are of that class or of none"
                if whole_class is not None
                else f"a whole total of no one class, and it may hold the {counted} beside it"
            )
            whole = whole_column(role)
            return cls(
                code=ToolErrorCode.PLACE_LEVELS_MIXED,
                message=(
                    f"The {role} parties in this frame cannot be pooled: {whole_total} is "
                    f"{held}, such as {', '.join(members)}. A sum across them would count "
                    "those members twice. Filter the frame with a category_match on "
                    f"{whole!r} keeping {WHOLE_CELL}, to sum the whole totals alone, or a "
                    f"category_match on {column!r} keeping {' and '.join(_POOLED_MEMBER_KINDS)}, "
                    "to sum the members alone, then retry with that result's frame_ref."
                ),
                fix={
                    "column": column,
                    "types": list(types),
                    "whole_total": whole_total,
                    "whole_class": whole_class,
                    "member_count": member_count,
                    "example_members": list(members),
                    "filters": [
                        {"column": whole, "values": [WHOLE_CELL]},
                        {"column": column, "values": list(_POOLED_MEMBER_KINDS)},
                    ],
                },
            )
        if role is not None and not types:
            return cls(
                code=ToolErrorCode.PLACE_LEVELS_MIXED,
                message=(
                    f"The {role} parties in this frame cannot be pooled or ranked: the frame "
                    f"carries no {column!r} column, so its parties cannot be told apart from "
                    "the group totals that hold them. Call get_data again for this spec_id, "
                    "with place_types=true where get_data takes that argument (otherwise it "
                    "adds the party kinds by default), and repeat the call with the frame_ref "
                    f"it returns, or filter the frame to a single {role} party."
                ),
                fix={"column": column, "types": []},
            )
        if role is not None:
            return cls(
                code=ToolErrorCode.PLACE_LEVELS_MIXED,
                message=(
                    f"The {role} parties in this frame cannot be pooled: {column!r} holds "
                    f"{', '.join(types)}. Parties of kind total, such as a DAC regional total, "
                    "are left out of sums and ranks beside their members, and totals alone are "
                    f"summed when {whole_column(role)!r} is true on every row. So a party here has "
                    "no declared kind, or every party is a total and not all are whole. For a "
                    "total across the whole side, search naming the population "
                    f'("{WHOLE_SIDE_POPULATIONS[role]}") with no operation over that side. '
                    f"Otherwise filter the frame with a category_match on {role!r} naming the "
                    f"parties to keep, or a category_match on {column!r}: to sum the members, "
                    f"keep party and residual; to compare or rank them, keep "
                    f"{', '.join(_MEMBER_KINDS)} alone. Then retry with that result's frame_ref; "
                    f"{class_column(role)!r} tells private parties from official ones."
                ),
                fix={"column": column, "types": list(types), "member_kinds": list(_MEMBER_KINDS)},
            )
        levels = sorted({level for cell in types for level in levels_of(cell) & PLACE_LEVELS})
        level = _COUNTRY_LEVEL if _COUNTRY_LEVEL in levels else next(iter(levels), None)
        keep_one_level = (
            "Repeat the call with place_level set to the level to keep, for example "
            'place_level="Country".'
            if takes_place_level
            else f"Filter the frame first with a place_level predicate on {column!r} naming "
            "the level to keep, such as Country, then retry with that result's frame_ref."
        )
        return cls(
            code=ToolErrorCode.PLACE_LEVELS_MIXED,
            message=(
                f"This frame's places share no level a filter can keep: {column!r} holds "
                f"{', '.join(types)}, and no one of {', '.join(sorted(PLACE_LEVELS))} is "
                "among every place's types. A frame that names no place holds the world, "
                "regions, World Bank groups, countries and their subdivisions side by side, "
                "and ranking or summing them together counts a place beside the places inside "
                "it. A type outside those levels, such as Place or ProvisionalNode, names no "
                f"level, and no level kept leaves the places that carry it alone. {keep_one_level}"
            ),
            fix={"column": column, "types": list(types)}
            | ({} if level is None or not takes_place_level else {"place_level": level}),
        )

    @classmethod
    def rolling_anchor_unresolved(
        cls, *, detail: str, reason: str, context: Mapping[str, Any] | None = None
    ) -> Self:
        """`detail` is the machine-readable name of the check that failed and
        `reason` says it in words. `context` adds the values the check read."""
        return cls(
            code=ToolErrorCode.ROLLING_ANCHOR_UNRESOLVED,
            message=(
                f"The latest period of the data is not established: {reason}. Call "
                "resolve_anchor with the populations and every root frame of the question, "
                "then retry with the anchor_key and the periods it gives."
            ),
            fix={"detail": detail, **(context or {})},
        )

    @classmethod
    def latest_year_unresolved(cls, *, fix: Mapping[str, Any], derived: Sequence[str] = ()) -> Self:
        """`rolling_anchor_unresolved` for a call with no `anchor_key` to resolve and
        retry with. `fix` is the runner's, whose `detail` names the check that failed.
        A year counted from the latest year stands in the call or in the recipe of one
        of its `derived` inputs, which the call computes again."""
        fixed_years = (
            f"State each such year as a fixed year, in this call and in the calls that made "
            f"{', '.join(derived)}, then retry with the frame_ref those return."
            if derived
            else "State the periods as fixed years instead."
        )
        return cls(
            code=ToolErrorCode.ROLLING_ANCHOR_UNRESOLVED,
            message=(
                f"The latest year of the data is not established ({fix.get('detail')}), so "
                f"a year counted from it cannot be worked out. {fixed_years}"
            ),
            fix=dict(fix),
        )

    @classmethod
    def rolling_reanchor_unavailable(cls, *, frame_refs: Sequence[Mapping[str, str]]) -> Self:
        """The derived (`ops_`) inputs `frame_refs` hold no stored recipe, so the
        runner cannot compute them again inside this call's tree, where every
        rolling year follows one latest year."""
        named = ", ".join(f"{ref['spec_id']}/{ref['content_version']}" for ref in frame_refs)
        return cls(
            code=ToolErrorCode.ROLLING_REANCHOR_UNAVAILABLE,
            message=(
                f"The derived frame {named} holds no stored recipe, so this call cannot "
                "compute it again under one latest year with its other inputs. Make it "
                "again: call get_data for the data it came from, repeat the verb calls "
                "that made it, and retry with the frame_ref the last one returns."
            ),
            fix={"frame_refs": [dict(ref) for ref in frame_refs], "tool": "get_data"},
        )

    @classmethod
    def frame_store_unavailable(
        cls, *, spec_id: str, content_version: str, retry_after_s: float
    ) -> Self:
        """The frame store failed a read of the mint record of a `get_data` frame,
        which names the dataset version the runner reads it as."""
        return cls(
            code=ToolErrorCode.SERVICE_UNAVAILABLE,
            message=(
                f"The record of the frame spec_id={spec_id!r}, "
                f"content_version={content_version!r} could not be read just now. Retry "
                f"the same call after {retry_after_s:.1f}s."
            ),
            fix={"service": "frame-store", "spec_id": spec_id, "content_version": content_version},
            retry_after_s=retry_after_s,
        )

    @classmethod
    def analysis_pending(cls, *, retry_after_s: float, operations: Sequence[str]) -> Self:
        """The runner answered 202 because a root dataset of the call is still
        building. `operations` are the rebuilds it waits on, none for a call the
        runner is already computing."""
        queued = f" ({len(operations)} rebuild(s) queued)" if operations else ""
        return cls(
            code=ToolErrorCode.SERVICE_UNAVAILABLE,
            message=(
                f"The data this call reads is still being prepared upstream{queued}. "
                f"Retry the same call after {retry_after_s:.1f}s."
            ),
            fix={"service": "dfo-apis", "operations": list(operations)},
            retry_after_s=retry_after_s,
        )

    @classmethod
    def rendering_not_recorded(
        cls, *, spec_id: str, content_version: str, options: Sequence[str]
    ) -> Self:
        """`options` are the mint options of the frame the runner cannot read a root with,
        empty for a frame no mint record states the rendering of."""
        frame = f"The frame spec_id={spec_id!r}, content_version={content_version!r}"
        return cls(
            code=ToolErrorCode.RENDERING_NOT_RECORDED,
            message=(
                f"{frame} was fetched with {', '.join(options)}, which the analysis service "
                "cannot read a frame with. Call get_data for the same data without those "
                "options and retry with that frame_ref."
                if options
                else f"{frame} is no longer held, and no record states how it was fetched, "
                "so it cannot be read again with the same rows. Call get_data for the same "
                "data and retry with that frame_ref."
            ),
            fix={
                "spec_id": spec_id,
                "content_version": content_version,
                "options": list(options),
                "tool": "get_data",
            },
        )

    @classmethod
    def upstream_error(
        cls,
        *,
        upstream_code: str | None,
        detail: str,
        retry_after_s: float | None,
        fix: Mapping[str, Any] | None = None,
    ) -> Self:
        """A runner refusal under a code this module does not define, or a call the
        runner could not answer. `retry_after_s` is `None` when a retry is not
        expected to succeed."""
        retry = (
            f"Retry after {retry_after_s:.1f}s."
            if retry_after_s is not None
            else "A repeated call is unlikely to succeed unmodified."
        )
        return cls(
            code=ToolErrorCode.UPSTREAM_ERROR,
            message=(
                f"The analysis service refused the call ({upstream_code or 'unclassified'}): "
                f"{detail}. {retry}"
            ),
            fix={"detail": detail, **(fix or {})},
            retry_after_s=retry_after_s,
            upstream_code=upstream_code,
        )

    @classmethod
    def internal_error(cls, *, tool: str) -> Self:
        """The registration-site catch-all's own refusal (see `server.py`'s
        module docstring), for an exception no verb's own typed handling
        anticipated. The exception's text stays in the log
        (`common.refusal.log_unexpected_error`): it can hold a bucket path or
        a service account, and it gives a retry nothing to act on, which is
        why this code is `do_not_retry`.
        """
        return cls(
            code=ToolErrorCode.INTERNAL_ERROR,
            message=internal_error_message(subject=tool),
            fix={"tool": tool},
        )


def to_tool_result(refusal: ToolRefusal, *, preamble: str | None = None) -> ToolResult:
    """Render `refusal` as the `ToolResult` a tool returns to signal an error.

    `preamble`, when given, is prepended to `refusal.message` in `content`
    -- the mechanism for a partial success: what resolved, followed by why
    the rest didn't. `structured_content` is unaffected either way.

    `content` is `common.refusal.refusal_text`: `refusal.message`, then a blank
    line, then `code`, `retry_mode`, `fix` and `retry_after_s` when set as one
    line of compact JSON, so a client that renders only `content` still sees
    them. `structured_content` carries `retry_after_s` under the same rule.

    `structured_content` also carries `upstream_code` when set, and `meta` is
    `common.refusal.refusal_meta` of `code` and `upstream_code`.
    """
    return ToolResult(
        content=refusal_text(
            message=refusal.message,
            code=str(refusal.code),
            retry_mode=refusal.retry_mode.value,
            fix=refusal.fix,
            retry_after_s=refusal.retry_after_s,
            preamble=preamble,
        ),
        structured_content={
            "code": str(refusal.code),
            "retry_mode": refusal.retry_mode.value,
            # `refusal.message` itself, not just `content`: a caller that
            # reads `structured_content` alone (`executor.classify.
            # parse_tool_failure`, reading the raw MCP payload rather than
            # this module's own `content` text) needs the same prose a
            # human reader of `content` gets, preamble aside.
            "message": refusal.message,
            "fix": refusal.fix,
        }
        | ({} if refusal.retry_after_s is None else {"retry_after_s": refusal.retry_after_s})
        | ({} if refusal.upstream_code is None else {"upstream_code": refusal.upstream_code}),
        meta=refusal_meta(code=str(refusal.code), upstream_code=refusal.upstream_code),
        is_error=True,
    )
