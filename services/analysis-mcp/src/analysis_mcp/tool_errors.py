"""Tool-facing refusal grammar, following `data_mcp.tool_errors`'s shape: a
`StrEnum` code, a `StrEnum` retry mode, a frozen kw-only `ToolRefusal` with
one classmethod factory per code, and `to_tool_result` to render one.

Simplified from that precedent: analysis-mcp makes no upstream HTTP calls of
its own, so there is no `upstream_code`/`trace_id`/`capability_version` to
carry -- this module's fields are `code`, `message`, `fix`, and the
`retry_after_s` only `service_unavailable` sets.

`unit_heterogeneous` and `facet_heterogeneous` carry a
`analysis_mcp.disambiguation.DisambiguationPayload` in `fix["disambiguation"]`.
`unit_heterogeneous` is retryable through whichever unit-pin parameter the
calling verb declares (`fix["parameter"]`); `facet_heterogeneous` has no pin,
so its payload names the axis and candidates for a caller to act on by
choosing a differently-scoped `frame_ref`. A unit refusal's `fix["units"]`
lists each candidate in the form the pin parameter accepts: a unit ref as
its string, a converted unit (a `UnitRequest` cell) as its request object.
A converted candidate's label names its price basis and currency, and its
`id` is the request object to pin. `role_ambiguous` names whichever
column-list parameter the calling verb actually has in `fix["parameter"]`,
never a hardcoded name borrowed from `aggregate`.

`comparison_periods_disjoint` fires when compare's two sides share no
overlapping period; `growth_gap_year` fires when no entity has a row in
both of growth's endpoint years. `cross_frame_unit_mismatch`/`cross_frame_unit_unbound`/
`cross_frame_facet_mismatch` fire when two independently-resolved sides
(`compare`'s left/right, `share_of`'s numerator/denominator) disagree on
unit or facet -- distinct from `unit_heterogeneous`/`facet_heterogeneous`,
which fire when *one* frame itself carries more than one value.
`cross_frame_unit_unbound` is the narrower case where one side's rows leave
`unit` unbound entirely rather than resolving to a second, differing unit.

`share_of_grain_mismatch` refuses rather than reporting both sides'
coverage when numerator and denominator periods don't match;
`fix["matched_periods"]` names the overlap the two sides do share, if any,
for a `filter`-then-retry repair; `fix["key"]` names the offending
`numerator_by` key (empty for an un-keyed call), since a keyed `share_of`
checks the identical-period-set rule per key, not pooled across every key.
`share_of_zero_denominator` is split out from `non_finite_result` because a
different `frame_ref` or entity binding can produce a nonzero sum, so it is
retryable.

`share_of_key_unknown` fires when `denominator_by` is non-empty but leaves
out a `numerator_by` column -- a partial mapping has no defined pairing for
the column(s) it omits (a `denominator_by` key `numerator_by` itself
doesn't carry refuses earlier, at the op-spec, as `op_spec_invalid`).
`share_of_key_unmatched` fires when a numerator key, realised from its own
frame's rows, has no matching denominator group under `denominator_by`'s
own mapping.

`filter_predicate_type_mismatch` fires when a predicate's column exists but
its declared cell type doesn't admit that predicate kind, or an
`ordered_threshold` operand carries the wrong type; it is retryable.
`filter_unit_conversion_unsupported` fires when a `category_match` on `unit`
meets a frame whose `unit` cells carry a converted unit, which no unit string
matches. Predicates on every other column still apply to that frame, so it is
retryable: without the unit predicate, or after a `get_data` call that binds
one unit.

`join_output_collision` fires when both sides of a `join` resolve to the
identical `stat_var`, unit, and facets, so a matched key's two rows would
collide on one grain. `join_no_common_key_columns` fires when `join_keys`
is left empty and the two frames declare no role column in common while
both declare some -- distinct from `join_output_collision`, which fires
after a match is found.

`mixed_time_grain` fires when a resolved input's `time_period` column
carries more than one grain, a span beside a point, or two different spans,
checked once at frame admission
(`frame_ref_resolution.resolve_input_frames`); `do_not_retry`, since the
stored frame doesn't change between calls. `frame_not_held` fires when a
resolved `frame_ref` names a pair the store cannot return -- never
registered, fabricated, or past the store's own retention. `frame_expired`
fires when a pair aged out past an in-memory store's TTL, with a `get_data`
retry only for a Datasets-family (`spec_...`) pair; a durable, bucket-backed
store never raises it, since a pair its lifecycle rule has deleted comes
back as the same `frame_not_held` a pair that was never registered would.
`frame_integrity_failed` fires when a durable store's read-time check finds
the stored object itself is wrong: its bytes don't match what was recorded
at write time, its row count differs from the count recorded beside it,
its envelope names a different reference than the one requested, or, when the recompute check is enabled, a recomputed
content_version disagrees with the value that check compares against. Every
one of those means the identical reference reaches the identical wrong
object, so the fix is a fresh result under a new reference.
`frame_too_large_before_decode` reuses `frame_too_large`'s code for a stored
object the same store refuses before decoding, over its encoded byte size,
row count, column count, or row-group uncompressed size. The bytes are
exactly what was written, just too large for this store to serve back, so
unlike `frame_integrity_failed` it is retryable with a narrower
`frame_ref`. `frame_too_large` itself fires on the row count the store
reports for a held frame before it is read, against `Settings.max_frame_rows`
-- a service-level ceiling, not the store's own resource-protection one. `frame_too_large_on_write` also
reuses `frame_too_large`'s code, for the same store's row ceiling checked
the other direction: a verb's own computed result, refused before `put`
writes anything, so no reference is ever minted for a caller to retry with
-- the fix is narrowing this call's own inputs, not a smaller `frame_ref`.

`derived_input_unusable` refuses a derived input a verb cannot read, with a
retry over a different `frame_ref`. `derived_input_unrecoverable` refuses,
for every verb, an input whose trace records an analysis-mcp engine before
0.3.0. Those engines labelled a collapsed or grown row with one borrowed
point, such as `2022` over a 2018-2022 sum, so their periods do not state the
years they cover and no retry over the same result fixes it.

`derived_input_ordinal` fires when a resolved input carries
`DerivedMarker.ordinal` -- a per-period `rank`'s own `value` column holds a
rank position, not a measured value, and no verb here may treat a position
as a quantity to ratio, grow at a percentage rate or rank again. Raised by
`share_of` on either side (a share has no meaning over a position), by
`growth` when `change="relative"` (a rank moves by places, not by a
percentage of one) and by `rank` (highest first over positions puts the
last-placed entity first). The message names the retry for `growth`
(`change="absolute"`) and for `rank` (the frame the per-period rank read,
with `limit` set on that call); `share_of` has none to name.

`place_levels_mixed` fires when `rank`, an `aggregate` whose `group_by`
leaves a place column out, or either side of a `share_of` reads a frame
carrying that column's `{role}_type` enrichment whose several places
share no level of `common.place_type.PLACE_LEVELS`
(`common.place_type.at_one_level`): the frame names no place, so it holds
the world, regions, World Bank groups, countries and their subdivisions
side by side. A type outside `PLACE_LEVELS`, such as `Place`,
`ProvisionalNode` or `UNGeoRegion`, names no level, so places sharing only
that type are refused, and one place alone passes whatever its type. It is
retryable over a `filter` result that keeps one level (`place_level` on the
type column).

`service_unavailable` fires when the process's frame row budget
(`common.row_budget`) has no room for a verb's input rows
(`frame_ref_resolution.resolve_input_frames`). It is the one
`retry_after_delay` code: the same call succeeds once other calls release
their rows.

`internal_error` is the registration-site catch-all's code (see
`server.py`): every other code names something a specific, anticipated
failure produced; `internal_error` names the absence of one. Every other
refusal `compare`, `growth`, `rank`, `share_of`, `join`, and `filter` can
reach reuses one of the existing codes/factories where the shape fits.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Self

from common.place_type import PLACE_LEVELS
from common.refusal import internal_error_message, refusal_meta, refusal_text
from common.row_budget import ROW_BUDGET_RETRY_AFTER_S
from dfo_frame_grammar.grammar import UnitRequest
from fastmcp.tools import ToolResult

from analysis_mcp.disambiguation import (
    MAX_LABEL_CHARS,
    DisambiguationCandidate,
    DisambiguationPayload,
)
from analysis_mcp.units import converted_unit_label, unit_json, unit_name


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
    JOIN_OUTPUT_COLLISION = "join_output_collision"
    JOIN_NO_COMMON_KEY_COLUMNS = "join_no_common_key_columns"
    DERIVED_INPUT_UNUSABLE = "derived_input_unusable"
    DERIVED_INPUT_UNRECOVERABLE = "derived_input_unrecoverable"
    DERIVED_INPUT_ORDINAL = "derived_input_ordinal"
    PLACE_LEVELS_MIXED = "place_levels_mixed"
    SERVICE_UNAVAILABLE = "service_unavailable"
    INTERNAL_ERROR = "internal_error"


# `do_not_retry` codes share one shape: nothing about the call's arguments
# was wrong, so retrying unchanged reaches the identical failure.
# `non_finite_result`/`internal_error`: no specific argument to change.
# `mixed_time_grain`: the frame `frame_ref` names doesn't change between calls.
# `derived_input_unrecoverable`: a result from an earlier engine whose
# period labels misstate its years, so only a fresh run from the source data
# fixes it.
# `derived_input_ordinal` stays `correct_and_retry`: `growth` retries with
# `change="absolute"` over the identical frame_ref, and `share_of` and `rank`
# retry with a frame_ref that isn't itself a per-period rank's own output -- the
# same "a different input fixes this" shape `derived_input_unusable` has.
# `frame_integrity_failed`: a corrupted or tampered stored object, so the
# identical frame_ref reaches the identical bytes.
# `service_unavailable` is `retry_after_delay`: the identical call fits once
# other calls release their frame rows.
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
    ToolErrorCode.JOIN_OUTPUT_COLLISION: RetryMode.CORRECT_AND_RETRY,
    ToolErrorCode.JOIN_NO_COMMON_KEY_COLUMNS: RetryMode.CORRECT_AND_RETRY,
    ToolErrorCode.DERIVED_INPUT_UNUSABLE: RetryMode.CORRECT_AND_RETRY,
    ToolErrorCode.DERIVED_INPUT_UNRECOVERABLE: RetryMode.DO_NOT_RETRY,
    ToolErrorCode.DERIVED_INPUT_ORDINAL: RetryMode.CORRECT_AND_RETRY,
    ToolErrorCode.PLACE_LEVELS_MIXED: RetryMode.CORRECT_AND_RETRY,
    ToolErrorCode.SERVICE_UNAVAILABLE: RetryMode.RETRY_AFTER_DELAY,
    ToolErrorCode.INTERNAL_ERROR: RetryMode.DO_NOT_RETRY,
}
if set(_RETRY_MODE_BY_CODE) != set(ToolErrorCode):
    # A plain `raise`, not `assert` -- `python -O` strips asserts, and this
    # import-time guard should hold even under `-O`.
    raise AssertionError("every ToolErrorCode needs exactly one _RETRY_MODE_BY_CODE entry")

# Caps the candidates a disambiguation payload carries in a refusal's `fix`,
# the same bound as `data_mcp.tool_errors.ENTITY_CANDIDATE_CAP`.
_DISAMBIGUATION_CANDIDATE_CAP = 10


# How the `filter` refusal over a converted unit tells a caller to get a
# frame with one unit.
_REFETCH_ONE_UNIT = (
    "call get_data again with scope.facets binding a single unit, then retry with the new frame_ref"
)


def _quoted_unit(unit: str | UnitRequest) -> str:
    """`unit` as a message quotes it: a ref as its `repr`, a converted unit
    as its label's."""
    return repr(unit_name(unit))


def _pin_values(units: Sequence[str | UnitRequest]) -> str:
    """`units` as a message lists them for a retry: each as the JSON value
    the unit-pin parameter accepts, a ref as a JSON string and a converted
    unit as its request object."""
    return ", ".join(json.dumps(unit_json(unit)) for unit in units)


def _candidate(value: str | UnitRequest) -> DisambiguationCandidate:
    """A unit as a candidate whose `id` is the value the unit-pin parameter
    accepts: a ref as its string, a converted unit as its request object."""
    if isinstance(value, str):
        return DisambiguationCandidate(id=value, label=value[:MAX_LABEL_CHARS])
    return DisambiguationCandidate(id=value.as_json(), label=converted_unit_label(value))


def _disambiguation_payload(candidates: Sequence[str | UnitRequest]) -> dict[str, Any] | None:
    """A `DisambiguationPayload`, dumped to JSON, or `None` if `candidates`
    is empty. `DisambiguationPayload.candidates` has `min_length=1`, and a
    frame whose every declared unit is `None` (see `UnitNotPresentError`)
    has zero candidates to offer, so an unconditional build would raise while
    constructing the refusal. Every candidate label is truncated to
    `MAX_LABEL_CHARS`, the bound `DisambiguationCandidate.label` enforces,
    since a unit or facet value has no length ceiling of its own before it
    reaches here.
    """
    if not candidates:
        return None
    capped = candidates[:_DISAMBIGUATION_CANDIDATE_CAP]
    payload = DisambiguationPayload(candidates=tuple(_candidate(value) for value in capped))
    return payload.model_dump(mode="json")


@dataclass(frozen=True, kw_only=True)
class ToolRefusal:
    """A tool's structured refusal, built by one of its typed factories."""

    code: ToolErrorCode
    message: str
    fix: dict[str, Any]
    retry_after_s: float | None = None

    @property
    def retry_mode(self) -> RetryMode:
        return _RETRY_MODE_BY_CODE[self.code]

    @classmethod
    def op_spec_invalid(cls, *, detail: str) -> Self:
        return cls(
            code=ToolErrorCode.OP_SPEC_INVALID,
            message=f"The op-spec was invalid: {detail}. Correct it and retry.",
            fix={"detail": detail},
        )

    @classmethod
    def stat_var_unresolved(cls, *, reason: str, distinct_stat_vars: list[str]) -> Self:
        return cls(
            code=ToolErrorCode.STAT_VAR_UNRESOLVED,
            message=f"Could not resolve a single StatVar to aggregate: {reason}.",
            fix={"reason": reason, "distinct_stat_vars": distinct_stat_vars},
        )

    @classmethod
    def role_ambiguous(cls, *, parameter: str, columns: list[str], available: list[str]) -> Self:
        """`parameter` names the calling verb's own column-list parameter
        this refusal is about (`group_by`, `entity_by`, `rank_by`,
        `join_keys`, `predicates`, or an `entity` binding such as
        `left_entity`) -- never hardcoded to one verb's own vocabulary,
        since every verb but `aggregate` reaches this factory too.
        """
        return cls(
            code=ToolErrorCode.ROLE_AMBIGUOUS,
            message=(
                f"{parameter} names column(s) this frame has no grain for: {columns}. "
                f"Available: {available}."
            ),
            fix={"parameter": parameter, "columns": columns, "available": available},
        )

    @classmethod
    def operation_uncovered(cls, *, aggregation: str, supported: list[str]) -> Self:
        return cls(
            code=ToolErrorCode.OPERATION_UNCOVERED,
            message=(f"{aggregation!r} is not a supported aggregation. Supported: {supported}."),
            fix={"aggregation": aggregation, "supported": supported},
        )

    @classmethod
    def frame_too_large(cls, *, row_count: int, ceiling: int) -> Self:
        return cls(
            code=ToolErrorCode.FRAME_TOO_LARGE,
            message=(
                f"The input frame has {row_count} rows, over the {ceiling}-row ceiling. "
                "Narrow the input (a smaller frame_ref) and retry."
            ),
            fix={"row_count": row_count, "ceiling": ceiling},
        )

    @classmethod
    def frame_too_large_before_decode(cls, *, reason: str) -> Self:
        """Reuses `frame_too_large`'s own code and retry posture: the
        durable store refused a stored object before decoding it, over one
        of its own encoded-byte, row, column, or row-group uncompressed-size
        ceilings. `reason` is the store's own description of which ceiling
        and by how much -- unlike `frame_too_large` itself, there is no
        single `row_count`/`ceiling` pair that fits every one of those
        shapes.
        """
        return cls(
            code=ToolErrorCode.FRAME_TOO_LARGE,
            message=(
                f"The stored object was refused before it could be read: {reason}. Narrow "
                "the input (a smaller frame_ref) and retry."
            ),
            fix={"reason": reason},
        )

    @classmethod
    def frame_too_large_on_write(cls, *, reason: str) -> Self:
        """Reuses `frame_too_large`'s own code and retry posture: the
        durable store refused to write this call's own RESULT, over its
        row ceiling, before any object was created. Unlike
        `frame_too_large_before_decode` (an already-stored object, refused
        on a later read, retryable with a narrower `frame_ref`), there is no
        stored reference here to narrow -- the fix is a smaller result,
        which only narrowing this call's own inputs can produce. `reason` is
        the store's own description of the row count and its ceiling.
        """
        return cls(
            code=ToolErrorCode.FRAME_TOO_LARGE,
            message=(
                f"This call's own result was too large to store: {reason}. Narrow the "
                "inputs to this operation (a smaller frame_ref, a coarser group_by, or an "
                "entity/predicate that selects fewer rows) and retry -- no result was stored."
            ),
            fix={"reason": reason},
        )

    @classmethod
    def unit_heterogeneous(cls, *, parameter: str, units: Sequence[str | UnitRequest]) -> Self:
        """`parameter` names the calling verb's own unit-pin parameter
        (`unit`, or, for a two-sided verb, both sides' own names together,
        e.g. `"left_unit or right_unit"`), since `compare`/`share_of`/`join`
        have no single `unit` field.

        `filter` refuses `category_match` on a converted unit, so when
        `units` holds one the message offers only the pin, listing each
        unit as the JSON value `parameter` accepts.
        """
        fix: dict[str, Any] = {"parameter": parameter, "units": [unit_json(u) for u in units]}
        disambiguation = _disambiguation_payload(units)
        if disambiguation is not None:
            fix["disambiguation"] = disambiguation
        names = [unit_name(unit) for unit in units]
        if all(isinstance(unit, str) for unit in units):
            retry = (
                f"Retry with `{parameter}` set to one of these to disambiguate, or filter this "
                "frame_ref with a category_match predicate on unit naming one of these "
                "values first, then retry with the resulting pair as frame_ref."
            )
        else:
            retry = f"Retry with `{parameter}` set to one of these values: {_pin_values(units)}."
        return cls(
            code=ToolErrorCode.UNIT_HETEROGENEOUS,
            message=f"The input frame carries more than one unit: {names}. {retry}",
            fix=fix,
        )

    @classmethod
    def unit_not_present(
        cls,
        *,
        parameter: str,
        pinned_unit: str | UnitRequest,
        units: Sequence[str | UnitRequest],
    ) -> Self:
        """`parameter` names the calling verb's own unit-pin parameter that
        `pinned_unit` was passed to. `unit_heterogeneous` documents the
        handling of a converted unit in `units`."""
        fix: dict[str, Any] = {
            "parameter": parameter,
            "pinned_unit": unit_json(pinned_unit),
            "units": [unit_json(unit) for unit in units],
        }
        disambiguation = _disambiguation_payload(units)
        if disambiguation is not None:
            fix["disambiguation"] = disambiguation
        names = [unit_name(unit) for unit in units]
        if all(isinstance(unit, str) for unit in units):
            retry = "Retry with one of these instead."
        else:
            retry = f"Retry with `{parameter}` set to one of these values: {_pin_values(units)}."
        return cls(
            code=ToolErrorCode.UNIT_HETEROGENEOUS,
            message=(
                f"{parameter}={_quoted_unit(pinned_unit)} is not one of the units this frame "
                f"declares: {names}. {retry}"
            ),
            fix=fix,
        )

    @classmethod
    def facet_heterogeneous(cls, *, axis: str, values: list[str]) -> Self:
        fix: dict[str, Any] = {"axis": axis, "values": values}
        disambiguation = _disambiguation_payload(values)
        if disambiguation is not None:
            fix["disambiguation"] = disambiguation
        return cls(
            code=ToolErrorCode.FACET_HETEROGENEOUS,
            message=(
                f"The input frame carries more than one declared {axis!r}: {values}. "
                "There is no way to pin this from a parameter yet -- retry with a "
                "frame_ref already scoped to one of these."
            ),
            fix=fix,
        )

    @classmethod
    def coverage_insufficient(
        cls, *, realised_coverage_fraction: float, floor: float, skipped_cell_count: int
    ) -> Self:
        return cls(
            code=ToolErrorCode.COVERAGE_INSUFFICIENT,
            message=(
                f"Only {realised_coverage_fraction:.1%} of the expected grid is present "
                f"({skipped_cell_count} cell(s) missing), below the {floor:.0%} floor. "
                "Narrow group_by to a coarser grain and retry."
            ),
            fix={
                "realised_coverage_fraction": realised_coverage_fraction,
                "floor": floor,
                "skipped_cell_count": skipped_cell_count,
            },
        )

    @classmethod
    def non_finite_result(cls, *, group_key: dict[str, str], aggregation: str) -> Self:
        return cls(
            code=ToolErrorCode.NON_FINITE_RESULT,
            message=(
                f"The {aggregation} for group {group_key} is not finite (overflow). "
                "Retrying the same call would reach the same result; narrow the input instead."
            ),
            fix={"group_key": group_key, "aggregation": aggregation},
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
    def mixed_time_grain(cls, *, grains: list[str], periods: list[str]) -> Self:
        """`grains` names every distinct grain the resolved input's own
        `time_period` column carries, in `year`/`month`/`day` order, and
        `periods` the cells that conflict: one per grain for a grain mix, or
        the spans and a point beside them for a span mix within one grain.
        """
        return cls(
            code=ToolErrorCode.MIXED_TIME_GRAIN,
            message=(
                f"The input frame's own time_period column mixes {periods} "
                f"(grains {grains}). Every verb here reads a frame whose periods are all "
                "points at one grain, or exactly one span such as 2018/2022, so this frame "
                "cannot be resolved. Retrying the identical frame_ref reaches the identical "
                "column; retry with a frame_ref whose time_period values are points at one "
                "grain, or one span alone."
            ),
            fix={"grains": grains, "periods": periods},
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
    def comparison_periods_disjoint(
        cls, *, left_periods: list[str], right_periods: list[str]
    ) -> Self:
        return cls(
            code=ToolErrorCode.COMPARISON_PERIODS_DISJOINT,
            message=(
                f"The two sides share no overlapping time_period. Left side has {left_periods}; "
                f"right side has {right_periods}. Retry with a frame_ref (or entity binding) "
                "whose periods actually overlap the other side's."
            ),
            fix={"left_periods": left_periods, "right_periods": right_periods},
        )

    @classmethod
    def cross_frame_unit_mismatch(
        cls,
        *,
        left_label: str,
        left_unit: str | UnitRequest,
        right_label: str,
        right_unit: str | UnitRequest,
    ) -> Self:
        """Two independently-resolved sides (`compare`'s left/right,
        `share_of`'s numerator/denominator) carry different units. Distinct
        from `unit_heterogeneous`, which fires when *one* frame itself
        carries more than one unit -- here each side resolved to exactly
        one unit on its own, and it's the pairing across sides that's
        unsafe to combine. When either side holds a converted unit, the
        message asks for both sides again in one unit.
        """
        if isinstance(left_unit, UnitRequest) or isinstance(right_unit, UnitRequest):
            retry = (
                "Call get_data for both sides with scope.facets binding the same unit, "
                "then retry with the new frame_refs."
            )
        else:
            retry = "Retry with both sides pinned to the same unit."
        return cls(
            code=ToolErrorCode.CROSS_FRAME_UNIT_MISMATCH,
            message=(
                f"{left_label} resolved to unit {_quoted_unit(left_unit)}, {right_label} to "
                f"{_quoted_unit(right_unit)} -- combining values across different units is "
                f"silently wrong, not merely inconvenient. {retry}"
            ),
            fix={
                left_label: unit_json(left_unit),
                right_label: unit_json(right_unit),
            },
        )

    @classmethod
    def cross_frame_unit_unbound(
        cls, *, unbound_label: str, bound_label: str, bound_unit: str | UnitRequest
    ) -> Self:
        """One side's own rows leave `unit` unbound entirely, while the
        other resolved to a real declared unit. Distinct from
        `cross_frame_unit_mismatch`, which fires when both sides resolve to
        a real, differing unit. Here one side has no declared unit at all,
        so there is no second unit to compare.
        """
        return cls(
            code=ToolErrorCode.CROSS_FRAME_UNIT_UNBOUND,
            message=(
                f"{unbound_label}'s own rows leave `unit` unbound entirely; {bound_label} "
                f"resolved to unit {_quoted_unit(bound_unit)}. Retry with a {unbound_label} whose "
                "own rows actually declare a unit."
            ),
            fix={
                "unbound_side": unbound_label,
                "bound_side": bound_label,
                "bound_unit": unit_json(bound_unit),
            },
        )

    @classmethod
    def cross_frame_facet_mismatch(
        cls,
        *,
        left_label: str,
        right_label: str,
        axis: str,
        left_ref: str,
        right_ref: str,
    ) -> Self:
        """Two independently-resolved sides (`compare`'s left/right,
        `share_of`'s numerator/denominator) disagree on one facet axis.
        Distinct from `facet_heterogeneous`, which fires when one side's own
        slice carries more than one declared value. Here each side is
        internally uniform, and it's the pairing across sides that's unsafe.
        """
        return cls(
            code=ToolErrorCode.CROSS_FRAME_FACET_MISMATCH,
            message=(
                f"{left_label} resolved facet {axis!r} to {left_ref!r}, {right_label} to "
                f"{right_ref!r} -- combining values across different {axis!r} selections is "
                "silently wrong, not merely inconvenient. Retry with both sides scoped to a "
                "frame_ref that agrees on this facet."
            ),
            fix={"axis": axis, left_label: left_ref, right_label: right_ref},
        )

    @classmethod
    def growth_gap_year(
        cls,
        *,
        window_start: str,
        window_end: str,
        entity_count: int,
        missing_start: int,
        missing_end: int,
        suggested_window: list[str],
        suggested_window_entities: int,
    ) -> Self:
        """No entity has a row in both endpoint years. The counts cover every
        entity the input holds. `suggested_window` is a window
        `suggested_window_entities` entities hold both ends of, `[]` when no
        entity holds two years inside the requested window.
        """
        if suggested_window:
            retry = (
                f"{suggested_window_entities} entit(y/ies) have rows in both "
                f"{suggested_window[0]} and {suggested_window[1]}; retry with window_start and "
                "window_end set to those years."
            )
        else:
            retry = (
                "No entity has rows in two years inside the window; retry with window_start "
                "and window_end set to years the entities report."
            )
        return cls(
            code=ToolErrorCode.GROWTH_GAP_YEAR,
            message=(
                f"No entity has rows in both {window_start} and {window_end}, and growth reads "
                f"each entity's value in those two endpoint years. Of {entity_count} "
                f"entit(y/ies), {missing_start} have no row in {window_start} and {missing_end} "
                f"have no row in {window_end}. {retry}"
            ),
            fix={
                "window_start": window_start,
                "window_end": window_end,
                "entity_count": entity_count,
                "missing_start": missing_start,
                "missing_end": missing_end,
                "suggested_window": suggested_window,
                "suggested_window_entities": suggested_window_entities,
            },
        )

    @classmethod
    def growth_non_finite_result(
        cls, *, entity_count: int, zero_or_sign_crossing_base: int, non_finite: int
    ) -> Self:
        """Every entity holding both endpoints was excluded for a growth value
        that is undefined or not finite. `non_finite_result`'s own code,
        with growth's per-cause entity counts in place of one group key."""
        causes = []
        if zero_or_sign_crossing_base:
            causes.append(
                f"{zero_or_sign_crossing_base} have a zero or sign-crossing base, which leaves "
                "a relative change undefined (change='absolute' is defined there)"
            )
        if non_finite:
            causes.append(f"{non_finite} compute to a non-finite value")
        return cls(
            code=ToolErrorCode.NON_FINITE_RESULT,
            message=(
                f"No entity has a finite growth value. Of the {entity_count} entit(y/ies) with "
                f"rows in both endpoint years, {' and '.join(causes)}. Retrying the same call "
                "would reach the same result; change the request instead."
            ),
            fix={
                "entity_count": entity_count,
                "zero_or_sign_crossing_base": zero_or_sign_crossing_base,
                "non_finite": non_finite,
            },
        )

    @classmethod
    def share_of_grain_mismatch(
        cls,
        *,
        numerator_periods: list[str],
        denominator_periods: list[str],
        numerator_coverage_fraction: float,
        denominator_coverage_fraction: float,
        matched_periods: list[str],
        key: dict[str, str] | None = None,
    ) -> Self:
        """`matched_periods` is the sorted intersection of the two sides'
        own periods -- possibly empty. When non-empty, it names an in-toolkit
        repair: `filter` each side's frame_ref with a `category_match`
        predicate on `time_period` naming exactly these values, then retry
        `share_of` on the two resulting pairs. `key` names the offending
        `numerator_by` key for a keyed call, `{}` (the default) for an
        un-keyed one, since the identical-period-set rule applies per key.
        """
        key = key or {}
        repair = (
            f"Both sides do share {matched_periods}: filter each side's frame_ref with a "
            "category_match predicate on time_period naming these values, then retry "
            "share_of on the two resulting pairs."
            if matched_periods
            else "The two sides share no periods at all -- narrow either side's frame_ref "
            "or entity binding to a source that actually overlaps the other's, and retry."
        )
        # Both sides empty is a distinct fault from a genuine mismatch: two
        # empty period lists are trivially equal as sets, so
        # `numerator_periods != denominator_periods` alone never catches this
        # (see `run_share_of`'s own `or not numerator_periods` guard). The
        # generic message below would otherwise read "The denominator's
        # periods [] do not match the numerator's periods []" -- two
        # identical empty lists framed as disagreeing, never naming the
        # actual fault: neither side's own entity binding matched any row at
        # all.
        if not numerator_periods and not denominator_periods:
            message = (
                "Neither the numerator's nor the denominator's own entity binding matched "
                "any row -- both sides summed to zero periods, not to two disagreeing period "
                "sets. share_of never reports one side's own coverage and proceeds on the "
                "other's. Retry with a numerator_entity and/or denominator_entity that "
                "actually narrows to rows this frame_ref carries, or a frame_ref that does."
            )
        else:
            message = (
                f"The denominator's periods {denominator_periods} do not match the "
                f"numerator's periods {numerator_periods}. share_of never reports one side's "
                f"own coverage and proceeds on the other's. {repair}"
            )
        return cls(
            code=ToolErrorCode.SHARE_OF_GRAIN_MISMATCH,
            message=message,
            fix={
                "numerator_periods": numerator_periods,
                "denominator_periods": denominator_periods,
                "numerator_coverage_fraction": numerator_coverage_fraction,
                "denominator_coverage_fraction": denominator_coverage_fraction,
                "matched_periods": matched_periods,
                "key": key,
            },
        )

    @classmethod
    def share_of_zero_denominator(cls, *, time_period: str) -> Self:
        """A summed-to-exactly-zero denominator is retryable (a different
        `frame_ref` or entity binding for the denominator can produce a
        nonzero sum) -- distinct from `non_finite_result`, where retrying
        the identical call reaches the identical, genuine overflow.
        """
        return cls(
            code=ToolErrorCode.SHARE_OF_ZERO_DENOMINATOR,
            message=(
                f"The denominator summed to exactly zero for time_period {time_period!r}, "
                "making the share undefined. Retry with a denominator_frame_ref or "
                "denominator_entity that sums to a nonzero value for this period."
            ),
            fix={"time_period": time_period},
        )

    @classmethod
    def share_of_key_unknown(
        cls, *, column: str, numerator_by: list[str], denominator_by_keys: list[str]
    ) -> Self:
        """`denominator_by` is non-empty but leaves `column` (one of
        `numerator_by`'s own columns) out -- a partial mapping has no
        defined pairing for the column(s) it omits."""
        return cls(
            code=ToolErrorCode.SHARE_OF_KEY_UNKNOWN,
            message=(
                f"denominator_by does not cover numerator_by column {column!r} -- a "
                f"partial mapping has no defined pairing for the column(s) it omits. "
                f"denominator_by names: {denominator_by_keys}; numerator_by: "
                f"{numerator_by}. Retry with denominator_by either empty (a pooled "
                "denominator, one total shared by every key) or naming every "
                "numerator_by column."
            ),
            fix={
                "column": column,
                "numerator_by": numerator_by,
                "denominator_by_keys": denominator_by_keys,
            },
        )

    @classmethod
    def share_of_key_unmatched(
        cls, *, key: dict[str, str], available: list[dict[str, str]]
    ) -> Self:
        """A numerator key, realised from its own frame's rows, has no
        matching denominator group under `denominator_by`'s own mapping.
        `available` lists every denominator key this call did realise."""
        return cls(
            code=ToolErrorCode.SHARE_OF_KEY_UNMATCHED,
            message=(
                f"No denominator group matches the numerator key {key}. Available "
                f"denominator keys: {available}. Retry with a denominator_frame_ref (or "
                "denominator_entity) whose own rows actually carry this entity under "
                "denominator_by's own mapping."
            ),
            fix={"key": key, "available": available},
        )

    @classmethod
    def filter_predicate_type_mismatch(
        cls, *, column: str, predicate_kind: str, actual_type: str
    ) -> Self:
        return cls(
            code=ToolErrorCode.FILTER_PREDICATE_TYPE_MISMATCH,
            message=(
                f"A {predicate_kind!r} predicate on column {column!r} doesn't match that "
                f"column's own declared cell type ({actual_type}). Retry with the correct "
                "predicate kind for this column, or a different column."
            ),
            fix={"column": column, "predicate_kind": predicate_kind, "actual_type": actual_type},
        )

    @classmethod
    def filter_predicate_value_type_mismatch(
        cls, *, column: str, actual_type: str, expected_type: str
    ) -> Self:
        """An `ordered_threshold` predicate targeting a legal column, whose
        own `value` carries the wrong operand type for it -- reuses
        `filter_predicate_type_mismatch`'s own code and retry posture: a
        different operand type for the same column is the same kind of
        correction a wrong predicate kind or column is.
        """
        operand_words = {
            "float": "a number",
            "str": "a period string",
            "point": "one period such as 2018",
            "interval": "an interval such as 2018/2022",
        }
        return cls(
            code=ToolErrorCode.FILTER_PREDICATE_TYPE_MISMATCH,
            message=(
                f"An 'ordered_threshold' predicate's value on column {column!r} must be "
                f"{operand_words.get(expected_type, expected_type)}, not "
                f"{operand_words.get(actual_type, actual_type)}. Retry with "
                f"{operand_words.get(expected_type, expected_type)} for this column."
            ),
            fix={"column": column, "actual_type": actual_type, "expected_type": expected_type},
        )

    @classmethod
    def filter_unit_conversion_unsupported(cls, *, column: str, predicate_kind: str) -> Self:
        return cls(
            code=ToolErrorCode.FILTER_UNIT_CONVERSION_UNSUPPORTED,
            message=(
                f"A {predicate_kind!r} predicate on column {column!r} cannot match this "
                "frame's units: its unit cells hold a converted unit, and category_match "
                "compares unit strings. Predicates on value, time_period and role columns "
                "still apply to this frame. Retry without the unit predicate, or, to select "
                f"one unit, {_REFETCH_ONE_UNIT}."
            ),
            fix={"column": column, "predicate_kind": predicate_kind, "actual_type": "UnitRequest"},
        )

    @classmethod
    def join_output_collision(cls, *, left_stat_var: str, right_stat_var: str) -> Self:
        return cls(
            code=ToolErrorCode.JOIN_OUTPUT_COLLISION,
            message=(
                f"Both sides resolved to StatVar {left_stat_var!r} with the same unit and "
                "facets -- the two rows a matched key would emit are indistinguishable, "
                "which Frame.build cannot represent. Retry with a right_frame_ref (or entity "
                "binding) whose stat_var, unit, or facets differ from the left side's."
            ),
            fix={"left_stat_var": left_stat_var, "right_stat_var": right_stat_var},
        )

    @classmethod
    def join_no_common_key_columns(
        cls, *, left_columns: list[str], right_columns: list[str]
    ) -> Self:
        return cls(
            code=ToolErrorCode.JOIN_NO_COMMON_KEY_COLUMNS,
            message=(
                f"join_keys was left empty, and auto-detection found no role column both "
                f"sides declare: left declares {left_columns}, right declares "
                f"{right_columns}. Joining on time_period alone would pair unrelated "
                "rows. Retry with a right_frame_ref whose own role columns actually "
                "overlap the left side's, or an explicit join_keys naming a column both "
                "sides genuinely share."
            ),
            fix={"left_columns": left_columns, "right_columns": right_columns},
        )

    @classmethod
    def derived_input_unusable(cls, *, span: str) -> Self:
        return cls(
            code=ToolErrorCode.DERIVED_INPUT_UNUSABLE,
            message=(
                f"This frame_ref names a result whose rows carry the span {span}. growth "
                "reads one row per period, so it has no years to window over a span. Retry "
                "growth over the frame this result was computed from."
            ),
            fix={"span": span},
        )

    @classmethod
    def derived_input_unrecoverable(cls, *, engine_version: str) -> Self:
        """`engine_version` is the analysis-mcp engine that computed the
        input, as its trace's `execution_identity` records it."""
        return cls(
            code=ToolErrorCode.DERIVED_INPUT_UNRECOVERABLE,
            message=(
                f"This frame_ref names a result computed by analysis-mcp engine "
                f"{engine_version}, an earlier engine whose period labels do not state the "
                "years they cover: a row labelled 2022 may total 2018 to 2022. No verb here "
                "can read its periods, and retrying the identical frame_ref reaches the "
                "identical result. Run the analysis again from the source data."
            ),
            fix={"engine_version": engine_version},
        )

    @classmethod
    def derived_input_ordinal(cls, *, reason: str) -> Self:
        return cls(
            code=ToolErrorCode.DERIVED_INPUT_ORDINAL,
            message=(
                "This frame_ref names a derived result whose own value column holds a rank "
                f"position, not a measured value: {reason}. A ratio, a relative growth rate or "
                "a ranking over a rank position is not a meaningful number. growth retries with "
                'change="absolute" over the identical frame_ref -- a rank moves by places, not '
                "by percentage; rank retries over the frame_ref the per-period rank itself "
                "read, with limit set on that one call to keep a top or bottom N; share_of has "
                "no such retry and needs a frame_ref that isn't itself a per-period rank's own "
                "output."
            ),
            fix={"reason": reason},
        )

    @classmethod
    def place_levels_mixed(cls, *, column: str, types: Sequence[str]) -> Self:
        """`types` are the distinct cells of `column`, sorted and capped by
        the engine that found them."""
        return cls(
            code=ToolErrorCode.PLACE_LEVELS_MIXED,
            message=(
                f"This frame's places share no level a filter can keep: {column!r} holds "
                f"{', '.join(types)}, and no one of {', '.join(sorted(PLACE_LEVELS))} is "
                "among every place's types. A frame that names no place holds the world, "
                "regions, World Bank groups, countries and their subdivisions side by side, "
                "and ranking or summing them together counts a place beside the places inside "
                "it. Filter the frame first with a place_level predicate on "
                f"{column!r} naming one of those levels, such as Country, then retry with that "
                "result's frame_ref. A type outside those levels, such as Place or "
                "ProvisionalNode, names no level, and no place_level filter keeps the places "
                "that carry it alone."
            ),
            fix={"column": column, "types": list(types)},
        )

    @classmethod
    def row_budget_full(cls, *, requested: int, free: int, max_rows: int) -> Self:
        return cls(
            code=ToolErrorCode.SERVICE_UNAVAILABLE,
            message=(
                f"This server is already working on as many frame rows as its memory allows "
                f"at once ({max_rows:,}). This call's inputs need room for {requested:,} rows "
                f"and {free:,} are free. Retry after {ROW_BUDGET_RETRY_AFTER_S:.1f}s."
            ),
            fix={
                "service": "analysis-mcp",
                "rows_requested": requested,
                "rows_free": free,
                "rows_in_flight_max": max_rows,
            },
            retry_after_s=ROW_BUDGET_RETRY_AFTER_S,
        )

    @classmethod
    def internal_error(cls, *, tool: str, trace_id: str) -> Self:
        """The registration-site catch-all's own refusal (see `server.py`'s
        module docstring), for an exception no verb's own typed handling
        anticipated. The exception's text stays in the log line `trace_id`
        names (`common.refusal.log_unexpected_error`): it can hold a bucket
        path or a service account, and it gives a retry nothing to act on,
        which is why this code is `do_not_retry`.
        """
        return cls(
            code=ToolErrorCode.INTERNAL_ERROR,
            message=internal_error_message(subject=tool, trace_id=trace_id),
            fix={"tool": tool, "trace_id": trace_id},
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

    `meta` is `common.refusal.refusal_meta` of `code`.
    """
    return ToolResult(
        content=refusal_text(
            message=refusal.message,
            code=refusal.code.value,
            retry_mode=refusal.retry_mode.value,
            fix=refusal.fix,
            retry_after_s=refusal.retry_after_s,
            preamble=preamble,
        ),
        structured_content={
            "code": refusal.code.value,
            "retry_mode": refusal.retry_mode.value,
            # `refusal.message` itself, not just `content`: a caller that
            # reads `structured_content` alone (`executor.classify.
            # parse_tool_failure`, reading the raw MCP payload rather than
            # this module's own `content` text) needs the same prose a
            # human reader of `content` gets, preamble aside.
            "message": refusal.message,
            "fix": refusal.fix,
        }
        | ({} if refusal.retry_after_s is None else {"retry_after_s": refusal.retry_after_s}),
        meta=refusal_meta(code=refusal.code.value),
        is_error=True,
    )
