"""What to do with a tool's refusal.

Every MCP tool answers a failure with a typed refusal carrying a `retry_mode`,
which says whether trying again can help, and a `code`, which says what went
wrong. The executor never corrects a call itself -- a corrected call is a
different plan -- so a refusal is retried when the tool says waiting helps,
handed to the planner when it carries a choice the user has to make, sent
back to discovery when only a fresh discovery can change what a retry would
get (a bare retry would just reproduce the same refusal), and stopped on
otherwise.

The mode decides the transient and refresh cases; the code decides the rest.
Every code either server can answer with sits in exactly one of the five sets
below, and `tests/executor/test_classify.py` reads both servers' enums and
fails when one is missing here -- so adding a server code means picking a
side, not landing in the stop bucket by omission:

- `METHOD_CHOICE_CODES`: the refusal's `fix` is a set of alternatives -- which
  entity, unit, facet, window, coverage or denominator -- and choosing one
  changes the plan. Goes to the planner as `MethodChoiceNeeded`, candidates
  attached. `share_of_key_unmatched` belongs here alongside
  `share_of_zero_denominator` and `share_of_grain_mismatch`. `share_of`
  leaves out and records a key with no denominator group or none sharing a
  period with it, and refuses only when no key has a group or none shares
  a period. Which denominator_frame_ref or entity binding covers the
  missing keys or years is a data-dependent choice, not something the
  compiler could have ruled out.
  `quotient_of_zero_denominator` and `quotient_of_no_key_matched` sit beside
  them for the same reason: `quotient_of` refuses only when every matched
  cell divides by zero, or when no numerator key pairs with a denominator
  key at all, and which denominator, window or `denominator_axis` gives the
  division something to divide by is the planner's to choose.
- `REDISCOVERY_CODES`: the source the step was approved to read is gone or
  has moved on, and only discovering it again, under a new plan revision,
  changes what a retry would get. `upstream_version_changed` says so in its
  mode. `definition_not_held` is `correct_and_retry` on the wire, but the
  approved `spec_id` has left the definition store and nothing this loop
  can do brings it back; its `fix` says `search`, which is discovery.
  `open_role_unavailable` is `correct_and_retry` too. The identity keeps an
  open role the StatVars it elects do not declare, so the same identity
  refuses every time, and its `fix` says `search`.
- `TRANSIENT_CODES`: waiting helps; retried up to the attempt limit.
- `COMPILER_PROVABLE_CODES`: stops, because the compiler already rules them
  out. It writes every column list (`group_by`, `entity_by`, `rank_by`,
  `join_keys`, `right_keys`) from the source's own roles, validates every
  argument against the tool's live schema, checks the capability implements
  the operation kind, writes every join key explicitly, so analysis-mcp has no
  keys to detect (`join_no_common_key_columns`), and writes every
  predicate in the frame grammar's own shape. One of these reaching the
  executor is a compiler bug, not a choice for the user. `share_of_key_unknown`
  belongs here too: the compiler writes `denominator_by` either empty
  (pooled) or total over `numerator_by` (per-axis), so a partial mapping
  reaching the tool means the compiler failed to keep that invariant. So
  does `place_levels_mixed`: `planner.shapes.rejects` refuses a rank, a
  total or a share over a requirement that names no place until a
  place-level filter keeps one level, and only that filter's retrieval asks
  `get_data` for place types. And `composed_unit_sum_unsupported`:
  `planner.shapes` refuses a sum over a quotient (`_rejects_summed_rate`,
  `_rejects_sum_of`, `_rejects_pooled_rate`), and the compiler refuses a
  `share_of` side holding several named entities un-keyed
  (`entity_required`, `share_axis_required`), so a summing `aggregate`,
  `rank` or `sum_of` reading a composed unit, or a `share_of` side summing
  several rows of one, is a plan the door let through.
- `STOP_CODES`: genuine dead ends. Either the tool says `do_not_retry`, or
  the correction it asks for is one the executor may not make and the user
  could not pick from offered candidates -- a different entity, frame,
  ceiling or aggregation is a different plan. `derived_input_ordinal` joins
  `derived_input_unusable`/`derived_input_unrecoverable` here for the same
  reason: each `fix` names a fact (a reason, a span, an engine version), not
  a set of candidates, and the actual correction (a different `frame_ref`,
  `change="absolute"`, or a fresh run from the source data) is a different
  plan for the planner to propose, not a choice the executor offers
  mid-run. `quotient_of_side_unit_unsupported` is the same kind: its `fix`
  names the side and why it gives no unit to compose (a share or a growth
  rate, which is a ratio already; a quotient; a unit-less sum), and
  dividing something else is a different plan. The planner's shape rules and the compiler refuse the sides they can
  read off the plan, but a side's unit and its trace are the frame's own, so
  the code is not compiler-provable. `over_ceiling` sits here as its default, but it is not
  uniform: see the `fix.ceiling_kind` carve-outs below.

`over_ceiling`'s own disposition depends on `fix.ceiling_kind`, which
`classify_refusal` reads directly rather than through the code-only tables
above. `rows`, `bytes` and `requests` ceilings narrow with those scope
members, so they are a method choice: which narrower request to try, same as
any other `METHOD_CHOICE_CODES` member. A `member_cardinality` or
`candidate_window` ceiling is fixed by the resolved spec_id's own five-tuple
and slots -- `scope.entities`/`scope.window` cannot narrow it -- and it is a
function of that spec_id's identity and the index version behind it: a retry
against the same index reproduces the identical refusal, and so does a replan
that rediscovers the same spec by paraphrase search. It stays at
`over_ceiling`'s `STOP_CODES` default, and the turn says so rather than
spending its one replan on a rediscovery that cannot change what a retry
gets. A refusal built without a
`ceiling_kind` at all -- an older upstream, or a raw payload assembled by
hand -- stays `STOP` for the same reason: the executor cannot tell whether
an unnamed ceiling narrows with scope, and guessing wrong spends a replan
reproducing the identical failing call.

`rolling_anchor_unresolved` stops under every `fix.detail`, and the detail
says whose stop it is. `no_qualifying_period` is `resolve_anchor` finding no
year that half of the series hold, and `pairing_shares_no_key` a keyed verb
whose two sides name no entity in common. The user is told of both. Every other
detail says the populations, root frames, key or periods the executor itself sent do
not fit together (`is_anchor_fault`); the run reports that as an
`internal_error`, since nothing the user asked for is wrong.

A stop does not throw the tool's offer away: `Refused.payload` carries the
refusal verbatim, `fix` included. It just does not act on it.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from enum import StrEnum
from typing import cast

from planner.records import StrictModel
from pydantic import Field, JsonValue, ValidationError

METHOD_CHOICE_CODES = frozenset(
    {
        "entity_ambiguous",
        "unit_heterogeneous",
        "facet_heterogeneous",
        "growth_gap_year",
        "comparison_periods_disjoint",
        "share_of_grain_mismatch",
        "cross_frame_unit_mismatch",
        "cross_frame_unit_unbound",
        "cross_frame_facet_mismatch",
        "coverage_insufficient",
        "share_of_zero_denominator",
        "share_of_key_unmatched",
        "quotient_of_zero_denominator",
        "quotient_of_no_key_matched",
    }
)

REDISCOVERY_CODES = frozenset(
    {"upstream_version_changed", "definition_not_held", "open_role_unavailable"}
)

TRANSIENT_CODES = frozenset(
    {
        "upstream_error",
        # data-mcp also answers this for a `get_data` mint POST that got no response
        # (`ToolRefusal.mint_unanswered`): apis may still be running it, and a
        # repeat call either rejoins the operation apis recorded for it or runs
        # the request again. data-mcp, analysis-mcp and presentation-mcp answer it
        # when the process's frame row budget (`common.row_budget`) has no room for
        # a call's rows, which the same call finds once other calls release theirs.
        "service_unavailable",
        # data-mcp: `get_data`'s poll reserve of the call deadline ran out before a
        # mint's 202 operation produced a manifest, with the operation last seen
        # `pending`/`running` or its status unconfirmed. A repeat `get_data` call for
        # the same spec_id and scope resumes that operation while the dataset's
        # coverage is unchanged, or starts a fresh one if coverage has been published
        # since.
        "operation_still_processing",
    }
)

COMPILER_PROVABLE_CODES = frozenset(
    {
        "role_ambiguous",
        "op_spec_invalid",
        "request_invalid",
        "operation_uncovered",
        "join_no_common_key_columns",
        "filter_predicate_type_mismatch",
        "share_of_key_unknown",
        "place_levels_mixed",
        "composed_unit_sum_unsupported",
    }
)

STOP_CODES = frozenset(
    {
        # data-mcp
        "facet_not_available",
        "default_facet_undefined",
        "over_ceiling",
        # The spec mints no dataset as posted: it describes a family with
        # nothing to export (`spec_not_exportable`), or the graph cannot
        # resolve its StatVars (`member_not_resolvable`). `do_not_retry` on
        # the wire, and no scope edit reaches it: a different spec is a
        # different plan.
        "spec_not_exportable",
        "member_not_resolvable",
        "capability_unavailable",
        "entity_not_found",
        "node_not_held",
        # apis cannot sum the family a `get_data` asked to sum (a gross/net
        # pair, a code it cannot decode, sources that disagree). `do_not_retry`
        # on the wire; reading the family separately is a different plan.
        "rollup_not_summable",
        # The skill tools' own refusals. No workflow step calls those tools
        # (`compiler.capabilities.EXCLUDED_TOOLS`).
        "skill_not_found",
        "skill_resource_not_found",
        # analysis-mcp
        "stat_var_unresolved",
        "frame_too_large",
        "non_finite_result",
        "frame_not_held",
        "frame_expired",
        # Both reproduce on the identical frame_ref: one frame mixes time
        # grains, the other no longer reads back as what was stored. A
        # different frame is a different plan, and only the planner makes one.
        "mixed_time_grain",
        "frame_integrity_failed",
        "filter_unit_conversion_unsupported",
        "join_output_collision",
        "derived_input_unusable",
        "derived_input_unrecoverable",
        "derived_input_ordinal",
        "quotient_of_side_unit_unsupported",
        # A verb read a frame fetched with a `breakdown` or a `stat_vars` sum,
        # a rendering dfo-apis' analysis runner reads no root with. Fetching
        # the frame another way is a different plan.
        "rendering_not_recorded",
        # The newest period a run's rolling steps stand on, which the executor
        # resolves and writes each step's periods from. No year is held by half
        # of the series, or the periods are no calendar years. Both are facts of
        # the retrieved data, which the same plan meets again. Every other
        # `rolling_anchor_unresolved` is a fault of the run (`is_anchor_fault`).
        "rolling_anchor_unresolved",
        "rolling_grain_unsupported",
        "internal_error",
    }
)

# How many times one call is attempted before its failure is final: a plan
# step's in `executor.run`, and one call of `harness.mcp_client.McpExecutor`.
MAX_ATTEMPTS_PER_STEP = 3

# Seconds to wait before a retry when the refusal names no delay of its own,
# doubled on each further attempt.
BACKOFF_SECONDS = 0.5

# The longest a refusal's own `retry_after_s` is honoured for, and the longest
# any jittered wait lasts. A server naming an hour does not get to hold a run
# for an hour.
MAX_RETRY_AFTER_SECONDS = 30.0

# How far above `retry_delay` a wait may spread: each wait is drawn uniformly
# between `retry_delay` and this multiple of it, so calls refused together
# (a full row budget refuses every call that arrives while it is full) retry
# at different moments instead of colliding again.
RETRY_JITTER_FACTOR = 2.0

# Draws a number uniformly between its two bounds: `random.uniform` in
# production, a seeded `random.Random(...).uniform` under test.
type Jitter = Callable[[float, float], float]


class Disposition(StrEnum):
    """What the executor does with a refusal."""

    RETRY_LATER = "retry_later"
    REDISCOVER = "rediscover"
    METHOD_CHOICE = "method_choice"
    STOP = "stop"


# What a `correct_and_retry` refusal calls for, by code. Anything not named
# here stops; the exhaustiveness test keeps "not named" from being a new code.
_CORRECTION_DISPOSITIONS: dict[str, Disposition] = (
    dict.fromkeys(METHOD_CHOICE_CODES, Disposition.METHOD_CHOICE)
    | dict.fromkeys(REDISCOVERY_CODES, Disposition.REDISCOVER)
    | dict.fromkeys(COMPILER_PROVABLE_CODES | STOP_CODES, Disposition.STOP)
)

# `over_ceiling` kinds `scope.entities`/`scope.window` can actually narrow --
# these go to `METHOD_CHOICE`. Every other kind, `member_cardinality` and
# `candidate_window` included, falls through to `over_ceiling`'s `STOP_CODES`
# default.
_NARROWABLE_CEILING_KINDS = frozenset({"rows", "bytes", "requests"})


def _ceiling_kind(fix: Mapping[str, JsonValue]) -> object:
    """`fix.get("ceiling_kind")`, or `None` when `fix` names none."""
    return fix.get("ceiling_kind")


class ExecutorCode(StrEnum):
    """The `code` of a refusal the executor mints itself."""

    # A payload that does not fit `ToolFailure` is parsed into this. It is the
    # executor's own code and appears in none of the five sets above, so it
    # always falls through `classify_refusal` to `Disposition.STOP`, whatever
    # mode the broken payload named.
    MALFORMED_REFUSAL = "malformed_refusal"
    # A retrieval asked to sum its family of StatVars returned them unsummed
    # (`executor.run._Run._check_summed_retrieval`). It travels with
    # `do_not_retry`, which `classify_refusal` reads as `Disposition.STOP`,
    # because the same plan draws the same unsummed frame from the same
    # deployment.
    FAMILY_NOT_SUMMED = "family_not_summed"
    # A rolling step's resolved start falls before the first year its dataset
    # holds (`executor.run._Run._check_rolled_start`). It travels with
    # `do_not_retry`, since the same plan resolves the same start over the
    # same data.
    ROLLING_WINDOW_OUTSIDE_DATASET = "rolling_window_outside_dataset"
    # The newest period a group of rolling steps resolves is before the one
    # the plan's rolling years state (`executor.run._Run._check_anchor_held`).
    # It travels with `do_not_retry`, since the same plan resolves the same
    # period over the same data, and the user asks again to plan from it.
    ROLLING_ANCHOR_BEHIND_PLAN = "rolling_anchor_behind_plan"
    # A fault of the run itself, as both servers report their own.
    INTERNAL_ERROR = "internal_error"
    # The run was stopped before it finished.
    CANCELLED = "cancelled"
    # A run still unfinished after its last resumption or past its age bound
    # (`executor.ledger`), by the stage it stood at.
    PUBLICATION_ABANDONED = "publication_abandoned"
    COMPUTATION_ABANDONED = "computation_abandoned"
    # A publication held for a later try that names no fault of its own.
    PUBLICATION_UNFINISHED = "publication_unfinished"
    # The publisher raised (`executor.completion`).
    PUBLICATION_ERROR = "publication_error"
    # The ledger's store could not admit the run, or could not record its
    # progress (`executor.completion.unrecorded_hand_back`).
    RUN_NOT_STARTED = "run_not_started"
    LEDGER_UNAVAILABLE = "ledger_unavailable"


# The codes that say the harness wrote a call wrong or a server failed of
# itself, so nothing the user asked for is wrong. They are a compiler-provable
# code, a payload that did not parse, and an internal error, the run's own
# (`executor.run._Run._anchor_fault`) or a server's.
HARNESS_FAULT_CODES = COMPILER_PROVABLE_CODES | {
    ExecutorCode.MALFORMED_REFUSAL,
    ExecutorCode.INTERNAL_ERROR,
}


# analysis-mcp's code for a newest period that is not established, and the
# `fix.detail`s of it that state a fact of the data: no year that half of the
# series hold, and a pairing whose two sides name no entity in common.
ROLLING_ANCHOR_UNRESOLVED = "rolling_anchor_unresolved"
PAIRING_SHARES_NO_KEY = "pairing_shares_no_key"
_DATA_DETAILS = frozenset({"no_qualifying_period", PAIRING_SHARES_NO_KEY})


class ToolFailure(StrictModel):
    """A tool's refusal, typed off the raw MCP payload.

    `code` and `retry_mode` are read against the five sets above, not
    against a closed enum here -- the served vocabulary belongs to
    `data_mcp.tool_errors.ToolErrorCode` and `analysis_mcp.tool_errors.ToolErrorCode`,
    and this model exists to stop `classify.py` and its callers from
    indexing an untyped mapping, not to duplicate those enums.
    """

    code: str
    message: str
    retry_mode: str
    fix: dict[str, JsonValue] = Field(default_factory=dict)


def parse_tool_failure(payload: Mapping[str, object]) -> ToolFailure:
    """`payload` as a `ToolFailure`.

    A payload that does not fit -- most often a missing `code`, since that
    is the one field `classify_refusal`'s correction lookup cannot proceed
    without -- parses into a `ToolFailure` naming the defect through
    `ExecutorCode.MALFORMED_REFUSAL` rather than raising. A server that answers its own
    refusal badly is exactly when the executor still has to report
    something back to the plan, not crash the run.
    """
    fix = payload.get("fix")
    try:
        return ToolFailure(
            code=cast(str, payload.get("code")),
            message=str(payload.get("message", "")),
            retry_mode=str(payload.get("retry_mode", "do_not_retry")),
            fix=cast(dict[str, JsonValue], dict(fix)) if isinstance(fix, Mapping) else {},
        )
    except ValidationError as exc:
        return ToolFailure(
            code=ExecutorCode.MALFORMED_REFUSAL,
            message=f"the refusal payload does not fit ToolFailure: {exc}",
            retry_mode="do_not_retry",
            fix={"payload": cast(JsonValue, dict(payload))},
        )


def classify_refusal(failure: ToolFailure) -> Disposition:
    """Decide what a refusal's `retry_mode` and `code` call for."""
    if failure.retry_mode == "retry_after_delay":
        return Disposition.RETRY_LATER
    if failure.retry_mode == "refresh_then_retry":
        return Disposition.REDISCOVER
    if failure.retry_mode == "correct_and_retry":
        if (
            failure.code == "over_ceiling"
            and _ceiling_kind(failure.fix) in _NARROWABLE_CEILING_KINDS
        ):
            return Disposition.METHOD_CHOICE
        return _CORRECTION_DISPOSITIONS.get(failure.code, Disposition.STOP)
    return Disposition.STOP


def is_anchor_fault(failure: ToolFailure) -> bool:
    """Whether `failure` says the run's own account of its newest period is wrong.

    `resolve_anchor` refuses populations and root frames that do not fit
    together, and a rolling `growth`, `rank` or `filter` refuses a key,
    population, root or period that its stored resolution does not cover. The
    executor writes all of them, so either is its own fault. Only a refusal
    whose `fix.detail` says no period qualified, or that a pairing's two sides
    share no key, is about the data.
    """
    return (
        failure.code == ROLLING_ANCHOR_UNRESOLVED and failure.fix.get("detail") not in _DATA_DETAILS
    )


def is_harness_fault(failure: ToolFailure) -> bool:
    """Whether `failure` is the harness's own fault (`HARNESS_FAULT_CODES`, `is_anchor_fault`)."""
    return failure.code in HARNESS_FAULT_CODES or is_anchor_fault(failure)


def retry_delay(payload: Mapping[str, object], attempt: int) -> float:
    """How long to wait before `attempt` (zero-based).

    Reads the raw payload rather than a `ToolFailure`, because
    `retry_after_s` is not one of `ToolFailure`'s four fields: it is a
    transport-timing detail, not part of the refusal's identity. The
    refusal's own `retry_after_s` is honoured up to `MAX_RETRY_AFTER_SECONDS`
    and ignored when it is negative or not finite; the doubling backoff
    applies otherwise, and as a floor.
    """
    backoff = BACKOFF_SECONDS * 2**attempt
    named = _named_delay(payload.get("retry_after_s"))
    if named is None:
        return backoff
    return min(max(named, backoff), MAX_RETRY_AFTER_SECONDS)


def jittered_retry_delay(payload: Mapping[str, object], attempt: int, *, jitter: Jitter) -> float:
    """`retry_delay` spread by jitter: a wait drawn between it and
    `RETRY_JITTER_FACTOR` times it, never past `MAX_RETRY_AFTER_SECONDS`
    unless the doubling backoff alone already is."""
    floor = retry_delay(payload, attempt)
    return jitter(floor, max(floor, min(floor * RETRY_JITTER_FACTOR, MAX_RETRY_AFTER_SECONDS)))


def delay_before_retry(
    payload: Mapping[str, object], attempt: int, *, jitter: Jitter
) -> float | None:
    """How long to wait before retrying the call `payload` refused on
    `attempt` (zero-based), or `None` when it is not retried.

    A refusal is retried only when `classify_refusal` reads it as
    `RETRY_LATER` (its mode is `retry_after_delay`) and `attempt` was not the
    last of `MAX_ATTEMPTS_PER_STEP`. The executor's step loop and
    `harness.mcp_client.McpExecutor` both ask here, so the plan path and the
    Studio graph's direct calls retry on one policy.
    """
    if attempt + 1 >= MAX_ATTEMPTS_PER_STEP:
        return None
    if classify_refusal(parse_tool_failure(payload)) is not Disposition.RETRY_LATER:
        return None
    return jittered_retry_delay(payload, attempt, jitter=jitter)


def _named_delay(value: object) -> float | None:
    """`value` as a usable delay, or `None` when it is absent, negative or not finite."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    seconds = float(value)
    return seconds if math.isfinite(seconds) and seconds >= 0 else None


def candidates_of(failure: ToolFailure) -> JsonValue:
    """The alternatives a refusal offers, as it offered them."""
    if "candidates" in failure.fix:
        return failure.fix["candidates"]
    return cast(JsonValue, failure.fix) if failure.fix else None
