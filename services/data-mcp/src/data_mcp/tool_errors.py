"""Tool-facing refusal grammar.

`ToolRefusal` is the stable code/retry-mode/fix/message contract a tool
returns via `to_tool_result` instead of letting a raw `UpstreamError`
propagate. Twenty codes span four retry modes (`RetryMode`), most with a
typed factory whose `fix` payload is the machine-readable detail a calling
model needs to correct and retry, wait, refresh, or give up.

`from_upstream_error` routes an apis service call failure onto this grammar.
`entity_not_found`, `entity_ambiguous`, `definition_not_held`, and
`node_not_held` come from resolver/lookup code instead, which already has
the query/role/candidates/definition_id/dcids in hand. `definition_not_held`
in particular is what a caller should raise on catching an `UpstreamError`
with `code="spec_not_found"` from the definitions door, using the id it
requested itself: `SpecNotFound`'s wire body carries that id only in
free-text `detail`, not in any structural field, so it isn't recoverable
generically from `UpstreamError` alone. `node_not_held` is never reached
through `from_upstream_error` either: the node batch route answers an unheld
dcid as a `not_held` entry, which `data_mcp.node_lookup` returns as data, so
this code has no
`_UPSTREAM_CODE_MAP` entry -- it is unreachable there. `operation_still_processing`
is never reached through `from_upstream_error` either: its one source is
`data_mcp.operations.resolve_operation_response`, when the call deadline ends its
wait on a 202 operation without a manifest. `service_unavailable` has five
sources: `from_upstream_error` (apis' own `service_unavailable` code, or a
transport error), `mint_unanswered` (a mint POST that got no response), an
unreachable definition store (`search` and `get_data`), `get_data`'s refusals
for work the call deadline left no time for, or for a mint the mint ledger
(`data_mcp.mint_ledger`) has no slot for (`mint_still_running`, `mint_slots_full`),
and `row_budget_full`, for a held pair or a fetched artifact whose rows the
process's frame row budget (`common.row_budget`) has no room for.

`skill_not_found` and `skill_resource_not_found` bypass `from_upstream_error`
too. `data_mcp.skills_client` reads the Skills API's own codes of those names
and builds each refusal with the id and path it asked for, since the problem
body carries neither as a structural field.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any, Literal, Self, cast

from common.refusal import refusal_meta, refusal_text
from common.row_budget import ROW_BUDGET_RETRY_AFTER_S
from fastmcp.tools import ToolResult

from data_mcp.errors import UpstreamError, UpstreamSurfaceUnavailable, UpstreamTransportError

logger = logging.getLogger(__name__)


class RetryMode(StrEnum):
    """How a caller should react to a `ToolRefusal`, distinct from the code that caused it."""

    CORRECT_AND_RETRY = "correct_and_retry"
    REFRESH_THEN_RETRY = "refresh_then_retry"
    RETRY_AFTER_DELAY = "retry_after_delay"
    DO_NOT_RETRY = "do_not_retry"


class ToolErrorCode(StrEnum):
    """The twenty refusal codes, each mapped to exactly one `RetryMode`."""

    FACET_NOT_AVAILABLE = "facet_not_available"
    DEFAULT_FACET_UNDEFINED = "default_facet_undefined"
    OVER_CEILING = "over_ceiling"
    CAPABILITY_UNAVAILABLE = "capability_unavailable"
    UPSTREAM_VERSION_CHANGED = "upstream_version_changed"
    ENTITY_NOT_FOUND = "entity_not_found"
    ENTITY_AMBIGUOUS = "entity_ambiguous"
    UPSTREAM_ERROR = "upstream_error"
    SERVICE_UNAVAILABLE = "service_unavailable"
    DEFINITION_NOT_HELD = "definition_not_held"
    REQUEST_INVALID = "request_invalid"
    NODE_NOT_HELD = "node_not_held"
    FRAME_INTEGRITY_FAILED = "frame_integrity_failed"
    SPEC_NOT_EXPORTABLE = "spec_not_exportable"
    OPERATION_STILL_PROCESSING = "operation_still_processing"
    SKILL_NOT_FOUND = "skill_not_found"
    SKILL_RESOURCE_NOT_FOUND = "skill_resource_not_found"
    ROLLUP_NOT_SUMMABLE = "rollup_not_summable"
    OPEN_ROLE_UNAVAILABLE = "open_role_unavailable"
    MEMBER_NOT_RESOLVABLE = "member_not_resolvable"


# Locked table: two upstream `retryable: false` codes land on different modes
# (`facet_not_available` vs. `capability_unavailable`), and two `retryable:
# true` codes do too (`upstream_version_changed` vs. `upstream_error`) — the
# upstream flag alone can't pick a mode, so this table is the source of truth.
_RETRY_MODE_BY_CODE: dict[ToolErrorCode, RetryMode] = {
    ToolErrorCode.FACET_NOT_AVAILABLE: RetryMode.CORRECT_AND_RETRY,
    ToolErrorCode.DEFAULT_FACET_UNDEFINED: RetryMode.CORRECT_AND_RETRY,
    ToolErrorCode.OVER_CEILING: RetryMode.CORRECT_AND_RETRY,
    ToolErrorCode.CAPABILITY_UNAVAILABLE: RetryMode.DO_NOT_RETRY,
    ToolErrorCode.UPSTREAM_VERSION_CHANGED: RetryMode.REFRESH_THEN_RETRY,
    ToolErrorCode.ENTITY_NOT_FOUND: RetryMode.CORRECT_AND_RETRY,
    ToolErrorCode.ENTITY_AMBIGUOUS: RetryMode.CORRECT_AND_RETRY,
    ToolErrorCode.UPSTREAM_ERROR: RetryMode.RETRY_AFTER_DELAY,
    ToolErrorCode.SERVICE_UNAVAILABLE: RetryMode.RETRY_AFTER_DELAY,
    ToolErrorCode.DEFINITION_NOT_HELD: RetryMode.CORRECT_AND_RETRY,
    ToolErrorCode.REQUEST_INVALID: RetryMode.CORRECT_AND_RETRY,
    ToolErrorCode.NODE_NOT_HELD: RetryMode.CORRECT_AND_RETRY,
    ToolErrorCode.FRAME_INTEGRITY_FAILED: RetryMode.DO_NOT_RETRY,
    ToolErrorCode.SPEC_NOT_EXPORTABLE: RetryMode.DO_NOT_RETRY,
    ToolErrorCode.OPERATION_STILL_PROCESSING: RetryMode.RETRY_AFTER_DELAY,
    ToolErrorCode.SKILL_NOT_FOUND: RetryMode.CORRECT_AND_RETRY,
    ToolErrorCode.SKILL_RESOURCE_NOT_FOUND: RetryMode.CORRECT_AND_RETRY,
    ToolErrorCode.ROLLUP_NOT_SUMMABLE: RetryMode.DO_NOT_RETRY,
    ToolErrorCode.OPEN_ROLE_UNAVAILABLE: RetryMode.CORRECT_AND_RETRY,
    ToolErrorCode.MEMBER_NOT_RESOLVABLE: RetryMode.DO_NOT_RETRY,
}
if set(_RETRY_MODE_BY_CODE) != set(ToolErrorCode):
    # A plain `raise` rather than `assert` — `python -O` strips asserts, and
    # this import-time guard should hold even under `-O`.
    raise AssertionError("every ToolErrorCode needs exactly one _RETRY_MODE_BY_CODE entry")

# Five keeps `entity_ambiguous` payloads bounded without starving the model
# of real choices.
ENTITY_CANDIDATE_CAP = 5

# The marker `entity_ambiguous` puts before each candidate's dcid, and that `get_data`
# reads as "a dcid, not a name to resolve" in `scope.entities`.
ENTITY_DCID_MARKER = "dcid:"

# The ceilings `over_ceiling`'s `fix.ceiling_kind` can name, each by the unit
# its `count` and `cap` measure. `_UPSTREAM_CEILING_KINDS` maps apis' own
# kinds onto these. `candidate_window` is apis' member walk stopping at its
# candidate page before it could prove the family complete;
# `member_cardinality` is a family counted and found over the member cap.
CeilingKind = Literal["bytes", "candidate_window", "member_cardinality", "requests", "rows"]


# The end-user remedy for the ceilings `scope.entities` and `scope.window`
# cannot narrow, keyed by kind.
_CEILING_REMEDIES: dict[CeilingKind, str] = {
    "member_cardinality": (
        "This spec_id's definition has more member series than one request can "
        "serve. scope.entities and scope.window do not change that. A narrower "
        "question, such as one naming a sector for aid flows, resolves to a "
        "smaller definition."
    ),
    "candidate_window": (
        "This spec_id's definition covers too broad a family of series to list in "
        "one request. scope.entities and scope.window do not change that. A "
        "narrower question, such as one naming a breakdown, resolves to a smaller "
        "definition."
    ),
}


@dataclass(frozen=True, kw_only=True)
class EntityCandidate:
    """One candidate in an `entity_ambiguous` refusal."""

    dcid: str
    label: str
    type: str


@dataclass(frozen=True, kw_only=True)
class FacetAlternative:
    """One `ref`/`label` pair apis offers in place of a rejected facet."""

    ref: str
    label: str


@dataclass(frozen=True, kw_only=True)
class FacetIssue:
    """One axis's worth of a `facet_not_available` refusal.

    Mirrors apis' `ProblemIssue` (`src/dfo_platform/problem.py`):
    `mint.py`'s `classify_mint_exception` aggregates every failing axis into
    a single `issues` list before raising, so a refusal can name more than
    one axis at once.
    """

    axis: str | None
    alternatives: list[FacetAlternative]
    truncated: bool
    # The issue's own cause, when apis sends one.
    code: str | None = None
    detail: str | None = None


@dataclass(frozen=True, kw_only=True)
class ToolRefusal:
    """A tool's structured refusal, built by one of its typed factories."""

    code: ToolErrorCode
    message: str
    fix: dict[str, Any]
    upstream_code: str | None = None
    retry_after_s: float | None = None
    # `to_tool_result` logs it and leaves it out of the result.
    trace_id: str | None = None
    capability_version: str | None = None

    @property
    def retry_mode(self) -> RetryMode:
        if self.code is ToolErrorCode.UPSTREAM_ERROR and self.retry_after_s is None:
            # `_RETRY_MODE_BY_CODE` pins this code to `retry_after_delay`,
            # which fits the six other `upstream_error` instances -- every
            # one of them has a numeric wait. This one doesn't: an unmapped
            # code upstream itself marked non-retryable, or a 2xx response
            # this service couldn't use, carries no delay to name, and
            # `upstream_error`'s own message already says a repeat is
            # unlikely to succeed unmodified. Reading `retry_after_delay`
            # off the table here would contradict that message with the one
            # field a model actually acts on.
            return RetryMode.DO_NOT_RETRY
        if self.code is ToolErrorCode.DEFAULT_FACET_UNDEFINED:
            # apis' `retryable` says whether the same request might succeed
            # (`dfo_platform/problem.py`). When it can't, a changed request can
            # only when an issue lists values to change to.
            if self.fix.get("retryable"):
                return RetryMode.RETRY_AFTER_DELAY
            if any(issue["alternatives"] for issue in self.fix.get("issues", [])):
                return RetryMode.CORRECT_AND_RETRY
            return RetryMode.DO_NOT_RETRY
        return _RETRY_MODE_BY_CODE[self.code]

    @classmethod
    def facet_not_available(
        cls,
        *,
        issues: Sequence[FacetIssue],
        detail: str | None = None,
        trace_id: str | None = None,
        capability_version: str | None = None,
    ) -> Self:
        # `mint.py`'s `classify_mint_exception` sets only `issues`
        # for this code, never a top-level `pointer`/`alternatives` — the
        # failed ref itself has no machine-readable field anywhere, only
        # `detail` prose. A failed operation with no stored refusal carries no
        # `issues`, and search omits `alternatives` for an axis with no live
        # value, so the message points at alternatives only when `issues` lists
        # some, and otherwise states the axis and that none was listed, which is
        # all the body shows.
        axes = [i.axis for i in issues if i.axis]
        axis_list = axes or ["the requested axis"]
        if any(i.alternatives for i in issues):
            message = (
                f"Facet unavailable on {axis_list}; pick one of the listed alternatives per "
                "axis and retry."
            )
        else:
            message = f"Facet unavailable on {axis_list}, and no alternative values were listed."
        return cls(
            code=ToolErrorCode.FACET_NOT_AVAILABLE,
            message=message,
            fix={"issues": [asdict(i) for i in issues], "detail": detail},
            trace_id=trace_id,
            capability_version=capability_version,
        )

    @classmethod
    def default_facet_undefined(
        cls,
        *,
        issues: Sequence[FacetIssue],
        detail: str | None,
        retryable: bool,
        spec_id: str | None = None,
        entities: Mapping[str, Sequence[str]] | None = None,
        places: str | None = None,
        trace_id: str | None = None,
        capability_version: str | None = None,
    ) -> Self:
        # Relays apis' `detail`, `issues[]` (each with its own code and
        # detail) and `retryable` in `fix`. The message names what was asked
        # for; `get_data` adds the spec_id and the places it was minting, which
        # the operation-poll path does not have.
        message = f"The platform refused the requested data{f' for {places}' if places else ''}."
        fix: dict[str, Any] = {
            "issues": [asdict(i) for i in issues],
            "detail": detail,
            "retryable": retryable,
        }
        if spec_id is not None:
            fix["spec_id"] = spec_id
        if entities is not None:
            fix["entities"] = {role: list(refs) for role, refs in entities.items()}
        return cls(
            code=ToolErrorCode.DEFAULT_FACET_UNDEFINED,
            message=message,
            fix=fix,
            trace_id=trace_id,
            capability_version=capability_version,
        )

    @classmethod
    def over_ceiling(
        cls,
        *,
        detail: str,
        ceiling_kind: CeilingKind | None = "rows",
        count: int | None = None,
        cap: int | None = None,
        trace_id: str | None = None,
        capability_version: str | None = None,
    ) -> Self:
        # `OverCeilingError` (`src/materialise/errors.py`) and
        # `classify_mint_exception` (`mint.py`) carry
        # `row_count`/`distinct_counts_by_role` only in `detail` prose
        # (`window.py`), never as structured fields, and the entity
        # sets behind them are discarded before `Coverage` is even built
        # (`ceiling.py`). `ceiling_kind` is a second, independent signal: a
        # member-cardinality or candidate-window ceiling is fixed by the
        # spec_id's own five-tuple and slots, so only a row, byte or request
        # ceiling narrows with `scope.entities` and `scope.window`.
        # `frame_materialisation.py` raises this factory for its own local
        # row-count ceiling with no upstream code to dispatch on, which is what
        # the "rows" default is for. `None` is a ceiling of unknown kind (a body
        # that names no kind this module maps), which `fix` leaves out, so no
        # caller reads it as one that narrows with scope.
        detail_sep = "" if detail.endswith(".") else "."
        # `remedy`, when set, is the one sentence a caller-facing close can
        # append verbatim: unlike `detail` (upstream's own prose about what
        # was over the ceiling), it is written for the end user, plain
        # words already, naming the thing they can actually do about it --
        # `harness.graph.execute._failed_step`/`narrate` read it off `fix` and
        # place it in the turn's closing text by code, not by asking a
        # narrator model to restate it.
        remedy = _CEILING_REMEDIES.get(ceiling_kind) if ceiling_kind is not None else None
        if remedy is not None:
            message = f"Over the coverage ceiling: {detail}{detail_sep} {remedy}"
        elif ceiling_kind is None:
            message = f"Over the coverage ceiling: {detail}{detail_sep}"
        else:
            message = f"Over the coverage ceiling: {detail}{detail_sep} Narrow the scope and retry."
        # `count` and `cap` are absent on the deployment's own member-cardinality
        # branch, where the enumeration can't page fast enough to produce a
        # count at all -- that's "unknown", not "zero", so the keys are left
        # out of `fix` rather than carrying a default that would read as zero.
        # They are in the ceiling's own unit, so a ceiling of unknown kind
        # carries neither.
        fix: dict[str, Any] = {"detail": detail}
        if ceiling_kind is not None:
            fix["ceiling_kind"] = ceiling_kind
            if count is not None:
                fix["count"] = count
            if cap is not None:
                fix["cap"] = cap
        if remedy is not None:
            fix["remedy"] = remedy
        return cls(
            code=ToolErrorCode.OVER_CEILING,
            message=message,
            fix=fix,
            trace_id=trace_id,
            capability_version=capability_version,
        )

    @classmethod
    def capability_unavailable(
        cls,
        *,
        detail: str | None,
        option: str | None = None,
        trace_id: str | None = None,
        capability_version: str | None = None,
    ) -> Self:
        # Without `option`, this surfaces the prose pointer upstream gives to a
        # separate capabilities call. With it, the refusal names the one tool
        # argument this deployment cannot serve, and the call succeeds without it.
        if option is None:
            return cls(
                code=ToolErrorCode.CAPABILITY_UNAVAILABLE,
                message=(
                    f"This capability isn't live: {detail}. Check capabilities "
                    "before calling this tool again."
                ),
                fix={"detail": detail},
                trace_id=trace_id,
                capability_version=capability_version,
            )
        return cls(
            code=ToolErrorCode.CAPABILITY_UNAVAILABLE,
            message=(
                f"The {option!r} option isn't live: {detail}. Do not retry with "
                f"{option!r}; the call without it is still served."
            ),
            fix={"detail": detail, "option": option},
            trace_id=trace_id,
            capability_version=capability_version,
        )

    @classmethod
    def upstream_version_changed(
        cls, *, capability_version: str | None, trace_id: str | None = None
    ) -> Self:
        # apis raises it when a source's revision, or a rate table's
        # provenance, moved during materialisation and its own retry did not
        # converge (`materialise/revision.py`, `materialise/rate_tables.py`).
        # This is the synchronous mint's refusal; a queued operation that fails
        # on it is retried in place (`_FAILED_OPERATION_CAUSES_RETRIED_IN_PLACE`).
        return cls(
            code=ToolErrorCode.UPSTREAM_VERSION_CHANGED,
            message=(
                "The source data refreshed while it was being retrieved, so the frame could "
                "not be built from one consistent version. Run the request again."
            ),
            fix={"capability_version": capability_version},
            capability_version=capability_version,
            trace_id=trace_id,
        )

    @classmethod
    def entity_not_found(
        cls,
        *,
        role: str,
        query: str,
        near_matches: Sequence[str],
        not_held: bool = False,
        memberless_kind: str | None = None,
        trace_id: str | None = None,
        capability_version: str | None = None,
    ) -> Self:
        # `not_held` is `data_mcp.resolution`'s `NotHeld` outcome: `query` names a real
        # entity, but this deployment's graph holds no node for it, so no data can bind it.
        # `memberless_kind` is a group or region of that kind that resolved with no member
        # country, so it scopes nothing. Both set `fix.not_held`, which makes "resolved, but
        # unusable here" machine-readable the way `had_near_matches` makes the
        # near-miss/total-whiff distinction readable, and `entity_type` names the kind where
        # the door gave one; a `not_held` entity has none.
        near = list(near_matches)
        if memberless_kind is not None:
            message = (
                f"{query!r} as {role} resolved to a {memberless_kind} with no member "
                "countries, so it names no entity to fetch data for."
            )
        elif not_held:
            message = (
                f"{query!r} as {role} names a real place or organisation, but this "
                "deployment's graph holds no node for it, so it has no data here -- "
                "distinct from matching nothing at all."
            )
        elif near:
            message = f"No match for {query!r} as {role}. Nearest: {near}."
        else:
            message = f"No match for {query!r} as {role}, and nothing close."
        fix: dict[str, Any] = {
            "role": role,
            "near_matches": near,
            "had_near_matches": bool(near),
            "not_held": not_held or memberless_kind is not None,
        }
        if memberless_kind is not None:
            fix["entity_type"] = memberless_kind
        return cls(
            code=ToolErrorCode.ENTITY_NOT_FOUND,
            message=message,
            fix=fix,
            trace_id=trace_id,
            capability_version=capability_version,
        )

    @classmethod
    def entity_ambiguous(
        cls,
        *,
        role: str,
        query: str,
        candidates: Sequence[EntityCandidate],
        trace_id: str | None = None,
        capability_version: str | None = None,
    ) -> Self:
        # `candidates` arrives here uncapped (`resolution.Ambiguous.candidates`), so `total` is the real count this refusal can see --
        # counted before `ENTITY_CANDIDATE_CAP` narrows what's actually shown. Interpolating
        # `len(capped)` here instead would state the capped count as if it were the true one:
        # a name matching nine held countries would tell the model "ambiguous among 5
        # candidates" with nothing marking the list partial, and the correct entity could be
        # the uncounted seventh.
        total = len(candidates)
        capped = list(candidates)[:ENTITY_CANDIDATE_CAP]
        # True when `ENTITY_CANDIDATE_CAP` cut the list, so the model is picking from a list
        # that may not include the entity it meant.
        list_truncated = total > len(capped)
        shown = f"{len(capped)} of {total}" if len(capped) < total else str(len(capped))
        # Each candidate's `entity`, its dcid behind `ENTITY_DCID_MARKER`, rides beside its
        # label in the message itself, not only in `fix["candidates"]`: a client that shows the
        # model only `content` (an external caller with no `structuredContent` surface, see the
        # module docstring) still needs the exact string `get_data` takes in place of the name.
        named = [f"{c.label} ({ENTITY_DCID_MARKER}{c.dcid})" for c in capped]
        return cls(
            code=ToolErrorCode.ENTITY_AMBIGUOUS,
            message=(
                f"{query!r} as {role} is ambiguous among {shown} candidates: "
                f"{named}. Retry with the one meant in place of {query!r}, written as shown "
                f"with its {ENTITY_DCID_MARKER} prefix."
            ),
            fix={
                "role": role,
                "candidates": [
                    {**asdict(c), "entity": f"{ENTITY_DCID_MARKER}{c.dcid}"} for c in capped
                ],
                "candidate_count": total,
                "truncated": list_truncated,
            },
            trace_id=trace_id,
            capability_version=capability_version,
        )

    @classmethod
    def upstream_error(
        cls,
        *,
        upstream_code: str | None,
        detail: str | None,
        retry_after_s: float | None,
        trace_id: str | None = None,
        capability_version: str | None = None,
    ) -> Self:
        message = (
            f"Upstream call failed ({upstream_code or 'unclassified'}): {detail}. "
            f"Retry after {retry_after_s:.1f}s."
            if retry_after_s is not None
            else f"Upstream call failed ({upstream_code or 'unclassified'}): {detail}. "
            "Upstream marked this non-retryable; a repeated call is unlikely to succeed unmodified."
        )
        return cls(
            code=ToolErrorCode.UPSTREAM_ERROR,
            message=message,
            fix={"detail": detail},
            upstream_code=upstream_code,
            retry_after_s=retry_after_s,
            trace_id=trace_id,
            capability_version=capability_version,
        )

    @classmethod
    def upstream_response_unusable(
        cls,
        *,
        detail: str,
        trace_id: str | None = None,
        capability_version: str | None = None,
    ) -> Self:
        # Shares `upstream_error`'s code rather than its factory: there is no
        # `UpstreamError` here to carry an `upstream_code`/`retryable` flag,
        # since the call itself succeeded -- what failed is turning a 2xx
        # body into the shape this service expects (bad JSON, a schema-drifted
        # manifest, or a manifest shaped for a request this tool never sends).
        # This factory is shared by every upstream call this service makes
        # (mint, search, resolve) rather than one -- `detail` is what names
        # which operation failed, not this classmethod, so the message itself
        # stays call-neutral. `upstream_error`'s own message text claims
        # "upstream marked this non-retryable", which would misattribute a
        # local parse failure to a flag that was never set, so this gets its
        # own message instead.
        # `do_not_retry`, not `retry_after_delay`: a body truncated in transit
        # surfaces as a transport error under HTTP's own content-length
        # framing, not as a clean 2xx, so the realistic cause of a 2xx this
        # service can't use is schema drift between this service and its own
        # upstream -- deterministic, not transient. Retrying the identical
        # call reaches the identical response, and no argument here was
        # wrong either, so `retry_after_s=None` puts this on `do_not_retry`
        # via the same instance-level rule `upstream_error` itself relies on
        # -- telling a model to retry a deterministic failure would just burn
        # its turns.
        return cls(
            code=ToolErrorCode.UPSTREAM_ERROR,
            message=(
                f"The upstream call succeeded, but its response couldn't be used: {detail}. "
                "Retrying the same call would reach the same response, and no argument "
                "here was wrong either."
            ),
            fix={"detail": detail},
            retry_after_s=None,
            trace_id=trace_id,
            capability_version=capability_version,
        )

    @classmethod
    def service_unavailable(
        cls,
        *,
        service: str,
        retry_after_s: float,
        detail: str | None = None,
        trace_id: str | None = None,
        capability_version: str | None = None,
    ) -> Self:
        message = f"{service} is unreachable right now"
        message += f": {detail}." if detail else "."
        message += f" Retry after {retry_after_s:.1f}s."
        fix: dict[str, Any] = {"service": service}
        if detail is not None:
            fix["detail"] = detail
        return cls(
            code=ToolErrorCode.SERVICE_UNAVAILABLE,
            message=message,
            fix=fix,
            retry_after_s=retry_after_s,
            trace_id=trace_id,
            capability_version=capability_version,
        )

    @classmethod
    def definition_not_held(
        cls,
        *,
        definition_id: str,
        trace_id: str | None = None,
        capability_version: str | None = None,
    ) -> Self:
        # `definition_not_held` means the store is up and the id simply
        # isn't registered, distinct from `service_unavailable`, where the
        # store itself is down.
        return cls(
            code=ToolErrorCode.DEFINITION_NOT_HELD,
            message=(
                f"No definition held for {definition_id!r}. This store is up — the id "
                "just isn't registered. Search again with a different id."
            ),
            fix={"definition_id": definition_id, "next_step": "search"},
            trace_id=trace_id,
            capability_version=capability_version,
        )

    @classmethod
    def open_role_unavailable(
        cls,
        *,
        pointer: str | None,
        detail: str | None,
        trace_id: str | None = None,
        capability_version: str | None = None,
    ) -> Self:
        # Search refuses a `dataset_resubmit` with this 422 when the posted
        # identity holds a role open and the StatVars the definition elects
        # declare no role on that axis. `pointer` is
        # the identity's `open_roles` and `detail` names the axis. The same
        # identity refuses the same way every time. A fresh `search` asking for
        # the role elects whichever source breaks the measure down by it.
        stated = f": {detail}" if detail else ""
        return cls(
            code=ToolErrorCode.OPEN_ROLE_UNAVAILABLE,
            message=(
                f"This definition can no longer give one series per member of the role "
                f"its identity holds open{stated}. Search again for the breakdown and "
                "use the spec_id it returns."
            ),
            fix={"pointer": pointer, "detail": detail, "next_step": "search"},
            trace_id=trace_id,
            capability_version=capability_version,
        )

    @classmethod
    def request_invalid(
        cls,
        *,
        pointer: str | None,
        detail: str | None,
        role: str | None = None,
        trace_id: str | None = None,
        capability_version: str | None = None,
    ) -> Self:
        # `pointer` is a JSON Pointer into the posted request body, naming the
        # argument at fault; it isn't among the six fields `UpstreamError`
        # names directly, so it arrives through `exc.extensions`. `role` is
        # `invalid_role_ref`/`role_incompatible`'s own addition (`mint.py`):
        # both name the offending scope role structurally, in `ProblemIssue.
        # axis`, so a caller can act on it directly rather than parsing it
        # back out of `pointer`'s trailing path segment.
        if pointer and role:
            message = (
                f"The request was invalid at {pointer} (role {role!r}): {detail}. "
                "Correct that argument and retry."
            )
        elif pointer:
            message = (
                f"The request was invalid at {pointer}: {detail}. Correct that argument and retry."
            )
        else:
            message = f"The request was invalid: {detail}. Correct the argument and retry."
        fix: dict[str, Any] = {"pointer": pointer, "detail": detail}
        if role is not None:
            fix["role"] = role
        return cls(
            code=ToolErrorCode.REQUEST_INVALID,
            message=message,
            fix=fix,
            trace_id=trace_id,
            capability_version=capability_version,
        )

    @classmethod
    def node_not_held(
        cls,
        *,
        dcids: Sequence[str],
        trace_id: str | None = None,
        capability_version: str | None = None,
    ) -> Self:
        # Every offending dcid is named in `message` itself, not only in
        # `fix` -- the two halves of a partial-success reply reach the model
        # through different paths (the digest preamble and this message),
        # and only the message is guaranteed to survive both.
        names = list(dcids)
        message = (
            f"This deployment's graph does not hold: {', '.join(names)}. "
            "Describe a different dcid, or drop these from the call and retry."
        )
        return cls(
            code=ToolErrorCode.NODE_NOT_HELD,
            message=message,
            fix={"dcids": names},
            trace_id=trace_id,
            capability_version=capability_version,
        )

    @classmethod
    def frame_integrity_failed(cls, *, spec_id: str, content_version: str, reason: str) -> Self:
        # Raised by `frame_materialisation.materialise_and_register`'s own cache check:
        # the frame store already holds an object under this exact
        # `{spec_id, content_version}` pair, and reading it back failed an integrity
        # check. The store is content-addressed and insert-only (`GcsMaterialisationStore`'s
        # own docstring), so re-registering under this identical pair cannot heal it --
        # a fresh write only lands when nothing is held there yet.
        return cls(
            code=ToolErrorCode.FRAME_INTEGRITY_FAILED,
            message=(
                f"The frame already held for spec_id={spec_id!r}, "
                f"content_version={content_version!r} failed an integrity check: {reason}. "
                "Retrying this identical call reaches the identical stored object -- call "
                "get_data again with a narrower scope.entities or scope.window instead, "
                "which mints a different content_version."
            ),
            fix={"spec_id": spec_id, "content_version": content_version, "reason": reason},
        )

    @classmethod
    def spec_not_exportable(
        cls,
        *,
        detail: str,
        trace_id: str | None = None,
        capability_version: str | None = None,
    ) -> Self:
        # Upstream's own raise site (`adapter.resolve_definition_members`, 409) fires when
        # a described-arm ask enumerates no member consistent with its own five-tuple and
        # slots against the graph -- a fact about the graph, not a retryable argument
        # mistake. `get_data` maps it through here rather than guessing at this outcome
        # itself ahead of the call.
        detail_sep = "" if detail.endswith(".") else "."
        return cls(
            code=ToolErrorCode.SPEC_NOT_EXPORTABLE,
            message=(
                f"This spec can't be exported: {detail}{detail_sep} This is a dead end for "
                "this definition -- search again for a narrower or different ask."
            ),
            fix={"detail": detail},
            trace_id=trace_id,
            capability_version=capability_version,
        )

    @classmethod
    def rollup_not_summable(
        cls,
        *,
        axes: Sequence[str],
        detail: str | None,
        trace_id: str | None = None,
        capability_version: str | None = None,
    ) -> Self:
        # apis refuses a `get_data(stat_vars="sum")` mint with this code when the
        # family's StatVars cannot be added into one total: two members are distinct
        # measures with the same codes (gross and net), a member has no decoded code, or
        # the rows span several sources (`datasets_api/groupby.py`). The 422 names the
        # axis in `issues[].axis` and the precondition in `detail` prose. A summed mint
        # that ran as an operation reports both when apis stored its refusal, and
        # otherwise only the code (`cause.code`), so both can be empty. The same
        # request refuses the same way every time. `detail` stays in `fix` for
        # diagnosis. `remedy` is written for the end user, like
        # `over_ceiling`'s: it offers a breakdown by each axis other than `stat_var`
        # the sum failed on, and apis names `provenance` when the rows span
        # several sources. The message's last sentence is for a caller that sets
        # `stat_vars` itself.
        axis_list = list(axes)
        remedy = "Ask about one of the series it combines instead of the total"
        breakdown_axes = [
            "source" if axis == "provenance" else axis for axis in axis_list if axis != "stat_var"
        ]
        if breakdown_axes:
            remedy = f"{remedy}, or for the figures broken down by {' or '.join(breakdown_axes)}"
        remedy = f"{remedy}."
        return cls(
            code=ToolErrorCode.ROLLUP_NOT_SUMMABLE,
            message=(
                "The series in this dataset can't be added into one total: a total could "
                f"count the same amount twice or merge different measures. {remedy} "
                "Repeating the call refuses the same way. A get_data call with stat_vars "
                '"separate" returns the series one by one.'
            ),
            fix={"axes": axis_list, "detail": detail, "remedy": remedy},
            trace_id=trace_id,
            capability_version=capability_version,
        )

    @classmethod
    def operation_still_processing(
        cls,
        *,
        operation_id: str,
        last_status: str,
        last_poll_error: str | None,
        elapsed_s: float,
        retry_after_s: float,
        trace_id: str | None = None,
        capability_version: str | None = None,
    ) -> Self:
        # Raised by `data_mcp.operations.resolve_operation_response` once a mint's own
        # 202 operation has not produced a manifest within its share of the call
        # deadline (`CallBudget.post_mint_reserve_s` is kept back). apis'
        # `datasets_api.operation_store.compute_operation_id` hashes
        # `{request_identity, coverage_stamp, rendering}`, so a repeat `get_data` call
        # for this spec_id and scope rejoins the same operation while the dataset's
        # coverage is unchanged, and starts a fresh one once newer coverage lands.
        # Either way the fix is to wait and repeat the call, never to pick a different
        # spec_id or narrow the request. `last_status` is the last status apis
        # reported, and `last_poll_error` names the last failure when every poll since
        # then failed. A `done`/`failed` operation never runs again, so for those the
        # message states the status and says only that its result was not fetched.
        if last_status in ("done", "failed"):
            detail = f"{elapsed_s:.0f}s of polling"
            if last_poll_error:
                detail += f"; the last attempt ended with {last_poll_error}"
            status = (
                f"dfo-apis reported operation {operation_id!r} {last_status}, but this call "
                f"could not fetch its result before its own budget ran out ({detail})."
            )
        elif last_poll_error is None:
            status = (
                f"This dataset is still being prepared upstream (operation {operation_id!r}, "
                f"{elapsed_s:.0f}s of polling within this call's own budget) -- it has not "
                "finished yet, not failed."
            )
        else:
            status = (
                f"This call could not confirm the status of operation {operation_id!r} "
                f"after {elapsed_s:.0f}s of polling: every poll since dfo-apis last reported "
                f"it {last_status} ended in error, the last with {last_poll_error}. It may "
                "still be running, or it may have finished or failed."
            )
        fix: dict[str, Any] = {
            "operation_id": operation_id,
            "last_status": last_status,
            "elapsed_s": round(elapsed_s, 1),
        }
        if last_poll_error is not None:
            fix["last_poll_error"] = last_poll_error
        return cls(
            code=ToolErrorCode.OPERATION_STILL_PROCESSING,
            message=(
                f"{status} Call get_data again with the same spec_id and scope: while the "
                "dataset's coverage is unchanged, the repeat call picks this same operation "
                "back up; if newer coverage has been published since, it starts a fresh one "
                f"against that instead. Retry after {retry_after_s:.1f}s."
            ),
            fix=fix,
            retry_after_s=retry_after_s,
            trace_id=trace_id,
            capability_version=capability_version,
        )

    @classmethod
    def mint_unanswered(
        cls,
        *,
        spec_id: str,
        elapsed_s: float,
        retry_after_s: float,
        waits_out_sync_mint: bool,
        trace_id: str | None = None,
        capability_version: str | None = None,
    ) -> Self:
        # Raised by `get_data._post_mint` on any `UpstreamTransportError` from the mint
        # POST: a timeout, whether its own `CallBudget.mint_timeout_s` cap or the call
        # deadline ended it, or a connection failure. Also raised for a call that joined
        # an open POST of the same request which had not answered by its deadline. A mint within apis' sync budget
        # runs inline and keeps running after its client leaves, with no operation
        # record; only a mint over that budget becomes an operation a repeat submit
        # rejoins (`datasets_api.operations.submit.submit_over_budget`). A repeat call is
        # therefore safe but may run the request again, and a connection failure started
        # nothing. `waits_out_sync_mint` is `CallBudget.waits_out_sync_mint`: when the
        # call's mint cap is shorter than apis' own synchronous mint timeout, the mint
        # ledger holds the mint until that timeout and `retry_after_s` is the time left.
        # apis' mint route materialises a repeat from the start, so a slow inline
        # mint runs out of time on every repeat, and the message names a narrower scope
        # as the way through.
        prefix = f"The mint for {spec_id!r} got no answer after {elapsed_s:.0f}s. "
        if waits_out_sync_mint:
            message = (
                f"{prefix}dfo-apis may still be running it. Calling get_data again with the "
                "same spec_id and scope is safe: a mint dfo-apis moved to a background "
                "operation is picked up while the dataset's coverage is unchanged, and "
                f"otherwise the request runs again. Retry after {retry_after_s:.1f}s."
            )
        else:
            message = (
                f"{prefix}dfo-apis may still be running it upstream for up to "
                f"{retry_after_s:.0f}s more, and this server instance refuses the same "
                "spec_id and scope until then. A call after that, or one that reaches "
                "another instance, runs the request again from the start, and a request "
                "this large can run out of time the same way on every call. A narrower "
                "scope, such as fewer entities or a shorter window, mints faster. Retry "
                f"after {retry_after_s:.1f}s."
            )
        return cls(
            code=ToolErrorCode.SERVICE_UNAVAILABLE,
            message=message,
            fix={"service": "dfo-apis", "spec_id": spec_id, "elapsed_s": round(elapsed_s, 1)},
            retry_after_s=retry_after_s,
            trace_id=trace_id,
            capability_version=capability_version,
        )

    @classmethod
    def mint_still_running(
        cls,
        *,
        spec_id: str,
        started_s_ago: float,
        retry_after_s: float,
        trace_id: str | None = None,
        capability_version: str | None = None,
    ) -> Self:
        # Raised by `MintLedger.admit` for a mint whose request matches one it holds: a
        # repeat call would start the same work in apis a second time.
        return cls(
            code=ToolErrorCode.SERVICE_UNAVAILABLE,
            message=(
                f"A get_data call with spec_id {spec_id!r} and this scope started a mint "
                f"{started_s_ago:.0f}s ago that dfo-apis may still be running, and this "
                "server instance does not start the same mint twice. A call after the "
                "wait, or one that reaches another instance, runs the request again from "
                "the start, and a request this large can run out of "
                "time the same way on every call. A narrower scope, such as fewer entities "
                f"or a shorter window, mints faster. Retry after {retry_after_s:.1f}s."
            ),
            fix={
                "service": "dfo-apis",
                "spec_id": spec_id,
                "started_s_ago": round(started_s_ago, 1),
            },
            retry_after_s=retry_after_s,
            trace_id=trace_id,
            capability_version=capability_version,
        )

    @classmethod
    def mint_slots_full(
        cls,
        *,
        limit: int,
        retry_after_s: float,
        trace_id: str | None = None,
        capability_version: str | None = None,
    ) -> Self:
        # Raised by `MintLedger.admit` when it already holds `limit` mints.
        return cls(
            code=ToolErrorCode.SERVICE_UNAVAILABLE,
            message=(
                "This server is already running as many dataset mints in dfo-apis as it "
                f"allows at once ({limit}). A mint that got no answer keeps its place until "
                f"dfo-apis finishes it. Retry after {retry_after_s:.1f}s."
            ),
            fix={"service": "dfo-apis", "limit": limit},
            retry_after_s=retry_after_s,
            trace_id=trace_id,
            capability_version=capability_version,
        )

    @classmethod
    def row_budget_full(cls, *, requested: int, free: int, max_rows: int) -> Self:
        # Raised when `common.row_budget` has no room for the rows of a held pair's
        # read or of a fetched artifact.
        return cls(
            code=ToolErrorCode.SERVICE_UNAVAILABLE,
            message=(
                f"This server is already working on as many frame rows as its memory allows "
                f"at once ({max_rows:,}). This dataset needs room for {requested:,} rows and "
                f"{free:,} are free. Retry after {ROW_BUDGET_RETRY_AFTER_S:.1f}s."
            ),
            fix={
                "service": "data-mcp",
                "rows_requested": requested,
                "rows_free": free,
                "rows_in_flight_max": max_rows,
            },
            retry_after_s=ROW_BUDGET_RETRY_AFTER_S,
        )

    @classmethod
    def skill_not_found(
        cls,
        *,
        skill_id: str,
        trace_id: str | None = None,
        capability_version: str | None = None,
    ) -> Self:
        # Raised by `data_mcp.skills_client` on the Skills API's own
        # `skill_not_found`, and for an id that API reserves for a route of its
        # own. Skill ids come from a `search` result, so the fix points there.
        return cls(
            code=ToolErrorCode.SKILL_NOT_FOUND,
            message=(
                f"No skill has the id {skill_id!r}. Skill ids come from the skills "
                "list beneath a search result; call get_skill with an id named there."
            ),
            fix={"skill_id": skill_id, "next_step": "search"},
            trace_id=trace_id,
            capability_version=capability_version,
        )

    @classmethod
    def skill_resource_not_found(
        cls,
        *,
        skill_id: str,
        path: str,
        trace_id: str | None = None,
        capability_version: str | None = None,
    ) -> Self:
        # Raised by `data_mcp.skills_client` on the Skills API's own
        # `skill_resource_not_found`: the skill exists and its manifest has no
        # such path. `get_skill` lists the paths it has.
        return cls(
            code=ToolErrorCode.SKILL_RESOURCE_NOT_FOUND,
            message=(
                f"Skill {skill_id!r} has no file {path!r}. Call "
                f'get_skill(skill_id="{skill_id}") for its file list and read a path '
                "named there; do not retry under a guessed path."
            ),
            fix={"skill_id": skill_id, "path": path, "next_step": "get_skill"},
            trace_id=trace_id,
            capability_version=capability_version,
        )

    @classmethod
    def skill_file_changed(
        cls, *, skill_id: str, path: str, digest: str, current_digest: str
    ) -> Self:
        # A paging continuation whose `digest` is not the file's current one:
        # a replica rebuilt between pages, so the pages read so far belong to
        # another version and the file restarts at offset 0.
        restart = f'get_skill_file(skill_id="{skill_id}", path="{path}", offset=0)'
        return cls(
            code=ToolErrorCode.UPSTREAM_VERSION_CHANGED,
            message=(
                f"Skill {skill_id!r} file {path!r} changed since the page read with digest "
                f"{digest}; its current digest is {current_digest}. Read it again from the "
                f"start with {restart}."
            ),
            fix={
                "skill_id": skill_id,
                "path": path,
                "digest": current_digest,
                "restart": restart,
            },
        )


def _derive_retry_after_s(exc: UpstreamError) -> float:
    """Wait at least as long as the failed attempt sequence already took.

    Heuristic: `elapsed_s` (filled in by the retry loop, per `errors.py`'s own
    docstring) floored at 1.0s so a first-attempt failure still gets a sane wait.
    """
    return max(1.0, round(exc.elapsed_s, 1))


def to_tool_result(refusal: ToolRefusal, *, preamble: str | None = None) -> ToolResult:
    """Render `refusal` as the `ToolResult` a tool returns to signal an error.

    Args:
        refusal: The refusal to render.
        preamble: Prepended to `refusal.message` in `content`, followed by a
            blank line -- the mechanism that lets a partial success (what
            resolved) and a refusal (what didn't) coexist in one reply.
            `structured_content` is unaffected: it stays exactly what
            `refusal` alone would produce, so a caller reading structured
            content sees an ordinary refusal envelope whether or not a
            preamble was rendered into `content`.

    `content` is `common.refusal.refusal_text`: `refusal.message`, then a blank
    line, then `code`, `retry_mode`, `fix` and `retry_after_s` when set as one
    line of compact JSON, so a client that renders only `content` still sees
    them. `upstream_code` and `capability_version` stay in `structured_content`
    alone: they serve a human or a log line reading the raw payload, not a
    model deciding whether and how to retry. `trace_id`, the upstream's own
    trace id for the refused call, is logged and left out of the result.

    `meta` is `common.refusal.refusal_meta` of `code` and `upstream_code`.
    """
    if refusal.trace_id:
        logger.warning(
            "tool refused code=%s upstream_code=%s upstream_trace_id=%s",
            refusal.code.value,
            refusal.upstream_code,
            refusal.trace_id,
        )
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
            "upstream_code": refusal.upstream_code,
            "retry_after_s": refusal.retry_after_s,
            "capability_version": refusal.capability_version,
        },
        meta=refusal_meta(code=refusal.code.value, upstream_code=refusal.upstream_code),
        is_error=True,
    )


# Upstream codes recognised by name. `upstream_error` is listed explicitly
# rather than left implicit, so a `code="upstream_error"` refusal and a
# genuinely unmapped code both reach the same fallback branch below.
_UPSTREAM_CODE_MAP: dict[str, ToolErrorCode] = {
    "facet_not_available": ToolErrorCode.FACET_NOT_AVAILABLE,
    "default_facet_undefined": ToolErrorCode.DEFAULT_FACET_UNDEFINED,
    "over_ceiling": ToolErrorCode.OVER_CEILING,
    # `parquet_size_bound` (`materialise/formats.py`'s `ParquetSizeBoundError`,
    # raised by `write_parquet`) is the same "the materialised output is too
    # big, narrow the request and retry" refusal as `over_ceiling`, just
    # measured in encoded bytes rather than row/entity count, and it carries
    # `detail` prose only, exactly like `over_ceiling` does. The `get_data`
    # tool always requests Parquet rendering, so this code is reachable on
    # every call.
    "parquet_size_bound": ToolErrorCode.OVER_CEILING,
    "capability_unavailable": ToolErrorCode.CAPABILITY_UNAVAILABLE,
    "upstream_version_changed": ToolErrorCode.UPSTREAM_VERSION_CHANGED,
    "upstream_error": ToolErrorCode.UPSTREAM_ERROR,
    "service_unavailable": ToolErrorCode.SERVICE_UNAVAILABLE,
    "request_invalid": ToolErrorCode.REQUEST_INVALID,
    # `invalid_role_ref`/`role_incompatible` (both 422, `mint.py`) fire when a
    # caller names a scope role a definition doesn't accept — the single most
    # likely mistake a model makes with `get_data`, guessing at role names.
    # Both carry a single-entry `issues[]` with `pointer`/`axis` naming the
    # offending role and no top-level `pointer`, a `request_invalid`-shaped
    # refusal in every way that matters, so they share its code and factory
    # rather than inventing a new one; the mapped factory below reads `role`
    # out of that issue so a caller isn't left parsing it back out of prose.
    "invalid_role_ref": ToolErrorCode.REQUEST_INVALID,
    "role_incompatible": ToolErrorCode.REQUEST_INVALID,
    # `DefinitionNotReproducible` (`datasets_api/doors/mint_errors.py`) fires
    # when a stored definition's identity binds a scope role the graph's
    # current account of the StatVar does not declare
    # (`adapter.py`'s `check_reproducible`). Its `issues[]` names the spec_id
    # as `member` and each undeclared role as `axis`, with no `pointer`, so
    # this shares `request_invalid`'s code but not its pointer/role factory;
    # the mapped factory below reads `member` for `spec_id` and every `axis`
    # for `roles`.
    "definition_not_reproducible": ToolErrorCode.REQUEST_INVALID,
    # `SpecNotExportable` (`datasets_api/doors/mint_errors.py`, 409) fires when a
    # described-arm ask enumerates no member consistent with its own five-tuple and slots
    # against the graph.
    "spec_not_exportable": ToolErrorCode.SPEC_NOT_EXPORTABLE,
    # `member_not_resolvable` (422, not retryable) fires when the definition's StatVars
    # cannot be resolved as it stands: the graph's arcs for a member are inconsistent,
    # so the platform cannot confirm what the member measures and does not mint the
    # spec. `_member_not_resolvable_refusal` names the StatVars.
    "member_not_resolvable": ToolErrorCode.MEMBER_NOT_RESOLVABLE,
    # `RollupNotSummableError` (`datasets_api/doors/mint_errors.py`, 422): a
    # `get_data(stat_vars="sum")` mint over StatVars that do not add into one total.
    "rollup_not_summable": ToolErrorCode.ROLLUP_NOT_SUMMABLE,
    # Search's `dataset_resubmit` (422, `resolve_definition`): the posted identity
    # holds open a role the elected StatVars do not declare.
    "open_role_unavailable": ToolErrorCode.OPEN_ROLE_UNAVAILABLE,
}


def facet_issues_from(extensions: Mapping[str, Any]) -> list[FacetIssue]:
    """Datasets aggregates failing axes into `issues[]`; Search sets a single
    top-level `pointer`/`alternatives` pair with no `issues` array — both
    wire shapes are live for a client that calls both surfaces.
    """
    raw_issues = _objects(extensions.get("issues"))
    if raw_issues:
        return [_facet_issue_from_raw(raw) for raw in raw_issues]
    # `search/engine/region/facet_refusals.py` sets `pointer`
    # unconditionally on every raise but collapses `alternatives` to `None`
    # (omitted from the wire) whenever the failing axis has zero live
    # values — `pointer` is the reliable signal, so the branch keys off it,
    # not off `alternatives`.
    pointer = extensions.get("pointer")
    if not isinstance(pointer, str):
        return []
    return [
        FacetIssue(
            axis=pointer.rsplit("/", 1)[-1],
            alternatives=_facet_alternatives_from_raw(extensions.get("alternatives")),
            truncated=bool(extensions.get("truncated")),
        )
    ]


def _facet_issue_from_raw(raw: Mapping[str, Any]) -> FacetIssue:
    axis, code, detail = raw.get("axis"), raw.get("code"), raw.get("detail")
    return FacetIssue(
        axis=axis if isinstance(axis, str) else None,
        alternatives=_facet_alternatives_from_raw(raw.get("alternatives")),
        truncated=bool(raw.get("truncated")),
        code=code if isinstance(code, str) else None,
        detail=detail if isinstance(detail, str) else None,
    )


def _facet_alternatives_from_raw(raw: object) -> list[FacetAlternative]:
    return [
        FacetAlternative(ref=a["ref"], label=a["label"])
        for a in _objects(raw)
        if "ref" in a and "label" in a
    ]


def _objects(value: object) -> list[Mapping[str, Any]]:
    """The entries of `value` that are objects, when `value` is a list.

    apis sends `issues[]` and `alternatives[]` as lists of objects.
    Anything else is skipped, so a malformed entry costs the refusal that one
    entry's fields and the refusal itself still reaches the caller.
    """
    if not isinstance(value, list):
        return []
    return [cast("Mapping[str, Any]", entry) for entry in value if isinstance(entry, Mapping)]


def _first_issue(extensions: Mapping[str, Any]) -> Mapping[str, Any]:
    """The first object in the problem's `issues[]`, or an empty mapping when it holds none."""
    return next(iter(_objects(extensions.get("issues"))), {})


def _issue_axes(extensions: Mapping[str, Any]) -> list[str]:
    """Every axis the problem's `issues[]` names as a non-empty string, in order."""
    return [
        axis
        for issue in _objects(extensions.get("issues"))
        if isinstance(axis := issue.get("axis"), str) and axis
    ]


def _request_invalid_refusal(exc: UpstreamError) -> ToolRefusal:
    """`request_invalid` itself sets a top-level `pointer` with no `issues`.
    `invalid_role_ref`/`role_incompatible` set no top-level `pointer`,
    carrying the offending scope role instead in a single-entry `issues[]`
    (`mint.py`'s `InvalidRoleRef`/`RoleIncompatible` branches) — `role` stays
    unset for the plain shape, which names no axis.
    """
    pointer = exc.extensions.get("pointer")
    role = None
    if pointer is None:
        issue = _first_issue(exc.extensions)
        pointer = issue.get("pointer")
        role = issue.get("axis")
    return ToolRefusal.request_invalid(
        pointer=pointer,
        role=role,
        detail=exc.detail,
        trace_id=exc.trace_id,
        capability_version=exc.capability_version,
    )


def _definition_not_reproducible_refusal(exc: UpstreamError) -> ToolRefusal:
    """`DefinitionNotReproducible` (`mint_errors.py`) names the spec_id in
    `issues[].member` and each entity role the stored identity binds and the
    graph's current account of the StatVar no longer declares in an
    `issues[].axis` (`adapter.py`'s `check_reproducible`). The caller is
    `get_data`, which takes a spec_id and can only edit `scope.entities`, so
    the message points to the two moves `get_data` can make: rerun `search`
    for a fresh spec_id, or retry `get_data` with those roles left out of
    `scope.entities`. A body that names no role, such as a failed operation
    with no stored refusal, offers only the `search` move.
    """
    spec_id = _first_issue(exc.extensions).get("member")
    roles = list(dict.fromkeys(_issue_axes(exc.extensions)))
    fix: dict[str, Any] = {}
    if spec_id is not None:
        fix["spec_id"] = spec_id
    fix["upstream_code"] = "definition_not_reproducible"
    fix["role_bound"] = True
    if roles:
        fix["roles"] = roles
    fix["next_tool"] = "search"
    rerun = "Run search again without the role phrase and use the spec_id it returns."
    if roles:
        named = ", ".join(roles)
        message = (
            "The stored definition could not be rebuilt for this mint because it binds the "
            f"entity role(s) {named}, which the source no longer declares. {rerun} Or call "
            f"get_data again with scope.entities, leaving out {named}."
        )
    else:
        message = (
            "The stored definition could not be rebuilt for this mint because it binds an "
            f"entity role the source no longer declares. {rerun}"
        )
    return ToolRefusal(
        code=ToolErrorCode.REQUEST_INVALID,
        message=message,
        fix=fix,
        trace_id=exc.trace_id,
        capability_version=exc.capability_version,
    )


def _rollup_not_summable_refusal(exc: UpstreamError) -> ToolRefusal:
    """`RollupNotSummableError`'s refusal names its axis in `issues[]`. A failed
    operation whose body names no axis carries only its refusal's code, and its
    `detail` is the poll route's sentence about the operation, which says
    nothing of the refusal and stays out of `fix`.
    """
    axes = _issue_axes(exc.extensions)
    carries_refusal = bool(axes) or exc.code != _OPERATION_FAILED_CODE
    return ToolRefusal.rollup_not_summable(
        axes=axes,
        detail=exc.detail if carries_refusal else None,
        trace_id=exc.trace_id,
        capability_version=exc.capability_version,
    )


# Every `ceiling_kind` apis' Datasets surface raises, mapped to the kind
# this module names it by. The `sync_` and `operation_` kinds are the admission
# ceilings of a preview (`refuse_preview_over_budget`) and of an operation
# (`refuse_async_over_ceiling`), counted in the same units as the inline ones.
_UPSTREAM_CEILING_KINDS: dict[str, CeilingKind] = {
    "rows": "rows",
    "sync_rows": "rows",
    "operation_rows": "rows",
    "bytes": "bytes",
    "sync_requests": "requests",
    "operation_requests": "requests",
    "member_cardinality": "member_cardinality",
    "candidate_window": "candidate_window",
}


def _ceiling_kind_from_extensions(extensions: Mapping[str, Any]) -> CeilingKind | None:
    """The kind the wire's `ceiling_kind` names, or `None` when it names none
    `_UPSTREAM_CEILING_KINDS` maps.

    apis publishes `ceiling_kind` as a top-level RFC 9457 problem-document
    member, which `_to_upstream_error` files into `UpstreamError.extensions`
    with every other member it doesn't name explicitly. A failed operation
    carries it when the worker stored the refusal. An absent or unmapped kind
    is a ceiling of unknown kind.
    """
    wire_kind = extensions.get("ceiling_kind")
    return _UPSTREAM_CEILING_KINDS.get(wire_kind) if isinstance(wire_kind, str) else None


def _int_member(extensions: Mapping[str, Any], key: str) -> int | None:
    """The integer extension member `key`, or `None` when it is absent or not an integer."""
    value = extensions.get(key)
    return value if isinstance(value, int) and not isinstance(value, bool) else None


# How every spec_id starts (apis `SpecIdInput.spec_id`). A `member_not_resolvable`
# issue names the spec_id as its `member` when the identity as a whole did not resolve.
_SPEC_ID_PREFIX = "spec_"


# The end-user remedy for a `member_not_resolvable` refusal.
_MEMBER_NOT_RESOLVABLE_REMEDY = (
    "This dataset holds indicators the platform cannot confirm, so its figures cannot be "
    "fetched. Ask for them from another source."
)


def _member_not_resolvable_refusal(exc: UpstreamError) -> ToolRefusal:
    """`member_not_resolvable` as a refusal listing its StatVars in `fix`.

    Each issue names one StatVar or spec_id whose graph arcs could not be resolved in
    `member`: `fix.members` lists them all, and the message counts the StatVars alone,
    as indicators, since a spec_id is the dataset and no StatVar of it. The message names
    no StatVar, since a model reads it and repeats it to the user. A failed operation
    with no stored refusal carries no issues, so its refusal counts no StatVar.
    `fix.remedy` is written for the end user, like `over_ceiling`'s.
    """
    members = [
        issue["member"]
        for issue in _objects(exc.extensions.get("issues"))
        if isinstance(issue.get("member"), str) and issue["member"]
    ]
    stat_vars = [member for member in members if not member.startswith(_SPEC_ID_PREFIX)]
    counted = f" ({len(stat_vars):,} the graph defines inconsistently)" if stat_vars else ""
    message = (
        f"This dataset holds indicators the platform cannot confirm{counted}, so this spec "
        "can't be minted. Retrying it refuses the same way -- search again for a "
        "different ask."
    )
    return ToolRefusal(
        code=ToolErrorCode.MEMBER_NOT_RESOLVABLE,
        message=message,
        fix={"members": members, "remedy": _MEMBER_NOT_RESOLVABLE_REMEDY},
        trace_id=exc.trace_id,
        capability_version=exc.capability_version,
    )


def _map_known_upstream_code(code: str | None, exc: UpstreamError) -> ToolRefusal | None:
    """Dispatch the upstream-exact codes to their typed factories.

    `code` is the refusal's own code (`refusal_code`). Several upstream codes
    can share one entry here — `_UPSTREAM_CODE_MAP` is what narrows many wire
    codes onto few `ToolErrorCode`s, this table only needs one factory per
    `ToolErrorCode`. Returns `None` for `upstream_error` and anything
    unmapped, so the caller falls through to the `upstream_error` factory. A
    dispatch table rather than an if-chain, so a growing set of exact codes
    doesn't trip PLR0911.

    A failed operation's body carries its refusal's code, and that refusal's
    own fields only when the worker stored it, so each factory falls back on
    what it cannot read: no `issues[]`, the operation's own `detail`, and no
    `ceiling_kind`.
    """
    tool_code = _UPSTREAM_CODE_MAP.get(code) if code is not None else None
    factories: dict[ToolErrorCode, Callable[[], ToolRefusal]] = {
        ToolErrorCode.FACET_NOT_AVAILABLE: lambda: ToolRefusal.facet_not_available(
            issues=facet_issues_from(exc.extensions),
            detail=exc.detail,
            trace_id=exc.trace_id,
            capability_version=exc.capability_version,
        ),
        ToolErrorCode.DEFAULT_FACET_UNDEFINED: lambda: ToolRefusal.default_facet_undefined(
            issues=facet_issues_from(exc.extensions),
            detail=exc.detail,
            retryable=exc.retryable,
            trace_id=exc.trace_id,
            capability_version=exc.capability_version,
        ),
        ToolErrorCode.OVER_CEILING: lambda: ToolRefusal.over_ceiling(
            # `OverCeilingError` and `ParquetSizeBoundError`
            # (`materialise/errors.py`, `materialise/formats.py`) both set
            # `detail` at every real raise site; the fallback only satisfies
            # `exc.detail`'s generic `str | None` parse-time type.
            detail=exc.detail or "the coverage ceiling was exceeded",
            ceiling_kind=_over_ceiling_kind(code, exc),
            # Siblings of `ceiling_kind` in the same problem document;
            # absent on the member-cardinality branch where the enumeration
            # can't page fast enough to produce either.
            count=_int_member(exc.extensions, "count"),
            cap=_int_member(exc.extensions, "cap"),
            trace_id=exc.trace_id,
            capability_version=exc.capability_version,
        ),
        ToolErrorCode.CAPABILITY_UNAVAILABLE: lambda: ToolRefusal.capability_unavailable(
            detail=exc.detail, trace_id=exc.trace_id, capability_version=exc.capability_version
        ),
        ToolErrorCode.UPSTREAM_VERSION_CHANGED: lambda: ToolRefusal.upstream_version_changed(
            capability_version=exc.capability_version, trace_id=exc.trace_id
        ),
        ToolErrorCode.SERVICE_UNAVAILABLE: lambda: ToolRefusal.service_unavailable(
            # Every raise site for this code emits the same generic body
            # (store, registry, graph/LLM infra, and door backpressure
            # alike), so "an apis service dependency" is what's actually
            # knowable. Most raise sites do name what's backed up in
            # `detail` (e.g. mint.py, register.py); only the search
            # engine's infra-failure path (app.py) carries none.
            service="a dfo-apis dependency",
            retry_after_s=_derive_retry_after_s(exc),
            detail=exc.detail,
            trace_id=exc.trace_id,
            capability_version=exc.capability_version,
        ),
        ToolErrorCode.REQUEST_INVALID: lambda: (
            _definition_not_reproducible_refusal(exc)
            if code == "definition_not_reproducible"
            else _request_invalid_refusal(exc)
        ),
        ToolErrorCode.SPEC_NOT_EXPORTABLE: lambda: ToolRefusal.spec_not_exportable(
            detail=exc.detail or "this spec resolves to no exportable member",
            trace_id=exc.trace_id,
            capability_version=exc.capability_version,
        ),
        ToolErrorCode.MEMBER_NOT_RESOLVABLE: lambda: _member_not_resolvable_refusal(exc),
        ToolErrorCode.ROLLUP_NOT_SUMMABLE: lambda: _rollup_not_summable_refusal(exc),
        ToolErrorCode.OPEN_ROLE_UNAVAILABLE: lambda: ToolRefusal.open_role_unavailable(
            pointer=exc.extensions.get("pointer"),
            detail=exc.detail,
            trace_id=exc.trace_id,
            capability_version=exc.capability_version,
        ),
    }
    factory = factories.get(tool_code) if tool_code is not None else None
    return factory() if factory is not None else None


# The poll route's code for an operation that ran and did not complete
# (apis' `OperationFailed`). Its body names the refusal that failed the
# operation as `cause: {code}`. When the worker stored that refusal, the body
# also carries its detail, pointer, issues and ceiling fields; otherwise its
# detail is the poll route's own sentence and it carries none of them.
_OPERATION_FAILED_CODE = "operation_failed"

# The causes a failed operation relays as `upstream_error` with a wait, where the
# synchronous refusal of the same code has a typed factory: submitting the same mint
# again fixes a source revision that moved mid-materialisation, and discovering the
# source again changes nothing about it.
_FAILED_OPERATION_CAUSES_RETRIED_IN_PLACE = frozenset({"upstream_version_changed"})


def _over_ceiling_kind(code: str | None, exc: UpstreamError) -> CeilingKind | None:
    """Which ceiling an `over_ceiling`-mapped refusal names, or `None` when it cannot say.

    `parquet_size_bound` is the one upstream code this map sends to
    `over_ceiling` that is never a row or member ceiling -- a structured wire
    fact, read from the code ahead of the wire's own `ceiling_kind`
    extension.
    """
    if code == "parquet_size_bound":
        return "bytes"
    return _ceiling_kind_from_extensions(exc.extensions)


def refusal_code(exc: UpstreamError) -> str | None:
    """The code of the refusal `exc` reports, which picks its `ToolRefusal`.

    A failed operation names the refusal that failed it in `cause.code`, the
    code a caller can act on, so that is read in place of
    `operation_failed`. Every other problem reports its own `code`, including
    one that carries a `cause` of its own (`recipe_not_declared`,
    `rate_coverage_gap`).
    """
    if exc.code != _OPERATION_FAILED_CODE:
        return exc.code
    cause = exc.extensions.get("cause")
    cause_code = cause.get("code") if isinstance(cause, Mapping) else None
    return cause_code if isinstance(cause_code, str) and cause_code else exc.code


def from_upstream_error(exc: UpstreamError) -> ToolRefusal:
    """Translate an apis service call failure into the tool-facing refusal grammar.

    The refusal's own code (`refusal_code`) picks the refusal: the problem's
    `code`, or for a failed operation (`operation_failed`) the code of the
    refusal that failed it. A mapped code reaches its typed factory either
    way, so a deferred mint that `rollup_not_summable` failed refuses with the
    code, retry mode, and remedy of the synchronous one. An unmapped code is
    an `upstream_error` refusal relayed under that code, and its `retryable`
    flag says whether submitting the mint again could succeed. A failed
    operation whose cause is in `_FAILED_OPERATION_CAUSES_RETRIED_IN_PLACE` is
    relayed the same way, always with a wait.
    """
    if isinstance(exc, UpstreamSurfaceUnavailable):
        # An unmounted surface counts as `capability_unavailable` —
        # `errors.py`'s own docstring says the id "may well exist once the
        # surface is deployed", which is `do_not_retry` framing.
        return ToolRefusal.capability_unavailable(
            detail=exc.detail, trace_id=exc.trace_id, capability_version=exc.capability_version
        )
    if isinstance(exc, UpstreamTransportError):
        # No response ever came back — closer to "the store is unreachable"
        # than to a graph/SDMX 502, so it gets `retry_after_delay` rather than
        # folding into the generic `upstream_error` bucket.
        return ToolRefusal.service_unavailable(
            service="dfo-apis",
            detail=exc.detail,
            retry_after_s=_derive_retry_after_s(exc),
            trace_id=exc.trace_id,
            capability_version=exc.capability_version,
        )

    code = refusal_code(exc)
    retried_in_place = (
        exc.code == _OPERATION_FAILED_CODE and code in _FAILED_OPERATION_CAUSES_RETRIED_IN_PLACE
    )
    if not retried_in_place:
        mapped = _map_known_upstream_code(code, exc)
        if mapped is not None:
            return mapped
    # The refusal's code — the explicit `upstream_error` entry or any other,
    # including `None` — survives verbatim as `upstream_code`. `retry_mode`
    # reads a fixed `retry_after_delay` from `_RETRY_MODE_BY_CODE` regardless
    # of `exc.retryable`, since the static per-code table can't vary by
    # instance. The mapped codes avoid this gap because their `retryable`
    # value is a fixed upstream constant per code, verified against real
    # apis raise sites. A failed operation records `retryable: false` for
    # a refusal whose own flag is false (apis `operations/worker.py`), so
    # a mapped cause keeps it too.
    retry_after_s = _derive_retry_after_s(exc) if exc.retryable or retried_in_place else None
    return ToolRefusal.upstream_error(
        upstream_code=code,
        detail=exc.detail,
        retry_after_s=retry_after_s,
        trace_id=exc.trace_id,
        capability_version=exc.capability_version,
    )
