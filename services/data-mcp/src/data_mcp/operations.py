"""Resolving a mint's 202 (`operation submitted`) response to a manifest.

`get_data_tool.py`'s own mint POST answers 202, not 200, once apis' admission
logic (`series_coverage`-keyed) finds a request over its sync budget: the body is an
`OperationResource` (apis' own `datasets_api.schema.OperationResource`,
mirrored here as this module's own pydantic models rather than imported, since this
service and apis do not share a schema package) naming an `operation_id` this
call can poll: `GET {poll}` (`datasets_api.doors.meta`'s operations route). This
module is what turns that 202 into the same manifest a synchronous (200) mint would
have returned, or a refusal if the operation fails, disappears, or does not finish
before the call deadline leaves only its post-mint reserve
(`CallBudget.post_mint_reserve_s`).

`resolve_operation_response` is the module's public seam: `get_data_tool
._mint_payload_or_refusal` calls it once, immediately after a mint POST comes back
202, and reads its `ResolvedOperation.manifest_payload` the same way a 200
response's own parsed body is treated -- ready for `data_mcp.manifest.strip_manifest`.

`operation_without_waiting` is the same seam for a caller that must not wait:
it returns a `QueuedOperation` at once for an operation still pending or running,
and fetches the manifest of one already done. `get_data_tool.mint_recorded_root`
calls it.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, replace
from typing import Any, Literal

from fastmcp.tools import ToolResult
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from data_mcp.client import SleepFn, UpstreamClient, is_retryable
from data_mcp.deadline import DEADLINE_RETRY_AFTER_S, CallDeadline, current_call_budget
from data_mcp.errors import UpstreamError
from data_mcp.tool_errors import ToolRefusal, from_upstream_error, to_tool_result

logger = logging.getLogger(__name__)


async def _default_sleep(seconds: float) -> None:
    """Real sleep; tests inject a fake to keep this module's own budget assertions fast."""
    await asyncio.sleep(seconds)


# The poll route's own path template. Doubles as `client.request`'s own `route=`
# span attribute -- kept as one literal rather than derived from a per-operation
# `poll` field, which carries the concrete `operation_id` and would blow up that
# attribute's cardinality across operations -- and as the only source `_poll_once`
# ever builds an actual request path from, never the upstream-supplied `poll` field.
_POLL_ROUTE_TEMPLATE = "/api/datasets/operations/{operation_id}"

_TERMINAL_STATUSES = ("done", "failed")

# How often `on_poll` repeats a report while the status hasn't moved. A status
# change reports immediately regardless of this interval; this only bounds the
# silence during a long `pending`/`running` stretch.
_POLL_REPORT_INTERVAL_S = 10.0


class OperationEnvelope(BaseModel):
    """This service's own mirror of apis' `datasets_api.schema.OperationEnvelope`.

    `retry_after_ms` is set only on the very first envelope this call ever sees (the
    202 submit response). Every poll response after that carries none, by apis'
    own contract (that field's wire docstring), so a later envelope's `None` here
    means "no pacing guidance", not "no delay".

    `poll` is apis' own suggested poll path, parsed but never dialled: `_poll_once`
    always builds its request path from `operation_id` and `_POLL_ROUTE_TEMPLATE`
    instead, so an upstream-controlled string can never become a request target this
    client's own auth headers would ride along with. `operation_id` is constrained to
    apis' own wire format (`op_` plus a sha256 hex digest,
    `datasets_api.operation_store.compute_operation_id`) so a malformed id refuses rather
    than reaching `_POLL_ROUTE_TEMPLATE.format` unchecked.
    """

    model_config = ConfigDict(extra="ignore")

    operation_id: str = Field(pattern=r"^op_[0-9a-f]{64}$")
    status: Literal["pending", "running", "done", "failed"]
    poll: str | None = None
    retry_after_ms: int | None = None


class OperationResource(BaseModel):
    """This service's own mirror of apis' `datasets_api.schema.OperationResource`.

    `manifest` is set only once a poll response's own `operation.status == "done"`;
    the 202 submit response never carries one even when `operation.status` is
    already `"done"` (a resubmit racing a worker's own completion) -- see
    `resolve_operation_response`'s own handling of that case. `error` is declared on
    the wire but never populated (apis' own `OperationResource.error` docstring:
    a `failed` operation raises an RFC 9457 problem instead), so it is not modelled
    here at all.
    """

    model_config = ConfigDict(extra="ignore")

    operation: OperationEnvelope
    manifest: dict[str, Any] | None = None


@dataclass(frozen=True, kw_only=True)
class ResolvedOperation:
    """A 202 operation resolved all the way to its manifest.

    `manifest_payload` is exactly what a synchronous (200) mint's own response body
    would have been -- the caller feeds it to `data_mcp.manifest.strip_manifest`
    unchanged. `poll_attempts` and `elapsed_s` are forwarded into the digest (see
    `data_mcp.get_data_digest.OperationPollNote`) so the model is told this mint
    queued rather than ran inline, not left to infer it from timing alone.
    """

    manifest_payload: dict[str, Any]
    operation_id: str
    poll_attempts: int
    elapsed_s: float


@dataclass(frozen=True, kw_only=True)
class QueuedOperation:
    """A 202 operation apis has not finished, as `operation_without_waiting`
    left it. `status` is the last apis reported. `retry_after_s` is the wait
    apis named on the submit response, else the caller's poll interval. Sending
    the same mint again after it rejoins this operation: apis derives
    `operation_id` from the request, and answers a repeat with the operation it
    already holds."""

    operation_id: str
    status: Literal["pending", "running", "done", "failed"]
    retry_after_s: float


@dataclass(frozen=True, kw_only=True)
class PollUpdate:
    """One `on_poll` report: what `resolve_operation_response` knows about the
    operation right now, for a caller to show as it waits."""

    operation_id: str
    status: Literal["pending", "running", "done", "failed"]
    elapsed_s: float
    poll_attempts: int


# Awaited once the 202 submit response is parsed, then again on a status
# change or every `_POLL_REPORT_INTERVAL_S` of waiting -- never on every poll.
# A callback that raises is logged and ignored: reporting progress never fails
# the call it describes. Kept as a plain callable, not a `Context` method,
# so this module stays free of any FastMCP import; `get_data_tool` supplies
# the callback that actually calls `Context.report_progress`.
OnPoll = Callable[[PollUpdate], Awaitable[None]]


def _malformed_operation_response(*, detail: str) -> ToolResult:
    return to_tool_result(ToolRefusal.upstream_response_unusable(detail=detail))


def _parse_operation_resource(
    payload: Mapping[str, Any], *, context: str
) -> OperationResource | ToolResult:
    try:
        return OperationResource.model_validate(payload)
    except ValidationError as exc:
        return _malformed_operation_response(
            detail=f"the {context} did not match the operation envelope shape this service expects: {exc}"
        )


async def _poll_once(
    client: UpstreamClient, envelope: OperationEnvelope, *, deadline: CallDeadline
) -> OperationResource | ToolResult:
    """One `GET` against `envelope`'s own poll route, bounded by `deadline`.

    `retry=False`: the caller owns retrying. `resolve_operation_response`'s wait loop
    polls again after a retryable failure, paced like any other poll, so a second
    retry loop underneath would only spend the call deadline twice.

    Raises:
        UpstreamError: the poll failed, including `operation_failed` (the operation
            ran and did not complete) and `operation_not_found` (the poll route does
            not know the id). The caller decides whether to poll again or refuse.
    """
    # Built from `operation_id`, never from `envelope.poll` -- see
    # `OperationEnvelope`'s own docstring for why that upstream-supplied field is
    # parsed but never used as a request target.
    poll_path = _POLL_ROUTE_TEMPLATE.format(operation_id=envelope.operation_id)
    response = await client.request(
        "GET", poll_path, route=_POLL_ROUTE_TEMPLATE, retry=False, deadline=deadline
    )
    try:
        payload = response.json()
    except ValueError as exc:
        return _malformed_operation_response(
            detail=f"the operation poll response body could not be parsed as JSON: {exc}"
        )
    return _parse_operation_resource(payload, context="operation poll response")


async def _poll_and_pin(
    client: UpstreamClient,
    envelope: OperationEnvelope,
    *,
    deadline: CallDeadline,
    operation_id: str,
) -> OperationResource | ToolResult:
    """One `_poll_once`, refusing a response that names a different operation.

    `operation_id` is the one pinned from the 202 submit response; a poll response
    naming another is malformed, whatever its own status, since this call has no
    way to know which operation's outcome it would be trusting.
    """
    polled = await _poll_once(client, envelope, deadline=deadline)
    if isinstance(polled, ToolResult):
        return polled
    if polled.operation.operation_id != operation_id:
        return _malformed_operation_response(
            detail=f"polled operation {operation_id!r} but the response named "
            f"{polled.operation.operation_id!r}"
        )
    return polled


def _delay_before_next_poll(
    envelope: OperationEnvelope, *, polled: bool, last_poll_failed: bool, default_s: float
) -> float:
    """How long to wait before the next poll.

    Zero for a `done`/`failed` submit envelope not yet polled, whose outcome is
    already decided upstream. `default_s` after a failed poll. Otherwise the
    envelope's own `retry_after_ms` when set, else `default_s`.
    """
    if not polled and envelope.status in _TERMINAL_STATUSES:
        return 0.0
    if last_poll_failed or envelope.retry_after_ms is None:
        return default_s
    return envelope.retry_after_ms / 1000.0


async def _wait_for_next_poll(
    delay_s: float, *, poll_deadline: CallDeadline, sleep: SleepFn
) -> bool:
    """Sleep out `delay_s`, clamped to what's left of `poll_deadline`.

    A server-named delay that would overrun the deadline is cut short, the same
    wait the caller was always going to spend checking one more time before giving
    up. Returns `False` when `poll_deadline` has passed, before or after sleeping,
    and `True` once it's safe to poll again.
    """
    remaining_s = poll_deadline.remaining()
    if remaining_s <= 0:
        return False
    if (clamped_s := min(delay_s, remaining_s)) > 0:
        await sleep(clamped_s)
    return poll_deadline.remaining() > 0


def _still_processing_refusal(
    envelope: OperationEnvelope,
    *,
    elapsed_s: float,
    last_poll_error: UpstreamError | None,
) -> ToolResult:
    """The refusal for a wait that ended without a manifest.

    `envelope` carries the last status apis reported, and `last_poll_error` the
    last of the upstream failures since then, if every poll since failed.
    """
    error_name = None
    if last_poll_error is not None:
        error_name = last_poll_error.code or "a transport error"
    logger.warning(
        "get_data operation poll: %s ended its wait after %.1fs, last reported %s%s",
        envelope.operation_id,
        elapsed_s,
        envelope.status,
        f", polls since ending with {error_name}" if error_name else "",
    )
    return to_tool_result(
        ToolRefusal.operation_still_processing(
            operation_id=envelope.operation_id,
            last_status=envelope.status,
            last_poll_error=error_name,
            elapsed_s=elapsed_s,
            retry_after_s=DEADLINE_RETRY_AFTER_S,
        )
    )


async def resolve_operation_response(
    *,
    client: UpstreamClient,
    submit_payload: Mapping[str, Any],
    deadline: CallDeadline,
    poll_interval_s: float,
    sleep: SleepFn = _default_sleep,
    on_poll: OnPoll | None = None,
    raise_upstream_error: bool = False,
) -> ResolvedOperation | ToolResult:
    """Resolve a mint POST's own 202 body to a manifest, polling as needed.

    Args:
        client: The same upstream client the mint POST itself used.
        submit_payload: The 202 response's own parsed JSON body -- an
            `OperationResource` with `operation.status` almost always `"pending"`,
            occasionally `"running"`, `"done"`, or `"failed"` (a resubmit racing a
            worker; see apis' own `operation_submitted_resource` docstring).
        deadline: The `get_data` call's deadline. The wait loop polls until the
            operation reaches `done` with a manifest, and stops once less than the
            current `CallBudget.post_mint_reserve_s` of it remains. A
            `"done"`/`"failed"` submit envelope carries no manifest or problem body, so
            its first fetch runs at once, against `deadline` itself since the outcome is
            already decided upstream; a retry of it waits and polls like any other.
        poll_interval_s: Fallback wait between polls when the current envelope
            names no `retry_after_ms` of its own -- `Settings
            .operation_poll_interval_s`. Every poll response but the first carries
            none, by apis' own contract (`OperationEnvelope.retry_after_ms`'s
            own docstring).
        sleep: Backoff sleep, replaced in tests to avoid waiting.
        on_poll: Awaited once the submit response is parsed, then again on a
            status change or every `_POLL_REPORT_INTERVAL_S` of waiting -- see
            `PollUpdate`'s own docstring. `None` skips reporting entirely.
        raise_upstream_error: Raise a non-retryable poll `UpstreamError` for a
            caller that can correct and retry the mint; defaults to a refusal.

    A poll that fails with an `UpstreamError` the client itself would retry
    (`data_mcp.client.is_retryable`: a transport failure, or the poll door's
    backpressure 503) does not end the wait: the loop polls again at
    `poll_interval_s`, still inside the poll deadline. Any other is refused through
    `from_upstream_error`. `operation_failed` is never polled again, whatever
    `retryable` its body carries (`data_mcp.client.NON_RETRYABLE_CODES`): the
    operation has ended, and the refusal carries the code of the refusal that
    failed it.

    The `operation_id` the 202 submit response names is pinned for the rest of the
    call: a later poll response naming a different one is malformed, whatever its
    own status, since this call has no way to know which operation's outcome it
    would be trusting.

    Returns:
        A `ResolvedOperation` once the operation reaches `done`, or a refusal
        `ToolResult` for `failed`/a non-retryable upstream error/a malformed
        body/the poll reserve reached. A poll response reporting `done` or `failed`
        without its manifest is malformed.
    """
    resource = _parse_operation_resource(submit_payload, context="202 operation response")
    if isinstance(resource, ToolResult):
        return resource

    # Pinned from the 202 submit response; every later poll must keep reporting
    # this same id, since a different one would mean this call started tracking
    # some other operation's outcome.
    operation_id = resource.operation.operation_id
    start = deadline.clock()
    poll_deadline = deadline.reserving(current_call_budget().post_mint_reserve_s)
    poll_attempts = 0
    # Whether `resource` came from a successful poll rather than the submit response.
    from_poll = False
    last_poll_error: UpstreamError | None = None

    # Throttling state for `on_poll`: the status and elapsed time of the last report
    # sent, so a repeat only fires on a status change or once `_POLL_REPORT_INTERVAL_S`
    # has passed since -- never on every poll.
    last_reported_status: str | None = None
    last_reported_at_s: float | None = None

    async def _report(envelope: OperationEnvelope, *, elapsed_s: float) -> None:
        nonlocal last_reported_status, last_reported_at_s
        if on_poll is None:
            return
        status_changed = envelope.status != last_reported_status
        due = (
            last_reported_at_s is None or elapsed_s - last_reported_at_s >= _POLL_REPORT_INTERVAL_S
        )
        if not status_changed and not due:
            return
        last_reported_status = envelope.status
        last_reported_at_s = elapsed_s
        try:
            await on_poll(
                PollUpdate(
                    operation_id=envelope.operation_id,
                    status=envelope.status,
                    elapsed_s=elapsed_s,
                    poll_attempts=poll_attempts,
                )
            )
        except Exception:
            logger.warning(
                "get_data operation poll: %s's on_poll callback raised; ignored",
                envelope.operation_id,
                exc_info=True,
            )

    # The 202 is accepted the moment its body parses; this is that one report.
    await _report(resource.operation, elapsed_s=deadline.clock() - start)

    while True:
        envelope = resource.operation
        if envelope.status == "done" and resource.manifest is not None:
            elapsed_s = deadline.clock() - start
            logger.info(
                "get_data operation poll: %s resolved after %d poll attempt(s), %.1fs",
                envelope.operation_id,
                poll_attempts,
                elapsed_s,
            )
            return ResolvedOperation(
                manifest_payload=resource.manifest,
                operation_id=envelope.operation_id,
                poll_attempts=poll_attempts,
                elapsed_s=elapsed_s,
            )
        if from_poll and envelope.status in _TERMINAL_STATUSES:
            # A poll response, not the submit one, reported a terminal state without
            # the manifest (or, for `failed`, the problem body) that state carries.
            return _malformed_operation_response(
                detail=f"operation {envelope.operation_id!r}'s poll response reports "
                f"{envelope.status!r} but carries no manifest"
            )

        first_terminal_fetch = poll_attempts == 0 and envelope.status in _TERMINAL_STATUSES
        bound = deadline if first_terminal_fetch else poll_deadline
        delay_s = _delay_before_next_poll(
            envelope,
            polled=poll_attempts > 0,
            last_poll_failed=last_poll_error is not None,
            default_s=poll_interval_s,
        )
        if not await _wait_for_next_poll(delay_s, poll_deadline=bound, sleep=sleep):
            return _still_processing_refusal(
                envelope,
                elapsed_s=deadline.clock() - start,
                last_poll_error=last_poll_error,
            )

        poll_attempts += 1
        try:
            polled = await _poll_and_pin(
                client, envelope, deadline=bound, operation_id=operation_id
            )
        except UpstreamError as exc:
            if not is_retryable(exc):
                if raise_upstream_error:
                    raise
                return to_tool_result(from_upstream_error(exc, tool="get_data"))
            if bound.remaining() > 0:
                logger.info(
                    "get_data operation poll: %s poll attempt %d failed retryably (%s); "
                    "polling again",
                    envelope.operation_id,
                    poll_attempts,
                    exc.code or exc.detail,
                )
                last_poll_error = exc
            # Otherwise this call's own bound cut the poll, which says nothing about
            # the operation: the last reported status stands, and the next wait ends
            # the loop.
            continue
        if isinstance(polled, ToolResult):
            return polled
        resource = polled
        from_poll = True
        last_poll_error = None
        await _report(resource.operation, elapsed_s=deadline.clock() - start)


async def operation_without_waiting(
    *,
    client: UpstreamClient,
    submit_payload: Mapping[str, Any],
    deadline: CallDeadline,
    poll_interval_s: float,
    raise_upstream_error: bool = False,
) -> ResolvedOperation | QueuedOperation | ToolResult:
    """Read a mint POST's 202 body without waiting for the operation to finish.

    An operation the submit response reports `pending` or `running` comes back as a
    `QueuedOperation` and nothing is polled. One it reports `done` or `failed`
    carries no manifest or problem body on that response, so it is polled once,
    at once, against `deadline`. `done` gives the `ResolvedOperation`, and `failed`
    the refusal `resolve_operation_response` gives it. A poll that fails
    retryably leaves the operation a `QueuedOperation` with the status the submit
    response reported.

    Args:
        client: The same upstream client the mint POST itself used.
        submit_payload: The 202 response's own parsed JSON body.
        deadline: Bounds the one poll.
        poll_interval_s: The `retry_after_s` of a `QueuedOperation` whose submit
            response named no `retry_after_ms`.
    """
    resource = _parse_operation_resource(submit_payload, context="202 operation response")
    if isinstance(resource, ToolResult):
        return resource
    submitted = resource.operation
    queued = QueuedOperation(
        operation_id=submitted.operation_id,
        status=submitted.status,
        retry_after_s=(
            poll_interval_s
            if submitted.retry_after_ms is None
            else submitted.retry_after_ms / 1000.0
        ),
    )
    if submitted.status not in _TERMINAL_STATUSES:
        return queued
    return await _fetch_finished(
        client,
        submitted,
        queued=queued,
        deadline=deadline,
        raise_upstream_error=raise_upstream_error,
    )


async def _fetch_finished(
    client: UpstreamClient,
    submitted: OperationEnvelope,
    *,
    queued: QueuedOperation,
    deadline: CallDeadline,
    raise_upstream_error: bool,
) -> ResolvedOperation | QueuedOperation | ToolResult:
    """`operation_without_waiting`'s one poll of an operation whose submit response
    reported it `done` or `failed`. `queued` is what a poll that settles nothing
    returns."""
    start = deadline.clock()
    try:
        polled = await _poll_and_pin(
            client, submitted, deadline=deadline, operation_id=submitted.operation_id
        )
    except UpstreamError as exc:
        if raise_upstream_error and not is_retryable(exc):
            raise
        return (
            queued
            if is_retryable(exc)
            else to_tool_result(from_upstream_error(exc, tool="get_data"))
        )
    if isinstance(polled, ToolResult):
        return polled
    if polled.operation.status == "done" and polled.manifest is not None:
        return ResolvedOperation(
            manifest_payload=polled.manifest,
            operation_id=submitted.operation_id,
            poll_attempts=1,
            elapsed_s=deadline.clock() - start,
        )
    if polled.operation.status in _TERMINAL_STATUSES:
        return _malformed_operation_response(
            detail=f"operation {submitted.operation_id!r}'s poll response reports "
            f"{polled.operation.status!r} but carries no manifest"
        )
    return replace(queued, status=polled.operation.status)
