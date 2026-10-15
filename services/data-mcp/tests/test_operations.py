"""Tests for `data_mcp.operations`: resolving a mint's 202 response to a manifest."""

from __future__ import annotations

import asyncio
import hashlib
from collections.abc import Awaitable, Callable
from typing import Any

import httpx2
import pytest
from data_mcp.deadline import DIRECT_CALL_BUDGET, CallDeadline
from data_mcp.operations import (
    PollUpdate,
    QueuedOperation,
    ResolvedOperation,
    operation_without_waiting,
    resolve_operation_response,
)
from fastmcp.tools import ToolResult

from .net_guard import no_network  # noqa: F401 -- imported for its autouse fixture
from .test_client import _client, _problem_body, _recording_sleep, _settings

# Shaped like apis' own `op_` + sha256-hex ids (`OperationEnvelope.operation_id`'s
# own pattern constraint) so these tests exercise the real validated shape rather than
# a placeholder a stricter pattern would itself reject.
_OPERATION_ID = "op_" + hashlib.sha256(b"test-operation").hexdigest()
_POLL_PATH = f"/api/datasets/operations/{_OPERATION_ID}"


def _deadline(*, poll_budget_s: float) -> CallDeadline:
    """A call deadline that leaves the wait loop `poll_budget_s` before its poll reserve."""
    return CallDeadline.after(poll_budget_s + DIRECT_CALL_BUDGET.post_mint_reserve_s)


def _submitted(
    *, status: str, retry_after_ms: int | None = None, poll: str | None = _POLL_PATH
) -> dict[str, object]:
    envelope: dict[str, object] = {"operation_id": _OPERATION_ID, "status": status}
    if poll is not None:
        envelope["poll"] = poll
    if retry_after_ms is not None:
        envelope["retry_after_ms"] = retry_after_ms
    return {"operation": envelope}


def _poll_response(*, status: str, manifest: dict[str, object] | None = None) -> httpx2.Response:
    body: dict[str, object] = {"operation": {"operation_id": _OPERATION_ID, "status": status}}
    if manifest is not None:
        body["manifest"] = manifest
    return httpx2.Response(200, json=body)


def _real_recording_sleep() -> tuple[list[float], Callable[[float], Awaitable[None]]]:
    """Records the delay it was asked for, and actually waits it out.

    `test_client.py`'s own `_recording_sleep` never really waits, which is right for
    proving what backoff a caller *asked* for -- but this module's own poll budget is
    judged against real wall-clock (`time.monotonic()`), so a test proving the budget
    itself expires needs the clock to really move. Kept sub-second throughout, the same
    way `test_client.py`'s own budget-expiry tests use a tiny *real* `upstream_retry_budget_s`
    rather than a faked clock.
    """
    delays: list[float] = []

    async def sleep(seconds: float) -> None:
        delays.append(seconds)
        await asyncio.sleep(seconds)

    return delays, sleep


async def test_202_then_done_returns_the_manifest() -> None:
    manifest_payload = {"content_version": "cv_1", "definition": {"spec_id": "spec_x"}}
    requests: list[str] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        requests.append(request.url.path)
        assert request.url.path == _POLL_PATH
        return _poll_response(status="done", manifest=manifest_payload)

    delays, sleep = _recording_sleep()
    async with _client(handler) as client:
        result = await resolve_operation_response(
            client=client,
            submit_payload=_submitted(status="pending", retry_after_ms=1500),
            deadline=_deadline(poll_budget_s=5.0),
            poll_interval_s=1.0,
            sleep=sleep,
        )

    assert isinstance(result, ResolvedOperation)
    assert result.manifest_payload == manifest_payload
    assert result.operation_id == _OPERATION_ID
    assert result.poll_attempts == 1
    assert requests == [_POLL_PATH]
    # The submit envelope's own `retry_after_ms` paces the one poll this call needed.
    assert delays == [1.5]


async def test_202_then_running_then_done_falls_back_to_the_poll_interval() -> None:
    """Only the very first envelope names `retry_after_ms`, by apis' own contract
    for every poll response after that. The second sleep here falls back to
    `poll_interval_s` rather than reusing the first envelope's own pacing.
    """
    manifest_payload = {"content_version": "cv_1"}
    responses = [
        _poll_response(status="running"),
        _poll_response(status="done", manifest=manifest_payload),
    ]

    def handler(request: httpx2.Request) -> httpx2.Response:
        assert request.url.path == _POLL_PATH
        return responses.pop(0)

    delays, sleep = _recording_sleep()
    async with _client(handler) as client:
        result = await resolve_operation_response(
            client=client,
            submit_payload=_submitted(status="pending", retry_after_ms=5),
            deadline=_deadline(poll_budget_s=5.0),
            poll_interval_s=0.25,
            sleep=sleep,
        )

    assert isinstance(result, ResolvedOperation)
    assert result.poll_attempts == 2
    assert delays == [0.005, 0.25]


async def test_202_then_failed_maps_to_a_typed_refusal() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        assert request.url.path == _POLL_PATH
        return httpx2.Response(
            500,
            content=_problem_body(
                code="operation_failed",
                retryable=False,
                detail=f"operation {_OPERATION_ID!r} did not complete successfully",
            ),
        )

    _, sleep = _recording_sleep()
    async with _client(handler) as client:
        result = await resolve_operation_response(
            client=client,
            submit_payload=_submitted(status="pending", retry_after_ms=1),
            deadline=_deadline(poll_budget_s=5.0),
            poll_interval_s=1.0,
            sleep=sleep,
        )

    assert isinstance(result, ToolResult)
    assert result.structured_content is not None
    assert result.structured_content["code"] == "upstream_error"
    assert result.structured_content["upstream_code"] == "operation_failed"
    assert result.structured_content["retry_mode"] == "do_not_retry"


@pytest.mark.parametrize("retryable", [False, True])
async def test_failed_operation_relays_the_cause_code_after_one_poll(retryable: bool) -> None:
    """A failed operation is terminal whatever `retryable` its problem carries: one
    poll, then a refusal under the code of the refusal that failed it. The flag says
    whether a fresh submit could succeed, so it sets the retry mode alone."""
    polls: list[str] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        polls.append(request.url.path)
        return httpx2.Response(
            500,
            content=_problem_body(
                code="operation_failed",
                retryable=retryable,
                cause={"code": "entities_unbound"},
            ),
        )

    _, sleep = _recording_sleep()
    async with _client(handler) as client:
        result = await resolve_operation_response(
            client=client,
            submit_payload=_submitted(status="pending", retry_after_ms=1),
            deadline=_deadline(poll_budget_s=5.0),
            poll_interval_s=1.0,
            sleep=sleep,
        )

    assert polls == [_POLL_PATH]
    assert isinstance(result, ToolResult)
    assert result.structured_content is not None
    assert result.structured_content["code"] == "upstream_error"
    assert result.structured_content["upstream_code"] == "entities_unbound"
    assert result.structured_content["retry_mode"] == (
        "retry_after_delay" if retryable else "do_not_retry"
    )


async def test_already_failed_retryable_operation_does_not_spend_the_call_deadline() -> None:
    """A submit envelope already reporting `failed` fetches its problem against the
    whole call deadline. A `retryable: true` body still ends the wait at that one
    fetch, with no sleep."""
    polls: list[str] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        polls.append(request.url.path)
        return httpx2.Response(
            500,
            content=_problem_body(
                code="operation_failed", retryable=True, cause={"code": "upstream_error"}
            ),
        )

    def unused_sleep(_seconds: float) -> Awaitable[None]:
        raise AssertionError("a failed operation is never waited on")

    async with _client(handler) as client:
        result = await resolve_operation_response(
            client=client,
            submit_payload=_submitted(status="failed"),
            deadline=_deadline(poll_budget_s=5.0),
            poll_interval_s=1.0,
            sleep=unused_sleep,
        )

    assert polls == [_POLL_PATH]
    assert isinstance(result, ToolResult)
    assert result.structured_content is not None
    assert result.structured_content["code"] == "upstream_error"
    assert result.structured_content["upstream_code"] == "upstream_error"


async def _failed_operation_refusal(cause_code: str) -> dict[str, Any]:
    """The refusal a poll answering `operation_failed` with `cause_code` resolves to."""

    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(
            500,
            content=_problem_body(
                code="operation_failed", retryable=False, cause={"code": cause_code}
            ),
        )

    _, sleep = _recording_sleep()
    async with _client(handler) as client:
        result = await resolve_operation_response(
            client=client,
            submit_payload=_submitted(status="pending", retry_after_ms=1),
            deadline=_deadline(poll_budget_s=5.0),
            poll_interval_s=1.0,
            sleep=sleep,
        )

    assert isinstance(result, ToolResult)
    assert result.structured_content is not None
    return result.structured_content


async def test_failed_operation_over_a_ceiling_states_no_ceiling_kind() -> None:
    """An operation failed with no stored refusal reaches the poll as `cause.code`
    alone, without `ceiling_kind`, `count` or `cap`, so the refusal names the code
    and guesses no kind."""
    refusal = await _failed_operation_refusal("over_ceiling")

    assert refusal["code"] == "over_ceiling"
    assert refusal["retry_mode"] == "correct_and_retry"
    assert refusal["upstream_code"] is None
    assert refusal["fix"] == {"detail": "because"}


async def test_failed_operation_over_a_stored_row_ceiling_names_its_kind() -> None:
    """The poll carries the worker's stored `over_ceiling` refusal: its own detail,
    `ceiling_kind`, `count` and `cap` beside `cause.code`."""
    detail = (
        "materialised coverage of 120000 rows exceeds the sync materialisation ceiling of "
        "100000 rows"
    )

    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(
            500,
            content=_problem_body(
                code="operation_failed",
                retryable=False,
                detail=detail,
                cause={"code": "over_ceiling", "entities": [], "truncated": False},
                ceiling_kind="rows",
                count=120000,
                cap=100000,
            ),
        )

    _, sleep = _recording_sleep()
    async with _client(handler) as client:
        result = await resolve_operation_response(
            client=client,
            submit_payload=_submitted(status="pending", retry_after_ms=1),
            deadline=_deadline(poll_budget_s=5.0),
            poll_interval_s=1.0,
            sleep=sleep,
        )

    assert isinstance(result, ToolResult)
    assert result.structured_content is not None
    assert result.structured_content["code"] == "over_ceiling"
    assert result.structured_content["retry_mode"] == "correct_and_retry"
    assert result.structured_content["fix"] == {
        "detail": detail,
        "ceiling_kind": "rows",
        "count": 120000,
        "cap": 100000,
    }


@pytest.mark.parametrize(
    ("cause_code", "code", "retry_mode"),
    [
        ("parquet_size_bound", "over_ceiling", "correct_and_retry"),
        ("facet_not_available", "facet_not_available", "correct_and_retry"),
        # The poll's body says `retryable: false`, as apis' own bodies do.
        ("default_facet_undefined", "default_facet_undefined", "do_not_retry"),
        ("definition_not_reproducible", "request_invalid", "correct_and_retry"),
        ("spec_not_exportable", "spec_not_exportable", "do_not_retry"),
        ("rollup_not_summable", "rollup_not_summable", "do_not_retry"),
    ],
)
async def test_failed_operation_refuses_with_its_causes_code_and_retry_mode(
    cause_code: str, code: str, retry_mode: str
) -> None:
    refusal = await _failed_operation_refusal(cause_code)

    assert refusal["code"] == code
    assert refusal["retry_mode"] == retry_mode
    assert refusal["upstream_code"] is None


async def test_failed_operation_caused_by_a_moved_source_revision_is_retried_in_place() -> None:
    refusal = await _failed_operation_refusal("upstream_version_changed")

    assert refusal["code"] == "upstream_error"
    assert refusal["retry_mode"] == "retry_after_delay"
    assert refusal["upstream_code"] == "upstream_version_changed"
    assert refusal["retry_after_s"] >= 1.0


async def test_failed_operation_that_could_not_sum_a_family_carries_the_users_remedy() -> None:
    """A summed mint deferred to a worker fails on the poll, and refuses with
    the remedy the synchronous 422 carries."""
    refusal = await _failed_operation_refusal("rollup_not_summable")

    assert refusal["fix"] == {
        "axes": [],
        "detail": None,
        "remedy": "Ask about one of the series it combines instead of the total.",
    }


async def test_202_already_done_polls_once_more_for_the_manifest() -> None:
    """The submit response never carries a manifest, even when it already reports
    `done` -- a resubmit racing a worker's own completion. One more bounded fetch is
    what actually resolves it, with no sleep beforehand.
    """
    manifest_payload = {"content_version": "cv_1"}
    calls = {"n": 0}

    def handler(request: httpx2.Request) -> httpx2.Response:
        calls["n"] += 1
        assert request.url.path == _POLL_PATH
        return _poll_response(status="done", manifest=manifest_payload)

    def unused_sleep(_seconds: float) -> Awaitable[None]:
        raise AssertionError("an already-done submit response should not sleep at all")

    async with _client(handler) as client:
        result = await resolve_operation_response(
            client=client,
            submit_payload=_submitted(status="done"),
            deadline=_deadline(poll_budget_s=5.0),
            poll_interval_s=1.0,
            sleep=unused_sleep,
        )

    assert isinstance(result, ResolvedOperation)
    assert result.manifest_payload == manifest_payload
    assert result.poll_attempts == 1
    assert calls["n"] == 1


async def test_202_already_failed_polls_once_more_and_refuses() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        assert request.url.path == _POLL_PATH
        return httpx2.Response(
            500, content=_problem_body(code="operation_failed", retryable=False, detail="failed")
        )

    def unused_sleep(_seconds: float) -> Awaitable[None]:
        raise AssertionError("an already-failed submit response should not sleep at all")

    async with _client(handler) as client:
        result = await resolve_operation_response(
            client=client,
            submit_payload=_submitted(status="failed"),
            deadline=_deadline(poll_budget_s=5.0),
            poll_interval_s=1.0,
            sleep=unused_sleep,
        )

    assert isinstance(result, ToolResult)
    assert result.structured_content is not None
    assert result.structured_content["upstream_code"] == "operation_failed"


async def test_budget_exhaustion_refuses_retryably_without_polling_again() -> None:
    def handler(_request: httpx2.Request) -> httpx2.Response:
        raise AssertionError("no poll should fire once the budget is already spent")

    _, sleep = _recording_sleep()
    async with _client(handler) as client:
        result = await resolve_operation_response(
            client=client,
            submit_payload=_submitted(status="pending", retry_after_ms=1500),
            deadline=_deadline(poll_budget_s=0.0),
            poll_interval_s=2.0,
            sleep=sleep,
        )

    assert isinstance(result, ToolResult)
    assert result.structured_content is not None
    assert result.structured_content["code"] == "operation_still_processing"
    assert result.structured_content["retry_mode"] == "retry_after_delay"
    assert result.structured_content["fix"]["operation_id"] == _OPERATION_ID
    # The wait before a repeat call, independent of the 2s poll interval.
    assert result.structured_content["retry_after_s"] == 5.0


async def test_retry_after_ms_is_clamped_to_the_remaining_budget() -> None:
    """The submit envelope's own `retry_after_ms` (1500ms) would overrun a 10ms
    budget entirely -- the actual sleep is cut short to what the budget allows, and
    the loop refuses on exhaustion right after, never polling again.
    """

    def handler(_request: httpx2.Request) -> httpx2.Response:
        raise AssertionError("the budget should exhaust during the clamped sleep itself")

    delays, sleep = _real_recording_sleep()
    async with _client(handler) as client:
        result = await resolve_operation_response(
            client=client,
            submit_payload=_submitted(status="pending", retry_after_ms=1500),
            deadline=_deadline(poll_budget_s=0.01),
            poll_interval_s=2.0,
            sleep=sleep,
        )

    assert isinstance(result, ToolResult)
    assert result.structured_content is not None
    assert result.structured_content["code"] == "operation_still_processing"
    assert len(delays) == 1
    # Clamped well below the server's own 1.5s -- proof the budget won, not the server.
    assert 0 < delays[0] <= 0.01


async def test_a_malformed_202_body_is_a_refusal_not_a_crash() -> None:
    async with _client(lambda request: httpx2.Response(200)) as client:
        result = await resolve_operation_response(
            client=client,
            submit_payload={"operation": {"operation_id": _OPERATION_ID}},  # missing `status`
            deadline=_deadline(poll_budget_s=5.0),
            poll_interval_s=1.0,
        )

    assert isinstance(result, ToolResult)
    assert result.structured_content is not None
    assert result.structured_content["code"] == "upstream_error"


async def test_202_operation_not_found_on_poll_maps_to_a_typed_refusal() -> None:
    """The id this call itself computed and just submitted is somehow unknown to the
    poll route -- a distinct upstream code from `operation_failed`, taking the same
    `UpstreamError` -> refusal path."""

    def handler(request: httpx2.Request) -> httpx2.Response:
        assert request.url.path == _POLL_PATH
        return httpx2.Response(
            404,
            content=_problem_body(
                code="operation_not_found", retryable=False, detail="no such operation"
            ),
        )

    _, sleep = _recording_sleep()
    async with _client(handler) as client:
        result = await resolve_operation_response(
            client=client,
            submit_payload=_submitted(status="pending", retry_after_ms=1),
            deadline=_deadline(poll_budget_s=5.0),
            poll_interval_s=1.0,
            sleep=sleep,
        )

    assert isinstance(result, ToolResult)
    assert result.structured_content is not None
    assert result.structured_content["code"] == "upstream_error"
    assert result.structured_content["upstream_code"] == "operation_not_found"


async def test_an_absolute_poll_url_is_ignored_in_favour_of_the_local_operation_id_path() -> None:
    """`envelope.poll` is parsed but never dialled: even a same-envelope absolute URL
    (what an SSRF via an upstream-controlled field would look like) is ignored in
    favour of the request path this client builds locally from `operation_id`."""
    manifest_payload = {"content_version": "cv_1"}
    requests: list[str] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        requests.append(str(request.url))
        assert request.url.path == _POLL_PATH
        return _poll_response(status="done", manifest=manifest_payload)

    _, sleep = _recording_sleep()
    async with _client(handler) as client:
        result = await resolve_operation_response(
            client=client,
            submit_payload=_submitted(
                status="pending", retry_after_ms=1, poll="http://attacker.example/steal"
            ),
            deadline=_deadline(poll_budget_s=5.0),
            poll_interval_s=1.0,
            sleep=sleep,
        )

    assert isinstance(result, ResolvedOperation)
    # The attacker-controlled absolute URL never appears; the request always lands on
    # `settings.dfo_apis_url` (this client's own base) plus the locally built path.
    assert requests == [f"http://localhost:8083{_POLL_PATH}"]


async def test_a_malformed_operation_id_is_refused_as_upstream_response_unusable() -> None:
    """`OperationEnvelope.operation_id` is constrained to apis' own `op_` +
    sha256-hex wire format; anything else refuses rather than reaching
    `_POLL_ROUTE_TEMPLATE.format` unchecked."""
    async with _client(lambda request: httpx2.Response(200)) as client:
        result = await resolve_operation_response(
            client=client,
            submit_payload={
                "operation": {"operation_id": "op_not-a-valid-id", "status": "pending"}
            },
            deadline=_deadline(poll_budget_s=5.0),
            poll_interval_s=1.0,
        )

    assert isinstance(result, ToolResult)
    assert result.structured_content is not None
    # `upstream_response_unusable` shares `upstream_error`'s own code (see its own
    # docstring in `tool_errors.py`) -- the same refusal a malformed 202 body itself
    # takes, since a malformed `operation_id` is a schema-drifted response too.
    assert result.structured_content["code"] == "upstream_error"


class _FakeClock:
    """A monotonic clock that moves only when a test moves it."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def _read_timeout(request: httpx2.Request) -> float:
    return request.extensions["timeout"]["read"]


async def test_a_retryable_poll_failure_keeps_the_wait_going() -> None:
    """apis' poll-door backpressure 503 (`retryable: true`) is one more poll that
    saw nothing new: the loop waits `poll_interval_s` and polls again."""
    manifest_payload = {"content_version": "cv_1"}
    responses = [
        httpx2.Response(503, content=_problem_body(code="service_unavailable", retryable=True)),
        _poll_response(status="done", manifest=manifest_payload),
    ]

    def handler(request: httpx2.Request) -> httpx2.Response:
        assert request.url.path == _POLL_PATH
        return responses.pop(0)

    delays, sleep = _recording_sleep()
    async with _client(handler) as client:
        result = await resolve_operation_response(
            client=client,
            submit_payload=_submitted(status="pending", retry_after_ms=5),
            deadline=_deadline(poll_budget_s=5.0),
            poll_interval_s=0.25,
            sleep=sleep,
        )

    assert isinstance(result, ResolvedOperation)
    assert result.manifest_payload == manifest_payload
    assert result.poll_attempts == 2
    assert delays == [0.005, 0.25]


async def test_a_non_retryable_poll_failure_refuses_after_one_poll() -> None:
    requests: list[str] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        requests.append(request.url.path)
        return httpx2.Response(
            404, content=_problem_body(code="operation_not_found", retryable=False)
        )

    _, sleep = _recording_sleep()
    async with _client(handler) as client:
        result = await resolve_operation_response(
            client=client,
            submit_payload=_submitted(status="pending", retry_after_ms=1),
            deadline=_deadline(poll_budget_s=5.0),
            poll_interval_s=0.25,
            sleep=sleep,
        )

    assert isinstance(result, ToolResult)
    assert result.structured_content is not None
    assert result.structured_content["upstream_code"] == "operation_not_found"
    assert requests == [_POLL_PATH]


async def test_the_wait_stops_at_the_poll_reserve_and_no_poll_outruns_it() -> None:
    """With 100s of the call deadline left, polling may use 70s of it. Each poll here
    takes 25s of fake time and is held to the client's 50s read timeout or what is
    left before the reserve, whichever is less; the loop refuses once the reserve is
    reached, without a fourth poll.
    """
    clock = _FakeClock()
    read_timeouts: list[float] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        read_timeouts.append(_read_timeout(request))
        clock.now += 25.0
        return _poll_response(status="running")

    async def sleep(_seconds: float) -> None:
        pass

    async with _client(handler, settings=_settings(upstream_read_timeout_s=50.0)) as client:
        result = await resolve_operation_response(
            client=client,
            submit_payload=_submitted(status="pending", retry_after_ms=1),
            deadline=CallDeadline.after(100.0, clock=clock),
            poll_interval_s=0.0,
            sleep=sleep,
        )

    assert isinstance(result, ToolResult)
    assert result.structured_content is not None
    assert result.structured_content["code"] == "operation_still_processing"
    assert read_timeouts == pytest.approx([50.0, 45.0, 20.0], abs=0.01)
    assert "still being prepared upstream" in result.structured_content["message"]


async def test_a_retryable_failure_fetching_an_already_done_operation_polls_again() -> None:
    """A resubmit of a finished operation answers 202 `done` with no manifest. A
    backpressure 503 on the fetch that follows is waited out like any other poll."""
    manifest_payload = {"content_version": "cv_1"}
    responses = [
        httpx2.Response(503, content=_problem_body(code="service_unavailable", retryable=True)),
        _poll_response(status="done", manifest=manifest_payload),
    ]

    def handler(request: httpx2.Request) -> httpx2.Response:
        assert request.url.path == _POLL_PATH
        return responses.pop(0)

    delays, sleep = _recording_sleep()
    async with _client(handler) as client:
        result = await resolve_operation_response(
            client=client,
            submit_payload=_submitted(status="done", retry_after_ms=1500),
            deadline=_deadline(poll_budget_s=5.0),
            poll_interval_s=0.25,
            sleep=sleep,
        )

    assert isinstance(result, ResolvedOperation)
    assert result.manifest_payload == manifest_payload
    assert result.poll_attempts == 2
    assert delays == [0.25]


async def test_a_poll_the_client_would_not_retry_refuses_even_when_marked_retryable() -> None:
    """`upstream_version_changed` arrives as a 503 with `retryable: true`, yet repeats
    identically; the wait loop reads it as the client's retry loop does."""
    requests: list[str] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        requests.append(request.url.path)
        return httpx2.Response(
            503, content=_problem_body(code="upstream_version_changed", retryable=True)
        )

    _, sleep = _recording_sleep()
    async with _client(handler) as client:
        result = await resolve_operation_response(
            client=client,
            submit_payload=_submitted(status="pending", retry_after_ms=1),
            deadline=_deadline(poll_budget_s=5.0),
            poll_interval_s=0.25,
            sleep=sleep,
        )

    assert isinstance(result, ToolResult)
    assert result.structured_content is not None
    assert result.structured_content["code"] == "upstream_version_changed"
    assert requests == [_POLL_PATH]


async def test_polls_failing_until_the_reserve_report_the_status_as_unconfirmed() -> None:
    """Every poll after the submit envelope fails retryably, so the refusal says the
    status could not be confirmed and names the last error, without claiming the
    operation is still being prepared."""
    clock = _FakeClock()

    def handler(request: httpx2.Request) -> httpx2.Response:
        assert request.url.path == _POLL_PATH
        clock.now += 20.0
        return httpx2.Response(
            503, content=_problem_body(code="service_unavailable", retryable=True)
        )

    async def sleep(_seconds: float) -> None:
        pass

    async with _client(handler) as client:
        result = await resolve_operation_response(
            client=client,
            submit_payload=_submitted(status="pending", retry_after_ms=1),
            deadline=CallDeadline.after(70.0, clock=clock),
            poll_interval_s=0.0,
            sleep=sleep,
        )

    assert isinstance(result, ToolResult)
    assert result.structured_content is not None
    assert result.structured_content["code"] == "operation_still_processing"
    assert result.structured_content["retry_mode"] == "retry_after_delay"
    message = result.structured_content["message"]
    assert "could not confirm the status" in message
    assert "service_unavailable" in message
    assert "still being prepared" not in message
    assert "same spec_id and scope" in message
    assert result.structured_content["fix"]["elapsed_s"] == 40.0


async def test_a_poll_reporting_done_without_a_manifest_is_malformed() -> None:
    def handler(_request: httpx2.Request) -> httpx2.Response:
        return _poll_response(status="done")

    _, sleep = _recording_sleep()
    async with _client(handler) as client:
        result = await resolve_operation_response(
            client=client,
            submit_payload=_submitted(status="done"),
            deadline=_deadline(poll_budget_s=5.0),
            poll_interval_s=0.25,
            sleep=sleep,
        )

    assert isinstance(result, ToolResult)
    assert result.structured_content is not None
    assert result.structured_content["code"] == "upstream_error"
    assert "carries no manifest" in result.structured_content["message"]


async def test_a_poll_naming_a_different_operation_id_is_malformed() -> None:
    """`operation_id` is pinned from the 202 submit response; a poll response naming
    some other id -- op_B where the submit named op_A -- is refused rather than
    trusted, whatever its own status."""
    other_operation_id = "op_" + hashlib.sha256(b"a-different-operation").hexdigest()

    def handler(_request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(
            200,
            json={
                "operation": {"operation_id": other_operation_id, "status": "done"},
                "manifest": {"content_version": "cv_1"},
            },
        )

    _, sleep = _recording_sleep()
    async with _client(handler) as client:
        result = await resolve_operation_response(
            client=client,
            submit_payload=_submitted(status="pending", retry_after_ms=1),
            deadline=_deadline(poll_budget_s=5.0),
            poll_interval_s=1.0,
            sleep=sleep,
        )

    assert isinstance(result, ToolResult)
    assert result.structured_content is not None
    assert result.structured_content["code"] == "upstream_error"
    assert _OPERATION_ID in result.structured_content["message"]
    assert other_operation_id in result.structured_content["message"]


async def test_an_already_done_operation_is_fetched_against_the_whole_deadline() -> None:
    """The mint POST used its whole share, so no poll time is left, but the outcome is
    already decided upstream: the first fetch runs against the call deadline itself."""
    manifest_payload = {"content_version": "cv_1"}

    def handler(request: httpx2.Request) -> httpx2.Response:
        assert request.url.path == _POLL_PATH
        return _poll_response(status="done", manifest=manifest_payload)

    async with _client(handler) as client:
        result = await resolve_operation_response(
            client=client,
            submit_payload=_submitted(status="done"),
            deadline=_deadline(poll_budget_s=0.0),
            poll_interval_s=1.0,
        )

    assert isinstance(result, ResolvedOperation)
    assert result.manifest_payload == manifest_payload
    assert result.poll_attempts == 1


@pytest.mark.parametrize("status", ["done", "failed"])
async def test_a_terminal_operation_whose_result_could_not_be_fetched_states_its_status(
    status: str,
) -> None:
    """A retryable failure on the first fetch, with no poll time left to try again. The
    refusal states the status apis reported and says only that the result was not
    fetched: a `done` or `failed` operation never runs again."""

    def handler(_request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(
            503, content=_problem_body(code="service_unavailable", retryable=True)
        )

    async with _client(handler) as client:
        result = await resolve_operation_response(
            client=client,
            submit_payload=_submitted(status=status),
            deadline=_deadline(poll_budget_s=0.0),
            poll_interval_s=1.0,
        )

    assert isinstance(result, ToolResult)
    assert result.structured_content is not None
    assert result.structured_content["code"] == "operation_still_processing"
    message = result.structured_content["message"]
    assert (
        f"reported operation {_OPERATION_ID!r} {status}, but this call could not fetch" in message
    )
    assert "the last attempt ended with service_unavailable" in message
    assert "may still be running" not in message
    assert "failed failed" not in message
    assert result.structured_content["fix"]["last_status"] == status


async def test_a_poll_cut_by_the_poll_deadline_keeps_the_last_reported_status() -> None:
    """Real time throughout, since the client cuts a poll with `asyncio.timeout` on the
    event loop's clock. Each poll answers `running` after 0.2s; the second starts with
    about 0.05s of the 0.5s poll window left and is cut. That cut is this call's own
    doing, so the refusal still reports the operation as being prepared."""

    async def handler(_request: httpx2.Request) -> httpx2.Response:
        await asyncio.sleep(0.2)
        return _poll_response(status="running")

    delays, sleep = _real_recording_sleep()
    async with _client(handler) as client:
        result = await resolve_operation_response(
            client=client,
            submit_payload=_submitted(status="pending", retry_after_ms=1),
            deadline=_deadline(poll_budget_s=0.5),
            poll_interval_s=0.25,
            sleep=sleep,
        )

    assert isinstance(result, ToolResult)
    assert result.structured_content is not None
    assert result.structured_content["code"] == "operation_still_processing"
    assert "still being prepared upstream" in result.structured_content["message"]
    assert "last_poll_error" not in result.structured_content["fix"]
    assert len(delays) == 2


# `on_poll` reporting


async def test_on_poll_reports_at_acceptance_and_on_status_change_not_every_poll() -> None:
    """One report when the submit response parses, one on `running`'s own first
    appearance, one on `done`'s -- the middle `running` poll, an unchanged status
    well inside the 10s cadence, reports nothing."""
    clock = _FakeClock()
    manifest_payload = {"content_version": "cv_1"}
    responses = [
        _poll_response(status="running"),
        _poll_response(status="running"),
        _poll_response(status="done", manifest=manifest_payload),
    ]

    def handler(request: httpx2.Request) -> httpx2.Response:
        assert request.url.path == _POLL_PATH
        clock.now += 1.0
        return responses.pop(0)

    async def sleep(_seconds: float) -> None:
        pass

    reports: list[PollUpdate] = []

    async def on_poll(update: PollUpdate) -> None:
        reports.append(update)

    async with _client(handler) as client:
        result = await resolve_operation_response(
            client=client,
            submit_payload=_submitted(status="pending", retry_after_ms=1),
            deadline=CallDeadline.after(60.0, clock=clock),
            poll_interval_s=1.0,
            sleep=sleep,
            on_poll=on_poll,
        )

    assert isinstance(result, ResolvedOperation)
    assert [(r.status, r.elapsed_s, r.poll_attempts) for r in reports] == [
        ("pending", 0.0, 0),
        ("running", 1.0, 1),
        ("done", 3.0, 3),
    ]
    assert all(r.operation_id == _OPERATION_ID for r in reports)


async def test_on_poll_repeats_every_10s_of_waiting_when_the_status_does_not_change() -> None:
    """A `running` status that never moves still gets a report roughly every 10s of
    elapsed wait, not on every one of the polls that produced it."""
    clock = _FakeClock()
    manifest_payload = {"content_version": "cv_1"}
    calls = {"n": 0}

    def handler(request: httpx2.Request) -> httpx2.Response:
        calls["n"] += 1
        clock.now += 6.0
        if calls["n"] >= 4:
            return _poll_response(status="done", manifest=manifest_payload)
        return _poll_response(status="running")

    async def sleep(_seconds: float) -> None:
        pass

    reports: list[PollUpdate] = []

    async def on_poll(update: PollUpdate) -> None:
        reports.append(update)

    async with _client(handler) as client:
        result = await resolve_operation_response(
            client=client,
            submit_payload=_submitted(status="running", retry_after_ms=1),
            deadline=CallDeadline.after(60.0, clock=clock),
            poll_interval_s=1.0,
            sleep=sleep,
            on_poll=on_poll,
        )

    assert isinstance(result, ResolvedOperation)
    # Accept at 0s; the 6s and 18s polls stay under the 10s-since-last-report bar and
    # report nothing; the 12s poll clears it; the 24s poll reports on the status change.
    assert [(r.status, r.elapsed_s) for r in reports] == [
        ("running", 0.0),
        ("running", 12.0),
        ("done", 24.0),
    ]


async def test_a_raising_on_poll_is_logged_and_ignored() -> None:
    """`on_poll` fires both at acceptance and after the resolving poll here -- both
    raise, and neither stops the call from returning its manifest."""
    manifest_payload = {"content_version": "cv_1"}

    def handler(request: httpx2.Request) -> httpx2.Response:
        assert request.url.path == _POLL_PATH
        return _poll_response(status="done", manifest=manifest_payload)

    async def on_poll(_update: PollUpdate) -> None:
        raise RuntimeError("boom")

    _, sleep = _recording_sleep()
    async with _client(handler) as client:
        result = await resolve_operation_response(
            client=client,
            submit_payload=_submitted(status="pending", retry_after_ms=1),
            deadline=_deadline(poll_budget_s=5.0),
            poll_interval_s=1.0,
            sleep=sleep,
            on_poll=on_poll,
        )

    assert isinstance(result, ResolvedOperation)
    assert result.manifest_payload == manifest_payload


# `operation_without_waiting`


@pytest.mark.parametrize(
    ("status", "retry_after_ms", "retry_after_s"),
    [("pending", 1500, 1.5), ("running", None, 2.0)],
)
async def test_an_unfinished_operation_is_returned_queued_without_a_poll(
    status: str, retry_after_ms: int | None, retry_after_s: float
) -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        raise AssertionError(f"unexpected call to {request.url.path}")

    async with _client(handler) as client:
        result = await operation_without_waiting(
            client=client,
            submit_payload=_submitted(status=status, retry_after_ms=retry_after_ms),
            deadline=_deadline(poll_budget_s=5.0),
            poll_interval_s=2.0,
        )

    assert result == QueuedOperation(
        operation_id=_OPERATION_ID, status=status, retry_after_s=retry_after_s
    )


async def test_an_operation_already_done_is_fetched_once_without_waiting() -> None:
    manifest_payload = {"content_version": "cv_1"}
    requests: list[str] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        requests.append(request.url.path)
        return _poll_response(status="done", manifest=manifest_payload)

    async with _client(handler) as client:
        result = await operation_without_waiting(
            client=client,
            submit_payload=_submitted(status="done"),
            deadline=_deadline(poll_budget_s=5.0),
            poll_interval_s=2.0,
        )

    assert isinstance(result, ResolvedOperation)
    assert result.manifest_payload == manifest_payload
    assert requests == [_POLL_PATH]


async def test_an_operation_already_failed_refuses_without_waiting() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(500, content=_problem_body(code="operation_failed", retryable=False))

    async with _client(handler) as client:
        result = await operation_without_waiting(
            client=client,
            submit_payload=_submitted(status="failed"),
            deadline=_deadline(poll_budget_s=5.0),
            poll_interval_s=2.0,
        )

    assert isinstance(result, ToolResult)
    assert result.structured_content is not None
    assert result.structured_content["upstream_code"] == "operation_failed"


async def test_a_done_operation_whose_fetch_fails_retryably_stays_queued() -> None:
    requests: list[str] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        requests.append(request.url.path)
        return httpx2.Response(
            503, content=_problem_body(code="service_unavailable", retryable=True)
        )

    async with _client(handler) as client:
        result = await operation_without_waiting(
            client=client,
            submit_payload=_submitted(status="done"),
            deadline=_deadline(poll_budget_s=5.0),
            poll_interval_s=2.0,
        )

    assert result == QueuedOperation(operation_id=_OPERATION_ID, status="done", retry_after_s=2.0)
    assert requests == [_POLL_PATH]
