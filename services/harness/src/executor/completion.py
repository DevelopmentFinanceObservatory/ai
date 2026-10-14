"""The completion gate: a run completes once each derived result is published.

`publish_results` takes a run the ledger holds as `publishing` and publishes,
through a `ResultPublisher`, each derived (`ops_`) terminal frame still
awaiting publication with its replay recipe. A root (`spec_`) frame needs
none. Each frame gets `PUBLISH_ATTEMPTS` tries, each retry after a jittered
wait no shorter than the one the fault named
(`executor.classify.jittered_retry_delay`). A fault the publisher calls final
fails the run under its code; one still retryable after the last try holds it
`pending` under its code, keeping its results so a carry-on resumes
publication (`executor.ledger.Resume` returns it to `publishing`). The fault's
reason is logged and kept in the ledger. An exception the publisher raises is
a fault too: a transport error (`OSError`) is retryable, any other is final
(`publication_error`). Once nothing awaits publication the run is
`completed`. A fault of the ledger's own store raises
`executor.ledger.LedgerUnavailableError` (`executor.ledger.record_change`),
leaving the run as last recorded.

`run_hand_back` states a run's standing in the shape the orchestrator reads,
codes and numbers alone, and `unrecorded_hand_back` the standing of a run
whose progress the ledger could not record.
"""

from __future__ import annotations

import logging
import math
import random
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Protocol

from common.materialisation import FrameRef

from executor.classify import (
    MAX_RETRY_AFTER_SECONDS,
    ExecutorCode,
    Jitter,
    jittered_retry_delay,
)
from executor.ledger import (
    Complete,
    Fail,
    Hold,
    Publication,
    Published,
    RunLedger,
    RunRecord,
    record_change,
)
from executor.outcomes import RefusalKind

logger = logging.getLogger(__name__)

# Tries at publishing one frame before the run is held `pending`.
PUBLISH_ATTEMPTS = 2


@dataclass(frozen=True, kw_only=True)
class FramePublished:
    """The frame is published with the recipe `replay_key` names."""

    replay_key: str


@dataclass(frozen=True, kw_only=True)
class PublishFault:
    """The frame is not published. `retryable` says a later try may succeed,
    after `retry_after_s` when the fault names a wait."""

    code: str
    reason: str
    retryable: bool
    retry_after_s: float | None = None


class ResultPublisher(Protocol):
    """Publishes one derived frame with the replay recipe built from the store.
    `harness.mcp_client.McpExecutor.publish_result` is one."""

    def __call__(self, frame: FrameRef) -> Awaitable[FramePublished | PublishFault]: ...


async def publish_results(
    run: RunRecord,
    *,
    ledger: RunLedger,
    publisher: ResultPublisher | None,
    sleep: Callable[[float], Awaitable[None]],
    jitter: Jitter = random.uniform,
) -> RunRecord:
    """Publish what `publishing` `run` still awaits, and return the run as
    the ledger then holds it: `completed`, `pending` or `failed`. With no
    `publisher`, a run that awaits publication is returned as it stands."""
    awaiting = [r.frame for r in run.results if r.publication is Publication.REQUIRED]
    if awaiting and publisher is None:
        return run
    for result in awaiting:
        assert publisher is not None
        answer = await _publish(
            FrameRef(spec_id=result.spec_id, content_version=result.content_version),
            publisher=publisher,
            sleep=sleep,
            jitter=jitter,
        )
        if isinstance(answer, FramePublished):
            run = await record_change(ledger, run.run_id, Published(result, answer.replay_key))
            continue
        logger.warning(
            "completion: publishing %s/%s failed (%s): %s",
            result.spec_id,
            result.content_version,
            answer.code,
            answer.reason,
        )
        if not answer.retryable:
            return await record_change(ledger, run.run_id, Fail(answer.code, answer.reason))
        hold = Hold(answer.code, answer.reason, retry_after_s=_named_wait(answer.retry_after_s))
        return await record_change(ledger, run.run_id, hold)
    return await record_change(ledger, run.run_id, Complete())


async def _publish(
    frame: FrameRef,
    *,
    publisher: ResultPublisher,
    sleep: Callable[[float], Awaitable[None]],
    jitter: Jitter,
) -> FramePublished | PublishFault:
    """`publisher`'s answer for `frame`, tried again after a retryable fault
    while tries remain."""
    answer = await _answer(frame, publisher=publisher)
    for attempt in range(PUBLISH_ATTEMPTS - 1):
        if isinstance(answer, FramePublished) or not answer.retryable:
            return answer
        await sleep(
            jittered_retry_delay({"retry_after_s": answer.retry_after_s}, attempt, jitter=jitter)
        )
        answer = await _answer(frame, publisher=publisher)
    return answer


def _named_wait(retry_after_s: float | None) -> float | None:
    """The wait a fault named, capped at `MAX_RETRY_AFTER_SECONDS`, or `None`
    when it named none or one that is negative or not finite."""
    if retry_after_s is None or not math.isfinite(retry_after_s) or retry_after_s < 0:
        return None
    return min(retry_after_s, MAX_RETRY_AFTER_SECONDS)


async def _answer(frame: FrameRef, *, publisher: ResultPublisher) -> FramePublished | PublishFault:
    """`publisher`'s answer for `frame`, an exception it raises made a fault."""
    try:
        return await publisher(frame)
    except OSError as exc:
        return PublishFault(
            code=RefusalKind.TRANSPORT_FAILED.value, reason=str(exc), retryable=True
        )
    except Exception as exc:
        logger.exception("completion: publishing %s raised", frame.spec_id)
        return PublishFault(
            code=ExecutorCode.PUBLICATION_ERROR.value,
            reason=f"{type(exc).__name__}: {exc}",
            retryable=False,
        )


def run_hand_back(run: RunRecord) -> dict[str, Any]:
    """`run`'s standing as the orchestrator reads it: its id and status, and
    for a `pending` run the code of the fault that holds it and when to look
    again, for a `failed` one its code. The ledger's reasons stay out: they
    carry platform text and exception messages."""
    hand_back: dict[str, Any] = {"run_id": run.run_id, "status": run.status.value}
    if run.pending is not None:
        hand_back["code"] = run.pending.code
        if run.pending.retry_after_s is not None:
            hand_back["retry_after_s"] = run.pending.retry_after_s
    if run.failure is not None:
        hand_back["code"] = run.failure.code
    return hand_back


def unrecorded_hand_back(run_id: str | None) -> dict[str, Any]:
    """Run `run_id`'s standing, in `run_hand_back`'s shape, when the ledger's
    store could not record its progress (`executor.ledger.
    LedgerUnavailableError`): `unfinished`, for the conversation's next turn
    to carry on. With no `run_id` the store could not admit the run, so
    nothing was run and nothing carries on: `not_started`."""
    if run_id is None:
        return {"status": "not_started", "code": ExecutorCode.RUN_NOT_STARTED.value}
    return {
        "run_id": run_id,
        "status": "unfinished",
        "code": ExecutorCode.LEDGER_UNAVAILABLE.value,
    }
