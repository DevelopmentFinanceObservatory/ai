"""When a completed run's result counts as delivered.

A completed run is delivered once a hand-back listing its frames is in the
conversation's session and every frame of it was read: described, or refused
`do_not_retry`, which a later turn would meet again (`frame_read`). `deliver`
decides it for every route that hands a completed run back: `plan_done` for a
run that completes in its own turn, and `harness.graph.resume` for one carried
on. A run with a frame refused for a retry, or not described at all, is
recorded `executor.ledger.Undelivered` before its hand-back is written, so its
standing says so (`RESULT_UNDELIVERED`), and the next turn carries it on and
lists it again until the delivery bound (`executor.ledger.MAX_DELIVERY_ATTEMPTS`,
`MAX_PENDING_AGE`) ends it.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Any

from executor.ledger import (
    Delivered,
    LedgerUnavailableError,
    RunLedger,
    Undelivered,
    record_change,
)

logger = logging.getLogger(__name__)

# `run_hand_back`'s `code` for a completed run whose frames could not all be
# read this turn.
RESULT_UNDELIVERED = "result_undelivered"


@dataclass(frozen=True)
class Delivery:
    """A completed run's delivery: `standing` holds the fields its standing
    gains, and `acknowledge` records it `Delivered`, to be called once the
    session holds the hand-back."""

    standing: dict[str, str]
    acknowledge: Callable[[], Awaitable[None]]


def frame_read(view: Mapping[str, Any]) -> bool:
    """Whether the `describe_frame` answer `view` reads its frame for delivery:
    described, or refused `do_not_retry`."""
    if view.get("outcome") == "ok":
        return True
    return (view.get("refusal") or {}).get("retry_mode") == "do_not_retry"


async def deliver(ledger: RunLedger, run_id: str, *, read: bool) -> Delivery:
    """The delivery of completed run `run_id` by a hand-back whose frames were
    all read (`read`, each by `frame_read`) or not.

    A read run is recorded `Delivered` by `acknowledge`, and a run whose
    delivery cannot be recorded is listed again by the next turn. An unread run
    is recorded `Undelivered` now and its standing carries `RESULT_UNDELIVERED`,
    with the reason no later turn lists it once the delivery bound has ended
    its carrying on. A store fault recording that leaves the run as it stands,
    for the next turn to list again.
    """
    if read:

        async def acknowledge() -> None:
            try:
                await ledger.record(run_id, Delivered())
            except Exception:
                logger.warning(
                    "delivery: run %s's delivery could not be recorded; the next turn lists it "
                    "again",
                    run_id,
                    exc_info=True,
                )

        return Delivery(standing={}, acknowledge=acknowledge)
    try:
        run = await record_change(ledger, run_id, Undelivered())
    except LedgerUnavailableError:
        logger.warning(
            "delivery: run %s's undelivered result could not be recorded; the next turn lists "
            "it again",
            run_id,
            exc_info=True,
        )
        return Delivery(standing={"code": RESULT_UNDELIVERED}, acknowledge=_nothing)
    if not run.delivered:
        return Delivery(standing={"code": RESULT_UNDELIVERED}, acknowledge=_nothing)
    standing = {
        "code": RESULT_UNDELIVERED,
        "reason": "its frames could not be read, so it is not listed again",
    }
    return Delivery(standing=standing, acknowledge=_nothing)


async def _nothing() -> None:
    return None
