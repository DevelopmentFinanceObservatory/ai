"""A wall-clock deadline one caller hands down through every upstream step of a call,
and the budget that sizes a `get_data` call's deadline for the entry point it came through."""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, fields, replace
from typing import Self

Clock = Callable[[], float]

# apis' default `mint_timeout_seconds` (`datasets_api/config.py`): the longest it runs
# a synchronous mint before answering 504.
UPSTREAM_SYNC_MINT_TIMEOUT_S = 120.0

# The wait a caller is told before repeating a `get_data` call its deadline refused.
DEADLINE_RETRY_AFTER_S = 5.0

# The longest the combined server (`harness.combined_mcp`) lets a call from an outside
# client run. At this limit it cancels the call and refuses it.
EXTERNAL_CALL_LIMIT_S = 45.0

# How far before `EXTERNAL_CALL_LIMIT_S` an outside `get_data` call's own deadline ends.
# At its deadline `get_data` writes its own refusal (`operation_still_processing`,
# `mint_unanswered`), which takes well under a second; the margin lets that refusal
# reach the caller before the combined server cancels the call.
EXTERNAL_CALL_MARGIN_S = 3.0


@dataclass(frozen=True, kw_only=True)
class CallDeadline:
    """The instant, on `clock`, by which a whole call must finish.

    Passed as an explicit argument, so each step reads the time left and
    bounds its own work with it. `clock` must be monotonic; tests inject a
    fake one to move time without sleeping.
    """

    expires_at: float
    clock: Clock = time.monotonic

    @classmethod
    def after(cls, seconds: float, *, clock: Clock = time.monotonic) -> Self:
        """A deadline `seconds` from now on `clock`."""
        return cls(expires_at=clock() + seconds, clock=clock)

    def remaining(self) -> float:
        """Seconds left before the deadline, negative once it has passed."""
        return self.expires_at - self.clock()

    def reserving(self, seconds: float) -> Self:
        """This deadline, brought forward by `seconds`.

        What a step passes to the work it bounds when it must itself finish with
        `seconds` still left on the original deadline, for whatever runs after it.
        """
        return replace(self, expires_at=self.expires_at - seconds)


@dataclass(frozen=True, kw_only=True)
class CallBudget:
    """How one `get_data` call shares out its deadline, in seconds.

    The call starts a `CallDeadline` of `deadline_s`, or of
    `Settings.get_data_deadline_s` when `deadline_s` is None. Reading the stored
    definition and resolving entity names must finish with `pre_mint_reserve_s` still
    left. A mint POST gets at most `mint_timeout_s`; with less than
    `min_mint_timeout_s` to offer it, the call refuses before sending anything. The
    mint POST, the operation poll and a re-mint's definition write must each finish
    with `post_mint_reserve_s` still left, the share kept for the frame-store read and
    write, the frame fetch, the statistics fetch and rendering.

    Raises:
        ValueError: a field is not finite and positive, or the reserves leave a
            phase no time: pre-mint work must leave room for the shortest mint
            before the post-mint reserve, and needs some of a set deadline itself.
            `Settings.get_data_deadline_s` enforces the same for the deadline it
            supplies.
    """

    deadline_s: float | None
    pre_mint_reserve_s: float
    mint_timeout_s: float
    min_mint_timeout_s: float
    post_mint_reserve_s: float

    def __post_init__(self) -> None:
        for field in fields(self):
            value = getattr(self, field.name)
            if value is None:
                continue
            if not (math.isfinite(value) and value > 0):
                raise ValueError(f"{field.name} must be finite and positive, got {value}")
        if self.pre_mint_reserve_s < self.post_mint_reserve_s + self.min_mint_timeout_s:
            raise ValueError(
                "pre_mint_reserve_s must cover min_mint_timeout_s plus post_mint_reserve_s"
            )
        if self.mint_timeout_s < self.min_mint_timeout_s:
            raise ValueError("mint_timeout_s must be at least min_mint_timeout_s")
        if self.deadline_s is not None and self.deadline_s <= self.pre_mint_reserve_s:
            raise ValueError("deadline_s must exceed pre_mint_reserve_s")

    @property
    def waits_out_sync_mint(self) -> bool:
        """Whether a mint POST given its full `mint_timeout_s` outlasts apis'
        synchronous mint timeout, so a synchronous mint always answers inside it."""
        return self.mint_timeout_s > UPSTREAM_SYNC_MINT_TIMEOUT_S


# A call through the harness's own mount. Its deadline is `Settings.get_data_deadline_s`,
# 165s by default, 15s under the harness's 180s `get_data` call timeout
# (`compiler/capabilities.py`), and the 125s mint cap outlasts apis' 120s
# synchronous mint timeout.
DIRECT_CALL_BUDGET = CallBudget(
    deadline_s=None,
    pre_mint_reserve_s=50.0,
    mint_timeout_s=125.0,
    min_mint_timeout_s=20.0,
    post_mint_reserve_s=30.0,
)

# A call through the combined server, which cancels an outside call at
# `EXTERNAL_CALL_LIMIT_S`, 45s. The deadline ends `EXTERNAL_CALL_MARGIN_S` before that,
# at 42s. Worked back from the end of the call:
#
# - The last 10s (`post_mint_reserve_s`) hold the frame-store read and write, the
#   frame fetch and rendering. The statistics fetch runs only with 8s left and ends
#   with 3s unspent (`get_data_tool._STATISTICS_FLOOR_S` and
#   `_STATISTICS_RESERVE_S`), so it fits inside them.
# - The mint POST and the operation poll therefore stop at 32s, and a poll still
#   waiting then refuses `operation_still_processing` with 10s in hand.
# - A mint needs at least 15s (`min_mint_timeout_s`) to be worth sending, so
#   pre-mint work stops at 17s: `pre_mint_reserve_s` is 15 + 10 = 25.
# - The mint cap is the 32s before the post-mint reserve, so only the deadline
#   bounds the POST.
#
# A repeat call made after the refusal's 5s wait rejoins the operation and polls
# for about 30s more, so an operation that takes 125s finishes on the fourth call.
# A synchronous mint gets at most 32s, under apis' 120s synchronous mint
# timeout: one that apis admits to its synchronous path and that runs longer
# refuses `mint_unanswered`, and keeps running upstream. `data_mcp.mint_ledger`
# holds its apis slot until the 120s plus its own margin have passed and
# refuses a repeat until then; a repeat after that runs the mint again from the
# start.
_EXTERNAL_DEADLINE_S = EXTERNAL_CALL_LIMIT_S - EXTERNAL_CALL_MARGIN_S
_EXTERNAL_POST_MINT_RESERVE_S = 10.0
EXTERNAL_CALL_BUDGET = CallBudget(
    deadline_s=_EXTERNAL_DEADLINE_S,
    pre_mint_reserve_s=25.0,
    mint_timeout_s=_EXTERNAL_DEADLINE_S - _EXTERNAL_POST_MINT_RESERVE_S,
    min_mint_timeout_s=15.0,
    post_mint_reserve_s=_EXTERNAL_POST_MINT_RESERVE_S,
)

_call_budget: ContextVar[CallBudget] = ContextVar(
    "get_data_call_budget", default=DIRECT_CALL_BUDGET
)


def current_call_budget() -> CallBudget:
    """The budget for a `get_data` call made now: the one `use_call_budget` set,
    else `DIRECT_CALL_BUDGET`."""
    return _call_budget.get()


@contextmanager
def use_call_budget(budget: CallBudget) -> Iterator[None]:
    """Run `get_data` calls made inside the block under `budget`.

    The entry point a call came through sets it: the combined server wraps each
    tool call in `use_call_budget(EXTERNAL_CALL_BUDGET)`. A test that drives a
    server through an in-memory client opens the client inside the block, since
    the server's tasks copy the context when the client connects.
    """
    token = _call_budget.set(budget)
    try:
        yield
    finally:
        _call_budget.reset(token)
