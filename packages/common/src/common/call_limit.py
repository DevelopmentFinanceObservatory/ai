"""How long a call from an outside client may run, and the deadline a call runs under.

The combined server (`harness.combined_mcp`) cancels a tool call from an outside client
at `EXTERNAL_CALL_LIMIT_S` and refuses it. Its outermost middleware starts the call's
deadline `EXTERNAL_CALL_MARGIN_S` before that limit (`data_mcp.deadline.use_call_budget`),
so a tool that bounds its work by `current_call_deadline` writes its own refusal before the
call is cancelled: `export_data` (`data_mcp.export_data_tool`) and a public analysis call
(`analysis_mcp.tool_pipeline`). `get_data` starts a deadline of its own, sized from the
same budget (`data_mcp.deadline.current_call_budget`), when its tool body starts.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace
from typing import Self

EXTERNAL_CALL_LIMIT_S = 45.0

# Writing a refusal takes well under a second, so this leaves room for it to reach
# the caller before the combined server cancels the call.
EXTERNAL_CALL_MARGIN_S = 3.0

Clock = Callable[[], float]


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


_call_deadline: ContextVar[CallDeadline | None] = ContextVar("call_deadline", default=None)


def current_call_deadline() -> CallDeadline | None:
    """The deadline of the call running now, as `use_call_deadline` set it, else `None`."""
    return _call_deadline.get()


@contextmanager
def use_call_deadline(deadline: CallDeadline | None) -> Iterator[None]:
    """Run the block under `deadline` (`current_call_deadline`)."""
    token = _call_deadline.set(deadline)
    try:
        yield
    finally:
        _call_deadline.reset(token)
