"""Writes that finish what they start whatever happens to their caller.

`run_to_completion` runs a coroutine to its end in a task of its own, so a
cancel of the caller cannot cut it short. `committing` runs one transaction
whose COMMIT, and the work that must follow every commit that lands (the
`studio.turn` lines of what it committed, `harness.studio_store` and
`harness.turn.store.TurnStore.pause`), run that way, so a caller cancelled
during the commit cannot land the write without that work. Every runtime
connection bounds each statement, the COMMIT included, with asyncpg's
`command_timeout` (`harness.db.engine.COMMAND_TIMEOUT_S`), so a COMMIT run to
its end on a stalled connection still ends. A write that carries a line
confirms a COMMIT that raised by its transaction's status
(`transaction_committed`), so a write that landed with its acknowledgement
lost still runs that work.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable, Coroutine
from contextlib import AbstractAsyncContextManager
from typing import Any

from sqlalchemy.ext.asyncio import AsyncConnection

logger = logging.getLogger(__name__)

# How often `transaction_committed` reads a status again.
READ_BACK_POLL_S = 0.05


async def run_to_completion[T](
    work: Coroutine[Any, Any, T],
    *,
    cancelled: Callable[[asyncio.Task[T]], None] | None = None,
) -> T:
    """Run `work` in a task of its own and wait for it to finish even when
    the caller is cancelled, then re-raise the caller's cancellation.
    `cancelled`, when given, hears each cancel of the caller with the task.

    A write made from a task the stream cancels (`renew`, on the heartbeat)
    or from a request's cleanup (`drop_lease`) is cancelled with
    `Task.cancel`, which no anyio shield stops. Cancelled mid-statement, the
    write never lands.

    The caller's cancellation wins over whatever `work` ended with. A
    heartbeat cancelled while its renewal fails must stop, or the stream it
    renews never finishes unwinding. What `work` raised is then logged."""
    task = asyncio.ensure_future(work)
    caught: asyncio.CancelledError | None = None
    while not task.done():
        try:
            await asyncio.wait((task,))
        except asyncio.CancelledError as exc:
            caught = exc
            if cancelled is not None:
                cancelled(task)
    if caught is not None:
        if not task.cancelled() and (failure := task.exception()) is not None:
            logger.warning(
                "db commit: a write ended with %r after its caller was cancelled", failure
            )
        raise caught
    return task.result()


async def transaction_committed(
    status: Callable[[], Awaitable[str | None]], *, what: str, limit_s: float
) -> bool:
    """Whether the transaction whose COMMIT raised committed, by its status
    as `status` reads it on a fresh connection (`pg_xact_status`:
    `committed`, `aborted`, `in progress`, or `None` once too old to tell).
    A status still `in progress`, a COMMIT the server has not finished, and
    a read that fails, as one in the outage that broke the COMMIT does, are
    read again every `READ_BACK_POLL_S` until `limit_s` has passed. Each read
    is bounded by the time left, and any exception it ends on, the
    cancellation of that bound turned into another error included, counts
    as unknown. `False` for `aborted` or `None`, and when the limit passes
    unsettled, which is logged at ERROR with `what`: such a COMMIT that lands
    later runs none of the work owed to it."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + limit_s
    failure: Exception | None = None
    while (left := deadline - loop.time()) > 0:
        try:
            async with asyncio.timeout(left):
                read = await status()
        except Exception as exc:
            failure = exc
        else:
            if read != "in progress":
                return read == "committed"
        await asyncio.sleep(min(READ_BACK_POLL_S, max(deadline - loop.time(), 0)))
    logger.error(
        "db commit: %s's COMMIT raised and its transaction did not settle within %ss",
        what,
        limit_s,
        exc_info=failure,
    )
    return False


async def committing[T](
    transaction: AbstractAsyncContextManager[AsyncConnection],
    write: Callable[[AsyncConnection], Awaitable[Callable[[], T]]],
    *,
    read_back: Callable[[], Awaitable[bool]] | None = None,
    forward_cut: bool = False,
    raise_confirmed: bool = False,
) -> T:
    """Run `write` in `transaction` and return what the function it returns
    gives once the transaction commits. That function is what follows the
    commit (the caller's bookkeeping and the turn lines of what the write
    committed) and must not await. The commit and it run in one task to the
    end (`run_to_completion`), so a cancel of the caller cannot land the
    commit without them. A cancel or a fault before the commit rolls the
    transaction back. A commit that raises runs nothing after it, unless
    `read_back`, awaited in the same task, confirms that the commit landed
    with its acknowledgement lost: the function then runs as for a commit
    that returned, and with `raise_confirmed` the COMMIT's error is raised
    after it, so the caller judges the write by what the store holds.

    With `forward_cut`, for a caller a cut must not hold up on a stalled
    COMMIT (`harness.turn.store.TurnStore.pause`), the first cancel of the
    caller that arrives while the COMMIT statement is awaited is passed into
    the commit task. The COMMIT then raises, which drops its connection, and
    `read_back` confirms it as for any COMMIT that raised, so the caller's
    cancellation is raised once that read ends, within its limit. A cancel
    that arrives at any other time waits for the task, as without it."""
    conn = await transaction.__aenter__()
    try:
        committed = await write(conn)
    except BaseException as exc:
        await transaction.__aexit__(type(exc), exc, exc.__traceback__)
        raise
    in_commit = forwarded = False

    async def commit() -> T:
        nonlocal in_commit
        in_commit = True
        try:
            await transaction.__aexit__(None, None, None)
        except (Exception, asyncio.CancelledError) as exc:
            in_commit = False
            if isinstance(exc, asyncio.CancelledError) and not forwarded:
                raise
            _take_back(forwarded)
            if read_back is None or not await read_back():
                raise
            result = committed()
            if raise_confirmed:
                raise
            return result
        in_commit = False
        _take_back(forwarded)
        return committed()

    def cut(task: asyncio.Task[T]) -> None:
        nonlocal forwarded
        if forward_cut and in_commit and not forwarded:
            forwarded = True
            task.cancel()

    return await run_to_completion(commit(), cancelled=cut)


def _take_back(forwarded: bool) -> None:
    """Take back the caller's cut `committing` passed into the commit task,
    so the read back and the work after the commit run whole."""
    if forwarded and (task := asyncio.current_task()) is not None:
        task.uncancel()
