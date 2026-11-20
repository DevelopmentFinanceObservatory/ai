"""Writes that finish what they start whatever happens to their caller.

`run_to_completion` runs a coroutine to its end in a task of its own, so a
cancel of the caller cannot cut it short. `committing` runs one transaction
whose COMMIT, and the work that must follow every commit that lands (the
`studio.turn` lines of what it committed, `harness.studio_store` and
`harness.turn.store.TurnStore.pause`), run that way, so a caller cancelled
during the commit cannot land the write without that work. Every runtime
connection bounds each statement, the COMMIT included, with asyncpg's
`command_timeout` (`harness.db.engine.COMMAND_TIMEOUT_S`), so a COMMIT run to
its end on a stalled connection still ends.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable, Coroutine
from contextlib import AbstractAsyncContextManager
from typing import Any

from sqlalchemy.ext.asyncio import AsyncConnection

logger = logging.getLogger(__name__)


async def run_to_completion[T](work: Coroutine[Any, Any, T]) -> T:
    """Run `work` in a task of its own and wait for it to finish even when
    the caller is cancelled, then re-raise the caller's cancellation.

    A write made from a task the stream cancels (`renew`, on the heartbeat)
    or from a request's cleanup (`drop_lease`) is cancelled with
    `Task.cancel`, which no anyio shield stops. Cancelled mid-statement, the
    write never lands.

    The caller's cancellation wins over whatever `work` ended with. A
    heartbeat cancelled while its renewal fails must stop, or the stream it
    renews never finishes unwinding. What `work` raised is then logged."""
    task = asyncio.ensure_future(work)
    cancelled: asyncio.CancelledError | None = None
    while not task.done():
        try:
            await asyncio.wait((task,))
        except asyncio.CancelledError as exc:
            cancelled = exc
    if cancelled is not None:
        if not task.cancelled() and (failure := task.exception()) is not None:
            logger.warning(
                "db commit: a write ended with %r after its caller was cancelled", failure
            )
        raise cancelled
    return task.result()


async def committing[T](
    transaction: AbstractAsyncContextManager[AsyncConnection],
    write: Callable[[AsyncConnection], Awaitable[Callable[[], T]]],
    *,
    read_back: Callable[[], Awaitable[bool]] | None = None,
) -> T:
    """Run `write` in `transaction` and return what the function it returns
    gives once the transaction commits. That function is what follows the
    commit (the caller's bookkeeping and the turn lines of what the write
    committed) and must not await. The commit and it run in one task to the
    end (`run_to_completion`), so a cancel of the caller cannot land the
    commit without them. A cancel or a fault before the commit rolls the
    transaction back. A commit that raises runs nothing after it, unless
    `read_back`, awaited in the same task, finds that the commit landed with
    its acknowledgement lost: the function then runs as for a commit that
    returned."""
    conn = await transaction.__aenter__()
    try:
        committed = await write(conn)
    except BaseException as exc:
        await transaction.__aexit__(type(exc), exc, exc.__traceback__)
        raise

    async def commit() -> T:
        try:
            await transaction.__aexit__(None, None, None)
        except Exception:
            if read_back is None or not await read_back():
                raise
        return committed()

    return await run_to_completion(commit())
