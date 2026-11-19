"""Best-effort reads and writes of the lineage rows a card or a turn stores.

A lineage row helps a later draw state its sources, and nothing waits on it
past a small bound or fails for want of it. `bounded_read` runs one read for
at most a budget, and `insert_in_savepoint` writes rows without risking the
write around them. The card lineage (`harness.studio_store`,
`harness.mcp_client.McpExecutor`) and the turn frame lineage
(`harness.turn.store.TurnStore`) use both.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Coroutine, Mapping, Sequence
from typing import Any

from compiler.surface import transient
from sqlalchemy import Executable
from sqlalchemy.ext.asyncio import AsyncConnection

_logger = logging.getLogger(__name__)


async def bounded_read[T](
    read: Coroutine[Any, Any, T],
    *,
    budget_s: float,
    held: set[asyncio.Task[Any]],
    logger: logging.Logger,
    what: str,
) -> T | None:
    """`read`'s result, or `None` when it fails or runs past `budget_s`,
    logged as a warning that starts with `what`. The read runs as a task of
    its own, so a connection that stalls while the read is cancelled holds up
    only that task. A read past its budget, or whose caller is cancelled, is
    cancelled and kept in `held` until it finishes its cleanup, and logged if
    that fails. A read whose own result can be `None` reads the same as one
    that failed."""
    task = asyncio.create_task(read)
    try:
        done, _ = await asyncio.wait({task}, timeout=budget_s)
    except asyncio.CancelledError:
        _abandon(task, held)
        raise
    if not done:
        _abandon(task, held)
        logger.warning("%s ran past %gs", what, budget_s)
        return None
    try:
        return task.result()
    except Exception:
        logger.warning("%s failed", what, exc_info=True)
        return None


def _abandon(task: asyncio.Task[Any], held: set[asyncio.Task[Any]]) -> None:
    """Cancel `task` and hold it in `held` until it finishes, without waiting."""
    task.cancel()
    held.add(task)

    def _done(finished: asyncio.Task[Any]) -> None:
        held.discard(finished)
        if not finished.cancelled() and (exc := finished.exception()) is not None:
            _logger.warning("an abandoned lineage read failed while it finished", exc_info=exc)

    task.add_done_callback(_done)


async def insert_in_savepoint(
    conn: AsyncConnection,
    statement: Executable,
    params: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    *,
    logger: logging.Logger,
    failure: str,
) -> bool:
    """Run `statement` with `params` on `conn` in a savepoint, so a failed
    insert leaves the caller's transaction to commit. A transient fault
    (`compiler.surface.transient`) raises, for the whole write to be retried;
    any other is logged at error on `logger` as `failure`, and this
    returns `False`. `True` means the rows were written."""
    try:
        async with conn.begin_nested():
            await conn.execute(statement, params if isinstance(params, Mapping) else list(params))
    except Exception as exc:
        if transient(exc):
            raise
        logger.exception(failure)
        return False
    return True
