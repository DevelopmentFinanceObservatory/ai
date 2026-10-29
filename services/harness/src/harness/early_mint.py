"""Early mints: a card's data asked for while the user reads the card.

Approving a card starts its run, and the run's first wait is the mint behind
each retrieval. `EarlyMints.offer` sends each retrieval's `get_data` when the
`approve` node offers the card (`harness.graph.planning.approve`), so apis
has the frame, or is building it, by the time the run asks again with the
same arguments. Every card passes through that offer: a first draft, an
amendment's or a run's replan, a switch, and a re-offer after a refused
draft.

Each call is sent once, marked background
(`harness.mcp_client.McpExecutor.get_data_in_background`). apis admits a
background mint only while it keeps one work slot and one materialise
permit free for other callers, and declines it otherwise. A background mint
apis admits runs to completion and holds its slot until then, whether or not
the card is approved. data-mcp answers a background call `posted` once its
mint POST has been written out, which is what this module logs, and keeps
the POST open until apis answers it, so apis keeps its CPU for the mint.
apis' verdict, `declined` among them, is on data-mcp's own
`background mint answered` line.

The calls go out at the offer, `max_concurrent` of a card's at a time, the
cap the run's own steps go out under (`executor_max_concurrent_steps`). The
ai service's CPU is throttled once a request ends, so the response that
carried the card waits for them before it closes (`settle`, called from
`harness.studio.StudioRunner`), for `settle_s` at most, after the turn's
lock is released. A call still pending then is left running and logged
`unsent_at_stream_end` at once, and logs nothing when it ends. Each call is
waited for once, by the response that offered it.

The arguments are each retrieval step's own `step.args`, the arguments
`executor.run` sends for a step with no bindings, so the run asks for the
frame the early call built. A run that asks while data-mcp still holds the
early call's mint POST open joins that POST and takes its answer
(`data_mcp.get_data_tool._post_mint`), so apis mints the frame once. It posts
its own request only when apis declined the early POST or that POST never went
out.

One set per conversation, keyed by the card's `plan_revision`: an offer of
the revision already started starts nothing. A new revision replaces the
record, and the replaced set's calls already sent run on. A call still
waiting for one of its card's slots is withdrawn, unsent and logged
`withdrawn`, when a newer revision is offered or the card is approved
(`withdraw`, called from `harness.graph.planning.approve`): the run then
asks for the frames itself. Nothing here touches the card,
the plan or the run ledger. Each call logs one line with its spec_id, the
revision, its outcome and the seconds since the card was offered, and a
failure is logged and dropped.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import OrderedDict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from compiler.caller_clients import MAX_CALLERS
from planner.plan import Plan, plan_revision
from planner.workflow import ToolId

from harness.conversation import current_conversation_id

logger = logging.getLogger(__name__)

# The tool a retrieval step calls.
GET_DATA = ToolId(server="data-mcp", name="get_data")

# Sends one `get_data` call marked background and returns its outcome as a
# word (`McpExecutor.get_data_in_background`).
SendGetData = Callable[[dict[str, Any]], Awaitable[str]]


@dataclass(kw_only=True)
class _Call:
    """One early call: the spec_id it asks for, when the card was offered,
    whether it has taken a slot and gone out, and whether its line has been
    logged."""

    spec_id: Any
    offered_at: float
    sent: bool = False
    logged: bool = False


@dataclass(kw_only=True)
class _MintSet:
    """One card's early calls: its revision, each call's task, the calls a
    `settle` has already waited for, and the cap on how many are in flight."""

    revision: str
    slots: asyncio.Semaphore
    calls: dict[asyncio.Task[None], _Call] = field(default_factory=dict)
    awaited: set[asyncio.Task[None]] = field(default_factory=set)


class EarlyMints:
    """Starts each offered card's retrievals before the user approves it.

    `send` makes one call, and at most `max_concurrent` of one card's calls
    are in flight at once. `settle` waits `settle_s` at most for the calls of
    the conversation's latest card. `clock` times each call.
    """

    def __init__(
        self,
        *,
        send: SendGetData,
        max_concurrent: int,
        settle_s: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._send = send
        self._max_concurrent = max_concurrent
        self._settle_s = settle_s
        self._clock = clock
        # The latest set of each conversation, least recently offered first,
        # bounded as the held MCP clients are.
        self._sets: OrderedDict[str, _MintSet] = OrderedDict()
        # Every call not yet finished, so none is collected mid-flight and
        # `aclose` can cancel them.
        self._tasks: set[asyncio.Task[None]] = set()

    def offer(self, plan: Plan) -> None:
        """Start `plan`'s retrievals, unless its revision is the one this
        conversation last started. Returns at once. The calls run as tasks of
        their own, in the current context, so they name the conversation's
        caller."""
        if plan.workflow is None:
            return
        revision = plan_revision(plan)
        conversation = current_conversation_id.get()
        latest = self._sets.get(conversation)
        if latest is not None:
            if latest.revision == revision:
                return
            self._withdraw(latest)
        mint_set = _MintSet(revision=revision, slots=asyncio.Semaphore(self._max_concurrent))
        self._sets[conversation] = mint_set
        self._sets.move_to_end(conversation)
        while len(self._sets) > MAX_CALLERS:
            self._sets.popitem(last=False)
        for step in plan.workflow.steps:
            if step.tool != GET_DATA or step.bindings:
                continue
            args = dict(step.args)
            call = _Call(spec_id=args.get("spec_id"), offered_at=self._clock())
            task = asyncio.create_task(self._mint(mint_set, call, args))
            mint_set.calls[task] = call
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)

    def withdraw(self) -> None:
        """Withdraw the calls of this conversation's latest card still waiting
        for a slot, once the card is approved."""
        mint_set = self._sets.get(current_conversation_id.get())
        if mint_set is not None:
            self._withdraw(mint_set)

    def _withdraw(self, mint_set: _MintSet) -> None:
        """Cancel each call of `mint_set` that has not gone out, and log it
        `withdrawn`."""
        for task, call in mint_set.calls.items():
            if not (call.sent or task.done()):
                task.cancel()
                self._log(mint_set, call, "withdrawn")

    async def settle(self) -> bool:
        """Wait, `settle_s` at most, for the calls of this conversation's
        latest card no earlier `settle` waited for, and log each one still
        pending then. Those are left running and log nothing more. Each call
        is waited for once, by the stream that offered it, so a later stream,
        such as the approval's, never waits on it. Returns whether it waited
        for any call."""
        mint_set = self._sets.get(current_conversation_id.get())
        if mint_set is None:
            return False
        pending = [
            task for task in mint_set.calls if not task.done() and task not in mint_set.awaited
        ]
        if not pending:
            return False
        mint_set.awaited.update(pending)
        _, unsent = await asyncio.wait(pending, timeout=self._settle_s)
        for task in unsent:
            self._log(mint_set, mint_set.calls[task], "unsent_at_stream_end")
        return True

    async def _mint(self, mint_set: _MintSet, call: _Call, args: dict[str, Any]) -> None:
        """Send one call, once one of the card's slots is free, and log how it
        ended unless `settle` logged it unsent."""
        async with mint_set.slots:
            call.sent = True
            try:
                outcome = await self._send(args)
            except Exception as exc:
                outcome = f"error:{type(exc).__name__}"
        if not call.logged:
            self._log(mint_set, call, outcome)

    def _log(self, mint_set: _MintSet, call: _Call, outcome: str) -> None:
        """Log `call`'s one line."""
        call.logged = True
        logger.info(
            "early mint: spec_id=%s plan_revision=%s outcome=%s elapsed_s=%.2f",
            call.spec_id,
            mint_set.revision,
            outcome,
            self._clock() - call.offered_at,
        )

    async def aclose(self) -> None:
        """Cancel every call still running, at shutdown."""
        tasks = list(self._tasks)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        self._sets.clear()
