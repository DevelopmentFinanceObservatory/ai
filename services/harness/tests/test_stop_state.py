"""The Stop on the store: the predicate that says whether a Stop is live
(`harness.studio_store.StopState.live`), and every short sequence of the
writes that touch a Stop, run against Postgres.

Each store write is one transaction that locks the conversation row before
it reads any Stop state, so the store's behaviours are the serial orders of
its writes. `test_every_sequence_of_stop_writes_keeps_the_invariants` runs
every sequence of up to `DEPTH` operations from a turn that is streaming,
and every sequence of up to `LAPSED_STOP_DEPTH` of the operations a Stop
on a lapsed lease needs to reach the next turn, and asserts six invariants
at each step that could break one:

- no claim that starts or resumes work commits while a Stop is live;
- a close `cancelled` leaves no `admitted` or `running` run;
- a close `cancelled` closes the turn the Stop names (`stop_turn`);
- a Stop names no turn later than the one the conversation was on when it
  was requested, so a Stop sent after a turn's cursor closed is not read as
  the next turn's. The one exception is a Stop requested on a `/turn`'s
  lease between its claim and the save that begins its turn: that Stop
  names the turn the save opens (`World.cancel`);
- a closing write that cleared the lease leaves no live Stop;
- a closing write that saved, or closed the turn on a Stop, leaves the
  conversation's record and the turn's cursor agreeing on whether the
  turn is closed, and the lease clear.

The search runs breadth first and merges sequences that end in the same
state (`World.state`): the stored row, cursors and runs with their ids
renamed in order of appearance, and what each actor holds. Two sequences
that end in one state behave alike from there, so each state is extended
once.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from typing import Any

import pytest
from harness import studio_state
from harness.studio_state import SegmentBusy, SegmentLost
from harness.studio_store import (
    SEQ_BLOCK,
    ConversationStale,
    SegmentClose,
    StopState,
    StudioStore,
)
from harness.turn.loop import abandon_on
from harness.turn.store import CLOSED, Cursor, CursorStale, TurnStatus, TurnStore, open_turn_on
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine

from .store_clock import StoreClock
from .test_turn_store import ADMITTED_RUN

PRINCIPAL = "user-1"
CONVERSATION = "conv-1"
# The longest sequence the search over every operation runs.
DEPTH = 5
# Six operations carry a Stop on a lapsed lease into the next turn (cancel,
# done, lapse, claim, open, renew), which `claim_turn_lease` must stamp with
# the turn it names. The search at that depth runs those operations only,
# since six of all twelve takes twice as long and finds nothing more.
LAPSED_STOP = ("cancel", "done", "lapse", "claim", "open", "renew")
LAPSED_STOP_DEPTH = 6
_START = 1_000.0


@pytest.mark.parametrize(
    ("cursor_status", "lease_live", "flagged", "stop_turn", "taken_id", "live"),
    [
        pytest.param(None, True, True, None, None, True, id="claimed-flagged"),
        pytest.param(None, False, True, None, None, False, id="unclaimed-flagged"),
        pytest.param(None, True, False, 3, None, True, id="claimed-named"),
        pytest.param("running", False, True, None, None, True, id="running-flagged"),
        pytest.param("running", True, False, 3, None, True, id="running-named"),
        pytest.param("running", True, False, 2, None, False, id="running-earlier-turn-named"),
        pytest.param("running", True, False, None, None, False, id="running-no-stop"),
        pytest.param("paused", True, False, 3, "stop_b", False, id="paused-stop-taken"),
        pytest.param("paused", True, False, 3, "stop_a", True, id="paused-earlier-stop-taken"),
        pytest.param("paused", True, True, 3, "stop_b", True, id="paused-taken-then-flagged"),
        pytest.param("done", True, True, 3, None, False, id="closed-flagged"),
        pytest.param("cancelled", False, False, 3, None, False, id="closed-named"),
    ],
)
def test_a_stop_is_live_while_the_turn_it_names_is_open_and_its_scope_did_not_take_it(
    cursor_status: str | None,
    lease_live: bool,
    flagged: bool,
    stop_turn: int | None,
    taken_id: str | None,
    live: bool,
) -> None:
    state = StopState(
        turn_index=3,
        flagged=flagged,
        stop_id="stop_b" if stop_turn is not None else None,
        stop_turn=stop_turn,
        cursor_status=cursor_status,
        taken_id=taken_id,
        lease_live=lease_live,
    )

    assert state.live is live


class World:
    """One conversation, the stores that stand for its instances, and what
    the instance streaming it holds: `holder`, the store that claimed the
    lease `segment_id`; `acting`, set when that claim acts on a Stop;
    `to_open`, set while the turn it claimed has not begun; `reported`, the
    Stop its last renewal reported live; and `requested_at`, the latest turn
    the last Stop request may name."""

    def __init__(self, engine: AsyncEngine, clock: StoreClock) -> None:
        self.engine = engine
        self.clock = clock
        self.turns = TurnStore(engine)
        self.holder: StudioStore | None = None
        self.segment_id: str | None = None
        self.acting = False
        self.to_open = False
        self.reported: str | None = None
        self.requested_at: int | None = None

    def instance(self) -> StudioStore:
        return StudioStore(self.engine, **self.clock.store_kwargs)

    async def reset(self) -> None:
        """An empty conversation table and the clock at its start, then a
        first turn claimed and opened by one instance."""
        async with self.engine.begin() as conn:
            await conn.execute(text("DELETE FROM conversations"))
        await self.clock.set(_START)
        self.requested_at = None
        store = self.instance()
        await store.create(PRINCIPAL, CONVERSATION)
        await self._claim(store, begin=True)
        await self.open()

    async def _claim(self, store: StudioStore, *, begin: bool, stopping: bool = False) -> bool:
        """Claim the conversation on `store` as the routes do: `/turn`'s
        lease alone when `begin`, which `open` then begins the turn under, and
        otherwise `/interrupt`'s claim save. Returns whether the claim
        committed."""
        if begin:
            try:
                lease = await store.claim_turn_lease(PRINCIPAL, CONVERSATION)
            except (SegmentBusy, ConversationStale):
                return False
            segment_id = lease.segment_id
        else:
            work = store.draft(PRINCIPAL, CONVERSATION)
            try:
                segment_id = store.claim_segment(PRINCIPAL, CONVERSATION, work)
                await store.save(PRINCIPAL, CONVERSATION, work, stopping=stopping)
            except (SegmentBusy, ConversationStale):
                store.drop_draft(PRINCIPAL, CONVERSATION)
                return False
        self.holder, self.segment_id = store, segment_id
        self.acting, self.to_open, self.reported = stopping, begin, None
        return True

    async def _open_cursor(self) -> Cursor | None:
        return await self.turns.open_cursor(PRINCIPAL, CONVERSATION)

    async def _turn(self, turn_index: int) -> Cursor | None:
        cursor = await self._open_cursor()
        if cursor is not None and cursor.turn_index == turn_index:
            return cursor
        async with self.engine.connect() as conn:
            turn_id = (
                await conn.execute(
                    text("SELECT id FROM turn_cursors WHERE turn_index = :turn_index"),
                    {"turn_index": turn_index},
                )
            ).scalar_one_or_none()
        return None if turn_id is None else await self.turns.turn(turn_id)

    async def _row(self) -> Any:
        async with self.engine.connect() as conn:
            return (
                await conn.execute(
                    text(
                        "SELECT id, (turn_state ->> 'turn_index')::int AS turn_index, "
                        "COALESCE((turn_state ->> 'turn_recorded')::boolean, false) "
                        "AS turn_recorded, lease_segment_id, lease_expires_at, lease_cancel, "
                        "stop_id, stop_turn FROM conversations"
                    )
                )
            ).one()

    async def _computing_runs(self) -> int:
        async with self.engine.connect() as conn:
            return (
                await conn.execute(
                    text(
                        "SELECT count(*) FROM runs "
                        "WHERE record ->> 'status' IN ('admitted', 'running')"
                    )
                )
            ).scalar_one()

    async def stop_live(self) -> bool:
        return await self.instance().stop_live(PRINCIPAL, CONVERSATION)

    def _streaming(self) -> bool:
        return self.holder is not None and not self.acting and not self.to_open

    # --- the operations -------------------------------------------------

    async def cancel(self) -> None:
        """A Stop request. One that lands on a `/turn`'s lease before the
        save that begins its turn is stamped by that turn's writes, so it
        may name the turn the save opens, one past the row's."""
        if await self.instance().request_cancel(PRINCIPAL, CONVERSATION):
            row = await self._row()
            beginning = self.to_open and row.lease_segment_id == self.segment_id
            self.requested_at = row.turn_index + (1 if beginning else 0)

    async def check_stop_turn(self) -> None:
        """A Stop names a turn no later than the one its request may name
        (`cancel`), whichever write stamped it."""
        stop_turn = (await self._row()).stop_turn
        assert stop_turn is None or (
            self.requested_at is not None and stop_turn <= self.requested_at
        ), "a Stop sent before a turn began names that turn"

    async def renew(self) -> None:
        """The holder's heartbeat, which `/turn` starts with the stream, once
        the save that begins its turn has committed."""
        if self.holder is None or self.segment_id is None or self.to_open:
            return
        renewal = await self.holder.renew(PRINCIPAL, CONVERSATION, self.segment_id)
        if not renewal.held:
            self.holder = None
            return
        self.reported = renewal.stop_id

    async def _pause(self, taken: str | None) -> None:
        cursor = await self._open_cursor()
        if not self._streaming() or cursor is None or cursor.status is not TurnStatus.RUNNING:
            return
        assert self.segment_id is not None
        try:
            await self.turns.pause(
                cursor,
                segment_id=self.segment_id,
                node="ask",
                node_input=None,
                kind="ask",
                interrupt_id=f"int_{cursor.round + 1}",
                payload={},
                state={},
                stop_taken_id=taken,
                lineage=None,
            )
        except SegmentLost:
            self.holder = None

    async def pause(self) -> None:
        await self._pause(None)

    async def pause_taken(self) -> None:
        if self.reported is not None:
            await self._pause(self.reported)

    async def done(self) -> None:
        cursor = await self._open_cursor()
        if not self._streaming() or cursor is None or cursor.status is not TurnStatus.RUNNING:
            return
        assert self.segment_id is not None
        try:
            await self.turns.finish(cursor, segment_id=self.segment_id, status=TurnStatus.DONE)
        except SegmentLost:
            self.holder = None

    async def cancelled(self) -> None:
        """The holder closes the turn `cancelled` on the Stop its renewal
        reported, or that its claim acts on."""
        cursor = await self._open_cursor()
        if self.holder is None or self.to_open or not (self.acting or self.reported):
            return
        if cursor is None:
            return
        assert self.segment_id is not None
        stop_turn = (await self._row()).stop_turn
        try:
            await self.turns.finish(cursor, segment_id=self.segment_id, status=TurnStatus.CANCELLED)
        except SegmentLost:
            self.holder = None
            return
        assert cursor.turn_index == stop_turn, "a turn the Stop does not name closed cancelled"
        assert await self._computing_runs() == 0, "a cancelled turn left a run computing"

    async def close(self) -> None:
        """The holder's closing write, as the stream's owner makes it: the
        record of a turn its cursor closed is closed on the live state
        first, as the stream's last frames close it, then `close_segment`
        saves it and clears the lease, closing the turn on a live Stop."""
        if self.holder is None or self.to_open or self.segment_id is None:
            return
        row = await self._row()
        state = self.holder.current(PRINCIPAL, CONVERSATION)
        assert state is not None
        cursor = await self._turn(row.turn_index)
        if cursor is not None and cursor.status in CLOSED and not state.turn_recorded:
            state.close_turn(
                final_text="",
                error=None if cursor.status is TurnStatus.DONE else cursor.status.value,
            )
        closed = await self.holder.close_segment(PRINCIPAL, CONVERSATION, self.segment_id)
        self.holder = None
        if closed is SegmentClose.LOST:
            return
        assert closed is not SegmentClose.FAILED, "a closing write failed with no fault"
        row = await self._row()
        assert row.lease_segment_id is None, "a closing write left the lease"
        assert not await self.stop_live(), "a closing write cleared the lease under a live Stop"
        cursor = await self._turn(row.turn_index)
        if cursor is not None:
            assert row.turn_recorded is (cursor.status in CLOSED), (
                "a closing write left the record and the cursor disagreeing"
            )
        if closed is SegmentClose.STOPPED:
            assert cursor is not None and cursor.status is TurnStatus.CANCELLED
            assert await self._computing_runs() == 0, "a cancelled turn left a run computing"

    async def _working_claim(self, *, begin: bool) -> None:
        store = self.instance()
        await store.load(PRINCIPAL, CONVERSATION)
        cursor = await self._open_cursor()
        if not begin and (cursor is None or cursor.status is not TurnStatus.PAUSED):
            return
        live = await self.stop_live()
        if await self._claim(store, begin=begin):
            assert not live, "a claim that starts work committed under a live Stop"
            if not begin:
                assert cursor is not None and cursor.pause is not None
                assert self.segment_id is not None
                await self.turns.record_answer(
                    cursor,
                    segment_id=self.segment_id,
                    interrupt_id=cursor.pause.interrupt_id,
                    round=cursor.pause.round,
                    answer={},
                )

    async def claim(self) -> None:
        """A `/turn` claim, which begins the next turn."""
        await self._working_claim(begin=True)

    async def resume(self) -> None:
        """An `/interrupt` claim, which answers the open pause."""
        await self._working_claim(begin=False)

    async def act(self) -> None:
        """A claim that acts on a live Stop, as a route does before it starts work."""
        if not await self.stop_live():
            return
        store = self.instance()
        await store.load(PRINCIPAL, CONVERSATION)
        await self._claim(store, begin=False, stopping=True)

    async def open(self) -> None:
        """The save that begins the turn the holder claimed, as `/turn`
        makes it: in one transaction it abandons the turn left open, opens
        the claimed one and moves the row to it. The opened turn then admits
        a run."""
        if self.holder is None or not self.to_open or self.segment_id is None:
            return
        store, segment_id = self.holder, self.segment_id
        await store.load(PRINCIPAL, CONVERSATION)
        work = store.draft(PRINCIPAL, CONVERSATION)
        left = await self._open_cursor()
        work.begin_turn()

        async def begin(conn: AsyncConnection, pk: int) -> None:
            if left is not None:
                await abandon_on(conn, left, segment_id=segment_id)
            await open_turn_on(
                conn,
                pk,
                segment_id=segment_id,
                turn_index=work.turn_index,
                node="seed",
                inputs={},
                state={},
            )

        try:
            await store.save(PRINCIPAL, CONVERSATION, work, alongside=begin, seq_reserved=SEQ_BLOCK)
        except (SegmentLost, ConversationStale):
            store.drop_draft(PRINCIPAL, CONVERSATION)
            self.holder = None
            return
        except CursorStale:
            store.drop_draft(PRINCIPAL, CONVERSATION)
            raise
        self.to_open = False
        row = await self._row()
        async with self.engine.begin() as conn:
            await conn.execute(
                text(
                    "INSERT INTO runs (conversation, run_id, record) "
                    "VALUES (:pk, :run_id, CAST(:record AS jsonb))"
                ),
                {
                    "pk": row.id,
                    "run_id": f"run_{row.turn_index}",
                    "record": json.dumps(ADMITTED_RUN),
                },
            )

    async def lapse(self) -> None:
        await self.clock.advance(studio_state.SEGMENT_TTL_S + 1)

    # --- the state ------------------------------------------------------

    async def state(self) -> tuple[Any, ...]:
        """Everything the operations read, with Stop ids renamed in order of
        appearance and turn indices made relative to the row's."""
        row = await self._row()
        async with self.engine.connect() as conn:
            cursors = (
                await conn.execute(
                    text(
                        "SELECT turn_index, status, stop_taken_id FROM turn_cursors "
                        "ORDER BY turn_index"
                    )
                )
            ).all()
            runs = (
                await conn.execute(
                    text("SELECT run_id, record ->> 'status' FROM runs ORDER BY run_id")
                )
            ).all()
        names: dict[str, int] = {}

        def name(stop_id: str | None) -> int | None:
            return None if stop_id is None else names.setdefault(stop_id, len(names))

        def turn(index: int | None) -> int | None:
            return None if index is None else index - row.turn_index

        lease = (
            None
            if row.lease_segment_id is None
            else (row.lease_segment_id == self.segment_id, row.lease_expires_at > self.clock.at)
        )
        return (
            lease,
            row.lease_cancel,
            name(row.stop_id),
            turn(row.stop_turn),
            tuple((turn(c.turn_index), c.status, name(c.stop_taken_id)) for c in cursors),
            tuple(status for _, status in runs),
            self.holder is not None,
            self.acting,
            self.to_open,
            name(self.reported),
        )


_OPERATIONS: dict[str, Callable[[World], Awaitable[None]]] = {
    name: getattr(World, name)
    for name in (
        "cancel",
        "renew",
        "pause",
        "pause_taken",
        "done",
        "cancelled",
        "close",
        "claim",
        "resume",
        "act",
        "open",
        "lapse",
    )
}


async def _run(world: World, sequence: tuple[str, ...]) -> tuple[Any, ...]:
    await world.reset()
    for step, name in enumerate(sequence):
        try:
            await _OPERATIONS[name](world)
        except CursorStale:
            # The holder's view of the cursor moved under another writer.
            pass
        except AssertionError as exc:
            raise AssertionError(f"after {sequence[: step + 1]}: {exc}") from exc
        try:
            await world.check_stop_turn()
        except AssertionError as exc:
            raise AssertionError(f"after {sequence[: step + 1]}: {exc}") from exc
    return await world.state()


@pytest.mark.parametrize(
    ("operations", "depth"),
    [(tuple(_OPERATIONS), DEPTH), (LAPSED_STOP, LAPSED_STOP_DEPTH)],
    ids=["every-operation", "lapsed-stop"],
)
async def test_every_sequence_of_stop_writes_keeps_the_invariants(
    pg_engine: AsyncEngine, operations: tuple[str, ...], depth: int
) -> None:
    """Breadth first over every sequence of up to `depth` of `operations`,
    each state extended once (module docstring).
    Slow: each sequence replays from an empty table, thousands of short transactions."""
    world = World(pg_engine, await StoreClock.start(pg_engine, at=_START))
    frontier = {await _run(world, ()): ()}
    seen = set(frontier)
    # The operations that changed a state, so none is a no-op throughout.
    moved: set[str] = set()
    for _ in range(depth):
        reached: dict[tuple[Any, ...], tuple[str, ...]] = {}
        for before, sequence in frontier.items():
            for name in operations:
                extended = (*sequence, name)
                state = await _run(world, extended)
                if state != before:
                    moved.add(name)
                if state not in seen:
                    seen.add(state)
                    reached[state] = extended
        frontier = reached

    assert moved == set(operations)
