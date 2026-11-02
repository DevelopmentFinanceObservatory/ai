"""The own loop's store: one cursor row per turn, per-agent records and accepted
answers, in Postgres, every write a compare-and-swap on the cursor's version.

`Store(cas=False)` is the control for the race: it reads, checks in Python and
writes unconditionally, so two writers that read the same cursor both win.
"""

import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

import asyncpg

SCHEMA = """
create table if not exists turns (
    turn_id text primary key,
    state text not null,
    round int not null default 0,
    pending_interrupt text,
    version bigint not null default 0,
    lease_owner text,
    lease_until timestamptz,
    cancel_requested boolean not null default false,
    data jsonb not null default '{}'
);
create table if not exists agent_records (
    turn_id text not null references turns,
    agent text not null,
    body jsonb not null,
    primary key (turn_id, agent)
);
create table if not exists answers (
    turn_id text not null references turns,
    interrupt_id text not null,
    round int not null,
    answer jsonb not null,
    owner text not null,
    primary key (turn_id, interrupt_id, round)
);
"""

CANCEL_CHANNEL = "turn_cancel"
DONE = "done"


class Conflict(Exception):  # noqa: N818 -- named for the wire 409
    """A write lost its compare-and-swap, or the answer names no open pause."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass
class Cursor:
    turn_id: str
    state: str
    round: int
    pending_interrupt: str | None
    version: int
    lease_owner: str | None
    lease_live: bool
    cancel_requested: bool
    data: dict[str, Any]
    records: dict[str, Any] = field(default_factory=dict)


_SELECT = """
select turn_id, state, round, pending_interrupt, version, lease_owner,
       coalesce(lease_until > now(), false) as lease_live, cancel_requested, data
from turns where turn_id = $1
"""


def _cursor(row: asyncpg.Record, records: dict[str, Any]) -> Cursor:
    return Cursor(**{**dict(row), "data": json.loads(row["data"])}, records=records)


async def connect(dsn: str) -> asyncpg.Pool:
    pool = await asyncpg.create_pool(dsn, min_size=1, max_size=4)
    async with pool.acquire() as conn:
        await conn.execute("select pg_advisory_lock(7)")
        await conn.execute(SCHEMA)
        await conn.execute("select pg_advisory_unlock(7)")
    return pool


@dataclass
class Store:
    pool: asyncpg.Pool
    ttl: float = 2.0
    cas: bool = True
    match_round: bool = True
    # Awaited between reading the cursor and writing the claim; tests use it
    # to hold two claimants on the same read.
    after_read: Callable[[], Awaitable[None]] | None = None

    async def load(self, turn_id: str) -> Cursor:
        async with self.pool.acquire() as conn:
            return await self._load(conn, turn_id)

    async def _load(self, conn: asyncpg.Connection, turn_id: str) -> Cursor:
        row = await conn.fetchrow(_SELECT, turn_id)
        if row is None:
            raise Conflict("no such turn")
        rows = await conn.fetch("select agent, body from agent_records where turn_id = $1", turn_id)
        return _cursor(row, {r["agent"]: json.loads(r["body"]) for r in rows})

    async def create(self, turn_id: str, *, state: str, owner: str, data: dict) -> Cursor:
        async with self.pool.acquire() as conn:
            await conn.execute(
                "insert into turns (turn_id, state, lease_owner, lease_until, data)"
                " values ($1, $2, $3, now() + make_interval(secs => $4), $5)",
                turn_id,
                state,
                owner,
                self.ttl,
                json.dumps(data),
            )
            return await self._load(conn, turn_id)

    async def _swap(
        self,
        conn: asyncpg.Connection,
        cursor: Cursor,
        *,
        guard: str,
        sets: str,
        args: list[Any],
    ) -> None:
        """Bump the version and apply `sets` where the cursor still holds the
        version this writer read and `guard` holds; `Conflict` otherwise. With
        `cas=False` the guard and version check were done on the read, and the
        write is unconditional."""
        check = f"version = $2 and ({guard})"
        # Without the swap the check is still in the statement, so every
        # parameter keeps a type, but `true or` makes it decide nothing.
        where = f"turn_id = $1 and ({check})" if self.cas else f"turn_id = $1 and (true or {check})"
        status = await conn.execute(
            f"update turns set version = version + 1, {sets} where {where}",
            cursor.turn_id,
            cursor.version,
            *args,
        )
        if status != "UPDATE 1":
            raise Conflict("stale cursor")

    async def advance(
        self,
        cursor: Cursor,
        *,
        owner: str,
        state: str,
        round: int,
        data: dict[str, Any],
        records: dict[str, Any],
        pending: str | None = None,
    ) -> Cursor:
        """Move the cursor and write the records changed with it in one
        transaction, fenced on this owner's lease. A pause (`pending`) releases
        the lease."""
        lease = "lease_owner = null, lease_until = null" if pending else "lease_owner = $3"
        async with self.pool.acquire() as conn, conn.transaction():
            await self._swap(
                conn,
                cursor,
                guard="lease_owner = $3",
                sets=f"state = $4, round = $5, data = $6, pending_interrupt = $7, {lease}",
                args=[owner, state, round, json.dumps(data), pending],
            )
            for agent, body in records.items():
                await conn.execute(
                    "insert into agent_records values ($1, $2, $3)"
                    " on conflict (turn_id, agent) do update set body = excluded.body",
                    cursor.turn_id,
                    agent,
                    json.dumps(body),
                )
            return await self._load(conn, cursor.turn_id)

    async def claim_answer(
        self, turn_id: str, *, interrupt_id: str, round: int, answer: dict, owner: str
    ) -> tuple[Cursor, dict]:
        """Accept `answer` to the pause `(interrupt_id, round)` and take the lease.

        The answer the turn resumes with comes back beside the cursor. An
        answer already accepted for that pause, on a turn not done whose lease
        has lapsed, takes the lease over and returns the stored answer: the
        turn continues from its last checkpoint. Anything else is a `Conflict`.
        """
        async with self.pool.acquire() as conn, conn.transaction():
            cursor = await self._load(conn, turn_id)
            if self.after_read is not None:
                await self.after_read()
            stored = await conn.fetchval(
                "select answer from answers where turn_id = $1 and interrupt_id = $2"
                " and round = $3",
                turn_id,
                interrupt_id,
                round,
            )
            # The controls know no answered pause: without the swap nothing
            # fences it, and without rounds a pause has no stable identity.
            if stored is not None and self.cas and self.match_round:
                if cursor.state == DONE or cursor.lease_live or cursor.pending_interrupt:
                    raise Conflict("already answered")
                await self._swap(
                    conn,
                    cursor,
                    guard="lease_owner is null or lease_until < now()",
                    sets="lease_owner = $3, lease_until = now() + make_interval(secs => $4)",
                    args=[owner, self.ttl],
                )
                return await self._load(conn, turn_id), json.loads(stored)
            open_pause = cursor.pending_interrupt == interrupt_id and (
                cursor.round == round or not self.match_round
            )
            if not open_pause:
                raise Conflict("no open pause")
            await self._swap(
                conn,
                cursor,
                guard="pending_interrupt = $5 and round = $6",
                sets=(
                    "pending_interrupt = null, lease_owner = $3,"
                    " lease_until = now() + make_interval(secs => $4)"
                ),
                args=[owner, self.ttl, interrupt_id, cursor.round],
            )
            await conn.execute(
                "insert into answers values ($1, $2, $3, $4, $5) on conflict do nothing",
                turn_id,
                interrupt_id,
                cursor.round,
                json.dumps(answer),
                owner,
            )
            return await self._load(conn, turn_id), answer

    async def claim_resume(self, turn_id: str, *, owner: str) -> Cursor:
        """Take the lease of a turn stopped between pauses, such as after a cancel."""
        async with self.pool.acquire() as conn, conn.transaction():
            cursor = await self._load(conn, turn_id)
            if cursor.state == DONE or cursor.pending_interrupt or cursor.lease_live:
                raise Conflict("not resumable")
            await self._swap(
                conn,
                cursor,
                guard="lease_owner is null or lease_until < now()",
                sets="lease_owner = $3, lease_until = now() + make_interval(secs => $4)",
                args=[owner, self.ttl],
            )
            return await self._load(conn, turn_id)

    async def renew(self, turn_id: str, *, owner: str) -> bool | None:
        """Push the lease forward without a version bump, so the runner's cursor
        stays current. The turn's cancel flag, or `None` when this owner no
        longer holds the lease."""
        async with self.pool.acquire() as conn:
            return await conn.fetchval(
                "update turns set lease_until = now() + make_interval(secs => $3)"
                " where turn_id = $1 and lease_owner = $2 returning cancel_requested",
                turn_id,
                owner,
                self.ttl,
            )

    async def release(self, turn_id: str, *, owner: str) -> None:
        """Give the lease up where the turn stands and consume a cancel."""
        async with self.pool.acquire() as conn:
            await conn.execute(
                "update turns set lease_owner = null, lease_until = null,"
                " cancel_requested = false where turn_id = $1 and lease_owner = $2",
                turn_id,
                owner,
            )

    async def request_cancel(self, turn_id: str) -> None:
        """Flag the turn and notify every listener in one transaction."""
        async with self.pool.acquire() as conn, conn.transaction():
            await conn.execute(
                "update turns set cancel_requested = true where turn_id = $1", turn_id
            )
            await conn.execute("select pg_notify($1, $2)", CANCEL_CHANNEL, turn_id)
