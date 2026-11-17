"""A clock a test moves for both of `StudioStore`'s clocks at once.

The store decides lease expiry on the database's clock, read through the SQL
expression it is given (`StudioStore(time_sql=...)`), and stamps run records
on its instance clock (`clock`). `StoreClock` holds one reading in a one-row
table of the test's schema and hands the store both: `SQL` reads the row,
and calling the clock returns the same value. Every store built on the
schema, standing for every instance, shares the one reading.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine


class StoreClock:
    SQL = "(SELECT at FROM store_clock)"

    def __init__(self, engine: AsyncEngine, at: float) -> None:
        self._engine = engine
        self.at = at

    @classmethod
    async def start(cls, engine: AsyncEngine, at: float = 1_000.0) -> StoreClock:
        """A clock reading `at`, its table created in `engine`'s schema."""
        async with engine.begin() as conn:
            await conn.execute(text("CREATE TABLE store_clock (at double precision NOT NULL)"))
            await conn.execute(text("INSERT INTO store_clock (at) VALUES (:at)"), {"at": at})
        return cls(engine, at)

    def __call__(self) -> float:
        return self.at

    async def set(self, at: float) -> None:
        async with self._engine.begin() as conn:
            await conn.execute(text("UPDATE store_clock SET at = :at"), {"at": at})
        self.at = at

    async def advance(self, seconds: float) -> None:
        await self.set(self.at + seconds)

    @property
    def store_kwargs(self) -> dict[str, Any]:
        """`StudioStore`'s keyword arguments that put it on this clock."""
        return {"clock": self, "time_sql": self.SQL}
