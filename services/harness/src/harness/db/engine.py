"""Builds the async SQLAlchemy engine selected by `harness.settings.Settings`.

`build_database` chooses from which of `Settings`' database fields are set
(`Settings._validate_session_store_backend` already refuses the combinations
that would make the choice ambiguous): no engine when none is set, an engine
over a plain SQLAlchemy URL when `database_url` is set, or an engine built on
the Cloud SQL Python Connector (PSC, IAM auth) when
`session_store_instance_connection_name` is set, the shape a deployed `ai`
Cloud Run instance connecting to Postgres over IAM auth needs.

The connector path mirrors apis' `dataset_store.connection.build_connector`.
It uses `refresh_strategy="lazy"`, so a Cloud Run freeze/thaw cycle does not
hand out a stale token, and `ip_type=IPTypes.PSC`, the only reachable path to
the platform Postgres instance. It diverges from apis only because
SQLAlchemy's async engine needs an async DB-API connection:
`create_async_connector` / `connect_async(..., "asyncpg", ...)` /
`close_async()` replace apis' synchronous `Connector` / `pg8000` / `close()`
calls one level down.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from google.cloud.sql.connector import IPTypes, create_async_connector
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from harness.settings import Settings

logger = logging.getLogger(__name__)

# The Postgres async driver every consumer of the engine expects. The
# connector also supports pg8000 and psycopg, and only asyncpg gives
# SQLAlchemy an async engine.
_CONNECTOR_DRIVER = "asyncpg"

# The pool of `DatabaseHandle.lineage_engine`: four connections and no
# overflow, for the card lineage reads and the turn frame lineage restores of
# the turns that resume at once. A read that finds it busy waits at most
# `pool_timeout`, inside its caller's budget
# (`harness.mcp_client._LINEAGE_READ_BUDGET_S` and
# `harness.turn.store._FRAME_LINEAGE_READ_BUDGET_S`, 2 s each), which lets
# the caller go on without it.
_LINEAGE_POOL: dict[str, Any] = {"pool_size": 4, "max_overflow": 0, "pool_timeout": 1.5}

# asyncpg's bound on each statement over the lineage engine, cleanup's
# rollback included, so a read the caller abandoned on a stalled statement
# ends and gives its connection back.
_LINEAGE_COMMAND_TIMEOUT_S = 3.0


@dataclass
class DatabaseHandle:
    """The engines plus their async teardown.

    `engine` holds the session store's pool. `lineage_engine` is a pool of
    four connections, no overflow, on the same settings, for the best-effort
    lineage reads: a card's (`StudioStore.card_lineage`) and a resuming
    turn's frames' (`harness.turn.store.TurnStore.restore_frame_lineage`), so
    those reads never take a connection the lease renewals, cursor writes and
    card writes need, and several turns resuming at once do not queue on one
    connection.
    `aclose` disposes both and, on the Cloud SQL Connector path, closes the
    connector their connections were dialled through. Call it from the
    harness lifespan's shutdown phase.
    """

    engine: AsyncEngine
    lineage_engine: AsyncEngine
    aclose: Callable[[], Awaitable[None]]


@dataclass
class _ConnectorSource:
    """One Cloud SQL Python Connector, built lazily on the first connection.

    `create_async_connector` needs a running event loop
    (`asyncio.get_running_loop()`), but `build_database` runs synchronously
    inside `harness.server.create_app`, before uvicorn's own loop starts, so
    the connector cannot be built there. It is built the first time `connect`
    runs, which SQLAlchemy's pool only does from inside a running loop, under
    a lock so two connections racing to be first build one connector.
    """

    instance_connection_name: str
    database: str
    runtime_db_user: str
    connect_timeout_s: float
    _connector: Any = field(default=None, init=False)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False)

    async def connect(self, **driver_kwargs: Any) -> Any:
        """A new connection, with `driver_kwargs` passed on to asyncpg."""
        if self._connector is None:
            async with self._lock:
                if self._connector is None:
                    self._connector = await create_async_connector(
                        refresh_strategy="lazy",
                        ip_type=IPTypes.PSC,
                        timeout=int(self.connect_timeout_s),
                    )
        return await self._connector.connect_async(
            self.instance_connection_name,
            _CONNECTOR_DRIVER,
            user=self.runtime_db_user,
            db=self.database,
            enable_iam_auth=True,
            **driver_kwargs,
        )

    async def aclose(self) -> None:
        if self._connector is not None:
            await self._connector.close_async()


def build_database(*, settings: Settings) -> DatabaseHandle | None:
    """Return the database handle selected by `settings`, or `None` when no
    database is configured.

    See the module docstring for the two backends and the selection order;
    `Settings` itself refuses a `settings` where both could apply.
    """
    if settings.session_store_instance_connection_name is not None:
        return _build_connector_backed_database(settings=settings)
    if settings.database_url is not None:
        logger.info("Using the database at DATABASE_URL")
        # `pool_pre_ping` drops a connection the server closed while it sat in
        # the pool.
        engine = create_async_engine(settings.database_url, pool_pre_ping=True)
        lineage_engine = build_lineage_engine(settings.database_url)

        async def aclose_url() -> None:
            await engine.dispose()
            await lineage_engine.dispose()

        return DatabaseHandle(engine=engine, lineage_engine=lineage_engine, aclose=aclose_url)
    return None


def build_lineage_engine(
    url: str, *, async_creator: Callable[[], Awaitable[Any]] | None = None
) -> AsyncEngine:
    """The engine of `DatabaseHandle.lineage_engine` over `url`: one pooled
    connection, no overflow, a bounded checkout wait and a bounded statement
    (`_LINEAGE_POOL`, `_LINEAGE_COMMAND_TIMEOUT_S`). With `async_creator`,
    which then sets the statement bound itself, connections come from it."""
    if async_creator is not None:
        return create_async_engine(
            url, async_creator=async_creator, pool_pre_ping=True, **_LINEAGE_POOL
        )
    return create_async_engine(
        url,
        pool_pre_ping=True,
        connect_args={"command_timeout": _LINEAGE_COMMAND_TIMEOUT_S},
        **_LINEAGE_POOL,
    )


def _build_connector_backed_database(*, settings: Settings) -> DatabaseHandle:
    connection_name = settings.session_store_instance_connection_name
    database = settings.session_store_database
    runtime_db_user = settings.session_store_runtime_db_user
    if connection_name is None or database is None or runtime_db_user is None:
        raise ValueError(
            "session_store_instance_connection_name requires session_store_database "
            "and session_store_runtime_db_user; Settings validation should already "
            "have refused this combination."
        )

    source = _ConnectorSource(
        instance_connection_name=connection_name,
        database=database,
        runtime_db_user=runtime_db_user,
        connect_timeout_s=settings.session_store_connect_timeout_s,
    )
    # The URL carries only the dialect+driver. `async_creator` supplies the
    # actual connection, and SQLAlchemy ignores the URL's parameters when it
    # is set.
    engine = create_async_engine(
        "postgresql+asyncpg://",
        async_creator=source.connect,
        pool_size=settings.session_store_pool_size,
        max_overflow=settings.session_store_max_overflow,
        pool_pre_ping=True,
    )
    lineage_engine = build_lineage_engine(
        "postgresql+asyncpg://",
        async_creator=lambda: source.connect(command_timeout=_LINEAGE_COMMAND_TIMEOUT_S),
    )

    async def aclose() -> None:
        await engine.dispose()
        await lineage_engine.dispose()
        await source.aclose()

    logger.info("Using the database over the Cloud SQL Connector")
    return DatabaseHandle(engine=engine, lineage_engine=lineage_engine, aclose=aclose)
