"""Builds the ADK session-service backend selected by `harness.settings.Settings`.

Three backends, chosen by `build_session_service` from which of `Settings`'
session fields are set (`Settings._validate_session_store_backend` already
refuses the combinations that would make the choice ambiguous): in-memory
when none of them is set, `DatabaseSessionService` over a plain SQLAlchemy
URL when `database_url` is set, or `DatabaseSessionService` over an async
engine built on the Cloud SQL Python Connector (PSC, IAM auth) when
`session_store_instance_connection_name` is set -- the shape a deployed
`ai` Cloud Run instance connecting to Postgres over IAM auth needs.

The connector path mirrors apis' `dataset_store.connection.build_connector`
(`refresh_strategy="lazy"`, so a Cloud Run freeze/thaw cycle never hands out
a stale token; `ip_type=IPTypes.PSC`, the only reachable path to the
platform Postgres instance) and diverges from it only because ADK's
`DatabaseSessionService` needs a SQLAlchemy `AsyncEngine`, not a bare
DB-API connection pool: `create_async_connector` / `connect_async(...,
"asyncpg", ...)` / `close_async()` replace apis' synchronous
`Connector` / `pg8000` / `close()` calls one level down.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from google.adk.sessions import BaseSessionService, DatabaseSessionService, InMemorySessionService
from google.cloud.sql.connector import IPTypes, create_async_connector
from sqlalchemy.ext.asyncio import create_async_engine

from harness.settings import Settings

logger = logging.getLogger(__name__)

# The one Postgres async driver ADK's DatabaseSessionService expects
# (`_ASYNC_DRIVER_BY_BACKEND["postgresql"]`); the connector also supports
# pg8000 and psycopg, but only asyncpg satisfies ADK's async engine.
_CONNECTOR_DRIVER = "asyncpg"


async def _aclose_noop() -> None:
    """`InMemorySessionService` holds nothing to dispose -- `BaseSessionService`
    declares no `close()` at all, and only `DatabaseSessionService` does."""


@dataclass
class SessionServiceHandle:
    """The session service plus its async teardown.

    `service` is what `harness.server.create_app` stores at
    `app.state.session_service`. `aclose` disposes whatever this handle's
    construction path opened; call it from the harness lifespan's shutdown
    phase. It is a no-op for the in-memory backend. For the plain-URL
    backend it is `service.close`, which disposes the engine
    `DatabaseSessionService` built for itself (`_owns_db_engine = True`
    there). For the Cloud SQL Connector path it disposes the engine this
    module built and closes the connector directly, since ADK's own
    `close()` does not dispose an engine it did not build (`db_engine=`
    construction sets `_owns_db_engine = False`).
    """

    service: BaseSessionService
    aclose: Callable[[], Awaitable[None]]


@dataclass
class _ConnectorSource:
    """One Cloud SQL Python Connector, built lazily on the first connection.

    `create_async_connector` needs a running event loop
    (`asyncio.get_running_loop()`), but `build_session_service` runs
    synchronously inside `harness.server.create_app`, before uvicorn's own
    loop starts -- so the connector can't be built there. It is built
    instead the first time `connect` runs, which SQLAlchemy's pool only
    ever does from inside a running loop, guarded by a lock so two
    connections racing to be first build only one connector.
    """

    instance_connection_name: str
    database: str
    runtime_db_user: str
    connect_timeout_s: float
    _connector: Any = field(default=None, init=False)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False)

    async def connect(self) -> Any:
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
        )

    async def aclose(self) -> None:
        if self._connector is not None:
            await self._connector.close_async()


def build_session_service(*, settings: Settings) -> SessionServiceHandle:
    """Return the session-service handle selected by `settings`.

    See the module docstring for the three backends and the selection
    order; `Settings` itself refuses a `settings` where more than one of
    them could apply.
    """
    if settings.session_store_instance_connection_name is not None:
        return _build_connector_backed_service(settings=settings)
    if settings.database_url is not None:
        logger.info("Using database session service over DATABASE_URL")
        service = DatabaseSessionService(db_url=settings.database_url)
        return SessionServiceHandle(service=service, aclose=service.close)
    logger.info("Using in-memory session service")
    return SessionServiceHandle(service=InMemorySessionService(), aclose=_aclose_noop)


def _build_connector_backed_service(*, settings: Settings) -> SessionServiceHandle:
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
    # The URL carries only the dialect+driver; `async_creator` supplies the
    # actual connection, and SQLAlchemy's own docs for it say the URL's
    # parameters are never applied. `pool_pre_ping=True` matches ADK's own
    # default for every non-sqlite backend built from a `db_url`.
    engine = create_async_engine(
        "postgresql+asyncpg://",
        async_creator=source.connect,
        pool_size=settings.session_store_pool_size,
        max_overflow=settings.session_store_max_overflow,
        pool_pre_ping=True,
    )

    async def aclose() -> None:
        await engine.dispose()
        await source.aclose()

    logger.info("Using database session service over the Cloud SQL Connector")
    return SessionServiceHandle(service=DatabaseSessionService(db_engine=engine), aclose=aclose)
