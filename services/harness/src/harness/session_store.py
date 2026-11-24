"""Builds the ADK session service over the database `harness.settings.Settings`
configures.

`build_session_service` runs `DatabaseSessionService` over the engine
`harness.db.engine.build_database` yields (a plain SQLAlchemy URL or the Cloud
SQL Connector; see `harness.db.engine`), and refuses settings that configure
no database. `Settings._validate_session_store_backend` already refuses the
combinations that would make the choice ambiguous.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from google.adk.sessions import DatabaseSessionService
from sqlalchemy.ext.asyncio import AsyncEngine

from harness.db.engine import build_database
from harness.settings import Settings

logger = logging.getLogger(__name__)


@dataclass
class SessionServiceHandle:
    """The session service, the engine behind it, and the async teardown.

    `service` is what `harness.server.create_app` stores at
    `app.state.session_service`. `engine` is the engine `service` runs on.
    `aclose` disposes the engine and, on the Cloud SQL Connector path, closes
    the connector; call it from the harness lifespan's shutdown phase.
    """

    service: DatabaseSessionService
    engine: AsyncEngine
    aclose: Callable[[], Awaitable[None]]


def build_session_service(*, settings: Settings) -> SessionServiceHandle:
    """Return the session-service handle over the database `settings`
    configures. Raises `RuntimeError` when it configures none."""
    database = build_database(settings=settings)
    if database is None:
        raise RuntimeError(
            "the harness stores sessions and conversations in Postgres: set DATABASE_URL "
            "to a Postgres URL or the SESSION_STORE_* variables of the Cloud SQL Connector"
        )
    logger.info("Using database session service")
    return SessionServiceHandle(
        service=DatabaseSessionService(db_engine=database.engine),
        engine=database.engine,
        aclose=database.aclose,
    )
