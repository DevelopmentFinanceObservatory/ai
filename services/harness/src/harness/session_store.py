"""Builds the ADK session service over the database `harness.settings.Settings`
configures.

`build_session_service` runs `DatabaseSessionService` over the engine
`harness.db.engine.build_database` yields (a plain SQLAlchemy URL or the Cloud
SQL Connector; see `harness.db.engine`), and refuses settings that configure
no database. `Settings._validate_session_store_backend` already refuses the
combinations that would make the choice ambiguous. `adk_turn_specs` reads an
open turn's spec blocks out of its ADK session for `harness.studio_store`.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from google.adk.sessions import BaseSessionService, DatabaseSessionService
from google.adk.sessions.base_session_service import GetSessionConfig
from sqlalchemy.ext.asyncio import AsyncEngine

from harness.db.engine import build_database
from harness.graph.state import ORCHESTRATOR_SPECS_KEY
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


def adk_turn_specs(
    service: BaseSessionService, *, app_name: str
) -> Callable[[str, str], Awaitable[dict[str, Any]]]:
    """`harness.studio_store.StudioStore`'s `turn_specs` reader over
    `service`: the spec blocks (`ORCHESTRATOR_SPECS_KEY`) the turn session
    `session_id` under `app_name` holds, for `ConversationState.close_turn` to
    keep those of the sources its plans used. Empty when the session holds
    none or does not exist yet."""

    async def turn_specs(principal: str, session_id: str) -> dict[str, Any]:
        session = await service.get_session(
            app_name=app_name,
            user_id=principal,
            session_id=session_id,
            config=GetSessionConfig(num_recent_events=0),
        )
        return dict((session.state.get(ORCHESTRATOR_SPECS_KEY) or {}) if session else {})

    return turn_specs
