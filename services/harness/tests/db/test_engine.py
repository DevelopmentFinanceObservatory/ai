"""`harness.db.engine.build_database`: every runtime connection carries
asyncpg's `command_timeout` (`COMMAND_TIMEOUT_S`), on the `DATABASE_URL`
engine and on the Cloud SQL Connector's."""

from __future__ import annotations

from typing import Any

import harness.db.engine as engine_module
import pytest
from harness.settings import Settings
from sqlalchemy import text


async def test_a_statement_that_outlasts_the_command_timeout_raises(
    pg_database_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A statement still running past `COMMAND_TIMEOUT_S` raises
    `TimeoutError` on the `DATABASE_URL` engine, and one that ends in time
    returns. The bound is shrunk so the test waits under a second."""
    monkeypatch.setattr(engine_module, "COMMAND_TIMEOUT_S", 0.2)
    database = engine_module.build_database(settings=Settings(database_url=pg_database_url))
    assert database is not None
    try:
        async with database.engine.connect() as conn:
            assert (await conn.execute(text("SELECT 1"))).scalar_one() == 1
        with pytest.raises(TimeoutError):
            async with database.engine.connect() as conn:
                await conn.execute(text("SELECT pg_sleep(2)"))
    finally:
        await database.aclose()


async def test_the_connector_dials_each_connection_with_the_command_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The connector `build_database` builds passes `COMMAND_TIMEOUT_S` to
    asyncpg's connect, beside its IAM and database arguments."""
    dialled: list[dict[str, Any]] = []

    class _Connector:
        async def connect_async(self, name: str, driver: str, **kwargs: Any) -> object:
            dialled.append({"name": name, "driver": driver, **kwargs})
            return object()

    async def create_async_connector(**_: Any) -> _Connector:
        return _Connector()

    monkeypatch.setattr(engine_module, "create_async_connector", create_async_connector)
    sources: list[Any] = []
    real_source = engine_module._ConnectorSource

    def recording_source(**kwargs: Any) -> Any:
        sources.append(real_source(**kwargs))
        return sources[-1]

    monkeypatch.setattr(engine_module, "_ConnectorSource", recording_source)
    database = engine_module.build_database(
        settings=Settings(
            session_store_instance_connection_name="project:region:instance",
            session_store_database="ai",
            session_store_runtime_db_user="ai-runtime",
        )
    )
    assert database is not None
    (source,) = sources

    await source.connect()

    assert dialled == [
        {
            "name": "project:region:instance",
            "driver": "asyncpg",
            "user": "ai-runtime",
            "db": "ai",
            "enable_iam_auth": True,
            "command_timeout": engine_module.COMMAND_TIMEOUT_S,
        }
    ]
