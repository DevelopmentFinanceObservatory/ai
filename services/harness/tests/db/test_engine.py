"""`harness.db.engine.build_database`: every runtime connection carries
asyncpg's `command_timeout` (`COMMAND_TIMEOUT_S`), on the `DATABASE_URL`
engine and on the Cloud SQL Connector's."""

from __future__ import annotations

from types import SimpleNamespace
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


@pytest.mark.parametrize(
    ("url", "connect_args"),
    [
        ("postgresql+asyncpg://u@db/ai", {"command_timeout": engine_module.COMMAND_TIMEOUT_S}),
        ("postgresql+psycopg://u@db/ai", {}),
    ],
    ids=["asyncpg", "psycopg"],
)
def test_only_an_asyncpg_url_gets_the_command_timeout(
    monkeypatch: pytest.MonkeyPatch, url: str, connect_args: dict[str, Any]
) -> None:
    """`command_timeout` is asyncpg's own argument, so a `DATABASE_URL` on
    another driver connects without it."""
    built: list[dict[str, Any]] = []

    def create_async_engine(url: str, **kwargs: Any) -> Any:
        built.append(kwargs)
        return SimpleNamespace(dispose=None)

    monkeypatch.setattr(engine_module, "create_async_engine", create_async_engine)
    engine_module.build_database(settings=Settings(database_url=url))

    assert [kwargs["connect_args"] for kwargs in built] == [connect_args]
