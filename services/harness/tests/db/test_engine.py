from __future__ import annotations

from typing import Any

import pytest
from google.cloud.sql.connector import IPTypes
from harness.db import engine as engine_module
from harness.db.engine import build_database
from harness.settings import Settings
from sqlalchemy import text
from sqlalchemy.exc import DBAPIError


class _FakeConnector:
    """Captures every argument `_ConnectorSource` passes it, instead of
    dialling a real Cloud SQL instance."""

    def __init__(self) -> None:
        self.connect_calls: list[dict[str, Any]] = []
        self.closed = False

    async def connect_async(self, instance_connection_string: str, driver: str, **kwargs: Any):
        self.connect_calls.append(
            {"instance_connection_string": instance_connection_string, "driver": driver, **kwargs}
        )
        raise RuntimeError("test double: never dials a real Cloud SQL instance")

    async def close_async(self) -> None:
        self.closed = True


@pytest.fixture
def fake_connector(monkeypatch) -> _FakeConnector:
    connector = _FakeConnector()
    captured_kwargs: dict[str, Any] = {}

    async def fake_create_async_connector(**kwargs: Any) -> _FakeConnector:
        captured_kwargs.update(kwargs)
        return connector

    monkeypatch.setattr(engine_module, "create_async_connector", fake_create_async_connector)
    connector.create_async_connector_kwargs = captured_kwargs  # type: ignore[attr-defined]
    return connector


async def test_connector_backend_passes_iam_auth_and_psc_settings_through(
    fake_connector,
) -> None:
    """The connector path builds one connector with `refresh_strategy="lazy"`
    and `ip_type=IPTypes.PSC`, then dials with the asyncpg driver and the
    settings-provided user/database, authenticating as the runtime service
    account over IAM rather than a password."""
    settings = Settings(
        session_store_instance_connection_name="my-project:my-region:my-instance",
        session_store_database="sessions",
        session_store_runtime_db_user="ai-run",
        session_store_connect_timeout_s=12.0,
    )
    database = build_database(settings=settings)
    assert database is not None

    with pytest.raises(RuntimeError, match="never dials a real Cloud SQL instance"):
        async with database.engine.connect():
            pass

    assert fake_connector.create_async_connector_kwargs == {  # type: ignore[attr-defined]
        "refresh_strategy": "lazy",
        "ip_type": IPTypes.PSC,
        "timeout": 12,
    }
    assert fake_connector.connect_calls == [
        {
            "instance_connection_string": "my-project:my-region:my-instance",
            "driver": "asyncpg",
            "user": "ai-run",
            "db": "sessions",
            "enable_iam_auth": True,
        }
    ]

    await database.aclose()


async def test_connector_backend_aclose_disposes_engine_and_closes_connector(
    fake_connector,
) -> None:
    settings = Settings(
        session_store_instance_connection_name="my-project:my-region:my-instance",
        session_store_database="sessions",
        session_store_runtime_db_user="ai-run",
    )
    database = build_database(settings=settings)
    assert database is not None

    with pytest.raises(RuntimeError, match="never dials a real Cloud SQL instance"):
        async with database.engine.connect():
            pass

    await database.aclose()

    assert fake_connector.closed is True


async def test_connector_backend_builds_the_connector_only_once_across_pooled_connections(
    fake_connector,
) -> None:
    """Two connections through the pool share one connector -- built lazily
    on the first, not rebuilt on the second."""
    settings = Settings(
        session_store_instance_connection_name="my-project:my-region:my-instance",
        session_store_database="sessions",
        session_store_runtime_db_user="ai-run",
        session_store_pool_size=1,
    )
    database = build_database(settings=settings)
    assert database is not None

    for _ in range(2):
        with pytest.raises(RuntimeError, match="never dials a real Cloud SQL instance"):
            async with database.engine.connect():
                pass

    assert len(fake_connector.connect_calls) == 2

    await database.aclose()


@pytest.mark.parametrize(
    "settings",
    [
        pytest.param(
            Settings(database_url="postgresql+asyncpg://u@localhost/db"), id="database-url"
        ),
        pytest.param(
            Settings(
                session_store_instance_connection_name="my-project:my-region:my-instance",
                session_store_database="sessions",
                session_store_runtime_db_user="ai-run",
                session_store_pool_size=4,
                session_store_max_overflow=2,
            ),
            id="connector",
        ),
    ],
)
async def test_lineage_reads_get_a_pool_of_their_own(settings: Settings, fake_connector) -> None:
    """The lineage engine shares no pool with the session store's engine, and
    holds three connections with no overflow, so four instances hold at
    most 4 x (2 + 3) = 20 connections."""
    database = build_database(settings=settings)
    assert database is not None

    assert database.lineage_engine is not database.engine
    assert database.lineage_engine.pool is not database.engine.pool
    # `size` and `_max_overflow` are on the `AsyncAdaptedQueuePool` the
    # engine builds, not on the `Pool` it is typed as.
    assert database.lineage_engine.pool.size() == 3  # type: ignore[attr-defined]
    assert database.lineage_engine.pool._max_overflow == 0  # type: ignore[attr-defined]
    assert database.lineage_engine.pool.timeout() == 1.5  # type: ignore[attr-defined]
    assert database.engine.sync_engine.hide_parameters
    assert database.lineage_engine.sync_engine.hide_parameters

    await database.aclose()


async def test_the_connector_backed_lineage_engine_bounds_each_statement(fake_connector) -> None:
    """Its connections are dialled with asyncpg's `command_timeout`, so a
    stalled lineage read cannot hold a pool connection."""
    settings = Settings(
        session_store_instance_connection_name="my-project:my-region:my-instance",
        session_store_database="sessions",
        session_store_runtime_db_user="ai-run",
    )
    database = build_database(settings=settings)
    assert database is not None

    with pytest.raises(RuntimeError, match="never dials a real Cloud SQL instance"):
        async with database.lineage_engine.connect():
            pass

    assert fake_connector.connect_calls[-1]["command_timeout"] == 3.0
    await database.aclose()


async def test_a_failed_statement_leaves_its_bound_values_out_of_the_error(
    pg_database_url: str,
) -> None:
    """The error of a failed write, as the turn loop logs it, carries no bound
    value, so a turn's transcript stays out of the logs."""
    planted = "PLANTED-TRANSCRIPT-TEXT"
    database = build_database(settings=Settings(database_url=pg_database_url))
    assert database is not None

    with pytest.raises(DBAPIError) as raised:
        async with database.engine.begin() as connection:
            await connection.execute(
                text("insert into no_such_table (body) values (:body)"), {"body": planted}
            )

    assert planted not in repr(raised.value)
    assert planted not in str(raised.value)

    await database.aclose()
