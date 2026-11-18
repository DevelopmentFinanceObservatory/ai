from __future__ import annotations

from typing import Any

import pytest
from google.cloud.sql.connector import IPTypes
from harness.db import engine as engine_module
from harness.db.engine import build_database
from harness.settings import Settings


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
