from __future__ import annotations

from typing import Any

import pytest
from google.adk.events import Event, EventActions
from google.adk.sessions import DatabaseSessionService, InMemorySessionService
from google.cloud.sql.connector import IPTypes
from harness import session_store as session_store_module
from harness.session_store import build_session_service
from harness.settings import Settings


def test_unset_backend_yields_in_memory() -> None:
    handle = build_session_service(settings=Settings())

    assert isinstance(handle.service, InMemorySessionService)


async def test_unset_backend_aclose_is_a_noop() -> None:
    handle = build_session_service(settings=Settings())

    await handle.aclose()


async def test_database_url_round_trips_a_session_and_disposes_on_aclose(
    tmp_path,
) -> None:
    """`DATABASE_URL=sqlite+aiosqlite://...` yields a working
    `DatabaseSessionService`: a session created, appended to, and reloaded
    carries the state the append wrote, and `aclose` disposes the engine it
    built for itself without raising."""
    settings = Settings(database_url=f"sqlite+aiosqlite:///{tmp_path}/sessions.db")
    handle = build_session_service(settings=settings)

    assert isinstance(handle.service, DatabaseSessionService)

    session = await handle.service.create_session(
        app_name="harness", user_id="user-1", session_id="conv-1"
    )
    await handle.service.append_event(
        session=session,
        event=Event(author="test", actions=EventActions(state_delta={"studio:turn_index": 1})),
    )

    reloaded = await handle.service.get_session(
        app_name="harness", user_id="user-1", session_id="conv-1"
    )

    assert reloaded is not None
    assert reloaded.state["studio:turn_index"] == 1

    await handle.aclose()


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

    monkeypatch.setattr(session_store_module, "create_async_connector", fake_create_async_connector)
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
    handle = build_session_service(settings=settings)

    assert isinstance(handle.service, DatabaseSessionService)

    with pytest.raises(RuntimeError, match="never dials a real Cloud SQL instance"):
        await handle.service.create_session(app_name="harness", user_id="u", session_id="s")

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

    await handle.aclose()


async def test_connector_backend_aclose_disposes_engine_and_closes_connector(
    fake_connector,
) -> None:
    settings = Settings(
        session_store_instance_connection_name="my-project:my-region:my-instance",
        session_store_database="sessions",
        session_store_runtime_db_user="ai-run",
    )
    handle = build_session_service(settings=settings)

    with pytest.raises(RuntimeError, match="never dials a real Cloud SQL instance"):
        await handle.service.create_session(app_name="harness", user_id="u", session_id="s")

    await handle.aclose()

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
    handle = build_session_service(settings=settings)

    for _ in range(2):
        with pytest.raises(RuntimeError, match="never dials a real Cloud SQL instance"):
            await handle.service.create_session(app_name="harness", user_id="u", session_id="s")

    assert len(fake_connector.connect_calls) == 2

    await handle.aclose()
