from __future__ import annotations

import math

import pytest
from harness.settings import Settings

_FIELDS = (
    "DATA_MCP_URL",
    "ANALYSIS_MCP_URL",
    "PRESENTATION_MCP_URL",
    "DATABASE_URL",
    "SESSION_STORE_INSTANCE_CONNECTION_NAME",
    "SESSION_STORE_DATABASE",
    "SESSION_STORE_RUNTIME_DB_USER",
    "SESSION_STORE_POOL_SIZE",
    "SESSION_STORE_MAX_OVERFLOW",
    "SESSION_STORE_CONNECT_TIMEOUT_S",
    "PORT",
    "MCP_API_TOKEN",
    "MCP_AUTH_DISABLED",
    "MCP_CALL_TIMEOUT_S",
    "MCP_LIST_TIMEOUT_S",
    "MCP_INIT_TIMEOUT_S",
    "EXECUTOR_MAX_CONCURRENT_STEPS",
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Isolate tests from any of these vars set in the ambient environment."""
    for var in _FIELDS:
        monkeypatch.delenv(var, raising=False)


def test_defaults_without_env() -> None:
    settings = Settings()

    assert settings.port == 8080
    assert settings.data_mcp_url == "http://127.0.0.1:8080/mcp/data/mcp"
    assert settings.analysis_mcp_url == "http://127.0.0.1:8080/mcp/analysis/mcp"
    assert settings.presentation_mcp_url == "http://127.0.0.1:8080/mcp/presentation/mcp"
    assert settings.database_url is None
    assert settings.session_store_instance_connection_name is None
    assert settings.session_store_database is None
    assert settings.session_store_runtime_db_user is None
    assert settings.session_store_pool_size == 2
    assert settings.session_store_max_overflow == 0
    assert settings.session_store_connect_timeout_s == 30.0
    assert settings.mcp_call_timeout_s == 60.0
    assert settings.mcp_list_timeout_s == 15.0
    assert settings.mcp_init_timeout_s == 10.0
    assert settings.executor_max_concurrent_steps == 4


def test_mcp_timeouts_read_from_env(monkeypatch) -> None:
    monkeypatch.setenv("MCP_CALL_TIMEOUT_S", "5")
    monkeypatch.setenv("MCP_LIST_TIMEOUT_S", "2.5")
    monkeypatch.setenv("MCP_INIT_TIMEOUT_S", "1")

    settings = Settings()

    assert settings.mcp_call_timeout_s == 5.0
    assert settings.mcp_list_timeout_s == 2.5
    assert settings.mcp_init_timeout_s == 1.0


def test_mcp_url_defaults_follow_port_when_unset(monkeypatch) -> None:
    """The loopback defaults track `port` -- the mounted apps live in the
    same process, on the same port, so a `PORT` override must too."""
    monkeypatch.setenv("PORT", "9000")

    settings = Settings()

    assert settings.data_mcp_url == "http://127.0.0.1:9000/mcp/data/mcp"
    assert settings.analysis_mcp_url == "http://127.0.0.1:9000/mcp/analysis/mcp"
    assert settings.presentation_mcp_url == "http://127.0.0.1:9000/mcp/presentation/mcp"


def test_reads_from_env(monkeypatch) -> None:
    monkeypatch.setenv("DATA_MCP_URL", "https://data-mcp.internal")
    monkeypatch.setenv("ANALYSIS_MCP_URL", "https://analysis-mcp.internal")
    monkeypatch.setenv("PRESENTATION_MCP_URL", "https://presentation-mcp.internal")
    monkeypatch.setenv("DATABASE_URL", "postgresql://db/harness")
    # An off-loopback URL refuses construction with no token (see the
    # off-loopback tests below) -- this test is about the URLs threading
    # through, so give it one.
    monkeypatch.setenv("MCP_API_TOKEN", "test-token")

    settings = Settings()

    assert settings.data_mcp_url == "https://data-mcp.internal"
    assert settings.analysis_mcp_url == "https://analysis-mcp.internal"
    assert settings.presentation_mcp_url == "https://presentation-mcp.internal"
    assert settings.database_url == "postgresql://db/harness"


def test_inherits_base_service_settings_fields() -> None:
    settings = Settings()

    assert settings.port == 8080
    assert settings.model_name == "gemini-3.8-flash"


def test_a_blank_studio_api_token_is_unset(monkeypatch) -> None:
    monkeypatch.setenv("STUDIO_API_TOKEN", "")
    assert Settings().studio_api_token is None
    monkeypatch.setenv("STUDIO_API_TOKEN", "   ")
    assert Settings().studio_api_token is None
    monkeypatch.setenv("STUDIO_API_TOKEN", "s3cret")
    assert Settings().studio_api_token == "s3cret"


def test_a_blank_mcp_api_token_is_unset(monkeypatch) -> None:
    monkeypatch.setenv("MCP_API_TOKEN", "")
    assert Settings().mcp_api_token is None
    monkeypatch.setenv("MCP_API_TOKEN", "s3cret")
    assert Settings().mcp_api_token == "s3cret"


def test_a_blank_database_url_is_unset(monkeypatch) -> None:
    monkeypatch.setenv("DATABASE_URL", "")
    assert Settings().database_url is None
    monkeypatch.setenv("DATABASE_URL", "   ")
    assert Settings().database_url is None
    monkeypatch.setenv("DATABASE_URL", "sqlite+aiosqlite:///:memory:")
    assert Settings().database_url == "sqlite+aiosqlite:///:memory:"


def test_blank_session_store_fields_are_unset(monkeypatch) -> None:
    monkeypatch.setenv("SESSION_STORE_INSTANCE_CONNECTION_NAME", "")
    monkeypatch.setenv("SESSION_STORE_DATABASE", "   ")
    monkeypatch.setenv("SESSION_STORE_RUNTIME_DB_USER", "")

    settings = Settings()

    assert settings.session_store_instance_connection_name is None
    assert settings.session_store_database is None
    assert settings.session_store_runtime_db_user is None


def test_session_store_connector_fields_read_from_env(monkeypatch) -> None:
    monkeypatch.setenv("SESSION_STORE_INSTANCE_CONNECTION_NAME", "proj:region:instance")
    monkeypatch.setenv("SESSION_STORE_DATABASE", "sessions")
    monkeypatch.setenv("SESSION_STORE_RUNTIME_DB_USER", "ai-run")
    monkeypatch.setenv("SESSION_STORE_POOL_SIZE", "5")
    monkeypatch.setenv("SESSION_STORE_MAX_OVERFLOW", "1")
    monkeypatch.setenv("SESSION_STORE_CONNECT_TIMEOUT_S", "10")

    settings = Settings()

    assert settings.session_store_instance_connection_name == "proj:region:instance"
    assert settings.session_store_database == "sessions"
    assert settings.session_store_runtime_db_user == "ai-run"
    assert settings.session_store_pool_size == 5
    assert settings.session_store_max_overflow == 1
    assert settings.session_store_connect_timeout_s == 10.0


def test_database_url_and_session_store_connection_name_together_is_refused(
    monkeypatch,
) -> None:
    monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://db/harness")
    monkeypatch.setenv("SESSION_STORE_INSTANCE_CONNECTION_NAME", "proj:region:instance")
    monkeypatch.setenv("SESSION_STORE_DATABASE", "sessions")
    monkeypatch.setenv("SESSION_STORE_RUNTIME_DB_USER", "ai-run")

    with pytest.raises(ValueError, match="DATABASE_URL and SESSION_STORE_INSTANCE_CONNECTION_NAME"):
        Settings()


def test_session_store_connection_name_without_database_or_user_is_refused(
    monkeypatch,
) -> None:
    monkeypatch.setenv("SESSION_STORE_INSTANCE_CONNECTION_NAME", "proj:region:instance")

    with pytest.raises(
        ValueError, match="SESSION_STORE_DATABASE and SESSION_STORE_RUNTIME_DB_USER"
    ):
        Settings()


def test_session_store_connection_name_without_only_the_user_is_refused(monkeypatch) -> None:
    monkeypatch.setenv("SESSION_STORE_INSTANCE_CONNECTION_NAME", "proj:region:instance")
    monkeypatch.setenv("SESSION_STORE_DATABASE", "sessions")

    with pytest.raises(ValueError, match="SESSION_STORE_RUNTIME_DB_USER"):
        Settings()


def test_session_store_connection_name_without_only_the_database_is_refused(monkeypatch) -> None:
    monkeypatch.setenv("SESSION_STORE_INSTANCE_CONNECTION_NAME", "proj:region:instance")
    monkeypatch.setenv("SESSION_STORE_RUNTIME_DB_USER", "ai-run")

    with pytest.raises(ValueError, match="SESSION_STORE_DATABASE"):
        Settings()


def test_session_store_database_alone_is_refused(monkeypatch) -> None:
    monkeypatch.setenv("SESSION_STORE_DATABASE", "sessions")

    with pytest.raises(
        ValueError,
        match="SESSION_STORE_INSTANCE_CONNECTION_NAME and SESSION_STORE_RUNTIME_DB_USER",
    ):
        Settings()


def test_session_store_runtime_db_user_alone_is_refused(monkeypatch) -> None:
    monkeypatch.setenv("SESSION_STORE_RUNTIME_DB_USER", "ai-run")

    with pytest.raises(
        ValueError,
        match="SESSION_STORE_INSTANCE_CONNECTION_NAME and SESSION_STORE_DATABASE",
    ):
        Settings()


def test_session_store_database_and_user_without_connection_name_is_refused(
    monkeypatch,
) -> None:
    monkeypatch.setenv("SESSION_STORE_DATABASE", "sessions")
    monkeypatch.setenv("SESSION_STORE_RUNTIME_DB_USER", "ai-run")

    with pytest.raises(ValueError, match="SESSION_STORE_INSTANCE_CONNECTION_NAME"):
        Settings()


def test_session_store_connection_name_with_database_and_user_is_accepted(monkeypatch) -> None:
    monkeypatch.setenv("SESSION_STORE_INSTANCE_CONNECTION_NAME", "proj:region:instance")
    monkeypatch.setenv("SESSION_STORE_DATABASE", "sessions")
    monkeypatch.setenv("SESSION_STORE_RUNTIME_DB_USER", "ai-run")

    settings = Settings()

    assert settings.session_store_instance_connection_name == "proj:region:instance"


def test_an_off_loopback_mount_url_refuses_construction_without_a_token(monkeypatch) -> None:
    monkeypatch.setenv("DATA_MCP_URL", "https://data-mcp.internal/mcp")

    with pytest.raises(ValueError, match="DATA_MCP_URL"):
        Settings()


def test_an_off_loopback_mount_url_is_fine_with_a_token(monkeypatch) -> None:
    monkeypatch.setenv("DATA_MCP_URL", "https://data-mcp.internal/mcp")
    monkeypatch.setenv("MCP_API_TOKEN", "test-token")

    assert Settings().data_mcp_url == "https://data-mcp.internal/mcp"


def test_an_off_loopback_mount_url_is_fine_when_auth_is_disabled(monkeypatch) -> None:
    monkeypatch.setenv("DATA_MCP_URL", "https://data-mcp.internal/mcp")
    monkeypatch.setenv("MCP_AUTH_DISABLED", "true")

    assert Settings().data_mcp_url == "https://data-mcp.internal/mcp"


def test_the_default_loopback_urls_never_trip_the_off_loopback_refusal() -> None:
    """No override at all -- `_default_mcp_urls_to_loopback` fills in the
    loopback defaults before the refusal check runs, so an unset
    `MCP_API_TOKEN` is fine as long as nothing points off loopback."""
    Settings()


def test_an_off_loopback_http_mount_url_refuses_construction_with_a_token(monkeypatch) -> None:
    """A token makes the URL reachable, but `harness.mcp_client` attaches that
    same shared bearer to every request over it -- cleartext HTTP would put
    it on the wire in the open, so this refuses even though the missing-token
    rule (`test_an_off_loopback_mount_url_refuses_construction_without_a_token`)
    is satisfied."""
    monkeypatch.setenv("DATA_MCP_URL", "http://data-mcp.internal/mcp")
    monkeypatch.setenv("MCP_API_TOKEN", "test-token")

    with pytest.raises(ValueError, match="DATA_MCP_URL"):
        Settings()


def test_an_off_loopback_https_mount_url_is_fine_with_a_token(monkeypatch) -> None:
    monkeypatch.setenv("DATA_MCP_URL", "https://data-mcp.internal/mcp")
    monkeypatch.setenv("MCP_API_TOKEN", "test-token")

    assert Settings().data_mcp_url == "https://data-mcp.internal/mcp"


def test_a_loopback_http_mount_url_is_unaffected_by_the_scheme_refusal(monkeypatch) -> None:
    """Local dev points these at plain `http://127.0.0.1` -- the scheme
    refusal only ever looks at off-loopback overrides."""
    monkeypatch.setenv("DATA_MCP_URL", "http://127.0.0.1:8080/mcp/data/mcp")
    monkeypatch.setenv("MCP_API_TOKEN", "test-token")

    assert Settings().data_mcp_url == "http://127.0.0.1:8080/mcp/data/mcp"


@pytest.mark.parametrize(
    "field", ["mcp_call_timeout_s", "mcp_list_timeout_s", "mcp_init_timeout_s"]
)
@pytest.mark.parametrize("value", [0.0, -1.0, -0.001, math.nan, math.inf, -math.inf])
def test_a_zero_negative_or_non_finite_mcp_timeout_is_refused(field, value) -> None:
    """`FastMCP` normalises `init_timeout=0` to `None` -- the unbounded wait
    these settings exist to rule out -- and a negative or non-finite value
    has no sane meaning as a deadline either way; refuse all three shapes at
    construction, for all three timeout fields."""
    with pytest.raises(ValueError, match="must be a finite number greater than zero"):
        Settings(**{field: value})


@pytest.mark.parametrize(
    "field", ["mcp_call_timeout_s", "mcp_list_timeout_s", "mcp_init_timeout_s"]
)
def test_a_positive_finite_mcp_timeout_is_accepted(field) -> None:
    settings = Settings(**{field: 5.0})

    assert getattr(settings, field) == 5.0


@pytest.mark.parametrize("value", [0.0, -1.0, -0.001, math.nan, math.inf, -math.inf])
def test_a_zero_negative_or_non_finite_session_store_connect_timeout_is_refused(value) -> None:
    """`session_store_connect_timeout_s` shares `_finite_positive_timeout`
    with the three MCP timeouts above, for the same reason: it bounds a
    connector dial attempt, and none of these shapes is a sane deadline."""
    with pytest.raises(ValueError, match="must be a finite number greater than zero"):
        Settings(session_store_connect_timeout_s=value)


def test_executor_max_concurrent_steps_reads_from_env(monkeypatch) -> None:
    monkeypatch.setenv("EXECUTOR_MAX_CONCURRENT_STEPS", "2")

    assert Settings().executor_max_concurrent_steps == 2


def test_executor_max_concurrent_steps_below_one_is_refused() -> None:
    with pytest.raises(ValueError, match="greater than or equal to 1"):
        Settings(executor_max_concurrent_steps=0)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (None, "LOW"),
        ("", "LOW"),
        ("  ", "LOW"),
        ("low", "LOW"),
        (" Medium ", "MEDIUM"),
        ("HIGH", "HIGH"),
    ],
)
def test_orchestrator_thinking_level_reads_from_env(monkeypatch, raw, expected) -> None:
    if raw is not None:
        monkeypatch.setenv("ORCHESTRATOR_THINKING_LEVEL", raw)

    assert Settings().orchestrator_thinking_level == expected


@pytest.mark.parametrize("value", ["minimal", "max", "0"])
def test_an_unknown_orchestrator_thinking_level_is_refused(monkeypatch, value) -> None:
    monkeypatch.setenv("ORCHESTRATOR_THINKING_LEVEL", value)

    with pytest.raises(ValueError, match="orchestrator_thinking_level"):
        Settings()
