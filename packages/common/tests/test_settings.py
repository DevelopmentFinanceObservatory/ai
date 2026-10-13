import pytest
from common.settings import BaseServiceSettings, FrameServiceSettings

_BASE_FIELDS = (
    "PORT",
    "MODEL_NAME",
    "LANGFUSE_PUBLIC_KEY",
    "LANGFUSE_SECRET_KEY",
    "LANGFUSE_BASE_URL",
    "MCP_API_TOKEN",
    "MCP_AUTH_DISABLED",
    "FRAME_ROWS_IN_FLIGHT_MAX",
    "MAX_FRAME_ROWS",
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Isolate tests from any of these vars set in the ambient environment."""
    for var in _BASE_FIELDS:
        monkeypatch.delenv(var, raising=False)


def test_defaults_without_env() -> None:
    settings = BaseServiceSettings()

    assert settings.port == 8080
    assert settings.model_name == "gemini-3.8-flash"
    assert settings.langfuse_public_key is None
    assert settings.mcp_api_token is None
    assert settings.mcp_auth_disabled is False


def test_a_blank_mcp_api_token_is_unset(monkeypatch) -> None:
    monkeypatch.setenv("MCP_API_TOKEN", "")
    assert BaseServiceSettings().mcp_api_token is None
    monkeypatch.setenv("MCP_API_TOKEN", "s3cret")
    assert BaseServiceSettings().mcp_api_token == "s3cret"


def test_mcp_auth_disabled_reads_from_env(monkeypatch) -> None:
    monkeypatch.setenv("MCP_AUTH_DISABLED", "true")
    assert BaseServiceSettings().mcp_auth_disabled is True


def test_reads_from_env(monkeypatch) -> None:
    monkeypatch.setenv("PORT", "9090")
    monkeypatch.setenv("MODEL_NAME", "gemini-pro")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")

    settings = BaseServiceSettings()

    assert settings.port == 9090
    assert settings.model_name == "gemini-pro"
    assert settings.langfuse_public_key == "pk-test"


def test_unknown_env_vars_are_ignored(monkeypatch) -> None:
    monkeypatch.setenv("DFO_APIS_URL", "https://example.org")

    settings = BaseServiceSettings()

    assert not hasattr(settings, "dfo_apis_url")


def test_subclass_adds_service_specific_fields(monkeypatch) -> None:
    class DataMcpSettings(BaseServiceSettings):
        dfo_apis_url: str = "http://localhost:8000"

    monkeypatch.setenv("DFO_APIS_URL", "https://search.internal")

    settings = DataMcpSettings()

    assert settings.dfo_apis_url == "https://search.internal"
    assert settings.port == 8080


def test_frame_rows_in_flight_max_defaults_to_the_two_gib_derivation() -> None:
    assert BaseServiceSettings().frame_rows_in_flight_max == 1_100_000


def test_frame_rows_in_flight_max_refuses_a_budget_below_one_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FRAME_ROWS_IN_FLIGHT_MAX", "0")

    with pytest.raises(ValueError, match="frame_rows_in_flight_max"):
        BaseServiceSettings()


def test_the_default_budget_admits_a_two_input_verb_at_the_default_ceiling() -> None:
    settings = FrameServiceSettings()

    assert settings.frame_rows_in_flight_max >= 2 * settings.max_frame_rows


def test_a_budget_of_exactly_two_inputs_at_the_ceiling_is_accepted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MAX_FRAME_ROWS", "300000")
    monkeypatch.setenv("FRAME_ROWS_IN_FLIGHT_MAX", "600000")

    assert FrameServiceSettings().frame_rows_in_flight_max == 600_000


def test_a_budget_below_two_inputs_at_the_ceiling_fails_naming_both_fixes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MAX_FRAME_ROWS", "300000")
    monkeypatch.setenv("FRAME_ROWS_IN_FLIGHT_MAX", "599999")

    with pytest.raises(ValueError, match=r"lower MAX_FRAME_ROWS or raise FRAME_ROWS_IN_FLIGHT_MAX"):
        FrameServiceSettings()
