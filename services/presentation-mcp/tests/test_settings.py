"""Settings tests: env-driven config over the shared base fields."""

from __future__ import annotations

import pytest
from presentation_mcp.settings import Settings
from pydantic import ValidationError

_PUBLIC_URL = "http://localhost:8080/mcp/presentation"
_ENV_VARS = (
    "PORT",
    "MODEL_NAME",
    "DFO_APIS_URL",
    "DATASETS_FIRST_PARTY_PRINCIPAL",
    "PRESENTATION_PUBLIC_URL",
    "PRESENTATION_RELEASE_REQUIRED",
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Isolate tests from any of these vars set in the ambient environment,
    then set the one variable `Settings` cannot be built without."""
    for var in _ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("PRESENTATION_PUBLIC_URL", _PUBLIC_URL)


def test_defaults_without_env() -> None:
    settings = Settings()

    assert settings.port == 8080
    assert settings.model_name == "gemini-3.8-flash"
    assert settings.dfo_apis_url == "http://localhost:8083"
    assert settings.upstream_connect_timeout_s == 5.0
    assert settings.upstream_read_timeout_s == 50.0
    assert settings.upstream_retry_budget_s == 120.0
    assert settings.upstream_max_attempts == 3
    assert settings.upstream_backoff_base_s == 1.0
    assert settings.max_schema_version == 1
    assert settings.presentation_release_required is True
    assert settings.release_retry_base_s == 1.0
    assert settings.release_retry_max_s == 60.0
    assert settings.release_startup_deadline_s == 90.0
    assert settings.datasets_first_party_principal is None
    assert settings.presentation_public_url == _PUBLIC_URL


def test_reads_datasets_first_party_principal_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATASETS_FIRST_PARTY_PRINCIPAL", "presentation-mcp")

    settings = Settings()

    assert settings.datasets_first_party_principal == "presentation-mcp"


def test_blank_datasets_first_party_principal_is_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATASETS_FIRST_PARTY_PRINCIPAL", "   ")

    settings = Settings()

    assert settings.datasets_first_party_principal is None


def test_reads_port_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PORT", "9090")

    settings = Settings()

    assert settings.port == 9090


def test_reads_dfo_apis_url_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DFO_APIS_URL", "https://dfo-apis.internal")

    settings = Settings()

    assert settings.dfo_apis_url == "https://dfo-apis.internal"


def test_reads_presentation_public_url_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PRESENTATION_PUBLIC_URL", "https://ai.example.org/mcp/presentation")

    settings = Settings()

    assert settings.presentation_public_url == "https://ai.example.org/mcp/presentation"


def test_presentation_public_url_is_stripped(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PRESENTATION_PUBLIC_URL", "  https://ai.example.org/mcp/presentation \n")

    assert Settings().presentation_public_url == "https://ai.example.org/mcp/presentation"


@pytest.mark.parametrize("value", [None, "", "   "], ids=["unset", "empty", "blank"])
def test_a_missing_presentation_public_url_fails_naming_the_variable(
    monkeypatch: pytest.MonkeyPatch, value: str | None
) -> None:
    if value is None:
        monkeypatch.delenv("PRESENTATION_PUBLIC_URL")
    else:
        monkeypatch.setenv("PRESENTATION_PUBLIC_URL", value)

    with pytest.raises(ValidationError) as failure:
        Settings()

    message = str(failure.value)
    assert "PRESENTATION_PUBLIC_URL" in message
    assert "it is not set" in message
    assert "https://<host>/mcp/presentation" in message
    assert "dfo-infra" in message


@pytest.mark.parametrize(
    "value",
    [
        "localhost:8080/mcp/presentation",
        "/mcp/presentation",
        "ftp://ai.example.org/mcp/presentation",
        "https://",
        "https:///mcp/presentation",
        "ai.example.org",
        "https://ai.example.org/mcp/presentation?x=1",
        "https://ai.example.org/mcp/presentation?",
        "https://ai.example.org/mcp/presentation#frag",
        "https://ai.example.org/mcp/presentation#",
        "https://user:pass@ai.example.org/mcp/presentation",
        "https://user@ai.example.org/mcp/presentation",
        "https://@ai.example.org/mcp/presentation",
        "https://ai.example.org:port/mcp/presentation",
        "https://a b.com/mcp/presentation",
        "https://a;b.com/mcp/presentation",
        "https://h.com'/mcp/presentation",
        "https://*.h.com/mcp/presentation",
        "https://h_h.com/mcp/presentation",
        "https://h.com:0/mcp/presentation",
        "https://h.com:65536/mcp/presentation",
        "https://[fe80::1%25eth0]/mcp/presentation",
        "https://[not-ipv6]/mcp/presentation",
        "https://ai.exa\tmple.org/mcp/presentation",
        "https://ai.example.org/mcp\n/presentation",
    ],
)
def test_a_malformed_presentation_public_url_fails_naming_the_variable_and_the_value(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setenv("PRESENTATION_PUBLIC_URL", value)

    with pytest.raises(ValidationError) as failure:
        Settings()

    message = str(failure.value)
    assert "PRESENTATION_PUBLIC_URL" in message
    assert f"got {value!r}" in message
    assert "https://<host>/mcp/presentation" in message
    assert "dfo-infra" in message


@pytest.mark.parametrize(
    "value",
    [
        "http://localhost:8080/mcp/presentation",
        "https://ai.example.org/mcp/presentation",
        "https://ai-1.example.org:65535/mcp/presentation",
        "https://h.com:1/mcp/presentation",
        "http://127.0.0.1:8090/mcp/presentation",
        "http://[::1]:8090/mcp/presentation",
    ],
)
def test_a_well_formed_presentation_public_url_is_accepted(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setenv("PRESENTATION_PUBLIC_URL", value)

    assert Settings().presentation_public_url == value


def test_a_presentation_public_url_passed_as_an_argument_is_validated_too() -> None:
    with pytest.raises(ValidationError, match="PRESENTATION_PUBLIC_URL"):
        Settings(presentation_public_url="not a url")
