import math

import pytest
from data_mcp.settings import Settings
from pydantic import ValidationError

_ENV_VARS = ("PORT", "MODEL_NAME", "DFO_APIS_URL", "DFO_SITE_ORIGIN", "K_SERVICE")

# Every float field that bounds a wait, per `Settings`' own docstring: each
# must reject zero, a negative value, infinity, and NaN. `upstream_backoff_base_s`
# is excluded -- it is a backoff multiplier, not a wait bound, and zero is its own
# legitimate "no backoff" configuration (see its own test below).
_WAIT_BOUND_FIELDS = (
    "upstream_connect_timeout_s",
    "upstream_read_timeout_s",
    "upstream_retry_budget_s",
    "artifact_fetch_budget_s",
    "get_data_deadline_s",
    "operation_poll_interval_s",
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Isolate tests from any of these vars set in the ambient environment."""
    for var in _ENV_VARS:
        monkeypatch.delenv(var, raising=False)


def test_defaults_without_env() -> None:
    settings = Settings()

    assert settings.dfo_apis_url == "http://localhost:8083"
    assert settings.upstream_connect_timeout_s == 5.0
    assert settings.upstream_read_timeout_s == 50.0
    assert settings.upstream_retry_budget_s == 120.0
    assert settings.upstream_max_attempts == 3
    assert settings.upstream_backoff_base_s == 1.0
    # Inherited from BaseServiceSettings, confirming the subclass picks up
    # the shared base fields alongside its own.
    assert settings.port == 8080
    assert settings.model_name == "gemini-3.8-flash"


def test_reads_dfo_apis_url_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DFO_APIS_URL", "https://dfo-apis.internal")

    settings = Settings()

    assert settings.dfo_apis_url == "https://dfo-apis.internal"


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, None),
        ("", None),
        ("https://developmentfinance.org", "https://developmentfinance.org"),
        (" https://developmentfinance.org/ ", "https://developmentfinance.org"),
        ("http://localhost:3000", "http://localhost:3000"),
    ],
)
def test_site_origin_off_cloud_run(
    value: str | None, expected: str | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    if value is not None:
        monkeypatch.setenv("DFO_SITE_ORIGIN", value)

    assert Settings().dfo_site_origin == expected


@pytest.mark.parametrize(
    "bad_value",
    [
        "developmentfinance.org",
        "ftp://developmentfinance.org",
        "https://developmentfinance.org/api",
        "https://developmentfinance.org?x=1",
        "https://user@developmentfinance.org",
        "https://developmentfinance.org:99999",
    ],
)
def test_site_origin_refuses_anything_but_an_origin(
    bad_value: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DFO_SITE_ORIGIN", bad_value)

    with pytest.raises(ValidationError, match="DFO_SITE_ORIGIN"):
        Settings()


def test_site_origin_is_required_on_cloud_run(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("K_SERVICE", "dfo-dev-ai")

    with pytest.raises(ValidationError, match="DFO_SITE_ORIGIN"):
        Settings()

    monkeypatch.setenv("DFO_SITE_ORIGIN", "https://developmentfinance.org")

    assert Settings().dfo_site_origin == "https://developmentfinance.org"


def test_still_reads_shared_base_fields_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PORT", "9090")

    settings = Settings()

    assert settings.port == 9090
    assert settings.dfo_apis_url == "http://localhost:8083"


# Wait-bound float fields -- finite and positive


@pytest.mark.parametrize("field", _WAIT_BOUND_FIELDS)
@pytest.mark.parametrize("bad_value", ["0", "-1", "inf", "nan"])
def test_wait_bound_field_rejects_zero_negative_or_non_finite(
    field: str, bad_value: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(field.upper(), bad_value)

    with pytest.raises(ValidationError):
        Settings()


@pytest.mark.parametrize("bad_value", ["-1", "inf", "nan"])
def test_backoff_base_rejects_negative_or_non_finite(
    bad_value: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("UPSTREAM_BACKOFF_BASE_S", bad_value)

    with pytest.raises(ValidationError):
        Settings()


def test_backoff_base_accepts_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("UPSTREAM_BACKOFF_BASE_S", "0")

    assert Settings().upstream_backoff_base_s == 0.0


def test_operation_poll_interval_floors_at_half_a_second(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPERATION_POLL_INTERVAL_S", "0.5")
    assert Settings().operation_poll_interval_s == 0.5

    monkeypatch.setenv("OPERATION_POLL_INTERVAL_S", "0.4")
    with pytest.raises(ValidationError):
        Settings()


def test_get_data_deadline_caps_at_170s(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GET_DATA_DEADLINE_S", "170")
    assert Settings().get_data_deadline_s == 170.0

    monkeypatch.setenv("GET_DATA_DEADLINE_S", "170.1")
    with pytest.raises(ValidationError):
        Settings()


def test_get_data_deadline_must_leave_the_direct_budget_time_to_mint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The direct budget's pre-mint work must finish with 50s left, so a deadline of
    50s or less would refuse every call before it minted."""
    monkeypatch.setenv("GET_DATA_DEADLINE_S", "50")
    with pytest.raises(ValidationError):
        Settings()

    monkeypatch.setenv("GET_DATA_DEADLINE_S", "168")
    assert Settings().get_data_deadline_s == 168.0


def test_external_mint_concurrency_defaults_to_one_and_refuses_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two harness instances at the default hold at most two of apis' four mint
    slots for outside callers."""
    monkeypatch.delenv("EXTERNAL_MINT_CONCURRENCY", raising=False)
    assert Settings().external_mint_concurrency == 1

    monkeypatch.setenv("EXTERNAL_MINT_CONCURRENCY", "0")
    with pytest.raises(ValidationError):
        Settings()


def test_default_get_data_deadline_leaves_headroom_under_the_harness_call_timeout() -> None:
    """`get_data_deadline_s`'s default must clear its own 170s cap, itself under the
    harness's 180s `call_timeout_s` for `get_data`
    (`services/harness/src/compiler/capabilities.py`)."""
    settings = Settings()

    assert 0 < settings.get_data_deadline_s <= 170.0
    assert math.isfinite(settings.get_data_deadline_s)
