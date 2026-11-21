"""Shared pydantic-settings base for dfo-ai services."""

from __future__ import annotations

from typing import Self

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class BaseServiceSettings(BaseSettings):
    """Env-driven config shared by every dfo-ai service.

    Subclass per service and add service-specific fields, e.g. `DFO_APIS_URL`
    on data-mcp, or `DATA_MCP_URL` and `ANALYSIS_MCP_URL` on harness. Reads
    from process env vars first, then an optional `.env` file; unknown env
    vars are ignored so services can share one `.env` in local dev.

    `mcp_api_token` and `mcp_auth_disabled` gate every service's MCP mount
    (`common.auth`). Every one of the four services reads the same env vars,
    since the harness co-mounts the other three in-process and the token has
    to agree with whatever `common.auth.apply_gate` resolves first. A blank
    `MCP_API_TOKEN=` in an env file leaves the setting unset, matching
    `harness.settings.Settings.studio_api_token`'s own precedent.

    `frame_rows_in_flight_max` bounds the frame rows this process's tool calls hold in
    memory at once (`common.row_budget`). It lives here so that data-mcp,
    presentation-mcp and the harness, in one harness process, read one value from one
    env var, `FRAME_ROWS_IN_FLIGHT_MAX`. `scripts/bench_frames.py --rows 250000
    --memory-mib <MiB>` prints the implied value:

        (instance memory - idle - headroom) / worst bytes a row

    Worst bytes a row is the largest `delta_mb` / claimed rows among the whole calls this
    process runs, `get_data` and `publish_call`, at 250,000 rows over frames that carry
    each flow party's kind, as `get_data(without_party_totals=True)` serves every flow
    retrieval. On a macOS run, `get_data` added 206.2 MiB over 250,000 rows, 865 bytes a
    row, and `publish_call` 144 MiB. Idle is assumed to be 400 MiB for the harness
    and three servers; the deployed figure is unmeasured. Headroom is 300 MiB, which
    absorbs the freed Arrow pages a long-lived allocator keeps. At 2 GiB the implied
    value is 1,348 MiB / 865 B = 1,634,335 rows.

    The default, 1,100,000 rows, sits below that implied value and admits four
    250,000-row calls at once. Recompute the implied value whenever `max_frame_rows`, the instance's
    memory, the measured idle or the columns a frame carries change; the script exits 1
    when this setting exceeds it. `FrameServiceSettings` refuses a value
    below `max_frame_rows`.
    """

    model_config = SettingsConfigDict(env_file=".env", extra="ignore", case_sensitive=False)

    port: int = 8080
    model_name: str = "gemini-3.8-flash"
    langfuse_public_key: str | None = None
    langfuse_secret_key: str | None = None
    langfuse_base_url: str | None = None
    mcp_api_token: str | None = None
    mcp_auth_disabled: bool = False
    frame_rows_in_flight_max: int = Field(default=1_100_000, ge=1)

    @field_validator("mcp_api_token", mode="before")
    @classmethod
    def _blank_mcp_token_is_unset(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip():
            return None
        return value


class FrameServiceSettings(BaseServiceSettings):
    """The settings data-mcp, analysis-mcp and presentation-mcp share as the services
    that read frames.

    `max_frame_rows` is the row ceiling data-mcp and presentation-mcp enforce: data-mcp
    on a fetched artifact, from its Parquet footer before any row is parsed, and both on
    a frame read back from the store, whose rows they claim at this ceiling when the
    store recorded no count. analysis-mcp inherits the field and reads it nowhere, since
    dfo-apis' analysis runner computes its verbs. It
    matches the durable store's own ceiling
    (`common.materialisation.GcsMaterialisationStore`).

    `frame_rows_in_flight_max` must be at least `max_frame_rows`. Every claim on the
    frame row budget (`common.row_budget`) is one frame of up to `max_frame_rows`
    rows: data-mcp's artifact or cache-hit read, presentation-mcp's publish, and the
    harness's reads of a stored frame. A budget smaller than that refuses such a call
    on every attempt. Settings that break the rule fail at startup.
    """

    max_frame_rows: int = 250_000

    @model_validator(mode="after")
    def _budget_admits_a_full_frame(self) -> Self:
        if self.frame_rows_in_flight_max < self.max_frame_rows:
            raise ValueError(
                f"frame_rows_in_flight_max ({self.frame_rows_in_flight_max}) is below "
                f"max_frame_rows ({self.max_frame_rows}), so a call on a max_frame_rows frame "
                "could never run; lower MAX_FRAME_ROWS or raise FRAME_ROWS_IN_FLIGHT_MAX"
            )
        return self
