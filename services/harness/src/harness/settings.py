"""Env-driven settings for the harness service."""

from __future__ import annotations

import math
from typing import Annotated, Literal
from urllib.parse import urlsplit

from common.mcp_telemetry import is_loopback_url
from common.settings import BaseServiceSettings
from pydantic import Field, ValidationInfo, field_validator, model_validator
from pydantic_settings import NoDecode

# Mount prefixes for the data-mcp / analysis-mcp / presentation-mcp ASGI
# apps that `harness.server.create_app` mounts in-process: the four
# workspace services deploy as one Cloud Run instance (one warm instance,
# no cold starts on agent tool calls, loopback instead of network hops), so
# these live here -- rather than duplicated as string literals in both
# `harness.server` and the defaults below -- to keep the mount points and
# the loopback URL defaults from drifting apart.
DATA_MCP_MOUNT_PATH = "/mcp/data"
ANALYSIS_MCP_MOUNT_PATH = "/mcp/analysis"
PRESENTATION_MCP_MOUNT_PATH = "/mcp/presentation"
# The combined server (`harness.combined_mcp`) that outside MCP clients call.
# Its MCP endpoint is the prefix itself, `/mcp/`, and it is mounted after the
# three above so their longer prefixes match first.
COMBINED_MCP_MOUNT_PATH = "/mcp"

# fastmcp's `FastMCP.http_app()` serves the MCP protocol at this path suffix
# by default (`fastmcp.settings.streamable_http_path`); the mounted apps
# don't override it, so the full in-container path is the mount prefix plus
# this suffix, e.g. "/mcp/data/mcp".
_MCP_PROTOCOL_SUFFIX = "/mcp"

# The Gemini thinking levels the orchestrator's reasoning effort can be set to.
type ThinkingLevelName = Literal["LOW", "MEDIUM", "HIGH"]

# The orchestrator runs at `LOW`: in an 18-turn eval per arm, `LOW` halved the
# median turn time (79.9 s to 38.0 s) with answer quality inside the spread
# between two default-level runs.
DEFAULT_ORCHESTRATOR_THINKING_LEVEL: ThinkingLevelName = "LOW"

# How long after its deadline `harness.live_refresh` cancels a refresh, and how
# long the website waits for one. A refresh answers by its deadline plus the
# grace, so the largest deadline accepted is the caller's timeout less the grace.
LIVE_REFRESH_HARD_STOP_GRACE_S = 8.0
LIVE_REFRESH_CALLER_TIMEOUT_S = 60.0
LIVE_REFRESH_DEADLINE_MAX_S = LIVE_REFRESH_CALLER_TIMEOUT_S - LIVE_REFRESH_HARD_STOP_GRACE_S


class Settings(BaseServiceSettings):
    """Config for the harness agentic-loop service.

    `data_mcp_url`, `analysis_mcp_url`, and `presentation_mcp_url` point at
    the three mounted MCP servers over streamable HTTP (`harness.mcp_client`).
    Left unset, all three default to the loopback path of the FastMCP
    apps mounted in-process by `harness.server.create_app`, derived from
    this service's own `port`. Setting `DATA_MCP_URL` / `ANALYSIS_MCP_URL`
    / `PRESENTATION_MCP_URL` always wins over that default -- that env
    override is the seam for a future split back into standalone services
    (point it at the split-out service's URL instead of the loopback path).
    `database_url` and the `session_store_*` fields together select ADK's
    session backend (`harness.session_store.build_session_service`):
    in-memory when none of them is set, `DatabaseSessionService` over a
    plain SQLAlchemy URL when `database_url` is set, or
    `DatabaseSessionService` over an async engine built on the Cloud SQL
    Python Connector (PSC, IAM auth) when
    `session_store_instance_connection_name` is set. Setting both
    `database_url` and `session_store_instance_connection_name` is a
    validation error, and so is setting any one or two of
    `session_store_instance_connection_name`, `session_store_database` and
    `session_store_runtime_db_user` without the rest. The connector path
    needs all three or none. `session_store_pool_size` and
    `session_store_max_overflow` size the pooled connections
    `create_async_engine` opens for that engine. The defaults (2 and 0) sum
    to two per instance. A rollout can leave an old revision still draining
    SSE streams running alongside the new one, so up to four instances hold
    connections at once, and four times two stays within the ten
    connections dfo-infra's platform Postgres instance reserves for it.
    `session_store_connect_timeout_s` bounds one connector dial attempt,
    matching dfo-apis' `DATASET_STORE_CONNECT_TIMEOUT` default. Every one of
    these five `session_store_*` fields, and `database_url` itself,
    collapses a blank value to unset, like `studio_api_token` below.

    `mcp_api_token` and `mcp_auth_disabled` (`common.settings.BaseServiceSettings`)
    gate the three mounts (`common.auth`) and are also the credential this
    service's own MCP clients present when calling them (`harness.mcp_client`).
    With `mcp_api_token` unset and `mcp_auth_disabled` false, construction
    refuses if any of the three URLs above has been overridden off loopback.
    A per-process minted token can never be known to a remote mount, so this
    refusal catches a split deployment that would otherwise 401 on every
    call at runtime instead of failing at boot. With `mcp_api_token` set,
    construction instead refuses an off-loopback override whose scheme
    isn't `https`: `harness.mcp_client` attaches that shared bearer to every
    request, and cleartext HTTP would put it on the wire in the open.

    `mcp_call_timeout_s`, `mcp_list_timeout_s` and `mcp_init_timeout_s` feed
    `harness.mcp_deadlines.McpTimeoutPolicy`, the one place that turns them
    (plus `compiler.capabilities.call_timeouts()`'s per-tool overrides) into
    the deadlines every FastMCP `Client` this service constructs carries:
    `compiler.surface.McpSurface` (`harness.agents.build_tool_surface`, the
    translator/compiler's discovery surface) and `harness.mcp_client`'s
    `McpResolver` / `McpExecutor` (the served request path) alike.
    `mcp_call_timeout_s` bounds a tool call a capability row's own
    `call_timeout_s` does not override; `mcp_list_timeout_s` bounds a
    `list_tools` read; `mcp_init_timeout_s` bounds a client's handshake.
    Each must be a finite number greater than zero -- FastMCP normalises a
    zero `init_timeout` to `None`, restoring the unbounded wait the setting
    exists to prevent, and a negative or non-finite value has no sane
    meaning as a deadline either.

    `studio_prefix_cache` turns on the shared explicit cache of each agent's
    static prompt prefix (`harness.prefix_cache`). The planner's and the
    orchestrator's system instruction and tool declarations are cached once
    per process and read from the cache on every call, in every
    conversation. Off, each call sends its whole prompt and Gemini's implicit
    caching serves whatever prefix it matches.

    `combined_mcp_requests_per_s` and `combined_mcp_burst` bound what outside
    clients can ask of the combined server at `/mcp/` (`harness.combined_mcp`).
    They size one token bucket per instance, refilled at `combined_mcp_requests_per_s`
    and holding at most `combined_mcp_burst`, that every tool call to `/mcp/`
    draws on. Initialize, `tools/list`, resource reads and pings do not draw
    on it. It is one bucket for every caller, a service-wide ceiling on spend.
    A holder of `MCP_API_TOKEN` can name any caller, so a bucket per caller
    would let one client pass the limit by changing a header. Every caller
    therefore shares, and can exhaust, one limit. A tool call over it refuses `rate_limited`. `search` spends a Gemini
    call in dfo-apis on every request, which is what the bucket protects. The
    dfo-apis mint slots outside `get_data` calls can hold are capped in
    data-mcp, per instance, by `EXTERNAL_MINT_CONCURRENCY`
    (`data_mcp.settings.Settings.external_mint_concurrency`), which counts a
    mint until dfo-apis has stopped running it. None of these limits applies
    to the per-server mounts.

    `combined_mcp_max_in_flight` caps the tool calls and resource reads the
    combined server runs at once, per instance. The token bucket limits how
    often calls start, and each call can run for up to 45 s
    (`data_mcp.deadline.EXTERNAL_CALL_LIMIT_S`), so without this cap a burst
    of slow calls piles up on one instance, each holding an upstream
    connection and memory. A call over the cap refuses `rate_limited` at
    once rather than queueing, since a queued call would spend its own 45 s
    waiting. The cap is per instance because nothing shares a count across
    instances, so n instances run up to n times it.

    `live_refresh_principal` (`LIVE_REFRESH_PRINCIPAL`) is the bearer
    `POST /api/studio/refresh` takes (`harness.live_refresh`), and the principal
    this service presents to dfo-apis' frame refresh and render routes. Unset or
    blank, the route refuses every caller. It must differ from
    `studio_api_token`: the studio routes refuse it, so one value for both would
    close them.

    `live_refresh_deadline_s` is how long one refresh has before its replay
    starts no new mint, verb, or publish. The run is cancelled
    `LIVE_REFRESH_HARD_STOP_GRACE_S` (8 s) later and answers `pending`, so no
    request runs past 53 s at the default of 45. The website gives the call
    `LIVE_REFRESH_CALLER_TIMEOUT_S` (60 s), which leaves it 7 s to receive that
    answer. A value above `LIVE_REFRESH_DEADLINE_MAX_S` (52, the caller's timeout
    less the grace) is refused: its hard stop would fall after the website had
    given up, so the `pending` answer would reach nobody. A root mint needs about
    25 s of the deadline left to start (`data_mcp.deadline.EXTERNAL_CALL_BUDGET`),
    and the reads before the replay take a few seconds, so a value under 30 never
    mints a root and is refused.

    `live_refresh_requests_per_s`, `live_refresh_burst`, and
    `live_refresh_max_in_flight` are the refresh route's own token bucket and
    in-flight cap, per instance, apart from the combined server's, so neither
    kind of traffic uses up the other's allowance. A request over either answers
    `pending` with a wait. The burst of 20 is one live canvas, which holds at
    most 20 data cards and asks for all of them as it opens. The rate of 2 a
    second is under half the combined server's 5: the website asks about a card
    once in 300 s whatever the traffic, so only a `pending` poll repeats sooner.
    The cap of 8 is half the combined server's 16, since a refresh that replays
    holds its slot for up to 53 s against a tool call's 45.

    `live_refresh_rows_in_flight_max` sizes a second `common.row_budget.RowBudget`
    that `harness.recipe_replay.replay_recipe` claims from: the input rows of
    each verb call, the root rows of an anchor call, and the published frame's
    rows. It caps the rows all replays in this process put through the tools at
    once. The tools and the publish also claim the same rows from the
    process-wide budget (`frame_rows_in_flight_max`), and a root mint claims
    there only. The default of 500,000 is one two-input verb on two full frames,
    the largest single claim, and leaves 600,000 of the process-wide 1,100,000
    to Studio conversations and outside MCP callers. A claim the rows in flight
    leave no room for, in either budget, answers `pending`. A replay step whose
    rows exceed this whole budget is refused `rows_over_refresh_budget`.

    The `turn_*` and `model_first_chunk_s` fields bound one request (one POST
    to `/api/studio/turn` or `/api/studio/interrupt`) through
    `harness.turn_budget.TurnBudget`, which every orchestrator and planner
    model call reads (`harness.turn_budget.BoundedGemini`):

    - `turn_model_time_s` (90) meters model time: the seconds the request's
      model calls spend waiting on their streams, from the moment each request
      is sent, genai's own HTTP retries included, to each chunk in turn. The
      prefix cache's create does not count, and neither do the tools ADK runs
      while a call waits at a response it has delivered. It ends 1 of 187 simple and 3 of 182 plan requests
      measured on dev. It cuts a call that is still streaming.
    - `turn_model_calls_max` (30) counts model calls, a first-chunk retry not
      included. Dev's most in one request was 61, local runs' 29.
    - `turn_wall_s` (600) meters wall-clock time from the request's start, tool
      time included, below Cloud Run's 900 s request limit. It cuts only model
      calls and the discovery calls inside a planner entry, which may take
      what is left of it (`planner.plan.discovery_time_left`). Other tool
      calls keep their own timeouts, and the next model call finds the bound
      spent.
    - `model_first_chunk_s` (12) is how long a streamed call may wait for its
      first chunk, of any kind, before it is resent once. Only the first call
      in a request to miss it is resent, and only before any chunk reached
      the caller. 0.61% of dev calls waited longer.

    A spent budget, cap or wall bound ends the request with its sentence
    (`harness.sentences.SEAM_SENTENCES`) as the final message, with no further
    model call, unless it closes a planner call while the last referral names
    sources to choose from: the planner then asks the user to pick one
    (`harness.planner_seam.PlannerSeam._after_planner_model`). `harness.graph.state.BUDGETS` counts tool calls apart from these.
    The Edit Card's one model call (`harness.generative.GeminiEdit`) carries
    `turn_model_time_s` as its HTTP timeout.
    """

    data_mcp_url: str = ""
    analysis_mcp_url: str = ""
    presentation_mcp_url: str = ""
    database_url: str | None = None
    # Presence selects the Cloud SQL Connector path over `database_url`'s
    # plain-URL path (see the class docstring). `project:region:instance`.
    session_store_instance_connection_name: str | None = None
    # Required together with the connection name above; the connector path
    # needs all three of these fields or none of them.
    session_store_database: str | None = None
    session_store_runtime_db_user: str | None = None
    # Pooled connections `create_async_engine` opens for the connector-backed
    # engine (see the class docstring for the budget these defaults sum to).
    session_store_pool_size: int = 2
    session_store_max_overflow: int = 0
    # Bounds one Cloud SQL Connector dial attempt, matching dfo-apis'
    # DATASET_STORE_CONNECT_TIMEOUT default.
    session_store_connect_timeout_s: float = 30.0
    mcp_call_timeout_s: float = 60.0
    mcp_list_timeout_s: float = 15.0
    mcp_init_timeout_s: float = 10.0
    # The dev-time shared secret for the studio bearer check (`harness.auth`).
    # Unset means any non-empty token is accepted (the token stays required
    # either way -- contract delta 5).
    studio_api_token: str | None = None
    # The orchestrator's reasoning effort (`ORCHESTRATOR_THINKING_LEVEL`),
    # read case-insensitively. Unset or blank means the default.
    orchestrator_thinking_level: ThinkingLevelName = DEFAULT_ORCHESTRATOR_THINKING_LEVEL
    # Browser origins the server answers CORS preflights for, as a
    # comma-separated env value (a locally served page testing against the
    # real agent on another port). Empty sends no CORS headers: the deployed
    # service is reached same-origin through the website's proxy.
    studio_cors_origins: Annotated[list[str], NoDecode] = Field(default_factory=list)
    # The shared cache of the agents' static prompt prefixes; see the class
    # docstring.
    studio_prefix_cache: bool = True
    # How many of one plan's wave's tool calls `executor.run._Run` may hold
    # in flight against the MCP servers at once (`harness.graph.workflow.
    # create_root_workflow(max_concurrent_steps=...)`); a wave wider than
    # this queues at the point of call rather than opening every call at once.
    executor_max_concurrent_steps: int = Field(default=4, ge=1)
    # The combined server's limits for outside callers; see the class docstring.
    combined_mcp_requests_per_s: float = Field(default=5.0, gt=0, allow_inf_nan=False)
    combined_mcp_burst: int = Field(default=20, ge=1)
    combined_mcp_max_in_flight: int = Field(default=16, ge=1)
    # The refresh route's credential, deadline, and limits; see the class docstring.
    live_refresh_principal: str | None = None
    live_refresh_deadline_s: float = Field(
        default=45.0, ge=30.0, le=LIVE_REFRESH_DEADLINE_MAX_S, allow_inf_nan=False
    )
    live_refresh_requests_per_s: float = Field(default=2.0, gt=0, allow_inf_nan=False)
    live_refresh_burst: int = Field(default=20, ge=1)
    live_refresh_max_in_flight: int = Field(default=8, ge=1)
    live_refresh_rows_in_flight_max: int = Field(default=500_000, ge=1)
    # One request's model-call bounds; see the class docstring.
    turn_model_time_s: float = Field(default=90.0, gt=0, allow_inf_nan=False)
    turn_model_calls_max: int = Field(default=30, ge=1)
    turn_wall_s: float = Field(default=600.0, gt=0, allow_inf_nan=False)
    model_first_chunk_s: float = Field(default=12.0, gt=0, allow_inf_nan=False)

    @field_validator(
        "mcp_call_timeout_s",
        "mcp_list_timeout_s",
        "mcp_init_timeout_s",
        "session_store_connect_timeout_s",
    )
    @classmethod
    def _finite_positive_timeout(cls, value: float, info: ValidationInfo) -> float:
        if not math.isfinite(value) or value <= 0:
            raise ValueError(
                f"{info.field_name} must be a finite number greater than zero, got {value!r}. "
                "A zero, negative, or non-finite deadline either has no sane meaning or is "
                "FastMCP's own signal for an unbounded wait -- exactly what these settings "
                "exist to rule out."
            )
        return value

    @field_validator("studio_api_token", "live_refresh_principal", mode="before")
    @classmethod
    def _blank_token_is_unset(cls, value: object) -> object:
        # A blank `STUDIO_API_TOKEN=` or `LIVE_REFRESH_PRINCIPAL=` in an env file
        # leaves the setting unset.
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @field_validator("orchestrator_thinking_level", mode="before")
    @classmethod
    def _normalise_thinking_level(cls, value: object) -> object:
        # `low`, ` Low ` and `LOW` all name one level; a blank value is the default.
        if isinstance(value, str):
            return value.strip().upper() or DEFAULT_ORCHESTRATOR_THINKING_LEVEL
        return value

    @field_validator(
        "database_url",
        "session_store_instance_connection_name",
        "session_store_database",
        "session_store_runtime_db_user",
        mode="before",
    )
    @classmethod
    def _blank_session_backend_field_is_unset(cls, value: object) -> object:
        # A blank `DATABASE_URL=` / `SESSION_STORE_*=` in an env file leaves
        # the field unset, matching `_blank_token_is_unset` above.
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @field_validator("studio_cors_origins", mode="before")
    @classmethod
    def _split_list(cls, value: object) -> object:
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return value

    @model_validator(mode="after")
    def _refuse_one_value_for_both_studio_credentials(self) -> Settings:
        if (
            self.live_refresh_principal is not None
            and self.live_refresh_principal == self.studio_api_token
        ):
            raise ValueError(
                "LIVE_REFRESH_PRINCIPAL and STUDIO_API_TOKEN hold the same value. The "
                "studio routes refuse the live-refresh principal (harness.auth), so "
                "they would refuse every caller: set two different secrets."
            )
        return self

    @model_validator(mode="after")
    def _default_mcp_urls_to_loopback(self) -> Settings:
        if not self.data_mcp_url:
            self.data_mcp_url = (
                f"http://127.0.0.1:{self.port}{DATA_MCP_MOUNT_PATH}{_MCP_PROTOCOL_SUFFIX}"
            )
        if not self.analysis_mcp_url:
            self.analysis_mcp_url = (
                f"http://127.0.0.1:{self.port}{ANALYSIS_MCP_MOUNT_PATH}{_MCP_PROTOCOL_SUFFIX}"
            )
        if not self.presentation_mcp_url:
            self.presentation_mcp_url = (
                f"http://127.0.0.1:{self.port}{PRESENTATION_MCP_MOUNT_PATH}{_MCP_PROTOCOL_SUFFIX}"
            )
        return self

    @model_validator(mode="after")
    def _refuse_off_loopback_mounts_without_a_token(self) -> Settings:
        # Runs after `_default_mcp_urls_to_loopback` (declaration order), so
        # an unset URL is already the loopback default here and never trips
        # this check. Only an explicit `DATA_MCP_URL` / `ANALYSIS_MCP_URL` /
        # `PRESENTATION_MCP_URL` override pointing elsewhere can trip it.
        #
        # `MCP_AUTH_DISABLED=true` exempts both rules below: no bearer is
        # ever attached (`harness.mcp_client`), so neither an unknowable token
        # nor a cleartext one is a risk -- that's the "deliberately open dev
        # server" case the first message itself points callers at.
        if self.mcp_auth_disabled:
            return self
        off_loopback = [
            (name, urlsplit(url))
            for name, url in (
                ("DATA_MCP_URL", self.data_mcp_url),
                ("ANALYSIS_MCP_URL", self.analysis_mcp_url),
                ("PRESENTATION_MCP_URL", self.presentation_mcp_url),
            )
            if not is_loopback_url(url)
        ]
        if not off_loopback:
            return self
        if self.mcp_api_token is None:
            names = ", ".join(name for name, _ in off_loopback)
            raise ValueError(
                f"{names} points off loopback with MCP_API_TOKEN unset. "
                "A per-process minted token can't be known to a remote mount: set "
                "MCP_API_TOKEN, or MCP_AUTH_DISABLED=true for a deliberately open dev "
                "server, before pointing these at a split-out deployment."
            )
        cleartext = [name for name, parts in off_loopback if parts.scheme != "https"]
        if cleartext:
            names = ", ".join(cleartext)
            raise ValueError(
                f"{names} points off loopback over a non-https scheme with "
                "MCP_API_TOKEN set. The shared bearer is attached to every request "
                "(harness.mcp_client) and would go out in cleartext: use https for these "
                "URLs before pointing them at a split-out deployment."
            )
        return self

    @model_validator(mode="after")
    def _validate_session_store_backend(self) -> Settings:
        if (
            self.database_url is not None
            and self.session_store_instance_connection_name is not None
        ):
            raise ValueError(
                "DATABASE_URL and SESSION_STORE_INSTANCE_CONNECTION_NAME are both set. "
                "They select two different session backends "
                "(harness.session_store.build_session_service) -- unset one of them."
            )
        # All three fields select the connector path or none of them do. A
        # partial combination (e.g. the database and user set without the
        # connection name) boots the in-memory backend silently instead of
        # the connector the caller meant to configure.
        connector_fields = (
            (
                "SESSION_STORE_INSTANCE_CONNECTION_NAME",
                self.session_store_instance_connection_name,
            ),
            ("SESSION_STORE_DATABASE", self.session_store_database),
            ("SESSION_STORE_RUNTIME_DB_USER", self.session_store_runtime_db_user),
        )
        present = [name for name, value in connector_fields if value is not None]
        missing = [name for name, value in connector_fields if value is None]
        if present and missing:
            raise ValueError(
                f"{' and '.join(present)} set without {' and '.join(missing)}. The Cloud "
                "SQL Connector path needs all three or none."
            )
        return self
