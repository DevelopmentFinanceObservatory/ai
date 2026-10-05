"""Environment-driven settings for data-mcp."""

from __future__ import annotations

from common.settings import FrameServiceSettings
from pydantic import Field

from data_mcp.deadline import DIRECT_CALL_BUDGET
from data_mcp.definition_store.caching import DEFAULT_MAX_ENTRIES


class Settings(FrameServiceSettings):
    """Configuration for data-mcp, combining the shared base fields with the dfo-apis upstream settings it wraps.

    `dfo_apis_url` is the single base URL of dfo-apis, read from
    `DFO_APIS_URL` per BaseServiceSettings' case-insensitive env matching.
    dfo-apis is one FastAPI app serving both `/api/search/*` (always
    mounted) and `/api/datasets/*` (gated by dfo-apis' own
    `DFO_DATASETS_SURFACE` flag, default off) on the same host, whether
    that is port 8083 in local dev or one internal Cloud Run URL in
    deployment, so one field covers both. The `SEARCH_PREFIX` /
    `DATASETS_PREFIX` split happens at the path level in `client.py`, not
    here.

    The `upstream_*` fields bound `UpstreamClient`'s retry loop.
    `upstream_read_timeout_s` clears dfo-apis' 45s search safety net
    (`SEARCH_REQUEST_TIMEOUT_S`) with headroom, and `upstream_retry_budget_s`
    is the wall-clock ceiling for *all* attempts of one call rather than a
    per-attempt timeout.

    `definition_store_bucket` unset means `server.py`'s lifespan builds an
    in-memory definition store, mirroring `harness.settings.database_url`'s
    unset-means-in-memory default. Set it to back the store with GCS instead.
    `definition_store_cache_max_entries` bounds the read cache `server.py`
    wraps either store in; the class default matches
    `definition_store.caching.DEFAULT_MAX_ENTRIES`.

    `max_frame_rows` comes from `common.FrameServiceSettings`, shared with
    analysis-mcp and presentation-mcp: this service bounds a fetched artifact's parse
    with it, from Parquet metadata alone, before any row is materialised, and the
    store read of a pair already held. `artifact_fetch_budget_s` caps the whole post-mint fetch step in
    `frame_materialisation.py` (every signed-URL re-sign and every GET against it); the
    call deadline below can cut it shorter.

    `frame_store_bucket` names the same GCS bucket `analysis_mcp.settings.Settings`'s
    own field of the same name points at, since this service never seeds fixtures
    into the frame store and only reads and writes through `server.py`'s lifespan
    and `frame_materialisation.py`. `None` (the default) leaves this service
    relying on being mounted alongside analysis-mcp, or on the frame-store holder's
    own lazy in-memory default when it is not. Do not set this on a deployment
    before the bucket's own lifecycle rule exists: that rule bounds retention, is
    provisioned in `dfo-infra`, not here, and until it exists a durable store's
    objects accumulate indefinitely.

    `frame_store_verify_content_version` names the same env var
    `analysis_mcp.settings.Settings`'s own field reads, since both services'
    lifespans can install the frame store (`server.py`'s own ownership rule) and
    must agree on whether the check runs. This service has no frame-grammar
    dependency to derive a content_version itself: when the flag is on, the store
    looks up the derivation function `analysis_mcp.frame_ref_resolution` registers into
    `common.materialisation.content_version` at its own import time. A deployment
    that sets this flag without also mounting analysis-mcp has nothing registered
    to check against, and `server.py`'s lifespan refuses to start rather than
    install a store with the check silently disabled.

    `get_data_deadline_s` is the wall-clock deadline for one whole `get_data` call
    through the harness's own mount, started at entry. How a call's deadline is shared
    out comes from the `data_mcp.deadline.CallBudget` its entry point set:
    `DIRECT_CALL_BUDGET` through the harness's own mount, which takes its deadline from
    this field, and `EXTERNAL_CALL_BUDGET` through the combined server, whose own 42s
    deadline and reserves are derived and explained in that module's own comment.
    Every network and storage wait
    in the call (the definition read, entity resolution, the mint POST, the operation
    poll, the re-mint definition write, the frame store read and write, the frame fetch,
    the statistics fetch) takes its bound from what remains of it, and a wait that runs
    out becomes a typed refusal or a degraded digest. The local work between those
    waits (hashing, Parquet parsing, rendering) is unbounded and runs in the reserves
    the waits leave. The default, 165s, sits 15s under the harness's 180s
    `call_timeout_s` for `get_data` (`services/harness/src/compiler/capabilities.py`).
    `operation_poll_interval_s` is the fallback wait between poll attempts: dfo-apis'
    own poll route names a pacing interval (`retry_after_ms`) only on the initial 202
    response, never on a later `pending`/`running` poll (`OperationEnvelope
    .retry_after_ms`'s own docstring), so every poll after the first falls back to
    this fixed interval, itself still clamped to what remains of the call deadline.

    Every float field below that bounds a wait must be finite and positive: a
    non-finite or non-positive value would leave the wait it bounds either
    unbounded or unable to run at all. `operation_poll_interval_s` additionally
    floors at 0.5s, since a shorter fallback interval would spin the poll loop
    against dfo-apis' own poll door. `get_data_deadline_s` must exceed the direct
    budget's `pre_mint_reserve_s` (50s), which pre-mint work leaves for the mint and
    what follows it, and caps at 170s, 10s under the harness's own 180s
    `call_timeout_s` for `get_data` (`services/harness/src/compiler/capabilities.py`),
    so the deadline this service enforces always fires before the harness's own call
    timeout would.
    `upstream_backoff_base_s` is not itself a wait bound but a multiplier
    `client.py` feeds into `random.uniform(0, upstream_backoff_base_s * 2
    ** (attempt - 1))`: zero is a legitimate "retry immediately, no backoff"
    configuration there, so it only has to stay finite and non-negative.

    `external_mint_concurrency` caps the upstream mints this process holds for
    `get_data` calls under `EXTERNAL_CALL_BUDGET`, the budget the combined server sets
    for outside callers (`data_mcp.mint_ledger`). A mint is held from its POST until
    dfo-apis answers it, or for dfo-apis' 120s synchronous mint timeout when the call
    stopped waiting first, since dfo-apis keeps running the mint, and keeps its slot,
    after the client leaves. A call past the cap refuses `service_unavailable`. The cap
    is per process, and the harness runs up to two instances (dfo-infra's `ai` service
    sets `max_instances = 2`), so outside callers can hold up to twice this many of the
    four mint slots a dfo-apis instance has (`mint_max_concurrency`), all of them on one
    instance at worst. The default of 1 leaves the harness's own calls two slots at that
    worst case. Each of those mints also takes
    one of dfo-apis' two materialisation slots (`materialise_max_concurrency`) while it
    rebuilds, so at the worst case one of the harness's own mints waits up to 30s for
    one (`materialise_wait_seconds`) and is then refused. Calls through the harness's
    own mount are not counted.
    """

    dfo_apis_url: str = "http://localhost:8083"
    upstream_connect_timeout_s: float = Field(default=5.0, gt=0, allow_inf_nan=False)
    upstream_read_timeout_s: float = Field(default=50.0, gt=0, allow_inf_nan=False)
    upstream_retry_budget_s: float = Field(default=120.0, gt=0, allow_inf_nan=False)
    upstream_max_attempts: int = 3
    upstream_backoff_base_s: float = Field(default=1.0, ge=0, allow_inf_nan=False)
    definition_store_bucket: str | None = None
    definition_store_cache_max_entries: int = DEFAULT_MAX_ENTRIES
    artifact_fetch_budget_s: float = Field(default=60.0, gt=0, allow_inf_nan=False)
    frame_store_bucket: str | None = None
    frame_store_verify_content_version: bool = False
    get_data_deadline_s: float = Field(
        default=165.0,
        gt=DIRECT_CALL_BUDGET.pre_mint_reserve_s,
        le=170.0,
        allow_inf_nan=False,
    )
    operation_poll_interval_s: float = Field(default=2.0, ge=0.5, allow_inf_nan=False)
    external_mint_concurrency: int = Field(default=1, ge=1)
