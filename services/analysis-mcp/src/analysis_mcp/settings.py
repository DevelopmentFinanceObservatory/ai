"""Environment-driven settings for analysis-mcp."""

from __future__ import annotations

from common.settings import FrameServiceSettings
from pydantic import SecretStr, field_validator


class Settings(FrameServiceSettings):
    """analysis-mcp config: the shared frame-service fields, plus the fixture-store
    seeding flag and the toolkit-wide ceilings a verb checks before it computes.

    `seed_fixture_frames` gates `analysis_mcp.fixtures.seed_fixture_store`
    in `server.py`'s lifespan -- default off, since production has no use
    for development-finance fixture data. Set `SEED_FIXTURE_FRAMES=true` for
    local development or an eval run against a live analysis-mcp instance.

    `max_frame_rows` (`common.FrameServiceSettings`, shared with data-mcp and
    presentation-mcp) is the row-count ceiling every verb checks on each input
    before it decodes it, not just `aggregate`. The store's row/column/size ceilings
    (`common.materialisation.GcsMaterialisationStore`) are fixed,
    store-side constants a caller cannot configure, since more than one
    service can build a store over the same bucket.

    `aggregate_coverage_floor` is the minimum realised coverage fraction
    (cells present / cells expected for the grouped grain, see
    `analysis_mcp.engine.aggregate`'s module docstring) below which
    `aggregate` and `rank` refuse rather than return a number computed from
    a grid too sparse to trust. 0.5 is a deliberately simple starting point
    -- "less than half the expected grid is present" is the plainest line to
    draw. It is not caller-overridable at this level.

    `frame_store_bucket` names the GCS bucket the materialisation store
    persists to; `None` (the default) keeps the in-process, restart-losing
    store. Do not set this on a deployment before the bucket's own lifecycle
    rule exists: that rule bounds retention, is provisioned in `infra`,
    not here, and until it exists a durable store's objects accumulate
    indefinitely.

    `frame_store_verify_content_version` gates the durable store's own
    expensive re-derivation check on every read (`GcsMaterialisationStore`'s
    own `verify_content_version_on_read`) -- off by default, since it costs
    a full content_version recompute per read. `data_mcp.settings.Settings`
    reads the same env var into the same field name, since either service's
    lifespan may be the one that installs the shared store.

    `dfo_apis_url` is the base URL of dfo-apis, whose analysis runner
    (`POST /api/datasets/analysis`) computes every verb and `resolve_anchor`
    (`runner_client`). `analysis_runner_timeout_s` bounds one runner call; it
    sits above the runner's own 120 s run limit, so the runner's `timeout`
    refusal arrives before this one.

    `dfo_caller_principal` (`DFO_CALLER_PRINCIPAL`) is the secret dfo-apis shares
    with the website. A call whose caller the website named as `ip:<digest>`
    carries it beside that caller (`x-dfo-caller-principal`, `x-dfo-caller`), so
    dfo-apis counts the website's client. Unset or blank, no caller is sent.
    """

    seed_fixture_frames: bool = False
    aggregate_coverage_floor: float = 0.5
    frame_store_bucket: str | None = None
    frame_store_verify_content_version: bool = False
    dfo_apis_url: str = "http://localhost:8083"
    analysis_runner_timeout_s: float = 130.0
    dfo_caller_principal: SecretStr | None = None

    @field_validator("dfo_caller_principal", mode="before")
    @classmethod
    def _blank_principal_is_unset(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip():
            return None
        return value
