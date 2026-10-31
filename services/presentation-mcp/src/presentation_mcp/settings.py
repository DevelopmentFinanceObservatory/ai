"""Environment-driven settings for presentation-mcp."""

from __future__ import annotations

import ipaddress
import re
from urllib.parse import urlsplit

from common.settings import FrameServiceSettings
from pydantic import Field, SecretStr, field_validator

# Letters, digits, and hyphens in dot-separated labels. The origin goes into a
# Content-Security-Policy source list, where a space, `;`, quote, or `*` would
# change the policy.
_HOSTNAME = re.compile(r"[A-Za-z0-9-]+(\.[A-Za-z0-9-]+)*")


def _is_host(host: str | None) -> bool:
    """Whether `host` (brackets already stripped) is a plain hostname or an
    IPv6 literal."""
    if not host:
        return False
    if _HOSTNAME.fullmatch(host):
        return True
    try:
        ipaddress.IPv6Address(host)
    except ValueError:
        return False
    return "%" not in host


def _is_browser_url(url: str) -> bool:
    """Whether `url` is an absolute `http(s)` URL with a hostname (letters,
    digits, hyphens, and dots) or bracketed IPv6 literal, a port from 1 to
    65535 where it has one, and nothing a stylesheet path cannot be appended
    to: no query, fragment, credentials, whitespace, or control character.
    `urlsplit` drops tabs and newlines before parsing, so the raw string is
    checked for them first.
    """
    if any(char.isspace() or not char.isprintable() for char in url):
        return False
    try:
        parts = urlsplit(url)
        port = parts.port
    except ValueError:
        return False
    return (
        parts.scheme in ("http", "https")
        and _is_host(parts.hostname)
        and (port is None or 1 <= port <= 65535)
        and "@" not in parts.netloc
        and "?" not in url
        and "#" not in url
    )


class Settings(FrameServiceSettings):
    """presentation-mcp config: the shared base fields plus the apis presentation surface it wraps.

    `dfo_apis_url` is the base URL of apis, which mounts
    `/api/presentation/*` alongside its other surfaces on the same host.

    `max_frame_rows` (`common.FrameServiceSettings`, shared with data-mcp and
    analysis-mcp) bounds a frame this service reads back from the store to
    publish (`frame_publish.publish_body_for`).

    The `upstream_*` fields bound `PlatformAssembler`'s retry loop, mirroring
    data-mcp's own `UpstreamClient` settings (see `data_mcp.settings.Settings`
    for the ADK call-budget rationale behind these defaults).

    `max_schema_version` is the admission bound `server.py`'s startup lifespan
    compares the platform's `release()` `schema_version` against.
    `presentation_release_required` decides what the lifespan does on any
    startup admission failure -- the platform is unreachable, its release
    fails admission, its tool schemas fail to assemble, or the frame store
    fails to install (see `frame_store_bucket` below). `True` (the default,
    and what every deployment runs) retries an unreachable platform before
    startup finishes: the first retry waits `release_retry_base_s`, each
    later one doubles the wait up to `release_retry_max_s`, and no fetch
    starts after `release_startup_deadline_s`. Any failure left at that
    point fails startup and exits the process. The 90 s default sits inside
    Cloud Run's default 240 s startup probe, since the port opens only once
    startup finishes. `False`, for local development and tests, fetches the
    release once and on any failure leaves the server tool-less with
    `/healthz` reporting `degraded` until it restarts.

    `frame_store_bucket` names the same GCS bucket `analysis_mcp.settings
    .Settings`'s own field of the same name points at, since this service
    never seeds or writes the frame store -- it only reads an already-minted
    `ops_`/`ocv_` frame back out, to publish it to the platform lazily
    (`client.PlatformAssembler`'s own publish hook). `None` (the default)
    leaves this service relying on being mounted alongside analysis-mcp
    (`harness.server`'s in-process mount, the only deployment topology this
    service ships with), or on the frame-store holder's own lazy in-memory
    default when it is not.

    `frame_store_verify_content_version` names the same env var
    `analysis_mcp.settings.Settings`'s own field reads, since both services'
    lifespans can install the frame store (see `data_mcp.settings.Settings`'s
    own field of the same name for the ownership rule) and must agree on
    whether the check runs. Set `True` with no `analysis_mcp` in this
    process (a standalone deployment of this service), `server.py`'s
    lifespan install fails with `ContentVersionDeriverUnregistered` -- only
    `analysis_mcp` ever registers a deriver -- and that failure runs through
    `presentation_release_required` exactly like every other startup
    admission failure, rather than crashing the lifespan outright.

    `datasets_first_party_principal` is the principal apis grants `publish`
    to, sent as the `x-dfo-principal` header (`client.PRINCIPAL_HEADER`) on
    the call `client.PlatformAssembler._ensure_published_pair` issues to
    `POST /api/datasets/frames/publish`, which apis' publish door authorizes
    by it (`datasets_api.doors.publish`), and on a component create and revise
    that the harness makes for a Studio conversation (`Caller.FIRST_PARTY`).
    apis spends one unit of a per-caller daily card allowance on a create or
    revise and exempts a principal holding the publish grant, so Studio's
    writes spend none. An external caller's write never carries it
    (`client.PlatformAssembler._card_write_headers`). No other apis route this
    service calls reads this principal. Must equal
    whatever the apis deployment's own `DATASETS_FIRST_PARTY_PRINCIPAL` env
    var holds. The two are separate deployments, and nothing keeps them in
    sync but a deployer setting the same value on both. `None` (the default,
    including a blank env value) sends no header, so Studio's cards spend the
    allowance of the address dfo-apis sees, and this service publishes nothing ahead of a read
    or a write, since a frame_ref an analysis verb returns is one dfo-apis'
    analysis runner published already and the platform answers for it.
    `publish_frame` refuses `publish_not_configured` without one.

    `dfo_caller_principal` (`DFO_CALLER_PRINCIPAL`) is the secret dfo-apis
    shares with the website, the same value analysis-mcp reads. An external
    caller's component create or revise whose request the website named as
    `x-dfo-caller: ip:<digest>` carries it beside that caller
    (`x-dfo-caller-principal`, `x-dfo-caller`), so dfo-apis counts the card
    against that client's allowance. Unset or blank, or for a caller named any
    other way, the write carries neither header, and dfo-apis counts it against
    the address it sees for this service, one allowance every such caller shares.

    `live_refresh_principal` (`LIVE_REFRESH_PRINCIPAL`) is the principal apis grants
    `refresh`, the value `harness.settings.Settings` reads from the same env var, since
    the harness co-mounts this service in its process. `rebuild_lost_frames` sends it to
    `POST /api/datasets/frames/refresh` for a lost `spec_` frame whose rows a newest read
    would not hold (`client.PlatformAssembler._refreshed`). Unset or blank, such a frame
    refuses `rendering_not_recorded`.

    `presentation_public_url` is the URL at which this service's routes are
    reachable from a browser, mount path included: the co-mounted harness
    serves them under `/mcp/presentation`, so a deployment sets it to its
    public `https` URL plus `/mcp/presentation` (the Cloud Run service must set
    it, in infra; `just dev` and its siblings pass
    `http://localhost:$PORT/mcp/presentation`).
    Every view's payload names its theme stylesheet as
    `<presentation_public_url>/theme-<hash>.css` (`theme.theme_url`), and the
    renderer's CSP admits that URL's origin (`theme.renderer_csp`). There is no
    default: any address a default could name is unreachable from some
    browser, and a view whose stylesheet cannot load renders unstyled without
    an error. `PRESENTATION_PUBLIC_URL` unset, blank, or not an absolute
    `http(s)` URL with a host, and with no query, fragment, or credentials,
    fails Settings construction, and so startup.

    `chart_render_timeout_s` is how long a write_component or
    describe_component result waits for its chart image from the worker
    process (`chart_worker.ChartWorker`) before it goes without one. A
    render inside `chart_image`'s limits took at most 105 ms on an Apple M5 Pro.
    `chart_worker_warm` starts the worker process and loads its fonts at
    startup, and again after a render kills an overrunning worker, so the
    next render waits for neither. The test suites turn it off, since most
    of their servers never draw a chart and each start costs about 0.3 s of
    CPU.
    """

    dfo_apis_url: str = "http://localhost:8083"
    upstream_connect_timeout_s: float = 5.0
    upstream_read_timeout_s: float = 50.0
    upstream_retry_budget_s: float = 120.0
    upstream_max_attempts: int = 3
    upstream_backoff_base_s: float = 1.0
    max_schema_version: int = 1
    presentation_release_required: bool = True
    release_retry_base_s: float = 1.0
    release_retry_max_s: float = 60.0
    release_startup_deadline_s: float = 90.0
    frame_store_bucket: str | None = None
    frame_store_verify_content_version: bool = False
    datasets_first_party_principal: str | None = None
    dfo_caller_principal: SecretStr | None = None
    live_refresh_principal: SecretStr | None = None
    presentation_public_url: str = Field(default="", validate_default=True)
    chart_render_timeout_s: float = 2.0
    chart_worker_warm: bool = True

    @field_validator(
        "datasets_first_party_principal",
        "dfo_caller_principal",
        "live_refresh_principal",
        mode="before",
    )
    @classmethod
    def _blank_principal_is_unset(cls, value: object) -> object:
        """A blank env value means "unconfigured", matching
        `BaseServiceSettings._blank_mcp_token_is_unset`'s own precedent --
        an env file that sets `DATASETS_FIRST_PARTY_PRINCIPAL=` or
        `DFO_CALLER_PRINCIPAL=` (present but empty) must degrade the same way as leaving it out entirely, not
        send an empty header value apis' own principal check would
        refuse as a non-matching principal instead of as "not configured".
        """
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @field_validator("presentation_public_url")
    @classmethod
    def _public_url_is_an_absolute_http_url(cls, value: str) -> str:
        """Refuse an unset, blank, or malformed `PRESENTATION_PUBLIC_URL`
        loudly at startup: a view whose theme URL a browser cannot resolve
        renders unstyled with no error to point at.
        """
        url = value.strip()
        if _is_browser_url(url):
            return url
        problem = f"got {url!r}" if url else "it is not set"
        raise ValueError(
            "PRESENTATION_PUBLIC_URL must be the absolute http(s) URL at which a "
            "browser reaches presentation-mcp, mount path included, e.g. "
            f"https://<host>/mcp/presentation ({problem}). The Cloud Run "
            "service must set it (infra owns that config); `just dev` sets it for a local run."
        )
