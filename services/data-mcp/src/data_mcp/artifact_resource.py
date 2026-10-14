"""The artifact resource: reads dfo-apis' snapshot route and returns an artifact's stable download link.

Each artifact in a manifest carries a root-relative link,
`/api/datasets/{id}/{content_version}/files/{digest}`. The link never
expires: a GET on it answers 302 to an object-store URL signed for a few
minutes, so the signed URL is minted at the moment of use and never sits in a
manifest, a tool result or this resource's response. This module holds no key
and signs nothing. It resolves the link against the dfo-apis origin this
service calls and returns the absolute URL beside the artifact's metadata.
`get_data_tool.py` strips the link from the manifest it returns; each
artifact's `resource_uri` (`manifest.py`) points here, so a caller pulls a link
only when it needs one.

The `dfo://artifact` URI is a bearer credential. Any session holding it can
read it and get the link, no matter which session's call built it.

A resource-read failure is a JSON-RPC code and message, thinner than the tool
path's typed refusal grammar (`tool_errors.py`): the code fastmcp assigns is
always `0` and carries no meaning, and `fix`/`retry_mode`/`retry_after_s` have
nowhere to live on that wire. So this module never reaches for
`ToolErrorCode`/`RetryMode` -- the four retry modes stay on the tool path --
and instead puts a stable, lowercase marker at the head of the `ResourceError`
message itself, as `"<code>: <prose>"` (`_refuse`). That marker mirrors
dfo-apis' own snapshot-route vocabulary verbatim (`spec_not_found`,
`snapshot_not_found`, `snapshot_revoked`, `artifact_expired`,
`service_unavailable`), plus three this module owns for a failure upstream never
sees (`upstream_response_unusable`, `request_invalid`,
`artifact_not_in_snapshot`) -- one vocabulary across the boundary, so an
upstream problem document is traceable straight from a model-facing string.

The link answers `410 artifact_expired` when the object aged out of the
working tier and cannot render again, and a missing digest answers 404. Those
answers come from the link itself, at the moment a caller follows it.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Mapping
from typing import Any, NoReturn
from urllib.parse import urlsplit

from common.refusal import internal_error_message, log_unexpected_error
from common.telemetry import current_trace_id
from fastmcp.exceptions import ResourceError
from pydantic import ValidationError

from data_mcp.client_holder import get_client
from data_mcp.errors import UpstreamError, UpstreamSurfaceUnavailable, UpstreamTransportError
from data_mcp.get_data_tool import _SPEC_ID_PATTERN
from data_mcp.manifest import ArtifactEntry, Manifest, render_validation_errors, strip_manifest

logger = logging.getLogger(__name__)

# The snapshot route's path *template*, passed as `route=` to
# `UpstreamClient.request` so span cardinality stays bounded the same way
# `get_data_tool.py` bounds it for the mint route.
_SNAPSHOT_ROUTE = "/api/datasets/{id}/{content_version}"

# The marker vocabulary -- see the module docstring for the mirrored-vs-owned
# split.
_SPEC_NOT_FOUND = "spec_not_found"
_SNAPSHOT_NOT_FOUND = "snapshot_not_found"
_SNAPSHOT_REVOKED = "snapshot_revoked"
_ARTIFACT_EXPIRED = "artifact_expired"
_ARTIFACT_NOT_IN_SNAPSHOT = "artifact_not_in_snapshot"
_SERVICE_UNAVAILABLE = "service_unavailable"
_UPSTREAM_RESPONSE_UNUSABLE = "upstream_response_unusable"
_REQUEST_INVALID = "request_invalid"

# Deliberately looser than whatever upstream actually mints
# (`datasets_api/schema.py:147` only publishes the `^cv_` prefix). This is this
# service's own bound on an unpublished grammar, so it stays loose.
_CONTENT_VERSION_PATTERN = re.compile(r"^cv_[A-Za-z0-9_.\-]{1,128}$")
_ARTIFACT_DIGEST_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")

_AGED_OUT_PROSE = (
    "This artifact has aged out of the working tier and is no longer servable. "
    "Call get_data again to re-mint it."
)

_NOT_IN_SNAPSHOT_PROSE = (
    "The snapshot at this content_version does not list this artifact digest. "
    "Check the digest against the resource link the mint returned. If that "
    "link is gone, call get_data again."
)


class ArtifactRefusalError(ResourceError):
    """A refusal this module writes for the caller, with its marker as `code`.

    Raised at `logging.DEBUG`, so fastmcp's own `Error reading resource '<uri>'`
    line, which carries the whole bearer URI, stays below the deployed level.
    `serve_artifact` logs the refusal instead, without the URI.
    """

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}", log_level=logging.DEBUG)
        self.code = code


def _refuse(code: str, detail: str) -> NoReturn:
    """Raise the one `ArtifactRefusalError` shape every refusal in this module uses.

    The JSON-RPC error `code` a client sees is always `0` and carries no
    meaning, so the machine-readable marker has to live at the head of the
    message string instead, as `"<code>: <prose>"`. Defined once here so that
    format can't drift between call sites.
    """
    raise ArtifactRefusalError(code, detail)


def _validate_pair(*, subject_id: str, content_version: str) -> None:
    """Validate the snapshot's two path parts before any upstream call. A malformed
    part refuses locally and never reaches the network."""
    if not re.fullmatch(_SPEC_ID_PATTERN, subject_id):
        _refuse(_REQUEST_INVALID, f"subject_id {subject_id!r} is not a valid spec_ id")
    if not _CONTENT_VERSION_PATTERN.fullmatch(content_version):
        _refuse(
            _REQUEST_INVALID, f"content_version {content_version!r} does not look like a cv_ id"
        )


def _validate_uri_parts(*, subject_id: str, content_version: str, artifact_digest: str) -> None:
    """Validate all three path parts before any upstream call.

    The URI is attacker-reachable on the standalone HTTP exposure and gets
    interpolated straight into an upstream path, so a malformed part refuses
    locally and never reaches the network.
    """
    _validate_pair(subject_id=subject_id, content_version=content_version)
    if not _ARTIFACT_DIGEST_PATTERN.fullmatch(artifact_digest):
        _refuse(_REQUEST_INVALID, f"artifact_digest {artifact_digest!r} is not a sha256: digest")


def _marker_for_upstream_error(exc: UpstreamError) -> tuple[str, str]:
    """Map a snapshot-call failure onto this module's marker vocabulary.

    Order matters: the two `UpstreamError` subclasses are checked first,
    since neither carries a snapshot-route `code` to match against. A `500
    internal_error` only promotes to `service_unavailable` when upstream
    itself marked it retryable. Upstream classifies a snapshot signing
    failure as non-retryable, unlike the mint route, and silently promoting
    it here would send a client hammering a call that will fail identically
    every time. The catch-all for an unclassified or bodyless failure splits
    the same way, on the same `exc.retryable` verdict `client.py` already
    computed.
    """
    # One if/elif chain assigning `code`/`detail`, and a single return at the
    # end, rather than a return per branch: this module's own vocabulary has
    # more than six cases, and a return-per-branch trips ruff's return-count
    # limit for no gain in clarity.
    if isinstance(exc, UpstreamSurfaceUnavailable):
        code, detail = _SERVICE_UNAVAILABLE, exc.detail or "the Datasets surface is unmounted"
    elif isinstance(exc, UpstreamTransportError):
        code, detail = (
            _SERVICE_UNAVAILABLE,
            exc.detail or "the snapshot call never received a response",
        )
    elif exc.code == _SPEC_NOT_FOUND:
        code, detail = _SPEC_NOT_FOUND, "no dataset is registered under this subject id"
    elif exc.code == _SNAPSHOT_NOT_FOUND:
        code, detail = _SNAPSHOT_NOT_FOUND, "no snapshot exists at this content_version"
    elif exc.code == _SNAPSHOT_REVOKED:
        code, detail = _SNAPSHOT_REVOKED, "this content_version has been taken down"
    elif exc.code == _ARTIFACT_EXPIRED:
        code, detail = _ARTIFACT_EXPIRED, _AGED_OUT_PROSE
    elif exc.code == "internal_error" and exc.retryable:
        code, detail = (
            _SERVICE_UNAVAILABLE,
            exc.detail or "the snapshot call failed transiently upstream",
        )
    elif exc.code == "internal_error":
        code, detail = (
            _UPSTREAM_RESPONSE_UNUSABLE,
            exc.detail or "the snapshot call failed upstream and is not safely retryable",
        )
    elif exc.code == _SERVICE_UNAVAILABLE:
        code, detail = _SERVICE_UNAVAILABLE, exc.detail or "a dfo-apis dependency is unavailable"
    elif (exc.code is None or (exc.status is not None and exc.status >= 500)) and exc.retryable:
        code, detail = (
            _SERVICE_UNAVAILABLE,
            exc.detail or f"an unclassified upstream failure ({exc.code})",
        )
    elif exc.code is None or (exc.status is not None and exc.status >= 500):
        code, detail = (
            _UPSTREAM_RESPONSE_UNUSABLE,
            exc.detail
            or f"an unclassified upstream failure ({exc.code}) that is not safely retryable",
        )
    else:
        code, detail = (
            _UPSTREAM_RESPONSE_UNUSABLE,
            exc.detail or f"an unrecognised upstream refusal ({exc.code})",
        )
    return code, detail


def _select_raw_artifact(payload: dict[str, Any], *, artifact_digest: str) -> dict[str, Any]:
    """Pick the first raw artifact entry matching `artifact_digest` out of `payload["artifacts"]`.

    An artifact digest names its bytes, so every entry sharing one serves the
    same content, even when upstream lists it under several renderings (a frame
    file is byte-identical to the long file of the same version and options).
    The first match in manifest order is taken; any entry would read the same.

    Runs over the raw payload, since `strip_manifest` has already dropped
    every `url`. This is the one place in the module that deliberately reads
    it back off the raw response, having already confirmed the payload
    validates as a `Manifest`.
    """
    matches = [a for a in payload["artifacts"] if a.get("artifact_digest") == artifact_digest]
    if not matches:
        # A digest absent from this snapshot's artifact list has two possible
        # causes: the artifact aged out of the working tier before signing
        # was attempted, or the caller supplied a digest this content_version
        # never minted, such as a mistyped or fabricated one. The payload
        # gives no way to tell these apart, so the marker stays neutral
        # between them rather than asserting either.
        _refuse(_ARTIFACT_NOT_IN_SNAPSHOT, _NOT_IN_SNAPSHOT_PROSE)
    return matches[0]


def _root_relative_link(url: object) -> str:
    """`url` if it is a root-relative path, else a refusal.

    `ArtifactEntryModel.url` upstream is a bare, unconstrained `str`. A scheme
    or a host (`//host/...`) would send the caller to an origin other than the
    dfo-apis this service calls. `urlsplit` can also raise `ValueError` on
    malformed input.
    """
    refusal = "the matched artifact entry's url is not a root-relative link"
    if not isinstance(url, str):
        _refuse(_UPSTREAM_RESPONSE_UNUSABLE, refusal)
    try:
        split = urlsplit(url)
    except ValueError:
        _refuse(_UPSTREAM_RESPONSE_UNUSABLE, refusal)
    if split.scheme or split.netloc or not split.path.startswith("/"):
        _refuse(_UPSTREAM_RESPONSE_UNUSABLE, refusal)
    return url


async def read_artifact(
    subject_id: str, content_version: str, artifact_digest: str
) -> dict[str, Any]:
    """Read one artifact's stable download link off dfo-apis' snapshot route.

    Reading this resource looks up the artifact identified by `subject_id`
    (a `spec_` id), `content_version`, and `artifact_digest` -- a metadata
    lookup, with no graph call and no re-derivation. The returned `url` is the
    absolute form of the artifact's stable link. It does not expire: a GET on
    it answers 302 to an object-store URL signed for a few minutes, so a caller
    follows redirects and reads the link again only if it lost it.

    Args:
        subject_id: The `spec_` id this artifact's manifest was minted
            under.
        content_version: The `cv_` id naming the exact retrieval.
        artifact_digest: The `sha256:` digest identifying this one artifact,
            as carried on the manifest's `resource_uri`.

    Returns:
        A document carrying the absolute `url` and the artifact's own
        metadata fields (`media_type`, `bytes`, `rendering`, `tier`).

    Raises:
        fastmcp.exceptions.ResourceError: Always, on every failure path. The
            message is `"<code>: <prose>"`; the JSON-RPC `code` fastmcp
            assigns carries no meaning and must not be matched on.
    """
    _validate_uri_parts(
        subject_id=subject_id, content_version=content_version, artifact_digest=artifact_digest
    )
    payload, manifest = await _read_snapshot(subject_id, content_version)
    raw_entry = _select_raw_artifact(payload, artifact_digest=artifact_digest)
    stripped_entry = next(
        (a for a in manifest.artifacts if a.artifact_digest == artifact_digest), None
    )
    if stripped_entry is None:
        # Unreachable: `strip_manifest` already validated every entry in
        # `payload["artifacts"]`, including the one `_select_raw_artifact`
        # just matched by the same digest field.
        raise AssertionError("digest matched in the raw payload but not in the stripped manifest")
    return _link_document(
        subject_id=subject_id,
        content_version=content_version,
        raw_entry=raw_entry,
        entry=stripped_entry,
    )


async def read_artifact_by_rendering(
    subject_id: str, content_version: str, rendering: Mapping[str, object]
) -> dict[str, Any] | None:
    """`read_artifact`'s document for the first artifact the snapshot lists under
    `rendering`, or `None` when it lists none.

    `frame_materialisation` reads a version's canonical frame this way: dfo-apis
    links it on the version record, so the snapshot lists it and a mint's own
    response does not. The digest comes from the snapshot, so the caller checks the
    bytes against the document's `artifact_digest`.

    Raises:
        fastmcp.exceptions.ResourceError: On every failure `read_artifact` raises
            for, except a missing artifact, which returns `None`.
    """
    _validate_pair(subject_id=subject_id, content_version=content_version)
    payload, manifest = await _read_snapshot(subject_id, content_version)
    # `strip_manifest` validated `payload["artifacts"]` entry for entry, in order.
    for raw_entry, entry in zip(payload["artifacts"], manifest.artifacts, strict=True):
        if entry.rendering == dict(rendering):
            return _link_document(
                subject_id=subject_id,
                content_version=content_version,
                raw_entry=raw_entry,
                entry=entry,
            )
    return None


async def _read_snapshot(subject_id: str, content_version: str) -> tuple[dict[str, Any], Manifest]:
    """The snapshot route's raw payload for the pair and its stripped manifest, or a
    refusal for a failed call or an unusable body."""
    client = get_client()
    try:
        response = await client.request(
            "GET",
            f"/api/datasets/{subject_id}/{content_version}",
            route=_SNAPSHOT_ROUTE,
            # The default timeout and `retry=True` (the client's own
            # default), safe here because the call is idempotent.
        )
    except UpstreamError as exc:
        code, detail = _marker_for_upstream_error(exc)
        _refuse(code, detail)

    try:
        payload = response.json()
    except ValueError as exc:
        _refuse(
            _UPSTREAM_RESPONSE_UNUSABLE,
            f"the snapshot response body could not be parsed as JSON: {exc}",
        )

    try:
        manifest = strip_manifest(payload)
    except ValidationError as exc:
        # Never `str(exc)` here: the snapshot response is full of artifact
        # links, and pydantic-core's default rendering embeds the raw,
        # un-stripped input for a failing field.
        _refuse(
            _UPSTREAM_RESPONSE_UNUSABLE,
            f"the snapshot response did not match the manifest shape this service expects: "
            f"{render_validation_errors(exc)}",
        )

    return payload, manifest


def _link_document(
    *,
    subject_id: str,
    content_version: str,
    raw_entry: dict[str, Any],
    entry: ArtifactEntry,
) -> dict[str, Any]:
    """The document `read_artifact` returns for one listed artifact: its absolute
    stable link beside its metadata."""
    link = _root_relative_link(raw_entry.get("url"))
    return {
        "subject_id": subject_id,
        "content_version": content_version,
        "artifact_digest": entry.artifact_digest,
        "url": get_client().absolute_url(link),
        "media_type": entry.media_type,
        "bytes": entry.bytes,
        "rendering": entry.rendering,
        "tier": entry.tier,
        "note": (
            "This link does not expire. A GET on it answers 302 to a URL signed "
            "for a few minutes, so follow redirects and fetch the bytes when you "
            "need them."
        ),
    }


async def serve_artifact(
    subject_id: str, content_version: str, artifact_digest: str
) -> dict[str, Any]:
    """`read_artifact` as the `dfo://artifact` resource serves it, logging each
    failure itself on one line that leaves out the URI, a bearer credential.

    A refusal logs its marker at WARNING. Any other exception is logged with
    its traceback and fails as `internal_error`, raised at
    `logging.DEBUG` and with no cause, since fastmcp would otherwise log it at
    ERROR as `Error reading resource '<uri>'` and wrap it in a message that
    repeats the URI.
    """
    try:
        return await read_artifact(subject_id, content_version, artifact_digest)
    except ArtifactRefusalError as exc:
        logger.warning("artifact read refused code=%s trace_id=%s", exc.code, current_trace_id())
        raise
    except Exception:
        log_unexpected_error(logger, "artifact read")
        raise ResourceError(
            "internal_error: " + internal_error_message(subject="Reading this resource"),
            log_level=logging.DEBUG,
        ) from None
