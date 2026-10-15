"""Errors raised by `UpstreamClient` for dfo-apis call failures.

Three classes, pending #26's ten codes and four retry modes. Enough to tell a
refusal from a call that never landed, plus the probe's verdict as its own
catchable type.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


class UpstreamError(Exception):
    """A dfo-apis call failed with a parsed or classified RFC 9457 problem.

    #26 reads `status` and `code` together. A refusal dfo-apis meant to send
    carries both, a response without a parseable problem body carries `status`
    alone, and a call that never landed carries neither. `code` is only ever
    read off the wire, keeping an unmapped code and an absent one apart.

    `extensions` is where a problem document's RFC 9457 extension members
    land — fields such as `default_facet_undefined`'s `axis` that aren't
    among the six the constructor already names.
    """

    def __init__(
        self,
        *,
        status: int | None,
        code: str | None,
        title: str | None = None,
        detail: str | None = None,
        retryable: bool = False,
        trace_id: str | None = None,
        capability_version: str | None = None,
        body: bytes = b"",
        extensions: Mapping[str, Any] | None = None,
    ) -> None:
        self.status = status
        self.code = code
        self.title = title
        self.detail = detail
        self.retryable = retryable
        self.trace_id = trace_id
        self.capability_version = capability_version
        self.body = body
        self.extensions: dict[str, Any] = dict(extensions or {})
        # Filled in by the retry loop. #26's `retry_after_delay` derives its
        # interval from them, since no upstream response carries one.
        self.attempts = 1
        self.elapsed_s = 0.0
        super().__init__(f"upstream call failed: status={status} code={code} title={title}")


class UpstreamTransportError(UpstreamError):
    """Raised when a call never got a response, e.g. a connection error or a
    read timeout. `status` and `code` stay `None`."""

    def __init__(self, *, detail: str) -> None:
        super().__init__(status=None, code=None, detail=detail, retryable=True)


class UpstreamAuthUnavailable(UpstreamTransportError):  # noqa: N818 -- a kind of transport error, named for its cause
    """Raised when the auth provider could not mint a dfo-apis ID token, so nothing was sent.

    A subclass so callers that map any unanswered call keep doing so, while a
    caller that must tell "never sent" from "sent but unanswered" can catch it first.
    """


class UpstreamSurfaceUnavailable(UpstreamError):  # noqa: N818 -- named for the probe's verdict, not a generic "Error"
    """Raised when the surface probe confirms dfo-apis has not mounted `prefix`.

    An unmounted surface answers `not_found`, `retryable: false` on every path
    beneath it, exactly as a genuine "no such id" refusal does, so this means
    the probe `GET <probe_path>` answered 404 as well. `probe_path` is the
    path `data_mcp.client.GATED_PREFIXES` names for `prefix`.

    `code` is synthesised rather than read off the wire, so #26 excludes it
    from the upstream codes it maps. `attempts` and `elapsed_s` come from the
    retry loop that produced the refusal this replaces.
    """

    def __init__(
        self,
        *,
        prefix: str,
        probe_path: str,
        trace_id: str | None = None,
        capability_version: str | None = None,
        attempts: int = 1,
        elapsed_s: float = 0.0,
    ) -> None:
        self.prefix = prefix
        super().__init__(
            status=404,
            code="surface_unavailable",
            title="Upstream surface unavailable",
            detail=(
                f"GET {probe_path} answered 404 as well, so dfo-apis has "
                f"not mounted the surface at {prefix}. The id in the original "
                "request may well exist once the surface is deployed."
            ),
            retryable=False,
            trace_id=trace_id,
            capability_version=capability_version,
        )
        self.attempts = attempts
        self.elapsed_s = elapsed_s
