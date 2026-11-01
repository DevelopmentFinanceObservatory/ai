"""Errors raised by `PlatformAssembler` for apis presentation-surface call failures.

Two classes: a parsed or classified RFC 9457 problem, and the transport
failure that never got a response. `client.py`'s `request()` raises one of
these on every non-2xx or unreachable call; its callers translate them into
the `Refusal` the `Assembler` Protocol returns.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


class PlatformError(Exception):
    """An apis service presentation call failed with a parsed or classified RFC 9457 problem.

    `code` is only ever read off the wire, so an unmapped code and a body
    that never parsed as a problem document (both leave `code` as `None`)
    stay distinguishable from one another. `cause_code` is the problem's `cause.code`,
    why a refusal fired beneath its `code`, `None` on a refusal with no cause.
    """

    def __init__(
        self,
        *,
        status: int | None,
        code: str | None,
        retryable: bool,
        detail: str | None = None,
        pointer: str | None = None,
        alternatives: Sequence[Mapping[str, Any]] | None = None,
        issues: Sequence[Mapping[str, Any]] | None = None,
        constraint_id: str | None = None,
        truncated: bool = False,
        trace_id: str | None = None,
        cause_code: str | None = None,
    ) -> None:
        self.status = status
        self.code = code
        self.retryable = retryable
        self.detail = detail
        self.pointer = pointer
        self.alternatives = alternatives
        self.issues = issues
        self.constraint_id = constraint_id
        self.truncated = truncated
        self.trace_id = trace_id
        self.cause_code = cause_code
        super().__init__(f"platform call failed: status={status} code={code}")


class PlatformTransportError(PlatformError):
    """Raised when a call never got a response, e.g. a connection error or a
    read timeout. `status` and `code` stay `None`.
    """

    def __init__(self, *, detail: str) -> None:
        super().__init__(status=None, code=None, retryable=True, detail=detail)
