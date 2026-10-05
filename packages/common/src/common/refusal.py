"""The refusal text an MCP tool result carries in `content`.

data-mcp, analysis-mcp and the harness's combined server each return
refusals whose `content` ends in the refusal envelope as one line of
compact JSON, so a client that shows its model only `content` still sees
the code, the retry mode and the fix. `refusal_text` renders that text for
all three. `refusal_code` reads the code from a refusal's
`structuredContent`, whichever of the two envelopes that holds.

Every refusal from those three and from presentation-mcp also carries its
code in `_meta` under `REFUSAL_META_KEY` (`refusal_meta`), where the
combined server reads it. `structuredContent` cannot serve: presentation-mcp
names its code `error`, and answers an MCP Apps host with a view there.
MCP hosts keep `_meta` out of the model's context.

An exception no typed handling anticipated answers `internal_error`. Its
text can hold a bucket path, a URL or a service account, so the reply
carries none of it: `log_unexpected_error` logs the exception under a trace
id, and `internal_error_message` is the fixed text that names the id.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from typing import Any

from common.telemetry import current_trace_id

# The tool result `_meta` key under which a refusal carries its code.
REFUSAL_META_KEY = "org.one.dfo/refusal"


def refusal_text(
    *,
    message: str,
    code: str,
    retry_mode: str,
    fix: dict[str, Any],
    retry_after_s: float | None = None,
    preamble: str | None = None,
) -> str:
    """`message`, a blank line, then `code`, `retry_mode`, `fix` and
    `retry_after_s` when set as one line of compact JSON.

    `preamble`, when given, comes first, followed by a blank line. A partial
    success names what resolved before the refusal says what did not.
    """
    envelope: dict[str, Any] = {"code": code, "retry_mode": retry_mode, "fix": fix}
    if retry_after_s is not None:
        envelope["retry_after_s"] = retry_after_s
    text = f"{message}\n\n{json.dumps(envelope, separators=(',', ':'))}"
    return f"{preamble}\n\n{text}" if preamble else text


def refusal_meta(*, code: str, upstream_code: str | None = None) -> dict[str, Any]:
    """The `_meta` of a refusal with `code`: `{"code": code}` under
    `REFUSAL_META_KEY`, with `upstream_code` added when the refusal names the
    upstream service's own code."""
    refusal = {"code": code}
    if upstream_code is not None:
        refusal["upstream_code"] = upstream_code
    return {REFUSAL_META_KEY: refusal}


def refusal_code(structured: Mapping[str, Any] | None) -> str | None:
    """The code of the refusal envelope in `structured`, or `None` when it
    names none.

    data-mcp, analysis-mcp and the combined server's own refusals carry the
    code as `code`. presentation-mcp's `Refusal` carries it as `error`.
    """
    body = structured or {}
    code = body.get("code") or body.get("error")
    return str(code) if code else None


def log_unexpected_error(log: logging.Logger, what: str) -> str:
    """Log the exception being handled at ERROR, with its traceback, and return
    the trace id the line names (`common.telemetry.current_trace_id`).

    Call it inside the `except` block, so the record carries the exception.
    `what` says what failed, such as the tool's name.
    """
    trace_id = current_trace_id()
    log.exception("%s raised an unexpected error trace_id=%s", what, trace_id)
    return trace_id


def internal_error_message(*, subject: str, trace_id: str) -> str:
    """The fixed text of an `internal_error` reply: `subject` failed, and
    `trace_id` finds the logged exception."""
    return (
        f"{subject} failed with an unexpected error. Repeating the same call is "
        f"unlikely to help. Quote trace id {trace_id} when reporting it."
    )
