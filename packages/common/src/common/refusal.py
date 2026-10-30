"""The refusal text an MCP tool result carries in `content`.

data-mcp, analysis-mcp, presentation-mcp and the harness's combined server
each return refusals whose `content` ends in the refusal envelope as one
line of compact JSON, so a client that shows its model only `content` still
sees the code, the retry mode and the fix. `refusal_text` renders that text
for all four. `refusal_code` reads the code from a refusal's
`structuredContent`. `detail_sentence` turns an upstream service's `detail`
fragment into a sentence of a refusal's message.

Every refusal from those four also carries its code in `_meta` under
`REFUSAL_META_KEY` (`refusal_meta`), where the combined server reads it.
`structuredContent` cannot serve: presentation-mcp answers an MCP Apps host
with a view there. MCP hosts keep `_meta` out of the model's context.

An exception no typed handling anticipated answers `internal_error`. Its
text can hold a bucket path, a URL or a service account, so the reply
carries none of it: `log_unexpected_error` logs the exception under the trace
id, and `internal_error_message` is the fixed text that answers.
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


def detail_sentence(detail: str) -> str:
    """`detail`, an upstream fragment, as its own sentence: a capital first letter and a
    closing stop. A fragment that opens with an identifier, a first word holding `_` or
    `.` such as `frame_ref`, keeps it as written. An empty fragment stays empty."""
    text = detail.strip()
    if not text:
        return text
    if not {"_", "."} & set(text.split(maxsplit=1)[0]):
        text = text[:1].upper() + text[1:]
    return text if text.endswith((".", "!", "?")) else f"{text}."


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
    names none."""
    body = structured or {}
    code = body.get("code")
    return str(code) if code else None


def log_unexpected_error(log: logging.Logger, what: str) -> None:
    """Log the exception being handled at ERROR, with its traceback and the
    trace id (`common.telemetry.current_trace_id`).

    Call it inside the `except` block, so the record carries the exception.
    `what` says what failed, such as the tool's name.
    """
    log.exception("%s raised an unexpected error trace_id=%s", what, current_trace_id())


def internal_error_message(*, subject: str) -> str:
    """The fixed text of an `internal_error` reply: `subject` failed on the
    server, and the failure can be reported with the time of the call and the
    name of what was called."""
    return (
        f"{subject} failed on the server with an unexpected error. Repeating the "
        "same call is unlikely to help. Report it with the time of the call and "
        "the name of what was called."
    )
