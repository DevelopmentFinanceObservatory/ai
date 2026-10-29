"""How apis admits a `get_data` call's mint: at once, or only when it has room.

A `get_data` call from the harness whose request `_meta` carries
`MINT_ADMISSION_META_KEY` set to `BACKGROUND_ADMISSION` mints in the
background (`requested_admission`): its mint POST carries
`MINT_ADMISSION_HEADER: background`, and apis admits it only while it holds
spare work capacity. Otherwise it answers 503 `service_unavailable` with
`constraint_id` `BACKGROUND_DECLINED`, which this service's refusal carries
on to the caller in its `fix` (`declined_in_background`). A background call
answers once its POST has been written out (`background_result`, outcome
`"posted"`), or with apis' answer when that comes first. The POST itself stays
open in a task of its own until apis answers, since apis runs the mint on CPU
that Cloud Run allocates only while the request is open. It never polls and
reads no frame. Its tool-call span carries `BACKGROUND_MINT_SPAN_ATTRIBUTE`.
The harness sends it for a card's mints before the user approves the card
(`harness.early_mint`).

A background call answers before its mint ends and leaves the POST open in a
task that outlives the request, so only the harness may ask for one: a call
that names a conversation caller (`common.caller.conversation_caller`) and
arrives outside the combined server, whose outside calls all run under
`EXTERNAL_CALL_BUDGET`. Every other call that carries the marker, an outside
client's included, mints in the foreground, holding its connection for the
mint as any call does.

The marker rides `_meta`, outside the tool's arguments, so no input schema,
tool description or capability pin carries it.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Any, Final, Literal

from common.caller import CONVERSATION_CALLER_PREFIX, current_caller
from common.refusal import refusal_code
from fastmcp.server.dependencies import fastmcp_request_ctx
from fastmcp.tools import ToolResult

from data_mcp.deadline import EXTERNAL_CALL_BUDGET, current_call_budget
from data_mcp.errors import UpstreamError
from data_mcp.tool_errors import ToolRefusal

# The request `_meta` key a caller marks a background mint under.
MINT_ADMISSION_META_KEY = "org.one.dfo/mint_admission"

# The one value of `MINT_ADMISSION_META_KEY` that changes anything.
BACKGROUND_ADMISSION: Final = "background"

# The header apis reads a mint POST's admission from.
MINT_ADMISSION_HEADER = "X-DFO-Mint-Admission"

# The `constraint_id` apis declines a background mint under, and the key a
# refusal's `fix` carries it on under.
BACKGROUND_DECLINED = "mint_background_declined"
CONSTRAINT_ID_KEY = "constraint_id"

# The structured-content key a background call's result states its outcome
# under: "minted" (apis answered 200), "queued" (202) or "posted" (written out,
# with apis' answer still to come).
BACKGROUND_OUTCOME_KEY = "background_mint"

# The span attribute a background call's tool-call span carries.
BACKGROUND_MINT_SPAN_ATTRIBUTE = "dfo.background_mint"

# A mint's admission: `"background"`, or `None` for a foreground mint.
type Admission = Literal["background"] | None


def requested_admission() -> Admission:
    """`"background"` when the current request's `_meta` asks for a background
    mint and the harness sent it: a conversation caller, outside the combined
    server's `EXTERNAL_CALL_BUDGET`. `None` for any other value, no value, any
    other caller, or no MCP request in flight."""
    request_context = fastmcp_request_ctx.get()
    meta = request_context.meta if request_context is not None else None
    asked = meta.get(MINT_ADMISSION_META_KEY) if meta else None
    if asked != BACKGROUND_ADMISSION or current_call_budget() is EXTERNAL_CALL_BUDGET:
        return None
    caller = current_caller()
    if caller is None or not caller.startswith(CONVERSATION_CALLER_PREFIX):
        return None
    return BACKGROUND_ADMISSION


def admission_headers(admission: Admission) -> dict[str, str]:
    """The headers a mint POST sends for `admission`: none for a foreground mint."""
    return {} if admission is None else {MINT_ADMISSION_HEADER: admission}


def declined_in_background(refusal: ToolRefusal, exc: UpstreamError) -> ToolRefusal:
    """`refusal`, made from `exc`, with `exc`'s `constraint_id` added to its
    `fix` when apis declined a background mint; unchanged otherwise."""
    if exc.extensions.get(CONSTRAINT_ID_KEY) != BACKGROUND_DECLINED:
        return refusal
    return dataclasses.replace(refusal, fix={**refusal.fix, CONSTRAINT_ID_KEY: BACKGROUND_DECLINED})


def background_result(
    *, spec_id: str, outcome: Literal["minted", "queued", "posted"]
) -> ToolResult:
    """A background call's answer once its mint POST has been written out, or answered."""
    return ToolResult(
        content=f"Background mint of {spec_id}: {outcome}.",
        structured_content={"spec_id": spec_id, BACKGROUND_OUTCOME_KEY: outcome},
    )


def background_outcome(structured: Mapping[str, Any] | None) -> str:
    """A background call's answer as one word, from its structured content:
    its `BACKGROUND_OUTCOME_KEY` (`"minted"`, `"queued"` or `"posted"`),
    `"declined"` when apis had no room for it, the refusal's code otherwise, or
    `"answered"` for an answer that is none of these."""
    body = structured or {}
    if outcome := body.get(BACKGROUND_OUTCOME_KEY):
        return str(outcome)
    fix = body.get("fix")
    if isinstance(fix, Mapping) and fix.get(CONSTRAINT_ID_KEY) == BACKGROUND_DECLINED:
        return "declined"
    return refusal_code(body) or "answered"
