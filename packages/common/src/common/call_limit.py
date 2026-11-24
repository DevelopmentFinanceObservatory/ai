"""How long a call from an outside client may run.

The combined server (`harness.combined_mcp`) cancels a tool call from an outside client
at `EXTERNAL_CALL_LIMIT_S` and refuses it. A tool that bounds its own work for such a
call stops `EXTERNAL_CALL_MARGIN_S` before that limit, so its own refusal reaches the
caller first: `get_data`'s deadline (`data_mcp.deadline`) and a public analysis call
sent once more (`analysis_mcp.tool_pipeline`).
"""

from __future__ import annotations

EXTERNAL_CALL_LIMIT_S = 45.0

# Writing a refusal takes well under a second, so this leaves room for it to reach
# the caller before the combined server cancels the call.
EXTERNAL_CALL_MARGIN_S = 3.0
