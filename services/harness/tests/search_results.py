"""A successful `search` result's parts, as data-mcp returns them.

`data_mcp.search_result` describes the three parts: the digest in `content`,
the small answer in `structuredContent`, and the upstream envelope in
`_meta`. `search_result_parts` builds the last two from an envelope, as
keyword arguments for `fastmcp.tools.ToolResult` and
`compiler.surface.ToolCallResult` alike.
"""

from __future__ import annotations

from typing import Any

from data_mcp.digest import render_answer
from data_mcp.search_result import SEARCH_ENVELOPE_META_KEY


def search_result_parts(envelope: dict[str, Any]) -> dict[str, Any]:
    """`structured_content` and `meta` for a success `search` result carrying `envelope`."""
    return {
        "structured_content": render_answer(envelope),
        "meta": {SEARCH_ENVELOPE_META_KEY: envelope},
    }
