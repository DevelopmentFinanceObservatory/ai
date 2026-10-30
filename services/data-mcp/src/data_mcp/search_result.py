"""The parts of a successful `search` result, and where each one rides.

A success result carries three things:

- `content`: the text digest (`data_mcp.digest.render_digest`).
- `structuredContent`: the small answer (`data_mcp.digest.render_answer`),
  which states what the digest states, as data. A host may hand it to a
  model, so it holds only a status, and per part a label, a status, the
  `spec_id`s a model can pass to `get_data`, and warnings.
- `_meta[SEARCH_ENVELOPE_META_KEY]`: the upstream resolve envelope, after
  its definitions were registered. Code reads it, and a host keeps `_meta`
  out of a model's context.

A refusal keeps its own `structuredContent` (`data_mcp.tool_errors`) and
carries no envelope.

A call may carry `_meta[SEARCH_CONTEXT_META_KEY]`, the conversation a
search is made in (`data_mcp.search_tool.SearchContext`). The harness sends
it; the tool's input schema leaves it out, so an outside client is never
asked for it.
"""

from __future__ import annotations

# The result `_meta` key under which `search` carries the upstream envelope.
SEARCH_ENVELOPE_META_KEY = "org.one.dfo/search_envelope"

# The request `_meta` key under which a caller sends `search` the conversation
# it searches in (`data_mcp.search_tool.SearchContext`).
SEARCH_CONTEXT_META_KEY = "org.one.dfo/search_context"
