"""The MCP annotations the read-only tools declare, defined once.

Every tool that reads DFO's own, bounded catalogue shares this value.
These tools store content they derive, each keyed by a hash of that content:

- `search` writes the definitions it resolves to into the definition store
  (`data_mcp.definition_writes`).
- `get_data` asks dfo-apis to mint a frame, which registers its definition
  there, and a mint can keep running after the caller's call has ended
  (`data_mcp.mint_ledger`). It stores the frame in the frame store.
- The analysis tools store their result frame in the frame store, and
  `resolve_anchor` stores the anchor record it resolves.
- `describe_frame`, for an analysis result, uploads that frame to dfo-apis'
  publish door before asking for its view, because the platform computes the
  view and cannot otherwise resolve the frame
  (`presentation_mcp.client._ensure_published_pair`). dfo-apis keeps accepted
  frames.

The label stays read-only because a repeat of any of these stores nothing new,
and none makes anything readable to a caller who does not already hold the id.
`destructive_hint` is stated as false on a read-only tool so a directory that
requires all three hints finds each as an explicit boolean. `idempotent_hint`
is left unset, because it means something only on a tool that writes.
"""

from __future__ import annotations

from mcp.types import ToolAnnotations

READ_ONLY_CATALOGUE = ToolAnnotations(
    read_only_hint=True,
    destructive_hint=False,
    open_world_hint=False,
)
