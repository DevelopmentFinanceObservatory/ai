"""How the combined MCP server lists and calls a tool whose public form differs
from its per-server mount's.

A child server module exports `PUBLIC_SHAPES`, mapping a tool's name to its
`PublicShape`, and `harness.combined_mcp` applies every shape the same way:

- `paragraphs` are appended to the mount's description, so the public text is
  the mount's text plus what holds only for public calls;
- every property named in `hidden` is left out, wherever the schema's
  `properties` hold one, and from `required` beside it;
- `forced` is written over the caller's arguments on every call.

A text about a hidden argument belongs in that argument's own description, so
hiding the argument takes the text with it.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, kw_only=True)
class PublicShape:
    """One tool's public form, as `harness.combined_mcp` lists and calls it."""

    paragraphs: tuple[str, ...] = ()
    hidden: frozenset[str] = frozenset()
    forced: Mapping[str, Any] = field(default_factory=dict)
