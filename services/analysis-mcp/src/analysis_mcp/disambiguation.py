"""The disambiguation payload: a candidate set a caller chooses from.

A refusal (`ToolRefusal`, the per-service typed-error grammar data-mcp's
`entity_ambiguous` already demonstrates) says a call failed and how to
correct it; a disambiguation payload says a call was understood but
resolved to more than one candidate, and names them so the next call can
pick one. A verb's own tool wires this shape into its refusal's `fix`
payload for the ambiguous case. This module only locks the shape, the same
way `op_spec.py` locks the verb shapes without implementing any verb.

This module also holds `escape_control_chars`/`truncate_value`, the one
choke point every frame-derived string passes through before it reaches
model-facing text -- `analysis_mcp.digest`'s renderers and
`analysis_mcp.tool_errors`'s refusal messages both interpolate row-derived
values (a role, entity, unit, or facet value) into text a model reads, and
both import from here rather than each keeping a private copy, so there is
exactly one place that guards the boundary.
"""

from __future__ import annotations

import unicodedata
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field

MAX_LABEL_CHARS = 200


def escape_control_chars(value: str) -> str:
    """`value`, with every Unicode category `Cc` (control) or `Cf` (format)
    character replaced by its `\\uXXXX` escape. `dfo_frame_grammar` places no
    character restriction on a role/entity value beyond rejecting lone UTF-16
    surrogates, so a raw newline, carriage return, or escape/bidi-override
    character can reach here verbatim; escaping them is what makes a
    digest's brackets (`[...]`, `role=value`) or a refusal message's own
    line structure real delimiters a value can never break out of or use to
    forge a line that reads as a directive. Every other codepoint, including
    any non-ASCII letter, passes through unchanged."""
    return "".join(
        f"\\u{ord(char):04x}" if unicodedata.category(char) in ("Cc", "Cf") else char
        for char in value
    )


def truncate_value(value: str) -> str:
    """`value`, escaped (`escape_control_chars`) and then capped to
    `MAX_LABEL_CHARS` -- a role, unit, facet, or entity-label value read
    straight off an input row carries no length ceiling before it reaches
    here. Escaping runs first: an escaped control character is six
    characters, so capping first and escaping the kept slice would let a
    value crafted to exploit that ordering render far past `MAX_LABEL_CHARS`
    once escaped."""
    value = escape_control_chars(value)
    if len(value) <= MAX_LABEL_CHARS:
        return value
    return value[:MAX_LABEL_CHARS] + "...(truncated)"


class DisambiguationCandidate(BaseModel):
    """One candidate in a disambiguation payload."""

    model_config = ConfigDict(extra="forbid")

    id: Annotated[str, Field(min_length=1)] | dict[str, object] = Field(
        description=(
            "The value a retry would pass to select this candidate: a string, or the "
            "object a converted unit pins by."
        )
    )
    label: str = Field(min_length=1, max_length=MAX_LABEL_CHARS, description="Human-readable name.")
    detail: dict[str, Any] = Field(
        default_factory=dict,
        description="Per-candidate detail a caller can use to tell candidates apart.",
    )


class DisambiguationPayload(BaseModel):
    """A candidate set a caller must choose from before a call can proceed."""

    model_config = ConfigDict(extra="forbid")

    candidates: tuple[DisambiguationCandidate, ...] = Field(min_length=1)
