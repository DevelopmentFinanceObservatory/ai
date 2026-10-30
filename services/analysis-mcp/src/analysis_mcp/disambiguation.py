"""The disambiguation payload: a candidate set a caller chooses from.

A refusal (`ToolRefusal`, the per-service typed-error grammar data-mcp's
`entity_ambiguous` already demonstrates) says a call failed and how to
correct it; a disambiguation payload says a call was understood but
resolved to more than one candidate, and names them so the next call can
pick one. A verb's own tool wires this shape into its refusal's `fix`
payload for the ambiguous case. This module only locks the shape, the same
way `op_spec.py` locks the verb shapes without implementing any verb.
"""

from __future__ import annotations

from typing import Annotated, Any

from common.echoed_text import MAX_LABEL_CHARS
from pydantic import BaseModel, ConfigDict, Field


class DisambiguationCandidate(BaseModel):
    """One candidate in a disambiguation payload."""

    model_config = ConfigDict(extra="forbid")

    id: Annotated[str, Field(min_length=1)] | dict[str, object] = Field(
        description=(
            "The value a retry would pass to select this candidate: a string, or the "
            "object a converted or composed unit pins by."
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
