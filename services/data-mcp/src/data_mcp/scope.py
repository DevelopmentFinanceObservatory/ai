"""Search's scope entries.

A scope entry is an option asked for on one axis of a part, in the shape
Search's `Entry` writes (`search.models.scope` in dfo-apis). The harness
reads them from the scope choices a stored search reading holds
(`harness.scope_choices.PartScope`).
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

# The parsed door's limits on an entry: the user's message an entry's words
# are read from, and every id or concept an entry names.
MAX_USER_MESSAGE_CHARS = 4000
MAX_ENTRY_ID_CHARS = 200

ScopeAxis = Literal["sector", "region", "measure_basis", "provider"]

# How a part's scope choice came to hold its active option: the user's
# words, a choice carried from earlier in the conversation, or Search's
# default.
ChoiceProvenance = Literal["explicit", "carried", "default"]

ScopeReasonCode = Literal[
    "provider_not_offered",
    "imputed_coverage",
    "basis_not_published",
    "residuals_need_summed_region",
    "conflicts_with",
    "carried_choice_not_applied",
    "stated_choice_not_applied",
    "option_not_covered",
    "contributions_counted",
]

# "<axis>:<value>": `sector:120+130`, `region:members_and_residuals`,
# `measure_basis:gross`, `provider:inclusive`. One option has one id on every part.
OptionId = Annotated[
    str,
    Field(
        pattern=r"^(sector|region|measure_basis|provider):[A-Za-z0-9_+]+$",
        max_length=MAX_ENTRY_ID_CHARS,
    ),
]


class TextOrigin(BaseModel):
    """The words of the user's message an explicit entry was read from."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["text"] = "text"
    text: str = Field(max_length=MAX_USER_MESSAGE_CHARS)
    start: int = Field(ge=0)
    end: int = Field(ge=0)


class ChoiceOrigin(BaseModel):
    """The scope choice an entry was switched or carried from."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["choice"] = "choice"
    choice_id: str = Field(min_length=1, max_length=MAX_ENTRY_ID_CHARS)


class ScopeEntry(BaseModel):
    """One explicit or carried entry: an option asked for on one axis.

    `concept` is the concept the entry was made for ("health"), so a carried
    entry attaches only to a part asking for that concept; `None` when the
    axis has no concept. `origin` is where the entry came from.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    axis: ScopeAxis
    option_id: OptionId
    provenance: Literal["explicit", "carried"]
    concept: str | None = Field(default=None, min_length=1, max_length=MAX_ENTRY_ID_CHARS)
    origin: Annotated[TextOrigin | ChoiceOrigin, Field(discriminator="kind")]
