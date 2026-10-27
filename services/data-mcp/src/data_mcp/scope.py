"""Search's scope entries and the words for its scope reason codes.

A scope entry is an option asked for on one axis of a part, in the shape
Search's `Entry` writes (`search.models.scope` in dfo-apis). The search
tool carries the entries a conversation settled to Search
(`data_mcp.search_tool.SearchContext`), and the harness reads them off a
part's `resolved.scope_choices` and posts them back
(`harness.scope_choices`), so both read and write them through
`ScopeEntry` and its door limits.

`REASON_COPY` words each reason code Search gives an option's
availability or a dropped entry, one clause each: the search digest names
why an entry was dropped with it (`data_mcp.digest`), and the harness's
basis line why an option is limited.
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
    "counts_core_contributions",
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


# What each scope reason code says to a reader, one clause.
REASON_COPY: dict[str, str] = {
    "provider_not_offered": "imputed multilateral aid is not offered for this question",
    "imputed_coverage": "imputed multilateral aid covers only some years",
    "basis_not_published": "this basis is not published for these figures",
    "residuals_need_summed_region": "regional buckets apply only to a summed region",
    "conflicts_with": "it conflicts with another choice",
    "carried_choice_not_applied": "an earlier choice did not apply here",
    "stated_choice_not_applied": "the option you asked for is not available here",
    "option_not_covered": "the data holds no series for this option",
    "counts_core_contributions": (
        "this total counts core contributions to multilaterals as the donor's aid, so it "
        "cannot be limited to bilateral flows"
    ),
    "contributions_counted": (
        "this total already counts core contributions to multilaterals as the donor's aid, "
        "in place of an imputed share of their spending"
    ),
}
