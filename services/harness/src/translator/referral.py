"""What the translator hands back when a plan cannot be carried out as stated.

A referral is typed because the planner has to act on it: a choice of source
is a decision to settle with the user, a requirement no data meets is one to
amend, a decision no operation performs is one to revisit, a data service
that would not answer is one to retry later rather than a fact about the
requirement's own content. `candidates` is filled by code from what
discovery recorded, never proposed by a model.

Three kinds carry `candidates`, for different reasons. `SOURCE_AMBIGUOUS`
always carries two or more: discovery found a real choice, and that is the
choice the planner puts to the user. `PINNED_SOURCE_NOT_FOUND` carries
whatever the discovery log holds for the requirement, zero sources or more:
the id the plan pinned did not survive a fresh search, and naming what did
survive -- nothing, one source, or the same several a `SOURCE_AMBIGUOUS`
referral would have named -- is the difference between a dead end and a
question the planner can still answer. `NO_DATA` carries the nearest real
sources `search` offered instead of the ask, zero or more, each one rendered
without its `spec_id` (`compact_source(selectable=False)`): they describe
what is nearby, not a source the planner may pin without a fresh search of
its own. `no_data_reason` names the closed code `search` gave for why nothing
was found -- an opaque string from another service, since a value it does
not yet recognise is still data to report, not a reason to fail the referral.
Every other kind carries neither.

`recommendation` is the translator's one surviving judgement, folded into
the referral rather than sent as a second call: a `source_ambiguous`
referral may carry the candidate the translator's own model ranked first,
and why, for the person who settles the choice to read as a suggestion,
not a pick the translator makes on its own. No other kind carries one.

`save_referral`, `load_referral` and `clear_referral` keep the last referral in
session state, under `REFERRAL_STATE_KEY`, so the planner can read why the
translator handed the plan back without it riding along on the plan itself.
"""

from __future__ import annotations

from enum import StrEnum

from planner.plan import SessionState
from planner.records import Identifier, SpecId, StrictModel
from planner.workflow import SelectedSource
from pydantic import Field, model_validator

# The session-state slot holding the last `Referral`. One key, since only the
# most recent referral is ever pending -- a new one replaces it.
REFERRAL_STATE_KEY = "translator:referral"


class ReferralKind(StrEnum):
    """Why the translator stopped."""

    SOURCE_AMBIGUOUS = "source_ambiguous"
    NO_DATA = "no_data"
    UNSUPPORTED_ASK = "unsupported_ask"
    REQUIREMENT_UNMEETABLE = "requirement_unmeetable"
    NO_OPERATION_COVERS = "no_operation_covers"
    DISCOVERY_BUDGET_EXHAUSTED = "discovery_budget_exhausted"
    PINNED_SOURCE_NOT_FOUND = "pinned_source_not_found"
    ASK_NOT_UNDERSTOOD = "ask_not_understood"
    # A `search` that timed out or that data-mcp refused outright (a
    # transport failure, or an envelope `outcome_from_envelope` could not
    # read) -- a fact about the data service right now, not about the
    # requirement's own content, so it is never `requirement_unmeetable`:
    # nothing about the requirement needs to change for a retry to work.
    DISCOVERY_UNAVAILABLE = "discovery_unavailable"
    # A `search` that answered a requirement with a dataset the turn dropped
    # (`planner.plan.dropped_seeds`): the user is told the plan leaves it out,
    # so the planner removes the requirement.
    SEED_DROPPED = "seed_dropped"


ABOUT_A_REQUIREMENT = frozenset(
    {
        ReferralKind.SOURCE_AMBIGUOUS,
        ReferralKind.NO_DATA,
        ReferralKind.UNSUPPORTED_ASK,
        ReferralKind.REQUIREMENT_UNMEETABLE,
        ReferralKind.PINNED_SOURCE_NOT_FOUND,
        ReferralKind.ASK_NOT_UNDERSTOOD,
        ReferralKind.DISCOVERY_UNAVAILABLE,
        ReferralKind.SEED_DROPPED,
    }
)

# The kinds a referral may carry `candidates` for. `SOURCE_AMBIGUOUS` must
# carry at least two -- see `_names_what_it_is_about`. `PINNED_SOURCE_NOT_FOUND`
# and `NO_DATA` may each carry as few as none: the discovery log holds
# whatever the requirement's latest matching search found, which is empty
# when that search answered `definite` with a single source that was not the
# one pinned, or when `no_data`'s own nearest-series relaxation found nothing.
MAY_CARRY_CANDIDATES = frozenset(
    {ReferralKind.SOURCE_AMBIGUOUS, ReferralKind.PINNED_SOURCE_NOT_FOUND, ReferralKind.NO_DATA}
)


class SourceRecommendation(StrictModel):
    """The translator's ranked pick among a `source_ambiguous` referral's own candidates.

    `spec_id` names which candidate it ranks first; `rationale` is written
    for the person who settles the choice, not only to justify the pick to
    a later model call -- it is the one place the translator's judgement
    reaches a reader rather than another tool.
    """

    spec_id: SpecId
    rationale: str = Field(min_length=1)


class Referral(StrictModel):
    """One reason the plan goes back to the planner, and what it is about."""

    kind: ReferralKind
    reason: str = Field(min_length=1)
    requirement_id: Identifier | None = None
    decision_id: Identifier | None = None
    candidates: tuple[SelectedSource, ...] = ()
    no_data_reason: str | None = None
    recommendation: SourceRecommendation | None = None

    @model_validator(mode="after")
    def _names_what_it_is_about(self) -> Referral:
        about_requirement = self.kind in ABOUT_A_REQUIREMENT
        about_decision = self.kind is ReferralKind.NO_OPERATION_COVERS
        if about_requirement != (self.requirement_id is not None):
            raise ValueError(
                f"a {self.kind.value} referral "
                + (
                    "names the requirement it is about"
                    if about_requirement
                    else "names no requirement"
                )
            )
        if about_decision != (self.decision_id is not None):
            raise ValueError(
                f"a {self.kind.value} referral "
                + ("names the decision it is about" if about_decision else "names no decision")
            )
        if self.kind is ReferralKind.SOURCE_AMBIGUOUS and len(self.candidates) < 2:
            raise ValueError("a source_ambiguous referral carries the candidate sources")
        if self.kind not in MAY_CARRY_CANDIDATES and self.candidates:
            raise ValueError(f"a {self.kind.value} referral carries no candidates")
        if self.kind is not ReferralKind.NO_DATA and self.no_data_reason is not None:
            raise ValueError(f"a {self.kind.value} referral carries no reason code")
        if self.recommendation is not None:
            if self.kind is not ReferralKind.SOURCE_AMBIGUOUS:
                raise ValueError(f"a {self.kind.value} referral carries no recommendation")
            if self.recommendation.spec_id not in {c.spec_id for c in self.candidates}:
                raise ValueError("a recommendation must name one of the referral's own candidates")
        return self


def save_referral(state: SessionState, referral: Referral) -> None:
    """Record `referral` as the session's pending referral."""
    state[REFERRAL_STATE_KEY] = referral.model_dump_json()


def load_referral(state: SessionState) -> Referral | None:
    """Return the session's pending referral, or `None` when none is."""
    raw = state.get(REFERRAL_STATE_KEY)
    return Referral.model_validate_json(raw) if raw else None


def clear_referral(state: SessionState) -> None:
    """Remove the session's pending referral, if one is recorded."""
    state[REFERRAL_STATE_KEY] = None
