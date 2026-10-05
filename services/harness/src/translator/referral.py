"""What the translator hands back when a plan cannot be carried out as stated.

A referral is typed because the planner has to act on it: a choice of source
is a decision to settle with the user, a requirement no data meets is one to
amend, a calculation no operation performs is one to decline, a data service
that would not answer is one to retry later rather than a fact about the
requirement's own content. `candidates` is filled by code from what
discovery recorded, never proposed by a model.

Every kind lands on one of five outcomes, `OUTCOME_OF`, from the planner's
own `Outcome` vocabulary: `choice_required`, `invalid_input` (input the
planner can repair), `unsupported`, `transient` and `internal`. A compiled
plan, the sixth, is no referral at all. `requirement_needs_repair` names the
compile issue code the requirement breaks as `constraint` and what the
planner may change as `repair`; `internal_failure` carries the `reference`
its log line is written under. `repeated` marks a referral for a translation
that already failed with the same issues earlier in the session
(`translator.mechanical`'s loop guard). The plan was not changed where it
failed. `budget_spent` marks a `discovery_unavailable` referral made once
a discovery budget is spent and the most recent call that spent it timed
out (`translator.discovery._budget_referral`). Translating the same revision
again cannot search, so retrying cannot succeed.

Three kinds carry `candidates`, for different reasons. `SOURCE_AMBIGUOUS`
always carries two or more: discovery found a real choice. The planner
records it as a source decision naming one of them, or puts the candidates
to the user. `PINNED_SOURCE_NOT_FOUND` carries
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

A `pinned_source_not_found` referral also carries, in `stale_pins`, every
other requirement of the same plan revision whose pin discovery did not find,
each with the sources found in its place, so one planner round repairs every
pin a change such as a moved period left stale.

`save_referral`, `load_referral` and `clear_referral` keep the last referral in
session state, under `REFERRAL_STATE_KEY`, so the planner can read why the
translator handed the plan back without it riding along on the plan itself.
"""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum

from compiler.compile import CompileIssueCode
from planner.plan import SessionState
from planner.records import Identifier, StrictModel
from planner.tool_results import Outcome
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
    # A compile issue the planner repairs in the requirement it names, carrying
    # the issue's code as `constraint` and the permitted change as `repair`.
    REQUIREMENT_NEEDS_REPAIR = "requirement_needs_repair"
    # The tool surface could not be listed to compile against: a fact about the
    # services right now.
    SURFACE_UNAVAILABLE = "surface_unavailable"
    # A fault in the translator or the services' configuration that no change to
    # the plan repairs, logged under the referral's `reference`.
    INTERNAL_FAILURE = "internal_failure"


# What each kind asks of the planner, in the planner's own `Outcome` vocabulary.
OUTCOME_OF: Mapping[ReferralKind, Outcome] = {
    ReferralKind.SOURCE_AMBIGUOUS: Outcome.CHOICE_REQUIRED,
    ReferralKind.PINNED_SOURCE_NOT_FOUND: Outcome.CHOICE_REQUIRED,
    ReferralKind.NO_DATA: Outcome.INVALID_INPUT,
    ReferralKind.REQUIREMENT_UNMEETABLE: Outcome.INVALID_INPUT,
    ReferralKind.ASK_NOT_UNDERSTOOD: Outcome.INVALID_INPUT,
    ReferralKind.DISCOVERY_BUDGET_EXHAUSTED: Outcome.INVALID_INPUT,
    ReferralKind.SEED_DROPPED: Outcome.INVALID_INPUT,
    ReferralKind.REQUIREMENT_NEEDS_REPAIR: Outcome.INVALID_INPUT,
    ReferralKind.UNSUPPORTED_ASK: Outcome.UNSUPPORTED,
    ReferralKind.NO_OPERATION_COVERS: Outcome.UNSUPPORTED,
    ReferralKind.DISCOVERY_UNAVAILABLE: Outcome.TRANSIENT,
    ReferralKind.SURFACE_UNAVAILABLE: Outcome.TRANSIENT,
    ReferralKind.INTERNAL_FAILURE: Outcome.INTERNAL,
}


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
        ReferralKind.NO_OPERATION_COVERS,
        ReferralKind.REQUIREMENT_NEEDS_REPAIR,
    }
)

# The kinds a referral may carry `candidates` for. `SOURCE_AMBIGUOUS` must
# carry at least two -- see `_names_what_it_is_about`. `PINNED_SOURCE_NOT_FOUND`
# and `NO_DATA` may each carry as few as none: the discovery log holds
# whatever the requirement's latest matching search found, which is empty
# when that search found no source a pin may be taken from, or when
# `no_data`'s own nearest-series relaxation found nothing.
MAY_CARRY_CANDIDATES = frozenset(
    {ReferralKind.SOURCE_AMBIGUOUS, ReferralKind.PINNED_SOURCE_NOT_FOUND, ReferralKind.NO_DATA}
)


class StalePin(StrictModel):
    """A requirement whose pin discovery did not find, and the sources found in its place."""

    requirement_id: Identifier
    candidates: tuple[SelectedSource, ...] = ()


class Referral(StrictModel):
    """One reason the plan goes back to the planner, and what it is about."""

    kind: ReferralKind
    reason: str = Field(min_length=1)
    requirement_id: Identifier | None = None
    candidates: tuple[SelectedSource, ...] = ()
    no_data_reason: str | None = None
    constraint: CompileIssueCode | None = None
    repair: str | None = Field(default=None, min_length=1)
    reference: str | None = Field(default=None, min_length=1)
    repeated: bool = False
    budget_spent: bool = False
    stale_pins: tuple[StalePin, ...] = ()

    @property
    def outcome(self) -> Outcome:
        """What this referral asks of the planner (`OUTCOME_OF`)."""
        return OUTCOME_OF[self.kind]

    def offered(self) -> dict[str, tuple[SelectedSource, ...]]:
        """The sources each requirement this referral is about may pin, by requirement id.

        Its own `requirement_id` with `candidates`, then each of `stale_pins`.
        """
        offers = {pin.requirement_id: pin.candidates for pin in self.stale_pins}
        if self.requirement_id is not None:
            offers = {self.requirement_id: self.candidates, **offers}
        return offers

    @model_validator(mode="after")
    def _names_what_it_is_about(self) -> Referral:
        about_requirement = self.kind in ABOUT_A_REQUIREMENT
        if about_requirement != (self.requirement_id is not None):
            raise ValueError(
                f"a {self.kind.value} referral "
                + (
                    "names the requirement it is about"
                    if about_requirement
                    else "names no requirement"
                )
            )
        if self.kind is ReferralKind.REQUIREMENT_NEEDS_REPAIR:
            if self.constraint is None or self.repair is None:
                raise ValueError(
                    f"a {self.kind.value} referral names the constraint and the permitted repair"
                )
        elif self.constraint is not None or self.repair is not None:
            raise ValueError(f"a {self.kind.value} referral names no constraint or repair")
        if (self.kind is ReferralKind.INTERNAL_FAILURE) != (self.reference is not None):
            raise ValueError(
                f"a {self.kind.value} referral "
                + (
                    "carries its diagnostic reference"
                    if self.kind is ReferralKind.INTERNAL_FAILURE
                    else "carries no diagnostic reference"
                )
            )
        if self.kind is ReferralKind.SOURCE_AMBIGUOUS and len(self.candidates) < 2:
            raise ValueError("a source_ambiguous referral carries the candidate sources")
        if self.kind not in MAY_CARRY_CANDIDATES and self.candidates:
            raise ValueError(f"a {self.kind.value} referral carries no candidates")
        if self.kind is not ReferralKind.NO_DATA and self.no_data_reason is not None:
            raise ValueError(f"a {self.kind.value} referral carries no reason code")
        if self.stale_pins and self.kind is not ReferralKind.PINNED_SOURCE_NOT_FOUND:
            raise ValueError(f"a {self.kind.value} referral carries no stale pins")
        if self.budget_spent and self.kind is not ReferralKind.DISCOVERY_UNAVAILABLE:
            raise ValueError(f"a {self.kind.value} referral is never marked budget_spent")
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
