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
again cannot search, so retrying cannot succeed. `not_issued` marks a
`discovery_unavailable` referral for a call the planner entry had no time
left to issue (`translator.discovery.NoTimeLeftError`): the service was never
asked, so the entry ran out of time rather than the service failing.
`stop_code` is the run's own code for a stop found before approval, when
reading the newest year a plan's rolling years count from stopped where the
plan's run would (`translator.tool._run_would_stop`).

Four kinds carry `candidates`, for different reasons. `SOURCE_AMBIGUOUS`
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
`REQUIREMENT_UNMEETABLE` carries them only when marked `source_mismatch`: the
requirement states a source that none of the sources discovery found names,
and the candidates are those sources, zero or more, the same ones a
`PINNED_SOURCE_NOT_FOUND` referral offers. Whether to take one in place of
the stated source is the user's choice. A `requirement_unmeetable` referral
for any other cause, a period or coverage the source cannot meet, carries
none, since another source found for the same search would not meet it
either. Every other kind carries neither.

A `pinned_source_not_found` referral also carries, in `stale_pins`, every
other requirement of the same plan revision whose pin discovery did not find,
each with the sources found in its place, so one planner round repairs every
pin a change such as a moved period left stale.

A source found in place of a pin whose own source discovery read and set
aside, because it held the requirement's `across` side closed, is offered
only when it keeps that source's scope (`dropped_scope`), or when a
requirement reading the pinned one reads that side per member. One that drops it,
leaving open a role the pinned source closed to a named group or named
members, is listed under `withheld` with the scope it drops, and never under
`candidates`, so `offered` does not name it and the planner cannot pin it.
"France to Africa", re-searched as "France to every recipient" for a total,
is the case. A rank, a filter or any other reader that reads the side per
member needs every member `across` asks for, so there the pin gives way and
the source is offered: "top donors to Malawi" pinned to the total of official
donors (DAC) is that case (`translator.discovery.DiscoveryLog.scope_kept`).

`save_referral`, `load_referral` and `clear_referral` keep the last referral in
session state, under `REFERRAL_STATE_KEY`, so the planner can read why the
translator handed the plan back without it riding along on the plan itself.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from enum import StrEnum

from compiler.compile import CompileIssueCode
from planner.parties import flow_side, party_role
from planner.plan import SessionState
from planner.records import AmbiguousSourceReading, Identifier, StrictModel, warn_left_out
from planner.tool_results import Outcome
from planner.workflow import SelectedSource
from pydantic import Field, ValidationError, model_validator

logger = logging.getLogger(__name__)

# The session-state slot holding the last `Referral`. One key, since only the
# most recent referral is ever pending -- a new one replaces it.
REFERRAL_STATE_KEY = "translator:referral"


class ReferralKind(StrEnum):
    """Why the translator stopped."""

    SOURCE_AMBIGUOUS = "source_ambiguous"
    SOURCE_NAME_AMBIGUOUS = "source_name_ambiguous"
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
    ReferralKind.SOURCE_NAME_AMBIGUOUS: Outcome.CHOICE_REQUIRED,
    ReferralKind.PINNED_SOURCE_NOT_FOUND: Outcome.CHOICE_REQUIRED,
    ReferralKind.NO_DATA: Outcome.INVALID_INPUT,
    ReferralKind.REQUIREMENT_UNMEETABLE: Outcome.INVALID_INPUT,
    ReferralKind.ASK_NOT_UNDERSTOOD: Outcome.INVALID_INPUT,
    ReferralKind.DISCOVERY_BUDGET_EXHAUSTED: Outcome.INVALID_INPUT,
    ReferralKind.REQUIREMENT_NEEDS_REPAIR: Outcome.INVALID_INPUT,
    ReferralKind.UNSUPPORTED_ASK: Outcome.UNSUPPORTED,
    ReferralKind.NO_OPERATION_COVERS: Outcome.UNSUPPORTED,
    ReferralKind.DISCOVERY_UNAVAILABLE: Outcome.TRANSIENT,
    ReferralKind.SURFACE_UNAVAILABLE: Outcome.TRANSIENT,
    ReferralKind.INTERNAL_FAILURE: Outcome.INTERNAL,
}

# The kinds that hand a requirement back because what it asks for cannot be done
# or names a source that could be several, so what to change is the user's to
# choose. `planner.agent` holds a plan over one of these until the user has
# answered, and `harness.planner_seam` asks the user itself when one carries
# candidates to pick from.
USER_DECIDES = frozenset(
    {
        ReferralKind.REQUIREMENT_UNMEETABLE,
        ReferralKind.UNSUPPORTED_ASK,
        ReferralKind.SOURCE_NAME_AMBIGUOUS,
    }
)


ABOUT_A_REQUIREMENT = frozenset(
    {
        ReferralKind.SOURCE_AMBIGUOUS,
        ReferralKind.SOURCE_NAME_AMBIGUOUS,
        ReferralKind.NO_DATA,
        ReferralKind.UNSUPPORTED_ASK,
        ReferralKind.REQUIREMENT_UNMEETABLE,
        ReferralKind.PINNED_SOURCE_NOT_FOUND,
        ReferralKind.ASK_NOT_UNDERSTOOD,
        ReferralKind.DISCOVERY_UNAVAILABLE,
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
# `REQUIREMENT_UNMEETABLE` carries them only when marked `source_mismatch`.
MAY_CARRY_CANDIDATES = frozenset(
    {
        ReferralKind.SOURCE_AMBIGUOUS,
        ReferralKind.PINNED_SOURCE_NOT_FOUND,
        ReferralKind.NO_DATA,
        ReferralKind.REQUIREMENT_UNMEETABLE,
    }
)


class WithheldSource(StrictModel):
    """A source found in place of a pin that drops the scope the pinned source held.

    `dropped` names each role column whose scope it drops, one sentence a
    column (`dropped_scope`).
    """

    source: SelectedSource
    dropped: tuple[str, ...] = Field(min_length=1)


def dropped_scope(held: SelectedSource, found: SelectedSource) -> tuple[str, ...]:
    """The scope `held`, a pinned dataset's source, holds on each role column that `found` drops.

    A column `held` closes, by binding entities on it or naming a group on
    it, is dropped when `found` leaves it open, when `found` binds on it
    fewer than every entity `held` binds there, or when `held` names its
    members only as a group and `found` names no group of that label there.
    A column `held` leaves open is never dropped: a source that closes it
    narrows the population to what the requirement names. Empty when `found`
    keeps the scope on every column.
    """
    found_open = {axis.column: axis.member_count for axis in found.open_entity_axes or ()}
    held_open = {axis.column for axis in held.open_entity_axes or ()}
    closed = {entity.role for entity in held.entities} | {group.role for group in held.groups}
    dropped: list[str] = []
    for column in sorted(closed - held_open):
        refs = {entity.ref for entity in held.entities if entity.role == column}
        group = next((group for group in held.groups if group.role == column), None)
        if group is not None:
            scope = group.label
        elif len(refs) == 1:
            scope = next(entity.label for entity in held.entities if entity.role == column)
        else:
            scope = f"{len(refs)} named members"
        if column in found_open:
            count = found_open[column]
            over = "every member" if count is None else f"{count} members"
            dropped.append(
                f"{column!r}: the pinned source holds it to {scope}; this one leaves it "
                f"open over {over}"
            )
            continue
        kept = refs & {entity.ref for entity in found.entities if entity.role == column}
        if kept != refs:
            dropped.append(
                f"{column!r}: the pinned source holds it to {scope}; this one binds "
                f"{len(kept)} of those {len(refs)} members"
            )
        elif not refs and not any(
            other.role == column and other.label.casefold() == scope.casefold()
            for other in found.groups
        ):
            dropped.append(
                f"{column!r}: the pinned source holds it to {scope}; this one names no "
                f"{scope} there"
            )
    return tuple(dropped)


def closing_group(held: SelectedSource, across: str) -> str | None:
    """The label of the one group `held` holds the side of a flow the `across` word names to, or `None`.

    `None` when the word names no side of a flow, or `held` names no group,
    or several, on that side.
    """
    side = flow_side(across)
    if side is None:
        return None
    labels = {group.label for group in held.groups if flow_side(group.role) == side}
    return next(iter(labels)) if len(labels) == 1 else None


def scope_repair(held: SelectedSource, across: str) -> str | None:
    """The change to a requirement that restates the scope `held` keeps on the side the `across` word opens, or `None`.

    `held` is a pinned dataset's source set aside because it holds that side
    closed. When it holds the side to one group, or to one entity, the
    requirement can name that member under the side's `PARTY_ROLES` word and
    drop the word from `across`, which brings the pinned source's scope back.
    Several named members, or a word naming no single role, give `None`.
    The repair leaves one member on the side, so it answers only a plan whose
    readers sum the side away (`translator.discovery.DiscoveryLog.scope_kept`).
    """
    role = party_role(across)
    side = flow_side(across)
    if role is None or side is None:
        return None
    groups = {group.label for group in held.groups if flow_side(group.role) == side}
    entities = {entity.label for entity in held.entities if flow_side(entity.role) == side}
    labels = groups or entities
    if len(labels) != 1:
        return None
    (label,) = labels
    return f"name {label!r} as {role} in `entities` and drop {across!r} from `across`"


class StalePin(StrictModel):
    """A requirement whose pin discovery did not find, and the sources found in its place.

    `withheld` are the sources found that drop the scope its pinned source
    held (`WithheldSource`), which it may not pin, and `repair` the change
    to the requirement that keeps that scope (`scope_repair`), when one does.
    """

    requirement_id: Identifier
    candidates: tuple[SelectedSource, ...] = ()
    withheld: tuple[WithheldSource, ...] = ()
    repair: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def _repairs_a_withheld_scope(self) -> StalePin:
        if self.repair is not None and not self.withheld:
            raise ValueError("a stale pin states a repair only for a scope it withholds")
        return self


class Referral(StrictModel):
    """One reason the plan goes back to the planner, and what it is about."""

    kind: ReferralKind
    reason: str = Field(min_length=1)
    requirement_id: Identifier | None = None
    candidates: tuple[SelectedSource, ...] = ()
    no_data_reason: str | None = None
    ambiguous_sources: tuple[AmbiguousSourceReading, ...] = ()
    constraint: CompileIssueCode | None = None
    repair: str | None = Field(default=None, min_length=1)
    reference: str | None = Field(default=None, min_length=1)
    repeated: bool = False
    budget_spent: bool = False
    not_issued: bool = False
    source_mismatch: bool = False
    stop_code: str | None = Field(default=None, min_length=1)
    stale_pins: tuple[StalePin, ...] = ()
    withheld: tuple[WithheldSource, ...] = ()

    @property
    def outcome(self) -> Outcome:
        """What this referral asks of the planner (`OUTCOME_OF`)."""
        return OUTCOME_OF[self.kind]

    def offered(self) -> dict[str, tuple[SelectedSource, ...]]:
        """The sources each requirement this referral is about may pin, by requirement id.

        Its own `requirement_id` with `candidates`, then each of `stale_pins`.
        A source under `withheld` is never offered.
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
        elif self.constraint is not None:
            raise ValueError(f"a {self.kind.value} referral names no constraint")
        elif self.repair is not None and not self.withheld:
            raise ValueError(
                f"a {self.kind.value} referral names a repair only for a scope it withholds"
            )
        if (self.kind is ReferralKind.INTERNAL_FAILURE) != (self.reference is not None):
            raise ValueError(
                f"a {self.kind.value} referral "
                + (
                    "carries its diagnostic reference"
                    if self.kind is ReferralKind.INTERNAL_FAILURE
                    else "carries no diagnostic reference"
                )
            )
        if self.kind is ReferralKind.SOURCE_NAME_AMBIGUOUS:
            if not self.ambiguous_sources or any(
                len(group.candidates) < 2 for group in self.ambiguous_sources
            ):
                raise ValueError(
                    "a source_name_ambiguous referral carries the offered source references"
                )
        elif self.ambiguous_sources:
            raise ValueError("only a source_name_ambiguous referral carries ambiguous_sources")
        if self.kind is ReferralKind.SOURCE_AMBIGUOUS and len(self.candidates) < 2:
            raise ValueError("a source_ambiguous referral carries the candidate sources")
        if self.kind not in MAY_CARRY_CANDIDATES and self.candidates:
            raise ValueError(f"a {self.kind.value} referral carries no candidates")
        if self.source_mismatch and self.kind is not ReferralKind.REQUIREMENT_UNMEETABLE:
            raise ValueError(f"a {self.kind.value} referral is never marked source_mismatch")
        if (
            self.kind is ReferralKind.REQUIREMENT_UNMEETABLE
            and self.candidates
            and not self.source_mismatch
        ):
            raise ValueError(
                "a requirement_unmeetable referral carries candidates only for a stated "
                "source the sources found do not name"
            )
        if self.kind is not ReferralKind.NO_DATA and self.no_data_reason is not None:
            raise ValueError(f"a {self.kind.value} referral carries no reason code")
        if self.stale_pins and self.kind is not ReferralKind.PINNED_SOURCE_NOT_FOUND:
            raise ValueError(f"a {self.kind.value} referral carries no stale pins")
        if self.withheld and self.kind is not ReferralKind.PINNED_SOURCE_NOT_FOUND:
            raise ValueError(f"a {self.kind.value} referral withholds no sources")
        if self.budget_spent and self.kind is not ReferralKind.DISCOVERY_UNAVAILABLE:
            raise ValueError(f"a {self.kind.value} referral is never marked budget_spent")
        if self.not_issued and self.kind is not ReferralKind.DISCOVERY_UNAVAILABLE:
            raise ValueError(f"a {self.kind.value} referral is never marked not_issued")
        return self


def save_referral(state: SessionState, referral: Referral) -> None:
    """Record `referral` as the session's pending referral."""
    state[REFERRAL_STATE_KEY] = referral.model_dump_json()


def load_referral(state: SessionState) -> Referral | None:
    """Return the session's pending referral, or `None` when none is.

    A referral stored in a shape `Referral` rejects, one a release with
    other fields wrote, is no referral, and the omission is logged. The
    sources it offered are in that stored shape too, so nothing can be
    pinned from it: the plan is translated again and discovery refers it
    back with what Search answers now.
    """
    raw = state.get(REFERRAL_STATE_KEY)
    if not raw:
        return None
    try:
        return Referral.model_validate_json(raw)
    except ValidationError as exc:
        warn_left_out(logger, REFERRAL_STATE_KEY, [exc])
        return None


def clear_referral(state: SessionState) -> None:
    """Remove the session's pending referral, if one is recorded."""
    state[REFERRAL_STATE_KEY] = None
