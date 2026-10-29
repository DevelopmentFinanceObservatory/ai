"""The plan object: method specification, control surface, and record in one.

A `Plan` states the question, each `Decision` the planner made (or proposed)
with the alternatives it considered and who resolved it, and the
`EvidenceRequirement`s those decisions fix. A dataset requirement is evidence
for a calculation the plan states -- the series a retrieved result rests on,
stated in fields code can check a retrieved frame against -- and every
calculation is itself a typed requirement (aggregate, growth, comparison,
rank, share of, quotient of, join, filter), citing the decisions that fixed it, if any
were weighed, and naming the requirements it reads. Once a plan states
requirements, every decision has to be cited by at least one of them, which
is how the plan shows what each decision is for.

A plan is filled in stages, and every stage returns a new instance. The
planner records the question, the decisions and the requirements, leaving a
`DRAFT`. The translator's `Translation` -- the operations that resolve the
requirements -- and the `Workflow` the compiler builds from it arrive together
through `with_translation`, which is what moves a `DRAFT` to
`PENDING_APPROVAL`, the first status the user sees. That workflow is what the
executor runs.

Every model here is a `StrictModel`: frozen, and closed to fields it does not
declare. A revision is therefore a value. `plan_revision` hashes one,
`approve` records that hash on the plan, and `start_execution` refuses a plan
whose content no longer matches what was approved.

`Plan.amend()` returns a new `Plan` with `version` incremented, so the original
stays in the record and a reviewer can reconstruct what changed. An amendment
is a patch. It names the decisions and requirements that change, by `slot` and
by `id`, and carries forward every item it does not mention. `removed_decisions`
and `removed_evidence_requirements` are the deletion channel, since an item
absent from a patch means unchanged. Whether a given revision amends at all is
`planner.agent`'s call: a call that states the version it amends routes through
`amend()`, while a call stating `starts_new_plan` begins a fresh `Plan` at
version 1 (see `_resolve_intent` there) -- unless the planner was built with
`question_fixed=True`, where a restart is refused instead of ever reaching
this module, and an amendment's own `question` is held to the one already
recorded.
"""

from __future__ import annotations

import functools
import hashlib
import itertools
import json
import logging
import re
from collections.abc import Callable, Collection, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime
from typing import Annotated, Any, Literal, NamedTuple, Protocol, get_args

import rfc8785
from common.hashed_fields import HASHED_WHEN_SET, revision_content
from compiler.sources import source_from_spec_block
from data_mcp.definition_store import canonicalize, spec_id_for
from data_mcp.definition_writes import windowed_identity
from data_mcp.scope import ChoiceProvenance, ScopeAxis
from pydantic import Field, ValidationError, field_validator, model_validator

from planner.operations import Translation
from planner.parties import (
    PARTY_ROLES,
    Party,
    PartyRole,
    across_party,
    party_among,
    role_party,
    same_party,
    singular,
    words_for_one_party,
)
from planner.records import (
    MAX_ENTITIES,
    MAX_ENTITY_NAME_CHARS,
    MAX_ENTITY_ROLE_CHARS,
    Breakdown,
    DenominatorScope,
    EntityIn,
    GroupBy,
    GrowthChange,
    Identifier,
    OpenPeriodFault,
    OptionStatus,
    Period,
    PeriodThreshold,
    PlaceLevel,
    Predicate,
    RankAmong,
    RankDirection,
    RankLimit,
    Reduction,
    RelativePeriod,
    ResolvedSourceMention,
    RollingIssue,
    RollingPlanError,
    RollingWindow,
    SpecId,
    StatVarReading,
    StrictModel,
    TaggedModel,
    TitlelessEnum,
    Year,
    check_rolling_window,
    warn_left_out,
)
from planner.shapes import (
    EndDrift,
    FrameShape,
    PeriodDrift,
    RoleDimension,
    ShapeVerb,
    TimeLabel,
    aggregated,
    anchor_misses,
    anchor_year_stays,
    compared,
    filtered,
    grown,
    growth_endpoint_leaves,
    joined,
    level_filter,
    open_on_whole_side,
    open_whole_side_reason,
    place_role,
    quotiented,
    ranked,
    retrieved,
    role_named,
    roles_named,
    shared,
    summed_sides,
    window_outside,
    windowed,
    year_filter,
    years_run_out,
)
from planner.shapes import rejects as shape_rejects
from planner.workflow import ResolvedGroup, Workflow

logger = logging.getLogger(__name__)

# The session-state slot holding the current `Plan`. One key, because a plan is
# filled in stages by more than one agent and each stage reads the last.
PLAN_STATE_KEY = "planner:plan"

# The session-state slot counting how many plans the planner has proposed in its
# current entry (see `MAX_PLAN_ROUNDS`). Kept beside the plan rather than on it:
# a count on `Plan` would enter `plan_revision`, and every re-proposal would
# change the very hash it counts.
PLAN_ROUNDS_STATE_KEY = "planner:rounds"

# The session-state slot holding the `UnsupportedOperation` of every
# `unsupported` answer `propose_plan` or its `translation` gave in the
# planner's current entry, each of which the recorded plan has to decline.
# Cleared with `PLAN_ROUNDS_STATE_KEY` when an entry starts, so an
# ask-the-user resume inside one entry keeps it.
UNSUPPORTED_STATE_KEY = "planner:unsupported"

# The session-state slot saying whether the planner's current entry opened
# with the user's own amendment of the plan at the approval card.
# `harness.planner_seam.PlannerSeam` writes it with `PLAN_ROUNDS_STATE_KEY`
# when an entry starts, so an ask-the-user resume inside one entry keeps it.
# Unset, the entry carries no amendment.
AMENDMENT_ENTRY_STATE_KEY = "planner:amendment_entry"

# The session-state slot holding the option the user picked for a planner
# decision on the plan card, as a `DecisionPick` dump, when the planner's
# current entry replans to hold that pick. `harness.planner_seam.PlannerSeam`
# writes it, or clears it, with `AMENDMENT_ENTRY_STATE_KEY` when an entry
# starts, so an ask-the-user resume inside one entry keeps it. Admission
# refuses a plan that does not state the pick (`submission.service.admit`).
DECISION_PICK_STATE_KEY = "planner:decision_pick"

# The session-state slot holding the `UnsupportedOperation` of every decline
# earned in the planner's current entry that the recorded plan may make but
# need not: a first-draft entry or a translation that refers a requirement
# back as unmeetable (`translator.tool._earned_decline`) earns one. It holds for the rest of the
# entry, whatever later plans change, and no plan owes it. Cleared and kept
# with `UNSUPPORTED_STATE_KEY`.
OPTIONAL_DECLINES_STATE_KEY = "planner:optional_declines"

# Optional declines that may end the entry when every ask is declined and
# no evidence requirements remain, including first-draft asks no kind states.
TERMINAL_DECLINES_STATE_KEY = "planner:terminal_declines"

# The asks a `period_reduction_unstated` refusal named in the planner's
# current entry, by number. No optional decline covers one of them
# (`submission.service._declinable`). Cleared and kept with
# `UNSUPPORTED_STATE_KEY`.
REDUCTION_REFUSED_ASKS_STATE_KEY = "planner:reduction_refused_asks"

# The session-state slot holding a `PinSource` for every spec_id the
# planner was shown this entry, keyed by spec_id in the order it was shown
# (`pin_sources`). `harness.planner_seam` writes it from the search
# readings its first message renders, one entry per spec_id they name.
# `propose_plan` (`planner.agent`) reads a settled entry's measure choice and
# the identity its block carries, and `translator.discovery.discover_source`
# builds a pinned requirement's source from an entry with no `search` call.
PIN_SOURCES_STATE_KEY = "planner:pin_sources"

# The session-state slot the host writes a seeded turn's seeds to, as JSON:
# one entry per Search result a Studio handoff carries (`TurnRequest.seed_spec`),
# as `harness.seeds.seeds_slot` compacts it, a `harness.wire.Seed` or the
# refusal its step fails with. `harness.graph`'s `seed` node reads each entry
# and clears the slot. Unset on a turn with no seeds.
SEEDS_STATE_KEY = "seeds"

# The session-state slot holding a `WindowedPin` for every pin of a dataset
# whose identity a search in the conversation recorded
# (`recorded_identity`), that the plan `planner.agent.propose_plan` last
# recorded moves to that identity over a dataset requirement's own period.
# `translator.discovery.discover_source` takes such a requirement's source
# from Search's answer for the copy's identity, with no `search` call. Each
# is stored whole, its identity included.
# Unset until a plan pins such a copy.
WINDOWED_PINS_STATE_KEY = "planner:windowed_pins"

# The session-state slot holding what `planner.agent.propose_plan` read about
# every year a pinned dataset holds, by the pin's spec_id, as its first year and
# its rows, or null when Search's answer over no window stated no first year.
# One read answers every requirement and every later proposal of the turn that
# pins the dataset. Unset until a start that follows the newest data needs one.
FULL_HISTORIES_STATE_KEY = "planner:full_histories"

# The session-state slot holding the outputs the user asked for, as a JSON
# list of strings numbered by position, at most `MAX_PLAN_ASKS` of at most
# `MAX_ASK_CHARS` characters each (`load_asks`). The orchestrator's
# `plan_start` node writes it beside the brief on every plan call, and the
# `plan` tool refuses a call that lists none
# (`harness.orchestrator_tools._asks_refusal`). `harness.planner_seam` lists the asks in the
# planner's first message, and `planner.agent.propose_plan` copies them onto
# every plan it records, so `Plan` refuses one that does not account for each
# exactly once. Unset, a plan carries no asks and no ask is accounted for.
ASKS_STATE_KEY = "orchestrator:asks"

# The session-state slot holding the parties the asks break down by, as a JSON
# list of `AskBreakdown`, one per ask that names any (`load_ask_breakdowns`).
# `plan_start` writes it beside `ASKS_STATE_KEY`, and `propose_plan` copies it
# onto every plan it records beside the asks. Unset, no ask breaks down by a
# party.
ASK_BREAKDOWNS_STATE_KEY = "orchestrator:ask_breakdowns"

# The session-state slot holding the numbers of the asks the user put to one
# figure, or one ranking, over several years taken together, as a JSON list of
# ints (`load_asks_across_years`). `plan_start` writes it beside
# `ASKS_STATE_KEY`, and `propose_plan` copies it onto every plan it records
# beside the asks. Unset, no ask is across years.
ASKS_ACROSS_YEARS_STATE_KEY = "orchestrator:asks_across_years"

# How many plans one entry into the planner may propose -- a first draft and
# three more, whether each patches the recorded plan or restarts. Each proposal
# opens a fresh discovery budget, so without a cap a refer-and-repropose loop
# is unbounded in calls and spend. The count starts over on every entry into
# the planner (a new turn, a user amendment, a replan), while `Plan.version`
# keeps climbing across entries, so a plan's version can exceed this.
MAX_PLAN_ROUNDS = 4

# The most decisions one plan may state. A plan is something a person approves
# from a card, and a decision is only recorded where the planner genuinely
# weighed alternatives; the largest plan any fixture in this repository builds
# states three, and `translator.discovery`'s own budget comment records typical
# plans as carrying one to three requirements. The cap leaves that headroom by
# an order of magnitude and still bounds what a single model call can draft.
MAX_PLAN_DECISIONS = 24

# The most evidence requirements one plan may state. A plan that spends
# `translator.discovery.DISCOVERY_CALL_BUDGET` (8, one search per dataset
# requirement) carries about eight dataset requirements and about as many
# calculations over them; the realistic upper end the seam brief is sized
# from is nine requirements in total. Sixteen dataset requirements with their
# calculations is roughly twice the budget, so a plan at the cap is refused
# for its size rather than for a discovery budget it can never spend.
MAX_PLAN_REQUIREMENTS = 24

# The most requirements a plan holds: the model writes at most
# `MAX_PLAN_REQUIREMENTS` (`Plan._record_fits_the_size_limit`), and
# `whole_history_split` copies each of them at most once.
MAX_HELD_REQUIREMENTS = 2 * MAX_PLAN_REQUIREMENTS

# The most asks one brief may list, and the longest each may run. An ask is a
# short line naming one output the user asked for, such as a chart or a
# figure.
MAX_PLAN_ASKS = 6
MAX_ASK_CHARS = 300

# Length limits on the plan's free text, and count limits on its collections.
# Each rides on the JSON schema as `maxLength` or `maxItems`, so the model reads
# it from the declaration and no field description restates one. Each sits
# several times above the longest value in the plans the eval sets under `evals/`
# record, and every one of those plans validates. A plan a limit refuses comes
# back to the model as an invalid plan naming the field, without spending a
# proposal round. The entity limits sit in `planner.records` beside `EntityIn`,
# which shares them. Per-field limits multiply through the nesting (a decision's
# `fixes` can list fifty entities and eight predicates of fifty names), so
# `MAX_PLAN_RECORD_CHARS` below bounds the plan as a whole.
#
# The question limit sits well above a question a person types. `Plan.question`
# restates the user's, and `cut_user_text` shortens a longer one before it
# reaches a plan or the brief.
MAX_QUESTION_CHARS = 2_000
MAX_DECISION_DESCRIPTION_CHARS = 500
MAX_DECISION_RATIONALE_CHARS = 800
# A decision's options list every candidate the planner weighed, among them each
# candidate a referral offers.
MAX_DECISION_OPTIONS = 24
MAX_OPTION_CHARS = 200
MAX_MEASURE_CHARS = 200
# The most characters `search` takes for one source's wording, so a plan never
# holds a source `search` would refuse.
MAX_SOURCE_CHARS = 120
# The most characters `search` takes for a publisher's series code.
MAX_SOURCE_CODE_CHARS = 64
MAX_PREDICATES = 8
MAX_GROUP_BY = 2

# The most characters the model-authored part of one plan may serialise to: its
# `question`, `decisions` and `evidence_requirements`, as compact JSON with
# non-ASCII characters left as written. The count is raw characters, which is
# what `harness.planner_seam` renders: free text as it is, and entities,
# predicates and a decision's `fixes` as JSON in the same encoding. The brief
# renders these fields in full, so this is what bounds the plan section to this
# many characters plus the render's own line framing. The largest plan in the
# eval sets serialises to 2,244 characters and the nine-requirement plan
# `tests/test_context_budget.py` sizes the brief from to 7,992; the limit is 7
# times the first and twice the second. `translation`, `workflow`,
# `approved_revision` and `approved_at` are the compiler's and `approve`'s, and
# do not count, so a compiled plan re-validates however large its workflow.
MAX_PLAN_RECORD_CHARS = 16_000

# What `cut_user_text` ends a cut text with, so a reader of the result can tell
# a cut from a short text.
CUT_MARKER = "[cut]"


def cut_user_text(text: str, cap: int) -> str:
    """Return `text` when it is at most `cap` characters, else a prefix of it ending in `CUT_MARKER`.

    The result is never longer than `cap`. For text the user wrote and a
    bounded field or a prompt line has to carry, where refusing it would
    end the turn over its length.
    """
    if len(text) <= cap:
        return text
    return (text[: max(cap - len(CUT_MARKER), 0)] + CUT_MARKER)[:cap]


# One word of a requirement's `across`, a party as the party word alone. A
# requirement states each party once, so it holds at most one word per
# `planner.parties.Party`.
AcrossWord = Annotated[str, Field(min_length=1, max_length=MAX_ENTITY_ROLE_CHARS)]
MAX_ACROSS_WORDS = len(get_args(Party))


# `role` is one of `PARTY_ROLES` in what a model writes: `propose_plan`
# refuses any other word. A plan read back from session state is not held to
# it, so one recorded with an older role word still loads. The docstring is
# the schema's description, so this note stays out of it.
class Entity(StrictModel):
    """Who or where a requirement is about, in the words of the question."""

    name: str = Field(
        min_length=1,
        max_length=MAX_ENTITY_NAME_CHARS,
        description="The entity as the question names it, in its own words.",
    )
    role: str | None = Field(
        default=None,
        min_length=1,
        max_length=MAX_ENTITY_ROLE_CHARS,
        description=(
            "This entity's side of a flow of aid, lending and debt, or trade. Unset for a "
            "place a measure is about, and unless the question or measure identifies it."
        ),
        json_schema_extra={"enum": list(PARTY_ROLES)},
    )
    # Left out of every dump while unset, so a plan naming no set's members
    # keeps its serialised form, rendering and size against
    # `MAX_PLAN_RECORD_CHARS`.
    members: Annotated[bool, HASHED_WHEN_SET] = Field(
        default=False,
        exclude_if=lambda members: not members,
        description=(
            "Set when `name` is a set, such as a region or a group of countries or donors, "
            "and the question asks about its members one by one: 'African countries' is "
            "'Africa' with `members` set. Unset, a set named alone is one entity, its own "
            "figure. A set is named once, never listed member by member."
        ),
    )
    dcid: Annotated[str | None, HASHED_WHEN_SET] = Field(
        default=None,
        min_length=1,
        max_length=200,
        exclude_if=lambda dcid: dcid is None,
        description="Graph node from an entity_name_ambiguous referral.",
    )


# Stated in the frame grammar's own `currency`, `prices` and `base_year`,
# `current` or `constant`, so a retrieved frame's unit compares to a
# requirement field by field.
class Unit(StrictModel):
    """The unit an acceptable result is stated in."""

    currency: str = Field(
        pattern=r"^[A-Z]{3}$",
        description="The three-letter currency code an acceptable result is denominated in.",
    )
    prices: Literal["current", "constant"] = Field(
        description=(
            "Whether an acceptable result holds prices at the time observed ('current') or with "
            "inflation removed to one reference year ('constant')."
        )
    )
    base_year: Year | None = Field(
        default=None,
        description="The reference year constant prices are expressed in; unset for current prices.",
    )

    @model_validator(mode="after")
    def _base_year_goes_with_constant_prices(self) -> Unit:
        if self.prices == "constant" and self.base_year is None:
            raise ValueError("constant prices need a base_year")
        if self.prices == "current" and self.base_year is not None:
            raise ValueError("current prices take no base_year")
        return self


class Coverage(StrictModel):
    """What an acceptable result does when its evidence cannot be confirmed."""

    on_missing: Literal["report_gaps", "fail"] = Field(
        default="report_gaps",
        description=(
            "What to do when a year or entity's coverage cannot be confirmed: 'report_gaps' "
            "accepts the result and records the gap, 'fail' refuses it."
        ),
    )


class ResolvedBy(TitlelessEnum):
    """Who settled a `Decision`."""

    USER = "user"
    PLANNER_DEFAULT = "planner_default"


class DecisionFixes(StrictModel):
    """The requirement fields a `Decision` fixes, with their values.

    `settle_decisions` writes it from the field names a `DecisionDraft`
    lists, reading each value off the requirements that cite the decision.
    `propose_plan` sets `source_spec_id` on the measure and scope decisions
    it writes for a pin (`submission.service._measure_decisions`,
    `_scope_decisions`) and moves it with the pin
    (`submission.service._repinned`). Every requirement citing a decision
    states each value it fixes, `source_spec_id` aside (`fix_disagreements`).
    """

    period: Period | None = Field(
        default=None, description="The period this decision fixes, when it fixes one."
    )
    unit: Unit | None = Field(
        default=None, description="The unit this decision fixes, when it fixes one."
    )
    entities: tuple[Entity, ...] | None = Field(
        default=None,
        min_length=1,
        max_length=MAX_ENTITIES,
        description="The entities this decision fixes, when it fixes them.",
    )
    across: Annotated[tuple[AcrossWord, ...] | None, HASHED_WHEN_SET] = Field(
        default=None,
        min_length=1,
        max_length=MAX_ACROSS_WORDS,
        description="The parties whose every member this decision fixes a requirement to cover, when it fixes them.",
    )
    measure: str | None = Field(
        default=None,
        min_length=1,
        max_length=MAX_MEASURE_CHARS,
        description="The measure this decision fixes, when it fixes one.",
    )
    source_spec_id: SpecId | None = Field(
        default=None,
        description="The dataset definition id this decision pins, when it pins a source.",
    )
    stat_vars: StatVarReading | None = Field(
        default=None,
        description=(
            "Whether a family of StatVars this decision fixes is read summed ('sum') or as "
            "separate series ('separate'), when it fixes that."
        ),
    )
    reduction: Reduction | None = Field(
        default=None,
        description="The reduction (sum, mean, max or min) this decision fixes, when it fixes one.",
    )
    group_by: tuple[GroupBy, ...] | None = Field(
        default=None,
        max_length=MAX_GROUP_BY,
        description=(
            "The group_by columns this decision fixes, when it fixes them; empty for one "
            "figure over every row."
        ),
    )
    window_start: Year | None = Field(
        default=None, description="The window's start year this decision fixes, when it fixes one."
    )
    window_end: Year | None = Field(
        default=None, description="The window's end year this decision fixes, when it fixes one."
    )
    direction: RankDirection | None = Field(
        default=None,
        description="The rank direction (highest or lowest first) this decision fixes, when it fixes one.",
    )
    predicates: tuple[Predicate, ...] | None = Field(
        default=None,
        min_length=1,
        max_length=MAX_PREDICATES,
        description="The predicates this decision fixes, when it fixes them.",
    )
    among: RankAmong | None = Field(
        default=None,
        description=(
            "Whether a rank this decision fixes compares named entities or the whole "
            "population, when it fixes that."
        ),
    )
    axis: str | None = Field(
        default=None,
        min_length=1,
        max_length=MAX_ENTITY_ROLE_CHARS,
        description="The role dimension this decision fixes, when it fixes one.",
    )
    denominator_scope: DenominatorScope | None = Field(
        default=None,
        description=(
            "Whether a share_of this decision fixes divides per axis key or by one pooled "
            "total, when it fixes that."
        ),
    )
    change: GrowthChange | None = Field(
        default=None,
        description=(
            "Whether a growth this decision fixes reads a relative ratio or an absolute "
            "difference, when it fixes that."
        ),
    )
    per_period: bool | None = Field(
        default=None,
        description="Whether a rank this decision fixes ranks each period on its own, when it fixes that.",
    )
    limit: Annotated[RankLimit | None, HASHED_WHEN_SET] = Field(
        default=None,
        description="How many places a rank this decision fixes keeps, when it fixes that.",
    )


class _Choice(StrictModel):
    """What a `Decision` and a `DecisionDraft` share: the choice, its alternatives and its reason."""

    slot: Identifier = Field(
        description="This decision's name; unique within a plan, so a requirement can cite it."
    )
    description: str = Field(
        min_length=1,
        max_length=MAX_DECISION_DESCRIPTION_CHARS,
        description="What choice this decision states, in plain words.",
    )
    options: tuple[Annotated[str, Field(max_length=MAX_OPTION_CHARS)], ...] = Field(
        max_length=MAX_DECISION_OPTIONS,
        description=(
            "The alternatives considered for this decision; at least two, since one option "
            "is no choice at all."
        ),
    )
    chosen: str = Field(
        min_length=1,
        max_length=MAX_OPTION_CHARS,
        description="Which of `options` this decision settles on.",
    )
    resolved_by: ResolvedBy = Field(
        description="Who settled this decision: the user, or the planner's own default."
    )
    rationale: str = Field(
        min_length=1,
        max_length=MAX_DECISION_RATIONALE_CHARS,
        description=(
            "The one-line reason `chosen` was picked over the other options; for a "
            "user's choice, saying so is enough."
        ),
    )

    @model_validator(mode="after")
    def _options_are_a_real_alternative(self) -> _Choice:
        """Reject a `Decision` with fewer than two distinct options.

        A decision whose only option is what was chosen records no decision:
        the reviewer sees a choice with nothing it was chosen over. The same
        option listed twice is one option dressed as two. The message tells
        the caller what to do about it, not only what the rule is, since a
        caller retrying past this error is prone to inventing a second,
        uninvestigated option rather than dropping the decision, recording a
        choice that was never actually weighed, worse than recording none.
        """
        if len(self.options) < 2:
            raise ValueError(
                f"a decision states at least two options to choose between; got {list(self.options)!r}. "
                "If nothing else was genuinely weighed, delete this decision rather than inventing a "
                "second option to pass this check -- a value the question already states outright is "
                "not a decision at all and needs no `Decision` entry."
            )
        _reject_repeats(self.options, what="decision option")
        return self

    @model_validator(mode="after")
    def _chosen_is_an_option(self) -> _Choice:
        """Reject a `Decision` whose `chosen` value isn't one of `options`.

        Catches a typo in `chosen` or `options` edited without updating
        `chosen` -- either way, a plan that states alternatives and a choice
        which don't agree with each other shouldn't construct successfully.
        The comparison is exact: normalizing case or whitespace here would
        just as easily paper over the same kind of mistake it's meant to
        catch.
        """
        if self.chosen not in self.options:
            raise ValueError(
                f"chosen {self.chosen!r} is not one of the stated options {list(self.options)!r}"
            )
        return self


# The longest scope choice id, option id and reason code a scope decision
# holds, and the most year spans a partial option lists. A choice id is a
# spec_id and an axis (51 characters at most), an option id an axis and a
# value such as `sector:120+130`, and a partial option covers one span or a
# few; each limit sits several times above that.
MAX_SCOPE_ID_CHARS = 200
MAX_SCOPE_YEAR_SPANS = 50


class ScopeOptionState(StrictModel):
    """One option of a scope choice as Search offered it for the pinned part:
    its id, whether it can be served, the years a partial option covers and
    the reason code of a partial or unavailable one."""

    option_id: str = Field(max_length=MAX_SCOPE_ID_CHARS)
    status: OptionStatus
    years: tuple[tuple[int, int], ...] | None = Field(default=None, max_length=MAX_SCOPE_YEAR_SPANS)
    code: str | None = Field(default=None, max_length=MAX_SCOPE_ID_CHARS)


class DecisionScope(StrictModel):
    """The scope choice a scope decision states (`harness.scope_choices.ScopeChoice`).

    `choice_id` names the choice and `axis` the axis it settles. `options`
    holds one state per `Decision.options` label, in the same order.
    `provenance` is how the choice came to hold its option, `"default"` when
    Search's default settled it, and `None` on a decision recorded without it.
    """

    choice_id: str = Field(max_length=MAX_SCOPE_ID_CHARS)
    axis: ScopeAxis
    options: tuple[ScopeOptionState, ...] = Field(max_length=MAX_DECISION_OPTIONS)
    provenance: Annotated[ChoiceProvenance | None, HASHED_WHEN_SET] = None


# How a calculation qualifies the answer where its inputs differ on one aligned
# axis, one caveat per axis (`submission.service._DIFFERENCES`).
DifferenceCaveat = Literal["providers_differ", "bases_differ"]

# How a decision `propose_plan` writes qualifies the answer (`Decision.caveat`): a
# calculation's inputs differing on an axis, or a whole compiled from its part.
DecisionCaveat = Literal[DifferenceCaveat, "derived_whole"]

# The most decisions qualifying the answer one requirement carries: a calculation
# states at most one per axis its inputs differ on, and a whole, a dataset
# requirement, only `derived_whole`.
MAX_REQUIREMENT_CAVEATS = len(get_args(DifferenceCaveat))


class Decision(_Choice):
    """One methodological choice the plan makes explicit."""

    fixes: DecisionFixes | None = Field(
        default=None,
        description=(
            "The requirement fields this decision settles; unset when it is a rationale only."
        ),
    )
    # Set only on a decision `propose_plan` writes for a scope choice of a
    # pinned source (`submission.service._scope_decisions`). Left out of
    # every dump while unset, so a plan stating none keeps its serialised
    # form and its `plan_revision`.
    scope: Annotated[DecisionScope | None, HASHED_WHEN_SET] = Field(
        default=None, exclude_if=lambda scope: scope is None
    )
    # Set only on a decision `propose_plan` writes for a choice that qualifies
    # the answer, whose `rationale` says how: a calculation whose inputs count
    # different providers (`providers_differ`) or are counted on different
    # measure bases (`bases_differ`), or a total compiled from its part
    # (`derived_whole`) (`submission.service._difference_decisions`,
    # `_whole_decisions`). Absent from `DecisionDraft`, so the planner cannot
    # set it.
    caveat: Annotated[DecisionCaveat | None, HASHED_WHEN_SET] = Field(
        default=None, exclude_if=lambda caveat: caveat is None
    )

    @model_validator(mode="after")
    def _scope_states_each_option(self) -> Decision:
        """Reject a scope decision whose option states do not match its options one for one."""
        if self.scope is not None and len(self.scope.options) != len(self.options):
            raise ValueError(
                f"a scope decision states {len(self.scope.options)} option states for "
                f"{len(self.options)} options"
            )
        return self

    @property
    def active_option_id(self) -> str | None:
        """The option id of `chosen` on a scope decision; `None` on any other."""
        if self.scope is None:
            return None
        return self.scope.options[self.options.index(self.chosen)].option_id


def search_pin_slot(spec_id: str) -> str:
    """The `Decision.slot` `propose_plan` records its decision for a pin of `spec_id` under.

    `propose_plan` writes that decision itself when a requirement pins a
    measure offered by a source a search settled on
    (`planner.agent._measure_decisions`).
    Derived from `spec_id`, so two pins never collide; `spec_id` already
    matches `Identifier`'s pattern, so the prefixed slot does too.
    """
    return f"source_{spec_id}"


def is_search_pin(decision: Decision) -> bool:
    """Whether `decision` is one `propose_plan` wrote for a pin of a source a search settled on."""
    fixes = decision.fixes
    return (
        fixes is not None
        and fixes.source_spec_id is not None
        and decision.slot == search_pin_slot(fixes.source_spec_id)
    )


def scope_choice_slot(spec_id: str, axis: ScopeAxis) -> str:
    """The `Decision.slot` `propose_plan` records the scope choice on `axis` of a pin of `spec_id` under.

    Derived from the pin as `search_pin_slot` is, so each source's choices
    have slots of their own.
    """
    return f"scope_{axis}_{spec_id}"


def is_scope_choice(decision: Decision) -> bool:
    """Whether `decision` is one `propose_plan` wrote for a scope choice of a pinned source."""
    fixes = decision.fixes
    return (
        decision.scope is not None
        and fixes is not None
        and fixes.source_spec_id is not None
        and decision.slot == scope_choice_slot(fixes.source_spec_id, decision.scope.axis)
    )


def pin_slot_moves(origin: str, target: str) -> dict[str, str]:
    """Each slot `propose_plan` may write for a pin of `origin`, mapped to the one a pin of `target` takes.

    The measure decision's (`search_pin_slot`) and one scope decision's per
    axis (`scope_choice_slot`), so a moved pin renames every decision code
    wrote for it and every citation of one.
    """
    moves = {search_pin_slot(origin): search_pin_slot(target)}
    for axis in get_args(ScopeAxis):
        moves[scope_choice_slot(origin, axis)] = scope_choice_slot(target, axis)
    return moves


# The name of each `DecisionFixes` field a `DecisionDraft` may list.
# `source_spec_id` is absent. A pin needs no decision, and `propose_plan`
# writes the one a measure pin states (`submission.service._measure_decisions`).
FixedField = Literal[
    "period",
    "unit",
    "entities",
    "across",
    "measure",
    "stat_vars",
    "reduction",
    "group_by",
    "window_start",
    "window_end",
    "direction",
    "predicates",
    "among",
    "axis",
    "denominator_scope",
    "change",
    "per_period",
    "limit",
]


# `settle_decisions` builds the `Decision` from a draft, reading each named
# field's value off the requirements that cite it.
class DecisionDraft(_Choice):
    """One methodological choice, as a proposal states it: the fields it fixes, by name."""

    fixes: tuple[FixedField, ...] | None = Field(
        default=None,
        max_length=len(DecisionFixes.model_fields),
        description=(
            "The requirement fields this decision settles, by name, whenever it settles "
            "what a requirement carries: the period, the unit, the entities, the measure, "
            "or a calculation field a derived requirement reads. Each value is read off the "
            "requirements citing it. Empty when it is a rationale only."
        ),
    )


# What a declared whole leaves open: the part's sector, or the party it names on one
# side of the flow (`WholeOf.axis`).
WholeAxis = Literal["sector", "donor", "recipient"]


class WholeOf(StrictModel):
    """The part a dataset requirement is the whole of: that part's own dataset with one scope axis left open.

    `propose_plan` compiles the whole from the part's pin with `axis` left
    open and every other choice held, and pins the requirement to it
    (`submission.service._declared_wholes`). The sector is opened by
    unbinding the slot the part's sector choice binds; a donor or recipient
    by dropping the role the part binds there, its region entry and its
    named set, so Search serves the total that holds the part.
    """

    part: Identifier = Field(description="The dataset requirement this one is the whole of.")
    axis: WholeAxis = Field(
        description=(
            "What the whole leaves open: 'sector' for every sector, 'donor' for every donor, "
            "'recipient' for every recipient."
        )
    )


# The compiler writes a pinned requirement's retrieval, and which column a
# retrieved frame carries for an entity is learnt only when data is found.
class DatasetRequirement(TaggedModel):
    """The series one part of the plan rests on, in fields a retrieved frame is checked against."""

    kind: Literal["dataset"] = Field(
        default="dataset", description="Tags this requirement as a dataset requirement."
    )
    id: Identifier = Field(
        description="This requirement's name; unique within a plan, so a calculation can cite it."
    )
    decision_ids: tuple[Identifier, ...] = Field(
        default=(),
        max_length=MAX_PLAN_DECISIONS,
        description=(
            "The decisions that fixed this requirement's fields; empty when the question "
            "states them outright and nothing was weighed."
        ),
    )
    measure: str = Field(
        min_length=1,
        max_length=MAX_MEASURE_CHARS,
        description=(
            "What this requirement retrieves, in the words of the question, naming how a "
            "flow is counted only when the question does."
        ),
    )
    entities: tuple[Entity, ...] = Field(
        default=(),
        max_length=MAX_ENTITIES,
        description="Who or where this requirement is about, by name.",
    )
    across: Annotated[tuple[AcrossWord, ...], HASHED_WHEN_SET] = Field(
        default=(),
        max_length=MAX_ACROSS_WORDS,
        description=(
            "The parties whose every member this requirement covers, each as the party word "
            "alone, such as 'donor', 'recipient' or 'country', when the question ranks them or "
            "its answer is their total, and names none of them: one word per party, so every "
            "donor by every recipient is ['donor', 'recipient']. No stand-in entity such as "
            "'World' is named for them, and no entity in a party it covers. A narrower "
            "population, such as a region, an income group or a donor type, is named itself "
            "in `entities` instead (see `members`). A world figure set beside a named entity names "
            "'World' instead."
        ),
    )
    period: Period = Field(description="The years an acceptable result has to cover.")
    period_end_open: Annotated[bool, HASHED_WHEN_SET] = Field(
        default=False,
        description=(
            "Set when the period follows the newest data: the dataset is retrieved with no "
            "end year, so later years reach it. `period` still states the years. Needs a "
            "pinned `source_spec_id`."
        ),
    )
    frequency: Literal["annual"] = Field(
        default="annual",
        description="How often an acceptable result is observed; annual is the only frequency this plan states.",
    )
    unit: Unit | None = Field(
        default=None,
        description=(
            "The unit and price basis an acceptable result has to be stated in. Unset unless "
            "the question itself names a currency or price basis: a unit read off a search "
            "result, a source or a skill is never stated here."
        ),
    )
    source: str | None = Field(
        default=None,
        min_length=1,
        max_length=MAX_SOURCE_CHARS,
        description="The source the question names for this requirement, when it names one, in the question's own words.",
    )
    source_ref: Annotated[str | None, HASHED_WHEN_SET] = Field(
        default=None,
        min_length=1,
        max_length=200,
        description="The admitted source reference the user chose from Search's source ambiguity candidates; leave unset before the user chooses.",
    )
    source_exclude: Annotated[bool, HASHED_WHEN_SET] = Field(
        default=False,
        description="Set when the question excludes the named source instead of requesting it.",
    )
    source_code: Annotated[str | None, HASHED_WHEN_SET] = Field(
        default=None,
        min_length=1,
        max_length=MAX_SOURCE_CODE_CHARS,
        description=(
            "A publisher's own series code for this requirement, such as 'DT.TDS.DEGG.CD', "
            "copied exactly from the question or from a search result that names it. Search "
            "finds the series by this code. `measure` keeps the question's words."
        ),
    )
    source_spec_id: SpecId | None = Field(
        default=None,
        description="The exact dataset definition discovery has to use for this requirement.",
    )
    coverage: Coverage = Field(
        default=Coverage(),
        description="What to do when this requirement's evidence cannot be confirmed.",
    )
    stat_vars: StatVarReading | None = Field(
        default=None,
        description=(
            "How to read a source that binds a family of StatVars, such as aid split by "
            "sector and flow type, or population by sex: 'sum' totals the family into one "
            "series, 'separate' keeps one series per StatVar. Set 'sum' when the question "
            "asks for the total and 'separate' when it asks for the parts, as 'by flow type' "
            "or 'by sector' does. Leave it unset to let the source's summable axes decide; a "
            "choice you weigh is a decision fixing stat_vars. A summed flow keeps one row per "
            "entity on a side the question leaves open, such as every recipient of a donor's "
            "aid, so charting or totalling it over time needs an aggregate summing by "
            "period. No effect on a source binding one StatVar."
        ),
    )
    breakdown: Annotated[Breakdown | None, HASHED_WHEN_SET] = Field(
        default=None,
        description=(
            "The classification to split the total by, at one level of its hierarchy: what a "
            "total by sector is split by, as is a ranking of which sectors received the most. "
            "Adds a role named for the dimension, which a rank's axis or an aggregate's "
            "group_by 'breakdown' reads. Needs stat_vars 'separate'."
        ),
    )
    # Left out of every dump while unset, as `Entity.members` is.
    whole_of: Annotated[WholeOf | None, HASHED_WHEN_SET] = Field(
        default=None,
        exclude_if=lambda whole_of: whole_of is None,
        description=(
            "Set on the total a share_of or quotient_of divides its part by, the part over "
            "every sector, donor or recipient: all US aid for US health aid, all aid to Kenya "
            "for US aid to Kenya. Counted from the part's own dataset, with no search. Read "
            "only as that denominator."
        ),
    )

    @field_validator("source_code", mode="before")
    @classmethod
    def _source_code_is_normalised(cls, code: object) -> object:
        """Strip and upper-case a series code, as Search reads it."""
        return code.strip().upper() if isinstance(code, str) else code

    @model_validator(mode="after")
    def _source_modifiers_name_a_source(self) -> DatasetRequirement:
        if self.source is None and (self.source_ref is not None or self.source_exclude):
            raise ValueError("source_ref and source_exclude require source wording")
        return self

    @model_validator(mode="after")
    def _entities_are_unique(self) -> DatasetRequirement:
        """Reject a requirement naming the same entity, role and node more than once."""
        keys = [(entity.name, entity.role, entity.dcid) for entity in self.entities]
        seen: set[tuple[str, str | None, str | None]] = set()
        repeated = sorted(
            {"/".join(part for part in key if part) for key in keys if key in seen or seen.add(key)}
        )
        if repeated:
            raise ValueError(
                f"every entity must be unique by name, role and node; repeated: {repeated!r}"
            )
        return self

    @model_validator(mode="after")
    def _nodes_are_unique(self) -> DatasetRequirement:
        """Reject two entities stating one `dcid`: Search reads rows pinning one node
        as one entity, so the second would lose its role or name the first again."""
        dcids = [entity.dcid for entity in self.entities if entity.dcid is not None]
        if repeated := sorted({dcid for dcid in dcids if dcids.count(dcid) > 1}):
            raise ValueError(f"each entity's dcid must be its own node; repeated: {repeated!r}")
        return self

    @model_validator(mode="after")
    def _breakdown_is_its_own_role(self) -> DatasetRequirement:
        """Reject a breakdown read summed, or one named like a role the requirement states.

        A summed family is one series with no code left to break down by, and
        the breakdown's role is the dimension's name, which a rank's `axis`
        could not tell apart from an entity role or `across` of the same name.
        """
        if self.breakdown is None:
            return self
        if self.stat_vars is StatVarReading.SUM:
            raise ValueError(
                "a breakdown splits the family by its codes, and stat_vars='sum' totals it "
                "into one series; state stat_vars='separate'"
            )
        role = self.breakdown.dimension.value
        if role in self.across or any(entity.role == role for entity in self.entities):
            raise ValueError(
                f"breakdown {role!r} names the role its own codes take, so no entity role or "
                f"`across` may also be {role!r}"
            )
        return self

    @model_validator(mode="after")
    def _states_who_it_covers(self) -> DatasetRequirement:
        """Reject a requirement about nobody, one stating a party twice in `across`, or one
        naming an entity in a party `across` covers whole.

        Parties are compared by what they name, so `role='donors'` sits in
        `across=('donor',)`, an entity with no role or a place role sits in
        `across=('country',)`, and 'donor' and 'provider' state one party. A
        word outside the party vocabulary is left for
        `compiler.evidence.open_axes_for` to refuse.
        """
        if not self.across:
            if not self.entities:
                raise ValueError(
                    "a dataset requirement names at least one entity, or states in `across` "
                    "the parties whose every member it covers"
                )
            return self
        parties = [across_party(word) for word in self.across]
        stated = [
            party or word.casefold() for word, party in zip(self.across, parties, strict=True)
        ]
        if repeated := sorted(
            word for word, key in zip(self.across, stated, strict=True) if stated.count(key) > 1
        ):
            raise ValueError(
                f"`across` states each party once, and {repeated!r} name the same one; keep one "
                "word for it"
            )
        for word, party in zip(self.across, parties, strict=True):
            self._names_none_of(word, party)
        return self

    def _names_none_of(self, word: str, party: Party | None) -> None:
        """Reject an entity named in the party the `across` word covers whole."""
        every = f"across word {word!r} covers every {singular(word)}"
        named_in_across = sorted(
            entity.name
            for entity in self.entities
            if entity.role is not None
            and (
                entity.role.casefold() == word.casefold()
                or (party is not None and role_party(entity.role) == party)
            )
        )
        if named_in_across:
            raise ValueError(
                f"{every}, so no entity may also be named in that party; named: "
                f"{named_in_across!r}. Drop them to cover every member, or drop the word from "
                "`across` and name every entity the answer needs"
            )
        unplaced = sorted(entity.name for entity in self.entities if entity.role is None)
        if party == "place" and unplaced:
            raise ValueError(
                f"{every}, and {unplaced!r} carry no `role`, so each names one of them and "
                "leaves no population open. To rank every member, drop them; to rank them "
                "against every member, drop the word from `across` and state the rank's "
                "among='population'"
            )


# --- Calculation vocabulary -----------------------------------------------
#
# Every calculation a plan needs beyond "retrieve this series" is a typed
# derived requirement below. Each states what analysis-mcp computes and what
# analysis-mcp fixes on its own -- the actual column names, the join keys --
# so the plan states only what a reviewer has to check.


class RequirementSide(StrictModel):
    """One side of a two-sided calculation.

    It names which requirement's result it reads, narrowed to one entity when
    it reads a single row of that result.
    """

    input_id: Identifier = Field(description="Which requirement's result this side reads.")
    entity: str | None = Field(
        default=None,
        min_length=1,
        max_length=MAX_ENTITY_NAME_CHARS,
        description="The one entity from `input_id`'s requirement this side narrows to; unset to take the whole frame.",
    )


class AggregateRequirement(TaggedModel):
    """Collapse `input_id`'s rows into one figure per group."""

    kind: Literal["aggregate"] = Field(
        default="aggregate", description="Tags this requirement as an aggregate."
    )
    id: Identifier = Field(
        description="This requirement's name; unique within a plan, so a calculation can cite it."
    )
    decision_ids: tuple[Identifier, ...] = Field(
        default=(),
        max_length=MAX_PLAN_DECISIONS,
        description="The decisions that fixed this requirement's fields.",
    )
    input_id: Identifier = Field(description="The requirement whose result this aggregate reduces.")
    reduction: Reduction = Field(
        description="How to collapse each group's rows into one figure: sum, mean, max or min."
    )
    group_by: tuple[GroupBy, ...] = Field(
        max_length=MAX_GROUP_BY,
        description=(
            "Which columns to group by before reducing: period, entity (the roles of the "
            "entities named, plus a breakdown's codes when the input has them; an `across` "
            "party is summed), breakdown (its codes alone), or two of them. Without period, "
            "the years reduce to one figure per group. Empty reduces every row to one figure, "
            "over every year, party and code."
        ),
    )

    @field_validator("group_by")
    @classmethod
    def _each_column_once(cls, group_by: tuple[GroupBy, ...]) -> tuple[GroupBy, ...]:
        """Reject a column named twice, which groups by nothing more than naming it once."""
        _reject_repeats(group_by, what="`group_by` column")
        return group_by


_ROLLING_WINDOW_DESCRIPTION = (
    "Set when the window follows the newest data: which ends are relative and how far back "
    "each sits. `window_start` and `window_end` still state the years. Needs "
    "`period_end_open` on the dataset requirement read."
)


def _years_disagree(requirement_id: str, fault: str) -> RollingPlanError:
    """`RollingIssue.YEARS_DISAGREE` for `fault`, naming the requirement."""
    return RollingPlanError(
        RollingIssue.YEARS_DISAGREE,
        f"{requirement_id!r}: {fault}",
        requirement_ids=[requirement_id],
    )


class GrowthRequirement(TaggedModel):
    """The growth of `input_id`'s value from `window_start` to `window_end`."""

    kind: Literal["growth"] = Field(
        default="growth", description="Tags this requirement as a growth."
    )
    id: Identifier = Field(
        description="This requirement's name; unique within a plan, so a calculation can cite it."
    )
    decision_ids: tuple[Identifier, ...] = Field(
        default=(),
        max_length=MAX_PLAN_DECISIONS,
        description="The decisions that fixed this requirement's fields.",
    )
    input_id: Identifier = Field(
        description="The requirement whose result this growth reads its two endpoint values from."
    )
    window_start: Year = Field(description="The year growth is measured from.")
    window_end: Year = Field(description="The year growth is measured to.")
    change: GrowthChange = Field(
        default=GrowthChange.RELATIVE,
        description=(
            "How the two endpoint values are compared: 'relative' divides the difference by the "
            "start value and reads a dimensionless ratio, so a share moving 10% to 15% reads +50%; "
            "'absolute' subtracts and reads a difference in the input's own unit, so the same move "
            "reads +5 percentage points."
        ),
    )
    rolling: Annotated[RollingWindow | None, HASHED_WHEN_SET] = Field(
        default=None, description=_ROLLING_WINDOW_DESCRIPTION
    )

    @model_validator(mode="after")
    def _window_ends_after_it_starts(self) -> GrowthRequirement:
        if self.window_end <= self.window_start:
            raise ValueError(
                f"growth window must end after it starts; got {self.window_start}-{self.window_end}"
            )
        return self

    @model_validator(mode="after")
    def _rolling_agrees_with_the_years(self) -> GrowthRequirement:
        check_rolling_window(
            self.rolling,
            window_start=self.window_start,
            window_end=self.window_end,
            error=functools.partial(_years_disagree, self.id),
        )
        return self


# What `accept_provider_difference` means to the planner, on each calculation
# that subtracts, divides or adds its inputs
# (`submission.service._PROVIDER_ALIGNED`).
ACCEPT_PROVIDER_DIFFERENCE = (
    "Set only when the user asks for inputs counting different providers' spending, as "
    "'use the DAC1 total' does beside CRS aid: each input keeps its own provider scope."
)


class ComparisonRequirement(TaggedModel):
    """The difference between two requirements' results, at each common period.

    A comparison answers the question and feeds nothing further, so no other
    requirement may read one.
    """

    kind: Literal["comparison"] = Field(
        default="comparison", description="Tags this requirement as a comparison."
    )
    id: Identifier = Field(
        description="This requirement's name; unique within a plan, so a calculation can cite it."
    )
    decision_ids: tuple[Identifier, ...] = Field(
        default=(),
        max_length=MAX_PLAN_DECISIONS,
        description="The decisions that fixed this requirement's fields.",
    )
    minuend: RequirementSide = Field(
        description="The side the difference starts from: the result is `minuend` minus `subtrahend`."
    )
    subtrahend: RequirementSide = Field(description="The side subtracted from `minuend`.")
    # Left out of every dump while unset, as `Entity.members` is.
    accept_provider_difference: Annotated[bool, HASHED_WHEN_SET] = Field(
        default=False,
        exclude_if=lambda accepted: not accepted,
        description=ACCEPT_PROVIDER_DIFFERENCE,
    )

    @model_validator(mode="after")
    def _sides_differ(self) -> ComparisonRequirement:
        if (self.minuend.input_id, self.minuend.entity) == (
            self.subtrahend.input_id,
            self.subtrahend.entity,
        ):
            raise ValueError("a comparison's minuend and subtrahend must differ")
        return self


class SumOfRequirement(TaggedModel):
    """Two single-series results added together, at each period both report.

    Each side reads one series: one entity's, or a result holding one. A
    period only one side reports is left out of the sum, never counted as
    zero.
    """

    kind: Literal["sum_of"] = Field(
        default="sum_of", description="Tags this requirement as a sum of two series."
    )
    id: Identifier = Field(
        description="This requirement's name; unique within a plan, so a calculation can cite it."
    )
    decision_ids: tuple[Identifier, ...] = Field(
        default=(),
        max_length=MAX_PLAN_DECISIONS,
        description="The decisions that fixed this requirement's fields.",
    )
    left: RequirementSide = Field(description="One of the two series added together.")
    right: RequirementSide = Field(description="The series added to `left`.")
    # Left out of every dump while unset, as `Entity.members` is.
    accept_provider_difference: Annotated[bool, HASHED_WHEN_SET] = Field(
        default=False,
        exclude_if=lambda accepted: not accepted,
        description=ACCEPT_PROVIDER_DIFFERENCE,
    )

    @model_validator(mode="after")
    def _sides_read_different_requirements(self) -> SumOfRequirement:
        if self.left.input_id == self.right.input_id:
            raise ValueError(
                "a sum_of adds two different sources, so its sides read different "
                f"requirements; both read {self.left.input_id!r}. To total several entities "
                "of one series, aggregate it"
            )
        return self


class RankRequirement(TaggedModel):
    """Rank `input_id`'s entities against each other.

    Entities tied on value share a rank.
    """

    kind: Literal["rank"] = Field(default="rank", description="Tags this requirement as a rank.")
    id: Identifier = Field(
        description="This requirement's name; unique within a plan, so a calculation can cite it."
    )
    decision_ids: tuple[Identifier, ...] = Field(
        default=(),
        max_length=MAX_PLAN_DECISIONS,
        description="The decisions that fixed this requirement's fields.",
    )
    input_id: Identifier = Field(description="The requirement whose result this rank orders.")
    direction: RankDirection = Field(
        description="Which end of the ranking comes first: the highest value or the lowest."
    )
    among: RankAmong = Field(
        description=(
            "Who this rank compares an entity against: 'named' for exactly the entities a "
            "filter has narrowed to, 'population' for every entity the retrieved definition "
            "holds: every member of the party its `across` states, or, where entities are "
            "named, those entities as the focus among the rest."
        )
    )
    axis: str | None = Field(
        default=None,
        min_length=1,
        max_length=MAX_ENTITY_ROLE_CHARS,
        description="Which role dimension of the input this rank keeps; required once the input carries more than one.",
    )
    reduction: Reduction | None = Field(
        default=None,
        description=(
            "How to collapse an entity's rows to one value first. Required when the input "
            "carries more than one role, to collapse the other role's rows, and required "
            "when the window leaves an entity more than one row."
        ),
    )
    window_start: Year | None = Field(
        default=None,
        description=(
            "The year this rank's window starts, when it is windowed. The same year as "
            "`window_end` ranks that one year."
        ),
    )
    window_end: Year | None = Field(
        default=None,
        description=(
            "The year this rank's window ends, when it is windowed. The same year as "
            "`window_start` ranks that one year."
        ),
    )
    per_period: bool = Field(
        default=False,
        description="Whether this rank orders each period on its own rather than collapsing the window into one value per entity.",
    )
    limit: Annotated[RankLimit | None, HASHED_WHEN_SET] = Field(
        default=None,
        description=(
            "Keep only the entities ranked at this place or better: 10 with 'highest_first' "
            "is the top 10, 2 with 'lowest_first' the bottom 2. A value shared at the last "
            "place kept keeps every entity sharing it. Applies within each period under "
            "per_period, whose kept rows then carry measured values, not rank positions, and "
            "each entity only in the periods it made the cut. Unset keeps every entity "
            "ranked."
        ),
    )
    rolling: Annotated[RollingWindow | None, HASHED_WHEN_SET] = Field(
        default=None, description=_ROLLING_WINDOW_DESCRIPTION
    )

    @model_validator(mode="after")
    def _window_is_paired_and_ordered(self) -> RankRequirement:
        if (self.window_start is None) != (self.window_end is None):
            raise ValueError("a rank window needs both window_start and window_end, or neither")
        if (
            self.window_start is not None
            and self.window_end is not None
            and self.window_end < self.window_start
        ):
            raise ValueError(
                f"rank window ends before it starts; got {self.window_start}-{self.window_end}"
            )
        return self

    @model_validator(mode="after")
    def _rolling_agrees_with_the_years(self) -> RankRequirement:
        check_rolling_window(
            self.rolling,
            window_start=self.window_start,
            window_end=self.window_end,
            error=functools.partial(_years_disagree, self.id),
        )
        return self


class ShareOfRequirement(TaggedModel):
    """The share `numerator` holds of `denominator`, at each period.

    Both sides cover the same years, because a period only one side covers
    has no defined share. Each member's share of all of them, as each donor's
    share of all ODA to one recipient, reads one side as both, keyed by `axis`
    and 'pooled', from a frame keeping that role open, which an aggregate by
    entity sums away. One named member's share divides that member's series
    by every member's total per period.
    """

    kind: Literal["share_of"] = Field(
        default="share_of", description="Tags this requirement as a share_of."
    )
    id: Identifier = Field(
        description="This requirement's name; unique within a plan, so a calculation can cite it."
    )
    decision_ids: tuple[Identifier, ...] = Field(
        default=(),
        max_length=MAX_PLAN_DECISIONS,
        description="The decisions that fixed this requirement's fields.",
    )
    numerator: RequirementSide = Field(description="The side this share is a share of.")
    denominator: RequirementSide = Field(description="The side `numerator` is divided by.")
    axis: str | None = Field(
        default=None,
        min_length=1,
        max_length=MAX_ENTITY_ROLE_CHARS,
        description="The role dimension to compute one share per entity on, rather than one figure overall.",
    )
    denominator_scope: DenominatorScope | None = Field(
        default=None,
        description=(
            "Whose total a keyed share divides by: 'per_axis' divides each entity by its own "
            "total, 'pooled' divides every entity by one total summed across the population."
        ),
    )
    # Left out of every dump while unset, as `Entity.members` is.
    accept_provider_difference: Annotated[bool, HASHED_WHEN_SET] = Field(
        default=False,
        exclude_if=lambda accepted: not accepted,
        description=ACCEPT_PROVIDER_DIFFERENCE,
    )

    @property
    def pools_its_own_axis(self) -> bool:
        """Whether one side is both numerator and denominator, each `axis` member's share of their total."""
        return self.numerator == self.denominator

    @model_validator(mode="after")
    def _sides_differ(self) -> ShareOfRequirement:
        # Keyed by `axis` and pooled across it, one side divides each member by
        # the members' total; read any other way it divides a series by itself.
        if self.pools_its_own_axis and self.denominator_scope is not DenominatorScope.POOLED:
            raise ValueError(
                "a share_of's numerator and denominator must differ, unless it keys that one "
                "side by `axis` with denominator_scope 'pooled', each member's share of the "
                "total across them"
            )
        return self

    @model_validator(mode="after")
    def _denominator_scope_needs_an_axis(self) -> ShareOfRequirement:
        if (self.axis is None) != (self.denominator_scope is None):
            raise ValueError(
                "denominator_scope is stated exactly when axis is: an un-keyed share has one "
                "denominator value and the question does not arise, a keyed share must state it"
            )
        return self


class QuotientRequirement(TaggedModel):
    """One measure divided by another, at each period both report.

    For a per-head, per-capita or "divide one by the other" figure, in the
    unit the two compose.
    """

    kind: Literal["quotient_of"] = Field(
        default="quotient_of", description="Tags this requirement as a quotient_of."
    )
    id: Identifier = Field(
        description="This requirement's name; unique within a plan, so a calculation can cite it."
    )
    decision_ids: tuple[Identifier, ...] = Field(
        default=(),
        max_length=MAX_PLAN_DECISIONS,
        description="The decisions that fixed this requirement's fields.",
    )
    numerator: RequirementSide = Field(description="The side that is divided.")
    denominator: RequirementSide = Field(description="The side `numerator` is divided by.")
    axis: str | None = Field(
        default=None,
        min_length=1,
        max_length=MAX_ENTITY_ROLE_CHARS,
        description=(
            "The numerator's role dimension to compute one quotient per entity on, rather "
            "than one figure overall. Alone, every entity divides by the denominator's one "
            "total."
        ),
    )
    denominator_axis: str | None = Field(
        default=None,
        min_length=1,
        max_length=MAX_ENTITY_ROLE_CHARS,
        description=(
            "The denominator's name for the same entities, so each divides by its own "
            "counterpart: the role they state there, or 'country' where they state none."
        ),
    )
    # Left out of every dump while unset, as `Entity.members` is.
    accept_provider_difference: Annotated[bool, HASHED_WHEN_SET] = Field(
        default=False,
        exclude_if=lambda accepted: not accepted,
        description=ACCEPT_PROVIDER_DIFFERENCE,
    )

    @model_validator(mode="after")
    def _sides_differ(self) -> QuotientRequirement:
        if (self.numerator.input_id, self.numerator.entity) == (
            self.denominator.input_id,
            self.denominator.entity,
        ):
            raise ValueError("a quotient_of's numerator and denominator must differ")
        return self

    @model_validator(mode="after")
    def _denominator_axis_needs_an_axis(self) -> QuotientRequirement:
        if self.denominator_axis is not None and self.axis is None:
            raise ValueError(
                "denominator_axis names the denominator's role for the entities axis keys the "
                "numerator by, so it is stated only with axis"
            )
        return self


# analysis-mcp inner-joins on `time_period` and the keys the compiler writes,
# the role pairs `planner.shapes.join_key_pairs` finds. `_join_shape` refuses
# a join with none.
class JoinRequirement(TaggedModel):
    """Join two requirements' results into one frame.

    Only the period and the entities both sides carry survive, so the two
    requirements' periods and entities have to overlap for the join to hold
    anything. A join answers the question and feeds nothing further, so no
    other requirement may read one.
    """

    kind: Literal["join"] = Field(default="join", description="Tags this requirement as a join.")
    id: Identifier = Field(
        description="This requirement's name; unique within a plan, so a calculation can cite it."
    )
    decision_ids: tuple[Identifier, ...] = Field(
        default=(),
        max_length=MAX_PLAN_DECISIONS,
        description="The decisions that fixed this requirement's fields.",
    )
    left_id: Identifier = Field(description="One of the two requirements this join combines.")
    right_id: Identifier = Field(description="The other requirement this join combines.")

    @model_validator(mode="after")
    def _ids_are_distinct(self) -> JoinRequirement:
        if self.left_id == self.right_id:
            raise ValueError(
                f"a join's left_id and right_id must differ; both are {self.left_id!r}"
            )
        return self


class FilterRequirement(TaggedModel):
    """Keep only `input_id`'s rows that satisfy every one of `predicates`."""

    kind: Literal["filter"] = Field(
        default="filter", description="Tags this requirement as a filter."
    )
    id: Identifier = Field(
        description="This requirement's name; unique within a plan, so a calculation can cite it."
    )
    decision_ids: tuple[Identifier, ...] = Field(
        default=(),
        max_length=MAX_PLAN_DECISIONS,
        description="The decisions that fixed this requirement's fields.",
    )
    input_id: Identifier = Field(description="The requirement whose rows this filter narrows.")
    predicates: tuple[Predicate, ...] = Field(
        min_length=1,
        max_length=MAX_PREDICATES,
        description="The conditions a row has to satisfy, all of them, to survive this filter.",
    )

    @model_validator(mode="after")
    def _keeps_at_most_one_level(self) -> FilterRequirement:
        levels = [p.level.value for p in self.predicates if isinstance(p, PlaceLevel)]
        if len(levels) > 1:
            raise ValueError(
                f"a filter keeps one level of place, so it states one place_level; got {levels!r}"
            )
        return self

    @model_validator(mode="after")
    def _states_each_predicate_once(self) -> FilterRequirement:
        """Reject a predicate stated twice, which analysis-mcp's filter op spec refuses.

        analysis-mcp compares predicates by their canonical form
        (`analysis_mcp.identity._sorted_unique_predicates`), which sorts an
        entity list, and two predicates equal here compile to one form
        (`compiler.compile._filter_arguments`). So do two whose names
        resolve to one entity, which only the sources show, and the
        compiler refuses those (`compiler.arguments.op_spec_errors`).
        """
        _reject_repeats((predicate_key(p) for p in self.predicates), what="predicate")
        return self


def predicate_key(predicate: Predicate) -> str:
    """`predicate` in a form two statements of it share, its names in any order."""
    if isinstance(predicate, EntityIn):
        predicate = predicate.model_copy(update={"names": tuple(sorted(set(predicate.names)))})
    return predicate.model_dump_json()


DerivedRequirement = (
    AggregateRequirement
    | GrowthRequirement
    | ComparisonRequirement
    | SumOfRequirement
    | RankRequirement
    | ShareOfRequirement
    | QuotientRequirement
    | JoinRequirement
    | FilterRequirement
)

# Pydantic does not discriminate on a default: JSON input has to carry
# `kind` explicitly. Constructing a concrete class from Python still gets
# the field's default, since the discriminator only governs how the union
# picks a member from an already-parsed mapping.
EvidenceRequirement = Annotated[
    DatasetRequirement | DerivedRequirement, Field(discriminator="kind")
]


def is_dataset(requirement: EvidenceRequirement) -> bool:
    """Return whether `requirement` is a `DatasetRequirement`."""
    return isinstance(requirement, DatasetRequirement)


def inputs_of(requirement: EvidenceRequirement) -> tuple[Identifier, ...]:
    """Return the requirement ids `requirement` reads.

    Empty for a dataset requirement, which reads nothing upstream. A
    comparison, sum_of, share_of or quotient_of reads both its sides, a
    comparison's minuend first; a join reads its left and right id; every
    other derived requirement reads its one `input_id`.
    """
    if isinstance(requirement, DatasetRequirement):
        return ()
    if isinstance(requirement, ComparisonRequirement):
        return (requirement.minuend.input_id, requirement.subtrahend.input_id)
    if isinstance(requirement, SumOfRequirement):
        return (requirement.left.input_id, requirement.right.input_id)
    if isinstance(requirement, ShareOfRequirement | QuotientRequirement):
        return (requirement.numerator.input_id, requirement.denominator.input_id)
    if isinstance(requirement, JoinRequirement):
        return (requirement.left_id, requirement.right_id)
    return (requirement.input_id,)


def _find_cycle(requirements_by_id: Mapping[str, EvidenceRequirement]) -> tuple[str, ...] | None:
    """Return one cycle of requirement ids, if the requirements do not form a DAG.

    A depth-first search with a coloring scheme: white is unvisited, gray is
    on the current path, black is fully explored. Finding a gray node again
    while walking `inputs_of` means the path back to it is a cycle. An input
    `requirements_by_id` does not hold reads nothing, so it closes no cycle.
    """
    white, gray, black = 0, 1, 2
    color: dict[str, int] = dict.fromkeys(requirements_by_id, white)
    path: list[str] = []

    def visit(requirement_id: str) -> tuple[str, ...] | None:
        color[requirement_id] = gray
        path.append(requirement_id)
        for input_id in inputs_of(requirements_by_id[requirement_id]):
            if color.get(input_id) == gray:
                return tuple(path[path.index(input_id) :])
            if color.get(input_id) == white:
                found = visit(input_id)
                if found is not None:
                    return found
        path.pop()
        color[requirement_id] = black
        return None

    for requirement_id in requirements_by_id:
        if color[requirement_id] == white:
            found = visit(requirement_id)
            if found is not None:
                return found
    return None


_TERMINAL_REQUIREMENTS: tuple[type, ...] = (ComparisonRequirement, JoinRequirement)

MAX_DECLINE_ASK_CHARS = 300


# `translator.referral.OUTCOME_OF` maps a referral `kind` to its outcome.
class UnsupportedOperation(TitlelessEnum):
    """An ask the system cannot answer, as an `unsupported` answer names it.

    `propose_plan` names one in `detail.operation`; its `translation` names
    one as the `kind` of a referral whose outcome is `unsupported`.
    """

    # Any calculation over a `comparison` or `join` result: both are terminal,
    # so nothing reads either.
    CALCULATION_OVER_COMBINED = "calculation_over_combined"
    # A requirement no operation the translator builds carries out.
    NO_OPERATION_COVERS = "no_operation_covers"
    # An ask search found the data cannot answer as stated.
    UNSUPPORTED_ASK = "unsupported_ask"
    # The parties an ask breaks its answer down by, where no source the ask's
    # own answer pins holds more of them than the answer keeps
    # (`submission.service`).
    BREAKDOWN_NOT_KEPT = "breakdown_not_kept"


# The most declines one plan may carry: one per code for the question as a
# whole and one per code for each ask.
MAX_DECLINES = len(UnsupportedOperation) * (MAX_PLAN_ASKS + 1)


class UnsupportedPlanError(ValueError):
    """A plan rule refused a calculation the system cannot carry out.

    Distinct from a malformed draft: no reshaping of the plan states the
    calculation, so `planner.agent.propose_plan` answers with the
    `unsupported` outcome and requires a `Decline` naming `operation`.
    """

    def __init__(self, operation: UnsupportedOperation, message: str) -> None:
        super().__init__(message)
        self.operation = operation


class Decline(StrictModel):
    """A part of the question the plan leaves unanswered."""

    ask: str = Field(
        min_length=1,
        max_length=MAX_DECLINE_ASK_CHARS,
        description="That part, in the user's words.",
    )
    code: UnsupportedOperation = Field(
        description=(
            "The code of the answer that earned this decline: its `detail.operation`, "
            "its referral `kind`, or the code a replan names; `breakdown_not_kept` "
            "when an `ask_role_dropped` problem names the ask and the source its "
            "answer pins cannot hold that party open."
        )
    )
    ask_ref: Annotated[int | None, HASHED_WHEN_SET] = Field(
        default=None,
        ge=0,
        lt=MAX_PLAN_ASKS,
        description="The number of the ask this declines, when the brief lists asks.",
    )

    @property
    def key(self) -> str:
        """What a plan holds one decline per: its code, and the ask it names if any."""
        return self.code.value if self.ask_ref is None else f"{self.code.value}@{self.ask_ref}"


class AskAnswer(StrictModel):
    """The requirements whose results answer one ask the brief lists."""

    ask: int = Field(ge=0, lt=MAX_PLAN_ASKS, description="The ask's number in the brief.")
    outputs: tuple[Identifier, ...] = Field(
        min_length=1,
        max_length=MAX_PLAN_REQUIREMENTS,
        description=(
            "The `id` of each requirement whose result answers it, whether or not another "
            "requirement reads that result too."
        ),
    )


class AskBreakdown(StrictModel):
    """The parties one ask the brief lists breaks its answer down by, and those it totals across.

    `by` holds one word for each side of the flow: one ask is one output,
    whose flow has one giving and one receiving party, so two words for one
    side (`planner.parties.words_for_one_party`) are two outputs and are
    rejected. `over` may hold several words for one side, as a total of aid
    and debt from every donor and creditor does: a total sums each away. At
    least one of them names a party.
    """

    ask: int = Field(ge=0, lt=MAX_PLAN_ASKS, description="The ask's number in the brief.")
    by: tuple[PartyRole, ...] = Field(
        default=(),
        max_length=len(PARTY_ROLES),
        description="Each party the answer lists one by one, as `search` takes `open_roles`.",
    )
    over: Annotated[tuple[PartyRole, ...], HASHED_WHEN_SET] = Field(
        default=(),
        max_length=len(PARTY_ROLES),
        description="Each party the answer totals across into one figure.",
    )

    @field_validator("by")
    @classmethod
    def _one_word_per_party(cls, by: tuple[PartyRole, ...]) -> tuple[PartyRole, ...]:
        """Reject two words for one side of the flow."""
        if shared := words_for_one_party(by):
            raise ValueError(
                f"{', '.join(shared)} name one side of a flow: an ask breaks down by one "
                "party on each side"
            )
        return by

    @model_validator(mode="after")
    def _names_a_party(self) -> AskBreakdown:
        if not self.by and not self.over:
            raise ValueError("an ask breakdown names at least one party in `by` or `over`")
        return self


class PeriodLicence(StrictModel):
    """A step the user's own words asked to combine its years, as `propose_plan` granted it.

    `propose_plan` writes it, never the model (`period_licences`), and it
    holds while the step keeps this id and `step_kind` and combines its
    years with this `reduction`, over however many years, or is a `rank`
    stating no `reduction` (`period_licences_held`).
    """

    step_id: Identifier = Field(description="The step that combines its years.")
    step_kind: Literal["aggregate", "rank"] = Field(description="The step's kind.")
    reduction: Reduction = Field(description="How the step combines the years.")


class RollingCopy(StrictModel):
    """A requirement `whole_history_split` added, as a copy of one the model wrote.

    `propose_plan` writes it, never the model (`Plan.rolling_copies`).
    """

    copy_id: Identifier = Field(description="The copy's id.")
    origin_id: Identifier = Field(description="The id of the requirement it copies.")


class FirstYear(StrictModel):
    """The first year on record of the dataset a dataset requirement pins, as `propose_plan` read it.

    `propose_plan` writes it, never the model (`Plan.first_years`). It keys
    on the model's requirement id, which a windowed pin never moves, and a
    copy of the requirement reads its origin's entry.
    """

    requirement_id: Identifier = Field(description="The dataset requirement.")
    year: Year = Field(description="The first year its dataset holds.")


class AsksUnaccountedError(ValueError):
    """A plan leaves an ask the brief lists unaccounted for, or answers or declines one twice.

    `planner.agent.propose_plan` answers it with a typed problem naming the
    asks, so the model adds the missing `answered` or `declined` entry
    rather than reshaping requirements.
    """

    def __init__(
        self, *, asks: Sequence[str], uncovered: Sequence[int], repeated: Sequence[int]
    ) -> None:
        self.uncovered = tuple(uncovered)
        self.repeated = tuple(repeated)
        parts = []
        if self.uncovered:
            parts.append(f"neither answered nor declined: {_named_asks(asks, self.uncovered)}")
        if self.repeated:
            parts.append(f"accounted for more than once: {_named_asks(asks, self.repeated)}")
        super().__init__(
            "every ask the brief lists is answered by one `answered` entry, declined by one "
            "`declined` entry naming it in `ask_ref`, or both when the plan computes part of "
            f"it and the decline names the part it does not; {'; '.join(parts)}"
        )


def _named_asks(asks: Sequence[str], numbers: Iterable[int]) -> str:
    return ", ".join(f"ask {number} ({asks[number]!r})" for number in numbers)


def _check_asks_accounted(
    asks: Sequence[str],
    answered: Sequence[AskAnswer],
    declined: Sequence[Decline],
    requirements: Sequence[EvidenceRequirement],
) -> None:
    """Reject a plan that leaves an ask unaccounted for, or answers or declines one twice.

    Each ask the brief lists is answered by one `AskAnswer`, declined by one
    `Decline` whose `ask_ref` names it, or both: an answer for the part the
    plan computes and a decline naming, in `Decline.ask`, the part it does
    not. An answer names any requirement the plan states, one another
    requirement reads included. `translator.mechanical.build_translation`
    makes each a terminal, so its result reaches the answer. A number past
    the asks listed is a malformed draft. A gap, two answers or two declines
    is `AsksUnaccountedError`.
    """
    answer_numbers = [answer.ask for answer in answered]
    decline_numbers = [decline.ask_ref for decline in declined if decline.ask_ref is not None]
    numbers = answer_numbers + decline_numbers
    unlisted = sorted(set(numbers) - set(range(len(asks))))
    if unlisted:
        listed = f"numbered 0 to {len(asks) - 1}" if asks else "none"
        raise ValueError(f"the brief lists {len(asks)} asks ({listed}); no ask {unlisted!r}")
    ids = {requirement.id for requirement in requirements}
    outputs = {output for answer in answered for output in answer.outputs}
    if unknown := sorted(outputs - ids):
        raise ValueError(
            f"an ask is answered by requirements this plan does not state: {unknown!r}"
        )
    uncovered = [number for number in range(len(asks)) if number not in numbers]
    repeated = [
        number
        for number in range(len(asks))
        if answer_numbers.count(number) > 1 or decline_numbers.count(number) > 1
    ]
    if uncovered or repeated:
        raise AsksUnaccountedError(asks=asks, uncovered=uncovered, repeated=repeated)


class DecisionsUncitedError(ValueError):
    """A plan stating requirements makes decisions none of them cites.

    `slots` are those decisions' slots, sorted (`check_references`).
    """

    def __init__(self, slots: Iterable[str]) -> None:
        self.slots = tuple(sorted(slots))
        super().__init__(f"no evidence requirement cites the decisions {list(self.slots)!r}")


class WholeIsItsMemberError(ValueError):
    """A share or quotient answering an ask that totals across a party divides by its own members of that party.

    `requirement_id` is the share or quotient, `ask` the ask it answers,
    `party` the party the ask totals across, `denominator_id` the dataset
    requirement its denominator reads, and `named` the entities that
    requirement names alone on that side, each of which the numerator names
    too (`check_wholes_span_their_totals`). `planner.agent.propose_plan`
    answers it with a typed problem naming the ways out.
    """

    def __init__(
        self,
        *,
        requirement_id: str,
        ask: int,
        ask_text: str,
        party: str,
        denominator_id: str,
        named: Collection[str],
    ) -> None:
        self.requirement_id = requirement_id
        self.ask = ask
        self.party = party
        self.denominator_id = denominator_id
        self.named = tuple(sorted(named))
        super().__init__(
            f"{requirement_id!r} answers ask {ask} ({ask_text!r}), which totals across "
            f"{party}, but its denominator ({denominator_id!r}) names {list(self.named)!r} "
            f"alone as {party}, each of which its numerator names too, so it divides the "
            "member by itself"
        )


class AskRoleDroppedError(ValueError):
    """A plan answers an ask without keeping a party the ask breaks its answer down by.

    `dropped` holds each such ask's dropped parties (`roles_not_kept`) and
    `together` every party of each of those asks that breaks down by both
    parties of a flow, which one output keeps at once, each in the word its
    `AskBreakdown.by` holds. `planner.agent.propose_plan` answers
    it with a typed problem naming each ask and party, so the model keeps
    the party or declines that part.
    """

    def __init__(
        self,
        *,
        asks: Sequence[str],
        dropped: Mapping[int, Sequence[str]],
        together: Mapping[int, Sequence[str]] | None = None,
    ) -> None:
        self.dropped = {number: tuple(roles) for number, roles in dropped.items()}
        self.together = {number: tuple(roles) for number, roles in (together or {}).items()}
        named = "; ".join(
            f"ask {number} ({asks[number]!r}) by {', '.join(roles)}"
            + (
                f", which no one output keeps with the rest of {', '.join(self.together[number])}"
                if number in self.together
                else ""
            )
            for number, roles in self.dropped.items()
        )
        super().__init__(
            "the requirements answering an ask keep each party it breaks down by, open in "
            f"`across` or named in `entities`, and these drop one: {named}"
        )


def roles_not_kept(plan: Plan) -> dict[int, tuple[PartyRole, ...]]:
    """Each ask of `plan.ask_breakdowns` whose answer drops a party it breaks down by, with those parties.

    An output keeps a party when a role dimension of its shape names it, in
    the ask's word or another on the same side of the flow
    (`planner.parties.same_party`), since the frame then holds one row per
    member: a dataset requirement's `across`, or its entities named with
    that role, carried to the output as the shape algebra carries it
    (`frame_shapes`). An output is one flow, with one party on each side,
    so side decides. A
    breakdown's dimension holds a classification's codes and keeps no party. The
    parties of one ask are one result, so an answer keeps the breakdown when
    one of its outputs keeps every party at once; an output per party keeps
    none of the pairs. Otherwise the parties dropped are the ones the output
    keeping the most of them lacks, the first such output on a tie: the
    shortest way to an output that keeps them all. An ask `answered` does
    not answer keeps none.
    """
    if not plan.ask_breakdowns:
        return {}
    shapes = frame_shapes(plan)
    outputs = {answer.ask: answer.outputs for answer in plan.answered}
    dropped: dict[int, tuple[PartyRole, ...]] = {}
    for breakdown in plan.ask_breakdowns:
        kept = [
            [
                dimension.role
                for dimension in shapes[output].roles
                if dimension.role is not None and not dimension.breakdown
            ]
            for output in outputs.get(breakdown.ask, ())
        ]
        lacking = [
            tuple(role for role in breakdown.by if not party_among(role, roles)) for roles in kept
        ]
        if missing := min(lacking, key=len, default=breakdown.by):
            dropped[breakdown.ask] = missing
    return dropped


def _check_asks_keep_their_parties(plan: Plan) -> None:
    """Reject a plan whose answer to an ask drops a party the ask breaks down by.

    Each breakdown names an ask the brief lists, once. An answered ask whose
    answer drops a party, or keeps its parties only in separate outputs
    (`roles_not_kept`), needs a `BREAKDOWN_NOT_KEPT`
    decline naming it, since a decline under another code names another part
    of the ask. Without one the plan raises `AskRoleDroppedError`.
    """
    asks, breakdowns = plan.asks, plan.ask_breakdowns
    numbers = [breakdown.ask for breakdown in breakdowns]
    if unlisted := sorted(set(numbers) - set(range(len(asks)))):
        raise ValueError(f"the brief lists {len(asks)} asks; no breakdown of ask {unlisted!r}")
    _reject_repeats([str(number) for number in numbers], what="ask breakdown")
    exempt = {
        decline.ask_ref
        for decline in plan.declined
        if decline.code is UnsupportedOperation.BREAKDOWN_NOT_KEPT
    }
    answer_numbers = {answer.ask for answer in plan.answered}
    dropped = {
        number: roles
        for number, roles in roles_not_kept(plan).items()
        if number in answer_numbers and number not in exempt
    }
    if dropped:
        together = {
            breakdown.ask: breakdown.by
            for breakdown in breakdowns
            if breakdown.ask in dropped and len(breakdown.by) > 1
        }
        raise AskRoleDroppedError(asks=asks, dropped=dropped, together=together)


def side_entities(
    requirements_by_id: Mapping[str, EvidenceRequirement], side: RequirementSide
) -> tuple[Entity, ...]:
    """The entities a calculation's `side` names, its requirements looked up in `requirements_by_id`.

    A side names the entities of the dataset requirement it reads, directly
    or through aggregates; narrowed to one `entity`, it names the entities
    of that name alone, or that name with no role when it reads no dataset
    requirement naming it. A side reading through any other kind names none.
    """
    dataset = _side_dataset(requirements_by_id, side.input_id)
    entities = () if dataset is None else dataset.entities
    if side.entity is None:
        return entities
    return tuple(entity for entity in entities if entity.name == side.entity) or (
        Entity(name=side.entity),
    )


def _side_dataset(
    requirements_by_id: Mapping[str, EvidenceRequirement], input_id: str
) -> DatasetRequirement | None:
    """The dataset requirement `input_id` reads, directly or through aggregates, or `None` through any other kind."""
    item = requirements_by_id[input_id]
    while isinstance(item, AggregateRequirement):
        item = requirements_by_id[item.input_id]
    return item if isinstance(item, DatasetRequirement) else None


def no_refs(_dataset: DatasetRequirement | None, _entity: Entity) -> frozenset[str]:
    """No graph ref, so the entity is compared by its name (`check_wholes_span_their_totals`)."""
    return frozenset()


def check_wholes_span_their_totals(
    plan: Plan,
    *,
    refs: Callable[[DatasetRequirement | None, Entity], frozenset[str]] = no_refs,
) -> None:
    """Reject a share or quotient whose denominator, on a party its ask totals across, names the numerator's members alone.

    An ask's `over` (`AskBreakdown.over`) is a party its answer totals
    across, as "the share of ODA to Kenya from the United States" totals
    across donors, so a share_of or quotient_of computing that answer, as
    an output of the ask or upstream of one, divides by a whole spanning
    that party. When every entity its denominator names on that side is one
    its numerator names, the whole holds the numerator's members alone and
    the result divides the member by itself, whatever source each side
    reads: the United States' ODA to Kenya over a total naming the United
    States and Kenya. A denominator naming no entity on that side keeps it
    open or reads a source's total, and one naming another entity there
    too, as G7 beside Germany or the United Kingdom beside the United
    States, holds a whole the user may have named; neither is refused. A
    share keyed by that party and pooled across it divides each member by
    the members' total, a whole spanning more than any one member once it
    holds two, so only a pool of one is refused. A share pooling its own
    axis from one side is checked by `_pools_one_member`.

    The check is open where `over` is empty: an ask the orchestrator handed
    with no party totalled across is not checked, which is right for a
    share by measure, as health ODA within all ODA to one donor and
    recipient. The orchestrator states `over` for the side a search's
    `share_of` operation runs over (`harness.orchestrator_tools._breakdowns_missing`).
    An ask listing a party in `by` is answered by an output keeping it
    (`_check_asks_keep_their_parties`), which an unkeyed share does not.

    Each entity is compared by the graph refs `refs` resolves it to, given
    the dataset requirement its side reads, `None` through any other kind,
    and by its name where either entity of a pair resolves none
    (`_among_members`). `Plan` validation resolves none (`no_refs`), so it
    compares names as the plan writes them, and an alias, or a group on one
    side beside its members on the other, is not seen there.
    `submission.service.admit` checks again with the refs each side's pinned
    source resolves its names to, so "United States" and "United States of
    America" compare equal when both resolve to one ref. The requirements
    are those the model wrote (`authored`), so a refusal names its ids.

    Raises:
        WholeIsItsMemberError: For the first such share or quotient.
    """
    over = {breakdown.ask: breakdown.over for breakdown in plan.ask_breakdowns if breakdown.over}
    if not over:
        return
    written = authored(plan).evidence_requirements
    by_id = {requirement.id: requirement for requirement in written}
    for answer in plan.answered:
        words = over.get(answer.ask, ())
        if not words:
            continue
        upstream = {key for output in answer.outputs for key in _upstream(output, by_id)}
        for requirement in written:
            if (
                requirement.id not in upstream
                or not isinstance(requirement, ShareOfRequirement | QuotientRequirement)
                or (isinstance(requirement, ShareOfRequirement) and requirement.pools_its_own_axis)
            ):
                continue
            numerator = side_entities(by_id, requirement.numerator)
            denominator = side_entities(by_id, requirement.denominator)
            numerator_dataset = _side_dataset(by_id, requirement.numerator.input_id)
            dataset = _side_dataset(by_id, requirement.denominator.input_id)
            for word in words:
                held = [(entity, refs(dataset, entity)) for entity in _on_side(denominator, word)]
                part = [
                    (entity, refs(numerator_dataset, entity))
                    for entity in _on_side(numerator, word)
                ]
                if not held or not all(_among_members(one, part) for one in held):
                    continue
                # A pool's members are every ref its entities resolve to, a
                # group counting each of its members, and the names that resolve none.
                members = {ref for _, found in held for ref in found} | {
                    entity.name for entity, found in held if not found
                }
                if len(members) > 1 and _pools_across(requirement, word):
                    continue
                raise WholeIsItsMemberError(
                    requirement_id=requirement.id,
                    ask=answer.ask,
                    ask_text=plan.asks[answer.ask],
                    party=word,
                    denominator_id=requirement.denominator.input_id
                    if dataset is None
                    else dataset.id,
                    named={entity.name for entity, _ in held},
                )


def _pools_across(requirement: ShareOfRequirement | QuotientRequirement, word: str) -> bool:
    """Whether `requirement` is a share keyed by the party `word` names and pooled across it."""
    return (
        isinstance(requirement, ShareOfRequirement)
        and requirement.denominator_scope is DenominatorScope.POOLED
        and requirement.axis is not None
        and same_party(requirement.axis, word)
    )


def _among_members(
    resolved: tuple[Entity, frozenset[str]], part: Sequence[tuple[Entity, frozenset[str]]]
) -> bool:
    """Whether an entity of a whole, with the refs it resolves to, is among the members `part` names.

    By ref, when every ref it resolves to is one `part`'s entities resolve
    to, so a group the part names stands for each of its members; by name,
    against an entity of `part` with the same name where either of the two
    resolves no ref.
    """
    entity, found = resolved
    if found and found <= {ref for _, theirs in part for ref in theirs}:
        return True
    return any(one.name == entity.name for one, theirs in part if not (found and theirs))


def _on_side(entities: Iterable[Entity], word: str) -> tuple[Entity, ...]:
    """The entities of `entities` whose role names the party `word` names (`planner.parties.same_party`)."""
    return tuple(
        entity for entity in entities if entity.role is not None and same_party(entity.role, word)
    )


def _check_asks_across_years(asks: Sequence[str], numbers: Sequence[int]) -> None:
    """Reject a plan whose across-years asks name an ask the brief does not list, or one twice."""
    if unlisted := sorted(set(numbers) - set(range(len(asks)))):
        raise ValueError(
            f"the brief lists {len(asks)} asks; no ask {unlisted!r} to put across years"
        )
    _reject_repeats([str(number) for number in numbers], what="ask across years")


# How a calculation reading a comparison or a join is named to the user.
_READER_PHRASES = {
    "aggregate": "a total or other summary",
    "growth": "a growth rate",
    "rank": "a ranking",
    "share_of": "a share",
    "quotient_of": "a quotient",
    "sum_of": "a sum",
    "filter": "a filter",
    "comparison": "a comparison",
    "join": "a join",
}
_COMBINED_PHRASES = {
    "comparison": "the difference between two series",
    "join": "two series combined",
}


class AnswerMode(TitlelessEnum):
    """What kind of answer a plan's requirements have to add up to.

    `series` answers a question that asks to see a level or a trend: a lone
    dataset requirement, unread by anything else, is itself the answer, its
    frame carrying one series per entity for a chart to draw. `computed`
    answers a question that asks for a figure the series does not state on
    its own -- a difference, growth, share, quotient, rank or total -- so every dataset
    requirement has to feed a calculation; one left unread would leave the
    plan without the figure the question asked for. Either way, `answered`
    may name a dataset requirement a calculation reads.
    """

    SERIES = "series"
    COMPUTED = "computed"


def check_references(
    requirements: Sequence[EvidenceRequirement],
    *,
    slots: Sequence[str],
    declines: Iterable[str],
    made: Collection[str] | None,
) -> None:
    """Reject a plan's parts that do not name each other once each.

    Slots (`slots`), declines by key (`declines`) and requirement ids are
    unique, so a citation is unambiguous. When `made`, every slot the plan
    makes, is given, every citation names one of them. Once `requirements`
    holds any, a requirement cites each of `slots` (`DecisionsUncitedError`),
    and every input a requirement reads names one of `requirements`, with no
    cycle.

    `Plan` validation passes every decision as both `slots` and `made`.
    `propose_plan` checks the model's drafts the same way before any Search
    read the plan waits on (`submission.service._draft_unreferenced`), since
    none of these rules reads a pin.

    Raises:
        DecisionsUncitedError: If a decision of `slots` is cited by none of
            `requirements`.
        ValueError: If any other rule is broken.
    """
    _reject_repeats(slots, what="decision slot")
    _reject_repeats(declines, what="decline")
    _reject_repeats(
        [requirement.id for requirement in requirements], what="evidence requirement id"
    )
    cited = {
        decision_id for requirement in requirements for decision_id in requirement.decision_ids
    }
    if made is not None:
        unknown = sorted(cited - set(made))
        if unknown:
            raise ValueError(
                f"evidence requirements cite decisions the plan does not make: {unknown!r}"
            )
    if requirements:
        uncited = set(slots) - cited
        if uncited:
            raise DecisionsUncitedError(uncited)
    _check_reads_resolve({r.id: r for r in requirements})


def _check_reads_resolve(requirements_by_id: Mapping[str, EvidenceRequirement]) -> None:
    """Reject requirements reading one the plan does not state, or reading in a cycle."""
    read_ids = {
        input_id
        for requirement in requirements_by_id.values()
        for input_id in inputs_of(requirement)
    }
    unknown_inputs = sorted(read_ids - set(requirements_by_id))
    if unknown_inputs:
        raise ValueError(
            f"evidence requirements read requirements this plan does not state: {unknown_inputs!r}"
        )
    check_acyclic(requirements_by_id)


def check_acyclic(requirements_by_id: Mapping[str, EvidenceRequirement]) -> None:
    """Reject requirements reading in a cycle (`_find_cycle`), so a walk down what each
    reads ends. An input `requirements_by_id` does not hold is left for
    `_check_reads_resolve` to refuse.
    """
    cycle = _find_cycle(requirements_by_id)
    if cycle is not None:
        raise ValueError(f"evidence requirements do not form a DAG; cycle: {list(cycle)!r}")


def _check_wholes_name_their_parts(
    requirements_by_id: Mapping[str, EvidenceRequirement],
) -> None:
    """Reject a `whole_of` that cannot be compiled from its part (`whole_of_fault`)."""
    for requirement in requirements_by_id.values():
        if isinstance(requirement, DatasetRequirement) and (
            fault := whole_of_fault(requirement, requirements_by_id)
        ):
            raise ValueError(fault)


def whole_of_fault(
    whole: DatasetRequirement, requirements_by_id: Mapping[str, EvidenceRequirement]
) -> str | None:
    """Why the `whole_of` `whole` states cannot be compiled from its part, `None` when
    it can or `whole` states none.

    The whole is compiled from the part's pin with only the part's scope
    widened, so the part is another dataset requirement, itself no whole.
    A whole over every sector counts the part's parties. A whole over every
    donor or recipient holds the part only where the part names a party on
    that side: it states the part's other entities, and `across` the part's
    words with one word for that side added (`_entity_whole_fault`). It
    follows its part wherever the part is counted, so it is read only as
    the denominator of a share or quotient of its part (`_whole_read_fault`).
    """
    if whole.whole_of is None:
        return None
    part_id = whole.whole_of.part
    part = requirements_by_id.get(part_id)
    if not isinstance(part, DatasetRequirement) or part_id == whole.id:
        return (
            f"{whole.id!r} states whole_of part {part_id!r}, which is not another dataset "
            "requirement of this plan"
        )
    if part.whole_of is not None:
        return (
            f"{whole.id!r} states whole_of part {part_id!r}, which is a whole itself; name the "
            "part both are counted from"
        )
    if whole.whole_of.axis != "sector":
        if fault := _entity_whole_fault(whole, part, whole.whole_of.axis):
            return fault
    elif {*whole.entities, *whole.across} != {*part.entities, *part.across}:
        return (
            f"{whole.id!r} is the whole of {part_id!r}, counted from its dataset for the same "
            "parties, so it states the same entities and `across`"
        )
    return _whole_read_fault(whole, part_id, requirements_by_id)


def _entity_whole_fault(
    whole: DatasetRequirement, part: DatasetRequirement, axis: str
) -> str | None:
    """Why `whole`, the whole of `part` over every `axis` (a donor or a recipient),
    does not hold `part`, `None` when it does.

    The part names a party on the side of the flow `axis` is on. The whole names
    the part's other entities and states the part's `across` words with one word
    for that side added, so it is the total over every `axis` for the part's
    other parties. Where Search answers the whole with that side bound to one
    group, as DAC1 binds DAC countries' total, the translator names the group on
    the side in place of the word (`translator.discovery.name_pinned_groups`), so
    a whole naming one entity on that side, other than the part's, and stating
    only the part's `across` words holds the part too. Which group Search binds
    is known only once the whole is compiled, so admission refuses a whole
    naming any other entity there (`submission.service._misnamed_total`).
    """
    side = role_party(axis)
    named = [e for e in part.entities if e.role is not None and role_party(e.role) == side]
    if not named and (roleless := [e.name for e in part.entities if e.role is None]):
        return (
            f"{whole.id!r} is the whole of {part.id!r} over every {axis}, and the side of "
            f"{part.id!r}'s entities ({', '.join(sorted(roleless))}) is not stated; state each "
            "entity's role, donor or recipient"
        )
    if not named:
        return (
            f"{whole.id!r} is the whole of {part.id!r} over every {axis}, and {part.id!r} names "
            f"no {axis} to leave open; state the whole over another axis, or a total of its own "
            "without `whole_of`"
        )
    others = {*part.entities} - {*named}
    spans = any(across_party(word) == side for word in whole.across)
    kept = {word for word in whole.across if across_party(word) != side}
    on_side = {e for e in whole.entities if e.role is not None and role_party(e.role) == side}
    # The side spanned by `across`, or closed to one group other than the part's party.
    opened = not on_side if spans else len(on_side) == 1 and not on_side & {*named}
    if {*whole.entities} - on_side != others or not opened or kept != {*part.across}:
        return (
            f"{whole.id!r} is the whole of {part.id!r} over every {axis}, so it states "
            f"{part.id!r}'s entities without its {axis}s "
            f"({', '.join(sorted(e.name for e in named))}), and `across` "
            f"{part.id!r}'s words with {axis!r} added"
        )
    return None


def _dataset_reach(
    requirements_by_id: Mapping[str, EvidenceRequirement],
) -> Callable[[str], set[str]]:
    """The dataset requirements a requirement or input reads, directly or through
    the calculations between, by its id."""
    reads: dict[str, Sequence[str]] = {
        input_id: ()
        for requirement in requirements_by_id.values()
        for input_id in inputs_of(requirement)
    }
    reads |= {rid: inputs_of(requirement) for rid, requirement in requirements_by_id.items()}

    def datasets(node: str) -> set[str]:
        return {
            rid
            for rid in reached(node, reads)
            if isinstance(requirements_by_id.get(rid), DatasetRequirement)
        }

    return datasets


def requirements_over(
    whole_id: str, requirements_by_id: Mapping[str, EvidenceRequirement]
) -> set[str]:
    """The ids of the whole `whole_id` and of every calculation over it alone, such as
    the whole summed by year: the reads that end in a divisor of the part."""
    datasets = _dataset_reach(requirements_by_id)
    return {rid for rid in requirements_by_id if datasets(rid) == {whole_id}}


def _whole_read_fault(
    whole: DatasetRequirement,
    part_id: str,
    requirements_by_id: Mapping[str, EvidenceRequirement],
) -> str | None:
    """Why `whole`, the whole of `part_id`, is read other than as the denominator of a
    share or quotient of its part, `None` when it is read only so.

    The whole is read directly or through calculations over it alone, such
    as the whole summed by year. Each such read ends in a share or quotient
    whose denominator reads it and whose numerator reads no dataset but the
    part, directly or summed the same way. Anything else reading it, and a
    read that ends unread, takes a plain total of its own.
    """
    datasets = _dataset_reach(requirements_by_id)
    over_whole = requirements_over(whole.id, requirements_by_id)
    readers: dict[str, list[EvidenceRequirement]] = {rid: [] for rid in over_whole}
    for reader in requirements_by_id.values():
        for input_id in inputs_of(reader):
            if input_id in readers:
                readers[input_id].append(reader)
    rule = (
        f"{whole.id!r} states `whole_of`, so it is read only as the denominator of a "
        f"share_of or quotient_of whose numerator is its part {part_id!r}"
    )
    for read_id in sorted(over_whole):
        if not readers[read_id]:
            return (
                f"{rule}, and nothing reads {read_id!r}. For a total of its own, state a "
                "plain total: another dataset requirement without `whole_of`"
            )
        for reader in readers[read_id]:
            if reader.id in over_whole or (
                isinstance(reader, ShareOfRequirement | QuotientRequirement)
                and reader.denominator.input_id == read_id
                and datasets(reader.numerator.input_id) == {part_id}
            ):
                continue
            return (
                f"{rule}, and {reader.id!r} reads {read_id!r} otherwise. For "
                f"{reader.id!r}, state a plain total: another dataset requirement without "
                "`whole_of`, read in its place"
            )
    return None


def _check_requirement_graph(
    requirements_by_id: dict[str, EvidenceRequirement],
    *,
    answer_mode: AnswerMode,
    whole_history: Mapping[str, int],
    origins: Mapping[str, str],
) -> None:
    """Reject requirements that do not agree with each other.

    The requirements read only requirements they state and form a DAG
    (`_check_reads_resolve`, which `Plan` runs on the requirements the model
    wrote; a copy the split adds reads only copies and their origins).
    Nothing reads a comparison or a join -- both are terminal, and a calculation over either is `UnsupportedPlanError`, the one
    rule here that refuses what the system cannot do rather than a malformed
    draft -- and every requirement's output shape is legal
    under `planner.shapes.rejects`: a side's entity or a rank axis names a
    role the shape it reads actually carries, a growth window fits that
    shape's own period and keeps both its years there as newer data arrives
    (`planner.shapes.growth_endpoint_leaves`), and a rank without a reduction leaves one value per
    entity to rank on. A period that follows the newest data rests only on
    dataset requirements stating `period_end_open`
    (`_check_rolling_reads_open_periods`), and every period a run reads from
    one newest year states that year, one the datasets it rests on hold
    (`_check_rolling_years_agree`). An `ne`
    removes a fixed year from fixed years (`_check_exclusions_fixed`).
    `whole_history` and `origins` are as `_build_shapes` takes them.

    Under `AnswerMode.SERIES`, a dataset requirement left unread by every
    other requirement is legal: it is the answer. Under `AnswerMode.COMPUTED`,
    the same requirement is rejected -- a retrieved series carries a
    reference and statistics, not the figure a computed answer asks for.
    """
    requirements = tuple(requirements_by_id.values())
    read_ids = {input_id for requirement in requirements for input_id in inputs_of(requirement)}
    terminal_kinds = {r.id: r.kind for r in requirements if isinstance(r, _TERMINAL_REQUIREMENTS)}
    over_combined = sorted(
        f"{_READER_PHRASES[reader.kind]} of {_COMBINED_PHRASES[terminal_kinds[input_id]]} "
        f"({reader.id!r} reads {input_id!r})"
        for reader in requirements
        for input_id in inputs_of(reader)
        if input_id in terminal_kinds
    )
    if over_combined:
        raise UnsupportedPlanError(
            UnsupportedOperation.CALCULATION_OVER_COMBINED,
            "this system cannot compute a calculation over the result of a comparison or a "
            f"join: {'; '.join(over_combined)}",
        )
    if answer_mode == AnswerMode.COMPUTED:
        # A copy read stands for its origin read in the plan the model wrote.
        read_ids |= {origins[input_id] for input_id in read_ids if input_id in origins}
        unread_datasets = sorted(
            r.id for r in requirements if is_dataset(r) and r.id not in read_ids
        )
        if unread_datasets:
            raise ValueError(
                "answer_mode is computed, so a retrieved series carries a reference and "
                "statistics, not the figure the question asks for; unread dataset "
                f"requirements: {unread_datasets!r}"
            )
    _check_exclusions_fixed(requirements_by_id)
    _check_rolling_reads_open_periods(requirements_by_id)
    _check_rolling_years_agree(requirements_by_id)
    _build_shapes(requirements_by_id, whole_history=whole_history, origins=origins)


def rolls(requirement: EvidenceRequirement) -> bool:
    """Whether a period `requirement` states follows the newest data."""
    if isinstance(requirement, GrowthRequirement | RankRequirement):
        return requirement.rolling is not None
    if isinstance(requirement, FilterRequirement):
        return any(
            isinstance(predicate, PeriodThreshold) and predicate.rolling is not None
            for predicate in requirement.predicates
        )
    return False


# The comparisons under which a threshold's year is the earliest year kept.
_LOWER_BOUND_OPS = frozenset({"gt", "gte", "eq"})


def rolls_from_its_start(requirement: EvidenceRequirement) -> bool:
    """Whether the earliest year `requirement` reads follows the newest data.

    A growth or a rank whose `rolling` names `window_start`, and a filter
    whose rolling threshold keeps the years at or after it. Such a
    requirement can read a year earlier than its dataset's stated period,
    once the newest year the data holds sits behind the one the plan read.
    """
    if isinstance(requirement, GrowthRequirement | RankRequirement):
        return requirement.rolling is not None and requirement.rolling.window_start is not None
    if isinstance(requirement, FilterRequirement):
        return any(
            isinstance(predicate, PeriodThreshold)
            and predicate.rolling is not None
            and predicate.op in _LOWER_BOUND_OPS
            for predicate in requirement.predicates
        )
    return False


def dataset_roots(
    requirements_by_id: Mapping[str, EvidenceRequirement], requirement_id: str
) -> frozenset[str]:
    """The ids of the dataset requirements `requirement_id`'s result rests on, itself included."""
    requirement = requirements_by_id[requirement_id]
    if isinstance(requirement, DatasetRequirement):
        return frozenset({requirement_id})
    return frozenset().union(
        *(dataset_roots(requirements_by_id, input_id) for input_id in inputs_of(requirement))
    )


def reached(node: str, reads: Mapping[str, Iterable[str]]) -> set[str]:
    """`node` and every node it reads under `reads`, directly or through others."""
    found: set[str] = set()
    pending = [node]
    while pending:
        current = pending.pop()
        if current not in found:
            found.add(current)
            pending.extend(reads[current])
    return found


def rolling_groups(
    reads: Mapping[str, Iterable[str]], *, rolling: Sequence[str]
) -> list[list[str]]:
    """`rolling` split into the groups that count from one newest year.

    Two nodes share a group when some node of `reads` reads both, directly
    or through others, and the relation is transitive. Each group and the
    list of them keep the order of `rolling`. A plan's requirements and a
    workflow's steps group alike (`compiler.rolling.anchor_groups`), so the
    rule a plan is held to is the one its run resolves by.
    """
    group_of = {node: {node} for node in rolling}
    for node in reads:
        below = reached(node, reads)
        together = [member for member in rolling if member in below]
        merged = set().union(*(group_of[member] for member in together))
        for member in merged:
            group_of[member] = merged
    return [
        [member for member in rolling if member in group_of[node]]
        for node in rolling
        if node == min(group_of[node], key=rolling.index)
    ]


def anchored_over(
    reads: Mapping[str, Iterable[str]], *, rolling: Sequence[str], retrievals: Collection[str]
) -> dict[str, frozenset[str]]:
    """The retrievals each node of `reads` has the newest year it counts from resolved over.

    A node counts from the newest year of the one group of `rolling` it
    reaches (`rolling_groups`), and the run resolves that year over the
    retrievals the group's members reach, whatever reads them beside or
    above (`compiler.rolling.anchor_groups`). A node reaching no rolling
    node maps to an empty set.
    """
    found = dict.fromkeys(reads, frozenset[str]())
    for members in rolling_groups(reads, rolling=rolling):
        below = set().union(*(reached(member, reads) for member in members))
        roots = frozenset(below.intersection(retrievals))
        for node in reads:
            if not reached(node, reads).isdisjoint(members):
                found[node] = roots
    return found


def _rolling_reads(
    requirements_by_id: Mapping[str, EvidenceRequirement], *, joining: str | None = None
) -> tuple[dict[str, tuple[Identifier, ...]], list[str]]:
    """What each requirement reads, and the requirements that roll, `joining` among them when named."""
    reads = {
        requirement_id: inputs_of(requirement)
        for requirement_id, requirement in requirements_by_id.items()
    }
    rolling = [
        requirement_id
        for requirement_id, requirement in requirements_by_id.items()
        if rolls(requirement) or requirement_id == joining
    ]
    return reads, rolling


def passes_observation_years(
    requirements_by_id: Mapping[str, EvidenceRequirement], reader_id: str, dataset_id: str
) -> bool:
    """Whether `reader_id` reads dataset requirement `dataset_id` through steps that keep its years.

    Every step on the way, `dataset_id` included and `reader_id` left out,
    has to pass the years each row was observed in up to the next: a growth
    does not, nor does a step that combines the years it reads
    (`_licence_for`), and a two-sided step does only when both its sides
    do. A start that follows the newest data, read through such steps,
    bounds the years the dataset holds; read above a collapse, it bounds
    the years the collapse picked or summed, and the dataset keeps its
    stated period (`whole_history_split`).
    """
    return any(
        dataset_id in dataset_roots(requirements_by_id, input_id)
        and _keeps_observation_years(requirements_by_id, input_id)
        for input_id in inputs_of(requirements_by_id[reader_id])
    )


def _keeps_observation_years(
    requirements_by_id: Mapping[str, EvidenceRequirement], step_id: str
) -> bool:
    """Whether `step_id` and every step it is computed from pass their input's years on (`passes_observation_years`)."""
    step = requirements_by_id[step_id]
    if isinstance(step, DatasetRequirement):
        return True
    if isinstance(step, GrowthRequirement) or _licence_for(step) is not None:
        return False
    return all(
        _keeps_observation_years(requirements_by_id, input_id) for input_id in inputs_of(step)
    )


class HistorySplit(NamedTuple):
    """A plan's requirements once every read of a dataset's whole history serves only its rolling-start readers.

    `requirements` are the requirements as split, `copies` each clone with
    its origin (the plan's own `rolling_copies` first), and `whole_history`
    each dataset requirement read from its first year on record, by id,
    with that year, the year its shape starts at (`_build_shapes`).
    `first_years_read` names the model's dataset requirements whose first
    year the split reads, and `first_year_unknown` those of them the plan's
    `first_years` holds no year for (`whole_history_split`).
    """

    requirements: tuple[EvidenceRequirement, ...]
    copies: tuple[RollingCopy, ...]
    whole_history: dict[str, int]
    first_years_read: tuple[str, ...]
    first_year_unknown: tuple[str, ...]

    def origins(self) -> dict[str, str]:
        """Each clone's id mapped to its origin's."""
        return copy_origins(self.copies)


def copy_origins(copies: Iterable[RollingCopy]) -> dict[str, str]:
    """Each copy's id in `copies` mapped to the id of the requirement it copies."""
    return {copy.copy_id: copy.origin_id for copy in copies}


def origin_of(split: HistorySplit, step_id: str) -> str:
    """The id of the model's requirement `step_id` copies, or `step_id` itself where it copies none.

    Outward text names a step by it: a refusal's requirement ids and the
    dataset requirement a period-reduction gap reads.
    """
    return split.origins().get(step_id, step_id)


def authored_text(text: str, origins: Mapping[str, str]) -> str:
    """`text` naming, in place of each copy `origins` maps, the requirement it copies.

    `origins` maps each copy's id to its origin's (`copy_origins`,
    `HistorySplit.origins`). Each copy id is matched as a whole identifier in
    prose, so an id the model wrote that only starts with one
    (`aid_history_2` beside `aid_history`) keeps its name, and so does a ref
    holding one: a path or dcid segment (`dc/topic/aid_history`,
    `/aid_history.csv`) or a prefixed name (`dc:aid_history`), where a `/`,
    `.` or `:` joins it to another identifier character or a `/` follows.
    Outward text built on the split plan reads through it: the refusals its
    rules raise (`Plan`'s validation), the referrals the translator hands
    back (`translator.tool.authored_answer`), and the prose of what a run
    reports (`harness.graph.plan_done.authored_report`).
    """
    if not origins:
        return text
    copy_id = re.compile(
        r"(?<![A-Za-z0-9_/.:])("
        + "|".join(map(re.escape, origins))
        + r")(?![A-Za-z0-9_/]|[.:][A-Za-z0-9_])"
    )
    return copy_id.sub(lambda match: origins[match[1]], text)


def whole_history_split(
    requirements: Sequence[EvidenceRequirement],
    *,
    answering: Collection[str],
    first_years: Iterable[FirstYear],
    copies: Iterable[RollingCopy],
) -> HistorySplit:
    """`requirements` split so a dataset read from its first year serves only its rolling-start readers.

    A dataset requirement D stating `period_end_open` is read from its first
    year on record by its rolling-start readers R: each requirement
    `rolls_from_its_start` holds for that reads D through steps keeping its
    years (`passes_observation_years`). Its first year is the one
    `first_years` holds for D, or for the origin `copies` names when D is a
    clone. Where that year is at or after D's stated start, the whole
    history and the stated period hold the same rows and nothing splits.

    Otherwise let H be the steps on R's paths from D, D included and R left
    out, and O those of H that a step outside H and R reads or that answer
    the plan (`answering`). With O empty, D itself is the whole-history
    read. Else O and every step of H a step of O is computed from are each
    copied under the first free id of `{stem}_history`, `{stem}_history_2`
    and on (`stem` the origin's first 52 characters), placed after the
    origin, with each input among them naming its copy, and every other
    step of H and every reader in R reading a step among them reads its
    copy instead. The copy of D is the whole-history read, and the
    originals keep their stated years for every other reader. Nothing is
    dropped, so no origin is left unread.

    On its own output, with its `copies`, it changes nothing, since every
    whole-history read is then read only by its rolling-start readers and
    the steps between.
    """
    years = {entry.requirement_id: entry.year for entry in first_years}
    made = list(copies)
    origins = copy_origins(made)
    split = list(requirements)
    whole: dict[str, int] = {}
    read: list[str] = []
    unknown: list[str] = []
    datasets = [r for r in requirements if isinstance(r, DatasetRequirement) and r.period_end_open]
    for dataset in datasets:
        by_id = {requirement.id: requirement for requirement in split}
        readers = {
            reader.id
            for reader in split
            if rolls_from_its_start(reader)
            and passes_observation_years(by_id, reader.id, dataset.id)
        }
        if not readers:
            continue
        origin = origins.get(dataset.id, dataset.id)
        read.append(origin)
        first_year = years.get(origin)
        if first_year is None:
            unknown.append(origin)
            continue
        if first_year >= dataset.period.start:
            continue
        cloned, rerouted = _history_copies(by_id, dataset.id, readers=readers, answering=answering)
        if not cloned:
            whole[dataset.id] = first_year
            continue
        split = _with_copies(split, cloned, rerouted=rerouted)
        made.extend(RollingCopy(copy_id=copy_id, origin_id=step_id) for step_id, copy_id in cloned)
        whole[dict(cloned)[dataset.id]] = first_year
    return HistorySplit(
        requirements=tuple(split),
        copies=tuple(made),
        whole_history=whole,
        first_years_read=tuple(read),
        first_year_unknown=tuple(unknown),
    )


def _history_copies(
    requirements_by_id: Mapping[str, EvidenceRequirement],
    dataset_id: str,
    *,
    readers: Collection[str],
    answering: Collection[str],
) -> tuple[list[tuple[str, str]], set[str]]:
    """The steps `whole_history_split` copies for `dataset_id`, each with its copy's id, and the steps to read the copies.

    The copies are in plan order, and none where no step between the
    dataset and its rolling-start `readers` is read by anything else or
    answers the plan. The steps to read them are the `readers` and the
    steps between left uncopied.
    """
    reads_of: dict[str, list[str]] = {step_id: [] for step_id in requirements_by_id}
    for requirement in requirements_by_id.values():
        for input_id in inputs_of(requirement):
            reads_of[input_id].append(requirement.id)
    between = {
        step_id
        for reader in readers
        for step_id in _upstream(reader, requirements_by_id)
        if step_id not in readers and dataset_id in _upstream(step_id, requirements_by_id)
    }
    shared = {
        step_id
        for step_id in between
        if step_id in answering
        or any(user not in between and user not in readers for user in reads_of[step_id])
    }
    if not shared:
        return [], set()
    copied = between & set().union(*(_upstream(step_id, requirements_by_id) for step_id in shared))
    taken = set(requirements_by_id)
    cloned = [
        (step_id, _copy_id(step_id, taken)) for step_id in requirements_by_id if step_id in copied
    ]
    return cloned, (between - copied) | set(readers)


def _copy_id(origin_id: str, taken: set[str]) -> str:
    """The first free id of `{stem}_history`, `{stem}_history_2` and on, which it then takes."""
    stem = origin_id[:52]
    copy_id = next(
        candidate
        for candidate in itertools.chain(
            (f"{stem}_history",), (f"{stem}_history_{n}" for n in itertools.count(2))
        )
        if candidate not in taken
    )
    taken.add(copy_id)
    return copy_id


def _with_copies(
    requirements: Sequence[EvidenceRequirement],
    cloned: Sequence[tuple[str, str]],
    *,
    rerouted: Collection[str],
) -> list[EvidenceRequirement]:
    """`requirements` with each step of `cloned` followed by its copy, and each step `rerouted` reading the copies."""
    renamed = dict(cloned)
    split: list[EvidenceRequirement] = []
    for requirement in requirements:
        if requirement.id in rerouted:
            split.append(_reading(requirement, renamed))
            continue
        split.append(requirement)
        if requirement.id in renamed:
            copy = _reading(requirement, renamed).model_copy(update={"id": renamed[requirement.id]})
            split.append(copy)
    return split


def _reading(requirement: EvidenceRequirement, renamed: Mapping[str, str]) -> EvidenceRequirement:
    """`requirement` reading, for each input `renamed` names, the requirement it maps to."""
    if isinstance(requirement, DatasetRequirement) or renamed.keys().isdisjoint(
        inputs_of(requirement)
    ):
        return requirement

    def side(read: RequirementSide) -> RequirementSide:
        return read.model_copy(update={"input_id": renamed.get(read.input_id, read.input_id)})

    if isinstance(requirement, ComparisonRequirement):
        update: dict[str, object] = {
            "minuend": side(requirement.minuend),
            "subtrahend": side(requirement.subtrahend),
        }
    elif isinstance(requirement, SumOfRequirement):
        update = {"left": side(requirement.left), "right": side(requirement.right)}
    elif isinstance(requirement, ShareOfRequirement | QuotientRequirement):
        update = {
            "numerator": side(requirement.numerator),
            "denominator": side(requirement.denominator),
        }
    elif isinstance(requirement, JoinRequirement):
        update = {
            "left_id": renamed.get(requirement.left_id, requirement.left_id),
            "right_id": renamed.get(requirement.right_id, requirement.right_id),
        }
    else:
        update = {"input_id": renamed.get(requirement.input_id, requirement.input_id)}
    return requirement.model_copy(update=update)


def _check_rolling_reads_open_periods(
    requirements_by_id: Mapping[str, EvidenceRequirement],
) -> None:
    """Reject a period that follows the newest data over a dataset retrieved with a closed end.

    A dataset retrieved over fixed years never holds a later one, so a
    relative period over it would stop moving at the plan's own end year.

    Raises:
        RollingPlanError: `RollingIssue.NEEDS_OPEN_PERIOD`, naming the dataset
            requirements to state `period_end_open` on.
    """
    fixed_end = {
        requirement.id
        for requirement in requirements_by_id.values()
        if isinstance(requirement, DatasetRequirement) and not requirement.period_end_open
    }
    closed: set[str] = set()
    readers: list[str] = []
    for requirement in requirements_by_id.values():
        if not rolls(requirement):
            continue
        unopened = dataset_roots(requirements_by_id, requirement.id) & fixed_end
        if unopened:
            closed |= unopened
            readers.append(requirement.id)
    if closed:
        raise RollingPlanError(
            RollingIssue.NEEDS_OPEN_PERIOD,
            f"{sorted(readers)!r} state a period that follows the newest data, over "
            f"{sorted(closed)!r}, retrieved with a fixed end year; state `period_end_open` "
            "on those dataset requirements, or drop `rolling` for fixed years",
            requirement_ids=sorted(closed),
            fault=OpenPeriodFault.DATASET_CLOSED,
        )


def _rolling_years(requirement: EvidenceRequirement) -> list[tuple[int, RelativePeriod]]:
    """Each year `requirement` states that follows the newest data, with its relative period."""
    if isinstance(requirement, GrowthRequirement | RankRequirement):
        rolling = requirement.rolling
        if rolling is None:
            return []
        bounds = (
            (requirement.window_start, rolling.window_start),
            (requirement.window_end, rolling.window_end),
        )
        return [
            (year, period) for year, period in bounds if year is not None and period is not None
        ]
    if isinstance(requirement, FilterRequirement):
        return [
            (predicate.year, predicate.rolling)
            for predicate in requirement.predicates
            if isinstance(predicate, PeriodThreshold) and predicate.rolling is not None
        ]
    return []


def _check_rolling_years_agree(requirements_by_id: Mapping[str, EvidenceRequirement]) -> None:
    """Reject rolling years that stand for different newest years where a run reads one, or for one the data lacks.

    The run resolves one newest year for each group of rolling requirements
    (`rolling_groups`, as `compiler.rolling.anchor_groups` groups their
    steps) and writes every rolling year of the group as that year moved by
    its offset (`compiler.rolling.resolved_arguments`). Every rule here
    reads the years a plan states, so all of a group's rolling years state
    one newest year (`RelativePeriod.latest_implied_by`): within one filter
    or window, and across the steps one result reads. That year is one the
    datasets the group rests on hold today: no later than the last year of
    any of their periods, the years Search found them over. A newest year
    before it stays, since translation restates a group's rolling years from
    the newest year at least half its series hold where that is earlier than
    the last year Search states (`restated_to_anchors`).

    Raises:
        RollingPlanError: `RollingIssue.YEARS_DISAGREE`, naming the group's
            requirements.
    """
    reads, rolling = _rolling_reads(requirements_by_id)
    for members in rolling_groups(reads, rolling=rolling):
        implied = {
            member: sorted(
                {
                    period.latest_implied_by(year)
                    for year, period in _rolling_years(requirements_by_id[member])
                }
            )
            for member in members
        }
        newest = {year for years in implied.values() for year in years}
        if len(newest) > 1:
            stated = "; ".join(f"{member!r} {years!r}" for member, years in implied.items())
            raise RollingPlanError(
                RollingIssue.YEARS_DISAGREE,
                f"{members!r} count from one newest year, and their rolling years, less their "
                f"offsets, give different ones ({stated}); state each rolling year as the "
                "newest year the data holds today moved by its offset",
                requirement_ids=members,
            )
        held = min(
            root.period.end
            for member in members
            for root_id in dataset_roots(requirements_by_id, member)
            if isinstance(root := requirements_by_id[root_id], DatasetRequirement)
        )
        if any(year > held for year in newest):
            raise RollingPlanError(
                RollingIssue.YEARS_DISAGREE,
                f"{members!r} state rolling years, less their offsets, as newest year {min(newest)}, "
                f"after {held}, the last year the datasets they read hold; state each rolling "
                "year as the newest year the data holds today moved by its offset",
                requirement_ids=members,
            )


def _check_exclusions_fixed(requirements_by_id: Mapping[str, EvidenceRequirement]) -> None:
    """Reject a filter's `ne` that follows the newest data, or that reads years newer data moves.

    A rolling `ne` removes another year each time newer data arrives. Over
    a dataset retrieved with `period_end_open`, a year a fixed `ne` removes
    can become the newest year, or the year a later bound that follows the
    newest data names, which then holds no row. So an `ne` names a fixed
    year over datasets of fixed years alone.

    Raises:
        RollingPlanError: `RollingIssue.EXCLUSION_ROLLS`, naming the filters.
    """
    open_ends = {
        requirement.id
        for requirement in requirements_by_id.values()
        if isinstance(requirement, DatasetRequirement) and requirement.period_end_open
    }
    refused = sorted(
        requirement.id
        for requirement in requirements_by_id.values()
        if isinstance(requirement, FilterRequirement)
        and any(
            isinstance(predicate, PeriodThreshold)
            and predicate.op == "ne"
            and (
                predicate.rolling is not None
                or not open_ends.isdisjoint(dataset_roots(requirements_by_id, requirement.id))
            )
            for predicate in requirement.predicates
        )
    )
    if refused:
        raise RollingPlanError(
            RollingIssue.EXCLUSION_ROLLS,
            f"{refused!r} remove a year with `ne` that follows the newest data, or from a "
            "dataset retrieved with `period_end_open`, so newer data can make the removed year "
            "the one a step reads; state `ne` on a fixed "
            "year over datasets retrieved without `period_end_open`, and to leave out the "
            "newest year keep the years up to the one before it with op `lte` and `rolling` "
            "at offset -1",
            requirement_ids=refused,
        )


def frame_shapes(plan: Plan) -> dict[str, FrameShape]:
    """Return each requirement's output shape, by id, for a plan that already validated.

    The shapes are those of the plan as it runs (`split_of`), so they hold
    a key for each copy the split adds beside the plan's own requirements.
    """
    return _split_shapes(split_of(plan))


def _split_shapes(split: HistorySplit) -> dict[str, FrameShape]:
    """The shapes of `split`'s requirements, each whole-history read starting at its first year."""
    return _build_shapes(
        {requirement.id: requirement for requirement in split.requirements},
        whole_history=split.whole_history,
        origins=split.origins(),
    )


def _build_shapes(
    requirements_by_id: dict[str, EvidenceRequirement],
    *,
    whole_history: Mapping[str, int],
    origins: Mapping[str, str],
) -> dict[str, FrameShape]:
    """Walk every requirement in the order the plan states them, each after its inputs
    (`requirement_step_order`), rejecting an illegal shape as it is found.

    Each requirement's output shape is computed from its inputs' -- already
    built, since the walk is topological -- through the one transform
    `planner.shapes` states for its kind, after `planner.shapes.rejects`
    confirms the input may feed it. Nothing here skips a check for want of a
    single upstream dataset requirement to walk to: a role dimension always
    names exactly one, by construction of every transform. `whole_history`
    names each dataset requirement read from the first year its dataset
    holds on record, with that year, where its shape starts
    (`_dataset_shape`). Every other one starts at its stated period.
    `origins` maps each copy `whole_history_split` added to the requirement
    it copies, which a refusal names in its place: a copy reads other
    years than its origin, and the model wrote only the origin. A copy
    that is not a dataset requirement passes its input's years on
    (`passes_observation_years`), so it writes no span named for itself,
    and its shape is built under its origin's id.

    Raises:
        RollingPlanError: `RollingIssue.SIDE_UNANCHORED`, naming the first
            two-sided step pairing an open side with a side on a newest year
            not resolved over it (`planner.shapes.anchor_misses`), or
            `RollingIssue.YEARS_RUN_OUT`, naming the first step whose years
            newer data leaves empty (`planner.shapes.years_run_out`), or
            the first growth whose window year newer data takes out of its
            input (`planner.shapes.growth_endpoint_leaves`).
    """
    reads, rolling = _rolling_reads(requirements_by_id)
    over = anchored_over(
        reads,
        rolling=rolling,
        retrievals=[
            requirement_id
            for requirement_id, requirement in requirements_by_id.items()
            if isinstance(requirement, DatasetRequirement)
        ],
    )
    shapes_by_id: dict[str, FrameShape] = {}
    for requirement_id in requirement_step_order(requirements_by_id):
        requirement = requirements_by_id[requirement_id]
        named = origins.get(requirement_id, requirement_id)
        if named != requirement_id and not isinstance(requirement, DatasetRequirement):
            requirement = requirement.model_copy(update={"id": named})
        shape = _shape_of(
            requirement,
            shapes_by_id,
            requirements_by_id,
            first_year=whole_history.get(requirement_id),
        )
        sides = [shapes_by_id[input_id] for input_id in reads[requirement_id]]
        missed = anchor_misses(sides, over[requirement_id]) if len(sides) > 1 else None
        if missed is not None:
            raise RollingPlanError(
                RollingIssue.SIDE_UNANCHORED,
                f"{named!r}: {missed}",
                requirement_ids=[named],
            )
        run_out = years_run_out(shape)
        if run_out is not None:
            raise RollingPlanError(
                RollingIssue.YEARS_RUN_OUT,
                f"{named!r}: {run_out}",
                requirement_ids=[named],
            )
        shapes_by_id[requirement_id] = shape
    return shapes_by_id


def requirement_step_order(requirements_by_id: Mapping[str, EvidenceRequirement]) -> list[str]:
    """Every requirement id in the order `requirements_by_id` holds them, each after
    every id it reads, one per requirement.

    A post-order depth-first walk taking the requirements in the order the
    mapping holds them, the order the plan states them, and each one's inputs
    (`inputs_of`) in the order it reads them. So `_build_shapes` meets the
    steps the model wrote in the order it wrote them, each after its inputs.
    An input missing from `requirements_by_id` is skipped. Assumes the
    requirements form a DAG, which `_check_reads_resolve` guarantees by calling
    `_find_cycle` before shapes are built.
    """
    order: list[str] = []
    visited: set[str] = set()

    def visit(requirement_id: str) -> None:
        if requirement_id in visited or requirement_id not in requirements_by_id:
            return
        visited.add(requirement_id)
        for input_id in inputs_of(requirements_by_id[requirement_id]):
            visit(input_id)
        order.append(requirement_id)

    for requirement_id in requirements_by_id:
        visit(requirement_id)
    return order


def _dataset_shape(requirement: DatasetRequirement, *, first_year: int | None) -> FrameShape:
    """The shape a fresh retrieval carries: one role dimension per role phrase the entities
    state, then one open dimension per `across` word, then the breakdown's.

    Its years run on past `period` with `period_end_open`, and start at
    `first_year` for a read of its dataset's whole history, the first year
    the dataset holds on record (`_build_shapes`).

    A requirement reading its source's StatVars separately carries an
    uncounted number of measures (`None`): the plan has no source yet to
    count them from, and a source binding one StatVar gives it one, so no
    single-measure verb refuses it here. The compiler counts them once the
    source is known and refuses a family of several there
    (`compiler.shapes`). Summed or unset, it carries one, and so does a
    breakdown, which sums the family within each code.
    """
    role_names = sorted(
        {entity.role for entity in requirement.entities}, key=lambda role: (role is None, role)
    )
    roles = tuple(
        RoleDimension(role=name, requirement_id=requirement.id, place=place_role(name))
        for name in role_names
    )
    roles = (
        *roles,
        *(
            RoleDimension(
                role=word,
                requirement_id=requirement.id,
                open=True,
                place=place_role(word),
                level_filterable=True,
            )
            for word in requirement.across
        ),
    )
    if requirement.breakdown is not None:
        breakdown = RoleDimension(
            role=requirement.breakdown.dimension.value,
            requirement_id=requirement.id,
            breakdown=True,
        )
        roles = (*roles, breakdown)
    separate = requirement.stat_vars is StatVarReading.SEPARATE and requirement.breakdown is None
    measures = None if separate else 1
    return retrieved(
        requirement_id=requirement.id,
        roles=roles,
        period=retrieved_period(requirement, first_year=first_year),
        measures=measures,
        end_open=requirement.period_end_open,
    )


def repinned(
    requirement: DatasetRequirement, spec_id: str, exact_codes: frozenset[str]
) -> DatasetRequirement:
    """`requirement` pinned to `spec_id`, a dataset code chose in place of the one it named.

    The source it stated goes with the pin it left: `source`, `source_ref`
    and `source_exclude`, and `source_code` unless `exact_codes`, the codes
    `spec_id` is exactly (`exact_codes`), hold it. Discovery checks a pin
    against a stated source and searches again by it when they differ, so a
    pin a scope switch (`harness.graph.switch`) or `propose_plan`'s alignment
    (`submission.service._aligned_bases`) chose would otherwise be searched
    back to the dataset it replaced.
    """
    update: dict[str, Any] = {
        "source_spec_id": spec_id,
        "source": None,
        "source_ref": None,
        "source_exclude": False,
    }
    if requirement.source_code not in exact_codes:
        update["source_code"] = None
    return requirement.model_copy(update=update)


def exact_codes(block: Mapping[str, Any] | None) -> frozenset[str]:
    """The publisher codes the dataset of the spec block `block` is exactly, none when it
    cannot be read (`compiler.sources.source_from_spec_block`)."""
    if block is None:
        return frozenset()
    try:
        return frozenset(source_from_spec_block(block).exact_codes)
    except ValueError:
        return frozenset()


def retrieved_period(requirement: DatasetRequirement, *, first_year: int | None) -> Period:
    """The years a retrieval of `requirement` reads.

    Its stated period, started at `first_year` for a read of its dataset's
    whole history (`HistorySplit.whole_history`). The plan's shapes and the
    compiler's both start a retrieval here, so both check a growth window
    against the same years (`planner.shapes.window_outside`).
    """
    if first_year is None:
        return requirement.period
    return Period(start=first_year, end=requirement.period.end)


def _aggregate_shape(requirement: AggregateRequirement, input_shape: FrameShape) -> FrameShape:
    by_entity = GroupBy.ENTITY in requirement.group_by
    by_period = GroupBy.PERIOD in requirement.group_by
    by_breakdown = GroupBy.BREAKDOWN in requirement.group_by
    reason = shape_rejects(
        ShapeVerb.AGGREGATE,
        input_shape,
        by_entity=by_entity,
        by_period=by_period,
        by_breakdown=by_breakdown,
        reduction=requirement.reduction,
    )
    if reason is not None:
        raise ValueError(f"{requirement.id!r}: {reason}")
    return aggregated(
        input_shape,
        reduction=requirement.reduction,
        by_entity=by_entity,
        by_period=by_period,
        by_breakdown=by_breakdown,
        step_id=requirement.id,
    )


def _growth_shape(requirement: GrowthRequirement, input_shape: FrameShape) -> FrameShape:
    window = Period(start=requirement.window_start, end=requirement.window_end)
    reason = shape_rejects(ShapeVerb.GROWTH, input_shape, change=requirement.change, window=window)
    if reason is not None:
        raise ValueError(f"{requirement.id!r}: {reason}")
    outside = window_outside(input_shape, window)
    if outside is not None:
        raise ValueError(f"{requirement.id!r}'s {outside}")
    leaves = growth_endpoint_leaves(input_shape, window, requirement.rolling)
    if leaves is not None:
        raise RollingPlanError(
            RollingIssue.YEARS_RUN_OUT,
            f"{requirement.id!r}: {leaves}",
            requirement_ids=[requirement.id],
        )
    return grown(
        input_shape,
        change=requirement.change,
        window=window,
        rolling=requirement.rolling,
        step_id=requirement.id,
    )


def _selected_role(roles: Sequence[RoleDimension], axis: str | None) -> RoleDimension:
    """Return the role dimension `axis` selects, once `shapes.rejects` has confirmed it may.

    `shapes.rejects` is what proves `axis` (or the lone role, when `axis` is
    `None`) resolves to exactly one dimension of `roles`; this only repeats
    that resolution (`shapes.role_named`), to read a field off the one it
    finds.
    """
    selected = roles[0] if axis is None else role_named(roles, axis)
    assert selected is not None
    return selected


def _rank_shape(requirement: RankRequirement, input_shape: FrameShape) -> FrameShape:
    window = (
        Period(start=requirement.window_start, end=requirement.window_end)
        if requirement.window_start is not None and requirement.window_end is not None
        else None
    )
    reason = shape_rejects(
        ShapeVerb.RANK,
        input_shape,
        axis=requirement.axis,
        among=requirement.among,
        window=window,
        per_period=requirement.per_period,
        reduction=requirement.reduction,
        rolling=requirement.rolling,
    )
    if reason is not None:
        raise ValueError(f"{requirement.id!r}: {reason}")
    axis_role = _selected_role(input_shape.roles, requirement.axis)
    if requirement.among is RankAmong.NAMED and not axis_role.narrowed:
        raise ValueError(
            f"{requirement.id!r} states among='named', but axis {axis_role.role!r} carries no "
            "narrowing filter; a named rank needs a visible FilterRequirement with an EntityIn "
            "on that role, or state among='population' instead"
        )
    reduction_reason = _rank_reduction_reason(requirement, input_shape)
    if reduction_reason is not None:
        raise ValueError(reduction_reason)
    return ranked(
        input_shape,
        axis=requirement.axis,
        window=window,
        per_period=requirement.per_period,
        limit=requirement.limit,
        rolling=requirement.rolling,
        step_id=requirement.id,
    )


def _rank_reduction_reason(requirement: RankRequirement, input_shape: FrameShape) -> str | None:
    """Return why `requirement` needs a reduction it does not state, or `None` when it has enough.

    A rank orders entities by one value each. An axis over an input carrying
    more than one role always needs a reduction, in either mode, to collapse
    the other role's rows to one value per axis entity per (period, in
    per-period mode). Short of that, `per_period=True` ranks each period on
    its own and never needs a reduction just to collapse years together;
    `per_period=False` needs one unless its input holds one row per entity,
    or the years of its input its window keeps (`_years_read`) are one and
    newer data adds none: a 2020-2025 window over 2015-2020 ranks each
    entity's 2020 row. 2024 alone today is 2024 and 2025 once 2025 arrives
    when the window's end alone follows the newest data, or when nothing
    fixes the last year of a dataset retrieved with no end year. A window
    stays one year with both ends fixed on it, or both following the newest
    data at one offset.
    """
    if requirement.reduction is not None:
        return None
    if len(input_shape.roles) > 1:
        return (
            f"{requirement.id!r} ranks by axis {requirement.axis!r} over an input carrying more "
            "than one role, which needs a reduction to collapse the other role's rows to one "
            "value per axis entity"
        )
    if requirement.per_period or input_shape.one_row_per_entity:
        return None
    held = _years_read(requirement, input_shape)
    if held is None or not held.several:
        return None
    if not held.follows_newest:
        return (
            f"{requirement.id!r} ranks entities over {held.phrase()} with no reduction, so no "
            "entity has a single value to rank on; to rank one year set window_start and "
            "window_end to that year, or, only for an ask marked (across years), state a "
            "reduction to combine the years"
        )
    return (
        f"{requirement.id!r} ranks entities over {held.phrase()} with no reduction, so no "
        "entity keeps a single value to rank on; to rank one year state window_start and "
        "window_end on that year, with `rolling` unset for a fixed year or naming both at "
        "one offset for a year that follows the newest data, or, only for an ask marked "
        "(across years), state a reduction to combine the years"
    )


def stated_under(entity: Entity, role: str | None) -> bool:
    """Whether the phrase `role` names the dimension `entity` sits on; any does when `role` is `None`.

    The rule is `planner.shapes.roles_named`'s: a phrase names the role it
    equals, and a place word names the place dimension, whose entities
    state no role or a place phrase.
    """
    if role is None or entity.role == role:
        return True
    return across_party(role) == "place" and place_role(entity.role)


def entities_named(requirement: DatasetRequirement, name: str, role: str | None) -> list[Entity]:
    """The entities of `requirement` called `name`, under `role` when one is given (`stated_under`)."""
    return [
        entity
        for entity in requirement.entities
        if entity.name == name and stated_under(entity, role)
    ]


def unknown_names(
    names: Iterable[str], role: str | None, requirements: Sequence[DatasetRequirement]
) -> list[str]:
    """Each of `names` that none of `requirements` states, under `role` when one is given.

    A filter keeps entities by the names the requirements behind its input
    state. A name they do not state matches no row the compiler can write,
    since a breakdown's codes are no entities and analysis-mcp matches the
    code labels apis decodes, which no plan knows. `_filter_shape` refuses such a
    filter and `compiler.compile` reports it as `unknown_entity`.
    """
    return [
        name
        for name in names
        if not any(entities_named(requirement, name, role) for requirement in requirements)
    ]


def stated_names(requirements: Sequence[DatasetRequirement], role: str | None) -> list[str]:
    """Every entity name `requirements` state, under `role` when one is given, sorted.

    What a refusal of `unknown_names` lists as the names a filter may keep.
    """
    return sorted(
        {
            entity.name
            for requirement in requirements
            for entity in requirement.entities
            if stated_under(entity, role)
        }
    )


def _requirements_behind(
    shape: FrameShape, requirements_by_id: Mapping[str, EvidenceRequirement]
) -> list[DatasetRequirement]:
    """The dataset requirements whose entities `shape`'s role dimensions hold, each once.

    A frame with no role dimension resting on one retrieval answers with
    that retrieval's requirement; one resting on several holds one series
    and no entity to name.
    """
    ids = list(dict.fromkeys(role.requirement_id for role in shape.roles))
    if not ids and len(shape.roots) == 1:
        ids = list(shape.roots)
    return [
        requirement
        for requirement_id in ids
        if isinstance(requirement := requirements_by_id[requirement_id], DatasetRequirement)
    ]


def _check_filter_names_known(
    requirement: FilterRequirement,
    input_shape: FrameShape,
    requirements_by_id: Mapping[str, EvidenceRequirement],
) -> None:
    """Reject an `EntityIn` naming what no requirement behind the filter's input states (`unknown_names`)."""
    behind = _requirements_behind(input_shape, requirements_by_id)
    for predicate in requirement.predicates:
        if not isinstance(predicate, EntityIn):
            continue
        unknown = unknown_names(predicate.names, predicate.role, behind)
        if not unknown:
            continue
        under = f" under role {predicate.role!r}" if predicate.role is not None else ""
        stated = stated_names(behind, predicate.role)
        codes = predicate.role is not None and any(
            role.breakdown for role in roles_named(input_shape.roles, predicate.role)
        )
        remedy = (
            f"; {predicate.role!r} is a breakdown, whose codes are no entities a filter keeps: "
            "to read one code, state a dataset requirement whose measure names it, or keep "
            "every code"
            if codes
            else ""
        )
        raise ValueError(
            f"{requirement.id!r} keeps {unknown!r}{under}, which no requirement behind "
            f"{requirement.input_id!r} names; the entities they name{under} are "
            f"{stated!r}{remedy}"
        )


def _filter_shape(
    requirement: FilterRequirement,
    input_shape: FrameShape,
    requirements_by_id: Mapping[str, EvidenceRequirement],
) -> FrameShape:
    _check_filter_names_known(requirement, input_shape, requirements_by_id)
    years = year_filter(input_shape.period, requirement.predicates, drift=input_shape.drift)
    if isinstance(years, str):
        raise ValueError(f"{requirement.id!r}: {years}")
    levels = level_filter(input_shape.roles, requirement.predicates)
    if isinstance(levels, str):
        raise ValueError(f"{requirement.id!r}: {levels}")
    reason = shape_rejects(
        ShapeVerb.FILTER,
        input_shape,
        place_level=keeps_one_level(requirement.predicates),
        place_level_available=levels.available,
        period_threshold=states_period_threshold(requirement.predicates),
        years=years,
    )
    if reason is not None:
        raise ValueError(f"{requirement.id!r}: {reason}")
    narrowed_roles = tuple(p.role for p in requirement.predicates if isinstance(p, EntityIn))
    return filtered(
        input_shape,
        narrowed_roles=narrowed_roles,
        years=years,
        levelled_role_keys=levels.role_keys,
    )


def keeps_one_level(predicates: Iterable[Predicate]) -> bool:
    """Whether `predicates` state a `PlaceLevel`, which keeps one level of place."""
    return any(isinstance(predicate, PlaceLevel) for predicate in predicates)


def states_period_threshold(predicates: Iterable[Predicate]) -> bool:
    """Whether `predicates` state a `PeriodThreshold`, which compares each row's year."""
    return any(isinstance(predicate, PeriodThreshold) for predicate in predicates)


def _check_side_entity_is_named(
    requirement_id: str,
    field: str,
    side: RequirementSide,
    side_shape: FrameShape,
    requirements_by_id: dict[str, EvidenceRequirement],
    *,
    keyed: str | None = None,
    one_figure: bool = False,
    keyable: bool = False,
) -> None:
    """Reject `side` narrowed to an entity none of its shape's role dimensions' requirements name,
    or read whole over a dimension nobody named.

    A side naming no entity is one series on every dimension but the one
    `keyed` names, so an open dimension left on it is refused
    (`planner.shapes.open_on_whole_side`), as the compiler refuses it.
    `one_figure` is whether the reader takes one figure from each side, and
    `keyable` whether `side` is an un-keyed reader's numerator, which `axis`
    would key (`planner.shapes.open_whole_side_reason`).
    """
    if side.entity is None:
        role = open_on_whole_side(side_shape.roles, keyed=keyed)
        if role is not None:
            raise ValueError(
                f"{requirement_id!r} reads its {field} ({side.input_id!r}) whole, "
                f"{open_whole_side_reason(role, one_figure=one_figure, keyable=keyable)}"
            )
        return
    names: set[str] = set()
    for role in side_shape.roles:
        dataset = requirements_by_id[role.requirement_id]
        assert isinstance(dataset, DatasetRequirement)
        names.update(entity.name for entity in dataset.entities)
    if not names:
        raise ValueError(
            f"{requirement_id!r} narrows its {field} ({side.input_id!r}) to entity "
            f"{side.entity!r}, but that input carries no entity: it is a total, so the side "
            "names none"
        )
    if side.entity not in names:
        raise ValueError(
            f"{requirement_id!r} narrows its {field} ({side.input_id!r}) to entity "
            f"{side.entity!r}, which is not one of {sorted(names)!r}"
        )


def _pools_one_member(
    requirement: ShareOfRequirement,
    shape: FrameShape,
    requirements_by_id: dict[str, EvidenceRequirement],
) -> str | None:
    """Why a share pooling its own axis divides one member by itself, or `None` when it does not.

    Its key holds every member of an open dimension, or the entities its
    requirement names on that dimension less any the side narrows to. One
    member is the whole of its own total, 100% every period.
    """
    assert requirement.axis is not None
    role = role_named(shape.roles, requirement.axis)
    if role is None or role.open:
        return None
    dataset = requirements_by_id[role.requirement_id]
    assert isinstance(dataset, DatasetRequirement)
    members = [entity.name for entity in dataset.entities if entity.role == role.role]
    if requirement.numerator.entity in members:
        held = f"narrows to {requirement.numerator.entity!r} on {requirement.axis!r}"
    elif len(members) < 2:
        held = f"names {members!r} alone on {requirement.axis!r}"
    else:
        return None
    return (
        f"{requirement.id!r} reads one side as both numerator and denominator, and that side "
        f"{held}, so the share divides the member by itself; for a named member's share, "
        "divide that member's series by every member's total per period"
    )


def _share_of_shape(
    requirement: ShareOfRequirement,
    shapes_by_id: dict[str, FrameShape],
    requirements_by_id: dict[str, EvidenceRequirement],
) -> FrameShape:
    numerator_shape = shapes_by_id[requirement.numerator.input_id]
    denominator_shape = shapes_by_id[requirement.denominator.input_id]
    one_figure = requirement.axis is None
    # A per-axis denominator reads one series per member, and one pooling its
    # own axis sums exactly the members its numerator keeps apart.
    keys_denominator = (
        requirement.denominator_scope is DenominatorScope.PER_AXIS or requirement.pools_its_own_axis
    )
    _check_side_entity_is_named(
        requirement.id,
        "numerator",
        requirement.numerator,
        numerator_shape,
        requirements_by_id,
        keyed=requirement.axis,
        one_figure=one_figure,
        keyable=one_figure,
    )
    _check_side_entity_is_named(
        requirement.id,
        "denominator",
        requirement.denominator,
        denominator_shape,
        requirements_by_id,
        keyed=requirement.axis if keys_denominator else None,
        one_figure=one_figure,
    )
    if requirement.pools_its_own_axis:
        reason = _pools_one_member(requirement, numerator_shape, requirements_by_id)
        if reason is not None:
            raise ValueError(reason)
    # The denominator's own *in* rule (measures, time, not ordinal) is
    # checked with no axis: the numerator's side alone gives the output its
    # role lineage, so only the numerator's call states the real axis, with
    # the denominator passed as `other` for the per-axis cross-check.
    reason = shape_rejects(
        ShapeVerb.SHARE_OF,
        denominator_shape,
        axis=None,
        entity=requirement.denominator.entity,
    )
    if reason is not None:
        raise ValueError(f"{requirement.id!r}: {reason}")
    reason = shape_rejects(
        ShapeVerb.SHARE_OF,
        numerator_shape,
        axis=requirement.axis,
        scope=requirement.denominator_scope,
        other=denominator_shape,
        entity=requirement.numerator.entity,
    )
    if reason is not None:
        raise ValueError(f"{requirement.id!r}: {reason}")
    return shared(
        numerator_shape,
        denominator_shape,
        axis=requirement.axis,
        scope=requirement.denominator_scope,
    )


def _quotient_of_shape(
    requirement: QuotientRequirement,
    shapes_by_id: dict[str, FrameShape],
    requirements_by_id: dict[str, EvidenceRequirement],
) -> FrameShape:
    numerator_shape = shapes_by_id[requirement.numerator.input_id]
    denominator_shape = shapes_by_id[requirement.denominator.input_id]
    one_figure = requirement.axis is None
    _check_side_entity_is_named(
        requirement.id,
        "numerator",
        requirement.numerator,
        numerator_shape,
        requirements_by_id,
        keyed=requirement.axis,
        one_figure=one_figure,
        keyable=one_figure,
    )
    _check_side_entity_is_named(
        requirement.id,
        "denominator",
        requirement.denominator,
        denominator_shape,
        requirements_by_id,
        keyed=requirement.denominator_axis,
        one_figure=one_figure,
    )
    # As for a share_of, the denominator's own *in* rule is checked with no
    # axis, and the numerator's call states both, with the denominator passed
    # as `other` for the periods and for `denominator_axis`.
    reason = shape_rejects(ShapeVerb.QUOTIENT_OF, denominator_shape)
    if reason is not None:
        raise ValueError(f"{requirement.id!r}: {reason}")
    reason = shape_rejects(
        ShapeVerb.QUOTIENT_OF,
        numerator_shape,
        axis=requirement.axis,
        denominator_axis=requirement.denominator_axis,
        other=denominator_shape,
    )
    if reason is not None:
        raise ValueError(f"{requirement.id!r}: {reason}")
    return quotiented(
        numerator_shape,
        denominator_shape,
        axis=requirement.axis,
        denominator_axis=requirement.denominator_axis,
    )


def _comparison_shape(
    requirement: ComparisonRequirement,
    shapes_by_id: dict[str, FrameShape],
    requirements_by_id: dict[str, EvidenceRequirement],
) -> FrameShape:
    minuend_shape = shapes_by_id[requirement.minuend.input_id]
    subtrahend_shape = shapes_by_id[requirement.subtrahend.input_id]
    _check_side_entity_is_named(
        requirement.id, "minuend", requirement.minuend, minuend_shape, requirements_by_id
    )
    _check_side_entity_is_named(
        requirement.id, "subtrahend", requirement.subtrahend, subtrahend_shape, requirements_by_id
    )
    for shape, other in ((minuend_shape, subtrahend_shape), (subtrahend_shape, minuend_shape)):
        reason = shape_rejects(ShapeVerb.COMPARE, shape, other=other)
        if reason is not None:
            raise ValueError(f"{requirement.id!r}: {reason}")
    return compared(minuend_shape, subtrahend_shape)


def _sum_of_shape(
    requirement: SumOfRequirement,
    shapes_by_id: dict[str, FrameShape],
    requirements_by_id: dict[str, EvidenceRequirement],
) -> FrameShape:
    left_shape = shapes_by_id[requirement.left.input_id]
    right_shape = shapes_by_id[requirement.right.input_id]
    _check_side_entity_is_named(
        requirement.id, "left", requirement.left, left_shape, requirements_by_id
    )
    _check_side_entity_is_named(
        requirement.id, "right", requirement.right, right_shape, requirements_by_id
    )
    for shape, other in ((left_shape, right_shape), (right_shape, left_shape)):
        reason = shape_rejects(ShapeVerb.SUM_OF, shape, other=other)
        if reason is not None:
            raise ValueError(f"{requirement.id!r}: {reason}")
    return summed_sides(left_shape, right_shape)


def _join_shape(requirement: JoinRequirement, shapes_by_id: dict[str, FrameShape]) -> FrameShape:
    left_shape = shapes_by_id[requirement.left_id]
    right_shape = shapes_by_id[requirement.right_id]
    for shape, other in ((left_shape, right_shape), (right_shape, left_shape)):
        reason = shape_rejects(ShapeVerb.JOIN, shape, other=other)
        if reason is not None:
            raise ValueError(f"{requirement.id!r}: {reason}")
    return joined(left_shape, right_shape)


def _shape_of(  # noqa: PLR0911 -- one isinstance branch per requirement kind narrows `ty`'s
    # type for each `_xxx_shape` call; a dict keyed by `type(requirement)` cannot narrow it
    # and every branch would need an `assert isinstance(...)` restoring the same length.
    requirement: EvidenceRequirement,
    shapes_by_id: dict[str, FrameShape],
    requirements_by_id: dict[str, EvidenceRequirement],
    *,
    first_year: int | None,
) -> FrameShape:
    """Return the output shape `requirement` produces, raising on the first illegal input.

    `first_year` starts a dataset requirement's shape (`_dataset_shape`).
    """
    if isinstance(requirement, DatasetRequirement):
        return _dataset_shape(requirement, first_year=first_year)
    if isinstance(requirement, AggregateRequirement):
        return _aggregate_shape(requirement, shapes_by_id[requirement.input_id])
    if isinstance(requirement, GrowthRequirement):
        return _growth_shape(requirement, shapes_by_id[requirement.input_id])
    if isinstance(requirement, RankRequirement):
        return _rank_shape(requirement, shapes_by_id[requirement.input_id])
    if isinstance(requirement, FilterRequirement):
        return _filter_shape(requirement, shapes_by_id[requirement.input_id], requirements_by_id)
    if isinstance(requirement, ShareOfRequirement):
        return _share_of_shape(requirement, shapes_by_id, requirements_by_id)
    if isinstance(requirement, QuotientRequirement):
        return _quotient_of_shape(requirement, shapes_by_id, requirements_by_id)
    if isinstance(requirement, ComparisonRequirement):
        return _comparison_shape(requirement, shapes_by_id, requirements_by_id)
    if isinstance(requirement, SumOfRequirement):
        return _sum_of_shape(requirement, shapes_by_id, requirements_by_id)
    assert isinstance(requirement, JoinRequirement)
    return _join_shape(requirement, shapes_by_id)


def _check_population_claims_are_cited(
    requirements: Iterable[EvidenceRequirement], decisions_by_slot: dict[str, Decision]
) -> None:
    """Reject a population-wide claim no cited decision states structurally.

    A rank stating `among='population'` and a keyed share_of stating
    `denominator_scope='pooled'` are each a figure computed over entities
    nobody named, so either needs a cited `Decision` whose own `fixes` states
    the same claim. Every slot in `decision_ids` is already known to name a
    decision the plan makes.
    """
    for requirement in requirements:
        if isinstance(requirement, RankRequirement) and requirement.among is RankAmong.POPULATION:
            field, value = "among", RankAmong.POPULATION
        elif (
            isinstance(requirement, ShareOfRequirement)
            and requirement.denominator_scope is DenominatorScope.POOLED
        ):
            field, value = "denominator_scope", DenominatorScope.POOLED
        else:
            continue
        cited = (decisions_by_slot[slot] for slot in requirement.decision_ids)
        if not any(
            decision.fixes is not None and getattr(decision.fixes, field) == value
            for decision in cited
        ):
            raise ValueError(
                f"{requirement.id!r} states {field}={value!r}, a claim over entities nobody "
                f"named, but cites no decision whose `fixes` names `{field}`"
            )


# The fields whose empty value is a value stated: an empty `group_by` is one
# figure over every row. Every other empty tuple states nothing, as `entities`
# and `across` leave it.
_EMPTY_STATED = frozenset({"group_by"})


def _stated(requirement: EvidenceRequirement, field: str) -> Any:
    """The value `requirement` states for `field`, a field its kind declares, or `None`."""
    value = getattr(requirement, field)
    return None if value == () and field not in _EMPTY_STATED else value


def _same(first: Any, second: Any) -> bool:
    """Whether two stated values agree, tuples (`entities`, `group_by`, `predicates`) as sets."""
    if isinstance(first, tuple) and isinstance(second, tuple):
        return set(first) == set(second)
    return first == second


# The `DecisionFixes` field no requirement is held to. Code writes
# `source_spec_id` on the decisions a pin states and moves it with the pin
# (`submission.service._measure_decisions`, `_scope_decisions`, `_repinned`);
# a draft cannot name it.
_UNCHECKED_FIXES = frozenset({"source_spec_id"})


class FixDisagreement(NamedTuple):
    """One value a cited decision fixes that the requirement citing it states otherwise."""

    field: str
    fixed: Any
    stated: Any


def bound_fixes(requirement: EvidenceRequirement, decision: Decision) -> dict[str, Any]:
    """The values `decision.fixes` states that `requirement`, citing it, is held to, by field.

    A field the decision leaves unset is left out, and so is one the
    requirement's kind does not declare and `source_spec_id`
    (`_UNCHECKED_FIXES`).
    """
    if decision.fixes is None:
        return {}
    declared = type(requirement).model_fields
    return {
        field: fixed
        for field, fixed in decision.fixes
        if fixed is not None and field in declared and field not in _UNCHECKED_FIXES
    }


def fix_disagreements(
    requirement: EvidenceRequirement, decision: Decision
) -> list[FixDisagreement]:
    """Each value of `bound_fixes` that `requirement` states otherwise, tuples compared as sets (`_same`)."""
    return [
        FixDisagreement(field, fixed, getattr(requirement, field))
        for field, fixed in bound_fixes(requirement, decision).items()
        if not _same(fixed, _stated(requirement, field))
    ]


def fixes_changes(
    before: DecisionFixes | None, after: DecisionFixes | None
) -> list[FixDisagreement]:
    """Each field whose value `after` states differently from `before`, tuples compared as sets.

    `fixed` is the value `before` states and `stated` the one `after` does,
    either `None` where it states none.
    """
    empty = DecisionFixes()
    first, second = before or empty, after or empty
    return [
        FixDisagreement(field, getattr(first, field), getattr(second, field))
        for field in DecisionFixes.model_fields
        if not _same(getattr(first, field), getattr(second, field))
    ]


def _check_requirements_honour_their_decisions(
    requirements: Iterable[EvidenceRequirement], decisions_by_slot: Mapping[str, Decision]
) -> None:
    """Reject a requirement stating a value a decision it cites fixes otherwise.

    `propose_plan` upholds this by construction: `settle_decisions` reads a
    drafted decision's values off its citing requirements, and
    `submission.service.admit` drops or refuses a carried decision an
    amendment contradicts. Every slot in `decision_ids` is already known to
    name a decision the plan makes.
    """
    for requirement in requirements:
        for slot in requirement.decision_ids:
            disagreements = fix_disagreements(requirement, decisions_by_slot[slot])
            if disagreements:
                field, fixed, stated = disagreements[0]
                raise ValueError(
                    f"{requirement.id!r} states {field} {stated!r}, but the decision "
                    f"{slot!r} it cites fixes {fixed!r}"
                )


def settle_decisions(
    drafts: Iterable[DecisionDraft], requirements: Sequence[EvidenceRequirement]
) -> list[Decision]:
    """Build each `Decision` from its draft, reading the fields it fixes off the requirements citing it.

    A named field takes the value the citing requirements that declare it
    state, when they all state the same one. A field none of them states,
    or that they state differently, is left out. `fixes` is read off the
    requirements, so the model never restates a value, nothing is refused
    for it, and every built `Decision` agrees with the requirements citing
    it (`fix_disagreements`). A draft naming no field that settles builds a
    `Decision` with no `fixes`.

    A field the citing requirements disagree on is dropped because its
    readers (`harness.turn_records._periods`, `harness.graph.answer.
    _reading_set_by`) only describe the plan and fall back to their default
    wording.

    Args:
        drafts: The decisions as `propose_plan` took them.
        requirements: Every requirement the plan states once the call lands,
            carried forward or new.
    """
    decisions: list[Decision] = []
    for draft in drafts:
        citing = [
            requirement for requirement in requirements if draft.slot in requirement.decision_ids
        ]
        fixed: dict[str, Any] = {}
        for field in dict.fromkeys(draft.fixes or ()):
            stated = [_stated(r, field) for r in citing if field in type(r).model_fields]
            if stated and stated[0] is not None and all(_same(stated[0], v) for v in stated):
                fixed[field] = stated[0]
        decisions.append(
            Decision(
                **draft.model_dump(exclude={"fixes"}),
                fixes=DecisionFixes(**fixed) if fixed else None,
            )
        )
    return decisions


class PlanStatus(TitlelessEnum):
    """Where a plan sits in the plan-approve-execute cycle.

    `DRAFT` is a plan whose decisions and requirements are recorded and whose
    workflow is not. `PENDING_APPROVAL` carries that workflow and awaits the
    user. `APPROVED` and `REJECTED` record their answer. `EXECUTING`,
    `COMPLETED` and `FAILED` belong to the run an approved plan starts: it
    completes when every step succeeds and every requirement is met, and fails
    when one is not.
    """

    DRAFT = "draft"
    PENDING_APPROVAL = "pending_approval"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"


# Statuses a plan can still be rejected from. Once a run has started, the
# answer to "may this run?" has been given and acted on.
_REJECTABLE = frozenset(
    {PlanStatus.DRAFT, PlanStatus.PENDING_APPROVAL, PlanStatus.APPROVED, PlanStatus.REJECTED}
)


# The `Plan` fields a model writes, which the size limit counts. `declined` is
# left out: one decline per `UnsupportedOperation`, each ask capped at
# `MAX_DECLINE_ASK_CHARS`, bounds it on its own.
_MODEL_AUTHORED = {"question", "answer_mode", "decisions", "evidence_requirements"}


def _serialised_chars(value: Any) -> int:
    """Characters of `value` as compact JSON with non-ASCII left as written, as the seam renders it."""
    return len(json.dumps(value, separators=(",", ":"), ensure_ascii=False))


class Plan(StrictModel):
    """The plan object.

    A plan is filled in stages, and every stage returns a new instance:
    `question`, `decisions` and `evidence_requirements` are what you record,
    and `translation` and `workflow` arrive once the plan is translated. Say
    something in a `Decision` only where you genuinely weighed more than one
    alternative. A value the question states outright belongs in a
    requirement's own fields.
    """

    question: str = Field(min_length=1, max_length=MAX_QUESTION_CHARS)
    answer_mode: AnswerMode = Field(
        default=AnswerMode.SERIES,
        description=(
            "Whether the answer is a computed figure or the series itself. `series` (the "
            "default, and what a plan persisted before this field existed means) leaves an "
            "unread dataset requirement as the answer in its own right; `computed` requires "
            "every dataset requirement to feed a calculation. The planner's proposal tool requires the "
            "model to pass this explicitly on a first draft or a restart, so the default "
            "only governs a plan built another way, such as a legacy record deserialising "
            "without the field."
        ),
    )
    # Past `MAX_PLAN_DECISIONS`, a plan carries the decisions qualifying its answer
    # (`Decision.caveat`, `_decisions_fit_the_cap`), at most `MAX_REQUIREMENT_CAVEATS`
    # for each requirement it states.
    decisions: tuple[Decision, ...] = Field(
        default=(),
        max_length=MAX_PLAN_DECISIONS + MAX_REQUIREMENT_CAVEATS * MAX_HELD_REQUIREMENTS,
        description=(
            f"The methodological choices this plan makes, at most "
            f"{MAX_PLAN_DECISIONS}; a plan names its parts by `slot`, so no two decisions "
            "share one. Every decision has to be cited by at least one evidence "
            "requirement."
        ),
    )
    evidence_requirements: tuple[EvidenceRequirement, ...] = Field(
        default=(),
        max_length=MAX_HELD_REQUIREMENTS,
        description=(
            f"What the answer rests on, at most {MAX_PLAN_REQUIREMENTS} requirements: a "
            "dataset requirement per measure and source, naming every entity, and a derived "
            "requirement per calculation over them. Requirements are named by `id`. Under "
            "`answer_mode='series'`, a dataset requirement no calculation reads is itself "
            "the answer; under `answer_mode='computed'`, every dataset requirement has to "
            "feed one."
        ),
    )
    declined: Annotated[tuple[Decline, ...], HASHED_WHEN_SET] = Field(
        default=(),
        max_length=MAX_DECLINES,
        description=(
            "The parts of the question this plan does not answer because an `unsupported` "
            "answer earned their decline, one per `code` and `ask_ref`."
        ),
    )
    asks: Annotated[
        tuple[Annotated[str, Field(min_length=1, max_length=MAX_ASK_CHARS)], ...],
        HASHED_WHEN_SET,
    ] = Field(
        default=(),
        max_length=MAX_PLAN_ASKS,
        description="The outputs the user asked for, numbered by position, as the brief lists them.",
    )
    answered: Annotated[tuple[AskAnswer, ...], HASHED_WHEN_SET] = Field(
        default=(),
        max_length=MAX_PLAN_ASKS,
        description="The requirements that answer each ask this plan does not decline.",
    )
    ask_breakdowns: Annotated[tuple[AskBreakdown, ...], HASHED_WHEN_SET] = Field(
        default=(),
        max_length=MAX_PLAN_ASKS,
        description="The parties each ask breaks its answer down by, for the asks that name any.",
    )
    asks_across_years: Annotated[tuple[int, ...], HASHED_WHEN_SET] = Field(
        default=(),
        max_length=MAX_PLAN_ASKS,
        description=(
            "The numbers of the asks the user put to one figure, or one ranking, over "
            "several years taken together."
        ),
    )
    period_licences: Annotated[tuple[PeriodLicence, ...], HASHED_WHEN_SET] = Field(
        default=(),
        max_length=MAX_PLAN_REQUIREMENTS,
        description=(
            "The steps whose years the user's own words, as the planner read them, ask "
            "to combine into one figure or one ranking, each held while its step "
            "combines them that way (`period_licences_held`)."
        ),
    )
    rolling_copies: Annotated[tuple[RollingCopy, ...], HASHED_WHEN_SET] = Field(
        default=(),
        max_length=MAX_PLAN_REQUIREMENTS,
        description=(
            "The requirements `propose_plan` added so that only a step rolling from its "
            "start reads a dataset's whole history, each with the requirement it copies "
            "(`whole_history_split`)."
        ),
    )
    first_years: Annotated[tuple[FirstYear, ...], HASHED_WHEN_SET] = Field(
        default=(),
        max_length=MAX_PLAN_REQUIREMENTS,
        description=(
            "The first year on record of the dataset each dataset requirement pins, for the "
            "requirements a step rolling from its start reads."
        ),
    )
    translation: Translation | None = None
    workflow: Workflow | None = None
    version: int = 1
    previous_version: int | None = None
    status: PlanStatus = PlanStatus.DRAFT
    approved_revision: str | None = None
    approved_at: Annotated[datetime | None, HASHED_WHEN_SET] = None

    @field_validator("question")
    @classmethod
    def _question_is_stated(cls, question: str) -> str:
        """Reject a question of whitespace alone, since a plan answers something."""
        if not question.strip():
            raise ValueError("the question is blank; state the question this plan answers")
        return question

    @model_validator(mode="after")
    def _decisions_fit_the_cap(self) -> Plan:
        """Reject a plan stating more than `MAX_PLAN_DECISIONS` decisions that qualify
        nothing. A decision qualifying the answer (`Decision.caveat`) is not counted:
        `propose_plan` writes it for the plan whatever else it states."""
        counted = sum(decision.caveat is None for decision in self.decisions)
        if counted > MAX_PLAN_DECISIONS:
            raise ValueError(
                f"the plan states {counted} decisions, over the {MAX_PLAN_DECISIONS} a plan "
                "may take"
            )
        return self

    @model_validator(mode="after")
    def _record_fits_the_size_limit(self) -> Plan:
        """Reject a plan whose model-authored part states more than `MAX_PLAN_REQUIREMENTS` requirements or serialises past `MAX_PLAN_RECORD_CHARS`.

        The field limits bound each string and collection on its own, and a
        plan that fills many of them at once is still far past what the
        planner's brief can carry. The message names the measured size and the
        parts that take the most of it, so the model can shorten those. Both
        limits measure the plan the model wrote (`authored`), without the
        copies `whole_history_split` adds. The size leaves out the scope
        decisions and the decisions qualifying the answer that `propose_plan`
        writes (`is_scope_choice`, `Decision.caveat`): the model cannot shorten
        them, and `MAX_PLAN_DECISIONS` and the requirements bound how many a
        plan holds.
        """
        written = authored(self)
        if len(written.evidence_requirements) > MAX_PLAN_REQUIREMENTS:
            raise ValueError(
                f"the plan states {len(written.evidence_requirements)} evidence requirements, "
                f"over the {MAX_PLAN_REQUIREMENTS} a plan may take"
            )
        record = written.model_dump(mode="json", include=_MODEL_AUTHORED)
        record["decisions"] = [
            dumped
            for dumped, decision in zip(record["decisions"], written.decisions, strict=True)
            if not is_scope_choice(decision) and decision.caveat is None
        ]
        size = _serialised_chars(record)
        if size <= MAX_PLAN_RECORD_CHARS:
            return self
        parts = sorted(
            [
                (_serialised_chars(decision), f"decision {decision['slot']!r}")
                for decision in record["decisions"]
            ]
            + [
                (_serialised_chars(requirement), f"requirement {requirement['id']!r}")
                for requirement in record["evidence_requirements"]
            ]
            + [(_serialised_chars(record["question"]), "the question")],
            reverse=True,
        )
        largest = ", ".join(f"{name} ({chars:,})" for chars, name in parts[:3])
        raise ValueError(
            f"the plan is {size:,} characters as serialised, over the {MAX_PLAN_RECORD_CHARS:,} "
            f"a plan may take. Shorten the longest parts, or drop entities, options and "
            f"predicates that do not change the answer: {largest}"
        )

    @model_validator(mode="after")
    def _references_resolve(self) -> Plan:
        """Reject a plan whose parts do not agree with each other.

        Slots, requirement ids and declines (by code and ask) are unique, so a
        citation is unambiguous (`check_references`).
        Every `decision_ids` entry names a decision the plan makes, and once
        the plan states any requirement, every decision is cited by one: a
        decision no requirement rests on is either a decision about nothing or
        a requirement the planner left out. Beyond decisions, the requirements
        themselves have to agree: every input a requirement reads names a
        requirement the plan states, the requirements form a DAG, nothing reads
        a comparison, a `RequirementSide.entity` or a growth window has to fit
        the one dataset requirement its input chain rests on, where that chain
        is single-input all the way down, a rank over more than one year per
        entity states a reduction, and under `answer_mode == AnswerMode.COMPUTED`
        every dataset requirement feeds a calculation. Then every requirement
        states each value a decision it cites fixes
        (`_check_requirements_honour_their_decisions`), and a population-wide
        claim (`among='population'`, `denominator_scope='pooled'`) has to cite
        a decision whose `fixes` states it. Last, each ask the plan
        carries is answered or declined exactly once, an answer naming any
        requirement the plan states, one another reads included
        (`_check_asks_accounted`), and an answer keeps in one output every
        party its ask breaks down by unless that part is declined
        (`_check_asks_keep_their_parties`), and a share or quotient answering
        an ask that totals across a party divides by a whole spanning more
        than the numerator's own members of it (`check_wholes_span_their_totals`).

        The requirements are judged as they run: split so each dataset read
        from its first year serves only its rolling-start readers
        (`split_of`), the split being the identity on a plan already split.
        A dataset requirement whose first year the split reads and
        `first_years` does not hold is refused before any shape is built
        (`OpenPeriodFault.FIRST_YEAR_UNKNOWN`). A plan carrying
        `rolling_copies` carries exactly the split of the plan the model
        wrote (`_check_copies_are_the_split`). A refusal of the split
        requirements names the requirements the model wrote, in its ids and
        its text (`_naming_origins`).
        """
        slots = [decision.slot for decision in self.decisions]
        check_references(
            self.evidence_requirements,
            slots=slots,
            declines=[decline.key for decline in self.declined],
            made=slots,
        )
        # The model's own requirements: a copy the split adds reads copies of a
        # whole's part.
        copied = copy_origins(self.rolling_copies)
        _check_wholes_name_their_parts(
            {r.id: _reading(r, copied) for r in self.evidence_requirements if r.id not in copied}
        )
        _check_first_years(self)
        split = split_of(self)
        if split.first_year_unknown:
            raise _first_year_unknown(self, split.first_year_unknown[0])
        if self.rolling_copies:
            _check_copies_are_the_split(self)
        try:
            _check_requirement_graph(
                {requirement.id: requirement for requirement in split.requirements},
                answer_mode=self.answer_mode,
                whole_history=split.whole_history,
                origins=split.origins(),
            )
        except ValueError as error:
            raise _naming_origins(error, split) from error
        decisions_by_slot = {decision.slot: decision for decision in self.decisions}
        _check_requirements_honour_their_decisions(self.evidence_requirements, decisions_by_slot)
        _check_population_claims_are_cited(self.evidence_requirements, decisions_by_slot)
        _check_asks_accounted(self.asks, self.answered, self.declined, self.evidence_requirements)
        _check_asks_keep_their_parties(self)
        check_wholes_span_their_totals(self)
        _check_asks_across_years(self.asks, self.asks_across_years)
        self._translation_and_workflow_agree()
        return self

    def requirement(self, requirement_id: str) -> EvidenceRequirement:
        """Return the requirement this plan states with id `requirement_id`.

        Raises:
            KeyError: If this plan states no requirement by that id.
        """
        for candidate in self.evidence_requirements:
            if candidate.id == requirement_id:
                return candidate
        raise KeyError(f"this plan states no requirement {requirement_id!r}")

    def _translation_and_workflow_agree(self) -> None:
        """Reject a translation or a workflow that does not prove this plan's requirements.

        The compiler builds a `Workflow` from a `Translation` only once every
        requirement is carried out, exactly once, by an operation that agrees
        with it field for field (`planner.proof.mismatches`); a step naming an
        operation the translation lacks, or evidence bound to a requirement
        the plan lacks or a source the workflow never selected, is a wiring
        fault rather than a proposal to report back.

        `planner.proof` is imported here, not at this module's own top level,
        to break the cycle its own docstring explains: `planner.proof`
        imports `planner.plan` for the requirement classes, so `planner.plan`
        cannot import it back at module scope.
        """
        if (self.translation is None) != (self.workflow is None):
            raise ValueError("a plan carries its translation and its workflow together")
        if self.translation is None or self.workflow is None:
            return

        from planner.proof import mismatches  # noqa: PLC0415 -- breaks a module-level import cycle

        by_id = {requirement.id: requirement for requirement in self.evidence_requirements}
        carried_out: dict[str, list[str]] = {rid: [] for rid in by_id}
        unknown_requirements = sorted(
            operation.id
            for operation in self.translation.operations
            if operation.requirement_id not in by_id
        )
        if unknown_requirements:
            raise ValueError(
                f"operations name requirements this plan does not state: {unknown_requirements!r}"
            )
        for operation in self.translation.operations:
            carried_out[operation.requirement_id].append(operation.id)
        unsatisfied = sorted(rid for rid, op_ids in carried_out.items() if not op_ids)
        if unsatisfied:
            raise ValueError(f"requirements no operation carries out: {unsatisfied!r}")
        duplicated = sorted(rid for rid, op_ids in carried_out.items() if len(op_ids) > 1)
        if duplicated:
            raise ValueError(f"requirements carried out by more than one operation: {duplicated!r}")

        satisfied_by = {rid: op_ids[0] for rid, op_ids in carried_out.items()}
        operations_by_id = {operation.id: operation for operation in self.translation.operations}
        disagreements = sorted(
            f"{op_id!r} on requirement {rid!r}: {problems!r}"
            for rid, op_id in satisfied_by.items()
            if (problems := mismatches(by_id[rid], operations_by_id[op_id], satisfied_by))
        )
        if disagreements:
            raise ValueError(f"operations disagree with what they carry out: {disagreements!r}")

        operation_ids = set(operations_by_id)
        orphaned = sorted({step.operation_id for step in self.workflow.steps} - operation_ids)
        if orphaned:
            raise ValueError(
                f"workflow steps compile operations the translation lacks: {orphaned!r}"
            )

        unknown_bindings = sorted(
            binding.requirement_id
            for binding in self.workflow.evidence_bindings
            if binding.requirement_id not in by_id
        )
        if unknown_bindings:
            raise ValueError(
                f"evidence is bound to requirements the plan lacks: {unknown_bindings!r}"
            )
        known_spec_ids = {source.spec_id for source in self.workflow.sources}
        unpinned_bindings = sorted(
            binding.step_id
            for binding in self.workflow.evidence_bindings
            if binding.spec_id not in known_spec_ids
        )
        if unpinned_bindings:
            raise ValueError(
                f"evidence is bound to a spec_id the workflow never selected: {unpinned_bindings!r}"
            )

    def amend(
        self,
        *,
        decisions: Sequence[Decision] = (),
        evidence_requirements: Sequence[EvidenceRequirement] = (),
        removed_decisions: Sequence[str] = (),
        removed_evidence_requirements: Sequence[str] = (),
        question: str | None = None,
        answer_mode: AnswerMode | None = None,
        declined: Sequence[Decline] = (),
        asks: Sequence[str] | None = None,
        ask_breakdowns: Sequence[AskBreakdown] = (),
        asks_across_years: Sequence[int] = (),
        answered: Sequence[AskAnswer] = (),
        first_years: Sequence[FirstYear] | None = None,
    ) -> Plan:
        """Return a new, incremented-version `Plan` with a patch applied.

        The patch is keyed: a decision by its `slot`, a requirement by its
        `id`. An item in `decisions` whose slot this plan already states
        replaces the decision under that slot. One whose slot is new is added.
        An item this plan states and the patch does not mention is carried
        forward unchanged. Removal is stated in `removed_decisions` and
        `removed_evidence_requirements`. It is never inferred from an item's
        absence, since absence means unchanged. `answer_mode` works the same
        way: omitted, the amended plan keeps the mode this plan already
        carries; given, it switches, and the requirements are re-validated
        under the new one.

        The patched plan is validated in full, exactly as a first draft is, so
        it cannot introduce a repeated slot, a citation of a decision it does
        not make, or an uncited decision. Returns to `DRAFT` and clears
        `approved_revision`, `approved_at`, `translation` and `workflow`. An amendment
        (whether it lands before approval or mid-execution) always needs a
        fresh look, and the operations translated from the previous decisions
        may no longer fit the new ones, so the new version starts where a first
        draft does. It clears `period_licences` too, which `propose_plan`
        writes on the amended plan from this one's
        (`period_licences_held`). The patch applies to the plan the model
        wrote (`authored`), so the amended plan carries no
        `rolling_copies`, and its validation splits it again.

        Args:
            decisions: The decisions this patch adds or replaces, keyed by
                `slot`. Omit a decision to leave it as it is.
            evidence_requirements: Likewise for the requirements, keyed by
                `id`.
            removed_decisions: The slots of decisions this patch removes.
            removed_evidence_requirements: The ids of requirements this patch
                removes.
            question: The question the amended plan answers. Omitted, the plan
                keeps its own. The requirements and the question are read
                together, so a patch that moves one states the other.
            answer_mode: The mode the amended plan answers under. Omitted, the
                plan keeps its own.
            declined: The declines this patch adds or replaces, keyed by
                `code` and `ask_ref`. A decline naming no ask stays for the
                plan's life: the system still cannot compute what it names.
                One naming an ask replaces whatever accounted for that ask
                before.
            asks: The asks the amended plan accounts for. Omitted, the plan
                keeps its own. A list other than this plan's drops every
                answer and every decline naming an ask, so the patch accounts
                for the new list from scratch.
            ask_breakdowns: The parties each of `asks` breaks down by, read
                with `asks`: omitted with them, the plan keeps its own.
            asks_across_years: The numbers of the asks of `asks` put to one
                figure over several years, read with `asks` as
                `ask_breakdowns` is.
            answered: The answers this patch adds or replaces, keyed by
                `ask`. One replaces whatever accounted for its ask before,
                a decline included.
            first_years: The first years on record the amended plan reads
                its datasets' whole history from (`whole_history_split`).
                Omitted, it keeps this plan's for the dataset requirements
                it still states.

        Returns:
            A new `Plan`; `self` is unchanged.

        Raises:
            ValueError: If a removal names a slot or id this plan does not
                state.
        """
        breakdowns = self.ask_breakdowns if asks is None else tuple(ask_breakdowns)
        across_years = self.asks_across_years if asks is None else tuple(asks_across_years)
        asks = self.asks if asks is None else tuple(asks)
        requirements = _patched(
            authored(self).evidence_requirements,
            evidence_requirements,
            removed_evidence_requirements,
            key="id",
        )
        if first_years is None:
            stated = {r.id for r in requirements if isinstance(r, DatasetRequirement)}
            first_years = [entry for entry in self.first_years if entry.requirement_id in stated]
        return Plan.model_validate(
            {
                **self.model_dump(),
                "question": self.question if question is None else question,
                "answer_mode": self.answer_mode if answer_mode is None else answer_mode,
                "decisions": _patched(self.decisions, decisions, removed_decisions, key="slot"),
                "evidence_requirements": requirements,
                "rolling_copies": (),
                "first_years": tuple(first_years),
                **_patched_accounts(
                    self, answered=answered, declined=declined, same_asks=asks == self.asks
                ),
                "asks": asks,
                "ask_breakdowns": breakdowns,
                "asks_across_years": across_years,
                "period_licences": (),
                "translation": None,
                "workflow": None,
                "version": self.version + 1,
                "previous_version": self.version,
                "status": PlanStatus.DRAFT,
                "approved_revision": None,
                "approved_at": None,
            }
        )

    def with_translation(self, *, translation: Translation, workflow: Workflow) -> Plan:
        """Return a new `PENDING_APPROVAL` `Plan` carrying what will run.

        `version` is unchanged: translating a draft completes it, so one
        version stays one thing the user reviewed and approves by number.

        Args:
            translation: The operations the plan's requirements resolve into.
            workflow: The steps the compiler built from them.

        Returns:
            A new `Plan`; `self` is unchanged.

        Raises:
            ValueError: If this plan is not a `DRAFT`, since translating a
                plan the user has already seen would overwrite what they saw;
                or if the workflow does not compile this translation.
        """
        self._require(PlanStatus.DRAFT, to_be="translated")
        return Plan.model_validate(
            {
                **self.model_dump(),
                "decisions": self.decisions,
                "evidence_requirements": self.evidence_requirements,
                "translation": translation,
                "workflow": workflow,
                "status": PlanStatus.PENDING_APPROVAL,
            }
        )

    def approve(self, *, at: datetime) -> Plan:
        """Return a new `Plan` marked `APPROVED`, recording what was approved and when.

        The harness workflow's `approve` node enforces the actual gate; this
        records the resulting state, the revision it applies to, which
        `start_execution` checks against, and `at` as `approved_at`, from
        which `executor.run.execute_approved_plan` measures how long the
        approval stays in force.

        Args:
            at: When the user approved the plan, timezone-aware.

        Raises:
            ValueError: If the plan is not awaiting approval. Approving a
                second time, or approving a plan already answered, would
                overwrite the record of the first answer.
        """
        self._require(PlanStatus.PENDING_APPROVAL, to_be="approved")
        return self.model_copy(
            update={
                "status": PlanStatus.APPROVED,
                "approved_revision": plan_revision(self),
                "approved_at": at,
            }
        )

    def reject(self) -> Plan:
        """Return a new `Plan` marked `REJECTED`.

        Nothing in this plan may run afterwards. `planner.agent`'s
        `declare_ready` checks the status and refuses. The planner's own tools
        stay callable, so the user can ask for a revision.

        Raises:
            ValueError: If a run has already started or finished; the answer
                that run rested on cannot be withdrawn after the fact.
        """
        self._require(*_REJECTABLE, to_be="rejected")
        return self.model_copy(update={"status": PlanStatus.REJECTED})

    def start_execution(self) -> Plan:
        """Return a new `Plan` marked `EXECUTING`.

        Raises:
            ValueError: If the plan is not `APPROVED`, or if its content no
                longer hashes to `approved_revision` -- the user approved
                something else.
        """
        self._require(PlanStatus.APPROVED, to_be="executed")
        current = plan_revision(self)
        if current != self.approved_revision:
            raise ValueError(
                f"this plan's content ({current}) is not what was approved "
                f"({self.approved_revision}); it needs approving again"
            )
        return self.model_copy(update={"status": PlanStatus.EXECUTING})

    def complete(self) -> Plan:
        """Return a new `Plan` marked `COMPLETED`; only a run in progress completes."""
        self._require(PlanStatus.EXECUTING, to_be="completed")
        return self.model_copy(update={"status": PlanStatus.COMPLETED})

    def fail(self) -> Plan:
        """Return a new `Plan` marked `FAILED`; only a run in progress fails."""
        self._require(PlanStatus.EXECUTING, to_be="marked failed")
        return self.model_copy(update={"status": PlanStatus.FAILED})

    def _require(self, *allowed: PlanStatus, to_be: str) -> None:
        if self.status not in allowed:
            wanted = " or ".join(status.value for status in allowed)
            raise ValueError(
                f"this plan is {self.status.value}, so it cannot be {to_be}; "
                f"that takes a plan that is {wanted}"
            )


def plan_revision(plan: Plan) -> str:
    """Return the content hash identifying one revision of `plan`.

    Covers everything a user reviews -- question, decisions, requirements,
    translation, workflow, version -- and leaves out `status`,
    `approved_revision` and `approved_at`, so approving a revision does not
    change the value that was approved. A `HASHED_WHEN_SET` field enters once it holds
    something other than its default (`common.hashed_fields.revision_content`), so
    a plan saved without the field keeps its revision. Canonical JSON
    (RFC 8785) under SHA-256, the same discipline the data services use for
    their own identities.
    """
    content = revision_content(plan, exclude={"status", "approved_revision", "approved_at"})
    return hashlib.sha256(rfc8785.dumps(content)).hexdigest()


def split_of(plan: Plan) -> HistorySplit:
    """`plan`'s requirements as they run, each whole-history read serving only its rolling-start readers.

    The split of its requirements under the results answering it and its
    `first_years`, continued from its `rolling_copies` (`whole_history_split`).
    On a plan already split it is the identity.
    """
    return whole_history_split(
        plan.evidence_requirements,
        answering=answering_results(plan),
        first_years=plan.first_years,
        copies=plan.rolling_copies,
    )


def authored(plan: Plan) -> Plan:
    """`plan` as the model wrote it, without the copies `rolling_copies` names and with both fields the split writes empty.

    Every input naming a copy names its origin again, and `rolling_copies`
    and `first_years` are empty. The split of the result under `plan`'s
    `first_years` is `plan` again, and on a plan with no copies this
    changes only `first_years`. The result is not validated, since a
    rolling-start read in it needs the first years it no longer carries.
    `translation` and `workflow` are `plan`'s own.
    """
    origins = copy_origins(plan.rolling_copies)
    return plan.model_copy(
        update={
            "evidence_requirements": tuple(
                _reading(requirement, origins)
                for requirement in plan.evidence_requirements
                if requirement.id not in origins
            ),
            "rolling_copies": (),
            "first_years": (),
        }
    )


def with_authored_requirements(plan: Plan, requirements: Sequence[EvidenceRequirement]) -> Plan:
    """`plan` stating `requirements` in place of those the model wrote, split again under its `first_years`.

    `requirements` stand for `authored(plan)`'s. The result carries their
    split (`split_of`) in `evidence_requirements` and `rolling_copies`, and a
    copy that copies the same origin in `plan` keeps `plan`'s pin for it,
    its own windowed pin (`submission.service._window_pins`).

    Raises:
        ValueError: If the plan stating `requirements`, or its split, does
            not validate.
    """
    origins = copy_origins(plan.rolling_copies)
    pins = {
        requirement.id: requirement.source_spec_id
        for requirement in plan.evidence_requirements
        if requirement.id in origins and isinstance(requirement, DatasetRequirement)
    }
    stated = Plan.model_validate(
        {
            **plan.model_dump(),
            "evidence_requirements": tuple(requirements),
            "rolling_copies": (),
        }
    )
    split = split_of(stated)
    if not split.copies:
        return stated
    kept = split.origins()
    return Plan.model_validate(
        {
            **stated.model_dump(),
            "evidence_requirements": tuple(
                requirement.model_copy(update={"source_spec_id": pins[requirement.id]})
                if requirement.id in pins and kept.get(requirement.id) == origins[requirement.id]
                else requirement
                for requirement in split.requirements
            ),
            "rolling_copies": split.copies,
        }
    )


def restated_to_anchors(plan: Plan, *, anchors: Mapping[str, int]) -> Plan:
    """`plan` with each rolling requirement's years stated from the newest year `anchors` gives it.

    `anchors` holds a newest year for every requirement of `plan` that rolls
    (`rolls`), copies included, as the run resolves it for the requirement's
    group. Each rolling year moves by that year less the newest year the
    requirement states (`RelativePeriod.latest_implied_by`), either way, and
    keeps its offset. A fixed year stays. A copy moves with the requirement
    it copies. Where no year moves, `plan` itself comes back, with its
    revision. Otherwise the result is `plan` stating the moved requirements,
    split again (`with_authored_requirements`), and so held to every rule a
    proposal is. `plan` is a draft. A translation states the years too, and
    `restated_translation` moves them by the same years.

    Raises:
        KeyError: `anchors` does not name exactly the rolling requirements of
            `plan`.
        RollingPlanError: `RollingIssue.YEARS_DISAGREE`, naming each
            requirement that `anchors` gives one newest year and a copy of it
            another, since the one year the model wrote cannot state both.
        ValueError: A moved requirement or the moved plan breaks a plan rule,
            raised as validating a proposal raises it.
    """
    moves = _rolling_moves(plan, anchors=anchors)
    if not any(moves.values()):
        return plan
    return with_authored_requirements(
        plan,
        [
            _moved(requirement, by=moves[requirement.id])
            if moves.get(requirement.id)
            else requirement
            for requirement in authored(plan).evidence_requirements
        ],
    )


def restated_translation(
    translation: Translation, *, plan: Plan, anchors: Mapping[str, int]
) -> Translation:
    """`translation` of `plan` with each operation's years moved as `restated_to_anchors` moves its requirement's.

    An operation repeats its requirement's years and `rolling`
    (`planner.proof.mismatches`), so each operation carrying out a rolling
    requirement, a copy's included, moves by that requirement's years. Where
    no year moves, `translation` itself comes back.

    Raises:
        KeyError, RollingPlanError, ValueError: As `restated_to_anchors`.
    """
    moves = _rolling_moves(plan, anchors=anchors)
    if not any(moves.values()):
        return translation
    return Translation.model_validate(
        {
            **translation.model_dump(),
            "operations": [
                _moved(operation, by=moves[operation.requirement_id])
                if moves.get(operation.requirement_id)
                else operation
                for operation in translation.operations
            ],
        }
    )


class Restatement(StrictModel):
    """One requirement's rolling years, as written and as translation stated them from the data.

    `anchor` is the newest year the data holds that they now count from,
    and `years_from` and `years_to` the years the requirement states that
    follow the newest data, in the order it states them (a window's first
    and last, a filter's predicates), before and after.
    """

    requirement_id: Identifier
    anchor: Year
    years_from: tuple[Year, ...]
    years_to: tuple[Year, ...]


def restatements(plan: Plan, restated: Plan) -> tuple[Restatement, ...]:
    """Each requirement of `plan` whose rolling years `restated` states otherwise, in plan order.

    `restated` is `plan` as `restated_to_anchors` returns it. A copy is left
    out: it moves with the requirement it copies, and the model wrote only
    that one.
    """
    copies = copy_origins(plan.rolling_copies)
    moved: list[Restatement] = []
    for requirement in plan.evidence_requirements:
        if requirement.id in copies or not rolls(requirement):
            continue
        after = restated.requirement(requirement.id)
        years_from = tuple(year for year, _ in _rolling_years(requirement))
        years_to = tuple(year for year, _ in _rolling_years(after))
        if years_from != years_to:
            moved.append(
                Restatement(
                    requirement_id=requirement.id,
                    anchor=_stated_newest(after),
                    years_from=years_from,
                    years_to=years_to,
                )
            )
    return tuple(moved)


def _rolling_moves(plan: Plan, *, anchors: Mapping[str, int]) -> dict[str, int]:
    """How many years each rolling requirement of `plan` moves to the newest year `anchors` gives it.

    Keyed by every rolling requirement's id, each copy's beside its origin's,
    the two moving alike. Raises as `restated_to_anchors`.
    """
    rolling = {requirement.id for requirement in plan.evidence_requirements if rolls(requirement)}
    if anchors.keys() != rolling:
        raise KeyError(
            f"anchors name {sorted(anchors)!r} where the plan's rolling requirements are "
            f"{sorted(rolling)!r}"
        )
    origins = copy_origins(plan.rolling_copies)
    newest: dict[str, set[int]] = {}
    for requirement_id, year in anchors.items():
        newest.setdefault(origins.get(requirement_id, requirement_id), set()).add(year)
    if torn := sorted(origin for origin, years in newest.items() if len(years) > 1):
        stated = "; ".join(f"{origin!r} {sorted(newest[origin])!r}" for origin in torn)
        raise RollingPlanError(
            RollingIssue.YEARS_DISAGREE,
            f"{torn!r} and their copies over the dataset's whole history count from different "
            f"newest years ({stated}), and one requirement states one; read the whole history "
            "and the stated years through requirements of their own",
            requirement_ids=torn,
        )
    moves: dict[str, int] = {}
    for requirement in plan.evidence_requirements:
        if requirement.id in rolling:
            (year,) = newest[origins.get(requirement.id, requirement.id)]
            moves[requirement.id] = year - _stated_newest(requirement)
    return moves


def _stated_newest(requirement: EvidenceRequirement) -> int:
    """The one newest year a rolling `requirement` of a validated plan states."""
    (year,) = {period.latest_implied_by(year) for year, period in _rolling_years(requirement)}
    return year


def _moved[T: StrictModel](carried: T, *, by: int) -> T:
    """`carried`, a rolling requirement or the operation carrying one out, with each year that follows the newest data moved `by` years, validated again.

    A growth or a rank states `rolling` beside its window, and a filter on
    each period predicate.
    """
    fields = carried.model_dump()
    if (rolling := fields.get("rolling")) is not None:
        for bound in ("window_start", "window_end"):
            if rolling[bound] is not None:
                fields[bound] += by
    else:
        fields["predicates"] = [
            {**predicate, "year": predicate["year"] + by}
            if predicate.get("rolling") is not None
            else predicate
            for predicate in fields["predicates"]
        ]
    return type(carried).model_validate(fields)


def _check_first_years(plan: Plan) -> None:
    """Reject `first_years` naming a requirement twice, or one that is no dataset requirement of `plan`."""
    named = [entry.requirement_id for entry in plan.first_years]
    _reject_repeats(named, what="first year")
    datasets = {r.id for r in plan.evidence_requirements if isinstance(r, DatasetRequirement)}
    if unknown := sorted(set(named) - datasets):
        raise ValueError(f"first years name no dataset requirement of this plan: {unknown!r}")


def _first_year_unknown(plan: Plan, requirement_id: str) -> RollingPlanError:
    """`RollingIssue.NEEDS_OPEN_PERIOD` for a dataset requirement read from a first year `plan` does not hold.

    It carries the requirement's pin as `spec_id`, `None` when it pins
    none. `submission.service` words it from what the session read.
    """
    requirement = plan.requirement(requirement_id)
    assert isinstance(requirement, DatasetRequirement), requirement
    return RollingPlanError(
        RollingIssue.NEEDS_OPEN_PERIOD,
        f"{requirement_id!r} is read from a start that follows the newest data, which needs "
        "the first year its dataset holds, and none is on record. Where the window covers "
        "several years, state the start as a fixed year and roll only the end; a window of "
        "one year cannot, so drop `rolling` and `period_end_open` there for fixed years",
        requirement_ids=[requirement_id],
        fault=OpenPeriodFault.FIRST_YEAR_UNKNOWN,
        spec_id=requirement.source_spec_id,
    )


def _check_copies_are_the_split(plan: Plan) -> None:
    """Reject `rolling_copies` other than the split of the plan the model wrote.

    The split of `authored(plan)` under `plan.first_years` has to give
    `plan`'s requirements and copies, apart from each copy's
    `source_spec_id`, which `submission.service._window_pins` moves to the
    copy's own windowed pin.
    """
    written = authored(plan)
    again = whole_history_split(
        written.evidence_requirements,
        answering=answering_results(written),
        first_years=plan.first_years,
        copies=(),
    )
    copies = {copy.copy_id for copy in plan.rolling_copies}

    def unpinned(requirements: Iterable[EvidenceRequirement]) -> list[EvidenceRequirement]:
        return [
            requirement.model_copy(update={"source_spec_id": None})
            if requirement.id in copies and isinstance(requirement, DatasetRequirement)
            else requirement
            for requirement in requirements
        ]

    if again.copies != plan.rolling_copies or unpinned(again.requirements) != unpinned(
        plan.evidence_requirements
    ):
        raise ValueError(
            "the plan's rolling copies are not the split of the requirements the model wrote"
        )


def _naming_origins(error: ValueError, split: HistorySplit) -> ValueError:
    """`error` naming, in its text and its requirements, the requirement each copy among them copies.

    The rules over the requirements as they run raise a `RollingPlanError`,
    an `UnsupportedPlanError` or a plain `ValueError`
    (`_check_requirement_graph`). Each comes back as its own kind.
    """
    message = authored_text(str(error), split.origins())
    if isinstance(error, RollingPlanError):
        named = dict.fromkeys(origin_of(split, step_id) for step_id in error.requirement_ids)
        return RollingPlanError(
            error.issue,
            message,
            requirement_ids=list(named),
            fault=error.fault,
            spec_id=error.spec_id,
        )
    if isinstance(error, UnsupportedPlanError):
        return UnsupportedPlanError(error.operation, message)
    if type(error) is ValueError:
        return ValueError(message)
    return error


class ResultScopeGap(NamedTuple):
    """A result given per member of a party nobody named, with no decision saying it should be.

    `output_id` is the requirement whose result answers, `ask` the number of
    the ask it answers (`None` for a terminal of a computed plan no ask
    names), `role` the `across` word of the party it keeps one figure per
    member of, and
    `dataset_id` the dataset requirement whose `across` opened that role.
    """

    output_id: str
    ask: int | None
    role: str
    dataset_id: str


def answering_results(plan: Plan) -> dict[str, int | None]:
    """The requirements whose results answer `plan`, each with the ask it answers.

    Every output `plan.answered` names, keyed to its ask, and, under
    `AnswerMode.COMPUTED` or for a plan that answers no ask, every
    requirement nothing else reads, keyed to `None`. The validator
    (`unstated_result_scopes`) and the plan card
    (`harness.plan_display.answer_scopes`) read this one set.
    """
    outputs: dict[str, int | None] = {}
    for answer in plan.answered:
        for output in answer.outputs:
            outputs.setdefault(output, answer.ask)
    if plan.answer_mode is AnswerMode.COMPUTED or not plan.answered:
        read = {
            input_id
            for requirement in plan.evidence_requirements
            for input_id in inputs_of(requirement)
        }
        for requirement in plan.evidence_requirements:
            if requirement.id not in read:
                outputs.setdefault(requirement.id, None)
    return outputs


def unstated_result_scopes(plan: Plan) -> list[ResultScopeGap]:
    """Each answering result of `plan` that keeps an open role no step or decision states.

    A dataset requirement's `across` reads every member of a party nobody
    named (`planner.shapes.RoleDimension.open`). Every verb but an
    aggregate carries that dimension through, so a growth read straight off
    it is one growth per member, which a question asking for the total does
    not want. A result kept per member of such a party passes when the plan
    states that scope: a step on its way to the result operates over the
    role by name (`_operand_roles`: a rank ordering it, a share or quotient
    keyed on it, or a place-level filter on it, since `_filter_shape`
    refuses an `EntityIn` naming a member of an open role, by
    `unknown_names`), or, like `among='population'`
    (`_check_population_claims_are_cited`), the dataset requirement that
    opened the role cites a decision whose `fixes` states its `across`, every
    word of it. An ask the result answers states the scope too when it
    breaks down by that party (`AskBreakdown.by`), in the role's word or
    another on the same side of the flow (`planner.parties.same_party`): it
    asks for one figure per member. A
    result that sums the role away first, through an aggregate, keeps no
    open role and passes.

    The results checked are those `answering_results` names, except a
    dataset requirement of an `AnswerMode.SERIES` plan that no ask names:
    that is the series retrieved as asked, one per member, for a chart to
    draw. Requires a plan whose requirements already validated, and reads
    them as they run (`split_of`), naming for each gap the dataset
    requirement the model wrote.
    """
    split = split_of(plan)
    requirements_by_id = {requirement.id: requirement for requirement in split.requirements}
    decisions = {decision.slot: decision for decision in plan.decisions}
    shapes = _split_shapes(split)
    by = {breakdown.ask: breakdown.by for breakdown in plan.ask_breakdowns}
    listed: dict[str, set[str]] = {}
    for answer in plan.answered:
        for output in answer.outputs:
            listed.setdefault(output, set()).update(by.get(answer.ask, ()))
    gaps: list[ResultScopeGap] = []
    for output_id, ask in answering_results(plan).items():
        if (
            ask is None
            and plan.answer_mode is AnswerMode.SERIES
            and is_dataset(requirements_by_id[output_id])
        ):
            continue
        operated = {
            key
            for step_id in _upstream(output_id, requirements_by_id)
            for key in _operand_roles(requirements_by_id[step_id], shapes)
        }
        for role in shapes[output_id].roles:
            if not role.open or (role.role, role.requirement_id) in operated:
                continue
            dataset = requirements_by_id[role.requirement_id]
            assert isinstance(dataset, DatasetRequirement)
            assert role.role is not None, "an open dimension is one `across` word"
            if party_among(role.role, listed.get(output_id, ())):
                continue
            if any(
                (fixes := decisions[slot].fixes) is not None and _same(fixes.across, dataset.across)
                for slot in dataset.decision_ids
            ):
                continue
            gaps.append(ResultScopeGap(output_id, ask, role.role, origin_of(split, dataset.id)))
    return gaps


class PeriodReductionGap(NamedTuple):
    """A step that turns several years into one figure for a result no ask across years asks for.

    `step_id` is the reducing step and `held` the years it reduces
    (`YearsHeld`), with the dataset requirement it rests on.
    `output_id` is the first answering result computed from it
    (`answering_results`) that answers an ask `plan.asks_across_years` does
    not number, or answers no ask, and `asks` the numbers of those asks,
    empty when it answers none. `asked_across` says another answering
    result computed from the step answers an ask across years, so the step
    itself is asked for. `together_with` names the other steps whose
    one-year changes the plan has to make with this one for
    `held.one_year` to be admitted, empty where it is admitted alone
    (`_one_year_offered`).
    """

    step_id: str
    output_id: str
    asks: tuple[int, ...]
    held: YearsHeld
    asked_across: bool
    together_with: tuple[str, ...] = ()


def unstated_period_reductions(plan: Plan) -> list[PeriodReductionGap]:
    """Each step of `plan` that reduces across years for a result no ask across years asks for.

    A dataset requirement's `period` is the window of the search that found
    its source, which for a question naming no years is every year the
    source holds. An `aggregate` whose `group_by` lacks `period`, or a `rank`
    with a `reduction` and `per_period` unset, turns the years of its input
    into one figure per entity (`_years_reduced`). Such a step passes when
    every answering result computed from it (`answering_results`), the step
    included, answers at least one ask and every ask it answers is one
    `plan.asks_across_years` numbers: the asks the orchestrator, which holds
    the user's words, marked as one figure over several years. It passes
    too when `plan.period_licences` holds a licence for it, with its kind
    and `reduction`: the planner, where it reads the user's own words, said
    they ask for that figure (`period_licences_held`). Otherwise a result
    that answers no ask, a step no answering result is computed from, and a
    plan with no asks license nothing.

    Requires a plan whose requirements already validated, and reads them as
    they run (`split_of`). A copy the split adds passes its input's years on
    (`passes_observation_years`), so it reduces none, and each gap's
    `held.dataset_id` names the dataset requirement the model wrote.
    """
    split = split_of(plan)
    requirements_by_id = {requirement.id: requirement for requirement in split.requirements}
    shapes = _split_shapes(split)
    across_years = set(plan.asks_across_years)
    licensed = set(plan.period_licences)
    asks_of: dict[str, list[int]] = {output: [] for output in answering_results(plan)}
    for answer in plan.answered:
        for output in answer.outputs:
            asks_of[output].append(answer.ask)
    gaps: list[PeriodReductionGap] = []
    for step in split.requirements:
        reduced = _years_reduced(step, shapes)
        if reduced is None or _licence_for(step) in licensed:
            continue
        reduced = reduced._replace(dataset_id=origin_of(split, reduced.dataset_id))
        computed = [
            output for output in asks_of if step.id in _upstream(output, requirements_by_id)
        ] or [step.id]
        offending: tuple[str, tuple[int, ...]] | None = None
        asked_across = False
        for output_id in computed:
            asks = asks_of.get(output_id, ())
            unmarked = tuple(ask for ask in asks if ask not in across_years)
            if unmarked or not asks:
                offending = offending or (output_id, unmarked)
            if any(ask in across_years for ask in asks):
                asked_across = True
        if offending is not None:
            gaps.append(
                PeriodReductionGap(
                    step_id=step.id,
                    output_id=offending[0],
                    asks=offending[1],
                    held=reduced,
                    asked_across=asked_across,
                )
            )
    return _one_year_offered(gaps, plan)


def _one_year_offered(gaps: list[PeriodReductionGap], plan: Plan) -> list[PeriodReductionGap]:
    """`gaps` with the one year a refusal offers each step: one `plan` is admitted with once it reads it.

    Each step is offered the first year tried (`_one_year_tried`) whose
    change, made to that step alone, gives a plan that passes
    (`_one_year_admits`). The steps not asked across years that no such
    year passes for are then tried together, each on its first year tried
    and then each on its last: the two sides of one pairing can each match
    the other only once both read one year. Where one of those plans
    passes, each of them is offered its year in it, together with the
    others (`PeriodReductionGap.together_with`). Every other step is
    offered none.
    """
    tried = {gap.step_id: _one_year_tried(gap.held, gap.step_id, plan) for gap in gaps}
    alone: dict[str, OneYearRead] = {}
    for step_id, candidates in tried.items():
        read = next((read for read in candidates if _one_year_admits(plan, {step_id: read})), None)
        if read is not None:
            alone[step_id] = read
    unmet = [
        gap.step_id
        for gap in gaps
        if tried[gap.step_id] and gap.step_id not in alone and not gap.asked_across
    ]
    together: dict[str, OneYearRead] = {}
    if len(unmet) > 1:
        joint = ({step_id: tried[step_id][at] for step_id in unmet} for at in (0, -1))
        together = next((reads for reads in joint if _one_year_admits(plan, reads)), {})
    return [
        gap._replace(
            held=gap.held._replace(one_year=alone.get(gap.step_id, together.get(gap.step_id))),
            together_with=tuple(step_id for step_id in together if step_id != gap.step_id)
            if gap.step_id in together
            else (),
        )
        for gap in gaps
    ]


def _one_year_tried(held: YearsHeld, step_id: str, plan: Plan) -> list[OneYearRead]:
    """The one years tried for `step_id` in `plan`, in order; none where `held` holds no year it can read alone.

    The year first tried is `held.one_year`. Where it follows the newest
    data, the change that reads it makes the step roll, or read a filter
    that rolls, so its rolling year joins the group the step's own rolling
    years would (`rolling_groups`) and is stated under the newest year the
    other rolling years of that group stand for. Where they stand for
    several, the year tried disagrees with one of them, so
    `_one_year_admits` refuses it (`rolling_years_disagree`). The year tried
    next is the last year the step holds today, fixed.
    """
    read = held.one_year
    if read is None:
        return []
    requirements_by_id = {requirement.id: requirement for requirement in plan.evidence_requirements}
    tried = [OneYearRead(year=held.years.end, offset=None)]
    if read.offset is not None:
        reads, rolling = _rolling_reads(requirements_by_id, joining=step_id)
        (members,) = [group for group in rolling_groups(reads, rolling=rolling) if step_id in group]
        newest = {
            period.latest_implied_by(year)
            for member in members
            if member != step_id
            for year, period in _rolling_years(requirements_by_id[member])
        }
        year = read.year if not newest else newest.pop() + read.offset
        tried.insert(0, OneYearRead(year=year, offset=read.offset))
    return tried


def _one_year_admits(plan: Plan, reads: Mapping[str, OneYearRead]) -> bool:
    """Whether `plan` passes once each step `reads` names reads its year there (`one_year_followed`).

    The change is made to the plan the model wrote (`authored`), as the
    planner would make it, and passes when that plan validates as a
    `Plan` under `plan`'s `first_years`, the rules every plan
    `propose_plan` records is built under: a rolling year can leave the
    step no year, a fixed year can be passed by a first year that follows
    the newest data, and a window a decision the rank cites fixes stays as
    the decision states it. Validation splits it (`split_of`), so a year
    that follows the newest data reads its dataset's whole history through
    copies of its own and every other result keeps the years it read. A
    dataset whose first year `plan` does not hold cannot be read so, and
    the change fails there.
    """
    followed = list(authored(plan).evidence_requirements)
    for step_id, read in reads.items():
        followed = one_year_followed(followed, step_id, read)
    return _admits(plan, followed)


def keep_years_admits(plan: Plan, step_id: str, change: Mapping[str, object]) -> bool:
    """Whether `plan` passes once the step `step_id` takes the fields `change` states.

    `change` is how a `period_reduction_unstated` refusal has the step keep
    its years (`submission.service._keep_years`): a `group_by` keeping
    `period`, or `per_period` true. It is made and checked as
    `_one_year_admits` makes and checks a one-year change, so a decision
    the step cites that fixes the field otherwise refuses it. A step the
    model did not write is no change it can make.
    """
    written = authored(plan).evidence_requirements
    if all(step.id != step_id for step in written):
        return False
    followed = [
        type(step).model_validate({**step.model_dump(), **change}) if step.id == step_id else step
        for step in written
    ]
    return _admits(plan, followed)


def _admits(plan: Plan, requirements: Sequence[EvidenceRequirement]) -> bool:
    """Whether the plan `plan`'s author wrote validates with `requirements`, under `plan`'s `first_years`."""
    try:
        Plan.model_validate(
            {
                **authored(plan).model_dump(),
                "evidence_requirements": requirements,
                "first_years": plan.first_years,
                "translation": None,
                "workflow": None,
            }
        )
    except ValueError:
        return False
    return True


def one_year_followed(
    requirements: Sequence[EvidenceRequirement], step_id: str, read: OneYearRead
) -> list[EvidenceRequirement]:
    """`requirements` once the step `step_id` reads the one year `read` names, and nothing else changes.

    This is the change a `period_reduction_unstated` refusal offers
    (`submission.service.OneYear`). A rank sets its own window on the year,
    both ends rolling at `read.offset` where it follows the newest data. An
    aggregate reads its own input through a new filter keeping the year with
    op `eq`, rolling at that offset, under the first free id of `one_year`,
    `one_year_2` and on.
    """
    rolling = None if read.offset is None else RelativePeriod(offset=read.offset)
    (index,) = [i for i, step in enumerate(requirements) if step.id == step_id]
    step = requirements[index]
    followed = list(requirements)
    if isinstance(step, RankRequirement):
        window = (
            None if rolling is None else RollingWindow(window_start=rolling, window_end=rolling)
        )
        followed[index] = step.model_copy(
            update={"window_start": read.year, "window_end": read.year, "rolling": window}
        )
        return followed
    assert isinstance(step, AggregateRequirement), step
    taken = {requirement.id for requirement in requirements}
    filter_id = next(
        candidate
        for candidate in itertools.chain(
            ("one_year",), (f"one_year_{n}" for n in itertools.count(2))
        )
        if candidate not in taken
    )
    one_year = FilterRequirement(
        id=filter_id,
        input_id=step.input_id,
        predicates=(PeriodThreshold(op="eq", year=read.year, rolling=rolling),),
    )
    followed[index : index + 1] = [one_year, step.model_copy(update={"input_id": filter_id})]
    return followed


def period_licences_held(
    plan: Plan, *, named: Sequence[str], previous: Plan | None
) -> tuple[PeriodLicence, ...]:
    """The licences `plan` holds: those of `previous` it keeps, and one for each step `named`.

    `named` is the `across_years` of the `propose_plan` call that states
    `plan`, the steps whose years the user's own words, as the planner read
    them, ask to combine. Only a call made where the planner reads those
    words names any (`submission.service.admit`). A licence of `previous`,
    the plan the call amends, holds while `plan` states a step of its id and
    kind that combines its years with the licensed reduction
    (`_keeps_licence`), whatever years it reads: a replan that narrows the
    years to one and a later one that widens them again keep the figure the
    user asked for. It lies unused while a `rank` of its id states no
    `reduction`, and admits the rank again once it states the licensed
    one. An `aggregate` grouping by `period`, a `rank` set `per_period`, a
    step stating another reduction, a step of another kind and a removed
    step drop it, since each states another figure. A step
    renamed together with its decisions holds no licence of its old id, and
    one named again in the same call takes a new one. A restart (`previous`
    `None`) holds only the steps it names.

    Raises:
        ValueError: If `named` names a step `plan` does not state, or one
            that keeps its years.
    """
    steps = {requirement.id: requirement for requirement in plan.evidence_requirements}
    granted = {name: _licence_for(steps.get(name)) for name in named}
    refused = sorted(name for name, licence in granted.items() if licence is None)
    if refused:
        raise ValueError(
            f"`across_years` names {', '.join(map(repr, refused))}, which the plan states as "
            "no step that combines years: name an `aggregate` whose `group_by` leaves out "
            "`period`, or a `rank` with a `reduction` and `per_period` false"
        )
    held = {
        licence.step_id: licence
        for licence in (previous.period_licences if previous is not None else ())
        if _keeps_licence(steps.get(licence.step_id), licence)
    }
    held.update((name, licence) for name, licence in granted.items() if licence is not None)
    return tuple(held[step_id] for step_id in steps if step_id in held)


def licences_restated(
    named: Sequence[str],
    *,
    requirements: Sequence[EvidenceRequirement],
    previous: Plan | None,
) -> frozenset[str]:
    """The steps of `named` whose licence `previous` already holds, as `requirements` state them.

    Naming such a step grants nothing `period_licences_held` would not
    carry anyway.
    """
    held = set(previous.period_licences) if previous is not None else set()
    return frozenset(
        requirement.id
        for requirement in requirements
        if requirement.id in named and _licence_for(requirement) in held
    )


def _keeps_licence(step: EvidenceRequirement | None, licence: PeriodLicence) -> bool:
    """Whether `step`, of the licence's id, keeps `licence`.

    It keeps it while it combines its years with the licensed reduction, or
    is a `rank` of that kind stating no `reduction`, which reads one year.
    An `aggregate` grouping by `period` and a `rank` set `per_period` state
    a figure for each year, which drops it.
    """
    if step is None or step.kind != licence.step_kind:
        return False
    if isinstance(step, RankRequirement) and step.reduction is None:
        return not step.per_period
    return _licence_for(step) == licence


def _licence_for(step: EvidenceRequirement | None) -> PeriodLicence | None:
    """The licence `step` needs to combine its years; `None` when it keeps them.

    An `aggregate` whose `group_by` lacks `period`, and a `rank` with a
    `reduction` and `per_period` false, combine the years they read.
    """
    if isinstance(step, AggregateRequirement):
        if GroupBy.PERIOD in step.group_by:
            return None
    elif isinstance(step, RankRequirement):
        if step.reduction is None or step.per_period:
            return None
    else:
        return None
    return PeriodLicence(step_id=step.id, step_kind=step.kind, reduction=step.reduction)


class OneYearRead(NamedTuple):
    """The one year a step can read in place of the years it holds, and how newer data moves it.

    `year` is the year it reads today, and `offset` the offset from the
    anchor at which a bound following the newest data keeps it, `None` for
    a fixed year.
    """

    year: int
    offset: int | None


class YearsHeld(NamedTuple):
    """The years a step reads from its input, one row per entity for each (`_years_held`).

    `years` are the years it holds today. `follows_newest` says newer data
    adds later years, and `until` the last year a fixed bound lets them
    reach, `None` when nothing stops them. `one_year` is the year the step
    can read alone and every series keeps holding as newer data arrives
    (`_one_year_read`). A refusal offers one the plan reading it is admitted
    with, `None` where no year is (`_one_year_offered`).
    """

    years: Period
    follows_newest: bool
    until: int | None
    one_year: OneYearRead | None
    dataset_id: str

    @property
    def several(self) -> bool:
        """Whether they are, or newer data makes them, more than one year."""
        return self.years.end > self.years.start or self.follows_newest

    def phrase(self) -> str:
        """The years as a refusal names them."""
        start, end = self.years.start, self.years.end
        stops = "" if self.until is None else f", as far as {self.until},"
        if self.follows_newest:
            return f"the years from {start} to the newest{stops}"
        return f"{start}-{end}"


def _years_reduced(
    requirement: EvidenceRequirement, shapes_by_id: Mapping[str, FrameShape]
) -> YearsHeld | None:
    """The years `requirement` turns into one figure per entity; `None` when it reduces none.

    A step that combines the years it reads (`_licence_for`) reduces the
    years its input holds (`_years_held`) when they are several
    (`YearsHeld.several`). An `aggregate` that groups by `period`, and a
    `rank` that is `per_period` or states no `reduction`, keep them.
    """
    if _licence_for(requirement) is None:
        return None
    assert isinstance(requirement, AggregateRequirement | RankRequirement)
    held = _years_held(requirement, shapes_by_id[requirement.input_id])
    return held if held is not None and held.several else None


def _years_held(
    requirement: AggregateRequirement | RankRequirement, input_shape: FrameShape
) -> YearsHeld | None:
    """The years `requirement` reads per entity from `input_shape`; `None` when it holds no series of years.

    The input holds years per entity when its rows are labelled with the
    years they were observed in (`TimeLabel.OBSERVATION`). Every
    `planner.shapes` transform keeps that reading: a step that collapses the
    years labels its rows a `SPAN`, one that picks a year per entity labels
    them `PICKED`, each two-sided verb takes the narrower label of its sides
    and the years both hold, and a filter or a window that leaves one year
    narrows `period` to it. The years are those `_years_read` finds.
    """
    if input_shape.time is not TimeLabel.OBSERVATION:
        return None
    return _years_read(requirement, input_shape)


def _years_read(
    requirement: AggregateRequirement | RankRequirement, input_shape: FrameShape
) -> YearsHeld | None:
    """The years of `input_shape` `requirement` reads; `None` when no dataset requirement bounds them.

    They are the input's `period` and its `drift` once a rank's own window
    is applied (`shapes.windowed`). Each `planner.shapes` transform states
    how newer data moves them: a dataset retrieved with no end year adds
    each newer year until a fixed bound stops it. They stay one year when
    the last year is fixed there, or when the first year and the last both
    follow the anchor, which a window does with both ends at one offset. A
    first year that follows the anchor ahead of an end open to the data
    keeps every later year a series reports (`PeriodDrift.widens`).
    """
    window: Period | None = None
    if (
        isinstance(requirement, RankRequirement)
        and requirement.window_start is not None
        and requirement.window_end is not None
    ):
        window = Period(start=requirement.window_start, end=requirement.window_end)
    rolling = requirement.rolling if isinstance(requirement, RankRequirement) else None
    years, drift = windowed(input_shape, window, rolling)
    if years is None:
        return None
    return YearsHeld(
        years=years,
        follows_newest=drift.widens,
        until=drift.until if drift.widens else None,
        one_year=_one_year_read(years, drift),
        dataset_id=input_shape.roots[0],
    )


def _one_year_read(years: Period, drift: PeriodDrift) -> OneYearRead:
    """The one year of `years`, moving as `drift` says, that a step can read and keep holding.

    It is the last year held today. A last year that newer data moves
    follows the anchor at the offset of the bound that sets it, or at 0
    where it is open to each series' newest, which at offset 0 the anchor
    marks for half the series at least. A fixed last year, and one a fixed
    bound stops (`until`), give a fixed year, since a year on the anchor
    over years a fixed bound caps runs out (`planner.shapes.anchor_year_stays`).
    """
    if not anchor_year_stays(drift):
        return OneYearRead(year=years.end, offset=None)
    if drift.end is EndDrift.DATA:
        return OneYearRead(year=years.end, offset=0)
    return OneYearRead(year=years.end, offset=drift.end_offset)


def _upstream(
    requirement_id: str, requirements_by_id: Mapping[str, EvidenceRequirement]
) -> set[str]:
    """`requirement_id` and every requirement its result is computed from."""
    seen: set[str] = set()
    pending = [requirement_id]
    while pending:
        current = pending.pop()
        if current not in seen:
            seen.add(current)
            pending.extend(inputs_of(requirements_by_id[current]))
    return seen


def _operand_roles(
    requirement: EvidenceRequirement, shapes_by_id: Mapping[str, FrameShape]
) -> set[tuple[str | None, str]]:
    """The role dimensions `requirement` operates over by name, each as its role and dataset requirement.

    A rank orders the dimension its `axis` selects, the lone one when it
    states none. A share_of or quotient_of states `axis` to give one figure
    per member of the numerator's dimension it names. A filter narrows the
    dimensions its `EntityIn` predicates name, and a predicate naming no
    role counts only over an input with one dimension, since its names may
    sit in any of several. An `EntityIn` names only what the requirements
    state (`unknown_names`), so it reaches an open dimension only where a
    named one shares its role phrase. A `PlaceLevel` keeps one level of the place
    dimension. Each is resolved against its input's shape the way the
    shape transform resolves it (`planner.shapes.roles_named`).
    """
    selected: list[RoleDimension] = []
    if isinstance(requirement, RankRequirement):
        selected.append(_selected_role(shapes_by_id[requirement.input_id].roles, requirement.axis))
    elif isinstance(requirement, ShareOfRequirement | QuotientRequirement):
        if requirement.axis is not None:
            selected.extend(
                roles_named(shapes_by_id[requirement.numerator.input_id].roles, requirement.axis)
            )
    elif isinstance(requirement, FilterRequirement):
        roles = shapes_by_id[requirement.input_id].roles
        for predicate in requirement.predicates:
            if isinstance(predicate, EntityIn):
                if predicate.role is not None:
                    selected.extend(roles_named(roles, predicate.role))
                elif len(roles) == 1:
                    selected.extend(roles)
            elif isinstance(predicate, PlaceLevel):
                if predicate.role is not None:
                    selected.extend(roles_named(roles, predicate.role))
                else:
                    selected.extend(role for role in roles if role.place)
    return {(role.role, role.requirement_id) for role in selected}


def _patched[T](
    current: tuple[T, ...],
    amended: Sequence[T],
    removed: Sequence[str],
    *,
    key: str,
) -> tuple[T, ...]:
    """Apply `amended` and `removed` to `current`, keyed on `key` in that order.

    An item in `amended` whose key `current` already holds replaces the item in
    place, so a patch that changes one field of one decision does not reorder
    the plan. One whose key is new is appended. Removal is applied after the
    replaces and additions, and a removal naming a key the patch has just
    added removes that item too. A removal naming no key at all is refused. A
    slot or id that names nothing is a mistake to report back, not a silently
    ignored line in the patch.
    """
    order = [getattr(item, key) for item in current]
    by_key: dict[str, T] = {getattr(item, key): item for item in current}
    for item in amended:
        name = getattr(item, key)
        if name not in by_key:
            order.append(name)
        by_key[name] = item
    unknown = sorted(set(removed) - set(by_key))
    if unknown:
        raise ValueError(f"every removed {key} must name one the plan states; unknown: {unknown!r}")
    dropped = set(removed)
    return tuple(by_key[name] for name in order if name not in dropped)


def _patched_accounts(
    plan: Plan,
    *,
    answered: Sequence[AskAnswer],
    declined: Sequence[Decline],
    same_asks: bool,
) -> dict[str, tuple[Any, ...]]:
    """Apply a patch's answers and declines to `plan`'s.

    An ask the patch answers or declines is accounted for by the patch alone:
    whatever answered or declined it before is dropped. A decline naming no
    ask replaces the one with its code, as `_patched` keys it. Unless
    `same_asks`, the patch accounts for every ask: `plan`'s answers and the
    declines naming an ask are numbered against a list that no longer holds.
    """
    restated = {answer.ask for answer in answered} | {
        decline.ask_ref for decline in declined if decline.ask_ref is not None
    }
    if not same_asks:
        restated |= {answer.ask for answer in plan.answered} | {
            decline.ask_ref for decline in plan.declined if decline.ask_ref is not None
        }
    return {
        "answered": _patched(
            tuple(answer for answer in plan.answered if answer.ask not in restated),
            answered,
            (),
            key="ask",
        ),
        "declined": _patched(
            tuple(decline for decline in plan.declined if decline.ask_ref not in restated),
            declined,
            (),
            key="key",
        ),
    }


def _reject_repeats(names: Iterable[str], *, what: str) -> None:
    seen: set[str] = set()
    repeated = sorted({name for name in names if name in seen or seen.add(name)})
    if repeated:
        raise ValueError(f"every {what} must be unique; repeated: {repeated!r}")


class SessionState(Protocol):
    """What the agents need of session state: read a key, write a key."""

    def get(self, key: str, default: Any = None, /) -> Any: ...

    def __setitem__(self, key: str, value: Any) -> None: ...


def period_window(period: Period, *, end_open: bool = False) -> dict[str, object]:
    """`period` as an identity's `requested_window`.

    With `end_open` the range states `start_year` alone, the form apis
    ends at the current year at every mint
    (`datasets_api.adapter.window_to_identity` writes no `end_year` key
    either), so the definition reaches years published after the plan.
    """
    if end_open:
        return open_window(period.start)
    return {"kind": "range", "start_year": period.start, "end_year": period.end}


def open_window(start_year: int) -> dict[str, object]:
    """The `requested_window` from `start_year` with no end year."""
    return {"kind": "range", "start_year": start_year}


def retrieves_period(
    window: object, requirement: DatasetRequirement, *, first_year: int | None
) -> bool:
    """Whether `window`, an identity's `requested_window`, is one `requirement` is retrieved over.

    A closed requirement is retrieved over its `period` exactly. One stating
    `period_end_open` is retrieved over a range with no end year, from
    `first_year` where it reads its dataset's whole history, the year the
    plan's split reads it from (`HistorySplit.whole_history`), and from its
    period's start otherwise (`submission.service._window_pins`).
    """
    if not requirement.period_end_open:
        return window == period_window(requirement.period)
    start = requirement.period.start if first_year is None else first_year
    return window == open_window(start)


class FullHistory(NamedTuple):
    """The first year a dataset holds, and the rows of every year it holds when Search counted them."""

    first_year: int
    rows: int | None


def _whole_year(value: object) -> int | None:
    """`value` as a year when it is one: an integer, or the text of one."""
    if isinstance(value, bool) or not isinstance(value, int | str):
        return None
    try:
        return int(value)
    except ValueError:
        return None


def full_history(block: Mapping[str, Any]) -> FullHistory | None:
    """What Search's answer `block` states about every year its dataset holds.

    Only an answer for a definition over no window (`requested_window` of
    kind `all`) states it: any other answer reads the years inside the
    window it was asked for. The first year is `resolved.download.window`'s,
    the first year of the points a mint writes, or else
    `resolved.stat_vars.data_date_range`'s, as `harness.tool_contracts`
    reads the `window` the planner is shown. The rows are
    `resolved.download.rows`, which counts through any projection years the
    definition holds, so they bound the rows of the same years retrieved up
    to the current one. `None` when the answer is over a window or states no
    first year.
    """
    definition = block.get("definition")
    identity = definition.get("identity") if isinstance(definition, Mapping) else None
    requested = identity.get("requested_window") if isinstance(identity, Mapping) else None
    if not isinstance(requested, Mapping) or requested.get("kind") != "all":
        return None
    resolved = block.get("resolved")
    if not isinstance(resolved, Mapping):
        return None
    download = resolved.get("download")
    download = download if isinstance(download, Mapping) else {}
    window = download.get("window")
    first_year = _whole_year(window.get("start_year")) if isinstance(window, Mapping) else None
    if first_year is None:
        stat_vars = resolved.get("stat_vars")
        dates = stat_vars.get("data_date_range") if isinstance(stat_vars, Mapping) else None
        first_year = _whole_year(dates.get("start")) if isinstance(dates, Mapping) else None
    if first_year is None:
        return None
    rows = download.get("rows")
    counted = isinstance(rows, int) and not isinstance(rows, bool)
    return FullHistory(first_year=first_year, rows=rows if counted else None)


class RecordedDefinition(Protocol):
    """A dataset's definition as a search in the conversation recorded it (`recorded_identity`).

    `identity` is the identity document its spec_id hashes from, and
    `groups` the sets a role's places were asked for as.
    """

    @property
    def identity(self) -> Mapping[str, Any]: ...

    @property
    def groups(self) -> tuple[ResolvedGroup, ...]: ...


class WindowedPin(StrictModel):
    """A pinned dataset whose identity is on record, copied over a requirement's window.

    `spec_id` and `identity` are the recorded identity of `windowed_from`
    over that window (`data_mcp.definition_writes.windowed_identity`), the
    ones `get_data`'s `scope.window` would mint (`windowed_pin`). The window
    is the requirement's period, or a range with no end year for a
    requirement stating `period_end_open`. `groups` are the sets its places
    were asked for as, posted again beside the copy.
    """

    spec_id: SpecId
    identity: dict[str, Any]
    windowed_from: SpecId
    resolved_sources: tuple[ResolvedSourceMention, ...] = Field(
        default=(), exclude_if=lambda mentions: not mentions
    )
    groups: tuple[ResolvedGroup, ...] = Field(default=(), exclude_if=lambda groups: not groups)


def full_histories(state: SessionState) -> dict[str, FullHistory | None]:
    """What `FULL_HISTORIES_STATE_KEY` holds, by spec_id; empty when unset.

    A spec_id held with `None` was read, and its answer stated no first year.
    """
    stored = state.get(FULL_HISTORIES_STATE_KEY) or {}
    return {
        spec_id: None if raw is None else FullHistory(raw["first_year"], raw["rows"])
        for spec_id, raw in stored.items()
    }


def save_full_history(state: SessionState, spec_id: str, history: FullHistory | None) -> None:
    """Record `history` as what a read of the dataset `spec_id` over no window stated."""
    stored = dict(state.get(FULL_HISTORIES_STATE_KEY) or {})
    stored[spec_id] = None if history is None else history._asdict()
    state[FULL_HISTORIES_STATE_KEY] = stored


def shown_identity(spec_id: str, source: PinSource) -> Mapping[str, Any] | None:
    """The identity the block the planner was shown for `spec_id` carries, when it hashes to it."""
    block = source.spec_block(spec_id)
    definition = block.get("definition") if block is not None else None
    identity = definition.get("identity") if isinstance(definition, Mapping) else None
    if not isinstance(identity, Mapping):
        return None
    try:
        hashed = spec_id_for(canonicalize(identity))
    except rfc8785.CanonicalizationError:
        return None
    return identity if hashed == spec_id else None


def windowed_pin(
    spec_id: str, recorded: RecordedDefinition, window: Mapping[str, Any]
) -> WindowedPin:
    """The dataset `spec_id`, whose definition is `recorded`, over `window` in place of its own years."""
    copy_id, copied = windowed_identity(recorded.identity, window)
    return WindowedPin(
        spec_id=copy_id,
        identity=copied,
        windowed_from=spec_id,
        groups=recorded.groups,
        resolved_sources=getattr(recorded, "resolved_sources", ()),
    )


def windowed_pins(state: SessionState) -> dict[str, WindowedPin]:
    """The copies `WINDOWED_PINS_STATE_KEY` holds, by spec_id; empty when unset.

    A copy stored in a shape `WindowedPin` rejects, one a release with other
    fields wrote, is left out, and the omission is logged. A requirement
    pinned to it is then discovered through `search`.
    """
    pins: dict[str, WindowedPin] = {}
    errors: list[ValidationError] = []
    for raw in state.get(WINDOWED_PINS_STATE_KEY) or []:
        try:
            pin = WindowedPin.model_validate(raw)
        except ValidationError as exc:
            errors.append(exc)
        else:
            pins[pin.spec_id] = pin
    if errors:
        warn_left_out(logger, WINDOWED_PINS_STATE_KEY, errors, entries=len(errors))
    return pins


def save_windowed_pins(state: SessionState, pins: Iterable[WindowedPin]) -> None:
    """Record `pins` as the copies the recorded plan pins."""
    state[WINDOWED_PINS_STATE_KEY] = [pin.model_dump(mode="json") for pin in pins]


# The current conversation's lookup of the spec block an earlier turn stored
# for a spec_id (`harness.studio_state.ConversationState.earlier_block`), set
# by `harness.studio` for each segment. A turn's session state holds no
# earlier turn's block, so a pin or a fetch of one reads it here.
current_earlier_block: ContextVar[Callable[[str], Mapping[str, Any] | None] | None] = ContextVar(
    "earlier_block", default=None
)


def earlier_block(spec_id: str) -> Mapping[str, Any] | None:
    """The spec block an earlier turn of the current conversation stored for
    `spec_id`; `None` outside a conversation or when no earlier turn stored
    one."""
    lookup = current_earlier_block.get()
    return lookup(spec_id) if lookup is not None else None


# The current conversation's lookup of the definition a search in it
# recorded for a spec_id, this turn or an earlier one
# (`harness.studio_state.ConversationState.recorded_identity`), set by
# `harness.studio` for each segment beside `current_earlier_block`. Every
# identity it returns hashes to its spec_id.
current_recorded_identity: ContextVar[Callable[[str], RecordedDefinition | None] | None] = (
    ContextVar("recorded_identity", default=None)
)


def recorded_identity(spec_id: str) -> RecordedDefinition | None:
    """The definition a search in the current conversation recorded for
    `spec_id`; `None` outside a conversation or when none recorded one."""
    lookup = current_recorded_identity.get()
    return lookup(spec_id) if lookup is not None else None


@contextmanager
def definitions_on_record(definitions: Mapping[str, RecordedDefinition]) -> Iterator[None]:
    """Put `definitions`, by spec_id, on record inside the block (`recorded_identity`).

    Each is looked up before what the conversation recorded. A plan card
    switch (`harness.graph.switch`) puts the datasets Search compiled for it
    on record while the plan is revised, since the conversation records a
    search only once the node that made it yields.
    """
    outer = current_recorded_identity.get()

    def lookup(spec_id: str) -> RecordedDefinition | None:
        found = definitions.get(spec_id)
        if found is None and outer is not None:
            return outer(spec_id)
        return found

    token = current_recorded_identity.set(lookup)
    try:
        yield
    finally:
        current_recorded_identity.reset(token)


class _ShownDefinition(NamedTuple):
    """The definition of a dataset the planner was shown, as its block carries it,
    with the sets its places were asked for as."""

    identity: Mapping[str, Any]
    groups: tuple[ResolvedGroup, ...] = ()
    resolved_sources: tuple[ResolvedSourceMention, ...] = ()


def identity_on_record(
    pin: str,
    *,
    shown: Mapping[str, PinSource],
    end_open: bool,
    definition_lookup: Callable[[str], RecordedDefinition | None] | None = None,
) -> RecordedDefinition | None:
    """The definition on record for `pin`; `None` when none is.

    The definition is the one a search in the conversation recorded
    (`recorded_identity`), else the one `definition_lookup` answers:
    `submission.service.admit` and the translator pass the discovery log's
    (`translator.discovery.DiscoveryLog.recorded_definition`), which holds
    each source the translator offered. With `end_open` it is otherwise the
    one the block the planner was shown carries (`shown_identity`). Admission
    asks for it only for a requirement stating `period_end_open`: the
    translator can find a closed requirement's pin by searching its stated
    years, and no search finds a definition with no end year. The translator
    always asks for it, since a pin compiled in code during the request (a
    scope choice switched on the plan card, `harness.graph.switch`, or an
    input put on its calculation's basis, `planner.agent.compile_on_basis`)
    is held there and nowhere a search recorded it. A shown block that is
    not readable as a source (`compiler.sources.source_from_spec_block`)
    puts none on record.
    """
    recorded: RecordedDefinition | None = recorded_identity(pin)
    if recorded is None and definition_lookup is not None:
        recorded = definition_lookup(pin)
    if recorded is not None or not end_open or pin not in shown:
        return recorded
    identity = shown_identity(pin, shown[pin])
    block = shown[pin].spec_block(pin)
    if identity is None or block is None:
        return None
    try:
        source = source_from_spec_block(block)
    except ValueError:
        return None
    return _ShownDefinition(identity, source.groups, source.resolved_sources)


# The seconds a discovery call made now may take inside a planner entry. That is
# what remains of the request's wall-clock bound (`harness.turn_budget.TurnBudget`).
# Set by `harness.planner_seam` for each planner run inside a request. Unset
# otherwise, where a call takes its capability row's own timeout.
current_discovery_time_left: ContextVar[Callable[[], float] | None] = ContextVar(
    "discovery_time_left", default=None
)


def discovery_time_left() -> float | None:
    """The seconds a discovery call made now may take; `None` outside a planner entry."""
    time_left = current_discovery_time_left.get()
    return time_left() if time_left is not None else None


class PinSource(StrictModel):
    """A dataset the planner was shown this entry, as a pin of it is honoured.

    `block` is the spec block this turn's search returned for it, `None` when
    none was carried forward. `warnings` are the warning codes of the search
    part that returned it. `settled` says the search settled on it: a
    `definite` part, or a candidate or nearest series the orchestrator
    fetched (`chosen`); a pin of a settled source's measure states the
    measure decision `propose_plan` writes (`submission.service.measure_choices`).
    `earlier_turn` says an earlier turn of the conversation
    read it and no search this turn returned it; its block is then not held
    here but read through `earlier_block` (`spec_block`). `other_count_of`
    is the spec_id of the source that offered this one as another way of
    counting its flow, when only a `fetch` of it settled it
    (`harness.turn_records.SearchReading.other_count_of`): its block then
    names its own measure, which search did not elect. A source offered as
    the same count in other prices has none: its block elects the count
    search elected for the source that offered it. `asked_roles` are the
    flow parties the search part that settled on it asked for, each a
    `planner.parties.PARTY_ROLES` word, empty when it named none.
    """

    provenance_ref: str | None = None
    provenance_account: str | None = None
    block: dict[str, Any] | None = None
    warnings: tuple[str, ...] = ()
    settled: bool = False
    earlier_turn: bool = False
    other_count_of: str | None = None
    asked_roles: tuple[str, ...] = Field(default=(), exclude_if=lambda roles: not roles)

    def spec_block(self, spec_id: str) -> Mapping[str, Any] | None:
        """The spec block a pin of `spec_id`, this source, reuses."""
        return earlier_block(spec_id) if self.earlier_turn else self.block


def pin_sources(state: SessionState) -> dict[str, PinSource]:
    """The datasets `PIN_SOURCES_STATE_KEY` holds, by spec_id in the order shown; empty when unset."""
    stored = state.get(PIN_SOURCES_STATE_KEY) or {}
    return {spec_id: PinSource.model_validate(raw) for spec_id, raw in stored.items()}


def save_pin_sources(state: SessionState, sources: Mapping[str, PinSource]) -> None:
    """Record `sources` as the datasets the planner was shown this entry."""
    state[PIN_SOURCES_STATE_KEY] = {
        spec_id: source.model_dump(mode="json") for spec_id, source in sources.items()
    }


def dataset_of(identity: Mapping[str, Any]) -> dict[str, Any]:
    """`identity` without its years: what a dataset and every copy of it over another window share."""
    return {key: value for key, value in identity.items() if key != "requested_window"}


def load_asks(state: SessionState) -> tuple[str, ...]:
    """Return the asks session `state` holds (`ASKS_STATE_KEY`); empty when none are."""
    return tuple(state.get(ASKS_STATE_KEY) or ())


def load_ask_breakdowns(state: SessionState) -> tuple[AskBreakdown, ...]:
    """Return the ask breakdowns session `state` holds (`ASK_BREAKDOWNS_STATE_KEY`); empty when none are."""
    return tuple(
        AskBreakdown.model_validate(raw) for raw in state.get(ASK_BREAKDOWNS_STATE_KEY) or ()
    )


def load_asks_across_years(state: SessionState) -> tuple[int, ...]:
    """Return the ask numbers session `state` holds across years (`ASKS_ACROSS_YEARS_STATE_KEY`); empty when none are."""
    return tuple(state.get(ASKS_ACROSS_YEARS_STATE_KEY) or ())


def stored_plan(raw: str | None, *, state_key: str) -> Plan | None:
    """The plan `raw` holds, the text stored under `state_key`; `None` when it holds none.

    A plan stored in a shape `Plan` rejects, one a release with other fields
    wrote, is no plan, and the omission is logged. Its revision cannot be
    computed, so the answer to its approval card names nothing, and its
    workflow cannot run: the question is planned again. A caller that owns
    the slot tells the two `None`s apart by `raw` and clears it.
    """
    if not raw:
        return None
    try:
        return Plan.model_validate_json(raw)
    except ValidationError as exc:
        warn_left_out(logger, state_key, [exc])
        return None


def load_plan(state: SessionState) -> Plan | None:
    """Return the plan held in session `state`, or `None` when none is (`stored_plan`)."""
    return stored_plan(state.get(PLAN_STATE_KEY), state_key=PLAN_STATE_KEY)


def save_plan(state: SessionState, plan: Plan) -> None:
    """Record `plan` as the one session `state` holds."""
    state[PLAN_STATE_KEY] = plan.model_dump_json()


def load_plan_rounds(state: SessionState) -> int:
    """Return how many plans the planner has proposed in its current entry; 0 when none."""
    raw = state.get(PLAN_ROUNDS_STATE_KEY)
    return int(raw) if raw else 0


def save_plan_rounds(state: SessionState, rounds: int) -> None:
    """Record `rounds` as the number of plans the planner has proposed in its current entry."""
    state[PLAN_ROUNDS_STATE_KEY] = rounds


def amendment_entry(state: SessionState) -> bool:
    """Whether the planner's current entry opened with the user's amendment (`AMENDMENT_ENTRY_STATE_KEY`)."""
    return bool(state.get(AMENDMENT_ENTRY_STATE_KEY))


def save_amendment_entry(state: SessionState, amended: bool) -> None:
    """Record whether the planner's current entry opened with the user's amendment of the plan."""
    state[AMENDMENT_ENTRY_STATE_KEY] = amended


class DecisionPick(StrictModel):
    """The option `chosen` the user picked on the plan card for the planner
    decision under `slot`, exactly as the decision's `options` states it.

    `origin_chosen` and `origin_fixes` are that decision as the card showed
    it when the user picked: every proposal of the replan is held against
    them (`submission.service._picked_decision`), so a later proposal cannot
    return the old plan under the picked label.
    """

    slot: str
    chosen: str
    origin_chosen: str
    origin_fixes: DecisionFixes


def decision_pick(state: SessionState) -> DecisionPick | None:
    """The pick on the plan card that the planner's current entry holds
    (`DECISION_PICK_STATE_KEY`), or `None`."""
    return pick_of(state.get(DECISION_PICK_STATE_KEY))


def pick_of(raw: object) -> DecisionPick | None:
    """The `DecisionPick` a pick stored as its JSON dump states, or `None`
    for anything else."""
    if not isinstance(raw, dict):
        return None
    try:
        return DecisionPick.model_validate(raw)
    except ValidationError:
        return None


def save_decision_pick(state: SessionState, pick: DecisionPick | None) -> None:
    """Record the pick the planner's current entry holds, or clear it with `None`."""
    state[DECISION_PICK_STATE_KEY] = None if pick is None else pick.model_dump(mode="json")


def load_unsupported(state: SessionState) -> frozenset[UnsupportedOperation]:
    """Return every `unsupported` answer's operation in the planner's current entry."""
    return frozenset(UnsupportedOperation(code) for code in state.get(UNSUPPORTED_STATE_KEY) or ())


def record_unsupported(state: SessionState, operation: UnsupportedOperation) -> None:
    """Add `operation` to the `unsupported` answers of the planner's current entry."""
    recorded = load_unsupported(state) | {operation}
    state[UNSUPPORTED_STATE_KEY] = sorted(code.value for code in recorded)


def load_optional_declines(state: SessionState) -> frozenset[UnsupportedOperation]:
    """Return every decline the planner's current entry earned without requiring it."""
    return frozenset(
        UnsupportedOperation(code) for code in state.get(OPTIONAL_DECLINES_STATE_KEY) or ()
    )


def load_terminal_declines(state: SessionState) -> frozenset[UnsupportedOperation]:
    """Return the optional declines that may close this planner entry on their own."""
    return frozenset(
        UnsupportedOperation(code) for code in state.get(TERMINAL_DECLINES_STATE_KEY) or ()
    )


def record_optional_decline(
    state: SessionState, operation: UnsupportedOperation, *, terminal: bool = False
) -> None:
    """Permit `operation` optionally, and with `terminal` allow it to close the entry alone."""
    recorded = load_optional_declines(state) | {operation}
    state[OPTIONAL_DECLINES_STATE_KEY] = sorted(code.value for code in recorded)
    if terminal:
        state[TERMINAL_DECLINES_STATE_KEY] = sorted(
            code.value for code in load_terminal_declines(state) | {operation}
        )


def load_reduction_refused_asks(state: SessionState) -> frozenset[int]:
    """Return every ask a `period_reduction_unstated` refusal named in the planner's current entry."""
    return frozenset(state.get(REDUCTION_REFUSED_ASKS_STATE_KEY) or ())


def record_reduction_refused_asks(state: SessionState, asks: Iterable[int]) -> None:
    """Add `asks` to those a `period_reduction_unstated` refusal named in the planner's current entry."""
    state[REDUCTION_REFUSED_ASKS_STATE_KEY] = sorted(load_reduction_refused_asks(state) | set(asks))


def clear_unsupported(state: SessionState) -> None:
    """Forget the `unsupported` answers, optional declines and refused asks, as a new planner entry starts."""
    state[UNSUPPORTED_STATE_KEY] = []
    state[OPTIONAL_DECLINES_STATE_KEY] = []
    state[TERMINAL_DECLINES_STATE_KEY] = []
    state[REDUCTION_REFUSED_ASKS_STATE_KEY] = []
