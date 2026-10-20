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
import json
import logging
from collections.abc import Callable, Iterable, Mapping, Sequence
from contextvars import ContextVar
from datetime import datetime
from typing import Annotated, Any, Literal, NamedTuple, Protocol, get_args

import rfc8785
from common.hashed_fields import HASHED_WHEN_SET, revision_content
from data_mcp.definition_store import canonicalize, spec_id_for
from data_mcp.definition_writes import windowed_identity
from pydantic import Field, ValidationError, field_validator, model_validator

from planner.operations import Translation
from planner.parties import (
    PARTY_ROLES,
    Party,
    PartyRole,
    across_party,
    party_among,
    role_party,
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
    Period,
    PeriodThreshold,
    PlaceLevel,
    Predicate,
    RankAmong,
    RankDirection,
    RankLimit,
    Reduction,
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
    FrameShape,
    RoleDimension,
    ShapeVerb,
    aggregated,
    compared,
    filtered,
    grown,
    joined,
    place_role,
    quotiented,
    ranked,
    retrieved,
    role_named,
    roles_named,
    shared,
    summed_sides,
    windowed_years,
    year_filter,
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

# The session-state slot holding the `UnsupportedOperation` of every decline
# earned in the planner's current entry that the recorded plan may make but
# need not: a translation that refers a requirement back as unmeetable
# (`translator.tool._earned_decline`) earns one. It holds for the rest of the
# entry, whatever later plans change, and no plan owes it. Cleared and kept
# with `UNSUPPORTED_STATE_KEY`.
OPTIONAL_DECLINES_STATE_KEY = "planner:optional_declines"

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
# `plan_start` node writes it beside the brief on every plan call, and clears
# it when the call lists none. `harness.planner_seam` lists the asks in the
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
    `propose_plan` sets `source_spec_id` on the measure decision it writes
    for a pin (`submission.service._measure_decisions`) and moves it with the
    pin (`submission.service._repinned`). Every requirement citing a decision
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
        min_length=1,
        max_length=MAX_GROUP_BY,
        description="The group_by columns this decision fixes, when it fixes them.",
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


class Decision(_Choice):
    """One methodological choice the plan makes explicit."""

    fixes: DecisionFixes | None = Field(
        default=None,
        description=(
            "The requirement fields this decision settles; unset when it is a rationale only."
        ),
    )


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
        """Reject a requirement naming the same entity and role more than once."""
        keys = [(entity.name, entity.role) for entity in self.entities]
        seen: set[tuple[str, str | None]] = set()
        repeated = sorted(
            {
                f"{name}/{role}" if role else name
                for name, role in keys
                if (name, role) in seen or seen.add((name, role))
            }
        )
        if repeated:
            raise ValueError(
                f"every entity must be unique by name and role; repeated: {repeated!r}"
            )
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


def names_no_place(requirement: DatasetRequirement) -> bool:
    """Whether `requirement` names no place: its `across` states the place party.

    Its retrieval holds every place with data at every level, or the
    countries where apis cannot list them all, so a rank, a total or a
    share over it needs a `PlaceLevel` filter first (`planner.shapes`).
    """
    return any(across_party(word) == "place" for word in requirement.across)


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
        min_length=1,
        max_length=MAX_GROUP_BY,
        description=(
            "Which columns to group by before reducing: period, entity (the roles of the "
            "entities named, plus a breakdown's codes when the input has them; an `across` "
            "party is summed), breakdown (its codes alone), or two of them."
        ),
    )


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
        default=None, description="The year this rank's window starts, when it is windowed."
    )
    window_end: Year | None = Field(
        default=None, description="The year this rank's window ends, when it is windowed."
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
            and self.window_end <= self.window_start
        ):
            raise ValueError(
                f"rank window must end after it starts; got {self.window_start}-{self.window_end}"
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
    has no defined share.
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

    @model_validator(mode="after")
    def _sides_differ(self) -> ShareOfRequirement:
        if (self.numerator.input_id, self.numerator.entity) == (
            self.denominator.input_id,
            self.denominator.entity,
        ):
            raise ValueError("a share_of's numerator and denominator must differ")
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


# analysis-mcp inner-joins on `time_period` and whatever role columns the two
# sources share. The compiler writes those keys.
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
    def _rolling_thresholds_share_a_newest_year(self) -> FilterRequirement:
        """Reject rolling thresholds whose years stand for different newest years.

        Raises:
            RollingPlanError: `RollingIssue.YEARS_DISAGREE`, naming this filter.
        """
        implied = sorted(
            {
                predicate.rolling.latest_implied_by(predicate.year)
                for predicate in self.predicates
                if isinstance(predicate, PeriodThreshold) and predicate.rolling is not None
            }
        )
        if len(implied) > 1:
            raise RollingPlanError(
                RollingIssue.YEARS_DISAGREE,
                f"{self.id!r}: its rolling thresholds stand for different newest years "
                f"({implied!r}); state the years their offsets give under one newest year",
                requirement_ids=[self.id],
            )
        return self


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


def _find_cycle(requirements_by_id: dict[str, EvidenceRequirement]) -> tuple[str, ...] | None:
    """Return one cycle of requirement ids, if the requirements do not form a DAG.

    A depth-first search with a coloring scheme: white is unvisited, gray is
    on the current path, black is fully explored. Finding a gray node again
    while walking `inputs_of` means the path back to it is a cycle.
    """
    white, gray, black = 0, 1, 2
    color: dict[str, int] = dict.fromkeys(requirements_by_id, white)
    path: list[str] = []

    def visit(requirement_id: str) -> tuple[str, ...] | None:
        color[requirement_id] = gray
        path.append(requirement_id)
        for input_id in inputs_of(requirements_by_id[requirement_id]):
            if color[input_id] == gray:
                return tuple(path[path.index(input_id) :])
            if color[input_id] == white:
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
    """The parties one ask the brief lists breaks its answer down by.

    `by` holds one word for each side of the flow: one ask is one output,
    whose flow has one giving and one receiving party, so two words for one
    side (`planner.parties.words_for_one_party`) are two outputs and are
    rejected.
    """

    ask: int = Field(ge=0, lt=MAX_PLAN_ASKS, description="The ask's number in the brief.")
    by: tuple[PartyRole, ...] = Field(
        min_length=1,
        max_length=len(PARTY_ROLES),
        description="Each party the answer lists one by one, as `search` takes `open_roles`.",
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


def roles_not_kept(
    breakdowns: Sequence[AskBreakdown],
    answered: Sequence[AskAnswer],
    requirements: Iterable[EvidenceRequirement],
) -> dict[int, tuple[PartyRole, ...]]:
    """Each ask in `breakdowns` whose answer drops a party it breaks down by, with those parties.

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
    if not breakdowns:
        return {}
    shapes = frame_shapes(requirements)
    outputs = {answer.ask: answer.outputs for answer in answered}
    dropped: dict[int, tuple[PartyRole, ...]] = {}
    for breakdown in breakdowns:
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


def _check_asks_keep_their_parties(
    asks: Sequence[str],
    breakdowns: Sequence[AskBreakdown],
    answered: Sequence[AskAnswer],
    declined: Sequence[Decline],
    requirements: Sequence[EvidenceRequirement],
) -> None:
    """Reject a plan whose answer to an ask drops a party the ask breaks down by.

    Each breakdown names an ask the brief lists, once. An answered ask whose
    answer drops a party, or keeps its parties only in separate outputs
    (`roles_not_kept`), needs a `BREAKDOWN_NOT_KEPT`
    decline naming it, since a decline under another code names another part
    of the ask. Without one the plan raises `AskRoleDroppedError`.
    """
    numbers = [breakdown.ask for breakdown in breakdowns]
    if unlisted := sorted(set(numbers) - set(range(len(asks)))):
        raise ValueError(f"the brief lists {len(asks)} asks; no breakdown of ask {unlisted!r}")
    _reject_repeats([str(number) for number in numbers], what="ask breakdown")
    exempt = {
        decline.ask_ref
        for decline in declined
        if decline.code is UnsupportedOperation.BREAKDOWN_NOT_KEPT
    }
    answer_numbers = {answer.ask for answer in answered}
    dropped = {
        number: roles
        for number, roles in roles_not_kept(breakdowns, answered, requirements).items()
        if number in answer_numbers and number not in exempt
    }
    if dropped:
        together = {
            breakdown.ask: breakdown.by
            for breakdown in breakdowns
            if breakdown.ask in dropped and len(breakdown.by) > 1
        }
        raise AskRoleDroppedError(asks=asks, dropped=dropped, together=together)


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


def _check_requirement_graph(
    requirements_by_id: dict[str, EvidenceRequirement], *, answer_mode: AnswerMode
) -> None:
    """Reject requirements that do not agree with each other.

    Every input a requirement reads names a requirement the plan states, the
    requirements form a DAG, nothing reads a comparison or a join -- both are
    terminal, and a calculation over either is `UnsupportedPlanError`, the one
    rule here that refuses what the system cannot do rather than a malformed
    draft -- and every requirement's output shape is legal
    under `planner.shapes.rejects`: a side's entity or a rank axis names a
    role the shape it reads actually carries, a growth window fits that
    shape's own period, and a rank without a reduction leaves one value per
    entity to rank on. A period that follows the newest data rests only on
    dataset requirements stating `period_end_open`
    (`_check_rolling_reads_open_periods`).

    Under `AnswerMode.SERIES`, a dataset requirement left unread by every
    other requirement is legal: it is the answer. Under `AnswerMode.COMPUTED`,
    the same requirement is rejected -- a retrieved series carries a
    reference and statistics, not the figure a computed answer asks for.
    """
    requirements = tuple(requirements_by_id.values())
    read_ids = {input_id for requirement in requirements for input_id in inputs_of(requirement)}
    unknown_inputs = sorted(read_ids - set(requirements_by_id))
    if unknown_inputs:
        raise ValueError(
            f"evidence requirements read requirements this plan does not state: {unknown_inputs!r}"
        )
    cycle = _find_cycle(requirements_by_id)
    if cycle is not None:
        raise ValueError(f"evidence requirements do not form a DAG; cycle: {list(cycle)!r}")
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
        unread_datasets = sorted(
            r.id for r in requirements if is_dataset(r) and r.id not in read_ids
        )
        if unread_datasets:
            raise ValueError(
                "answer_mode is computed, so a retrieved series carries a reference and "
                "statistics, not the figure the question asks for; unread dataset "
                f"requirements: {unread_datasets!r}"
            )
    _build_shapes(requirements_by_id)
    _check_rolling_reads_open_periods(requirements_by_id)


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


def rolling_readers(
    requirements_by_id: Mapping[str, EvidenceRequirement], dataset_id: str
) -> list[EvidenceRequirement]:
    """The requirements stating a period that follows the newest data, among those resting on `dataset_id`."""
    return [
        requirement
        for requirement in requirements_by_id.values()
        if rolls(requirement) and dataset_id in dataset_roots(requirements_by_id, requirement.id)
    ]


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


def frame_shapes(requirements: Iterable[EvidenceRequirement]) -> dict[str, FrameShape]:
    """Return each requirement's output shape, by id, for a plan that already validated."""
    return _build_shapes({requirement.id: requirement for requirement in requirements})


def _build_shapes(requirements_by_id: dict[str, EvidenceRequirement]) -> dict[str, FrameShape]:
    """Walk every requirement in topological order, rejecting an illegal shape as it is found.

    Each requirement's output shape is computed from its inputs' -- already
    built, since the walk is topological -- through the one transform
    `planner.shapes` states for its kind, after `planner.shapes.rejects`
    confirms the input may feed it. Nothing here skips a check for want of a
    single upstream dataset requirement to walk to: a role dimension always
    names exactly one, by construction of every transform.
    """
    shapes_by_id: dict[str, FrameShape] = {}
    for requirement_id in _topological_order(requirements_by_id):
        requirement = requirements_by_id[requirement_id]
        shapes_by_id[requirement_id] = _shape_of(requirement, shapes_by_id, requirements_by_id)
    return shapes_by_id


def _topological_order(requirements_by_id: dict[str, EvidenceRequirement]) -> list[str]:
    """Return every requirement id ordered so each comes after everything `inputs_of` names.

    A post-order depth-first walk. Assumes the requirements already form a
    DAG -- `_check_requirement_graph` calls `_find_cycle` before this runs.
    """
    order: list[str] = []
    visited: set[str] = set()

    def visit(requirement_id: str) -> None:
        if requirement_id in visited:
            return
        visited.add(requirement_id)
        for input_id in inputs_of(requirements_by_id[requirement_id]):
            visit(input_id)
        order.append(requirement_id)

    for requirement_id in requirements_by_id:
        visit(requirement_id)
    return order


def _dataset_shape(requirement: DatasetRequirement) -> FrameShape:
    """The shape a fresh retrieval carries: one role dimension per role phrase the entities
    state, then one open dimension per `across` word, then the breakdown's.

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
                role=word, requirement_id=requirement.id, open=True, place=place_role(word)
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
        period=requirement.period,
        measures=measures,
        place_typed=names_no_place(requirement),
    )


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
    )


def _growth_shape(requirement: GrowthRequirement, input_shape: FrameShape) -> FrameShape:
    reason = shape_rejects(ShapeVerb.GROWTH, input_shape, change=requirement.change)
    if reason is not None:
        raise ValueError(f"{requirement.id!r}: {reason}")
    if input_shape.period is not None and (
        requirement.window_start < input_shape.period.start
        or requirement.window_end > input_shape.period.end
    ):
        raise ValueError(
            f"{requirement.id!r}'s growth window ({requirement.window_start}-"
            f"{requirement.window_end}) falls outside its input's period "
            f"({input_shape.period.start}-{input_shape.period.end})"
        )
    window = Period(start=requirement.window_start, end=requirement.window_end)
    return grown(input_shape, change=requirement.change, window=window)


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
    reduction_reason = _rank_reduction_reason(
        requirement, input_shape, years=windowed_years(input_shape.period, window)
    )
    if reduction_reason is not None:
        raise ValueError(reduction_reason)
    return ranked(
        input_shape,
        axis=requirement.axis,
        window=window,
        per_period=requirement.per_period,
        limit=requirement.limit,
    )


def _rank_reduction_reason(
    requirement: RankRequirement, input_shape: FrameShape, *, years: Period | None
) -> str | None:
    """Return why `requirement` needs a reduction it does not state, or `None` when it has enough.

    A rank orders entities by one value each. An axis over an input carrying
    more than one role always needs a reduction, in either mode, to collapse
    the other role's rows to one value per axis entity per (period, in
    per-period mode). Short of that, `per_period=True` ranks each period on
    its own and never needs a reduction just to collapse years together;
    `per_period=False` needs one unless its input already holds one row per
    entity, or `years`, the years of its input its window keeps
    (`shapes.windowed_years`), are one: a 2020-2025 window over 2015-2020
    ranks each entity's 2020 row.
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
    if years is not None and years.end > years.start:
        return (
            f"{requirement.id!r} ranks entities over {years.start}-{years.end} with no "
            "reduction, so no entity has a single value to rank on; state a reduction"
        )
    return None


def _filter_shape(requirement: FilterRequirement, input_shape: FrameShape) -> FrameShape:
    place_level = keeps_one_level(requirement.predicates)
    reason = shape_rejects(
        ShapeVerb.FILTER,
        input_shape,
        place_level=place_level,
        period_threshold=states_period_threshold(requirement.predicates),
    )
    if reason is not None:
        raise ValueError(f"{requirement.id!r}: {reason}")
    narrowed_roles = tuple(p.role for p in requirement.predicates if isinstance(p, EntityIn))
    years = year_filter(input_shape.period, requirement.predicates)
    if isinstance(years, str):
        raise ValueError(f"{requirement.id!r}: {years}")
    return filtered(
        input_shape,
        narrowed_roles=narrowed_roles,
        period=years.period,
        pins_one_year=years.leaves_one_year,
        keeps_one_level=place_level,
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
) -> None:
    """Reject `side` narrowed to an entity none of its shape's role dimensions' requirements name."""
    if side.entity is None:
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


def _share_of_shape(
    requirement: ShareOfRequirement,
    shapes_by_id: dict[str, FrameShape],
    requirements_by_id: dict[str, EvidenceRequirement],
) -> FrameShape:
    numerator_shape = shapes_by_id[requirement.numerator.input_id]
    denominator_shape = shapes_by_id[requirement.denominator.input_id]
    _check_side_entity_is_named(
        requirement.id, "numerator", requirement.numerator, numerator_shape, requirements_by_id
    )
    _check_side_entity_is_named(
        requirement.id,
        "denominator",
        requirement.denominator,
        denominator_shape,
        requirements_by_id,
    )
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
    _check_side_entity_is_named(
        requirement.id, "numerator", requirement.numerator, numerator_shape, requirements_by_id
    )
    _check_side_entity_is_named(
        requirement.id,
        "denominator",
        requirement.denominator,
        denominator_shape,
        requirements_by_id,
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
) -> FrameShape:
    """Return the output shape `requirement` produces, raising on the first illegal input."""
    if isinstance(requirement, DatasetRequirement):
        return _dataset_shape(requirement)
    if isinstance(requirement, AggregateRequirement):
        return _aggregate_shape(requirement, shapes_by_id[requirement.input_id])
    if isinstance(requirement, GrowthRequirement):
        return _growth_shape(requirement, shapes_by_id[requirement.input_id])
    if isinstance(requirement, RankRequirement):
        return _rank_shape(requirement, shapes_by_id[requirement.input_id])
    if isinstance(requirement, FilterRequirement):
        return _filter_shape(requirement, shapes_by_id[requirement.input_id])
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


def _stated(requirement: EvidenceRequirement, field: str) -> Any:
    """The value `requirement` states for `field`, a field its kind declares; `None` when empty."""
    value = getattr(requirement, field)
    return None if value == () else value


def _same(first: Any, second: Any) -> bool:
    """Whether two stated values agree, tuples (`entities`, `group_by`, `predicates`) as sets."""
    if isinstance(first, tuple) and isinstance(second, tuple):
        return set(first) == set(second)
    return first == second


# The `DecisionFixes` field no requirement is held to. Code writes
# `source_spec_id` on the decision a pin states and moves it with the pin
# (`submission.service._measure_decisions`, `_repinned`); a draft cannot name it.
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
    decisions: tuple[Decision, ...] = Field(
        default=(),
        max_length=MAX_PLAN_DECISIONS,
        description=(
            f"The methodological choices this plan makes, at most "
            f"{MAX_PLAN_DECISIONS}; a plan names its parts by `slot`, so no two decisions "
            "share one. Every decision has to be cited by at least one evidence "
            "requirement."
        ),
    )
    evidence_requirements: tuple[EvidenceRequirement, ...] = Field(
        default=(),
        max_length=MAX_PLAN_REQUIREMENTS,
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
    def _record_fits_the_size_limit(self) -> Plan:
        """Reject a plan whose model-authored part serialises past `MAX_PLAN_RECORD_CHARS`.

        The field limits bound each string and collection on its own, and a
        plan that fills many of them at once is still far past what the
        planner's brief can carry. The message names the measured size and the
        parts that take the most of it, so the model can shorten those.
        """
        record = self.model_dump(mode="json", include=_MODEL_AUTHORED)
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
        citation is unambiguous.
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
        (`_check_asks_keep_their_parties`).
        """
        slots = [decision.slot for decision in self.decisions]
        _reject_repeats(slots, what="decision slot")
        _reject_repeats([decline.key for decline in self.declined], what="decline")
        _reject_repeats(
            [requirement.id for requirement in self.evidence_requirements],
            what="evidence requirement id",
        )
        cited = {
            decision_id
            for requirement in self.evidence_requirements
            for decision_id in requirement.decision_ids
        }
        unknown = sorted(cited - set(slots))
        if unknown:
            raise ValueError(
                f"evidence requirements cite decisions the plan does not make: {unknown!r}"
            )
        if self.evidence_requirements:
            uncited = sorted(set(slots) - cited)
            if uncited:
                raise ValueError(f"no evidence requirement cites the decisions {uncited!r}")
        _check_requirement_graph(
            {r.id: r for r in self.evidence_requirements}, answer_mode=self.answer_mode
        )
        decisions_by_slot = {decision.slot: decision for decision in self.decisions}
        _check_requirements_honour_their_decisions(self.evidence_requirements, decisions_by_slot)
        _check_population_claims_are_cited(self.evidence_requirements, decisions_by_slot)
        _check_asks_accounted(self.asks, self.answered, self.declined, self.evidence_requirements)
        _check_asks_keep_their_parties(
            self.asks,
            self.ask_breakdowns,
            self.answered,
            self.declined,
            self.evidence_requirements,
        )
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
        answered: Sequence[AskAnswer] = (),
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
        draft does.

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
            answered: The answers this patch adds or replaces, keyed by
                `ask`. One replaces whatever accounted for its ask before,
                a decline included.

        Returns:
            A new `Plan`; `self` is unchanged.

        Raises:
            ValueError: If a removal names a slot or id this plan does not
                state.
        """
        breakdowns = self.ask_breakdowns if asks is None else tuple(ask_breakdowns)
        asks = self.asks if asks is None else tuple(asks)
        return Plan.model_validate(
            {
                **self.model_dump(),
                "question": self.question if question is None else question,
                "answer_mode": self.answer_mode if answer_mode is None else answer_mode,
                "decisions": _patched(self.decisions, decisions, removed_decisions, key="slot"),
                "evidence_requirements": _patched(
                    self.evidence_requirements,
                    evidence_requirements,
                    removed_evidence_requirements,
                    key="id",
                ),
                **_patched_accounts(
                    self, answered=answered, declined=declined, same_asks=asks == self.asks
                ),
                "asks": asks,
                "ask_breakdowns": breakdowns,
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
    keyed on it, a filter naming its role or naming none over an input
    carrying that role alone), or, like `among='population'`
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
    draw. Requires a plan whose requirements already validated.
    """
    requirements_by_id = {requirement.id: requirement for requirement in plan.evidence_requirements}
    decisions = {decision.slot: decision for decision in plan.decisions}
    shapes = _build_shapes(requirements_by_id)
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
            gaps.append(ResultScopeGap(output_id, ask, role.role, dataset.id))
    return gaps


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
    sit in any of several. A `PlaceLevel` keeps one level of the place
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


def retrieves_period(window: object, requirement: DatasetRequirement) -> bool:
    """Whether `window`, an identity's `requested_window`, is one `requirement` is retrieved over.

    A closed requirement is retrieved over its `period` exactly. One stating
    `period_end_open` is retrieved over a range with no end year that
    starts no later than its period (`planner.agent._window_pins` starts it
    at the period's start or at the first year the dataset holds).
    """
    if not requirement.period_end_open:
        return window == period_window(requirement.period)
    if not isinstance(window, Mapping) or window.get("kind") != "range" or "end_year" in window:
        return False
    start = window.get("start_year")
    return isinstance(start, int) and start <= requirement.period.start


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


def record_optional_decline(state: SessionState, operation: UnsupportedOperation) -> None:
    """Add `operation` to the declines the planner's current entry earned without requiring."""
    recorded = load_optional_declines(state) | {operation}
    state[OPTIONAL_DECLINES_STATE_KEY] = sorted(code.value for code in recorded)


def clear_unsupported(state: SessionState) -> None:
    """Forget the `unsupported` answers and optional declines, as a new planner entry starts."""
    state[UNSUPPORTED_STATE_KEY] = []
    state[OPTIONAL_DECLINES_STATE_KEY] = []
