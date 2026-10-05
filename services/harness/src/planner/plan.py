"""The plan object: method specification, control surface, and record in one.

A `Plan` states the question, each `Decision` the planner made (or proposed)
with the alternatives it considered and who resolved it, and the
`EvidenceRequirement`s those decisions fix. A dataset requirement is evidence
for a calculation the plan states -- the series a retrieved result rests on,
stated in fields code can check a retrieved frame against -- and every
calculation is itself a typed requirement (aggregate, growth, comparison,
rank, share of, join, filter), citing the decisions that fixed it, if any
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

import hashlib
import json
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from contextvars import ContextVar
from typing import Annotated, Any, Literal, Protocol

import rfc8785
from common.hashed_fields import HASHED_WHEN_SET, revision_content
from data_mcp.definition_writes import windowed_identity
from pydantic import Field, field_validator, model_validator

from planner.operations import Translation
from planner.parties import PARTY_ROLES, across_party, role_party, singular
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
    Period,
    PeriodThreshold,
    PlaceLevel,
    Predicate,
    RankAmong,
    RankDirection,
    RankLimit,
    Reduction,
    SpecId,
    StatVarReading,
    StrictModel,
    TaggedModel,
    TitlelessEnum,
    Year,
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
    ranked,
    retrieved,
    shared,
    summed_sides,
    windowed_years,
    year_filter,
)
from planner.shapes import rejects as shape_rejects
from planner.workflow import SelectedSource, Workflow

# The session-state slot holding the current `Plan`. One key, because a plan is
# filled in stages by more than one agent and each stage reads the last.
PLAN_STATE_KEY = "planner:plan"

# The session-state slot counting how many plans the planner has proposed in its
# current entry (see `MAX_PLAN_ROUNDS`). Kept beside the plan rather than on it:
# a count on `Plan` would enter `plan_revision`, and every re-proposal would
# change the very hash it counts.
PLAN_ROUNDS_STATE_KEY = "planner:rounds"

# The session-state slot holding the `UnsupportedOperation` of every
# `unsupported` answer `propose_plan` or `translate` gave in the planner's
# current entry, each of which the recorded plan has to decline. Cleared with
# `PLAN_ROUNDS_STATE_KEY` when an entry starts, so an ask-the-user resume
# inside one entry keeps it.
UNSUPPORTED_STATE_KEY = "planner:unsupported"

# The session-state slot holding a `PinSource` for every spec_id the
# planner was shown this entry, keyed by spec_id in the order it was shown
# (`pin_sources`). `harness.planner_seam` writes it: on the orchestrator
# engine from the search readings its first message renders, one entry per
# spec_id they name; on the classic engine from the resolve envelope, one
# entry for the envelope's own `spec_id` when the planner was shown it as a
# pin (`harness.planner_seam._exposed_spec_id`), none otherwise.
# `propose_plan` (`planner.agent`) writes the citing `Decision` for a pin of
# a settled entry, and `translator.discovery.discover_source` builds a pinned
# requirement's source from an entry's block with no `search` call.
PIN_SOURCES_STATE_KEY = "planner:pin_sources"

# The session-state slot the host writes a seeded turn's seeds to, as JSON:
# one entry per Search result a Studio handoff carries (`TurnRequest.seed_spec`),
# as `harness.seeds.seeds_slot` compacts it, a `harness.wire.Seed` or the
# refusal its step fails with. `harness.graph`'s `seed` node reads each entry
# and clears the slot. Unset on a turn with no seeds.
SEEDS_STATE_KEY = "seeds"

# The session-state slot holding a `SeededSource` for every dataset the user
# handed over from Search whose identity `harness.graph`'s `seed` node
# resolved again through Search this turn, and for every copy of one over
# another window that the plan `planner.agent.propose_plan` last recorded pins
# (`seeded_sources`). Only those two write it, so it names exactly the user's
# own datasets:
# `planner.agent.propose_plan` counts each spec_id as the user's own election,
# and `translator.discovery.discover_source` takes a dataset requirement's
# source straight from the one its `source_spec_id` names, with no `search`
# call. It stores each seed whole and each copy as its seed's spec_id and its
# window, which `seeded_sources` rebuilds into the copy (`windowed_copy`).
# Unset on a turn the seeds did not start.
SEEDED_SOURCES_STATE_KEY = "planner:seeded_sources"

# The session-state slot holding a `SeededSource` for every dataset the turn
# dropped, each seed and each copy of it over another window
# (`dropped_seeds`). `harness.graph`'s `execute` node writes it when a seed's
# StatVars have left its dataset and the turn replans from the other seeds;
# the user is then told the plan leaves that dataset out. For the rest of the
# turn `planner.agent.propose_plan` refuses a plan that pins one, and
# `translator.discovery.discover_source` takes no source from a search that
# answers with one (`names_dropped_seed`). It is stored as
# `SEEDED_SOURCES_STATE_KEY` is. Unset on a turn that dropped none.
DROPPED_SEEDS_STATE_KEY = "planner:dropped_seeds"

# The session-state slot holding the outputs the user asked for, as a JSON
# list of strings numbered by position, at most `MAX_PLAN_ASKS` of at most
# `MAX_ASK_CHARS` characters each (`load_asks`). The orchestrator engine's
# `plan_start` node writes it beside the brief on every plan call, and clears
# it when the call lists none; the classic engine never writes it. `harness.planner_seam` lists the asks in the planner's first message,
# and `planner.agent.propose_plan` copies them onto every plan it records, so
# `Plan` refuses one that does not account for each exactly once. Unset, a
# plan carries no asks and no ask is accounted for.
ASKS_STATE_KEY = "orchestrator:asks"

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
MAX_SOURCE_CHARS = 200
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
# times the first and twice the second. `translation`, `workflow` and
# `approved_revision` are the compiler's and `approve`'s, and do not count, so a
# compiled plan re-validates however large its workflow.
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


class Entity(StrictModel):
    """Who or where a requirement is about, in the words of the question.

    `role` is one of `PARTY_ROLES` in what a model writes: `propose_plan`
    refuses any other word. A plan read back from session state is not held
    to it, so one recorded with an older role word still loads.
    """

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
            "This entity's side of a flow: 'donor' or 'recipient' for aid, 'creditor' or "
            "'debtor' for lending and debt, 'exporter' or 'importer' for trade. Unset for a "
            "place a measure is about, and unless the question or measure identifies it."
        ),
        json_schema_extra={"enum": list(PARTY_ROLES)},
    )


class Unit(StrictModel):
    """The unit an acceptable result is stated in.

    Stated in the frame grammar's own `currency`, `prices` and `base_year`,
    `current` or `constant`, so a retrieved frame's unit compares to a
    requirement field by field.
    """

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
    """The requirement fields a `Decision` fixes, with their values, so code can compare them.

    `settle_decisions` writes it from the field names a `DecisionDraft`
    lists, reading each value off the requirements that cite the decision;
    code that pins a source on the model's behalf writes it directly. Each
    requirement citing the decision is checked against the fields set here,
    where that requirement's own class declares them, so a requirement that
    contradicts the decision is refused.
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
    across: Annotated[str | None, HASHED_WHEN_SET] = Field(
        default=None,
        min_length=1,
        max_length=MAX_ENTITY_ROLE_CHARS,
        description="The party whose every member this decision fixes a requirement to cover, when it fixes one.",
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
    source a search settled on (`planner.agent._synthesize_source_decisions`).
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


# The name of each `DecisionFixes` field, as a `DecisionDraft` lists the
# fields it fixes.
FixedField = Literal[
    "period",
    "unit",
    "entities",
    "across",
    "measure",
    "source_spec_id",
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


class DecisionDraft(_Choice):
    """One methodological choice, as `propose_plan` takes it: the fields it fixes, by name.

    `settle_decisions` builds the `Decision` from it, reading each named
    field's value off the requirements that cite it.
    """

    fixes: tuple[FixedField, ...] | None = Field(
        default=None,
        max_length=len(DecisionFixes.model_fields),
        description=(
            "The requirement fields this decision settles, by name, whenever it settles "
            "what a requirement carries: the period, the unit, the entities, the measure, "
            "the pinned source, or a calculation field a derived requirement reads. Every "
            "requirement citing it states the same value for each field named. Empty when "
            "it is a rationale only."
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
        description="What this requirement retrieves, in the words of the question.",
    )
    entities: tuple[Entity, ...] = Field(
        default=(),
        max_length=MAX_ENTITIES,
        description="Who or where this requirement is about, by name.",
    )
    across: Annotated[str | None, HASHED_WHEN_SET] = Field(
        default=None,
        min_length=1,
        max_length=MAX_ENTITY_ROLE_CHARS,
        description=(
            "The party whose every member this requirement covers, as the party word alone, "
            "such as 'donor', 'recipient' or 'country', when the question ranks them or its "
            "answer is their total, and names none of them; no stand-in entity such as "
            "'World' is named for them, and no entity in the party it covers. A narrower "
            "population, such as a region, an income group or a donor type, names its "
            "members in `entities` instead. A world figure set beside a named entity names "
            "'World' instead."
        ),
    )
    period: Period = Field(description="The years an acceptable result has to cover.")
    frequency: Literal["annual"] = Field(
        default="annual",
        description="How often an acceptable result is observed; annual is the only frequency this plan states.",
    )
    unit: Unit | None = Field(
        default=None,
        description=(
            "The unit and price basis an acceptable result has to be stated in, when the "
            "question asks for a currency or price basis; a source's own unit is not one the "
            "user asked for."
        ),
    )
    source: str | None = Field(
        default=None,
        min_length=1,
        max_length=MAX_SOURCE_CHARS,
        description="The source the question names for this requirement, when it names one, in the question's own words.",
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
        if role == self.across or any(entity.role == role for entity in self.entities):
            raise ValueError(
                f"breakdown {role!r} names the role its own codes take, so no entity role or "
                f"`across` may also be {role!r}"
            )
        return self

    @model_validator(mode="after")
    def _states_who_it_covers(self) -> DatasetRequirement:
        """Reject a requirement about nobody, or one naming an entity in the party `across` covers whole.

        Parties are compared by what they name, so `role='donors'` sits in
        `across='donor'`, and an entity with no role, or a place role, sits in
        `across='country'`. An `across` outside the party vocabulary is left
        for `compiler.evidence.open_axis_for` to refuse.
        """
        if self.across is None:
            if not self.entities:
                raise ValueError(
                    "a dataset requirement names at least one entity, or states in `across` "
                    "the party whose every member it covers"
                )
            return self
        party = across_party(self.across)
        every = f"across={self.across!r} covers every {singular(self.across)}"
        named_in_across = sorted(
            entity.name
            for entity in self.entities
            if entity.role is not None
            and (
                entity.role.casefold() == self.across.casefold()
                or (party is not None and role_party(entity.role) == party)
            )
        )
        if named_in_across:
            raise ValueError(
                f"{every}, so no entity may also be named in that party; named: "
                f"{named_in_across!r}. Drop them to cover every member, or drop `across` and "
                "name every entity the answer needs"
            )
        unplaced = sorted(entity.name for entity in self.entities if entity.role is None)
        if party == "place" and unplaced:
            raise ValueError(
                f"{every}, and {unplaced!r} carry no `role`, so each names one of them and "
                "leaves no population open. To rank every member, drop them; to rank them "
                "against every member, drop `across` and state the rank's among='population'"
            )
        return self


def names_no_place(requirement: DatasetRequirement) -> bool:
    """Whether `requirement` names no place: its `across` is the place party.

    Its retrieval holds every place with data at every level, or the
    countries where dfo-apis cannot list them all, so a rank, a total or a
    share over it needs a `PlaceLevel` filter first (`planner.shapes`).
    """
    return requirement.across is not None and across_party(requirement.across) == "place"


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
            "Which columns to group by before reducing: period, entity (every role), "
            "breakdown (its codes alone), or two of them."
        ),
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
            "How the two endpoint values are compared: 'relative' reads a dimensionless ratio, "
            "'absolute' reads a difference in the input's own unit."
        ),
    )

    @model_validator(mode="after")
    def _window_ends_after_it_starts(self) -> GrowthRequirement:
        if self.window_end <= self.window_start:
            raise ValueError(
                f"growth window must end after it starts; got {self.window_start}-{self.window_end}"
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
    left: RequirementSide = Field(description="The side subtracted from `right`.")
    right: RequirementSide = Field(description="The side `left` is subtracted from.")

    @model_validator(mode="after")
    def _sides_differ(self) -> ComparisonRequirement:
        if (self.left.input_id, self.left.entity) == (self.right.input_id, self.right.entity):
            raise ValueError("a comparison's left and right sides must differ")
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


DerivedRequirement = (
    AggregateRequirement
    | GrowthRequirement
    | ComparisonRequirement
    | SumOfRequirement
    | RankRequirement
    | ShareOfRequirement
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
    comparison, sum_of or share_of reads both its sides; a join reads its
    left and right id; every other derived requirement reads its one
    `input_id`.
    """
    if isinstance(requirement, DatasetRequirement):
        return ()
    if isinstance(requirement, ComparisonRequirement | SumOfRequirement):
        return (requirement.left.input_id, requirement.right.input_id)
    if isinstance(requirement, ShareOfRequirement):
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


class UnsupportedOperation(TitlelessEnum):
    """An ask the system cannot answer, as an `unsupported` answer names it.

    `propose_plan` names one in `detail.operation`; `translate` names one as
    the `kind` of a referral whose outcome is `unsupported`
    (`translator.referral.OUTCOME_OF`).
    """

    # Any calculation over a `comparison` or `join` result: both are terminal,
    # so nothing reads either.
    CALCULATION_OVER_COMBINED = "calculation_over_combined"
    # A requirement no operation the translator builds carries out.
    NO_OPERATION_COVERS = "no_operation_covers"
    # An ask search found the data cannot answer as stated.
    UNSUPPORTED_ASK = "unsupported_ask"


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
        description="The refusing answer's `detail.operation`, or its referral `kind`."
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
        description="The `id` of each requirement that answers it; no requirement reads one.",
    )


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
    not. An answer names requirements the plan states that no other
    requirement reads, since only those results reach the answer. A number
    past the asks listed is a malformed draft; a gap, two answers or two
    declines is `AsksUnaccountedError`.
    """
    answer_numbers = [answer.ask for answer in answered]
    decline_numbers = [decline.ask_ref for decline in declined if decline.ask_ref is not None]
    numbers = answer_numbers + decline_numbers
    unlisted = sorted(set(numbers) - set(range(len(asks))))
    if unlisted:
        listed = f"numbered 0 to {len(asks) - 1}" if asks else "none"
        raise ValueError(f"the brief lists {len(asks)} asks ({listed}); no ask {unlisted!r}")
    ids = {requirement.id for requirement in requirements}
    read_ids = {input_id for requirement in requirements for input_id in inputs_of(requirement)}
    outputs = {output for answer in answered for output in answer.outputs}
    if unknown := sorted(outputs - ids):
        raise ValueError(
            f"an ask is answered by requirements this plan does not state: {unknown!r}"
        )
    if read := sorted(outputs & read_ids):
        raise ValueError(
            "an ask is answered only by requirements no other requirement reads, since only "
            f"those results reach the answer; these feed another: {read!r}"
        )
    uncovered = [number for number in range(len(asks)) if number not in numbers]
    repeated = [
        number
        for number in range(len(asks))
        if answer_numbers.count(number) > 1 or decline_numbers.count(number) > 1
    ]
    if uncovered or repeated:
        raise AsksUnaccountedError(asks=asks, uncovered=uncovered, repeated=repeated)


# How a calculation reading a comparison or a join is named to the user.
_READER_PHRASES = {
    "aggregate": "a total or other summary",
    "growth": "a growth rate",
    "rank": "a ranking",
    "share_of": "a share",
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

    `SERIES` answers a question that asks to see a level or a trend: a lone
    dataset requirement, unread by anything else, is itself the answer, its
    frame carrying one series per entity for a chart to draw. `COMPUTED`
    answers a question that asks for a figure the series does not state on
    its own -- a difference, growth, share, rank or total -- so every dataset
    requirement has to feed a calculation; one left unread would leave the
    plan without the figure the question asked for.
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
    entity to rank on.

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
    state, then `across`, then the breakdown's.

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
    if requirement.across is not None:
        role_names.append(requirement.across)
    roles = tuple(RoleDimension(role=name, requirement_id=requirement.id) for name in role_names)
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
        names_no_place=names_no_place(requirement),
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
    that resolution, to read a field off the one it finds.
    """
    if axis is None:
        return roles[0]
    return next(role for role in roles if role.role == axis)


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


def _comparison_shape(
    requirement: ComparisonRequirement,
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
        reason = shape_rejects(ShapeVerb.COMPARE, shape, other=other)
        if reason is not None:
            raise ValueError(f"{requirement.id!r}: {reason}")
    return compared(left_shape, right_shape)


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
    if isinstance(requirement, ComparisonRequirement):
        return _comparison_shape(requirement, shapes_by_id, requirements_by_id)
    if isinstance(requirement, SumOfRequirement):
        return _sum_of_shape(requirement, shapes_by_id, requirements_by_id)
    assert isinstance(requirement, JoinRequirement)
    return _join_shape(requirement, shapes_by_id)


def _check_requirements_honour_their_decisions(
    requirements: Iterable[EvidenceRequirement], decisions_by_slot: dict[str, Decision]
) -> None:
    """Reject a requirement that contradicts a decision it cites, or a population claim no decision backs.

    Every requirement kind is compared against what its cited decisions fix
    -- `_check_fixes_agree` skips a field the requirement's own class does
    not declare, so comparing every kind costs nothing where a field does
    not apply. A pinned dataset requirement additionally needs its pin
    resting on a decision that fixes that same source_spec_id; a rank stating
    `among='population'` or a keyed share_of stating
    `denominator_scope='pooled'` additionally needs a decision that states
    that claim structurally -- the same treatment a pinned source gets,
    since both are a figure over entities nobody named. Every slot in
    `decision_ids` is already known to name a decision the plan makes.
    """
    for requirement in requirements:
        cited = [decisions_by_slot[slot] for slot in requirement.decision_ids]
        for decision in cited:
            _check_fixes_agree(requirement, decision)
        if isinstance(requirement, DatasetRequirement):
            _check_pin_rests_on_a_fixing_decision(requirement, cited)
        elif isinstance(requirement, RankRequirement) and requirement.among is RankAmong.POPULATION:
            _check_population_claim_is_cited(
                requirement.id, cited, field="among", value=RankAmong.POPULATION
            )
        elif (
            isinstance(requirement, ShareOfRequirement)
            and requirement.denominator_scope is DenominatorScope.POOLED
        ):
            _check_population_claim_is_cited(
                requirement.id, cited, field="denominator_scope", value=DenominatorScope.POOLED
            )


def _check_fixes_agree(requirement: EvidenceRequirement, decision: Decision) -> None:
    """Reject a field of `requirement` that differs from what `decision.fixes` states.

    A field the decision leaves unset is not compared, and neither is a
    field the requirement's own class does not declare -- a decision fixing
    `reduction`, say, has nothing to compare against a `FilterRequirement`,
    which carries no such field. `entities`, `group_by` and `predicates`
    compare as sets: the order a plan lists them in carries no meaning.
    Every other field compares by equality.
    """
    if decision.fixes is None:
        return
    model_fields = type(requirement).model_fields
    for field, fixed in decision.fixes:
        if fixed is None or field not in model_fields:
            continue
        own = getattr(requirement, field)
        agree = set(own) == set(fixed) if isinstance(fixed, tuple) else own == fixed
        if not agree:
            raise ValueError(
                f"{requirement.id!r} states {field} {own!r}, but the decision "
                f"{decision.slot!r} it cites fixes {fixed!r}"
            )


def _check_pin_rests_on_a_fixing_decision(
    requirement: DatasetRequirement, cited: Sequence[Decision]
) -> None:
    """Reject a pinned source that no cited decision states structurally.

    A `source_spec_id` is the exact definition chosen among several, so the
    choice has to be on record in a decision whose `fixes` names
    `source_spec_id` with this exact pin. The
    pin is never compared with any decision's `chosen`: the choice is "OECD
    DAC", not a hash. Who resolved that decision is not checked here:
    `resolved_by=user` is model-self-reported, and this model has no session
    to check it against. `planner.agent.propose_plan` is the one that can --
    it checks a `resolved_by=user` pin against the spec_ids the user
    actually chose answering `adk_request_input`
    (`planner.agent._elected_spec_ids`, read off the session's own event
    history) or handed over from Search (`seeded_sources`), before a plan
    carrying one ever reaches this validator.
    """
    if requirement.source_spec_id is None:
        return
    if not any(
        decision.fixes is not None and decision.fixes.source_spec_id == requirement.source_spec_id
        for decision in cited
    ):
        raise ValueError(
            f"{requirement.id!r} pins source_spec_id {requirement.source_spec_id!r} but cites no "
            "decision whose `fixes` names `source_spec_id`; a pinned source rests on a decision "
            "that names it structurally"
        )


def _check_population_claim_is_cited(
    requirement_id: str, cited: Sequence[Decision], *, field: str, value: object
) -> None:
    """Reject a population-wide claim no cited decision states structurally.

    `among='population'` and `denominator_scope='pooled'` are each a figure
    computed over entities nobody named, so this rule gives them the same
    treatment a pinned source gets: no silent default, a cited `Decision`
    whose own `fixes` states the same claim.
    """
    if not any(
        decision.fixes is not None and getattr(decision.fixes, field) == value for decision in cited
    ):
        raise ValueError(
            f"{requirement_id!r} states {field}={value!r}, a claim over entities nobody named, "
            f"but cites no decision whose `fixes` names `{field}`"
        )


class UnsettledFixError(ValueError):
    """A field a `DecisionDraft` names that the requirements citing it do not settle.

    `citing` lists the requirements that cite the decision. `stating` is
    empty when none of them states the field. When they disagree it lists
    every citing requirement whose kind declares the field, because they
    state different values or some state it and others leave it empty.
    """

    def __init__(
        self, *, slot: str, field: str, citing: Sequence[str], stating: Sequence[str]
    ) -> None:
        self.slot = slot
        self.field = field
        self.citing = tuple(citing)
        self.stating = tuple(stating)
        if not citing:
            message = (
                f"decision {slot!r} fixes {field!r}, but no requirement cites it. Cite {slot!r} "
                f"in the `decision_ids` of the requirement that states {field}, or drop "
                f"{field!r} from its `fixes`."
            )
        elif not stating:
            message = (
                f"decision {slot!r} fixes {field!r}, but no requirement citing it "
                f"({', '.join(citing)}) states {field}. State {field} on a requirement that "
                f"cites {slot!r} and has that field, or drop {field!r} from its `fixes`."
            )
        else:
            message = (
                f"decision {slot!r} fixes {field!r}, but the requirements citing it state "
                f"different values for it, or leave it empty ({', '.join(stating)}). A decision "
                "fixes one value: "
                f"drop {field!r} from its `fixes` and keep the citations, or give each value "
                "a decision of its own."
            )
        super().__init__(message)


def _stated(requirement: EvidenceRequirement, field: str) -> Any:
    """The value `requirement` states for `field`, a field its kind declares; `None` when empty."""
    value = getattr(requirement, field)
    return None if value == () else value


def _same(first: Any, second: Any) -> bool:
    """Whether two stated values agree, the way `_check_fixes_agree` compares them."""
    if isinstance(first, tuple) and isinstance(second, tuple):
        return set(first) == set(second)
    return first == second


def settle_decisions(
    drafts: Iterable[DecisionDraft], requirements: Sequence[EvidenceRequirement]
) -> list[Decision]:
    """Build each `Decision` from its draft, reading the fields it fixes off the requirements citing it.

    A field's value is the one the citing requirements state, taken from the
    first of them in `requirements` order. Every citing requirement whose
    kind declares the field has to state that value, compared as
    `_check_fixes_agree` compares, so the built `Decision` agrees with every
    requirement citing it by construction. A draft naming no field builds a
    `Decision` with no `fixes`.

    Args:
        drafts: The decisions as `propose_plan` took them.
        requirements: Every requirement the plan states once the call lands,
            carried forward or new.

    Raises:
        UnsettledFixError: If a named field is stated by no requirement citing
            the draft, or two citing requirements declaring it disagree,
            one of them leaving it empty included.
    """
    decisions: list[Decision] = []
    for draft in drafts:
        citing = [
            requirement for requirement in requirements if draft.slot in requirement.decision_ids
        ]
        fixed: dict[str, Any] = {}
        for field in dict.fromkeys(draft.fixes or ()):
            declaring = [(r.id, _stated(r, field)) for r in citing if field in type(r).model_fields]
            stated = [value for _, value in declaring if value is not None]
            if not stated or any(
                value is None or not _same(stated[0], value) for _, value in declaring
            ):
                raise UnsettledFixError(
                    slot=draft.slot,
                    field=field,
                    citing=[requirement.id for requirement in citing],
                    stating=[requirement_id for requirement_id, _ in declaring] if stated else [],
                )
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
            "every dataset requirement to feed a calculation. `propose_plan` requires the "
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
            "The parts of the question this plan does not answer because `propose_plan` "
            "answered `unsupported` for them, one per `code` and `ask_ref`."
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
    translation: Translation | None = None
    workflow: Workflow | None = None
    version: int = 1
    previous_version: int | None = None
    status: PlanStatus = PlanStatus.DRAFT
    approved_revision: str | None = None

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
        every dataset requirement feeds a calculation. Last, a dataset
        requirement has to honour the decisions it cites: no field may
        contradict what a cited decision `fixes`, and a pinned `source_spec_id`
        has to rest on a decision that fixes that same source_spec_id. Last of
        all, each ask the plan carries is answered or declined exactly once
        (`_check_asks_accounted`).
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
        _check_requirements_honour_their_decisions(
            self.evidence_requirements, {decision.slot: decision for decision in self.decisions}
        )
        _check_asks_accounted(self.asks, self.answered, self.declined, self.evidence_requirements)
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
        `approved_revision`, `translation` and `workflow`. An amendment
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
            answered: The answers this patch adds or replaces, keyed by
                `ask`. One replaces whatever accounted for its ask before,
                a decline included.

        Returns:
            A new `Plan`; `self` is unchanged.

        Raises:
            ValueError: If a removal names a slot or id this plan does not
                state.
        """
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
                "translation": None,
                "workflow": None,
                "version": self.version + 1,
                "previous_version": self.version,
                "status": PlanStatus.DRAFT,
                "approved_revision": None,
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

    def approve(self) -> Plan:
        """Return a new `Plan` marked `APPROVED`, recording what was approved.

        The harness graph's own approval step enforces the actual gate; this
        records the resulting state and the revision it applies to, which
        `start_execution` checks against.

        Raises:
            ValueError: If the plan is not awaiting approval. Approving a
                second time, or approving a plan already answered, would
                overwrite the record of the first answer.
        """
        self._require(PlanStatus.PENDING_APPROVAL, to_be="approved")
        return self.model_copy(
            update={"status": PlanStatus.APPROVED, "approved_revision": plan_revision(self)}
        )

    def reject(self) -> Plan:
        """Return a new `Plan` marked `REJECTED`.

        Nothing in this plan may run afterwards -- `planner.agent`'s
        `submit_plan` checks the status and refuses. The planner's own tools
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
    translation, workflow, version -- and leaves out `status` and
    `approved_revision`, so approving a revision does not change the value
    that was approved. A `HASHED_WHEN_SET` field enters once it holds
    something other than its default (`common.hashed_fields.revision_content`), so
    a plan saved without the field keeps its revision. Canonical JSON
    (RFC 8785) under SHA-256, the same discipline the data services use for
    their own identities.
    """
    content = revision_content(plan, exclude={"status", "approved_revision"})
    return hashlib.sha256(rfc8785.dumps(content)).hexdigest()


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


class SeededSource(StrictModel):
    """One dataset the user handed over from Search, or a copy of one over another window.

    `source` is the dataset as the plan pins it. A seed's is read from the
    Search API's own answer for the seed's identity (`harness.graph`'s
    `seed` node, `compiler.sources.source_from_spec_block`). `seen_year` is
    the year whose figures Search showed when the question named no year and
    Search chose the latest one; `None` when the question named its years or
    Search showed no such headline. `windowed_from` is the spec_id of the
    seed this source copies over another window (`planner.agent`'s
    `propose_plan`); `None` on the seed itself. A copy is `windowed_copy`
    of its seed: its spec_id, its identity, the seed's elected StatVars and
    the seed's sentence, and no data or coverage, which the translator reads
    from Search's answer for the copy (`translator.discovery`).
    """

    source: SelectedSource
    seen_year: Year | None = None
    windowed_from: SpecId | None = None


def windowed_copy(seed: SeededSource, window: Mapping[str, Any]) -> SeededSource:
    """`seed` over `window` in place of its own years, as a copy of it.

    The copy's identity and spec_id are the seed's identity over `window`
    (`data_mcp.definition_writes.windowed_identity`), the ones `get_data`'s
    `scope.window` would mint. It keeps the seed's elected StatVars and the
    seed's sentence as a name, and states nothing Search stated for the
    seed's own years.

    Raises:
        ValueError: If `seed` is a copy itself or states no identity.
    """
    if seed.windowed_from is not None or seed.source.identity is None:
        raise ValueError(f"{seed.source.spec_id!r} is not a seed with an identity to copy")
    spec_id, identity = windowed_identity(seed.source.identity, window)
    return SeededSource(
        source=SelectedSource(
            spec_id=spec_id,
            sentence=seed.source.sentence,
            has_data=None,
            stat_vars=seed.source.stat_vars,
            identity=identity,
        ),
        windowed_from=seed.source.spec_id,
    )


class _StoredCopy(StrictModel):
    """A copy as session state stores it: the seed it copies and the window it copies it over."""

    windowed_from: SpecId
    window: dict[str, Any]


class _StoredSources(StrictModel):
    """Seeded sources as session state stores them: each seed whole, each copy by reference."""

    seeds: list[SeededSource]
    copies: list[_StoredCopy]


def _store_sources(sources: Sequence[SeededSource]) -> str:
    """`sources` as session state holds them, every copy reduced to its seed and window.

    Raises:
        ValueError: If a copy's seed is not among `sources`, or a copy is not
            `windowed_copy` of its seed, so reading it back would change it.
    """
    seeds = [item for item in sources if item.windowed_from is None]
    by_spec_id = {item.source.spec_id: item for item in seeds}
    copies: list[_StoredCopy] = []
    for item in sources:
        if item.windowed_from is None:
            continue
        seed = by_spec_id.get(item.windowed_from)
        if seed is None:
            raise ValueError(
                f"the copy {item.source.spec_id!r} names the seed {item.windowed_from!r}, "
                "which the sources do not hold"
            )
        window = (item.source.identity or {}).get("requested_window")
        if not isinstance(window, dict) or windowed_copy(seed, window) != item:
            raise ValueError(
                f"{item.source.spec_id!r} is not its seed {item.windowed_from!r} over another "
                "window (`windowed_copy`)"
            )
        copies.append(_StoredCopy(windowed_from=item.windowed_from, window=window))
    stored = _StoredSources(seeds=seeds, copies=copies)
    return json.dumps(stored.model_dump(mode="json"))


def _read_sources(raw: str | None) -> list[SeededSource]:
    """The sources `_store_sources` stored in `raw`, each copy rebuilt from its seed; empty when unset.

    Raises:
        ValueError: If a copy names a seed the stored sources do not hold.
    """
    if not raw:
        return []
    stored = _StoredSources.model_validate_json(raw)
    seeds = {seed.source.spec_id: seed for seed in stored.seeds}
    copies: list[SeededSource] = []
    for copy in stored.copies:
        seed = seeds.get(copy.windowed_from)
        if seed is None:
            raise ValueError(
                f"a stored copy over {copy.window!r} names the seed {copy.windowed_from!r}, "
                "which the stored sources do not hold"
            )
        copies.append(windowed_copy(seed, copy.window))
    return [*stored.seeds, *copies]


def seeded_sources(state: SessionState) -> list[SeededSource]:
    """The datasets `SEEDED_SOURCES_STATE_KEY` holds; empty when the seeds did not start this turn.

    The seeds come first, then the copies, each rebuilt from its seed
    (`windowed_copy`).

    Raises:
        ValueError: If a copy names a seed the slot does not hold.
    """
    return _read_sources(state.get(SEEDED_SOURCES_STATE_KEY))


def save_seeded_sources(state: SessionState, sources: Sequence[SeededSource]) -> None:
    """Record `sources` as the datasets the user handed over from Search.

    Each seed is stored whole and each copy as its seed's spec_id and its
    window, so a copy costs the slot about 130 characters whatever its seed
    holds.

    Raises:
        ValueError: If two of them share a spec_id, since every reader looks a
            seeded dataset up by spec_id (`seeded_source`), so one spec_id
            names one entry; or if a copy's seed is not among them, or the
            copy is not `windowed_copy` of it.
    """
    spec_ids = [item.source.spec_id for item in sources]
    repeated = sorted({spec_id for spec_id in spec_ids if spec_ids.count(spec_id) > 1})
    if repeated:
        raise ValueError(f"seeded sources name {repeated!r} more than once")
    state[SEEDED_SOURCES_STATE_KEY] = _store_sources(sources)


def seeded_source(sources: Sequence[SeededSource], spec_id: str | None) -> SeededSource | None:
    """The one of `sources` whose spec_id is `spec_id`, a seed or a copy of one; `None` when none is."""
    return next((item for item in sources if item.source.spec_id == spec_id), None)


def seed_of(item: SeededSource) -> str:
    """The spec_id of the seed `item` is: its own, or the seed it copies over another window."""
    return item.windowed_from or item.source.spec_id


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


class PinSource(StrictModel):
    """A dataset the planner was shown this entry, as a pin of it is honoured.

    `block` is the spec block this turn's search returned for it, `None` when
    none was carried forward. `warnings` are the warning codes of the search
    part that returned it. `settled` says the search settled on it: a
    `definite` part, a fetched (`chosen`) candidate, or the resolve
    envelope's own `spec_id`. A pin of a settled dataset needs no decision
    from the planner; a pin of an unsettled one, a candidate, needs the
    planner's own. `earlier_turn` says an earlier turn of the conversation
    read it and no search this turn returned it; its block is then not held
    here but read through `earlier_block` (`spec_block`).
    """

    block: dict[str, Any] | None = None
    warnings: tuple[str, ...] = ()
    settled: bool = False
    earlier_turn: bool = False

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


def split_dropped(
    sources: Sequence[SeededSource], seeds: Collection[str]
) -> tuple[list[SeededSource], list[SeededSource]]:
    """`sources` as those a drop of `seeds` keeps and those it takes.

    A drop takes each named seed with every copy of it over another window
    (`seed_of`), and nothing else: another seed over the same dataset and
    other years is its own Search result and stays.
    """
    kept = [item for item in sources if seed_of(item) not in seeds]
    taken = [item for item in sources if seed_of(item) in seeds]
    return kept, taken


def dropped_seeds(state: SessionState) -> list[SeededSource]:
    """The datasets `DROPPED_SEEDS_STATE_KEY` holds; empty when the turn dropped none.

    The seeds come first, then the copies, each rebuilt from its seed
    (`windowed_copy`).

    Raises:
        ValueError: If a copy names a seed the slot does not hold.
    """
    return _read_sources(state.get(DROPPED_SEEDS_STATE_KEY))


def save_dropped_seeds(state: SessionState, sources: Sequence[SeededSource]) -> None:
    """Record `sources` as the datasets the turn dropped, stored as `save_seeded_sources` stores them.

    Raises:
        ValueError: If a copy's seed is not among `sources`, or the copy is
            not `windowed_copy` of it.
    """
    state[DROPPED_SEEDS_STATE_KEY] = _store_sources(sources)


def dataset_of(identity: Mapping[str, Any]) -> dict[str, Any]:
    """`identity` without its years: what a seed and every copy of it over another window share."""
    return {key: value for key, value in identity.items() if key != "requested_window"}


def names_dropped_seed(
    state: SessionState, *, spec_id: str | None, identity: Mapping[str, Any] | None = None
) -> bool:
    """Whether a dataset is one the turn dropped, which no plan may pin and no retrieval may take.

    The one rule every reader asks: `planner.agent.propose_plan` of a pin,
    with its spec_id alone, and `translator.discovery` of a source, with its
    identity too. A dataset the turn still holds as a seed or a copy of one
    (`seeded_sources`, what `split_dropped` kept) is not dropped, whatever
    else matches: a second seed over the dropped one's dataset and other
    years stays usable. Otherwise it is dropped when its spec_id is a dropped
    one, or when its identity equals a dropped one's apart from
    `requested_window`: a search for the dropped dataset over other years
    resolves the same StatVar family under another spec_id
    (`data_mcp.definition_writes.windowed_identity`).

    A pin states no identity, so a pin on the dropped dataset under a
    spec_id the turn never held passes here and is caught where its source
    is read, with its identity: the translator refers it back as
    `seed_dropped`. A held seed pinned over a dropped sibling's own window
    moves to the dropped spec_id, which `propose_plan` checks before it
    records any copy, so that spec_id is never held again.
    """
    if seeded_source(seeded_sources(state), spec_id) is not None:
        return False
    dropped = dropped_seeds(state)
    if seeded_source(dropped, spec_id) is not None:
        return True
    if identity is None:
        return False
    dataset = dataset_of(identity)
    return any(
        item.source.identity is not None and dataset_of(item.source.identity) == dataset
        for item in dropped
    )


def load_asks(state: SessionState) -> tuple[str, ...]:
    """Return the asks session `state` holds (`ASKS_STATE_KEY`); empty when none are."""
    return tuple(state.get(ASKS_STATE_KEY) or ())


def load_plan(state: SessionState) -> Plan | None:
    """Return the plan held in session `state`, or `None` when none is."""
    raw = state.get(PLAN_STATE_KEY)
    return Plan.model_validate_json(raw) if raw else None


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


def clear_unsupported(state: SessionState) -> None:
    """Forget the `unsupported` answers, as a new planner entry starts."""
    state[UNSUPPORTED_STATE_KEY] = []
