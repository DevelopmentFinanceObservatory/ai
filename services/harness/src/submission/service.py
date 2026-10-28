"""The checks a proposed plan passes before it is recorded, over any session.

`admit` runs the whole chain a `propose_plan` call runs, in order: the lists
it states (`_Patch`), which plan it means (`_resolve_intent`) and that plan
as the model wrote it (`planner.plan.authored`), the pins it
moves back from their windowed copies (`_unwindowed_pins`), a referral the
user has not answered (`_referral_unanswered`), the inputs of each
calculation put on one measure basis and provider scope (`_aligned_bases`), the decisions it
settles (`planner.plan.settle_decisions`), the measure decisions and then the
scope decisions written on the model's behalf (`_measure_decisions`,
`_scope_decisions`), the carried decisions it
contradicts (`_contradicted_carried_decisions`: a planner's is dropped
unless another requirement still rests on it), the carried planner
decisions it leaves uncited (`_uncited_carried_decisions`), a user's
decision it drops or changes with no word from the user since the plan was
recorded (`_user_decision_refused`), a decision of its own restating
the measure a pin decides (`_measure_fixed_by_pin`), the `Plan` it
builds under the first years on record (`_first_years_on_record`), that
plan split so a dataset read from its first year serves only the steps
rolling from their start (`planner.plan.split_of`), the pins it windows
(`_window_pins`), and what a built plan may not record
(`_unrecordable`). It answers the refusal that stops the chain first, or an
`Admission` carrying the plan to record. `submit` records an admission and
answers `ok`, which is all `planner.agent.propose_plan` does with its call.
A `Proposer` names the tool the proposal came through, so every `action` a
refusal carries calls a tool the model was given.

The chain reads the session through a `SessionView`: the state slots
`planner.plan` and the translator keep, what the user answered and was asked
(read off the session's history by whatever implements the view), and the
two writes the chain makes, an `unsupported` answer and the admitted plan.
`planner.agent.ToolContextSession` is the view over an ADK tool call.

Every refusal is `planner.tool_results`' typed `problem`. Translation is the
caller's.
"""

from __future__ import annotations

import functools
import hashlib
import logging
import unicodedata
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from enum import StrEnum
from itertools import chain
from typing import Annotated, Any, NamedTuple, Protocol, cast, get_args

import rfc8785
from compiler.evidence import named_members, named_refs, open_axes_for
from compiler.sources import (
    PROVIDER_ACCOUNT,
    block_concept_label,
    block_group,
    countries_apart,
    measure_named_by_query,
    open_entity_axes_of,
    readable_groups,
    readable_source,
    slot_codes,
    stat_var_reading,
)
from data_mcp.scope import ChoiceOrigin
from data_mcp.settings import Settings as DataMcpSettings
from google.adk.workflow.utils._workflow_hitl_utils import REQUEST_INPUT_FUNCTION_CALL_NAME
from harness.plan_display import plan_words
from harness.scope_choices import (
    ANY_REPORTERS,
    BILATERAL_REPORTERS,
    MULTILATERAL_REPORTERS,
    PROVIDER_AXIS,
    PROVIDER_STREAMS,
    Option,
    PartScope,
    ProviderMoney,
    ProviderScope,
    ProviderStream,
    ScopeChoice,
    SlotBinding,
    block_dataset,
    holds_streams,
    option_axis,
    planner_axes,
    scope_money,
)
from planner.parties import (
    PARTY_ROLES,
    FlowSide,
    across_party,
    flow_side,
    party_among,
    party_role,
)
from planner.plan import (
    MAX_DECISION_OPTIONS,
    MAX_DECISION_RATIONALE_CHARS,
    MAX_DECLINES,
    MAX_GROUP_BY,
    MAX_OPTION_CHARS,
    MAX_PLAN_ASKS,
    MAX_PLAN_DECISIONS,
    MAX_PLAN_REQUIREMENTS,
    MAX_PLAN_ROUNDS,
    AggregateRequirement,
    AnswerMode,
    AskAnswer,
    AskRoleDroppedError,
    AsksUnaccountedError,
    ComparisonRequirement,
    DatasetRequirement,
    Decision,
    DecisionDraft,
    DecisionFixes,
    DecisionScope,
    Decline,
    DifferenceCaveat,
    Entity,
    EvidenceRequirement,
    FilterRequirement,
    FirstYear,
    FixDisagreement,
    FullHistory,
    OneYearRead,
    PinSource,
    Plan,
    QuotientRequirement,
    RankRequirement,
    RecordedDefinition,
    RequirementSide,
    ResolvedBy,
    ScopeOptionState,
    SessionState,
    ShareOfRequirement,
    SumOfRequirement,
    UnsupportedOperation,
    UnsupportedPlanError,
    WholeIsItsMemberError,
    WindowedPin,
    amendment_entry,
    authored,
    bound_fixes,
    check_acyclic,
    check_wholes_span_their_totals,
    cut_user_text,
    dataset_roots,
    exact_codes,
    fix_disagreements,
    fixes_changes,
    full_histories,
    full_history,
    identity_on_record,
    inputs_of,
    is_scope_choice,
    is_search_pin,
    keep_years_admits,
    licences_restated,
    load_ask_breakdowns,
    load_asks,
    load_asks_across_years,
    load_optional_declines,
    load_plan,
    load_plan_rounds,
    load_reduction_refused_asks,
    load_terminal_declines,
    load_unsupported,
    open_window,
    period_licences_held,
    period_window,
    pin_slot_moves,
    pin_sources,
    predicate_key,
    reached,
    repinned,
    requirements_over,
    roles_not_kept,
    scope_choice_slot,
    search_pin_slot,
    settle_decisions,
    split_of,
    unstated_period_reductions,
    unstated_result_scopes,
    whole_of_fault,
    windowed_pin,
    windowed_pins,
)
from planner.records import (
    MAX_IDENTIFIER_CHARS,
    AmbiguousEntityReading,
    Breakdown,
    EntityCandidate,
    EntityIn,
    EntityKey,
    GroupBy,
    Identifier,
    OpenPeriodFault,
    Reduction,
    RollingIssue,
    RollingPlanError,
    StatVarReading,
    StrictModel,
    Year,
)
from planner.shapes import PLACE_PHRASE, pooling_remedy
from planner.tool_results import Action, Outcome, ProblemCode, ok, problem
from planner.workflow import OpenEntityAxis, SelectedSource
from pydantic import BaseModel, Field, JsonValue, TypeAdapter, ValidationError, model_validator
from pydantic_core import to_jsonable_python
from translator.discovery import (
    load_log,
    requirement_open_roles,
    requirement_revision,
    search_part,
)
from translator.referral import USER_DECIDES, Referral, ReferralKind, load_referral
from translator.tool import authored_answer

# What a combination ask asks about: the `sum_of` requirement's id and the
# datasets its sides read (`_sum_datasets`).
type Combination = tuple[str, frozenset[str]]

logger = logging.getLogger(__name__)


class ReferralAnswers(NamedTuple):
    """What the user said to the asks raised after the pending referral (`SessionView.answers_since_referral`).

    `answered` says whether any of those asks has an answer. `picked` holds the
    candidate ids the answers picked, in the order they were given, so a later
    answer about the same choice comes after the one it replaces. `unanswered`
    holds every candidate id an ask offered that has no answer, skipped or
    dismissed included; an ask the tool or ADK refused offered nothing.
    `picked_ever` holds the user's standing picks from asks raised under any
    referral of the session, which a later referral does not clear: a later
    answer to an ask offering the same ids replaces them, and a message the
    user typed or an amendment of the approval card releases them all.
    """

    answered: bool
    picked: tuple[str, ...]
    unanswered: frozenset[str]
    picked_ever: tuple[str, ...]


class SessionView(Protocol):
    """The session a proposal is checked against, as the chain reads and writes it.

    `state` holds the slots `planner.plan` and the translator keep (the
    recorded plan, its rounds, pins shown and windowed, histories read,
    asks, the pending referral, the discovery log), read through their own
    loaders. The four readers answer from what the user was asked and
    answered in this session. The four writes are the chain's only side
    effects: an `unsupported` answer, an optional decline and the asks a
    `PERIOD_REDUCTION_UNSTATED` refusal names, each recorded even though the
    call is refused, and an admitted plan.
    """

    @property
    def state(self) -> SessionState:
        """The session's state slots."""
        ...

    def elected_spec_ids(self) -> frozenset[str]:
        """Every spec_id the user chose answering an ask that offered it."""
        ...

    def answers_since_referral(self) -> ReferralAnswers:
        """What the user answered to the asks raised after the pending referral."""
        ...

    def combination_picks(self) -> tuple[dict[Combination, str | None], bool]:
        """The option the user last picked for each combination asked about, and whether any answer declined to add."""
        ...

    def user_spoke_since(self, version: int) -> bool:
        """Whether the user has said anything (an answer, an amendment, a new message) since plan `version` was recorded."""
        ...

    def record_unsupported(self, operation: UnsupportedOperation) -> None:
        """Record that a proposal in this planner entry was answered `unsupported` for `operation`."""
        ...

    def record_optional_decline(self, operation: UnsupportedOperation) -> None:
        """Permit a decline of `operation` for the rest of this planner entry, without requiring one."""
        ...

    def record_reduction_refused(self, asks: Iterable[int]) -> None:
        """Record `asks` as named by a `PERIOD_REDUCTION_UNSTATED` refusal in this planner entry."""
        ...

    def record_admission(self, admission: Admission) -> None:
        """Record `admission`'s plan, its round count and its windowed pins, and clear the referral it supersedes."""
        ...


@dataclass(frozen=True)
class Proposal:
    """What one proposal states, as `propose_plan` takes its arguments.

    The lists arrive as the caller received them, unvalidated: `admit`
    validates them first (`_Patch`).
    """

    question: str | None = None
    answer_mode: AnswerMode | None = None
    decisions: Sequence[Any] = ()
    evidence_requirements: Sequence[Any] = ()
    removed_decisions: Sequence[Any] = ()
    removed_evidence_requirements: Sequence[Any] = ()
    declined: Sequence[Any] = ()
    answered: Sequence[Any] = ()
    across_years: Sequence[Any] = ()
    amends_version: int | None = None
    starts_new_plan: bool = False


# The most failing fields one `propose_plan` refusal names. A plan can fail on
# dozens at once, and the model corrects the first few before it proposes again.
_MAX_REPORTED_ERRORS = 10


@dataclass(frozen=True)
class Proposer:
    """The tool a proposal came through, as the chain's answers address it.

    `call` is the tool a refusal's `action` names to propose again.
    """

    call: str

    def action(
        self, arguments: Mapping[str, JsonValue] | None = None, needs: tuple[str, ...] = ()
    ) -> Action:
        """The call to this tool that states `arguments` and leaves `needs` to the model.

        An action stating no arguments leaves `arguments` unset.
        """
        return Action(call=self.call, arguments=arguments or None, needs=needs)


# `planner.agent.propose_plan`'s proposer.
PROPOSE_PLAN = Proposer(call="propose_plan")


@dataclass(frozen=True)
class Admission:
    """A plan the chain admitted, and what recording it writes.

    `rounds` is the proposal count the call brings the planner entry to.
    `windowed_pins` is every windowed copy the plan pins, to replace those
    on record, or `None` when neither this plan nor the session holds any,
    so nothing is written.
    """

    plan: Plan
    rounds: int
    windowed_pins: tuple[WindowedPin, ...] | None


def submit(
    view: SessionView,
    proposal: Proposal,
    *,
    question_fixed: bool,
    reads_user_words: bool,
    proposer: Proposer,
) -> dict[str, object]:
    """Check `proposal` against `view` and record it, answering as `propose_plan` does.

    Returns `ok` with `status` `"draft"`, the new `version`, the
    `previous_version` it amended (`None` on a first draft) and the
    `question` as stored, or the refusal `admit` answered, recording
    nothing but an `unsupported` answer. `proposer` is the tool `proposal`
    came through.
    """
    admitted = admit(
        view,
        proposal,
        question_fixed=question_fixed,
        reads_user_words=reads_user_words,
        proposer=proposer,
    )
    if not isinstance(admitted, Admission):
        return admitted
    view.record_admission(admitted)
    plan = admitted.plan
    # The answer carries only what the model cannot read off its own call:
    # the question as stored, and the `version` and predecessor
    # `Plan.amend` assigns, which a later patch names as `amends_version`. The decisions and requirements sent are not echoed,
    # since translation reads the plan from session state.
    return ok(
        status=plan.status.value,
        version=plan.version,
        previous_version=plan.previous_version,
        question=plan.question,
    )


def admit(  # noqa: PLR0911 -- one return per stage of the chain that can refuse
    view: SessionView,
    proposal: Proposal,
    *,
    question_fixed: bool,
    reads_user_words: bool,
    proposer: Proposer,
) -> Admission | dict[str, object]:
    """Run the chain over `proposal`, answering the plan to record or the refusal that stops it.

    `question_fixed` is `_resolve_intent`'s. `reads_user_words` says the
    planner reads the user's own words in every entry, as the standalone
    planner and `harness_pair_v1` do; under `harness.planner_seam` it reads
    them only in an entry the user's amendment opened
    (`planner.plan.amendment_entry`). Only where it reads them may
    `proposal.across_years` name a step, and each step it names takes a
    licence to combine its years (`planner.plan.period_licences_held`).
    `proposer` is the tool every refusal's `action` names. Writes nothing
    but an `unsupported` answer (`SessionView.record_unsupported`) and the
    asks a `PERIOD_REDUCTION_UNSTATED` refusal names
    (`SessionView.record_reduction_refused`).
    """
    try:
        # ADK's arg preprocessing silently falls back to raw dicts for the
        # *whole* list if any single item fails validation (e.g. a
        # Decision's `chosen` not in `options`), and skips conversion
        # entirely for a value that isn't a list at all. Re-validating here
        # turns either case into a retryable tool problem, keeping
        # unvalidated dicts out of a Plan and the exception out of the turn.
        patch = _Patch.model_validate(
            {
                "decisions": proposal.decisions,
                "evidence_requirements": proposal.evidence_requirements,
                "removed_decisions": proposal.removed_decisions,
                "removed_evidence_requirements": proposal.removed_evidence_requirements,
                "declined": proposal.declined,
                "answered": proposal.answered,
                "across_years": proposal.across_years,
            }
        )
    except ValidationError as exc:
        return _invalid_data(exc, proposer)
    reads_user_words = reads_user_words or amendment_entry(view.state)
    question = proposal.question
    answer_mode = proposal.answer_mode
    evidence_requirements = patch.evidence_requirements
    removed_decisions = patch.removed_decisions
    removed_evidence_requirements = patch.removed_evidence_requirements
    existing = load_plan(view.state)
    referral = _authored_referral(view, existing)
    resolved = _resolve_intent(
        existing,
        amends_version=proposal.amends_version,
        starts_new_plan=proposal.starts_new_plan,
        rounds_so_far=load_plan_rounds(view.state),
        question=question,
        question_fixed=question_fixed,
    )
    if isinstance(resolved, dict):
        return resolved
    previous, rounds = resolved
    if previous is not None:
        # The chain reads the plan as the model wrote it. The model is shown
        # no copy the split added, so a patch naming a copy's id names a
        # requirement of its own, and the split copies under the next free id.
        previous = authored(previous)
    held_copies = windowed_pins(view.state)
    to_origin = {copy.spec_id: copy.windowed_from for copy in held_copies.values()}
    if to_origin:
        _, evidence_requirements = _unwindowed_pins((), evidence_requirements, to_origin)
    if previous is not None and to_origin:
        unwindowed, unwindowed_requirements = _unwindowed_pins(
            previous.decisions, previous.evidence_requirements, to_origin
        )
        previous = previous.model_copy(
            update={
                "decisions": tuple(unwindowed),
                "evidence_requirements": tuple(unwindowed_requirements),
            }
        )
    pins = pin_sources(view.state)
    sources = SourceReadings.of_shown(pins)
    measures = measure_choices(
        pins,
        compiled=previous.workflow.sources if previous is not None and previous.workflow else (),
        offered=chain.from_iterable(referral.offered().values()) if referral else (),
        sources=sources,
    )
    user_settled = _user_settled_measures(measures, view, pins)
    evidence_requirements = _stated_source_given_way(evidence_requirements, referral, view)
    stated_after = _requirements_after(
        previous, evidence_requirements, removed_evidence_requirements
    )
    try:
        # Every stage below walks what each requirement reads, down to its datasets.
        check_acyclic({requirement.id: requirement for requirement in stated_after})
    except ValueError as exc:
        return _invalid_data(exc, proposer)
    realigned = _aligned_bases(
        stated_after,
        evidence_requirements,
        shown=pins,
        sources=sources,
        aligned=aligned_pins(view.state),
        searched=searched_pins(view.state),
        view=view,
        proposer=proposer,
    )
    if isinstance(realigned, dict):
        return realigned
    evidence_requirements = realigned
    # Every requirement once aligned; the decisions below change citations alone.
    aligned_after = _requirements_after(
        previous, evidence_requirements, removed_evidence_requirements
    )
    wholes = _compiled_wholes(aligned_after, aligned=aligned_pins(view.state))
    settled, evidence_requirements = _without_replaced_whole_decisions(
        settle_decisions(patch.decisions, aligned_after),
        evidence_requirements,
        carried=previous.decisions if previous is not None else (),
        after=aligned_after,
        wholes=wholes.keys(),
    )
    settled, evidence_requirements = _measure_decisions(
        settled,
        evidence_requirements,
        previous=previous,
        measures=measures,
        user_settled=user_settled,
    )
    scopes = scope_choice_decisions(pins, wholes=_whole_only_pins(aligned_after, wholes))
    settled, evidence_requirements = _scope_decisions(
        settled,
        evidence_requirements,
        previous=previous,
        scopes=scopes,
        wholes=_whole_decisions(wholes, aligned_after, shown=pins),
        calculations=_difference_decisions(
            aligned_after,
            sources=sources,
            plan_decisions=_decisions_after(previous, settled, removed=removed_decisions),
            scope_decisions=scopes,
        ),
        removed=removed_decisions,
    )
    requirements_after = _requirements_after(
        previous, evidence_requirements, removed_evidence_requirements
    )
    read = full_histories(view.state)
    definition_lookup = load_log(view.state).recorded_definition
    settled = [
        *settled,
        *_user_settled_rewrites(
            previous,
            settled,
            requirements_after,
            measures=measures,
            user_settled=user_settled,
        ),
    ]
    # Between the user's turns, their decisions stand as they made them.
    user_spoke = previous is None or view.user_spoke_since(previous.version)
    contradicted = _contradicted_carried_decisions(
        previous, settled, requirements_after, removed=removed_decisions, user_spoke=user_spoke
    )
    if contradicted:
        evidence_requirements = _without_citations(
            requirements_after,
            evidence_requirements,
            slots=frozenset(c.decision.slot for c in contradicted),
        )
        requirements_after = _requirements_after(
            previous, evidence_requirements, removed_evidence_requirements
        )
    removed_decisions = [
        *removed_decisions,
        *_uncited_carried_decisions(
            previous,
            settled,
            requirements_after,
            removed=removed_decisions,
            user_spoke=user_spoke,
        ),
    ]
    # A pending referral outranks an overridden user decision: answering it
    # may change the patch.
    blocked = (
        _across_years_unread(
            patch.across_years,
            requirements=requirements_after,
            previous=previous,
            reads_user_words=reads_user_words,
        )
        or _referral_unanswered(existing, referral, view)
        or (
            None
            if user_spoke
            else _user_decision_refused(
                previous, settled, requirements_after, removed=removed_decisions
            )
        )
        or _shared_decision_contradicted(contradicted, proposer)
        or _measure_fixed_by_pin(
            patch.decisions,
            requirements_after,
            _decisions_after(previous, settled, removed=removed_decisions),
            measures,
            proposer,
        )
    )
    if blocked is not None:
        return blocked
    try:
        if previous is not None:
            plan = previous.amend(
                question=question,
                answer_mode=answer_mode,
                decisions=settled,
                evidence_requirements=evidence_requirements,
                removed_decisions=removed_decisions,
                removed_evidence_requirements=removed_evidence_requirements,
                declined=patch.declined,
                asks=load_asks(view.state),
                ask_breakdowns=load_ask_breakdowns(view.state),
                asks_across_years=load_asks_across_years(view.state),
                answered=patch.answered,
                first_years=_first_years_on_record(requirements_after, shown=pins, read=read),
            )
        elif question is None or answer_mode is None:
            missing = " and ".join(
                clause
                for clause, absent in (
                    ("the `question` it answers", question is None),
                    ("the `answer_mode` it answers under", answer_mode is None),
                )
                if absent
            )
            raise ValueError(
                f"a first draft states {missing}"
                if existing is None
                else f"a `starts_new_plan` call states {missing}, for its new plan"
            )
        else:
            plan = Plan(
                question=question,
                answer_mode=answer_mode,
                decisions=settled,
                evidence_requirements=evidence_requirements,
                declined=patch.declined,
                asks=load_asks(view.state),
                ask_breakdowns=load_ask_breakdowns(view.state),
                asks_across_years=load_asks_across_years(view.state),
                answered=patch.answered,
                first_years=_first_years_on_record(requirements_after, shown=pins, read=read),
            )
        stated = plan.model_copy(
            update={
                "period_licences": period_licences_held(
                    plan, named=patch.across_years, previous=previous
                )
            }
        )
        # `Plan` validation compares the sides of a share by name; the
        # pins resolve them to graph refs, so an alias divides by itself too.
        check_wholes_span_their_totals(stated, refs=_entity_refs(view, sources))
        split = split_of(stated)
        # The recorded plan carries the copies, and `Plan` validation checks
        # they are the split of `stated` before `_window_pins` gives each pin
        # its window.
        materialized = (
            Plan.model_validate(
                {
                    **stated.model_dump(),
                    "evidence_requirements": split.requirements,
                    "rolling_copies": split.copies,
                }
            )
            if split.copies
            else stated
        )
        plan, copies = _window_pins(materialized, shown=pins, definition_lookup=definition_lookup)
    except (ValidationError, ValueError) as exc:
        # The plan-level rules: unique slots and ids, every citation naming
        # a decision, every decision cited, the length and count limits on
        # each field, a non-blank question, and the size limit on the whole. A removal naming an item
        # the plan does not state is a `ValueError` from `Plan.amend` itself,
        # and a first draft or restart with no `question`/`answer_mode` is
        # one raised above.
        cause = _unsupported_cause(exc)
        if cause is not None:
            view.record_unsupported(cause.operation)
        unknown = _first_year_unknown_cause(exc)
        if unknown is not None:
            exc = _first_year_refusal(
                unknown, shown=pins, read=read, definition_lookup=definition_lookup
            )
        return _invalid_data(exc, proposer)
    blocked = _unrecordable(
        plan,
        view,
        referral=referral,
        stated=stated,
        previous=previous,
        reads_user_words=reads_user_words,
        proposer=proposer,
        sources=sources,
    )
    if blocked is not None:
        return blocked
    # The plan keeps the first years its split reads. A plan reading no
    # whole history carries no `first_years`, so they leave its
    # `plan_revision` unchanged.
    whole = set(split_of(plan).first_years_read)
    plan = plan.model_copy(
        update={
            "first_years": tuple(
                entry for entry in plan.first_years if entry.requirement_id in whole
            )
        }
    )
    # The copies this plan pins. A copy that only a replaced plan pinned is
    # let go, so however many drafts and amendments the turn makes, the
    # state holds at most one copy per dataset requirement of the recorded
    # plan.
    return Admission(
        plan=plan,
        rounds=rounds,
        windowed_pins=tuple(copies) if copies or held_copies else None,
    )


class _Patch(BaseModel):
    """The lists one `propose_plan` call states, checked before they reach a `Plan`.

    ADK hands `propose_plan` raw values wherever it could not convert them, and
    builds the tool's declaration from the signature with every `Annotated`
    limit stripped, so the count and shape of each list is enforced here. A
    removal names a slot or an id, so it is held to `Identifier`; each list
    holds no more items than a plan does.
    """

    decisions: Annotated[list[DecisionDraft], Field(max_length=MAX_PLAN_DECISIONS)]
    evidence_requirements: Annotated[
        list[EvidenceRequirement], Field(max_length=MAX_PLAN_REQUIREMENTS)
    ]
    removed_decisions: Annotated[list[Identifier], Field(max_length=MAX_PLAN_DECISIONS)]
    removed_evidence_requirements: Annotated[
        list[Identifier], Field(max_length=MAX_PLAN_REQUIREMENTS)
    ]
    declined: Annotated[list[Decline], Field(max_length=MAX_DECLINES)]
    answered: Annotated[list[AskAnswer], Field(max_length=MAX_PLAN_ASKS)]
    across_years: Annotated[list[Identifier], Field(max_length=MAX_PLAN_REQUIREMENTS)]

    @model_validator(mode="after")
    def _roles_are_party_words(self) -> _Patch:
        """Reject an entity `role` outside `PARTY_ROLES`, naming the word it stands for when one does.

        A place word is no role: a place the measure is about states none,
        and an axis names its dimension by a place word all the same
        (`planner.shapes.roles_named`).
        """
        refused = []
        for requirement in self.evidence_requirements:
            if not isinstance(requirement, DatasetRequirement):
                continue
            for entity in requirement.entities:
                if entity.role is None or entity.role in PARTY_ROLES:
                    continue
                meant = party_role(entity.role)
                advice = ""
                if meant is not None:
                    advice = f", write {meant!r}"
                elif across_party(entity.role) == "place":
                    advice = f", leave it unset: an axis names such places {PLACE_PHRASE!r}"
                refused.append(f"{requirement.id}/{entity.name}: {entity.role!r}{advice}")
        if refused:
            raise ValueError(
                f"an entity's role is one of {', '.join(PARTY_ROLES)}, or unset for a place "
                f"the measure is about; {'; '.join(refused)}"
            )
        return self


# The longest field name in an error location the refusal carries. A location names
# a field the plan declares, except where the model supplied an unknown key, which
# arrives as a name of any length.
_MAX_LOC_ELEMENT_CHARS = 80


def _cut_loc_element(element: str | int) -> str | int:
    """Return `element` with a string cut to `_MAX_LOC_ELEMENT_CHARS`, an index unchanged."""
    return cut_user_text(element, _MAX_LOC_ELEMENT_CHARS) if isinstance(element, str) else element


def _invalid_plan(message: str) -> dict[str, object]:
    """Return the `INVALID_PLAN` problem carrying `message`."""
    return problem(ProblemCode.INVALID_PLAN, message, outcome=Outcome.INVALID_INPUT)


def _invalid_data(exc: ValueError, proposer: Proposer) -> dict[str, object]:
    """Return the `INVALID_PLAN` problem for data a plan's own rules refused.

    A `ValidationError` is reported as each failing field's path, the rule it
    broke and the rule's limit, for the first `_MAX_REPORTED_ERRORS`
    fields and a count of the rest. The refused value is left out: the response lands in
    the model's next request, so echoing it would spend the space the refusal
    exists to protect and repeat it in the history on every retry. A location
    can still carry model-supplied text (an unknown key on a requirement), so
    each string in it is cut at `_MAX_LOC_ELEMENT_CHARS`. A `Decision`
    validator's message echoes `chosen` and `options`, which their own
    length caps bound. A plan rule's typed error raised outside validation
    answers as it does inside (`_rule_refused`); any other `ValueError` is a
    plan rule stated in words, so its text is the message. A rule that refused what the system cannot do
    (`UnsupportedPlanError`) is no malformed draft, so it answers `_unsupported`.
    An ask left unaccounted for (`AsksUnaccountedError`) answers
    `_asks_unaccounted`, and a party an answer drops (`AskRoleDroppedError`)
    answers `_ask_role_dropped`, and a share dividing a member by itself
    (`WholeIsItsMemberError`) answers `_whole_is_its_member`. A rolling period a rule refused
    (`RollingPlanError`) answers `_rolling_refused`, which names the rule.

    `exc.errors()` carries a `loc` tuple and, for a `model_validator` failure,
    the raised exception itself in `ctx["error"]`. Neither is JSON on its own,
    so `to_jsonable_python(..., serialize_unknown=True)` renders both as plain
    JSON (a list, a string) rather than `problem()`'s own validation rejecting
    the detail it was just handed.
    """
    if isinstance(exc, RollingPlanError):
        return _rolling_refused(exc)
    if not isinstance(exc, ValidationError):
        return _rule_refused(exc, proposer) or _invalid_plan(f"Invalid plan data: {exc}")
    cause = _unsupported_cause(exc)
    if cause is not None:
        return _unsupported(cause, proposer)
    errors = exc.errors(include_url=False, include_input=False)
    for error in errors:
        refused = _rule_refused(error.get("ctx", {}).get("error"), proposer)
        if refused is not None:
            return refused
    reported = [
        {**error, "loc": tuple(_cut_loc_element(element) for element in error["loc"])}
        for error in errors[:_MAX_REPORTED_ERRORS]
    ]
    lines = [
        f"{'.'.join(map(str, error['loc'])) or 'the plan'}: {error['msg']}" for error in reported
    ]
    if len(errors) > len(reported):
        lines.append(f"and {len(errors) - len(reported)} more")
    return problem(
        ProblemCode.INVALID_PLAN,
        f"Invalid plan data: {'; '.join(lines)}",
        outcome=Outcome.INVALID_INPUT,
        detail={"errors": to_jsonable_python(reported, serialize_unknown=True)},
    )


def _rule_refused(raised: object, proposer: Proposer) -> dict[str, object] | None:
    """The typed problem for `raised` when it is an error a plan rule raised, else `None`."""
    if isinstance(raised, AsksUnaccountedError):
        return _asks_unaccounted(raised, proposer)
    if isinstance(raised, AskRoleDroppedError):
        return _ask_role_dropped(raised)
    if isinstance(raised, WholeIsItsMemberError):
        return _whole_is_its_member(raised)
    if isinstance(raised, RollingPlanError):
        return _rolling_refused(raised)
    return None


def _rolling_refused(exc: RollingPlanError) -> dict[str, object]:
    """Return the `INVALID_PLAN` problem for a rolling period a plan rule refused.

    `detail.issue` is the rule (`planner.records.RollingIssue`),
    `detail.requirement_ids` the requirements to change, and `detail.fault`,
    under `rolling_needs_open_period`, which part of it holds
    (`planner.records.OpenPeriodFault`). `detail.spec_id` is the pinned
    dataset whose first year is missing, when that is the fault.
    """
    detail: dict[str, Any] = {
        "issue": exc.issue.value,
        "requirement_ids": list(exc.requirement_ids),
    }
    if exc.fault is not None:
        detail["fault"] = exc.fault.value
    if exc.spec_id is not None:
        detail["spec_id"] = exc.spec_id
    return problem(
        ProblemCode.INVALID_PLAN,
        f"Invalid plan data: {exc}",
        outcome=Outcome.INVALID_INPUT,
        detail=detail,
    )


def _unsupported_cause(exc: ValueError) -> UnsupportedPlanError | None:
    """The `UnsupportedPlanError` a plan rule raised under `exc`, if one did."""
    if not isinstance(exc, ValidationError):
        return None
    for error in exc.errors(include_url=False, include_input=False):
        cause = error.get("ctx", {}).get("error")
        if isinstance(cause, UnsupportedPlanError):
            return cause
    return None


def _unsupported(exc: UnsupportedPlanError, proposer: Proposer) -> dict[str, object]:
    """Return the `UNSUPPORTED` problem for a calculation no plan can state.

    The `action` carries the decline's `code` and leaves its `ask`, the user's
    own words, to the model.
    """
    code = exc.operation.value
    return problem(
        ProblemCode.OPERATION_UNSUPPORTED,
        f"Unsupported: {exc}. This is a limit of the system, not a fault in the plan, so no "
        f"reshaping makes it work. Record the part of the question that asks for it in "
        f"`declined` with code {code!r}, in the user's words and, when the brief lists asks, "
        "with the ask's number as `ask_ref`, and plan only what can still be answered, such "
        "as each series on its own; when nothing can, propose `declined` alone, with no "
        "requirements. The user is told that part was not computed.",
        outcome=Outcome.UNSUPPORTED,
        action=proposer.action({"declined": [{"code": code}]}, needs=("declined.ask",)),
        detail={"operation": code},
    )


def _asks_unaccounted(exc: AsksUnaccountedError, proposer: Proposer) -> dict[str, object]:
    """Return the `ASKS_UNACCOUNTED` problem for a plan that drops or doubles an ask.

    The `action` names each ask left unaccounted for and leaves the outputs
    that answer it to the model.
    """
    return problem(
        ProblemCode.ASKS_UNACCOUNTED,
        f"{exc}. Answer an ask in `answered` with the ids of the requirements whose results "
        "answer it, or, when `declined` permits it, "
        "decline it in `declined` with its number as `ask_ref`; an ask the plan computes in "
        "part takes one "
        "of each. A patch that answers or declines an ask replaces whatever accounted for it "
        "before, so state each once.",
        outcome=Outcome.INVALID_INPUT,
        action=proposer.action(
            {"answered": [{"ask": number} for number in exc.uncovered]},
            needs=("answered.outputs",),
        )
        if exc.uncovered
        else None,
        detail={"uncovered": list(exc.uncovered), "repeated": list(exc.repeated)},
    )


def _ask_role_dropped(exc: AskRoleDroppedError) -> dict[str, object]:
    """Return the `ASK_ROLE_DROPPED` problem for a plan whose answer drops a party its ask breaks down by.

    `detail.dropped` holds each ask's number and the parties its answer
    drops, with every party one output has to keep as `together` for an
    ask that breaks down by several.
    """
    declined = UnsupportedOperation.BREAKDOWN_NOT_KEPT.value
    joint = (
        " The parties an ask lists together are one result: one output keeps them all."
        if exc.together
        else ""
    )
    return problem(
        ProblemCode.ASK_ROLE_DROPPED,
        f"{exc}.{joint} Keep each party to the answer: name its members as `entities` with "
        "that role when the question names them; otherwise pin a source whose search reading "
        "holds it open and name it in the dataset requirement's `across`. Carry it to the "
        "output, through a rank whose `axis` is that party or no aggregate that sums it away. When "
        "the source this ask's answer pins cannot hold the party open, answer the ask "
        f"with what the plan computes and decline the breakdown in `declined` with code "
        f"{declined!r} and the ask's number as `ask_ref`.",
        outcome=Outcome.INVALID_INPUT,
        detail={
            "dropped": [
                {"ask": number, "by": list(roles)}
                | ({"together": list(exc.together[number])} if number in exc.together else {})
                for number, roles in exc.dropped.items()
            ]
        },
    )


def _whole_is_its_member(exc: WholeIsItsMemberError) -> dict[str, object]:
    """Return the `WHOLE_IS_ITS_MEMBER` problem for a share or quotient dividing a member by itself.

    Keeping the party open always passes the rule, so the refusal never
    leaves the planner without a plan it admits. A party no source holds
    open is then translation's to report, and its answer earns the decline
    of the ask when it names a code. `detail` names the step, the ask, the
    party, the denominator's dataset requirement and the entities it names.
    """
    party = exc.party
    return problem(
        ProblemCode.WHOLE_IS_ITS_MEMBER,
        f"{exc}. Keep {party} open in {exc.denominator_id!r}'s `across`, naming no {party} in "
        "its `entities`, so its total spans every one; or name there the group the user "
        f"named as the whole; or, when translation finds no source holding every {party}, "
        f"decline ask {exc.ask} in `declined` with the code its answer names and the ask's "
        "number as `ask_ref`.",
        outcome=Outcome.INVALID_INPUT,
        detail={
            "requirement_id": exc.requirement_id,
            "ask": exc.ask,
            "party": party,
            "denominator_id": exc.denominator_id,
            "named": list(exc.named),
        },
    )


def _decline_missing(plan: Plan, view: SessionView, proposer: Proposer) -> dict[str, object] | None:
    """Return the refusal for a plan that drops an ask answered `unsupported`.

    Each `unsupported` answer `propose_plan` or its translation gave in this
    planner entry (`load_unsupported`) needs a `Decline` with its operation
    as `code`, so the ask stays on the record instead of vanishing from a
    reshaped plan. An optional decline (`load_optional_declines`) is owed by
    no plan.
    """
    declined = {decline.code for decline in plan.declined}
    missing = sorted(code.value for code in load_unsupported(view.state) - declined)
    if not missing:
        return None
    return problem(
        ProblemCode.DECLINE_MISSING,
        f"An `unsupported` answer named {missing!r} earlier in this turn, and this plan does "
        "not say so. State each in `declined`, with the part of the question it leaves "
        "unanswered, in the user's words.",
        outcome=Outcome.INVALID_INPUT,
        action=proposer.action(
            {"declined": [{"code": code} for code in missing]}, needs=("declined.ask",)
        ),
        detail={"operations": list(missing)},
    )


def _decline_unearned(
    plan: Plan, previous: Plan | None, view: SessionView
) -> dict[str, object] | None:
    """Return the refusal for a plan declining what no tool answered `unsupported`.

    A decline this plan adds needs its code among the `unsupported` answers
    recorded in this planner entry (`load_unsupported`): `propose_plan`'s or
    its translation's, or one a replan's run-time refusal stands for
    (`harness.planner_seam._earned_by_refusal`); or among its optional
    declines (`load_optional_declines`), which hold `no_operation_covers`
    in a first-draft entry (`harness.planner_seam`), for an ask no
    `PERIOD_REDUCTION_UNSTATED` refusal named (`_declinable`). A decline carried
    unchanged from `previous`, the plan this one amends, was earned when it
    was recorded; one restated with the same `Decline.key` in other words is
    the same decline. A `BREAKDOWN_NOT_KEPT` decline is earned by its own
    rule, which checks a carried one too (`_breakdown_decline_unearned`).
    `detail.asks` lists the refused asks a decline named, when there are any.
    """
    carried = {decline.key for decline in previous.declined} if previous is not None else set()
    declines = [
        decline
        for decline in plan.declined
        if decline.key not in carried
        and decline.code is not UnsupportedOperation.BREAKDOWN_NOT_KEPT
        and not _declinable(decline, view, permitted=load_optional_declines(view.state))
    ]
    if not declines:
        return None
    unearned = sorted({decline.code.value for decline in declines})
    refused = sorted(
        {decline.ask_ref for decline in declines if decline.ask_ref is not None}
        & load_reduction_refused_asks(view.state)
    )
    message = (
        f"`declined` names {unearned!r}, and nothing in this planner entry earned that "
        "code. A decline is earned when a proposal or its `translation` answers `unsupported` "
        "with its code, when a `requirement_unmeetable` translation's reason says it earns "
        "that code, or when the refusal this entry replans from means no operation "
        "covers the step. To earn it, propose the computation the ask states; if nothing "
        "carries it out, that answer names the code."
    )
    if refused:
        message += (
            f" A `period_reduction_unstated` refusal named ask {_ask_list(refused)} in this "
            "entry, so a computation states it: make the change that refusal offers."
        )
    return problem(
        ProblemCode.DECLINE_UNEARNED,
        message,
        outcome=Outcome.INVALID_INPUT,
        detail={"operations": unearned} | ({"asks": refused} if refused else {}),
    )


def _declinable(
    decline: Decline, view: SessionView, *, permitted: frozenset[UnsupportedOperation]
) -> bool:
    """Whether this planner entry earned `decline`, with `permitted` the declines it allows unasked.

    A tool's `unsupported` answer with the decline's code earns it
    (`load_unsupported`). An entry's permission (`permitted`) covers no ask
    a `PERIOD_REDUCTION_UNSTATED` refusal named in the entry
    (`load_reduction_refused_asks`): that refusal found a computation the
    ask states, and offered the change that answers it.
    """
    if decline.code in load_unsupported(view.state):
        return True
    return decline.code in permitted and decline.ask_ref not in load_reduction_refused_asks(
        view.state
    )


def _ask_list(asks: Sequence[int]) -> str:
    """Name `asks` by number, joined with "and"."""
    return " and ".join(str(ask) for ask in asks)


def _breakdown_decline_unearned(plan: Plan, view: SessionView) -> dict[str, object] | None:
    """Return the refusal for a `BREAKDOWN_NOT_KEPT` decline the turn's search did not earn.

    Such a decline names an ask (`ask_ref`) whose answer drops a party it
    breaks down by, or keeps its parties only in separate outputs
    (`planner.plan.roles_not_kept`). It is earned unless a dataset
    requirement under the ask's own answer can keep more of those parties,
    on the source it pins, than the answer's best output keeps
    (`_answer_pin_holding`). Where one can, the refusal names the
    requirement, its spec_id and every party it can keep: each
    party to state in its `across`, with the word there that the party
    replaces, and then all of them to keep to one output. A party already in `across`, or
    one whose members the requirement names, only has to reach the output.
    `detail.held_open` holds one `{ask, role, spec_id}` entry per party the
    source holds open, and `detail.held_named`, present when there are any,
    one per party the requirement names members for. Two pins each holding
    one of two parties, under an answer keeping one, earn the decline:
    neither states a pair.

    `plan` pins the sources as the planner named them, before `_window_pins`
    moves each to its copy over the requirement's period. `pin_sources`
    holds a dataset under the spec_id its search returned, and `admit` moves
    every pin the amended plan carries back to that spec_id
    (`_unwindowed_pins`). A copy keeps the definition of the dataset it
    copies, so that dataset's open axes decide for the copy.

    Every such decline is checked against `plan`, one the plan it amends
    carried included: an amendment can move the ask's answer to a source
    that holds the parties, or to an output that keeps them. A carried
    decline whose answering pins are the ones it was recorded with gets the
    answer it got then.
    """
    declines = [
        decline
        for decline in plan.declined
        if decline.code is UnsupportedOperation.BREAKDOWN_NOT_KEPT
    ]
    if not declines:
        return None
    dropped = roles_not_kept(plan)
    parties = {breakdown.ask: breakdown.by for breakdown in plan.ask_breakdowns}
    sources = pin_sources(view.state)
    requirements = {requirement.id: requirement for requirement in plan.evidence_requirements}
    problems: list[str] = []
    held: dict[str, list[dict[str, object]]] = {"held_open": [], "held_named": []}
    for decline in declines:
        if decline.ask_ref is None:
            problems.append(
                "a `breakdown_not_kept` decline names no ask: give the number of the ask whose "
                "answer drops a party it breaks down by as `ask_ref`"
            )
            continue
        if decline.ask_ref not in dropped:
            problems.append(
                f"ask {decline.ask_ref}'s answer keeps every party the ask breaks down by, so "
                "it has no breakdown to decline"
            )
            continue
        by = parties[decline.ask_ref]
        outputs = next((a.outputs for a in plan.answered if a.ask == decline.ask_ref), ())
        holding = _answer_pin_holding(
            requirements, outputs, sources, by, kept=len(by) - len(dropped[decline.ask_ref])
        )
        if holding is None:
            continue
        for key, roles in (("held_open", holding.open), ("held_named", holding.named)):
            held[key] += [
                {"ask": decline.ask_ref, "role": role, "spec_id": holding.spec_id} for role in roles
            ]
        root = holding.root
        stating = " and ".join(
            f"{role!r} in place of {word!r}" if word is not None else repr(role)
            for role, word in holding.to_state
        )
        together = [role for role in by if role in {*holding.open, *holding.named}]
        problems.append(
            f"{holding.spec_id}, which {root!r} pins for ask {decline.ask_ref}, can keep "
            f"{' and '.join(together)}: "
            + (f"state {stating} in {root!r}'s `across`, and " if stating else "")
            + f"keep {' and '.join(together)} to one output, with no aggregate summing "
            f"{'it' if len(together) == 1 else 'them'} away"
        )
    if not problems:
        return None
    return problem(
        ProblemCode.DECLINE_UNEARNED,
        f"`declined` names 'breakdown_not_kept', and this turn's search did not earn it: "
        f"{'; '.join(problems)}.",
        outcome=Outcome.INVALID_INPUT,
        detail={
            "operations": [UnsupportedOperation.BREAKDOWN_NOT_KEPT.value],
            "held_open": held["held_open"],
            **({"held_named": held["held_named"]} if held["held_named"] else {}),
        },
    )


@dataclass(frozen=True)
class _PinHolding:
    """The parties of an ask a dataset requirement can keep on the source it pins (`_answer_pin_holding`).

    `open` are the parties the source holds on an open axis and `named` the
    ones the requirement names members for, each in the ask's order.
    `to_state` lists each open party no word of the requirement's `across`
    names (`planner.parties.same_party`), with the word there that covers
    its axis and that it replaces, `None` when no word does.
    """

    root: str
    spec_id: str
    open: tuple[str, ...]
    named: tuple[str, ...]
    to_state: tuple[tuple[str, str | None], ...]


def _answer_pin_holding(
    requirements: Mapping[str, EvidenceRequirement],
    outputs: Sequence[str],
    sources: Mapping[str, PinSource],
    by: Sequence[str],
    *,
    kept: int,
) -> _PinHolding | None:
    """The dataset requirement under `outputs` that can keep more than `kept` of the parties in `by` on the source it pins, with those parties.

    The chain runs from the ask's answering `outputs` through `dataset_roots`
    to each root's `source_spec_id` and the `PinSource` shown for it. A root
    keeps a party of `by` in one of two ways. It names entities with that
    role: the frame has one row per named member, and the axis on that side
    of the flow is not open to cover. Or the source holds the party open: an
    open axis on the party's side of the flow holds other than one member,
    and either a word of the root's `across` already covers that axis, or
    the search that settled on the source asked for the party or named no
    party. The root and its pin are one source, whose flow has one party on
    each side, so side decides: a role, an asked party or an `across` word
    names a party of `by` when the two words are on the same side of the
    flow (`planner.parties.same_party`), and a search that asked for
    "donor" asked for the creditor of a debt flow. `by` holds one word for
    each side (`planner.plan.AskBreakdown`).

    A word of `across` covers an axis as `open_axes_for` matches them: a
    party word covers the axis on its side, and a place word the flow axis
    `open_axes_for` puts it on when the root's `across` fits the source as
    written, as "country" covers IDS's `destination`. The open parties are
    confirmed as one set, since one of two open parties alone leaves the
    other uncovered: `open_axes_for` has to cover the root's `across`
    restated with each of them as its own word, in place of the word
    covering its axis when there is one. A restatement it does not cover
    leaves the root keeping only the parties it names. A root with an empty
    `across` and no open party has no word to restate, and nothing is
    asked of `open_axes_for`.

    The root keeping the most parties of `by` is returned, the first on a
    tie, when that is more than `kept`, the number the answer's best output
    keeps.

    A missing link names no holder, so the decline stands: an ask with no
    answer, a root with no pin, a source the search did not settle on, and a
    pin with no block here, which is an earlier turn's pin or a spec_id no
    search this entry returned.
    """
    best: _PinHolding | None = None
    most = kept
    roots = sorted({root for output in outputs for root in dataset_roots(requirements, output)})
    for root_id in roots:
        root = requirements[root_id]
        if not isinstance(root, DatasetRequirement) or root.source_spec_id is None:
            continue
        source = sources.get(root.source_spec_id)
        if source is None or not source.settled or source.block is None:
            continue
        named_roles = [entity.role for entity in root.entities if entity.role is not None]
        named = tuple(role for role in by if party_among(role, named_roles))
        named_sides = {side for role in named_roles if (side := flow_side(role)) is not None}
        axes = [
            axis
            for axis in open_entity_axes_of(source.block.get("resolved") or {})
            if flow_side(axis.column) not in named_sides
        ]
        open_sides = {flow_side(axis.column) for axis in axes if axis.member_count != 1}
        covers = _sides_covered(root.across, axes)
        candidates = tuple(
            role
            for role in by
            if role not in named
            and (side := flow_side(role)) in open_sides
            and (
                side in covers.values()
                or not source.asked_roles
                or party_among(role, source.asked_roles)
            )
        )
        stated = {flow_side(role): role for role in reversed(candidates)}
        restated = dict.fromkeys(
            (*(stated.get(covers[word], word) for word in root.across), *candidates)
        )
        if restated and isinstance(open_axes_for(tuple(restated), axes), str):
            candidates = ()
        if len(named) + len(candidates) <= most:
            continue
        most = len(named) + len(candidates)
        replaced = {side: word for word, side in reversed(covers.items()) if side is not None}
        best = _PinHolding(
            root=root_id,
            spec_id=root.source_spec_id,
            open=candidates,
            named=named,
            to_state=tuple(
                (role, replaced.get(flow_side(role)))
                for role in candidates
                if not party_among(role, root.across)
            ),
        )
    return best


def _sides_covered(
    across: Sequence[str], axes: Sequence[OpenEntityAxis]
) -> dict[str, FlowSide | None]:
    """The side of the flow whose open axis each `across` word covers, `None` for a word covering none.

    Where `across` fits `axes` as written, each word covers the side of the
    axis `open_axes_for` matches it to, so a place word standing for the one
    open party of a flow covers that party's side. Where it does not fit, a
    party word covers its own side and a place word none. An empty `across`
    covers no axis, and `open_axes_for` is asked only about words.
    """
    if not across:
        return {}
    fit = open_axes_for(across, axes)
    if isinstance(fit, str):
        return {
            word: None if (party := across_party(word)) == "place" else party for word in across
        }
    return {word: flow_side(axis.column) for word, axis in zip(across, fit, strict=True)}


def _nothing_to_plan(plan: Plan, view: SessionView) -> dict[str, object] | None:
    """Return the answer for a plan that declines every ask and states nothing else.

    The entry's required declines and terminal optional declines can close
    it, each as `_declinable` allows. An optional decline of incompatible sources still leaves their
    separate figures to show, so that permission alone cannot close it.
    """
    terminal = load_terminal_declines(view.state)
    if not plan.declined or not all(
        _declinable(decline, view, permitted=terminal) for decline in plan.declined
    ):
        return None
    return problem(
        ProblemCode.NOTHING_TO_PLAN,
        "Nothing in the question can be planned once its unsupported asks are declined.",
        outcome=Outcome.REFUSED,
        detail={"declined": [decline.model_dump(mode="json") for decline in plan.declined]},
    )


# The refusal for a `propose_plan` call after `MAX_PLAN_ROUNDS` proposals in one
# entry into the planner. The plan's version can exceed the cap, since only the
# count restarts on a new entry. It names what remains possible: approving the
# recorded version when it compiled, and the user's next message.
_BUDGET_EXHAUSTED = (
    f"{MAX_PLAN_ROUNDS} plans have been proposed in this planner run, each with its own "
    "discovery budget; no further version will be recorded."
)

# The same refusal for a `starts_new_plan` call. The call already claims to be a
# different question, so the text says a restart counts against the same bound.
_BUDGET_EXHAUSTED_RESTART = (
    f"A restart is a proposal like any other, and {MAX_PLAN_ROUNDS} have been made in this "
    "planner run, each with its own discovery budget; no further plan will be recorded, "
    "however the call is stated."
)

# What follows either refusal. The approval is offered only where the recorded plan
# compiled, since a plan that never did has no card for the user to approve.
_BUDGET_APPROVABLE = (
    "The last version that compiled can still be approved, and the user's next message "
    "opens a fresh budget."
)
_BUDGET_FRESH = "The user's next message opens a fresh budget."

# The refusal for a `starts_new_plan` call while `question_fixed`. The call is
# how the model tells the harness the user's reply asks a different question,
# so the message says what happened and that nothing further is needed from
# it. `harness.planner_seam` ends the turn on this code as it does on
# `BUDGET_EXHAUSTED`.
_QUESTION_FIXED_RESTART = (
    "This turn's question is fixed, so `starts_new_plan` cannot open a plan for a "
    "different one; nothing is recorded, and the plan already on record stands "
    "unchanged. This is exactly how a different question is reported: the turn ends "
    "here, the user is told to send it as its own message, and nothing further is "
    "needed from you."
)


def _resolve_intent(
    existing: Plan | None,
    *,
    amends_version: int | None,
    starts_new_plan: bool,
    rounds_so_far: int,
    question: str | None = None,
    question_fixed: bool = False,
) -> tuple[Plan | None, int] | dict[str, object]:
    """Which plan a `propose_plan` call means, from what the call states.

    Returns `(previous, rounds)` (the plan to patch, and the proposal count
    this call brings the entry to) or, when the call does not state its
    intent unambiguously, the problem to answer with. Which of the two a call
    is comes from the call's own `amends_version`/`starts_new_plan` alone,
    regardless of how `question`'s wording compares to the recorded plan's. A
    model that rewords the question while meaning to amend would otherwise be
    read as starting a fresh plan and drop every item it did not restate.

    With nothing recorded, the call is a first draft. With a plan recorded,
    an `amends_version` naming that same version patches it. `starts_new_plan`
    begins a plan of its own. Anything else is refused, naming the version to
    pass so the next call is correct.

    `amends_version` and `starts_new_plan` are read in that order, so a call
    stating both is treated as an amendment, since a payload shaped as a
    patch is read as one. `rounds_so_far` counts the proposals made in this
    entry into the planner. `PlannerSeam` zeroes it on every entry that is not
    a mid-turn resume. An amendment or a restart increments it, since every
    version opens a fresh discovery budget, and past `MAX_PLAN_ROUNDS` the call
    is refused rather than recorded, with no `action` named, since any tool
    named there would restart the loop. A restart begins at version 1 and is
    counted like an amendment, so restarting does not escape the bound.

    `question_fixed` (`harness.planner_seam` builds the planner with it
    set; standalone and `harness_pair_v1` leave it `False`) narrows both
    readings once a plan is recorded, against `existing.question` rather than
    the call's own wording of the question it is answering:

    - A restart (`starts_new_plan` with no `amends_version`) is refused with
      `QUESTION_FIXED` before the round cap is even checked, ahead of
      `BUDGET_EXHAUSTED`. Both refusals record nothing, so nothing is lost
      by checking this one first, and a restart can never succeed under a
      fixed question however fresh the budget is. Leading with a budget
      message here would read as an invitation to retry once the user's next
      message opens one, which is false.
    - An amendment (`amends_version` naming the recorded version) whose own
      `question` states something other than what is recorded is refused as
      `INVALID_PLAN`, checked after the round cap like the shape checks below
      it: a correctly-shaped amendment that merely lost the race for the last
      round in this entry is exactly what the cap exists to stop, so the round
      check still takes priority there, the opposite ordering from the restart
      case above.
    """
    if existing is None:
        return None, 1
    blocked = _question_fixed_restart_blocked(
        existing,
        amends_version=amends_version,
        starts_new_plan=starts_new_plan,
        question_fixed=question_fixed,
    ) or _round_cap_blocked(
        existing,
        amends_version=amends_version,
        starts_new_plan=starts_new_plan,
        rounds_so_far=rounds_so_far,
    )
    if blocked is not None:
        return blocked
    if amends_version is None and starts_new_plan:
        return None, rounds_so_far + 1
    invalid = _amendment_version_invalid(
        existing, amends_version
    ) or _question_fixed_amendment_invalid(
        existing, question=question, question_fixed=question_fixed
    )
    if invalid is not None:
        return invalid
    return existing, rounds_so_far + 1


def _question_fixed_restart_blocked(
    existing: Plan, *, amends_version: int | None, starts_new_plan: bool, question_fixed: bool
) -> dict[str, object] | None:
    """Return the `QUESTION_FIXED` refusal for a restart while `question_fixed`, or `None`.

    Split out of `_resolve_intent` to keep its own branch count within this
    project's `PLR0911` budget.
    """
    if not (question_fixed and amends_version is None and starts_new_plan):
        return None
    return problem(
        ProblemCode.QUESTION_FIXED,
        _QUESTION_FIXED_RESTART,
        outcome=Outcome.REFUSED,
        detail={"version": existing.version},
    )


def _round_cap_blocked(
    existing: Plan, *, amends_version: int | None, starts_new_plan: bool, rounds_so_far: int
) -> dict[str, object] | None:
    """Return the `BUDGET_EXHAUSTED` refusal past `MAX_PLAN_ROUNDS`, or `None`.

    Split out of `_resolve_intent` for the same reason as
    `_question_fixed_restart_blocked`, above.
    """
    if rounds_so_far + 1 <= MAX_PLAN_ROUNDS:
        return None
    restart = amends_version is None and starts_new_plan
    return problem(
        ProblemCode.BUDGET_EXHAUSTED,
        f"{_BUDGET_EXHAUSTED_RESTART if restart else _BUDGET_EXHAUSTED} "
        f"{_BUDGET_APPROVABLE if existing.translation is not None else _BUDGET_FRESH}",
        outcome=Outcome.REFUSED,
        detail={
            "scope": "plan_rounds",
            "rounds": MAX_PLAN_ROUNDS,
            "version": existing.version,
        },
    )


def _amendment_version_invalid(
    existing: Plan, amends_version: int | None
) -> dict[str, object] | None:
    """Return the `INVALID_PLAN` refusal for an amendment naming no version, or the wrong
    one, or `None` when `amends_version` is exactly `existing.version`.

    Split out of `_resolve_intent` for the same reason as
    `_question_fixed_restart_blocked`, above.
    """
    if amends_version is None:
        return _invalid_plan(
            f"A plan is already recorded (version {existing.version}), so this call has "
            "to say which it is: pass "
            f"`amends_version={existing.version}` to patch that plan, or "
            "`starts_new_plan=true` to begin a plan of its own for a different question. "
            "A patch states only what changes; everything it does not mention is carried "
            "forward.",
        )
    if amends_version != existing.version:
        return _invalid_plan(
            f"This call amends version {amends_version}, but version {existing.version} is "
            f"the one recorded. Pass `amends_version={existing.version}`, or "
            "`starts_new_plan=true` to begin a plan of its own for a different question.",
        )
    return None


def _question_fixed_amendment_invalid(
    existing: Plan, *, question: str | None, question_fixed: bool
) -> dict[str, object] | None:
    """Return the `INVALID_PLAN` refusal for an amendment that rewords a fixed question, or `None`.

    Split out of `_resolve_intent` for the same reason as
    `_question_fixed_restart_blocked`, above.
    """
    if not (question_fixed and question is not None and question != existing.question):
        return None
    return _invalid_plan(
        f"This call amends version {existing.version}, but its `question` "
        f"({question!r}) does not match the one recorded ({existing.question!r}) -- "
        "the question is fixed for this turn, so a patch may not change it. Omit "
        "`question` to leave it as recorded, or restate it unchanged. Pass "
        "`starts_new_plan=true` instead if the user's reply genuinely asks a "
        "different question; that ends the turn rather than recording a plan for it.",
    )


def _referral_unanswered(
    existing: Plan | None, referral: Referral | None, view: SessionView
) -> dict[str, object] | None:
    """Return the refusal for a `propose_plan` call over a referral the user has not
    answered, or `None` to let the call through.

    A first draft, with no plan recorded, is never held.

    A patch replaces a requirement whole, and a proposal clears the referral, so a
    plan proposed over an `unsupported_ask` or `requirement_unmeetable`
    referral would drop what the referral names and go on to translate. Every
    other kind is the planner's to settle, so the call goes through. The
    refusal points at the tool that asks the user, since `propose_plan` is
    what it stops. `referral` is the pending one as the planner reads it
    (`_authored_referral`).
    """
    if (
        existing is None
        or referral is None
        or referral.kind not in USER_DECIDES
        or view.answers_since_referral().answered
    ):
        return None
    return problem(
        ProblemCode.REFERRAL_PENDING,
        f"The translator handed requirement `{referral.requirement_id}` back as "
        f"{referral.kind.value}, and the user has said nothing since. Tell them what "
        f"cannot be done and put the choice to them through `{REQUEST_INPUT_FUNCTION_CALL_NAME}` "
        "before proposing again.",
        outcome=Outcome.REFUSED,
        action=Action(call=REQUEST_INPUT_FUNCTION_CALL_NAME),
    )


# The option of a combination ask that adds the two datasets.
COMBINE_OPTION = "add them"

# The options a combination ask offers, as `_combination_unasked`'s refusal
# hands them to `adk_request_input`: each `id` is what the user's pick comes
# back as, and Studio shows each `label`.
COMBINATION_OPTIONS: tuple[dict[str, str], ...] = (
    {"id": COMBINE_OPTION, "label": "Add them into one total"},
    {"id": "show them separately", "label": "Show them side by side, without adding them"},
)


def _combination_ask() -> Action:
    """The `adk_request_input` call that asks the user whether to add two datasets.

    It carries `COMBINATION_OPTIONS` and needs only the question. What it asks
    about is the refusal that carries it (`SessionView.combination_picks`).
    """
    return Action(
        call=REQUEST_INPUT_FUNCTION_CALL_NAME,
        arguments={"options": [dict(option) for option in COMBINATION_OPTIONS]},
        needs=("message",),
    )


def _stated_source_given_way(
    requirements: Sequence[EvidenceRequirement], referral: Referral | None, view: SessionView
) -> list[EvidenceRequirement]:
    """`requirements`, with the stated `source` dropped from a requirement that pins the source the user chose.

    A pending `requirement_unmeetable` referral (`referral`, as
    `_authored_referral` gives it) marked `source_mismatch` hands a requirement back because the source it states names none of the
    sources discovery found, and offers those. A pin of one of them that the
    user picked, answering an ask raised after the referral
    (`SessionView.answers_since_referral`), names the exact source they chose, so the
    stated source gives way to it. Left in place, it would refuse that source
    as `compiler.evidence`'s `SOURCE_MISMATCH` and refer the plan back the
    same way. Any other pin keeps the stated source, including one the planner
    settled itself after a free-text answer and one the user picked before the
    referral, so the compiler still refuses a source the user neither named
    nor chose.
    """
    if referral is None or not referral.source_mismatch:
        return list(requirements)
    offered = {candidate.spec_id for candidate in referral.candidates}
    chosen = offered.intersection(view.answers_since_referral().picked)
    return [
        requirement.model_copy(update={"source": None})
        if isinstance(requirement, DatasetRequirement)
        and requirement.id == referral.requirement_id
        and requirement.source_spec_id in chosen
        else requirement
        for requirement in requirements
    ]


def _authored_referral(view: SessionView, recorded: Plan | None) -> Referral | None:
    """The session's pending referral as the planner reads it, naming each requirement by the id the model wrote.

    The referral saved in session state keys on the ids of `recorded`, the
    plan on record, where a copy the split added names its own
    (`translator.tool.authored_answer`). `admit` reads it once and hands it
    to each check that reads the referral.
    """
    referral = load_referral(view.state)
    if referral is None or recorded is None or not recorded.rolling_copies:
        return referral
    return Referral.model_validate(
        authored_answer(referral.model_dump(mode="json"), split_of(recorded))
    )


def _choice_not_offered(
    plan: Plan, referral: Referral | None, view: SessionView
) -> dict[str, object] | None:
    """Return the refusal for a pin a pending source choice did not offer, or `None`.

    A `choice_required` `referral`, as `_authored_referral` gives it, names the sources each requirement it is
    about may pin (`Referral.offered`): its `candidates`, and each stale
    pin's, which discovery recorded and no model wrote. While it is pending,
    each such requirement as `plan` states it either pins one of them or
    pins nothing. A requirement the plan no longer states passes.

    A requirement whose found sources the referral withholds, because they
    drop the scope of the pinned source it set aside, may pin nothing only
    when it also changes its `entities`, `across` or `source` from the plan the
    referral is about: stated unpinned and otherwise unchanged, it would
    have the translator's search find those sources again
    (`_withheld_scope_dropped`).

    An `entity_name_ambiguous` referral offers graph nodes and no sources,
    so it leaves every pin as it is (`_entity_dcid_unoffered` checks its
    choice).
    """
    if (
        referral is None
        or referral.outcome is not Outcome.CHOICE_REQUIRED
        or referral.kind is ReferralKind.ENTITY_NAME_AMBIGUOUS
    ):
        return None
    for requirement_id, candidates in referral.offered().items():
        try:
            requirement = plan.requirement(requirement_id)
        except KeyError:
            continue
        if not isinstance(requirement, DatasetRequirement):
            continue
        if requirement.source_spec_id is None:
            refused = _withheld_scope_dropped(requirement, referral, view)
            if refused is not None:
                return refused
            continue
        offered = [candidate.spec_id for candidate in candidates]
        if requirement.source_spec_id in offered:
            continue
        return problem(
            ProblemCode.SOURCE_NOT_OFFERED,
            f"{requirement.id!r} pins {requirement.source_spec_id!r}, which the "
            f"{referral.kind.value} referral did not offer. Pin one of its candidates, put "
            "them to the user, or unpin the requirement and change what it asks for.",
            outcome=Outcome.REFUSED,
            detail={"requirement_id": requirement.id, "candidates": offered},
        )
    return None


def _withheld_scope_dropped(
    requirement: DatasetRequirement, referral: Referral, view: SessionView
) -> dict[str, object] | None:
    """Return the refusal for `requirement`, stated unpinned, that would retrieve a source `referral` withholds, or `None`.

    It is refused when `referral` withholds sources found in its place
    (`translator.referral.WithheldSource`) and it asks `search` for what the
    recorded plan the referral is about asked (`_search_scope`).
    """
    if requirement.id == referral.requirement_id:
        withheld, repair = referral.withheld, referral.repair
    else:
        stale = next(
            (pin for pin in referral.stale_pins if pin.requirement_id == requirement.id), None
        )
        withheld, repair = (stale.withheld, stale.repair) if stale is not None else ((), None)
    recorded = load_plan(view.state)
    if not withheld or recorded is None:
        return None
    try:
        referred = recorded.requirement(requirement.id)
    except KeyError:
        return None
    if not isinstance(referred, DatasetRequirement) or (
        _search_scope(requirement) != _search_scope(referred)
    ):
        return None
    return problem(
        ProblemCode.SOURCE_NOT_OFFERED,
        f"{requirement.id!r} is unpinned but asks for what it asked before, so the search "
        "would find again the sources the referral withholds for dropping the pinned "
        "source's scope. "
        + (
            f"Make the referral's repair, {repair}, and propose again."
            if repair is not None
            else f"Ask the user through `{REQUEST_INPUT_FUNCTION_CALL_NAME}` what to change."
        ),
        outcome=Outcome.REFUSED,
        detail={"requirement_id": requirement.id, "repair": repair},
    )


def _search_scope(
    requirement: DatasetRequirement,
) -> tuple[
    frozenset[tuple[str, str | None, bool]],
    frozenset[str],
    tuple[str | None, str | None, bool],
    str | None,
]:
    """The scope `search` is asked for `requirement`: its entities, open roles, named source and series code.

    The entities are a set of names, casefolded with runs of whitespace as
    one space, each with the `PARTY_ROLES` word its role stands for
    (`party_role`) and whether it asks for the set's members, and the open
    roles are the set `translator.discovery.requirement_open_roles` gives.
    Two requirements stating their entities or their `across` words in
    another order, or their `across` in other words for one role
    ("recipient", "recipients"), have one scope. The named source follows
    Search's word matching: casefold, punctuation as spaces, a leading "the"
    dropped and whitespace collapsed.
    """
    entities = frozenset(
        (
            " ".join(entity.name.casefold().split()),
            party_role(entity.role) if entity.role is not None else None,
            entity.members,
        )
        for entity in requirement.entities
    )
    source = None
    if requirement.source is not None:
        words = "".join(
            " " if unicodedata.category(char).startswith("P") else char
            for char in requirement.source.casefold()
        ).split()
        if len(words) > 1 and words[0] == "the":
            words = words[1:]
        source = " ".join(words)
    return (
        entities,
        frozenset(requirement_open_roles(requirement)),
        (source, requirement.source_ref, requirement.source_exclude),
        requirement.source_code,
    )


class SourceReadings:
    """The spec block each spec_id has and the source it states
    (`compiler.sources.readable_source`), each looked up and read once, on first use.

    One proposal reads its pins' sources through one of these at every site
    that names, labels or resolves by a pin's source, so each block is parsed
    once and an unreadable one logs its warning once.
    """

    def __init__(self, block_of: Callable[[str], Mapping[str, Any] | None]) -> None:
        self._block_of = block_of
        self._blocks: dict[str, Mapping[str, Any] | None] = {}
        self._sources: dict[str, SelectedSource | None] = {}

    @classmethod
    def of_shown(cls, shown: Mapping[str, PinSource]) -> SourceReadings:
        """The readings of the blocks of the datasets the planner was shown (`_pin_block`)."""
        return cls(lambda spec_id: _pin_block(spec_id, shown))

    def block(self, spec_id: str) -> Mapping[str, Any] | None:
        """The spec block of `spec_id`, `None` when none is held."""
        if spec_id not in self._blocks:
            self._blocks[spec_id] = self._block_of(spec_id)
        return self._blocks[spec_id]

    def get(self, spec_id: str) -> SelectedSource | None:
        """The source `spec_id`'s block states, `None` where no block is held or it is not readable."""
        if spec_id not in self._sources:
            block = self.block(spec_id)
            self._sources[spec_id] = readable_source(block) if block is not None else None
        return self._sources[spec_id]


def pinned_sources(
    requirements: Iterable[EvidenceRequirement], sources: SourceReadings
) -> dict[str, SelectedSource]:
    """The source each dataset requirement of `requirements` pins, by spec_id, as
    `sources` reads it, leaving out a pin whose block is not held or not readable."""
    return {
        requirement.source_spec_id: source
        for requirement in requirements
        if isinstance(requirement, DatasetRequirement)
        and requirement.source_spec_id is not None
        and (source := sources.get(requirement.source_spec_id)) is not None
    }


def _pinned_source(
    spec_id: str | None, view: SessionView, sources: SourceReadings
) -> SelectedSource | None:
    """The source `spec_id` pins, as the record that states it; `None` when nothing on record does.

    The source is a block's the planner was shown, as `sources` reads it, or
    one the translator's discovery recorded, such as a referral's candidate
    (`translator.discovery.DiscoveryLog.recorded_source`).
    """
    if spec_id is None:
        return None
    if sources.block(spec_id) is not None:
        return sources.get(spec_id)
    return load_log(view.state).recorded_source(spec_id)


def _entity_refs(
    view: SessionView, sources: SourceReadings
) -> Callable[[DatasetRequirement | None, Entity], frozenset[str]]:
    """The graph refs `check_wholes_span_their_totals` compares an entity by, given its side's dataset requirement.

    A name is resolved against the source its dataset requirement pins
    (`_pinned_source`), by label or by the refs Search bound for it
    (`compiler.evidence.named_refs`), as
    `translator.discovery._divides_its_member_by_itself` compares a pin's
    members; a group names every member the source binds on its side. A
    name the pin does not resolve, or on a requirement pinning nothing on
    record, resolves no ref and is compared by its name.
    """

    # `sources` caches a shown block's source; this also caches the discovery log's
    # source for a pin with no shown block, which `load_log` rebuilds on every read.
    @functools.cache
    def source(spec_id: str) -> SelectedSource | None:
        return _pinned_source(spec_id, view, sources)

    def refs(dataset: DatasetRequirement | None, entity: Entity) -> frozenset[str]:
        if dataset is None or dataset.source_spec_id is None:
            return frozenset()
        pinned = source(dataset.source_spec_id)
        return frozenset() if pinned is None else frozenset(named_refs((entity,), pinned))

    return refs


class _SumDatasets(NamedTuple):
    """The datasets both sides of a `sum_of` read, and what their pins state of them.

    `stated` says a pin states each dataset's provenance. `provider_flows`
    says each pin also states its provenance's account is `PROVIDER_ACCOUNT`:
    the OECD and ONE aid tables, whose skill says how they add up.
    """

    datasets: frozenset[str]
    stated: bool
    provider_flows: bool


def _sum_datasets(
    plan: Plan, requirement: SumOfRequirement, view: SessionView, sources: SourceReadings
) -> _SumDatasets:
    """The datasets both sides of `requirement` read in `plan`, and what their pins state.

    A dataset requirement whose pin states its dataset (`_pinned_source`)
    reads that dataset's provenance ref. One whose dataset no pin states
    reads a dataset of its own: `spec:<source_spec_id>` for a pin whose
    dataset this turn cannot read, `requirement:<id>:<revision>` for an unpinned one, where the revision
    (`requirement_revision`) hashes what the requirement asks for. So two such
    sides read one dataset only when they share the pin, or the requirement
    with the content it was asked about.
    """
    datasets: set[str] = set()
    stated = True
    provider_flows = True
    pending = [requirement.left.input_id, requirement.right.input_id]
    while pending:
        read = plan.requirement(pending.pop())
        if isinstance(read, DatasetRequirement):
            source = _pinned_source(read.source_spec_id, view, sources)
            prior = pin_sources(view.state).get(read.source_spec_id or "")
            dataset = (
                source.provenance_ref
                if source is not None
                else prior.provenance_ref
                if prior is not None
                else None
            )
            account = (
                source.provenance_account
                if source is not None
                else prior.provenance_account
                if prior is not None
                else None
            )
            if dataset is None:
                stated = False
                dataset = (
                    f"requirement:{read.id}:{requirement_revision(read)}"
                    if read.source_spec_id is None
                    else f"spec:{read.source_spec_id}"
                )
            if account != PROVIDER_ACCOUNT:
                provider_flows = False
            datasets.add(dataset)
        pending.extend(inputs_of(read))
    return _SumDatasets(frozenset(datasets), stated, provider_flows)


def _combination_unasked(
    plan: Plan, view: SessionView, *, previous: Plan | None, sources: SourceReadings
) -> dict[str, object] | None:
    """Return the refusal for a `sum_of` of two datasets the user was not asked to combine, or `None`.

    Whether two datasets add up without counting a flow twice is the
    user's to settle, since ODA loans, for one, count in both aid and debt
    flows. So a `sum_of` passes only when the user's latest pick for it,
    its id and the datasets its sides read (`SessionView.combination_picks`), is
    `COMBINE_OPTION`, and it cites a `Decision` resolved by `user` whose
    `chosen` is that option, unless its sides read one dataset, or only
    tables of a provider's flows (`_SumDatasets.provider_flows`), as the
    pins of their dataset requirements state. A `user` decision is the model's own claim,
    so citing one, even a source pin the user did elect, settles nothing
    without the pick. A pick for one sum, or for the same sum over other
    datasets, settles nothing for another.

    `plan` pins the sources the planner named, before `_window_pins` moves
    a pin to a copy over the requirement's period, which no store records a
    dataset for until the plan is recorded. A side with an unpinned dataset
    requirement, or a pin whose dataset this turn cannot read
    (`_pinned_source`), reads a dataset of its own, named by that
    requirement or pin (`_sum_datasets`). A Studio turn opens its own
    session, so a plan carried into a later turn has no pick on record.
    A `sum_of` that `previous`, the plan this call amends, states unchanged,
    over the same datasets and citing the same `user` decisions to add
    them, was checked when `previous` was recorded and keeps its sum,
    unless some combination ask in this session was answered with anything
    but `COMBINE_OPTION`. The refusal's action carries the `COMBINATION_OPTIONS`
    the ask offers, so the user picks one of them and the planner phrases
    only the question, and its `detail` names the sum and its datasets,
    which is what the ask that follows asks about.
    """
    picks, declined = view.combination_picks()
    for requirement in plan.evidence_requirements:
        if not isinstance(requirement, SumOfRequirement):
            continue
        datasets, stated, provider_flows = _sum_datasets(plan, requirement, view, sources)
        if stated and (len(datasets) == 1 or provider_flows):
            continue
        consent = _consenting_decisions(plan, requirement)
        pick = picks.get((requirement.id, datasets))
        if consent and pick == COMBINE_OPTION:
            continue
        if (
            consent
            and previous is not None
            and not declined
            and requirement in previous.evidence_requirements
            and _sum_datasets(previous, requirement, view, sources).datasets == datasets
            and _consenting_decisions(previous, requirement) == consent
        ):
            continue
        return problem(
            ProblemCode.COMBINATION_UNASKED,
            f"{requirement.id!r} adds two datasets, and whether they add up without "
            "counting a flow twice is the user's to decide: ODA loans, for one, count in "
            f"both aid and debt flows. Ask the user through `{REQUEST_INPUT_FUNCTION_CALL_NAME}` "
            "with this action's `options` unchanged and a `message` naming that overlap. "
            f"If they pick {COMBINE_OPTION!r}, cite a `Decision` resolved by `user` whose "
            "`options` are those option ids and whose `chosen` is the one they picked; "
            f"otherwise remove {requirement.id!r} and show the two series separately.",
            outcome=Outcome.REFUSED,
            action=_combination_ask(),
            detail={
                "requirement_id": requirement.id,
                "datasets": sorted(datasets),
            },
        )
    return None


def _consenting_decisions(plan: Plan, requirement: SumOfRequirement) -> dict[str, Decision]:
    """The decisions `requirement` cites in `plan` resolved by `user` to `COMBINE_OPTION`, by slot."""
    return {
        decision.slot: decision
        for decision in plan.decisions
        if decision.slot in requirement.decision_ids
        and decision.resolved_by is ResolvedBy.USER
        and decision.chosen == COMBINE_OPTION
    }


def _source_ref_unpicked(
    plan: Plan, referral: Referral | None, view: SessionView, previous: Plan | None
) -> dict[str, object] | None:
    """A new source reference must be a source option the user actually picked.

    The options are the candidates the open `source_name_ambiguous` referral
    (`referral`, as `_authored_referral` gives it) offers for the requirement's `source` wording, or for its `source_code`
    when more than one publisher holds the code.
    """
    picked = view.answers_since_referral().picked
    for requirement in plan.evidence_requirements:
        if not isinstance(requirement, DatasetRequirement) or requirement.source_ref is None:
            continue
        old = (
            next((r for r in previous.evidence_requirements if r.id == requirement.id), None)
            if previous
            else None
        )
        if (
            isinstance(old, DatasetRequirement)
            and old.source_ref == requirement.source_ref
            and old.source == requirement.source
            and old.source_exclude == requirement.source_exclude
        ):
            continue
        asked = {
            " ".join(wording.casefold().split())
            for wording in (requirement.source, requirement.source_code)
            if wording is not None
        }
        offered = (
            {
                candidate.ref
                for group in referral.ambiguous_sources
                if " ".join(group.text.casefold().split()) in asked
                for candidate in group.candidates
            }
            if referral
            and referral.kind is ReferralKind.SOURCE_NAME_AMBIGUOUS
            and referral.requirement_id == requirement.id
            else set()
        )
        if requirement.source_ref not in picked or requirement.source_ref not in offered:
            return _invalid_plan(
                "The source_ref must be a source reference Search offered and the user picked. Ask the user before resolving source ambiguity."
            )
    return None


def _kept_entities(
    previous: Plan | None, requirement_id: str
) -> set[tuple[str, str | None, str | None]]:
    """The `(name, role, dcid)` of each entity the previous plan states for `requirement_id`."""
    if previous is None:
        return set()
    old = next((r for r in previous.evidence_requirements if r.id == requirement_id), None)
    if not isinstance(old, DatasetRequirement):
        return set()
    return {(e.name, e.role, e.dcid) for e in old.entities}


def _entity_dcid_unoffered(
    plan: Plan, referral: Referral | None, previous: Plan | None
) -> dict[str, object] | None:
    """A new entity `dcid` must be a graph node Search offered for that entity.

    The options are the candidates of the reading the open
    `entity_name_ambiguous` referral (`referral`, as `_authored_referral`
    gives it) holds for the entity's `(name, role)` in its requirement. The planner settles the choice when the
    question decides it, so the user need not have picked it. A candidate
    restated by a group's name (`EntityCandidate.restate_as`) offers no
    node, since Search refuses a pinned group asked for its members; the
    entity restating it states no `dcid` (`_restated`). An entity
    whose `(name, role, dcid)` the previous plan states for the requirement
    carries its settled choice forward.
    """
    for requirement in plan.evidence_requirements:
        if not isinstance(requirement, DatasetRequirement):
            continue
        kept = _kept_entities(previous, requirement.id)
        readings: tuple[AmbiguousEntityReading, ...] = ()
        if (
            referral is not None
            and referral.kind is ReferralKind.ENTITY_NAME_AMBIGUOUS
            and referral.requirement_id == requirement.id
        ):
            readings = referral.ambiguous_entities
        for entity in requirement.entities:
            if entity.dcid is None or (entity.name, entity.role, entity.dcid) in kept:
                continue
            offered = {
                candidate.ref
                for reading in readings
                if reading.entity == EntityKey(name=entity.name, role=entity.role)
                for candidate in reading.candidates
                if candidate.restate_as is None
            }
            if entity.dcid not in offered:
                return _invalid_plan(
                    "An entity's dcid must be a graph node an entity_name_ambiguous referral "
                    "offered for that entity. Leave dcid unset unless Search asked which node "
                    "the entity is."
                )
    return None


def _entity_dcid_unpicked(
    plan: Plan, referral: Referral | None, view: SessionView, previous: Plan | None
) -> dict[str, object] | None:
    """An entity must state the candidate the user picked for it, when they were asked.

    An ask raised after the open `entity_name_ambiguous` referral (`referral`,
    as `_authored_referral` gives it) offers one reading's candidates under
    their `pick` ids (`planner.records.entity_pick`), so an answer names the
    entity as well as the candidate. The latest pick for a reading fixes how
    its entity is stated. An entity with a `dcid` is the reading of its
    `(name, role)` in the referral's requirement, and has to state the picked
    node. An entity restating a group's name (`_restated`) no longer carries
    the reading's name, so it is the reading of each candidate it restates,
    and has to be the pick of one of them. A reading an ask offered that the
    user has not answered stays unsettled until they do. A reading nobody
    asked about, and one the user answered in their own words, the planner
    settles (`_entity_dcid_unoffered` checks the node was offered). An entity
    the previous plan states alike for the requirement carries its settled
    choice forward. A pick names its requirement (`entity_pick`), so a pick
    made under an earlier referral about the same requirement still binds its
    reading when this one offers it again, and the planner cannot drop a
    picked node and state another once Search refers the entity anew. Only
    asks raised since this referral hold a reading unanswered.
    """
    if referral is None or referral.kind is not ReferralKind.ENTITY_NAME_AMBIGUOUS:
        return None
    readings = {
        candidate.pick: (reading.entity, candidate)
        for reading in referral.ambiguous_entities
        for candidate in reading.candidates
    }
    answers = view.answers_since_referral()
    chosen = {
        readings[pick][0]: readings[pick][1] for pick in answers.picked_ever if pick in readings
    }
    held = {readings[pick][0] for pick in answers.unanswered if pick in readings} - chosen.keys()
    for requirement in plan.evidence_requirements:
        if (
            not isinstance(requirement, DatasetRequirement)
            or requirement.id != referral.requirement_id
        ):
            continue
        kept = _kept_entities(previous, requirement.id)
        for entity in requirement.entities:
            if (entity.name, entity.role, entity.dcid) in kept:
                continue
            if entity.dcid is not None:
                keys = [EntityKey(name=entity.name, role=entity.role)]
            elif not (keys := _restated(entity, referral.ambiguous_entities)):
                continue
            stated = (entity.dcid, None if entity.dcid is not None else entity.name)
            if any(
                key not in held and (key not in chosen or _stated_as(chosen[key]) == stated)
                for key in keys
            ):
                continue
            key = keys[0]
            if key in chosen:
                return _invalid_plan(
                    f"The user picked {chosen[key].label!r} for the entity {key.name!r} "
                    f"(role {key.role!r}). State it with {_statement(chosen[key])}."
                )
            return _invalid_plan(
                f"The user was asked which node the entity {key.name!r} "
                f"(role {key.role!r}) names and has not picked one. Leave its dcid "
                "unset or ask again; do not choose it for them."
            )
    return None


def _restated(entity: Entity, readings: Sequence[AmbiguousEntityReading]) -> list[EntityKey]:
    """The readings whose candidate `entity` restates by a group's name, empty for any other entity.

    Such an entity states no `dcid`, sets `members` and carries as its name
    a candidate's `restate_as` for a reading of the same role. The name
    replaces the reading's own, so the entity is keyed by the candidate it
    restates: by the `(name, role)` of each reading offering that candidate.
    """
    if entity.dcid is not None or not entity.members:
        return []
    return [
        reading.entity
        for reading in readings
        if reading.entity.role == entity.role
        and any(candidate.restate_as == entity.name for candidate in reading.candidates)
    ]


def _stated_as(candidate: EntityCandidate) -> tuple[str | None, str | None]:
    """The `(dcid, name)` an entity stating `candidate` carries; the name only for a restated group."""
    if candidate.restate_as is None:
        return candidate.ref, None
    return None, candidate.restate_as


def _statement(candidate: EntityCandidate) -> str:
    """How an entity states `candidate`, as `ENTITY_CHOICE_REPAIR` tells the planner."""
    if candidate.restate_as is None:
        return f"{candidate.ref!r} as its dcid"
    return f"the name {candidate.restate_as!r}, members true and no dcid"


def _unrecordable(
    plan: Plan,
    view: SessionView,
    *,
    referral: Referral | None,
    stated: Plan,
    previous: Plan | None,
    reads_user_words: bool,
    proposer: Proposer,
    sources: SourceReadings,
) -> dict[str, object] | None:
    """Return the refusal for a built plan `propose_plan` may not record, or `None`.

    A plan states at least one requirement, unless it declines every ask
    (`_nothing_to_plan`); one that states none and makes a decline nothing
    earned is refused for the decline (`_decline_unearned`). A patch carries forward what it
    does not name, so the amended plan is checked whole for a source
    reference the user did not pick (`_source_ref_unpicked`), an entity
    node no referral offered (`_entity_dcid_unoffered`) or other than the one
    the user picked for it (`_entity_dcid_unpicked`), and a pin a
    pending source choice did not offer (`_choice_not_offered`), each
    against `stated`, the plan before its split and `_window_pins`, with
    `referral`, the pending referral naming the requirements the model wrote
    (`_authored_referral`), for a
    sum of two datasets the user was not asked to combine
    (`_combination_unasked`, against `stated`, the plan before `_window_pins`
    moved its pins), for a
    decline no `unsupported` answer earned (`_decline_unearned`, against
    `previous`, the plan it amends), for a decline of a party a source this
    turn holds open (`_breakdown_decline_unearned`, against `stated`), for a decline every `unsupported`
    answer this turn requires (`_decline_missing`), for a result given
    per member of a party nobody named that no decision states
    (`_result_scope_unstated`), and for a step reducing across years for a
    result no ask across years asks for (`_period_reduction_unstated`,
    against `stated`, read with `reads_user_words`).
    """
    if not plan.evidence_requirements:
        return (
            _nothing_to_plan(plan, view)
            or _decline_unearned(plan, previous, view)
            or _invalid_plan(
                "A plan states at least one evidence requirement: what data the answer "
                "rests on, citing the decisions that fixed it. State the requirements this "
                "question needs, or name the ones a patch removes in "
                "`removed_evidence_requirements`."
            )
        )
    return (
        _source_ref_unpicked(stated, referral, view, previous)
        or _entity_dcid_unoffered(stated, referral, previous)
        or _entity_dcid_unpicked(stated, referral, view, previous)
        or _choice_not_offered(stated, referral, view)
        or _combination_unasked(stated, view, previous=previous, sources=sources)
        or _decline_unearned(plan, previous, view)
        or _breakdown_decline_unearned(stated, view)
        or _decline_missing(plan, view, proposer)
        or _result_scope_unstated(plan)
        or _period_reduction_unstated(stated, view, reads_user_words=reads_user_words)
    )


def _result_scope_unstated(plan: Plan) -> dict[str, object] | None:
    """Return the `RESULT_SCOPE_UNSTATED` refusal naming every result `plan` gives per member of a party nobody named, or `None`.

    The rule is `planner.plan.unstated_result_scopes`'. For each such result
    the refusal names the result, the ask it answers, the party, and the two
    ways to settle it: sum the party away before the calculation, or state
    per-member results in a decision the dataset requirement cites.
    `detail.gaps` holds one entry per result, in the order the rule finds them.
    """
    gaps = unstated_result_scopes(plan)
    if not gaps:
        return None
    sentences = []
    for gap in gaps:
        answers = (
            f"answers ask {gap.ask} ({plan.asks[gap.ask]!r})"
            if gap.ask is not None
            else "is the plan's answer"
        )
        # Every calculation that keeps an open party past the shape checks reads
        # the years apart, so no result here takes one figure over them.
        pool = pooling_remedy(one_figure=False)
        sentences.append(
            f"{gap.output_id!r} {answers} with one figure per {gap.role}, because "
            f"{gap.dataset_id!r} reads every {gap.role} (`across`) and nothing sums them. "
            f"If the ask wants them taken together, total {gap.dataset_id!r} with reduction "
            f"`sum` and have any calculation read the total: {pool}, with `entity` in a "
            "per-year `group_by` to keep the entities it names apart. "
            f"If it wants a figure per {gap.role}, cite on "
            f"{gap.dataset_id!r} a decision whose `fixes` names `across`."
        )
    return problem(
        ProblemCode.RESULT_SCOPE_UNSTATED,
        " ".join(sentences),
        outcome=Outcome.INVALID_INPUT,
        detail={
            "gaps": [
                {
                    "requirement_id": gap.output_id,
                    "ask": gap.ask,
                    "role": gap.role,
                    "dataset_requirement_id": gap.dataset_id,
                }
                for gap in gaps
            ]
        },
    )


class OneYear(StrEnum):
    """The change a `PERIOD_REDUCTION_UNSTATED` refusal offers to have a step read one year.

    Each edits the refused step alone, so no other result's years or inputs
    move (`planner.plan.one_year_followed`), save those of the refused
    steps it is offered together with. `WINDOW` sets a rank's own
    window to the year. `FILTER` has an aggregate read its own input through
    an `eq` filter on the year. The year is one every series keeps holding
    as newer data arrives (`planner.plan.YearsHeld.one_year`): the last one
    the step's input holds, fixed where that year is fixed or a fixed bound
    stops it, and rolling at the offset newer data moves it by otherwise,
    or that last year fixed where the rolling one gives a plan
    `planner.plan._one_year_offered` does not pass. A change is offered
    only where the plan it gives, alone or with the changes it is offered
    together with, validates and retrieves every dataset over
    the window the refused plan does, so it meets no rule the refused plan
    did not and leaves every other result's years as they were.
    """

    WINDOW = "window"
    FILTER = "filter"


def _one_year_rolling(offset: int | None, *, rolls: str) -> str:
    """How a one-year remedy states `rolling` at `offset`, `None` for a fixed year; `rolls` names what rolls."""
    if offset is None:
        return "`rolling` unset"
    year = "the newest year" if offset == 0 else f"the year {-offset} before the newest"
    return f"{rolls} at offset {offset}, so it keeps {year} as newer data arrives"


def _one_year_offer(
    change: OneYear, read: OneYearRead, together_with: Sequence[str]
) -> dict[str, object]:
    """The `one_year` a gap of a `PERIOD_REDUCTION_UNSTATED` refusal offers: `change` on `read`, with the steps it is made together with."""
    return {
        "change": change,
        "year": read.year,
        "offset": read.offset,
        "together_with": list(together_with),
    }


def _one_year_together(together_with: Sequence[str]) -> str:
    """The sentence saying a one-year change is admitted only together with the ones offered to `together_with`, and an empty string when it names none."""
    if not together_with:
        return ""
    steps = " and ".join(repr(step) for step in together_with)
    return (
        f" Make this change together with the one-year change offered to {steps}: the plan "
        "is admitted with all of them, and refused with any one alone."
    )


def _period_reduction_unstated(
    plan: Plan, view: SessionView, *, reads_user_words: bool
) -> dict[str, object] | None:
    """Return the `PERIOD_REDUCTION_UNSTATED` refusal naming every step of `plan` that reduces across years for a result no ask across years asks for, or `None`.

    The rule is `planner.plan.unstated_period_reductions`, over the
    licences `admit` wrote on `plan`. For each such step the refusal names
    the step, the result computed from it, the asks that result answers and
    the years reduced, and says those asks do not ask for one figure over
    them.

    When another result computed from the step answers an ask across years,
    the step is asked for, and the refusal points at the unmarked asks. Each
    is answered by a result of its own that keeps the years or reads one
    year, or, for a result answering no ask, the result is listed under an
    ask across years or removed.

    Otherwise the refusal gives two ways to settle the step: keep the years
    (`_keep_the_years`), or read one year, the second only where the step
    holds a year it can read alone (`planner.plan.YearsHeld.one_year`).
    Where the planner reads the user's own words (`reads_user_words`), it
    adds a third, naming the step in `across_years` when those words ask for
    the figure.

    `detail.gaps` holds one entry per step, in the order the rule finds
    them. Its `one_year` is the change offered to read one year, with the
    `OneYear` `change`, the `year`, the `offset` from the anchor it rolls at
    (`None` for a fixed year), and `together_with`, the other steps whose
    changes the plan has to make with it (empty where it passes alone).
    `one_year` is `None` where the step is asked for and keeps its years,
    and where it holds no year it can read in a plan
    `planner.plan._one_year_offered` passes, alone or with the others.

    The asks the refusal names are recorded for the rest of the planner
    entry (`SessionView.record_reduction_refused`), where no permission to
    decline unasked covers them (`_declinable`).
    """
    gaps = unstated_period_reductions(plan)
    if not gaps:
        return None
    view.record_reduction_refused(ask for gap in gaps for ask in gap.asks)
    steps = {requirement.id: requirement for requirement in plan.evidence_requirements}
    sentences = []
    offered: list[dict[str, object] | None] = []
    kept: list[dict[str, object] | None] = []
    for gap in gaps:
        held = gap.held
        offer: dict[str, object] | None = None
        keep: dict[str, object] | None = None
        read = held.one_year
        years = held.phrase()
        step = repr(gap.step_id)
        output = repr(gap.output_id)
        result = "which" if gap.output_id == gap.step_id else f"and {output}, computed from it,"
        if gap.asks:
            answers = " and ".join(f"ask {ask} ({plan.asks[ask]!r})" for ask in gap.asks)
            unasked = (
                "That ask is not marked (across years) and does not ask"
                if len(gap.asks) == 1
                else "Those asks are not marked (across years) and do not ask"
            ) + " for one figure over those years"
        else:
            answers = "no ask"
            unasked = "No ask asks for that figure"
        reducing = steps[gap.step_id]
        if gap.asked_across and gap.asks:
            listed = "that ask's" if len(gap.asks) == 1 else "each of those asks'"
            remedies = (
                f"{step} also feeds an ask marked (across years), so keep it as it is. Answer "
                f"{answers} with a result of its own that keeps the years or reads one year, "
                f"and take {output} out of {listed} `answered` outputs."
            )
        elif gap.asked_across:
            remedies = (
                f"{step} also feeds an ask marked (across years), so keep it as it is and list "
                f"{output} in `answered` under an ask marked (across years) that it answers, "
                "or remove it."
            )
        elif isinstance(reducing, RankRequirement):
            keep = _keep_years(plan, gap.step_id, {"per_period": True})
            remedies = f"To keep the years, set `per_period` to true on {step}." if keep else ""
            if read is not None:
                offer = _one_year_offer(OneYear.WINDOW, read, gap.together_with)
                remedies += (
                    f" To read one year, set `window_start` and `window_end` on {step} to "
                    f"{read.year}, with "
                    f"{_one_year_rolling(read.offset, rolls='`rolling` naming both')}."
                    f"{_one_year_together(gap.together_with)}"
                )
            if reads_user_words:
                remedies += (
                    f" If the user's own words ask for one ranking over {years}, leave "
                    f"`per_period` false on {step} and name {step} in `across_years`."
                )
        else:
            assert isinstance(reducing, AggregateRequirement)
            kept_by = _years_kept(reducing)
            keep = _keep_years(plan, gap.step_id, {"group_by": kept_by})
            remedies = f"To keep the years, {_keep_the_years(reducing, kept_by)}." if keep else ""
            if read is not None:
                offer = _one_year_offer(OneYear.FILTER, read, gap.together_with)
                remedies += (
                    f" To read one year, have {step} read its input {reducing.input_id!r} "
                    f"through a `filter` whose `period_threshold` keeps {read.year} alone, with "
                    f"op `eq` and {_one_year_rolling(read.offset, rolls='`rolling`')}."
                    f"{_one_year_together(gap.together_with)}"
                )
            if reads_user_words:
                remedies += (
                    f" If the user's own words ask for one figure over {years}, name {step} "
                    "in `across_years`."
                )
        offered.append(offer)
        kept.append(keep)
        sentences.append(
            f"{step} reduces {years} to one figure, {result} answers {answers}. {unasked}. "
            f"{remedies.lstrip()}"
        )
    return problem(
        ProblemCode.PERIOD_REDUCTION_UNSTATED,
        " ".join(sentences),
        outcome=Outcome.INVALID_INPUT,
        detail={
            "gaps": [
                {
                    "requirement_id": gap.step_id,
                    "output_id": gap.output_id,
                    "asks": list(gap.asks),
                    "years": {"start": gap.held.years.start, "end": gap.held.years.end},
                    "follows_newest": gap.held.follows_newest,
                    "until": gap.held.until,
                    "dataset_requirement_id": gap.held.dataset_id,
                    "keep_years": keep_years,
                    "one_year": one_year,
                }
                for gap, keep_years, one_year in zip(gaps, kept, offered, strict=True)
            ]
        },
    )


def _across_years_unread(
    across_years: Sequence[str],
    *,
    requirements: Sequence[EvidenceRequirement],
    previous: Plan | None,
    reads_user_words: bool,
) -> dict[str, object] | None:
    """Return the `INVALID_PLAN` refusal of an `across_years` naming steps where the planner reads none of the user's own words, or `None`.

    There the orchestrator holds them, and its (across years) mark on an ask
    is the only statement that the user asked for one figure over the years.
    A step whose licence `previous` holds as `requirements` state it is a
    restatement, which grants nothing (`planner.plan.licences_restated`).
    """
    if reads_user_words:
        return None
    restated = licences_restated(across_years, requirements=requirements, previous=previous)
    unread = [name for name in across_years if name not in restated]
    if not unread:
        return None
    return _invalid_plan(
        "`across_years` names the steps whose years the user's own words ask to combine, "
        "and in this entry you read the orchestrator's brief, not the user's words. Take "
        f"{', '.join(map(repr, unread))} out of `across_years`. A step the plan already "
        "licenses keeps its licence without being named, and any other figure or ranking "
        "over several years answers only an ask the brief marks (across years)."
    )


def _years_kept(aggregate: AggregateRequirement) -> tuple[GroupBy, ...]:
    """The `group_by` that keeps the years of `aggregate`, whose `group_by` lacks `period`.

    It adds `period` while the group names fewer than `MAX_GROUP_BY`
    columns, an empty group included. At the cap it groups by `entity` and
    `breakdown`, and `entity` already keeps a breakdown's codes, so `period`
    takes `breakdown`'s place.
    """
    if len(aggregate.group_by) < MAX_GROUP_BY:
        return (*aggregate.group_by, GroupBy.PERIOD)
    assert set(aggregate.group_by) == {GroupBy.ENTITY, GroupBy.BREAKDOWN}
    return (GroupBy.ENTITY, GroupBy.PERIOD)


def _keep_years(plan: Plan, step_id: str, change: dict[str, object]) -> dict[str, object] | None:
    """`change` as `detail.gaps[].keep_years` states it, or `None` where the plan it gives fails.

    The check is `planner.plan.keep_years_admits`, as `one_year` is
    checked: a decision the step cites that fixes `group_by` or
    `per_period` otherwise refuses it.
    """
    if not keep_years_admits(plan, step_id, change):
        return None
    return to_jsonable_python(change)


def _keep_the_years(aggregate: AggregateRequirement, kept: tuple[GroupBy, ...]) -> str:
    """How `aggregate` keeps the years, grouped by `kept` (`_years_kept`), as its refusal says it."""
    step = repr(aggregate.id)
    if kept == (*aggregate.group_by, GroupBy.PERIOD):
        return f"add `period` to the `group_by` of {step}"
    return (
        f"group {step} by `entity` and `period`: `entity` keeps the breakdown's codes too, "
        "and a group names two columns at most"
    )


@dataclass(frozen=True)
class MeasureChoice:
    """How a flow is counted, as the decision `propose_plan` writes for a pin of one of its measures states it.

    `options` are the measure search elected, then each it offers counting
    the flow another way (`planner.workflow.MeasureAlternative`, axis
    `what`), by label, and `spec_ids` the spec_id of each in the same order.
    `chosen` is the label of the one pinned.
    """

    options: tuple[str, ...]
    spec_ids: tuple[str, ...]
    chosen: str


def measure_choices(
    shown: Mapping[str, PinSource],
    *,
    compiled: Sequence[SelectedSource] = (),
    offered: Iterable[SelectedSource] = (),
    sources: SourceReadings | None = None,
) -> dict[str, MeasureChoice]:
    """The measure choice a pin of each spec_id states, for every measure a settled source shown offers.

    Families from the prior compiled plan are retained on an amendment, and
    a source a pending referral offers in place of a pin (`offered`,
    `translator.referral.Referral.offered`) counts as settled: discovery
    found it, so its `elected_measure` is search's own. Each overrides the
    last, and a shown source with the same id takes precedence.

    A settled source whose block elects a measure (`SelectedSource.
    elected_measure`) and offers another way of counting it maps to the
    choice with its own label chosen, and each of those others to the same
    choice with that one's label chosen. A source settled only as another
    count of one (`PinSource.other_count_of`) takes its choice from the
    source that offered it, since its own block names its own measure as
    if search had elected it, and states none when that source is not
    shown. A label over `MAX_OPTION_CHARS`, a
    repeated one and any past `MAX_DECISION_OPTIONS` are left out, and a
    source left with one option offers no choice. Price alternatives (axis
    `unit`) count the flow the same way in other prices and offer no choice here.
    Each shown block is read through `sources`, the proposal's readings of
    `shown`, or readings of its own when none are given.
    """
    if sources is None:
        sources = SourceReadings.of_shown(shown)
    elections = {source.spec_id: source.elected_measure for source in (*compiled, *offered)}
    for spec_id, source in shown.items():
        if not source.settled or source.other_count_of:
            continue
        if (read := sources.get(spec_id)) is not None:
            elections[spec_id] = read.elected_measure
    choices: dict[str, MeasureChoice] = {}
    for spec_id, elected in elections.items():
        if elected is None or len(elected.label) > MAX_OPTION_CHARS:
            continue
        labels = {elected.label: spec_id}
        for alternative in elected.other_counts():
            if len(alternative.label) <= MAX_OPTION_CHARS and len(labels) < MAX_DECISION_OPTIONS:
                labels.setdefault(alternative.label, alternative.spec_id)
        if len(labels) < 2:
            continue
        options, spec_ids = tuple(labels), tuple(labels.values())
        choices[spec_id] = MeasureChoice(options=options, spec_ids=spec_ids, chosen=elected.label)
        for label, measure in labels.items():
            # A source search elected states its own choice, whatever
            # another source offers it as.
            choices.setdefault(
                measure, MeasureChoice(options=options, spec_ids=spec_ids, chosen=label)
            )
    return choices


def _repinned(
    decisions: Sequence[Decision],
    requirements: Sequence[EvidenceRequirement],
    *,
    decision_pins: Mapping[str, str],
    requirement_pins: Mapping[str, str],
    slots: Mapping[str, str] = {},
) -> tuple[list[Decision], list[EvidenceRequirement]]:
    """`decisions` and `requirements` with each source pin moved as the maps say.

    `decision_pins` maps a decision's slot, `requirement_pins` a dataset
    requirement's id, to the spec_id it pins after the move; anything either
    map does not name is returned as it was. `slots` maps the slot of a
    decision `propose_plan` wrote for a pin, a search pin's
    (`planner.plan.is_search_pin`) or a scope choice's
    (`planner.plan.is_scope_choice`), to the one its moved pin takes
    (`planner.plan.pin_slot_moves`): that decision is rewritten for its new
    pin, and every citation of it cites the new slot, so it stays one code
    wrote. A measure choice (`MeasureChoice`) keeps its options, and a scope
    choice its choice and options, since each reads the same over any period.
    """
    moved_decisions: list[Decision] = []
    for decision in decisions:
        if decision.fixes is None or decision.slot not in decision_pins:
            moved_decisions.append(decision)
            continue
        pin = decision_pins[decision.slot]
        update: dict[str, object] = {
            "fixes": decision.fixes.model_copy(update={"source_spec_id": pin})
        }
        if (is_search_pin(decision) or is_scope_choice(decision)) and decision.slot in slots:
            update["slot"] = slots[decision.slot]
        moved_decisions.append(decision.model_copy(update=update))
    moved_requirements: list[EvidenceRequirement] = []
    for requirement in requirements:
        update = {}
        if isinstance(requirement, DatasetRequirement) and requirement.id in requirement_pins:
            update["source_spec_id"] = requirement_pins[requirement.id]
        if slots.keys() & set(requirement.decision_ids):
            update["decision_ids"] = tuple(
                slots.get(slot, slot) for slot in requirement.decision_ids
            )
        moved_requirements.append(requirement.model_copy(update=update) if update else requirement)
    return moved_decisions, moved_requirements


def _unwindowed_pins(
    decisions: Sequence[Decision],
    requirements: Sequence[EvidenceRequirement],
    to_origin: Mapping[str, str],
) -> tuple[list[Decision], list[EvidenceRequirement]]:
    """`decisions` and `requirements` with every pin of a windowed copy moved back to the pin it copies.

    `to_origin` maps each copy's spec_id to the pin it copies over another
    period (`planner.plan.WindowedPin.windowed_from`).
    `propose_plan` reasons about a pin by the spec_id the planner was shown,
    whatever period an earlier call recorded it over (`_window_pins`), so a
    patch and the plan it amends always agree on what they pin.
    """
    return _repinned(
        decisions,
        requirements,
        decision_pins={
            decision.slot: origin
            for decision in decisions
            if decision.fixes is not None
            and (origin := to_origin.get(decision.fixes.source_spec_id or ""))
        },
        requirement_pins={
            requirement.id: origin
            for requirement in requirements
            if isinstance(requirement, DatasetRequirement)
            and (origin := to_origin.get(requirement.source_spec_id or ""))
        },
        slots={
            slot: moved
            for copy, origin in to_origin.items()
            for slot, moved in pin_slot_moves(copy, origin).items()
        },
    )


def _first_year_on_record(
    pin: str,
    *,
    shown: Mapping[str, PinSource],
    read: Mapping[str, FullHistory | None],
) -> FullHistory | None:
    """What is on record about every year the dataset `pin` holds; `None` when nothing is.

    Read from the Search answer the planner was shown for `pin`
    (`planner.plan.full_history`), or else from what `propose_plan` read for
    it over no window (`read`, `planner.plan.full_histories`). A first year
    outside the years a plan states (`planner.records.Year`) is none.
    """
    source = shown.get(pin)
    block = source.spec_block(pin) if source is not None else None
    history = full_history(block) if block is not None else None
    if history is None or not _is_year(history.first_year):
        history = read.get(pin)
    return history if history is not None and _is_year(history.first_year) else None


_YEAR = TypeAdapter(Year)


def _is_year(year: int) -> bool:
    """Whether `year` is one a plan can state (`planner.records.Year`)."""
    try:
        _YEAR.validate_python(year)
    except ValidationError:
        return False
    return True


def _first_years_on_record(
    requirements: Iterable[EvidenceRequirement],
    *,
    shown: Mapping[str, PinSource],
    read: Mapping[str, FullHistory | None],
) -> list[FirstYear]:
    """The first year on record for each pinned dataset requirement of `requirements` stating `period_end_open`.

    A plan reads a dataset's whole history from that year
    (`planner.plan.whole_history_split`), so a requirement gets one only
    where the frame can hold it: Search counted no more rows for every
    year than one frame may hold (data-mcp's `max_frame_rows`), or counted
    none, and then `get_data` itself refuses one too large
    (`over_ceiling`). A requirement with none that the split reads from
    its first year is refused (`_first_year_refusal`).
    """
    ceiling = DataMcpSettings().max_frame_rows
    years = []
    for requirement in requirements:
        if not isinstance(requirement, DatasetRequirement) or not requirement.period_end_open:
            continue
        pin = requirement.source_spec_id
        history = _first_year_on_record(pin, shown=shown, read=read) if pin is not None else None
        if history is None or (history.rows is not None and history.rows > ceiling):
            continue
        years.append(FirstYear(requirement_id=requirement.id, year=history.first_year))
    return years


def _first_year_unknown_cause(exc: ValueError) -> RollingPlanError | None:
    """The `OpenPeriodFault.FIRST_YEAR_UNKNOWN` refusal plan validation raised under `exc`, if it did."""
    raised: list[object] = [exc]
    if isinstance(exc, ValidationError):
        raised = [error.get("ctx", {}).get("error") for error in exc.errors(include_url=False)]
    for cause in raised:
        if (
            isinstance(cause, RollingPlanError)
            and cause.fault is OpenPeriodFault.FIRST_YEAR_UNKNOWN
        ):
            return cause
    return None


def _first_year_refusal(
    refused: RollingPlanError,
    *,
    shown: Mapping[str, PinSource],
    read: Mapping[str, FullHistory | None],
    definition_lookup: Callable[[str], RecordedDefinition | None] | None,
) -> RollingPlanError:
    """`refused`, a requirement read from a first year the plan holds none for, restated to match what the session read.

    `refused.spec_id` is the pin, as the planner named it. With none, or
    one with no identity on record (`identity_on_record`), the requirement
    names no dataset whose years can be read
    (`OpenPeriodFault.NO_SOURCE_ON_RECORD`). Otherwise the first year is
    unknown before the dataset was read over no window (`refused` as it
    is, which `planner.agent.propose_plan_reading` answers by reading it),
    `FIRST_YEAR_READ_FAILED` when that read's answer stated none (`read`
    holds the pin with `None`), and `RollingIssue.FRAME_TOO_LARGE` when the
    history is past the rows one frame may hold
    (`_first_years_on_record`).
    """
    (requirement_id,) = refused.requirement_ids
    pin = refused.spec_id
    if (
        pin is None
        or identity_on_record(pin, shown=shown, end_open=True, definition_lookup=definition_lookup)
        is None
    ):
        return RollingPlanError(
            RollingIssue.NEEDS_OPEN_PERIOD,
            f"{requirement_id!r} states `period_end_open`, which takes a dataset on "
            "record to retrieve with no end year: pin a search result's `spec_id` as its "
            "`source_spec_id`, or drop `period_end_open` and every `rolling` reading it "
            "for fixed years",
            requirement_ids=[requirement_id],
            fault=OpenPeriodFault.NO_SOURCE_ON_RECORD,
        )
    history = _first_year_on_record(pin, shown=shown, read=read)
    if history is None and pin in read:
        return RollingPlanError(
            RollingIssue.NEEDS_OPEN_PERIOD,
            f"{requirement_id!r} is read from a start that follows the newest data, and "
            f"Search's answer for {pin} over all its years states no first year, so the "
            "retrieval cannot be started there. This is a limit of what was read, not a "
            "fault in the question. Where the window covers several years, state the "
            "start as a fixed year and roll only the end; a window of one year cannot, so "
            "drop `rolling` and `period_end_open` there for fixed years",
            requirement_ids=[requirement_id],
            fault=OpenPeriodFault.FIRST_YEAR_READ_FAILED,
            spec_id=pin,
        )
    if history is None:
        return RollingPlanError(
            RollingIssue.NEEDS_OPEN_PERIOD,
            f"{requirement_id!r} is read from a start that follows the newest data, which "
            f"needs the first year {pin} holds, and no search result over all its years "
            "states one. Where the window covers several years, state the start as a "
            "fixed year and roll only the end; a window of one year cannot, so drop "
            "`rolling` and `period_end_open` there for fixed years",
            requirement_ids=[requirement_id],
            fault=OpenPeriodFault.FIRST_YEAR_UNKNOWN,
            spec_id=pin,
        )
    ceiling = DataMcpSettings().max_frame_rows
    assert history.rows is not None and history.rows > ceiling, history
    return RollingPlanError(
        RollingIssue.FRAME_TOO_LARGE,
        f"{requirement_id!r} is read from a start that follows the newest data, so it "
        f"retrieves every year {pin} holds: {history.rows:,} rows, over the {ceiling:,} "
        "one retrieval may return. Name fewer entities, or, where the window covers "
        "several years, state the start as a fixed year and roll only the end, or "
        "drop `rolling` and `period_end_open` for fixed years",
        requirement_ids=[requirement_id],
    )


def _open_window(
    requirement: DatasetRequirement, *, whole_history: Mapping[str, int]
) -> dict[str, object]:
    """The window with no end year a pinned `requirement` stating `period_end_open` is retrieved over.

    It starts at the requirement's own first year, unless the plan reads the
    dataset's whole history there (`planner.plan.HistorySplit.whole_history`,
    each such requirement by id with its dataset's first year on record):
    then it starts at that year, and the frame holds every year the dataset
    holds.
    """
    first_year = whole_history.get(requirement.id)
    if first_year is None:
        return period_window(requirement.period, end_open=True)
    return open_window(first_year)


def first_year_reads(state: SessionState) -> list[Mapping[str, Any]]:
    """The identity of each dataset the planner was shown whose first year `propose_plan`
    can still read in `state`.

    `propose_plan` reads a pinned dataset's first year
    (`planner.agent.read_full_history`) when a requirement rolls from its start and
    nothing on record states that year (`_first_year_refusal`), once per dataset. Which dataset
    it reads depends on the plan, so this is every dataset the planner was shown
    (`planner.plan.pin_sources`) that has an identity on record, has not been read, and
    whose block states no first year.
    """
    shown = pin_sources(state)
    read = full_histories(state)
    reads = []
    for pin in shown:
        if pin in read or _first_year_on_record(pin, shown=shown, read=read) is not None:
            continue
        recorded = identity_on_record(pin, shown=shown, end_open=True)
        if recorded is not None:
            reads.append(recorded.identity)
    return reads


def _window_pins(
    plan: Plan,
    *,
    shown: Mapping[str, PinSource] = {},
    definition_lookup: Callable[[str], RecordedDefinition | None] | None = None,
) -> tuple[Plan, list[WindowedPin]]:
    """`plan` with each pin whose identity is on record moved to that identity over the requirement's own period.

    A pin's identity carries the years of the search that found it, the
    window of the question that search answered. A plan's retrieval mints
    the pinned spec_id as it is (`compiler.compile`'s
    `_retrieve_arguments`). So a dataset requirement whose pin's identity is
    over another period is pinned instead to that identity over its own
    (`data_mcp.definition_writes.windowed_identity`, the spec_id `get_data`'s
    `scope.window` would mint), and so is the decision that fixes the pin.

    The identity on record is the one a search in the conversation recorded
    (`planner.plan.recorded_identity`), else the one `definition_lookup`
    answers, the discovery log's
    (`translator.discovery.DiscoveryLog.recorded_definition`) under
    `admit`. One of a `latest` window is copied
    like any other: the copy replaces the window whole, as `get_data`'s
    scope surgery does, and the translator confirms the copy through Search
    before it is retrieved.

    A requirement stating `period_end_open` is pinned to that identity over
    a range with no end year (`_open_window`), which apis ends at the
    current year at every mint, so its retrieval reaches years published
    after the plan. `shown` are the datasets the planner was shown this
    entry (`planner.plan.pin_sources`). A pinned one needs an identity on
    record (`identity_on_record`). An unpinned one is searched for from its
    first year on (`translator.discovery`), and Search answers with a
    definition that has no end year; plan validation refuses one the plan
    reads from its dataset's first year, since only a dataset on record
    has one (`_first_year_refusal`).

    Returns the moved plan and every copy it pins, each a
    `planner.plan.WindowedPin`. A copy states nothing Search stated for the
    original's years; the translator reads those from Search's answer for
    the copy when it honours the pin (`translator.discovery`).

    One decision fixing the pin of requirements over different periods
    cannot name every copy with its one `source_spec_id`. That decision
    keeps the pin it fixes, and each of those requirements its own copy,
    since the translator reads a requirement's pin and ignores its
    decision's (`translator.discovery`).

    Raises:
        RollingPlanError: If a pinned requirement stating `period_end_open`
            has no identity on record.
    """
    whole_history = split_of(plan).whole_history
    pinning: dict[str, str] = {}
    copies: dict[str, WindowedPin | None] = {}
    for requirement in plan.evidence_requirements:
        if not isinstance(requirement, DatasetRequirement):
            continue
        pin = requirement.source_spec_id
        recorded = (
            identity_on_record(
                pin,
                shown=shown,
                end_open=requirement.period_end_open,
                definition_lookup=definition_lookup,
            )
            if pin is not None
            else None
        )
        if requirement.period_end_open and pin is not None and recorded is None:
            raise RollingPlanError(
                RollingIssue.NEEDS_OPEN_PERIOD,
                f"{requirement.id!r} states `period_end_open`, which takes a dataset on "
                "record to retrieve with no end year: pin a search result's `spec_id` as its "
                "`source_spec_id`, or drop `period_end_open` and every `rolling` reading it "
                "for fixed years",
                requirement_ids=[requirement.id],
                fault=OpenPeriodFault.NO_SOURCE_ON_RECORD,
            )
        if pin is None or recorded is None:
            continue
        pinning[requirement.id] = pin
        if requirement.period_end_open:
            window = _open_window(requirement, whole_history=whole_history)
        else:
            window = period_window(requirement.period)
        if recorded.identity.get("requested_window") == window:
            copies[requirement.id] = None
        else:
            copies[requirement.id] = windowed_pin(pin, recorded, window)
    targets: dict[str, set[str]] = {}
    for requirement in plan.evidence_requirements:
        if requirement.id not in copies:
            continue
        copy = copies[requirement.id]
        target = pinning[requirement.id] if copy is None else copy.spec_id
        for slot in requirement.decision_ids:
            targets.setdefault(slot, set()).add(target)
    origins = set(pinning.values())
    decision_pins: dict[str, str] = {}
    slots: dict[str, str] = {}
    for decision in plan.decisions:
        pin = decision.fixes.source_spec_id if decision.fixes is not None else None
        if pin is None or pin not in origins:
            continue
        pinned = targets.get(decision.slot, set())
        if len(pinned) == 1 and pinned != {pin}:
            decision_pins[decision.slot] = next(iter(pinned))
            if is_search_pin(decision) or is_scope_choice(decision):
                slots[decision.slot] = pin_slot_moves(pin, decision_pins[decision.slot])[
                    decision.slot
                ]
    requirement_pins = {
        requirement_id: copy.spec_id for requirement_id, copy in copies.items() if copy is not None
    }
    if not requirement_pins and not decision_pins:
        return plan, []
    recorded_copies: dict[str, WindowedPin] = {}
    for copy in copies.values():
        if copy is not None:
            recorded_copies.setdefault(copy.spec_id, copy)
    decisions, requirements = _repinned(
        plan.decisions,
        plan.evidence_requirements,
        decision_pins=decision_pins,
        requirement_pins=requirement_pins,
        slots=slots,
    )
    moved = plan.model_copy(
        update={"decisions": tuple(decisions), "evidence_requirements": tuple(requirements)}
    )
    return Plan.model_validate(moved.model_dump()), list(recorded_copies.values())


def _measure_decisions(
    decisions: Sequence[Decision],
    evidence_requirements: Sequence[EvidenceRequirement],
    *,
    previous: Plan | None,
    measures: Mapping[str, MeasureChoice],
    user_settled: Mapping[str, UserChoice] = {},
) -> tuple[list[Decision], list[EvidenceRequirement]]:
    """Add the decision a pin of a measure a shown source offers states, and cite it.

    A pin of a spec_id in `measures` chooses how the flow is counted, so
    this writes that choice as a `Decision` on the model's behalf
    (`_measure_decision`). Its options are the measure search elected and
    the others counting the flow another way, by label, and it chooses the
    pinned one's. The user settled it (`resolved_by=user`) when the pin is
    in `user_settled`, else the planner did. Each requirement pinning such a
    spec_id cites the decision, under the slot `search_pin_slot` derives
    from the pin. A requirement still citing a decision written here for a
    pin it no longer states has that citation dropped, so a re-pin needs no
    edit of `decision_ids`.

    Only a requirement in this call's own `evidence_requirements` is
    examined, never one carried unchanged from `previous`; `previous.decisions`
    is read only to check whether an existing citation already covers the
    pin. A requirement whose citations do not yet cover the pin gets the
    citation added even when the decision's slot is already known
    elsewhere. Only the append to `decisions` is skipped then.

    Args:
        decisions: This `propose_plan` call's own decisions, before they are
            merged with any prior plan.
        evidence_requirements: Likewise, this call's own requirements.
        previous: The plan this call amends, or `None` on a first draft or a
            restart.
        measures: The measure choice a pin of each spec_id states
            (`measure_choices`).
        user_settled: The spec_ids of `measures` whose measure the user
            chose, each with how they chose it (`_user_settled_measures`).

    Returns:
        `decisions` with each new measure decision appended, and
        `evidence_requirements` with each pin of a measure citing its
        decision.
    """
    known = {decision.slot: decision for decision in (previous.decisions if previous else ())}
    known.update({decision.slot: decision for decision in decisions})

    def _uncited_pin(requirement: EvidenceRequirement) -> str | None:
        if not isinstance(requirement, DatasetRequirement):
            return None
        pin = requirement.source_spec_id
        if pin is None or pin not in measures:
            return None
        for slot in requirement.decision_ids:
            cited = known.get(slot)
            if (
                cited is not None
                and cited.fixes is not None
                and cited.fixes.source_spec_id == pin
                and not is_scope_choice(cited)
            ):
                return None
        return pin

    def _stale_citation(requirement: EvidenceRequirement, slot: str) -> bool:
        """Whether `slot` is a decision this function wrote for a pin `requirement` no longer states."""
        cited = known.get(slot)
        return (
            isinstance(requirement, DatasetRequirement)
            and cited is not None
            and is_search_pin(cited)
            and cited.fixes is not None
            and cited.fixes.source_spec_id != requirement.source_spec_id
        )

    written: list[Decision] = []
    patched_requirements: list[EvidenceRequirement] = []
    for stated in evidence_requirements:
        kept = tuple(slot for slot in stated.decision_ids if not _stale_citation(stated, slot))
        requirement = (
            stated
            if kept == stated.decision_ids
            else stated.model_copy(update={"decision_ids": kept})
        )
        pin = _uncited_pin(requirement)
        if pin is None:
            patched_requirements.append(requirement)
            continue
        slot = search_pin_slot(pin)
        patched_requirements.append(
            requirement.model_copy(update={"decision_ids": (*requirement.decision_ids, slot)})
        )
        if slot in known:
            continue
        known[slot] = _measure_decision(slot, pin, measures[pin], user=user_settled.get(pin))
        written.append(known[slot])
    return [*decisions, *written], patched_requirements


class UserChoice(StrEnum):
    """How the user chose a measure: `PICKED` answering an ask that offered
    it, `NAMED` in the words of their question (`compiler.sources.
    measure_named_by_query`)."""

    PICKED = "picked"
    NAMED = "named"


def _measure_decision(
    slot: str, pin: str, choice: MeasureChoice, *, user: UserChoice | None
) -> Decision:
    """The decision `_measure_decisions` writes for a pin of `pin`, one
    measure of `choice`; `user` says how the user chose it, `None` when
    they did not."""
    elected = choice.options[0]
    if user is UserChoice.PICKED:
        rationale = f"The user chose {choice.chosen}."
    elif user is UserChoice.NAMED:
        rationale = f"The user's question named {choice.chosen}."
    elif choice.chosen == elected:
        rationale = f"Search elected {elected} for this source."
    else:
        rationale = f"Chosen in place of {elected}, the measure search elected."
    return Decision(
        slot=slot,
        description="How the flow is counted.",
        options=choice.options,
        chosen=choice.chosen,
        resolved_by=ResolvedBy.PLANNER_DEFAULT if user is None else ResolvedBy.USER,
        rationale=rationale,
        fixes=DecisionFixes(source_spec_id=pin),
    )


def _user_settled_measures(
    measures: Mapping[str, MeasureChoice], view: SessionView, shown: Mapping[str, PinSource]
) -> dict[str, UserChoice]:
    """The spec_ids of `measures` whose measure the user chose, with how.

    The user picked one answering an `adk_request_input` call: the
    orchestrator's `ask_user` offers each measure under its own spec_id, so
    a pick names it (`SessionView.elected_spec_ids`). A typed answer or an
    amendment at approval names no spec_id, so the measure it leads to
    stays the planner's. The user's question named the measure of a
    settled source in `shown` whose block says so
    (`compiler.sources.measure_named_by_query`); a source settled as
    another count of one (`PinSource.other_count_of`) names its own measure
    in its block, so its block says nothing of the question."""
    chosen = dict.fromkeys(view.elected_spec_ids() & measures.keys(), UserChoice.PICKED)
    for spec_id, source in shown.items():
        if spec_id in chosen or spec_id not in measures:
            continue
        if not source.settled or source.other_count_of is not None:
            continue
        block = source.spec_block(spec_id)
        if block is not None and measure_named_by_query(block):
            chosen[spec_id] = UserChoice.NAMED
    return chosen


def _user_settled_rewrites(
    previous: Plan | None,
    decisions: Sequence[Decision],
    requirements: Sequence[EvidenceRequirement],
    *,
    measures: Mapping[str, MeasureChoice],
    user_settled: Mapping[str, UserChoice],
) -> list[Decision]:
    """Each measure decision `previous` records as the planner's that the user has since settled, as theirs.

    `_measure_decisions` writes a measure decision when a
    requirement first pins the measure, so a pick answering an ask raised
    after that leaves the planner's decision standing on the same pin. Only
    a decision some requirement of the patched plan (`requirements`) cites,
    and that this call's `decisions` do not restate, is rewritten.
    """
    if previous is None:
        return []
    restated = {decision.slot for decision in decisions}
    cited = {slot for requirement in requirements for slot in requirement.decision_ids}
    rewritten: list[Decision] = []
    for decision in previous.decisions:
        pin = decision.fixes.source_spec_id if decision.fixes is not None else None
        choice = measures.get(pin) if pin is not None else None
        if (
            pin is not None
            and choice is not None
            and (user := user_settled.get(pin)) is not None
            and is_search_pin(decision)
            and decision.resolved_by is ResolvedBy.PLANNER_DEFAULT
            and decision.slot in cited
            and decision.slot not in restated
        ):
            rewritten.append(_measure_decision(decision.slot, pin, choice, user=user))
    return rewritten


# Why the planner moved an input's choice on each axis it aligns (`_aligned_bases`).
_ALIGNED_REASONS: dict[str, str] = {
    "measure_basis": "so that every input of the calculation is on one basis",
    "provider": "so that every input of the calculation counts the same spending",
}

# The rationale of a scope decision Search's default settled. The planner
# reads it. The plan card hides the reason of such a decision
# (`DecisionScope.provenance`), since it says only that nobody chose
# (`harness.studio_translate`).
SEARCH_DEFAULT_RATIONALE = "Search's default for this question."

# What each scope decision asks, by the axis its choice settles.
_SCOPE_QUESTIONS: dict[str, str] = {
    "sector": "Which sector codes the figures count.",
    "region": "Which places the region's figures count.",
    "measure_basis": "Which basis the figures are measured on.",
    "provider": "Whose spending the figures count.",
}


def scope_choice_decisions(
    shown: Mapping[str, PinSource], *, wholes: Collection[str] = ()
) -> dict[str, tuple[Decision, ...]]:
    """The scope decisions a pin of each spec_id of `shown` states, in the order Search lists its choices.

    Read from the source's spec block (`resolved.scope_choices`,
    `harness.scope_choices.PartScope`). A block carrying none, or one this
    release cannot read, states none. Each choice is one decision
    (`_scope_decision`), when it states one; a choice on an axis the block
    marks as the planner's (`harness.scope_choices.PLANNER_AXES_BLOCK_KEY`)
    is the planner's decision. The pin of a declared whole (`wholes`, by
    spec_id, `_whole_only_pins`) states none. Its choices hold its part's
    options, so the part's rows move both, and any one option on the axis
    it leaves open would count a single slice of what the whole counts in
    full. A pin a plain requirement shares with a declared whole is no
    whole's alone, and states its rows.
    """
    decisions: dict[str, tuple[Decision, ...]] = {}
    for spec_id, source in shown.items():
        if spec_id in wholes:
            continue
        block = source.spec_block(spec_id)
        scope = part_scope(spec_id, block)
        if scope is None:
            continue
        planner = planner_axes(block)
        written = tuple(
            decision
            for choice in scope.choices
            if (decision := _scope_decision(spec_id, choice, scope=scope, planner=planner))
            is not None
        )
        if written:
            decisions[spec_id] = written
    return decisions


def part_scope(spec_id: str, block: Mapping[str, Any] | None) -> PartScope | None:
    """The scope choices `block`, the spec block of `spec_id`, carries; `None` when it carries none or none this release reads."""
    raw = ((block or {}).get("resolved") or {}).get("scope_choices")
    if raw is None:
        return None
    try:
        return PartScope.from_search(raw)
    except ValidationError as exc:
        logger.warning("scope choices unreadable: spec_id=%s: %s", spec_id, exc)
        return None


def _scope_decision(
    pin: str, choice: ScopeChoice, *, scope: PartScope, planner: Collection[str] = frozenset()
) -> Decision | None:
    """The decision a pin of `pin` states for `choice`, or `None` when it states none.

    It states one when another option than the active one can be served,
    whole or for some years (`available` or `partial`), or is blocked only
    by another choice (`conflicts_with`), which the user can reach by
    changing that choice. Its options are the
    choice's labels, unavailable ones included, so the card can say why
    each is out of reach. A choice a `Decision` cannot state, its labels
    repeated or past a limit of the plan's, states none. The user settled it
    when an explicit or carried entry of `scope` did: an explicit one from a
    choice (`ChoiceOrigin`) is a switch on the plan card. Else Search's
    default did. A choice on an axis of `planner`, the axes the planner's
    own words or alignment settled (`harness.scope_choices.PLANNER_AXES_BLOCK_KEY`),
    is the planner's: a switch of the measure basis or provider scope there
    compiled the pin on the option of a calculation's other inputs
    (`_aligned_bases`), and is stated whatever else the pin could be served on.
    """
    active = choice.active_option
    switched = any(
        entry.axis == choice.axis and entry.origin.kind == "choice" for entry in scope.entries
    )
    by_planner = choice.axis in planner and choice.provenance != "default"
    aligned = by_planner and switched and choice.axis in _ALIGNED_REASONS
    if active is None or not (
        aligned
        or any(
            option.availability.status != "unavailable"
            or option.availability.code == "conflicts_with"
            for option in choice.options
            if option.option_id != choice.active
        )
    ):
        return None
    if aligned:
        rationale = f"Counted as {active.label} {_ALIGNED_REASONS[choice.axis]}."
    elif by_planner:
        rationale = f"The plan's wording of this input asked for {active.label}."
    elif choice.provenance == "explicit" and switched:
        rationale = f"The user chose {active.label} on the plan."
    elif choice.provenance == "explicit":
        rationale = f"The user's question asked for {active.label}."
    elif choice.provenance == "carried":
        rationale = f"The user chose {active.label} earlier in the conversation."
    else:
        rationale = SEARCH_DEFAULT_RATIONALE
    try:
        return Decision(
            slot=scope_choice_slot(pin, choice.axis),
            description=_SCOPE_QUESTIONS[choice.axis],
            options=tuple(option.label for option in choice.options),
            chosen=active.label,
            resolved_by=ResolvedBy.PLANNER_DEFAULT
            if by_planner or choice.provenance == "default"
            else ResolvedBy.USER,
            rationale=rationale,
            fixes=DecisionFixes(source_spec_id=pin),
            scope=DecisionScope(
                choice_id=choice.id,
                axis=choice.axis,
                provenance=choice.provenance,
                options=tuple(
                    ScopeOptionState(
                        option_id=option.option_id,
                        status=option.availability.status,
                        years=option.availability.years,
                        code=option.availability.code,
                    )
                    for option in choice.options
                ),
            ),
        )
    except ValidationError as exc:
        logger.warning("scope decision not stated: spec_id=%s axis=%s: %s", pin, choice.axis, exc)
        return None


def _scope_decisions(
    decisions: Sequence[Decision],
    evidence_requirements: Sequence[EvidenceRequirement],
    *,
    previous: Plan | None,
    scopes: Mapping[str, Sequence[Decision]],
    wholes: Mapping[str, Sequence[Decision]],
    calculations: Mapping[str, Sequence[Decision]],
    removed: Collection[str],
) -> tuple[list[Decision], list[EvidenceRequirement]]:
    """Add the scope decisions each pin in this call's requirements states, and cite them.

    Runs after the model's decisions and the measure decisions
    (`_measure_decisions`), so those come first under the cap. Each dataset
    requirement of `evidence_requirements` pinning a spec_id of `scopes`
    cites that pin's scope decisions; one citing a scope decision, or a
    whole's (`_whole_decisions`), for a pin it does not state has that
    citation dropped, as does a declared whole (`DatasetRequirement.whole_of`)
    citing a scope decision and a requirement declaring none citing a
    whole's. Each calculation of `calculations`, this call's or
    one `previous` carries, cites the decisions it states
    (`_difference_decisions`), restated where their words changed,
    and one citing such a decision it no longer states has that citation
    dropped. A carried calculation whose citations change is patched. A
    decision whose slot is
    new is written while the plan holds fewer than `MAX_PLAN_DECISIONS`;
    past that it is neither written nor cited, and the basis line still
    states its choice. A decision qualifying the answer (`Decision.caveat`)
    is always written and is not counted, as `Plan` counts it. The count is
    every slot without a caveat this call states and every
    slot of `previous` it does not remove that stays: one a requirement
    still cites once this call's citations are patched, the scope decisions
    of this call's pins among them, or a user's, which
    `_uncited_carried_decisions` may keep. A planner's decision no
    requirement cites any more, such as a re-pinned dataset's scope
    decision, is swept and not counted. The count never falls short of the
    plan's own, so the cap never refuses. One `previous` records as Search's
    default is restated when Search now says otherwise, unless this call
    states its slot; a user's stands. Code writes the decisions of `wholes`
    and `calculations`, so a decision of this call's on one of their slots is
    dropped and code's is written or kept.

    Args:
        decisions: This call's decisions, the measure decisions included.
        evidence_requirements: This call's own requirements.
        previous: The plan this call amends, or `None`.
        scopes: The scope decisions a pin of each spec_id states
            (`scope_choice_decisions`).
        wholes: The decision each whole compiled from its part states, by
            requirement id (`_whole_decisions`).
        calculations: The provider-scope and measure-basis decisions each
            comparison, share, quotient or sum states, by requirement id, empty
            for one stating none (`_difference_decisions`).
        removed: The slots this call removes.

    Returns:
        `decisions` with each new or restated scope decision appended, and
        `evidence_requirements` citing them, with each carried calculation
        whose citations changed.
    """
    # Code writes the decisions qualifying the answer, so a draft of the
    # planner's restating one of their slots gives way to code's.
    derived_slots = {
        derived.slot
        for stated_by in (*wholes.values(), *calculations.values())
        for derived in stated_by
    }
    decisions = [decision for decision in decisions if decision.slot not in derived_slots]
    carried = previous.decisions if previous is not None else ()
    known = {decision.slot: decision for decision in (*carried, *decisions)}
    stated = {decision.slot for decision in decisions}
    restated = {requirement.id for requirement in evidence_requirements}
    # A calculation the plan carries states its decisions again too, since an
    # input re-pinned under it can change what they say.
    own_count = len(evidence_requirements)
    evidence_requirements = [
        *evidence_requirements,
        *(
            requirement
            for requirement in (previous.evidence_requirements if previous is not None else ())
            if requirement.id in calculations and requirement.id not in restated
        ),
    ]
    citations = [
        [
            slot
            for slot in requirement.decision_ids
            if not (
                (decision := known.get(slot)) is not None
                and (is_scope_choice(decision) or decision.caveat == "derived_whole")
                and decision.fixes is not None
                and (
                    decision.fixes.source_spec_id != requirement.source_spec_id
                    # A declared whole states no scope row, and only a declared
                    # whole states the whole's decision.
                    or (decision.caveat == "derived_whole") == (requirement.whole_of is None)
                )
            )
        ]
        if isinstance(requirement, DatasetRequirement)
        else [
            slot
            for slot in requirement.decision_ids
            if (decision := known.get(slot)) is None
            or decision.caveat not in _CALCULATION_CAVEATS
            or slot in {derived.slot for derived in calculations.get(requirement.id, ())}
        ]
        for requirement in evidence_requirements
    ]

    def derived_for(requirement: EvidenceRequirement) -> Sequence[Decision]:
        if not isinstance(requirement, DatasetRequirement):
            return calculations.get(requirement.id, ())
        pin = requirement.source_spec_id
        own = scopes.get(pin, ()) if pin is not None and requirement.whole_of is None else ()
        return (*own, *wholes.get(requirement.id, ()))

    cited_after = {
        *(slot for cited in citations for slot in cited),
        *(
            slot
            for requirement in (previous.evidence_requirements if previous is not None else ())
            if requirement.id not in restated
            for slot in requirement.decision_ids
        ),
        *(
            derived.slot
            for requirement in evidence_requirements
            for derived in derived_for(requirement)
        ),
    }
    held = stated | {
        decision.slot
        for decision in carried
        if decision.slot not in removed
        and decision.caveat is None
        and (decision.slot in cited_after or decision.resolved_by is not ResolvedBy.PLANNER_DEFAULT)
    }
    written: list[Decision] = []
    patched: list[EvidenceRequirement] = []
    for index, (requirement, cited) in enumerate(
        zip(evidence_requirements, citations, strict=True)
    ):
        for derived in derived_for(requirement):
            standing = known.get(derived.slot)
            if standing is None:
                if derived.caveat is None:
                    if len(held) >= MAX_PLAN_DECISIONS:
                        continue
                    held.add(derived.slot)
                known[derived.slot] = derived
                written.append(derived)
            elif (
                standing != derived
                and standing.resolved_by is ResolvedBy.PLANNER_DEFAULT
                and derived.slot not in stated
            ):
                known[derived.slot] = derived
                written.append(derived)
            if derived.slot not in cited:
                cited.append(derived.slot)
        if tuple(cited) != requirement.decision_ids:
            patched.append(requirement.model_copy(update={"decision_ids": tuple(cited)}))
        elif index < own_count:
            patched.append(requirement)
    return [*decisions, *written], patched


# The session slot holding each pin compiled to put a calculation's inputs
# on one measure basis or provider scope (`AlignedPin`), which `_aligned_bases` reads and
# `planner.agent.with_source_reads` writes.
ALIGNED_PINS_STATE_KEY = "planner:aligned_pins"

# The identity slots whose bound code makes a part a sector or restricted
# one: the CRS purpose code, type of finance and cooperation modality. They
# mirror the slots dfo-apis reads a part's shape from for its basis
# convention (`search.scope.basis.identity_part_shape` there).
_NARROWING_SLOTS = frozenset({"crsPurpose", "crsTypeOfFinance", "crsCooperationModality"})

_BASIS_AXIS = "measure_basis"
# The graph type of a country, which Search never marks a multilateral (`_own_flows`).
_COUNTRY_TYPE = "Country"
# What a sum's double-count refusal calls each money two inputs can both count
# (`_shared_money`).
_MONEY_WORDS: Mapping[ProviderMoney, str] = {
    "bilateral": "the donor's bilateral flows",
    "multilateral": "the donor's aid through multilaterals",
}

# The axes a calculation's inputs are counted on one option of (`_aligned_bases`),
# and what figures counted on different options of each mix, or of either for an
# input with no pin to name an option on.
_ALIGNED_AXES = (_BASIS_AXIS, PROVIDER_AXIS)
_MIXED: dict[str | None, str] = {
    _BASIS_AXIS: "counted on different measure bases",
    PROVIDER_AXIS: "counting different providers' spending",
    None: "counted on different measure bases or provider scopes",
}

# The calculations whose inputs are counted on one provider scope: those that
# subtract, divide or add their inputs. A join sets its inputs side by side for
# display (`_aligns`).
_PROVIDER_ALIGNED = (
    ComparisonRequirement,
    ShareOfRequirement,
    QuotientRequirement,
    SumOfRequirement,
)

# The axis a part's sector is chosen on, which names a share's side for the
# reader (`_side_name`).
_SECTOR_AXIS = "sector"


class AlignedPin(StrictModel):
    """A pin compiled from the pin `origin` on the options `option_ids`, which Search answered as `spec_id`.

    `option_ids` holds one option per axis moved, measure basis or provider
    scope, sorted (`Realignment.option_ids`). `digest` is Search's answer
    as a search's digest (`harness.tool_contracts.resolver_digest`), which
    the planner seam stores as a reading of the conversation.

    A declared whole (`whole_axis` set, `DatasetRequirement.whole_of`) is
    the part pinned to `origin` with its choice on `whole_axis` left open
    and every other choice on the part's options, and moves no option
    (`option_ids` empty, `Realignment.whole_axis`).
    """

    origin: str
    option_ids: tuple[str, ...]
    spec_id: str
    digest: dict[str, Any]
    whole_axis: str | None = None

    @property
    def key(self) -> tuple[str, tuple[str, ...], str | None]:
        """What this pin was compiled for: its origin, its options and the axis it leaves open."""
        return self.origin, self.option_ids, self.whole_axis


def aligned_pins(state: SessionState) -> tuple[AlignedPin, ...]:
    """The pins `ALIGNED_PINS_STATE_KEY` holds, oldest first; empty when unset."""
    return tuple(AlignedPin.model_validate(raw) for raw in state.get(ALIGNED_PINS_STATE_KEY) or ())


def save_aligned_pin(state: SessionState, pin: AlignedPin) -> None:
    """Hold `pin`, in place of any pin compiled for the same thing (`AlignedPin.key`)."""
    kept = [held for held in aligned_pins(state) if held.key != pin.key]
    state[ALIGNED_PINS_STATE_KEY] = [held.model_dump(mode="json") for held in (*kept, pin)]


# The session slot holding each search code made for a calculation's input
# the planner pinned no dataset for (`SearchedPin`), which `_aligned_bases`
# reads and `planner.agent.with_source_reads` writes.
SEARCHED_PINS_STATE_KEY = "planner:searched_pins"


class SearchedPin(StrictModel):
    """Search's answer for a dataset requirement the planner pinned no dataset for.

    `requirement_key` is the requirement's searched content
    (`translator.discovery.requirement_revision`), so a requirement restated
    with other words, parties or years is searched again. `spec_id` is the
    dataset Search settled on and `digest` its answer as a search's digest
    (`harness.tool_contracts.resolver_digest`), which the planner seam stores
    as a reading of the conversation; both are `None` when Search settled on
    no one dataset, and `_aligned_bases` then refuses the plan. `status` is
    the status Search answered the part with (`definite`, `ambiguous`,
    `candidates`, `no_data`), `refused` for a refusal that cannot succeed
    later, or `unreadable` for an answer holding no one part. `asked_roles` and `open_roles` are the parties the search named
    and held open.
    """

    requirement_key: str
    status: str
    spec_id: str | None = None
    digest: dict[str, Any] | None = None
    asked_roles: tuple[str, ...] = ()
    open_roles: tuple[str, ...] = ()


def searched_pins(state: SessionState) -> tuple[SearchedPin, ...]:
    """The searches `SEARCHED_PINS_STATE_KEY` holds, oldest first; empty when unset."""
    return tuple(
        SearchedPin.model_validate(raw) for raw in state.get(SEARCHED_PINS_STATE_KEY) or ()
    )


def save_searched_pin(state: SessionState, pin: SearchedPin) -> None:
    """Hold `pin`, in place of any search of the same requirement content."""
    kept = [held for held in searched_pins(state) if held.requirement_key != pin.requirement_key]
    state[SEARCHED_PINS_STATE_KEY] = [held.model_dump(mode="json") for held in (*kept, pin)]


class Unpinned(StrictModel):
    """One input of a calculation the planner pinned no dataset for, to search for in code.

    `requirement_id` is the dataset requirement and `step_id` the
    calculation whose inputs it joins. `requirement_key` is its searched
    content (`translator.discovery.requirement_revision`), `part` the
    `search` part the translator's own discovery sends for it
    (`translator.discovery.search_part`), and `asked_roles` and `open_roles`
    the parties that part names and holds open.
    """

    requirement_id: str
    step_id: str
    requirement_key: str
    part: dict[str, Any]
    asked_roles: tuple[str, ...]
    open_roles: tuple[str, ...]


def _unpinned(requirement: DatasetRequirement, step_id: str) -> Unpinned:
    """The search for `requirement`, an unpinned input of `step_id`."""
    return Unpinned(
        requirement_id=requirement.id,
        step_id=step_id,
        requirement_key=requirement_revision(requirement),
        part=search_part(requirement, requirement.measure),
        asked_roles=tuple(entity.role for entity in requirement.entities if entity.role),
        open_roles=tuple(requirement_open_roles(requirement)),
    )


def unpinned_wanted(result: Mapping[str, object]) -> list[Unpinned]:
    """The inputs a `BASIS_UNKNOWN` refusal asks to search for; empty for any other answer."""
    if result.get("code") != ProblemCode.BASIS_UNKNOWN.value:
        return []
    detail = cast("Mapping[str, Any]", result.get("detail") or {})
    return [Unpinned.model_validate(entry) for entry in detail.get("search") or ()]


class Realignment(NamedTuple):
    """One input a calculation needs compiled on another measure basis, provider scope, or both.

    `spec_id` is the pin as Search first answered it, `option_ids` the
    options to compile it on, one per axis it moves on, sorted so one set
    of options has one form. `requirement_id` is the dataset requirement
    pinning it and `step_id` the calculation that wants its first option.

    With `whole_axis` set, the input is a declared whole
    (`DatasetRequirement.whole_of`): `spec_id` is its part's pin, compiled
    with its choice on `whole_axis` left open, `option_ids` is empty, and
    `requirement_id` and `step_id` both name the whole (`_declared_wholes`).
    """

    spec_id: str
    option_ids: tuple[str, ...]
    requirement_id: str
    step_id: str
    whole_axis: str | None = None

    def detail(self) -> dict[str, object]:
        """This realignment as a refusal's detail states it, `whole_axis` only when set."""
        detail: dict[str, object] = {**self._asdict(), "option_ids": list(self.option_ids)}
        if self.whole_axis is None:
            del detail["whole_axis"]
        return detail


def realignments_wanted(result: Mapping[str, object]) -> list[Realignment]:
    """The inputs a `BASES_UNALIGNED` refusal asks to compile on other options; empty for any other answer."""
    if result.get("code") != ProblemCode.BASES_UNALIGNED.value:
        return []
    detail = cast("Mapping[str, Any]", result.get("detail") or {})
    return [
        Realignment(
            spec_id=entry["spec_id"],
            option_ids=tuple(entry["option_ids"]),
            requirement_id=entry["requirement_id"],
            step_id=entry["step_id"],
            whole_axis=entry.get("whole_axis"),
        )
        for entry in detail.get("realign") or ()
    ]


@dataclass(frozen=True)
class _InputBasis:
    """The option one dataset requirement's pin is counted on, on one aligned axis.

    `pin` is the spec_id the requirement states and `origin` the pin it was
    compiled from (itself, unless a proposal of this entry put it on another
    option). `choice` is the origin's choice on the axis (`_ALIGNED_AXES`), so
    an option the user stated reads as theirs whatever the pin was aligned
    to. `stated` says the user stated it: an explicit entry settled the
    choice, or a carried one, the user's choice earlier in the conversation,
    and the origin's block does not mark the axis as the planner's
    (`harness.scope_choices.PLANNER_AXES_BLOCK_KEY`), or Search settled the
    provider row on the words it searched (`_provider_read_in_words`). `held`
    is the origin as alignment reads it on every axis (`_Origin`), and `reads`
    what the requirement reads of it (`_Reads`), which alignment sees before
    `_window_pins` puts each pin on its own period, so two inputs holding one
    `held.part`, one option on every axis and one `reads` read one series.
    """

    requirement_id: str
    pin: str
    origin: str
    choice: ScopeChoice
    stated: bool
    held: _Origin
    reads: _Reads

    def label(self, option_id: str) -> str:
        """The reader's label of `option_id` on this input, or the id when the choice lists none."""
        return next(
            (option.label for option in self.choice.options if option.option_id == option_id),
            option_id,
        )


class _Reads(NamedTuple):
    """What one dataset requirement reads of the part its pin compiles to (`_InputBasis.reads`).

    `period` is the requirement's period and whether its end is open.
    `entities` holds each entity it names as the refs of the members its pin's
    source resolves the name to (`compiler.evidence.named_members`), so
    "United States" and "United States of America" bound to one ref read
    alike, or as its case-folded name where the source resolves none.
    `across`, `breakdown` and `stat_vars` are what else shapes the frame its
    retrieval carries (`compiler.compile._retrieve_arguments`): the sides it
    keeps open, the classification it splits the total by, and how it reads a
    family of StatVars, as the compiler settles it (`stat_var_reading`), so
    each recipient's aid and the recipients' total read different series.
    """

    period: tuple[str, bool]
    entities: frozenset[frozenset[tuple[str, str]] | str]
    across: frozenset[str]
    breakdown: Breakdown | None
    stat_vars: StatVarReading | None

    @classmethod
    def of(cls, requirement: DatasetRequirement, source: SelectedSource | None) -> _Reads:
        """What `requirement` reads, its names resolved against `source`, its pin's
        origin's source, `None` where its block is not held or not readable."""
        return cls(
            period=(requirement.period.model_dump_json(), requirement.period_end_open),
            entities=frozenset(
                frozenset((member.role, member.ref) for member in members)
                if source is not None and (members := named_members(entity, source))
                else entity.name.casefold()
                for entity in requirement.entities
            ),
            across=frozenset(requirement.across),
            breakdown=requirement.breakdown,
            stat_vars=_reading(requirement, source),
        )


def _reading(
    requirement: DatasetRequirement, source: SelectedSource | None
) -> StatVarReading | None:
    """How a retrieval for `requirement` reads its StatVars, `source` its pin's source:
    as the compiler settles it (`compiler.sources.stat_var_reading`), a breakdown read
    separately, and as the plan states it where no source is held."""
    requested = (
        StatVarReading.SEPARATE if requirement.breakdown is not None else requirement.stat_vars
    )
    return requested if source is None else stat_var_reading(source, requested).stat_var_reading


@dataclass(frozen=True)
class _Flows:
    """Where one part's flows lie, as its scope record binds them.

    `donors` and `recipients` are the parties the part binds on its `origin`
    and `destination` roles (`bound_parties`), empty for a part counting every
    one. `slots` are the codes each summable slot the part binds admits, the
    codes the code index widened its bound codes to included, by constraint
    property (`compiler.sources.slot_codes`), a slot whose codes are unknown
    mapping to `None`, and empty where the block lists no readable slots.
    Search marks a slot summable (`Slot.summable`) where its code system gives
    each record one code and sibling codes add up to their parent, so two
    summable bindings sharing no code count no record in common. Other slots,
    such as DAC1's aid type, where the total line holds the bilateral one, or
    the netting basis, which counts one flow gross and net, are left out.
    `years` are the first and last year its identity requests, the last
    `None` for a window open at its end, and `years` is `None` for a window
    stating no range.
    """

    donors: frozenset[str]
    recipients: frozenset[str]
    slots: Mapping[str, frozenset[str] | None]
    years: tuple[int, int | None] | None

    @classmethod
    def of(cls, scope: PartScope, block: Mapping[str, Any] | None) -> _Flows:
        """The flows of the part `scope` records, `block` its spec block."""
        resolved = (block or {}).get("resolved") or {}
        summable = {
            slot["key"]["property"].get("ref")
            for slot in resolved.get("slots") or ()
            if isinstance(slot, Mapping)
            and slot.get("summable") is True
            and isinstance(slot.get("key"), Mapping)
            and isinstance(slot["key"].get("property"), Mapping)
        }
        codes = slot_codes(resolved) or {}
        window = scope.base.identity.get("requested_window") or {}
        start, end = window.get("start_year"), window.get("end_year")
        return cls(
            donors=frozenset(bound_parties(scope, "origin")),
            recipients=frozenset(bound_parties(scope, "destination")),
            slots={
                slot: frozenset(refs) if refs is not None else None
                for slot, refs in codes.items()
                if slot in summable
            },
            years=(start, end)
            if window.get("kind") == "range" and isinstance(start, int)
            else None,
        )

    def meet(self, other: _Flows) -> bool:
        """Whether one flow can lie in both parts: no axis keeps them apart.

        Two parts are apart on donors or recipients when both bind countries
        alone with none in common (`compiler.sources.countries_apart`), on a
        summable slot both bind to known codes sharing none, as grants and
        loans are, and on years when both request ranges that share none.
        Anything a part leaves open or does not state, and a slot whose codes
        are unknown, keeps them together.
        """
        if countries_apart(self.donors, other.donors) or countries_apart(
            self.recipients, other.recipients
        ):
            return False
        for slot, codes in self.slots.items():
            theirs = other.slots.get(slot)
            if codes is not None and theirs is not None and codes.isdisjoint(theirs):
                return False
        if self.years is None or other.years is None:
            return True
        (start, end), (other_start, other_end) = self.years, other.years
        return (end is None or other_start <= end) and (other_end is None or start <= other_end)


def _part_rank(identity: Mapping[str, Any]) -> int:
    """How narrow a part of `identity` is: 2 when it binds a sector or restriction
    (`_NARROWING_SLOTS`), 1 when it binds a recipient, 0 for a donor-only total."""
    bound = {
        slot.get("property")
        for slot in identity.get("slots") or ()
        if isinstance(slot, Mapping)
        and (slot.get("binding") or {}).get("state") in ("value", "set")
    }
    if bound & _NARROWING_SLOTS:
        return 2
    return 1 if (identity.get("entities") or {}).get("destination") else 0


@dataclass(frozen=True)
class _Origin:
    """A pin's origin as alignment reads it on every axis (`_input_bases`), parsed once
    per proposal: its spec block, `None` where it was not shown, and the scope
    record the block states (`part_scope`).

    `rank` is how narrow its part is (`_part_rank`), `flows` where the part's
    flows lie (`_Flows`) and `own_flows` the money its reporters' own flows count
    (`_own_flows`), read on first use, since only the provider axis reads it.
    `part` is the spec_id of the part before any choice (`PartScope.base`) and
    `actives` the option each of its choices holds, by axis. The two together
    are what the origin compiles to.
    """

    block: Mapping[str, Any] | None
    scope: PartScope

    @functools.cached_property
    def rank(self) -> int:
        return _part_rank(self.scope.base.identity)

    @functools.cached_property
    def flows(self) -> _Flows:
        return _Flows.of(self.scope, self.block)

    @functools.cached_property
    def own_flows(self) -> frozenset[ProviderMoney]:
        return _own_flows(self.scope, self.block)

    @property
    def part(self) -> str:
        return self.scope.base.spec_id

    @functools.cached_property
    def actives(self) -> tuple[tuple[str, str], ...]:
        return tuple((choice.axis, choice.active) for choice in self.scope.choices)


class _Input(NamedTuple):
    """One dataset requirement's pin as alignment reads it on every axis (`_read_inputs`):
    the spec_id it states, the `origin` it was compiled from, that origin as read
    (`held`), `None` where it carries no choice on an aligned axis
    (`_ALIGNED_AXES`), and what the requirement reads of it (`reads`)."""

    pin: str
    origin: str
    held: _Origin | None
    reads: _Reads


def _read_inputs(
    requirements: Sequence[EvidenceRequirement],
    *,
    sources: SourceReadings,
    origins: Mapping[str, str],
) -> dict[str, _Input]:
    """Each pinned dataset requirement among `requirements`, its pin's origin
    (`origins`) as `_input_bases` reads it on every axis, by requirement id. Each
    origin is parsed once, its block and source read off `sources`."""
    held: dict[str, _Origin | None] = {}
    inputs: dict[str, _Input] = {}
    for requirement in requirements:
        if not isinstance(requirement, DatasetRequirement) or requirement.source_spec_id is None:
            continue
        pin = requirement.source_spec_id
        origin = origins.get(pin, pin)
        if origin not in held:
            block = sources.block(origin)
            scope = part_scope(origin, block)
            held[origin] = (
                _Origin(block=block, scope=scope)
                if scope is not None
                and any(choice.axis in _ALIGNED_AXES for choice in scope.choices)
                else None
            )
        inputs[requirement.id] = _Input(
            pin=pin,
            origin=origin,
            held=held[origin],
            reads=_Reads.of(requirement, sources.get(origin)),
        )
    return inputs


def _input_bases(inputs: Mapping[str, _Input], *, axis: str) -> dict[str, _InputBasis]:
    """The option each dataset requirement of `inputs` is counted on along `axis`, by requirement id.

    `inputs` holds each requirement as `_read_inputs` read it. A requirement
    whose origin carries no choice on `axis` has none.
    """
    bases: dict[str, _InputBasis] = {}
    for requirement_id, read in inputs.items():
        held = read.held
        if held is None:
            continue
        choice = next((c for c in held.scope.choices if c.axis == axis), None)
        if choice is None:
            continue
        bases[requirement_id] = _InputBasis(
            requirement_id=requirement_id,
            pin=read.pin,
            origin=read.origin,
            choice=choice,
            stated=choice.provenance != "default"
            and (
                axis not in planner_axes(held.block) or _provider_read_in_words(held.scope, choice)
            ),
            held=held,
            reads=read.reads,
        )
    return bases


def _provider_read_in_words(scope: PartScope, choice: ScopeChoice) -> bool:
    """Whether Search settled `choice`, a provider choice of the part `scope` records,
    on the words it searched: an explicit choice no entry from a choice
    (`ChoiceOrigin`) holds on its axis.

    Such a row is the one the words picked ("multilateral ODA"), so it counts as
    stated even on a pin the planner searched for (`planner.agent.pin_by_search`),
    whose block marks the axis as the planner's and so carries nothing to a later
    search (`harness.scope_choices.carried_entries`). An entry from a choice is one
    the planner posted to align the pin (`planner.agent.compile_on_basis`) or a
    carried one. A basis read in the planner's words stays the planner's.
    """
    return (
        choice.axis == PROVIDER_AXIS
        and choice.provenance == "explicit"
        and not any(
            entry.axis == choice.axis and isinstance(entry.origin, ChoiceOrigin)
            for entry in scope.entries
        )
    )


def _own_flows(scope: PartScope, block: Mapping[str, Any] | None) -> frozenset[ProviderMoney]:
    """The money the reporters' own flows count in the part `scope` records, `block`
    its spec block: bilateral money where every donor is a country, multilateral
    money where every donor is a multilateral organisation, whose outflows spend
    the core contributions its funders paid it, and either where some are or may
    be.

    Search states it on the donor side. Each donor bound there counts what the set
    naming it states (`ResolvedGroup.holds_multilaterals`), where it is that
    set's total or one of its members, or where the set names neither and Search
    does not mark the donor a multilateral; any other donor counts `multilateral`
    where Search marks it one (`_donor_money`), as IDA bound beside a set of DAC
    members does. The part counts every money any of its donors counts, so DAC
    countries' total beside IDA counts both. For a donor role left open, the donors there state it
    (`role_axis_facts`). Search marks only parties the graph types `Organization`,
    so an unmarked country is none. Anything else states nothing: an unmarked
    organisation or total, or an open role stating nothing, may hold one, as an
    unreadable axis counts as overlapping. Where Search's donor set is not readable
    (`compiler.sources.block_group`), its marker is not read, so each donor
    counts what Search marks of it alone (`_donor_money`).
    """
    resolved = (block or {}).get("resolved") or {}
    donors = bound_parties(scope, "origin")
    if not donors:
        fact = next(
            (
                fact
                for fact in resolved.get("role_axis_facts") or ()
                if isinstance(fact, Mapping) and fact.get("role") == "origin"
            ),
            {},
        )
        stated = fact.get("holds_multilaterals")
        if isinstance(stated, bool):
            return ANY_REPORTERS if stated else BILATERAL_REPORTERS
        return ANY_REPORTERS
    try:
        group = block_group(block or {}, "origin")
    except ValueError:
        group = None
    marker = group.holds_multilaterals if group is not None else None
    named = {group.total, *group.members} - {None} if group is not None else set()
    entities = {
        entity.get("ref"): entity
        for entity in resolved.get("entities") or ()
        if isinstance(entity, Mapping) and entity.get("role") == "origin"
    }
    money: set[ProviderMoney] = set()
    for donor in donors:
        entity = entities.get(donor)
        # A set naming no one speaks for every donor but one Search marks a
        # multilateral of its own, bound beside the set.
        covered = donor in named or (not named and (entity or {}).get("multilateral") is not True)
        if marker is not None and covered:
            money |= ANY_REPORTERS if marker else BILATERAL_REPORTERS
        else:
            money |= _donor_money(entity)
    return frozenset(money)


def _donor_money(entity: Mapping[str, Any] | None) -> frozenset[ProviderMoney]:
    """The money one bound donor's own flows count, `entity` its resolved entity
    (`_own_flows`)."""
    if entity is not None and entity.get("multilateral") is True:
        return MULTILATERAL_REPORTERS
    if entity is not None and entity.get("entity_type") == _COUNTRY_TYPE:
        return BILATERAL_REPORTERS
    return ANY_REPORTERS


def _aligned_bases(
    requirements: Sequence[EvidenceRequirement],
    evidence_requirements: Sequence[EvidenceRequirement],
    *,
    shown: Mapping[str, PinSource],
    sources: SourceReadings,
    aligned: Sequence[AlignedPin],
    searched: Sequence[SearchedPin],
    view: SessionView,
    proposer: Proposer,
) -> list[EvidenceRequirement] | dict[str, object]:
    """This call's requirements with every input of a calculation pinned on one measure basis and one provider scope, or the refusal that stops the plan.

    A dataset input the planner pinned no dataset for is pinned first to the
    dataset Search settled on for it (`searched`), since its basis is
    unknown until a dataset is pinned. An input of a calculation another of
    whose inputs carries a basis or provider choice, with no search of it
    held yet, answers `BASIS_UNKNOWN`, naming each to search for
    (`Unpinned`). `planner.agent.with_source_reads` searches for them and
    proposes again.
    One whose search settled on no one dataset answers
    `OPERATION_UNSUPPORTED`, since one basis or provider scope cannot be
    guaranteed for it.

    A whole the planner declared (`DatasetRequirement.whole_of`) is pinned
    to its part's dataset with the declared axis left open, compiled with
    no search and whatever pin the planner wrote on it (`_declared_wholes`).
    A part with no pin is searched for first. A declared whole is left
    out of alignment. Its calculation's other inputs are aligned among themselves
    first, and the whole is then compiled from its part's final pin, the
    one alignment moved the part to, so it follows its part by construction.
    Nothing reads a whole but the denominator of a share or quotient of its
    part (`planner.plan.whole_of_fault`), so alignment never reaches it
    through another input. A whole not compiled yet answers `BASES_UNALIGNED` naming it
    (`Realignment.whole_axis`), and a part Search cannot leave the axis open
    on answers `SHARE_WHOLE_UNDERIVABLE`. A share
    declaring nothing is left as written, and its inputs are aligned as any
    calculation's are.

    A calculation reading several inputs (a comparison, sum, share, quotient
    or join) contrasts sector and region, and keeps those as stated. On
    each axis of `_ALIGNED_AXES`, the measure basis and the provider scope
    (`harness.scope_choices.ProviderScope`), it
    contrasts its inputs only when the user stated different options on
    them (explicit or carried entries naming different options, an entry
    this planner compiled excluded, as `_InputBasis.stated` reads them), and
    a comparison, share, quotient or sum contrasting provider scopes says so
    (`_difference_decisions`).
    Otherwise its dataset inputs with a choice on the axis are counted on
    one option, except a calculation's whose two sides would then read one
    series, which keep their own (`_sides_one_series`). Only a comparison, share, quotient or sum aligns the
    provider scope, and not one the user asked to keep its inputs' own
    (`accept_provider_difference`); a join keeps each input's own
    (`_aligns`). Calculations sharing an input, such as a sum read by a
    share, form one group and take one option for every input any of them
    combines (`_shared_basis`): the one the user stated on any of them, else
    the one the narrowest part is counted on (`_part_rank`), the first such
    input in plan order on a tie. A group whose inputs the user stated on
    different options has no one option, and each of its calculations takes
    its own. A sum whose inputs the user stated on different provider
    scopes, or one accepting a provider difference whose inputs are on
    different scopes, is refused where two of them can count one flow
    (`_Flows.meet`: no donor, recipient, summable slot or year keeps them apart),
    where they count one money (`_shared_money`), since an inclusive figure
    already holds the bilateral one and the sum would count it twice. A
    multilateral organisation's own outflows count multilateral money
    (`_own_flows`), since they spend the core contributions its funders paid
    it, so every donor's core contributions plus IDA's outflows is refused
    too. A sum over disjoint donors, such as France's
    bilateral aid plus Germany's inclusive aid, over disjoint sectors, such
    as US bilateral health aid plus US inclusive education aid, or over
    inputs sharing no money, such as US bilateral aid plus US core
    contributions to multilaterals, is served as stated. A sum whose inputs
    are aligned on one provider scope is refused the same way where their
    reporters' own flows count different money (`_double_counted`), as DAC
    countries' inclusive aid plus IDA's outflows both count the donors' aid
    through multilaterals, or may count different money, as donors left open
    may, unless their flows are apart: grants plus loans to Kenya from all
    donors on the inclusive scope is served.

    A calculation aligns each group of inputs it reads (`_input_groups`). A
    share or quotient beneath another calculation is one figure, its own
    inputs left on the options it put them on, except that a calculation over
    ratios alone aligns their numerators with each other and their
    denominators with each other. It leaves a position's provider scopes as they stand when a ratio
    supplying it accepts a provider difference, since that ratio's contract
    stands. So comparing two shares of the DAC1 total that each accept a
    provider difference keeps both denominators.

    An input already on its options keeps its origin pin. Another is pinned
    to the pin compiled from its origin on the options it moves to, one per
    axis (`aligned`), and the requirement is restated under it, appended
    when only the plan this call amends states it. An input with no such
    pin yet answers `BASES_UNALIGNED`, naming each to compile with every
    option it moves to (`Realignment`). `planner.agent.with_source_reads`
    compiles them and proposes again. An input whose choice does not offer
    the provider scope its calculation counts, or offers it unavailable,
    answers `PROVIDERS_UNALIGNED` (`_providers_unaligned`); the same on the
    measure basis, and an input two calculations want on different options
    of one axis, answer `OPERATION_UNSUPPORTED`.
    Every requirement pinned to a compiled whole or a realigned pin drops
    the source it stated (`planner.plan.repinned`), which discovery would
    otherwise search back to the dataset it replaced.

    Args:
        requirements: Every requirement the plan states once this call's
            patch lands (`_requirements_after`).
        evidence_requirements: This call's own requirements.
        shown: The datasets the planner was shown (`planner.plan.pin_sources`).
        sources: The proposal's readings of the blocks of `shown` (`SourceReadings`).
        aligned: The pins compiled so far (`aligned_pins`).
        searched: The searches made so far for unpinned inputs (`searched_pins`).
        view: The session, where an `unsupported` answer is recorded.
        proposer: The tool a refusal's `action` names.
    """
    held = {pin.requirement_key: pin for pin in searched}
    searchable = [
        requirement
        for requirement in requirements
        if isinstance(requirement, DatasetRequirement)
        and requirement.source_spec_id is None
        and requirement.whole_of is None
    ]
    found = {
        requirement.id: pin.spec_id
        for requirement in searchable
        if (pin := held.get(requirement_revision(requirement))) is not None
        and pin.spec_id is not None
    }
    unsearched = {
        requirement.id: requirement
        for requirement in searchable
        if requirement_revision(requirement) not in held
    }
    unsettled = {
        requirement.id: pin
        for requirement in searchable
        if (pin := held.get(requirement_revision(requirement))) is not None and pin.spec_id is None
    }
    requirements = [_pinned(requirement, found) for requirement in requirements]
    evidence_requirements = [_pinned(requirement, found) for requirement in evidence_requirements]
    try:
        declared = _whole_parts(requirements, unsearched=unsearched)
    except _Unservable as exc:
        return exc.problem
    origins = {pin.spec_id: pin.origin for pin in aligned if pin.whole_axis is None}
    compiled = {
        (pin.origin, pin.option_ids): pin.spec_id for pin in aligned if pin.whole_axis is None
    }
    # Alignment leaves a declared whole alone. It is compiled from its part's
    # final pin below.
    whole_ids = {whole.id for whole, _, _ in declared}
    aligning = [requirement for requirement in requirements if requirement.id not in whole_ids]
    inputs = _read_inputs(aligning, sources=sources, origins=origins)
    by_axis = {axis: _input_bases(inputs, axis=axis) for axis in _ALIGNED_AXES}
    steps = [step for step in requirements if len(inputs_of(step)) >= 2]
    groups = _input_groups(requirements, source_of=lambda pin: sources.get(origins.get(pin, pin)))
    unpinned: dict[str, Unpinned] = {}
    for step in steps:
        under = set().union(*(group.members for group in groups[step.id]))
        if not any(
            bases.keys() & under for axis, bases in by_axis.items() if _reads_on(step, axis)
        ):
            continue
        for requirement_id in sorted(unsettled.keys() & under):
            return _basis_unservable(
                view,
                proposer,
                step_id=step.id,
                requirement_id=requirement_id,
                reason=(
                    f"Search settles {requirement_id!r} on no one dataset "
                    f"({unsettled[requirement_id].status}), so how it is counted cannot be matched "
                    "to the other inputs'"
                ),
            )
        for requirement_id in sorted(unsearched.keys() & under):
            unpinned.setdefault(requirement_id, _unpinned(unsearched[requirement_id], step.id))
    if unpinned:
        return _basis_unknown(list(unpinned.values()))
    try:
        targets = _alignment_targets(
            steps,
            groups,
            by_axis,
            reads={requirement_id: read.reads for requirement_id, read in inputs.items()},
            view=view,
            proposer=proposer,
        )
        wanted, realigned = _realigned_pins(targets, by_axis, compiled=compiled)
        if wanted:
            raise _Unservable(_bases_unaligned(wanted, by_axis))
        wholes = _declared_wholes(
            declared,
            realigned=realigned,
            shown=shown,
            compiled=[pin for pin in aligned if pin.whole_axis is not None],
        )
    except _Unservable as exc:
        return exc.problem
    moved = {**wholes, **realigned}
    return [
        repinned(requirement, pin, exact_codes(_pin_block(pin, shown)))
        if isinstance(requirement, DatasetRequirement)
        and (pin := moved.get(requirement.id)) is not None
        else requirement
        for requirement in _with_found(evidence_requirements, requirements, {**found, **moved})
    ]


class _InputGroup(NamedTuple):
    """Inputs one calculation counts on one option (`_input_groups`).

    `members` are requirement ids, the datasets among them the inputs.
    `accepted` says a ratio beneath the calculation supplying a member accepts a
    provider difference (`accept_provider_difference`), so the calculation
    leaves the group's provider scopes as they stand. `sides` are the
    derivations of the two sides of a comparison, sum, share or quotient
    (`_side_derivations`), for one reading its sides as one group, so alignment can
    keep the two sides apart (`_sides_one_series`); `None` for any other group,
    and for every group read without a source (`_input_groups`).
    """

    members: frozenset[str]
    accepted: bool = False
    sides: tuple[_Derivation, _Derivation] | None = None


class _Derivation(NamedTuple):
    """What one side of a calculation computes, from its sources up (`_derivation`).

    `shape` is the canonical JSON of every operation between the sources and
    the calculation, except the steps `_passed_through` drops, with its
    parameters and entity selectors, each source written as its position in
    `leaves`. `leaves` are the dataset requirements read, in that order, each
    with the spec_id it pins. Two sides of one `shape` read one series where
    each pair of leaves at one position does.
    """

    shape: bytes
    leaves: tuple[tuple[str, str | None], ...]


# The fields through which a derived requirement names a requirement it reads
# (`planner.plan.inputs_of`).
_READ_FIELDS = frozenset({"input_id", "left_id", "right_id"})


def _derivation(
    side: RequirementSide,
    requirements: Mapping[str, EvidenceRequirement],
    source_of: Callable[[str], SelectedSource | None],
) -> _Derivation:
    """The derivation of `side` (`_Derivation`): the requirement it reads, narrowed
    to its entity, down to every dataset requirement beneath it.

    A derived requirement is its every field but its id and decisions, each
    requirement it reads replaced by that requirement's own derivation, and
    each list whose order computes nothing in one order (`_canonical`). A step
    that leaves its input's frame as it is (`_passed_through`) is its input's
    derivation. A dataset requirement, or one the plan does not state, is a
    leaf.
    """
    leaves: list[tuple[str, str | None]] = []

    def read(requirement_id: str) -> JsonValue:
        requirement = requirements.get(requirement_id)
        if requirement is None or isinstance(requirement, DatasetRequirement):
            pin = None if requirement is None else requirement.source_spec_id
            leaves.append((requirement_id, pin))
            return {"leaf": len(leaves) - 1}
        if (through := _passed_through(requirement, requirements, source_of)) is not None:
            return read(through)
        return substituted(_canonical(requirement))

    def substituted(value: JsonValue) -> JsonValue:
        if isinstance(value, dict):
            return {
                key: read(held)
                if key in _READ_FIELDS and isinstance(held, str)
                else substituted(held)
                for key, held in value.items()
            }
        if isinstance(value, list):
            return [substituted(held) for held in value]
        return value

    shape = rfc8785.dumps({"entity": side.entity, "reads": read(side.input_id)})
    return _Derivation(shape, tuple(leaves))


def _canonical(requirement: EvidenceRequirement) -> dict[str, JsonValue]:
    """`requirement`'s fields but its id and decisions, each list whose order
    computes nothing sorted. These are a filter's predicates, each entity list
    among them too (`planner.plan.predicate_key`), since a row survives every
    predicate in any order, and an aggregate's `group_by`, whose order sets
    only the order of the result's columns."""
    fields = requirement.model_dump(mode="json", exclude={"id", "decision_ids"})
    if isinstance(requirement, FilterRequirement):
        fields["predicates"] = sorted(predicate_key(held) for held in requirement.predicates)
    if isinstance(requirement, AggregateRequirement):
        fields["group_by"] = sorted(held.value for held in requirement.group_by)
    return fields


def _passed_through(
    requirement: EvidenceRequirement,
    requirements: Mapping[str, EvidenceRequirement],
    source_of: Callable[[str], SelectedSource | None],
) -> str | None:
    """The dataset requirement `requirement` reads, where the plan proves it leaves
    that requirement's frame as it is; else `None`. `source_of` is the source
    of a pin, `None` where none is held.

    Two steps are proved so, each reading a dataset requirement directly:

    - A filter whose every predicate is an `entity_in` over a dataset
      requirement that covers no party whole (`across`), each predicate naming
      every entity, and only those, that the dataset states under the
      predicate's role. A predicate naming no role passes only where the
      dataset states every entity under one role. The frame's column for that
      role holds the entities the dataset states, each resolved as the filter
      resolves it, so every row matches.
    - An aggregate summing grouped by period over a dataset requirement holding
      one series (`_one_series`): each year holds one row, which its sum is.

    Every other step is kept, such as a sum by period folding several
    recipients, an across party or a breakdown into one figure a year.
    """
    if isinstance(requirement, AggregateRequirement):
        dataset = requirements.get(requirement.input_id)
        kept = (
            requirement.reduction is Reduction.SUM
            and GroupBy.PERIOD in requirement.group_by
            and isinstance(dataset, DatasetRequirement)
            and _one_series(dataset, _source_of_pin(dataset, source_of))
        )
    elif isinstance(requirement, FilterRequirement):
        dataset = requirements.get(requirement.input_id)
        kept = isinstance(dataset, DatasetRequirement) and _keeps_every_row(requirement, dataset)
    else:
        return None
    return requirement.input_id if kept else None


def _source_of_pin(
    dataset: DatasetRequirement, source_of: Callable[[str], SelectedSource | None]
) -> SelectedSource | None:
    """The source `dataset` pins (`source_of`), `None` where it pins none."""
    return None if dataset.source_spec_id is None else source_of(dataset.source_spec_id)


def _keeps_every_row(requirement: FilterRequirement, dataset: DatasetRequirement) -> bool:
    """Whether `requirement`, a filter reading `dataset`, keeps every row of it
    (`_passed_through`)."""
    if dataset.across:
        return False
    roles = {entity.role for entity in dataset.entities}
    for predicate in requirement.predicates:
        if not isinstance(predicate, EntityIn) or (predicate.role is None and len(roles) != 1):
            return False
        named = {
            entity.name
            for entity in dataset.entities
            if predicate.role is None or entity.role == predicate.role
        }
        if set(predicate.names) != named:
            return False
    return True


def _one_series(dataset: DatasetRequirement, source: SelectedSource | None) -> bool:
    """Whether the plan shows `dataset` retrieves one series (one row a year),
    `source` its pin's source.

    It covers no party whole (`across`) and breaks nothing down, reads its
    StatVars as one measure, as the compiler reads them (`_reading`: a family
    the plan leaves unset is read separately where the compiler's rule reads it
    so), and names at most one entity per role, none of them a set read member
    by member. This is the compiler's rule for a side read whole
    (`compiler.compile._names_on_column`: one entity per role column is one
    series) read from the plan, before any source resolves the names.
    """
    roles = [entity.role for entity in dataset.entities]
    return (
        not dataset.across
        and dataset.breakdown is None
        and _reading(dataset, source) is not StatVarReading.SEPARATE
        and len(set(roles)) == len(roles)
        and not any(entity.members for entity in dataset.entities)
    )


def _side_derivations(
    step: EvidenceRequirement,
    requirements: Mapping[str, EvidenceRequirement],
    source_of: Callable[[str], SelectedSource | None],
) -> tuple[_Derivation, _Derivation] | None:
    """The derivations of the two sides of `step`, a comparison, sum, share or
    quotient (`_derivation`); `None` for any other."""
    if isinstance(step, ComparisonRequirement):
        sides = step.minuend, step.subtrahend
    elif isinstance(step, SumOfRequirement):
        sides = step.left, step.right
    elif isinstance(step, ShareOfRequirement | QuotientRequirement):
        sides = step.numerator, step.denominator
    else:
        return None
    return (
        _derivation(sides[0], requirements, source_of),
        _derivation(sides[1], requirements, source_of),
    )


def _input_groups(
    requirements: Sequence[EvidenceRequirement],
    *,
    source_of: Callable[[str], SelectedSource | None] | None = None,
) -> dict[str, list[_InputGroup]]:
    """The groups of requirements each calculation of `requirements` reading two or
    more inputs counts on one option, by its id. `source_of` is the source of a
    pin (`_passed_through`); a group reading its calculation's sides as one holds
    their derivations (`_InputGroup.sides`) only where it is given, as alignment
    reads them and nothing else does.

    A calculation reads one group: what it reads, and what a calculation it
    reads reads, down to a share or quotient, which it reads as one figure,
    its inputs counted as the ratio itself settled them, aligned or accepted
    as different (`accept_provider_difference`). One whose every input is a
    ratio, read directly or through calculations reading one input each,
    whose sides hold no ratio, reads the ratios position by position: their
    numerators as one group and their denominators as another, each
    `accepted` when any of those ratios accepts a provider difference. So
    comparing two shares of one total aligns the numerators with each other
    and leaves each share's own contract between numerator and denominator
    standing, and comparing gross ODA per person with net ODA per person
    counts the two ODA figures on one basis.
    """
    ratios = {
        requirement.id: requirement
        for requirement in requirements
        if isinstance(requirement, ShareOfRequirement | QuotientRequirement)
    }
    # An input the plan does not state reads nothing here, and `Plan` refuses it.
    reads: dict[str, Sequence[str]] = {
        input_id: () for requirement in requirements for input_id in inputs_of(requirement)
    }
    reads |= {requirement.id: inputs_of(requirement) for requirement in requirements}
    cut = {node: () if node in ratios else inputs for node, inputs in reads.items()}

    def ratio_of(node: str) -> str | None:
        """The one ratio `node` is, or reads through calculations reading one input each."""
        below = reached(node, cut)
        found = below & ratios.keys()
        chain = all(len(reads[other]) == 1 for other in below - found)
        return next(iter(found)) if len(found) == 1 and chain else None

    by_id = {requirement.id: requirement for requirement in requirements}
    groups: dict[str, list[_InputGroup]] = {}
    for step in requirements:
        inputs = inputs_of(step)
        if len(inputs) < 2:
            continue
        under = [ratios[ratio] for input_id in inputs if (ratio := ratio_of(input_id))]
        sides = [inputs_of(ratio) for ratio in under]
        if len(sides) == len(inputs) and not any(
            reached(side, cut) & ratios.keys() for pair in sides for side in pair
        ):
            accepted = any(ratio.accept_provider_difference for ratio in under)
            groups[step.id] = [
                _InputGroup(
                    frozenset().union(*(reached(pair[position], cut) for pair in sides)),
                    accepted,
                )
                for position in (0, 1)
            ]
        else:
            groups[step.id] = [
                _InputGroup(
                    frozenset(reached(step.id, {**cut, step.id: inputs})),
                    sides=(
                        _side_derivations(step, by_id, source_of) if source_of is not None else None
                    ),
                )
            ]
    return groups


def _alignment_targets(
    steps: Sequence[EvidenceRequirement],
    input_groups: Mapping[str, Sequence[_InputGroup]],
    by_axis: Mapping[str, Mapping[str, _InputBasis]],
    *,
    reads: Mapping[str, _Reads],
    view: SessionView,
    proposer: Proposer,
) -> dict[tuple[str, str], tuple[str, str]]:
    """The option each input of `steps` is counted on, by its requirement id and axis, with the calculation that wants it (`_aligned_bases`).

    Each step counts each of its groups of inputs (`_input_groups`) on one option. On each axis, the calculations whose inputs share one option
    (`_shared_basis`) are joined into connected groups by the inputs they
    share, and each takes its group's option, or its own when the group's
    inputs were stated on different options.

    Alignment keeps the two sides of a comparison, sum, share or quotient on
    different series, whichever calculation moves them. Once an axis's options
    are chosen, a calculation whose two sides alignment has put on one series,
    each computed alike from inputs on one part and options
    (`_sides_one_series`, each input read as `reads` holds it, by requirement
    id), has every group that moved one of those inputs left unaligned on that axis, and the options are chosen again until no group
    changes. The axis is what the calculation
    contrasts, as "what share of US ODA goes to multilaterals" reads US
    multilateral ODA over US total ODA, and counting the total on the
    multilateral row, whether for the share or for a comparison of the total
    with France's multilateral ODA, would answer 100%, the difference of the
    two 0, and their sum twice the multilateral row. Every input of a
    calculation left unaligned keeps its own option, so a group moving one of
    them is left unaligned too, and the outcome is the same in any step order.
    The inputs keep their own options, as those the user stated on different
    options do. The difference is stated on the axis kept apart
    (`_difference_decisions`), and a sum whose inputs then count one
    money is refused (`_refuse_double_count`), as US bilateral ODA plus US
    total ODA is, the total already holding the bilateral flows.

    Raises:
        _Unservable: When the inputs of one calculation cannot be counted on one option.
    """
    targets: dict[tuple[str, str], tuple[str, str]] = {}
    for axis, bases in by_axis.items():
        candidates: list[_Aligning] = []
        for step in steps:
            if not _reads_on(step, axis):
                continue
            for group in input_groups[step.id]:
                inputs = [
                    basis
                    for requirement_id, basis in bases.items()
                    if requirement_id in group.members
                ]
                if len(inputs) < 2:
                    continue
                aligns = _aligns(step, axis) and not (axis == PROVIDER_AXIS and group.accepted)
                own = _shared_basis(inputs) if aligns else None
                candidates.append(_Aligning(step, group, inputs, aligns, own))
        # Each candidate counting its inputs on one option, by its index in
        # `candidates`, with the option its inputs share.
        aligning = [
            (index, own)
            for index, candidate in enumerate(candidates)
            if (own := candidate.own) is not None
        ]
        while True:
            chosen = _chosen_options([(candidates[index], own) for index, own in aligning], bases)
            counted: dict[str, set[str]] = {}
            for (index, _), target in zip(aligning, chosen, strict=True):
                for basis in candidates[index].inputs:
                    counted.setdefault(basis.requirement_id, set()).add(target)
            moved = {
                move
                for candidate in candidates
                for move in _sides_one_series(candidate.group, bases, reads, axis, targets, counted)
            }
            # The inputs of a calculation kept apart stay on their own options.
            aligned = {index for index, _ in aligning}
            held = {
                basis.requirement_id
                for index, candidate in enumerate(candidates)
                if candidate.own is not None and index not in aligned
                for basis in candidate.inputs
            }
            kept = [
                (index, own)
                for (index, own), target in zip(aligning, chosen, strict=True)
                if not any(
                    (basis.requirement_id, target) in moved
                    or (basis.requirement_id in held and target != basis.choice.active)
                    for basis in candidates[index].inputs
                )
            ]
            if len(kept) == len(aligning):
                break
            aligning = kept
        aligned = {index for index, _ in aligning}
        for index, candidate in enumerate(candidates):
            if index in aligned:
                continue
            # A group left unaligned to keep two sides apart refuses as one
            # accepting a provider difference does, on every input.
            kept_apart = candidate.own is not None
            _refuse_double_count(
                candidate.step,
                axis,
                [basis for basis in candidate.inputs if basis.stated]
                if candidate.aligns and not kept_apart
                else candidate.inputs,
                view=view,
                proposer=proposer,
            )
        for (index, _), target in zip(aligning, chosen, strict=True):
            candidate = candidates[index]
            for basis in candidate.inputs:
                _claim(targets, basis, axis, target, candidate.step, view=view, proposer=proposer)
            _refuse_double_count(
                candidate.step,
                axis,
                [_counted_on(basis, target) for basis in candidate.inputs],
                view=view,
                proposer=proposer,
            )
    return targets


class _Aligning(NamedTuple):
    """One group of inputs a calculation reads on one axis (`_alignment_targets`).

    `aligns` says the calculation counts them on one option (`_aligns`), and
    `own` is the option they share (`_shared_basis`), `None` where it leaves
    them on their own.
    """

    step: EvidenceRequirement
    group: _InputGroup
    inputs: list[_InputBasis]
    aligns: bool
    own: str | None


def _chosen_options(
    aligning: Sequence[tuple[_Aligning, str]], bases: Mapping[str, _InputBasis]
) -> list[str]:
    """The option each group of `aligning`, with the option its own inputs share,
    counts its inputs on, in order.

    The groups sharing an input are joined, and each takes the option its
    joined inputs share (`_shared_basis`), or its own where they were stated on
    different options, which `_claim` refuses where it moves an input another
    group counts on another option.
    """
    joined_groups: list[set[str]] = []
    for candidate, _ in aligning:
        joined = {basis.requirement_id for basis in candidate.inputs}
        for held in [held for held in joined_groups if held & joined]:
            joined |= held
            joined_groups.remove(held)
        joined_groups.append(joined)
    shared = {
        requirement_id: _shared_basis([b for r, b in bases.items() if r in joined])
        for joined in joined_groups
        for requirement_id in joined
    }
    return [shared[candidate.inputs[0].requirement_id] or own for candidate, own in aligning]


def _sides_one_series(
    group: _InputGroup,
    bases: Mapping[str, _InputBasis],
    reads: Mapping[str, _Reads],
    axis: str,
    targets: Mapping[tuple[str, str], tuple[str, str]],
    counted: Mapping[str, Collection[str]],
) -> frozenset[tuple[str, str]]:
    """The moves alignment on `axis` makes that put the two sides of `group`, a
    comparison's, sum's, share's or quotient's, on one series: each input moved
    and the option it moves to.

    Two sides read one series where their derivations (`_side_derivations`)
    hold one shape and each pair of leaves at one position reads one series, so
    a total compared with an average of one series reads two, and a total by
    year over the series itself reads one. Two leaves read one series where
    both have a choice on `axis` (`bases`), share a part (`_Origin.part`),
    read it alike (`_InputBasis.reads`: one period, one set of entities, one
    frame), and hold one option on every axis (`_Origin.actives`), which on `axis`
    is an option `counted` gives or the input's own, and on an earlier axis the
    one in `targets`, or where neither has a choice on `axis`, both pin one
    spec_id and read it alike (`reads`, by requirement id). A part and its
    options are what a pin compiles to. `counted` holds every option any aligning group counts an input on, so a pair is
    caught on whichever group moves it, in any step order. Two sides on one
    series before alignment moves an input are left alone, since alignment did
    not put them there. A group with no sides (`_InputGroup.sides`) has none
    to keep apart.
    """
    if group.sides is None:
        return frozenset()
    first, second = group.sides
    if first.shape != second.shape:
        return frozenset()

    def readings(leaf: tuple[str, str | None]) -> list[tuple[str | None, object]]:
        """Each option `leaf` can be counted on along `axis`, with the series it reads there."""
        requirement_id, pin = leaf
        basis = bases.get(requirement_id)
        if basis is None:
            return [(None, (pin, reads.get(requirement_id)))]
        others = {
            held: targets.get((requirement_id, held), (active, ""))[0]
            for held, active in basis.held.actives
            if held != axis
        }
        options = {basis.choice.active, *counted.get(requirement_id, ())}
        return [
            (option, (basis.held.part, basis.reads, {**others, axis: option}))
            for option in sorted(options)
        ]

    pairs = list(zip(first.leaves, second.leaves, strict=True))

    def matched(
        position: int, chosen: dict[str, tuple[str | None, object]]
    ) -> Iterable[dict[str, tuple[str | None, object]]]:
        """Each reading of every leaf, extending `chosen`, under which the pairs from `position` on read one series."""
        if position == len(pairs):
            yield chosen
            return
        one, other = pairs[position]
        for one_reading in [chosen[one[0]]] if one[0] in chosen else readings(one):
            held = {**chosen, one[0]: one_reading}
            for other_reading in [held[other[0]]] if other[0] in held else readings(other):
                if one_reading[1] == other_reading[1]:
                    yield from matched(position + 1, {**held, other[0]: other_reading})

    moves = [
        {
            (requirement_id, option)
            for requirement_id, (option, _) in chosen.items()
            if option is not None and option != bases[requirement_id].choice.active
        }
        for chosen in matched(0, {})
    ]
    # A reading moving nothing is the two sides on one series before alignment.
    if not all(moves):
        return frozenset()
    return frozenset().union(*moves)


def _counted_on(basis: _InputBasis, option_id: str) -> _InputBasis:
    """`basis` as its input is counted once aligned on `option_id` (`_alignment_targets`)."""
    if basis.choice.active == option_id:
        return basis
    return replace(basis, choice=basis.choice.model_copy(update={"active": option_id}))


def _refuse_double_count(
    step: EvidenceRequirement,
    axis: str,
    inputs: Sequence[_InputBasis],
    *,
    view: SessionView,
    proposer: Proposer,
) -> None:
    """Refuse a sum whose `inputs`, each on the option of `axis` it is counted on,
    count one provider's flows twice (`_double_counted`): of a sum aligning them,
    every input on the option it aligns to, or those the user stated where they
    stated different ones, and of one leaving a provider difference, every one.

    Raises:
        _Unservable: When two such inputs, on scopes counting one money
            (`_shared_money`), can count one flow (`_Flows.meet`).
    """
    if axis != PROVIDER_AXIS or not isinstance(step, SumOfRequirement):
        return
    pair = _double_counted(inputs)
    if pair is None:
        return
    first, other = pair
    money = _shared_money(first, other)
    shared = " and ".join(_MONEY_WORDS[held] for held in sorted(money))
    wider = _wider(first, other)
    counted = (
        f"both count {shared}"
        if wider is None
        else f"the figure counted as {wider.label(wider.choice.active)} already holds {shared}"
    )
    spending = next(
        (
            f", since the donors of {basis.requirement_id!r} "
            f"{'are' if basis.held.own_flows == MULTILATERAL_REPORTERS else 'may include'} "
            "multilateral organisations, whose outflows spend the core contributions "
            "their funders paid them"
            for basis in (first, other)
            if "multilateral" in basis.held.own_flows
            and "multilateral" in money
            and "bilateral" in _streams(basis)
        ),
        "",
    )
    raise _Unservable(
        _basis_unservable(
            view,
            proposer,
            step_id=step.id,
            requirement_id=other.requirement_id,
            spec_id=other.origin,
            option_id=first.choice.active,
            reason=(
                f"{step.id!r} adds {first.requirement_id!r}, counted as "
                f"{first.label(first.choice.active)}, to {other.requirement_id!r}, "
                f"counted as {other.label(other.choice.active)}, and {counted}{spending}, "
                "so the sum counts them twice"
            ),
        )
    )


def _streams(basis: _InputBasis) -> frozenset[ProviderStream]:
    """The streams `basis`'s active provider scope counts, every one where it states none."""
    scope = _active_scope(basis)
    return PROVIDER_STREAMS[scope] if scope is not None else frozenset(get_args(ProviderStream))


def _claim(
    targets: dict[tuple[str, str], tuple[str, str]],
    basis: _InputBasis,
    axis: str,
    target: str,
    step: EvidenceRequirement,
    *,
    view: SessionView,
    proposer: Proposer,
) -> None:
    """Count `basis` on `target` along `axis` for `step` in `targets`, unless
    another calculation counts it on another option or its dataset does not
    serve `target`.

    Raises:
        _Unservable: With `PROVIDERS_UNALIGNED` for a provider scope the input's
            dataset does not serve, else `OPERATION_UNSUPPORTED`.
    """
    held_target, held_by = targets.setdefault((basis.requirement_id, axis), (target, step.id))
    option = next((o for o in basis.choice.options if o.option_id == target), None)
    unserved = option is None or option.availability.status == "unavailable"
    if held_target == target and unserved and axis == PROVIDER_AXIS:
        raise _Unservable(
            _providers_unaligned(step, basis, target, permit=view.record_optional_decline)
        )
    if held_target != target:
        reason = (
            f"{basis.requirement_id!r} is also an input of {held_by!r}, which "
            f"counts it as {basis.label(held_target)}, and one dataset "
            "requirement is counted one way"
        )
    elif unserved:
        reason = (
            f"{step.id!r} counts its inputs as {basis.label(target)}, and "
            f"{basis.requirement_id!r}'s dataset, counted as "
            f"{basis.label(basis.choice.active)}, is not served that way"
        )
    else:
        return
    raise _Unservable(
        _basis_unservable(
            view,
            proposer,
            step_id=step.id,
            requirement_id=basis.requirement_id,
            spec_id=basis.origin,
            option_id=target,
            reason=reason,
        )
    )


def _realigned_pins(
    targets: Mapping[tuple[str, str], tuple[str, str]],
    by_axis: Mapping[str, Mapping[str, _InputBasis]],
    *,
    compiled: Mapping[tuple[str, tuple[str, ...]], str],
) -> tuple[list[Realignment], dict[str, str]]:
    """The inputs still to compile on their `targets`, and the pins of the inputs compiled already, by requirement id (`_aligned_bases`).

    An input is compiled from its origin once on every option it moves to,
    one per axis, so a gross inclusive total wanted net and bilateral is
    one compile, and `compiled` holds it under the origin and those options.
    """
    moves: dict[str, list[tuple[str, str]]] = {}
    for (requirement_id, axis), (target, step_id) in targets.items():
        moved = moves.setdefault(requirement_id, [])
        if by_axis[axis][requirement_id].choice.active != target:
            moved.append((target, step_id))
    wanted: list[Realignment] = []
    repinned: dict[str, str] = {}
    for requirement_id, moved in moves.items():
        basis = next(bases[requirement_id] for bases in by_axis.values() if requirement_id in bases)
        option_ids = tuple(sorted(target for target, _ in moved))
        pin = compiled.get((basis.origin, option_ids)) if moved else basis.origin
        if pin is None:
            wanted.append(Realignment(basis.origin, option_ids, requirement_id, moved[0][1]))
        elif pin != basis.pin:
            repinned[requirement_id] = pin
    return wanted, repinned


def _providers_unaligned(
    step: EvidenceRequirement,
    basis: _InputBasis,
    target: str,
    *,
    permit: Callable[[UnsupportedOperation], None],
) -> dict[str, object]:
    """Record and return the `PROVIDERS_UNALIGNED` problem for an input of `step` its
    dataset does not serve on `target`, the provider scope `step`'s other inputs are
    counted on.

    The planner can repair it: declare a share's total the part over every
    sector, donor or recipient (`DatasetRequirement.whole_of`), or pin an input counted on that
    scope. `accept_provider_difference` is named only for a user who asked for
    that input's source, so the planner does not take it as a way past the
    refusal. Where no repair applies, the calculation is declined: `permit`
    records an `unsupported_ask` decline as optional, so the decline is earned
    and a repaired plan owes none. `detail.requirement_id` names the input, as
    every refusal about one requirement does, for its sentence
    (`harness.sentences`).
    """
    permit(UnsupportedOperation.UNSUPPORTED_ASK)
    return problem(
        ProblemCode.PROVIDERS_UNALIGNED,
        f"{step.id!r} subtracts, divides or adds its inputs, so they count one provider scope, "
        f"{basis.label(target)}, and {basis.requirement_id!r}'s dataset counts "
        f"{basis.label(basis.choice.active)} and is not served that way. Align them: when "
        f"{step.id!r} is a share of a figure in its total over every sector, donor or "
        "recipient, state that total with `whole_of` naming its part and that axis; otherwise "
        "pin for "
        f"{basis.requirement_id!r} a dataset counted as {basis.label(target)}. Set "
        f"`accept_provider_difference` on {step.id!r} only when the user asked for "
        f"{basis.requirement_id!r}'s source as it stands. When none of these applies, record "
        f"the part of the question asking for {step.id!r} in `declined` with code "
        f"{UnsupportedOperation.UNSUPPORTED_ASK.value!r}.",
        outcome=Outcome.INVALID_INPUT,
        detail={
            "requirement_id": basis.requirement_id,
            "provider": {
                "step_id": step.id,
                "requirement_id": basis.requirement_id,
                "spec_id": basis.origin,
                "option_id": target,
            },
        },
    )


class _Unservable(Exception):  # noqa: N818 -- carries the refusal `_aligned_bases` answers
    """Carries the refusal `_aligned_bases` answers out of an alignment helper."""

    def __init__(self, problem: dict[str, object]) -> None:
        super().__init__(problem.get("code"))
        self.problem = problem


def _double_counted(kept: Sequence[_InputBasis]) -> tuple[_InputBasis, _InputBasis] | None:
    """The first two inputs of `kept`, the inputs of a sum each on the option it is
    counted on, whose options count one money (`_shared_money`) by different
    streams and can count one flow (`_Flows.meet`); `None` when none can.

    Two inputs count a money by different streams where they hold different
    options, or one option whose reporters' own flows count different money
    (`_Origin.own_flows`) where the option counts multilateral money by another
    stream too: DAC countries' inclusive aid and IDA's outflows on the inclusive
    scope both count the donors' aid through multilaterals, one as an imputed share
    and one as what IDA spends. Two inputs on one option whose reporters' own
    flows are known to count the same money count it by the same streams, and
    whether their flows overlap is no question of provider scope. `ANY_REPORTERS`
    states that the money is unknown, so two inputs holding it may each be
    a country or a multilateral and go on like any other pair: two organisations
    Search states nothing of, on the inclusive scope, may be one's outflows
    beside the other's imputed share of them.

    `_refuse_double_count` gives the inputs (`_alignment_targets`)."""
    return next(
        (
            (first, other)
            for index, first in enumerate(kept)
            for other in kept[index + 1 :]
            if (
                first.choice.active != other.choice.active
                or first.held.own_flows != other.held.own_flows
                or first.held.own_flows == ANY_REPORTERS
            )
            and _shared_money(first, other)
            and first.held.flows.meet(other.held.flows)
        ),
        None,
    )


def _wider(first: _InputBasis, other: _InputBasis) -> _InputBasis | None:
    """Of `first` and `other`, the input whose active provider scope holds every stream
    the other's counts (`harness.scope_choices.holds_streams`) and every money
    (`_money`); `None` where neither does or an input's active option states no
    scope."""
    first_scope, other_scope = _active_scope(first), _active_scope(other)
    if first_scope is None or other_scope is None:
        return None
    if holds_streams(first_scope, other_scope) and _money(other) <= _money(first):
        return first
    if holds_streams(other_scope, first_scope) and _money(first) <= _money(other):
        return other
    return None


def _shared_money(first: _InputBasis, other: _InputBasis) -> frozenset[ProviderMoney]:
    """The money `first` and `other`, on the provider options they hold active, both
    count by different streams (`_money`).

    On different options, or one stating no scope, that is every money both count.
    On one option a stream both count counts each input's money the same way, so
    they share money only where one input's reporters' own flows, the bilateral
    stream, count the multilateral money the option's other streams count for the
    other: IDA's outflows beside the US's imputed share of them. US bilateral aid
    beside IDA's, both counting bilateral flows alone, share none, whatever IDA is,
    and neither do two inputs on an option that counts no reporter's own flows
    (`provider:multilateral`, core contributions alone)."""
    scope = _active_scope(first)
    if first.choice.active != other.choice.active or scope is None:
        return _money(first) & _money(other)
    if "bilateral" not in PROVIDER_STREAMS[scope]:
        return frozenset()
    # No own flows: the money the option's streams other than the bilateral one count.
    pooled = scope_money(scope, own_flows=frozenset())
    return (first.held.own_flows | other.held.own_flows) & pooled


def _money(basis: _InputBasis) -> frozenset[ProviderMoney]:
    """The money `basis` counts on the provider option it holds active
    (`harness.scope_choices.scope_money`), its reporters' own flows counting what
    its donors are (`_Origin.own_flows`); any money where the option states no scope."""
    return scope_money(_active_scope(basis), own_flows=basis.held.own_flows)


def _active_scope(basis: _InputBasis) -> ProviderScope | None:
    """The provider scope `basis`'s active option counts (`Option.provider_scope`),
    `None` where the option states none or the choice lists no such option."""
    option = basis.choice.active_option
    return option.provider_scope if option is not None else None


def _shared_basis(inputs: Sequence[_InputBasis]) -> str | None:
    """The option `inputs` are counted on together on their axis, or `None` when the user stated different ones.

    The one option the user stated on any of them (`_InputBasis.stated`),
    else the one the narrowest part is counted on (`_part_rank`), the first
    such input on a tie.
    """
    stated = {basis.choice.active for basis in inputs if basis.stated}
    if len(stated) > 1:
        return None
    return next(iter(stated)) if stated else max(inputs, key=lambda b: b.held.rank).choice.active


def _reads_on(step: EvidenceRequirement, axis: str) -> bool:
    """Whether `step` reads its inputs' options on `axis`: every calculation on the
    measure basis, and on the provider scope a comparison, share, quotient or
    sum (`_PROVIDER_ALIGNED`), whether it aligns them (`_aligns`) or accepts a
    provider difference, where a sum refuses counting flows twice and the
    difference is stated (`_difference_decisions`)."""
    return axis != PROVIDER_AXIS or isinstance(step, _PROVIDER_ALIGNED)


def _aligns(step: EvidenceRequirement, axis: str) -> bool:
    """Whether `step` counts its inputs on one option of `axis`: every calculation
    on the measure basis, and on the provider scope a comparison, share,
    quotient or sum (`_PROVIDER_ALIGNED`), unless the user asked for its inputs as each dataset
    counts them (`accept_provider_difference`)."""
    if axis != PROVIDER_AXIS:
        return True
    return isinstance(step, _PROVIDER_ALIGNED) and not step.accept_provider_difference


def _whole_parts(
    requirements: Sequence[EvidenceRequirement], *, unsearched: Mapping[str, DatasetRequirement]
) -> list[tuple[DatasetRequirement, DatasetRequirement, str]]:
    """Each whole the planner declared (`DatasetRequirement.whole_of`), with its part and the axis it leaves open (`_aligned_bases`).

    A whole is read only as the denominator of a share or quotient of its
    part, so alignment never reaches it through any other calculation.

    Raises:
        _Unservable: With `INVALID_PLAN` for a whole that cannot name its
            part so or that another calculation reads (`planner.plan.whole_of_fault`),
            `BASIS_UNKNOWN`, naming each part pinned to no
            dataset and not searched for yet (`unsearched`), the whole as the
            step wanting it, or `SHARE_WHOLE_UNDERIVABLE` when Search settled a
            part on no one dataset.
    """
    by_id = {requirement.id: requirement for requirement in requirements}
    wholes = [
        requirement
        for requirement in requirements
        if isinstance(requirement, DatasetRequirement) and requirement.whole_of is not None
    ]
    for whole in wholes:
        if (fault := whole_of_fault(whole, by_id)) is not None:
            raise _Unservable(_invalid_plan(fault))
    declared: list[tuple[DatasetRequirement, DatasetRequirement, str]] = [
        (whole, cast(DatasetRequirement, by_id[whole.whole_of.part]), whole.whole_of.axis)
        for whole in wholes
        if whole.whole_of is not None
    ]
    searches = {
        part.id: _unpinned(unsearched[part.id], whole.id)
        for whole, part, _ in declared
        if part.id in unsearched
    }
    if searches:
        raise _Unservable(_basis_unknown(list(searches.values())))
    for whole, part, _ in declared:
        if part.source_spec_id is None:
            raise _Unservable(
                share_whole_underivable(
                    whole_id=whole.id,
                    part_spec_id=None,
                    reason=f"Search settled {part.id!r} on no one dataset",
                    cause="part_unpinned",
                )
            )
    return declared


def _declared_wholes(
    declared: Sequence[tuple[DatasetRequirement, DatasetRequirement, str]],
    *,
    realigned: Mapping[str, str],
    shown: Mapping[str, PinSource],
    compiled: Sequence[AlignedPin],
) -> dict[str, str]:
    """The pin of each declared whole of `declared` (`_whole_parts`), by requirement id (`_aligned_bases`).

    A declared whole is its part's final pin, the one alignment moved the
    part to (`realigned`) else its own, with the declared axis left open
    (`whole_axis_held`, `planner.agent.compile_whole`) and every other choice
    held. `compiled` holds the wholes compiled so far
    (`AlignedPin.key`), so wholes declaring one part and axis share one
    compile.

    Raises:
        _Unservable: With `BASES_UNALIGNED`, naming each whole not compiled
            yet, once per part pin and axis (`Realignment.whole_axis`),
            or `SHARE_WHOLE_UNDERIVABLE` when Search's scope records for a
            part's pin hold nothing on the axis to leave open.
    """
    pins = {pin.key: pin.spec_id for pin in compiled}
    wanted: dict[tuple[str, str], Realignment] = {}
    wholes: dict[str, str] = {}
    for whole, part, axis in declared:
        pin = realigned.get(part.id) or cast(str, part.source_spec_id)
        block = _pin_block(pin, shown)
        scope = part_scope(pin, block)
        if scope is None or not whole_axis_held(scope, axis):
            reason, cause = unread_part(pin, axis, source=shown.get(pin), block=block, scope=scope)
            raise _Unservable(
                share_whole_underivable(
                    whole_id=whole.id, part_spec_id=pin, reason=reason, cause=cause
                )
            )
        held = pins.get((pin, (), axis))
        if held is not None:
            wholes[whole.id] = held
        else:
            wanted.setdefault(
                (pin, axis), Realignment(pin, (), whole.id, whole.id, whole_axis=axis)
            )
    if wanted:
        raise _Unservable(_bases_unaligned(list(wanted.values()), {}))
    return wholes


def _whole_slot(requirement_id: str) -> str:
    """The `Decision.slot` of the decision the whole `requirement_id`, compiled from
    its part, states (`_whole_decisions`). Nothing reads the slot back, since such a
    decision is known by its `caveat`."""
    return _derived_slot("share_whole_", requirement_id)


def _compiled_wholes(
    requirements: Sequence[EvidenceRequirement], *, aligned: Sequence[AlignedPin]
) -> dict[str, AlignedPin]:
    """The whole compiled from its part (`AlignedPin.whole_axis`) each declared whole
    of `requirements` (`DatasetRequirement.whole_of`) pins, by requirement id.

    A requirement stating no `whole_of` is no whole, though it pins a whole an
    earlier plan compiled: the rules for wholes read the plan as it stands,
    and use the compiled pins only for what they record."""
    compiled = {pin.spec_id: pin for pin in aligned if pin.whole_axis is not None}
    return {
        requirement.id: compiled[requirement.source_spec_id]
        for requirement in requirements
        if isinstance(requirement, DatasetRequirement)
        and requirement.whole_of is not None
        and requirement.source_spec_id in compiled
    }


def _whole_only_pins(
    requirements: Sequence[EvidenceRequirement], wholes: Mapping[str, AlignedPin]
) -> set[str]:
    """The spec_ids `wholes` (`_compiled_wholes`) pin that no requirement of
    `requirements` other than a declared whole pins, which state no scope rows. A
    plain requirement the planner pins to a whole's dataset is no whole: its rows stay."""
    return {pin.spec_id for pin in wholes.values()} - {
        requirement.source_spec_id
        for requirement in requirements
        if isinstance(requirement, DatasetRequirement) and requirement.whole_of is None
    }


def _without_replaced_whole_decisions(
    decisions: Sequence[Decision],
    evidence_requirements: Sequence[EvidenceRequirement],
    *,
    carried: Sequence[Decision],
    after: Sequence[EvidenceRequirement],
    wholes: Collection[str],
) -> tuple[list[Decision], list[EvidenceRequirement]]:
    """`decisions` and `evidence_requirements` less each planner decision only the
    wholes `wholes` cite, by requirement id, that fixes the whole's measure,
    and those citations.

    A planner decision fixing a total's measure chose the dataset it reads,
    and the total is compiled from its part's own dataset, so the
    decision can say something false of the plan, as "Source for total aid
    denominator: dac1 total aid" over a CRS whole. `_whole_decisions` states
    the whole in its place. A planner fixes no provider scope, measure basis
    or sector, since Search's choices state those, and the scope decisions
    written for them stay. Any other decision, such as the year a
    whole reads, stays cited by the whole, as do a user's decision and one a
    requirement other than those wholes cites. Citations are
    read over `after`, the plan's requirements once the patch applies, so a
    requirement the patch carries counts as well as one it states.

    The candidates are this call's `decisions` and those of the plan it
    amends (`carried`) it does not restate, so a whole restated with
    `whole_of` drops a citation an earlier turn wrote, and
    `_uncited_carried_decisions` then removes the decision.
    """
    citers: dict[str, set[str]] = {}
    for requirement in after:
        for slot in requirement.decision_ids:
            citers.setdefault(slot, set()).add(requirement.id)
    restated = {decision.slot for decision in decisions}
    replaced = {
        decision.slot
        for decision in [
            *decisions,
            *(decision for decision in carried if decision.slot not in restated),
        ]
        if decision.resolved_by is ResolvedBy.PLANNER_DEFAULT
        and decision.fixes is not None
        and decision.fixes.measure is not None
        and not (is_scope_choice(decision) or is_search_pin(decision))
        and decision.caveat != "derived_whole"
        and citers.get(decision.slot)
        and citers[decision.slot] <= set(wholes)
    }
    if not replaced:
        return list(decisions), list(evidence_requirements)
    return [decision for decision in decisions if decision.slot not in replaced], [
        requirement.model_copy(
            update={
                "decision_ids": tuple(
                    slot for slot in requirement.decision_ids if slot not in replaced
                )
            }
        )
        if replaced & set(requirement.decision_ids)
        else requirement
        for requirement in evidence_requirements
    ]


def _whole_decisions(
    wholes: Mapping[str, AlignedPin],
    requirements: Sequence[EvidenceRequirement],
    *,
    shown: Mapping[str, PinSource],
) -> dict[str, tuple[Decision, ...]]:
    """The decision each whole compiled from its part (`AlignedPin.whole_axis`) states,
    by the whole's requirement id, `wholes` holding each one's pin.

    It says, in reader words, that the share (or quotient, as the
    calculation reading the whole is one) divides by the total of its
    part's own dataset over the axis left open (`_whole_total`), and what
    both sides count alike (`_whole_shares`), naming the part's dataset by
    its title where that fits the rationale's limit (`_fitting_rationale`). The part is
    the one the whole declares (`DatasetRequirement.whole_of`), else the pin
    it was compiled from (`AlignedPin.origin`), so two wholes compiled to one
    pin each name their own. It is the planner's, qualifies the answer
    (`Decision.caveat` `derived_whole`), and is cited by the whole's
    requirement as a scope decision is (`_scope_decisions`).
    """
    by_id = {requirement.id: requirement for requirement in requirements}
    decisions: dict[str, tuple[Decision, ...]] = {}
    for requirement_id, whole in wholes.items():
        declared = by_id.get(requirement_id)
        part = (
            by_id.get(declared.whole_of.part)
            if isinstance(declared, DatasetRequirement) and declared.whole_of is not None
            else None
        )
        origin = (
            part.source_spec_id
            if isinstance(part, DatasetRequirement) and part.source_spec_id is not None
            else whole.origin
        )
        over_whole = requirements_over(requirement_id, by_id)
        divides = (
            "quotient"
            if any(
                isinstance(reader, QuotientRequirement)
                and reader.denominator.input_id in over_whole
                for reader in requirements
            )
            else "share"
        )
        part_block = _pin_block(origin, shown)
        whole_block = whole.digest.get("spec_block")
        part_held, whole_held = _block_scope(part_block), _block_scope(whole_block)
        dataset = _dataset_title(part_block)
        axis = cast(str, whole.whole_axis)
        total = _whole_total(axis, whole_block)
        alike = _whole_shares(axis, part_block, part_held, whole_held)
        chosen = f"the {dataset} {total}" if dataset is not None else f"the {total}"
        if len(chosen) > MAX_OPTION_CHARS:
            chosen = f"the {total}"

        # The dataset and the part's name, from the fullest to the shortest.
        titled_side = _side_name(part_block, part_held)
        side = _side_name(part_block, part_held, titled=False)
        named = (
            (
                f"the {dataset}, the same dataset as"
                if dataset is not None
                else "the same dataset as",
                f"the {titled_side} figures" if titled_side is not None else "its part",
            ),
            ("the same dataset as", f"the {side} figures" if side is not None else "its part"),
            ("the same dataset as", "its part"),
        )
        try:
            decision = Decision(
                slot=_whole_slot(requirement_id),
                description=f"Which total the {divides} divides by.",
                options=(chosen, "a total from another dataset"),
                chosen=chosen,
                resolved_by=ResolvedBy.PLANNER_DEFAULT,
                rationale=_fitting_rationale(
                    *(
                        f"The {divides} divides by the {total} from {source} {part_named}{alike}"
                        for source, part_named in named
                    )
                ),
                fixes=DecisionFixes(source_spec_id=whole.spec_id),
                caveat="derived_whole",
            )
        except ValidationError as exc:
            logger.warning("whole not noted: requirement=%s: %s", requirement_id, exc)
            continue
        decisions[requirement_id] = (decision,)
    return decisions


# What a whole and its part can count alike, by the axis the whole leaves open
# (`planner.plan.WholeOf.axis`), besides the providers (`_whole_shares`).
_WHOLE_ALIKE: Mapping[str, tuple[str, ...]] = {
    "sector": ("donors", "recipients", "measure basis"),
    "donor": ("recipients", "sectors", "measure basis"),
    "recipient": ("donors", "sectors", "measure basis"),
}
# The role column each party side of `_WHOLE_ALIKE` reads, and the scope axis each
# other name reads (`_dataset_holds`).
_ALIKE_COLUMNS: Mapping[str, str] = {"donors": "origin", "recipients": "destination"}
_ALIKE_AXES: Mapping[str, str] = {"sectors": "sector", "measure basis": "measure_basis"}


def _whole_total(axis: str, block: Mapping[str, Any] | None) -> str:
    """The total a whole over every `axis` is, `block` its compiled spec block, as
    its caveat names it: "total over every sector"; over a donor or recipient, the
    set Search binds on that role's column (`WHOLE_COLUMNS`), "total for official
    donors (DAC)", else, or where Search's groups are not readable, "total over every
    donor"."""
    column = WHOLE_COLUMNS.get(axis)
    label = next(
        (group.label for group in readable_groups(block or {}) or () if group.role == column),
        None,
    )
    return f"total for {label}" if label is not None else f"total over every {axis}"


def _whole_shares(
    axis: str,
    part_block: Mapping[str, Any] | None,
    part_scope_held: PartScope | None,
    whole_scope_held: PartScope | None,
) -> str:
    """What a whole over every `axis` and its part count alike, as the end of its
    caveat's rationale: ", so both sides count the same donors, recipients, measure
    basis and providers." `part_block` is the part's spec block, and
    `part_scope_held` and `whole_scope_held` the scope records of the part and the
    whole. It names only what the part's dataset has (`_dataset_holds`), so a DAC1
    donor total names no recipients or sectors, and the providers only where both
    hold the same provider option. Where they hold different options, or one states
    none, it says what each counts (`_providers_counted`), and where they share
    nothing else it says they count different providers only where both state
    one."""
    alike = [
        name for name in _WHOLE_ALIKE[axis] if _dataset_holds(part_block, part_scope_held, name)
    ]
    part, whole = (
        _active_option(part_scope_held, PROVIDER_AXIS),
        _active_option(whole_scope_held, PROVIDER_AXIS),
    )
    same = part is not None and whole is not None and part.option_id == whole.option_id
    if same:
        alike.append("providers")
    named = ", ".join(alike[:-1]) + f" and {alike[-1]}" if len(alike) > 1 else "".join(alike)
    if same or (part is None and whole is None):
        return (
            f", so both sides count the same {named}." if named else ", so both sides count alike."
        )
    providers = _providers_counted(whole, part)
    if named:
        return f", so both sides count the same {named}. {providers}"
    if part is not None and whole is not None:
        return f", so both sides count different providers. {providers}"
    return f". {providers}"


def _providers_counted(whole: Option | None, part: Option | None) -> str:
    """The sentence saying which providers a whole and its part count, `whole` and
    `part` the provider options they hold, `None` for one that states none."""
    if whole is None:
        return (
            f"The total does not say which providers it counts, and the part counts "
            f"{cast(Option, part).label}."
        )
    if part is None:
        return (
            f"The total counts {whole.label}, and the part does not say which providers it counts."
        )
    return f"The total counts {whole.label}, the part {part.label}."


def _dataset_holds(block: Mapping[str, Any] | None, scope: PartScope | None, name: str) -> bool:
    """Whether the dataset of `block`, a part's spec block, `scope` its scope record,
    has what `name` of `_WHOLE_ALIKE` counts: a party side the part names parties on
    or Search states open (`resolved.role_axis_facts`), or a scope axis the part's
    records hold a choice on."""
    column = _ALIKE_COLUMNS.get(name)
    if column is not None:
        definition = (block or {}).get("definition") or {}
        entities = (definition.get("identity") or {}).get("entities") or {}
        facts = ((block or {}).get("resolved") or {}).get("role_axis_facts") or ()
        return bool(entities.get(column)) or any(
            isinstance(fact, Mapping) and fact.get("role") == column for fact in facts
        )
    return scope is not None and any(c.axis == _ALIKE_AXES[name] for c in scope.choices)


def _derived_slot(prefix: str, requirement_id: str) -> str:
    """The `Decision.slot` of a decision the requirement `requirement_id` states
    (`_whole_slot`, `_difference_decisions`): `prefix` and the id, or `prefix`
    and a digest of the id where both pass an identifier's 64 characters."""
    slot = f"{prefix}{requirement_id}"
    if len(slot) <= MAX_IDENTIFIER_CHARS:
        return slot
    return f"{prefix}{hashlib.sha256(requirement_id.encode()).hexdigest()[:16]}"


def _card_names(
    requirements: Sequence[EvidenceRequirement],
    *,
    decisions: Sequence[Decision],
    scope_decisions: Mapping[str, Sequence[Decision]],
    sources: SourceReadings,
) -> dict[str, str]:
    """What the plan card names each retrieval of `requirements` by, by requirement
    id (`harness.plan_display.plan_words`), read off the plan they state once this
    call lands, not compiled yet: `decisions`, and each retrieval citing the scope
    decisions its pin states (`scope_decisions`, `scope_choice_decisions`), as
    `_scope_decisions` cites them, since its sector decision names it where Search
    states no concept. Each retrieval reads the source of the block its pin was
    shown with (`sources`), and is named afresh as the card of the compiled plan
    names it (`harness.plan_display.card_rows`), whatever the card it revises
    named it. Retrievals named alike are numbered in the order the plan states
    them, as the card numbers them (`harness.plan_display._card_order`)."""
    datasets = [
        requirement.model_copy(
            update={
                "decision_ids": (
                    *requirement.decision_ids,
                    *(
                        decision.slot
                        for decision in scope_decisions.get(requirement.source_spec_id, ())
                    ),
                )
            }
        )
        if requirement.source_spec_id is not None
        else requirement
        for requirement in requirements
        if isinstance(requirement, DatasetRequirement)
    ]
    # The card names a retrieval off the retrievals and decisions alone. The
    # calculations are left out, since the plan is not validated yet and one can
    # still read a requirement the patch removed.
    draft = Plan.model_construct(
        decisions=(*decisions, *(d for held in scope_decisions.values() for d in held)),
        evidence_requirements=tuple(datasets),
    )
    return dict(plan_words(draft, pinned=pinned_sources(datasets, sources)).names)


class _Difference(NamedTuple):
    """How a calculation states that its inputs differ on one axis (`_difference_decisions`).

    `prefix` starts the decision's slot (`_derived_slot`), `caveat` is its
    `Decision.caveat`, `description` and `options` its question and options,
    the first chosen, and `lead` its rationale's first sentence. `counts` and
    `unstated` end the sentence naming what one input counts, the first
    formatted with the label of the option its pin holds active, the second
    for an input whose pin holds none (`_difference_sentence`).
    """

    prefix: str
    caveat: DifferenceCaveat
    description: str
    options: tuple[str, str]
    lead: str
    counts: str
    unstated: str


# What each aligned axis's difference decision says (`_difference_decisions`).
_DIFFERENCES: Mapping[str, _Difference] = {
    PROVIDER_AXIS: _Difference(
        prefix="providers_differ_",
        caveat="providers_differ",
        description="Whether the inputs count the same providers' spending.",
        options=("each input as its dataset counts it", "every input on one provider scope"),
        lead="The inputs are set to different provider scopes, so they count different "
        "providers' spending.",
        counts="counts {label}",
        unstated="does not say which providers it counts",
    ),
    _BASIS_AXIS: _Difference(
        prefix="bases_differ_",
        caveat="bases_differ",
        description="Whether the inputs are counted on one measure basis.",
        options=("each input on its own measure basis", "every input on one measure basis"),
        lead="The inputs are counted on different measure bases, so their figures "
        "are not measured alike.",
        counts="is counted as {label}",
        unstated="does not say which measure basis it is counted on",
    ),
}
# The lead of a provider difference the question asked for (`accept_provider_difference`).
_ACCEPTED_PROVIDER_LEAD = (
    "The question asks for the inputs as their datasets count them, so they may "
    "count different providers' spending."
)
# The caveats a calculation's own decisions carry (`_difference_decisions`).
_CALCULATION_CAVEATS = frozenset(difference.caveat for difference in _DIFFERENCES.values())


def _difference_decisions(
    requirements: Sequence[EvidenceRequirement],
    *,
    sources: SourceReadings,
    plan_decisions: Sequence[Decision] = (),
    scope_decisions: Mapping[str, Sequence[Decision]] | None = None,
) -> dict[str, tuple[Decision, ...]]:
    """The decisions each comparison, share, quotient or sum states about its inputs'
    provider scopes and measure bases (`_DIFFERENCES`), by its requirement id.

    Every such calculation (`_PROVIDER_ALIGNED`) has an entry, empty when it
    states none, so a calculation the plan carries is restated as one it
    states (`_scope_decisions`). On each axis it reads each group of its
    inputs (`_input_groups`) apart, and names the inputs of every group that
    differs: a comparison of two shares compares numerators with numerators
    and denominators with denominators, each share stating its own. It states
    one when two inputs' records hold different options on the axis, as when
    the user set its inputs to different ones or alignment kept two sides
    apart (`_alignment_targets`), whatever an input with no such record
    counts; on the provider axis the options are compared by the scope they
    count (`Option.provider_scope`). One accepting a provider difference, the
    user having asked for its inputs as each dataset counts them
    (`accept_provider_difference`), states a provider one unless every pinned
    dataset input's scope record holds a provider scope and all match: an
    input whose record states none may count other providers, since nothing
    says what it counts. A group a ratio beneath supplies under an accepted
    provider difference (`_InputGroup.accepted`) reads as accepted, and a
    decision naming one is worded as the user's request. The decision says
    so in reader words, one sentence per input naming what it counts, or
    that its dataset does not say (`_difference_sentence`), each input named
    as the plan card names it on the plan `requirements`, `plan_decisions`
    and the scope decisions of their pins (`scope_decisions`) state
    (`_card_names`); where that passes the
    rationale's limit, each input is named without its dataset title, and
    where that still does, the first sentence is followed by how many inputs
    the calculation reads (`_fitting_rationale`). It is the planner's,
    qualifies the answer (`Decision.caveat`), and is cited by the calculation
    (`_scope_decisions`). A declared whole (`DatasetRequirement.whole_of`) is
    left out of a group holding its part: Search counts it on the options it
    offers for the whole, and the whole's own caveat says what it and its
    part count (`_whole_shares`).
    """
    pinned = [
        (requirement, block, _block_scope(block))
        for requirement in requirements
        if isinstance(requirement, DatasetRequirement) and requirement.source_spec_id is not None
        for block in [sources.block(requirement.source_spec_id)]
    ]
    # The card's names are read only once a difference needs its inputs named.
    names: dict[str, str] | None = None
    whole_parts = {
        requirement.id: requirement.whole_of.part
        for requirement in requirements
        if isinstance(requirement, DatasetRequirement) and requirement.whole_of is not None
    }
    groups = _input_groups(requirements)
    decisions: dict[str, tuple[Decision, ...]] = {}
    for step in requirements:
        if not isinstance(step, _PROVIDER_ALIGNED):
            continue
        stated: list[Decision] = []
        for axis, difference in _DIFFERENCES.items():
            held = [
                (requirement, block, _active_option(scope, axis))
                for requirement, block, scope in pinned
            ]
            differing: set[str] = set()
            flagged = axis == PROVIDER_AXIS and step.accept_provider_difference
            for group in groups[step.id]:
                accepted = axis == PROVIDER_AXIS and (
                    step.accept_provider_difference or group.accepted
                )
                members = group.members - {
                    whole for whole, part in whole_parts.items() if part in group.members
                }
                counted = {
                    _counted_as(option, axis)
                    for requirement, _, option in held
                    if requirement.id in members
                }
                if len(counted - {None}) > 1 or (accepted and None in counted):
                    differing |= members
                    flagged |= accepted and group.accepted
            inputs = [entry for entry in held if entry[0].id in differing]
            if not inputs:
                continue
            if names is None:
                names = _card_names(
                    requirements,
                    decisions=plan_decisions,
                    scope_decisions=scope_decisions or {},
                    sources=sources,
                )
            lead = _ACCEPTED_PROVIDER_LEAD if flagged else difference.lead
            try:
                decision = Decision(
                    slot=_derived_slot(difference.prefix, step.id),
                    description=difference.description,
                    options=difference.options,
                    chosen=difference.options[0],
                    resolved_by=ResolvedBy.PLANNER_DEFAULT,
                    rationale=_fitting_rationale(
                        *(
                            " ".join(
                                (
                                    lead,
                                    *(
                                        _difference_sentence(
                                            difference,
                                            names[requirement.id],
                                            block,
                                            option,
                                            titled=titled,
                                        )
                                        for requirement, block, option in inputs
                                    ),
                                )
                            )
                            for titled in (True, False)
                        ),
                        f"{lead} The calculation reads {len(inputs)} inputs.",
                    ),
                    caveat=difference.caveat,
                )
            except ValidationError as exc:
                logger.warning("%s not noted: step=%s: %s", difference.caveat, step.id, exc)
                continue
            stated.append(decision)
        decisions[step.id] = tuple(stated)
    return decisions


def _counted_as(option: Option | None, axis: str) -> object:
    """What an input holding `option` active on `axis` counts, for telling inputs
    apart (`_difference_decisions`): the provider scope it counts on the provider
    axis, the option itself on any other, `None` where it states none."""
    if option is None:
        return None
    return option.provider_scope if axis == PROVIDER_AXIS else option.option_id


def _fitting_rationale(*forms: str) -> str:
    """The first of `forms`, a caveat's rationale from its fullest form to its
    shortest, within `MAX_DECISION_RATIONALE_CHARS`, else the last, so a
    caveat too long for its fullest form takes the shortest and stays."""
    return next((form for form in forms if len(form) <= MAX_DECISION_RATIONALE_CHARS), forms[-1])


def _difference_sentence(
    difference: _Difference,
    name: str,
    block: Mapping[str, Any] | None,
    option: Option | None,
    *,
    titled: bool = True,
) -> str:
    """One sentence saying what an input counts on `difference`'s axis: "<name>
    counts <provider label>, from <dataset>." with `name` the input's name on the
    plan card (`_card_names`) and `option` the option its pin holds active, or that
    its dataset does not say. The dataset is left out where Search names none
    (`_dataset_title`), or without `titled`, the short form a long rationale falls
    back to."""
    concept = f"{name[:1].upper()}{name[1:]}"
    title = _dataset_title(block) if titled else None
    if option is None:
        if title is None:
            return f"{concept} {difference.unstated}."
        return f"{concept}, from {title}, {difference.unstated}."
    counts = difference.counts.format(label=option.label)
    if title is None:
        return f"{concept} {counts}."
    return f"{concept} {counts}, from {title}."


def _block_scope(block: Mapping[str, Any] | None) -> PartScope | None:
    """The scope record `block`, a spec block, states (`part_scope`), `None` where it
    states none this release reads."""
    return part_scope(str((block or {}).get("spec_id")), block)


def held_sector(block: Mapping[str, Any] | None) -> Option | None:
    """The option the scope choices of `block`, a spec block, hold on their sector
    choice (`_active_option`), `None` where they state none."""
    return _active_option(_block_scope(block), _SECTOR_AXIS)


def _dataset_title(block: Mapping[str, Any] | None) -> str | None:
    """The tables `block`'s figures are published in (`resolved.published_tables`),
    else the name of the dataset it counts from (`BlockDataset.name`), `None` when
    Search names neither or no block is held."""
    tables = ((block or {}).get("resolved") or {}).get("published_tables")
    if isinstance(tables, str) and tables:
        return tables
    return block_dataset(block).name


def _side_name(
    block: Mapping[str, Any] | None, scope: PartScope | None, *, titled: bool = True
) -> str | None:
    """A share side's name for the reader, `block` its spec block and `scope` its
    scope record: what Search says its series count (`compiler.sources.block_concept_label`),
    else the option its scope's sector choice holds, else, with `titled`, the dataset
    it counts from (`_dataset_title`), `None` when none is stated."""
    held = _active_option(scope, _SECTOR_AXIS)
    named = block_concept_label(block) or (held.label if held is not None else None)
    if named is not None:
        return named
    return _dataset_title(block) if titled else None


def _active_option(scope: PartScope | None, axis: str) -> Option | None:
    """The option `scope`, a part's scope record, holds active on `axis`, `None`
    where it states none."""
    if scope is None:
        return None
    return next((c.active_option for c in scope.choices if c.axis == axis), None)


def _pin_block(pin: str, shown: Mapping[str, PinSource]) -> Mapping[str, Any] | None:
    """The spec block of `pin` among the datasets the planner was shown, `None` when not shown."""
    source = shown.get(pin)
    return source.spec_block(pin) if source is not None else None


def axis_slot(scope: PartScope, axis: str) -> str | None:
    """The identity slot `scope`'s choice on `axis` binds, `None` when it offers no
    choice on `axis` binding one slot of its base identity."""
    choice = next((c for c in scope.choices if c.axis == axis), None)
    if choice is None:
        return None
    slots = {o.binding.slot for o in choice.options if isinstance(o.binding, SlotBinding)}
    held = {slot.get("property") for slot in scope.base.identity.get("slots") or ()}
    return next(iter(slots)) if len(slots) == 1 and slots <= held else None


# The role column a declared whole over every donor or recipient leaves open, by
# `planner.plan.WholeOf.axis`.
WHOLE_COLUMNS: Mapping[str, str] = {"donor": "origin", "recipient": "destination"}


def bound_parties(scope: PartScope, column: str) -> tuple[str, ...]:
    """The parties the part `scope` records binds on the role column `column`: those
    of the identity its active options compile to (`PartScope.compiled`), so a party
    a choice's option adds there (`EntitiesBinding`), as a region option adds its
    donors, counts as the base identity's own do."""
    return tuple((scope.compiled.identity.get("entities") or {}).get(column) or ())


def whole_axis_held(scope: PartScope, axis: str) -> bool:
    """Whether the part `scope` records binds something on `axis` a declared whole can
    leave open: for the sector, a choice binding one slot (`axis_slot`); for a donor
    or recipient, a party on that role's column (`WHOLE_COLUMNS`, `bound_parties`)."""
    column = WHOLE_COLUMNS.get(axis)
    if column is None:
        return axis_slot(scope, axis) is not None
    return bool(bound_parties(scope, column))


def unread_part(
    part_pin: str,
    axis: str,
    *,
    source: PinSource | None,
    block: Mapping[str, Any] | None,
    scope: PartScope | None,
) -> tuple[str, str]:
    """Why a declared whole's part pinned `part_pin` gives no choice on `axis` to
    leave open, as a clause and its `share_whole_underivable` cause: the first
    missing of its pin source (`part_not_shown`), that source's block
    (`part_block_missing`), the block's scope choices (`scope_unreadable`) and
    what the whole leaves open on `axis` (`whole_axis_held`, `axis_not_held`).
    `_declared_wholes` and `planner.agent.compile_whole` both name a failure by it."""
    if source is None:
        return f"{part_pin} is not among the datasets Search showed this turn", "part_not_shown"
    if block is None:
        return f"no record of {part_pin} from Search is held", "part_block_missing"
    if scope is None:
        return (
            f"Search's scope choices for {part_pin} are missing or unreadable",
            "scope_unreadable",
        )
    if axis in WHOLE_COLUMNS:
        return f"Search binds no {axis} on {part_pin} to leave open", "axis_not_held"
    return f"Search offers no {axis} choice on {part_pin} to leave open", "axis_not_held"


def share_whole_underivable(
    *, whole_id: str, part_spec_id: str | None, reason: str, cause: str
) -> dict[str, object]:
    """Return the `SHARE_WHOLE_UNDERIVABLE` problem for a declared whole that cannot be counted from its part's dataset.

    The planner can repair it: pin the part to a dataset Search offers the
    choice on, or pin a total of the whole's own and drop `whole_of`.
    `part_spec_id` is the part's pin, `None` where it has none, `reason`
    says why no whole was compiled, as a clause, and `cause` names that
    reason as a code:

    - `part_unpinned`: the part has no pin.
    - `part_not_shown`, `part_block_missing`, `scope_unreadable`,
      `axis_not_held`: the part's pin gives no choice on the axis to leave
      open (`unread_part`).
    - `groups_unreadable`: the part names a set on a role the whole keeps that
      this release cannot read.
    - `whole_refused`: Search refuses the whole, its code in `reason`
      (`requested_option_unavailable` where it cannot hold an option the part
      holds), or answers with no one dataset.
    - `part_outside_whole`, `whole_members_unstated`: Search's whole does not
      hold, or does not show it holds, a party the part names
      (`planner.agent._uncontained`).

    `_declared_wholes` answers the first two kinds, and
    `planner.agent.compile_whole` every kind but the first.
    `detail.requirement_id` names the whole for the refusal's sentence
    (`harness.sentences`).
    """
    return problem(
        ProblemCode.SHARE_WHOLE_UNDERIVABLE,
        f"{whole_id!r} states `whole_of`, a total counted from its part's own dataset with "
        "its sector, donor or recipient left open, so both count everything else alike. "
        f"{reason}. Pin the part to a dataset Search offers that choice on, or "
        f"drop `whole_of` and pin a total for {whole_id!r} of its own.",
        outcome=Outcome.INVALID_INPUT,
        detail={
            "requirement_id": whole_id,
            "whole": {"whole_id": whole_id, "part_spec_id": part_spec_id, "cause": cause},
        },
    )


def _pinned(requirement: EvidenceRequirement, pins: Mapping[str, str]) -> EvidenceRequirement:
    """`requirement` pinned to the dataset `pins` names for it, or as it stands."""
    if isinstance(requirement, DatasetRequirement) and requirement.id in pins:
        return requirement.model_copy(update={"source_spec_id": pins[requirement.id]})
    return requirement


def _with_found(
    evidence_requirements: Sequence[EvidenceRequirement],
    requirements: Sequence[EvidenceRequirement],
    pins: Mapping[str, str],
) -> list[EvidenceRequirement]:
    """This call's requirements, each pinned as `pins` says, then every other requirement `pins` re-pins, restated.

    The others are requirements only the plan this call amends states.
    """
    restated = [_pinned(requirement, pins) for requirement in evidence_requirements]
    own = {requirement.id for requirement in evidence_requirements}
    restated.extend(
        _pinned(requirement, pins)
        for requirement in requirements
        if requirement.id in pins and requirement.id not in own
    )
    return restated


def _basis_unknown(unpinned: Sequence[Unpinned]) -> dict[str, object]:
    """Return the `BASIS_UNKNOWN` problem naming each unpinned input of a calculation to search for.

    `planner.agent.with_source_reads` answers it by searching for each and
    proposing again, so the planner sees it only where no tool surface
    reaches Search.
    """
    named = "; ".join(f"{item.requirement_id!r}, an input of {item.step_id!r}" for item in unpinned)
    return problem(
        ProblemCode.BASIS_UNKNOWN,
        f"A calculation's inputs must share one measure basis and provider scope, and these "
        f"pin no dataset, so how they are counted is unknown: {named}. Search is asked for each before the plan is "
        "recorded, and could not be reached to do so; propose the plan again.",
        outcome=Outcome.REFUSED,
        detail={"search": [item.model_dump(mode="json") for item in unpinned]},
    )


def _bases_unaligned(
    wanted: Sequence[Realignment], by_axis: Mapping[str, Mapping[str, _InputBasis]]
) -> dict[str, object]:
    """Return the `BASES_UNALIGNED` problem naming each input to compile on its calculation's basis or provider scope.

    `by_axis` holds each input's choice by axis. `planner.agent.with_source_reads`
    answers it by compiling each and proposing again, so the planner sees it
    only where no tool surface reaches Search. `detail.requirement_id` names
    the first input for the refusal's sentence (`harness.sentences`).
    """

    def move(item: Realignment) -> str:
        if item.whole_axis is not None:
            return (
                f"{item.requirement_id!r} is wanted as the total over every "
                f"{item.whole_axis} of {item.spec_id}, its part's dataset"
            )
        bases = [
            (by_axis[option_axis(option_id)][item.requirement_id], option_id)
            for option_id in item.option_ids
        ]
        wanted = " and as ".join(
            f"{basis.label(option_id)} in place of {basis.label(basis.choice.active)}"
            for basis, option_id in bases
        )
        return f"{item.requirement_id!r}, an input of {item.step_id!r}, is wanted as {wanted}"

    moves = "; ".join(move(item) for item in wanted)
    return problem(
        ProblemCode.BASES_UNALIGNED,
        f"A calculation's inputs are counted on different measure bases or provider scopes: "
        f"{moves}. Search compiles each on its calculation's options before the plan is "
        "recorded, and could not be reached to do so; propose the plan again.",
        outcome=Outcome.REFUSED,
        detail={
            "requirement_id": wanted[0].requirement_id,
            "realign": [item.detail() for item in wanted],
        },
    )


def _basis_unservable(
    view: SessionView,
    proposer: Proposer,
    *,
    step_id: str,
    requirement_id: str,
    spec_id: str | None = None,
    option_id: str | None = None,
    reason: str,
) -> dict[str, object]:
    """Record and return the `OPERATION_UNSUPPORTED` problem for a calculation whose inputs cannot share one basis or provider scope.

    `detail.basis` names the calculation, the input, and its pin and the
    option it cannot be counted on, `None` and no option for an input with
    no pin.
    """
    return basis_unservable(
        spec_id=spec_id,
        option_ids=(option_id,) if option_id is not None else (),
        requirement_id=requirement_id,
        step_id=step_id,
        reason=reason,
        record=view.record_unsupported,
        proposer=proposer,
    )


def basis_unservable(
    *,
    spec_id: str | None,
    option_ids: Sequence[str],
    requirement_id: str,
    step_id: str,
    reason: str,
    record: Callable[[UnsupportedOperation], None],
    proposer: Proposer,
) -> dict[str, object]:
    """Record and return the `OPERATION_UNSUPPORTED` problem for an input that cannot be counted on its calculation's basis or provider scope.

    The arguments are a `Realignment`'s fields: the calculation
    (`step_id`), the input, its pin, `None` where it has none, and the
    options it is wanted on. Figures mixed on one axis are named by that
    axis, and with no option or options on both axes by either (`_MIXED`).
    `reason` says why, as a clause. `record` records the `unsupported`
    answer. `detail.basis` holds the fields.
    """
    axes = {option_axis(option_id) for option_id in option_ids}
    mixed = _MIXED[next(iter(axes)) if len(axes) == 1 else None]
    operation = UnsupportedOperation.UNSUPPORTED_ASK
    record(operation)
    refused = _unsupported(
        UnsupportedPlanError(operation, f"{reason}, so {step_id!r} would mix figures {mixed}"),
        proposer,
    )
    detail = cast("dict[str, object]", refused["detail"])
    detail["basis"] = {
        "spec_id": spec_id,
        "option_ids": list(option_ids),
        "requirement_id": requirement_id,
        "step_id": step_id,
    }
    return refused


def _uncited_carried_decisions(
    previous: Plan | None,
    decisions: Sequence[Decision],
    requirements: Sequence[EvidenceRequirement],
    *,
    removed: Collection[str],
    user_spoke: bool,
) -> list[str]:
    """The slot of each decision `previous` records that no requirement of the patched plan cites.

    `previous` validated, so every decision it records was cited. One the
    patched plan no longer cites lost its last citation to this patch, which
    unpinned, re-pinned, rewrote or removed every requirement resting on it.
    It goes with them, so the model never names a decision only to remove
    it. A slot this call restates or removes is left as the call says, since
    a decision the call states is the model's to cite. A user's decision is
    swept only once the user has spoken since `previous` (`user_spoke`),
    since the patch then acts on their new words; before that
    `_user_decision_refused` refuses the patch instead.

    Args:
        previous: The plan this call amends, or `None`.
        decisions: This call's own decisions, synthesized ones included.
        requirements: Every requirement the patched plan states.
        removed: The slots this call removes.
        user_spoke: Whether the user has spoken since `previous` was recorded.
    """
    if previous is None:
        return []
    restated = {decision.slot for decision in decisions} | set(removed)
    cited = {slot for requirement in requirements for slot in requirement.decision_ids}
    return [
        decision.slot
        for decision in previous.decisions
        if decision.slot not in restated
        and decision.slot not in cited
        and (user_spoke or decision.resolved_by is ResolvedBy.PLANNER_DEFAULT)
    ]


class _Contradiction(NamedTuple):
    """A carried decision fixing a value a requirement citing it now states otherwise.

    `backing` is every other requirement still citing the decision that it
    binds (`planner.plan.bound_fixes`) and that agrees with it.
    """

    decision: Decision
    requirement_id: str
    disagreement: FixDisagreement
    backing: tuple[str, ...]


def _contradicted_carried_decisions(
    previous: Plan | None,
    decisions: Sequence[Decision],
    requirements: Sequence[EvidenceRequirement],
    *,
    removed: Collection[str],
    user_spoke: bool,
) -> list[_Contradiction]:
    """Each decision `previous` records, carried unchanged, that a requirement still citing it contradicts.

    A patch replaces a requirement whole, so it can move a value (a period,
    say) a carried decision fixes. A decision this call states or removes is
    left out: `settle_decisions` read a stated one's values off the patched
    requirements. Only the requirements citing the decision are read: one
    the patch moved off it by citing another decision no longer rests on it.
    A user's decision is read only once the user has spoken since
    `previous` (`user_spoke`); before that it is `_user_decision_refused`'s.
    Each decision is named
    once, with the first requirement contradicting it and the first field
    (`planner.plan.fix_disagreements`).

    Args:
        previous: The plan this call amends, or `None`.
        decisions: This call's own decisions, synthesized ones included.
        requirements: Every requirement the patched plan states.
        removed: The slots this call removes.
        user_spoke: Whether the user has spoken since `previous` was recorded.
    """
    if previous is None:
        return []
    restated = {decision.slot for decision in decisions} | set(removed)
    found: list[_Contradiction] = []
    for decision in previous.decisions:
        if decision.slot in restated or not (
            user_spoke or decision.resolved_by is ResolvedBy.PLANNER_DEFAULT
        ):
            continue
        citing = [r for r in requirements if decision.slot in r.decision_ids]
        contradiction = next(
            (
                (requirement.id, disagreements[0])
                for requirement in citing
                if (disagreements := fix_disagreements(requirement, decision))
            ),
            None,
        )
        if contradiction is None:
            continue
        backing = tuple(
            requirement.id
            for requirement in citing
            if bound_fixes(requirement, decision) and not fix_disagreements(requirement, decision)
        )
        found.append(_Contradiction(decision, *contradiction, backing))
    return found


def _without_citations(
    requirements: Sequence[EvidenceRequirement],
    patch: Sequence[EvidenceRequirement],
    *,
    slots: frozenset[str],
) -> list[EvidenceRequirement]:
    """`patch` with every requirement of `requirements` citing one of `slots` restated without those citations.

    `requirements` is every requirement the patched plan states. One `patch`
    states is replaced in place; a carried one is appended, so `Plan.amend`
    replaces it. The decisions left uncited go with
    `_uncited_carried_decisions`.
    """
    rewritten = {
        requirement.id: requirement.model_copy(
            update={
                "decision_ids": tuple(
                    slot for slot in requirement.decision_ids if slot not in slots
                )
            }
        )
        for requirement in requirements
        if not slots.isdisjoint(requirement.decision_ids)
    }
    stated = {requirement.id for requirement in patch}
    return [
        *(rewritten.get(requirement.id, requirement) for requirement in patch),
        *(requirement for rid, requirement in rewritten.items() if rid not in stated),
    ]


def _user_decision_refused(
    previous: Plan | None,
    decisions: Sequence[Decision],
    requirements: Sequence[EvidenceRequirement],
    *,
    removed: Collection[str],
) -> dict[str, object] | None:
    """The refusal for a patch that drops or changes a decision the user made, or `None`.

    `admit` calls it only when the user has said nothing since `previous`
    was recorded, so nothing the patch does can rest on their words. Every
    `resolved_by=user` decision `previous` records then stands as the user
    made it: not removed (`removed`), cited by some requirement of the
    patched plan (`requirements`), and, as this call states it
    (`decisions`) or carries it, with the same `chosen`, `resolved_by` and
    `fixes`, which no requirement citing it contradicts. How the patch got
    there (a moved value, a renamed requirement, a dropped citation, a
    restatement) does not matter. `USER_DECISION_DROPPED` answers a removal
    or a lost last citation, `USER_DECISION_CONTRADICTED` any change.
    """
    if previous is None:
        return None
    stated = {decision.slot: decision for decision in decisions}
    for decided in previous.decisions:
        if decided.resolved_by is not ResolvedBy.USER:
            continue
        citing = [r for r in requirements if decided.slot in r.decision_ids]
        if decided.slot in removed or not citing:
            return _user_decision_dropped(decided, removed=decided.slot in removed)
        change = _user_decision_change(decided, stated.get(decided.slot, decided), citing)
        if change is not None:
            return _user_decision_contradicted(decided, *change)
    return None


def _user_decision_change(
    decided: Decision, standing: Decision, citing: Sequence[EvidenceRequirement]
) -> tuple[str | None, FixDisagreement] | None:
    """The first way `standing` departs from the user's `decided`, and the requirement showing it, or `None`.

    `chosen` and `resolved_by` first, then each field `fixes` states
    (`planner.plan.fixes_changes`), then a requirement citing it that states
    a value it fixes otherwise (`planner.plan.fix_disagreements`). The
    requirement is the first of `citing` stating the moved field otherwise
    than the user decided, `None` when none does.
    """
    for name in ("chosen", "resolved_by"):
        if getattr(standing, name) != getattr(decided, name):
            return None, FixDisagreement(name, getattr(decided, name), getattr(standing, name))
    for change in fixes_changes(decided.fixes, standing.fixes):
        showing = next(
            (
                requirement.id
                for requirement in citing
                if any(d.field == change.field for d in fix_disagreements(requirement, decided))
            ),
            None,
        )
        return showing, change
    for requirement in citing:
        disagreements = fix_disagreements(requirement, standing)
        if disagreements:
            return requirement.id, disagreements[0]
    return None


def _user_decision_dropped(decision: Decision, *, removed: bool) -> dict[str, object]:
    """The `USER_DECISION_DROPPED` refusal for a patch dropping `decision`.

    `detail` names the decision's `slot`, what the user `chose`, and whether
    the patch `removed` it or left no requirement citing it.
    """
    how = "names it in `removed_decisions`" if removed else "leaves no requirement citing it"
    return problem(
        ProblemCode.USER_DECISION_DROPPED,
        f"The user decided {decision.chosen!r} in decision {decision.slot!r}, and this patch "
        f"{how}. They have said nothing since the plan was recorded, so it stands: keep it "
        "cited on the requirements it fixes, or ask them through "
        f"`{REQUEST_INPUT_FUNCTION_CALL_NAME}`; once they answer, a patch may drop it.",
        outcome=Outcome.REFUSED,
        action=Action(call=REQUEST_INPUT_FUNCTION_CALL_NAME),
        detail={"slot": decision.slot, "chosen": decision.chosen, "removed": removed},
    )


def _user_decision_contradicted(
    decision: Decision, requirement_id: str | None, disagreement: FixDisagreement
) -> dict[str, object]:
    """The `USER_DECISION_CONTRADICTED` refusal for a patch changing what the user decided.

    `detail` names the decision's `slot`, the `field` (a `fixes` field,
    `chosen` or `resolved_by`), the value the user `decided`, the value the
    patch `stated`, and the `requirement_id` stating it, `None` when the
    change is the decision's own.
    """
    field, fixed, stated = disagreement
    decided = to_jsonable_python(fixed)
    now = to_jsonable_python(stated)
    where = f"{requirement_id!r} states" if requirement_id is not None else "this patch states"
    return problem(
        ProblemCode.USER_DECISION_CONTRADICTED,
        f"{where} {field} {now}, but the user decided {decided} in decision "
        f"{decision.slot!r} ({decision.chosen!r}), and they have said nothing since the plan "
        f"was recorded, so it stands as they made it. Keep {field} as they decided, or ask "
        f"them through `{REQUEST_INPUT_FUNCTION_CALL_NAME}`; once they answer, a patch may "
        "change it.",
        outcome=Outcome.REFUSED,
        action=Action(call=REQUEST_INPUT_FUNCTION_CALL_NAME),
        detail={
            "slot": decision.slot,
            "field": field,
            "decided": decided,
            "stated": now,
            "requirement_id": requirement_id,
        },
    )


def _shared_decision_contradicted(
    contradicted: Sequence[_Contradiction], proposer: Proposer
) -> dict[str, object] | None:
    """The `SHARED_DECISION_CONTRADICTED` refusal for the first of `contradicted` other requirements still rest on, or `None`.

    `admit` drops a planner's decision a patch contradicts, with its
    citations. One that still binds another requirement citing it (a period
    and a rank's `among`, say) would take that requirement's backing with
    it, so the patch is refused instead. `detail` names the decision's
    `slot`, the `field`, the value it `decided`, the value the patch
    `stated`, the `requirement_id` stating it, and the requirements
    `still_cited_by`. Its `action` calls `proposer` to restate the decision.
    """
    contradiction = next((c for c in contradicted if c.backing), None)
    if contradiction is None:
        return None
    decision = contradiction.decision
    moved = contradiction.requirement_id
    field, fixed, stated = contradiction.disagreement
    decided = to_jsonable_python(fixed)
    now = to_jsonable_python(stated)
    backing = ", ".join(repr(requirement_id) for requirement_id in contradiction.backing)
    return problem(
        ProblemCode.SHARED_DECISION_CONTRADICTED,
        f"{moved!r} now states {field} {now}, but decision {decision.slot!r} fixes "
        f"{decided}, and {backing} still rest on it, so it cannot be dropped. Restate "
        f"decision {decision.slot!r} in `decisions`: its values are read again off every "
        "requirement citing it, and a field they state differently is left off. Or cite on "
        f"{moved!r} a decision of its own and leave {decision.slot!r} out of its "
        "`decision_ids`.",
        outcome=Outcome.REFUSED,
        action=proposer.action(needs=("decisions",)),
        detail={
            "slot": decision.slot,
            "field": field,
            "decided": decided,
            "stated": now,
            "requirement_id": moved,
            "still_cited_by": list(contradiction.backing),
        },
    )


def _measure_fixed_by_pin(
    drafts: Sequence[DecisionDraft],
    requirements: Sequence[EvidenceRequirement],
    decisions: Iterable[Decision],
    measures: Mapping[str, MeasureChoice],
    proposer: Proposer,
) -> dict[str, object] | None:
    """The `MEASURE_FIXED_BY_PIN` refusal for the first of `drafts` restating a pin's measure, or `None`.

    A draft listing `measure` in `fixes` restates the flow when a requirement
    citing it also cites another decision `propose_plan` wrote for that
    requirement's own pin (`is_search_pin`, from `_measure_decisions`), so
    the plan would list the flow twice. `decisions` are every decision the
    patched plan holds, this call's settled ones and the carried ones it
    keeps. A draft restating the pin's own decision, or cited by a
    requirement whose pin has no such decision beside it, is let through.
    Only this call's own `drafts` are read: a decision carried unchanged
    from the plan being amended stands, and `Plan` validation never refuses
    one, so a stored plan holding one still loads. `detail` names the
    draft's `slot`, the `requirement_id` citing it, the pinned `spec_id`,
    the `measure_decision` recording the choice, the measure `chosen`, and
    the spec_id of each count the source offers by label (`counts`, from
    `measures`). Its `action` calls `proposer` to restate `decisions`.
    """
    by_slot = {decision.slot: decision for decision in decisions}
    for draft in drafts:
        if "measure" not in (draft.fixes or ()):
            continue
        for requirement in requirements:
            if (
                not isinstance(requirement, DatasetRequirement)
                or draft.slot not in requirement.decision_ids
            ):
                continue
            pinned = next(
                (
                    decision
                    for slot in requirement.decision_ids
                    if slot != draft.slot
                    and (decision := by_slot.get(slot)) is not None
                    and is_search_pin(decision)
                    and decision.fixes is not None
                    and decision.fixes.source_spec_id == requirement.source_spec_id
                ),
                None,
            )
            if pinned is not None:
                return _measure_fixed(draft, requirement, pinned, measures, proposer)
    return None


def _measure_fixed(
    draft: DecisionDraft,
    requirement: DatasetRequirement,
    pinned: Decision,
    measures: Mapping[str, MeasureChoice],
    proposer: Proposer,
) -> dict[str, object]:
    """The `MEASURE_FIXED_BY_PIN` refusal of `draft`, which `requirement` cites beside `pinned`, its pin's flow decision."""
    pin = requirement.source_spec_id
    choice = measures.get(pin) if pin is not None else None
    counts = dict(zip(choice.options, choice.spec_ids, strict=True)) if choice else {}
    others = ", ".join(
        f"{label} ({spec_id})" for label, spec_id in counts.items() if spec_id != pin
    )
    another_way = (
        f" To count the flow another way, pin that count's own `spec_id`: {others}."
        if others
        else ""
    )
    return problem(
        ProblemCode.MEASURE_FIXED_BY_PIN,
        f"Decision {draft.slot!r} lists `measure` in `fixes`, but {requirement.id!r} pins "
        f"{pin}, and a pin decides how its flow is counted: `propose_plan` records "
        f"{pinned.chosen} as decision {pinned.slot!r}. Cite {draft.slot!r} on no "
        "requirement pinning such a source, and leave it out of `decisions` when no other "
        f"requirement rests on it.{another_way} To choose between candidate readings, pin "
        "the chosen candidate's `spec_id` and state a `Decision` naming the candidates you "
        "weighed, with no `measure` in its `fixes`.",
        outcome=Outcome.REFUSED,
        action=proposer.action(needs=("decisions",)),
        detail={
            "slot": draft.slot,
            "requirement_id": requirement.id,
            "spec_id": pin,
            "measure_decision": pinned.slot,
            "chosen": pinned.chosen,
            "counts": counts,
        },
    )


def _decisions_after(
    previous: Plan | None, decisions: Sequence[Decision], *, removed: Collection[str]
) -> list[Decision]:
    """Every decision the plan holds once this call's patch lands: `previous`'s
    less those `removed` and those `decisions` restate, then `decisions`."""
    restated = {decision.slot for decision in decisions}
    carried = previous.decisions if previous is not None else ()
    return [
        *(d for d in carried if d.slot not in removed and d.slot not in restated),
        *decisions,
    ]


def _requirements_after(
    previous: Plan | None,
    requirements: Sequence[EvidenceRequirement],
    removed: Collection[str],
) -> list[EvidenceRequirement]:
    """Every requirement the plan states once this call's patch lands, in `Plan.amend`'s order.

    A removal naming no requirement is left for `Plan.amend` to refuse.
    """
    by_id = {
        requirement.id: requirement
        for requirement in (previous.evidence_requirements if previous else ())
    }
    by_id.update({requirement.id: requirement for requirement in requirements})
    return [
        requirement
        for requirement_id, requirement in by_id.items()
        if requirement_id not in removed
    ]
