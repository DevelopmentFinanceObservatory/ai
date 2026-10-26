"""The checks a proposed plan passes before it is recorded, over any session.

`admit` runs the whole chain a `propose_plan` call runs, in order: the lists
it states (`_Patch`), which plan it means (`_resolve_intent`) and that plan
as the model wrote it (`planner.plan.authored`), the pins it
moves back from their windowed copies (`_unwindowed_pins`), a referral the
user has not answered (`_referral_unanswered`), the decisions it settles
(`planner.plan.settle_decisions`), the measure decisions written on the
model's behalf (`_measure_decisions`), the carried decisions it
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

import unicodedata
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from itertools import chain
from typing import Annotated, Any, NamedTuple, Protocol

from compiler.evidence import open_axes_for
from compiler.sources import (
    PROVIDER_ACCOUNT,
    measure_named_by_query,
    open_entity_axes_of,
    source_from_spec_block,
)
from data_mcp.settings import Settings as DataMcpSettings
from google.adk.workflow.utils._workflow_hitl_utils import REQUEST_INPUT_FUNCTION_CALL_NAME
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
    DatasetRequirement,
    Decision,
    DecisionDraft,
    DecisionFixes,
    Decline,
    EvidenceRequirement,
    FirstYear,
    FixDisagreement,
    FullHistory,
    OneYearRead,
    PinSource,
    Plan,
    RankRequirement,
    RecordedDefinition,
    ResolvedBy,
    SessionState,
    SumOfRequirement,
    UnsupportedOperation,
    UnsupportedPlanError,
    WholeIsItsMemberError,
    WindowedPin,
    amendment_entry,
    authored,
    bound_fixes,
    cut_user_text,
    dataset_roots,
    fix_disagreements,
    fixes_changes,
    full_histories,
    full_history,
    inputs_of,
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
    pin_sources,
    recorded_identity,
    roles_not_kept,
    search_pin_slot,
    settle_decisions,
    shown_identity,
    split_of,
    unstated_period_reductions,
    unstated_result_scopes,
    windowed_pin,
    windowed_pins,
)
from planner.records import (
    AmbiguousEntityReading,
    EntityKey,
    GroupBy,
    Identifier,
    OpenPeriodFault,
    ResolvedSourceMention,
    RollingIssue,
    RollingPlanError,
    Year,
)
from planner.shapes import PLACE_PHRASE, pooling_remedy
from planner.tool_results import Action, Outcome, ProblemCode, ok, problem
from planner.workflow import OpenEntityAxis, ResolvedGroup, SelectedSource
from pydantic import BaseModel, Field, JsonValue, TypeAdapter, ValidationError, model_validator
from pydantic_core import to_jsonable_python
from translator.discovery import load_log, requirement_open_roles, requirement_revision
from translator.referral import USER_DECIDES, Referral, ReferralKind, load_referral
from translator.tool import authored_answer

# What a combination ask asks about: the `sum_of` requirement's id and the
# datasets its sides read (`_sum_datasets`).
type Combination = tuple[str, frozenset[str]]


class SessionView(Protocol):
    """The session a proposal is checked against, as the chain reads and writes it.

    `state` holds the slots `planner.plan` and the translator keep (the
    recorded plan, its rounds, pins shown and windowed, histories read,
    asks, the pending referral, the discovery log), read through their own
    loaders. The four readers answer from what the user was asked and
    answered in this session. The three writes are the chain's only side
    effects: an `unsupported` answer and the asks a `PERIOD_REDUCTION_UNSTATED`
    refusal names, both recorded even though the call is refused, and an
    admitted plan.
    """

    @property
    def state(self) -> SessionState:
        """The session's state slots."""
        ...

    def elected_spec_ids(self) -> frozenset[str]:
        """Every spec_id the user chose answering an ask that offered it."""
        ...

    def answers_since_referral(self) -> tuple[bool, frozenset[str]]:
        """Whether the user answered an ask raised after the pending referral, and the spec_ids those answers picked."""
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


def admit(
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
    measures = measure_choices(
        pins,
        compiled=previous.workflow.sources if previous is not None and previous.workflow else (),
        offered=chain.from_iterable(referral.offered().values()) if referral else (),
    )
    user_settled = _user_settled_measures(measures, view, pins)
    evidence_requirements = _stated_source_given_way(evidence_requirements, referral, view)
    settled, evidence_requirements = _measure_decisions(
        settle_decisions(
            patch.decisions,
            _requirements_after(previous, evidence_requirements, removed_evidence_requirements),
        ),
        evidence_requirements,
        previous=previous,
        measures=measures,
        user_settled=user_settled,
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
    length caps bound. Any other `ValueError` is a plan rule stated in words,
    so its text is the message. A rule that refused what the system cannot do
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
        return _invalid_plan(f"Invalid plan data: {exc}")
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
        or view.answers_since_referral()[0]
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
    chosen = offered & view.answers_since_referral()[1]
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


def _pinned_source(spec_id: str | None, view: SessionView) -> SelectedSource | None:
    """The source `spec_id` pins, as the record that states it; `None` when nothing on record does.

    The source is a block's the planner was shown, or one the
    translator's discovery recorded, such as a referral's candidate
    (`translator.discovery.DiscoveryLog.recorded_source`).
    """
    if spec_id is None:
        return None
    shown = pin_sources(view.state).get(spec_id)
    block = shown.spec_block(spec_id) if shown is not None else None
    if block is not None:
        try:
            return source_from_spec_block(block)
        except ValueError:
            return None
    return load_log(view.state).recorded_source(spec_id)


class _SumDatasets(NamedTuple):
    """The datasets both sides of a `sum_of` read, and what their pins state of them.

    `stated` says a pin states each dataset's provenance. `provider_flows`
    says each pin also states its provenance's account is `PROVIDER_ACCOUNT`:
    the OECD and ONE aid tables, whose skill says how they add up.
    """

    datasets: frozenset[str]
    stated: bool
    provider_flows: bool


def _sum_datasets(plan: Plan, requirement: SumOfRequirement, view: SessionView) -> _SumDatasets:
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
            source = _pinned_source(read.source_spec_id, view)
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
    plan: Plan, view: SessionView, *, previous: Plan | None
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
        datasets, stated, provider_flows = _sum_datasets(plan, requirement, view)
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
            and _sum_datasets(previous, requirement, view).datasets == datasets
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
    picked = view.answers_since_referral()[1]
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


def _entity_dcid_unoffered(
    plan: Plan, referral: Referral | None, previous: Plan | None
) -> dict[str, object] | None:
    """A new entity `dcid` must be a graph node Search offered for that entity.

    The options are the candidates of the reading the open
    `entity_name_ambiguous` referral (`referral`, as `_authored_referral`
    gives it) holds for the entity's `(name, role)` in its requirement. The planner settles the choice when the
    question decides it, so the user need not have picked it. An entity
    whose `(name, role, dcid)` the previous plan states for the requirement
    carries its settled choice forward.
    """
    for requirement in plan.evidence_requirements:
        if not isinstance(requirement, DatasetRequirement):
            continue
        kept: set[tuple[str, str | None, str | None]] = set()
        if previous is not None:
            old = next((r for r in previous.evidence_requirements if r.id == requirement.id), None)
            if isinstance(old, DatasetRequirement):
                kept = {(e.name, e.role, e.dcid) for e in old.entities}
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
            }
            if entity.dcid not in offered:
                return _invalid_plan(
                    "An entity's dcid must be a graph node an entity_name_ambiguous referral "
                    "offered for that entity. Leave dcid unset unless Search asked which node "
                    "the entity is."
                )
    return None


def _unrecordable(
    plan: Plan,
    view: SessionView,
    *,
    referral: Referral | None,
    stated: Plan,
    previous: Plan | None,
    reads_user_words: bool,
    proposer: Proposer,
) -> dict[str, object] | None:
    """Return the refusal for a built plan `propose_plan` may not record, or `None`.

    A plan states at least one requirement, unless it declines every ask
    (`_nothing_to_plan`); one that states none and makes a decline nothing
    earned is refused for the decline (`_decline_unearned`). A patch carries forward what it
    does not name, so the amended plan is checked whole for a source
    reference the user did not pick (`_source_ref_unpicked`), an entity
    node no referral offered (`_entity_dcid_unoffered`) and a pin a
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
        or _choice_not_offered(stated, referral, view)
        or _combination_unasked(stated, view, previous=previous)
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
    """
    elections = {source.spec_id: source.elected_measure for source in (*compiled, *offered)}
    for spec_id, source in shown.items():
        block = source.spec_block(spec_id) if source.settled and not source.other_count_of else None
        if block is None:
            continue
        try:
            elected = source_from_spec_block(block).elected_measure
        except ValueError:
            continue
        elections[spec_id] = elected
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
    decision `propose_plan` wrote for a search pin (`planner.plan.
    is_search_pin`) to the one its moved pin takes (`search_pin_slot`): that
    decision is rewritten for its new pin, and every citation of it cites the
    new slot, so it stays a search pin. A measure choice (`MeasureChoice`)
    keeps its options, since it reads the same over any period.
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
        if is_search_pin(decision) and decision.slot in slots:
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
            search_pin_slot(copy): search_pin_slot(origin) for copy, origin in to_origin.items()
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


@dataclass(frozen=True)
class _ShownDefinition:
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
    (`planner.plan.recorded_identity`), else the one `definition_lookup`
    answers: `admit` passes the discovery log's
    (`translator.discovery.DiscoveryLog.recorded_definition`), which holds
    a candidate the translator offered. For a requirement stating
    `period_end_open` it is otherwise the one the block the planner was
    shown carries (`planner.plan.shown_identity`): the translator can find
    a closed requirement's pin by searching its stated years, and no search
    finds a definition with no end year. A shown block that is not readable
    as a source (`source_from_spec_block`) puts none on record.
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
            if is_search_pin(decision):
                slots[decision.slot] = search_pin_slot(decision_pins[decision.slot])
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
            if cited is not None and cited.fixes is not None and cited.fixes.source_spec_id == pin:
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
