"""The studio endpoints: the v1 wire contract served over the turn loop.

The routes (`docs/studio-contract-v1.md`):

* `POST /api/studio/turn`: runs one turn and streams v1 frames as SSE.
* `POST /api/studio/interrupt`: resumes a paused turn; the response is a
  NEW SSE stream continuing it (contract delta 1).
* `POST /api/studio/resume`: carries on a running turn no segment holds,
  one whose client left; the response is a NEW SSE stream continuing it.
* `POST /api/studio/cancel`: stops the conversation's turn.
* `POST /api/studio/composition`: applies canvas mutations and returns the
  new manifest as plain JSON.
* `GET /api/studio/state`: the conversation as stored, read with no lease.

A turn runs on the turn loop (`harness.turn`) over the graph
`harness.graph.workflow.studio_graph` builds: `/turn` opens it at the
graph's start and runs its first segment, and `/interrupt` answers its
open pause and runs the next. The translator from the segment's events to
frames is `harness.studio_translate`; per-conversation state and the
composition rules are `harness.studio_state`; the Edit Card route is
`harness.studio_edit`. This module owns the routes and the `StudioRunner`
that runs a segment. Every stream (a segment of the turn loop, from `/turn`
or `/interrupt`; an `/interrupt` that rejects the plan; an Edit Card) runs
through an owner and a response of its own (`harness.studio_stream`), with
this module's `_SegmentWork`, `_RejectWork` or `_EditWork` as the work.

Conversation identity (contract delta 5): the client mints
`conversation_id`; the bearer token resolves to a principal
(`harness.auth`); the store and the turn's cursor are keyed on the pair.
Each `turn.request` opens a FRESH turn whose state starts empty. Turn 2's
planner drafts from a clean state, with no plan left over from turn 1,
while the composition manifest and the turn records persist across turns
in the store. Cross-turn context reaches the models as the turn's inputs:
the canvas digest (block titles + fact summaries) and the conversation
digest (the newest turn records); anything past their bounds is served on
demand by the recall function (`make_recall_part`).

Who asks (`studio_identity`): the website's proxy names a signed-in account in
`x-dfo-account` and the visitor's address in `x-dfo-caller`. The harness reads
both only on a request the bearer check admitted with `STUDIO_API_TOKEN` set,
keeps them on the turn (`harness.studio_state.TurnUsage`), and writes them on
the turn's spans and its `studio.turn` log line.
"""

from __future__ import annotations

import asyncio
import functools
import json
import logging
import math
import re
from collections.abc import AsyncIterator, Awaitable, Callable, Coroutine, Mapping, Sequence
from contextlib import (
    suppress,
)
from datetime import UTC, datetime
from typing import Any, ClassVar, Literal

from common.caller import ADDRESS_CALLER, set_supplied_caller, set_supplied_turn
from common.telemetry import flush_telemetry
from compiler.surface import SpecUse, ToolSurface, current_spec_uses
from executor.ledger import UNFINISHED_STATUSES, submission_content
from executor.run import UnrunnablePlan, approved_work, run_candidate
from fastapi import APIRouter, Header, HTTPException, Query, Request
from fastapi.responses import JSONResponse, Response
from fastapi.routing import APIRoute
from planner.plan import (
    PlanStatus,
    current_earlier_block,
    current_recorded_identity,
    plan_revision,
    stored_plan,
)
from planner.stated_scope import LICENCE_STATE_KEY
from pydantic import BaseModel, ValidationError
from sqlalchemy.ext.asyncio import AsyncConnection
from starlette.types import Receive, Scope

from harness.approval import ApprovalDecision, ApprovalRecord
from harness.auth import principal_for
from harness.card_unit import UnitMove, stored_unit_change, unit_move
from harness.conversation import (
    current_conversation_caller,
    current_conversation_id,
    current_conversation_key,
    current_turn_id,
    current_user_message,
    ledger_conversation_id,
)
from harness.decision_pick import is_decision_choice
from harness.early_mint import EarlyMints
from harness.graph.events import GraphEvent, StepProgress
from harness.graph.interfaces import Executor
from harness.graph.state import (
    APPLY_CHOICE_STATE_KEY,
    ORCHESTRATOR_PRIOR_LEDGER_KEY,
    TurnInput,
)
from harness.provenance import ProvenanceStore
from harness.run_ledger import ApprovedRun, StudioRunLedger
from harness.seeds import seeds_slot
from harness.settings import Settings
from harness.studio_edit import (
    CardSwap,
    EditClosed,
    EditEvent,
    EditFailed,
    edit_event_frames,
    edit_events,
    unit_events,
)
from harness.studio_state import (
    HEARTBEAT_INTERVAL_S,
    REPLACED_CARD_KEY,
    ConversationState,
    SegmentBusy,
    SegmentLost,
    TurnUsage,
    UnstreamedStatus,
    _apply_operations,
    _next_manifest,
    _PendingInterrupt,
    _PendingQuestion,
    _recorded_reply,
    _tool_error_response,
    _wire_id,
    record_unstreamed_span,
    renew_lease_now,
)
from harness.studio_store import (
    SEQ_BLOCK,
    ConversationExists,
    ConversationStale,
    Standing,
    StopPending,
    StoredConversation,
    StudioStore,
    TurnMoved,
    TurnStanding,
    asked_for_question,
    pause_entry,
    record_closed_turn,
    retry_delays,
    set_aside_question,
    stop_names_turn,
    turn_standing,
    write_pause_line,
)
from harness.studio_stream import Emit, LiveOwners, LockHold, open_stream
from harness.studio_translate import (
    RecordMarks,
    SegmentEvent,
    _EventRecords,
    _RecordingExecutor,
    _RecordingSurface,
    _stream_fatal,
    _Translator,
    closed_on_reply,
    held_labels,
    keep_turn_records,
    plan_approved,
    recorded_fatal,
)
from harness.turn import transcript as entries
from harness.turn.events import Paused, TurnEvent, TurnFailed, pause_state
from harness.turn.loop import (
    Answer,
    Graph,
    Outcome,
    Segment,
    abandon_on,
    reject,
    run_segment,
    stop_turn_on,
)
from harness.turn.store import (
    Cursor,
    CursorStale,
    TurnStatus,
    TurnStore,
    open_turn_on,
)
from harness.turn_budget import start_turn_budget
from harness.turn_records import CHIP_STALE_NOTE, is_count_choice, stored_choice, stored_count
from harness.unit_offer import (
    offered_unit_refusal,
    row_units,
    unit_option_id,
    unit_phrase,
    unit_words,
    whole_units,
)
from harness.wire import (
    MAX_CANCEL_REQUEST_BYTES,
    MAX_CLOSED_SINCE,
    MAX_COMPOSITION_REQUEST_BYTES,
    MAX_INTERRUPT_REQUEST_BYTES,
    MAX_RESUME_REQUEST_BYTES,
    MAX_TURN_REQUEST_BYTES,
    STORED_INTERRUPT,
    AddOp,
    ApplyChoice,
    ApplyUnit,
    CancelRequest,
    CancelSkipped,
    ClosedTurn,
    ComponentRefIntent,
    CompositionActor,
    CompositionManifest,
    CompositionMutation,
    ConversationRead,
    ElicitationAnswer,
    ElicitationInput,
    ErrorCode,
    ErrorFrame,
    FactSummary,
    InterruptDecision,
    InterruptResponse,
    OptionAnswer,
    OptionsAnswer,
    PendingPause,
    PlanApproved,
    PlanProposed,
    PlanRevisionStaleError,
    ResumeRequest,
    RetryMode,
    SkippedAnswer,
    StoredCard,
    TextAnswer,
    ToolError,
    TurnRequest,
    TurnState,
    TurnStatusName,
    UnitAnswer,
    UnitChoice,
    UnitOffer,
    declared_features,
)

logger = logging.getLogger(__name__)


# The form of the account digest the website sends in `x-dfo-account`: the
# hex HMAC-SHA256 of the account id, as its event export writes it.
_ACCOUNT_DIGEST = re.compile(r"[0-9a-f]{64}")


def studio_identity(
    *, account: str | None, caller: str | None, settings: Settings
) -> tuple[str | None, str | None]:
    """The account and visitor a Studio request names: `account` when it is
    an account digest, `caller` when it is an address caller
    (`common.caller.ADDRESS_CALLER`), `None` for each otherwise. Call it only
    once `principal_for` admitted the request, since the proxy that holds the
    bearer token vouches for both headers. With no
    `settings.studio_api_token` the route is open, nothing vouches, and both
    are `None`."""
    if settings.studio_api_token is None:
        return None, None
    account = (account or "").strip().lower()
    caller = (caller or "").strip().lower()
    return (
        account if _ACCOUNT_DIGEST.fullmatch(account) else None,
        caller if ADDRESS_CALLER.fullmatch(caller) else None,
    )


def _spec_use_sink(
    store: StudioStore, principal: str, conversation_id: str, trace_id: str
) -> Callable[[SpecUse, str], None]:
    """Where the turn `trace_id` of `(principal, conversation_id)` hears each
    spec it used (`compiler.surface.current_spec_uses`). It goes to the
    conversation the store holds live when the use is heard, so a state the
    store reloads meanwhile still gets it. A use heard once that conversation
    holds another turn, or once the turn is recorded
    (`ConversationState.turn_recorded`), is dropped, since the turn's record,
    which its log line and outcome are written from, was built without it."""

    def note(use: SpecUse, spec_id: str) -> None:
        conversation = store.current(principal, conversation_id)
        if (
            conversation is not None
            and conversation.trace_id == trace_id
            and not conversation.turn_recorded
        ):
            conversation.note_spec_use(use, spec_id)

    return note


# The wire decision on a plan interrupt, as the record `/interrupt` keeps of it.
_APPROVAL_DECISION = {
    InterruptDecision.APPROVE: ApprovalDecision.APPROVED,
    InterruptDecision.AMEND: ApprovalDecision.AMENDED,
    InterruptDecision.REJECT: ApprovalDecision.REJECTED,
}


# The answer kinds each question input takes.
_ANSWER_KINDS: dict[ElicitationInput, frozenset[str]] = {
    ElicitationInput.SINGLE_SELECT: frozenset({"option", "text", "skipped"}),
    ElicitationInput.MULTI_SELECT: frozenset({"options", "text", "skipped"}),
    ElicitationInput.TEXT: frozenset({"text", "skipped"}),
    ElicitationInput.UNIT: frozenset({"unit", "skipped"}),
}


def _answer_kinds(question: _PendingQuestion) -> frozenset[str]:
    """The answer kinds `question` takes: those of the input its frame sent
    (`_ANSWER_KINDS`), save free text on a unit question sent as a
    `single_select`, which offers none. A unit question that lists its whole units
    as options, as one restored from the conversation row does
    (`harness.studio_store`), also takes a pick of one, the answer of a page that
    was sent it as a `single_select`."""
    kinds = _ANSWER_KINDS[question.input]
    if question.native_input is ElicitationInput.UNIT:
        return kinds - {"text"}
    if question.input is ElicitationInput.UNIT and question.options:
        return kinds | {"option"}
    return kinds


def _answers_refusal(
    questions: tuple[_PendingQuestion, ...], answers: dict[str, ElicitationAnswer]
) -> str | None:
    """What is wrong with `answers` to `questions`, or `None`. Valid answers
    name every question exactly once, each of a kind its question's input
    takes (`_ANSWER_KINDS`), pick only that question's offered options, name
    a unit inside a unit question's offer, and skip only a skippable question.
    A question is checked in the form its frame sent (`_PendingQuestion.input`),
    so a unit or `multi_select` question sent as a `single_select` takes one
    `option`."""
    asked = [question.id for question in questions]
    missing = [question_id for question_id in asked if question_id not in answers]
    extra = sorted(set(answers) - set(asked))
    if missing or extra:
        return f"answers must name exactly the questions {asked}: missing {missing}, extra {extra}"
    for question in questions:
        if problem := _answer_refusal(question, answers[question.id]):
            return problem
    return None


def _answer_refusal(question: _PendingQuestion, answer: ElicitationAnswer) -> str | None:
    """What is wrong with `answer` to `question` (`_answers_refusal`), or `None`."""
    kinds = _answer_kinds(question)
    if answer.kind not in kinds:
        return f"question {question.id!r} is answered by {sorted(kinds)}, not {answer.kind!r}"
    match answer:
        case OptionAnswer(option_id=option_id) if option_id not in question.options:
            return f"option {option_id!r} is not offered by question {question.id!r}"
        case OptionsAnswer(option_ids=option_ids) if unknown := [
            option_id for option_id in option_ids if option_id not in question.options
        ]:
            return f"options {unknown} are not offered by question {question.id!r}"
        case UnitAnswer() if question.unit is not None and (
            problem := offered_unit_refusal(question.unit, _answered_unit(answer))
        ):
            return f"question {question.id!r}: {problem}"
        case SkippedAnswer() if not question.skippable:
            return f"question {question.id!r} cannot be skipped"
    return None


def _answered_unit(answer: UnitAnswer) -> UnitChoice:
    return UnitChoice(currency=answer.currency, prices=answer.prices, base_year=answer.base_year)


def _elicitation_refusal(pending: _PendingInterrupt, body: InterruptResponse) -> str | None:
    """What is wrong with an `answer` or `skip` to an elicitation, or `None`.
    A skip dismisses a skippable pause and carries nothing. An answer
    carries `answers` (`_answers_refusal`). A legacy entry
    (`_PendingInterrupt.legacy`) also takes its one answer as `payload`."""
    if body.decision is InterruptDecision.SKIP:
        if body.payload is not None or body.answers is not None:
            return "decision 'skip' takes no payload or answers"
        return None if pending.skippable else "this elicitation cannot be skipped as a whole"
    if body.payload is not None:
        if body.answers is not None or not pending.legacy:
            return "an elicitation answer carries answers, not a payload"
        return None if body.payload else "decision 'answer' requires answers"
    if body.answers is None:
        return "decision 'answer' requires answers, one per question id"
    return _answers_refusal(pending.questions, body.answers)


def _interrupt_refusal(
    pending_by_id: dict[str, _PendingInterrupt], body: InterruptResponse
) -> JSONResponse | None:
    """Everything that can be wrong with an interrupt.response, as one HTTP
    refusal: unknown id, a decision foreign to the interrupt kind, an
    elicitation's answers (`_elicitation_refusal`), the plan payload
    discipline (delta 12: amend carries the text it acts on, the other
    decisions must not smuggle one), an approve's revision, and a switch's
    revision and option (`_switch_refusal`).

    An answer naming a plan revision that is not the one awaiting an answer
    gets that plan's card back (`_stale_plan_refusal`), including when its
    interrupt is gone because a later card replaced it."""
    pending = pending_by_id.get(body.interrupt_id)
    if pending is None:
        current = _current_plan(pending_by_id)
        if (
            body.plan_revision is not None
            and current is not None
            and body.plan_revision != current.plan_revision
        ):
            return _stale_plan_refusal(current)
        return _tool_error_response(
            ToolError(
                error=ErrorCode.NOT_FOUND,
                retry=RetryMode.DO_NOT_RETRY,
                detail=f"no pending interrupt {body.interrupt_id!r} on this conversation",
            ),
            status_code=404,
        )
    allowed = (
        {InterruptDecision.ANSWER, InterruptDecision.SKIP}
        if pending.kind == "elicitation"
        else {
            InterruptDecision.APPROVE,
            InterruptDecision.AMEND,
            InterruptDecision.REJECT,
            InterruptDecision.SWITCH,
        }
    )
    detail: str | None = None
    fix: str | None = None
    if body.decision not in allowed:
        detail = f"decision {body.decision.value!r} does not apply to a {pending.kind} interrupt"
        fix = f"send one of: {', '.join(sorted(d.value for d in allowed))}"
    elif pending.kind == "elicitation":
        detail = _elicitation_refusal(pending, body)
    elif body.answers is not None:
        detail = "a plan interrupt takes no answers"
    elif body.decision is InterruptDecision.AMEND:
        if not body.payload:
            detail = "decision 'amend' requires a text payload"
    elif body.payload is not None:
        detail = f"decision {body.decision.value!r} does not take a payload"
    elif body.decision is InterruptDecision.SWITCH:
        return _switch_refusal(pending, body)
    elif body.plan_revision is not None and body.plan_revision != pending.plan_revision:
        return _stale_plan_refusal(pending)
    if detail is None:
        return None
    return _tool_error_response(
        ToolError(
            error=ErrorCode.BINDING_INVALID,
            retry=RetryMode.CORRECT_AND_RETRY,
            detail=detail,
            fix=fix,
        ),
        status_code=422,
    )


def _row_unit_offer(pending: _PendingInterrupt, choice_id: str) -> UnitOffer | None:
    """The units the row `choice_id` of `pending`'s plan card offers
    (`PlanDecision.unit`), `None` for a row offering none or a pause stored
    with no card."""
    decisions = (pending.card.decisions if pending.card is not None else None) or ()
    return next((d.unit for d in decisions if d.choice_id == choice_id), None)


def _switch_answer(body: InterruptResponse) -> dict[str, Any]:
    """What a `switch` answer hands the paused `approve` node under `switch`
    (`harness.graph.approve.decide`): the row and option, and the unit a
    unit row's switch names, when it names one."""
    switch: dict[str, Any] = {"choice_id": body.choice_id, "option_id": body.option_id}
    if body.unit is not None:
        switch["unit"] = body.unit.model_dump(mode="json", exclude_none=True)
    return switch


def _switch_refusal(pending: _PendingInterrupt, body: InterruptResponse) -> JSONResponse | None:
    """What is wrong with a `switch` on a plan card, as one HTTP refusal, or `None`.

    A switch names the revision of the plan the user saw, or is refused
    409 `plan_revision_stale`. It moves a choice the card shows
    (`_PendingInterrupt.choices`), a scope choice or a planner decision
    (`harness.decision_pick.is_decision_choice`), to an option the card
    offers as available or partial, other than the active one, or is refused
    422 `switch_option_unavailable`. A row offering a unit
    (`PlanDecision.unit`) also takes a switch carrying `unit`, any unit its
    offer holds other than the one it is in, named by `option_id` as
    `harness.unit_offer.unit_option_id` names it; a `unit` on any other row
    is refused the same way. `detail` is copy the card shows.
    """
    if body.plan_revision != pending.plan_revision:
        return _stale_plan_refusal(pending)
    choice = pending.choices.get(body.choice_id or "")
    option = choice.options.get(body.option_id or "") if choice is not None else None
    offer = _row_unit_offer(pending, body.choice_id or "")
    if choice is None:
        detail = "That choice cannot be switched on this plan."
    elif body.unit is not None:
        if offer is None:
            detail = "That choice offers no unit."
        elif unit_option_id(body.unit) != body.option_id:
            detail = "That option is not the unit picked."
        elif (refusal := offered_unit_refusal(offer, body.unit)) is not None:
            detail = f"That unit is not offered for this choice: {refusal}."
        elif body.option_id == choice.active:
            detail = f"The plan already uses {unit_phrase(body.unit, offer)}."
        else:
            return None
    elif option is None:
        detail = "That option is not offered for this choice."
    elif body.option_id == choice.active:
        verb = "uses" if is_decision_choice(body.choice_id or "") else "counts"
        detail = f"The plan already {verb} {option.label}."
    elif option.status == "unavailable":
        detail = f"{option.label} cannot be counted for this plan."
    else:
        return None
    return _tool_error_response(
        ToolError(
            error=ErrorCode.CONSTRAINT_VIOLATION,
            retry=RetryMode.CORRECT_AND_RETRY,
            constraint_id="switch_option_unavailable",
            detail=detail,
            fix="choose an option the plan card offers",
        ),
        status_code=422,
    )


def _chip_label(conversation: ConversationState, pick: ApplyChoice) -> str | JSONResponse:
    """The label of the option a chip moves to, or the HTTP refusal of a chip that cannot apply.

    A chip applies only on the answer that offered it. It is refused 422
    `scope_choice_stale` when that answer's turn (`ApplyChoice.turn`) is not
    the conversation's latest answered turn, or when the newest reading
    holding its choice (`harness.turn_records.stored_choice`), over the
    recorded turns and a turn still paused on its plan, is not the reading
    the chip switches (`ApplyChoice.spec_id`): a plan card's switch in a
    later turn compiles a new reading of the same choice. The chip is stale
    too when its choice is already on its option. An option the choice does
    not offer, or one Search cannot serve, is refused 422
    `switch_option_unavailable`, as a plan card's switch is. A count chip
    (`harness.turn_records.is_count_choice`) is read the same way from its
    count choice (`harness.turn_records.stored_count`). It is stale when the
    part reads a count other than that of `ApplyChoice.spec_id`, since a
    later fetch read one of its other counts, and its option must be one of
    the other counts that reading offers. `detail` is copy the client shows.
    """
    answered = [
        number
        for number, turn in enumerate(conversation.turns, start=1)
        if turn.get("assistant") and not turn.get("error")
    ]
    readings = [reading for turn in conversation.turns for reading in turn.get("readings") or ()]
    if not conversation.turn_recorded:
        readings.extend(conversation.turn_readings)
    if is_count_choice(pick.choice_id):
        refused = _count_chip_refusal(readings, pick, latest=answered[-1:] == [pick.turn])
    else:
        refused = _scope_chip_refusal(readings, pick, latest=answered[-1:] == [pick.turn])
    if isinstance(refused, str):
        return refused
    constraint, detail = refused
    return _tool_error_response(
        ToolError(
            error=ErrorCode.CONSTRAINT_VIOLATION,
            retry=RetryMode.DO_NOT_RETRY,
            constraint_id=constraint,
            detail=detail,
            pointer="/apply_choice",
        ),
        status_code=422,
    )


def _scope_chip_refusal(
    readings: list[dict[str, Any]], pick: ApplyChoice, *, latest: bool
) -> str | tuple[str, str]:
    """The label of the option a scope chip moves to, or its refusal's
    constraint and detail (`_chip_label`). `latest` is whether the chip's
    answer is the conversation's latest."""
    found = stored_choice(readings, pick.choice_id)
    reading, choice = found if found is not None else (None, None)
    option = (
        next((o for o in choice.options if o.option_id == pick.option_id), None)
        if choice is not None
        else None
    )
    if (
        not latest
        or reading is None
        or choice is None
        or reading.spec_id != pick.spec_id
        or (option is not None and pick.option_id == choice.active)
    ):
        return "scope_choice_stale", CHIP_STALE_NOTE
    if option is None:
        return "switch_option_unavailable", "That option is not offered here."
    if option.availability.status == "unavailable":
        return (
            "switch_option_unavailable",
            f"{option.label} cannot be counted for this answer.",
        )
    return option.label


def _count_chip_refusal(
    readings: list[dict[str, Any]], pick: ApplyChoice, *, latest: bool
) -> str | tuple[str, str]:
    """The label of the other count a count chip moves to, or its refusal's
    constraint and detail (`_chip_label`). `latest` is whether the chip's
    answer is the conversation's latest."""
    counted = stored_count(readings, pick.choice_id)
    if (
        not latest
        or counted is None
        or counted.reading.spec_id != pick.spec_id
        or counted.current != pick.spec_id
        or pick.option_id == pick.spec_id
    ):
        return "scope_choice_stale", CHIP_STALE_NOTE
    other = next((o for o in counted.reading.alternatives if o.spec_id == pick.option_id), None)
    if other is None:
        return "switch_option_unavailable", "That option is not offered here."
    return other.label


def _unit_move(conversation: ConversationState, pick: ApplyUnit) -> UnitMove | JSONResponse:
    """The unit change `pick` asks of `conversation` (`harness.card_unit.unit_move`),
    or its HTTP refusal: 422 `scope_choice_stale` for a card off the canvas or
    reading a spec other than `pick.spec_id`, and 422 `unit_not_offered` for a unit
    its offer does not hold. `detail` is copy the client shows, and the
    pointer is `/apply_unit`."""
    move = unit_move(conversation, pick)
    if isinstance(move, UnitMove):
        return move
    constraint, detail = move
    return _tool_error_response(
        ToolError(
            error=ErrorCode.CONSTRAINT_VIOLATION,
            retry=RetryMode.DO_NOT_RETRY,
            constraint_id=constraint,
            detail=detail,
            pointer="/apply_unit/unit" if constraint == "unit_not_offered" else "/apply_unit",
        ),
        status_code=422,
    )


def _unit_request_line(move: UnitMove) -> str:
    """The history record's text for a unit change until it lands:
    "Change Aid to Kenya to constant 2023 US dollars."."""
    words = unit_words(move.unit, move.offer)
    return f"Change {move.title} to {words[:1].lower()}{words[1:]}."


def _turn_refusal_or_chip(
    conversation: ConversationState, body: TurnRequest
) -> JSONResponse | str | None:
    """The refusal of `body` on `conversation`, else the label of the option
    its chip moves to (`_chip_label`), or `None` when it applies no chip. A
    follow-up about a card the conversation does not hold is refused 404:
    it would run as a plain turn with no card in view."""
    ref = body.component_ref
    if (
        ref is not None
        and ref.intent is ComponentRefIntent.FOLLOW_UP
        and ref.component_id not in conversation.components
    ):
        return _tool_error_response(
            ToolError(
                error=ErrorCode.NOT_FOUND,
                retry=RetryMode.DO_NOT_RETRY,
                detail=(
                    "This card is not part of this conversation, so the agent "
                    "cannot answer about it. Ask again without the card."
                ),
                pointer="/component_ref/component_id",
            ),
            status_code=404,
        )
    if body.apply_choice is None:
        return None
    return _chip_label(conversation, body.apply_choice)


def _turn_opening(
    conversation: ConversationState, body: TurnRequest, chip_label: str | None
) -> tuple[dict[str, Any], dict[str, Any], str]:
    """What the turn `body` begins on `conversation`, the working copy the
    begin-turn save stores, opens its cursor with: its inputs, its state and
    the user's text its transcript starts on."""
    focus = (
        body.component_ref.component_id
        if body.component_ref is not None
        and body.component_ref.intent is ComponentRefIntent.FOLLOW_UP
        else None
    )
    # The agent reads the canvas through `recall` and starts from
    # counts. The digests are the JSON text the agent reads.
    digest = conversation.canvas_summary_state(focus=focus)
    history = conversation.conversation_digest_state()
    seeds: list[Any] | dict[str, Any] | None = None
    if isinstance(body.seed_spec, list):
        # Each seed as the `seed` node takes it, or the refusal its
        # step fails with: no posted Spec block enters the turn.
        seeds = seeds_slot(body.seed_spec)
    elif body.seed_spec is not None:
        # One reference, the reserved shape: `graph.seed` reports it unused.
        seeds = body.seed_spec.model_dump(mode="json")
    text = body.message if chip_label is None else f"Count {chip_label}."
    inputs = TurnInput(
        message=text,
        canvas_digest=json.dumps(digest) if digest else None,
        conversation_digest=json.dumps(history) if history else None,
        seeds=seeds,
    )
    state: dict[str, Any] = {"surface": body.surface.value}
    # The orchestrator's planner renders the earlier turns' readings;
    # their blocks stay in the conversation (`current_earlier_block`).
    ledger = conversation.prior_ledger_state()
    if ledger:
        state[ORCHESTRATOR_PRIOR_LEDGER_KEY] = ledger
    # The conversation's licence to read partly reported and projected
    # years, which the turn's read of the user's words updates and
    # `keep_turn_records` keeps back.
    if conversation.licence:
        state[LICENCE_STATE_KEY] = dict(conversation.licence)
    if body.apply_choice is not None:
        # `graph.seed` applies the chip, and the turn neither searches nor plans.
        state[APPLY_CHOICE_STATE_KEY] = body.apply_choice.model_dump(mode="json")
    return inputs.model_dump(mode="json"), state, text


def _current_plan(pending_by_id: dict[str, _PendingInterrupt]) -> _PendingInterrupt | None:
    """The plan pause awaiting an answer, or `None`: the latest plan entry
    no segment is resuming."""
    return next(
        (
            pending
            for pending in reversed(pending_by_id.values())
            if pending.kind == "plan" and pending.resuming_segment is None
        ),
        None,
    )


def _stale_plan_refusal(current: _PendingInterrupt) -> JSONResponse:
    """409 `plan_revision_stale`: the answer names a plan the user is no
    longer being asked about. The body carries `current`'s card as the
    stream sent it, for the client to show in place of the one answered."""
    return _tool_error_response(
        PlanRevisionStaleError(
            error=ErrorCode.EDIT_CONFLICT,
            retry=RetryMode.REFRESH_THEN_RETRY,
            constraint_id="plan_revision_stale",
            detail="This plan has changed since you chose. Choose again on the plan shown now.",
            fix="answer the plan in current_plan",
            current_plan=current.card,
        ),
        status_code=409,
    )


def _given_answers(
    pending: _PendingInterrupt, body: InterruptResponse
) -> dict[str, ElicitationAnswer]:
    """An accepted `answer`'s answers by question id. A legacy entry's
    `payload` is its one question's answer. An offered option id is a pick,
    anything else free text."""
    if body.answers is not None:
        return body.answers
    assert body.payload  # `_elicitation_refusal` admits only a non-empty legacy payload
    (question,) = pending.questions
    if body.payload in question.options:
        return {question.id: OptionAnswer(kind="option", option_id=body.payload)}
    # `body` holds the payload within the cap on a request's own words, or is rebuilt
    # from a stored answer, which is carried whole (`STORED_INTERRUPT`).
    return {
        question.id: TextAnswer.model_validate(
            {"kind": "text", "text": body.payload}, context={STORED_INTERRUPT: True}
        )
    }


def _native_answer(question: _PendingQuestion, answer: ElicitationAnswer) -> ElicitationAnswer:
    """`answer` in the form of `question`'s own input. A pick on a unit question,
    sent as one (`_answer_kinds`) or in another form (`_PendingQuestion.native_input`),
    is the whole unit the option names (`whole_units`), and on a question sent as a
    `multi_select` one, one tick. Any other answer is itself."""
    if not isinstance(answer, OptionAnswer):
        return answer
    if ElicitationInput.UNIT in (question.input, question.native_input) and question.unit:
        unit = whole_units(question.unit)[answer.option_id]
        return UnitAnswer(kind="unit", **unit.model_dump())
    if question.native_input is ElicitationInput.MULTI_SELECT:
        return OptionsAnswer(kind="options", option_ids=[answer.option_id])
    return answer


def _resume_answer(question: _PendingQuestion, answer: ElicitationAnswer) -> dict[str, Any]:
    """One answer to `question` as the graph reads it
    (`harness.graph.state.resume_answers`): a pick names its candidate id,
    several picks theirs, a unit its parts, each in the form of the
    question's own input (`_native_answer`)."""
    match _native_answer(question, answer):
        case OptionAnswer(option_id=option_id):
            return {"kind": "option", "spec_id": option_id}
        case OptionsAnswer(option_ids=option_ids):
            return {"kind": "options", "spec_ids": list(option_ids)}
        case UnitAnswer() as unit:
            return {
                "kind": "unit",
                **_answered_unit(unit).model_dump(mode="json", exclude_none=True),
            }
        case TextAnswer(text=text):
            return {"kind": "text", "text": text}
        case SkippedAnswer():
            return {"kind": "skipped"}


def _answered_lines(pending: _PendingInterrupt, answers: dict[str, ElicitationAnswer]) -> list[str]:
    """The answers the turn's record keeps, readable: an option's label, with
    its spec_id when the option carries one, the labels of several picks, a
    unit in words ("Constant 2023 US dollars"), or the user's own words. A
    lone question's answer stands alone. With several, each follows its
    question. Skipped questions, and a legacy pick whose label was never
    kept, add nothing."""
    lines: list[str] = []
    for question in pending.questions:
        said: str | None
        match answers[question.id]:
            case OptionAnswer(option_id=option_id):
                said = question.options.get(option_id)
                if said and option_id.startswith("spec_"):
                    said = f"{said} ({option_id})"
            case OptionsAnswer(option_ids=option_ids):
                labels = [question.options.get(option_id) for option_id in option_ids]
                said = "; ".join(label for label in labels if label) or None
            case UnitAnswer() as answer:
                said = unit_words(_answered_unit(answer), question.unit)
            case TextAnswer(text=text):
                said = text
            case SkippedAnswer():
                said = None
        if said:
            lines.append(said if len(pending.questions) == 1 else f"{question.question} {said}")
    return lines


def _answer_turn_lines(
    pending: _PendingInterrupt, body: InterruptResponse
) -> list[tuple[str, bool]]:
    """The lines an answer adds to the turn's record, each with whether the
    user wrote it (`ConversationState.add_to_turn`): an elicitation's
    answers, a switch's choice, an amendment. The record must carry them or
    the next turn's digest shows only the un-refined ask. A switch's line
    names an option's label, which Search would read as scope the user
    stated, so it is no user's words. An approval, a rejection and a skip
    add none."""
    if pending.kind == "elicitation":
        if body.decision is InterruptDecision.SKIP:
            return []
        return [(line, True) for line in _answered_lines(pending, _given_answers(pending, body))]
    match body.decision:
        case InterruptDecision.SWITCH:
            assert body.choice_id is not None and body.option_id is not None
            if (offer := _row_unit_offer(pending, body.choice_id)) is not None:
                # A unit row names the unit in one form whether the unit is
                # one of its options or not (`_switch_refusal`).
                unit = body.unit or row_units(offer)[body.option_id]
                label = unit_phrase(unit, offer)
            else:
                label = pending.choices[body.choice_id].options[body.option_id].label
            verb = "use" if is_decision_choice(body.choice_id) else "count"
            return [(f"{verb} {label}", False)]
        case InterruptDecision.AMEND:
            return [(str(body.payload), True)]
    return []


def _composition_refusal(
    manifest: CompositionManifest | None, body: CompositionMutation
) -> JSONResponse | None:
    """Everything that can already be wrong with a composition.mutate on a
    conversation the store DOES hold: an unknown or mismatched composition
    id, or a stale `expected_revision`. `None` means the mutation may
    proceed."""
    bootstrapping = (
        manifest is None
        and body.composition_id is None
        and all(isinstance(op, AddOp) for op in body.operations)
    )
    no_composition = manifest is None and not bootstrapping
    mismatched = (
        manifest is not None
        and body.composition_id is not None
        and manifest.composition_id != body.composition_id
    )
    if no_composition or mismatched:
        return _tool_error_response(
            ToolError(
                error=ErrorCode.NOT_FOUND,
                retry=RetryMode.DO_NOT_RETRY,
                detail=f"no composition {body.composition_id!r} on this conversation",
            ),
            status_code=404,
        )
    current_revision = manifest.revision if manifest is not None else 0
    if body.expected_revision is not None and body.expected_revision != current_revision:
        return _tool_error_response(
            ToolError(
                error=ErrorCode.EDIT_CONFLICT,
                retry=RetryMode.REFRESH_THEN_RETRY,
                detail=f"expected revision {body.expected_revision}, current is {current_revision}",
                fix="re-render from the current manifest and retry",
            ),
            status_code=409,
        )
    return None


# ---------------------------------------------------------------------------
# The runner and routes
# ---------------------------------------------------------------------------


class StudioRunner:
    """Runs the segments of Studio turns on the turn loop.

    `create_graph` is called once with the recording executor and the
    recording surface already wrapped around the injected ones, so every
    turn runs on one graph while every executor call and every plan-step
    tool call is observed. `tool_surface` is `None` for a caller that has
    no `ToolSurface` to give it (a test that does not exercise plan steps);
    `create_graph` gets `None` through in that case, and the graph's
    `execute` node is what refuses loudly if it is ever reached. `turns`
    holds each turn's cursor (`harness.turn.store`), on the database
    `store` keeps conversations in. `early_mints` is the graph's own
    (`harness.early_mint`): a segment's response waits for the calls of the
    card it offered before it closes, while the request still has CPU
    (the owner's `settle`). A stream renews the conversation's lease every
    `heartbeat_seconds`. `owners` holds the owners of the streams running on
    this instance (`harness.studio_stream`), which the app's lifespan
    shutdown waits for before it closes the database. `run_ledger` is the
    graph's ledger, through which `/interrupt` admits an approved plan's run
    in the save that records the approval
    (`harness.run_ledger.StudioRunLedger.admit_with_approval`); with none, `execute`
    admits every run.
    """

    def __init__(
        self,
        *,
        create_graph: Callable[[_RecordingExecutor, _RecordingSurface | None], Graph[GraphEvent]],
        executor: Executor,
        store: StudioStore,
        turns: TurnStore,
        tool_surface: ToolSurface | None = None,
        edit_model: Any | None = None,
        provenance: ProvenanceStore | None = None,
        early_mints: EarlyMints | None = None,
        heartbeat_seconds: float = HEARTBEAT_INTERVAL_S,
        run_ledger: StudioRunLedger | None = None,
    ) -> None:
        self.edit_model = edit_model
        # The bare surface a card's unit change re-reads its spec through
        # (`harness.studio_edit.unit_events`), off the Activity feed.
        self.definition_surface = tool_surface
        self.run_ledger = run_ledger
        self.early_mints = early_mints
        self.store = store
        self.turns = turns
        self.heartbeat_seconds = heartbeat_seconds
        self.owners = LiveOwners()
        # `_RecordingExecutor` binds the store, to keep a create's write
        # token (`harness.studio_translate`) on the same conversations the
        # rest of this runner reads and writes.
        self.recorder = _RecordingExecutor(executor, store=self.store)
        self.surface_recorder = (
            _RecordingSurface(tool_surface, provenance=provenance)
            if tool_surface is not None
            else None
        )
        self.graph = create_graph(self.recorder, self.surface_recorder)

    async def aclose(self) -> None:
        """Release the injected executor's resources, through the recorder
        wrapping it. Called once, from `harness.server._lifespan`'s
        shutdown. The server's executor is `McpExecutor`, which opens one
        long-lived presentation-mcp client per conversation
        (`harness.mcp_client.McpExecutor._presentation_client`) and needs
        exactly this call to give them back; nothing else on the served path
        closes it."""
        await self.recorder.aclose()

    async def _heartbeat(
        self,
        principal: str,
        conversation_id: str,
        segment_id: str,
        renew_now: asyncio.Event,
    ) -> Literal[Outcome.CANCELLED, Outcome.LOST]:
        """Renew `segment_id`'s lease every `heartbeat_seconds`, or at once
        when `renew_now` is set, for a stream that runs no segment of the
        turn loop (the Edit Card stream, `_EditWork`), and return why the
        stream must stop: a renewal reported a Stop from any instance
        (`cancelled`), or found another instance's claim in its place
        (`lost`).

        A renewal that raises is retried every interval until the lease's
        deadline (`StudioStore.lease_deadline`, on this instance's own
        clock), and the stream then stops as lost: an instance this deaf to
        its own renewal cannot be trusted to still own the lease. The store's
        writes stop retrying at the same deadline (`DatabaseUnavailable`).
        """
        while True:
            with suppress(TimeoutError):
                await asyncio.wait_for(renew_now.wait(), self.heartbeat_seconds)
            renew_now.clear()
            try:
                renewal = await self.store.renew(principal, conversation_id, segment_id)
            except Exception:
                logger.exception(
                    "studio segment heartbeat: conversation %s segment %s renewal failed",
                    conversation_id,
                    segment_id,
                )
                if self.store.now() < self.store.lease_deadline(principal, conversation_id):
                    continue
                return Outcome.LOST
            if not renewal.held:
                return Outcome.LOST
            if renewal.stop_id is not None:
                return Outcome.CANCELLED


# An event of a segment's drive, marked with the calls recorded before it.
type _MarkedEvent = tuple[SegmentEvent, RecordMarks]


def event_key(event: SegmentEvent) -> tuple[str, str] | None:
    """The key a segment's event waits for its owner under (`Emit`): a plan
    step's progress report replaces a waiting report of the same step, so a
    stream that falls behind the run shows where each step is now."""
    return ("progress", event.step_id) if isinstance(event, StepProgress) else None


class _SegmentWork:
    """The work of a claimed segment that runs the turn loop, for its owner
    (`harness.studio_stream.Work`): one segment of the turn at `cursor`
    (`harness.turn.loop.run_segment`) under `segment_id`, whose events the
    owner translates into wire frames (`_Translator`). `/turn` opens the turn
    and runs its first segment; `/interrupt` gives `answer`, the answer to
    the turn's open pause (`None` when a retry carries on a running turn),
    and `approving`, the plan card a first answer approves, which
    `plan.approved` states once the graph takes the approval, and
    `restored`, the `plan.approved` frames of approvals the turn took on an
    earlier segment, which it records on the turn (`record_approved`) and
    sends before its first frame. With
    `replay`, the segment runs no node: it emits the pause the paused
    cursor holds and ends `paused` (`_replay_pause`).

    While it runs, `conversation.renew_now` is the segment's, so a Stop
    posted on this instance reaches its heartbeat at once
    (`harness.studio_state.renew_lease_now`). A segment that pauses or ends
    closes on `_Translator.finish`'s frames, with the turn's records as its
    cursor stores them (`keep_turn_records`). A turn that failed ends on the
    translator's frame for its `TurnFailed` event, or on one worded from the
    error the cursor records when the close committed with no event
    (`recorded_fatal`). A taken Stop whose close could not be written
    (`unclosed` with no `TurnFailed`) leaves the turn open with that Stop
    live, so the work ends `cancelled` and the owner's closing write closes
    the turn. A turn another instance abandoned ends `abandoned`. One this
    segment closed `abandoned` on a node's failure ends `failed` on that
    failure's frame. Cancelling the drive cancels the segment, which leaves
    the cursor as last written and its runs unfinished, for the next request
    to carry on. A step the executor finished after the last translated
    event belongs to this segment, so the recorders are emptied however it
    ends."""

    def __init__(
        self,
        runner: StudioRunner,
        *,
        principal: str,
        conversation: ConversationState,
        cursor: Cursor,
        segment_id: str,
        answer: Answer | None = None,
        approving: PlanProposed | None = None,
        restored: Sequence[PlanApproved] = (),
        replay: bool = False,
    ) -> None:
        self._runner = runner
        self._principal = principal
        self._conversation = conversation
        self._cursor = cursor
        self._segment_id = segment_id
        self._answer = answer
        self._replay = replay
        self._restored = tuple(restored)
        self._translator = _Translator(
            conversation=conversation,
            recorder=runner.recorder,
            surface_recorder=runner.surface_recorder,
            approving=approving,
            restored=restored,
        )
        # The drive's half of the translation: it marks each event with the
        # calls recorded before it, and reads nothing else.
        self._marks = _EventRecords(runner.recorder, runner.surface_recorder)
        self._renew_now = asyncio.Event()
        # Whether a `TurnFailed` event was translated.
        self._failed = False

    def begin(self) -> None:
        """Make the segment's `renew_now` the conversation's, and keep the
        turn's records so far, for a segment that resumes a turn another
        instance began."""
        self._conversation.renew_now = self._renew_now
        keep_turn_records(self._conversation, self._cursor.state)
        for approved in self._restored:
            self._conversation.record_approved(approved)

    async def drive(self, emit: Emit[_MarkedEvent]) -> Segment:
        """Run the segment, emitting each event marked with the calls
        recorded before it (`_EventRecords`). A plan step's progress report
        replaces one of the same step that is waiting."""
        runner, marks = self._runner, self._marks

        def recorded(event: GraphEvent | TurnEvent) -> None:
            emit(marks.recorded(event), key=event_key(event))

        if self._replay:
            # The pause the cursor holds, as the segment that wrote it
            # emitted it, with no node run.
            pause = self._cursor.pause
            assert pause is not None
            recorded(
                Paused(
                    kind=pause.kind,
                    interrupt_id=pause.interrupt_id,
                    round=pause.round,
                    payload=pause.payload,
                    state=pause_state(self._cursor.state),
                )
            )
            return Segment(cursor=self._cursor, outcome=Outcome.PAUSED)
        return await run_segment(
            runner.turns,
            runner.store,
            self._cursor,
            principal=self._principal,
            conversation_id=self._conversation.conversation_id,
            segment_id=self._segment_id,
            graph=runner.graph,
            emit=recorded,
            answer=self._answer,
            heartbeat_seconds=runner.heartbeat_seconds,
            renew_now=self._renew_now,
            paused=self._paused,
        )

    def _paused(self, cursor: Cursor) -> None:
        """Write the `studio.turn` line and `turn.pause` outcome of the pause
        `cursor` holds, once its write committed (`run_segment`'s `paused`):
        a line exists exactly when the cursor stores the pause, whatever cut
        or failed closing write follows it."""
        write_pause_line(self._conversation.conversation_id, self._conversation)

    async def frames(self, event: _MarkedEvent) -> list[BaseModel]:
        """The frames of one event `drive` emitted."""
        taken, marks = event
        self._failed = self._failed or isinstance(taken, TurnFailed)
        return await self._translator.translate_recorded(taken, marks)

    async def end(self, result: Segment) -> tuple[Outcome, list[BaseModel]]:
        """How the segment `drive` returned ended, and its closing frames."""
        segment = result
        conversation = self._conversation
        keep_turn_records(conversation, segment.cursor.state)
        outcome = segment.outcome
        assert outcome is not None
        match outcome:
            case Outcome.DONE:
                return outcome, await self._translator.finish(
                    new_question=set_aside_question(segment.cursor),
                    asked_for_question=asked_for_question(segment.cursor),
                )
            case Outcome.PAUSED:
                return outcome, await self._translator.finish()
            case Outcome.FAILED if not self._failed:
                return outcome, [recorded_fatal(conversation, segment.cursor.error or {})]
            case Outcome.UNCLOSED if not self._failed:
                return Outcome.CANCELLED, []
            case Outcome.ABANDONED if self._failed:
                return Outcome.FAILED, []
        return outcome, []

    def fatal(self, error: BaseException) -> ErrorFrame:
        """The stream-fatal frame for `error` (`_stream_fatal`), which
        records the turn failed on it. A cancellation of the work's own is a
        fault of the segment's frames."""
        if not isinstance(error, Exception):
            error = RuntimeError("the segment's frames were cancelled")
        return _stream_fatal(self._conversation, error)

    def close(self) -> None:
        """Let go of `renew_now` and empty the recorders."""
        _close_work(self._conversation, self._renew_now, self._runner)


class _RejectWork:
    """The work of an `/interrupt` that declines the plan, for its owner
    (`harness.studio_stream.Work`). The route closed the turn's cursor
    before the stream (`harness.turn.loop.reject`); `begin`, which the owner
    runs before anything else, records the turn closed on its reply, so a
    client that leaves before the first frame still leaves it recorded.
    The stream then sends that reply and `done`. Nothing was computed, so
    its drive runs nothing."""

    REPLY = "Plan rejected."

    def __init__(self, conversation: ConversationState) -> None:
        self._conversation = conversation
        self._frames: list[BaseModel] = []

    def begin(self) -> None:
        """Record the turn closed on its reply."""
        self._frames = [*closed_on_reply(self._conversation, self.REPLY, suggestions=False)]

    async def drive(self, emit: Emit[None]) -> None:
        return None

    async def frames(self, event: None) -> list[BaseModel]:
        raise AssertionError("the rejection emits no events")

    async def end(self, result: None) -> tuple[Outcome, list[BaseModel]]:
        """`done`, with the reply's frames."""
        return Outcome.DONE, self._frames

    def fatal(self, error: BaseException) -> ErrorFrame:
        """The stream-fatal frame for `error` (`_stream_fatal`)."""
        if not isinstance(error, Exception):
            error = RuntimeError("the rejection's frames were cancelled")
        return _stream_fatal(self._conversation, error)

    def close(self) -> None:
        return None


class _EditWork:
    """The work of an Edit Card stream, for its owner
    (`harness.studio_stream.Work`): the card edit's calls
    (`harness.studio_edit.edit_events`) under `segment_id`, which runs no
    turn loop, beside a heartbeat that renews the lease
    (`StudioRunner._heartbeat`), and the owner's frames of the edit's events
    (`harness.studio_edit.edit_event_frames`). While it runs,
    `conversation.renew_now` is the heartbeat's, so a Stop posted on this
    instance reaches it at once.

    A card's unit change (`harness.studio_edit.unit_events`) runs as this
    work too, `events` giving its events in place of the edit's.

    The edit ends on its reply and `done`, a card swap
    (`harness.studio_edit.CardSwap`) carrying its reply, or on a stream-fatal `error`
    frame that records the turn failed, and the work ends `done` or
    `failed`. A heartbeat that reads a Stop first cancels the edit and the
    work ends `cancelled`. One that finds the lease another's, or cannot
    renew it before the lease's deadline, ends it `lost`, as does a card
    write fenced out by another instance's takeover or unable to reach the
    database (`StudioStore.change_canvas`). An ending the edit emitted
    before the heartbeat stopped it stands."""

    def __init__(
        self,
        runner: StudioRunner,
        *,
        principal: str,
        conversation: ConversationState,
        segment_id: str,
        events: Callable[[], AsyncIterator[EditEvent]],
    ) -> None:
        self._runner = runner
        self._principal = principal
        self._conversation = conversation
        self._segment_id = segment_id
        self._events = events
        self._renew_now = asyncio.Event()

    def begin(self) -> None:
        """Make the heartbeat's `renew_now` the conversation's."""
        self._conversation.renew_now = self._renew_now

    async def drive(self, emit: Emit[EditEvent]) -> Outcome:
        """Run the edit's calls, emitting each event, beside the heartbeat,
        and return how the edit ended."""
        runner, conversation = self._runner, self._conversation
        ended: EditEvent | None = None

        async def edit() -> None:
            nonlocal ended
            async for event in self._events():
                emit(event)
                if isinstance(event, (EditClosed, EditFailed, CardSwap)):
                    ended = event

        editing = asyncio.create_task(edit())
        heartbeat = asyncio.create_task(
            runner._heartbeat(
                self._principal, conversation.conversation_id, self._segment_id, self._renew_now
            )
        )
        try:
            await asyncio.wait({editing, heartbeat}, return_when=asyncio.FIRST_COMPLETED)
            if not editing.done():
                editing.cancel()
                await asyncio.wait({editing})
            if ended is not None:
                return Outcome.FAILED if isinstance(ended, EditFailed) else Outcome.DONE
            if editing.cancelled():
                return heartbeat.result()
            editing.result()
            raise RuntimeError("the edit ended with no ending frame")
        finally:
            editing.cancel()
            heartbeat.cancel()
            await asyncio.wait({editing, heartbeat})

    async def frames(self, event: EditEvent) -> list[BaseModel]:
        """The frames of one event the edit emitted."""
        return await edit_event_frames(
            event, conversation=self._conversation, store=self._runner.store
        )

    async def end(self, result: Outcome) -> tuple[Outcome, list[BaseModel]]:
        """How the edit ended; its frames are the events' own."""
        return result, []

    def fatal(self, error: BaseException) -> ErrorFrame:
        """The stream-fatal frame for `error` (`_stream_fatal`), which
        records the turn failed on it."""
        if not isinstance(error, Exception):
            error = RuntimeError("the edit's frames were cancelled")
        return _stream_fatal(self._conversation, error)

    def close(self) -> None:
        """Let go of `renew_now` and empty the recorders."""
        _close_work(self._conversation, self._renew_now, self._runner)


def _close_work(
    conversation: ConversationState, renew_now: asyncio.Event, runner: StudioRunner
) -> None:
    """Let go of a work's `renew_now`, when the conversation still holds it,
    and empty the recorders of the calls its stream did not translate,
    their counts included, since no event of the segment is left to drain
    (`harness.studio_translate._Records.forget`)."""
    if conversation.renew_now is renew_now:
        conversation.renew_now = None
    runner.recorder.release(current_conversation_id.get())
    if runner.surface_recorder is not None:
        runner.surface_recorder.release(current_conversation_id.get())


def _turn_moved_response(*, after_turn: int | None, latest: int) -> JSONResponse:
    """The 409 `edit_conflict` (`constraint_id` `turn_moved`) for a `/turn`
    whose `after_turn` is not the conversation's latest turn, `latest`: a
    request on another page began a turn since the page read the state."""
    return _tool_error_response(
        ToolError(
            error=ErrorCode.EDIT_CONFLICT,
            retry=RetryMode.REFRESH_THEN_RETRY,
            constraint_id="turn_moved",
            pointer="/after_turn",
            detail=f"the request follows turn {after_turn}, and the conversation's latest "
            f"turn is {latest}",
            fix="read the conversation's state and show the turns it holds",
        ),
        status_code=409,
    )


def _streaming_conflict_response(detail: str, *, retry_after_seconds: int = 5) -> JSONResponse:
    """The 409 `TRANSIENT_UNAVAILABLE` every claim conflict answers with,
    whichever guard caught it: the process-local lock (`_acquire_or_conflict`),
    a live segment lease held by any instance (`SegmentBusy`), or the store's
    compare-and-swap on the `/turn` claim (`ConversationStale`).
    `retry_after_seconds` defaults to a flat 5s for the process-local and
    claim guards, which carry no TTL of their own to report. A `SegmentBusy`
    conflict passes its own remaining TTL."""
    return _tool_error_response(
        ToolError(
            error=ErrorCode.TRANSIENT_UNAVAILABLE,
            retry=RetryMode.RETRY_AFTER_DELAY,
            detail=detail,
            retry_after_seconds=retry_after_seconds,
        ),
        status_code=409,
    )


class _BodyTooLarge(HTTPException):
    """A body past its route's cap (`_BoundedRoute.max_body_bytes`). An
    `HTTPException`, since FastAPI re-raises those from its body read and turns
    any other exception there into a 400."""

    def __init__(self, size: str, cap: int) -> None:
        super().__init__(
            status_code=413,
            detail=f"the request body is {size} bytes, past the {cap:,}-byte cap",
        )


class _BoundedRequest(Request):
    """A request whose body read stops at `cap` bytes.

    A declared `Content-Length` past the cap is refused before anything is
    read; a body sent without one is refused as soon as it crosses the cap.
    """

    def __init__(self, scope: Scope, receive: Receive, *, cap: int) -> None:
        super().__init__(scope, receive)
        self._cap = cap

    async def body(self) -> bytes:
        if not hasattr(self, "_body"):
            declared = self.headers.get("content-length", "")
            if declared.isdigit() and int(declared) > self._cap:
                raise _BodyTooLarge(f"{int(declared):,}", self._cap)
            chunks: list[bytes] = []
            size = 0
            async for chunk in self.stream():
                size += len(chunk)
                if size > self._cap:
                    raise _BodyTooLarge(f"more than {self._cap:,}", self._cap)
                chunks.append(chunk)
            # Starlette's own cache for the body, which `Request.body` reads back.
            self._body = b"".join(chunks)
        return self._body


class _BoundedRoute(APIRoute):
    """A route whose body is read through `_BoundedRequest`, so a request past
    `max_body_bytes` is refused 413 `resource_exhausted` before FastAPI parses it."""

    max_body_bytes: ClassVar[int]

    def get_route_handler(self) -> Callable[[Request], Coroutine[Any, Any, Response]]:
        handler = super().get_route_handler()
        cap = self.max_body_bytes

        async def bounded(request: Request) -> Response:
            try:
                return await handler(_BoundedRequest(request.scope, request.receive, cap=cap))
            except _BodyTooLarge as exc:
                return _tool_error_response(
                    ToolError(
                        error=ErrorCode.RESOURCE_EXHAUSTED,
                        retry=RetryMode.DO_NOT_RETRY,
                        detail=str(exc.detail),
                    ),
                    status_code=exc.status_code,
                )

        return bounded


class _BoundedTurnRoute(_BoundedRoute):
    """The `/turn` route, bounded at `MAX_TURN_REQUEST_BYTES`. Each seed is
    checked on its own (`harness.seeds.seeds_slot`) and refused in its own
    step; this bound is what keeps an oversized request from being parsed at
    all."""

    max_body_bytes = MAX_TURN_REQUEST_BYTES


class _BoundedInterruptRoute(_BoundedRoute):
    """The `/interrupt` route, bounded at `MAX_INTERRUPT_REQUEST_BYTES`, so the
    answers, the amendment text and the lists a client posts are bounded before
    they are parsed."""

    max_body_bytes = MAX_INTERRUPT_REQUEST_BYTES


def _resume_refusal(standing: TurnStanding, turn_index: int) -> JSONResponse | None:
    """The 409 `edit_conflict` a `/resume` of turn `turn_index`, the
    conversation's latest turn, answers for `standing`, or `None` when the
    resume claims the turn: a closed turn is no open turn to carry on, and a
    turn paused on a question awaits its answer on `/interrupt`. A `CARRY`
    the route carries on before its claim; after the claim it means another
    client's answer landed in between, and the client reads the state again.
    A `HELD` turn goes on to the claim, which answers the 409
    `transient_unavailable` every claim conflict answers."""
    match standing.standing:
        case Standing.CLOSED:
            return _not_open_turn_response(turn_index)
        case Standing.AWAITING_ANSWER | Standing.CARRY:
            return _resume_conflict(
                f"turn {turn_index} is paused on a question, which an answer carries on"
            )
        case Standing.HELD | Standing.RESUMABLE | Standing.REPLAY:
            return None


def _taken_approval(
    conversation: ConversationState, cursor: Cursor, card: PlanProposed | None
) -> PlanApproved | None:
    """The `plan.approved` frame of `card`, the plan card an answer
    approved, when the turn at `cursor` already took that approval: an
    earlier segment's `approve` node approved the plan of the card's
    revision, so no `approve` node of a segment carrying the turn on states
    it. The frame the turn recorded for that revision
    (`ConversationState.turn_approved`) when it holds one, else one built
    from the card. `None` when the approval is not yet taken."""
    if card is None:
        return None
    raw = cursor.state.get("plan")
    plan = stored_plan(raw if isinstance(raw, str) else None, state_key="plan")
    if plan is None or plan.approved_at is None or plan_revision(plan) != card.revision:
        return None
    recorded = next(
        (kept for kept in conversation.turn_approved if kept.get("revision") == card.revision),
        None,
    )
    if recorded is not None:
        return PlanApproved.model_validate(recorded)
    return plan_approved(card, seq=0, baseline=conversation.turn_choice_baseline)


def _turn_approvals(
    conversation: ConversationState, taken: PlanApproved | None
) -> list[PlanApproved]:
    """Every `plan.approved` the turn recorded, then `taken` when the turn
    holds none of its revision: what a `/resume` stream sends first, for a
    page that never read them."""
    approvals = [PlanApproved.model_validate(kept) for kept in conversation.turn_approved]
    if taken is not None and all(kept.revision != taken.revision for kept in approvals):
        approvals.append(taken)
    return approvals


def _cursor_standing(
    cursor: Cursor | None, state: ConversationState, *, lease_live: bool
) -> TurnStanding:
    """Where `state`'s latest turn stands (`turn_standing`) by `cursor`, the
    conversation's open turn (`TurnStore.open_cursor`), `None` once every
    turn closed. An open cursor of another turn says nothing of the latest
    one's, and a closed cursor of the latest turn stands as `None` does:
    `StudioStore.load`, which ran first, recorded a turn whose cursor closed
    under an open record when no lease was live, and a live lease holds the
    turn whatever its cursor says."""
    current = cursor if cursor is not None and cursor.turn_index == state.turn_index else None
    return turn_standing(
        cursor_status=current.status if current is not None else None,
        pause_id=(
            current.pause.interrupt_id
            if current is not None and current.pause is not None
            else None
        ),
        lease_live=lease_live,
        state=state,
    )


def _carried_answer(response: Response, turn_index: int) -> Response:
    """What `/resume` answers for the `/interrupt` retry it delegated a
    carried answer to (`Standing.CARRY`): the stream, or a busy 409
    `transient_unavailable`, as the retry answered, and the 409
    `edit_conflict` `/resume` answers for any other refusal, a 404 for an
    answer no longer pending, a 422, or a stale plan revision, since the
    turn moved between the resume's read and the retry."""
    if response.status_code == 409:
        refused = json.loads(bytes(response.body))
        if refused.get("error") != ErrorCode.EDIT_CONFLICT.value:
            return response
    elif response.status_code not in (404, 422):
        return response
    return _resume_conflict(f"turn {turn_index} moved before its answer was carried on")


def _not_open_turn_response(turn_index: int) -> JSONResponse:
    """The 409 `edit_conflict` for a `/resume` of a turn that is not open."""
    return _resume_conflict(f"turn {turn_index} is not the conversation's open turn")


def _resume_conflict(detail: str) -> JSONResponse:
    """A `/resume` refused for the turn it names: 409 `edit_conflict`."""
    return _tool_error_response(
        ToolError(
            error=ErrorCode.EDIT_CONFLICT,
            retry=RetryMode.REFRESH_THEN_RETRY,
            detail=detail,
            pointer="/turn_index",
            fix="read the conversation's state and follow what it shows",
        ),
        status_code=409,
    )


def _stored_card(
    component_id: str, entry: Mapping[str, Any], state: ConversationState, *, swapped: bool
) -> StoredCard | None:
    """The card `entry`, a stored component of `state`
    (`ConversationState.components`), for the state read. A card on the
    canvas carries the unit its stored `UnitCard` can change within
    (`stored_unit_change`), as `apply_unit`, which refuses a card off the
    canvas, reads it. A card `swapped` in by the turn being read names the
    card whose slot it took (`REPLACED_CARD_KEY`). A `fact_summary` or a
    unit offer that cannot be read is left out and logged, and an entry
    with no type or title gives no card."""
    if not entry.get("type") or not entry.get("title"):
        return None
    try:
        summary = (
            FactSummary.model_validate(entry["fact_summary"])
            if entry.get("fact_summary") is not None
            else None
        )
    except ValidationError:
        logger.warning(
            "studio state: component %s's stored fact summary does not validate",
            component_id,
            exc_info=True,
        )
        summary = None
    unit_change = None
    if _on_canvas(state, component_id):
        try:
            unit_change = stored_unit_change(entry, state.turn_specs, state.turns)
        except Exception:
            logger.warning(
                "studio state: component %s's stored unit offer cannot be read",
                component_id,
                exc_info=True,
            )
    replaces = entry.get(REPLACED_CARD_KEY) if swapped else None
    return StoredCard(
        component_id=component_id,
        type=str(entry["type"]),
        title=str(entry["title"]),
        revision=int(entry.get("revision") or 1),
        fact_summary=summary,
        unit_change=unit_change,
        replaces=replaces if isinstance(replaces, str) else None,
    )


def _on_canvas(state: ConversationState, component_id: str) -> bool:
    """Whether `state`'s canvas holds card `component_id`."""
    blocks = state.composition.blocks if state.composition is not None else []
    return any(block.component_id == component_id for block in blocks)


def _open_user_message(state: ConversationState) -> str:
    """The latest turn's message as `_closed_turn` words it: the display line
    its record keeps once it closed with one, else the turn's own line."""
    if state.turn_recorded and state.turns:
        display = state.turns[-1].get("user_display")
        if isinstance(display, str) and display:
            return display
    return state.turn_user_message


def _closed_turn(turn_index: int, state: ConversationState) -> ClosedTurn:
    """Turn `turn_index` of `state` as its stored record holds it, with the
    stored card of each card it made (`_stored_card`). The reply is the
    answer the user was shown with the notes and the basis line the stream
    appended to it (`harness.studio_state._recorded_reply`), as its
    `message.complete` sent it, and empty for a record whose stream sent none
    (`reply_shown: false`).
    The user's message is the record's display line where it keeps one
    (`user_display`), else its `user` line."""
    record = state.turns[turn_index - 1]
    reply = "" if record.get("reply_shown") is False else _recorded_reply(record)
    listed = [
        card
        for card in record.get("cards") or []
        if isinstance(card, Mapping) and isinstance(card.get("component_id"), str)
    ]
    component_ids = [card["component_id"] for card in listed]
    # The cards this turn swapped into another card's slot
    # (`harness.studio_edit._swapped`).
    swapped = {card["component_id"] for card in listed if card.get("action") == "recreated"}
    cards = [
        card
        for component_id in component_ids
        if (entry := state.components.get(component_id)) is not None
        and (card := _stored_card(component_id, entry, state, swapped=component_id in swapped))
        is not None
    ]
    return ClosedTurn(
        turn_index=turn_index,
        user_message=record.get("user_display") or record.get("user") or "",
        reply=reply,
        component_ids=component_ids,
        cards=cards,
        approved=_approved_plans(record.get("approved")),
        error=record.get("error"),
        new_question=record.get("new_question"),
    )


def _approved_plans(stored: Any) -> list[PlanApproved] | None:
    """The `plan.approved` frames a turn recorded (`ConversationState.
    turn_approved`), for the state read, `None` when it recorded none. One
    that does not validate is left out."""
    approved: list[PlanApproved] = []
    for frame in stored if isinstance(stored, list) else ():
        try:
            approved.append(PlanApproved.model_validate(frame))
        except ValidationError:
            logger.warning("studio state: a stored approved plan does not validate", exc_info=True)
    return approved or None


def _answer_again(
    conversation_id: str, interrupt_id: str, pending: _PendingInterrupt
) -> InterruptResponse:
    """The `interrupt.response` that recorded `pending`'s answer, rebuilt
    from what `/interrupt` stored (`_PendingInterrupt.answer`), so a retry
    of it carries the turn on as the client's own retry would. A switch
    names the revision of the plan it switched on, the entry's own."""
    answer = pending.answer or {}
    switch = answer.get("switch") or {}
    # A stored answer is carried whole, whatever its length (`STORED_INTERRUPT`).
    return InterruptResponse.model_validate(
        {
            "kind": "interrupt.response",
            "schema_version": 1,
            "conversation_id": conversation_id,
            "interrupt_id": interrupt_id,
            "decision": answer.get("decision"),
            **({"payload": answer["payload"]} if answer.get("payload") is not None else {}),
            **({"answers": answer["answers"]} if "answers" in answer else {}),
            **({**switch, "plan_revision": pending.plan_revision} if switch else {}),
        },
        context={STORED_INTERRUPT: True},
    )


def _stored_standing(stored: StoredConversation) -> TurnStanding:
    """Where `stored`'s latest turn stands (`turn_standing`)."""
    return turn_standing(
        cursor_status=stored.turn_status,
        pause_id=stored.pause_id,
        lease_live=stored.lease_seconds is not None,
        state=stored.state,
    )


def _read_status(standing: TurnStanding) -> TurnStatusName:
    """The status the state read gives a turn of `standing`: a closed turn's
    own, `paused` for one whose pause awaits an answer, and `running` for
    the rest, which a page waits out (`HELD`) or carries on with `/resume`
    (`RESUMABLE`, `CARRY`, `REPLAY`)."""
    match standing.standing:
        case Standing.CLOSED:
            assert standing.closed is not None
            return TurnStatusName(standing.closed.value)
        case Standing.AWAITING_ANSWER:
            return TurnStatusName.PAUSED
        case Standing.HELD | Standing.RESUMABLE | Standing.CARRY | Standing.REPLAY:
            return TurnStatusName.RUNNING


def _conversation_read(stored: StoredConversation, *, since: int | None) -> ConversationRead:
    """`GET /api/studio/state`'s answer for `stored`: the turn, the closed
    turns after `since` (none when it is `None`), the open pause, and the
    canvas. A conversation's records are kept one per turn in turn order
    (`ConversationState.close_turn`), so record `n - 1` is turn `n`'s."""
    state = stored.state
    standing = _stored_standing(stored) if state.turn_index else None
    status = _read_status(standing) if standing is not None else None
    turn = (
        TurnState(
            index=state.turn_index,
            status=status,
            trace_id=state.trace_id,
            user_message=_open_user_message(state),
            approved=(
                _approved_plans(state.turn_approved)
                if status in (TurnStatusName.RUNNING, TurnStatusName.PAUSED)
                else None
            ),
            lease_live=stored.lease_seconds is not None,
            retry_after_seconds=(
                math.ceil(stored.lease_seconds) if stored.lease_seconds is not None else None
            ),
        )
        if status is not None
        else None
    )
    indexes = range(since + 1, len(state.turns) + 1) if since is not None else range(0)
    closed = [_closed_turn(index, state) for index in indexes[:MAX_CLOSED_SINCE]]
    pause = None
    if standing is not None and standing.standing is Standing.AWAITING_ANSWER:
        found = pause_entry(state.pending, stored.pause_id)
        assert found is not None  # a turn awaits an answer on the entry that holds its pause
        _, entry = found
        pause = PendingPause.model_validate(
            {"interrupt_id": entry.adk_id, "kind": entry.kind, "frames": entry.frames}
        )
    return ConversationRead(
        conversation_id=state.conversation_id,
        turn=turn,
        closed_since=closed,
        closed_total=len(state.turns),
        pause=pause,
        manifest=state.composition,
    )


def _no_pending_interrupt_response(interrupt_id: str) -> JSONResponse:
    """The 404 for an answer to an interrupt the conversation does not hold
    pending."""
    return _tool_error_response(
        ToolError(
            error=ErrorCode.NOT_FOUND,
            retry=RetryMode.DO_NOT_RETRY,
            detail=f"no pending interrupt {interrupt_id!r} on this conversation",
        ),
        status_code=404,
    )


def _unknown_conversation_response(conversation_id: str) -> JSONResponse:
    """The 404 every route answers with for a conversation id its store
    does not hold. The wire has no "new conversation" signal, so this is
    what tells a lost or mistyped id apart from a fresh one (`/turn` is the
    only route that creates)."""
    return _tool_error_response(
        ToolError(
            error=ErrorCode.NOT_FOUND,
            retry=RetryMode.DO_NOT_RETRY,
            detail=f"no conversation {conversation_id!r}",
            pointer="/conversation_id",
        ),
        status_code=404,
    )


async def _acquire_or_conflict(
    store: StudioStore, principal: str, conversation_id: str
) -> LockHold | JSONResponse:
    """Take the conversation's lock BEFORE loading or touching any turn
    state.

    The check-and-acquire pair has no suspension point between them on
    the uncontended fast path, so two concurrent requests cannot both
    pass. A lazily-acquired lock let them race past the 409 guard and
    clobber each other's turn state and cancel event. The request lets go
    of the hold on every early-exit path (`_release_claim`), or hands it to
    the stream's owner (`harness.studio_stream.open_stream`), which lets go
    once the segment ends. Returns the request's hold on the lock, not a
    conversation, since a lock exists (and can be contended) before this
    store has ever loaded or created the conversation it names.
    """
    lock = store.lock_for(principal, conversation_id)
    if lock.locked():
        return _streaming_conflict_response("another request is streaming on this conversation")
    await lock.acquire()
    return LockHold(lock)


class _ClaimLost(Exception):  # noqa: N818 -- names the outcome, not an error kind
    """A Stop's own close of the turn gave way to another writer: its claim
    lost to another request, the lease it claimed was taken over, or the
    turn moved since it was read. The turn did not close. When the claim
    committed, its lease is dropped, and what committed under it (the Stop
    it recorded, and perhaps the turn's record saved closed) leaves the turn
    open with a Stop live, which whoever claims next acts on. The caller
    decides again from the stored state."""


async def _act_on_stop(
    studio: StudioRunner, principal: str, conversation_id: str
) -> Response | None:
    """Act on a Stop live for the conversation's turn
    (`StudioStore.stop_live`) before a request claims the conversation: the
    open turn, running or paused, is closed `cancelled` under a claim of its
    own (`_stop_turn_at`). Returns `None` once the turn is closed or none is
    open, and otherwise the error to answer: the 503 `_stop_turn_at` answers
    when the close fails, or a retryable 409 when its claim lost to another
    request or the turn moved first. With no open turn the Stop names a turn
    claimed and not yet opened, whose live lease refuses this request's
    claim."""
    cursor = await studio.turns.open_cursor(principal, conversation_id)
    if cursor is None:
        return None
    try:
        return await _stop_turn_at(studio, principal, conversation_id, cursor)
    except _ClaimLost:
        return _streaming_conflict_response(
            "the conversation's previous turn is still being stopped", retry_after_seconds=1
        )


async def _close_stopped_turn(
    studio: StudioRunner, principal: str, conversation_id: str
) -> Response | None:
    """The Stop handling `/turn` and `/interrupt` share when a Stop is live
    for the conversation's open turn, found before the claim
    (`StudioStore.stop_live`) or by the claim save (`StopPending`): give up
    the request's working copy, reload the conversation and close the turn
    under a claim of its own (`_act_on_stop`). Returns that act's refusal,
    `None` once the turn is closed. The caller holds the conversation lock
    throughout and decides when to release it."""
    studio.store.drop_draft(principal, conversation_id)
    await studio.store.load(principal, conversation_id)
    return await _act_on_stop(studio, principal, conversation_id)


async def _answer_after_stop(
    studio: StudioRunner,
    principal: str,
    conversation_id: str,
    hold: LockHold,
    then: Response,
    *,
    turn: ConversationState | None = None,
) -> Response:
    """Close the turn a live Stop names (`_close_stopped_turn`), let go of
    `hold` (`_unclaimed`, given `turn`), and answer that close's refusal,
    else `then`."""
    refused = await _close_stopped_turn(studio, principal, conversation_id)
    return await _unclaimed(
        studio.store, principal, conversation_id, hold, refused or then, turn=turn
    )


async def _release_claim(
    store: StudioStore,
    principal: str,
    conversation_id: str,
    hold: LockHold,
    segment_id: str | None,
    *,
    cause: BaseException | None = None,
    turn: ConversationState | None = None,
    unsure: str | None = None,
) -> None:
    """Let go of a request whose stream never started, on every path,
    cancellation included. A claim that never committed gives up its
    working copy (`StudioStore.drop_draft`). A claim that committed
    (`segment_id`) drops its lease (`StudioStore.drop_lease`, which runs to
    the end under cancellation). A claim save that raised with the live
    state not holding its segment (`unsure`) may still have committed, a
    COMMIT that landed and then timed out say, so that lease is dropped all
    the same: `drop_lease` clears only that segment's lease, and a failure
    there is logged and leaves the lease to its TTL. A claim save that
    closes the open turn `abandoned` confirms a COMMIT that raised by its
    transaction's status, and returns with the turn's line once it finds it
    committed (`StudioStore.save`), so that line is lost only when the
    status stayed unknown for its limit. A request opens no span before its stream,
    whose owner opens one: `turn`, the working copy its claim began or
    resumed a turn on, gets a zero-length `turn.segment` span on the turn's
    trace (`record_unstreamed_span`) whose status says what let it go. `cause`,
    the exception that ended the request, gives `cancelled` when it is this
    task's own cancellation (the client gone) and `failed` otherwise, a
    `CancelledError` an inner task let out among them, and a refusal, with
    no `cause`, gives `not_started`. Then `hold` lets go of the lock,
    whatever recording the span raised, and that span is exported, and with
    it any turn line the claim save wrote, an abandoned turn's close, which no
    stream then exports. It does nothing once `hold` has let go: the working copy and the lease are then
    another request's, so a request cancelled while it exports the span
    leaves the next request's claim alone."""
    if not hold.held:
        return
    task = asyncio.current_task()
    status: UnstreamedStatus
    if cause is None:
        status = "not_started"
    elif isinstance(cause, asyncio.CancelledError) and task is not None and task.cancelling() > 0:
        status = "cancelled"
    else:
        status = "failed"
    try:
        store.drop_draft(principal, conversation_id)
        if segment_id is not None:
            await store.drop_lease(principal, conversation_id, segment_id)
        elif unsure is not None:
            try:
                await store.drop_lease(principal, conversation_id, unsure)
            except Exception:
                logger.warning(
                    "studio: conversation %s could not drop the lease its claim save may "
                    "have committed; it lapses by its TTL",
                    conversation_id,
                    exc_info=True,
                )
    finally:
        try:
            if turn is not None:
                record_unstreamed_span(
                    conversation_id, trace_id=turn.trace_id, status=status, usage=turn.turn_usage
                )
        finally:
            hold.release()
        if turn is not None:
            await flush_telemetry()


async def _stop_open_turn(
    studio: StudioRunner, principal: str, conversation_id: str
) -> Response | None:
    """Close what a Stop that found no live segment leaves, closing the turn
    the conversation holds open and clearing its pending interrupts, and
    save it. Returns the error to answer when the turn could not be closed,
    and `None` otherwise.

    An open turn, paused on a question or left running by a segment whose
    lease lapsed or whose client left, is closed `cancelled` under a lease
    this claims and drops at the end (`_stop_turn_at`), which ends the runs
    it left computing, so no later turn carries them on. When that close
    fails, a retryable 503 is answered and the turn stays open with a Stop
    live, which the next claimant or a retried Stop acts on. With no open
    turn, the lease on record is cleared. It has lapsed, since
    `request_cancel` found no live segment and the save refuses if one went
    live since, and clearing it fences out a stream running under it, whose
    writes are fenced on the lease id alone.

    Raises:
        _ClaimLost: another request claimed the conversation, or moved its
            turn, since this request found no live lease.
    """
    cursor = await studio.turns.open_cursor(principal, conversation_id)
    if cursor is None:
        try:
            wrote = await _save_turn_closed(
                studio, principal, conversation_id, None, clear_lease=True
            )
        except ConversationStale as exc:
            raise _ClaimLost from exc
        if wrote:
            # No stream is left to export the close's outcome.
            await flush_telemetry()
        return None
    return await _stop_turn_at(studio, principal, conversation_id, cursor)


async def _stop_turn_at(
    studio: StudioRunner, principal: str, conversation_id: str, cursor: Cursor
) -> Response | None:
    """Close the open turn at `cursor` `cancelled` under a lease this claims
    with a save that acts on any Stop live for it (`StudioStore.save` with
    `stopping`), and drop the lease. Under the lease this records a Stop of
    its own (`StudioStore.request_cancel`), then, in one transaction, saves
    the turn's record closed with its pending interrupts cleared
    (`_save_turn_closed`) and closes the turn's cursor, which ends the runs
    it left computing and ends the Stop (`harness.turn.loop.stop_turn_on`),
    so the record commits only with the cursor closed. Returns `None` once
    the turn is closed, and a retryable 503 when a write fails. The turn
    then stays open, record and cursor, with a Stop live, so the next
    claimant closes it (`_act_on_stop`) and no later turn carries its runs
    on. The save of the
    turn's record closed writes its `cancelled` line and `turn` outcome,
    which this exports, since no stream runs to.

    Raises:
        _ClaimLost: the claim lost to another request, the lease was taken
            over before the writes landed, or the turn moved since `cursor`
            was read.
    """
    segment_id: str | None = None
    try:
        segment_id = studio.store.claim_segment(principal, conversation_id)
        await studio.store.save(principal, conversation_id, stopping=True)
    except (SegmentBusy, ConversationStale) as exc:
        # Each is raised before the claim save's COMMIT.
        raise _ClaimLost from exc
    except BaseException:
        # Any other failure of the claim save may have committed its lease,
        # a cancel during the COMMIT say, so the lease is dropped
        # (`StudioStore.drop_lease` clears only this segment's).
        if segment_id is not None:
            await studio.store.drop_lease(principal, conversation_id, segment_id)
        raise
    assert segment_id is not None
    wrote = False
    try:
        try:
            if not await studio.store.request_cancel(principal, conversation_id):
                raise _ClaimLost

            async def close_cursor(conn: AsyncConnection, _pk: int) -> None:
                assert segment_id is not None
                await stop_turn_on(
                    conn,
                    cursor,
                    segment_id=segment_id,
                    now=datetime.fromtimestamp(studio.store.now(), UTC),
                )

            wrote = await _save_turn_closed(
                studio, principal, conversation_id, cursor, alongside=close_cursor
            )
        except _ClaimLost:
            raise
        except (ConversationStale, CursorStale, SegmentLost) as exc:
            raise _ClaimLost from exc
        except Exception:
            logger.exception(
                "studio stop: conversation %s's turn could not be closed; its Stop stays live",
                conversation_id,
            )
            return _tool_error_response(
                ToolError(
                    error=ErrorCode.TRANSIENT_UNAVAILABLE,
                    retry=RetryMode.RETRY_AFTER_DELAY,
                    detail="the turn could not be stopped",
                    retry_after_seconds=5,
                ),
                status_code=503,
            )
    finally:
        await studio.store.drop_lease(principal, conversation_id, segment_id)
        if wrote:
            # No stream is left to export the close's outcome.
            await flush_telemetry()
    return None


async def _save_turn_closed(
    studio: StudioRunner,
    principal: str,
    conversation_id: str,
    open_turn: Cursor | None,
    *,
    clear_lease: bool = False,
    alongside: Callable[[AsyncConnection, int], Awaitable[None]] | None = None,
) -> bool:
    """Save the conversation's turn closed `cancelled` when it is open,
    keeping the records `open_turn`, its cursor, holds (`keep_turn_records`),
    with its pending interrupts cleared, so an answer to one is refused.
    With `clear_lease`, the lease on record is cleared too. `alongside` runs
    in the save's transaction (`StudioStore.save`), so what it writes, the
    close of the turn's cursor, commits with the record or not at all. The change is made
    on a working copy (`StudioStore.draft`) and saved (`StudioStore.save`):
    only once the save commits does the live state take it. The save
    writes the close's `cancelled` line and `turn` outcome once it commits
    (`StudioStore.save`). Returns whether it wrote one, for a caller with no
    stream to export it. Raises what the save raises, with the live state as
    it was."""
    work = studio.store.draft(principal, conversation_id)
    work.pending.clear()
    if work.turn_index and not work.turn_recorded:
        if open_turn is not None:
            keep_turn_records(work, open_turn.state)
        work.close_turn(final_text=work.turn_final_text, error=ErrorCode.CANCELLED.value)
    if clear_lease:
        work.segment = None
    try:
        return await studio.store.save(principal, conversation_id, work, alongside=alongside)
    except BaseException:
        studio.store.drop_draft(principal, conversation_id)
        raise


def _set_turn_context(
    principal: str, conversation_id: str, conversation: ConversationState, settings: Settings
) -> None:
    """Make the turn's context current for the request: set in the request
    task's context (each request gets a fresh one, so nothing leaks across
    requests), the streamed generator and every task spawned under it
    inherit the values. The conversation routes the executor's
    presentation client, and with it the platform canvas session, scoped
    by PRINCIPAL as well as conversation, since two principals minting the
    same conversation_id must not share a canvas. The user's words are what
    a search sends. The budget bounds every model call the segment makes
    (`harness.turn_budget`)."""
    current_conversation_id.set(ledger_conversation_id(principal, conversation_id))
    current_conversation_key.set((principal, conversation_id))
    current_user_message.set(conversation.turn_user_words)
    current_earlier_block.set(conversation.earlier_block)
    current_recorded_identity.set(conversation.recorded_identity)
    start_turn_budget(settings)


def _same_open_pause(before: Cursor | None, after: Cursor | None) -> bool:
    """Whether `after`, the turn read after a claim, is where `before`, the
    turn read before it, stood: both closed, or both open with the same
    status and pause. A running turn may have moved on to a later node."""
    if before is None or after is None:
        return before is after
    return after.status is before.status and after.pause == before.pause


def _approval_admission(
    studio: StudioRunner,
    state: Mapping[str, Any],
    pending: _PendingInterrupt,
    *,
    at: datetime,
    principal: str,
    conversation_id: str,
) -> Callable[[AsyncConnection, int], Awaitable[None]] | None:
    """The write that admits the run of the plan the turn's `state` holds,
    approved at `at` through plan interrupt `pending`, for `StudioStore.save`
    to make in the transaction that records the approval (`alongside`).

    The plan is the one `approve.decide` reads from the same cursor, approved
    when it awaits approval as the revision `pending` is keyed on, as
    `decide` approves it; its run is the candidate `execute_approved_plan`
    builds (`approved_work`, `run_candidate`), which `studio.run_ledger`
    stores with its plan and admits by `execute`'s rule
    (`StudioRunLedger.admit_with_approval`). So `execute` finds the run its
    key names and carries it on, and a turn that ends before `execute`
    leaves the run for the next turn's `seed`. When the admission leaves
    that run unfinished, the write names it on `pending` (`admitted_run`),
    which the same save stores, so the answer to `approve` names it on this
    attempt and on every retry resuming it.

    None when there is nothing to admit: no Studio ledger, no plan the state
    holds that this release reads, a plan keyed on another revision
    (`decide` amends), a plan that no longer awaits approval (`decide`
    raises in the turn), or one `approved_work` finds nothing to run for
    (`execute` raises in the turn). It raises nothing."""
    ledger = studio.run_ledger
    if ledger is None:
        return None
    raw = state.get("plan")
    plan = stored_plan(raw if isinstance(raw, str) else None, state_key="plan")
    if (
        plan is None
        or plan.status is not PlanStatus.PENDING_APPROVAL
        or plan_revision(plan) != pending.plan_revision
    ):
        return None
    approved = plan.approve(at=at)
    work = approved_work(
        approved,
        approver=principal,
        conversation_id=ledger_conversation_id(principal, conversation_id),
    )
    if isinstance(work, UnrunnablePlan):
        return None

    def build(now: datetime) -> ApprovedRun:
        return ApprovedRun(
            candidate=run_candidate(
                work.submission,
                authorisation=work.authorisation,
                idempotency_key=work.idempotency_key,
                now=now,
            ),
            submission=submission_content(work.submission),
            revision=plan_revision(approved),
            plan=approved.model_dump_json(),
        )

    async def admit(conn: AsyncConnection, pk: int) -> None:
        admitted = await ledger.admit_with_approval(conn, pk, build)
        if admitted is not None and admitted.run.status in UNFINISHED_STATUSES:
            pending.admitted_run = admitted.run.run_id

    return admit


async def _replays_later_pause(
    turns: TurnStore,
    cursor: Cursor | None,
    interrupt_id: str,
    requested_answer: dict[str, Any],
) -> bool:
    """Whether an `/interrupt` answering `interrupt_id` with
    `requested_answer` is a retry of the answer that resumed the turn, which
    then paused again on a later question, `cursor`'s open pause: the turn
    accepted an answer to `interrupt_id` at an earlier round than that
    pause, and the client sent it as `requested_answer`
    (`TurnStore.answered`). The stored answer decides it whether or not the
    conversation still holds the answered entry, which the closing write
    that stored the later pause drops."""
    if (
        cursor is None
        or cursor.status is not TurnStatus.PAUSED
        or cursor.pause is None
        or cursor.pause.interrupt_id == interrupt_id
    ):
        return False
    accepted = await turns.answered(cursor, interrupt_id=interrupt_id)
    return (
        accepted is not None
        and accepted.round < cursor.pause.round
        and accepted.request == requested_answer
    )


async def _replay_pause(
    studio: StudioRunner,
    *,
    principal: str,
    conversation_id: str,
    interrupt_id: str,
    work: ConversationState,
    cursor: Cursor,
    segment_id: str,
    hold: LockHold,
    settings: Settings,
    turn: ConversationState | None,
    stale: JSONResponse | None = None,
) -> Response:
    """Stream the later pause the paused `cursor` holds to a retry of the
    answer to `interrupt_id` (`_replays_later_pause`), under the segment
    `segment_id` the retry claimed, with no node run. The answered entry,
    when the conversation still holds it, is marked resuming under
    `segment_id` and saved first, so the stream's closing write drops it
    and stores the replayed pause's entry, which the next answer answers. A
    closing write that fails, or a client that leaves, keeps the entry
    resuming, for the same retry to replay the pause again. A save that
    finds the conversation moved answers `stale`, the caller's 409, or else
    an `/interrupt` answer's."""
    if (answered := work.pending.get(interrupt_id)) is not None:
        answered.resuming_segment = segment_id
    try:
        await studio.store.save(principal, conversation_id, work)
    except ConversationStale:
        await _release_claim(studio.store, principal, conversation_id, hold, segment_id, turn=turn)
        if stale is not None:
            return stale
        return _tool_error_response(
            ToolError(
                error=ErrorCode.EDIT_CONFLICT,
                retry=RetryMode.REFRESH_THEN_RETRY,
                detail="the conversation changed before this answer was saved",
                fix="reload the conversation and answer what it now shows",
            ),
            status_code=409,
        )
    conversation = studio.store.current(principal, conversation_id)
    assert conversation is not None
    _set_turn_context(principal, conversation_id, conversation, settings)
    # The segment's owner takes the lease and the lock hold from here on.
    return open_stream(
        _SegmentWork(
            studio,
            principal=principal,
            conversation=conversation,
            cursor=cursor,
            segment_id=segment_id,
            replay=True,
        ),
        owners=studio.owners,
        store=studio.store,
        conversation=conversation,
        principal=principal,
        conversation_id=conversation_id,
        segment_id=segment_id,
        hold=hold,
    )


async def _unclaimed(
    store: StudioStore,
    principal: str,
    conversation_id: str,
    hold: LockHold,
    response: Response,
    *,
    turn: ConversationState | None = None,
) -> Response:
    """Answer `response` for a request whose claim save never committed,
    once it has let go of it (`_release_claim`, given `turn`), so the live
    state stays as loaded."""
    await _release_claim(store, principal, conversation_id, hold, None, turn=turn)
    return response


# How long, by wall time, `/cancel` waits on a lock held with no live lease
# (a request before its claim save, or a stream after its closing write)
# before it answers a retryable 503.
CANCEL_LOCK_WAIT_S = 5.0


async def _lock_or_claim(lock: asyncio.Lock, claimed: asyncio.Event, *, timeout: float) -> bool:
    """Wait up to `timeout` for `lock` or for `claimed`, whichever comes
    first. Once `timeout` is spent, a free lock with no waiter is taken at
    once. Returns `True` when this holds `lock`, and `False` when a claim
    committed first or the time ran out. Cancelled after it took the lock,
    it releases the lock."""
    if claimed.is_set():
        return False
    if timeout <= 0:
        # An uncontended acquire returns without suspending, so the spent
        # timeout cancels only an acquire that queues, behind a waiter the
        # lock was just handed to.
        try:
            async with asyncio.timeout(0):
                return await lock.acquire()
        except TimeoutError:
            return False
    acquiring = asyncio.ensure_future(lock.acquire())
    claiming = asyncio.ensure_future(claimed.wait())
    waited = False
    try:
        await asyncio.wait(
            {acquiring, claiming}, timeout=timeout, return_when=asyncio.FIRST_COMPLETED
        )
        waited = True
    finally:
        claiming.cancel()
        if not acquiring.done():
            acquiring.cancel()
        with suppress(asyncio.CancelledError):
            await acquiring
        if not waited and not acquiring.cancelled():
            # Cancelled after the lock was taken: let it go.
            lock.release()
    if acquiring.cancelled():
        return False
    if claimed.is_set():
        # The claim came first: decide again before holding the lock.
        lock.release()
        return False
    return True


async def _stop_under_lock(
    studio: StudioRunner, principal: str, conversation_id: str, *, turn_index: int | None
) -> Response:
    """`/cancel`'s answer once it holds the conversation lock: 204 when a
    lease went live since (another instance claimed while this one waited,
    and the Stop is recorded on it), when the store holds no such
    conversation, or once what an open turn left is closed
    (`_stop_open_turn`). It answers that close's 503 when it fails. A Stop
    for turn `turn_index` of a conversation whose latest turn is another
    is answered `_cancel_skipped` and leaves the conversation as it was.

    Raises:
        _ClaimLost: this route's claim lost to another request.
    """
    try:
        if await studio.store.request_cancel(principal, conversation_id, turn_index=turn_index):
            return Response(status_code=204)
    except TurnMoved as exc:
        return _cancel_skipped(exc.latest)
    state = await studio.store.load(principal, conversation_id)
    if state is None:
        return Response(status_code=204)
    if turn_index is not None and state.turn_index != turn_index:
        return _cancel_skipped(state.turn_index)
    # A turn paused on a question, or one its client left while it ran, is
    # open with no stream to close it.
    refused = await _stop_open_turn(studio, principal, conversation_id)
    return refused or Response(status_code=204)


def _cancel_skipped(latest: int) -> JSONResponse:
    """`/cancel`'s 200 answer to a Stop for a turn that is not the
    conversation's latest, `latest`, which leaves the conversation as it
    was."""
    return JSONResponse(
        content=CancelSkipped(code="turn_moved", turn_index=latest).model_dump(mode="json")
    )


class _BoundedCancelRoute(_BoundedRoute):
    """The `/cancel` route, bounded at `MAX_CANCEL_REQUEST_BYTES`."""

    max_body_bytes = MAX_CANCEL_REQUEST_BYTES


class _BoundedResumeRoute(_BoundedRoute):
    """The `/resume` route, bounded at `MAX_RESUME_REQUEST_BYTES`."""

    max_body_bytes = MAX_RESUME_REQUEST_BYTES


class _BoundedCompositionRoute(_BoundedRoute):
    """The `/composition` route, bounded at `MAX_COMPOSITION_REQUEST_BYTES`, so the
    operations a client posts are bounded before they are parsed."""

    max_body_bytes = MAX_COMPOSITION_REQUEST_BYTES


def create_studio_router(studio: StudioRunner, *, settings: Settings) -> APIRouter:
    router = APIRouter(prefix="/api/studio")

    async def turn(  # noqa: PLR0911 -- one distinct refusal per claim guard the route enforces
        body: TurnRequest,
        authorization: str | None = Header(default=None),
        x_dfo_account: str | None = Header(default=None),
        x_dfo_caller: str | None = Header(default=None),
    ) -> Response:
        principal = principal_for(authorization, settings=settings)
        account, visitor = studio_identity(
            account=x_dfo_account, caller=x_dfo_caller, settings=settings
        )
        hold = await _acquire_or_conflict(studio.store, principal, body.conversation_id)
        if isinstance(hold, JSONResponse):
            return hold
        # The segment this request claimed once its claim save landed, which
        # every failure before the stream starts releases, and the working
        # copy its claim began the turn on, whose trace records how it let
        # go (`_release_claim`).
        claimed: str | None = None
        # The lease id this request's lease claim writes, from the claim's
        # start until it returns or refuses: a claim that raised may have
        # committed it (`_release_claim`'s `unsure`).
        leasing: str | None = None
        begun: ConversationState | None = None
        try:
            conversation = await studio.store.load(principal, body.conversation_id)
            if conversation is None:
                if (
                    body.component_ref is not None
                    or body.apply_choice is not None
                    or body.apply_unit is not None
                ):
                    # The client refers to state (a card to edit, to follow up
                    # on or to change the unit of, or a chip under an answer)
                    # this store never had.
                    hold.release()
                    return _unknown_conversation_response(body.conversation_id)
                try:
                    conversation = await studio.store.create(principal, body.conversation_id)
                except ConversationExists:
                    # Another instance's `/turn` created this same
                    # never-before-seen conversation between this request's
                    # own load and create. Load what it created and continue
                    # to the claim below. Its compare-and-swap decides the
                    # race, and a lost claim is the `ConversationStale`
                    # 409, not a bare 500.
                    conversation = await studio.store.load(principal, body.conversation_id)
                    assert conversation is not None
            elif await studio.store.stop_live(principal, body.conversation_id):
                # A Stop live for the turn left open, by a stream whose close
                # failed or whose lease lapsed, ends that turn before this
                # one begins.
                refused = await _close_stopped_turn(studio, principal, body.conversation_id)
                if refused is not None:
                    return await _unclaimed(
                        studio.store, principal, body.conversation_id, hold, refused
                    )
                conversation = await studio.store.load(principal, body.conversation_id)
                assert conversation is not None
            # Checked again on the conversation reloaded under the claim
            # below, which is what the turn begins on.
            checked = _turn_refusal_or_chip(conversation, body)
            if isinstance(checked, JSONResponse):
                hold.release()
                return checked
            moved = _unit_move(conversation, body.apply_unit) if body.apply_unit else None
            if isinstance(moved, JSONResponse):
                hold.release()
                return moved
            chip_label = checked
            pick = body.apply_choice
            user_message = body.message if pick is None else f"count {chip_label}"
            if moved is not None:
                user_message = _unit_request_line(moved)
            # The conversation's identity, set in the request task's context
            # (`_set_turn_context` sets it again with the rest of the turn's)
            # before the turn begins, so the turn records the caller its MCP
            # clients send. The graph's in-process dfo-apis calls (the
            # executor's statistics and labels, the translator's lookups) run
            # outside any MCP request, so they name the conversation and turn
            # through the supplied ones.
            current_conversation_id.set(ledger_conversation_id(principal, body.conversation_id))
            current_conversation_key.set((principal, body.conversation_id))
            set_supplied_caller(current_conversation_caller())
            usage = TurnUsage(
                caller=current_conversation_caller(),
                account=account,
                visitor=visitor,
                surface=body.surface.value,
            )
            # The claim: the lease, written alone (`StudioStore.claim_turn_lease`),
            # before the turn opens and before the SSE response starts. A
            # segment whose lease lapsed can still write the turn left open
            # (its cursor, its record) until the claim commits. From then on
            # its writes are fenced out, so the turn is read below, under
            # this request's lease, as that segment last wrote it.
            # `SegmentBusy` and `ConversationStale` mean a live segment, on
            # this instance or another, already owns the conversation's
            # stream. `StopPending` means a Stop is live for the open turn,
            # which ends that turn here, and the request is answered a
            # retryable 409. Claiming first leaves no crash window: a crash
            # after the claim leaves the turn index where it was, and the
            # segment expires by its own TTL. The turn's trace is minted
            # here, so a claim that fails ends a span on the trace the turn
            # would have had (`_release_claim`).
            begun = ConversationState(
                conversation_id=body.conversation_id,
                trace_id=_wire_id("tr"),
                turn_user_message=user_message,
                turn_usage=usage.copy(),
            )
            leasing = _wire_id("seg")
            try:
                lease = await studio.store.claim_turn_lease(
                    principal, body.conversation_id, segment_id=leasing, after_turn=body.after_turn
                )
            except SegmentBusy as exc:
                # Each refusal is raised before the claim's COMMIT.
                leasing = None
                return await _unclaimed(
                    studio.store,
                    principal,
                    body.conversation_id,
                    hold,
                    _streaming_conflict_response(
                        "another request is streaming on this conversation",
                        retry_after_seconds=exc.retry_after_seconds,
                    ),
                    turn=begun,
                )
            except TurnMoved as exc:
                leasing = None
                return await _unclaimed(
                    studio.store,
                    principal,
                    body.conversation_id,
                    hold,
                    _turn_moved_response(after_turn=body.after_turn, latest=exc.latest),
                    turn=begun,
                )
            except StopPending:
                # A `ConversationStale` of its own kind, so it is caught first.
                leasing = None
                return await _answer_after_stop(
                    studio,
                    principal,
                    body.conversation_id,
                    hold,
                    _streaming_conflict_response(
                        "the conversation's previous turn was stopped", retry_after_seconds=1
                    ),
                    turn=begun,
                )
            except ConversationStale:
                leasing = None
                return await _unclaimed(
                    studio.store,
                    principal,
                    body.conversation_id,
                    hold,
                    _streaming_conflict_response(
                        "another request is streaming on this conversation"
                    ),
                    turn=begun,
                )
            claimed = segment_id = lease.segment_id
            leasing = None
            # Under the lease, the turn begins on the conversation and the
            # turn left open as stored. A turn still open, paused on a
            # question or left running by the lapsed segment, is closed
            # `abandoned`, keeping its records. A turn whose cursor closed
            # with its record left open (a closing write that never landed)
            # is recorded as its cursor holds it, as `StudioStore.load`
            # records one with no lease live. A turn the tables record as
            # open with no cursor closes on what the conversation holds.
            # The conversation is read again under the lease: an edit on
            # another instance can change the canvas without moving the
            # version. Every change is made on a working copy, which the save
            # swaps into the live state once it commits, with the new turn's
            # frame numbers reserved.
            await studio.store.load(principal, body.conversation_id)
            work = studio.store.draft(principal, body.conversation_id)
            # The card and the chip are checked on the conversation as
            # reloaded: a turn on another instance between the first load
            # and the claim can answer anew, which makes the chip stale, or
            # drop the card a follow-up names.
            checked = _turn_refusal_or_chip(work, body)
            if isinstance(checked, JSONResponse):
                await _release_claim(
                    studio.store, principal, body.conversation_id, hold, claimed, turn=begun
                )
                return checked
            moved = _unit_move(work, body.apply_unit) if body.apply_unit else None
            if isinstance(moved, JSONResponse):
                await _release_claim(
                    studio.store, principal, body.conversation_id, hold, claimed, turn=begun
                )
                return moved
            chip_label = checked
            user_message = body.message if pick is None else f"count {chip_label}"
            if moved is not None:
                user_message = _unit_request_line(moved)
            left_open = await studio.turns.open_cursor(principal, body.conversation_id)
            if left_open is not None:
                if not work.turn_recorded:
                    keep_turn_records(work, left_open.state)
            elif work.turn_index and not work.turn_recorded:
                closed = await studio.turns.turn_at(lease.conversation, work.turn_index)
                if closed is not None:
                    record_closed_turn(work, closed)
            # The record of the turn left open, which `begin_turn` below closes
            # `abandoned`, or `None` when that turn was recorded already.
            left_record = (
                work.turn_index - 1 if left_open is not None and not work.turn_recorded else None
            )
            # An unrecorded previous turn this closes as `abandoned` rides the
            # claim save, which writes its line once it commits
            # (`StudioStore.save`).
            work.begin_turn(usage, trace_id=begun.trace_id)
            begun = work
            # The turn's MCP clients name its id (`current_caller_turn`), and
            # so do the in-process dfo-apis calls.
            current_turn_id.set(work.trace_id)
            set_supplied_turn(work.trace_id)
            current_spec_uses.set(
                _spec_use_sink(studio.store, principal, body.conversation_id, work.trace_id)
            )
            work.auto_add = body.auto_add
            if body.client_features is not None:
                work.client_features = declared_features(body.client_features)
            work.turn_user_message = user_message
            if moved is not None:
                # The change is the user's pick, not words Search should read.
                work.turn_user_words = ""
            elif pick is None:
                work.turn_user_words = body.message
            else:
                # The history record names the option the chip counts. Search
                # would read its label as scope the user stated, so the words
                # later searches send leave it out, as a plan card's switch does.
                work.turn_user_words = ""
                # The record's `spec_id` is the reading the `seed` node compiles,
                # set once the chip applies, so the chip's own origin stays out.
                work.turn_applied_choice = pick.model_dump(include={"choice_id", "option_id"})
            editing = moved is not None or (
                body.component_ref is not None
                and body.component_ref.intent is ComponentRefIntent.EDIT
            )
            opened: list[Cursor] = []

            async def begin_turn(conn: AsyncConnection, pk: int) -> None:
                # The turn left open closes `abandoned` and the new turn
                # opens (the Edit Card route opens none) in the save's
                # transaction, so a save that fails leaves the turn left
                # open as it stood, its pause still answerable. A Stop that
                # names the turn left open, one that landed on this request's
                # lease after its claim, closes it `cancelled`, cursor and
                # record, as `_act_on_stop` would have before the claim. The
                # new turn's opening is read after that close, so its model
                # reads the turn left open as it closed.
                if left_open is not None:
                    if await stop_names_turn(conn, pk, left_open.turn_index):
                        await stop_turn_on(
                            conn,
                            left_open,
                            segment_id=segment_id,
                            now=datetime.fromtimestamp(studio.store.now(), UTC),
                        )
                        if left_record is not None:
                            work.turns[left_record]["error"] = ErrorCode.CANCELLED.value
                    else:
                        await abandon_on(conn, left_open, segment_id=segment_id)
                if not editing:
                    inputs, state, text = _turn_opening(work, body, chip_label)
                    opened.append(
                        await open_turn_on(
                            conn,
                            pk,
                            segment_id=segment_id,
                            turn_index=work.turn_index,
                            node=studio.graph.start,
                            inputs=inputs,
                            state=state,
                            records=[entries.user_message(text)],
                        )
                    )

            await studio.store.save(
                principal,
                body.conversation_id,
                work,
                alongside=begin_turn,
                seq_reserved=SEQ_BLOCK,
            )
            conversation = studio.store.current(principal, body.conversation_id)
            assert conversation is not None
            _set_turn_context(principal, body.conversation_id, conversation, settings)
            if moved is not None:
                # A card's unit change runs in code as an edit does: no graph,
                # no model, the card redrawn in its slot.
                return open_stream(
                    _EditWork(
                        studio,
                        principal=principal,
                        conversation=conversation,
                        segment_id=segment_id,
                        events=functools.partial(
                            unit_events,
                            recorder=studio.recorder,
                            surface=studio.definition_surface,
                            conversation=conversation,
                            move=moved,
                        ),
                    ),
                    owners=studio.owners,
                    store=studio.store,
                    conversation=conversation,
                    principal=principal,
                    conversation_id=body.conversation_id,
                    segment_id=segment_id,
                    hold=hold,
                )
            if body.component_ref is not None and editing:
                # The Edit Card route bypasses the graph entirely: no
                # resolve, no plan, no turn on the loop, just the card, the
                # model's patch-or-recreate decision, and the platform write.
                # Its owner streams it under the segment the claim save above
                # just persisted.
                return open_stream(
                    _EditWork(
                        studio,
                        principal=principal,
                        conversation=conversation,
                        segment_id=segment_id,
                        events=functools.partial(
                            edit_events,
                            recorder=studio.recorder,
                            edit_model=studio.edit_model,
                            conversation=conversation,
                            component_id=body.component_ref.component_id,
                            message=body.message,
                        ),
                    ),
                    owners=studio.owners,
                    store=studio.store,
                    conversation=conversation,
                    principal=principal,
                    conversation_id=body.conversation_id,
                    segment_id=segment_id,
                    hold=hold,
                )
            (cursor,) = opened
            # The segment's owner takes the lease and the lock hold from here on.
            return open_stream(
                _SegmentWork(
                    studio,
                    principal=principal,
                    conversation=conversation,
                    cursor=cursor,
                    segment_id=segment_id,
                ),
                owners=studio.owners,
                store=studio.store,
                conversation=conversation,
                principal=principal,
                conversation_id=body.conversation_id,
                segment_id=segment_id,
                hold=hold,
                settle=studio.early_mints.settle if studio.early_mints is not None else None,
            )
        except BaseException as exc:
            # The stream's owner releases the segment and lets go of the
            # lock hold. A failure before the owner starts releases them
            # here (`_release_claim`), which does nothing once a refusal
            # above let go of the hold. A failure after the lease's claim
            # drops the lease. One before the begin-turn save commits also
            # gives up the unsaved working copy, and leaves the turn left
            # open as it stood, its pause answerable. Once that save
            # commits, the abandoned turn and the opened one stay as
            # written, and the next `/turn` closes the open one as it closes
            # any open turn. A lease claim that raised may have committed
            # its lease, a COMMIT that landed and then timed out say, so
            # that lease is dropped too (`leasing`).
            await _release_claim(
                studio.store,
                principal,
                body.conversation_id,
                hold,
                claimed,
                cause=exc,
                turn=begun,
                unsure=leasing,
            )
            raise

    router.add_api_route("/turn", turn, methods=["POST"], route_class_override=_BoundedTurnRoute)

    async def cancel(
        body: CancelRequest, authorization: str | None = Header(default=None)
    ) -> Response:
        """Stop the conversation's turn (the Stop button). Idempotent: an
        unknown conversation, and an idle one, are both a no-op 204 (a Stop
        can reach an instance before the first `/turn` has claimed the
        conversation there at all).

        A Stop that names its turn (`CancelRequest.turn_index`) stops that
        turn alone. When the conversation's latest turn is another, the
        route answers 200 with a `CancelSkipped` (`code` `turn_moved`) and
        the conversation stays as it was, so a later turn, begun on another
        tab say, runs on. On a live lease the Stop is recorded at once naming
        its turn, while the row holds that turn (`StudioStore.request_cancel`),
        so a turn a `/turn` begins under that lease after the Stop runs on. A
        Stop naming no turn stops whatever turn the conversation runs, as the
        table below gives it, and so does one naming the latest turn.

        A Stop the route acts on is answered 204 after one of three things: it recorded
        the Stop on a live lease (`StudioStore.request_cancel`), it closed
        the open turn itself under the conversation lock, or under the lock
        it found no open turn and no live lease. A Stop recorded on a lease
        names the turn that lease holds, the row's turn index
        (`harness.studio_store.StopState`), and is live until that turn
        closes, whatever the status; no write marks it read. The holder of
        the lease acts on it: its heartbeat while it drives the turn, its
        closing write (`StudioStore.close_segment`) once the drive has
        returned. Every stream ends on one frame, sent after that write, so a
        Stop the closing write acts on ends the stream on `cancelled` alone.
        When the holder cannot, the next claim does. A claim that starts or
        resumes work refuses while a Stop is live for the open turn
        (`StopPending`), so `/turn` and `/interrupt` first close that turn
        `cancelled` under a claim of their own (`_act_on_stop`). Then `/turn`
        claims and `/interrupt` answers 404. A cancelled close ends the turn's
        unfinished runs in its own transaction
        (`harness.turn.store.TurnStore.finish`). The one exception is the
        first Stop while a node holds the turn's `StopScope` armed: the
        scope takes it with no write, and the turn's next pause write
        records it taken, which ends it.

        | When the Stop commits | Acted on by | Turn after | Client is told |
        |---|---|---|---|
        | Lease live, a node running | heartbeat | `cancelled` | `cancelled` frame |
        | Lease live, `StopScope` armed, first Stop | the scope, whose re-offer's pause write records it taken | paused on the re-offered plan | `plan.proposed`, whose card can be answered |
        | The same, a second Stop before the pause write | heartbeat | `cancelled` | `cancelled` frame |
        | The same, a second Stop after the pause write, before the closing write | the closing write | `cancelled`, pending cleared | `cancelled` frame in place of `plan.proposed` |
        | The scope took the first Stop, then the client left before the pause write | the closing write | `cancelled` | nothing more |
        | The scope took the first Stop, and the pause write failed | the drive, which closes the turn on the failure | `failed` | `error` frame |
        | The scope took the first Stop, and its node ended the turn or left on an edge that is not a stop edge, or the node after it did not pause | the drive, in place of that write, and when that close fails, the closing write or the next claimant | `cancelled` | `cancelled` frame |
        | Lease live, pause committed, the drive in after-commit work | heartbeat, `harness.turn.loop._settle` | `cancelled` | `cancelled` frame |
        | Lease live, the segment returned paused, before the closing write | the closing write | `cancelled`, pending cleared | `cancelled` frame in place of the pause frame |
        | Lease live, turn closed, before the closing write | nobody, since the close ended the Stop | unchanged | nothing more |
        | `/turn` claimed, before it abandons the turn left open | the new turn's heartbeat | the old turn `abandoned`, the new `cancelled` | `cancelled` frame |
        | The same, the Stop naming the turn left open | the save that begins the new turn (`stop_names_turn`) | the old turn `cancelled`, the new running | 204, and the new turn's stream runs on |
        | Lease live, an Edit Card stream | its heartbeat | the turn's record `cancelled` | `cancelled` frame |
        | The same, the client gone before the heartbeat reads it | the closing write | the turn's record `cancelled` | nothing more |
        | The closing write failed or timed out, such as when ending the turn's runs fails | the next claimant, once the lease lapses (up to the lease's 45 s expiry, and a request before then answers a retryable 409) | `cancelled` | `cancelled` frame when the work ended on the Stop, `transient_unavailable` when it had paused, and the next `/turn` streams while an `/interrupt` answers 404 |
        | The same, and the next `/turn`'s close of the turn is refused | the `/turn` retried after its 409 | `cancelled` | nothing more, since that `/turn` answers a retryable 409 and starts nothing, and its retry streams |
        | A lapsed lease, after the next claimant's check and before its claim save | that claimant, whose save refuses | `cancelled` | nothing more, since that `/turn` answers a retryable 409 and its retry streams, and an `/interrupt` answers 404 |
        | No live lease, turn paused | this route, under the lock | `cancelled`, pending cleared | 204 |
        | No live lease, turn left running | this route, under the lock | `cancelled`, its runs ended | 204, or 503 when the close fails, which leaves a Stop live for the next claimant |
        | No live lease, no open turn | this route, under the lock, finding nothing open | unchanged | 204 |
        | No live lease, and this route's own claim loses to another request | the holder of the lease that request claimed, as this route decides again | as the row it lands on | 204, or 503 after `CANCEL_LOCK_WAIT_S` |
        | Lock held here, no live lease | the claim's holder once it claims, or this route once the lock is free | as the row it lands on | 204, or 503 after `CANCEL_LOCK_WAIT_S` |
        | The same, the claim committing just after this route found no lease | the claim's holder | as the row it lands on | 204 |
        | The same, this request cancelled once the lock passed to it | nobody, since the Stop was never recorded | unchanged | nothing, and the lock is let go |
        | The same, and the request then abandons its claim | the next claimant | `cancelled` | nothing more, and the next `/turn` streams |

        A live segment, on any instance, is stopped through its lease.
        `request_cancel` sets `lease_cancel`, or records a Stop naming its
        turn when given one, the segment's next renewal reports it as the
        Stop of the lease's turn, and the
        heartbeat stops the turn (`harness.turn.loop.run_segment`). A segment
        streaming on this instance is made to renew at once
        (`renew_lease_now`), so the Stop reaches it without waiting for the
        beat. Another instance's segment reads it within
        `HEARTBEAT_INTERVAL_S`. A
        Stop that commits after the segment's last renewal is read by the
        segment's closing write under the row lock, which closes an open
        turn `cancelled` in the same transaction. This route answers 204
        without waiting for either.

        A request on this instance that holds the lock with no lease live (a
        `/turn` or `/interrupt` before its claim save, or a stream after its
        closing write) is waited on. The Stop is recorded once the claim
        lands, or taken under the lock once it is free, and a retryable 503
        is answered after `CANCEL_LOCK_WAIT_S` of wall time.

        With no live segment on record anywhere, this takes the lock,
        reloads, and closes what is left. A turn paused on a question, or
        left running by a segment whose lease lapsed, is closed `cancelled`
        under a lease this route claims (`harness.turn.loop.stop_turn`), its
        record is closed `cancelled`, and the conversation's pending
        interrupts are cleared, so an answer to one is refused. A close that
        fails, such as ending the turn's runs, answers a retryable 503 and
        leaves the turn open with a Stop live, which the next claimant acts
        on (`_stop_open_turn`). A claim of this route's that loses to
        another request decides again from the top, within the same
        `CANCEL_LOCK_WAIT_S`.
        """
        principal = principal_for(authorization, settings=settings)
        lock = studio.store.lock_for(principal, body.conversation_id)
        loop = asyncio.get_running_loop()
        deadline = loop.time() + CANCEL_LOCK_WAIT_S
        busy = _tool_error_response(
            ToolError(
                error=ErrorCode.TRANSIENT_UNAVAILABLE,
                retry=RetryMode.RETRY_AFTER_DELAY,
                detail="the conversation is starting or ending a turn",
                retry_after_seconds=1,
            ),
            status_code=503,
        )
        delays = retry_delays()
        # Whether this route's own claim just lost to another request.
        lost = False
        while True:
            # Taken before the lease is read, so a claim that commits after
            # the read sets this very event.
            claimed = studio.store.claimed_event(principal, body.conversation_id)
            try:
                signalled = await studio.store.request_cancel(
                    principal, body.conversation_id, turn_index=body.turn_index
                )
            except TurnMoved as exc:
                return _cancel_skipped(exc.latest)
            if signalled:
                live = studio.store.current(principal, body.conversation_id)
                if live is not None:
                    renew_lease_now(live)
                return Response(status_code=204)
            if lost:
                # The request that won found no lease live either, or let it
                # go: back off before deciding again, on the store's own
                # schedule (`retry_delays`).
                if loop.time() >= deadline:
                    return busy
                await asyncio.sleep(min(next(delays), deadline - loop.time()))
                lost = False
                continue
            # No live lease: take the lock, or, while a request on this
            # instance holds it (a `/turn` or `/interrupt` before its claim
            # save, or a stream after its closing write), wait for its claim to
            # commit or for the lock, then decide again. Once the deadline
            # has passed, a free lock is taken.
            if await _lock_or_claim(lock, claimed, timeout=deadline - loop.time()):
                try:
                    return await _stop_under_lock(
                        studio, principal, body.conversation_id, turn_index=body.turn_index
                    )
                except _ClaimLost:
                    logger.info(
                        "studio cancel: conversation %s was claimed since its load; deciding again",
                        body.conversation_id,
                    )
                    lost = True
                finally:
                    lock.release()
                continue
            if claimed.is_set():
                continue
            if loop.time() >= deadline:
                return busy

    async def interrupt(  # noqa: PLR0911 -- one distinct refusal per claim guard the route enforces
        body: InterruptResponse,
        authorization: str | None = Header(default=None),
        x_dfo_account: str | None = Header(default=None),
        x_dfo_caller: str | None = Header(default=None),
    ) -> Response:
        principal = principal_for(authorization, settings=settings)
        account, visitor = studio_identity(
            account=x_dfo_account, caller=x_dfo_caller, settings=settings
        )
        hold = await _acquire_or_conflict(studio.store, principal, body.conversation_id)
        if isinstance(hold, JSONResponse):
            return hold
        # The working copy of a claim about to be saved, whose trace records
        # how a request let go of its claim before its stream started
        # (`_release_claim`), the segment that save claims, and that segment
        # from the save's start until it returns or refuses, which a save
        # that raised may have committed (`saving`).
        claiming: ConversationState | None = None
        segment_id: str | None = None
        saving: str | None = None
        try:
            conversation = await studio.store.load(principal, body.conversation_id)
            if conversation is None:
                hold.release()
                return _unknown_conversation_response(body.conversation_id)
            if await studio.store.stop_live(principal, body.conversation_id):
                # A Stop live for the turn ends it first, which leaves no
                # pause to answer.
                return await _answer_after_stop(
                    studio,
                    principal,
                    body.conversation_id,
                    hold,
                    _no_pending_interrupt_response(body.interrupt_id),
                )
            requested_answer: dict[str, Any] = {
                "decision": body.decision.value,
                "payload": body.payload,
            }
            if body.answers is not None:
                # The whole answer, so a retry with other answers is told
                # apart from the one that resumed.
                requested_answer["answers"] = {
                    question_id: answer.model_dump(mode="json")
                    for question_id, answer in body.answers.items()
                }
            if body.decision is InterruptDecision.SWITCH:
                # Likewise a retry switching to another option.
                requested_answer["switch"] = _switch_answer(body)
            # The answer reaches the paused node's `resume`. A retry of an
            # answer the turn already took finds the turn running and carries
            # it on from its cursor, when the node owes the answer its
            # `resume` or the turn is past it. A retried rejection finds
            # the turn already closed. A retry that finds the turn paused
            # again on a later question replays that question
            # (`_replays_later_pause`). Anything else answers no pause, and
            # is refused before the save records an approval of it or marks
            # it resuming.
            cursor = await studio.turns.open_cursor(principal, body.conversation_id)
            replaying = await _replays_later_pause(
                studio.turns, cursor, body.interrupt_id, requested_answer
            )
            if not replaying:
                refusal = _interrupt_refusal(conversation.pending, body)
                if refusal is not None:
                    hold.release()
                    return refusal
            # Every change up to the claim save is made on a working copy,
            # which the save swaps into the live state once it commits. A
            # request that claims nothing leaves the live state as loaded.
            work = studio.store.draft(principal, body.conversation_id)
            # Only a replay can find the entry gone (`_replays_later_pause`).
            pending = work.pending.get(body.interrupt_id)
            resuming = pending is not None and pending.resuming_segment is not None
            paused_here = (
                pending is not None
                and cursor is not None
                and cursor.status is TurnStatus.PAUSED
                and cursor.pause is not None
                and cursor.pause.interrupt_id == pending.adk_id
            )
            retried = resuming and (
                body.decision is InterruptDecision.REJECT
                or (cursor is not None and cursor.status is TurnStatus.RUNNING)
            )
            answerable = paused_here or retried or replaying
            if not (answerable or resuming):
                return await _unclaimed(
                    studio.store,
                    principal,
                    body.conversation_id,
                    hold,
                    _no_pending_interrupt_response(body.interrupt_id),
                )
            # The segment lease: claimed before the save below persists it.
            # `SegmentBusy` here means a segment is still live, either
            # this interrupt's own earlier resume, still running, or (on a
            # first answer) a double answer racing in from another
            # instance. Either way this request cannot resume right now,
            # whatever it is answering with: a live resume already under
            # way may yet land cleanly on the answer it started with.
            try:
                segment_id = studio.store.claim_segment(principal, body.conversation_id, work)
            except SegmentBusy as exc:
                return await _unclaimed(
                    studio.store,
                    principal,
                    body.conversation_id,
                    hold,
                    _streaming_conflict_response(
                        "another request is streaming on this conversation",
                        retry_after_seconds=exc.retry_after_seconds,
                    ),
                )
            if not answerable:
                # A retry of an answer that resumed the turn is refused only
                # once no segment resuming it is live: while one is, it is
                # answered 409 above, as for any retry.
                return await _unclaimed(
                    studio.store,
                    principal,
                    body.conversation_id,
                    hold,
                    _no_pending_interrupt_response(body.interrupt_id),
                )
            if pending is not None and resuming and pending.answer != requested_answer:
                # The claim above only succeeds once the earlier resume's
                # segment has expired (its owner died before reaching its
                # closing write) or was dropped (it failed before its stream
                # started), so this is a genuinely dead attempt, and
                # this request's answer does not match what actually
                # resumed it. Refused for good: a live segment above would
                # already have caught a retry of the SAME answer, so a
                # mismatch reaching here can never become a match later.
                return await _unclaimed(
                    studio.store,
                    principal,
                    body.conversation_id,
                    hold,
                    _tool_error_response(
                        ToolError(
                            error=ErrorCode.EDIT_CONFLICT,
                            retry=RetryMode.REFRESH_THEN_RETRY,
                            detail=(
                                f"interrupt {body.interrupt_id!r} already resumed with a "
                                "different decision"
                            ),
                            fix="reload the conversation and follow its recorded outcome",
                        ),
                        status_code=409,
                    ),
                )
            # The resumed segment's span is its owner's, on the turn's
            # trace, whichever instance runs it.
            claiming = work
            # The turn keeps who asked it from `/turn`. A turn that recorded
            # no one takes this request's account, visitor and caller, and
            # the surface its graph state names (`_turn_opening`), so the
            # resumed segment's span and the turn's line name them, and the
            # claim save below persists them with the segment. The caller is
            # read once the conversation it names is set. The resume
            # continues the paused turn, so it names that turn's id, as the
            # in-process dfo-apis calls do.
            current_conversation_id.set(ledger_conversation_id(principal, body.conversation_id))
            current_conversation_key.set((principal, body.conversation_id))
            current_turn_id.set(work.trace_id)
            set_supplied_caller(current_conversation_caller())
            set_supplied_turn(work.trace_id)
            surface = cursor.state.get("surface") if cursor is not None else None
            work.take_identity(
                account=account,
                visitor=visitor,
                caller=current_conversation_caller(),
                surface=surface if isinstance(surface, str) else None,
            )
            # The claim is saved alone, before the answer is recorded: a
            # double answer to the same interrupt across two instances loses
            # cleanly here, with a 409, rather than each resuming its own
            # segment of the turn.
            saving = segment_id
            try:
                await studio.store.save(principal, body.conversation_id, work)
            except StopPending:
                # A Stop that landed since the check above ends the turn
                # first, which leaves no pause to answer. Like the conflict
                # below, it is raised before the save's COMMIT.
                saving = None
                return await _answer_after_stop(
                    studio,
                    principal,
                    body.conversation_id,
                    hold,
                    _no_pending_interrupt_response(body.interrupt_id),
                    turn=claiming,
                )
            except ConversationStale:
                saving = None
                return await _unclaimed(
                    studio.store,
                    principal,
                    body.conversation_id,
                    hold,
                    _streaming_conflict_response("another request already answered this interrupt"),
                    turn=claiming,
                )
            saving = None
        except BaseException as exc:
            # A claim save that raised once its write committed, cancelled
            # during the commit say, still handed its segment to the live
            # state (`StudioStore.save`), whose lease is dropped as any
            # committed claim's. One the live state does not hold may still
            # have committed, a COMMIT that landed and then timed out say,
            # so its lease is dropped too (`unsure`). A failure before the
            # save started drops nothing. Either way the span ends as `exc`
            # reads.
            live = studio.store.current(principal, body.conversation_id)
            held = (
                segment_id is not None
                and live is not None
                and live.segment is not None
                and live.segment.id == segment_id
            )
            await _release_claim(
                studio.store,
                principal,
                body.conversation_id,
                hold,
                segment_id if held else None,
                cause=exc,
                turn=claiming,
                unsure=None if held else saving,
            )
            raise

        # Every failure from here to the stream's start drops the lease the
        # save above claimed, and releases the lock. It drops only the lease.
        # Once the answer is saved below, the pending entry stays resuming
        # under the dropped segment with the answer recorded, so a retry of
        # the same answer claims at once and resumes, with the answer's turn
        # lines saved beside it.
        try:
            # The read above only gated the claim. A segment whose lease
            # lapsed can still write the turn until the claim commits, so
            # the answer runs on the cursor read after it, and is refused
            # when that cursor does not hold the same open pause. Nothing
            # of the answer is saved yet, so the refusal leaves the entry
            # to be answered again as the reloaded conversation shows it.
            claimed = await studio.turns.open_cursor(principal, body.conversation_id)
            if not _same_open_pause(cursor, claimed):
                await _release_claim(
                    studio.store, principal, body.conversation_id, hold, segment_id, turn=claiming
                )
                return _tool_error_response(
                    ToolError(
                        error=ErrorCode.EDIT_CONFLICT,
                        retry=RetryMode.REFRESH_THEN_RETRY,
                        detail="the turn moved while this answer was claimed",
                        fix="reload the conversation and answer what it now shows",
                    ),
                    status_code=409,
                )
            cursor = claimed
            work = studio.store.draft(principal, body.conversation_id)
            if replaying:
                assert cursor is not None
                return await _replay_pause(
                    studio,
                    principal=principal,
                    conversation_id=body.conversation_id,
                    interrupt_id=body.interrupt_id,
                    work=work,
                    cursor=cursor,
                    segment_id=segment_id,
                    hold=hold,
                    settings=settings,
                    turn=claiming,
                )
            pending = work.pending[body.interrupt_id]
            # The resumed turn asks its next questions in the forms this
            # request declares, when it declares any; the pending one keeps
            # the form it was sent in.
            if body.client_features is not None:
                work.client_features = declared_features(body.client_features)
            admission: Callable[[AsyncConnection, int], Awaitable[None]] | None = None
            # A switch revises the plan and approves nothing, so it leaves
            # no approval record.
            if (
                not resuming
                and pending.kind == "plan"
                and body.decision is not InterruptDecision.SWITCH
            ):
                # Recorded once per interrupt id, on its first answer only:
                # a resumed retry of the SAME answer (below) must not
                # duplicate it. The record is of the act, not of what the
                # graph goes on to do with it
                # (`harness.approval.ApprovalRecord`'s docstring).
                #
                # Appended before the pending entry is marked resuming: if
                # constructing the record ever raises, the entry is
                # untouched and the interrupt can still be answered fresh.
                assert pending.plan_revision is not None  # every plan interrupt carries one
                record = ApprovalRecord(
                    revision=pending.plan_revision,
                    approver=principal,
                    decision=_APPROVAL_DECISION[body.decision],
                    at=datetime.now(UTC),
                    conversation_id=body.conversation_id,
                    interrupt_id=body.interrupt_id,
                    plan_version=pending.plan_version,
                    auth_enforced=settings.studio_api_token is not None,
                )
                work.approvals.append(record)
                if body.decision is InterruptDecision.APPROVE:
                    # The approval and its run commit together: the save
                    # below admits the run `execute` would, so a client
                    # that leaves at any point after it has the plan carried
                    # on by the next turn's `seed`.
                    # A first answer answers the open pause on the claimed cursor.
                    assert cursor is not None
                    admission = _approval_admission(
                        studio,
                        cursor.state,
                        pending,
                        at=record.at,
                        principal=principal,
                        conversation_id=body.conversation_id,
                    )
            # The entry is kept, marked resuming under this fresh segment.
            # A crash between this save and the resumed stream's closing
            # write still leaves a durable, retryable record. The stream's
            # owner drops it in that write on every ending but a client
            # that left (`harness.studio_stream`). A client that left, or a
            # crash, leaves it for this same retry path to pick back up.
            if not resuming:
                # The answer's lines commit with the answer, so every
                # resumed segment, and a replay of the pause it reaches, finds
                # them stored. A retry resuming the same answer adds none.
                for line, user_words in _answer_turn_lines(pending, body):
                    work.add_to_turn(line, user_words=user_words)
            pending.answer = requested_answer
            pending.resuming_segment = segment_id
            if body.decision is InterruptDecision.SWITCH and pending.card is not None:
                # What each choice held before the turn's first switch of
                # it, which `plan.approved` names as `changed_from`.
                work.turn_choice_baseline = {
                    **held_labels(pending.card),
                    **work.turn_choice_baseline,
                }
            try:
                await studio.store.save(principal, body.conversation_id, work, alongside=admission)
            except ConversationStale:
                # The lease this request holds refuses every other save, so
                # only a takeover after it lapsed moves the conversation
                # here, and nothing of the answer was stored.
                await _release_claim(
                    studio.store, principal, body.conversation_id, hold, segment_id, turn=claiming
                )
                return _tool_error_response(
                    ToolError(
                        error=ErrorCode.EDIT_CONFLICT,
                        retry=RetryMode.REFRESH_THEN_RETRY,
                        detail="the conversation changed before this answer was saved",
                        fix="reload the conversation and answer what it now shows",
                    ),
                    status_code=409,
                )
            conversation = studio.store.current(principal, body.conversation_id)
            assert conversation is not None
            pending = conversation.pending[body.interrupt_id]
            # A workflow's own steps run as concurrent waves, whose recording
            # order `_RecordingSurface` needs the plan's own step order to
            # undo (`_PendingInterrupt.step_order`'s docstring). It and the
            # spec-use sink are set before anything below can run the graph
            # or close the turn.
            current_spec_uses.set(
                _spec_use_sink(studio.store, principal, body.conversation_id, conversation.trace_id)
            )
            if studio.surface_recorder is not None:
                studio.surface_recorder.set_step_order(
                    ledger_conversation_id(principal, body.conversation_id),
                    pending.step_order,
                    pending.step_labels,
                )

            if body.decision is InterruptDecision.REJECT:
                # The plan is declined and the turn ends; nothing was computed. Its
                # early retrievals, if any, are left to finish or lapse. A retry
                # after the close committed finds the turn closed and ends it
                # on the same two frames.
                if paused_here:
                    assert cursor is not None and cursor.pause is not None
                    await reject(
                        studio.turns,
                        cursor,
                        segment_id=segment_id,
                        answer=Answer(
                            interrupt_id=pending.adk_id,
                            round=cursor.pause.round,
                            value={"rejected": True},
                        ),
                    )
                # The segment's owner takes the lease and the lock hold from here on.
                return open_stream(
                    _RejectWork(conversation),
                    owners=studio.owners,
                    store=studio.store,
                    conversation=conversation,
                    principal=principal,
                    conversation_id=body.conversation_id,
                    segment_id=segment_id,
                    hold=hold,
                )

            assert cursor is not None  # a paused or running turn, checked before the claim
            if pending.kind == "elicitation":
                # skip dismisses the whole pause; an answer resumes with every
                # question's answer, keyed by question id.
                if body.decision is InterruptDecision.SKIP:
                    answer: dict[str, Any] = {"skip": True}
                else:
                    given = _given_answers(pending, body)
                    answer = {
                        "answers": {
                            question.id: _resume_answer(question, given[question.id])
                            for question in pending.questions
                        }
                    }
            elif body.decision is InterruptDecision.APPROVE:
                # The plan is keyed on its revision hash (`plan_revision`), and
                # the approval dated by the record of its first answer, so a
                # retry resuming the same answer later approves as of the same
                # moment. The graph's `approve` node approves nothing without it.
                answer = {"approved": pending.plan_revision}
                if pending.admitted_run is not None:
                    # The run the approval's save admitted, for `execute`.
                    answer["run_id"] = pending.admitted_run
                recorded = next(
                    (
                        r
                        for r in reversed(conversation.approvals)
                        if r.interrupt_id == body.interrupt_id
                    ),
                    None,
                )
                if recorded is not None:
                    answer["at"] = recorded.at.isoformat()
            elif body.decision is InterruptDecision.SWITCH:
                # The graph's `switch` node revises the plan in code and offers
                # it again; a planner decision's switch (`is_decision_choice`)
                # replans with the pick held (`harness.decision_pick`). The
                # turn's history record carries the choice (`_answer_turn_lines`).
                assert body.choice_id is not None and body.option_id is not None
                answer = {"approved": None, "switch": _switch_answer(body)}
            else:  # amend: a stated non-approval routes the graph to the planner
                answer = {"approved": None, "amendment": body.payload}
            # The user's words as amended, so a search after the resume reads the
            # refined ask and no switch's label.
            _set_turn_context(principal, body.conversation_id, conversation, settings)
            # The approved plan goes out when the graph's `approve` node takes
            # the approval. A retry finds that node run already when the dead
            # attempt's write of it committed, and sends it first instead, since
            # the client that left may never have read it.
            approving = pending.card if body.decision is InterruptDecision.APPROVE else None
            taken = _taken_approval(conversation, cursor, approving)
            # The segment's owner takes the lease and the lock hold from here on.
            return open_stream(
                _SegmentWork(
                    studio,
                    principal=principal,
                    conversation=conversation,
                    cursor=cursor,
                    segment_id=segment_id,
                    answer=(
                        Answer(
                            interrupt_id=pending.adk_id,
                            round=cursor.pause.round,
                            value=answer,
                            request=requested_answer,
                        )
                        if paused_here and cursor.pause is not None
                        else None
                    ),
                    approving=None if taken is not None else approving,
                    restored=() if taken is None else (taken,),
                ),
                owners=studio.owners,
                store=studio.store,
                conversation=conversation,
                principal=principal,
                conversation_id=body.conversation_id,
                segment_id=segment_id,
                hold=hold,
                settle=studio.early_mints.settle if studio.early_mints is not None else None,
            )
        except BaseException as exc:
            await _release_claim(
                studio.store,
                principal,
                body.conversation_id,
                hold,
                segment_id,
                cause=exc,
                turn=claiming,
            )
            raise

    router.add_api_route(
        "/interrupt", interrupt, methods=["POST"], route_class_override=_BoundedInterruptRoute
    )

    async def resume(  # noqa: PLR0911 -- one distinct refusal per claim guard the route enforces
        body: ResumeRequest,
        authorization: str | None = Header(default=None),
        x_dfo_account: str | None = Header(default=None),
        x_dfo_caller: str | None = Header(default=None),
    ) -> Response:
        """Carry on turn `turn_index` from its cursor, on a new stream, when
        it is the conversation's open turn, running, and no segment holds
        it: a turn whose client left while it ran, whose owner then
        cancelled the drive and cleared the lease
        (`harness.studio_stream`). This is the path an `/interrupt` retry
        takes when it finds the turn running, with no answer: the segment
        runs the node at the cursor from its start, and a node that owes an
        accepted answer to its `resume` reads it from the turn
        (`harness.turn.loop.run_segment`). An answer to an approved plan
        whose approval the turn has not yet taken sends `plan.approved`
        once, as that retry does.

        The route acts on the turn's standing (`turn_standing`), the one
        the state read reports, derived before the claim and again from the
        cursor read after it, since a segment whose lease lapsed can move
        the turn until the claim commits. A turn that is not the latest, a
        closed turn, or a turn paused on a question answers 409
        `edit_conflict` (`_resume_refusal`). A live lease answers the 409 `transient_unavailable` every
        claim conflict answers, with the lease's remaining time, and a Stop
        live for the turn closes it first and answers `edit_conflict`.

        A pending entry the departed client's segment left resuming is
        marked resuming under this segment and saved before the stream, as
        `_replay_pause` marks one, so the stream's closing write drops it on
        every ending but another client that leaves, which keeps it for the
        next resume."""
        principal = principal_for(authorization, settings=settings)
        account, visitor = studio_identity(
            account=x_dfo_account, caller=x_dfo_caller, settings=settings
        )
        conversation_id = body.conversation_id
        hold = await _acquire_or_conflict(studio.store, principal, conversation_id)
        if isinstance(hold, JSONResponse):
            return hold
        # As in `/interrupt`: the working copy of the claim, whose trace
        # records how the request let go of it, the segment the claim save
        # claims, and that segment while the save runs (`saving`).
        claiming: ConversationState | None = None
        segment_id: str | None = None
        saving: str | None = None
        # The answer a client left before the turn took it, carried on as a
        # retry of it once this request let go of the conversation.
        carry: InterruptResponse | None = None
        try:
            conversation = await studio.store.load(principal, conversation_id)
            if conversation is None:
                hold.release()
                return _unknown_conversation_response(conversation_id)
            if await studio.store.stop_live(principal, conversation_id):
                # A Stop live for the turn closes it, which leaves no turn to
                # carry on.
                return await _answer_after_stop(
                    studio,
                    principal,
                    conversation_id,
                    hold,
                    _not_open_turn_response(body.turn_index),
                )
            cursor = await studio.turns.open_cursor(principal, conversation_id)
            standing = _cursor_standing(
                cursor,
                conversation,
                lease_live=studio.store.lease_live(principal, conversation_id),
            )
            if body.turn_index == conversation.turn_index and standing.standing is Standing.CARRY:
                # A client left after its answer was recorded and before the
                # turn took it: carry the turn on as a retry of that answer
                # does, which claims the conversation itself.
                assert cursor is not None and cursor.pause is not None
                found = pause_entry(conversation.pending, cursor.pause.interrupt_id)
                assert found is not None
                carry = _answer_again(conversation_id, *found)
                hold.release()
            else:
                refusal = (
                    _resume_refusal(standing, body.turn_index)
                    if body.turn_index == conversation.turn_index
                    else _not_open_turn_response(body.turn_index)
                )
                if refusal is not None:
                    hold.release()
                    return refusal
                work = studio.store.draft(principal, conversation_id)
                try:
                    segment_id = studio.store.claim_segment(principal, conversation_id, work)
                except SegmentBusy as exc:
                    return await _unclaimed(
                        studio.store,
                        principal,
                        conversation_id,
                        hold,
                        _streaming_conflict_response(
                            "another request is streaming on this conversation",
                            retry_after_seconds=exc.retry_after_seconds,
                        ),
                    )
                claiming = work
                # The resumed turn keeps who asked it, and names its own id, as
                # an `/interrupt` resume does.
                current_conversation_id.set(ledger_conversation_id(principal, conversation_id))
                current_conversation_key.set((principal, conversation_id))
                current_turn_id.set(work.trace_id)
                set_supplied_caller(current_conversation_caller())
                set_supplied_turn(work.trace_id)
                surface = cursor.state.get("surface") if cursor is not None else None
                work.take_identity(
                    account=account,
                    visitor=visitor,
                    caller=current_conversation_caller(),
                    surface=surface if isinstance(surface, str) else None,
                )
                saving = segment_id
                try:
                    await studio.store.save(principal, conversation_id, work)
                except StopPending:
                    saving = None
                    return await _answer_after_stop(
                        studio,
                        principal,
                        conversation_id,
                        hold,
                        _not_open_turn_response(body.turn_index),
                        turn=claiming,
                    )
                except ConversationStale:
                    saving = None
                    return await _unclaimed(
                        studio.store,
                        principal,
                        conversation_id,
                        hold,
                        _streaming_conflict_response(
                            "another request is streaming on this conversation"
                        ),
                        turn=claiming,
                    )
                saving = None
        except BaseException as exc:
            # As in `/interrupt`: a claim save that raised once it committed
            # handed its segment to the live state, and one the live state
            # does not hold may still have committed.
            live = studio.store.current(principal, conversation_id)
            held = (
                segment_id is not None
                and live is not None
                and live.segment is not None
                and live.segment.id == segment_id
            )
            await _release_claim(
                studio.store,
                principal,
                conversation_id,
                hold,
                segment_id if held else None,
                cause=exc,
                turn=claiming,
                unsure=None if held else saving,
            )
            raise

        if carry is not None:
            # The retry claims the conversation itself, outside this
            # request's cleanup, which holds no claim of its own here.
            return _carried_answer(
                await interrupt(
                    carry,
                    authorization=authorization,
                    x_dfo_account=x_dfo_account,
                    x_dfo_caller=x_dfo_caller,
                ),
                body.turn_index,
            )
        assert segment_id is not None
        try:
            claimed = await studio.turns.open_cursor(principal, conversation_id)
            live = studio.store.current(principal, conversation_id)
            assert live is not None
            # The claim is this request's own lease, so it holds nothing
            # against the resume.
            standing = _cursor_standing(claimed, live, lease_live=False)
            refusal = (
                _resume_refusal(standing, body.turn_index)
                if body.turn_index == live.turn_index
                else _not_open_turn_response(body.turn_index)
            )
            if refusal is not None:
                await _release_claim(
                    studio.store, principal, conversation_id, hold, segment_id, turn=claiming
                )
                return refusal
            assert claimed is not None
            work = studio.store.draft(principal, conversation_id)
            left = [p for p in work.pending.values() if p.resuming_segment is not None]
            for pending in left:
                pending.resuming_segment = segment_id
            if standing.standing is Standing.REPLAY:
                # No entry holds the pause the cursor stands on: send it
                # again with no node run, and its stream's closing write
                # stores the entry the next answer answers.
                assert claimed.pause is not None
                return await _replay_pause(
                    studio,
                    principal=principal,
                    conversation_id=conversation_id,
                    interrupt_id=claimed.pause.interrupt_id,
                    work=work,
                    cursor=claimed,
                    segment_id=segment_id,
                    hold=hold,
                    settings=settings,
                    turn=claiming,
                    stale=_resume_conflict("the conversation changed before this resume was saved"),
                )
            if left:
                try:
                    await studio.store.save(principal, conversation_id, work)
                except ConversationStale:
                    await _release_claim(
                        studio.store, principal, conversation_id, hold, segment_id, turn=claiming
                    )
                    return _resume_conflict("the conversation changed before this resume was saved")
            else:
                studio.store.drop_draft(principal, conversation_id)
            conversation = studio.store.current(principal, conversation_id)
            assert conversation is not None
            answered = next(
                (p for p in conversation.pending.values() if p.resuming_segment == segment_id),
                None,
            )
            current_spec_uses.set(
                _spec_use_sink(studio.store, principal, conversation_id, conversation.trace_id)
            )
            if answered is not None and studio.surface_recorder is not None:
                studio.surface_recorder.set_step_order(
                    ledger_conversation_id(principal, conversation_id),
                    answered.step_order,
                    answered.step_labels,
                )
            _set_turn_context(principal, conversation_id, conversation, settings)
            approving = (
                answered.card
                if answered is not None
                and answered.answer is not None
                and answered.answer.get("decision") == InterruptDecision.APPROVE.value
                else None
            )
            # An approval the turn took before its client left goes out
            # first, with every other the turn recorded, since the page may
            # have read none of them.
            taken = _taken_approval(conversation, claimed, approving)
            # The segment's owner takes the lease and the lock hold from here on.
            return open_stream(
                _SegmentWork(
                    studio,
                    principal=principal,
                    conversation=conversation,
                    cursor=claimed,
                    segment_id=segment_id,
                    approving=None if taken is not None else approving,
                    restored=_turn_approvals(conversation, taken),
                ),
                owners=studio.owners,
                store=studio.store,
                conversation=conversation,
                principal=principal,
                conversation_id=conversation_id,
                segment_id=segment_id,
                hold=hold,
                settle=studio.early_mints.settle if studio.early_mints is not None else None,
            )
        except BaseException as exc:
            await _release_claim(
                studio.store,
                principal,
                conversation_id,
                hold,
                segment_id,
                cause=exc,
                turn=claiming,
            )
            raise

    async def composition(
        body: CompositionMutation, authorization: str | None = Header(default=None)
    ) -> Response:
        # Routed through the composition lock, not the turn lock: a canvas
        # edit must land while a turn streams on the same conversation. The
        # turn writes each of its own card changes under the same row lock
        # (`StudioStore.change_canvas`), so neither overwrites the other, and
        # each card it lands commits its digest entry with its block, so the
        # rows this route reads hold every card a stream has landed. The
        # composition lock also serialises two same-instance `/composition`
        # calls, so they read-mutate-write in turn rather than interleaving.
        principal = principal_for(authorization, settings=settings)
        async with studio.store.composition_lock_for(principal, body.conversation_id):

            def mutate(
                manifest: CompositionManifest | None, components: dict[str, dict[str, Any]]
            ) -> CompositionManifest | JSONResponse:
                refusal = _composition_refusal(manifest, body)
                if refusal is not None:
                    return refusal
                # Everything below works on LOCAL copies; the manifest this
                # route hands back to `mutate_composition` is built once,
                # after every operation validated. A refused op must not
                # leave a half-applied canvas behind (mutations are
                # transactional).
                applied = _apply_operations(components, manifest, body.operations)
                if isinstance(applied, JSONResponse):
                    return applied
                blocks, layout_hints = applied
                if (
                    manifest is not None
                    and blocks == list(manifest.blocks)
                    and layout_hints == manifest.layout_hints
                ):
                    # Nothing changed (e.g. an add of an already-composed
                    # component): minting a revision anyway would hand every
                    # optimistic-concurrency client a spurious edit
                    # conflict. Returning the same object back signals the
                    # no-op to `mutate_composition`.
                    return manifest
                return _next_manifest(
                    manifest, blocks, actor=CompositionActor.USER, layout_hints=layout_hints
                )

            result = await studio.store.mutate_composition(
                principal,
                body.conversation_id,
                mutate,
                component_ids={op.component_id for op in body.operations if isinstance(op, AddOp)},
            )
            if result is None:
                return _unknown_conversation_response(body.conversation_id)
            if isinstance(result, JSONResponse):
                return result
            return JSONResponse(content=json.loads(result.model_dump_json(exclude_none=True)))

    @router.get("/state")
    async def state(
        conversation_id: str,
        since: int | None = Query(default=None, ge=0),
        authorization: str | None = Header(default=None),
    ) -> Response:
        """The conversation as stored (`harness.wire.ConversationRead`), read
        with no lock and no lease (`StudioStore.peek`), so it answers while a
        turn streams on any instance and changes nothing. `since` names the
        last turn the client holds; the closed turns after it come back,
        oldest first, at most `MAX_CLOSED_SINCE`. An unknown conversation
        answers 404 `not_found`. Members with no value are left out, as a
        stream's frames leave them out."""
        principal = principal_for(authorization, settings=settings)
        stored = await studio.store.peek(principal, conversation_id)
        if stored is None:
            return _unknown_conversation_response(conversation_id)
        read = _conversation_read(stored, since=since)
        return JSONResponse(content=json.loads(read.model_dump_json(exclude_none=True)))

    # Each body is bounded before it is parsed (`_BoundedRoute`).
    router.add_api_route(
        "/cancel", cancel, methods=["POST"], route_class_override=_BoundedCancelRoute
    )
    router.add_api_route(
        "/resume", resume, methods=["POST"], route_class_override=_BoundedResumeRoute
    )
    router.add_api_route(
        "/composition", composition, methods=["POST"], route_class_override=_BoundedCompositionRoute
    )

    return router
