"""The studio endpoints: the v1 wire contract served over the orchestrator workflow.

Three routes (`docs/studio-contract-v1.md`):

* `POST /api/studio/turn`: runs one turn and streams v1 frames as SSE.
* `POST /api/studio/interrupt`: resumes a paused turn; the response is a
  NEW SSE stream continuing it (contract delta 1).
* `POST /api/studio/composition`: applies canvas mutations and returns the
  new manifest as plain JSON.

The translator from ADK events to frames is `harness.studio_translate`;
per-conversation state and the composition rules are `harness.studio_state`;
the Edit Card route is `harness.studio_edit`. This module owns the routes,
the `StudioRunner` that drives a turn, and the SSE transport.

Conversation identity (contract delta 5): the client mints
`conversation_id`; the bearer token resolves to a principal
(`harness.auth`); the store and ADK sessions are keyed on the pair. Each
`turn.request` starts a FRESH ADK session (`{conversation_id}#t{n}`) --
graph state is per-turn by design (turn 2's planner drafts from a clean
state, with no plan left over from turn 1), while the composition
manifest and the turn records persist across turns in the store. Cross-turn
context reaches the models as DIGESTS seeded into each turn's session
state: the canvas digest (block titles + fact summaries) and the
conversation digest (the newest turn records); anything past their bounds
is served on demand by the recall function (`make_recall_part`). Interrupts resume
the current turn's session.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import AsyncGenerator, AsyncIterator, Callable, Coroutine
from contextlib import (
    AbstractContextManager,
    aclosing,
    asynccontextmanager,
    nullcontext,
    suppress,
)
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from typing import Any, Literal, override

import anyio
from common.telemetry import flush_telemetry
from compiler.surface import ToolSurface
from fastapi import APIRouter, Header, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from fastapi.routing import APIRoute
from google.adk.agents.run_config import RunConfig, StreamingMode
from google.adk.apps import App
from google.adk.errors.already_exists_error import AlreadyExistsError
from google.adk.events import Event
from google.adk.runners import Runner
from google.adk.workflow.utils._workflow_hitl_utils import (
    create_request_input_response,
)
from google.genai import types
from planner.plan import SEEDS_STATE_KEY, current_earlier_block, current_recorded_identity
from starlette.requests import ClientDisconnect
from starlette.types import Receive, Scope, Send

from harness.approval import ApprovalDecision, ApprovalRecord
from harness.auth import principal_for
from harness.conversation import (
    current_conversation_id,
    current_conversation_key,
    current_user_message,
)
from harness.decision_pick import is_decision_choice
from harness.early_mint import EarlyMints
from harness.graph.interfaces import Executor
from harness.graph.state import (
    AGENT_NAME,
    APPLY_CHOICE_STATE_KEY,
    ORCHESTRATOR_APPLIED_KEY,
    ORCHESTRATOR_PLAN_RECORDS_KEY,
    ORCHESTRATOR_PRIOR_LEDGER_KEY,
    ORCHESTRATOR_SEARCHES_KEY,
    ORCHESTRATOR_SPECS_KEY,
)
from harness.graph.switch import SWITCH_STOP
from harness.provenance import ProvenanceStore
from harness.seeds import seeds_slot
from harness.sentences import failure_sentence
from harness.settings import Settings
from harness.studio_edit import edit_frames
from harness.studio_state import (
    HEARTBEAT_INTERVAL_S,
    STREAM_LOSSES,
    ClosedTurn,
    ConversationState,
    DatabaseUnavailable,
    SegmentBusy,
    SegmentLost,
    StreamLoss,
    _apply_operations,
    _end_stream,
    _next_manifest,
    _PendingInterrupt,
    _PendingQuestion,
    _tool_error_response,
    record_closed_turn_span,
    stream_loss,
)
from harness.studio_store import (
    ConversationExists,
    ConversationHeld,
    ConversationStale,
    StudioStore,
)
from harness.studio_translate import (
    _checkpoint_failed_frame,
    _RecordingExecutor,
    _RecordingSurface,
    _stream_fatal,
    _Translator,
    held_labels,
    segment_lost_frame,
)
from harness.turn_budget import start_turn_budget
from harness.turn_records import CHIP_STALE_NOTE, stored_choice
from harness.wire import (
    MAX_TURN_REQUEST_BYTES,
    ActivityStep,
    AddOp,
    ApplyChoice,
    CancelRequest,
    ComponentRefIntent,
    CompositionActor,
    CompositionManifest,
    CompositionMutation,
    Done,
    ElicitationAnswer,
    ErrorCode,
    ErrorFrame,
    InterruptDecision,
    InterruptResponse,
    MessageComplete,
    OptionAnswer,
    PlanProposed,
    PlanRevisionStaleError,
    RetryMode,
    SkippedAnswer,
    StepStatus,
    TextAnswer,
    ToolError,
    TurnRequest,
)

logger = logging.getLogger(__name__)

HEARTBEAT_SECONDS = 5.0
SSE_MEDIA_TYPE = "text/event-stream"
# How long a stream a Stop reached while a plan-card switch ran waits for the
# `switch` node to offer the plan again before it ends as any Stop does.
SWITCH_REOFFER_SECONDS = 10.0

# The wire decision on a plan interrupt, as the record `/interrupt` keeps of it.
_APPROVAL_DECISION = {
    InterruptDecision.APPROVE: ApprovalDecision.APPROVED,
    InterruptDecision.AMEND: ApprovalDecision.AMENDED,
    InterruptDecision.REJECT: ApprovalDecision.REJECTED,
}


def _answers_refusal(
    questions: tuple[_PendingQuestion, ...], answers: dict[str, ElicitationAnswer]
) -> str | None:
    """What is wrong with `answers` to `questions`, or `None`. Valid answers
    name every question exactly once, pick only that question's offered
    options, and skip only a skippable question."""
    asked = [question.id for question in questions]
    missing = [question_id for question_id in asked if question_id not in answers]
    extra = sorted(set(answers) - set(asked))
    if missing or extra:
        return f"answers must name exactly the questions {asked}: missing {missing}, extra {extra}"
    for question in questions:
        answer = answers[question.id]
        if isinstance(answer, OptionAnswer) and answer.option_id not in question.options:
            return f"option {answer.option_id!r} is not offered by question {question.id!r}"
        if isinstance(answer, SkippedAnswer) and not question.skippable:
            return f"question {question.id!r} cannot be skipped"
    return None


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


def _switch_refusal(pending: _PendingInterrupt, body: InterruptResponse) -> JSONResponse | None:
    """What is wrong with a `switch` on a plan card, as one HTTP refusal, or `None`.

    A switch names the revision of the plan the user saw, or is refused
    409 `plan_revision_stale`. It moves a choice the card shows
    (`_PendingInterrupt.choices`), a scope choice or a planner decision
    (`harness.decision_pick.is_decision_choice`), to an option the card
    offers as available or partial, other than the active one, or is refused
    422 `switch_option_unavailable`. `detail` is copy the card shows.
    """
    if body.plan_revision != pending.plan_revision:
        return _stale_plan_refusal(pending)
    choice = pending.choices.get(body.choice_id or "")
    option = choice.options.get(body.option_id or "") if choice is not None else None
    if choice is None:
        detail = "That choice cannot be switched on this plan."
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
    `switch_option_unavailable`, as a plan card's switch is. `detail` is copy
    the client shows.
    """
    answered = [
        number
        for number, turn in enumerate(conversation.turns, start=1)
        if turn.get("assistant") and not turn.get("error")
    ]
    readings = [reading for turn in conversation.turns for reading in turn.get("readings") or ()]
    if not conversation.turn_recorded:
        readings.extend(conversation.turn_readings)
    found = stored_choice(readings, pick.choice_id)
    reading, choice = found if found is not None else (None, None)
    option = (
        next((o for o in choice.options if o.option_id == pick.option_id), None)
        if choice is not None
        else None
    )
    if (
        answered[-1:] != [pick.turn]
        or reading is None
        or choice is None
        or reading.spec_id != pick.spec_id
        or (option is not None and pick.option_id == choice.active)
    ):
        constraint, detail = "scope_choice_stale", CHIP_STALE_NOTE
    elif option is None:
        constraint, detail = "switch_option_unavailable", "That option is not offered here."
    elif option.availability.status == "unavailable":
        constraint, detail = (
            "switch_option_unavailable",
            f"{option.label} cannot be counted for this answer.",
        )
    else:
        return option.label
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
    return {question.id: TextAnswer(kind="text", text=body.payload)}


def _resume_answer(answer: ElicitationAnswer) -> dict[str, str]:
    """One answer as the graph reads it: a pick names its candidate id."""
    match answer:
        case OptionAnswer(option_id=option_id):
            return {"kind": "option", "spec_id": option_id}
        case TextAnswer(text=text):
            return {"kind": "text", "text": text}
        case SkippedAnswer():
            return {"kind": "skipped"}


def _answered_lines(pending: _PendingInterrupt, answers: dict[str, ElicitationAnswer]) -> list[str]:
    """The answers the turn's record keeps, readable: an option's label, with
    its spec_id when the option carries one, or the user's own words. A lone
    question's answer stands alone. With several, each follows its question.
    Skipped questions, and a legacy pick whose label was never kept, add
    nothing."""
    lines: list[str] = []
    for question in pending.questions:
        match answers[question.id]:
            case OptionAnswer(option_id=option_id):
                said = question.options.get(option_id)
                if said and option_id.startswith("spec_"):
                    said = f"{said} ({option_id})"
            case TextAnswer(text=text):
                said = text
            case SkippedAnswer():
                said = None
        if said:
            lines.append(said if len(pending.questions) == 1 else f"{question.question} {said}")
    return lines


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


def _sse_data(frame: Any) -> str:
    return f"data: {frame.model_dump_json(exclude_none=True)}\n\n"


class _OpenSteps:
    """The Activity lines one stream has opened and not yet closed.

    Studio shows a line as pending while its latest frame carries no
    `duration_ms`: a `running` or `pending` step report. A later frame on
    the same `step_id` with a `duration_ms` closes it. A stream that ends
    early sends no such frame, so it closes each open line itself.
    """

    def __init__(self) -> None:
        self._open: dict[str, tuple[ActivityStep, float]] = {}

    def observe(self, frame: Any) -> None:
        if not isinstance(frame, ActivityStep):
            return
        if frame.duration_ms is None and frame.status in (StepStatus.RUNNING, StepStatus.PENDING):
            self._open.setdefault(frame.step_id, (frame, time.monotonic()))
        else:
            self._open.pop(frame.step_id, None)

    def close(self, conversation: ConversationState, *, detail: str) -> list[ActivityStep]:
        """A `failed` frame for every open line, timed from its first frame."""
        now = time.monotonic()
        frames = [
            ActivityStep(
                event="activity.step",
                seq=conversation.next_seq(),
                step_id=step_id,
                label=opened.label,
                status=StepStatus.FAILED,
                tool=opened.tool,
                detail=detail,
                duration_ms=int((now - started) * 1000),
            )
            for step_id, (opened, started) in self._open.items()
        ]
        self._open.clear()
        return frames


async def _sse_stream(
    frames: AsyncGenerator[Any],
    *,
    conversation: ConversationState | None = None,
    switch: bool = False,
    heartbeat_seconds: float = HEARTBEAT_SECONDS,
    ending: _StreamEnding | None = None,
) -> AsyncGenerator[str]:
    """Serialise frames as SSE `data:` lines, one loop owning all three
    transport concerns: `: ka` comment heartbeats while the next frame is
    slow; cancellation via the conversation's cancel event when one is
    supplied; and client disconnect. This changes no turn state. What it
    reads goes on `ending` (`_StreamEnding`), and the segment that `frames`
    is (`StudioRunner._locked`) acts on it in its own task, before its
    release saves the turn.

    The cancel event fires for two different reasons, told apart by
    `conversation.cancel_reason` (set alongside the event by whoever armed
    it, `POST /cancel` or `StudioRunner._locked`'s heartbeat). `"cancelled"`
    is a client's Stop, from this instance or another, except on a stream
    running a plan-card switch (`switch`, which arms
    `conversation.switch_stop` and `SWITCH_STOP` for the frames this stream
    reads, and clears the one it armed when it ends): the first Stop there
    sets `switch_stop` and the stream reads on, ending on the
    `plan.proposed` the `switch` node offers with the plan unchanged
    (`harness.graph.switch.SWITCH_STOP`). The cancel event is armed again,
    so a second Stop ends the stream as any Stop does, and so does the
    stream reaching `SWITCH_REOFFER_SECONDS` after the first with no
    `plan.proposed`. A Stop never leaves a stream open. Once that frame
    passes, the deadline is met, so a slow unwind after it leaves the
    paused turn open. `"segment_lost"` (this stream's segment lease was
    taken over: the heartbeat's renewal, a card removal or the run ledger
    found it so) and `"database_unavailable"` (the heartbeat's renewals or
    a card removal could not reach the database before the lease's
    deadline) are a loss: another instance now owns the segment, or may
    claim it once its lease lapses, and may still complete it.

    A Stop or a loss is read once (`_StreamEnding.read_stop`), and the
    stream reads `frames` on to their end: the segment ends its own frames
    and sends the frames that end the stream (`StudioRunner._locked`). An
    ending frame, once sent, fixes how the stream ended, with one exception:
    after `done` or a stream-fatal error frame, a Stop or a loss is ignored,
    and after a pause frame a loss is, while a Stop still ends the stream on
    `cancelled`. A stream that ignores one stops listening, clearing the
    cancel event, so a later `/cancel` takes its own path. A disconnect, at
    a wait or at any `yield`, records that the client left, and unwinds the
    segment.

    A stream that has a `conversation` runs inside its segment's telemetry
    span (`ConversationState.in_turn_span`), on the turn's trace, so a turn
    resumed by a second request exports on the same trace, and the task
    that produces each frame is created inside that scope. The scope is
    left on the task that iterates the response. When that task is
    cancelled while this generator sits at a `yield`, `_ClaimedStream`
    closes the generator as the response ends, in the request task. Under
    ASGI spec 2.3 Starlette iterates the body in a task of its own, so the
    close runs in another context, where leaving the scope detaches nothing
    (`common.telemetry.use_turn_span`). On either path the finished spans
    are exported (`flush_telemetry`) as the
    stream ends, while the request still has CPU. Each frame is recorded
    on `ending` as it goes out (`_StreamEnding.sent`), which names the
    stream's ending, and its Activity lines (`_StreamEnding.open_steps`)."""
    turn_span: AbstractContextManager[None] = (
        conversation.in_turn_span() if conversation is not None else nullcontext()
    )
    ending = ending or _StreamEnding()
    cancel_wait: asyncio.Future[Any] | None = None
    # The loop time by which a stopped switch must have ended the stream.
    reoffer_by: float | None = None
    loop = asyncio.get_running_loop()
    switch_stop: asyncio.Event | None = None
    # The cancel event this stream last armed, so its `finally` clears only
    # that one and not the event of a request that took the conversation after
    # `_locked` released it.
    armed_cancel: asyncio.Event | None = None
    next_frame: asyncio.Future[Any] | None = None
    if conversation is not None:
        armed_cancel = conversation.cancel_event = asyncio.Event()
        # A reason left by an earlier stream was aimed at that stream.
        conversation.cancel_reason = None
        cancel_wait = asyncio.ensure_future(armed_cancel.wait())
        if switch:
            # Armed here, by the stream the switch runs on, so an earlier
            # stream's `finally` cannot clear it.
            switch_stop = conversation.switch_stop = asyncio.Event()
            SWITCH_STOP.set(switch_stop)
    with turn_span:
        try:
            async with aclosing(frames) as source:
                iterator = source.__aiter__()
                try:
                    while True:
                        next_frame = asyncio.ensure_future(anext(iterator))
                        while True:
                            race = {next_frame, cancel_wait} if cancel_wait else {next_frame}
                            done, _ = await asyncio.wait(
                                race,
                                timeout=heartbeat_seconds
                                if reoffer_by is None
                                else max(0.0, min(heartbeat_seconds, reoffer_by - loop.time())),
                                return_when=asyncio.FIRST_COMPLETED,
                            )
                            if (
                                next_frame in done
                                and not next_frame.cancelled()
                                and next_frame.exception() is None
                                and isinstance(next_frame.result(), PlanProposed)
                            ):
                                # The stopped switch offered its plan in time.
                                reoffer_by = None
                            # Checked on every frame too, since a stopped switch
                            # that keeps sending frames is not idle.
                            overdue = reoffer_by is not None and loop.time() >= reoffer_by
                            if next_frame in done and not overdue:
                                break
                            if overdue or (cancel_wait is not None and cancel_wait in done):
                                assert conversation is not None
                                if overdue and conversation.cancel_reason is None:
                                    # The stopped switch offered nothing in
                                    # time. A takeover recorded in the same
                                    # wait keeps its own reason.
                                    conversation.cancel_reason = "cancelled"
                                stop = conversation.switch_stop
                                if (
                                    conversation.cancel_reason == "cancelled"
                                    and stop is not None
                                    and not stop.is_set()
                                ):
                                    stop.set()
                                    reoffer_by = loop.time() + SWITCH_REOFFER_SECONDS
                                    armed_cancel = conversation.cancel_event = asyncio.Event()
                                    conversation.cancel_reason = None
                                    cancel_wait = asyncio.ensure_future(armed_cancel.wait())
                                    continue
                                reason = conversation.cancel_reason
                                lost = reason in STREAM_LOSSES
                                # Read once: the stream stops listening, and
                                # reads its frames to their end.
                                cancel_wait = reoffer_by = None
                                if ending.final or (ending.status == "paused" and lost):
                                    # The stream's ending frame went out
                                    # first, and stands. The stream stops
                                    # listening, so `/cancel` takes its own
                                    # path.
                                    if conversation.cancel_event is armed_cancel:
                                        conversation.cancel_event = None
                                    continue
                                # The segment ends its frames, and sends the
                                # frames that end the stream
                                # (`StudioRunner._locked`).
                                ending.read_stop(reason if lost else "cancelled")
                                continue
                            yield ": ka\n\n"
                        if next_frame.cancelled():
                            # A segment with no ending frames of its own.
                            return
                        try:
                            frame = next_frame.result()
                        except StopAsyncIteration:
                            return
                        ending.open_steps.observe(frame)
                        ending.sent(frame)
                        yield _sse_data(frame)
                except (asyncio.CancelledError, GeneratorExit):
                    # CancelledError: the response task was cancelled while
                    # this waited. GeneratorExit: the consumer closed us at a
                    # yield, a frame's or a heartbeat's. Either way the
                    # client is gone, and the segment acts on it
                    # (`StudioRunner._locked`). Without unwinding here,
                    # aclosing(source) would find the anext still running
                    # and raise.
                    ending.client_gone = True
                    if next_frame is not None:
                        try:
                            await _unwind(next_frame)
                        except StopAsyncIteration:
                            pass
                        except Exception:
                            logger.exception("studio: a segment failed as its client left")
                    raise
        finally:
            if cancel_wait is not None:
                await _unwind(cancel_wait)
            if conversation is not None:
                if conversation.cancel_event is armed_cancel:
                    conversation.cancel_event = None
                    conversation.cancel_reason = None
                if conversation.switch_stop is switch_stop:
                    conversation.switch_stop = None
            await flush_telemetry()


def _keep_turn_records(conversation: ConversationState, event: Event, store: StudioStore) -> None:
    """Keep the plan records, search readings, spec blocks and applied chip
    `event` writes to the turn's session state, for
    `ConversationState.close_turn` to store with the turn. Each key holds the
    turn's whole value, so the latest write replaces. The blocks stay in
    process (`turn_specs`) until the turn closes, converted as `load` reads
    them (`StudioStore.adk_convert_specs`), since a turn the previous
    release opened holds blocks in the form that release stored. They are
    converted until a conversion changes nothing (`turn_specs_settled`). A
    turn this release opened, whose graph writes the current form, needs no
    conversion."""
    delta = event.actions.state_delta
    if ORCHESTRATOR_PLAN_RECORDS_KEY in delta:
        conversation.turn_plans = list(delta[ORCHESTRATOR_PLAN_RECORDS_KEY])
    if ORCHESTRATOR_SEARCHES_KEY in delta:
        conversation.turn_readings = list(delta[ORCHESTRATOR_SEARCHES_KEY])
    if ORCHESTRATOR_SPECS_KEY in delta:
        specs = dict(delta[ORCHESTRATOR_SPECS_KEY])
        if not conversation.turn_specs_settled:
            converted = store.adk_convert_specs(conversation.session_id, specs)
            conversation.turn_specs_settled = converted is specs
            specs = converted
        conversation.turn_specs = specs
    if ORCHESTRATOR_APPLIED_KEY in delta:
        conversation.turn_applied_choice = dict(delta[ORCHESTRATOR_APPLIED_KEY])


async def _unwind(pending: asyncio.Future[Any]) -> None:
    """Cancel `pending` once and WAIT for it: a bare `.cancel()` leaves the
    generator mid-frame and `aclosing` then raises "asynchronous generator
    is already running".

    The wait is shielded. Awaiting a task forwards every cancellation of the
    waiting task to it, and Starlette's anyio task group re-cancels the
    response task on every loop iteration once the response is cancelled.
    Unshielded, that re-cancels `pending`'s own cleanup (the session
    service's rollback and connection close, `StudioStore.release`) at
    each of its awaits, and a SQLAlchemy connection torn down twice over
    asyncpg leaves a task that never finishes."""
    pending.cancel()
    with anyio.CancelScope(shield=True), suppress(asyncio.CancelledError):
        await pending


@asynccontextmanager
async def _closing(source: AsyncGenerator[Any], what: str) -> AsyncIterator[AsyncGenerator[Any]]:
    """`aclosing(source)`, except that a close that raises while another
    exception is leaving the block (most often the stream's cancellation)
    is logged and dropped, so the exception in flight goes on and the stream
    still unwinds to its ending frame. A close that raises on a clean exit
    raises."""
    try:
        yield source
    except BaseException:
        try:
            await source.aclose()
        except Exception:
            logger.exception("studio: closing %s failed while the stream was ending", what)
        raise
    await source.aclose()


# The frames `StudioRunner._checkpointed` durably saves the conversation
# before yielding: the two actionable pauses and the turn's own terminal
# success. `message.complete` is not in this set. It precedes `done` on
# the SAME segment when a turn ends outright, and checkpointing twice one
# frame apart buys nothing (`ElicitationRequest`'s and `PlanProposed`'s own
# frame carries no `message.complete` of its own).
_CHECKPOINT_BEFORE_EVENTS = frozenset({"plan.proposed", "elicitation.request", "done"})


# ---------------------------------------------------------------------------
# The runner wrapper and routes
# ---------------------------------------------------------------------------


class StudioRunner:
    """Owns the ADK Runner, the executor and surface recorders, and the store.

    `create_workflow` is called once with the recording executor and the
    recording surface already wrapped around the injected ones, so the
    workflow the Runner drives is the same object across turns while every
    executor call and every plan-step tool call is observed. `tool_surface`
    is `None` for a caller that has no `ToolSurface` to give it (a test that
    does not exercise plan steps); `create_workflow` gets `None` through in
    that case, and the graph's `execute` node is what refuses loudly if it
    is ever reached. `early_mints` is the workflow's own
    (`harness.early_mint`): a segment's response waits for the calls of the
    card it offered before it closes, while the request still has CPU
    (`_settling`).
    """

    def __init__(
        self,
        *,
        create_workflow: Any,
        executor: Executor,
        session_service: Any,
        store: StudioStore,
        tool_surface: ToolSurface | None = None,
        app_name: str | None = None,
        edit_model: Any | None = None,
        provenance: ProvenanceStore | None = None,
        early_mints: EarlyMints | None = None,
    ) -> None:
        self.edit_model = edit_model
        self.early_mints = early_mints
        # `AGENT_NAME` is the fixed name every `create_root_workflow` call
        # gives its `Workflow`, so it matches `workflow.name` in every real
        # caller. `store` reads an open turn's spec blocks from the per-turn
        # sessions this runner creates, so its caller builds it over the same
        # `session_service` and app name. `_RecordingExecutor` binds it too,
        # to keep a create's write token (`harness.studio_translate`) on the
        # same conversations the rest of this runner reads and writes.
        resolved_app_name = app_name or AGENT_NAME
        self.store = store
        self.recorder = _RecordingExecutor(executor, store=self.store)
        self.surface_recorder = (
            _RecordingSurface(tool_surface, provenance=provenance)
            if tool_surface is not None
            else None
        )
        workflow = create_workflow(self.recorder, self.surface_recorder)
        # One `App` for the whole graph. The agents' models cache their
        # static prompt prefixes themselves, across sessions
        # (`harness.prefix_cache`).
        app = App(name=resolved_app_name, root_agent=workflow)
        self.runner = Runner(app=app, session_service=session_service)

    async def aclose(self) -> None:
        """Release the injected executor's resources, through the recorder
        wrapping it. Called once, from `harness.server._lifespan`'s
        shutdown. The server's executor is `McpExecutor`, which opens one
        long-lived presentation-mcp client per conversation
        (`harness.mcp_client.McpExecutor._presentation_client`) and needs
        exactly this call to give them back; nothing else on the served path
        closes it."""
        await self.recorder.aclose()

    async def _open_session(
        self,
        *,
        user_id: str,
        conversation_id: str,
        session_id: str,
        state: dict[str, Any],
    ) -> bool:
        """Create the per-turn ADK session (`{cid}#t{n}`) of the turn the
        `/turn` route is about to claim, before its claim save. True when it
        created the session.

        A session already under that id is met in one of two ways. When
        ADK's ledger session moved since the load, an instance still running
        on ADK sessions may have opened this turn, so nothing is deleted and
        `ConversationStale` asks the route to answer a retryable 409 with
        nothing claimed: the retry's load re-imports what that instance
        wrote (`harness.adk_import.reimport_on_access`) and meets its live
        lease. Otherwise the session is a leftover of a turn the tables never
        recorded (a request of this release whose claim lost or crashed, a
        conversation served as absent whose ADK turn sessions remain, or a
        `--prefer-adk` copy whose `turn_index` is below the turns the tables
        recorded), and False asks the route to replace it once its claim
        has won (`_replace_leftover_session`). A racing request of this
        release may have created it, and only the winning claim may delete
        it."""
        session_service = self.runner.session_service
        try:
            await session_service.create_session(
                app_name=self.runner.app_name,
                user_id=user_id,
                session_id=session_id,
                state=state,
            )
        except AlreadyExistsError:
            if await self.store.adk_moved(user_id, conversation_id):
                raise ConversationStale from None
            return False
        return True

    async def _replace_leftover_session(
        self, *, user_id: str, session_id: str, state: dict[str, Any]
    ) -> None:
        """Delete the leftover per-turn session `_open_session` met and create
        it fresh, for a turn whose claim has won."""
        logger.warning(
            "replacing leftover per-turn session %s: the turn it belonged to was never recorded",
            session_id,
        )
        session_service = self.runner.session_service
        await session_service.delete_session(
            app_name=self.runner.app_name, user_id=user_id, session_id=session_id
        )
        await session_service.create_session(
            app_name=self.runner.app_name, user_id=user_id, session_id=session_id, state=state
        )

    async def _frames(
        self,
        *,
        principal: str,
        conversation: ConversationState,
        message: types.Content,
        approving: PlanProposed | None = None,
    ) -> AsyncGenerator[Any]:
        """Run one segment (fresh turn or resume) and yield wire frames.

        `approving` is the plan card a first answer to its interrupt approves,
        which `plan.approved` states once the graph takes the approval
        (`_Translator`).

        Failures are serialised into the stream as a stream-fatal `error`
        frame. SSE headers are already on the wire, so an exception can't
        become an HTTP status. The exception is logged with the trace id;
        it is not re-raised into the transport. A lost segment or an
        unreachable database (`SegmentLost`, `DatabaseUnavailable`) is
        re-raised, and `_checkpointed` ends the stream on it.
        """
        translator = _Translator(
            conversation=conversation,
            recorder=self.recorder,
            surface_recorder=self.surface_recorder,
            approving=approving,
        )
        try:
            async with aclosing(
                self.runner.run_async(
                    user_id=principal,
                    session_id=conversation.session_id,
                    new_message=message,
                    # The agent's text and thoughts stream as they are generated.
                    run_config=RunConfig(streaming_mode=StreamingMode.SSE),
                )
            ) as events:
                async for event in events:
                    _keep_turn_records(conversation, event, self.store)
                    for frame in await translator.translate(event):
                        yield frame
            for frame in await translator.finish():
                yield frame
        except (SegmentLost, DatabaseUnavailable):
            # A card's write fenced out by another instance's takeover, or one
            # that could not reach the database before the lease's deadline
            # (`StudioStore.change_canvas`), ends the stream where every save
            # that loses the segment does (`StudioRunner._checkpointed`).
            raise
        except Exception as exc:
            logger.exception("studio turn failed (trace_id=%s)", conversation.trace_id)
            yield _stream_fatal(conversation, exc)
        finally:
            # A step the executor finished after the last translated event
            # belongs to this segment. Emptying the queue here keeps a failed
            # or abandoned turn's steps and cards out of the next turn.
            self.recorder.drain(current_conversation_id.get())
            if self.surface_recorder is not None:
                self.surface_recorder.drain(current_conversation_id.get())

    async def _settling(self, stream: AsyncGenerator[str]) -> AsyncGenerator[str]:
        """`stream`, then the wait for the early calls of the card it offered
        (`harness.early_mint.EarlyMints.settle`).

        The wait starts once `stream` has ended, so the conversation lock and
        the segment are released and an answer to the card is taken at once.
        It sends nothing and only keeps the response, and so the request's
        CPU, open. A client that leaves first ends it. `stream` flushed the
        telemetry as it ended, so a wait that ran flushes again, exporting
        the spans of the calls it waited for.
        """
        async with aclosing(stream):
            async for chunk in stream:
                yield chunk
        if self.early_mints is not None and await self.early_mints.settle():
            await flush_telemetry()

    async def _heartbeat(
        self,
        principal: str,
        conversation_id: str,
        segment_id: str,
        conversation: ConversationState,
    ) -> None:
        """Renew `segment_id`'s TTL every `HEARTBEAT_INTERVAL_S` for as long
        as this stream runs. A `renew` that reports `"cancelled"` (a Stop,
        from any instance) or `"lost"` (another instance's claim outlived
        this one's TTL) ends the stream by arming `conversation.cancel_event`
        with `"cancelled"` or `"segment_lost"`. `_sse_stream`'s cancel branch
        reads it, one mechanism for every kind of mid-stream stop.

        On a stream running a plan-card switch whose Stop has not come yet
        (`conversation.switch_stop` armed and unset), the renewal clears the
        flag it reports (`StudioStore.renew`'s `consume_cancel`) and the
        heartbeat keeps watching: that first Stop only stops the switch
        (`_sse_stream`), and a second one, from any instance, sets the flag
        again and ends the stream.

        A `renew` call that raises is retried every interval until the
        lease's deadline (`StudioStore.lease_deadline`, on this instance's own
        clock, anchored just before the read or renewal that set the lease's
        expiry), then ends the stream with the reason
        `"database_unavailable"`. An instance this deaf to its own renewal
        cannot be trusted to still own the segment, even though nothing else
        has claimed it yet. The turn stays as stored, its lease lapses by its
        TTL, and a retry resumes it or closes it as
        `ConversationState.begin_turn` does. Measuring the deadline on one
        clock keeps the gap between this clock and the database's, which
        decides the lease, out of it. The store's writes stop retrying at the
        same deadline.
        """
        while True:
            await asyncio.sleep(HEARTBEAT_INTERVAL_S)
            stop = conversation.switch_stop
            switching = stop is not None and not stop.is_set()
            try:
                result = await self.store.renew(
                    principal, conversation_id, segment_id, consume_cancel=switching
                )
            except Exception:
                logger.exception(
                    "studio segment heartbeat: conversation %s segment %s renewal failed",
                    conversation_id,
                    segment_id,
                )
                segment = conversation.segment
                if segment is None or segment.id != segment_id:
                    return
                if self.store.now() < self.store.lease_deadline(principal, conversation_id):
                    continue
                _end_stream(conversation, "database_unavailable")
                return
            if result == "ok":
                continue
            _end_stream(conversation, "cancelled" if result == "cancelled" else "segment_lost")
            if result == "cancelled" and switching:
                continue
            return

    async def _checkpointed(
        self,
        principal: str,
        conversation_id: str,
        segment_id: str,
        conversation: ConversationState,
        frames: AsyncGenerator[Any],
        *,
        ending: _StreamEnding,
    ) -> AsyncGenerator[Any]:
        """Relay `frames`, each once its `seq` lies inside the segment's
        reserved frame numbers (`StudioStore.reserve_frames`), checkpointing
        durably before every actionable pause (`plan.proposed`,
        `elicitation.request`) and before `done`. The client never sees a
        frame that says "paused" or "done" before the store agrees. `done`'s frame is rebuilt to carry the canvas
        the checkpoint read back from the store, so a `/composition` edit
        another instance made meanwhile reaches the client on it. A
        checkpoint that fails (the segment was taken over, or the write
        itself failed) yields a stream-fatal error frame instead of the
        pause or terminal frame it would have carried, and ends the stream
        there. Once a `plan.approved` frame is sent, the interrupt it answers
        records it (`ConversationState.mark_approved_sent`) and is
        checkpointed. A lost segment, or a database unreachable until the
        lease's deadline, ends the stream the same way, and any other
        failure there is logged alone. A card's canvas write in `frames`
        that loses the segment, or cannot reach the database, ends `frames`
        with the same frame, which is relayed, its number reserved first,
        like any other; when that reservation fails too, the card write's
        frame goes out alone, and the reservation's failure is logged with
        it. `frames` is closed whenever this stops relaying it (`_closing`).
        This is the one interception point for every route this runs under
        (`/turn`, its edit-card branch, and `/interrupt`).

        A stream-fatal error frame, from `frames` or built here, goes out
        after a `failed` frame for every Activity line the stream left open
        (`_StreamEnding.open_steps`), since no step will report on this
        stream again (a resumed segment reports the same `step_id` again,
        and the client's line-patching keys on it), renumbered past them so
        `seq` still rises along the stream. Every ending frame is offered
        to `ending` with its cause (`_StreamEnding.offer`): `done` and
        `paused` once their checkpoint landed, the loss (`stream_loss`) for
        a save that lost the segment, and otherwise the error frame's own
        code, for a save that failed (`transient_unavailable`, the turn
        recorded `state_not_saved`) and for a stream-fatal frame of
        `frames`."""

        def failed(what: str, exc: Exception) -> tuple[ErrorFrame, str]:
            _log_save_failure(what, conversation, segment_id, exc)
            frame = _checkpoint_failed_frame(conversation, exc)
            status = (
                stream_loss(exc)
                if isinstance(exc, (SegmentLost, DatabaseUnavailable))
                else frame.error.error.value
            )
            return frame, status

        def ending_with(frame: ErrorFrame, status: str) -> list[Any]:
            return _ending_frames(conversation, ending, frame, status)

        card_write_lost: tuple[ErrorFrame, str] | None = None

        async def relayed() -> AsyncGenerator[Any]:
            nonlocal card_write_lost
            async with _closing(frames, "the segment's frames"):
                try:
                    async for frame in frames:
                        yield frame
                except (SegmentLost, DatabaseUnavailable) as exc:
                    card_write_lost = failed("card write", exc)
                    yield card_write_lost[0]

        async with _closing(relayed(), "the relayed frames") as source:
            async for frame in source:
                lost_card = (
                    card_write_lost
                    if card_write_lost is not None and frame is card_write_lost[0]
                    else None
                )
                try:
                    await self.store.reserve_frames(principal, conversation_id, segment_id)
                except Exception as exc:
                    if lost_card is not None:
                        # The card write that lost the segment has logged its
                        # loss and built its frame already.
                        _log_save_failure("frame reservation", conversation, segment_id, exc)
                        batch = ending_with(*lost_card)
                    else:
                        batch = ending_with(*failed("frame reservation", exc))
                    for outgoing in batch:
                        yield outgoing
                    return
                event = getattr(frame, "event", None)
                outgoing = frame
                if event in _CHECKPOINT_BEFORE_EVENTS:
                    conversation.drop_resuming(segment_id)
                    try:
                        await self.store.checkpoint(principal, conversation_id, segment_id)
                    except Exception as exc:
                        for outgoing in ending_with(*failed("checkpoint", exc)):
                            yield outgoing
                        return
                    if event == "done":
                        outgoing = frame.model_copy(
                            update={"composition": conversation.composition}
                        )
                    ending.offer(outgoing, "done" if event == "done" else "paused")
                if isinstance(frame, ErrorFrame):
                    status = lost_card[1] if lost_card is not None else frame.error.error.value
                    for outgoing in ending_with(frame, status):
                        yield outgoing
                else:
                    yield outgoing
                if event == "plan.approved":
                    # Saved once sent, so a retry of the same answer after this
                    # segment dies does not send it again. A crash before the
                    # save sends it twice, which beats never.
                    conversation.mark_approved_sent(segment_id)
                    try:
                        await self.store.checkpoint(principal, conversation_id, segment_id)
                    except Exception as exc:
                        if isinstance(exc, (SegmentLost, DatabaseUnavailable)):
                            for outgoing in ending_with(
                                *failed("checkpoint after plan.approved", exc)
                            ):
                                yield outgoing
                            return
                        _log_save_failure(
                            "checkpoint after plan.approved", conversation, segment_id, exc
                        )

    async def _locked(
        self,
        principal: str,
        conversation_id: str,
        conversation: ConversationState,
        segment_id: str,
        frames: AsyncGenerator[Any],
        ending: _StreamEnding | None = None,
        *,
        hold: _LockHold,
    ) -> AsyncGenerator[Any]:
        """Stream frames, checkpointing before any that report success
        (`_checkpointed`), then release the segment and let go of the
        conversation lock through `hold`, the route's hold on it.

        The ROUTE acquires the lock (eagerly, before touching turn state), so
        ownership is settled while the request handler still controls the
        response. A lazily-acquired lock let two requests pass the 409
        guard and clobber each other's turn state and cancel event.

        A heartbeat task renews `segment_id` for as long as this generator is
        live, so the segment's lease outlives a single checkpoint's TTL push;
        it is cancelled and awaited before release, so it never races the
        release's own write (`StudioStore.release`). On every path out, the
        heartbeat is stopped, then the segment released (with this
        segment's resuming pending entries), then the local lock, so a
        heartbeat that ends with a non-cancel exception still reaches both
        releases; `_unwind` re-raises that exception once they have run. The
        release itself never raises on a DB error (`StudioStore.release`),
        and releasing the lock would not depend on it if it did.

        Every change a Stop, a loss or a disconnect makes to the turn is made
        here, in this generator's own task, before the release saves it.
        `_sse_stream` only records what it read (`_StreamEnding`). Each frame
        is pulled from `_checkpointed` in a task of its own, raced against a
        Stop or loss the stream reads (`_StreamEnding.stop_read`). One that
        still applies (`_StreamEnding.stop_applies`) cancels that task and
        nothing more is relayed, even from frames that swallow the
        cancellation or raise in its place. A cancellation of this generator
        is always the client's. A Stop then closes the turn as `cancelled` and
        drops its pending interrupts. The frames that end the stream, a
        `failed` frame for every open Activity line and then the `cancelled`
        frame or a lost segment's `transient_unavailable` frame, are numbered
        (reserved first for a Stop) before the release, which saves the turn
        as the Stop left it, and are sent after it. A Stop read while the
        release writes is applied the same way once it has, with a save of its
        own (`_stop_frames`), and the turn stays as released after a loss read
        then. A client that leaves before the stream sent an ending frame
        closes a running turn as `cancelled` before the release. A turn
        paused for an answer stays open.

        The segment's span ends here, after the last frame this sent
        (`harness.studio_state.span_level`'s table). Its status is the cause
        of the last ending frame the stream sent (`_StreamEnding`). A stream
        that sent none ended because its frames ran out (`failed`) or because
        it was closed or cancelled first (`disconnected`). Its
        `stored` is the turn as this instance last saved it
        (`StudioStore.stored_turn`), read after the release, which runs to
        the end even when this generator is cancelled while it waits.
        """
        ending = ending or _StreamEnding()
        ran_out = False
        released = False
        stop_applied = False

        async def release() -> None:
            nonlocal released
            released = True
            await self.store.release(principal, conversation_id, segment_id)

        heartbeat = asyncio.ensure_future(
            self._heartbeat(principal, conversation_id, segment_id, conversation)
        )

        def stops() -> bool:
            return ending.stop_applies(conversation)

        # Set once the relay broke on a Stop or loss, and held, so it
        # outlasts frames that record the turn as they unwind. Settled once
        # the frames are unwound.
        stopped = False

        reading: asyncio.Future[Any] | None = None
        try:
            async with _closing(
                self._checkpointed(
                    principal, conversation_id, segment_id, conversation, frames, ending=ending
                ),
                "the segment's frames",
            ) as checked:
                pulling: asyncio.Future[Any] | None = None
                try:
                    while True:
                        if ending.stop_spent():
                            # The stream sent its ending frame first. With
                            # no stream to act on the Stop or loss, `/cancel`
                            # takes its own path.
                            ending.drop_stop()
                            conversation.cancel_event = None
                        if stops():
                            stopped = True
                            break
                        pulling = asyncio.ensure_future(anext(checked))
                        while not pulling.done():
                            if ending.stop is not None:
                                # Read while the frames end on their own, so
                                # they run on to their ending frame.
                                await asyncio.wait({pulling})
                                break
                            if reading is None:
                                reading = asyncio.ensure_future(ending.stop_read.wait())
                            await asyncio.wait(
                                {pulling, reading}, return_when=asyncio.FIRST_COMPLETED
                            )
                            if reading.done():
                                reading = None
                            if not pulling.done() and stops():
                                stopped = True
                                break
                        if not pulling.done():
                            # A Stop or loss ends the frames where they are.
                            break
                        try:
                            item = pulling.result()
                        except StopAsyncIteration:
                            ran_out = True
                            break
                        except asyncio.CancelledError as exc:
                            task = asyncio.current_task()
                            if task is not None and task.cancelling():
                                raise
                            # The frames failed with a cancellation of their
                            # own, this segment's task uncancelled.
                            pulling = None
                            ran_out = True
                            closing_failed = await self._frames_failed(
                                principal, conversation_id, segment_id, conversation, ending, exc
                            )
                            for frame in closing_failed:
                                yield frame
                            break
                        pulling = None
                        yield item
                finally:
                    if pulling is not None and not pulling.done():
                        # Cancelled in a task of its own, which keeps the
                        # cancellation inside the frames. What they yield or
                        # raise from here stays unrelayed.
                        cause = "a Stop" if stopped else "the client leaving"
                        try:
                            await _unwind(pulling)
                        except StopAsyncIteration:
                            pass
                        except Exception:
                            logger.exception(
                                "studio: conversation %s's frames failed as %s ended them",
                                conversation_id,
                                cause,
                            )
                    # An offered ending frame left unsent is withdrawn.
                    ending.withdraw()
                    if reading is not None:
                        await _unwind(reading)
                    stopped = stopped or stops()
            await _unwind(heartbeat)
            closing: list[Any] = []
            if stopped:
                stop_applied = True
                closing = await self._stop_frames(
                    principal, conversation_id, segment_id, conversation, ending
                )
            await release()
            if not stop_applied and ending.stop == "cancelled" and stops():
                stop_applied = True
                closing = await self._stop_frames(
                    principal, conversation_id, segment_id, conversation, ending, released=True
                )
            # From here a Stop finds no stream to end: `/cancel` takes its
            # own path.
            conversation.cancel_event = None
            for frame in closing:
                yield frame
        finally:
            try:
                if not released:
                    try:
                        await _unwind(heartbeat)
                    finally:
                        if not stop_applied and ending.stop == "cancelled" and stopped:
                            # A Stop the segment's exit kept from its own
                            # path still closes the turn.
                            _apply_stop(conversation)
                        elif (
                            ending.client_gone
                            and ending.status is None
                            and not conversation.awaiting_answer
                        ):
                            conversation.close_turn(
                                final_text=conversation.turn_final_text,
                                error=ErrorCode.CANCELLED.value,
                            )
                        await release()
            finally:
                try:
                    # The segment ends here on this instance, and its span
                    # with it. The stream's end exports the span
                    # (`_sse_stream`).
                    conversation.end_segment_span(
                        ending.status or ("failed" if ran_out else "disconnected"),
                        stored=self._stored_turn(principal, conversation_id),
                    )
                finally:
                    hold.release()

    async def _frames_failed(
        self,
        principal: str,
        conversation_id: str,
        segment_id: str,
        conversation: ConversationState,
        ending: _StreamEnding,
        exc: asyncio.CancelledError,
    ) -> list[Any]:
        """The frames that end a stream whose frames failed with a cancellation
        of their own, the segment's task uncancelled (an awaited future
        cancelled under them, say). As with any ending (`_StreamEnding`), after an
        ending frame went out (`done`, a stream-fatal frame or a pause), it
        sends nothing, keeps that frame's status and logs a warning. Otherwise
        it logs an error and the turn closes with a stream-fatal frame
        (`_stream_fatal`), after a `failed` frame for every open Activity line,
        its number reserved (logged when that fails). The span then ends
        `failed`, as for frames that ran out."""
        message = (
            "studio: conversation %s's frames ended cancelled, the segment's task uncancelled "
            "(trace_id=%s)"
        )
        if ending.status is not None:
            logger.warning(message, conversation_id, conversation.trace_id, exc_info=exc)
            return []
        logger.error(message, conversation_id, conversation.trace_id, exc_info=exc)
        frames = _ending_frames(
            conversation,
            ending,
            _stream_fatal(conversation, RuntimeError("the segment's frames were cancelled")),
            "failed",
        )
        try:
            await self.store.reserve_frames(principal, conversation_id, segment_id)
        except Exception as reservation:
            _log_save_failure("frame reservation", conversation, segment_id, reservation)
        return frames

    async def _stop_frames(
        self,
        principal: str,
        conversation_id: str,
        segment_id: str,
        conversation: ConversationState,
        ending: _StreamEnding,
        *,
        released: bool = False,
    ) -> list[Any]:
        """Apply the Stop or loss the stream read (`ending.stop`) to the
        turn, before the release saves it, and build the frames that end the
        stream: a `failed` frame for every open Activity line, then the
        `cancelled` frame, the turn closed and its pending interrupts
        dropped, or a lost segment's `transient_unavailable` frame, the turn
        left as stored. A Stop's frames have their numbers reserved; when
        that fails, they go out all the same, the failure logged. A Stop the
        stream read while the release wrote (`released`) is applied after
        it, as `/cancel` applies one to a segment that has ended: its frames
        go out unreserved, and the closed turn, with their numbers, is saved
        on its own, the failure of that save logged."""
        if ending.stop != "cancelled":
            return _ending_frames(
                conversation,
                ending,
                segment_lost_frame(conversation),
                ending.stop or "segment_lost",
            )
        _apply_stop(conversation)
        stopped = failure_sentence(ErrorCode.CANCELLED, RetryMode.DO_NOT_RETRY)
        frames = _ending_frames(
            conversation,
            ending,
            ErrorFrame(
                event="error",
                seq=conversation.next_seq(),
                error=ToolError(
                    error=ErrorCode.CANCELLED, retry=RetryMode.DO_NOT_RETRY, detail=stopped
                ),
            ),
            "cancelled",
        )
        if released:
            try:
                await self.store.save(principal, conversation_id)
            except Exception:
                # The span names the `cancelled` frame sent, and its stored
                # turn stays as the release saved it.
                logger.exception(
                    "studio: a Stop read while conversation %s released its segment was not saved",
                    conversation_id,
                )
        else:
            try:
                await self.store.reserve_frames(principal, conversation_id, segment_id)
            except Exception as exc:
                _log_save_failure("frame reservation", conversation, segment_id, exc)
        return frames

    def _stored_turn(self, principal: str, conversation_id: str) -> str | None:
        """`StudioStore.stored_turn`, or `None` (logged) when it fails, so
        a failed read cannot keep the segment's span from ending."""
        try:
            return self.store.stored_turn(principal, conversation_id)
        except Exception:
            logger.exception(
                "studio: could not read how conversation %s was stored", conversation_id
            )
            return None

    def _segment_stream(
        self,
        claim: _Claim,
        conversation: ConversationState,
        frames: AsyncGenerator[Any],
        *,
        switch: bool = False,
        settle: bool = False,
    ) -> _ClaimedStream:
        """The SSE response of `claim`'s segment: `frames` under `_locked`,
        which lets go of the claim's hold, relayed by `_sse_stream`, both
        sharing one `_StreamEnding`, then, with `settle`, the wait for the
        early calls of the card it offered (`_settling`)."""
        assert claim.segment_id is not None
        ending = _StreamEnding()
        stream = _sse_stream(
            self._locked(
                claim.principal,
                claim.conversation_id,
                conversation,
                claim.segment_id,
                frames,
                ending,
                hold=claim.hold,
            ),
            conversation=conversation,
            switch=switch,
            ending=ending,
        )
        return _ClaimedStream(self._settling(stream) if settle else stream, claim=claim)


@dataclass
class _StreamEnding:
    """How one segment's stream ends, shared by `_sse_stream`, which reads
    the client and the cancel event, and `StudioRunner._locked`, which
    changes the turn in its own task.

    `status` is the cause of the last ending frame the stream sent, in the
    turn record's vocabulary (`harness.studio_state.span_level`'s table),
    `None` while it sent none. Each ending frame is offered with its cause
    before it is relayed (`offer`), and `_sse_stream` sets `status` as that
    frame goes out (`sent`). It is set on sending, since `_sse_stream` can
    be closed at a heartbeat's yield while a frame `_locked` yielded waits
    unread. `stop` is a Stop or a loss the stream read (`read_stop`), which
    `_locked` acts on, and `client_gone` a client that left. `open_steps`
    are the Activity lines the stream has sent open."""

    status: str | None = None
    stop: Literal["cancelled"] | StreamLoss | None = None
    client_gone: bool = False
    open_steps: _OpenSteps = field(default_factory=_OpenSteps)
    stop_read: asyncio.Event = field(default_factory=asyncio.Event)
    _offered: tuple[Any, str] | None = None

    def offer(self, frame: Any, status: str) -> None:
        """Name `status` as the cause `frame` ends the stream with."""
        self._offered = (frame, status)

    def sent(self, frame: Any) -> None:
        """Record that `frame` went out; the offered ending frame sets
        `status`."""
        if self._offered is not None and self._offered[0] is frame:
            self.status = self._offered[1]
            self._offered = None

    def withdraw(self) -> None:
        """Withdraw an offered ending frame that will not be relayed. An
        offered loss frame stays. The segment is another instance's to end,
        and the frame keeps blocking a Stop."""
        if self._offered is not None and self._offered[1] not in STREAM_LOSSES:
            self._offered = None

    @property
    def final(self) -> bool:
        """Whether the stream sent `done` or a stream-fatal frame."""
        return self.status not in (None, "paused")

    def read_stop(self, stop: Literal["cancelled"] | StreamLoss) -> None:
        """Record a Stop or a loss the stream read, for `_locked`."""
        self.stop = stop
        self.stop_read.set()

    def stop_spent(self) -> bool:
        """Whether the Stop or loss read came too late: the stream has sent
        `done` or a stream-fatal frame, or a pause frame for a loss."""
        return self.stop is not None and (
            self.final or (self.status == "paused" and self.stop != "cancelled")
        )

    def stop_applies(self, conversation: ConversationState) -> bool:
        """Whether the Stop or loss read ends the segment now: one was read,
        it came in time (`stop_spent`), no offered ending frame waits to go
        out, and the segment has not recorded the turn
        (`ConversationState.segment_closed_turn`)."""
        return (
            self.stop is not None
            and self._offered is None
            and not conversation.segment_closed_turn
            and not self.stop_spent()
        )

    def drop_stop(self) -> None:
        """Forget a Stop or loss that came too late."""
        self.stop = None
        self.stop_read.clear()


def _apply_stop(conversation: ConversationState) -> None:
    """A Stop's effect on the turn: closed as `cancelled`, with no pending
    interrupt left to answer."""
    conversation.close_turn(
        final_text=conversation.turn_final_text, error=ErrorCode.CANCELLED.value
    )
    conversation.pending.clear()


def _ending_frames(
    conversation: ConversationState, ending: _StreamEnding, frame: ErrorFrame, status: str
) -> list[Any]:
    """The frames that end a stream on the stream-fatal `frame`: a `failed`
    frame for every Activity line the stream left open
    (`_StreamEnding.open_steps`), since no step will report on this stream
    again (a resumed segment reports the same `step_id` again, and the
    client's line-patching keys on it), then `frame`, renumbered past them
    so `seq` still rises, offered to `ending` with `status`."""
    closing = ending.open_steps.close(conversation, detail=frame.error.detail)
    if closing:
        frame = frame.model_copy(update={"seq": conversation.next_seq()})
    ending.offer(frame, status)
    return [*closing, frame]


def _log_save_failure(
    what: str, conversation: ConversationState, segment_id: str, exc: Exception
) -> None:
    """Log a save `_checkpointed` made that failed: a lost segment or a
    database unreachable until the lease's deadline (every transient fault
    the store's retries did not outlast) as a warning, which ends the
    stream as the heartbeat does, and anything else, a fault no retry
    mends such as a lock timeout, a failed login or a bug, as an error with
    its traceback."""
    if isinstance(exc, (SegmentLost, DatabaseUnavailable)):
        logger.warning(
            "studio %s ended the stream: conversation %s segment %s (trace_id=%s): %s",
            what,
            conversation.conversation_id,
            segment_id,
            conversation.trace_id,
            exc,
        )
    else:
        logger.exception(
            "studio %s failed: conversation %s segment %s (trace_id=%s)",
            what,
            conversation.conversation_id,
            segment_id,
            conversation.trace_id,
            exc_info=exc,
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
    """A `/turn` body past `MAX_TURN_REQUEST_BYTES`. An `HTTPException`, since
    FastAPI re-raises those from its body read and turns any other exception
    there into a 400."""

    def __init__(self, size: str) -> None:
        super().__init__(
            status_code=413,
            detail=f"the request body is {size} bytes, past the {MAX_TURN_REQUEST_BYTES:,}-byte cap",
        )


class _BoundedRequest(Request):
    """A request whose body read stops at `MAX_TURN_REQUEST_BYTES`.

    A declared `Content-Length` past the cap is refused before anything is
    read; a body sent without one is refused as soon as it crosses the cap.
    """

    async def body(self) -> bytes:
        if not hasattr(self, "_body"):
            declared = self.headers.get("content-length", "")
            if declared.isdigit() and int(declared) > MAX_TURN_REQUEST_BYTES:
                raise _BodyTooLarge(f"{int(declared):,}")
            chunks: list[bytes] = []
            size = 0
            async for chunk in self.stream():
                size += len(chunk)
                if size > MAX_TURN_REQUEST_BYTES:
                    raise _BodyTooLarge(f"more than {MAX_TURN_REQUEST_BYTES:,}")
                chunks.append(chunk)
            # Starlette's own cache for the body, which `Request.body` reads back.
            self._body = b"".join(chunks)
        return self._body


class _StudioRoute(APIRoute):
    """Every Studio route: a conversation `StudioStore.load` refuses as held
    (`ConversationHeld`), wherever the route meets it, answers
    `_held_response()`. A route lets go of its lock before the refusal
    leaves it, as for any other failure."""

    def get_route_handler(self) -> Callable[[Request], Coroutine[Any, Any, Response]]:
        handler = super().get_route_handler()

        async def held_answered(request: Request) -> Response:
            try:
                return await handler(request)
            except ConversationHeld:
                return _held_response()

        return held_answered


class _BoundedTurnRoute(_StudioRoute):
    """The `/turn` route, whose body is read through `_BoundedRequest`, so a
    request past the size cap is refused before FastAPI parses it. Each seed
    is checked on its own (`harness.seeds.seeds_slot`) and refused in its own
    step; this bound is what keeps an oversized request from being parsed at
    all."""

    def get_route_handler(self) -> Callable[[Request], Coroutine[Any, Any, Response]]:
        handler = super().get_route_handler()

        async def bounded(request: Request) -> Response:
            try:
                return await handler(_BoundedRequest(request.scope, request.receive))
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


def _unknown_conversation_response(conversation_id: str) -> JSONResponse:
    """The 404 every route answers with for a conversation id its store
    does not hold -- the wire has no "new conversation" signal, so this is
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


def _held_response() -> JSONResponse:
    """The 409 every route answers for a conversation `StudioStore.load`
    cannot serve until an operator acts (`ConversationHeld`). A retry
    cannot succeed, so it is `do_not_retry`. The operator's command is in
    the log line the store's `adk.reimport` wrote."""
    return _tool_error_response(
        ToolError(
            error=ErrorCode.CONSTRAINT_VIOLATION,
            retry=RetryMode.DO_NOT_RETRY,
            constraint_id="conversation_diverged",
            detail=(
                "This conversation changed in two places during an upgrade, so it cannot be "
                "continued until it is repaired."
            ),
            pointer="/conversation_id",
        ),
        status_code=409,
    )


class _LockHold:
    """One request's hold on its conversation's process-local lock. The
    request lets go through `release`, once: a later call does nothing, so
    a request that let go never releases the lock another request took
    since. `asyncio.Lock.locked()` says only that someone holds it."""

    def __init__(self, lock: asyncio.Lock) -> None:
        self.lock = lock
        self.held = True

    def release(self) -> None:
        if self.held:
            self.held = False
            self.lock.release()


async def _acquire_or_conflict(
    store: StudioStore, principal: str, conversation_id: str
) -> _LockHold | JSONResponse:
    """Take the conversation's lock BEFORE loading or touching any turn
    state.

    The check-and-acquire pair has no suspension point between them on
    the uncontended fast path, so two concurrent requests cannot both
    pass. A lazily-acquired lock let them race past the 409 guard and
    clobber each other's turn state and cancel event. The request's
    stream (`StudioRunner._locked`) lets go of the hold, or `_ClaimedStream`
    when that stream never starts; early-exit route paths let go
    explicitly. Returns the request's hold on the lock, not a
    conversation, since a lock exists (and can be contended) before this
    store has ever loaded or created the conversation it names.
    """
    hold = await _try_hold(store, principal, conversation_id)
    if hold is None:
        return _streaming_conflict_response("another request is streaming on this conversation")
    return hold


async def _try_hold(store: StudioStore, principal: str, conversation_id: str) -> _LockHold | None:
    """The conversation's lock, taken at once, or `None` when a request
    holds it. A free lock is taken with no suspension point between the
    check and the acquire."""
    lock = store.lock_for(principal, conversation_id)
    if lock.locked():
        return None
    await lock.acquire()
    return _LockHold(lock)


def _turn_session_state(conversation: ConversationState, body: TurnRequest) -> dict[str, Any]:
    """The state a `/turn`'s per-turn ADK session starts with: the surface,
    the canvas and conversation digests, the earlier turns' readings, the
    posted seeds and the chip to apply."""
    focus = (
        body.component_ref.component_id
        if body.component_ref is not None
        and body.component_ref.intent is ComponentRefIntent.FOLLOW_UP
        else None
    )
    # The agent reads the canvas through `recall` and starts from counts.
    digest = conversation.canvas_summary_state(focus=focus)
    state: dict[str, Any] = {"surface": body.surface.value}
    if digest:
        state["canvas_digest"] = json.dumps(digest)
    history = conversation.conversation_digest_state()
    if history:
        state["conversation_digest"] = json.dumps(history)
    # The orchestrator's planner renders the earlier turns' readings; their
    # blocks stay in the conversation (`current_earlier_block`).
    ledger = conversation.prior_ledger_state()
    if ledger:
        state[ORCHESTRATOR_PRIOR_LEDGER_KEY] = ledger
    if isinstance(body.seed_spec, list):
        # Each seed as the `seed` node takes it, or the refusal its step
        # fails with. No posted Spec block enters the session.
        state[SEEDS_STATE_KEY] = json.dumps(seeds_slot(body.seed_spec))
    elif body.seed_spec is not None:
        # One reference, the reserved shape: `graph.seed` reports it unused.
        state[SEEDS_STATE_KEY] = json.dumps(body.seed_spec.model_dump(mode="json"))
    if body.apply_choice is not None:
        # `graph.seed` applies the chip, and the turn skips search and planning.
        state[APPLY_CHOICE_STATE_KEY] = json.dumps(body.apply_choice.model_dump())
    return state


def _ending_status(cause: BaseException | None, *, response: bool) -> str:
    """The status a released claim's segment span ends with, by `cause`,
    the exception that ended the request, `None` when its response
    returned (span_level's table). `response` says the release runs as
    the response ends (`_ClaimedStream`), where the client's own
    disconnect is `disconnected`: `ClientDisconnect` or `OSError` from a
    send, or a return, since under ASGI spec 2.3 Starlette's response
    returns once the client's disconnect cancelled its send task. On the
    route's own path an `OSError` is the route's failure, a database
    connection reset among them. On either path the request task's own
    cancellation is `cancelled`, and anything else `failed`, a
    `CancelledError` an inner task let out among them."""
    if response and (cause is None or isinstance(cause, ClientDisconnect | OSError)):
        return "disconnected"
    task = asyncio.current_task()
    if isinstance(cause, asyncio.CancelledError) and task is not None and task.cancelling() > 0:
        return "cancelled"
    return "failed"


async def _release_claim(
    claim: _Claim, *, cause: BaseException | None = None, response: bool = False
) -> None:
    """Let go of a request whose stream never started, on every path,
    cancellation included. A claim that never committed gives up its
    working copy (`StudioStore.drop_draft`), whose span ends `not_started`.
    A claim that committed (`claim.segment_id`) drops its lease
    (`StudioStore.drop_lease`, which runs to the end under cancellation),
    and the segment's span, if one opened, ends as `_ending_status` reads
    `cause` on the route's path, or with `response` on the response's. Then the claim's hold lets go of the lock, and any span that
    ended is exported, whatever the span end raised. It does nothing once
    the hold has let go: the working copy and the lease are then another
    request's."""
    store, principal, conversation_id = claim.store, claim.principal, claim.conversation_id
    if not claim.hold.held:
        return
    ended = False
    try:
        ended = store.drop_draft(principal, conversation_id)
        if claim.segment_id is not None:
            await store.drop_lease(principal, conversation_id, claim.segment_id)
    finally:
        try:
            try:
                live = store.current(principal, conversation_id)
                if claim.segment_id is not None and live is not None:
                    ended = live.end_segment_span(_ending_status(cause, response=response)) or ended
            finally:
                claim.hold.release()
        finally:
            if ended:
                await flush_telemetry()


async def _unclaimed(claim: _Claim, response: Response) -> Response:
    """Answer `response` for a request whose claim save never committed,
    once it has let go of it (`_release_claim`)."""
    await _release_claim(claim)
    return response


@dataclass(frozen=True)
class _Claim:
    """A request's claim on its conversation: its lock hold, and the
    segment once the claim save committed it (`segment_id`). What
    `_release_claim` lets go of when the segment's stream never does."""

    store: StudioStore
    principal: str
    conversation_id: str
    hold: _LockHold
    segment_id: str | None = None


class _ClaimedStream(StreamingResponse):
    """The SSE response of a claimed segment
    (`StudioRunner._segment_stream`). Its body releases the segment and
    lets go of the request's lock hold as it ends. A body that never starts
    runs none of that: under ASGI spec 2.3 Starlette cancels the response's
    send task when the client has already left, possibly before the task
    pulls the first frame. So once the response ends, this closes the body,
    in the request task, which runs the ending of a body suspended at a
    frame, and ends a claim the body never let go of as a failure before
    the stream starts ends one (`_release_claim`): the segment never ran.
    The lease is dropped, a pending entry stays resuming under it, no turn
    line the route added is saved, and the turn stays open for the next
    `/turn` or the close job to record `abandoned`. The segment's span ends
    as `_ending_status` reads how the response ended: `disconnected` when
    the client left, `cancelled` when the request's task was cancelled, and
    `failed` otherwise."""

    body_iterator: AsyncGenerator[str]

    def __init__(self, content: AsyncGenerator[str], *, claim: _Claim) -> None:
        super().__init__(content, media_type=SSE_MEDIA_TYPE)
        self._claim = claim

    @override
    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        cause: BaseException | None = None
        try:
            await super().__call__(scope, receive, send)
        except BaseException as exc:
            cause = exc
            raise
        finally:
            try:
                await self.body_iterator.aclose()
            finally:
                await _release_claim(self._claim, cause=cause, response=True)


def create_studio_router(studio: StudioRunner, *, settings: Settings) -> APIRouter:
    router = APIRouter(prefix="/api/studio", route_class=_StudioRoute)

    async def turn(  # noqa: PLR0911 -- one distinct refusal per claim guard the route enforces
        body: TurnRequest, authorization: str | None = Header(default=None)
    ) -> Response:
        principal = principal_for(authorization, settings=settings)
        hold = await _acquire_or_conflict(studio.store, principal, body.conversation_id)
        if isinstance(hold, JSONResponse):
            return hold
        # The request's claim, which every failure before the stream starts
        # releases, with its segment once the claim save landed.
        claim = _Claim(studio.store, principal, body.conversation_id, hold)
        try:
            conversation = await studio.store.load(principal, body.conversation_id)
            if conversation is None:
                if body.component_ref is not None or body.apply_choice is not None:
                    # The client refers to state (a card to edit or to follow
                    # up on, or a chip under an answer) this store never had.
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
            if (
                body.component_ref is not None
                and body.component_ref.intent is ComponentRefIntent.FOLLOW_UP
                and body.component_ref.component_id not in conversation.components
            ):
                # A follow-up about a card this conversation never stored
                # would run as a plain turn with no card in view.
                hold.release()
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
            chip_label: str | None = None
            if (pick := body.apply_choice) is not None:
                label = _chip_label(conversation, pick)
                if isinstance(label, JSONResponse):
                    hold.release()
                    return label
                chip_label = label
            # Every change up to the claim save is made on a working copy,
            # which the save swaps into the live state once it commits. A
            # request that claims nothing leaves the live state as loaded.
            work = studio.store.draft(principal, body.conversation_id)
            abandoned = work.begin_turn()
            work.auto_add = body.auto_add
            if pick is None:
                work.turn_user_message = body.message
                work.turn_user_words = body.message
            else:
                # The history record names the option the chip counts. Search
                # would read its label as scope the user stated, so the words
                # later searches send leave it out, as a plan card's switch does.
                work.turn_user_message = f"count {chip_label}"
                work.turn_user_words = ""
                # The record's `spec_id` is the reading the `seed` node compiles,
                # set once the chip applies, so the chip's own origin stays out.
                work.turn_applied_choice = pick.model_dump(include={"choice_id", "option_id"})
            # The segment lease: claimed on the working copy before the claim
            # save below persists it, so the save's own compare-and-swap
            # covers both in one write. `SegmentBusy` means a live segment,
            # on this instance or another, already owns the conversation's
            # stream. Its own `retry_after_seconds` is the ceiling of that
            # segment's remaining TTL.
            try:
                segment_id = studio.store.claim_segment(principal, body.conversation_id, work)
            except SegmentBusy as exc:
                return await _unclaimed(
                    claim,
                    _streaming_conflict_response(
                        "another request is streaming on this conversation",
                        retry_after_seconds=exc.retry_after_seconds,
                    ),
                )
            edit = (
                body.component_ref is not None
                and body.component_ref.intent is ComponentRefIntent.EDIT
            )
            # The turn's ADK session is created before the claim save, so a
            # turn the previous release opened under the same id is met while
            # nothing is claimed (`_open_session`). The Edit Card route runs
            # no ADK session.
            session_state: dict[str, Any] = {}
            fresh = True
            if not edit:
                session_state = _turn_session_state(work, body)
                try:
                    fresh = await studio._open_session(
                        user_id=principal,
                        conversation_id=body.conversation_id,
                        session_id=work.session_id,
                        state=session_state,
                    )
                except ConversationStale:
                    # An instance still running on ADK sessions opened this
                    # turn since the load. Nothing is claimed, so the retry's
                    # load re-imports what it wrote and meets its lease.
                    return await _unclaimed(
                        claim,
                        _streaming_conflict_response(
                            "another request is streaming on this conversation"
                        ),
                    )
            # The claim: advancing turn_index (and the segment above) is
            # saved before the SSE response starts. This save is a
            # compare-and-swap against what this instance last loaded, so a
            # `ConversationStale` here means another instance's `/turn`
            # claimed a turn on this conversation since. The session this
            # request created is then a leftover, which the next claim of
            # the same turn replaces. A crash between the create and the
            # claim leaves the same leftover. A crash after the claim leaves
            # the store at the new turn_index, and the segment it claimed
            # expires by its own TTL.
            try:
                await studio.store.save(principal, body.conversation_id, work)
            except ConversationStale:
                return await _unclaimed(
                    claim,
                    _streaming_conflict_response(
                        "another request is streaming on this conversation"
                    ),
                )
            claim = replace(claim, segment_id=segment_id)
            conversation = studio.store.current(principal, body.conversation_id)
            assert conversation is not None
            if abandoned is not None:
                # The unrecorded turn this one closed as `abandoned` ends its
                # trace as its record does. The stream's end exports the
                # span.
                record_closed_turn_span(body.conversation_id, abandoned)
            # Set in the request task's context (each request gets a fresh
            # one, so nothing leaks across requests); the streamed generator
            # and every task spawned under it inherit the value, which is
            # what routes the executor's presentation client, and with it
            # the platform canvas session. Scoped by PRINCIPAL as well as
            # conversation: two principals minting the same conversation_id
            # must not share a canvas.
            current_conversation_id.set(f"{principal}/{body.conversation_id}")
            current_conversation_key.set((principal, body.conversation_id))
            current_user_message.set(conversation.turn_user_words)
            current_earlier_block.set(conversation.earlier_block)
            current_recorded_identity.set(conversation.recorded_identity)
            # This request's model-call bounds, read by every model call the
            # streamed run makes (`harness.turn_budget`).
            start_turn_budget(settings)
            if not fresh:
                # Replaced only once the claim has won, so a session another
                # instance of this release created for the same turn is
                # never deleted under its stream.
                await studio._replace_leftover_session(
                    user_id=principal, session_id=conversation.session_id, state=session_state
                )
            if edit:
                assert body.component_ref is not None
                # The Edit Card route bypasses the graph entirely: no
                # resolve, no plan, no ADK session, just the card, the
                # model's patch-or-recreate decision, and the platform write.
                # It streams under the same segment the claim save above
                # just persisted.
                return studio._segment_stream(
                    claim,
                    conversation,
                    edit_frames(
                        recorder=studio.recorder,
                        edit_model=studio.edit_model,
                        conversation=conversation,
                        component_id=body.component_ref.component_id,
                        message=body.message,
                    ),
                )
            text = body.message if chip_label is None else f"Count {chip_label}."
            message = types.Content(role="user", parts=[types.Part.from_text(text=text)])
            return studio._segment_stream(
                claim,
                conversation,
                studio._frames(principal=principal, conversation=conversation, message=message),
                settle=True,
            )
        except BaseException as exc:
            # The stream wrapper normally releases the segment and the lock.
            # A failure before the response starts streaming releases them
            # here (`_release_claim`). Before the claim save commits, the
            # working copy is given up and the live state stays as loaded.
            # After it, the lease is dropped, and the claim save already
            # covers `begin_turn`'s effects, so the turn stays open and the
            # next `/turn` closes it as `begin_turn` closes any open turn.
            await _release_claim(claim, cause=exc)
            raise

    router.add_api_route("/turn", turn, methods=["POST"], route_class_override=_BoundedTurnRoute)

    @router.post("/cancel")
    async def cancel(
        body: CancelRequest, authorization: str | None = Header(default=None)
    ) -> Response:
        """Abort the conversation's in-flight segment (the Stop button), and
        invalidate any pause it left behind. Idempotent: an unknown
        conversation, and an idle one, are both a no-op 204 (a Stop can
        reach an instance before the first `/turn` has claimed the
        conversation there at all).

        `cancel_event` only reaches a stream that is running on THIS
        instance right now: by the time a turn is paused on an ask or a
        plan, the stream that emitted the pause has already ended and reset
        the event to `None` in its `finally` (`_sse_stream`); the lock is
        released too, at the same point. So a live local stream is signalled
        directly, without the lock, exactly the state cancel exists to
        interrupt. Otherwise `request_cancel` sets the segment's durable
        `cancel` flag, whoever owns it (this instance a moment before its
        own stream arms the event, or any other instance), and returns
        whether a live segment was found. The owner's own heartbeat picks
        the flag up within `HEARTBEAT_INTERVAL_S` and ends its stream, so
        this instance answers 204 without waiting for that to happen. Only
        once neither is true (nothing streaming locally, and no live
        segment on record anywhere) does this take the lock, reload, and
        invalidate a pause the store might be holding.
        """
        principal = principal_for(authorization, settings=settings)
        live = studio.store.current(principal, body.conversation_id)
        if live is not None and live.cancel_event is not None:
            live.cancel_reason = "cancelled"
            live.cancel_event.set()
            return Response(status_code=204)
        if await studio.store.request_cancel(principal, body.conversation_id):
            return Response(status_code=204)
        hold = await _try_hold(studio.store, principal, body.conversation_id)
        if hold is None:
            # A request holds the lock: a claim on its way, or a stream.
            return Response(status_code=204)
        closed: ClosedTurn | None = None
        try:
            conversation = await studio.store.load(principal, body.conversation_id)
            if conversation is None:
                return Response(status_code=204)
            if conversation.awaiting_answer:
                closed = ClosedTurn(
                    conversation.trace_id, conversation.turn_user_message, ErrorCode.CANCELLED.value
                )
                conversation.close_turn(
                    final_text=conversation.turn_final_text, error=ErrorCode.CANCELLED.value
                )
            # The pending interrupts are cleared unconditionally, the same
            # abandonment `begin_turn` already does for a fresh turn on a
            # paused conversation; the parked ADK invocation needs nothing
            # beyond that, since it stays in its own per-turn session
            # either way.
            conversation.pending.clear()
            # `request_cancel` above found no live segment, and the save
            # below refuses if the load found one since, so the lease on
            # record, if any, has lapsed. Clearing it fences out a stream
            # still running under it: its checkpoint is fenced on the lease
            # id alone, and would otherwise write over the turn this closes.
            conversation.segment = None
            # With no live segment there is nothing for a release to fence
            # against. A plain save persists the clear. A stale write (a
            # segment claimed the instant after `request_cancel` checked) is
            # a rare enough race that losing this save is acceptable. The
            # claiming stream's own release still checkpoints whatever it
            # leaves behind.
            try:
                await studio.store.save(principal, body.conversation_id)
            except ConversationStale:
                logger.warning(
                    "studio cancel: conversation %s changed underneath this save; nothing "
                    "persisted",
                    body.conversation_id,
                )
                closed = None
        finally:
            hold.release()
        if closed is not None:
            # The paused turn's trace ends as its record does.
            record_closed_turn_span(body.conversation_id, closed)
            await flush_telemetry()
        return Response(status_code=204)

    @router.post("/interrupt")
    async def interrupt(  # noqa: PLR0911 -- one distinct refusal per claim guard the route enforces
        body: InterruptResponse, authorization: str | None = Header(default=None)
    ) -> Response:
        principal = principal_for(authorization, settings=settings)
        hold = await _acquire_or_conflict(studio.store, principal, body.conversation_id)
        if isinstance(hold, JSONResponse):
            return hold
        claim = _Claim(studio.store, principal, body.conversation_id, hold)
        try:
            conversation = await studio.store.load(principal, body.conversation_id)
            if conversation is None:
                hold.release()
                return _unknown_conversation_response(body.conversation_id)
            pending = conversation.pending.get(body.interrupt_id)
            refusal = _interrupt_refusal(conversation.pending, body)
            if refusal is not None:
                hold.release()
                return refusal
            assert pending is not None  # narrowed by _interrupt_refusal
            # Every change up to the claim save is made on a working copy,
            # which the save swaps into the live state once it commits. A
            # request that claims nothing leaves the live state as loaded.
            work = studio.store.draft(principal, body.conversation_id)
            pending = work.pending[body.interrupt_id]
            # A stream that runs a plan-card switch arms the switch's Stop:
            # one that answers the switch, and one that answers a question
            # the planner asked while replanning for a pick on the card.
            switching = body.decision is InterruptDecision.SWITCH or pending.pick_replan
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
                requested_answer["switch"] = {
                    "choice_id": body.choice_id,
                    "option_id": body.option_id,
                }
            resuming = pending.resuming_segment is not None
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
                work.approvals.append(
                    ApprovalRecord(
                        revision=pending.plan_revision,
                        approver=principal,
                        decision=_APPROVAL_DECISION[body.decision],
                        at=datetime.now(UTC),
                        conversation_id=body.conversation_id,
                        interrupt_id=body.interrupt_id,
                        plan_version=pending.plan_version,
                        auth_enforced=settings.studio_api_token is not None,
                    )
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
                    claim,
                    _streaming_conflict_response(
                        "another request is streaming on this conversation",
                        retry_after_seconds=exc.retry_after_seconds,
                    ),
                )
            if resuming and pending.answer != requested_answer:
                # The claim above only succeeds once the earlier resume's
                # segment has expired (its owner died before reaching a
                # checkpoint) or was dropped (it failed before its stream
                # started), so this is a genuinely dead attempt, and
                # this request's answer does not match what actually
                # resumed it. Refused for good: a live segment above would
                # already have caught a retry of the SAME answer, so a
                # mismatch reaching here can never become a match later.
                return await _unclaimed(
                    claim,
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
            # The entry is kept, marked resuming under this fresh segment.
            # A crash between this save and the resumed stream's own
            # checkpoint still leaves a durable, retryable record.
            # `_checkpointed` drops it once the resumed segment reaches its
            # own pause or terminal frame. Its release drops it on every
            # other ending (error, cancel, disconnect). Only a crash
            # leaves it for this same retry path to pick back up.
            pending.answer = requested_answer
            pending.resuming_segment = segment_id
            if body.decision is InterruptDecision.SWITCH and pending.card is not None:
                # What each choice held before the turn's first switch of
                # it, which `plan.approved` names as `changed_from`.
                work.turn_choice_baseline = {
                    **held_labels(pending.card),
                    **work.turn_choice_baseline,
                }
            # The resumed segment opens its own telemetry span on the turn's
            # trace, whichever instance runs it; the save hands it to the
            # live state.
            work.open_segment_span()
            # Saved before the resume stream starts: a double answer to the
            # same interrupt across two instances loses cleanly here, with a
            # 409, rather than each resuming its own segment of the turn.
            try:
                await studio.store.save(principal, body.conversation_id, work)
            except ConversationStale:
                return await _unclaimed(
                    claim,
                    _streaming_conflict_response("another request already answered this interrupt"),
                )
        except BaseException:
            await _release_claim(claim)
            raise

        # Every failure from here to the stream's start drops the lease the
        # save above claimed, and releases the lock, and so does a response
        # whose body never starts (`_ClaimedStream`). The pending entry stays
        # resuming under the dropped segment with the answer recorded, so a
        # retry of the same answer claims at once and resumes. The turn lines
        # added below stay unsaved, and the retry's load reads the row afresh.
        claim = replace(claim, segment_id=segment_id)
        try:
            conversation = studio.store.current(principal, body.conversation_id)
            assert conversation is not None
            pending = conversation.pending[body.interrupt_id]
            # A workflow's own steps run as concurrent waves, whose recording
            # order `_RecordingSurface` needs the plan's own step order to
            # undo (`_PendingInterrupt.step_order`'s docstring). It is set
            # before anything below can run the graph or close the turn.
            if studio.surface_recorder is not None:
                studio.surface_recorder.set_step_order(
                    f"{principal}/{body.conversation_id}", pending.step_order, pending.step_labels
                )

            if body.decision is InterruptDecision.REJECT:
                # The plan is declined and the turn ends; nothing was computed. Its
                # early retrievals, if any, are left to finish or lapse.
                # The paused ADK invocation is abandoned in its per-turn session.
                async def _rejected() -> AsyncGenerator[Any]:
                    text = "Plan rejected."
                    conversation.close_turn(final_text=text)
                    yield MessageComplete(
                        event="message.complete",
                        seq=conversation.next_seq(),
                        text=text,
                    )
                    yield Done(
                        event="done",
                        seq=conversation.next_seq(),
                        trace_id=conversation.trace_id,
                        composition=conversation.composition,
                    )

                return studio._segment_stream(claim, conversation, _rejected())

            if pending.kind == "elicitation":
                # skip dismisses the whole pause; an answer resumes with every
                # question's answer, keyed by question id.
                if body.decision is InterruptDecision.SKIP:
                    answer: dict[str, Any] = {"skip": True}
                else:
                    given = _given_answers(pending, body)
                    answer = {
                        "answers": {
                            question.id: _resume_answer(given[question.id])
                            for question in pending.questions
                        }
                    }
                    # The turn's history record must carry the answers or the
                    # next turn's digest shows only the un-refined ask.
                    for line in _answered_lines(pending, given):
                        conversation.add_to_turn(line, user_words=True)
            elif body.decision is InterruptDecision.APPROVE:
                # The plan is keyed on its revision hash (`plan_revision`), and
                # the approval dated by the record of its first answer, so a
                # retry resuming the same answer later approves as of the same
                # moment. The graph's `approve` node approves nothing without it.
                answer = {"approved": pending.plan_revision}
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
                # turn's history record carries the choice.
                assert body.choice_id is not None and body.option_id is not None
                answer = {
                    "approved": None,
                    "switch": {"choice_id": body.choice_id, "option_id": body.option_id},
                }
                label = pending.choices[body.choice_id].options[body.option_id].label
                verb = "use" if is_decision_choice(body.choice_id) else "count"
                conversation.add_to_turn(f"{verb} {label}", user_words=False)
            else:  # amend: a stated non-approval routes the graph to the planner
                answer = {"approved": None, "amendment": body.payload}
                # An amendment refines the question, so the turn's history record
                # carries it.
                conversation.add_to_turn(str(body.payload), user_words=True)
            current_conversation_id.set(f"{principal}/{body.conversation_id}")
            current_conversation_key.set((principal, body.conversation_id))
            # The user's words as amended, so a search after the resume reads the
            # refined ask and no switch's label.
            current_user_message.set(conversation.turn_user_words)
            current_earlier_block.set(conversation.earlier_block)
            current_recorded_identity.set(conversation.recorded_identity)
            start_turn_budget(settings)
            part = create_request_input_response(pending.adk_id, {"result": answer})
            message = types.Content(role="user", parts=[part])
            # The approved plan goes out once per interrupt: a retry resuming
            # the same answer sends it only when the dead attempt did not.
            approving = (
                pending.card
                if body.decision is InterruptDecision.APPROVE and not pending.approved_sent
                else None
            )
            frames = studio._frames(
                principal=principal,
                conversation=conversation,
                message=message,
                approving=approving,
            )
            return studio._segment_stream(
                claim, conversation, frames, switch=switching, settle=True
            )
        except BaseException as exc:
            await _release_claim(claim, cause=exc)
            raise

    @router.post("/composition")
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

    return router
