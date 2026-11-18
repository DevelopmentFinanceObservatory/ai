"""The studio endpoints: the v1 wire contract served over the turn loop.

Three routes (`docs/studio-contract-v1.md`):

* `POST /api/studio/turn`: runs one turn and streams v1 frames as SSE.
* `POST /api/studio/interrupt`: resumes a paused turn; the response is a
  NEW SSE stream continuing it (contract delta 1).
* `POST /api/studio/composition`: applies canvas mutations and returns the
  new manifest as plain JSON.

A turn runs on the turn loop (`harness.turn`) over the graph
`harness.graph.workflow.studio_graph` builds: `/turn` opens it at the
graph's start and runs its first segment, and `/interrupt` answers its
open pause and runs the next. The translator from the segment's events to
frames is `harness.studio_translate`; per-conversation state and the
composition rules are `harness.studio_state`; the Edit Card route is
`harness.studio_edit`. This module owns the routes, the `StudioRunner` that
runs a segment, and the SSE transport of the `/interrupt` and Edit Card
streams. A `/turn` segment's stream has an owner and a response of its own
(`harness.studio_stream`), with this module's `_SegmentWork` as its work.

Conversation identity (contract delta 5): the client mints
`conversation_id`; the bearer token resolves to a principal
(`harness.auth`); the store and the turn's cursor are keyed on the pair.
Each `turn.request` opens a FRESH turn whose state starts empty: turn 2's
planner drafts from a clean state, with no plan left over from turn 1,
while the composition manifest and the turn records persist across turns
in the store. Cross-turn context reaches the models as the turn's inputs:
the canvas digest (block titles + fact summaries) and the conversation
digest (the newest turn records); anything past their bounds is served on
demand by the recall function (`make_recall_part`).
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
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Literal

import anyio
from common.telemetry import flush_telemetry
from compiler.surface import ToolSurface
from fastapi import APIRouter, Header, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from fastapi.routing import APIRoute
from planner.plan import current_earlier_block, current_recorded_identity
from pydantic import BaseModel

from harness.approval import ApprovalDecision, ApprovalRecord
from harness.auth import principal_for
from harness.conversation import (
    current_conversation_id,
    current_conversation_key,
    current_user_message,
)
from harness.decision_pick import is_decision_choice
from harness.early_mint import EarlyMints
from harness.graph.events import GraphEvent
from harness.graph.interfaces import Executor
from harness.graph.state import (
    APPLY_CHOICE_STATE_KEY,
    ORCHESTRATOR_PRIOR_LEDGER_KEY,
    TurnInput,
)
from harness.provenance import ProvenanceStore
from harness.seeds import seeds_slot
from harness.sentences import failure_sentence
from harness.settings import Settings
from harness.studio_edit import edit_frames
from harness.studio_state import (
    HEARTBEAT_INTERVAL_S,
    ClosedTurn,
    ConversationState,
    DatabaseUnavailable,
    SegmentBusy,
    SegmentLost,
    _apply_operations,
    _next_manifest,
    _PendingInterrupt,
    _PendingQuestion,
    _tool_error_response,
    record_closed_turn_span,
    record_unstreamed_span,
    renew_lease_now,
    stream_loss,
)
from harness.studio_store import (
    ConversationExists,
    ConversationStale,
    StopPending,
    StudioStore,
    retry_delays,
    run_to_completion,
)
from harness.studio_stream import LiveOwners, open_stream
from harness.studio_translate import (
    SEGMENT_LOST_DETAIL,
    _checkpoint_failed_frame,
    _RecordingExecutor,
    _RecordingSurface,
    _stream_fatal,
    _Translator,
    held_labels,
    keep_turn_records,
    recorded_fatal,
    segment_lost_frame,
)
from harness.turn import transcript as entries
from harness.turn.events import TurnEvent, TurnFailed
from harness.turn.loop import (
    Answer,
    Graph,
    Outcome,
    Segment,
    abandon,
    reject,
    run_segment,
    stop_turn,
)
from harness.turn.store import Cursor, CursorStale, TurnStatus, TurnStore
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


@dataclass(frozen=True)
class _Stopped:
    """The last item of a stream whose work stopped before it ended: a Stop
    ended the turn (`"cancelled"`), or the conversation's lease is no longer
    this stream's (`"segment_lost"`). `_sse_stream` answers it with the
    frames that end the stream (`_stopped_frames`)."""

    reason: Literal["cancelled", "segment_lost"]


def _stopped_frames(
    conversation: ConversationState, open_steps: _OpenSteps, stopped: _Stopped
) -> list[Any]:
    """The frames that end a stream on `stopped`: a `failed` frame for every
    Activity line it left open (`_OpenSteps`), since no step reports on
    this stream again (a resumed segment reports the same `step_id` again,
    and the client's line-patching keys on it), then the stream-fatal
    `error` frame. A Stop closes the turn as `cancelled`. A lost lease
    leaves it to the instance that holds the lease now, which may still
    finish it, and answers `transient_unavailable` (`segment_lost_frame`)."""
    if stopped.reason == "segment_lost":
        return [
            *open_steps.close(conversation, detail=SEGMENT_LOST_DETAIL),
            segment_lost_frame(conversation),
        ]
    conversation.close_turn(
        final_text=conversation.turn_final_text, error=ErrorCode.CANCELLED.value
    )
    cancelled = failure_sentence(ErrorCode.CANCELLED, RetryMode.DO_NOT_RETRY)
    return [
        *open_steps.close(conversation, detail=cancelled),
        ErrorFrame(
            event="error",
            seq=conversation.next_seq(),
            error=ToolError(
                error=ErrorCode.CANCELLED, retry=RetryMode.DO_NOT_RETRY, detail=cancelled
            ),
        ),
    ]


@dataclass
class _StreamEnding:
    """How one segment's stream ended, shared by `_sse_stream`, which sends
    its frames, and `StudioRunner._locked`, which ends the segment's span
    with it.

    `status` is the cause of the last ending frame the stream sent, in the
    turn record's vocabulary (`harness.studio_state.span_level`'s table),
    `None` while it sent none. `StudioRunner._checkpointed` offers each
    ending frame it relays with its cause (`offer`), and `_sse_stream` sets
    `status` as that frame goes out (`sent`), or as it sends the frames that
    end a stream on `_Stopped`. It is set on sending, since `_sse_stream`
    can be closed at a heartbeat's yield while a frame `_locked` yielded
    waits unread."""

    status: str | None = None
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


async def _sse_stream(
    frames: AsyncGenerator[Any],
    *,
    conversation: ConversationState | None = None,
    heartbeat_seconds: float = HEARTBEAT_SECONDS,
    ending: _StreamEnding | None = None,
) -> AsyncGenerator[str]:
    """Serialise frames as SSE `data:` lines, one loop owning the transport:
    `: ka` comment heartbeats while the next frame is slow, the frames that
    end a stream whose work stopped (`_Stopped`), and client disconnect.

    A disconnect, at a wait or at any `yield`, unwinds `frames`, which
    cancels the segment it runs: the turn's cursor stays as last written and
    its runs unfinished, and the turn stays open, unrecorded. The next
    `/turn` abandons it and carries its runs on; a retried `/interrupt`
    carries it on from the cursor.

    A `_Stopped` item ends the stream on `_stopped_frames`, except after
    `done`: a turn that ended stays ended, and the stream reads `frames` on
    to their end. After a stream-fatal frame a `_Stopped` item still ends
    the stream `cancelled`, as when the failed turn's close went unwritten
    and the release acts on the Stop live for it (`StudioRunner._locked`).
    A stream-fatal `error` frame from `frames`
    first closes every Activity line the stream left open with a `failed`
    frame, as a Stop does, and is then re-stamped with the next `seq`, so
    `seq` still rises along the stream. Each frame is recorded on `ending`
    as it goes out (`_StreamEnding.sent`), which names the stream's ending.

    A stream that has a `conversation` runs inside its segment's telemetry
    span (`ConversationState.in_turn_span`), on the turn's trace, and the
    task that produces each frame is created inside that scope. The scope is
    left on the task that iterates the response. When the response task is
    cancelled while this generator sits at a `yield`, Starlette abandons the
    generator and the event loop closes it later from another context, where
    leaving the scope detaches nothing (`common.telemetry.use_turn_span`). On
    either path the finished spans are exported (`flush_telemetry`) as the
    stream ends, while the request still has CPU."""
    turn_span: AbstractContextManager[None] = (
        conversation.in_turn_span() if conversation is not None else nullcontext()
    )
    ending = ending or _StreamEnding()
    open_steps = _OpenSteps()
    next_frame: asyncio.Future[Any] | None = None
    with turn_span:
        try:
            async with aclosing(frames) as source:
                iterator = source.__aiter__()
                try:
                    while True:
                        next_frame = asyncio.ensure_future(anext(iterator))
                        while True:
                            done, _ = await asyncio.wait({next_frame}, timeout=heartbeat_seconds)
                            if done:
                                break
                            yield ": ka\n\n"
                        if next_frame.cancelled():
                            # Frames that ended on a cancellation of their
                            # own, with no ending frame for it.
                            return
                        try:
                            frame = next_frame.result()
                        except StopAsyncIteration:
                            return
                        if isinstance(frame, _Stopped):
                            if ending.status == "done":
                                # The turn ended first, and its `done`
                                # stands.
                                continue
                            assert conversation is not None
                            closing = _stopped_frames(conversation, open_steps, frame)
                            ending.status = frame.reason
                            for outgoing in closing:
                                yield _sse_data(outgoing)
                            return
                        ending.sent(frame)
                        if conversation is not None and isinstance(frame, ErrorFrame):
                            closed = open_steps.close(conversation, detail=frame.error.detail)
                            for outgoing in closed:
                                yield _sse_data(outgoing)
                            if closed:
                                frame = frame.model_copy(update={"seq": conversation.next_seq()})
                        open_steps.observe(frame)
                        yield _sse_data(frame)
                except (asyncio.CancelledError, GeneratorExit):
                    # CancelledError: the response task was cancelled while
                    # this waited. GeneratorExit: the consumer closed us at a
                    # yield, a frame's or a heartbeat's. Either way the
                    # client is gone. Without unwinding here, aclosing(source)
                    # would find the anext still running and raise. A frame
                    # task that finished first is unwound as it ended.
                    if next_frame is not None:
                        try:
                            await _unwind(next_frame)
                        except StopAsyncIteration:
                            pass
                        except Exception:
                            logger.exception("studio: a segment failed as its client left")
                    raise
        finally:
            await flush_telemetry()


async def _until_stopped(
    frames: AsyncGenerator[Any], stopped: asyncio.Future[Literal["cancelled", "segment_lost"]]
) -> AsyncGenerator[Any]:
    """`frames`, until `stopped` resolves first: then the frame being made
    is unwound and the stream ends on `_Stopped` with its reason."""
    async with aclosing(frames) as source:
        iterator = source.__aiter__()
        while True:
            next_frame = asyncio.ensure_future(anext(iterator))
            try:
                await asyncio.wait({next_frame, stopped}, return_when=asyncio.FIRST_COMPLETED)
            except BaseException:
                await _unwind(next_frame)
                raise
            if not next_frame.done():
                await _unwind(next_frame)
                yield _Stopped(stopped.result())
                return
            try:
                frame = next_frame.result()
            except StopAsyncIteration:
                return
            yield frame


async def _unwind(pending: asyncio.Future[Any]) -> None:
    """Cancel `pending` once and WAIT for it: a bare `.cancel()` leaves the
    generator mid-frame and `aclosing` then raises "asynchronous generator
    is already running".

    The wait is shielded. Awaiting a task forwards every cancellation of the
    waiting task to it, and Starlette's anyio task group re-cancels the
    response task on every loop iteration once the response is cancelled.
    Unshielded, that re-cancels `pending`'s own cleanup (a segment's closing
    write, `StudioStore.release`) at each of its awaits, and a SQLAlchemy
    connection torn down twice over asyncpg leaves a task that never
    finishes."""
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


# How a stream's release ends it (`StudioRunner._release`): released as
# usual, or on a Stop it acted on, with the turn closed or left open with
# the Stop live.
type _StopEnding = Literal["released", "closed", "unclosed"]


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
    (`_settling`, or the owner's `settle` on a `/turn` stream). A stream renews the conversation's lease every
    `heartbeat_seconds`. `owners` holds the owners of the `/turn` streams
    running on this instance (`harness.studio_stream`), which the app's
    lifespan shutdown waits for before it closes the database.
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
    ) -> None:
        self.edit_model = edit_model
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

    async def _frames(
        self,
        *,
        principal: str,
        conversation: ConversationState,
        cursor: Cursor,
        segment_id: str,
        answer: Answer | None = None,
        approving: PlanProposed | None = None,
    ) -> AsyncGenerator[Any]:
        """Run one segment of the turn at `cursor` (`harness.turn.loop.run_segment`)
        under `segment_id` and yield its wire frames as its events come.

        `answer` answers the turn's open pause, and is `None` for a turn the
        segment opens or carries on from a running cursor. `approving` is
        the plan card a first answer to its interrupt approves, which
        `plan.approved` states once the graph takes the approval
        (`_Translator`). The stream's `renew_now` reaches the segment's
        heartbeat (`_locked`).

        A segment that pauses or ends closes on `_Translator.finish`'s
        frames, with the turn's records as its cursor stores them
        (`keep_turn_records`). A failed one ends on its error frame: the
        translator's, from the `TurnFailed` event, or one worded from the
        error the cursor records when the close committed with no event
        (`recorded_fatal`). A Stop or a lost lease ends the stream on
        `_Stopped`, as does a turn another instance abandoned once it took
        the lapsed lease. A `SegmentLost` or `DatabaseUnavailable` that
        reaches here propagates, and the stream relaying this ends on it
        (`_checkpointed`). Any other failure is serialised as a stream-fatal
        `error` frame, since SSE headers are already on the wire; it is
        logged with the trace id and not re-raised into the transport.
        Leaving the stream early cancels the segment, which leaves the
        cursor as last written and its runs unfinished, and marks the
        segment unfinished (`ConversationState.unfinished_segment`), so
        `release` keeps a resuming answer for a retry to carry the turn on.
        """
        conversation_id = conversation.conversation_id
        translator = _Translator(
            conversation=conversation,
            recorder=self.recorder,
            surface_recorder=self.surface_recorder,
            approving=approving,
        )
        failed = False

        def emit(event: GraphEvent | TurnEvent) -> None:
            nonlocal failed
            failed = failed or isinstance(event, TurnFailed)
            translator.emit(event)

        segment_task: asyncio.Future[Any] | None = None
        try:
            # The turn's records so far, for a segment that resumes a turn
            # another instance began.
            keep_turn_records(conversation, cursor.state)
            segment_task = asyncio.ensure_future(
                run_segment(
                    self.turns,
                    self.store,
                    cursor,
                    principal=principal,
                    conversation_id=conversation_id,
                    segment_id=segment_id,
                    graph=self.graph,
                    emit=emit,
                    answer=answer,
                    heartbeat_seconds=self.heartbeat_seconds,
                    renew_now=conversation.renew_now,
                )
            )
            async for frame in translator.frames(segment_task):
                yield frame
            segment = segment_task.result()
            keep_turn_records(conversation, segment.cursor.state)
            match segment.outcome:
                case Outcome.CANCELLED:
                    yield _Stopped("cancelled")
                case Outcome.LOST:
                    yield _Stopped("segment_lost")
                case Outcome.ABANDONED if not failed:
                    # Another instance's `/turn` took the lapsed lease and
                    # closed the turn (`abandon`).
                    yield _Stopped("segment_lost")
                case Outcome.FAILED if not failed:
                    yield recorded_fatal(conversation, segment.cursor.error or {})
                case Outcome.UNCLOSED if not failed:
                    # The close on the Stop a `StopScope` took could not be
                    # written: the turn stays open with that Stop live, and
                    # the release, or the next claimant, closes it
                    # `cancelled` (`harness.turn.loop.drive`).
                    yield _Stopped("cancelled")
                case Outcome.FAILED | Outcome.ABANDONED | Outcome.UNCLOSED:
                    # The translator sent the `TurnFailed` event's error
                    # frame. An `unclosed` turn's close could not be
                    # written, and it stays open for a Stop or the next turn.
                    pass
                case _:
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
            if segment_task is not None and not segment_task.done():
                # The stream stopped reading while the segment ran: the
                # client left, or a frame's save ended the stream. The
                # cursor stays as last written and the run it computes
                # unfinished, for the next request to carry on.
                conversation.unfinished_segment = segment_id
                await _unwind(segment_task)
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
        renew_now: asyncio.Event,
    ) -> Literal["cancelled", "segment_lost"]:
        """Renew `segment_id`'s lease every `heartbeat_seconds`, or at once
        when `renew_now` is set, for a stream that runs no segment of its
        own (the Edit Card stream), and return why the stream must stop: a
        renewal reported a Stop from any instance (`"cancelled"`), or found
        another instance's claim in its place (`"segment_lost"`).

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
                return "segment_lost"
            if not renewal.held:
                return "segment_lost"
            if renewal.stop_id is not None:
                return "cancelled"

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
        A `_Stopped` item passes through. This is the one interception point
        for every stream this runs under (the Edit Card branch of `/turn`,
        and `/interrupt`).

        Every ending frame is offered to `ending` with its cause
        (`_StreamEnding.offer`): `done` and `paused` once their checkpoint
        landed, the loss (`stream_loss`) for a save that lost the segment,
        and otherwise the error frame's own code, for a save that failed
        (`transient_unavailable`, the turn recorded `state_not_saved`) and
        for a stream-fatal frame of `frames`."""

        def failed(what: str, exc: Exception) -> ErrorFrame:
            _log_save_failure(what, conversation, segment_id, exc)
            frame = _checkpoint_failed_frame(conversation, exc)
            ending.offer(
                frame,
                stream_loss(exc)
                if isinstance(exc, (SegmentLost, DatabaseUnavailable))
                else frame.error.error.value,
            )
            return frame

        card_write_lost: ErrorFrame | None = None

        async def relayed() -> AsyncGenerator[Any]:
            nonlocal card_write_lost
            async with _closing(frames, "the segment's frames"):
                try:
                    async for frame in frames:
                        yield frame
                except (SegmentLost, DatabaseUnavailable) as exc:
                    card_write_lost = failed("card write", exc)
                    yield card_write_lost

        async with _closing(relayed(), "the relayed frames") as source:
            async for frame in source:
                if isinstance(frame, _Stopped):
                    yield frame
                    continue
                try:
                    await self.store.reserve_frames(principal, conversation_id, segment_id)
                except Exception as exc:
                    if frame is card_write_lost:
                        # The card write that lost the segment has logged its
                        # loss and offered its frame already.
                        _log_save_failure("frame reservation", conversation, segment_id, exc)
                        yield frame
                    else:
                        yield failed("frame reservation", exc)
                    return
                event = getattr(frame, "event", None)
                outgoing = frame
                if event in _CHECKPOINT_BEFORE_EVENTS:
                    conversation.drop_resuming(segment_id)
                    try:
                        await self.store.checkpoint(principal, conversation_id, segment_id)
                    except Exception as exc:
                        yield failed("checkpoint", exc)
                        return
                    if event == "done":
                        outgoing = frame.model_copy(
                            update={"composition": conversation.composition}
                        )
                    ending.offer(outgoing, "done" if event == "done" else "paused")
                elif isinstance(frame, ErrorFrame) and frame is not card_write_lost:
                    ending.offer(frame, frame.error.error.value)
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
                            yield failed("checkpoint after plan.approved", exc)
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
        *,
        renewing: bool = False,
        ending: _StreamEnding | None = None,
    ) -> AsyncGenerator[Any]:
        """Stream frames, checkpointing before any that report success
        (`_checkpointed`), then release the segment and the conversation
        lock.

        The ROUTE acquires the lock (eagerly, before touching turn state), so
        ownership settles while the request handler still controls the
        response. A lazily-acquired lock let two requests pass the 409
        guard and clobber each other's turn state.

        While the stream runs, `conversation.renew_now` is its own, so a
        Stop posted on this instance reaches it at once. A segment's
        frames (`_frames`) renew the lease through the turn loop; with
        `renewing`, for a stream that runs no segment, a heartbeat task
        renews it (`_heartbeat`) and the stream stops as it says
        (`_until_stopped`). `frames` is closed before the release, however
        the stream ends, so the segment has stopped when the lease is let go
        and `release` reads whether it stopped unfinished. The heartbeat is
        cancelled and awaited before release, so it never races
        `store.release`'s own write. A release that keeps the lease for a
        Stop live on the open turn closes the turn under it (`_release`);
        the stream then reserves frame numbers and sends the `cancelled`
        frame while it still holds the lease and the lock, and releases the
        lease once more after it, or drops it when the close failed and the
        Stop stays live for the next claimant. On a stream that ended
        abnormally the same close runs and no frame is sent. Frames that end
        on a cancellation of their own, this generator's task uncancelled,
        end the stream by `_frames_failed`. Unwinding the heartbeat,
        releasing the segment (and this segment's resuming pending entries),
        ending the segment's span and releasing the local lock each sit in
        their own nested `finally`, so a failure in one still reaches the
        others.

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
        renew_now = conversation.renew_now = asyncio.Event()
        heartbeat = (
            asyncio.ensure_future(
                self._heartbeat(principal, conversation_id, segment_id, renew_now)
            )
            if renewing
            else None
        )
        released = False
        ran_out = False
        stopping: _StopEnding = "released"

        async def unwind_heartbeat() -> None:
            if conversation.renew_now is renew_now:
                conversation.renew_now = None
            if heartbeat is not None:
                await _unwind(heartbeat)

        async def release() -> None:
            # Runs to the end as its own task, so what it did is known even
            # when the caller is cancelled while it runs.
            nonlocal released, stopping
            released = True
            stopping = await self._release(principal, conversation_id, segment_id)

        try:
            async with _closing(
                self._checkpointed(
                    principal,
                    conversation_id,
                    segment_id,
                    conversation,
                    _until_stopped(frames, heartbeat) if heartbeat is not None else frames,
                    ending=ending,
                ),
                "the segment's frames",
            ) as checked:
                try:
                    async for frame in checked:
                        yield frame
                except asyncio.CancelledError as exc:
                    task = asyncio.current_task()
                    if task is not None and task.cancelling():
                        raise
                    # The frames failed with a cancellation of their own,
                    # this generator's task uncancelled.
                    for frame in await self._frames_failed(
                        principal, conversation_id, segment_id, conversation, ending, exc
                    ):
                        yield frame
            ran_out = True
            await unwind_heartbeat()
            await run_to_completion(release())
            if stopping != "released":
                # The closing frames go out under the lease and the lock.
                with suppress(SegmentLost, DatabaseUnavailable):
                    await self.store.reserve_frames(principal, conversation_id, segment_id)
                yield _Stopped("cancelled")
        finally:
            try:
                await unwind_heartbeat()
            finally:
                try:
                    if not released:
                        await run_to_completion(release())
                    if stopping == "closed":
                        await self.store.release(principal, conversation_id, segment_id)
                    elif stopping == "unclosed":
                        await self.store.drop_lease(principal, conversation_id, segment_id)
                finally:
                    try:
                        # The segment ends here on this instance, and its
                        # span with it; the stream's end exports the span
                        # (`_sse_stream`).
                        conversation.end_segment_span(
                            ending.status or ("failed" if ran_out else "disconnected"),
                            stored=self._stored_turn(principal, conversation_id),
                        )
                    finally:
                        conversation.lock.release()

    async def _frames_failed(
        self,
        principal: str,
        conversation_id: str,
        segment_id: str,
        conversation: ConversationState,
        ending: _StreamEnding,
        exc: asyncio.CancelledError,
    ) -> list[Any]:
        """The frames that end a stream whose frames failed with a
        cancellation of their own, the segment's task uncancelled (an
        awaited future cancelled under them, say), by the rule every ending
        follows. After an ending frame went out (`done`, a stream-fatal
        frame or a pause), it sends nothing, keeps that frame's status and
        logs a warning. Otherwise it logs an error and the turn closes with
        a stream-fatal frame (`_stream_fatal`), its number reserved (logged
        when that fails), offered as `failed`."""
        message = (
            "studio: conversation %s's frames ended cancelled, the segment's task uncancelled "
            "(trace_id=%s)"
        )
        if ending.status is not None:
            logger.warning(message, conversation_id, conversation.trace_id, exc_info=exc)
            return []
        logger.error(message, conversation_id, conversation.trace_id, exc_info=exc)
        frame = _stream_fatal(conversation, RuntimeError("the segment's frames were cancelled"))
        ending.offer(frame, "failed")
        try:
            await self.store.reserve_frames(principal, conversation_id, segment_id)
        except Exception as reservation:
            _log_save_failure("frame reservation", conversation, segment_id, reservation)
        return [frame]

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
        principal: str,
        conversation_id: str,
        conversation: ConversationState,
        segment_id: str,
        frames: AsyncGenerator[Any],
        *,
        renewing: bool = False,
    ) -> AsyncGenerator[str]:
        """The SSE stream of a claimed segment: `frames` under `_locked`,
        relayed by `_sse_stream`, both sharing one `_StreamEnding`."""
        ending = _StreamEnding()
        return _sse_stream(
            self._locked(
                principal,
                conversation_id,
                conversation,
                segment_id,
                frames,
                renewing=renewing,
                ending=ending,
            ),
            conversation=conversation,
            ending=ending,
        )

    async def _release(self, principal: str, conversation_id: str, segment_id: str) -> _StopEnding:
        """Release `segment_id` (`StudioStore.release`) and say how the
        stream ends. A release that keeps the lease for a Stop live on the
        open turn makes this stream the party that acts on it: the turn
        closes `cancelled` under the lease (`_close_on_stop`), and this
        returns `"closed"` with the lease still held for the closing frames
        and a last release. When that close fails it returns `"unclosed"`:
        the stream still ends `cancelled`, since the Stop stays live and the
        next claimant closes the turn `cancelled` before anything else, and
        the lease is then dropped (`drop_lease`). Otherwise the lease is
        released and this returns `"released"`."""
        if not await self.store.release(principal, conversation_id, segment_id):
            return "released"
        try:
            closed = await _close_on_stop(self, principal, conversation_id, segment_id)
        except Exception:
            logger.exception(
                "studio stream: conversation %s's turn could not be closed on a Stop; it "
                "stays open with the Stop live for the next claimant",
                conversation_id,
            )
            return "unclosed"
        if not closed:
            await self.store.release(principal, conversation_id, segment_id)
            return "released"
        return "closed"


class _SegmentWork:
    """The work of a claimed `/turn` segment, for its owner
    (`harness.studio_stream.Work`): one segment of the turn at `cursor`
    (`harness.turn.loop.run_segment`) under `segment_id`, its events
    translated into wire frames (`_Translator`).

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
    the turn. A turn another instance abandoned ends `abandoned`; one this
    segment closed `abandoned` on a node's failure ends `failed` on that
    failure's frame. A card write fenced out by another instance's takeover,
    or one that could not reach the database before the lease's deadline
    (`StudioStore.change_canvas`), ends it `lost`. Cancelling it cancels the
    segment and waits for it, which leaves the cursor as last written and
    its runs unfinished, for the next request to carry on. A step the
    executor finished after the last translated event belongs to this
    segment, so the recorders are emptied however it ends."""

    def __init__(
        self,
        runner: StudioRunner,
        *,
        principal: str,
        conversation: ConversationState,
        cursor: Cursor,
        segment_id: str,
    ) -> None:
        self._runner = runner
        self._principal = principal
        self._conversation = conversation
        self._cursor = cursor
        self._segment_id = segment_id

    async def run(self, send: Callable[[BaseModel], None]) -> Outcome:
        """Run the segment, `send` each frame, and return how it ended."""
        runner, conversation = self._runner, self._conversation
        translator = _Translator(
            conversation=conversation,
            recorder=runner.recorder,
            surface_recorder=runner.surface_recorder,
        )
        failed = False

        def emit(event: GraphEvent | TurnEvent) -> None:
            nonlocal failed
            failed = failed or isinstance(event, TurnFailed)
            translator.emit(event)

        renew_now = conversation.renew_now = asyncio.Event()
        segment_task: asyncio.Task[Segment] | None = None
        try:
            # The turn's records so far, for a segment that resumes a turn
            # another instance began.
            keep_turn_records(conversation, self._cursor.state)
            segment_task = asyncio.create_task(
                run_segment(
                    runner.turns,
                    runner.store,
                    self._cursor,
                    principal=self._principal,
                    conversation_id=conversation.conversation_id,
                    segment_id=self._segment_id,
                    graph=runner.graph,
                    emit=emit,
                    heartbeat_seconds=runner.heartbeat_seconds,
                    renew_now=renew_now,
                )
            )
            async for frame in translator.frames(segment_task):
                send(frame)
            segment = segment_task.result()
            keep_turn_records(conversation, segment.cursor.state)
            outcome = segment.outcome
            match outcome:
                case Outcome.PAUSED | Outcome.DONE:
                    for frame in await translator.finish():
                        send(frame)
                case Outcome.FAILED if not failed:
                    send(recorded_fatal(conversation, segment.cursor.error or {}))
                case Outcome.UNCLOSED if not failed:
                    return Outcome.CANCELLED
                case Outcome.ABANDONED if failed:
                    return Outcome.FAILED
            assert outcome is not None
            return outcome
        except (SegmentLost, DatabaseUnavailable) as exc:
            _log_save_failure("card write", conversation, self._segment_id, exc)
            return Outcome.LOST
        finally:
            if conversation.renew_now is renew_now:
                conversation.renew_now = None
            if segment_task is not None and not segment_task.done():
                segment_task.cancel()
                await asyncio.wait({segment_task})
            runner.recorder.drain(current_conversation_id.get())
            if runner.surface_recorder is not None:
                runner.surface_recorder.drain(current_conversation_id.get())

    def fatal(self, error: BaseException) -> ErrorFrame:
        """The stream-fatal frame for `error` (`_stream_fatal`), which
        records the turn failed on it; a cancellation of the work's own is a
        fault of the segment's frames."""
        if not isinstance(error, Exception):
            error = RuntimeError("the segment's frames were cancelled")
        return _stream_fatal(self._conversation, error)


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


class _BoundedTurnRoute(APIRoute):
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


async def _acquire_or_conflict(
    store: StudioStore, principal: str, conversation_id: str
) -> asyncio.Lock | JSONResponse:
    """Take the conversation's lock BEFORE loading or touching any turn
    state.

    The check-and-acquire pair has no suspension point between them on
    the uncontended fast path, so two concurrent requests cannot both
    pass. A lazily-acquired lock let them race past the 409 guard and
    clobber each other's turn state and cancel event. The streaming
    wrapper (`StudioRunner._locked`) releases; early-exit route paths
    must release explicitly. Returns the lock itself, not a conversation --
    a lock exists (and can be contended) before this store has ever loaded
    or created the conversation it names.
    """
    lock = store.lock_for(principal, conversation_id)
    if lock.locked():
        return _streaming_conflict_response("another request is streaming on this conversation")
    await lock.acquire()
    return lock


async def _close_on_stop(
    studio: StudioRunner, principal: str, conversation_id: str, segment_id: str
) -> bool:
    """Close the conversation's open turn `cancelled` on a live Stop, under
    `segment_id`, the lease the caller holds: save the turn's record closed
    with its pending interrupts cleared (`_save_turn_closed`), then close the
    turn, which ends the runs it left computing in the same transaction
    (`harness.turn.store.TurnStore.finish`). Returns `False` when no turn is
    open. Raises what either write raises. The record goes first because
    only the turn's close ends the Stop: whichever write fails, the turn
    stays open with its Stop live, for the next claimant to close
    (`_act_on_stop`), which finds the record already closed when it was.
    The stream that holds the lease ends its own segment's span on the
    `cancelled` frame it sends."""
    cursor = await studio.turns.open_cursor(principal, conversation_id)
    if cursor is None:
        return False
    await _save_turn_closed(studio, principal, conversation_id, cursor)
    await stop_turn(studio.turns, cursor, segment_id=segment_id)
    return True


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
    own (`_stop_turn_at`). Returns the error to answer, `None` once the turn
    is closed or none is open: the 503 `_stop_turn_at` answers when the
    close fails, or a retryable 409 when its claim lost to another request
    or the turn moved first. With no open turn the Stop names a turn
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
    if studio.store.drop_draft(principal, conversation_id):
        await flush_telemetry()
    await studio.store.load(principal, conversation_id)
    return await _act_on_stop(studio, principal, conversation_id)


async def _answer_after_stop(
    studio: StudioRunner,
    principal: str,
    conversation_id: str,
    lock: asyncio.Lock,
    then: Response,
    *,
    turn: ConversationState | None = None,
) -> Response:
    """Close the turn a live Stop names (`_close_stopped_turn`), release
    `lock` (`_unclaimed`, given `turn`), and answer that close's refusal,
    else `then`."""
    refused = await _close_stopped_turn(studio, principal, conversation_id)
    return await _unclaimed(
        studio.store, principal, conversation_id, lock, refused or then, turn=turn
    )


async def _release_claim(
    store: StudioStore,
    principal: str,
    conversation_id: str,
    lock: asyncio.Lock,
    segment_id: str | None,
    *,
    cause: BaseException | None = None,
    turn: ConversationState | None = None,
) -> None:
    """Let go of a request whose stream never started, on every path,
    cancellation included. A claim that never committed gives up its
    working copy (`StudioStore.drop_draft`), whose span ends `not_started`.
    A claim that committed (`segment_id`) drops its lease
    (`StudioStore.drop_lease`, which runs to the end under cancellation),
    and the segment's span, if one opened, ends by what let it go: `cause`,
    the exception that ended the request, gives `cancelled` when it is this
    task's own cancellation (the client gone) and `failed` otherwise, a
    `CancelledError` an inner task let out among them, and a refusal, with
    no `cause`, gives `not_started`. A `/turn` request opens no span before
    its stream, whose owner opens one: `turn`, the working copy its claim
    began a turn on, gets a zero-length span with that status on the turn's
    trace when no span ended (`record_unstreamed_span`). Then `lock` is
    released and any span that ended is exported."""
    task = asyncio.current_task()
    if cause is None:
        status = "not_started"
    elif isinstance(cause, asyncio.CancelledError) and task is not None and task.cancelling() > 0:
        status = "cancelled"
    else:
        status = "failed"
    ended = False
    try:
        ended = store.drop_draft(principal, conversation_id)
        if segment_id is not None:
            await store.drop_lease(principal, conversation_id, segment_id)
    finally:
        live = store.current(principal, conversation_id)
        if segment_id is not None and live is not None:
            ended = live.end_segment_span(status) or ended
        if turn is not None and not ended:
            record_unstreamed_span(
                conversation_id,
                trace_id=turn.trace_id,
                user_message=turn.turn_user_message,
                status=status,
            )
            ended = True
        lock.release()
        if ended:
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
    turn, the lease on record is cleared: it has lapsed, since
    `request_cancel` found no live segment and the save refuses if one went
    live since, and clearing it fences out a stream still running under it,
    whose checkpoint is fenced on the lease id alone.

    Raises:
        _ClaimLost: another request claimed the conversation, or moved its
            turn, since this request found no live lease.
    """
    cursor = await studio.turns.open_cursor(principal, conversation_id)
    if cursor is None:
        try:
            closed = await _save_turn_closed(
                studio, principal, conversation_id, None, clear_lease=True
            )
        except ConversationStale as exc:
            raise _ClaimLost from exc
        await _record_closed(conversation_id, closed)
        return None
    return await _stop_turn_at(studio, principal, conversation_id, cursor)


async def _stop_turn_at(
    studio: StudioRunner, principal: str, conversation_id: str, cursor: Cursor
) -> Response | None:
    """Close the open turn at `cursor` `cancelled` under a lease this claims
    with a save that acts on any Stop live for it (`StudioStore.save` with
    `stopping`), and drop the lease. Under the lease this records a Stop of
    its own (`StudioStore.request_cancel`), saves the turn's record closed
    with its pending interrupts cleared (`_save_turn_closed`), then closes
    the turn, which ends the runs it left computing in the same transaction
    (`harness.turn.loop.stop_turn`) and ends the Stop. Returns `None` once
    the turn is closed, and a retryable 503 when a write fails: the turn
    then stays open with a Stop live, so the next claimant closes it
    (`_act_on_stop`) and no later turn carries its runs on. A turn closed
    here ran no segment of its own, so its trace ends on a closing span
    (`_record_closed`).

    Raises:
        _ClaimLost: the claim lost to another request, the lease was taken
            over before the writes landed, or the turn moved since `cursor`
            was read.
    """
    try:
        segment_id = studio.store.claim_segment(principal, conversation_id)
        await studio.store.save(principal, conversation_id, stopping=True)
    except (SegmentBusy, ConversationStale) as exc:
        raise _ClaimLost from exc
    closed: ClosedTurn | None = None
    try:
        try:
            if not await studio.store.request_cancel(principal, conversation_id):
                raise _ClaimLost
            closed = await _save_turn_closed(studio, principal, conversation_id, cursor)
            await stop_turn(studio.turns, cursor, segment_id=segment_id)
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
    await _record_closed(conversation_id, closed)
    return None


async def _record_closed(conversation_id: str, closed: ClosedTurn | None) -> None:
    """End the trace of a turn closed with no segment of its own running,
    `closed`, on its closing span (`record_closed_turn_span`), and export
    it. A no-op for `None`."""
    if closed is not None:
        record_closed_turn_span(conversation_id, closed)
        await flush_telemetry()


async def _save_turn_closed(
    studio: StudioRunner,
    principal: str,
    conversation_id: str,
    open_turn: Cursor | None,
    *,
    clear_lease: bool = False,
) -> ClosedTurn | None:
    """Save the conversation's turn closed `cancelled` when it is open,
    keeping the records `open_turn`, its cursor, holds (`keep_turn_records`),
    with its pending interrupts cleared, so an answer to one is refused;
    with `clear_lease`, the lease on record cleared too. The change is made
    on a working copy (`StudioStore.draft`) and saved (`StudioStore.save`):
    only once the save commits does the live state take it. Returns the
    turn it closed, `None` when none was open, for a caller that closed it
    with no segment of its own running to record its closing span
    (`_record_closed`). Raises what the save raises, with the live state as
    it was."""
    work = studio.store.draft(principal, conversation_id)
    work.pending.clear()
    closed: ClosedTurn | None = None
    if work.turn_index and not work.turn_recorded:
        closed = ClosedTurn(work.trace_id, work.turn_user_message, "cancelled")
        if open_turn is not None:
            keep_turn_records(work, open_turn.state)
        work.close_turn(final_text=work.turn_final_text, error=ErrorCode.CANCELLED.value)
    if clear_lease:
        work.segment = None
    try:
        await studio.store.save(principal, conversation_id, work)
    except BaseException:
        studio.store.drop_draft(principal, conversation_id)
        raise
    return closed


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
    current_conversation_id.set(f"{principal}/{conversation_id}")
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


async def _unclaimed(
    store: StudioStore,
    principal: str,
    conversation_id: str,
    lock: asyncio.Lock,
    response: Response,
    *,
    turn: ConversationState | None = None,
) -> Response:
    """Answer `response` for a request whose claim save never committed,
    once it has let go of it (`_release_claim`, given `turn`), so the live
    state stays as loaded."""
    await _release_claim(store, principal, conversation_id, lock, None, turn=turn)
    return response


# How long, by wall time, `/cancel` waits on a lock held with no live lease
# (a request before its claim save, or a stream after its release) before it
# answers a retryable 503.
CANCEL_LOCK_WAIT_S = 5.0


async def _lock_or_claim(lock: asyncio.Lock, claimed: asyncio.Event, *, timeout: float) -> bool:
    """Wait up to `timeout` for `lock` or for `claimed`, whichever comes
    first. Past the deadline (`timeout` spent) a free lock with no waiter is
    still taken at once. Returns whether this now holds `lock`; `False` when
    a claim committed first or the time ran out. Cancelled after it took the
    lock, it releases the lock."""
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


async def _stop_under_lock(studio: StudioRunner, principal: str, conversation_id: str) -> Response:
    """`/cancel`'s answer once it holds the conversation lock: 204 when a
    lease went live since (another instance claimed while this one waited,
    and the Stop is recorded on it), when the store holds no such
    conversation, or once what an open turn left is closed
    (`_stop_open_turn`); that close's 503 when it fails.

    Raises:
        _ClaimLost: this route's claim lost to another request.
    """
    if await studio.store.request_cancel(principal, conversation_id):
        return Response(status_code=204)
    if await studio.store.load(principal, conversation_id) is None:
        return Response(status_code=204)
    # A turn paused on a question, or one its client left while it ran, is
    # open with no stream to close it.
    refused = await _stop_open_turn(studio, principal, conversation_id)
    return refused or Response(status_code=204)


def create_studio_router(studio: StudioRunner, *, settings: Settings) -> APIRouter:
    router = APIRouter(prefix="/api/studio")

    async def turn(  # noqa: PLR0911 -- one distinct refusal per claim guard the route enforces
        body: TurnRequest, authorization: str | None = Header(default=None)
    ) -> Response:
        principal = principal_for(authorization, settings=settings)
        lock = await _acquire_or_conflict(studio.store, principal, body.conversation_id)
        if isinstance(lock, JSONResponse):
            return lock
        # The segment this request claimed once its claim save landed, which
        # every failure before the stream starts releases, and the working
        # copy its claim began the turn on, whose trace records how it let
        # go (`_release_claim`).
        claimed: str | None = None
        begun: ConversationState | None = None
        try:
            conversation = await studio.store.load(principal, body.conversation_id)
            if conversation is None:
                if body.component_ref is not None or body.apply_choice is not None:
                    # The client refers to state (a card to edit or to follow
                    # up on, or a chip under an answer) this store never had.
                    lock.release()
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
                        studio.store, principal, body.conversation_id, lock, refused
                    )
                conversation = await studio.store.load(principal, body.conversation_id)
                assert conversation is not None
            if (
                body.component_ref is not None
                and body.component_ref.intent is ComponentRefIntent.FOLLOW_UP
                and body.component_ref.component_id not in conversation.components
            ):
                # A follow-up about a card this conversation never stored
                # would run as a plain turn with no card in view.
                lock.release()
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
                    lock.release()
                    return label
                chip_label = label
            # A turn still open on the conversation, paused on a question or
            # left running by a segment whose lease lapsed, is abandoned once
            # this request holds the lease. Its records are taken first, so
            # `begin_turn` keeps them in the record it closes the turn with.
            # A turn the tables record as open with no cursor closes on what
            # the conversation holds. Every change up to the claim save is
            # made on a working copy, which the save swaps into the live
            # state once it commits; a request that claims nothing leaves
            # the live state as loaded.
            work = studio.store.draft(principal, body.conversation_id)
            left_open = await studio.turns.open_cursor(principal, body.conversation_id)
            if left_open is not None and not work.turn_recorded:
                keep_turn_records(work, left_open.state)
            abandoned = work.begin_turn()
            begun = work
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
                    studio.store,
                    principal,
                    body.conversation_id,
                    lock,
                    _streaming_conflict_response(
                        "another request is streaming on this conversation",
                        retry_after_seconds=exc.retry_after_seconds,
                    ),
                    turn=work,
                )
            # The claim: advancing turn_index (and the segment above) is
            # saved before the turn opens and before the SSE response
            # starts. This save is a compare-and-swap against what this
            # instance last loaded, so a `ConversationStale` here means
            # another instance's `/turn` claimed a turn on this conversation
            # since. Claiming first leaves no crash window: a crash between
            # the claim and the turn's opening leaves the store at the new
            # turn_index with no turn open, and the next `/turn` claims the
            # one after it; the segment it also claimed simply expires by
            # its own TTL. The save refuses while a Stop is live for the
            # open turn (`StopPending`), so the turn index never moves past
            # one: a Stop that landed since the check above ends that turn
            # here, and the request is answered a retryable 409.
            try:
                await studio.store.save(principal, body.conversation_id, work)
            except StopPending:
                return await _answer_after_stop(
                    studio,
                    principal,
                    body.conversation_id,
                    lock,
                    _streaming_conflict_response(
                        "the conversation's previous turn was stopped", retry_after_seconds=1
                    ),
                    turn=work,
                )
            except ConversationStale:
                return await _unclaimed(
                    studio.store,
                    principal,
                    body.conversation_id,
                    lock,
                    _streaming_conflict_response(
                        "another request is streaming on this conversation"
                    ),
                    turn=work,
                )
            claimed = segment_id
            conversation = studio.store.current(principal, body.conversation_id)
            assert conversation is not None
            if abandoned is not None:
                # The unrecorded turn this one closed as `abandoned` ends its
                # trace as its record does. The stream's end exports the
                # span.
                record_closed_turn_span(body.conversation_id, abandoned)
            _set_turn_context(principal, body.conversation_id, conversation, settings)
            if left_open is not None:
                await abandon(studio.turns, left_open, segment_id=segment_id)
            if (
                body.component_ref is not None
                and body.component_ref.intent is ComponentRefIntent.EDIT
            ):
                # The Edit Card route bypasses the graph entirely: no
                # resolve, no plan, no turn on the loop, just the card, the
                # model's patch-or-recreate decision, and the platform write.
                # It streams under the same segment the claim save above
                # just persisted, in a span of its own.
                conversation.open_segment_span()
                return StreamingResponse(
                    studio._segment_stream(
                        principal,
                        body.conversation_id,
                        conversation,
                        segment_id,
                        edit_frames(
                            recorder=studio.recorder,
                            edit_model=studio.edit_model,
                            conversation=conversation,
                            component_id=body.component_ref.component_id,
                            message=body.message,
                        ),
                        renewing=True,
                    ),
                    media_type=SSE_MEDIA_TYPE,
                )
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
            if body.apply_choice is not None:
                # `graph.seed` applies the chip, and the turn neither searches nor plans.
                state[APPLY_CHOICE_STATE_KEY] = body.apply_choice.model_dump(mode="json")
            cursor = await studio.turns.open_turn(
                principal,
                body.conversation_id,
                segment_id=segment_id,
                turn_index=conversation.turn_index,
                node=studio.graph.start,
                inputs=inputs.model_dump(mode="json"),
                state=state,
                records=[entries.user_message(text)],
            )
            # The segment's owner takes the lease and the lock from here on.
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
                settle=studio.early_mints.settle if studio.early_mints is not None else None,
            )
        except BaseException as exc:
            # The stream's owner, or the Edit Card stream's wrapper, releases
            # the segment and the lock. A failure before the response starts
            # streaming releases them here (`_release_claim`). Before the
            # claim save commits, the working copy is given up and the live
            # state stays as loaded. After it, the lease is dropped. The
            # claim save already covers `begin_turn`'s effects, and a turn
            # opened here stays open, so the next `/turn` closes both as it
            # closes any open turn.
            if lock.locked():
                await _release_claim(
                    studio.store,
                    principal,
                    body.conversation_id,
                    lock,
                    claimed,
                    cause=exc,
                    turn=begun,
                )
            raise

    router.add_api_route("/turn", turn, methods=["POST"], route_class_override=_BoundedTurnRoute)

    @router.post("/cancel")
    async def cancel(
        body: CancelRequest, authorization: str | None = Header(default=None)
    ) -> Response:
        """Stop the conversation's turn (the Stop button). Idempotent: an
        unknown conversation, and an idle one, are both a no-op 204 (a Stop
        can reach an instance before the first `/turn` has claimed the
        conversation there at all).

        This route answers 204 only after one of three things: it recorded
        the Stop on a live lease (`StudioStore.request_cancel`), it closed
        the open turn itself under the conversation lock, or under the lock
        it found no open turn and no live lease. A Stop recorded on a lease
        names the turn that lease holds, the row's turn index
        (`harness.studio_store.StopState`), and is live until that turn
        closes, whatever the status; no write marks it read. The holder of
        the lease acts on it: its heartbeat while it drives the turn, the
        release of its lease once the drive has returned. When the holder
        cannot, the next claim does: a claim that starts or resumes work
        refuses while a Stop is live for the open turn (`StopPending`), so
        `/turn` and `/interrupt` first close that turn `cancelled` under a
        claim of their own (`_act_on_stop`); then `/turn` claims and
        `/interrupt` answers 404. A cancelled close ends the turn's
        unfinished runs in its own transaction
        (`harness.turn.store.TurnStore.finish`). The one exception is the
        first Stop while a node holds the turn's `StopScope` armed: the
        scope takes it with no write, and the turn's next pause write
        records it taken, which ends it.

        | When the Stop commits | Acted on by | Turn after | Client is told |
        |---|---|---|---|
        | Lease live, a node running | heartbeat | `cancelled` | `cancelled` frame |
        | Lease live, `StopScope` armed, first Stop | the scope; the re-offer's pause write records it taken | paused on the re-offered plan | `plan.proposed`, whose card can be answered |
        | The same, a second Stop before the pause write | heartbeat | `cancelled` | `cancelled` frame |
        | The same, a second Stop after the pause write, lease not yet released | the release, under the lease it keeps | `cancelled`, pending cleared | `cancelled` frame after `plan.proposed` |
        | The scope took the first Stop, then the client left before the pause write | the release, under the lease it keeps | `cancelled` | nothing more |
        | The scope took the first Stop, and the pause write failed | the drive, which closes the turn on the failure | `failed` | `error` frame |
        | The scope took the first Stop, and its node ended the turn or left on an edge that is not a stop edge, or the node after it did not pause | the drive, in place of that write; when that close fails, the release or the next claimant | `cancelled` | `cancelled` frame |
        | Lease live, pause committed, the drive in after-commit work | heartbeat, `_settle` | `cancelled` | `cancelled` frame |
        | Lease live, the segment returned paused, lease not yet released | the release, under the lease it keeps | `cancelled`, pending cleared | `cancelled` frame after the pause frame, while the stream is open; another request meanwhile answers 409 |
        | Lease live, turn closed, lease not yet released | nobody: the close ended the Stop | unchanged | nothing more |
        | `/turn` claimed, before it abandons the turn left open | the new turn's heartbeat | the old turn `abandoned`, the new `cancelled` | `cancelled` frame |
        | Lease live, an Edit Card stream | its heartbeat | no turn is open | `cancelled` frame |
        | The release kept the lease and closing failed, such as ending the turn's runs | the next claimant, as the lease is dropped | `cancelled` | `cancelled` frame; the next `/turn` streams, an `/interrupt` answers 404 |
        | The release's write failed or timed out | the next claimant, once the lease lapses | `cancelled` | nothing more; the next `/turn` streams, an `/interrupt` answers 404 |
        | The same, and the next `/turn`'s close of the turn is refused | the `/turn` retried after its 409 | `cancelled` | nothing more; that `/turn` answers a retryable 409 and starts nothing, and its retry streams |
        | A lapsed lease, after the next claimant's check and before its claim save | that claimant, whose save refuses | `cancelled` | nothing more; that `/turn` answers a retryable 409 and its retry streams, an `/interrupt` answers 404 |
        | No live lease, turn paused | this route, under the lock | `cancelled`, pending cleared | 204 |
        | No live lease, turn left running | this route, under the lock | `cancelled`, its runs ended | 204, or 503 when the close fails, which leaves a Stop live for the next claimant |
        | No live lease, no open turn | this route, under the lock, finding nothing open | unchanged | 204 |
        | No live lease, and this route's own claim loses to another request | the holder of the lease that request claimed, as this route decides again | as the row it lands on | 204, or 503 after `CANCEL_LOCK_WAIT_S` |
        | Lock held here, no live lease | the claim's holder once it claims, or this route once the lock is free | as the row it lands on | 204, or 503 after `CANCEL_LOCK_WAIT_S` |
        | The same, the claim committing just after this route found no lease | the claim's holder | as the row it lands on | 204 |
        | The same, this request cancelled once the lock passed to it | nobody: the Stop was never recorded | unchanged | nothing; the lock is let go |
        | The same, and the request then abandons its claim | the next claimant | `cancelled` | nothing more; the next `/turn` streams |

        A live segment, on any instance, is stopped through its lease:
        `request_cancel` sets `lease_cancel`, the segment's next renewal
        records it as the Stop of the lease's turn and reports it, and the
        heartbeat stops the turn (`harness.turn.loop.run_segment`). A segment
        streaming on this instance is made to renew at once
        (`renew_lease_now`), so the Stop reaches it without waiting for the
        beat; another instance's reads it within `HEARTBEAT_INTERVAL_S`. A
        Stop that commits after the segment's last renewal is read by the
        release of the lease, under the row lock: a release that finds it
        live on an open turn keeps the lease and closes the turn under it
        before releasing (`StudioStore.release`, `StudioRunner._release`).
        This route answers 204 without waiting for either.

        A request on this instance that holds the lock with no lease live (a
        `/turn` or `/interrupt` before its claim save, or a stream after its
        release) is waited on: the Stop is recorded once the claim lands,
        or taken under the lock once it is free, and a retryable 503 is
        answered after `CANCEL_LOCK_WAIT_S` of wall time.

        With no live segment on record anywhere, this takes the lock,
        reloads, and closes what is left: a turn paused on a question, or
        left running by a segment whose lease lapsed, is closed `cancelled`
        under a lease this route claims (`harness.turn.loop.stop_turn`), its
        record is closed `cancelled`, and the conversation's pending
        interrupts are cleared, so an answer to one is refused. A close that
        fails, such as ending the turn's runs, answers a retryable 503 and
        leaves the turn open with a Stop live, which the next claimant acts
        on (`_stop_open_turn`). A claim of this
        route's that loses to another request decides again from the top,
        within the same `CANCEL_LOCK_WAIT_S`.
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
            if await studio.store.request_cancel(principal, body.conversation_id):
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
            # save, or a stream after its release), wait for its claim to
            # commit or for the lock, then decide again. Past the deadline
            # a free lock is still taken.
            if await _lock_or_claim(lock, claimed, timeout=deadline - loop.time()):
                try:
                    return await _stop_under_lock(studio, principal, body.conversation_id)
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

    @router.post("/interrupt")
    async def interrupt(  # noqa: PLR0911 -- one distinct refusal per claim guard the route enforces
        body: InterruptResponse, authorization: str | None = Header(default=None)
    ) -> Response:
        principal = principal_for(authorization, settings=settings)
        lock = await _acquire_or_conflict(studio.store, principal, body.conversation_id)
        if isinstance(lock, JSONResponse):
            return lock
        try:
            conversation = await studio.store.load(principal, body.conversation_id)
            if conversation is None:
                lock.release()
                return _unknown_conversation_response(body.conversation_id)
            if await studio.store.stop_live(principal, body.conversation_id):
                # A Stop live for the turn ends it first, which leaves no
                # pause to answer.
                return await _answer_after_stop(
                    studio,
                    principal,
                    body.conversation_id,
                    lock,
                    _no_pending_interrupt_response(body.interrupt_id),
                )
            pending = conversation.pending.get(body.interrupt_id)
            refusal = _interrupt_refusal(conversation.pending, body)
            if refusal is not None:
                lock.release()
                return refusal
            assert pending is not None  # narrowed by _interrupt_refusal
            # Every change up to the claim save is made on a working copy,
            # which the save swaps into the live state once it commits. A
            # request that claims nothing leaves the live state as loaded.
            work = studio.store.draft(principal, body.conversation_id)
            pending = work.pending[body.interrupt_id]
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
            # The answer reaches the paused node's `resume`. A retry of an
            # answer the turn already took finds the turn running and carries
            # it on from its cursor: the node still owes the answer its
            # `resume`, or the turn moved past it; a retried rejection finds
            # the turn already closed. Anything else answers no pause, and is
            # refused before the save below records an approval of it or
            # marks it resuming.
            cursor = await studio.turns.open_cursor(principal, body.conversation_id)
            paused_here = (
                cursor is not None
                and cursor.status is TurnStatus.PAUSED
                and cursor.pause is not None
                and cursor.pause.interrupt_id == pending.adk_id
            )
            retried = resuming and (
                body.decision is InterruptDecision.REJECT
                or (cursor is not None and cursor.status is TurnStatus.RUNNING)
            )
            if not (paused_here or retried):
                return await _unclaimed(
                    studio.store,
                    principal,
                    body.conversation_id,
                    lock,
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
                    lock,
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
                    studio.store,
                    principal,
                    body.conversation_id,
                    lock,
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
            # The resumed segment opens its own telemetry span on the turn's
            # trace, whichever instance runs it; the save hands it to the
            # live state.
            work.open_segment_span()
            # The claim is saved alone, before the answer is recorded: a
            # double answer to the same interrupt across two instances loses
            # cleanly here, with a 409, rather than each resuming its own
            # segment of the turn.
            try:
                await studio.store.save(principal, body.conversation_id, work)
            except StopPending:
                # A Stop that landed since the check above ends the turn
                # first, which leaves no pause to answer.
                return await _answer_after_stop(
                    studio,
                    principal,
                    body.conversation_id,
                    lock,
                    _no_pending_interrupt_response(body.interrupt_id),
                )
            except ConversationStale:
                return await _unclaimed(
                    studio.store,
                    principal,
                    body.conversation_id,
                    lock,
                    _streaming_conflict_response("another request already answered this interrupt"),
                )
        except BaseException:
            if lock.locked():
                await _release_claim(studio.store, principal, body.conversation_id, lock, None)
            raise

        # Every failure from here to the stream's start drops the lease the
        # save above claimed, and releases the lock. Only the lease: once the
        # answer is saved below, the pending entry stays resuming under the
        # dropped segment with the answer recorded, so a retry of the same
        # answer claims at once and resumes. The turn lines added below stay
        # unsaved, and the retry's load reads the row afresh.
        try:
            # The read above only gated the claim. A segment whose lease
            # lapsed can still write the turn until the claim commits, so
            # the answer runs on the cursor read after it, and is refused
            # when that cursor no longer holds the same open pause. Nothing
            # of the answer is saved yet, so the refusal leaves the entry
            # to be answered again as the reloaded conversation shows it.
            claimed = await studio.turns.open_cursor(principal, body.conversation_id)
            if not _same_open_pause(cursor, claimed):
                await _release_claim(
                    studio.store, principal, body.conversation_id, lock, segment_id
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
            pending = work.pending[body.interrupt_id]
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
            # The entry is kept, marked resuming under this fresh segment.
            # A crash between this save and the resumed stream's own
            # checkpoint still leaves a durable, retryable record.
            # `_checkpointed` drops it once the resumed segment reaches its
            # own pause or terminal frame; `release` drops it on every
            # other ending (error, cancel, disconnect); only a crash
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
            try:
                await studio.store.save(principal, body.conversation_id, work)
            except ConversationStale:
                # The lease this request holds refuses every other save, so
                # only a takeover after it lapsed moves the conversation
                # here; nothing of the answer was stored.
                await _release_claim(
                    studio.store, principal, body.conversation_id, lock, segment_id
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
            # undo (`_PendingInterrupt.step_order`'s docstring). It is set
            # before anything below can run the graph or close the turn.
            if studio.surface_recorder is not None:
                studio.surface_recorder.set_step_order(
                    f"{principal}/{body.conversation_id}", pending.step_order, pending.step_labels
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

                return StreamingResponse(
                    studio._segment_stream(
                        principal, body.conversation_id, conversation, segment_id, _rejected()
                    ),
                    media_type=SSE_MEDIA_TYPE,
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
            # The user's words as amended, so a search after the resume reads the
            # refined ask and no switch's label.
            _set_turn_context(principal, body.conversation_id, conversation, settings)
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
                cursor=cursor,
                segment_id=segment_id,
                answer=(
                    Answer(interrupt_id=pending.adk_id, round=cursor.pause.round, value=answer)
                    if paused_here and cursor.pause is not None
                    else None
                ),
                approving=approving,
            )
            return StreamingResponse(
                studio._settling(
                    studio._segment_stream(
                        principal, body.conversation_id, conversation, segment_id, frames
                    )
                ),
                media_type=SSE_MEDIA_TYPE,
            )
        except BaseException as exc:
            if lock.locked():
                await _release_claim(
                    studio.store, principal, body.conversation_id, lock, segment_id, cause=exc
                )
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
