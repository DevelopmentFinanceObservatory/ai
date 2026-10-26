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
is served on demand by the recall seam (`make_recall_part`). Interrupts resume
the current turn's session.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import AsyncGenerator, AsyncIterator, Callable, Coroutine
from contextlib import AbstractContextManager, aclosing, nullcontext, suppress
from datetime import UTC, datetime
from typing import Any

import anyio
from common.telemetry import flush_telemetry
from compiler.surface import ToolSurface
from fastapi import APIRouter, Header, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from fastapi.routing import APIRoute
from google.adk.agents.run_config import RunConfig, StreamingMode
from google.adk.apps import App
from google.adk.errors import StaleSessionError
from google.adk.errors.already_exists_error import AlreadyExistsError
from google.adk.events import Event
from google.adk.runners import Runner
from google.adk.workflow.utils._workflow_hitl_utils import (
    create_request_input_response,
)
from google.genai import types
from planner.plan import SEEDS_STATE_KEY, current_earlier_block, current_recorded_identity

from harness.approval import ApprovalDecision, ApprovalRecord
from harness.auth import principal_for
from harness.conversation import (
    current_conversation_id,
    current_conversation_key,
    current_user_message,
)
from harness.graph.seams import Executor
from harness.graph.state import (
    AGENT_NAME,
    ORCHESTRATOR_PLAN_RECORDS_KEY,
    ORCHESTRATOR_PRIOR_LEDGER_KEY,
    ORCHESTRATOR_SEARCHES_KEY,
    ORCHESTRATOR_SPECS_KEY,
)
from harness.provenance import ProvenanceStore
from harness.seeds import seeds_slot
from harness.sentences import failure_sentence
from harness.settings import Settings
from harness.studio_edit import edit_frames
from harness.studio_state import (
    HEARTBEAT_INTERVAL_S,
    ConversationState,
    SegmentBusy,
    SegmentLost,
    StudioStore,
    _apply_operations,
    _next_manifest,
    _PendingInterrupt,
    _PendingQuestion,
    _tool_error_response,
)
from harness.studio_translate import (
    _RecordingExecutor,
    _RecordingSurface,
    _stream_fatal,
    _Translator,
    failure_detail,
)
from harness.tool_contracts import HarnessCode, harness_refusal
from harness.turn_budget import start_turn_budget
from harness.wire import (
    MAX_TURN_REQUEST_BYTES,
    ActivityStep,
    AddOp,
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
    pending: _PendingInterrupt | None, body: InterruptResponse
) -> JSONResponse | None:
    """Everything that can be wrong with an interrupt.response, as one HTTP
    refusal: unknown id, a decision foreign to the interrupt kind, an
    elicitation's answers (`_elicitation_refusal`), the plan payload
    discipline (delta 12: amend carries the text it acts on, the other
    decisions must not smuggle one), and a switch's revision and option
    (`_switch_refusal`)."""
    if pending is None:
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
    (`_PendingInterrupt.choices`) to an option the card offers as available
    or partial, other than the active one, or is refused 422
    `switch_option_unavailable`. `detail` is copy the card shows.
    """
    if body.plan_revision != pending.plan_revision:
        return _tool_error_response(
            ToolError(
                error=ErrorCode.EDIT_CONFLICT,
                retry=RetryMode.REFRESH_THEN_RETRY,
                constraint_id="plan_revision_stale",
                detail="This plan has changed since you chose. Choose again on the plan shown now.",
                fix="reload the conversation and switch on its current plan",
            ),
            status_code=409,
        )
    choice = pending.choices.get(body.choice_id or "")
    option = choice.options.get(body.option_id or "") if choice is not None else None
    if choice is None:
        detail = "That choice cannot be switched on this plan."
    elif option is None:
        detail = "That option is not offered for this choice."
    elif body.option_id == choice.active:
        detail = f"The plan already counts {option.label}."
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
    heartbeat_seconds: float = HEARTBEAT_SECONDS,
) -> AsyncIterator[str]:
    """Serialise frames as SSE `data:` lines, one loop owning all three
    transport concerns: `: ka` comment heartbeats while the next frame is
    slow; cancellation via the conversation's cancel event when one is
    supplied; and client disconnect, which unwinds the same way minus the
    frame nobody is left to read. A disconnect records a running turn as
    cancelled (`close_turn`). A turn that is `awaiting_answer` (paused on an
    interrupt) is left open, so `/interrupt` still resumes it inside the
    same turn span and trace, and `/cancel` or the next `begin_turn` still
    closes it out.

    The cancel event fires for two different reasons, told apart by
    `conversation.cancel_reason` (set alongside the event by whoever armed
    it, `POST /cancel` or `StudioRunner._locked`'s heartbeat). `"cancelled"`
    (a client's Stop, from this instance or another) closes the turn and
    yields the stream-fatal `cancelled` error frame, same as always.
    `"segment_lost"` (this stream's segment lease was taken over, or this
    instance's own renewal could not keep it alive) yields a stream-fatal
    `transient_unavailable` frame without closing the turn. Another
    instance now owns the segment and may still complete it. Either way the
    segment generator is closed, unwinding the ADK run.

    A Stop, a `segment_lost` transient-unavailable frame, or a stream-fatal
    `error` frame from `frames`, first closes every Activity line the
    stream left open (`_OpenSteps`) with a `failed` frame, since no step
    will report on THIS stream again -- a resumed segment reports the same
    `step_id` again, and the client's line-patching keys on it. A source
    error frame is then re-stamped with the next `seq`, so `seq` still
    rises along the stream. A disconnect sends nothing: the turn is
    closed, and the segment's own `finally` discards its unsent step
    records.

    A stream that has a `conversation` runs inside the turn's telemetry span
    (`ConversationState.in_turn_span`), so a turn resumed by a second request
    exports as one trace, and the task that produces each frame is created
    inside that scope. The scope is left on the task that iterates the
    response. When the response task is cancelled while this generator sits at
    a `yield`, Starlette abandons the generator and the event loop closes it
    later from another context, where leaving the scope detaches nothing
    (`common.telemetry.use_turn_span`). On either path the finished spans are
    exported (`flush_telemetry`) as the stream ends, while the request still
    has CPU."""
    turn_span: AbstractContextManager[None] = (
        conversation.in_turn_span() if conversation is not None else nullcontext()
    )
    cancel_wait: asyncio.Future[Any] | None = None
    open_steps = _OpenSteps()
    if conversation is not None:
        conversation.cancel_event = asyncio.Event()
        cancel_wait = asyncio.ensure_future(conversation.cancel_event.wait())
    with turn_span:
        try:
            async with aclosing(frames) as source:
                iterator = source.__aiter__()
                while True:
                    next_frame = asyncio.ensure_future(anext(iterator))
                    try:
                        while True:
                            race = {next_frame, cancel_wait} if cancel_wait else {next_frame}
                            done, _ = await asyncio.wait(
                                race,
                                timeout=heartbeat_seconds,
                                return_when=asyncio.FIRST_COMPLETED,
                            )
                            if next_frame in done:
                                break
                            if cancel_wait is not None and cancel_wait in done:
                                await _unwind(next_frame)
                                assert conversation is not None
                                if conversation.cancel_reason == "segment_lost":
                                    # Another instance took the segment over.
                                    lost = failure_sentence(
                                        ErrorCode.TRANSIENT_UNAVAILABLE,
                                        RetryMode.RETRY_AFTER_DELAY,
                                    )
                                    for closing in open_steps.close(conversation, detail=lost):
                                        yield _sse_data(closing)
                                    yield _sse_data(
                                        ErrorFrame(
                                            event="error",
                                            seq=conversation.next_seq(),
                                            error=ToolError(
                                                error=ErrorCode.TRANSIENT_UNAVAILABLE,
                                                retry=RetryMode.RETRY_AFTER_DELAY,
                                                detail=lost,
                                                retry_after_seconds=30,
                                            ),
                                        )
                                    )
                                    return
                                conversation.close_turn(
                                    final_text=conversation.turn_final_text,
                                    error=ErrorCode.CANCELLED.value,
                                )
                                stopped = failure_sentence(
                                    ErrorCode.CANCELLED, RetryMode.DO_NOT_RETRY
                                )
                                for closing in open_steps.close(conversation, detail=stopped):
                                    yield _sse_data(closing)
                                yield _sse_data(
                                    ErrorFrame(
                                        event="error",
                                        seq=conversation.next_seq(),
                                        error=ToolError(
                                            error=ErrorCode.CANCELLED,
                                            retry=RetryMode.DO_NOT_RETRY,
                                            detail=stopped,
                                        ),
                                    )
                                )
                                return
                            yield ": ka\n\n"
                    except (asyncio.CancelledError, GeneratorExit):
                        # CancelledError: the response task was cancelled.
                        # GeneratorExit: the consumer closed us AT the heartbeat
                        # yield above. Without unwinding here, aclosing(source)
                        # would find the anext still running and raise. Either
                        # way the client is gone. A running turn is recorded as
                        # cancelled (a completed turn's close_turn already ran
                        # and this is a no-op). A turn paused for an answer
                        # stays open: its `plan.proposed` or elicitation frame
                        # was sent, `/interrupt` can still resume it, and
                        # closing it here would end the turn span and put the
                        # resumed segment on a trace of its own.
                        if conversation is not None and not conversation.awaiting_answer:
                            conversation.close_turn(
                                final_text=conversation.turn_final_text,
                                error=ErrorCode.CANCELLED.value,
                            )
                        await _unwind(next_frame)
                        raise
                    try:
                        frame = next_frame.result()
                    except StopAsyncIteration:
                        return
                    if conversation is not None and isinstance(frame, ErrorFrame):
                        closed = open_steps.close(conversation, detail=frame.error.detail)
                        for closing in closed:
                            yield _sse_data(closing)
                        if closed:
                            frame = frame.model_copy(update={"seq": conversation.next_seq()})
                    open_steps.observe(frame)
                    yield _sse_data(frame)
        finally:
            if cancel_wait is not None:
                await _unwind(cancel_wait)
            if conversation is not None:
                conversation.cancel_event = None
                conversation.cancel_reason = None
            await flush_telemetry()


def _keep_turn_records(conversation: ConversationState, event: Event) -> None:
    """Keep the plan records, search readings and spec blocks `event` writes
    to the turn's session state, for `ConversationState.close_turn` to store
    with the turn. Each key holds the turn's whole value, so the latest
    write replaces. The blocks stay in process (`turn_specs`) until the turn
    closes."""
    delta = event.actions.state_delta
    if ORCHESTRATOR_PLAN_RECORDS_KEY in delta:
        conversation.turn_plans = list(delta[ORCHESTRATOR_PLAN_RECORDS_KEY])
    if ORCHESTRATOR_SEARCHES_KEY in delta:
        conversation.turn_readings = list(delta[ORCHESTRATOR_SEARCHES_KEY])
    if ORCHESTRATOR_SPECS_KEY in delta:
        conversation.turn_specs = dict(delta[ORCHESTRATOR_SPECS_KEY])


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
    aiosqlite leaves a task that never finishes."""
    pending.cancel()
    with anyio.CancelScope(shield=True), suppress(asyncio.CancelledError):
        await pending


# The frames `StudioRunner._checkpointed` durably saves the conversation
# before yielding: the two actionable pauses and the turn's own terminal
# success. `message.complete` is not in this set. It precedes `done` on
# the SAME segment when a turn ends outright, and checkpointing twice one
# frame apart buys nothing (`ElicitationRequest`'s and `PlanProposed`'s own
# frame carries no `message.complete` of its own).
_CHECKPOINT_BEFORE_EVENTS = frozenset({"plan.proposed", "elicitation.request", "done"})


def _end_stream(conversation: ConversationState, reason: str) -> None:
    """Arm `conversation.cancel_event` with `reason`, read back by
    `_sse_stream`'s cancel branch to pick the frame. A no-op when no stream
    is currently armed to receive it (the segment ended some other way in
    the gap between the heartbeat's check and this call)."""
    conversation.cancel_reason = reason
    if conversation.cancel_event is not None:
        conversation.cancel_event.set()


def _checkpoint_failed_frame(conversation: ConversationState, exc: Exception) -> ErrorFrame:
    """The stream-fatal frame `StudioRunner._checkpointed` yields when a
    checkpoint before a pause or terminal frame fails, in place of the
    frame it would have carried. A `SegmentLost` means another instance's
    claim has already outlived this one's TTL and now owns the
    conversation, so the turn is left open for it rather than closed here;
    anything else (an exhausted retry, a DB error) closes the turn, since
    nothing else will finish it. The turn's record names `state_not_saved`,
    and the frame, whose grammar has no such member, the transient code."""
    refusal = harness_refusal(
        code=HarnessCode.STATE_NOT_SAVED, message=f"the turn's state could not be saved: {exc}"
    )
    detail = failure_detail(refusal)
    if not isinstance(exc, SegmentLost):
        conversation.close_turn(
            final_text=conversation.turn_final_text or detail, error=refusal["code"]
        )
    return ErrorFrame(
        event="error",
        seq=conversation.next_seq(),
        error=ToolError(
            error=ErrorCode.TRANSIENT_UNAVAILABLE,
            retry=RetryMode.RETRY_AFTER_DELAY,
            detail=detail,
            retry_after_seconds=30,
        ),
    )


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
    is ever reached.
    """

    def __init__(
        self,
        *,
        create_workflow: Any,
        executor: Executor,
        session_service: Any,
        tool_surface: ToolSurface | None = None,
        store: StudioStore | None = None,
        app_name: str | None = None,
        edit_model: Any | None = None,
        provenance: ProvenanceStore | None = None,
    ) -> None:
        self.edit_model = edit_model
        # Resolved before the workflow exists (`AGENT_NAME` is the fixed
        # name every `create_root_workflow` call gives its `Workflow`, so
        # this matches `workflow.name` in every real caller) because the
        # store must be built -- for the recorder to bind, below -- before
        # `create_workflow` runs, which is the only place `workflow.name`
        # itself becomes available.
        resolved_app_name = app_name or AGENT_NAME
        # Built before the recorder: `_RecordingExecutor` binds it too, to
        # keep a create's write token (`harness.studio_translate`) on the
        # same conversations the rest of this runner reads and writes. A
        # caller giving its own `store` (`harness.server`, so `make_recall_part` can bind it before
        # this constructor runs) is responsible for building it over the
        # same `session_service`.
        self.store = store or StudioStore(session_service, app_name=resolved_app_name)
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

    async def _ensure_session(
        self, *, user_id: str, session_id: str, state: dict[str, Any] | None = None
    ) -> None:
        """Create the turn's per-turn ADK session (`{cid}#t{n}`). Create-only:
        `session_id` names a fresh turn index the `/turn` route's ledger
        claim has just won, so this session must not already exist. Raises
        `AlreadyExistsError` when it does -- another instance claimed the
        same turn index concurrently -- for the caller to answer as the
        existing 409 `TRANSIENT_UNAVAILABLE`, the same as a stale claim."""
        await self.runner.session_service.create_session(
            app_name=self.runner.app_name,
            user_id=user_id,
            session_id=session_id,
            state=state,
        )

    async def _frames(
        self,
        *,
        principal: str,
        conversation: ConversationState,
        message: types.Content,
    ) -> AsyncGenerator[Any]:
        """Run one segment (fresh turn or resume) and yield wire frames.

        Failures are serialised into the stream as a stream-fatal `error`
        frame. SSE headers are already on the wire, so an exception can't
        become an HTTP status. The exception is logged with the trace id;
        it is not re-raised into the transport.
        """
        translator = _Translator(
            conversation=conversation,
            recorder=self.recorder,
            surface_recorder=self.surface_recorder,
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
                    _keep_turn_records(conversation, event)
                    for frame in translator.translate(event):
                        yield frame
            for frame in translator.finish():
                yield frame
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
        with that reason. `_sse_stream`'s cancel branch turns it into the
        matching frame, one mechanism for every kind of mid-stream stop.

        A `renew` call that raises is retried every interval until
        `SEGMENT_TTL_S - 10` seconds of the segment's own remaining life are
        left, then also ends the stream as lost: an instance this deaf to
        its own renewal cannot be trusted to still own the segment, even
        though nothing else has claimed it yet.
        """
        while True:
            await asyncio.sleep(HEARTBEAT_INTERVAL_S)
            try:
                result = await self.store.renew(principal, conversation_id, segment_id)
            except Exception:
                logger.exception(
                    "studio segment heartbeat: conversation %s segment %s renewal failed",
                    conversation_id,
                    segment_id,
                )
                segment = conversation.segment
                if segment is None or segment.id != segment_id:
                    return
                if self.store.now() < segment.expires_at - 10:
                    continue
                _end_stream(conversation, "segment_lost")
                return
            if result == "ok":
                continue
            _end_stream(conversation, "cancelled" if result == "cancelled" else "segment_lost")
            return

    async def _checkpointed(
        self,
        principal: str,
        conversation_id: str,
        segment_id: str,
        conversation: ConversationState,
        frames: AsyncGenerator[Any],
    ) -> AsyncGenerator[Any]:
        """Relay `frames`, checkpointing durably before every actionable
        pause (`plan.proposed`, `elicitation.request`) and before `done`.
        The client never sees a frame that says "paused" or "done" before
        the ledger agrees. `done`'s frame is rebuilt to carry the
        post-checkpoint `conversation.composition`, so a concurrent
        `/composition` edit merged in by the checkpoint reaches the client
        on the same frame it reaches the ledger. A checkpoint that fails
        (the segment was taken over, or the write itself failed) yields a
        stream-fatal error frame instead of the pause or terminal frame it
        would have carried, and ends the stream there. This is the one
        interception point for every route this runs under (`/turn`, its
        edit-card branch, and `/interrupt`)."""
        async for frame in frames:
            event = getattr(frame, "event", None)
            outgoing = frame
            if event in _CHECKPOINT_BEFORE_EVENTS:
                conversation.drop_resuming(segment_id)
                try:
                    await self.store.checkpoint(principal, conversation_id, segment_id)
                except Exception as exc:
                    logger.exception(
                        "studio checkpoint failed: conversation %s segment %s (trace_id=%s)",
                        conversation_id,
                        segment_id,
                        conversation.trace_id,
                    )
                    yield _checkpoint_failed_frame(conversation, exc)
                    return
                if event == "done":
                    outgoing = frame.model_copy(update={"composition": conversation.composition})
            yield outgoing

    async def _locked(
        self,
        principal: str,
        conversation_id: str,
        conversation: ConversationState,
        segment_id: str,
        frames: AsyncGenerator[Any],
    ) -> AsyncGenerator[Any]:
        """Stream frames, checkpointing before any that report success
        (`_checkpointed`), then release the segment and the conversation
        lock.

        The ROUTE acquires the lock (eagerly, before touching turn state), so
        ownership is settled while the request handler still controls the
        response. A lazily-acquired lock let two requests pass the 409
        guard and clobber each other's turn state and cancel event.

        A heartbeat task renews `segment_id` for as long as this generator
        is live, so the segment's lease outlives a single checkpoint's TTL
        push; it is cancelled and awaited before release, so it never races
        `store.release`'s own write. Unwinding it, releasing the segment
        (and this segment's resuming pending entries), and releasing the
        local lock each sit in their own nested `finally`, so a heartbeat
        that ends with a non-cancel exception still reaches both releases;
        `_unwind` re-raises that exception once they have run. A DB error
        `store.release` cannot itself raise (it never does; see
        `StudioStore.release`), and still could not strand the lock, since
        releasing it does not depend on the release call having succeeded.
        """
        heartbeat = asyncio.ensure_future(
            self._heartbeat(principal, conversation_id, segment_id, conversation)
        )
        try:
            async for frame in self._checkpointed(
                principal, conversation_id, segment_id, conversation, frames
            ):
                yield frame
        finally:
            try:
                await _unwind(heartbeat)
            finally:
                try:
                    await self.store.release(principal, conversation_id, segment_id)
                finally:
                    conversation.lock.release()


def _streaming_conflict_response(detail: str, *, retry_after_seconds: int = 5) -> JSONResponse:
    """The 409 `TRANSIENT_UNAVAILABLE` every claim conflict answers with,
    whichever guard caught it: the process-local lock (`_acquire_or_conflict`),
    a live segment lease held by any instance (`SegmentBusy`), the ledger's
    compare-and-swap on the `/turn` claim (`StaleSessionError`), or a
    concurrent claim on the per-turn ADK session (`AlreadyExistsError`).
    `retry_after_seconds` defaults to a flat 5s for the process-local and
    session-create guards, which carry no TTL of their own to report; a
    `SegmentBusy` conflict passes its own remaining TTL instead."""
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


def create_studio_router(studio: StudioRunner, *, settings: Settings) -> APIRouter:
    router = APIRouter(prefix="/api/studio")

    async def turn(  # noqa: PLR0911 -- one distinct refusal per claim guard the route enforces
        body: TurnRequest, authorization: str | None = Header(default=None)
    ) -> Response:
        principal = principal_for(authorization, settings=settings)
        lock = await _acquire_or_conflict(studio.store, principal, body.conversation_id)
        if isinstance(lock, JSONResponse):
            return lock
        try:
            conversation = await studio.store.load(principal, body.conversation_id)
            if conversation is None:
                if body.component_ref is not None:
                    # The client refers to state (a card to edit, or to
                    # follow up on) this store never had.
                    lock.release()
                    return _unknown_conversation_response(body.conversation_id)
                try:
                    conversation = await studio.store.create(principal, body.conversation_id)
                except AlreadyExistsError:
                    # Another instance's `/turn` created this same
                    # never-before-seen conversation between this request's
                    # own load and create. Load what it created and continue
                    # to the claim below -- its compare-and-swap decides the
                    # race; a lost claim is the existing `StaleSessionError`
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
            conversation.begin_turn()
            conversation.auto_add = body.auto_add
            conversation.turn_user_message = body.message
            conversation.turn_user_words = body.message
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
            # The segment lease: claimed on the live state before the claim
            # save below persists it, so the save's own compare-and-swap
            # covers both in one write. `SegmentBusy` means a live segment,
            # on this instance or another, already owns the conversation's
            # stream; its own `retry_after_seconds` is the ceiling of that
            # segment's remaining TTL.
            try:
                segment_id = studio.store.claim_segment(principal, body.conversation_id)
            except SegmentBusy as exc:
                lock.release()
                return _streaming_conflict_response(
                    "another request is streaming on this conversation",
                    retry_after_seconds=exc.retry_after_seconds,
                )
            # The claim: advancing turn_index (and the segment above) is
            # saved before the per-turn ADK session is created and before
            # the SSE response starts. This save is a compare-and-swap
            # against what this instance last loaded, so a
            # `StaleSessionError` here means another instance's `/turn`
            # claimed a turn on this conversation since, the same case as
            # the `AlreadyExistsError` from the session create below.
            # Claiming first leaves no crash window: a crash between the
            # claim and the create leaves the ledger at the new turn_index
            # with no matching session, and the next `/turn` claims the one
            # after it; the segment it also claimed simply expires by its
            # own TTL.
            try:
                await studio.store.save(principal, body.conversation_id)
            except StaleSessionError:
                lock.release()
                return _streaming_conflict_response(
                    "another request is streaming on this conversation"
                )
            if (
                body.component_ref is not None
                and body.component_ref.intent is ComponentRefIntent.EDIT
            ):
                # The Edit Card route bypasses the graph entirely: no
                # resolve, no plan, no ADK session, just the card, the
                # model's patch-or-recreate decision, and the platform write.
                # It streams under the same segment the claim save above
                # just persisted.
                return StreamingResponse(
                    _sse_stream(
                        studio._locked(
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
                        ),
                        conversation=conversation,
                    ),
                    media_type=SSE_MEDIA_TYPE,
                )
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
            # The orchestrator's planner renders the earlier turns' readings;
            # their blocks stay in the conversation (`current_earlier_block`).
            ledger = conversation.prior_ledger_state()
            if ledger:
                state[ORCHESTRATOR_PRIOR_LEDGER_KEY] = ledger
            if isinstance(body.seed_spec, list):
                # Each seed as the `seed` node takes it, or the refusal its
                # step fails with: no posted Spec block enters the session.
                state[SEEDS_STATE_KEY] = json.dumps(seeds_slot(body.seed_spec))
            elif body.seed_spec is not None:
                # One reference, the reserved shape: `graph.seed` reports it unused.
                state[SEEDS_STATE_KEY] = json.dumps(body.seed_spec.model_dump(mode="json"))
            try:
                await studio._ensure_session(
                    user_id=principal, session_id=conversation.session_id, state=state
                )
            except AlreadyExistsError:
                lock.release()
                return _streaming_conflict_response(
                    "another request is streaming on this conversation"
                )
            message = types.Content(role="user", parts=[types.Part.from_text(text=body.message)])
            return StreamingResponse(
                _sse_stream(
                    studio._locked(
                        principal,
                        body.conversation_id,
                        conversation,
                        segment_id,
                        studio._frames(
                            principal=principal, conversation=conversation, message=message
                        ),
                    ),
                    conversation=conversation,
                ),
                media_type=SSE_MEDIA_TYPE,
            )
        except BaseException:
            # The lock is normally released by the stream wrapper; a failure
            # before the response starts streaming must release it here.
            # Nothing durable changed since the claim save above that this
            # request is responsible for flushing -- the claim already
            # covers `begin_turn`'s effects and the segment it claimed
            # alongside it, which simply expires by its own TTL.
            if lock.locked():
                lock.release()
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
        invalidate a pause the ledger might be holding.
        """
        principal = principal_for(authorization, settings=settings)
        live = studio.store.current(principal, body.conversation_id)
        if live is not None and live.cancel_event is not None:
            live.cancel_reason = "cancelled"
            live.cancel_event.set()
            return Response(status_code=204)
        if await studio.store.request_cancel(principal, body.conversation_id):
            return Response(status_code=204)
        lock = studio.store.lock_for(principal, body.conversation_id)
        if lock.locked():
            return Response(status_code=204)
        await lock.acquire()
        try:
            conversation = await studio.store.load(principal, body.conversation_id)
            if conversation is None:
                return Response(status_code=204)
            closed_here = False
            if conversation.awaiting_answer:
                conversation.close_turn(
                    final_text=conversation.turn_final_text, error=ErrorCode.CANCELLED.value
                )
                closed_here = True
            # The pending interrupts are cleared unconditionally, the same
            # abandonment `begin_turn` already does for a fresh turn on a
            # paused conversation; the parked ADK invocation needs nothing
            # beyond that, since it stays in its own per-turn session
            # either way.
            conversation.pending.clear()
            # No segment is live here (`request_cancel` above found none),
            # so there is nothing for `release` to fence against; a plain
            # save persists the clear. A stale write (a segment claimed the
            # instant after `request_cancel` checked) is a rare enough race
            # that losing this save is acceptable. The claiming stream's
            # own release still checkpoints whatever it leaves behind.
            try:
                await studio.store.save(principal, body.conversation_id)
            except StaleSessionError:
                logger.warning(
                    "studio cancel: conversation %s changed underneath this save; nothing "
                    "persisted",
                    body.conversation_id,
                )
            if closed_here:
                # No stream is left to export the turn span that was just ended.
                await flush_telemetry()
        finally:
            lock.release()
        return Response(status_code=204)

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
            pending = conversation.pending.get(body.interrupt_id)
            refusal = _interrupt_refusal(pending, body)
            if refusal is not None:
                lock.release()
                return refusal
            assert pending is not None  # narrowed by _interrupt_refusal
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
                conversation.approvals.append(
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
                segment_id = studio.store.claim_segment(principal, body.conversation_id)
            except SegmentBusy as exc:
                lock.release()
                return _streaming_conflict_response(
                    "another request is streaming on this conversation",
                    retry_after_seconds=exc.retry_after_seconds,
                )
            if resuming and pending.answer != requested_answer:
                # The claim above only succeeds once the earlier resume's
                # segment has expired (its owner died before reaching a
                # checkpoint), so this is a genuinely dead attempt, and
                # this request's answer does not match what actually
                # resumed it. Refused for good: a live segment above would
                # already have caught a retry of the SAME answer, so a
                # mismatch reaching here can never become a match later.
                lock.release()
                return _tool_error_response(
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
            if resuming:
                # A retry resuming the same answer again: the dead attempt
                # may already have sent frames the client saw, carrying a
                # `seq` this stream must never repeat.
                conversation.seq += 1000
            # Saved before the resume stream starts: a double answer to the
            # same interrupt across two instances loses cleanly here, with a
            # 409, rather than each resuming its own segment of the turn.
            try:
                await studio.store.save(principal, body.conversation_id)
            except StaleSessionError:
                lock.release()
                return _streaming_conflict_response(
                    "another request already answered this interrupt"
                )
        except BaseException:
            if not lock.locked():
                raise
            lock.release()
            raise

        # The instance that resumes a paused turn may not be the one that
        # opened it: its telemetry span is process-local and never persisted,
        # and a workflow's own steps run as concurrent waves whose recording
        # order `_RecordingSurface` needs the plan's own step order to undo
        # (`_PendingInterrupt.step_order`'s docstring). Both are restored
        # before anything below can run the graph or close the turn.
        conversation.ensure_turn_span()
        if studio.surface_recorder is not None:
            studio.surface_recorder.set_step_order(
                f"{principal}/{body.conversation_id}", pending.step_order, pending.step_labels
            )

        if body.decision is InterruptDecision.REJECT:
            # The plan is declined and the turn ends; nothing was computed.
            # The paused ADK invocation is abandoned in its per-turn session.
            async def _rejected() -> AsyncGenerator[Any]:
                text = "Plan rejected; nothing was retrieved or computed."
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
                _sse_stream(
                    studio._locked(
                        principal, body.conversation_id, conversation, segment_id, _rejected()
                    )
                ),
                media_type=SSE_MEDIA_TYPE,
            )

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
            # it again; the turn's history record carries the choice.
            assert body.choice_id is not None and body.option_id is not None
            answer = {
                "approved": None,
                "switch": {"choice_id": body.choice_id, "option_id": body.option_id},
            }
            label = pending.choices[body.choice_id].options[body.option_id].label
            conversation.add_to_turn(f"count {label}", user_words=False)
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
        try:
            part = create_request_input_response(pending.adk_id, {"result": answer})
            message = types.Content(role="user", parts=[part])
            return StreamingResponse(
                _sse_stream(
                    studio._locked(
                        principal,
                        body.conversation_id,
                        conversation,
                        segment_id,
                        studio._frames(
                            principal=principal, conversation=conversation, message=message
                        ),
                    ),
                    conversation=conversation,
                ),
                media_type=SSE_MEDIA_TYPE,
            )
        except BaseException:
            lock.release()
            raise

    @router.post("/composition")
    async def composition(
        body: CompositionMutation, authorization: str | None = Header(default=None)
    ) -> Response:
        # Routed through the composition lock, not the turn lock: a canvas
        # edit must land while a turn streams on the same conversation, so a
        # concurrent turn's own checkpoint replays its composition ops
        # instead of overwriting this route's write. The composition lock
        # also serialises two same-instance `/composition` calls, so they
        # read-mutate-write in turn rather than interleaving.
        principal = principal_for(authorization, settings=settings)
        turn_lock = studio.store.lock_for(principal, body.conversation_id)
        async with studio.store.composition_lock_for(principal, body.conversation_id):
            if turn_lock.locked():
                # A stream holds the turn lock: `entry.state` carries its
                # unsaved turn fields (turn_components, turn_user_message,
                # ...), so this reads it live rather than reloading over
                # them. The composition manifest itself is never read off
                # `conversation` here. `mutate_composition` below fetches
                # and rebuilds it straight from the ledger, so this choice
                # only decides which snapshot of the conversation's OTHER
                # fields (`.components`, read by `_apply_operations` below)
                # the route sees.
                conversation = studio.store.current(principal, body.conversation_id)
            else:
                conversation = await studio.store.load(principal, body.conversation_id)
            if conversation is None:
                return _unknown_conversation_response(body.conversation_id)

            def mutate(
                manifest: CompositionManifest | None,
            ) -> CompositionManifest | None | JSONResponse:
                refusal = _composition_refusal(manifest, body)
                if refusal is not None:
                    return refusal
                # Everything below works on LOCAL copies; the manifest this
                # route hands back to `mutate_composition` is built once,
                # after every operation validated. A refused op must not
                # leave a half-applied canvas behind (mutations are
                # transactional).
                applied = _apply_operations(conversation, manifest, body.operations)
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

            try:
                result = await studio.store.mutate_composition(
                    principal, body.conversation_id, mutate
                )
            except StaleSessionError:
                # `_COMPOSITION_MUTATE_RETRY_ATTEMPTS` fetch-rebuild-mutate
                # cycles in a row all found another writer had landed first:
                # the same conflict the expected_revision check above
                # answers, just caught after sustained contention instead
                # of at the first read.
                logger.warning(
                    "studio composition mutate: conversation %s stayed stale after retries; "
                    "refusing as an edit conflict",
                    body.conversation_id,
                )
                return _tool_error_response(
                    ToolError(
                        error=ErrorCode.EDIT_CONFLICT,
                        retry=RetryMode.REFRESH_THEN_RETRY,
                        detail="the composition changed since this request loaded it",
                        fix="re-render from the current manifest and retry",
                    ),
                    status_code=409,
                )
            if isinstance(result, JSONResponse):
                return result
            assert result is not None  # every non-refusal `mutate` return here is a manifest
            return JSONResponse(content=json.loads(result.model_dump_json(exclude_none=True)))

    return router
