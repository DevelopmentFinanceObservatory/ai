"""Ask the studio harness a question from a terminal and read the answer.

Drives one turn end to end: POST /api/studio/turn, render every frame as it
streams, and when the turn pauses on a plan or an elicitation, prompt for a
decision and resume on POST /api/studio/interrupt -- no interrupt_id to copy
by hand. The stream ending on a pause frame is the contract, not a hang.

    uv run --package harness python scripts/ask.py "how has Ethiopia's health aid changed since 2020?"

`STUDIO_API_TOKEN` supplies the bearer token; a server started with it
unset accepts any non-empty token. `--url` points at a different harness
(default http://localhost:8080).
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import re
import sys
import threading
import time
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlsplit

import httpx2
from harness.wire import (
    STUDIO_FRAME_ADAPTER,
    ActivityStep,
    ComponentCreated,
    ComponentRevised,
    CompositionUpdated,
    Done,
    ElicitationAnswer,
    ElicitationQuestion,
    ElicitationRequest,
    ErrorFrame,
    InterruptDecision,
    InterruptResponse,
    MessageComplete,
    MessageDelta,
    OptionAnswer,
    PlanProposed,
    PlanVersioned,
    SkippedAnswer,
    StudioFrame,
    Suggestions,
    TextAnswer,
    TurnRequest,
)
from pydantic import ValidationError

_DEFAULT_URL = "http://localhost:8080"
# Any non-empty value works against a server started with STUDIO_API_TOKEN
# unset (dev servers run that way); it only has to match when the server
# enforces a real token.
_DEFAULT_TOKEN = "ask-cli"

# A dev server on localhost/127.0.0.1/::1 legitimately runs without TLS; any
# other host must be https, since the bearer token goes on every request.
_LOOPBACK_HOSTS = {"localhost", "127.0.0.1", "::1"}


class StudioError(Exception):
    """A non-200 response from the studio API. The message is the detail."""


def _check_url(url: str) -> None:
    """Refuses a URL that would send the bearer token over cleartext HTTP to
    anything but the local machine. `--url`/`STUDIO_URL` are trusted inputs
    (the person running the CLI chose them), but a typo'd or copy-pasted
    `http://` remote host is exactly the kind of mistake this guards against
    -- the token leaks to the network path, not just to the intended server.
    """
    parts = urlsplit(url)
    if parts.scheme == "https":
        return
    if parts.scheme == "http" and (parts.hostname or "").lower() in _LOOPBACK_HOSTS:
        return
    raise StudioError(
        f"refusing to send the bearer token to {url!r} over plain HTTP to a "
        "non-loopback host -- use https:// for a remote server"
    )


# Matches a full ANSI escape sequence: OSC (ESC ']' ... BEL or ST), CSI
# (ESC '[' ... final byte), or any other two-byte Fe sequence. Used to strip
# escapes from text this CLI did not generate itself.
_ESCAPE_SEQUENCE = re.compile(
    r"\x1b(?:\][^\x07\x1b]*(?:\x07|\x1b\\)|\[[0-?]*[ -/]*[@-~]|[@-Z\\-_])"
)
# C0 control bytes other than tab and newline, which are ordinary text
# formatting a model or the server may legitimately send.
_CONTROL_BYTE = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")


def _strip_control_sequences(value: Any) -> Any:
    """Strips ANSI/OSC escape sequences and stray C0 control bytes from every
    string in a frame payload, recursively, right after it comes off the
    wire and before it is validated into a `StudioFrame` -- this is the one
    place text from a model, a tool, or a remote server enters the program.
    Everything downstream, including `Console.paint`'s own ANSI codes and
    the strings this script composes itself, is trusted output this CLI
    generates and never runs through this filter.
    """
    if isinstance(value, str):
        return _CONTROL_BYTE.sub("", _ESCAPE_SEQUENCE.sub("", value))
    if isinstance(value, dict):
        return {key: _strip_control_sequences(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_strip_control_sequences(item) for item in value]
    return value


@dataclass(frozen=True)
class Console:
    """Writes to stdout, colouring and redrawing only when it looks safe
    to. `tty` gates the live waiting line separately from `color`: a dumb
    terminal (TERM=dumb) still redraws in place but without escape codes
    for colour."""

    color: bool
    tty: bool = False

    def paint(self, text: str, code: str) -> str:
        return f"\033[{code}m{text}\033[0m" if self.color else text

    def write(self, text: str) -> None:
        print(text, end="", flush=True)

    def line(self, text: str = "") -> None:
        print(text, flush=True)

    def waiting(self, text: str) -> None:
        """Redraws the one live status line in place. No-op unless stdout
        is a tty -- piped or redirected output stays quiet."""
        if self.tty:
            print(f"\r\033[K{text}", end="", flush=True)

    def clear_waiting(self) -> None:
        if self.tty:
            print("\r\033[K", end="", flush=True)


@dataclass
class RenderState:
    """Tracks narration in flight across one SSE segment, so a following
    frame starts on its own line and `message.complete` doesn't reprint text
    its deltas already streamed. `failed_steps` holds the step ids whose
    latest frame was `failed`, so a later try's `complete` frame on one of
    them prints its detail under the failed line it settles."""

    mid_line: bool = False
    narrated: bool = False
    failed_steps: set[str] = field(default_factory=set)

    def break_line(self, console: Console) -> None:
        if self.mid_line:
            console.line()
            self.mid_line = False


# Redraw interval for the waiting line. No live model round trip resolves in
# under this, so it never competes with a real frame for the terminal.
_REDRAW_INTERVAL = 0.5


@dataclass
class Waiter:
    """Tracks one status line across the whole turn: how long it has run
    and which phase the stream implies, for the redraw loop to show while
    no frame has arrived. Elapsed counts from the turn's first request, not
    from the last frame -- it answers "is this alive", not "how long was
    the last gap". Also outlives any one SSE segment (unlike `RenderState`,
    which is per-segment), so `unhandled_frame` is a whole-turn flag: once
    `render_frame` hits a variant it has no case for, the turn is marked as
    having dropped content even if a later segment renders cleanly."""

    start: float
    phase: str = "resolving the question"
    detail: str = ""
    unhandled_frame: bool = False

    def elapsed(self) -> int:
        return int(time.monotonic() - self.start)

    def render(self, console: Console) -> str:
        return console.paint(f"  … {self.phase}{self.detail} ({self.elapsed()}s)", "2")

    def note(self, frame: StudioFrame) -> None:
        """Updates the phase from a frame actually rendered, so the next
        redraw (if any) reflects it."""
        match frame:
            case ActivityStep(step_id="resolve"):
                self.phase, self.detail = "planning", ""
            case ActivityStep(progress=progress) if progress is not None:
                self.phase = "executing"
                self.detail = f" ({progress.completed}/{progress.total})"
            case ActivityStep():
                self.phase, self.detail = "executing", ""
            case PlanVersioned():
                self.phase, self.detail = "revising the plan", ""

    def note_decision(
        self, pause: PlanProposed | ElicitationRequest, decision: InterruptDecision
    ) -> None:
        """Updates the phase for the segment about to open after a decision
        is sent, since it may start silent (e.g. a re-plan)."""
        if isinstance(pause, PlanProposed) and decision in (
            InterruptDecision.REJECT,
            InterruptDecision.AMEND,
        ):
            self.phase, self.detail = "revising the plan", ""
        elif isinstance(pause, PlanProposed):
            self.phase, self.detail = "executing", ""
        else:
            self.phase, self.detail = "resolving the question", ""


def _color_enabled() -> bool:
    return (
        sys.stdout.isatty()
        and os.environ.get("NO_COLOR") is None
        and os.environ.get("TERM") != "dumb"
    )


def _short(identifier: str) -> str:
    return identifier[:12]


def _component_title(payload: dict[str, Any]) -> str | None:
    """The resolved revision's `component.title.text`, as
    `harness.studio_state._block_title` reads it. Returns `None` for a
    member that carries no title (a note, a source card)."""
    component = payload.get("component")
    if isinstance(component, dict):
        title = component.get("title")
        if isinstance(title, dict) and isinstance(title.get("text"), str):
            return title["text"]
    return None


# ---------------------------------------------------------------------------
# Frame rendering
# ---------------------------------------------------------------------------


def _render_plan(frame: PlanProposed, console: Console) -> None:
    console.line()
    header = f"Plan: {frame.title} (v{frame.version}, rev {_short(frame.revision or frame.plan_version_id)})"
    console.line(console.paint(header, "1;36"))
    if frame.decisions:
        console.line(console.paint("  Decisions:", "2"))
        for decision in frame.decisions:
            console.line(f"    - {decision.question} -> {decision.chosen}")
            console.line(
                console.paint(f"      ({decision.resolved_by.value}: {decision.reason})", "2")
            )
    if frame.asks:
        labels = {step.id: step.label for step in frame.steps}
        console.line(console.paint("  You asked for:", "2"))
        for ask in frame.asks:
            answers = "; ".join(labels.get(step, step) for step in ask.answered_by)
            if ask.declined is None:
                console.line(f"    - {ask.ask} -> {answers}")
            elif ask.not_computed is not None:
                console.line(
                    f"    - {ask.ask} -> {answers}; not computed: {ask.not_computed} "
                    f"({ask.declined})"
                )
            else:
                console.line(f"    - {ask.ask} -> not computed: {ask.declined}")
    if frame.requirements:
        console.line(console.paint("  Requirements:", "2"))
        for requirement in frame.requirements:
            console.line(f"    - {requirement.sentence}")
    console.line(console.paint("  Steps:", "2"))
    for index, step in enumerate(frame.steps, start=1):
        console.line(f"    {index}. {step.label}")
        if step.description:
            console.line(console.paint(f"       {step.description}", "2"))
    console.line()


def _render_plan_versioned(frame: PlanVersioned, console: Console) -> None:
    header = f"Plan amended -> v{frame.version} (supersedes {_short(frame.supersedes)})"
    console.line(console.paint(header, "33"))
    if frame.changed:
        console.line("  changed: " + ", ".join(frame.changed))


_STEP_SYMBOL = {
    "pending": "·",
    "running": "…",
    "complete": "✓",
    "failed": "✗",
    "skipped": "-",
}
_STEP_COLOR = {"complete": "32", "failed": "1;31", "running": "33"}


def _render_activity_step(frame: ActivityStep, console: Console, state: RenderState) -> None:
    status = frame.status.value
    symbol = _STEP_SYMBOL.get(status, "·")
    progress = f" ({frame.progress.completed}/{frame.progress.total})" if frame.progress else ""
    duration = f" {frame.duration_ms}ms" if frame.duration_ms is not None else ""
    line = f"  {symbol} {frame.label}{progress}{duration}"
    console.line(console.paint(line, _STEP_COLOR.get(status, "2")))
    if status == "failed":
        state.failed_steps.add(frame.step_id)
        if frame.detail:
            console.line(console.paint(f"      {frame.detail}", "31"))
    elif frame.step_id in state.failed_steps:
        state.failed_steps.discard(frame.step_id)
        if status == "complete" and frame.detail:
            console.line(console.paint(f"      {frame.detail}", "32"))


def _render_component_created(frame: ComponentCreated, console: Console) -> None:
    console.line()
    title = _component_title(frame.payload) or frame.component_id
    console.line(console.paint(f"Component: {title}", "1;35"))
    console.line(f"  type: {frame.type.value}  revision: {frame.revision}")
    console.line(f"  {frame.provenance.citation}")
    console.line("  sources: " + ", ".join(source.name for source in frame.provenance.sources))
    freshness = frame.provenance.freshness
    partial = " (partial period)" if freshness.partial_period else ""
    console.line(
        f"  data through {freshness.data_through}, refreshed {freshness.last_refreshed}{partial}"
    )
    for warning in frame.provenance.warnings:
        console.line(console.paint(f"  ! [{warning.severity.value}] {warning.message}", "33"))
    console.line()


def _render_component_revised(frame: ComponentRevised, console: Console) -> None:
    changed = ", ".join(frame.changed)
    console.line(
        console.paint(
            f"Component {frame.component_id} revised -> rev {frame.revision} ({changed})", "35"
        )
    )


def _render_composition_updated(frame: CompositionUpdated, console: Console) -> None:
    manifest = frame.manifest
    console.line(
        console.paint(f"Canvas: {manifest.counts.all} card(s), rev {manifest.revision}", "2")
    )


def _render_elicitation(frame: ElicitationRequest, console: Console) -> None:
    console.line()
    count = len(frame.questions)
    for number, question in enumerate(frame.questions, start=1):
        heading = f"Question {number} of {count}" if count > 1 else "Question"
        console.line(console.paint(f"{heading}: {question.question}", "1;36"))
        for index, option in enumerate(question.options or [], start=1):
            detail = f" -- {option.description}" if option.description else ""
            mark = " (recommended)" if option.recommended else ""
            console.line(f"    {index}. {option.label}{detail}{mark}")
        if question.rationale:
            console.line(console.paint(f"    -> {question.rationale}", "2"))
    console.line()


def _render_suggestions(frame: Suggestions, console: Console) -> None:
    if frame.items:
        chips = " | ".join(item.label for item in frame.items)
        console.line(console.paint(f"Follow-ups: {chips}", "2"))


def _render_error(frame: ErrorFrame, console: Console) -> None:
    console.line()
    error = frame.error
    console.line(console.paint(f"Error [{error.error.value}] retry: {error.retry.value}", "1;31"))
    console.line(f"  {error.detail}")
    if error.fix:
        console.line(f"  fix: {error.fix}")
    console.line()


def _render_done(frame: Done, console: Console) -> None:
    if frame.composition:
        console.line(
            console.paint(f"Done. Canvas: {frame.composition.counts.all} card(s).", "1;32")
        )
    else:
        console.line(console.paint("Done.", "1;32"))


def _render_unhandled(frame: StudioFrame, console: Console) -> None:
    """Makes an unrecognised-but-valid frame loud instead of invisible.

    `_read_frames.handle` already guards the sibling failure -- a raw SSE
    payload whose `event` doesn't match any wire model at all -- by printing
    it and moving on. This is the other half: a frame that DID parse as a
    `StudioFrame` (so it is one of the union's current members) but that
    `render_frame`'s own `match` has no case for, because a variant was
    added to the union after this file's case list, or the two drifted
    apart some other way. A frame type silently rendering as nothing is
    exactly the failure this CLI exists to make visible, so this prints to
    both stdout (in the transcript, where a person reading along sees it)
    and stderr (so a redirected or piped run doesn't bury it), and the
    caller marks the whole turn as having dropped content."""
    message = (
        f"[unhandled frame type: {type(frame).__name__} (event={getattr(frame, 'event', '?')!r})]"
    )
    console.line(console.paint(message, "1;31"))
    console.line(json.dumps(frame.model_dump(mode="json"), indent=2))
    print(message, file=sys.stderr)


def render_frame(frame: StudioFrame, console: Console, state: RenderState) -> bool:
    """Renders one parsed wire frame, updating narration state as it goes.

    Returns False for a frame `match` has no case for (see
    `_render_unhandled`); the caller uses that to flag the turn.
    """
    if not isinstance(frame, MessageDelta):
        state.break_line(console)
    match frame:
        case MessageDelta():
            console.write(frame.text)
            state.mid_line = True
            state.narrated = True
        case MessageComplete():
            if not state.narrated:
                console.line(frame.text)
        case PlanProposed():
            _render_plan(frame, console)
        case PlanVersioned():
            _render_plan_versioned(frame, console)
        case ActivityStep():
            _render_activity_step(frame, console, state)
        case ComponentCreated():
            _render_component_created(frame, console)
        case ComponentRevised():
            _render_component_revised(frame, console)
        case CompositionUpdated():
            _render_composition_updated(frame, console)
        case ElicitationRequest():
            _render_elicitation(frame, console)
        case Suggestions():
            _render_suggestions(frame, console)
        case ErrorFrame():
            _render_error(frame, console)
        case Done():
            _render_done(frame, console)
        case _:
            _render_unhandled(frame, console)
            return False
    return True


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------


async def _ainput(prompt: str = "") -> str:
    """Reads one line from stdin without blocking the event loop's thread.

    A bare `input()` awaited nowhere runs synchronously inside the
    coroutine, and asyncio's Ctrl-C handling cancels the running task at
    its next await point -- a task with none, parked in a blocking read
    syscall, cannot be cancelled until that call returns on its own. Reading
    on a daemon thread and awaiting its result gives the task an await
    point, so SIGINT (and a closed stdin) interrupt it immediately instead
    of hanging until the next keystroke.
    """
    loop = asyncio.get_running_loop()
    future: asyncio.Future[str] = loop.create_future()

    def read() -> None:
        try:
            result = input(prompt)
        except BaseException as exc:  # EOFError on closed stdin, most often
            loop.call_soon_threadsafe(future.set_exception, exc)
        else:
            loop.call_soon_threadsafe(future.set_result, result)

    threading.Thread(target=read, daemon=True).start()
    return await future


async def prompt_plan_decision(
    console: Console, *, auto_yes: bool
) -> tuple[InterruptDecision, str | None]:
    if auto_yes:
        console.line(console.paint("  -> approved (--yes)", "2"))
        return InterruptDecision.APPROVE, None
    while True:
        console.write(console.paint("Approve this plan? [Enter=approve, r=reject, a=amend] ", "1"))
        choice = (await _ainput()).strip().lower()
        if choice in ("", "y", "yes", "approve"):
            console.line(console.paint("  -> approved, sending...", "2"))
            return InterruptDecision.APPROVE, None
        if choice in ("r", "n", "reject"):
            console.line(console.paint("  -> rejected, sending...", "2"))
            return InterruptDecision.REJECT, None
        if choice in ("a", "amend"):
            note = (await _ainput(console.paint("Amendment: ", "1"))).strip()
            console.line(console.paint("  -> amendment noted, sending...", "2"))
            return InterruptDecision.AMEND, note
        console.line("  (enter, r, or a)")


async def prompt_elicitation_decision(
    frame: ElicitationRequest, console: Console, *, auto_yes: bool
) -> tuple[InterruptDecision, dict[str, ElicitationAnswer] | None]:
    """The decision and per-question answers for one elicitation pause.

    Each question takes an option number, free text when it allows it, or
    "s" to leave it to the agent when it is skippable. "s" on a question
    that is not skippable dismisses the whole pause when the frame allows
    that. `--yes` picks each question's recommended option, else its first.
    """
    answers: dict[str, ElicitationAnswer] = {}
    count = len(frame.questions)
    for number, question in enumerate(frame.questions, start=1):
        if auto_yes:
            answer = _auto_answer(question)
            if answer is None:
                console.line(console.paint("  -> skipped (--yes)", "2"))
                return InterruptDecision.SKIP, None
            answers[question.id] = answer
            labels = {option.id: option.label for option in question.options or []}
            shown = labels[answer.option_id] if isinstance(answer, OptionAnswer) else "left to you"
            console.line(console.paint(f"  -> answered '{shown}' (--yes)", "2"))
            continue
        label = f"Answer {number} of {count}" if count > 1 else "Answer"
        answer = await _prompt_question(question, console, label=label, dismissable=frame.skippable)
        if answer is None:
            console.line(console.paint("  -> skipped, sending...", "2"))
            return InterruptDecision.SKIP, None
        answers[question.id] = answer
    if not auto_yes:
        console.line(console.paint("  -> answered, sending...", "2"))
    return InterruptDecision.ANSWER, answers


def _auto_answer(question: ElicitationQuestion) -> ElicitationAnswer | None:
    """The question's recommended option, else its first; with no options,
    the question left to the agent when it allows that, else None for a
    whole-pause dismiss."""
    if question.options:
        chosen = next((o for o in question.options if o.recommended), question.options[0])
        return OptionAnswer(kind="option", option_id=chosen.id)
    if question.skippable:
        return SkippedAnswer(kind="skipped")
    return None


async def _prompt_question(
    question: ElicitationQuestion, console: Console, *, label: str, dismissable: bool
) -> ElicitationAnswer | None:
    """One question's answer read from the terminal, or None when the user
    dismissed the whole pause."""
    options = question.options or []
    skippable = question.skippable or dismissable
    hints = []
    if question.allow_free_text:
        hints.append("or type an answer")
    if skippable:
        hints.append("s=skip")
    suffix = f" ({', '.join(hints)})" if hints else ""
    while True:
        console.write(console.paint(f"{label}{suffix}: ", "1"))
        choice = (await _ainput()).strip()
        if skippable and choice.lower() in ("s", "skip"):
            return SkippedAnswer(kind="skipped") if question.skippable else None
        if options and choice.isdigit() and 1 <= int(choice) <= len(options):
            return OptionAnswer(kind="option", option_id=options[int(choice) - 1].id)
        if question.allow_free_text and choice:
            return TextAnswer(kind="text", text=choice)
        console.line(
            "  (pick a number"
            + (", type free text" if question.allow_free_text else "")
            + (", or 's' to skip" if skippable else "")
            + ")"
        )


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------


async def _read_frames(
    response: httpx2.Response, console: Console, waiter: Waiter, state: RenderState
) -> AsyncIterator[StudioFrame]:
    """Parses and renders each frame in `response` as it arrives, updating
    `waiter`'s phase, and yields it. Off a tty it just reads the stream. On
    a tty, redraws `waiter`'s one status line during an idle stretch --
    before the resolve step, between resolve and the plan, and between
    execution steps, which between them are the only silent stretches this
    SSE stream has.

    `asyncio.wait` on the pending read with a timeout is what lets an idle
    stretch redraw without ever cancelling the read itself -- a timeout
    leaves the read running, it just isn't done yet, so no real frame is
    delayed or lost.
    """

    def handle(payload: str) -> StudioFrame | None:
        try:
            raw = json.loads(payload)
        except json.JSONDecodeError:
            state.break_line(console)
            console.line(console.paint(f"[malformed frame: {payload!r}]", "33"))
            waiter.unhandled_frame = True
            return None
        raw = _strip_control_sequences(raw)
        try:
            frame = STUDIO_FRAME_ADAPTER.validate_python(raw)
        except ValidationError:
            state.break_line(console)
            console.line(console.paint(f"[unrecognized frame: {raw.get('event', '?')}]", "33"))
            console.line(json.dumps(raw, indent=2))
            waiter.unhandled_frame = True
            return None
        if not render_frame(frame, console, state):
            waiter.unhandled_frame = True
        waiter.note(frame)
        return frame

    if not console.tty:
        async for line in response.aiter_lines():
            if not line.startswith("data: "):
                continue
            frame = handle(line[len("data: ") :])
            if frame is not None:
                yield frame
        return

    lines = response.aiter_lines().__aiter__()
    pending = asyncio.ensure_future(lines.__anext__())
    shown = False
    try:
        while True:
            done, _ = await asyncio.wait({pending}, timeout=_REDRAW_INTERVAL)
            if pending not in done:
                if not state.mid_line:
                    console.waiting(waiter.render(console))
                    shown = True
                continue
            try:
                line = pending.result()
            except StopAsyncIteration:
                return
            pending = asyncio.ensure_future(lines.__anext__())
            if not line.startswith("data: "):
                continue
            if shown:
                console.clear_waiting()
                shown = False
            frame = handle(line[len("data: ") :])
            if frame is not None:
                yield frame
    finally:
        pending.cancel()
        with contextlib.suppress(BaseException):
            await pending
        if shown:
            console.clear_waiting()


async def stream_segment(
    client: httpx2.AsyncClient, path: str, body: dict[str, Any], console: Console, waiter: Waiter
) -> PlanProposed | ElicitationRequest | Done | ErrorFrame | None:
    """Streams one SSE segment, rendering every frame as it arrives.

    Returns the pausing frame when the segment ends on `plan.proposed` or
    `elicitation.request`, the terminal frame when it ends on `done` or
    `error` -- the caller tells the two apart, since one is a completed
    answer and the other is a failure the server reported back -- or None
    if the stream closes without reaching any of those.

    Reads the stream to its end before returning. The server holds the
    conversation until the segment has saved and released it, which is
    after the pausing frame. The stream's end signals that it has, so an
    `/interrupt` sent then is not refused as a concurrent request, and the
    server's release does not run under a client disconnect.
    """
    state = RenderState()
    ending: PlanProposed | ElicitationRequest | Done | ErrorFrame | None = None
    async with client.stream("POST", path, json=body) as response:
        if response.status_code != 200:
            await response.aread()
            raise StudioError(f"{path} -> HTTP {response.status_code}: {response.text[:300]}")
        async for frame in _read_frames(response, console, waiter, state):
            if isinstance(frame, (PlanProposed, ElicitationRequest, Done, ErrorFrame)):
                ending = frame
    return ending


async def ask(
    client: httpx2.AsyncClient, question: str, *, auto_yes: bool, console: Console
) -> int:
    """Drives one turn to completion, resuming through every pause. Returns
    the process exit code for the turn's outcome: 0 for a clean `done`, 1
    for a real `error.frame` the server sent back, 2 if the turn dropped
    anything -- a StudioFrame variant `render_frame` had no case for, a
    frame failed wire validation, a line wasn't valid JSON, or the stream
    ended without reaching `done`, `error`, or a pause."""
    conversation_id = f"ask_{uuid.uuid4().hex[:12]}"
    request = TurnRequest(
        kind="turn.request", schema_version=1, conversation_id=conversation_id, message=question
    )
    path = "/api/studio/turn"
    body = request.model_dump(mode="json", exclude_none=True)
    waiter = Waiter(start=time.monotonic())
    while True:
        frame = await stream_segment(client, path, body, console, waiter)
        if isinstance(frame, Done):
            return 2 if waiter.unhandled_frame else 0
        if isinstance(frame, ErrorFrame):
            return 1
        if frame is None:
            # The stream closed without a done, error, or pause frame --
            # protocol drift this CLI has no case for, same severity as an
            # unhandled frame type.
            return 2
        pause = frame
        payload: str | None = None
        answers: dict[str, ElicitationAnswer] | None = None
        if isinstance(pause, PlanProposed):
            decision, payload = await prompt_plan_decision(console, auto_yes=auto_yes)
        else:
            decision, answers = await prompt_elicitation_decision(pause, console, auto_yes=auto_yes)
        waiter.note_decision(pause, decision)
        resume = InterruptResponse(
            kind="interrupt.response",
            schema_version=1,
            conversation_id=conversation_id,
            interrupt_id=pause.interrupt_id,
            decision=decision,
            payload=payload,
            answers=answers,
        )
        path = "/api/studio/interrupt"
        body = resume.model_dump(mode="json", exclude_none=True)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ask the studio harness a question and read the answer."
    )
    # nargs="+": `just ask <question>` hands the question to `just` as one
    # shell-quoted argument, but just's variadic recipe parameter re-splits
    # it on spaces before this script ever sees it, so it arrives as several
    # argv entries. Joining them back is the fix, not `nargs=1`.
    parser.add_argument("question", nargs="+", help="the message to send as the turn")
    parser.add_argument(
        "--url",
        # `or`, not `.get(key, default)`: `just env` carries a var forward
        # from .env.example even when its value is blank (see STUDIO_URL /
        # STUDIO_API_TOKEN below), so a present-but-empty value must fall
        # back the same way an absent one does.
        default=os.environ.get("STUDIO_URL") or _DEFAULT_URL,
        help=f"harness base URL (default: {_DEFAULT_URL})",
    )
    parser.add_argument(
        "--token",
        # Same `or` fallback, and for the same reason `settings.py`'s
        # `_blank_token_is_unset` validator and evals/studio/run.py's
        # `_AUTH` both treat a blank `STUDIO_API_TOKEN` as unset: a bare
        # `STUDIO_API_TOKEN=` line in .env must not become a literal empty
        # bearer token, which httpx2 rejects before any request leaves the
        # process.
        default=os.environ.get("STUDIO_API_TOKEN") or _DEFAULT_TOKEN,
        help="bearer token (default: $STUDIO_API_TOKEN, else a placeholder any unauthenticated server accepts)",
    )
    parser.add_argument(
        "--yes",
        "-y",
        action="store_true",
        help="approve plans and take the first elicitation option without prompting",
    )
    return parser.parse_args(argv)


async def _main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    console = Console(color=_color_enabled(), tty=sys.stdout.isatty())
    headers = {"Authorization": f"Bearer {args.token}"}
    try:
        _check_url(args.url)
        async with httpx2.AsyncClient(
            base_url=args.url, headers=headers, timeout=httpx2.Timeout(300.0)
        ) as client:
            return await ask(client, " ".join(args.question), auto_yes=args.yes, console=console)
    except StudioError as error:
        console.line(console.paint(f"studio error: {error}", "1;31"))
        return 1
    except httpx2.ConnectError:
        console.line(
            console.paint(
                f"could not reach {args.url} -- is the harness running? (just dev / just studio-live)",
                "1;31",
            )
        )
        return 1
    except httpx2.HTTPError as error:
        console.line(console.paint(f"connection dropped: {error}", "1;31"))
        return 1


def main() -> None:
    try:
        raise SystemExit(asyncio.run(_main()))
    except (KeyboardInterrupt, EOFError):
        # EOFError: stdin closed (piped, redirected from /dev/null, or the
        # terminal detached) while a plan or elicitation prompt was waiting
        # on input() -- the same "no answer is coming" case as Ctrl-C.
        print()
        raise SystemExit(130) from None


if __name__ == "__main__":
    main()
