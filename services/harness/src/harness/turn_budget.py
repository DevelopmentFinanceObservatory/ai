"""One request's bounds on its model calls, applied to each streamed Gemini call.

A request is one POST to `/api/studio/turn` or `/api/studio/interrupt`. The
route builds one `TurnBudget` from `harness.settings.Settings` and sets it in
`current_turn_budget` before the stream starts, so every model call the run
makes, the orchestrator's and the planner's alike, reads the same one. It
holds four bounds (the `turn_*` and `model_first_chunk_s` settings say what
each meters):

- model time, 90 s;
- a cap of 30 model calls;
- a wall-clock bound of 600 s from the request's start, tool time included;
- a first-chunk deadline of 12 s, after which the call is resent once. Only
  the first call in the request to miss it is resent, and only before any
  chunk reached the caller. A provider slow period defeats a retry, so later
  calls wait under the budget alone.

`bounded_by` applies them around one streamed google-genai call, starting the
clock before genai's own HTTP retries so it covers them. The budget cuts a call
that is still streaming. A call that finds a bound spent, or that a bound cuts,
makes no further request and ends on a `Closing`: that bound's sentence
(`harness.sentences.STOP_SENTENCES`) as the model's final text. A caller that
passes chunks on as they arrive shows `Closing.shown` after the text it
already passed on. One that holds a call's chunks until it ends answers with
`Closing.content` in their place, as the planner's model does
(`harness.gemini_wire`). In the planner, `harness.planner_adapter` turns that
answer into a question to the user when the last referral names sources to
choose from, and otherwise the run ends on it and the entry refuses with the
code on `TurnBudget.closed`.

A user's stop or a client disconnect cancels the call. The cancellation
propagates and closes the stream, and the call is not retried. The span current
around the call (Pydantic AI's `chat <model>`) is marked as an error when the
call is cancelled or cut. It also carries
`langfuse.observation.completion_start_time` at the first chunk and the gaps
between chunks (`dfo.model.chunk_gaps_s`), so an idle rule can be sized from
the traces.

With no budget (an eval runner, a test driving a model directly) a call runs
unbounded.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncGenerator, Callable
from contextlib import aclosing
from contextvars import ContextVar
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from typing import Any

from google.genai import types
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

from harness.sentences import StopCode, refusal_sentence
from harness.settings import Settings

# `langfuse.LangfuseOtelSpanAttributes.OBSERVATION_COMPLETION_START_TIME`: Langfuse
# reads a generation's time to first token from it. Spelled out so this module
# does not import the Langfuse client.
COMPLETION_START_TIME = "langfuse.observation.completion_start_time"

type Chunk = types.GenerateContentResponse


class TurnBudget:
    """The model-time budget, call cap, wall-clock bound and first-chunk
    deadline of one request, and what its calls have spent of them.

    `clock` stands in for `time.monotonic` in a test that moves time without
    sleeping. The cuts themselves wait on the event loop's clock, which is
    `time.monotonic` too.
    """

    def __init__(
        self,
        *,
        model_time_s: float,
        model_calls_max: int,
        wall_s: float,
        first_chunk_s: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.model_time_s = model_time_s
        self.model_calls_max = model_calls_max
        self.wall_s = wall_s
        self.first_chunk_s = first_chunk_s
        self.clock = clock
        self.started_at = clock()
        self.model_time_spent = 0.0
        self.model_calls = 0
        # Set once a call in this request has been resent after its first-chunk
        # deadline. No later call is resent.
        self.retry_spent = False
        # The refusal of the bound that closed a call, once one has.
        self.closed: dict[str, Any] | None = None

    @classmethod
    def from_settings(
        cls, settings: Settings, *, clock: Callable[[], float] = time.monotonic
    ) -> TurnBudget:
        return cls(
            model_time_s=settings.turn_model_time_s,
            model_calls_max=settings.turn_model_calls_max,
            wall_s=settings.turn_wall_s,
            first_chunk_s=settings.model_first_chunk_s,
            clock=clock,
        )

    def wall_left(self) -> float:
        """The seconds left before the wall-clock bound."""
        return self.wall_s - (self.clock() - self.started_at)

    def model_time_left(self) -> float:
        """The seconds of model time left, counting the waits of a call in flight."""
        return self.model_time_s - self.model_time_spent

    def spent_bound(self) -> StopCode | None:
        """The bound that leaves no room for another model call, if one does."""
        if self.wall_left() <= 0:
            return StopCode.TURN_WALL_TIME_SPENT
        if self.model_time_left() <= 0:
            return StopCode.TURN_MODEL_TIME_SPENT
        if self.model_calls >= self.model_calls_max:
            return StopCode.TURN_MODEL_CALLS_SPENT
        return None

    def close(self, code: StopCode) -> None:
        """Record `code` as the bound that closed a call, unless one already has."""
        if self.closed is None:
            self.closed = {"code": code, "detail": {}}


current_turn_budget: ContextVar[TurnBudget | None] = ContextVar("current_turn_budget", default=None)


def start_turn_budget(settings: Settings) -> TurnBudget:
    """Build this request's budget from `settings` and make it current.

    Called in the route's own task before the stream starts, as the route sets
    the conversation's other context. The streamed run and every task it
    spawns inherit it, and each request gets its own.
    """
    budget = TurnBudget.from_settings(settings)
    current_turn_budget.set(budget)
    return budget


@dataclass(frozen=True, slots=True)
class Closing:
    """The end of a call the bound `code` closed: its sentence as the model's
    final text. `after_text` says the call had already streamed answer text,
    so a reader that streams shows the sentence as a paragraph of its own
    (`shown`). `sent` says the call had been sent: false for a call the bound
    closed before it sent anything."""

    code: StopCode
    sentence: str
    after_text: bool
    sent: bool = True

    @property
    def shown(self) -> str:
        """The sentence as streamed after what the call already sent."""
        return f"\n\n{self.sentence}" if self.after_text else self.sentence

    @property
    def content(self) -> types.Content:
        """The sentence as the model turn a call that sends nothing answers with."""
        return types.Content(role="model", parts=[types.Part(text=self.sentence)])


async def bounded_by(
    budget: TurnBudget | None,
    request: Callable[[], AsyncGenerator[Chunk]],
) -> AsyncGenerator[Chunk | Closing]:
    """The chunks of one streamed model call, `request()`, kept to `budget`.

    `request` opens the call afresh each time it is invoked, so a first-chunk
    retry sends it again. With no budget the call runs unbounded. A call a
    bound closes ends on one `Closing`, after the chunks it delivered.
    """
    if budget is None:
        async with aclosing(request()) as responses:
            async for response in responses:
                yield response
        return

    if (spent := budget.spent_bound()) is not None:
        budget.close(spent)
        yield replace(_closing(spent, after_text=False), sent=False)
        return
    budget.model_calls += 1
    span = trace.get_current_span()
    while True:
        may_retry = not budget.retry_spent
        opened = budget.clock()
        first_chunk_by = opened + budget.first_chunk_s
        delivered = False
        delivered_text = False
        delivered_thought = False
        streamed = 0.0
        gaps: list[float] = []
        cut: StopCode | None = None
        retry = False
        try:
            async with aclosing(request()) as responses:
                while True:
                    # Only the waits on the stream count as model time. A caller
                    # that streams runs its own work while this generator sits
                    # at its yield.
                    wall_left, model_left = budget.wall_left(), budget.model_time_left()
                    before = budget.clock()
                    first_chunk_left = first_chunk_by - before
                    waits_for_first = (
                        may_retry
                        and not delivered
                        and first_chunk_left < min(wall_left, model_left)
                    )
                    limit = first_chunk_left if waits_for_first else min(wall_left, model_left)
                    timeout = asyncio.timeout(max(limit, 0.0))
                    try:
                        async with timeout:
                            response = await anext(responses)
                    except StopAsyncIteration:
                        return
                    except TimeoutError:
                        if not timeout.expired():
                            raise
                        if waits_for_first:
                            retry = True
                        else:
                            cut = (
                                StopCode.TURN_WALL_TIME_SPENT
                                if wall_left <= model_left
                                else StopCode.TURN_MODEL_TIME_SPENT
                            )
                        break
                    finally:
                        budget.model_time_spent += budget.clock() - before
                    waited = round(budget.clock() - before, 3)
                    streamed += waited
                    if not delivered:
                        _record_first_chunk(span, seconds=waited)
                    else:
                        gaps.append(waited)
                        _record_chunk_gaps(span, gaps)
                    delivered = True
                    if not delivered_thought and _has_thought(response):
                        delivered_thought = True
                        _record_streamed(span, "dfo.model.first_thought_s", seconds=streamed)
                    if not delivered_text and _has_text(response):
                        delivered_text = True
                        _record_streamed(span, "dfo.model.first_text_s", seconds=streamed)
                    _record_thoughts_tokens(span, response)
                    yield response
        except asyncio.CancelledError:
            _mark_failed(span, "cancelled")
            raise
        if retry:
            budget.retry_spent = True
            if span.is_recording():
                span.add_event(
                    "dfo.model.first_chunk_timeout",
                    {"dfo.model.first_chunk_s": budget.first_chunk_s},
                )
            continue
        assert cut is not None
        _mark_failed(span, cut.value)
        budget.close(cut)
        yield _closing(cut, after_text=delivered_text)
        return


def _closing(code: StopCode, *, after_text: bool) -> Closing:
    """The end of a call the bound `code` closed, after answer text or none."""
    return Closing(code=code, sentence=refusal_sentence({"code": code}), after_text=after_text)


def _parts(chunk: Chunk) -> list[types.Part]:
    """The parts of `chunk`'s first candidate, the one Pydantic AI reads."""
    if not chunk.candidates or chunk.candidates[0].content is None:
        return []
    return chunk.candidates[0].content.parts or []


def _has_text(chunk: Chunk) -> bool:
    """Whether `chunk` carries answer text, which Studio streams to the user."""
    return any(part.text and not part.thought for part in _parts(chunk))


def _has_thought(chunk: Chunk) -> bool:
    """Whether `chunk` carries thought-summary text, which Studio hides."""
    return any(part.text and part.thought for part in _parts(chunk))


def _record_streamed(span: trace.Span, key: str, *, seconds: float) -> None:
    """Stamp `key` on `span` with `seconds`, the time this attempt waited on
    the stream up to the chunk at hand.

    Measured as `dfo.model.first_chunk_s` is, on the waits alone, so time the
    caller spends at the generator's yield is left out and the three compare.
    `dfo.model.first_thought_s` and `dfo.model.first_text_s` beside
    `dfo.model.first_chunk_s` tell a model thinking apart from a provider
    queueing before it sends anything."""
    if span.is_recording():
        span.set_attribute(key, round(seconds, 3))


def _record_thoughts_tokens(span: trace.Span, response: Chunk) -> None:
    """Stamp the thought tokens the response's usage reports as
    `dfo.model.thoughts_token_count`, on the span that carries the call's
    `dfo.model.first_*` times, so one span tells how long the model thought
    and how many tokens that took."""
    usage = response.usage_metadata
    if usage is not None and usage.thoughts_token_count is not None and span.is_recording():
        span.set_attribute("dfo.model.thoughts_token_count", usage.thoughts_token_count)


def _record_chunk_gaps(span: trace.Span, gaps: list[float]) -> None:
    """Stamp the gaps between the call's chunks so far as
    `dfo.model.chunk_gaps_s`, before the latest chunk is yielded.

    The attribute is rewritten at each chunk, so a span that ends before the
    stream does still carries the gaps so far."""
    if span.is_recording():
        span.set_attribute("dfo.model.chunk_gaps_s", gaps)


def _record_first_chunk(span: trace.Span, *, seconds: float) -> None:
    if span.is_recording():
        span.set_attribute(COMPLETION_START_TIME, json.dumps(datetime.now(UTC).isoformat()))
        span.set_attribute("dfo.model.first_chunk_s", round(seconds, 3))


def _mark_failed(span: trace.Span, reason: str) -> None:
    if span.is_recording():
        span.set_status(Status(StatusCode.ERROR, reason))


__all__ = [
    "COMPLETION_START_TIME",
    "Chunk",
    "Closing",
    "TurnBudget",
    "bounded_by",
    "current_turn_budget",
    "start_turn_budget",
]
