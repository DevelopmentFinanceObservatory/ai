"""Tests for `harness.turn_budget`: one request's bounds on its model calls.

Each call is a scripted stream: a text chunk, or `_STALL`, which waits until
the call is cut or cancelled. The bounds are fractions of a second, so a cut
fires at once, and the wall bound reads a fake clock.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Iterator
from contextlib import contextmanager
from typing import Any

import pytest
from google.genai import types
from harness.sentences import StopCode, refusal_sentence
from harness.turn_budget import (
    COMPLETION_START_TIME,
    Chunk,
    Closing,
    TurnBudget,
    bounded_by,
    current_turn_budget,
)
from langfuse import LangfuseOtelSpanAttributes
from opentelemetry.sdk.trace import Span, TracerProvider
from opentelemetry.trace import StatusCode

_STALL = None


class _Calls:
    """Each invocation opens the next scripted stream and counts it."""

    def __init__(self, *scripts: list[str | None]) -> None:
        self.opened = 0
        self._scripts = list(scripts)

    def __call__(self) -> AsyncGenerator[Chunk]:
        script = self._scripts[self.opened]
        self.opened += 1
        return _stream(script)


def _chunk(*parts: types.Part, usage: Any = None) -> Chunk:
    return types.GenerateContentResponse(
        candidates=[types.Candidate(content=types.Content(role="model", parts=list(parts)))],
        usage_metadata=usage,
    )


async def _stream(script: list[str | None]) -> AsyncGenerator[Chunk]:
    for step in script:
        if step is _STALL:
            await asyncio.Event().wait()
        yield _chunk(types.Part(text=step))


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def _budget(**overrides: Any) -> TurnBudget:
    bounds: dict[str, Any] = {
        "model_time_s": 0.5,
        "model_calls_max": 30,
        "wall_s": 600.0,
        "first_chunk_s": 0.05,
    }
    return TurnBudget(**{**bounds, **overrides})


@contextmanager
def _current(budget: TurnBudget | None) -> Iterator[None]:
    token = current_turn_budget.set(budget)
    try:
        yield
    finally:
        current_turn_budget.reset(token)


@contextmanager
def _generate_content_span() -> Iterator[Span]:
    tracer = TracerProvider().get_tracer(__name__)
    with tracer.start_as_current_span("generate_content gemini-3.8-flash") as span:
        assert isinstance(span, Span)
        yield span


def _text(item: Chunk | Closing) -> str:
    if isinstance(item, Closing):
        return item.shown
    content = item.candidates[0].content if item.candidates else None
    return "".join(part.text or "" for part in (content.parts or [])) if content else ""


async def _texts(calls: _Calls) -> list[str]:
    return [_text(item) async for item in bounded_by(current_turn_budget.get(), calls)]


def _closed(code: StopCode) -> str:
    return refusal_sentence({"code": code})


async def test_a_call_with_no_budget_runs_unbounded() -> None:
    calls = _Calls(["a", "b"])

    with _current(None):
        assert await _texts(calls) == ["a", "b"]


async def test_a_first_chunk_timeout_is_resent_once_before_anything_is_delivered() -> None:
    budget = _budget()
    calls = _Calls([_STALL], ["a", "b"])

    with _current(budget):
        texts = await _texts(calls)

    assert texts == ["a", "b"]
    assert calls.opened == 2
    assert budget.retry_spent
    assert budget.model_calls == 1
    assert budget.closed is None


async def test_a_call_that_delivered_a_chunk_is_not_resent_and_the_budget_cuts_it() -> None:
    budget = _budget(model_time_s=0.2)
    calls = _Calls(["thinking", _STALL], ["never sent"])

    with _current(budget), _generate_content_span() as span:
        texts = await _texts(calls)

    assert calls.opened == 1
    assert not budget.retry_spent
    sentence = _closed(StopCode.TURN_MODEL_TIME_SPENT)
    # The closing shows the sentence after the streamed text.
    assert texts == ["thinking", f"\n\n{sentence}"]
    assert budget.closed == {"code": StopCode.TURN_MODEL_TIME_SPENT, "detail": {}}
    assert span.status.status_code is StatusCode.ERROR


async def test_a_second_call_in_the_request_gets_no_retry() -> None:
    budget = _budget(first_chunk_s=0.02, model_time_s=0.2)
    first = _Calls([_STALL], ["a"])
    second = _Calls([_STALL], ["never sent"])

    with _current(budget):
        assert await _texts(first) == ["a"]
        texts = await _texts(second)

    # The second call's stall outlasts its first-chunk deadline, so only the
    # budget ends it.
    assert second.opened == 1
    assert texts[-1] == _closed(StopCode.TURN_MODEL_TIME_SPENT)
    assert budget.model_time_spent >= 0.2


@pytest.mark.parametrize(
    ("spend", "code"),
    [
        ({"model_time_spent": 0.5}, StopCode.TURN_MODEL_TIME_SPENT),
        ({"model_calls": 30}, StopCode.TURN_MODEL_CALLS_SPENT),
    ],
    ids=["model-time", "call-cap"],
)
async def test_a_spent_budget_ends_on_its_sentence_with_no_model_call(
    spend: dict[str, float], code: StopCode
) -> None:
    budget = _budget()
    for name, value in spend.items():
        setattr(budget, name, value)
    calls = _Calls(["never sent"])

    with _current(budget):
        texts = await _texts(calls)

    assert calls.opened == 0
    assert texts == [_closed(code)]
    assert budget.closed == {"code": code, "detail": {}}


async def test_model_time_leaves_out_what_the_caller_does_between_chunks() -> None:
    """A caller that streams runs its own work while the generator waits at
    its yield, so only the waits on the stream count. A caller that spends 100 s
    after each chunk, more than the whole budget, neither cuts the call nor
    spends any model time."""
    clock = _Clock()
    budget = _budget(model_time_s=90.0, clock=clock)
    texts = []

    with _current(budget):
        async for response in bounded_by(budget, _Calls(["a", "b"])):
            texts.append(_text(response))
            clock.now += 100.0

    assert texts == ["a", "b"]
    assert budget.model_time_spent == 0.0
    assert budget.closed is None


async def test_the_wall_bound_counts_time_outside_model_calls() -> None:
    clock = _Clock()
    budget = _budget(wall_s=600.0, clock=clock)
    calls = _Calls(["a"], ["never sent"])

    with _current(budget):
        assert await _texts(calls) == ["a"]
        # A tool call ran for ten minutes between the two model calls.
        clock.now += 600.0
        texts = await _texts(calls)

    assert calls.opened == 1
    assert texts == [_closed(StopCode.TURN_WALL_TIME_SPENT)]


async def test_the_wall_bound_cuts_a_streaming_call_when_it_comes_first() -> None:
    budget = _budget(model_time_s=60.0, wall_s=0.1)
    calls = _Calls(["thinking", _STALL])

    with _current(budget):
        texts = await _texts(calls)

    assert texts == ["thinking", f"\n\n{_closed(StopCode.TURN_WALL_TIME_SPENT)}"]
    assert budget.closed == {"code": StopCode.TURN_WALL_TIME_SPENT, "detail": {}}


async def test_a_user_cancel_propagates_and_is_never_retried() -> None:
    budget = _budget(first_chunk_s=5.0)
    calls = _Calls([_STALL], ["never sent"])

    with _current(budget), _generate_content_span() as span:
        task = asyncio.create_task(_texts(calls))
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    assert calls.opened == 1
    assert not budget.retry_spent
    assert budget.closed is None
    assert span.status.status_code is StatusCode.ERROR
    assert span.status.description == "cancelled"


async def test_the_span_records_the_first_chunk_and_the_gaps_after_it() -> None:
    budget = _budget()
    calls = _Calls(["a", "b", "c"])

    with _current(budget), _generate_content_span() as span:
        await _texts(calls)

    assert COMPLETION_START_TIME == LangfuseOtelSpanAttributes.OBSERVATION_COMPLETION_START_TIME
    assert span.attributes is not None
    assert COMPLETION_START_TIME in span.attributes
    assert len(span.attributes["dfo.model.chunk_gaps_s"]) == 2


async def test_the_span_records_thought_tokens_and_when_thought_and_text_first_arrived() -> None:
    """A call that thinks before it answers stamps when its first thought and
    its first text arrived, counted as the first chunk is, on the stream's
    waits alone, so the 10 s the caller spends on each chunk is left out, and
    the thought tokens its usage reports."""
    clock = _Clock()
    budget = _budget(clock=clock, model_time_s=60.0)

    async def thinks_then_answers() -> AsyncGenerator[Chunk]:
        clock.now += 1.0
        yield _chunk(types.Part(text="hm", thought=True))
        clock.now += 3.0
        yield _chunk(types.Part(text="answer"))
        yield _chunk(
            types.Part(text="answer"),
            usage=types.GenerateContentResponseUsageMetadata(thoughts_token_count=321),
        )

    with _current(budget), _generate_content_span() as span:
        async for _ in bounded_by(budget, thinks_then_answers):
            clock.now += 10.0

    assert span.attributes is not None
    assert span.attributes["dfo.model.first_thought_s"] == 1.0
    assert span.attributes["dfo.model.first_text_s"] == 4.0
    assert span.attributes["dfo.model.thoughts_token_count"] == 321
