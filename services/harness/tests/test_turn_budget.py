"""Tests for `harness.turn_budget`: one request's bounds on its model calls.

Each call is a scripted stream: a text chunk, or `_STALL`, which waits until
the call is cut or cancelled. The bounds are fractions of a second, so a cut
fires at once, and the wall bound reads a fake clock.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Iterator
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any

import pytest
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.genai import Client, types
from harness.sentences import SeamCode, refusal_sentence
from harness.turn_budget import (
    COMPLETION_START_TIME,
    BoundedGemini,
    TurnBudget,
    bounded,
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

    def __call__(self) -> AsyncGenerator[LlmResponse]:
        script = self._scripts[self.opened]
        self.opened += 1
        return _stream(script)


async def _stream(script: list[str | None]) -> AsyncGenerator[LlmResponse]:
    for step in script:
        if step is _STALL:
            await asyncio.Event().wait()
        yield LlmResponse(
            content=types.Content(role="model", parts=[types.Part(text=step)]), partial=True
        )


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


async def _texts(calls: _Calls, *, stream: bool = True) -> list[str]:
    return [
        "".join(part.text or "" for part in response.content.parts)
        async for response in bounded(calls, stream=stream)
        if response.content and response.content.parts
    ]


def _closed(code: SeamCode) -> str:
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
    sentence = _closed(SeamCode.TURN_MODEL_TIME_SPENT)
    # A partial chunk shows the sentence after the streamed text; the final
    # response is the sentence alone.
    assert texts == ["thinking", f"\n\n{sentence}", sentence]
    assert budget.closed == {"code": SeamCode.TURN_MODEL_TIME_SPENT, "detail": {}}
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
    assert texts[-1] == _closed(SeamCode.TURN_MODEL_TIME_SPENT)
    assert budget.model_time_spent >= 0.2


async def test_a_non_streamed_call_has_no_first_chunk_deadline() -> None:
    budget = _budget(first_chunk_s=0.01, model_time_s=0.1)
    calls = _Calls([_STALL], ["never sent"])

    with _current(budget):
        texts = await _texts(calls, stream=False)

    assert calls.opened == 1
    assert texts == [_closed(SeamCode.TURN_MODEL_TIME_SPENT)]


@pytest.mark.parametrize(
    ("spend", "code"),
    [
        ({"model_time_spent": 0.5}, SeamCode.TURN_MODEL_TIME_SPENT),
        ({"model_calls": 30}, SeamCode.TURN_MODEL_CALLS_SPENT),
    ],
    ids=["model-time", "call-cap"],
)
async def test_a_spent_budget_ends_on_its_sentence_with_no_model_call(
    spend: dict[str, float], code: SeamCode
) -> None:
    budget = _budget()
    for name, value in spend.items():
        setattr(budget, name, value)
    calls = _Calls(["never sent"])

    with _current(budget):
        texts = await _texts(calls, stream=False)

    assert calls.opened == 0
    assert texts == [_closed(code)]
    assert budget.closed == {"code": code, "detail": {}}


async def test_model_time_leaves_out_what_the_caller_does_between_chunks() -> None:
    """ADK runs a response's tools while the model's generator waits at its
    yield, so only the waits on the stream count. A caller that spends 100 s
    after each chunk, more than the whole budget, neither cuts the call nor
    spends any model time."""
    clock = _Clock()
    budget = _budget(model_time_s=90.0, clock=clock)
    texts = []

    with _current(budget):
        async for response in bounded(_Calls(["a", "b"]), stream=True):
            assert response.content is not None and response.content.parts is not None
            texts.append(response.content.parts[0].text)
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
        texts = await _texts(calls, stream=False)

    assert calls.opened == 1
    assert texts == [_closed(SeamCode.TURN_WALL_TIME_SPENT)]


async def test_the_wall_bound_cuts_a_streaming_call_when_it_comes_first() -> None:
    budget = _budget(model_time_s=60.0, wall_s=0.1)
    calls = _Calls(["thinking", _STALL])

    with _current(budget):
        texts = await _texts(calls)

    assert texts[-1] == _closed(SeamCode.TURN_WALL_TIME_SPENT)
    assert budget.closed == {"code": SeamCode.TURN_WALL_TIME_SPENT, "detail": {}}


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
    its first text arrived, counted from when the call opened, and the
    thought tokens its usage reports."""
    clock = _Clock()
    budget = _budget(clock=clock, model_time_s=60.0)

    async def thinks_then_answers() -> AsyncGenerator[LlmResponse]:
        clock.now = 1.0
        yield LlmResponse(
            content=types.Content(role="model", parts=[types.Part(text="hm", thought=True)]),
            partial=True,
        )
        clock.now = 4.0
        yield LlmResponse(
            content=types.Content(role="model", parts=[types.Part(text="answer")]), partial=True
        )
        yield LlmResponse(
            content=types.Content(role="model", parts=[types.Part(text="answer")]),
            usage_metadata=types.GenerateContentResponseUsageMetadata(thoughts_token_count=321),
        )

    with _current(budget), _generate_content_span() as span:
        async for _ in bounded(thinks_then_answers, stream=True):
            pass

    assert span.attributes is not None
    assert span.attributes["dfo.model.first_thought_s"] == 1.0
    assert span.attributes["dfo.model.first_text_s"] == 4.0
    assert span.attributes["dfo.model.thoughts_token_count"] == 321


class _StallingModels:
    """genai's `aio.models`: the first stream stalls before its first chunk."""

    def __init__(self) -> None:
        self.streams = 0

    async def generate_content_stream(self, *, model: str, contents: Any, config: Any) -> Any:
        self.streams += 1
        stalls = self.streams == 1

        async def chunks() -> AsyncGenerator[types.GenerateContentResponse]:
            if stalls:
                await asyncio.Event().wait()
            yield types.GenerateContentResponse(
                candidates=[
                    types.Candidate(
                        content=types.Content(role="model", parts=[types.Part(text="done")]),
                        finish_reason=types.FinishReason.STOP,
                    )
                ]
            )

        return chunks()


class _FakeClient(Client):
    def __init__(self) -> None:
        self.models_calls = _StallingModels()

    @property
    def aio(self) -> Any:
        return SimpleNamespace(models=self.models_calls)

    @property
    def vertexai(self) -> bool:
        return False

    def close(self) -> None:
        pass


async def test_bounded_gemini_resends_the_genai_request_after_a_first_chunk_timeout() -> None:
    client = _FakeClient()
    model = BoundedGemini(model="gemini-3.8-flash", client=client)
    request = LlmRequest(
        model="gemini-3.8-flash",
        contents=[types.Content(role="user", parts=[types.Part(text="question")])],
        config=types.GenerateContentConfig(),
    )

    with _current(_budget()):
        responses = [
            response
            async for response in model.generate_content_async(request, stream=True)
            if not response.partial
        ]

    assert client.models_calls.streams == 2
    assert responses[-1].content is not None
    assert responses[-1].content.parts is not None
    assert responses[-1].content.parts[0].text == "done"
