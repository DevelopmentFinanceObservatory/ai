"""One orchestrator script, played by either framework's model.

A script is a list of `Step`s, one per model call, consumed in order. Each step
streams its thought chunks, then its text chunks, then makes its tool calls.

- `StreamingScriptLlm` plays it as an ADK `BaseLlm` does under
  `StreamingMode.SSE`: a partial `LlmResponse` per chunk, then one aggregated
  final response holding the thought, the text and the calls. It stands in for
  `tests.orchestrator_fakes.ScriptLlm`, which answers each call in one piece.
- `pai_model` plays it as a Pydantic AI `FunctionModel` stream function: a
  `DeltaThinkingPart` per thought chunk, a string per text chunk, and a
  `DeltaToolCall` per call.

Both keep what they were sent (`requests`), and `log` records one line per model
call with the number of messages (Pydantic AI) or contents (ADK) it was given.
"""

import asyncio
import json
from collections.abc import AsyncGenerator, AsyncIterator, Sequence
from dataclasses import dataclass, field
from typing import Any

from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_response import LlmResponse
from google.genai import types
from pydantic import Field
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models.function import (
    AgentInfo,
    DeltaThinkingCalls,
    DeltaThinkingPart,
    DeltaToolCall,
    DeltaToolCalls,
    FunctionModel,
)


@dataclass(frozen=True)
class Step:
    """One model call: thought chunks, then text chunks, then tool calls."""

    thought: Sequence[str] = ()
    text: Sequence[str] = ()
    calls: Sequence[tuple[str, dict[str, Any]]] = ()
    # Seconds the model waits after its first text chunk.
    stall: float = 0.0


def plan_step(brief: str, *, thought: Sequence[str] = ()) -> Step:
    return Step(thought=thought, calls=[("plan", {"brief": brief, "asks": [brief]})])


def ask_step(question: str, options: Sequence[str], *, thought: Sequence[str] = ()) -> Step:
    return Step(
        thought=thought,
        calls=[
            (
                "ask_user",
                {"questions": [{"question": question, "options": [{"label": o} for o in options]}]},
            )
        ],
    )


def search_step(query: str, variable: str) -> Step:
    return Step(calls=[("search", {"query": query, "parts": [{"variable": variable}]})])


@dataclass
class Played:
    """What one framework's model was sent and did."""

    steps: list[Step]
    requests: list[Any] = field(default_factory=list)
    log: list[str] = field(default_factory=list)
    calls_made: int = 0

    def next(self) -> Step:
        return self.steps.pop(0) if self.steps else Step(text=["done"])


class StreamingScriptLlm(BaseLlm):
    """An ADK model playing a `Played` script, partial chunks first under SSE."""

    model: str = "script-llm"
    played: Any = Field(default=None)

    async def generate_content_async(
        self, llm_request: Any, stream: bool = False
    ) -> AsyncGenerator[LlmResponse]:
        played: Played = self.played
        played.requests.append(llm_request)
        played.log.append(f"model call after {len(llm_request.contents)} contents")
        step = played.next()
        if stream:
            for chunk in step.thought:
                yield _partial(types.Part(text=chunk, thought=True))
            for index, chunk in enumerate(step.text):
                yield _partial(types.Part(text=chunk))
                if index == 0 and step.stall:
                    await asyncio.sleep(step.stall)
        parts: list[types.Part] = []
        if step.thought:
            parts.append(types.Part(text="".join(step.thought), thought=True))
        if step.text:
            parts.append(types.Part(text="".join(step.text)))
        parts += [types.Part.from_function_call(name=name, args=args) for name, args in step.calls]
        yield LlmResponse(content=types.Content(role="model", parts=parts))


def _partial(part: types.Part) -> LlmResponse:
    return LlmResponse(content=types.Content(role="model", parts=[part]), partial=True)


def pai_model(played: Played) -> FunctionModel:
    """A `FunctionModel` playing `played`; tool call ids are `call-<n>` in call order."""

    async def stream(
        messages: list[ModelMessage], info: AgentInfo
    ) -> AsyncIterator[str | DeltaToolCalls | DeltaThinkingCalls]:
        played.requests.append(list(messages))
        played.log.append(f"model call after {len(messages)} messages")
        step = played.next()
        for chunk in step.thought:
            yield {0: DeltaThinkingPart(content=chunk)}
        for index, chunk in enumerate(step.text):
            yield chunk
            if index == 0 and step.stall:
                await asyncio.sleep(step.stall)
        for index, (name, args) in enumerate(step.calls, start=1):
            played.calls_made += 1
            call_id = f"call-{played.calls_made}"
            yield {
                index: DeltaToolCall(name=name, json_args=json.dumps(args), tool_call_id=call_id)
            }

    return FunctionModel(stream_function=stream, model_name="script-llm")
