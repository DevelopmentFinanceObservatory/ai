"""`harness.turn_budget` carried onto a Pydantic AI model through `WrapperModel`.

`BoundedModel` reads the request's `TurnBudget` (`current_turn_budget`) around each
streamed model request, as `BoundedGemini` does around each ADK call:

- a call that finds a bound spent makes no request and answers with that bound's
  sentence as the model's text, through a one-call `FunctionModel`;
- every other call counts against the call cap, and its stream waits under
  `asyncio.timeout` for the model time or wall time left, whichever is less.

What it does not carry over: the first-chunk resend, the span attributes, and the
closing sentence after a cut mid-stream. A cut here raises `TimeoutError` out of
the consumer's iteration, so the run fails; `BoundedGemini` instead ends the call
on the bound's sentence (`turn_budget._closing`). Doing that needs a
`StreamedResponse` of our own that relays the wrapped one's events and its parts.
"""

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from harness.sentences import refusal_sentence
from harness.turn_budget import current_turn_budget
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models import ModelRequestParameters, StreamedResponse
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.wrapper import WrapperModel
from pydantic_ai.settings import ModelSettings


def _saying(text: str) -> FunctionModel:
    async def say(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
        yield text

    return FunctionModel(stream_function=say, model_name="turn-budget")


class BoundedModel(WrapperModel):
    """The wrapped model, each streamed request kept to the current `TurnBudget`."""

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
        run_context: Any = None,
    ) -> AsyncIterator[StreamedResponse]:
        budget = current_turn_budget.get()
        if budget is None:
            async with self.wrapped.request_stream(
                messages, model_settings, model_request_parameters, run_context
            ) as response:
                yield response
            return
        if (spent := budget.spent_bound()) is not None:
            budget.close(spent)
            closing = _saying(refusal_sentence({"code": spent}))
            async with closing.request_stream(
                messages, model_settings, model_request_parameters, run_context
            ) as response:
                yield response
            return
        budget.model_calls += 1
        started = budget.clock()
        try:
            async with (
                asyncio.timeout(max(0.0, min(budget.wall_left(), budget.model_time_left()))),
                self.wrapped.request_stream(
                    messages, model_settings, model_request_parameters, run_context
                ) as response,
            ):
                yield response
        finally:
            budget.model_time_spent += budget.clock() - started
