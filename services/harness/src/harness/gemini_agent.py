"""What a Pydantic AI agent on Gemini needs of the harness, whichever agent it is.

`GEMINI_PROFILE` is the model profile an agent's `GoogleModel` is built on,
the one `harness.gemini_wire.in_canonical_form` expects. `replayed_history`
is the history processor that puts a run's messages in the form they are
stored and replayed in. `response_parts` turns a model response back into the
google-genai parts Gemini sent, and `as_function_response` a tool's answer
into the function response the model reads. `RecordingState` is the view a
model response's tool calls write state through, so the keys they wrote are
known.
"""

from __future__ import annotations

import base64
from collections.abc import Awaitable, Callable, MutableMapping, Sequence
from dataclasses import replace
from typing import Any, cast

from google.genai import types
from pydantic_ai.exceptions import UnexpectedModelBehavior
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelRequestPart,
    ModelResponse,
    ModelResponsePart,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.profiles.google import GoogleModelProfile

# The harness's changes to Gemini's profile, which `GoogleModel(profile=...)`
# merges over the provider's own for the model. Declarations are sent as given
# (no JSON schema transformer), and nothing is added to the instruction about
# system messages mid-conversation (inline system prompts supported).
GEMINI_PROFILE = GoogleModelProfile(
    json_schema_transformer=None, supports_inline_system_prompts=True
)


class RecordingState:
    """A view over a state mapping that reads and writes through to it and
    records each key written through the view, in first-write order.

    The view is a `planner.plan.SessionState`. Writes made to the mapping by
    anything but the view are unrecorded.
    """

    def __init__(self, state: MutableMapping[str, Any]) -> None:
        self._state = state
        self._written: dict[str, None] = {}

    @property
    def written(self) -> tuple[str, ...]:
        """Every key written through the view, in first-write order."""
        return tuple(self._written)

    def get(self, key: str, default: Any = None, /) -> Any:
        return self._state.get(key, default)

    def __getitem__(self, key: str) -> Any:
        return self._state[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self._state[key] = value
        self._written[key] = None

    def __contains__(self, key: str) -> bool:
        return key in self._state


class EmptyResponseError(UnexpectedModelBehavior):
    """The model's response held neither answer text nor a function call
    (`holds_no_answer`), so the run has nothing to go on with."""


def holds_no_answer(response: ModelResponse) -> bool:
    """Whether `response` holds neither answer text nor a function call: no
    parts, thoughts alone, or text that is empty.

    Neither agent sends such a response back for another try. Pydantic AI
    would, with a retry prompt added to the instruction, which breaks the
    one request form (`harness.gemini_wire.in_canonical_form`) and the cached
    prefix, and which no Gemini model of the harness was ever sent. The
    planner's run ends on it with no output (`planner.pydantic_agent`), and
    the orchestrator's raises `EmptyResponseError` (`orchestrator.run`).
    """
    return not any(
        isinstance(part, ToolCallPart) or (isinstance(part, TextPart) and part.content)
        for part in response.parts
    )


def tool_not_found(name: str, tools: Sequence[str]) -> dict[str, str]:
    """The answer to a call of `name`, a tool the agent lacks, naming the tools
    it can call in the order they are declared. It is google-adk's answer to a
    call of a tool an agent lacks, word for word, so stored conversations read
    the same."""
    available = ", ".join(tools) or "none"
    return {
        "error": (
            f"Invoking `{name}()` failed as no tool with that name is available. "
            f"The tools you can call are: {available}. You could retry, but it is "
            "IMPORTANT that you only call a tool from that list."
        )
    }


def replayed_history(
    tools: Sequence[str],
) -> Callable[[list[ModelMessage]], Awaitable[list[ModelMessage]]]:
    """A history processor that puts a run's messages in the form they are
    stored: each call to a tool not in `tools` answered with `tool_not_found`,
    in place of the retry Pydantic AI answers it with, and each response less
    its unsigned thought summaries (`without_unsigned_thoughts`). It runs
    before every model request, and Pydantic AI keeps what it returns as the
    run's messages."""

    async def process(messages: list[ModelMessage]) -> list[ModelMessage]:
        return [
            replace(message, parts=[_answered_if_unknown(part, tools) for part in message.parts])
            if isinstance(message, ModelRequest)
            else without_unsigned_thoughts(message)
            for message in messages
        ]

    return process


def _answered_if_unknown(part: ModelRequestPart, tools: Sequence[str]) -> ModelRequestPart:
    """`part`, or `tool_not_found`'s answer in its place when it answers a call
    to a tool not in `tools`."""
    if not isinstance(part, ToolReturnPart) or part.tool_name in tools:
        return part
    return ToolReturnPart(
        tool_name=part.tool_name,
        content=tool_not_found(part.tool_name, tools),
        tool_call_id=part.tool_call_id,
    )


def without_unsigned_thoughts(response: ModelResponse) -> ModelResponse:
    """`response` less each thought summary that carries no thought signature,
    unless every part is one, so no response is left empty.

    A part with a signature is kept whether or not it is a thought, since
    Gemini requires the signature on a later request. Pydantic AI's parts
    manager merges a run of streamed thought chunks into one `ThinkingPart`,
    signed when any chunk of the run was.
    """
    kept = [part for part in response.parts if not _unsigned_thought(part)]
    if not kept or len(kept) == len(response.parts):
        return response
    return replace(response, parts=kept)


def _unsigned_thought(part: ModelResponsePart) -> bool:
    return isinstance(part, ThinkingPart) and not _signature(part)


def _signature(part: ModelResponsePart) -> bytes | None:
    """The thought signature Gemini sent on `part`, which Pydantic AI keeps
    base64-encoded in the part's `provider_details`."""
    encoded = (part.provider_details or {}).get("thought_signature")
    return base64.b64decode(encoded) if encoded else None


def response_parts(response: ModelResponse) -> tuple[types.Part, ...]:
    """`response`'s parts as the google-genai parts Gemini sent for them, each
    with the thought signature it carried.

    Raises:
        ValueError: A part is none of a function call, text or a thought.
    """
    parts: list[types.Part] = []
    for part in response.parts:
        signature = _signature(part)
        match part:
            case ToolCallPart():
                call = types.FunctionCall(
                    id=part.tool_call_id, name=part.tool_name, args=part.args_as_dict()
                )
                parts.append(types.Part(function_call=call, thought_signature=signature))
            case TextPart():
                parts.append(types.Part(text=part.content, thought_signature=signature))
            case ThinkingPart():
                parts.append(
                    types.Part(text=part.content, thought=True, thought_signature=signature)
                )
            case _:
                raise ValueError(f"the response holds a {part.part_kind} part")
    return tuple(parts)


def as_function_response(value: object) -> dict[str, Any]:
    """`value` as a function response carries it: a dict as it is, anything else
    under `result`."""
    if isinstance(value, dict):
        # A function response's keys are the JSON object's, so strings.
        return cast("dict[str, Any]", value)
    return {"result": value}


__all__ = [
    "GEMINI_PROFILE",
    "EmptyResponseError",
    "RecordingState",
    "as_function_response",
    "holds_no_answer",
    "replayed_history",
    "response_parts",
    "tool_not_found",
    "without_unsigned_thoughts",
]
