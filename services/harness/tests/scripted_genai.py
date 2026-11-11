"""A scripted google-genai client that answers a model's streamed calls without a network.

`ScriptedGenai` holds a list of `Reply` objects. Each `generate_content_stream`
call on its `client` answers with the next reply and records the request it was
sent. Pydantic AI's `GoogleProvider(client=...)` and ADK's `Gemini(client=...)`
both accept the client and run their own request building and response parsing
against it, so the recorded requests are what each would put on the wire.

The part builders (`text_part`, `call_part`, `thought_part`), `usage_metadata`,
`reply` and `Reply` describe a response; `Reply` options make a stream stall, fail
at open, be cut, come back empty or come back blocked.
"""

from __future__ import annotations

import asyncio
import copy
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from typing import Any

from google.genai import Client, errors, types


class ScriptExhaustedError(RuntimeError):
    """A request arrived after every scripted reply had been used."""


class StreamCutError(RuntimeError):
    """The default exception a reply raises when its stream is cut part-way."""


@dataclass(frozen=True)
class Request:
    """One `generate_content_stream` call as the model code made it."""

    model: str
    contents: Any
    config: Any

    @property
    def generate_config(self) -> types.GenerateContentConfig:
        """The config as a `GenerateContentConfig`, whether the caller sent a dict or a model."""
        return types.GenerateContentConfig.model_validate(self.config or {})

    @property
    def content_list(self) -> list[types.Content]:
        """The contents as `Content` objects, whether the caller sent dicts or models."""
        return [types.Content.model_validate(content) for content in self.contents]


@dataclass(frozen=True)
class Reply:
    """What one scripted call answers with.

    Args:
        chunks: The parts of each streamed chunk, in order. A reply with several
            chunks is a response split across them.
        usage: Usage metadata, attached to the last chunk.
        finish_reason: Finish reason, attached to the last chunk.
        stall_s: Seconds to wait before the first chunk.
        open_error: Raised when the stream is opened, before any chunk.
        cut_after: Number of chunks to deliver before the stream fails.
        cut_with: The exception raised at the cut, `StreamCutError` by default. Pass
            `asyncio.CancelledError()` for a cancellation.
        prompt_feedback: Prompt feedback sent in place of any candidate, as a
            blocked prompt is answered.
    """

    chunks: Sequence[Sequence[types.Part]] = ()
    usage: types.GenerateContentResponseUsageMetadata | None = None
    finish_reason: types.FinishReason | None = types.FinishReason.STOP
    stall_s: float = 0.0
    open_error: Exception | None = None
    cut_after: int | None = None
    cut_with: BaseException | None = None
    prompt_feedback: types.GenerateContentResponsePromptFeedback | None = None


def text_part(text: str) -> types.Part:
    """A part of plain answer text."""
    return types.Part(text=text)


def call_part(
    name: str,
    args: dict[str, Any] | None = None,
    *,
    id: str | None = None,
    signature: bytes | None = None,
) -> types.Part:
    """A function call part, with a call id and a thought signature when given."""
    return types.Part(
        function_call=types.FunctionCall(name=name, args=args or {}, id=id),
        thought_signature=signature,
    )


def thought_part(text: str, *, signature: bytes | None = b"scripted-signature") -> types.Part:
    """A thought summary part carrying a thought signature."""
    return types.Part(text=text, thought=True, thought_signature=signature)


def usage_metadata(
    *,
    prompt: int = 0,
    cached: int = 0,
    candidates: int = 0,
    thoughts: int = 0,
) -> types.GenerateContentResponseUsageMetadata:
    """Usage metadata with the given prompt, cached, candidate and thought token counts."""
    return types.GenerateContentResponseUsageMetadata(
        prompt_token_count=prompt,
        cached_content_token_count=cached or None,
        candidates_token_count=candidates,
        thoughts_token_count=thoughts or None,
        total_token_count=prompt + candidates + thoughts,
    )


def reply(
    *parts: types.Part,
    usage: types.GenerateContentResponseUsageMetadata | None = None,
    finish_reason: types.FinishReason | None = types.FinishReason.STOP,
) -> Reply:
    """A reply that streams `parts` as one chunk."""
    return Reply(chunks=[parts], usage=usage, finish_reason=finish_reason)


def empty_reply() -> Reply:
    """A reply whose one candidate carries no parts."""
    return Reply(chunks=[[]])


def blocked_reply(
    reason: types.BlockedReason = types.BlockedReason.SAFETY,
) -> Reply:
    """A reply that is prompt feedback with a block reason and no candidate."""
    return Reply(
        chunks=[[]],
        finish_reason=None,
        prompt_feedback=types.GenerateContentResponsePromptFeedback(block_reason=reason),
    )


def client_error(status: int, message: str = "scripted refusal") -> errors.ClientError:
    """The error google-genai raises for an HTTP `status` answer, for `Reply.open_error`."""
    return errors.ClientError(status, {"error": {"code": status, "message": message}})


class ScriptedGenai:
    """A google-genai client that answers streamed calls from `replies`, in order.

    `client` is a real `google.genai.Client` whose
    `aio.models.generate_content_stream` is replaced, so everything else a model
    class reads off a client (`vertexai`, HTTP options) is present. `requests`
    holds a copy of each call's model, contents and config in call order.
    """

    def __init__(self, replies: Sequence[Reply]) -> None:
        self.replies = list(replies)
        self.requests: list[Request] = []
        self.client = Client(api_key="scripted")
        self.client.aio.models.generate_content_stream = self._open  # type: ignore[method-assign]

    async def _open(
        self, *, model: str, contents: Any, config: Any = None
    ) -> AsyncIterator[types.GenerateContentResponse]:
        self.requests.append(
            Request(model=model, contents=copy.deepcopy(contents), config=copy.deepcopy(config))
        )
        if len(self.requests) > len(self.replies):
            raise ScriptExhaustedError(
                f"{len(self.requests)} requests were made against "
                f"{len(self.replies)} scripted replies"
            )
        step = self.replies[len(self.requests) - 1]
        if step.open_error is not None:
            raise step.open_error
        return _stream(step)


async def _stream(step: Reply) -> AsyncIterator[types.GenerateContentResponse]:
    if step.stall_s:
        await asyncio.sleep(step.stall_s)
    last = len(step.chunks) - 1
    for position, parts in enumerate(step.chunks):
        if position == step.cut_after:
            raise step.cut_with or StreamCutError()
        yield _chunk(step, list(parts), final=position == last)
    if step.cut_after == len(step.chunks):
        raise step.cut_with or StreamCutError()


def _chunk(step: Reply, parts: list[types.Part], *, final: bool) -> types.GenerateContentResponse:
    # A copy per chunk: ADK writes its own call ids into the parts it is handed.
    parts = copy.deepcopy(parts)
    if step.prompt_feedback is not None:
        return types.GenerateContentResponse(prompt_feedback=step.prompt_feedback)
    content = types.Content(role="model", parts=parts) if parts else None
    return types.GenerateContentResponse(
        candidates=[
            types.Candidate(content=content, finish_reason=step.finish_reason if final else None)
        ],
        usage_metadata=step.usage if final else None,
    )
