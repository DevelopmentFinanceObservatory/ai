"""A scripted google-genai client that answers a model's streamed calls without a network.

`ScriptedGenai` holds a list of `Reply` objects. Each `generate_content_stream`
call on its `client` answers with the next reply and records the request it was
sent. Pydantic AI's `GoogleProvider(client=...)` and ADK's `Gemini(client=...)`
both accept the client and run their own request building and response parsing
against it, so the recorded requests are what each would put on the wire.

The part builders (`text_part`, `call_part`, `thought_part`), `usage_metadata`,
`reply` and `Reply` describe a response; `Reply` options make a stream stall, fail
at open, be cut, come back empty or come back blocked. A script entry may instead
be a function of the request it answers (`Step`), which computes the reply from
what the model was sent, or runs code when the call arrives; `held` is one that
waits for a test to release it.
"""

from __future__ import annotations

import asyncio
import copy
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Any

from google.genai import Client, errors, types


class ScriptExhaustedError(RuntimeError):
    """A request arrived after every scripted reply had been used."""


class UnscriptedCallError(RuntimeError):
    """A call the scripted client does not script, which would otherwise reach the network."""


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


# A script entry: a reply, or a function of the request it answers that returns
# one, awaited when it returns an awaitable. The function runs in the caller's
# context when the call arrives, after the request is recorded.
type Step = Reply | Callable[[Request], Reply | Awaitable[Reply]]


def held(step: Step, *, paused: asyncio.Event, release: asyncio.Event) -> Step:
    """`step`, answered only once `release` is set: the call sets `paused` when
    it arrives, then waits, so a test can act while the generation is held."""

    async def answer(request: Request) -> Reply:
        paused.set()
        await release.wait()
        return await _resolved(step, request)

    return answer


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


def live_call_reply(
    *calls: types.Part,
    usage: types.GenerateContentResponseUsageMetadata | None = None,
) -> Reply:
    """A function-call reply as live Gemini streams one: a chunk of the calls,
    then a chunk of one empty text part, which carries the finish reason and
    the usage."""
    return Reply(chunks=[calls, [types.Part(text="")]], usage=usage)


def malformed_call_reply(
    usage: types.GenerateContentResponseUsageMetadata | None = None,
) -> Reply:
    """A reply as live Gemini streams a function call it could not parse: two
    chunks of one empty text part each, the second signed and carrying the
    `MALFORMED_FUNCTION_CALL` finish reason."""
    return Reply(
        chunks=[
            [types.Part(text="")],
            [types.Part(text="", thought_signature=b"scripted-signature")],
        ],
        usage=usage,
        finish_reason=types.FinishReason.MALFORMED_FUNCTION_CALL,
    )


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

    Each entry of `replies` is a `Step`: a `Reply`, or a function of the
    request it answers.

    `client` is a real `google.genai.Client` with a scripted
    `aio.models.generate_content_stream`, so everything else a model class
    reads off a client (`vertexai`, HTTP options) is present. Every other
    call of its `models`, `aio.models`, `caches` and `aio.caches` raises
    `UnscriptedCallError` naming the call, so no test reaches the network. A test
    that scripts one of them assigns over it. `requests` holds a copy of each
    call's model, contents and config in call order.
    """

    def __init__(self, replies: Sequence[Step]) -> None:
        self.replies = list(replies)
        self.requests: list[Request] = []
        self.client = Client(api_key="scripted")
        namespaces = {
            "models": self.client.models,
            "aio.models": self.client.aio.models,
            "caches": self.client.caches,
            "aio.caches": self.client.aio.caches,
        }
        for path, namespace in namespaces.items():
            for name in dir(namespace):
                if not name.startswith("_") and callable(getattr(namespace, name)):
                    setattr(namespace, name, _refusing(f"{path}.{name}"))
        self.client.aio.models.generate_content_stream = self._open  # type: ignore[method-assign]

    async def _open(
        self, *, model: str, contents: Any, config: Any = None
    ) -> AsyncIterator[types.GenerateContentResponse]:
        request = Request(
            model=model, contents=copy.deepcopy(contents), config=copy.deepcopy(config)
        )
        self.requests.append(request)
        if len(self.requests) > len(self.replies):
            raise ScriptExhaustedError(
                f"{len(self.requests)} requests were made against "
                f"{len(self.replies)} scripted replies"
            )
        step = await _resolved(self.replies[len(self.requests) - 1], request)
        if step.open_error is not None:
            raise step.open_error
        return _stream(step)


async def _resolved(step: Step, request: Request) -> Reply:
    """The reply `step` answers `request` with."""
    if isinstance(step, Reply):
        return step
    answered = step(request)
    return answered if isinstance(answered, Reply) else await answered


def _refusing(call: str) -> Callable[..., Any]:
    """A stand-in for the client's `call` that refuses it."""

    def refuse(*args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        raise UnscriptedCallError(
            f"{call} was called; ScriptedGenai scripts aio.models.generate_content_stream only"
        )

    return refuse


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
    # Each chunk gets a copy because ADK writes its own call ids into the parts it is handed.
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
