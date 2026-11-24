"""The Gemini wire: how a model call reaches google-genai and what comes back.

`GeminiStream` is the streamed call a Pydantic AI `GoogleModel` makes, and
`wrapped_client` puts it in a google-genai client's place. Each request is
first put in one fixed form (`in_canonical_form`). Each call, in order:

1. asks `before_request` for a model turn to answer with; given one, the call
   answers with it and sends nothing, so no budget is spent;
2. looks the request's static prefix up in the shared `PrefixCache` and sends
   the request against it (`harness.prefix_cache.against_cache`). A cache the
   API reports gone (`CACHE_GONE_CODES`) before any chunk arrived is
   forgotten, looked up again and the request resent once;
3. sends the request under the turn's `TurnBudget`
   (`harness.turn_budget.bounded_by`), with the model-time, call-cap and
   wall-clock bounds, the first-chunk resend, and the HTTP retries
   (`model_retry_options`);
4. passes the streamed chunks on by the stream's rule (`_Relay`). A stream
   that holds (the planner's) holds every chunk until the call ends. Pydantic
   AI would run a tool from the chunk that carried it, so a call the budget
   cuts after a function call was streamed runs no tool. A stream that passes
   text on (the orchestrator's, whose answer reaches the user as it arrives)
   holds chunks until the first chunk of answer text and then passes them on
   in order, so the thought summaries a call opens with stay ahead of its
   text. From the first chunk holding a function call it holds again, to the
   call's end, for the same reason. A call a bound closes answers with the
   closing sentence `bounded_by` gives, as a paragraph of its own after text
   already passed on (`Closing.shown`), and its held chunks are dropped;
5. hands `after_response` the code of the bound that closed the call, if one
   did, and the usage the API reported for it; a model turn it returns
   replaces the response. Only a stream that holds takes the hook, since
   text passed on cannot be replaced;
6. when Gemini ended the call on `MALFORMED_FUNCTION_CALL`, no bound closed
   it and no chunk was passed on, discards the held chunks and runs steps 1
   to 5 once more, so the second attempt is counted, bounded and logged as a
   call of its own. The span carries both attempts' usage summed and the
   number of attempts discarded (`_MALFORMED_DISCARDED`). Otherwise, and
   after the second attempt whatever its outcome, the held chunks are
   delivered. After answer text was passed on the call ends as Gemini ended
   it, since what the user read cannot be taken back.

A function call the model sent without an id reaches the caller with one the
harness makes (`client_call_id`), so it is stored under that id, and each
request is sent less those ids (`_without_client_call_ids`), so the model
reads its calls as it sent them. Every request streams. A non-streamed call
refuses, since it would bypass all of the above.

Each call's span, Pydantic AI's `chat <model>`, is current while the call runs
and carries the usage Langfuse reads from a GenAI span, which Pydantic AI
leaves out (`_record_usage`). That is the reasoning tokens and the usage
Langfuse derives from such a span, set outright so that the usage details
Pydantic AI adds are not summed into the call's total. A turn put in
place of a request no attempt was sent for is no model call, so its span is
marked a plain span for Langfuse. Every attempt sent records on the span how
it ended, whether or not Gemini reported usage for it: the finish reason
Gemini gave (`_FINISH_REASON`, `none` when the stream carried none), or the
class of the exception the call raised (`_ERROR`). Pydantic AI's own
`gen_ai.response.finish_reasons` maps several of Gemini's reasons onto one
and leaves out a response that carried none.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from collections.abc import AsyncGenerator, AsyncIterator, Callable
from contextlib import aclosing
from dataclasses import dataclass, field
from typing import Any, cast

from google.genai import Client, errors, types
from opentelemetry import trace
from pydantic_ai.exceptions import ModelAPIError

from harness.prefix_cache import CACHE_GONE_CODES, PrefixCache, against_cache
from harness.sentences import StopCode
from harness.turn_budget import Chunk, Closing, TurnBudget, bounded_by

logger = logging.getLogger(__name__)

# The schema Pydantic AI declares a tool with when its declaration has none.
# Pydantic AI always sends `parameters_json_schema`, and `in_canonical_form`
# strips this value.
NO_PARAMETERS: dict[str, Any] = {"type": "object", "properties": {}}

type Usage = types.GenerateContentResponseUsageMetadata

# What every id `client_call_id` makes starts with. It is the prefix earlier
# releases gave such ids, so the planner records and conversations stored under
# them read the same.
CLIENT_CALL_ID_PREFIX = "adk-"

# A planner call is sent at most this many times. The second send happens when
# Gemini ends the first on a function call it could not parse.
_ATTEMPTS = 2

# Langfuse's `LangfuseOtelSpanAttributes.OBSERVATION_TYPE` is the observation
# type Langfuse gives a span, ahead of the one it infers from the span's GenAI
# attributes. `OBSERVATION_USAGE_DETAILS` is the usage it reads ahead of every
# `gen_ai.usage.*` attribute.
_OBSERVATION_TYPE = "langfuse.observation.type"
_USAGE_DETAILS = "langfuse.observation.usage_details"
# The OpenTelemetry GenAI attribute a call's thought tokens go under.
_REASONING_TOKENS = "gen_ai.usage.reasoning.output_tokens"
# The number of a call's attempts discarded for ending on a function call
# Gemini could not parse, set only when one was.
_MALFORMED_DISCARDED = "dfo.model.malformed_attempts_discarded"
# How the latest attempt sent ended: Gemini's finish reason as it named it, or
# `none` when no chunk carried one. `_ERROR` is the class of the exception the
# call raised, set only when it raised.
_FINISH_REASON = "dfo.model.finish_reason"
_ERROR = "dfo.model.error"

# A model turn to answer the next request with in place of the model, or None
# to send the request.
type BeforeRequest = Callable[[], types.Content | None]


# Called with the code of the bound that closed the call (None when none did)
# and the last usage any of its chunks carried. It gives a model turn to answer
# with in place of the response, or None to keep it.
type AfterResponse = Callable[[StopCode | None, Usage | None], types.Content | None]


def model_retry_options(*, attempts: int = 3, max_delay: float = 2.0) -> types.HttpRetryOptions:
    """The HTTP retries a Gemini model call makes: up to `attempts` attempts on a
    transient error (408, 429, 5xx), the first wait one second, no wait longer
    than `max_delay` seconds."""
    return types.HttpRetryOptions(attempts=attempts, initial_delay=1.0, max_delay=max_delay)


@dataclass
class _Relay:
    """One sent attempt's streamed chunks: those held, whether any was passed
    on, and how the call ended (step 4 of the module docstring).

    `passes_text` says whether the stream passes answer text on as it arrives
    or holds every chunk to the call's end.
    """

    passes_text: bool
    held: list[Chunk] = field(default_factory=list)
    passed: bool = False
    calling: bool = False
    usage: Usage | None = None
    finish_reason: str | None = None
    closing: Closing | None = None

    def take(self, chunk: Chunk) -> list[Chunk]:
        """The chunks to pass on now that `chunk` arrived, in order."""
        if chunk.usage_metadata is not None:
            # The last usage any chunk carried.
            self.usage = chunk.usage_metadata
        if chunk.candidates and (reason := chunk.candidates[0].finish_reason) is not None:
            self.finish_reason = reason.value
        self.calling = self.calling or any(part.function_call for part in _parts(chunk))
        if not self.passes_text or self.calling or not (self.passed or _has_answer_text(chunk)):
            self.held.append(chunk)
            return []
        passing = [*self.held, chunk]
        self.held = []
        self.passed = True
        return passing

    @property
    def closed_by(self) -> StopCode | None:
        """The bound that closed the call, if one did."""
        return self.closing.code if self.closing is not None else None

    @property
    def malformed(self) -> bool:
        """Whether Gemini ended the call, no bound having closed it, on a function
        call it could not parse."""
        return (
            self.closing is None
            and self.finish_reason == types.FinishReason.MALFORMED_FUNCTION_CALL.value
        )

    def closing_turn(self) -> types.Content:
        """The closing sentence as the model's turn: a paragraph of its own after
        text already passed on, else the sentence alone."""
        assert self.closing is not None
        text = self.closing.shown if self.passed else self.closing.sentence
        return types.Content(role="model", parts=[types.Part(text=text)])


@dataclass
class GeminiStream:
    """The streamed call `GoogleModel` makes, with its hooks, cache and budget.

    `passes_text` passes answer text on as it arrives; left false, every chunk
    is held to the call's end (step 4 of the module docstring).

    Raises:
        ValueError: A stream that passes text on was given `after_response`.
    """

    client: Client
    budget: TurnBudget | None
    prefix_cache: PrefixCache | None
    before_request: BeforeRequest | None
    after_response: AfterResponse | None
    passes_text: bool = False

    def __post_init__(self) -> None:
        if self.passes_text and self.after_response is not None:
            raise ValueError("a stream that passes text on cannot replace its response")

    async def __call__(
        self, *, model: str, contents: Any, config: Any = None
    ) -> AsyncIterator[Chunk]:
        return _with_client_call_ids(
            self._respond(
                model=model,
                contents=_without_client_call_ids(contents),
                config=in_canonical_form(config),
            )
        )

    async def _respond(
        self, *, model: str, contents: Any, config: types.GenerateContentConfig
    ) -> AsyncGenerator[Chunk]:
        span = trace.get_current_span()
        spent: Usage | None = None
        sent_any = False
        for attempt in range(_ATTEMPTS):
            if self.before_request is not None and (turn := self.before_request()) is not None:
                # After a sent attempt the span carries that call's usage, so it
                # stays a generation.
                if not sent_any and span.is_recording():
                    span.set_attribute(_OBSERVATION_TYPE, "span")
                yield _chunk(turn, usage=None)
                return
            relay = _Relay(passes_text=self.passes_text)
            try:
                async with aclosing(
                    self._send(model=model, contents=contents, config=config)
                ) as sent:
                    async for item in sent:
                        if isinstance(item, Closing):
                            relay.closing = item
                            continue
                        for chunk in relay.take(item):
                            yield chunk
            except (Exception, asyncio.CancelledError) as exc:
                if span.is_recording():
                    span.set_attribute(_ERROR, type(exc).__name__)
                raise
            unsent = relay.closing is not None and not relay.closing.sent and not sent_any
            if span.is_recording():
                if unsent:
                    # Nothing went to the model on any attempt, so the span is no
                    # generation.
                    span.set_attribute(_OBSERVATION_TYPE, "span")
                else:
                    span.set_attribute(_FINISH_REASON, relay.finish_reason or "none")
            sent_any = sent_any or (relay.closing is None or relay.closing.sent)
            usage = relay.usage
            spent = _added(spent, usage)
            _record_usage(span, spent)
            replaced = (
                self.after_response(relay.closed_by, usage)
                if self.after_response is not None
                else None
            )
            if (
                replaced is None
                and relay.malformed
                and not relay.passed
                and attempt + 1 < _ATTEMPTS
            ):
                logger.warning(
                    "model call to %s ended MALFORMED_FUNCTION_CALL after %s output tokens, "
                    "sending it again",
                    model,
                    usage.candidates_token_count if usage else None,
                )
                if span.is_recording():
                    span.set_attribute(_MALFORMED_DISCARDED, attempt + 1)
                continue
            if replaced is not None:
                yield _chunk(replaced, usage=usage)
            elif relay.closing is not None:
                yield _chunk(relay.closing_turn(), usage=usage)
            else:
                for chunk in relay.held:
                    yield chunk
            return

    async def _send(
        self, *, model: str, contents: Any, config: types.GenerateContentConfig
    ) -> AsyncGenerator[Chunk | Closing]:
        """The call sent against the request's cached prefix, or with no
        prefix cache as `config` stands, and resent once, against a fresh
        cache or none, when the API reports the cache gone before any chunk
        arrived."""
        cached = await self._cached(model=model, config=config)
        for resend in (False, True):
            sending = config if cached is None else against_cache(config, cached[1])
            arrived = False
            try:
                async with aclosing(
                    self._bounded(model=model, contents=contents, config=sending)
                ) as items:
                    async for item in items:
                        arrived = True
                        yield item
                return
            except errors.ClientError as exc:
                if resend or cached is None or arrived or exc.code not in CACHE_GONE_CODES:
                    raise
                key, name = cached
                logger.warning(
                    "prefix cache %s: %s gone (%s), recreating", key[:16], name, exc.code
                )
                assert self.prefix_cache is not None
                self.prefix_cache.forget(key, name)
                cached = await self._cached(model=model, config=config)

    async def _cached(
        self, *, model: str, config: types.GenerateContentConfig
    ) -> tuple[str, str] | None:
        """The prefix key and cache name `config` is sent against, or `None`
        with no prefix cache or no cache for it (`PrefixCache.lookup`)."""
        if self.prefix_cache is None:
            return None
        return await self.prefix_cache.lookup(client=self.client, model=model, config=config)

    def _bounded(
        self, *, model: str, contents: Any, config: types.GenerateContentConfig
    ) -> AsyncGenerator[Chunk | Closing]:
        """One call under the budget."""

        async def request() -> AsyncGenerator[Chunk]:
            stream = await self.client.aio.models.generate_content_stream(
                model=model, contents=contents, config=config
            )
            async with aclosing(cast(AsyncGenerator[Chunk], stream)) as chunks:
                async for chunk in chunks:
                    yield chunk

        return bounded_by(self.budget, request)


class StreamResumedError(ConnectionError):
    """A call's stream dropped and google-genai resumed it from a continuation
    token that does not cover all the output it had delivered, so Pydantic AI
    refused the repeat (`resumed_over_output`). It is a dropped connection,
    which a later attempt may not meet, and is raised from Pydantic AI's
    error."""


def resumed_over_output(exc: BaseException) -> bool:
    """Whether `exc` is the error Pydantic AI's Gemini stream parser
    (`pydantic_ai.models.google.GeminiStreamedResponse`) raises when
    google-genai resumed a dropped stream over output no continuation token
    covers.

    The parser raises that error as a plain `ModelAPIError` from no other
    exception. Each other plain `ModelAPIError` on the streaming path is
    raised from the error it maps: a google-genai `APIError` with a status
    below 400 (`_map_api_error`), or a body the SDK could not decode
    (`pydantic_ai.models._decode_errors.map_decode_errors`). Its subclasses
    carry a status or a category of their own. The message is not read.
    """
    return type(exc) is ModelAPIError and exc.__cause__ is None


def client_call_id() -> str:
    """A new id for a function call the model sent without one:
    `CLIENT_CALL_ID_PREFIX` and a random UUID."""
    return f"{CLIENT_CALL_ID_PREFIX}{uuid.uuid4()}"


async def _with_client_call_ids(chunks: AsyncIterator[Chunk]) -> AsyncGenerator[Chunk]:
    """`chunks`, each function call that arrived without an id given one
    (`client_call_id`).

    The call is stored under that id, given before its tool runs. Pydantic AI
    would invent an id of its own form, so this runs on every chunk the model
    hands Pydantic AI, a turn put in the model's place and a closing turn
    included. A chunk carries each call whole in one part, and Pydantic AI
    reads each function call part as one call, so one part gets one id.
    """
    async for chunk in chunks:
        if not any(_id_less_calls(chunk)):
            yield chunk
            continue
        named = chunk.model_copy(deep=True)
        for call in _id_less_calls(named):
            call.id = client_call_id()
        yield named


def _id_less_calls(chunk: Chunk) -> list[types.FunctionCall]:
    """The function calls of `chunk`'s first candidate that carry no id."""
    return [
        part.function_call
        for part in _parts(chunk)
        if part.function_call is not None and not part.function_call.id
    ]


def _added(spent: Usage | None, usage: Usage | None) -> Usage | None:
    """The token counts `_record_usage` reads, summed over `spent` and `usage`."""
    if spent is None or usage is None:
        return spent or usage
    counts = (
        "prompt_token_count",
        "tool_use_prompt_token_count",
        "candidates_token_count",
        "thoughts_token_count",
        "cached_content_token_count",
        "total_token_count",
    )
    total = spent.model_copy()
    for name in counts:
        before, after = getattr(spent, name), getattr(usage, name)
        if before is not None or after is not None:
            setattr(total, name, (before or 0) + (after or 0))
    return total


def _record_usage(span: trace.Span, usage: Usage | None) -> None:
    """Put `usage`, summed over a call's attempts (`_added`), on `span` as
    Langfuse reads it from a GenAI span.

    A call's usage is its input (prompt and tool-use tokens), output (candidate
    and thought tokens), cached and thought tokens, each only when Gemini
    reported it, and nothing when it reported neither input nor output.
    Langfuse takes the cached tokens out of the input and the thought tokens
    out of the output. Pydantic AI records the thought tokens only as a
    `gen_ai.usage.details.*` attribute, and Langfuse adds each of those to the
    call's total, so the thought tokens go on under `_REASONING_TOKENS` and the
    usage Langfuse derives goes on as its usage details, which it reads ahead
    of the rest.
    """
    if usage is None or not span.is_recording():
        return
    prompt, tool = usage.prompt_token_count, usage.tool_use_prompt_token_count
    candidates, thoughts = usage.candidates_token_count, usage.thoughts_token_count
    cached = usage.cached_content_token_count
    details: dict[str, int] = {}
    if prompt is not None or tool is not None:
        details["input"] = max((prompt or 0) + (tool or 0) - (cached or 0), 0)
    if candidates is not None or thoughts is not None:
        details["output"] = candidates or 0
    if not details:
        return
    if cached is not None:
        details["input_cached_tokens"] = cached
    if thoughts is not None:
        details["output_reasoning_tokens"] = thoughts
        span.set_attribute(_REASONING_TOKENS, thoughts)
    span.set_attribute(_USAGE_DETAILS, json.dumps(details))


def _without_client_call_ids(contents: Any) -> list[Any]:
    """`contents`, as `GoogleModel` built them, with each content that holds
    a function call or function response id starting with
    `CLIENT_CALL_ID_PREFIX` replaced by a copy less those ids. The other
    contents are passed on as they are, uncopied.

    A call the model sent without an id carries one (`_with_client_call_ids`),
    as does a turn put in the model's place
    (`harness.planner_adapter.directive_turn`), and a later request of the
    run replays it. The model sent no id, so it is sent none.
    """
    return [
        _stripped(content) if _holds_client_call_id(content) else content for content in contents
    ]


def _field(value: Any, name: str) -> Any:
    """`value`'s field `name`, whether `value` is a google-genai model or the
    dict form `GoogleModel` builds."""
    return value.get(name) if isinstance(value, dict) else getattr(value, name, None)


def _client_id(part: Any, field: str) -> bool:
    ref = _field(part, field)
    return ref is not None and (_field(ref, "id") or "").startswith(CLIENT_CALL_ID_PREFIX)


def _holds_client_call_id(content: Any) -> bool:
    return any(
        _client_id(part, "function_call") or _client_id(part, "function_response")
        for part in _field(content, "parts") or ()
    )


def _stripped(content: Any) -> types.Content:
    """A copy of `content` less its client call ids."""
    stripped = types.Content.model_validate(content).model_copy(deep=True)
    for part in stripped.parts or ():
        if (call := part.function_call) and (call.id or "").startswith(CLIENT_CALL_ID_PREFIX):
            call.id = None
        if (response := part.function_response) and (response.id or "").startswith(
            CLIENT_CALL_ID_PREFIX
        ):
            response.id = None
    return stripped


class UnexpectedRequestFormError(ValueError):
    """A request config `GoogleModel` built is not in a form
    `in_canonical_form` rewrites.

    Gemini's behaviour follows a request's bytes, so a form the rewrite does
    not know is refused.
    """


def in_canonical_form(config: Any) -> types.GenerateContentConfig:
    """`config`, as `GoogleModel` built it, put in the one form every request
    takes.

    Gemini's behaviour follows the request's bytes as well as its meaning. A
    parameterless tool declared with an empty object schema, where this form
    declares none, made Gemini reason for some 12,000 tokens on one call in six
    of a live task's entries, and never without it. The prefix cache is keyed
    on the same fields (`harness.prefix_cache.prefix_key`), and the request
    that creates the cache is in this form, so the cache holds the same form.

    Each field Pydantic AI sends is rewritten, and a value of any other shape
    raises `UnexpectedRequestFormError`:

    - the system instruction, one user `Content` of one text part, becomes
      that text;
    - one `Tool` per function declaration becomes one `Tool` of them all, in
      order;
    - a declaration whose parameters schema is the empty object
      (`NO_PARAMETERS`) loses it;
    - a tool config of nothing but the `AUTO` function calling mode is
      dropped, as `AUTO` is Gemini's default with tools;
    - `response_modalities` of `TEXT` alone is dropped, as text is the
      default;
    - the orchestrator's HTTP retries (`model_retry_options`) and
      `automatic_continuation` are set.

    Pydantic AI's own HTTP headers stay, as does the placeholder signature it
    gives an unsigned replayed call in the contents.
    """
    built = types.GenerateContentConfig.model_validate(config or {})
    _require_only_auto(built.tool_config)
    _require_only_text(built.response_modalities)
    http_options = built.http_options or types.HttpOptions()
    return built.model_copy(
        update={
            "system_instruction": _instruction_as_text(built.system_instruction),
            "tools": _as_one_tool(built.tools),
            "tool_config": None,
            "response_modalities": None,
            "http_options": http_options.model_copy(
                update={"retry_options": model_retry_options()}
            ),
            "automatic_continuation": True,
        }
    )


def _instruction_as_text(instruction: Any) -> str | None:
    """The text of `instruction`, one `Content` of one text part."""
    if isinstance(instruction, types.Content) and instruction.parts and len(instruction.parts) == 1:
        part = instruction.parts[0]
        if part.model_dump(exclude_none=True).keys() == {"text"}:
            return part.text
    raise UnexpectedRequestFormError(
        f"the system instruction is not one text part: {instruction!r}"
    )


def _as_one_tool(tools: list[Any] | None) -> list[types.Tool]:
    """`tools` as one `Tool` holding every function declaration in order, each
    less an empty parameters schema. Every tool must hold function
    declarations alone."""
    if not tools or not all(
        isinstance(tool, types.Tool)
        and tool.model_dump(exclude_none=True).keys() == {"function_declarations"}
        for tool in tools
    ):
        raise UnexpectedRequestFormError(
            f"the tools are not function declarations alone: {tools!r}"
        )
    return [
        types.Tool(
            function_declarations=[
                _without_empty_parameters(declaration)
                for tool in tools
                for declaration in tool.function_declarations or ()
            ]
        )
    ]


def _without_empty_parameters(
    declaration: types.FunctionDeclaration,
) -> types.FunctionDeclaration:
    if declaration.parameters_json_schema != NO_PARAMETERS:
        return declaration
    return declaration.model_copy(update={"parameters_json_schema": None})


def _require_only_auto(tool_config: types.ToolConfig | None) -> None:
    """Check that `tool_config` sets nothing but the `AUTO` function calling
    mode, which is Gemini's default with tools, so none need be sent."""
    if tool_config is None or tool_config.model_dump(mode="json", exclude_none=True) != {
        "function_calling_config": {"mode": "AUTO"}
    }:
        raise UnexpectedRequestFormError(
            f"the tool config is not the AUTO mode alone: {tool_config!r}"
        )


def _require_only_text(modalities: list[str] | None) -> None:
    """Check that `modalities` is `TEXT` alone, which is the default, so none
    need be sent."""
    if modalities != [types.Modality.TEXT]:
        raise UnexpectedRequestFormError(
            f"the response modalities are not TEXT alone: {modalities!r}"
        )


def _chunk(content: types.Content, *, usage: Usage | None) -> Chunk:
    """A model turn as the one streamed chunk google-genai would deliver for it."""
    return types.GenerateContentResponse(
        candidates=[types.Candidate(content=content, finish_reason=types.FinishReason.STOP)],
        usage_metadata=usage,
    )


def _parts(chunk: Chunk) -> list[types.Part]:
    """The parts of `chunk`'s first candidate, the one Pydantic AI reads."""
    if not chunk.candidates or chunk.candidates[0].content is None:
        return []
    return chunk.candidates[0].content.parts or []


def _has_answer_text(chunk: Chunk) -> bool:
    """Whether `chunk` carries answer text, which the user reads, as opposed to
    thought summaries."""
    return any(part.text and not part.thought for part in _parts(chunk))


def wrapped_client(client: Client, stream: GeminiStream) -> Client:
    """`client` with `stream` answering `aio.models.generate_content_stream`.

    The wrapper reads as a `Client` everywhere `GoogleModel` reads one, and
    leaves `client` unchanged.
    """
    return cast(Client, _Client(client, stream))


class _Models:
    """`client.aio.models` with `generate_content_stream` answered by `stream`."""

    def __init__(self, models: Any, stream: GeminiStream) -> None:
        self._models = models
        self.generate_content_stream = stream

    async def generate_content(self, **_: Any) -> types.GenerateContentResponse:
        raise TypeError("the model streams every request")

    def __getattr__(self, name: str) -> Any:
        return getattr(self._models, name)


class _AsyncClient:
    """`client.aio` with `_Models` in place of its models."""

    def __init__(self, aio: Any, stream: GeminiStream) -> None:
        self._aio = aio
        self.models = _Models(aio.models, stream)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._aio, name)


class _Client:
    """`client` with `_AsyncClient` in place of its `aio`, read by `GoogleModel`
    as the client itself."""

    def __init__(self, client: Client, stream: GeminiStream) -> None:
        self._client = client
        self.aio = _AsyncClient(client.aio, stream)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)


__all__ = [
    "CLIENT_CALL_ID_PREFIX",
    "NO_PARAMETERS",
    "AfterResponse",
    "BeforeRequest",
    "GeminiStream",
    "StreamResumedError",
    "UnexpectedRequestFormError",
    "Usage",
    "client_call_id",
    "in_canonical_form",
    "model_retry_options",
    "resumed_over_output",
    "wrapped_client",
]
