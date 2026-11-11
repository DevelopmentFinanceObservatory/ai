"""The Gemini model the Pydantic AI planner runs on, kept to the turn's budget.

`planner_model` builds Pydantic AI's `GoogleModel` over a google-genai client
whose streamed call is wrapped. Each request is first put in the form the ADK
planner sends (`_in_adk_planner_form`). Each call the planner makes, in order:

1. asks `before_request` for a model turn to answer with; given one, the call
   answers with it and sends nothing, so no budget is spent;
2. looks the request's static prefix up in the shared `PrefixCache` and sends
   the request against it (`harness.prefix_cache.against_cache`). A cache the
   API reports gone (`CACHE_GONE_CODES`) before any chunk arrived is
   forgotten, looked up again and the request resent once, as
   `PrefixCachedGemini` does;
3. sends the request under the turn's `TurnBudget` (`harness.turn_budget.bounded_by`),
   with the model-time, call-cap and wall-clock bounds, the first-chunk
   resend, and the HTTP retries (`orchestrator.agent.model_retry_options`)
   the ADK planner's model gets;
4. holds every streamed chunk until the budget wrapper returns. ADK hands its
   agent a function call only once the stream has ended, and Pydantic AI would
   run a tool from the chunk that carried it, so a call the budget cuts after a
   function call was streamed runs no tool. A call a bound closes answers with
   the closing sentence the ADK planner's model gives;
5. hands `after_response` the code of the bound that closed the call, if one
   did, and the usage the API reported for it; a model turn it returns
   replaces the response; otherwise the held chunks are delivered.

A function call the model sent without an id reaches the planner with one of
the form ADK gives it (`_with_client_call_ids`), so both planners store it
under such an id, and each request is sent less those ids, as the ADK planner
sends it (`_without_client_call_ids`). Every request streams. A non-streamed
call refuses, since it would bypass all of the above.

Each call's span, Pydantic AI's `chat <model>`, is current while the call runs
and carries what Langfuse reads of the ADK planner's `generate_content` span
that Pydantic AI leaves out (`_record_usage`). That is the reasoning tokens and
the usage Langfuse derives from the ADK planner's span, set outright so that the
usage details Pydantic AI adds are not summed into the call's total. A turn put
in place of a request is no model call and the ADK planner opens no span for
it, so its span is marked a plain span for Langfuse.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncGenerator, AsyncIterator, Callable
from contextlib import aclosing
from dataclasses import dataclass, field
from typing import Any, cast

from google.adk.flows.llm_flows import functions as adk_functions
from google.adk.models.llm_response import LlmResponse
from google.genai import Client, errors, types
from opentelemetry import trace
from orchestrator.agent import model_retry_options
from pydantic_ai.exceptions import ModelAPIError
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.profiles import ModelProfile
from pydantic_ai.providers.google import GoogleProvider

from harness.prefix_cache import CACHE_GONE_CODES, PrefixCache, against_cache
from harness.sentences import SeamCode
from harness.turn_budget import CLOSED_BY_KEY, TurnBudget, bounded_by

logger = logging.getLogger(__name__)

type Usage = types.GenerateContentResponseUsageMetadata

# Langfuse's `LangfuseOtelSpanAttributes.OBSERVATION_TYPE` is the observation
# type Langfuse gives a span, ahead of the one it infers from the span's GenAI
# attributes. `OBSERVATION_USAGE_DETAILS` is the usage it reads ahead of every
# `gen_ai.usage.*` attribute.
_OBSERVATION_TYPE = "langfuse.observation.type"
_USAGE_DETAILS = "langfuse.observation.usage_details"
# The OpenTelemetry GenAI attribute ADK records a call's thought tokens under.
_REASONING_TOKENS = "gen_ai.usage.reasoning.output_tokens"

# A model turn to answer the next request with in place of the model, or None to send it.
type BeforeRequest = Callable[[], types.Content | None]


# Called with the code of the bound that closed the call (None when none did)
# and the last usage any of its chunks carried. It gives a model turn to answer
# with in place of the response, or None to keep it.
type AfterResponse = Callable[[SeamCode | None, Usage | None], types.Content | None]


def planner_model(
    model_name: str,
    *,
    client: Client,
    budget: TurnBudget | None,
    prefix_cache: PrefixCache | None = None,
    before_request: BeforeRequest | None = None,
    after_response: AfterResponse | None = None,
    profile: ModelProfile | None = None,
) -> GoogleModel:
    """The `GoogleModel` the Pydantic AI planner runs on.

    Args:
        model_name: The Gemini model, as google-genai names it.
        client: The google-genai client the model's calls go through. It is
            left unchanged. The model calls a wrapper over its streamed call.
        budget: The turn's budget, or None to run every call unbounded.
        prefix_cache: The process's shared cache of static prompt prefixes,
            or None to send every request whole.
        before_request: Run before each request (`BeforeRequest`).
        after_response: Run on each response the model gave or a bound
            closed (`AfterResponse`).
        profile: The model profile, which decides how requests are built.
            None gives Gemini's own (`google_model_profile`).

    Returns:
        A `GoogleModel` on `profile` whose streamed calls go through the
        steps above.
    """
    stream = _PlannerStream(
        client=client,
        budget=budget,
        prefix_cache=prefix_cache,
        before_request=before_request,
        after_response=after_response,
    )
    # `GoogleProvider` takes a `Client`, and the wrapper reads as one everywhere
    # `GoogleModel` reads it.
    provider = GoogleProvider(client=cast(Client, _Client(client, stream)))
    return GoogleModel(model_name, provider=provider, profile=profile)


@dataclass
class _Answer:
    """One sent call: its streamed chunks, held, and the closing turn and code
    when a bound closed it."""

    chunks: list[types.GenerateContentResponse] = field(default_factory=list)
    closing: types.Content | None = None
    closed_by: SeamCode | None = None

    @property
    def usage(self) -> Usage | None:
        """The last usage any of the call's chunks carried, as ADK's stream
        aggregator reads it."""
        usages = [chunk.usage_metadata for chunk in self.chunks if chunk.usage_metadata]
        return usages[-1] if usages else None


@dataclass
class _PlannerStream:
    """The streamed call `GoogleModel` makes, with the planner's hooks, cache and budget."""

    client: Client
    budget: TurnBudget | None
    prefix_cache: PrefixCache | None
    before_request: BeforeRequest | None
    after_response: AfterResponse | None

    async def __call__(
        self, *, model: str, contents: Any, config: Any = None
    ) -> AsyncIterator[types.GenerateContentResponse]:
        return _with_client_call_ids(
            self._respond(
                model=model,
                contents=_without_client_call_ids(contents),
                config=_in_adk_planner_form(config),
            )
        )

    async def _respond(
        self, *, model: str, contents: Any, config: types.GenerateContentConfig
    ) -> AsyncGenerator[types.GenerateContentResponse]:
        span = trace.get_current_span()
        if self.before_request is not None and (turn := self.before_request()) is not None:
            if span.is_recording():
                span.set_attribute(_OBSERVATION_TYPE, "span")
            yield _chunk(turn, usage=None)
            return
        answer = await self._send(model=model, contents=contents, config=config)
        usage = answer.usage
        _record_usage(span, usage)
        replaced = (
            self.after_response(answer.closed_by, usage)
            if self.after_response is not None
            else None
        )
        if replaced is not None:
            yield _chunk(replaced, usage=usage)
        elif answer.closing is not None:
            yield _chunk(answer.closing, usage=usage)
        else:
            for chunk in answer.chunks:
                yield chunk

    async def _send(
        self, *, model: str, contents: Any, config: types.GenerateContentConfig
    ) -> _Answer:
        """The call sent against the request's cached prefix, and resent once,
        against a fresh cache or none, when the API reports the cache gone
        before any chunk arrived."""
        if self.prefix_cache is None:
            return await self._bounded(model=model, contents=contents, config=config)
        cached = await self.prefix_cache.lookup(client=self.client, model=model, config=config)
        if cached is None:
            return await self._bounded(model=model, contents=contents, config=config)

        key, name = cached
        answer = _Answer()
        try:
            return await self._bounded(
                model=model, contents=contents, config=against_cache(config, name), answer=answer
            )
        except errors.ClientError as exc:
            if answer.chunks or exc.code not in CACHE_GONE_CODES:
                raise
            logger.warning("prefix cache %s: %s gone (%s), recreating", key[:16], name, exc.code)

        self.prefix_cache.forget(key, name)
        cached = await self.prefix_cache.lookup(client=self.client, model=model, config=config)
        resent = config if cached is None else against_cache(config, cached[1])
        return await self._bounded(model=model, contents=contents, config=resent)

    async def _bounded(
        self,
        *,
        model: str,
        contents: Any,
        config: types.GenerateContentConfig,
        answer: _Answer | None = None,
    ) -> _Answer:
        """One call under the budget, its chunks held in `answer`."""
        answer = answer or _Answer()

        def request() -> AsyncGenerator[LlmResponse]:
            return self._raw(model=model, contents=contents, config=config, held=answer.chunks)

        async with aclosing(bounded_by(self.budget, request, stream=True)) as responses:
            async for response in responses:
                code = (response.custom_metadata or {}).get(CLOSED_BY_KEY)
                if code is not None:
                    answer.closing = response.content
                    answer.closed_by = SeamCode(code)
        return answer

    async def _raw(
        self,
        *,
        model: str,
        contents: Any,
        config: types.GenerateContentConfig,
        held: list[types.GenerateContentResponse],
    ) -> AsyncGenerator[LlmResponse]:
        """One attempt at the call: each chunk held, and handed to the budget as
        the partial response ADK's `Gemini` would make of it."""
        stream = await self.client.aio.models.generate_content_stream(
            model=model, contents=contents, config=config
        )
        async with aclosing(cast(AsyncGenerator[types.GenerateContentResponse], stream)) as chunks:
            async for chunk in chunks:
                held.append(chunk)
                content = chunk.candidates[0].content if chunk.candidates else None
                yield LlmResponse(
                    content=content, usage_metadata=chunk.usage_metadata, partial=True
                )


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


async def _with_client_call_ids(
    chunks: AsyncIterator[types.GenerateContentResponse],
) -> AsyncGenerator[types.GenerateContentResponse]:
    """`chunks`, each function call that arrived without an id given one as ADK
    gives it (`generate_client_function_call_id`).

    The ADK planner gives such a call an `adk-` id before its tools run and
    stores it under that id. Pydantic AI would invent an id of its own form, so
    this runs on every chunk the model hands Pydantic AI, a turn put in the
    model's place and a closing turn included. A chunk carries each call whole
    in one part, and both frameworks read each function call part as one call,
    so one part gets one id.
    """
    async for chunk in chunks:
        if not any(_id_less_calls(chunk)):
            yield chunk
            continue
        named = chunk.model_copy(deep=True)
        for call in _id_less_calls(named):
            call.id = adk_functions.generate_client_function_call_id()
        yield named


def _id_less_calls(chunk: types.GenerateContentResponse) -> list[types.FunctionCall]:
    """The function calls of `chunk`'s first candidate that carry no id."""
    return [
        part.function_call
        for part in _parts(chunk)
        if part.function_call is not None and not part.function_call.id
    ]


def _record_usage(span: trace.Span, usage: Usage | None) -> None:
    """Put `usage`, one call's, on `span` as Langfuse reads it from the ADK
    planner's `generate_content` span.

    ADK records a call's input (prompt and tool-use tokens), output (candidate
    and thought tokens), cached and thought tokens, each only when Gemini
    reported it, and nothing when it reported neither input nor output.
    Langfuse takes the cached tokens out of the input and the thought tokens
    out of the output. Pydantic AI records the thought tokens only as a
    `gen_ai.usage.details.*` attribute, and Langfuse adds each of those to the
    call's total, so the thought tokens go on under ADK's attribute and the
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


def _without_client_call_ids(contents: Any) -> list[types.Content]:
    """`contents`, as `GoogleModel` built them, less the call ids ADK gives a
    call the model sent without one.

    A turn put in the model's place carries such an id
    (`harness.planner_node.directive_turn`), and a later request of the run
    replays it when the call does not end the run. The ADK planner strips those
    ids from every request it sends (`remove_client_function_call_id`), so this
    applies ADK's own rule to a copy of each content.
    """
    stripped = [types.Content.model_validate(content).model_copy(deep=True) for content in contents]
    for content in stripped:
        adk_functions.remove_client_function_call_id(content)
    return stripped


# The parameters schema Pydantic AI gives a declaration with no parameters
# (`pydantic_ai.models.google._function_declaration_from_tool`).
_NO_PARAMETERS_SCHEMA = {"type": "object", "properties": {}}


def _in_adk_planner_form(config: Any) -> types.GenerateContentConfig:
    """`config`, as `GoogleModel` built it, put in the form the ADK planner sends.

    Gemini's behaviour follows the request's bytes, not only its meaning: a
    parameterless tool declared with an empty object schema, where the ADK
    planner declares none, made Gemini reason for some 12,000 tokens on one
    call in six of a live task's entries, and never without it. The prefix
    cache is keyed on the same fields (`harness.prefix_cache.prefix_key`), so a
    request in this form shares the ADK planner's cache. The request that
    creates the cache is this one, so the cache holds the same form.

    Each field Pydantic AI sends otherwise is rewritten, and kept as it came
    when it holds anything the ADK planner's form has no equal for:

    - the system instruction, one user `Content` of one text part, becomes
      that text;
    - one `Tool` per function declaration becomes one `Tool` of them all, in
      order;
    - a declaration whose parameters schema is the empty object
      (`_NO_PARAMETERS_SCHEMA`) loses it;
    - a tool config of nothing but the `AUTO` function calling mode is
      dropped, as `AUTO` is Gemini's default with tools;
    - `response_modalities` of `TEXT` alone is dropped, as text is the
      default;
    - the ADK planner's HTTP retries (`model_retry_options`) and
      `automatic_continuation` are set, as `planner.agent.create_planner_agent`
      sets them.

    Each framework's own HTTP headers stay, as does the placeholder signature
    Pydantic AI gives an unsigned replayed call in the contents.
    """
    built = types.GenerateContentConfig.model_validate(config or {})
    http_options = built.http_options or types.HttpOptions()
    return built.model_copy(
        update={
            "system_instruction": _instruction_as_text(built.system_instruction),
            "tools": _as_one_tool(built.tools),
            "tool_config": None if _only_auto(built.tool_config) else built.tool_config,
            "response_modalities": (
                None
                if built.response_modalities == [types.Modality.TEXT]
                else built.response_modalities
            ),
            "http_options": http_options.model_copy(
                update={"retry_options": model_retry_options()}
            ),
            "automatic_continuation": True,
        }
    )


def _instruction_as_text(instruction: Any) -> Any:
    """`instruction` as its text when it is one `Content` of one text part,
    else as it is."""
    if not isinstance(instruction, types.Content) or not instruction.parts:
        return instruction
    if len(instruction.parts) != 1:
        return instruction
    part = instruction.parts[0]
    dumped = part.model_dump(exclude_none=True)
    return part.text if dumped.keys() == {"text"} else instruction


def _as_one_tool(tools: list[Any] | None) -> list[Any] | None:
    """`tools` as one `Tool` holding every function declaration in order, each
    less an empty parameters schema, when every tool holds function
    declarations alone, else as they are."""
    if not tools or not all(
        isinstance(tool, types.Tool)
        and tool.model_dump(exclude_none=True).keys() == {"function_declarations"}
        for tool in tools
    ):
        return tools
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
    if declaration.parameters_json_schema != _NO_PARAMETERS_SCHEMA:
        return declaration
    return declaration.model_copy(update={"parameters_json_schema": None})


def _only_auto(tool_config: types.ToolConfig | None) -> bool:
    """Whether `tool_config` sets nothing but the `AUTO` function calling mode."""
    return tool_config is not None and tool_config.model_dump(mode="json", exclude_none=True) == {
        "function_calling_config": {"mode": "AUTO"}
    }


def _chunk(content: types.Content, *, usage: Usage | None) -> types.GenerateContentResponse:
    """A model turn as the one streamed chunk google-genai would deliver for it."""
    return types.GenerateContentResponse(
        candidates=[types.Candidate(content=content, finish_reason=types.FinishReason.STOP)],
        usage_metadata=usage,
    )


def _parts(chunk: types.GenerateContentResponse) -> list[types.Part]:
    """The parts of `chunk`'s first candidate, the one ADK and Pydantic AI read."""
    if not chunk.candidates or chunk.candidates[0].content is None:
        return []
    return chunk.candidates[0].content.parts or []


class _Models:
    """`client.aio.models` with `generate_content_stream` answered by `stream`."""

    def __init__(self, models: Any, stream: _PlannerStream) -> None:
        self._models = models
        self.generate_content_stream = stream

    async def generate_content(self, **_: Any) -> types.GenerateContentResponse:
        raise TypeError("the planner's model streams every request")

    def __getattr__(self, name: str) -> Any:
        return getattr(self._models, name)


class _AsyncClient:
    """`client.aio` with `_Models` in place of its models."""

    def __init__(self, aio: Any, stream: _PlannerStream) -> None:
        self._aio = aio
        self.models = _Models(aio.models, stream)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._aio, name)


class _Client:
    """`client` with `_AsyncClient` in place of its `aio`, read by `GoogleModel`
    as the client itself."""

    def __init__(self, client: Client, stream: _PlannerStream) -> None:
        self._client = client
        self.aio = _AsyncClient(client.aio, stream)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)


__all__ = [
    "AfterResponse",
    "BeforeRequest",
    "StreamResumedError",
    "Usage",
    "planner_model",
    "resumed_over_output",
]
