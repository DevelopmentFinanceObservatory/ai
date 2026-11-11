"""The Gemini model the Pydantic AI planner runs on, kept to the turn's budget.

`planner_model` builds Pydantic AI's `GoogleModel` over a google-genai client
whose streamed call is wrapped. Each call the planner makes, in order:

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
   did, and the usage the API reported; a model turn it returns replaces the
   response;
6. otherwise delivers the held chunks less their unsigned thought summaries
   (`planner.settings.without_unsigned_thoughts`), so the planner's next
   request carries the parts the ADK planner's session keeps.

Every request streams. A non-streamed call refuses, since it would bypass all
of the above.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator, AsyncIterator, Callable
from contextlib import aclosing
from dataclasses import dataclass, field
from typing import Any, cast

from google.adk.models.llm_response import LlmResponse
from google.genai import Client, errors, types
from orchestrator.agent import model_retry_options
from planner.settings import without_unsigned_thoughts
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.profiles import ModelProfile
from pydantic_ai.providers.google import GoogleProvider

from harness.prefix_cache import CACHE_GONE_CODES, PrefixCache, against_cache
from harness.sentences import SeamCode
from harness.turn_budget import CLOSED_BY_KEY, TurnBudget, bounded_by

logger = logging.getLogger(__name__)

type Usage = types.GenerateContentResponseUsageMetadata

# A model turn to answer the next request with in place of the model, or None to send it.
type BeforeRequest = Callable[[], types.Content | None]

# Called with the code of the bound that closed the call (None when none did)
# and the usage the API reported. It gives a model turn to answer with in place
# of the response, or None to keep it.
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
            left unchanged; the model calls a wrapper over its streamed call.
        budget: The turn's budget, or None to run every call unbounded.
        prefix_cache: The process's shared cache of static prompt prefixes,
            or None to send every request whole.
        before_request: Run before each request (`BeforeRequest`).
        after_response: Run on each response the model gave or a bound
            closed (`AfterResponse`).
        profile: The model profile, which decides how requests are built;
            None for Gemini's own (`google_model_profile`).

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
    # `GoogleProvider` takes a `Client`; the wrapper reads as one everywhere
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
        """The last usage the API reported on the call."""
        reported = [chunk.usage_metadata for chunk in self.chunks if chunk.usage_metadata]
        return reported[-1] if reported else None


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
        return self._respond(model=model, contents=contents, config=_with_retries(config))

    async def _respond(
        self, *, model: str, contents: Any, config: types.GenerateContentConfig
    ) -> AsyncGenerator[types.GenerateContentResponse]:
        if self.before_request is not None and (turn := self.before_request()) is not None:
            yield _chunk(turn, usage=None)
            return
        answer = await self._send(model=model, contents=contents, config=config)
        usage = answer.usage
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
            for chunk in _without_unsigned_thoughts(answer.chunks):
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


def _with_retries(config: Any) -> types.GenerateContentConfig:
    """`config`, as `GoogleModel` built it, carrying the ADK planner's HTTP retries."""
    built = types.GenerateContentConfig.model_validate(config or {})
    http_options = built.http_options or types.HttpOptions()
    return built.model_copy(
        update={
            "http_options": http_options.model_copy(update={"retry_options": model_retry_options()})
        }
    )


def _chunk(content: types.Content, *, usage: Usage | None) -> types.GenerateContentResponse:
    """A model turn as the one streamed chunk google-genai would deliver for it."""
    return types.GenerateContentResponse(
        candidates=[types.Candidate(content=content, finish_reason=types.FinishReason.STOP)],
        usage_metadata=usage,
    )


def _without_unsigned_thoughts(
    chunks: list[types.GenerateContentResponse],
) -> list[types.GenerateContentResponse]:
    """`chunks`, one response's, less the parts of each merged part
    `without_unsigned_thoughts` drops.

    The rule reads the response as an agent holds it: ADK's stream aggregator
    and Pydantic AI's parts manager both merge consecutive text parts of one
    kind, thought or answer, into one part that keeps the first thought
    signature among them (`_text_runs`). A merged thought part is dropped or
    kept whole, so a run of thought chunks with a signature on any one of them
    stays, as on the ADK planner.
    """
    runs = _text_runs([part for chunk in chunks for part in _parts(chunk)])
    merged = [_merged(run) for run in runs]
    kept = without_unsigned_thoughts(merged)
    if kept is None:
        return chunks
    kept_runs = {id(part) for part in kept}
    kept_ids = {
        id(part)
        for run, whole in zip(runs, merged, strict=True)
        if id(whole) in kept_runs
        for part in run
    }
    return [_keeping(chunk, kept_ids) for chunk in chunks]


def _text_runs(parts: list[types.Part]) -> list[list[types.Part]]:
    """`parts` in runs: consecutive parts with text and one `thought` value
    together, every other part alone."""
    runs: list[list[types.Part]] = []
    for part in parts:
        last = runs[-1][-1] if runs else None
        if last is not None and part.text and last.text and part.thought == last.thought:
            runs[-1].append(part)
        else:
            runs.append([part])
    return runs


def _merged(run: list[types.Part]) -> types.Part:
    """The one part an agent makes of `run`."""
    if len(run) == 1:
        return run[0]
    signature = next((part.thought_signature for part in run if part.thought_signature), None)
    return types.Part(
        text="".join(part.text or "" for part in run),
        thought=run[0].thought,
        thought_signature=signature,
    )


def _parts(chunk: types.GenerateContentResponse) -> list[types.Part]:
    """The parts of `chunk`'s first candidate, the one ADK and Pydantic AI read."""
    if not chunk.candidates or chunk.candidates[0].content is None:
        return []
    return chunk.candidates[0].content.parts or []


def _keeping(
    chunk: types.GenerateContentResponse, kept_ids: set[int]
) -> types.GenerateContentResponse:
    """`chunk` with only the parts of its first candidate whose ids are in `kept_ids`."""
    if not chunk.candidates:
        return chunk
    first, *rest = chunk.candidates
    content = first.content
    if content is None:
        return chunk
    parts = [part for part in content.parts or [] if id(part) in kept_ids]
    first = first.model_copy(update={"content": content.model_copy(update={"parts": parts})})
    return chunk.model_copy(update={"candidates": [first, *rest]})


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


__all__ = ["AfterResponse", "BeforeRequest", "Usage", "planner_model"]
