"""Tests for `harness.prefix_cache`: one shared cache per static prefix.

A fake genai client records every `caches.create` and every generate call's
model and config, and fails on cue the way Gemini and its transports do. A
request against a gone cache raises 403 when the response is first read,
streamed or not.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any

import aiohttp
import httpx
import pytest
from google.adk.models.google_llm import Gemini
from google.adk.models.llm_request import LlmRequest
from google.genai import Client, errors, types
from harness.prefix_cache import (
    CACHE_TTL_S,
    CREATE_COOLDOWN_S,
    REFRESH_MARGIN_S,
    PrefixCache,
    PrefixCachedGemini,
)

MODEL = "gemini-3.8-flash"
TOOL = types.Tool(
    function_declarations=[types.FunctionDeclaration(name="lookup", description="Look up.")]
)


def _gone() -> errors.ClientError:
    return errors.ClientError(
        403,
        {
            "error": {
                "code": 403,
                "message": "CachedContent not found (or permission denied)",
                "status": "PERMISSION_DENIED",
            }
        },
    )


def _chunk(text: str) -> types.GenerateContentResponse:
    return types.GenerateContentResponse(
        candidates=[
            types.Candidate(
                content=types.Content(role="model", parts=[types.Part(text=text)]),
                finish_reason=types.FinishReason.STOP,
            )
        ]
    )


class _Caches:
    """`create` fails with each of `failures` in turn, None meaning success."""

    def __init__(self, failures: list[Exception | None], release: asyncio.Event | None) -> None:
        self.attempts = 0
        self.models: list[str] = []
        self.created: list[types.CreateCachedContentConfig] = []
        self._failures = failures
        self._release = release

    async def create(self, *, model: str, config: types.CreateCachedContentConfig) -> Any:
        self.attempts += 1
        if self._release is not None:
            await self._release.wait()
        if self._failures and (failure := self._failures.pop(0)) is not None:
            raise failure
        self.models.append(model)
        self.created.append(config)
        return types.CachedContent(name=f"cachedContents/c{len(self.created)}")


class _Models:
    def __init__(self, gone: set[str], gone_after_chunk: set[str], error: Exception | None) -> None:
        self.models: list[str] = []
        self.configs: list[types.GenerateContentConfig] = []
        self._gone = gone
        self._gone_after_chunk = gone_after_chunk
        self._error = error

    def _check(self, model: str, config: types.GenerateContentConfig) -> None:
        self.models.append(model)
        self.configs.append(config)
        if config.cached_content in self._gone:
            raise _gone()
        if self._error is not None:
            raise self._error

    async def generate_content(self, *, model: str, contents: Any, config: Any) -> Any:
        self._check(model, config)
        return _chunk("done")

    async def generate_content_stream(self, *, model: str, contents: Any, config: Any) -> Any:
        async def chunks() -> AsyncIterator[types.GenerateContentResponse]:
            self._check(model, config)
            yield _chunk("do")
            if config.cached_content in self._gone_after_chunk:
                raise _gone()
            yield _chunk("ne")

        return chunks()


class _FakeClient(Client):
    """A `Client` holding fake `caches` and `models`, with no API key or I/O."""

    def __init__(
        self,
        *,
        create_failures: list[Exception | None] | None = None,
        gone: set[str] | None = None,
        gone_after_chunk: set[str] | None = None,
        release_create: asyncio.Event | None = None,
        generate_error: Exception | None = None,
    ) -> None:
        self.cache_calls = _Caches(create_failures or [], release_create)
        self.model_calls = _Models(gone or set(), gone_after_chunk or set(), generate_error)

    @property
    def aio(self) -> Any:
        return SimpleNamespace(caches=self.cache_calls, models=self.model_calls)

    @property
    def vertexai(self) -> bool:
        return False

    def close(self) -> None:
        pass


class _Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def _model(cache: PrefixCache, client: _FakeClient) -> PrefixCachedGemini:
    return PrefixCachedGemini(model=MODEL, client=client, prefix_cache=cache)


def _request(system: str = "You are the planner.", *, model: str = MODEL) -> LlmRequest:
    return LlmRequest(
        model=model,
        contents=[types.Content(role="user", parts=[types.Part(text="question")])],
        config=types.GenerateContentConfig(
            system_instruction=system,
            tools=[TOOL],
            tool_config=types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(mode="AUTO")
            ),
        ),
    )


async def _call(model: Gemini, request: LlmRequest, *, stream: bool = False) -> list[str]:
    return [
        "".join(part.text or "" for part in response.content.parts)
        async for response in model.generate_content_async(request, stream=stream)
        if response.content and response.content.parts and not response.partial
    ]


def _sent_against(config: types.GenerateContentConfig) -> str | None:
    """The cache a sent config names, after checking it carries none of the
    fields the API refuses alongside one."""
    if config.cached_content is not None:
        assert config.system_instruction is None
        assert config.tools is None
        assert config.tool_config is None
    return config.cached_content


async def test_one_cache_serves_five_calls_across_two_sessions() -> None:
    cache = PrefixCache()
    client = _FakeClient()
    sessions = [_model(cache, client), _model(cache, client)]

    for model in [*sessions, *sessions, sessions[0]]:
        await _call(model, _request())

    assert len(client.cache_calls.created) == 1
    created = client.cache_calls.created[0]
    assert created.system_instruction == "You are the planner."
    assert created.tools == [TOOL]
    assert created.ttl == f"{CACHE_TTL_S}s"
    assert [_sent_against(config) for config in client.model_calls.configs] == [
        "cachedContents/c1"
    ] * 5


async def test_the_traced_request_keeps_its_instruction_and_tools() -> None:
    client = _FakeClient()
    request = _request()

    await _call(_model(PrefixCache(), client), request, stream=True)

    assert _sent_against(client.model_calls.configs[0]) == "cachedContents/c1"
    assert request.config.cached_content is None
    assert request.config.system_instruction == "You are the planner."
    assert request.config.tools == [TOOL]
    assert request.config.tool_config is not None


@pytest.mark.parametrize("stream", [False, True])
async def test_a_gone_cache_is_created_again_and_the_request_resent_once(stream: bool) -> None:
    cache = PrefixCache()
    client = _FakeClient(gone={"cachedContents/c1"})
    model = _model(cache, client)

    texts = await _call(model, _request(), stream=stream)
    await _call(model, _request(), stream=stream)

    assert texts == ["done"]
    assert len(client.cache_calls.created) == 2
    assert [_sent_against(config) for config in client.model_calls.configs] == [
        "cachedContents/c1",
        "cachedContents/c2",
        "cachedContents/c2",
    ]


async def test_a_gone_cache_that_cannot_be_created_again_is_resent_uncached() -> None:
    client = _FakeClient(
        gone={"cachedContents/c1"}, create_failures=[None, aiohttp.ClientPayloadError("reset")]
    )

    texts = await _call(_model(PrefixCache(), client), _request())

    assert texts == ["done"]
    assert [_sent_against(config) for config in client.model_calls.configs] == [
        "cachedContents/c1",
        None,
    ]


async def test_a_cache_gone_after_the_first_chunk_is_not_resent() -> None:
    client = _FakeClient(gone_after_chunk={"cachedContents/c1"})

    with pytest.raises(errors.ClientError) as raised:
        await _call(_model(PrefixCache(), client), _request(), stream=True)

    assert raised.value.code == 403
    assert len(client.cache_calls.created) == 1
    assert len(client.model_calls.configs) == 1


async def test_a_prefix_the_api_refuses_to_cache_is_sent_whole_from_then_on() -> None:
    too_small = errors.ClientError(
        400, {"error": {"code": 400, "message": "too small", "status": "INVALID_ARGUMENT"}}
    )
    client = _FakeClient(create_failures=[too_small])
    model = _model(PrefixCache(), client)

    await _call(model, _request())
    await _call(model, _request())

    assert client.cache_calls.created == []
    assert [_sent_against(config) for config in client.model_calls.configs] == [None, None]
    assert client.model_calls.configs[1].system_instruction == "You are the planner."


@pytest.mark.parametrize(
    "failure",
    [
        errors.ServerError(503, {"error": {"code": 503, "status": "UNAVAILABLE"}}),
        errors.ClientError(429, {"error": {"code": 429, "status": "RESOURCE_EXHAUSTED"}}),
        aiohttp.ClientPayloadError("response payload is not completed"),
        aiohttp.ClientConnectionError("connection reset"),
        httpx.ReadError("connection reset"),
        ConnectionResetError("connection reset"),
    ],
    ids=["503", "429", "aiohttp-payload", "aiohttp-connection", "httpx-read", "os-reset"],
)
async def test_a_transient_create_failure_goes_uncached_until_the_cooldown_ends(
    failure: Exception,
) -> None:
    clock = _Clock()
    client = _FakeClient(create_failures=[failure])
    model = _model(PrefixCache(clock=clock), client)

    await _call(model, _request())
    clock.now += CREATE_COOLDOWN_S - 1
    await _call(model, _request())
    attempts_in_cooldown = client.cache_calls.attempts
    clock.now += 1
    await _call(model, _request())

    assert attempts_in_cooldown == 1
    assert client.cache_calls.attempts == 2
    assert [_sent_against(config) for config in client.model_calls.configs] == [
        None,
        None,
        "cachedContents/c1",
    ]


async def test_a_model_error_on_the_uncached_request_reaches_the_caller() -> None:
    client = _FakeClient(
        create_failures=[aiohttp.ClientPayloadError("reset")],
        generate_error=errors.ServerError(500, {"error": {"code": 500, "status": "INTERNAL"}}),
    )

    with pytest.raises(errors.ServerError) as raised:
        await _call(_model(PrefixCache(), client), _request())

    assert raised.value.code == 500
    assert [_sent_against(config) for config in client.model_calls.configs] == [None]


async def test_a_create_that_never_returns_goes_uncached_within_the_bound() -> None:
    client = _FakeClient(release_create=asyncio.Event())
    model = _model(PrefixCache(create_timeout_s=0.05), client)

    async with asyncio.timeout(1):
        texts = await _call(model, _request())

    assert texts == ["done"]
    assert client.cache_calls.created == []
    assert [_sent_against(config) for config in client.model_calls.configs] == [None]


async def test_concurrent_calls_on_a_create_that_never_returns_share_one_bound() -> None:
    # Four calls waiting one bound each in turn would take 0.8 s.
    client = _FakeClient(release_create=asyncio.Event())
    cache = PrefixCache(create_timeout_s=0.2)

    async with asyncio.timeout(0.35):
        texts = await asyncio.gather(*(_call(_model(cache, client), _request()) for _ in range(4)))

    assert texts == [["done"]] * 4
    assert client.cache_calls.attempts == 1
    assert [_sent_against(config) for config in client.model_calls.configs] == [None] * 4


async def test_a_cancelled_call_propagates_and_leaves_the_shared_create_running() -> None:
    release = asyncio.Event()
    client = _FakeClient(release_create=release)
    cache = PrefixCache()
    cancelled = asyncio.create_task(_call(_model(cache, client), _request()))
    waiting = asyncio.create_task(_call(_model(cache, client), _request()))

    await asyncio.sleep(0.01)
    cancelled.cancel()
    with pytest.raises(asyncio.CancelledError):
        await cancelled
    release.set()
    await waiting

    assert client.cache_calls.attempts == 1
    assert [_sent_against(config) for config in client.model_calls.configs] == ["cachedContents/c1"]


async def test_concurrent_first_calls_on_one_prefix_create_one_cache() -> None:
    release = asyncio.Event()
    client = _FakeClient(release_create=release)
    cache = PrefixCache()
    calls = asyncio.gather(
        _call(_model(cache, client), _request()), _call(_model(cache, client), _request())
    )

    await asyncio.sleep(0.01)
    release.set()
    await calls

    assert len(client.cache_calls.created) == 1
    assert [_sent_against(config) for config in client.model_calls.configs] == [
        "cachedContents/c1"
    ] * 2


async def test_a_changed_instruction_gets_its_own_cache() -> None:
    client = _FakeClient()
    model = _model(PrefixCache(), client)

    await _call(model, _request("You are the planner."))
    await _call(model, _request("You are the orchestrator."))
    await _call(model, _request("You are the planner."))

    assert [created.system_instruction for created in client.cache_calls.created] == [
        "You are the planner.",
        "You are the orchestrator.",
    ]
    assert [_sent_against(config) for config in client.model_calls.configs] == [
        "cachedContents/c1",
        "cachedContents/c2",
        "cachedContents/c1",
    ]


async def test_a_cache_near_its_expiry_is_replaced_before_use() -> None:
    clock = _Clock()
    client = _FakeClient()
    model = _model(PrefixCache(clock=clock), client)

    await _call(model, _request())
    clock.now += CACHE_TTL_S - REFRESH_MARGIN_S - 1
    await _call(model, _request())
    clock.now += 2
    await _call(model, _request())

    assert [_sent_against(config) for config in client.model_calls.configs] == [
        "cachedContents/c1",
        "cachedContents/c1",
        "cachedContents/c2",
    ]


async def test_a_request_with_no_instruction_or_tools_goes_uncached() -> None:
    client = _FakeClient()
    request = LlmRequest(
        model=MODEL, contents=[types.Content(role="user", parts=[types.Part(text="q")])]
    )

    await _call(_model(PrefixCache(), client), request)

    assert client.cache_calls.created == []
    assert _sent_against(client.model_calls.configs[0]) is None


@pytest.mark.parametrize(
    ("model_name", "cached"),
    [("gemini-3.8-flash", True), ("gemma-3-27b-it", False), ("openai/gpt-4o", False)],
)
def test_only_a_gemini_model_is_wrapped(model_name: str, cached: bool) -> None:
    model = PrefixCache().model_for(model_name)

    assert isinstance(model, PrefixCachedGemini) is cached
    if not cached:
        assert model == model_name


async def test_a_provider_prefixed_gemini_name_is_cached_and_sent_without_its_prefix() -> None:
    wrapped = PrefixCache().model_for("Gemini:gemini-3.8-flash")
    assert isinstance(wrapped, PrefixCachedGemini)
    client = _FakeClient()
    model = wrapped.model_copy(update={"client": client})

    # ADK names the request's model from the agent's model (`basic.py`).
    await _call(model, _request(model=model.model))

    assert model.model == MODEL
    assert client.cache_calls.models == [MODEL]
    assert client.model_calls.models == [MODEL]
