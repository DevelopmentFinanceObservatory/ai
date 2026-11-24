"""Tests for `harness.prefix_cache`: one shared cache per static prefix.

Each call goes through `harness.gemini_wire.GeminiStream`, the stream that
looks a request's prefix up and sends the request against its cache. A fake
genai client records every `caches.create` and every streamed call's model
and config, and fails on cue the way Gemini and its transports do. A request
against a gone cache raises 403 when its stream is first read.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any

import aiohttp
import httpx
import pytest
from google.genai import Client, errors, types
from harness.gemini_wire import GeminiStream
from harness.prefix_cache import (
    CACHE_TTL_S,
    CREATE_COOLDOWN_S,
    REFRESH_MARGIN_S,
    PrefixCache,
    gemini_model_name,
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


def _stream(cache: PrefixCache, client: _FakeClient) -> GeminiStream:
    return GeminiStream(
        client=client, budget=None, prefix_cache=cache, before_request=None, after_response=None
    )


def _config(system: str = "You are the planner.") -> types.GenerateContentConfig:
    """A request's config as Pydantic AI's `GoogleModel` builds it."""
    return types.GenerateContentConfig(
        system_instruction=types.Content(role="user", parts=[types.Part(text=system)]),
        tools=[TOOL],
        tool_config=types.ToolConfig(
            function_calling_config=types.FunctionCallingConfig(mode="AUTO")
        ),
        response_modalities=[types.Modality.TEXT],
    )


async def _call(stream: GeminiStream, config: types.GenerateContentConfig | None = None) -> str:
    """The text of the streamed answer to one request."""
    chunks = await stream(
        model=MODEL,
        contents=[types.Content(role="user", parts=[types.Part(text="question")])],
        config=config or _config(),
    )
    return "".join([chunk.text or "" async for chunk in chunks])


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
    sessions = [_stream(cache, client), _stream(cache, client)]

    for model in [*sessions, *sessions, sessions[0]]:
        await _call(model)

    assert len(client.cache_calls.created) == 1
    created = client.cache_calls.created[0]
    assert created.system_instruction == "You are the planner."
    assert created.tools == [TOOL]
    assert created.ttl == f"{CACHE_TTL_S}s"
    assert [_sent_against(config) for config in client.model_calls.configs] == [
        "cachedContents/c1"
    ] * 5


async def test_a_gone_cache_is_created_again_and_the_request_resent_once() -> None:
    cache = PrefixCache()
    client = _FakeClient(gone={"cachedContents/c1"})
    model = _stream(cache, client)

    texts = await _call(model)
    await _call(model)

    assert texts == "done"
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

    texts = await _call(_stream(PrefixCache(), client))

    assert texts == "done"
    assert [_sent_against(config) for config in client.model_calls.configs] == [
        "cachedContents/c1",
        None,
    ]


async def test_a_cache_gone_after_the_first_chunk_is_not_resent() -> None:
    client = _FakeClient(gone_after_chunk={"cachedContents/c1"})

    with pytest.raises(errors.ClientError) as raised:
        await _call(_stream(PrefixCache(), client))

    assert raised.value.code == 403
    assert len(client.cache_calls.created) == 1
    assert len(client.model_calls.configs) == 1


async def test_a_prefix_the_api_refuses_to_cache_is_sent_whole_from_then_on() -> None:
    too_small = errors.ClientError(
        400, {"error": {"code": 400, "message": "too small", "status": "INVALID_ARGUMENT"}}
    )
    client = _FakeClient(create_failures=[too_small])
    model = _stream(PrefixCache(), client)

    await _call(model)
    await _call(model)

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
    model = _stream(PrefixCache(clock=clock), client)

    await _call(model)
    clock.now += CREATE_COOLDOWN_S - 1
    await _call(model)
    attempts_in_cooldown = client.cache_calls.attempts
    clock.now += 1
    await _call(model)

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
        await _call(_stream(PrefixCache(), client))

    assert raised.value.code == 500
    assert [_sent_against(config) for config in client.model_calls.configs] == [None]


async def test_a_create_that_never_returns_goes_uncached_within_the_bound() -> None:
    client = _FakeClient(release_create=asyncio.Event())
    model = _stream(PrefixCache(create_timeout_s=0.05), client)

    async with asyncio.timeout(1):
        texts = await _call(model)

    assert texts == "done"
    assert client.cache_calls.created == []
    assert [_sent_against(config) for config in client.model_calls.configs] == [None]


async def test_concurrent_calls_on_a_create_that_never_returns_share_one_bound() -> None:
    # Four calls waiting one bound each in turn would take 0.8 s.
    client = _FakeClient(release_create=asyncio.Event())
    cache = PrefixCache(create_timeout_s=0.2)

    async with asyncio.timeout(0.35):
        texts = await asyncio.gather(*(_call(_stream(cache, client)) for _ in range(4)))

    assert texts == ["done"] * 4
    assert client.cache_calls.attempts == 1
    assert [_sent_against(config) for config in client.model_calls.configs] == [None] * 4


async def test_a_cancelled_call_propagates_and_leaves_the_shared_create_running() -> None:
    release = asyncio.Event()
    client = _FakeClient(release_create=release)
    cache = PrefixCache()
    cancelled = asyncio.create_task(_call(_stream(cache, client)))
    waiting = asyncio.create_task(_call(_stream(cache, client)))

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
    calls = asyncio.gather(_call(_stream(cache, client)), _call(_stream(cache, client)))

    await asyncio.sleep(0.01)
    release.set()
    await calls

    assert len(client.cache_calls.created) == 1
    assert [_sent_against(config) for config in client.model_calls.configs] == [
        "cachedContents/c1"
    ] * 2


async def test_a_changed_instruction_gets_its_own_cache() -> None:
    client = _FakeClient()
    model = _stream(PrefixCache(), client)

    await _call(model, _config("You are the planner."))
    await _call(model, _config("You are the orchestrator."))
    await _call(model, _config("You are the planner."))

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
    model = _stream(PrefixCache(clock=clock), client)

    await _call(model)
    clock.now += CACHE_TTL_S - REFRESH_MARGIN_S - 1
    await _call(model)
    clock.now += 2
    await _call(model)

    assert [_sent_against(config) for config in client.model_calls.configs] == [
        "cachedContents/c1",
        "cachedContents/c1",
        "cachedContents/c2",
    ]


async def test_a_request_with_no_instruction_or_tools_goes_uncached() -> None:
    client = _FakeClient()

    found = await PrefixCache().lookup(
        client=client, model=MODEL, config=types.GenerateContentConfig()
    )

    assert found is None
    assert client.cache_calls.attempts == 0


@pytest.mark.parametrize(
    ("model_name", "sent"),
    [
        pytest.param("gemini-3.8-flash", "gemini-3.8-flash", id="gemini"),
        pytest.param("gemma-4-31b-it", "gemma-4-31b-it", id="gemma-4"),
        pytest.param("gemma-3-27b-it", None, id="gemma-3"),
        pytest.param("openai/gpt-4o", None, id="litellm-path"),
        pytest.param("claude-3-5-sonnet", None, id="claude"),
        pytest.param("Gemini:gemini-3.8-flash", "gemini-3.8-flash", id="gemini-prefix"),
        pytest.param("gemini:gemma-3-27b-it", "gemma-3-27b-it", id="gemini-prefix-any-name"),
        pytest.param("openai:gemini-3.8-flash", None, id="other-prefix"),
        pytest.param(
            "projects/p/locations/l/endpoints/e",
            "projects/p/locations/l/endpoints/e",
            id="endpoint",
        ),
        pytest.param(
            "projects/p/locations/l/publishers/google/models/gemini-3.8-flash",
            "projects/p/locations/l/publishers/google/models/gemini-3.8-flash",
            id="vertex-path",
        ),
    ],
)
def test_a_name_gemini_serves_is_sent_as_gemini_names_it(model_name: str, sent: str | None) -> None:
    assert gemini_model_name(model_name) == sent
