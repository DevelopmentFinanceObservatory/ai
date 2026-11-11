"""Tests for `harness.planner_model`: the Pydantic AI planner's Gemini model.

Each behaviour runs on the same script through the Pydantic AI planner's model
and through the ADK planner's model stack (`BoundedGemini` or
`PrefixCachedGemini` over the scripted client), and both must end the same
way. Bounds are fractions of a second, so a cut fires at once.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterator, Sequence
from contextlib import contextmanager
from typing import Any

import pytest
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.genai import errors, types
from harness.planner_model import ResponseMetadata, planner_model
from harness.prefix_cache import PrefixCache, PrefixCachedGemini
from harness.sentences import SeamCode, refusal_sentence
from harness.turn_budget import BoundedGemini, TurnBudget, current_turn_budget
from orchestrator.agent import _retrying, model_retry_options
from planner.settings import drop_thought_parts, without_unsigned_thoughts
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    UserPromptPart,
)
from pydantic_ai.models import Model, ModelRequestParameters
from pydantic_ai.tools import ToolDefinition

from .scripted_genai import (
    Reply,
    ScriptedGenai,
    call_part,
    client_error,
    reply,
    text_part,
    thought_part,
    usage_metadata,
)

MODEL = "gemini-3.8-flash"
INSTRUCTION = "Plan the user's question."
SCHEMA: dict[str, Any] = {"type": "object", "properties": {"key": {"type": "string"}}}
CACHE_NAME = "cachedContents/c1"


class _HangingGenai(ScriptedGenai):
    """A `ScriptedGenai` whose streams wait for ever after `hang_after` chunks,
    as a model does that stops sending mid-answer."""

    def __init__(self, replies: Sequence[Reply], *, hang_after: int) -> None:
        super().__init__(replies)
        self._hang_after = hang_after

    async def _open(
        self, *, model: str, contents: Any, config: Any = None
    ) -> AsyncIterator[types.GenerateContentResponse]:
        stream = await super()._open(model=model, contents=contents, config=config)
        return _hanging(stream, after=self._hang_after)


async def _hanging(
    stream: AsyncIterator[types.GenerateContentResponse], *, after: int
) -> AsyncIterator[types.GenerateContentResponse]:
    delivered = 0
    async for chunk in stream:
        if delivered == after:
            await asyncio.Event().wait()
        delivered += 1
        yield chunk


class _Caches:
    """`caches.create` answering with `CACHE_NAME`, then failing with each of
    `failures` in turn."""

    def __init__(self, failures: Sequence[Exception] = ()) -> None:
        self.created: list[types.CreateCachedContentConfig] = []
        self._failures = list(failures)

    async def create(self, *, model: str, config: types.CreateCachedContentConfig) -> Any:
        if self.created and self._failures:
            raise self._failures.pop(0)
        self.created.append(config)
        return types.CachedContent(name=f"cachedContents/c{len(self.created)}")


def _with_caches(scripted: ScriptedGenai, caches: _Caches) -> ScriptedGenai:
    scripted.client.aio.caches.create = caches.create  # type: ignore[method-assign]
    return scripted


def _budget(**overrides: Any) -> TurnBudget:
    bounds: dict[str, Any] = {
        "model_time_s": 5.0,
        "model_calls_max": 30,
        "wall_s": 600.0,
        "first_chunk_s": 5.0,
    }
    return TurnBudget(**{**bounds, **overrides})


@contextmanager
def _current(budget: TurnBudget | None) -> Iterator[None]:
    token = current_turn_budget.set(budget)
    try:
        yield
    finally:
        current_turn_budget.reset(token)


async def _pai(model: Model) -> ModelResponse:
    """The planner model's response to one request with one tool."""
    messages = [
        ModelRequest(parts=[UserPromptPart("Total ODA to Ghana?")], instructions=INSTRUCTION)
    ]
    parameters = ModelRequestParameters(
        function_tools=[ToolDefinition(name="lookup", parameters_json_schema=SCHEMA)]
    )
    async with model.request_stream(messages, None, parameters) as stream:
        async for _ in stream:
            pass
    return stream.get()


async def _adk(model: BaseLlm, budget: TurnBudget | None) -> list[LlmResponse]:
    """The ADK planner's model's responses, partial and final, to the same request."""
    request = LlmRequest(
        model=MODEL,
        contents=[types.Content(role="user", parts=[text_part("Total ODA to Ghana?")])],
        config=types.GenerateContentConfig(
            system_instruction=INSTRUCTION,
            tools=[
                types.Tool(
                    function_declarations=[
                        types.FunctionDeclaration(name="lookup", parameters_json_schema=SCHEMA)
                    ]
                )
            ],
        ),
    )
    with _current(budget):
        return [response async for response in model.generate_content_async(request, stream=True)]


def _adk_final(responses: list[LlmResponse]) -> list[types.Part]:
    """The parts of the ADK model's last final response, the one its agent reads."""
    final = [response for response in responses if not response.partial]
    content = final[-1].content
    assert content is not None
    return list(content.parts or [])


def _pai_text(response: ModelResponse) -> str:
    return "".join(part.content for part in response.parts if isinstance(part, TextPart))


def _adk_text(parts: list[types.Part]) -> str:
    return "".join(part.text or "" for part in parts if not part.thought)


class _After:
    """An `after_response` hook that records its calls and answers with `replace`."""

    def __init__(self, replace: types.Content | None = None) -> None:
        self.calls: list[tuple[SeamCode | None, ResponseMetadata]] = []
        self._replace = replace

    def __call__(
        self, closed_by: SeamCode | None, metadata: ResponseMetadata
    ) -> types.Content | None:
        self.calls.append((closed_by, metadata))
        return self._replace


async def test_a_cut_after_a_function_call_chunk_delivers_only_the_closing_sentence() -> None:
    script = [Reply(chunks=[[call_part("lookup", {"key": "gh"})], [text_part("late")]])]
    pai_scripted = _HangingGenai(script, hang_after=1)
    adk_scripted = _HangingGenai(script, hang_after=1)
    pai_budget = _budget(model_time_s=0.05)
    adk_budget = _budget(model_time_s=0.05)
    after = _After()

    response = await _pai(
        planner_model(MODEL, client=pai_scripted.client, budget=pai_budget, after_response=after)
    )
    adk = await _adk(BoundedGemini(model=MODEL, client=adk_scripted.client), adk_budget)

    sentence = refusal_sentence({"code": SeamCode.TURN_MODEL_TIME_SPENT})
    assert not any(isinstance(part, ToolCallPart) for part in response.parts)
    assert not any(
        part.function_call
        for r in adk
        if not r.partial and r.content
        for part in r.content.parts or []
    )
    assert _pai_text(response) == _adk_text(_adk_final(adk)) == sentence
    assert pai_budget.closed == adk_budget.closed
    assert pai_budget.closed is not None
    assert pai_budget.closed["code"] == SeamCode.TURN_MODEL_TIME_SPENT
    assert after.calls == [(SeamCode.TURN_MODEL_TIME_SPENT, ResponseMetadata())]


async def test_a_first_chunk_stall_is_resent_once_and_the_second_answer_delivered() -> None:
    script = [
        Reply(chunks=[[text_part("stalled")]], stall_s=5.0),
        reply(text_part("Accra")),
    ]
    pai_scripted, adk_scripted = ScriptedGenai(script), ScriptedGenai(script)

    response = await _pai(
        planner_model(MODEL, client=pai_scripted.client, budget=_budget(first_chunk_s=0.05))
    )
    adk = await _adk(
        BoundedGemini(model=MODEL, client=adk_scripted.client), _budget(first_chunk_s=0.05)
    )

    assert len(pai_scripted.requests) == len(adk_scripted.requests) == 2
    assert _pai_text(response) == _adk_text(_adk_final(adk)) == "Accra"


@pytest.mark.parametrize(
    ("budget", "code", "sent"),
    [
        pytest.param({"model_calls_max": 0}, SeamCode.TURN_MODEL_CALLS_SPENT, 0, id="call-cap"),
        pytest.param({"wall_s": 0.0}, SeamCode.TURN_WALL_TIME_SPENT, 0, id="wall"),
        pytest.param(
            {"model_time_s": 0.05, "first_chunk_s": 0.01},
            SeamCode.TURN_MODEL_TIME_SPENT,
            2,
            id="model-time",
        ),
    ],
)
async def test_each_bound_closes_the_call_with_the_same_code_on_both_models(
    budget: dict[str, Any], code: SeamCode, sent: int
) -> None:
    # Each stream stalls past every bound; the first stall spends the resend.
    script = [Reply(chunks=[[text_part("x")]], stall_s=5.0)] * 2
    pai_scripted, adk_scripted = ScriptedGenai(script), ScriptedGenai(script)
    pai_budget, adk_budget = _budget(**budget), _budget(**budget)

    response = await _pai(planner_model(MODEL, client=pai_scripted.client, budget=pai_budget))
    adk = await _adk(BoundedGemini(model=MODEL, client=adk_scripted.client), adk_budget)

    assert len(pai_scripted.requests) == len(adk_scripted.requests) == sent
    assert pai_budget.closed == adk_budget.closed == {"code": code, "detail": {}}
    assert _pai_text(response) == _adk_text(_adk_final(adk)) == refusal_sentence({"code": code})


def _cache_fields(config: types.GenerateContentConfig) -> tuple[Any, ...]:
    return (
        config.cached_content,
        config.system_instruction is None,
        config.tools is None,
        config.tool_config is None,
    )


async def test_a_cached_request_names_the_cache_and_omits_what_it_holds() -> None:
    pai_caches, adk_caches = _Caches(), _Caches()
    pai_scripted = _with_caches(ScriptedGenai([reply(text_part("Accra"))]), pai_caches)
    adk_scripted = _with_caches(ScriptedGenai([reply(text_part("Accra"))]), adk_caches)

    await _pai(
        planner_model(
            MODEL, client=pai_scripted.client, budget=_budget(), prefix_cache=PrefixCache()
        )
    )
    await _adk(
        PrefixCachedGemini(model=MODEL, client=adk_scripted.client, prefix_cache=PrefixCache()),
        _budget(),
    )

    pai_config = pai_scripted.requests[0].generate_config
    adk_config = adk_scripted.requests[0].generate_config
    assert _cache_fields(pai_config) == _cache_fields(adk_config) == (CACHE_NAME, True, True, True)
    # The cache holds the instruction and the tool the request leaves out.
    for caches in (pai_caches, adk_caches):
        (created,) = caches.created
        assert created.system_instruction is not None
        assert [
            d.name for tool in created.tools or [] for d in tool.function_declarations or []
        ] == ["lookup"]


@pytest.mark.parametrize("status", [403, 404])
async def test_a_cache_refused_at_open_is_forgotten_and_the_request_resent_whole(
    status: int,
) -> None:
    script = [
        Reply(chunks=[[text_part("x")]], open_error=client_error(status)),
        reply(text_part("Accra")),
    ]
    # The cache cannot be made again, so the resend goes uncached.
    failure = errors.ServerError(503, {"error": {"code": 503, "message": "unavailable"}})
    pai_caches, adk_caches = _Caches([failure]), _Caches([failure])
    pai_scripted = _with_caches(ScriptedGenai(script), pai_caches)
    adk_scripted = _with_caches(ScriptedGenai(script), adk_caches)

    response = await _pai(
        planner_model(
            MODEL, client=pai_scripted.client, budget=_budget(), prefix_cache=PrefixCache()
        )
    )
    adk = await _adk(
        PrefixCachedGemini(model=MODEL, client=adk_scripted.client, prefix_cache=PrefixCache()),
        _budget(),
    )

    for scripted in (pai_scripted, adk_scripted):
        first, resent = (request.generate_config for request in scripted.requests)
        assert first.cached_content == CACHE_NAME
        assert resent.cached_content is None
        assert resent.system_instruction is not None
        assert resent.tools is not None
    # Forgetting the entry is what sent the lookup back to `create`.
    assert pai_caches._failures == adk_caches._failures == []
    assert _pai_text(response) == _adk_text(_adk_final(adk)) == "Accra"


async def test_before_request_answers_with_no_request_and_no_budget_spent() -> None:
    scripted = ScriptedGenai([])
    budget = _budget()
    turn = types.Content(role="model", parts=[text_part("Which source?")])
    after = _After()

    response = await _pai(
        planner_model(
            MODEL,
            client=scripted.client,
            budget=budget,
            before_request=lambda: turn,
            after_response=after,
        )
    )

    assert _pai_text(response) == "Which source?"
    assert scripted.requests == []
    assert (budget.model_calls, budget.closed, after.calls) == (0, None, [])


async def test_after_response_sees_what_the_api_reported_and_its_turn_replaces_the_response() -> (
    None
):
    usage = usage_metadata(prompt=40, cached=30, candidates=5)
    scripted = ScriptedGenai(
        [
            reply(
                call_part("lookup", {"key": "gh"}),
                usage=usage,
                finish_reason=types.FinishReason.MAX_TOKENS,
            )
        ]
    )
    _reporting_version(scripted, "gemini-3.8-flash-001")
    replacement = types.Content(role="model", parts=[text_part("Asked instead.")])
    after = _After(replace=replacement)

    response = await _pai(
        planner_model(MODEL, client=scripted.client, budget=_budget(), after_response=after)
    )

    assert after.calls == [
        (
            None,
            ResponseMetadata(
                usage=usage,
                finish_reason=types.FinishReason.MAX_TOKENS,
                model_version="gemini-3.8-flash-001",
                error_code="MAX_TOKENS",
            ),
        )
    ]
    assert [type(part) for part in response.parts] == [TextPart]
    assert _pai_text(response) == "Asked instead."
    assert response.usage.input_tokens == 40


def _reporting_version(scripted: ScriptedGenai, version: str) -> None:
    """Make every chunk `scripted` streams report `version` as its model version."""
    opened = scripted.client.aio.models.generate_content_stream

    async def open_stream(**request: Any) -> AsyncIterator[types.GenerateContentResponse]:
        stream = await opened(**request)

        async def versioned() -> AsyncIterator[types.GenerateContentResponse]:
            async for chunk in stream:
                yield chunk.model_copy(update={"model_version": version})

        return versioned()

    scripted.client.aio.models.generate_content_stream = open_stream  # type: ignore[method-assign]


async def test_every_request_carries_the_adk_planners_retries() -> None:
    scripted = ScriptedGenai([reply(text_part("Accra"))])

    await _pai(planner_model(MODEL, client=scripted.client, budget=_budget()))

    http_options = scripted.requests[0].generate_config.http_options
    assert http_options is not None
    adk_model = _retrying(MODEL)
    assert isinstance(adk_model, BoundedGemini)
    assert http_options.retry_options == adk_model.retry_options == model_retry_options()


async def test_a_non_streamed_request_refuses() -> None:
    scripted = ScriptedGenai([reply(text_part("Accra"))])
    model = planner_model(MODEL, client=scripted.client, budget=_budget())

    with pytest.raises(TypeError, match="streams every request"):
        await model.request(
            [ModelRequest(parts=[UserPromptPart("hi")])], None, ModelRequestParameters()
        )
    assert scripted.requests == []


_SIGNED = thought_part("weighed the sources", signature=b"sig")
_UNSIGNED = thought_part("thinking aloud", signature=None)
_ANSWER = text_part("Accra")


@pytest.mark.parametrize(
    ("parts", "kept"),
    [
        pytest.param([_SIGNED, _UNSIGNED, _ANSWER], [_SIGNED, _ANSWER], id="signed-and-unsigned"),
        pytest.param([_UNSIGNED, _ANSWER], [_ANSWER], id="unsigned"),
        pytest.param([_UNSIGNED], [_UNSIGNED], id="only-unsigned"),
        pytest.param([_SIGNED, _ANSWER], [_SIGNED, _ANSWER], id="none-to-drop"),
    ],
)
def test_the_thought_rule_and_the_adk_callback_keep_the_same_parts(
    parts: list[types.Part], kept: list[types.Part]
) -> None:
    """Every unsigned thought summary is dropped, a signed thought kept, and a
    response made only of unsigned thoughts kept whole."""
    response = LlmResponse(content=types.Content(role="model", parts=parts))

    dropped = drop_thought_parts(None, response)  # type: ignore[arg-type]
    ruled = without_unsigned_thoughts(parts)

    adk_parts = parts if dropped is None or dropped.content is None else dropped.content.parts
    assert (parts if ruled is None else ruled) == kept
    assert adk_parts == kept


_MORE = thought_part("and more", signature=None)


@pytest.mark.parametrize(
    ("chunks", "thoughts", "text", "calls"),
    [
        pytest.param(
            [[_SIGNED], [_UNSIGNED], [_ANSWER]],
            [("weighed the sourcesthinking aloud", True)],
            "Accra",
            [],
            id="signed-then-unsigned-chunk",
        ),
        pytest.param(
            [[_UNSIGNED], [_SIGNED], [_ANSWER]],
            [("thinking aloudweighed the sources", True)],
            "Accra",
            [],
            id="unsigned-then-signed-chunk",
        ),
        pytest.param([[_UNSIGNED], [_MORE], [_ANSWER]], [], "Accra", [], id="unsigned-run"),
        pytest.param(
            [[_UNSIGNED], [_ANSWER], [_SIGNED]],
            [("weighed the sources", True)],
            "Accra",
            [],
            id="split-by-text",
        ),
        pytest.param(
            [[_UNSIGNED], [call_part("lookup", {"key": "gh"}, signature=b"call")]],
            [],
            "",
            ["lookup"],
            id="before-a-call",
        ),
        pytest.param(
            [[_UNSIGNED], [_MORE]], [("thinking aloudand more", False)], "", [], id="only-unsigned"
        ),
    ],
)
async def test_a_response_reaches_the_agent_with_the_thoughts_the_adk_planner_keeps(
    chunks: list[list[types.Part]],
    thoughts: list[tuple[str, bool]],
    text: str,
    calls: list[str],
) -> None:
    """Consecutive thought chunks merge into one part, signed when any chunk of
    them is; the rule then drops each unsigned one unless nothing else is left.
    Both planners keep `thoughts` (each part's text and whether it is signed),
    `text` and `calls`."""
    script = [Reply(chunks=chunks)]
    pai_scripted, adk_scripted = ScriptedGenai(script), ScriptedGenai(script)

    response = await _pai(planner_model(MODEL, client=pai_scripted.client, budget=_budget()))
    adk = await _adk(BoundedGemini(model=MODEL, client=adk_scripted.client), _budget())
    final = adk[-1]
    assert not final.partial
    kept = drop_thought_parts(None, final) or final  # type: ignore[arg-type]
    assert kept.content is not None
    adk_parts = list(kept.content.parts or [])

    assert [
        (part.content, bool(part.provider_details))
        for part in response.parts
        if isinstance(part, ThinkingPart)
    ] == thoughts
    assert [(part.text, bool(part.thought_signature)) for part in adk_parts if part.thought] == (
        thoughts
    )
    assert _pai_text(response) == _adk_text(adk_parts) == text
    assert [part.tool_name for part in response.parts if isinstance(part, ToolCallPart)] == calls
    assert [part.function_call.name for part in adk_parts if part.function_call] == calls
