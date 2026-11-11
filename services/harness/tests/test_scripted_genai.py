"""The scripted google-genai client drives both model stacks and each stream behaviour."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

import pytest
from google.adk.models import Gemini
from google.adk.models.llm_request import LlmRequest
from google.genai import errors, types
from pydantic_ai import Agent
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.providers.google import GoogleProvider

from .scripted_genai import (
    Reply,
    ScriptedGenai,
    ScriptExhaustedError,
    StreamCutError,
    blocked_reply,
    call_part,
    client_error,
    empty_reply,
    reply,
    text_part,
    usage_metadata,
)

INSTRUCTIONS = "Answer in one word."


def _declared_names(config: Any) -> list[str]:
    return [declaration.name for tool in config.tools for declaration in tool.function_declarations]


async def test_pydantic_ai_streams_text_and_a_function_call_through_the_client() -> None:
    scripted = ScriptedGenai(
        [
            reply(call_part("lookup", {"key": "ghana"}, id="c1"), usage=usage_metadata(prompt=10)),
            reply(text_part("Accra"), usage=usage_metadata(prompt=12, candidates=1)),
        ]
    )
    agent = Agent(
        GoogleModel("gemini-test", provider=GoogleProvider(client=scripted.client)),
        instructions=INSTRUCTIONS,
    )

    @agent.tool_plain
    def lookup(key: str) -> str:
        """Look a key up."""
        return key.upper()

    async with agent.run_stream("capital?") as result:
        output = await result.get_output()

    assert output == "Accra"
    assert len(scripted.requests) == 2
    first = scripted.requests[0]
    assert first.model == "gemini-test"
    # Pydantic AI sends the system instruction as a `Content` and adds its own preamble part.
    assert INSTRUCTIONS in [p.text for p in first.generate_config.system_instruction.parts]
    assert _declared_names(first.generate_config) == ["lookup"]
    returned = scripted.requests[1].content_list[-1].parts[0].function_response
    assert (returned.name, returned.response) == ("lookup", {"return_value": "GHANA"})


async def test_adk_gemini_streams_one_response_through_the_client() -> None:
    scripted = ScriptedGenai([reply(text_part("Accra"), usage=usage_metadata(prompt=5))])
    # `Gemini.client` is a public field, so the scripted client is handed over as one.
    model = Gemini(model="gemini-test", client=scripted.client)
    request = LlmRequest(
        model="gemini-test",
        contents=[types.Content(role="user", parts=[text_part("capital?")])],
        config=types.GenerateContentConfig(system_instruction=INSTRUCTIONS),
    )

    responses = [r async for r in model.generate_content_async(request, stream=True)]

    assert (
        "".join(
            part.text
            for r in responses
            if not r.partial and r.content
            for part in r.content.parts or []
        )
        == "Accra"
    )
    assert scripted.requests[0].generate_config.system_instruction == INSTRUCTIONS


async def _drain(scripted: ScriptedGenai) -> list[types.GenerateContentResponse]:
    stream = await scripted.client.aio.models.generate_content_stream(
        model="m", contents=[], config=None
    )
    return [chunk async for chunk in stream]


async def test_a_reply_split_across_chunks_streams_each_with_usage_on_the_last() -> None:
    scripted = ScriptedGenai(
        [
            Reply(
                chunks=[[text_part("Ac")], [text_part("cra")]],
                usage=usage_metadata(prompt=3, cached=2, candidates=1, thoughts=4),
            )
        ]
    )

    chunks = await _drain(scripted)

    assert [c.candidates[0].content.parts[0].text for c in chunks] == ["Ac", "cra"]
    assert chunks[0].usage_metadata is None
    last = chunks[-1]
    assert last.candidates[0].finish_reason == types.FinishReason.STOP
    assert (
        last.usage_metadata.prompt_token_count,
        last.usage_metadata.cached_content_token_count,
        last.usage_metadata.candidates_token_count,
        last.usage_metadata.thoughts_token_count,
    ) == (3, 2, 1, 4)


@pytest.mark.parametrize(
    ("script", "check"),
    [
        pytest.param(
            Reply(chunks=[[text_part("x")]], open_error=client_error(404)),
            lambda e: isinstance(e, errors.ClientError) and e.code == 404,
            id="open-error",
        ),
        pytest.param(
            Reply(chunks=[[text_part("a")], [text_part("b")]], cut_after=1),
            lambda e: isinstance(e, StreamCutError),
            id="cut-after-a-chunk",
        ),
        pytest.param(
            Reply(
                chunks=[[text_part("a")], [text_part("b")]],
                cut_after=0,
                cut_with=asyncio.CancelledError(),
            ),
            lambda e: isinstance(e, asyncio.CancelledError),
            id="cancelled",
        ),
    ],
)
async def test_a_failing_stream_raises_where_scripted(
    script: Reply, check: Callable[[BaseException], bool]
) -> None:
    scripted = ScriptedGenai([script])

    with pytest.raises((Exception, asyncio.CancelledError)) as raised:
        await _drain(scripted)

    assert check(raised.value)


async def test_a_stalled_stream_waits_before_its_first_chunk() -> None:
    scripted = ScriptedGenai([Reply(chunks=[[text_part("x")]], stall_s=3600)])

    with pytest.raises(TimeoutError):
        await asyncio.wait_for(_drain(scripted), timeout=0.05)


async def test_an_empty_reply_has_no_parts_and_a_blocked_reply_no_candidate() -> None:
    scripted = ScriptedGenai([empty_reply(), blocked_reply(types.BlockedReason.SAFETY)])

    (empty,) = await _drain(scripted)
    (blocked,) = await _drain(scripted)

    assert not empty.candidates[0].content
    assert not blocked.candidates
    assert blocked.prompt_feedback.block_reason == types.BlockedReason.SAFETY


async def test_an_exhausted_script_raises_naming_the_request_count() -> None:
    scripted = ScriptedGenai([reply(text_part("once"))])
    await _drain(scripted)

    with pytest.raises(ScriptExhaustedError, match="2 requests"):
        await _drain(scripted)
