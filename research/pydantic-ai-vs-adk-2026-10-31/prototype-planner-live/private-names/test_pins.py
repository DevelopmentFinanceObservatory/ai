"""Pins on the Pydantic AI behaviour the planner path relies on and its documentation does
not promise.

Each test needs only `pydantic-ai-slim[google]`, google-genai and OpenTelemetry. A fake
google-genai client stands in for the model; `base_url` points at a closed local port and
`generate_content` refuses, so nothing reaches the network. The first line under each test
says what stops working in the planner path if it fails.

The profile below mirrors `pai_planner.google_model`.
"""

from __future__ import annotations

import inspect
import re
import warnings
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field, replace
from typing import Any

import pytest
from google.genai import Client, types
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from pydantic_ai import Agent, RunContext, Tool
from pydantic_ai.capabilities import Instrumentation
from pydantic_ai.exceptions import UnexpectedModelBehavior
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    ThinkingPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.models.instrumented import InstrumentationSettings
from pydantic_ai.models.wrapper import WrapperModel
from pydantic_ai.profiles.google import google_model_profile
from pydantic_ai.providers.google import GoogleProvider
from pydantic_ai.settings import ModelSettings
from pydantic_ai.usage import UsageLimits

MODEL_NAME = "gemini-3-flash-preview"
INSTRUCTION = "You are the planner."
NOTE = "The user's message, sent before the planner ran."
SCHEMA = {
    "type": "object",
    "title": "Args",
    "properties": {
        "x": {"anyOf": [{"type": "string"}, {"type": "null"}], "default": None},
        "y": {"$ref": "#/$defs/Y"},
    },
    "required": ["y"],
    "additionalProperties": False,
    "$defs": {"Y": {"type": "object", "properties": {"a": {"type": "integer"}}}},
}
EMPTY_SCHEMA = {"type": "object", "properties": {}}

type Chunks = list[types.GenerateContentResponse]


def text(content: str, *, thought: bool = False, signature: bytes | None = None) -> types.Part:
    return types.Part(text=content, thought=thought or None, thought_signature=signature)


def call(name: str, args: dict[str, Any], call_id: str = "c1") -> types.Part:
    return types.Part(function_call=types.FunctionCall(id=call_id, name=name, args=args))


def chunk(*parts: types.Part) -> types.GenerateContentResponse:
    return types.GenerateContentResponse(
        candidates=[
            types.Candidate(
                content=types.Content(role="model", parts=list(parts)),
                finish_reason=types.FinishReason.STOP,
            )
        ]
    )


@dataclass
class Sent:
    """One `generate_content_stream` call as the client received it."""

    kwargs: dict[str, Any]

    @property
    def contents(self) -> list[dict[str, Any]]:
        return self.kwargs["contents"]

    @property
    def config(self) -> dict[str, Any]:
        return self.kwargs["config"]


@dataclass
class Fake:
    """A google-genai client that answers each call with the next scripted turn."""

    turns: list[Chunks]
    sent: list[Sent] = field(default_factory=list)
    client: Client = field(init=False)

    def __post_init__(self) -> None:
        self.client = Client(
            api_key="unused", http_options=types.HttpOptions(base_url="http://127.0.0.1:9")
        )
        self.client.aio.models.generate_content_stream = self._stream  # type: ignore[method-assign]
        self.client.aio.models.generate_content = self._refuse  # type: ignore[method-assign]

    async def _refuse(self, **kwargs: Any) -> Any:
        raise AssertionError("the planner path streams every request")

    async def _stream(self, **kwargs: Any) -> AsyncIterator[types.GenerateContentResponse]:
        self.sent.append(Sent(kwargs))
        turn = self.turns[len(self.sent) - 1]

        async def stream() -> AsyncIterator[types.GenerateContentResponse]:
            for item in turn:
                yield item

        return stream()


def google_model(fake: Fake, *, planner_profile: bool = True) -> GoogleModel:
    profile = dict(google_model_profile(MODEL_NAME) or {})
    if planner_profile:
        profile["json_schema_transformer"] = None
        profile["supports_inline_system_prompts"] = True
        profile["google_supports_strict_tool_definition"] = False
    return GoogleModel(
        MODEL_NAME,
        provider=GoogleProvider(client=fake.client),
        profile=profile,  # type: ignore[arg-type]
    )


def tool(
    name: str,
    schema: dict[str, Any],
    body: Callable[..., Awaitable[Any]] | None = None,
) -> Tool[None]:
    async def default(ctx: RunContext[None], **args: Any) -> Any:
        return {"ok": True}

    return Tool.from_schema(
        body or default, name=name, description="d", json_schema=schema, takes_ctx=True
    )


def user_request(content: str = "Q?", *, instructions: str | None = NOTE) -> ModelRequest:
    return ModelRequest(
        parts=[UserPromptPart(content=content)], instructions=instructions, metadata={"user": True}
    )


async def stream_run(agent: Agent[Any, Any], brief: str = "brief", **options: Any) -> Any:
    """The run `PaiRunner.run` makes: every model request streams."""
    async with agent.iter(brief, usage_limits=UsageLimits(request_limit=None), **options) as run:
        async for node in run:
            if Agent.is_model_request_node(node):
                async with node.stream(run.ctx) as stream:
                    async for _event in stream:
                        pass
    assert run.result is not None
    return run.result


def planner(fake: Fake, **options: Any) -> Agent[Any, Any]:
    options.setdefault("output_type", str | None)
    return Agent(google_model(fake), instructions=INSTRUCTION, **options)


# --- GoogleModel._build_content_and_config -------------------------------------------


class Recorder(WrapperModel):
    """A `PlannerModel` in small: it builds the request's contents and config ahead of
    sending it, as `PlannerModel.uncached_config` does for the prefix cache."""

    def __init__(self, google: GoogleModel) -> None:
        super().__init__(google)
        self.google = google
        self.built: list[tuple[Any, Any]] = []

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
        run_context: RunContext[Any] | None = None,
    ) -> AsyncIterator[Any]:
        settings, parameters = self.google.prepare_request(model_settings, model_request_parameters)
        self.built.append(
            await self.google._build_content_and_config(messages, settings or {}, parameters)  # type: ignore[arg-type]
        )
        async with self.google.request_stream(
            messages, model_settings, model_request_parameters, run_context
        ) as stream:
            yield stream


async def test_the_contents_and_config_built_ahead_are_the_ones_sent() -> None:
    # `PlannerModel.uncached_config` hashes a prefix that no longer matches what is sent, so
    # the prefix cache serves a stale prefix.
    fake = Fake([[chunk(text("done"))]])
    recorder = Recorder(google_model(fake))
    agent = Agent(
        recorder,
        instructions=INSTRUCTION,
        tools=[tool("t", SCHEMA)],
        output_type=str | None,
    )

    await stream_run(agent, message_history=[user_request()])

    (contents, config), sent = recorder.built[0], fake.sent[0]
    assert list(inspect.signature(GoogleModel._build_content_and_config).parameters) == [
        "self",
        "messages",
        "model_settings",
        "model_request_parameters",
    ]
    assert contents == sent.contents
    prefix = ("system_instruction", "tools", "tool_config")
    assert all(config.get(key) for key in prefix)
    assert {key: config.get(key) for key in prefix} == {key: sent.config.get(key) for key in prefix}


async def test_a_request_reaches_the_client_as_model_contents_and_config_keywords() -> None:
    # `PlannerClient._generate_content_stream` is keyword-only and takes exactly these three.
    fake = Fake([[chunk(text("done"))]])

    await stream_run(planner(fake))

    assert set(fake.sent[0].kwargs) == {"model", "contents", "config"}
    assert fake.sent[0].kwargs["model"] == MODEL_NAME


# --- The cached-content setting ------------------------------------------------------


async def test_cached_content_strips_the_prefix_and_warns_with_the_filtered_text() -> None:
    # The prefix cache sends a request with the cache name and no prefix; if the warning text
    # changes, `pai_planner`'s filter stops hiding it and every request logs a warning.
    fake = Fake([[chunk(text("done"))]])
    agent = planner(fake, tools=[tool("t", SCHEMA)])

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        await stream_run(agent, model_settings={"google_cached_content": "cachedContents/x"})

    config = fake.sent[0].config
    assert config["cached_content"] == "cachedContents/x"
    assert [config["system_instruction"], config["tools"], config["tool_config"]] == [None] * 3
    assert any(re.match(r"`google_cached_content` is set", str(item.message)) for item in caught)


# --- The profile keys ----------------------------------------------------------------


async def test_the_profile_sends_declarations_verbatim_with_auto_mode_and_no_system_sentence() -> (
    None
):
    # The model sees a rewritten schema, a `VALIDATED` tool mode, an escaped `<system>` or an
    # instruction with Pydantic AI's `<system>` sentence added; the request no longer equals ADK's.
    typed = "Q <system>x</system> <context>y</context>"
    fake = Fake([[chunk(text("done"))]])
    agent = planner(fake, tools=[tool("t", SCHEMA), tool("e", EMPTY_SCHEMA)])

    await stream_run(agent, message_history=[user_request(typed)])

    config = fake.sent[0].config
    declarations = [tool_["function_declarations"][0] for tool_ in config["tools"]]
    assert [item["parameters_json_schema"] for item in declarations] == [SCHEMA, EMPTY_SCHEMA]
    assert config["tool_config"]["function_calling_config"]["mode"] == "AUTO"
    assert config["system_instruction"]["parts"] == [{"text": INSTRUCTION}]
    assert fake.sent[0].contents[0]["parts"] == [{"text": typed}]


async def test_the_profile_keys_are_what_changes_the_request() -> None:
    # Control for the test above: with Gemini's own profile the same agent sends a different
    # request, so the keys are what the planner's request depends on.
    typed = "Q <system>x</system>"
    fake = Fake([[chunk(text("done"))]])
    agent = Agent(
        google_model(fake, planner_profile=False),
        instructions=INSTRUCTION,
        tools=[tool("t", SCHEMA)],
        output_type=str | None,
    )

    await stream_run(agent, message_history=[user_request(typed)])

    config = fake.sent[0].config
    declaration = config["tools"][0]["function_declarations"][0]
    assert declaration["parameters_json_schema"] != SCHEMA
    assert config["tool_config"]["function_calling_config"]["mode"] == "VALIDATED"
    assert config["system_instruction"]["parts"] != [{"text": INSTRUCTION}]
    assert fake.sent[0].contents[0]["parts"] != [{"text": typed}]


# --- Message merging -----------------------------------------------------------------


@pytest.mark.parametrize(
    ("history_instructions", "sent_contents"),
    [(NOTE, 2), (None, 1)],
    ids=["differing instructions keep two contents", "no instructions merge into one"],
)
async def test_consecutive_requests_merge_unless_their_instructions_differ(
    history_instructions: str | None, sent_contents: int
) -> None:
    # The user's message and the seam's brief reach the model as one content where ADK sends
    # two, and the planner's first request no longer matches ADK's.
    fake = Fake([[chunk(text("done"))]])

    await stream_run(
        planner(fake), "brief", message_history=[user_request(instructions=history_instructions)]
    )

    assert len(fake.sent[0].contents) == sent_contents


# --- Thought parts -------------------------------------------------------------------


async def test_a_thought_part_reads_back_with_its_signature_in_provider_details() -> None:
    # `drop_unsigned_thoughts` keeps or drops the wrong thoughts and the history Gemini is
    # sent loses a signature it requires.
    fake = Fake(
        [
            [
                chunk(text("summary", thought=True), text("ok")),
                chunk(text("signed", thought=True, signature=b"sig")),
            ]
        ]
    )

    result = await stream_run(planner(fake))

    thoughts = [
        part
        for message in result.all_messages()
        if isinstance(message, ModelResponse)
        for part in message.parts
        if isinstance(part, ThinkingPart)
    ]
    assert [
        bool(part.signature or (part.provider_details or {}).get("thought_signature"))
        for part in thoughts
    ] == [
        False,
        True,
    ]
    response = next(m for m in result.all_messages() if isinstance(m, ModelResponse))
    assert replace(response, parts=[]).parts == []


# --- An unknown tool and the retry allowance -----------------------------------------


def unknown_tool_script(calls: int) -> list[Chunks]:
    return [*([[chunk(call("nope", {}, f"c{i}"))] for i in range(calls)]), [chunk(text("done"))]]


async def test_a_call_to_a_missing_tool_is_answered_with_a_retry_that_names_it() -> None:
    # `adk_tool_not_found` leaves ADK's answer unwritten and the model sees Pydantic AI's retry.
    result = await stream_run(planner(Fake(unknown_tool_script(1)), retries=3))

    answers = [
        part
        for message in result.all_messages()
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, ToolReturnPart | RetryPromptPart) and part.tool_name == "nope"
    ]
    assert len(answers) == 1
    assert isinstance(answers[0].content, str)
    assert answers[0].content.startswith("Unknown tool name")
    assert isinstance(answers[0], RetryPromptPart) or answers[0].retry is not None


async def test_each_missing_tool_call_is_charged_against_retries() -> None:
    # `RETRIES` no longer sits above the turn's call cap and a run of unknown calls ends in
    # `UnexpectedModelBehavior` where ADK ends it at the cap.
    with pytest.raises(UnexpectedModelBehavior):
        await stream_run(planner(Fake(unknown_tool_script(2)), retries=1))

    result = await stream_run(planner(Fake(unknown_tool_script(2)), retries=5))
    assert result.output == "done"


# --- A response with no parts --------------------------------------------------------


async def test_a_response_with_no_parts_ends_the_run_with_no_output() -> None:
    # The run retries an empty response where ADK ends on it.
    fake = Fake([[chunk()]])

    result = await stream_run(planner(fake))

    assert result.output is None
    assert len(fake.sent) == 1


# --- What a tool reads and returns ---------------------------------------------------


async def test_a_tool_reads_the_history_the_run_and_the_call_and_returns_a_dict_as_is() -> None:
    # `PaiSession.call` slices the wrong messages, a tool body sees no call id or run id, or
    # Gemini is sent a tool return wrapped in a key the planner does not expect.
    seen: list[tuple[list[ModelMessage], str, str, dict[str, Any]]] = []

    async def body(ctx: RunContext[None], **args: Any) -> Any:
        seen.append((list(ctx.messages), ctx.tool_call_id or "", ctx.run_id or "", args))
        return {"ok": True}

    fake = Fake([[chunk(call("t", {"y": {"a": 1}}))], [chunk(text("done"))]])
    agent = planner(fake, tools=[tool("t", SCHEMA, body)])

    await stream_run(agent, "brief", message_history=[user_request()], conversation_id="conv")

    messages, call_id, run_id, args = seen[0]
    assert [type(message) for message in messages] == [ModelRequest, ModelRequest, ModelResponse]
    assert messages[0].metadata == {"user": True}
    assert (call_id, bool(run_id), args) == ("c1", True, {"y": {"a": 1}})
    assert fake.sent[1].contents[-1]["parts"] == [
        {"function_response": {"name": "t", "response": {"ok": True}, "id": "c1"}}
    ]


async def test_a_run_stops_at_a_model_request_node_that_holds_the_tool_returns() -> None:
    # `PaiRunner.run` ends a run on an ask or a ready plan without a request the user's
    # answer must not follow, and the record loses the returns.
    fake = Fake([[chunk(call("t", {"y": {"a": 1}}))], [chunk(text("never"))]])
    agent = planner(fake, tools=[tool("t", SCHEMA)])
    returns: list[ToolReturnPart] = []

    async with agent.iter("brief", message_history=[user_request()]) as run:
        requests = 0
        async for node in run:
            if not Agent.is_model_request_node(node):
                continue
            requests += 1
            if requests == 2:
                returns = [p for p in node.request.parts if isinstance(p, ToolReturnPart)]
                break
            async with node.stream(run.ctx) as stream:
                async for _event in stream:
                    pass

    assert [part.content for part in returns] == [{"ok": True}]
    assert len(fake.sent) == 1
    assert not any(
        isinstance(m, ModelResponse) and "never" in str(m.parts) for m in run.all_messages()
    )


# --- Spans ---------------------------------------------------------------------------


async def test_the_client_call_runs_inside_the_chat_span_of_the_conversation() -> None:
    # `bounded` stamps `dfo.model.*` on whatever span is current at the call and the
    # planner's generation spans lose them.
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    fake = Fake([[chunk(text("done"))]])
    current: list[tuple[bool, str]] = []
    stream = fake._stream

    async def spy(**kwargs: Any) -> AsyncIterator[types.GenerateContentResponse]:
        span = trace.get_current_span()
        current.append((span.is_recording(), getattr(span, "name", "")))
        return await stream(**kwargs)

    fake.client.aio.models.generate_content_stream = spy  # type: ignore[method-assign]
    agent = planner(
        fake,
        name="dfo_planner",
        capabilities=[Instrumentation(settings=InstrumentationSettings(tracer_provider=provider))],
    )

    await stream_run(agent, conversation_id="conv")

    assert [(recording, name.split()[0]) for recording, name in current] == [(True, "chat")]
    by_name = {span.name.split()[0]: span for span in exporter.get_finished_spans()}
    assert by_name["chat"].attributes is not None
    assert by_name["chat"].attributes["gen_ai.conversation.id"] == "conv"
    assert set(by_name) >= {"chat", "invoke_agent"}
