"""The served planner, rebuilt as a Pydantic AI agent on `GoogleModel`.

Everything the planner is shown and every rule the seam applies come from production
objects, so the only thing this module changes is the agent loop:

- `PlannerSeam` itself runs the entry. Its `__call__` seeds state and renders the brief,
  its `_propose` sets the per-call tracker, and its callbacks
  (`_before_planner_model`, `_after_planner_model`, `_after_planner_tool`,
  `_on_planner_tool_error`) are called here at the points ADK calls them. Only
  `ctx.run_node`, the ADK call that runs the planner, is replaced (`PaiNodeContext`).
- The instruction and every tool declaration are read off the ADK agent the seam
  builds (`PlannerParts`), and each tool call runs the ADK tool object's own
  `run_async`, argument checks included.
- The four event-history readers `submission.service.SessionView` needs run unchanged
  over ADK `Event`s rebuilt from the Pydantic AI message history (`adk_events`).
- Each model call goes through `harness.turn_budget.bounded` at the google-genai client,
  as `BoundedGemini` does, and reads its prefix from `harness.prefix_cache.PrefixCache`
  through `GoogleModelSettings.google_cached_content` (`PlannerModel`).
- Thought summaries are dropped from history as `planner.settings.drop_thought_parts`
  drops them (`drop_unsigned_thoughts`).

Private names used here: `PlannerSeam._before_planner_model`, `_after_planner_model`,
`_after_planner_tool`, `_on_planner_tool_error`, `_agent`; ADK `FunctionTool._get_declaration`;
Pydantic AI `GoogleModel._build_content_and_config`.
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import AsyncIterator, Callable, Sequence
from contextlib import AsyncExitStack, aclosing, asynccontextmanager
from dataclasses import dataclass, field, replace
from types import SimpleNamespace
from typing import Any, Literal

from google.adk import Event
from google.adk.events import EventActions
from google.adk.models.llm_response import LlmResponse
from google.adk.tools.base_tool import BaseTool
from google.genai import Client, types
from harness.planner_seam import PlannerSeam
from harness.prefix_cache import PrefixCache, prefix_key
from harness.turn_budget import CLOSED_BY_KEY, bounded
from orchestrator.agent import _retrying
from planner.agent import AGENT_NAME
from planner.settings import AGENT_THINKING_CONFIG
from pydantic_ai import Agent, CallDeferred, RunContext, Tool
from pydantic_ai.capabilities.process_history import ProcessHistory
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.models.google import GoogleModel, GoogleModelSettings
from pydantic_ai.models.wrapper import WrapperModel
from pydantic_ai.profiles.google import google_model_profile
from pydantic_ai.providers.google import GoogleProvider
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import DeferredToolRequests

logger = logging.getLogger(__name__)

Mode = Literal["like-adk", "validated"]

# The branch `PlannerSeam` runs the planner on (`harness.planner_seam._PLANNER_BRANCH`).
PLANNER_BRANCH = "dfo_planner"

# Gemini answers a request whose `cached_content` is gone with 403 (404 on Vertex);
# `harness.prefix_cache._CACHE_GONE_CODES`.
_CACHE_GONE_CODES = frozenset({403, 404})

# What `custom_metadata` carries the raw google-genai chunk under while `bounded` holds it.
_RAW_CHUNK = "pai.raw_chunk"

# The thinking config ADK sends (`planner.settings.AGENT_THINKING_CONFIG`).
THINKING_SETTINGS = GoogleModelSettings(
    google_thinking_config=AGENT_THINKING_CONFIG.model_dump(mode="json", exclude_none=True)
)

# `GoogleModelSettings.google_cached_content` warns each time it strips the prefix
# from a request; stripping it is the point here.
warnings.filterwarnings("ignore", message=r"`google_cached_content` is set")


# --- What the ADK planner is built from --------------------------------------------


@dataclass(frozen=True)
class PlannerParts:
    """The instruction, the ADK tool objects and their declarations, read off the seam's
    ADK agent, in the order ADK declares them."""

    instruction: str
    tools: tuple[BaseTool, ...]
    declarations: tuple[types.FunctionDeclaration, ...]

    @classmethod
    def of(cls, seam: PlannerSeam) -> PlannerParts:
        agent = seam._agent
        instruction = agent.instruction
        assert isinstance(instruction, str), "the planner's instruction is static text"
        tools = tuple(tool for tool in agent.tools if isinstance(tool, BaseTool))
        declarations = []
        for tool in tools:
            declaration = tool._get_declaration()
            assert declaration is not None, f"{tool.name} declares nothing"
            declarations.append(declaration)
        return cls(instruction=instruction, tools=tools, declarations=tuple(declarations))


# The JSON schema Pydantic AI gets for a declaration ADK sends with no parameters
# (`reject_plan`). Pydantic AI always sends `parameters_json_schema`, so this one is sent.
EMPTY_PARAMETERS: dict[str, Any] = {"type": "object", "properties": {}}


# --- The session the readers and the record see -------------------------------------


def adk_events(messages: Sequence[ModelMessage]) -> list[Event]:
    """The ADK session events a run with `messages` leaves, as `PlannerSeam` runs it.

    The first user prompt is the user's own message: authored `user`, no branch. Every
    later user prompt is a message the seam sent, on the planner's branch. A model
    response is the planner's event, and so are the tool returns that follow it, the
    user's answer to a deferred ask included: the readers and the record read a
    function response's id, name and body, never its author.
    """
    events: list[Event] = []
    seen_user_text = False
    for message in messages:
        if isinstance(message, ModelResponse):
            parts = []
            for part in message.parts:
                if isinstance(part, ToolCallPart):
                    parts.append(
                        types.Part(
                            function_call=types.FunctionCall(
                                id=part.tool_call_id, name=part.tool_name, args=part.args_as_dict()
                            )
                        )
                    )
                elif isinstance(part, TextPart):
                    parts.append(types.Part(text=part.content))
            if parts:
                events.append(
                    Event(
                        author=AGENT_NAME,
                        branch=PLANNER_BRANCH,
                        content=types.Content(role="model", parts=parts),
                    )
                )
            continue
        assert isinstance(message, ModelRequest)
        responses: list[types.Part] = []
        for part in message.parts:
            if isinstance(part, UserPromptPart):
                text = part.content if isinstance(part.content, str) else str(part.content)
                events.append(
                    Event(
                        author="user",
                        branch=PLANNER_BRANCH if seen_user_text else None,
                        content=types.Content(role="user", parts=[types.Part(text=text)]),
                    )
                )
                seen_user_text = True
            elif isinstance(part, ToolReturnPart):
                body = part.content if isinstance(part.content, dict) else {"result": part.content}
                responses.append(
                    types.Part(
                        function_response=types.FunctionResponse(
                            id=part.tool_call_id, name=part.tool_name, response=body
                        )
                    )
                )
        if responses:
            events.append(
                Event(
                    author=AGENT_NAME,
                    branch=PLANNER_BRANCH,
                    content=types.Content(role="user", parts=responses),
                )
            )
    return events


def drop_unsigned_thoughts(messages: list[ModelMessage]) -> list[ModelMessage]:
    """`messages` with every thinking part that carries no thought signature removed,
    as `planner.settings.drop_thought_parts` removes a thought part with none from
    ADK's session history. A thinking part with a signature stays whole."""
    kept: list[ModelMessage] = []
    for message in messages:
        if isinstance(message, ModelResponse):
            parts = [part for part in message.parts if not _unsigned_thought(part)]
            if len(parts) != len(message.parts):
                kept.append(replace(message, parts=parts))
                continue
        kept.append(message)
    return kept


def _unsigned_thought(part: object) -> bool:
    if not isinstance(part, ThinkingPart):
        return False
    details = part.provider_details or {}
    return not (part.signature or details.get("thought_signature"))


# --- The tool context ADK tools and the seam's callbacks read ------------------------


@dataclass
class PlannerDeps:
    """One entry's run state: the session state ADK's planner reads and writes, and
    whether a tool asked ADK to end the run without another model call."""

    state: dict[str, Any]
    session_id: str
    skip_summarization: bool = False


def tool_context(ctx: RunContext[PlannerDeps]) -> SimpleNamespace:
    """The `ToolContext` an ADK tool and the seam's tool callbacks read for this call:
    the session's state, its events rebuilt from the run's messages, the call's id and
    a fresh `EventActions`."""
    state = ctx.deps.state
    return SimpleNamespace(
        state=state,
        session=SimpleNamespace(
            id=ctx.deps.session_id, state=state, events=adk_events(ctx.messages)
        ),
        function_call_id=ctx.tool_call_id,
        invocation_id=ctx.run_id,
        actions=EventActions(),
        tool_confirmation=None,
    )


def planner_tools(seam: PlannerSeam, parts: PlannerParts) -> list[Tool[PlannerDeps]]:
    """One Pydantic AI tool per ADK tool, declared as ADK declares it, running the ADK
    tool object and the seam's tool callbacks as ADK's tool caller does
    (`google.adk.flows.llm_flows.tools._caller`)."""
    return [
        Tool.from_schema(
            _tool_body(seam, tool),
            name=declaration.name or tool.name,
            description=declaration.description,
            json_schema=(
                dict(declaration.parameters_json_schema)
                if declaration.parameters_json_schema is not None
                else EMPTY_PARAMETERS
            ),
            takes_ctx=True,
        )
        for tool, declaration in zip(parts.tools, parts.declarations, strict=True)
    ]


def _tool_body(seam: PlannerSeam, tool: BaseTool) -> Callable[..., Any]:
    async def call(ctx: RunContext[PlannerDeps], **args: Any) -> dict[str, Any]:
        context = tool_context(ctx)
        try:
            result = await tool.run_async(args=args, tool_context=context)  # type: ignore[arg-type]
        except Exception as exc:
            result = seam._on_planner_tool_error(tool, args, context, exc)
            if result is None:
                raise
        seam._after_planner_tool(tool, args, context, result)
        if context.actions.skip_summarization:
            ctx.deps.skip_summarization = True
        if tool.is_long_running and not result:
            raise CallDeferred
        return result if isinstance(result, dict) else {"result": result}

    call.__name__ = tool.name
    return call


# --- The model: before-model, prefix cache, budget, after-model ----------------------


@dataclass
class CallHooks:
    """What runs around each of the planner's model calls, at the points ADK runs it.

    `before` is the probe's before-model callback (`evals/planner_entry/entry.Probe`),
    which ADK runs after the seam's own; `after` is the probe's after-model callback,
    which ADK runs before the seam's own.
    """

    seam: PlannerSeam
    before: Callable[[Any, Any], Any] = lambda _ctx, _request: None
    after: Callable[[Any, LlmResponse], Any] = lambda _ctx, _response: None


@dataclass
class _Pending:
    """The model turn the next client call answers with instead of the model."""

    response: LlmResponse | None = None


@dataclass
class PlannerClient:
    """The google-genai client the planner's model calls through, its
    `generate_content_stream` replaced so each call keeps to the current `TurnBudget`
    (`harness.turn_budget.bounded`, as `BoundedGemini` applies it) and runs the
    after-model callbacks on its answer.

    A call the budget closes answers with the closing sentence, or with the ask the
    seam's after-model callback puts in its place. A call `PlannerModel` answers for
    the seam (a synthetic response) reaches neither the model nor the budget.
    """

    client: Client
    hooks: CallHooks
    pending: _Pending = field(default_factory=_Pending)
    sent: Callable[..., Any] = field(init=False)

    def __post_init__(self) -> None:
        self.sent = self.client.aio.models.generate_content_stream
        self.client.aio.models.generate_content_stream = self._generate_content_stream  # type: ignore[method-assign]

    async def _generate_content_stream(
        self, *, model: str, contents: Any, config: Any
    ) -> AsyncIterator[types.GenerateContentResponse]:
        return self._stream(model=model, contents=contents, config=config)

    async def _stream(
        self, *, model: str, contents: Any, config: Any
    ) -> AsyncIterator[types.GenerateContentResponse]:
        synthetic, self.pending.response = self.pending.response, None
        if synthetic is not None:
            yield _as_chunk(synthetic)
            return
        usage: types.GenerateContentResponseUsageMetadata | None = None
        closed = False
        responses = bounded(
            lambda: self._raw(model=model, contents=contents, config=config), stream=True
        )
        async with aclosing(responses) as stream:
            async for response in stream:
                metadata = response.custom_metadata or {}
                if _RAW_CHUNK in metadata:
                    chunk: types.GenerateContentResponse = metadata[_RAW_CHUNK]
                    usage = chunk.usage_metadata or usage
                    yield chunk
                elif CLOSED_BY_KEY in metadata:
                    # The budget closed the call: ADK's session keeps the final answer
                    # alone, after the after-model callbacks have seen it.
                    closed = True
                    self.hooks.after(None, response)
                    replaced = self.hooks.seam._after_planner_model(None, response)
                    yield _as_chunk(replaced or response)
                # The partial copy of a closing sentence is Studio's, never history's.
        if not closed:
            whole = LlmResponse(usage_metadata=usage, partial=False)
            self.hooks.after(None, whole)
            self.hooks.seam._after_planner_model(None, whole)

    async def _raw(self, *, model: str, contents: Any, config: Any) -> AsyncIterator[LlmResponse]:
        stream = await self.sent(model=model, contents=contents, config=config)
        async with aclosing(stream) as chunks:
            async for chunk in chunks:
                content = chunk.candidates[0].content if chunk.candidates else None
                yield LlmResponse(
                    content=content, partial=True, custom_metadata={_RAW_CHUNK: chunk}
                )


def _as_chunk(response: LlmResponse) -> types.GenerateContentResponse:
    """A model turn as the one streamed chunk google-genai would deliver for it."""
    return types.GenerateContentResponse(
        candidates=[
            types.Candidate(
                content=response.content or types.Content(role="model", parts=[]),
                finish_reason=types.FinishReason.STOP,
            )
        ],
        usage_metadata=response.usage_metadata,
    )


class PlannerModel(WrapperModel):
    """`GoogleModel` with the seam's before-model callback and the shared prefix cache.

    Each request first asks the seam whether to answer it itself
    (`PlannerSeam._before_planner_model`); if so the client answers with that turn and
    the model is not called. Otherwise the probe's before-model callback runs, the
    request's prefix is looked up in `prefix_cache` and the request is sent against it
    (`google_cached_content`). A cache the API reports gone (403 or 404) before any
    chunk arrived is forgotten, looked up again and the request resent once, as
    `harness.prefix_cache.PrefixCachedGemini` does. In `like-adk` mode the cached
    prefix carries ADK's tool config (none); in `validated` mode it carries the one
    Pydantic AI sends uncached (`VALIDATED`).
    """

    def __init__(
        self,
        google: GoogleModel,
        *,
        client: PlannerClient,
        prefix_cache: PrefixCache | None,
        mode: Mode,
    ) -> None:
        super().__init__(google)
        self.google = google
        self.planner_client = client
        self.prefix_cache = prefix_cache
        self.mode = mode
        self.cache_keys: list[str] = []

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
        run_context: RunContext[Any] | None = None,
    ) -> AsyncIterator[Any]:
        hooks = self.planner_client.hooks
        synthetic = hooks.seam._before_planner_model(None, None)
        settings: dict[str, Any] = dict(model_settings or {})
        cached: tuple[str, str] | None = None
        if synthetic is not None:
            self.planner_client.pending.response = synthetic
        else:
            # ADK hands its before-model callbacks a context whose `state` is the session's.
            hooks.before(
                SimpleNamespace(state=run_context.deps.state) if run_context else None, None
            )
            cached = await self._cached(messages, settings, model_request_parameters)
        async with AsyncExitStack() as stack:
            for resend in (False, True):
                if cached is not None:
                    settings["google_cached_content"] = cached[1]
                else:
                    settings.pop("google_cached_content", None)
                try:
                    stream = await stack.enter_async_context(
                        self.google.request_stream(
                            messages, settings, model_request_parameters, run_context
                        )
                    )
                except ModelHTTPError as exc:
                    if resend or cached is None or exc.status_code not in _CACHE_GONE_CODES:
                        raise
                    logger.warning("prefix cache %s: gone (%s), recreating", cached[0][:16], exc)
                    assert self.prefix_cache is not None
                    self.prefix_cache.forget(*cached)
                    cached = await self._cached(messages, settings, model_request_parameters)
                    continue
                break
            yield stream

    def prefix(self, config: Any) -> types.GenerateContentConfig:
        """The prefix a request built as `config` is cached under, for this mode."""
        built = types.GenerateContentConfig.model_validate(
            {key: config.get(key) for key in ("system_instruction", "tools", "tool_config")}
        )
        if self.mode == "like-adk":
            built.tool_config = None
        return built

    async def uncached_config(
        self,
        messages: list[ModelMessage],
        settings: dict[str, Any],
        parameters: ModelRequestParameters,
    ) -> tuple[Any, Any]:
        """The contents and config Pydantic AI builds for this request with no cache."""
        uncached = {key: value for key, value in settings.items() if key != "google_cached_content"}
        prepared_settings, prepared = self.google.prepare_request(uncached, parameters)
        return await self.google._build_content_and_config(
            messages,
            prepared_settings or {},  # type: ignore[arg-type]
            prepared,
        )

    async def _cached(
        self,
        messages: list[ModelMessage],
        settings: dict[str, Any],
        parameters: ModelRequestParameters,
    ) -> tuple[str, str] | None:
        """The prefix key and cache name this request goes against, or `None` to send it
        uncached, as `PrefixCachedGemini._cached` answers."""
        if self.prefix_cache is None:
            return None
        try:
            _, config = await self.uncached_config(messages, settings, parameters)
            prefix = self.prefix(config)
            key = prefix_key(self.google.model_name, prefix)
            if key is None:
                return None
            self.cache_keys.append(key)
            name = await self.prefix_cache.cache_name(
                key,
                client=self.planner_client.client,
                model=self.google.model_name,
                config=prefix,
            )
        except Exception as exc:
            logger.warning("prefix cache: lookup failed (%s): %s", type(exc).__name__, exc)
            return None
        return None if name is None else (key, name)


def retry_options(model_name: str) -> types.HttpRetryOptions:
    """The retry options production's planner model is built with
    (`orchestrator.agent._retrying`), read off the model it builds."""
    model = _retrying(model_name)
    options = getattr(model, "retry_options", None)
    assert isinstance(options, types.HttpRetryOptions)
    return options


def genai_client(model_name: str, *, api_key: str | None = None) -> Client:
    """A google-genai client configured as ADK's `Gemini.api_client` is for the planner:
    production's retry options. ADK's tracking headers are left out."""
    http_options = types.HttpOptions(retry_options=retry_options(model_name))
    return Client(api_key=api_key, http_options=http_options)


def google_model(model_name: str, client: Client, *, mode: Mode) -> GoogleModel:
    """`GoogleModel` on `client`, declaring each tool's JSON schema as given.

    The profile is Gemini's own less its JSON schema transformer, which would rewrite
    the declarations ADK sends verbatim, and with inline system prompts on, which keeps
    Pydantic AI's `<system>` tag sentence out of the instruction. In `like-adk` mode it also turns off strict
    tool definitions, so a request sent uncached states `AUTO`, the mode ADK's absent
    tool config means, and not `VALIDATED`.
    """
    profile = dict(google_model_profile(model_name) or {})
    profile["json_schema_transformer"] = None
    # Off, Pydantic AI puts a sentence on `<system>`-tagged user text ahead of the
    # instruction on every request, and escapes `<system>` in user prompts and tool
    # returns (`Model.prepare_messages`, `pydantic_ai/models/__init__.py` lines 1777 and
    # 1866-1876). The planner sends no system prompt part, so on it changes nothing else.
    profile["supports_inline_system_prompts"] = True
    if mode == "like-adk":
        profile["google_supports_strict_tool_definition"] = False
    return GoogleModel(
        model_name,
        provider=GoogleProvider(client=client),
        profile=profile,  # type: ignore[arg-type]
    )


def planner_agent(
    *,
    seam: PlannerSeam,
    model: PlannerModel,
    parts: PlannerParts | None = None,
    keep_thoughts: bool = False,
) -> Agent[PlannerDeps, str | DeferredToolRequests]:
    """The planner as a Pydantic AI agent: the seam's instruction and tools, at LOW
    thinking with summaries, dropping unsigned thoughts from history unless
    `keep_thoughts`."""
    parts = parts or PlannerParts.of(seam)
    return Agent(
        model,
        instructions=parts.instruction,
        tools=planner_tools(seam, parts),
        deps_type=PlannerDeps,
        output_type=[str, DeferredToolRequests],
        model_settings=THINKING_SETTINGS,
        capabilities=[] if keep_thoughts else [ProcessHistory(drop_unsigned_thoughts)],
        name=AGENT_NAME,
    )


__all__ = [
    "EMPTY_PARAMETERS",
    "PLANNER_BRANCH",
    "THINKING_SETTINGS",
    "CallHooks",
    "Mode",
    "PlannerClient",
    "PlannerDeps",
    "PlannerModel",
    "PlannerParts",
    "adk_events",
    "drop_unsigned_thoughts",
    "genai_client",
    "google_model",
    "planner_agent",
    "tool_context",
]
