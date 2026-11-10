"""The served planner as a Pydantic AI agent on `GoogleModel`.

What the planner is shown and every rule the seam applies come from production objects.
This module changes the agent loop only:

- The instruction and each tool's declaration are read off the ADK agent the seam builds
  (`PlannerParts`). A declaration is data: ADK builds it from the tool function's name,
  signature and docstring, and Pydantic AI is given it verbatim. No ADK tool object runs.
- Each tool is a Pydantic AI tool whose body calls the neutral function in
  `planner.tools` or `planner.skills` with a `PaiCall` over the run's messages and an
  `EndRun` that stops the run before its next model request (`PlannerDeps.end_run`).
  The seam's tool callbacks run around it as ADK's tool caller runs them.
- Each model request goes through the seam's before-model callback and the shared prefix
  cache (`PlannerModel`). Each call at the google-genai client keeps to the request's
  `TurnBudget` and runs the after-model callbacks (`PlannerClient`).
- Thought summaries leave the history as `planner.settings.drop_thought_parts` removes
  them (`drop_unsigned_thoughts`).

Private names relied on: `PlannerSeam._agent`, `_before_planner_model`,
`_after_planner_model`, `_after_planner_tool`, `_on_planner_tool_error` (harness);
`FunctionTool._get_declaration`, `_preprocess_args` and `_get_mandatory_args` (ADK); `GoogleModel._build_content_and_config`
(Pydantic AI); `orchestrator.agent._retrying` (harness).
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import AsyncIterator, Callable, Mapping
from contextlib import AsyncExitStack, aclosing, asynccontextmanager
from dataclasses import dataclass, field, replace
from types import SimpleNamespace
from typing import Any

from compiler.surface import ToolSurface
from google.adk.models.llm_response import LlmResponse
from google.adk.tools.base_tool import BaseTool
from google.genai import Client, types
from harness.planner_seam import PlannerSeam
from harness.prefix_cache import PrefixCache, prefix_key
from harness.turn_budget import CLOSED_BY_KEY, bounded
from orchestrator.agent import _retrying
from pai_call import PaiCall
from planner.agent import AGENT_NAME
from planner.settings import AGENT_THINKING_CONFIG
from planner.skills import GET_SKILL, GET_SKILL_FILE, read_skill_page
from planner.tools import EndRun, ask_user, propose, propose_and_translate, reject
from pydantic_ai import Agent, RunContext, Tool
from pydantic_ai.capabilities.process_history import ProcessHistory
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.messages import ModelMessage, ModelResponse, ThinkingPart
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.models.google import GoogleModel, GoogleModelSettings
from pydantic_ai.models.wrapper import WrapperModel
from pydantic_ai.profiles.google import google_model_profile
from pydantic_ai.providers.google import GoogleProvider
from pydantic_ai.settings import ModelSettings
from submission.service import Proposal
from translator.tool import Translate

logger = logging.getLogger(__name__)

# Gemini answers a request whose `cached_content` is gone with 403 (404 on Vertex);
# `harness.prefix_cache._CACHE_GONE_CODES`.
_CACHE_GONE_CODES = frozenset({403, 404})

# What `custom_metadata` carries the raw google-genai chunk under while `bounded` holds it.
_RAW_CHUNK = "pai.raw_chunk"

# The thinking config ADK sends (`planner.settings.AGENT_THINKING_CONFIG`).
THINKING_SETTINGS = GoogleModelSettings(
    google_thinking_config=AGENT_THINKING_CONFIG.model_dump(mode="json", exclude_none=True)
)

# `GoogleModelSettings.google_cached_content` warns each time it strips the prefix from a
# request; stripping it is the point here.
warnings.filterwarnings("ignore", message=r"`google_cached_content` is set")

# ADK ends a run on any response with no function call, a response with no parts included.
# Pydantic AI retries a response with no text output unless the output type allows `None`.
OUTPUT_TYPE = str | None

# The JSON schema Pydantic AI is given for a declaration ADK sends with no parameters
# (`reject_plan`). Pydantic AI always sends `parameters_json_schema`.
EMPTY_PARAMETERS: dict[str, Any] = {"type": "object", "properties": {}}


# --- What the ADK planner is built from ----------------------------------------------


@dataclass(frozen=True)
class PlannerParts:
    """The instruction, and each tool's name and declaration, read off the seam's ADK
    agent in the order ADK declares them. The tool objects are kept for their names, which
    the seam's tool callbacks read; none is run."""

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


# --- The session -------------------------------------------------------------------


@dataclass
class PaiSession:
    """The record of one entry's session: every message of every run, in order.

    It opens with the user's message (`pai_call.user_message`). A run starts from that
    message alone and the seam's brief, as ADK's `single_turn` run reads only its own node
    path, and its messages join the record when it ends. The user's answer to an ask
    joins it before the run that answer resumes (`pai_call.answer_message`).
    """

    id: str
    record: list[ModelMessage]

    def history(self) -> list[ModelMessage]:
        """What a run starts from: the user's message."""
        return [self.record[0]]

    def call(self, ctx: RunContext[PlannerDeps]) -> PaiCall:
        """The `PlannerCall` for the tool call `ctx` serves: the record, then the current
        run's own messages (those after the user's message it started from)."""
        return PaiCall(
            messages=[*self.record, *ctx.messages[1:]],
            state=ctx.deps.state,
            call_id=ctx.tool_call_id,
        )


@dataclass
class PlannerDeps:
    """One entry's run state: the session state ADK's planner reads and writes, the
    session, and whether a tool ended the run."""

    state: dict[str, Any]
    session: PaiSession
    ended: bool = False

    def end_run(self) -> None:
        """The `EndRun` the tools end the planner's run with: Pydantic AI stops before the
        next model request (`pai_entry.PaiRunner.run`), where ADK skips summarisation."""
        self.ended = True


# --- The tools ---------------------------------------------------------------------


@dataclass(frozen=True)
class ToolWiring:
    """What the planner's tool functions close over in `planner.agent.build_planner_tools`,
    for the seam's planner (`PlannerSeam.__init__` builds it with `question_fixed=True`,
    `reads_user_words=False`, its surface and its translation)."""

    surface: ToolSurface
    translate: Translate
    question_fixed: bool = True
    reads_user_words: bool = False


def _proposal(arguments: Mapping[str, Any]) -> Proposal:
    """The proposal `propose_plan`'s arguments state. The lists default to `[]`, as the
    tool function's signature does: `submission.service` tests lists by type."""
    lists = {
        name: []
        for name in (
            "decisions",
            "evidence_requirements",
            "removed_decisions",
            "removed_evidence_requirements",
            "declined",
            "answered",
            "removed_asks",
            "across_years",
        )
    }
    return Proposal(**{**lists, **arguments})


_PROPOSAL_FIELDS = frozenset(Proposal.__dataclass_fields__)


async def run_tool(
    name: str,
    args: Mapping[str, Any],
    call: PaiCall,
    end_run: EndRun,
    wiring: ToolWiring,
) -> Any:
    """What the planner's tool `name` answers for `args`, through its neutral body."""
    match name:
        case "get_skill":
            return await read_skill_page(
                wiring.surface, GET_SKILL, {"skill_id": args["skill_id"]}, call=call, path=None
            )
        case "get_skill_file":
            arguments: dict[str, object] = {
                "skill_id": args["skill_id"],
                "path": args["path"],
                "offset": args.get("offset", 0),
            }
            if args.get("digest") is not None:
                arguments["digest"] = args["digest"]
            return await read_skill_page(
                wiring.surface, GET_SKILL_FILE, arguments, call=call, path=args["path"]
            )
        case "ask_user_question":
            return ask_user(
                call,
                end_run,
                message=args["message"],
                candidates=args.get("candidates", []),
                options=args.get("options", []),
                recommended_spec_id=args.get("recommended_spec_id"),
                rationale=args.get("rationale"),
            )
        case "propose_plan":
            stated = {key: value for key, value in args.items() if key in _PROPOSAL_FIELDS}
            return await propose_and_translate(
                call,
                end_run,
                arguments=stated,
                draft=lambda: propose(
                    call,
                    _proposal(stated),
                    question_fixed=wiring.question_fixed,
                    reads_user_words=wiring.reads_user_words,
                ),
                translate=wiring.translate,
                surface=wiring.surface,
            )
        case "reject_plan":
            return reject(call)
    raise ValueError(f"the planner has no tool {name!r}")


def planner_tools(
    seam: PlannerSeam, parts: PlannerParts, wiring: ToolWiring
) -> list[Tool[PlannerDeps]]:
    """One Pydantic AI tool per planner tool, declared as ADK declares it, calling the
    neutral body and the seam's tool callbacks as ADK's tool caller does
    (`google.adk.flows.llm_flows.tools._caller`)."""
    return [
        Tool.from_schema(
            _body(seam, tool, wiring),
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


def convert_arguments(tool: BaseTool, args: Mapping[str, Any]) -> dict[str, Any]:
    """`args` as ADK hands them to the tool function: each value a `BaseModel` type stands
    for converted to it, and an item that does not validate left as the dict the model wrote
    (`FunctionTool._preprocess_args`). `submission.service` reads a dict and a model
    differently where a refusal names the pins a proposal states."""
    return tool._preprocess_args(dict(args))  # type: ignore[attr-defined]


def missing_arguments(tool: BaseTool, args: Mapping[str, Any]) -> dict[str, str] | None:
    """ADK's answer to a call that leaves out a mandatory argument, which it makes before it
    runs the tool function (`FunctionTool.run_async`), or `None` when none is missing."""
    missing = [name for name in tool._get_mandatory_args() if name not in args]  # type: ignore[attr-defined]
    if not missing:
        return None
    names = "\n".join(missing)
    return {
        "error": f"""Invoking `{tool.name}()` failed as the following mandatory input parameters are not present:
{names}
You could retry calling this tool, but it is IMPORTANT for you to provide all the mandatory parameters."""
    }


def _body(seam: PlannerSeam, tool: BaseTool, wiring: ToolWiring) -> Callable[..., Any]:
    async def call(ctx: RunContext[PlannerDeps], **args: Any) -> Any:
        # The seam's tool callbacks log the session and invocation of a failing call.
        logged = SimpleNamespace(
            session=SimpleNamespace(id=ctx.deps.session.id), invocation_id=ctx.run_id
        )
        try:
            result = missing_arguments(tool, args)
            if result is None:
                result = await run_tool(
                    tool.name,
                    convert_arguments(tool, args),
                    ctx.deps.session.call(ctx),
                    ctx.deps.end_run,
                    wiring,
                )
        except Exception as exc:
            result = seam._on_planner_tool_error(tool, args, logged, exc)
            if result is None:
                raise
        seam._after_planner_tool(tool, args, logged, result)
        # ADK answers the model with `{"result": value}` for a value that is no dict;
        # Pydantic AI would send `{"return_value": value}`.
        return result if isinstance(result, dict) else {"result": result}

    call.__name__ = tool.name
    return call


# --- The history -------------------------------------------------------------------


def drop_unsigned_thoughts(messages: list[ModelMessage]) -> list[ModelMessage]:
    """`messages` with every thinking part that carries no thought signature removed, as
    `planner.settings.drop_thought_parts` removes a thought part with none from ADK's
    session history. A thinking part with a signature stays whole."""
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

    A call the budget closes answers with the closing sentence, or with the ask the seam's
    after-model callback puts in its place. A call `PlannerModel` answers for the seam (a
    synthetic response) reaches neither the model nor the budget.
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
                    # The budget closed the call: the session keeps the final answer alone,
                    # after the after-model callbacks have seen it.
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
    (`PlannerSeam._before_planner_model`); if so the client answers with that turn and the
    model is not called. Otherwise the probe's before-model callback runs, the request's
    prefix is looked up in `prefix_cache` and the request is sent against it
    (`google_cached_content`). A cache the API reports gone (403 or 404) before any chunk
    arrived is forgotten, looked up again and the request resent once, as
    `harness.prefix_cache.PrefixCachedGemini` does. The cached prefix carries no tool
    config, as ADK's carries none.
    """

    def __init__(
        self, google: GoogleModel, *, client: PlannerClient, prefix_cache: PrefixCache | None
    ) -> None:
        super().__init__(google)
        self.google = google
        self.planner_client = client
        self.prefix_cache = prefix_cache
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
        """The prefix a request built as `config` is cached under: its system instruction
        and tools, with no tool config."""
        built = types.GenerateContentConfig.model_validate(
            {key: config.get(key) for key in ("system_instruction", "tools", "tool_config")}
        )
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
    production's retry options."""
    return Client(
        api_key=api_key, http_options=types.HttpOptions(retry_options=retry_options(model_name))
    )


def google_model(model_name: str, client: Client) -> GoogleModel:
    """`GoogleModel` on `client`, declaring each tool's JSON schema as given.

    The profile is Gemini's own with three changes:

    - no JSON schema transformer, which would rewrite the declarations ADK sends verbatim;
    - inline system prompts on, which keeps Pydantic AI's `<system>` tag sentence out of
      the instruction (`Model.prepare_messages`, `pydantic_ai/models/__init__.py` lines
      1777 and 1866-1876). The planner sends no system prompt part, so nothing else changes;
    - strict tool definitions off, so a request sent uncached states `AUTO`, the mode
      ADK's absent tool config means, and not `VALIDATED`.
    """
    profile = dict(google_model_profile(model_name) or {})
    profile["json_schema_transformer"] = None
    profile["supports_inline_system_prompts"] = True
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
    wiring: ToolWiring,
    parts: PlannerParts | None = None,
    keep_thoughts: bool = False,
) -> Agent[PlannerDeps, str | None]:
    """The planner as a Pydantic AI agent: the seam's instruction and tools, at LOW thinking
    with summaries, dropping unsigned thoughts from history unless `keep_thoughts`."""
    parts = parts or PlannerParts.of(seam)
    return Agent(
        model,
        instructions=parts.instruction,
        tools=planner_tools(seam, parts, wiring),
        deps_type=PlannerDeps,
        output_type=OUTPUT_TYPE,
        model_settings=THINKING_SETTINGS,
        capabilities=[] if keep_thoughts else [ProcessHistory(drop_unsigned_thoughts)],
        name=AGENT_NAME,
    )
