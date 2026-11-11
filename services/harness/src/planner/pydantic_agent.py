"""The Pydantic AI planner: its agent (`planner_agent`) and its run (`run_planner`).

The agent serves each `planner.tool_spec.ToolSpec` with the declaration the
ADK planner shows for it (`function_declaration`), and a call runs the spec's
body over a `planner.pydantic_call.PydanticCall` (`run_tool`), with the
model's arguments as sent. `run_planner` drives one run with `Agent.iter`
from the user's message and the brief, as the ADK planner's `single_turn`
run starts from the turn's question and the brief (`planner.agent.
prepend_brief`), and yields each model response and each call's answer, with
the state the call wrote, as the run makes them. A tool that ends the run (`planner.tools.EndRun`) stops it
before the next model request.

What the model is sent equals the ADK planner's request for the same turn,
apart from how Pydantic AI shapes a request on the wire: the system
instruction as a `Content`, one `Tool` per declaration, an empty parameters
schema on a tool that takes none, an explicit `AUTO` tool config,
`response_modalities`, its own HTTP headers, and no `automatic_continuation`,
which google-genai reads as on. The model must be built with
`planner_profile`, which sends each declaration's schema as given and adds
nothing to the instruction.
"""

from __future__ import annotations

import base64
import copy
import inspect
from collections.abc import AsyncIterator, Callable, MutableMapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from google.genai import types
from pydantic_ai import Agent, RunContext, Tool
from pydantic_ai.messages import (
    FunctionToolResultEvent,
    ModelRequest,
    ModelResponse,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    UserPromptPart,
)
from pydantic_ai.models import Model
from pydantic_ai.models.google import GoogleModelSettings
from pydantic_ai.profiles.google import GoogleModelProfile, google_model_profile
from pydantic_ai.usage import RequestUsage, UsageLimits

from planner.call import RecordingState, Step
from planner.pydantic_call import PydanticCall, as_function_response
from planner.settings import AGENT_THINKING_CONFIG
from planner.tool_spec import ToolSpec, function_declaration, run_tool

# The schema Pydantic AI declares a tool with when its declaration has none:
# Pydantic AI always sends `parameters_json_schema`.
_NO_PARAMETERS: dict[str, Any] = {"type": "object", "properties": {}}

# The turn budget bounds a run's model calls (`harness.turn_budget`), so Pydantic
# AI's own default cap of 50 requests is lifted.
_NO_REQUEST_LIMIT = UsageLimits(request_limit=None)

# Carried by the request holding the user's message. Pydantic AI merges
# consecutive requests into one content unless their `instructions` differ,
# and the brief's request carries the agent's instruction, so this keeps the
# user's message and the brief two contents, as the ADK planner sends them.
# Only the latest request's instructions reach the model, so the model never
# reads this text.
_QUESTION_INSTRUCTIONS = "The user's message, which opened the turn."

type ToolAnswered = Callable[[str, dict[str, Any]], None]
type ToolFailed = Callable[[str, BaseException], dict[str, Any]]


def planner_profile(model_name: str) -> GoogleModelProfile:
    """Pydantic AI's profile for the Gemini `model_name`, with the declarations
    sent as given (no JSON schema transformer) and nothing added to the
    instruction about system messages mid-conversation (inline system prompts
    supported)."""
    return GoogleModelProfile(
        {
            **(google_model_profile(model_name) or {}),
            "json_schema_transformer": None,
            "supports_inline_system_prompts": True,
        }
    )


@dataclass
class _Call:
    """One tool call of the run: its tool, the view it writes through, and its
    answer once it has one."""

    name: str
    view: RecordingState
    answer: dict[str, Any] | None = None
    writes: dict[str, Any] | None = None


@dataclass
class PlannerDeps:
    """One run's dependencies, which every tool call of the run reads, and what
    the run has not yet yielded.

    `state` is the session's state. Each call writes it through a
    `RecordingState` of its own, so the keys one call wrote are told apart
    from another's while every call reads the others' values. `earlier` is
    the steps of the session before the run. `tool_answered` is told each
    tool's answer and `tool_failed` each tool that raised. `ended` says
    whether a tool ended the run.
    """

    state: MutableMapping[str, Any]
    earlier: Sequence[Step]
    tool_answered: ToolAnswered
    tool_failed: ToolFailed
    ended: bool = False
    _calls: dict[str, _Call] = field(default_factory=dict, init=False)

    def end_run(self) -> None:
        """The run's `planner.tools.EndRun`: no model request follows the current calls."""
        self.ended = True

    def unyielded_writes(self) -> list[PlannerCallWrites]:
        """The writes of every call the run has not yielded an answer for, in the
        order the calls began.

        A run whose iteration is cancelled or raises while calls are running
        yields nothing more, so this is where the state those calls wrote,
        finished or still in flight, is read. Each call's writes are as they
        stand now, or as they stood when it answered.
        """
        return [
            PlannerCallWrites(
                name=call.name,
                call_id=call_id,
                writes=call.writes if call.writes is not None else self._written(call.view),
            )
            for call_id, call in self._calls.items()
        ]

    def _written(self, view: RecordingState) -> dict[str, Any]:
        """The keys written through `view`, with a copy of their values. A key
        deleted since holds no value and is left out."""
        return {key: copy.deepcopy(self.state[key]) for key in view.written if key in self.state}


@dataclass(frozen=True)
class PlannerResponse:
    """One model response of the run, as soon as it is received and before its calls run.

    `parts` are its parts as google-genai parts: each function call with its
    id, text, and each thought with `thought` set, each with its thought
    signature. `usage` is the request's usage as Pydantic AI reads it.
    """

    parts: tuple[types.Part, ...]
    usage: RequestUsage


@dataclass(frozen=True)
class PlannerToolAnswer:
    """One call's answer, as soon as the call has answered.

    `writes` is every state key the call wrote, with its value when it
    answered. The answers of one response come in the order their calls
    finish.
    """

    name: str
    call_id: str
    answer: dict[str, Any]
    writes: dict[str, Any]


@dataclass(frozen=True)
class PlannerCallWrites:
    """The state keys a call wrote, with their values, for a call whose answer
    the run has not yielded (`PlannerDeps.unyielded_writes`)."""

    name: str
    call_id: str
    writes: dict[str, Any]


type PlannerItem = PlannerResponse | PlannerToolAnswer


def planner_agent(
    specs: Sequence[ToolSpec], *, instruction: str, model: Model
) -> Agent[PlannerDeps, str | None]:
    """The planner as a Pydantic AI agent on `model`, with `instruction` and one tool per spec.

    Each tool is declared as `function_declaration(spec)` states it, its
    schema as it stands, and runs the spec's body over the model's arguments
    as sent (`_tool_function`). Every tool is declared not strict, which keeps
    the request's function calling mode `AUTO`, the mode the ADK planner's
    request leaves Gemini on. A response with no tool call ends the run, an
    empty one included, as it ends the ADK planner's. Thinking is
    `planner.settings.AGENT_THINKING_CONFIG`.
    """
    return Agent(
        model,
        instructions=instruction,
        tools=[_tool(spec) for spec in specs],
        deps_type=PlannerDeps,
        output_type=str | None,
        model_settings=GoogleModelSettings(
            google_thinking_config=AGENT_THINKING_CONFIG.model_dump(mode="json", exclude_none=True)
        ),
    )


def _tool(spec: ToolSpec) -> Tool[PlannerDeps]:
    declaration = function_declaration(spec)
    schema = declaration.parameters_json_schema
    tool = Tool.from_schema(
        _tool_function(spec),
        name=spec.name,
        description=declaration.description,
        json_schema=dict(schema) if schema is not None else _NO_PARAMETERS,
        takes_ctx=True,
    )
    tool.strict = False
    return tool


def _tool_function(spec: ToolSpec) -> Callable[..., Any]:
    """The function Pydantic AI calls for `spec`, with the model's arguments unvalidated.

    It runs the spec's body (`run_tool`) over the call's `PydanticCall`, whose
    state is a view of the call's own. An answer that is no dict is wrapped
    under `result`, as ADK wraps one. A body that raises is answered with
    `tool_failed`'s answer, as the ADK planner's tool error callback answers
    it, and a cancellation propagates. Either answer goes to `tool_answered`
    and is kept, with the call's writes, for the run to yield.
    """

    async def call(ctx: RunContext[PlannerDeps], **raw: Any) -> dict[str, Any]:
        deps = ctx.deps
        record = _Call(name=spec.name, view=RecordingState(deps.state))
        if ctx.tool_call_id is not None:
            deps._calls[ctx.tool_call_id] = record
        planner_call = PydanticCall(
            state=record.view,
            call_id=ctx.tool_call_id,
            earlier=deps.earlier,
            messages=tuple(ctx.messages),
        )
        try:
            answer = run_tool(spec, raw, call=planner_call, end_run=deps.end_run)
            if inspect.isawaitable(answer):
                answer = await answer
        except Exception as exc:
            answer = deps.tool_failed(spec.name, exc)
        response = as_function_response(answer)
        record.writes = deps._written(record.view)
        record.answer = response
        deps.tool_answered(spec.name, response)
        return response

    call.__name__ = spec.name
    return call


async def run_planner(
    agent: Agent[PlannerDeps, str | None], deps: PlannerDeps, *, question: str, brief: str
) -> AsyncIterator[PlannerItem]:
    """Run the planner once on `brief`, after the user's `question`, yielding as it goes.

    The run's first request holds `question`, then `brief`, as two user
    contents. Each model response is yielded as a `PlannerResponse` once
    received, before its calls run, and each call's answer as a
    `PlannerToolAnswer` once that call has answered. The run stops before its
    next model request once a tool ended it, and ends on a response with no
    tool call.

    `deps` holds the session's state, the earlier steps and the callbacks
    (`PlannerDeps`). A run whose iteration stops early leaves the writes of
    the calls it did not yield in `deps.unyielded_writes()`.
    """
    history = [ModelRequest(parts=[UserPromptPart(question)], instructions=_QUESTION_INSTRUCTIONS)]
    async with agent.iter(
        brief, message_history=history, deps=deps, usage_limits=_NO_REQUEST_LIMIT
    ) as run:
        async for node in run:
            if Agent.is_model_request_node(node):
                if deps.ended:
                    return
                async with node.stream(run.ctx) as stream:
                    async for _event in stream:
                        pass
                    response = stream.response
                yield PlannerResponse(parts=response_parts(response), usage=response.usage)
            elif Agent.is_call_tools_node(node):
                async with node.stream(run.ctx) as events:
                    async for event in events:
                        if not isinstance(event, FunctionToolResultEvent):
                            continue
                        answered = _answered(deps, event.tool_call_id)
                        if answered is not None:
                            yield answered


def _answered(deps: PlannerDeps, call_id: str) -> PlannerToolAnswer | None:
    """The answer of the call `call_id`, taken out of what the run has not yielded,
    or `None` for a call no planner tool served."""
    call = deps._calls.get(call_id)
    if call is None or call.answer is None or call.writes is None:
        return None
    del deps._calls[call_id]
    return PlannerToolAnswer(
        name=call.name, call_id=call_id, answer=call.answer, writes=call.writes
    )


def response_parts(response: ModelResponse) -> tuple[types.Part, ...]:
    """`response`'s parts as the google-genai parts Gemini sent for them.

    A thought signature is restored to the part that carried it, which
    Pydantic AI keeps in the part's `provider_details`.

    Raises:
        ValueError: A part is none of a function call, text or a thought.
    """
    parts: list[types.Part] = []
    for part in response.parts:
        details = part.provider_details or {}
        encoded = details.get("thought_signature")
        signature = base64.b64decode(encoded) if encoded else None
        match part:
            case ToolCallPart():
                call = types.FunctionCall(
                    id=part.tool_call_id, name=part.tool_name, args=part.args_as_dict()
                )
                parts.append(types.Part(function_call=call, thought_signature=signature))
            case TextPart():
                parts.append(types.Part(text=part.content, thought_signature=signature))
            case ThinkingPart():
                parts.append(
                    types.Part(text=part.content, thought=True, thought_signature=signature)
                )
            case _:
                raise ValueError(f"the planner's response holds a {part.part_kind} part")
    return tuple(parts)
