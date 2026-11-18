"""The planner: its Pydantic AI agent (`planner_agent`) and its run (`run_planner`).

The agent serves each `planner.tool_spec.ToolSpec` with its declaration
(`function_declaration`), and a call runs the spec's body (`serve_tool`) over
the `planner.call.PlannerCall` its run's `PlannerDeps` makes, with the
model's arguments as sent. `run_planner` drives one run with `Agent.iter`
from the turn's question and the brief, and yields each model response as it
arrives and then its calls' answers, with the state they wrote, once every
call has answered. A tool that ends the run (`planner.tools.EndRun`) stops it
before the next model request.

Pydantic AI shapes a request its own way: the system instruction as a
`Content`, one `Tool` per declaration, an empty parameters schema on a tool
that takes none, an explicit `AUTO` tool config and `response_modalities`.
The planner's model (`harness.gemini_wire`) puts each request in one fixed
form before it is sent. The model is built with
`harness.gemini_agent.GEMINI_PROFILE`, which sends each declaration's schema
as given and adds nothing to the instruction.
"""

from __future__ import annotations

import copy
import logging
from collections.abc import AsyncGenerator, Callable, MutableMapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from google.genai import types
from harness.gemini_agent import (
    NO_REQUEST_LIMIT,
    QUESTION_INSTRUCTIONS,
    UNLIMITED_RETRIES,
    RecordingState,
    as_function_response,
    declared_tool,
    replayed_history,
    response_parts,
    tool_not_found,
    without_unsigned_thoughts,
)
from pydantic_ai import Agent, InstrumentationSettings, RunContext, Tool
from pydantic_ai.capabilities import Instrumentation, ProcessHistory
from pydantic_ai.exceptions import UnexpectedModelBehavior
from pydantic_ai.messages import ModelRequest, ModelResponse, ToolReturnPart, UserPromptPart
from pydantic_ai.models import Model
from pydantic_ai.models.google import GoogleModelSettings

from planner.call import PlannerCall
from planner.plan import SessionState
from planner.settings import AGENT_THINKING_CONFIG
from planner.tool_spec import ToolSpec, function_declaration, serve_tool

logger = logging.getLogger(__name__)

# Makes the `PlannerCall` of one tool call from the state view the call's
# response writes through and the call's id.
type CallFor = Callable[[SessionState, str | None], PlannerCall]
type ToolAnswered = Callable[[str, dict[str, Any]], None]
type ToolFailed = Callable[[str, BaseException], dict[str, Any]]


@dataclass
class PlannerDeps:
    """One run's dependencies, which every tool call of the run reads.

    `state` is the session's state. The calls of one model response write it
    through one `RecordingState`, which `run_planner` replaces before each
    model request, so the keys a response's calls wrote are told apart from
    the keys of the responses before it. `call_for` makes the `PlannerCall` a
    tool call runs over, from that state view and the call's id, so the
    planner reads the session through whatever stores it. `tool_answered` is
    told each tool's answer and `tool_failed` each tool that raised. `ended`
    says whether a tool ended the run.
    """

    state: MutableMapping[str, Any]
    call_for: CallFor
    tool_answered: ToolAnswered
    tool_failed: ToolFailed
    ended: bool = False
    _response_view: RecordingState = field(init=False)

    def __post_init__(self) -> None:
        self._response_view = RecordingState(self.state)

    def end_run(self) -> None:
        """Mark the run ended, so no model request follows the current calls."""
        self.ended = True


@dataclass(frozen=True)
class PlannerResponse:
    """One model response of the run, yielded on receipt, before its calls run.

    `parts` are its parts as google-genai parts: each function call with its
    id, text, and each thought with `thought` set, each with its thought
    signature, less its unsigned thought summaries
    (`harness.gemini_agent.without_unsigned_thoughts`).
    A blocked prompt is a response with no parts. The usage the API reported is
    read off the stream by `harness.gemini_wire`.
    """

    parts: tuple[types.Part, ...]


@dataclass(frozen=True)
class PlannerAnswers:
    """The answers of one model response's calls, once every call has answered.

    `answers` holds each call's answer as a function response, with the call's
    id and tool, in the order of the calls in the response. `state` is the
    session's value, after the last call answered, of every key the
    response's calls wrote. A key deleted since holds no value and is left out.
    """

    answers: tuple[types.FunctionResponse, ...]
    state: dict[str, Any]


type PlannerItem = PlannerResponse | PlannerAnswers


@dataclass(frozen=True)
class PydanticPlanner:
    """The planner's Pydantic AI agent (`planner_agent`) and its tools' names,
    in the order they are declared."""

    agent: Agent[PlannerDeps, str | None]
    tools: tuple[str, ...]


def planner_agent(
    specs: Sequence[ToolSpec],
    *,
    instruction: str,
    model: Model | None = None,
    name: str | None = None,
    include_content: bool = True,
) -> PydanticPlanner:
    """The planner as a Pydantic AI agent named `name` on `model`, with
    `instruction` and one tool per spec. With no `model` each run names its
    own (`run_planner`), so one agent serves concurrent runs, each on a model
    of its own.

    Each tool is declared as `function_declaration(spec)` states it, its
    schema as it stands, and runs the spec's body over the model's arguments
    as sent (`_tool_function`). Every tool is declared not strict, which keeps
    the request's function calling mode `AUTO`, Gemini's default with tools.
    A response with no tool call ends the run, an empty one included.
    Thinking is `planner.settings.AGENT_THINKING_CONFIG`.

    A call to a tool the planner lacks is answered with
    `harness.gemini_agent.tool_not_found`, and the run goes on however many
    such calls the model makes (`harness.gemini_agent.UNLIMITED_RETRIES`). Each request replays
    the run's earlier responses less their unsigned thought summaries, as the
    session keeps them (`harness.gemini_agent.replayed_history`).

    The agent traces through the global tracer provider: a span per run
    (`invoke_agent <name>`), per model call (`chat <model>`) and per tool call
    (`execute_tool <tool>`), with the OpenTelemetry GenAI attributes. With
    `include_content` the spans carry the model's messages, each response's
    thought summaries among them, the instruction and each call's arguments
    and answer. The request parameters are left off, since the tool
    declarations are on each model call's span.
    """
    tools = tuple(spec.name for spec in specs)
    agent = Agent(
        model,
        instructions=instruction,
        tools=[_tool(spec) for spec in specs],
        deps_type=PlannerDeps,
        output_type=str | None,
        model_settings=GoogleModelSettings(
            google_thinking_config=AGENT_THINKING_CONFIG.model_dump(mode="json", exclude_none=True)
        ),
        retries={"tools": UNLIMITED_RETRIES},
        capabilities=[
            ProcessHistory(replayed_history(tools)),
            Instrumentation(
                InstrumentationSettings(
                    include_content=include_content, include_model_request_parameters=False
                )
            ),
        ],
        name=name,
    )
    return PydanticPlanner(agent=agent, tools=tools)


def _tool(spec: ToolSpec) -> Tool[PlannerDeps]:
    return declared_tool(
        _tool_function(spec), name=spec.name, declaration=function_declaration(spec)
    )


def _tool_function(spec: ToolSpec) -> Callable[..., Any]:
    """The function Pydantic AI calls for `spec`, taking the arguments as sent.

    It runs the spec's body (`serve_tool`) over the call's `PlannerCall`, whose
    state is the view the response's calls write through. An answer that is
    no dict is wrapped under `result`. A body that raises is
    answered with `tool_failed`'s answer, and a cancellation propagates. An
    answer of the body's holding an `error` key, the refusal of a call missing a required
    argument (`planner.tool_spec.missing_arguments`), marks the call's span
    failed (`planner.tool_spec.serve_tool`). Either answer goes to
    `tool_answered` as the call finishes, and what that raises fails the run.
    """

    async def call(ctx: RunContext[PlannerDeps], **raw: Any) -> dict[str, Any]:
        deps = ctx.deps
        planner_call = deps.call_for(deps._response_view, ctx.tool_call_id)
        try:
            answer = await serve_tool(spec, raw, call=planner_call, end_run=deps.end_run)
        except Exception as exc:
            answer = deps.tool_failed(spec.name, exc)
        response = as_function_response(answer)
        deps.tool_answered(spec.name, response)
        return response

    call.__name__ = spec.name
    return call


async def run_planner(
    planner: PydanticPlanner,
    deps: PlannerDeps,
    *,
    question: types.Content | str | None,
    brief: str,
    model: Model | None = None,
    conversation_id: str | None = None,
) -> AsyncGenerator[PlannerItem]:
    """Run the planner once on the user's `question` and `brief`, yielding as it goes.

    The run's first request holds `question`, then `brief`, as two user
    contents, or `brief` alone for a turn with no question. A question given as
    a `Content` is sent with its parts as the user sent them
    (`_question_prompt`). Each model response is yielded as a
    `PlannerResponse` once received, before its calls run. Once every call
    has answered, their answers are yielded as one `PlannerAnswers`, before
    the next model request. They are the tool returns Pydantic AI collects
    into that request (`ModelRequestNode.request`), in call order, with
    `tool_not_found`'s answer for a call to a tool the planner lacks. The run
    stops after yielding them once a tool ended it, and ends on a response
    with no tool call.

    A response Pydantic AI refuses (`UnexpectedModelBehavior`, as for a stream
    with no chunk or a blocked prompt) ends the run, as a response with no tool
    call does, and raises nothing. A blocked prompt's response is yielded
    first, with no parts. Any other exception is raised. A run that ends either
    way while a response's calls run yields none of their answers. Closing the
    iterator closes the run (`Agent.iter`) and any stream it holds open.

    `deps` holds the session's state, how a tool call reads it, and the callbacks
    (`PlannerDeps`). `model` is the model the run calls, or None for the
    agent's own. `conversation_id` is the conversation the run's spans name
    (`gen_ai.conversation.id`). None leaves Pydantic AI to make one up.
    """
    history = (
        [
            ModelRequest(
                parts=[UserPromptPart(_question_prompt(question))],
                instructions=QUESTION_INSTRUCTIONS,
            )
        ]
        if question is not None
        else []
    )
    async with planner.agent.iter(
        brief,
        model=model,
        message_history=history,
        deps=deps,
        usage_limits=NO_REQUEST_LIMIT,
        conversation_id=conversation_id,
    ) as run:
        try:
            async for node in run:
                if not Agent.is_model_request_node(node):
                    continue
                returns = [part for part in node.request.parts if isinstance(part, ToolReturnPart)]
                if returns:
                    yield _answers(returns, deps, planner.tools)
                if deps.ended:
                    return
                deps._response_view = RecordingState(deps.state)
                async with node.stream(run.ctx) as stream:
                    async for _event in stream:
                        pass
                    response = stream.response
                yield _planner_response(response)
        except UnexpectedModelBehavior as exc:
            # The exception's text can hold the response (its `body`, a refusal,
            # a part's repr), so only its class is logged.
            logger.warning(
                "planner run ended on a response Pydantic AI refused: %s", type(exc).__name__
            )


class UnsentQuestionPartError(ValueError):
    """The turn's question holds a part the planner does not send
    (`_question_prompt`)."""


def _question_prompt(question: types.Content | str) -> str | list[str]:
    """`question`'s parts as the user prompt Pydantic AI sends as one content of
    the same parts: one text part as its text, several as a list of them.

    Studio sends one text part (`harness.studio`). Any other part is refused,
    and so is a question with no part, which Gemini rejects as an empty content.

    Raises:
        UnsentQuestionPartError: A part is not text, the message naming the
            part's fields, or the question holds no part.
    """
    if isinstance(question, str):
        return question
    if not question.parts:
        raise UnsentQuestionPartError("the question holds no part; the planner takes text parts")
    prompt: list[str] = []
    for part in question.parts:
        if part.text is None:
            received = sorted(part.model_dump(exclude_none=True))
            raise UnsentQuestionPartError(
                f"the question holds a part with {received}; the planner takes text parts only"
            )
        prompt.append(part.text)
    return prompt[0] if len(prompt) == 1 else prompt


def _planner_response(response: ModelResponse) -> PlannerResponse:
    return PlannerResponse(parts=response_parts(without_unsigned_thoughts(response)))


def _answers(
    returns: Sequence[ToolReturnPart], deps: PlannerDeps, tools: Sequence[str]
) -> PlannerAnswers:
    """The answers `returns`, a response's tool returns in call order, stand for:
    a planner tool's own answer, or `tool_not_found`'s for a tool the planner
    lacks, with the state the response's calls wrote."""
    view = deps._response_view
    return PlannerAnswers(
        answers=tuple(
            types.FunctionResponse(
                id=part.tool_call_id,
                name=part.tool_name,
                response=as_function_response(part.content)
                if part.tool_name in tools
                else tool_not_found(part.tool_name, tools),
            )
            for part in returns
        ),
        state={key: copy.deepcopy(deps.state[key]) for key in view.written if key in deps.state},
    )
