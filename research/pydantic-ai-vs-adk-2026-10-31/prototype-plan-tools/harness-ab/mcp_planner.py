"""DFO's planner with its planning tools reached over MCP, on plan-mcp.

`McpPlannerSeam` is `harness.planner_seam.PlannerSeam` with the same agent,
model, settings, callbacks and skill tools, but its planning tools are
plan-mcp's, called through `compiler.surface.McpSurface`, the client the
harness reaches data-mcp and analysis-mcp through:

- the seam opens a plan session with `start_plan` (the orchestrator's brief, asks
  and searches) and gives the planner the first message plan-mcp renders, in
  place of rendering it in process;
- the model's `propose_plan` is plan-mcp's `propose_plan`, declared as plan-mcp
  lists it, with `plan_session` filled in by the harness;
- the model's ask is plan-mcp's `record_user_answer`, declared as plan-mcp lists
  it less `plan_session` and `answer`. Calling it pauses the turn, as the
  planner's own ask does; when the turn resumes with the user's answer, the
  seam records the question and the answer through `record_user_answer` before
  the planner runs again.

`HarnessPlanMcp` is plan-mcp with one change for a trusted client: an answer
the harness sends is recorded `by: harness`, since the harness relays what the
user said in the Studio. It also times each tool's own work, so a call's
transport overhead is its client time less its server time.
"""

from __future__ import annotations

import json
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, override

from compiler.surface import McpSurface, ToolId, ToolSchema
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext
from google.genai import types
from harness.planner_seam import PlannerSeam
from harness.settings import Settings
from harness.turn_events import REQUEST_INPUT_FUNCTION_CALL_NAME
from outward import ASK_TOOL, guidance_text, outward
from plan_tools import SESSION_UNKNOWN, refusal
from planner.agent import AGENT_NAME as PLANNER_AGENT
from planner.tool_results import ProblemCode
from server import PlanMcp

SERVER = "plan-mcp"
PROPOSE_PLAN = "propose_plan"
START_PLAN = "start_plan"
BY_HARNESS = "harness"

# Harness session state: the plan session this turn's planner drafts in, the
# first message plan-mcp rendered for it, and the asks already recorded there.
PLAN_SESSION_KEY = "planner:plan_session"
MESSAGE_KEY = "planner:plan_message"
RECORDED_ASKS_KEY = "planner:recorded_asks"

# The planner's instruction names `reject_plan`, which plan-mcp does not serve.
REJECT_SENTENCE = (
    " Call `reject_plan` when the user declines the plan and asks for nothing in its place."
)
_SKILL_TOOLS = ("get_skill", "get_skill_file")


def mcp_instruction(instruction: str) -> str:
    """The planner's instruction for the tools plan-mcp serves: its ask tool named
    `record_user_answer` (`outward.guidance_text`) and no `reject_plan`."""
    if REJECT_SENTENCE not in instruction:
        raise ValueError("the planner's instruction no longer names reject_plan as expected")
    return guidance_text(instruction.replace(REJECT_SENTENCE, ""))


def without(schema: Mapping[str, Any], hidden: Sequence[str]) -> dict[str, Any]:
    """`schema` with the top-level properties `hidden` left out of it and of `required`."""
    out = dict(schema)
    out["properties"] = {k: v for k, v in schema["properties"].items() if k not in hidden}
    if "required" in schema:
        out["required"] = [name for name in schema["required"] if name not in hidden]
    return out


@dataclass
class McpCall:
    """One tool call the harness made to plan-mcp, as the client timed it."""

    tool: str
    client_s: float
    is_error: bool
    chars: int


@dataclass
class PlanClient:
    """The harness's client of plan-mcp: an `McpSurface` over one source, the tools
    it listed, and every call's time."""

    surface: McpSurface
    listed: dict[str, ToolSchema]
    calls: list[McpCall] = field(default_factory=list)

    @classmethod
    async def connect(cls, surface: McpSurface) -> PlanClient:
        listed = {tool.name: tool for tool in await surface.list_tools()}
        return cls(surface=surface, listed=listed)

    async def call(self, tool: str, arguments: Mapping[str, Any]) -> dict[str, Any]:
        started = time.perf_counter()
        # `ToolId.server` admits only data-mcp and analysis-mcp; a served version
        # would add plan-mcp to that literal.
        tool_id = ToolId.model_construct(server=SERVER, name=tool)
        result = await self.surface.call_tool(tool=tool_id, arguments=dict(arguments))
        answer = dict(result.structured_content or {})
        if not answer and result.text:
            answer = {"outcome": "error", "message": result.text}
        self.calls.append(
            McpCall(
                tool=tool,
                client_s=time.perf_counter() - started,
                is_error=result.is_error,
                chars=len(result.text),
            )
        )
        return answer


# What the eval runs on each `propose_plan` answer before the model reads it: it may
# raise to end the entry, as the eval's translate step does in process.
type AnswerHook = Callable[[Mapping[str, Any], ToolContext], None]


class McpPlanTool(BaseTool):
    """A plan-mcp tool as the planner's model sees it: plan-mcp's listed declaration
    less the arguments the harness fills, its answer the tool's structured result."""

    def __init__(
        self,
        client: PlanClient,
        name: str,
        *,
        hidden: Sequence[str] = ("plan_session",),
        on_answer: AnswerHook | None = None,
        declared: types.FunctionDeclaration | None = None,
    ) -> None:
        listed = client.listed[name]
        description = listed.description if declared is None else str(declared.description)
        super().__init__(name=name, description=description)
        self._client = client
        self._schema = (
            without(listed.parameters, hidden)
            if declared is None
            else dict(declared.parameters_json_schema or {})
        )
        self._on_answer = on_answer

    @override
    def _get_declaration(self) -> types.FunctionDeclaration:
        return types.FunctionDeclaration(
            name=self.name, description=self.description, parameters_json_schema=self._schema
        )

    @override
    async def run_async(self, *, args: dict[str, Any], tool_context: ToolContext) -> Any:
        session = tool_context.state.get(PLAN_SESSION_KEY)
        answer = await self._client.call(self.name, {"plan_session": session, **args})
        if self._on_answer is not None:
            self._on_answer(answer, tool_context)
        return answer


class McpAskTool(McpPlanTool):
    """`record_user_answer` as the model calls it: it pauses the turn, as the
    planner's own ask does, and the seam records it once the user answers."""

    def __init__(self, client: PlanClient) -> None:
        super().__init__(client, ASK_TOOL, hidden=("plan_session", "answer"))
        self.is_long_running = True

    @override
    async def run_async(self, *, args: dict[str, Any], tool_context: ToolContext) -> Any:
        del args, tool_context


# The orchestrator's hand-off a plan session opens from: its asks and searches.
type HandOff = Callable[[], Awaitable[tuple[list[Any], list[dict[str, Any]]]]]


class McpPlannerSeam(PlannerSeam):
    """`PlannerSeam` with its planning tools on plan-mcp, for a turn's first entry.

    `hand_off` gives the asks and searches of the orchestrator's `plan` call.
    `served_declaration` declares `propose_plan` as the served planner declares
    it, bounds and defaults included, in place of plan-mcp's lean listing. The
    other arguments are `PlannerSeam`'s; `translate` is never called, since
    translation runs inside plan-mcp's `propose_plan`.
    """

    def __init__(
        self,
        *,
        client: PlanClient,
        hand_off: HandOff,
        on_propose: AnswerHook | None = None,
        served_declaration: bool = False,
        **seam: Any,
    ) -> None:
        super().__init__(settings=Settings(), translate=_translated_elsewhere, **seam)
        self._client = client
        self._hand_off = hand_off
        tools = {getattr(tool, "name", ""): tool for tool in self._agent.tools}
        declared = tools[PROPOSE_PLAN]._get_declaration() if served_declaration else None
        self._agent.tools = [
            *(tools[name] for name in _SKILL_TOOLS if name in tools),
            McpAskTool(client),
            McpPlanTool(client, PROPOSE_PLAN, on_answer=on_propose, declared=declared),
        ]
        self._agent.instruction = mcp_instruction(str(self._agent.instruction))

    @override
    async def __call__(
        self,
        *,
        ctx: Any,
        question: str,
        readings: Sequence[Mapping[str, Any]],
        prior_plan: dict[str, Any] | None = None,
        refusal: dict[str, Any] | None = None,
        amendment: str | None = None,
        canvas_digest: dict[str, Any] | None = None,
        conversation_digest: dict[str, Any] | None = None,
        on_progress: Callable[[dict[str, Any]], None] | None = None,
        pick: Any = None,
    ) -> dict[str, Any]:
        del readings
        if any((prior_plan, refusal, amendment, canvas_digest, conversation_digest, pick)):
            raise NotImplementedError("this seam drafts a turn's first plan only")
        state = ctx.session.state
        if ctx.resume_inputs and PLAN_SESSION_KEY in state:
            await self._record_answers(ctx)
            message = state[MESSAGE_KEY]
        else:
            asks, searches = await self._hand_off()
            opened = await self._client.call(
                START_PLAN, {"brief": question, "asks": asks, "searches": searches}
            )
            if opened.get("outcome") != "ok":
                return self._refused(opened)
            state[PLAN_SESSION_KEY] = opened["plan_session"]
            state[MESSAGE_KEY] = message = opened["planner_message"]
            state[RECORDED_ASKS_KEY] = []
        try:
            submitted, _plan, refused = await self._propose(
                ctx=ctx, message=message, on_progress=on_progress
            )
        finally:
            for key in (PLAN_SESSION_KEY, MESSAGE_KEY, RECORDED_ASKS_KEY):
                ctx.state[key] = state.get(key)
        if refused is not None:
            return self._refused(refused)
        assert submitted is not None
        return {
            "plan_session": state[PLAN_SESSION_KEY],
            "summary": submitted["summary"],
            "card": submitted.get("card"),
        }

    async def _record_answers(self, ctx: Any) -> None:
        """Record on plan-mcp each `record_user_answer` call of the planner the user
        has answered since the last entry, with the answer the turn resumed with."""
        state = ctx.session.state
        recorded = list(state.get(RECORDED_ASKS_KEY) or [])
        for call, response in _answered_asks(ctx.session.events):
            if call.id in recorded:
                continue
            answered = await self._client.call(
                ASK_TOOL,
                {
                    "plan_session": state[PLAN_SESSION_KEY],
                    **dict(call.args or {}),
                    "answer": user_answer(response),
                },
            )
            if answered.get("outcome") != "ok":
                raise RuntimeError(f"plan-mcp refused the user's answer: {answered}")
            recorded.append(call.id)
        state[RECORDED_ASKS_KEY] = recorded


def user_answer(response: Mapping[str, Any]) -> dict[str, Any]:
    """The answer a resumed ask carries, in plan-mcp's `answer` shape: the first
    question's answer the Studio's interrupt route sends."""
    result = response.get("result", response)
    answers = result.get("answers") if isinstance(result, Mapping) else None
    if not isinstance(answers, Mapping) or not answers:
        raise ValueError(f"no answer in {json.dumps(response)[:300]}")
    return dict(next(iter(answers.values())))


def _answered_asks(events: Sequence[Any]) -> list[tuple[types.FunctionCall, dict[str, Any]]]:
    calls = {
        call.id: call
        for event in events
        if event.author == PLANNER_AGENT
        for call in event.get_function_calls()
        if call.name == ASK_TOOL
    }
    return [
        (calls[response.id], dict(response.response or {}))
        for event in events
        for response in event.get_function_responses()
        if response.id in calls
    ]


async def _translated_elsewhere(tool_context: Any) -> dict[str, object]:
    raise AssertionError("translation runs inside plan-mcp's propose_plan")


@dataclass
class ServerTime:
    tool: str
    server_s: float


class HarnessPlanMcp(PlanMcp):
    """plan-mcp for the harness: answers it relays are recorded `by: harness`, and
    each tool's own work is timed in `server_times`."""

    def __init__(self, service: Any) -> None:
        super().__init__(service, elicitation=False, guidance_in_answer=False)
        self.server_times: list[ServerTime] = []

    def _timed(self, tool: str, started: float) -> None:
        self.server_times.append(ServerTime(tool, time.perf_counter() - started))

    @override
    async def start(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        started = time.perf_counter()
        try:
            return await super().start(arguments)
        finally:
            self._timed(START_PLAN, started)

    @override
    async def propose(self, arguments: Mapping[str, Any], *, schema: Mapping[str, Any]) -> Any:
        started = time.perf_counter()
        try:
            return await super().propose(arguments, schema=schema)
        finally:
            self._timed(PROPOSE_PLAN, started)

    @override
    async def record_user_answer(self, arguments: Mapping[str, Any]) -> Any:
        started = time.perf_counter()
        try:
            session = self.find(str(arguments.get("plan_session", "")))
            if session is None:
                return refusal(SESSION_UNKNOWN, "no plan session by that id")
            if "answer" not in arguments:
                return refusal(ProblemCode.INVALID_ARGUMENTS.value, "send `answer`")
            asked = {
                key: arguments[key]
                for key in ("message", "candidates", "options", "recommended_spec_id", "rationale")
                if key in arguments
            }
            answered = await self.service.answer(session, asked, arguments["answer"], by=BY_HARNESS)
            return outward(answered)
        finally:
            self._timed(ASK_TOOL, started)


__all__ = [
    "BY_HARNESS",
    "PLAN_SESSION_KEY",
    "REQUEST_INPUT_FUNCTION_CALL_NAME",
    "HarnessPlanMcp",
    "McpPlannerSeam",
    "PlanClient",
    "mcp_instruction",
]
