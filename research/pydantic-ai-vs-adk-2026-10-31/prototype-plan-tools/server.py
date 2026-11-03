"""`plan-mcp`: four tools that let an outside model plan and run over DFO's planner code.

- `start_plan` opens a plan session from the spec_ids the client's own `search`
  calls returned (or searches it sends to run again), its brief and its asks,
  and answers the session id, the planner's guidance and the planner's first
  message.
- `propose_plan` is the served planner's own tool: its arguments and its
  refusals, with `plan_session` added. An accepted plan's answer gains `card`,
  what the user approves.
- `record_user_answer` asks the user a question a refusal leaves to them. A
  client that supports MCP elicitation has this server ask the user directly;
  any other sends the user's answer.
- `run_plan` runs an accepted version once the user approves it, asked through
  elicitation where the client supports it, and answers the hand-back dev's
  orchestrator receives with each frame's rows and the pair the public tools
  read it by. A run that outlasts `wait_s` answers `running`; the same call
  again waits on the same run.

Every tool is listed at version `TOOLS_VERSION`, marked `strict: false` in its
`_meta`, and reads an empty value in an optional field as absent
(`outward.drop_unset`). Sessions are keyed on the id `start_plan` returns and
bound to the caller the `x-dfo-caller` header names. `SessionKey.TRANSPORT`
keys them on the MCP transport session instead, for the control that shows why
they are not.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping
from enum import StrEnum
from typing import Any, override

from fastmcp import FastMCP
from fastmcp.server.dependencies import get_context, get_http_headers
from fastmcp.server.elicitation import (
    AcceptedElicitation,
    handle_elicit_accept,
    parse_elicit_response_type,
)
from fastmcp.tools import Tool
from fastmcp.tools.base import InputRequiredToolResult, ToolResult
from google.adk.tools import FunctionTool
from harness import plan_display
from harness.orchestrator_tools import build_orchestrator_tools
from mcp.types import (
    ElicitRequest,
    ElicitRequestFormParams,
    InputRequiredResult,
    TextContent,
    ToolAnnotations,
)
from outward import ASK_TOOL, drop_unset, guidance_text, lean, outward
from plan_tools import (
    BY_CLIENT,
    BY_USER,
    RUN_FAILED,
    SESSION_UNKNOWN,
    PlanningService,
    PlanSession,
    Run,
    _NoExecutor,
    refusal,
)
from planner.plan import Plan, load_plan
from planner.tool_results import ProblemCode
from pydantic import BaseModel, Field

CALLER_HEADER = "x-dfo-caller"
DEFAULT_WAIT_S = 30.0
TOOLS_VERSION = "2"
GUIDANCE_URI = "plan-mcp://guidance"
# Marks each tool as one a strict-mode client must not make strict.
NOT_STRICT = {"strict": False}

# What an outside model reads before the planner's own instruction.
OUTSIDE_PREAMBLE = """\
These are DFO's planner's instructions; `propose_plan` here takes `plan_session`
as well. `get_skill` and `get_skill_file` are the public DFO tools. There is no
`reject_plan`: when the user turns a plan down, propose again or stop. When an
accepted answer carries `card`, show the user the card, then call `run_plan`.

"""


class SessionKey(StrEnum):
    RETURNED = "returned"
    TRANSPORT = "transport"


class Approval(BaseModel):
    """What the user answers to a plan card asked through elicitation."""

    approve: bool = Field(description="Run this plan as shown.")
    comment: str = Field(default="", description="What to change, if you do not approve.")


def _result(answer: Mapping[str, Any]) -> ToolResult:
    text = json.dumps(answer, separators=(",", ":"), default=str)
    return ToolResult(
        content=[TextContent(type="text", text=text)], structured_content=dict(answer)
    )


def _caller() -> str | None:
    return get_http_headers(include={CALLER_HEADER}).get(CALLER_HEADER)


def card(plan: Plan) -> dict[str, Any]:
    """What the user approves: the approval card's title, summary, steps and asks
    (`harness.plan_display`, as `harness.studio_translate` builds the card)."""
    return {
        "title": plan_display.card_title(plan),
        "summary": plan_display.card_summary(plan),
        "steps": [
            {"label": row["label"], "description": row["description"]}
            for row in plan_display.card_rows(plan)
        ],
        "asks": [
            {key: row[key] for key in ("ask", "answered_by", "reason") if row.get(key)}
            for row in plan_display.ask_rows(plan)
        ],
    }


def card_text(shown: Mapping[str, Any]) -> str:
    """A card as the words an elicitation shows the user."""
    lines = [f"Run this plan? {shown['title']}"]
    if shown.get("summary"):
        lines.append(str(shown["summary"]))
    lines += [f"- {step['label']}: {step['description']}" for step in shown["steps"]]
    return "\n".join(lines)


def _elicits(enabled: bool) -> bool:
    """Whether the calling client declared MCP elicitation, and this server asks
    through it."""
    if not enabled:
        return False
    params = getattr(get_context().session, "client_params", None)
    capabilities = getattr(params, "capabilities", None)
    return getattr(capabilities, "elicitation", None) is not None


def _reply_sent(key: str) -> bool:
    """Whether this leg carries the client's reply to `key`."""
    return key in (get_context().input_responses or {})


async def _ask(key: str, message: str, response_type: Any) -> Any:
    """Ask the user `message` through elicitation: the `AcceptedElicitation` of their
    reply, `None` when they declined or cancelled, or, on an MCP 2026-07-28
    connection that has not answered yet, the `InputRequiredToolResult` this leg
    returns. The client answers it and makes the same call again with the reply
    under `key`. A handshake-era connection is asked through `ctx.elicit`."""
    ctx = get_context()
    config = parse_elicit_response_type(response_type)
    if not ctx._is_modern_protocol():
        asked = await ctx.elicit(message, response_type)
        return asked if isinstance(asked, AcceptedElicitation) else None
    reply = (ctx.input_responses or {}).get(key)
    if reply is None:
        request = ElicitRequest(
            params=ElicitRequestFormParams(message=message, requested_schema=config.schema)
        )
        return InputRequiredToolResult(InputRequiredResult(input_requests={key: request}))
    if reply.action != "accept":
        return None
    accepted = handle_elicit_accept(config, reply.content or {})
    return accepted if isinstance(accepted, AcceptedElicitation) else None


class PlanMcp:
    """The server and the sessions it reads, keyed as `key` says.

    `elicitation` lets the server ask the user directly when a client supports
    it; only a control turns it off. `guidance_in_answer` sends the guidance in
    `start_plan`'s answer; a client that already holds the planner's instruction
    leaves it out.
    """

    def __init__(
        self,
        service: PlanningService,
        *,
        key: SessionKey = SessionKey.RETURNED,
        elicitation: bool = True,
        guidance_in_answer: bool = True,
    ) -> None:
        self.service = service
        self.key = key
        self.elicitation = elicitation
        self.guidance_in_answer = guidance_in_answer
        self.transport_keys: dict[str, str] = {}
        self.mcp = FastMCP("plan-mcp")
        self.guidance = OUTSIDE_PREAMBLE + guidance_text(str(service._seam._agent.instruction))
        tools = [_StartPlan(self), _ProposePlan(self), _RecordUserAnswer(self), _RunPlan(self)]
        # Each tool's full schema, which its arguments are read against.
        self.schemas = {tool.name: tool.schema for tool in tools}
        for tool in tools:
            self.mcp.add_tool(tool)
        self.mcp.resource(
            GUIDANCE_URI, name="plan_guidance", mime_type="text/plain", description=_GUIDANCE
        )(self._guidance)
        self.mcp.prompt(name="plan_guidance", description=_GUIDANCE)(self._guidance)

    def _guidance(self) -> str:
        return self.guidance

    def find(self, plan_session: str) -> PlanSession | None:
        if self.key is SessionKey.TRANSPORT:
            plan_session = self.transport_keys.get(get_context().session_id, "")
        return self.service.session(plan_session, caller=_caller())

    async def start(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        opened = await self.service.open(
            brief=arguments["brief"],
            asks=arguments["asks"],
            spec_ids=arguments.get("spec_ids") or [],
            searches=arguments.get("searches") or [],
            caller=_caller(),
        )
        if isinstance(opened, dict):
            return opened
        if self.key is SessionKey.TRANSPORT:
            self.transport_keys[get_context().session_id] = opened.session_id
        answer = {"outcome": "ok", "plan_session": opened.session_id}
        if self.guidance_in_answer:
            answer["guidance"] = self.guidance
        answer["planner_message"] = opened.first_message
        return answer

    async def propose(self, arguments: Mapping[str, Any], *, schema: Mapping[str, Any]) -> Any:
        arguments = dict(arguments)
        session = self.find(str(arguments.pop("plan_session", "")))
        if session is None:
            return refusal(SESSION_UNKNOWN, "no plan session by that id; call start_plan")
        answer = outward(await self.service.propose(session, arguments), schema=schema)
        plan = load_plan(session.state)
        if "summary" in answer and plan is not None:
            answer["card"] = card(plan)
        return answer

    async def record_user_answer(self, arguments: Mapping[str, Any]) -> Any:
        session = self.find(str(arguments.get("plan_session", "")))
        if session is None:
            return refusal(SESSION_UNKNOWN, "no plan session by that id")
        asked = {
            key: arguments[key]
            for key in ("message", "candidates", "options", "recommended_spec_id", "rationale")
            if key in arguments
        }
        if not _elicits(self.elicitation):
            if "answer" not in arguments:
                return refusal(
                    ProblemCode.INVALID_ARGUMENTS.value,
                    "send `answer`: what the user said when you asked them",
                )
            answered = await self.service.answer(session, asked, arguments["answer"], by=BY_CLIENT)
            return outward(answered)
        choices = {
            str(choice.get("id") or choice.get("spec_id")): str(choice.get("label") or "")
            for choice in [*asked.get("options", []), *asked.get("candidates", [])]
        }
        if not _reply_sent("answer"):
            checked = await self.service.check_ask(session, asked)
            if checked is not None:
                return outward(checked)
        reply = await _ask(
            "answer",
            str(asked.get("message", "")),
            {key: {"title": label or key} for key, label in choices.items()} if choices else str,
        )
        if isinstance(reply, InputRequiredToolResult):
            return reply
        answer: dict[str, Any] = {"kind": "skipped"}
        if reply is not None:
            answer = (
                {"kind": "option", "spec_id": reply.data}
                if choices
                else {"kind": "text", "text": str(reply.data or "")}
            )
        return outward(await self.service.answer(session, asked, answer, by=BY_USER))

    async def run_plan(self, arguments: Mapping[str, Any]) -> Any:
        session = self.find(str(arguments.get("plan_session", "")))
        if session is None:
            return refusal(SESSION_UNKNOWN, "no plan session by that id")
        version = int(arguments["version"])
        wait_s = float(arguments.get("wait_s", DEFAULT_WAIT_S))
        found = self.service.runnable(session, version)
        if isinstance(found, dict):
            return found
        if isinstance(found, Plan) and not (session.approvals.get(version) or {}).get("approved"):
            if _elicits(self.elicitation):
                reply = await _ask("approval", card_text(card(found)), Approval)
                if isinstance(reply, InputRequiredToolResult):
                    return reply
                given = reply.data if reply is not None else None
                self.service.approve(
                    session,
                    version,
                    approved=bool(given and given.approve),
                    by=BY_USER,
                    comment=given.comment if given else "",
                )
            elif arguments.get("approved") is True:
                self.service.approve(session, version, approved=True, by=BY_CLIENT)
        started = self.service.start_run(session, version)
        if isinstance(started, dict):
            return started
        answer = await _wait(started, wait_s)
        if answer.get("outcome") == "ok":
            answer["approved_by"] = session.approvals[version]["by"]
        return answer


async def _wait(run: Run, wait_s: float) -> dict[str, Any]:
    if run.task is not None and not run.task.done():
        await asyncio.wait({run.task}, timeout=max(wait_s, 0.0))
    if run.result is None:
        return {"outcome": "ok", "status": "running", "run_id": run.run_id, "version": run.version}
    if run.status == RUN_FAILED:
        return dict(run.result)
    return {"outcome": "ok", "run_id": run.run_id, "version": run.version, **run.result}


def _declaration(tool: Any) -> Any:
    declaration = tool._get_declaration()
    assert declaration is not None
    return declaration


_GUIDANCE = "How to draft a DFO plan with these tools: the planner's own instructions."
_PLAN_SESSION = {"type": "string", "description": "The id `start_plan` returned."}
_SPEC_IDS = {
    "type": "array",
    "items": {"type": "string"},
    "description": "The spec_ids your `search` calls returned that the plan may use.",
}
_SEARCHES = {
    "type": "array",
    "description": (
        "Searches to run here instead of spec_ids: each `query` and the `parts` you sent "
        "to `search`."
    ),
    "items": {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "parts": {"type": "array", "items": {"type": "object"}},
        },
        "required": ["query", "parts"],
    },
}


class _PlanTool(Tool):
    """A plan tool: listed lean (`outward.lean`) at `TOOLS_VERSION` and not strict,
    its arguments read against the full schema with unset values left out."""

    def __init__(
        self, server: PlanMcp, *, name: str, description: str, schema: dict[str, Any]
    ) -> None:
        super().__init__(
            name=name,
            description=description,
            parameters=lean(schema),
            annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False),
            meta=NOT_STRICT,
            version=TOOLS_VERSION,
        )
        self._server = server
        self._schema = schema

    @property
    def schema(self) -> dict[str, Any]:
        return self._schema

    async def answer(self, arguments: dict[str, Any]) -> Any:
        raise NotImplementedError

    @override
    async def run(self, arguments: dict[str, Any]) -> ToolResult:
        answered = await self.answer(drop_unset(arguments, self._schema))
        if isinstance(answered, ToolResult):
            return answered
        return _result(answered)


class _StartPlan(_PlanTool):
    """`start_plan`: the orchestrator's `plan` tool's brief and asks, and the sources."""

    def __init__(self, server: PlanMcp) -> None:
        tools = {
            tool.__name__: tool
            for tool in build_orchestrator_tools(resolver=_no_search, executor=_NoExecutor())
        }
        plan = dict(_declaration(FunctionTool(tools["plan"])).parameters_json_schema)
        properties = plan["properties"]
        schema = {
            **plan,
            "properties": {
                "brief": properties["brief"],
                "asks": properties["asks"],
                "spec_ids": _SPEC_IDS,
                "searches": _SEARCHES,
            },
        }
        super().__init__(
            server,
            name="start_plan",
            description=(
                "Open a plan session for the user's question after you searched DFO. Pass "
                "the spec_ids your searches returned. Answers `plan_session`, the planning "
                "guidance (also the resource " + GUIDANCE_URI + ") and the planner's first "
                "message, which lists the sources the plan may use."
            ),
            schema=schema,
        )

    async def answer(self, arguments: dict[str, Any]) -> Any:
        return await self._server.start(arguments)


class _ProposePlan(_PlanTool):
    """`propose_plan`: the served planner's declaration with `plan_session` added."""

    def __init__(self, server: PlanMcp) -> None:
        declaration = _declaration(server.service.propose_tool)
        schema = dict(declaration.parameters_json_schema)
        schema["properties"] = {"plan_session": _PLAN_SESSION, **schema["properties"]}
        schema["required"] = ["plan_session", *schema.get("required", [])]
        super().__init__(
            server,
            name="propose_plan",
            description=str(declaration.description),
            schema=schema,
        )

    async def answer(self, arguments: dict[str, Any]) -> Any:
        return await self._server.propose(arguments, schema=self._schema)


class _RecordUserAnswer(_PlanTool):
    def __init__(self, server: PlanMcp) -> None:
        declaration = _declaration(server.service.ask_tool)
        ask = dict(declaration.parameters_json_schema)
        schema = {
            **ask,
            "properties": {
                "plan_session": _PLAN_SESSION,
                **ask["properties"],
                "answer": {
                    "type": "object",
                    "description": (
                        "What the user said, when this server could not ask them itself: "
                        '`{"kind": "option", "spec_id": <option id or candidate spec_id>}`, '
                        '`{"kind": "text", "text": ...}` or `{"kind": "skipped"}`.'
                    ),
                },
            },
            "required": ["plan_session", *ask.get("required", [])],
        }
        super().__init__(
            server,
            name=ASK_TOOL,
            description=(
                "Ask the user what a refusal's `action` leaves to them, offering exactly its "
                "`options` or the referral's `candidates`, then propose again. When your host "
                "supports elicitation the user is asked directly; otherwise ask them yourself "
                "and send their reply as `answer`."
            ),
            schema=schema,
        )

    async def answer(self, arguments: dict[str, Any]) -> Any:
        return await self._server.record_user_answer(arguments)


class _RunPlan(_PlanTool):
    def __init__(self, server: PlanMcp) -> None:
        schema = {
            "type": "object",
            "properties": {
                "plan_session": _PLAN_SESSION,
                "version": {"type": "integer", "description": "The accepted `version`."},
                "approved": {
                    "type": "boolean",
                    "description": (
                        "True once the user approved the card, when your host cannot show it "
                        "to them through elicitation."
                    ),
                },
                "wait_s": {
                    "type": "number",
                    "minimum": 0,
                    "description": "Seconds to wait before answering `running` (default 30).",
                },
            },
            "required": ["plan_session", "version"],
        }
        super().__init__(
            server,
            name="run_plan",
            description=(
                "Run an accepted version the user approved, and answer its result: each "
                "frame's rows when small, and its `frame_ref` for describe_frame, export_data "
                "and the analysis tools. A run past `wait_s` answers `running`; call again "
                "with the same version. Only the current version runs."
            ),
            schema=schema,
        )

    async def answer(self, arguments: dict[str, Any]) -> Any:
        return await self._server.run_plan(arguments)


async def _no_search(**kwargs: Any) -> dict[str, Any]:
    raise AssertionError("the declaration's tools run no search")


__all__ = ["PlanMcp", "SessionKey", "card"]
