"""`plan-mcp`: four tools that let an outside model plan and run over DFO's planner code.

- `start_plan` opens a plan session from the searches the client ran, its brief
  and its asks, and answers the session id, the planner's guidance and the
  planner's first message.
- `propose_plan` is the served planner's own tool: its declaration, its
  arguments and its refusals, with `plan_session` added. An accepted plan's
  answer gains `card`, what the user approves.
- `record_user_answer` records a question the client asked its user and their
  answer, where a refusal's `action` names `adk_request_input`.
- `run_plan` runs an accepted version once the user approves it, and answers
  the hand-back dev's orchestrator receives. A run that outlasts `wait_s`
  answers `running`; the same call again waits on the same run.

Sessions are keyed on the id `start_plan` returns and bound to the caller the
`x-dfo-caller` header names. `SessionKey.TRANSPORT` keys them on the MCP
transport session instead, for the control that shows why they are not.
"""

from __future__ import annotations

import asyncio
import json
from enum import StrEnum
from typing import Any, override

from fastmcp import FastMCP
from fastmcp.server.dependencies import get_context, get_http_headers
from fastmcp.tools import Tool
from fastmcp.tools.base import ToolResult
from google.adk.tools import FunctionTool
from harness import plan_display
from harness.orchestrator_tools import build_orchestrator_tools
from mcp.types import TextContent, ToolAnnotations
from plan_tools import (
    RUN_FAILED,
    SESSION_UNKNOWN,
    PlanningService,
    PlanSession,
    Run,
    _NoExecutor,
    refusal,
)
from planner.plan import Plan, load_plan

CALLER_HEADER = "x-dfo-caller"
DEFAULT_WAIT_S = 30.0

# What an outside model reads before the planner's own instruction: how the
# instruction's tool names map onto this server's.
OUTSIDE_PREAMBLE = """\
You are drafting a DFO plan through the plan-mcp tools. The guidance below is
written for DFO's own planner; read its tool names this way. `propose_plan`
is this server's `propose_plan`: pass `plan_session` with every call.
`get_skill` and `get_skill_file` are the public DFO tools of the same names.
Where the guidance or a refusal's `action` names `adk_request_input`, ask
your user in the conversation, offering exactly the refusal's `options` or
the referral's `candidates`, then call `record_user_answer` with what you
asked and what they answered, and propose again. When an accepted answer
carries `card`, show the user the card and run it with `run_plan` only once
they approve it.

"""


class SessionKey(StrEnum):
    RETURNED = "returned"
    TRANSPORT = "transport"


def _result(answer: dict[str, Any]) -> ToolResult:
    text = json.dumps(answer, separators=(",", ":"), default=str)
    return ToolResult(content=[TextContent(type="text", text=text)], structured_content=answer)


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


class PlanMcp:
    """The server and the sessions it reads, keyed as `key` says."""

    def __init__(self, service: PlanningService, *, key: SessionKey = SessionKey.RETURNED) -> None:
        self.service = service
        self.key = key
        self.transport_keys: dict[str, str] = {}
        self.mcp = FastMCP("plan-mcp")
        self.guidance = OUTSIDE_PREAMBLE + str(service._seam._agent.instruction)
        self.mcp.add_tool(_StartPlan(self))
        self.mcp.add_tool(_ProposePlan(self))
        self.mcp.tool(
            name="record_user_answer",
            annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False),
        )(self.record_user_answer)
        self.mcp.tool(
            name="run_plan",
            annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False),
        )(self.run_plan)
        self.mcp.prompt(name="plan_guidance", description="How to draft a DFO plan.")(
            self._guidance
        )

    def _guidance(self) -> str:
        """The planning guidance `start_plan` answers, for hosts that offer prompts."""
        return self.guidance

    def find(self, plan_session: str) -> PlanSession | None:
        if self.key is SessionKey.TRANSPORT:
            plan_session = self.transport_keys.get(get_context().session_id, "")
        return self.service.session(plan_session, caller=_caller())

    async def start(self, arguments: dict[str, Any]) -> dict[str, Any]:
        opened = await self.service.open(
            brief=arguments["brief"],
            asks=arguments["asks"],
            searches=arguments.get("searches") or [],
            caller=_caller(),
        )
        if isinstance(opened, dict):
            return opened
        if self.key is SessionKey.TRANSPORT:
            self.transport_keys[get_context().session_id] = opened.session_id
        return {
            "outcome": "ok",
            "plan_session": opened.session_id,
            "guidance": self.guidance,
            "planner_message": opened.first_message,
        }

    async def propose(self, arguments: dict[str, Any]) -> dict[str, Any]:
        arguments = dict(arguments)
        session = self.find(str(arguments.pop("plan_session", "")))
        if session is None:
            return refusal(SESSION_UNKNOWN, "no plan session by that id; call start_plan")
        answer = await self.service.propose(session, arguments)
        plan = load_plan(session.state)
        if "summary" in answer and plan is not None:
            answer = {**answer, "card": card(plan)}
        return answer

    async def record_user_answer(
        self,
        plan_session: str,
        message: str,
        answer: dict[str, Any],
        candidates: list[dict[str, str]] | None = None,
        options: list[dict[str, str]] | None = None,
        recommended_spec_id: str | None = None,
        rationale: str | None = None,
    ) -> ToolResult:
        """Record a question you asked your user and their answer, where a refusal's
        `action` names `adk_request_input`.

        Args:
            plan_session: The id `start_plan` returned.
            message: The question you asked, in plain words.
            answer: What the user said: `{"kind": "option", "spec_id": ...}` for a
                pick of one of the options or candidates you offered, by its `id`
                or `spec_id`, `{"kind": "text", "text": ...}` for their own words,
                or `{"kind": "skipped"}`.
            candidates: The referral's candidates you offered, each
                `{"spec_id", "label"}`. Omit with `options`.
            options: The `options` the refusal's action carries, unchanged.
            recommended_spec_id: The candidate you recommended, if any.
            rationale: Why you recommended it.
        """
        session = self.find(plan_session)
        if session is None:
            return _result(refusal(SESSION_UNKNOWN, "no plan session by that id"))
        asked: dict[str, Any] = {"message": message}
        if candidates:
            asked["candidates"] = candidates
        if options:
            asked["options"] = options
        if recommended_spec_id:
            asked["recommended_spec_id"] = recommended_spec_id
        if rationale:
            asked["rationale"] = rationale
        return _result(await self.service.answer(session, asked, answer))

    async def run_plan(
        self, plan_session: str, version: int, wait_s: float = DEFAULT_WAIT_S
    ) -> ToolResult:
        """Run an accepted plan version the user approved, and answer what it found.

        Answers `finished` with `hand_back`: the frames, step summaries, totals,
        each ask's status and the plan record. A run still going after `wait_s`
        seconds answers `running`; call again with the same version to wait
        on the same run. Only the current version runs: an amended plan
        supersedes the version it amends.

        Args:
            plan_session: The id `start_plan` returned.
            version: The `version` the accepted `propose_plan` answer carried.
            wait_s: Seconds to wait for the run before answering `running`.
        """
        session = self.find(plan_session)
        if session is None:
            return _result(refusal(SESSION_UNKNOWN, "no plan session by that id"))
        started = self.service.start_run(session, version)
        if isinstance(started, dict):
            return _result(started)
        return _result(await _wait(started, wait_s))


async def _wait(run: Run, wait_s: float) -> dict[str, Any]:
    done, _ = await asyncio.wait({run.task}, timeout=max(wait_s, 0.0))
    if not done:
        return {"outcome": "ok", "status": "running", "run_id": run.run_id, "version": run.version}
    if run.task.exception() is not None:
        return refusal(RUN_FAILED, f"the run failed: {run.task.exception()!r}")
    return {"outcome": "ok", "run_id": run.run_id, "version": run.version, **run.task.result()}


def _declaration(tool: Any) -> Any:
    declaration = tool._get_declaration()
    assert declaration is not None
    return declaration


_PLAN_SESSION = {
    "type": "string",
    "description": "The id `start_plan` returned. Every call of one plan names it.",
}
_SEARCHES = {
    "type": "array",
    "description": (
        "Each `search` call whose results the plan may use, in the order you made "
        "them: `query`, the user's question in their words, and `parts`, the parts "
        "you sent to `search`, unchanged. The searches run again here, and the plan "
        "may pin only the datasets they return."
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


class _StartPlan(Tool):
    """`start_plan`: the orchestrator's `plan` tool's brief and asks, and the searches."""

    def __init__(self, server: PlanMcp) -> None:
        tools = {
            tool.__name__: tool
            for tool in build_orchestrator_tools(resolver=_no_search, executor=_NoExecutor())
        }
        declaration = _declaration(FunctionTool(tools["plan"]))
        schema = dict(declaration.parameters_json_schema)
        schema["properties"] = {**schema["properties"], "searches": _SEARCHES}
        schema["required"] = [*schema.get("required", []), "searches"]
        super().__init__(
            name="start_plan",
            description=(
                "Open a plan session for a question you searched DFO for, and read "
                "the planning guidance before your first proposal.\n\n"
                + str(declaration.description)
            ),
            parameters=schema,
            annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False),
        )
        self._server = server

    @override
    async def run(self, arguments: dict[str, Any]) -> ToolResult:
        return _result(await self._server.start(arguments))


class _ProposePlan(Tool):
    """`propose_plan`: the served planner's declaration with `plan_session` added."""

    def __init__(self, server: PlanMcp) -> None:
        declaration = _declaration(server.service.propose_tool)
        schema = dict(declaration.parameters_json_schema)
        schema["properties"] = {"plan_session": _PLAN_SESSION, **schema["properties"]}
        schema["required"] = ["plan_session", *schema.get("required", [])]
        super().__init__(
            name="propose_plan",
            description=str(declaration.description),
            parameters=schema,
            annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False),
        )
        self._server = server

    @override
    async def run(self, arguments: dict[str, Any]) -> ToolResult:
        return _result(await self._server.propose(arguments))


async def _no_search(**kwargs: Any) -> dict[str, Any]:
    raise AssertionError("the declaration's tools run no search")
