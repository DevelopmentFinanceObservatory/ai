"""The studio, scripted: real workflow, real planner seam, no model.

This module builds the studio `harness.server` serves and replaces only what
a model or an upstream would answer:

- the orchestrator agent's model answers from a script of tool calls and
  text, one entry per model call, across every turn of a case;
- the planner is the real `harness.planner_seam.PlannerSeam`. Its model
  either proposes the canned plan `harness.scripted_plan` builds, through a
  `propose_plan` that declares the plan on record ready
  (`harness.scripted.use_compiled_proposals`), so the seam renders its real
  first message and the workflow's approval, execution and hand-back nodes
  run as served, or, when the case scripts it (`planner`),
  answers from that script and drives the real `propose_plan`, whose
  translation discovers through the case's canned `translator_search`
  results and compiles against the real servers' tool schemas;
- search answers from the case's canned envelopes, run through the digest the
  served resolver builds (`harness.tool_contracts.resolver_digest`);
- the tool surface and executor are `harness.scripted`'s, wrapped to log
  every call and to refuse a retrieval the case names.

Every model call is recorded as a plain dict (`log`), next to the calls the
plan's steps make, so `checks.py`'s request checks read what each model was
shown without any model having run. The studio is driven over its own HTTP
routes (`create_studio_router`), the surface a browser uses.
"""

from __future__ import annotations

import copy
from collections.abc import AsyncGenerator, AsyncIterator, Callable, Mapping
from contextlib import asynccontextmanager
from typing import Any

import httpx2
from analysis_mcp.server import mcp as analysis_mcp
from compiler.surface import McpSurface, ToolCallResult, ToolSchema, ToolSurfaceUnavailableError
from data_mcp.digest import render_answer
from data_mcp.search_result import SEARCH_ENVELOPE_META_KEY
from data_mcp.server import mcp as data_mcp
from executor.ledger import InMemoryRunLedger
from executor.run import no_label_refs
from fastapi import FastAPI
from google.adk import Context
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_response import LlmResponse
from google.adk.sessions import InMemorySessionService
from google.adk.tools.base_tool import BaseTool
from google.genai import types
from harness.graph.workflow import create_root_workflow
from harness.orchestrator_tools import build_orchestrator_tools
from harness.planner_seam import PlannerSeam
from harness.scripted import (
    SCRIPTED_SPEC,
    ScriptedExecutor,
    ScriptedSurface,
    scripted_publish_result,
    use_compiled_proposals,
)
from harness.scripted_plan import scripted_fetch_series_stats, scripted_plan
from harness.settings import Settings
from harness.studio import StudioRunner, create_studio_router
from harness.tool_contracts import ResolverError, resolver_digest
from harness.wiring import handing_back_planner_failures
from orchestrator.agent import create_orchestrator_agent
from planner.plan import Plan, SessionState, save_plan
from pydantic import Field

# What a model says once its script runs out. The run records the exhaustion
# (`Log.exhausted`) and reports the case as under-scripted, so a turn nothing
# drove cannot pass.
_EXHAUSTED_TEXT = "done"
# The bearer token sent when `Settings` names none: the studio accepts any non-empty one.
_TOKEN = "orchestrator-eval"


class Log:
    """What a case recorded, in order: one dict per model request and per tool call.

    A request entry is `{"kind": "request", "agent": ..., "system": str,
    "contents": [Content as JSON]}`; a call entry is `{"kind": "call",
    "layer": "step" | "executor", "tool": ..., "arguments": ..., "outcome":
    "ok" | "refused", "code": ...}`, and one from a scripted planner's own tool
    call carries `"layer": "planner"`, the tool's own `outcome` and its whole
    `response`; a `{"kind": "turn", "index": n}` marker opens each message
    turn.
    """

    def __init__(self) -> None:
        self.entries: list[dict[str, Any]] = []
        self.exhausted: list[str] = []

    def mark_turn(self, index: int) -> None:
        self.entries.append({"kind": "turn", "index": index})

    def record_request(self, agent: str, request: Any) -> None:
        instruction = request.config.system_instruction if request.config else None
        self.entries.append(
            {
                "kind": "request",
                "agent": agent,
                "system": instruction if isinstance(instruction, str) else "",
                "contents": [
                    content.model_dump(mode="json", exclude_none=True)
                    for content in request.contents
                ],
            }
        )


def _part_from_step(step: Mapping[str, Any]) -> types.Part:
    """One script entry: `{"call": tool, "args": {...}}` or `{"text": ...}`."""
    if "call" in step:
        return types.Part.from_function_call(name=step["call"], args=dict(step.get("args") or {}))
    return types.Part.from_text(text=step["text"])


class OrchestratorLlm(BaseLlm):
    """Answers each orchestrator model call with the next scripted part."""

    model: str = "orchestrator-script"
    script: list[types.Part] = Field(default_factory=list)
    log: Any = None

    async def generate_content_async(
        self, llm_request: Any, stream: bool = False
    ) -> AsyncGenerator[LlmResponse]:
        del stream
        self.log.record_request("orchestrator", llm_request)
        if self.script:
            part = self.script.pop(0)
        else:
            self.log.exhausted.append("orchestrator")
            part = types.Part.from_text(text=_EXHAUSTED_TEXT)
        yield LlmResponse(content=types.Content(role="model", parts=[part]))


class PlannerLlm(BaseLlm):
    """The planner's model: it proposes the plan it was handed, once per entry.

    The plan is written into the session's planner state the way a translated
    `propose_plan` leaves a compiled plan there, then a `propose_plan` call
    declares it ready (`harness.scripted.use_compiled_proposals`). Every other
    generation closes the run.
    """

    model: str = "planner-script"
    log: Any = None
    state: Any = None
    offered: Any = None

    def offer(self, state: SessionState, plan: Plan) -> None:
        self.state = state
        self.offered = plan

    async def generate_content_async(
        self, llm_request: Any, stream: bool = False
    ) -> AsyncGenerator[LlmResponse]:
        del stream
        self.log.record_request("planner", llm_request)
        plan, self.offered = self.offered, None
        if plan is None:
            part = types.Part.from_text(text=_EXHAUSTED_TEXT)
        else:
            save_plan(self.state, plan)
            part = types.Part.from_function_call(name="propose_plan", args={})
        yield LlmResponse(content=types.Content(role="model", parts=[part]))


class ScriptedPlannerLlm(BaseLlm):
    """The planner's model from the case's `planner` script, one part per model call."""

    model: str = "planner-script"
    script: list[types.Part] = Field(default_factory=list)
    log: Any = None

    async def generate_content_async(
        self, llm_request: Any, stream: bool = False
    ) -> AsyncGenerator[LlmResponse]:
        del stream
        self.log.record_request("planner", llm_request)
        if self.script:
            part = self.script.pop(0)
        else:
            self.log.exhausted.append("planner")
            part = types.Part.from_text(text=_EXHAUSTED_TEXT)
        yield LlmResponse(content=types.Content(role="model", parts=[part]))


def planner_tool_log(
    log: Log,
) -> Callable[[BaseTool, dict[str, Any], Context, dict[str, Any]], None]:
    """An `after_tool_callback` recording each planner tool call and its response in `log`."""

    def record(
        tool: BaseTool, args: dict[str, Any], tool_context: Context, tool_response: dict[str, Any]
    ) -> None:
        del tool_context
        response = tool_response if isinstance(tool_response, dict) else {}
        log.entries.append(
            {
                "kind": "call",
                "layer": "planner",
                "tool": tool.name,
                "arguments": copy.deepcopy(dict(args)),
                "outcome": response.get("outcome"),
                "code": response.get("code"),
                "response": copy.deepcopy(response),
            }
        )

    return record


class ScriptedPairPlanner:
    """`harness.graph.seams.PlannerModel` over the real `PlannerSeam` and a `PlannerLlm`.

    A first draft offers `scripted_plan`; an amendment or a replan offers the
    next version of the plan the graph holds, as `harness.scripted.
    ScriptedPlannerModel` does.
    """

    def __init__(self, *, seam: PlannerSeam, llm: PlannerLlm, spec_id: str) -> None:
        self._seam = seam
        self._llm = llm
        self._spec_id = spec_id

    async def __call__(
        self, *, ctx: Any, question: str, prior_plan: dict[str, Any] | None = None, **kwargs: Any
    ) -> dict[str, Any]:
        plan = scripted_plan(question, spec_id=self._spec_id)
        if prior_plan is not None:
            prior_version = int(prior_plan.get("version", 1))
            plan = plan.model_copy(
                update={"version": prior_version + 1, "previous_version": prior_version}
            )
        self._llm.offer(ctx.session.state, plan)
        return await self._seam(ctx=ctx, question=question, prior_plan=prior_plan, **kwargs)


def search_envelope(result: Mapping[str, Any]) -> dict[str, Any]:
    """A data-mcp search envelope for a case's canned search result.

    `result["parts"]` holds one entry per part: `variable`, `status`
    (`definite`, `candidates`, `no_data` or `unsupported`), `warnings`
    (`[{"code", "message"}]`), and for a definite part its `spec_id`,
    `sentence`, `coverage` (`[first_year, last_year]`) and `entities` (labels,
    Kenya when unset). A candidates part lists its `candidates`, each shaped
    as a definite part.
    """
    parts = []
    for part in result["parts"]:
        built: dict[str, Any] = {
            "status": part["status"],
            "variable_text": part["variable"],
            "warnings": [
                {"severity": "warning", **warning} for warning in part.get("warnings") or []
            ],
        }
        if part["status"] == "definite":
            built["spec"] = _spec_block(part)
        elif part["status"] == "candidates":
            built["candidates"] = {
                "specs": [_spec_block(candidate) for candidate in part["candidates"]],
                "truncated": False,
            }
        parts.append(built)
    return {"schema_version": 1, "kind": "resolution", "status": parts[0]["status"], "parts": parts}


def _spec_block(part: Mapping[str, Any]) -> dict[str, Any]:
    first, last = part.get("coverage", [2010, 2024])
    return {
        "spec_id": part["spec_id"],
        "resolved": {
            "rendered_sentence": part["sentence"],
            "confidence": "high",
            "coverage": {
                "kind": "exact",
                "has_data": True,
                "window": {"start_year": first, "end_year": last},
            },
            "stat_vars": {
                "member_count": 1,
                "member_count_exact": True,
                "served_count": 1,
                "data_date_range": {"start": str(first), "end": str(last)},
            },
            "entities": [
                {"role": "observationAbout", "label": label, "ref": f"country/{label[:3].upper()}"}
                for label in part.get("entities", ["Kenya"])
            ],
            "facets": {},
        },
        "definition": {
            "schema_version": 1,
            "spec_id": part["spec_id"],
            "identity": {"stat_var": "LifeExpectancy_Person"},
        },
    }


class CannedResolver:
    """`harness.orchestrator_tools.Resolver` answering from the case's search results, in order.

    Builds the digest from each envelope with `resolver_digest`, as
    `harness.mcp_client.McpResolver` does.
    """

    def __init__(self, results: list[Mapping[str, Any]]) -> None:
        self._results = list(results)
        self.questions: list[str] = []

    async def __call__(self, *, question: str, **kwargs: Any) -> dict[str, Any]:
        del kwargs
        self.questions.append(question)
        if not self._results:
            raise ResolverError("the case scripted no further search result")
        return resolver_digest(search_envelope(self._results.pop(0)))


class LoggingExecutor(ScriptedExecutor):
    """`ScriptedExecutor` that logs each call."""

    def __init__(self, *, log: Log) -> None:
        self._log = log

    async def __call__(self, *, tool: str, args: dict[str, Any], on_progress: Any = None) -> Any:
        result = await super().__call__(tool=tool, args=args, on_progress=on_progress)
        refusal = result.get("refusal") or {}
        self._log.entries.append(
            {
                "kind": "call",
                "layer": "executor",
                "tool": tool,
                "arguments": copy.deepcopy(args),
                "outcome": "ok" if result.get("outcome") == "ok" else "refused",
                "code": refusal.get("code"),
            }
        )
        return result


class LoggingSurface(ScriptedSurface):
    """`ScriptedSurface` that logs each plan-step call and refuses `get_data`
    for the spec ids a case names with that refusal's `code` and `retry_mode`.

    With `real_schemas`, it declares the real servers' tool schemas, the ones
    a plan the translator compiled is pinned to."""

    def __init__(
        self, *, log: Log, refusals: Mapping[str, Mapping[str, str]], real_schemas: bool = False
    ) -> None:
        super().__init__()
        self._log = log
        self._refusals = refusals
        self._real_schemas = real_schemas

    async def list_tools(self) -> list[ToolSchema]:
        return await _real_schemas() if self._real_schemas else await super().list_tools()

    async def call_tool(
        self, *, tool: Any, arguments: Mapping[str, Any], on_progress: Any = None
    ) -> ToolCallResult:
        refusal = (
            self._refusals.get(str(arguments.get("spec_id"))) if tool.name == "get_data" else None
        )
        if refusal is not None:
            body = {"fix": {}, "message": "the case's scripted refusal", **refusal}
            result = ToolCallResult(is_error=True, structured_content=body, text=body["message"])
        else:
            result = await super().call_tool(
                tool=tool, arguments=arguments, on_progress=on_progress
            )
        self._log.entries.append(
            {
                "kind": "call",
                "layer": "step",
                "tool": tool.name,
                "arguments": copy.deepcopy(dict(arguments)),
                "outcome": "refused" if result.is_error else "ok",
                "code": (result.structured_content or {}).get("code") if result.is_error else None,
            }
        )
        return result


class CannedDataSurface(ScriptedSurface):
    """The tool surface the translator discovers and compiles on, for a scripted planner.

    `list_tools` answers the real data-mcp and analysis-mcp schemas, which the
    capability table is pinned to, as the case's `LoggingSurface` does, so
    the compiled workflow runs on the scripted executor.
    `search` answers the case's `translator_search` results in order, each
    logged as a `discovery` call; `describe` refuses, which leaves a source
    undescribed. Everything else is `ScriptedSurface`'s.
    """

    def __init__(self, *, log: Log, searches: list[Mapping[str, Any]]) -> None:
        super().__init__()
        self._log = log
        self._searches = searches

    async def list_tools(self) -> list[ToolSchema]:
        return await _real_schemas()

    async def call_tool(
        self, *, tool: Any, arguments: Mapping[str, Any], on_progress: Any = None
    ) -> ToolCallResult:
        if tool.name == "describe":
            return ToolCallResult(is_error=True, structured_content={"code": "unavailable"})
        if tool.name != "search":
            return await super().call_tool(tool=tool, arguments=arguments, on_progress=on_progress)
        if not self._searches:
            raise ToolSurfaceUnavailableError("the case scripted no further translator search")
        envelope = search_envelope(self._searches.pop(0))
        self._log.entries.append(
            {
                "kind": "call",
                "layer": "discovery",
                "tool": "search",
                "arguments": copy.deepcopy(dict(arguments)),
                "outcome": "ok",
                "code": None,
            }
        )
        return ToolCallResult(
            is_error=False,
            structured_content=render_answer(envelope),
            meta={SEARCH_ENVELOPE_META_KEY: envelope},
        )


_REAL_SCHEMAS: list[ToolSchema] = []


async def _real_schemas() -> list[ToolSchema]:
    """The real servers' tool schemas, read once from the in-process servers."""
    if not _REAL_SCHEMAS:
        _REAL_SCHEMAS.extend(
            await McpSurface(
                sources={"data-mcp": data_mcp, "analysis-mcp": analysis_mcp}
            ).list_tools()
        )
    return _REAL_SCHEMAS


def build_studio(
    scenario: Mapping[str, Any], *, log: Log
) -> tuple[FastAPI, OrchestratorLlm, CannedResolver]:
    """The studio for one case: the app, its model and its resolver."""
    settings = Settings()
    sessions = InMemorySessionService()

    orchestrator_llm = OrchestratorLlm(
        script=[_part_from_step(step) for step in scenario["orchestrator"]], log=log
    )
    resolver = CannedResolver(copy.deepcopy(scenario.get("search") or []))
    planner_script = scenario.get("planner")
    surface = LoggingSurface(
        log=log,
        refusals=scenario.get("get_data_refusals") or {},
        real_schemas=planner_script is not None,
    )
    if planner_script is None:
        planner_llm = PlannerLlm(log=log)
        seam = PlannerSeam(
            settings=settings,
            surface=surface,
        )
        seam._agent.model = planner_llm
        use_compiled_proposals(seam._agent)
        planner = handing_back_planner_failures(
            ScriptedPairPlanner(
                seam=seam, llm=planner_llm, spec_id=scenario.get("plan_spec_id", SCRIPTED_SPEC)
            )
        )
    else:
        seam = PlannerSeam(
            settings=settings,
            surface=CannedDataSurface(
                log=log, searches=copy.deepcopy(scenario.get("translator_search") or [])
            ),
        )
        seam._agent.model = ScriptedPlannerLlm(
            script=[_part_from_step(step) for step in planner_script], log=log
        )
        seam._agent.after_tool_callback = [seam._after_planner_tool, planner_tool_log(log)]
        planner = handing_back_planner_failures(seam)

    def create_workflow(recorder: Any, surface_recorder: Any) -> Any:
        agent = create_orchestrator_agent(
            model_name="unused",
            tools=build_orchestrator_tools(
                resolver=resolver, executor=recorder, surface=surface_recorder
            ),
        ).model_copy(update={"model": orchestrator_llm})
        return create_root_workflow(
            planner=planner,
            orchestrator=agent,
            executor=recorder,
            tool_surface=surface_recorder,
            definition_surface=surface,
            publisher=scripted_publish_result,
            run_ledger=InMemoryRunLedger(),
            max_concurrent_steps=settings.executor_max_concurrent_steps,
            fetch_series_stats=scripted_fetch_series_stats,
            label_refs=no_label_refs,
        )

    studio = StudioRunner(
        create_workflow=create_workflow,
        executor=LoggingExecutor(log=log),
        session_service=sessions,
        tool_surface=surface,
    )
    app = FastAPI()
    app.state.settings = settings
    app.include_router(create_studio_router(studio, settings=settings))
    return app, orchestrator_llm, resolver


@asynccontextmanager
async def orchestrator_client(
    scenario: Mapping[str, Any],
) -> AsyncIterator[tuple[httpx2.AsyncClient, Log]]:
    """An in-process client on the studio for `scenario`, and its log."""
    log = Log()
    app, _llm, _resolver = build_studio(scenario, log=log)
    token = app.state.settings.studio_api_token or _TOKEN
    transport = httpx2.ASGITransport(app=app)
    async with httpx2.AsyncClient(
        transport=transport,
        base_url="http://studio-eval.local",
        timeout=30.0,
        headers={"Authorization": f"Bearer {token}"},
    ) as client:
        yield client, log


__all__ = ["Log", "build_studio", "orchestrator_client", "search_envelope"]
