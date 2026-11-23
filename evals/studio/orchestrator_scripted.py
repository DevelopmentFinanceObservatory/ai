"""The studio, scripted: real turn graph, real agents, no model.

This module builds the studio `harness.server` serves and replaces only what
a model or an upstream would answer:

- the orchestrator agent is the real Pydantic AI agent
  (`orchestrator.agent.orchestrator_agent`) on its real model
  (`orchestrator.model.orchestrator_model`), whose google-genai client
  (`harness.scripted.scripted_client`) answers from a script of tool calls
  and text, one entry per model call, across every turn of a case;
- the planner is the real `harness.planner_adapter.PlannerAdapter`, its model
  calls answered by a scripted google-genai client. Its model either proposes
  the canned plan `harness.scripted_plan` builds, through a `propose_plan`
  that declares the plan on record ready
  (`harness.scripted.use_compiled_proposals`), so the adapter renders its real
  first message and the graph's approval, execution and hand-back nodes run
  as served. When the case scripts the planner (`planner`), it answers from
  that script and drives the real `propose_plan`, whose translation
  discovers through the case's canned `translator_search` results and
  compiles against the real servers' tool schemas;
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
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
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
from google.genai import types
from harness.conversation import current_conversation_id
from harness.graph.events import GraphEvent
from harness.graph.workflow import studio_graph
from harness.orchestrator_tools import build_orchestrator_tools
from harness.planner_adapter import PlannerAdapter, content_on_spans
from harness.scripted import (
    SCRIPTED_SPEC,
    ModelRequest,
    ScriptedExecutor,
    ScriptedSurface,
    scripted_client,
    scripted_publish_result,
    use_compiled_proposals,
)
from harness.scripted_plan import scripted_fetch_series_stats, scripted_plan
from harness.settings import Settings
from harness.studio import StudioRunner, create_studio_router
from harness.studio_store import StudioStore
from harness.tool_contracts import ResolverError, resolver_digest
from harness.turn.loop import Graph
from harness.turn.node import NodeCtx
from harness.turn.store import TurnStore
from harness.turn_budget import current_turn_budget
from harness.wiring import build_stated_scope_reader, handing_back_planner_failures
from orchestrator.agent import orchestrator_agent
from orchestrator.model import orchestrator_model
from planner.plan import Plan, SessionState, save_plan
from planner.specs import planner_specs
from planner.stated_scope import StatedScopeReader
from planner.tool_spec import ToolSpec
from sqlalchemy.ext.asyncio import AsyncEngine
from translator.tool import build_translate

# What a model says once its script runs out. The run records the exhaustion
# (`Log.exhausted`) and reports the case as under-scripted, so a turn nothing
# drove cannot pass.
_EXHAUSTED_TEXT = "done"
# The bearer token sent when `Settings` names none: the studio accepts any non-empty one.
_TOKEN = "orchestrator-eval"
# The model name the scripted orchestrator's requests carry.
_ORCHESTRATOR_MODEL = "orchestrator-script"


class Log:
    """What a case recorded, in order: one dict per model request and per tool call.

    A request entry is `{"kind": "request", "agent": ..., "system": str,
    "contents": [Content as JSON]}`; a call entry is `{"kind": "call",
    "layer": "step" | "executor", "tool": ..., "arguments": ..., "outcome":
    "ok" | "refused", "code": ...}`, and one from a scripted planner's tool
    call carries `"layer": "planner"`, no arguments, and the `outcome`, `code`
    and whole `response` of the answer the model receives
    (`planner_tool_answered`); a `{"kind": "turn", "index": n}` marker opens
    each message turn.
    """

    def __init__(self) -> None:
        self.entries: list[dict[str, Any]] = []
        self.exhausted: list[str] = []

    def mark_turn(self, index: int) -> None:
        self.entries.append({"kind": "turn", "index": index})

    def planner_tool_answered(self, tool: str, answer: dict[str, Any] | None) -> None:
        """Record a planner tool call by the answer the model receives for it: the
        tool's own, an argument refusal, or the adapter's answer for a tool that
        raised (`PlannerAdapter`'s `on_tool_answered`)."""
        response = answer or {}
        self.entries.append(
            {
                "kind": "call",
                "layer": "planner",
                "tool": tool,
                "outcome": response.get("outcome"),
                "code": response.get("code"),
                "response": copy.deepcopy(response),
            }
        )

    def record_request(self, agent: str, *, system: str, contents: Sequence[types.Content]) -> None:
        self.entries.append(
            {
                "kind": "request",
                "agent": agent,
                "system": system,
                "contents": [
                    content.model_dump(mode="json", exclude_none=True) for content in contents
                ],
            }
        )


def _part_from_step(step: Mapping[str, Any]) -> types.Part:
    """One script entry: `{"call": tool, "args": {...}}` or `{"text": ...}`."""
    if "call" in step:
        return types.Part.from_function_call(name=step["call"], args=dict(step.get("args") or {}))
    return types.Part.from_text(text=step["text"])


class OrchestratorScript:
    """The orchestrator's model from the case's `orchestrator` script, one part per model call."""

    def __init__(self, *, script: list[types.Part], log: Log) -> None:
        self._script = script
        self._log = log

    def __call__(self, request: ModelRequest) -> types.Part:
        self._log.record_request(
            "orchestrator", system=request.system_text, contents=request.contents
        )
        if self._script:
            return self._script.pop(0)
        self._log.exhausted.append("orchestrator")
        return types.Part.from_text(text=_EXHAUSTED_TEXT)


class OfferedPlans:
    """The planner's model when the case scripts none: it proposes the plan it
    was handed, once per entry.

    The plan is written into the turn's planner state the way a translated
    `propose_plan` leaves a compiled plan there, then a `propose_plan` call
    declares it ready (`harness.scripted.use_compiled_proposals`). Every other
    generation closes the run.
    """

    def __init__(self, *, log: Log) -> None:
        self._log = log
        self._state: SessionState | None = None
        self._offered: Plan | None = None

    def offer(self, state: SessionState, plan: Plan) -> None:
        self._state = state
        self._offered = plan

    def __call__(self, request: ModelRequest) -> types.Part:
        self._log.record_request("planner", system=request.system_text, contents=request.contents)
        plan, self._offered = self._offered, None
        if plan is None or self._state is None:
            return types.Part.from_text(text=_EXHAUSTED_TEXT)
        save_plan(self._state, plan)
        return types.Part.from_function_call(name="propose_plan", args={})


class PlannerScript:
    """The planner's model from the case's `planner` script, one part per model call."""

    def __init__(self, *, script: list[types.Part], log: Log) -> None:
        self._script = script
        self._log = log

    def __call__(self, request: ModelRequest) -> types.Part:
        self._log.record_request("planner", system=request.system_text, contents=request.contents)
        if self._script:
            return self._script.pop(0)
        self._log.exhausted.append("planner")
        return types.Part.from_text(text=_EXHAUSTED_TEXT)


class ScriptedPairPlanner:
    """`harness.graph.interfaces.PlannerModel` over the real `PlannerAdapter` and `OfferedPlans`.

    A first draft offers `scripted_plan`; an amendment or a replan offers the
    next version of the plan the graph holds, as `harness.scripted.
    ScriptedPlannerModel` does.
    """

    def __init__(self, *, adapter: PlannerAdapter, plans: OfferedPlans, spec_id: str) -> None:
        self._adapter = adapter
        self._plans = plans
        self._spec_id = spec_id

    async def __call__(
        self,
        *,
        ctx: NodeCtx[GraphEvent],
        question: str,
        prior_plan: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        plan = scripted_plan(question, spec_id=self._spec_id)
        if prior_plan is not None:
            prior_version = int(prior_plan.get("version", 1))
            plan = plan.model_copy(
                update={"version": prior_version + 1, "previous_version": prior_version}
            )
        self._plans.offer(ctx.state, plan)
        return await self._adapter(ctx=ctx, question=question, prior_plan=prior_plan, **kwargs)

    async def revise(self, **kwargs: Any) -> dict[str, Any]:
        """The adapter's own revision, which calls no model."""
        return await self._adapter.revise(**kwargs)


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

    async def __call__(self, **kwargs: Any) -> dict[str, Any]:
        del kwargs
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
            raise ToolSurfaceUnavailableError(
                "the case scripted no further translator search", retryable=False
            )
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


def _planner_adapter(
    settings: Settings,
    *,
    surface: ScriptedSurface,
    answer: OfferedPlans | PlannerScript,
    tools: Callable[[Sequence[ToolSpec]], Sequence[ToolSpec]] = tuple,
    on_tool_answered: Callable[[str, dict[str, Any] | None], None] | None = None,
    stated_scope: StatedScopeReader | None = None,
) -> PlannerAdapter:
    """The real adapter, built with `settings`, over `surface`, its model calls
    answered by `answer`, its tools the ones the adapter builds for itself
    (`planner.specs.planner_specs` over `surface` and its translation) as
    `tools` changes them, and each tool's answer told to `on_tool_answered`.
    `stated_scope` is the reader of the provider scope the user's words state, as
    production's planner has (`harness.wiring.build_stated_scope_reader`)."""
    translate = build_translate(surface=surface)
    specs = planner_specs(translate=translate, surface=surface, stated_scope=stated_scope)
    return PlannerAdapter(
        settings=settings,
        surface=surface,
        translate=translate,
        specs=tools(specs),
        client=scripted_client(answer),
        on_tool_answered=on_tool_answered,
        stated_scope=stated_scope,
    )


def build_studio(scenario: Mapping[str, Any], *, log: Log, engine: AsyncEngine) -> FastAPI:
    """The studio for one case, its conversations and turns stored on `engine`."""
    settings = Settings()
    # Closed with the app by `orchestrator_client` (`app.state.stated_scope`).
    stated_scope = build_stated_scope_reader(settings)
    orchestrator_genai = scripted_client(
        OrchestratorScript(
            script=[_part_from_step(step) for step in scenario["orchestrator"]], log=log
        )
    )
    resolver = CannedResolver(copy.deepcopy(scenario.get("search") or []))
    planner_script = scenario.get("planner")
    surface = LoggingSurface(
        log=log,
        refusals=scenario.get("get_data_refusals") or {},
        real_schemas=planner_script is not None,
    )
    if planner_script is None:
        plans = OfferedPlans(log=log)
        adapter = _planner_adapter(
            settings,
            surface=surface,
            answer=plans,
            tools=use_compiled_proposals,
            stated_scope=stated_scope,
        )
        planner = handing_back_planner_failures(
            ScriptedPairPlanner(
                adapter=adapter, plans=plans, spec_id=scenario.get("plan_spec_id", SCRIPTED_SPEC)
            )
        )
    else:
        script = PlannerScript(script=[_part_from_step(step) for step in planner_script], log=log)
        adapter = _planner_adapter(
            settings,
            surface=CannedDataSurface(
                log=log, searches=copy.deepcopy(scenario.get("translator_search") or [])
            ),
            answer=script,
            on_tool_answered=log.planner_tool_answered,
            stated_scope=stated_scope,
        )
        planner = handing_back_planner_failures(adapter)

    # Each conversation's runs are its own, as the served `StudioRunLedger`
    # keeps them.
    run_ledger = InMemoryRunLedger(scope=current_conversation_id.get)

    def create_graph(recorder: Any, surface_recorder: Any) -> Graph[GraphEvent]:
        tools = build_orchestrator_tools(
            resolver=resolver, executor=recorder, surface=surface_recorder
        )
        return studio_graph(
            planner=planner,
            orchestrator=orchestrator_agent(tools, include_content=content_on_spans()),
            # One model per orchestrator run, on the turn's budget, as served.
            model=lambda: orchestrator_model(
                _ORCHESTRATOR_MODEL, client=orchestrator_genai, budget=current_turn_budget.get()
            ),
            tools=tools,
            executor=recorder,
            tool_surface=surface_recorder,
            definition_surface=surface,
            publisher=scripted_publish_result,
            run_ledger=run_ledger,
            max_concurrent_steps=settings.executor_max_concurrent_steps,
            fetch_series_stats=scripted_fetch_series_stats,
            label_refs=no_label_refs,
        )

    store = StudioStore(engine)
    studio = StudioRunner(
        create_graph=create_graph,
        executor=LoggingExecutor(log=log),
        store=store,
        turns=TurnStore(engine, clock=store.now),
        tool_surface=surface,
    )
    app = FastAPI()
    app.state.settings = settings
    app.state.stated_scope = stated_scope
    app.include_router(create_studio_router(studio, settings=settings))
    return app


@asynccontextmanager
async def orchestrator_client(
    scenario: Mapping[str, Any], *, engine: AsyncEngine
) -> AsyncIterator[tuple[httpx2.AsyncClient, Log]]:
    """An in-process client on the studio for `scenario`, and its log. The
    studio stores its conversations on `engine`."""
    log = Log()
    app = build_studio(scenario, log=log, engine=engine)
    token = app.state.settings.studio_api_token or _TOKEN
    transport = httpx2.ASGITransport(app=app)
    async with httpx2.AsyncClient(
        transport=transport,
        base_url="http://studio-eval.local",
        timeout=30.0,
        headers={"Authorization": f"Bearer {token}"},
    ) as client:
        try:
            yield client, log
        finally:
            if app.state.stated_scope is not None:
                await app.state.stated_scope.aclose()


__all__ = ["Log", "build_studio", "orchestrator_client", "search_envelope"]
