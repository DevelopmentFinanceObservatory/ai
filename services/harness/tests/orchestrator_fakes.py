"""Scripted stand-ins for driving the Studio routes on the turn loop in tests.

`orchestrator_app` serves the studio routes on the turn graph, wired as
`harness.server` wires it, its orchestrator answering from a script
(`ScriptLlm`), and `stream` posts to a route and reads its SSE frames back.
`call`, `search_call`, `fetch_call`, `make_charts_call` and `plan_call`
build the script's parts; `request_text` and `handed_back` read what the
model was sent.

`CannedSearch` answers `search` from data-mcp search envelopes that
`definite_envelope` builds, and `frame_for` names the frame a `fetch` of a
spec_id registers when `harness.scripted.ScriptedExecutor` answers it.
"""

from __future__ import annotations

import asyncio
import copy
import json
from collections.abc import Awaitable, Callable, Mapping, Sequence
from datetime import UTC, datetime
from typing import Any

import httpx2
from executor.ledger import InMemoryRunLedger
from fastapi import FastAPI
from google.genai import types
from harness.conversation import current_conversation_id
from harness.graph.state import frame_id_for
from harness.graph.workflow import studio_graph
from harness.orchestrator_tools import build_orchestrator_tools
from harness.planner_adapter import PlannerAdapterError, content_on_spans
from harness.scripted import (
    SCRIPTED_CV,
    SCRIPTED_SPEC,
    ScriptedExecutor,
    ScriptedPlannerModel,
    scripted_publish_result,
)
from harness.scripted_plan import scripted_fetch_series_stats
from harness.settings import Settings
from harness.studio import StudioRunner, create_studio_router
from harness.studio_state import make_recall_part
from harness.studio_store import StudioStore
from harness.tool_contracts import ResolverError, resolver_digest
from harness.turn.store import TurnStore
from harness.turn_budget import TurnBudget, current_turn_budget
from harness.wire import STUDIO_FRAME_ADAPTER
from harness.wiring import handing_back_planner_failures
from orchestrator.agent import orchestrator_agent
from orchestrator.model import orchestrator_model
from planner.plan import AskBreakdown, load_ask_breakdowns, load_asks, load_asks_across_years
from pydantic_ai.models.google import GoogleModel
from sqlalchemy.ext.asyncio import AsyncEngine

from .data_mcp_definitions import DataMcpDefinitions
from .scripted_genai import Reply, Request, ScriptedGenai, _stream
from .turn_driver import MODEL, NO_ADK, no_turn_specs, unconnected_store

__all__ = ["NO_ADK", "no_turn_specs", "unconnected_store"]

AUTH = {"Authorization": "Bearer test-token"}
CONVERSATION = "conv_01test0000"

# A script entry: a part the model answers with, a whole reply, or a function
# of the request that returns either, awaited when it returns an awaitable.
type ScriptEntry = (
    types.Part | Reply | Callable[[Request], types.Part | Reply | Awaitable[types.Part | Reply]]
)


class ScriptLlm(ScriptedGenai):
    """The orchestrator's model client, answering each call with the next
    entry of `script` and recording the request (`requests`). A part is
    answered as one chunk. Once the script runs out it answers "done" in
    plain text, so a turn with no script is a plain-text answer.
    `model` builds the orchestrator's model on it."""

    def __init__(self, script: Sequence[ScriptEntry] = ()) -> None:
        super().__init__([])
        self.script: list[ScriptEntry] = list(script)

    def model(self, *, budget: TurnBudget | None = None) -> GoogleModel:
        """The orchestrator's model (`orchestrator.model.orchestrator_model`) on this client."""
        return orchestrator_model(MODEL, client=self.client, budget=budget)

    async def _open(self, *, model: str, contents: Any, config: Any = None) -> Any:
        request = Request(
            model=model, contents=copy.deepcopy(contents), config=copy.deepcopy(config)
        )
        self.requests.append(request)
        entry: Any = self.script.pop(0) if self.script else types.Part.from_text(text="done")
        if callable(entry):
            entry = entry(request)
            if asyncio.iscoroutine(entry) or isinstance(entry, Awaitable):
                entry = await entry
        reply = entry if isinstance(entry, Reply) else Reply(chunks=[[entry]])
        if reply.open_error is not None:
            raise reply.open_error
        return _stream(reply)


class FailingPlanner:
    """A fake planner that raises `PlannerAdapterError`, the way `PlannerAdapter` fails."""

    def __init__(
        self,
        message: str = (
            "a plan that is draft was declared ready, not pending_approval; "
            "refusing to report it ready"
        ),
    ) -> None:
        self.message = message

    async def __call__(self, **kwargs: Any) -> dict[str, Any]:
        raise PlannerAdapterError(self.message)


class RefusingPlanner:
    """A fake planner that refuses to draft, as `PlannerAdapter` does at its round cap."""

    def __init__(self, *, final: bool = False) -> None:
        self.questions: list[str] = []
        self.readings: list[Any] = []
        self.asks: list[tuple[str, ...]] = []
        self.breakdowns: list[tuple[AskBreakdown, ...]] = []
        self.across_years: list[tuple[int, ...]] = []
        self.final = final

    async def __call__(self, **kwargs: Any) -> dict[str, Any]:
        self.questions.append(kwargs["question"])
        self.readings.append(kwargs.get("readings"))
        self.asks.append(load_asks(kwargs["ctx"].state))
        self.breakdowns.append(load_ask_breakdowns(kwargs["ctx"].state))
        self.across_years.append(load_asks_across_years(kwargs["ctx"].state))
        return {"refused": {"text": "the planner ran out of rounds", "final": self.final}}


async def no_resolver(**kwargs: Any) -> dict[str, Any]:
    raise AssertionError("no search expected")


async def no_executor(*, tool: str, args: dict[str, Any], on_progress: Any = None) -> Any:
    raise AssertionError(f"no tool call expected, got {tool}")


def definite_envelope(
    *, spec_id: str = SCRIPTED_SPEC, sentence: str = "Life expectancy in Kenya"
) -> dict[str, Any]:
    """A data-mcp search envelope with one definite part reading `spec_id`."""
    spec = {
        "spec_id": spec_id,
        "resolved": {
            "rendered_sentence": sentence,
            "confidence": "high",
            "coverage": {
                "kind": "exact",
                "has_data": True,
                "window": {"start_year": 1960, "end_year": 2023},
            },
            "stat_vars": {
                "member_count": 1,
                "member_count_exact": True,
                "served_count": 1,
                "data_date_range": {"start": "1960", "end": "2023"},
            },
            "entities": [{"role": "observationAbout", "label": "Kenya", "ref": "country/KEN"}],
            "facets": {},
        },
        "definition": {
            "schema_version": 1,
            "spec_id": spec_id,
            "identity": {"stat_var": "LifeExpectancy_Person"},
        },
    }
    part = {"status": "definite", "variable_text": sentence, "warnings": [], "spec": spec}
    return {"schema_version": 1, "kind": "resolution", "status": "definite", "parts": [part]}


class CannedSearch:
    """A `Resolver` answering each search with the next envelope, digested as
    `harness.mcp_client.McpResolver` digests it. `calls` records each call's
    arguments. A search past the last envelope raises `ResolverError`."""

    def __init__(self, *envelopes: Mapping[str, Any]) -> None:
        self._envelopes = [dict(envelope) for envelope in envelopes]
        self.calls: list[dict[str, Any]] = []

    async def __call__(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        if not self._envelopes:
            raise ResolverError("no further search result was scripted")
        return resolver_digest(self._envelopes.pop(0))


def frame_for(spec_id: str = SCRIPTED_SPEC) -> str:
    """The frame id a `fetch` of `spec_id` registers when
    `harness.scripted.ScriptedExecutor` answers its `get_data`."""
    return frame_id_for({"spec_id": spec_id, "content_version": SCRIPTED_CV})


def call(name: str, **args: Any) -> types.Part:
    """A model turn calling the tool `name` with `args`."""
    return types.Part.from_function_call(name=name, args=args)


def search_call(*variables: str, text: str | None = None) -> types.Part:
    """A parsed `search` call with one part per variable; `text` makes it a retry."""
    args: dict[str, Any] = {"parts": [{"variable": v} for v in variables]}
    if text is not None:
        args["text"] = text
    return call("search", **args)


def fetch_call(spec_id: str = SCRIPTED_SPEC, **args: Any) -> types.Part:
    """A `fetch` call; `args` are its `scope`, `stat_vars` or `breakdown`."""
    return call("fetch", spec_id=spec_id, **args)


def chart(
    frame: str,
    *,
    type_: str = "line_chart",
    title: str = "Life expectancy in Kenya",
    select: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """One `make_charts` chart on `frame`, its `select` sent as `select_json`.
    Left unset, the select names the one series of a scripted frame
    (`harness.scripted.ScriptedExecutor`), whose rows no store holds for an
    empty select to be checked against."""
    if select is None:
        select = {"stat_var": "dcid:LifeExpectancy_Person"}
    return {"frame": frame, "type": type_, "title": title, "select_json": json.dumps(select)}


def make_charts_call(*charts: Mapping[str, Any]) -> types.Part:
    """A `make_charts` call drawing `charts`, each built with `chart`."""
    return call("make_charts", charts=list(charts))


def plan_call(brief: str, *asks: str) -> types.Part:
    """A `plan` call handing `brief` and its `asks` to the planner. Given no
    asks, it lists `brief` as the one output asked for, since the `plan` tool
    refuses a call that lists none."""
    return call("plan", brief=brief, asks=list(asks or (brief,)))


def approve_answer(revision: object, *, at: datetime | None = None) -> dict[str, Any]:
    """An answer approving the plan card that names `revision`, given at `at`
    (now by default), as the studio's interrupt route builds it."""
    return {"approved": revision, "at": (at or datetime.now(UTC)).isoformat()}


def request_text(request: Request) -> str:
    """Every text part of a model request, and every hand-back it shows as
    the result of a `plan` or `ask_user` call (`hand_back_text`), one per line."""
    return "\n".join(_part_text(p) for c in request.content_list for p in c.parts or [])


def _part_text(part: types.Part) -> str:
    response = part.function_response
    if response is not None and response.name in _HANDED_BACK_TO:
        return hand_back_text(response.response or {})
    return part.text or ""


# The calls `harness.graph.orchestrate` answers with a hand-back.
_HANDED_BACK_TO = frozenset({"plan", "ask_user"})


def hand_back_text(result: dict[str, Any]) -> str:
    """A hand-back's result (`harness.graph.orchestrate.AgentHandBack.response`)
    as lines: its text, its structured fields as one JSON object, then its
    plan record. Empty for a result that holds no hand-back."""
    if "result" not in result:
        return ""
    fields = {key: value for key, value in result.items() if key not in ("result", "plan_record")}
    lines = [str(result["result"])]
    if fields:
        lines.append(json.dumps(fields))
    if "plan_record" in result:
        lines.append(str(result["plan_record"]))
    return "\n".join(lines)


def handed_back(model: ScriptLlm) -> tuple[str, dict[str, Any]]:
    """The last request's text and the last `plan` result in it."""
    request = model.requests[-1]
    (result, *_) = [
        part.function_response.response or {}
        for content in reversed(request.content_list)
        for part in content.parts or []
        if part.function_response is not None and part.function_response.name == "plan"
    ]
    return request_text(request), result


def orchestrator_app(
    *script: ScriptEntry,
    resolver: Any = None,
    executor: Any = None,
    planner: Any = None,
    tool_surface: Any = None,
    engine: AsyncEngine | None = None,
    store: StudioStore | None = None,
    settings: Settings | None = None,
    edit_model: Any = None,
    served: Sequence[Mapping[str, Any]] = (),
    publisher: Any = scripted_publish_result,
    run_ledger: Any = None,
    definition_surface: Any = None,
    early_mints: Any = None,
    heartbeat_seconds: float | None = None,
) -> tuple[FastAPI, ScriptLlm]:
    """The studio routes on the turn graph, and the agent's model client.

    Wired as `harness.server` wires the studio: the agent's tools and the
    `execute` node run through the studio's recorder, the planner hands its
    failures back to the agent, and turns stream over SSE. The model
    answers from `script`, then "done" in plain text.

    `resolver` answers `search` and fails the test by default. `executor`
    answers `get_data`, `describe_frame` and `write_component`
    (`harness.scripted.ScriptedExecutor` by default). `planner` drafts
    plans (`ScriptedPlannerModel` by default). `tool_surface` runs an
    approved plan's steps, and with `None` `execute` refuses. `served` are
    the seeds or Spec blocks data-mcp's definition lookup answers with, and
    `definition_surface` replaces that lookup whole.
    `publisher` publishes a run's derived results, all at once by default,
    and `run_ledger` keeps its runs, an in-memory ledger that keeps each
    conversation's runs apart by default. `early_mints` starts each offered
    card's retrievals, none by default. Conversations and their turns are
    stored in `store`, or in a store built on `engine` (the `pg_engine`
    fixture's); one of the two is required. A stream renews its lease every
    `heartbeat_seconds`, the server's interval by default.
    """
    if store is None:
        if engine is None:
            raise TypeError("orchestrator_app needs a store, or the engine to build one on")
        store = StudioStore(engine, adk=NO_ADK, turn_specs=no_turn_specs)
    resolved_settings = settings or Settings()
    model = ScriptLlm(script)
    resolver = resolver or no_resolver
    planner_model = handing_back_planner_failures(planner or ScriptedPlannerModel())
    ledger = (
        run_ledger
        if run_ledger is not None
        else InMemoryRunLedger(scope=current_conversation_id.get)
    )

    def create_graph(recorder: Any, surface_recorder: Any) -> Any:
        tools = build_orchestrator_tools(
            resolver=resolver,
            executor=recorder,
            surface=surface_recorder,
            recall_part=make_recall_part(store),
        )
        return studio_graph(
            planner=planner_model,
            orchestrator=orchestrator_agent(tools, include_content=content_on_spans()),
            model=lambda: model.model(budget=current_turn_budget.get()),
            tools=tools,
            executor=recorder,
            tool_surface=surface_recorder,
            definition_surface=definition_surface or DataMcpDefinitions(served=served),
            publisher=publisher,
            run_ledger=ledger,
            max_concurrent_steps=resolved_settings.executor_max_concurrent_steps,
            fetch_series_stats=scripted_fetch_series_stats,
            early_mints=early_mints,
        )

    studio = StudioRunner(
        create_graph=create_graph,
        executor=executor or ScriptedExecutor(),
        store=store,
        turns=TurnStore(store.engine),
        tool_surface=tool_surface,
        edit_model=edit_model,
        early_mints=early_mints,
        **({"heartbeat_seconds": heartbeat_seconds} if heartbeat_seconds is not None else {}),
    )
    app = FastAPI()
    app.state.studio = studio
    app.include_router(create_studio_router(studio, settings=resolved_settings))
    return app, model


def studio_client(app: FastAPI) -> httpx2.AsyncClient:
    """An in-process client on `app`'s routes."""
    return httpx2.AsyncClient(
        transport=httpx2.ASGITransport(app=app), base_url="http://studio.test"
    )


def turn_body(message: str, *, conversation_id: str = CONVERSATION) -> dict[str, Any]:
    """A `turn.request` body for `/api/studio/turn`."""
    return {
        "kind": "turn.request",
        "schema_version": 1,
        "conversation_id": conversation_id,
        "message": message,
    }


async def stream(
    client: httpx2.AsyncClient, path: str, body: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """POST `body` to `path` and read the SSE stream to its end, each frame
    validated against the studio wire contract."""
    frames: list[dict[str, Any]] = []
    async with client.stream("POST", path, json=body, headers=AUTH) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        async for line in response.aiter_lines():
            if line.startswith("data: "):
                frame = json.loads(line.removeprefix("data: "))
                STUDIO_FRAME_ADAPTER.validate_python(frame)
                frames.append(frame)
    return frames
