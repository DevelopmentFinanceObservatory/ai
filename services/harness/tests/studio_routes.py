"""The Studio routes on the turn graph with a scripted orchestrator, for tests.

`orchestrator_app` serves the studio routes on the turn graph, wired as
`harness.server` wires it, its orchestrator answering from a script
(`ScriptLlm`), and `stream` posts to a route and reads its SSE frames back.
`studio_client` is an in-process client on the routes, `turn_body` a turn
request's body, and `frame_for` the frame a `fetch` of a spec_id registers
when `harness.scripted.ScriptedExecutor` answers it.

The parts a script holds are built with `tests.scripted_tools`.
"""

from __future__ import annotations

import asyncio
import copy
import json
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Any

import httpx2
from executor.ledger import InMemoryRunLedger
from fastapi import FastAPI
from google.genai import types
from harness.conversation import current_conversation_id
from harness.graph.state import frame_id_for
from harness.graph.workflow import studio_graph
from harness.orchestrator_tools import build_orchestrator_tools
from harness.planner_adapter import content_on_spans
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
from harness.turn.store import TurnStore
from harness.turn_budget import TurnBudget, current_turn_budget
from harness.wire import STUDIO_FRAME_ADAPTER
from harness.wiring import handing_back_planner_failures
from orchestrator.agent import orchestrator_agent
from orchestrator.model import orchestrator_model
from pydantic_ai.models.google import GoogleModel
from sqlalchemy.ext.asyncio import AsyncEngine

from .data_mcp_definitions import DataMcpDefinitions
from .scripted_genai import Reply, Request, ScriptedGenai, _stream
from .scripted_tools import no_resolver
from .turn_driver import CONVERSATION, MODEL

AUTH = {"Authorization": "Bearer test-token"}

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


def frame_for(spec_id: str = SCRIPTED_SPEC) -> str:
    """The frame id a `fetch` of `spec_id` registers when
    `harness.scripted.ScriptedExecutor` answers its `get_data`."""
    return frame_id_for({"spec_id": spec_id, "content_version": SCRIPTED_CV})


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
        store = StudioStore(engine)
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
        run_ledger=ledger,
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
