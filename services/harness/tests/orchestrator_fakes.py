"""Scripted stand-ins for driving the orchestrator topology in tests.

`ScriptLlm` answers each orchestrator model call with the next part of a
script; `call`, `search_call`, `fetch_call`, `make_charts_call` and
`plan_call` build those parts. Two drivers run a script:

- `orchestrator_runner` runs the root workflow under a bare ADK `Runner`,
  with `open_session`, `run_turn` and `single_pause` to drive it and
  `request_text` and `handed_back` to read what the model was sent.
- `orchestrator_app` serves the studio routes on the orchestrator topology,
  wired as `harness.server` wires it, and `stream` posts to a route and
  reads its SSE frames back.

`CannedSearch` answers `search` from data-mcp search envelopes that
`definite_envelope` builds, and `frame_for` names the frame a `fetch` of a
spec_id registers when `harness.scripted.ScriptedExecutor` answers it.
"""

from __future__ import annotations

import json
from collections.abc import AsyncGenerator, Mapping, Sequence
from datetime import UTC, datetime
from typing import Any

import httpx2
from fastapi import FastAPI
from google.adk.apps import App
from google.adk.events import Event
from google.adk.flows.llm_flows._fencing import OTHER_AGENT_CONTEXT_PREAMBLE, QUOTED_CONTENT_BEGIN
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_response import LlmResponse
from google.adk.runners import Runner
from google.adk.sessions import BaseSessionService, InMemorySessionService
from google.genai import types
from harness.graph.state import AGENT_NAME, frame_id_for
from harness.graph.workflow import create_root_workflow
from harness.orchestrator_tools import build_orchestrator_tools
from harness.planner_seam import PlannerSeamError
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
from harness.studio_state import StudioStore, make_recall_part
from harness.studio_translate import _node_name
from harness.tool_contracts import ResolverError, resolver_digest
from harness.wire import STUDIO_FRAME_ADAPTER
from harness.wiring import handing_back_planner_failures
from orchestrator.agent import create_orchestrator_agent
from planner.plan import AskBreakdown, load_ask_breakdowns, load_asks
from pydantic import Field

from .data_mcp_definitions import DataMcpDefinitions

AUTH = {"Authorization": "Bearer test-token"}
CONVERSATION = "conv_01test0000"


class ScriptLlm(BaseLlm):
    """Answers each model call with the next scripted part, recording requests.

    Once the script runs out it answers "done" in plain text, so a turn
    with no script is a plain-text answer.
    """

    model: str = "script-llm"
    script: list[types.Part] = Field(default_factory=list)
    requests: list[Any] = Field(default_factory=list)

    async def generate_content_async(
        self, llm_request: Any, stream: bool = False
    ) -> AsyncGenerator[LlmResponse]:
        self.requests.append(llm_request)
        part = self.script.pop(0) if self.script else types.Part.from_text(text="done")
        yield LlmResponse(content=types.Content(role="model", parts=[part]))


class FailingPlanner:
    """A planner seam that raises `PlannerSeamError`, the way `PlannerSeam` fails."""

    def __init__(
        self, message: str = "the planner seam ran past 60 events without settling"
    ) -> None:
        self.message = message

    async def __call__(self, **kwargs: Any) -> dict[str, Any]:
        raise PlannerSeamError(self.message)


class RefusingPlanner:
    """A planner seam that refuses to draft, as `PlannerSeam` does at its round cap."""

    def __init__(self, *, final: bool = False) -> None:
        self.questions: list[str] = []
        self.readings: list[Any] = []
        self.asks: list[tuple[str, ...]] = []
        self.breakdowns: list[tuple[AskBreakdown, ...]] = []
        self.final = final

    async def __call__(self, **kwargs: Any) -> dict[str, Any]:
        self.questions.append(kwargs["question"])
        self.readings.append(kwargs.get("readings"))
        self.asks.append(load_asks(kwargs["ctx"].state))
        self.breakdowns.append(load_ask_breakdowns(kwargs["ctx"].state))
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


def search_call(query: str, *variables: str) -> types.Part:
    """A `search` call with one part per variable, or one part reading `query`."""
    return call("search", query=query, parts=[{"variable": v} for v in variables or (query,)])


def fetch_call(spec_id: str = SCRIPTED_SPEC, **args: Any) -> types.Part:
    """A `fetch` call; `args` are its `scope`, `stat_vars` or `breakdown`."""
    return call("fetch", spec_id=spec_id, **args)


def chart(
    frame: str,
    *,
    type_: str = "line_chart",
    title: str = "Life expectancy in Kenya, 1960 to 2023",
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
    """A `plan` call handing `brief` and its `asks` to the planner."""
    return call("plan", brief=brief, asks=list(asks))


def approve_answer(revision: object, *, at: datetime | None = None) -> dict[str, Any]:
    """An answer approving the plan card that names `revision`, given at `at`
    (now by default), as the studio's interrupt route builds it."""
    return {"approved": revision, "at": (at or datetime.now(UTC)).isoformat()}


def create_request_input_response(interrupt_id: str, response: dict[str, Any]) -> types.Part:
    """The user's answer to a pause, as ADK's `adk_request_input` tool reads it."""
    return types.Part(
        function_response=types.FunctionResponse(
            id=interrupt_id, name="adk_request_input", response=response
        )
    )


def request_text(request: Any) -> str:
    """Every text part of a model request, one per line."""
    return "\n".join(p.text or "" for c in request.contents for p in c.parts or [])


def content_text(content: types.Content) -> str:
    return "".join(p.text or "" for p in content.parts or [])


def quotes_another_agent(request: Any) -> bool:
    """Whether ADK fenced any content of `request` as another agent's."""
    text = request_text(request)
    return QUOTED_CONTENT_BEGIN in text or OTHER_AGENT_CONTEXT_PREAMBLE in text


def carries_an_interrupt(request: Any) -> bool:
    """Whether `request` holds a pause's `adk_request_input` call, as a call or quoted."""
    return "adk_request_input" in "\n".join(str(p) for c in request.contents for p in c.parts or [])


def handed_back(model: ScriptLlm) -> tuple[str, dict[str, Any]]:
    """The last request's text and the plan result's `{"frames": ...}` body in it."""
    text = request_text(model.requests[-1])
    return text, json.loads(next(line for line in text.split("\n") if line.startswith('{"frames"')))


def orchestrator_runner(
    *script: types.Part,
    planner: Any = None,
    tool_surface: Any = None,
    executor: Any = None,
    resolver: Any = None,
    definition_surface: Any = None,
    publisher: Any = scripted_publish_result,
    run_ledger: Any = None,
) -> tuple[Runner, ScriptLlm]:
    """The root workflow on the orchestrator topology under a bare ADK
    `Runner` (app `t`), its model answering from `script`. The planner
    refuses unless one is given; search and every executor call fail the
    test unless a resolver or executor is given. `definition_surface`
    resolves seeds, data-mcp over a Search API that serves nothing unless
    one is given. `publisher` publishes a run's derived results, all of
    them at once by default, and `run_ledger` keeps its runs, the
    workflow's in-memory ledger by default."""
    model = ScriptLlm(script=list(script))
    agent = create_orchestrator_agent(
        model_name="unused",
        tools=build_orchestrator_tools(
            resolver=resolver or no_resolver, executor=executor or no_executor
        ),
    ).model_copy(update={"model": model})
    workflow = create_root_workflow(
        planner=planner or RefusingPlanner(),
        orchestrator=agent,
        executor=executor or no_executor,
        tool_surface=tool_surface,
        definition_surface=definition_surface or DataMcpDefinitions(),
        publisher=publisher,
        run_ledger=run_ledger,
        fetch_series_stats=scripted_fetch_series_stats,
    )
    runner = Runner(
        app=App(name="t", root_agent=workflow), session_service=InMemorySessionService()
    )
    return runner, model


async def open_session(runner: Runner, **state: Any) -> None:
    """Create `orchestrator_runner`'s session on the studio surface, with `state`."""
    await runner.session_service.create_session(
        app_name="t", user_id="u", session_id="s", state={"surface": "studio", **state}
    )


async def run_turn(runner: Runner, message: types.Content) -> list[Event]:
    """Run `message` through `orchestrator_runner`'s session; every event it yields."""
    return [
        event async for event in runner.run_async(user_id="u", session_id="s", new_message=message)
    ]


def single_pause(events: list[Event]) -> types.FunctionCall:
    """The one `adk_request_input` call among `events`."""
    calls = [c for e in events for c in e.get_function_calls() if c.name == "adk_request_input"]
    assert len(calls) == 1, [c.id for c in calls]
    return calls[0]


def visited_nodes(events: list[Event]) -> list[str]:
    """The workflow node each node event came from, in order."""
    return [_node_name(e) for e in events if e.node_info]


def orchestrator_app(
    *script: types.Part,
    resolver: Any = None,
    executor: Any = None,
    planner: Any = None,
    tool_surface: Any = None,
    session_service: BaseSessionService | None = None,
    store: StudioStore | None = None,
    settings: Settings | None = None,
    edit_model: Any = None,
    served: Sequence[Mapping[str, Any]] = (),
    publisher: Any = scripted_publish_result,
    run_ledger: Any = None,
) -> tuple[FastAPI, ScriptLlm]:
    """The studio routes on the orchestrator topology, and the agent's model.

    Wired as `harness.server` wires the studio: the agent's tools and the
    `execute` node run through the studio's recorder, the planner hands its
    failures back to the agent, and turns stream over SSE. The model
    answers from `script`, then "done" in plain text.

    `resolver` answers `search` and fails the test by default. `executor`
    answers `get_data`, `describe_frame` and `write_component`
    (`harness.scripted.ScriptedExecutor` by default). `planner` drafts
    plans (`ScriptedPlannerModel` by default). `tool_surface` runs an
    approved plan's steps, and with `None` `execute` refuses. `served` are
    the seeds or Spec blocks data-mcp's definition lookup answers with.
    `publisher` publishes a run's derived results, all at once by default,
    and `run_ledger` keeps its runs, the workflow's in-memory ledger by
    default.
    """
    session_service = session_service or InMemorySessionService()
    store = store or StudioStore(session_service, app_name=AGENT_NAME)
    resolved_settings = settings or Settings()
    model = ScriptLlm(script=list(script))
    resolver = resolver or no_resolver
    seam = handing_back_planner_failures(planner or ScriptedPlannerModel())

    def create_workflow(recorder: Any, surface_recorder: Any) -> Any:
        agent = create_orchestrator_agent(
            model_name="unused",
            tools=build_orchestrator_tools(
                resolver=resolver,
                executor=recorder,
                surface=surface_recorder,
                recall_part=make_recall_part(store),
            ),
        ).model_copy(update={"model": model})
        return create_root_workflow(
            planner=seam,
            orchestrator=agent,
            executor=recorder,
            tool_surface=surface_recorder,
            definition_surface=DataMcpDefinitions(served=served),
            publisher=publisher,
            run_ledger=run_ledger,
            max_concurrent_steps=resolved_settings.executor_max_concurrent_steps,
            fetch_series_stats=scripted_fetch_series_stats,
        )

    studio = StudioRunner(
        create_workflow=create_workflow,
        executor=executor or ScriptedExecutor(),
        session_service=session_service,
        tool_surface=tool_surface,
        store=store,
        edit_model=edit_model,
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
