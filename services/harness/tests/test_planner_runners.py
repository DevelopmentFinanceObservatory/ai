"""The planner runner the settings select, and turns that cross from one runner to the other.

`Settings.planner_runner` selects the planner `PlannerSeam` runs. Both planners
store the same session events, so a turn paused on an ask under one runner
resumes under the other. Each test here runs a script on a scripted
google-genai client with every run on the ADK planner, with every run on the
Pydantic AI planner, and, for the turns that cross, with the runs split
between the two, each switch building a seam afresh over the same session as a
deploy or a rollback would. The runs are compared on what the seam answers,
the progress it reports, the requests the model received, the events the
session stores, the steps `session_steps` reads from them and the stored
state, as `tests.test_planner_node` compares them.
"""

from __future__ import annotations

import asyncio
import itertools
import json
import re
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from datetime import UTC, datetime
from typing import Any
from unittest import mock

import pytest
from analysis_mcp.server import mcp as analysis_mcp
from compiler.surface import FixtureSurface, McpSurface, ToolSchema
from data_mcp.definition_store import InMemoryDefinitionStore
from data_mcp.definition_writes import windowed_identity
from data_mcp.server import mcp as data_mcp
from google.adk import Context, Event
from google.adk.agents import LlmAgent
from google.adk.models import Gemini
from google.adk.runners import Runner as AdkRunner
from google.adk.sessions import InMemorySessionService
from google.adk.tools.long_running_tool import LongRunningFunctionTool
from google.adk.workflow import START, Workflow, node
from google.adk.workflow.utils._workflow_hitl_utils import create_request_input_response
from google.genai import Client, types
from harness import planner_seam
from harness import studio as studio_module
from harness.graph import switch as switch_module
from harness.graph.state import APPROVE_INTERRUPT, ORCHESTRATOR_SEARCHES_KEY
from harness.planner_node import SKIP_THOUGHT_SIGNATURE
from harness.planner_run import failed_tool_answer
from harness.planner_seam import PlannerSeam
from harness.prefix_cache import PrefixCache, prefix_key
from harness.scripted import _compiled_proposal, instruction_text
from harness.sentences import SeamCode
from harness.settings import Settings
from harness.turn_budget import TurnBudget
from harness.turn_events import PLANNER_ASK
from harness.turn_records import SearchReading
from planner import agent as planner_agent_module
from planner.agent import session_steps
from planner.call import PlannerCall
from planner.plan import (
    DatasetRequirement,
    Plan,
    ResolvedBy,
    load_plan,
    save_plan,
)
from planner.settings import AgentSettings
from planner.specs import PROPOSE_PLAN
from planner.tool_results import Outcome, ProblemCode
from planner.tool_spec import ToolSpec
from planner.tools import EndRun
from translator.tool import build_translate

from .data_mcp_definitions import DataMcpDefinitions, SearchStub
from .orchestrator_fakes import open_session, orchestrator_runner, plan_call
from .planner.planner_pairs import assert_pinned_wire_differences, seam_specs
from .scripted_genai import Reply, Request, ScriptedGenai, call_part, live_call_reply, reply
from .test_graph import (
    _BRIEF as _GRAPH_BRIEF,
)
from .test_graph import (
    _PLAN_WINDOW,
    ScriptedExecutor,
    _hashed_scope,
    _output_of,
    _pending_interrupt,
    _row_id,
    _scoped_plan,
    _session_state,
    _switched_answer,
)
from .test_planner_model import CACHE_NAME, _Caches, _with_caches
from .test_planner_node import (
    _MODEL,
    _QUESTION,
    _REFUSED,
    _SIGNED,
    _SPEC,
    _SSE,
    Runner,
    _ask_call,
    _assert_same,
    _both,
    _dumped,
    _entries,
    _Entry,
    _frames,
    _numbered_call_ids,
    _planned,
    _planner_events,
    _proposals,
    _referral,
    _Script,
    _seam,
    _serving_proposals,
    _without_trailing_empty_text,
)
from .test_planner_seam import (
    _APP,
    _PICKED,
    _USER,
    _pending_ask_interrupt_id,
    _rediscovery_refusal,
    _SeamHarness,
    _translated_plan,
    adk_request_input,
)
from .test_studio import (
    _CHOICE_ID,
    AUTH,
    CONVERSATION,
    PRINCIPAL,
    _app,
    _client,
    _events,
    _frame,
    _plan_with_a_choice,
    _stream,
    _switch_body,
)

_RUNNERS: tuple[Runner, Runner] = ("adk", "pydantic_ai")


@pytest.fixture
def plan(decision: Any, requirement: Any, derived_requirement: Any) -> Plan:
    return _translated_plan(decision, requirement, derived_requirement)


def _request_dumped(request: Request) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    return (
        request.generate_config.model_dump(mode="json", exclude_none=True),
        [content.model_dump(mode="json", exclude_none=True) for content in request.content_list],
    )


def _assert_crossed_same(crossed: _Entry, adk: _Entry, pydantic: _Entry) -> None:
    """`crossed`, whose runs were split between the runners, lands where a turn run
    on one runner lands, and each of its requests equals the one the runner that
    sent it sent in a turn run on it alone."""
    _assert_same(adk, pydantic)
    assert crossed.outputs == adk.outputs
    assert crossed.reports == adk.reports
    assert set(crossed.senders) == set(_RUNNERS), "the turn crossed from one runner to the other"
    alone = {"adk": adk, "pydantic_ai": pydantic}
    assert len(crossed.requests) == len(adk.requests)
    for position, (request, sender) in enumerate(
        zip(crossed.requests, crossed.senders, strict=True)
    ):
        assert _request_dumped(request) == _request_dumped(alone[sender].requests[position])
    assert _dumped(crossed.events) == _dumped(adk.events)
    assert list(session_steps(crossed.events)) == list(session_steps(adk.events))
    assert crossed.state == adk.state


async def _crossed(runners: Sequence[Runner], script: _Script) -> tuple[_Entry, _Entry, _Entry]:
    """`script` with its runs on `runners`, then on each runner alone."""
    with _numbered_call_ids():
        crossed = await _entries(runners, script)
    adk, pydantic = await _both(script)
    return crossed, adk, pydantic


def _mixed(runs: int) -> list[tuple[Runner, ...]]:
    """Every order of `runs` runs that uses both runners."""
    return [order for order in itertools.product(_RUNNERS, repeat=runs) if len(set(order)) == 2]


# --- The setting --------------------------------------------------------------------------


@pytest.mark.parametrize(("runner", "agents"), [("adk", 1), ("pydantic_ai", 0)])
def test_the_setting_selects_the_planner_and_only_the_adk_planner_builds_an_agent(
    runner: Runner, agents: int
) -> None:
    with mock.patch.object(planner_agent_module, "LlmAgent", wraps=LlmAgent) as built:
        seam = PlannerSeam(
            settings=Settings(planner_runner=runner), surface=FixtureSurface(schemas=[])
        )

    assert built.call_count == agents
    assert (seam._agent is not None) is (runner == "adk")


def test_the_adk_planner_serves_the_tools_and_instruction_the_seam_built() -> None:
    seam = PlannerSeam(settings=Settings(planner_runner="adk"), surface=FixtureSurface(schemas=[]))

    assert seam._agent is not None
    assert [tool.spec for tool in seam._agent.tools] == list(seam._specs)
    assert seam._agent.instruction == seam._instruction


def test_the_pydantic_ai_planner_refuses_an_adk_model() -> None:
    with pytest.raises(ValueError, match="ADK planner only"):
        PlannerSeam(
            settings=Settings(planner_runner="pydantic_ai"),
            surface=FixtureSurface(schemas=[]),
            model=Gemini(model=_MODEL),
        )


@pytest.mark.parametrize("named", [_MODEL, f"Gemini:{_MODEL}"])
async def test_both_planners_call_the_gemini_model_adks_registry_resolves(
    plan: Plan, named: str
) -> None:
    """A planner model name ADK's registry resolves, a class prefix included,
    names the same model in every request on both planners."""
    sent: dict[Runner, list[str]] = {}
    for runner in ("adk", "pydantic_ai"):
        scripted = ScriptedGenai([reply(call_part(PROPOSE_PLAN, {}, id="p1", signature=_SIGNED))])
        surface = FixtureSurface(schemas=[])
        seam = PlannerSeam(
            settings=Settings(planner_runner=runner),
            agent_settings=AgentSettings(planner_model_name=named),
            surface=surface,
            specs=_proposals()(
                seam_specs(surface=surface, translate=build_translate(surface=surface))
            ),
            client=scripted.client,
        )
        await _SeamHarness().run(seam, run_config=_SSE, readings=[], **_planned(plan))
        sent[runner] = [request.model for request in scripted.requests]

    assert sent["pydantic_ai"] == sent["adk"] == [_MODEL]


async def test_both_planners_send_one_request_form_and_share_one_prefix_cache(plan: Plan) -> None:
    """A planned turn on the ADK planner, then the same turn on the Pydantic AI
    planner over one `PrefixCache`: the first creates the cache, the second
    finds it by the same key and sends against it."""
    caches = _Caches()
    scripted = _with_caches(
        ScriptedGenai(
            [reply(call_part(PROPOSE_PLAN, {}, id=f"p{n}", signature=_SIGNED)) for n in (1, 2)]
        ),
        caches,
    )
    cache = PrefixCache()
    surface = FixtureSurface(schemas=[])
    specs = _proposals()(seam_specs(surface=surface, translate=build_translate(surface=surface)))
    for runner in _RUNNERS:
        seam = PlannerSeam(
            settings=Settings(planner_runner=runner),
            surface=surface,
            specs=specs,
            prefix_cache=cache,
            client=scripted.client,
        )
        await _SeamHarness().run(seam, run_config=_SSE, readings=[], **_planned(plan))

    assert len(caches.created) == 1
    assert [request.generate_config.cached_content for request in scripted.requests] == [
        CACHE_NAME,
        CACHE_NAME,
    ]
    adk, pydantic = scripted.requests
    assert_pinned_wire_differences(adk, pydantic)


def _in_wire_order(config: types.GenerateContentConfig | types.CreateCachedContentConfig) -> str:
    """`config` as JSON in the order its fields and lists go on the wire, less the
    HTTP options each framework sets its own."""
    dumped = config.model_dump(mode="json", exclude_none=True)
    dumped.pop("http_options", None)
    return json.dumps(dumped)


@pytest.mark.parametrize("cached", [False, True], ids=["plain-request", "cache-create"])
async def test_both_planners_send_the_same_config_in_wire_order(plan: Plan, cached: bool) -> None:
    """A planned turn's first request, or the create of its prefix cache, holds
    the same config on both planners, field for field and list for list in the
    order sent, and the two key the prefix cache alike. Equal keys alone would
    not show the order, since the key sorts."""
    sent: dict[Runner, str] = {}
    keys: dict[Runner, str | None] = {}
    surface = FixtureSurface(schemas=[])
    specs = _proposals()(seam_specs(surface=surface, translate=build_translate(surface=surface)))
    for runner in _RUNNERS:
        caches = _Caches()
        scripted = _with_caches(
            ScriptedGenai([reply(call_part(PROPOSE_PLAN, {}, id="p1", signature=_SIGNED))]),
            caches,
        )
        seam = PlannerSeam(
            settings=Settings(planner_runner=runner),
            surface=surface,
            specs=specs,
            prefix_cache=PrefixCache() if cached else None,
            client=scripted.client,
        )
        await _SeamHarness().run(seam, run_config=_SSE, readings=[], **_planned(plan))
        (request, *_) = scripted.requests
        if cached:
            (created,) = caches.created
            sent[runner] = _in_wire_order(created)
            whole = types.GenerateContentConfig(
                system_instruction=created.system_instruction,
                tools=created.tools,
                tool_config=created.tool_config,
            )
            keys[runner] = prefix_key(request.model, whole)
        else:
            sent[runner] = _in_wire_order(request.generate_config)
            keys[runner] = prefix_key(request.model, request.generate_config)

    assert sent["pydantic_ai"] == sent["adk"]
    assert keys["adk"] is not None and keys["pydantic_ai"] == keys["adk"]


def test_the_pydantic_ai_planner_refuses_a_model_that_is_not_gemini() -> None:
    with pytest.raises(ValueError, match="Gemini model only"):
        PlannerSeam(
            settings=Settings(planner_runner="pydantic_ai"),
            agent_settings=AgentSettings(planner_model_name="claude-3-5-sonnet"),
            surface=FixtureSurface(schemas=[]),
        )


def _genai_client(*, vertexai: bool) -> Client:
    """A google-genai client that uses Vertex AI or the Gemini developer API,
    built without credentials."""
    client = mock.create_autospec(Client, instance=True)
    client.vertexai = vertexai
    return client


# A Gemini model named by its Vertex AI resource path.
_VERTEX_MODEL = f"projects/p/locations/global/publishers/google/models/{_MODEL}"


_ON_VERTEX = "client uses Vertex AI"
_RESOURCE_PATH = "Vertex AI resource path"


@pytest.mark.parametrize("runner", ["adk", "pydantic_ai"])
@pytest.mark.parametrize(
    ("environment", "model_name", "vertexai", "refusals"),
    [
        pytest.param({}, _MODEL, None, {}, id="default-client-developer-api"),
        pytest.param(
            {"GOOGLE_GENAI_USE_VERTEXAI": "1"},
            _MODEL,
            None,
            {"adk": _ON_VERTEX, "pydantic_ai": _ON_VERTEX},
            id="default-client-vertex",
        ),
        pytest.param(
            {},
            _VERTEX_MODEL,
            None,
            {"adk": _ON_VERTEX, "pydantic_ai": _RESOURCE_PATH},
            id="default-client-vertex-model-name",
        ),
        pytest.param(
            {},
            _VERTEX_MODEL,
            False,
            {"pydantic_ai": _RESOURCE_PATH},
            id="developer-client-vertex-model-name",
        ),
        pytest.param(
            {}, _MODEL, True, {"adk": _ON_VERTEX, "pydantic_ai": _ON_VERTEX}, id="client-on-vertex"
        ),
        pytest.param({"GOOGLE_GENAI_USE_VERTEXAI": "1"}, _MODEL, False, {}, id="client-decides"),
    ],
)
def test_both_planners_refuse_a_model_that_would_call_vertex_ai(
    monkeypatch: pytest.MonkeyPatch,
    runner: Runner,
    environment: dict[str, str],
    model_name: str,
    vertexai: bool | None,
    refusals: dict[Runner, str],
) -> None:
    """A given client decides by its own `vertexai`. With none, the client the
    runner builds decides: ADK's `Gemini` sends a `projects/...` name to Vertex
    AI, and otherwise the environment decides. The Pydantic AI planner runs on
    the developer API, so it refuses a `projects/...` name, a Vertex AI resource
    path, whatever the client."""
    for name in ("GOOGLE_GENAI_USE_ENTERPRISE", "GOOGLE_GENAI_USE_VERTEXAI"):
        monkeypatch.delenv(name, raising=False)
    for name, value in environment.items():
        monkeypatch.setenv(name, value)
    client = _genai_client(vertexai=vertexai) if vertexai is not None else None

    def build() -> PlannerSeam:
        return PlannerSeam(
            settings=Settings(planner_runner=runner),
            agent_settings=AgentSettings(planner_model_name=model_name),
            surface=FixtureSurface(schemas=[]),
            client=client,
        )

    refusal = refusals.get(runner)
    if refusal is not None:
        with pytest.raises(RuntimeError, match=refusal):
            build()
    else:
        build()


def test_the_adk_planner_runs_a_model_given_in_place_of_the_named_one_unchecked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A `model` replacing the settings' model calls through no client the seam
    builds, so the Vertex AI rule for the replaced model's name refuses nothing."""
    monkeypatch.setenv("GOOGLE_GENAI_USE_VERTEXAI", "1")
    given = Gemini(model=_MODEL)

    seam = PlannerSeam(
        settings=Settings(planner_runner="adk"),
        agent_settings=AgentSettings(planner_model_name=_VERTEX_MODEL),
        surface=FixtureSurface(schemas=[]),
        model=given,
    )

    assert seam._agent is not None and seam._agent.model is given


@pytest.mark.parametrize(
    ("model_name", "refused"),
    [("claude-3-5-sonnet", ValueError), (_VERTEX_MODEL, RuntimeError)],
    ids=["not-gemini", "vertex-resource-path"],
)
def test_the_pydantic_ai_planners_model_is_refused_before_the_seam_builds_anything(
    model_name: str, refused: type[Exception]
) -> None:
    with (
        mock.patch.object(planner_seam, "build_tool_surface") as surface,
        mock.patch.object(planner_seam, "planner_specs") as specs,
        pytest.raises(refused),
    ):
        PlannerSeam(
            settings=Settings(planner_runner="pydantic_ai"),
            agent_settings=AgentSettings(planner_model_name=model_name),
        )

    surface.assert_not_called()
    specs.assert_not_called()


@pytest.mark.parametrize(
    "instruction",
    [
        "Plan. Then propose.",
        types.Content(
            role="user", parts=[types.Part(text="Plan. "), types.Part(text="Then propose.")]
        ),
        types.Part(text="Plan. Then propose."),
        {"role": "user", "parts": [{"text": "Plan. "}, {"text": "Then propose."}]},
        {"text": "Plan. Then propose."},
        [types.Part(text="Plan. "), {"text": "Then propose."}],
    ],
    ids=["str", "content", "part", "content-dict", "part-dict", "list"],
)
def test_an_instructions_text_reads_the_same_in_every_shape(instruction: object) -> None:
    assert instruction_text(instruction) == "Plan. Then propose."


@pytest.mark.parametrize("runner", _RUNNERS)
async def test_the_seams_observer_sees_each_answer_the_model_reads(
    plan: Plan, runner: Runner
) -> None:
    """`on_tool_answered` is told each planner tool's name and answer once: a
    refused proposal, an ask refused for a missing argument, a proposal whose
    body raised and, in the next entry, an accepted one. Each is the function
    response the session stores for the call, the one the model reads."""
    raised = RuntimeError("broken")
    calls = itertools.count()

    def body(call: PlannerCall, end_run: EndRun, arguments: dict[str, Any]) -> object:
        match next(calls):
            case 0:
                return _REFUSED
            case 1:
                raise raised
            case _:
                return _compiled_proposal(call, end_run, arguments)

    scripted = ScriptedGenai(
        [
            reply(
                call_part(PROPOSE_PLAN, {}, id="p1", signature=_SIGNED),
                call_part(PLANNER_ASK, {}, id="a1"),
            ),
            reply(call_part(PROPOSE_PLAN, {}, id="p2", signature=_SIGNED)),
            reply(call_part(PROPOSE_PLAN, {}, id="p3", signature=_SIGNED)),
        ]
    )
    surface = FixtureSurface(schemas=[])
    specs = seam_specs(surface=surface, translate=build_translate(surface=surface))
    observed: list[tuple[str, dict[str, Any] | None]] = []
    seam = PlannerSeam(
        settings=Settings(planner_runner=runner),
        surface=surface,
        specs=_serving_proposals(specs, body),
        client=scripted.client,
        on_tool_answered=lambda name, answer: observed.append((name, answer)),
    )
    stored: list[tuple[str | None, dict[str, Any] | None]] = []
    for _ in range(2):
        harness = _SeamHarness()
        await harness.run(seam, run_config=_SSE, readings=[], **_planned(plan))
        stored += [
            (response.name, response.response)
            for event in _planner_events(_Entry([], [], [], await harness.events(), {}))
            for response in event.get_function_responses()
        ]

    def key(item: tuple[str | None, dict[str, Any] | None]) -> str:
        return repr(item)

    assert sorted(observed, key=key) == sorted(stored, key=key)
    answers = dict(observed[:2])
    assert answers[PROPOSE_PLAN] == _REFUSED
    assert answers[PLANNER_ASK] is not None and set(answers[PLANNER_ASK]) == {"error"}
    assert observed[2] == (PROPOSE_PLAN, failed_tool_answer(PROPOSE_PLAN, raised))
    assert observed[3][1] is not None and observed[3][1]["outcome"] == Outcome.OK
    assert len(observed) == 4


# --- Turns that cross runners -------------------------------------------------------------


@pytest.mark.parametrize("shape", [reply, live_call_reply], ids=["one-chunk", "live-shaped"])
@pytest.mark.parametrize("runners", _mixed(2), ids="-then-".join)
async def test_an_ask_answered_under_the_other_runner_lands_the_same_plan(
    plan: Plan, runners: tuple[Runner, ...], shape: Callable[..., Reply]
) -> None:
    """A live-shaped reply (`live_call_reply`) leaves ADK's trailing empty text
    part on a model turn the ADK planner stores, which the other runner's resume
    reads as it reads its own, and the turn lands the same plan; with that part
    taken off the ADK planner's side, everything else is equal."""
    crossed, adk, pydantic = await _crossed(
        runners,
        _Script(
            replies=[
                shape(_ask_call("a1")),
                shape(call_part(PROPOSE_PLAN, {}, id="p1", signature=_SIGNED)),
            ],
            runs=[_planned(plan), {**_planned(plan), "answer": {"result": "2020"}}],
            setup=_proposals(),
        ),
    )

    assert adk.outputs[0] is None, "the ask pauses the first run"
    assert adk.outputs[1] is not None and "plan" in adk.outputs[1]
    if shape is live_call_reply:
        adk, taken = _without_trailing_empty_text(adk)
        crossed, _ = _without_trailing_empty_text(crossed)
        assert taken > 0
    _assert_crossed_same(crossed, adk, pydantic)


@pytest.mark.parametrize("runners", _mixed(3), ids="-then-".join)
async def test_two_asks_answered_across_runners_land_the_same_plan(
    plan: Plan, runners: tuple[Runner, ...]
) -> None:
    crossed, adk, pydantic = await _crossed(
        runners,
        _Script(
            replies=[
                reply(_ask_call("a1")),
                reply(_ask_call("a2", message="Which years?")),
                reply(call_part(PROPOSE_PLAN, {}, id="p1", signature=_SIGNED)),
            ],
            runs=[
                _planned(plan),
                {**_planned(plan), "answer": {"result": "external debt"}},
                {**_planned(plan), "answer": {"result": "2020"}},
            ],
            setup=_proposals(),
        ),
    )

    assert adk.outputs[:2] == [None, None], "each ask pauses its run"
    assert adk.outputs[2] is not None and "plan" in adk.outputs[2]
    _assert_crossed_same(crossed, adk, pydantic)


async def _legacy_answered(runner: Runner, plan: Plan) -> _Entry:
    """A turn the ADK planner paused on its long-running `adk_request_input` call,
    answered on `runner`, whose seam is built afresh."""
    question = {"message": "Which source did you mean?"}
    scripted = ScriptedGenai(
        [
            reply(call_part("adk_request_input", question, id="r1", signature=_SIGNED)),
            reply(call_part(PROPOSE_PLAN, {}, id="p1", signature=_SIGNED)),
        ]
    )
    harness = _SeamHarness()
    paused = _seam("adk", scripted, tools=_proposals())
    assert paused._agent is not None
    paused._agent.tools = [*paused._agent.tools, LongRunningFunctionTool(adk_request_input)]
    reports: list[dict[str, Any]] = []
    with _numbered_call_ids():
        outputs = [
            await harness.run(
                paused, on_progress=reports.append, run_config=_SSE, readings=[], **_planned(plan)
            )
        ]
        answer = types.Content(
            role="user",
            parts=[
                create_request_input_response(
                    _pending_ask_interrupt_id(await harness.events()), {"result": "2020"}
                )
            ],
        )
        senders: list[Runner] = ["adk"] * len(scripted.requests)
        resumed = _seam(runner, scripted, tools=_proposals())
        outputs.append(
            await harness.run(
                resumed,
                on_progress=reports.append,
                run_config=_SSE,
                new_message=answer,
                readings=[],
                **_planned(plan),
            )
        )
        senders += [runner] * (len(scripted.requests) - len(senders))
    return _Entry(
        outputs=outputs,
        reports=[{k: v for k, v in report.items() if k != "duration_ms"} for report in reports],
        requests=scripted.requests,
        events=await harness.events(),
        state=dict(await harness.state()),
        senders=senders,
    )


async def test_a_legacy_ask_answered_under_the_pydantic_ai_runner_lands_the_adk_planners_plan(
    plan: Plan,
) -> None:
    """A turn paused on the planner's `adk_request_input` call, as the ADK planner
    paused before its node paused on its asks, holds no pending ask. Its answer
    re-enters the Pydantic AI planner in a fresh run, as it re-enters the ADK
    planner."""
    adk = await _legacy_answered("adk", plan)
    pydantic = await _legacy_answered("pydantic_ai", plan)

    assert adk.outputs[0] is None, "the legacy ask pauses the first run"
    assert adk.outputs[1] is not None and "plan" in adk.outputs[1]
    assert pydantic.senders[-1] == "pydantic_ai"
    assert pydantic.outputs == adk.outputs
    assert pydantic.reports == adk.reports
    for position, (adk_request, pydantic_request) in enumerate(
        zip(adk.requests, pydantic.requests, strict=True)
    ):
        if pydantic.senders[position] == "adk":
            assert _request_dumped(pydantic_request) == _request_dumped(adk_request)
        else:
            assert_pinned_wire_differences(adk_request, pydantic_request)
    assert _dumped(pydantic.events) == _dumped(adk.events)
    assert pydantic.state == adk.state


# --- Amendments and replans ---------------------------------------------------------------

_AMENDMENT = "Show one total for the whole period."


def _amended(plan: Plan) -> dict[str, Any]:
    return {**_planned(plan), "amendment": _AMENDMENT}


def _text(request: Request) -> str:
    return "".join(
        part.text or "" for content in request.content_list for part in content.parts or ()
    )


@pytest.mark.parametrize("scenario", ["amendment", "ask-during-amendment"])
async def test_an_amendment_is_the_same_on_both_planners(plan: Plan, scenario: str) -> None:
    """An amendment opens a turn of its own, so its entry is its session's first."""
    proposed = reply(call_part(PROPOSE_PLAN, {}, id="p1", signature=_SIGNED))
    scripts = {
        "amendment": _Script(replies=[proposed], runs=[_amended(plan)], setup=_proposals()),
        "ask-during-amendment": _Script(
            replies=[reply(_ask_call("a1")), proposed],
            runs=[_amended(plan), {**_amended(plan), "answer": {"result": "2020"}}],
            setup=_proposals(),
        ),
    }

    adk, pydantic = await _both(scripts[scenario])

    assert adk.outputs[-1] is not None and "plan" in adk.outputs[-1], adk.outputs
    assert all(_AMENDMENT in _text(request) for request in adk.requests)
    _assert_same(adk, pydantic)


async def _replanned(runner: Runner, plan: Plan, opened_by: Mapping[str, Any]) -> _Entry:
    """A first entry and a replan `opened_by` its refusal or pick, run in one
    invocation of one session, as the graph's planner node runs a replan in the
    turn it planned in. The replan's run has a node path of its own and follows
    no other user message."""
    scripted = ScriptedGenai(
        [
            reply(call_part(PROPOSE_PLAN, {}, id="p1", signature=_SIGNED)),
            reply(call_part(PROPOSE_PLAN, {}, id="p2", signature=_SIGNED)),
        ]
    )
    seam = _seam(runner, scripted, tools=_proposals())
    reports: list[dict[str, Any]] = []
    outputs: list[dict[str, Any] | None] = []

    async def entries(ctx: Context, node_input: Any) -> AsyncIterator[Event]:
        del node_input
        entry = {"ctx": ctx, "readings": [], "on_progress": reports.append, **_planned(plan)}
        first = await seam(**entry)
        outputs.extend([first, await seam(**entry, **opened_by)])
        yield Event(output={"entries": len(outputs)})

    sessions = InMemorySessionService()
    session = await sessions.create_session(app_name=_APP, user_id=_USER)
    workflow = Workflow(
        name="replan_wf", edges=[(START, node(entries, name="entries", rerun_on_resume=True))]
    )
    # The refusal's time is recorded with it (`_record_refusal`). One clock serves
    # both runners.
    clock = mock.Mock(wraps=datetime)
    clock.now.return_value = datetime(2026, 11, 11, tzinfo=UTC)
    with _numbered_call_ids(), mock.patch.object(planner_seam, "datetime", clock):
        async for _ in AdkRunner(node=workflow, app_name=_APP, session_service=sessions).run_async(
            user_id=_USER,
            session_id=session.id,
            new_message=types.Content(role="user", parts=[types.Part(text=_QUESTION)]),
            run_config=_SSE,
        ):
            pass
    stored = await sessions.get_session(app_name=_APP, user_id=_USER, session_id=session.id)
    assert stored is not None
    return _Entry(
        outputs=outputs,
        reports=[{k: v for k, v in report.items() if k != "duration_ms"} for report in reports],
        requests=scripted.requests,
        events=stored.events,
        state=dict(stored.state),
    )


@pytest.mark.parametrize(
    "opened_by",
    [{"refusal": _rediscovery_refusal(_SPEC)}, {"pick": _PICKED}],
    ids=["refusal", "pick"],
)
async def test_a_replan_in_the_turn_it_planned_in_is_the_same_on_both_planners(
    plan: Plan, opened_by: dict[str, Any]
) -> None:
    adk = await _replanned("adk", plan, opened_by)
    pydantic = await _replanned("pydantic_ai", plan, opened_by)

    assert [output is not None and "plan" in output for output in adk.outputs] == [True, True]
    assert len(adk.requests) == 2
    _assert_same(adk, pydantic)


# --- Two bounds closing in one request ----------------------------------------------------


async def test_a_second_bound_closing_in_the_same_request_refuses_with_the_same_code(
    plan: Plan,
) -> None:
    """The call cap closes the first entry's second call. The request's wall
    clock then runs out before a replan the plan card's pick opens in the same
    request, whose first call the wall bound closes. `TurnBudget.closed` keeps
    the call cap; both planners refuse the replan on the wall bound that closed
    its own call."""

    def script() -> _Script:
        now = [0.0]
        budget = TurnBudget(
            model_time_s=60.0,
            model_calls_max=1,
            wall_s=30.0,
            first_chunk_s=10.0,
            clock=lambda: now[0],
        )
        handed = itertools.count()

        def shared() -> TurnBudget:
            if next(handed) == 1:
                now[0] = 31.0
            return budget

        return _Script(
            replies=[reply(call_part(PROPOSE_PLAN, {}, id="p1", signature=_SIGNED))],
            runs=[_planned(plan), {**_planned(plan), "pick": _PICKED}],
            setup=_proposals(_REFUSED),
            budget=shared,
        )

    with _numbered_call_ids():
        adk = await _entries("adk", script())
    with _numbered_call_ids():
        pydantic = await _entries("pydantic_ai", script())

    assert [output["refused"]["code"] for output in adk.outputs if output is not None] == [
        SeamCode.TURN_MODEL_CALLS_SPENT,
        SeamCode.TURN_WALL_TIME_SPENT,
    ]
    _assert_same(adk, pydantic)


# --- Call ids and signatures on the wire --------------------------------------------------


def _refusing_asks(specs: Sequence[ToolSpec]) -> list[ToolSpec]:
    """`specs` with an ask tool that refuses every ask, and `propose_plan`
    answering as `_proposals` does after a referral the user decides."""
    referral = _referral({"spec_id": _SPEC, "sentence": "external debt"})
    referral["translation"]["kind"] = "source_name_ambiguous"
    referral["translation"]["ambiguous_sources"] = [
        {"candidates": [{"ref": _SPEC, "label": "external debt"}]}
    ]

    def refuses(call: PlannerCall, end_run: EndRun, arguments: dict[str, Any]) -> object:
        del call, end_run, arguments
        return {
            "outcome": "refused",
            "code": ProblemCode.INVALID_ARGUMENTS.value,
            "message": "Ask in your own words.",
        }

    ask = ToolSpec(name=PLANNER_ASK, description="Ask the user.", parameters=(), body=refuses)
    return [ask if spec.name == PLANNER_ASK else spec for spec in _proposals(referral)(specs)]


async def test_a_refused_ask_put_in_the_models_place_is_replayed_without_its_call_id(
    plan: Plan,
) -> None:
    """A referral the user decides puts an ask in the model's place under an
    `adk-` call id. The ask tool refuses it, the run goes on, and the next
    request replays the ask and its answer with ADK's own call id removed and
    the placeholder signature kept, on both planners."""
    adk, pydantic = await _both(
        _Script(
            replies=[
                reply(call_part(PROPOSE_PLAN, {}, id="p1", signature=_SIGNED)),
                reply(call_part(PROPOSE_PLAN, {}, id="p2", signature=_SIGNED)),
            ],
            runs=[_planned(plan)],
            setup=_refusing_asks,
        )
    )

    for entry in (adk, pydantic):
        assert len(entry.requests) == 2
        (asked,) = [
            part
            for content in entry.requests[1].content_list
            for part in content.parts or ()
            if part.function_call is not None and part.function_call.name == PLANNER_ASK
        ]
        assert asked.function_call is not None and asked.function_call.id is None
        assert asked.thought_signature == SKIP_THOUGHT_SIGNATURE
        (answered,) = [
            part
            for content in entry.requests[1].content_list
            for part in content.parts or ()
            if part.function_response is not None and part.function_response.name == PLANNER_ASK
        ]
        assert answered.function_response is not None and answered.function_response.id is None
    (stored,) = [
        call
        for event in _planner_events(adk)
        for call in event.get_function_calls()
        if call.name == PLANNER_ASK
    ]
    assert stored.id == "adk-1", "the stored ask keeps its id"
    _assert_same(adk, pydantic)


async def test_an_unsigned_first_call_is_replayed_with_the_placeholder_only_by_pydantic_ai(
    plan: Plan,
) -> None:
    """Gemini signs the first function call of a response it thought before. One
    it sent unsigned, the ADK planner replays with no signature and the Pydantic AI
    planner with Gemini's placeholder. `assert_pinned_wire_differences` pins
    that one difference."""
    adk, pydantic = await _both(
        _Script(
            replies=[
                reply(call_part(PROPOSE_PLAN, {}, id="p1")),
                reply(call_part(PROPOSE_PLAN, {}, id="p2", signature=_SIGNED)),
            ],
            runs=[_planned(plan)],
            setup=_proposals(_REFUSED),
        )
    )

    def replayed(entry: _Entry) -> types.Part:
        (part,) = [
            part
            for content in entry.requests[1].content_list
            for part in content.parts or ()
            if part.function_call is not None and part.function_call.id == "p1"
        ]
        return part

    assert replayed(adk).thought_signature is None
    assert replayed(pydantic).thought_signature == SKIP_THOUGHT_SIGNATURE
    _assert_same(adk, pydantic)


async def test_calls_without_ids_are_stored_and_replayed_alike_on_both_planners(
    plan: Plan,
) -> None:
    """Gemini sends its function calls without ids. Both planners give each one
    an `adk-` id, store the call and its answer under it, and send the next
    request without it."""
    adk, pydantic = await _both(
        _Script(
            replies=[
                reply(call_part(PROPOSE_PLAN, {}, signature=_SIGNED)),
                reply(call_part(PROPOSE_PLAN, {}, signature=_SIGNED)),
            ],
            runs=[_planned(plan)],
            setup=_proposals(_REFUSED),
        )
    )

    for entry in (adk, pydantic):
        ids = [call.id for event in _planner_events(entry) for call in event.get_function_calls()]
        assert ids == ["adk-1", "adk-2"]
        assert all(
            part.function_call.id is None
            for content in entry.requests[1].content_list
            for part in content.parts or ()
            if part.function_call is not None
        )
    _assert_same(adk, pydantic)


# --- A Stop while a switch runs -----------------------------------------------------------


async def _stopped_switch(
    runner: Runner, plan: Plan, monkeypatch: pytest.MonkeyPatch
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """The frames of a turn whose plan the planner on `runner` drafted, and of a
    switch on its card that the user stops while the revision runs."""
    entered = asyncio.Event()

    async def revising(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        entered.set()
        await asyncio.sleep(5)
        raise AssertionError("a stopped revision is cancelled")

    monkeypatch.setattr(switch_module, "_switched_plan", revising)
    monkeypatch.setattr(studio_module, "SWITCH_REOFFER_SECONDS", 0.2)
    scripted = ScriptedGenai([reply(call_part(PROPOSE_PLAN, {}, id="p1", signature=_SIGNED))])
    seam = _seam(runner, scripted, tools=_proposals(records=plan))
    app = _app(plan_call(_QUESTION), planner=seam)
    stop = {"kind": "turn.cancel", "schema_version": 1, "conversation_id": CONVERSATION}
    async with _client(app) as client:
        with _numbered_call_ids():
            proposed = await _plan_with_a_choice(app, client)
        conversation = app.state.studio.store.current(PRINCIPAL, CONVERSATION)
        resumed = asyncio.ensure_future(
            _stream(client, "/api/studio/interrupt", _switch_body(proposed))
        )
        await asyncio.wait_for(entered.wait(), timeout=5)
        armed = conversation.cancel_event
        assert armed is not None and not armed.is_set()
        cancelled = await client.post("/api/studio/cancel", json=stop, headers=AUTH)
        assert cancelled.status_code == 204
        frames = await asyncio.wait_for(resumed, timeout=5)

    assert _events(frames)[-1] == "plan.proposed"
    again = _frame(frames, "plan.proposed")
    assert again["interrupt_id"] != proposed["interrupt_id"]
    assert again["revision"] == proposed["revision"]
    assert again["switch_failed"]["choice_id"] == _CHOICE_ID
    assert again["switch_failed"]["code"] == "switch_cancelled"
    assert list(conversation.pending) == [again["interrupt_id"]]
    assert conversation.awaiting_answer
    assert conversation.switch_stop is None
    return [proposed], frames


async def test_a_stop_while_a_switch_runs_offers_the_plan_unchanged_after_either_planner(
    plan: Plan, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The plan card a Pydantic AI entry drafted takes a switch and a Stop as the
    ADK planner's card does. The stream ends on the same plan re-offered under a
    fresh interrupt, carrying `switch_cancelled`."""
    adk = await _stopped_switch("adk", plan, monkeypatch)
    pydantic = await _stopped_switch("pydantic_ai", plan, monkeypatch)

    assert [_frames(frames) for frames in pydantic] == [_frames(frames) for frames in adk]


# --- A switch through the graph after an entry ---------------------------------------------


@pytest.fixture
async def real_schemas() -> list[ToolSchema]:
    """The real servers' tool schemas (`tests.test_graph.real_schemas`)."""
    return await McpSurface(
        sources={"data-mcp": data_mcp, "analysis-mcp": analysis_mcp}
    ).list_tools()


async def _graph_turn(
    graph: AdkRunner, interrupt_id: str | None = None, answer: dict[str, Any] | None = None
) -> list[Event]:
    """One turn of `graph`'s session under Studio's streaming run config: the
    brief that opens it, or `answer` to the pause `interrupt_id`."""
    if interrupt_id is None:
        await open_session(graph)
        message = types.UserContent(_GRAPH_BRIEF)
    else:
        part = create_request_input_response(interrupt_id, {"result": answer})
        message = types.Content(role="user", parts=[part])
    return [
        event
        async for event in graph.run_async(
            user_id="u", session_id="s", new_message=message, run_config=_SSE
        )
    ]


async def _switched_after_an_entry(
    runner: Runner, schemas: list[ToolSchema]
) -> tuple[list[Event], list[Event], Any, str]:
    """A turn whose planner, on `runner`, asks which source the user means, is
    answered with the scope's pin, and drafts the plan pinning it; the plan card
    then takes a sector switch through the graph's real `switch` node, which
    re-pins the plan through `PlannerSeam.revise` with no model call. Returns the
    switch's events, the session as stored, its state with each time it records
    put as `<time>`, and the pin the revised plan reads, which is the copy, over the
    plan's years, of the one Search answered the switch with."""
    scope = _hashed_scope()
    pin = scope["compiled"]["spec_id"]
    scoped = _scoped_plan(pin, scope, schemas=schemas)
    # The plan answers the brief's one ask, as the switch's revision checks.
    plan = Plan.model_validate(
        {
            **scoped.model_dump(),
            "asks": [_GRAPH_BRIEF],
            "answered": [{"ask": 0, "outputs": ["totals"]}],
        }
    )
    _, envelope = _switched_answer(scope)
    # Admission copies the pin Search answered over the years the plan reads.
    copy_pin, _ = windowed_identity(
        envelope["parts"][0]["spec"]["definition"]["identity"], _PLAN_WINDOW
    )

    async def translate(call: PlannerCall) -> dict[str, object]:
        draft = load_plan(call.state)
        assert draft is not None
        assert plan.translation is not None and plan.workflow is not None
        save_plan(
            call.state,
            draft.with_translation(translation=plan.translation, workflow=plan.workflow),
        )
        return {"outcome": "ok"}

    scripted = ScriptedGenai(
        [
            reply(_ask_call("a1", candidates=[{"spec_id": pin, "label": "health aid"}])),
            reply(call_part(PROPOSE_PLAN, {}, id="p1", signature=_SIGNED)),
        ]
    )
    surface = FixtureSurface(schemas=[])
    seam = PlannerSeam(
        settings=Settings(planner_runner=runner),
        surface=surface,
        translate=translate,
        specs=_proposals(records=plan)(seam_specs(surface=surface, translate=translate)),
        client=scripted.client,
    )
    graph, _ = orchestrator_runner(
        plan_call(_GRAPH_BRIEF),
        planner=seam,
        executor=ScriptedExecutor(),
        definition_surface=DataMcpDefinitions(
            InMemoryDefinitionStore(), search=SearchStub(answer={scope["id"]: envelope})
        ),
    )
    with _numbered_call_ids():
        asked = await _graph_turn(graph)
        ask = _pending_interrupt(asked)
        assert ask is not None and ask != APPROVE_INTERRUPT, "the planner's ask pauses the turn"
        carded = await _graph_turn(graph, ask, {"spec_id": pin})
        assert _pending_interrupt(carded) == APPROVE_INTERRUPT
        reading = SearchReading.from_part(
            {"variable": "health aid", "status": "definite", "spec_id": pin, "scope": scope}
        )
        _session_state(graph)[ORCHESTRATOR_SEARCHES_KEY] = [reading.model_dump(mode="json")]
        switched = await _graph_turn(
            graph,
            APPROVE_INTERRUPT,
            {
                "approved": None,
                "switch": {"choice_id": _row_id(plan, "sector"), "option_id": "sector:120+130"},
            },
        )
    stored = await graph.session_service.get_session(
        app_name=graph.app_name, user_id="u", session_id="s"
    )
    assert stored is not None
    return switched, stored.events, _timeless(stored.state), copy_pin


# An ISO time a turn records as it runs (a reading's `observed_at`, a resolve's time).
_RECORDED_TIME = re.compile(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d(\.\d+)?(Z|\+00:00)")


def _untimed(events: Sequence[Event]) -> Any:
    """`events` dumped as `_dumped` dumps them, less each progress line's
    duration and with each time recorded in them put as `<time>`."""
    dumped = _dumped(events)
    for event in dumped:
        event.get("custom_metadata", {}).get("progress", {}).pop("duration_ms", None)
    return _timeless({"events": dumped})["events"]


def _timeless(state: Mapping[str, Any]) -> Any:
    """`state` as JSON, each time recorded in it put as `<time>`."""
    return json.loads(_RECORDED_TIME.sub("<time>", json.dumps(dict(state), default=str)))


async def test_a_switch_after_an_entry_revises_the_same_plan_on_both_planners(
    real_schemas: list[ToolSchema],
) -> None:
    """The switch node's revision reads the entry's stored ask and answer
    (`PlannerSeam.revise`), so the sector decision it makes is the user's and
    the revised plan, the turn's events and the session's state are the same
    after an entry on either planner."""
    adk_switched, adk_events, adk_state, copy_pin = await _switched_after_an_entry(
        "adk", real_schemas
    )
    _, pydantic_events, pydantic_state, _ = await _switched_after_an_entry(
        "pydantic_ai", real_schemas
    )

    offered = _output_of(adk_switched, "switch")
    assert "switch_failed" not in offered, offered.get("switch_failed")
    revised = Plan.model_validate_json(adk_state["plan"])
    (dataset,) = [r for r in revised.evidence_requirements if isinstance(r, DatasetRequirement)]
    assert dataset.source_spec_id == copy_pin
    sector = next(d for d in revised.decisions if d.slot.startswith("scope_sector_"))
    assert sector.resolved_by is ResolvedBy.USER
    assert _untimed(pydantic_events) == _untimed(adk_events)
    assert list(session_steps(pydantic_events)) == list(session_steps(adk_events))
    assert pydantic_state == adk_state
