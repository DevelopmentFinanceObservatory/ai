"""`PlannerSeam`'s planner on a scripted google-genai client: its model, its
requests and its sessions over turns that amend, replan and switch, and its
resume of the sessions in `fixtures/adk_planner_sessions`.

Each test runs a script on a scripted google-genai client through the seam. An
entry is checked against the stored expectation of its test
(`tests.planner_expectations`) on what the seam answers, the progress it reports,
the requests the model received, the events the session stores, the steps
`session_steps` reads from them and the stored state, as `tests.test_planner_node`
checks them.
"""

from __future__ import annotations

import asyncio
import itertools
import json
import re
from collections.abc import AsyncIterator, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest import mock

import pytest
from analysis_mcp.server import mcp as analysis_mcp
from compiler.surface import FixtureSurface, McpSurface, ToolSchema
from data_mcp.definition_store import InMemoryDefinitionStore
from data_mcp.definition_writes import windowed_identity
from data_mcp.server import mcp as data_mcp
from google.adk import Context, Event
from google.adk.runners import Runner as AdkRunner
from google.adk.sessions import InMemorySessionService
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
from planner.call import PlannerCall
from planner.plan import (
    DatasetRequirement,
    Plan,
    ResolvedBy,
    load_plan,
    save_plan,
)
from planner.session import session_steps
from planner.settings import AgentSettings
from planner.specs import PROPOSE_PLAN
from planner.tool_results import Outcome, ProblemCode
from planner.tool_spec import ToolSpec
from planner.tools import PENDING_ASK_STATE_KEY, EndRun
from translator.tool import build_translate

from .data_mcp_definitions import DataMcpDefinitions, SearchStub
from .orchestrator_fakes import open_session, orchestrator_runner, plan_call
from .planner.planner_pairs import seam_specs
from .planner_expectations import expect, expectation_name, expected, shared
from .scripted_genai import Request, ScriptedGenai, call_part, reply
from .scripted_planner import scripted_planner
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
    _ELECTED_KEY,
    _MODEL,
    _QUESTION,
    _READ_KEY,
    _REFUSED,
    _SIGNED,
    _SPEC,
    _SSE,
    _ask_call,
    _assert_expected,
    _Entry,
    _entry,
    _frames,
    _numbered_call_ids,
    _planned,
    _planner_events,
    _proposals,
    _read,
    _referral,
    _Script,
    _serving_proposals,
    _steps,
)
from .test_planner_seam import (
    _APP,
    _PICKED,
    _USER,
    SPEC,
    _brief_follows_the_question,
    _pending_ask_interrupt_id,
    _rediscovery_refusal,
    _SeamHarness,
    _translated_plan,
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


@pytest.fixture
def plan(decision: Any, requirement: Any, derived_requirement: Any) -> Plan:
    return _translated_plan(decision, requirement, derived_requirement)


# --- The model ----------------------------------------------------------------------------


@pytest.mark.parametrize("named", [_MODEL, f"Gemini:{_MODEL}"])
async def test_the_planner_calls_the_gemini_model_adks_registry_resolves(
    plan: Plan, named: str
) -> None:
    """A planner model name ADK's registry resolves, a class prefix included,
    names the same model in every request."""
    scripted = ScriptedGenai([reply(call_part(PROPOSE_PLAN, {}, id="p1", signature=_SIGNED))])
    surface = FixtureSurface(schemas=[])
    seam = PlannerSeam(
        settings=Settings(),
        agent_settings=AgentSettings(planner_model_name=named),
        surface=surface,
        specs=_proposals()(seam_specs(surface=surface, translate=build_translate(surface=surface))),
        client=scripted.client,
    )
    await _SeamHarness().run(seam, run_config=_SSE, readings=[], **_planned(plan))

    assert [request.model for request in scripted.requests] == [_MODEL]


async def test_two_planned_turns_over_one_prefix_cache_create_it_once(plan: Plan) -> None:
    """A planned turn, then the same turn on a seam built afresh over one
    `PrefixCache`: the first creates the cache, the second finds it by the same
    key and sends against it."""
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
    for _ in range(2):
        seam = PlannerSeam(
            settings=Settings(),
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


def _in_wire_order(config: types.GenerateContentConfig | types.CreateCachedContentConfig) -> str:
    """`config` as JSON in the order its fields and lists go on the wire, less the
    HTTP options each framework sets its own."""
    dumped = config.model_dump(mode="json", exclude_none=True)
    dumped.pop("http_options", None)
    return json.dumps(dumped)


@pytest.mark.parametrize("cached", [False, True], ids=["plain-request", "cache-create"])
async def test_a_planned_turns_config_is_sent_in_wire_order(
    plan: Plan, cached: bool, request: pytest.FixtureRequest
) -> None:
    """A planned turn's first request, or the create of its prefix cache, holds
    the stored config, field for field and list for list in the order sent, and
    keys the prefix cache with the stored key. Equal keys alone would not show
    the order, since the key sorts. The stored config is compared as the text it
    dumps to, which keeps its order."""
    surface = FixtureSurface(schemas=[])
    specs = _proposals()(seam_specs(surface=surface, translate=build_translate(surface=surface)))
    caches = _Caches()
    scripted = _with_caches(
        ScriptedGenai([reply(call_part(PROPOSE_PLAN, {}, id="p1", signature=_SIGNED))]),
        caches,
    )
    seam = PlannerSeam(
        settings=Settings(),
        surface=surface,
        specs=specs,
        prefix_cache=PrefixCache() if cached else None,
        client=scripted.client,
    )
    await _SeamHarness().run(seam, run_config=_SSE, readings=[], **_planned(plan))
    (sent, *_) = scripted.requests
    if cached:
        (created,) = caches.created
        in_order = _in_wire_order(created)
        whole = types.GenerateContentConfig(
            system_instruction=created.system_instruction,
            tools=created.tools,
            tool_config=created.tool_config,
        )
        key = prefix_key(sent.model, whole)
    else:
        in_order = _in_wire_order(sent.generate_config)
        key = prefix_key(sent.model, sent.generate_config)

    name = expectation_name(request)
    assert key is not None
    held = "wire_order_cache_create_config" if cached else "wire_order_request_config"
    expect(name, {"config": shared(json.loads(in_order), held), "key": key})
    stored = expected(name)
    assert json.dumps(stored["config"]) == in_order
    assert stored["key"] == key


def test_the_planner_refuses_a_model_that_is_not_gemini() -> None:
    with pytest.raises(ValueError, match="Gemini model only"):
        PlannerSeam(
            settings=Settings(),
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


@pytest.mark.parametrize(
    ("environment", "model_name", "vertexai", "refusal"),
    [
        pytest.param({}, _MODEL, None, None, id="default-client-developer-api"),
        pytest.param(
            {"GOOGLE_GENAI_USE_VERTEXAI": "1"}, _MODEL, None, _ON_VERTEX, id="default-client-vertex"
        ),
        pytest.param(
            {}, _VERTEX_MODEL, None, _RESOURCE_PATH, id="default-client-vertex-model-name"
        ),
        pytest.param(
            {}, _VERTEX_MODEL, False, _RESOURCE_PATH, id="developer-client-vertex-model-name"
        ),
        pytest.param({}, _MODEL, True, _ON_VERTEX, id="client-on-vertex"),
        pytest.param({"GOOGLE_GENAI_USE_VERTEXAI": "1"}, _MODEL, False, None, id="client-decides"),
    ],
)
def test_the_planner_refuses_a_model_that_would_call_vertex_ai(
    monkeypatch: pytest.MonkeyPatch,
    environment: dict[str, str],
    model_name: str,
    vertexai: bool | None,
    refusal: str | None,
) -> None:
    """A given client decides by its own `vertexai`. With none, the environment
    decides. The planner runs on the developer API, so it refuses a `projects/...`
    name, a Vertex AI resource path, whatever the client."""
    for name in ("GOOGLE_GENAI_USE_ENTERPRISE", "GOOGLE_GENAI_USE_VERTEXAI"):
        monkeypatch.delenv(name, raising=False)
    for name, value in environment.items():
        monkeypatch.setenv(name, value)
    client = _genai_client(vertexai=vertexai) if vertexai is not None else None

    def build() -> PlannerSeam:
        return PlannerSeam(
            settings=Settings(),
            agent_settings=AgentSettings(planner_model_name=model_name),
            surface=FixtureSurface(schemas=[]),
            client=client,
        )

    if refusal is not None:
        with pytest.raises(RuntimeError, match=refusal):
            build()
    else:
        build()


@pytest.mark.parametrize(
    ("model_name", "refused"),
    [("claude-3-5-sonnet", ValueError), (_VERTEX_MODEL, RuntimeError)],
    ids=["not-gemini", "vertex-resource-path"],
)
def test_the_planners_model_is_refused_before_the_seam_builds_anything(
    model_name: str, refused: type[Exception]
) -> None:
    with (
        mock.patch.object(planner_seam, "build_tool_surface") as surface,
        mock.patch.object(planner_seam, "planner_specs") as specs,
        pytest.raises(refused),
    ):
        PlannerSeam(
            settings=Settings(),
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


async def test_the_seams_observer_sees_each_answer_the_model_reads(plan: Plan) -> None:
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
        settings=Settings(),
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


# --- Sessions the ADK planner stored ------------------------------------------------------

_ADK_SESSIONS = Path(__file__).parent / "fixtures" / "adk_planner_sessions"
# The question the planner asked in each of `_ADK_SESSIONS`' sessions.
_ASKED = {"message": "Which source did you mean?"}


@pytest.mark.parametrize(
    ("recording", "call_id", "asked", "answer", "picked", "elected"),
    [
        (
            "paused_on_a_planner_ask",
            "a1",
            {**_ASKED, "candidates": [{"spec_id": SPEC, "label": "a candidate"}]},
            {"result": {"spec_id": SPEC}},
            f"They picked 'a candidate', id `{SPEC}`.",
            [SPEC],
        ),
        (
            "paused_on_adk_request_input",
            "r1",
            _ASKED,
            {"result": "2020"},
            "They picked '2020', id `2020`.",
            [],
        ),
    ],
    ids=["planner-ask", "adk-request-input"],
)
async def test_a_session_the_adk_planner_paused_on_an_ask_resumes_to_the_plan(
    plan: Plan,
    recording: str,
    call_id: str,
    asked: dict[str, Any],
    answer: dict[str, Any],
    picked: str,
    elected: list[str],
) -> None:
    """A conversation the ADK planner left paused on an ask, as it stored it
    (`fixtures/adk_planner_sessions`), resumes on the user's answer: the planner
    reads the ask and its answer from the stored events, puts the question in
    the brief, replays no empty text part, and lands the plan."""
    stored = json.loads((_ADK_SESSIONS / f"{recording}.json").read_text())
    harness = _SeamHarness()
    await harness.restore(
        [Event.model_validate(event) for event in stored["events"]], stored["state"]
    )
    scripted = ScriptedGenai([reply(call_part(PROPOSE_PLAN, {}, id="p1", signature=_SIGNED))])
    seam = scripted_planner(scripted, tools=_proposals())
    assert _pending_ask_interrupt_id(await harness.events()) == call_id
    resume = types.Content(role="user", parts=[create_request_input_response(call_id, answer)])

    output = await harness.run(seam, new_message=resume, readings=[], **_planned(plan))

    assert (
        output is not None
        and output["plan"]["workflow"] == plan.model_dump(mode="json")["workflow"]
    )
    steps = list(session_steps(await harness.events()))
    assert [(step.kind, step.call_id) for step in steps] == [
        ("message", None),
        ("call", call_id),
        ("answer", call_id),
        ("response", None),
    ]
    assert steps[1].args == steps[2].args == asked
    assert steps[2].body == answer
    state = await harness.state()
    assert state[_READ_KEY] == {
        "steps": [["message", None], ["call", call_id], ["answer", call_id]],
        "response_calls": [[PROPOSE_PLAN, "p1"]],
    }
    assert state[_ELECTED_KEY] == elected
    # The planner-ask recording holds the ask it paused on; the answer clears it.
    assert state.get(PENDING_ASK_STATE_KEY) is None
    (request,) = scripted.requests
    brief = _brief_follows_the_question(request)
    assert _ASKED["message"] in brief
    assert picked in brief
    assert not [
        part
        for content in request.content_list
        if content.role == "model"
        for part in content.parts or ()
        if part.text == ""
    ], "the ADK planner's trailing empty text part is replayed"


# --- Amendments and replans ---------------------------------------------------------------

_AMENDMENT = "Show one total for the whole period."


def _amended(plan: Plan) -> dict[str, Any]:
    return {**_planned(plan), "amendment": _AMENDMENT}


def _text(request: Request) -> str:
    return "".join(
        part.text or "" for content in request.content_list for part in content.parts or ()
    )


@pytest.mark.parametrize("scenario", ["amendment", "ask-during-amendment"])
async def test_an_amendment_opens_a_turn_of_its_own_and_lands_a_plan(
    plan: Plan, scenario: str, request: pytest.FixtureRequest
) -> None:
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

    entry = await _entry(scripts[scenario])

    assert entry.outputs[-1] is not None and "plan" in entry.outputs[-1], entry.outputs
    assert all(_AMENDMENT in _text(sent) for sent in entry.requests)
    _assert_expected(request, entry)


async def _replanned(plan: Plan, opened_by: Mapping[str, Any]) -> _Entry:
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
    seam = scripted_planner(scripted, tools=_proposals())
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
    # The refusal's time is recorded with it (`_record_refusal`).
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
async def test_a_replan_in_the_turn_it_planned_in_lands_a_plan(
    plan: Plan, opened_by: dict[str, Any], request: pytest.FixtureRequest
) -> None:
    entry = await _replanned(plan, opened_by)

    assert [output is not None and "plan" in output for output in entry.outputs] == [True, True]
    assert len(entry.requests) == 2
    _assert_expected(request, entry)


# --- Two bounds closing in one request ----------------------------------------------------


async def test_a_second_bound_closing_in_the_same_request_refuses_with_its_own_code(
    plan: Plan, request: pytest.FixtureRequest
) -> None:
    """The call cap closes the first entry's second call. The request's wall
    clock then runs out before a replan the plan card's pick opens in the same
    request, whose first call the wall bound closes. `TurnBudget.closed` keeps
    the call cap; the replan is refused on the wall bound that closed its own call."""

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

    entry = await _entry(script())

    assert [output["refused"]["code"] for output in entry.outputs if output is not None] == [
        SeamCode.TURN_MODEL_CALLS_SPENT,
        SeamCode.TURN_WALL_TIME_SPENT,
    ]
    _assert_expected(request, entry)


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
    plan: Plan, request: pytest.FixtureRequest
) -> None:
    """A referral the user decides puts an ask in the model's place under an
    `adk-` call id. The ask tool refuses it, the run goes on, and the next
    request replays the ask and its answer with ADK's own call id removed and
    the placeholder signature kept."""
    entry = await _entry(
        _Script(
            replies=[
                reply(call_part(PROPOSE_PLAN, {}, id="p1", signature=_SIGNED)),
                reply(call_part(PROPOSE_PLAN, {}, id="p2", signature=_SIGNED)),
            ],
            runs=[_planned(plan)],
            setup=_refusing_asks,
        ),
    )

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
        for event in _planner_events(entry)
        for call in event.get_function_calls()
        if call.name == PLANNER_ASK
    ]
    assert stored.id == "adk-1", "the stored ask keeps its id"
    _assert_expected(request, entry, config="request_config_refusing_ask")


async def test_an_unsigned_first_call_is_replayed_with_the_placeholder_signature(
    plan: Plan, request: pytest.FixtureRequest
) -> None:
    """Gemini signs the first function call of a response it thought before. One
    it sent unsigned is replayed with Gemini's placeholder signature."""
    entry = await _entry(
        _Script(
            replies=[
                reply(call_part(PROPOSE_PLAN, {}, id="p1")),
                reply(call_part(PROPOSE_PLAN, {}, id="p2", signature=_SIGNED)),
            ],
            runs=[_planned(plan)],
            setup=_proposals(_REFUSED),
        ),
    )

    (replayed,) = [
        part
        for content in entry.requests[1].content_list
        for part in content.parts or ()
        if part.function_call is not None and part.function_call.id == "p1"
    ]
    assert replayed.thought_signature == SKIP_THOUGHT_SIGNATURE
    _assert_expected(request, entry)


async def test_calls_without_ids_are_stored_under_adk_ids_and_replayed_without_them(
    plan: Plan, request: pytest.FixtureRequest
) -> None:
    """Gemini sends its function calls without ids. The planner gives each one
    an `adk-` id, stores the call and its answer under it, and sends the next
    request without it."""
    entry = await _entry(
        _Script(
            replies=[
                reply(call_part(PROPOSE_PLAN, {}, signature=_SIGNED)),
                reply(call_part(PROPOSE_PLAN, {}, signature=_SIGNED)),
            ],
            runs=[_planned(plan)],
            setup=_proposals(_REFUSED),
        ),
    )

    ids = [call.id for event in _planner_events(entry) for call in event.get_function_calls()]
    assert ids == ["adk-1", "adk-2"]
    assert all(
        part.function_call.id is None
        for content in entry.requests[1].content_list
        for part in content.parts or ()
        if part.function_call is not None
    )
    _assert_expected(request, entry)


# --- A Stop while a switch runs -----------------------------------------------------------


async def _stopped_switch(
    plan: Plan, monkeypatch: pytest.MonkeyPatch
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """The frames of a turn whose plan the planner drafted, and of a
    switch on its card that the user stops while the revision runs."""
    entered = asyncio.Event()

    async def revising(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        entered.set()
        await asyncio.sleep(5)
        raise AssertionError("a stopped revision is cancelled")

    monkeypatch.setattr(switch_module, "_switched_plan", revising)
    monkeypatch.setattr(studio_module, "SWITCH_REOFFER_SECONDS", 0.2)
    scripted = ScriptedGenai([reply(call_part(PROPOSE_PLAN, {}, id="p1", signature=_SIGNED))])
    seam = scripted_planner(scripted, tools=_proposals(records=plan))
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


async def test_a_stop_while_a_switch_runs_offers_the_plan_unchanged(
    plan: Plan, monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
) -> None:
    """The plan card an entry drafted takes a switch and a Stop. The stream ends
    on the same plan re-offered under a fresh interrupt, carrying
    `switch_cancelled`."""
    streamed = await _stopped_switch(plan, monkeypatch)

    expect(expectation_name(request), [_frames(frames) for frames in streamed])


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
    schemas: list[ToolSchema],
) -> tuple[list[Event], list[Event], Any, str]:
    """A turn whose planner asks which source the user means, is
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
        settings=Settings(),
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
    """`events` as `_read` reads them, less each progress line's
    duration and with each time recorded in them put as `<time>`."""
    dumped = _read(events)
    for event in dumped:
        event.get("custom_metadata", {}).get("progress", {}).pop("duration_ms", None)
    return _timeless({"events": dumped})["events"]


def _timeless(state: Mapping[str, Any]) -> Any:
    """`state` as JSON, each time recorded in it put as `<time>`."""
    return json.loads(_RECORDED_TIME.sub("<time>", json.dumps(dict(state), default=str)))


async def test_a_switch_after_an_entry_revises_the_plan(
    real_schemas: list[ToolSchema], request: pytest.FixtureRequest
) -> None:
    """The switch node's revision reads the entry's stored ask and answer
    (`PlannerSeam.revise`), so the sector decision it makes is the user's and
    the revised plan, the turn's events and the session's state are the stored
    ones."""
    switched, events, state, copy_pin = await _switched_after_an_entry(real_schemas)

    offered = _output_of(switched, "switch")
    assert "switch_failed" not in offered, offered.get("switch_failed")
    revised = Plan.model_validate_json(state["plan"])
    (dataset,) = [r for r in revised.evidence_requirements if isinstance(r, DatasetRequirement)]
    assert dataset.source_spec_id == copy_pin
    sector = next(d for d in revised.decisions if d.slot.startswith("scope_sector_"))
    assert sector.resolved_by is ResolvedBy.USER
    expect(
        expectation_name(request),
        {"events": _untimed(events), "steps": _steps(events), "state": state},
    )
