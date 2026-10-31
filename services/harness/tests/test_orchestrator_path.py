"""The orchestrator's path through the workflow, run by a real ADK runner
with a scripted model: answer, ask, plan, and what the agent reads after each."""

from __future__ import annotations

import copy
import dataclasses
import json
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from analysis_mcp import digest as analysis_digest
from analysis_mcp.digest import render_rank_digest
from analysis_mcp.engine.rank import run_rank
from analysis_mcp.execution_identity import current_execution_identity
from common.materialisation import FrameRef
from compiler.surface import FixtureSurface, ToolCallResult
from data_mcp import get_data_digest
from data_mcp.tool_errors import ToolErrorCode
from dfo_frame_grammar import Frame
from executor.completion import PublishFault
from executor.ledger import InMemoryRunLedger
from google.adk.events import EventActions
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService, State
from google.adk.workflow import START, Workflow
from google.genai import types
from harness import plan_display
from harness.conversation import current_conversation_id
from harness.graph import execute as execute_module
from harness.graph import plan_done as plan_done_module
from harness.graph.orchestrate import orchestrate_node
from harness.graph.plan_done import (
    _DROPPED,
    HandBack,
    StepSummary,
    _ask_status,
    _partial_result,
    _stop_causes,
)
from harness.graph.state import (
    APPROVE_INTERRUPT,
    CLARIFY_INTERRUPT,
    ORCHESTRATOR_ASKED_KEY,
    ORCHESTRATOR_BRIEF_KEY,
    ORCHESTRATOR_CALLS_KEY,
    ORCHESTRATOR_FRAMES_KEY,
    ORCHESTRATOR_PENDING_KEY,
    ORCHESTRATOR_PLAN_RECORDS_KEY,
    ORCHESTRATOR_PRIOR_LEDGER_KEY,
    ORCHESTRATOR_READINGS_KEY,
    ORCHESTRATOR_SEARCHES_KEY,
    ORCHESTRATOR_SPECS_KEY,
    ORCHESTRATOR_STATUS_FAILURE_KEY,
    ORCHESTRATOR_UNAVAILABLE_KEY,
    UNREADABLE_PLAN_NOTE,
    AskOption,
    AskQuestion,
    GraphState,
    PendingAsk,
    ask_questions,
    pending_action,
)
from harness.graph.workflow import planner_entry
from harness.orchestrator_tools import HANDED_OFF, build_orchestrator_tools
from harness.planner_seam import PlannerSeam, PlannerSeamError
from harness.provenance import ProvenanceStore
from harness.scripted import ScriptedPlannerModel, ScriptedSurface, scripted_publish_result
from harness.sentences import DECLINE_REASONS
from harness.settings import Settings
from harness.studio_translate import _RecordingSurface
from harness.tool_contracts import HarnessCode
from harness.turn_records import PlanAsk, PlanRecord, SearchReading, WindowReading
from harness.wiring import handing_back_planner_failures
from planner.agent import ToolContextSession
from planner.plan import (
    ASK_BREAKDOWNS_STATE_KEY,
    ASKS_ACROSS_YEARS_STATE_KEY,
    ASKS_STATE_KEY,
    AskAnswer,
    AskBreakdown,
    Decline,
    Plan,
    UnsupportedOperation,
    load_asks,
)

from .context_probe import RecordingLlm
from .orchestrator_fakes import (
    CannedSearch,
    FailingPlanner,
    RefusingPlanner,
    ScriptLlm,
    approve_answer,
    call,
    carries_an_interrupt,
    chart,
    create_request_input_response,
    definite_envelope,
    fetch_call,
    frame_for,
    handed_back,
    make_charts_call,
    no_executor,
    no_resolver,
    open_session,
    orchestrator_app,
    orchestrator_runner,
    quotes_another_agent,
    request_text,
    run_turn,
    search_call,
    single_pause,
    stream,
    studio_client,
    turn_body,
    visited_nodes,
)


class AskingPlanner(ScriptedPlannerModel):
    """`ScriptedPlannerModel` whose plan answers the brief's first ask with
    its final step and declines every other ask as `unsupported_ask`."""

    async def __call__(self, **kwargs: Any) -> dict[str, Any]:
        plan = Plan.model_validate((await super().__call__(**kwargs))["plan"])
        asks = load_asks(kwargs["ctx"].state)
        final = plan.evidence_requirements[-1].id
        plan = plan.model_copy(
            update={
                "asks": asks,
                "answered": (AskAnswer(ask=0, outputs=(final,)),),
                "declined": tuple(
                    Decline(ask=ask, code=UnsupportedOperation.UNSUPPORTED_ASK, ask_ref=number)
                    for number, ask in enumerate(asks)
                    if number
                ),
            }
        )
        return {"plan": plan.model_dump(mode="json"), "summary": {"version": plan.version}}


async def _describes(*, tool: str, args: dict[str, Any], on_progress: Any = None) -> Any:
    assert tool == "describe_frame", tool
    return {"outcome": "ok", "view": {"dimensions": [{"name": "observationAbout"}]}}


# The spec_id `harness.scripted_plan.scripted_plan` retrieves when no search pinned one.
_SCRIPTED_SPEC = "spec_" + "f" * 32


async def test_an_answered_turn_ends_after_the_agent_without_a_pause():
    runner, model = orchestrator_runner(
        types.Part.from_text(text="Kenya's GDP was $101 billion in 2020.")
    )
    await open_session(runner)

    events = await run_turn(runner, types.UserContent("Kenya's GDP in 2020?"))

    assert request_text(model.requests[0]).count("Kenya's GDP in 2020?") == 1
    assert not [e for e in events if e.error_code]
    assert not [c for e in events for c in e.get_function_calls() if c.name == "adk_request_input"]
    assert "route" in visited_nodes(events) and "ask" not in visited_nodes(events)


_CONGO = {
    "question": "Which Congo?",
    "options": [{"label": "Republic of the Congo"}, {"label": "DR Congo"}],
}
_SETTLED = "Each answered choice is settled. Do not ask it again."


def _answers(**answers: dict[str, Any]) -> dict[str, Any]:
    """The clarify interrupt's resume value: one answer per question id."""
    return {"answers": answers}


@pytest.mark.parametrize(
    "answer",
    [_answers(q1={"kind": "option", "spec_id": "o1"}), {"spec_id": "o1"}],
    ids=["answers", "legacy_pick"],
)
async def test_ask_pauses_with_generated_option_ids_and_hands_the_choice_back_in_words(
    answer: dict[str, Any],
):
    runner, model = orchestrator_runner(
        call("ask_user", questions=[_CONGO]),
        types.Part.from_text(text="The Republic of the Congo's GDP was $15.7 billion in 2024."),
    )
    await open_session(runner)

    pause = single_pause(await run_turn(runner, types.UserContent("GDP of Congo?")))
    assert pause.id == CLARIFY_INTERRUPT
    assert pause.args["payload"] == {
        "questions": [
            {
                "id": "q1",
                "question": "Which Congo?",
                "candidates": [
                    {"spec_id": "o1", "label": "Republic of the Congo"},
                    {"spec_id": "o2", "label": "DR Congo"},
                ],
            }
        ]
    }
    reply = create_request_input_response(pause.id, {"result": answer})
    events = await run_turn(runner, types.Content(role="user", parts=[reply]))

    resumed = request_text(model.requests[-1])
    assert "GDP of Congo?" in resumed
    assert (
        f"You asked the user: Which Congo?\nThe user chose: Republic of the Congo\n{_SETTLED}"
        in resumed
    )
    assert not carries_an_interrupt(model.requests[-1])
    assert not quotes_another_agent(model.requests[-1])
    assert not [e for e in events if e.error_code]


_GROSS, _NET = "spec_" + "a" * 32, "spec_" + "c" * 32


@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        (
            _answers(q1={"kind": "option", "spec_id": _NET}, q2={"kind": "skipped"}),
            [
                f"1. How should aid be counted? The user chose: net disbursements ({_NET})",
                "2. In which prices? The user left this to you. Go on with Current US dollars.",
                _SETTLED,
            ],
        ),
        (
            _answers(
                q1={"kind": "option", "spec_id": "o3"},
                q2={"kind": "text", "text": "2015 dollars"},
            ),
            [
                "1. How should aid be counted? The user chose: Either",
                "2. In which prices? The user answered in their own words: 2015 dollars",
                _SETTLED,
            ],
        ),
        (
            _answers(q2={"kind": "option", "spec_id": "o2"}),
            [
                "1. How should aid be counted? The user left this to you. Go on with "
                "gross disbursements.",
                "2. In which prices? The user chose: Constant 2023 US dollars",
                _SETTLED,
            ],
        ),
        (
            {"skip": True},
            [
                "1. How should aid be counted?",
                "2. In which prices?",
                "The user skipped the questions. Do not answer them",
            ],
        ),
    ],
    ids=["count_and_skipped", "other_and_free_text", "unanswered", "dismissed"],
)
async def test_several_questions_pause_together_and_each_answer_is_handed_back(
    answer: dict[str, Any], expected: list[str]
):
    block = {
        "spec_id": _GROSS,
        "resolved": {
            "measure_label": "gross disbursements",
            "alternative_measures": [
                {"label": "net disbursements", "axis": "what", "definition": {"spec_id": _NET}}
            ],
        },
    }
    runner, model = orchestrator_runner(
        call(
            "ask_user",
            questions=[
                {
                    "question": "How should aid be counted?",
                    "options": [
                        {"label": "gross disbursements", "spec_id": _GROSS},
                        {"label": "net disbursements", "spec_id": _NET},
                        {"label": "Either"},
                    ],
                },
                {
                    "question": "In which prices?",
                    "options": [
                        {"label": "Current US dollars"},
                        {"label": "Constant 2023 US dollars"},
                    ],
                },
            ],
        ),
        types.Part.from_text(text="Done."),
    )
    await open_session(runner, **{ORCHESTRATOR_SPECS_KEY: {_GROSS: block}})

    pause = single_pause(await run_turn(runner, types.UserContent("Aid from France to Kenya?")))
    assert pause.args["message"] == "How should aid be counted?"
    assert pause.args["payload"]["questions"] == [
        {
            "id": "q1",
            "question": "How should aid be counted?",
            "candidates": [
                {"spec_id": _GROSS, "label": "gross disbursements"},
                {"spec_id": _NET, "label": "net disbursements"},
                {"spec_id": "o3", "label": "Either"},
            ],
        },
        {
            "id": "q2",
            "question": "In which prices?",
            "candidates": [
                {"spec_id": "o1", "label": "Current US dollars"},
                {"spec_id": "o2", "label": "Constant 2023 US dollars"},
            ],
        },
    ]
    reply = create_request_input_response(pause.id, {"result": answer})
    events = await run_turn(runner, types.Content(role="user", parts=[reply]))

    assert "\n".join(["You asked the user:", *expected]) in request_text(model.requests[-1])
    assert not [e for e in events if e.error_code]


# Search's answer to "Bilateral plus imputed multilateral aid from France to
# Kenya": gross disbursements in current prices, offering net disbursements.
_FRANCE_KENYA = json.loads(
    (
        Path(__file__).parents[2]
        / "data-mcp"
        / "tests"
        / "fixtures"
        / "resolve_aid_france_kenya.json"
    ).read_text()
)["parts"][0]["spec"]
_FK_GROSS = _FRANCE_KENYA["spec_id"]
_FK_NET = "spec_c67524ce5da3bfd4d94994a76c560b28"
_FK_CONSTANT = "spec_" + "e" * 32


def _priced(alternative: dict[str, Any], qualifier: str) -> dict[str, Any]:
    """`alternative` with its measure in `qualifier` prices."""
    priced = copy.deepcopy(alternative)
    priced["definition"]["identity"]["shape"]["measurement_qualifier"] = qualifier
    for value in priced["measure_axis_values"]:
        if value["axis"] == "measurementQualifier":
            value["value"] = {"ref": qualifier, "label": qualifier}
    return priced


def _france_kenya_with_constant_prices(*, net_qualifier: str) -> dict[str, Any]:
    """The France to Kenya block offering, besides net disbursements in
    `net_qualifier` prices, the gross count in constant prices as dfo-apis
    lists it: axis "unit", labelled by its prices."""
    block = copy.deepcopy(_FRANCE_KENYA)
    resolved = block["resolved"]
    (net,) = resolved["alternative_measures"]
    gross = {
        "definition": copy.deepcopy(block["definition"]),
        "measure_axis_values": resolved["measure_axis_values"],
    }
    constant = _priced(gross, "RealValue")
    constant["definition"]["spec_id"] = _FK_CONSTANT
    resolved["alternative_measures"] = [
        _priced(net, net_qualifier),
        {**constant, "label": "constant prices", "axis": "unit"},
    ]
    return block


def _open_reading(spec_id: str) -> dict[str, Any]:
    """A `definite` reading of `spec_id` over 2018 to 2022, as `store_search` keeps it."""
    return SearchReading(
        text=None,
        status="definite",
        spec_id=spec_id,
        window=WindowReading(start=2018, end=2022, source="requested"),
    ).model_dump(mode="json")


_USD_BLOCK_UNIT = {
    "basis": "fallback",
    "ref": "CurrencyUSD",
    "label": "US Dollar",
    "currency": "USD",
}


def _usd_block(spec_id: str) -> dict[str, Any]:
    """A spec block in current US dollars, so its prices are open over several years."""
    return {"spec_id": spec_id, "resolved": {"facets": {"unit": _USD_BLOCK_UNIT}}}


_OTHER = "spec_" + "b" * 32


async def test_a_prices_question_takes_its_options_from_its_part_s_open_prices():
    """Two parts whose prices are open, worded alike. Each prices question
    names its part, and code gives it that part's `open_prices` entries,
    marked with the part and their prices. A pick of constant prices no
    search listed asks for a search of that part."""
    runner, model = orchestrator_runner(
        call(
            "ask_user",
            questions=[
                {"question": "Aid in which prices?", "prices_of": _GROSS},
                {"question": "GDP in which prices?", "prices_of": _OTHER},
            ],
        ),
        types.Part.from_text(text="Done."),
    )
    await open_session(
        runner,
        **{
            ORCHESTRATOR_SPECS_KEY: {spec: _usd_block(spec) for spec in (_GROSS, _OTHER)},
            ORCHESTRATOR_SEARCHES_KEY: [_open_reading(spec) for spec in (_GROSS, _OTHER)],
        },
    )

    pause = single_pause(await run_turn(runner, types.UserContent("Aid and GDP, 2018-2022?")))
    assert [q["candidates"] for q in pause.args["payload"]["questions"]] == [
        [
            {"spec_id": "o1", "label": "current prices", "part": part, "prices": "current"},
            {"spec_id": "o2", "label": "constant prices", "part": part, "prices": "constant"},
        ]
        for part in (_GROSS, _OTHER)
    ]
    reply = create_request_input_response(
        pause.id,
        {
            "result": _answers(
                q1={"kind": "option", "spec_id": "o1"}, q2={"kind": "option", "spec_id": "o2"}
            )
        },
    )
    await run_turn(runner, types.Content(role="user", parts=[reply]))

    asked = request_text(model.requests[-1])
    assert (
        f"The user chose: constant prices, which no search listed. Search once for that part "
        f"({_OTHER}) in constant prices." in asked
    )
    assert f"({_GROSS}) in constant prices" not in asked


async def test_two_prices_questions_on_one_part_are_refused():
    """Two questions pricing one part would let the later answer override the
    earlier one in silence, so `ask_user` refuses the call and nothing pauses."""
    runner, _ = orchestrator_runner(
        call(
            "ask_user",
            questions=[
                {"question": "Aid in which prices?", "prices_of": _GROSS},
                {"question": "And in which prices again?", "prices_of": _GROSS},
            ],
        ),
        types.Part.from_text(text="Done."),
    )
    await open_session(
        runner,
        **{
            ORCHESTRATOR_SPECS_KEY: {_GROSS: _usd_block(_GROSS)},
            ORCHESTRATOR_SEARCHES_KEY: [_open_reading(_GROSS)],
        },
    )

    events = await run_turn(runner, types.UserContent("Aid, 2018-2022?"))

    assert not [c for e in events for c in e.get_function_calls() if c.name == "adk_request_input"]
    (refused,) = [
        r.response for e in events for r in e.get_function_responses() if r.name == "ask_user"
    ]
    assert refused is not None
    assert refused["code"] == ToolErrorCode.REQUEST_INVALID.value
    assert f"{_GROSS} is priced by more than one question" in refused["refusal"]


@pytest.mark.parametrize("same_measure", [True, False], ids=["two-windows", "two-counts"])
async def test_two_windows_of_one_measure_take_one_prices_question(same_measure: bool):
    """Two readings of one measure over different years are one part, so two
    prices questions on them are refused, and nothing pauses. Two readings of
    different counts each take their own question."""
    other = copy.deepcopy(_FRANCE_KENYA)
    other["spec_id"] = other["definition"]["spec_id"] = _OTHER
    identity = other["definition"]["identity"]
    if same_measure:
        identity["requested_window"] = {"kind": "range", "start": 2010, "end": 2014}
    else:
        for slot in identity["slots"]:
            if slot["property"] == "nettingBasis":
                slot["binding"]["ref"] = "Net"
    runner, _ = orchestrator_runner(
        call(
            "ask_user",
            questions=[
                {"question": "Aid to 2014 in which prices?", "prices_of": _OTHER},
                {"question": "Aid since 2018 in which prices?", "prices_of": _FK_GROSS},
            ],
        ),
        types.Part.from_text(text="Done."),
    )
    await open_session(
        runner,
        **{
            ORCHESTRATOR_SPECS_KEY: {_FK_GROSS: _FRANCE_KENYA, _OTHER: other},
            ORCHESTRATOR_SEARCHES_KEY: [_open_reading(_FK_GROSS), _open_reading(_OTHER)],
        },
    )

    events = await run_turn(runner, types.UserContent("Aid to 2014 against since 2018?"))

    paused = [c for e in events for c in e.get_function_calls() if c.name == "adk_request_input"]
    assert (not paused) is same_measure
    if same_measure:
        (refused,) = [
            r.response for e in events for r in e.get_function_responses() if r.name == "ask_user"
        ]
        assert refused is not None
        assert refused["code"] == ToolErrorCode.REQUEST_INVALID.value
        assert "measure one thing over other years" in refused["refusal"]


_COUNT_AND_PRICES = [
    {
        "question": "How should aid be counted?",
        "options": [
            {"label": "Gross disbursements", "spec_id": _FK_GROSS},
            {"label": "Net disbursements", "spec_id": _FK_NET},
        ],
    },
    {
        "question": "In which prices?",
        "options": [
            {"label": "Current US dollars"},
            {"label": "Constant US dollars", "spec_id": _FK_CONSTANT},
        ],
    },
]
_NET_LINE = f"1. How should aid be counted? The user chose: Net disbursements ({_FK_NET})"
_CONSTANT_LINE = f"2. In which prices? The user chose: Constant US dollars ({_FK_CONSTANT})"


@pytest.mark.parametrize(
    ("net_qualifier", "answer", "expected", "elected"),
    [
        (
            "RealValue",
            _answers(
                q1={"kind": "option", "spec_id": _FK_NET},
                q2={"kind": "option", "spec_id": _FK_CONSTANT},
            ),
            [
                _NET_LINE,
                _CONSTANT_LINE,
                "Together the answers ask for net disbursements in constant prices. Fetch "
                f"{_FK_NET}, which this search listed with both.",
            ],
            {_FK_NET},
        ),
        (
            "Nominal",
            _answers(
                q1={"kind": "option", "spec_id": _FK_NET},
                q2={"kind": "option", "spec_id": _FK_CONSTANT},
            ),
            [
                _NET_LINE,
                _CONSTANT_LINE,
                "Together the answers ask for net disbursements in constant prices, which this "
                "search did not list. Search once for that reading.",
            ],
            {_FK_NET},
        ),
        (
            "Nominal",
            _answers(
                q1={"kind": "option", "spec_id": _FK_NET}, q2={"kind": "option", "spec_id": "o1"}
            ),
            [_NET_LINE, "2. In which prices? The user chose: Current US dollars"],
            {_FK_NET},
        ),
        (
            "Nominal",
            _answers(q1={"kind": "skipped"}, q2={"kind": "option", "spec_id": _FK_CONSTANT}),
            [
                "1. How should aid be counted? The user left this to you. Go on with Gross "
                "disbursements.",
                _CONSTANT_LINE,
            ],
            set(),
        ),
        (
            "Nominal",
            _answers(
                q1={"kind": "option", "spec_id": _FK_GROSS},
                q2={"kind": "option", "spec_id": _FK_CONSTANT},
            ),
            [
                f"1. How should aid be counted? The user chose: Gross disbursements ({_FK_GROSS})",
                _CONSTANT_LINE,
                "Together the answers ask for gross disbursements in constant prices. Fetch "
                f"{_FK_CONSTANT}, which this search listed with both.",
            ],
            {_FK_GROSS, _FK_CONSTANT},
        ),
        (
            "RealValue",
            _answers(
                q1={"kind": "option", "spec_id": _FK_NET}, q2={"kind": "option", "spec_id": "o1"}
            ),
            [
                _NET_LINE,
                "2. In which prices? The user chose: Current US dollars",
                "This search listed net disbursements only in constant prices. For other "
                "prices, search once for that reading.",
            ],
            {_FK_NET},
        ),
    ],
    ids=[
        "both_resolved",
        "both_unlisted",
        "only_count",
        "only_prices",
        "elected_picked",
        "count_only_in_other_prices",
    ],
)
async def test_a_count_and_prices_picked_together_name_the_one_reading_with_both(
    net_qualifier: str, answer: dict[str, Any], expected: list[str], elected: set[str]
):
    """A count picked in one answer, search's own reading included, and
    other prices in another, both of one search result, name the listed
    reading measuring both, or ask for one search when none is listed. A
    count listed only at the other price basis is said to be. Any other pair
    reads answer by answer."""
    block = _france_kenya_with_constant_prices(net_qualifier=net_qualifier)
    runner, model = orchestrator_runner(
        call("ask_user", questions=_COUNT_AND_PRICES), types.Part.from_text(text="Done.")
    )
    await open_session(runner, **{ORCHESTRATOR_SPECS_KEY: {_FK_GROSS: block}})
    pause = single_pause(await run_turn(runner, types.UserContent("Aid from France to Kenya?")))

    reply = create_request_input_response(pause.id, {"result": answer})
    events = await run_turn(runner, types.Content(role="user", parts=[reply]))

    assert not [e for e in events if e.error_code]
    assert "\n".join(["You asked the user:", *expected, _SETTLED]) in request_text(
        model.requests[-1]
    )
    session = await runner.session_service.get_session(app_name="t", user_id="u", session_id="s")
    assert session is not None
    tool_context = SimpleNamespace(state=session.state, session=session)
    chosen = ToolContextSession(cast(Any, tool_context)).elected_spec_ids()
    assert {spec_id for spec_id in chosen if spec_id.startswith("spec_")} == elected


@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        ({"text": "the big one"}, "The user answered in their own words: the big one"),
        (
            _answers(q1={"kind": "text", "text": "the big one"}),
            "The user answered in their own words: the big one",
        ),
        ({"skip": True}, "The user skipped the question. Do not answer it"),
    ],
    ids=["legacy_text", "answers_text", "dismissed"],
)
async def test_a_free_text_answer_and_a_skip_are_handed_back_as_such(
    answer: dict[str, Any], expected: str
):
    runner, model = orchestrator_runner(
        call("ask_user", questions=[_CONGO]),
        types.Part.from_text(text="Done."),
    )
    await open_session(runner)
    pause = single_pause(await run_turn(runner, types.UserContent("GDP of Congo?")))

    reply = create_request_input_response(pause.id, {"result": answer})
    await run_turn(runner, types.Content(role="user", parts=[reply]))

    assert expected in request_text(model.requests[-1])


@pytest.mark.parametrize(
    "options",
    [
        ["Republic of the Congo", "DR Congo"],
        [{"label": "Republic of the Congo", "spec_id": None}, {"label": "DR Congo"}],
    ],
    ids=["bare_labels", "options"],
)
@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        ({"spec_id": "o2"}, "The user chose: DR Congo"),
        (_answers(q1={"kind": "option", "spec_id": "o2"}), "The user chose: DR Congo"),
        ({"text": "the big one"}, "The user answered in their own words: the big one"),
        ({"skip": True}, "The user skipped the question. Do not answer it"),
    ],
    ids=["legacy_pick", "answers_pick", "text", "dismissed"],
)
async def test_a_question_paused_in_the_one_question_shape_resumes(
    options: list[Any], answer: dict[str, Any], expected: str
):
    """A question held in the one-question shape `{kind, question, options}`
    resumes, whether its options are bare labels or option objects."""
    runner, model = orchestrator_runner(
        call("ask_user", questions=[_CONGO]),
        types.Part.from_text(text="Done."),
    )
    await open_session(runner)
    pause = single_pause(await run_turn(runner, types.UserContent("GDP of Congo?")))
    _stored_state(runner)[ORCHESTRATOR_ASKED_KEY] = {
        "kind": "ask",
        "question": "Which Congo?",
        "options": options,
    }

    reply = create_request_input_response(pause.id, {"result": answer})
    events = await run_turn(runner, types.Content(role="user", parts=[reply]))

    assert not [e for e in events if e.error_code]
    assert f"You asked the user: Which Congo?\n{expected}" in request_text(model.requests[-1])


def test_a_pending_question_stored_in_the_one_question_shape_reads_as_one_question():
    state = {
        ORCHESTRATOR_PENDING_KEY: {
            "kind": "ask",
            "question": "Which Congo?",
            "options": ["Republic of the Congo", "DR Congo"],
        }
    }

    pending = pending_action(state)

    assert isinstance(pending, PendingAsk)
    assert ask_questions(pending.questions) == [
        {
            "id": "q1",
            "question": "Which Congo?",
            "candidates": [
                {"spec_id": "o1", "label": "Republic of the Congo"},
                {"spec_id": "o2", "label": "DR Congo"},
            ],
        }
    ]


async def test_a_planner_that_refuses_hands_back_and_the_agent_keeps_the_question():
    planner = RefusingPlanner()
    runner, model = orchestrator_runner(
        call(
            "plan",
            brief="Niger's Global Fund receipts as a share of health spending",
            asks=["Niger's Global Fund receipts as a share of health spending"],
        ),
        types.Part.from_text(text="The share could not be calculated."),
        planner=planner,
    )
    searched = {"text": "Global Fund receipts", "status": "no_data", "refusal_code": "no_match"}
    earlier = {"text": "Niger health spending", "status": "definite", "spec_id": "spec_h"}
    await open_session(
        runner,
        **{
            ORCHESTRATOR_SEARCHES_KEY: [searched],
            ORCHESTRATOR_PRIOR_LEDGER_KEY: {"readings": [earlier], "spec_ids": [], "warnings": {}},
        },
    )

    events = await run_turn(
        runner, types.UserContent("How much does Niger get from the Global Fund?")
    )

    assert planner.questions == ["Niger's Global Fund receipts as a share of health spending"]
    # An earlier turn's readings come first, then this turn's.
    assert planner.readings == [[earlier, searched]]
    after_plan = request_text(model.requests[-1])
    assert "How much does Niger get from the Global Fund?" in after_plan
    assert "The planner could not make a plan: the planner ran out of rounds" in after_plan
    assert not [e for e in events if e.error_code]
    session = await runner.session_service.get_session(app_name="t", user_id="u", session_id="s")
    assert session.state.get("question") in (None, "")
    assert session.state[ORCHESTRATOR_BRIEF_KEY].startswith("Niger's")


@pytest.mark.parametrize(
    ("tool", "arguments", "questions", "asks"),
    [
        (
            "plan",
            {"brief": "Kenya's aid since 2015", "asks": ["the total"]},
            ["Kenya's aid since 2015"],
            [("the total",)],
        ),
        (
            "ask_user",
            {
                "questions": [
                    AskQuestion(question="Which years?", options=[AskOption(label="2015")])
                ]
            },
            [],
            [],
        ),
    ],
)
async def test_the_planner_entry_runs_alone_from_a_plan_calls_hand_off(
    tool: str, arguments: dict[str, Any], questions: list[str], asks: list[tuple[str, ...]]
):
    """`planner_entry`'s nodes, run without the rest of the turn, give the planner the
    brief and asks a `plan` call handed off and end on its answer. An `ask_user` hand-off
    stops at `route`."""
    tools = {
        tool.__name__: tool
        for tool in build_orchestrator_tools(resolver=no_resolver, executor=no_executor)
    }
    handed = SimpleNamespace(state={}, actions=EventActions())
    tools[tool](**arguments, tool_context=handed)
    planner = RefusingPlanner(final=True)
    entry = planner_entry(planner)
    runner = Runner(
        node=Workflow(
            name="planner_entry",
            state_schema=GraphState,
            edges=[(START, entry.route), *entry.edges],
        ),
        app_name="t",
        session_service=InMemorySessionService(),
    )
    await open_session(runner, **handed.state)

    events = await run_turn(runner, types.UserContent("How much aid has Kenya had?"))

    assert (planner.questions, planner.asks) == (questions, asks)
    assert not [e for e in events if e.error_code]


async def test_a_failed_plan_gives_the_plan_call_back_once():
    planner = RefusingPlanner()
    runner, _ = orchestrator_runner(
        call("plan", brief="Kenya's aid as a share of GDP", asks=["Kenya's aid as a share of GDP"]),
        call(
            "plan",
            brief="Kenya's CRS aid as a share of GDP",
            asks=["Kenya's CRS aid as a share of GDP"],
        ),
        call(
            "plan",
            brief="Kenya's aid as a share of GDP, any source",
            asks=["Kenya's aid as a share of GDP, any source"],
        ),
        types.Part.from_text(text="Could not be calculated."),
        planner=planner,
    )
    await open_session(runner)

    events = await run_turn(runner, types.UserContent("Kenya's aid as a share of GDP?"))

    assert len(planner.questions) == 2
    third = [
        r.response
        for e in events
        for r in e.get_function_responses()
        if r.name == "plan" and r.response and "error" in r.response
    ]
    assert len(third) == 1 and "limit" in third[0]["error"]


@pytest.mark.parametrize("final", [False, True])
async def test_only_a_refusal_another_brief_could_avoid_offers_another_plan(final: bool):
    """A final refusal (an internal or repeated failure, or a question declined
    whole) fails the same way for any brief, so the plan call is not given back
    and the hand-back does not offer one."""
    runner, model = orchestrator_runner(
        call("plan", brief="Kenya's aid as a share of GDP", asks=["Kenya's aid as a share of GDP"]),
        types.Part.from_text(text="Could not be calculated."),
        planner=RefusingPlanner(final=final),
    )
    await open_session(runner)

    await run_turn(runner, types.UserContent("Kenya's aid as a share of GDP?"))

    session = await runner.session_service.get_session(app_name="t", user_id="u", session_id="s")
    calls = session.state[ORCHESTRATOR_CALLS_KEY]
    assert bool(calls.get("plan_given_back")) is not final
    assert ("planned once more" in request_text(model.requests[-1])) is not final


async def test_each_plan_call_holds_the_planner_to_its_own_asks_alone():
    """A plan call lists the asks for its own plan only: a later call with
    one ask and no breakdowns leaves no earlier call's asks, breakdowns, or
    across-years marks for the planner to account for."""
    planner = RefusingPlanner()
    asks = ["a chart of France's aid to Guatemala", "the total since 2010 by donor"]
    runner, _ = orchestrator_runner(
        call(
            "plan",
            brief="France's aid to Guatemala since 2010",
            asks=[asks[0], {"text": asks[1], "by": ["donor"], "across_years": True}],
        ),
        call(
            "plan",
            brief="France's bilateral aid to Guatemala",
            asks=["France's bilateral aid to Guatemala"],
        ),
        types.Part.from_text(text="Could not be calculated."),
        planner=planner,
    )
    await open_session(runner)

    await run_turn(runner, types.UserContent("France's aid to Guatemala since 2010?"))

    assert planner.asks == [tuple(asks), ("France's bilateral aid to Guatemala",)]
    assert planner.breakdowns == [(AskBreakdown(ask=1, by=("donor",)),), ()]
    assert planner.across_years == [(1,), ()]
    session = await runner.session_service.get_session(app_name="t", user_id="u", session_id="s")
    assert session.state.get(ASKS_STATE_KEY) == ["France's bilateral aid to Guatemala"]
    assert session.state.get(ASK_BREAKDOWNS_STATE_KEY) is None
    assert session.state.get(ASKS_ACROSS_YEARS_STATE_KEY) is None


async def test_a_planner_seam_failure_is_handed_back_when_wrapped_and_fatal_when_not():
    runner, model = orchestrator_runner(
        call("plan", brief="Kenya's debt", asks=["Kenya's debt"]),
        types.Part.from_text(text="Could not be calculated."),
        planner=handing_back_planner_failures(FailingPlanner()),
    )
    await open_session(runner)

    events = await run_turn(runner, types.UserContent("Kenya's debt?"))

    assert (
        "The planner could not make a plan: the planner seam ran past 60 events without settling"
        in request_text(model.requests[-1])
    )
    assert not [e for e in events if e.error_code]
    # The failure's code travels with the status that fails the plan's step.
    failures = [
        e.custom_metadata[ORCHESTRATOR_STATUS_FAILURE_KEY]
        for e in events
        if e.custom_metadata and ORCHESTRATOR_STATUS_FAILURE_KEY in e.custom_metadata
    ]
    assert failures == [{"code": "internal_error"}]

    bare, _ = orchestrator_runner(
        call("plan", brief="Kenya's debt", asks=["Kenya's debt"]), planner=FailingPlanner()
    )
    await open_session(bare)
    with pytest.raises(PlannerSeamError):
        await run_turn(bare, types.UserContent("Kenya's debt?"))


async def _breaking_resolver(**kwargs: Any) -> dict[str, Any]:
    raise RuntimeError("upstream at http://internal.example/v2 broke")


def _tool_answers(request: Any) -> dict[str, Any]:
    """Each function response in `request`, by tool name."""
    return {
        p.function_response.name: p.function_response.response
        for c in request.contents
        for p in c.parts or []
        if p.function_response
    }


def _results_of(content: types.Content, name: str) -> list[dict[str, Any]]:
    """The results `content` gives calls to `name`."""
    return [
        p.function_response.response or {}
        for p in content.parts or []
        if p.function_response and p.function_response.name == name
    ]


def _paired_results(request: Any) -> list[tuple[str, list[dict[str, Any]]]]:
    """Each function call in `request`, in order, by name, with the results
    the content after it gives that name. The request carries no call ids,
    so Gemini pairs a call with the responses that follow it."""
    contents = request.contents
    pairs = []
    for place, content in enumerate(contents):
        following = contents[place + 1] if place + 1 < len(contents) else None
        for part in content.parts or []:
            if part.function_call:
                name = part.function_call.name or ""
                pairs.append((name, _results_of(following, name) if following else []))
    return pairs


def _assert_one_result_per_call(request: Any) -> None:
    """Every call in `request` is followed by exactly one result, and no
    result stands without its call."""
    pairs = _paired_results(request)
    assert [len(results) for _, results in pairs] == [1] * len(pairs)
    responses = [p for c in request.contents for p in c.parts or [] if p.function_response]
    assert len(responses) == len(pairs)


def _the_hand_back(request: Any, name: str) -> dict[str, Any]:
    """The one result `request` shows for the agent's call to `name`."""
    ((result,),) = [results for called, results in _paired_results(request) if called == name]
    assert HANDED_OFF not in result
    return result


async def test_a_trailing_hand_back_is_the_one_result_of_its_plan_call():
    """The model's request right after a plan ran ends on the plan's
    hand-back. ADK merges every response to the call there, the later
    replacing the earlier, so the call shows the hand-back and not `_pend`'s
    record of the hand-off."""
    runner, model = orchestrator_runner(
        call("plan", brief="Kenya's debt service since 2015", asks=["Kenya's debt service"]),
        types.Part.from_text(text="Charted."),
        planner=ScriptedPlannerModel(),
        tool_surface=ScriptedSurface(),
        executor=_describes,
    )
    await open_session(runner)
    approval = single_pause(await run_turn(runner, types.UserContent("Kenya's debt service?")))
    revision = approval.args["payload"]["revision"]
    approve = create_request_input_response(approval.id, {"result": approve_answer(revision)})
    await run_turn(runner, types.Content(role="user", parts=[approve]))

    request = model.requests[-1]
    assert _results_of(request.contents[-1], "plan")
    result = _the_hand_back(request, "plan")
    assert result["result"] == "The approved plan ran. Chart its frames and answer."
    assert result["frames"]
    _assert_one_result_per_call(request)


async def test_a_hand_back_in_history_is_the_one_result_of_its_plan_call():
    """A later request holds the plan's hand-back in its history, with the
    agent's `ask_user` call and its answer after it. ADK pairs a call in
    history with its last response, so the plan call still shows the
    hand-back alone, and the question its answer alone."""
    runner, model = orchestrator_runner(
        call("plan", brief="Kenya's debt service", asks=["Kenya's debt service"]),
        call("ask_user", questions=[_CONGO]),
        types.Part.from_text(text="Kenya's debt service could not be calculated."),
    )
    await open_session(runner)
    pause = single_pause(await run_turn(runner, types.UserContent("Kenya's debt service?")))
    answer = create_request_input_response(pause.id, {"result": {"spec_id": "o1"}})
    await run_turn(runner, types.Content(role="user", parts=[answer]))

    request = model.requests[-1]
    assert not _results_of(request.contents[-1], "plan")
    assert _the_hand_back(request, "plan")["result"].startswith("The planner could not make a plan")
    assert _the_hand_back(request, "ask_user")["result"].startswith(
        "You asked the user: Which Congo?\nThe user chose: Republic of the Congo"
    )
    _assert_one_result_per_call(request)


@pytest.mark.parametrize("answers", ["plan", "ask_user"])
async def test_an_answer_no_call_waits_on_is_refused_and_not_read_as_a_seed(answers: str):
    """A hand-back that answers a call finds no call of the agent's waiting
    on it: `orchestrate` raises, and records no seed hand-back."""
    orchestrate = orchestrate_node(
        orchestrator=cast("Any", SimpleNamespace(name="dfo_orchestrator"))
    )
    ctx = SimpleNamespace(session=SimpleNamespace(events=[]), state={}, event_author="orchestrate")
    hand_back = {"answers": answers, "status": "Reading", "text": "The approved plan ran."}

    with pytest.raises(RuntimeError, match=f"no {answers} call"):
        [_ async for _ in orchestrate(cast("Any", ctx), hand_back)]
    assert ctx.state == {}


async def test_a_tool_that_raises_answers_the_agent_with_its_class_and_the_turn_goes_on(caplog):
    runner, model = orchestrator_runner(
        call("search", query="Kenya's GDP?", parts=[{"variable": "GDP"}]),
        types.Part.from_text(text="Search failed, so I cannot give Kenya's GDP."),
        resolver=_breaking_resolver,
    )
    await open_session(runner)

    with caplog.at_level(logging.ERROR, logger="orchestrator.tool_failures"):
        events = await run_turn(runner, types.UserContent("Kenya's GDP?"))

    answer = _tool_answers(model.requests[-1])["search"]
    assert answer["error"].startswith("search failed inside DFO (RuntimeError).")
    assert "internal.example" not in answer["error"]
    assert (answer["code"], answer["retry_mode"]) == (HarnessCode.RAISED, "do_not_retry")
    assert not [e for e in events if e.error_code]
    (record,) = [r for r in caplog.records if r.name == "orchestrator.tool_failures"]
    assert record.levelno == logging.ERROR
    assert record.exc_info is not None and record.exc_info[0] is RuntimeError
    assert record.args[0] == "search"


async def test_a_made_up_tool_name_keeps_adk_answer_and_logs_no_failure(caplog):
    runner, model = orchestrator_runner(
        call("summon_data", topic="GDP"), types.Part.from_text(text="Sorry.")
    )
    await open_session(runner)

    with caplog.at_level(logging.ERROR, logger="orchestrator.tool_failures"):
        events = await run_turn(runner, types.UserContent("Kenya's GDP?"))

    answer = _tool_answers(model.requests[-1])["summon_data"]
    assert "The tools you can call are: " in answer["error"]
    assert not [e for e in events if e.error_code]
    assert not [r for r in caplog.records if r.name == "orchestrator.tool_failures"]


async def test_an_approved_plan_runs_and_its_final_frames_come_back_to_the_agent():
    runner, model = orchestrator_runner(
        call(
            "plan",
            brief="Kenya's debt service since 2015",
            asks=["Kenya's debt service since 2015"],
        ),
        types.Part.from_text(text="Charted."),
        types.Part.from_text(text="Uganda next."),
        planner=ScriptedPlannerModel(),
        tool_surface=ScriptedSurface(),
        executor=_describes,
    )
    await open_session(runner)

    approval = single_pause(await run_turn(runner, types.UserContent("Kenya's debt service?")))
    assert approval.id == APPROVE_INTERRUPT
    revision = approval.args["payload"]["revision"]
    approve = create_request_input_response(approval.id, {"result": approve_answer(revision)})
    events = await run_turn(runner, types.Content(role="user", parts=[approve]))

    request = model.requests[-1]
    (hand_back,) = [c for c in request.contents if _results_of(c, "plan")]
    assert hand_back.role == "user"
    (body,) = _results_of(hand_back, "plan")
    assert body["result"] == "The approved plan ran. Chart its frames and answer."
    assert body["frames"] and all(f["frame"].startswith("f") for f in body["frames"])
    assert body["step_summaries"]
    assert body["plan_record"].startswith(f"Plan {revision[:12]} (approved)")
    assert not carries_an_interrupt(request)
    assert not quotes_another_agent(request)
    session = await runner.session_service.get_session(app_name="t", user_id="u", session_id="s")
    assert set(session.state[ORCHESTRATOR_FRAMES_KEY]) == {f["frame"] for f in body["frames"]}
    assert not [e for e in events if e.error_code]

    follow_up = await run_turn(runner, types.UserContent("And Uganda?"))

    assert not [e for e in follow_up if e.error_code]
    assert "The approved plan ran" in request_text(model.requests[-1])
    assert not carries_an_interrupt(model.requests[-1])


async def test_a_searched_and_approved_turn_copies_no_large_container_it_reads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ADK's `State` deep-copies a dict or list on its first read
    (`_take_snapshot`), which costs milliseconds per read for the frames,
    spec blocks and readings a turn holds. A turn reads them through
    `stored_mapping` and `stored_sequence`, which take no copy; the small
    containers it reads through `State` are still copied, which shows the
    spy sees ADK's reads."""
    copied: list[str] = []
    take_snapshot = State._take_snapshot

    def spy(self: State, key: str, value: Any) -> None:
        copied.append(key)
        take_snapshot(self, key, value)

    monkeypatch.setattr(State, "_take_snapshot", spy)
    runner, _ = orchestrator_runner(
        search_call("Kenya debt service", "debt service"),
        call(
            "plan",
            brief="Kenya's debt service since 2015",
            asks=["Kenya's debt service since 2015"],
        ),
        types.Part.from_text(text="Charted."),
        planner=ScriptedPlannerModel(),
        tool_surface=ScriptedSurface(),
        executor=_describes,
        resolver=CannedSearch(definite_envelope(sentence="Kenya debt service")),
    )
    large: dict[str, Any] = {
        ORCHESTRATOR_FRAMES_KEY: {"f" + "0" * 30: {"ref": {}, "view": {"dimensions": []}}},
        ORCHESTRATOR_READINGS_KEY: {_SCRIPTED_SPEC: ["sv/1"]},
        ORCHESTRATOR_PRIOR_LEDGER_KEY: {"readings": [], "spec_ids": [], "warnings": {}},
        ORCHESTRATOR_PLAN_RECORDS_KEY: [],
    }
    await open_session(runner, **large)

    approval = single_pause(await run_turn(runner, types.UserContent("Kenya's debt service?")))
    approve = create_request_input_response(
        approval.id, {"result": approve_answer(approval.args["payload"]["revision"])}
    )
    events = await run_turn(runner, types.Content(role="user", parts=[approve]))

    assert not [e for e in events if e.error_code]
    assert ORCHESTRATOR_CALLS_KEY in copied
    assert not {*copied} & {*large, ORCHESTRATOR_SPECS_KEY, ORCHESTRATOR_SEARCHES_KEY}


@pytest.mark.parametrize("describes", [True, False])
async def test_the_hand_back_reports_each_ask_with_its_output_status_or_decline(describes: bool):
    runner, model = orchestrator_runner(
        call(
            "plan",
            brief="Kenya's debt service since 2015, ranked against East Africa",
            asks=["Kenya's debt service each year", "a ranking of East African countries"],
        ),
        types.Part.from_text(text="Answered."),
        planner=AskingPlanner(),
        tool_surface=ScriptedSurface(),
        executor=_describes if describes else _refuses_to_describe,
    )
    await open_session(runner)

    approval = single_pause(await run_turn(runner, types.UserContent("Kenya's debt service?")))
    approve = create_request_input_response(
        approval.id, {"result": approve_answer(approval.args["payload"]["revision"])}
    )
    await run_turn(runner, types.Content(role="user", parts=[approve]))

    after_plan = request_text(model.requests[-1])
    body = json.loads(next(line for line in after_plan.split("\n") if line.startswith('{"frames"')))
    session = await runner.session_service.get_session(app_name="t", user_id="u", session_id="s")
    workflow = Plan.model_validate_json(session.state["plan"]).workflow
    assert workflow is not None
    (final_step,) = workflow.terminal_steps
    output = (
        {"step": final_step, "status": "data", "frame": body["frames"][0]["frame"]}
        if describes
        else {
            "step": final_step,
            "status": "not_chartable",
            "reason": body["frames_not_chartable"][0]["reason"],
        }
    )
    assert body["asks"] == [
        {"ask": "Kenya's debt service each year", "answered_by": [output]},
        {
            "ask": "a ranking of East African countries",
            "declined": DECLINE_REASONS[UnsupportedOperation.UNSUPPORTED_ASK],
        },
    ]


class _AmendedPlanner(ScriptedPlannerModel):
    """`ScriptedPlannerModel` whose plan answers each brief ask but the last
    with its final step and records the last as removed by the user's
    amendment, as `Plan.amend` does when the amendment removes its answer."""

    async def __call__(self, **kwargs: Any) -> dict[str, Any]:
        plan = Plan.model_validate((await super().__call__(**kwargs))["plan"])
        asks = load_asks(kwargs["ctx"].state)
        final = plan.evidence_requirements[-1].id
        plan = plan.model_copy(
            update={
                "asks": asks,
                "answered": tuple(
                    AskAnswer(ask=number, outputs=(final,)) for number in range(len(asks) - 1)
                ),
                "dropped_asks": (len(asks) - 1,),
            }
        )
        return {"plan": plan.model_dump(mode="json"), "summary": {"version": plan.version}}


class _AmendedRefusingReplanner(_AmendedPlanner):
    """`_AmendedPlanner` that refuses to draft again once a run has stopped."""

    async def __call__(self, **kwargs: Any) -> dict[str, Any]:
        if kwargs.get("refusal") is None:
            return await super().__call__(**kwargs)
        return {"refused": {"text": "no plan avoids the empty sum", "final": False}}


@pytest.mark.parametrize("ending", ["ran", "stopped", "refused", "refused_after_computed"])
async def test_the_hand_back_reports_an_ask_the_users_amendment_removed_as_removed(ending: str):
    """Whether the run ran, stopped, or stopped and the planner refused to
    replan, the hand-back quotes the user's amendments in order and its lead
    keeps the removed ask out of the answer, so the agent does not report it
    as a failure or fetch it."""
    planner, surface = {
        "ran": (_AmendedPlanner(), ScriptedSurface()),
        "stopped": (_AmendedPlanner(), RefusingGetDataSurface("member_not_resolvable")),
        "refused": (_AmendedRefusingReplanner(), _EmptyAggregateSurface()),
        "refused_after_computed": (_RemovingRefusingReplanner(), _EmptyAggregateSurface()),
    }[ending]
    runner, model = orchestrator_runner(
        call(
            "plan",
            brief="ODA received by Ghana, Kenya and Nigeria",
            asks=["ODA to Ghana", "ODA to Kenya", "ODA to Nigeria"],
        ),
        types.Part.from_text(text="Answered."),
        planner=planner,
        tool_surface=surface,
        executor=_describes,
    )
    await open_session(runner)

    amendments = ["Show each year.", "Drop Nigeria, just Ghana and Kenya."]
    approval = single_pause(await run_turn(runner, types.UserContent("ODA to three countries?")))
    for words in amendments:
        amend = create_request_input_response(
            approval.id, {"result": {"approved": None, "amendment": words}}
        )
        approval = single_pause(await run_turn(runner, types.Content(role="user", parts=[amend])))
    approve = create_request_input_response(
        approval.id, {"result": approve_answer(approval.args["payload"]["revision"])}
    )
    await run_turn(runner, types.Content(role="user", parts=[approve]))

    after_plan, body = handed_back(model)
    assert body["amendments"] == amendments
    assert (
        'The user removed from the plan "ODA to Nigeria": leave it out of the answer'
        in (body["result"])
    )
    if ending.startswith("refused"):
        assert body["result"].startswith("The planner could not make a plan")
        assert "Every other ask was not computed" not in body["result"]
        if ending == "refused":
            assert body["asks"] == [{"ask": "ODA to Nigeria", "dropped": _DROPPED}]
            return
        ghana, nigeria = body["asks"]
        assert ghana["ask"] == "ODA to Ghana"
        assert ghana["earlier_plan"] is True
        assert [entry["status"] for entry in ghana["answered_by"]] == ["data"]
        assert nigeria == {"ask": "ODA to Nigeria", "dropped": _DROPPED}
        assert "Every other ask the plan kept was not computed." in body["result"]
        return
    ghana, kenya, nigeria = body["asks"]
    status = "failed" if ending == "stopped" else "data"
    assert [entry["status"] for entry in ghana["answered_by"] + kenya["answered_by"]] == [
        status,
        status,
    ]
    assert set(nigeria) == {"ask", "dropped"}
    assert '- 2 "ODA to Nigeria": removed by the user\'s amendment' in after_plan.split("\n")
    if ending == "stopped":
        assert 'Not computed: "ODA to Ghana"; "ODA to Kenya".' in body["result"]


class _BreakdownRefusingSurface(ScriptedSurface):
    """`ScriptedSurface` whose `get_data` refuses the breakdown it was asked for,
    as apis refuses a code system it cannot read."""

    async def call_tool(self, **kwargs: Any) -> ToolCallResult:
        if kwargs["tool"].name != "get_data":
            return await super().call_tool(**kwargs)
        return ToolCallResult(
            is_error=True,
            structured_content={
                "code": "capability_unavailable",
                "retry_mode": "do_not_retry",
                "message": "this source cannot be broken down by that code system",
                "fix": {"pointer": "/breakdown/code_system"},
            },
            text="refused",
        )


async def test_a_breakdown_refused_at_run_time_hands_back_a_failed_step_and_ask():
    runner, model = orchestrator_runner(
        call("plan", brief="Kenya's debt service since 2015", asks=["Kenya's debt service"]),
        types.Part.from_text(text="It could not be computed."),
        planner=AskingPlanner(),
        tool_surface=_BreakdownRefusingSurface(),
        executor=_describes,
    )
    await open_session(runner)

    approval = single_pause(await run_turn(runner, types.UserContent("Kenya's debt service?")))
    approve = create_request_input_response(
        approval.id, {"result": approve_answer(approval.args["payload"]["revision"])}
    )
    await run_turn(runner, types.Content(role="user", parts=[approve]))

    after_plan = request_text(model.requests[-1])
    body = json.loads(next(line for line in after_plan.split("\n") if line.startswith('{"frames"')))
    assert body["failed_step"]["step_id"] == "retrieve"
    assert body["failed_step"]["code"] == "capability_unavailable"
    # The ask's output reads the refused retrieval, so it failed with it.
    failed = {"status": "failed", "failed_at": "retrieve", "code": "capability_unavailable"}
    assert body["asks"] == [
        {"ask": "Kenya's debt service", "answered_by": [{"step": "totals", **failed}]}
    ]
    # An output the failed step does not feed, and that never ran, was not run.
    session = await runner.session_service.get_session(app_name="t", user_id="u", session_id="s")
    stopped = _stop_causes(Plan.model_validate_json(session.state["plan"]), "totals")
    assert stopped == {"totals": "totals"}
    upstream = PlanAsk(ask="the debt service series", outputs=("retrieve",))
    assert _ask_status(upstream, {}, {"step_id": "totals"}, stopped)["answered_by"] == [
        {"step": "retrieve", "status": "not_run"}
    ]


def test_an_ask_answered_in_part_hands_back_its_steps_and_the_part_not_computed() -> None:
    code = UnsupportedOperation.CALCULATION_OVER_COMBINED
    ask = PlanAsk(
        ask="bilateral, imputed and their total",
        outputs=("joined",),
        declined=code,
        not_computed="their total",
    )

    assert _ask_status(ask, {}, None, {}) == {
        "ask": "bilateral, imputed and their total",
        "answered_by": [{"step": "joined", "status": "not_run"}],
        "not_computed": {"part": "their total", "reason": DECLINE_REASONS[code]},
    }


def _ranking(
    entities: int,
    *,
    limit: int | None = None,
    label: str = "",
    years: tuple[str, ...] = ("2023",),
    warnings: tuple[str, ...] = (),
) -> StepSummary:
    """The summary of a rank step over `entities` countries named `label` after
    their number, as `analysis_mcp` renders its digest, with the row count of
    the frame it keeps and, under a `limit`, the entries it ranked. Several
    `years` rank each year on its own. `warnings` are the run's."""
    frame = Frame.build(
        role_keys=["observationAbout"],
        rows=[
            {
                "stat_var": "dcid:LifeExpectancy",
                "observationAbout": f"country/C{number:02d}{label}",
                "time_period": year,
                "value": 80.0 - number,
                "unit": "dcid:Years",
            }
            for number in range(entities)
            for year in years
        ],
    )
    result = run_rank(
        frame,
        rank_by=("observationAbout",),
        aggregation=None,
        window_start=None,
        window_end=None,
        ascending=False,
        per_period=len(years) > 1,
        limit=limit,
    )
    digest = render_rank_digest(
        rank_by=("observationAbout",),
        result=dataclasses.replace(result, warnings=warnings),
        op_spec_hash="ops_" + "1" * 32,
        content_version="ocv_" + "1" * 32,
        execution_identity=current_execution_identity(),
    )
    ranked = len(result.entity_ranks) if limit is not None else None
    return StepSummary(digest, "rank", result.frame.num_rows, ranked)


def _handed_back(
    digest: str,
    tool: str | None,
    rows: int | None = None,
    ranked_before_limit: int | None = None,
) -> list[str]:
    """The lines the orchestrator reads of one `tool` step's `digest` in the hand-back."""
    summary = StepSummary(digest, tool, rows, ranked_before_limit)
    summaries = HandBack(_refuses_to_describe, summaries=[summary]).listing({}, None)[
        "step_summaries"
    ]
    return summaries[0].split("\n")


def _rows_of(digest: str) -> list[str]:
    return [line for line in digest.split("\n") if line.startswith("  #")]


def test_a_ranking_of_twelve_hands_back_its_first_row_its_last_row_and_its_count() -> None:
    ranking = _ranking(12)
    rows = _rows_of(ranking.digest)

    handed = _handed_back(*ranking)

    assert len(rows) == 12
    assert "12 ranked entit(y/ies):" in handed
    assert [row for row in rows if row in handed] == [rows[0], rows[-1]]


def test_a_ranking_of_four_is_handed_back_whole() -> None:
    ranking = _ranking(4)

    assert _rows_of(ranking.digest) == [
        line for line in _handed_back(*ranking) if line.startswith("  #")
    ]


@pytest.mark.parametrize("limit", [10, 20], ids=["limit-10", "limit-20"])
def test_a_limited_ranking_hands_back_every_row_its_frame_keeps_and_no_others(limit: int) -> None:
    """The digest lists the rows the frame keeps, a `limit` of 20 or fewer of the
    70 ranked entities, and they are handed back whole."""
    ranking = _ranking(70, limit=limit)
    rows = _rows_of(ranking.digest)

    handed = _handed_back(*ranking)

    assert ranking.rows == limit and len(rows) == limit
    assert f"{limit} ranked entit(y/ies):" in handed
    assert f"limit {limit} (kept {limit} of 70 ranked entities)" in handed[0]
    assert _rows_of("\n".join(handed)) == rows


@pytest.mark.parametrize("periods", [3, 20], ids=["3-periods", "20-periods"])
def test_a_limited_ranking_of_each_period_hands_back_its_header_and_closing_lines(
    periods: int,
) -> None:
    """Blocks of 3 rows hold few lines each, yet a line per period would pass the
    hand-back cap and cut the lines after them."""
    years = tuple(str(2000 + offset) for offset in range(periods))
    ranking = _ranking(40, limit=3, years=years, warnings=("the window is sparse",))

    handed = _handed_back(*ranking)

    assert f"kept {3 * periods} of {40 * periods} ranked rows" in handed[0]
    assert not [line for line in handed if "ranked entit(y/ies)" in line or "#" in line]
    assert any(line.startswith("unit: ") for line in handed)
    assert "  - the window is sparse" in handed


@pytest.mark.parametrize(
    ("limit", "kept"), [(None, 70), (24, 24), (30, 30)], ids=["unlimited", "limit-24", "limit-30"]
)
def test_a_ranking_whose_frame_keeps_more_rows_than_the_digest_lists_hands_back_no_closing_row(
    limit: int | None, kept: int
) -> None:
    """The digest prints 20 rows. Rank 20 is the last row only of a frame that
    keeps 20, so only the first row is handed back, with a marker that says
    how many of the frame's rows the digest lists."""
    ranking = _ranking(70, limit=limit)
    rows = _rows_of(ranking.digest)

    handed = _handed_back(*ranking)

    assert ranking.rows == kept and len(rows) == 20
    assert [row for row in rows if row in handed] == [rows[0]]
    assert any(f"20 of the frame's {kept} rows" in line for line in handed)


def test_a_limited_ranking_over_the_card_cap_hands_back_its_first_and_last_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A frame that keeps more rows than a card draws is trimmed even when the
    digest lists them all (here a card draws 9 and the frame keeps 10)."""
    monkeypatch.setattr(plan_done_module, "MAX_SERIES_PER_COMPONENT", 9)
    ranking = _ranking(70, limit=10)
    rows = _rows_of(ranking.digest)

    handed = _handed_back(*ranking)

    assert [row for row in rows if row in handed] == [rows[0], rows[9]]


def test_a_ranking_stored_without_its_limit_is_trimmed_to_its_first_and_last_kept_row() -> None:
    """A summary stored before `ranked_before_limit` was carried has no value."""
    ranking = _ranking(70, limit=10)._replace(ranked_before_limit=None)
    rows = _rows_of(ranking.digest)

    handed = _handed_back(*ranking)

    assert [row for row in rows if row in handed] == [rows[0], rows[9]]
    assert (
        StepSummary.of({"tool": "rank", "digest": ranking.digest, "rows": 10}).ranked_before_limit
        is None
    )


def test_a_limited_ranking_too_long_to_hand_back_whole_is_trimmed_not_cut_mid_row() -> None:
    ranking = _ranking(70, limit=20, label="x" * 150)
    rows = _rows_of(ranking.digest)

    handed = _handed_back(*ranking)

    assert sum(len(line) + 1 for line in rows[:20]) > plan_done_module._WHOLE_RANKING_CHARS
    assert [row for row in rows if row in handed] == [rows[0], rows[19]]


@pytest.mark.parametrize("rows", [None, 0], ids=["unknown", "empty"])
def test_a_ranking_of_unknown_or_empty_frame_hands_back_its_count_and_no_rows(
    rows: int | None,
) -> None:
    ranking = _ranking(12)

    handed = _handed_back(ranking.digest, "rank", rows)

    assert "12 ranked entit(y/ies):" in handed
    assert _rows_of("\n".join(handed)) == []


# The shapes `analysis_mcp.digest` writes for the six other per-entity digests:
# the digest's opening line, the heading that states the count, one line.
_COUNT_ONLY_SHAPES = {
    "aggregate": (
        "aggregate: sum grouped by reporter",
        "6 group(s):",
        "  reporter=dcid:USA [1 row]: 1.0",
    ),
    "growth": (
        "growth: 2020..2023, entity_by reporter",
        "6 entit(y/ies):",
        "  reporter=dcid:USA: +10.0% (1 -> 2)",
    ),
    "filter": (
        "filter: 1 predicate(s) applied, 6/6 row(s) kept (0 dropped)",
        "6 kept row(s):",
        "  reporter=dcid:USA: 1.0 [Count]",
    ),
    "join": (
        "join: on reporter + time_period, 6 matched key(s) (0 left-only, 0 right-only, dropped)",
        "6 matched key(s):",
        "  reporter=dcid:USA [2021]: left=1.0, right=2.0",
    ),
    "quotient_of": (
        "quotient_of: 6 matched period(s), in Count",
        "6 quotient(s), numerator / denominator, in Count:",
        "  2020: 1 / 2 = 0.5",
    ),
    "share_of": (
        "share_of: 6 matched period(s), in per cent (Percent)",
        "6 period share(s):",
        "  2020: 1 / 2 = 50.000%",
    ),
}
_KEYED_SHARE = (
    "share_of: 1 key(s), 6 matched (key, period) pair(s), in per cent (Percent)",
    "reporter=dcid:USA [6 period(s)], first 2020: 50.000%; latest 2025: 50.000%:",
    "  2020: 1 / 2 = 50.000%",
)


def _entity_digest(shape: tuple[str, str, str], *, lines: int) -> str:
    opening, heading, line = shape
    return "\n".join([opening, "", heading, *[line] * lines, "", "unit: Count"])


@pytest.mark.parametrize(
    ("tool", "shape"),
    [*_COUNT_ONLY_SHAPES.items(), ("share_of", _KEYED_SHARE)],
    ids=[*_COUNT_ONLY_SHAPES, "keyed share_of"],
)
def test_a_per_entity_digest_past_four_lines_hands_back_its_count_and_none_of_its_lines(
    tool: str, shape: tuple[str, str, str]
) -> None:
    _, heading, line = shape

    handed = _handed_back(_entity_digest(shape, lines=5), tool)

    assert heading in handed
    assert line not in handed
    assert _handed_back(_entity_digest(shape, lines=4), tool).count(line) == 4


@pytest.mark.parametrize("tool", [None, "compare", "get_data"], ids=["no-tool", "compare", "data"])
def test_a_digest_of_a_step_not_known_to_print_one_line_per_entity_is_handed_back_whole(
    tool: str | None,
) -> None:
    """A summary stored without its tool, as a run paused before the tool was
    kept leaves it, reads whole."""
    digest = _entity_digest(_COUNT_ONLY_SHAPES["growth"], lines=9)

    assert _handed_back(digest, tool) == digest.split("\n")


async def _refuses_to_describe(*, tool: str, args: dict[str, Any], on_progress: Any = None) -> Any:
    assert tool == "describe_frame", tool
    return {
        "outcome": "refusal",
        "refusal": {"code": "publish_not_configured", "retry_mode": "do_not_retry"},
    }


class RefusingGetDataSurface(ScriptedSurface):
    """`ScriptedSurface` whose `get_data` answers the refusal `code` names."""

    def __init__(self, code: str) -> None:
        super().__init__()
        self.code = code

    async def call_tool(self, *, tool: Any, arguments: Any, on_progress: Any = None) -> Any:
        if tool.name != "get_data":
            return await super().call_tool(tool=tool, arguments=arguments, on_progress=on_progress)
        refusal = {
            "code": self.code,
            "retry_mode": "do_not_retry",
            "fix": {},
            "message": "this deployment cannot serve the request",
        }
        return ToolCallResult(is_error=True, structured_content=refusal, text=refusal["message"])


@pytest.mark.parametrize("code", ["capability_unavailable", "member_not_resolvable"])
async def test_a_step_refused_for_a_missing_capability_closes_the_part_it_served(code: str):
    """A capability the platform lacks fails any brief and any fetch of the
    step's data, so the plan call is not given back and the step's spec_id
    is refused to `fetch`; another refusal leaves both open."""
    runner, model = orchestrator_runner(
        call(
            "plan",
            brief="Kenya's debt service since 2015",
            asks=["Kenya's debt service since 2015"],
        ),
        types.Part.from_text(text="That part is unavailable."),
        planner=ScriptedPlannerModel(),
        tool_surface=RefusingGetDataSurface(code),
        executor=_describes,
    )
    await open_session(runner)

    approval = single_pause(await run_turn(runner, types.UserContent("Kenya's debt service?")))
    approve = create_request_input_response(
        approval.id, {"result": approve_answer(approval.args["payload"]["revision"])}
    )
    await run_turn(runner, types.Content(role="user", parts=[approve]))

    after_plan = request_text(model.requests[-1])
    body = json.loads(next(line for line in after_plan.split("\n") if line.startswith('{"frames"')))
    assert body["failed_step"]["code"] == code
    session = await runner.session_service.get_session(app_name="t", user_id="u", session_id="s")
    missing = code == "capability_unavailable"
    assert bool(session.state[ORCHESTRATOR_CALLS_KEY].get("plan_given_back")) is not missing
    assert (_SCRIPTED_SPEC in (session.state.get(ORCHESTRATOR_UNAVAILABLE_KEY) or {})) is missing


async def test_a_plan_whose_frames_cannot_be_described_hands_back_the_reason_not_a_frame():
    runner, model = orchestrator_runner(
        call(
            "plan",
            brief="Kenya's debt service since 2015",
            asks=["Kenya's debt service since 2015"],
        ),
        types.Part.from_text(text="Here is what the plan computed."),
        planner=ScriptedPlannerModel(),
        tool_surface=ScriptedSurface(),
        executor=_refuses_to_describe,
    )
    await open_session(runner)

    approval = single_pause(await run_turn(runner, types.UserContent("Kenya's debt service?")))
    approve = create_request_input_response(
        approval.id, {"result": approve_answer(approval.args["payload"]["revision"])}
    )
    await run_turn(runner, types.Content(role="user", parts=[approve]))

    after_plan = request_text(model.requests[-1])
    assert "its result cannot be drawn" in after_plan
    body = json.loads(
        after_plan.split("Only a frame listed under frames can be charted.\n", 1)[1].split("\n", 1)[
            0
        ]
    )
    assert body["frames"] == []
    assert body["frames_not_chartable"] == [
        {"step": "aggregate", "reason": "this step's data view could not be prepared"}
    ]
    assert "publish_not_configured" not in after_plan


async def test_an_amendment_redrafts_and_the_amended_plan_runs():
    runner, model = orchestrator_runner(
        call(
            "plan",
            brief="Kenya's debt service since 2015",
            asks=["Kenya's debt service since 2015"],
        ),
        types.Part.from_text(text="Charted."),
        planner=ScriptedPlannerModel(),
        tool_surface=ScriptedSurface(),
        executor=_describes,
    )
    await open_session(runner)

    first = single_pause(await run_turn(runner, types.UserContent("Kenya's debt service?")))
    amend = create_request_input_response(
        first.id, {"result": {"approved": None, "amendment": "only 2020"}}
    )
    second = single_pause(await run_turn(runner, types.Content(role="user", parts=[amend])))
    assert second.args["payload"]["version"] == first.args["payload"]["version"] + 1
    approve = create_request_input_response(
        second.id, {"result": approve_answer(second.args["payload"]["revision"])}
    )
    await run_turn(runner, types.Content(role="user", parts=[approve]))

    assert "The approved plan ran." in request_text(model.requests[-1])


class _CountingSurface(ScriptedSurface):
    """`ScriptedSurface` that records the name of every tool a plan step calls."""

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[str] = []

    async def call_tool(self, **kwargs: Any) -> ToolCallResult:
        self.calls.append(kwargs["tool"].name)
        return await super().call_tool(**kwargs)


# Three plans awaiting approval as the release that listed members stored
# them, each beside the revision its card named. Their workflow's sources
# hold `stat_vars`, `stat_var_truncated` and `pinned`, which
# `planner.workflow.SelectedSource` no longer has.
_PLANS_WITH_MEMBER_LISTS: dict[str, Any] = json.loads(
    (Path(__file__).parent / "fixtures" / "plans_with_member_lists.json").read_text()
)
_UNREADABLE_PLAN = json.dumps(_PLANS_WITH_MEMBER_LISTS["aggregate"]["plan"])


def _stored_state(runner: Runner) -> dict[str, Any]:
    """The live state of the session `open_session` opened."""
    service = cast(InMemorySessionService, runner.session_service)
    return service.sessions["t"]["u"]["s"].state


def _plan_warnings(caplog: pytest.LogCaptureFixture) -> list[tuple[object, ...]]:
    """The arguments of each warning `planner.plan` logged."""
    return [
        cast("tuple[object, ...]", record.args)
        for record in caplog.records
        if record.name == "planner.plan" and record.levelno == logging.WARNING
    ]


@pytest.mark.parametrize(
    ("stored", "answer"),
    [
        ("aggregate", "approves"),
        ("rank", "approves"),
        ("declined", "approves"),
        ("aggregate", "amends"),
    ],
)
async def test_an_answer_to_a_plan_stored_with_member_lists_gets_a_new_plan(
    stored: str, answer: str, caplog: pytest.LogCaptureFixture
):
    """A plan awaiting approval across the release cannot be read. Approving
    it or amending it retrieves nothing: the question is planned from the
    start, the new card says why, and approving that one runs it."""
    surface = _CountingSurface()
    runner, model = orchestrator_runner(
        call(
            "plan",
            brief="Kenya's debt service since 2015",
            asks=["Kenya's debt service since 2015"],
        ),
        types.Part.from_text(text="Charted."),
        planner=ScriptedPlannerModel(),
        tool_surface=surface,
        executor=_describes,
    )
    await open_session(runner)
    first = single_pause(await run_turn(runner, types.UserContent("Kenya's debt service?")))
    held = _PLANS_WITH_MEMBER_LISTS[stored]
    _stored_state(runner)["plan"] = json.dumps(held["plan"])
    result = (
        approve_answer(held["revision"])
        if answer == "approves"
        else {"approved": None, "amendment": "only 2020"}
    )

    with caplog.at_level(logging.WARNING, logger="planner.plan"):
        second = single_pause(
            await run_turn(
                runner,
                types.Content(
                    role="user",
                    parts=[create_request_input_response(first.id, {"result": result})],
                ),
            )
        )

    assert second.id == APPROVE_INTERRUPT
    assert second.args["payload"]["refusal_note"] == UNREADABLE_PLAN_NOTE
    assert second.args["payload"]["version"] == 1
    assert surface.calls == []
    ((state_key, entries, fields),) = _plan_warnings(caplog)
    assert (state_key, entries) == ("plan", 1)
    assert "workflow.sources.stat_vars" in str(fields)
    assert held["plan"]["question"] not in caplog.text

    approve = create_request_input_response(
        second.id, {"result": approve_answer(second.args["payload"]["revision"])}
    )
    await run_turn(runner, types.Content(role="user", parts=[approve]))

    assert "get_data" in surface.calls
    assert "The approved plan ran." in request_text(model.requests[-1])


async def test_a_plan_call_drafts_over_an_earlier_plan_stored_with_member_lists():
    """A session's earlier plan call left a plan the release cannot read.
    The next plan call drafts its own plan as it would with none on record,
    and its card says nothing of the earlier one, which no card of this
    call showed."""
    runner, _ = orchestrator_runner(
        call(
            "plan",
            brief="Kenya's debt service since 2015",
            asks=["Kenya's debt service since 2015"],
        ),
        planner=ScriptedPlannerModel(),
        tool_surface=ScriptedSurface(),
        executor=_describes,
    )
    await open_session(runner, plan=_UNREADABLE_PLAN)

    approval = single_pause(await run_turn(runner, types.UserContent("Kenya's debt service?")))

    assert approval.id == APPROVE_INTERRUPT
    assert "refusal_note" not in approval.args["payload"]
    assert Plan.model_validate_json(_stored_state(runner)["plan"]).version == 1


class _StoredWithMemberLists(Plan):
    """A plan that is stored as `_UNREADABLE_PLAN`, whatever it holds.

    Each node writes the plan it just built before the next reads it, so no
    answer and no tool call lands a stored shape between `approve` and
    `execute`, or between `execute` and `plan_done`. This puts one there.
    """

    def model_dump_json(self, **kwargs: Any) -> str:
        del kwargs
        return _UNREADABLE_PLAN


async def test_execute_plans_again_when_the_approved_plan_was_stored_with_member_lists(
    monkeypatch: pytest.MonkeyPatch,
):
    """`execute` runs nothing of a plan it cannot read and spends no replan
    on it: the question is planned from the start and the card says why."""
    approve_plan = Plan.approve
    monkeypatch.setattr(
        Plan,
        "approve",
        lambda self, *, at: _StoredWithMemberLists.model_validate(
            approve_plan(self, at=at).model_dump()
        ),
    )
    surface = _CountingSurface()
    runner, _ = orchestrator_runner(
        call(
            "plan",
            brief="Kenya's debt service since 2015",
            asks=["Kenya's debt service since 2015"],
        ),
        planner=ScriptedPlannerModel(),
        tool_surface=surface,
        executor=_describes,
    )
    await open_session(runner)
    first = single_pause(await run_turn(runner, types.UserContent("Kenya's debt service?")))
    approve = create_request_input_response(
        first.id, {"result": approve_answer(first.args["payload"]["revision"])}
    )

    second = single_pause(await run_turn(runner, types.Content(role="user", parts=[approve])))

    assert second.id == APPROVE_INTERRUPT
    assert second.args["payload"]["refusal_note"] == UNREADABLE_PLAN_NOTE
    assert surface.calls == []
    assert not _stored_state(runner).get("replans")


async def test_the_hand_back_carries_no_record_of_a_plan_stored_with_member_lists(
    monkeypatch: pytest.MonkeyPatch,
):
    """`plan_done` hands back the frames the run produced as it does when no
    plan is on record: without the plan record, and with the slot cleared."""
    run_plan = execute_module.execute_approved_plan

    async def stored_unreadable(plan: Plan, **kwargs: Any) -> tuple[Plan, Any]:
        finished, outcome = await run_plan(plan, **kwargs)
        return _StoredWithMemberLists.model_validate(finished.model_dump()), outcome

    monkeypatch.setattr(execute_module, "execute_approved_plan", stored_unreadable)
    runner, model = orchestrator_runner(
        call(
            "plan",
            brief="Kenya's debt service since 2015",
            asks=["Kenya's debt service since 2015"],
        ),
        types.Part.from_text(text="Charted."),
        planner=ScriptedPlannerModel(),
        tool_surface=ScriptedSurface(),
        executor=_describes,
    )
    await open_session(runner)
    approval = single_pause(await run_turn(runner, types.UserContent("Kenya's debt service?")))
    revision = approval.args["payload"]["revision"]
    approve = create_request_input_response(approval.id, {"result": approve_answer(revision)})

    events = await run_turn(runner, types.Content(role="user", parts=[approve]))

    handed_back = request_text(model.requests[-1])
    assert "The approved plan ran. Chart its frames and answer." in handed_back
    assert f"Plan {revision[:12]}" not in handed_back
    assert not [e for e in events if e.error_code]
    state = _stored_state(runner)
    assert state["plan"] is None
    assert not state.get(ORCHESTRATOR_PLAN_RECORDS_KEY)


async def test_approving_a_superseded_revision_retrieves_nothing_and_proposes_again():
    surface = _CountingSurface()
    runner, _ = orchestrator_runner(
        call(
            "plan",
            brief="Kenya's debt service since 2015",
            asks=["Kenya's debt service since 2015"],
        ),
        planner=ScriptedPlannerModel(),
        tool_surface=surface,
        executor=_describes,
    )
    await open_session(runner)

    first = single_pause(await run_turn(runner, types.UserContent("Kenya's debt service?")))
    amend = create_request_input_response(
        first.id, {"result": {"approved": None, "amendment": "only 2020"}}
    )
    second = single_pause(await run_turn(runner, types.Content(role="user", parts=[amend])))
    stale = create_request_input_response(
        second.id, {"result": approve_answer(first.args["payload"]["revision"])}
    )
    third = single_pause(await run_turn(runner, types.Content(role="user", parts=[stale])))

    assert third.id == APPROVE_INTERRUPT
    assert third.args["payload"]["version"] == second.args["payload"]["version"] + 1
    assert "get_data" not in surface.calls


async def test_a_hand_back_is_still_in_view_after_a_later_pause():
    runner, model = orchestrator_runner(
        call("plan", brief="Kenya's debt service", asks=["Kenya's debt service"]),
        call(
            "ask_user",
            questions=[
                {
                    "question": "Which years?",
                    "options": [{"label": "Since 2015"}, {"label": "Since 2020"}],
                }
            ],
        ),
        types.Part.from_text(text="Kenya's debt service since 2015 could not be calculated."),
    )
    await open_session(runner)

    pause = single_pause(await run_turn(runner, types.UserContent("Kenya's debt service?")))
    answer = create_request_input_response(pause.id, {"result": {"spec_id": "o1"}})
    await run_turn(runner, types.Content(role="user", parts=[answer]))

    resumed = request_text(model.requests[-1])
    assert "Kenya's debt service?" in resumed
    assert "The planner could not make a plan" in resumed
    assert "The user chose: Since 2015" in resumed


async def test_a_second_plan_call_runs_a_planner_that_never_sees_the_first_hand_back():
    seam = PlannerSeam(
        settings=Settings(),
        surface=FixtureSurface(schemas=[]),
    )
    planner_llm = RecordingLlm(script=[], agent_name="planner", scenario="replan")
    seam._agent.model = planner_llm
    runner, model = orchestrator_runner(
        call("plan", brief="Kenya's debt, first brief", asks=["Kenya's debt, first brief"]),
        call("plan", brief="Kenya's debt, second brief", asks=["Kenya's debt, second brief"]),
        types.Part.from_text(text="Could not be calculated."),
        planner=handing_back_planner_failures(seam),
    )
    await open_session(runner)

    events = await run_turn(runner, types.UserContent("Kenya's debt?"))

    hand_back = "The planner could not make a plan"
    second_run = [r for r in planner_llm.captured if "second brief" in request_text(r)]
    assert second_run
    assert not [r for r in second_run if hand_back in request_text(r)]
    assert hand_back in request_text(model.requests[-1])
    assert not quotes_another_agent(model.requests[-1])
    # The hand-back's branch stays on the hand-back: the graph's own events
    # reach the same agents after each hand-back as before the first.
    graph_events = [e for e in events if e.author == "dfo_analyst_graph"]
    assert graph_events and {e.branch for e in graph_events} == {None}
    assert not [e for e in events if e.error_code]


def _rows_led_by_a_wide_step(step_rows: Any) -> Any:
    """`plan_display.step_rows` with a first row whose arguments run past
    4,000 characters, and a reading on every retrieval after it."""

    def rows(plan: Plan) -> list[dict[str, object]]:
        wide = {
            "id": "wide",
            "label": "Fetch every member",
            "args": {
                "spec_id": "spec_wide",
                "resolved_stat_var_dcids": [f"Wide_StatVar_{n:04d}" for n in range(300)],
            },
        }
        read = [
            {**row, "args": {**cast("dict[str, Any]", row["args"]), "stat_vars": "sum"}}
            if "spec_id" in cast("dict[str, Any]", row["args"])
            else row
            for row in step_rows(plan)
        ]
        return [wide, *read]

    return rows


@pytest.mark.parametrize(
    ("prior_spec_ids", "origin"),
    [([], "translator"), ([_SCRIPTED_SPEC], "search")],
    ids=["found_by_the_translator", "searched_in_an_earlier_turn"],
)
async def test_the_plan_record_and_every_reading_reach_the_agent_past_wide_step_arguments(
    monkeypatch: pytest.MonkeyPatch, prior_spec_ids: list[str], origin: str
):
    """A source an earlier turn's search returned is recorded as searched, so
    its block in that turn's record stays the one a later turn pins from."""
    monkeypatch.setattr(plan_display, "step_rows", _rows_led_by_a_wide_step(plan_display.step_rows))
    runner, model = orchestrator_runner(
        call(
            "plan",
            brief="Kenya's debt service since 2015",
            asks=["Kenya's debt service since 2015"],
        ),
        types.Part.from_text(text="Charted."),
        planner=ScriptedPlannerModel(),
        tool_surface=ScriptedSurface(),
        executor=_describes,
    )
    prior = {"readings": [], "spec_ids": prior_spec_ids, "warnings": {}}
    await open_session(runner, **{ORCHESTRATOR_PRIOR_LEDGER_KEY: prior})

    approval = single_pause(await run_turn(runner, types.UserContent("Kenya's debt service?")))
    revision = approval.args["payload"]["revision"]
    approve = create_request_input_response(approval.id, {"result": approve_answer(revision)})
    events = await run_turn(runner, types.Content(role="user", parts=[approve]))

    after_plan = request_text(model.requests[-1])
    assert f"Plan {revision[:12]} (approved)" in after_plan
    assert f"({_SCRIPTED_SPEC}, sum)" in after_plan
    assert "Wide_StatVar_" not in after_plan
    session = await runner.session_service.get_session(app_name="t", user_id="u", session_id="s")
    assert session.state[ORCHESTRATOR_READINGS_KEY] == {_SCRIPTED_SPEC: "sum"}
    (stored,) = session.state[ORCHESTRATOR_PLAN_RECORDS_KEY]
    record = PlanRecord.model_validate(stored)
    assert record.revision == revision
    assert [source.origin for source in record.sources] == [origin]
    # The run the plan executed as, which the hand-back names.
    assert record.run_id is not None and f'"run_id": "{record.run_id}"' in after_plan
    assert not [e for e in events if e.error_code]


class _NoisySurface(ScriptedSurface):
    """`ScriptedSurface` whose `get_data` digest holds the plumbing lines data-mcp
    and analysis-mcp render, each from its producer's own renderer."""

    async def call_tool(self, **kwargs: Any) -> ToolCallResult:
        result = await super().call_tool(**kwargs)
        if kwargs["tool"].name != "get_data":
            return result
        text = "\n\n".join(
            [
                "Retrieved life expectancy for Kenya.",
                get_data_digest._frame_reference_line(
                    FrameRef(spec_id="spec_" + "1" * 32, content_version="cv_" + "1" * 32)
                ),
                "Source: World Bank.",
                get_data_digest._operation_poll_line(
                    get_data_digest.OperationPollNote(
                        operation_id="op_123", poll_attempts=2, elapsed_s=31.0
                    )
                ),
                analysis_digest._identity_block(
                    op_spec_hash="ops_" + "1" * 32,
                    content_version="ocv_" + "1" * 32,
                    execution_identity=current_execution_identity(),
                ),
                "Coverage 1960-2023.",
            ]
        )
        return result.model_copy(update={"text": text})


async def test_step_summaries_reach_the_agent_without_frame_references_or_operation_notes():
    runner, model = orchestrator_runner(
        call("plan", brief="Kenya's life expectancy", asks=["Kenya's life expectancy"]),
        types.Part.from_text(text="Charted."),
        planner=ScriptedPlannerModel(),
        tool_surface=_NoisySurface(),
        executor=_describes,
    )
    await open_session(runner)

    approval = single_pause(await run_turn(runner, types.UserContent("Kenya's life expectancy?")))
    approve = create_request_input_response(
        approval.id, {"result": approve_answer(approval.args["payload"]["revision"])}
    )
    await run_turn(runner, types.Content(role="user", parts=[approve]))

    after_plan = request_text(model.requests[-1])
    body = json.loads(after_plan.split("Chart its frames and answer.\n", 1)[1].split("\n", 1)[0])
    assert (
        "Retrieved life expectancy for Kenya.\n\nSource: World Bank.\n\nCoverage 1960-2023."
        in body["step_summaries"]
    )


class TwoAsksPlanner(ScriptedPlannerModel):
    """`ScriptedPlannerModel` whose plan answers the brief's first ask with the
    retrieval and its second with the aggregate that reads it, so the
    retrieval is a terminal step another step reads, declining `declined`.
    Records each refusal it is handed."""

    def __init__(self, declined: tuple[Decline, ...] = ()) -> None:
        self.declined = declined
        self.refusals: list[dict[str, Any] | None] = []

    async def __call__(self, **kwargs: Any) -> dict[str, Any]:
        self.refusals.append(kwargs.get("refusal"))
        plan = Plan.model_validate((await super().__call__(**kwargs))["plan"])
        assert plan.translation is not None and plan.workflow is not None
        series, mean = (requirement.id for requirement in plan.evidence_requirements)
        terminal = tuple(step.id for step in plan.workflow.steps)
        plan = plan.model_copy(
            update={
                "asks": load_asks(kwargs["ctx"].state),
                "answered": (
                    AskAnswer(ask=0, outputs=(series,)),
                    AskAnswer(ask=1, outputs=(mean,)),
                ),
                "declined": self.declined,
                "translation": plan.translation.model_copy(update={"terminal": terminal}),
                "workflow": plan.workflow.model_copy(update={"terminal_steps": terminal}),
            }
        )
        return {"plan": plan.model_dump(mode="json"), "summary": {"version": plan.version}}


class _AggregateRefusingSurface(ScriptedSurface):
    """`ScriptedSurface` whose `aggregate` refuses with a code no replan reaches."""

    async def call_tool(self, **kwargs: Any) -> ToolCallResult:
        if kwargs["tool"].name != "aggregate":
            return await super().call_tool(**kwargs)
        refusal = {
            "code": "growth_gap_year",
            "retry_mode": "do_not_retry",
            "message": "no entity reports both endpoint years",
            "fix": {},
        }
        return ToolCallResult(is_error=True, structured_content=refusal, text="refused")


class _EmptyAggregateSurface(ScriptedSurface):
    """`ScriptedSurface` whose `aggregate` matches no rows, a stop the graph replans."""

    async def call_tool(self, **kwargs: Any) -> ToolCallResult:
        result = await super().call_tool(**kwargs)
        if kwargs["tool"].name != "aggregate":
            return result
        return result.model_copy(
            update={"structured_content": {**(result.structured_content or {}), "rows": 0}}
        )


_TWO_ASKS = ["Kenya's life expectancy each year", "the mean across the period"]
# The first ask answered in part: its history computed, its forecast declined.
_FORECAST = Decline(
    ask="its forecast to 2030", code=UnsupportedOperation.NO_OPERATION_COVERS, ask_ref=0
)
_IN_PART = pytest.mark.parametrize("declined", [(), (_FORECAST,)], ids=["whole", "in_part"])


def _first_ask(declined: tuple[Decline, ...]) -> tuple[dict[str, Any], str]:
    """The first ask's `not_computed` entry under `declined`, and the ask as
    the charted sentence names it."""
    if not declined:
        return {}, json.dumps(_TWO_ASKS[0])
    reason = DECLINE_REASONS[_FORECAST.code]
    return (
        {"not_computed": {"part": _FORECAST.ask, "reason": reason}},
        f"{json.dumps(_TWO_ASKS[0])} except {json.dumps(_FORECAST.ask)}, "
        f"not computed because {reason}",
    )


_STORE_VERSION = "cv_" + "5" * 32


class _StoredApart:
    """`surface` whose `get_data` frame the store holds under `_STORE_VERSION`,
    a pair of data-mcp's own beside the manifest's, as for a retrieval read
    with extra columns."""

    def __init__(self, surface: ScriptedSurface) -> None:
        self._surface = surface

    def __getattr__(self, name: str) -> Any:
        return getattr(self._surface, name)

    async def call_tool(self, **kwargs: Any) -> ToolCallResult:
        result = await self._surface.call_tool(**kwargs)
        if kwargs["tool"].name != "get_data" or result.is_error:
            return result
        content = result.structured_content or {}
        store_ref = {**content["frame_ref"], "content_version": _STORE_VERSION}
        return result.model_copy(update={"structured_content": {**content, "frame_ref": store_ref}})


async def _store_versions(runner: Runner) -> dict[str, str]:
    """Each registered frame's id and the `content_version` the store holds it under."""
    session = await runner.session_service.get_session(app_name="t", user_id="u", session_id="s")
    assert session is not None
    return {
        frame_id: frame["store_ref"]["content_version"]
        for frame_id, frame in session.state[ORCHESTRATOR_FRAMES_KEY].items()
    }


async def _approved_two_asks(
    surface: Any,
    planner: TwoAsksPlanner | None = None,
    *,
    publisher: Any = scripted_publish_result,
    run_ledger: Any = None,
) -> tuple[ScriptLlm, TwoAsksPlanner]:
    """Run the two-asks brief, approving every plan the planner drafts."""
    _, model, planner = await _run_two_asks(
        surface, planner, publisher=publisher, run_ledger=run_ledger
    )
    return model, planner


async def _run_two_asks(
    surface: Any,
    planner: TwoAsksPlanner | None = None,
    *,
    publisher: Any = scripted_publish_result,
    run_ledger: Any = None,
) -> tuple[Runner, ScriptLlm, TwoAsksPlanner]:
    """`_approved_two_asks`, with the runner whose session holds the turn's state."""
    planner = planner or TwoAsksPlanner()
    runner, model = orchestrator_runner(
        call("plan", brief="Kenya's life expectancy and its mean", asks=_TWO_ASKS),
        types.Part.from_text(text="Answered."),
        planner=planner,
        tool_surface=surface,
        executor=_describes,
        publisher=publisher,
        run_ledger=run_ledger,
    )
    await open_session(runner)
    events = await run_turn(runner, types.UserContent("Kenya's life expectancy?"))
    while any(c.name == "adk_request_input" for e in events for c in e.get_function_calls()):
        approval = single_pause(events)
        approve = create_request_input_response(
            approval.id, {"result": approve_answer(approval.args["payload"]["revision"])}
        )
        events = await run_turn(runner, types.Content(role="user", parts=[approve]))
    return runner, model, planner


async def test_a_terminal_step_another_step_reads_comes_back_on_an_accepted_plan():
    model, _ = await _approved_two_asks(ScriptedSurface())

    _, body = handed_back(model)
    series, mean = (frame["frame"] for frame in body["frames"])
    assert body["asks"] == [
        {
            "ask": _TWO_ASKS[0],
            "answered_by": [{"step": "retrieve", "status": "data", "frame": series}],
        },
        {"ask": _TWO_ASKS[1], "answered_by": [{"step": "totals", "status": "data", "frame": mean}]},
    ]


async def _publication_pending(frame: FrameRef) -> PublishFault:
    return PublishFault(code="service_unavailable", reason="busy", retryable=True)


async def test_a_pending_run_hands_back_its_retrieved_terminal_and_withholds_the_derived_one():
    model, _ = await _approved_two_asks(ScriptedSurface(), publisher=_publication_pending)

    text, body = handed_back(model)
    (series,) = (frame["frame"] for frame in body["frames"])
    assert body["run"]["status"] == "pending"
    assert body["asks"] == [
        {
            "ask": _TWO_ASKS[0],
            "answered_by": [{"step": "retrieve", "status": "data", "frame": series}],
        },
        {"ask": _TWO_ASKS[1], "answered_by": [{"step": "totals", "status": "not_run"}]},
    ]
    assert "The frames listed hold retrieved data" in text


async def test_a_pending_run_admitted_again_on_a_fresh_surface_keeps_its_retrieved_provenance():
    """The process that computed the run is gone with its provenance store:
    the retrieved terminal it hands back rests on the retrievals it sealed."""
    ledger = InMemoryRunLedger(scope=current_conversation_id.get)
    first = _RecordingSurface(ScriptedSurface(), provenance=ProvenanceStore())
    await _approved_two_asks(first, publisher=_publication_pending, run_ledger=ledger)
    provenance = ProvenanceStore()
    second = _RecordingSurface(ScriptedSurface(), provenance=provenance)
    model, _ = await _approved_two_asks(second, publisher=_publication_pending, run_ledger=ledger)

    _, body = handed_back(model)
    assert body["run"]["status"] == "pending"
    run = await ledger.get(body["run"]["run_id"])
    assert run is not None
    (retrieved,) = (
        result.frame for result in run.results if not result.frame.spec_id.startswith("ops_")
    )
    lineage = provenance.get(
        conversation=current_conversation_id.get(),
        spec_id=retrieved.spec_id,
        content_version=retrieved.content_version,
    )
    assert lineage is not None
    assert {member.spec_id for member in lineage} == {retrieved.spec_id}


@_IN_PART
async def test_a_failed_plan_hands_back_the_terminal_frames_it_finished(
    declined: tuple[Decline, ...],
):
    runner, model, _ = await _run_two_asks(
        _StoredApart(_AggregateRefusingSurface()), TwoAsksPlanner(declined)
    )

    text, body = handed_back(model)
    (series,) = (frame["frame"] for frame in body["frames"])
    assert await _store_versions(runner) == {series: _STORE_VERSION}
    failed = {"status": "failed", "failed_at": "totals", "code": "growth_gap_year"}
    not_computed, charted = _first_ask(declined)
    assert body["asks"] == [
        {
            "ask": _TWO_ASKS[0],
            "answered_by": [{"step": "retrieve", "status": "data", "frame": series}],
            **not_computed,
        },
        {"ask": _TWO_ASKS[1], "answered_by": [{"step": "totals", **failed}]},
    ]
    assert f"can be charted: {charted}." in text
    assert f"Not computed: {json.dumps(_TWO_ASKS[1])}." in text


async def test_a_failed_plan_hands_back_no_step_that_is_only_read():
    """The retrieval finished, but only the refused aggregate answers an ask."""
    runner, model = orchestrator_runner(
        call("plan", brief="Kenya's life expectancy", asks=["Kenya's mean life expectancy"]),
        types.Part.from_text(text="It could not be computed."),
        planner=AskingPlanner(),
        tool_surface=_AggregateRefusingSurface(),
        executor=_describes,
    )
    await open_session(runner)
    approval = single_pause(await run_turn(runner, types.UserContent("Kenya's life expectancy?")))
    approve = create_request_input_response(
        approval.id, {"result": approve_answer(approval.args["payload"]["revision"])}
    )
    await run_turn(runner, types.Content(role="user", parts=[approve]))

    _, body = handed_back(model)
    assert body["frames"] == []
    assert body["failed_step"]["step_id"] == "totals"


async def test_a_replan_names_the_asks_the_failed_run_computed():
    _, planner = await _approved_two_asks(_EmptyAggregateSurface())

    refusal = planner.refusals[-1]
    assert refusal is not None
    assert refusal["computed_asks"] == [0]


class _RefusingReplanner(TwoAsksPlanner):
    """`TwoAsksPlanner` that refuses to draft again once a run has stopped."""

    async def __call__(self, **kwargs: Any) -> dict[str, Any]:
        if kwargs.get("refusal") is None:
            return await super().__call__(**kwargs)
        self.refusals.append(kwargs["refusal"])
        return {"refused": {"text": "no plan avoids the empty mean", "final": False}}


class _RemovingRefusingReplanner(_RefusingReplanner):
    """`_RefusingReplanner` whose plan records the brief's third ask as removed
    by the user's amendment, so a run that computes the first ask and stops
    on the second leaves a computed ask, a failed one and a removed one."""

    async def __call__(self, **kwargs: Any) -> dict[str, Any]:
        drafted = await super().__call__(**kwargs)
        if "plan" in drafted:
            drafted["plan"]["dropped_asks"] = [2]
        return drafted


class _RetrievalRefusingOnReplanSurface(_EmptyAggregateSurface):
    """`_EmptyAggregateSurface` whose second `get_data` call refuses, so the
    replanned run stops before any ask is computed."""

    def __init__(self) -> None:
        super().__init__()
        self.retrievals = 0

    async def call_tool(self, **kwargs: Any) -> ToolCallResult:
        if kwargs["tool"].name == "get_data":
            self.retrievals += 1
            if self.retrievals > 1:
                refusal = {
                    "code": "growth_gap_year",
                    "retry_mode": "do_not_retry",
                    "message": "refused on the replan",
                    "fix": {},
                }
                return ToolCallResult(is_error=True, structured_content=refusal, text="refused")
        return await super().call_tool(**kwargs)


@_IN_PART
async def test_a_planner_refusal_after_a_stopped_run_hands_back_the_asks_it_computed(
    declined: tuple[Decline, ...],
):
    model, planner = await _approved_two_asks(
        _EmptyAggregateSurface(), _RefusingReplanner(declined)
    )

    text, body = handed_back(model)
    (series,) = (frame["frame"] for frame in body["frames"])
    not_computed, charted = _first_ask(declined)
    assert planner.refusals[-1] is not None
    assert body["asks"] == [
        {
            "ask": _TWO_ASKS[0],
            "answered_by": [{"step": "retrieve", "status": "data", "frame": series}],
            **not_computed,
            "earlier_plan": True,
        }
    ]
    assert "The planner could not make a plan: no plan avoids the empty mean" in text
    assert f"can be charted: {charted}." in text


@_IN_PART
async def test_a_replanned_run_that_stops_earlier_keeps_the_asks_the_first_run_computed(
    declined: tuple[Decline, ...],
):
    runner, model, _ = await _run_two_asks(
        _StoredApart(_RetrievalRefusingOnReplanSurface()), TwoAsksPlanner(declined)
    )

    text, body = handed_back(model)
    (series,) = (frame["frame"] for frame in body["frames"])
    assert await _store_versions(runner) == {series: _STORE_VERSION}
    failed = {"status": "failed", "failed_at": "retrieve", "code": "growth_gap_year"}
    not_computed, charted = _first_ask(declined)
    assert body["failed_step"]["step_id"] == "retrieve"
    assert body["asks"] == [
        {
            "ask": _TWO_ASKS[0],
            "answered_by": [{"step": "retrieve", "status": "data", "frame": series}],
            **not_computed,
            "earlier_plan": True,
        },
        {"ask": _TWO_ASKS[1], "answered_by": [{"step": "totals", **failed}]},
    ]
    assert f"can be charted: {charted}." in text
    assert f"Not computed: {json.dumps(_TWO_ASKS[1])}." in text


@pytest.mark.parametrize(
    ("status", "charted"), [({"status": "data"}, True), ({"status": "empty"}, False)]
)
def test_a_partial_result_offers_to_chart_only_frames_that_hold_series(
    status: dict[str, str], charted: bool
):
    asks = [{"ask": "the mean", "answered_by": [{"step": "totals", **status, "frame": "f1"}]}]

    text = _partial_result(asks, frames=True)

    assert ('can be charted: "the mean".' in text) is charted
    assert (
        'holds no series or cannot be drawn, as each step\'s status says: "the mean".' in text
    ) is not charted


async def test_a_plain_text_turn_streams_the_agents_answer_over_the_studio_routes():
    app, model = orchestrator_app(types.Part.from_text(text="Kenya's GDP was $101 billion."))

    async with studio_client(app) as client:
        frames = await stream(client, "/api/studio/turn", turn_body("Kenya's GDP in 2020?"))

    assert [f["event"] for f in frames][-4:] == [
        "message.delta",
        "message.complete",
        "suggestions",
        "done",
    ]
    assert frames[-3]["text"] == "Kenya's GDP was $101 billion."
    assert len(model.requests) == 1
    assert "Kenya's GDP in 2020?" in request_text(model.requests[0])


async def test_a_search_fetch_and_chart_turn_lands_a_card_over_the_studio_routes():
    search = CannedSearch(definite_envelope())
    app, model = orchestrator_app(
        search_call("life expectancy in Kenya", "life expectancy"),
        fetch_call(),
        make_charts_call(chart(frame_for())),
        types.Part.from_text(text="Life expectancy in Kenya rose to 62 years."),
        resolver=search,
    )

    async with studio_client(app) as client:
        frames = await stream(client, "/api/studio/turn", turn_body("Life expectancy in Kenya?"))

    assert [c["question"] for c in search.calls] == ["life expectancy in Kenya"]
    tools = [f.get("tool") for f in frames if f["event"] == "activity.step"]
    assert {"search", "fetch", "make_charts", "write_component"} <= set(tools)
    created = [f for f in frames if f["event"] == "component.created"]
    assert [f["component_id"] for f in created] == ["cmp_scripted0002"]
    assert frames[-1]["event"] == "done"
    complete = next(f for f in frames if f["event"] == "message.complete")
    assert complete["text"] == "Life expectancy in Kenya rose to 62 years."
    drawn = model.requests[-1].contents[-1].parts[0].function_response
    assert drawn.name == "make_charts"
    (card,) = drawn.response["charts"]
    assert card["holds"]["series"] == ["Kenya"]
    assert card["reply_figures"] == [
        {"series": "Kenya", "value": 63.6, "unit": "Year", "period": "2023"}
    ]
