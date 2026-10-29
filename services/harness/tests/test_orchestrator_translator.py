"""`_Translator` over real ADK events, in the order a streamed run yields them."""

from __future__ import annotations

import copy
import json
import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
from analysis_mcp.result import VerbResult
from common.period_note import period_note
from compiler.surface import ProgressCallback, ToolCallResult, ToolSchema
from data_mcp.catalogue_counts import COUNTS_UNAVAILABLE
from executor.run import current_step_id
from google.adk.events import Event
from google.adk.events.event import NodeInfo
from google.adk.sessions import InMemorySessionService
from google.adk.workflow.utils._workflow_hitl_utils import REQUEST_INPUT_FUNCTION_CALL_NAME
from google.genai import types
from harness.chart_binding import TopMembers
from harness.conversation import current_conversation_id
from harness.graph.state import (
    APPROVE_INTERRUPT,
    ORCHESTRATOR_PLAN_FAILED_STATUS,
    ORCHESTRATOR_PLAN_RAN_STATUS,
    ORCHESTRATOR_STATUS_FAILURE_KEY,
    ORCHESTRATOR_STATUS_KEY,
    STEP_PROGRESS_KEY,
)
from harness.orchestrator_tools import HANDED_OFF
from harness.provenance import ProvenanceStore
from harness.reply_figures import DrawnSeries, card_holds
from harness.scripted_plan import scripted_plan
from harness.sentences import (
    TERMINAL_PROPOSE_CODES,
    SeamCode,
    failure_sentence,
    refusal_sentence,
)
from harness.studio_state import (
    ConversationState,
    StudioStore,
    _next_manifest,
    land_component,
    remove_cards,
)
from harness.studio_translate import (
    _ExecutorRecord,
    _record_detail,
    _record_label,
    _RecordingExecutor,
    _RecordingSurface,
    _tool_call_label,
    _Translator,
)
from harness.wire import ActivityStep, CompositionActor, CompositionManifest, StepStatus
from planner.plan import plan_revision
from planner.workflow import ToolId

from .failure_text import classed_sentence, is_user_sentence
from .test_studio_orchestrator_eval import checks


async def _no_executor(*, tool: str, args: dict[str, object]) -> dict[str, object]:
    raise AssertionError(f"no tool call expected, got {tool}")


def _translator() -> _Translator:
    return _Translator(
        conversation=ConversationState(conversation_id="c1"),
        recorder=_RecordingExecutor(_no_executor, store=StudioStore(InMemorySessionService())),
    )


def _event(
    *parts: types.Part, partial: bool | None = None, author: str = "dfo_orchestrator"
) -> Event:
    return Event(author=author, partial=partial, content=types.Content(parts=list(parts)))


def _frames(translator: _Translator, *events: Event) -> list:
    return [frame for event in events for frame in translator.translate(event)]


def _steps(frames: list) -> list[tuple[str, StepStatus]]:
    return [(f.label, f.status) for f in frames if f.event == "activity.step"]


def test_streamed_text_is_sent_once_and_completed_whole():
    translator = _translator()

    frames = _frames(
        translator,
        _event(types.Part(text="Life expectancy was "), partial=True),
        _event(types.Part(text="54.5 years."), partial=True),
        _event(types.Part(text="Life expectancy was 54.5 years.")),
    )
    closing = translator.finish()

    assert [f.text for f in frames if f.event == "message.delta"] == [
        "Life expectancy was ",
        "54.5 years.",
    ]
    assert [f.text for f in closing if f.event == "message.complete"] == [
        "Life expectancy was 54.5 years."
    ]
    assert closing[-1].event == "done"


def test_the_closing_message_ends_with_a_note_on_each_loose_match_the_turn_used():
    """The note is sent as one more delta, so the deltas add up to the message."""
    translator = _translator()
    conversation = translator._conversation
    conversation.turn_readings = [
        {
            "text": "climate finance",
            "status": "chosen",
            "spec_id": "spec_c",
            "warnings": [{"code": "retrieval_score_weak"}],
            "asked_roles": ["recipient"],
            "fetch": {"summary": "12 observations"},
        }
    ]
    conversation.turn_specs = {
        "spec_c": {
            "resolved": {
                "facets": {
                    "provenance": {
                        "ref": "provenance/OECDCreditorReportingSystem",
                        "account": "provider",
                    }
                },
                "sentence_plan": [
                    {"kind": "slot", "text": "Climate support worldwide.", "segment": "heading"}
                ],
            }
        }
    }

    frames = _frames(translator, _event(types.Part(text="Climate support grew.")))
    closing = translator.finish()

    deltas = [f.text for f in frames + closing if f.event == "message.delta"]
    complete = [f.text for f in closing if f.event == "message.complete"]
    assert complete == [
        'Climate support grew.\n\nThe closest series to "climate finance" is a loose '
        "match: Climate support worldwide, with no donor or recipient detail."
    ]
    assert "".join(deltas) == complete[0]
    assert conversation.turn_final_text == complete[0]


@pytest.mark.parametrize(
    ("name", "args", "answer", "step", "reading"),
    [
        (
            "search",
            {"query": "life expectancy in Kenya"},
            {"parts": []},
            "Searching for data",
            "Reading what the search found",
        ),
        ("fetch", {"spec_id": "spec_x"}, {"frames": []}, "Fetching data", "Looking at the data"),
        (
            "make_charts",
            {"charts": [{}]},
            {"charts": [{"title": "A"}]},
            "Making 1 chart",
            "Going over the charts",
        ),
        (
            "recall",
            {"part": "plan"},
            {"plans": []},
            "Looking back at an earlier turn",
            "Reading the earlier turn",
        ),
    ],
)
def test_a_tool_call_completes_on_its_response_and_a_line_names_what_the_model_reads(
    name, args, answer, step, reading
):
    """The tool's complete frame comes first, so a page showing the last
    frame shows the model reading the result."""
    translator = _translator()
    call = types.FunctionCall(id="call_1", name=name, args=args)
    response = types.FunctionResponse(id="call_1", name=name, response=answer)

    frames = _frames(
        translator,
        _event(types.Part(function_call=call)),
        _event(types.Part(function_response=response)),
    )

    assert [(f.label, f.status, f.tool) for f in frames if f.event == "activity.step"] == [
        (step, StepStatus.RUNNING, name),
        (step, StepStatus.COMPLETE, name),
        (reading, StepStatus.RUNNING, None),
    ]


def test_text_before_a_tool_call_is_interim_and_the_last_text_is_the_answer():
    translator = _translator()
    call = types.FunctionCall(id="call_1", name="fetch", args={"spec_id": "spec_x"})
    response = types.FunctionResponse(id="call_1", name="fetch", response={"frames": []})

    _frames(
        translator,
        _event(types.Part(text="Let me look that up."), partial=True),
        _event(types.Part(text="Let me look that up."), types.Part(function_call=call)),
        _event(types.Part(function_response=response)),
        _event(types.Part(text="Kenya's GDP was $101 billion.")),
    )
    closing = translator.finish()

    assert [f.text for f in closing if f.event == "message.complete"] == [
        "Kenya's GDP was $101 billion."
    ]


def test_text_after_a_hand_off_is_not_shown_until_the_next_run_starts():
    translator = _translator()
    call = types.FunctionCall(id="call_1", name="ask_user", args={"question": "Which Congo?"})
    taken = types.FunctionResponse(
        id="call_1", name="ask_user", response={"status": "Recorded.", HANDED_OFF: "ask"}
    )

    frames = _frames(
        translator,
        _event(types.Part(function_call=call)),
        _event(types.Part(function_response=taken)),
        _event(types.Part(text="I have asked the user."), partial=True),
        _event(types.Part(text="I have asked the user.")),
        Event(author="t", custom_metadata={ORCHESTRATOR_STATUS_KEY: "Reading your answer"}),
        _event(types.Part(text="The Republic of the Congo's GDP was $15.7 billion."), partial=True),
    )

    assert [f.text for f in frames if f.event == "message.delta"] == [
        "The Republic of the Congo's GDP was $15.7 billion."
    ]
    assert ("Reading your answer", StepStatus.RUNNING) in _steps(frames)
    assert ("Reading your answer", StepStatus.COMPLETE) in _steps(frames)


@pytest.mark.parametrize(
    ("answer", "code", "retry_mode"),
    [
        # An answer that states no code still fails its step with a sentence.
        ({"error": "search is down"}, None, None),
        (
            {"refusal": "request_invalid: /input/scope/window", "code": "request_invalid"},
            "request_invalid",
            None,
        ),
        ({"refusal": None, "code": "over_ceiling"}, "over_ceiling", None),
        (
            {
                "error": "fetch has been called 6 time(s) this turn, its limit.",
                "code": "tool_budget_spent",
                "retry_mode": "do_not_retry",
            },
            "tool_budget_spent",
            "do_not_retry",
        ),
        # A refusal body states its own code.
        (
            {
                "outcome": "refusal",
                "refusal": {"code": "empty_result", "message": "no observations for country/KEN"},
            },
            "empty_result",
            None,
        ),
    ],
    ids=["no-code", "refusal-prose", "no-reason", "budget", "refusal-body"],
)
def test_a_tool_that_returned_an_error_or_a_refusal_is_a_failed_step_with_its_codes_sentence(
    answer, code, retry_mode
):
    """The reason is written for the model, so the failed line states the
    sentence of the answer's code and none of its prose."""
    translator = _translator()
    call = types.FunctionCall(id="call_1", name="fetch", args={"spec_id": "spec_x"})
    failed = types.FunctionResponse(id="call_1", name="fetch", response=answer)

    frames = _frames(
        translator,
        _event(types.Part(function_call=call)),
        _event(types.Part(function_response=failed)),
    )

    running, failed = [f for f in frames if f.event == "activity.step"][:2]
    assert (running.status, running.detail) == (StepStatus.RUNNING, None)
    assert (failed.status, failed.detail) == (StepStatus.FAILED, failure_sentence(code, retry_mode))
    assert is_user_sentence(failed.detail)


def test_step_ids_stay_unique_across_the_segments_of_one_turn():
    """Studio resumes a paused turn into one Activity entry keyed by step_id,
    and each segment gets a fresh translator over the same conversation."""
    conversation = ConversationState(conversation_id="c1")
    recorder = _RecordingExecutor(_no_executor, store=StudioStore(InMemorySessionService()))
    first = _Translator(conversation=conversation, recorder=recorder)
    second = _Translator(conversation=conversation, recorder=recorder)
    ask = types.FunctionCall(id="call_1", name="ask_user", args={"question": "Which?"})
    explain = types.FunctionCall(id="call_2", name="explain_dfo", args={})
    status = Event(author="t", custom_metadata={ORCHESTRATOR_STATUS_KEY: "Reading your answer"})

    before = _frames(first, _event(types.Part(function_call=ask)), status)
    after = _frames(second, _event(types.Part(function_call=explain)), status)

    ids = [f.step_id for f in [*before, *after] if f.event == "activity.step"]
    assert len(ids) == len(set(ids)) == 4


@pytest.mark.parametrize(
    "answer",
    [
        # Its budget spent.
        {"error": "plan has been called 1 time(s) this turn, its limit."},
        # Its asks past the planner's limits: a refusal, with no "error".
        {"refusal": "plan recorded nothing: asks lists 7 outputs, more than 6.", "code": "x"},
    ],
)
def test_a_refused_hand_off_leaves_the_models_text_showing(answer):
    translator = _translator()
    call = types.FunctionCall(id="call_1", name="plan", args={"brief": "b"})
    refused = types.FunctionResponse(id="call_1", name="plan", response=answer)

    frames = _frames(
        translator,
        _event(types.Part(function_call=call)),
        _event(types.Part(function_response=refused)),
        _event(types.Part(text="I cannot plan this; here is what I found."), partial=True),
        _event(types.Part(text="I cannot plan this; here is what I found.")),
    )
    closing = translator.finish()

    assert [f.text for f in frames if f.event == "message.delta"] == [
        "I cannot plan this; here is what I found."
    ]
    assert [f.text for f in closing if f.event == "message.complete"] == [
        "I cannot plan this; here is what I found."
    ]
    assert ("Working on a plan", StepStatus.FAILED) in _steps(frames)


def test_the_models_thinking_sends_no_thought_text():
    translator = _translator()
    call = types.FunctionCall(id="call_1", name="explain_dfo", args={})

    frames = _frames(
        translator,
        _event(
            types.Part(text="**Reviewing the System Instructions**\n\nspec_x", thought=True),
            partial=True,
        ),
        _event(types.Part(function_call=call)),
        _event(
            types.Part(
                function_response=types.FunctionResponse(
                    id="call_1", name="explain_dfo", response={}
                )
            )
        ),
        _event(types.Part(text="**Next** frame f07", thought=True), partial=True),
        _event(types.Part(text="Here it is."), partial=True),
    )

    assert _steps(frames) == [
        ("Reading about DFO", StepStatus.RUNNING),
        ("Reading about DFO", StepStatus.COMPLETE),
        ("Going over the notes on DFO", StepStatus.RUNNING),
        ("Going over the notes on DFO", StepStatus.COMPLETE),
    ]


@pytest.mark.parametrize(
    ("counts", "detail"),
    [
        ("DFO holds 92 datasets.", "DFO holds 92 datasets."),
        # The model's instruction to quote no count is not shown to the user.
        (COUNTS_UNAVAILABLE, None),
    ],
)
def test_explain_dfos_step_shows_the_counts_it_read(counts, detail):
    call = types.FunctionCall(id="call_1", name="explain_dfo", args={})
    read = types.FunctionResponse(
        id="call_1", name="explain_dfo", response={"explainer": "DFO is", "counts": counts}
    )

    frames = _frames(
        _translator(),
        _event(types.Part(function_call=call)),
        _event(types.Part(function_response=read)),
    )

    assert [(f.status, f.detail) for f in frames if f.tool == "explain_dfo"] == [
        (StepStatus.RUNNING, None),
        (StepStatus.COMPLETE, detail),
    ]


# A refusal detail with every slot a `SEAM_SENTENCES` template reads filled.
_FILLED_DETAIL = {
    "tool": "propose_plan",
    "reference": "4f2a9c0d1e7b4a55",
    "requirement_id": "aid_by_donor",
    "kind": "no_data",
    "rounds": 7,
    "declined": [{"ask": "forecast U.S. aid to 2030"}],
}
_FILLED_VALUES = ("propose", "4f2a9c0d", "aid_by_donor", "forecast", "U.S.", "7")


def _plan_failed_step(failure: object) -> ActivityStep:
    """The failed step a plan hand-back with no plan shows for `failure`."""
    failed_status = {ORCHESTRATOR_STATUS_KEY: ORCHESTRATOR_PLAN_FAILED_STATUS}
    if failure is not None:
        failed_status[ORCHESTRATOR_STATUS_FAILURE_KEY] = failure
    frames = _frames(
        _translator(),
        Event(author="t", custom_metadata={ORCHESTRATOR_STATUS_KEY: "Reading your question"}),
        Event(author="t", custom_metadata=failed_status),
    )
    assert _steps(frames) == [
        ("Reading your question", StepStatus.RUNNING),
        ("Reading your question", StepStatus.COMPLETE),
        ("No plan could be made", StepStatus.FAILED),
        (ORCHESTRATOR_PLAN_FAILED_STATUS, StepStatus.RUNNING),
    ]
    return next(f for f in frames if f.event == "activity.step" and f.status is StepStatus.FAILED)


@pytest.mark.parametrize("code", [*SeamCode, *TERMINAL_PROPOSE_CODES])
def test_a_plan_the_planner_refused_shows_its_code_s_sentence_and_none_of_its_detail(code):
    """The refusal's own sentence is filled for the agent with a tool name, a
    reference, a requirement id or the declined ask. The step reads the
    failure table by the code alone."""
    refused = {"code": code, "detail": _FILLED_DETAIL}
    reason = refusal_sentence(refused)

    failed = _plan_failed_step({"code": code, "reason": reason})

    assert failed.detail == classed_sentence(code)
    assert not [value for value in _FILLED_VALUES if value in failed.detail]
    assert is_user_sentence(failed.detail)


@pytest.mark.parametrize(
    ("failure", "code"),
    [
        # A reason written for the agent stays out, and the code's sentence shows.
        (
            {"code": "internal_error", "reason": "the planner seam ran past 400 events"},
            "internal_error",
        ),
        ({"code": None, "reason": "the pair did not converge"}, None),
        (None, None),
    ],
    ids=["failure-code", "no-code", "no-failure"],
)
def test_a_plan_the_planner_could_not_make_shows_as_a_failed_step_with_its_reason(failure, code):
    failed = _plan_failed_step(failure)

    assert failed.detail == failure_sentence(code)
    assert is_user_sentence(failed.detail)


def test_a_plan_that_ran_is_read_from_the_execute_nodes_result_through_the_hand_back():
    """The line opens at the execute node's result, so it covers `plan_done`,
    and the hand-back's status keeps it running until the agent's next move."""
    translator = _translator()
    executed = Event(
        author="dfo_analyst_graph",
        node_info=NodeInfo(path="dfo_analyst_graph/execute@1"),
        output={"frame_refs": []},
        route="ok",
    )
    hand_back = Event(
        author="t", custom_metadata={ORCHESTRATOR_STATUS_KEY: ORCHESTRATOR_PLAN_RAN_STATUS}
    )
    call = types.FunctionCall(id="call_1", name="make_charts", args={"charts": [{}]})

    frames = _frames(translator, executed, hand_back, _event(types.Part(function_call=call)))

    assert _steps(frames) == [
        ("Reading the plan's results", StepStatus.RUNNING),
        ("Reading the plan's results", StepStatus.COMPLETE),
        ("Making 1 chart", StepStatus.RUNNING),
    ]


def test_thinking_leaves_the_reading_line_running_until_the_answer_text():
    """Studio hides thought text, so a thought chunk, streamed or in a
    finished call, leaves the line the model reads under running. The
    answer's first text completes it."""
    translator = _translator()
    call = types.FunctionCall(id="call_1", name="make_charts", args={"charts": [{}]})
    response = types.FunctionResponse(
        id="call_1", name="make_charts", response={"charts": [{"title": "A"}]}
    )
    thought = types.Part(text="**Weighing the charts**", thought=True)

    thinking = _frames(
        translator,
        _event(types.Part(function_call=call)),
        _event(types.Part(function_response=response)),
        _event(thought, partial=True),
        _event(thought),
    )
    answering = _frames(translator, _event(types.Part(text="Here it is."), partial=True))

    assert _steps(thinking)[-1] == ("Going over the charts", StepStatus.RUNNING)
    assert _steps(answering) == [("Going over the charts", StepStatus.COMPLETE)]


def test_a_plan_sent_back_for_a_replan_opens_no_reading_line():
    executed = Event(
        author="dfo_analyst_graph",
        node_info=NodeInfo(path="dfo_analyst_graph/execute@1"),
        output={"refusal": {"code": "x"}},
        route="replan",
    )

    assert _steps(_frames(_translator(), executed)) == []


def test_a_successful_hand_off_to_the_planner_shows_only_the_planner_status_line():
    """The hand-off call shows no step; "Working on a plan" opens and the
    planner's own status lines follow it."""
    translator = _translator()
    call = types.FunctionCall(id="call_1", name="plan", args={"brief": "b", "asks": ["a"]})
    taken = types.FunctionResponse(
        id="call_1", name="plan", response={"status": "Recorded.", HANDED_OFF: "plan"}
    )

    frames = _frames(
        translator,
        _event(types.Part(function_call=call)),
        _event(types.Part(function_response=taken)),
        _event(types.Part(text="**Waiting**", thought=True), partial=True),
    )

    assert _steps(frames) == [("Working on a plan", StepStatus.RUNNING)]


def test_a_failed_skill_read_shows_and_a_successful_one_does_not():
    translator = _translator()
    read = types.FunctionCall(id="call_1", name="get_skill", args={"skill_id": "oda-measures"})
    missing = types.FunctionCall(id="call_2", name="get_skill", args={"skill_id": "nope"})

    frames = _frames(
        translator,
        _event(types.Part(function_call=read), types.Part(function_call=missing)),
        _event(
            types.Part(
                function_response=types.FunctionResponse(
                    id="call_1", name="get_skill", response={"text": "notes"}
                )
            ),
            types.Part(
                function_response=types.FunctionResponse(
                    id="call_2", name="get_skill", response={"error": "no skill nope"}
                )
            ),
        ),
    )

    (failed,) = [f for f in frames if f.event == "activity.step"]
    assert (failed.label, failed.status) == ("Reading background notes", StepStatus.FAILED)
    assert is_user_sentence(failed.detail)
    assert "nope" not in failed.detail


_FAULT = {"error": "boom", "code": "internal_error"}
# data-mcp's refusal of a call whose arguments the model must correct.
_SLIP = {
    "error": "open_roles word 'purpose' names no recognised role",
    "code": "request_invalid",
    "retry_mode": "correct_and_retry",
}
_SEARCH = {"query": "ODA to Kenya"}


def _call(call_id: str, name: str, args: dict[str, Any]) -> Event:
    return _event(types.Part(function_call=types.FunctionCall(id=call_id, name=name, args=args)))


def _response(call_id: str, name: str, answer: dict[str, Any]) -> Event:
    return _event(
        types.Part(function_response=types.FunctionResponse(id=call_id, name=name, response=answer))
    )


def _plan_card() -> Event:
    """The approval interrupt a planner that made a plan pauses on."""
    plan = scripted_plan("kenya life expectancy", spec_id=None)
    payload = {**plan.model_dump(mode="json"), "revision": plan_revision(plan)}
    call = types.FunctionCall(
        id=APPROVE_INTERRUPT, name=REQUEST_INPUT_FUNCTION_CALL_NAME, args={"payload": payload}
    )
    return Event(
        author="dfo_analyst_graph", content=types.Content(parts=[types.Part(function_call=call)])
    )


def _no_plan() -> Event:
    return Event(
        author="t",
        custom_metadata={
            ORCHESTRATOR_STATUS_KEY: ORCHESTRATOR_PLAN_FAILED_STATUS,
            ORCHESTRATOR_STATUS_FAILURE_KEY: {"code": "internal_error"},
        },
    )


def _settled_failures(frames: list) -> list[ActivityStep]:
    """The latest frame of each step that failed, as Studio's Activity record
    keeps it: the latest frame on a step_id wins."""
    steps = [frame for frame in frames if frame.event == "activity.step"]
    failed = {frame.step_id for frame in steps if frame.status is StepStatus.FAILED}
    latest = {frame.step_id: frame for frame in steps if frame.step_id in failed}
    assert latest, "no step failed"
    return list(latest.values())


_PLANNED = {"status": "Recorded.", HANDED_OFF: "plan"}
# A hand-off whose ask lists donors one by one and totals across them, and
# the orchestrator's refusal of it.
_ASKED_TWICE = {"brief": "b", "asks": [{"by": ["donor"], "over": ["donor"], "across": "2022"}]}
_ASK_SLIP = {
    "error": "plan: asks.0: `by` lists donor and `over` totals across donor.",
    "code": "tool_arguments_invalid",
    "retry_mode": "correct_and_retry",
}


@pytest.mark.parametrize(
    "events",
    [
        lambda: [
            _call("c1", "plan", _ASKED_TWICE),
            _response("c1", "plan", _ASK_SLIP),
            _call("c2", "plan", {"brief": "b", "asks": [{"by": ["donor"], "across": "2022"}]}),
            _response("c2", "plan", _PLANNED),
        ],
        # A skill read under a wrong name.
        lambda: [
            _call("c1", "get_skill", {"name": "wrong"}),
            _response("c1", "get_skill", {**_SLIP, "code": "skill_not_found"}),
            _call("c2", "get_skill", {"name": "right"}),
            _response("c2", "get_skill", {"skill": "text"}),
        ],
    ],
    ids=["hand-off", "skill-read"],
)
def test_a_quiet_call_the_model_corrects_shows_no_failed_step(events):
    """A hand-off or a skill read shows only when it fails, and a refusal
    the model corrects is no failure: a line stating one, even settled later,
    reads as a fault in the system."""
    translator = _translator()

    frames = [*_frames(translator, *events()), *translator.finish()]

    steps = [frame for frame in frames if frame.event == "activity.step"]
    assert [frame for frame in steps if frame.status is StepStatus.FAILED] == []


@pytest.mark.parametrize(
    "events",
    [
        # f7b: the same search, tried again.
        lambda: [
            _call("c1", "search", _SEARCH),
            _response("c1", "search", _FAULT),
            _call("c2", "search", _SEARCH),
            _response("c2", "search", {"parts": []}),
        ],
        # A refusal that asked for a corrected call, then the corrected call.
        lambda: [
            _call("c1", "search", {**_SEARCH, "open_roles": ["purpose"]}),
            _response("c1", "search", _SLIP),
            _call("c2", "search", _SEARCH),
            _response("c2", "search", {"parts": []}),
        ],
        # f6a: a hand-off that hit a fault, then one the planner took.
        lambda: [
            _call("c1", "plan", {"brief": "b", "asks": ["a"]}),
            _response("c1", "plan", _FAULT),
            _call("c2", "plan", {"brief": "b again", "asks": ["a"]}),
            _response("c2", "plan", _PLANNED),
        ],
        # A planner that made no plan, then a plan made on a second hand-off.
        lambda: [
            _no_plan(),
            _call("c2", "plan", {"brief": "b"}),
            _response("c2", "plan", _PLANNED),
            _plan_card(),
        ],
    ],
    ids=[
        "search-retried",
        "search-corrected",
        "hand-off-retried",
        "replanned",
    ],
)
def test_a_failed_step_a_later_try_made_good_settles_as_complete(events):
    """A failed line standing beside an answer reads as a failure the user
    must weigh, so a later try of the same step settles it on its own id."""
    translator = _translator()

    frames = _frames(translator, *events())

    (settled,) = _settled_failures([*frames, *translator.finish()])
    assert settled.status is StepStatus.COMPLETE
    assert is_user_sentence(settled.detail)


@pytest.mark.parametrize(
    "events",
    [
        # A different search is another step, not a retry.
        lambda: [
            _call("c1", "search", _SEARCH),
            _response("c1", "search", _FAULT),
            _call("c2", "search", {"query": "FDI to Kenya"}),
            _response("c2", "search", {"parts": []}),
        ],
        # f5b: no plan, then an answer from separate fetches.
        lambda: [
            _no_plan(),
            _call("c2", "fetch", {"spec_id": "spec_x"}),
            _response("c2", "fetch", {"summary": "12 observations"}),
            _event(types.Part(text="Here are both figures.")),
        ],
        # A hand-off the planner took, which then made no plan.
        lambda: [
            _call("c1", "plan", {"brief": "b"}),
            _response("c1", "plan", _PLANNED),
            _no_plan(),
        ],
        # A failure no later step tried again: the turn ends on it.
        lambda: [_call("c1", "search", _SEARCH), _response("c1", "search", _FAULT)],
        # A hand-off refused as one to correct, which the model never
        # corrected: held to the segment's end, it shows then.
        lambda: [
            _call("c1", "plan", _ASKED_TWICE),
            _response("c1", "plan", _ASK_SLIP),
            _event(types.Part(text="Here is what I found.")),
        ],
    ],
    ids=[
        "other-search",
        "answered-without-a-plan",
        "no-plan-after-hand-off",
        "terminal",
        "hand-off-never-corrected",
    ],
)
def test_a_failure_no_later_try_made_good_stays_failed(events):
    translator = _translator()

    frames = _frames(translator, *events())

    settled = _settled_failures([*frames, *translator.finish()])
    assert [frame.status for frame in settled] == [StepStatus.FAILED] * len(settled)


_SLIPPED = {**_SEARCH, "open_roles": ["purpose"]}
_FOUND = {"parts": []}
_CHARTS = {"charts": [{"title": "Aid to Kenya"}]}
_OTHER_CHARTS = {"charts": [{"title": "Aid to Kenya by donor"}]}
_BAR = {"frame": "f1", "type": "bar_chart"}
_NAMED_CHARTS = {"charts": [{**_BAR, "title": "Aid to Kenya in 2024"}]}
_CORRECTED_CHARTS = {"charts": [{**_BAR, "title": "Aid to Kenya"}]}
# `make_charts` refusing a chart whose title names a year of its frame.
_TITLE_REFUSED = {
    "charts": [
        {
            "title": "Aid to Kenya in 2024",
            "error": "The title names 2024.",
            "code": "title_names_period",
            "retry_mode": "correct_and_retry",
        }
    ]
}


def _calls(*calls: tuple[str, str, dict[str, Any]]) -> Event:
    """One model call that makes several tool calls at once."""
    return _event(
        *(
            types.Part(function_call=types.FunctionCall(id=call_id, name=name, args=args))
            for call_id, name, args in calls
        )
    )


def _responses(*responses: tuple[str, str, dict[str, Any]]) -> Event:
    return _event(
        *(
            types.Part(
                function_response=types.FunctionResponse(id=call_id, name=name, response=answer)
            )
            for call_id, name, answer in responses
        )
    )


@pytest.mark.parametrize(
    ("events", "settled"),
    [
        # Two refusals asking for a correction, then a call that completes.
        (
            lambda: [
                _call("c1", "search", _SLIPPED),
                _response("c1", "search", _SLIP),
                _call("c2", "search", {**_SEARCH, "open_roles": ["sector"]}),
                _response("c2", "search", _SLIP),
                _call("c3", "search", _SEARCH),
                _response("c3", "search", _FOUND),
            ],
            [StepStatus.COMPLETE, StepStatus.COMPLETE],
        ),
        # A correction that hits a fault leaves the slip waiting; the fault
        # itself is keyed by its own arguments.
        (
            lambda: [
                _call("c1", "search", _SLIPPED),
                _response("c1", "search", _SLIP),
                _call("c2", "search", {**_SEARCH, "open_roles": ["sector"]}),
                _response("c2", "search", _FAULT),
                _call("c3", "search", _SEARCH),
                _response("c3", "search", _FOUND),
            ],
            [StepStatus.COMPLETE, StepStatus.FAILED],
        ),
        (
            lambda: [
                _call("c1", "search", _SLIPPED),
                _response("c1", "search", _SLIP),
                _call("c2", "search", _SEARCH),
                _response("c2", "search", _FAULT),
            ],
            [StepStatus.FAILED, StepStatus.FAILED],
        ),
        # Parallel corrections, one failing and one completing, either order.
        (
            lambda: [
                _call("c1", "search", _SLIPPED),
                _response("c1", "search", _SLIP),
                _calls(("c2", "search", _SEARCH), ("c3", "search", {"query": "FDI to Kenya"})),
                _responses(("c2", "search", _FAULT), ("c3", "search", _FOUND)),
            ],
            [StepStatus.COMPLETE, StepStatus.FAILED],
        ),
        (
            lambda: [
                _call("c1", "search", _SLIPPED),
                _response("c1", "search", _SLIP),
                _calls(("c2", "search", _SEARCH), ("c3", "search", {"query": "FDI to Kenya"})),
                _responses(("c2", "search", _FOUND), ("c3", "search", _FAULT)),
            ],
            [StepStatus.COMPLETE, StepStatus.FAILED],
        ),
        # Another tool's call corrects nothing.
        (
            lambda: [
                _call("c1", "search", _SLIPPED),
                _response("c1", "search", _SLIP),
                _call("c2", "recall", {}),
                _response("c2", "recall", {"turns": []}),
            ],
            [StepStatus.FAILED],
        ),
        # A call made beside the refused one, in the same batch, is no
        # correction of it.
        (
            lambda: [
                _calls(("c1", "search", _SLIPPED), ("c2", "search", {"query": "FDI to Kenya"})),
                _responses(("c1", "search", _SLIP), ("c2", "search", _FOUND)),
            ],
            [StepStatus.FAILED],
        ),
        # A correction failed by its outcome, with no error in its answer,
        # leaves the slip waiting too.
        (
            lambda: [
                _call("c1", "make_charts", {"charts": "Aid to Kenya"}),
                _response("c1", "make_charts", {**_SLIP, "code": "tool_arguments_invalid"}),
                _call("c2", "make_charts", _OTHER_CHARTS),
                _response(
                    "c2",
                    "make_charts",
                    {"charts": [{"title": "Aid to Kenya by donor", **_FAULT}]},
                ),
                _call("c3", "make_charts", _CHARTS),
                _response("c3", "make_charts", _CHARTS),
            ],
            [StepStatus.COMPLETE, StepStatus.FAILED],
        ),
        # A chart refused for a title the model corrects, then the corrected call.
        (
            lambda: [
                _call("c1", "make_charts", _NAMED_CHARTS),
                _response("c1", "make_charts", _TITLE_REFUSED),
                _call("c2", "make_charts", _CORRECTED_CHARTS),
                _response("c2", "make_charts", _CORRECTED_CHARTS),
            ],
            [StepStatus.COMPLETE],
        ),
        # The corrected chart drawn beside others corrects it too.
        (
            lambda: [
                _call("c1", "make_charts", _NAMED_CHARTS),
                _response("c1", "make_charts", _TITLE_REFUSED),
                _call(
                    "c2",
                    "make_charts",
                    {"charts": [{"frame": "f2", "type": "table"}, *_CORRECTED_CHARTS["charts"]]},
                ),
                _response("c2", "make_charts", {"charts": [{"title": "t"}, {"title": "Aid"}]}),
            ],
            [StepStatus.COMPLETE],
        ),
        (
            lambda: [
                _call("c1", "make_charts", _NAMED_CHARTS),
                _response("c1", "make_charts", _TITLE_REFUSED),
            ],
            [StepStatus.FAILED],
        ),
        # A chart of another frame or another type drawn later leaves the
        # refused one undrawn.
        (
            lambda: [
                _call("c1", "make_charts", _NAMED_CHARTS),
                _response("c1", "make_charts", _TITLE_REFUSED),
                _call("c2", "make_charts", {"charts": [{**_BAR, "type": "table"}]}),
                _response("c2", "make_charts", _CHARTS),
            ],
            [StepStatus.FAILED],
        ),
        (
            lambda: [
                _call("c1", "make_charts", _NAMED_CHARTS),
                _response("c1", "make_charts", _TITLE_REFUSED),
                _call("c2", "make_charts", {"charts": [{**_BAR, "frame": "f2"}]}),
                _response("c2", "make_charts", _CHARTS),
            ],
            [StepStatus.FAILED],
        ),
        # A chart the platform failed beside it is no slip to correct.
        (
            lambda: [
                _call(
                    "c1",
                    "make_charts",
                    {"charts": [*_NAMED_CHARTS["charts"], {"frame": "f2", "type": "table"}]},
                ),
                _response(
                    "c1",
                    "make_charts",
                    {
                        "charts": [
                            *_TITLE_REFUSED["charts"],
                            {"title": "Aid to Kenya by donor", **_FAULT},
                        ]
                    },
                ),
                _call("c2", "make_charts", _CORRECTED_CHARTS),
                _response("c2", "make_charts", _CORRECTED_CHARTS),
            ],
            [StepStatus.FAILED],
        ),
    ],
    ids=[
        "slip-slip-success",
        "slip-fault-success",
        "slip-fault-end",
        "parallel-fault-first",
        "parallel-success-first",
        "other-tool",
        "beside-the-slip",
        "keyless-failure-then-success",
        "chart-refused-then-corrected",
        "chart-corrected-beside-others",
        "chart-refused-uncorrected",
        "chart-refused-then-another-type",
        "chart-refused-then-another-frame",
        "chart-refused-beside-a-fault",
    ],
)
def test_a_later_call_of_its_tool_that_completes_settles_a_slip(events, settled):
    """A refusal that asks the model to correct its call is settled by any
    later call of the same tool that completes, whatever its arguments."""
    translator = _translator()

    frames = _frames(translator, *events())

    steps = _settled_failures([*frames, *translator.finish()])
    assert [frame.status for frame in steps] == settled


def test_a_fetch_names_the_measure_its_search_found_and_carries_its_summary():
    translator = _translator()
    search = types.FunctionCall(
        id="call_1", name="search", args={"query": "life expectancy and infant mortality in Rwanda"}
    )
    found = {
        "parts": [
            {
                "variable": "life expectancy",
                "status": "definite",
                "spec_id": "spec_le",
                "resolved": "Life expectancy at birth in Rwanda",
            },
            {"variable": "infant mortality", "status": "definite", "spec_id": "spec_im"},
            {"variable": "maternal deaths", "status": "no_data", "spec_id": None},
            {"variable": "aid per person", "status": "unsupported", "spec_id": None},
        ]
    }
    fetches = [
        types.FunctionCall(id="call_2", name="fetch", args={"spec_id": "spec_le"}),
        types.FunctionCall(id="call_3", name="fetch", args={"spec_id": "spec_im"}),
    ]
    summary = "24 observations, 2000-2023 · World Development Indicators"
    note = "The data is for 2023, which covers 8 donors, while 2024 covers 2 donors so far."
    answers = {"call_2": {"summary": summary}, "call_3": {"summary": summary, "period_note": note}}
    # The search tool's state write, which `studio._keep_turn_records` keeps
    # on the conversation before the translator sees the event.
    translator._conversation.turn_readings = [
        {"text": "life expectancy", "status": "definite", "spec_id": "spec_le"},
        {"text": "infant mortality", "status": "definite", "spec_id": "spec_im"},
    ]

    frames = _frames(
        translator,
        _event(types.Part(function_call=search)),
        _event(
            types.Part(
                function_response=types.FunctionResponse(id="call_1", name="search", response=found)
            )
        ),
        _event(*(types.Part(function_call=call) for call in fetches)),
        _event(
            *(
                types.Part(
                    function_response=types.FunctionResponse(
                        id=call.id, name="fetch", response=answers[call.id or ""]
                    )
                )
                for call in fetches
            )
        ),
    )

    lines = [(f.label, f.detail) for f in frames if f.event == "activity.step" and f.tool is None]
    # Several fetches read as one line naming each measure in its detail.
    assert lines[-1] == (
        "Looking at the data",
        f"life expectancy: {summary}; infant mortality: {summary}",
    )
    steps = [(f.label, f.status, f.detail) for f in frames if f.event == "activity.step" and f.tool]
    assert steps == [
        (
            "Searching for data",
            StepStatus.RUNNING,
            "life expectancy and infant mortality in Rwanda",
        ),
        (
            "Searching for data",
            StepStatus.COMPLETE,
            "life expectancy and infant mortality in Rwanda · "
            "Found Life expectancy at birth in Rwanda, infant mortality; "
            "no data for maternal deaths; "
            "no supported dataset for aid per person",
        ),
        ("Fetching life expectancy", StepStatus.RUNNING, None),
        ("Fetching infant mortality", StepStatus.RUNNING, None),
        ("Fetching life expectancy", StepStatus.COMPLETE, summary),
        # A fetch's line states the period note its result carries.
        ("Fetching infant mortality", StepStatus.COMPLETE, f"{summary} \u00b7 {note}"),
    ]


def test_a_fetch_names_its_measure_from_the_conversations_readings_after_a_resume():
    """A fetch in a resumed segment, or in a later turn, finds the measure in
    the readings the conversation kept, a chosen candidate included."""
    conversation = ConversationState(conversation_id="c1")
    conversation.turns = [
        {
            "user": "q1",
            "assistant": "a1",
            "readings": [
                {"text": "GDP", "status": "definite", "spec_id": "spec_gdp"},
            ],
        },
    ]
    conversation.turn_readings = [
        {
            "text": "debt",
            "status": "candidates",
            "spec_id": None,
            "candidates": [{"spec_id": "spec_debt_gni", "label": "External debt (% of GNI)"}],
        },
    ]
    resumed = _Translator(
        conversation=conversation,
        recorder=_RecordingExecutor(_no_executor, store=StudioStore(InMemorySessionService())),
    )
    calls = [
        types.FunctionCall(id="call_1", name="fetch", args={"spec_id": "spec_debt_gni"}),
        types.FunctionCall(id="call_2", name="fetch", args={"spec_id": "spec_gdp"}),
        types.FunctionCall(id="call_3", name="fetch", args={"spec_id": "spec_unknown"}),
    ]

    frames = _frames(resumed, _event(*(types.Part(function_call=call) for call in calls)))

    assert [f.label for f in frames if f.event == "activity.step"] == [
        "Fetching debt",
        "Fetching GDP",
        "Fetching data",
    ]


def _refused_chart(title: str, code: str) -> dict[str, Any]:
    return {"title": title, "error": "no such column", "code": code, "retry_mode": "do_not_retry"}


@pytest.mark.parametrize(
    ("asked", "answer", "reasons"),
    [
        (
            ["A", "B"],
            {
                "charts": [
                    {"title": "A", "holds": {}, "reply_figures": []},
                    _refused_chart("B", "binding_invalid"),
                ]
            },
            [("binding_invalid", ["B"])],
        ),
        # The charts past the limit of one call are named too.
        (
            ["A", "B"],
            {
                "charts": [{"title": "A", "holds": {}, "reply_figures": []}],
                "not_drawn": "1 charts past the 1 one call may draw",
            },
            [("chart_not_drawn", ["B"])],
        ),
        # Charts refused for one reason share it, in the order they were asked.
        (
            ["A", "B", "C"],
            {
                "charts": [
                    _refused_chart("A", "series_limit_exceeded"),
                    _refused_chart("B", "transient_unavailable"),
                    _refused_chart("C", "series_limit_exceeded"),
                ]
            },
            [("series_limit_exceeded", ["A", "C"]), ("transient_unavailable", ["B"])],
        ),
    ],
    ids=["refused", "past-the-limit", "two-reasons"],
)
def test_a_chart_call_that_drew_only_some_of_its_charts_fails(asked, answer, reasons):
    """The failed line gives each reason as the sentence of its code, then the
    cards it holds for, and nothing of the refusal's own message."""
    translator = _translator()
    charts = [{"title": title} for title in asked]
    call = types.FunctionCall(id="call_1", name="make_charts", args={"charts": charts})

    frames = _frames(
        translator,
        _event(types.Part(function_call=call)),
        _event(
            types.Part(
                function_response=types.FunctionResponse(
                    id="call_1", name="make_charts", response=answer
                )
            )
        ),
    )

    running, failed = [f for f in frames if f.event == "activity.step"][:2]
    label = f"Making {len(asked)} charts"
    assert (running.label, running.status, running.detail) == (label, StepStatus.RUNNING, None)
    assert (failed.label, failed.status) == (label, StepStatus.FAILED)
    said = [
        f"{classed_sentence(code, 'do_not_retry').removesuffix('.')}: {'; '.join(titles)}"
        for code, titles in reasons
    ]
    # The first reason and its cards are whole, and the rest follow as far as
    # the Activity log shows.
    assert failed.detail.startswith(said[0])
    assert "; ".join(said).startswith(failed.detail.removesuffix("\u2026"))
    assert is_user_sentence(failed.detail)
    assert "no such column" not in failed.detail


def test_a_failed_chart_lines_reason_survives_titles_too_long_for_the_activity_log():
    """The Activity log shows the first 139 characters of a detail. The
    reason leads, so the cut falls in a title."""
    translator = _translator()
    titles = ["Official development assistance to " + "sub-Saharan Africa " * 4] * 3
    call = types.FunctionCall(
        id="call_1", name="make_charts", args={"charts": [{"title": title} for title in titles]}
    )
    answer = {"charts": [_refused_chart(title, "binding_invalid") for title in titles]}

    frames = _frames(
        translator,
        _event(types.Part(function_call=call)),
        _event(
            types.Part(
                function_response=types.FunctionResponse(
                    id="call_1", name="make_charts", response=answer
                )
            )
        ),
    )

    failed = next(f for f in frames if f.event == "activity.step" and f.status is StepStatus.FAILED)
    reason = classed_sentence("binding_invalid", "do_not_retry").removesuffix(".")
    assert failed.detail.startswith(f"{reason}: Official development assistance")
    assert is_user_sentence(failed.detail)


def _cut_chart(title: str, *, drawn: int, holding: int, other_kinds: int, unvalued: int) -> dict:
    """The `make_charts` entry of a card cut to its `drawn` largest members,
    with what it holds as `harness.reply_figures.card_holds` states it."""
    cut = TopMembers(
        members=[f"m{n}" for n in range(drawn)],
        period="2024",
        holding=holding,
        unvalued=[f"u{n}" for n in range(unvalued)],
        other_kinds=other_kinds,
    )
    series = [
        DrawnSeries(id=member, label=member, unit="USD", points=(("2024", 1.0),))
        for member in cut.members
    ]
    return {"title": title, "holds": card_holds(series, cut=cut), "reply_figures": []}


def _charts_step(*charts: dict[str, Any]) -> ActivityStep:
    """The closing line of a `make_charts` call that answered `charts`."""
    call = types.FunctionCall(
        id="call_1", name="make_charts", args={"charts": [{"title": c["title"]} for c in charts]}
    )
    answered = types.FunctionResponse(
        id="call_1", name="make_charts", response={"charts": list(charts)}
    )
    frames = _frames(
        _translator(),
        _event(types.Part(function_call=call)),
        _event(types.Part(function_response=answered)),
    )
    return next(f for f in frames if f.event == "activity.step" and f.status is StepStatus.COMPLETE)


def test_a_chart_line_states_the_counts_of_each_card_it_cut():
    """A cut card's counts reach the stream in its step's detail: how many it
    draws, of how many with a value, and how many it left out of each kind.
    A zero count is omitted, and a card drawn whole states nothing."""
    whole = {"title": "Aid over time", "holds": {"series_count": 2}, "reply_figures": []}
    cut = _cut_chart("Aid by donor", drawn=24, holding=70, other_kinds=15, unvalued=3)
    plain = _cut_chart("Aid by sector", drawn=20, holding=31, other_kinds=0, unvalued=0)

    assert _charts_step(whole).detail is None
    detail = _charts_step(whole, cut).detail
    assert checks._numbers_in_prose(detail) == {24, 70, 2024, 15, 3}
    assert is_user_sentence(detail)
    assert checks._numbers_in_prose(_charts_step(plain).detail) == {20, 31, 2024}
    # One sentence a cut card, in the order drawn.
    both = _charts_step(cut, plain).detail
    assert both == f"{detail} {_charts_step(plain).detail}"


_YEAR_COUNTS_MANIFEST = json.loads(
    (Path(__file__).parent / "fixtures" / "mint_manifest_recent_years.json").read_text()
)


class _PlanSurface:
    """A `ToolSurface` answering each plan step's call with the structured
    content scripted for its tool."""

    def __init__(self, results: Mapping[str, Mapping[str, Any]]) -> None:
        self._results = results

    async def list_tools(self) -> list[ToolSchema]:
        raise AssertionError("not called")

    async def call_tool(
        self,
        *,
        tool: ToolId,
        arguments: Mapping[str, Any],
        on_progress: ProgressCallback | None = None,
    ) -> ToolCallResult:
        return ToolCallResult(
            text="digest", structured_content=dict(self._results[tool.name]), is_error=False
        )


async def _plan_steps(results: Mapping[str, Mapping[str, Any]]) -> dict[str, ActivityStep]:
    """The closing line of each plan step, by tool, after a plan whose steps
    call each tool of `results` once through the recording surface."""
    current_conversation_id.set("c1")
    surface = _RecordingSurface(_PlanSurface(results), provenance=ProvenanceStore())
    translator = _Translator(
        conversation=ConversationState(conversation_id="c1"),
        recorder=_RecordingExecutor(_no_executor, store=StudioStore(InMemorySessionService())),
        surface_recorder=surface,
    )
    for name in results:
        server = "data-mcp" if name == "get_data" else "analysis-mcp"
        token = current_step_id.set(f"step_{name}")
        try:
            await surface.call_tool(tool=ToolId(server=server, name=name), arguments={})
        finally:
            current_step_id.reset(token)
    return {f.tool: f for f in translator._drain_recorder() if f.event == "activity.step"}


def _verb_result(rows: int, *, ranked_before_limit: int | None = None) -> dict[str, Any]:
    """An analysis verb's structured content, as analysis-mcp builds it."""
    return VerbResult(
        spec_id="ops_1",
        content_version="v1",
        frame_content_version="cv_1",
        rows=rows,
        stat_var_count=1,
        ranked_before_limit=ranked_before_limit,
    ).model_dump(mode="json")


async def test_a_plan_steps_line_states_the_period_note_and_the_rows_its_result_holds():
    """A plan's retrieval states the period note of its manifest's coverage
    after its summary, and the summary alone when the coverage gives none. A
    ranking and any other analysis step state the rows their result holds."""
    coverage = _YEAR_COUNTS_MANIFEST["coverage"]
    note = period_note(coverage)
    assert note is not None
    unnoted = copy.deepcopy(_YEAR_COUNTS_MANIFEST)
    del unnoted["coverage"]["recent_years"]

    steps = await _plan_steps(
        {
            "get_data": _YEAR_COUNTS_MANIFEST,
            "rank": _verb_result(10, ranked_before_limit=70),
            "growth": _verb_result(1),
        }
    )
    plain = (await _plan_steps({"get_data": unnoted}))["get_data"].detail

    assert plain and "\u00b7 The data is for" not in plain
    assert steps["get_data"].detail == f"{plain} \u00b7 {note}"
    assert checks._numbers_in_prose(steps["rank"].detail) == {10, 70}
    unlimited = (await _plan_steps({"rank": _verb_result(10)}))["rank"].detail
    assert checks._numbers_in_prose(unlimited) == {10}
    assert checks._numbers_in_prose(steps["growth"].detail) == {1}
    assert all(is_user_sentence(steps[tool].detail) for tool in ("rank", "growth"))


async def test_a_reply_stating_the_facts_of_its_steps_is_grounded_by_their_frames():
    """The Studio eval grounds a reply's numbers in the stream alone
    (`evals/studio/checks.py`'s `check_grounding`). The period note's year and
    count, a cut card's counts and a ranking's row count are grounded by the
    step frames that state them, and by nothing else in the turn."""
    manifest = copy.deepcopy(_YEAR_COUNTS_MANIFEST)
    counts = {2025: 9, 2024: 85, 2023: 84}
    for dimension in manifest["coverage"]["dimensions"]:
        if dimension["label"] == "financingProvider":
            dimension["count"] = 85
    for year in manifest["coverage"]["recent_years"]:
        year["role_counts"]["financingProvider"] = counts[year["year"]]
    note = period_note(manifest["coverage"])
    assert note is not None

    fetch = types.FunctionCall(id="call_f", name="fetch", args={"spec_id": "spec_oda"})
    fetched = types.FunctionResponse(
        id="call_f", name="fetch", response={"summary": "170 observations", "period_note": note}
    )
    steps = [
        *_frames(
            _translator(),
            _event(types.Part(function_call=fetch)),
            _event(types.Part(function_response=fetched)),
        ),
        _charts_step(_cut_chart("Aid", drawn=24, holding=70, other_kinds=15, unvalued=31)),
        *(
            await _plan_steps(
                {"get_data": manifest, "rank": _verb_result(25, ranked_before_limit=64)}
            )
        ).values(),
    ]
    reply = {
        "event": "message.complete",
        "text": (
            "The data is for 2024, which covers 85 donors, while 2025 covers 9 donors so far. "
            "The chart shows the 24 largest of 70 donors, excluding 15 group totals and 31 "
            "with no value. The ranking keeps 25 of 64 ranked donors."
        ),
    }
    turn = [frame.model_dump(mode="json") for frame in steps]

    assert checks.check_grounding([[*turn, reply]]) == []
    # The same turn with its step details withheld grounds none of them.
    bare = [{**frame, "detail": None} for frame in turn]
    (violation,) = checks.check_grounding([[*bare, reply]])
    assert all(number in violation for number in ("2025", "85", "70", "15", "31", "25", "64"))


def _record(tool: str, *, step_id: str | None = None, **result: object) -> _ExecutorRecord:
    return _ExecutorRecord(
        tool=tool, args={}, result=result, duration_ms=5, conversation="c1", step_id=step_id
    )


@pytest.mark.parametrize(
    ("record", "shown"),
    [
        # A fetch's own retrieval and check: its fetch line stands for them.
        (_record("get_data", outcome="ok", summary="24 observations"), False),
        (_record("describe_frame", outcome="ok"), False),
        (_record("get_skill", outcome="ok"), False),
        # The same calls failing still show.
        (_record("get_data", outcome="refusal", refusal={"code": "x", "message": "m"}), True),
        (_record("describe_frame", outcome="refusal", refusal={"code": "x", "message": "m"}), True),
        # A plan step's call, and a card, always show.
        (_record("get_data", step_id="aid_received", outcome="ok"), True),
        (_record("write_component", outcome="ok", summary="table: Aid"), True),
    ],
)
def test_calls_an_agent_tool_made_on_its_own_behalf_show_only_when_they_fail(record, shown):
    assert _translator()._unshown(record) is not shown


def test_a_hand_off_to_the_planner_is_one_line_until_the_planner_drafts():
    """The hand-off opens the line the planner node opens as it starts, so
    the user sees one "Working on a plan" from the hand-off on; the agent's
    run winds down after it, and that thinking or text must not close it."""
    translator = _translator()
    call = types.FunctionCall(
        id="call_1", name="plan", args={"brief": "growth", "asks": ["growth"]}
    )
    taken = types.FunctionResponse(
        id="call_1", name="plan", response={"status": "Recorded.", HANDED_OFF: "plan"}
    )
    planner_started = Event(
        author="dfo_analyst_graph",
        node_info=NodeInfo(path="dfo_analyst_graph/planner@1"),
        custom_metadata={"planner_started": {"revising": False}},
    )

    frames = _frames(
        translator,
        _event(types.Part(function_call=call)),
        _event(types.Part(function_response=taken)),
        _event(
            types.Part(text="**Wrapping up**\n\nThe planner has it.", thought=True), partial=True
        ),
        _event(types.Part(text="Handed off."), partial=True),
        _event(types.Part(text="Handed off.")),
        planner_started,
    )

    lines = [f for f in frames if f.event == "activity.step" and f.tool is None]
    assert [(f.label, f.status) for f in lines] == [("Working on a plan", StepStatus.RUNNING)]
    assert "Reading the results" not in [label for label, _ in _steps(frames)]


def test_a_hand_off_to_a_question_opens_no_line():
    translator = _translator()
    call = types.FunctionCall(id="call_1", name="ask_user", args={"question": "Which Congo?"})
    taken = types.FunctionResponse(
        id="call_1", name="ask_user", response={"status": "Recorded.", HANDED_OFF: "ask"}
    )

    frames = _frames(
        translator,
        _event(types.Part(function_call=call)),
        _event(types.Part(function_response=taken)),
    )

    assert [f for f in frames if f.event == "activity.step" and f.tool is None] == []


def test_plan_path_step_progress_becomes_a_running_fetch_line():
    progress = Event(
        author="t",
        custom_metadata={
            STEP_PROGRESS_KEY: {"step_id": "req_1", "tool": "get_data", "message": "minting"}
        },
    )

    frames = _frames(_translator(), progress)

    assert [(f.label, f.detail, f.status) for f in frames] == [
        ("Fetching data", "minting", StepStatus.RUNNING)
    ]


def test_tool_call_labels_come_from_the_arguments():
    assert _tool_call_label("search", {"query": "GDP per capita in Kenya"}) == "Searching for data"
    assert _tool_call_label("fetch", {"spec_id": "spec_x"}) == "Fetching data"
    assert _tool_call_label("fetch", {"spec_id": "spec_x"}, "GDP") == "Fetching GDP"
    assert _tool_call_label("make_charts", {"charts": [{}, {}]}) == "Making 2 charts"
    assert _tool_call_label("make_charts", {"charts": [{}]}) == "Making 1 chart"
    assert _tool_call_label("get_skill", {"skill_id": "oda-measures"}) == "Reading background notes"
    assert _tool_call_label("plan", {"brief": "b"}) == "Working on a plan"
    assert _tool_call_label("explain_dfo", {}) == "Reading about DFO"
    assert _tool_call_label("recall", {"part": "plan"}) == "Looking back at an earlier turn"
    assert _tool_call_label("recall", {"part": "canvas"}) == "Looking at your canvas"
    assert (
        _tool_call_label("recall", {"part": "canvas", "component_id": "cmp_1"})
        == "Looking back at an earlier turn"
    )
    # A tool with no label is never shown by its machine name.
    assert _tool_call_label("sources", {}) == "Working on your request"


def test_a_card_line_names_its_kind_and_title_and_repeats_neither_in_its_detail():
    card = _record("write_component", outcome="ok", summary="line_chart: Aid to Kenya, 2000-2023")
    untitled = _record("write_component", outcome="ok", summary=None)
    step = _record("filter", step_id="kept_1990", outcome="ok", summary="filter: 1/34 row(s) kept")

    assert _record_label(card, {}) == "Line chart: Aid to Kenya, 2000-2023"
    assert _record_detail(card) is None
    assert _record_label(untitled, {}) == "Adding a chart"
    # A plan step's line takes the plan's label and keeps its tool's result.
    assert _record_label(step, {"kept_1990": "Filter life expectancy (1)"}) == (
        "Filter life expectancy (1)"
    )
    assert _record_detail(step) == "filter: 1/34 row(s) kept"
    assert _record_label(step, {}) == "Filtering data"


def _reading_lines(frames: list) -> list[tuple[str, StepStatus, str | None]]:
    return [
        (f.label, f.status, f.detail)
        for f in frames
        if f.event == "activity.step" and f.tool is None
    ]


def _batch(*pairs: tuple[str, dict[str, Any], dict[str, Any]]) -> tuple[Event, Event]:
    """One model turn's parallel calls, and their responses in one event."""
    calls = [
        types.FunctionCall(id=f"call_{index}", name=name, args=args)
        for index, (name, args, _) in enumerate(pairs)
    ]
    responses = [
        types.FunctionResponse(id=f"call_{index}", name=name, response=answer)
        for index, (name, _, answer) in enumerate(pairs)
    ]
    return (
        _event(*(types.Part(function_call=call) for call in calls)),
        _event(*(types.Part(function_response=response) for response in responses)),
    )


def test_a_search_line_carries_the_limits_its_result_states_in_words():
    found = {
        "kind": "resolution",
        "parts": [
            {
                "variable": "aid to Kenya",
                "status": "definite",
                "spec_id": "spec_aid",
                "resolved": "Health aid to Kenya",
                "window": {"start": 2000, "end": 2023, "source": "coverage"},
                "warnings": [{"code": "x", "message": "dcid/Thing was dropped"}],
            },
            {
                "variable": "debt",
                "status": "candidates",
                "spec_id": None,
                "candidates": [{"spec_id": "spec_a"}, {"spec_id": "spec_b"}],
            },
            {"variable": "maternal deaths", "status": "no_data", "reason": "no_series"},
            {"variable": "aid per person", "status": "unsupported"},
        ],
    }

    frames = _frames(_translator(), *_batch(("search", {"query": "q"}, found)))

    assert _reading_lines(frames) == [
        (
            "Reading what the search found",
            StepStatus.RUNNING,
            "Health aid to Kenya, 2000\u20132023; debt: 2 possible readings; "
            "no data for maternal deaths; no supported dataset for aid per person",
        )
    ]


class _NoSurface:
    """A `ToolSurface` the translator only reads manifests beside."""

    async def list_tools(self) -> list[ToolSchema]:
        raise AssertionError("not called")

    async def call_tool(
        self,
        *,
        tool: ToolId,
        arguments: Mapping[str, Any],
        on_progress: ProgressCallback | None = None,
    ) -> ToolCallResult:
        raise AssertionError("not called")


@pytest.mark.parametrize(
    ("window", "label"),
    [
        ({"start_year": 2000, "end_year": 2023}, "Looking at life expectancy, 2000\u20132023"),
        ({"start_year": 2023, "end_year": 2023}, "Looking at life expectancy, 2023"),
        # A retrieval that states no years: the summary's own are not parsed.
        ({}, "Looking at life expectancy"),
    ],
)
async def test_one_fetch_names_its_measure_and_the_years_its_retrieval_covers(window, label):
    provenance = ProvenanceStore()
    ref = {"spec_id": "spec_le", "content_version": "v1"}
    summary = "24 observations, 2000-2023 \u00b7 World Development Indicators"

    async def get_data(*, tool: str, args: dict[str, object], on_progress=None) -> dict:
        manifest = {"frame_ref": ref, "coverage": {"rows": 24, "window": window}}
        provenance.put(
            conversation=current_conversation_id.get(),
            spec_id="spec_le",
            content_version="v1",
            manifest=manifest,
        )
        return {"outcome": "ok", "frame_refs": [ref], "summary": summary}

    conversation = ConversationState(conversation_id="c1")
    conversation.turn_readings = [
        {"text": "life expectancy", "status": "definite", "spec_id": "spec_le"}
    ]
    recorder = _RecordingExecutor(get_data, store=StudioStore(InMemorySessionService()))
    translator = _Translator(
        conversation=conversation,
        recorder=recorder,
        surface_recorder=_RecordingSurface(_NoSurface(), provenance=provenance),
    )
    called, responded = _batch(("fetch", {"spec_id": "spec_le"}, {"summary": summary}))

    frames = translator.translate(called)
    # The fetch tool's own retrieval, recorded before its response arrives.
    await recorder(tool="get_data", args={"spec_id": "spec_le"})
    frames += translator.translate(responded)

    assert _reading_lines(frames) == [(label, StepStatus.RUNNING, summary)]


@pytest.mark.parametrize(
    ("batch", "label"),
    [
        # The later stage names a mixed batch.
        (
            [("search", {"query": "q"}, {"parts": []}), ("fetch", {"spec_id": "s"}, {})],
            "Looking at the data",
        ),
        (
            [("fetch", {"spec_id": "s"}, {}), ("make_charts", {"charts": [{}]}, {"charts": []})],
            "Going over the charts",
        ),
        (
            [("recall", {"part": "plan"}, {}), ("search", {"query": "q"}, {"parts": []})],
            "Reading what the search found",
        ),
        # A batch that hands off opens no line.
        (
            [
                ("search", {"query": "q"}, {"parts": []}),
                ("ask_user", {"question": "Which?"}, {HANDED_OFF: "ask"}),
            ],
            None,
        ),
        ([("explain_dfo", {}, {"explainer": "DFO is"})], "Going over the notes on DFO"),
        # A refused hand-off leaves the agent to choose its next move.
        (
            [
                (
                    "ask_user",
                    {"question": "Which?"},
                    {"error": "ask_user needs at least one option to offer."},
                )
            ],
            "Working out the next step",
        ),
        (
            [("plan", {"brief": "b"}, {"error": "plan has been called 1 time(s) this turn."})],
            "Working out the next step",
        ),
    ],
)
def test_parallel_calls_give_one_line_after_the_last_response(batch, label):
    frames = _frames(_translator(), *_batch(*batch))

    lines = _reading_lines(frames)
    assert lines == ([(label, StepStatus.RUNNING, None)] if label else [])


@pytest.mark.parametrize(
    "next_event",
    [
        _event(types.Part(function_call=types.FunctionCall(id="n", name="fetch", args={}))),
        _event(types.Part(text="Kenya's"), partial=True),
        _event(types.Part(text="Kenya's GDP was $101 billion.")),
        _event(
            types.Part(function_call=types.FunctionCall(id="n", name="plan", args={"brief": "b"}))
        ),
    ],
    ids=["tool call", "streamed text", "whole text", "hand-off"],
)
def test_a_reading_line_closes_when_anything_else_happens(next_event):
    translator = _translator()
    _frames(translator, *_batch(("search", {"query": "q"}, {"parts": []})))

    closing = translator.translate(next_event)

    line = next(f for f in closing if f.event == "activity.step" and f.tool is None)
    assert (line.label, line.status) == ("Reading what the search found", StepStatus.COMPLETE)
    assert line.duration_ms is not None


def test_a_reading_line_closes_when_the_segment_ends():
    translator = _translator()
    _frames(translator, *_batch(("search", {"query": "q"}, {"parts": []})))

    assert _steps(translator.finish()) == [("Reading what the search found", StepStatus.COMPLETE)]


def test_a_skill_read_keeps_the_reading_line_running():
    translator = _translator()
    frames = _frames(
        translator,
        *_batch(("search", {"query": "q"}, {"parts": []})),
        *_batch(("get_skill", {"skill_id": "oda-measures"}, {"text": "notes"})),
    )

    assert _reading_lines(frames) == [("Reading what the search found", StepStatus.RUNNING, None)]


def test_cards_the_agent_removed_reach_the_client_as_a_composition_update():
    """The website drops a card only on a manifest without it; a refused
    removal changes nothing and sends none."""
    conversation = ConversationState(conversation_id="c1")
    for cid in ("cmp_01", "cmp_02"):
        land_component(
            conversation,
            {"component_id": cid, "revision": 1, "type": "line_chart", "title": "Aid"},
        )
    translator = _Translator(
        conversation=conversation,
        recorder=_RecordingExecutor(_no_executor, store=StudioStore(InMemorySessionService())),
    )
    remove_cards(conversation, ["cmp_01"], all_cards=False)

    frames = _frames(
        translator,
        *_batch(("remove_cards", {"component_ids": ["cmp_01"]}, {"removed": ["cmp_01"]})),
        *_batch(("remove_cards", {"component_ids": []}, {"error": "names no card"})),
        *_batch(("remove_cards", {"component_ids": ["cmp_01"]}, {"removed": [], "note": "n"})),
    )

    updates = [f for f in frames if f.event == "composition.updated"]
    assert len(updates) == 1
    assert [b.component_id for b in updates[0].manifest.blocks] == ["cmp_02"]
    assert ("Removing cards from the canvas", StepStatus.FAILED) in _steps(frames)


async def test_a_removal_reported_after_a_newer_canvas_edit_carries_that_edit():
    """`/composition` can commit a newer revision between the agent's removal
    and the drain that reports it, here a pin. The client renders every
    update, so the removal's update carries the stored manifest: never a
    revision below the one the pin committed, and the pin with it."""
    store = StudioStore(InMemorySessionService(), app_name="app")
    await store.create("principal", "c1")
    conversation = store.current("principal", "c1")
    assert conversation is not None
    for cid in ("cmp_01", "cmp_02"):
        land_component(
            conversation,
            {"component_id": cid, "revision": 1, "type": "line_chart", "title": "Aid"},
        )
    translator = _Translator(
        conversation=conversation,
        recorder=_RecordingExecutor(_no_executor, store=store),
    )
    remove_cards(conversation, ["cmp_01"], all_cards=False)

    def pin(manifest: CompositionManifest | None) -> CompositionManifest:
        assert manifest is not None
        blocks = [block.model_copy(update={"pinned": True}) for block in manifest.blocks]
        return _next_manifest(manifest, blocks, actor=CompositionActor.USER)

    pinned = await store.mutate_composition("principal", "c1", pin)
    assert isinstance(pinned, CompositionManifest)

    frames = _frames(
        translator,
        *_batch(("remove_cards", {"component_ids": ["cmp_01"]}, {"removed": ["cmp_01"]})),
    )

    (update,) = [f for f in frames if f.event == "composition.updated"]
    assert update.manifest.revision >= pinned.revision
    assert [(b.component_id, b.pinned) for b in update.manifest.blocks] == [("cmp_02", True)]


def test_a_manifest_older_than_one_the_stream_sent_is_logged_and_not_sent(caplog):
    """No path gives the translator a manifest below a revision it sent. One
    forced on it is withheld and logged with both revisions, and a manifest
    at the revision last sent goes out."""
    conversation = ConversationState(conversation_id="c1")
    for cid in ("cmp_01", "cmp_02", "cmp_03"):
        land_component(
            conversation,
            {"component_id": cid, "revision": 1, "type": "line_chart", "title": "Aid"},
        )
    translator = _Translator(
        conversation=conversation,
        recorder=_RecordingExecutor(_no_executor, store=StudioStore(InMemorySessionService())),
    )
    removal = ("remove_cards", {"component_ids": ["cmp_01"]}, {"removed": ["cmp_01"]})
    older = conversation.composition
    remove_cards(conversation, ["cmp_01"], all_cards=False)
    sent = conversation.composition
    assert older is not None and sent is not None

    def revisions(*events: Event) -> list[int]:
        frames = _frames(translator, *events)
        return [f.manifest.revision for f in frames if f.event == "composition.updated"]

    assert revisions(*_batch(removal)) == [sent.revision]
    # The same revision again, as after a `/composition` response the client holds.
    conversation._removal_unreported = True
    assert revisions(*_batch(removal)) == [sent.revision]

    conversation.composition = older
    conversation._removal_unreported = True
    with caplog.at_level(logging.ERROR, logger="harness.studio_translate"):
        assert revisions(*_batch(removal)) == []
    (record,) = caplog.records
    assert record.args == (older.revision, sent.revision)


def test_a_canvas_listing_is_read_under_its_own_line():
    translator = _translator()

    frames = _frames(translator, *_batch(("recall", {"part": "canvas"}, {"on_canvas": []})))

    assert ("Going over your canvas", StepStatus.RUNNING) in _steps(frames)
