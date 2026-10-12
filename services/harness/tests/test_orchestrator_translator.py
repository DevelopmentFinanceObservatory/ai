"""`_Translator` over real ADK events, in the order a streamed run yields them."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest
from compiler.surface import ProgressCallback, ToolCallResult, ToolSchema
from google.adk.events import Event
from google.adk.events.event import NodeInfo
from google.adk.sessions import InMemorySessionService
from google.genai import types
from harness.conversation import current_conversation_id
from harness.graph.state import (
    ORCHESTRATOR_PLAN_FAILED_STATUS,
    ORCHESTRATOR_PLAN_RAN_STATUS,
    ORCHESTRATOR_STATUS_KEY,
    STEP_PROGRESS_KEY,
)
from harness.orchestrator_tools import HANDED_OFF
from harness.provenance import ProvenanceStore
from harness.studio_state import ConversationState, StudioStore, land_component, remove_cards
from harness.studio_translate import (
    _ExecutorRecord,
    _record_detail,
    _record_label,
    _RecordingExecutor,
    _RecordingSurface,
    _tool_call_label,
    _Translator,
)
from harness.wire import StepStatus
from planner.workflow import ToolId


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
    ("answer", "detail"),
    [
        ({"error": "search is down"}, "search is down"),
        ({"refusal": "too big", "code": "over_ceiling"}, "too big"),
        # A refusal that states no reason still fails its step.
        ({"refusal": None, "code": "over_ceiling"}, None),
    ],
)
def test_a_tool_that_returned_an_error_or_a_refusal_is_a_failed_step_with_the_reason(
    answer, detail
):
    translator = _translator()
    call = types.FunctionCall(id="call_1", name="fetch", args={"spec_id": "spec_x"})
    failed = types.FunctionResponse(id="call_1", name="fetch", response=answer)

    frames = _frames(
        translator,
        _event(types.Part(function_call=call)),
        _event(types.Part(function_response=failed)),
    )

    assert [(f.status, f.detail) for f in frames if f.event == "activity.step"][:2] == [
        (StepStatus.RUNNING, None),
        (StepStatus.FAILED, detail),
    ]


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
    assert ("Handing the question to the planner", StepStatus.FAILED) in _steps(frames)


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


def test_a_plan_the_planner_could_not_make_shows_as_a_failed_step():
    translator = _translator()

    frames = _frames(
        translator,
        Event(author="t", custom_metadata={ORCHESTRATOR_STATUS_KEY: "Reading your question"}),
        Event(
            author="t", custom_metadata={ORCHESTRATOR_STATUS_KEY: ORCHESTRATOR_PLAN_FAILED_STATUS}
        ),
    )

    assert _steps(frames) == [
        ("Reading your question", StepStatus.RUNNING),
        ("Reading your question", StepStatus.COMPLETE),
        ("The planner could not make a plan", StepStatus.FAILED),
        (ORCHESTRATOR_PLAN_FAILED_STATUS, StepStatus.RUNNING),
    ]


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

    assert [(f.label, f.status, f.detail) for f in frames if f.event == "activity.step"] == [
        ("Reading background notes", StepStatus.FAILED, "no skill nope"),
    ]


def test_a_fetch_names_the_measure_its_search_found_and_carries_its_summary():
    translator = _translator()
    search = types.FunctionCall(
        id="call_1", name="search", args={"query": "life expectancy and infant mortality in Rwanda"}
    )
    found = {
        "parts": [
            {"variable": "life expectancy", "status": "definite", "spec_id": "spec_le"},
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
                        id=call.id, name="fetch", response={"summary": summary}
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
            "Found life expectancy, infant mortality; no data for maternal deaths; "
            "no supported dataset for aid per person",
        ),
        ("Fetching life expectancy", StepStatus.RUNNING, None),
        ("Fetching infant mortality", StepStatus.RUNNING, None),
        ("Fetching life expectancy", StepStatus.COMPLETE, summary),
        ("Fetching infant mortality", StepStatus.COMPLETE, summary),
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


@pytest.mark.parametrize(
    ("answer", "detail"),
    [
        (
            {"charts": [{"title": "A", "facts": {}}, {"title": "B", "error": "no such column"}]},
            "1 of 2 could not be drawn: B: no such column",
        ),
        (
            {
                "charts": [{"title": "A", "facts": {}}],
                "not_drawn": "1 charts past the 1 one call may draw",
            },
            "1 of 2 could not be drawn: 1 charts past the 1 one call may draw",
        ),
    ],
)
def test_a_chart_call_that_drew_only_some_of_its_charts_fails(answer, detail):
    translator = _translator()
    call = types.FunctionCall(id="call_1", name="make_charts", args={"charts": [{}, {}]})

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

    assert [(f.label, f.status, f.detail) for f in frames if f.event == "activity.step"][:2] == [
        ("Making 2 charts", StepStatus.RUNNING, None),
        ("Making 2 charts", StepStatus.FAILED, detail),
    ]


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
    call = types.FunctionCall(id="call_1", name="plan", args={"brief": "growth", "asks": []})
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
    assert _tool_call_label("plan", {"brief": "b"}) == "Handing the question to the planner"
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


def test_a_canvas_listing_is_read_under_its_own_line():
    translator = _translator()

    frames = _frames(translator, *_batch(("recall", {"part": "canvas"}, {"on_canvas": []})))

    assert ("Going over your canvas", StepStatus.RUNNING) in _steps(frames)
