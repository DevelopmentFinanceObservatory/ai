"""`_OrchestratorTranslator` over real ADK events, in the order a streamed run yields them."""

from __future__ import annotations

from google.adk.events import Event
from google.adk.sessions import InMemorySessionService
from google.genai import types
from harness.graph import ORCHESTRATOR_STATUS_KEY, STEP_PROGRESS_KEY
from harness.studio_state import ConversationState, StudioStore
from harness.studio_translate import (
    _OrchestratorTranslator,
    _RecordingExecutor,
    _thought_heading,
    _tool_call_label,
)
from harness.wire import StepStatus


async def _no_executor(*, tool: str, args: dict[str, object]) -> dict[str, object]:
    raise AssertionError(f"no tool call expected, got {tool}")


def _translator() -> _OrchestratorTranslator:
    return _OrchestratorTranslator(
        conversation=ConversationState(conversation_id="c1"),
        recorder=_RecordingExecutor(_no_executor, store=StudioStore(InMemorySessionService())),
    )


def _event(
    *parts: types.Part, partial: bool | None = None, author: str = "dfo_orchestrator"
) -> Event:
    return Event(author=author, partial=partial, content=types.Content(parts=list(parts)))


def _frames(translator: _OrchestratorTranslator, *events: Event) -> list:
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


def test_a_tool_call_runs_from_the_call_and_completes_on_its_response():
    translator = _translator()
    call = types.FunctionCall(
        id="call_1", name="search", args={"query": "life expectancy in Kenya"}
    )
    response = types.FunctionResponse(id="call_1", name="search", response={"parts": []})

    frames = _frames(
        translator,
        _event(types.Part(function_call=call)),
        _event(types.Part(function_response=response)),
    )

    assert _steps(frames) == [
        ("Searching: life expectancy in Kenya", StepStatus.RUNNING),
        ("Searching: life expectancy in Kenya", StepStatus.COMPLETE),
        ("Reading the results", StepStatus.RUNNING),
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
    taken = types.FunctionResponse(id="call_1", name="ask_user", response={"status": "Recorded."})

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


def test_a_tool_that_returned_an_error_is_a_failed_step_with_the_reason():
    translator = _translator()
    call = types.FunctionCall(id="call_1", name="search", args={"query": "x"})
    failed = types.FunctionResponse(
        id="call_1", name="search", response={"error": "search is down"}
    )

    frames = _frames(
        translator,
        _event(types.Part(function_call=call)),
        _event(types.Part(function_response=failed)),
    )

    assert [(f.status, f.detail) for f in frames if f.event == "activity.step"][:2] == [
        (StepStatus.RUNNING, None),
        (StepStatus.FAILED, "search is down"),
    ]


def test_step_ids_stay_unique_across_the_segments_of_one_turn():
    """Studio resumes a paused turn into one Activity entry keyed by step_id,
    and each segment gets a fresh translator over the same conversation."""
    conversation = ConversationState(conversation_id="c1")
    recorder = _RecordingExecutor(_no_executor, store=StudioStore(InMemorySessionService()))
    first = _OrchestratorTranslator(conversation=conversation, recorder=recorder)
    second = _OrchestratorTranslator(conversation=conversation, recorder=recorder)
    ask = types.FunctionCall(id="call_1", name="ask_user", args={"question": "Which?"})
    explain = types.FunctionCall(id="call_2", name="explain_dfo", args={})
    thought = types.Part(text="**Reading**\n\nhmm", thought=True)

    before = _frames(first, _event(types.Part(function_call=ask)), _event(thought, partial=True))
    after = _frames(
        second, _event(types.Part(function_call=explain)), _event(thought, partial=True)
    )

    ids = [f.step_id for f in [*before, *after] if f.event == "activity.step"]
    assert len(ids) == len(set(ids)) == 4


def test_a_refused_hand_off_leaves_the_models_text_showing():
    translator = _translator()
    call = types.FunctionCall(id="call_1", name="plan", args={"brief": "b"})
    refused = types.FunctionResponse(id="call_1", name="plan", response={"error": "limit"})

    frames = _frames(
        translator,
        _event(types.Part(function_call=call)),
        _event(types.Part(function_response=refused)),
        _event(types.Part(text="I cannot plan this; here is what I found."), partial=True),
    )

    assert [f.text for f in frames if f.event == "message.delta"] == [
        "I cannot plan this; here is what I found."
    ]


def test_a_thought_shows_its_latest_heading_while_the_model_thinks():
    translator = _translator()

    frames = _frames(
        translator,
        _event(
            types.Part(text="**Reading the question**\n\nThe user wants", thought=True),
            partial=True,
        ),
        _event(
            types.Part(text=" a trend.\n\n**Choosing a chart**\n\nA line", thought=True),
            partial=True,
        ),
        _event(types.Part(text="…", thought=True), types.Part(text="Here it is.")),
    )

    assert _steps(frames) == [
        ("Reading the question", StepStatus.RUNNING),
        ("Choosing a chart", StepStatus.RUNNING),
        ("Choosing a chart", StepStatus.COMPLETE),
    ]
    assert len({f.step_id for f in frames if f.event == "activity.step"}) == 1


def test_a_thought_heading_is_a_short_first_line_of_the_latest_paragraph():
    assert (
        _thought_heading("**Reading the question**\n\nThe user wants a trend.")
        == "Reading the question"
    )
    assert _thought_heading("# Choosing\nmore") == "Choosing"
    assert _thought_heading("**" + "x" * 81 + "**") is None
    assert _thought_heading("Just prose, no heading.") is None
    assert _thought_heading("") is None


def test_plan_path_events_go_through_the_graphs_own_translation():
    progress = Event(
        author="t",
        custom_metadata={
            STEP_PROGRESS_KEY: {"step_id": "req_1", "tool": "get_data", "message": "minting"}
        },
    )

    frames = _frames(_translator(), progress)

    assert [(f.label, f.detail, f.status) for f in frames] == [
        ("Fetch the data", "minting", StepStatus.RUNNING)
    ]


def test_tool_call_labels_come_from_the_arguments():
    assert _tool_call_label("search", {"query": "GDP per capita in Kenya"}) == (
        "Searching: GDP per capita in Kenya"
    )
    assert _tool_call_label("make_charts", {"charts": [{}, {}]}) == "Drawing 2 charts"
    assert _tool_call_label("get_skill", {"skill_id": "oda-measures"}) == (
        "Reading the oda-measures skill"
    )
    assert _tool_call_label("get_skill_file", {"skill_id": "oda-measures", "path": "a.md"}) == (
        "Reading the oda-measures skill"
    )
    assert _tool_call_label("plan", {"brief": "b"}) == "Handing the question to the planner"
    assert _tool_call_label("explain_dfo", {}) == "Reading about DFO"
