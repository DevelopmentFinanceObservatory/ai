"""`orchestrator.run.run_orchestrator` on the streaming orchestrator model.

The model streams through a scripted google-genai client. A stream that
stalls is a call the turn budget cuts.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
from collections.abc import AsyncGenerator
from typing import Any

import pytest
from compiler.surface import transient
from google.genai import types
from harness.gemini_agent import EmptyResponseError
from harness.gemini_wire import StreamResumedError
from harness.graph.state import ORCHESTRATOR_SPECS_KEY
from harness.orchestrator_tools import _DRAWN_CARDS_KEY, build_orchestrator_tools
from harness.prefix_cache import PrefixCache
from harness.sentences import StopCode, refusal_sentence
from harness.studio_state import CardRemoval, ConversationState, land_component, remove_cards
from harness.turn_budget import TurnBudget
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from orchestrator.agent import INSTRUCTION, orchestrator_agent
from orchestrator.model import orchestrator_model
from orchestrator.run import (
    Answered,
    HandBack,
    HandedOff,
    ModelResponded,
    NoCallWaitsError,
    Opening,
    StoredHistory,
    TextDelta,
    ToolsAnswered,
    run_orchestrator,
)
from pydantic_ai.exceptions import ModelAPIError
from pydantic_ai.messages import ModelMessagesTypeAdapter

from ..scripted_genai import (
    Reply,
    ScriptedGenai,
    call_part,
    empty_reply,
    malformed_call_reply,
    reply,
    text_part,
    thought_part,
)
from .pai_runs import TODAY, CannedSearch, definite_envelope

_STALL = object()
_SEARCH = call_part("search", {"parts": [{"variable": "life expectancy"}]})


def _budget(**overrides: Any) -> TurnBudget:
    bounds: dict[str, Any] = {
        "model_time_s": 5.0,
        "model_calls_max": 30,
        "wall_s": 600.0,
        "first_chunk_s": 5.0,
    }
    return TurnBudget(**{**bounds, **overrides})


def _chunk(*parts: types.Part, finish: types.FinishReason | None = None) -> Any:
    return types.GenerateContentResponse(
        candidates=[
            types.Candidate(
                content=types.Content(role="model", parts=list(parts)), finish_reason=finish
            )
        ]
    )


def _streams(scripted: ScriptedGenai, *scripts: list[Any]) -> list[Any]:
    """Answer each call with the next script: chunks in order, `_STALL` waiting
    until the call is cut. Returns the requests, as `scripted.requests` would."""
    calls: list[Any] = []
    pending = list(scripts)

    async def stream(*, model: str, contents: Any, config: Any = None) -> Any:
        calls.append(config)
        script = pending.pop(0)

        async def chunks() -> AsyncGenerator[Any]:
            for step in script:
                if step is _STALL:
                    await asyncio.Event().wait()
                yield step

        return chunks()

    scripted.client.aio.models.generate_content_stream = stream  # type: ignore[method-assign]
    return calls


class _Searches(CannedSearch):
    def __init__(self) -> None:
        super().__init__(definite_envelope(), definite_envelope())
        self.calls = 0

    async def __call__(self, **kwargs: Any) -> dict[str, Any]:
        self.calls += 1
        return await super().__call__(**kwargs)


async def _run(
    scripted: ScriptedGenai,
    *,
    budget: TurnBudget | None = None,
    searches: _Searches | None = None,
    prefix_cache: PrefixCache | None = None,
    tools: Any = None,
    seeded: str | None = None,
    held: dict[str, Any] | None = None,
) -> tuple[list[Any], dict[str, Any]]:
    orchestrator = orchestrator_agent(
        tools
        if tools is not None
        else build_orchestrator_tools(resolver=searches or _Searches(), executor=_no_executor)
    )
    state: dict[str, Any] = {"surface": "studio", **(held or {})}
    model = orchestrator_model(
        "gemini-test", client=scripted.client, budget=budget, prefix_cache=prefix_cache
    )
    events = [
        event
        async for event in run_orchestrator(
            orchestrator,
            history=None,
            state=state,
            turn_id="turn-1",
            model=model,
            opening=Opening(question="Life expectancy in Kenya?", today=TODAY, seeded=seeded),
        )
    ]
    return events, state


async def _no_executor(**kwargs: Any) -> Any:
    raise AssertionError("no executor call expected")


def _texts(events: list[Any]) -> list[str]:
    return [event.text for event in events if isinstance(event, TextDelta)]


# --- the streaming budget ----------------------------------------------------


async def test_a_cut_mid_answer_ends_on_the_closing_sentence_after_the_text_sent() -> None:
    scripted = ScriptedGenai([])
    _streams(
        scripted,
        [_chunk(thought_part("Answer.")), _chunk(text_part("Life expectancy rose")), _STALL],
    )
    budget = _budget(model_time_s=0.1)

    events, _ = await _run(scripted, budget=budget)

    sentence = refusal_sentence({"code": StopCode.TURN_MODEL_TIME_SPENT})
    assert _texts(events) == ["Life expectancy rose", f"\n\n{sentence}"]
    assert events[-1] == Answered(text=f"Life expectancy rose\n\n{sentence}")
    assert budget.closed == {"code": StopCode.TURN_MODEL_TIME_SPENT, "detail": {}}


async def test_a_cut_after_a_function_call_streamed_runs_no_tool() -> None:
    scripted = ScriptedGenai([])
    _streams(scripted, [_chunk(_SEARCH), _STALL])
    searches = _Searches()

    events, _ = await _run(scripted, budget=_budget(model_time_s=0.1), searches=searches)

    sentence = refusal_sentence({"code": StopCode.TURN_MODEL_TIME_SPENT})
    assert searches.calls == 0
    assert _texts(events) == [sentence]
    assert not any(isinstance(event, ToolsAnswered) for event in events)
    assert events[-1] == Answered(text=sentence)


async def test_a_spent_budget_answers_with_its_sentence_and_sends_nothing() -> None:
    scripted = ScriptedGenai([])
    budget = _budget()
    budget.model_calls = budget.model_calls_max

    events, _ = await _run(scripted, budget=budget)

    sentence = refusal_sentence({"code": StopCode.TURN_MODEL_CALLS_SPENT})
    assert scripted.requests == []
    assert events[-1] == Answered(text=sentence)


async def test_answer_text_streams_as_it_arrives_and_thoughts_wait_for_it() -> None:
    """Thoughts are held until the first answer text, then the text passes on chunk
    by chunk: the second chunk is seen before the third arrives."""
    scripted = ScriptedGenai([])
    seen: list[str] = []
    third = asyncio.Event()

    async def stream(*, model: str, contents: Any, config: Any = None) -> Any:
        async def chunks() -> AsyncGenerator[Any]:
            yield _chunk(thought_part("Think."))
            yield _chunk(text_part("One. "))
            yield _chunk(text_part("Two. "))
            await asyncio.wait_for(third.wait(), timeout=2)
            yield _chunk(text_part("Three."), finish=types.FinishReason.STOP)

        return chunks()

    scripted.client.aio.models.generate_content_stream = stream  # type: ignore[method-assign]
    orchestrator = orchestrator_agent(
        build_orchestrator_tools(resolver=_Searches(), executor=_no_executor)
    )
    async for event in run_orchestrator(
        orchestrator,
        history=None,
        state={"surface": "studio"},
        turn_id="t",
        model=orchestrator_model("gemini-test", client=scripted.client, budget=_budget()),
        opening=Opening(question="q", today=TODAY),
    ):
        if isinstance(event, TextDelta):
            seen.append(event.text)
            if event.text == "Two. ":
                third.set()
        if isinstance(event, ModelResponded):
            assert [part.thought for part in event.parts] == [True, None]
    assert seen == ["One. ", "Two. ", "Three."]


async def test_a_malformed_call_before_any_text_is_sent_once_more() -> None:
    scripted = ScriptedGenai([malformed_call_reply(), reply(text_part("Life expectancy is 62."))])

    events, _ = await _run(scripted, budget=_budget())

    assert len(scripted.requests) == 2
    assert _texts(events) == ["Life expectancy is 62."]
    assert events[-1] == Answered(text="Life expectancy is 62.")


async def test_a_malformed_call_after_text_is_not_sent_again() -> None:
    scripted = ScriptedGenai([])
    calls = _streams(
        scripted,
        [
            _chunk(text_part("Life expectancy")),
            _chunk(text_part(""), finish=types.FinishReason.MALFORMED_FUNCTION_CALL),
        ],
    )

    events, _ = await _run(scripted, budget=_budget())

    assert len(calls) == 1
    assert events[-1] == Answered(text="Life expectancy")


# --- tools -------------------------------------------------------------------


async def test_a_raising_tool_is_answered_raised_and_the_run_goes_on(
    caplog: pytest.LogCaptureFixture,
) -> None:
    async def broken(**kwargs: Any) -> Any:
        raise RuntimeError("http://internal.example/secret")

    scripted = ScriptedGenai([reply(_SEARCH), reply(text_part("Search failed."))])
    tools = build_orchestrator_tools(resolver=broken, executor=_no_executor)

    events, _ = await _run(scripted, tools=tools)

    (answered,) = [e for e in events if isinstance(e, ToolsAnswered)]
    sent = json.dumps(scripted.requests[1].content_list[-1].model_dump(mode="json"))
    assert answered.answers[0].response is not None
    assert answered.answers[0].response["code"] == "raised"
    assert "secret" not in sent
    assert events[-1] == Answered(text="Search failed.")
    (logged,) = [r for r in caplog.records if r.name == "orchestrator.agent"]
    assert logged.levelno == logging.ERROR
    assert "turn_id=turn-1" in logged.getMessage()


async def test_a_cancelled_tool_call_ends_the_run_and_logs_nothing(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A cancellation is no tool failure: it propagates out of the run, unanswered."""

    async def cancelled(**kwargs: Any) -> Any:
        raise asyncio.CancelledError

    scripted = ScriptedGenai([reply(_SEARCH), reply(text_part("unreached"))])
    tools = build_orchestrator_tools(resolver=cancelled, executor=_no_executor)

    with caplog.at_level(logging.ERROR), pytest.raises(asyncio.CancelledError):
        await _run(scripted, tools=tools)

    assert len(scripted.requests) == 1
    assert not [r for r in caplog.records if r.name == "orchestrator.agent"]


async def test_a_tool_that_takes_no_call_record_runs_without_one() -> None:
    """`explain_dfo` takes no `call`; with no dfo-faq surface it says it cannot
    describe DFO, and does not raise."""
    scripted = ScriptedGenai([reply(call_part("explain_dfo", {})), reply(text_part("Hello."))])

    events, _ = await _run(scripted)

    (answered,) = [e for e in events if isinstance(e, ToolsAnswered)]
    (response,) = answered.answers
    assert response.name == "explain_dfo"
    assert response.response is not None
    assert response.response.get("code") != "raised"
    assert events[-1] == Answered(text="Hello.")


async def test_a_call_to_a_tool_the_agent_lacks_is_answered_with_the_tools_it_has() -> None:
    scripted = ScriptedGenai([reply(call_part("draw", {})), reply(text_part("Done."))])

    events, _ = await _run(scripted)

    (response,) = [
        part.function_response
        for part in scripted.requests[1].content_list[-1].parts or []
        if part.function_response
    ]
    assert response.name == "draw"
    assert "The tools you can call are: search, fetch" in json.dumps(response.response)
    (answered,) = [e for e in events if isinstance(e, ToolsAnswered)]
    assert [answer.response for answer in answered.answers] == [response.response]
    assert events[-1] == Answered(text="Done.")


async def test_a_response_s_tool_answers_carry_the_state_its_calls_wrote() -> None:
    """The event holds a copy of what the calls wrote: later changes to the
    state, a block the calls wrote changed in place included, leave it as
    sent. A spec block held before the response is shared, since a block is
    replaced whole."""
    scripted = ScriptedGenai([reply(_SEARCH), reply(text_part("Done."))])
    held = {"spec_id": "spec_held"}

    events, state = await _run(scripted, held={ORCHESTRATOR_SPECS_KEY: {"spec_held": held}})

    (answered,) = [e for e in events if isinstance(e, ToolsAnswered)]
    assert answered.state
    assert all(state[key] == value for key, value in answered.state.items())
    assert "orchestrator:searches" in answered.state
    sent = json.dumps(answered.state, sort_keys=True)
    state["orchestrator:searches"].append({"written": "later"})
    state[ORCHESTRATOR_SPECS_KEY]["spec_later"] = {"written": "later"}
    specs = answered.state[ORCHESTRATOR_SPECS_KEY]
    assert specs["spec_held"] is held
    wrote = [spec_id for spec_id in specs if spec_id != "spec_held"]
    assert wrote
    for spec_id in wrote:
        state[ORCHESTRATOR_SPECS_KEY][spec_id]["changed"] = "in place"
    assert json.dumps(answered.state, sort_keys=True) == sent


async def test_a_stream_resumed_over_output_ends_the_run_as_a_dropped_connection() -> None:
    """Pydantic AI's Gemini parser raises a plain `ModelAPIError` when
    google-genai resumed a dropped stream over output no continuation token
    covers; the run raises it as the dropped connection it is."""
    resumed = ModelAPIError("gemini-test", "the stream resumed over a repeated call")
    scripted = ScriptedGenai([Reply(chunks=[[text_part("Kenya")]], cut_after=1, cut_with=resumed)])

    with pytest.raises(StreamResumedError) as raised:
        await _run(scripted)

    assert raised.value.__cause__ is resumed
    assert transient(raised.value)


# --- hand-off and resume -----------------------------------------------------


async def test_a_plan_call_hands_off_and_the_hand_back_answers_it_after_storage() -> None:
    scripted = ScriptedGenai(
        [
            reply(
                thought_part("Plan it.", signature=b"sig-plan"),
                call_part("plan", {"brief": "rank", "asks": ["rank"]}),
            ),
            reply(text_part("Kenya ranks 3rd.")),
        ]
    )
    orchestrator = orchestrator_agent(
        build_orchestrator_tools(resolver=_Searches(), executor=_no_executor)
    )
    state: dict[str, Any] = {"surface": "studio"}

    def model() -> Any:
        return orchestrator_model("gemini-test", client=scripted.client, budget=None)

    first = [
        e
        async for e in run_orchestrator(
            orchestrator,
            history=None,
            state=state,
            turn_id="turn-1",
            model=model(),
            opening=Opening(question="Rank Kenya", today=TODAY),
        )
    ]
    (handed,) = [e for e in first if isinstance(e, HandedOff)]
    stored = handed.history.model_dump_json()
    assert len(scripted.requests) == 1
    assert handed.tool == "plan"
    assert state["orchestrator:pending"]["kind"] == "plan"
    assert json.loads(stored)["schema_version"] == 1

    second = [
        e
        async for e in run_orchestrator(
            orchestrator,
            history=StoredHistory.model_validate_json(stored),
            state=state,
            turn_id="turn-1",
            model=model(),
            hand_back=HandBack(answers="plan", result={"result": "Kenya is 3rd of 54."}),
        )
    ]

    (opening, *_) = second
    assert isinstance(opening, ToolsAnswered)
    assert [(a.name, a.response) for a in opening.answers] == [
        ("plan", {"result": "Kenya is 3rd of 54."})
    ]
    resumed = scripted.requests[1].content_list
    model_turn = resumed[-2].model_dump(mode="json", exclude_none=True)
    assert model_turn["parts"][0]["thought_signature"] == base64.b64encode(b"sig-plan").decode()
    assert resumed[-1].parts[0].function_response.response == {"result": "Kenya is 3rd of 54."}
    assert resumed[-1].parts[0].function_response.id is None
    assert second[-1] == Answered(text="Kenya ranks 3rd.")


async def test_a_stored_history_holds_no_copy_of_the_instruction_and_resumes_as_unstored() -> None:
    scripted = ScriptedGenai(
        [
            reply(_SEARCH),
            reply(call_part("plan", {"brief": "rank", "asks": ["rank"]})),
        ]
    )
    orchestrator = orchestrator_agent(
        build_orchestrator_tools(resolver=_Searches(), executor=_no_executor)
    )
    model = orchestrator_model("gemini-test", client=scripted.client, budget=None)
    (handed,) = [
        e
        async for e in run_orchestrator(
            orchestrator,
            state={"surface": "studio"},
            turn_id="turn-1",
            model=model,
            opening=Opening("Rank Kenya", TODAY),
        )
        if isinstance(e, HandedOff)
    ]
    held = handed.history.conversation.messages
    stored = handed.history.model_dump_json()
    loaded = StoredHistory.model_validate_json(stored)

    first_line = INSTRUCTION.splitlines()[0]
    assert sum(first_line in json.dumps(m.instructions) for m in held if m.kind == "request") == 2
    assert first_line not in stored
    assert ModelMessagesTypeAdapter.dump_json(
        loaded.conversation.messages
    ) == ModelMessagesTypeAdapter.dump_json(held)


async def test_a_second_hand_off_in_one_response_is_answered_refused() -> None:
    questions = [{"question": "Which Congo?", "options": [{"label": "DR Congo"}]}]
    scripted = ScriptedGenai(
        [
            reply(
                call_part("ask_user", {"questions": questions}),
                call_part("plan", {"brief": "rank", "asks": ["rank"]}),
            )
        ]
    )

    events, state = await _run(scripted)

    (answered,) = [e for e in events if isinstance(e, ToolsAnswered)]
    (refused,) = answered.answers
    assert refused.name == "plan"
    assert refused.response is not None
    assert refused.response["code"] == "hand_off_pending"
    handed = events[-1]
    assert isinstance(handed, HandedOff)
    assert handed.tool == "ask_user"
    assert state["orchestrator:pending"]["kind"] == "ask"
    assert len(scripted.requests) == 1


async def test_a_resumed_run_names_the_conversation_its_caller_gives(
    exported_spans: InMemorySpanExporter,
) -> None:
    """A stored history carries the conversation id its first run opened with;
    a resume given the turn's conversation id runs under that id."""
    scripted = ScriptedGenai(
        [reply(call_part("plan", {"brief": "rank", "asks": ["rank"]})), reply(text_part("Done."))]
    )
    orchestrator = orchestrator_agent(
        build_orchestrator_tools(resolver=_Searches(), executor=_no_executor)
    )
    state: dict[str, Any] = {"surface": "studio"}
    model = orchestrator_model("gemini-test", client=scripted.client, budget=None)
    (handed,) = [
        e
        async for e in run_orchestrator(
            orchestrator,
            state=state,
            turn_id="conv_1/1",
            model=model,
            opening=Opening("Rank Kenya", TODAY),
            conversation_id="conv_1",
        )
        if isinstance(e, HandedOff)
    ]
    stored = StoredHistory.model_validate_json(handed.history.model_dump_json())

    resumed = [
        e
        async for e in run_orchestrator(
            orchestrator,
            state=state,
            turn_id="conv_1/1",
            model=model,
            history=stored,
            hand_back=HandBack(answers="plan", result={"result": "Kenya is 3rd."}),
            conversation_id="conv_1",
        )
    ]

    assert resumed[-1] == Answered(text="Done.")
    runs = [s for s in exported_spans.get_finished_spans() if s.name.startswith("invoke_agent")]
    assert [(s.attributes or {}).get("gen_ai.conversation.id") for s in runs] == ["conv_1"] * 2


@pytest.mark.parametrize(
    "replies",
    [
        [empty_reply()],
        [reply(thought_part("Thinking."))],
        [reply(_SEARCH), empty_reply()],
        [reply(text_part("Looking it up."), _SEARCH), empty_reply()],
    ],
    ids=["no-parts", "thought-only", "after-a-call", "after-narration"],
)
async def test_a_response_with_no_answer_in_a_turn_that_showed_no_card_fails_transiently(
    replies: list[Any],
) -> None:
    """A response with neither answer text nor a function call is not retried: no
    further request is sent. A turn that showed the user no card, whatever calls it
    answered and whatever narration it streamed beside them, raises
    `EmptyResponseError`, a fault a later attempt may not meet."""
    scripted = ScriptedGenai(replies)

    with pytest.raises(EmptyResponseError) as raised:
        await _run(scripted, budget=_budget())

    assert transient(raised.value)
    assert len(scripted.requests) == len(replies)


async def test_a_resumed_run_whose_first_response_holds_no_answer_fails_transiently() -> None:
    """The hand-back that opens a resumed run is no answer the user sees: a resumed
    run whose response holds no answer fails, so the user's answer is not met with
    silence."""
    asked = {"questions": [{"question": "Which Congo?", "options": [{"label": "DR Congo"}]}]}
    scripted = ScriptedGenai([reply(call_part("ask_user", asked)), reply(thought_part("Hmm."))])
    orchestrator = orchestrator_agent(
        build_orchestrator_tools(resolver=_Searches(), executor=_no_executor)
    )
    state: dict[str, Any] = {"surface": "studio"}
    model = orchestrator_model("gemini-test", client=scripted.client, budget=None)
    (handed,) = [
        e
        async for e in run_orchestrator(
            orchestrator, state=state, turn_id="turn-1", model=model, opening=Opening("q", TODAY)
        )
        if isinstance(e, HandedOff)
    ]

    with pytest.raises(EmptyResponseError):
        async for _ in run_orchestrator(
            orchestrator,
            state=state,
            turn_id="turn-1",
            model=model,
            history=handed.history,
            hand_back=HandBack(answers="ask_user", result={"result": "DR Congo"}),
        ):
            pass


class _DrawingSearches(_Searches):
    """`_Searches` whose every search also records a card drawn this turn in `drawn`,
    the turn's drawn cards as `make_charts` keeps them."""

    def __init__(self, drawn: dict[str, Any]) -> None:
        super().__init__()
        self.drawn = drawn

    async def __call__(self, **kwargs: Any) -> dict[str, Any]:
        self.drawn[f"card-{self.calls}"] = {"component_id": f"cmp_{self.calls}", "revision": 1}
        return await super().__call__(**kwargs)


async def test_a_response_with_no_answer_after_a_card_drawn_this_run_ends_it_answered() -> None:
    """A response with no answer after the run drew a card ends it answered, with no
    text, and sends no further request: the card is the answer."""
    scripted = ScriptedGenai([reply(text_part("Looking it up."), _SEARCH), empty_reply()])
    drawn: dict[str, Any] = {}
    tools = build_orchestrator_tools(resolver=_DrawingSearches(drawn), executor=_no_executor)

    events, _ = await _run(
        scripted,
        budget=_budget(),
        tools=tools,
        held={_DRAWN_CARDS_KEY: {"turn": "turn-1", "drawn": drawn}},
    )

    assert events[-1] == Answered(text="")
    assert len(scripted.requests) == 2


async def test_a_resumed_run_ends_answered_on_a_card_drawn_before_the_hand_back() -> None:
    """A card drawn in the run before a hand-back is one the turn showed the user, so
    a resumed run whose response holds no answer ends answered."""
    asked = {"questions": [{"question": "Which Congo?", "options": [{"label": "DR Congo"}]}]}
    scripted = ScriptedGenai(
        [reply(_SEARCH), reply(call_part("ask_user", asked)), reply(thought_part("Hmm."))]
    )
    drawn: dict[str, Any] = {}
    orchestrator = orchestrator_agent(
        build_orchestrator_tools(resolver=_DrawingSearches(drawn), executor=_no_executor)
    )
    state: dict[str, Any] = {
        "surface": "studio",
        _DRAWN_CARDS_KEY: {"turn": "turn-1", "drawn": drawn},
    }
    model = orchestrator_model("gemini-test", client=scripted.client, budget=None)
    (handed,) = [
        e
        async for e in run_orchestrator(
            orchestrator, state=state, turn_id="turn-1", model=model, opening=Opening("q", TODAY)
        )
        if isinstance(e, HandedOff)
    ]

    resumed = [
        e
        async for e in run_orchestrator(
            orchestrator,
            state=state,
            turn_id="turn-1",
            model=model,
            history=handed.history,
            hand_back=HandBack(answers="ask_user", result={"result": "DR Congo"}),
        )
    ]

    assert resumed[-1] == Answered(text="")


async def test_a_hand_back_for_a_call_that_does_not_wait_is_refused() -> None:
    scripted = ScriptedGenai([reply(call_part("plan", {"brief": "rank", "asks": ["rank"]}))])
    orchestrator = orchestrator_agent(
        build_orchestrator_tools(resolver=_Searches(), executor=_no_executor)
    )
    state: dict[str, Any] = {"surface": "studio"}
    model = orchestrator_model("gemini-test", client=scripted.client, budget=None)
    (handed,) = [
        e
        async for e in run_orchestrator(
            orchestrator,
            history=None,
            state=state,
            turn_id="turn-1",
            model=model,
            opening=Opening("q", TODAY),
        )
        if isinstance(e, HandedOff)
    ]

    with pytest.raises(NoCallWaitsError):
        async for _ in run_orchestrator(
            orchestrator,
            history=handed.history,
            state=state,
            turn_id="turn-1",
            model=model,
            hand_back=HandBack(answers="ask_user", result={"result": "x"}),
        ):
            pass


async def test_a_seed_hand_back_follows_the_question_on_every_request() -> None:
    """As the recorded requests place it: digests, question, seed, then the run."""
    scripted = ScriptedGenai([reply(_SEARCH), reply(text_part("Done."))])

    await _run(scripted, seeded="The user brought Life expectancy from Search.")

    for request in scripted.requests:
        texts = [content.parts[0].text for content in request.content_list[:3]]
        assert texts[0].startswith("Today's date (UTC): 2026-11-17")
        assert texts[1:] == [
            "Life expectancy in Kenya?",
            "The user brought Life expectancy from Search.",
        ]


# --- the prefix cache --------------------------------------------------------


async def test_every_call_reads_its_prefix_from_the_shared_cache() -> None:
    scripted = ScriptedGenai([reply(_SEARCH), reply(text_part("Done."))])
    created: list[Any] = []

    async def create(*, model: str, config: Any) -> Any:
        created.append(config)
        return types.CachedContent(name="cachedContents/c1")

    scripted.client.aio.caches.create = create  # type: ignore[method-assign]

    await _run(scripted, prefix_cache=PrefixCache())

    assert len(created) == 1
    assert [r.generate_config.cached_content for r in scripted.requests] == [
        "cachedContents/c1"
    ] * 2
    assert all(r.generate_config.system_instruction is None for r in scripted.requests)
    assert all(r.generate_config.tools is None for r in scripted.requests)


async def test_a_stop_mid_answer_cancels_the_model_stream() -> None:
    scripted = ScriptedGenai([])
    closed = asyncio.Event()
    first = asyncio.Event()

    async def stream(*, model: str, contents: Any, config: Any = None) -> Any:
        async def chunks() -> AsyncGenerator[Any]:
            try:
                yield _chunk(text_part("Life expectancy"))
                await asyncio.Event().wait()
            finally:
                closed.set()

        return chunks()

    scripted.client.aio.models.generate_content_stream = stream  # type: ignore[method-assign]

    async def consume() -> None:
        async for event in run_orchestrator(
            orchestrator_agent(
                build_orchestrator_tools(resolver=_Searches(), executor=_no_executor)
            ),
            history=None,
            state={"surface": "studio"},
            turn_id="t",
            model=orchestrator_model("gemini-test", client=scripted.client, budget=_budget()),
            opening=Opening(question="q", today=TODAY),
        ):
            if isinstance(event, TextDelta):
                first.set()

    task = asyncio.create_task(consume())
    await asyncio.wait_for(first.wait(), timeout=2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    await asyncio.wait_for(closed.wait(), timeout=1)


# --- a turn that only takes cards off ------------------------------------------


def _canvas_of_two(*, pinned: str | None = None) -> ConversationState:
    """A canvas with two charts, "Aid to Kenya" (`cmp_01`) and "Aid to Ghana"
    (`cmp_02`), with `pinned` pinned."""
    canvas = ConversationState(conversation_id="c1")
    for cid, title in (("cmp_01", "Aid to Kenya"), ("cmp_02", "Aid to Ghana")):
        land_component(
            canvas,
            {
                "component_id": cid,
                "revision": 1,
                "type": "line_chart",
                "payload": {"component": {"title": {"text": title}}},
            },
        )
    assert canvas.composition is not None
    for block in canvas.composition.blocks:
        block.pinned = block.component_id == pinned
    return canvas


def _removing_tools(canvas: ConversationState) -> Any:
    async def remove(*, component_ids: list[str], all_cards: bool) -> CardRemoval:
        return remove_cards(canvas, component_ids, all_cards=all_cards)

    return build_orchestrator_tools(
        resolver=_Searches(), executor=_no_executor, remove_cards=remove
    )


_REMOVE_FIRST = call_part("remove_cards", {"component_ids": ["cmp_01"]})


@pytest.mark.parametrize(
    ("pinned", "names", "reply_text"),
    [
        (None, ["cmp_01"], 'I took "Aid to Kenya" off the canvas.'),
        (None, ["cmp_01", "cmp_02"], 'I took "Aid to Kenya" and "Aid to Ghana" off the canvas.'),
        (
            "cmp_02",
            ["cmp_01", "cmp_02"],
            'I took "Aid to Kenya" off the canvas. "Aid to Ghana" is pinned, so it stays. '
            "To remove it, choose Unpin from Canvas in its menu, then ask again.",
        ),
        (
            "cmp_01",
            ["cmp_01"],
            '"Aid to Kenya" is pinned, so it stays. '
            "To remove it, choose Unpin from Canvas in its menu, then ask again.",
        ),
    ],
)
async def test_a_turn_whose_only_call_takes_cards_off_ends_on_the_harness_reply(
    pinned: str | None, names: list[str], reply_text: str
) -> None:
    """The model's one response makes a lone `remove_cards` call and no text.
    The harness writes the reply from the call's answer, naming each card
    taken off and each pinned card kept by title, and the run ends without a
    second model call."""
    canvas = _canvas_of_two(pinned=pinned)
    scripted = ScriptedGenai(
        [
            reply(thought_part("Remove it."), call_part("remove_cards", {"component_ids": names})),
            reply(text_part("Removed.")),
        ]
    )

    events, _ = await _run(scripted, tools=_removing_tools(canvas))

    assert len(scripted.requests) == 1
    assert _texts(events) == [reply_text]
    assert events[-1] == Answered(text=reply_text)
    assert isinstance(events[-3], ToolsAnswered)


@pytest.mark.parametrize(
    "first",
    [
        pytest.param([text_part("Taking it off."), _REMOVE_FIRST], id="text-beside-the-call"),
        pytest.param([_SEARCH, _REMOVE_FIRST], id="another-call-beside-it"),
        pytest.param(
            [call_part("remove_cards", {"component_ids": ["cmp_01", "cmp_01"]}), _SEARCH],
            id="removal-first-then-another-call",
        ),
    ],
)
async def test_a_removal_beside_text_or_another_call_keeps_the_model_reply(
    first: list[types.Part],
) -> None:
    canvas = _canvas_of_two()
    scripted = ScriptedGenai([reply(*first), reply(text_part("Done."))])

    events, _ = await _run(scripted, tools=_removing_tools(canvas))

    assert len(scripted.requests) == 2
    assert events[-1] == Answered(text="Done.")


async def test_a_removal_after_another_call_keeps_the_model_reply() -> None:
    canvas = _canvas_of_two()
    scripted = ScriptedGenai([reply(_SEARCH), reply(_REMOVE_FIRST), reply(text_part("Done."))])

    events, _ = await _run(scripted, tools=_removing_tools(canvas))

    assert len(scripted.requests) == 3
    assert events[-1] == Answered(text="Done.")


@pytest.mark.parametrize(
    "args",
    [
        pytest.param({"component_ids": ["cmp_99"]}, id="refused"),
        pytest.param({"component_ids": ["cmp_01", "cmp_99"]}, id="an-id-never-held"),
        pytest.param({"component_ids": ["cmp_03"]}, id="already-off"),
    ],
)
async def test_a_removal_the_reply_cannot_name_whole_keeps_the_model_reply(
    args: dict[str, Any],
) -> None:
    """A refused call, or one whose answer holds more than cards taken off and
    pinned cards kept, goes back to the model."""
    canvas = _canvas_of_two()
    land_component(
        canvas,
        {"component_id": "cmp_03", "revision": 1, "type": "line_chart"},
    )
    remove_cards(canvas, ["cmp_03"], all_cards=False)
    scripted = ScriptedGenai([reply(call_part("remove_cards", args)), reply(text_part("Done."))])

    events, _ = await _run(scripted, tools=_removing_tools(canvas))

    assert len(scripted.requests) == 2
    assert events[-1] == Answered(text="Done.")


async def test_a_removal_reply_ends_the_agent_run_before_it_is_yielded() -> None:
    """The graph's `orchestrate` node stops reading at `Answered`, and the
    generator is closed later from another task. The agent's run has closed
    by then, so closing the generator resets no context variable of it."""
    canvas = _canvas_of_two()
    scripted = ScriptedGenai([reply(_REMOVE_FIRST), reply(text_part("Removed."))])
    run = run_orchestrator(
        orchestrator_agent(_removing_tools(canvas)),
        history=None,
        state={"surface": "studio"},
        turn_id="turn-1",
        model=orchestrator_model("gemini-test", client=scripted.client, budget=_budget()),
        opening=Opening(question="Remove the first chart.", today=TODAY),
    )
    async for event in run:
        if isinstance(event, Answered):
            break

    await asyncio.create_task(run.aclose())

    assert event == Answered(text='I took "Aid to Kenya" off the canvas.')
