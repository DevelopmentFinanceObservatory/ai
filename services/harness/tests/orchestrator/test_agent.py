"""The orchestrator agent (`orchestrator.agent`): its instruction, thinking and
the request it opens a turn with, through a scripted google-genai client."""

from __future__ import annotations

import hashlib
import json
from datetime import date
from typing import Any

import pytest
from google.genai import types
from harness import turn_records
from harness.gemini_wire import model_retry_options
from harness.orchestrator_tools import MAX_ASK_QUESTIONS, build_orchestrator_tools
from harness.prefix_cache import PrefixCache
from orchestrator.agent import INSTRUCTION, orchestrator_agent
from orchestrator.model import orchestrator_model
from orchestrator.run import Opening, run_orchestrator

from ..scripted_genai import ScriptedGenai, reply, text_part


async def _unused(**kwargs: Any) -> Any:
    raise AssertionError("no call expected")


async def _first_request(
    *,
    thinking_level: Any = None,
    canvas_digest: str | None = None,
    conversation_digest: str | None = None,
    prefix_cache: PrefixCache | None = None,
) -> Any:
    scripted = ScriptedGenai([reply(text_part("done"))])

    async def create(*, model: str, config: Any) -> Any:
        return types.CachedContent(name="cachedContents/c1")

    scripted.client.aio.caches.create = create  # type: ignore[method-assign]
    tools = build_orchestrator_tools(resolver=_unused, executor=_unused)
    orchestrator = (
        orchestrator_agent(tools)
        if thinking_level is None
        else orchestrator_agent(tools, thinking_level=thinking_level)
    )
    async for _ in run_orchestrator(
        orchestrator,
        history=None,
        state={"surface": "studio"},
        turn_id="t",
        model=orchestrator_model(
            "gemini-test", client=scripted.client, budget=None, prefix_cache=prefix_cache
        ),
        opening=Opening(
            question="and Ghana?",
            today=date(2026, 10, 5),
            canvas_digest=canvas_digest,
            conversation_digest=conversation_digest,
        ),
    ):
        pass
    return scripted.requests[0]


@pytest.mark.parametrize("cached", [False, True], ids=["whole", "cached"])
async def test_every_request_carries_the_model_retries(cached: bool) -> None:
    request = await _first_request(prefix_cache=PrefixCache() if cached else None)

    assert (request.generate_config.cached_content is not None) == cached
    http_options = request.generate_config.http_options
    assert http_options is not None
    assert http_options.retry_options == model_retry_options()


@pytest.mark.parametrize(
    ("level", "expected"),
    [
        ("LOW", types.ThinkingLevel.LOW),
        ("MEDIUM", types.ThinkingLevel.MEDIUM),
        ("HIGH", types.ThinkingLevel.HIGH),
    ],
)
async def test_the_thinking_level_reaches_the_request(level: str, expected: Any) -> None:
    thinking = (await _first_request(thinking_level=level)).generate_config.thinking_config

    assert thinking.thinking_level == expected
    assert thinking.include_thoughts is True


async def test_an_agent_built_without_a_thinking_level_runs_at_low() -> None:
    thinking = (await _first_request()).generate_config.thinking_config

    assert thinking.thinking_level == types.ThinkingLevel.LOW


def _prefix_hash(request: Any) -> str:
    config = request.generate_config
    prefix = [
        config.system_instruction,
        [tool.model_dump(mode="json") for tool in config.tools or []],
    ]
    return hashlib.sha256(json.dumps(prefix, sort_keys=True).encode()).hexdigest()


async def test_the_static_prefix_is_the_same_whatever_the_canvas() -> None:
    kenya = json.dumps({"blocks": [{"title": "Debt service, Kenya"}]})
    ghana = json.dumps({"blocks": [{"title": "Debt service, Ghana"}]})
    history = json.dumps({"turns": [{"user": "Kenya's debt service"}]})

    first = await _first_request(canvas_digest=kenya)
    second = await _first_request(canvas_digest=ghana, conversation_digest=history)

    assert _prefix_hash(first) == _prefix_hash(second)
    assert first.generate_config.automatic_continuation is True
    assert first.content_list[0].parts[0].text == (
        f"Today's date (UTC): 2026-10-05\n\nThe user's canvas: {kenya}"
    )
    assert second.content_list[0].parts[0].text == (
        f"Today's date (UTC): 2026-10-05\n\nThe user's canvas: {ghana}"
        f"\n\nEarlier turns in this conversation: {history}"
    )
    assert [content.parts[0].text for content in first.content_list[1:]] == ["and Ghana?"]


async def test_the_model_reads_the_instruction_as_written() -> None:
    """No identity line or other text is added to the instruction; Pydantic AI
    strips its trailing newline."""
    request = await _first_request()

    assert request.generate_config.system_instruction == INSTRUCTION.rstrip("\n")


async def test_a_turn_with_no_digest_opens_on_the_date_and_the_question() -> None:
    request = await _first_request()

    assert [content.parts[0].text for content in request.content_list] == [
        "Today's date (UTC): 2026-10-05",
        "and Ghana?",
    ]


def test_the_instruction_says_what_the_canvas_digests_focus_card_is():
    """`focus_component_id` is how a card's "Ask follow up" reaches the
    model; unexplained, the model may answer about every card."""
    assert "focus_component_id" in INSTRUCTION


def test_the_instruction_says_when_cards_come_off_and_that_pinned_ones_stay():
    """Cards accumulate across turns. The model takes them off when the user
    asks or the conversation changes subject, and pinned cards stay."""
    text = " ".join(INSTRUCTION.split())
    assert "each turn's cards join the ones already on the canvas" in text
    assert "when the conversation turns to another subject" in text
    assert "remove_cards leaves pinned cards on the canvas and names them as kept_pinned" in text
    assert "A pinned card comes off once the user unpins it from the card's menu" in text
    # The ids of the earlier subject's cards come from the canvas listing.
    assert (
        "or turns to another subject so the earlier subject's cards come off, call recall "
        "with part canvas first to see them and their ids" in text
    )


def test_the_instruction_keeps_a_measures_creditor_scope_in_the_reply():
    text = " ".join(INSTRUCTION.split())
    assert 'debt service "owed to all creditors" is reported in those words' in text


def test_the_instruction_lets_a_reply_about_dfo_state_the_figures_explain_dfo_gives():
    assert "stating only the figures it gives" in INSTRUCTION
    text = " ".join(INSTRUCTION.split())
    assert "Do not state a figure in a reply to a greeting or small talk" in text


def test_the_instruction_has_a_reply_state_only_the_figures_a_drawn_card_returns():
    """The reply states a drawn card's `reply_figures` and nothing else of
    it, the chart names the series the user asked a value of, an empty list
    gets what the card holds, and the conclusion comes from `holds`."""
    text = " ".join(INSTRUCTION.split())
    assert "state only the values its reply_figures list, each with its period" in text
    assert "Name in state_values_for only a series whose value the user asked to be told" in text
    assert (
        "Calling make_charts again for a card already drawn returns its figures for the "
        "series it names, without drawing it twice" in text
    )
    assert "say what the card holds, its measure, places and years, and state no value" in text
    assert "Close with one conclusion drawn from holds" in text
    assert "compute nothing yourself" in text
    assert "state its last value and period from the card's facts" not in text


def test_the_instruction_tells_the_agent_to_state_the_years_an_event_period_reads_as():
    text = " ".join(INSTRUCTION.split())
    assert "names a period by an event or an administration" in text
    assert "say the data ends earlier and answer with what it holds" in text
    assert "ask about candidates or prices search left open in one more ask_user call" in text
    assert "the reading search chose" not in text


_NUMBER_WORDS = {2: "two", 3: "three", 4: "four"}


def test_the_instruction_has_search_elect_the_measure_and_asks_only_what_search_left_open():
    text = " ".join(INSTRUCTION.split())
    assert (
        "Put a word for how a flow is counted into `variable` only when the user said it, "
        "and let search elect the rest" in text
    )
    assert (
        "A count search chose (`counted_as`) is settled too, even when it lists other counts "
        "under `alternatives`: go on with it and do not ask about the count" in text
    )
    # The harness states the count basis after the reply (`turn_records.basis_line`)
    # and offers the other options and the other counts as chips
    # (`turn_records.scope_chips`).
    assert f'"{turn_records._BASIS_LEAD}...", naming the sector, region, provider' in text
    assert (
        "and the other counts a part lists under `alternatives`, as buttons the user can "
        "switch with, so leave those out of the reply" in text
    )
    assert "No button offers" not in text
    assert "name the source with that count" not in text
    assert "with the count search chose" not in text
    assert "switch to another count" not in text
    assert "Search leaves a count open only when a part lists `candidates`" in text
    assert (
        "ask the unit with ask_user in one call: one question per measure, with `unit_of` "
        "set to a part's spec_id, and `prices` set when a skill prefers current or constant. "
        "Parts that measure one thing over other years share one unit question, and one call "
        f"holds at most {_NUMBER_WORDS[MAX_ASK_QUESTIONS]} questions" in text
    )
    assert "one question per part" not in text
    assert "count question" not in text
    assert "a count or prices still open" not in text
    assert "even when the result lists no `other_prices`" not in text
    assert "The answer to a unit question names the spec_id to fetch" in text
    assert "search again with that unit on the part" not in text
    assert (
        "A request that adds or combines measures from different sources, such as aid and debt, "
        "goes to plan straight after search, without ask_user. The planner asks how to combine "
        "them" in text
    )
    assert (
        "in the user's words, except a part's candidates or open prices, whose options are set "
        "out in their own paragraphs" in text
    )
    assert "takes one question a turn" not in text
    assert "gross or net" not in text
