"""The agent's turn and its question to the user, on the turn loop over
Postgres: an answered turn, the `ask` node's pause, and what the agent reads
once the user answers.

The real `seed`, `orchestrate` and `ask` nodes run in a test-only graph that
ends where they end (`tests.graph_turns.opening_graph`), the orchestrator on
its real tools over a scripted Gemini client.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest
from data_mcp.tool_errors import ToolErrorCode
from google.genai import types
from harness.graph.orchestrate import orchestrate_node
from harness.graph.state import (
    ORCHESTRATOR_SEARCHES_KEY,
)
from harness.turn.loop import Outcome
from harness.turn.node import NodeCtx, StopScope, TurnKey
from harness.turn.transcript import Transcript
from harness.turn_records import SearchReading, WindowReading
from orchestrator.run import ToolsAnswered
from sqlalchemy.ext.asyncio import AsyncEngine

from .graph_turns import agent_read, given, opening_graph, opening_nodes, visited
from .scripted_genai import call_part, reply, text_part
from .scripted_tools import scripted_orchestrator
from .turn_driver import ScriptedModels, SegmentRun, TurnDriver, scripted_models


async def _turn(
    engine: AsyncEngine,
    message: str,
    *script: types.Part,
    specs: Mapping[str, Any] | None = None,
    state: Mapping[str, Any] | None = None,
) -> tuple[TurnDriver, SegmentRun, ScriptedModels]:
    """Run turn `message` to its first pause or its end, the agent answering
    each call with the next part of `script`, after a `given` node hands the
    turn `specs` and `state`."""
    models = scripted_models(orchestrator=[reply(part) for part in script])
    nodes = opening_nodes(scripted_orchestrator(), models)
    before = given(specs=specs, state=state) if specs or state else None
    driver = TurnDriver(engine, graph=opening_graph(nodes, before=before))
    run = await driver.turn(message)
    return driver, run, models


def _ask_user(questions: Sequence[Mapping[str, Any]]) -> types.Part:
    return call_part("ask_user", {"questions": list(questions)})


async def test_an_answered_turn_ends_after_the_agent_without_a_pause(
    pg_engine: AsyncEngine,
) -> None:
    _, run, models = await _turn(
        pg_engine, "Kenya's GDP in 2020?", text_part("Kenya's GDP was $101 billion in 2020.")
    )

    assert agent_read(models.orchestrator.requests[0]).count("Kenya's GDP in 2020?") == 1
    assert run.outcome is Outcome.DONE
    assert visited(run.events) == ["seed", "orchestrate"]


_CONGO = {
    "question": "Which Congo?",
    "options": [{"label": "Republic of the Congo"}, {"label": "DR Congo"}],
}
_SETTLED = "Each answered choice is settled. Do not ask it again."


def _answers(**answers: dict[str, Any]) -> dict[str, Any]:
    """The `ask` pause's answer: one answer per question id."""
    return {"answers": answers}


async def test_ask_pauses_with_generated_option_ids_and_hands_the_choice_back_in_words(
    pg_engine: AsyncEngine,
) -> None:
    driver, paused, models = await _turn(
        pg_engine,
        "GDP of Congo?",
        _ask_user([_CONGO]),
        text_part("The Republic of the Congo's GDP was $15.7 billion in 2024."),
    )

    pause = paused.pause
    assert pause.kind == "ask"
    assert pause.payload == {
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
    resumed_run = await driver.answer(_answers(q1={"kind": "option", "spec_id": "o1"}), to=pause)

    resumed = agent_read(models.orchestrator.requests[-1])
    assert "GDP of Congo?" in resumed
    assert (
        f"You asked the user: Which Congo?\nThe user chose: Republic of the Congo\n{_SETTLED}"
        in resumed
    )
    assert resumed_run.outcome is Outcome.DONE


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
    pg_engine: AsyncEngine, answer: dict[str, Any], expected: list[str]
) -> None:
    block = {
        "spec_id": _GROSS,
        "resolved": {
            "measure_label": "gross disbursements",
            "alternative_measures": [
                {"label": "net disbursements", "axis": "what", "definition": {"spec_id": _NET}}
            ],
        },
    }
    driver, paused, models = await _turn(
        pg_engine,
        "Aid from France to Kenya?",
        _ask_user(
            [
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
            ]
        ),
        text_part("Done."),
        specs={_GROSS: block},
    )

    pause = paused.pause
    assert pause.message == "How should aid be counted?"
    assert pause.payload["questions"] == [
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
    resumed = await driver.answer(answer, to=pause)

    assert "\n".join(["You asked the user:", *expected]) in agent_read(
        models.orchestrator.requests[-1]
    )
    assert resumed.outcome is Outcome.DONE


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


async def test_a_prices_question_takes_its_options_from_its_part_s_open_prices(
    pg_engine: AsyncEngine,
) -> None:
    """Two parts whose prices are open, worded alike. Each prices question
    names its part, and code gives it that part's `open_prices` entries,
    marked with the part and their prices. A pick of constant prices no
    search listed asks for a search of that part."""
    driver, paused, models = await _turn(
        pg_engine,
        "Aid and GDP, 2018-2022?",
        _ask_user(
            [
                {"question": "Aid in which prices?", "prices_of": _GROSS},
                {"question": "GDP in which prices?", "prices_of": _OTHER},
            ]
        ),
        text_part("Done."),
        specs={spec: _usd_block(spec) for spec in (_GROSS, _OTHER)},
        state={ORCHESTRATOR_SEARCHES_KEY: [_open_reading(spec) for spec in (_GROSS, _OTHER)]},
    )

    pause = paused.pause
    assert [q["candidates"] for q in pause.payload["questions"]] == [
        [
            {"spec_id": "o1", "label": "current prices", "part": part, "prices": "current"},
            {"spec_id": "o2", "label": "constant prices", "part": part, "prices": "constant"},
        ]
        for part in (_GROSS, _OTHER)
    ]
    await driver.answer(
        _answers(q1={"kind": "option", "spec_id": "o1"}, q2={"kind": "option", "spec_id": "o2"}),
        to=pause,
    )

    asked = agent_read(models.orchestrator.requests[-1])
    assert (
        f"The user chose: constant prices, which no search listed. Search once for that part "
        f"({_OTHER}) in constant prices." in asked
    )
    assert f"({_GROSS}) in constant prices" not in asked


def _ask_user_answer(run: SegmentRun) -> dict[str, Any]:
    """The one `ask_user` answer among the turn's events."""
    (answer,) = [
        response.response
        for event in run.events
        if isinstance(event, ToolsAnswered)
        for response in event.answers
        if response.name == "ask_user"
    ]
    assert answer is not None
    return answer


async def test_two_prices_questions_on_one_part_are_refused(pg_engine: AsyncEngine) -> None:
    """Two questions pricing one part would let the later answer override the
    earlier one in silence, so `ask_user` refuses the call and nothing pauses."""
    _, run, _ = await _turn(
        pg_engine,
        "Aid, 2018-2022?",
        _ask_user(
            [
                {"question": "Aid in which prices?", "prices_of": _GROSS},
                {"question": "And in which prices again?", "prices_of": _GROSS},
            ]
        ),
        text_part("Done."),
        specs={_GROSS: _usd_block(_GROSS)},
        state={ORCHESTRATOR_SEARCHES_KEY: [_open_reading(_GROSS)]},
    )

    assert run.outcome is Outcome.DONE
    refused = _ask_user_answer(run)
    assert refused["code"] == ToolErrorCode.REQUEST_INVALID.value
    assert f"{_GROSS} is priced by more than one question" in refused["refusal"]


@pytest.mark.parametrize("same_measure", [True, False], ids=["two-windows", "two-counts"])
async def test_two_windows_of_one_measure_take_one_prices_question(
    pg_engine: AsyncEngine, same_measure: bool
) -> None:
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
    _, run, _ = await _turn(
        pg_engine,
        "Aid to 2014 against since 2018?",
        _ask_user(
            [
                {"question": "Aid to 2014 in which prices?", "prices_of": _OTHER},
                {"question": "Aid since 2018 in which prices?", "prices_of": _FK_GROSS},
            ]
        ),
        text_part("Done."),
        specs={_FK_GROSS: _FRANCE_KENYA, _OTHER: other},
        state={ORCHESTRATOR_SEARCHES_KEY: [_open_reading(_FK_GROSS), _open_reading(_OTHER)]},
    )

    assert (run.outcome is not Outcome.PAUSED) is same_measure
    if same_measure:
        refused = _ask_user_answer(run)
        assert refused["code"] == ToolErrorCode.REQUEST_INVALID.value
        assert "measure one thing over other years" in refused["refusal"]


@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        (
            _answers(q1={"kind": "text", "text": "the big one"}),
            "The user answered in their own words: the big one",
        ),
        ({"skip": True}, "The user skipped the question. Do not answer it"),
    ],
    ids=["answers_text", "dismissed"],
)
async def test_a_free_text_answer_and_a_skip_are_handed_back_as_such(
    pg_engine: AsyncEngine, answer: dict[str, Any], expected: str
) -> None:
    driver, paused, models = await _turn(
        pg_engine, "GDP of Congo?", _ask_user([_CONGO]), text_part("Done.")
    )

    await driver.answer(answer, to=paused.pause)

    assert expected in agent_read(models.orchestrator.requests[-1])


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
        (_answers(q1={"kind": "option", "spec_id": "o2"}), "The user chose: DR Congo"),
        (
            _answers(q1={"kind": "text", "text": "the big one"}),
            "The user answered in their own words: the big one",
        ),
        ({"skip": True}, "The user skipped the question. Do not answer it"),
    ],
    ids=["pick", "text", "dismissed"],
)
async def test_an_ask_whose_options_the_model_sent_as_bare_labels_resumes(
    pg_engine: AsyncEngine, options: list[Any], answer: dict[str, Any], expected: str
) -> None:
    """Gemini does not hold a call to its declaration, so an `ask_user`
    option can arrive as its bare label; the question pauses and resumes as
    one whose options are objects."""
    driver, paused, models = await _turn(
        pg_engine,
        "GDP of Congo?",
        _ask_user([{"question": "Which Congo?", "options": options}]),
        text_part("Done."),
    )

    resumed = await driver.answer(answer, to=paused.pause)

    assert resumed.outcome is Outcome.DONE
    assert f"You asked the user: Which Congo?\n{expected}" in agent_read(
        models.orchestrator.requests[-1]
    )


@pytest.mark.parametrize("answers", ["plan", "ask_user"])
async def test_an_answer_no_call_waits_on_is_refused_and_not_read_as_a_seed(answers: str) -> None:
    """A hand-back that answers a call finds no conversation of the agent's
    waiting on it: `orchestrate` raises, and records no seed hand-back."""
    models = scripted_models()
    node = orchestrate_node(orchestrator=scripted_orchestrator(), model=models.orchestrator_model)
    state: dict[str, Any] = {}
    ctx = NodeCtx(
        key=TurnKey("user-1", "conv", 1),
        inputs={"message": "Kenya's debt service?"},
        state=state,
        transcript=Transcript(),
        emit=lambda event: None,
        stop=StopScope(),
    )
    hand_back = {"answers": answers, "status": "Reading", "text": "The approved plan ran."}

    with pytest.raises(RuntimeError, match=f"{answers} hand-back"):
        await node.run(ctx, hand_back)
    assert state == {}
    assert models.orchestrator.requests == []
