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
from compiler.surface import ProgressCallback, ToolCallResult, ToolId
from data_mcp.search_result import SEARCH_ENVELOPE_META_KEY
from data_mcp.tool_errors import ToolErrorCode
from google.genai import types
from harness.graph.orchestrate import answer_replies, ask_node, orchestrate_node
from harness.graph.state import (
    ORCHESTRATOR_SEARCHES_KEY,
    ORCHESTRATOR_SPECS_KEY,
    AskQuestion,
    ask_questions,
    open_parts,
    prices_forms,
    settle_unit,
    unit_reads,
    unit_settled,
)
from harness.turn.loop import Outcome
from harness.turn.node import NodeCtx, StopScope, TurnKey
from harness.turn.transcript import Transcript
from harness.turn_records import CandidateReading, SearchReading, WindowReading
from orchestrator.run import ToolsAnswered
from planner.call import _picks
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
    definition_surface: Any = None,
) -> tuple[TurnDriver, SegmentRun, ScriptedModels]:
    """Run turn `message` to its first pause or its end, the agent answering
    each call with the next part of `script`, after a `given` node hands the
    turn `specs` and `state`. A unit answer re-reads its parts through
    `definition_surface`."""
    models = scripted_models(orchestrator=[reply(part) for part in script])
    nodes = opening_nodes(scripted_orchestrator(), models, definition_surface=definition_surface)
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
# The recorded France-Kenya search predates `price_bases`; its tests that need
# open prices carry this hand-written pair.
_BOTH_BASES = {
    "price_bases": [
        {"prices": "current", "via": "served"},
        {"prices": "constant", "via": "unit_request"},
    ],
    "unit_conversion": {
        "currencies": [
            {
                "code": "USD",
                "noun": "US dollars",
                "current_years": {"start_year": 2000, "end_year": 2023},
                "constant_years": {"start_year": 2000, "end_year": 2023},
                "base_year_min": 2000,
                "base_year_max": 2023,
            }
        ],
        "default_base_year": 2023,
    },
}

_FRANCE_KENYA = json.loads(
    (
        Path(__file__).parents[2]
        / "data-mcp"
        / "tests"
        / "fixtures"
        / "resolve_aid_france_kenya.json"
    ).read_text()
)["parts"][0]["spec"]
_FRANCE_KENYA["resolved"].update(_BOTH_BASES)
_FK_GROSS = _FRANCE_KENYA["spec_id"]
# The net count the France-Kenya block offers (`resolved.alternative_measures`).
_FK_NET = "spec_c67524ce5da3bfd4d94994a76c560b28"


def _open_reading(
    spec_id: str, *, text: str | None = None, other_prices: Sequence[Mapping[str, str]] = ()
) -> dict[str, Any]:
    """A `definite` reading of `spec_id` over 2018 to 2022, as `store_search` keeps it."""
    return SearchReading(
        text=text,
        status="definite",
        spec_id=spec_id,
        window=WindowReading(start=2018, end=2022, source="requested"),
        other_prices=tuple(CandidateReading.model_validate(offer) for offer in other_prices),
    ).model_dump(mode="json")


_USD_BLOCK_UNIT = {
    "basis": "fallback",
    "ref": "CurrencyUSD",
    "label": "US Dollar",
    "currency": "USD",
}


def _usd_block(spec_id: str) -> dict[str, Any]:
    """A spec block in current US dollars, so its prices are open over several years."""
    return {
        "spec_id": spec_id,
        "resolved": {
            "facets": {"unit": _USD_BLOCK_UNIT},
            **_BOTH_BASES,
        },
    }


_OTHER = "spec_" + "b" * 32


_CONSTANT = "spec_" + "d" * 32
# The offer of a part whose Search converts it into US dollars, current or
# constant at a base year from 2000 to 2023 (`_BOTH_BASES`).
_USD_OFFER = {
    "currencies": [
        {
            "code": "USD",
            "name": "US dollars",
            "prices": ["current", "constant"],
            "base_year_min": 2000,
            "base_year_max": 2023,
        }
    ],
    "presets": ["USD"],
    "start": {"currency": "USD", "prices": "constant", "base_year": 2023},
}


async def test_a_part_converted_takes_a_unit_question_and_one_with_a_constant_reading_two_options(
    pg_engine: AsyncEngine,
) -> None:
    """Two parts whose prices are open. Search serves aid in constant prices
    only as another reading, so its question offers that reading and the
    part's own. Search converts GDP, so its question offers units, and a unit
    the part is served in needs no new search."""
    aid_block = {
        "spec_id": _GROSS,
        "resolved": {
            "facets": {"unit": _USD_BLOCK_UNIT},
            "price_bases": [
                {"prices": "current", "via": "served"},
                {"prices": "constant", "via": "alternative"},
            ],
        },
    }
    driver, paused, models = await _turn(
        pg_engine,
        "Aid and GDP, 2018-2022?",
        _ask_user(
            [
                {"question": "Aid in which prices?", "unit_of": _GROSS},
                {"question": "GDP in which prices?", "unit_of": _OTHER},
            ]
        ),
        text_part("Done."),
        specs={_GROSS: aid_block, _CONSTANT: {"spec_id": _CONSTANT}, _OTHER: _usd_block(_OTHER)},
        state={
            ORCHESTRATOR_SEARCHES_KEY: [
                _open_reading(
                    _GROSS, other_prices=[{"spec_id": _CONSTANT, "label": "constant prices"}]
                ),
                _open_reading(_OTHER, text="GDP"),
            ]
        },
    )

    pause = paused.pause
    assert pause.payload["questions"] == [
        {
            "id": "q1",
            "question": "Aid in which prices?",
            "candidates": [
                {"spec_id": "o1", "label": "current prices", "part": _GROSS, "prices": "current"},
                {
                    "spec_id": _CONSTANT,
                    "label": "constant prices",
                    "part": _GROSS,
                    "prices": "constant",
                },
            ],
        },
        {
            "id": "q2",
            "question": "GDP in which prices?",
            "candidates": [],
            "unit": {"parts": [_OTHER], "offer": _USD_OFFER},
        },
    ]
    await driver.answer(
        _answers(
            q1={"kind": "option", "spec_id": _CONSTANT},
            q2={"kind": "unit", "currency": "USD", "prices": "current"},
        ),
        to=pause,
    )

    asked = agent_read(models.orchestrator.requests[-1])
    assert f"1. Aid in which prices? The user chose: constant prices ({_CONSTANT})" in asked
    assert (
        "2. GDP in which prices? The user chose current US dollars. "
        f"GDP ({_OTHER}) is served in current US dollars: fetch it as it stands." in asked
    )


class _UnitSearch:
    """data-mcp's `resolve_definition` for a unit answer: each identity posted
    is answered with the France-Kenya block under the posted spec_id and
    identity, or refused with `refusal`."""

    def __init__(self, *, refusal: dict[str, object] | None = None) -> None:
        self.refusal = refusal
        self.posted: list[dict[str, Any]] = []

    async def call_tool(
        self, *, tool: ToolId, arguments: Any, on_progress: ProgressCallback | None = None
    ) -> ToolCallResult:
        del tool, on_progress
        self.posted.append(dict(arguments))
        if self.refusal is not None:
            return ToolCallResult(is_error=True, structured_content=self.refusal, text="refused")
        spec = copy.deepcopy(_FRANCE_KENYA)
        spec["spec_id"] = arguments["spec_id"]
        spec["definition"] = {
            **spec["definition"],
            "spec_id": arguments["spec_id"],
            "identity": arguments["identity"],
        }
        part = {"status": "definite", "variable_text": arguments["spec_id"], "spec": spec}
        return ToolCallResult(is_error=False, meta={SEARCH_ENVELOPE_META_KEY: {"parts": [part]}})


_AID = "aid from France to Kenya"


def _other_window() -> dict[str, Any]:
    """The France-Kenya block over 2010 to 2014 under `_OTHER`: the same measure, so
    one unit question covers it beside `_FK_GROSS`."""
    other = copy.deepcopy(_FRANCE_KENYA)
    other["spec_id"] = other["definition"]["spec_id"] = _OTHER
    other["definition"]["identity"]["requested_window"] = {
        "kind": "range",
        "start": 2010,
        "end": 2014,
    }
    return other


@pytest.mark.parametrize(
    ("answer", "lead", "words", "base_year"),
    [
        (
            _answers(
                q1={"kind": "unit", "currency": "USD", "prices": "constant", "base_year": 2020}
            ),
            "The user chose constant 2020 US dollars.",
            "constant 2020 US dollars",
            2020,
        ),
        (
            {"skip": True},
            "The user left this to you: go on in constant 2023 US dollars.",
            "constant 2023 US dollars",
            2023,
        ),
    ],
    ids=["answered", "skipped"],
)
async def test_a_unit_answer_reads_each_part_it_covers_again_in_that_unit(
    pg_engine: AsyncEngine, answer: dict[str, Any], lead: str, words: str, base_year: int
) -> None:
    """A unit question covers its part and the part measuring the same thing
    over other years. Its answer, or a skip, which goes on with the unit the
    chips started on, re-reads both through Search in that unit, and the agent
    is told which spec_ids to fetch."""
    search = _UnitSearch()
    driver, paused, models = await _turn(
        pg_engine,
        "Aid to Kenya to 2014 and since 2018?",
        _ask_user([{"question": "In which prices?", "unit_of": _FK_GROSS}]),
        text_part("Done."),
        specs={_FK_GROSS: _FRANCE_KENYA, _OTHER: _other_window()},
        state={
            ORCHESTRATOR_SEARCHES_KEY: [
                _open_reading(_FK_GROSS, text=_AID),
                _open_reading(_OTHER, text=_AID),
            ]
        },
        definition_surface=search,
    )
    (question,) = paused.pause.payload["questions"]
    assert question["unit"] == {"parts": [_FK_GROSS, _OTHER], "offer": _USD_OFFER}

    resumed = await driver.answer(answer, to=paused.pause)

    assert resumed.outcome is Outcome.DONE
    unit = {"currency": "USD", "prices": "constant", "base_year": base_year}
    assert [posted["identity"]["facets"]["unit"]["request"] for posted in search.posted] == [
        unit,
        unit,
    ]
    expected = " ".join(
        [
            lead,
            *(
                f"Search read {_AID} in {words} as {posted['spec_id']}: fetch it in place of "
                f"{part}."
                for posted, part in zip(search.posted, (_FK_GROSS, _OTHER), strict=True)
            ),
        ]
    )
    asked = agent_read(models.orchestrator.requests[-1])
    assert f"You asked the user: In which prices?\n{expected}\n{_SETTLED}" in asked


@pytest.mark.parametrize(
    ("unit", "refusal"),
    [
        ({"currency": "USD", "prices": "current"}, None),
        ({"currency": "USD", "prices": "constant", "base_year": 2023}, None),
        (
            {"currency": "USD", "prices": "constant", "base_year": 2023},
            {"code": "requested_option_unavailable"},
        ),
    ],
    ids=["served", "read-again", "refused"],
)
async def test_a_part_the_user_settled_the_unit_of_lists_no_open_prices_and_is_not_asked_again(
    pg_engine: AsyncEngine, unit: dict[str, Any], refusal: dict[str, object] | None
) -> None:
    """A unit answer settles the part it covers, whether the part is served
    in that unit, Search re-reads it in that unit or Search refuses the unit.
    The part's prices close, and a second unit question on it is refused."""
    state = {ORCHESTRATOR_SEARCHES_KEY: [_open_reading(_FK_GROSS, text=_AID)]}
    specs = {_FK_GROSS: _FRANCE_KENYA}
    driver, paused, models = await _turn(
        pg_engine,
        "Aid to Kenya since 2018?",
        _ask_user([{"question": "In which prices?", "unit_of": _FK_GROSS}]),
        _ask_user([{"question": "And again?", "unit_of": _FK_GROSS}]),
        text_part("Done."),
        specs=specs,
        state=state,
        definition_surface=_UnitSearch(refusal=refusal),
    )

    resumed = await driver.answer(_answers(q1={"kind": "unit", **unit}), to=paused.pause)

    assert resumed.outcome is Outcome.DONE
    refused = [
        str(part.function_response.response)
        for content in models.orchestrator.requests[-1].content_list
        for part in content.parts or []
        if part.function_response is not None and part.function_response.name == "ask_user"
    ]
    assert [
        f"asks the unit of {_FK_GROSS}, which lists no `open_prices` this turn" in r
        for r in refused
    ] == [False, True]


def test_a_settled_part_is_closed_to_the_prices_gate() -> None:
    """Control for the row above: the same part is open until it is settled."""
    state: dict[str, Any] = {
        ORCHESTRATOR_SEARCHES_KEY: [_open_reading(_FK_GROSS, text=_AID)],
        ORCHESTRATOR_SPECS_KEY: {_FK_GROSS: _FRANCE_KENYA},
    }
    assert list(open_parts(state)) == [_FK_GROSS]
    assert prices_forms(state) == {_FK_GROSS: "unit"}

    settle_unit(state, _FK_GROSS)

    assert open_parts(state) == {}
    assert prices_forms(state) == {}


async def test_a_unit_search_does_not_serve_is_handed_back_to_fetch_as_served(
    pg_engine: AsyncEngine,
) -> None:
    """A unit Search refuses ends no turn: the agent is told to fetch the part
    in the unit it is served in and to say so."""
    search = _UnitSearch(refusal={"code": "requested_option_unavailable"})
    driver, paused, models = await _turn(
        pg_engine,
        "Aid to Kenya since 2018?",
        _ask_user([{"question": "In which prices?", "unit_of": _FK_GROSS}]),
        text_part("Done."),
        specs={_FK_GROSS: _FRANCE_KENYA},
        state={ORCHESTRATOR_SEARCHES_KEY: [_open_reading(_FK_GROSS, text=_AID)]},
        definition_surface=search,
    )

    resumed = await driver.answer(
        _answers(q1={"kind": "unit", "currency": "USD", "prices": "constant", "base_year": 2023}),
        to=paused.pause,
    )

    assert resumed.outcome is Outcome.DONE
    assert len(search.posted) == 1
    assert (
        f"The user chose constant 2023 US dollars. Search could not serve constant 2023 US "
        f"dollars for {_AID}: Search cannot serve {_AID} in that unit. Fetch {_FK_GROSS} as "
        "served, in current US dollars, and say so."
    ) in agent_read(models.orchestrator.requests[-1])


async def test_a_count_picked_beside_a_unit_answer_is_read_again_in_that_unit(
    pg_engine: AsyncEngine,
) -> None:
    """The user picks the net count and constant prices in one pause: Search
    re-reads the net count, not the gross count the unit question named."""
    search = _UnitSearch()
    driver, paused, models = await _turn(
        pg_engine,
        "Aid to Kenya since 2018?",
        _ask_user(
            [
                {
                    "question": "How should aid be counted?",
                    "options": [
                        {"label": "gross disbursements", "spec_id": _FK_GROSS},
                        {"label": "net disbursements", "spec_id": _FK_NET},
                    ],
                },
                {"question": "In which prices?", "unit_of": _FK_GROSS},
            ]
        ),
        text_part("Done."),
        specs={_FK_GROSS: _FRANCE_KENYA},
        state={ORCHESTRATOR_SEARCHES_KEY: [_open_reading(_FK_GROSS, text=_AID)]},
        definition_surface=search,
    )

    await driver.answer(
        _answers(
            q1={"kind": "option", "spec_id": _FK_NET},
            q2={"kind": "unit", "currency": "USD", "prices": "constant", "base_year": 2023},
        ),
        to=paused.pause,
    )

    (posted,) = search.posted
    (net,) = _FRANCE_KENYA["resolved"]["alternative_measures"]
    assert {**posted["identity"], "facets": None} == {
        **net["definition"]["identity"],
        "facets": None,
    }
    assert (
        f"Search read {_AID} in constant 2023 US dollars as {posted['spec_id']}: fetch it in "
        f"place of {_FK_NET}."
    ) in agent_read(models.orchestrator.requests[-1])


async def test_a_count_read_again_in_a_unit_keeps_the_part_s_roles_and_no_pair_note(
    pg_engine: AsyncEngine,
) -> None:
    """The net count re-read beside a unit answer is stored with the roles the
    gross part's reading asked for. The block lists the net count only in
    constant prices, and the hand-back carries no note telling the agent to
    search for the count's prices: code has read them. Both counts are
    settled."""
    del pg_engine
    search = _UnitSearch()
    block = copy.deepcopy(_FRANCE_KENYA)
    (net,) = block["resolved"]["alternative_measures"]
    for value in net["measure_axis_values"]:
        if value["axis"] == "measurementQualifier":
            value["value"] = {"ref": "RealValue", "label": "RealValue"}
    reading = SearchReading.model_validate(
        _open_reading(_FK_GROSS, text=_AID) | {"asked_roles": ["donor", "recipient"]}
    )
    state: dict[str, Any] = {
        ORCHESTRATOR_SPECS_KEY: {_FK_GROSS: block},
        ORCHESTRATOR_SEARCHES_KEY: [reading.model_dump(mode="json")],
    }
    ctx = NodeCtx(
        key=TurnKey("user-1", "conv", 1),
        inputs={"message": "Aid to Kenya since 2018?"},
        state=state,
        transcript=Transcript(),
        emit=lambda event: None,
        stop=StopScope(),
    )
    questions = [
        {
            "question": "How should aid be counted?",
            "options": [
                {"label": "gross disbursements", "spec_id": _FK_GROSS},
                {"label": "net disbursements", "spec_id": _FK_NET},
            ],
        },
        {"question": "In which prices?", "unit_of": _FK_GROSS},
    ]
    answer = _answers(
        q1={"kind": "option", "spec_id": _FK_NET},
        q2={"kind": "unit", "currency": "USD", "prices": "constant", "base_year": 2023},
    )
    node = ask_node(definition_surface=search)
    assert node.resume is not None

    handed = await node.resume(ctx, {"questions": questions}, answer)

    (posted,) = search.posted
    stored = [r for r in state[ORCHESTRATOR_SEARCHES_KEY] if r["spec_id"] == posted["spec_id"]]
    assert [tuple(r["asked_roles"]) for r in stored] == [("donor", "recipient")]
    assert "search once for that reading" not in handed.output["text"]
    assert {_FK_GROSS, _FK_NET} <= set(unit_settled(state))
    assert unit_reads(state) == {_FK_GROSS: [posted["spec_id"]]}


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


async def test_a_unit_question_whose_parts_share_no_currency_is_refused(
    pg_engine: AsyncEngine,
) -> None:
    """Two windows of one measure, one converted into US dollars and one into
    euros: no unit serves both, so `ask_user` refuses and nothing pauses."""
    euros = _other_window()
    (currency,) = euros["resolved"]["unit_conversion"]["currencies"]
    euros["resolved"]["unit_conversion"] = {
        **euros["resolved"]["unit_conversion"],
        "currencies": [{**currency, "code": "EUR", "noun": "euros"}],
    }
    _, run, _ = await _turn(
        pg_engine,
        "Aid to Kenya to 2014 and since 2018?",
        _ask_user([{"question": "In which prices?", "unit_of": _FK_GROSS}]),
        text_part("Done."),
        specs={_FK_GROSS: _FRANCE_KENYA, _OTHER: euros},
        state={ORCHESTRATOR_SEARCHES_KEY: [_open_reading(_FK_GROSS), _open_reading(_OTHER)]},
    )

    assert run.outcome is Outcome.DONE
    refused = _ask_user_answer(run)
    assert refused["code"] == ToolErrorCode.REQUEST_INVALID.value
    assert "no currency serves both current and constant prices" in refused["refusal"]


async def test_two_prices_questions_on_one_part_are_refused(pg_engine: AsyncEngine) -> None:
    """Two questions pricing one part would let the later answer override the
    earlier one in silence, so `ask_user` refuses the call and nothing pauses."""
    _, run, _ = await _turn(
        pg_engine,
        "Aid, 2018-2022?",
        _ask_user(
            [
                {"question": "Aid in which prices?", "unit_of": _GROSS},
                {"question": "And in which prices again?", "unit_of": _GROSS},
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
                {"question": "Aid to 2014 in which prices?", "unit_of": _OTHER},
                {"question": "Aid since 2018 in which prices?", "unit_of": _FK_GROSS},
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


def test_a_multi_select_question_is_asked_and_answered_by_its_labels() -> None:
    """A `multi_select` question's pause entry says so, its candidates are the
    labels under `o{n}` ids, and several ticks read back as the labels joined
    by "; ". The planner's readers of picks match `kind == "option"` only, so a
    label-only multi pick elects nothing."""
    question = AskQuestion.model_validate(
        {
            "question": "Which sectors?",
            "multi_select": True,
            "options": [{"label": "Health"}, {"label": "Education"}, {"label": "Water"}],
        }
    )
    (entry,) = ask_questions([question])
    ticked = {"kind": "options", "spec_ids": ["o1", "o3"]}

    assert entry["multi_select"] is True
    assert [c["spec_id"] for c in entry["candidates"]] == ["o1", "o2", "o3"]
    assert answer_replies([question], {"q1": ticked}) == ["The user chose: Health; Water"]
    assert answer_replies([question], {"q1": {"kind": "text", "text": "all"}}) == [
        "The user answered in their own words: all"
    ]
    assert answer_replies([question], {"q1": {"kind": "skipped"}}) == [
        "The user left this to you. Go on with Health."
    ]
    assert (
        "multi_select"
        not in ask_questions([question.model_copy(update={"multi_select": False})])[0]
    )
    offered = {"q1": frozenset({"o1", "o3"})}
    assert _picks({"answers": {"q1": ticked}}, offered) == ()
    assert _picks({"answers": {"q1": {"kind": "option", "spec_id": "o1"}}}, offered) == ("o1",)
