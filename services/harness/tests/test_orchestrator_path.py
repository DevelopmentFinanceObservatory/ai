"""The orchestrator's path through the turn graph on the turn loop, with a
scripted model: ask, plan, and what the agent reads after each.

Each turn runs the real nodes a plan call passes through
(`tests.plan_turns.plan_nodes`), the orchestrator on its real tools over a
scripted Gemini client, on Postgres (`tests.turn_driver.TurnDriver`).
"""

from __future__ import annotations

import copy
import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import pytest
from analysis_mcp import digest as analysis_digest
from analysis_mcp.execution_identity import current_execution_identity
from common.materialisation import FrameRef
from compiler.surface import FixtureSurface, ToolCallResult
from data_mcp import get_data_digest
from executor.completion import PublishFault
from executor.ledger import InMemoryRunLedger
from google.genai import types
from harness import plan_display
from harness.conversation import current_conversation_id
from harness.graph import execute as execute_module
from harness.graph.events import AgentStatus
from harness.graph.plan_done import _DROPPED
from harness.graph.state import (
    ORCHESTRATOR_BRIEF_KEY,
    ORCHESTRATOR_CALLS_KEY,
    ORCHESTRATOR_FRAMES_KEY,
    ORCHESTRATOR_PLAN_RECORDS_KEY,
    ORCHESTRATOR_PRIOR_LEDGER_KEY,
    ORCHESTRATOR_READINGS_KEY,
    ORCHESTRATOR_SEARCHES_KEY,
    ORCHESTRATOR_SPECS_KEY,
    UNREADABLE_PLAN_NOTE,
)
from harness.orchestrator_tools import HANDED_OFF
from harness.planner_adapter import PlannerAdapter, PlannerAdapterError
from harness.provenance import ProvenanceStore
from harness.scripted import ScriptedPlannerModel, ScriptedSurface, scripted_publish_result
from harness.sentences import DECLINE_REASONS
from harness.settings import Settings
from harness.studio_translate import _RecordingSurface
from harness.tool_contracts import HarnessCode
from harness.turn.events import Paused, TurnEvent, TurnFailed
from harness.turn_records import PlanRecord
from harness.wiring import handing_back_planner_failures
from planner.call import CallSession
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
from planner.session import SessionCall
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from .graph_turns import agent_read, given, opening_graph
from .plan_turns import approve_answer, plan_nodes
from .planner_host import FailingPlanner, RefusingPlanner
from .scripted_genai import ScriptedGenai, call_part, reply, text_part
from .scripted_tools import no_executor, no_resolver, scripted_orchestrator
from .test_plan_path import (
    SCRIPTED_SPEC,
    AskingPlanner,
    RefusingGetDataSurface,
    describes,
)
from .turn_driver import CONVERSATION, SegmentRun, TurnDriver, scripted_models


def call(name: str, **args: Any) -> types.Part:
    """A model turn calling the tool `name` with `args`."""
    return call_part(name, args)


class _Turn:
    """One conversation's turn on the plan path, built on its first
    `run_turn`: `model` is the agent's scripted client, and `last` the
    segment that ran last."""

    def __init__(
        self,
        engine: AsyncEngine,
        script: list[types.Part],
        *,
        conversation_id: str,
        turn_index: int,
        **wiring: Any,
    ) -> None:
        self.engine = engine
        self.conversation_id = conversation_id
        self.turn_index = turn_index
        # The agent answers from `script`, then in plain text.
        self.model = ScriptedGenai(
            [*(reply(part) for part in script), *(reply(text_part("done")) for _ in range(4))]
        )
        self.wiring = wiring
        self.opening: dict[str, Any] = {}
        self.driver: TurnDriver | None = None
        self.last: SegmentRun | None = None

    def build(self) -> TurnDriver:
        """The instance running the turn, its graph opening on the state
        `open_session` gave it."""
        models = scripted_models()
        models.orchestrator = self.model
        executor = self.wiring.pop("executor", None) or no_executor
        resolver = self.wiring.pop("resolver", None) or no_resolver
        planner = self.wiring.pop("planner", None) or RefusingPlanner()
        nodes = plan_nodes(
            scripted_orchestrator(resolver=resolver, executor=executor),
            models,
            executor=executor,
            run_ledger=self.wiring.pop("run_ledger", None) or InMemoryRunLedger(),
            planner=planner,
            **self.wiring,
        )
        state = dict(self.opening)
        specs = state.pop(ORCHESTRATOR_SPECS_KEY, None)
        before = given(specs=specs, state=state) if state or specs else None
        return TurnDriver(
            self.engine,
            graph=opening_graph(nodes, before=before),
            conversation_id=self.conversation_id,
        )

    async def run(self, segment: Callable[[TurnDriver], Any]) -> list[TurnEvent]:
        """Run the segment `segment` starts on the turn's instance, keep it
        as `last`, and return its events. A node that raised fails the
        turn, and its error is raised here."""
        if self.driver is None:
            self.driver = self.build()
        self.last = await segment(self.driver)
        failed = [event for event in self.last.events if isinstance(event, TurnFailed)]
        if failed:
            raise failed[-1].error
        return self.last.events


def orchestrator_runner(
    engine: AsyncEngine,
    *script: types.Part,
    planner: Any = None,
    tool_surface: Any = None,
    executor: Any = None,
    resolver: Any = None,
    publisher: Any = scripted_publish_result,
    run_ledger: Any = None,
    conversation_id: str = CONVERSATION,
    turn_index: int = 1,
) -> tuple[_Turn, ScriptedGenai]:
    """Turn `turn_index` of conversation `conversation_id`, whose agent
    answers from `script`, then "done" in plain text, and the agent's
    scripted client. The planner refuses (`RefusingPlanner`) unless
    `planner` is given; search and every executor call fail the test unless
    `resolver` or `executor` is given. `tool_surface` runs an approved plan,
    `publisher` publishes a run's results and `run_ledger` keeps its runs."""
    turn = _Turn(
        engine,
        list(script),
        conversation_id=conversation_id,
        turn_index=turn_index,
        planner=planner,
        tool_surface=tool_surface,
        executor=executor,
        resolver=resolver,
        publisher=publisher,
        run_ledger=run_ledger,
    )
    return turn, turn.model


async def open_session(turn: _Turn, **state: Any) -> None:
    """Open the turn on `state`, as its earlier turns and searches left it:
    spec blocks under `ORCHESTRATOR_SPECS_KEY`, the rest as state."""
    turn.opening = state


async def run_turn(turn: _Turn, message: str) -> list[TurnEvent]:
    """Run the turn on the user's `message` to its first pause or its end."""
    return await turn.run(lambda driver: driver.turn(message, turn_index=turn.turn_index))


async def answer_turn(turn: _Turn, answer: dict[str, Any]) -> list[TurnEvent]:
    """Answer the pause the turn holds with `answer`, and run on."""
    assert turn.last is not None
    pause = turn.last.pause
    return await turn.run(lambda driver: driver.answer(answer, to=pause))


def single_pause(events: list[TurnEvent]) -> Paused:
    """The one pause the segment ended on."""
    (pause,) = [event for event in events if isinstance(event, Paused)]
    return pause


def turn_state(turn: _Turn) -> dict[str, Any]:
    """The turn's state as its last segment stored it."""
    assert turn.last is not None
    return turn.last.cursor.state


async def store_state(turn: _Turn, key: str, value: Any) -> None:
    """Store `value` under `key` in the paused turn's state, for a test that
    needs state no node of this release would write."""
    assert turn.last is not None
    state = {**turn.last.cursor.state, key: value}
    async with turn.engine.begin() as conn:
        await conn.execute(
            text("UPDATE turn_cursors SET state = CAST(:state AS json) WHERE id = :id"),
            {"state": json.dumps(state), "id": turn.last.cursor.id},
        )


def handed_back(model: ScriptedGenai) -> tuple[str, dict[str, Any]]:
    """The agent's last request as it reads it, and the last `plan` result in it."""
    request = model.requests[-1]
    (result, *_) = [
        part.function_response.response or {}
        for content in reversed(request.content_list)
        for part in content.parts or []
        if part.function_response is not None and part.function_response.name == "plan"
    ]
    return agent_read(request), result


def request_text(request: Any) -> str:
    """`request` as the agent reads it (`tests.graph_turns.agent_read`)."""
    return agent_read(request)


_CONGO = {
    "question": "Which Congo?",
    "options": [{"label": "Republic of the Congo"}, {"label": "DR Congo"}],
}
_SETTLED = "Each answered choice is settled. Do not ask it again."


def _answers(**answers: dict[str, Any]) -> dict[str, Any]:
    """The clarify interrupt's resume value: one answer per question id."""
    return {"answers": answers}


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
    pg_engine: AsyncEngine,
    net_qualifier: str,
    answer: dict[str, Any],
    expected: list[str],
    elected: set[str],
):
    """A count picked in one answer, search's own reading included, and
    other prices in another, both of one search result, name the listed
    reading measuring both, or ask for one search when none is listed. A
    count listed only at the other price basis is said to be. Any other pair
    reads answer by answer."""
    block = _france_kenya_with_constant_prices(net_qualifier=net_qualifier)
    runner, model = orchestrator_runner(
        pg_engine, call("ask_user", questions=_COUNT_AND_PRICES), text_part("Done.")
    )
    await open_session(runner, **{ORCHESTRATOR_SPECS_KEY: {_FK_GROSS: block}})
    single_pause(await run_turn(runner, "Aid from France to Kenya?"))

    events = await answer_turn(runner, answer)

    assert not [e for e in events if isinstance(e, TurnFailed)]
    assert "\n".join(["You asked the user:", *expected, _SETTLED]) in request_text(
        model.requests[-1]
    )
    assert runner.driver is not None and runner.last is not None
    records = await runner.driver.records(runner.last.cursor)
    specs = await runner.driver.turns.specs(runner.last.cursor)
    state = {**turn_state(runner), ORCHESTRATOR_SPECS_KEY: specs}
    session = SessionCall(records=records, state=state, call_id=None)
    chosen = CallSession(session).elected_spec_ids()
    assert {spec_id for spec_id in chosen if spec_id.startswith("spec_")} == elected


async def test_a_planner_that_refuses_hands_back_and_the_agent_keeps_the_question(
    pg_engine: AsyncEngine,
):
    planner = RefusingPlanner()
    runner, model = orchestrator_runner(
        pg_engine,
        call(
            "plan",
            brief="Niger's Global Fund receipts as a share of health spending",
            asks=["Niger's Global Fund receipts as a share of health spending"],
        ),
        text_part("The share could not be calculated."),
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

    events = await run_turn(runner, "How much does Niger get from the Global Fund?")

    assert planner.questions == ["Niger's Global Fund receipts as a share of health spending"]
    # An earlier turn's readings come first, then this turn's.
    assert planner.readings == [[earlier, searched]]
    after_plan = request_text(model.requests[-1])
    assert "How much does Niger get from the Global Fund?" in after_plan
    assert "The planner could not make a plan: the planner ran out of rounds" in after_plan
    assert not [e for e in events if isinstance(e, TurnFailed)]
    state = turn_state(runner)
    assert state.get("question") in (None, "")
    assert state[ORCHESTRATOR_BRIEF_KEY].startswith("Niger's")


async def test_a_failed_plan_gives_the_plan_call_back_once(pg_engine: AsyncEngine):
    planner = RefusingPlanner()
    runner, model = orchestrator_runner(
        pg_engine,
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
        text_part("Could not be calculated."),
        planner=planner,
    )
    await open_session(runner)

    await run_turn(runner, "Kenya's aid as a share of GDP?")

    assert len(planner.questions) == 2
    third = [
        part.function_response.response
        for content in model.requests[-1].content_list
        for part in content.parts or []
        if part.function_response is not None
        and part.function_response.name == "plan"
        and part.function_response.response
        and "error" in part.function_response.response
    ]
    assert len(third) == 1 and "limit" in third[0]["error"]


@pytest.mark.parametrize("final", [False, True])
async def test_only_a_refusal_another_brief_could_avoid_offers_another_plan(
    pg_engine: AsyncEngine, final: bool
):
    """A final refusal (an internal or repeated failure, or a question declined
    whole) fails the same way for any brief, so the plan call is not given back
    and the hand-back does not offer one."""
    runner, model = orchestrator_runner(
        pg_engine,
        call("plan", brief="Kenya's aid as a share of GDP", asks=["Kenya's aid as a share of GDP"]),
        text_part("Could not be calculated."),
        planner=RefusingPlanner(final=final),
    )
    await open_session(runner)

    await run_turn(runner, "Kenya's aid as a share of GDP?")

    state = turn_state(runner)
    calls = state[ORCHESTRATOR_CALLS_KEY]
    assert bool(calls.get("plan_given_back")) is not final
    assert ("planned once more" in request_text(model.requests[-1])) is not final


async def test_each_plan_call_holds_the_planner_to_its_own_asks_alone(pg_engine: AsyncEngine):
    """A plan call lists the asks for its own plan only: a later call with
    one ask and no breakdowns leaves no earlier call's asks, breakdowns, or
    across-years marks for the planner to account for."""
    planner = RefusingPlanner()
    asks = ["a chart of France's aid to Guatemala", "the total since 2010 by donor"]
    runner, _ = orchestrator_runner(
        pg_engine,
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
        text_part("Could not be calculated."),
        planner=planner,
    )
    await open_session(runner)

    await run_turn(runner, "France's aid to Guatemala since 2010?")

    assert planner.asks == [tuple(asks), ("France's bilateral aid to Guatemala",)]
    assert planner.breakdowns == [(AskBreakdown(ask=1, by=("donor",)),), ()]
    assert planner.across_years == [(1,), ()]
    state = turn_state(runner)
    assert state.get(ASKS_STATE_KEY) == ["France's bilateral aid to Guatemala"]
    assert state.get(ASK_BREAKDOWNS_STATE_KEY) is None
    assert state.get(ASKS_ACROSS_YEARS_STATE_KEY) is None


async def test_a_planner_adapter_failure_is_handed_back_when_wrapped_and_fatal_when_not(
    pg_engine: AsyncEngine,
):
    runner, model = orchestrator_runner(
        pg_engine,
        call("plan", brief="Kenya's debt", asks=["Kenya's debt"]),
        text_part("Could not be calculated."),
        planner=handing_back_planner_failures(FailingPlanner()),
    )
    await open_session(runner)

    events = await run_turn(runner, "Kenya's debt?")

    assert (
        "The planner could not make a plan: a plan that is draft was declared ready, not "
        "pending_approval; refusing to report it ready" in request_text(model.requests[-1])
    )
    assert not [e for e in events if isinstance(e, TurnFailed)]
    # The failure's code travels with the status that fails the plan's step.
    failures = [e.failure for e in events if isinstance(e, AgentStatus) and e.failure]
    assert failures == [{"code": "internal_error"}]

    bare, _ = orchestrator_runner(
        pg_engine,
        call("plan", brief="Kenya's debt", asks=["Kenya's debt"]),
        planner=FailingPlanner(),
        conversation_id="conv_bare",
    )
    await open_session(bare)
    with pytest.raises(PlannerAdapterError):
        await run_turn(bare, "Kenya's debt?")


async def _breaking_resolver(**kwargs: Any) -> dict[str, Any]:
    raise RuntimeError("upstream at http://internal.example/v2 broke")


def _tool_answers(request: Any) -> dict[str, Any]:
    """Each function response in `request`, by tool name."""
    return {
        p.function_response.name: p.function_response.response
        for c in request.content_list
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
    contents = request.content_list
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
    responses = [p for c in request.content_list for p in c.parts or [] if p.function_response]
    assert len(responses) == len(pairs)


def _the_hand_back(request: Any, name: str) -> dict[str, Any]:
    """The one result `request` shows for the agent's call to `name`."""
    ((result,),) = [results for called, results in _paired_results(request) if called == name]
    assert HANDED_OFF not in result
    return result


async def test_a_trailing_hand_back_is_the_one_result_of_its_plan_call(pg_engine: AsyncEngine):
    """The model's request right after a plan ran ends on the plan's
    hand-back. The request holds every response to the call there, the later
    replacing the earlier, so the call shows the hand-back and not `_pend`'s
    record of the hand-off."""
    runner, model = orchestrator_runner(
        pg_engine,
        call("plan", brief="Kenya's debt service since 2015", asks=["Kenya's debt service"]),
        text_part("Charted."),
        planner=ScriptedPlannerModel(),
        tool_surface=ScriptedSurface(),
        executor=describes,
    )
    await open_session(runner)
    approval = single_pause(await run_turn(runner, "Kenya's debt service?"))
    revision = approval.payload["revision"]
    await answer_turn(runner, approve_answer(revision))

    request = model.requests[-1]
    assert _results_of(request.content_list[-1], "plan")
    result = _the_hand_back(request, "plan")
    assert result["result"] == "The approved plan ran. Chart its frames and answer."
    assert result["frames"]
    _assert_one_result_per_call(request)


async def test_a_hand_back_in_history_is_the_one_result_of_its_plan_call(pg_engine: AsyncEngine):
    """A later request holds the plan's hand-back in its history, with the
    agent's `ask_user` call and its answer after it. A call in
    history pairs with its last response, so the plan call still shows the
    hand-back alone, and the question its answer alone."""
    runner, model = orchestrator_runner(
        pg_engine,
        call("plan", brief="Kenya's debt service", asks=["Kenya's debt service"]),
        call("ask_user", questions=[_CONGO]),
        text_part("Kenya's debt service could not be calculated."),
    )
    await open_session(runner)
    single_pause(await run_turn(runner, "Kenya's debt service?"))
    await answer_turn(runner, {"answers": {"q1": {"kind": "option", "spec_id": "o1"}}})

    request = model.requests[-1]
    assert not _results_of(request.content_list[-1], "plan")
    assert _the_hand_back(request, "plan")["result"].startswith("The planner could not make a plan")
    assert _the_hand_back(request, "ask_user")["result"].startswith(
        "You asked the user: Which Congo?\nThe user chose: Republic of the Congo"
    )
    _assert_one_result_per_call(request)


async def test_a_tool_that_raises_answers_the_agent_with_its_class_and_the_turn_goes_on(
    pg_engine: AsyncEngine, caplog
):
    runner, model = orchestrator_runner(
        pg_engine,
        call("search", parts=[{"variable": "GDP"}]),
        text_part("Search failed, so I cannot give Kenya's GDP."),
        resolver=_breaking_resolver,
    )
    await open_session(runner)

    with caplog.at_level(logging.ERROR, logger="orchestrator.agent"):
        events = await run_turn(runner, "Kenya's GDP?")

    answer = _tool_answers(model.requests[-1])["search"]
    assert answer["error"].startswith("search failed inside DFO (RuntimeError).")
    assert "internal.example" not in answer["error"]
    assert (answer["code"], answer["retry_mode"]) == (HarnessCode.RAISED, "do_not_retry")
    assert not [e for e in events if isinstance(e, TurnFailed)]
    (record,) = [r for r in caplog.records if r.name == "orchestrator.agent"]
    assert record.levelno == logging.ERROR
    assert record.exc_info is not None and record.exc_info[0] is RuntimeError
    assert record.args[0] == "search"


async def test_a_made_up_tool_name_keeps_adk_answer_and_logs_no_failure(
    pg_engine: AsyncEngine, caplog
):
    runner, model = orchestrator_runner(
        pg_engine, call("summon_data", topic="GDP"), text_part("Sorry.")
    )
    await open_session(runner)

    with caplog.at_level(logging.ERROR, logger="orchestrator.agent"):
        events = await run_turn(runner, "Kenya's GDP?")

    answer = _tool_answers(model.requests[-1])["summon_data"]
    assert "The tools you can call are: " in answer["error"]
    assert not [e for e in events if isinstance(e, TurnFailed)]
    assert not [r for r in caplog.records if r.name == "orchestrator.agent"]


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
async def test_the_hand_back_reports_an_ask_the_users_amendment_removed_as_removed(
    pg_engine: AsyncEngine, ending: str
):
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
        pg_engine,
        call(
            "plan",
            brief="ODA received by Ghana, Kenya and Nigeria",
            asks=["ODA to Ghana", "ODA to Kenya", "ODA to Nigeria"],
        ),
        text_part("Answered."),
        planner=planner,
        tool_surface=surface,
        executor=describes,
    )
    await open_session(runner)

    amendments = ["Show each year.", "Drop Nigeria, just Ghana and Kenya."]
    approval = single_pause(await run_turn(runner, "ODA to three countries?"))
    for words in amendments:
        approval = single_pause(await answer_turn(runner, {"approved": None, "amendment": words}))
    await answer_turn(runner, approve_answer(approval.payload["revision"]))

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


async def test_an_amendment_redrafts_and_the_amended_plan_runs(pg_engine: AsyncEngine):
    runner, model = orchestrator_runner(
        pg_engine,
        call(
            "plan",
            brief="Kenya's debt service since 2015",
            asks=["Kenya's debt service since 2015"],
        ),
        text_part("Charted."),
        planner=ScriptedPlannerModel(),
        tool_surface=ScriptedSurface(),
        executor=describes,
    )
    await open_session(runner)

    first = single_pause(await run_turn(runner, "Kenya's debt service?"))
    second = single_pause(await answer_turn(runner, {"approved": None, "amendment": "only 2020"}))
    assert second.payload["version"] == first.payload["version"] + 1
    await answer_turn(runner, approve_answer(second.payload["revision"]))

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
    pg_engine: AsyncEngine, stored: str, answer: str, caplog: pytest.LogCaptureFixture
):
    """A plan awaiting approval across the release cannot be read. Approving
    it or amending it retrieves nothing: the question is planned from the
    start, the new card says why, and approving that one runs it."""
    surface = _CountingSurface()
    runner, model = orchestrator_runner(
        pg_engine,
        call(
            "plan",
            brief="Kenya's debt service since 2015",
            asks=["Kenya's debt service since 2015"],
        ),
        text_part("Charted."),
        planner=ScriptedPlannerModel(),
        tool_surface=surface,
        executor=describes,
    )
    await open_session(runner)
    single_pause(await run_turn(runner, "Kenya's debt service?"))
    held = _PLANS_WITH_MEMBER_LISTS[stored]
    await store_state(runner, "plan", json.dumps(held["plan"]))
    result = (
        approve_answer(held["revision"])
        if answer == "approves"
        else {"approved": None, "amendment": "only 2020"}
    )

    with caplog.at_level(logging.WARNING, logger="planner.plan"):
        second = single_pause(await answer_turn(runner, result))

    assert second.kind == "approve"
    assert second.payload["refusal_note"] == UNREADABLE_PLAN_NOTE
    assert second.payload["version"] == 1
    assert surface.calls == []
    ((state_key, entries, fields),) = _plan_warnings(caplog)
    assert (state_key, entries) == ("plan", 1)
    assert "workflow.sources.stat_vars" in str(fields)
    assert held["plan"]["question"] not in caplog.text

    await answer_turn(runner, approve_answer(second.payload["revision"]))

    assert "get_data" in surface.calls
    assert "The approved plan ran." in request_text(model.requests[-1])


async def test_a_plan_call_drafts_over_an_earlier_plan_stored_with_member_lists(
    pg_engine: AsyncEngine,
):
    """A session's earlier plan call left a plan the release cannot read.
    The next plan call drafts its own plan as it would with none on record,
    and its card says nothing of the earlier one, which no card of this
    call showed."""
    runner, _ = orchestrator_runner(
        pg_engine,
        call(
            "plan",
            brief="Kenya's debt service since 2015",
            asks=["Kenya's debt service since 2015"],
        ),
        planner=ScriptedPlannerModel(),
        tool_surface=ScriptedSurface(),
        executor=describes,
    )
    await open_session(runner, plan=_UNREADABLE_PLAN)

    approval = single_pause(await run_turn(runner, "Kenya's debt service?"))

    assert approval.kind == "approve"
    assert "refusal_note" not in approval.payload
    assert Plan.model_validate_json(turn_state(runner)["plan"]).version == 1


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
    pg_engine: AsyncEngine,
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
        pg_engine,
        call(
            "plan",
            brief="Kenya's debt service since 2015",
            asks=["Kenya's debt service since 2015"],
        ),
        planner=ScriptedPlannerModel(),
        tool_surface=surface,
        executor=describes,
    )
    await open_session(runner)
    first = single_pause(await run_turn(runner, "Kenya's debt service?"))

    second = single_pause(await answer_turn(runner, approve_answer(first.payload["revision"])))

    assert second.kind == "approve"
    assert second.payload["refusal_note"] == UNREADABLE_PLAN_NOTE
    assert surface.calls == []
    assert not turn_state(runner).get("replans")


async def test_the_hand_back_carries_no_record_of_a_plan_stored_with_member_lists(
    pg_engine: AsyncEngine,
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
        pg_engine,
        call(
            "plan",
            brief="Kenya's debt service since 2015",
            asks=["Kenya's debt service since 2015"],
        ),
        text_part("Charted."),
        planner=ScriptedPlannerModel(),
        tool_surface=ScriptedSurface(),
        executor=describes,
    )
    await open_session(runner)
    approval = single_pause(await run_turn(runner, "Kenya's debt service?"))
    revision = approval.payload["revision"]

    events = await answer_turn(runner, approve_answer(revision))

    handed_back = request_text(model.requests[-1])
    assert "The approved plan ran. Chart its frames and answer." in handed_back
    assert f"Plan {revision[:12]}" not in handed_back
    assert not [e for e in events if isinstance(e, TurnFailed)]
    state = turn_state(runner)
    assert state["plan"] is None
    assert not state.get(ORCHESTRATOR_PLAN_RECORDS_KEY)


async def test_approving_a_superseded_revision_retrieves_nothing_and_proposes_again(
    pg_engine: AsyncEngine,
):
    surface = _CountingSurface()
    runner, _ = orchestrator_runner(
        pg_engine,
        call(
            "plan",
            brief="Kenya's debt service since 2015",
            asks=["Kenya's debt service since 2015"],
        ),
        planner=ScriptedPlannerModel(),
        tool_surface=surface,
        executor=describes,
    )
    await open_session(runner)

    first = single_pause(await run_turn(runner, "Kenya's debt service?"))
    second = single_pause(await answer_turn(runner, {"approved": None, "amendment": "only 2020"}))
    third = single_pause(await answer_turn(runner, approve_answer(first.payload["revision"])))

    assert third.kind == "approve"
    assert third.payload["version"] == second.payload["version"] + 1
    assert "get_data" not in surface.calls


async def test_a_hand_back_is_still_in_view_after_a_later_pause(pg_engine: AsyncEngine):
    runner, model = orchestrator_runner(
        pg_engine,
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
        text_part("Kenya's debt service since 2015 could not be calculated."),
    )
    await open_session(runner)

    single_pause(await run_turn(runner, "Kenya's debt service?"))
    await answer_turn(runner, {"answers": {"q1": {"kind": "option", "spec_id": "o1"}}})

    resumed = request_text(model.requests[-1])
    assert "Kenya's debt service?" in resumed
    assert "The planner could not make a plan" in resumed
    assert "The user chose: Since 2015" in resumed


async def test_a_second_plan_call_runs_a_planner_that_never_sees_the_first_hand_back(
    pg_engine: AsyncEngine,
):
    # The planner's model answers each run with text, which ends it with no plan.
    scripted = ScriptedGenai([reply(text_part("done"))] * 2)
    adapter = PlannerAdapter(
        settings=Settings(), surface=FixtureSurface(schemas=[]), client=scripted.client
    )
    runner, model = orchestrator_runner(
        pg_engine,
        call("plan", brief="Kenya's debt, first brief", asks=["Kenya's debt, first brief"]),
        call("plan", brief="Kenya's debt, second brief", asks=["Kenya's debt, second brief"]),
        text_part("Could not be calculated."),
        planner=handing_back_planner_failures(adapter),
    )
    await open_session(runner)

    events = await run_turn(runner, "Kenya's debt?")

    hand_back = "The planner could not make a plan"
    planner_texts = [
        "\n".join(part.text or "" for content in r.content_list for part in content.parts or [])
        for r in scripted.requests
    ]
    second_run = [text for text in planner_texts if "second brief" in text]
    assert second_run
    assert not [text for text in second_run if hand_back in text]
    assert hand_back in request_text(model.requests[-1])
    assert not [e for e in events if isinstance(e, TurnFailed)]


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
    [([], "translator"), ([SCRIPTED_SPEC], "search")],
    ids=["found_by_the_translator", "searched_in_an_earlier_turn"],
)
async def test_the_plan_record_and_every_reading_reach_the_agent_past_wide_step_arguments(
    pg_engine: AsyncEngine, monkeypatch: pytest.MonkeyPatch, prior_spec_ids: list[str], origin: str
):
    """A source an earlier turn's search returned is recorded as searched, so
    its block in that turn's record stays the one a later turn pins from."""
    monkeypatch.setattr(plan_display, "step_rows", _rows_led_by_a_wide_step(plan_display.step_rows))
    runner, model = orchestrator_runner(
        pg_engine,
        call(
            "plan",
            brief="Kenya's debt service since 2015",
            asks=["Kenya's debt service since 2015"],
        ),
        text_part("Charted."),
        planner=ScriptedPlannerModel(),
        tool_surface=ScriptedSurface(),
        executor=describes,
    )
    prior = {"readings": [], "spec_ids": prior_spec_ids, "warnings": {}}
    await open_session(runner, **{ORCHESTRATOR_PRIOR_LEDGER_KEY: prior})

    approval = single_pause(await run_turn(runner, "Kenya's debt service?"))
    revision = approval.payload["revision"]
    events = await answer_turn(runner, approve_answer(revision))

    after_plan = request_text(model.requests[-1])
    assert f"Plan {revision[:12]} (approved)" in after_plan
    assert f"({SCRIPTED_SPEC}, sum)" in after_plan
    assert "Wide_StatVar_" not in after_plan
    state = turn_state(runner)
    assert state[ORCHESTRATOR_READINGS_KEY] == {SCRIPTED_SPEC: "sum"}
    (stored,) = state[ORCHESTRATOR_PLAN_RECORDS_KEY]
    record = PlanRecord.model_validate(stored)
    assert record.revision == revision
    assert [source.origin for source in record.sources] == [origin]
    # The run the plan executed as, which the hand-back names.
    assert record.run_id is not None and f'"run_id": "{record.run_id}"' in after_plan
    assert not [e for e in events if isinstance(e, TurnFailed)]


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


async def test_step_summaries_reach_the_agent_without_frame_references_or_operation_notes(
    pg_engine: AsyncEngine,
):
    runner, model = orchestrator_runner(
        pg_engine,
        call("plan", brief="Kenya's life expectancy", asks=["Kenya's life expectancy"]),
        text_part("Charted."),
        planner=ScriptedPlannerModel(),
        tool_surface=_NoisySurface(),
        executor=describes,
    )
    await open_session(runner)

    approval = single_pause(await run_turn(runner, "Kenya's life expectancy?"))
    await answer_turn(runner, approve_answer(approval.payload["revision"]))

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


async def _store_versions(runner: _Turn) -> dict[str, str]:
    """Each registered frame's id and the `content_version` the store holds it under."""
    state = turn_state(runner)
    return {
        frame_id: frame["store_ref"]["content_version"]
        for frame_id, frame in state[ORCHESTRATOR_FRAMES_KEY].items()
    }


async def _approved_two_asks(
    pg_engine: AsyncEngine,
    surface: Any,
    planner: TwoAsksPlanner | None = None,
    *,
    publisher: Any = scripted_publish_result,
    run_ledger: Any = None,
    turn_index: int = 1,
) -> tuple[ScriptedGenai, TwoAsksPlanner]:
    """Run the two-asks brief as turn `turn_index`, approving every plan the
    planner drafts."""
    _, model, planner = await _run_two_asks(
        pg_engine,
        surface,
        planner,
        publisher=publisher,
        run_ledger=run_ledger,
        turn_index=turn_index,
    )
    return model, planner


async def _run_two_asks(
    pg_engine: AsyncEngine,
    surface: Any,
    planner: TwoAsksPlanner | None = None,
    *,
    publisher: Any = scripted_publish_result,
    run_ledger: Any = None,
    turn_index: int = 1,
) -> tuple[_Turn, ScriptedGenai, TwoAsksPlanner]:
    """`_approved_two_asks`, with the turn that holds the turn's state."""
    planner = planner or TwoAsksPlanner()
    runner, model = orchestrator_runner(
        pg_engine,
        call("plan", brief="Kenya's life expectancy and its mean", asks=_TWO_ASKS),
        text_part("Answered."),
        planner=planner,
        tool_surface=surface,
        executor=describes,
        publisher=publisher,
        run_ledger=run_ledger,
        turn_index=turn_index,
    )
    await open_session(runner)
    events = await run_turn(runner, "Kenya's life expectancy?")
    while any(isinstance(e, Paused) for e in events):
        approval = single_pause(events)
        events = await answer_turn(runner, approve_answer(approval.payload["revision"]))
    return runner, model, planner


async def test_a_terminal_step_another_step_reads_comes_back_on_an_accepted_plan(
    pg_engine: AsyncEngine,
):
    model, _ = await _approved_two_asks(pg_engine, ScriptedSurface())

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


async def test_a_pending_run_hands_back_its_retrieved_terminal_and_withholds_the_derived_one(
    pg_engine: AsyncEngine,
):
    model, _ = await _approved_two_asks(
        pg_engine, ScriptedSurface(), publisher=_publication_pending
    )

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


async def test_a_pending_run_admitted_again_on_a_fresh_surface_keeps_its_retrieved_provenance(
    pg_engine: AsyncEngine,
):
    """The process that computed the run is gone with its provenance store:
    the retrieved terminal it hands back rests on the retrievals it sealed."""
    ledger = InMemoryRunLedger(scope=current_conversation_id.get)
    first = _RecordingSurface(ScriptedSurface(), provenance=ProvenanceStore())
    await _approved_two_asks(pg_engine, first, publisher=_publication_pending, run_ledger=ledger)
    provenance = ProvenanceStore()
    second = _RecordingSurface(ScriptedSurface(), provenance=provenance)
    model, _ = await _approved_two_asks(
        pg_engine, second, publisher=_publication_pending, run_ledger=ledger, turn_index=2
    )

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
    pg_engine: AsyncEngine,
    declined: tuple[Decline, ...],
):
    runner, model, _ = await _run_two_asks(
        pg_engine, _StoredApart(_AggregateRefusingSurface()), TwoAsksPlanner(declined)
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


async def test_a_failed_plan_hands_back_no_step_that_is_only_read(pg_engine: AsyncEngine):
    """The retrieval finished, but only the refused aggregate answers an ask."""
    runner, model = orchestrator_runner(
        pg_engine,
        call("plan", brief="Kenya's life expectancy", asks=["Kenya's mean life expectancy"]),
        text_part("It could not be computed."),
        planner=AskingPlanner(),
        tool_surface=_AggregateRefusingSurface(),
        executor=describes,
    )
    await open_session(runner)
    approval = single_pause(await run_turn(runner, "Kenya's life expectancy?"))
    await answer_turn(runner, approve_answer(approval.payload["revision"]))

    _, body = handed_back(model)
    assert body["frames"] == []
    assert body["failed_step"]["step_id"] == "totals"


async def test_a_replan_names_the_asks_the_failed_run_computed(pg_engine: AsyncEngine):
    _, planner = await _approved_two_asks(pg_engine, _EmptyAggregateSurface())

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
    pg_engine: AsyncEngine,
    declined: tuple[Decline, ...],
):
    model, planner = await _approved_two_asks(
        pg_engine, _EmptyAggregateSurface(), _RefusingReplanner(declined)
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
    pg_engine: AsyncEngine,
    declined: tuple[Decline, ...],
):
    runner, model, _ = await _run_two_asks(
        pg_engine, _StoredApart(_RetrievalRefusingOnReplanSurface()), TwoAsksPlanner(declined)
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
