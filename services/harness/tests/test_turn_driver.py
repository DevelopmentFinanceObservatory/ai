"""The scripted turn driver (`tests.turn_driver`) runs the real orchestrator
through the turn loop on Postgres.

The orchestrator sits in a one-node graph built here: the node runs
`orchestrator.run.run_orchestrator`, ends the turn on its answer, and pauses
the turn on a hand-off with the run's history in the turn state, which the
node's `resume` runs on with the user's answer as the hand-back.
"""

from __future__ import annotations

from datetime import date
from typing import Any

from google.genai import types
from harness.graph.state import ORCHESTRATOR_PENDING_KEY, ORCHESTRATOR_SPECS_KEY
from harness.scripted import SCRIPTED_SPEC
from harness.turn import transcript as entries
from harness.turn.events import NodeDone
from harness.turn.loop import END, Graph, Outcome
from harness.turn.node import Next, Node, NodeCtx, Pause
from harness.turn.store import TurnStatus
from orchestrator.agent import Orchestrator
from orchestrator.run import (
    Answered,
    HandBack,
    HandedOff,
    Opening,
    OrchestratorEvent,
    StoredHistory,
    run_orchestrator,
)
from sqlalchemy.ext.asyncio import AsyncEngine

from .scripted_genai import reply, text_part
from .scripted_tools import (
    CannedSearch,
    ask_user_call,
    definite_envelope,
    scripted_orchestrator,
    search_call,
)
from .turn_driver import (
    ScriptedModels,
    TurnDriver,
    numbered_call_ids,
    request_text,
    scripted_models,
)

_HISTORY = "test:orchestrator_history"
_TODAY = date(2026, 11, 17)


def _graph(orchestrator: Orchestrator, models: ScriptedModels, seen: list[Any]) -> Graph:
    """One `orchestrate` node over `orchestrator` on `models`, recording each
    orchestrator event in `seen`."""

    async def finish(ctx: NodeCtx, runs: Any) -> Next | Pause:
        async for event in runs:
            seen.append(event)
            if isinstance(event, HandedOff):
                ctx.state[_HISTORY] = event.history.model_dump(mode="json")
                return Pause(kind="ask", payload=ctx.state[ORCHESTRATOR_PENDING_KEY])
            if isinstance(event, Answered):
                return Next("answered", {"text": event.text})
        raise AssertionError("the run ended without an answer or a hand-off")

    def run_on(ctx: NodeCtx, **how: Any) -> Any:
        return run_orchestrator(
            orchestrator,
            state=ctx.state,
            turn_id=f"{ctx.key.conversation_id}/{ctx.key.turn_index}",
            model=models.orchestrator_model(),
            **how,
        )

    async def run(ctx: NodeCtx, node_input: Any) -> Next | Pause:
        opening = Opening(question=ctx.inputs["message"], today=_TODAY)
        return await finish(ctx, run_on(ctx, opening=opening))

    async def resume(ctx: NodeCtx, node_input: Any, answer: dict[str, Any]) -> Next | Pause:
        history = StoredHistory.model_validate(ctx.state.pop(_HISTORY))
        ctx.state.pop(ORCHESTRATOR_PENDING_KEY)
        hand_back = HandBack(answers="ask_user", result=answer)
        return await finish(ctx, run_on(ctx, history=history, hand_back=hand_back))

    return Graph(
        start="orchestrate",
        nodes={"orchestrate": Node(run=run, resume=resume)},
        edges={("orchestrate", "answered"): END},
    )


async def test_a_plain_answer_closes_the_turn_with_the_state_its_tools_wrote(
    pg_engine: AsyncEngine,
) -> None:
    models = scripted_models(
        orchestrator=[
            reply(search_call("life expectancy")),
            reply(text_part("Life expectancy in Kenya was 62 years.")),
        ]
    )
    searches = CannedSearch(definite_envelope())
    seen: list[OrchestratorEvent] = []
    driver = TurnDriver(
        pg_engine, graph=_graph(scripted_orchestrator(resolver=searches), models, seen)
    )

    with numbered_call_ids():
        run = await driver.turn("Life expectancy in Kenya?")

    assert run.outcome is Outcome.DONE
    assert run.events == [
        NodeDone(
            node="orchestrate",
            label="answered",
            output={"text": "Life expectancy in Kenya was 62 years."},
        )
    ]
    assert run.cursor.status is TurnStatus.DONE
    assert list(run.cursor.state[ORCHESTRATOR_SPECS_KEY]) == [SCRIPTED_SPEC]
    assert [record.kind for record in await driver.records(run.cursor)] == [entries.USER]
    assert len(searches.calls) == 1
    assert len(models.orchestrator.requests) == 2
    assert "Life expectancy in Kenya?" in request_text(models.orchestrator.requests[0])
    assert [event.text for event in seen if isinstance(event, Answered)] == [
        "Life expectancy in Kenya was 62 years."
    ]


async def test_a_hand_off_pauses_and_another_instance_resumes_it_with_the_answer(
    pg_engine: AsyncEngine,
) -> None:
    """The run's history crosses the pause through the stored cursor: the
    instance that resumes holds no orchestrator object of the first."""
    models = scripted_models(
        orchestrator=[
            reply(ask_user_call("Which Congo?", "Republic of the Congo", "DR Congo")),
            reply(text_part("DR Congo it is.")),
        ]
    )
    seen: list[OrchestratorEvent] = []
    first = TurnDriver(pg_engine, graph=_graph(scripted_orchestrator(), models, seen))
    second = TurnDriver(pg_engine, graph=_graph(scripted_orchestrator(), models, seen))

    with numbered_call_ids():
        paused = await first.turn("GDP of Congo?")
        pause = paused.pause
        resumed = await second.answer({"answer": "DR Congo"}, to=pause)

    assert paused.cursor.status is TurnStatus.PAUSED
    assert (pause.kind, pause.round) == ("ask", 1)
    assert pause.payload["questions"][0]["question"] == "Which Congo?"
    assert resumed.outcome is Outcome.DONE
    assert resumed.events == [
        NodeDone(node="orchestrate", label="answered", output={"text": "DR Congo it is."})
    ]
    assert _HISTORY not in resumed.cursor.state
    assert [record.kind for record in await second.records(resumed.cursor)] == [
        entries.USER,
        entries.PAUSE,
        entries.ANSWER,
    ]
    [hand_back] = [
        part.function_response
        for content in models.orchestrator.requests[1].content_list
        for part in content.parts or []
        if part.function_response is not None
    ]
    assert isinstance(hand_back, types.FunctionResponse)
    assert (hand_back.name, hand_back.response) == ("ask_user", {"answer": "DR Congo"})
    assert [event.call_id for event in seen if isinstance(event, HandedOff)] == ["adk-1"]
