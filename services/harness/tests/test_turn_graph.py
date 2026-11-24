"""The turn's edges (`harness.graph.workflow`) and its first nodes, `seed`,
`orchestrate` and `ask`, on the turn loop over Postgres.

The node tests run the real nodes in a test-only graph that ends where
they end (`tests.graph_turns.opening_graph`), the orchestrator on its real
tools over a scripted Gemini client.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
from harness.gemini_agent import EmptyResponseError
from harness.graph.events import AgentStatus
from harness.graph.orchestrate import turn_id
from harness.graph.resume import Resumed
from harness.graph.state import (
    ORCHESTRATOR_HISTORY_KEY,
    ORCHESTRATOR_PENDING_KEY,
    ORCHESTRATOR_SEARCHES_KEY,
    ORCHESTRATOR_SPECS_KEY,
    PendingPlan,
    stored_mapping,
    stored_sequence,
)
from harness.graph.workflow import EDGES, START, turn_graph
from harness.turn.events import NodeDone, Paused, TurnFailed
from harness.turn.loop import END, Outcome
from harness.turn.node import Next, Node, NodeCtx
from harness.turn.store import TurnStatus
from orchestrator.run import ModelResponded, TextDelta, ToolsAnswered
from pydantic_ai.exceptions import UnexpectedModelBehavior
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from .graph_turns import GIVEN, agent_read, given, opening_graph, opening_nodes, visited
from .scripted_genai import Reply, blocked_reply, empty_reply, reply, text_part
from .scripted_tools import ask_user_call, plan_call, scripted_orchestrator
from .turn_driver import TurnDriver, numbered_call_ids, scripted_models


async def _stored_state(engine: AsyncEngine, turn: int) -> dict[str, Any]:
    """The state column of turn `turn` as stored, without the spec blocks
    the store keeps as rows."""
    async with engine.connect() as conn:
        return (
            await conn.execute(text("SELECT state FROM turn_cursors WHERE id = :id"), {"id": turn})
        ).scalar_one()


_NODES = {"seed", "orchestrate", "ask", "plan_start", "planner", "approve", "switch", "execute"}
_NODES |= {"plan_done"}


async def _goes(ctx: NodeCtx, node_input: Any) -> Next:
    del ctx, node_input
    return Next("next")


def test_every_edge_leaves_a_node_of_the_turn_and_every_node_but_the_start_is_reached() -> None:
    sources = {source for source, _ in EDGES}
    targets = set(EDGES.values())

    assert sources == _NODES
    assert targets - {END} == _NODES - {START}
    assert EDGES[("orchestrate", "answered")] == END
    assert {label for source, label in EDGES if source == "orchestrate"} == {
        "plan",
        "ask",
        "answered",
    }


def test_the_turn_graph_opens_at_seed_and_refuses_a_missing_node() -> None:
    graph = turn_graph(dict.fromkeys(_NODES, Node(run=_goes)))

    assert graph.start == START == "seed"
    assert dict(graph.edges) == dict(EDGES)
    with pytest.raises(ValueError, match="unknown node"):
        turn_graph(dict.fromkeys(_NODES - {"ask"}, Node(run=_goes)))


async def test_an_answered_turn_shows_its_status_streams_the_run_and_ends(
    pg_engine: AsyncEngine,
) -> None:
    models = scripted_models(orchestrator=[reply(text_part("Kenya's GDP was $101 billion."))])
    nodes = opening_nodes(scripted_orchestrator(), models)
    driver = TurnDriver(pg_engine, graph=opening_graph(nodes))

    run = await driver.turn("Kenya's GDP in 2020?", inputs={"canvas_digest": '{"cards": []}'})

    assert run.outcome is Outcome.DONE
    assert visited(run.events) == ["seed", "orchestrate"]
    assert run.events[1] == AgentStatus(text="Reading your question")
    assert run.events[2] == TextDelta(text="Kenya's GDP was $101 billion.")
    assert isinstance(run.events[3], ModelResponded)
    assert run.events[-1] == NodeDone(node="orchestrate", label="answered")
    read = agent_read(models.orchestrator.requests[0])
    assert 'The user\'s canvas: {"cards": []}' in read
    assert read.endswith("\nKenya's GDP in 2020?")
    assert ORCHESTRATOR_HISTORY_KEY not in run.cursor.state
    assert ORCHESTRATOR_SPECS_KEY not in await _stored_state(pg_engine, run.cursor.id)


async def test_a_plan_hand_off_leaves_on_the_plan_edge_with_the_action_and_keeps_the_run(
    pg_engine: AsyncEngine,
) -> None:
    models = scripted_models(orchestrator=[reply(plan_call("rank donors", "the top 5"))])
    nodes = opening_nodes(scripted_orchestrator(), models)
    driver = TurnDriver(pg_engine, graph=opening_graph(nodes))

    run = await driver.turn("Top donors?")

    assert run.outcome is Outcome.DONE
    done = run.events[-1]
    assert isinstance(done, NodeDone)
    assert (done.node, done.label) == ("orchestrate", "plan")
    assert PendingPlan.model_validate(done.output) == PendingPlan(
        brief="rank donors", asks=["the top 5"]
    )
    assert ORCHESTRATOR_PENDING_KEY not in run.cursor.state
    assert run.cursor.state[ORCHESTRATOR_HISTORY_KEY]["schema_version"] == 1


async def test_an_ask_pauses_and_another_instance_hands_the_answer_back_to_the_waiting_call(
    pg_engine: AsyncEngine,
) -> None:
    """The run's conversation crosses the pause in the turn's state. The
    resumed run's first `ToolsAnswered` carries the hand-back, and no other
    event of the segment does."""
    models = scripted_models(
        orchestrator=[
            reply(ask_user_call("Which Congo?", "Republic of the Congo", "DR Congo")),
            reply(text_part("DR Congo it is.")),
        ]
    )
    first = TurnDriver(
        pg_engine, graph=opening_graph(opening_nodes(scripted_orchestrator(), models))
    )
    second = TurnDriver(
        pg_engine, graph=opening_graph(opening_nodes(scripted_orchestrator(), models))
    )

    with numbered_call_ids():
        paused = await first.turn("GDP of Congo?")
        pause = paused.pause
        resumed = await second.answer(
            {"answers": {"q1": {"kind": "option", "spec_id": "o2"}}}, to=pause
        )

    assert visited(paused.events) == ["seed", "orchestrate"]
    assert paused.events[-2] == NodeDone(
        node="orchestrate",
        label="ask",
        output={
            "kind": "ask",
            "questions": [
                {
                    "question": "Which Congo?",
                    "options": [
                        {"label": "Republic of the Congo", "spec_id": None},
                        {"label": "DR Congo", "spec_id": None},
                    ],
                    "unit_of": None,
                    "prices": None,
                    "multi_select": False,
                }
            ],
        },
    )
    assert isinstance(pause, Paused)
    assert (pause.kind, pause.message) == ("ask", "Which Congo?")
    assert paused.cursor.node == "ask"

    assert resumed.outcome is Outcome.DONE
    assert visited(resumed.events) == ["ask", "orchestrate"]
    assert resumed.events[1] == AgentStatus(text="Reading your answer")
    answered = [event for event in resumed.events if isinstance(event, ToolsAnswered)]
    assert answered[0] is resumed.events[2]
    hand_back = answered[0].answers[0]
    assert (hand_back.id, hand_back.name) == ("adk-1", "ask_user")
    assert hand_back.response == {
        "result": "You asked the user: Which Congo?\nThe user chose: DR Congo\n"
        "Each answered choice is settled. Do not ask it again."
    }
    assert sum(hand_back in event.answers for event in answered) == 1
    assert "The user chose: DR Congo" in agent_read(models.orchestrator.requests[1])
    assert ORCHESTRATOR_HISTORY_KEY not in resumed.cursor.state


@pytest.mark.parametrize("answer", [blocked_reply, empty_reply], ids=["blocked", "empty"])
async def test_a_model_response_with_no_answer_fails_the_turn_at_orchestrate(
    pg_engine: AsyncEngine, answer: Callable[[], Reply]
) -> None:
    models = scripted_models(orchestrator=[answer()])
    nodes = opening_nodes(scripted_orchestrator(), models)
    driver = TurnDriver(pg_engine, graph=opening_graph(nodes))

    run = await driver.turn("Kenya's GDP?")

    assert run.outcome is Outcome.FAILED
    failed = run.events[-1]
    assert isinstance(failed, TurnFailed)
    assert failed.node == "orchestrate"
    assert run.cursor.status is TurnStatus.FAILED
    assert run.cursor.error is not None
    assert run.cursor.error["node"] == "orchestrate"
    if answer is blocked_reply:
        assert isinstance(failed.error, UnexpectedModelBehavior)
    else:
        assert isinstance(failed.error, EmptyResponseError)
        assert run.cursor.error["type"] == "EmptyResponseError"
        assert run.cursor.error["transient"] is True
        assert len(models.orchestrator.requests) == 1


async def test_seed_hands_the_runs_it_carried_on_to_the_agent_and_acknowledges_after_the_commit(
    pg_engine: AsyncEngine,
) -> None:
    acknowledged: list[str] = []
    driver: TurnDriver

    async def acknowledge() -> None:
        cursor = await driver.open_cursor()
        acknowledged.append(cursor.node)

    async def resume(ctx: NodeCtx) -> Resumed:
        assert turn_id(ctx.key) == "conv_01test0000/1"
        hand_back = {"status": "Finishing an earlier calculation", "text": "Run r1 finished."}
        return Resumed(hand_back=hand_back, acknowledge=acknowledge)

    models = scripted_models(orchestrator=[reply(text_part("Run r1 is done."))])
    nodes = opening_nodes(scripted_orchestrator(), models, resume=resume)
    driver = TurnDriver(pg_engine, graph=opening_graph(nodes))

    run = await driver.turn("And the run?")

    assert run.outcome is Outcome.DONE
    assert acknowledged == ["orchestrate"]
    assert run.events[1] == AgentStatus(text="Finishing an earlier calculation")
    assert agent_read(models.orchestrator.requests[0]).endswith("\nAnd the run?\nRun r1 finished.")


async def test_a_given_node_hands_the_turn_specs_that_the_loop_stores_apart_from_state(
    pg_engine: AsyncEngine,
) -> None:
    models = scripted_models(orchestrator=[reply(text_part("Done."))])
    nodes = opening_nodes(scripted_orchestrator(), models)
    before = given(specs={"spec_a": {"spec_id": "spec_a"}}, state={"surface": "studio"})
    driver = TurnDriver(pg_engine, graph=opening_graph(nodes, before=before))

    run = await driver.turn("Hello")

    assert next(e.node for e in run.events if isinstance(e, NodeDone)) == GIVEN
    assert await driver.turns.specs(run.cursor) == {"spec_a": {"spec_id": "spec_a"}}
    assert run.cursor.state[ORCHESTRATOR_SPECS_KEY] == {"spec_a": {"spec_id": "spec_a"}}
    assert await _stored_state(pg_engine, run.cursor.id) == {"surface": "studio"}


def test_a_stored_read_refuses_a_write_into_it() -> None:
    state: dict[str, Any] = {ORCHESTRATOR_SPECS_KEY: {}, ORCHESTRATOR_SEARCHES_KEY: []}

    with pytest.raises(TypeError):
        stored_mapping(state, ORCHESTRATOR_SPECS_KEY)["spec_a"] = {}  # ty: ignore[invalid-assignment]
    with pytest.raises(AttributeError):
        stored_sequence(state, ORCHESTRATOR_SEARCHES_KEY).append({})  # ty: ignore[unresolved-attribute]
    assert state == {ORCHESTRATOR_SPECS_KEY: {}, ORCHESTRATOR_SEARCHES_KEY: []}
