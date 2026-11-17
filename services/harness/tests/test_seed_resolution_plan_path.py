"""Seeded turns on the plan path: a seeded turn paused on a plan's approval,
and a seed a later turn's planner reads.

These run the turn through the planner, `approve` and the Studio routes, so
they move onto the turn loop with those nodes and routes. The seeded turns
`seed` and `orchestrate` run alone are in `test_seed_resolution.py`.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from data_mcp.definition_store import InMemoryDefinitionStore
from google.genai import types
from harness.graph.state import APPROVE_INTERRUPT
from harness.scripted import ScriptedPlannerModel, ScriptedSurface
from harness.seeds import SEED_STEP_TOOL
from planner.plan import SEEDS_STATE_KEY
from sqlalchemy.ext.asyncio import AsyncEngine

from .data_mcp_definitions import DataMcpDefinitions
from .orchestrator_fakes import (
    CannedSearch,
    RefusingPlanner,
    approve_answer,
    call,
    create_request_input_response,
    fetch_call,
    open_session,
    orchestrator_app,
    orchestrator_runner,
    plan_call,
    request_text,
    run_turn,
    single_pause,
    stream,
    studio_client,
    turn_body,
    visited_nodes,
)
from .test_seed_resolution import _KENYA, _QUESTION, _SEEDED, _seeds, _Steps


async def test_a_seed_is_fetched_without_a_search_and_a_later_turns_planner_reads_it(
    definition_store: InMemoryDefinitionStore, pg_engine: AsyncEngine
) -> None:
    """Over the studio routes: the agent fetches the seed's spec_id with no search,
    and the next turn's planner reads the seed among the earlier readings, with the
    fetch it served."""
    del definition_store
    search = CannedSearch()
    planner = RefusingPlanner()
    app, model = orchestrator_app(
        fetch_call(_KENYA["spec_id"]),
        types.Part.from_text(text="Here are the donors."),
        plan_call("rank donors of education aid to Kenya"),
        types.Part.from_text(text="I could not plan that."),
        resolver=search,
        planner=planner,
        engine=pg_engine,
        served=[_KENYA],
    )
    body = {**turn_body(_QUESTION), "seed_spec": _seeds([_KENYA])}

    async with studio_client(app) as client:
        first = await stream(client, "/api/studio/turn", body)
        await stream(client, "/api/studio/turn", turn_body("rank them"))

    steps = [f for f in first if f["event"] == "activity.step" and f.get("tool")]
    assert [(f["tool"], f["status"]) for f in steps if f["status"] != "running"] == [
        ("receive_seeds", "complete"),
        ("fetch", "complete"),
    ]
    fetched = model.requests[1].contents[-1].parts[0].function_response
    assert fetched.name == "fetch"
    assert fetched.response["frames"]
    assert search.calls == []
    (readings,) = planner.readings
    seeded = [r for r in readings if r["spec_id"] == _KENYA["spec_id"]]
    assert [(r["origin"], r["status"], r["fetch"] is not None) for r in seeded] == [
        ("seed", "definite", True)
    ]


class _Describes(_Steps):
    """`_Steps` that answers the `describe_frame` calls an approved plan's run makes."""

    async def __call__(self, *, tool: str, args: dict[str, Any], **_: Any) -> Any:
        assert tool == "describe_frame", tool
        return {"outcome": "ok", "view": {"dimensions": [{"name": "observationAbout"}]}}


def _approval(pause: types.FunctionCall) -> dict[str, Any]:
    assert pause.id == APPROVE_INTERRUPT
    return {"result": approve_answer(pause.args["payload"]["revision"])}


_PAUSES = {
    "plan": (
        lambda: call(
            "plan",
            brief="rank donors of education aid to Kenya",
            asks=["rank donors of education aid to Kenya"],
        ),
        _approval,
    ),
}


@pytest.mark.parametrize("pausing", sorted(_PAUSES))
async def test_a_seeded_turn_takes_its_seeds_once_across_a_pause(
    definition_store: InMemoryDefinitionStore, pausing: str
) -> None:
    """A seeded turn that pauses on `ask_user` or on a plan's approval resumes
    without running `seed` again: one hand-back and one set of its steps
    across the turn, which then completes. The hand-back follows the user's
    question in every request, the resumed one included."""
    del definition_store
    pausing_call, answer = _PAUSES[pausing]
    steps = _Describes()
    runner, model = orchestrator_runner(
        pausing_call(),
        types.Part.from_text(text="Here are the donors."),
        planner=ScriptedPlannerModel(),
        tool_surface=ScriptedSurface(),
        executor=steps,
        definition_surface=DataMcpDefinitions(served=[_KENYA]),
    )
    await open_session(runner, **{SEEDS_STATE_KEY: json.dumps(_seeds([_KENYA]))})

    pause = single_pause(await run_turn(runner, types.UserContent(_QUESTION)))
    reply = create_request_input_response(pause.id, answer(pause))
    resumed = await run_turn(runner, types.Content(role="user", parts=[reply]))

    assert "seed" not in visited_nodes(resumed)
    assert [step for tool, step in steps.recorded if tool == SEED_STEP_TOOL] == ["seeds"]
    assert request_text(model.requests[-1]).count(_SEEDED) == 1
    for request in (model.requests[0], model.requests[-1]):
        texts = ["".join(p.text or "" for p in c.parts or []) for c in request.contents]
        assert _SEEDED in texts[texts.index(_QUESTION) + 1]
    assert not [e for e in resumed if e.error_code]
    assert not [c for e in resumed for c in e.get_function_calls() if c.name == "adk_request_input"]
    assert model.script == []
