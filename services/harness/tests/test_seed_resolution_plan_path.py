"""A seed a later turn's planner reads, over the Studio routes.

It runs two turns through the Studio routes, so it moves onto the turn loop
with them. The other seeded turns are in `test_seed_resolution.py`.
"""

from __future__ import annotations

from data_mcp.definition_store import InMemoryDefinitionStore
from google.genai import types
from sqlalchemy.ext.asyncio import AsyncEngine

from .orchestrator_fakes import (
    CannedSearch,
    RefusingPlanner,
    fetch_call,
    orchestrator_app,
    plan_call,
    stream,
    studio_client,
    turn_body,
)
from .test_seed_resolution import _KENYA, _QUESTION, _seeds


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
