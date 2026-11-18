"""The orchestrator's turns over the Studio routes: a plain answer, and a
search, fetch and chart that lands a card."""

from __future__ import annotations

from google.genai import types
from sqlalchemy.ext.asyncio import AsyncEngine

from .graph_turns import agent_read
from .scripted_tools import (
    CannedSearch,
    chart,
    definite_envelope,
    fetch_call,
    make_charts_call,
    search_call,
)
from .studio_routes import frame_for, orchestrator_app, stream, studio_client, turn_body


async def test_a_plain_text_turn_streams_the_agents_answer_over_the_studio_routes(
    pg_engine: AsyncEngine,
) -> None:
    app, model = orchestrator_app(
        types.Part.from_text(text="Kenya's GDP was $101 billion."), engine=pg_engine
    )

    async with studio_client(app) as client:
        frames = await stream(client, "/api/studio/turn", turn_body("Kenya's GDP in 2020?"))

    assert [f["event"] for f in frames][-4:] == [
        "message.delta",
        "message.complete",
        "suggestions",
        "done",
    ]
    assert frames[-3]["text"] == "Kenya's GDP was $101 billion."
    assert len(model.requests) == 1
    assert "Kenya's GDP in 2020?" in agent_read(model.requests[0])


async def test_a_search_fetch_and_chart_turn_lands_a_card_over_the_studio_routes(
    pg_engine: AsyncEngine,
) -> None:
    search = CannedSearch(definite_envelope())
    app, model = orchestrator_app(
        search_call("life expectancy"),
        fetch_call(),
        make_charts_call(chart(frame_for())),
        types.Part.from_text(text="Life expectancy in Kenya rose to 62 years."),
        resolver=search,
        engine=pg_engine,
    )

    async with studio_client(app) as client:
        frames = await stream(client, "/api/studio/turn", turn_body("Life expectancy in Kenya?"))

    assert [(c["question"], c["parts"]) for c in search.calls] == [
        (None, [{"variable": "life expectancy"}])
    ]
    tools = [f.get("tool") for f in frames if f["event"] == "activity.step"]
    assert {"search", "fetch", "make_charts", "write_component"} <= set(tools)
    created = [f for f in frames if f["event"] == "component.created"]
    assert [f["component_id"] for f in created] == ["cmp_scripted0002"]
    assert frames[-1]["event"] == "done"
    complete = next(f for f in frames if f["event"] == "message.complete")
    assert complete["text"] == "Life expectancy in Kenya rose to 62 years."
    drawn = model.requests[-1].contents[-1].parts[0].function_response
    assert drawn.name == "make_charts"
    (card,) = drawn.response["charts"]
    assert card["holds"]["series"] == ["Kenya"]
    assert card["reply_figures"] == [
        {"series": "Kenya", "value": 63.6, "unit": "Year", "period": "2023"}
    ]
