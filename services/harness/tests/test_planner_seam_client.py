"""The google-genai client `PlannerSeam(client=...)` routes the ADK planner's model through.

The planner entry eval (`evals/planner_entry`) passes the seam a client on either
runner and reads every model call off it, so the ADK planner's Gemini model has to call
through that client and stay the budget or prefix-cache model the seam built, and the
seam has to refuse what would leave the client unused.
"""

from __future__ import annotations

from typing import Any

import pytest
from compiler.surface import FixtureSurface
from google.genai import Client
from harness.planner_seam import PlannerSeam
from harness.prefix_cache import PrefixCache, PrefixCachedGemini
from harness.settings import Settings
from harness.turn_budget import BoundedGemini
from planner.settings import AgentSettings


@pytest.mark.parametrize(
    ("prefix_cache", "model_type"),
    [(None, BoundedGemini), (PrefixCache(), PrefixCachedGemini)],
    ids=["uncached", "cached"],
)
def test_the_adk_planners_model_calls_through_the_given_client(
    prefix_cache: PrefixCache | None, model_type: type[Any]
) -> None:
    client = Client(api_key="scripted")

    seam = PlannerSeam(
        settings=Settings(planner_runner="adk"),
        surface=FixtureSurface(schemas=[]),
        prefix_cache=prefix_cache,
        client=client,
    )

    assert seam._agent is not None
    model = seam._agent.model
    assert type(model) is model_type
    assert model.api_client is client


@pytest.mark.parametrize(
    "refused",
    [
        {"model": BoundedGemini(model="gemini-2.5-flash")},
        {"agent_settings": AgentSettings(planner_model_name="not-a-gemini-model")},
    ],
    ids=["beside-a-model", "on-another-provider"],
)
def test_the_adk_planner_refuses_a_client_it_would_leave_unused(refused: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="client"):
        PlannerSeam(
            settings=Settings(planner_runner="adk"),
            surface=FixtureSurface(schemas=[]),
            client=Client(api_key="scripted"),
            **refused,
        )
