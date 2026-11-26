"""Drives the Pydantic AI orchestrator over a scripted google-genai client.

`run_scenario` replays one of the four turns `fixtures/orchestrator_requests`
recorded from the ADK orchestrator: the same model replies, the same search
result, the same hand-back. It returns every request the client was sent and
every event the run yielded, so a test can hold them against ADK's.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from datetime import date
from pathlib import Path
from typing import Any

from harness.orchestrator_tools import build_orchestrator_tools
from harness.scripted import SCRIPTED_SPEC
from harness.tool_contracts import resolver_digest
from harness.turn_budget import TurnBudget
from orchestrator.agent import orchestrator_agent
from orchestrator.model import orchestrator_model
from orchestrator.run import (
    HandBack,
    HandedOff,
    Opening,
    OrchestratorEvent,
    run_orchestrator,
)

from ..scripted_genai import ScriptedGenai, Step, call_part, reply, text_part, thought_part

RECORDED = Path(__file__).parent.parent / "fixtures" / "orchestrator_requests"
TODAY = date(2026, 11, 17)
DIGESTS = {
    "canvas_digest": json.dumps({"cards": 2, "hidden": 0}),
    "conversation_digest": json.dumps([{"turn": 1, "asked": "Kenya's GDP"}]),
}
# The model replies of each recorded turn, as the recorder scripted them.
SCENARIOS: dict[str, list[Any]] = {
    "answer": [
        reply(
            thought_part("Search first."),
            call_part("search", {"parts": [{"variable": "life expectancy"}]}),
        ),
        reply(text_part("Life expectancy in Kenya "), text_part("was 62 years.")),
    ],
    "plan": [
        reply(call_part("search", {"parts": [{"variable": "life expectancy"}]})),
        reply(
            thought_part("A ranking needs a plan."),
            call_part("plan", {"brief": "rank Kenya", "asks": ["rank"]}),
        ),
        reply(text_part("The plan could not be drafted.")),
    ],
    "ask": [
        reply(
            thought_part("Which Congo?", signature=b"sig-ask"),
            call_part(
                "ask_user",
                {
                    "questions": [
                        {
                            "question": "Which Congo?",
                            "options": [{"label": "Republic of the Congo"}, {"label": "DR Congo"}],
                        }
                    ]
                },
            ),
        ),
        reply(text_part("The Republic of the Congo it is.")),
    ],
    "ask_search": [
        reply(
            call_part("search", {"parts": [{"variable": "life expectancy"}]}),
            call_part(
                "ask_user",
                {
                    "questions": [
                        {
                            "question": "Which years?",
                            "options": [{"label": "2021 to the latest"}, {"label": "2015 to 2020"}],
                        }
                    ]
                },
            ),
        ),
        reply(text_part("From 2021, life expectancy rose.")),
    ],
}

QUESTIONS = {
    "answer": "Life expectancy in Kenya?",
    "plan": "Rank Kenya's life expectancy",
    "ask": "GDP of Congo?",
    "ask_search": "Life expectancy in Kenya since the election?",
}


def recorded(scenario: str) -> list[dict[str, Any]]:
    return json.loads((RECORDED / f"flash_{scenario}.json").read_text())


def definite_envelope(spec_id: str = SCRIPTED_SPEC) -> dict[str, Any]:
    """A definite search result for life expectancy in Kenya, read as `spec_id`."""
    spec = {
        "spec_id": spec_id,
        "resolved": {
            "rendered_sentence": "Life expectancy in Kenya",
            "confidence": "high",
            "coverage": {
                "kind": "exact",
                "has_data": True,
                "window": {"start_year": 1960, "end_year": 2023},
            },
            "stat_vars": {
                "member_count": 1,
                "member_count_exact": True,
                "served_count": 1,
                "data_date_range": {"start": "1960", "end": "2023"},
            },
            "entities": [{"role": "observationAbout", "label": "Kenya", "ref": "country/KEN"}],
            "facets": {},
        },
        "definition": {
            "schema_version": 1,
            "spec_id": spec_id,
            "identity": {"stat_var": "LifeExpectancy_Person"},
        },
    }
    part = {
        "status": "definite",
        "variable_text": "Life expectancy in Kenya",
        "warnings": [],
        "spec": spec,
    }
    return {"schema_version": 1, "kind": "resolution", "status": "definite", "parts": [part]}


class CannedSearch:
    def __init__(self, *envelopes: Mapping[str, Any]) -> None:
        self._envelopes = [dict(e) for e in envelopes]

    async def __call__(self, **kwargs: Any) -> dict[str, Any]:
        return resolver_digest(self._envelopes.pop(0))


async def _no_executor(**kwargs: Any) -> Any:
    raise AssertionError("no executor call expected")


def hand_back_of(scenario: str) -> HandBack | None:
    """The hand-back ADK's last request showed as the deferred call's result."""
    tool = {"plan": "plan", "ask": "ask_user", "ask_search": "ask_user"}.get(scenario)
    if tool is None:
        return None
    last = recorded(scenario)[-1]
    (result,) = [
        part["function_response"]["response"]
        for content in last["contents"]
        for part in content["parts"]
        if part.get("function_response", {}).get("name") == tool
    ]
    return HandBack(answers=tool, result=result)


async def run_scenario(
    scenario: str,
    replies: Sequence[Step],
    *,
    budget: TurnBudget | None = None,
    store: bool = True,
    hand_back: HandBack | None = None,
) -> tuple[ScriptedGenai, list[OrchestratorEvent], dict[str, Any]]:
    """Run `scenario`'s turn on the Pydantic AI orchestrator: its first run, and
    when it hands off, a second run from the stored history, serialised to JSON
    and back when `store`. The second run reads `hand_back`, by default the one
    ADK's recording showed (`hand_back_of`)."""
    scripted = ScriptedGenai(list(replies))
    tools = build_orchestrator_tools(
        resolver=CannedSearch(definite_envelope(), definite_envelope()), executor=_no_executor
    )
    orchestrator = orchestrator_agent(tools)
    state: dict[str, Any] = {"surface": "studio", **DIGESTS}

    def model() -> Any:
        return orchestrator_model("gemini-test", client=scripted.client, budget=budget)

    events: list[OrchestratorEvent] = [
        event
        async for event in run_orchestrator(
            orchestrator,
            history=None,
            state=state,
            turn_id="turn-1",
            model=model(),
            opening=Opening(question=QUESTIONS[scenario], today=TODAY, **DIGESTS),
        )
    ]
    handed = [e for e in events if isinstance(e, HandedOff)]
    if hand_back is None:
        hand_back = hand_back_of(scenario)
    if handed and hand_back is not None:
        history = handed[0].history
        if store:
            history = type(history).model_validate_json(history.model_dump_json())
        events += [
            event
            async for event in run_orchestrator(
                orchestrator,
                history=history,
                state=state,
                turn_id="turn-1",
                model=model(),
                hand_back=hand_back,
            )
        ]
    return scripted, events, state
