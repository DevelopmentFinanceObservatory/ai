"""The Gemini requests the orchestrator sends on the Hydrogen surface, against recorded ones.

`fixtures/orchestrator_requests/adk_*.json` pin Flash's bytes
(`test_request_parity.py`, `test_declarations.py`). `hydrogen_*.json` beside them
pin Hydrogen's: each scenario of `pai_runs.SCENARIOS` replayed with the same model
replies, search results and hand-back, the refused plan's next step worded as
`plan_done` words it on Hydrogen, and each tool's declaration.

A request is stored without its HTTP options, which configure the client and
never reach the model. Rewrite the files after a change that moves what the
orchestrator sends on Hydrogen, then run the tests again without the variable:

    ORCHESTRATOR_REQUESTS=write \\
        uv run --package harness pytest services/harness/tests/orchestrator/test_hydrogen_requests.py

A write run fails each test whose file it changed.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest
from harness.graph.plan_done import _refusal_step
from harness.orchestrator_tools import TOOL_NAMES, build_orchestrator_tools
from harness.surface import FLASH_SURFACE, HYDROGEN_SURFACE, set_studio_surface, studio_surface
from orchestrator.declarations import tool_declaration
from orchestrator.run import HandBack

from .pai_runs import RECORDED, SCENARIOS, hand_back_of, run_scenario
from .test_request_parity import diff

pytestmark = pytest.mark.usefixtures("hydrogen_studio")

_WRITING = os.environ.get("ORCHESTRATOR_REQUESTS") == "write"


def _dump(request: Any) -> dict[str, Any]:
    config = request.generate_config.model_dump(mode="json", exclude_none=True)
    config.pop("http_options", None)
    return {
        "model": request.model,
        "contents": [c.model_dump(mode="json", exclude_none=True) for c in request.content_list],
        "config": config,
    }


def _hydrogen_hand_back(scenario: str) -> HandBack | None:
    """ADK's recorded hand-back, with a refused plan's next step in Hydrogen's
    words, as `plan_done` hands it back on that surface."""
    hand_back = hand_back_of(scenario)
    if hand_back is None or hand_back.answers != "plan":
        return hand_back
    surface = studio_surface()
    try:
        set_studio_surface(FLASH_SURFACE)
        flash = _refusal_step()
        set_studio_surface(HYDROGEN_SURFACE)
        hydrogen = _refusal_step()
    finally:
        set_studio_surface(surface)
    text = hand_back.result["result"]
    assert text.count(flash) == 1
    return HandBack(answers="plan", result={"result": text.replace(flash, hydrogen)})


async def _unused(**kwargs: Any) -> Any:
    raise AssertionError("no call expected")


def _declarations() -> dict[str, dict[str, Any]]:
    tools = build_orchestrator_tools(resolver=_unused, executor=_unused)
    return {
        tool.__name__: tool_declaration(
            tool,
            variant=HYDROGEN_SURFACE.variant,
            descriptions=HYDROGEN_SURFACE.tool_descriptions,
        ).model_dump(mode="json", exclude_none=True)
        for tool in tools
    }


def _check(path: Path, produced: Any) -> None:
    """Hold `produced` against `path`, or write it there on a write run."""
    stored = json.loads(path.read_text()) if path.exists() else None
    if _WRITING and stored != produced:
        path.write_text(json.dumps(produced, indent=2, ensure_ascii=False) + "\n")
        pytest.fail(f"wrote {path.name}")
    assert stored is not None, f"{path.name} is missing; record it with ORCHESTRATOR_REQUESTS=write"
    assert diff(stored, produced) == []


@pytest.mark.parametrize("scenario", list(SCENARIOS))
async def test_every_request_is_the_recorded_one(scenario: str) -> None:
    scripted, _, _ = await run_scenario(
        scenario, SCENARIOS[scenario], hand_back=_hydrogen_hand_back(scenario)
    )

    _check(RECORDED / f"hydrogen_{scenario}.json", [_dump(r) for r in scripted.requests])


def test_each_declaration_is_the_recorded_one() -> None:
    built = _declarations()

    assert list(built) == list(TOOL_NAMES)
    _check(RECORDED / "hydrogen_declarations.json", built)


@pytest.mark.parametrize("scenario", list(SCENARIOS))
def test_every_recorded_request_sends_the_recorded_declarations(scenario: str) -> None:
    declarations = json.loads((RECORDED / "hydrogen_declarations.json").read_text())

    for request in json.loads((RECORDED / f"hydrogen_{scenario}.json").read_text()):
        (tool,) = request["config"]["tools"]
        assert tool["function_declarations"] == [declarations[n] for n in TOOL_NAMES]
