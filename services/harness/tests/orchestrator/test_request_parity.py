"""The Gemini requests the Pydantic AI orchestrator sends, against ADK's recorded ones.

Each scenario replays a turn `fixtures/orchestrator_requests` recorded from the
ADK orchestrator (`pai_runs.SCENARIOS`), with the same model replies and the
same hand-back, its stored history serialised to JSON and back across the
pause, and compares every request field by field. A difference must be one
`_known` accepts:

- HTTP headers and the request's retry options, which configure the client and
  never reach the model;
- the system instruction's trailing newline, which Pydantic AI strips;
- the placeholder signature Pydantic AI gives the first function call of a
  replayed response that Gemini sent unsigned (the scripted replies sign
  none; Gemini signs the first call of a thinking response).
"""

from __future__ import annotations

import base64
import copy
from typing import Any

import pytest

from .pai_runs import SCENARIOS, recorded, run_scenario

_PLACEHOLDER = base64.b64encode(b"skip_thought_signature_validator").decode()


def _dump(request: Any) -> dict[str, Any]:
    return {
        "model": request.model,
        "contents": [c.model_dump(mode="json", exclude_none=True) for c in request.content_list],
        "config": request.generate_config.model_dump(mode="json", exclude_none=True),
    }


def diff(left: Any, right: Any, path: str = "") -> list[tuple[str, Any, Any]]:
    """Field-level differences between two JSON values, ADK's on the left."""
    if isinstance(left, dict) and isinstance(right, dict):
        found = []
        for key in sorted(set(left) | set(right), key=str):
            where = f"{path}.{key}" if path else str(key)
            found += diff(left.get(key, "<absent>"), right.get(key, "<absent>"), where)
        return found
    if isinstance(left, list) and isinstance(right, list) and len(left) == len(right):
        return [
            h
            for i, (a, b) in enumerate(zip(left, right, strict=True))
            for h in diff(a, b, f"{path}[{i}]")
        ]
    return [] if left == right else [(path, left, right)]


def _known(path: str, adk: Any, pai: Any) -> bool:
    if path.startswith("config.http_options."):
        return True
    if path == "config.system_instruction":
        return adk == f"{pai}\n"
    if path.startswith("contents[") and path.endswith(".thought_signature"):
        return adk == "<absent>" and pai == _PLACEHOLDER
    return False


def _unexplained(adk: list[dict[str, Any]], pai: list[dict[str, Any]]) -> list[Any]:
    assert len(adk) == len(pai)
    return [
        (index, path, left, right)
        for index, (a, b) in enumerate(zip(adk, pai, strict=True))
        for path, left, right in diff(a, b)
        if not _known(path, left, right)
    ]


@pytest.mark.parametrize("scenario", list(SCENARIOS))
async def test_every_request_is_adk_s_but_for_the_known_differences(scenario: str) -> None:
    scripted, _, _ = await run_scenario(scenario, SCENARIOS[scenario])

    assert _unexplained(recorded(scenario), [_dump(r) for r in scripted.requests]) == []


async def test_a_hand_back_shown_otherwise_is_caught() -> None:
    """Control: a hand-back the model reads in other words is a difference."""
    scripted, _, _ = await run_scenario("ask", SCENARIOS["ask"])
    adk = copy.deepcopy(recorded("ask"))
    for content in adk[-1]["contents"]:
        for part in content["parts"]:
            if "function_response" in part:
                part["function_response"]["response"]["result"] += " Changed."

    found = _unexplained(adk, [_dump(r) for r in scripted.requests])

    assert [path for _, path, _, _ in found] == [
        "contents[3].parts[0].function_response.response.result"
    ]
