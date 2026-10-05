"""harness-planner-skills-v1's scoring and its canned skill surface, with no model.

The set itself runs outside CI (`evals/README.md`). These pin the two pieces it
rests on: `skills_loaded_before_plan` passes a turn only when every required
skill and no forbidden one was loaded before the first `propose_plan`, and `SkillSurface` answers the
planner's real skill tools for every skill a case names.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from google.adk.evaluation.eval_case import IntermediateData, Invocation
from google.genai import types as genai_types
from planner.skills import build_skill_tools

_AGENTS_DIR = Path(__file__).resolve().parents[3] / "evals" / "agents"
if str(_AGENTS_DIR) not in sys.path:
    sys.path.insert(0, str(_AGENTS_DIR))

from harness_planner_skills_v1.cases import CASES  # noqa: E402
from harness_planner_skills_v1.skill_metric import turn_score  # noqa: E402
from harness_planner_skills_v1.skill_surface import SkillSurface  # noqa: E402


def _skill(skill_id: str) -> genai_types.FunctionCall:
    return genai_types.FunctionCall(name="get_skill", args={"skill_id": skill_id})


_PLAN = genai_types.FunctionCall(name="propose_plan", args={"decisions": []})


def _turn(calls: list[genai_types.FunctionCall]) -> Invocation:
    return Invocation(
        user_content=genai_types.Content(role="user", parts=[genai_types.Part(text="q")]),
        intermediate_data=IntermediateData(tool_uses=calls),
    )


@pytest.mark.parametrize(
    ("required", "forbidden", "actual", "score"),
    [
        (["oda-crs", "oda-measures"], [], [_skill("oda-measures"), _skill("oda-crs"), _PLAN], 1.0),
        (["oda-crs"], [], [_skill("oda-dac1"), _skill("oda-crs"), _PLAN, _PLAN], 1.0),
        (["oda-crs", "oda-measures"], [], [_skill("oda-measures"), _PLAN, _skill("oda-crs")], 0.0),
        (["oda-crs"], [], [_PLAN, _skill("oda-crs"), _PLAN], 0.0),
        (["oda-crs"], [], [_skill("oda-crs")], 0.0),
        ([], [], [_skill("oda-dac1"), _PLAN], 1.0),
        ([], [], [], 0.0),
        (
            ["oda-crs"],
            ["policy-markers"],
            [_skill("policy-markers"), _skill("oda-crs"), _PLAN],
            0.0,
        ),
        (
            ["oda-crs"],
            ["policy-markers"],
            [_skill("oda-crs"), _PLAN, _skill("policy-markers")],
            1.0,
        ),
    ],
    ids=[
        "both-before",
        "unjudged-skill-allowed",
        "second-after-plan",
        "loaded-after-first-plan",
        "no-plan",
        "none-required",
        "none-required-no-plan",
        "forbidden-before-plan",
        "forbidden-after-plan",
    ],
)
def test_a_turn_passes_when_the_first_plan_follows_every_required_load_and_no_forbidden_one(
    required: list[str],
    forbidden: list[str],
    actual: list[genai_types.FunctionCall],
    score: float,
) -> None:
    expected = _turn([_skill(skill_id) for skill_id in required])

    assert turn_score(_turn(actual), expected, forbidden=forbidden) == score


def test_every_case_forbids_only_skills_its_envelope_names_and_does_not_require() -> None:
    misnamed = [
        case.eval_id
        for case in CASES
        if not set(case.forbidden) <= set(case.skills) - set(case.required)
    ]

    assert misnamed == []


@pytest.mark.parametrize("skill_id", sorted({s for case in CASES for s in case.skills}))
async def test_the_planner_skill_tools_load_every_skill_a_case_names(skill_id: str) -> None:
    get_skill, get_skill_file = (tool.func for tool in build_skill_tools(SkillSurface()))

    loaded = await get_skill(skill_id)

    assert loaded.startswith(f"---\nname: {skill_id}\n")
    assert await get_skill_file(skill_id, "SKILL.md") in loaded
