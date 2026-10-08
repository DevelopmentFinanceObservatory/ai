"""The harness-planner-skills-v1 cases, and the eval set built from them.

Each case is a question, the resolve envelope the search step hands the
planner for it, and the skills the planner has to load before it drafts. The
user message is the brief `harness.planner_seam` renders for the served
planner (`_envelope_context`, then the question), so the envelope's `skills`
reach the model in the words and order the harness uses.

`harness-planner-skills-v1.evalset.json` is built from `CASES` by
`eval_set()`. Edit a case here, then rewrite the file:

    PYTHONPATH=evals/agents uv run python -m harness_planner_skills_v1.cases

`services/harness/tests/test_eval_sets.py` fails while the file and `CASES`
disagree.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from google.adk.evaluation.eval_case import Invocation
from harness.planner_seam import _envelope_context

from harness_planner_skills_v1.skill_surface import SKILLS

EVAL_SET_ID = "harness-planner-skills-v1"
EVAL_SET_PATH = Path(__file__).resolve().parent / f"{EVAL_SET_ID}.evalset.json"

_DESCRIPTION = (
    "Whether the planner loads the skill a question needs before it drafts. Each case "
    "hands the planner the brief the harness renders: a resolve envelope whose `skills` "
    "names candidate skills, then the question. The skills_loaded_before_plan metric "
    "reads the recorded tool calls and passes a turn when propose_plan was called and every skill "
    "the turn's tool_uses names was loaded with get_skill before the first propose_plan, "
    "and no skill the case forbids (cases.py) was. A named skill in neither set may be "
    "loaded or skipped. workflow_outcome "
    "checks that a propose_plan was accepted. No judge model scores this set."
)


@dataclass(frozen=True)
class Case:
    """One case: the question, what the resolve found, and the skills it needs.

    `required` skills have to be loaded before the first `propose_plan`, and
    `forbidden` ones must not be. A named skill in neither set may be loaded
    or skipped: it is one that plausibly helps without being needed.
    """

    eval_id: str
    question: str
    spec_ids: tuple[str, ...]
    skills: tuple[str, ...]
    required: tuple[str, ...]
    forbidden: tuple[str, ...]

    def envelope(self) -> dict[str, Any]:
        """The envelope `harness.tool_contracts.graph_envelope` returns for a definite resolve."""
        return {
            "status": "definite",
            "spec_id": self.spec_ids[0],
            "choice_points": 0,
            "parts": len(self.spec_ids),
            "wide": False,
            "warnings": [],
            "candidates": [],
            "skills": [
                {"id": skill_id, "description": SKILLS[skill_id].description}
                for skill_id in self.skills
            ],
        }

    def brief(self) -> str:
        """The planner's user message, as `PlannerSeam._message` renders a fresh turn."""
        return _envelope_context(self.envelope()) + f"\nQuestion: {self.question}\n"


def _spec(digit: str) -> str:
    return "spec_" + digit * 32


CASES = (
    Case(
        eval_id="grant-equivalent-measure",
        question=(
            "How much ODA did Germany give in 2023 on a grant-equivalent basis, and how "
            "does that compare with its net disbursements?"
        ),
        spec_ids=(_spec("1"),),
        skills=("oda-dac1", "oda-measures", "oda-policy-markers"),
        required=("oda-measures",),
        forbidden=("oda-policy-markers",),
    ),
    Case(
        eval_id="oda-gni-target",
        question="Which DAC members met the 0.7% ODA/GNI target in 2023?",
        spec_ids=(_spec("2"),),
        skills=("oda-dac1", "oda-concepts", "oda-policy-markers"),
        required=("oda-concepts",),
        forbidden=("oda-policy-markers",),
    ),
    Case(
        eval_id="gender-marker-share",
        question=(
            "What share of the UK's bilateral ODA in 2022 had gender equality as its "
            "principal objective?"
        ),
        spec_ids=(_spec("3"),),
        skills=("oda-crs", "oda-policy-markers", "oda-dac2a"),
        required=("oda-policy-markers",),
        # DAC2A carries no policy marker, and the share's denominator is the
        # UK's screened CRS activities, so it has nothing to add here.
        forbidden=("oda-dac2a",),
    ),
    Case(
        eval_id="crs-purpose-code-sector",
        question=(
            "How much bilateral aid did donors give for basic education in Kenya in 2022, "
            "using the CRS purpose codes, broken down by donor?"
        ),
        spec_ids=(_spec("4"),),
        skills=("oda-crs", "oda-dac2a", "oda-policy-markers"),
        required=("oda-crs",),
        forbidden=("oda-policy-markers",),
    ),
    Case(
        eval_id="grant-equivalent-and-purpose-codes",
        question=(
            "Compare France's grant-equivalent ODA in 2022 with the amount of its "
            "bilateral aid that went to health, by CRS purpose code."
        ),
        spec_ids=(_spec("5"), _spec("6")),
        skills=("oda-crs", "oda-measures", "oda-dac1", "oda-policy-markers"),
        required=("oda-crs", "oda-measures"),
        forbidden=("oda-policy-markers",),
    ),
    Case(
        eval_id="population-needs-no-skill",
        question="What was Nigeria's total population in 2020?",
        spec_ids=(_spec("7"),),
        skills=("oda-policy-markers", "oda-dac2a"),
        required=(),
        forbidden=("oda-policy-markers", "oda-dac2a"),
    ),
)


def _turn(case: Case) -> dict[str, Any]:
    return {
        "invocation_id": "ask",
        "user_content": {"role": "user", "parts": [{"text": case.brief()}]},
        "intermediate_data": {
            "tool_uses": [
                {"name": "get_skill", "args": {"skill_id": skill_id}} for skill_id in case.required
            ],
            "tool_responses": [{"name": "propose_plan", "response": {"outcome": "ok"}}],
        },
    }


def case_for(invocation: Invocation) -> Case:
    """The case whose brief is `invocation`'s user message.

    Raises:
        LookupError: If no case renders that message, as when the eval set
            and `CASES` disagree.
    """
    parts = invocation.user_content.parts or []
    text = "".join(part.text or "" for part in parts)
    for case in CASES:
        if case.brief() == text:
            return case
    raise LookupError("no case in CASES renders this user message; rewrite the eval set")


def eval_set() -> dict[str, Any]:
    """The eval set `CASES` describe, as `adk eval` reads it."""
    return {
        "eval_set_id": EVAL_SET_ID,
        "name": EVAL_SET_ID,
        "description": _DESCRIPTION,
        "eval_cases": [
            {
                "eval_id": case.eval_id,
                "conversation": [_turn(case)],
                "session_input": {
                    "app_name": "harness_planner_skills_v1",
                    "user_id": "eval-user",
                    "state": {},
                },
            }
            for case in CASES
        ],
    }


if __name__ == "__main__":
    EVAL_SET_PATH.write_text(json.dumps(eval_set(), indent=2, ensure_ascii=False) + "\n")
