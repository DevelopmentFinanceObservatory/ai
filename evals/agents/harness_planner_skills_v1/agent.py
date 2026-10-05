"""Backs the harness-planner-skills-v1 eval set: the served planner, minus its translator.

Score it with `just eval-planner-skills`. The planner is built as the
harness graph's seam builds it (`harness.agents.create_planner_with_translator`
with `question_fixed=True`) in every respect but one: the translator is left
out, so a case ends at `propose_plan`. With a tool surface it carries
`get_skill` and `get_skill_file`, and its instruction carries the same
skills paragraph the served planner reads. The surface is
`skill_surface.SkillSurface`, which answers both tools from fixed skills and
reaches no server.
"""

from __future__ import annotations

from planner.agent import create_planner_agent
from planner.settings import AgentSettings

from .skill_surface import SkillSurface

# The same field the deployed planner is built from; `PLANNER_MODEL_NAME`
# overrides it here exactly as it does there.
_MODEL_NAME = AgentSettings().planner_model_name

root_agent = create_planner_agent(
    model_name=_MODEL_NAME, surface=SkillSurface(), question_fixed=True
)
