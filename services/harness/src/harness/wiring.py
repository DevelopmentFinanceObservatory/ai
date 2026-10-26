"""Assembly of the planner seam for the live server.

The one place that knows which implementation stands behind the workflow's
`PlannerModel`: the planner-translator pair behind `harness.planner_seam`.
`harness.server.create_app` calls this.
"""

from __future__ import annotations

import logging
from typing import Any

from compiler.surface import ToolSurface

from harness.graph.seams import PlannerModel
from harness.planner_seam import PlannerSeam, PlannerSeamError
from harness.prefix_cache import PrefixCache
from harness.settings import Settings

logger = logging.getLogger(__name__)


def build_planner(
    *,
    settings: Settings,
    surface: ToolSurface,
    prefix_cache: PrefixCache | None = None,
) -> PlannerSeam:
    """The planner seam: `planner.agent`'s planner with the translation from
    `translator.tool.build_translate` attached to `propose_plan`, adapted by
    `harness.planner_seam.PlannerSeam`. `surface` is the tool surface the
    translation discovers on, the compiler checks against, and the planner's
    `get_skill` and `get_skill_file` call through. Pass the one
    `harness.server.create_app` already built for the `execute` node, so the
    two compile against one set of schemas, read once. `prefix_cache` is the
    process's shared prompt-prefix cache, passed on to the planner's model.
    """
    return PlannerSeam(settings=settings, surface=surface, prefix_cache=prefix_cache)


def handing_back_planner_failures(seam: PlannerModel) -> PlannerModel:
    """`seam`, answering a draft or a revision it cannot finish (`PlannerSeamError`)
    with a refusal instead of raising. The planner node then takes its "refused"
    route and `plan_done` hands the reason back to the agent, which answers
    with what it can; the `switch` node offers the plan unchanged. The refusal
    states the failure's code, which words the failed step the user sees. The
    failure is logged in full."""
    return _HandingBackFailures(seam)


class _HandingBackFailures:
    """The `PlannerModel` `handing_back_planner_failures` returns."""

    def __init__(self, seam: PlannerModel) -> None:
        self._seam = seam

    async def __call__(self, **kwargs: Any) -> dict[str, Any]:
        try:
            return await self._seam(**kwargs)
        except PlannerSeamError as exc:
            return _handed_back(exc)

    async def revise(self, **kwargs: Any) -> dict[str, Any]:
        try:
            return await self._seam.revise(**kwargs)
        except PlannerSeamError as exc:
            return _handed_back(exc)


def _handed_back(exc: PlannerSeamError) -> dict[str, Any]:
    """The refusal a `PlannerSeamError` is answered with, logged in full."""
    logger.error("planner seam failed; handing the reason back to the orchestrator: %s", exc)
    return {"refused": {"text": str(exc), "code": exc.refusal.get("code")}}


__all__ = ["build_planner", "handing_back_planner_failures"]
