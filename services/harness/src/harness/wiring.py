"""Assembly of the planner adapter for the live server.

The one place that knows which implementation stands behind the workflow's
`PlannerModel`: the planner-translator pair behind `harness.planner_adapter`.
`harness.server.create_app` calls this.
"""

from __future__ import annotations

import logging
from typing import Any

from compiler.surface import ToolSurface

from harness.graph.interfaces import PlannerModel
from harness.planner_adapter import PlannerAdapter, PlannerAdapterError
from harness.prefix_cache import PrefixCache
from harness.settings import Settings

logger = logging.getLogger(__name__)


def build_planner(
    *,
    settings: Settings,
    surface: ToolSurface,
    prefix_cache: PrefixCache | None = None,
) -> PlannerAdapter:
    """The planner adapter: the planner with the translation from
    `translator.tool.build_translate` attached to `propose_plan`, adapted by
    `harness.planner_adapter.PlannerAdapter`. `surface` is the tool surface the
    translation discovers on, the compiler checks against, and the planner's
    `get_skill` calls through. Pass the one
    `harness.server.create_app` already built for the `execute` node, so the
    two compile against one set of schemas, read once. `prefix_cache` is the
    process's shared prompt-prefix cache, passed on to the planner's model.
    """
    return PlannerAdapter(settings=settings, surface=surface, prefix_cache=prefix_cache)


def handing_back_planner_failures(planner: PlannerModel) -> PlannerModel:
    """`planner`, answering a draft or a revision it cannot finish (`PlannerAdapterError`)
    with a refusal instead of raising. The planner node then takes its "refused"
    route and `plan_done` hands the reason back to the agent, which answers
    with what it can; the `switch` node offers the plan unchanged. The refusal
    states the failure's code, which words the failed step the user sees. The
    failure is logged in full."""
    return _HandingBackFailures(planner)


class _HandingBackFailures:
    """The `PlannerModel` `handing_back_planner_failures` returns."""

    def __init__(self, planner: PlannerModel) -> None:
        self._planner = planner

    async def __call__(self, **kwargs: Any) -> dict[str, Any]:
        try:
            return await self._planner(**kwargs)
        except PlannerAdapterError as exc:
            return _handed_back(exc)

    async def revise(self, **kwargs: Any) -> dict[str, Any]:
        try:
            return await self._planner.revise(**kwargs)
        except PlannerAdapterError as exc:
            return _handed_back(exc)


def _handed_back(exc: PlannerAdapterError) -> dict[str, Any]:
    """The refusal a `PlannerAdapterError` is answered with, logged in full."""
    logger.error("planner failed; handing the reason back to the orchestrator: %s", exc)
    return {"refused": {"text": str(exc), "code": exc.refusal.get("code")}}


__all__ = ["build_planner", "handing_back_planner_failures"]
