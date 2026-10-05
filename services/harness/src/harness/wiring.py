"""Assembly of the graph's model seams for the live server.

The one place that knows which implementation stands behind each seam: the
constrained Gemini calls in `harness.generative` and the planner-translator
pair behind `harness.planner_seam`. `harness.server.create_app` calls
this; tests and the scripted mode build their own `GenerativeNodes`.
"""

from __future__ import annotations

import logging
from typing import Any

from compiler.surface import ToolSurface
from google.genai import Client

from harness.generative import GeminiCompose, GeminiConverse, GeminiNarrate, GeminiRefusal
from harness.graph import GenerativeNodes, PlannerModel
from harness.planner_seam import PlannerSeam, PlannerSeamError
from harness.settings import Settings

logger = logging.getLogger(__name__)


def build_generative_nodes(
    *,
    settings: Settings,
    model_name: str,
    client: Client | None = None,
    surface: ToolSurface | None = None,
) -> GenerativeNodes:
    """The seams as a `GenerativeNodes`.

    The model seams here are single constrained calls that build their genai
    client lazily on first use; pass `client` to share one across them. The
    planner is `planner.agent`'s planner with `translator.agent`'s translator
    attached, adapted by `harness.planner_seam.PlannerSeam`. `surface`
    is the tool surface the translator discovers on, the compiler checks
    against, the planner's `get_skill` and `get_skill_file` call through,
    and converse reads dfo-faq through for a reply about DFO itself;
    pass the one `harness.server.create_app` already built for the
    `execute` node, so the two compile against one set of schemas, read
    once, rather than each fetching its own copy. Left unset, the seam
    builds its own from `settings`, for callers with no surface of their own
    to share (tests, the scripted-mode wiring), and converse answers a
    question about DFO with `harness.dfo_faq.CANNOT_DESCRIBE_DFO`.
    """
    shared = client
    return GenerativeNodes(
        converse=GeminiConverse(model_name=model_name, client=shared, surface=surface),
        planner=PlannerSeam(settings=settings, surface=surface),
        compose=GeminiCompose(model_name=model_name, client=shared),
        narrate=GeminiNarrate(model_name=model_name, client=shared),
        refusal=GeminiRefusal(model_name=model_name, client=shared),
    )


def handing_back_planner_failures(seam: PlannerModel) -> PlannerModel:
    """`seam`, answering a draft it cannot finish (`PlannerSeamError`) with a
    refusal instead of raising. On the orchestrator engine the planner node
    then takes its "refused" route and `plan_done` hands the reason back to
    the agent, which answers with what it can. The failure is logged in full."""

    async def planner(**kwargs: Any) -> dict[str, Any]:
        try:
            return await seam(**kwargs)
        except PlannerSeamError as exc:
            logger.error(
                "planner seam failed; handing the reason back to the orchestrator: %s", exc
            )
            return {"refused": {"text": str(exc)}}

    return planner


__all__ = ["build_generative_nodes", "handing_back_planner_failures"]
