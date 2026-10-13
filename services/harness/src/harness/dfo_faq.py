"""dfo-faq, the published skill a question about DFO itself is answered from.

The orchestrator's `explain_dfo` (`harness.orchestrator_tools`) reads it with
data-mcp's `get_skill`, the tool and id the planner's instruction names
(`planner.skills`), so both answer from the one published text.
`read_dfo_faq` reads it once per question about DFO and holds no copy.

The counts of what DFO holds come from the live search catalogue, read
through data-mcp's harness-only `catalogue_counts` tool
(`read_catalogue_counts`), so `explain_dfo` quotes the figures `sources`
quotes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from common.refusal import refusal_code
from compiler.surface import ToolSurface, ToolSurfaceUnavailableError
from data_mcp.catalogue_counts import COUNTS_UNAVAILABLE
from planner.skills import DFO_FAQ_SKILL_ID, GET_SKILL
from planner.workflow import ToolId

logger = logging.getLogger(__name__)

# The reply when this deployment serves no Skills API or no dfo-faq.
CANNOT_DESCRIBE_DFO = (
    "I can't describe the Development Finance Observatory on this deployment, "
    "because its description is not published here. I can still look up data for you."
)
# The reply when the read did not complete, or the reply drafted from it came
# back empty. Asking again can succeed.
DFO_FAQ_NOT_READ = (
    "I couldn't read the description of the Development Finance Observatory just "
    "now. Please ask again in a moment."
)
# data-mcp's counts of what DFO holds, from the search catalogue.
CATALOGUE_COUNTS = ToolId(server="data-mcp", name="catalogue_counts")
# Refusals that mean this deployment does not serve dfo-faq at all.
_NOT_SERVED = frozenset({"capability_unavailable", "skill_not_found"})


@dataclass(frozen=True)
class DfoFaqUnavailable:
    """dfo-faq could not be read; `reply` is what to say in its place."""

    reply: str


async def read_dfo_faq(surface: ToolSurface | None) -> str | DfoFaqUnavailable:
    """dfo-faq's text as `get_skill` renders it, or why it could not be read.

    A deployment with no surface, no Skills API or no dfo-faq answers
    `CANNOT_DESCRIBE_DFO`. Any other refusal, or a call that never
    completed, answers `DFO_FAQ_NOT_READ`, as the planner's skill tools
    answer a call that did not complete with a sentence and carry on.
    """
    if surface is None:
        return DfoFaqUnavailable(CANNOT_DESCRIBE_DFO)
    try:
        result = await surface.call_tool(tool=GET_SKILL, arguments={"skill_id": DFO_FAQ_SKILL_ID})
    except ToolSurfaceUnavailableError as exc:
        logger.warning("get_skill(%s) did not complete: %s", DFO_FAQ_SKILL_ID, exc)
        return DfoFaqUnavailable(DFO_FAQ_NOT_READ)
    if not result.is_error:
        return result.text
    code = refusal_code(result.structured_content)
    logger.warning("get_skill(%s) refused %s: %s", DFO_FAQ_SKILL_ID, code, result.text)
    return DfoFaqUnavailable(CANNOT_DESCRIBE_DFO if code in _NOT_SERVED else DFO_FAQ_NOT_READ)


async def read_catalogue_counts(surface: ToolSurface | None) -> str:
    """The counts of what DFO holds as data-mcp words them, or
    `COUNTS_UNAVAILABLE` when there is no surface, data-mcp refused (a 503
    from a deployment with no published catalogue among them), or the call
    did not complete."""
    if surface is None:
        return COUNTS_UNAVAILABLE
    try:
        result = await surface.call_tool(tool=CATALOGUE_COUNTS, arguments={})
    except ToolSurfaceUnavailableError as exc:
        logger.warning("catalogue_counts did not complete: %s", exc)
        return COUNTS_UNAVAILABLE
    if result.is_error:
        code = refusal_code(result.structured_content)
        logger.warning("catalogue_counts refused %s: %s", code, result.text)
        return COUNTS_UNAVAILABLE
    return result.text


__all__ = [
    "CANNOT_DESCRIBE_DFO",
    "CATALOGUE_COUNTS",
    "DFO_FAQ_NOT_READ",
    "DfoFaqUnavailable",
    "read_catalogue_counts",
    "read_dfo_faq",
]
