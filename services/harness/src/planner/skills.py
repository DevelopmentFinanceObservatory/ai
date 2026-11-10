"""The planner's skill tool, `get_skill`, over no agent framework.

It is data-mcp's `get_skill`, called through the harness's
`compiler.surface.ToolSurface` (`read_skill_page`) in the shape the public MCP
surface offers: the skill's first page, or one file of it when `path` is given,
with `offset` and `digest` to read on. `planner.specs.skill_tool_spec` declares it
as the planner's `get_skill` tool. A call answers with data-mcp's text as it came back, a
refusal's included: an unknown id refuses `skill_not_found`, and a deployment
with no Skills API refuses `capability_unavailable`, which the planner reads and
drafts on without the skill. A call that does not complete answers with one
sentence saying so.

Each answer data-mcp serves without an error is also recorded in session
state (`SKILLS_READ_STATE_KEY`), so a later run of the same plan can be shown
what an earlier run read (`harness.planner_seam`) instead of reading it again.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from compiler.surface import ToolSurface, ToolSurfaceUnavailableError

from planner.plan import SessionState
from planner.workflow import ToolId

if TYPE_CHECKING:
    from planner.call import PlannerCall

GET_SKILL = ToolId(server="data-mcp", name="get_skill")
# The skill a question about DFO itself is answered from. The planner's
# instruction names it, and the orchestrator's `explain_dfo` loads it
# (`harness.dfo_faq`).
DFO_FAQ_SKILL_ID = "dfo-faq"

# Every skill page the planner read without an error, in the order it read
# them, each once: `{"skill_id", "path", "offset"}`, `path` `None` for a
# skill's first page. Each page's text sits under its own key (`skill_page_key`),
# so every read writes its own text once and this list stays small.
# `harness.planner_seam` clears it on a plan's first draft.
SKILLS_READ_STATE_KEY = "planner:skills_read"

logger = logging.getLogger(__name__)


def skills_read(state: SessionState) -> list[dict[str, Any]]:
    """Every skill page `state` records as read (`SKILLS_READ_STATE_KEY`), in read order."""
    return list(state.get(SKILLS_READ_STATE_KEY) or ())


def skill_page_key(page: Mapping[str, Any]) -> str:
    """The state key a skill page's text is stored under, from its skill, file and offset."""
    return f"planner:skill_page:{page['skill_id']}:{page['path'] or ''}:{page['offset']}"


async def read_skill_page(
    surface: ToolSurface,
    arguments: Mapping[str, object],
    *,
    call: PlannerCall,
) -> str:
    """data-mcp's text for one `get_skill` call, or a sentence saying it did not complete.

    An answer without an error is recorded under `SKILLS_READ_STATE_KEY`,
    with its text under `skill_page_key`, unless that page is already there.
    """
    try:
        result = await surface.call_tool(tool=GET_SKILL, arguments=arguments)
    except ToolSurfaceUnavailableError as exc:
        logger.warning(
            "planner get_skill(%s) did not complete; drafting without it: %s", dict(arguments), exc
        )
        return "get_skill could not reach data-mcp just now. Draft the plan without this skill."
    if not result.is_error:
        page = {
            "skill_id": arguments["skill_id"],
            "path": arguments.get("path"),
            "offset": arguments.get("offset", 0),
        }
        recorded = skills_read(call.state)
        if page not in recorded:
            call.state[skill_page_key(page)] = result.text
            call.state[SKILLS_READ_STATE_KEY] = [*recorded, page]
    return result.text
