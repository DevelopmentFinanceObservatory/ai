"""The planner's skill tool, `get_skill`.

It is an ADK function tool over data-mcp's `get_skill`, called through the
harness's `compiler.surface.ToolSurface`, in the shape the public MCP surface
offers: the skill's first page, or one file of it when `path` is given, with
`offset` and `digest` to read on. Every call sends `NAMES_GET_SKILL`, so each
page names `get_skill` and nothing the model reads names a tool it lacks. A call
answers with data-mcp's text as it came back, a refusal's included: an unknown id
refuses `skill_not_found`, and a deployment with no Skills API refuses
`capability_unavailable`, which the planner reads and drafts on without the
skill. A call that does not complete answers with one sentence saying so.

Each answer data-mcp serves without an error is also recorded in session
state (`SKILLS_READ_STATE_KEY`), so a later run of the same plan can be shown
what an earlier run read (`harness.planner_seam`) instead of reading it again.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Annotated, Any

from compiler.surface import ToolSurface, ToolSurfaceUnavailableError
from data_mcp.get_skill_tool import NAMES_GET_SKILL
from google.adk.tools.function_tool import FunctionTool
from google.adk.tools.tool_context import ToolContext
from pydantic import Field

from planner.plan import SessionState
from planner.workflow import ToolId

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


async def _call(
    surface: ToolSurface,
    arguments: Mapping[str, object],
    *,
    tool_context: ToolContext,
) -> str:
    """data-mcp's text for one `get_skill` call, or a sentence saying it did not complete.

    An answer without an error is recorded under `SKILLS_READ_STATE_KEY`,
    with its text under `skill_page_key`, unless that page is already there.
    """
    try:
        result = await surface.call_tool(tool=GET_SKILL, arguments=arguments)
    except ToolSurfaceUnavailableError as exc:
        sent = {name: value for name, value in arguments.items() if name != NAMES_GET_SKILL}
        logger.warning("planner get_skill(%s) did not complete; drafting without it: %s", sent, exc)
        return "get_skill could not reach data-mcp just now. Draft the plan without this skill."
    if not result.is_error:
        page = {
            "skill_id": arguments["skill_id"],
            "path": arguments.get("path"),
            "offset": arguments.get("offset", 0),
        }
        recorded = skills_read(tool_context.state)
        if page not in recorded:
            tool_context.state[skill_page_key(page)] = result.text
            tool_context.state[SKILLS_READ_STATE_KEY] = [*recorded, page]
    return result.text


# Keep the description to one sentence saying what the call returns: a longer one
# makes Gemini reason at length before some calls (`tests/planner/test_skills.py`
# bounds its length). ADK sends a docstring's `Args:` block inside the description,
# so the argument text sits in the parameters' `Field` descriptions.
_GET_SKILL_DOC = (
    "Returns the SKILL.md of the skill with the given id and the list of its files, "
    "or one file of the skill, verbatim, when `path` is given."
)


def build_skill_tools(surface: ToolSurface) -> list[FunctionTool]:
    """Build `get_skill`, which calls data-mcp through `surface`."""

    async def get_skill(
        skill_id: Annotated[
            str, Field(description=f"A skill id, or `{DFO_FAQ_SKILL_ID}` for DFO itself.")
        ],
        tool_context: ToolContext,
        path: Annotated[
            str | None, Field(description="A file's path exactly as the list gives it.")
        ] = None,
        offset: Annotated[
            int, Field(description="Where a page starts: 0, or the offset a closing line names.")
        ] = 0,
        digest: Annotated[
            str | None, Field(description="The digest a closing line names, on a continuation.")
        ] = None,
    ) -> str:
        arguments: dict[str, object] = {"skill_id": skill_id, NAMES_GET_SKILL: True}
        if path is not None:
            arguments["path"] = path
        if offset:
            arguments["offset"] = offset
        if digest is not None:
            arguments["digest"] = digest
        return await _call(surface, arguments, tool_context=tool_context)

    get_skill.__doc__ = _GET_SKILL_DOC
    return [FunctionTool(get_skill)]
