"""The planner's two skill tools, `get_skill` and `get_skill_file`.

Each is an ADK function tool over data-mcp's tool of the same name, called
through the harness's `compiler.surface.ToolSurface`, so the planner and an
outside model load the same text by the same names. A tool answers with
data-mcp's text as it came back, a refusal's included: an unknown id refuses
`skill_not_found`, and a deployment with no Skills API refuses
`capability_unavailable`, which the planner reads and drafts on without the
skill. A call that does not complete answers with one sentence saying so.

Each answer data-mcp serves without an error is also recorded in session
state (`SKILLS_READ_STATE_KEY`), so a later run of the same plan can be shown
what an earlier run read (`harness.planner_seam`) instead of reading it again.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

from compiler.surface import ToolSurface, ToolSurfaceUnavailableError
from google.adk.tools.function_tool import FunctionTool
from google.adk.tools.tool_context import ToolContext

from planner.plan import SessionState
from planner.workflow import ToolId

GET_SKILL = ToolId(server="data-mcp", name="get_skill")
GET_SKILL_FILE = ToolId(server="data-mcp", name="get_skill_file")
# The skill a question about DFO itself is answered from. The planner's
# instruction names it, and the orchestrator's `explain_dfo` loads it
# (`harness.dfo_faq`).
DFO_FAQ_SKILL_ID = "dfo-faq"

# Every skill page the planner read without an error, in the order it read
# them, each once: `{"skill_id", "path", "offset"}`, `path` `None` for a
# `get_skill` read. Each page's text sits under its own key (`skill_page_key`),
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
    tool: ToolId,
    arguments: Mapping[str, object],
    *,
    tool_context: ToolContext,
    path: str | None,
) -> str:
    """data-mcp's text for one call, or a sentence saying the call did not complete.

    An answer without an error is recorded under `SKILLS_READ_STATE_KEY`,
    with its text under `skill_page_key`, unless that page is already there.
    """
    try:
        result = await surface.call_tool(tool=tool, arguments=arguments)
    except ToolSurfaceUnavailableError as exc:
        logger.warning(
            "planner %s(%s) did not complete; drafting without it: %s",
            tool.name,
            dict(arguments),
            exc,
        )
        return f"{tool.name} could not reach data-mcp just now. Draft the plan without this skill."
    if not result.is_error:
        page = {
            "skill_id": arguments["skill_id"],
            "path": path,
            "offset": arguments.get("offset", 0),
        }
        recorded = skills_read(tool_context.state)
        if page not in recorded:
            tool_context.state[skill_page_key(page)] = result.text
            tool_context.state[SKILLS_READ_STATE_KEY] = [*recorded, page]
    return result.text


# `get_skill`'s description, as the planner's model reads it.
_GET_SKILL_DOC = """Load one skill: its SKILL.md verbatim, then the files it links.

        A skill carries the definitions, caveats and methods an answer in its
        domain needs. Loading one returns instructions and does not finish the
        task: carry on drafting the plan once you have read them. An id no
        skill carries refuses `skill_not_found`.

        Args:
            skill_id: An id the search readings' `Skills named` line lists, or
                `{dfo_faq}` for a question about DFO itself.
        """


def build_skill_tools(surface: ToolSurface) -> list[FunctionTool]:
    """Build `get_skill` and `get_skill_file`, each calling data-mcp through `surface`."""

    async def get_skill(skill_id: str, tool_context: ToolContext) -> str:
        return await _call(
            surface, GET_SKILL, {"skill_id": skill_id}, tool_context=tool_context, path=None
        )

    # Set here so the declaration names the skill id from its one constant.
    get_skill.__doc__ = _GET_SKILL_DOC.format(dfo_faq=DFO_FAQ_SKILL_ID)

    async def get_skill_file(
        skill_id: str,
        path: str,
        tool_context: ToolContext,
        offset: int = 0,
        digest: str | None = None,
    ) -> str:
        """Read one file of a skill `get_skill` loaded, verbatim.

        Pass the path exactly as `get_skill` lists it; SKILL.md is accepted
        too. A file refused as not found is not retried under a guessed path:
        call `get_skill` for the list of files. A page that stops short of the
        end closes with a line naming the exact next call, with its `offset`
        and `digest`; make that call to read on.

        Args:
            skill_id: The skill's id.
            path: The file's path exactly as `get_skill` lists it.
            offset: Where the page starts, in characters: 0, or the offset a
                closing line names.
            digest: The digest a page's closing line names, on a continuation.
        """
        arguments: dict[str, object] = {"skill_id": skill_id, "path": path, "offset": offset}
        if digest is not None:
            arguments["digest"] = digest
        return await _call(surface, GET_SKILL_FILE, arguments, tool_context=tool_context, path=path)

    return [FunctionTool(get_skill), FunctionTool(get_skill_file)]
