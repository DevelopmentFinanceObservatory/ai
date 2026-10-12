"""The planner's two skill tools, `get_skill` and `get_skill_file`.

Each is an ADK function tool over data-mcp's tool of the same name, called
through the harness's `compiler.surface.ToolSurface`, so the planner and an
outside model load the same text by the same names. A tool answers with
data-mcp's text as it came back, a refusal's included: an unknown id refuses
`skill_not_found`, and a deployment with no Skills API refuses
`capability_unavailable`, which the planner reads and drafts on without the
skill. A call that does not complete answers with one sentence saying so.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping

from compiler.surface import ToolSurface, ToolSurfaceUnavailableError
from google.adk.tools.function_tool import FunctionTool

from planner.workflow import ToolId

GET_SKILL = ToolId(server="data-mcp", name="get_skill")
GET_SKILL_FILE = ToolId(server="data-mcp", name="get_skill_file")
# The skill a question about DFO itself is answered from. The planner's
# instruction names it, and the orchestrator's `explain_dfo` loads it
# (`harness.dfo_faq`).
DFO_FAQ_SKILL_ID = "dfo-faq"

logger = logging.getLogger(__name__)


async def _call(surface: ToolSurface, tool: ToolId, arguments: Mapping[str, object]) -> str:
    """data-mcp's text for one call, or a sentence saying the call did not complete."""
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

    async def get_skill(skill_id: str) -> str:
        return await _call(surface, GET_SKILL, {"skill_id": skill_id})

    # Set here so the declaration names the skill id from its one constant.
    get_skill.__doc__ = _GET_SKILL_DOC.format(dfo_faq=DFO_FAQ_SKILL_ID)

    async def get_skill_file(
        skill_id: str, path: str, offset: int = 0, digest: str | None = None
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
        return await _call(surface, GET_SKILL_FILE, arguments)

    return [FunctionTool(get_skill), FunctionTool(get_skill_file)]
