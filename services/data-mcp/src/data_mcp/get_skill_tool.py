"""The `get_skill` tool: one skill's `SKILL.md` and its file list.

`search` names the skills that apply to its results, by id and description.
`get_skill` loads one through `data_mcp.skills_client`: the first page of
`SKILL.md` verbatim, front matter included, then the skill's other files with
their sizes and the call that reads one, and one `resource_link` block per
file. Structured content is the skill's SEP-2640 entry: `uri`, `frontmatter`,
and `resources` as the full manifest of `uri`, `digest` and `size`.
"""

from __future__ import annotations

from typing import Annotated

from fastmcp.tools import ToolResult
from mcp.types import ResourceLink, TextContent
from pydantic import Field

from data_mcp.skills_client import (
    SKILL_ID_MAX_CHARS,
    SKILL_MAIN_FILE,
    Skill,
    SkillFile,
    SkillReadRefused,
    continuation_line,
    fetch_skill,
    page_of,
    skill_entry,
    skill_uri,
    with_closing_line,
)
from data_mcp.tool_errors import to_tool_result


def _file_size(size: int) -> str:
    """A byte count as a reader scans it: `812 B`, `5.5 KB`."""
    return f"{size} B" if size < 1024 else f"{size / 1024:.1f} KB"


def _files_block(skill: Skill) -> str:
    return "\n".join(
        [
            f'Files in this skill; read one with get_skill_file(skill_id="{skill.id}", path=…):',
            *(f"- {file.path} ({_file_size(file.size)})" for file in skill.other_files),
        ]
    )


def _resource_link(skill_id: str, file: SkillFile) -> ResourceLink:
    return ResourceLink(
        type="resource_link",
        uri=skill_uri(skill_id, file.path),
        name=file.path,
        mimeType=file.media_type,
        size=file.size,
    )


def _render(skill: Skill) -> str:
    """The first page of `SKILL.md`, its closing line when it goes on, then the file list."""
    page = page_of(skill.content_text, 0)
    rendered = page.text
    if page.next_offset is not None:
        rendered = with_closing_line(
            rendered,
            continuation_line(
                skill_id=skill.id,
                path=SKILL_MAIN_FILE,
                page=page,
                length=len(skill.content_text),
                digest=skill.main_file.digest,
            ),
        )
    if skill.other_files:
        rendered = with_closing_line(rendered, _files_block(skill))
    return rendered


async def get_skill(
    skill_id: Annotated[
        str,
        Field(
            min_length=1,
            max_length=SKILL_ID_MAX_CHARS,
            description="A skill id named in a search result's skills list.",
        ),
    ],
) -> ToolResult:
    """Load one of the skills a `search` result names.

    A skill carries the definitions, caveats and methods an answer in its
    domain needs. Call this with an id from the skills list beneath a
    `search` result when its description bears on the question. A skill is
    reference material for the question, so loading one does not answer it.

    Returns the skill's SKILL.md verbatim, front matter included, then the
    skill's other files with their sizes. Read one with `get_skill_file`,
    passing its path exactly as listed. A file refused as not found is not
    retried under a guessed path. A SKILL.md past 32,000 characters is cut at
    a line break, with a closing line naming the `get_skill_file` call that
    reads the rest. An id no skill carries refuses `skill_not_found`.

    Args:
        skill_id: The skill's id, as a search result names it.

    Returns:
        The skill's SKILL.md and file list.
    """
    try:
        skill = (await fetch_skill(skill_id)).value
    except SkillReadRefused as refused:
        return to_tool_result(refused.refusal)
    return ToolResult(
        content=[
            TextContent(type="text", text=_render(skill)),
            *(_resource_link(skill.id, file) for file in skill.other_files),
        ],
        structured_content=skill_entry(skill),
        is_error=False,
    )
