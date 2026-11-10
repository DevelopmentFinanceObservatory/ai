"""The `get_skill` tool: one skill's `SKILL.md` and its file list, or one of its files.

`search` names the skills that apply to its results, by id and description.
`get_skill` loads one through `data_mcp.skills_client`: the first page of
`SKILL.md` verbatim, front matter included, then the skill's other files with
their sizes and the call that reads one, and one `resource_link` block per
file. Structured content is the skill's SEP-2640 entry: `uri`, `frontmatter`,
and `resources` as the full manifest of `uri`, `digest` and `size`.

With a `path` it reads one file through `read_skill_file`, by the bundle-relative
path `SKILL.md` links it under, `SKILL.md` itself included. A file past 32,000
characters comes in pages: each page stops short of the end with a closing line
naming the exact next call, carrying the file's digest, because the combined
server strips structured content and the text is the whole surface there. A
continuation whose digest is not the file's current one refuses
`upstream_version_changed` and names the call that restarts at offset 0. A
continuation without a digest refuses `request_invalid` before any read, since
nothing would tie its page to the revision the earlier pages came from.
"""

from __future__ import annotations

from typing import Annotated

from fastmcp.tools import ToolResult
from mcp.types import ResourceLink, TextContent
from pydantic import Field

from data_mcp.skills_client import (
    PATH_MAX_CHARS,
    SKILL_ID_MAX_CHARS,
    SKILL_MAIN_FILE,
    Skill,
    SkillFile,
    SkillReadRefused,
    check_digest,
    continuation_line,
    fetch_skill,
    fetch_skill_file,
    offset_refusal,
    page_of,
    skill_entry,
    skill_uri,
    with_closing_line,
)
from data_mcp.tool_errors import ToolRefusal, to_tool_result


def _file_size(size: int) -> str:
    """A byte count as a reader scans it: `812 B`, `5.5 KB`."""
    return f"{size} B" if size < 1024 else f"{size / 1024:.1f} KB"


def _files_block(skill: Skill) -> str:
    return "\n".join(
        [
            f'Files in this skill; read one with get_skill(skill_id="{skill.id}", path=…):',
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
    path: Annotated[
        str | None,
        Field(
            min_length=1,
            max_length=PATH_MAX_CHARS,
            description=(
                "A file's path exactly as listed, such as references/core-oda.md. Left out, "
                "the tool loads the skill: SKILL.md and its file list. A file refused as "
                "not found is not retried under a guessed path."
            ),
        ),
    ] = None,
    offset: Annotated[
        int,
        Field(
            ge=0,
            description="Where a page starts, in characters: 0, or the offset a closing line names.",
        ),
    ] = 0,
    digest: Annotated[
        str | None,
        Field(
            description="The digest a page's closing line names; required when offset is past 0."
        ),
    ] = None,
) -> ToolResult:
    """Returns a skill's SKILL.md verbatim and the list of its other files, or one of
    those files verbatim when `path` is given. Text past 32,000 characters comes in
    pages; a page that stops short closes with the call that reads on, carrying its
    `offset` and `digest`.
    """
    if path is not None:
        return await read_skill_file(skill_id=skill_id, path=path, offset=offset, digest=digest)
    if offset > 0 or digest is not None:
        return to_tool_result(
            ToolRefusal.request_invalid(
                tool="get_skill",
                pointer="/path",
                detail=(
                    "offset and digest page a file; pass the path the closing line names, "
                    f'such as path="{SKILL_MAIN_FILE}"'
                ),
            )
        )
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


def _digest_required(*, skill_id: str, path: str) -> ToolRefusal:
    restart = f'get_skill(skill_id="{skill_id}", path="{path}", offset=0)'
    return ToolRefusal.request_invalid(
        tool="get_skill",
        pointer="/digest",
        detail=(
            "a page past offset 0 needs the digest the previous page's closing line names; "
            "make that line's call as written, or read from the start with "
            f"{restart}"
        ),
    )


async def read_skill_file(
    *, skill_id: str, path: str, offset: int, digest: str | None
) -> ToolResult:
    """One page of a skill file, or the refusal for it."""
    if offset > 0 and digest is None:
        return to_tool_result(_digest_required(skill_id=skill_id, path=path))
    if digest is not None and (malformed := check_digest(digest)) is not None:
        return to_tool_result(malformed)
    try:
        file = await fetch_skill_file(skill_id, path)
    except SkillReadRefused as refused:
        return to_tool_result(refused.refusal)
    if digest is not None and digest != file.digest:
        return to_tool_result(
            ToolRefusal.skill_file_changed(
                skill_id=skill_id,
                path=path,
                digest=digest,
                current_digest=file.digest,
            )
        )
    text = file.content_text
    refusal = offset_refusal(offset, path=path, length=len(text))
    if refusal is not None:
        return to_tool_result(refusal)
    page = page_of(text, offset)
    rendered = page.text
    if page.next_offset is not None:
        rendered = with_closing_line(
            rendered,
            continuation_line(
                skill_id=skill_id,
                path=path,
                page=page,
                length=len(text),
                digest=file.digest,
            ),
        )
    return ToolResult(
        content=rendered,
        structured_content={
            "uri": skill_uri(skill_id, path),
            "path": path,
            "size": file.size,
            "digest": file.digest,
            "offset": page.offset,
            "next_offset": page.next_offset,
        },
        is_error=False,
    )
