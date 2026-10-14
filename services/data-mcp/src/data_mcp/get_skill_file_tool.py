"""The `get_skill_file` tool: one file of a skill, verbatim, a page at a time.

`get_skill` lists a skill's files. `get_skill_file` reads one through
`data_mcp.skills_client`, by the bundle-relative path `SKILL.md` links it
under, `SKILL.md` itself included. A file past 32,000 characters comes in
pages: each page stops short of the end with a closing line naming the exact
next call, carrying the file's digest, because the combined server strips
structured content and the text is the whole surface there. A continuation
whose digest is not the file's current one refuses `upstream_version_changed`
and names the call that restarts at offset 0. A continuation without a digest
refuses `request_invalid` before any read, since nothing would tie its page to
the revision the earlier pages came from.
"""

from __future__ import annotations

from typing import Annotated

from fastmcp.tools import ToolResult
from pydantic import Field

from data_mcp.skills_client import (
    PATH_MAX_CHARS,
    SKILL_ID_MAX_CHARS,
    SkillReadRefused,
    check_digest,
    continuation_line,
    fetch_skill_file,
    offset_refusal,
    page_of,
    skill_uri,
    with_closing_line,
)
from data_mcp.tool_errors import ToolRefusal, to_tool_result


def _digest_required(*, skill_id: str, path: str) -> ToolRefusal:
    restart = f'get_skill_file(skill_id="{skill_id}", path="{path}", offset=0)'
    return ToolRefusal.request_invalid(
        pointer="/digest",
        detail=(
            "a page past offset 0 needs the digest the previous page's closing line names; "
            "make that line's call as written, or read from the start with "
            f"{restart}"
        ),
    )


async def get_skill_file(
    skill_id: Annotated[
        str,
        Field(
            min_length=1,
            max_length=SKILL_ID_MAX_CHARS,
            description="A skill id named in a search result's skills list.",
        ),
    ],
    path: Annotated[
        str,
        Field(
            min_length=1,
            max_length=PATH_MAX_CHARS,
            description=(
                "The file's path exactly as get_skill lists it, such as "
                "references/core-oda.md, or SKILL.md."
            ),
        ),
    ],
    offset: Annotated[
        int,
        Field(
            ge=0,
            description="Where the page starts, in characters. Use 0, or the offset a closing line names.",
        ),
    ] = 0,
    digest: Annotated[
        str | None,
        Field(
            description="The digest a page's closing line names. Required when offset is past 0."
        ),
    ] = None,
) -> ToolResult:
    """Read one file of a skill `get_skill` loaded, verbatim.

    Use it after `get_skill`, to read a file the skill lists. Pass the path
    exactly as `get_skill` lists it. SKILL.md is accepted too. A file refused
    as not found is not retried under a guessed path. Call `get_skill` for the
    list of files. A skill's files are reference material for the question, so
    reading one does not answer it.

    A file past 32,000 characters comes in pages. A page that stops short of
    the end closes with a line naming the exact next call, with its `offset`
    and `digest`; make that call to read on. If the file changed between
    pages, the continuation refuses `upstream_version_changed` and names the
    call that reads it again from offset 0. An `offset` past 0 without a
    `digest` refuses `request_invalid`.

    Args:
        skill_id: The skill's id, as a search result names it.
        path: The file's bundle-relative path.
        offset: Where the page starts, in characters of the file's text.
        digest: The file digest a previous page's closing line names.

    Returns:
        One page of the file.
    """
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
                skill_id=skill_id, path=path, digest=digest, current_digest=file.digest
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
                skill_id=skill_id, path=path, page=page, length=len(text), digest=file.digest
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
