"""Reads dfo-apis' Skills API for every data-mcp layer that serves skills.

The module holds the skill id and bundle-relative path grammar, the models of
the three doors data-mcp reads (`GET /api/skills`, `GET /api/skills/{skill_id}`
and `GET /api/skills/{skill_id}/resources/{path}`), the SEP-2640 skill entry
built from a list or skill door entry, the mapping of the Skills API's
refusals onto the tool refusal grammar, and the paging both skill tools
share. Requests go through `get_client()`, whose probe of
`/api/skills/healthz` tells an unmounted Skills API from an unknown id.

A read that fails raises `SkillReadRefused` carrying the mapped
`ToolRefusal` and the Skills API's own code, so a tool returns the refusal and
the resource layer and the Skills over MCP extension word it for their own
callers.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Self

import httpx2
from pydantic import BaseModel, ConfigDict, ValidationError, model_validator

from data_mcp.client_holder import get_client
from data_mcp.errors import UpstreamError, UpstreamSurfaceUnavailable
from data_mcp.tool_errors import ToolRefusal, from_upstream_error

# The Agent Skills name rule the Skills API reads ids by: kebab-case segments
# of lower-case letters and digits, at most 64 characters.
SKILL_ID_MAX_CHARS = 64
_SKILL_ID_PATTERN = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")

# The Skills API routes this segment to a door of its own, so no skill
# carries it as an id.
_RESERVED_SKILL_IDS = frozenset({"healthz"})

# A bundle-relative path: `SKILL.md`, or kebab-case segments whose last ends
# in one extension, such as `references/core-oda.md`. A path with `..`, a
# leading slash or a backslash falls outside it.
SKILL_MAIN_FILE = "SKILL.md"
PATH_MAX_CHARS = 512
_SEGMENT = r"[a-z0-9]+(?:-[a-z0-9]+)*"
_PATH_PATTERN = re.compile(rf"{_SEGMENT}(?:/{_SEGMENT})*\.[a-z0-9]+")

_DIGEST_PATTERN = re.compile(r"sha256:[0-9a-f]{64}")

_LIST_ROUTE = "/api/skills"
_SKILL_ROUTE = "/api/skills/{skill_id}"
_FILE_ROUTE = "/api/skills/{skill_id}/resources/{path}"

# The most characters one page of a skill file carries. A page ends at the
# last line break in its second half, or at this count when there is none.
PAGE_CHARS = 32_000

_UNMOUNTED_DETAIL = "this deployment does not serve the Skills API"

_MAX_AGE = re.compile(r"(?:^|,)\s*max-age\s*=\s*(\d+)\s*(?:,|$)", re.IGNORECASE)


class SkillReadRefused(Exception):  # noqa: N818 -- carries a refusal, not a fault
    """A Skills API read that ended in a refusal a caller can act on.

    `upstream_code` is the `code` of the `UpstreamError` the read failed with,
    and `None` for a read refused before it was sent or whose answer was
    unusable.
    """

    def __init__(self, refusal: ToolRefusal, *, upstream_code: str | None = None) -> None:
        super().__init__(refusal.message)
        self.refusal = refusal
        self.upstream_code = upstream_code


class SkillFile(BaseModel):
    """One manifest entry: a file of the skill with its size and digest."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    path: str
    size: int
    digest: str
    media_type: str


class SkillSummary(BaseModel):
    """The fields of one Skills API list entry that data-mcp reads."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    id: str
    frontmatter: dict[str, Any]
    attached: bool
    excluded: bool
    resources: list[SkillFile]

    @model_validator(mode="after")
    def _manifest_opens_with_skill_md(self) -> Self:
        if not self.resources or self.resources[0].path != SKILL_MAIN_FILE:
            raise ValueError(f"the manifest does not open with {SKILL_MAIN_FILE}")
        return self

    @property
    def main_file(self) -> SkillFile:
        """The manifest entry for `SKILL.md`, which the manifest lists first."""
        return self.resources[0]

    @property
    def other_files(self) -> list[SkillFile]:
        """Every manifest entry after `SKILL.md`, in path order."""
        return self.resources[1:]

    @property
    def searchable(self) -> bool:
        """Whether search can name this skill: attached and not excluded."""
        return self.attached and not self.excluded

    @property
    def description(self) -> str | None:
        """The front matter's `description`, when it is a string."""
        description = self.frontmatter.get("description")
        return description if isinstance(description, str) else None


class Skill(SkillSummary):
    """The fields of the Skills API's skill door that data-mcp reads."""

    content_text: str


class SkillList(BaseModel):
    """The fields of the Skills API's list door that data-mcp reads."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    skills: list[SkillSummary]


class SkillFileContent(BaseModel):
    """The Skills API's resources door: one file and its manifest facts."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    skill_id: str
    path: str
    size: int
    digest: str
    media_type: str
    content_text: str


@dataclass(frozen=True, kw_only=True)
class Page:
    """One page of a text: its characters and where the next page starts."""

    text: str
    offset: int
    next_offset: int | None


@dataclass(frozen=True, kw_only=True)
class Cacheable[T]:
    """A Skills API answer and the seconds its `Cache-Control: max-age` lets it be held.

    `max_age_s` is 0 when the answer carried no `max-age`.
    """

    value: T
    max_age_s: int


def skill_uri(skill_id: str, path: str = SKILL_MAIN_FILE) -> str:
    """The `skill://` URI of one file of a skill."""
    return f"skill://{skill_id}/{path}"


def skill_entry(skill: SkillSummary) -> dict[str, Any]:
    """`skill` as a SEP-2640 skill entry: `uri`, `frontmatter`, and `resources`
    as the full manifest of `uri`, `digest` and `size`."""
    return {
        "uri": skill_uri(skill.id),
        "frontmatter": skill.frontmatter,
        "resources": [
            {"uri": skill_uri(skill.id, file.path), "digest": file.digest, "size": file.size}
            for file in skill.resources
        ],
    }


def check_skill_id(skill_id: str) -> ToolRefusal | None:
    """A refusal for an id no skill can carry, or `None` for one that might."""
    if not _SKILL_ID_PATTERN.fullmatch(skill_id) or len(skill_id) > SKILL_ID_MAX_CHARS:
        return ToolRefusal.request_invalid(
            pointer="/skill_id",
            detail=(
                f"{skill_id!r} is not a skill id; an id is lower-case letters and digits in "
                f"hyphen-separated words, at most {SKILL_ID_MAX_CHARS} characters, as a "
                "search result names it"
            ),
        )
    if skill_id in _RESERVED_SKILL_IDS:
        return ToolRefusal.skill_not_found(skill_id=skill_id)
    return None


def check_path(path: str) -> ToolRefusal | None:
    """A refusal for a path no skill file can carry, or `None` for one that might."""
    if path == SKILL_MAIN_FILE or (len(path) <= PATH_MAX_CHARS and _PATH_PATTERN.fullmatch(path)):
        return None
    return ToolRefusal.request_invalid(
        pointer="/path",
        detail=(
            f"{path!r} is not a skill file path; a path is {SKILL_MAIN_FILE} or one "
            "get_skill lists, such as references/core-oda.md"
        ),
    )


def check_digest(digest: str) -> ToolRefusal | None:
    """A refusal for a digest that is not `sha256:` and 64 hex digits."""
    if _DIGEST_PATTERN.fullmatch(digest):
        return None
    return ToolRefusal.request_invalid(
        pointer="/digest",
        detail=(
            f"{digest!r} is not a file digest; pass the digest a page's closing line names, "
            "sha256: followed by 64 hex digits"
        ),
    )


async def fetch_skill_list() -> Cacheable[SkillList]:
    """Read the catalogue from `GET /api/skills`, with its `max-age`.

    Raises:
        SkillReadRefused: the Skills API refused, or its answer is not a list
            of skills whose manifests open with `SKILL.md`.
    """
    response = await _get("/api/skills", route=_LIST_ROUTE)
    return Cacheable(
        value=_parse(SkillList, _json(response), what="skill list"),
        max_age_s=_max_age_s(response),
    )


async def fetch_skill(skill_id: str) -> Cacheable[Skill]:
    """Read one skill from `GET /api/skills/{skill_id}`, with its `max-age`.

    Raises:
        SkillReadRefused: the id is malformed, the Skills API refused, or its
            answer is not a skill whose manifest opens with `SKILL.md`.
    """
    _raise_if_refused(check_skill_id(skill_id))
    response = await _get(f"/api/skills/{skill_id}", route=_SKILL_ROUTE, skill_id=skill_id)
    return Cacheable(
        value=_parse(Skill, _json(response), what="skill"), max_age_s=_max_age_s(response)
    )


async def fetch_skill_file(skill_id: str, path: str) -> SkillFileContent:
    """Read one file from `GET /api/skills/{skill_id}/resources/{path}`.

    Raises:
        SkillReadRefused: the id or path is malformed, the Skills API refused,
            or its answer is not a skill file.
    """
    _raise_if_refused(check_skill_id(skill_id) or check_path(path))
    response = await _get(
        f"/api/skills/{skill_id}/resources/{path}", route=_FILE_ROUTE, skill_id=skill_id, path=path
    )
    return _parse(SkillFileContent, _json(response), what="skill file")


def page_of(text: str, offset: int) -> Page:
    """The page of `text` that starts at character `offset`.

    A page holds at most `PAGE_CHARS` characters and, when the text goes on
    past it, ends just after the last line break in its second half, or at
    `PAGE_CHARS` when that half has none. `next_offset` is where the next
    page starts, and `None` on the last page. The caller checks that
    `offset` lies inside `text`.
    """
    end = offset + PAGE_CHARS
    if end >= len(text):
        return Page(text=text[offset:], offset=offset, next_offset=None)
    line_break = text.rfind("\n", offset + PAGE_CHARS // 2, end)
    cut = line_break + 1 if line_break != -1 else end
    return Page(text=text[offset:cut], offset=offset, next_offset=cut)


def offset_refusal(offset: int, *, path: str, length: int) -> ToolRefusal | None:
    """A refusal for an `offset` outside a text of `length` characters.

    Offset 0 is always valid, so an empty file still reads as one empty page.
    """
    if offset == 0 or 0 < offset < length:
        return None
    return ToolRefusal.request_invalid(
        pointer="/offset",
        detail=(
            f"offset {offset} is outside {path}, which is {length} characters; pass 0 or "
            "the offset a page's closing line names"
        ),
    )


def continuation_line(*, skill_id: str, path: str, page: Page, length: int, digest: str) -> str:
    """The closing line of a page that stops short of the end: the exact next call."""
    return (
        f"This page ends at character {page.next_offset:,} of {length:,}. Read the rest with "
        f'get_skill_file(skill_id="{skill_id}", path="{path}", offset={page.next_offset}, '
        f'digest="{digest}").'
    )


def with_closing_line(page_text: str, line: str) -> str:
    """`page_text` followed by `line` after a blank line."""
    separator = "\n" if page_text.endswith("\n") else "\n\n"
    return f"{page_text}{separator}{line}" if page_text else line


def _raise_if_refused(refusal: ToolRefusal | None) -> None:
    if refusal is not None:
        raise SkillReadRefused(refusal)


async def _get(
    url: str, *, route: str, skill_id: str | None = None, path: str | None = None
) -> httpx2.Response:
    try:
        return await get_client().request("GET", url, route=route)
    except UpstreamError as exc:
        raise SkillReadRefused(
            _refusal_from_upstream(exc, skill_id=skill_id, path=path), upstream_code=exc.code
        ) from exc


def _json(response: httpx2.Response) -> Any:
    try:
        return response.json()
    except ValueError as exc:
        raise SkillReadRefused(
            ToolRefusal.upstream_response_unusable(
                detail=f"the Skills API answered with a body that is not JSON: {exc}"
            )
        ) from exc


def _max_age_s(response: httpx2.Response) -> int:
    match = _MAX_AGE.search(response.headers.get("cache-control", ""))
    return int(match[1]) if match else 0


def _parse[M: BaseModel](model: type[M], body: Any, *, what: str) -> M:
    try:
        return model.model_validate(body)
    except ValidationError as exc:
        raise SkillReadRefused(
            ToolRefusal.upstream_response_unusable(
                detail=f"the {what} response did not match the shape this service expects: {exc}"
            )
        ) from exc


def _refusal_from_upstream(
    exc: UpstreamError, *, skill_id: str | None, path: str | None
) -> ToolRefusal:
    """Map a failed Skills API read onto the refusal grammar.

    The Skills API's `skill_not_found` and `skill_resource_not_found` carry
    neither the id nor the path as a field, so the refusal is built with the
    ones this read asked for. An unmounted Skills API arrives as the client's
    `UpstreamSurfaceUnavailable`, confirmed by its `healthz` probe. Every other
    code, `service_unavailable` while the index builds among them, maps as
    any dfo-apis refusal does.
    """
    if exc.code == "skill_not_found" and skill_id is not None:
        return ToolRefusal.skill_not_found(
            skill_id=skill_id, trace_id=exc.trace_id, capability_version=exc.capability_version
        )
    if exc.code == "skill_resource_not_found" and skill_id is not None and path is not None:
        return ToolRefusal.skill_resource_not_found(
            skill_id=skill_id,
            path=path,
            trace_id=exc.trace_id,
            capability_version=exc.capability_version,
        )
    if isinstance(exc, UpstreamSurfaceUnavailable):
        return ToolRefusal.capability_unavailable(
            detail=_UNMOUNTED_DETAIL,
            trace_id=exc.trace_id,
            capability_version=exc.capability_version,
        )
    return from_upstream_error(exc)
