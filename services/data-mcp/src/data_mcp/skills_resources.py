"""The `skill://` resources: every skill file, in the layout FastMCP's skill provider uses.

`SkillsProvider` serves three URI shapes, read from dfo-apis' Skills API
through `data_mcp.skills_client`:

- `skill://{id}/SKILL.md`: `SKILL.md` verbatim, with the front matter's
  `description` as the resource's, which FastMCP's `list_skills` client
  helper reads.
- `skill://{id}/_manifest`: `{"skill": id, "files": [{"path", "size",
  "hash"}]}` as JSON, which FastMCP's `get_skill_manifest` and
  `download_skill` client helpers read. `hash` is the Skills API's `digest`.
- `skill://{id}/{path*}`: one file, through a resource template.

`resources/list` names the first two for each skill search can name, one the
list door marks `attached` and not `excluded`. A direct read serves any skill
by id. Each read fetches its bytes at read time and carries the media type the
Skills API reports.

A read that fails raises `ResourceError` with no chained cause, written for
the caller as `"<code>: <prose>"`, so the combined server passes it through as
written. A `resources/list` whose Skills API read fails lists no skills and
logs why. The skill tools refuse loudly in the same state.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from typing import Any, NoReturn, override

from fastmcp.exceptions import ResourceError
from fastmcp.resources.base import Resource, ResourceContent, ResourceResult
from fastmcp.resources.template import ResourceTemplate
from fastmcp.server.providers.base import Provider
from fastmcp.utilities.components import FastMCPComponent
from fastmcp.utilities.versions import VersionSpec
from pydantic import AnyUrl

from data_mcp.skills_client import (
    SKILL_MAIN_FILE,
    SkillReadRefused,
    SkillSummary,
    fetch_skill,
    fetch_skill_file,
    fetch_skill_list,
    skill_uri,
)
from data_mcp.tool_errors import ToolErrorCode

logger = logging.getLogger(__name__)

SKILL_SCHEME = "skill://"
MANIFEST_PATH = "_manifest"
SKILL_FILE_TEMPLATE = "skill://{skill_id}/{path*}"
_MANIFEST_MEDIA_TYPE = "application/json"


def refusal_prose(refused: SkillReadRefused, *, skill_id: str, path: str | None = None) -> str:
    """What a failed read of `skill_id` (and `path`) tells a resource or extension caller.

    An unmounted Skills API and an index still building get fixed prose. A
    `service_unavailable` with no Skills API code behind it means dfo-apis is
    unreachable, and the refusal's own message says so. Every other refusal
    keeps its own message.
    """
    refusal = refused.refusal
    if refusal.code is ToolErrorCode.SKILL_NOT_FOUND:
        return f"No skill has the id {skill_id!r}."
    if refusal.code is ToolErrorCode.SKILL_RESOURCE_NOT_FOUND:
        return f"Skill {skill_id!r} has no file {path!r}; get_skill lists its files."
    if refusal.code is ToolErrorCode.CAPABILITY_UNAVAILABLE:
        return "This deployment serves no skills."
    if (
        refusal.code is ToolErrorCode.SERVICE_UNAVAILABLE
        and refused.upstream_code == "service_unavailable"
    ):
        return "The skill catalogue is still loading; retry shortly."
    return refusal.message


def parse_skill_uri(uri: str) -> tuple[str, str] | None:
    """The skill id and bundle path of a `skill://{id}/{path}` URI, or `None` for any other.

    The parts are returned as written. The Skills API reads check their grammar.
    """
    if not uri.startswith(SKILL_SCHEME):
        return None
    skill_id, slash, path = uri.removeprefix(SKILL_SCHEME).partition("/")
    if not skill_id or not slash or not path:
        return None
    return skill_id, path


def _refuse(refused: SkillReadRefused, *, skill_id: str, path: str) -> NoReturn:
    prose = refusal_prose(refused, skill_id=skill_id, path=path)
    raise ResourceError(f"{refused.refusal.code}: {prose}") from None


class _SkillFile(Resource):
    """One file of a skill, read from the Skills API's resources door."""

    skill_id: str
    path: str

    @override
    async def read(self) -> ResourceResult:
        try:
            file = await fetch_skill_file(self.skill_id, self.path)
        except SkillReadRefused as refused:
            _refuse(refused, skill_id=self.skill_id, path=self.path)
        return ResourceResult([ResourceContent(file.content_text, mime_type=file.media_type)])


class _SkillManifest(Resource):
    """A skill's file list in FastMCP's manifest shape, read from the skill door."""

    skill_id: str

    @override
    async def read(self) -> ResourceResult:
        try:
            skill = (await fetch_skill(self.skill_id)).value
        except SkillReadRefused as refused:
            _refuse(refused, skill_id=self.skill_id, path=MANIFEST_PATH)
        manifest = {
            "skill": skill.id,
            "files": [
                {"path": file.path, "size": file.size, "hash": file.digest}
                for file in skill.resources
            ],
        }
        return ResourceResult(
            [ResourceContent(json.dumps(manifest, indent=2), mime_type=_MANIFEST_MEDIA_TYPE)]
        )


class _SkillFileTemplate(ResourceTemplate):
    """`skill://{skill_id}/{path*}`: any file of any skill."""

    @override
    async def create_resource(self, uri: str, params: dict[str, Any]) -> Resource:
        return _skill_file(uri, skill_id=str(params["skill_id"]), path=str(params["path"]))


def _skill_file(
    uri: str,
    *,
    skill_id: str,
    path: str,
    mime_type: str = "text/markdown",
    description: str | None = None,
) -> Resource:
    return _SkillFile(
        uri=AnyUrl(uri),
        name=f"{skill_id}/{path}",
        description=description,
        mime_type=mime_type,
        skill_id=skill_id,
        path=path,
    )


def _manifest(skill_id: str) -> Resource:
    return _SkillManifest(
        uri=AnyUrl(skill_uri(skill_id, MANIFEST_PATH)),
        name=f"{skill_id}/{MANIFEST_PATH}",
        description=f"File listing for {skill_id}",
        mime_type=_MANIFEST_MEDIA_TYPE,
        skill_id=skill_id,
    )


def _listed(skill: SkillSummary) -> list[Resource]:
    main = skill.main_file
    return [
        _skill_file(
            skill_uri(skill.id),
            skill_id=skill.id,
            path=main.path,
            mime_type=main.media_type,
            description=skill.description,
        ),
        _manifest(skill.id),
    ]


_TEMPLATE = _SkillFileTemplate(
    uri_template=SKILL_FILE_TEMPLATE,
    name="skill_file",
    description=(
        "One file of a skill, by the skill's id and the path its manifest "
        f"(skill://{{skill_id}}/{MANIFEST_PATH}) lists."
    ),
    mime_type="text/markdown",
    parameters={
        "type": "object",
        "properties": {"skill_id": {"type": "string"}, "path": {"type": "string"}},
        "required": ["skill_id", "path"],
    },
)


class SkillsProvider(Provider):
    """Serves the Skills API's skills as `skill://` resources."""

    @override
    async def get_tasks(self) -> Sequence[FastMCPComponent]:
        # No skill resource runs as a background task, and the base class
        # would list resources, a Skills API read, at every server startup.
        return []

    @override
    async def _list_resources(self) -> Sequence[Resource]:
        try:
            catalogue = await fetch_skill_list()
        except SkillReadRefused as refused:
            logger.warning(
                "skill resources: the Skills API list read failed, so none are listed: %s",
                refused.refusal.message,
            )
            return []
        return [
            resource
            for skill in catalogue.value.skills
            if skill.searchable
            for resource in _listed(skill)
        ]

    @override
    async def _get_resource(self, uri: str, version: VersionSpec | None = None) -> Resource | None:
        parsed = parse_skill_uri(uri)
        if parsed is None:
            return None
        skill_id, path = parsed
        if path == MANIFEST_PATH:
            return _manifest(skill_id)
        if path == SKILL_MAIN_FILE:
            return _skill_file(uri, skill_id=skill_id, path=path)
        return None

    @override
    async def _list_resource_templates(self) -> Sequence[ResourceTemplate]:
        return [_TEMPLATE]
