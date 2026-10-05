"""The Skills over MCP extension (SEP-2640, `io.modelcontextprotocol/skills`).

`SkillsExtension` answers two methods from dfo-apis' Skills API:

- `skills/list` reads `GET /api/skills` once and returns every skill search
  can name, the ones `skills_resources.SkillsProvider` lists, as SEP-2640
  entries.
- `skills/get` takes a skill's `skill://{id}/SKILL.md` URI, reads the skill
  door, and returns that skill's entry under `skill`. It serves any skill by
  id.

Both results carry `resultType: "complete"`, `ttlMs`, the answering door's
`Cache-Control: max-age` in milliseconds, and `cacheScope: "public"`, since
the catalogue is the same for every caller. Files are read with ordinary
`resources/read` against `SkillsProvider`, so the digests a host checks are
the Skills API's own. `settings()` is empty, so `directoryRead` is not offered.

A malformed URI or an unknown skill fails with `-32602`, and every other
failure, an unmounted Skills API or an index still building among them, with
`-32603`. Each message reads `<code>: <prose>`, the refusal's code and the
prose `skills_resources.refusal_prose` writes, the form the skill resources'
errors and the registering server's own extension errors take.

A mounted server's extensions do not reach the server that mounts it, so the
server an outside client connects to registers this one. `guard` wraps each
handler as that server needs.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, NoReturn, override

from fastmcp.server.extensions import ExtensionRequestHandler, MethodBinding, ServerExtension
from mcp.server.context import ServerRequestContext
from mcp.shared.exceptions import MCPError
from mcp_types import INTERNAL_ERROR, INVALID_PARAMS, RequestParams

from data_mcp.skills_client import (
    SKILL_MAIN_FILE,
    Cacheable,
    SkillReadRefused,
    fetch_skill,
    fetch_skill_list,
    skill_entry,
)
from data_mcp.skills_resources import parse_skill_uri, refusal_prose
from data_mcp.tool_errors import ToolErrorCode

SKILLS_EXTENSION_ID = "io.modelcontextprotocol/skills"
SKILLS_LIST_METHOD = "skills/list"
SKILLS_GET_METHOD = "skills/get"

# Refusals the caller caused by what it sent, which SEP-2640 answers `-32602`.
_INVALID_PARAMS_CODES = frozenset({ToolErrorCode.REQUEST_INVALID, ToolErrorCode.SKILL_NOT_FOUND})

type HandlerGuard = Callable[[str, ExtensionRequestHandler], ExtensionRequestHandler]


class SkillsListParams(RequestParams):
    """`skills/list` params: none beyond `_meta`, since the catalogue fits one page."""


class SkillsGetParams(RequestParams):
    """`skills/get` params: the skill's `SKILL.md` URI."""

    uri: str


class SkillsExtension(ServerExtension):
    """Serves `skills/list` and `skills/get` over the Skills API.

    `guard`, given a method name and its handler, returns the handler the
    server binds, so the registering server can wrap each in its own
    admission, deadline and span. Without one the handlers run bare.
    """

    identifier = SKILLS_EXTENSION_ID

    def __init__(self, *, guard: HandlerGuard | None = None) -> None:
        self._guard = guard

    @override
    def methods(self) -> Sequence[MethodBinding]:
        bindings = (
            (SKILLS_LIST_METHOD, SkillsListParams, _skills_list),
            (SKILLS_GET_METHOD, SkillsGetParams, _skills_get),
        )
        return [
            MethodBinding(
                method=method,
                params_type=params_type,
                handler=self._guard(method, handler) if self._guard else handler,
            )
            for method, params_type, handler in bindings
        ]


async def _skills_list(
    _ctx: ServerRequestContext[Any, Any], _params: SkillsListParams
) -> dict[str, Any]:
    try:
        catalogue = await fetch_skill_list()
    except SkillReadRefused as refused:
        _fail(refused, skill_id="")
    return _complete(
        catalogue,
        skills=[skill_entry(skill) for skill in catalogue.value.skills if skill.searchable],
    )


async def _skills_get(
    _ctx: ServerRequestContext[Any, Any], params: SkillsGetParams
) -> dict[str, Any]:
    parsed = parse_skill_uri(params.uri)
    if parsed is None or parsed[1] != SKILL_MAIN_FILE:
        raise MCPError(
            code=INVALID_PARAMS,
            message=(
                f"{ToolErrorCode.REQUEST_INVALID}: {params.uri!r} is not a skill URI; "
                f"skills/get takes skill://{{id}}/{SKILL_MAIN_FILE}, as skills/list names it."
            ),
        )
    skill_id = parsed[0]
    try:
        skill = await fetch_skill(skill_id)
    except SkillReadRefused as refused:
        _fail(refused, skill_id=skill_id)
    return _complete(skill, skill=skill_entry(skill.value))


def _complete(answer: Cacheable[Any], **result: Any) -> dict[str, Any]:
    return {
        "resultType": "complete",
        **result,
        "ttlMs": answer.max_age_s * 1000,
        "cacheScope": "public",
    }


def _fail(refused: SkillReadRefused, *, skill_id: str) -> NoReturn:
    code = INVALID_PARAMS if refused.refusal.code in _INVALID_PARAMS_CODES else INTERNAL_ERROR
    prose = refusal_prose(refused, skill_id=skill_id)
    raise MCPError(code=code, message=f"{refused.refusal.code}: {prose}") from None
