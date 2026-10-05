"""A stubbed Skills API in dfo-apis' wire shape, for the skill tool, resource and
extension tests."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from typing import Any

import httpx2
from data_mcp import client_holder
from data_mcp.server import mcp
from fastmcp import Client
from fastmcp.client.client import CallToolResult

from .test_client import _client, _problem_body

Handler = Callable[[httpx2.Request], httpx2.Response]

SKILL_ID = "oda-concepts"
SKILL_MD = (
    "---\nname: oda-concepts\ndescription: >-\n  Core ODA concepts: grant equivalents.\n"
    "version: 1.10\n---\n\n# ODA concepts\n\nRead references/core-oda.md first.\n"
)
# The `Cache-Control: max-age` every 200 from the stub carries.
MAX_AGE_S = 120
CACHE_CONTROL = {"cache-control": f"public, max-age={MAX_AGE_S}"}

REFERENCES = {
    "references/core-oda.md": "# Core ODA\n\n" + "Grant equivalents count loans.\n" * 180,
    "references/donor-groupings.md": "# Donor groupings\n\nDAC members.\n",
}


def digest_of(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode()).hexdigest()


def _manifest_entry(path: str, text: str) -> dict[str, Any]:
    return {
        "path": path,
        "size": len(text.encode()),
        "digest": digest_of(text),
        "media_type": "text/markdown",
    }


def skill_body(
    *, content_text: str = SKILL_MD, references: dict[str, str] | None = None
) -> dict[str, Any]:
    """`GET /api/skills/{skill_id}` as the Skills API serves it."""
    files = REFERENCES if references is None else references
    return {
        "id": SKILL_ID,
        "name": SKILL_ID,
        "description": "Core ODA concepts: grant equivalents.",
        "frontmatter": {
            "name": SKILL_ID,
            "description": "Core ODA concepts: grant equivalents.",
            "version": "1.10",
        },
        "content_version": "sha256:" + "c" * 64,
        "url": f"https://github.com/ONEcampaign/dfo-skills/tree/main/skills/{SKILL_ID}",
        "attached": True,
        "excluded": False,
        "statvar_count": 4695,
        "statvar_count_truncated": False,
        "warnings": [],
        "content_text": content_text,
        "resources": [
            _manifest_entry("SKILL.md", content_text),
            *(_manifest_entry(path, text) for path, text in sorted(files.items())),
        ],
    }


def list_entry(
    *, skill_id: str = SKILL_ID, attached: bool = True, excluded: bool = False
) -> dict[str, Any]:
    """One `GET /api/skills` entry: the skill door's body without `content_text`."""
    entry = {key: value for key, value in skill_body().items() if key != "content_text"}
    return {
        **entry,
        "id": skill_id,
        "name": skill_id,
        "frontmatter": {**entry["frontmatter"], "name": skill_id},
        "attached": attached,
        "excluded": excluded,
    }


def file_body(path: str, text: str) -> dict[str, Any]:
    """`GET /api/skills/{skill_id}/resources/{path}` as the Skills API serves it."""
    return {"skill_id": SKILL_ID, **_manifest_entry(path, text), "content_text": text}


def serving_skill(
    *,
    content_text: str = SKILL_MD,
    references: dict[str, str] | None = None,
    requests: list[httpx2.Request] | None = None,
    catalogue: list[dict[str, Any]] | None = None,
) -> Handler:
    """A Skills API holding one skill, answering the list, skill and resources doors.

    The list door answers `catalogue`, by default the one skill's entry.
    """
    files = {"SKILL.md": content_text, **(REFERENCES if references is None else references)}
    prefix = f"/api/skills/{SKILL_ID}"

    listed = [list_entry()] if catalogue is None else catalogue
    bodies = {
        "/api/skills/healthz": {"status": "ok"},
        "/api/skills": {"skills": listed, "warnings": []},
        prefix: skill_body(content_text=content_text, references=references),
        **{f"{prefix}/resources/{path}": file_body(path, text) for path, text in files.items()},
    }

    def handler(request: httpx2.Request) -> httpx2.Response:
        if requests is not None:
            requests.append(request)
        path = request.url.path
        if path in bodies:
            return httpx2.Response(200, json=bodies[path], headers=CACHE_CONTROL)
        if path.startswith(f"{prefix}/resources/"):
            return refusal(404, "skill_resource_not_found")
        if path.startswith("/api/skills/"):
            return refusal(404, "skill_not_found")
        return refusal(404, "not_found")

    return handler


def refusal(status: int, code: str) -> httpx2.Response:
    return httpx2.Response(status, content=_problem_body(code=code, retryable=False))


def refusing(status: int, code: str) -> Handler:
    return lambda _request: refusal(status, code)


async def call_tool(handler: Handler, tool: str, arguments: dict[str, Any]) -> CallToolResult:
    async with _client(handler) as upstream:
        client_holder.set_client(upstream)
        async with Client(mcp) as client:
            return await client.call_tool(tool, arguments, raise_on_error=False)


def refusal_of(result: CallToolResult) -> dict[str, Any]:
    assert result.is_error is True
    assert result.structured_content is not None
    return result.structured_content
