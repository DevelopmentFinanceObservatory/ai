"""Tests for the Skills over MCP extension: `skills/list` and `skills/get`."""

from __future__ import annotations

from typing import Any

import httpx2
import pytest
from data_mcp import client_holder
from data_mcp.skills_extension import (
    SKILLS_EXTENSION_ID,
    SkillsExtension,
    SkillsGetParams,
    SkillsListParams,
)
from fastmcp import Client, FastMCP
from fastmcp.server.extensions import ExtensionRequestHandler
from mcp import MCPError
from mcp import types as mt
from mcp_types import INTERNAL_ERROR, INVALID_PARAMS
from pydantic import BaseModel, ConfigDict

from .net_guard import no_network  # noqa: F401 -- imported for its autouse fixture
from .skills_fixtures import (
    MAX_AGE_S,
    REFERENCES,
    SKILL_ID,
    SKILL_MD,
    Handler,
    digest_of,
    list_entry,
    refusal,
    refusing,
    serving_skill,
    skill_body,
)
from .test_client import _client


class _AnyResult(BaseModel):
    """Any result, every field kept as sent."""

    model_config = ConfigDict(extra="allow")


async def _request(
    handler: Handler,
    method: str,
    params: dict[str, Any],
    *,
    extension: SkillsExtension | None = None,
) -> dict[str, Any]:
    server = FastMCP(name="skills")
    server.add_extension(extension or SkillsExtension())
    async with _client(handler) as upstream:
        client_holder.set_client(upstream)
        async with Client(server, mode="2026-07-28") as client:
            # A bare `mt.Request` validates `params` as `RequestParams` and drops
            # every field but `_meta`, so they are sent as the extension's own model.
            params_type = SkillsGetParams if method == "skills/get" else SkillsListParams
            result = await client.session.send_request(
                mt.Request(method=method, params=params_type.model_validate(params)), _AnyResult
            )
    return {key: value for key, value in (result.model_extra or {}).items() if key != "_meta"}


async def _refused(handler: Handler, method: str, params: dict[str, Any]) -> MCPError:
    with pytest.raises(MCPError) as raised:
        await _request(handler, method, params)
    return raised.value


def _entry(skill_id: str = SKILL_ID) -> dict[str, Any]:
    files = {"SKILL.md": SKILL_MD, **REFERENCES}
    return {
        "uri": f"skill://{skill_id}/SKILL.md",
        "frontmatter": {
            "name": skill_id,
            "description": "Core ODA concepts: grant equivalents.",
            "version": "1.10",
        },
        "resources": [
            {
                "uri": f"skill://{skill_id}/{path}",
                "digest": digest_of(text),
                "size": len(text.encode()),
            }
            for path, text in files.items()
        ],
    }


async def test_skills_list_names_only_the_skills_search_can_name() -> None:
    catalogue = [
        list_entry(skill_id="dfo-faq", attached=False),
        list_entry(skill_id="oda-channels", excluded=True),
        list_entry(skill_id="oda-concepts"),
    ]
    requests: list[httpx2.Request] = []

    result = await _request(
        serving_skill(catalogue=catalogue, requests=requests), "skills/list", {}
    )

    assert [request.url.path for request in requests] == ["/api/skills"]
    assert result == {
        "resultType": "complete",
        "skills": [_entry("oda-concepts")],
        "ttlMs": MAX_AGE_S * 1000,
        "cacheScope": "public",
    }


async def test_skills_get_returns_one_skills_entry_by_its_skill_md_uri() -> None:
    catalogue = [list_entry(attached=False)]

    result = await _request(
        serving_skill(catalogue=catalogue), "skills/get", {"uri": f"skill://{SKILL_ID}/SKILL.md"}
    )

    assert result == {
        "resultType": "complete",
        "skill": _entry(),
        "ttlMs": MAX_AGE_S * 1000,
        "cacheScope": "public",
    }


async def test_an_answer_without_a_max_age_gives_a_ttl_of_zero() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        del request
        return httpx2.Response(200, json=skill_body())

    result = await _request(handler, "skills/get", {"uri": f"skill://{SKILL_ID}/SKILL.md"})

    assert result["ttlMs"] == 0


@pytest.mark.parametrize(
    "uri",
    [
        "dfo://artifact/x",
        f"skill://{SKILL_ID}",
        f"skill://{SKILL_ID}/references/core-oda.md",
        f"skill://{SKILL_ID}/_manifest",
    ],
)
async def test_skills_get_refuses_a_uri_that_is_not_a_skill_md_uri(uri: str) -> None:
    requests: list[httpx2.Request] = []

    error = await _refused(serving_skill(requests=requests), "skills/get", {"uri": uri})

    assert requests == []
    assert error.code == INVALID_PARAMS
    assert error.message == (
        f"request_invalid: {uri!r} is not a skill URI; skills/get takes "
        "skill://{id}/SKILL.md, as skills/list names it."
    )


async def test_skills_get_refuses_a_malformed_id_as_invalid_params() -> None:
    error = await _refused(serving_skill(), "skills/get", {"uri": "skill://Not_An_Id/SKILL.md"})

    assert error.code == INVALID_PARAMS
    assert error.message.startswith("request_invalid: ")
    assert "'Not_An_Id' is not a skill id" in error.message


async def test_skills_get_refuses_an_unknown_skill_as_invalid_params() -> None:
    error = await _refused(serving_skill(), "skills/get", {"uri": "skill://no-such-skill/SKILL.md"})

    assert (error.code, error.message) == (
        INVALID_PARAMS,
        "skill_not_found: No skill has the id 'no-such-skill'.",
    )


def _unmounted(request: httpx2.Request) -> httpx2.Response:
    del request
    return refusal(404, "not_found")


@pytest.mark.parametrize("method", ["skills/list", "skills/get"])
@pytest.mark.parametrize(
    ("handler", "message"),
    [
        (_unmounted, "capability_unavailable: This deployment serves no skills."),
        (
            refusing(503, "service_unavailable"),
            "service_unavailable: The skill catalogue is still loading; retry shortly.",
        ),
    ],
    ids=["unmounted", "building"],
)
async def test_an_unmounted_or_building_skills_api_is_an_internal_error(
    method: str, handler: Handler, message: str
) -> None:
    error = await _refused(handler, method, {"uri": f"skill://{SKILL_ID}/SKILL.md"})

    assert (error.code, error.message) == (INTERNAL_ERROR, message)


async def test_the_guard_wraps_each_method_the_extension_binds() -> None:
    guarded: list[str] = []

    def guard(method: str, handler: ExtensionRequestHandler) -> ExtensionRequestHandler:
        async def wrapped(ctx: Any, params: Any) -> Any:
            guarded.append(method)
            return await handler(ctx, params)

        return wrapped

    extension = SkillsExtension(guard=guard)
    await _request(serving_skill(), "skills/list", {}, extension=extension)

    assert [binding.method for binding in extension.methods()] == ["skills/list", "skills/get"]
    assert guarded == ["skills/list"]


def test_the_extension_declares_the_skills_identifier_with_no_settings() -> None:
    extension = SkillsExtension()

    assert (extension.identifier, extension.settings()) == (SKILLS_EXTENSION_ID, {})
    assert SKILLS_EXTENSION_ID == "io.modelcontextprotocol/skills"
