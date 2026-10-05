"""Tests for the `skill://` resources data-mcp serves over the Skills API."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import httpx2
import pytest
from data_mcp import client_holder
from data_mcp.server import mcp
from fastmcp import Client
from fastmcp.utilities.skills import download_skill, list_skills
from mcp import MCPError
from mcp.types import TextResourceContents

from .net_guard import no_network  # noqa: F401 -- imported for its autouse fixture
from .skills_fixtures import (
    REFERENCES,
    SKILL_ID,
    SKILL_MD,
    Handler,
    digest_of,
    file_body,
    list_entry,
    refusal,
    refusing,
    serving_skill,
)
from .test_client import _client


@asynccontextmanager
async def _connected(handler: Handler) -> AsyncIterator[Client]:
    async with _client(handler) as upstream:
        client_holder.set_client(upstream)
        async with Client(mcp) as client:
            yield client


async def _read(handler: Handler, uri: str) -> TextResourceContents:
    async with _connected(handler) as client:
        [content] = await client.read_resource(uri)
    assert isinstance(content, TextResourceContents)
    return content


async def _refused(handler: Handler, uri: str) -> str:
    async with _connected(handler) as client:
        with pytest.raises(MCPError) as raised:
            await client.read_resource(uri)
    return str(raised.value)


# Listing


async def test_each_skill_search_can_name_lists_its_skill_md_and_manifest() -> None:
    catalogue = [
        list_entry(skill_id="oda-concepts"),
        list_entry(skill_id="dfo-faq", attached=False),
        list_entry(skill_id="oda-channels", excluded=True),
        list_entry(skill_id="oda-measures"),
    ]

    async with _connected(serving_skill(catalogue=catalogue)) as client:
        listed = await client.list_resources()

    assert [(str(r.uri), r.mime_type, r.description) for r in listed] == [
        (
            "skill://oda-concepts/SKILL.md",
            "text/markdown",
            "Core ODA concepts: grant equivalents.",
        ),
        ("skill://oda-concepts/_manifest", "application/json", "File listing for oda-concepts"),
        (
            "skill://oda-measures/SKILL.md",
            "text/markdown",
            "Core ODA concepts: grant equivalents.",
        ),
        ("skill://oda-measures/_manifest", "application/json", "File listing for oda-measures"),
    ]


async def test_fastmcps_list_skills_helper_reads_the_listed_skills() -> None:
    async with _connected(serving_skill()) as client:
        skills = await list_skills(client)

    assert [(skill.name, skill.description) for skill in skills] == [
        (SKILL_ID, "Core ODA concepts: grant equivalents.")
    ]


async def test_a_list_read_that_fails_lists_no_skills() -> None:
    async with _connected(refusing(503, "service_unavailable")) as client:
        listed = await client.list_resources()

    assert listed == []


async def test_the_file_template_is_listed_beside_the_artifact_template() -> None:
    async with _connected(serving_skill()) as client:
        templates = await client.list_resource_templates()

    assert "skill://{skill_id}/{path*}" in [template.uri_template for template in templates]


# Reading


async def test_skill_md_reads_verbatim_from_the_resources_door() -> None:
    requests: list[httpx2.Request] = []

    content = await _read(serving_skill(requests=requests), f"skill://{SKILL_ID}/SKILL.md")

    assert [request.url.path for request in requests] == [
        f"/api/skills/{SKILL_ID}/resources/SKILL.md"
    ]
    assert (content.text, content.mime_type) == (SKILL_MD, "text/markdown")


async def test_the_manifest_lists_every_file_with_its_size_and_digest() -> None:
    content = await _read(serving_skill(), f"skill://{SKILL_ID}/_manifest")

    files = {"SKILL.md": SKILL_MD, **REFERENCES}
    assert content.mime_type == "application/json"
    assert json.loads(content.text) == {
        "skill": SKILL_ID,
        "files": [
            {"path": path, "size": len(text.encode()), "hash": digest_of(text)}
            for path, text in files.items()
        ],
    }


async def test_a_template_read_carries_the_media_type_the_skills_api_reports() -> None:
    path = "references/core-oda.md"
    served = serving_skill()

    def handler(request: httpx2.Request) -> httpx2.Response:
        if request.url.path.endswith(f"/resources/{path}"):
            return httpx2.Response(
                200, json={**file_body(path, REFERENCES[path]), "media_type": "text/x-reference"}
            )
        return served(request)

    content = await _read(handler, f"skill://{SKILL_ID}/{path}")

    assert (content.text, content.mime_type) == (REFERENCES[path], "text/x-reference")


async def test_a_read_serves_a_skill_the_list_leaves_out() -> None:
    catalogue = [list_entry(attached=False)]

    content = await _read(serving_skill(catalogue=catalogue), f"skill://{SKILL_ID}/SKILL.md")

    assert content.text == SKILL_MD


async def test_fastmcps_download_skill_helper_fetches_every_file(tmp_path: Path) -> None:
    async with _connected(serving_skill()) as client:
        skill_dir = await download_skill(client, SKILL_ID, tmp_path)

    written = {
        str(path.relative_to(skill_dir)): path.read_text()
        for path in skill_dir.rglob("*")
        if path.is_file()
    }
    assert written == {"SKILL.md": SKILL_MD, **REFERENCES}


# Refusals, each written for the caller


def _unmounted(request: httpx2.Request) -> httpx2.Response:
    del request
    return refusal(404, "not_found")


@pytest.mark.parametrize(
    ("handler", "uri", "message"),
    [
        (
            serving_skill(),
            "skill://no-such-skill/SKILL.md",
            "skill_not_found: No skill has the id 'no-such-skill'.",
        ),
        (
            serving_skill(),
            "skill://no-such-skill/_manifest",
            "skill_not_found: No skill has the id 'no-such-skill'.",
        ),
        (
            serving_skill(),
            f"skill://{SKILL_ID}/references/missing.md",
            f"skill_resource_not_found: Skill '{SKILL_ID}' has no file "
            "'references/missing.md'; get_skill lists its files.",
        ),
        (
            _unmounted,
            f"skill://{SKILL_ID}/SKILL.md",
            "capability_unavailable: This deployment serves no skills.",
        ),
        (
            refusing(503, "service_unavailable"),
            f"skill://{SKILL_ID}/references/core-oda.md",
            "service_unavailable: The skill catalogue is still loading; retry shortly.",
        ),
    ],
    ids=["unknown-skill", "unknown-skill-manifest", "unknown-file", "unmounted", "building"],
)
async def test_each_skills_api_refusal_fails_the_read_as_written(
    handler: Handler, uri: str, message: str
) -> None:
    assert await _refused(handler, uri) == message


async def test_a_malformed_skill_id_fails_before_any_request() -> None:
    requests: list[httpx2.Request] = []

    message = await _refused(serving_skill(requests=requests), "skill://Not_An_Id/SKILL.md")

    assert requests == []
    assert message.startswith("request_invalid: ")
    assert "'Not_An_Id' is not a skill id" in message


async def test_an_unmounted_skills_api_is_confirmed_by_its_healthz_probe() -> None:
    requests: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        return refusal(404, "not_found")

    await _refused(handler, f"skill://{SKILL_ID}/SKILL.md")

    assert [request.url.path for request in requests][-1] == "/api/skills/healthz"
