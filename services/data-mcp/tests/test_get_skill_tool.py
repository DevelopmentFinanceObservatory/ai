"""Tests for the `get_skill` tool: SKILL.md, the file list, the SEP-2640 entry, and refusals."""

from __future__ import annotations

import httpx2
import pytest
from data_mcp.server import mcp
from data_mcp.skills_client import PAGE_CHARS
from fastmcp.client.client import CallToolResult
from mcp.types import ResourceLink

from .net_guard import no_network  # noqa: F401 -- imported for its autouse fixture
from .skills_fixtures import (
    REFERENCES,
    SKILL_ID,
    SKILL_MD,
    Handler,
    call_tool,
    digest_of,
    refusal_of,
    refusing,
    serving_skill,
    skill_body,
)
from .test_client import _problem_body, text_of


async def _get_skill(handler: Handler, skill_id: str = SKILL_ID) -> CallToolResult:
    return await call_tool(handler, "get_skill", {"skill_id": skill_id})


# A served skill


async def test_skill_md_renders_verbatim_then_the_file_list() -> None:
    requests: list[httpx2.Request] = []

    result = await _get_skill(serving_skill(requests=requests))

    assert [request.url.path for request in requests] == [f"/api/skills/{SKILL_ID}"]
    assert result.is_error is False
    assert text_of(result) == (
        SKILL_MD
        + "\n"
        + 'Files in this skill; read one with get_skill_file(skill_id="oda-concepts", path=…):\n'
        + "- references/core-oda.md (5.5 KB)\n"
        + "- references/donor-groupings.md (32 B)"
    )


async def test_each_file_after_skill_md_rides_as_a_resource_link() -> None:
    result = await _get_skill(serving_skill())

    links = [block for block in result.content if isinstance(block, ResourceLink)]
    assert [(str(link.uri), link.name, link.mime_type) for link in links] == [
        (
            "skill://oda-concepts/references/core-oda.md",
            "references/core-oda.md",
            "text/markdown",
        ),
        (
            "skill://oda-concepts/references/donor-groupings.md",
            "references/donor-groupings.md",
            "text/markdown",
        ),
    ]
    assert links[0].size == len(REFERENCES["references/core-oda.md"].encode())


async def test_the_structured_result_is_the_sep_2640_skill_entry() -> None:
    result = await _get_skill(serving_skill())

    body = skill_body()
    assert result.structured_content == {
        "uri": "skill://oda-concepts/SKILL.md",
        "frontmatter": body["frontmatter"],
        "resources": [
            {
                "uri": f"skill://oda-concepts/{entry['path']}",
                "digest": entry["digest"],
                "size": entry["size"],
            }
            for entry in body["resources"]
        ],
    }


async def test_a_skill_with_no_other_files_renders_skill_md_alone() -> None:
    result = await _get_skill(serving_skill(references={}))

    assert text_of(result) == SKILL_MD
    assert len(result.content) == 1


async def test_a_long_skill_md_returns_its_first_page_pointing_at_get_skill_file() -> None:
    line = "x" * 99 + "\n"
    content_text = line * (PAGE_CHARS // len(line) + 50)

    result = await _get_skill(serving_skill(content_text=content_text, references={}))

    page_end = PAGE_CHARS
    assert text_of(result) == (
        content_text[:page_end]
        + f"\nThis page ends at character {page_end:,} of {len(content_text):,}. Read the rest "
        + f'with get_skill_file(skill_id="oda-concepts", path="SKILL.md", offset={page_end}, '
        + f'digest="{digest_of(content_text)}").'
    )


async def test_the_description_carries_the_two_skill_loading_sentences() -> None:
    tools = {tool.name: tool for tool in await mcp.list_tools(run_middleware=False)}

    description = " ".join((tools["get_skill"].description or "").split())
    assert "Loading a skill returns instructions and does not finish the task" in description
    assert "A file refused as not found is not retried under a guessed path" in description


# Refusals


async def test_an_unknown_id_refuses_skill_not_found_pointing_at_search() -> None:
    result = await _get_skill(serving_skill(), "no-such-skill")

    refusal = refusal_of(result)
    assert refusal["code"] == "skill_not_found"
    assert refusal["retry_mode"] == "correct_and_retry"
    assert refusal["fix"] == {"skill_id": "no-such-skill", "next_step": "search"}
    assert refusal["trace_id"] == "trace-1"
    assert "search result" in text_of(result)


async def test_an_unmounted_skills_api_refuses_capability_unavailable() -> None:
    requests: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        return refusing(404, "not_found")(request)

    result = await _get_skill(handler)

    refusal = refusal_of(result)
    assert refusal["code"] == "capability_unavailable"
    assert refusal["retry_mode"] == "do_not_retry"
    assert refusal["fix"] == {"detail": "this deployment does not serve the Skills API"}
    assert [request.url.path for request in requests] == [
        f"/api/skills/{SKILL_ID}",
        "/api/skills/healthz",
    ]


async def test_a_bare_not_found_from_a_mounted_skills_api_is_not_called_unmounted() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        if request.url.path == "/api/skills/healthz":
            return httpx2.Response(200, json={"status": "ok"})
        return refusing(404, "not_found")(request)

    result = await _get_skill(handler)

    assert refusal_of(result)["code"] == "upstream_error"


async def test_an_index_still_building_refuses_service_unavailable() -> None:
    def handler(_request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(
            503, content=_problem_body(code="service_unavailable", retryable=False)
        )

    result = await _get_skill(handler)

    refusal = refusal_of(result)
    assert refusal["code"] == "service_unavailable"
    assert refusal["retry_mode"] == "retry_after_delay"


async def test_a_body_that_is_not_a_skill_refuses_upstream_error() -> None:
    result = await _get_skill(lambda _request: httpx2.Response(200, json={"status": "ok"}))

    refusal = refusal_of(result)
    assert refusal["code"] == "upstream_error"
    assert refusal["retry_mode"] == "do_not_retry"


async def test_a_manifest_that_does_not_open_with_skill_md_refuses_upstream_error() -> None:
    body = skill_body()
    body["resources"] = body["resources"][1:]

    result = await _get_skill(lambda _request: httpx2.Response(200, json=body))

    assert refusal_of(result)["code"] == "upstream_error"


async def test_a_body_that_is_not_json_refuses_upstream_error() -> None:
    result = await _get_skill(lambda _request: httpx2.Response(200, content=b"<html>"))

    assert refusal_of(result)["code"] == "upstream_error"


@pytest.mark.parametrize("skill_id", ["dc/skill/x", "has space", "..", "ODA-Concepts", "a--b"])
async def test_a_malformed_id_refuses_request_invalid_without_a_call(skill_id: str) -> None:
    requests: list[httpx2.Request] = []

    result = await _get_skill(serving_skill(requests=requests), skill_id)

    refusal = refusal_of(result)
    assert refusal["code"] == "request_invalid"
    assert refusal["fix"]["pointer"] == "/skill_id"
    assert requests == []


async def test_the_healthz_route_name_refuses_skill_not_found_without_a_call() -> None:
    requests: list[httpx2.Request] = []

    result = await _get_skill(serving_skill(requests=requests), "healthz")

    assert refusal_of(result)["code"] == "skill_not_found"
    assert requests == []
