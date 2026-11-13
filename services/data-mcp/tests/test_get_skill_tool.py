"""Tests for the `get_skill` tool: SKILL.md, the file list, the SEP-2640 entry, and refusals."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

import httpx2
import pytest
from common.apis_auth import ApisAuthError
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
        + 'Files in this skill; read one with get_skill(skill_id="oda-concepts", path=…):\n'
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


async def test_a_long_skill_md_returns_its_first_page_naming_the_next_call() -> None:
    line = "x" * 99 + "\n"
    content_text = line * (PAGE_CHARS // len(line) + 50)

    result = await _get_skill(serving_skill(content_text=content_text, references={}))

    page_end = PAGE_CHARS
    assert text_of(result) == (
        content_text[:page_end]
        + f"\nThis page ends at character {page_end:,} of {len(content_text):,}. Read the rest "
        + f'with get_skill(skill_id="oda-concepts", path="SKILL.md", offset={page_end}, '
        + f'digest="{digest_of(content_text)}").'
    )


# The public description says what a call returns and what `path`, `offset` and
# `digest` do, in a few sentences.
DESCRIPTION_CEILING = 330


async def test_the_description_stays_short() -> None:
    tools = {tool.name: tool for tool in await mcp.list_tools(run_middleware=False)}

    assert len(tools["get_skill"].description or "") <= DESCRIPTION_CEILING
    assert "get_skill_file" not in tools


async def test_the_path_parameter_says_a_refused_file_is_not_retried_under_a_guess() -> None:
    tools = {tool.name: tool for tool in await mcp.list_tools(run_middleware=False)}

    path = tools["get_skill"].parameters["properties"]["path"]
    assert "not retried under a guessed path" in path["description"]


# A file, by path

_CORE = "references/core-oda.md"


async def _read(handler: Handler, **arguments: Any) -> CallToolResult:
    return await call_tool(handler, "get_skill", {"skill_id": SKILL_ID, **arguments})


def _long_text() -> str:
    """Three pages of 100-character lines, the last one short."""
    line = "y" * 99 + "\n"
    return line * (2 * PAGE_CHARS // len(line) + 7)


async def test_a_reference_is_served_verbatim_from_the_resources_door() -> None:
    requests: list[httpx2.Request] = []

    result = await _read(serving_skill(requests=requests), path=_CORE)

    assert [request.url.path for request in requests] == [
        f"/api/skills/{SKILL_ID}/resources/{_CORE}"
    ]
    text = REFERENCES[_CORE]
    assert text_of(result) == text
    assert result.structured_content == {
        "uri": f"skill://{SKILL_ID}/{_CORE}",
        "path": _CORE,
        "size": len(text.encode()),
        "digest": digest_of(text),
        "offset": 0,
        "next_offset": None,
    }


async def test_skill_md_itself_is_accepted() -> None:
    result = await _read(serving_skill(), path="SKILL.md")

    assert text_of(result) == SKILL_MD


async def test_a_long_file_pages_by_its_digest_to_the_end_skipping_nothing() -> None:
    text = _long_text()
    handler = serving_skill(references={_CORE: text})
    digest = digest_of(text)

    first = await _read(handler, path=_CORE)
    assert first.structured_content is not None
    assert first.structured_content["next_offset"] == PAGE_CHARS
    assert text_of(first).endswith(
        f'get_skill(skill_id="{SKILL_ID}", path="{_CORE}", offset={PAGE_CHARS}, digest="{digest}").'
    )

    second = await _read(handler, path=_CORE, offset=PAGE_CHARS, digest=digest)
    assert second.structured_content is not None
    assert second.structured_content["offset"] == PAGE_CHARS
    assert second.structured_content["next_offset"] == 2 * PAGE_CHARS

    third = await _read(handler, path=_CORE, offset=2 * PAGE_CHARS, digest=digest)
    assert third.structured_content is not None
    assert third.structured_content["next_offset"] is None
    assert text_of(third) == text[2 * PAGE_CHARS :]


async def test_a_page_breaks_at_the_last_line_break_in_its_second_half() -> None:
    text = "a" * 20_000 + "\n" + "b" * 20_000
    handler = serving_skill(references={_CORE: text})

    result = await _read(handler, path=_CORE)

    assert result.structured_content is not None
    assert result.structured_content["next_offset"] == 20_001
    assert text_of(result).startswith("a" * 20_000 + "\n\nThis page ends at character 20,001")


async def test_a_page_with_no_line_break_in_its_second_half_cuts_at_the_page_size() -> None:
    text = "a" * 10_000 + "\n" + "b" * 30_000
    handler = serving_skill(references={_CORE: text})

    result = await _read(handler, path=_CORE)

    assert result.structured_content is not None
    assert result.structured_content["next_offset"] == PAGE_CHARS
    assert text_of(result).startswith(text[:PAGE_CHARS] + "\n\nThis page ends")


async def test_an_empty_file_reads_as_one_empty_page() -> None:
    result = await _read(serving_skill(references={_CORE: ""}), path=_CORE)

    assert result.is_error is False
    assert result.structured_content is not None
    assert result.structured_content["next_offset"] is None


async def test_a_continuation_on_a_changed_file_refuses_with_the_restart_call() -> None:
    text = _long_text()

    result = await _read(
        serving_skill(references={_CORE: text}),
        path=_CORE,
        offset=PAGE_CHARS,
        digest="sha256:" + "0" * 64,
    )

    refusal = refusal_of(result)
    restart = f'get_skill(skill_id="{SKILL_ID}", path="{_CORE}", offset=0)'
    assert refusal["code"] == "upstream_version_changed"
    assert refusal["retry_mode"] == "refresh_then_retry"
    assert refusal["fix"]["digest"] == digest_of(text)
    assert refusal["fix"]["restart"] == restart
    assert restart in text_of(result)


async def test_a_path_the_manifest_lacks_refuses_pointing_at_get_skill() -> None:
    result = await _read(serving_skill(), path="references/guessed.md")

    refusal = refusal_of(result)
    assert refusal["code"] == "skill_resource_not_found"
    assert refusal["retry_mode"] == "correct_and_retry"
    assert refusal["fix"] == {
        "skill_id": SKILL_ID,
        "path": "references/guessed.md",
        "next_step": "get_skill",
    }
    assert "do not retry under a guessed path" in text_of(result)


async def test_an_unknown_skill_refuses_skill_not_found() -> None:
    result = await _read(serving_skill(), skill_id="no-such-skill", path="SKILL.md")

    assert refusal_of(result)["code"] == "skill_not_found"


@pytest.mark.parametrize("offset", [len(SKILL_MD), len(SKILL_MD) + 1])
async def test_an_offset_at_or_past_the_end_refuses_request_invalid(offset: int) -> None:
    result = await _read(
        serving_skill(), path="SKILL.md", offset=offset, digest=digest_of(SKILL_MD)
    )

    refusal = refusal_of(result)
    assert refusal["code"] == "request_invalid"
    assert refusal["fix"]["problems"][0].startswith("offset: ")


@pytest.mark.parametrize(
    "path", ["../SKILL.md", "/references/core-oda.md", "references\\core-oda.md", "references/"]
)
async def test_a_malformed_path_refuses_request_invalid_without_a_call(path: str) -> None:
    requests: list[httpx2.Request] = []

    result = await _read(serving_skill(requests=requests), path=path)

    refusal = refusal_of(result)
    assert refusal["code"] == "request_invalid"
    assert refusal["fix"]["problems"][0].startswith("path: ")
    assert requests == []


async def test_a_continuation_without_a_digest_refuses_request_invalid_without_a_call() -> None:
    """Without the digest a continuation could join pages of two revisions of the file."""
    requests: list[httpx2.Request] = []

    result = await _read(serving_skill(requests=requests), path=_CORE, offset=PAGE_CHARS)

    refusal = refusal_of(result)
    assert refusal["code"] == "request_invalid"
    assert refusal["fix"]["problems"][0].startswith("digest: ")
    assert f'get_skill(skill_id="{SKILL_ID}", path="{_CORE}", offset=0)' in (refusal["message"])
    assert requests == []


async def test_a_malformed_digest_refuses_request_invalid_without_a_call() -> None:
    requests: list[httpx2.Request] = []

    result = await _read(serving_skill(requests=requests), path=_CORE, offset=1, digest="md5:abc")

    refusal = refusal_of(result)
    assert refusal["code"] == "request_invalid"
    assert refusal["fix"]["problems"][0].startswith("digest: ")
    assert requests == []


async def test_an_offset_without_a_path_refuses_request_invalid_without_a_call() -> None:
    requests: list[httpx2.Request] = []

    result = await call_tool(
        serving_skill(requests=requests),
        "get_skill",
        {"skill_id": SKILL_ID, "offset": PAGE_CHARS, "digest": digest_of(SKILL_MD)},
    )

    refusal = refusal_of(result)
    assert refusal["code"] == "request_invalid"
    assert refusal["fix"]["problems"][0].startswith("path: ")
    assert requests == []


# Refusals


async def test_an_unknown_id_refuses_skill_not_found_pointing_at_search(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING, logger="data_mcp.tool_errors"):
        result = await _get_skill(serving_skill(), "no-such-skill")

    refusal = refusal_of(result)
    assert refusal["code"] == "skill_not_found"
    assert refusal["retry_mode"] == "correct_and_retry"
    assert refusal["fix"] == {"skill_id": "no-such-skill", "next_step": "search"}
    assert "trace_id" not in refusal
    [record] = [r for r in caplog.records if r.name == "data_mcp.tool_errors"]
    assert record.levelno == logging.WARNING
    assert "trace-1" in record.getMessage()
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


async def test_a_failed_token_mint_refuses_service_unavailable_and_sends_nothing() -> None:
    requests: list[httpx2.Request] = []

    async def no_token() -> Mapping[str, str]:
        raise ApisAuthError("metadata server answered 500")

    result = await call_tool(
        serving_skill(requests=requests), "get_skill", {"skill_id": SKILL_ID}, auth=no_token
    )

    refusal = refusal_of(result)
    assert refusal["code"] == "service_unavailable"
    assert refusal["retry_mode"] == "retry_after_delay"
    assert requests == []


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
    assert refusal["fix"]["problems"][0].startswith("skill_id: ")
    assert requests == []


async def test_the_healthz_route_name_refuses_skill_not_found_without_a_call() -> None:
    requests: list[httpx2.Request] = []

    result = await _get_skill(serving_skill(requests=requests), "healthz")

    assert refusal_of(result)["code"] == "skill_not_found"
    assert requests == []
