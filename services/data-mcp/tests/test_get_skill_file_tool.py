"""Tests for the `get_skill_file` tool: one file verbatim, paging with a digest, and refusals."""

from __future__ import annotations

from typing import Any

import httpx2
import pytest
from data_mcp.server import mcp
from data_mcp.skills_client import PAGE_CHARS
from fastmcp.client.client import CallToolResult

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
)
from .test_client import text_of

_CORE = "references/core-oda.md"


async def _read(handler: Handler, **arguments: Any) -> CallToolResult:
    return await call_tool(handler, "get_skill_file", {"skill_id": SKILL_ID, **arguments})


def _long_text() -> str:
    """Three pages of 100-character lines, the last one short."""
    line = "y" * 99 + "\n"
    return line * (2 * PAGE_CHARS // len(line) + 7)


# A served file


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
        f'get_skill_file(skill_id="{SKILL_ID}", path="{_CORE}", offset={PAGE_CHARS}, '
        f'digest="{digest}").'
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


async def test_the_description_carries_the_two_skill_loading_sentences() -> None:
    tools = {tool.name: tool for tool in await mcp.list_tools(run_middleware=False)}

    description = " ".join((tools["get_skill_file"].description or "").split())
    assert "returns instructions and does not finish the task" in description
    assert "A file refused as not found is not retried under a guessed path" in description


# Refusals


async def test_a_continuation_on_a_changed_file_refuses_with_the_restart_call() -> None:
    text = _long_text()

    result = await _read(
        serving_skill(references={_CORE: text}),
        path=_CORE,
        offset=PAGE_CHARS,
        digest="sha256:" + "0" * 64,
    )

    refusal = refusal_of(result)
    restart = f'get_skill_file(skill_id="{SKILL_ID}", path="{_CORE}", offset=0)'
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
    result = await call_tool(
        serving_skill(), "get_skill_file", {"skill_id": "no-such-skill", "path": "SKILL.md"}
    )

    assert refusal_of(result)["code"] == "skill_not_found"


async def test_an_unmounted_skills_api_refuses_capability_unavailable() -> None:
    result = await _read(refusing(404, "not_found"), path=_CORE)

    assert refusal_of(result)["code"] == "capability_unavailable"


@pytest.mark.parametrize("offset", [len(SKILL_MD), len(SKILL_MD) + 1])
async def test_an_offset_at_or_past_the_end_refuses_request_invalid(offset: int) -> None:
    result = await _read(
        serving_skill(), path="SKILL.md", offset=offset, digest=digest_of(SKILL_MD)
    )

    refusal = refusal_of(result)
    assert refusal["code"] == "request_invalid"
    assert refusal["fix"]["pointer"] == "/offset"


@pytest.mark.parametrize(
    "path", ["../SKILL.md", "/references/core-oda.md", "references\\core-oda.md", "references/"]
)
async def test_a_malformed_path_refuses_request_invalid_without_a_call(path: str) -> None:
    requests: list[httpx2.Request] = []

    result = await _read(serving_skill(requests=requests), path=path)

    refusal = refusal_of(result)
    assert refusal["code"] == "request_invalid"
    assert refusal["fix"]["pointer"] == "/path"
    assert requests == []


async def test_a_continuation_without_a_digest_refuses_request_invalid_without_a_call() -> None:
    """Without the digest a continuation could join pages of two revisions of the file."""
    requests: list[httpx2.Request] = []

    result = await _read(serving_skill(requests=requests), path=_CORE, offset=PAGE_CHARS)

    refusal = refusal_of(result)
    assert refusal["code"] == "request_invalid"
    assert refusal["fix"]["pointer"] == "/digest"
    assert (
        f'get_skill_file(skill_id="{SKILL_ID}", path="{_CORE}", offset=0)' in (refusal["message"])
    )
    assert requests == []


async def test_a_malformed_digest_refuses_request_invalid_without_a_call() -> None:
    requests: list[httpx2.Request] = []

    result = await _read(serving_skill(requests=requests), path=_CORE, offset=1, digest="md5:abc")

    refusal = refusal_of(result)
    assert refusal["code"] == "request_invalid"
    assert refusal["fix"]["pointer"] == "/digest"
    assert requests == []
