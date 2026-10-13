"""Tests for the harness-only `catalogue_counts` tool: the counts sentence and its refusals."""

from __future__ import annotations

import httpx2
import pytest
from data_mcp import client_holder
from data_mcp.server import mcp
from fastmcp import Client
from fastmcp.client.client import CallToolResult

from .net_guard import no_network  # noqa: F401 -- imported for its autouse fixture
from .test_client import _client, _problem_body, text_of
from .test_sources_tool import _COUNTS_SENTENCE, Handler, _search_catalogue


async def _call_catalogue_counts(handler: Handler) -> CallToolResult:
    async with _client(handler) as upstream:
        client_holder.set_client(upstream)
        async with Client(mcp) as client:
            return await client.call_tool("catalogue_counts", {}, raise_on_error=False)


async def test_catalogue_counts_answers_the_counts_sentence_and_the_counts() -> None:
    requests: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        return httpx2.Response(200, json=_search_catalogue())

    result = await _call_catalogue_counts(handler)

    assert result.is_error is False
    assert [request.url.path for request in requests] == ["/api/search/catalogue"]
    assert text_of(result) == _COUNTS_SENTENCE
    assert result.structured_content == {
        "catalogue_version": "2026-10-11T04:00:00Z",
        "counts": _search_catalogue()["counts"],
    }


@pytest.mark.parametrize(
    ("response", "code"),
    [
        (
            httpx2.Response(
                503, content=_problem_body(code="service_unavailable", retryable=False)
            ),
            "service_unavailable",
        ),
        (
            httpx2.Response(200, json={"catalogue_version": "v", "counts": {}}),
            "upstream_error",
        ),
    ],
    ids=["no_catalogue_published", "wrong_shape"],
)
async def test_a_catalogue_that_cannot_be_read_refuses_with_its_code(
    response: httpx2.Response, code: str
) -> None:
    requests: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        return response

    result = await _call_catalogue_counts(handler)

    assert result.is_error is True
    assert result.structured_content is not None
    assert result.structured_content["code"] == code
    # A 503 means no catalogue is published; the read is not repeated.
    assert len(requests) == 1
