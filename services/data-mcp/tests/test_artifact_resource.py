"""Tests for the artifact resource: the snapshot read, the download link, and the JSON-RPC refusal."""

from __future__ import annotations

import json
from collections.abc import Callable, Coroutine
from typing import Any

import httpx2
import pytest
from data_mcp import client_holder
from data_mcp.server import mcp
from fastmcp import Client
from mcp import MCPError

from .net_guard import no_network  # noqa: F401 -- imported for its autouse fixture
from .test_client import _client, _problem_body, _settings

Handler = (
    Callable[[httpx2.Request], httpx2.Response]
    | Callable[[httpx2.Request], Coroutine[None, None, httpx2.Response]]
)

_SUBJECT_ID = "spec_" + "1" * 32
_CONTENT_VERSION = "cv_1"
_ARTIFACT_DIGEST = "sha256:" + "1" * 64
_LINK = f"/api/datasets/{_SUBJECT_ID}/{_CONTENT_VERSION}/files/{_ARTIFACT_DIGEST}"
# `_client` builds its `UpstreamClient` over `Settings.dfo_apis_url`'s default.
_ORIGIN = "http://localhost:8083"


def _uri(
    *,
    subject_id: str = _SUBJECT_ID,
    content_version: str = _CONTENT_VERSION,
    artifact_digest: str = _ARTIFACT_DIGEST,
) -> str:
    return f"dfo://artifact/{subject_id}/{content_version}/{artifact_digest}"


def _snapshot_response(
    *,
    subject_id: str = _SUBJECT_ID,
    content_version: str = _CONTENT_VERSION,
    artifact_digest: str = _ARTIFACT_DIGEST,
    url: str = _LINK,
    tier: str = "working",
) -> dict[str, Any]:
    """Mirrors `test_get_data_tool.py`'s `_manifest_response`, but with `url` populated --
    this is the snapshot route's shape, the same `Manifest` shape mint returns, with each
    artifact's `url` the root-relative stable link.
    """
    return {
        "schema_version": 1,
        "definition": {
            "schema_version": 1,
            "spec_id": subject_id,
            "identity": {},
        },
        "content_version": content_version,
        "coverage": {"rows": 1, "dimensions": []},
        "citation": {
            "text": "OECD DAC2a",
            "sources": [{"ref": "OECD_DAC2A", "label": "OECD DAC2a"}],
        },
        "trace": {},
        "warnings": [],
        "artifacts": [
            {
                "artifact_digest": artifact_digest,
                "rendering": {"bundle": False, "format": "parquet", "layout": "frame"},
                "bytes": 100,
                "media_type": "application/octet-stream",
                "created_at": "2026-08-11T00:00:00+00:00",
                "tier": tier,
                "url": url,
            }
        ],
    }


async def _call_read_artifact(
    handler: Handler, uri: str, *, site_origin: str | None = None
) -> list[Any]:
    async with _client(handler, settings=_settings(dfo_site_origin=site_origin)) as upstream:
        client_holder.set_client(upstream)
        async with Client(mcp) as client:
            return await client.read_resource(uri)


def _doc(contents: list[Any]) -> dict[str, Any]:
    assert len(contents) == 1
    return json.loads(contents[0].text)


# Happy path


async def test_happy_path_reads_the_absolute_download_link_and_metadata() -> None:
    requests: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        return httpx2.Response(200, json=_snapshot_response())

    contents = await _call_read_artifact(handler, _uri())

    assert len(requests) == 1
    assert requests[0].method == "GET"
    assert requests[0].url.path == f"/api/datasets/{_SUBJECT_ID}/{_CONTENT_VERSION}"
    assert requests[0].url.query == b""
    assert requests[0].content == b""

    assert contents[0].mime_type == "application/json"
    doc = _doc(contents)
    assert doc["subject_id"] == _SUBJECT_ID
    assert doc["content_version"] == _CONTENT_VERSION
    assert doc["artifact_digest"] == _ARTIFACT_DIGEST
    assert doc["url"] == _ORIGIN + _LINK
    assert doc["media_type"] == "application/octet-stream"
    assert doc["bytes"] == 100
    assert doc["tier"] == "working"
    assert doc["rendering"] == {"bundle": False, "format": "parquet", "layout": "frame"}
    assert "url_expires_at" not in doc
    assert "expires_in_s" not in doc


async def test_the_link_sits_on_the_site_origin_when_one_is_set() -> None:
    """The website proxies the same root-relative path to dfo-apis with its own ID
    token. The snapshot read itself goes to dfo-apis."""
    requests: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        return httpx2.Response(200, json=_snapshot_response())

    contents = await _call_read_artifact(handler, _uri(), site_origin="https://site.example")

    assert _doc(contents)["url"] == "https://site.example" + _LINK
    assert [r.url.host for r in requests] == ["localhost"]


async def test_call_uses_retry_true_and_the_default_timeout() -> None:
    requests: list[httpx2.Request] = []
    attempts = 0

    def handler(request: httpx2.Request) -> httpx2.Response:
        nonlocal attempts
        requests.append(request)
        attempts += 1
        if attempts <= 2:
            return httpx2.Response(
                503, content=_problem_body(code="service_unavailable", retryable=True)
            )
        return httpx2.Response(200, json=_snapshot_response())

    contents = await _call_read_artifact(handler, _uri())

    assert attempts == 3
    doc = _doc(contents)
    assert doc["url"] == _ORIGIN + _LINK
    # No per-call override was passed, so every attempt used the client's own
    # default read timeout rather than mint's 125s override.
    for request in requests:
        assert request.extensions["timeout"]["read"] == 1.0


# Aged out: the upstream 410 `artifact_expired` code


async def test_aged_out_410_refuses_with_artifact_expired_marker() -> None:
    def handler(_request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(410, content=_problem_body(code="artifact_expired", retryable=False))

    with pytest.raises(MCPError) as exc_info:
        await _call_read_artifact(handler, _uri())

    assert exc_info.value.error.message.startswith("artifact_expired: ")


# Absent from the snapshot: a digest this content_version never minted, its own marker


async def test_digest_absent_from_snapshot_refuses_with_its_own_marker() -> None:
    payload = _snapshot_response(artifact_digest="sha256:" + "2" * 64)

    def handler(_request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, json=payload)

    with pytest.raises(MCPError) as exc_info:
        await _call_read_artifact(handler, _uri())

    message = exc_info.value.error.message
    assert message.startswith("artifact_not_in_snapshot: ")
    assert "aged out" not in message


# The rest of the snapshot vocabulary, mirrored verbatim


@pytest.mark.parametrize(
    ("code", "status"),
    [
        ("spec_not_found", 404),
        ("snapshot_not_found", 404),
        ("snapshot_revoked", 410),
    ],
)
async def test_upstream_not_found_and_revoked_codes_map_to_their_own_marker(
    code: str, status: int
) -> None:
    def handler(_request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(status, content=_problem_body(code=code, retryable=False))

    with pytest.raises(MCPError) as exc_info:
        await _call_read_artifact(handler, _uri())

    assert exc_info.value.error.message.startswith(f"{code}: ")


async def test_503_maps_to_service_unavailable() -> None:
    def handler(_request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(
            503, content=_problem_body(code="service_unavailable", retryable=True)
        )

    with pytest.raises(MCPError) as exc_info:
        await _call_read_artifact(handler, _uri())

    assert exc_info.value.error.message.startswith("service_unavailable: ")


async def test_transport_failure_maps_to_service_unavailable() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ConnectError("boom", request=request)

    with pytest.raises(MCPError) as exc_info:
        await _call_read_artifact(handler, _uri())

    assert exc_info.value.error.message.startswith("service_unavailable: ")


async def test_non_retryable_500_internal_error_maps_to_upstream_response_unusable() -> None:
    def handler(_request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(500, content=_problem_body(code="internal_error", retryable=False))

    with pytest.raises(MCPError) as exc_info:
        await _call_read_artifact(handler, _uri())

    assert exc_info.value.error.message.startswith("upstream_response_unusable: ")


async def test_retryable_500_internal_error_maps_to_service_unavailable() -> None:
    def handler(_request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(500, content=_problem_body(code="internal_error", retryable=True))

    with pytest.raises(MCPError) as exc_info:
        await _call_read_artifact(handler, _uri())

    assert exc_info.value.error.message.startswith("service_unavailable: ")


async def test_opaque_non_retryable_500_maps_to_upstream_response_unusable() -> None:
    """A 500 with no parseable problem body carries `code=None`, `retryable=False`
    (500 falls outside the client's fixed 502/503/504 retryable fallback) -- the
    same non-retryable verdict `internal_error` gets above, so the catch-all must
    honour it the same way rather than promoting to `service_unavailable`.
    """

    def handler(_request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(500, content=b"<html>server error</html>")

    with pytest.raises(MCPError) as exc_info:
        await _call_read_artifact(handler, _uri())

    assert exc_info.value.error.message.startswith("upstream_response_unusable: ")


@pytest.mark.parametrize("status", [502, 503, 504])
async def test_bodyless_5xx_in_the_retryable_fallback_set_still_maps_to_service_unavailable(
    status: int,
) -> None:
    def handler(_request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(status, content=b"")

    with pytest.raises(MCPError) as exc_info:
        await _call_read_artifact(handler, _uri())

    assert exc_info.value.error.message.startswith("service_unavailable: ")


@pytest.mark.parametrize(
    ("retryable", "expected_marker"),
    [(True, "service_unavailable"), (False, "upstream_response_unusable")],
)
async def test_unmapped_code_at_5xx_splits_on_retryable(
    retryable: bool, expected_marker: str
) -> None:
    def handler(_request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(
            500, content=_problem_body(code="a_code_this_module_never_named", retryable=retryable)
        )

    with pytest.raises(MCPError) as exc_info:
        await _call_read_artifact(handler, _uri())

    assert exc_info.value.error.message.startswith(f"{expected_marker}: ")


# No tool-error grammar on this path


async def test_resource_read_failure_carries_no_tool_error_grammar() -> None:
    """The JSON-RPC `code` fastmcp assigns is always `0` and carries no meaning -- the
    machine-readable marker is the message prefix, asserted elsewhere. This test's own
    business is the negative: none of the tool path's retry-mode vocabulary or structured
    `fix` payload leaks onto this wire, which has nowhere to put either.
    """

    def handler(_request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(410, content=_problem_body(code="artifact_expired", retryable=False))

    with pytest.raises(MCPError) as exc_info:
        await _call_read_artifact(handler, _uri())

    message = exc_info.value.error.message
    for banned in (
        "retry_mode",
        "correct_and_retry",
        "refresh_then_retry",
        "retry_after_delay",
        "do_not_retry",
    ):
        assert banned not in message
    assert exc_info.value.error.data is None


# Malformed / ambiguous snapshot responses -- never leak the link


async def test_malformed_artifact_entry_refuses_without_leaking_its_link() -> None:
    payload = _snapshot_response()
    payload["artifacts"] = [{"url": "/api/should-not-leak"}]

    def handler(_request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, json=payload)

    with pytest.raises(MCPError) as exc_info:
        await _call_read_artifact(handler, _uri())

    assert exc_info.value.error.message.startswith("upstream_response_unusable: ")
    assert "/api/should-not-leak" not in exc_info.value.error.message


async def test_two_entries_sharing_a_digest_read_the_first_ones_bytes() -> None:
    payload = _snapshot_response()
    other_rendering = dict(payload["artifacts"][0])
    other_rendering["rendering"] = {"bundle": False, "format": "parquet", "layout": "long"}
    other_rendering["url"] = "/api/a-second-rendering"
    payload["artifacts"].append(other_rendering)

    def handler(_request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, json=payload)

    doc = _doc(await _call_read_artifact(handler, _uri()))

    assert doc["url"] == _ORIGIN + _LINK
    assert doc["rendering"] == {"bundle": False, "format": "parquet", "layout": "frame"}


@pytest.mark.parametrize(
    "bad_url",
    [
        "",
        "   ",
        "files/relative",
        "https://signed.example/absolute",
        "//other-origin.example/api/datasets",
        "http://[::1",
        "https:///path",
    ],
)
async def test_a_url_that_is_not_a_root_relative_link_refuses_without_leaking_its_value(
    bad_url: str,
) -> None:
    payload = _snapshot_response(url=bad_url)

    def handler(_request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, json=payload)

    with pytest.raises(MCPError) as exc_info:
        await _call_read_artifact(handler, _uri())

    message = exc_info.value.error.message
    assert message.startswith("upstream_response_unusable: ")
    if bad_url:
        assert bad_url not in message


# Local validation of the three URI parts -- no upstream call


@pytest.mark.parametrize(
    "uri",
    [
        _uri(subject_id="not-a-spec-id"),
        _uri(content_version="not-a-content-version"),
        _uri(artifact_digest="not-a-digest"),
    ],
)
async def test_malformed_uri_parts_refuse_locally_with_no_upstream_call(uri: str) -> None:
    requests: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        raise AssertionError("local validation should have refused before any upstream call")

    with pytest.raises(MCPError) as exc_info:
        await _call_read_artifact(handler, uri)

    assert exc_info.value.error.message.startswith("request_invalid: ")
    assert requests == []


# The note, and the document's own JSON shape


async def test_the_document_is_valid_json_with_the_right_mime_type_and_a_note() -> None:
    def handler(_request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, json=_snapshot_response())

    contents = await _call_read_artifact(handler, _uri())

    assert contents[0].mime_type == "application/json"
    doc = _doc(contents)
    assert "does not expire" in doc["note"]
