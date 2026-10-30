"""Tests for the `export_data` tool against a faked dfo-apis exports door."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import httpx2
import pytest
from common.caller import (
    CALLER_HEADER,
    CALLER_PRINCIPAL_HEADER,
    conversation_caller,
    supplied_caller,
)
from data_mcp import client_holder
from data_mcp.server import mcp
from fastmcp import Client
from fastmcp.client.client import CallToolResult
from mcp.types import ResourceLink

from .net_guard import no_network  # noqa: F401 -- imported for its autouse fixture
from .test_client import _client, _problem_body, _settings, text_of

Handler = Callable[[httpx2.Request], httpx2.Response]

SITE = "https://developmentfinance.org"
EXPORT_ID = "exp_" + "a" * 32
SIGNED = "https://storage.googleapis.com/pinned/sha256:abc?X-Goog-Signature=secret"
PRINCIPAL = "principal-secret"
ADDRESS = "ip:" + "0" * 32

FETCHED = {"spec_id": "spec_" + "1" * 32, "content_version": "cv_" + "2" * 32}
DERIVED = {"spec_id": "ops_" + "3" * 32, "content_version": "ocv_" + "4" * 32}
# The version the exports door resolves a bare spec_id to: the newest it holds.
NEWEST = "cv_" + "5" * 32

_MEDIA_TYPES = {
    "csv": "text/csv",
    "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "parquet": "application/vnd.apache.parquet",
}


def _facts(body: dict[str, Any]) -> dict[str, Any]:
    """The 200 body the exports door answers for the prepare `body`."""
    file_format = body.get("format", "csv")
    bundle = body.get("bundle", False)
    return {
        "export_id": EXPORT_ID,
        "url_path": f"/api/datasets/exports/{EXPORT_ID}",
        "file_name": f"oda-ghana.{'zip' if bundle else file_format}",
        "media_type": "application/zip" if bundle else _MEDIA_TYPES[file_format],
        "format": file_format,
        "bundle": bundle,
        "bytes": 48213,
        "rows": 1240,
        "columns": 14,
        "sha256": "f" * 64,
        "frame": {"content_version": NEWEST, **body["frame"]},
        "citation": "OECD (2026), Creditor Reporting System.",
        "licences": [
            {
                "source": "dc/s/Oecd",
                "label": "OECD CRS",
                "licence": "OECD Terms and Conditions",
                "url": "https://www.oecd.org/terms",
            },
            {"source": "dc/s/WorldBank", "label": "World Bank WDI", "licence": None, "url": None},
        ],
        "pinned": True,
    }


def _door(requests: list[httpx2.Request]) -> Handler:
    """The exports door: the prepare answers its facts, and the link answers 302 to a
    signed URL, which the tool must never request."""

    def handler(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        if request.method == "GET" and request.url.path.startswith("/api/datasets/exports/"):
            return httpx2.Response(302, headers={"Location": SIGNED})
        assert (request.method, request.url.path) == ("POST", "/api/datasets/exports")
        return httpx2.Response(200, json=_facts(json.loads(request.content)))

    return handler


def _refusing(status: int, code: str, **extra: object) -> Handler:
    headers = extra.pop("headers", {})
    assert isinstance(headers, dict)

    def handler(request: httpx2.Request) -> httpx2.Response:
        # The probe a `not_found` sends to tell an unmounted surface from a missing id.
        if request.url.path == "/api/datasets/capabilities":
            return httpx2.Response(200, json={})
        return httpx2.Response(
            status,
            content=_problem_body(code=code, retryable=status >= 500, **extra),
            headers=headers,
        )

    return handler


async def _call(handler: Handler, arguments: dict[str, Any], **settings: object) -> CallToolResult:
    async with _client(handler, settings=_settings(dfo_site_origin=SITE, **settings)) as upstream:
        client_holder.set_client(upstream)
        async with Client(mcp) as client:
            return await client.call_tool("export_data", arguments, raise_on_error=False)


@pytest.mark.parametrize("bundle", [False, True], ids=["file", "bundle"])
@pytest.mark.parametrize("file_format", ["csv", "xlsx", "parquet"])
@pytest.mark.parametrize("frame", [FETCHED, DERIVED], ids=["fetched", "derived"])
async def test_an_export_answers_the_public_link_and_the_file_facts(
    frame: dict[str, str], file_format: str, bundle: bool
) -> None:
    requests: list[httpx2.Request] = []

    result = await _call(
        _door(requests), {"frame_ref": frame, "format": file_format, "bundle": bundle}
    )

    assert not result.is_error
    assert [json.loads(r.content) for r in requests] == [
        {"frame": frame, "format": file_format, "bundle": bundle}
    ]
    url = f"{SITE}/api/datasets/exports/{EXPORT_ID}"
    facts = _facts({"frame": frame, "format": file_format, "bundle": bundle})
    assert result.structured_content == {
        "url": url,
        "file_name": facts["file_name"],
        "media_type": facts["media_type"],
        "bytes": 48213,
        "rows": 1240,
        "columns": 14,
        "sha256": "f" * 64,
        "frame_ref": frame,
        "citation": facts["citation"],
        "licences": facts["licences"],
        "pinned": True,
    }
    (link,) = [block for block in result.content if isinstance(block, ResourceLink)]
    assert (str(link.uri), link.mime_type, link.size) == (url, facts["media_type"], 48213)
    digest = text_of(result)
    assert url in digest
    (columns,) = [line for line in digest.splitlines() if line.startswith("Columns:")]
    assert ("_label" in columns) == frame["spec_id"].startswith("spec_")


async def test_the_digest_states_the_file_facts_in_order() -> None:
    """File name, format, rows and columns, size, link, citation, each licence, how long the
    file is kept, then the columns it holds."""
    result = await _call(_door([]), {"frame_ref": DERIVED})

    lines = text_of(result).splitlines()
    labels = [line.split(":", 1)[0] for line in lines if not line.startswith("- ")]
    assert labels == [
        "File",
        "Format",
        "Rows and columns",
        "Size",
        "Link",
        "Citation",
        "Licences",
        "Kept",
        "Frame",
        "Columns",
    ]
    licence_lines = [line for line in lines if line.startswith("- ")]
    assert lines.index(licence_lines[0]) == labels.index("Licences") + 1
    assert len(licence_lines) == 2


async def test_no_signed_url_reaches_the_result() -> None:
    requests: list[httpx2.Request] = []

    result = await _call(_door(requests), {"frame_ref": FETCHED})

    assert [r.method for r in requests] == ["POST"]
    dumped = json.dumps(result.structured_content) + "".join(
        str(block.model_dump()) for block in result.content
    )
    assert "storage.googleapis.com" not in dumped
    assert "Signature" not in dumped


async def test_a_bare_spec_id_exports_the_version_the_platform_resolves() -> None:
    requests: list[httpx2.Request] = []

    result = await _call(_door(requests), {"frame_ref": {"spec_id": FETCHED["spec_id"]}})

    assert not result.is_error
    assert [json.loads(r.content)["frame"] for r in requests] == [{"spec_id": FETCHED["spec_id"]}]
    resolved = {"spec_id": FETCHED["spec_id"], "content_version": NEWEST}
    assert result.structured_content is not None
    assert result.structured_content["frame_ref"] == resolved
    assert f"Frame: {FETCHED['spec_id']} at content_version {NEWEST}" in text_of(result)


async def test_a_bare_derived_id_refuses_frame_ref_incomplete_without_a_call() -> None:
    requests: list[httpx2.Request] = []

    result = await _call(_door(requests), {"frame_ref": {"spec_id": DERIVED["spec_id"]}})

    assert result.is_error
    assert result.structured_content is not None
    assert result.structured_content["code"] == "frame_ref_incomplete"
    assert result.structured_content["retry_mode"] == "correct_and_retry"
    assert requests == []


@pytest.mark.parametrize(
    ("caller", "principal", "sent"),
    [
        (ADDRESS, PRINCIPAL, {CALLER_PRINCIPAL_HEADER: PRINCIPAL, CALLER_HEADER: ADDRESS}),
        (ADDRESS, None, {}),
        (conversation_caller("c1"), PRINCIPAL, {}),
    ],
    ids=["website-caller", "no-principal", "harness-caller"],
)
async def test_the_prepare_names_the_caller_the_website_named(
    caller: str, principal: str | None, sent: dict[str, str]
) -> None:
    requests: list[httpx2.Request] = []

    with supplied_caller(caller):
        await _call(_door(requests), {"frame_ref": FETCHED}, dfo_caller_principal=principal)

    (request,) = requests
    named = {
        name: request.headers[name]
        for name in (CALLER_PRINCIPAL_HEADER, CALLER_HEADER)
        if name in request.headers
    }
    assert named == sent


@pytest.mark.parametrize(
    ("handler", "code", "retry_mode", "fix", "retry_after_s"),
    [
        (
            _refusing(422, "over_ceiling", ceiling_kind="rows", count=300000, cap=250000),
            "over_ceiling",
            "correct_and_retry",
            {"format": "parquet"},
            None,
        ),
        (
            _refusing(429, "allowance_exhausted", headers={"Retry-After": "3600"}),
            "allowance_exhausted",
            "do_not_retry",
            {},
            3600.0,
        ),
        (
            _refusing(503, "service_unavailable", headers={"Retry-After": "7"}),
            "service_unavailable",
            "retry_after_delay",
            None,
            7.0,
        ),
        (
            _refusing(422, "provenance_unavailable"),
            "provenance_unavailable",
            "do_not_retry",
            {},
            None,
        ),
        (
            _refusing(403, "licence_not_redistributable"),
            "licence_not_redistributable",
            "do_not_retry",
            {},
            None,
        ),
    ],
    ids=[
        "over_ceiling",
        "allowance_exhausted",
        "service_unavailable",
        "provenance_unavailable",
        "licence_not_redistributable",
    ],
)
async def test_a_refused_prepare_maps_to_the_envelope(
    handler: Handler,
    code: str,
    retry_mode: str,
    fix: dict[str, Any] | None,
    retry_after_s: float | None,
) -> None:
    result = await _call(handler, {"frame_ref": FETCHED, "format": "csv"})

    assert result.is_error
    envelope = result.structured_content
    assert envelope is not None
    assert (envelope["code"], envelope["retry_mode"], envelope["retry_after_s"]) == (
        code,
        retry_mode,
        retry_after_s,
    )
    if fix is not None:
        assert envelope["fix"] == fix
    assert json.loads(text_of(result).rsplit("\n\n", 1)[1])["code"] == code


@pytest.mark.parametrize(
    ("status", "upstream_code"),
    [
        (404, "snapshot_not_found"),
        (410, "artifact_expired"),
        (404, "spec_not_found"),
        (404, "not_found"),
    ],
)
@pytest.mark.parametrize(
    ("frame", "fix"),
    [(FETCHED, {"next_step": "get_data"}), (DERIVED, {})],
    ids=["fetched", "derived"],
)
async def test_a_frame_the_platform_cannot_export_refuses_frame_unavailable(
    status: int, upstream_code: str, frame: dict[str, str], fix: dict[str, str]
) -> None:
    requests: list[httpx2.Request] = []
    refuse = _refusing(status, upstream_code)

    def handler(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        return refuse(request)

    result = await _call(handler, {"frame_ref": frame})

    envelope = result.structured_content
    assert envelope is not None
    assert (
        envelope["code"],
        envelope["retry_mode"],
        envelope["upstream_code"],
        envelope["fix"],
    ) == (
        "frame_unavailable",
        "refresh_then_retry",
        upstream_code,
        fix,
    )
    # No fallback to another version: the one prepare is the only export call.
    assert [r.method for r in requests if r.url.path == "/api/datasets/exports"] == ["POST"]
