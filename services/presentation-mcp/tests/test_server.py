"""Server-scaffold tests: startup wiring, `/healthz`, the safety net, and refusal passthrough.

The lifespan is what actually wires this server: it fetches the platform's
release over HTTP, admits it, and only then mounts the three tools. Every
test here drives that lifespan for real, via `fastmcp.Client(mcp)` -- its
in-memory transport enters the server's own lifespan before the client
session opens (`fastmcp.client.transports.memory._enter_server_lifespan`),
so mount/degrade behaviour is exercised exactly as it runs in production,
not simulated around it.

`server.PlatformAssembler` is the one seam substituted per test: either the
real class wired to an `httpx2.MockTransport`, or a small fake standing in
for an unreachable or misbehaving platform. Tool *behaviour* once mounted
(safety net, refusal passthrough) runs against a `StubAssembler` substituted
the same way, since it already implements the `Assembler` Protocol.
"""

from __future__ import annotations

import asyncio
import base64
import copy
import dataclasses
import inspect
import json
import logging
import os
import re
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator
from contextlib import asynccontextmanager, nullcontext
from html.parser import HTMLParser
from importlib.metadata import version
from pathlib import Path
from typing import Any, cast

import httpx2
import jsonschema
import pytest
import resvg_py
from common import refusal as refusal_module
from common.apis_auth import AuthProvider, IdTokenProvider
from common.caller import CALLER_HEADER, supplied_caller
from common.declaration import declared_schema
from common.materialisation import InMemoryMaterialisationStore, set_store
from common.materialisation import content_version as content_version_module
from common.mcp_telemetry import LOOPBACK_SCOPE_KEY
from common.refusal import REFUSAL_META_KEY, internal_error_message, refusal_text
from common.row_budget import RowBudgetExhausted
from fastmcp import Client
from fastmcp.apps import UI_EXTENSION_ID, UI_MIME_TYPE, ResourceCSP
from fastmcp.client.client import CallToolResult
from fastmcp.client.transports import FastMCPTransport, StreamableHttpTransport
from fastmcp.server.providers.addressing import hashed_backend_name
from fastmcp.tools import ToolResult
from mcp import ClientSession
from mcp import types as mcp_types
from mcp.shared._httpx_utils import McpHttpClientFactory
from mcp.types import (
    CLIENT_CAPABILITIES_META_KEY,
    PROTOCOL_VERSION_META_KEY,
    ImageContent,
    TextContent,
)
from presentation_mcp import client as client_module
from presentation_mcp import server
from presentation_mcp.app_view import app_data, app_shell, notice_view, view_state
from presentation_mcp.assembler_holder import get_assembler
from presentation_mcp.client import PlatformAssembler
from presentation_mcp.errors import PlatformError, PlatformTransportError
from presentation_mcp.platform_contract import (
    RESOLVED_PAYLOAD_META_KEY,
    VIEW_STATE_META_KEY,
    Assembler,
    Caller,
    Refusal,
    RetryMode,
    WriteReply,
)
from presentation_mcp.projection import (
    ROWLESS_NOTICE,
    UNKNOWN_HOST_ROWLESS_NOTICE,
    delta_projection,
    text_projection,
)
from presentation_mcp.server import ViewSupport, mcp
from presentation_mcp.settings import Settings
from presentation_mcp.theme import (
    THEME_CSS,
    THEME_HASH,
    renderer_csp,
    renderer_html,
    renderer_uri,
    theme_url,
)
from presentation_mcp.tool_schema import (
    describe_component_schema,
    describe_frame_schema,
    write_component_schema,
)

from .net_guard import no_network  # noqa: F401 -- imported for its autouse fixture
from .stub_assembler import StubAssembler, load_release, resolved_revision

RELEASE = load_release()
_BASE_URL = server.settings.presentation_public_url
_KNOWN_TRACE_ID = "0af7651916cd43dd8448eb211c80319c"
_RENDERER_URI = renderer_uri(_BASE_URL)

# The caller every in-memory client names, as the harness's `x-dfo-caller`
# header would.
CALLER = "conversation:u/c1"

Handler = Callable[[httpx2.Request], httpx2.Response]


def _release_only_transport(release: dict[str, Any]) -> httpx2.MockTransport:
    """A platform stand-in that serves `/release` and nothing else."""

    def handler(request: httpx2.Request) -> httpx2.Response:
        assert request.url.path == "/api/presentation/release"
        return httpx2.Response(200, json=release)

    return httpx2.MockTransport(handler)


def _mock_platform_assembler(release: dict[str, Any]) -> Any:
    """A factory dropped in for `server.PlatformAssembler`, backed by a `MockTransport`.

    Matches `PlatformAssembler(settings=settings, auth=auth)`'s call shape; the
    real class does the rest, so this exercises the real HTTP/retry/refusal path
    against a served release document rather than a hand-rolled fake.
    """

    def factory(*, settings: Settings, auth: AuthProvider | None = None) -> PlatformAssembler:
        return PlatformAssembler(
            settings=settings, auth=auth, transport=_release_only_transport(release)
        )

    return factory


def _stub_platform_assembler(stub: StubAssembler) -> Any:
    """A factory dropped in for `server.PlatformAssembler`, backed by a `StubAssembler`.

    `StubAssembler` already implements the `Assembler` Protocol -- including
    `release()` and `aclose()` -- so it is a drop-in substitute for the real
    HTTP client wherever a test wants mounted tools without touching a wire.
    """

    def factory(*, settings: Settings, auth: AuthProvider | None = None) -> StubAssembler:
        del settings, auth
        return stub

    return factory


def _write_token_line(result: CallToolResult | mcp_types.CallToolResult) -> str | None:
    """The token on the last line of a create's content, or `None` when the
    last line names none."""
    last = _all_text(result).splitlines()[-1]
    return last.removeprefix("write_token: ") if last.startswith("write_token: ") else None


def _text(result: CallToolResult) -> str:
    """The text of the model vocabulary's content block, as JSON.

    It is the last text block, except on a create, which ends with a block
    carrying the write token. A result carrying an envelope leads with the
    text projection, and one for a client that cannot draw the view may
    carry an image block after the vocabulary.
    """
    blocks = [block for block in result.content if isinstance(block, TextContent)]
    if _write_token_line(result):
        blocks = blocks[:-1]
    return blocks[-1].text


def _content_refusal(result: CallToolResult | mcp_types.CallToolResult) -> dict[str, Any]:
    """The refusal envelope a refusal's one text block states: its message,
    a blank line, then `code`, `retry_mode`, `fix` and `retry_after_s` as one
    line of compact JSON."""
    (block,) = [block for block in result.content if isinstance(block, TextContent)]
    message, _, line = block.text.rpartition("\n\n")
    return {"message": message, **json.loads(line)}


def _all_text(result: CallToolResult) -> str:
    return "\n".join(block.text for block in result.content if isinstance(block, TextContent))


async def _call_result(
    client: Client[Any], name: str, arguments: dict[str, Any], *, expect_error: bool = False
) -> CallToolResult:
    """Call `name` on an open `client` and return the whole result.

    `expect_error` must match the tool's `isError`. A refusal is only
    readable this way with `raise_on_error=False`, since fastmcp's `Client`
    otherwise raises a `ToolError` on an `isError` result. Asserts the
    parsed text and `structured_content` carry the identical payload, and
    that a write token in `structured_content` is the one on the content's
    last line.
    """
    result = await client.call_tool(name, arguments, raise_on_error=not expect_error)
    assert result.is_error is expect_error
    if expect_error:
        assert result.structured_content == _content_refusal(result)
        return result
    structured = dict(result.structured_content or {})
    assert structured.pop("write_token", None) == _write_token_line(result)
    assert json.loads(_text(result)) == structured
    return result


@asynccontextmanager
async def _client(*, caller: str | None = CALLER) -> AsyncIterator[Client[Any]]:
    """`Client(mcp)` naming `caller`, or no caller for `None`.

    The in-memory transport carries no headers, so `supplied_caller` stands
    in for `x-dfo-caller`, and the client opens inside its block.
    """
    with supplied_caller(caller) if caller is not None else nullcontext():
        async with Client(mcp) as client:
            yield client


async def _call(
    name: str, arguments: dict[str, Any], *, expect_error: bool = False
) -> dict[str, Any]:
    """Call `name` in a fresh client session and return its parsed JSON payload,
    or its refusal envelope.

    The parsed text and `structured_content` carry the identical payload
    (`_call_result` asserts it), so asserting the returned payload covers
    both.
    """
    async with _client() as client:
        result = await _call_result(client, name, arguments, expect_error=expect_error)
    return dict(result.structured_content or {}) if expect_error else json.loads(_text(result))


def _envelope(result: CallToolResult) -> Any:
    """The resolved envelope a result carries in `_meta`, or `None`."""
    return (result.meta or {}).get(RESOLVED_PAYLOAD_META_KEY)


def _assert_model_vocabulary_only(result: CallToolResult) -> None:
    """Neither model-facing part of `result` carries the envelope."""
    assert result.structured_content is not None
    assert "resolved_payload" not in result.structured_content
    assert "resolved_payload" not in _all_text(result)
    assert RESOLVED_PAYLOAD_META_KEY not in _all_text(result)


# --- mounting from a reachable, admissible release --------------------------


async def test_tools_mount_from_a_mock_transport_served_release(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(server, "PlatformAssembler", _mock_platform_assembler(RELEASE))

    async with Client(mcp) as client:
        tools = {tool.name: tool for tool in await client.list_tools()}

    assert set(tools) == {"write_component", "describe_frame", "describe_component"}
    assembled = {
        "write_component": write_component_schema(release=RELEASE),
        "describe_frame": describe_frame_schema(release=RELEASE),
        "describe_component": describe_component_schema(),
    }
    for name, schema in assembled.items():
        tool = tools[name]
        assert tool.input_schema == declared_schema(schema, description=tool.description or "")
    # A patch's null clears a field, so the declared schema keeps it.
    jsonschema.validate(
        {"intent": {"title": None}, "component_id": "cmp_existing", "write_token": "cmw_existing"},
        tools["write_component"].input_schema,
    )


def _schema_rule_breaks(description: str, schema: Any) -> list[str]:
    """Where a listed tool breaks the declaration rules: a `$comment`, an
    empty `required`, a null branch beside a `default` of null, a `$defs`
    or `$ref`, or a sentence of seven words or more stated twice across
    `description` and the schema's descriptions."""
    breaks: list[str] = []
    texts = [description]

    def walk(node: Any, path: str) -> None:
        if isinstance(node, list):
            for index, item in enumerate(node):
                walk(item, f"{path}/{index}")
            return
        if not isinstance(node, dict):
            return
        breaks.extend(f"{path}/{key}" for key in ("$comment", "$defs", "$ref") if key in node)
        if node.get("required") == []:
            breaks.append(f"{path}/required")
        required = set(node.get("required", ()))
        for name, prop in (node.get("properties") or {}).items():
            stated_null = "default" in prop and prop["default"] is None
            if name not in required and stated_null and {"type": "null"} in prop.get("anyOf", ()):
                breaks.append(f"{path}/properties/{name}/anyOf")
        for key, value in node.items():
            if key == "description" and isinstance(value, str):
                texts.append(value)
            else:
                walk(value, f"{path}/{key}")

    walk(schema, "")
    seen: set[str] = set()
    for text in texts:
        for sentence in re.split(r"(?<=[.!?])\s+", " ".join(text.split())):
            if len(sentence.split()) >= 7:
                if sentence in seen:
                    breaks.append(f"repeated: {sentence}")
                seen.add(sentence)
    return breaks


def _release_with_new_schema_notes() -> dict[str, Any]:
    """`RELEASE` with what a later release document could add in new places:
    a definition carrying a `$comment`, a property defaulting to null referencing
    it in two members, and a sentence the two share."""
    release = copy.deepcopy(RELEASE)
    style = {
        "$comment": "A developer note.",
        "description": "How the marks are drawn on every chart that offers it.",
        "enum": ["solid", "dashed"],
    }
    members = release["intent_schemas_tolerant"]
    for member in members.values():
        member["$defs"]["style"] = style
    for name in ("line_chart", "bar_chart"):
        members[name]["properties"]["style"] = {
            "anyOf": [{"$ref": "#/$defs/style"}, {"type": "null"}],
            "default": None,
        }
    return release


@pytest.mark.parametrize(
    "release",
    [RELEASE, _release_with_new_schema_notes()],
    ids=["fixture-release", "release-with-new-notes"],
)
async def test_every_listed_schema_keeps_the_declaration_rules(
    release: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(server, "PlatformAssembler", _mock_platform_assembler(release))

    async with Client(mcp) as client:
        tools = await client.list_tools()

    assert {tool.name for tool in tools} == {
        "write_component",
        "describe_frame",
        "describe_component",
    }
    for tool in tools:
        assert _schema_rule_breaks(tool.description or "", tool.input_schema) == [], tool.name


async def test_each_intent_members_frame_ref_takes_a_bare_spec_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(server, "PlatformAssembler", _mock_platform_assembler(RELEASE))

    async with Client(mcp) as client:
        tools = {tool.name: tool for tool in await client.list_tools()}

    members = tools["write_component"].input_schema["properties"]["intent"]["anyOf"]
    frame_refs = [m["properties"]["frame_ref"] for m in members if "frame_ref" in m["properties"]]
    assert len(frame_refs) == 5
    for frame_ref in frame_refs:
        assert frame_ref["required"] == ["spec_id"]
        assert set(frame_ref["properties"]) == {"spec_id", "content_version"}
    assert frame_refs[0]["description"].endswith(
        "Leave content_version out to pass a spec_id and have the server fetch it."
    )


async def test_healthz_reports_ok_once_mounted(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(server, "PlatformAssembler", _mock_platform_assembler(RELEASE))

    async with Client(mcp):
        assert server._health == {"status": "ok"}


async def test_on_cloud_run_the_assembler_sends_an_id_token_for_the_apis_origin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("K_SERVICE", "dfo-dev-ai")
    monkeypatch.setattr(
        server, "settings", Settings(dfo_apis_url="https://dfo-dev-apis-abc123.a.run.app/")
    )
    stub_factory = _stub_platform_assembler(StubAssembler())
    auths: list[AuthProvider | None] = []

    def factory(*, settings: Settings, auth: AuthProvider | None = None) -> StubAssembler:
        auths.append(auth)
        return stub_factory(settings=settings, auth=auth)

    monkeypatch.setattr(server, "PlatformAssembler", factory)

    async with Client(mcp):
        pass

    (auth,) = auths
    assert isinstance(auth, IdTokenProvider)
    assert auth.audience == "https://dfo-dev-apis-abc123.a.run.app"


# --- frame store install failure -----------------------------------------------


async def test_frame_store_install_failure_mounts_nothing_and_degrades_healthz(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`frame_store_verify_content_version` with no `analysis_mcp` in this
    process to register a deriver raises `ContentVersionDeriverUnregistered`
    from `install_store_if_unset` -- this runs ahead of the platform release
    fetch, so `PlatformAssembler` is never even substituted with a working
    factory here, and the failure still degrades exactly like every later
    admission check.
    """
    # Importing `analysis_mcp` registers a deriver for the whole process.
    monkeypatch.setattr(content_version_module, "_deriver", None)
    monkeypatch.setattr(
        server,
        "settings",
        Settings(frame_store_bucket="test-bucket", frame_store_verify_content_version=True),
    )

    async with Client(mcp) as client:
        tools = await client.list_tools()
        assert tools == []
        assert server._health["status"] == "degraded"
        assert "frame store" in server._health["reason"]


async def test_frame_store_install_failure_fails_startup_when_release_is_required(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Importing `analysis_mcp` registers a deriver for the whole process.
    monkeypatch.setattr(content_version_module, "_deriver", None)
    monkeypatch.setattr(
        server,
        "settings",
        Settings(
            frame_store_bucket="test-bucket",
            frame_store_verify_content_version=True,
            presentation_release_required=True,
        ),
    )

    with pytest.raises(RuntimeError, match="content_version deriver"):
        async with Client(mcp):
            pass


# --- admission bound ----------------------------------------------------------


async def test_a_newer_schema_version_mounts_nothing_and_degrades_healthz(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    newer = RELEASE | {"schema_version": server.settings.max_schema_version + 1}
    monkeypatch.setattr(server, "PlatformAssembler", _mock_platform_assembler(newer))

    async with Client(mcp) as client:
        tools = await client.list_tools()
        assert tools == []
        assert server._health["status"] == "degraded"
        assert "schema_version" in server._health["reason"]


async def test_newer_schema_version_fails_startup_when_release_is_required(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    newer = RELEASE | {"schema_version": server.settings.max_schema_version + 1}
    monkeypatch.setattr(server, "PlatformAssembler", _mock_platform_assembler(newer))
    monkeypatch.setattr(server, "settings", Settings(presentation_release_required=True))

    with pytest.raises(RuntimeError, match="schema_version"):
        async with Client(mcp):
            pass


# --- a malformed schema_version ------------------------------------------------


async def test_missing_schema_version_mounts_nothing_and_degrades_healthz(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    missing = {k: v for k, v in RELEASE.items() if k != "schema_version"}
    monkeypatch.setattr(server, "PlatformAssembler", _mock_platform_assembler(missing))

    async with Client(mcp) as client:
        tools = await client.list_tools()
        assert tools == []
        assert server._health["status"] == "degraded"
        assert "schema_version" in server._health["reason"]


async def test_missing_schema_version_fails_startup_when_release_is_required(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    missing = {k: v for k, v in RELEASE.items() if k != "schema_version"}
    monkeypatch.setattr(server, "PlatformAssembler", _mock_platform_assembler(missing))
    monkeypatch.setattr(server, "settings", Settings(presentation_release_required=True))

    with pytest.raises(RuntimeError, match="schema_version"):
        async with Client(mcp):
            pass


async def test_non_comparable_schema_version_mounts_nothing_and_degrades_healthz(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stringy = RELEASE | {"schema_version": "not-a-number"}
    monkeypatch.setattr(server, "PlatformAssembler", _mock_platform_assembler(stringy))

    async with Client(mcp) as client:
        tools = await client.list_tools()
        assert tools == []
        assert server._health["status"] == "degraded"
        assert "schema_version" in server._health["reason"]


async def test_non_comparable_schema_version_fails_startup_when_release_is_required(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stringy = RELEASE | {"schema_version": "not-a-number"}
    monkeypatch.setattr(server, "PlatformAssembler", _mock_platform_assembler(stringy))
    monkeypatch.setattr(server, "settings", Settings(presentation_release_required=True))

    with pytest.raises(RuntimeError, match="schema_version"):
        async with Client(mcp):
            pass


# --- a release document that fails schema assembly ----------------------------


def _release_with_corrupted_defs() -> dict[str, Any]:
    """`RELEASE`, with one member's `$defs` diverging from the rest.

    `tool_schema._shared_defs` asserts all six `intent_schemas_tolerant`
    members carry an identical `$defs` map; this breaks that invariant the
    same way a genuinely inconsistent release document would.
    """
    corrupted = copy.deepcopy(RELEASE)
    corrupted["intent_schemas_tolerant"]["table"]["$defs"] = {}
    return corrupted


def _release_with_a_ref(ref: str, defs: dict[str, Any]) -> dict[str, Any]:
    """`RELEASE` with `defs` added to every member's `$defs` and a line_chart
    property referring to `ref`, which `declared_schema` cannot inline."""
    release = copy.deepcopy(RELEASE)
    members = release["intent_schemas_tolerant"]
    for member in members.values():
        member["$defs"].update(defs)
    members["line_chart"]["properties"]["style"] = {"$ref": ref}
    return release


# Each release this deployment cannot mount, with the text its refusal names.
_UNMOUNTABLE_RELEASES = {
    "diverging-defs": (_release_with_corrupted_defs(), "differs"),
    "non-local-ref": (
        _release_with_a_ref("https://example.org/style", {}),
        "refers outside the schema",
    ),
    "recursive-ref": (
        _release_with_a_ref("#/$defs/style", {"style": {"items": {"$ref": "#/$defs/style"}}}),
        r"keeps \$defs",
    ),
}


@pytest.mark.parametrize("case", list(_UNMOUNTABLE_RELEASES))
async def test_an_unmountable_release_mounts_nothing_and_degrades_healthz(
    case: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    release, reason = _UNMOUNTABLE_RELEASES[case]
    monkeypatch.setattr(server, "PlatformAssembler", _mock_platform_assembler(release))

    async with Client(mcp) as client:
        tools = await client.list_tools()
        assert tools == []
        assert server._health["status"] == "degraded"
        assert re.search(reason, server._health["reason"])

    with pytest.raises(RuntimeError, match="no assembler is wired"):
        get_assembler()


@pytest.mark.parametrize("case", list(_UNMOUNTABLE_RELEASES))
async def test_an_unmountable_release_fails_startup_when_release_is_required(
    case: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    release, reason = _UNMOUNTABLE_RELEASES[case]
    monkeypatch.setattr(server, "PlatformAssembler", _mock_platform_assembler(release))
    monkeypatch.setattr(server, "settings", Settings(presentation_release_required=True))

    with pytest.raises(RuntimeError, match=reason):
        async with Client(mcp):
            pass

    with pytest.raises(RuntimeError, match="no assembler is wired"):
        get_assembler()


# --- an unreachable platform --------------------------------------------------

_TOOL_NAMES = {"write_component", "describe_frame", "describe_component"}


def _required(*, base_s: float = 0.01, max_s: float = 0.02, deadline_s: float = 5.0) -> Settings:
    """Deployed settings, `presentation_release_required` on, with retries
    shortened so a test runs in well under a second."""
    return Settings(
        presentation_release_required=True,
        release_retry_base_s=base_s,
        release_retry_max_s=max_s,
        release_startup_deadline_s=deadline_s,
    )


class _PlatformDownAtBoot(StubAssembler):
    """A stub platform whose first `failures` release fetches fail, then
    answer with `release`."""

    def __init__(self, *, failures: int, release: dict[str, Any] | None = None) -> None:
        super().__init__()
        self.failures = failures
        self.fetches = 0
        self._served = release

    async def release(self) -> Any:
        self.fetches += 1
        if self.fetches <= self.failures:
            raise ConnectionError("platform down")
        return self._served if self._served is not None else await super().release()


async def test_an_unreachable_platform_fails_startup_without_ever_yielding_when_required(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No request, `tools/list` included, reaches a required server that
    never mounted its tools: the lifespan raises before it yields."""
    platform = _PlatformDownAtBoot(failures=10**6)
    monkeypatch.setattr(server, "PlatformAssembler", _stub_platform_assembler(platform))
    monkeypatch.setattr(server, "settings", _required(deadline_s=0.1))

    with pytest.raises(
        RuntimeError, match=r"could not reach the platform within 0.1s \(fetches: \d+"
    ):
        async with server._lifespan(mcp):
            pytest.fail(f"the lifespan yielded with tools {await mcp.list_tools()}")

    assert platform.fetches > 1
    assert await mcp.list_tools(run_middleware=False) == []
    with pytest.raises(RuntimeError, match="no assembler is wired"):
        get_assembler()


async def test_a_platform_that_answers_on_the_third_attempt_mounts_every_tool_before_startup_ends(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    platform = _PlatformDownAtBoot(failures=2)
    monkeypatch.setattr(server, "PlatformAssembler", _stub_platform_assembler(platform))
    monkeypatch.setattr(server, "settings", _required())

    async with server._lifespan(mcp):
        assert platform.fetches == 3
        tools = await mcp.list_tools(run_middleware=False)
        assert {tool.name for tool in tools} == _TOOL_NAMES
        assert server._health == {"status": "ok"}

    assert await mcp.list_tools(run_middleware=False) == []


async def test_an_in_flight_fetch_is_cancelled_at_the_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _HangingPlatform(StubAssembler):
        async def release(self) -> Any:
            await asyncio.sleep(3600)

    monkeypatch.setattr(server, "PlatformAssembler", _stub_platform_assembler(_HangingPlatform()))
    monkeypatch.setattr(server, "settings", _required(deadline_s=0.05))

    with pytest.raises(RuntimeError, match="TimeoutError"):
        async with asyncio.timeout(5), server._lifespan(mcp):
            pytest.fail("the lifespan yielded")


async def test_a_release_that_answers_inadmissible_during_retry_fails_startup_at_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    platform = _PlatformDownAtBoot(failures=1, release={**RELEASE, "schema_version": 99})
    monkeypatch.setattr(server, "PlatformAssembler", _stub_platform_assembler(platform))
    monkeypatch.setattr(server, "settings", _required())

    with pytest.raises(RuntimeError, match="exceeds"):
        async with server._lifespan(mcp):
            pytest.fail("the lifespan yielded")

    assert platform.fetches == 2


async def test_the_retry_backoff_doubles_up_to_its_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    platform = _PlatformDownAtBoot(failures=5)
    monkeypatch.setattr(
        server,
        "settings",
        _required(base_s=1.0, max_s=4.0, deadline_s=60),
    )
    delays: list[float] = []

    async def sleep(seconds: float) -> None:
        delays.append(seconds)

    mounted: tuple[str, ...] = ()
    try:
        mounted = await server._mount_within_deadline(
            cast(PlatformAssembler, platform), sleep=sleep
        )
    finally:
        for name in mounted:
            mcp.local_provider.remove_tool(name)
        server.set_assembler(None)

    assert delays == [1.0, 2.0, 4.0, 4.0, 4.0]
    assert set(mounted) == _TOOL_NAMES


async def test_an_unreachable_platform_mounts_nothing_and_degrades_healthz_when_not_required(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Local development's opt-out: one fetch, then a tool-less server
    that stays tool-less."""
    platform = _PlatformDownAtBoot(failures=1)
    monkeypatch.setattr(server, "PlatformAssembler", _stub_platform_assembler(platform))
    monkeypatch.setattr(server, "settings", Settings(presentation_release_required=False))

    async with Client(mcp) as client:
        assert await client.list_tools() == []
        assert server._health["status"] == "degraded"
        assert "platform down" in server._health["reason"]
        await asyncio.sleep(0.05)
        assert await client.list_tools() == []

    assert platform.fetches == 1


# --- tool behaviour once mounted ----------------------------------------------


async def test_not_found_refusal_from_describe_component_carries_is_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(server, "PlatformAssembler", _stub_platform_assembler(StubAssembler()))

    reply = await _call("describe_component", {"component_id": "cmp_missing"}, expect_error=True)

    assert reply["code"] == "not_found"
    assert reply["retry_mode"] == "correct_and_retry"


async def test_schema_invalid_refusal_from_write_component_carries_is_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(server, "PlatformAssembler", _stub_platform_assembler(StubAssembler()))
    # `frame_ref` dropped from an otherwise-minimal stat_card intent.
    intent = {"type": "stat_card", "select": {"statvar": "health_oda"}}

    reply = await _call("write_component", {"intent": intent}, expect_error=True)

    assert reply["code"] == "schema_invalid"
    assert reply["retry_mode"] == "correct_and_retry"
    assert "frame_ref" in reply["message"]


async def test_write_component_round_trips_through_the_mounted_stub(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(server, "PlatformAssembler", _stub_platform_assembler(StubAssembler()))
    intent = RELEASE["minimal_intents"]["line_chart"]

    reply = await _call("write_component", {"intent": intent})

    assert reply["component_id"] == "cmp_1"
    assert reply["revision"] == 1


# --- the resolved envelope rides in _meta ------------------------------------


async def test_write_component_carries_the_envelope_in_meta_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(server, "PlatformAssembler", _stub_platform_assembler(StubAssembler()))
    intent = RELEASE["minimal_intents"]["line_chart"]

    async with _client() as client:
        created = await _call_result(client, "write_component", {"intent": intent})
        revised = await _call_result(
            client,
            "write_component",
            {"component_id": "cmp_1", "write_token": "cmw_1", "intent": {"title": "New"}},
        )

    for result, revision in ((created, 1), (revised, 2)):
        _assert_model_vocabulary_only(result)
        envelope = _envelope(result)
        assert envelope["component_id"] == "cmp_1"
        assert envelope["revision"] == revision
        assert envelope["provenance"]["inputs"] == [intent["frame_ref"]]
    assert _envelope(revised)["component"]["title"]["text"] == "New"


async def test_describe_component_carries_the_requested_revisions_envelope_in_meta_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(server, "PlatformAssembler", _stub_platform_assembler(StubAssembler()))
    intent = RELEASE["minimal_intents"]["line_chart"]

    async with _client() as client:
        await _call_result(client, "write_component", {"intent": intent})
        await _call_result(
            client,
            "write_component",
            {"component_id": "cmp_1", "write_token": "cmw_1", "intent": {"title": "New"}},
        )
        current = await _call_result(client, "describe_component", {"component_id": "cmp_1"})
        first = await _call_result(
            client, "describe_component", {"component_id": "cmp_1", "revision": 1}
        )

    _assert_model_vocabulary_only(current)
    _assert_model_vocabulary_only(first)
    assert current.structured_content is not None
    assert current.structured_content["intent"]["title"] == "New"
    assert _envelope(current)["revision"] == 2
    assert _envelope(current)["component"]["title"]["text"] == "New"
    assert _envelope(first)["revision"] == 1
    assert _envelope(first)["component"]["title"]["text"] != "New"


async def test_a_write_without_an_envelope_still_succeeds_and_logs_a_warning(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    class _EnvelopelessAssembler(StubAssembler):
        async def create_component(self, **kwargs: Any) -> Any:
            reply = await super().create_component(**kwargs)
            assert isinstance(reply, WriteReply)
            return dataclasses.replace(reply, resolved_payload=None)

    envelopeless: Assembler = _EnvelopelessAssembler()
    monkeypatch.setattr(server, "PlatformAssembler", _stub_platform_assembler(envelopeless))
    intent = RELEASE["minimal_intents"]["line_chart"]

    with caplog.at_level(logging.WARNING, logger=server.__name__):
        async with _client() as client:
            result = await _call_result(client, "write_component", {"intent": intent})

    assert result.structured_content == {
        "component_id": "cmp_1",
        "revision": 1,
        "fact_summary": {},
        "changed": [],
        "write_token": "cmw_1",
    }
    assert _envelope(result) is None
    assert len(result.content) == 2
    assert _write_token_line(result) == "cmw_1"
    assert any("carried no resolved_payload" in r.getMessage() for r in caplog.records)


async def test_refusals_and_frame_views_carry_no_envelope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(server, "PlatformAssembler", _stub_platform_assembler(StubAssembler()))

    async with _client() as client:
        refused = await _call_result(
            client, "describe_component", {"component_id": "cmp_missing"}, expect_error=True
        )
        frame = await _call_result(
            client,
            "describe_frame",
            {"frame_ref": {"spec_id": "spec_1", "content_version": "cv_1"}},
        )

    assert _envelope(refused) is None
    assert _envelope(frame) is None


async def test_describe_frame_round_trips_through_the_mounted_stub(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(server, "PlatformAssembler", _stub_platform_assembler(StubAssembler()))

    reply = await _call(
        "describe_frame", {"frame_ref": {"spec_id": "spec_1", "content_version": "cv_1"}}
    )

    assert reply["spec_id"] == "spec_7c2f91"
    assert "recipient" in reply["dimensions"]


async def test_safety_net_turns_exceptions_into_internal_error_refusals(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    class _BrokenAssembler(StubAssembler):
        async def get_frame_view(self, **_: Any) -> dict[str, Any]:
            raise RuntimeError("boom")

    broken: Assembler = _BrokenAssembler()
    monkeypatch.setattr(server, "PlatformAssembler", _stub_platform_assembler(broken))
    monkeypatch.setattr(refusal_module, "current_trace_id", lambda: _KNOWN_TRACE_ID)

    with caplog.at_level(logging.ERROR, logger=server.__name__):
        reply = await _call(
            "describe_frame",
            {"frame_ref": {"spec_id": "spec_1", "content_version": "cv_1"}},
            expect_error=True,
        )

    assert reply["code"] == "internal_error"
    assert reply["retry_mode"] == "do_not_retry"
    assert reply["message"] == internal_error_message(subject="This call")
    assert "boom" not in json.dumps(reply)
    assert _KNOWN_TRACE_ID not in json.dumps(reply)
    [record] = [r for r in caplog.records if r.name == server.__name__]
    assert _KNOWN_TRACE_ID in record.getMessage()


async def test_a_write_forwards_the_caller_type_and_no_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[dict[str, Any]] = []

    class _SpyAssembler(StubAssembler):
        async def create_component(self, **kwargs: Any) -> Any:
            seen.append(kwargs)
            return await super().create_component(**kwargs)

    spy: Assembler = _SpyAssembler()
    monkeypatch.setattr(server, "PlatformAssembler", _stub_platform_assembler(spy))
    intent = RELEASE["minimal_intents"]["line_chart"]

    await _call("write_component", {"intent": intent})

    assert len(seen) == 1
    assert set(seen[0]) == {"intent", "caller", "rebuild_lost_frames"}
    assert seen[0]["rebuild_lost_frames"] is False
    assert seen[0]["caller"] is Caller.EXTERNAL


async def test_a_component_written_with_no_caller_is_revised_and_described_by_its_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(server, "PlatformAssembler", _stub_platform_assembler(StubAssembler()))
    intent = RELEASE["minimal_intents"]["line_chart"]

    async with _client(caller=None) as client:
        created = await _call_result(client, "write_component", {"intent": intent})
        revised = await _call_result(
            client,
            "write_component",
            {"component_id": "cmp_1", "write_token": "cmw_1", "intent": {"title": "New"}},
        )
        described = await _call_result(client, "describe_component", {"component_id": "cmp_1"})
        frame = await _call_result(
            client,
            "describe_frame",
            {"frame_ref": {"spec_id": "spec_1", "content_version": "cv_1"}},
        )

    assert created.structured_content is not None
    assert created.structured_content["component_id"] == "cmp_1"
    assert revised.structured_content is not None
    assert revised.structured_content["revision"] == 2
    assert _envelope(described) == _envelope(revised)
    assert frame.structured_content is not None
    assert "dimensions" in frame.structured_content


async def test_another_caller_describes_a_component_by_its_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(server, "PlatformAssembler", _stub_platform_assembler(StubAssembler()))
    intent = RELEASE["minimal_intents"]["line_chart"]

    async with _client(caller=CALLER) as writer:
        await _call_result(writer, "write_component", {"intent": intent})
    async with _client(caller="conversation:u/c2") as reader:
        described = await _call_result(reader, "describe_component", {"component_id": "cmp_1"})

    assert described.structured_content is not None
    assert described.structured_content["component_id"] == "cmp_1"


# --- the write token ----------------------------------------------------------


async def test_a_revise_without_a_write_token_is_refused_before_the_platform(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    revises: list[dict[str, Any]] = []

    class _SpyAssembler(StubAssembler):
        async def revise_component(self, **kwargs: Any) -> Any:
            revises.append(kwargs)
            return await super().revise_component(**kwargs)

    spy: Assembler = _SpyAssembler()
    monkeypatch.setattr(server, "PlatformAssembler", _stub_platform_assembler(spy))
    intent = RELEASE["minimal_intents"]["line_chart"]

    async with _client() as client:
        await _call_result(client, "write_component", {"intent": intent})
        refused = await _call_result(
            client,
            "write_component",
            {"component_id": "cmp_1", "intent": {"title": "New"}},
            expect_error=True,
        )

    assert revises == []
    refusal = _content_refusal(refused)
    assert refusal["code"] == "request_invalid"
    assert refusal["fix"]["tool"] == "write_component"
    (problem,) = refusal["fix"]["problems"]
    assert problem.startswith("write_token: ")
    assert "cmp_1" in refusal["message"]
    assert "create a new component" in refusal["message"]
    assert "describe_component" in refusal["message"]


async def test_a_revise_with_the_create_token_succeeds_and_a_wrong_one_is_write_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(server, "PlatformAssembler", _stub_platform_assembler(StubAssembler()))
    intent = RELEASE["minimal_intents"]["line_chart"]

    async with _client() as client:
        created = await _call_result(client, "write_component", {"intent": intent})
        assert created.structured_content is not None
        token = created.structured_content["write_token"]
        revised = await _call_result(
            client,
            "write_component",
            {"component_id": "cmp_1", "write_token": token, "intent": {"title": "New"}},
        )
        refused = await _call_result(
            client,
            "write_component",
            {"component_id": "cmp_1", "write_token": "cmw_9", "intent": {"title": "Other"}},
            expect_error=True,
        )

    assert token == _write_token_line(created) == "cmw_1"
    assert revised.structured_content is not None
    assert revised.structured_content["revision"] == 2
    assert "write_token" not in revised.structured_content
    assert _write_token_line(revised) is None
    refusal = _content_refusal(refused)
    assert refusal["code"] == "write_refused"
    assert refusal["retry_mode"] == "do_not_retry"


async def _token_results_over_http(*, meta: dict[str, Any] | None) -> list[dict[str, Any]]:
    """A create, a revise, and a describe as raw stateless JSON-RPC results,
    each request carrying `meta` as its `_meta`."""
    app = mcp.http_app(transport="http", stateless_http=True, json_response=True)
    calls = [
        ("write_component", {"intent": RELEASE["minimal_intents"]["line_chart"]}),
        ("write_component", {"component_id": "cmp_1", "write_token": "cmw_1", "intent": {}}),
        ("describe_component", {"component_id": "cmp_1"}),
    ]
    headers = {"accept": "application/json, text/event-stream", CALLER_HEADER: CALLER}
    if meta is not None:
        headers["mcp-protocol-version"] = meta[PROTOCOL_VERSION_META_KEY]
        headers["mcp-method"] = "tools/call"
    results = []
    async with app.router.lifespan_context(app):
        transport = httpx2.ASGITransport(app=app)
        async with httpx2.AsyncClient(transport=transport, base_url="http://testserver") as http:
            for number, (name, arguments) in enumerate(calls):
                params: dict[str, Any] = {"name": name, "arguments": arguments}
                if meta is not None:
                    params["_meta"] = meta
                    headers["mcp-name"] = name
                call = {"jsonrpc": "2.0", "id": number, "method": "tools/call", "params": params}
                response = await http.post("/mcp", json=call, headers=headers)
                results.append(response.json()["result"])
    return results


def _declaring(extensions: dict[str, Any]) -> dict[str, Any]:
    return {
        PROTOCOL_VERSION_META_KEY: "2026-07-28",
        CLIENT_CAPABILITIES_META_KEY: {"extensions": extensions},
    }


@pytest.mark.parametrize(
    ("meta", "apps"),
    [
        (_declaring({UI_EXTENSION_ID: {"mimeTypes": [UI_MIME_TYPE]}}), True),
        (_declaring({}), False),
        (None, True),
    ],
    ids=["declared", "undeclared", "unknown"],
)
async def test_the_write_token_never_reaches_meta_the_app_shell_or_a_later_result(
    monkeypatch: pytest.MonkeyPatch, meta: dict[str, Any] | None, apps: bool
) -> None:
    monkeypatch.setattr(server, "PlatformAssembler", _stub_platform_assembler(StubAssembler()))
    monkeypatch.setattr(mcp, "auth", None)

    created, revised, described = await _token_results_over_http(meta=meta)

    assert created["content"][-1]["text"].splitlines()[-1] == "write_token: cmw_1"
    assert "cmw_" not in json.dumps(created["_meta"])
    if apps:
        assert "$prefab" in created["structuredContent"]
        assert "cmw_" not in json.dumps(created["structuredContent"])
    else:
        assert created["structuredContent"]["write_token"] == "cmw_1"
    for later in (revised, described):
        assert not later.get("isError")
        assert "cmw_" not in json.dumps(later)


# --- the MCP App --------------------------------------------------------------

_UI = {UI_EXTENSION_ID: {"mimeTypes": [UI_MIME_TYPE]}}


async def test_the_two_component_tools_list_as_prefab_apps_without_an_output_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(server, "PlatformAssembler", _stub_platform_assembler(StubAssembler()))

    async with Client(mcp) as client:
        tools = {tool.name: tool for tool in await client.list_tools()}

    for name in ("write_component", "describe_component"):
        assert tools[name].meta is not None
        assert tools[name].meta["ui"]["resourceUri"] == _RENDERER_URI
        assert tools[name].output_schema is None
    assert "ui" not in (tools["describe_frame"].meta or {})
    assert tools["describe_frame"].output_schema is None


async def test_the_renderer_is_prefabs_cdn_page_pinned_to_the_installed_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(server, "PlatformAssembler", _stub_platform_assembler(StubAssembler()))
    uri = _RENDERER_URI

    async with Client(mcp) as client:
        listed = {str(resource.uri): resource for resource in await client.list_resources()}
        (read,) = await client.read_resource(uri)

    assert uri.startswith("ui://presentation-mcp/")
    assert set(listed) == {uri}
    assert listed[uri].mime_type == UI_MIME_TYPE
    assert read.mime_type == UI_MIME_TYPE
    assert isinstance(read, mcp_types.TextResourceContents)
    assert read.text == renderer_html()
    assert read.text.lstrip().lower().startswith("<!doctype html>")
    base = f"https://cdn.jsdelivr.net/npm/@prefecthq/prefab-ui@{version('prefab-ui')}/dist/app"
    assert _asset_urls(read.text) == [f"{base}/renderer.css", f"{base}/renderer.js"]


async def test_an_old_renderer_uri_returns_what_the_listed_uri_returns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A host keeps the URI from when its user added the app, so any
    `renderer-{version}.html` answers with the current page, MIME type, and CSP."""
    monkeypatch.setattr(server, "PlatformAssembler", _stub_platform_assembler(StubAssembler()))
    old = "ui://presentation-mcp/renderer-00000000.html"
    assert old != _RENDERER_URI

    async with Client(mcp) as client:
        (current,) = await client.read_resource(_RENDERER_URI)
        (stale,) = await client.read_resource(old)

    assert isinstance(current, mcp_types.TextResourceContents)
    assert isinstance(stale, mcp_types.TextResourceContents)
    assert stale.text == current.text == renderer_html()
    assert stale.mime_type == current.mime_type == UI_MIME_TYPE
    assert stale.meta == current.meta
    assert stale.meta is not None
    declared = ResourceCSP.model_validate(renderer_csp(_BASE_URL)).resource_domains
    assert stale.meta["ui"]["csp"] == {"resourceDomains": declared}


async def test_the_renderer_page_carries_the_view_state_script(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(server, "PlatformAssembler", _stub_platform_assembler(StubAssembler()))

    async with Client(mcp) as client:
        (read,) = await client.read_resource(_RENDERER_URI)

    assert isinstance(read, mcp_types.TextResourceContents)
    assert json.dumps(VIEW_STATE_META_KEY) in read.text
    assert read.text.index("ui/notifications/tool-result") < read.text.index("</head>")


async def test_the_renderer_declares_the_csp_built_from_the_configured_public_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The listing and the read, where MCP Apps hosts look, both carry
    `renderer_csp` of `settings.presentation_public_url`. What that CSP holds is
    pinned in test_theme.py."""
    monkeypatch.setattr(server, "PlatformAssembler", _stub_platform_assembler(StubAssembler()))

    async with Client(mcp) as client:
        (listed,) = await client.list_resources()
        (read,) = await client.read_resource(_RENDERER_URI)

    declared = ResourceCSP.model_validate(renderer_csp(_BASE_URL)).resource_domains or []
    for meta in (listed.meta, read.meta):
        assert meta is not None
        assert meta["ui"]["csp"]["resourceDomains"] == declared


def _asset_urls(page: str) -> list[str]:
    """The URLs `page`'s stylesheet links and script tags load, in page order.
    A string inside an inline script is script text, which `HTMLParser` leaves
    unparsed."""
    urls: list[str] = []

    class _Tags(HTMLParser):
        def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
            attributes = dict(attrs)
            if tag == "script" and attributes.get("src"):
                urls.append(attributes["src"])
            if tag == "link" and attributes.get("rel") == "stylesheet" and attributes.get("href"):
                urls.append(attributes["href"])

    _Tags().feed(page)
    return urls


async def test_a_plain_client_gets_the_projection_then_the_model_vocabulary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(server, "PlatformAssembler", _stub_platform_assembler(StubAssembler()))
    intent = RELEASE["minimal_intents"]["line_chart"]

    async with _client() as client:
        created = await _call_result(client, "write_component", {"intent": intent})
        described = await _call_result(client, "describe_component", {"component_id": "cmp_1"})

    for result in (created, described):
        projection, vocabulary = result.content[:2]
        assert isinstance(projection, TextContent)
        assert projection.text == text_projection(_envelope(result))
        assert isinstance(vocabulary, TextContent)
        assert json.loads(vocabulary.text) == {
            k: v for k, v in (result.structured_content or {}).items() if k != "write_token"
        }
    assert created.structured_content == {
        "component_id": "cmp_1",
        "revision": 1,
        "fact_summary": {},
        "changed": [],
        "write_token": "cmw_1",
    }
    assert [type(block) for block in created.content] == [
        TextContent,
        TextContent,
        ImageContent,
        TextContent,
    ]
    assert _write_token_line(created) == "cmw_1"
    assert [type(block) for block in described.content] == [TextContent, TextContent, ImageContent]
    assert "write_token" not in _all_text(described)


async def test_a_revise_that_leaves_the_visible_data_alone_leads_with_a_delta_line(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(server, "PlatformAssembler", _stub_platform_assembler(StubAssembler()))
    intent = RELEASE["minimal_intents"]["line_chart"]

    async with _client() as client:
        await _call_result(client, "write_component", {"intent": intent})
        revised = await _call_result(
            client,
            "write_component",
            {"component_id": "cmp_1", "write_token": "cmw_1", "intent": {"title": "New"}},
        )

    delta, vocabulary, image = revised.content
    assert isinstance(image, ImageContent)
    assert isinstance(delta, TextContent)
    assert delta.text == "title changed, values unchanged (`cmp_1`, revision 2)."
    assert delta.text == delta_projection(_envelope(revised), [{"field": "title"}])
    assert isinstance(vocabulary, TextContent)
    assert json.loads(vocabulary.text) == revised.structured_content


@pytest.mark.parametrize(
    ("patch", "include_projection"),
    [
        ({"window": {"start": "2016", "end": "2020"}}, None),
        ({"select": {"statvar": "health_oda"}}, None),
        ({"title": "New"}, "full"),
    ],
)
async def test_a_revise_that_changes_the_visible_data_or_asks_for_full_gets_the_projection(
    monkeypatch: pytest.MonkeyPatch, patch: dict[str, Any], include_projection: str | None
) -> None:
    monkeypatch.setattr(server, "PlatformAssembler", _stub_platform_assembler(StubAssembler()))
    intent = RELEASE["minimal_intents"]["line_chart"]
    arguments: dict[str, Any] = {"component_id": "cmp_1", "write_token": "cmw_1", "intent": patch}
    if include_projection is not None:
        arguments["include_projection"] = include_projection

    async with _client() as client:
        await _call_result(client, "write_component", {"intent": intent})
        revised = await _call_result(client, "write_component", arguments)

    projection, *_ = revised.content
    assert isinstance(projection, TextContent)
    assert projection.text == text_projection(_envelope(revised))


@pytest.mark.parametrize(
    ("member", "changed"),
    [
        ("line_chart", [{"field": "title", "prior": "Old", "new": "New"}]),
        ("line_chart", [{"field": "scale", "prior": "linear", "new": "log"}]),
        ("bar_chart", [{"field": "range"}, {"field": "title", "prior": None, "new": "New"}]),
        ("line_chart", []),
    ],
)
def test_a_revise_of_presentation_only_fields_leads_with_the_delta(
    member: str, changed: list[dict[str, Any]]
) -> None:
    envelope = resolved_revision(member)

    lead = server._projection(
        envelope, changed, include_projection="auto", revise=True, views=ViewSupport.UNDECLARED
    )

    assert lead == delta_projection(envelope, changed)


@pytest.mark.parametrize(
    ("member", "changed"),
    [
        ("stat_card", [{"field": "change"}]),
        ("note_narration", [{"field": "text", "prior": "Old words.", "new": "New words."}]),
        ("line_chart", [{"field": "select"}]),
        ("line_chart", [{"field": "window"}]),
        ("line_chart", [{"field": "title", "prior": "Old", "new": "New"}, {"field": "window"}]),
        ("line_chart", [{"field": "a_field_a_later_catalogue_adds"}]),
        ("table", [{"field": "scale"}]),
        ("note_narration", [{"field": "title"}]),
    ],
)
def test_a_revise_of_any_other_field_leads_with_the_full_projection(
    member: str, changed: list[dict[str, Any]]
) -> None:
    envelope = resolved_revision(member)

    lead = server._projection(
        envelope, changed, include_projection="auto", revise=True, views=ViewSupport.UNDECLARED
    )

    assert lead == text_projection(envelope)


@pytest.mark.parametrize("member", ["line_chart", "bar_chart", "table"])
@pytest.mark.parametrize(("revise", "changed"), [(False, []), (True, [{"field": "select"}])])
@pytest.mark.parametrize(
    ("views", "notice"),
    [
        (ViewSupport.DECLARED, ROWLESS_NOTICE),
        (ViewSupport.UNKNOWN, UNKNOWN_HOST_ROWLESS_NOTICE),
    ],
)
def test_an_apps_client_gets_the_projection_without_its_data_table_under_auto(
    member: str, revise: bool, changed: list[dict[str, Any]], views: ViewSupport, notice: str
) -> None:
    envelope = resolved_revision(member)

    lead = server._projection(
        envelope, changed, include_projection="auto", revise=revise, views=views
    )

    assert lead == text_projection(envelope, rowless_notice=notice)
    assert lead != text_projection(envelope)


def test_the_unknown_client_notice_claims_nothing_about_drawing() -> None:
    """A text client whose capabilities the server cannot see must not be
    told the component is on screen."""
    assert "since this client draws" not in UNKNOWN_HOST_ROWLESS_NOTICE
    assert "cannot tell whether this client draws" in UNKNOWN_HOST_ROWLESS_NOTICE
    assert (
        "`describe_component` with `include_projection` set to `full`"
        in UNKNOWN_HOST_ROWLESS_NOTICE
    )


def test_an_apps_client_still_gets_the_delta_for_a_presentation_only_revise() -> None:
    envelope = resolved_revision("line_chart")
    changed = [{"field": "title"}]

    lead = server._projection(
        envelope, changed, include_projection="auto", revise=True, views=ViewSupport.DECLARED
    )

    assert lead == delta_projection(envelope, changed)


@pytest.mark.parametrize("views", list(ViewSupport))
@pytest.mark.parametrize("revise", [False, True])
def test_include_projection_full_sends_the_data_table_to_any_client(
    views: ViewSupport, revise: bool
) -> None:
    envelope = resolved_revision("line_chart")

    lead = server._projection(
        envelope, [{"field": "title"}], include_projection="full", revise=revise, views=views
    )

    assert lead == text_projection(envelope)


@pytest.mark.parametrize("views", list(ViewSupport))
def test_include_projection_none_suppresses_the_projection_for_any_client(
    views: ViewSupport,
) -> None:
    lead = server._projection(
        resolved_revision("line_chart"), [], include_projection="none", revise=False, views=views
    )

    assert lead is None


@pytest.mark.parametrize("changed", [[], [{"field": "title", "prior": "Old", "new": "New"}]])
def test_a_revise_of_an_unknown_member_type_leads_with_the_full_projection(
    changed: list[dict[str, Any]],
) -> None:
    envelope = resolved_revision("line_chart")
    envelope["component"]["type"] = "scatter_chart"

    lead = server._projection(
        envelope, changed, include_projection="auto", revise=True, views=ViewSupport.UNDECLARED
    )

    assert lead == text_projection(envelope)


async def test_include_projection_none_leaves_the_model_vocabulary_alone_in_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(server, "PlatformAssembler", _stub_platform_assembler(StubAssembler()))
    intent = RELEASE["minimal_intents"]["line_chart"]

    async with _client() as client:
        created = await _call_result(
            client, "write_component", {"intent": intent, "include_projection": "none"}
        )
        revised = await _call_result(
            client,
            "write_component",
            {
                "component_id": "cmp_1",
                "write_token": "cmw_1",
                "intent": {"title": "New"},
                "include_projection": "none",
            },
        )

    for result in (created, revised):
        vocabulary = result.content[0]
        assert isinstance(vocabulary, TextContent)
        assert json.loads(vocabulary.text) == {
            k: v for k, v in (result.structured_content or {}).items() if k != "write_token"
        }
        assert _envelope(result) is not None
    assert len(created.content) == 2
    assert _write_token_line(created) == "cmw_1"
    assert len(revised.content) == 1


_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def _images(result: CallToolResult | ToolResult) -> list[ImageContent]:
    return [block for block in result.content if isinstance(block, ImageContent)]


@pytest.fixture
async def chart_worker() -> AsyncIterator[None]:
    """The server's chart worker, running and warm, for a test that calls
    `server._component_reply` outside the lifespan. The warm-up finishes
    before the test starts, so no render's timeout counts a spawn."""
    server._chart_worker.start()
    if (warming := server._chart_worker._warming) is not None:
        await warming
    yield
    await server._chart_worker.stop()


async def _reply_for(
    envelope: dict[str, Any],
    *,
    views: ViewSupport,
    include_projection: server.IncludeProjection = "auto",
) -> ToolResult:
    """`server._component_reply` for a create that resolved to `envelope`."""
    reply = {"component_id": "cmp_1", "revision": 1, "resolved_payload": envelope}
    return await server._component_reply(
        reply, tool="write_component", views=views, include_projection=include_projection
    )


async def test_an_undeclared_client_gets_one_png_of_the_chart_before_the_write_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(server, "PlatformAssembler", _stub_platform_assembler(StubAssembler()))
    intent = RELEASE["minimal_intents"]["line_chart"]

    async with _client() as client:
        created = await _call_result(client, "write_component", {"intent": intent})
        described = await _call_result(client, "describe_component", {"component_id": "cmp_1"})

    for result in (created, described):
        (image,) = _images(result)
        assert image.mime_type == "image/png"
        assert base64.b64decode(image.data, validate=True).startswith(_PNG_SIGNATURE)
        assert result.content.index(image) == 2
    assert _write_token_line(created) == "cmw_1"
    assert created.content[-1] is not _images(created)[0]


@pytest.mark.parametrize("views", [ViewSupport.DECLARED, ViewSupport.UNKNOWN])
@pytest.mark.usefixtures("chart_worker")
async def test_a_client_that_can_draw_or_may_draw_the_view_gets_no_image(
    views: ViewSupport,
) -> None:
    result = await _reply_for(resolved_revision("line_chart"), views=views)

    assert _images(result) == []


async def test_a_declaring_host_gets_no_image_over_the_wire(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(server, "PlatformAssembler", _stub_platform_assembler(StubAssembler()))
    ui = {UI_EXTENSION_ID: {"mimeTypes": [UI_MIME_TYPE]}}

    async with _session(extensions=ui) as session:
        results = await _write_and_describe(session)

    for result in results:
        assert not [b for b in result.content if isinstance(b, mcp_types.ImageContent)]


@pytest.mark.parametrize("member", ["line_chart", "bar_chart"])
@pytest.mark.usefixtures("chart_worker")
async def test_include_projection_none_gets_no_image(member: str) -> None:
    envelope = resolved_revision(member)

    without = await _reply_for(envelope, views=ViewSupport.UNDECLARED, include_projection="none")
    auto = await _reply_for(envelope, views=ViewSupport.UNDECLARED)
    full = await _reply_for(envelope, views=ViewSupport.UNDECLARED, include_projection="full")

    assert _images(without) == []
    assert len(_images(auto)) == len(_images(full)) == 1


@pytest.mark.parametrize(
    ("member", "axis"), [("line_chart", "y_axis"), ("bar_chart", "value_axis")]
)
@pytest.mark.usefixtures("chart_worker")
async def test_a_log_axis_chart_gets_no_image(member: str, axis: str) -> None:
    envelope = resolved_revision(member)
    envelope["component"][axis]["scale"] = "log"

    result = await _reply_for(envelope, views=ViewSupport.UNDECLARED)

    assert _images(result) == []


@pytest.mark.parametrize("member", ["table", "stat_card", "note_narration", "source_card"])
@pytest.mark.usefixtures("chart_worker")
async def test_a_member_that_is_not_a_chart_gets_no_image(member: str) -> None:
    result = await _reply_for(resolved_revision(member), views=ViewSupport.UNDECLARED)

    assert _images(result) == []


@pytest.fixture
def worker_calls(monkeypatch: pytest.MonkeyPatch) -> list[object]:
    """The component id of each render asked of the server's chart worker,
    which draws nothing."""
    calls: list[object] = []

    async def _render(_: Any, *, component_id: object) -> None:
        calls.append(component_id)

    monkeypatch.setattr(server._chart_worker, "render", _render)
    return calls


@pytest.mark.parametrize("member", ["line_chart", "bar_chart"])
async def test_a_chart_for_an_undeclared_client_goes_to_the_worker(
    member: str, worker_calls: list[object]
) -> None:
    await _reply_for(resolved_revision(member), views=ViewSupport.UNDECLARED)

    assert worker_calls == ["cmp_1"]


@pytest.mark.parametrize("member", ["table", "stat_card", "note_narration", "source_card"])
async def test_a_member_that_is_not_a_chart_never_reaches_the_worker(
    member: str, worker_calls: list[object]
) -> None:
    await _reply_for(resolved_revision(member), views=ViewSupport.UNDECLARED)

    assert worker_calls == []


@pytest.mark.parametrize(
    ("member", "axis"), [("line_chart", "y_axis"), ("bar_chart", "value_axis")]
)
async def test_a_log_axis_chart_never_reaches_the_worker(
    member: str, axis: str, worker_calls: list[object]
) -> None:
    envelope = resolved_revision(member)
    envelope["component"][axis]["scale"] = "log"

    await _reply_for(envelope, views=ViewSupport.UNDECLARED)

    assert worker_calls == []


async def test_a_render_that_raises_still_succeeds_without_an_image_and_logs_an_error(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    async def _raise(_: Any, *, component_id: object) -> bytes:
        raise RuntimeError("resvg failed")

    monkeypatch.setattr(server, "PlatformAssembler", _stub_platform_assembler(StubAssembler()))
    monkeypatch.setattr(server._chart_worker, "render", _raise)
    intent = RELEASE["minimal_intents"]["line_chart"]

    with caplog.at_level(logging.ERROR, logger=server.__name__):
        async with _client() as client:
            created = await _call_result(client, "write_component", {"intent": intent})

    assert not created.is_error
    assert _images(created) == []
    assert _write_token_line(created) == "cmw_1"
    assert _envelope(created) is not None
    (record,) = [r for r in caplog.records if r.name == server.__name__]
    assert record.levelno == logging.ERROR
    assert record.exc_info is not None
    assert "cmp_1" in record.getMessage()


async def test_the_chart_is_drawn_in_the_worker_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """resvg fails in this process, so the image can only come from the worker."""

    def _refuse(**_: Any) -> bytes:
        raise AssertionError("the chart was rasterised on the server's own process")

    monkeypatch.setattr(resvg_py, "svg_to_bytes", _refuse)
    monkeypatch.setattr(server, "PlatformAssembler", _stub_platform_assembler(StubAssembler()))
    intent = RELEASE["minimal_intents"]["bar_chart"]

    async with _client() as client:
        created = await _call_result(client, "write_component", {"intent": intent})

    (image,) = _images(created)
    assert base64.b64decode(image.data).startswith(_PNG_SIGNATURE)


async def test_the_lifespan_starts_the_chart_worker_and_stops_it() -> None:
    async with server._lifespan(mcp):
        await server._chart_worker.render(resolved_revision("line_chart"), component_id="c")
        process = server._chart_worker._process
        assert process is not None
        pid = process.pid
        assert pid is not None
        assert pid != os.getpid()

    assert server._chart_worker._process is None
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


@pytest.mark.parametrize("failing", ["text_projection", "app_shell", "view_state"])
async def test_a_render_failure_after_the_write_still_succeeds_with_the_envelope_in_meta(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, failing: str
) -> None:
    """The platform has stored the write, so a render failure must not read
    as a refusal the model would retry into a duplicate."""

    def _raise(*_: Any, **__: Any) -> Any:
        raise KeyError("series")

    monkeypatch.setattr(server, "PlatformAssembler", _stub_platform_assembler(StubAssembler()))
    monkeypatch.setattr(server, failing, _raise)
    ui = {UI_EXTENSION_ID: {"mimeTypes": [UI_MIME_TYPE]}}

    with caplog.at_level(logging.ERROR, logger=server.__name__):
        async with _session(extensions=ui) as session:
            created, described = await _write_and_describe(session)

    for result in (created, described):
        assert not result.is_error
        assert result.meta is not None
        assert result.meta[RESOLVED_PAYLOAD_META_KEY]["component_id"] == "cmp_1"
        vocabulary = result.content[0]
        assert isinstance(vocabulary, TextContent)
        assert json.loads(vocabulary.text)["component_id"] == "cmp_1"
    assert len(created.content) == 2
    assert _write_token_line(created) == "cmw_1"
    assert len(described.content) == 1
    records = [r for r in caplog.records if r.name == server.__name__]
    assert len(records) == 2
    for result, record in zip((created, described), records, strict=True):
        assert re.search(r"trace_id=[0-9a-f]{32}", record.getMessage())
        assert "resolved revision of cmp_1" in record.getMessage()
        assert record.exc_info is not None
        assert result.structured_content == notice_view(
            "This revision of the component could not be drawn.", base_url=_BASE_URL
        )


@asynccontextmanager
async def _session(*, extensions: dict[str, Any] | None) -> AsyncIterator[ClientSession]:
    """An mcp `ClientSession` on `mcp` naming `CALLER`, initialised declaring `extensions`.

    An MCP Apps host names `io.modelcontextprotocol/ui` among its client
    capabilities' `extensions`, which `ClientSession` declares on
    `initialize`.
    """
    with supplied_caller(CALLER):
        async with FastMCPTransport(mcp).connect_session(
            extensions=extensions,
            client_info=mcp_types.Implementation(name="apps-host", version="1"),
        ) as session:
            await session.initialize()
            yield session


async def _write_and_describe(session: ClientSession) -> list[mcp_types.CallToolResult]:
    intent = RELEASE["minimal_intents"]["line_chart"]
    created = await session.call_tool("write_component", {"intent": intent})
    described = await session.call_tool("describe_component", {"component_id": "cmp_1"})
    return [created, described]


async def test_a_client_declaring_mcp_apps_gets_the_prefab_shell(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(server, "PlatformAssembler", _stub_platform_assembler(StubAssembler()))

    async with _session(extensions=_UI) as session:
        results = await _write_and_describe(session)

    for result in results:
        assert not result.is_error
        assert result.meta is not None
        envelope = result.meta[RESOLVED_PAYLOAD_META_KEY]
        assert result.structured_content == app_shell(envelope, base_url=_BASE_URL)
        projection, vocabulary = result.content[:2]
        assert isinstance(projection, TextContent)
        assert projection.text == text_projection(envelope, rowless_notice=ROWLESS_NOTICE)
        assert ROWLESS_NOTICE in projection.text
        assert "412.3" not in projection.text
        assert isinstance(vocabulary, TextContent)
        assert json.loads(vocabulary.text)["component_id"] == "cmp_1"
        assert "resolved_payload" not in vocabulary.text
        assert RESOLVED_PAYLOAD_META_KEY not in json.dumps(result.structured_content)


async def test_an_apps_client_gets_the_data_table_from_describe_component_under_full(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(server, "PlatformAssembler", _stub_platform_assembler(StubAssembler()))

    async with _session(extensions=_UI) as session:
        await _write_and_describe(session)
        described = await session.call_tool(
            "describe_component", {"component_id": "cmp_1", "include_projection": "full"}
        )

    assert not described.is_error
    assert described.meta is not None
    projection, _ = described.content
    assert isinstance(projection, TextContent)
    assert projection.text == text_projection(described.meta[RESOLVED_PAYLOAD_META_KEY])
    assert "| 2015 | 412.3 |" in projection.text


async def test_an_apps_client_gets_a_notice_view_for_a_refusal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(server, "PlatformAssembler", _stub_platform_assembler(StubAssembler()))

    ui = {UI_EXTENSION_ID: {"mimeTypes": [UI_MIME_TYPE]}}
    async with _session(extensions=ui) as session:
        refused = await session.call_tool("describe_component", {"component_id": "cmp_missing"})

    assert refused.is_error
    refusal = _content_refusal(refused)
    assert refusal["code"] == "not_found"
    assert refused.structured_content == notice_view(
        f"The request was refused: {refusal['message']}", base_url=_BASE_URL
    )


def _write_refusing_platform_assembler() -> Any:
    """A factory for `server.PlatformAssembler` whose platform serves the
    release and answers every revise with the apis service's write_refused body,
    whose detail names the write-token header and REST routes."""
    problem = {
        "type": "/codes/write_refused",
        "title": "Write Refused",
        "status": 403,
        "detail": (
            "the x-dfo-write-token sent does not grant writing component 'cmp_1' fix: "
            "create a new component with POST /api/presentation/components; read this one "
            "first with GET /api/presentation/components/cmp_1 to reuse its intent"
        ),
        "code": "write_refused",
        "retryable": False,
        "trace_id": "trace-1",
    }

    def handler(request: httpx2.Request) -> httpx2.Response:
        if request.url.path == "/api/presentation/release":
            return httpx2.Response(200, json=RELEASE)
        assert request.method == "PATCH"
        return httpx2.Response(403, json=problem)

    def factory(*, settings: Settings, auth: AuthProvider | None = None) -> PlatformAssembler:
        return PlatformAssembler(
            settings=settings, auth=auth, transport=httpx2.MockTransport(handler)
        )

    return factory


@pytest.mark.parametrize("apps_host", [True, False], ids=["apps-host", "plain-host"])
async def test_a_write_refused_reply_names_tools_and_no_platform_route_or_header(
    monkeypatch: pytest.MonkeyPatch, apps_host: bool
) -> None:
    """The refusal's content, its structuredContent and an Apps host's
    notice speak of write_component's arguments, never the apis service's REST
    routes or its write-token header."""
    monkeypatch.setattr(server, "PlatformAssembler", _write_refusing_platform_assembler())

    ui = {UI_EXTENSION_ID: {"mimeTypes": [UI_MIME_TYPE]}}
    async with _session(extensions=ui if apps_host else None) as session:
        refused = await session.call_tool(
            "write_component",
            {"component_id": "cmp_1", "write_token": "cmw_wrong", "intent": {"title": "New"}},
        )

    assert refused.is_error
    (block,) = refused.content
    assert isinstance(block, TextContent)
    refusal = _content_refusal(refused)
    assert refusal["code"] == "write_refused"
    assert "write_token" in refusal["message"]
    assert "write_component" in refusal["message"]
    assert "describe_component" in refusal["message"]
    assert "Create a new component" in refusal["message"]
    expected = (
        notice_view(f"The request was refused: {refusal['message']}", base_url=_BASE_URL)
        if apps_host
        else refusal
    )
    assert refused.structured_content == expected
    for text in (block.text, json.dumps(refused.structured_content)):
        assert "/api/" not in text
        assert "x-dfo-" not in text


@pytest.mark.parametrize(
    "extensions",
    [{UI_EXTENSION_ID: {"mimeTypes": [UI_MIME_TYPE]}}, None],
    ids=["apps-host", "plain-host"],
)
async def test_a_refusal_carries_its_code_in_meta_with_or_without_a_view(
    extensions: dict[str, Any] | None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The harness's combined server reads a refusal's code from `_meta`, since
    an Apps client's `structuredContent` holds a notice view. A reply that
    succeeds carries none."""
    monkeypatch.setattr(server, "PlatformAssembler", _stub_platform_assembler(StubAssembler()))

    async with _session(extensions=extensions) as session:
        refused = await session.call_tool("describe_component", {"component_id": "cmp_missing"})
        frame = await session.call_tool(
            "describe_frame", {"frame_ref": {"spec_id": "spec_1", "content_version": "cv_1"}}
        )

    assert refused.is_error
    assert ("code" in (refused.structured_content or {})) is (extensions is None)
    assert (refused.meta or {})[REFUSAL_META_KEY]["code"] == "not_found"
    assert not frame.is_error
    assert REFUSAL_META_KEY not in (frame.meta or {})


class _RefusingAssembler(StubAssembler):
    """A `StubAssembler` answering every tool call with one `Refusal`, or
    raising for none, which the safety net turns into `internal_error`."""

    def __init__(self, refusal: Refusal | None) -> None:
        super().__init__()
        self._refusal = refusal

    def _answer(self) -> Refusal:
        if self._refusal is None:
            raise RuntimeError("boom")
        return self._refusal

    async def create_component(self, **_: Any) -> Refusal:
        return self._answer()

    async def revise_component(self, **_: Any) -> Refusal:
        return self._answer()

    async def get_frame_view(self, **_: Any) -> Refusal:
        return self._answer()

    async def get_component(self, **_: Any) -> Refusal:
        return self._answer()


_FRAME_REF = {"spec_id": "spec_1", "content_version": "cv_1"}

# One call per refusal path in `server.py`: a create and a revise through
# `_component_reply`, describe_component through it too, and describe_frame
# through `_reply`.
_REFUSED_CALLS = {
    "create": ("write_component", {"intent": RELEASE["minimal_intents"]["line_chart"]}),
    "revise": (
        "write_component",
        {"component_id": "cmp_1", "write_token": "cmw_1", "intent": {"title": "New"}},
    ),
    "describe_component": ("describe_component", {"component_id": "cmp_1"}),
    "describe_frame": ("describe_frame", {"frame_ref": _FRAME_REF}),
}

# Each shape a `Refusal` takes, with the envelope fields it answers with.
_REFUSAL_SHAPES = {
    "no-fix": (
        Refusal(error="not_found", retry=RetryMode.CORRECT_AND_RETRY, detail="Not held."),
        {"code": "not_found", "retry_mode": "correct_and_retry", "message": "Not held.", "fix": {}},
    ),
    "text-fix-and-pointer": (
        Refusal(
            error="binding_invalid",
            retry=RetryMode.CORRECT_AND_RETRY,
            detail="No such recipient.",
            fix="recipient: Kenya (country/KEN)",
            pointer="/select/recipient",
        ),
        {
            "code": "binding_invalid",
            "retry_mode": "correct_and_retry",
            "message": "No such recipient. Recipient: Kenya (country/KEN).",
            "fix": {},
        },
    ),
    "mapping-fix-and-wait": (
        Refusal(
            error="service_unavailable",
            retry=RetryMode.RETRY_AFTER_DELAY,
            detail="Busy.",
            fix={"rows_requested": 5},
            retry_after_s=2.0,
        ),
        {
            "code": "service_unavailable",
            "retry_mode": "retry_after_delay",
            "message": "Busy.",
            "fix": {"rows_requested": 5},
            "retry_after_s": 2.0,
        },
    ),
}


def _assert_refusal_envelope(result: CallToolResult) -> dict[str, Any]:
    """Assert `result` is an error whose `structuredContent` is the refusal
    envelope, whose one text block is `refusal_text` of it, and whose `_meta`
    names its code. Return the envelope."""
    assert result.is_error
    envelope = dict(result.structured_content or {})
    assert set(envelope) - {"retry_after_s"} == {"code", "retry_mode", "message", "fix"}
    assert isinstance(envelope["fix"], dict)
    assert envelope["retry_mode"] in {mode.value for mode in RetryMode}
    (block,) = result.content
    assert isinstance(block, TextContent)
    assert block.text == refusal_text(
        message=envelope["message"],
        code=envelope["code"],
        retry_mode=envelope["retry_mode"],
        fix=envelope["fix"],
        retry_after_s=envelope.get("retry_after_s"),
    )
    assert (result.meta or {})[REFUSAL_META_KEY]["code"] == envelope["code"]
    return envelope


@pytest.mark.parametrize("shape", list(_REFUSAL_SHAPES))
@pytest.mark.parametrize("call", list(_REFUSED_CALLS))
async def test_every_refusal_path_answers_the_envelope_in_both_channels(
    call: str, shape: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    refusal, expected = _REFUSAL_SHAPES[shape]
    monkeypatch.setattr(
        server, "PlatformAssembler", _stub_platform_assembler(_RefusingAssembler(refusal))
    )
    name, arguments = _REFUSED_CALLS[call]

    async with _client() as client:
        result = await client.call_tool(name, arguments, raise_on_error=False)

    assert _assert_refusal_envelope(result) == expected


@pytest.mark.parametrize(
    ("name", "arguments", "code"),
    [
        ("describe_frame", {"frame_ref": _FRAME_REF}, "internal_error"),
        ("write_component", {"component_id": "cmp_1", "intent": {}}, "request_invalid"),
    ],
    ids=["safety-net", "missing-write-token"],
)
async def test_the_servers_own_refusals_answer_the_envelope_in_both_channels(
    name: str, arguments: dict[str, Any], code: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        server, "PlatformAssembler", _stub_platform_assembler(_RefusingAssembler(None))
    )

    async with _client() as client:
        result = await client.call_tool(name, arguments, raise_on_error=False)

    assert _assert_refusal_envelope(result)["code"] == code


@pytest.mark.parametrize(
    ("name", "arguments"),
    [
        ("describe_frame", {"frame_ref": {"spec_id": "ops_1"}}),
        (
            "write_component",
            {"intent": {"type": "table", "frame_ref": {"spec_id": "ops_1"}, "select": {}}},
        ),
    ],
    ids=["describe_frame", "write_component"],
)
async def test_a_bare_ops_id_refuses_frame_ref_incomplete_in_the_envelope(
    name: str, arguments: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(server, "PlatformAssembler", _mock_platform_assembler(RELEASE))

    async with _client() as client:
        result = await client.call_tool(name, arguments, raise_on_error=False)

    envelope = _assert_refusal_envelope(result)
    assert (envelope["code"], envelope["retry_mode"]) == (
        "frame_ref_incomplete",
        "correct_and_retry",
    )
    assert envelope["fix"] == {}
    # The message opens with the argument's name as the schema writes it.
    assert envelope["message"].startswith("frame_ref names ops_1 ")


_SPEC_ID = "spec_" + "0123456789abcdef" * 2
_LOST_VERSION = "cv_" + "d" * 32
_NEWEST_VIEW = {"spec_id": _SPEC_ID, "content_version": "cv_9", "dimensions": {}}
_LOST = {"code": "frame_unavailable", "retryable": False, "detail": "gone"}


def _frame_platform_assembler(
    frame_requests: list[httpx2.Request], *frame_replies: httpx2.Response
) -> Any:
    """A factory for `server.PlatformAssembler` whose platform serves the
    release and answers each other request, recorded in `frame_requests`, with
    the next of `frame_replies`."""
    replies = iter(frame_replies)

    def handler(request: httpx2.Request) -> httpx2.Response:
        if request.url.path == "/api/presentation/release":
            return httpx2.Response(200, json=RELEASE)
        frame_requests.append(request)
        return next(replies)

    def factory(*, settings: Settings, auth: AuthProvider | None = None) -> PlatformAssembler:
        return PlatformAssembler(
            settings=settings, auth=auth, transport=httpx2.MockTransport(handler)
        )

    return factory


@pytest.fixture
def frame_store() -> Iterator[None]:
    """An in-process frame store, so a `spec_` pair's mint record lookup finds none."""
    set_store(InMemoryMaterialisationStore())
    try:
        yield
    finally:
        set_store(None)


@pytest.mark.usefixtures("frame_store")
@pytest.mark.parametrize(
    ("arguments", "frame_replies", "named"),
    [
        pytest.param(
            {"frame_ref": {"spec_id": _SPEC_ID}},
            (httpx2.Response(200, json=_NEWEST_VIEW),),
            ("cv_9",),
            id="bare spec_id",
        ),
        pytest.param(
            {
                "frame_ref": {"spec_id": _SPEC_ID, "content_version": _LOST_VERSION},
                "rebuild_lost_frames": True,
            },
            (httpx2.Response(404, json=_LOST), httpx2.Response(200, json=_NEWEST_VIEW)),
            ("cv_9", _LOST_VERSION),
            id="pair apis no longer holds",
        ),
    ],
)
async def test_describe_frame_names_the_version_it_read(
    arguments: dict[str, Any],
    frame_replies: tuple[httpx2.Response, ...],
    named: tuple[str, ...],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(server, "PlatformAssembler", _frame_platform_assembler([], *frame_replies))

    async with _client() as client:
        result = await _call_result(client, "describe_frame", arguments)

    text = _text(result)
    assert json.loads(text)["content_version"] == "cv_9"
    for named_version in named:
        assert named_version in text


def _lost_pair_call(name: str, frame_ref: dict[str, str]) -> dict[str, Any]:
    if name == "describe_frame":
        return {"frame_ref": frame_ref}
    return {"intent": {"type": "table", "frame_ref": frame_ref, "select": {}}}


@pytest.mark.usefixtures("frame_store")
@pytest.mark.parametrize("name", ["describe_frame", "write_component"])
@pytest.mark.parametrize(
    "frame_ref",
    [
        pytest.param({"spec_id": _SPEC_ID, "content_version": _LOST_VERSION}, id="spec_"),
        pytest.param(
            {"spec_id": "ops_" + "e" * 32, "content_version": "ocv_" + "e" * 32}, id="ops_"
        ),
    ],
)
async def test_the_mount_refuses_a_lost_pair_frame_unavailable(
    name: str, frame_ref: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The harness's call sets no `rebuild_lost_frames`, so a lost pair reaches
    it as `frame_unavailable` after one request, and its own replay answers."""
    frame_requests: list[httpx2.Request] = []
    monkeypatch.setattr(
        server,
        "PlatformAssembler",
        _frame_platform_assembler(frame_requests, httpx2.Response(404, json=_LOST)),
    )

    async with _client() as client:
        result = await client.call_tool(
            name, _lost_pair_call(name, frame_ref), raise_on_error=False
        )

    assert _assert_refusal_envelope(result)["code"] == "frame_unavailable"
    assert len(frame_requests) == 1


async def test_an_apps_client_gets_a_notice_view_for_a_reply_without_an_envelope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _EnvelopelessAssembler(StubAssembler):
        async def create_component(self, **kwargs: Any) -> Any:
            reply = await super().create_component(**kwargs)
            assert isinstance(reply, WriteReply)
            return dataclasses.replace(reply, resolved_payload=None)

    envelopeless: Assembler = _EnvelopelessAssembler()
    monkeypatch.setattr(server, "PlatformAssembler", _stub_platform_assembler(envelopeless))
    intent = RELEASE["minimal_intents"]["line_chart"]

    ui = {UI_EXTENSION_ID: {"mimeTypes": [UI_MIME_TYPE]}}
    async with _session(extensions=ui) as session:
        created = await session.call_tool("write_component", {"intent": intent})

    assert not created.is_error
    block, _ = created.content
    assert isinstance(block, TextContent)
    assert json.loads(block.text)["component_id"] == "cmp_1"
    assert _write_token_line(created) == "cmw_1"
    assert created.structured_content == notice_view(
        "The platform returned no revision of this component to draw.", base_url=_BASE_URL
    )


async def test_an_apps_result_carries_the_view_data_in_meta_and_none_in_structured_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(server, "PlatformAssembler", _stub_platform_assembler(StubAssembler()))

    async with _session(extensions=_UI) as session:
        results = await _write_and_describe(session)

    for result in results:
        assert result.meta is not None
        envelope = result.meta[RESOLVED_PAYLOAD_META_KEY]
        state = result.meta[VIEW_STATE_META_KEY]
        assert state == {"data": app_data(envelope), "loaded": True}
        assert state["data"]["rows"][0] == {"period": "2015", "s0": 412.3}
        structured = json.dumps(result.structured_content)
        assert "412.3" not in structured
        assert VIEW_STATE_META_KEY not in structured
        assert result.structured_content is not None
        assert result.structured_content["state"]["data"] == {"rows": []}
        assert result.structured_content["state"]["loaded"] is False


async def test_a_view_needs_no_tool_call_after_the_result_and_no_shared_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A host that opens a new MCP session per request, as ChatGPT does, gets
    the whole view from the write's result alone."""
    monkeypatch.setattr(server, "PlatformAssembler", _stub_platform_assembler(StubAssembler()))
    intent = RELEASE["minimal_intents"]["line_chart"]

    async with _session(extensions=_UI) as session:
        created = await session.call_tool("write_component", {"intent": intent})

    assert not created.is_error
    assert created.meta is not None
    assert created.structured_content is not None
    assert "toolCall" not in json.dumps(created.structured_content)
    envelope = created.meta[RESOLVED_PAYLOAD_META_KEY]
    assert created.meta[VIEW_STATE_META_KEY] == view_state(envelope)
    assert created.meta[VIEW_STATE_META_KEY]["data"]["rows"]


@pytest.mark.parametrize("extensions", [None, _UI])
async def test_render_component_is_absent_from_tools_list_and_from_a_hashed_call(
    monkeypatch: pytest.MonkeyPatch, extensions: dict[str, Any] | None
) -> None:
    monkeypatch.setattr(server, "PlatformAssembler", _stub_platform_assembler(StubAssembler()))
    hashed = hashed_backend_name("presentation", "render_component")

    async with _session(extensions=extensions) as session:
        await _write_and_describe(session)
        listed = await session.list_tools()
        refused = [
            await session.call_tool(name, {"component_id": "cmp_1", "revision": 1})
            for name in ("render_component", hashed)
        ]

    assert {tool.name for tool in listed.tools} == {
        "write_component",
        "describe_frame",
        "describe_component",
    }
    for result in refused:
        assert result.is_error
        assert result.structured_content is None
        (block,) = result.content
        assert isinstance(block, TextContent)
        assert "Unknown tool" in block.text


@pytest.mark.parametrize("extensions", [None, {"io.example/other": {}}])
async def test_a_client_not_declaring_mcp_apps_gets_the_model_vocabulary(
    monkeypatch: pytest.MonkeyPatch, extensions: dict[str, Any] | None
) -> None:
    monkeypatch.setattr(server, "PlatformAssembler", _stub_platform_assembler(StubAssembler()))

    async with _session(extensions=extensions) as session:
        results = await _write_and_describe(session)

    for result in results:
        assert result.meta is not None
        assert RESOLVED_PAYLOAD_META_KEY in result.meta
        vocabulary = result.content[1]
        assert isinstance(vocabulary, TextContent)
        assert result.structured_content is not None
        assert {
            k: v for k, v in result.structured_content.items() if k != "write_token"
        } == json.loads(vocabulary.text)
        assert "$prefab" not in result.structured_content


async def test_a_stateless_request_gets_the_apps_shape_and_the_unknown_client_notice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stateless HTTP request on the 2025-11-25 revision carries no
    `initialize`, so the server cannot see what the client declared. It gets
    the Prefab shell and the view state, as a declaring host does, and a
    projection whose notice claims nothing about drawing the component."""
    monkeypatch.setattr(server, "PlatformAssembler", _stub_platform_assembler(StubAssembler()))
    monkeypatch.setattr(mcp, "auth", None)
    app = mcp.http_app(transport="http", stateless_http=True, json_response=True)
    call = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {
            "name": "write_component",
            "arguments": {"intent": RELEASE["minimal_intents"]["line_chart"]},
        },
    }

    async with app.router.lifespan_context(app):
        transport = httpx2.ASGITransport(app=app)
        async with httpx2.AsyncClient(transport=transport, base_url="http://testserver") as http:
            response = await http.post(
                "/mcp",
                json=call,
                headers={"accept": "application/json, text/event-stream", CALLER_HEADER: CALLER},
            )

    result = response.json()["result"]
    envelope = result["_meta"][RESOLVED_PAYLOAD_META_KEY]
    assert result["structuredContent"] == app_shell(envelope, base_url=_BASE_URL)
    assert result["_meta"][VIEW_STATE_META_KEY] == view_state(envelope)
    projection = result["content"][0]["text"]
    assert projection == text_projection(envelope, rowless_notice=UNKNOWN_HOST_ROWLESS_NOTICE)
    assert ROWLESS_NOTICE not in projection
    assert result["content"][-1]["text"].splitlines()[-1] == "write_token: cmw_1"


@pytest.mark.parametrize(
    ("extensions", "apps"),
    [({UI_EXTENSION_ID: {"mimeTypes": [UI_MIME_TYPE]}}, True), ({}, False)],
)
async def test_a_2026_request_is_answered_by_the_capabilities_it_declares_without_client_info(
    monkeypatch: pytest.MonkeyPatch, extensions: dict[str, Any], apps: bool
) -> None:
    """A 2026-07-28 request declares its capabilities in `_meta`, and client
    info is optional there. The declaration alone decides the vocabulary."""
    monkeypatch.setattr(server, "PlatformAssembler", _stub_platform_assembler(StubAssembler()))
    monkeypatch.setattr(mcp, "auth", None)
    app = mcp.http_app(transport="http", stateless_http=True, json_response=True)
    call = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {
            "name": "write_component",
            "arguments": {"intent": RELEASE["minimal_intents"]["line_chart"]},
            "_meta": {
                PROTOCOL_VERSION_META_KEY: "2026-07-28",
                CLIENT_CAPABILITIES_META_KEY: {"extensions": extensions},
            },
        },
    }
    headers = {
        "accept": "application/json, text/event-stream",
        "mcp-protocol-version": "2026-07-28",
        "mcp-method": "tools/call",
        "mcp-name": "write_component",
    }

    async with app.router.lifespan_context(app):
        transport = httpx2.ASGITransport(app=app)
        async with httpx2.AsyncClient(transport=transport, base_url="http://testserver") as http:
            response = await http.post("/mcp", json=call, headers=headers)

    result = response.json()["result"]
    envelope = result["_meta"][RESOLVED_PAYLOAD_META_KEY]
    if apps:
        assert result["structuredContent"] == app_shell(envelope, base_url=_BASE_URL)
        assert result["content"][0]["text"] == text_projection(
            envelope, rowless_notice=ROWLESS_NOTICE
        )
    else:
        structured = result["structuredContent"]
        assert "$prefab" not in structured
        assert structured.pop("write_token") == "cmw_1"
        assert structured == json.loads(result["content"][1]["text"])
        assert result["content"][0]["text"] == text_projection(envelope)


@asynccontextmanager
async def _http_client(app: Any, *, caller: str | None) -> AsyncIterator[Client[Any]]:
    """A client reaching `app` over streamable HTTP in process, naming `caller`
    in its `x-dfo-caller` header, or no caller for `None`."""

    def factory(**kwargs: Any) -> httpx2.AsyncClient:
        return httpx2.AsyncClient(
            transport=httpx2.ASGITransport(app=app), base_url="http://testserver", **kwargs
        )

    transport = StreamableHttpTransport(
        "http://testserver/mcp",
        headers={CALLER_HEADER: caller} if caller is not None else {},
        httpx_client_factory=cast(McpHttpClientFactory, factory),
    )
    async with Client(transport) as client:
        yield client


async def test_a_component_id_alone_revises_and_describes_on_the_sessionless_revision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """On the 2026-07-28 revision every request is its own connection, and
    FastMCP's `Context.session_id` is new on each call. The id alone ties the
    revise and the describe to the write, with no caller header on any request."""
    monkeypatch.setattr(server, "PlatformAssembler", _stub_platform_assembler(StubAssembler()))
    monkeypatch.setattr(mcp, "auth", None)
    app = mcp.http_app(path="/mcp")
    intent = RELEASE["minimal_intents"]["line_chart"]

    async with app.router.lifespan_context(app), _http_client(app, caller=None) as client:
        await client.call_tool("write_component", {"intent": intent})
        revised = await client.call_tool(
            "write_component",
            {"component_id": "cmp_1", "write_token": "cmw_1", "intent": {"title": "New"}},
        )
        described = await client.call_tool("describe_component", {"component_id": "cmp_1"})
        revision = client.protocol_version

    assert revision == "2026-07-28"
    assert revised.structured_content is not None
    assert revised.structured_content["revision"] == 2
    assert _envelope(described) == _envelope(revised)
    assert _envelope(described)["revision"] == 2


@pytest.mark.parametrize(
    ("loopback_mark", "expected"),
    [
        pytest.param(True, Caller.FIRST_PARTY, id="harness-loopback-mount"),
        pytest.param(None, Caller.EXTERNAL, id="combined-or-standalone"),
        pytest.param("yes", Caller.EXTERNAL, id="mark-not-true"),
    ],
)
async def test_a_write_reads_its_caller_from_the_loopback_mark(
    monkeypatch: pytest.MonkeyPatch, loopback_mark: object, expected: Caller
) -> None:
    """Only a request the harness's loopback guard marked writes as first
    party, so only Studio's cards go out under the publish principal."""
    seen: list[Caller] = []

    class _SpyAssembler(StubAssembler):
        async def create_component(self, **kwargs: Any) -> Any:
            seen.append(kwargs["caller"])
            return await super().create_component(**kwargs)

        async def revise_component(self, **kwargs: Any) -> Any:
            seen.append(kwargs["caller"])
            return await super().revise_component(**kwargs)

    monkeypatch.setattr(server, "PlatformAssembler", _stub_platform_assembler(_SpyAssembler()))
    monkeypatch.setattr(mcp, "auth", None)
    inner = mcp.http_app(path="/mcp")

    async def marked(scope: Any, receive: Any, send: Any) -> None:
        if loopback_mark is not None:
            scope = {**scope, LOOPBACK_SCOPE_KEY: loopback_mark}
        await inner(scope, receive, send)

    intent = RELEASE["minimal_intents"]["line_chart"]
    async with inner.router.lifespan_context(inner), _http_client(marked, caller=None) as client:
        await client.call_tool("write_component", {"intent": intent})
        await client.call_tool(
            "write_component",
            {"component_id": "cmp_1", "write_token": "cmw_1", "intent": {"title": "New"}},
        )

    assert seen == [expected, expected]


# --- healthz custom route ------------------------------------------------------


async def test_healthz_custom_route_serves_the_current_health_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pins the route's mechanics -- it serves `server._health` verbatim --
    independent of the lifespan, which is what actually sets that state (see
    the mount/degrade tests above). `mcp.http_app`'s ASGI lifespan never runs
    under a plain `httpx2.ASGITransport`, so driving it here would test
    nothing this module doesn't already cover elsewhere.
    """
    monkeypatch.setattr(server, "_health", {"status": "ok"})
    app = mcp.http_app(transport="http")
    transport = httpx2.ASGITransport(app=app)

    async with httpx2.AsyncClient(transport=transport, base_url="http://testserver") as http_client:
        response = await http_client.get("/healthz")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


# --- theme custom route --------------------------------------------------------


async def _get_theme(name: str) -> httpx2.Response:
    app = mcp.http_app(transport="http")
    transport = httpx2.ASGITransport(app=app)
    async with httpx2.AsyncClient(transport=transport, base_url="http://testserver") as http_client:
        return await http_client.get(name)


async def test_the_theme_route_serves_the_current_css_immutably_and_without_a_token() -> None:
    """No `Authorization` header: the view's iframe cannot send the bearer."""
    response = await _get_theme(f"/theme-{THEME_HASH}.css")

    assert response.status_code == 200
    assert response.text == THEME_CSS
    assert response.headers["content-type"] == "text/css; charset=utf-8"
    assert response.headers["access-control-allow-origin"] == "*"
    assert response.headers["cache-control"] == "public, max-age=31536000, immutable"


async def test_the_theme_route_serves_the_current_css_for_an_old_hash_without_caching() -> None:
    response = await _get_theme("/theme-0123456789ab.css")

    assert response.status_code == 200
    assert response.text == THEME_CSS
    assert response.headers["content-type"] == "text/css; charset=utf-8"
    assert response.headers["access-control-allow-origin"] == "*"
    assert response.headers["cache-control"] == "no-cache"


async def test_the_theme_url_a_payload_names_resolves_to_the_route() -> None:
    path = theme_url(_BASE_URL).removeprefix(_BASE_URL)

    response = await _get_theme(path)

    assert response.headers["cache-control"] == "public, max-age=31536000, immutable"


async def _platform_refusal(error: PlatformError) -> Refusal:
    """The refusal `PlatformAssembler` relays for the platform error `error`."""
    async with PlatformAssembler(
        settings=server.settings, transport=_release_only_transport(RELEASE)
    ) as assembler:
        return assembler._refusal(error)


async def _safety_net_refusal() -> Refusal:
    async def boom() -> None:
        raise RuntimeError("boom")

    return await server._safely(boom())


def _relayed(code: str) -> Callable[[], Awaitable[Refusal]]:
    """A platform refusal of `code` naming a field and its legal values."""
    error = PlatformError(
        status=400,
        code=code,
        retryable=False,
        detail="the platform refused it",
        pointer="/intent/select/recipient",
        alternatives=[{"label": "Kenya", "ref": "country/KEN"}],
    )
    return lambda: _platform_refusal(error)


# Every refusal presentation-mcp writes itself, by the tool that answers it, and
# one relayed from the platform for each code the release's retry table names.
_REFUSAL_SAMPLES: dict[str, tuple[str, Callable[[], Refusal | Awaitable[Refusal]]]] = {
    "component-id": ("describe_component", lambda: client_module._component_id_refusal("x y")),
    "write-token": ("write_component", lambda: client_module._write_token_refusal("x")),
    "missing-write-token": ("write_component", lambda: server._missing_write_token("cmp_1")),
    "spec-id": ("describe_frame", lambda: client_module._frame_ref_or_refusal({"spec_id": "x"})),
    "content-version": (
        "describe_frame",
        lambda: client_module._frame_ref_or_refusal({"spec_id": "spec_1", "content_version": "x"}),
    ),
    "frame-ref-incomplete": (
        "describe_frame",
        lambda: client_module._frame_ref_incomplete("ops_1"),
    ),
    "publish-body": (
        "write_component",
        lambda: client_module._publish_body_refusal(
            body_bytes=100, rows=10, ceiling=50, ceiling_kind="publish_body_bytes", what="gzip"
        ),
    ),
    "publish-not-configured": ("write_component", lambda: client_module._PUBLISH_NOT_CONFIGURED),
    "row-budget": (
        "write_component",
        lambda: client_module._row_budget_refusal(
            RowBudgetExhausted(requested=10, free=1, max_rows=5)
        ),
    ),
    "safety-net": ("describe_frame", _safety_net_refusal),
    "transport": (
        "describe_frame",
        lambda: _platform_refusal(PlatformTransportError(detail="connection reset")),
    ),
    "unparsed-4xx": (
        "describe_frame",
        lambda: _platform_refusal(PlatformError(status=400, code=None, retryable=False)),
    ),
    "unparsed-5xx": (
        "describe_frame",
        lambda: _platform_refusal(PlatformError(status=502, code=None, retryable=True)),
    ),
    **{f"relayed-{code}": ("write_component", _relayed(code)) for code in RELEASE["retry_modes"]},
}


async def _sample(name: str) -> tuple[str, Refusal]:
    tool, build = _REFUSAL_SAMPLES[name]
    built = build()
    refusal = await built if inspect.isawaitable(built) else built
    assert isinstance(refusal, Refusal), name
    return tool, refusal


async def test_every_presentation_code_has_a_refusal_sample() -> None:
    """The samples cover every code this service writes itself and every code the
    release's retry table names."""
    sources = (Path(server.__file__).parent / name for name in ("client.py", "server.py"))
    written = {
        code
        for source in sources
        for code in re.findall(r'error="([a-z_]+)"', source.read_text(encoding="utf-8"))
    }
    sampled = {(await _sample(name))[1].error for name in _REFUSAL_SAMPLES}
    assert written | set(RELEASE["retry_modes"]) <= sampled


@pytest.mark.parametrize("sample", list(_REFUSAL_SAMPLES))
async def test_a_presentation_fix_names_only_arguments_and_values_its_tool_accepts(
    sample: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A fix key that is an argument of any presentation tool is one of this tool's,
    with a value its schema accepts. Display text and pointers stay out of the fix,
    and a `request_invalid` fix names the tool, its accepted arguments, and problems
    on those arguments."""
    monkeypatch.setattr(server, "PlatformAssembler", _mock_platform_assembler(RELEASE))
    tool, refusal = await _sample(sample)

    async with Client(mcp) as client:
        schemas = {listed.name: listed.input_schema for listed in await client.list_tools()}
        envelope = server._refusal_result(refusal, tool=tool).structured_content

    assert envelope is not None
    arguments = set().union(*(schema["properties"] for schema in schemas.values()))
    properties = schemas[tool]["properties"]
    fix = envelope["fix"]
    assert not {"advice", "pointer", "detail"} & set(fix), (sample, fix)
    for key, value in fix.items():
        if key in arguments:
            assert key in properties, (sample, tool, key)
            jsonschema.validate(value, properties[key])
    if envelope["code"] == "request_invalid":
        assert fix["tool"] == tool
        # The public listing, which leaves out what the tool's public shape hides.
        hidden = server.PUBLIC_SHAPES[tool].hidden if tool in server.PUBLIC_SHAPES else ()
        assert fix["accepted"] == [name for name in properties if name not in hidden]
        for problem in fix["problems"]:
            field, separator, _ = problem.partition(": ")
            if separator and " " not in field:
                assert field.split(".")[0] in properties, (sample, problem)
