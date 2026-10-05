"""Tests for the `describe` tool: caps, grammar, partial success, and refusals."""

from __future__ import annotations

from collections.abc import Callable, Coroutine

import httpx2
import pytest
from data_mcp import client_holder, node_lookup
from data_mcp.describe_tool import (
    DESCRIBE_ID_CAP,
    _cap_and_grammar_refusal,
    _strip_dcid_prefix,
    describe,
)
from data_mcp.node_lookup import EntityNode
from data_mcp.server import mcp
from fastmcp import Client
from fastmcp.client.client import CallToolResult

from .net_guard import no_network  # noqa: F401 -- imported for its autouse fixture
from .node_fixtures import load_node_fixture
from .test_client import _client, text_of

Handler = (
    Callable[[httpx2.Request], httpx2.Response]
    | Callable[[httpx2.Request], Coroutine[None, None, httpx2.Response]]
)


async def _call_describe(handler: Handler, arguments: dict[str, object]) -> CallToolResult:
    async with _client(handler) as upstream:
        client_holder.set_client(upstream)
        async with Client(mcp) as client:
            return await client.call_tool("describe", arguments, raise_on_error=False)


def _entity_handler() -> Handler:
    def handler(request: httpx2.Request) -> httpx2.Response:
        dcid = request.url.path.removeprefix("/api/graph/node/")
        return httpx2.Response(
            200,
            json={
                "shape": "entity",
                "dcid": dcid,
                "label": dcid,
                "entity_type": "geo.country",
                "contained_in": [],
            },
        )

    return handler


def _not_held_response(dcid: str) -> httpx2.Response:
    return httpx2.Response(
        404,
        json={
            "code": "node_not_held",
            "title": "not held",
            "detail": f"{dcid} is not held",
            "retryable": False,
        },
    )


# 1. Eight ids return one successful result whose digest carries all eight.


async def test_eight_ids_return_one_successful_digest_with_all_eight() -> None:
    dcids = [f"country/{i:03d}" for i in range(8)]

    result = await _call_describe(_entity_handler(), {"ids": dcids})

    assert result.is_error is False
    text = text_of(result)
    for dcid in dcids:
        assert dcid in text


# 2. Nine ids are rejected by the published schema before the tool runs.


async def test_nine_ids_rejected_by_the_published_schema() -> None:
    dcids = [f"country/{i:03d}" for i in range(9)]

    def handler(_request: httpx2.Request) -> httpx2.Response:
        raise AssertionError("no upstream call expected when the cap is exceeded")

    result = await _call_describe(handler, {"ids": dcids})

    assert result.is_error is True


# 3. Zero ids are rejected by the published schema before the tool runs.


async def test_zero_ids_rejected_by_the_published_schema() -> None:
    def handler(_request: httpx2.Request) -> httpx2.Response:
        raise AssertionError("no upstream call expected for an empty call")

    result = await _call_describe(handler, {"ids": []})

    assert result.is_error is True


# In-process callers bypass fastmcp's own schema validation, so
# `_cap_and_grammar_refusal` still guards the cap directly and produces a
# readable `request_invalid`.


def test_cap_and_grammar_refusal_rejects_an_empty_ids_list() -> None:
    refusal = _cap_and_grammar_refusal([])

    assert refusal is not None
    assert refusal.is_error is True
    assert refusal.structured_content is not None
    assert refusal.structured_content["code"] == "request_invalid"
    assert "at least one dcid per call" in text_of(refusal)


def test_cap_and_grammar_refusal_rejects_an_over_cap_ids_list() -> None:
    dcids = [f"country/{i:03d}" for i in range(9)]

    refusal = _cap_and_grammar_refusal(dcids)

    assert refusal is not None
    assert refusal.is_error is True
    assert refusal.structured_content is not None
    assert refusal.structured_content["code"] == "request_invalid"
    assert "at most 8 dcids per call, got 9" in text_of(refusal)


# 4. A malformed id refuses request_invalid and issues no upstream request.


@pytest.mark.parametrize("malformed", ["", "a b", "/leading"])
async def test_malformed_id_refuses_request_invalid_with_no_upstream_call(
    malformed: str,
) -> None:
    def handler(_request: httpx2.Request) -> httpx2.Response:
        raise AssertionError("no upstream call expected for a malformed dcid")

    result = await _call_describe(handler, {"ids": [malformed]})

    assert result.is_error is True
    assert result.structured_content is not None
    assert result.structured_content["code"] == "request_invalid"
    assert result.structured_content["fix"]["pointer"] == "/ids/0"


# A dot-segment dcid refuses request_invalid and issues no upstream request.


@pytest.mark.parametrize("traversal", ["..", ".", "country/../../api/datasets", "a/.."])
async def test_dot_segment_dcid_refuses_request_invalid_with_no_upstream_call(
    traversal: str,
) -> None:
    requests: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        raise AssertionError("no upstream call expected for a dot-segment dcid")

    result = await _call_describe(handler, {"ids": [traversal]})

    assert result.is_error is True
    assert result.structured_content is not None
    assert result.structured_content["code"] == "request_invalid"
    assert result.structured_content["fix"]["pointer"] == "/ids/0"
    assert requests == []


# 5. One unheld dcid among eight: partial success.


async def test_one_unheld_among_eight_is_partial_success() -> None:
    dcids = [f"country/{i:03d}" for i in range(8)]
    not_held_dcid = dcids[3]

    def handler(request: httpx2.Request) -> httpx2.Response:
        dcid = request.url.path.removeprefix("/api/graph/node/")
        if dcid == not_held_dcid:
            return _not_held_response(dcid)
        return httpx2.Response(
            200,
            json={
                "shape": "entity",
                "dcid": dcid,
                "label": dcid,
                "entity_type": "geo.country",
                "contained_in": [],
            },
        )

    result = await _call_describe(handler, {"ids": dcids})

    assert result.is_error is True
    assert result.structured_content is not None
    assert result.structured_content["code"] == "node_not_held"
    text = text_of(result)
    for dcid in dcids:
        if dcid != not_held_dcid:
            assert dcid in text
    assert not_held_dcid in text


# 6. Every dcid unheld: refusal with no rendered descriptions.


async def test_every_dcid_unheld_refuses_with_no_rendered_descriptions() -> None:
    dcids = ["country/AAA", "country/BBB"]

    def handler(request: httpx2.Request) -> httpx2.Response:
        dcid = request.url.path.removeprefix("/api/graph/node/")
        return _not_held_response(dcid)

    result = await _call_describe(handler, {"ids": dcids})

    assert result.is_error is True
    assert result.structured_content is not None
    assert result.structured_content["code"] == "node_not_held"
    text = text_of(result)
    assert "country/AAA" in text
    assert "country/BBB" in text
    assert "label" not in text.lower()


# 7. structured_content on the partial-success case is exactly the refusal
# envelope, unchanged by the preamble.


async def test_partial_success_structured_content_matches_a_plain_refusal() -> None:
    dcids = ["country/AAA", "country/BBB"]

    def handler(request: httpx2.Request) -> httpx2.Response:
        dcid = request.url.path.removeprefix("/api/graph/node/")
        if dcid == "country/AAA":
            return httpx2.Response(
                200,
                json={
                    "shape": "entity",
                    "dcid": dcid,
                    "label": dcid,
                    "entity_type": "geo.country",
                    "contained_in": [],
                },
            )
        return _not_held_response(dcid)

    result = await _call_describe(handler, {"ids": dcids})

    assert result.is_error is True
    assert result.structured_content is not None
    assert result.structured_content["code"] == "node_not_held"
    assert result.structured_content["fix"] == {"dcids": ["country/BBB"]}


# 8. A successful call's structured_content carries `nodes`, and a field
# present in the digest is also present in structured content.


async def test_successful_call_structured_content_carries_nodes() -> None:
    dcid = "country/KEN"

    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, json=load_node_fixture("node_entity"))

    result = await _call_describe(handler, {"ids": [dcid]})

    assert result.is_error is False
    assert result.structured_content is not None
    nodes = result.structured_content["nodes"]
    assert len(nodes) == 1
    assert nodes[0]["label"] == "Kenya"
    assert "Kenya" in text_of(result)


# 9. A 503 from the door refuses service_unavailable with retry_after_delay.


async def test_503_from_the_door_refuses_service_unavailable() -> None:
    def handler(_request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(
            503,
            json={
                "code": "service_unavailable",
                "title": "pool exhausted",
                "detail": "no reader slots",
                "retryable": True,
            },
        )

    result = await _call_describe(handler, {"ids": ["country/KEN"]})

    assert result.is_error is True
    assert result.structured_content is not None
    assert result.structured_content["code"] == "service_unavailable"
    assert result.structured_content["retry_mode"] == "retry_after_delay"


# 10. A 2xx body that will not validate refuses upstream_error / do_not_retry.


async def test_unvalidatable_2xx_body_refuses_upstream_response_unusable() -> None:
    def handler(_request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, json={"shape": "not_a_real_shape"})

    result = await _call_describe(handler, {"ids": ["country/KEN"]})

    assert result.is_error is True
    assert result.structured_content is not None
    assert result.structured_content["code"] == "upstream_error"
    assert result.structured_content["retry_mode"] == "do_not_retry"


# A well-formed node body naming a different dcid than the one requested
# refuses upstream_response_unusable rather than being rendered.


async def test_response_dcid_mismatch_refuses_upstream_response_unusable() -> None:
    def handler(_request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(
            200,
            json={
                "shape": "entity",
                "dcid": "country/OTHER",
                "label": "Someone Else",
                "entity_type": "geo.country",
                "contained_in": [],
            },
        )

    result = await _call_describe(handler, {"ids": ["country/KEN"]})

    assert result.is_error is True
    assert result.structured_content is not None
    assert result.structured_content["code"] == "upstream_error"
    assert result.structured_content["retry_mode"] == "do_not_retry"


# 11. With `node_lookup.describe_nodes` monkeypatched to a stub, the tool
# produces its full digest and issues no HTTP request -- proving the tool
# never reaches past the node lookup seam.


async def test_stubbed_node_lookup_produces_the_full_digest_with_no_http_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stub_node = EntityNode(
        shape="entity",
        dcid="country/KEN",
        label="Kenya",
        entity_type="geo.country",
        contained_in=[],
    )

    async def stub_describe_nodes(dcids: list[str]) -> dict[str, object]:
        return dict.fromkeys(dcids, stub_node)

    monkeypatch.setattr(node_lookup, "describe_nodes", stub_describe_nodes)

    def handler(_request: httpx2.Request) -> httpx2.Response:
        raise AssertionError("no HTTP request expected when describe_nodes is stubbed")

    result = await _call_describe(handler, {"ids": ["country/KEN"]})

    assert result.is_error is False
    assert "Kenya" in text_of(result)


# 12. `describe` is among the registered tools -- see test_server.py.

# 13. `to_tool_result(refusal, preamble=...)` -- see test_tool_errors.py.

# 14. The published input schema carries the cap as `minItems`/`maxItems`,
# not just as a runtime check.


async def test_published_schema_carries_the_ids_length_bounds() -> None:
    tool = await mcp.get_tool("describe")

    ids_schema = tool.parameters["properties"]["ids"]

    assert ids_schema["minItems"] == 1
    assert ids_schema["maxItems"] == DESCRIBE_ID_CAP


# 15. The docstring states that labels return in English regardless of the
# query's language.


def test_docstring_states_labels_return_in_english() -> None:
    assert describe.__doc__ is not None
    assert "English" in describe.__doc__


# 16. The Field description and docstring name the parameter `ids`, not
# `dcids` -- the keyword `describe`'s own prose used to imply.


async def test_published_schema_ids_description_says_ids_not_dcids() -> None:
    tool = await mcp.get_tool("describe")

    ids_schema = tool.parameters["properties"]["ids"]

    assert "ids to describe" in ids_schema["description"]
    assert "dcids" not in ids_schema["description"]


def test_docstring_says_ids_not_dcids_per_call() -> None:
    assert describe.__doc__ is not None
    assert "1 to 8 ids per call" in describe.__doc__
    assert "dcids per call" not in describe.__doc__


# 17. A leading `dcid:` prefix is stripped before grammar validation and
# dispatch, matching the exact form `get_data`'s own errors quote.


def test_strip_dcid_prefix_removes_only_the_leading_prefix() -> None:
    assert _strip_dcid_prefix(["dcid:sdg/DC_TOF_HLTHL", "country/JPN"]) == [
        "sdg/DC_TOF_HLTHL",
        "country/JPN",
    ]


async def test_leading_dcid_prefix_is_stripped_before_dispatch() -> None:
    requests: list[str] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        dcid = request.url.path.removeprefix("/api/graph/node/")
        requests.append(dcid)
        return httpx2.Response(
            200,
            json={
                "shape": "entity",
                "dcid": dcid,
                "label": dcid,
                "entity_type": "geo.country",
                "contained_in": [],
            },
        )

    result = await _call_describe(handler, {"ids": ["dcid:country/KEN"]})

    assert result.is_error is False
    assert requests == ["country/KEN"]
    assert result.structured_content is not None
    assert result.structured_content["nodes"][0]["dcid"] == "country/KEN"
    assert "dcid:country/KEN" not in text_of(result)


async def test_dcid_prefix_stripped_before_grammar_check_still_rejects_malformed() -> None:
    def handler(_request: httpx2.Request) -> httpx2.Response:
        raise AssertionError("no upstream call expected for a malformed dcid")

    result = await _call_describe(handler, {"ids": ["dcid:a b"]})

    assert result.is_error is True
    assert result.structured_content is not None
    assert result.structured_content["code"] == "request_invalid"


# 18. A classification-value node and a genuine entity, described together,
# both render correctly and distinctly -- `describe(["Japan"])` and
# `describe(["country/JPN"])` stay individually correct.


async def test_classification_value_and_entity_render_distinctly_in_one_call() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        dcid = request.url.path.removeprefix("/api/graph/node/")
        if dcid == "Japan":
            return httpx2.Response(
                200,
                json={
                    "shape": "node",
                    "dcid": "Japan",
                    "label": "Japan",
                    "types": ["GeographicalIndicatorEnum"],
                    "properties": {},
                },
            )
        return httpx2.Response(
            200,
            json={
                "shape": "entity",
                "dcid": "country/JPN",
                "label": "Japan",
                "entity_type": "Country",
                "contained_in": [],
            },
        )

    result = await _call_describe(handler, {"ids": ["Japan", "country/JPN"]})

    assert result.is_error is False
    text = text_of(result)
    assert "classification value" in text
    assert "type: Country" in text
    nodes = result.structured_content["nodes"]
    generic_node = next(n for n in nodes if n["dcid"] == "Japan")
    entity_node = next(n for n in nodes if n["dcid"] == "country/JPN")
    assert generic_node["is_classification_value"] is True
    assert "is_classification_value" not in entity_node
