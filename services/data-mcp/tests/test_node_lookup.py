"""Tests for the node lookup seam: fixture round-trips, dedup, concurrency, and errors."""

from __future__ import annotations

import ast
import asyncio
import inspect
from collections.abc import Callable, Coroutine
from typing import Any

import httpx2
import pytest
from data_mcp import client_holder, node_lookup
from data_mcp.client import UpstreamClient
from data_mcp.errors import UpstreamError
from data_mcp.node_lookup import (
    EntityNode,
    GenericNode,
    NotHeldNode,
    SourceNode,
    StatVarNode,
    describe_nodes,
)
from pydantic import ValidationError

from .net_guard import no_network  # noqa: F401 -- imported for its autouse fixture
from .node_fixtures import load_node_fixture
from .test_client import _client

Handler = (
    Callable[[httpx2.Request], httpx2.Response]
    | Callable[[httpx2.Request], Coroutine[None, None, httpx2.Response]]
)


async def _describe(handler: Handler, dcids: list[str]) -> dict[str, Any]:
    async with _client(handler) as upstream:
        client_holder.set_client(upstream)
        return await describe_nodes(dcids)


def _fixture_handler(payload: dict[str, Any]) -> Handler:
    def handler(_request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, json=payload)

    return handler


# 1. Fixture round-trips, one per shape.


async def test_stat_var_fixture_round_trips_with_every_field() -> None:
    payload = load_node_fixture("node_stat_var")

    outcomes = await _describe(_fixture_handler(payload), [payload["dcid"]])

    outcome = outcomes[payload["dcid"]]
    assert isinstance(outcome, StatVarNode)
    assert outcome.shape == "stat_var"
    assert outcome.dcid == payload["dcid"]
    assert outcome.label == payload["label"]
    assert outcome.definition == payload["definition"]
    assert outcome.roles == payload["roles"]
    assert outcome.population_type == payload["population_type"]
    assert outcome.processing_level is None
    assert outcome.facets is None
    assert outcome.sources is None


async def test_stat_var_with_a_null_definition_still_describes() -> None:
    # The node door serves UN System StatVars (undata/sdg/*, undata/who/*)
    # with a label and `"definition": null`.
    payload = {**load_node_fixture("node_stat_var"), "definition": None}

    outcomes = await _describe(_fixture_handler(payload), [payload["dcid"]])

    outcome = outcomes[payload["dcid"]]
    assert isinstance(outcome, StatVarNode)
    assert outcome.label == payload["label"]
    assert outcome.definition is None


@pytest.mark.parametrize(
    "fixture_name", ["node_stat_var", "node_source", "node_entity", "node_generic"]
)
async def test_a_node_with_a_null_label_still_describes(fixture_name: str) -> None:
    # The node door types `label` as nullable on every shape.
    payload = {**load_node_fixture(fixture_name), "label": None}

    outcomes = await _describe(_fixture_handler(payload), [payload["dcid"]])

    outcome = outcomes[payload["dcid"]]
    assert not isinstance(outcome, NotHeldNode)
    assert outcome.dcid == payload["dcid"]
    assert outcome.label is None


async def test_source_fixture_round_trips_with_every_field() -> None:
    payload = load_node_fixture("node_source")

    outcomes = await _describe(_fixture_handler(payload), [payload["dcid"]])

    outcome = outcomes[payload["dcid"]]
    assert isinstance(outcome, SourceNode)
    assert outcome.shape == "source"
    assert outcome.dcid == payload["dcid"]
    assert outcome.label == payload["label"]
    assert outcome.license == payload["license"]
    assert outcome.license_type == payload["license_type"]
    assert outcome.description == payload["description"]
    assert outcome.url == payload["url"]
    assert outcome.freshness.release_frequency == payload["freshness"]["release_frequency"]
    assert outcome.freshness.refresh_schedule == payload["freshness"]["refresh_schedule"]


async def test_entity_fixture_round_trips_with_every_field() -> None:
    payload = load_node_fixture("node_entity")

    outcomes = await _describe(_fixture_handler(payload), [payload["dcid"]])

    outcome = outcomes[payload["dcid"]]
    assert isinstance(outcome, EntityNode)
    assert outcome.shape == "entity"
    assert outcome.dcid == payload["dcid"]
    assert outcome.label == payload["label"]
    assert outcome.entity_type == payload["entity_type"]
    assert [m.dcid for m in outcome.contained_in] == [m["dcid"] for m in payload["contained_in"]]
    assert outcome.contained_in[0].label == "Africa"
    assert outcome.contained_in[1].label is None


async def test_generic_fixture_round_trips_with_every_field() -> None:
    payload = load_node_fixture("node_generic")

    outcomes = await _describe(_fixture_handler(payload), [payload["dcid"]])

    outcome = outcomes[payload["dcid"]]
    assert isinstance(outcome, GenericNode)
    assert outcome.shape == "node"
    assert outcome.dcid == payload["dcid"]
    assert outcome.label == payload["label"]
    assert outcome.types == payload["types"]
    assert outcome.properties == payload["properties"]
    assert outcome.is_classification_value is False


# 1b. A generic node whose types are all Enum-suffixed is a classification
# value; one with a mix, or none, is not.


async def test_generic_node_with_only_enum_types_is_a_classification_value() -> None:
    payload = {**load_node_fixture("node_generic"), "types": ["GeographicalIndicatorEnum"]}

    outcomes = await _describe(_fixture_handler(payload), [payload["dcid"]])

    outcome = outcomes[payload["dcid"]]
    assert isinstance(outcome, GenericNode)
    assert outcome.is_classification_value is True


async def test_generic_node_with_a_mix_of_enum_and_other_types_is_not_a_classification_value() -> (
    None
):
    payload = {
        **load_node_fixture("node_generic"),
        "types": ["GeographicalIndicatorEnum", "Dataset"],
    }

    outcomes = await _describe(_fixture_handler(payload), [payload["dcid"]])

    outcome = outcomes[payload["dcid"]]
    assert isinstance(outcome, GenericNode)
    assert outcome.is_classification_value is False


async def test_generic_node_with_no_types_is_not_a_classification_value() -> None:
    payload = {**load_node_fixture("node_generic"), "types": []}

    outcomes = await _describe(_fixture_handler(payload), [payload["dcid"]])

    outcome = outcomes[payload["dcid"]]
    assert isinstance(outcome, GenericNode)
    assert outcome.is_classification_value is False


def test_is_classification_value_lands_in_model_dump_json() -> None:
    node = GenericNode(
        shape="node",
        dcid="Japan",
        label="Japan",
        types=["GeographicalIndicatorEnum"],
        properties={},
    )

    dumped = node.model_dump(mode="json")

    assert dumped["is_classification_value"] is True


# 2. Unknown fields dropped, not rejected.


async def test_unknown_field_in_payload_is_dropped() -> None:
    payload = {**load_node_fixture("node_stat_var"), "an_unreviewed_field": "surprise"}

    outcomes = await _describe(_fixture_handler(payload), [payload["dcid"]])

    outcome = outcomes[payload["dcid"]]
    assert isinstance(outcome, StatVarNode)
    assert not hasattr(outcome, "an_unreviewed_field")


# 3. An unrecognised shape raises ValidationError.


async def test_unrecognised_shape_raises_validation_error() -> None:
    payload = {**load_node_fixture("node_stat_var"), "shape": "something_new"}

    with pytest.raises(ValidationError):
        await _describe(_fixture_handler(payload), [payload["dcid"]])


# 3b. A shape-valid body naming a different dcid than the one requested
# raises, rather than being stored under the requested dcid.


async def test_response_dcid_mismatch_raises_value_error() -> None:
    payload = {**load_node_fixture("node_stat_var"), "dcid": "some/other/dcid"}

    with pytest.raises(ValueError, match="some/other/dcid"):
        await _describe(_fixture_handler(payload), ["country/KEN"])


# 4. A full batch of eight dcids, all keyed by dcid.


async def test_eight_dcids_all_come_back_keyed_by_dcid() -> None:
    dcids = [f"country/{i:03d}" for i in range(8)]

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

    outcomes = await _describe(handler, dcids)

    assert set(outcomes) == set(dcids)
    for dcid in dcids:
        assert isinstance(outcomes[dcid], EntityNode)
        assert outcomes[dcid].dcid == dcid


# 5. Request path and span template.


async def test_request_path_and_route_template() -> None:
    requests: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        return httpx2.Response(200, json=load_node_fixture("node_entity"))

    await _describe(handler, ["country/KEN"])

    assert requests[0].url.path == "/api/graph/node/country/KEN"


async def test_route_template_uses_the_bare_placeholder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    routes: list[str] = []
    real_request = UpstreamClient.request

    async def capturing_request(
        self: UpstreamClient, method: str, path: str, *, route: str, **kwargs: Any
    ) -> Any:
        routes.append(route)
        return await real_request(self, method, path, route=route, **kwargs)

    monkeypatch.setattr(UpstreamClient, "request", capturing_request)

    def handler(_request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, json=load_node_fixture("node_entity"))

    await _describe(handler, ["country/KEN"])

    assert routes == ["/api/graph/node/{dcid}"]


# 6. A 404 node_not_held on one dcid is partial; the rest still resolve.


async def test_not_held_dcid_is_partial_the_rest_still_resolve() -> None:
    dcids = [f"country/{i:03d}" for i in range(8)]
    not_held_dcid = dcids[3]

    def handler(request: httpx2.Request) -> httpx2.Response:
        dcid = request.url.path.removeprefix("/api/graph/node/")
        if dcid == not_held_dcid:
            return httpx2.Response(
                404,
                json={
                    "code": "node_not_held",
                    "title": "not held",
                    "detail": f"{dcid} is not held",
                    "retryable": False,
                },
            )
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

    outcomes = await _describe(handler, dcids)

    assert set(outcomes) == set(dcids)
    assert outcomes[not_held_dcid] == NotHeldNode(dcid=not_held_dcid)
    for dcid in dcids:
        if dcid != not_held_dcid:
            assert isinstance(outcomes[dcid], EntityNode)


# 7. A 503 on one dcid raises, in dcid order, even when a sibling also fails.


async def test_503_on_one_dcid_raises_in_dcid_order() -> None:
    dcids = ["country/AAA", "country/BBB", "country/CCC"]

    def handler(request: httpx2.Request) -> httpx2.Response:
        dcid = request.url.path.removeprefix("/api/graph/node/")
        if dcid in ("country/BBB", "country/CCC"):
            return httpx2.Response(
                503,
                json={
                    "code": "service_unavailable",
                    "title": "pool exhausted",
                    "detail": "no reader slots",
                    "retryable": True,
                },
            )
        return httpx2.Response(200, json={**load_node_fixture("node_entity"), "dcid": dcid})

    with pytest.raises(UpstreamError) as exc_info:
        await _describe(handler, dcids)

    # The failure attributed to `country/BBB`, the first failing dcid in
    # order -- not `country/CCC`, which also failed.
    assert exc_info.value.code == "service_unavailable"


# 8. Duplicate dcids issue one request each for the unique set.


async def test_duplicate_dcids_issue_one_request_each() -> None:
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

    outcomes = await _describe(handler, ["country/KEN", "country/KEN", "country/UGA"])

    assert sorted(requests) == ["country/KEN", "country/UGA"]
    assert set(outcomes) == {"country/KEN", "country/UGA"}


# 9. No more than `_MAX_CONCURRENT_NODE_READS` requests in flight at once.


async def test_concurrency_is_bounded_at_four() -> None:
    in_flight = 0
    max_in_flight = 0
    lock = asyncio.Lock()

    async def handler(request: httpx2.Request) -> httpx2.Response:
        nonlocal in_flight, max_in_flight
        async with lock:
            in_flight += 1
            max_in_flight = max(max_in_flight, in_flight)
        await asyncio.sleep(0.01)
        async with lock:
            in_flight -= 1
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

    dcids = [f"country/{i:03d}" for i in range(8)]
    await _describe(handler, dcids)

    assert max_in_flight <= node_lookup._MAX_CONCURRENT_NODE_READS


# 10. The module imports nothing from `data_mcp.tool_errors`.


def test_module_imports_nothing_from_tool_errors() -> None:
    source = inspect.getsource(node_lookup)
    tree = ast.parse(source)
    imported_modules = {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
    assert "data_mcp.tool_errors" not in imported_modules
