"""Tests for the node lookup seam: fixture round-trips, batching, dedup, errors, and the
planner line."""

from __future__ import annotations

import ast
import inspect
import json
from collections.abc import Callable
from typing import Any

import httpx2
import pytest
from data_mcp import client_holder, node_lookup
from data_mcp.client import UpstreamClient
from data_mcp.errors import UpstreamError
from data_mcp.graph_wire import (
    Constraint,
    CountryNode,
    Measure,
    NodeRef,
    NotHeldEntry,
    StatVarNode,
)
from data_mcp.node_lookup import (
    definition_line,
    describe_nodes,
    described_variable_text,
    variable_text,
)
from pydantic import ValidationError

from .net_guard import no_network  # noqa: F401 -- imported for its autouse fixture
from .node_fixtures import NODE_FIXTURES, load_node_fixture, node_fixture
from .test_client import _client

Handler = Callable[[httpx2.Request], httpx2.Response]


async def _describe(handler: Handler, dcids: list[str]) -> dict[str, Any]:
    async with _client(handler) as upstream:
        client_holder.set_client(upstream)
        return await describe_nodes(dcids)


def _batch_handler(
    entry_for: Callable[[str], dict[str, Any]], requests: list[list[str]] | None = None
) -> Handler:
    """Answer each batch request with `entry_for(dcid)` per requested dcid, in order."""

    def handler(request: httpx2.Request) -> httpx2.Response:
        dcids = json.loads(request.content)["dcids"]
        if requests is not None:
            requests.append(dcids)
        return httpx2.Response(200, json={"nodes": [entry_for(dcid) for dcid in dcids]})

    return handler


def _country(dcid: str) -> dict[str, Any]:
    return {**load_node_fixture("node_country"), "ref": dcid, "label": dcid}


def _fixture_handler(payload: dict[str, Any]) -> Handler:
    return _batch_handler(lambda _dcid: payload)


# 1. Fixture round-trips, one per kind.


@pytest.mark.parametrize("fixture_name", NODE_FIXTURES)
async def test_each_fixture_round_trips_as_its_kind(fixture_name: str) -> None:
    payload = load_node_fixture(fixture_name)

    outcomes = await _describe(_fixture_handler(payload), [payload["ref"]])

    outcome = outcomes[payload["ref"]]
    assert outcome == node_fixture(fixture_name)
    assert outcome.kind == payload["kind"]
    assert outcome.model_dump(mode="json", by_alias=True) == payload


@pytest.mark.parametrize("fixture_name", NODE_FIXTURES)
async def test_a_node_with_a_null_label_still_describes(fixture_name: str) -> None:
    # The envelope types `label` as nullable on every kind.
    payload = {**load_node_fixture(fixture_name), "label": None}

    outcomes = await _describe(_fixture_handler(payload), [payload["ref"]])

    assert outcomes[payload["ref"]].label is None


async def test_a_stat_var_with_no_availability_still_describes() -> None:
    # A failed profile read or an unloaded coverage index serves `availability: null`.
    payload = load_node_fixture("node_stat_var")
    payload["stat_var"] = {**payload["stat_var"], "availability": None}

    outcomes = await _describe(_fixture_handler(payload), [payload["ref"]])

    outcome = outcomes[payload["ref"]]
    assert isinstance(outcome, StatVarNode)
    assert outcome.stat_var.availability is None


# 2. Unknown fields dropped, not rejected.


async def test_unknown_field_in_payload_is_dropped() -> None:
    payload = {**load_node_fixture("node_stat_var"), "an_unreviewed_field": "surprise"}

    outcomes = await _describe(_fixture_handler(payload), [payload["ref"]])

    outcome = outcomes[payload["ref"]]
    assert isinstance(outcome, StatVarNode)
    assert not hasattr(outcome, "an_unreviewed_field")


# 3. An unrecognised kind raises ValidationError.


async def test_unrecognised_kind_raises_validation_error() -> None:
    payload = {**load_node_fixture("node_country"), "kind": "something_new"}

    with pytest.raises(ValidationError):
        await _describe(_fixture_handler(payload), [payload["ref"]])


# 3b. A valid entry naming a different dcid than the one requested raises, rather than
# being stored under the requested dcid; so does an answer of the wrong length.


async def test_response_ref_mismatch_raises_value_error() -> None:
    payload = {**load_node_fixture("node_country"), "ref": "some/other/dcid"}

    with pytest.raises(ValueError, match="some/other/dcid"):
        await _describe(_fixture_handler(payload), ["country/KEN"])


async def test_an_answer_with_fewer_entries_than_dcids_raises_value_error() -> None:
    def handler(_request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, json={"nodes": [_country("country/KEN")]})

    with pytest.raises(ValueError, match="zip"):
        await _describe(handler, ["country/KEN", "country/UGA"])


# 4. Eight dcids come back keyed by dcid, from one batch request.


async def test_eight_dcids_come_back_keyed_by_dcid_from_one_request() -> None:
    dcids = [f"country/{i:03d}" for i in range(8)]
    requests: list[list[str]] = []

    outcomes = await _describe(_batch_handler(_country, requests), dcids)

    assert requests == [dcids]
    assert list(outcomes) == dcids
    for dcid in dcids:
        assert isinstance(outcomes[dcid], CountryNode)
        assert outcomes[dcid].ref == dcid


async def test_more_dcids_than_one_batch_takes_split_into_batches_of_fifty() -> None:
    dcids = [f"country/{i:03d}" for i in range(120)]
    requests: list[list[str]] = []

    outcomes = await _describe(_batch_handler(_country, requests), dcids)

    assert sorted(len(batch) for batch in requests) == [20, 50, 50]
    assert list(outcomes) == dcids


# 5. Request path, method, body and span template.


async def test_request_posts_the_dcids_to_the_nodes_route(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[httpx2.Request] = []
    routes: list[str] = []
    real_request = UpstreamClient.request

    async def capturing_request(
        self: UpstreamClient, method: str, path: str, *, route: str, **kwargs: Any
    ) -> Any:
        routes.append(route)
        return await real_request(self, method, path, route=route, **kwargs)

    monkeypatch.setattr(UpstreamClient, "request", capturing_request)

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen.append(request)
        return httpx2.Response(200, json={"nodes": [_country("country/KEN")]})

    await _describe(handler, ["country/KEN"])

    assert seen[0].method == "POST"
    assert seen[0].url.path == "/api/graph/nodes"
    assert json.loads(seen[0].content) == {"dcids": ["country/KEN"]}
    assert routes == ["/api/graph/nodes"]


# 6. A `not_held` entry is partial; the rest still describe.


async def test_not_held_entry_is_partial_the_rest_still_describe() -> None:
    dcids = [f"country/{i:03d}" for i in range(8)]
    not_held_dcid = dcids[3]

    def entry_for(dcid: str) -> dict[str, Any]:
        if dcid == not_held_dcid:
            return {"ref": dcid, "status": "not_held"}
        return _country(dcid)

    outcomes = await _describe(_batch_handler(entry_for), dcids)

    assert outcomes[not_held_dcid] == NotHeldEntry(ref=not_held_dcid, status="not_held")
    for dcid in dcids:
        if dcid != not_held_dcid:
            assert isinstance(outcomes[dcid], CountryNode)


# 7. A failed batch request raises.


async def test_503_on_the_batch_raises() -> None:
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

    with pytest.raises(UpstreamError) as exc_info:
        await _describe(handler, ["country/AAA", "country/BBB"])

    assert exc_info.value.code == "service_unavailable"


# 8. Duplicate dcids are requested once.


async def test_duplicate_dcids_are_requested_once() -> None:
    requests: list[list[str]] = []

    outcomes = await _describe(
        _batch_handler(_country, requests), ["country/KEN", "country/KEN", "country/UGA"]
    )

    assert requests == [["country/KEN", "country/UGA"]]
    assert list(outcomes) == ["country/KEN", "country/UGA"]


async def test_no_dcids_makes_no_request() -> None:
    requests: list[list[str]] = []

    assert await _describe(_batch_handler(_country, requests), []) == {}
    assert requests == []


# 9. The planner line: the description, else one line composed from the definition.


def _ref(ref: str, label: str | None = None) -> NodeRef:
    return NodeRef(ref=ref, label=label)


def test_variable_text_is_the_description_where_present() -> None:
    node = node_fixture("node_stat_var")
    assert isinstance(node, StatVarNode)

    assert variable_text(node) == "Gross domestic product at current prices."


def test_variable_text_composes_the_definition_without_a_description() -> None:
    node = node_fixture("node_stat_var", description=None)
    assert isinstance(node, StatVarNode)

    assert variable_text(node) == "Measures gross domestic product of Economic activity."


def test_definition_line_names_measure_and_constraints_by_label_else_ref() -> None:
    node = node_fixture("node_stat_var")
    assert isinstance(node, StatVarNode)
    definition = node.stat_var.definition.model_copy(
        update={
            "measured_property": _ref("amount", "amount"),
            "population_type": _ref("FinancialTransaction"),
            "measure": Measure(label="gross disbursements"),
            "constraints": [
                Constraint(property=_ref("sector", "sector"), value=_ref("CRS_121", "Health")),
                Constraint(property=_ref("financingRecipient"), value=_ref("country/KEN")),
            ],
        }
    )

    assert definition_line(definition) == (
        "Measures amount of FinancialTransaction as gross disbursements; "
        "sector Health; financingRecipient country/KEN."
    )


def test_definition_line_is_none_when_the_definition_names_no_part() -> None:
    node = node_fixture("node_stat_var")
    assert isinstance(node, StatVarNode)
    definition = node.stat_var.definition.model_copy(
        update={"measured_property": None, "population_type": None, "constraints": None}
    )

    assert definition_line(definition) is None


def test_described_variable_text_reads_a_dumped_envelope_and_skips_other_kinds() -> None:
    stat_var = load_node_fixture("node_stat_var")

    assert described_variable_text(stat_var) == "Gross domestic product at current prices."
    assert described_variable_text(load_node_fixture("node_country")) is None


# 10. The module imports nothing from `data_mcp.tool_errors`.


def test_module_imports_nothing_from_tool_errors() -> None:
    source = inspect.getsource(node_lookup)
    tree = ast.parse(source)
    imported_modules = {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
    assert "data_mcp.tool_errors" not in imported_modules
