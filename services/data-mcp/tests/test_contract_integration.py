"""Contract check against a REAL apis. Requires DFO_APIS_URL.

Skipped automatically otherwise (no network access in CI). Run manually
against a live apis Search API to catch a contract drift between this
service's client.py and the upstream API before it turns into a runtime
error, e.g.:

    DFO_APIS_URL=http://localhost:8083 \\
        UV_FROZEN=1 uv run --package data-mcp pytest services/data-mcp -k contract
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import httpx2
import jsonschema
import pytest
from data_mcp.definition_store import canonicalize, spec_id_for
from data_mcp.graph_wire import (
    CountryNode,
    DatasetNode,
    NodeResponse,
    ProvenanceNode,
    SourceNode,
    StatVarNode,
)
from data_mcp.search_tool import SearchPart, SearchSource, to_wire_request

_GRAPH_SCHEMA_SNAPSHOT = Path(__file__).resolve().parent / "fixtures" / "graph_api_schema.json"

pytestmark = pytest.mark.skipif(
    not os.environ.get("DFO_APIS_URL"),
    reason="set DFO_APIS_URL to a live apis to run this contract check",
)


async def test_schema_endpoint_reports_expected_contract_version() -> None:
    base_url = os.environ["DFO_APIS_URL"]
    async with httpx2.AsyncClient(base_url=base_url, timeout=10.0) as client:
        response = await client.get("/api/search/schema")

    response.raise_for_status()
    body = response.json()

    # A loose check on purpose, guarding against a MAJOR contract break (a
    # schema_version bump, or the endpoint disappearing/reshaping entirely)
    # rather than pinning every field this client happens to read today.
    assert body.get("schema_version") == 1
    assert isinstance(body.get("request"), dict)
    assert isinstance(body.get("response"), dict)


async def test_wire_request_validates_against_the_published_request_schema() -> None:
    """Pins `to_wire_request()`'s output against the deployment's own published schema.

    Validates the wire shape itself, a tagged union on `kind`, via a full
    `jsonschema` validation rather than a version check, so it catches a
    contract drift a field-name walk would miss.
    """
    base_url = os.environ["DFO_APIS_URL"]
    async with httpx2.AsyncClient(base_url=base_url, timeout=10.0) as client:
        response = await client.get("/api/search/schema")
    response.raise_for_status()
    request_schema = response.json()["request"]

    part = SearchPart(
        variable="debt service",
        sources=[
            SearchSource(
                text="IDS", ref="provenance/WorldBankInternationalDebtStatistics", exclude=True
            )
        ],
    )
    wire = to_wire_request([part], raw_query=None)

    jsonschema.validate(instance=wire, schema=request_schema)


async def test_resolve_round_trip_reports_the_integer_envelope_schema_version() -> None:
    """The envelope carries the integer contract schema_version."""
    base_url = os.environ["DFO_APIS_URL"]
    wire = to_wire_request([SearchPart(variable="malaria grants")], raw_query=None)

    async with httpx2.AsyncClient(base_url=base_url, timeout=30.0) as client:
        response = await client.post("/api/search/resolve", json=wire)

    response.raise_for_status()
    envelope = response.json()

    assert envelope.get("schema_version") == 1
    assert envelope.get("status") in {"definite", "candidates", "no_data", "unsupported"}


async def test_healthz_endpoint_is_reachable() -> None:
    base_url = os.environ["DFO_APIS_URL"]
    async with httpx2.AsyncClient(base_url=base_url, timeout=10.0) as client:
        response = await client.get("/api/search/healthz")

    response.raise_for_status()
    assert response.json().get("status") == "ok"


# Graph node door: validates a real node response against the generated wire models,
# distinct from the fixture round-trips in test_node_lookup.py, which replay apis'
# documented examples.


async def test_live_graph_schema_matches_the_committed_snapshot() -> None:
    """A difference means `just graph-schema` and `just graph-models` are due."""
    base_url = os.environ["DFO_APIS_URL"]
    async with httpx2.AsyncClient(base_url=base_url, timeout=10.0) as client:
        response = await client.get("/api/graph/schema")

    response.raise_for_status()
    assert response.json() == json.loads(_GRAPH_SCHEMA_SNAPSHOT.read_text())


async def test_live_node_door_country_ken_validates_as_country_node() -> None:
    base_url = os.environ["DFO_APIS_URL"]
    async with httpx2.AsyncClient(base_url=base_url, timeout=10.0) as client:
        response = await client.get("/api/graph/node/country/KEN")

    response.raise_for_status()
    node = NodeResponse.model_validate(response.json()).root

    assert isinstance(node, CountryNode)
    assert node.country.contained_in


@pytest.mark.skipif(
    not os.environ.get("DESCRIBE_LIVE_STAT_VAR_DCID"),
    reason="set DESCRIBE_LIVE_STAT_VAR_DCID to a real StatVar dcid to run this check",
)
async def test_live_node_door_stat_var_validates_with_nonempty_roles() -> None:
    base_url = os.environ["DFO_APIS_URL"]
    dcid = os.environ["DESCRIBE_LIVE_STAT_VAR_DCID"]
    async with httpx2.AsyncClient(base_url=base_url, timeout=10.0) as client:
        response = await client.get(f"/api/graph/node/{dcid}")

    response.raise_for_status()
    node = NodeResponse.model_validate(response.json()).root

    assert isinstance(node, StatVarNode)
    assert node.stat_var.definition.roles


@pytest.mark.skipif(
    not os.environ.get("DESCRIBE_LIVE_SOURCE_DCID"),
    reason="set DESCRIBE_LIVE_SOURCE_DCID to a real source/provenance dcid to run this check",
)
async def test_live_node_door_source_reports_licence_and_freshness_presence() -> None:
    """Prints what the live wire actually returns for licence and freshness rather than
    asserting either is populated -- a null value here is a recorded gap, not a failure. Run
    with `-s` to see the printed values; that output is the record of whether these fields
    are served for a real source node.
    """
    base_url = os.environ["DFO_APIS_URL"]
    dcid = os.environ["DESCRIBE_LIVE_SOURCE_DCID"]
    async with httpx2.AsyncClient(base_url=base_url, timeout=10.0) as client:
        response = await client.get(f"/api/graph/node/{dcid}")

    response.raise_for_status()
    node = NodeResponse.model_validate(response.json()).root

    assert isinstance(node, SourceNode | DatasetNode | ProvenanceNode)
    block = getattr(node, node.kind)
    print(f"license={block.license!r} license_type={block.license_type!r}")
    print(
        f"freshness.release_frequency={block.freshness.release_frequency!r} "
        f"freshness.refresh_type={block.freshness.refresh_type!r}"
    )


async def test_snapshot_route_carries_a_stable_link_that_redirects() -> None:
    """The only place the stable-link wire is checked against a running deployment
    rather than against a fixture this suite wrote itself. Mints a real manifest, then
    reads the snapshot route for the returned spec_id/content_version and checks every
    artifact's `url` is the root-relative link, carries no `url_expires_at`, and
    answers 302.

    Skips cleanly when the Datasets surface itself is unmounted
    (`DFO_DATASETS_SURFACE` off -> 404 on `/api/datasets/capabilities`), matching this
    module's own guard for the Search surface.
    """
    base_url = os.environ["DFO_APIS_URL"]
    identity = {
        "identity_version": "spec1",
        "shape": {"id": "MalariaGrantDisbursements", "kind": "stat_var"},
        "entities": {"recipient": ["country/KEN"]},
        "facets": {"provenance": {"ref": "OECD_DAC2A", "state": "value"}},
        "requested_window": {"kind": "all"},
        "slots": [],
    }
    canonical_bytes = canonicalize(identity)
    spec_id = spec_id_for(canonical_bytes)

    async with httpx2.AsyncClient(base_url=base_url, timeout=30.0) as client:
        capabilities = await client.get("/api/datasets/capabilities")
        if capabilities.status_code == 404:
            pytest.skip("the Datasets surface is unmounted (DFO_DATASETS_SURFACE off)")
        capabilities.raise_for_status()

        mint_response = await client.post(
            "/api/datasets",
            json={
                "schema_version": 1,
                "input": {
                    "kind": "spec_id",
                    "spec_id": spec_id,
                    "definition": {"schema_version": 1, "spec_id": spec_id, "identity": identity},
                    "scope": {},
                },
                "options": {"format": "parquet", "bundle": False, "layout": "frame"},
            },
        )
        mint_response.raise_for_status()
        manifest = mint_response.json()

        snapshot_response = await client.get(
            f"/api/datasets/{manifest['definition']['spec_id']}/{manifest['content_version']}"
        )

    snapshot_response.raise_for_status()
    snapshot = snapshot_response.json()

    assert snapshot["artifacts"], "expected at least one still-servable artifact"
    async with httpx2.AsyncClient(base_url=base_url, timeout=30.0) as client:
        for artifact in snapshot["artifacts"]:
            assert "url_expires_at" not in artifact
            assert artifact["url"] == (
                f"/api/datasets/{manifest['definition']['spec_id']}/"
                f"{manifest['content_version']}/files/{artifact['artifact_digest']}"
            )
            link_response = await client.get(artifact["url"])
            assert link_response.status_code == 302
