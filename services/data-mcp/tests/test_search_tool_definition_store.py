"""Tests for `search`'s write path: registering definitions into the definition store."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
from data_mcp import client_holder
from data_mcp.definition_store import (
    DefinitionStore,
    DefinitionStoreUnavailable,
    StoredDefinition,
    canonicalize,
    set_store,
    spec_id_for,
)
from data_mcp.server import mcp
from fastmcp import Client
from fastmcp.client.client import CallToolResult

from .test_client import _client, text_of
from .test_digest import _envelope, _spec
from .test_search_tool import Handler, _json_handler, _reset_client

_FIXTURES = Path(__file__).parent / "fixtures"

__all__ = ["_reset_client"]  # re-exported so pytest picks it up as an autouse fixture here too


@pytest.fixture(autouse=True)
async def _reset_store() -> Any:
    """Every test wires its own store; nothing should leak to the next test."""
    yield
    set_store(None)


class _SpyStore:
    """Records every definition `put_many` is asked to write; never fails."""

    def __init__(self) -> None:
        self.written: list[StoredDefinition] = []

    async def put_many(self, definitions: Sequence[StoredDefinition]) -> None:
        self.written.extend(definitions)

    async def get(self, spec_id: str, *, schema_version: int) -> StoredDefinition | None:
        return None

    async def aclose(self) -> None:
        pass


class _FailingStore:
    """A store whose backend is unreachable: every `put_many` refuses."""

    async def put_many(self, definitions: Sequence[StoredDefinition]) -> None:
        raise DefinitionStoreUnavailable("backend unreachable")

    async def get(self, spec_id: str, *, schema_version: int) -> StoredDefinition | None:
        return None

    async def aclose(self) -> None:
        pass


def _valid_definition(seed: str, *, schema_version: int = 1) -> dict[str, Any]:
    """A `{schema_version, spec_id, identity}` sidecar whose identity genuinely reproduces its spec_id."""
    identity_doc = {"identity_version": "spec1", "seed": seed}
    canonical_bytes = canonicalize(identity_doc)
    return {
        "schema_version": schema_version,
        "spec_id": spec_id_for(canonical_bytes),
        "identity": identity_doc,
    }


def _stored(definition: dict[str, Any]) -> StoredDefinition:
    return StoredDefinition(
        spec_id=definition["spec_id"],
        schema_version=definition["schema_version"],
        canonical_bytes=canonicalize(definition["identity"]),
    )


def _with_elected(spec: dict[str, Any], dcids: list[str]) -> dict[str, Any]:
    """Overlay the wire's real `resolved.stat_vars.items[].ref` shape onto a `_spec()` fixture.

    `_spec()`'s own `stat_vars` parameter builds a bare list, a shape nothing
    in `digest.py` or this module reads; the real search envelope nests
    elected members under `resolved.stat_vars.items`, each item carrying its
    dcid as `ref` (see `fixtures/resolve_population_guatemala.json`'s own
    `parts[0].spec.resolved.stat_vars.items`).
    """
    resolved = {
        **spec["resolved"],
        "stat_vars": {
            "items": [{"ref": dcid, "label": dcid, "slot_values": []} for dcid in dcids],
            "member_count": len(dcids),
            "truncated": False,
        },
    }
    return {**spec, "resolved": resolved}


async def _call_search_with_store(
    handler: Handler, parts: list[dict[str, object]], store: DefinitionStore
) -> CallToolResult:
    async with _client(handler) as upstream:
        client_holder.set_client(upstream)
        async with Client(mcp) as client:
            # `server.py`'s lifespan already built and set its own store on
            # entering `Client(mcp)`; this override lands before the tool
            # call itself runs, since `search` only reads the store via
            # `get_store()` once the request is dispatched below.
            set_store(store)
            return await client.call_tool("search", {"parts": parts}, raise_on_error=False)


# Writes across every status


async def test_definite_interpretation_definition_is_written() -> None:
    definition = _valid_definition("definite")
    spec = {**_spec(spec_id=definition["spec_id"]), "definition": definition}
    envelope = _envelope("definite", spec=spec)
    store = _SpyStore()

    result = await _call_search_with_store(
        _json_handler(200, envelope), [{"variable": "malaria grants"}], store
    )

    assert result.is_error is False
    assert store.written == [_stored(definition)]


async def test_candidates_spec_definitions_are_written() -> None:
    definition_a = _valid_definition("candidate-a")
    definition_b = _valid_definition("candidate-b")
    spec_a = {**_spec(spec_id=definition_a["spec_id"]), "definition": definition_a}
    spec_b = {**_spec(spec_id=definition_b["spec_id"]), "definition": definition_b}
    envelope = _envelope(
        "candidates",
        candidates={
            "ordering": "intent_rank",
            "max_candidates": 5,
            "specs": [spec_a, spec_b],
            "truncated": False,
        },
    )
    store = _SpyStore()

    result = await _call_search_with_store(
        _json_handler(200, envelope), [{"variable": "malaria grants"}], store
    )

    assert result.is_error is False
    assert store.written == [_stored(definition_a), _stored(definition_b)]


async def test_no_data_nearest_real_definitions_are_written() -> None:
    definition = _valid_definition("nearest-real")
    spec = {**_spec(spec_id=definition["spec_id"]), "definition": definition}
    envelope = _envelope(
        "no_data",
        no_data={"reason": "no_observations", "category": "absent", "nearest_real": [spec]},
    )
    store = _SpyStore()

    result = await _call_search_with_store(
        _json_handler(200, envelope), [{"variable": "malaria grants"}], store
    )

    assert result.is_error is False
    assert store.written == [_stored(definition)]


async def test_no_data_flagged_match_definition_is_written_as_a_single_object() -> None:
    definition = _valid_definition("flagged-match")
    # `flagged_match` is a single `FlaggedMatchSpec | None`, never a list.
    flagged_match = {**_spec(spec_id=definition["spec_id"]), "definition": definition}
    envelope = _envelope(
        "no_data",
        no_data={
            "reason": "quality_gate",
            "category": "absent",
            "gate": "quality",
            "flagged_match": flagged_match,
        },
    )
    store = _SpyStore()

    result = await _call_search_with_store(
        _json_handler(200, envelope), [{"variable": "malaria grants"}], store
    )

    assert result.is_error is False
    assert store.written == [_stored(definition)]


async def test_a_sibling_measure_definition_is_written_beside_its_spec() -> None:
    """The recorded aid envelope's sibling, net disbursements, resolves in `get_data`."""
    envelope = json.loads((_FIXTURES / "resolve_aid_france_kenya.json").read_text())
    spec = envelope["parts"][0]["spec"]
    sibling = spec["resolved"]["alternative_measures"][0]["definition"]
    store = _SpyStore()

    result = await _call_search_with_store(
        _json_handler(200, envelope), [{"variable": "aid"}], store
    )

    assert result.is_error is False
    assert store.written == [_stored(spec["definition"]), _stored(sibling)]


async def test_an_unhashable_sibling_is_skipped_and_named_nowhere_in_the_answer(
    caplog: pytest.LogCaptureFixture,
) -> None:
    envelope = json.loads((_FIXTURES / "resolve_aid_france_kenya.json").read_text())
    spec = envelope["parts"][0]["spec"]
    sibling = spec["resolved"]["alternative_measures"][0]
    tampered = {
        **sibling,
        "label": "constant prices",
        "axis": "unit",
        "definition": {**sibling["definition"], "spec_id": "spec_" + "0" * 32},
    }
    spec["resolved"]["alternative_measures"].append(tampered)
    store = _SpyStore()

    result = await _call_search_with_store(
        _json_handler(200, envelope), [{"variable": "aid"}], store
    )

    assert result.is_error is False
    assert store.written == [_stored(spec["definition"]), _stored(sibling["definition"])]
    logged = [record.getMessage() for record in caplog.records]
    assert any(
        tampered["definition"]["spec_id"] in line and spec["spec_id"] in line for line in logged
    )
    shown = text_of(result) + json.dumps(result.structured_content)
    assert sibling["definition"]["spec_id"] in shown
    assert tampered["definition"]["spec_id"] not in shown


# The elected StatVars: `resolved.stat_vars.items[].ref`


async def test_elected_stat_vars_are_neither_named_to_the_caller_nor_stored() -> None:
    """The digest counts them and names none; the store holds the identity alone."""
    definition = _valid_definition("definite-elected")
    spec = _with_elected(
        {**_spec(spec_id=definition["spec_id"]), "definition": definition},
        ["Count_Person_Female", "Count_Person_Male"],
    )
    envelope = _envelope("definite", spec=spec)
    store = _SpyStore()

    result = await _call_search_with_store(
        _json_handler(200, envelope), [{"variable": "population"}], store
    )

    assert result.is_error is False
    assert store.written == [_stored(definition)]
    assert "2 StatVars" in text_of(result)
    assert "Count_Person_" not in text_of(result)
    assert result.structured_content is not None
    assert "Count_Person_" not in json.dumps(result.structured_content)


# Skips


async def test_unsupported_status_definition_is_skipped() -> None:
    envelope = _envelope(
        "unsupported",
        unsupported={
            "code": "facet_binding_unsupported",
            "pointer": "/identity/facets/provenance",
            "detail": "no binding exists",
            # `unsupported`'s own definition omits spec_id by construction.
            "definition": {},
        },
    )
    store = _SpyStore()

    result = await _call_search_with_store(
        _json_handler(200, envelope), [{"variable": "malaria grants"}], store
    )

    assert result.is_error is False
    assert store.written == []


async def test_spec_with_no_definition_is_skipped() -> None:
    # `_spec()` carries no `definition` member at all, matching real fixtures
    # that predate the write path.
    envelope = _envelope("definite", spec=_spec())
    store = _SpyStore()

    result = await _call_search_with_store(
        _json_handler(200, envelope), [{"variable": "malaria grants"}], store
    )

    assert result.is_error is False
    assert store.written == []


# Refusals


async def test_failing_store_refuses_with_service_unavailable_never_definition_not_held() -> None:
    definition = _valid_definition("unavailable")
    spec = {**_spec(spec_id=definition["spec_id"]), "definition": definition}
    envelope = _envelope("definite", spec=spec)

    result = await _call_search_with_store(
        _json_handler(200, envelope), [{"variable": "malaria grants"}], _FailingStore()
    )

    assert result.is_error is True
    assert result.structured_content is not None
    assert result.structured_content["code"] == "service_unavailable"
    assert result.structured_content["code"] != "definition_not_held"
    assert result.structured_content["retry_mode"] == "retry_after_delay"


async def test_failing_store_hands_out_no_spec_id_in_content_or_structured_content() -> None:
    # The refusal above pins the error code; this pins the actual acceptance
    # criterion: a failed write must never let the id it couldn't hold leak
    # out through either half of the response.
    definition = _valid_definition("unavailable-id-leak")
    spec = {**_spec(spec_id=definition["spec_id"]), "definition": definition}
    envelope = _envelope("definite", spec=spec)

    result = await _call_search_with_store(
        _json_handler(200, envelope), [{"variable": "malaria grants"}], _FailingStore()
    )

    assert result.is_error is True
    assert definition["spec_id"] not in text_of(result)
    assert result.structured_content is not None
    assert definition["spec_id"] not in json.dumps(result.structured_content)


async def test_identity_mismatch_refuses_with_capability_unavailable() -> None:
    definition = _valid_definition("tampered")
    definition["spec_id"] = "spec_" + "0" * 32  # does not reproduce this identity's hash
    spec = {**_spec(spec_id=definition["spec_id"]), "definition": definition}
    envelope = _envelope("definite", spec=spec)

    result = await _call_search_with_store(
        _json_handler(200, envelope), [{"variable": "malaria grants"}], _SpyStore()
    )

    assert result.is_error is True
    assert result.structured_content is not None
    assert result.structured_content["code"] == "capability_unavailable"
    assert result.structured_content["retry_mode"] == "do_not_retry"
