"""Tests for definition writes: the definitions a Search envelope carries, the check a caller's definition passes, and an identity over other years."""

from __future__ import annotations

import copy
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
from data_mcp.definition_store import (
    DefinitionStoreUnavailable,
    IdentityMismatch,
    InMemoryDefinitionStore,
    StoredDefinition,
    canonicalize,
    get_store,
    set_store,
    spec_id_for,
)
from data_mcp.definition_writes import (
    MAX_IDENTITY_BYTES,
    IdentityUnusable,
    caller_definition,
    put_definitions,
    stored_definition,
    windowed_identity,
)

_FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(autouse=True)
def _memory_store() -> Any:
    set_store(InMemoryDefinitionStore())
    yield
    set_store(None)


class _FailingStore:
    """A store whose backend is unreachable."""

    async def put_many(self, definitions: Sequence[StoredDefinition]) -> None:
        raise DefinitionStoreUnavailable("backend unreachable")

    async def get(self, spec_id: str, *, schema_version: int) -> StoredDefinition | None:
        raise DefinitionStoreUnavailable("backend unreachable")

    async def aclose(self) -> None:
        pass


def _spec(fixture: str) -> dict[str, Any]:
    """The definite Spec block of a recorded Search API answer; its identity hashes to its spec_id."""
    return json.loads((_FIXTURES / fixture).read_text())["parts"][0]["spec"]


async def test_a_searched_definition_is_stored_as_its_identity_alone() -> None:
    spec = _spec("resolve_health_spending_broad.json")

    definition = stored_definition(spec)
    assert definition is not None
    assert await put_definitions([definition]) is None
    held = await get_store().get(spec["spec_id"], schema_version=1)

    assert held == StoredDefinition(
        spec_id=spec["spec_id"],
        schema_version=1,
        canonical_bytes=canonicalize(spec["definition"]["identity"]),
    )


def test_a_block_without_a_definition_stores_nothing() -> None:
    spec = _spec("resolve_population_guatemala.json")

    assert stored_definition({**spec, "definition": None}) is None
    assert stored_definition({**spec, "definition": {"schema_version": 1}}) is None


def test_a_definition_that_does_not_hash_to_its_spec_id_is_refused() -> None:
    spec = copy.deepcopy(_spec("resolve_population_guatemala.json"))
    spec["definition"]["identity"]["requested_window"] = {"kind": "latest"}

    with pytest.raises(IdentityMismatch) as refused:
        stored_definition(spec)

    assert refused.value.expected_spec_id == spec["spec_id"]


async def test_an_unreachable_store_refuses_retryably() -> None:
    set_store(_FailingStore())
    definition = stored_definition(_spec("resolve_population_guatemala.json"))
    assert definition is not None

    result = await put_definitions([definition])

    assert result is not None
    assert result.structured_content is not None
    assert result.structured_content["code"] == "service_unavailable"


async def test_nothing_to_write_never_reaches_the_store() -> None:
    set_store(_FailingStore())

    assert await put_definitions([]) is None


def test_a_searched_definition_is_its_identity_alone() -> None:
    spec = _spec("resolve_health_spending_broad.json")

    assert stored_definition(spec) == caller_definition(spec["definition"])


def test_a_caller_definition_that_does_not_hash_to_its_spec_id_is_refused() -> None:
    spec = copy.deepcopy(_spec("resolve_population_guatemala.json"))
    spec["definition"]["identity"]["requested_window"] = {"kind": "latest"}

    with pytest.raises(IdentityMismatch):
        caller_definition(spec["definition"])


def _hashed(identity: dict[str, Any]) -> dict[str, Any]:
    """A `{schema_version, spec_id, identity}` sidecar carrying `identity`."""
    return {
        "schema_version": 1,
        "spec_id": spec_id_for(canonicalize(identity)),
        "identity": identity,
    }


def test_a_caller_identity_past_the_registry_cap_is_refused() -> None:
    """dfo-apis's registry refuses an identity past 65,536 bytes of compact JSON."""
    identity = {"identity_version": "spec1", "padding": "x" * MAX_IDENTITY_BYTES}

    with pytest.raises(IdentityUnusable, match="past the 65,536-byte cap"):
        caller_definition(_hashed(identity))


def test_a_caller_identity_at_the_registry_cap_is_taken() -> None:
    bare = {"identity_version": "spec1", "padding": ""}
    room = MAX_IDENTITY_BYTES - len(json.dumps(bare, separators=(",", ":")).encode())
    identity = {**bare, "padding": "x" * room}

    definition = caller_definition(_hashed(identity))

    assert len(definition.canonical_bytes) == MAX_IDENTITY_BYTES


def test_a_caller_identity_with_no_canonical_form_is_refused() -> None:
    """RFC 8785 has no form for an integer past 2**53, so no spec_id can be computed."""
    definition = {
        "schema_version": 1,
        "spec_id": "spec_" + "0" * 32,
        "identity": {"identity_version": "spec1", "count": 2**70},
    }

    with pytest.raises(IdentityUnusable, match="no canonical JSON form"):
        caller_definition(definition)


def test_a_windowed_identity_hashes_as_the_scope_surgery_would() -> None:
    spec = _spec("resolve_health_spending_broad.json")
    window = {"kind": "range", "start_year": 2024, "end_year": 2024}

    spec_id, identity = windowed_identity(spec["definition"]["identity"], window)

    # dfo-apis's `apply_scope_surgery` replaces `requested_window` whole and
    # hashes the result, so this is the spec_id `get_data`'s scope mints.
    expected = {**spec["definition"]["identity"], "requested_window": window}
    assert identity == expected
    assert spec_id == spec_id_for(canonicalize(expected))
    assert spec_id != spec["spec_id"]


@pytest.mark.parametrize(
    "answer",
    json.loads((_FIXTURES / "scoped_mint_2024.json").read_text())["answers"],
    ids=["education-aid-to-kenya", "education-aid-every-donor"],
)
def test_a_windowed_identity_is_the_one_dfo_apis_mints_for_the_scoped_seed(
    answer: dict[str, Any],
) -> None:
    """Recorded from the dev dfo-apis: `get_data(seed, scope={"window": 2024})`.

    The Kenya mint answered its definition. The every-donor mint refused
    `entities_unbound` after dfo-apis registered the scoped definition, so
    only its spec_id is recorded: the dev registry answers 422 for that id,
    and 404 for one it never registered.
    """
    window = json.loads((_FIXTURES / "scoped_mint_2024.json").read_text())["window"]

    spec_id, identity = windowed_identity(answer["seed"]["identity"], window)

    assert spec_id == answer["minted"]["spec_id"]
    if answer["minted"]["identity"] is not None:
        assert identity == answer["minted"]["identity"]


@pytest.mark.parametrize(
    ("fixture", "start_year", "minted"),
    [
        ("resolve_health_spending_broad.json", 2000, "spec_45564d96005936f5ee6d649e84d6df64"),
        ("resolve_gdp_per_capita_guatemala.json", 1990, "spec_184dd9f8d800d0a2059882c1c381b50f"),
    ],
)
def test_an_identity_over_a_range_with_no_end_year_is_the_one_dfo_apis_serves(
    fixture: str, start_year: int, minted: str
) -> None:
    """A range stating `start_year` alone hashes as dfo-apis hashes `scope.window` without `end_year`.

    Each `minted` id is what dfo-apis at b1bb24896 (dfo-spec-kit adbebda, the
    commit this repository pins) computes for the fixture's identity under
    `Scope(window=WindowRange(start_year=start_year))`, by the steps of
    `datasets_api.kinds.spec_id._served_identity`: `apply_scope_surgery`,
    whose `window_to_identity` writes no `end_year` key, then
    `code_index.members.canonical_identity` over the packaged code index,
    then `spec_registry.recompute_identity(...).spec_id`.
    """
    seed = _spec(fixture)["definition"]["identity"]
    window = {"kind": "range", "start_year": start_year}

    spec_id, identity = windowed_identity(seed, window)

    assert spec_id == minted
    assert identity["requested_window"] == window
