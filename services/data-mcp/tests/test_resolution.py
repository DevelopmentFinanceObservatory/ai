"""Tests for the entity resolution seam: wire translation, dedup, chunking, and validation."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Coroutine
from typing import Any

import httpx2
import pytest
from data_mcp import client_holder
from data_mcp.errors import UpstreamError
from data_mcp.resolution import (
    Ambiguous,
    InvalidName,
    NotFound,
    NotHeld,
    OutOfScope,
    RegionMember,
    ResolvedEntity,
    ResolvedRegion,
    UnheldRegionMember,
    resolve_entities,
)

from .net_guard import no_network  # noqa: F401 -- imported for its autouse fixture
from .test_client import _client

Handler = (
    Callable[[httpx2.Request], httpx2.Response]
    | Callable[[httpx2.Request], Coroutine[None, None, httpx2.Response]]
)


def _json_handler(payload: dict[str, Any]) -> Handler:
    def handler(_request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, json=payload)

    return handler


async def _resolve(handler: Handler, scope: dict[str, list[str]]) -> dict[str, dict[str, Any]]:
    async with _client(handler) as upstream:
        client_holder.set_client(upstream)
        return await resolve_entities(scope)


# The six wire statuses


async def test_resolved_status_maps_to_resolved_entity() -> None:
    payload = {
        "results": [
            {
                "status": "resolved",
                "entity": {"dcid": "country/FRA", "name": "France", "entity_type": "geo.country"},
                "confidence": 0.98,
            }
        ]
    }

    outcomes = await _resolve(_json_handler(payload), {"donor": ["France"]})

    outcome = outcomes["donor"]["France"]
    assert outcome == ResolvedEntity(dcid="country/FRA", name="France", entity_type="geo.country")


async def test_resolved_region_status_preserves_held_and_unheld_members() -> None:
    payload = {
        "results": [
            {
                "status": "resolved_region",
                "region": {"name": "Sub-Saharan Africa", "entity_type": "geo.subregion"},
                "members": [{"dcid": "country/KEN", "name": "Kenya"}],
                "members_not_held": [{"name": "Chad", "entity_type": "geo.country"}],
                "confidence": 0.8,
            }
        ]
    }

    outcomes = await _resolve(_json_handler(payload), {"recipient": ["Sub-Saharan Africa"]})

    outcome = outcomes["recipient"]["Sub-Saharan Africa"]
    assert isinstance(outcome, ResolvedRegion)
    assert outcome.name == "Sub-Saharan Africa"
    assert outcome.entity_type == "geo.subregion"
    assert outcome.members == [RegionMember(dcid="country/KEN", name="Kenya")]
    assert outcome.members_not_held == [UnheldRegionMember(name="Chad")]


async def test_ambiguous_status_preserves_candidates_and_parent_field() -> None:
    payload = {
        "results": [
            {
                "status": "ambiguous",
                "candidates": [
                    {
                        "dcid": "geoId/13",
                        "name": "Georgia",
                        "entity_type": "geo.country",
                        "parent": None,
                    },
                    {
                        "dcid": "country/GEO",
                        "name": "Georgia",
                        "entity_type": "geo.country",
                        "parent": "continent/AS",
                    },
                ],
                "truncated": False,
            }
        ]
    }

    outcomes = await _resolve(_json_handler(payload), {"recipient": ["Georgia"]})

    outcome = outcomes["recipient"]["Georgia"]
    assert isinstance(outcome, Ambiguous)
    assert outcome.truncated is False
    assert len(outcome.candidates) == 2
    assert outcome.candidates[0].parent is None
    assert outcome.candidates[1].parent == "continent/AS"


async def test_not_found_with_near_matches_signals_a_probable_misspelling() -> None:
    payload = {
        "results": [
            {
                "status": "not_found",
                "near_matches": [
                    {
                        "name": "France",
                        "entity_type": "geo.country",
                        "dcid": "country/FRA",
                        "confidence": 0.6,
                    }
                ],
            }
        ]
    }

    outcomes = await _resolve(_json_handler(payload), {"donor": ["Fance"]})

    outcome = outcomes["donor"]["Fance"]
    assert isinstance(outcome, NotFound)
    assert len(outcome.near_matches) == 1
    assert outcome.near_matches[0].name == "France"


async def test_not_found_with_no_near_matches_signals_nonsense() -> None:
    """An organisation name matches nothing and has nothing close to offer either."""
    payload = {"results": [{"status": "not_found", "near_matches": []}]}

    outcomes = await _resolve(_json_handler(payload), {"donor": ["World Health Organization"]})

    outcome = outcomes["donor"]["World Health Organization"]
    assert isinstance(outcome, NotFound)
    assert outcome.near_matches == []


async def test_not_held_individual_entity_is_not_a_region() -> None:
    payload = {"results": [{"status": "not_held", "name": "Andorra", "entity_type": "geo.country"}]}

    outcomes = await _resolve(_json_handler(payload), {"donor": ["Andorra"]})

    outcome = outcomes["donor"]["Andorra"]
    assert isinstance(outcome, NotHeld)
    assert outcome.is_region is False


async def test_not_held_region_with_all_members_unheld_is_a_region() -> None:
    payload = {
        "results": [{"status": "not_held", "name": "Micronesia", "entity_type": "geo.subregion"}]
    }

    outcomes = await _resolve(_json_handler(payload), {"recipient": ["Micronesia"]})

    outcome = outcomes["recipient"]["Micronesia"]
    assert isinstance(outcome, NotHeld)
    assert outcome.is_region is True


async def test_out_of_scope_with_a_held_dcid_preserves_it() -> None:
    payload = {
        "results": [
            {
                "status": "out_of_scope",
                "name": "World Bank",
                "entity_type": "geo.organization",
                "dcid": "dc/s/WorldBank",
            }
        ]
    }

    outcomes = await _resolve(_json_handler(payload), {"donor": ["World Bank"]})

    outcome = outcomes["donor"]["World Bank"]
    assert outcome == OutOfScope(
        name="World Bank", entity_type="geo.organization", dcid="dc/s/WorldBank"
    )


async def test_out_of_scope_with_no_held_dcid_is_none() -> None:
    payload = {
        "results": [
            {
                "status": "out_of_scope",
                "name": "African Union",
                "entity_type": "geo.continental_union",
            }
        ]
    }

    outcomes = await _resolve(_json_handler(payload), {"donor": ["African Union"]})

    outcome = outcomes["donor"]["African Union"]
    assert isinstance(outcome, OutOfScope)
    assert outcome.dcid is None


# Dedup


async def test_same_name_under_two_roles_is_resolved_once() -> None:
    requests: list[list[str]] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        texts = json.loads(request.content)["texts"]
        requests.append(texts)
        return httpx2.Response(
            200,
            json={
                "results": [
                    {
                        "status": "resolved",
                        "entity": {
                            "dcid": "country/FRA",
                            "name": "France",
                            "entity_type": "geo.country",
                        },
                        "confidence": 0.95,
                    }
                    for _ in texts
                ]
            },
        )

    outcomes = await _resolve(handler, {"donor": ["France"], "recipient": ["France"]})

    assert requests == [["France"]]
    assert outcomes["donor"]["France"] == outcomes["recipient"]["France"]


# Chunking and index alignment


async def test_chunking_preserves_index_alignment_across_the_25_boundary() -> None:
    requests: list[list[str]] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        texts = json.loads(request.content)["texts"]
        requests.append(texts)
        results = [
            {
                "status": "resolved",
                "entity": {"dcid": f"dcid:{text}", "name": text, "entity_type": "geo.country"},
                "confidence": 0.9,
            }
            for text in texts
        ]
        return httpx2.Response(200, json={"results": results})

    names = [f"Country{i}" for i in range(30)]

    outcomes = await _resolve(handler, {"role": names})

    assert len(requests) == 2
    assert len(requests[0]) == 25
    assert len(requests[1]) == 5
    for i, name in enumerate(names):
        outcome = outcomes["role"][name]
        assert isinstance(outcome, ResolvedEntity)
        # Each name's own dcid, not a neighbour's -- proves the zip stayed aligned across
        # the chunk boundary.
        assert outcome.dcid == f"dcid:Country{i}"
        assert outcome.name == name


# Local validation


async def test_empty_after_strip_name_needs_no_upstream_call() -> None:
    calls = 0

    def handler(_request: httpx2.Request) -> httpx2.Response:
        nonlocal calls
        calls += 1
        return httpx2.Response(200, json={"results": []})

    outcomes = await _resolve(handler, {"donor": ["   "]})

    assert calls == 0
    outcome = outcomes["donor"]["   "]
    assert isinstance(outcome, InvalidName)
    assert outcome.name == "   "


async def test_overlength_name_needs_no_upstream_call() -> None:
    calls = 0
    name = "x" * 201

    def handler(_request: httpx2.Request) -> httpx2.Response:
        nonlocal calls
        calls += 1
        return httpx2.Response(200, json={"results": []})

    outcomes = await _resolve(handler, {"donor": [name]})

    assert calls == 0
    outcome = outcomes["donor"][name]
    assert isinstance(outcome, InvalidName)


async def test_valid_name_alongside_an_invalid_one_still_reaches_the_door() -> None:
    """One bad name in a role's list shouldn't stop the rest of the batch from resolving."""
    requests: list[list[str]] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        texts = json.loads(request.content)["texts"]
        requests.append(texts)
        return httpx2.Response(
            200,
            json={
                "results": [
                    {
                        "status": "resolved",
                        "entity": {
                            "dcid": "country/FRA",
                            "name": "France",
                            "entity_type": "geo.country",
                        },
                        "confidence": 0.95,
                    }
                ]
            },
        )

    outcomes = await _resolve(handler, {"donor": ["France", ""]})

    assert requests == [["France"]]
    assert isinstance(outcomes["donor"]["France"], ResolvedEntity)
    assert isinstance(outcomes["donor"][""], InvalidName)


# UpstreamError propagation


async def test_upstream_error_propagates_rather_than_being_swallowed() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ConnectError("boom", request=request)

    async with _client(handler) as upstream:
        client_holder.set_client(upstream)
        with pytest.raises(UpstreamError):
            await resolve_entities({"donor": ["France"]})


# Concurrent chunk dispatch


async def test_chunks_are_dispatched_concurrently_not_serially() -> None:
    """Chunk 1's handler blocks until chunk 2's handler has run and unblocked it.

    Under a serial `for chunk in chunks: await client.request(...)` loop, chunk 2 is
    never sent until chunk 1's request completes, so chunk 1 would wait forever on an
    event only chunk 2 can set -- deadlock. `asyncio.wait_for` bounds the whole call so
    a regression fails fast instead of hanging the suite.
    """
    chunk_two_started = asyncio.Event()

    async def handler(request: httpx2.Request) -> httpx2.Response:
        texts = json.loads(request.content)["texts"]
        if len(texts) == 25:
            await chunk_two_started.wait()
        else:
            chunk_two_started.set()
        results = [
            {
                "status": "resolved",
                "entity": {"dcid": f"dcid:{text}", "name": text, "entity_type": "geo.country"},
                "confidence": 0.9,
            }
            for text in texts
        ]
        return httpx2.Response(200, json={"results": results})

    names = [f"Country{i}" for i in range(30)]

    outcomes = await asyncio.wait_for(_resolve(handler, {"role": names}), timeout=5.0)

    assert len(outcomes["role"]) == 30
    for name in names:
        assert isinstance(outcomes["role"][name], ResolvedEntity)


async def test_one_failing_chunk_raises_while_the_other_still_completes() -> None:
    """One chunk's upstream failure surfaces as `UpstreamError`, not swallowed.

    Both chunks are in flight under concurrent dispatch, so the successful chunk's
    request runs to completion in the background rather than being left orphaned --
    there is nothing here for pytest to report as an unretrieved task exception.
    """

    def handler(request: httpx2.Request) -> httpx2.Response:
        texts = json.loads(request.content)["texts"]
        if len(texts) == 25:
            raise httpx2.ConnectError("boom", request=request)
        results = [
            {
                "status": "resolved",
                "entity": {"dcid": f"dcid:{text}", "name": text, "entity_type": "geo.country"},
                "confidence": 0.9,
            }
            for text in texts
        ]
        return httpx2.Response(200, json={"results": results})

    names = [f"Country{i}" for i in range(30)]

    with pytest.raises(UpstreamError):
        await asyncio.wait_for(_resolve(handler, {"role": names}), timeout=5.0)
