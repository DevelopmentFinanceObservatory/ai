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
    AmbiguousCandidate,
    InvalidName,
    NotFound,
    NotHeld,
    RegionMember,
    ResolvedEntity,
    ResolvedRegion,
    resolve_entities,
)
from pydantic import ValidationError

from .net_guard import no_network  # noqa: F401 -- imported for its autouse fixture
from .resolve_results import resolve_result, resolved_result
from .test_client import _client

Handler = (
    Callable[[httpx2.Request], httpx2.Response]
    | Callable[[httpx2.Request], Coroutine[None, None, httpx2.Response]]
)

_G7_MEMBERS = [
    {"ref": "country/CAN", "label": "Canada"},
    {"ref": "country/DEU", "label": "Germany"},
    {"ref": "country/FRA", "label": "France"},
    {"ref": "country/GBR", "label": "United Kingdom"},
    {"ref": "country/ITA", "label": "Italy"},
    {"ref": "country/JPN", "label": "Japan"},
    {"ref": "country/USA", "label": "United States"},
]


def _json_handler(*results: dict[str, Any]) -> Handler:
    def handler(_request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, json={"results": list(results)})

    return handler


def _echo_handler(requests: list[dict[str, Any]] | None = None) -> Handler:
    """Resolve every text to a country named by the text itself."""

    def handler(request: httpx2.Request) -> httpx2.Response:
        body = json.loads(request.content)
        if requests is not None:
            requests.append(body)
        results = [resolved_result(text, f"dcid:{text}") for text in body["texts"]]
        return httpx2.Response(200, json={"results": results})

    return handler


async def _resolve(handler: Handler, scope: dict[str, list[str]]) -> dict[str, dict[str, Any]]:
    async with _client(handler) as upstream:
        client_holder.set_client(upstream)
        return await resolve_entities(scope)


# The request


async def test_request_posts_the_texts_with_members_expanded() -> None:
    requests: list[dict[str, Any]] = []

    await _resolve(_echo_handler(requests), {"donor": ["France"]})

    assert requests == [{"texts": ["France"], "expand": ["members"]}]


# The four wire statuses


async def test_resolved_status_maps_to_resolved_entity() -> None:
    outcomes = await _resolve(
        _json_handler(resolved_result("France", "country/FRA")), {"donor": ["France"]}
    )

    assert outcomes["donor"]["France"] == ResolvedEntity(
        dcid="country/FRA", name="France", entity_type="country"
    )


async def test_resolved_organisation_maps_to_resolved_entity() -> None:
    result = resolved_result("World Bank", "WorldBank", kind="organisation")

    outcomes = await _resolve(_json_handler(result), {"provider": ["World Bank"]})

    assert outcomes["provider"]["World Bank"] == ResolvedEntity(
        dcid="WorldBank", name="World Bank", entity_type="organisation"
    )


async def test_resolved_group_maps_to_its_inline_members() -> None:
    result = resolve_result(
        "G7",
        "resolved",
        entity={"ref": "GroupOfSeven", "label": "Group of Seven", "kind": "group"},
        matched={"text": "G7", "as": "alias"},
        reading="entity",
        projection="countries",
        member_count=7,
        members=_G7_MEMBERS,
    )

    outcomes = await _resolve(_json_handler(result), {"observationAbout": ["G7"]})

    assert outcomes["observationAbout"]["G7"] == ResolvedRegion(
        name="Group of Seven",
        entity_type="group",
        members=[RegionMember(dcid=m["ref"], name=m["label"]) for m in _G7_MEMBERS],
    )


async def test_resolved_region_maps_its_members_and_names_an_unlabelled_one_by_ref() -> None:
    result = resolved_result(
        "North America",
        "northamerica",
        kind="continent",
        projection="countries",
        member_count=2,
        members=[{"ref": "country/CAN", "label": "Canada"}, {"ref": "country/XKX", "label": None}],
    )

    outcomes = await _resolve(_json_handler(result), {"observationAbout": ["North America"]})

    assert outcomes["observationAbout"]["North America"] == ResolvedRegion(
        name="North America",
        entity_type="continent",
        members=[
            RegionMember(dcid="country/CAN", name="Canada"),
            RegionMember(dcid="country/XKX", name="country/XKX"),
        ],
    )


async def test_a_region_the_membership_policy_does_not_hold_resolves_as_an_entity() -> None:
    # Its `projection`, `member_count` and `members` are `null`.
    result = resolved_result("Polynesia", "polynesia", kind="subregion")

    outcomes = await _resolve(_json_handler(result), {"observationAbout": ["Polynesia"]})

    assert outcomes["observationAbout"]["Polynesia"] == ResolvedEntity(
        dcid="polynesia", name="Polynesia", entity_type="subregion"
    )


@pytest.mark.parametrize(
    "members",
    [
        "/api/graph/node/GroupOfSeven/members?relation=members&projection=countries",
        _G7_MEMBERS[:6],
    ],
    ids=["a link", "fewer than member_count"],
)
async def test_a_group_without_all_its_members_inline_raises(members: object) -> None:
    result = resolved_result("G7", "GroupOfSeven", kind="group", member_count=7, members=members)

    with pytest.raises(ValueError, match="members inline"):
        await _resolve(_json_handler(result), {"observationAbout": ["G7"]})


async def test_ambiguous_status_preserves_candidates() -> None:
    result = resolve_result(
        "Congo",
        "ambiguous",
        candidates=[
            {"ref": "country/COD", "label": "Congo [DRC]", "kind": "country"},
            {"ref": "country/COG", "label": "Congo [Republic]", "kind": "country"},
        ],
    )

    outcomes = await _resolve(_json_handler(result), {"recipient": ["Congo"]})

    assert outcomes["recipient"]["Congo"] == Ambiguous(
        candidates=[
            AmbiguousCandidate(dcid="country/COD", name="Congo [DRC]", entity_type="country"),
            AmbiguousCandidate(dcid="country/COG", name="Congo [Republic]", entity_type="country"),
        ]
    )


async def test_not_found_with_near_matches_signals_a_probable_misspelling() -> None:
    result = resolve_result(
        "Lagos",
        "not_found",
        near_matches=[{"ref": "country/LAO", "label": "Laos", "kind": "country"}],
    )

    outcomes = await _resolve(_json_handler(result), {"donor": ["Lagos"]})

    outcome = outcomes["donor"]["Lagos"]
    assert isinstance(outcome, NotFound)
    assert [m.name for m in outcome.near_matches] == ["Laos"]


async def test_not_found_with_no_near_matches_signals_nonsense() -> None:
    result = resolve_result("Atlantis Prime", "not_found", near_matches=[])

    outcomes = await _resolve(_json_handler(result), {"donor": ["Atlantis Prime"]})

    assert outcomes["donor"]["Atlantis Prime"] == NotFound(near_matches=[])


async def test_not_held_maps_to_not_held() -> None:
    result = resolve_result(
        "Andorra", "not_held", entity={"ref": "country/AND", "label": "Andorra", "kind": None}
    )

    outcomes = await _resolve(_json_handler(result), {"donor": ["Andorra"]})

    assert outcomes["donor"]["Andorra"] == NotHeld(name="Andorra")


@pytest.mark.parametrize(
    "result",
    [
        {"input": "France", "status": "out_of_scope"},
        resolve_result("France", "resolved_region"),
    ],
    ids=["retired status", "unknown status"],
)
async def test_an_unrecognised_status_raises_a_validation_error(result: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        await _resolve(_json_handler(result), {"donor": ["France"]})


async def test_a_result_missing_the_field_its_status_sets_raises() -> None:
    with pytest.raises(ValueError, match="entity"):
        await _resolve(_json_handler(resolve_result("France", "resolved")), {"donor": ["France"]})


# Dedup


async def test_same_name_under_two_roles_is_resolved_once() -> None:
    requests: list[dict[str, Any]] = []

    outcomes = await _resolve(
        _echo_handler(requests), {"donor": ["France"], "recipient": ["France"]}
    )

    assert [body["texts"] for body in requests] == [["France"]]
    assert outcomes["donor"]["France"] == outcomes["recipient"]["France"]


# Chunking and index alignment


async def test_chunking_preserves_index_alignment_across_the_25_boundary() -> None:
    requests: list[dict[str, Any]] = []
    names = [f"Country{i}" for i in range(30)]

    outcomes = await _resolve(_echo_handler(requests), {"role": names})

    assert [len(body["texts"]) for body in requests] == [25, 5]
    for i, name in enumerate(names):
        outcome = outcomes["role"][name]
        assert isinstance(outcome, ResolvedEntity)
        # Each name's own dcid, not a neighbour's -- proves the zip stayed aligned across
        # the chunk boundary.
        assert outcome.dcid == f"dcid:Country{i}"
        assert outcome.name == name


# Local validation


async def test_empty_after_strip_name_needs_no_upstream_call() -> None:
    requests: list[dict[str, Any]] = []

    outcomes = await _resolve(_echo_handler(requests), {"donor": ["   "]})

    assert requests == []
    outcome = outcomes["donor"]["   "]
    assert isinstance(outcome, InvalidName)
    assert outcome.name == "   "


async def test_overlength_name_needs_no_upstream_call() -> None:
    requests: list[dict[str, Any]] = []
    name = "x" * 201

    outcomes = await _resolve(_echo_handler(requests), {"donor": [name]})

    assert requests == []
    assert isinstance(outcomes["donor"][name], InvalidName)


async def test_valid_name_alongside_an_invalid_one_still_reaches_the_door() -> None:
    """One bad name in a role's list shouldn't stop the rest of the batch from resolving."""
    requests: list[dict[str, Any]] = []

    outcomes = await _resolve(_echo_handler(requests), {"donor": ["France", ""]})

    assert [body["texts"] for body in requests] == [["France"]]
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
    echo = _echo_handler()

    async def handler(request: httpx2.Request) -> httpx2.Response:
        if len(json.loads(request.content)["texts"]) == 25:
            await chunk_two_started.wait()
        else:
            chunk_two_started.set()
        response = echo(request)
        assert isinstance(response, httpx2.Response)
        return response

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
    echo = _echo_handler()

    def handler(request: httpx2.Request) -> httpx2.Response:
        if len(json.loads(request.content)["texts"]) == 25:
            raise httpx2.ConnectError("boom", request=request)
        response = echo(request)
        assert isinstance(response, httpx2.Response)
        return response

    names = [f"Country{i}" for i in range(30)]

    with pytest.raises(UpstreamError):
        await asyncio.wait_for(_resolve(handler, {"role": names}), timeout=5.0)
