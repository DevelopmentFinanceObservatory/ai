"""Tests for `resolve_definition`: one identity resolved again through Search's `dataset_resubmit` door."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import httpx2
from data_mcp import client_holder
from data_mcp.definition_store import (
    DefinitionStore,
    InMemoryDefinitionStore,
    set_store,
)
from data_mcp.search_result import SEARCH_ENVELOPE_META_KEY
from data_mcp.server import mcp
from fastmcp import Client
from fastmcp.client.client import CallToolResult

from .test_client import _client, _problem_body
from .test_search_tool import Handler

# The dev Search API's answer for "population of Guatemala": one definite part.
_RECORDED = json.loads(
    (Path(__file__).parent / "fixtures" / "resolve_population_guatemala.json").read_text()
)
_SPEC = _RECORDED["parts"][0]["spec"]
_STAT_VARS = [item["ref"] for item in _SPEC["resolved"]["stat_vars"]["items"]]


def _arguments(**overrides: Any) -> dict[str, Any]:
    return {
        "spec_id": _SPEC["spec_id"],
        "identity": _SPEC["definition"]["identity"],
        "stat_vars": _STAT_VARS,
        **overrides,
    }


def _recorded_answer(posted: list[dict[str, Any]], envelope: dict[str, Any] = _RECORDED) -> Handler:
    def handler(request: httpx2.Request) -> httpx2.Response:
        assert request.url.path == "/api/search/resolve"
        posted.append(json.loads(request.content))
        return httpx2.Response(200, json=envelope)

    return handler


async def _call(
    handler: Handler, arguments: dict[str, Any], store: DefinitionStore
) -> CallToolResult:
    async with _client(handler) as upstream:
        client_holder.set_client(upstream)
        async with Client(mcp) as client:
            # The lifespan sets a store of its own on entry; the tool reads it at call time.
            set_store(store)
            return await client.call_tool("resolve_definition", arguments, raise_on_error=False)


async def test_an_identity_is_posted_as_its_own_base_and_search_s_answer_is_returned_and_stored() -> (
    None
):
    posted: list[dict[str, Any]] = []
    store = InMemoryDefinitionStore()

    result = await _call(_recorded_answer(posted), _arguments(), store)

    assert result.is_error is False
    assert result.structured_content == {"spec_id": _SPEC["spec_id"], "status": "definite"}
    assert result.meta is not None
    assert result.meta[SEARCH_ENVELOPE_META_KEY] == _RECORDED
    (body,) = posted
    assert body == {
        "schema_version": 1,
        "input": {
            "kind": "dataset_resubmit",
            "entries": [
                {
                    "entry_kind": "identity",
                    "variable_text": _SPEC["spec_id"],
                    "identity": _SPEC["definition"]["identity"],
                    "base_spec_id": _SPEC["spec_id"],
                    "base_identity": _SPEC["definition"]["identity"],
                    "resolved_stat_var_dcids": _STAT_VARS,
                }
            ],
        },
        "response_options": {"include_sentence": True},
    }
    held = await store.get(_SPEC["spec_id"], schema_version=1)
    assert held is not None
    assert json.loads(held.canonical_bytes) == _SPEC["definition"]["identity"]


async def test_no_statvars_posts_none() -> None:
    posted: list[dict[str, Any]] = []

    await _call(_recorded_answer(posted), _arguments(stat_vars=None), InMemoryDefinitionStore())

    (body,) = posted
    assert "resolved_stat_var_dcids" not in body["input"]["entries"][0]


async def test_an_identity_that_does_not_hash_to_its_spec_id_is_refused_before_search() -> None:
    posted: list[dict[str, Any]] = []
    identity = {**_SPEC["definition"]["identity"], "forged": True}
    store = InMemoryDefinitionStore()

    result = await _call(_recorded_answer(posted), _arguments(identity=identity), store)

    assert result.is_error is True
    assert result.structured_content is not None
    assert result.structured_content["code"] == "request_invalid"
    assert posted == []
    assert await store.get(_SPEC["spec_id"], schema_version=1) is None


async def test_a_refusal_from_search_passes_through_and_stores_nothing() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(422, content=_problem_body(code="invalid_role_ref", retryable=False))

    store = InMemoryDefinitionStore()

    result = await _call(handler, _arguments(), store)

    assert result.is_error is True
    assert result.structured_content is not None
    assert result.structured_content["retry_mode"] != "retry_after_delay"
    assert await store.get(_SPEC["spec_id"], schema_version=1) is None


async def test_an_answer_whose_definition_does_not_hash_is_refused_and_stores_nothing() -> None:
    envelope = copy.deepcopy(_RECORDED)
    envelope["parts"][0]["spec"]["definition"]["identity"]["forged"] = True
    store = InMemoryDefinitionStore()

    result = await _call(_recorded_answer([], envelope), _arguments(), store)

    assert result.is_error is True
    assert result.structured_content is not None
    assert result.structured_content["code"] == "capability_unavailable"
    assert await store.get(_SPEC["spec_id"], schema_version=1) is None
