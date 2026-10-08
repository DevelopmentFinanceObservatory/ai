"""data-mcp's `resolve_definition`, called in process, over a definition store and a Search API a test chooses.

`harness.graph`'s `seed` node resolves each seed's identity again through
data-mcp's `resolve_definition`, and `translator.discovery` resolves a seed's
copy over a plan's years the same way, on the data-mcp `get_data` reaches.
`SearchStub` stands in for the Search API's resolve route that
`resolve_definition` posts to. A test that gives data-mcp a store of its
own, apart from the one the harness process holds, stands in for a
deployment that runs data-mcp as its own service.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from typing import Any

import httpx2
from compiler.surface import ProgressCallback, ToolCallResult, ToolSchema
from data_mcp import client_holder
from data_mcp.client import UpstreamClient
from data_mcp.definition_store import DefinitionStore, get_store, set_store
from data_mcp.definition_store.identity import canonicalize, spec_id_for
from data_mcp.server import mcp as data_mcp
from data_mcp.settings import Settings as DataMcpSettings
from fastmcp.exceptions import ToolError
from mcp.types import TextContent
from planner.workflow import ToolId
from pydantic import ValidationError


def served_block(seed_or_block: Mapping[str, Any]) -> dict[str, Any]:
    """The Spec block of a `harness.wire.Seed` as the browser posts it, or a Spec block itself."""
    return dict(seed_or_block.get("spec", seed_or_block))


class SearchStub:
    """The Search API's resolve route, answering each `dataset_resubmit` from recorded Spec blocks.

    `blocks` are the Spec blocks Search serves, each for the identity its
    definition carries. A posted identity whose spec_id one of them names is
    answered with a one-part `definite` envelope carrying it; any other with
    a 422 `request_invalid`, as Search refuses an identity it cannot
    resolve. `answer` replaces the envelope for a spec_id, and `refuse`
    answers a spec_id with that 422 problem body instead. `delay_s` holds
    each answer for that long, by spec_id. `posted` records each request
    body, `spans` when each was received and answered (`time.monotonic`),
    and `peak` the most requests held at once.
    """

    def __init__(
        self,
        blocks: Iterable[Mapping[str, Any]] = (),
        *,
        answer: Mapping[str, Mapping[str, Any]] | None = None,
        refuse: Mapping[str, Mapping[str, Any]] | None = None,
        delay_s: Mapping[str, float] | None = None,
    ) -> None:
        self._blocks = {block["spec_id"]: block for block in map(served_block, blocks)}
        self._answer = dict(answer or {})
        self._refuse = dict(refuse or {})
        self._delay_s = dict(delay_s or {})
        self._held = 0
        self.peak = 0
        self.posted: list[dict[str, Any]] = []
        self.spans: list[tuple[float, float]] = []

    async def __call__(self, request: httpx2.Request) -> httpx2.Response:
        assert request.url.path == "/api/search/resolve", request.url
        body = json.loads(request.content)
        self.posted.append(body)
        (entry,) = body["input"]["entries"]
        spec_id = spec_id_for(canonicalize(entry["identity"]))
        received = time.monotonic()
        self._held += 1
        self.peak = max(self.peak, self._held)
        try:
            await asyncio.sleep(self._delay_s.get(spec_id, 0.0))
        finally:
            self._held -= 1
            self.spans.append((received, time.monotonic()))
        if spec_id in self._refuse:
            return httpx2.Response(422, json=self._refuse[spec_id])
        if spec_id in self._answer:
            return httpx2.Response(200, json=self._answer[spec_id])
        block = self._blocks.get(spec_id)
        if block is None:
            return httpx2.Response(
                422,
                json={
                    "code": "request_invalid",
                    "title": "refused",
                    "detail": f"no Spec resolves from the identity of {spec_id}",
                    "retryable": False,
                },
            )
        return httpx2.Response(
            200,
            json={
                "schema_version": 1,
                "status": "definite",
                "parts": [{"status": "definite", "spec": block}],
            },
        )


class GroupsSearchStub(SearchStub):
    """`SearchStub` answering a resubmit as dfo-apis#1222 does: a block's
    `resolved.entity_groups` is served only when the request posts groups beside
    the identity, and the members alone otherwise."""

    async def __call__(self, request: httpx2.Request) -> httpx2.Response:
        response = await super().__call__(request)
        (entry,) = json.loads(request.content)["input"]["entries"]
        if "entity_groups" in entry or response.status_code != 200:
            return response
        envelope = response.json()
        envelope["parts"][0]["spec"]["resolved"].pop("entity_groups", None)
        return httpx2.Response(200, json=envelope)


@contextmanager
def search_api(search: SearchStub) -> Iterator[None]:
    """`search` as the Search API data-mcp's tools post to while the block runs.

    For a test that reaches data-mcp's `resolve_definition` through a surface
    of its own; `DataMcpDefinitions` installs its own.
    """
    client_holder.set_client(
        UpstreamClient(settings=DataMcpSettings(), transport=httpx2.MockTransport(search))
    )
    try:
        yield
    finally:
        client_holder.set_client(None)


class DataMcpDefinitions:
    """A `ToolSurface` that calls data-mcp's `resolve_definition` in process.

    Each call runs the tool through data-mcp's server object, with its
    middleware and argument checks and without a client session, so calls
    run side by side as the seed node makes them. `search` answers the
    upstream Search API, a `SearchStub` over `served` when not given.
    `store` is data-mcp's definition store. `None` is whichever store the
    process holds, as when data-mcp runs in the harness's own process. The
    upstream client and `store` are installed while any call runs, and the
    process holds what it held before once the last one ends. `calls`
    records each call's arguments.
    """

    def __init__(
        self,
        store: DefinitionStore | None = None,
        *,
        served: Iterable[Mapping[str, Any]] = (),
        search: SearchStub | None = None,
    ) -> None:
        self._store = store
        self.search = search or SearchStub(served)
        self._upstream = UpstreamClient(
            settings=DataMcpSettings(), transport=httpx2.MockTransport(self.search)
        )
        self._running = 0
        self._held_store: DefinitionStore | None = None
        self.calls: list[dict[str, Any]] = []

    async def list_tools(self) -> list[ToolSchema]:
        return []

    @contextmanager
    def _installed(self) -> Iterator[None]:
        if self._running == 0:
            client_holder.set_client(self._upstream)
            if self._store is not None:
                self._held_store = get_store()
                set_store(self._store)
        self._running += 1
        try:
            yield
        finally:
            self._running -= 1
            if self._running == 0:
                client_holder.set_client(None)
                if self._store is not None:
                    set_store(self._held_store)

    async def call_tool(
        self,
        *,
        tool: ToolId,
        arguments: Mapping[str, object],
        on_progress: ProgressCallback | None = None,
    ) -> ToolCallResult:
        del on_progress
        assert tool.server == "data-mcp", tool
        assert tool.name == "resolve_definition", tool
        self.calls.append(dict(arguments))
        with self._installed():
            try:
                result = await data_mcp.call_tool(tool.name, dict(arguments))
            except (ToolError, ValidationError) as exc:
                return ToolCallResult(is_error=True, text=str(exc))
        structured = result.structured_content
        return ToolCallResult(
            is_error=bool(result.is_error),
            structured_content=dict(structured) if isinstance(structured, Mapping) else None,
            text="\n".join(
                block.text for block in result.content if isinstance(block, TextContent)
            ),
            meta=dict(result.meta) if result.meta else None,
        )
