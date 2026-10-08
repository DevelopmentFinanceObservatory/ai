"""`POST /api/studio/refresh` against the real services: a card drawn on a
filter-then-growth frame is refreshed after its dataset moves.

The verbs run through the mounted analysis-mcp app and the root is minted through
data-mcp's `mint_recorded_root` over a canned mint, as in
`test_recipe_replay_integration.py`. dfo-apis is
`live_refresh_support.RecordedApis`, whose answers on the live-refresh routes are
the recorded ones, and the store is the in-memory one. No socket opens.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Callable, Mapping
from typing import Any

import httpx2
import pytest
from common.materialisation import FrameRef, get_store
from common.recipe import build_recipe
from common.row_budget import ROW_BUDGET_RETRY_AFTER_S, RowBudget, row_budget
from compiler.surface import McpSurface, ToolCallResult
from data_mcp import client_holder, frame_materialisation
from data_mcp.client import UpstreamClient
from data_mcp.settings import Settings as DataMcpSettings
from fastapi import FastAPI
from harness.live_refresh import LiveRefresh
from harness.recipe_replay import Pending, Refused, Replayed, replay_recipe
from harness.server import create_app
from harness.settings import ANALYSIS_MCP_MOUNT_PATH, DATA_MCP_MOUNT_PATH, Settings
from planner.workflow import ToolId

from .live_refresh_support import CARD, LIVE_PRINCIPAL, RECORDED_RENDER, RecordedApis
from .replay_support import FIRST, SECOND, SECOND_VALUES, SPEC, Upstream, pair

_KEPT = ["dcid:FRA", "dcid:GBR"]
_REDRAWN = {"status": "redrawn", "render": RECORDED_RENDER}


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in ("DATA_MCP_URL", "ANALYSIS_MCP_URL", "PRESENTATION_MCP_URL", "DATABASE_URL", "PORT"):
        monkeypatch.delenv(var, raising=False)


@pytest.fixture(autouse=True)
async def _reset_data_mcp_client() -> AsyncIterator[None]:
    """`data_mcp.server`'s lifespan leaves `client_holder` alone, so a client set
    here would leak into later tests across the whole workspace suite."""
    yield
    client_holder.set_client(None)


class Harness:
    """The served app with its mounted tools, the canned mint, and dfo-apis.

    `service` builds a `LiveRefresh` over them as `build_live_refresh` wires the
    served one, counting the replays and the verb calls. Each call is a harness
    instance of its own: it shares only the store and dfo-apis with the others.
    """

    def __init__(self, app: FastAPI, surface: McpSurface, upstream: Upstream) -> None:
        self.app = app
        self.surface = surface
        self.upstream = upstream
        self.apis = RecordedApis()
        self.assembler = self.apis.assembler()
        self.verbs: list[str] = []
        self.replays = 0

    async def call(self, server: str, name: str, arguments: Mapping[str, object]) -> dict[str, Any]:
        result = await self.surface.call_tool(
            tool=ToolId(server=server, name=name), arguments=arguments
        )
        assert result.is_error is False, result.text
        assert result.structured_content is not None
        return result.structured_content

    async def _analysis(self, *, name: str, arguments: Mapping[str, object]) -> ToolCallResult:
        self.verbs.append(name)
        return await self.surface.call_tool(
            tool=ToolId(server="analysis-mcp", name=name), arguments=arguments
        )

    async def _replay(
        self, recipe: Mapping[str, Any], **arguments: Any
    ) -> Replayed | Pending | Refused:
        self.replays += 1
        return await replay_recipe(recipe, **arguments)

    def service(self, **overrides: Any) -> LiveRefresh:
        options: dict[str, Any] = {
            "apis": self.apis.client(),
            "call_tool": self._analysis,
            "publish": self.assembler.publish_frame,
            "row_budget": RowBudget(max_rows=100_000),
            "deadline_s": 45.0,
            "requests_per_s": 100.0,
            "burst": 100,
            "max_in_flight": 100,
            "replay": self._replay,
        }
        return LiveRefresh(**(options | overrides))

    async def refresh(
        self, service: LiveRefresh, *, token: str = LIVE_PRINCIPAL
    ) -> httpx2.Response:
        self.app.state.live_refresh = service
        async with httpx2.AsyncClient(
            transport=httpx2.ASGITransport(app=self.app), base_url="http://harness"
        ) as client:
            return await client.post(
                "/api/studio/refresh",
                json={"component_id": CARD, "revision": 1},
                headers={"Authorization": f"Bearer {token}"},
            )

    async def draw_card(self) -> FrameRef:
        """Compute growth over two of the dataset's reporters, publish the frame
        with its recipe, and store the card on it. Returns the frame."""
        root = (await self.call("data-mcp", "get_data", {"spec_id": SPEC}))["frame_ref"]
        kept = await self.call(
            "analysis-mcp",
            "filter",
            {
                "frame_ref": root,
                "predicates": [{"kind": "category_match", "column": "reporter", "values": _KEPT}],
            },
        )
        grown = await self.call(
            "analysis-mcp",
            "growth",
            {
                "frame_ref": {key: kept[key] for key in ("spec_id", "content_version")},
                "window_start": "2019",
                "window_end": "2022",
                "entity_by": ["reporter"],
            },
        )
        frame = FrameRef(spec_id=grown["spec_id"], content_version=grown["content_version"])
        recipe = await build_recipe(frame, store=get_store())
        assert await self.assembler.publish_frame(
            spec_id=frame.spec_id, content_version=frame.content_version, recipe=recipe
        )
        self.apis.store_card(frame)
        self.apis.heads[SPEC] = FIRST
        return frame

    def move_dataset(self) -> None:
        self.upstream.move(SECOND_VALUES, to=SECOND)
        self.apis.heads[SPEC] = SECOND

    def growth_of(self, frame: tuple[str, str]) -> dict[str, float]:
        return {row["reporter"]: row["value"] for row in self.apis.frames[frame]}


@pytest.fixture
async def harness(
    monkeypatch: pytest.MonkeyPatch,
    running_lifespan: Callable[..., Any],
    mounted_mcp_transport: Callable[..., Any],
) -> AsyncIterator[Harness]:
    upstream = Upstream()
    monkeypatch.setattr(frame_materialisation, "_fetch_artifact_bytes", upstream.fetch)
    client_holder.set_client(
        UpstreamClient(settings=DataMcpSettings(), transport=httpx2.MockTransport(upstream.handler))
    )
    app = create_app(
        settings=Settings(mcp_api_token="test-token", live_refresh_principal=LIVE_PRINCIPAL)
    )
    async with running_lifespan(app):
        surface = McpSurface(
            sources={
                "data-mcp": mounted_mcp_transport(
                    app, f"{DATA_MCP_MOUNT_PATH}/mcp", token="test-token"
                ),
                "analysis-mcp": mounted_mcp_transport(
                    app, f"{ANALYSIS_MCP_MOUNT_PATH}/mcp", token="test-token"
                ),
            }
        )
        made = Harness(app, surface, upstream)
        yield made
        await made.assembler.aclose()
        await surface.aclose()


async def test_a_card_on_current_data_stands_and_one_with_a_rebuilding_root_waits(
    harness: Harness,
) -> None:
    await harness.draw_card()
    mints = harness.upstream.mints

    current = await harness.refresh(harness.service())
    harness.apis.rebuilding[SPEC] = 1.5
    pending = await harness.refresh(harness.service())

    assert (current.status_code, current.json()) == (200, {"status": "current"})
    assert (pending.status_code, pending.json()) == (
        200,
        {"status": "pending", "retry_after_s": 1.5},
    )
    assert (harness.replays, harness.verbs, harness.upstream.mints) == (0, [], mints)
    assert harness.apis.calls("/recipes/replays/") == []
    assert harness.apis.calls("/render") == []


async def test_a_moved_card_is_replayed_once_and_later_requests_render_the_stored_frame(
    harness: Harness,
) -> None:
    original = await harness.draw_card()
    harness.move_dataset()

    first = await harness.refresh(harness.service())

    assert first.status_code == 200
    # The render is dfo-apis' recorded answer, member for member and in its order.
    assert first.content == json.dumps(_REDRAWN, ensure_ascii=False, separators=(",", ":")).encode()
    (target,) = harness.apis.rendered_on()
    assert target != pair(original)
    stored, replayed = harness.apis.recipes[pair(original)], harness.apis.recipes[target]
    assert replayed["normal_form_hash"] == stored["normal_form_hash"]
    assert replayed["roots"][0]["dataset"] == {"spec_id": SPEC, "content_version": SECOND}
    # 2019 to 2022 is 150 to 180 for France and 300 to 330 for Britain at the second version.
    assert harness.growth_of(target) == pytest.approx({"dcid:FRA": 0.2, "dcid:GBR": 0.1})
    assert (harness.replays, harness.verbs) == (1, ["filter", "growth"])

    # The same instance and another one find the frame under its replay key.
    mints, publishes = harness.upstream.mints, len(harness.apis.published())
    again = await harness.refresh(harness.service())

    assert again.json() == _REDRAWN
    assert harness.apis.rendered_on() == [target, target]
    assert (harness.replays, harness.verbs) == (1, ["filter", "growth"])
    assert (harness.upstream.mints, len(harness.apis.published())) == (mints, publishes)

    # An instance that started before the first published replays to the same
    # pairs: every step is in the store, and dfo-apis records nothing new.
    del harness.apis.recipes[target]
    late = await harness.refresh(harness.service())

    assert late.json() == _REDRAWN
    assert harness.apis.rendered_on() == [target, target, target]
    assert (harness.replays, harness.verbs) == (2, ["filter", "growth"])
    assert harness.apis.recipes[target]["replay_key"] == replayed["replay_key"]


async def test_concurrent_requests_for_a_moved_card_replay_it_once(harness: Harness) -> None:
    await harness.draw_card()
    harness.move_dataset()
    service = harness.service()

    responses = await asyncio.gather(*(harness.refresh(service) for _ in range(5)))

    assert [response.json() for response in responses] == [_REDRAWN] * 5
    assert (harness.replays, harness.verbs) == (1, ["filter", "growth"])
    assert len(harness.apis.rendered_on()) == 1


async def test_a_deadline_too_short_to_mint_waits_and_the_next_call_redraws(
    harness: Harness,
) -> None:
    await harness.draw_card()
    harness.move_dataset()
    mints = harness.upstream.mints

    short = await harness.refresh(harness.service(deadline_s=1.0))

    assert (short.status_code, short.json()["status"]) == (200, "pending")
    assert (harness.verbs, harness.upstream.mints) == ([], mints)
    assert harness.apis.calls("/render") == []

    assert (await harness.refresh(harness.service())).json() == _REDRAWN


async def test_a_full_refresh_row_budget_waits_and_leaves_the_shared_budget_alone(
    harness: Harness,
) -> None:
    await harness.draw_card()
    harness.move_dataset()
    publishes = len(harness.apis.published())

    # Another replay holds every row of the refresh budget.
    budget = RowBudget(max_rows=1_000)
    with budget.claim(budget.max_rows):
        response = await harness.refresh(harness.service(row_budget=budget))

    assert (response.status_code, response.json()) == (
        200,
        {"status": "pending", "retry_after_s": ROW_BUDGET_RETRY_AFTER_S},
    )
    assert harness.verbs == []
    assert len(harness.apis.published()) == publishes
    assert row_budget().in_flight == 0


async def test_a_step_larger_than_the_whole_refresh_row_budget_is_refused_for_good(
    harness: Harness,
) -> None:
    """No later request could claim the rows, so the route must not answer `pending`."""
    await harness.draw_card()
    harness.move_dataset()

    response = await harness.refresh(harness.service(row_budget=RowBudget(max_rows=1)))

    assert (response.status_code, response.json()["code"]) == (422, "rows_over_refresh_budget")
    assert harness.verbs == []
    assert row_budget().in_flight == 0


async def test_a_card_whose_frame_has_no_recipe_and_a_bad_credential_are_refused(
    harness: Harness,
) -> None:
    original = await harness.draw_card()
    harness.move_dataset()
    asked = len(harness.apis.requests)

    unauthorized = await harness.refresh(harness.service(), token="studio-token")

    assert (unauthorized.status_code, unauthorized.json()["code"]) == (401, "unauthorized")
    assert len(harness.apis.requests) == asked

    del harness.apis.recipes[pair(original)]
    missing = await harness.refresh(harness.service())

    assert (missing.status_code, missing.json()["code"]) == (422, "recipe_missing")
    assert (harness.replays, harness.apis.calls("/frames/refresh")) == (0, [])
