"""The replayer against the real services: a recipe built by tool calls on one
dataset version is replayed after the dataset moves.

`get_data`, `filter`, `growth` and `resolve_anchor` run through the mounted
data-mcp and analysis-mcp apps, as in `test_executor_integration.py`. The replay
mints its root through data-mcp's `mint_recorded_root`, whose upstream is a
canned dfo-apis mint with the artifact fetch patched at the seam nothing threads
a transport into, and publishes through presentation-mcp's client over a fake
publish door, so no socket opens.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Mapping
from typing import Any

import httpx2
import pytest
from common.materialisation import FrameRef, get_store
from common.recipe import build_recipe
from common.row_budget import RowBudget
from compiler.surface import McpSurface, ToolCallResult
from data_mcp import client_holder, frame_materialisation
from data_mcp.client import UpstreamClient
from data_mcp.deadline import CallDeadline
from data_mcp.settings import Settings as DataMcpSettings
from harness.recipe_replay import Pending, Refused, Replayed, replay_recipe
from harness.server import create_app
from harness.settings import ANALYSIS_MCP_MOUNT_PATH, DATA_MCP_MOUNT_PATH, Settings
from planner.workflow import ToolId

from .replay_support import (
    REPORTERS,
    SECOND,
    SECOND_VALUES,
    SPEC,
    FakeApis,
    Upstream,
    pair,
    yearly,
)

_THIRD = "cv_" + "3" * 32
_KEPT = ["dcid:FRA", "dcid:GBR"]
_LATEST = {"anchor": "latest", "offset": 0, "rule": "half1"}
_POPULATIONS = [{"spec_id": SPEC, "bindings": {}}]


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


class Services:
    """The mounted services' tools, the canned upstream, and dfo-apis' publish door."""

    def __init__(self, surface: McpSurface, upstream: Upstream) -> None:
        self.surface = surface
        self.upstream = upstream
        self.apis = FakeApis()
        self.assembler = self.apis.assembler()
        self.verbs: list[tuple[str, dict[str, Any]]] = []

    async def call(self, server: str, name: str, arguments: Mapping[str, object]) -> dict[str, Any]:
        result = await self.surface.call_tool(
            tool=ToolId(server=server, name=name), arguments=arguments
        )
        assert result.is_error is False, result.text
        assert result.structured_content is not None
        return result.structured_content

    async def analysis(self, *, name: str, arguments: Mapping[str, object]) -> ToolCallResult:
        """The replayer's tool seam: analysis-mcp over the surface the executor uses,
        recording each call."""
        self.verbs.append((name, dict(arguments)))
        return await self.surface.call_tool(
            tool=ToolId(server="analysis-mcp", name=name), arguments=arguments
        )

    async def published(self, content: Mapping[str, Any]) -> tuple[FrameRef, dict[str, Any]]:
        """The frame a verb answered with, published with its recipe."""
        ref = FrameRef(spec_id=content["spec_id"], content_version=content["content_version"])
        recipe = await build_recipe(ref, store=get_store())
        assert await self.assembler.publish_frame(
            spec_id=ref.spec_id, content_version=ref.content_version, recipe=recipe
        )
        return ref, recipe

    async def replay(self, recipe: Mapping[str, Any]) -> Replayed | Pending | Refused:
        async with self.apis.client() as client:
            return await replay_recipe(
                recipe,
                deadline=CallDeadline.after(40.0),
                call_tool=self.analysis,
                store=get_store(),
                apis=client,
                publish=self.assembler.publish_frame,
                row_budget=RowBudget(max_rows=100_000),
            )

    def called(self, name: str) -> list[dict[str, Any]]:
        return [arguments for verb, arguments in self.verbs if verb == name]

    def growth_of(self, frame: FrameRef) -> dict[str, float]:
        return {row["reporter"]: row["value"] for row in self.apis.frames[pair(frame)]}


@pytest.fixture
async def services(
    monkeypatch: pytest.MonkeyPatch,
    running_lifespan: Callable[..., Any],
    mounted_mcp_transport: Callable[..., Any],
) -> AsyncIterator[Services]:
    upstream = Upstream()
    monkeypatch.setattr(frame_materialisation, "_fetch_artifact_bytes", upstream.fetch)
    client_holder.set_client(
        UpstreamClient(settings=DataMcpSettings(), transport=httpx2.MockTransport(upstream.handler))
    )
    app = create_app(settings=Settings(mcp_api_token="test-token"))
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
        made = Services(surface, upstream)
        yield made
        await made.assembler.aclose()
        await surface.aclose()


async def test_a_filter_then_growth_recipe_is_replayed_after_its_dataset_moves(
    services: Services,
) -> None:
    root = (await services.call("data-mcp", "get_data", {"spec_id": SPEC}))["frame_ref"]
    kept = await services.call(
        "analysis-mcp",
        "filter",
        {
            "frame_ref": root,
            "predicates": [{"kind": "category_match", "column": "reporter", "values": _KEPT}],
        },
    )
    original, recipe = await services.published(
        await services.call(
            "analysis-mcp",
            "growth",
            {
                "frame_ref": {key: kept[key] for key in ("spec_id", "content_version")},
                "window_start": "2019",
                "window_end": "2022",
                "entity_by": ["reporter"],
            },
        )
    )
    services.upstream.move(SECOND_VALUES, to=SECOND)

    replayed = await services.replay(recipe)

    assert isinstance(replayed, Replayed), replayed
    assert replayed.frame != original
    assert replayed.newly_published is True
    assert replayed.recipe["normal_form_hash"] == recipe["normal_form_hash"]
    assert replayed.recipe["replay_key"] != recipe["replay_key"]
    assert replayed.recipe["roots"][0]["dataset"] == {"spec_id": SPEC, "content_version": SECOND}
    assert replayed.recipe["anchor"] is None
    # 2019 to 2022 is 100 to 130 in the first version and 150 to 180 in the second.
    assert services.growth_of(original) == pytest.approx({"dcid:FRA": 0.3, "dcid:GBR": 0.15})
    assert services.growth_of(replayed.frame) == pytest.approx({"dcid:FRA": 0.2, "dcid:GBR": 0.1})
    assert [name for name, _ in services.verbs] == ["filter", "growth"]

    verbs, posts, mints = (
        len(services.verbs),
        len(services.apis.published()),
        services.upstream.mints,
    )
    again = await services.replay(recipe)

    assert isinstance(again, Replayed)
    assert (again.frame, again.recipe, again.newly_published) == (
        replayed.frame,
        replayed.recipe,
        False,
    )
    assert len(services.verbs) == verbs
    assert len(services.apis.published()) == posts
    # The root is minted again on every replay.
    assert services.upstream.mints == mints + 1


async def test_a_rolling_recipe_keeps_its_latest_year_until_half_the_series_hold_the_next(
    services: Services,
) -> None:
    root = (await services.call("data-mcp", "get_data", {"spec_id": SPEC}))["frame_ref"]
    anchor = await services.call(
        "analysis-mcp", "resolve_anchor", {"populations": _POPULATIONS, "roots": [root]}
    )
    assert anchor["period"] == "2022"
    _, recipe = await services.published(
        await services.call(
            "analysis-mcp",
            "growth",
            {
                "frame_ref": root,
                "window_start": "2019",
                "window_end": "2022",
                "entity_by": ["reporter"],
                "rolling": {
                    "window_start": _LATEST | {"offset": -3},
                    "window_end": _LATEST,
                    "population": _POPULATIONS,
                },
                "anchor_key": anchor["anchor_key"],
            },
        )
    )
    assert recipe["anchor"]["period"] == "2022"
    france, britain, america = REPORTERS

    # One of three reporters holds 2023: under half, so the latest year stands.
    services.upstream.move(
        {
            france: yearly(range(2019, 2024), start=150.0),
            britain: yearly(range(2019, 2023), start=300.0),
            america: yearly(range(2019, 2023), start=450.0),
        },
        to=SECOND,
    )
    held = await services.replay(recipe)

    assert isinstance(held, Replayed), held
    assert held.recipe["anchor"]["period"] == "2022"
    assert held.recipe["normal_form_hash"] == recipe["normal_form_hash"]
    assert services.called("resolve_anchor") == [
        {"populations": _POPULATIONS, "roots": [{"spec_id": SPEC, "content_version": SECOND}]}
    ]
    (stood,) = services.called("growth")
    assert (stood["window_start"], stood["window_end"]) == ("2019", "2022")
    assert stood["anchor_key"].startswith("rak_")
    assert services.growth_of(held.frame) == pytest.approx(
        {france: 0.2, britain: 0.1, america: 30.0 / 450.0}
    )

    # Two of three hold 2023: at least half, so the window moves a year on, and
    # the reporter without 2023 is named as left out.
    services.verbs.clear()
    services.upstream.move(
        {
            france: yearly(range(2019, 2024), start=150.0),
            britain: yearly(range(2019, 2024), start=300.0),
            america: yearly(range(2019, 2023), start=450.0),
        },
        to=_THIRD,
    )
    moved = await services.replay(recipe)

    assert isinstance(moved, Replayed), moved
    assert moved.frame != held.frame
    assert moved.recipe["anchor"]["period"] == "2023"
    assert [series["coordinates"]["reporter"] for series in moved.recipe["anchor"]["left_out"]] == [
        america
    ]
    assert moved.recipe["normal_form_hash"] == recipe["normal_form_hash"]
    (rolled,) = services.called("growth")
    assert (rolled["window_start"], rolled["window_end"]) == ("2020", "2023")
    assert rolled["anchor_key"] != stood["anchor_key"]
    # 2020 to 2023 is 160 to 190 for France and 310 to 340 for Britain.
    assert services.growth_of(moved.frame) == pytest.approx(
        {france: 30.0 / 160.0, britain: 30.0 / 310.0}
    )
