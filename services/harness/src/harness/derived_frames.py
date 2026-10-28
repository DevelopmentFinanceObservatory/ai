"""The frames dfo-apis' analysis runner stored for the verbs, as the harness reads them.

Every analysis verb computes on the runner, which stores the frame it returns
with its recipe, so the materialisation store holds no `ops_` frame. Each
harness reader of a verb's result reads it from dfo-apis instead
(`analysis_mcp.runner_client`):

- `derived_recipe` is the frame's recipe as the runner stored it, from the
  frame's manifest: what `harness.rolling_cards` replays and
  `harness.mcp_client.McpExecutor.publish_result` takes the replay key of.
- `derived_file` is the frame's rows as its Parquet file, with whether its
  values are rank positions (`holds_rank_positions`): what `make_charts` ranks
  and checks a select against (`harness.orchestrator_tools`).

Both raise `analysis_mcp.runner_client.DerivedUnread` for a frame dfo-apis does
not serve.
"""

from __future__ import annotations

import io
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import pyarrow.parquet as pq
from analysis_mcp.runner_client import read_derived_file, read_derived_manifest
from common.materialisation import FrameRef
from common.recipe import RECIPE_SCHEMA_VERSION
from data_mcp.frame_materialisation import frame_from_parquet_bytes
from dfo_frame_grammar import Frame

__all__ = ["DerivedFile", "derived_file", "derived_recipe", "holds_rank_positions"]

_RANK = "rank"


@dataclass(frozen=True, kw_only=True)
class DerivedFile:
    """A derived frame's rows as one Parquet file, read before any row is decoded:
    `num_rows` from its footer, and whether its values are rank positions."""

    data: bytes
    num_rows: int
    rank_positions: bool

    def frame(self) -> Frame:
        """The frame the file holds. The runner serves a derived frame as it serves a
        dataset version, in the frame layout whose `unit` is text, which `get_data`
        decodes the same way (`data_mcp.frame_materialisation`). Decodes the whole
        file."""
        return frame_from_parquet_bytes(self.data)

    def rows(self) -> list[dict[str, Any]]:
        """Every row of `frame`, each as a mapping, as the store's frames give them."""
        frame = self.frame()
        return [dict(zip(frame.columns, row, strict=True)) for row in frame.rows]


async def derived_recipe(ref: FrameRef) -> dict[str, Any] | None:
    """The recipe the runner stored for `ref`, in the form apis' publish door
    stores a recipe (`common.recipe`): its manifest's `recipe` with the manifest's
    `anchor` and `left_out`. `None` for a frame stored without a recipe.

    The tree's nodes carry no `frame`: a derived node is `{op_spec, inputs}` and a
    root `{root}`, an index into `roots`.
    """
    manifest = await read_derived_manifest(ref)
    recipe = manifest.get("recipe")
    if not isinstance(recipe, Mapping):
        return None
    left_out = manifest.get("left_out")
    return {
        "schema_version": RECIPE_SCHEMA_VERSION,
        **recipe,
        "anchor": manifest.get("anchor"),
        "left_out": None
        if left_out is None
        else {"series": left_out, "complete": manifest.get("left_out_complete") is not False},
    }


def holds_rank_positions(tree: Mapping[str, Any]) -> bool:
    """Whether the values of the frame `tree` computes are rank positions: some step
    is a per-period `rank` with no `limit`, whose rows carry each entity's position.

    A verb carries that mark from any of its inputs onward, and every verb that
    would read a position as a quantity refuses it, so a step anywhere in the tree
    marks the frame (`analysis_mcp.frame_ref_resolution.DerivedMarker.ordinal`)."""
    if "root" in tree:
        return False
    op_spec = tree["op_spec"]
    parameters = op_spec["parameters"]
    if (
        op_spec["verb"] == _RANK
        and parameters.get("per_period") is True
        and parameters.get("limit") is None
    ):
        return True
    return any(holds_rank_positions(child) for child in tree["inputs"].values())


async def derived_file(ref: FrameRef) -> DerivedFile:
    """The rows file of the derived frame `ref`, and its manifest's word on whether
    they are rank positions.

    Raises:
        DerivedUnread: dfo-apis serves no such frame, or the read failed.
    """
    manifest = await read_derived_manifest(ref)
    recipe = manifest.get("recipe")
    data = await read_derived_file(ref)
    return DerivedFile(
        data=data,
        num_rows=pq.ParquetFile(io.BytesIO(data)).metadata.num_rows,
        rank_positions=isinstance(recipe, Mapping) and holds_rank_positions(recipe["tree"]),
    )
