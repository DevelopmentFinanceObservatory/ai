"""Loader for the node envelope fixtures.

Each fixture is one node envelope of apis' Graph wire: `node_stat_var`, `node_group`,
`node_continent` and `node_country` are the response examples of apis'
`src/graph_api/README.md`, served by the app its `tests/graph_api` fixtures build;
`node_source` and `node_other` follow the same schema. Every fixture validates through the
generated `data_mcp.graph_wire` models, so a schema change that a fixture no longer meets
fails the tests that load it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from data_mcp.graph_wire import NodeResponse
from data_mcp.node_lookup import HeldNode

_FIXTURES_DIR = Path(__file__).parent / "fixtures"

NODE_FIXTURES = (
    "node_stat_var",
    "node_group",
    "node_continent",
    "node_country",
    "node_source",
    "node_other",
)


def load_node_fixture(name: str) -> dict[str, Any]:
    """Load one node envelope payload by its file stem, e.g. `"node_stat_var"`."""
    return json.loads((_FIXTURES_DIR / f"{name}.json").read_text())


def node_fixture(name: str, **overrides: Any) -> HeldNode:
    """The validated envelope of fixture `name`, with `overrides` replacing its members."""
    return NodeResponse.model_validate({**load_node_fixture(name), **overrides}).root
