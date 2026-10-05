"""Loader for the node door's four shape fixtures.

These fixtures are authored against this service's own contract, not
captured from a running upstream -- drift against the live wire is caught
by the environment-gated checks in `test_contract_integration.py`, not by
these.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_FIXTURES_DIR = Path(__file__).parent / "fixtures"


def load_node_fixture(name: str) -> dict[str, Any]:
    """Load one node-door fixture payload by its file stem, e.g. `"node_stat_var"`."""
    return json.loads((_FIXTURES_DIR / f"{name}.json").read_text())
