"""Search answers for a share of US health aid in the US donor total, each input
carrying a measure-basis scope choice, for the tests of putting a calculation's
inputs on one basis (`submission.service._aligned_bases`)."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from compiler.surface import ToolCallResult
from data_mcp.search_result import SEARCH_ENVELOPE_META_KEY
from planner.workflow import ToolId

from .test_turn_records import _switchable_scope

_FRANCE_KENYA = json.loads(
    (
        Path(__file__).parents[2]
        / "data-mcp"
        / "tests"
        / "fixtures"
        / "resolve_aid_france_kenya.json"
    ).read_text()
)["parts"][0]["spec"]

HEALTH = f"spec_{'1' * 32}"
TOTAL = f"spec_{'2' * 32}"
TOTAL_GROSS = f"spec_{'3' * 32}"


def basis_block(
    spec_id: str,
    *,
    narrow: bool,
    active: str,
    provenance: str = "default",
    gross: str = "available",
) -> dict[str, Any]:
    """Search's France-to-Kenya aid block under `spec_id`, carrying the scope
    choices of a part counted as `active` by `provenance`: US health aid on
    the CRS when `narrow`, else the US donor total, whose part binds no
    sector and no recipient and offers gross disbursements as `gross`."""
    scope = _switchable_scope()
    base = f"{spec_id}_base"
    scope["id"] = scope["base"]["spec_id"] = base
    scope["compiled"]["spec_id"] = spec_id
    if not narrow:
        identity = scope["base"]["identity"]
        identity["slots"] = [s for s in identity["slots"] if s["property"] != "crsPurpose"]
        identity["entities"] = {"origin": ["country/USA"]}
        scope["choices"] = [c for c in scope["choices"] if c["axis"] == "measure_basis"]
        scope["entries"] = []
    for choice in scope["choices"]:
        choice["id"] = f"{base}:{choice['axis']}"
    basis = next(c for c in scope["choices"] if c["axis"] == "measure_basis")
    basis["active"], basis["provenance"] = active, provenance
    for option in basis["options"]:
        option["availability"] = {"status": "available", "detail": {}}
        if option["option_id"] == "measure_basis:gross" and gross == "unavailable":
            option["availability"] = {
                "status": "unavailable",
                "code": "basis_not_published",
                "detail": {},
            }
    if provenance == "explicit":
        scope["entries"] = [
            *scope["entries"],
            {
                "axis": "measure_basis",
                "option_id": active,
                "provenance": "explicit",
                "origin": {"kind": "text", "text": "net", "start": 0, "end": 3},
            },
        ]
    return {
        **_FRANCE_KENYA,
        "spec_id": spec_id,
        "resolved": {**_FRANCE_KENYA["resolved"], "scope_choices": scope},
    }


class BasisCompiles:
    """A tool surface whose `resolve_definition` compiles the US donor total on gross
    disbursements as `TOTAL_GROSS` over the base identity and entries posted,
    its basis settled by the explicit entry posted, recording each call."""

    def __init__(self) -> None:
        self.calls: list[Mapping[str, Any]] = []

    async def list_tools(self) -> list[Any]:
        return []

    async def call_tool(
        self, *, tool: ToolId, arguments: Mapping[str, Any], on_progress: Any = None
    ) -> ToolCallResult:
        del on_progress
        assert tool.name == "resolve_definition", tool
        self.calls.append(arguments)
        block = basis_block(TOTAL_GROSS, narrow=False, active="measure_basis:gross")
        scope = block["resolved"]["scope_choices"]
        scope["id"] = scope["base"]["spec_id"] = arguments["spec_id"]
        scope["base"]["identity"] = arguments["identity"]
        scope["entries"] = arguments["scope_entries"]
        for choice in scope["choices"]:
            choice["id"] = f"{arguments['spec_id']}:{choice['axis']}"
            choice["provenance"] = "explicit"
        part = {"status": "definite", "spec": block}
        return ToolCallResult(is_error=False, meta={SEARCH_ENVELOPE_META_KEY: {"parts": [part]}})
