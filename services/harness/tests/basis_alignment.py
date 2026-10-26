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
TOTAL_NET = f"spec_{'4' * 32}"

# The spec_id Search answers the US donor total under, by the basis posted.
_COMPILED = {"measure_basis:gross": TOTAL_GROSS, "measure_basis:net": TOTAL_NET}


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
    """A tool surface whose `resolve_definition` compiles the US donor total on the
    basis its explicit entry posts, gross disbursements as `TOTAL_GROSS` and net
    as `TOTAL_NET`, over the base identity and entries posted, and whose `search`
    answers the US donor total as `TOTAL`, counted as grant equivalents by
    default, recording each call's tool and arguments in `calls`."""

    def __init__(self) -> None:
        self.calls: list[Mapping[str, Any]] = []
        self.tools: list[str] = []

    async def list_tools(self) -> list[Any]:
        return []

    async def call_tool(
        self, *, tool: ToolId, arguments: Mapping[str, Any], on_progress: Any = None
    ) -> ToolCallResult:
        del on_progress
        self.calls.append(arguments)
        self.tools.append(tool.name)
        if tool.name == "search":
            searched = basis_block(TOTAL, narrow=False, active="measure_basis:grant_equivalent")
            part = {"status": "definite", "spec": searched}
            return ToolCallResult(
                is_error=False, meta={SEARCH_ENVELOPE_META_KEY: {"parts": [part]}}
            )
        assert tool.name == "resolve_definition", tool
        (basis,) = [
            e["option_id"] for e in arguments["scope_entries"] if e["axis"] == "measure_basis"
        ]
        block = basis_block(_COMPILED[basis], narrow=False, active=basis)
        scope = block["resolved"]["scope_choices"]
        scope["id"] = scope["base"]["spec_id"] = arguments["spec_id"]
        scope["base"]["identity"] = arguments["identity"]
        scope["entries"] = arguments["scope_entries"]
        for choice in scope["choices"]:
            choice["id"] = f"{arguments['spec_id']}:{choice['axis']}"
            choice["provenance"] = "explicit"
        part = {"status": "definite", "spec": block}
        return ToolCallResult(is_error=False, meta={SEARCH_ENVELOPE_META_KEY: {"parts": [part]}})


# The spec_id Search answers the US donor total under counting bilateral flows
# alone, compiled from `TOTAL` (`ProviderCompiles`).
TOTAL_BILATERAL = f"spec_{'5' * 32}"

_PROVIDER_LABELS = {
    "provider:bilateral": "bilateral aid only",
    "provider:inclusive": "bilateral plus imputed multilateral aid",
}


def provider_block(
    spec_id: str,
    *,
    narrow: bool,
    active: str,
    provenance: str = "default",
    basis: str = "measure_basis:gross",
    bilateral: str = "available",
) -> dict[str, Any]:
    """`basis_block` counted gross by default as `basis`, carrying a provider
    choice counted as `active` by `provenance`, its bilateral option `bilateral`,
    as dfo-apis writes the provider axis (`ProviderBinding`)."""
    block = basis_block(spec_id, narrow=narrow, active=basis)
    scope = block["resolved"]["scope_choices"]
    scope["choices"].append(
        {
            "id": f"{scope['id']}:provider",
            "axis": "provider",
            "rule": "provider_scope",
            "provenance": provenance,
            "active": active,
            "options": [
                {
                    "option_id": option_id,
                    "label": label,
                    "binding": {"kind": "provider", "scope": option_id.split(":")[1]},
                    "availability": (
                        {"status": "unavailable", "code": "provider_not_offered", "detail": {}}
                        if option_id == "provider:bilateral" and bilateral == "unavailable"
                        else {"status": "available", "detail": {}}
                    ),
                }
                for option_id, label in _PROVIDER_LABELS.items()
            ],
        }
    )
    if provenance == "explicit":
        scope["entries"] = [
            *scope["entries"],
            {
                "axis": "provider",
                "option_id": active,
                "provenance": "explicit",
                "origin": {"kind": "text", "text": "aid", "start": 0, "end": 3},
            },
        ]
    return block


class ProviderCompiles:
    """A tool surface whose `resolve_definition` compiles the US donor total on the
    provider scope its explicit entry posts, bilateral flows alone as
    `TOTAL_BILATERAL`, recording each call's arguments in `calls`."""

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
        (provider,) = [
            e["option_id"] for e in arguments["scope_entries"] if e["axis"] == "provider"
        ]
        assert provider == "provider:bilateral", provider
        block = provider_block(TOTAL_BILATERAL, narrow=False, active=provider)
        scope = block["resolved"]["scope_choices"]
        scope["id"] = scope["base"]["spec_id"] = arguments["spec_id"]
        scope["base"]["identity"] = arguments["identity"]
        scope["entries"] = arguments["scope_entries"]
        for choice in scope["choices"]:
            choice["id"] = f"{arguments['spec_id']}:{choice['axis']}"
        provider_choice = next(c for c in scope["choices"] if c["axis"] == "provider")
        provider_choice["provenance"] = "explicit"
        part = {"status": "definite", "spec": block}
        return ToolCallResult(is_error=False, meta={SEARCH_ENVELOPE_META_KEY: {"parts": [part]}})
