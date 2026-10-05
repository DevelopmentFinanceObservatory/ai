"""A canned data-mcp for the translator's discovery, with the real servers' schemas.

The translator eval sets grade what the translator does with what discovery
finds, not the search backend, so `search` here answers by the ask: one debt
service source, one population source, two aid candidates, and nothing for
anything else. Every answer resolves exactly the entities the search part
named -- role and label as they were sent, `ref` as `country/<code>` -- the
way the real server resolves what it is asked about, so a plan naming Kenya
gets Kenya's frame and one naming Kenya and Ghana gets both. Its `describe`
names only an identifier this stub can ever put in an answer, and refuses any
other the way the real server refuses a dcid its graph does not hold, so a
translator cannot confirm an identifier it invented. That set is fixed rather
than gathered from the answers a run happened to give, so which case ran first
cannot change what `describe` will confirm.

`EvalSurface` reads tool schemas from the real data-mcp and analysis-mcp,
which the capability table is pinned to, and sends discovery calls to the
canned server. The translator never calls `get_data` or a verb, so no other
call reaches either side.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from analysis_mcp.server import mcp as analysis_mcp
from compiler.surface import McpSurface, ToolCallResult, ToolSchema
from data_mcp.digest import render_answer
from data_mcp.search_result import SEARCH_ENVELOPE_META_KEY
from data_mcp.server import mcp as data_mcp
from data_mcp.tool_errors import ToolRefusal, to_tool_result
from fastmcp import FastMCP
from fastmcp.tools import ToolResult
from planner.workflow import ToolId

DEBT_SPEC = "spec_" + "d" * 32
POPULATION_SPEC = "spec_" + "9" * 32
AID_SPECS = ("spec_" + "b" * 32, "spec_" + "c" * 32)

# ISO 3166-1 alpha-3 codes for the names the seeded plans send, plus a few a
# live planner drafting its own plan is likely to; `country/<code>` is the ref
# the real server resolves a country to. A name outside the table is an
# entity the stub cannot resolve, answered as no data.
_COUNTRY_CODES = {
    "kenya": "KEN",
    "ghana": "GHA",
    "nigeria": "NGA",
    "ethiopia": "ETH",
    "uganda": "UGA",
    "tanzania": "TZA",
}
# The role column a source resolves an entity under when the part named no
# role. `recipient` because that is what the graph actually answers with for
# an unqualified country, and what the non-eval canned server uses: a stub
# that invents a role no real server emits shows a live model a string it
# would never meet.
_UNROLED = "recipient"
_USD = {"ref": "USD", "label": "US dollars"}
_WINDOW = {
    "kind": "exact",
    "has_data": True,
    "observation_count": 30,
    "window": {"start_year": 2010, "end_year": 2024},
}


def _resolved_entities(part: Mapping[str, Any]) -> list[dict[str, str]] | None:
    """The entities `part` names, resolved as the real server would.

    Returns `None` when a name is not a country this stub knows, which the
    caller answers as no data rather than inventing a ref for it.
    """
    resolved: list[dict[str, str]] = []
    for entity in part.get("entities", []):
        text = str(entity.get("text", "")).strip()
        code = _COUNTRY_CODES.get(text.casefold())
        if code is None:
            return None
        role = entity.get("role") or _UNROLED
        resolved.append({"role": str(role), "label": text, "ref": f"country/{code}"})
    return resolved


def _about(entities: list[dict[str, str]]) -> str:
    return " and ".join(entity["label"] for entity in entities) or "all countries"


def _spec(
    spec_id: str,
    sentence: str,
    stat_var: str,
    unit: dict[str, Any] | None,
    entities: list[dict[str, str]],
) -> dict[str, Any]:
    return {
        "spec_id": spec_id,
        "resolved": {
            "rendered_sentence": sentence,
            "confidence": "high",
            "coverage": _WINDOW,
            "stat_vars": {
                "items": [{"ref": stat_var, "label": stat_var, "slot_values": []}],
                "member_count": 1,
                "truncated": False,
            },
            "entities": entities,
            "facets": {"unit": unit} if unit else {},
        },
        "definition": {"schema_version": 1, "spec_id": spec_id, "identity": {"stat_var": stat_var}},
    }


# Every identifier this stub can ever put in an answer. `describe` holds ids to
# this rather than to what the current run happened to search for: `adk eval`
# runs a whole set in one process, so a closure over past answers would let a
# later case describe an id an earlier one named and make the refusal depend on
# case order. The point of the refusal is that an id the stub never emits
# cannot be confirmed, and that holds run to run.
NAMED_IDS = frozenset(
    {DEBT_SPEC, POPULATION_SPEC, *AID_SPECS}
    | {"DebtServicePaid", "Population", "AidDisbursements", "AidReceived"}
    | {str(_USD["ref"])}
    | {f"country/{code}" for code in _COUNTRY_CODES.values()}
)


def _envelope(status: str, **payload: Any) -> dict[str, Any]:
    """A single-part envelope: one entry in `parts`, carrying its own status."""
    part = {"status": status, "variable_text": "asked", "warnings": [], **payload}
    return {"schema_version": 1, "status": status, "parts": [part]}


def _no_data(reason: str) -> dict[str, Any]:
    return _envelope("no_data", no_data={"reason": reason, "category": "unresolved"})


def _specs_for(variable: str, entities: list[dict[str, str]]) -> list[dict[str, Any]]:
    """The spec blocks the ask in `variable` finds, over `entities`; empty for anything else."""
    about = _about(entities)
    # Whole words: "paid" is not "aid".
    if re.search(r"\b(aid|disbursements?)\b", variable):
        return [
            _spec(
                AID_SPECS[0],
                f"gross aid disbursements received by {about} (OECD DAC)",
                "AidDisbursements",
                _USD,
                entities,
            ),
            _spec(
                AID_SPECS[1],
                f"aid received by {about}, all donors (UN)",
                "AidReceived",
                _USD,
                entities,
            ),
        ]
    if re.search(r"\b(population|people|inhabitants)\b", variable):
        return [
            _spec(POPULATION_SPEC, f"total population of {about}", "Population", None, entities)
        ]
    if re.search(r"\bdebt\b", variable):
        return [
            _spec(
                DEBT_SPEC,
                f"external debt service paid by {about}, 2010 to 2024",
                "DebtServicePaid",
                _USD,
                entities,
            )
        ]
    return []


def _node(dcid: str) -> dict[str, Any]:
    if dcid.startswith("country/"):
        shape = "entity"
    elif dcid.startswith("spec_"):
        shape = "spec"
    else:
        shape = "stat_var"
    return {
        "shape": shape,
        "dcid": dcid,
        "label": dcid,
        "definition": f"{dcid}, as the graph defines it.",
    }


def make_discovery() -> FastMCP:
    """Build a canned data-mcp whose `describe` knows only the ids this stub can answer with."""
    discovery = FastMCP(name="data-mcp")

    @discovery.tool(name="search")
    async def search(parts: list[dict[str, Any]], raw_query: str | None = None) -> ToolResult:
        """Answer by the variable asked about, over the entities the part names."""
        part = parts[0]
        variable = str(part.get("variable", "")).casefold()
        entities = _resolved_entities(part)
        if entities is None:
            envelope = _no_data("an entity named is not a country this graph holds")
        else:
            specs = _specs_for(variable, entities)
            if not specs:
                envelope = _no_data("no series matches the ask")
            elif len(specs) == 1:
                envelope = _envelope("definite", spec=specs[0])
            else:
                envelope = _envelope("candidates", candidates={"specs": specs, "truncated": False})
        return ToolResult(
            content=f"search: {envelope['status']}",
            structured_content=render_answer(envelope),
            meta={SEARCH_ENVELOPE_META_KEY: envelope},
        )

    @discovery.tool(name="describe")
    async def describe(ids: list[str]) -> ToolResult:
        """Name each identifier this stub can answer with; refuse any other."""
        unknown = [dcid for dcid in ids if dcid not in NAMED_IDS]
        if unknown:
            return to_tool_result(ToolRefusal.node_not_held(dcids=unknown))
        nodes = [_node(dcid) for dcid in ids]
        return ToolResult(content="nodes", structured_content={"nodes": nodes})

    return discovery


class EvalSurface:
    """Schemas from the real servers, discovery calls to the canned one.

    Each surface gets its own canned server; what `describe` knows is the fixed
    set of ids the stub can answer with, so it does not vary with case order.
    """

    def __init__(self) -> None:
        self._schemas = McpSurface(sources={"data-mcp": data_mcp, "analysis-mcp": analysis_mcp})
        self._discovery = McpSurface(sources={"data-mcp": make_discovery()})

    async def list_tools(self) -> list[ToolSchema]:
        return await self._schemas.list_tools()

    async def call_tool(self, *, tool: ToolId, arguments: Mapping[str, object]) -> ToolCallResult:
        return await self._discovery.call_tool(tool=tool, arguments=arguments)
