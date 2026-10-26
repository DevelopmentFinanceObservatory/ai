"""A canned data-mcp for the translator's discovery, with the real servers' schemas.

The eval sets and the tests that run translation grade what translation does
with what discovery finds, not the search backend, so `search` here answers by
the ask: one debt service source, one population source, one exports source,
one official development assistance source, two aid candidates, and nothing for
anything else. A part naming "Congo" bare is answered with the ambiguity
between its two graph nodes, as the parsed door answers it, and a row
carrying one of them as its `dcid` resolves to that node under its graph
label. Every other answer resolves exactly the entities the search part named -- role
and label as they were sent, `ref` as `country/<code>` -- the way the real
server resolves what it is asked about, so a plan naming Kenya gets Kenya's
frame and one naming Kenya and Ghana gets both. Its `describe` answers
the real tool's output for the graph's node envelope -- built from the
generated `data_mcp.graph_wire` models and rendered by the real digest -- for
the countries and stat vars this stub can ever put in an answer, and refuses any
other the way the real server refuses a dcid its graph does not hold (a
spec_id is no graph node, so it is refused too), so a translator cannot
confirm an identifier it invented. That set is fixed rather
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
from data_mcp.describe_digest import render_describe_digest
from data_mcp.digest import render_answer
from data_mcp.graph_wire import (
    CountryNode,
    PlaceBlock,
    StatVarBlock,
    StatVarNode,
    UnitBlock,
    UnitNode,
    VariableDefinition,
)
from data_mcp.search_result import SEARCH_ENVELOPE_META_KEY
from data_mcp.server import mcp as data_mcp
from data_mcp.tool_errors import ToolRefusal, to_tool_result
from fastmcp import FastMCP
from fastmcp.tools import ToolResult
from planner.workflow import ToolId

DEBT_SPEC = "spec_" + "d" * 32
POPULATION_SPEC = "spec_" + "9" * 32
AID_SPECS = ("spec_" + "b" * 32, "spec_" + "c" * 32)
ODA_SPEC = "spec_" + "e" * 32
EXPORTS_SPEC = "spec_" + "a" * 32

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
# The two graph nodes the bare word "Congo" can name. Search answers a part
# naming it bare with an ambiguity over these, and a row carrying one of
# them as its `dcid` with that node.
_CONGOS = {
    "country/COD": "Democratic Republic of the Congo",
    "country/COG": "Republic of the Congo",
}
# The role column a source resolves an entity under when the part named no
# role. `recipient` because that is what the graph actually answers with for
# an unqualified country, and what the non-eval canned server uses: a stub
# that invents a role no real server emits shows a live model a string it
# would never meet. A population is not about a party to a flow, so its
# countries resolve under `observationAbout`, the column the graph holds them on.
_UNROLED = "recipient"
_POPULATION_UNROLED = "observationAbout"
_POPULATION = re.compile(r"\b(population|people|inhabitants)\b")
_USD = {"ref": "USD", "label": "US dollars"}
_WINDOW = {
    "kind": "exact",
    "has_data": True,
    "observation_count": 30,
    "window": {"start_year": 2010, "end_year": 2024},
}


def _resolved_entities(part: Mapping[str, Any], *, unroled: str) -> list[dict[str, str]] | None:
    """The entities `part` names, resolved as the real server would.

    Returns `None` when a name is not a country this stub knows, which the
    caller answers as no data rather than inventing a ref for it.
    """
    resolved: list[dict[str, str]] = []
    for entity in part.get("entities", []):
        text = str(entity.get("text", "")).strip()
        role = str(entity.get("role") or unroled)
        dcid = entity.get("dcid")
        if dcid in _CONGOS:
            resolved.append({"role": role, "label": _CONGOS[dcid], "ref": dcid})
            continue
        code = _COUNTRY_CODES.get(text.casefold())
        if code is None:
            return None
        resolved.append({"role": role, "label": text, "ref": f"country/{code}"})
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
            "stat_vars": {"member_count": 1, "member_count_exact": True, "served_count": 1},
            "entities": entities,
            "facets": {"unit": unit} if unit else {},
        },
        "definition": {
            "schema_version": 1,
            "spec_id": spec_id,
            "identity": {"shape": {"kind": "stat_var", "id": stat_var}},
        },
    }


# Every graph identifier this stub can ever put in an answer. `describe` holds ids to
# this rather than to what the current run happened to search for: `adk eval`
# runs a whole set in one process, so a closure over past answers would let a
# later case describe an id an earlier one named and make the refusal depend on
# case order. The point of the refusal is that an id the stub never emits
# cannot be confirmed, and that holds run to run.
STAT_VARS = frozenset(
    {"DebtServicePaid", "Population", "AidDisbursements", "AidReceived", "OdaReceived", "Exports"}
)
_COUNTRY_NAMES = {
    **{f"country/{code}": name.title() for name, code in _COUNTRY_CODES.items()},
    **_CONGOS,
}
NAMED_IDS = STAT_VARS | {str(_USD["ref"])} | _COUNTRY_NAMES.keys()


def _envelope(status: str, **payload: Any) -> dict[str, Any]:
    """A single-part envelope: one entry in `parts`, carrying its own status."""
    part = {"status": status, "variable_text": "asked", "warnings": [], **payload}
    return {"schema_version": 1, "status": status, "parts": [part]}


def _no_data(reason: str) -> dict[str, Any]:
    return _envelope("no_data", no_data={"reason": reason, "category": "unresolved"})


def _bare_congo(entity: Mapping[str, Any]) -> bool:
    return str(entity.get("text", "")).strip().casefold() == "congo" and not entity.get("dcid")


def _congo_ambiguous(part: Mapping[str, Any]) -> dict[str, Any]:
    """The parsed door's answer for a part naming "Congo" bare: one group, whose
    options pin every bare "Congo" row, since rows sent alike share one identity."""
    candidates = [{"ref": ref, "label": label, "kind": "country"} for ref, label in _CONGOS.items()]
    sent = list(part.get("entities", []))
    options = [
        {
            "mentions": ["Congo"],
            "label": candidate["label"],
            "entity_choice": 0,
            "data_query": {
                "parts": [
                    {
                        "variable": part.get("variable"),
                        "entities": [
                            {**row, "dcid": candidate["ref"]} if _bare_congo(row) else row
                            for row in sent
                        ],
                    }
                ]
            },
        }
        for candidate in candidates
    ]
    return _envelope(
        "no_data",
        no_data={
            "category": "unresolved",
            "reason": "entity_ambiguous",
            "ambiguous_entities": [{"text": "Congo", "candidates": candidates}],
            "options": options,
        },
    )


def _specs_for(variable: str, entities: list[dict[str, str]]) -> list[dict[str, Any]]:
    """The spec blocks the ask in `variable` finds, over `entities`; empty for anything else."""
    about = _about(entities)
    if re.search(r"\bofficial development assistance\b", variable):
        return [
            _spec(
                ODA_SPEC,
                f"official development assistance received by {about}",
                "OdaReceived",
                _USD,
                entities,
            )
        ]
    if re.search(r"\bexports?\b", variable):
        return [
            _spec(
                EXPORTS_SPEC,
                f"exports of goods and services by {about}",
                "Exports",
                _USD,
                entities,
            )
        ]
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
    if _POPULATION.search(variable):
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


def _node(dcid: str) -> CountryNode | StatVarNode | UnitNode:
    """The node envelope the Graph API serves for `dcid`, a name this stub can answer with."""
    if dcid in _COUNTRY_NAMES:
        return CountryNode(
            ref=dcid,
            links={},
            built_from={"code_index": "eval-stub"},
            kind="country",
            label=_COUNTRY_NAMES[dcid],
            types=["Country"],
            description=None,
            gaps=[],
            country=PlaceBlock(contained_in=[], membership=None),
        )
    if dcid in STAT_VARS:
        return StatVarNode(
            ref=dcid,
            links={},
            built_from={"code_index": "eval-stub"},
            kind="stat_var",
            label=dcid,
            types=["StatisticalVariable"],
            description=f"{dcid}, as the graph defines it.",
            gaps=["stat_var.availability"],
            stat_var=StatVarBlock(
                availability=None,
                definition=VariableDefinition(
                    constraints=[],
                    definitions=[],
                    measure=None,
                    measured_property=None,
                    population_type=None,
                    roles=["observationAbout"],
                    source_indicator_codes=[],
                    stat_type=None,
                ),
            ),
        )
    return UnitNode(
        ref=dcid,
        links={},
        built_from={"code_index": "eval-stub"},
        kind="unit",
        label=str(_USD["label"]),
        types=["UnitOfMeasure"],
        description=None,
        gaps=[],
        unit=UnitBlock(
            changes_magnitude=False,
            currency="USD",
            currency_noun=str(_USD["label"]),
            price_basis=None,
            unit_class="currency",
        ),
    )


def make_discovery() -> FastMCP:
    """Build a canned data-mcp whose `describe` knows only the ids this stub can answer with."""
    discovery = FastMCP(name="data-mcp")

    @discovery.tool(name="search")
    async def search(parts: list[dict[str, Any]], raw_query: str | None = None) -> ToolResult:
        """Answer by the variable asked about, over the entities the part names."""
        part = parts[0]
        variable = str(part.get("variable", "")).casefold()
        unroled = _POPULATION_UNROLED if _POPULATION.search(variable) else _UNROLED
        entities = _resolved_entities(part, unroled=unroled)
        if any(_bare_congo(entity) for entity in part.get("entities", [])):
            envelope = _congo_ambiguous(part)
        elif entities is None:
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
        """Describe each graph identifier this stub can answer with; refuse any other."""
        unknown = [dcid for dcid in ids if dcid not in NAMED_IDS]
        if unknown:
            return to_tool_result(ToolRefusal.node_not_held(dcids=unknown))
        unique_ids = list(dict.fromkeys(ids))
        outcomes = {dcid: _node(dcid) for dcid in unique_ids}
        # The shape `describe_tool.describe` returns on success: the digest for
        # the model, the dumped envelopes in structured content for code.
        return ToolResult(
            content=render_describe_digest(dcids=ids, outcomes=outcomes),
            structured_content={
                "nodes": [
                    outcomes[dcid].model_dump(mode="json", by_alias=True) for dcid in unique_ids
                ]
            },
        )

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
