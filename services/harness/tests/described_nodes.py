"""Builder for the `nodes` a `describe` answer carries: dfo-apis' Graph node envelopes, as
data-mcp's `describe` tool dumps them from `data_mcp.graph_wire`."""

from __future__ import annotations

from typing import Any


def stat_var_node(
    ref: str,
    *,
    label: str | None = None,
    description: str | None = None,
    measured_property: str | None = None,
    population_type: str | None = None,
    roles: tuple[str, ...] = ("observationAbout",),
) -> dict[str, Any]:
    """One `stat_var` envelope for `ref`, with no availability and the definition parts
    named, each a `{ref, label}` whose label is its ref."""

    def part(name: str | None) -> dict[str, str] | None:
        return None if name is None else {"ref": name, "label": name}

    return {
        "ref": ref,
        "label": label,
        "types": ["StatisticalVariable"],
        "description": description,
        "links": {"facets": f"/api/graph/node/{ref}/members?relation=facets"},
        "gaps": ["stat_var.availability"],
        "built_from": {},
        "kind": "stat_var",
        "stat_var": {
            "definition": {
                "population_type": part(population_type),
                "measured_property": part(measured_property),
                "stat_type": None,
                "measure": None,
                "roles": list(roles),
                "source_indicator_codes": [],
                "constraints": [],
                "definitions": [],
            },
            "availability": None,
        },
    }
