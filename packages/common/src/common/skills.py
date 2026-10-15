"""The order a search envelope's skills are presented in.

apis' resolve envelope carries front matter once, on its `skills` map
keyed by id, and each spec's `resolved.skills` lists the ids that apply to it,
most specific first. data-mcp's search digest and the harness planner's brief
both list those skills, and both read the order from `ordered_skill_ids` so the
two surfaces name the same skills first.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def ordered_skill_ids(envelope: Mapping[str, Any]) -> list[str]:
    """The ids in `envelope`'s `skills` map, in presentation order.

    Ids come in first-appearance order across the specs each part renders,
    in part order, so the first result's most specific skill leads. Ids the
    map carries that no rendered spec names (an equivalent reading's own)
    follow in id order. A spec's id the map lacks is skipped. A `None` or
    empty map gives an empty list.
    """
    skills = envelope.get("skills") or {}
    if not skills:
        return []
    named: dict[str, None] = {}
    for part in envelope.get("parts") or []:
        for spec in _rendered_specs(part):
            for skill_id in (spec.get("resolved") or {}).get("skills") or []:
                if skill_id in skills:
                    named.setdefault(skill_id, None)
    return [*named, *sorted(skills.keys() - named.keys())]


def _rendered_specs(part: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """A part's specs in the order a digest block renders them."""
    status = part.get("status")
    if status == "definite":
        spec = part.get("spec")
        return [spec] if spec else []
    if status == "candidates":
        return list((part.get("candidates") or {}).get("specs") or [])
    if status == "no_data":
        no_data = part.get("no_data") or {}
        specs = list(no_data.get("nearest_real") or [])
        specs.extend(
            match
            for match in (no_data.get("flagged_match"), no_data.get("alternate_match"))
            if match
        )
        return specs
    return []
