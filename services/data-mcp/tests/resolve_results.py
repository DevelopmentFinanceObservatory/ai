"""Builders for `POST /api/graph/resolve` results, in the shape dfo-apis' `src/graph_api/README.md`
documents: every result carries every field, `null` where its status does not use it, and a
group or region resolved with `expand: ["members"]` carries its members inline."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any


def resolve_result(text: str, status: str, **fields: Any) -> dict[str, Any]:
    """One wire result for `text`, every field `fields` leaves unset `null`."""
    return {
        "input": text,
        "status": status,
        "entity": None,
        "matched": None,
        "reading": None,
        "projection": None,
        "member_count": None,
        "members": None,
        "candidates": None,
        "near_matches": None,
        **fields,
    }


def resolved_result(label: str, ref: str, kind: str = "country", **fields: Any) -> dict[str, Any]:
    """A `resolved` result naming the entity `ref`, matched by its name `label`."""
    return resolve_result(
        label,
        "resolved",
        entity={"ref": ref, "label": label, "kind": kind},
        matched={"text": label, "as": "name"},
        reading="entity",
        **fields,
    )


def members_result(
    label: str, ref: str, kind: str, members: Sequence[tuple[str, str | None]]
) -> dict[str, Any]:
    """A `resolved` group or region with its `(ref, label)` members inline."""
    return resolved_result(
        label,
        ref,
        kind,
        projection="countries",
        member_count=len(members),
        members=[{"ref": member, "label": name} for member, name in members],
    )
