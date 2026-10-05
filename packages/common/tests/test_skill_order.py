"""Tests for `common.skills.ordered_skill_ids`: the order a search envelope's skills take."""

from __future__ import annotations

from typing import Any

from common.skills import ordered_skill_ids


def _spec(*skill_ids: str) -> dict[str, Any]:
    return {"resolved": {"skills": list(skill_ids)}}


def _skills(*skill_ids: str) -> dict[str, Any]:
    return {skill_id: {"id": skill_id, "description": skill_id} for skill_id in skill_ids}


def test_ids_follow_first_appearance_across_candidates_most_specific_first() -> None:
    envelope = {
        "parts": [
            {
                "status": "candidates",
                "candidates": {"specs": [_spec("malaria", "oda"), _spec("oda", "deflators")]},
            }
        ],
        "skills": _skills("deflators", "oda", "malaria"),
    }

    assert ordered_skill_ids(envelope) == ["malaria", "oda", "deflators"]


def test_parts_are_walked_in_order_through_each_status_shape() -> None:
    envelope = {
        "parts": [
            {"status": "unsupported"},
            {
                "status": "no_data",
                "no_data": {
                    "nearest_real": [_spec("near")],
                    "flagged_match": _spec("flagged"),
                    "alternate_match": _spec("alternate"),
                },
            },
            {"status": "definite", "spec": _spec("definite")},
        ],
        "skills": _skills("definite", "alternate", "flagged", "near"),
    }

    assert ordered_skill_ids(envelope) == ["near", "flagged", "alternate", "definite"]


def test_ids_no_rendered_spec_names_follow_in_id_order() -> None:
    envelope = {
        "parts": [{"status": "definite", "spec": _spec("zeta")}],
        "skills": _skills("zeta", "beta", "alpha"),
    }

    assert ordered_skill_ids(envelope) == ["zeta", "alpha", "beta"]


def test_a_spec_id_the_map_lacks_is_skipped() -> None:
    envelope = {
        "parts": [{"status": "definite", "spec": _spec("absent", "present")}],
        "skills": _skills("present"),
    }

    assert ordered_skill_ids(envelope) == ["present"]


def test_an_absent_or_empty_map_orders_nothing() -> None:
    parts = [{"status": "definite", "spec": _spec("oda")}]

    assert ordered_skill_ids({"parts": parts}) == []
    assert ordered_skill_ids({"parts": parts, "skills": None}) == []
    assert ordered_skill_ids({"parts": parts, "skills": {}}) == []
