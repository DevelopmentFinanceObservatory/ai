"""Tests for `render_answer`, the small `structuredContent` a `search` success carries.

The answer states what the digest states, as data, so a host that hands
`structuredContent` to a model shows it nothing the digest text does not.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from data_mcp.digest import render_answer, render_digest

from .digest_corpus import CASES
from .test_digest import _envelope, _part, _skill, _spec, _with_skills

_FIXTURES = Path(__file__).parent / "fixtures"
_CAPTURED = (
    "resolve_population_guatemala",
    "resolve_gdp_per_capita_guatemala",
    "resolve_aid_france_kenya",
)
_CAPTURED_BROAD = "resolve_health_spending_broad"

_ANSWER_KEYS = {"kind", "status", "parts", "skills", "warnings"}
_PART_KEYS = {"label", "status", "specs", "warnings"}
_SPEC_KEYS = {
    "spec_id",
    "label",
    "measure",
    "member_count",
    "member_count_exact",
    "varies_by",
    "counted_as",
    "alternatives",
}

# The answer for a single-part recorded envelope stays well under one kilobyte,
# a size a model's context takes comfortably.
_SINGLE_PART_ANSWER_BUDGET_BYTES = 1024


def _size(value: Any) -> int:
    return len(json.dumps(value, separators=(",", ":")).encode())


def _captured(name: str) -> dict[str, Any]:
    return json.loads((_FIXTURES / f"{name}.json").read_text())


def _spec_ids(answer: dict[str, Any]) -> list[str]:
    return [spec["spec_id"] for part in answer["parts"] for spec in part["specs"]]


def test_a_definite_part_names_its_spec_and_status() -> None:
    envelope = _envelope("definite", spec=_spec("spec_1", sentence="malaria grants to Togo"))

    assert render_answer(envelope) == {
        "status": "definite",
        "parts": [
            {
                "label": "malaria grants",
                "status": "definite",
                "specs": [{"spec_id": "spec_1", "label": "malaria grants to Togo"}],
            }
        ],
    }


def test_candidates_are_capped_as_the_digest_caps_them() -> None:
    specs = [_spec(f"spec_{index}") for index in range(8)]
    envelope = _envelope("candidates", candidates={"specs": specs, "truncated": False})

    answer = render_answer(envelope)

    assert _spec_ids(answer) == [f"spec_{index}" for index in range(6)]
    digest = render_digest(envelope)
    assert all(spec_id in digest for spec_id in _spec_ids(answer))
    assert "spec_6" not in digest


def test_a_no_data_part_offers_its_nearest_specs_and_leaves_out_the_flagged_match() -> None:
    envelope = _envelope(
        "no_data",
        no_data={
            "reason": "role_incompatible",
            "category": "absent",
            "nearest_real": [_spec("spec_1"), _spec("spec_2"), _spec("spec_3")],
            "flagged_match": _spec("spec_flagged"),
        },
    )

    answer = render_answer(envelope)

    assert _spec_ids(answer) == ["spec_1", "spec_2"]
    assert "spec_flagged" not in json.dumps(answer)


def test_an_unsupported_part_offers_no_spec_and_keeps_only_caller_input_warnings() -> None:
    envelope = _envelope(
        "unsupported",
        parts=[
            _part(
                "unsupported",
                unsupported={
                    "code": "facet_binding_unsupported",
                    "pointer": "/identity/facets/unit",
                },
                warnings=[
                    {"code": "role_text_not_recognised", "message": "role 'x' was not recognised"},
                    {"code": "place_scope_defaulted", "message": "a default scope was used"},
                ],
            )
        ],
    )

    part = render_answer(envelope)["parts"][0]

    assert part["specs"] == []
    assert [warning["code"] for warning in part["warnings"]] == ["role_text_not_recognised"]


@pytest.mark.parametrize(
    ("stat_vars", "facts"),
    [
        # A family whose membership settled.
        (
            {"member_count": 1_454, "member_count_exact": True, "served_count": 1_454},
            {"member_count": 1_454, "member_count_exact": True},
        ),
        # A recall sample: the members Search confirmed, a floor on the family.
        (
            {"member_count": 5, "member_count_exact": False, "served_count": None},
            {"member_count": 5, "member_count_exact": False},
        ),
    ],
    ids=["settled-family", "recall-sample"],
)
def test_a_spec_states_its_measure_count_and_axes_and_names_no_member(
    stat_vars: dict[str, Any], facts: dict[str, Any]
) -> None:
    spec = _spec()
    spec["resolved"]["stat_vars"] = stat_vars
    spec["resolved"]["shape"] = {"label": "population"}
    spec["resolved"]["slots"] = [
        {
            "key": {"property": {"dcid": "gender", "label": "gender"}, "label": "sex"},
            "binding": {"kind": "unbound"},
        },
        {
            "key": {"property": {"dcid": "age", "label": "age"}, "label": "age"},
            "binding": {"kind": "value", "value": {"ref": {"dcid": "Years15Onwards"}}},
        },
        {"key": {"axis": "where", "label": "place"}, "binding": {"kind": "unbound"}},
    ]
    envelope = _envelope("definite", spec=spec)

    answer = render_answer(envelope)["parts"][0]["specs"][0]

    assert answer == {
        "spec_id": "spec_1",
        "label": spec["resolved"]["rendered_sentence"],
        "measure": "population",
        **facts,
        "varies_by": ["sex"],
    }


def test_a_recorded_definite_spec_states_its_elected_measure_and_sibling() -> None:
    spec = render_answer(_captured("resolve_aid_france_kenya"))["parts"][0]["specs"][0]

    assert spec["counted_as"] == "gross disbursements"
    assert spec["alternatives"] == [
        {
            "spec_id": "spec_c67524ce5da3bfd4d94994a76c560b28",
            "label": "net disbursements",
            "axis": "what",
        }
    ]


def test_a_spec_with_no_elected_stat_vars_states_no_count() -> None:
    spec = render_answer(_envelope("definite", spec=_spec()))["parts"][0]["specs"][0]

    assert "member_count" not in spec


def test_a_part_with_no_warnings_has_no_warnings_key() -> None:
    part = render_answer(_envelope("definite", spec=_spec()))["parts"][0]

    assert "warnings" not in part


def test_diagnostics_warnings_sit_beside_the_parts() -> None:
    envelope = _envelope(
        "definite",
        spec=_spec(),
        warnings=[
            {"code": "time_window_defaulted", "message": "no time was given", "spec_id": "s"}
        ],
    )

    assert render_answer(envelope)["warnings"] == [
        {"code": "time_window_defaulted", "message": "no time was given", "spec_id": "s"}
    ]


def test_a_mixed_envelope_states_its_status_and_each_part() -> None:
    envelope = _envelope(
        "mixed",
        parts=[
            _part("definite", variable_text="a", spec=_spec("spec_a")),
            _part("no_data", variable_text="b", no_data={"reason": "no_observations"}),
        ],
    )

    answer = render_answer(envelope)

    assert answer["status"] == "mixed"
    assert [(part["label"], part["status"]) for part in answer["parts"]] == [
        ("a", "definite"),
        ("b", "no_data"),
    ]


def test_skill_ids_sit_beside_the_parts_in_the_digests_order_and_under_its_cap() -> None:
    skill_ids = [f"skill-{index}" for index in range(7)]
    envelope = _envelope("definite", spec=_with_skills(_spec(), *skill_ids))
    envelope["skills"] = {skill_id: _skill(skill_id, "x") for skill_id in skill_ids}

    assert render_answer(envelope)["skills"] == skill_ids[:5]


def test_an_envelope_without_skills_has_no_skills_key() -> None:
    assert "skills" not in render_answer(_envelope("definite", spec=_spec()))


def _every_envelope() -> list[tuple[str, dict[str, Any]]]:
    recorded = [(name, _captured(name)) for name in (*_CAPTURED, _CAPTURED_BROAD)]
    return [*recorded, *((f"corpus:{case.id}", case.envelope) for case in CASES)]


@pytest.mark.parametrize(("name", "envelope"), _every_envelope())
def test_the_answer_states_only_what_the_digest_states(name: str, envelope: dict[str, Any]) -> None:
    answer = render_answer(envelope)
    digest = render_digest(envelope)

    assert set(answer) <= _ANSWER_KEYS
    for part in answer["parts"]:
        assert set(part) <= _PART_KEYS
        for spec in part["specs"]:
            assert set(spec) <= _SPEC_KEYS
            assert spec["spec_id"] in digest
            assert not spec["label"] or spec["label"] in digest
            assert "measure" not in spec or spec["measure"] in digest
            assert "member_count" not in spec or f"{spec['member_count']:,} StatVar" in digest
            assert "counted_as" not in spec or f"Counted as {spec['counted_as']}" in digest
            for alternative in spec.get("alternatives", []):
                assert f"{alternative['label']} [{alternative['spec_id']}]" in digest
        for warning in part.get("warnings", []):
            assert warning["message"] in digest
    for skill_id in answer.get("skills", []):
        assert f"- {skill_id}" in digest
    for warning in answer.get("warnings", []):
        assert warning["message"] in digest


@pytest.mark.parametrize("name", [*_CAPTURED, _CAPTURED_BROAD])
def test_a_recorded_single_part_answer_is_a_small_fraction_of_its_envelope(name: str) -> None:
    envelope = _captured(name)

    answer = render_answer(envelope)

    assert _size(answer) < _SINGLE_PART_ANSWER_BUDGET_BYTES
    assert _size(answer) * 10 < _size(envelope)
