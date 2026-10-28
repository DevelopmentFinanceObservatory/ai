"""Tests for digest rendering: parts, their own statuses, and warnings."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from data_mcp.digest import render_answer, render_digest

_FIXTURES = Path(__file__).resolve().parent / "fixtures"

# Envelope factories. `spec_id`/`sentence`/`confidence`/`coverage`/`stat_vars`/
# `entities`/`facets` mirror one resolved spec; `_part` wraps one status
# payload into a `parts` entry; `_envelope` wraps one or more parts.


def _spec(
    spec_id: str = "spec_1",
    *,
    sentence: str = "malaria grants from France to Togo, 2015 to 2024",
    confidence: str | None = "high",
    coverage: dict[str, Any] | None = None,
    member_count: int = 0,
    entities: list[dict[str, Any]] | None = None,
    facets: dict[str, Any] | None = None,
    facet_options: list[dict[str, Any]] | None = None,
    observations: dict[str, Any] | None = None,
) -> dict[str, Any]:
    resolved: dict[str, Any] = {
        "rendered_sentence": sentence,
        "confidence": confidence,
        "coverage": coverage if coverage is not None else {"kind": "bare", "has_data": True},
        "stat_vars": {
            "member_count": member_count,
            "member_count_exact": True,
            "served_count": member_count,
        },
        "entities": entities if entities is not None else [],
        "facets": facets if facets is not None else {},
        "facet_options": facet_options if facet_options is not None else [],
    }
    if observations is not None:
        resolved["observations"] = observations
    return {"spec_id": spec_id, "resolved": resolved}


def _facet_option(dcid: str, label: str) -> dict[str, Any]:
    """One `SlotOption`-shaped facet option value, keyed by dcid and label."""
    return {"value": {"ref": {"ref": dcid, "label": label}, "value_kind": "enum_value"}}


def _part(
    status: str,
    *,
    variable_text: str | None = "malaria grants",
    warnings: list[dict[str, Any]] | None = None,
    **status_payload: Any,
) -> dict[str, Any]:
    return {
        "status": status,
        "variable_text": variable_text,
        "warnings": warnings if warnings is not None else [],
        **status_payload,
    }


def _envelope(
    status: str,
    *,
    variable_text: str | None = "malaria grants",
    warnings: list[dict[str, Any]] | None = None,
    parts: list[dict[str, Any]] | None = None,
    sentence_plan: list[dict[str, Any]] | None = None,
    **status_payload: Any,
) -> dict[str, Any]:
    """A single-part envelope, unless `parts` supplies the parts list directly."""
    envelope = {
        "schema_version": 1,
        "query_echo": {"entry_path": "parsed", "extract_skipped": True},
        "diagnostics": {
            "engine_build": "test",
            "trace_id": "trace-1",
            "warnings": warnings if warnings is not None else [],
        },
        "status": status,
        "parts": parts
        if parts is not None
        else [_part(status, variable_text=variable_text, **status_payload)],
    }
    if sentence_plan is not None:
        envelope["sentence_plan"] = sentence_plan
    return envelope


# Digest rendering, definite status


def test_definite_digest_carries_sentence_spec_id_confidence_coverage() -> None:
    envelope = _envelope(
        "definite",
        spec=_spec(
            spec_id="spec_9f2c",
            coverage={
                "kind": "exact",
                "has_data": True,
                "observation_count": 10,
                "window": {"start_year": 2015, "end_year": 2024},
            },
            member_count=1,
            entities=[
                {"role": "provider", "label": "France", "ref": "country/FRA"},
                {"role": "recipient", "label": "Togo", "ref": "country/TGO"},
            ],
            facets={
                "provenance": {
                    "basis": "bound",
                    "ref": "oecd_dac_2a",
                    "label": "OECD DAC table 2a",
                },
                "unit": {"basis": "declared_default", "ref": "usd", "label": "US dollars"},
            },
        ),
    )

    digest = render_digest(envelope)

    assert "spec_9f2c" in digest
    assert "confidence high" in digest
    assert "10 observations, 2015-2024" in digest
    assert "has data" in digest
    assert "provider France (country/FRA)" in digest
    assert "recipient Togo (country/TGO)" in digest
    assert "source OECD DAC table 2a (bound)" in digest
    assert "unit US dollars (default)" in digest


def test_definite_digest_per_series_facet_renders_as_varies() -> None:
    envelope = _envelope(
        "definite", spec=_spec(facets={"unit": {"basis": "bound", "selection": "per_series"}})
    )

    digest = render_digest(envelope)

    assert "unit varies per series" in digest


def test_definite_digest_renders_facet_options_one_line_per_axis_capped_at_three() -> None:
    envelope = _envelope(
        "definite",
        spec=_spec(
            facet_options=[
                {
                    "axis": "provenance",
                    "options": [
                        _facet_option("oecd_dac_2a", "OECD DAC table 2a"),
                        _facet_option("oecd_dac_1", "OECD DAC table 1"),
                        _facet_option("unesco_uis", "UNESCO UIS"),
                        _facet_option("world_bank_wdi", "World Development Indicators"),
                        _facet_option("un_comtrade", "UN Comtrade"),
                    ],
                    "truncated": True,
                    "source": "observed",
                },
                {
                    "axis": "unit",
                    "options": [
                        _facet_option("usd", "US dollars"),
                        _facet_option("eur", "euros"),
                    ],
                    "truncated": False,
                    "source": "observed",
                },
            ]
        ),
    )

    digest = render_digest(envelope)

    assert (
        "provenance options for get_data: oecd_dac_2a (OECD DAC table 2a), "
        "oecd_dac_1 (OECD DAC table 1), unesco_uis (UNESCO UIS), and 2 more."
    ) in digest
    assert "unit options for get_data: usd (US dollars), eur (euros)." in digest


def test_definite_digest_facet_options_name_the_raw_wire_axis_and_carry_the_ref() -> None:
    envelope = _envelope(
        "definite",
        spec=_spec(
            facet_options=[
                {
                    "axis": "provenance",
                    "options": [_facet_option("oecd_dac_2a", "OECD DAC table 2a")],
                    "truncated": False,
                    "source": "observed",
                }
            ]
        ),
    )

    digest = render_digest(envelope)

    assert "oecd_dac_2a" in digest
    assert "source options" not in digest


def test_definite_digest_facet_axis_line_notes_truncation_with_no_local_remainder() -> None:
    envelope = _envelope(
        "definite",
        spec=_spec(
            facet_options=[
                {
                    "axis": "provenance",
                    "options": [
                        _facet_option("oecd_dac_2a", "OECD DAC table 2a"),
                        _facet_option("oecd_dac_1", "OECD DAC table 1"),
                        _facet_option("unesco_uis", "UNESCO UIS"),
                    ],
                    "truncated": True,
                    "source": "observed",
                }
            ]
        ),
    )

    digest = render_digest(envelope)

    assert (
        "provenance options for get_data: oecd_dac_2a (OECD DAC table 2a), "
        "oecd_dac_1 (OECD DAC table 1), unesco_uis (UNESCO UIS), and more."
    ) in digest


def test_definite_digest_facet_option_falls_back_to_dcid_when_no_label() -> None:
    envelope = _envelope(
        "definite",
        spec=_spec(
            facet_options=[
                {
                    "axis": "measurementMethod",
                    "options": [{"value": {"ref": {"ref": "dcid:CensusACS1yr"}}}],
                    "truncated": False,
                    "source": "observed",
                }
            ]
        ),
    )

    digest = render_digest(envelope)

    assert "measurementMethod options for get_data: dcid:CensusACS1yr." in digest


def test_definite_digest_caps_facet_axis_lines_at_three_with_a_tail() -> None:
    axes = [
        {
            "axis": axis,
            "options": [_facet_option(f"{axis}_dcid", f"{axis} label")],
            "truncated": False,
            "source": "observed",
        }
        for axis in [
            "provenance",
            "unit",
            "measurementMethod",
            "observationPeriod",
            "scalingFactor",
        ]
    ]
    envelope = _envelope("definite", spec=_spec(facet_options=axes))

    digest = render_digest(envelope)

    assert "provenance options for get_data:" in digest
    assert "unit options for get_data:" in digest
    assert "measurementMethod options for get_data:" in digest
    assert "observationPeriod options for get_data:" not in digest
    assert "scalingFactor options for get_data:" not in digest
    assert "and 2 more axes with options." in digest


def test_definite_digest_omits_axis_tail_when_three_or_fewer_axes() -> None:
    axes = [
        {
            "axis": axis,
            "options": [_facet_option(f"{axis}_dcid", f"{axis} label")],
            "truncated": False,
            "source": "observed",
        }
        for axis in ["provenance", "unit", "measurementMethod"]
    ]
    envelope = _envelope("definite", spec=_spec(facet_options=axes))

    digest = render_digest(envelope)

    assert "more axes with options" not in digest


def test_definite_digest_omits_facet_options_line_when_absent() -> None:
    envelope = _envelope("definite", spec=_spec())

    digest = render_digest(envelope)

    assert "options:" not in digest


def test_candidates_digest_never_renders_facet_options() -> None:
    envelope = _envelope(
        "candidates",
        candidates={
            "ordering": "intent_rank",
            "max_candidates": 24,
            "specs": [
                _spec(
                    spec_id="spec_cand",
                    facet_options=[
                        {
                            "axis": "provenance",
                            "options": [_facet_option("oecd_dac_2a", "OECD DAC table 2a")],
                            "truncated": False,
                            "source": "observed",
                        }
                    ],
                )
            ],
            "truncated": False,
        },
    )

    digest = render_digest(envelope)

    assert "options:" not in digest


def test_multi_part_envelope_labels_each_block_by_its_variable_text() -> None:
    envelope = _envelope(
        "definite",
        parts=[
            _part("definite", variable_text="health spending", spec=_spec(spec_id="spec_1")),
            _part(
                "definite",
                variable_text="malaria grants",
                spec=_spec(spec_id="spec_2", sentence="second ask"),
            ),
        ],
    )

    digest = render_digest(envelope)

    assert digest.index('"health spending"') < digest.index("spec_1")
    assert digest.index('"malaria grants"') < digest.index("spec_2")


def test_single_part_envelope_renders_with_no_part_label() -> None:
    envelope = _envelope("definite", spec=_spec())

    digest = render_digest(envelope)

    assert "Part 1" not in digest


_COVERAGE = [
    {"kind": "text", "segment": "coverage", "text": "Answered 1 of 2 parts."},
    {"kind": "text", "segment": "coverage", "text": " "},
    {"kind": "text", "segment": "heading", "text": "Health spending", "part_index": 0},
]

# An unowned heading token first, as a refusal frame two parts share puts it.
_SHARED_HEADING = [
    {"kind": "text", "segment": "heading", "text": "No match for "},
    {"kind": "text", "segment": "heading", "text": "health spending", "part_index": 0},
]


@pytest.mark.parametrize(
    ("sentence_plan", "leading_line"),
    [(_COVERAGE, "Answered 1 of 2 parts."), (None, None), (_SHARED_HEADING, None)],
    ids=["coverage", "no_plan", "unowned_heading"],
)
def test_leading_line_is_the_answered_line(
    sentence_plan: list[dict[str, Any]] | None, leading_line: str | None
) -> None:
    """A mixed envelope leads with Search's coverage tokens. Without them, whatever its
    plan opens with, the digest opens with the first part's block."""
    envelope = _envelope(
        "mixed",
        sentence_plan=sentence_plan,
        parts=[
            _part("definite", variable_text="health spending", spec=_spec(spec_id="spec_1")),
            _part(
                "no_data",
                variable_text="malaria grants",
                no_data={"reason": "no_observations", "category": "absent"},
            ),
        ],
    )

    first_block = render_digest(envelope).split("\n\n")[0]

    if leading_line is None:
        assert first_block.startswith('Part 1: "health spending"')
    else:
        assert first_block == leading_line


def test_part_own_warning_renders_inside_its_block() -> None:
    envelope = _envelope(
        "definite",
        variable_text="malaria grants",
        warnings=[
            {
                "code": "role_not_resolved",
                "severity": "warning",
                "message": "role could not be resolved",
            }
        ],
        spec=_spec(),
    )

    digest = render_digest(envelope)

    assert "role could not be resolved" in digest


def test_diagnostics_warning_renders_as_a_trailer_with_spec_id() -> None:
    envelope = _envelope(
        "no_data",
        warnings=[{"code": "x", "severity": "warning", "message": "a note", "spec_id": "spec_1"}],
        no_data={"reason": "no_observations", "category": "absent"},
    )

    digest = render_digest(envelope)

    assert "a note" in digest
    assert "spec_1" in digest


def test_no_parts_renders_a_minimal_digest() -> None:
    envelope = _envelope("definite", parts=[])

    digest = render_digest(envelope)

    assert "No parts were returned" in digest


def test_unknown_part_status_renders_a_minimal_block() -> None:
    envelope = _envelope("mixed", parts=[_part("something_new")])

    digest = render_digest(envelope)

    assert "something_new" in digest


@pytest.mark.parametrize(
    ("stat_vars", "counted"),
    [
        # A family whose membership settled.
        (
            {"member_count": 1_454, "member_count_exact": True, "served_count": 1_454},
            "1,454 StatVars",
        ),
        # A recall sample: the members Search confirmed, a floor on the family.
        (
            {"member_count": 5, "member_count_exact": False, "served_count": None},
            "at least 5 StatVars",
        ),
        ({"member_count": 1, "member_count_exact": True, "served_count": 1}, "1 StatVar"),
    ],
    ids=["settled-family", "recall-sample", "one"],
)
def test_a_spec_states_its_measure_count_and_axes_and_names_no_member(
    stat_vars: dict[str, Any], counted: str
) -> None:
    """The digest counts the members and names none."""
    spec = _spec()
    spec["resolved"]["stat_vars"] = stat_vars
    spec["resolved"]["shape"] = {"label": "population"}
    spec["resolved"]["slots"] = [
        {
            "key": {"property": {"ref": "gender", "label": "gender"}, "label": "sex"},
            "binding": {"kind": "unbound"},
        },
        {
            "key": {"property": {"ref": "age", "label": "age"}, "label": "age"},
            "binding": {"kind": "value", "value": {"ref": {"ref": "Years15Onwards"}}},
        },
        {
            "key": {"property": {"ref": "religion", "label": "religion"}},
            "binding": {"kind": "unbound"},
        },
        {"key": {"axis": "where", "label": "place"}, "binding": {"kind": "unbound"}},
    ]
    spec["resolved"]["inapplicable_slots"] = ["religion"]
    envelope = _envelope("definite", spec=spec)

    digest = render_digest(envelope)

    assert f"Measure: population · {counted} varying by sex." in digest
    assert "Count_Person" not in digest
    assert "group" not in digest


@pytest.mark.parametrize(
    ("measure_label", "alternatives", "line"),
    [
        # Counts first, then prices, whatever the wire order.
        (
            "gross disbursements",
            [
                {"definition": {"spec_id": "spec_c"}, "label": "constant prices", "axis": "unit"},
                {"definition": {"spec_id": "spec_n"}, "label": "net disbursements", "axis": "what"},
                {"definition": {"spec_id": "spec_k"}, "label": "commitments", "axis": "what"},
            ],
            "Counted as gross disbursements; other counts: net disbursements [spec_n], "
            "commitments [spec_k]; other prices: constant prices [spec_c].",
        ),
        # A served partition whose siblings are all unlabelled publishes [].
        ("gross disbursements", [], None),
        # An entry with no spec_id names nothing get_data could take.
        ("gross disbursements", [{"definition": {}, "label": "net", "axis": "what"}], None),
        # No measure family: both fields are null.
        (None, None, None),
    ],
    ids=["counts-and-prices", "no-sibling", "no-spec-id", "no-family"],
)
def test_a_definite_spec_names_its_elected_measure_and_each_sibling(
    measure_label: str | None, alternatives: list[dict[str, Any]] | None, line: str | None
) -> None:
    spec = _spec()
    spec["resolved"]["measure_label"] = measure_label
    spec["resolved"]["alternative_measures"] = alternatives

    digest = render_digest(_envelope("definite", spec=spec))

    if line is None:
        assert "Counted as" not in digest
    else:
        assert line in digest.splitlines()


def _scope_spec() -> dict[str, Any]:
    """A spec carrying the recorded US health aid to Africa `PartScope` as served."""
    part_scope = json.loads((_FIXTURES / "scope_choices_us_health_africa.json").read_text())
    spec = _spec()
    spec["resolved"]["scope_choices"] = part_scope
    return spec


def test_each_scope_choice_renders_one_line_with_its_active_option_provenance_and_others() -> None:
    digest = render_digest(_envelope("definite", spec=_scope_spec()))

    lines = [line for line in digest.splitlines() if line.startswith("Scope choice")]
    assert lines == [
        "Scope choice sector: sector:120 (explicit); 1 other option available [sector:120+130].",
        "Scope choice region: region:members_only (carried); "
        "1 other option available [region:members_and_residuals].",
        # measure_basis:net is unavailable, so only grant equivalents are offered.
        "Scope choice measure_basis: measure_basis:gross (default); "
        "1 other option available [measure_basis:grant_equivalent].",
    ]


def test_a_scope_choice_with_no_other_option_says_so_and_a_partial_option_is_offered() -> None:
    spec = _scope_spec()
    choices = spec["resolved"]["scope_choices"]["choices"]
    for option in choices[0]["options"]:
        option["availability"] = {"status": "unavailable", "code": "conflicts_with", "detail": {}}
    choices[1]["options"][0]["availability"] = {"status": "partial", "years": [[2011, 2024]]}

    lines = [
        line
        for line in render_digest(_envelope("definite", spec=spec)).splitlines()
        if line.startswith("Scope choice")
    ]

    assert lines[0] == "Scope choice sector: sector:120 (explicit); no other option available."
    assert "[region:members_and_residuals]" in lines[1]


def test_the_structured_answer_carries_the_scope_choices_of_a_definite_spec() -> None:
    answer = render_answer(_envelope("definite", spec=_scope_spec()))

    assert answer["parts"][0]["specs"][0]["scope_choices"][1] == {
        "axis": "region",
        "provenance": "carried",
        "active": {"option_id": "region:members_only", "label": "region:members_only"},
        "other_options": ["region:members_and_residuals"],
    }


def _inclusive_request_served_bilateral() -> dict[str, Any]:
    """The scope spec with a provider choice served bilateral, where the user's words asked
    for bilateral plus imputed multilateral and the imputed table covers only 2011 to 2024."""
    spec = _scope_spec()
    scope = spec["resolved"]["scope_choices"]
    scope["choices"].append(
        {
            "id": f"{scope['id']}:provider",
            "axis": "provider",
            "rule": "test_provider_convention",
            "provenance": "default",
            "active": "provider:bilateral",
            "options": [
                {
                    "option_id": "provider:bilateral",
                    "label": "bilateral",
                    "binding": {"kind": "provider", "scope": "bilateral"},
                    "availability": {"status": "available", "detail": {}},
                },
                {
                    "option_id": "provider:inclusive",
                    "label": "bilateral plus imputed multilateral",
                    "binding": {"kind": "provider", "scope": "inclusive"},
                    "availability": {
                        "status": "partial",
                        "years": [[2011, 2024]],
                        "code": "imputed_coverage",
                        "detail": {},
                    },
                },
            ],
        }
    )
    scope["dropped_entries"] = [
        {
            "entry": {
                "axis": "provider",
                "option_id": "provider:inclusive",
                "provenance": "explicit",
                "origin": {
                    "kind": "text",
                    "text": "bilateral plus imputed multilateral",
                    "start": 0,
                    "end": 35,
                },
            },
            "code": "stated_choice_not_applied",
            "cause": "imputed_coverage",
        }
    ]
    return spec


def test_a_dropped_entry_states_the_option_asked_for_and_why_in_text_and_structured_content() -> (
    None
):
    envelope = _envelope("definite", spec=_inclusive_request_served_bilateral())

    lines = render_digest(envelope).splitlines()
    answer = render_answer(envelope)["parts"][0]["specs"][0]

    assert (
        "Scope choice provider not applied: asked for bilateral plus imputed multilateral "
        "(explicit); served bilateral, because imputed multilateral aid covers only some "
        "years (2011 to 2024)."
    ) in lines
    assert answer["dropped_scope_entries"] == [
        {
            "axis": "provider",
            "requested": {
                "option_id": "provider:inclusive",
                "label": "bilateral plus imputed multilateral",
            },
            "provenance": "explicit",
            "served": {"option_id": "provider:bilateral", "label": "bilateral"},
            "cause": "imputed_coverage",
            "reason": "imputed multilateral aid covers only some years (2011 to 2024)",
        }
    ]


@pytest.mark.parametrize(
    ("cause", "because"),
    [
        ("axis_absent", "this answer has no provider choice"),
        ("cause_from_a_later_release", "Search could not apply it [cause_from_a_later_release]"),
    ],
    ids=["axis-absent", "unknown-cause"],
)
def test_a_dropped_entry_on_an_axis_the_part_lacks_names_its_option_id_and_no_served_option(
    cause: str, because: str
) -> None:
    """A cause this release has no words for says only that Search could not
    apply the entry, never that it conflicts with another choice."""
    spec = _scope_spec()
    spec["resolved"]["scope_choices"]["dropped_entries"] = [
        {
            "entry": {
                "axis": "provider",
                "option_id": "provider:inclusive",
                "provenance": "carried",
                "origin": {"kind": "choice", "choice_id": "earlier:provider"},
            },
            "cause": cause,
        }
    ]

    envelope = _envelope("definite", spec=spec)

    assert (
        "Scope choice provider not applied: asked for provider:inclusive (carried), "
        f"because {because}."
    ) in render_digest(envelope).splitlines()
    dropped = render_answer(envelope)["parts"][0]["specs"][0]["dropped_scope_entries"]
    assert dropped[0]["served"] is None


@pytest.mark.parametrize(
    ("availability", "because"),
    [
        (
            {
                "status": "unavailable",
                "code": "conflicts_with",
                "detail": {
                    "entry": "measure_basis:net",
                    "cause": "imputed_gross_disbursements_only",
                },
            },
            "it conflicts with another choice [imputed_gross_disbursements_only]",
        ),
        (
            {"status": "available", "detail": {}},
            "Search could not apply it [imputed_gross_disbursements_only]",
        ),
        (
            {"status": "unavailable", "code": "conflicts_with", "detail": "measure_basis:net"},
            "Search could not apply it [imputed_gross_disbursements_only]",
        ),
    ],
    ids=["declared-conflict", "no-conflict-stated", "detail-not-a-mapping"],
)
def test_a_dropped_entry_a_declared_conflict_kept_out_reads_as_a_conflict(
    availability: dict[str, Any], because: str
) -> None:
    """An entry dropped under a declared conflict's own cause, which the option
    it asked for names as the conflict blocking it, reads as a conflict; the
    same cause on an option naming no conflict, or naming one in a `detail`
    that is no mapping, says only that Search could not apply it."""
    spec = _inclusive_request_served_bilateral()
    scope = spec["resolved"]["scope_choices"]
    (provider,) = [c for c in scope["choices"] if c["axis"] == "provider"]
    provider["options"][1]["availability"] = availability
    scope["dropped_entries"][0]["cause"] = "imputed_gross_disbursements_only"

    envelope = _envelope("definite", spec=spec)

    assert (
        "Scope choice provider not applied: asked for bilateral plus imputed multilateral "
        f"(explicit); served bilateral, because {because}."
    ) in render_digest(envelope).splitlines()
    dropped = render_answer(envelope)["parts"][0]["specs"][0]["dropped_scope_entries"]
    assert dropped[0]["reason"] == because


def test_no_dropped_entry_adds_no_line_and_no_answer_field() -> None:
    envelope = _envelope("definite", spec=_scope_spec())

    assert "not applied" not in render_digest(envelope)
    assert "dropped_scope_entries" not in render_answer(envelope)["parts"][0]["specs"][0]


def test_a_spec_without_scope_choices_renders_no_scope_line_and_no_answer_field() -> None:
    envelope = _envelope("definite", spec=_spec())

    assert "Scope choice" not in render_digest(envelope)
    assert "scope_choices" not in render_answer(envelope)["parts"][0]["specs"][0]


def test_a_spec_with_no_elected_stat_vars_renders_no_measure_line() -> None:
    assert "StatVar" not in render_digest(_envelope("definite", spec=_spec()))


# Digest rendering, candidates status


def test_candidates_digest_lists_each_spec_and_notes_truncation() -> None:
    envelope = _envelope(
        "candidates",
        candidates={
            "ordering": "intent_rank",
            "max_candidates": 5,
            "specs": [_spec(spec_id="spec_a"), _spec(spec_id="spec_b")],
            "truncated": True,
        },
    )

    digest = render_digest(envelope)

    assert "2 candidate interpretations" in digest
    assert "spec_id is the handle for retrieving the data" in digest
    assert "spec_a" in digest
    assert "spec_b" in digest
    assert "truncated" in digest.lower() or "more candidates" in digest.lower()


def test_each_candidate_counts_its_own_stat_vars_naming_none() -> None:
    envelope = _envelope(
        "candidates",
        candidates={
            "ordering": "intent_rank",
            "max_candidates": 5,
            "specs": [
                _spec(spec_id="spec_a", member_count=1),
                _spec(spec_id="spec_b"),
            ],
            "truncated": False,
        },
    )

    digest = render_digest(envelope)

    assert (
        '1. "malaria grants from France to Togo, 2015 to 2024"  [spec_a]\n'
        "   confidence high · has data\n"
        "   1 StatVar.\n\n"
        "2. "
    ) in digest
    assert digest.count("StatVar") == 1
    assert "Count_Person" not in digest
    assert "People" not in digest


def test_candidates_digest_renders_bare_coverage() -> None:
    envelope = _envelope(
        "candidates",
        candidates={
            "ordering": "broadest_first",
            "max_candidates": 5,
            "specs": [_spec(coverage={"kind": "bare", "has_data": False})],
            "truncated": False,
        },
    )

    digest = render_digest(envelope)

    assert "no data" in digest


def test_candidates_digest_at_the_cap_is_unchanged() -> None:
    specs = [_spec(spec_id=f"spec_{i}") for i in range(6)]
    envelope = _envelope(
        "candidates",
        candidates={"ordering": "intent_rank", "max_candidates": 6, "specs": specs},
    )

    digest = render_digest(envelope)

    assert "6 candidate interpretations" in digest
    for i in range(6):
        assert f"spec_{i}" in digest
    assert "more candidates" not in digest.lower()


def test_candidates_digest_beyond_the_cap_renders_only_the_first_six() -> None:
    specs = [_spec(spec_id=f"spec_{i}") for i in range(12)]
    envelope = _envelope(
        "candidates",
        candidates={"ordering": "intent_rank", "max_candidates": 12, "specs": specs},
    )

    digest = render_digest(envelope)

    for i in range(6):
        assert f"spec_{i}" in digest
    for i in range(6, 12):
        assert f"spec_{i}" not in digest


def test_candidates_digest_header_counts_what_is_shown_not_what_was_offered() -> None:
    specs = [_spec(spec_id=f"spec_{i}") for i in range(12)]
    envelope = _envelope(
        "candidates",
        candidates={"ordering": "intent_rank", "max_candidates": 12, "specs": specs},
    )

    digest = render_digest(envelope)

    assert "6 candidate interpretations" in digest
    assert "12 candidate interpretations" not in digest


def test_candidates_digest_notes_truncation_when_cut_locally_even_if_the_envelope_says_not() -> (
    None
):
    specs = [_spec(spec_id=f"spec_{i}") for i in range(12)]
    envelope = _envelope(
        "candidates",
        candidates={
            "ordering": "intent_rank",
            "max_candidates": 12,
            "specs": specs,
            "truncated": False,
        },
    )

    digest = render_digest(envelope)

    assert "More candidates exist than are shown here." in digest


# Digest rendering, no_data status


def test_no_data_unresolved_category_reads_as_unresolved() -> None:
    envelope = _envelope(
        "no_data", no_data={"reason": "entity_not_resolved", "category": "unresolved"}
    )

    digest = render_digest(envelope)

    assert "could not be resolved" in digest
    assert "entity_not_resolved" in digest


def test_no_data_absent_category_reads_as_no_data_available() -> None:
    envelope = _envelope("no_data", no_data={"reason": "no_observations", "category": "absent"})

    digest = render_digest(envelope)

    assert "no data is available" in digest
    assert "no_observations" in digest


def test_no_data_message_renders_after_the_header_and_its_absence_changes_nothing() -> None:
    no_data = {"reason": "no_observations", "category": "absent"}
    message = "No observations exist for Kenya in 1850."

    plain = render_digest(_envelope("no_data", no_data=no_data))
    stated = render_digest(
        _envelope("no_data", no_data={**no_data, "message": message, "added_later": 1})
    )

    assert message not in plain
    assert stated.split("\n\n") == [*plain.split("\n\n")[:1], message, *plain.split("\n\n")[1:]]


def test_no_data_with_nearest_real_renders_up_to_two() -> None:
    envelope = _envelope(
        "no_data",
        no_data={
            "reason": "no_observations",
            "category": "absent",
            "nearest_real": [
                {
                    **_spec(spec_id="spec_near_1"),
                    "relaxation": {"axis": "where", "dropped_value_dcids": []},
                },
                {
                    **_spec(spec_id="spec_near_2"),
                    "relaxation": {"axis": "when", "dropped_value_dcids": []},
                },
            ],
        },
    )

    digest = render_digest(envelope)

    assert "spec_near_1" in digest
    assert "spec_near_2" in digest
    assert "relaxed where" in digest
    assert "relaxed when" in digest


def test_a_nearest_real_spec_counts_its_stat_vars_naming_none() -> None:
    envelope = _envelope(
        "no_data",
        no_data={
            "reason": "no_observations",
            "category": "absent",
            "nearest_real": [
                {
                    **_spec(spec_id="spec_near", member_count=1),
                    "relaxation": {"axis": "where", "dropped_value_dcids": []},
                }
            ],
        },
    )

    digest = render_digest(envelope)

    assert "[spec_near] — relaxed where\n1 StatVar." in digest
    assert "Count_Person" not in digest


def test_no_data_without_nearest_real_omits_the_section() -> None:
    envelope = _envelope("no_data", no_data={"reason": "no_observations", "category": "absent"})

    digest = render_digest(envelope)

    assert "Nearby" not in digest


@pytest.mark.parametrize(
    ("checked", "line"), [(1, "Checked 1 StatVar."), (5, "Checked 5 StatVars.")]
)
def test_no_data_counts_the_stat_vars_it_checked_naming_none(checked: int, line: str) -> None:
    envelope = _envelope(
        "no_data",
        no_data={
            "reason": "no_observations",
            "category": "absent",
            "elected_count": checked,
        },
    )

    assert line in render_digest(envelope)


def test_no_data_without_elected_count_omits_the_line() -> None:
    envelope = _envelope("no_data", no_data={"reason": "no_observations", "category": "absent"})

    digest = render_digest(envelope)

    assert "Checked" not in digest


def test_no_data_flagged_match_with_gate_renders_the_gate_sentence() -> None:
    envelope = _envelope(
        "no_data",
        no_data={
            "reason": "no_observations",
            "category": "absent",
            "flagged_match": _spec(spec_id="spec_flagged"),
            "gate": "min_countries_5",
        },
    )

    digest = render_digest(envelope)

    assert "spec_flagged" in digest
    assert "min_countries_5" in digest
    assert "same-variable match exists" in digest
    assert "failed quality gate" in digest


def test_no_data_role_incompatible_names_the_role_mismatch_with_no_gate_wording() -> None:
    envelope = _envelope(
        "no_data",
        no_data={
            "reason": "role_incompatible",
            "category": "absent",
            "flagged_match": _spec(spec_id="spec_flagged"),
            "gate": None,
        },
    )

    digest = render_digest(envelope)

    assert "spec_flagged" in digest
    assert "declared roles cannot support this request" in digest
    assert "donor/recipient direction" not in digest
    assert "quality gate" not in digest


def test_no_data_qualifier_not_bound_preserves_each_disposition_in_query_order() -> None:
    envelope = _envelope(
        "no_data",
        no_data={
            "reason": "qualifier_not_bound",
            "category": "absent",
            "flagged_match": _spec(spec_id="spec_flagged"),
            "gate": None,
            "unbound_qualifiers": [
                {"text": "malaria", "disposition": "unbound"},
                {"text": "untied", "disposition": "no_vocabulary"},
            ],
        },
    )

    digest = render_digest(envelope)

    assert "spec_flagged" in digest
    assert "does not satisfy every requested qualifier" in digest
    assert 'It shows: "malaria grants from France to Togo, 2015 to 2024"' in digest
    assert "quality gate" not in digest
    assert '"malaria" names a category the search could not apply.' in digest
    assert '"untied" is not a distinction this data records.' in digest
    assert digest.index('"malaria"') < digest.index('"untied"')


@pytest.mark.parametrize(
    ("texts", "named"),
    [
        ([], "every phrase of this ask"),
        (["global"], '"global"'),
        (["HIV", "AIDS", "HIV"], '"HIV" and "AIDS"'),
        (["HIV", "AIDS", "malaria"], '"HIV", "AIDS", and "malaria"'),
    ],
)
def test_no_data_qualifier_not_bound_with_a_flagged_match_is_headed_as_a_refusal(
    texts: list[str], named: str
) -> None:
    """The data exists without the phrase, so the part says search would not
    apply it."""
    envelope = _envelope(
        "no_data",
        no_data={
            "reason": "qualifier_not_bound",
            "category": "absent",
            "flagged_match": _spec(spec_id="spec_flagged"),
            "gate": None,
            "unbound_qualifiers": [{"text": text, "disposition": "unbound"} for text in texts],
        },
    )

    digest = render_digest(envelope)

    assert "no data is available" not in digest
    assert (
        f"Search found matching data but could not apply {named} to it, so this part has "
        "no answer (reason: qualifier_not_bound)."
    ) in digest


def test_no_data_qualifier_not_bound_without_a_flagged_match_keeps_the_no_data_heading() -> None:
    envelope = _envelope(
        "no_data",
        no_data={
            "reason": "qualifier_not_bound",
            "category": "absent",
            "unbound_qualifiers": [{"text": "global", "disposition": "unbound"}],
        },
    )

    assert "Resolved, but no data is available" in render_digest(envelope)


def test_no_data_qualifier_not_bound_without_unbound_qualifiers_omits_the_line() -> None:
    envelope = _envelope(
        "no_data",
        no_data={
            "reason": "qualifier_not_bound",
            "category": "absent",
            "flagged_match": _spec(spec_id="spec_flagged"),
            "gate": None,
        },
    )

    digest = render_digest(envelope)

    assert "names a category" not in digest
    assert "not a distinction" not in digest


# Digest rendering, unsupported status


def test_unsupported_facet_binding_names_the_field_and_gives_a_concrete_step() -> None:
    envelope = _envelope(
        "unsupported",
        unsupported={
            "code": "facet_binding_unsupported",
            "pointer": "/identity/facets/provenance",
            "detail": "no free-text binding exists for the provenance axis",
            "definition": {"identity": {"facets": {"provenance": {"state": "value", "ref": "x"}}}},
        },
    )

    digest = render_digest(envelope)

    assert "No free-text binding exists for the provenance axis." in digest
    assert "the source field" in digest
    assert "Drop `source` and retry." in digest
    assert "identity/facets/provenance" not in digest
    assert "{" not in digest


def test_unsupported_other_code_with_a_recognised_field_gets_a_generic_step() -> None:
    envelope = _envelope(
        "unsupported",
        unsupported={
            "code": "unit_request_unsupported",
            "pointer": "/identity/facets/unit",
            "detail": "PPP terms are not representable",
            "definition": {"identity": {}},
        },
    )

    digest = render_digest(envelope)

    assert "PPP terms are not representable" in digest
    assert "the unit field" in digest
    assert "Drop the field named `unit` and retry." in digest
    assert "identity/facets/unit" not in digest


def test_unsupported_pointer_resolution_never_raises_on_a_bad_pointer() -> None:
    envelope = _envelope(
        "unsupported",
        unsupported={
            "code": "unit_request_unsupported",
            "pointer": "/does/not/exist",
            "detail": "PPP terms are not representable",
            "definition": {"identity": {}},
        },
    )

    digest = render_digest(envelope)

    assert "PPP terms are not representable" in digest


def test_unsupported_pointer_with_a_trailing_slash_names_no_field() -> None:
    envelope = _envelope(
        "unsupported",
        unsupported={
            "code": "unit_request_unsupported",
            "pointer": "/identity/facets/",
            "detail": "no axis was given",
        },
    )

    digest = render_digest(envelope)

    assert digest == "Not supported: No axis was given."


def test_unsupported_with_no_pointer_omits_the_field_sentence() -> None:
    envelope = _envelope(
        "unsupported",
        unsupported={"code": "unit_request_unsupported", "detail": "no pointer was given"},
    )

    digest = render_digest(envelope)

    assert digest == "Not supported: No pointer was given."


def test_unsupported_detail_is_normalised_into_a_sentence_with_a_field() -> None:
    envelope = _envelope(
        "unsupported",
        unsupported={
            "code": "unit_request_unsupported",
            "pointer": "/identity/facets/unit",
            "detail": "the requested unit's currency is not carried",
        },
    )

    digest = render_digest(envelope)

    assert "The requested unit's currency is not carried." in digest


def test_unsupported_detail_normalisation_does_not_double_a_trailing_stop() -> None:
    envelope = _envelope(
        "unsupported",
        unsupported={"code": "unit_request_unsupported", "detail": "already a full sentence."},
    )

    digest = render_digest(envelope)

    assert digest == "Not supported: Already a full sentence."


def test_unsupported_part_renders_only_caller_input_warnings() -> None:
    part = _part(
        "unsupported",
        warnings=[
            {
                "code": "near_neighbor_ambiguous",
                "severity": "info",
                "message": "an unrelated disambiguation note from an earlier step",
            },
            {
                "code": "default_not_declared",
                "severity": "info",
                "message": "the unit fell back to a deployment default",
            },
        ],
        unsupported={
            "code": "facet_binding_unsupported",
            "pointer": "/identity/facets/provenance",
            "detail": "no free-text binding exists for the provenance axis",
        },
    )
    envelope = _envelope("unsupported", parts=[part])

    digest = render_digest(envelope)

    assert "the unit fell back to a deployment default" in digest
    assert "an unrelated disambiguation note from an earlier step" not in digest


def test_no_data_part_keeps_every_warning_regardless_of_code() -> None:
    part = _part(
        "no_data",
        warnings=[
            {
                "code": "near_neighbor_ambiguous",
                "severity": "info",
                "message": "a disambiguation note that is not a caller-input code",
            }
        ],
        no_data={"reason": "no_observations", "category": "absent"},
    )
    envelope = _envelope("no_data", parts=[part])

    digest = render_digest(envelope)

    assert "a disambiguation note that is not a caller-input code" in digest


# No observation values ever leak


def test_no_observation_values_leak_from_a_malicious_envelope() -> None:
    malicious_observations = {
        "series": [
            {
                "stat_var": {"ref": "MalariaGrantDisbursements"},
                "entities": [],
                "points": [{"date": "2024", "value": 999999999.0}],
                "total_obs": 1,
            }
        ],
        "total_obs": 1,
        "truncated": False,
    }
    envelope = _envelope(
        "definite",
        spec=_spec(observations=malicious_observations),
    )

    digest = render_digest(envelope)

    assert "999999999" not in digest
    assert "MalariaGrantDisbursements" not in digest


# Skills


def _with_skills(spec: dict[str, Any], *skill_ids: str) -> dict[str, Any]:
    spec["resolved"]["skills"] = list(skill_ids)
    return spec


def _skill(skill_id: str, description: str) -> dict[str, Any]:
    return {
        "id": skill_id,
        "name": skill_id,
        "description": description,
        "content_version": None,
    }


def test_skills_render_by_id_and_description_between_the_blocks_and_the_warnings() -> None:
    envelope = _envelope(
        "definite",
        warnings=[{"code": "x", "severity": "warning", "message": "a trailing note"}],
        spec=_with_skills(_spec(), "health-financing"),
    )
    envelope["skills"] = {
        "health-financing": _skill("health-financing", "Health aid purpose codes.")
    }

    digest = render_digest(envelope)

    skills_section = (
        "Skills for these results; get_skill loads one by id:\n"
        "- health-financing: Health aid purpose codes."
    )
    assert skills_section in digest
    assert digest.index("[spec_1]") < digest.index(skills_section)
    assert digest.index(skills_section) < digest.index("a trailing note")


def test_skills_follow_first_appearance_across_candidates_most_specific_first() -> None:
    envelope = _envelope(
        "candidates",
        candidates={
            "specs": [
                _with_skills(_spec("spec_a"), "malaria", "oda-basics"),
                _with_skills(_spec("spec_b"), "oda-basics", "deflators"),
            ]
        },
    )
    envelope["skills"] = {
        "deflators": _skill("deflators", "d"),
        "oda-basics": _skill("oda-basics", "o"),
        "malaria": _skill("malaria", "m"),
    }

    digest = render_digest(envelope)

    assert "- malaria: m\n- oda-basics: o\n- deflators: d" in digest


def test_skills_named_by_nearest_real_specs_render_on_a_no_data_part() -> None:
    envelope = _envelope(
        "no_data",
        no_data={
            "reason": "no_observations",
            "category": "absent",
            "nearest_real": [_with_skills(_spec("spec_near"), "oda-basics")],
        },
    )
    envelope["skills"] = {"oda-basics": _skill("oda-basics", "What counts as ODA.")}

    assert "- oda-basics: What counts as ODA." in render_digest(envelope)


def test_a_skill_no_walked_spec_names_follows_the_named_ones_sorted() -> None:
    envelope = _envelope("definite", spec=_with_skills(_spec(), "zeta"))
    envelope["skills"] = {
        "zeta": _skill("zeta", "z"),
        "beta": _skill("beta", "b"),
        "alpha": _skill("alpha", "a"),
    }

    assert "- zeta: z\n- alpha: a\n- beta: b" in render_digest(envelope)


def test_skills_cap_at_five_and_count_the_rest() -> None:
    skill_ids = [f"skill-{index}" for index in range(7)]
    envelope = _envelope("definite", spec=_with_skills(_spec(), *skill_ids))
    envelope["skills"] = {skill_id: _skill(skill_id, "x") for skill_id in skill_ids}

    digest = render_digest(envelope)

    assert "- skill-4: x\nand 2 more skills." in digest
    assert "skill-5" not in digest


def test_one_skill_past_the_cap_is_counted_in_the_singular() -> None:
    skill_ids = [f"skill-{index}" for index in range(6)]
    envelope = _envelope("definite", spec=_with_skills(_spec(), *skill_ids))
    envelope["skills"] = {skill_id: _skill(skill_id, "x") for skill_id in skill_ids}

    assert render_digest(envelope).endswith("- skill-4: x\nand 1 more skill.")


def test_a_skill_description_renders_on_one_line_cut_at_160_characters() -> None:
    description = "Line one of the description.\n\n  " + "word " * 60
    envelope = _envelope("definite", spec=_with_skills(_spec(), "long"))
    envelope["skills"] = {"long": _skill("long", description)}

    line = next(line for line in render_digest(envelope).splitlines() if line.startswith("- long"))

    rendered = line.removeprefix("- long: ")
    assert len(rendered) <= 160
    assert rendered.startswith("Line one of the description. word word")
    assert rendered.endswith("…")


def test_an_empty_or_absent_skills_map_renders_no_skills_section() -> None:
    absent = _envelope("definite", spec=_spec())
    empty = {**absent, "skills": {}}

    assert render_digest(absent) == render_digest(empty)
    assert "Skills for these results" not in render_digest(empty)


# The resolve answer's kinds: a catalogue or declined answer carries no parts


def _kind_envelope(kind: str, **fields: Any) -> dict[str, Any]:
    """A `catalogue` or `declined` answer as `POST /api/search/resolve` serves it after
    the resolve door: `kind`, `scope`, and for `declined` a `reason`; no parts."""
    return {
        "schema_version": 1,
        "kind": kind,
        "query_echo": {"entry_path": "raw_text", "extract_skipped": False},
        "diagnostics": {"engine_build": "test", "trace_id": "trace-1", "warnings": []},
        "scope": {},
        **fields,
    }


def test_a_catalogue_answer_states_searchs_reading_and_points_at_sources() -> None:
    text = render_digest(_kind_envelope("catalogue"))

    assert text == (
        "Search read this question as asking what data DFO holds, not as naming data to "
        "retrieve, so it resolved no parts. The `sources` tool lists what DFO holds and "
        "documents each dataset."
    )


def test_a_catalogue_answer_names_the_confirmed_country() -> None:
    envelope = _kind_envelope(
        "catalogue", scope={"place": {"ref": "country/KEN", "label": "Kenya"}}
    )

    text = render_digest(envelope)

    assert "so it resolved no parts. It named Kenya. The `sources` tool" in text
    assert render_answer(envelope) == {
        "kind": "catalogue",
        "status": None,
        "parts": [],
        "scope": {"place": {"ref": "country/KEN", "label": "Kenya"}},
    }


def test_a_declined_answer_states_its_reason_in_plain_words() -> None:
    out_of_scope = _kind_envelope("declined", reason="out_of_scope")
    judgement = _kind_envelope("declined", reason="requires_analysis")

    assert render_digest(out_of_scope) == (
        "Search declined this question: it is not about data DFO holds; it resolved no parts."
    )
    assert render_digest(judgement) == (
        "Search declined this question: it asks for a judgement and names no data to show; "
        "it resolved no parts."
    )
    assert render_answer(out_of_scope) == {
        "kind": "declined",
        "status": None,
        "parts": [],
        "reason": "out_of_scope",
    }


def test_a_declined_answer_keeps_the_diagnostics_warnings_as_its_trailer() -> None:
    envelope = _kind_envelope("declined", reason="out_of_scope")
    envelope["diagnostics"]["warnings"] = [{"code": "slow", "message": "Extract ran slowly."}]

    text = render_digest(envelope)

    assert text.endswith("it resolved no parts.\n\nExtract ran slowly.")
    assert render_answer(envelope)["warnings"] == [
        {"code": "slow", "message": "Extract ran slowly."}
    ]


def test_a_kind_this_service_does_not_know_names_itself() -> None:
    assert render_digest(_kind_envelope("forecast")) == (
        "Search answered with kind 'forecast', which this service does not know; it carries no parts."
    )


def test_a_resolution_answer_renders_its_parts_as_before_and_states_its_kind() -> None:
    envelope = _envelope("definite", spec=_spec())
    without_kind = render_answer(envelope)
    envelope["kind"] = "resolution"

    with_kind = render_answer(envelope)

    assert "kind" not in without_kind
    assert with_kind == {**without_kind, "kind": "resolution"}
    assert render_digest(envelope) == render_digest(
        {k: v for k, v in envelope.items() if k != "kind"}
    )


# Digest rendering, sources the ask named

_IDS = "provenance/WorldBankInternationalDebtStatistics"
_CRS = "provenance/OECDCreditorReportingSystem"
_TIME_HINT = {
    "code": "time_window_defaulted",
    "severity": "warn",
    "message": "no time_text was given; the answer was resolved across every period with data.",
}


def _restatement(*sources: dict[str, Any]) -> dict[str, Any]:
    return {"variable_text": "debt service", "entities": [], "sources": list(sources)}


def test_a_named_source_reads_as_asked_for_with_the_label_the_spec_gives() -> None:
    spec = _spec(
        facets={"provenance": {"ref": _IDS, "label": "World Bank IDS", "basis": "fallback"}}
    )
    source = {"text": "IDS", "ref": _IDS, "polarity": "include"}

    digest = render_digest(_envelope("definite", spec=spec, restatement=_restatement(source)))

    assert f'Sources: asked for "IDS", read as World Bank IDS ({_IDS}).' in digest


def test_an_excluded_source_reads_as_excluded_with_its_ref_when_the_spec_has_no_label() -> None:
    spec = _spec(facets={"provenance": {"ref": "provenance/OECDDAC2A", "label": "OECD DAC2A"}})
    source = {"text": "CRS", "ref": _CRS, "polarity": "exclude"}

    digest = render_digest(_envelope("definite", spec=spec, restatement=_restatement(source)))

    assert f'Sources: excluded "CRS", read as {_CRS}.' in digest
    assert "asked for" not in digest


def test_a_source_label_comes_from_the_facet_options_too() -> None:
    options = [
        {"axis": "provenance", "options": [_facet_option(_CRS, "OECD Creditor Reporting System")]}
    ]
    source = {"text": "CRS", "ref": _CRS, "polarity": "exclude"}

    digest = render_digest(
        _envelope("definite", spec=_spec(facet_options=options), restatement=_restatement(source))
    )

    assert f'excluded "CRS", read as OECD Creditor Reporting System ({_CRS})' in digest


def test_several_sources_share_one_line() -> None:
    sources = [
        {"text": "IDS", "ref": _IDS, "polarity": "include"},
        {"text": "CRS", "ref": _CRS, "polarity": "exclude"},
    ]

    digest = render_digest(_envelope("definite", spec=_spec(), restatement=_restatement(*sources)))

    (line,) = [line for line in digest.splitlines() if line.startswith("Sources:")]
    assert line == f'Sources: asked for "IDS", read as {_IDS} · excluded "CRS", read as {_CRS}.'


def test_a_part_that_named_no_source_gets_no_sources_line() -> None:
    digest = render_digest(_envelope("definite", spec=_spec(), restatement=_restatement()))

    assert "Sources:" not in digest


# Digest rendering, source refusals

_UNRESOLVED_SOURCE = {
    "reason": "source_not_resolved",
    "category": "unresolved",
    "message": 'Could not match "XYZ statistics" to a data source. Try a publisher or '
    "dataset name, or leave the source out.",
    "source_terms": ["XYZ statistics"],
    "ambiguous_sources": [],
}


def _refused(no_data: dict[str, Any], **part_fields: Any) -> dict[str, Any]:
    """An envelope whose one `no_data` part carries `part_fields` (its own warnings, restatement)."""
    return _envelope("no_data", parts=[_part("no_data", no_data=no_data, **part_fields)])


def test_a_source_not_resolved_refusal_drops_the_defaulted_time_window_hint() -> None:
    digest = render_digest(
        _refused(
            _UNRESOLVED_SOURCE,
            warnings=[_TIME_HINT],
            restatement=_restatement({"text": "XYZ statistics", "polarity": "include"}),
        )
    )

    assert "time_text" not in digest
    assert 'Could not match "XYZ statistics" to a data source.' in digest
    assert 'Sources: asked for "XYZ statistics".' in digest


def test_an_entity_not_resolved_refusal_drops_the_defaulted_time_window_hint_too() -> None:
    digest = render_digest(
        _refused({"reason": "entity_not_resolved", "category": "unresolved"}, warnings=[_TIME_HINT])
    )

    assert "time_text" not in digest


def test_a_refusal_keeps_the_warnings_about_the_callers_input() -> None:
    warning = {"code": "time_text_not_parsed", "message": "time_text 'sometime' was not parsed."}

    digest = render_digest(_refused(_UNRESOLVED_SOURCE, warnings=[_TIME_HINT, warning]))

    assert warning["message"] in digest


def test_data_that_resolved_keeps_the_defaulted_time_window_hint() -> None:
    digest = render_digest(
        _envelope("definite", parts=[_part("definite", spec=_spec(), warnings=[_TIME_HINT])])
    )

    assert _TIME_HINT["message"] in digest


def test_a_source_ambiguous_refusal_lists_each_candidate_with_its_ref_and_kind() -> None:
    candidates = [
        {"ref": "dc/s/A", "label": "Source A", "kind": "source"},
        {"ref": "dc/d/B", "label": "Dataset B", "kind": "dataset"},
        {"ref": "provenance/C", "label": "Table C", "kind": "provenance"},
    ]
    no_data = {
        "reason": "source_ambiguous",
        "category": "unresolved",
        "message": '"ambiguous" matches more than one source. Name one of them.',
        "source_terms": ["ambiguous"],
        "ambiguous_sources": [{"text": "ambiguous", "candidates": candidates}],
    }

    digest = render_digest(_refused(no_data, warnings=[_TIME_HINT]))

    assert (
        '"ambiguous" could mean: Source A (dc/s/A, source); Dataset B (dc/d/B, dataset); '
        "Table C (provenance/C, provenance)."
    ) in digest
    assert "time_text" not in digest


def test_ambiguous_source_candidates_are_capped_with_a_count_of_the_rest() -> None:
    candidates = [{"ref": f"dc/s/{n}", "label": f"Source {n}", "kind": "source"} for n in range(8)]
    no_data = {
        "reason": "source_ambiguous",
        "category": "unresolved",
        "ambiguous_sources": [{"text": "s", "candidates": candidates}],
    }

    digest = render_digest(_envelope("no_data", no_data=no_data))

    assert "Source 5 (dc/s/5, source), and 2 more." in digest
    assert "Source 6" not in digest


_IDS_REF = "provenance/WorldBankInternationalDebtStatistics"


@pytest.mark.parametrize(
    ("facets", "line"),
    [
        (
            {"provenance": {"ref": _IDS_REF, "label": "World Bank IDS"}},
            "2 StatVars · Source codes: World Bank IDS DT.TDS.DEGG.CD, DT.TDS.DEGG.KD.",
        ),
        (
            {"provenance": {"ref": "provenance/OECDDAC1", "label": "OECD DAC1"}},
            "2 StatVars · Source codes: DT.TDS.DEGG.CD, DT.TDS.DEGG.KD.",
        ),
        ({}, "2 StatVars · Source codes: DT.TDS.DEGG.CD, DT.TDS.DEGG.KD."),
    ],
    ids=["named", "other-source-bound", "no-source-bound"],
)
def test_the_measure_line_names_the_family_source_codes(facets: dict[str, Any], line: str) -> None:
    spec = _spec(member_count=2, facets=facets)
    spec["resolved"]["stat_vars"]["source_codes"] = [
        {"provenance": _IDS_REF, "code": "DT.TDS.DEGG.CD"},
        {"provenance": _IDS_REF, "code": "DT.TDS.DEGG.KD"},
    ]

    assert line in render_digest(_envelope("definite", spec=spec))


def test_a_spec_with_no_source_codes_states_none() -> None:
    assert "Source codes" not in render_digest(_envelope("definite", spec=_spec(member_count=2)))
