"""`harness.tool_contracts.graph_envelope` against the search envelope's parts."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, get_args

import pytest
from data_mcp.digest import _dropped_cause_words
from data_mcp.search_tool import SearchContext
from harness.scope_choices import REASON_COPY, PartScope, ScopeEntry, ScopeReasonCode
from harness.tool_contracts import graph_envelope, reading_facts
from pydantic import ValidationError

_DATA_MCP_FIXTURES = Path(__file__).parents[2] / "data-mcp" / "tests" / "fixtures"


def _definite(
    spec_id: str,
    facets: dict | None = None,
    warnings: list | None = None,
    identity: dict | None = None,
    member_count: int | None = None,
    member_count_exact: bool | None = None,
    served_count: int | None = None,
    data_date_range: dict | None = None,
) -> dict:
    stat_vars: dict = {"served_count": served_count, "data_date_range": data_date_range}
    if member_count is not None:
        stat_vars["member_count"] = member_count
    if member_count_exact is not None:
        stat_vars["member_count_exact"] = member_count_exact
    resolved: dict = {
        "rendered_sentence": "life expectancy in Kenya",
        "facets": facets or {},
        "stat_vars": stat_vars,
    }
    spec: dict = {"spec_id": spec_id, "resolved": resolved}
    if identity is not None:
        spec["definition"] = {"identity": identity}
    return {
        "status": "definite",
        "variable_text": "life expectancy",
        "warnings": warnings or [],
        "spec": spec,
    }


# "Population for Guatemala since 2010", every demographic slot unbound (the
# envelope committed at
# `services/data-mcp/tests/fixtures/resolve_population_guatemala.json`).
_STANDARD_IDENTITY = {
    "shape": {"kind": "statvar_schema"},
    "entities": {"observationAbout": ["country/GTM"]},
    "slots": [
        {"property": "age", "binding": {"state": "unbound"}},
        {"property": "gender", "binding": {"state": "unbound"}},
    ],
}
_ROLE_BEARING_IDENTITY = {
    "shape": {"kind": "statvar_schema"},
    "entities": {"observationAbout": ["country/KEN"], "destination": ["country/KEN"]},
}
_PINNED_IDENTITY = {"shape": {"kind": "stat_var", "id": "Count_Person"}}


@pytest.mark.parametrize("separate", [True, False, None])
def test_part_summary_preserves_explicit_comparison_intent(separate: bool | None) -> None:
    part = _definite("spec_1")
    if separate is not None:
        part["separate"] = separate
    (summary,) = graph_envelope({"parts": [part]})["part_digests"]
    if separate is True:
        assert summary["separate"] is True
    else:
        assert "separate" not in summary


def _scope_choices() -> dict[str, Any]:
    """A part's `resolved.scope_choices` as dfo-apis writes it."""
    return json.loads((_DATA_MCP_FIXTURES / "scope_choices_us_health_africa.json").read_text())


def test_a_stored_reading_holds_the_parts_scope_choices_and_the_model_sees_none() -> None:
    part = _definite("spec_1")
    part["spec"]["resolved"]["scope_choices"] = _scope_choices()

    (summary,) = graph_envelope({"parts": [part]})["part_digests"]
    scope = reading_facts(part["spec"])["scope"]

    assert PartScope.model_validate(scope) == PartScope.model_validate(_scope_choices())
    # The identities stay as Search wrote them, so a switch posts them back unchanged.
    assert scope["base"] == _scope_choices()["base"]
    assert scope["compiled"]["identity"] == _scope_choices()["compiled"]["identity"]
    shown = {key: value for key, value in summary.items() if key != "spec_block"}
    assert "scope" not in shown
    assert "scope_choices" not in json.dumps(shown)


def test_a_field_search_adds_to_scope_choices_is_ignored_and_never_stored() -> None:
    """A field a later dfo-apis release adds, on the record or inside an
    option, leaves the part's scope choices readable and out of the stored
    reading, which a strict read still takes back."""
    added = _scope_choices()
    added["rule_digest"] = "r1"
    added["choices"][0]["options"][0]["availability"]["note"] = "new"
    part = _definite("spec_1")
    part["spec"]["resolved"]["scope_choices"] = added

    scope = reading_facts(part["spec"])["scope"]

    assert scope == PartScope.model_validate(_scope_choices()).model_dump(mode="json")
    assert PartScope.model_validate(scope) == PartScope.from_search(added)
    with pytest.raises(ValidationError):
        PartScope.model_validate(added)


def test_a_value_search_adds_to_an_open_vocabulary_degrades_only_what_holds_it() -> None:
    """A later dfo-apis release adds an axis, a reason code, a component role and
    a dropped entry's code. The record stays readable: the choice and entry on the
    unknown axis are left out, the unknown reason code reads as none, the role is
    kept as written, and the dropped entry's code follows its provenance."""
    added = _scope_choices()
    unknown_option = {
        "option_id": "channel:ngo",
        "label": "through NGOs",
        "binding": {"kind": "provider", "scope": "bilateral"},
        "availability": {"status": "available", "detail": {}},
    }
    added["choices"].append(
        {
            "id": f"{added['id']}:channel",
            "axis": "channel",
            "rule": "test_channel",
            "provenance": "explicit",
            "active": "channel:ngo",
            "options": [unknown_option],
        }
    )
    unknown_entry = {
        "axis": "channel",
        "option_id": "channel:ngo",
        "provenance": "explicit",
        "origin": {"kind": "text", "text": "NGOs", "start": 0, "end": 4},
    }
    added["entries"].append(unknown_entry)
    added["choices"][2]["options"][1]["availability"]["code"] = "netting_withdrawn"
    added["compiled"]["components"] = [{"role": "pooled", "identity": {}}]
    added["dropped_entries"] = [
        {"entry": added["entries"][0], "code": "entry_superseded", "cause": "superseded"},
        {"entry": unknown_entry, "cause": "axis_absent"},
    ]
    part = _definite("spec_1")
    part["spec"]["resolved"]["scope_choices"] = added

    stored = reading_facts(part["spec"])["scope"]

    assert stored is not None
    scope = PartScope.model_validate(stored)
    assert [choice.axis for choice in scope.choices] == ["sector", "region", "measure_basis"]
    assert [entry.axis for entry in scope.entries] == ["sector", "region"]
    assert scope.choices[2].options[1].availability.code is None
    assert [component.role for component in scope.compiled.components] == ["pooled"]
    assert [(d.entry.axis, d.code) for d in scope.dropped_entries] == [
        ("sector", "stated_choice_not_applied")
    ]


def test_scope_choices_this_release_cannot_read_are_logged_and_left_out(
    caplog: pytest.LogCaptureFixture,
) -> None:
    unreadable = _scope_choices()
    del unreadable["base"]
    part = _definite("spec_1")
    part["spec"]["resolved"]["scope_choices"] = unreadable

    assert reading_facts(part["spec"])["scope"] is None
    assert any("spec_1" in record.getMessage() for record in caplog.records)


def test_definite_part_yields_its_spec_id_and_counts_defaulted_facets() -> None:
    facets = {
        "source": {"basis": "bound"},
        "unit": {"basis": "declared_default"},
        "measurementMethod": {"basis": "bound", "selection": "per_series"},
    }
    digest = graph_envelope({"status": "definite", "parts": [_definite("spec_1", facets)]})
    assert digest["status"] == "definite"
    assert digest["spec_id"] == "spec_1"
    assert digest["choice_points"] == 2
    assert digest["parts"] == 1
    assert digest["candidates"] == []


def test_the_first_part_decides_and_the_part_count_is_reported() -> None:
    envelope = {
        "status": "mixed",
        "parts": [
            _definite("spec_1"),
            {
                "status": "no_data",
                "variable_text": "malaria deaths",
                "no_data": {"category": "absent"},
            },
        ],
    }
    digest = graph_envelope(envelope)
    assert digest["status"] == "definite"
    assert digest["spec_id"] == "spec_1"
    assert digest["parts"] == 2


def test_every_part_is_kept_as_a_summary() -> None:
    envelope = {
        "status": "mixed",
        "parts": [
            _definite("spec_1"),
            {
                "status": "candidates",
                "variable_text": "health spending",
                "candidates": {
                    "specs": [{"spec_id": "spec_a", "resolved": {"rendered_sentence": "total"}}]
                },
            },
        ],
    }
    digest = graph_envelope(envelope)

    assert digest["spec_id"] == "spec_1"  # the first part still decides
    assert digest["part_digests"] == [
        {
            "variable": "life expectancy",
            "status": "definite",
            "spec_id": "spec_1",
            "resolved": "life expectancy in Kenya",
            "warnings": [],
            "candidates": [],
            "wide": False,
        },
        {
            "variable": "health spending",
            "status": "candidates",
            "spec_id": None,
            "resolved": None,
            "warnings": [],
            "candidates": [{"spec_id": "spec_a", "label": "total", "wide": False}],
        },
    ]


_NO_IDENTITY_FACTS = {
    "wide": False,
    "stat_var_shape_kind": None,
    "stat_var_item_count": None,
    "stat_var_member_count_exact": True,
    "entity_roles": [],
    "stat_var_served_count": None,
    "open_entity_axes": [],
}


def test_candidates_part_lists_each_spec_with_its_sentence() -> None:
    envelope = {
        "status": "candidates",
        "parts": [
            {
                "status": "candidates",
                "candidates": {
                    "specs": [
                        {
                            "spec_id": "spec_a",
                            "resolved": {"rendered_sentence": "military spending"},
                        },
                        {"spec_id": "spec_b"},
                    ]
                },
            }
        ],
    }
    digest = graph_envelope(envelope)
    assert digest["status"] == "candidates"
    assert digest["spec_id"] is None
    assert digest["choice_points"] == 2
    assert digest["candidates"] == [
        {"spec_id": "spec_a", "label": "military spending", **_NO_IDENTITY_FACTS},
        {"spec_id": "spec_b", "label": "spec_b", **_NO_IDENTITY_FACTS},
    ]


def test_candidates_carry_the_same_wide_verdict_a_definite_reading_would() -> None:
    """A clarify choice's width must be known at a clarify re-entry without
    a second `search` call, so each offered candidate carries the same
    single-series verdict `graph_envelope`'s `definite` branch computes for
    the chosen reading -- not just `spec_id`/`label`."""
    envelope = {
        "status": "candidates",
        "parts": [
            {
                "status": "candidates",
                "candidates": {
                    "specs": [
                        {
                            "spec_id": "spec_role_bearing",
                            "resolved": {
                                "rendered_sentence": "flows into Kenya",
                                "stat_vars": {"member_count": 1, "served_count": None},
                            },
                            "definition": {"identity": _ROLE_BEARING_IDENTITY},
                        },
                        {
                            "spec_id": "spec_single_series",
                            "resolved": {
                                "rendered_sentence": "life expectancy in Kenya",
                                "stat_vars": {"member_count": 3, "served_count": 1},
                            },
                            "definition": {"identity": _STANDARD_IDENTITY},
                        },
                    ]
                },
            }
        ],
    }
    digest = graph_envelope(envelope)
    by_id = {c["spec_id"]: c for c in digest["candidates"]}
    assert by_id["spec_role_bearing"]["wide"] is True
    assert by_id["spec_single_series"]["wide"] is False


def test_unresolved_no_data_keeps_no_data_status_with_nearest_real_as_candidates() -> None:
    envelope = {
        "status": "no_data",
        "parts": [
            {
                "status": "no_data",
                "no_data": {
                    "category": "unresolved",
                    "reason": "no statvar matched",
                    "nearest_real": [
                        {"spec_id": "spec_n", "resolved": {"rendered_sentence": "near"}}
                    ],
                },
            }
        ],
    }
    digest = graph_envelope(envelope)
    assert digest["status"] == "no_data"
    assert digest["candidates"] == [{"spec_id": "spec_n", "label": "near", **_NO_IDENTITY_FACTS}]


def test_absent_no_data_maps_to_unsupported_with_the_reason_as_a_warning() -> None:
    envelope = {
        "status": "no_data",
        "parts": [
            {"status": "no_data", "no_data": {"category": "absent", "reason": "no observations"}}
        ],
    }
    digest = graph_envelope(envelope)
    assert digest["status"] == "unsupported"
    assert {"code": "no_data", "severity": "warn", "message": "no observations"} in digest[
        "warnings"
    ]


@pytest.mark.parametrize(
    "no_data",
    [
        {"category": "unresolved", "reason": "role_not_resolved"},
        {"category": "absent", "reason": "no_observations"},
        {
            "category": "unresolved",
            "reason": "variable_not_resolved",
            "nearest_real": [{"spec_id": "spec_n", "resolved": {"rendered_sentence": "near"}}],
        },
    ],
)
def test_a_no_data_part_summary_carries_the_reason_code(no_data: dict[str, Any]) -> None:
    digest = graph_envelope({"parts": [{"status": "no_data", "no_data": no_data}]})

    assert digest["part_digests"][0]["reason"] == no_data["reason"]


def test_a_no_data_part_summary_carries_the_message_when_search_gave_one() -> None:
    no_data = {"category": "absent", "reason": "no_observations"}
    message = "No observations exist for Kenya in 1850."

    plain = graph_envelope({"parts": [{"status": "no_data", "no_data": no_data}]})
    stated = graph_envelope(
        {
            "parts": [
                {
                    "status": "no_data",
                    "no_data": {**no_data, "message": message, "a_member_added_later": 1},
                }
            ]
        }
    )

    assert "message" not in plain["part_digests"][0]
    assert stated["part_digests"][0]["message"] == message


def test_a_qualifier_not_bound_summary_names_the_phrases_and_the_flagged_match() -> None:
    """The orchestrator reads which phrases search could not apply, and what it
    found without them, from the summary alone."""
    no_data = {
        "category": "absent",
        "reason": "qualifier_not_bound",
        "unbound_qualifiers": [
            {"text": "HIV", "disposition": "unbound"},
            {"text": "AIDS", "disposition": "unbound"},
            {"text": "HIV", "disposition": "unbound"},
            {"disposition": "unbound"},
        ],
        "flagged_match": {
            "spec_id": "spec_health",
            "resolved": {"rendered_sentence": "Gross ODA disbursements to Kenya for health."},
        },
    }

    summary = graph_envelope({"parts": [{"status": "no_data", "no_data": no_data}]})[
        "part_digests"
    ][0]

    assert summary["unbound_qualifiers"] == ["HIV", "AIDS"]
    assert summary["flagged_match"] == "Gross ODA disbursements to Kenya for health."


def test_a_gate_failure_summary_leaves_its_flagged_match_out() -> None:
    no_data = {
        "category": "absent",
        "reason": "gate_failed",
        "gate": "min_countries_5",
        "flagged_match": {"spec_id": "spec_f", "resolved": {"rendered_sentence": "flagged"}},
    }

    summary = graph_envelope({"parts": [{"status": "no_data", "no_data": no_data}]})[
        "part_digests"
    ][0]

    assert "flagged_match" not in summary
    assert "unbound_qualifiers" not in summary


def test_a_part_that_is_not_no_data_summarises_without_a_reason() -> None:
    digest = graph_envelope({"parts": [_definite("spec_a")]})

    assert "reason" not in digest["part_digests"][0]


def test_a_part_summary_carries_the_departures_search_states() -> None:
    departure = {
        "code": "closest_match",
        "text": "You asked for climate finance. This shows ODA, the closest measure.",
        "asked": "climate finance",
        "binding_ref": None,
    }
    plain = graph_envelope({"parts": [_definite("spec_a")]})
    departed = graph_envelope({"parts": [{**_definite("spec_a"), "departures": [departure]}]})

    assert "departures" not in plain["part_digests"][0]
    assert departed["part_digests"][0]["departures"] == [
        {"code": departure["code"], "text": departure["text"]}
    ]


@pytest.mark.parametrize(
    ("total", "readings"),
    [
        (
            "GroupOfSeven",
            [
                "The figures for G7 are the total the source publishes for the group as a whole "
                "(G7 countries, Total), not its members' figures added up."
            ],
        ),
        (None, None),
    ],
    ids=["total", "members"],
)
def test_a_part_summary_says_a_set_is_read_as_its_published_total(
    total: str | None, readings: list[str] | None
) -> None:
    part = _definite("spec_a")
    part["spec"]["resolved"] |= {
        "entities": [{"role": "origin", "ref": "GroupOfSeven", "label": "G7 countries, Total"}],
        "entity_groups": {"origin": {"ref": None, "label": "G7", "total": total}},
    }

    (summary,) = graph_envelope({"parts": [part]})["part_digests"]

    assert summary.get("group_readings") == readings


@pytest.mark.parametrize(
    ("alternatives", "offered"),
    [
        (
            [{"label": "net disbursements", "axis": "what", "definition": {"spec_id": "spec_n"}}],
            {"alternatives": [{"spec_id": "spec_n", "label": "net disbursements"}]},
        ),
        # Net and gross are both published, but the basis convention settles which one
        # answers, so search lists no basis sibling and the part offers no count to ask.
        (
            [{"label": "constant prices", "axis": "unit", "definition": {"spec_id": "spec_c"}}],
            {"other_prices": [{"spec_id": "spec_c", "label": "constant prices"}]},
        ),
        ([], {}),
    ],
    ids=["basis-sibling", "convention-settled-basis", "no-sibling"],
)
def test_a_part_summary_offers_a_count_only_when_search_lists_a_basis_sibling(
    alternatives: list[dict[str, Any]], offered: dict[str, Any]
) -> None:
    part = _definite("spec_a")
    part["spec"]["resolved"] |= {
        "measure_label": "gross disbursements",
        "alternative_measures": alternatives,
    }

    (summary,) = graph_envelope({"parts": [part]})["part_digests"]

    assert summary.get("alternatives") == offered.get("alternatives")
    assert summary.get("other_prices") == offered.get("other_prices")
    assert ("counted_as" in summary) is bool(offered)


def test_a_part_summary_with_a_set_of_an_unknown_class_says_no_reading() -> None:
    part = _definite("spec_a")
    part["spec"]["resolved"] |= {
        "entity_groups": {
            "origin": {"ref": None, "label": "DAC", "total": "DAC", "whole_of": ["party"]}
        },
    }

    (summary,) = graph_envelope({"parts": [part]})["part_digests"]

    assert "group_readings" not in summary


@pytest.mark.parametrize(
    ("facets", "codes"),
    [
        (
            {"provenance": {"ref": "provenance/IDS", "label": "World Bank IDS"}},
            [{"code": "DT.TDS.DEGG.CD", "source": "World Bank IDS"}],
        ),
        (
            {"provenance": {"ref": "provenance/OECDDAC1", "label": "OECD DAC1"}},
            [{"code": "DT.TDS.DEGG.CD"}],
        ),
        ({}, [{"code": "DT.TDS.DEGG.CD"}]),
    ],
    ids=["named", "other-source-bound", "no-source-bound"],
)
def test_a_part_summary_lists_the_source_codes_of_what_search_found(
    facets: dict[str, Any], codes: list[dict[str, str]]
) -> None:
    part = _definite("spec_a")
    plain = graph_envelope({"parts": [part]})["part_digests"][0]
    part["spec"]["resolved"]["facets"] = facets
    part["spec"]["resolved"]["stat_vars"]["source_codes"] = [
        {"provenance": "provenance/IDS", "code": "DT.TDS.DEGG.CD"}
    ]

    (summary,) = graph_envelope({"parts": [part]})["part_digests"]

    assert "source_codes" not in plain
    assert summary["source_codes"] == codes


def test_unsupported_part_carries_its_detail_as_a_warning() -> None:
    envelope = {
        "status": "unsupported",
        "parts": [
            {
                "status": "unsupported",
                "unsupported": {"code": "out_of_scope", "detail": "not a data question"},
            }
        ],
    }
    digest = graph_envelope(envelope)
    assert digest["status"] == "unsupported"
    assert digest["warnings"][-1]["message"] == "not a data question"


def test_request_warnings_come_before_the_first_parts_own() -> None:
    envelope = {
        "status": "definite",
        "diagnostics": {
            "warnings": [{"code": "slow", "severity": "info", "message": "took a while"}]
        },
        "parts": [
            _definite(
                "spec_1",
                warnings=[
                    {
                        "code": "time_text_not_parsed",
                        "severity": "warning",
                        "message": "since when?",
                    }
                ],
            )
        ],
    }
    codes = [w["code"] for w in graph_envelope(envelope)["warnings"]]
    assert codes == ["slow", "time_text_not_parsed"]


def test_an_envelope_with_no_parts_is_unsupported() -> None:
    digest = graph_envelope({"status": "unsupported"})
    assert digest["status"] == "unsupported"
    assert digest["spec_id"] is None
    assert digest["parts"] == 1


def test_the_captured_guatemala_fixture_is_narrow() -> None:
    """The captured Guatemala resolve envelope
    (`services/data-mcp/tests/fixtures/resolve_population_guatemala.json`):
    twelve unbound slots, `observationAbout` only, and the mint serves one
    StatVar, the family's root `Count_Person`. It must not reach the planner
    and the approval gate as `wide`."""
    envelope = json.loads((_DATA_MCP_FIXTURES / "resolve_population_guatemala.json").read_text())

    assert graph_envelope(envelope)["wide"] is False


@pytest.mark.parametrize(
    ("identity", "served_count", "wide"),
    [
        pytest.param(_STANDARD_IDENTITY, 1, False, id="described-one-served"),
        pytest.param(_ROLE_BEARING_IDENTITY, 1, False, id="role-bearing-one-served"),
        pytest.param(_STANDARD_IDENTITY, 3, True, id="described-several-served"),
        pytest.param(_STANDARD_IDENTITY, None, True, id="described-unsettled"),
        pytest.param(_ROLE_BEARING_IDENTITY, None, True, id="role-bearing-unsettled"),
        pytest.param(_PINNED_IDENTITY, None, False, id="pinned-arm"),
        pytest.param(None, None, False, id="no-identity"),
        pytest.param(None, 3, True, id="no-identity-several-served"),
        pytest.param({"shape": {"kind": "something_new"}}, None, False, id="unknown-shape"),
    ],
)
def test_a_spec_is_wide_unless_it_mints_one_series(
    identity: dict | None, served_count: int | None, wide: bool
) -> None:
    """One series is a pinned arm, or any identity `search` states serves one
    StatVar once its membership settled. A described identity is otherwise
    wide, its membership unsettled included."""
    digest = graph_envelope(
        {
            "status": "definite",
            "parts": [
                _definite("spec_1", identity=identity, member_count=3, served_count=served_count)
            ],
        }
    )

    assert digest["wide"] is wide


@pytest.mark.parametrize(
    ("member_count", "member_count_exact", "item_count", "exact"),
    [
        (None, None, None, True),
        (1_454, None, 1_454, True),
        # Search counted at least this many.
        (3_687, False, 3_687, False),
    ],
)
def test_the_raw_facts_ride_alongside_the_derived_verdict(
    member_count: int | None, member_count_exact: bool | None, item_count: int | None, exact: bool
) -> None:
    """A later change to the predicate must not need another trip into
    `mcp_client.py`'s raw envelope: the facts it was computed from are
    already in the digest."""
    digest = graph_envelope(
        {
            "status": "definite",
            "parts": [
                _definite(
                    "spec_1",
                    identity=_STANDARD_IDENTITY,
                    member_count=member_count,
                    member_count_exact=member_count_exact,
                    served_count=member_count,
                )
            ],
        }
    )
    assert digest["stat_var_shape_kind"] == "statvar_schema"
    assert digest["stat_var_item_count"] == item_count
    assert digest["stat_var_member_count_exact"] is exact
    assert digest["entity_roles"] == ["observationAbout"]
    assert digest["stat_var_served_count"] == member_count


_FIXTURES = Path(__file__).parent / "fixtures"


@pytest.mark.parametrize(
    ("fixture", "entity_roles", "open_entity_axes"),
    [
        # Education aid to Kenya: Kenya bound, every donor left open.
        (
            "search_spec_role_open.json",
            ["destination"],
            [{"column": "origin", "members": 55}],
        ),
        # Health spending per person: no place named, every country left open.
        (
            "search_spec_place_scope.json",
            [],
            [{"column": "observationAbout", "members": 193}],
        ),
    ],
)
def test_the_digest_states_each_axis_search_left_open_with_its_member_count(
    fixture: str, entity_roles: list[str], open_entity_axes: list[dict[str, object]]
) -> None:
    spec = json.loads((_FIXTURES / fixture).read_text())["spec"]

    digest = graph_envelope({"status": "definite", "parts": [{"status": "definite", "spec": spec}]})

    assert digest["entity_roles"] == entity_roles
    assert digest["open_entity_axes"] == open_entity_axes


_TWO_COUNTRY_ENVELOPE = (
    Path(__file__).parent / "fixtures" / "resolve_life_expectancy_two_countries.json"
)


_SERVED_1960_2023 = {"start": "1960", "end": "2023"}


def test_the_captured_two_country_fixture_carries_the_coverage_window() -> None:
    """The captured "life expectancy in Nigeria and Guatemala" envelope
    (`fixtures/resolve_life_expectancy_two_countries.json`) asked no time
    text at all, so the identity's `requested_window` is `{"kind": "all"}`,
    data-mcp's own vocabulary for "no specific span"
    (`data_mcp.get_data_digest._coverage_window_text`), naming no range of
    its own. The spec-level `data_date_range` over the served members,
    `{"start": "1960", "end": "2023"}`, is the resolver's actual coverage, so `graph_envelope`
    states that instead of leaving the planner to guess."""
    envelope = json.loads(_TWO_COUNTRY_ENVELOPE.read_text())
    digest = graph_envelope(envelope)
    assert digest["window"] == {"start": 1960, "end": 2023, "source": "coverage"}


def test_a_range_window_with_both_endpoints_wins_over_coverage() -> None:
    """The real wire shape (`resolve_population_guatemala.json`,
    `mint_statvar_set_guatemala.json`): `requested_window.kind == "range"`,
    with the years at `start_year`/`end_year`, not `start`/`end`. Both
    endpoints present is a complete answer the question named outright, so
    it is stated as `source: "requested"` even though the served members'
    (different) coverage and the headline's year are also stated."""
    identity = {
        **_STANDARD_IDENTITY,
        "requested_window": {"kind": "range", "start_year": "1990", "end_year": "2023"},
    }
    part = _definite("spec_1", identity=identity, data_date_range=_SERVED_1960_2023)
    part["spec"]["resolved"]["headline"] = {"year": 2024, "series": 85}
    digest = graph_envelope({"status": "definite", "parts": [part]})
    assert digest["window"] == {"start": 1990, "end": 2023, "source": "requested"}


def test_window_is_left_out_when_the_identity_states_none_and_nothing_served_covers() -> None:
    digest = graph_envelope(
        {"status": "definite", "parts": [_definite("spec_1", identity=_STANDARD_IDENTITY)]}
    )
    assert "window" not in digest


def test_the_kind_is_all_falls_back_to_the_served_coverage() -> None:
    identity = {**_STANDARD_IDENTITY, "requested_window": {"kind": "all"}}
    digest = graph_envelope(
        {
            "status": "definite",
            "parts": [_definite("spec_1", identity=identity, data_date_range=_SERVED_1960_2023)],
        }
    )
    assert digest["window"] == {"start": 1960, "end": 2023, "source": "coverage"}


def test_the_download_window_wins_over_the_served_coverage() -> None:
    """The points a mint writes start in 1973, which `resolved.download.window`
    states; the served members' range is the narrower read."""
    identity = {**_STANDARD_IDENTITY, "requested_window": {"kind": "all"}}
    spec = _definite(
        "spec_1",
        identity=identity,
        member_count=1_454,
        data_date_range={"start": "1990", "end": "2024"},
    )
    spec["spec"]["resolved"]["download"] = {"window": {"start_year": 1973, "end_year": 2024}}

    digest = graph_envelope({"status": "definite", "parts": [spec]})

    assert digest["window"] == {"start": 1973, "end": 2024, "source": "coverage"}


def test_the_kind_is_all_and_nothing_served_covers_leaves_window_out() -> None:
    identity = {**_STANDARD_IDENTITY, "requested_window": {"kind": "all"}}
    digest = graph_envelope(
        {"status": "definite", "parts": [_definite("spec_1", identity=identity)]}
    )
    assert "window" not in digest


@pytest.mark.parametrize(
    ("headline", "window"),
    [
        pytest.param(
            {"year": 2024, "series": 85},
            {"start": 2024, "end": 2024, "source": "elected"},
            id="elected-year",
        ),
        pytest.param({"year": None, "series": 85}, None, id="headline-without-a-year"),
        pytest.param(None, None, id="no-headline"),
    ],
)
def test_a_latest_window_is_the_year_the_headline_states(
    headline: dict | None, window: dict | None
) -> None:
    """`latest` names one year: the one the answer's headline states, never
    the served members' multi-year coverage. With no year stated the window
    is left out."""
    identity = {**_STANDARD_IDENTITY, "requested_window": {"kind": "latest"}}
    part = _definite("spec_1", identity=identity, data_date_range=_SERVED_1960_2023)
    if headline is not None:
        part["spec"]["resolved"]["headline"] = headline
    digest = graph_envelope({"status": "definite", "parts": [part]})
    assert digest.get("window") == window


def test_an_open_ended_range_is_completed_from_the_served_coverage() -> None:
    """ "Since 1990" is a `range` with a `start_year` and no `end_year`,
    the question's own lower bound. `graph_envelope` completes it with
    the coverage's own latest year."""
    identity = {
        **_STANDARD_IDENTITY,
        "requested_window": {"kind": "range", "start_year": "1990"},
    }
    digest = graph_envelope(
        {
            "status": "definite",
            "parts": [_definite("spec_1", identity=identity, data_date_range=_SERVED_1960_2023)],
        }
    )
    assert digest["window"] == {"start": 1990, "end": 2023, "source": "coverage"}


def test_an_open_ended_range_with_no_coverage_is_left_out() -> None:
    identity = {
        **_STANDARD_IDENTITY,
        "requested_window": {"kind": "range", "start_year": "1990"},
    }
    digest = graph_envelope(
        {"status": "definite", "parts": [_definite("spec_1", identity=identity)]}
    )
    assert "window" not in digest


def test_a_window_that_does_not_parse_as_years_is_left_out() -> None:
    identity = {
        **_STANDARD_IDENTITY,
        "requested_window": {"kind": "range", "start_year": "unknown", "end_year": "2023"},
    }
    digest = graph_envelope(
        {"status": "definite", "parts": [_definite("spec_1", identity=identity)]}
    )
    assert "window" not in digest


def test_tool_contracts_import_stays_a_leaf() -> None:
    """`data_mcp.get_data_tool` drags `httpx2`, `fastmcp`, `mcp.types` and
    `data_mcp.client` into whatever imports it -- the upstream client, fastmcp's own
    tool machinery, and the mint call's request/response types. `harness.tool_contracts`
    must import neither `data_mcp.get_data_tool` nor `data_mcp.search_tool`.

    Run in a fresh interpreter, not in this test process: by the time any test here
    runs, pytest's own collection and other harness modules may already have imported
    some of these names for reasons that have nothing to do with this import line, which
    would make an in-process check pass or fail on accident.
    """
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, harness.tool_contracts\n"
            "leaked = sorted(\n"
            "    m for m in ('httpx2', 'fastmcp', 'mcp.types', 'data_mcp.client') if m in sys.modules\n"
            ")\n"
            "print(leaked)\n",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == "[]", result.stdout + result.stderr


def _named_skills_envelope(parts: list, skill_ids: list[str]) -> dict:
    return {
        "status": "mixed",
        "parts": parts,
        "skills": {
            skill_id: {"id": skill_id, "description": f"When  {skill_id}\n applies."}
            for skill_id in skill_ids
        },
    }


def _skilled(spec_id: str, *skill_ids: str) -> dict:
    part = _definite(spec_id)
    part["spec"]["resolved"]["skills"] = list(skill_ids)
    return part


def test_skills_read_every_part_in_digest_order_with_descriptions() -> None:
    envelope = _named_skills_envelope(
        [
            _skilled("spec_a", "oda-measures", "oda-data-sources"),
            _skilled("spec_b", "oda-data-sources", "debt-basics"),
        ],
        ["debt-basics", "oda-data-sources", "oda-measures"],
    )

    digest = graph_envelope(envelope)

    assert digest["spec_id"] == "spec_a"
    assert digest["skills"] == [
        {"id": "oda-measures", "description": "When oda-measures applies."},
        {"id": "oda-data-sources", "description": "When oda-data-sources applies."},
        {"id": "debt-basics", "description": "When debt-basics applies."},
    ]


def test_skills_keep_every_named_skill() -> None:
    skill_ids = [f"skill-{index}" for index in range(7)]
    envelope = _named_skills_envelope([_skilled("spec_a", *skill_ids)], skill_ids)

    digest = graph_envelope(envelope)

    assert [skill["id"] for skill in digest["skills"]] == skill_ids


def test_an_envelope_with_no_skills_map_gives_an_empty_list_on_every_branch() -> None:
    for part in (
        _definite("spec_a"),
        {"status": "candidates", "candidates": {"specs": []}},
        {"status": "no_data", "no_data": {"category": "absent"}},
        {"status": "unsupported"},
    ):
        assert graph_envelope({"parts": [part]})["skills"] == []
    assert graph_envelope({"kind": "catalogue", "scope": {}})["skills"] == []


def test_a_catalogue_answer_is_unsupported_with_searchs_reading_as_its_warning() -> None:
    digest = graph_envelope(
        {
            "kind": "catalogue",
            "scope": {"place": {"ref": "country/KEN", "label": "Kenya"}},
            "diagnostics": {"warnings": []},
        }
    )

    assert digest["status"] == "unsupported"
    assert digest["spec_id"] is None
    assert digest["candidates"] == []
    assert digest["kind"] == "catalogue"
    assert digest["scope_place"] == "Kenya"
    assert digest["part_digests"] == []
    assert digest["warnings"] == [
        {
            "code": "catalogue",
            "severity": "warn",
            "message": (
                "Search read this question as asking what data DFO holds, not as naming data "
                "to retrieve. It named Kenya."
            ),
        }
    ]


def test_a_declined_answer_carries_its_reason_in_plain_words() -> None:
    out_of_scope = graph_envelope({"kind": "declined", "reason": "out_of_scope", "scope": {}})
    judgement = graph_envelope({"kind": "declined", "reason": "requires_analysis", "scope": {}})

    assert out_of_scope["status"] == judgement["status"] == "unsupported"
    assert out_of_scope["reason"] == "out_of_scope"
    assert out_of_scope["warnings"][-1]["message"] == (
        "Search read this question as not about data DFO holds."
    )
    assert judgement["warnings"][-1]["message"] == (
        "Search read this question as asking for a judgement and naming no data to show."
    )
    assert out_of_scope["scope_place"] is None


_ENTRY = {
    "axis": "sector",
    "option_id": "sector:120",
    "provenance": "carried",
    "concept": "health",
    "origin": {"kind": "choice", "choice_id": "spec_1:sector"},
}


@pytest.mark.parametrize(
    "entry",
    [
        pytest.param({**_ENTRY, "concept": ""}, id="empty-concept"),
        pytest.param(
            {**_ENTRY, "origin": {"kind": "choice", "choice_id": "c" * 201}}, id="long-id"
        ),
        pytest.param(
            {**_ENTRY, "origin": {"kind": "text", "text": "net", "start": -1, "end": 3}},
            id="negative-offset",
        ),
        pytest.param({**_ENTRY, "option_id": f"sector:{'1' * 200}"}, id="long-option"),
    ],
)
def test_an_entry_search_context_refuses_is_one_a_scope_record_refuses(
    entry: dict[str, Any],
) -> None:
    """A scope entry the harness reads is one it can carry to a later search."""
    with pytest.raises(ValidationError):
        SearchContext(carried_entries=[entry])
    with pytest.raises(ValidationError):
        ScopeEntry.model_validate(entry)


def test_a_dropped_entry_and_a_basis_line_word_each_reason_code_alike() -> None:
    """The search digest's words for a dropped entry's cause and the basis line's
    words for a reason code are the same for every reason code."""
    for code in get_args(ScopeReasonCode):
        assert _dropped_cause_words(code, axis="sector", availability={}) == REASON_COPY[code]
