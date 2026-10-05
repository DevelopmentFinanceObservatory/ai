"""`harness.tool_contracts.graph_envelope` against the search envelope's parts."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from compiler.sources import elected_stat_vars, source_from_spec_block
from harness.planner_seam import _render_envelope
from harness.tool_contracts import graph_envelope

_DATA_MCP_FIXTURES = Path(__file__).parents[2] / "data-mcp" / "tests" / "fixtures"


def _definite(
    spec_id: str,
    facets: dict | None = None,
    warnings: list | None = None,
    identity: dict | None = None,
    items: list | None = None,
    truncated: bool = False,
    member_count: int | None = None,
    slots: list | None = None,
) -> dict:
    # `resolved.stat_vars` is always the real wire object -- `{items,
    # member_count, truncated}` -- never a bare list. A hand-built helper
    # that defaults it to `[]` masks the very field this predicate reads.
    stat_vars: dict = {"items": items or [], "truncated": truncated}
    if member_count is not None:
        stat_vars["member_count"] = member_count
    resolved: dict = {
        "rendered_sentence": "life expectancy in Kenya",
        "facets": facets or {},
        "stat_vars": stat_vars,
    }
    if slots is not None:
        resolved["slots"] = slots
    spec: dict = {"spec_id": spec_id, "resolved": resolved}
    if identity is not None:
        spec["definition"] = {"identity": identity}
    return {
        "status": "definite",
        "variable_text": "life expectancy",
        "warnings": warnings or [],
        "spec": spec,
    }


# Shapes taken from two real resolve envelopes: "population for Guatemala
# since 2010", every demographic slot unbound (the same envelope committed at
# `services/data-mcp/tests/fixtures/resolve_population_guatemala.json`), and
# "female population in Mexico" from dev search, with `gender` bound. Both
# are `statvar_schema` with `observationAbout` only, and each elected one
# item that is its family's root -- both must route narrow.
_STANDARD_IDENTITY = {
    "shape": {"kind": "statvar_schema"},
    "entities": {"observationAbout": ["country/GTM"]},
    "slots": [
        {"property": "age", "binding": {"state": "unbound"}},
        {"property": "gender", "binding": {"state": "unbound"}},
    ],
}
_GENDER_BOUND_IDENTITY = {
    "shape": {"kind": "statvar_schema"},
    "entities": {"observationAbout": ["country/MEX"]},
    "slots": [
        {"property": "age", "binding": {"state": "unbound"}},
        {"property": "gender", "binding": {"state": "value", "ref": "Female"}},
    ],
}
# Search fills an item's `slot_values` from the identity's bound slots alone and lists
# every constraint the StatVar asserts, bound or not, in `constraint_properties`.
_GENDER_SLOT_VALUE = {"key": {"property": {"dcid": "gender"}}, "value": {"ref": {"dcid": "Female"}}}
# The block's `resolved.slots` entry for the bound `gender` slot. `Female` has no
# descendant codes, so Search serves it no `expansion`.
_GENDER_BLOCK_SLOT = {
    "key": {"axis": "how", "property": {"dcid": "gender", "label": "gender"}, "label": "gender"},
    "binding": {"kind": "value", "value": {"ref": {"dcid": "Female", "label": "Female"}}},
}
_ELECTED_UNBOUND_ITEM = [
    {
        "ref": "Count_Person",
        "label": "Total population",
        "slot_values": [],
        "constraint_properties": [],
    }
]
_ELECTED_BOUND_ITEM = [
    {
        "ref": "Count_Person_Female",
        "label": "Female population",
        "slot_values": [_GENDER_SLOT_VALUE],
        "constraint_properties": ["gender"],
    }
]
_ROLE_BEARING_IDENTITY = {
    "shape": {"kind": "statvar_schema"},
    "entities": {"observationAbout": ["country/KEN"], "destination": ["country/KEN"]},
}
_DONOR_ROLE_IDENTITY = {
    "shape": {"kind": "statvar_schema"},
    "entities": {"observationAbout": ["country/KEN"], "donor": ["country/FRA"]},
}
_PINNED_IDENTITY = {"shape": {"kind": "stat_var", "id": "Count_Person"}}
_ELECTED_TWO_CONSTRAINED_ITEMS = [
    {
        "ref": "Count_Person_Female",
        "label": "Female population",
        "slot_values": [],
        "constraint_properties": ["gender"],
    },
    {
        "ref": "Count_Person_Male",
        "label": "Male population",
        "slot_values": [],
        "constraint_properties": ["gender"],
    },
]


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
    "stat_var_item_count": 0,
    "entity_roles": [],
    "stat_var_truncated": False,
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
        {"spec_id": "spec_a", "label": "military spending", "stat_vars": [], **_NO_IDENTITY_FACTS},
        {"spec_id": "spec_b", "label": "spec_b", "stat_vars": [], **_NO_IDENTITY_FACTS},
    ]


def test_each_candidate_carries_the_statvars_search_elected_for_it() -> None:
    """A `get_data` call for the chosen candidate posts them as
    `resolved_stat_var_dcids`, and a clarify re-entry holds no spec block to
    read them from. A list the source reader refuses offers none of its
    entries, as a searched block's does."""
    envelope = {
        "status": "candidates",
        "parts": [
            {
                "status": "candidates",
                "candidates": {
                    "specs": [
                        {
                            "spec_id": "spec_two",
                            "resolved": {"stat_vars": {"items": _ELECTED_TWO_CONSTRAINED_ITEMS}},
                        },
                        {
                            "spec_id": "spec_one",
                            "resolved": {"stat_vars": {"items": _ELECTED_UNBOUND_ITEM}},
                        },
                        {
                            "spec_id": "spec_malformed",
                            "resolved": {
                                "stat_vars": {"items": [{"ref": "Count_Person"}, {"ref": 7}]}
                            },
                        },
                        {
                            "spec_id": "spec_prefixed",
                            "resolved": {"stat_vars": {"items": [{"ref": "dcid:Count_Person"}]}},
                        },
                    ]
                },
            }
        ],
    }

    by_id = {c["spec_id"]: c["stat_vars"] for c in graph_envelope(envelope)["candidates"]}

    assert by_id == {
        "spec_two": ["Count_Person_Female", "Count_Person_Male"],
        "spec_one": ["Count_Person"],
        "spec_malformed": [],
        "spec_prefixed": [],
    }


def test_a_candidate_and_a_definite_block_state_the_same_statvars() -> None:
    """One reader for both: the list a chosen candidate posts is the list the
    same block posts when `search` returns it as the `definite` reading."""
    block = json.loads((_DATA_MCP_FIXTURES / "resolve_population_guatemala.json").read_text())[
        "parts"
    ][0]["spec"]
    envelope = {
        "status": "candidates",
        "parts": [{"status": "candidates", "candidates": {"specs": [block]}}],
    }

    (candidate,) = graph_envelope(envelope)["candidates"]

    assert candidate["stat_vars"] == list(source_from_spec_block(block).stat_vars)
    assert candidate["stat_vars"] == list(elected_stat_vars(block["resolved"]))
    assert candidate["stat_vars"] == ["Count_Person"]


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
                                "stat_vars": {"items": _ELECTED_UNBOUND_ITEM},
                            },
                            "definition": {"identity": _ROLE_BEARING_IDENTITY},
                        },
                        {
                            "spec_id": "spec_single_series",
                            "resolved": {
                                "rendered_sentence": "life expectancy in Kenya",
                                "stat_vars": {"items": _ELECTED_UNBOUND_ITEM},
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
    assert digest["candidates"] == [
        {"spec_id": "spec_n", "label": "near", "stat_vars": [], **_NO_IDENTITY_FACTS}
    ]


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
    """The captured Guatemala resolve envelope's worked example
    (`services/data-mcp/tests/fixtures/resolve_population_guatemala.json`):
    twelve unbound slots, one elected item (`Count_Person`,
    `constraint_properties: []`), `observationAbout` only. The one elected member is the family's
    root (`data_mcp.substitution.root_member`), so it must not reach the
    planner and the approval gate as `wide`."""
    digest = graph_envelope(
        {
            "status": "definite",
            "parts": [
                _definite("spec_1", identity=_STANDARD_IDENTITY, items=_ELECTED_UNBOUND_ITEM)
            ],
        }
    )
    assert digest["wide"] is False


def test_the_mexico_bound_slot_shape_is_also_narrow() -> None:
    """ "Female population in Mexico" as dev search resolves it: `gender`
    bound, one elected item constrained by `gender` alone, `member_count` 2
    and `truncated` true. The item's constraint properties equal the bound
    slot properties, so it is the root of the family the binding names and
    this is narrow exactly like the all-unbound case, whatever `truncated`
    says."""
    digest = graph_envelope(
        {
            "status": "definite",
            "parts": [
                _definite(
                    "spec_1",
                    identity=_GENDER_BOUND_IDENTITY,
                    items=_ELECTED_BOUND_ITEM,
                    truncated=True,
                    member_count=2,
                    slots=[_GENDER_BLOCK_SLOT],
                )
            ],
        }
    )
    assert digest["wide"] is False


def test_a_bound_value_that_widens_to_descendant_codes_is_wide() -> None:
    """The mint serves a hierarchical code's whole subtree, so the one elected
    item is not the only series even though it is constrained by the bound
    slots alone. The block's slot carries the `expansion` Search walked."""
    widened = {**_GENDER_BLOCK_SLOT, "expansion": {"added": [{"dcid": "Female_Adult"}]}}
    for slots in ([widened], None):
        digest = graph_envelope(
            {
                "status": "definite",
                "parts": [
                    _definite(
                        "spec_1",
                        identity=_GENDER_BOUND_IDENTITY,
                        items=_ELECTED_BOUND_ITEM,
                        slots=slots,
                    )
                ],
            }
        )
        assert digest["wide"] is True


def test_an_elected_item_constrained_beyond_the_bound_slots_is_wide() -> None:
    """`Count_Person_Female_Urban` is one of several members under the family
    the `gender` binding names, and the root dominates it: the mint resolves
    the rest too. Its `slot_values` show `gender` alone, so only
    `constraint_properties` tells it from the root."""
    item = {
        "ref": "Count_Person_Female_Urban",
        "slot_values": [_GENDER_SLOT_VALUE],
        "constraint_properties": ["gender", "placeOfResidenceClassification"],
    }
    digest = graph_envelope(
        {
            "status": "definite",
            "parts": [_definite("spec_1", identity=_GENDER_BOUND_IDENTITY, items=[item])],
        }
    )
    assert digest["wide"] is True


def test_a_constrained_item_under_an_unbound_identity_is_wide() -> None:
    """Nothing is bound, so Search serves `slot_values: []` for
    `Count_Person_Female` too."""
    item = {"ref": "Count_Person_Female", "slot_values": [], "constraint_properties": ["gender"]}
    digest = graph_envelope(
        {
            "status": "definite",
            "parts": [_definite("spec_1", identity=_STANDARD_IDENTITY, items=[item])],
        }
    )
    assert digest["wide"] is True


def test_an_item_without_constraint_properties_is_wide() -> None:
    """A dfo-apis Search that predates `constraint_properties` serves
    `slot_values` alone, which cannot tell the root from a dominated member."""
    item = {"ref": "Count_Person_Female", "slot_values": [_GENDER_SLOT_VALUE]}
    digest = graph_envelope(
        {
            "status": "definite",
            "parts": [_definite("spec_1", identity=_GENDER_BOUND_IDENTITY, items=[item])],
        }
    )
    assert digest["wide"] is True


def test_a_set_bound_slot_is_wide() -> None:
    identity = {
        **_GENDER_BOUND_IDENTITY,
        "slots": [{"property": "gender", "binding": {"state": "set", "refs": ["Female", "Male"]}}],
    }
    digest = graph_envelope(
        {
            "status": "definite",
            "parts": [_definite("spec_1", identity=identity, items=_ELECTED_BOUND_ITEM)],
        }
    )
    assert digest["wide"] is True


def test_nothing_elected_is_wide() -> None:
    """Described-arm with no `resolved.stat_vars.items` at all: nothing
    is the root, so the mint will enumerate the whole family."""
    digest = graph_envelope(
        {"status": "definite", "parts": [_definite("spec_1", identity=_STANDARD_IDENTITY)]}
    )
    assert digest["wide"] is True


def test_a_role_bearing_identity_is_wide_even_with_an_elected_item() -> None:
    """The CRS `FinancialFlow` case: `entities` binds `destination` beyond
    `observationAbout`, so the root test deliberately leaves it alone -- an
    elected item does not make it narrow."""
    digest = graph_envelope(
        {
            "status": "definite",
            "parts": [
                _definite("spec_1", identity=_ROLE_BEARING_IDENTITY, items=_ELECTED_UNBOUND_ITEM)
            ],
        }
    )
    assert digest["wide"] is True


def test_a_donor_role_identity_is_wide_even_with_an_elected_item() -> None:
    """The role-bearing check is a whitelist of exactly `{"observationAbout"}`, not
    a denylist of `origin`/`destination` alone -- an aid flow's `donor` role must
    route wide the same way."""
    digest = graph_envelope(
        {
            "status": "definite",
            "parts": [
                _definite("spec_1", identity=_DONOR_ROLE_IDENTITY, items=_ELECTED_UNBOUND_ITEM)
            ],
        }
    )
    assert digest["wide"] is True


def test_two_elected_items_are_wide() -> None:
    """The single-series rule needs exactly one elected item
    (`data_mcp.substitution.root_member`), so a non-empty `items` alone must not
    read as narrow -- the mint enumerates the family."""
    digest = graph_envelope(
        {
            "status": "definite",
            "parts": [
                _definite(
                    "spec_1",
                    identity=_STANDARD_IDENTITY,
                    items=_ELECTED_TWO_CONSTRAINED_ITEMS,
                )
            ],
        }
    )
    assert digest["wide"] is True


def test_a_pinned_arm_identity_is_narrow() -> None:
    """Already a single series (`shape.kind == "stat_var"`) -- there is
    nothing for the root test to apply to, and it needs none."""
    digest = graph_envelope(
        {"status": "definite", "parts": [_definite("spec_1", identity=_PINNED_IDENTITY)]}
    )
    assert digest["wide"] is False


def test_no_identity_at_all_is_not_wide() -> None:
    # An older upstream, or a fixture that never populated `definition`.
    digest = graph_envelope({"status": "definite", "parts": [_definite("spec_1")]})
    assert digest["wide"] is False


def test_a_pinned_arm_wins_over_truncated() -> None:
    """A pinned arm always wins over `truncated`, not just over an absent one:
    `shape.kind == "stat_var"` already names a single series, so a `true`
    here must not be allowed to route it to the planner as `wide`."""
    digest = graph_envelope(
        {
            "status": "definite",
            "parts": [_definite("spec_1", identity=_PINNED_IDENTITY, truncated=True)],
        }
    )
    assert digest["wide"] is False


def test_a_root_wins_over_truncated_for_a_described_identity() -> None:
    """The post-#893 shape of the captured Guatemala fixture: dfo-apis#893
    changes `member_count` from `len(items)` to the confirmed-at-place
    population and makes `truncated` follow it (`len(items) < member_count`),
    but `items` itself does not change -- an ordinary standard identity
    still elects exactly one winner. If `truncated` could override a root,
    this flagship case (the one the root test exists to keep off the
    slow path) would flip wide the day #893 ships, and nothing in this
    file's fixtures -- all built from today's values -- would catch it.
    This is that regression, asserted directly: same identity and item as
    `resolve_population_guatemala.json`, `member_count` and `truncated`
    given their post-#893 values. It must still route narrow."""
    digest = graph_envelope(
        {
            "status": "definite",
            "parts": [
                _definite(
                    "spec_1",
                    identity=_STANDARD_IDENTITY,
                    items=_ELECTED_UNBOUND_ITEM,
                    truncated=True,
                    member_count=40,
                )
            ],
        }
    )
    assert digest["wide"] is False


def test_truncated_still_sharpens_for_an_unrecognised_shape() -> None:
    """`truncated` keeps its sharpening role for the one case the root test
    does not cover: an identity whose shape this predicate does not even recognise as
    described-arm or pinned-arm (an older upstream, or a shape kind this
    code does not yet know)."""
    identity = {"shape": {"kind": "future_shape"}, "entities": {"observationAbout": ["x"]}}
    digest = graph_envelope(
        {
            "status": "definite",
            "parts": [_definite("spec_1", identity=identity, truncated=True)],
        }
    )
    assert digest["wide"] is True


def test_truncated_false_never_narrows_a_role_bearing_identity() -> None:
    digest = graph_envelope(
        {
            "status": "definite",
            "parts": [
                _definite(
                    "spec_1",
                    identity=_ROLE_BEARING_IDENTITY,
                    items=_ELECTED_UNBOUND_ITEM,
                    truncated=False,
                )
            ],
        }
    )
    assert digest["wide"] is True


def test_the_raw_facts_ride_alongside_the_derived_verdict() -> None:
    """A later change to the predicate must not need another trip into
    `mcp_client.py`'s raw envelope: the facts it was computed from are
    already in the digest."""
    digest = graph_envelope(
        {
            "status": "definite",
            "parts": [
                _definite("spec_1", identity=_STANDARD_IDENTITY, items=_ELECTED_UNBOUND_ITEM)
            ],
        }
    )
    assert digest["stat_var_shape_kind"] == "statvar_schema"
    assert digest["stat_var_item_count"] == 1
    assert digest["entity_roles"] == ["observationAbout"]
    assert digest["stat_var_truncated"] is False


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


_ITEM_WITH_COVERAGE = [
    {
        "ref": "LifeExpectancy_Person",
        "label": "Life expectancy at birth",
        "slot_values": [],
        "constraint_properties": [],
        "data_date_range": {"start": "1960", "end": "2023"},
    }
]


def test_the_captured_two_country_fixture_carries_the_coverage_window() -> None:
    """The captured "life expectancy in Nigeria and Guatemala" envelope
    (`fixtures/resolve_life_expectancy_two_countries.json`) asked no time
    text at all, so the identity's `requested_window` is `{"kind": "all"}`,
    data-mcp's own vocabulary for "no specific span"
    (`data_mcp.get_data_digest._coverage_window_text`), naming no range of
    its own. The elected item's own `data_date_range`, `{"start": "1960",
    "end": "2023"}`, is the resolver's actual coverage, so `graph_envelope`
    states that instead of leaving the planner to guess."""
    envelope = json.loads(_TWO_COUNTRY_ENVELOPE.read_text())
    digest = graph_envelope(envelope)
    assert digest["window"] == {"start": 1960, "end": 2023, "source": "coverage"}


def test_a_range_window_with_both_endpoints_wins_over_coverage() -> None:
    """The real wire shape (`resolve_population_guatemala.json`,
    `mint_statvar_set_guatemala.json`): `requested_window.kind == "range"`,
    with the years at `start_year`/`end_year`, not `start`/`end`. Both
    endpoints present is a complete answer the question named outright, so
    it is stated as `source: "requested"` even though the item's own
    (different) coverage would otherwise be available."""
    identity = {
        **_STANDARD_IDENTITY,
        "requested_window": {"kind": "range", "start_year": "1990", "end_year": "2023"},
    }
    digest = graph_envelope(
        {
            "status": "definite",
            "parts": [_definite("spec_1", identity=identity, items=_ITEM_WITH_COVERAGE)],
        }
    )
    assert digest["window"] == {"start": 1990, "end": 2023, "source": "requested"}


def test_window_is_left_out_when_the_identity_states_none_and_no_item_covers() -> None:
    digest = graph_envelope(
        {"status": "definite", "parts": [_definite("spec_1", identity=_STANDARD_IDENTITY)]}
    )
    assert "window" not in digest


def test_the_kind_is_all_falls_back_to_the_items_own_coverage() -> None:
    identity = {**_STANDARD_IDENTITY, "requested_window": {"kind": "all"}}
    digest = graph_envelope(
        {
            "status": "definite",
            "parts": [_definite("spec_1", identity=identity, items=_ITEM_WITH_COVERAGE)],
        }
    )
    assert digest["window"] == {"start": 1960, "end": 2023, "source": "coverage"}


def test_the_kind_is_all_and_no_item_covers_leaves_window_out() -> None:
    identity = {**_STANDARD_IDENTITY, "requested_window": {"kind": "all"}}
    digest = graph_envelope(
        {"status": "definite", "parts": [_definite("spec_1", identity=identity)]}
    )
    assert "window" not in digest


def test_window_is_left_out_when_the_kind_is_latest_even_with_coverage() -> None:
    """`latest` names a single most-recent observation. An item's own
    multi-year coverage must not be read as one."""
    identity = {**_STANDARD_IDENTITY, "requested_window": {"kind": "latest"}}
    digest = graph_envelope(
        {
            "status": "definite",
            "parts": [_definite("spec_1", identity=identity, items=_ITEM_WITH_COVERAGE)],
        }
    )
    assert "window" not in digest


def test_an_open_ended_range_is_completed_from_the_items_own_coverage() -> None:
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
            "parts": [_definite("spec_1", identity=identity, items=_ITEM_WITH_COVERAGE)],
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
    must reach the single-series rule through the leaf `data_mcp.substitution` module
    instead, never through `data_mcp.get_data_tool` or `data_mcp.search_tool`.

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


def test_skills_keep_every_named_skill_for_the_seam_to_count() -> None:
    skill_ids = [f"skill-{index}" for index in range(7)]
    envelope = _named_skills_envelope([_skilled("spec_a", *skill_ids)], skill_ids)

    digest = graph_envelope(envelope)

    assert [skill["id"] for skill in digest["skills"]] == skill_ids
    rendered = _render_envelope(digest).splitlines()
    skills_at = rendered.index("- skills (7):")
    assert rendered[skills_at + 1 : skills_at + 7] == [
        *(f"  - {skill_id} -- When {skill_id} applies." for skill_id in skill_ids[:5]),
        "  - and 2 more",
    ]


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
            "scope": {"place": {"dcid": "country/KEN", "label": "Kenya"}},
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
