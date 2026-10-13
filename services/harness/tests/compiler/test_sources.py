"""Tests for `compiler.sources`: reading a `search` envelope into selectable sources."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from compiler.sources import (
    DiscoveryStatus,
    is_family,
    measure_named_by_query,
    outcome_from_envelope,
    pinned_stat_var,
    source_from_spec_block,
    stat_var_count,
    wire_groups,
)
from planner.records import Period
from planner.workflow import OpenEntityAxis

SPEC_KENYA = "spec_" + "a" * 32
SPEC_OTHER = "spec_" + "b" * 32

_PLACE_SCOPE_BLOCK = Path(__file__).parent.parent / "fixtures" / "search_spec_place_scope.json"


_ROLE_OPEN_BLOCK = Path(__file__).parent.parent / "fixtures" / "search_spec_role_open.json"


_MEASURE_NAMED_BLOCKS = Path(__file__).parent.parent / "fixtures" / "search_measure_named.json"


def _place_scope_block() -> dict[str, Any]:
    """dev `search`'s own block for health spending per person, every country left open."""
    return json.loads(_PLACE_SCOPE_BLOCK.read_text())["spec"]


def _role_open_block() -> dict[str, Any]:
    """dev `search`'s own block for education aid to Kenya, every donor left open."""
    return json.loads(_ROLE_OPEN_BLOCK.read_text())["spec"]


def _spec(spec_id: str = SPEC_KENYA, **resolved: Any) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "rendered_sentence": "external debt service paid by Kenya, 2015 to 2024",
        "confidence": "high",
        "coverage": {
            "kind": "exact",
            "has_data": True,
            "observation_count": 10,
            "window": {"start_year": 2010, "end_year": 2024},
        },
        "stat_vars": {"member_count": 1, "member_count_exact": True, "served_count": 1},
        "entities": [{"role": "destination", "label": "Kenya", "ref": "country/KEN"}],
        "facets": {
            "unit": {"basis": "declared_default", "ref": "usd", "label": "US dollars"},
            "provenance": {"basis": "bound", "ref": "oecd/dac/crs", "label": "OECD DAC CRS"},
        },
    }
    return {
        "spec_id": spec_id,
        "resolved": fields | resolved,
        "definition": {"schema_version": 1, "spec_id": spec_id, "identity": {"stat_var": "x"}},
    }


def _part(status: str, **payload: Any) -> dict[str, Any]:
    return {"status": status, "variable_text": "debt service", "warnings": [], **payload}


def _envelope(status: str, **payload: Any) -> dict[str, Any]:
    """A single-part envelope, the shape one `discover_source` call gets back."""
    return {"schema_version": 1, "status": status, "parts": [_part(status, **payload)]}


def test_a_definite_answer_yields_the_one_source_it_resolved() -> None:
    outcome = outcome_from_envelope(_envelope("definite", spec=_spec()))

    assert outcome.status is DiscoveryStatus.DEFINITE
    (source,) = outcome.sources
    assert source.spec_id == SPEC_KENYA
    assert source.has_data is True
    assert (source.stat_var_member_count, source.served_count) == (1, 1)
    assert [(e.role, e.ref, e.label) for e in source.entities] == [
        ("destination", "country/KEN", "Kenya")
    ]
    assert (source.unit_ref, source.unit_label) == ("usd", "US dollars")
    assert (source.provenance_ref, source.provenance_label) == ("oecd/dac/crs", "OECD DAC CRS")
    assert source.coverage_kind == "exact"
    assert source.coverage_window == Period(start=2010, end=2024)
    assert source.identity == {"stat_var": "x"}
    assert outcome.warnings == ()


@pytest.mark.parametrize(
    ("unit", "read"),
    [
        ({"basis": "bound", "selection": "per_series"}, (None, True)),
        ({"basis": "declared_default", "ref": "usd", "label": "US dollars"}, ("usd", None)),
    ],
    ids=["per-series", "one-unit"],
)
def test_a_unit_that_varies_per_series_is_read_as_such(
    unit: dict[str, Any], read: tuple[str | None, bool | None]
) -> None:
    source = source_from_spec_block(_spec(facets={"unit": unit}))

    assert (source.unit_ref, source.unit_per_series) == read


@pytest.mark.parametrize(
    ("stat_vars", "read"),
    [
        (
            {"member_count": 250, "member_count_exact": True, "served_count": 250},
            (250, True, 250),
        ),
        (
            {"member_count": 40, "member_count_exact": False, "served_count": None},
            (40, False, None),
        ),
        # The shape data-mcp sent before `resolved.stat_vars` was read as an
        # object: a bare list states none of the three.
        ([{"ref": "DebtServicePaid"}], (None, None, None)),
    ],
    ids=["settled", "unsettled", "a-bare-list"],
)
def test_the_member_counts_are_read_from_the_stat_vars_object(
    stat_vars: object, read: tuple[int | None, bool | None, int | None]
) -> None:
    source = source_from_spec_block(_spec(stat_vars=stat_vars))

    assert (
        source.stat_var_member_count,
        source.stat_var_member_count_exact,
        source.served_count,
    ) == read


def test_the_shape_s_member_count_and_digest_are_read_as_the_binding() -> None:
    block = _spec(shape={"member_count": 183, "member_digest": "7f60" + "0" * 60})

    source = source_from_spec_block(block)

    assert (source.shape_member_count, source.shape_member_digest) == (183, "7f60" + "0" * 60)
    assert source.stat_var_member_count == 1


@pytest.mark.parametrize(
    "shape",
    [None, {}, {"member_count": 8, "member_digest": None}],
    ids=["no-shape", "no-member-fields", "inexact-population"],
)
def test_a_shape_without_a_digest_reads_as_none(shape: dict[str, Any] | None) -> None:
    block = _spec() if shape is None else _spec(shape=shape)

    source = source_from_spec_block(block)

    assert source.shape_member_digest is None
    assert source.shape_member_count == (shape or {}).get("member_count")


def test_a_source_without_a_provenance_facet_has_no_provenance() -> None:
    block = _spec()
    del block["resolved"]["facets"]["provenance"]

    source = source_from_spec_block(block)

    assert (source.provenance_ref, source.provenance_label) == (None, None)


def test_candidates_yield_every_candidate() -> None:
    outcome = outcome_from_envelope(
        _envelope(
            "candidates",
            candidates={"specs": [_spec(), _spec(SPEC_OTHER)], "truncated": False},
        )
    )

    assert outcome.status is DiscoveryStatus.CANDIDATES
    assert [source.spec_id for source in outcome.sources] == [SPEC_KENYA, SPEC_OTHER]


def test_no_data_carries_the_reason_and_the_nearest_real_series() -> None:
    outcome = outcome_from_envelope(
        _envelope(
            "no_data",
            no_data={
                "reason": "no series matches the ask",
                "category": "unresolved",
                "nearest_real": [_spec(SPEC_OTHER), {"axis": "where", "label": "not a spec"}],
            },
        )
    )

    assert outcome.status is DiscoveryStatus.NO_DATA
    assert outcome.detail == "no series matches the ask"
    assert outcome.reason == "no series matches the ask"
    assert [source.spec_id for source in outcome.sources] == [SPEC_OTHER]


def test_no_data_carries_a_reason_this_service_does_not_yet_recognise() -> None:
    # The reason vocabulary belongs to another service: a value it has not
    # been taught yet is still data to carry, not grounds to refuse the
    # envelope. `entity_set_too_large` is real on dfo-apis' own feature
    # branch and absent from its published `dev` vocabulary at the time of
    # writing -- exactly the kind of value this has to survive.
    outcome = outcome_from_envelope(
        _envelope("no_data", no_data={"reason": "entity_set_too_large", "category": "unresolved"})
    )

    assert outcome.status is DiscoveryStatus.NO_DATA
    assert outcome.reason == "entity_set_too_large"


def test_unsupported_carries_the_detail_and_names_no_source() -> None:
    outcome = outcome_from_envelope(
        _envelope("unsupported", unsupported={"detail": "source without unit", "pointer": "unit"})
    )

    assert outcome.status is DiscoveryStatus.UNSUPPORTED
    assert outcome.detail == "source without unit unit"
    assert outcome.reason is None
    assert outcome.sources == ()


def test_an_unrecognised_status_is_refused() -> None:
    with pytest.raises(ValueError, match="unrecognised status 'partial'"):
        outcome_from_envelope(_envelope("partial"))


def test_a_definite_part_without_a_spec_is_refused() -> None:
    with pytest.raises(ValueError, match="names no spec"):
        outcome_from_envelope(_envelope("definite"))


def test_an_envelope_with_other_than_one_part_is_refused() -> None:
    # Discovery asks one part per call; a compound answer has no single status.
    two = {
        "schema_version": 1,
        "status": "mixed",
        "parts": [_part("definite", spec=_spec()), _part("no_data", no_data={"reason": "none"})],
    }
    with pytest.raises(ValueError, match="2 parts"):
        outcome_from_envelope(two)
    with pytest.raises(ValueError, match="0 parts"):
        outcome_from_envelope({"schema_version": 1, "status": "definite", "parts": []})


def test_a_source_without_a_definition_sidecar_has_no_identity() -> None:
    # Unknown, not empty: drift detection cannot compare against an identity
    # that was never held.
    block = _spec()
    del block["definition"]

    assert source_from_spec_block(block).identity is None


def test_a_definition_carrying_an_empty_identity_keeps_it_empty() -> None:
    block = _spec()
    block["definition"]["identity"] = {}

    assert source_from_spec_block(block).identity == {}


def test_a_source_reports_when_it_has_no_data() -> None:
    block = _spec(coverage={"kind": "bare", "has_data": False})

    source = source_from_spec_block(block)

    assert source.has_data is False
    assert source.coverage_kind == "bare"
    assert source.coverage_window is None


def test_a_source_stating_no_coverage_has_unknown_data() -> None:
    # data-mcp's own digest reads `has_data` with no default; so does this.
    block = _spec()
    del block["resolved"]["coverage"]

    source = source_from_spec_block(block)

    assert source.has_data is None
    assert source.coverage_kind is None
    assert source.coverage_window is None


def test_a_breadth_coverage_carries_no_series_window() -> None:
    block = _spec(
        coverage={
            "kind": "breadth",
            "has_data": True,
            "window": {"start_year": 2010, "end_year": 2024},
        }
    )

    source = source_from_spec_block(block)

    assert source.coverage_kind == "breadth"
    assert source.coverage_window is None


def test_request_level_warnings_come_before_the_parts_own() -> None:
    envelope = _envelope("definite", spec=_spec())
    envelope["diagnostics"] = {
        "warnings": [
            {"code": "role_dropped", "severity": "warning", "message": "role 'creditor' dropped"}
        ]
    }
    envelope["parts"][0]["warnings"] = [
        {
            "code": "time_dropped",
            "severity": "info",
            "message": "time phrase dropped",
            "spec_id": SPEC_KENYA,
        }
    ]

    outcome = outcome_from_envelope(envelope)

    assert [(w.code, w.severity, w.message, w.spec_id) for w in outcome.warnings] == [
        ("role_dropped", "warning", "role 'creditor' dropped", None),
        ("time_dropped", "info", "time phrase dropped", SPEC_KENYA),
    ]


def test_a_warning_with_an_unknown_code_or_severity_is_carried_not_refused() -> None:
    envelope = _envelope(
        "no_data",
        no_data={"reason": "none", "category": "unresolved", "nearest_real": []},
    )
    envelope["parts"][0]["warnings"] = [{"code": "brand_new", "severity": "fatal"}]

    (warning,) = outcome_from_envelope(envelope).warnings

    assert (warning.code, warning.severity, warning.message) == ("brand_new", "fatal", "")


def test_a_block_with_a_malformed_spec_id_is_refused() -> None:
    with pytest.raises(ValueError, match="not usable as a source"):
        source_from_spec_block(_spec("spec-abc"))


def test_an_entity_without_its_role_is_refused() -> None:
    # A role is the column the entity binds to; without it a source cannot
    # be compiled against, so the block is refused rather than trimmed.
    with pytest.raises(ValueError, match="not usable as a source"):
        source_from_spec_block(_spec(entities=[{"label": "Kenya", "ref": "country/KEN"}]))


# --- Whether a source mints as one StatVar ------------------------------------------

_GENDER_BLOCK_SLOT = {
    "key": {"axis": "how", "property": {"ref": "gender", "label": "gender"}},
    "binding": {"kind": "value", "value": {"ref": {"ref": "Female", "label": "Female"}}},
}


def _described(*, slots: list[dict[str, Any]]) -> dict[str, Any]:
    """A spec block whose identity describes the population family of one place."""
    block = _spec(slots=slots)
    block["definition"]["identity"] = {
        "shape": {"kind": "statvar_schema", "population_type": "Person"},
        "entities": {"observationAbout": ["country/MEX"]},
        "slots": [
            {"property": "age", "binding": {"state": "unbound"}},
            {"property": "gender", "binding": {"state": "unbound"}},
        ],
    }
    return block


@pytest.mark.parametrize(
    ("served_count", "member_count", "family", "counted"),
    [(1, 1, False, 1), (3, 3, True, 3), (None, 1, True, 1), (None, None, True, 0)],
    ids=["one-served", "several-served", "unsettled-one-confirmed", "nothing-stated"],
)
def test_a_source_is_one_statvar_exactly_when_search_states_it_serves_one(
    served_count: int | None, member_count: int | None, family: bool, counted: int
) -> None:
    """Search states `served_count` only once the membership settled, so an
    unsettled membership reads as a family whatever it confirmed."""
    stat_vars = {"member_count": member_count, "served_count": served_count}
    source = source_from_spec_block(_spec(stat_vars=stat_vars))

    assert is_family(source) is family
    assert stat_var_count(source) == counted


@pytest.mark.parametrize(
    ("identity", "pinned"),
    [
        ({"shape": {"kind": "stat_var", "id": "Count_Person"}}, "Count_Person"),
        ({"shape": {"kind": "statvar_schema", "population_type": "Person"}}, None),
        ({"stat_var": "x"}, None),
        (None, None),
    ],
    ids=["pinned-arm", "described-arm", "no-shape", "no-identity"],
)
def test_only_a_pinned_identity_names_its_statvar(
    identity: dict[str, Any] | None, pinned: str | None
) -> None:
    assert pinned_stat_var(identity) == pinned


_AGE_UNBOUND = {
    "key": {"axis": "how", "property": {"ref": "age", "label": "age"}},
    "binding": {"kind": "unbound"},
}
_PLACE = {"key": {"axis": "where", "label": "place"}, "binding": {"kind": "unbound"}}


@pytest.mark.parametrize(
    ("gender_slot", "codes"),
    [
        pytest.param(_GENDER_BLOCK_SLOT, {"gender": ("Female",)}, id="one-code"),
        pytest.param(
            {**_GENDER_BLOCK_SLOT, "expansion": {"added": [{"ref": "Female_Adult"}]}},
            {"gender": ("Female", "Female_Adult")},
            id="a-value-and-its-descendants",
        ),
        pytest.param(
            {
                **_GENDER_BLOCK_SLOT,
                "binding": {
                    "kind": "set",
                    "values": [{"ref": {"ref": "Male"}}, {"ref": {"ref": "Female"}}],
                },
            },
            {"gender": None},
            id="a-set-with-no-expansion",
        ),
        pytest.param(
            {
                **_GENDER_BLOCK_SLOT,
                "binding": {"kind": "set", "values": [{"ref": {"ref": "Female"}}]},
                "expansion": {"added": [{"ref": "Female_Adult"}]},
            },
            {"gender": ("Female", "Female_Adult")},
            id="a-set-and-its-descendants",
        ),
        pytest.param(
            {**_GENDER_BLOCK_SLOT, "binding": {"kind": "range"}}, None, id="an-unknown-binding"
        ),
        pytest.param(
            {**_GENDER_BLOCK_SLOT, "binding": {"kind": "value", "value": {}}},
            None,
            id="a-value-without-a-ref",
        ),
    ],
)
def test_slot_codes_hold_every_code_a_bound_slot_admits(
    gender_slot: dict[str, Any], codes: dict[str, tuple[str, ...] | None] | None
) -> None:
    block = _described(slots=[_AGE_UNBOUND, gender_slot, _PLACE])

    assert source_from_spec_block(block).slot_codes == codes


def test_a_defaulted_place_scope_opens_its_axis_with_every_member_counted() -> None:
    source = source_from_spec_block(_place_scope_block())

    assert source.entities == ()
    assert source.open_entity_axes == (OpenEntityAxis(column="observationAbout", member_count=193),)


def test_a_worldwide_total_opens_its_axis_with_one_member() -> None:
    # `search` scopes a worldwide total to `["Earth"]` alone: open, but a
    # population of one, which a rank over it cannot use.
    block = _place_scope_block()
    block["resolved"]["scope"]["members"] = [
        {"ref": "Earth", "label": "World", "entity_type": "Place", "role": "observationAbout"}
    ]

    source = source_from_spec_block(block)

    assert source.open_entity_axes == (OpenEntityAxis(column="observationAbout", member_count=1),)


_ABSENT = object()


@pytest.mark.parametrize("members", [_ABSENT, None, 193, "every country", {"count": 193}])
def test_a_scope_whose_members_are_no_list_reads_as_uncounted(members: object) -> None:
    # A population nobody counted is unknown, not empty: zero would let a
    # rank over it through as a ranking of nothing.
    block = _place_scope_block()
    if members is _ABSENT:
        del block["resolved"]["scope"]["members"]
    else:
        block["resolved"]["scope"]["members"] = members

    assert source_from_spec_block(block).open_entity_axes == (
        OpenEntityAxis(column="observationAbout", member_count=None),
    )


def test_a_scope_listing_its_members_as_refs_counts_every_one() -> None:
    block = _place_scope_block()
    block["resolved"]["scope"]["members"] = ["Earth"]

    assert source_from_spec_block(block).open_entity_axes == (
        OpenEntityAxis(column="observationAbout", member_count=1),
    )


def test_an_open_flow_side_is_read_from_its_role_axis_fact() -> None:
    # Education aid to Kenya: the recipient is bound, the donors are left
    # open, and a flow block states them only as a role axis fact, with
    # `scope` null.
    block = _spec(
        role_axis_facts=[{"role": "origin", "member_count": 55, "members_available": True}],
        scope=None,
    )

    source = source_from_spec_block(block)

    assert [(e.role, e.label) for e in source.entities] == [("destination", "Kenya")]
    assert source.open_entity_axes == (OpenEntityAxis(column="origin", member_count=55),)


def test_a_recorded_open_donor_side_is_read_from_its_role_axis_fact() -> None:
    block = _role_open_block()
    assert "scope" not in block["resolved"]

    source = source_from_spec_block(block)

    assert [(e.role, e.ref) for e in source.entities] == [("destination", "country/KEN")]
    assert source.open_entity_axes == (OpenEntityAxis(column="origin", member_count=55),)
    assert source.identity is not None
    assert source.identity["entities"] == {"destination": ["country/KEN"]}


@pytest.mark.parametrize("count", [-1, True, "55"])
def test_a_role_axis_fact_count_that_is_no_count_reads_as_unknown(count: object) -> None:
    block = _spec(role_axis_facts=[{"role": "origin", "member_count": count}])

    assert source_from_spec_block(block).open_entity_axes == (
        OpenEntityAxis(column="origin", member_count=None),
    )


def test_both_open_flow_sides_are_read_and_an_uncounted_one_stays_unknown() -> None:
    block = _spec(
        entities=[],
        role_axis_facts=[
            {"role": "origin", "member_count": None, "members_available": False},
            {"role": "destination", "member_count": 140, "members_available": True},
        ],
    )

    source = source_from_spec_block(block)

    assert source.open_entity_axes == (
        OpenEntityAxis(column="destination", member_count=140),
        OpenEntityAxis(column="origin", member_count=None),
    )


def test_a_scope_and_a_role_axis_fact_on_one_column_keep_the_scopes_count() -> None:
    # The scope lists the members it counts; the fact only counts them.
    donors = [
        {"ref": ref, "label": ref, "entity_type": "Country", "role": "origin"}
        for ref in ("country/FRA", "country/GBR", "country/USA")
    ]
    block = _spec(
        entities=[],
        scope={"axis": "origin", "members": donors},
        role_axis_facts=[{"role": "origin", "member_count": 55, "members_available": True}],
    )

    assert source_from_spec_block(block).open_entity_axes == (
        OpenEntityAxis(column="origin", member_count=3),
    )


def test_an_axis_a_resolved_entity_binds_is_not_open() -> None:
    block = _spec(
        role_axis_facts=[
            {"role": "destination", "member_count": 1, "members_available": True},
            {"role": "origin", "member_count": 55, "members_available": True},
        ]
    )

    assert source_from_spec_block(block).open_entity_axes == (
        OpenEntityAxis(column="origin", member_count=55),
    )


def test_a_block_stating_no_scope_and_no_role_axis_fact_opens_nothing() -> None:
    assert source_from_spec_block(_spec()).open_entity_axes == ()


# Africa as a DAC2A recipient, bound to the one party DAC publishes as its
# total, and the G7 read as its members, as dfo-apis states `entity_groups`.
_GROUPS = {
    "destination": {
        "ref": "africa",
        "label": "Africa",
        "entity_type": "Continent",
        "total": "DACAfrica",
    },
    "origin": {"ref": None, "label": "G7", "entity_type": None, "total": None},
}
_ENTITIES = [
    {"role": "destination", "label": "Africa, Total", "ref": "DACAfrica"},
    {"role": "origin", "label": "France", "ref": "country/FRA"},
]


def test_a_set_bound_as_its_published_total_is_read_with_that_party() -> None:
    source = source_from_spec_block(_spec(entities=_ENTITIES, entity_groups=_GROUPS))

    assert [(group.role, group.ref, group.total) for group in source.groups] == [
        ("destination", "africa", "DACAfrica"),
        ("origin", None, None),
    ]


def test_a_resubmit_posts_each_set_without_its_total() -> None:
    source = source_from_spec_block(_spec(entities=_ENTITIES, entity_groups=_GROUPS))

    assert wire_groups(source.groups) == {
        "destination": {"label": "Africa", "ref": "africa"},
        "origin": {"label": "G7"},
    }


def test_a_source_saved_before_entity_refs_dumps_as_it_did() -> None:
    source = source_from_spec_block(_spec())

    dumped = source.model_dump(mode="json")

    assert "entity_refs" not in dumped
    assert "elected_measure" not in dumped


def _alternative(spec_id: str, label: str, axis: str) -> dict[str, Any]:
    """One `resolved.alternative_measures` entry as dfo-apis states it."""
    return {
        "definition": {"schema_version": 1, "spec_id": spec_id, "identity": {"stat_var": "y"}},
        "label": label,
        "axis": axis,
        "measure_axis_values": [],
        "served_rows": 9,
        "served_rows_exact": True,
    }


SPEC_CONSTANT = "spec_" + "c" * 32


def test_the_elected_measure_is_read_with_every_alternative_search_offers() -> None:
    block = _spec(
        shape={"label": "official development assistance"},
        measure_label="gross disbursements",
        alternative_measures=[
            _alternative(SPEC_OTHER, "net disbursements", "what"),
            _alternative(SPEC_CONSTANT, "constant prices", "unit"),
        ],
    )

    source = source_from_spec_block(block)

    assert source.measure_label == "official development assistance"
    assert source.elected_measure is not None
    assert source.elected_measure.label == "gross disbursements"
    assert source.elected_measure.alternatives[0].identity == {"stat_var": "y"}
    assert [(a.spec_id, a.label, a.axis) for a in source.elected_measure.alternatives] == [
        (SPEC_OTHER, "net disbursements", "what"),
        (SPEC_CONSTANT, "constant prices", "unit"),
    ]


@pytest.mark.parametrize(
    "alternative",
    [
        {"definition": {}, "label": "net disbursements", "axis": "what"},
        {"definition": {"spec_id": SPEC_OTHER}, "label": "", "axis": "what"},
        {"definition": {"spec_id": SPEC_OTHER}, "label": "net disbursements"},
    ],
    ids=["no-spec-id", "no-label", "no-axis"],
)
def test_an_alternative_naming_no_spec_label_or_axis_is_left_out(
    alternative: dict[str, Any],
) -> None:
    block = _spec(measure_label="gross disbursements", alternative_measures=[alternative])

    source = source_from_spec_block(block)

    assert source.elected_measure is not None
    assert source.elected_measure.alternatives == ()


def test_a_block_stating_no_measure_label_elects_no_measure() -> None:
    block = _spec(
        shape={"label": "external debt service"},
        alternative_measures=[_alternative(SPEC_OTHER, "net disbursements", "what")],
    )

    assert source_from_spec_block(block).elected_measure is None


def _measure_block(variable: str) -> dict[str, Any]:
    """dev `search`'s own block for `variable` from Japan to Kenya, gross disbursements elected."""
    return json.loads(_MEASURE_NAMED_BLOCKS.read_text())["blocks"][variable]


def _untraced(block: dict[str, Any]) -> dict[str, Any]:
    return {**block, "resolved": {k: v for k, v in block["resolved"].items() if k != "trace"}}


@pytest.mark.parametrize(
    ("block", "named"),
    [
        (_measure_block("gross ODA disbursements"), True),
        (_measure_block("ODA disbursements"), False),
        (_measure_block("gross ODA"), True),
        (_untraced(_measure_block("gross ODA disbursements")), False),
    ],
    ids=["gross-disbursements", "disbursements", "gross-oda", "no-trace"],
)
def test_the_query_names_the_measure_only_where_its_words_set_every_differing_slot(
    block: dict[str, Any], named: bool
) -> None:
    assert measure_named_by_query(block) is named


def test_ignored_source_constraints_are_not_saved_as_applied():
    envelope = {
        "parts": [
            {
                "status": "definite",
                "restatement": {"sources": [{"text": "XYZ", "ref": None, "polarity": "exclude"}]},
                "warnings": [
                    {
                        "code": "query_qualifier_dropped",
                        "severity": "warning",
                        "message": 'The requested exclusion "XYZ" was not applied.',
                    }
                ],
                "spec": {"spec_id": "spec_" + "a" * 32, "resolved": {"rendered_sentence": "aid"}},
            }
        ]
    }
    assert outcome_from_envelope(envelope).sources[0].resolved_sources == ()
