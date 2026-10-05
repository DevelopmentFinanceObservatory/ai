"""Tests for `compiler.sources`: reading a `search` envelope into selectable sources."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from compiler.sources import (
    DiscoveryStatus,
    outcome_from_envelope,
    pinned_stat_var,
    source_from_spec_block,
)
from data_mcp.substitution import MAX_RESOLVED_STAT_VAR_DCIDS
from planner.records import Period
from planner.workflow import OpenEntityAxis

SPEC_KENYA = "spec_" + "a" * 32
SPEC_OTHER = "spec_" + "b" * 32

_PLACE_SCOPE_BLOCK = Path(__file__).parent.parent / "fixtures" / "search_spec_place_scope.json"


_ROLE_OPEN_BLOCK = Path(__file__).parent.parent / "fixtures" / "search_spec_role_open.json"


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
        "stat_vars": {
            "items": [{"ref": "DebtServicePaid", "label": "Debt service paid", "slot_values": []}],
            "member_count": 1,
            "truncated": False,
        },
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
    assert source.stat_vars == ("DebtServicePaid",)
    assert (source.stat_var_member_count, source.stat_var_truncated) == (1, False)
    assert [(e.role, e.ref, e.label) for e in source.entities] == [
        ("destination", "country/KEN", "Kenya")
    ]
    assert (source.unit_ref, source.unit_label) == ("usd", "US dollars")
    assert (source.provenance_ref, source.provenance_label) == ("oecd/dac/crs", "OECD DAC CRS")
    assert source.coverage_kind == "exact"
    assert source.coverage_window == Period(start=2010, end=2024)
    assert source.identity == {"stat_var": "x"}
    assert outcome.warnings == ()


def test_stat_vars_is_read_as_the_object_the_wire_actually_sends() -> None:
    # `resolved.stat_vars` is an object -- `items`, `member_count`,
    # `truncated` -- never a bare list. A reader that went back to treating
    # it as a list would find no `items` key on this mapping and read every
    # field back empty, so this fails first against that regression.
    block = _spec(
        stat_vars={
            "items": [{"ref": "DebtServicePaid", "slot_values": []}],
            "member_count": 250,
            "truncated": True,
        }
    )

    source = source_from_spec_block(block)

    assert source.stat_vars == ("DebtServicePaid",)
    assert (source.stat_var_member_count, source.stat_var_truncated) == (250, True)


def test_a_bare_list_stat_vars_reads_as_wholly_unknown() -> None:
    # The shape data-mcp actually sent before the fix: `search`'s own object
    # replaced by a bare list. `stat_vars` comes back empty exactly as
    # before, but `member_count`/`truncated` come back `None` rather than a
    # value the object shape would have supplied, so the caller can tell
    # "the envelope stated nothing" from "the definition binds nothing".
    block = _spec(stat_vars=[{"ref": "DebtServicePaid", "label": "Debt service paid"}])

    source = source_from_spec_block(block)

    assert source.stat_vars == ()
    assert (source.stat_var_member_count, source.stat_var_truncated) == (None, None)


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


@pytest.mark.parametrize(
    "items",
    [
        ["DebtServicePaid"],
        [{"ref": 7, "slot_values": []}],
        [{"ref": "", "slot_values": []}],
        [{"label": "Debt service paid", "slot_values": []}],
        {"ref": "DebtServicePaid"},
    ],
    ids=["a-bare-string", "a-number-ref", "an-empty-ref", "no-ref", "items-an-object"],
)
def test_an_elected_statvar_without_a_string_ref_is_refused(items: object) -> None:
    # Every elected StatVar is posted to `get_data`, so an item whose dcid
    # cannot be read refuses the block. Skipping it would post a shorter list
    # than the block states, and `str()` would post a dcid nobody elected.
    block = _spec(stat_vars={"items": items, "member_count": 1, "truncated": False})

    with pytest.raises(ValueError, match=r"not usable as a source: resolved\.stat_vars\.items"):
        source_from_spec_block(block)


def _items(refs: list[str]) -> dict[str, Any]:
    return {
        "items": [{"ref": ref, "slot_values": []} for ref in refs],
        "member_count": len(refs),
        "truncated": False,
    }


def test_an_elected_statvar_with_the_dcid_prefix_is_refused() -> None:
    # dfo-apis's mint refuses a posted `dcid:` ref, so the block's StatVars
    # could never be posted.
    block = _spec(stat_vars=_items(["DebtServicePaid", "dcid:DebtServiceDue"]))

    with pytest.raises(
        ValueError, match=r"items\[1\]\.ref 'dcid:DebtServiceDue' is not a bare dcid"
    ):
        source_from_spec_block(block)


def test_more_statvars_than_get_data_takes_are_refused() -> None:
    block = _spec(
        stat_vars=_items([f"Var_{index}" for index in range(MAX_RESOLVED_STAT_VAR_DCIDS + 1)])
    )

    with pytest.raises(
        ValueError,
        match=f"holds {MAX_RESOLVED_STAT_VAR_DCIDS + 1} StatVars, past the {MAX_RESOLVED_STAT_VAR_DCIDS}",
    ):
        source_from_spec_block(block)


def test_as_many_statvars_as_get_data_takes_are_read() -> None:
    refs = [f"Var_{index}" for index in range(MAX_RESOLVED_STAT_VAR_DCIDS)]

    assert source_from_spec_block(_spec(stat_vars=_items(refs))).stat_vars == tuple(refs)


def test_a_block_stating_null_items_elects_none() -> None:
    block = _spec(stat_vars={"items": None, "member_count": 0, "truncated": False})

    assert source_from_spec_block(block).stat_vars == ()


# --- Whether a described identity mints as one StatVar -------------------------------

_GENDER_SLOT_VALUE = {"key": {"property": {"dcid": "gender"}}, "value": {"ref": {"dcid": "Female"}}}
_GENDER_BLOCK_SLOT = {
    "key": {"axis": "how", "property": {"dcid": "gender", "label": "gender"}},
    "binding": {"kind": "value", "value": {"ref": {"dcid": "Female", "label": "Female"}}},
}


def _described(
    items: list[dict[str, Any]],
    *,
    gender: dict[str, Any] | None = None,
    entities: dict[str, list[str]] | None = None,
    slots: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """A spec block whose identity describes the population family of one place,
    its `gender` slot bound as `gender` states and unbound when it states nothing."""
    block = _spec(
        stat_vars={"items": items, "member_count": len(items), "truncated": False},
        slots=slots if slots is not None else ([_GENDER_BLOCK_SLOT] if gender else []),
    )
    block["definition"]["identity"] = {
        "shape": {"kind": "statvar_schema", "population_type": "Person"},
        "entities": entities if entities is not None else {"observationAbout": ["country/MEX"]},
        "slots": [
            {"property": "age", "binding": {"state": "unbound"}},
            {"property": "gender", "binding": gender or {"state": "unbound"}},
        ],
    }
    return block


_FEMALE = {"state": "value", "ref": "Female"}
_TOTAL = {"ref": "Count_Person", "slot_values": [], "constraint_properties": []}
_FEMALE_TOTAL = {
    "ref": "Count_Person_Female",
    "slot_values": [_GENDER_SLOT_VALUE],
    "constraint_properties": ["gender"],
}


def test_one_elected_statvar_asserting_no_constraint_is_the_root_of_an_unbound_family() -> None:
    block = _described([_TOTAL])

    assert source_from_spec_block(block).pinned is True
    assert (
        pinned_stat_var(block["definition"]["identity"], [_TOTAL], slots=block["resolved"]["slots"])
        == "Count_Person"
    )


def test_one_elected_statvar_constrained_by_the_bound_slot_alone_is_the_root() -> None:
    assert source_from_spec_block(_described([_FEMALE_TOTAL], gender=_FEMALE)).pinned is True


@pytest.mark.parametrize(
    "item",
    [
        {"ref": "Count_Person", "slot_values": []},
        {"ref": "Count_Person", "slot_values": [], "constraint_properties": None},
        {"ref": "Count_Person", "slot_values": [], "constraint_properties": [7]},
        {"ref": "Count_Person", "constraint_properties": []},
        {"ref": "Count_Person_Female", "slot_values": [], "constraint_properties": ["gender"]},
    ],
    ids=[
        "constraints-missing",
        "constraints-null",
        "constraints-malformed",
        "slot-values-missing",
        "constrained-beyond-the-bound-slots",
    ],
)
def test_an_elected_statvar_search_does_not_show_to_be_the_root_is_not_pinned(
    item: dict[str, Any],
) -> None:
    # A `null` or missing `constraint_properties` says Search does not know
    # what the StatVar asserts, which is not that it asserts nothing.
    assert source_from_spec_block(_described([item])).pinned is False


def test_a_statvar_constrained_beyond_the_bound_slot_is_not_pinned() -> None:
    # Its `slot_values` show the bound `gender` alone, so only
    # `constraint_properties` tells it from the root.
    urban = {
        "ref": "Count_Person_Female_Urban",
        "slot_values": [_GENDER_SLOT_VALUE],
        "constraint_properties": ["gender", "placeOfResidenceClassification"],
    }

    assert source_from_spec_block(_described([urban], gender=_FEMALE)).pinned is False


def test_a_statvar_holding_another_code_than_the_slot_binds_is_not_pinned() -> None:
    # Search lists a bound slot's value on an item only when the StatVar holds
    # that value, so a member constrained to a descendant code shows none.
    descendant = {
        "ref": "Count_Person_Female_Adult",
        "slot_values": [],
        "constraint_properties": ["gender"],
    }

    assert source_from_spec_block(_described([descendant], gender=_FEMALE)).pinned is False


def test_two_elected_statvars_are_not_pinned_even_when_one_is_the_root() -> None:
    assert source_from_spec_block(_described([_TOTAL, _FEMALE_TOTAL])).pinned is False


@pytest.mark.parametrize(
    "slots",
    [
        [{**_GENDER_BLOCK_SLOT, "expansion": {"added": [{"dcid": "Female_Adult"}]}}],
        [],
    ],
    ids=["expanded", "slot-not-listed"],
)
def test_a_bound_value_the_block_does_not_show_to_be_one_code_is_not_pinned(
    slots: list[dict[str, Any]],
) -> None:
    # dfo-apis mints a hierarchical code's whole subtree, which Search reports
    # as the slot's `expansion`.
    block = _described([_FEMALE_TOTAL], gender=_FEMALE, slots=slots)

    assert source_from_spec_block(block).pinned is False


def test_a_set_bound_slot_is_not_pinned() -> None:
    block = _described([_FEMALE_TOTAL], gender={"state": "set", "refs": ["Female", "Male"]})

    assert source_from_spec_block(block).pinned is False


@pytest.mark.parametrize(
    "entities",
    [{"origin": ["country/FRA"]}, {"observationAbout": ["country/MEX"], "origin": ["x"]}, {}],
    ids=["a-flow-role", "a-place-and-a-flow-role", "no-role"],
)
def test_an_identity_keyed_on_anything_but_one_place_role_is_not_pinned(
    entities: dict[str, list[str]],
) -> None:
    assert source_from_spec_block(_described([_TOTAL], entities=entities)).pinned is False


def test_an_identity_that_names_its_statvar_states_no_pin_verdict() -> None:
    block = _described([_TOTAL])
    block["definition"]["identity"]["shape"] = {"kind": "stat_var", "id": "Count_Person"}

    assert source_from_spec_block(block).pinned is None


def test_a_block_carrying_no_identity_states_no_pin_verdict() -> None:
    block = _described([_TOTAL])
    del block["definition"]

    assert source_from_spec_block(block).pinned is None


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
