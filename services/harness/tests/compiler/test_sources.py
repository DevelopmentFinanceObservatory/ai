"""Tests for `compiler.sources`: reading a `search` envelope into selectable sources."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from compiler.sources import (
    DiscoveryStatus,
    MembersAnswer,
    block_wire_groups,
    bound_refs,
    is_family,
    measure_named_by_query,
    members_reading,
    outcome_from_envelope,
    pinned_stat_var,
    readable_source,
    readable_wire_groups,
    source_from_spec_block,
    stat_var_count,
    total_reading,
    wire_groups,
)
from planner.records import EntityCandidate, EntityKey, Period
from planner.workflow import BoundEntity, OpenEntityAxis, SelectedSource

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


def _rows(*rows: dict[str, Any]) -> dict[str, Any]:
    """The restatement entity rows of a part."""
    return {"entities": list(rows)}


_CONGO_PART = {
    "restatement": _rows(
        {"text": "Congo", "dcid": "country/COD", "role_text": "from"},
        {"text": "Congo", "dcid": "country/COG", "role_text": "to"},
    ),
    "entity_refs": [
        {"text": "Congo", "refs": ["country/COD"]},
        {"text": "Congo", "refs": ["country/COG"]},
    ],
}


@pytest.mark.parametrize(
    ("part", "expected"),
    [
        (
            {
                "restatement": _rows(
                    {"text": "G7", "role_text": "from"}, {"text": "Africa", "role_text": "to"}
                ),
                "entity_refs": [
                    {"text": "G7", "refs": ["GroupOfSeven"]},
                    {"text": "Africa", "refs": ["DACAfrica"]},
                ],
            },
            [
                BoundEntity(identity="G7", text="G7", role="from", refs=("GroupOfSeven",)),
                BoundEntity(identity="Africa", text="Africa", role="to", refs=("DACAfrica",)),
            ],
        ),
        # A row Search bound nothing for names nothing by ref.
        (
            {
                "restatement": _rows({"text": "Atlantis"}),
                "entity_refs": [{"text": "Atlantis", "refs": []}],
            },
            [],
        ),
        # A Search that predates `entity_refs` states none, whatever it restates.
        ({"restatement": _rows({"text": "G7", "role_text": "from"})}, []),
        # Nor does one stating an entry per text rather than per row.
        (
            {
                "restatement": _rows({"text": "Congo"}, {"text": "Congo"}),
                "entity_refs": [{"text": "Congo", "refs": ["country/COD", "country/COG"]}],
            },
            [],
        ),
        # Two rows written alike stay two entities, each keyed by what it pins or names.
        (
            _CONGO_PART,
            [
                BoundEntity(
                    identity="country/COD", text="Congo", role="from", refs=("country/COD",)
                ),
                BoundEntity(identity="country/COG", text="Congo", role="to", refs=("country/COG",)),
            ],
        ),
        (
            {
                "restatement": _rows(
                    {"text": "Congo", "name": "Republic of the Congo"},
                    {"text": "Congo"},
                ),
                "entity_refs": [
                    {"text": "Congo", "refs": ["country/COG"]},
                    {"text": "Congo", "refs": ["country/COD"]},
                ],
            },
            [
                BoundEntity(identity="Republic of the Congo", text="Congo", refs=("country/COG",)),
                BoundEntity(identity="Congo", text="Congo", refs=("country/COD",)),
            ],
        ),
    ],
    ids=["texts", "unbound", "no-entity-refs", "per-text", "pinned-alike", "named-alike"],
)
def test_each_entity_row_keeps_the_refs_it_bound(
    part: dict[str, Any], expected: list[BoundEntity]
) -> None:
    assert list(bound_refs(part)) == expected


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


@pytest.mark.parametrize(
    ("labels", "read"),
    [
        (
            {
                "concept_label": "health",
                "published_tables": "OECD Creditor Reporting System (CRS) and ONE Imputed "
                "Multilateral Sectors",
            },
            (
                "health",
                "OECD Creditor Reporting System (CRS) and ONE Imputed Multilateral Sectors",
            ),
        ),
        ({"concept_label": None, "published_tables": None}, (None, None)),
        ({}, (None, None)),
    ],
    ids=["stated", "null", "before-the-fields"],
)
def test_the_concept_and_tables_search_states_are_read_as_stated(
    labels: dict[str, Any], read: tuple[str | None, str | None]
) -> None:
    source = source_from_spec_block(_spec(**labels))

    assert (source.concept_label, source.published_tables) == read
    # A source stating neither keeps the serialised form of one saved before them.
    dumped = source.model_dump()
    assert ("concept_label" in dumped, "published_tables" in dumped) == (
        read[0] is not None,
        read[1] is not None,
    )


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
    # envelope. `entity_set_too_large` is real on apis' own feature
    # branch and absent from its published `dev` vocabulary at the time of
    # writing -- exactly the kind of value this has to survive.
    outcome = outcome_from_envelope(
        _envelope("no_data", no_data={"reason": "entity_set_too_large", "category": "unresolved"})
    )

    assert outcome.status is DiscoveryStatus.NO_DATA
    assert outcome.reason == "entity_set_too_large"


_CONGO_CANDIDATES = [
    {"ref": "country/COD", "label": "Democratic Republic of the Congo", "kind": "country"},
    {"ref": "country/COG", "label": "Republic of the Congo", "kind": "country"},
]


def _entity_ambiguous(*rows: dict[str, Any]) -> dict[str, Any]:
    """Search's `entity_ambiguous` answer to the parsed door's `rows`, as C50c and C50e state it.

    Rows sent by text alone share one identity, so every "Congo" row is one
    group, and each option pins all of them to its candidate. Every option
    also pins a row that resolved to one node, here "France".
    """

    def pinned(row: dict[str, Any], ref: str) -> dict[str, Any]:
        return {**row, "dcid": ref if row["text"] == "Congo" else "country/FRA"}

    return {
        "reason": "entity_ambiguous",
        "category": "unresolved",
        "ambiguous_entities": [{"text": "Congo", "candidates": _CONGO_CANDIDATES}],
        "options": [
            {
                "mentions": ["Congo"],
                "label": candidate["label"],
                "entity_choice": 0,
                "data_query": {
                    "parts": [
                        {
                            "variable": "aid",
                            "entities": [pinned(row, candidate["ref"]) for row in rows],
                        }
                    ]
                },
            }
            for candidate in _CONGO_CANDIDATES
        ],
    }


_FROM_CONGO = {"text": "Congo", "role_text": "donor"}
_TO_CONGO = {"text": "Congo", "role_text": "recipient"}
_TO_FRANCE = {"text": "France", "role_text": "recipient"}


@pytest.mark.parametrize(
    ("rows", "pinned"),
    [
        ((_FROM_CONGO, _TO_CONGO), (0, 1)),
        ((_FROM_CONGO, _TO_FRANCE), (0,)),
    ],
    ids=["both_ends", "one_end"],
)
def test_no_data_reads_an_ambiguous_wording_once_per_entity_it_pins(
    rows: tuple[dict[str, Any], ...], pinned: tuple[int, ...]
) -> None:
    entities = [EntityKey(name=row["text"], role=row["role_text"]) for row in rows]

    outcome = outcome_from_envelope(
        _envelope("no_data", no_data=_entity_ambiguous(*rows)), entities=entities
    )

    # One reading per entity the group pins, each its own choice.
    readings = outcome.ambiguous_entities
    assert [reading.entity for reading in readings] == [entities[i] for i in pinned]
    for reading in readings:
        assert reading.text == "Congo"
        assert reading.candidates == (
            EntityCandidate(ref="country/COD", label="Democratic Republic of the Congo"),
            EntityCandidate(ref="country/COG", label="Republic of the Congo"),
        )


def _no_option_settles_the_group(no_data: dict[str, Any]) -> None:
    for option in no_data["options"]:
        option["entity_choice"] = 1


def _an_option_drops_a_row(no_data: dict[str, Any]) -> None:
    no_data["options"][0]["data_query"]["parts"][0]["entities"].pop()


def _the_options_pin_alike(no_data: dict[str, Any]) -> None:
    no_data["options"][1] = no_data["options"][0]


def _a_candidate_has_no_ref(no_data: dict[str, Any]) -> None:
    del no_data["ambiguous_entities"][0]["candidates"][0]["ref"]


@pytest.mark.parametrize(
    "malform",
    [
        _no_option_settles_the_group,
        _an_option_drops_a_row,
        _the_options_pin_alike,
        _a_candidate_has_no_ref,
    ],
)
def test_an_ambiguous_entity_group_pinning_no_sent_row_is_refused(
    malform: Callable[[dict[str, Any]], None],
) -> None:
    no_data = _entity_ambiguous(_FROM_CONGO, _TO_CONGO)
    malform(no_data)
    entities = [EntityKey(name="Congo", role="donor"), EntityKey(name="Congo", role="recipient")]

    with pytest.raises(ValueError, match=r"ambiguous entity group|ref"):
        outcome_from_envelope(_envelope("no_data", no_data=no_data), entities=entities)


_CENTRAL_AFRICA = [
    {"ref": "region/UN_CentralAfrica", "label": "Central Africa (UN)", "kind": "group"},
    {"ref": "region/AU_CentralAfrica", "label": "Central Africa (AU)", "kind": "region"},
    {"ref": "region/WB_CentralAfrica", "label": "Central Africa (WB)", "kind": "region"},
]


def _members_ambiguous(*options: dict[str, Any]) -> dict[str, Any]:
    """Search's answer to one row asked for its members (`"members": true`), as
    `conjoin._entity_options` builds its options: a group node posts the group's
    registered name and no dcid, any other candidate its dcid."""
    return {
        "reason": "entity_ambiguous",
        "category": "unresolved",
        "ambiguous_entities": [{"text": "Central Africa", "candidates": _CENTRAL_AFRICA}],
        "options": [
            {
                "mentions": ["Central Africa"],
                "label": label,
                "entity_choice": 0,
                "data_query": {
                    "parts": [{"variable": "aid", "entities": [{**row, "members": True}]}]
                },
            }
            for label, row in options
        ],
    }


_BY_GROUP_NAME = ("Central Africa (UN)", {"text": "Central Africa (UN)"})
_AU = ("Central Africa (AU)", {"text": "Central Africa", "dcid": "region/AU_CentralAfrica"})
_WB = ("Central Africa (WB)", {"text": "Central Africa", "dcid": "region/WB_CentralAfrica"})
_UN_KEPT = ("region/UN_CentralAfrica", "Central Africa (UN)")
_AU_KEPT = ("region/AU_CentralAfrica", None)
_WB_KEPT = ("region/WB_CentralAfrica", None)


@pytest.mark.parametrize(
    ("options", "kept"),
    [
        ((_BY_GROUP_NAME, _AU, _WB), [_UN_KEPT, _AU_KEPT, _WB_KEPT]),
        ((_BY_GROUP_NAME, _AU), [_UN_KEPT, _AU_KEPT]),
        # A group node with no registered name has no option, so it is no candidate.
        ((_AU, _WB), [_AU_KEPT, _WB_KEPT]),
    ],
    ids=["un-au-wb", "un-au", "no-group-name"],
)
def test_a_members_row_keeps_a_group_node_by_the_name_its_option_posts(
    options: tuple[tuple[str, dict[str, Any]], ...], kept: list[tuple[str, str | None]]
) -> None:
    entities = [EntityKey(name="Central Africa", role="recipient")]

    outcome = outcome_from_envelope(
        _envelope("no_data", no_data=_members_ambiguous(*options)), entities=entities
    )

    (reading,) = outcome.ambiguous_entities
    assert reading.entity == entities[0]
    assert [(c.ref, c.restate_as) for c in reading.candidates] == kept


def _shared_label(no_data: dict[str, Any]) -> None:
    no_data["ambiguous_entities"][0]["candidates"][1]["label"] = "Central Africa (UN)"


def _not_members(no_data: dict[str, Any]) -> None:
    del no_data["options"][0]["data_query"]["parts"][0]["entities"][0]["members"]


def _no_text(no_data: dict[str, Any]) -> None:
    no_data["options"][0]["data_query"]["parts"][0]["entities"][0]["text"] = ""


@pytest.mark.parametrize("others", [(_AU,), (_AU, _WB)], ids=["au", "au-wb"])
@pytest.mark.parametrize(
    "malform",
    [_shared_label, _not_members, _no_text],
    ids=["shared-label", "not-members", "no-text"],
)
def test_a_members_option_no_one_candidate_can_be_stated_by_is_refused(
    malform: Callable[[dict[str, Any]], None], others: tuple[tuple[str, dict[str, Any]], ...]
) -> None:
    # The UN option cannot be replayed as one candidate's, so the group is
    # refused whatever choice the others would leave.
    no_data = _members_ambiguous(_BY_GROUP_NAME, *others)
    malform(no_data)

    with pytest.raises(ValueError, match="ambiguous entity group 0 has an option"):
        outcome_from_envelope(
            _envelope("no_data", no_data=no_data),
            entities=[EntityKey(name="Central Africa", role="recipient")],
        )


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


def test_a_readable_source_is_none_with_a_warning_for_an_unusable_block(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A caller naming by the source reads an unusable block as none, logged once,
    and a usable one as `source_from_spec_block` reads it."""
    block = _spec()
    assert readable_source(block) == source_from_spec_block(block)
    with caplog.at_level("WARNING", logger="compiler.sources"):
        assert readable_source(_spec("spec-abc")) is None
    assert [record.levelname for record in caplog.records] == ["WARNING"]


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


@pytest.mark.parametrize(
    ("column", "fact_count", "count"),
    [
        # On a flow side the fact counts the parties and the scope also lists
        # group totals, which a flow retrieval does not return.
        pytest.param("origin", 2, 2, id="flow-side-fact-counts-parties"),
        pytest.param("origin", None, 3, id="flow-side-uncounted-fact-keeps-scope"),
        # On the place axis the scope lists the members it counts.
        pytest.param("observationAbout", 55, 3, id="place-axis-keeps-scope"),
    ],
)
def test_a_scope_and_a_role_axis_fact_on_one_column_count_parties_on_a_flow_side(
    column: str, fact_count: int | None, count: int
) -> None:
    members = [
        {"ref": ref, "label": ref, "entity_type": "Country", "role": column}
        for ref in ("country/FRA", "country/GBR", "DACCountries")
    ]
    block = _spec(
        entities=[],
        scope={"axis": column, "members": members},
        role_axis_facts=[{"role": column, "member_count": fact_count, "members_available": True}],
    )

    assert source_from_spec_block(block).open_entity_axes == (
        OpenEntityAxis(column=column, member_count=count),
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
# total, and the G7 read as its members, as apis states `entity_groups`.
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


def test_a_resubmit_posts_the_registry_id_of_a_set_the_graph_holds_no_node_for() -> None:
    # Search keeps a set with no node of its own, read as its members, only through
    # its registry id, which restores the total its dataset publishes for it.
    asia = {"label": "Asia", "group_id": "groups/OECD.DAC.Region.Asia", "total": None}
    source = source_from_spec_block(_spec(entities=_ENTITIES, entity_groups={"destination": asia}))

    assert wire_groups(source.groups) == {
        "destination": {"label": "Asia", "group_id": "groups/OECD.DAC.Region.Asia"}
    }


def test_a_resubmit_reads_each_sets_posted_identity_whatever_else_it_states() -> None:
    # Africa names a class of party this release does not know, so the full set is
    # not readable; a resubmit posts only its label, ref and registry id.
    africa = {**_GROUPS["destination"], "whole_of": ["sovereign_fund"], "group_id": None}
    block = _spec(entities=_ENTITIES, entity_groups={**_GROUPS, "destination": africa})
    with pytest.raises(ValueError, match="sovereign_fund"):
        source_from_spec_block(block)

    assert wire_groups(block_wire_groups(block)) == {
        "destination": {"label": "Africa", "ref": "africa"},
        "origin": {"label": "G7"},
    }
    assert wire_groups(block_wire_groups(block, leaving_out="destination")) == {
        "origin": {"label": "G7"}
    }


@pytest.mark.parametrize(
    "africa",
    [
        pytest.param({"label": "Africa", "ref": 5}, id="ref-not-text"),
        pytest.param({"label": "Africa", "group_id": 7}, id="group-id-not-text"),
        pytest.param("Africa", id="not-an-object"),
    ],
)
def test_a_resubmit_refuses_a_set_whose_posted_identity_does_not_read(africa: object) -> None:
    block = _spec(entities=_ENTITIES, entity_groups={**_GROUPS, "destination": africa})

    with pytest.raises(ValueError, match="destination"):
        block_wire_groups(block)
    assert readable_wire_groups(block) is None
    assert wire_groups(block_wire_groups(block, leaving_out="destination")) == {
        "origin": {"label": "G7"}
    }


@pytest.mark.parametrize(
    "africa",
    [
        pytest.param({"ref": "africa", "total": "DACAfrica"}, id="no-label"),
        pytest.param({"label": "", "ref": "africa"}, id="empty-label"),
    ],
)
def test_a_resubmit_posts_without_a_set_stating_no_label(africa: object) -> None:
    # A set stating no label names nothing a resubmit can post, and the source
    # reading the block skips it too.
    block = _spec(entities=_ENTITIES, entity_groups={**_GROUPS, "destination": africa})

    assert [group.role for group in source_from_spec_block(block).groups] == ["origin"]
    assert wire_groups(readable_wire_groups(block) or ()) == {"origin": {"label": "G7"}}


# "ODA to Ethiopia from all donors" binds the OECD's official-donors total; the
# union "all donors including private" binds that total and the private one.
_WHOLE_ENTITIES = [
    {"role": "origin", "label": "Official Donors, Total", "ref": "DACOfficialDonors"},
    {"role": "origin", "label": "Private Donors, Total", "ref": "DACPrivateDonors"},
]
_WHOLE_TOTAL = {
    "origin": {
        "ref": None,
        "label": "all donors",
        "entity_type": None,
        "total": "DACOfficialDonors",
        "whole_of": ["official"],
    }
}
_WHOLE_SET = {
    "origin": {
        "ref": None,
        "label": "all donors including private",
        "entity_type": None,
        "total": None,
        "whole_of": ["official", "private"],
    }
}


def test_a_group_carries_the_classes_its_whole_total_holds() -> None:
    sided = {"origin": _WHOLE_TOTAL["origin"] | {"side_classes": ["official", "private"]}}
    single = source_from_spec_block(_spec(entities=_WHOLE_ENTITIES, entity_groups=sided))
    union = source_from_spec_block(_spec(entities=_WHOLE_ENTITIES, entity_groups=_WHOLE_SET))
    members = source_from_spec_block(_spec(entities=_ENTITIES, entity_groups=_GROUPS))

    assert [(g.total, g.whole_of, g.side_classes) for g in single.groups] == [
        ("DACOfficialDonors", ("official",), ("official", "private"))
    ]
    assert [(g.total, g.whole_of, g.side_classes) for g in union.groups] == [
        (None, ("official", "private"), ())
    ]
    assert [(g.whole_of, g.side_classes) for g in members.groups] == [((), ()), ((), ())]


def test_a_group_naming_a_party_kind_as_its_class_fails_the_block() -> None:
    """`party` is a frame kind and no class of party, so a block from a Search
    that wrote kinds into `whole_of` fails rather than being read as some class."""
    kinds = {"origin": _WHOLE_TOTAL["origin"] | {"whole_of": ["party"]}}

    with pytest.raises(ValueError):
        source_from_spec_block(_spec(entities=_WHOLE_ENTITIES, entity_groups=kinds))


@pytest.mark.parametrize(
    ("side_classes", "outside"),
    [
        (["official", "private", "residual"], " (private donors are outside it)"),
        (["official", "residual"], ""),
        (None, ""),
    ],
    ids=["dac2a-publishes-private-donors", "dac2b-publishes-none", "search-without-the-field"],
)
def test_a_whole_total_is_read_with_the_population_it_holds(
    side_classes: list[str] | None, outside: str
) -> None:
    labels = {entity["ref"]: entity["label"] for entity in _WHOLE_ENTITIES}
    total = _WHOLE_TOTAL["origin"] | (
        {} if side_classes is None else {"side_classes": side_classes}
    )
    (single,) = source_from_spec_block(
        _spec(entities=_WHOLE_ENTITIES, entity_groups={"origin": total})
    ).groups
    (union,) = source_from_spec_block(
        _spec(entities=_WHOLE_ENTITIES, entity_groups=_WHOLE_SET)
    ).groups

    assert total_reading(single, labels) == (
        "The figures for all donors are Official Donors, Total, the total the source "
        f"publishes for all official donors{outside}, "
        "not its members' figures added up."
    )
    assert total_reading(union, labels) == (
        "The figure for all donors including private adds up the totals the source "
        "publishes for all official donors and private donors, not its members' figures."
    )


def test_a_set_read_as_its_whole_total_is_not_also_read_as_members() -> None:
    (union,) = source_from_spec_block(
        _spec(entities=_WHOLE_ENTITIES, entity_groups=_WHOLE_SET)
    ).groups

    assert members_reading(union, answer=MembersAnswer.SUMMED) is None


# Sets summed as their members, as Search states them: `left_out` names the
# residuals the set's published total counts beside the members, absent where
# the members leave none out and for a set with no published total, such as
# the G7 read as its members. `sum_differs` says the source publishes its own
# figure for the set, which can differ from the members' sum, in a sentence of
# its own, so the residuals' list ends before it.
_AFRICA_LEFT_OUT = (
    " and leaves out what the data allocates to no single country: "
    "Africa, unallocated by country (DAC); North of Sahara, unallocated by country (DAC)."
)
_PUBLISHED = (
    " The source publishes its own figure for {label}, "
    "which can differ from one built from its members."
)


@pytest.mark.parametrize(
    ("group", "tail"),
    [
        (
            {"origin": {"ref": "DAC", "label": "DAC members", "total": None}},
            ".",
        ),
        (
            {"destination": {"ref": "africa", "label": "Africa", "total": None}},
            ".",
        ),
        (
            {"origin": {"ref": None, "label": "G7", "total": None}},
            ".",
        ),
        (
            {
                "destination": {
                    "ref": "africa",
                    "label": "Africa",
                    "total": None,
                    "left_out": [
                        "Africa, unallocated by country (DAC)",
                        "North of Sahara, unallocated by country (DAC)",
                    ],
                }
            },
            _AFRICA_LEFT_OUT,
        ),
        (
            {"origin": {"ref": None, "label": "G7", "total": None, "sum_differs": True}},
            "." + _PUBLISHED.format(label="G7"),
        ),
        (
            {
                "destination": {
                    "ref": "africa",
                    "label": "Africa",
                    "total": None,
                    "left_out": [
                        "Africa, unallocated by country (DAC)",
                        "North of Sahara, unallocated by country (DAC)",
                    ],
                    "sum_differs": True,
                }
            },
            _AFRICA_LEFT_OUT + _PUBLISHED.format(label="Africa"),
        ),
        (
            {
                "origin": {
                    "ref": "DAC",
                    "label": "DAC members",
                    "total": None,
                    "left_out": ["DAC donors, unspecified"],
                }
            },
            " and leaves out what the source's total for DAC members counts beside them: "
            "DAC donors, unspecified.",
        ),
    ],
    ids=[
        "donor-members-leave-nothing-out",
        "recipient-members-leave-nothing-out",
        "donor-set-with-no-published-total",
        "recipient-members-leave-out-residuals",
        "donor-total-differs-from-its-members-sum",
        "residuals-end-before-the-published-figure",
        "donor-members-leave-out-residuals",
    ],
)
def test_summed_members_state_only_the_residuals_search_names(
    group: dict[str, dict[str, object]], tail: str
) -> None:
    (read,) = source_from_spec_block(_spec(entities=_ENTITIES, entity_groups=group)).groups

    assert members_reading(read, answer=MembersAnswer.SUMMED) == (
        f"The figure for {read.label} adds up its members' figures{tail}"
    )


def test_a_stored_set_reads_the_residuals_it_was_stored_with() -> None:
    stated = {"ref": "africa", "label": "Africa", "total": None}
    blocks = [
        {"destination": stated},
        {"destination": {**stated, "left_out": ["Africa, unallocated by country (DAC)"]}},
    ]
    stored = [
        SelectedSource.model_validate(
            source_from_spec_block(_spec(entities=_ENTITIES, entity_groups=block)).model_dump(
                mode="json"
            )
        )
        for block in blocks
    ]

    assert [
        members_reading(source.groups[0], answer=MembersAnswer.SUMMED) for source in stored
    ] == [
        "The figure for Africa adds up its members' figures.",
        "The figure for Africa adds up its members' figures and leaves out what the data "
        "allocates to no single country: Africa, unallocated by country (DAC).",
    ]


def test_a_resubmit_posts_a_whole_total_without_its_marker() -> None:
    source = source_from_spec_block(_spec(entities=_WHOLE_ENTITIES, entity_groups=_WHOLE_TOTAL))

    assert wire_groups(source.groups) == {"origin": {"label": "all donors"}}


def test_a_block_carries_its_parts_bound_rows_back_into_the_source() -> None:
    rows = bound_refs(_CONGO_PART)
    block = {**_spec(), "_bound_refs": [row.model_dump(mode="json") for row in rows]}

    assert source_from_spec_block(block).bound_refs == rows


def test_a_block_holding_bound_refs_by_text_is_not_usable() -> None:
    block = {**_spec(), "_bound_refs": {"congo": ["country/COD", "country/COG"]}}

    with pytest.raises(ValueError, match="not usable as a source"):
        source_from_spec_block(block)


def test_a_source_saved_before_entity_refs_dumps_as_it_did() -> None:
    source = source_from_spec_block(_spec())

    dumped = source.model_dump(mode="json")

    assert "entity_refs" not in dumped
    assert "elected_measure" not in dumped


def _alternative(spec_id: str, label: str, axis: str) -> dict[str, Any]:
    """One `resolved.alternative_measures` entry as apis states it."""
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
