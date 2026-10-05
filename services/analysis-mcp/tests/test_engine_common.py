"""`narrow_and_resolve`: unit refusal/pin, facet auto-select vs. refusal,
entity narrowing; `record_exclusion`'s entity cap."""

from __future__ import annotations

import pytest
from analysis_mcp.engine.common import (
    Exclusions,
    FacetHeterogeneousError,
    ResolvedUnitAndFacets,
    UnitHeterogeneousError,
    UnitNotPresentError,
    narrow_and_resolve,
    record_exclusion,
    smallest_distinct,
)
from dfo_frame_grammar import Frame
from dfo_frame_grammar.grammar import FACET_IDENTITY_COLUMNS


def _frame(rows: list[dict[str, object]], *, role_keys: list[str] | None = None) -> Frame:
    return Frame.build(role_keys=role_keys or ["reporter"], rows=rows)


def _row(
    *, reporter: str = "dcid:USA", year: str = "2020", value: float = 1.0, **extra: object
) -> dict:
    return {
        "stat_var": "dcid:X",
        "reporter": reporter,
        "time_period": year,
        "value": value,
        **extra,
    }


def _resolved(
    frame: Frame, *, pinned_unit: str | None = None, exclude_unitless: bool = False
) -> ResolvedUnitAndFacets:
    _table, resolved, _exclusions = narrow_and_resolve(
        frame, pinned_unit=pinned_unit, exclude_unitless=exclude_unitless
    )
    return resolved


def test_single_declared_unit_resolves_cleanly() -> None:
    frame = _frame([_row(unit="dcid:USDollar"), _row(reporter="dcid:GBR", unit="dcid:USDollar")])
    resolved = _resolved(frame)
    assert resolved.unit == "dcid:USDollar"
    assert resolved.warnings == ()


def test_no_declared_unit_resolves_to_none() -> None:
    frame = _frame([_row()])
    resolved = _resolved(frame)
    assert resolved.unit is None


def test_heterogeneous_units_refuse() -> None:
    frame = _frame([_row(unit="dcid:USDollar"), _row(reporter="dcid:GBR", unit="dcid:Percent")])
    with pytest.raises(UnitHeterogeneousError) as exc_info:
        _resolved(frame)
    assert exc_info.value.units == ("dcid:Percent", "dcid:USDollar")


def test_pinned_unit_narrows_a_heterogeneous_frame() -> None:
    frame = _frame([_row(unit="dcid:USDollar"), _row(reporter="dcid:GBR", unit="dcid:Percent")])
    resolved = _resolved(frame, pinned_unit="dcid:USDollar")
    assert resolved.unit == "dcid:USDollar"


def test_pinned_unit_absent_from_the_frame_refuses() -> None:
    frame = _frame([_row(unit="dcid:USDollar")])
    with pytest.raises(UnitNotPresentError) as exc_info:
        _resolved(frame, pinned_unit="dcid:Percent")
    assert exc_info.value.units == ("dcid:USDollar",)


def test_facet_axis_wholly_unbound_resolves_to_nothing() -> None:
    frame = _frame([_row(), _row(reporter="dcid:GBR")])
    resolved = _resolved(frame)
    assert resolved.facet_selections == ()
    assert resolved.warnings == ()


def test_facet_axis_uniformly_declared_selects_silently() -> None:
    frame = _frame(
        [
            _row(provenance="dcid:OECD_DAC2A"),
            _row(reporter="dcid:GBR", provenance="dcid:OECD_DAC2A"),
        ]
    )
    resolved = _resolved(frame)
    assert resolved.facet_selections[0].axis == "provenance"
    assert resolved.facet_selections[0].ref == "dcid:OECD_DAC2A"
    assert resolved.warnings == ()


def test_facet_axis_partially_bound_selects_with_a_warning() -> None:
    """One row declares the axis, another leaves it unbound -- clearly
    selectable."""
    frame = _frame([_row(provenance="dcid:OECD_DAC2A"), _row(reporter="dcid:GBR")])
    resolved = _resolved(frame)
    assert resolved.facet_selections[0].ref == "dcid:OECD_DAC2A"
    assert len(resolved.warnings) == 1
    assert "provenance" in resolved.warnings[0]


def test_facet_axis_with_two_distinct_values_refuses() -> None:
    frame = _frame(
        [
            _row(provenance="dcid:OECD_DAC2A"),
            _row(reporter="dcid:GBR", provenance="dcid:OECD_DAC1"),
        ]
    )
    with pytest.raises(FacetHeterogeneousError) as exc_info:
        _resolved(frame)
    assert exc_info.value.axis == "provenance"
    assert exc_info.value.values == ("dcid:OECD_DAC1", "dcid:OECD_DAC2A")


# `record_exclusion`: the entity cap ------------------------------------------


def test_record_exclusion_caps_entities_at_twenty_and_sets_truncated() -> None:
    """More than 20 distinct values on one role column: `entities` lists
    only the first 20 (sorted), `entities_truncated` is set, and `row_count`
    still names the true total dropped -- the excess values are just not
    listed, not lost from the count."""
    rows = [_row(reporter=f"dcid:R{i:02d}") for i in range(25)]
    frame = _frame(rows)
    exclusions = record_exclusion(
        Exclusions(), frame=frame, reason="predicate", dropped=frame.table
    )
    assert len(exclusions.records) == 1
    record = exclusions.records[0]
    assert record.reason == "predicate"
    assert record.row_count == 25
    assert record.entities_truncated is True
    assert len(record.entities["reporter"]) == 20
    assert record.entities["reporter"] == tuple(sorted(f"dcid:R{i:02d}" for i in range(25)))[:20]


def test_record_exclusion_returns_the_same_accumulator_when_nothing_is_dropped() -> None:
    frame = _frame([_row()])
    exclusions = Exclusions()
    dropped = frame.table.slice(0, 0)
    assert (
        record_exclusion(exclusions, frame=frame, reason="predicate", dropped=dropped) is exclusions
    )


def test_smallest_distinct_matches_sorting_the_distinct_values() -> None:
    """Unsorted values with repeats: the twenty smallest distinct, in order,
    and the flag, as sorting every distinct value and cutting would give."""
    values = [f"dcid:R{i % 30:02d}" for i in range(90, 0, -1)]
    assert smallest_distinct(values) == (tuple(sorted(set(values)))[:20], True)
    assert smallest_distinct(values[:5]) == (tuple(sorted(set(values[:5]))), False)


# `narrow_and_resolve`: entity narrowing ---------------------------------------


def test_entity_narrowing_records_the_dropped_rows_as_an_exclusion() -> None:
    """An `entity` binding that narrows a frame to one reporter must account
    for every other reporter's own rows it drops, not just silently narrow
    past them."""
    frame = _frame(
        [
            _row(reporter="dcid:USA"),
            _row(reporter="dcid:GBR"),
            _row(reporter="dcid:DEU"),
        ]
    )
    table, _resolved, exclusions = narrow_and_resolve(
        frame, entity={"reporter": "dcid:USA"}, side="left"
    )
    assert table.num_rows == 1
    entity_records = [
        record for record in exclusions.records if record.reason == "entity_not_selected"
    ]
    assert len(entity_records) == 1
    assert entity_records[0].row_count == 2
    assert entity_records[0].side == "left"
    assert entity_records[0].entities["reporter"] == ("dcid:DEU", "dcid:GBR")


def test_entity_narrowing_that_drops_every_row_still_records_the_exclusion() -> None:
    """An `entity` binding matching nothing returns immediately with
    `unit=None` (see `narrow_and_resolve`'s own docstring), but the rows it
    narrowed away before hitting the empty slice must still be accounted
    for -- the early return must not discard them."""
    frame = _frame([_row(reporter="dcid:USA"), _row(reporter="dcid:GBR")])
    table, resolved, exclusions = narrow_and_resolve(frame, entity={"reporter": "dcid:NOPE"})
    assert table.num_rows == 0
    assert resolved.unit is None
    entity_records = [
        record for record in exclusions.records if record.reason == "entity_not_selected"
    ]
    assert len(entity_records) == 1
    assert entity_records[0].row_count == 2


def test_an_already_empty_entity_binding_over_an_empty_frame_records_no_exclusion() -> None:
    """`entity={}` over an already-empty frame narrows nothing -- there is
    no dropped row to attribute to the binding."""
    frame = Frame.build(role_keys=["reporter"], rows=[])
    _table, _resolved, exclusions = narrow_and_resolve(frame, entity={})
    assert exclusions.records == ()


@pytest.mark.parametrize(
    ("pinned_unit", "exclude_unitless"), [(None, True), ("dcid:USDollar", False), (None, False)]
)
def test_two_partly_bound_facet_axes_warn_in_axis_order(
    pinned_unit: str | None, exclude_unitless: bool
) -> None:
    rows = [
        {
            "stat_var": "dcid:X",
            "donor": f"dcid:Donor{index}",
            "time_period": "2020",
            "value": float(index),
            "unit": "dcid:USDollar" if index % 3 else None,
            "measurementMethod": "dcid:MethodA" if index % 2 else None,
            "provenance": "dcid:Provenance1" if index % 5 else None,
        }
        for index in range(30)
    ]
    frame = Frame.build(role_keys=["donor"], rows=rows)
    warnings = _resolved(frame, pinned_unit=pinned_unit, exclude_unitless=exclude_unitless).warnings
    assert len(warnings) == len(FACET_IDENTITY_COLUMNS) == 2
    for axis, warning in zip(FACET_IDENTITY_COLUMNS, warnings, strict=True):
        assert warning.startswith(f"facet {axis!r} was left unbound on some rows")
