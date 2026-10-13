"""Unit refusals that involve a converted unit (`UnitRequest`) or a composed
unit (`ComposedUnit`).

Every verb tool maps `UnitHeterogeneousError` and `UnitNotPresentError` to a
`unit_heterogeneous` refusal that lists such a unit by label and offers it as
a pin in the form the pin parameter accepts, its object. `compare` and
`share_of` map a cross-side unit mismatch or an unbound side to a refusal
that names the unit by label and carries its object in `fix`. `quotient_of`
divides any pair of units, and refuses a side whose unit is composed already."""

from __future__ import annotations

import json
import typing
from types import ModuleType
from typing import Any

import pytest
from analysis_mcp import (
    aggregate_tool,
    compare_tool,
    growth_tool,
    join_tool,
    quotient_of_tool,
    rank_tool,
    share_of_tool,
)
from analysis_mcp.engine.common import UnitHeterogeneousError, UnitNotPresentError
from analysis_mcp.engine.compare import run_compare
from analysis_mcp.engine.quotient_of import QuotientOfSideUnitUnsupportedError
from analysis_mcp.engine.share_of import run_share_of
from analysis_mcp.tool_errors import ToolRefusal, to_tool_result
from analysis_mcp.units import unit_json
from dfo_frame_grammar import Frame
from dfo_frame_grammar.grammar import ComposedUnit, StatVarPart, UnitCell, UnitRequest

USD = "dcid:USDollar"
USD_REQUEST = UnitRequest(currency="USD", prices="current")
EUR_REQUEST = UnitRequest(currency="EUR", prices="constant", base_year=2020)

USD_REQUEST_CANDIDATE = {
    "id": {"currency": "USD", "prices": "current"},
    "label": "converted to current prices, US dollars",
    "detail": {},
}
EUR_REQUEST_CANDIDATE = {
    "id": {"currency": "EUR", "prices": "constant", "base_year": 2020},
    "label": "converted to constant 2020 prices, euros",
    "detail": {},
}
USD_CANDIDATE = {"id": USD, "label": USD, "detail": {}}

USD_PER_PERSON = ComposedUnit(numerator=USD, denominator=StatVarPart(stat_var="dcid:Count_Person"))
EUR_PER_KM = ComposedUnit(numerator=EUR_REQUEST, denominator="dcid:SquareKilometer")
USD_PER_PERSON_LABEL = "dcid:USDollar per Count_Person"
EUR_PER_KM_LABEL = "constant 2020 prices, euros per dcid:SquareKilometer"
USD_PER_PERSON_CANDIDATE = {
    "id": {"numerator": USD, "denominator": {"stat_var": "dcid:Count_Person"}},
    "label": USD_PER_PERSON_LABEL,
    "detail": {},
}
EUR_PER_KM_CANDIDATE = {
    "id": {
        "numerator": {"currency": "EUR", "prices": "constant", "base_year": 2020},
        "denominator": "dcid:SquareKilometer",
    },
    "label": EUR_PER_KM_LABEL,
    "detail": {},
}

TOOLS: dict[ModuleType, str] = {
    aggregate_tool: "unit",
    growth_tool: "unit",
    rank_tool: "unit",
    compare_tool: "left_unit or right_unit",
    share_of_tool: "numerator_unit or denominator_unit",
    quotient_of_tool: "numerator_unit or denominator_unit",
    join_tool: "left_unit or right_unit",
}

# Each error, with the candidates and `fix["units"]` its refusal carries,
# and the message it renders for a verb whose unit-pin parameter is `{p}`.
SHAPES = {
    "heterogeneous_ref_and_request": (
        UnitHeterogeneousError([USD, USD_REQUEST]),
        [USD_CANDIDATE, USD_REQUEST_CANDIDATE],
        "The input frame carries more than one unit: ['dcid:USDollar', 'converted to "
        "current prices, US dollars']. Retry with `{p}` set to one of these values: "
        '"dcid:USDollar", {{"currency": "USD", "prices": "current"}}.',
    ),
    "heterogeneous_two_requests": (
        UnitHeterogeneousError([EUR_REQUEST, USD_REQUEST]),
        [EUR_REQUEST_CANDIDATE, USD_REQUEST_CANDIDATE],
        "The input frame carries more than one unit: ['converted to constant 2020 prices, "
        "euros', 'converted to current prices, US dollars']. Retry with `{p}` set to one "
        'of these values: {{"currency": "EUR", "prices": "constant", "base_year": 2020}}, '
        '{{"currency": "USD", "prices": "current"}}.',
    ),
    "not_present_ref_and_request": (
        UnitNotPresentError("dcid:Euro", [USD, USD_REQUEST]),
        [USD_CANDIDATE, USD_REQUEST_CANDIDATE],
        "{p}='dcid:Euro' is not one of the units this frame declares: ['dcid:USDollar', "
        "'converted to current prices, US dollars']. Retry with `{p}` set to one of these "
        'values: "dcid:USDollar", {{"currency": "USD", "prices": "current"}}.',
    ),
    "not_present_two_requests": (
        UnitNotPresentError("dcid:Euro", [EUR_REQUEST, USD_REQUEST]),
        [EUR_REQUEST_CANDIDATE, USD_REQUEST_CANDIDATE],
        "{p}='dcid:Euro' is not one of the units this frame declares: ['converted to "
        "constant 2020 prices, euros', 'converted to current prices, US dollars']. Retry "
        'with `{p}` set to one of these values: {{"currency": "EUR", "prices": "constant", '
        '"base_year": 2020}}, {{"currency": "USD", "prices": "current"}}.',
    ),
    "not_present_one_request": (
        UnitNotPresentError("dcid:Euro", [USD_REQUEST]),
        [USD_REQUEST_CANDIDATE],
        "{p}='dcid:Euro' is not one of the units this frame declares: ['converted to "
        "current prices, US dollars']. Retry with `{p}` set to one of these values: "
        '{{"currency": "USD", "prices": "current"}}.',
    ),
    "heterogeneous_two_composed": (
        UnitHeterogeneousError([EUR_PER_KM, USD_PER_PERSON]),
        [EUR_PER_KM_CANDIDATE, USD_PER_PERSON_CANDIDATE],
        f"The input frame carries more than one unit: ['{EUR_PER_KM_LABEL}', "
        f"'{USD_PER_PERSON_LABEL}']. Retry with `{{p}}` set to one of these values: "
        '{{"numerator": {{"currency": "EUR", "prices": "constant", "base_year": 2020}}, '
        '"denominator": "dcid:SquareKilometer"}}, '
        '{{"numerator": "dcid:USDollar", "denominator": {{"stat_var": "dcid:Count_Person"}}}}.',
    ),
    "not_present_composed_pin": (
        UnitNotPresentError(USD_PER_PERSON, [USD]),
        [USD_CANDIDATE],
        f"{{p}}='{USD_PER_PERSON_LABEL}' is not one of the units this frame declares: "
        "['dcid:USDollar']. Retry with one of these instead.",
    ),
}


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("tool", TOOLS, ids=lambda tool: tool.__name__.rsplit(".", 1)[-1])
def test_converted_unit_candidates_refuse_with_a_labelled_pinnable_candidate(
    tool: ModuleType, shape: str
) -> None:
    error, candidates, message = SHAPES[shape]
    parameter = TOOLS[tool]

    refusal = tool._refusal_for_engine_error(error)

    assert refusal.code.value == "unit_heterogeneous"
    assert refusal.message == message.format(p=parameter)
    assert refusal.fix["parameter"] == parameter
    assert refusal.fix["units"] == [
        unit if isinstance(unit, str) else unit.as_json() for unit in error.units
    ]
    assert refusal.fix["disambiguation"] == {"candidates": candidates}
    if isinstance(error, UnitNotPresentError):
        assert refusal.fix["pinned_unit"] == unit_json(error.pinned_unit)
    json.dumps(refusal.fix)


def test_a_currency_without_a_name_is_labelled_by_its_code() -> None:
    refusal = aggregate_tool._refusal_for_engine_error(
        UnitHeterogeneousError([UnitRequest(currency="JPY", prices="current"), USD_REQUEST])
    )
    labels = [candidate["label"] for candidate in refusal.fix["disambiguation"]["candidates"]]
    assert labels == ["converted to current prices, JPY", "converted to current prices, US dollars"]


def _frame(unit: UnitCell | None) -> Frame:
    cell = {} if unit is None else {"unit": unit_json(unit)}
    rows: list[dict[str, object]] = [
        {"stat_var": "dcid:X", "donor": "dcid:D1", "time_period": period, "value": 1.0, **cell}
        for period in ("2020", "2021")
    ]
    return Frame.build(role_keys=["donor"], rows=rows)


def _structured(refusal: ToolRefusal) -> dict[str, Any]:
    structured = to_tool_result(refusal).structured_content
    assert structured is not None
    return structured


def _refuse_compare(left: Frame, right: Frame) -> dict[str, Any]:
    with pytest.raises(typing.get_args(compare_tool._EngineError)) as caught:
        run_compare(left, right, stat_var="dcid:ops_" + "d" * 32, left_entity={}, right_entity={})
    return _structured(compare_tool._refusal_for_engine_error(caught.value))


def _refuse_share_of(left: Frame, right: Frame) -> dict[str, Any]:
    with pytest.raises(typing.get_args(share_of_tool._EngineError)) as caught:
        run_share_of(
            left, right, stat_var="dcid:ops_" + "c" * 32, numerator_entity={}, denominator_entity={}
        )
    return _structured(share_of_tool._refusal_for_engine_error(caught.value))


# Each verb's side labels and how it reaches a refusal from two frames.
CROSS_SIDE_VERBS = {
    "compare": (("left", "right"), _refuse_compare),
    "share_of": (("numerator", "denominator"), _refuse_share_of),
}

USD_REQUEST_LABEL = "converted to current prices, US dollars"
EUR_REQUEST_LABEL = "converted to constant 2020 prices, euros"
CONVERTED_RETRY = (
    "Call get_data for both sides with scope.facets binding the same unit, then retry "
    "with the new frame_refs."
)

COMPOSED_RETRY = (
    "A unit of the form 'X per Y' is the unit quotient_of gives one frame divided by "
    "another, and no unit pin or get_data binding changes it. Retry with two sides whose "
    "units match: each a quotient_of result over inputs in the same units as the other."
)

# Each pair of first- and second-side units, with the text each side's unit
# reads as in the message and the retry the message gives.
MISMATCHES = {
    "request_and_ref": (
        USD_REQUEST,
        USD,
        f"'{USD_REQUEST_LABEL}'",
        "'dcid:USDollar'",
        CONVERTED_RETRY,
    ),
    "ref_and_request": (
        USD,
        EUR_REQUEST,
        "'dcid:USDollar'",
        f"'{EUR_REQUEST_LABEL}'",
        CONVERTED_RETRY,
    ),
    "two_requests": (
        USD_REQUEST,
        EUR_REQUEST,
        f"'{USD_REQUEST_LABEL}'",
        f"'{EUR_REQUEST_LABEL}'",
        CONVERTED_RETRY,
    ),
    "two_composed": (
        USD_PER_PERSON,
        EUR_PER_KM,
        f"'{USD_PER_PERSON_LABEL}'",
        f"'{EUR_PER_KM_LABEL}'",
        COMPOSED_RETRY,
    ),
    "composed_and_ref": (
        USD_PER_PERSON,
        USD,
        f"'{USD_PER_PERSON_LABEL}'",
        "'dcid:USDollar'",
        COMPOSED_RETRY,
    ),
    "request_and_composed": (
        USD_REQUEST,
        EUR_PER_KM,
        f"'{USD_REQUEST_LABEL}'",
        f"'{EUR_PER_KM_LABEL}'",
        COMPOSED_RETRY,
    ),
}


@pytest.mark.parametrize("mismatch", MISMATCHES)
@pytest.mark.parametrize("verb", CROSS_SIDE_VERBS)
def test_a_cross_side_mismatch_with_a_unit_object_refuses_by_its_label(
    verb: str, mismatch: str
) -> None:
    (first, second), refuse = CROSS_SIDE_VERBS[verb]
    first_unit, second_unit, first_text, second_text, retry = MISMATCHES[mismatch]

    refusal = refuse(_frame(first_unit), _frame(second_unit))

    assert refusal["code"] == "cross_frame_unit_mismatch"
    assert refusal["message"] == (
        f"{first} resolved to unit {first_text}, {second} to {second_text} -- combining "
        "values across different units is silently wrong, not merely inconvenient. "
        f"{retry}"
    )
    assert refusal["fix"] == {first: unit_json(first_unit), second: unit_json(second_unit)}
    json.dumps(refusal)


def test_two_sides_with_equal_composed_units_compare() -> None:
    result = run_compare(
        _frame(USD_PER_PERSON),
        _frame(USD_PER_PERSON),
        stat_var="dcid:ops_" + "d" * 32,
        left_entity={},
        right_entity={},
    )

    assert result.left_resolved_unit == USD_PER_PERSON
    assert result.right_resolved_unit == USD_PER_PERSON
    assert len(result.frame.rows) == 2


@pytest.mark.parametrize("verb", CROSS_SIDE_VERBS)
def test_an_unbound_side_against_a_converted_unit_refuses_by_its_label(verb: str) -> None:
    (first, second), refuse = CROSS_SIDE_VERBS[verb]

    refusal = refuse(_frame(None), _frame(EUR_REQUEST))

    assert refusal["code"] == "cross_frame_unit_unbound"
    assert refusal["message"] == (
        f"{first}'s own rows leave `unit` unbound entirely; {second} resolved to unit "
        f"'{EUR_REQUEST_LABEL}'. Retry with a {first} whose own rows actually declare a unit."
    )
    assert refusal["fix"] == {
        "unbound_side": first,
        "bound_side": second,
        "bound_unit": EUR_REQUEST.as_json(),
    }
    json.dumps(refusal)


@pytest.mark.parametrize("unit", [USD_PER_PERSON, EUR_PER_KM], ids=["ref", "converted"])
def test_a_composed_side_of_a_quotient_refuses_with_the_unit_object(unit: ComposedUnit) -> None:
    error = QuotientOfSideUnitUnsupportedError(side="denominator", reason="composed", unit=unit)

    refusal = _structured(quotient_of_tool._refusal_for_engine_error(error))

    assert refusal["code"] == "quotient_of_side_unit_unsupported"
    assert refusal["fix"]["side"] == "denominator"
    assert refusal["fix"]["reason"] == "composed"
    assert refusal["fix"]["unit"] == unit_json(unit)
    json.dumps(refusal)
