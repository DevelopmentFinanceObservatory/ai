"""Tests for `submission.catalogue`: cards, contracts, their rules and the revision digest."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest
from jsonschema import Draft202012Validator
from planner import plan as plan_module
from planner.declaration import SHARED_FIELDS
from planner.plan import (
    EvidenceRequirement,
    JoinRequirement,
    Plan,
    inputs_of,
    unstated_result_scopes,
)
from planner.proof import KIND_OF_REQUIREMENT
from pydantic import TypeAdapter, ValidationError
from submission import catalogue
from submission.catalogue import (
    Rule,
    catalogue_revision,
    models_reached,
    render_cards,
    requirement_kinds,
    select_contracts,
)
from translator.mechanical import _BUILDERS

_REQUIREMENT = TypeAdapter(EvidenceRequirement)

_SIDES = {"left": {"input_id": "grants"}, "right": {"input_id": "loans"}}
_DIFFERENCE = {"minuend": {"input_id": "loans"}, "subtrahend": {"input_id": "grants"}}
_DIFFERENCE_OF_AID = {"minuend": {"input_id": "loans"}, "subtrahend": {"input_id": "aid"}}
_PARTS = {"numerator": {"input_id": "grants"}, "denominator": {"input_id": "loans"}}

# One requirement of each kind that the models accept.
_VALID: dict[str, dict[str, Any]] = {
    "dataset": {
        "kind": "dataset",
        "id": "aid",
        "measure": "ODA",
        "entities": [{"name": "Ghana"}],
        "period": {"start": 2015, "end": 2022},
    },
    "aggregate": {
        "kind": "aggregate",
        "id": "total",
        "input_id": "aid",
        "reduction": "sum",
        "group_by": ["period"],
    },
    "growth": {
        "kind": "growth",
        "id": "change",
        "input_id": "aid",
        "window_start": 2015,
        "window_end": 2022,
    },
    "comparison": {"kind": "comparison", "id": "gap", **_DIFFERENCE},
    "sum_of": {"kind": "sum_of", "id": "both", **_SIDES},
    "rank": {
        "kind": "rank",
        "id": "order",
        "input_id": "aid",
        "direction": "highest_first",
        "among": "population",
    },
    "share_of": {"kind": "share_of", "id": "share", **_PARTS},
    "quotient_of": {"kind": "quotient_of", "id": "ratio", **_PARTS},
    "join": {"kind": "join", "id": "joined", "left_id": "grants", "right_id": "loans"},
    "filter": {
        "kind": "filter",
        "id": "kept",
        "input_id": "aid",
        "predicates": [{"kind": "value_threshold", "op": "gt", "value": 0}],
    },
}


def _with(kind: str, **fields: Any) -> dict[str, Any]:
    return _VALID[kind] | fields


def _threshold(year: int, offset: int) -> dict[str, Any]:
    return {"kind": "period_threshold", "op": "gte", "year": year, "rolling": {"offset": offset}}


def _place(level: str) -> dict[str, Any]:
    return {"kind": "place_level", "level": level}


# Per registered rule, a requirement the rule admits and its minimal change
# the rule refuses.
_RULE_SAMPLES: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {
    "DatasetRequirement._source_modifiers_name_a_source": (
        _with("dataset", source="DAC", source_exclude=True),
        _with("dataset", source_exclude=True),
    ),
    "DatasetRequirement._entities_are_unique": (
        _with("dataset", entities=[{"name": "Ghana"}, {"name": "Kenya"}]),
        _with("dataset", entities=[{"name": "Ghana"}, {"name": "Ghana"}]),
    ),
    "DatasetRequirement._breakdown_is_its_own_role": (
        _with("dataset", breakdown={"dimension": "sector", "level": 0}, stat_vars="separate"),
        _with("dataset", breakdown={"dimension": "sector", "level": 0}, stat_vars="sum"),
    ),
    "DatasetRequirement._states_who_it_covers": (
        _with("dataset", entities=[], across=["donor"]),
        _with("dataset", entities=[]),
    ),
    "Period._ends_no_earlier_than_it_starts": (
        _with("dataset", period={"start": 2015, "end": 2015}),
        _with("dataset", period={"start": 2015, "end": 2014}),
    ),
    "Unit._base_year_goes_with_constant_prices": (
        _with("dataset", unit={"currency": "USD", "prices": "constant", "base_year": 2022}),
        _with("dataset", unit={"currency": "USD", "prices": "constant"}),
    ),
    "RollingWindow._names_a_bound": (
        _with("growth", rolling={"window_end": {"offset": 0}}),
        _with("growth", rolling={}),
    ),
    "GrowthRequirement._window_ends_after_it_starts": (
        _with("growth", window_start=2015, window_end=2016),
        _with("growth", window_start=2016, window_end=2016),
    ),
    "GrowthRequirement._rolling_agrees_with_the_years": (
        _with("growth", rolling={"window_start": {"offset": -7}, "window_end": {"offset": 0}}),
        _with("growth", rolling={"window_start": {"offset": -6}, "window_end": {"offset": 0}}),
    ),
    "ComparisonRequirement._sides_differ": (
        _with("comparison", minuend={"input_id": "grants", "entity": "Ghana"}),
        _with("comparison", minuend={"input_id": "grants"}),
    ),
    "SumOfRequirement._sides_read_different_requirements": (
        _with("sum_of", left={"input_id": "grants", "entity": "Ghana"}),
        _with(
            "sum_of",
            left={"input_id": "grants", "entity": "Ghana"},
            right={"input_id": "grants", "entity": "Kenya"},
        ),
    ),
    "RankRequirement._window_is_paired_and_ordered": (
        _with("rank", window_start=2015, window_end=2022),
        _with("rank", window_start=2015),
    ),
    "RankRequirement._rolling_agrees_with_the_years": (
        _with("rank", window_start=2015, window_end=2022, rolling={"window_end": {"offset": 0}}),
        _with("rank", rolling={"window_end": {"offset": 0}}),
    ),
    "ShareOfRequirement._sides_differ": (
        _with("share_of", denominator={"input_id": "grants", "entity": "Ghana"}),
        _with("share_of", denominator={"input_id": "grants"}),
    ),
    "ShareOfRequirement._denominator_scope_needs_an_axis": (
        _with("share_of", axis="recipient", denominator_scope="per_axis"),
        _with("share_of", axis="recipient"),
    ),
    "QuotientRequirement._sides_differ": (
        _with("quotient_of", denominator={"input_id": "grants", "entity": "Ghana"}),
        _with("quotient_of", denominator={"input_id": "grants"}),
    ),
    "QuotientRequirement._denominator_axis_needs_an_axis": (
        _with("quotient_of", axis="recipient", denominator_axis="country"),
        _with("quotient_of", denominator_axis="country"),
    ),
    "JoinRequirement._ids_are_distinct": (
        _VALID["join"],
        _with("join", right_id="grants"),
    ),
    "FilterRequirement._keeps_at_most_one_level": (
        _with("filter", predicates=[_place("Country")]),
        _with("filter", predicates=[_place("Country"), _place("City")]),
    ),
    "FilterRequirement._rolling_thresholds_share_a_newest_year": (
        _with("filter", predicates=[_threshold(2020, -2), _threshold(2021, -1)]),
        _with("filter", predicates=[_threshold(2020, -2), _threshold(2021, -2)]),
    ),
}


_GHANA_AID = _VALID["dataset"]
_LOANS = _with("dataset", id="loans", measure="Loans")
_DONORS = _with(
    "dataset",
    id="donors",
    entities=[{"name": "France", "role": "donor"}, {"name": "Germany", "role": "donor"}],
    period={"start": 2022, "end": 2022},
)
_EVERY_DONOR = _with(
    "dataset",
    id="every_donor",
    entities=[{"name": "Ghana", "role": "recipient"}],
    across=["donor"],
)
_PLAN_RULE_PREFIX = "planner.plan."


def _decision(slot: str, **fixes: Any) -> dict[str, Any]:
    return {
        "slot": slot,
        "description": "A choice.",
        "options": ["one", "other"],
        "chosen": "one",
        "resolved_by": "planner_default",
        "rationale": "A reason.",
        "fixes": fixes,
    }


def _plan(*requirements: dict[str, Any], **fields: Any) -> dict[str, Any]:
    return {"question": "A question.", "evidence_requirements": list(requirements), **fields}


def _growth_of(input_id: str, **fields: Any) -> dict[str, Any]:
    return _with("growth", input_id=input_id, **fields)


def _rank_of(input_id: str, **fields: Any) -> dict[str, Any]:
    return _with("rank", input_id=input_id, **fields)


# Per registered plan rule, a plan the rule admits and its minimal change the
# rule refuses.
_PLAN_RULE_SAMPLES: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {
    "planner.plan._check_requirement_graph": (
        _plan(_GHANA_AID, _LOANS, _VALID["comparison"] | _DIFFERENCE_OF_AID, _growth_of("aid")),
        _plan(_GHANA_AID, _LOANS, _VALID["comparison"] | _DIFFERENCE_OF_AID, _growth_of("gap")),
    ),
    "planner.plan._check_rolling_reads_open_periods": (
        _plan(
            _with("dataset", period_end_open=True),
            _growth_of("aid", rolling={"window_end": {"offset": 0}}),
        ),
        _plan(_GHANA_AID, _growth_of("aid", rolling={"window_end": {"offset": 0}})),
    ),
    "planner.plan._growth_shape": (
        _plan(_GHANA_AID, _growth_of("aid", window_start=2015)),
        _plan(_GHANA_AID, _growth_of("aid", window_start=2014)),
    ),
    "planner.plan._rank_shape": (
        _plan(
            _DONORS,
            _with(
                "filter",
                id="named",
                input_id="donors",
                predicates=[{"kind": "entity_in", "names": ["France", "Germany"], "role": "donor"}],
            ),
            _rank_of("named", among="named"),
        ),
        _plan(_DONORS, _rank_of("donors", among="named")),
    ),
    "planner.plan._check_population_claims_are_cited": (
        _plan(
            _DONORS,
            _rank_of("donors", decision_ids=["who"]),
            decisions=[_decision("who", among="population")],
        ),
        _plan(
            _DONORS,
            _rank_of("donors", decision_ids=["who"]),
            decisions=[_decision("who", direction="highest_first")],
        ),
    ),
    "planner.plan.unstated_result_scopes": (
        _plan(
            _EVERY_DONOR | {"decision_ids": ["scope"]},
            _growth_of("every_donor"),
            answer_mode="computed",
            decisions=[_decision("scope", across=["donor"])],
        ),
        _plan(
            _EVERY_DONOR | {"decision_ids": ["scope"]},
            _growth_of("every_donor"),
            answer_mode="computed",
            decisions=[_decision("scope", measure="ODA")],
        ),
    ),
}


def _plan_refused(enforced_by: str, sample: dict[str, Any]) -> bool:
    """Whether `sample` fails the plan rule `enforced_by` names, through the real plan checks."""
    try:
        plan = Plan.model_validate(sample)
    except ValidationError as refusal:
        assert [error["type"] for error in refusal.errors()] == ["value_error"]
        return True
    if enforced_by == "planner.plan.unstated_result_scopes":
        return bool(unstated_result_scopes(plan))
    return False


@pytest.fixture(autouse=True)
def _fresh_catalogue() -> Iterator[None]:
    """Rebuild the cached catalogue around each test, so a patched registration never leaks."""
    catalogue._catalogue.cache_clear()
    catalogue.catalogue_revision.cache_clear()
    yield
    catalogue._catalogue.cache_clear()
    catalogue.catalogue_revision.cache_clear()


def _every_kind() -> dict[str, Any]:
    """The selection of every kind's contract, with the shared `$defs`."""
    return select_contracts(card.kind for card in catalogue.cards()).payload


def _schema(kind: str) -> dict[str, Any]:
    """The contract schema of `kind`, with the definitions it refers to."""
    payload = _every_kind()
    return payload["contracts"][kind]["schema"] | {"$defs": payload["$defs"]}


def _registered_rules() -> dict[str, dict[str, str]]:
    return {
        rule["enforced_by"]: rule
        for contract in _every_kind()["contracts"].values()
        for rule in contract["rules"]
    }


def _refs(node: object) -> set[str]:
    if isinstance(node, list):
        return {name for item in node for name in _refs(item)}
    if not isinstance(node, dict):
        return set()
    names = {name for value in node.values() for name in _refs(value)}
    if isinstance(ref := node.get("$ref"), str):
        names.add(ref.removeprefix("#/$defs/"))
    return names


def _refs_with_siblings(node: object) -> list[dict[str, Any]]:
    if isinstance(node, list):
        return [found for item in node for found in _refs_with_siblings(item)]
    if not isinstance(node, dict):
        return []
    here = [node] if "$ref" in node and len(node) > 1 else []
    return here + [found for value in node.values() for found in _refs_with_siblings(value)]


def test_every_kind_has_a_card_a_contract_and_a_compiler_mapping() -> None:
    kinds = [catalogue.kind_of(model) for model in requirement_kinds()]

    assert kinds == list(_VALID)
    assert [card.kind for card in catalogue.cards()] == kinds
    assert list(_every_kind()["contracts"]) == kinds
    assert set(requirement_kinds()) == set(KIND_OF_REQUIREMENT) == set(_BUILDERS)
    assert [line.split(":")[0] for line in render_cards().splitlines()] == [
        f"- {kind}" for kind in kinds
    ]


def test_card_inputs_are_the_requirements_a_kind_reads() -> None:
    for card in catalogue.cards():
        sample = _VALID[card.kind]
        named = {
            sample[name]["input_id"] if isinstance(sample[name], dict) else sample[name]
            for name in card.inputs
        }
        assert named == set(inputs_of(_REQUIREMENT.validate_python(sample))), card.kind


def test_every_model_validator_a_kind_reaches_is_a_registered_model_rule() -> None:
    validators = {
        f"{model.__name__}.{name}"
        for kind in requirement_kinds()
        for model in models_reached(kind)
        for name in model.__pydantic_decorators__.model_validators
    }

    model_rules = {name for name in _registered_rules() if not name.startswith(_PLAN_RULE_PREFIX)}
    assert model_rules == validators


def test_every_plan_rule_names_a_plan_check() -> None:
    for enforced_by in _registered_rules():
        if enforced_by.startswith(_PLAN_RULE_PREFIX):
            assert callable(getattr(plan_module, enforced_by.removeprefix(_PLAN_RULE_PREFIX)))


@pytest.mark.parametrize("enforced_by", sorted(_PLAN_RULE_SAMPLES))
def test_a_registered_plan_rule_refuses_its_sample(enforced_by: str) -> None:
    admitted, refused = _PLAN_RULE_SAMPLES[enforced_by]
    assert enforced_by in _registered_rules()

    assert not _plan_refused(enforced_by, admitted)
    assert _plan_refused(enforced_by, refused)


@pytest.mark.parametrize("enforced_by", sorted(_RULE_SAMPLES))
def test_a_registered_rule_refuses_what_its_contract_admits(enforced_by: str) -> None:
    """The model accepts the admitted sample and refuses its twin, which the schema alone admits."""
    admitted, refused = _RULE_SAMPLES[enforced_by]
    assert enforced_by in _registered_rules()
    schema = _schema(refused["kind"])

    _REQUIREMENT.validate_python(admitted)
    assert Draft202012Validator(schema).is_valid(admitted)
    assert Draft202012Validator(schema).is_valid(refused)
    with pytest.raises(ValidationError) as refusal:
        _REQUIREMENT.validate_python(refused)
    assert [error["type"] for error in refusal.value.errors()] == ["value_error"]


def test_every_registered_rule_has_samples() -> None:
    assert set(_RULE_SAMPLES) | set(_PLAN_RULE_SAMPLES) == set(_registered_rules())


def test_every_valid_sample_meets_its_contract() -> None:
    for kind, sample in _VALID.items():
        assert Draft202012Validator(_schema(kind)).is_valid(sample), kind


def test_no_ref_has_siblings_and_every_ref_resolves() -> None:
    selection = select_contracts(list(_VALID))
    definitions = selection.payload["$defs"]

    assert _refs_with_siblings(selection.payload) == []
    for kind in _every_kind()["contracts"]:
        assert _refs_with_siblings(_schema(kind)) == [], kind
        Draft202012Validator.check_schema(_schema(kind))
    for kind in selection.kinds:
        schema = selection.payload["contracts"][kind]["schema"] | {"$defs": definitions}
        assert _refs(schema) <= set(definitions), kind
        assert Draft202012Validator(schema).is_valid(_VALID[kind]), kind


def test_a_selection_states_each_kind_and_definition_once() -> None:
    selection = select_contracts(["growth", "dataset", "pie_chart", "growth"])

    assert selection.kinds == ("dataset", "growth")
    assert selection.unknown == ("pie_chart",)
    assert list(selection.payload["contracts"]) == ["dataset", "growth"]
    assert set(selection.payload["$defs"]) == _refs(selection.payload["contracts"])
    assert selection.payload["shared_fields"].keys() == SHARED_FIELDS
    assert selection.digest == select_contracts(["dataset", "growth"]).digest


def test_the_revision_is_stable_and_moves_with_a_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    revision = catalogue_revision()
    join_digest = select_contracts(["join"]).digest
    growth_digest = select_contracts(["growth"]).digest
    assert catalogue_revision() == revision

    monkeypatch.setitem(
        catalogue._RULES, JoinRequirement, (Rule("Changed.", "JoinRequirement._ids_are_distinct"),)
    )
    catalogue._catalogue.cache_clear()
    catalogue.catalogue_revision.cache_clear()

    assert catalogue_revision() != revision
    assert select_contracts(["join"]).digest != join_digest
    assert select_contracts(["growth"]).digest == growth_digest
