"""The requirement catalogue: a card and a contract per requirement kind.

A card is what a planner reads to choose a kind: its purpose, the
requirements it reads, the fields it needs, what it gives, and how it differs
from its neighbours. A contract is what a planner fills a chosen kind from:
the kind's compact JSON Schema (`planner.declaration.compact_schema`) and the
conditional rules no JSON Schema states.

Both come from the requirement models in `planner.plan`. What is written
here by hand is each kind's output and distinction line, each rule's text,
and the meaning of the fields every kind shares, which `compact_schema`
leaves undescribed. A model rule is registered against the model class whose
validator enforces it and names that validator, so a contract carries the
rules of every model its kind reaches. A plan rule, which `planner.plan`
checks across requirements, is registered against each kind it constrains
and names the function that enforces it. `tests/submission/test_catalogue.py`
proves each rule refuses what its text says it refuses.

`catalogue_revision()` digests every card, contract and shared field
meaning, so a submission naming a revision states which catalogue it was
written against.
"""

from __future__ import annotations

import copy
import hashlib
import inspect
import typing
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from functools import cache
from typing import Any

import rfc8785
from planner.declaration import SHARED_FIELDS, compact_schema
from planner.plan import (
    AggregateRequirement,
    ComparisonRequirement,
    DatasetRequirement,
    EvidenceRequirement,
    FilterRequirement,
    GrowthRequirement,
    JoinRequirement,
    QuotientRequirement,
    RankRequirement,
    RequirementSide,
    ShareOfRequirement,
    SumOfRequirement,
    Unit,
)
from planner.records import Identifier, Period, RollingWindow
from pydantic import BaseModel
from pydantic.fields import FieldInfo


@dataclass(frozen=True)
class Rule:
    """A conditional rule a contract states, and the model validator that enforces it.

    `enforced_by` reads `Model._validator`, the class and the validator's own name.
    """

    text: str
    enforced_by: str


@dataclass(frozen=True)
class Card:
    """What a planner reads to choose a requirement kind."""

    kind: str
    purpose: str
    inputs: tuple[str, ...]
    required: tuple[str, ...]
    output: str
    distinction: str


@dataclass(frozen=True)
class Contract:
    """What a planner fills one requirement kind from.

    `schema` is self-contained: its `$defs` hold every definition it refers to.
    """

    kind: str
    schema: dict[str, Any]
    rules: tuple[Rule, ...]


@dataclass(frozen=True)
class ContractSelection:
    """The contracts for a selection of kinds, each stated once.

    `payload` maps `contracts` to each known kind's schema, without `$defs`,
    and its rules, states the definitions every schema refers to once under
    `$defs`, and the meaning of each field every kind shares under
    `shared_fields`. `kinds` lists the known kinds in catalogue order and
    `unknown` the names no kind carries, in the order first named.
    """

    kinds: tuple[str, ...]
    unknown: tuple[str, ...]
    payload: dict[str, Any]
    digest: str


# What each kind gives and how it differs from the kinds a planner could
# mistake it for, which its model does not state.
_OUTPUTS: Mapping[type[BaseModel], tuple[str, str]] = {
    DatasetRequirement: (
        "One series per entity, and per code under a breakdown, for each year of `period`.",
        "The only kind that retrieves data; every other kind reads a requirement's result.",
    ),
    AggregateRequirement: (
        "One value per group of `group_by`; an `across` party not grouped by is summed away.",
        "Reduces rows within one result; sum_of adds two different series.",
    ),
    GrowthRequirement: (
        "One value per series over the window: a ratio for a relative change, a difference "
        "in the input's unit for an absolute one.",
        "Compares one series' two endpoint years; comparison subtracts two series at every period.",
    ),
    ComparisonRequirement: (
        "One difference per period both sides report. Terminal: nothing reads it.",
        "Subtracts two different results; growth compares one result's two years, and join "
        "sets two results side by side without computing.",
    ),
    SumOfRequirement: (
        "One series: the two sides added at each period both report.",
        "Adds two different single-series results; aggregate totals the entities of one result.",
    ),
    RankRequirement: (
        "Each entity's rank position on `axis`, or its measured value under `limit` with "
        "`per_period`, once per entity or once per entity and period.",
        "Orders entities and can keep the top or bottom places; filter keeps rows by a condition.",
    ),
    ShareOfRequirement: (
        "The numerator's share of the denominator at each period, per entity of `axis` when "
        "stated.",
        "A part of a whole in one unit; quotient_of divides one measure by another.",
    ),
    QuotientRequirement: (
        "The numerator divided by the denominator at each period, in the unit the two "
        "compose, per entity of `axis` when stated.",
        "Divides different measures, as a per-capita figure does; share_of takes a part of a "
        "whole.",
    ),
    JoinRequirement: (
        "One frame with both results' values over the periods and entities they share. "
        "Terminal: nothing reads it.",
        "Sets two results side by side without computing; comparison subtracts them.",
    ),
    FilterRequirement: (
        "The input's rows that satisfy every predicate, with the input's own columns.",
        "Keeps rows by a value, year, entity or place-level condition; rank keeps places by order.",
    ),
}

_RULES: Mapping[type[BaseModel], tuple[Rule, ...]] = {
    DatasetRequirement: (
        Rule(
            "`source_ref` and `source_exclude` are stated only with `source`.",
            "DatasetRequirement._source_modifiers_name_a_source",
        ),
        Rule(
            "Each entity appears once for its `name` and `role`.",
            "DatasetRequirement._entities_are_unique",
        ),
        Rule(
            "A `breakdown` is refused with `stat_vars` 'sum', and no entity `role` or `across` "
            "may equal its dimension.",
            "DatasetRequirement._breakdown_is_its_own_role",
        ),
        Rule(
            "Name at least one entity or state `across`. With `across`, no entity sits in the "
            "party it covers, and with a place party every entity states a `role`.",
            "DatasetRequirement._states_who_it_covers",
        ),
    ),
    Period: (Rule("`end` is no earlier than `start`.", "Period._ends_no_earlier_than_it_starts"),),
    Unit: (
        Rule(
            "`base_year` is stated exactly when `prices` is 'constant'.",
            "Unit._base_year_goes_with_constant_prices",
        ),
    ),
    RollingWindow: (
        Rule(
            "`rolling` names `window_start`, `window_end` or both.",
            "RollingWindow._names_a_bound",
        ),
    ),
    GrowthRequirement: (
        Rule(
            "`window_end` is after `window_start`.",
            "GrowthRequirement._window_ends_after_it_starts",
        ),
        Rule(
            "With both `rolling` bounds relative, their offsets span the same years as "
            "`window_start` to `window_end`.",
            "GrowthRequirement._rolling_agrees_with_the_years",
        ),
    ),
    ComparisonRequirement: (
        Rule(
            "`minuend` and `subtrahend` differ in `input_id` or `entity`.",
            "ComparisonRequirement._sides_differ",
        ),
    ),
    SumOfRequirement: (
        Rule(
            "`left` and `right` read different `input_id`s.",
            "SumOfRequirement._sides_read_different_requirements",
        ),
    ),
    RankRequirement: (
        Rule(
            "`window_start` and `window_end` are stated together, and `window_end` is after "
            "`window_start`.",
            "RankRequirement._window_is_paired_and_ordered",
        ),
        Rule(
            "`rolling` needs `window_start` and `window_end`; with both its bounds relative, "
            "their offsets span the same years as the window.",
            "RankRequirement._rolling_agrees_with_the_years",
        ),
    ),
    ShareOfRequirement: (
        Rule(
            "`numerator` and `denominator` differ in `input_id` or `entity`.",
            "ShareOfRequirement._sides_differ",
        ),
        Rule(
            "`denominator_scope` is stated exactly when `axis` is.",
            "ShareOfRequirement._denominator_scope_needs_an_axis",
        ),
    ),
    QuotientRequirement: (
        Rule(
            "`numerator` and `denominator` differ in `input_id` or `entity`.",
            "QuotientRequirement._sides_differ",
        ),
        Rule(
            "`denominator_axis` is stated only with `axis`.",
            "QuotientRequirement._denominator_axis_needs_an_axis",
        ),
    ),
    JoinRequirement: (
        Rule("`left_id` and `right_id` differ.", "JoinRequirement._ids_are_distinct"),
    ),
    FilterRequirement: (
        Rule(
            "At most one `place_level` predicate.",
            "FilterRequirement._keeps_at_most_one_level",
        ),
        Rule(
            "Rolling `period_threshold` predicates stand for one newest year: `year` minus "
            "`rolling.offset` is the same for each.",
            "FilterRequirement._rolling_thresholds_share_a_newest_year",
        ),
    ),
}

# The constraints an `Identifier` field carries, which tell it from any other string.
_IDENTIFIER_METADATA = typing.get_args(Identifier)[1].metadata

_COMBINED = Rule(
    "Nothing reads a comparison or a join: a calculation over either is unsupported.",
    "planner.plan._check_requirement_graph",
)
_OPEN_PERIOD = Rule(
    "A `rolling` window or threshold reads only dataset requirements stating `period_end_open`.",
    "planner.plan._check_rolling_reads_open_periods",
)
_POPULATION_CLAIM = Rule(
    "A rank with `among` 'population', or a share_of with `denominator_scope` 'pooled', "
    "cites a decision whose `fixes` states that same value.",
    "planner.plan._check_population_claims_are_cited",
)

# Rules `planner.plan` checks across a plan's requirements, under each kind
# they constrain.
_PLAN_RULES: Mapping[type[BaseModel], tuple[Rule, ...]] = {
    DatasetRequirement: (
        Rule(
            "A result that answers and keeps one figure per member of an `across` party "
            "states that scope. Either this requirement cites a decision whose `fixes` "
            "states that `across`, or a step on the way to the result names the party: a "
            "rank ordering it, a share_of or quotient_of whose `axis` names it, or a filter "
            "whose `entity_in` states its role, or states no role over an input carrying no "
            "other party. Otherwise aggregate the party away first.",
            "planner.plan.unstated_result_scopes",
        ),
    ),
    GrowthRequirement: (
        Rule(
            "The window lies within the period of the result it reads.",
            "planner.plan._growth_shape",
        ),
        _OPEN_PERIOD,
    ),
    ComparisonRequirement: (_COMBINED,),
    RankRequirement: (
        Rule(
            "`among` 'named' reads a filter whose `entity_in` narrows the ranked role.",
            "planner.plan._rank_shape",
        ),
        _POPULATION_CLAIM,
        _OPEN_PERIOD,
    ),
    ShareOfRequirement: (_POPULATION_CLAIM,),
    JoinRequirement: (_COMBINED,),
    FilterRequirement: (_OPEN_PERIOD,),
}

# The meaning of each field every kind carries, which `compact_schema` strips
# from each kind's schema.
_SHARED_FIELDS: Mapping[str, str] = {
    "kind": "The requirement's kind, as its card names it.",
    "id": "This requirement's name, unique within the plan, which other requirements and "
    "answers cite.",
    "decision_ids": "The slots of the decisions that fixed this requirement's fields; empty "
    "when the question states them outright.",
    "input_id": "The id of the requirement whose result this one reads.",
}
if set(_SHARED_FIELDS) != SHARED_FIELDS:
    raise RuntimeError("describe every field `planner.declaration.SHARED_FIELDS` names")


def requirement_kinds() -> tuple[type[BaseModel], ...]:
    """Every member of the `EvidenceRequirement` union, in the union's order."""
    union, _discriminator = typing.get_args(EvidenceRequirement)
    return typing.get_args(union)


def kind_of(model: type[BaseModel]) -> str:
    """The `kind` tag `model` carries."""
    return model.model_fields["kind"].default


def models_reached(model: type[BaseModel]) -> tuple[type[BaseModel], ...]:
    """`model` and every model its fields reach, depth first, each once."""
    reached: list[type[BaseModel]] = []

    def visit(annotation: Any) -> None:
        if isinstance(annotation, type) and issubclass(annotation, BaseModel):
            if annotation not in reached:
                reached.append(annotation)
                for field in annotation.model_fields.values():
                    visit(field.annotation)
            return
        for argument in typing.get_args(annotation):
            visit(argument)

    visit(model)
    return tuple(reached)


def _reads_a_requirement(name: str, field: FieldInfo) -> bool:
    """Whether `field` names a requirement whose result the kind reads."""
    if field.annotation is RequirementSide:
        return True
    return name != "id" and field.annotation is str and field.metadata == _IDENTIFIER_METADATA


def _purpose(model: type[BaseModel]) -> str:
    """The first paragraph of `model`'s docstring, on one line."""
    first = inspect.cleandoc(model.__doc__ or "").split("\n\n")[0]
    return " ".join(first.split())


def _card(model: type[BaseModel]) -> Card:
    output, distinction = _OUTPUTS[model]
    inputs = tuple(
        name for name, field in model.model_fields.items() if _reads_a_requirement(name, field)
    )
    required = tuple(
        name
        for name, field in model.model_fields.items()
        if field.is_required() and name != "id" and name not in inputs
    )
    return Card(
        kind=kind_of(model),
        purpose=_purpose(model),
        inputs=inputs,
        required=required,
        output=output,
        distinction=distinction,
    )


def _contract(model: type[BaseModel]) -> Contract:
    return Contract(
        kind=kind_of(model),
        schema=compact_schema(model.model_json_schema()),
        rules=(
            *(rule for reached in models_reached(model) for rule in _RULES.get(reached, ())),
            *_PLAN_RULES.get(model, ()),
        ),
    )


@cache
def _catalogue() -> tuple[dict[str, Card], dict[str, Contract]]:
    models = requirement_kinds()
    return (
        {kind_of(model): _card(model) for model in models},
        {kind_of(model): _contract(model) for model in models},
    )


def cards() -> tuple[Card, ...]:
    """Every requirement kind's card, in the union's order."""
    return tuple(_catalogue()[0].values())


def render_cards() -> str:
    """Every card as compact text for an instruction, one line per kind."""
    lines = []
    for card in cards():
        reads = f" Reads {', '.join(card.inputs)}." if card.inputs else ""
        needs = f" Needs {', '.join(card.required)}." if card.required else ""
        lines.append(
            f"- {card.kind}: {card.purpose}{reads}{needs} Gives: {card.output} {card.distinction}"
        )
    return "\n".join(lines)


def _digest(content: dict[str, Any]) -> str:
    return hashlib.sha256(rfc8785.dumps(content)).hexdigest()


def _rule_payload(rule: Rule) -> dict[str, str]:
    return {"rule": rule.text, "enforced_by": rule.enforced_by}


@cache
def catalogue_revision() -> str:
    """The digest of every card, contract and shared field meaning.

    A submission names it to state the catalogue it was written against.

    Canonical JSON (RFC 8785) under SHA-256, as `planner.plan.plan_revision`.
    """
    by_kind, by_contract = _catalogue()
    return _digest(
        {
            "cards": [asdict(card) for card in by_kind.values()],
            "contracts": {
                kind: {
                    "schema": contract.schema,
                    "rules": [_rule_payload(rule) for rule in contract.rules],
                }
                for kind, contract in by_contract.items()
            },
            "shared_fields": dict(_SHARED_FIELDS),
        }
    )


def select_contracts(kinds: Iterable[str]) -> ContractSelection:
    """The contracts for `kinds`, each kind once, with their definitions stated once.

    Raises:
        ValueError: two kinds' schemas define one name differently, which
            one shared `$defs` would misstate.
    """
    by_contract = _catalogue()[1]
    named = list(dict.fromkeys(kinds))
    known = tuple(kind for kind in by_contract if kind in named)
    unknown = tuple(kind for kind in named if kind not in by_contract)
    definitions: dict[str, Any] = {}
    selected: dict[str, Any] = {}
    for kind in known:
        contract = by_contract[kind]
        for name, definition in contract.schema.get("$defs", {}).items():
            if definitions.setdefault(name, definition) != definition:
                raise ValueError(f"two selected kinds define `{name}` differently")
        selected[kind] = {
            "schema": {key: value for key, value in contract.schema.items() if key != "$defs"},
            "rules": [_rule_payload(rule) for rule in contract.rules],
        }
    # A copy, so a caller editing its payload leaves the cached contracts alone.
    payload = copy.deepcopy(
        {
            "$defs": dict(sorted(definitions.items())),
            "shared_fields": dict(_SHARED_FIELDS),
            "contracts": selected,
        }
    )
    return ContractSelection(kinds=known, unknown=unknown, payload=payload, digest=_digest(payload))
