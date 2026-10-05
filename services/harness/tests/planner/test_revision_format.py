"""The revision format: what `plan_revision` and `requirement_revision` hash.

`fixtures/revision_base_plans.json` holds two plans as the code at commit
2563638 serialised them, each beside the revisions that code computed: the
scripted plan with a decision that fixes its reduction (`aggregate`), and a
population rank compiled against the tool schemas the servers declared at
that commit (`rank`). It also holds the scripted plan with a decline
(`declined`), as the code at d8fd80b serialised it before `Decline` gained
`ask_ref`. `fixtures/revision_base_fields.json` lists every field
of every model a plan holds at that commit, the base revision format. A field
outside that list carries `common.hashed_fields.HASHED_WHEN_SET`, which keeps it
out of the hash while it holds its default.
"""

from __future__ import annotations

import dataclasses
import json
import typing
from pathlib import Path
from typing import Any

import pytest
from common.hashed_fields import HASHED_WHEN_SET, revision_content
from planner.plan import DatasetRequirement, Plan, RankRequirement, plan_revision
from pydantic import BaseModel
from translator.discovery import requirement_revision

_FIXTURES = Path(__file__).parents[1] / "fixtures"
_RECORDED: dict[str, dict[str, Any]] = json.loads(
    (_FIXTURES / "revision_base_plans.json").read_text()
)
_BASE_FIELDS: dict[str, list[str]] = json.loads(
    (_FIXTURES / "revision_base_fields.json").read_text()
)


def _is_hashed_when_set(model: type[BaseModel], name: str) -> bool:
    return any(item is HASHED_WHEN_SET for item in model.model_fields[name].metadata)


def _collect(annotation: object, found: dict[str, type[BaseModel]]) -> None:
    """Add to `found` every model `annotation` can hold, at any depth."""
    if isinstance(annotation, typing.TypeAliasType):
        annotation = annotation.__value__
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        key = f"{annotation.__module__}.{annotation.__qualname__}"
        if key in found:
            return
        found[key] = annotation
        for field in annotation.model_fields.values():
            _collect(field.annotation, found)
        return
    # `_drop_unset` walks models, mappings and sequences only.
    assert not (isinstance(annotation, type) and dataclasses.is_dataclass(annotation)), annotation
    for argument in typing.get_args(annotation):
        _collect(argument, found)


def _hashed_models() -> dict[str, type[BaseModel]]:
    found: dict[str, type[BaseModel]] = {}
    _collect(Plan, found)
    return found


@pytest.mark.parametrize("name", sorted(_RECORDED))
def test_a_plan_serialised_at_the_base_format_keeps_its_revision(name: str) -> None:
    recorded = _RECORDED[name]

    plan = Plan.model_validate(recorded["plan"])

    assert plan_revision(plan) == recorded["revision"]
    assert plan_revision(Plan.model_validate_json(plan.model_dump_json())) == recorded["revision"]


@pytest.mark.parametrize("name", sorted(_RECORDED))
def test_a_requirement_serialised_at_the_base_format_keeps_its_revision(name: str) -> None:
    recorded = _RECORDED[name]
    plan = Plan.model_validate(recorded["plan"])

    revisions = {
        requirement.id: requirement_revision(requirement)
        for requirement in plan.evidence_requirements
        if isinstance(requirement, DatasetRequirement)
    }

    assert revisions == recorded["requirement_revisions"]


def test_the_recorded_plans_hold_every_field_added_since_the_base_format() -> None:
    # Each added field has to sit, at its default, in a recorded plan: a
    # recorded plan holding no instance of a model proves nothing about it.
    held: set[tuple[str, str]] = set()

    def walk(value: object) -> None:
        if isinstance(value, BaseModel):
            key = f"{type(value).__module__}.{type(value).__qualname__}"
            for name in type(value).model_fields:
                held.add((key, name))
                walk(getattr(value, name))
        elif isinstance(value, dict):
            for item in value.values():
                walk(item)
        elif isinstance(value, (tuple, list)):
            for item in value:
                walk(item)

    for recorded in _RECORDED.values():
        walk(Plan.model_validate(recorded["plan"]))
    added = {
        (key, name)
        for key, model in _hashed_models().items()
        if key in _BASE_FIELDS
        for name in model.model_fields
        if _is_hashed_when_set(model, name)
    }

    assert added
    assert added <= held


def test_a_field_added_since_the_base_format_enters_the_revision_once_set() -> None:
    plan = Plan.model_validate(_RECORDED["rank"]["plan"])
    content = plan.model_dump(mode="json")
    ranked = next(r for r in content["evidence_requirements"] if r["kind"] == "rank")
    operation = next(o for o in content["translation"]["operations"] if o["kind"] == "rank")
    ranked["limit"] = operation["limit"] = 3

    limited = Plan.model_validate(content)

    assert plan_revision(limited) != plan_revision(plan)
    (requirement,) = (r for r in limited.evidence_requirements if isinstance(r, RankRequirement))
    assert revision_content(requirement)["limit"] == 3
    assert "limit" not in revision_content(requirement.model_copy(update={"limit": None}))


def test_every_hashed_field_is_in_the_base_format_or_declared_hashed_when_set() -> None:
    """A field a hashed model gains moves every saved plan's revision unless it is declared.

    A field added to a model here carries `HASHED_WHEN_SET` in its
    annotation. A model that is new as a whole lists its fields in
    `fixtures/revision_base_fields.json`, since no saved plan holds one.
    """
    models = _hashed_models()
    undeclared = {
        key: sorted(name for name in model.model_fields if not _is_hashed_when_set(model, name))
        for key, model in models.items()
    }

    assert undeclared == {key: sorted(names) for key, names in _BASE_FIELDS.items()}


def test_every_hashed_when_set_field_has_a_default() -> None:
    required = [
        f"{key}.{name}"
        for key, model in _hashed_models().items()
        for name, field in model.model_fields.items()
        if _is_hashed_when_set(model, name) and field.is_required()
    ]

    assert required == []
