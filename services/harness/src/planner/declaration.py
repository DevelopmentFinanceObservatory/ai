"""The declaration `propose_plan` reaches the model with.

ADK builds a function tool's declaration from its signature: pydantic's
`model_json_schema()` of the arguments, with the whole docstring as the
tool's description and no description on any argument. `PlanTool` keeps that
schema, gives each argument its own description, and rewrites the result with
`compact_schema`, so the patch rules and limits a model writes a call from sit
on the argument they govern.

`compact_schema` removes what tells a model nothing its key does not, and
applies to every definition the plan's models reach, so a field added later
gets the same treatment:

- every `title`, which repeats the property or class name it sits under;
- the `null` branch of an optional field's `anyOf`, leaving the field's own
  schema in its place: an optional field is one `required` does not list;
- a `null` or `[]` default, which the same absence from `required` says, and
  a default equal to the `const` beside it;
- the description of a field named in `SHARED_FIELDS`, which every
  requirement kind carries with the same meaning. The `evidence_requirements`
  argument states that meaning once;
- the description of an enum named in `FIELD_DESCRIBED_ENUMS`, whose values
  the field using it describes.

It then states two schemas once under `$defs` and refers to them: an
`Identifier`, which names every requirement and decision, and the
`decision_ids` list every requirement kind carries.

Gemini reads nothing beside a `$ref`, so no `$ref` in the result has sibling
keys. Where a field's schema is a reference with a description or default of
its own, the field holds a copy of the definition instead, with the field's
keys on top: its `description` replaces the definition's own and its
`default` stays. A definition nothing refers to any more is dropped.

The validator is untouched: `propose_plan` still validates what the model
sends against the models themselves, so an explicit `null` is accepted as
before.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, override

from google.adk.features import FeatureName, is_feature_enabled
from google.adk.tools.function_tool import FunctionTool
from google.genai import types
from pydantic import TypeAdapter

from planner.records import Identifier

# Fields every requirement kind declares with one meaning, described once on
# the `evidence_requirements` argument rather than on each kind.
SHARED_FIELDS = frozenset({"kind", "id", "decision_ids", "input_id"})

# Enums whose every value the field using them describes, so the enum's own
# description would only repeat it: `stat_vars`, `among` and
# `denominator_scope`.
FIELD_DESCRIBED_ENUMS = frozenset({"StatVarReading", "RankAmong", "DenominatorScope"})

# The schemas stated once under `$defs`, by the name they are stated under.
_IDENTIFIER_DEF = "Identifier"
_DECISION_IDS_DEF = "DecisionIds"
_IDENTIFIER_SCHEMA: dict[str, Any] = TypeAdapter(Identifier).json_schema()

# Why a `PlanTool` refuses to build without a JSON schema to describe.
_NO_JSON_SCHEMA = (
    "{name} carries its argument descriptions on a JSON schema, which ADK builds only "
    "with its JSON_SCHEMA_FOR_FUNC_DECL feature enabled; enable it"
)

# Keys whose value maps names to schemas, so a name there is never a keyword.
_NAMED_SCHEMAS = frozenset({"properties", "$defs"})

# What every `$ref` in a pydantic schema starts with, before the definition's name.
_DEFS_PREFIX = "#/$defs/"


def _without_null_branch(schema: dict[str, Any]) -> dict[str, Any]:
    """`schema` with an `anyOf: [X, {"type": "null"}]` replaced by `X`'s own keys."""
    branches = schema.get("anyOf")
    if not isinstance(branches, list) or len(branches) != 2:
        return schema
    others = [branch for branch in branches if branch != {"type": "null"}]
    if len(others) != 1:
        return schema
    merged = {key: value for key, value in schema.items() if key != "anyOf"}
    return {**others[0], **merged}


def _compact(node: Any) -> Any:
    if isinstance(node, list):
        return [_compact(item) for item in node]
    if not isinstance(node, dict):
        return node
    compacted: dict[str, Any] = {}
    for key, value in node.items():
        if key in _NAMED_SCHEMAS:
            compacted[key] = {
                name: _compact(_shared_undescribed(name, schema) if key == "properties" else schema)
                for name, schema in value.items()
            }
        elif key == "title" or (
            key == "default"
            and (value is None or value == [] or ("const" in node and value == node["const"]))
        ):
            continue
        else:
            compacted[key] = _compact(value)
    return _without_null_branch(compacted)


def _shared_undescribed(name: str, schema: Any) -> Any:
    if name in SHARED_FIELDS and isinstance(schema, dict):
        return {key: value for key, value in schema.items() if key != "description"}
    return schema


def _ref(name: str) -> dict[str, str]:
    return {"$ref": f"{_DEFS_PREFIX}{name}"}


def _refer_identifiers(node: Any, found: list[Any]) -> Any:
    """`node` with every `Identifier` schema replaced by a reference, its other keys kept.

    Each schema replaced is appended to `found`.
    """
    if isinstance(node, list):
        return [_refer_identifiers(item, found) for item in node]
    if not isinstance(node, dict):
        return node
    referred = {key: _refer_identifiers(value, found) for key, value in node.items()}
    if _IDENTIFIER_SCHEMA.items() <= referred.items():
        found.append(_IDENTIFIER_SCHEMA)
        kept = {key: value for key, value in referred.items() if key not in _IDENTIFIER_SCHEMA}
        return _ref(_IDENTIFIER_DEF) | kept
    return referred


def _refer_decision_ids(node: Any, found: list[Any]) -> Any:
    """`node` with every `decision_ids` property replaced by a reference, collected in `found`."""
    if isinstance(node, list):
        return [_refer_decision_ids(item, found) for item in node]
    if not isinstance(node, dict):
        return node
    referred = {key: _refer_decision_ids(value, found) for key, value in node.items()}
    properties = referred.get("properties")
    if isinstance(properties, dict) and "decision_ids" in properties:
        found.append(properties["decision_ids"])
        referred["properties"] = properties | {"decision_ids": _ref(_DECISION_IDS_DEF)}
    return referred


def _factored(schema: dict[str, Any]) -> dict[str, Any]:
    """`schema` stating `Identifier` and `decision_ids` once under `$defs`.

    Raises when two requirement kinds declare `decision_ids` differently,
    since one definition would then misstate one of them.
    """
    identifiers: list[Any] = []
    decision_ids: list[Any] = []
    factored = _refer_decision_ids(_refer_identifiers(schema, identifiers), decision_ids)
    defs = {
        name: {key: value for key, value in enum.items() if key != "description"}
        if name in FIELD_DESCRIBED_ENUMS
        else enum
        for name, enum in factored.get("$defs", {}).items()
    }
    if identifiers:
        defs[_IDENTIFIER_DEF] = _IDENTIFIER_SCHEMA
    if decision_ids:
        if any(declared != decision_ids[0] for declared in decision_ids):
            raise ValueError("requirement kinds declare `decision_ids` differently")
        defs[_DECISION_IDS_DEF] = decision_ids[0]
    return factored | {"$defs": defs}


def _referred(node: Any) -> set[str]:
    """The `$defs` names every `$ref` in `node` points at."""
    if isinstance(node, list):
        return {name for item in node for name in _referred(item)}
    if not isinstance(node, dict):
        return set()
    names = {name for value in node.values() for name in _referred(value)}
    ref = node.get("$ref")
    if isinstance(ref, str):
        names.add(ref.removeprefix(_DEFS_PREFIX))
    return names


def _inline_ref_siblings(schema: dict[str, Any]) -> dict[str, Any]:
    """`schema` with every `$ref` that has sibling keys replaced by a copy of its target.

    The siblings sit on top of the copy, so a sibling `description` replaces
    the definition's own and a sibling `default` stays. A bare `$ref` stays a
    reference, and a definition nothing reaches any more is dropped.
    `schema` carries `$defs`, as `_factored` returns it.

    Raises when a definition refers to itself with sibling keys, since its
    copy would never end.
    """
    defs: dict[str, Any] = schema["$defs"]

    def inline(node: Any, inlining: tuple[str, ...]) -> Any:
        if isinstance(node, list):
            return [inline(item, inlining) for item in node]
        if not isinstance(node, dict):
            return node
        ref = node.get("$ref")
        if isinstance(ref, str) and len(node) > 1:
            name = ref.removeprefix(_DEFS_PREFIX)
            if name in inlining:
                raise ValueError(f"`{name}` refers to itself beside other keys")
            siblings = {key: value for key, value in node.items() if key != "$ref"}
            return inline(defs[name] | siblings, (*inlining, name))
        return {key: inline(value, inlining) for key, value in node.items()}

    inlined = {key: inline(value, ()) for key, value in schema.items() if key != "$defs"}
    inlined_defs = {name: inline(definition, (name,)) for name, definition in defs.items()}
    reached: set[str] = set()
    pending = _referred(inlined)
    while pending:
        name = pending.pop()
        reached.add(name)
        pending |= _referred(inlined_defs[name]) - reached
    kept = {name: definition for name, definition in inlined_defs.items() if name in reached}
    return {"$defs": kept} | inlined


def compact_schema(schema: Mapping[str, Any]) -> dict[str, Any]:
    """`schema` compacted, with `Identifier` and `decision_ids` stated once under `$defs`
    and no `$ref` beside another key."""
    return _inline_ref_siblings(_factored(_compact(dict(schema))))


class PlanTool(FunctionTool):
    """A function tool whose declaration is compacted and describes each argument.

    `arguments` maps every argument the model sees to its description; an
    argument without one, or a description naming no argument, raises when
    the declaration is built, so the two cannot drift apart.

    The descriptions ride on the JSON schema ADK builds only with its
    `JSON_SCHEMA_FOR_FUNC_DECL` feature on. With it off the declaration would
    reach the model with no argument described, so construction raises, as
    does building a declaration that has no JSON schema.
    """

    def __init__(self, func: Callable[..., Any], *, arguments: Mapping[str, str]) -> None:
        super().__init__(func)
        if not is_feature_enabled(FeatureName.JSON_SCHEMA_FOR_FUNC_DECL):
            raise RuntimeError(_NO_JSON_SCHEMA.format(name=self.name))
        self._arguments = dict(arguments)

    @override
    def _get_declaration(self) -> types.FunctionDeclaration | None:
        declaration = super()._get_declaration()
        if declaration is None or declaration.parameters_json_schema is None:
            raise RuntimeError(_NO_JSON_SCHEMA.format(name=self.name))
        schema = declaration.parameters_json_schema
        properties: dict[str, Any] = schema["properties"]
        if set(properties) != set(self._arguments):
            raise ValueError(
                f"{self.name} declares {sorted(properties)} and describes "
                f"{sorted(self._arguments)}; describe each argument exactly once"
            )
        described = {
            name: {**argument, "description": self._arguments[name]}
            for name, argument in properties.items()
        }
        declaration.parameters_json_schema = compact_schema(schema | {"properties": described})
        return declaration
