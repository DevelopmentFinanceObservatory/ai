"""The declaration `propose_plan` reaches the model with.

ADK builds a function tool's declaration from its signature: pydantic's
`model_json_schema()` of the arguments, with the whole docstring as the
tool's description and no description on any argument. `PlanTool` keeps that
schema and rewrites it with `compact_schema`, then gives each argument its
own description, so the patch rules and limits a model writes a call from sit
on the argument they govern.

`compact_schema` removes what tells a model nothing its key does not, and
applies to every definition the plan's models reach, so a field added later
gets the same treatment:

- every `title`, which repeats the property or class name it sits under;
- the `null` branch of an optional field's `anyOf`, leaving the field's own
  schema in its place: an optional field is one `required` does not list;
- a `null` or `[]` default, which the same absence from `required` says;
- the description of a field named in `SHARED_FIELDS`, which every
  requirement kind carries with the same meaning. The `evidence_requirements`
  argument states that meaning once.

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

# Fields every requirement kind declares with one meaning, described once on
# the `evidence_requirements` argument rather than on each kind.
SHARED_FIELDS = frozenset({"kind", "id", "decision_ids", "input_id"})

# Why a `PlanTool` refuses to build without a JSON schema to describe.
_NO_JSON_SCHEMA = (
    "{name} carries its argument descriptions on a JSON schema, which ADK builds only "
    "with its JSON_SCHEMA_FOR_FUNC_DECL feature enabled; enable it"
)

# Keys whose value maps names to schemas, so a name there is never a keyword.
_NAMED_SCHEMAS = frozenset({"properties", "$defs"})


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
        elif key == "title" or (key == "default" and (value is None or value == [])):
            continue
        else:
            compacted[key] = _compact(value)
    return _without_null_branch(compacted)


def _shared_undescribed(name: str, schema: Any) -> Any:
    if name in SHARED_FIELDS and isinstance(schema, dict):
        return {key: value for key, value in schema.items() if key != "description"}
    return schema


def compact_schema(schema: Mapping[str, Any]) -> dict[str, Any]:
    """`schema` without titles, `null` branches, empty defaults and shared field descriptions."""
    return _compact(dict(schema))


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
        schema = compact_schema(declaration.parameters_json_schema)
        properties: dict[str, Any] = schema["properties"]
        if set(properties) != set(self._arguments):
            raise ValueError(
                f"{self.name} declares {sorted(properties)} and describes "
                f"{sorted(self._arguments)}; describe each argument exactly once"
            )
        for name, description in self._arguments.items():
            properties[name] = {**properties[name], "description": description}
        declaration.parameters_json_schema = schema
        return declaration
