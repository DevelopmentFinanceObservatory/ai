"""What plan-mcp changes between the planner's tools and an outside model.

- `drop_unset` reads an empty string, an empty list or object, `None`, and a
  zero the field's schema does not admit, in an optional field, as absent. A
  client whose tools are strict fills every optional field with such a value.
- `outward` is a planner answer as an outside model reads it: actions and
  messages name `record_user_answer` where the planner names its own ask
  tool, a translation referral answers `referred` under its own code, and an
  `invalid_plan` refusal names the expected shape of each field that failed.
- `lean` is a declaration's schema without the bounds and flags a server
  checks anyway: the refusal names them when a value breaks one.
- `table` is a frame's rows in the compact shape `run_plan` returns them in.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from harness.turn_events import REQUEST_INPUT_FUNCTION_CALL_NAME

ASK_TOOL = "record_user_answer"
REFERRED = "referred"

# What `lean` removes from a declaration: bounds and flags the planner's own
# validation enforces, which a refusal names (`expected`) when a value breaks one.
_CHECKED_KEYWORDS = frozenset(
    {
        "additionalProperties",
        "default",
        "exclusiveMaximum",
        "exclusiveMinimum",
        "maxItems",
        "maxLength",
        "maximum",
        "minItems",
        "minLength",
        "minimum",
        "title",
    }
)
# How much of a schema an `expected` entry shows, and for how many fields.
_FRAGMENT_CHARS = 600
_FRAGMENTS = 3


# --- Strict-mode clients ---------------------------------------------------------


def _resolve(node: Any, root: Mapping[str, Any]) -> Any:
    """`node` with a local `$ref` followed."""
    while isinstance(node, Mapping) and isinstance(node.get("$ref"), str):
        name = node["$ref"].rsplit("/", 1)[-1]
        node = (root.get("$defs") or root.get("definitions") or {}).get(name, {})
    return node


def _branches(node: Mapping[str, Any], root: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return [_resolve(b, root) for b in (node.get("oneOf") or node.get("anyOf") or [])]


def _branch_for(value: Mapping[str, Any], node: Mapping[str, Any], root: Mapping[str, Any]) -> Any:
    """The object branch of a union `value` takes: the one whose `kind` constant it
    names, else the only object branch, else `None`."""
    objects = [b for b in _branches(node, root) if isinstance(b.get("properties"), Mapping)]
    kind = value.get("kind")
    for branch in objects:
        const = (branch["properties"].get("kind") or {}).get("const")
        if const is not None and const == kind:
            return branch
    return objects[0] if len(objects) == 1 else None


def _admits_zero(node: Mapping[str, Any], root: Mapping[str, Any]) -> bool:
    """Whether a field's schema states a lower bound at or below zero."""
    nodes = [node, *_branches(node, root)]
    return any(
        isinstance(n.get("minimum"), int | float) and n["minimum"] <= 0
        for n in nodes
        if isinstance(n, Mapping)
    )


def _unset(value: Any, node: Mapping[str, Any], root: Mapping[str, Any]) -> bool:
    if value is None or value in ("", [], {}):
        return True
    if isinstance(value, bool) or not isinstance(value, int | float):
        return False
    return value == 0 and not _admits_zero(node, root)


def drop_unset(value: Any, schema: Mapping[str, Any], root: Mapping[str, Any] | None = None) -> Any:
    """`value` with every optional field `_unset` reads as absent left out, at every depth
    the schema describes. A required field keeps whatever it holds."""
    root = schema if root is None else root
    node = _resolve(schema, root)
    if isinstance(value, list):
        items = _resolve(node.get("items") or {}, root) if isinstance(node, Mapping) else {}
        return [drop_unset(item, items, root) for item in value]
    if not isinstance(value, Mapping) or not isinstance(node, Mapping):
        return value
    if not isinstance(node.get("properties"), Mapping):
        branch = _branch_for(value, node, root)
        if branch is None:
            return dict(value)
        node = branch
    properties: Mapping[str, Any] = node["properties"]
    required = set(node.get("required") or ())
    kept: dict[str, Any] = {}
    for key, item in value.items():
        field = _resolve(properties.get(key) or {}, root)
        cleaned = drop_unset(item, field, root)
        if key not in required and _unset(cleaned, field, root):
            continue
        kept[key] = cleaned
    return kept


# --- Answers as an outside model reads them ----------------------------------------


def _renamed(text: str) -> str:
    return text.replace(f"`{REQUEST_INPUT_FUNCTION_CALL_NAME}`", f"`{ASK_TOOL}`")


def _fragment(node: Any, root: Mapping[str, Any], depth: int = 0) -> Any:
    """`node` without its prose, objects below the first level shown by their fields."""
    node = _resolve(node, root)
    if isinstance(node, list):
        return [_fragment(item, root, depth) for item in node]
    if not isinstance(node, Mapping):
        return node
    if depth > 1 and isinstance(node.get("properties"), Mapping):
        return {"type": "object", "fields": sorted(node["properties"])}
    return {
        key: {name: _fragment(prop, root, depth + 1) for name, prop in value.items()}
        if key == "properties" and isinstance(value, Mapping)
        else _fragment(value, root, depth)
        for key, value in node.items()
        if key not in ("description", "title")
    }


def expected_shape(schema: Mapping[str, Any], loc: Sequence[Any]) -> tuple[str, Any] | None:
    """The path a pydantic error's `loc` names and the schema at it, without prose,
    or `None` when the error names no field. A union tag in `loc` (the requirement's
    `kind`) picks its branch."""
    node: Any = _resolve(schema, schema)
    path = ""
    for step in loc:
        if not isinstance(node, Mapping):
            return None
        if isinstance(step, int):
            node = _resolve(node.get("items") or {}, schema)
            path += f"[{step}]"
            continue
        properties = node.get("properties")
        if isinstance(properties, Mapping) and step in properties:
            node = _resolve(properties[step], schema)
            path += f".{step}" if path else str(step)
            continue
        tagged = [
            b
            for b in _branches(node, schema)
            if ((b.get("properties") or {}).get("kind") or {}).get("const") == step
        ]
        if not tagged:
            break
        node = tagged[0]
    if not path:
        return None
    return path, _fragment(node, schema)


def _expected(answer: Mapping[str, Any], schema: Mapping[str, Any]) -> dict[str, Any]:
    shapes: dict[str, Any] = {}
    for error in (answer.get("detail") or {}).get("errors") or []:
        found = expected_shape(schema, error.get("loc") or ())
        if found is None or found[0] in shapes:
            continue
        path, fragment = found
        text = json.dumps(fragment, separators=(",", ":"))
        shapes[path] = fragment if len(text) <= _FRAGMENT_CHARS else text[:_FRAGMENT_CHARS] + "..."
        if len(shapes) == _FRAGMENTS:
            break
    return shapes


def outward(
    answer: Mapping[str, Any], *, schema: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """A planner tool's `answer` as plan-mcp returns it.

    - An action naming the planner's ask tool names `record_user_answer`, and so
      does the message.
    - An `ok` answer whose translation referred a requirement back, so no plan is
      ready, answers `referred` under the referral's `kind`.
    - An `invalid_plan` refusal adds `expected`: for each field an error names,
      that field's schema in `schema` without its prose.
    """
    out = dict(answer)
    if isinstance(out.get("message"), str):
        out["message"] = _renamed(out["message"])
    action = out.get("action")
    if isinstance(action, Mapping) and action.get("call") == REQUEST_INPUT_FUNCTION_CALL_NAME:
        out["action"] = {**action, "call": ASK_TOOL}
    translation = out.get("translation")
    if (
        out.get("outcome") == "ok"
        and "summary" not in out
        and isinstance(translation, Mapping)
        and translation.get("outcome") not in (None, "ok")
    ):
        kind = translation.get("kind") or translation.get("outcome")
        out = {
            "outcome": REFERRED,
            "code": kind,
            **{k: v for k, v in out.items() if k != "outcome"},
        }
    shapes = _expected(out, schema) if out.get("code") == "invalid_plan" and schema else None
    if shapes:
        out["expected"] = shapes
    return out


def guidance_text(instruction: str) -> str:
    """The planner's instruction with its ask tool named as plan-mcp serves it."""
    return _renamed(instruction)


# --- Declarations ------------------------------------------------------------------


def lean(schema: Any) -> Any:
    """`schema` without `_CHECKED_KEYWORDS`, at every depth. Property names are
    names, not keywords, so a field called `default` stays."""
    if isinstance(schema, list):
        return [lean(item) for item in schema]
    if not isinstance(schema, Mapping):
        return schema
    return {
        key: {name: lean(prop) for name, prop in value.items()}
        if key in ("properties", "$defs")
        else lean(value)
        for key, value in schema.items()
        if key not in _CHECKED_KEYWORDS
    }


# --- Rows --------------------------------------------------------------------------


def table(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """`rows` as `{"fixed", "columns", "rows"}`: each column holding one value in every
    row once under `fixed`, and the others as one list per row, in column order."""
    columns = list(dict.fromkeys(key for row in rows for key in row))
    fixed = {
        column: rows[0].get(column)
        for column in columns
        if len(rows) > 1 and all(row.get(column) == rows[0].get(column) for row in rows)
    }
    varying = [column for column in columns if column not in fixed]
    return {
        "fixed": fixed,
        "columns": varying,
        "rows": [[row.get(column) for column in varying] for row in rows],
    }
