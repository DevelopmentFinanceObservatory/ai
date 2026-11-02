"""What Anthropic's and OpenAI's tool APIs would reject in a tool declaration.

Each rule is a documented limit, checked statically on the declaration's name and
JSON schema. No API is called. Both vendors accept a plain JSON Schema object for a
non-strict tool; the strict modes (Anthropic `strict: true`, OpenAI
`strict: true` on a function) restrict the schema further.

- Both: a name matching `^[a-zA-Z0-9_-]{1,64}$` and a top-level `type: object`.
- Anthropic strict: every object sets `additionalProperties: false`; no recursive
  schema; no `minimum`, `maximum`, `multipleOf`, `minLength`, `maxLength`; no
  array constraint beyond `items` (`minItems` of 0 or 1 is allowed).
- OpenAI strict: every object sets `additionalProperties: false` and lists every
  property in `required` (an optional field is a union with `null`); no `default`;
  none of `minLength`, `maxLength`, `minProperties`, `maxProperties`,
  `patternProperties`, `unevaluatedProperties`, `propertyNames`, `uniqueItems`,
  `contains`, `allOf`, `not`, `if`; a root that is not `anyOf`; at most 5,000
  properties, 10 levels of nesting and 1,000 enum values in all.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from typing import Any

NAME = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")
ANTHROPIC_STRICT_UNSUPPORTED = frozenset(
    {"minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "multipleOf"}
    | {"minLength", "maxLength", "uniqueItems", "contains", "maxItems"}
)
OPENAI_STRICT_UNSUPPORTED = frozenset(
    {"default", "minLength", "maxLength", "minProperties", "maxProperties"}
    | {"patternProperties", "unevaluatedProperties", "propertyNames", "uniqueItems"}
    | {"contains", "allOf", "not", "if"}
)
OPENAI_MAX_PROPERTIES = 5000
OPENAI_MAX_DEPTH = 10
OPENAI_MAX_ENUM_VALUES = 1000


def _schemas(node: Any, path: str = "#") -> Iterator[tuple[str, dict[str, Any]]]:
    """Every schema in `node`, with its JSON pointer."""
    if not isinstance(node, dict):
        return
    yield path, node
    for key in ("properties", "$defs", "definitions"):
        for name, child in (node.get(key) or {}).items():
            yield from _schemas(child, f"{path}/{key}/{name}")
    for key in ("items", "additionalProperties", "not"):
        if isinstance(node.get(key), dict):
            yield from _schemas(node[key], f"{path}/{key}")
    for key in ("anyOf", "oneOf", "allOf", "prefixItems"):
        for index, child in enumerate(node.get(key) or []):
            yield from _schemas(child, f"{path}/{key}/{index}")


def _recursive_defs(schema: dict[str, Any]) -> list[str]:
    """The definitions that reach themselves through `$ref`."""
    defs = schema.get("$defs") or {}
    edges = {
        name: {
            ref.removeprefix("#/$defs/")
            for _, node in _schemas(definition)
            if isinstance(ref := node.get("$ref"), str)
        }
        for name, definition in defs.items()
    }

    def reaches(start: str) -> bool:
        seen, stack = set(), list(edges.get(start, ()))
        while stack:
            name = stack.pop()
            if name == start:
                return True
            if name not in seen:
                seen.add(name)
                stack.extend(edges.get(name, ()))
        return False

    return sorted(name for name in edges if reaches(name))


def _depth(node: Any, defs: dict[str, Any], seen: tuple[str, ...] = ()) -> int:
    """The deepest nesting of objects and arrays, following `$ref` once per path."""
    if not isinstance(node, dict):
        return 0
    if isinstance(ref := node.get("$ref"), str):
        name = ref.removeprefix("#/$defs/")
        return 0 if name in seen else _depth(defs.get(name), defs, (*seen, name))
    children = [
        *(node.get("properties") or {}).values(),
        *([node["items"]] if isinstance(node.get("items"), dict) else []),
        *(node.get("anyOf") or []),
    ]
    nests = node.get("type") in ("object", "array") or "properties" in node
    return int(nests) + max((_depth(child, defs, seen) for child in children), default=0)


def faults(name: str, schema: dict[str, Any]) -> dict[str, list[str]]:
    """Each API mode's faults in a declaration: `both`, `anthropic_strict`, `openai_strict`."""
    found: dict[str, list[str]] = {"both": [], "anthropic_strict": [], "openai_strict": []}
    if not NAME.match(name):
        found["both"].append(f"name {name!r}")
    if schema.get("type") != "object":
        found["both"].append("top-level type is not object")
    if "anyOf" in schema:
        found["openai_strict"].append("root anyOf")
    if recursive := _recursive_defs(schema):
        found["anthropic_strict"].append(f"recursive $defs {recursive}")
    properties = enums = 0
    for path, node in _schemas(schema):
        is_object = node.get("type") == "object" or "properties" in node
        properties += len(node.get("properties") or {})
        enums += len(node.get("enum") or [])
        if is_object and node.get("additionalProperties") is not False:
            for mode in ("anthropic_strict", "openai_strict"):
                found[mode].append(f"{path}: additionalProperties is not false")
        if is_object and set(node.get("properties") or {}) - set(node.get("required") or []):
            found["openai_strict"].append(f"{path}: optional properties")
        for keyword in ANTHROPIC_STRICT_UNSUPPORTED & set(node):
            found["anthropic_strict"].append(f"{path}: {keyword}")
        if node.get("minItems", 0) > 1:
            found["anthropic_strict"].append(f"{path}: minItems {node['minItems']}")
        for keyword in OPENAI_STRICT_UNSUPPORTED & set(node):
            found["openai_strict"].append(f"{path}: {keyword}")
    if properties > OPENAI_MAX_PROPERTIES:
        found["openai_strict"].append(f"{properties} properties")
    if enums > OPENAI_MAX_ENUM_VALUES:
        found["openai_strict"].append(f"{enums} enum values")
    if (depth := _depth(schema, schema.get("$defs") or {})) > OPENAI_MAX_DEPTH:
        found["openai_strict"].append(f"nesting depth {depth}")
    return found


def summary(found: dict[str, list[str]]) -> dict[str, Any]:
    """Each mode's fault count and the distinct kinds of fault, paths dropped."""
    return {
        mode: {
            "faults": len(items),
            "kinds": sorted({item.split(": ", 1)[-1] for item in items}),
        }
        for mode, items in found.items()
    }
