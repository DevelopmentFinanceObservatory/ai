"""The input schema a tool declares, cut to what a model reads.

Pydantic derives a tool's schema from its signature with developer-facing
parts a client pays for on every turn: definitions stated under `$defs` and
referred to by `$ref`, an `anyOf: [X, {"type": "null"}]` on every argument
defaulting to null, and sentences the tool description already says.
`declared_schema` returns the schema with:

1. every `$comment` removed;
2. every empty `required: []` removed;
3. each optional property that states `default: null` written as its own
   type, since that default says an absent property already means null. Every
   other nullable property keeps its null branch, as in a presentation patch,
   whose fields state no default because omitting one keeps the stored value
   and null clears it;
4. each sentence of seven words or more kept once across the description and
   the schema, the first occurrence staying, in the description when it is
   there. Sentences compare with whitespace collapsed, since a description
   taken from a docstring keeps its line wraps;
5. every definition inlined, with no `$defs` or `$ref`, since Gemini and Codex
   drop a description that sits beside a `$ref`.

FastMCP parses a call's arguments against the function signature, so a call
the declared schema no longer shows, such as an explicit null, still parses.
"""

from __future__ import annotations

import copy
import re
from collections.abc import Callable, Iterator
from typing import Any

from fastmcp.tools import Tool
from fastmcp.utilities.json_schema import dereference_refs

type Schema = dict[str, Any]

_NULL: Schema = {"type": "null"}

# A sentence ends at `.`, `!` or `?` before whitespace, except the stop that closes
# "e.g." or "i.e.".
_SENTENCE_BREAK = re.compile(r"(?<!\be\.g\.)(?<!\bi\.e\.)(?<=[.!?])\s+")
_MIN_WORDS = 7


def declared_tool(tool: Tool) -> Tool:
    """`tool` with its parameters replaced by `declared_schema`."""
    tool.parameters = declared_schema(tool.parameters, description=tool.description or "")
    return tool


def declared_schema(schema: Schema, *, description: str) -> Schema:
    """`schema` under the five rules in this module's docstring.

    Raises:
        ValueError: `schema` holds a `$ref` outside the document, such as an
            `https://` one, or a `$ref` or `$defs` that inlining cannot remove,
            such as a reference to a missing or recursive definition. The
            message names the JSON pointer of the first one met.
    """
    for pointer, node in _nodes(schema):
        ref = node.get("$ref")
        if isinstance(ref, str) and not ref.startswith("#"):
            raise ValueError(f"{pointer or '/'} refers outside the schema: $ref {ref!r}")
    result = dereference_refs(copy.deepcopy(schema))
    for pointer, node in _nodes(result):
        if left := sorted({"$ref", "$defs"} & node.keys()):
            raise ValueError(f"{pointer or '/'} keeps {' and '.join(left)} after inlining")
    _walk(result, _drop_comment_and_empty_required)
    _post_order(result, _collapse_optional_nullables)
    _drop_repeated_sentences(result, description=description)
    return result


def _nodes(node: Any, pointer: str = "") -> Iterator[tuple[str, Schema]]:
    """Every dict in `node` with its JSON pointer, parents before children."""
    if isinstance(node, dict):
        yield pointer, node
        for key, value in node.items():
            yield from _nodes(value, f"{pointer}/{_escaped(str(key))}")
    elif isinstance(node, list):
        for index, item in enumerate(node):
            yield from _nodes(item, f"{pointer}/{index}")


def _escaped(token: str) -> str:
    """`token` as one JSON pointer segment (RFC 6901)."""
    return token.replace("~", "~0").replace("/", "~1")


def _walk(node: Any, visit: Callable[[Schema], None]) -> None:
    """Call `visit` on every dict in `node`, parents before children."""
    if isinstance(node, dict):
        visit(node)
        for value in list(node.values()):
            _walk(value, visit)
    elif isinstance(node, list):
        for item in node:
            _walk(item, visit)


def _post_order(node: Any, visit: Callable[[Schema], None]) -> None:
    """Call `visit` on every dict in `node`, children before parents."""
    if isinstance(node, dict):
        for value in list(node.values()):
            _post_order(value, visit)
        visit(node)
    elif isinstance(node, list):
        for item in node:
            _post_order(item, visit)


def _drop_comment_and_empty_required(node: Schema) -> None:
    node.pop("$comment", None)
    if node.get("required") == []:
        del node["required"]


def _collapse_optional_nullables(node: Schema) -> None:
    """Replace `anyOf: [X, null]` with X on each optional property that states
    `default: null`.

    A required property keeps its null branch, since there null is a value
    the caller has to be able to send. So does a property with no default or
    another one, where null and omission differ.
    """
    properties = node.get("properties")
    if not isinstance(properties, dict):
        return
    required = set(node.get("required", ()))
    for name, prop in properties.items():
        if (
            name not in required
            and isinstance(prop, dict)
            and isinstance(prop.get("anyOf"), list)
            and _NULL in prop["anyOf"]
            and "default" in prop
            and prop["default"] is None
        ):
            properties[name] = _without_null(prop)


def _without_null(prop: Schema) -> Schema:
    """`prop` without its null branch and its `default` of null.

    Where both the property and its one remaining branch carry a description,
    the property's own wins. It is written for this argument, and the
    branch's is the docstring of a type the argument shares.
    """
    branches = [branch for branch in prop["anyOf"] if branch != _NULL]
    outer = {key: value for key, value in prop.items() if key not in ("anyOf", "default")}
    if len(branches) > 1:
        return {**outer, "anyOf": branches}
    return {**branches[0], **outer}


def _sentences(text: str) -> list[str]:
    return _SENTENCE_BREAK.split(text.strip())


def _normalised(sentence: str) -> str:
    return " ".join(sentence.split())


def _drop_repeated_sentences(schema: Schema, *, description: str) -> None:
    """Keep each sentence of `_MIN_WORDS` words or more once, in reading order.

    A schema description that loses a sentence is rejoined with single
    spaces. One that loses none keeps its text as written.
    """
    seen = {
        _normalised(sentence)
        for sentence in _sentences(description)
        if len(sentence.split()) >= _MIN_WORDS
    }

    def visit(node: Schema) -> None:
        text = node.get("description")
        if not isinstance(text, str):
            return
        kept: list[str] = []
        for sentence in _sentences(text):
            key = _normalised(sentence)
            if len(sentence.split()) >= _MIN_WORDS:
                if key in seen:
                    continue
                seen.add(key)
            kept.append(sentence)
        if not kept:
            del node["description"]
        elif len(kept) < len(_sentences(text)):
            node["description"] = " ".join(_normalised(sentence) for sentence in kept)

    _walk(schema, visit)
