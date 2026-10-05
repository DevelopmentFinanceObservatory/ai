"""Does a set of arguments fit a tool's own JSON Schema?

A tool schema is untrusted input, so one that jsonschema itself rejects is
reported as unusable rather than raising from inside the library part way
through a check, following analysis-mcp's own rule that every failure at a
tool boundary becomes a typed refusal.

A bound argument -- one a `Binding` fills at run time -- is held out of the
literal check and dropped from the schema's `required` list: a value the plan
cannot know yet would fail every time, and demanding it would fail every time
too. `undeclared_parameters` covers what can be checked about a bound
argument now, which is whether the tool declares it at all.
"""

from __future__ import annotations

import copy
from collections.abc import Collection, Mapping
from typing import cast

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError


def unusable_schema_reason(schema: Mapping[str, object]) -> str | None:
    """Return why `schema` is not valid JSON Schema, or `None` if it is."""
    try:
        Draft202012Validator.check_schema(dict(schema))
    except SchemaError as exc:
        return exc.message
    return None


def declared_parameters(schema: Mapping[str, object]) -> set[str]:
    """Return the argument names `schema` declares.

    Absent `properties` means a tool that declares no arguments. Callers run
    `unusable_schema_reason` first, so jsonschema has already rejected any
    shape this cast would misread.
    """
    properties = cast(dict[str, object], schema.get("properties", {}))
    return set(properties)


def undeclared_parameters(names: Collection[str], schema: Mapping[str, object]) -> list[str]:
    """Return the names in `names` that `schema` does not declare, sorted."""
    return sorted(set(names) - declared_parameters(schema))


def literal_argument_errors(
    *,
    arguments: Mapping[str, object],
    schema: Mapping[str, object],
    bound: Collection[str] = (),
) -> list[str]:
    """Validate literal arguments against `schema`, holding `bound` ones out.

    Args:
        arguments: The literal arguments to validate.
        schema: The tool's own `inputSchema`, already known to be usable.
        bound: Argument names a binding fills at run time. They are held out
            of the payload and dropped from `required`.

    Returns:
        One message per error, sorted by the argument's location and then the
        message, so two runs report the same sequence.
    """
    validator = Draft202012Validator(_schema_not_requiring(schema, set(bound)))
    errors = sorted(
        validator.iter_errors(dict(arguments)),
        key=lambda error: (list(map(str, error.absolute_path)), error.message),
    )
    messages: list[str] = []
    for error in errors:
        location = "/".join(str(part) for part in error.absolute_path)
        messages.append(f"{location}: {error.message}" if location else error.message)
    return messages


def _schema_not_requiring(schema: Mapping[str, object], bound: set[str]) -> dict[str, object]:
    """Return `schema` with `bound` arguments dropped from `required`.

    Their entries in `properties` stay: `properties` constrains only the keys
    a payload actually carries, and a bound argument is held out of the
    payload, so its subschema never applies either way.
    """
    parameters = copy.deepcopy(dict(schema))
    required = cast(list[str], parameters.get("required", []))
    parameters["required"] = [name for name in required if name not in bound]
    return parameters
