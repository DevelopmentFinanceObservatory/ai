"""Does a set of arguments fit a tool's own JSON Schema, and an analysis verb's op-spec?

A tool schema is untrusted input, so one that jsonschema itself rejects is
reported as unusable rather than raising from inside the library part way
through a check, following analysis-mcp's own rule that every failure at a
tool boundary becomes a typed refusal.

A bound argument -- one a `Binding` fills at run time -- is held out of the
literal check and dropped from the schema's `required` list: a value the plan
cannot know yet would fail every time, and demanding it would fail every time
too. `undeclared_parameters` covers what can be checked about a bound
argument now, which is whether the tool declares it at all.

An analysis-mcp verb also refuses some arguments its JSON Schema admits (a
predicate stated twice, for one) as `op_spec_invalid`. `op_spec_errors` runs
the verb's own op-spec checks, so every such call is refused here.
"""

from __future__ import annotations

import copy
from collections.abc import Collection, Mapping
from typing import Final, cast

import rfc8785
from analysis_mcp.identity import OpSpecIdentityError, op_spec_hash_for
from analysis_mcp.op_spec import OpSpecAdapter, Verb
from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError
from pydantic import ValidationError

# The frame a binding fills at run time, as `op_spec_errors` reads it. Any
# well-formed pair stands in, since no op-spec check compares frames.
_BOUND_FRAME: Final = {"spec_id": "ops_" + "0" * 32, "content_version": "ocv_" + "0" * 32}


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


def op_spec_errors(
    *, tool: str, arguments: Mapping[str, object], bound: Collection[str] = ()
) -> list[str]:
    """Why analysis-mcp's `tool` refuses `arguments` as `op_spec_invalid`, one message per error.

    Builds the verb's op-spec and mints its `op_spec_hash` as every verb
    tool does before it runs (`analysis_mcp.tool_pipeline.
    build_op_spec_or_refusal`, `analysis_mcp.frame_ref_resolution.
    op_spec_hash_or_refusal`): the model's validators, then the identity
    step, which refuses an array member stated twice. A verb's parameters
    are its op-spec's fields, with `anchor_key` beside them, which the
    compiler never writes. Each `bound` argument is a frame a binding fills
    at run time. Empty for a tool that is no verb.
    """
    if tool not in Verb:
        return []
    document = {**arguments, **dict.fromkeys(bound, _BOUND_FRAME), "verb": tool}
    try:
        op_spec_hash_for(OpSpecAdapter.validate_python(document))
    except ValidationError as exc:
        return [f"{'/'.join(map(str, error['loc'][1:]))}: {error['msg']}" for error in exc.errors()]
    except (OpSpecIdentityError, rfc8785.CanonicalizationError) as exc:
        return [str(exc)]
    return []
