"""A planner tool as data: its declaration, its argument conversion and its body.

A `ToolSpec` names a tool, describes it, lists its parameters in the order the
model reads them and holds the body a call runs. `function_declaration` builds
the declaration the model is shown from it, and `convert_arguments` and
`missing_arguments` prepare a call's arguments for the body. `run_tool` runs a
call through both and then the body, over the `planner.call.PlannerCall` and
`planner.tools.EndRun` whatever serves the tool supplies. `serve_tool` awaits
that result and marks the call's span failed when the answer is an error.
None of them reads a framework.

The declaration and the conversion are the ones ADK's `FunctionTool` makes
from an equivalent Python signature (google-adk 2.13, with
`JSON_SCHEMA_FOR_FUNC_DECL` on and `FUNCTION_TOOL_ARG_VALIDATION` off, ADK's
defaults), so the model reads the same bytes and a body receives the same
values whichever serves it.
"""

from __future__ import annotations

import inspect
import logging
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from types import UnionType
from typing import Annotated, Any, TypeIs, Union, get_args, get_origin

from google.genai import types
from opentelemetry import trace
from opentelemetry.semconv.attributes.error_attributes import ERROR_TYPE
from opentelemetry.trace import Status, StatusCode
from pydantic import BaseModel, Field, TypeAdapter, create_model
from pydantic_core import PydanticUndefined

from planner.call import PlannerCall
from planner.declaration import compact_schema
from planner.tools import EndRun

logger = logging.getLogger(__name__)

# The default of a parameter the model must state.
REQUIRED: Any = inspect.Parameter.empty

# What a tool's body does with a call: its answer, or an awaitable of it.
type ToolBody = Callable[[PlannerCall, EndRun, dict[str, Any]], object | Awaitable[object]]


@dataclass(frozen=True)
class Parameter:
    """One argument a tool declares.

    `annotation` is the type the model's value is converted to and declared
    as. `default` is `REQUIRED` for an argument the model must state, and is
    otherwise both declared and handed to the body when the call omits it.
    `description` is the argument's own text, if it has one.
    """

    name: str
    annotation: Any
    default: Any = REQUIRED
    description: str | None = None


@dataclass(frozen=True)
class ToolSpec:
    """One planner tool: its name, description, parameters and body.

    `parameters` are in the order the declaration lists them. `body` runs a
    call over the arguments `convert_arguments` gives, with every parameter
    the call omits at its default.

    `compact` declares the tool the way `propose_plan` is declared: every
    parameter carries a description, which overrides any in the parameter's own
    schema, and the whole schema is rewritten by
    `planner.declaration.compact_schema`. Otherwise a description sits in the
    parameter's schema as pydantic places a field's.
    """

    name: str
    description: str
    parameters: tuple[Parameter, ...]
    body: ToolBody
    compact: bool = False


def _parameters_schema(spec: ToolSpec) -> dict[str, Any] | None:
    fields: dict[str, Any] = {}
    for parameter in spec.parameters:
        default = PydanticUndefined if parameter.default is REQUIRED else parameter.default
        if parameter.description is not None and not spec.compact:
            fields[parameter.name] = (
                parameter.annotation,
                Field(default, description=parameter.description),
            )
        else:
            fields[parameter.name] = (parameter.annotation, default)
    if not fields:
        return None
    return create_model(f"{spec.name}Params", **fields).model_json_schema()


def _compacted(spec: ToolSpec, schema: dict[str, Any]) -> dict[str, Any]:
    described = {
        parameter.name: parameter.description
        for parameter in spec.parameters
        if parameter.description is not None
    }
    properties: dict[str, Any] = schema["properties"]
    if set(properties) != set(described):
        raise ValueError(
            f"{spec.name} declares {sorted(properties)} and describes "
            f"{sorted(described)}; describe each argument exactly once"
        )
    placed = {
        name: {**argument, "description": described[name]} for name, argument in properties.items()
    }
    return compact_schema(schema | {"properties": placed})


def function_declaration(spec: ToolSpec) -> types.FunctionDeclaration:
    """The declaration the model is shown for `spec`.

    The parameters' schema is pydantic's schema of a model named
    `<name>Params` whose fields are the parameters, as ADK builds a function
    tool's, rewritten by `compact_schema` when `spec.compact` is set. A tool
    with no parameters declares no schema. This is ADK's declaration on the
    Gemini developer API, which the harness runs on. It carries no response
    schema and keeps optional `anyOf` schemas as pydantic writes them. Vertex
    AI rejects that form, and `planner.agent.refuse_vertex_ai` stops the
    planner building when its model's google-genai client would use Vertex AI.

    Raises:
        ValueError: `spec` is compact and some parameter has no description.
    """
    declaration = types.FunctionDeclaration(name=spec.name, description=spec.description or None)
    schema = _parameters_schema(spec)
    if schema is not None:
        declaration.parameters_json_schema = _compacted(spec, schema) if spec.compact else schema
    return declaration


def _strip_annotated(annotation: Any) -> Any:
    return get_args(annotation)[0] if get_origin(annotation) is Annotated else annotation


def _is_model(annotation: Any) -> TypeIs[type[BaseModel]]:
    return inspect.isclass(annotation) and issubclass(annotation, BaseModel)


def _validated(name: str, validate: Callable[[Any], Any], target: Any, value: Any) -> Any:
    """`validate(value)`, or `value` as sent when it does not validate."""
    try:
        return validate(value)
    except Exception as exc:
        logger.warning("Failed to convert %r to %s: %s", name, target, exc)
        return value


def _converted(name: str, annotation: Any, value: Any) -> Any:
    """`value` converted to `annotation` where ADK converts it, else unchanged."""
    target = _strip_annotated(annotation)
    if get_origin(target) in (Union, UnionType):
        others = [_strip_annotated(arg) for arg in get_args(target) if arg is not type(None)]
        if len(others) > 1 and all(_is_model(other) for other in others):
            if value is None or isinstance(value, tuple(others)):
                return value
            return _validated(name, TypeAdapter(target).validate_python, target, value)
        if len(others) == 1:
            target = others[0]
    return _converted_to(name, target, value)


def _converted_to(name: str, target: Any, value: Any) -> Any:
    # A session store that keeps call arguments as a proto `Struct` returns
    # every number as a double, so an integral float is an int again.
    if target is int and type(value) is float:
        return int(value) if value.is_integer() else value
    if get_origin(target) is list and get_args(target)[:1] == (int,) and isinstance(value, list):
        return [int(item) if type(item) is float and item.is_integer() else item for item in value]
    if _is_model(target):
        if value is None or isinstance(value, target):
            return value
        return _validated(name, target.model_validate, target, value)
    item_type = _strip_annotated(get_args(target)[0]) if get_origin(target) is list else None
    if isinstance(value, list) and _is_model(item_type):
        return [
            _validated(name, item_type.model_validate, item_type, item)
            if isinstance(item, dict)
            else item
            for item in value
        ]
    return value


def convert_arguments(spec: ToolSpec, raw: Mapping[str, Any]) -> dict[str, Any]:
    """The model's `raw` arguments as `spec`'s body receives them.

    A key `spec` declares no parameter for is dropped. A value for a model
    parameter, an optional one, or a list of them item by item becomes that
    model. A value or item that does not validate stays as sent, for the body
    to refuse. An integral float for an `int` or `list[int]` parameter becomes
    an int. Nothing else is checked or converted.
    """
    annotations = {parameter.name: parameter.annotation for parameter in spec.parameters}
    return {
        name: _converted(name, annotations[name], value)
        for name, value in raw.items()
        if name in annotations
    }


def missing_arguments(spec: ToolSpec, arguments: Mapping[str, Any]) -> dict[str, str] | None:
    """The answer to a call whose `arguments` lack a parameter `spec` requires, or `None`.

    The answer is ADK's own, word for word, naming each missing parameter
    in declaration order.
    """
    missing = [
        parameter.name
        for parameter in spec.parameters
        if parameter.default is REQUIRED and parameter.name not in arguments
    ]
    if not missing:
        return None
    names = "\n".join(missing)
    return {
        "error": (
            f"Invoking `{spec.name}()` failed as the following mandatory input parameters "
            f"are not present:\n{names}\nYou could retry calling this tool, but it is "
            "IMPORTANT for you to provide all the mandatory parameters."
        )
    }


def run_tool(
    spec: ToolSpec, raw: Mapping[str, Any], *, call: PlannerCall, end_run: EndRun
) -> object | Awaitable[object]:
    """Run `spec`'s body over the model's `raw` arguments, as a served call does.

    The arguments are converted (`convert_arguments`) and checked
    (`missing_arguments`), and the body runs with every parameter they omit at
    its default. A call missing a required argument answers with
    `missing_arguments`' answer and runs nothing. The result is the body's: an
    awaitable when the body is a coroutine function.
    """
    arguments = convert_arguments(spec, raw)
    refusal = missing_arguments(spec, arguments)
    if refusal is not None:
        return refusal
    filled = {
        parameter.name: arguments.get(parameter.name, parameter.default)
        for parameter in spec.parameters
    }
    return spec.body(call, end_run, filled)


# The `error.type` ADK records on a `FunctionTool`'s span when its answer holds
# an `error` key.
_TOOL_ERROR = "TOOL_ERROR"


async def serve_tool(
    spec: ToolSpec, raw: Mapping[str, Any], *, call: PlannerCall, end_run: EndRun
) -> object:
    """The awaited `run_tool` answer to a call to `spec` with the model's `raw` arguments.

    An answer that is a dict holding an `error` key, such as the refusal of a
    call missing a required argument (`missing_arguments`), marks the current
    span, the call's, failed with `error.type` `TOOL_ERROR`, as ADK marks a
    `FunctionTool`'s.
    """
    answer = run_tool(spec, raw, call=call, end_run=end_run)
    if inspect.isawaitable(answer):
        answer = await answer
    if isinstance(answer, dict) and answer.get("error"):
        span = trace.get_current_span()
        span.set_attribute(ERROR_TYPE, _TOOL_ERROR)
        span.set_status(Status(StatusCode.ERROR, _TOOL_ERROR))
    return answer
