"""The function declaration Gemini is sent for each orchestrator tool.

A tool is a Python function: its name, its docstring and its parameters other
than `call` (`orchestrator.call.OrchestratorCall`) make the declaration. The
parameters become one pydantic model, `<name>Params`, whose JSON schema is the
declaration's `parameters_json_schema`, so an `Annotated[..., Field(...)]`
parameter's description and constraints reach the model, and a tool with no
other parameter declares none. The docstring is cleaned as `inspect.getdoc`
cleans it. The declarations are byte for byte the ones google-adk's
`FunctionTool` built for the same functions, recorded in
`tests/fixtures/orchestrator_requests` (`tests/orchestrator/test_declarations.py`),
so the model reads the same tools.
"""

from __future__ import annotations

import inspect
from types import FunctionType
from typing import Annotated, Any, get_origin, get_type_hints

from google.genai import types
from pydantic import create_model
from pydantic.fields import FieldInfo
from pydantic_core import PydanticUndefined

# The parameter a tool reads its call record from, which the model never sees.
CALL_PARAMETER = "call"


def tool_declaration(tool: FunctionType) -> types.FunctionDeclaration:
    """The declaration of `tool`, from its name, docstring and parameters.

    A wrapper made with `functools.wraps` declares the function it wraps.

    Raises:
        TypeError: A parameter is positional-only, `*args` or `**kwargs`, which
            no model call can fill by name.
    """
    name = tool.__name__
    declaration = types.FunctionDeclaration(name=name, description=inspect.getdoc(tool) or None)
    fields = _fields(tool)
    if fields:
        declaration.parameters_json_schema = create_model(
            f"{name}Params", **fields
        ).model_json_schema()
    return declaration


def _fields(tool: FunctionType) -> dict[str, Any]:
    hints = get_type_hints(tool, include_extras=True)
    fields: dict[str, Any] = {}
    for name, parameter in inspect.signature(tool).parameters.items():
        if name == CALL_PARAMETER:
            continue
        if parameter.kind not in (parameter.POSITIONAL_OR_KEYWORD, parameter.KEYWORD_ONLY):
            raise TypeError(f"{tool.__name__}: parameter {name} cannot be sent by name")
        annotation = hints.get(name, Any)
        default = (
            PydanticUndefined if parameter.default is inspect.Parameter.empty else parameter.default
        )
        if get_origin(annotation) is Annotated:
            info = FieldInfo.from_annotation(annotation)
            info.default = default
            fields[name] = (info.annotation, info)
        else:
            fields[name] = (annotation, default)
    return fields


__all__ = ["CALL_PARAMETER", "tool_declaration"]
