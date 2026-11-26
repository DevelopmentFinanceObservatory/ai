"""The function declaration Gemini is sent for each orchestrator tool.

A tool is a Python function: its name, its docstring and its parameters other
than `call` (`orchestrator.call.OrchestratorCall`) make the declaration. The
parameters become one pydantic model, `<name>Params`, whose JSON schema is the
declaration's `parameters_json_schema`, so an `Annotated[..., Field(...)]`
parameter's description and constraints reach the model, and a tool with no
other parameter declares none. The docstring is cleaned as `inspect.getdoc`
cleans it. The declarations are pinned byte for byte by the requests
recorded in `tests/fixtures/orchestrator_requests`
(`tests/orchestrator/test_declarations.py`).

With `variant="hydrogen"` (`harness.surface.Surface`), the parameter
descriptions `harness.orchestrator_tools` states for it
(`HYDROGEN_PARAM_DESCRIPTIONS`) replace the tool's own. With `descriptions="minimal"`, the
tool's description and each parameter description come from `orchestrator.minimal_surface`, and
the name, types, enums, required fields and structure stay as built. A text the
minimal surface lacks keeps the full one, logged once per process, unless
`MINIMAL_FULL_PATHS` keeps it on purpose.
"""

from __future__ import annotations

import copy
import importlib
import inspect
import logging
from types import FunctionType
from typing import TYPE_CHECKING, Annotated, Any, get_origin, get_type_hints

from google.genai import types
from pydantic import create_model
from pydantic.fields import FieldInfo
from pydantic_core import PydanticUndefined

if TYPE_CHECKING:
    from collections.abc import Collection, Mapping

    from harness.surface import PromptVariant, ToolDescriptions

logger = logging.getLogger(__name__)

# The parameter a tool reads its call record from, which the model never sees.
CALL_PARAMETER = "call"

# The `tool` or `tool:path` of each full text kept for want of a minimal one,
# logged once per process.
_KEPT_FULL: set[str] = set()


def tool_declaration(
    tool: FunctionType,
    *,
    variant: PromptVariant = "default",
    descriptions: ToolDescriptions = "full",
) -> types.FunctionDeclaration:
    """The declaration of `tool`, from its name, docstring and parameters,
    with the texts `variant` and `descriptions` name (see the module docstring).

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
    if variant == "hydrogen":
        declaration = _hydrogen(declaration)
    if descriptions == "minimal":
        return _minimal(declaration)
    return declaration


def _hydrogen(declaration: types.FunctionDeclaration) -> types.FunctionDeclaration:
    """`declaration` with `harness.orchestrator_tools`' `hydrogen` texts in
    place of the parameter descriptions they name; everything else stays as
    built."""
    tools = importlib.import_module("harness.orchestrator_tools")
    name = declaration.name or ""
    schema = declaration.parameters_json_schema
    texts = tools.HYDROGEN_PARAM_DESCRIPTIONS.get(name)
    if texts and isinstance(schema, dict):
        schema = copy.deepcopy(schema)
        _describe(schema, "", texts, schema.get("$defs", {}), name, set(), set(), partial=True)
    return declaration.model_copy(update={"parameters_json_schema": schema})


def _minimal(declaration: types.FunctionDeclaration) -> types.FunctionDeclaration:
    """`declaration` with `orchestrator.minimal_surface`'s texts in place of
    its description and of each parameter description it has."""
    surface = importlib.import_module("orchestrator.minimal_surface")
    name = declaration.name or ""
    description = surface.MINIMAL_TOOL_DESCRIPTIONS.get(name)
    if description is None:
        _keep_full(name)
        description = declaration.description
    schema = declaration.parameters_json_schema
    if isinstance(schema, dict):
        schema = copy.deepcopy(schema)
        texts = surface.MINIMAL_PARAM_DESCRIPTIONS.get(name, {})
        kept = surface.MINIMAL_FULL_PATHS.get(name, frozenset())
        _describe(schema, "", texts, schema.get("$defs", {}), name, set(), set(), kept=kept)
    return declaration.model_copy(
        update={"description": description, "parameters_json_schema": schema}
    )


def described_paths(declaration: types.FunctionDeclaration) -> set[str]:
    """The dotted path of every parameter description `declaration` carries,
    as `orchestrator.minimal_surface.MINIMAL_PARAM_DESCRIPTIONS` keys them."""
    schema = copy.deepcopy(declaration.parameters_json_schema)
    found: set[str] = set()
    if isinstance(schema, dict):
        _describe(schema, "", None, schema.get("$defs", {}), declaration.name or "", set(), found)
    return found


def _describe(
    node: Any,
    path: str,
    texts: Mapping[str, str] | None,
    defs: Mapping[str, Any],
    tool: str,
    seen: set[tuple[str, str]],
    found: set[str],
    *,
    partial: bool = False,
    kept: Collection[str] = (),
) -> None:
    """Replace each description under `node`, a schema at the dotted `path`,
    with `texts[path]`, and add each such path to `found`; with `texts` None,
    only collect. A property extends the path, and `items`, `anyOf`, `oneOf`,
    `allOf` and a `$ref`'s definition keep it, so a model's docstring takes the
    text of the property that holds it. No description is added where none is,
    as one beside a bare `$ref` never reaches Gemini. With `partial`, or for a
    path in `kept`, a path `texts` lacks keeps its text without a log."""
    if not isinstance(node, dict):
        return
    if "description" in node and path:
        found.add(path)
        if texts is not None and path in texts:
            node["description"] = texts[path]
        elif texts is not None and not partial and path not in kept:
            _keep_full(f"{tool}:{path}")
    for key, value in node.items():
        if key == "properties" and isinstance(value, dict):
            for child, schema in value.items():
                _describe(
                    schema,
                    f"{path}.{child}" if path else child,
                    texts,
                    defs,
                    tool,
                    seen,
                    found,
                    partial=partial,
                    kept=kept,
                )
        elif key == "items":
            _describe(value, path, texts, defs, tool, seen, found, partial=partial, kept=kept)
        elif key in ("anyOf", "oneOf", "allOf") and isinstance(value, list):
            for schema in value:
                _describe(schema, path, texts, defs, tool, seen, found, partial=partial, kept=kept)
        elif key == "$ref" and isinstance(value, str):
            target = value.rsplit("/", 1)[-1]
            if (target, path) not in seen and target in defs:
                seen.add((target, path))
                _describe(
                    defs[target], path, texts, defs, tool, seen, found, partial=partial, kept=kept
                )


def _keep_full(where: str) -> None:
    if where not in _KEPT_FULL:
        _KEPT_FULL.add(where)
        logger.warning("minimal tool descriptions: none for %s, so the full text stays", where)


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


__all__ = ["CALL_PARAMETER", "described_paths", "tool_declaration"]
