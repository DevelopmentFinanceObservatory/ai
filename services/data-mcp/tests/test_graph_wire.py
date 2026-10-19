"""`data_mcp.graph_wire` is generated from the committed Graph API schema snapshot.

`fixtures/graph_api_schema.json` is the body of apis' `GET /api/graph/schema`. The test
renders the node and resolve models from it with datamodel-code-generator and fails when the
committed module differs. `just graph-models` rewrites the module from the snapshot, and the
diff of that regeneration is the review artifact. `just graph-schema` refreshes the snapshot
from a deployed apis.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from datamodel_code_generator import (
    DataModelType,
    Formatter,
    InputFileType,
    PythonVersion,
    generate,
)

_TESTS = Path(__file__).resolve().parent
_SNAPSHOT = _TESTS / "fixtures" / "graph_api_schema.json"
_MODULE = _TESTS.parent / "src" / "data_mcp" / "graph_wire.py"
_WORKSPACE_PYPROJECT = _TESTS.parents[2] / "pyproject.toml"

# The response bodies data-mcp reads, keyed by the snapshot's name for each, with the model
# name each takes in the module.
_SCHEMAS = {
    "resolve_response": "ResolveResponse",
    "nodes_response": "NodesResponse",
    "node_response": "NodeResponse",
}

_HEADER = (
    "# Generated from services/data-mcp/tests/fixtures/graph_api_schema.json by\n"
    "# `just graph-models`. Do not edit by hand."
)


def _components() -> dict[str, object]:
    """Each served schema and the definitions it names, as one set of named schemas.

    Every served schema carries its own `$defs`. A definition two schemas share must be the
    same in both, since the module holds one model for it.
    """
    served = json.loads(_SNAPSHOT.read_text())["schemas"]
    components: dict[str, object] = {}
    for key, name in _SCHEMAS.items():
        schema = dict(served[key])
        for def_name, definition in schema.pop("$defs", {}).items():
            if components.setdefault(def_name, definition) != definition:
                raise ValueError(f"{key} defines {def_name} differently from another schema")
        components[name] = schema
    return components


def render_graph_wire() -> str:
    """The module the snapshot generates.

    The schemas go in as an OpenAPI document's components, so every one becomes a model and
    no root model is added. Their `$defs` references point at the components instead.
    """
    document = {
        "openapi": "3.1.0",
        "info": {"title": "dfo-apis Graph API", "version": "snapshot"},
        "paths": {},
        "components": {"schemas": _components()},
    }
    text = json.dumps(document).replace('"#/$defs/', '"#/components/schemas/')
    rendered = generate(
        text,
        input_file_type=InputFileType.OpenAPI,
        output_model_type=DataModelType.PydanticV2BaseModel,
        target_python_version=PythonVersion.PY_313,
        enum_field_as_literal="all",
        use_annotated=True,
        field_constraints=True,
        snake_case_field=True,
        collapse_root_models=True,
        use_schema_description=True,
        use_field_description=True,
        use_double_quotes=True,
        disable_timestamp=True,
        custom_file_header=_HEADER,
        formatters=[Formatter.RUFF_CHECK, Formatter.RUFF_FORMAT],
        # The suite runs in a temporary directory, so the formatters take the workspace's
        # ruff settings from here, as `just graph-models` does.
        settings_path=_WORKSPACE_PYPROJECT,
    )
    assert isinstance(rendered, str)
    # The formatters leave the last line unterminated.
    return rendered.rstrip("\n") + "\n"


def test_graph_wire_matches_the_schema_snapshot() -> None:
    rendered = render_graph_wire()
    if os.environ.get("GRAPH_WIRE_UPDATE"):
        _MODULE.write_text(rendered)
    assert _MODULE.read_text() == rendered, (
        "data_mcp/graph_wire.py differs from what the schema snapshot generates; "
        "run `just graph-models` and review the diff"
    )
