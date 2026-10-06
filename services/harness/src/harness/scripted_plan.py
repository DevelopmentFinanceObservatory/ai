"""The canned plan scripted mode approves, and the schemas it is compiled against.

`scripted_plan` builds a real `planner.plan.Plan`: one `Decision`, the
`DatasetRequirement` and `AggregateRequirement` behind it, a `Translation`,
and the `Workflow` `compiler.compile.compile_plan` builds from it against
`SCRIPTED_TOOL_SCHEMAS`. Compiling is what pins the workflow's
`input_schema_hash`, `output_schema_hash` and `implementation_version`: they
come out of the same `ToolSchema` objects `harness.scripted.ScriptedSurface`
declares through `list_tools()`, so `executor.run._Run._surface_as_compiled`
reads the plan's pins against the surface's own schemas and finds them
already equal, by construction rather than by a hash copied in by hand.

Retrieval and the one analysis step it feeds mirror `ScriptedExecutor`'s own
two-tool vocabulary: `get_data`, then `aggregate`. `_SCRIPTED_CAPABILITIES`
re-pins the production capability table's `get_data`/`aggregate` rows to
`SCRIPTED_TOOL_SCHEMAS`, the same technique `tests/compiler/conftest.py` uses
to compile against a fixture surface without tripping the drift gate the
table exists to hold against the real servers.
"""

from __future__ import annotations

from typing import Any

import pyarrow as pa
from compiler.capabilities import CAPABILITIES, ToolCapability
from compiler.compile import compile_plan
from compiler.surface import ToolSchema
from data_mcp.manifest import Manifest, SeriesStatisticsModel
from data_mcp.series_statistics import series_statistics_table
from planner.operations import AggregateOperation, RetrieveOperation, Translation
from planner.plan import (
    MAX_QUESTION_CHARS,
    AggregateRequirement,
    DatasetRequirement,
    Decision,
    Entity,
    Period,
    Plan,
    ResolvedBy,
    cut_user_text,
)
from planner.records import GroupBy, Reduction
from planner.workflow import ResolvedEntity, SelectedSource

# The requirement, the retrieval operation and the aggregate operation the
# canned plan always states, under these fixed ids.
_REQUIREMENT_ID = "life_expectancy"
_DERIVED_ID = "life_expectancy_by_period"
_RETRIEVE_STEP_ID = "retrieve"
_AGGREGATE_STEP_ID = "totals"

# What `get_data` mints for a retrieval: `ops_`/`ocv_` ids distinct from the
# `spec_`/`cv_` ids `harness.scripted`'s own frames use, so a scripted-plan
# frame is never mistaken for one `ScriptedExecutor` answers directly.
_RESULT_SPEC_ID = "ops_" + "9" * 32
_RESULT_CONTENT_VERSION = "ocv_" + "9" * 32
_MANIFEST_CONTENT_VERSION = "cv_" + "9" * 32

_GET_DATA_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "spec_id": {"type": "string"},
        "scope": {"type": "object"},
    },
    "required": ["spec_id"],
    "additionalProperties": False,
}

_FRAME_REF_PARAMETER: dict[str, Any] = {
    "type": "object",
    "additionalProperties": {"type": "string"},
}

_AGGREGATE_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "frame_ref": _FRAME_REF_PARAMETER,
        "aggregation": {"type": "string", "minLength": 1, "maxLength": 64},
        "group_by": {"type": "array", "items": {"type": "string"}, "default": []},
        "unit": {"anyOf": [{"type": "string"}, {"type": "null"}], "default": None},
    },
    "required": ["frame_ref", "aggregation"],
    "additionalProperties": False,
}

# The scripted surface's own declared tools: `get_data` and the one analysis
# verb the canned plan's workflow calls. `ScriptedSurface.list_tools()`
# returns exactly these, so a plan compiled against them passes
# `_surface_as_compiled` by construction.
SCRIPTED_TOOL_SCHEMAS: list[ToolSchema] = [
    ToolSchema(
        server="data-mcp",
        name="get_data",
        description="Retrieve the data behind a spec_id.",
        parameters=_GET_DATA_PARAMETERS,
    ),
    ToolSchema(
        server="analysis-mcp",
        name="aggregate",
        description="Group a stored frame and reduce each group.",
        parameters=_AGGREGATE_PARAMETERS,
    ),
]

_SCHEMAS_BY_NAME = {schema.name: schema for schema in SCRIPTED_TOOL_SCHEMAS}

# The production table's `get_data`/`aggregate` rows, re-pinned to
# `SCRIPTED_TOOL_SCHEMAS`. `implements`/`inputs`/`outputs` are structural,
# which tool implements which operation kind, and carry over unchanged; only
# the schema-hash and version pins, which the production rows read off the
# real servers, would otherwise read as drift against a surface that declares
# neither.
_SCRIPTED_CAPABILITIES: tuple[ToolCapability, ...] = tuple(
    capability.model_copy(
        update={
            "input_schema_hash": schema.input_schema_hash,
            "output_schema_hash": schema.output_schema_hash,
            "implementation_version": schema.version,
        }
    )
    for capability in CAPABILITIES
    if (schema := _SCHEMAS_BY_NAME.get(capability.tool.name)) is not None
)


def _scripted_source(spec_id: str) -> SelectedSource:
    """The discovered source `scripted_plan` compiles its retrieval against."""
    return SelectedSource(
        spec_id=spec_id,
        sentence="life expectancy at birth in Kenya, 1960 to 2023",
        confidence="high",
        has_data=True,
        served_count=1,
        entities=(ResolvedEntity(role="country", ref="country/KEN", label="Kenya"),),
        coverage_window=Period(start=1960, end=2023),
        identity={"stat_var": "LifeExpectancy_Person"},
    )


_SCRIPTED_SERIES: list[dict[str, Any]] = [
    {
        "stat_var": "dcid:LifeExpectancy_Person",
        "entities": [{"role": "country", "ref": "country/KEN"}],
        "facet": {},
        "observations": 64,
        "first": {"value": 48.5, "period": "1960"},
        "latest": {"value": 63.6, "period": "2023"},
        "minimum": {"value": 45.0, "period": "1960"},
        "maximum": {"value": 63.6, "period": "2023"},
    }
]


def scripted_manifest(spec_id: str) -> dict[str, Any]:
    """The `get_data` manifest `ScriptedSurface` answers a retrieval of `spec_id` with.

    Sixty-four annual observations, 1960 to 2023, matching `_scripted_source`
    identically (same spec_id, same identity, same resolved entity), so
    `executor.run._Run._check_evidence` finds no drift between what was
    approved and what came back. Carries no statistics artifact of its own
    -- `series_count` states there is one series, and `scripted_fetch_series_stats`
    (this module's own `FetchSeriesStats`, wired in wherever scripted mode
    builds its workflow) answers `_SCRIPTED_SERIES` for it directly, without
    a real fetch: `STUDIO_SCRIPTED`'s whole point is no upstream call.
    """
    return {
        "schema_version": 1,
        "definition": {
            "schema_version": 1,
            "spec_id": spec_id,
            "identity": {"stat_var": "LifeExpectancy_Person"},
        },
        "content_version": _MANIFEST_CONTENT_VERSION,
        "coverage": {"rows": 64, "dimensions": []},
        "citation": {"text": "World Development Indicators", "sources": []},
        "trace": {"resolved_stat_vars": [{"ref": "dcid:LifeExpectancy_Person", "roles": []}]},
        "warnings": [],
        "artifacts": [],
        "series_count": len(_SCRIPTED_SERIES),
        "frame_ref": {"spec_id": spec_id, "content_version": _MANIFEST_CONTENT_VERSION},
    }


async def scripted_fetch_series_stats(manifest: Manifest) -> pa.Table | None:
    """`executor.run.FetchSeriesStats` for scripted mode: `_SCRIPTED_SERIES`, always,
    never a real fetch. `scripted_manifest` is the only manifest this mode ever mints, so
    there is nothing to key this on; a caller passing this in for any other manifest shape
    is a wiring bug, not a case this function needs to detect.
    """
    del manifest
    return series_statistics_table(
        [SeriesStatisticsModel.model_validate(row) for row in _SCRIPTED_SERIES]
    )


def scripted_aggregate_result() -> dict[str, Any]:
    """The `aggregate` result `ScriptedSurface` answers the canned plan's terminal step with."""
    return {
        "spec_id": _RESULT_SPEC_ID,
        "content_version": _RESULT_CONTENT_VERSION,
        "rows": 64,
        "role_keys": [],
        "stat_var_count": 1,
    }


def scripted_plan(question: str, *, spec_id: str | None, parts: int) -> Plan:
    """Build the canned scripted plan a `PlannerModel` proposes in scripted mode.

    One `Decision`, the `DatasetRequirement` and `AggregateRequirement` it
    fixes, a `Translation` retrieving that requirement and reducing it to one
    figure per period, and the `Workflow` the compiler builds from them
    against `SCRIPTED_TOOL_SCHEMAS`. Walks `DRAFT` to `PENDING_APPROVAL`
    through `with_translation()`, the same transition a model-drafted plan
    goes through.

    Args:
        question: The plan's stated question, cut to `MAX_QUESTION_CHARS` by
            `cut_user_text` when it is longer.
        spec_id: The source to retrieve, `harness.scripted.SCRIPTED_SPEC` or
            `SCRIPTED_MISNAMED_SPEC`, as `ScriptedResolver` resolves the
            question to. Falls back to `SCRIPTED_SPEC` when `None`, which
            `ScriptedResolver` never returns for a question this is called
            for, but the signature states it as optional to mirror the
            resolver's own return shape.
        parts: Unused; scripted mode always resolves to one part, so this
            always builds a single dataset requirement. Accepted for
            signature parity with the resolved envelope a real
            `PlannerModel` reads.

    Returns:
        A `Plan` in `PENDING_APPROVAL`, its workflow's pins equal to what
        `harness.scripted.ScriptedSurface.list_tools()` declares.
    """
    del parts
    resolved_spec_id = spec_id if spec_id is not None else "spec_" + "f" * 32

    decision = Decision(
        slot="measure",
        description="Which measure answers the scripted question.",
        options=["life expectancy at birth", "healthy life expectancy"],
        chosen="life expectancy at birth",
        resolved_by=ResolvedBy.PLANNER_DEFAULT,
        rationale="Life expectancy at birth is the measure scripted mode always answers with.",
    )
    requirement = DatasetRequirement(
        id=_REQUIREMENT_ID,
        decision_ids=[decision.slot],
        measure="life expectancy at birth",
        entities=[Entity(name="Kenya")],
        period=Period(start=1960, end=2023),
    )
    derived = AggregateRequirement(
        id=_DERIVED_ID,
        decision_ids=[decision.slot],
        input_id=_REQUIREMENT_ID,
        reduction=Reduction.MEAN,
        group_by=(GroupBy.PERIOD,),
    )
    draft = Plan(
        question=cut_user_text(question, MAX_QUESTION_CHARS),
        decisions=[decision],
        evidence_requirements=[requirement, derived],
    )
    translation = Translation(
        operations=[
            RetrieveOperation(
                id=_RETRIEVE_STEP_ID, requirement_id=_REQUIREMENT_ID, spec_id=resolved_spec_id
            ),
            AggregateOperation(
                id=_AGGREGATE_STEP_ID,
                requirement_id=_DERIVED_ID,
                input=_RETRIEVE_STEP_ID,
                reduction=Reduction.MEAN,
                group_by=(GroupBy.PERIOD,),
            ),
        ],
        terminal=[_AGGREGATE_STEP_ID],
    )
    compiled = compile_plan(
        plan=draft,
        translation=translation,
        schemas=_SCHEMAS_BY_NAME,
        discovered={_REQUIREMENT_ID: _scripted_source(resolved_spec_id)},
        capabilities=_SCRIPTED_CAPABILITIES,
    )
    if compiled.workflow is None:
        raise RuntimeError(
            f"the canned scripted plan did not compile against its own fixed schemas: "
            f"{compiled.issues!r}"
        )
    return draft.with_translation(translation=translation, workflow=compiled.workflow)
