"""What each tool implements, takes and returns: the compiler's mapping table.

Hand-maintained. A `ToolCapability` names the tool, the operation kinds it
implements, the arguments that take a value another step produced and what
kind of value each returns, and the hash of the argument schema the row was
written against. The MCP server stays the source of truth for the full
schema; this table says how the compiler uses it.

The pinned hash is a gate. When a server changes a tool's arguments, the live
hash no longer matches, the compiler refuses to compile against that tool,
and `tests/compiler/test_capabilities.py` fails until someone re-reads the
schema and re-pins the row. That is the intended cost of a change to a tool
the compiler builds calls for. `output_schema_hash` and
`implementation_version` are the same gate for a tool's output schema and its
declared version, once a server states either: `schema_drift` reports a row
that pins one and no longer matches (`capability_schema_drift`) and, just as
much, a row that pins neither while the live tool now declares one
(`capability_unpinned`) -- silently trusting an unpinned schema is exactly
what the gate exists to rule out.

`EXCLUDED_TOOLS` lists the tools a server serves that the harness never
calls, each with its reason. Every tool on the live surface sits in exactly
one of the two tables, so a new tool is classified before the harness runs
beside it.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from enum import StrEnum
from typing import Literal

from planner.operations import OperationKind
from planner.records import StrictModel
from planner.workflow import ArtifactKind, ToolId
from pydantic import field_validator

from compiler.surface import ToolSchema, UnhashableSchemaError


class Phase(StrEnum):
    """When a tool may be called."""

    DISCOVERY = "discovery"
    WORKFLOW = "workflow"


class Extract(StrEnum):
    """How a value of the declared artifact kind is read from a tool's structured result."""

    SPEC_IDS_FROM_ENVELOPE = "spec_ids_from_envelope"
    FRAME_REF_FROM_MANIFEST = "frame_ref_from_manifest"
    FRAME_REF_PAIR = "frame_ref_pair"


class ToolInput(StrictModel):
    """An argument that takes a value another step produced."""

    parameter: str
    artifact: ArtifactKind


class ToolOutput(StrictModel):
    """A value a tool's structured result carries, and how to read it."""

    extract: Extract
    artifact: ArtifactKind


class ToolCapability(StrictModel):
    """One tool's row in the mapping table.

    `implementation_version` is the version this row pins for the tool, or
    `None` while it pins none; `output_schema_hash` is the same for the
    tool's declared output schema. `schema_drift` compares both fields
    strictly against the live surface: a row that pins a value the live tool
    no longer declares is drift, and so is a row that pins nothing while the
    live tool now declares something (`CAPABILITY_UNPINNED`) -- fail-closed
    both ways, once a tool states either.

    `retry_on_transport_failure` says whether the executor may retry a call
    to this tool after a transport failure -- one where the call never
    reached the server, or its answer never came back, so whether the
    server-side effect happened is unknown. Default `True`. `get_data`'s
    mint is not safely retryable and sets this `False`; every other tool's
    calculation is idempotent under a retry (the frame store is keyed by
    content, `common.materialisation`), so the default holds. `call_timeout_s`
    is the per-call deadline `harness.mcp_deadlines.McpTimeoutPolicy` reads
    through `call_timeouts()`, for both `compiler.surface.McpSurface` (wired
    in by `harness.agents.build_tool_surface`) and `harness.mcp_client`'s
    `McpResolver` / `McpExecutor`; `None` leaves the tool at the general
    default (`Settings.mcp_call_timeout_s`). Must be a finite number greater
    than zero when set -- see `_finite_positive_when_set` below.

    `empty_is_an_answer` says whether a terminal call to this tool may
    legitimately return zero rows. `filter` is the one row that sets it --
    a filter matching nothing is a real answer; every other verb either
    refuses on an empty input already or would be answering a question with
    nothing.
    """

    tool: ToolId
    phase: Phase
    implements: tuple[OperationKind, ...] = ()
    inputs: tuple[ToolInput, ...] = ()
    outputs: tuple[ToolOutput, ...] = ()
    implementation_version: str | None = None
    input_schema_hash: str
    output_schema_hash: str | None = None
    empty_is_an_answer: bool = False
    retry_on_transport_failure: bool = True
    call_timeout_s: float | None = None

    @field_validator("call_timeout_s")
    @classmethod
    def _finite_positive_when_set(cls, value: float | None) -> float | None:
        if value is not None and (not math.isfinite(value) or value <= 0):
            raise ValueError(
                f"call_timeout_s must be a finite number greater than zero, got {value!r}. "
                "A zero, negative, or non-finite deadline either has no sane meaning or is "
                "FastMCP's own signal for an unbounded wait -- exactly what this override "
                "exists to rule out."
            )
        return value


def _data(name: str) -> ToolId:
    return ToolId(server="data-mcp", name=name)


def _analysis(name: str) -> ToolId:
    return ToolId(server="analysis-mcp", name=name)


_FRAME_IN = ToolInput(parameter="frame_ref", artifact=ArtifactKind.FRAME_REF)
_FRAME_OUT = ToolOutput(extract=Extract.FRAME_REF_PAIR, artifact=ArtifactKind.FRAME_REF)

CAPABILITIES: tuple[ToolCapability, ...] = (
    ToolCapability(
        tool=_data("search"),
        phase=Phase.DISCOVERY,
        outputs=(
            ToolOutput(extract=Extract.SPEC_IDS_FROM_ENVELOPE, artifact=ArtifactKind.SPEC_ID),
        ),
        implementation_version="1",
        input_schema_hash="8802e5d3c1925c5dac13947cd16bf0901e2ebb58f50a67ab0151b7f366c81cc7",
        call_timeout_s=30.0,
    ),
    ToolCapability(
        tool=_data("describe"),
        phase=Phase.DISCOVERY,
        implementation_version="1",
        input_schema_hash="181e4b14d6e6af3b18d39786d8a8feeed51d6303473837b97d19d0316b778c1a",
        call_timeout_s=30.0,
    ),
    # The harness's own read: `harness.graph`'s `seed` node resolves each seed's
    # identity again through Search, as does `translator.discovery` for a seed's
    # identity over a plan's years. data-mcp stores the definition as it answers.
    ToolCapability(
        tool=_data("resolve_definition"),
        phase=Phase.DISCOVERY,
        implementation_version="1",
        input_schema_hash="3051ccd467a906bdb6bd16e744d756ac5e0aaafa2ddf8c45a60ddff6d1739c91",
        call_timeout_s=30.0,
    ),
    ToolCapability(
        tool=_data("get_data"),
        phase=Phase.WORKFLOW,
        implements=(OperationKind.RETRIEVE,),
        inputs=(ToolInput(parameter="spec_id", artifact=ArtifactKind.SPEC_ID),),
        outputs=(
            ToolOutput(extract=Extract.FRAME_REF_FROM_MANIFEST, artifact=ArtifactKind.FRAME_REF),
        ),
        implementation_version="1",
        # Pins the `place_types` parameter.
        input_schema_hash="0cef787e58d146ecd015c5f3f29241391d032ba6e9d505111cc9ca9a735ec0a5",
        # A mint is not safely retryable (`data_mcp.get_data_tool._post_mint`).
        # The call timeout sits above data-mcp's `Settings.get_data_deadline_s`,
        # so get_data returns a result or a typed refusal before it fires.
        retry_on_transport_failure=False,
        call_timeout_s=180.0,
    ),
    ToolCapability(
        tool=_analysis("aggregate"),
        phase=Phase.WORKFLOW,
        implements=(OperationKind.AGGREGATE,),
        inputs=(_FRAME_IN,),
        outputs=(_FRAME_OUT,),
        implementation_version="1",
        input_schema_hash="e5a48166b52d8429bb91744f9c7ed27a4e9b7633a5199b27399aa3f4ef101ce8",
        output_schema_hash="12f6d8462ff7ba69118484d92181e7eb66869baa932fe27c44c6a3efab86f2de",
    ),
    ToolCapability(
        tool=_analysis("growth"),
        phase=Phase.WORKFLOW,
        implements=(OperationKind.GROWTH,),
        inputs=(_FRAME_IN,),
        outputs=(_FRAME_OUT,),
        implementation_version="1",
        # Pins the `change` parameter.
        input_schema_hash="5df8ddf49059e6ee34c0372719c4bf66fce33e76c0a6df24f6c2735f0092e97e",
        output_schema_hash="12f6d8462ff7ba69118484d92181e7eb66869baa932fe27c44c6a3efab86f2de",
    ),
    ToolCapability(
        tool=_analysis("compare"),
        phase=Phase.WORKFLOW,
        implements=(OperationKind.COMPARE,),
        inputs=(_FRAME_IN, ToolInput(parameter="right_frame_ref", artifact=ArtifactKind.FRAME_REF)),
        outputs=(_FRAME_OUT,),
        implementation_version="1",
        input_schema_hash="cfeb33c86ed1db23ec65ca70ee792b81051540e45bfaa7780063b15290096720",
        output_schema_hash="12f6d8462ff7ba69118484d92181e7eb66869baa932fe27c44c6a3efab86f2de",
    ),
    ToolCapability(
        tool=_analysis("rank"),
        phase=Phase.WORKFLOW,
        implements=(OperationKind.RANK,),
        inputs=(_FRAME_IN,),
        outputs=(_FRAME_OUT,),
        implementation_version="1",
        # Pins the `per_period` and `limit` parameters.
        input_schema_hash="86103f75945bb2c5c03606c77d3249ae42ee7907dfb2b818376f23d67af73471",
        output_schema_hash="12f6d8462ff7ba69118484d92181e7eb66869baa932fe27c44c6a3efab86f2de",
    ),
    ToolCapability(
        tool=_analysis("share_of"),
        phase=Phase.WORKFLOW,
        implements=(OperationKind.SHARE_OF,),
        inputs=(
            _FRAME_IN,
            ToolInput(parameter="denominator_frame_ref", artifact=ArtifactKind.FRAME_REF),
        ),
        outputs=(_FRAME_OUT,),
        implementation_version="1",
        # Pins the `numerator_by`/`denominator_by` parameters.
        input_schema_hash="88a6d4eccdcb1c4a0116b23d49ca87a1f4f4082c88934efa472ad9021fb2deec",
        output_schema_hash="12f6d8462ff7ba69118484d92181e7eb66869baa932fe27c44c6a3efab86f2de",
    ),
    ToolCapability(
        tool=_analysis("join"),
        phase=Phase.WORKFLOW,
        implements=(OperationKind.JOIN,),
        inputs=(_FRAME_IN, ToolInput(parameter="right_frame_ref", artifact=ArtifactKind.FRAME_REF)),
        outputs=(_FRAME_OUT,),
        implementation_version="1",
        input_schema_hash="1901fa26209ee959653ed0c2938792655e65f78242963a21df120309840eee24",
        output_schema_hash="12f6d8462ff7ba69118484d92181e7eb66869baa932fe27c44c6a3efab86f2de",
    ),
    ToolCapability(
        tool=_analysis("filter"),
        phase=Phase.WORKFLOW,
        implements=(OperationKind.FILTER,),
        inputs=(_FRAME_IN,),
        outputs=(_FRAME_OUT,),
        implementation_version="1",
        # Pins the `place_level` predicate and the 21-character period operand.
        input_schema_hash="11fa15d982a856f7d05d7adac23875e5a6e59f1c0ae507a427bbdcdfe77f53c8",
        output_schema_hash="12f6d8462ff7ba69118484d92181e7eb66869baa932fe27c44c6a3efab86f2de",
        empty_is_an_answer=True,
    ),
)


class ExcludedTool(StrictModel):
    """A tool a server serves that neither the planner nor the translator may use."""

    tool: ToolId
    reason: str


_SKILL_TOOL_REASON = (
    "the planner reads skills through its own tools, and no workflow step takes a skill"
)

EXCLUDED_TOOLS: tuple[ExcludedTool, ...] = (
    ExcludedTool(tool=_data("get_skill"), reason=_SKILL_TOOL_REASON),
    ExcludedTool(tool=_data("get_skill_file"), reason=_SKILL_TOOL_REASON),
    ExcludedTool(
        tool=_data("sources"),
        reason=(
            "Serves the source catalogue and a dataset's documentation to an outside "
            "model answering a question about the data itself; no workflow step reads it."
        ),
    ),
)


def capabilities_for(
    kind: OperationKind, capabilities: Sequence[ToolCapability] = CAPABILITIES
) -> tuple[ToolCapability, ...]:
    """Return every workflow-phase capability implementing `kind`.

    The compiler wants exactly one; zero and several are each an issue it
    reports, so this returns all of them rather than choosing.
    """
    return tuple(
        capability
        for capability in capabilities
        if capability.phase is Phase.WORKFLOW and kind in capability.implements
    )


def call_timeouts(capabilities: Sequence[ToolCapability] = CAPABILITIES) -> dict[ToolId, float]:
    """The per-tool call deadline each row states, for a surface to be built with.

    Only rows that pin `call_timeout_s` appear; a tool absent from the
    mapping gets the surface's own default (`harness.agents.build_tool_surface`
    wires this in as `McpSurface`'s `call_timeouts`).
    """
    return {
        capability.tool: capability.call_timeout_s
        for capability in capabilities
        if capability.call_timeout_s is not None
    }


class SchemaDrift(StrictModel):
    """A tool whose live schema or version no longer matches what its row pins.

    `field` says which of the three a row pins independently: its argument
    schema, its output schema, or its declared version. `pinned` is `None`
    exactly when the row pins nothing for that field while the live tool now
    declares something -- unpinned, not merely stale.
    """

    tool: ToolId
    field: Literal["input_schema", "output_schema", "version"]
    pinned: str | None
    live: str | None


def schema_drift(
    capabilities: Sequence[ToolCapability], schemas: Mapping[str, ToolSchema]
) -> list[SchemaDrift]:
    """Return every capability whose tool is on the surface with a different hash or version.

    Checks all three: the argument schema (always pinned), the output schema,
    and the declared version. A tool missing from the surface is not drift --
    the compiler reports that as an unknown tool -- so it is left out here.

    A live schema RFC 8785 cannot hash (`UnhashableSchemaError`) skips that
    capability entirely rather than raising: `compile.py` reports the same
    tool as `BAD_TOOL_SCHEMA`, and a drift row for a hash that cannot be
    computed would say nothing.
    """
    drift: list[SchemaDrift] = []
    for capability in capabilities:
        schema = schemas.get(capability.tool.name)
        if schema is None or schema.server != capability.tool.server:
            continue
        try:
            input_hash = schema.input_schema_hash
            output_hash = schema.output_schema_hash
        except UnhashableSchemaError:
            continue
        if input_hash != capability.input_schema_hash:
            drift.append(
                SchemaDrift(
                    tool=capability.tool,
                    field="input_schema",
                    pinned=capability.input_schema_hash,
                    live=input_hash,
                )
            )
        if output_hash != capability.output_schema_hash:
            drift.append(
                SchemaDrift(
                    tool=capability.tool,
                    field="output_schema",
                    pinned=capability.output_schema_hash,
                    live=output_hash,
                )
            )
        if schema.version != capability.implementation_version:
            drift.append(
                SchemaDrift(
                    tool=capability.tool,
                    field="version",
                    pinned=capability.implementation_version,
                    live=schema.version,
                )
            )
    return drift
