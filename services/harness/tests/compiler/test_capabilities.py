"""Tests for the capability table, including its pin against the real servers.

`test_every_pinned_hash_matches_the_live_schema` is the drift gate: a change to
a tool the compiler builds calls for fails here until the row is re-pinned.
`test_every_pinned_version_matches_the_live_surface` is the same gate over the
other two fields `schema_drift` checks: a row's declared output-schema hash
and implementation version, once every analysis-mcp row pins both and every
data-mcp row pins a version.
"""

from __future__ import annotations

import math

import pytest
from analysis_mcp.engine.aggregate import AGGREGATIONS
from analysis_mcp.server import mcp as analysis_mcp
from compiler.capabilities import (
    CAPABILITIES,
    EXCLUDED_TOOLS,
    Phase,
    SchemaDrift,
    ToolCapability,
    call_timeouts,
    capabilities_for,
    schema_drift,
)
from compiler.surface import McpSurface, ToolSchema
from data_mcp.resolve_definition_tool import RESUBMIT_CALL_BUDGET_S, RESUBMIT_READ_TIMEOUT_S
from data_mcp.search_tool import SEARCH_CALL_BUDGET_S
from data_mcp.server import mcp as data_mcp
from data_mcp.settings import Settings as DataMcpSettings
from harness.agents import build_tool_surface
from harness.settings import Settings
from planner.operations import OperationKind, Reduction
from planner.workflow import ToolId
from pydantic import ValidationError


async def _live_schemas() -> dict[str, ToolSchema]:
    surface = McpSurface(sources={"data-mcp": data_mcp, "analysis-mcp": analysis_mcp})
    return {schema.name: schema for schema in await surface.list_tools()}


async def test_every_pinned_hash_matches_the_live_schema() -> None:
    # Failing here means a server changed a tool's arguments. Re-read the
    # schema, check the row's inputs still name the right seams, and re-pin.
    live = await _live_schemas()

    stale = [
        (c.tool.name, c.input_schema_hash, live[c.tool.name].input_schema_hash)
        for c in CAPABILITIES
        if live[c.tool.name].input_schema_hash != c.input_schema_hash
    ]

    assert stale == []


async def test_every_capability_names_a_tool_its_server_serves() -> None:
    live = await _live_schemas()

    assert all(live[c.tool.name].server == c.tool.server for c in CAPABILITIES)


async def test_declared_inputs_name_real_parameters() -> None:
    live = await _live_schemas()

    for capability in CAPABILITIES:
        declared = set(live[capability.tool.name].parameters["properties"])
        assert {i.parameter for i in capability.inputs} <= declared, capability.tool.name


async def test_every_served_tool_is_either_a_capability_or_excluded() -> None:
    # Failing here means a server serves a tool the harness has not classified.
    # Give it a capability row, or list it in EXCLUDED_TOOLS with its reason.
    live = await _live_schemas()
    served = {ToolId(server=schema.server, name=schema.name) for schema in live.values()}
    rows = {capability.tool for capability in CAPABILITIES}
    excluded = {entry.tool for entry in EXCLUDED_TOOLS}

    assert rows.isdisjoint(excluded)
    assert served == rows | excluded


def test_the_skill_tools_are_excluded_from_the_translator_for_one_reason() -> None:
    skill_tools = {
        ToolId(server="data-mcp", name="get_skill"),
        ToolId(server="data-mcp", name="get_skill_file"),
    }
    excluded = [entry for entry in EXCLUDED_TOOLS if entry.tool in skill_tools]

    assert {entry.tool for entry in excluded} == skill_tools
    assert {entry.reason for entry in excluded} == {
        "the planner reads skills through its own tools, and no workflow step takes a skill"
    }
    assert all(capability.tool not in skill_tools for capability in CAPABILITIES)


def test_every_operation_kind_has_exactly_one_implementing_tool() -> None:
    for kind in OperationKind:
        assert len(capabilities_for(kind)) == 1, kind


def test_discovery_tools_implement_no_operation() -> None:
    discovery = [c for c in CAPABILITIES if c.phase is Phase.DISCOVERY]

    assert {c.tool.name for c in discovery} == {
        "search",
        "describe",
        "resolve_definition",
    }
    assert all(c.implements == () for c in discovery)


def test_the_reduction_vocabulary_is_the_engines() -> None:
    # `Reduction` restates analysis-mcp's closed set so a proposal is checked
    # before it reaches the tool; the two have to agree.
    assert {reduction.value for reduction in Reduction} == set(AGGREGATIONS)


def test_drift_names_a_tool_whose_live_hash_differs() -> None:
    row = next(c for c in CAPABILITIES if c.tool.name == "aggregate")
    changed = ToolSchema(server="analysis-mcp", name="aggregate", parameters={"type": "object"})
    # A row pinning nothing beyond the input hash, isolated from the real
    # row's own output-schema/version pins so only the input hash is compared.
    isolated = ToolCapability(
        tool=row.tool,
        phase=Phase.WORKFLOW,
        implements=(OperationKind.AGGREGATE,),
        input_schema_hash=row.input_schema_hash,
    )

    drift = schema_drift([isolated], {"aggregate": changed})

    assert drift == [
        SchemaDrift(
            tool=row.tool,
            field="input_schema",
            pinned=row.input_schema_hash,
            live=changed.input_schema_hash,
        )
    ]


def test_a_tool_missing_from_the_surface_is_not_drift() -> None:
    assert schema_drift(CAPABILITIES, {}) == []


def test_a_declared_version_is_compared_when_the_row_pins_one() -> None:
    schema = ToolSchema(server="analysis-mcp", name="aggregate", parameters={}, version="0.3.0")
    schema_tool = next(c.tool for c in CAPABILITIES if c.tool.name == "aggregate")
    pinned = ToolCapability(
        tool=schema_tool,
        phase=Phase.WORKFLOW,
        implements=(OperationKind.AGGREGATE,),
        implementation_version="0.2.0",
        input_schema_hash=schema.input_schema_hash,
    )

    assert schema_drift([pinned], {"aggregate": schema}) == [
        SchemaDrift(tool=schema_tool, field="version", pinned="0.2.0", live="0.3.0")
    ]


def test_a_declared_output_schema_the_row_does_not_pin_is_unpinned_not_drifted() -> None:
    # `pinned=None` alongside a live value is the "row pins nothing while the
    # tool now declares something" case -- `compiler.compile` reports it as
    # `capability_unpinned`, not `capability_schema_drift`.
    row = next(c for c in CAPABILITIES if c.tool.name == "aggregate")
    declaring = ToolSchema(
        server="analysis-mcp",
        name="aggregate",
        parameters={},
        output_schema={"type": "object"},
    )
    # A row pinning only the input hash, isolated from the real row's own
    # output-schema/version pins so the row's own output schema stays unpinned.
    row = ToolCapability(
        tool=row.tool,
        phase=Phase.WORKFLOW,
        implements=(OperationKind.AGGREGATE,),
        input_schema_hash=declaring.input_schema_hash,
    )

    drift = schema_drift([row], {"aggregate": declaring})

    assert drift == [
        SchemaDrift(
            tool=row.tool, field="output_schema", pinned=None, live=declaring.output_schema_hash
        )
    ]


def test_output_schema_drift_is_reported_only_when_the_pinned_and_live_hashes_differ() -> None:
    # No real server declares an output schema yet, so the branch comparing
    # two hashes has never run against one; a fixture schema stands in.
    tool = next(c.tool for c in CAPABILITIES if c.tool.name == "aggregate")
    silent = ToolSchema(server="analysis-mcp", name="aggregate", parameters={})
    declaring = silent.model_copy(
        update={"output_schema": {"type": "object", "properties": {"frame_ref": {}}}}
    )
    redeclared = silent.model_copy(
        update={"output_schema": {"type": "object", "properties": {"rows": {}}}}
    )

    def row(output_schema_hash: str | None) -> ToolCapability:
        return ToolCapability(
            tool=tool,
            phase=Phase.WORKFLOW,
            implements=(OperationKind.AGGREGATE,),
            input_schema_hash=silent.input_schema_hash,
            output_schema_hash=output_schema_hash,
        )

    # Neither side states one: nothing to compare, nothing reported.
    assert schema_drift([row(None)], {"aggregate": silent}) == []
    # The row pins exactly what the tool declares.
    assert schema_drift([row(declaring.output_schema_hash)], {"aggregate": declaring}) == []
    # The row pins one hash and the tool now declares another: drift, not unpinned.
    assert schema_drift([row(declaring.output_schema_hash)], {"aggregate": redeclared}) == [
        SchemaDrift(
            tool=tool,
            field="output_schema",
            pinned=declaring.output_schema_hash,
            live=redeclared.output_schema_hash,
        )
    ]


async def test_every_pinned_version_matches_the_live_surface() -> None:
    # Failing here means a server's declared output schema or implementation
    # version no longer matches what a row pins -- the CAPABILITY_UNPINNED /
    # CAPABILITY_SCHEMA_DRIFT check `compile._check_capabilities` runs, made
    # visible directly against the live surface rather than through a
    # compiled plan. Re-read the schema and version, and re-pin the row.
    live = await _live_schemas()

    assert schema_drift(CAPABILITIES, live) == []


def test_an_unhashable_live_schema_is_skipped_not_raised() -> None:
    row = next(c for c in CAPABILITIES if c.tool.name == "aggregate")
    unhashable = ToolSchema(server="analysis-mcp", name="aggregate", parameters={"maximum": 2**53})
    other = next(c for c in CAPABILITIES if c.tool.name == "growth")
    changed = ToolSchema(server="analysis-mcp", name="growth", parameters={"type": "object"})
    # Isolated from the real rows' own output-schema/version pins, so only
    # the input-hash comparison this test is about is at stake.
    isolated_row = ToolCapability(
        tool=row.tool,
        phase=Phase.WORKFLOW,
        implements=(OperationKind.AGGREGATE,),
        input_schema_hash=row.input_schema_hash,
    )
    isolated_other = ToolCapability(
        tool=other.tool,
        phase=Phase.WORKFLOW,
        implements=(OperationKind.GROWTH,),
        input_schema_hash=other.input_schema_hash,
    )

    drift = schema_drift(
        [isolated_row, isolated_other], {"aggregate": unhashable, "growth": changed}
    )

    # No row at all for the unhashable tool -- `compile.py` reports it as
    # `BAD_TOOL_SCHEMA` instead -- while the other tool's drift still shows.
    assert drift == [
        SchemaDrift(
            tool=other.tool,
            field="input_schema",
            pinned=other.input_schema_hash,
            live=changed.input_schema_hash,
        )
    ]


def test_get_data_is_the_only_row_that_is_not_safely_retried() -> None:
    non_retryable = [c.tool.name for c in CAPABILITIES if not c.retry_on_transport_failure]

    assert non_retryable == ["get_data"]


def test_call_timeouts_returns_exactly_the_rows_that_state_one() -> None:
    timeouts = call_timeouts()

    assert timeouts == {
        ToolId(server="data-mcp", name="search"): 60.0,
        ToolId(server="data-mcp", name="describe"): 30.0,
        ToolId(server="data-mcp", name="resolve_definition"): 110.0,
        ToolId(server="data-mcp", name="get_data"): 180.0,
    }


# dfo-apis's search request deadline: `request_timeout_s` (env `SEARCH_REQUEST_TIMEOUT_S`)
# in dfo-apis `src/search/engine/config.py`.
DFO_APIS_SEARCH_DEADLINE_S = 45.0

# dfo-apis's deadline for a resubmit whose member walk enters the full subject read.
DFO_APIS_RESUBMIT_DEADLINE_S = 90.0


def test_search_timeouts_are_ordered_dfo_apis_then_data_mcp_then_harness() -> None:
    """Each layer's deadline outlasts the one it waits on, so a slow search surfaces
    dfo-apis's own typed 503 or 504 instead of a timeout raised a layer above it.
    data-mcp's retries run inside `SEARCH_CALL_BUDGET_S`, which ends before the harness's.

    Read from the shipped defaults, so an environment override cannot mask a
    default that crowds the next layer.
    """
    data_mcp_read_s = DataMcpSettings.model_fields["upstream_read_timeout_s"].default
    harness_call_s = call_timeouts()[ToolId(server="data-mcp", name="search")]

    assert DFO_APIS_SEARCH_DEADLINE_S < data_mcp_read_s < SEARCH_CALL_BUDGET_S < harness_call_s


def test_resubmit_timeouts_are_ordered_dfo_apis_then_data_mcp_then_harness() -> None:
    """A resubmit whose member walk reads the whole subject set runs up to dfo-apis's
    90 s resubmit deadline; data-mcp's read and budget and the harness's
    `resolve_definition` timeout each outlast the layer below."""
    harness_call_s = call_timeouts()[ToolId(server="data-mcp", name="resolve_definition")]

    assert (
        DFO_APIS_RESUBMIT_DEADLINE_S
        < RESUBMIT_READ_TIMEOUT_S
        < RESUBMIT_CALL_BUDGET_S
        < harness_call_s
    )


def test_get_data_call_timeout_outlasts_data_mcps_call_deadline() -> None:
    """get_data must return its own result or refusal before the harness gives up on it.

    Read from the field's default, so an environment override cannot mask a
    shipped default that crowds the timeout.
    """
    call_timeout_s = call_timeouts()[ToolId(server="data-mcp", name="get_data")]
    deadline_s = DataMcpSettings.model_fields["get_data_deadline_s"].default

    assert call_timeout_s - deadline_s >= 10.0


@pytest.mark.parametrize("value", [0.0, -1.0, -0.001, math.nan, math.inf, -math.inf])
def test_a_zero_negative_or_non_finite_call_timeout_override_is_refused(value) -> None:
    """A capability row's own `call_timeout_s` override is the same kind of
    deadline as the three `Settings` fields (`test_settings.py`'s
    `test_a_zero_negative_or_non_finite_mcp_timeout_is_refused`), and must
    be refused the same way rather than silently degrading to no deadline."""
    tool = next(c.tool for c in CAPABILITIES if c.tool.name == "aggregate")

    with pytest.raises(ValidationError, match="call_timeout_s"):
        ToolCapability(
            tool=tool,
            phase=Phase.WORKFLOW,
            implements=(OperationKind.AGGREGATE,),
            input_schema_hash="0" * 32,
            call_timeout_s=value,
        )


def test_a_positive_finite_call_timeout_override_is_accepted() -> None:
    tool = next(c.tool for c in CAPABILITIES if c.tool.name == "aggregate")

    row = ToolCapability(
        tool=tool,
        phase=Phase.WORKFLOW,
        implements=(OperationKind.AGGREGATE,),
        input_schema_hash="0" * 32,
        call_timeout_s=5.0,
    )

    assert row.call_timeout_s == 5.0


def test_build_tool_surface_carries_the_pinned_deadlines() -> None:
    settings = Settings(
        data_mcp_url="https://data-mcp.internal",
        analysis_mcp_url="https://analysis-mcp.internal",
        mcp_api_token="test-token",
    )

    surface = build_tool_surface(settings=settings)

    # Reads the private mapping because `list_tools()`/`call_tool()` are the
    # only public alternatives and both connect to a server.
    assert surface._call_timeouts == call_timeouts()
    assert surface._default_call_timeout_s == settings.mcp_call_timeout_s
    assert surface._list_timeout_s == settings.mcp_list_timeout_s
    assert surface._init_timeout_s == settings.mcp_init_timeout_s
