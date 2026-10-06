"""Fixtures for the translator suite: a canned data server and a draft plan.

`data_server` is a `FastMCP` named `data-mcp` whose `search` answers by the
`variable` it is asked about -- the requirement's own `measure`, since
`discover_source` never takes one as an argument any more -- so a test states
which answer it wants by the measure the requirement it discovers on carries.
Its definite answer resolves Kenya alone, exactly as wide as the shared
`requirement`, since the compiler refuses a source resolving an entity the
requirement never named. A phrase in `WARNING_PHRASES` adds that warning to
whatever the answer is, so a test can have an answer arrive with a dropped
role or an advisory note. It is served over `fastmcp.Client`'s in-memory
transport, so the discovery tools are exercised through the MCP protocol.
`get_data` is there to show the translator never wraps it.

The agent tests compile against the real tool schemas, read once from the real
in-memory servers, with canned answers for the calls a test makes.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
from analysis_mcp.server import mcp as analysis_mcp
from compiler.surface import McpSurface, ToolSchema
from data_mcp.digest import render_answer
from data_mcp.search_result import SEARCH_ENVELOPE_META_KEY
from data_mcp.server import mcp as data_mcp
from fastmcp import FastMCP
from fastmcp.tools import ToolResult
from planner.plan import PLAN_STATE_KEY, Decision, EvidenceRequirement, Plan, PlanStatus

SPEC_KENYA = "spec_" + "a" * 32
SPEC_OTHER = "spec_" + "b" * 32
SPEC_THIRD = "spec_" + "c" * 32

# A phrase in the variable asked about, and the warning `search` attaches to
# its answer when it sees one. The first two are the codes under which part
# of the ask was dropped; the third is advisory.
WARNING_PHRASES: dict[str, str] = {
    "dropped role": "role_text_not_recognised",
    "dropped time": "time_text_not_parsed",
    "undeclared default": "default_not_declared",
}


def _spec(spec_id: str, sentence: str, **resolved: Any) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "rendered_sentence": sentence,
        "confidence": "high",
        "coverage": {
            "kind": "exact",
            "has_data": True,
            "window": {"start_year": 2010, "end_year": 2024},
        },
        "stat_vars": {"member_count": 1, "member_count_exact": True, "served_count": 1},
        "entities": [{"role": "recipient", "label": "Kenya", "ref": "country/KEN"}],
        "facets": {"unit": {"basis": "declared_default", "ref": "usd", "label": "US dollars"}},
    }
    return {
        "spec_id": spec_id,
        "resolved": fields | resolved,
        "definition": {
            "schema_version": 1,
            "spec_id": spec_id,
            "identity": {"shape": {"kind": "stat_var", "id": "DebtServicePaid"}},
        },
    }


def _envelope(status: str, warnings: list[dict[str, str]], **payload: Any) -> dict[str, Any]:
    """A single-part envelope: one entry in `parts`, carrying its own status and warnings."""
    part = {"status": status, "variable_text": "asked", "warnings": warnings, **payload}
    return {"schema_version": 1, "status": status, "parts": [part]}


def _warnings_for(variable: str) -> list[dict[str, str]]:
    """The part warnings the phrases in `variable` ask for."""
    return [
        {"code": code, "severity": "warning", "message": f"{phrase} in {variable!r}"}
        for phrase, code in WARNING_PHRASES.items()
        if phrase in variable
    ]


@pytest.fixture
def search_calls() -> list[dict[str, Any]]:
    """Every argument set `data_server`'s `search` received, in order."""
    return []


@pytest.fixture
def data_server(search_calls: list[dict[str, Any]]) -> FastMCP:
    """A `data-mcp` stand-in whose `search` answers by the variable asked about."""
    server = FastMCP(name="data-mcp")

    @server.tool(name="search")
    async def search(parts: list[dict[str, Any]], raw_query: str | None = None) -> ToolResult:
        """Find what data exists for the ask."""
        search_calls.append({"parts": parts, "raw_query": raw_query})
        (part,) = parts
        variable = str(part["variable"])
        if variable == "refuse":
            return ToolResult(
                content="refused",
                structured_content={"code": "request_invalid", "retry_mode": "correct_and_retry"},
                is_error=True,
            )
        warnings = _warnings_for(variable)
        if variable == "garbage":
            envelope = _envelope("partial", warnings)
        elif "unsupported" in variable:
            envelope = _envelope(
                "unsupported", warnings, unsupported={"detail": "cannot serve this ask"}
            )
        elif "unicorn" in variable:
            # A `no_data` answer for a reason a rewording cannot fix: the
            # entity itself never resolved, so no nearest series is offered
            # either.
            envelope = _envelope("no_data", warnings, no_data={"reason": "entity_not_resolved"})
        elif "nearly" in variable:
            # No data at the resolved entity, but a real series nearby: the
            # one place a `no_data` answer names a spec, for the referral to
            # carry as its nearest series.
            envelope = _envelope(
                "no_data",
                warnings,
                no_data={
                    "reason": "no_observations",
                    "nearest_real": [_spec(SPEC_OTHER, "total debt service paid by Kenya")],
                },
            )
        elif variable.startswith("aid"):
            envelope = _envelope(
                "candidates",
                warnings,
                candidates={
                    "specs": [
                        _spec(SPEC_OTHER, "aid from France"),
                        _spec(SPEC_THIRD, "aid to Togo"),
                    ],
                    "truncated": False,
                },
            )
        else:
            envelope = _envelope(
                "definite", warnings, spec=_spec(SPEC_KENYA, "debt service paid by Kenya")
            )
        return ToolResult(
            content=f"digest for {variable}",
            structured_content=render_answer(envelope),
            meta={SEARCH_ENVELOPE_META_KEY: envelope},
        )

    @server.tool(name="describe")
    async def describe(ids: list[str]) -> ToolResult:
        """Describe graph identifiers; refuses every id "refuse" among them, naming it in `fix`.

        Mirrors `data_mcp.describe_tool.describe`'s own shape for a mixed
        batch: any "refuse" id in the call makes the whole `ToolResult`
        `is_error=True`, with `fix.dcids` naming every such id -- not only
        the ones actually held are described in a retry-usable form.
        """
        unheld = [dcid for dcid in ids if dcid == "refuse"]
        if unheld:
            return ToolResult(
                content="refused",
                structured_content={
                    "code": "node_not_held",
                    "retry_mode": "correct_and_retry",
                    "fix": {"dcids": unheld},
                },
                is_error=True,
            )
        nodes = [
            {
                "shape": "stat_var",
                "dcid": dcid,
                "label": dcid,
                "definition": "d",
                "roles": ["recipient"],
            }
            for dcid in ids
        ]
        return ToolResult(content="nodes", structured_content={"nodes": nodes})

    @server.tool(name="get_data")
    async def get_data(spec_id: str, scope: dict[str, Any] | None = None) -> str:
        """Retrieve observations. The translator must never reach this."""
        return "observations"

    return server


@pytest.fixture
def surface(data_server: FastMCP) -> McpSurface:
    """The canned data server, reached over the MCP protocol in-process."""
    return McpSurface(sources={"data-mcp": data_server})


@pytest.fixture
async def real_schemas() -> list[ToolSchema]:
    """The real servers' tool schemas, which the capability table is pinned to."""
    return await McpSurface(
        sources={"data-mcp": data_mcp, "analysis-mcp": analysis_mcp}
    ).list_tools()


@pytest.fixture
def draft(
    decision: Callable[..., Decision],
    requirements: tuple[EvidenceRequirement, ...],
) -> Callable[..., Plan]:
    """Return a factory for a draft plan with one decision and the `requirements` pair.

    `requirements` is the shared, minimal, valid pair: a dataset requirement
    and the aggregate that reads it. A test wanting different requirements
    passes its own `evidence_requirements`, built from the `requirement` and
    `derived_requirement` factories.
    """

    def _draft(**overrides: Any) -> Plan:
        fields: dict[str, Any] = {
            "question": "How much does Kenya pay to service its debt each year?",
            "decisions": [decision()],
            "evidence_requirements": list(requirements),
            "status": PlanStatus.DRAFT,
        }
        return Plan(**(fields | overrides))

    return _draft


class FakeToolContext:
    """A minimal stand-in for ADK's `ToolContext`: only `.state` is read."""

    def __init__(self) -> None:
        self.state: dict[str, str] = {}


@pytest.fixture
def empty_context() -> FakeToolContext:
    """A tool context holding no plan."""
    return FakeToolContext()


@pytest.fixture
def context_holding() -> Callable[[Plan], FakeToolContext]:
    """Return a factory for a tool context holding `plan` in session state."""

    def _holding(plan: Plan) -> FakeToolContext:
        context = FakeToolContext()
        context.state[PLAN_STATE_KEY] = plan.model_dump_json()
        return context

    return _holding
