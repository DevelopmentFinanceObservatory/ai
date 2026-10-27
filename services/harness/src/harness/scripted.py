"""Deterministic stand-ins for the studio's seams, for tests and the scripted eval.

Each answers from fixed data with no LLM, no MCP call and no upstream:
`ScriptedExecutor` (the agent's and the canvas's tool calls),
`ScriptedPlannerModel` (the planner), `ScriptedSurface` (an approved
plan's steps) and `scripted_publish_result` (the completion gate's
publisher). Every frame is the one Kenyan life-expectancy series.
`use_compiled_proposals` gives a real planner a `propose_plan` that declares
the plan already on record ready, for a scripted planner model that drives no
translation.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from common.materialisation import FrameRef
from common.mcp_telemetry import WRITE_TOKEN_KEY
from compiler.surface import (
    ProgressCallback,
    ProgressReport,
    ToolCallResult,
    ToolSchema,
    ToolSurfaceUnavailableError,
    deliver_progress,
)
from executor.completion import FramePublished
from google.adk import Context
from google.adk.agents import LlmAgent
from google.adk.tools.function_tool import FunctionTool
from google.adk.tools.tool_context import ToolContext
from planner.agent import declare_ready
from planner.plan import load_plan, plan_revision
from planner.tool_results import ok
from planner.workflow import ToolId

from harness.scripted_plan import (
    SCRIPTED_TOOL_SCHEMAS,
    scripted_aggregate_result,
    scripted_manifest,
    scripted_plan,
)

SCRIPTED_SPEC = "spec_" + "f" * 32
SCRIPTED_CV = "cv_" + "f" * 32
_FRAME_REF = {"spec_id": SCRIPTED_SPEC, "content_version": SCRIPTED_CV}

# The one-series fact summary a Kenyan card carries.
_FACT_SUMMARY = {
    "component_id": "cmp_scripted0001",
    "revision": 1,
    "series": [
        {
            "id": "ken",
            "label": "Kenya",
            "point_count": 64,
            "coordinates": {
                "unit": "Year",
                "stat_var": "dcid:LifeExpectancy_Person",
                "observationAbout": "country/KEN",
            },
            "first": 48.5,
            "last": 63.6,
        }
    ],
    "attribution": {
        "label": "WorldDevelopmentIndicators · 1 indicator(s)",
        "source_count": 1,
        "indicator_count": 1,
    },
    "window": {"start": "1960", "end": "2023"},
}


_PROVENANCE = {
    "inputs": [_FRAME_REF],
    "sources": [
        {
            "id": "dc/base/WorldDevelopmentIndicators",
            "name": "World Development Indicators",
            "licence": "CC BY 4.0",
        }
    ],
    "freshness": {
        "data_through": "2023",
        "last_refreshed": "2026-06-30",
        "partial_period": False,
    },
    "warnings": [],
    "generated_at": "2026-01-01T00:00:00Z",
    "citation": "World Development Indicators (2026).",
}


# The envelope's own provenance block, in the platform's resolved-revision
# shape (`inputs` is set per card).
_ENVELOPE_SOURCE_REF = "src_wdi"
_ENVELOPE_PROVENANCE = {
    "execution_ref": None,
    "sources": [
        {
            "id": _ENVELOPE_SOURCE_REF,
            "name": "World Development Indicators",
            "url": "https://databank.worldbank.org/source/world-development-indicators",
            "licence": "CC BY 4.0",
            "retrieved": "2026-06-30",
        }
    ],
    "freshness": {
        "data_through": "2023",
        "last_refreshed": "2026-06-30",
        "next_expected": None,
        "partial_period": False,
    },
    "warnings": [],
    "generated_at": "2026-01-01T00:00:00Z",
    "citation": "World Development Indicators (2026).",
    "refreshed_from": None,
}
_UNIT = "Year"
_Y_LABEL = "Life expectancy"


async def scripted_publish_result(frame: FrameRef) -> FramePublished:
    """Publish every frame at once, under a replay key read off its content
    version: there is no apis to publish to."""
    return FramePublished(replay_key=f"rpk_{frame.content_version.removeprefix('ocv_')}")


def _points(first: float, last: float, *, start: int, end: int) -> list[dict[str, Any]]:
    """One point a year from `start` to `end`, on a straight line from
    `first` to `last`, matching the fact summary's `point_count`."""
    span = end - start
    return [
        {"period": str(start + step), "value": round(first + (last - first) * step / span, 1)}
        for step in range(span + 1)
    ]


def _resolved_component(*, title: str, fact_summary: dict[str, Any]) -> dict[str, Any]:
    """The envelope's `component` for a scripted card, a line chart drawn
    from `fact_summary`'s series and window, in the shape of dfo-spec-kit's
    catalogue fixtures."""
    window = fact_summary["window"]
    series: list[dict[str, Any]] = [
        {
            "id": entry["id"],
            "label": entry["label"],
            "unit": _UNIT,
            "source_ref": _ENVELOPE_SOURCE_REF,
            "points": _points(
                entry["first"], entry["last"], start=int(window["start"]), end=int(window["end"])
            ),
        }
        for entry in fact_summary["series"]
    ]
    return {
        "type": "line_chart",
        "title": {"text": title, "authorship": "model"},
        "series": series,
        "x_axis": {"window": {"start": window["start"], "end": window["end"]}},
        "y_axis": {"label": _Y_LABEL, "scale": "linear", "range": {"min": None, "max": None}},
    }


def _component(
    component_id: str, *, title: str, frame_ref: dict[str, Any] | None
) -> dict[str, Any]:
    """A scripted `component.created` body: the resolved revision in
    `payload`, beside the harness's own fact summary and provenance."""
    resolved_ref = frame_ref or _FRAME_REF
    fact_summary = _FACT_SUMMARY
    catalog_version = "2026.8.0"
    return {
        "component_id": component_id,
        "revision": 1,
        "type": "line_chart",
        "catalog_version": catalog_version,
        "min_catalog_version": catalog_version,
        "payload": {
            "schema_version": 1,
            "catalog_version": catalog_version,
            "component_id": component_id,
            "revision": 1,
            "provenance": {"inputs": [resolved_ref], **_ENVELOPE_PROVENANCE},
            "component": _resolved_component(title=title, fact_summary=fact_summary),
        },
        "fact_summary": {**fact_summary, "component_id": component_id, "revision": 1},
        "provenance": {**_PROVENANCE, "inputs": [resolved_ref]},
    }


def _scripted_view() -> dict[str, Any]:
    """The describe_frame result for a scripted frame: one series."""
    return {
        "outcome": "ok",
        "frame_refs": [],
        "view": {
            "dimensions": [{"name": "stat_var", "values": ["dcid:LifeExpectancy_Person"]}],
            "period_grain": "year",
            "coverage": ["1960", "2023"],
            "series_count": 1,
        },
    }


class ScriptedExecutor:
    """Fixed results for `get_data`, `describe_frame` and a new card's
    `write_component`. `get_data` carries no component of its own: a card
    comes only from `write_component`, and every card is a line chart."""

    async def __call__(
        self, *, tool: str, args: dict[str, Any], on_progress: ProgressCallback | None = None
    ) -> dict[str, Any]:
        del on_progress  # a fixed result reports no progress
        if tool == "get_data":
            return {
                "outcome": "ok",
                "frame_refs": [
                    {**_FRAME_REF, "spec_id": str(args.get("spec_id") or SCRIPTED_SPEC)}
                ],
                "summary": "64 observations · 1960 to 2023 · WorldDevelopmentIndicators",
            }
        if tool == "describe_frame":
            return _scripted_view()
        if tool == "write_component" and "component_id" not in args:
            intent = args.get("intent") or {}
            return {
                "outcome": "ok",
                "frame_refs": [],
                "component": _component(
                    "cmp_scripted0002",
                    title=str(intent.get("title") or "Scripted card"),
                    frame_ref=intent.get("frame_ref"),
                ),
                "summary": "created",
                # The grant a later Edit Card patch revises this card with
                # (`common.mcp_telemetry.WRITE_TOKEN_KEY`).
                WRITE_TOKEN_KEY: "cmw_scripted0002",
            }
        raise ValueError(f"no scripted answer for {tool!r}")


def use_compiled_proposals(agent: LlmAgent) -> None:
    """Replace `agent`'s `propose_plan` with one that records nothing and
    answers as a proposal whose translation compiled the plan on record.

    The answer goes through `planner.agent.declare_ready`, so a plan on
    record that is `PENDING_APPROVAL` is declared ready and ends the run, as
    a compiled proposal does, and any other is refused under `submission`.
    """

    def propose_plan(tool_context: ToolContext) -> dict[str, object]:
        plan = load_plan(tool_context.state)
        assert plan is not None, "use_compiled_proposals needs a plan on record"
        answer = ok(
            status=plan.status.value,
            version=plan.version,
            previous_version=plan.previous_version,
            question=plan.question,
            translation=ok(),
        )
        return declare_ready(answer, tool_context)

    agent.tools = [
        FunctionTool(propose_plan) if getattr(tool, "name", None) == "propose_plan" else tool
        for tool in agent.tools
    ]


class ScriptedPlannerModel:
    """A `PlannerModel` seam with no model.

    Drafts the canned plan `harness.scripted_plan.scripted_plan` builds. A
    re-entry -- a replan after a refusal, or a user amendment -- redrafts the
    same steps as the next version rather than reasoning about which one it
    is, since there is no model to reason with.
    """

    async def __call__(
        self,
        *,
        ctx: Context,
        question: str,
        readings: Sequence[Mapping[str, Any]],
        prior_plan: dict[str, Any] | None = None,
        refusal: dict[str, Any] | None = None,
        amendment: str | None = None,
        canvas_digest: dict[str, Any] | None = None,
        conversation_digest: dict[str, Any] | None = None,
        on_progress: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        # Nothing to report and nothing to run through the workflow's
        # session: this seam has no model round trips to relay and no
        # agent to host via `ctx.run_node`. It drafts the same canned plan
        # whatever the readings name.
        del ctx, readings, refusal, amendment, canvas_digest, conversation_digest, on_progress
        plan = scripted_plan(question, spec_id=None)
        if prior_plan is not None:
            prior_version = int(prior_plan.get("version", 1))
            plan = plan.model_copy(
                update={"version": prior_version + 1, "previous_version": prior_version}
            )
        return {
            "plan": plan.model_dump(mode="json"),
            "summary": {"version": plan.version, "revision": plan_revision(plan)},
        }

    async def revise(
        self,
        *,
        ctx: Context,
        prior_plan: dict[str, Any],
        evidence_requirements: Sequence[Mapping[str, Any]],
        inherits: Mapping[str, str],
    ) -> dict[str, Any]:
        """Refuse: the canned plan pins no source, so no switch reaches it."""
        del ctx, prior_plan, evidence_requirements, inherits
        return {"refused": {"text": "The scripted planner revises no plan.", "code": None}}


class ScriptedSurface:
    """A `ToolSurface` over `harness.scripted_plan`'s own declared schemas.

    `list_tools()` answers `SCRIPTED_TOOL_SCHEMAS` verbatim -- the same
    objects `scripted_plan`'s workflow is compiled against -- so a canned
    plan's `input_schema_hash`/`output_schema_hash`/`implementation_version`
    pins always match what this surface declares. `call_tool` answers by
    `(tool.name, arguments)`: a fixed manifest for `get_data`, and for
    `aggregate` the fixed result `scripted_plan.scripted_aggregate_result`
    states. Any other tool is unavailable.

    `get_data_progress` holds the reports a `get_data` call hands its
    caller's `on_progress`, in order, before it answers, standing in for
    data-mcp's reports while it waits on a background operation. It is
    empty unless a caller scripts it. A raising callback fails nothing
    (`compiler.surface.deliver_progress`).
    """

    def __init__(self, *, get_data_progress: Sequence[ProgressReport] = ()) -> None:
        self._get_data_progress = tuple(get_data_progress)

    async def list_tools(self) -> list[ToolSchema]:
        return list(SCRIPTED_TOOL_SCHEMAS)

    async def call_tool(
        self,
        *,
        tool: ToolId,
        arguments: Mapping[str, Any],
        on_progress: ProgressCallback | None = None,
    ) -> ToolCallResult:
        if tool.name == "get_data":
            if on_progress is not None:
                for report in self._get_data_progress:
                    deliver_progress(on_progress, report)
            spec_id = str(arguments.get("spec_id") or "")
            return ToolCallResult(
                is_error=False,
                structured_content=scripted_manifest(spec_id),
                text="scripted get_data result",
            )
        if tool.name == "aggregate":
            return ToolCallResult(
                is_error=False,
                structured_content=scripted_aggregate_result(),
                text="scripted aggregate result",
            )
        raise ToolSurfaceUnavailableError(f"no scripted answer for {tool.name!r}")
