"""Behavioral tests for the minimal root workflow graph.

Each test pins one property the graph's topology claims: status routing,
the bounded clarify loop, depth gating, version-bound approval, executor
retry-mode branching, and verbatim frame-ref handoff. Everything runs
offline against injected scripted fakes -- no model, no MCP server.
"""

from __future__ import annotations

import asyncio
import copy
import json
import logging
import re
import threading
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from pathlib import Path
from typing import Any, cast

import httpx2
import pytest
from analysis_mcp.server import mcp as analysis_mcp
from compiler.surface import (
    FixtureSurface,
    McpSurface,
    ProgressCallback,
    ProgressReport,
    ToolCallResult,
    ToolSchema,
    ToolSurface,
)
from data_mcp import client_holder
from data_mcp.client import UpstreamClient
from data_mcp.definition_store import (
    DefinitionStoreUnavailable,
    InMemoryDefinitionStore,
    canonicalize,
    spec_id_for,
)
from data_mcp.definition_store import set_store as set_definition_store
from data_mcp.server import mcp as data_mcp
from data_mcp.settings import Settings as DataMcpSettings
from executor.classify import ToolFailure
from executor.outcomes import (
    Accepted,
    MethodChoiceNeeded,
    RediscoveryNeeded,
    RefusalKind,
    Refused,
)
from executor.run import StepProgressCallback, current_step_id, execute_approved_plan
from fastmcp import Client
from google.adk.events import Event
from google.adk.runners import InMemoryRunner
from google.adk.workflow._errors import NodeInterruptedError
from google.adk.workflow.utils._workflow_hitl_utils import (
    create_request_input_response,
    get_request_input_interrupt_ids,
)
from google.genai import types
from harness import graph as graph_module
from harness.depth import decide_depth
from harness.graph import (
    APPROVE_INTERRUPT,
    CLARIFY_INTERRUPT,
    FETCH_STEP_ID,
    LINE_CHART_MAX_VALUES,
    MAX_CLARIFY_ROUNDS,
    PLANNER_STARTED_KEY,
    REFUSED_WRITE_VOCABULARY,
    STEP_PROGRESS_KEY,
    GenerativeNodes,
    GraphState,
    _drop_seeds_from_turn,
    _failed_step,
    _frame_unavailable_reason,
    _mark_masked_step_failed,
    _refused_write_reason,
    _replan_refusal,
    _select_binding_ok,
    _start_relayed,
    create_root_workflow,
)
from harness.mcp_client import _refusal_outcome
from harness.scripted import ScriptedPlannerModel, ScriptedSurface
from harness.scripted_plan import scripted_fetch_series_stats
from harness.seeds import (
    NO_OPERATIONS_ON_DOWNLOAD,
    ONE_DATASET_PER_DOWNLOAD,
    SEED_DROPPED,
    SEED_DUPLICATE,
    SEED_REFERENCE_UNUSED,
    SEED_STEP_TOOL,
    SEEDS_STEP_ID,
    DroppedSeed,
    TakenSeed,
    dropped_seeds_note,
    resolve_seed,
    seeded_envelope,
    stale_reason,
    stale_seeds,
    unplanned_seeds_text,
)
from harness.tool_contracts import graph_envelope
from harness.wire import Seed
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from planner.operations import AggregateOperation, GrowthOperation, RetrieveOperation, Translation
from planner.plan import (
    SEEDS_STATE_KEY,
    AggregateRequirement,
    DatasetRequirement,
    Decision,
    Entity,
    GrowthRequirement,
    Period,
    PlanStatus,
    ResolvedBy,
    SeededSource,
    dropped_seeds,
    plan_revision,
    seeded_sources,
)
from planner.plan import Plan as PlannerPlan
from planner.records import EvidenceIssue, EvidenceIssueCode, GroupBy, Reduction, Severity
from planner.workflow import EvidenceBinding, SelectedSource, ToolId, Workflow, WorkflowStep
from pydantic import ValidationError
from translator.discovery import PartFault, PartNotResolvedError

from .data_mcp_definitions import DataMcpDefinitions

SPEC_A = "spec_" + "a" * 32
SPEC_B = "spec_" + "b" * 32
REF_A = {"spec_id": SPEC_A, "content_version": "cv_" + "1" * 32}
REF_B = {"spec_id": SPEC_B, "content_version": "cv_" + "2" * 32}


def _get_data_manifest(*, spec_id: str = SPEC_A) -> dict[str, Any]:
    """A minimal valid `get_data` manifest. The hand-built workflows below
    carry no `evidence_bindings`, so only its shape (a definition manifest and
    the `frame_ref` beside it, which `extract_frame_ref` reads) matters."""
    return {
        "schema_version": 1,
        "definition": {"schema_version": 1, "spec_id": spec_id, "identity": {}},
        "content_version": "cv_" + "1" * 32,
        "coverage": {"rows": 10, "dimensions": []},
        "citation": {"text": "Test source", "sources": []},
        "trace": {},
        "artifacts": [],
        "frame_ref": {"spec_id": spec_id, "content_version": "cv_" + "1" * 32},
    }


def _analysis_result(*, spec_id: str = "ops_" + "2" * 32) -> dict[str, Any]:
    """A minimal analysis-mcp result: one row of one StatVar, matching a
    `WorkflowStep`'s default `expected_role_keys=()`."""
    return {
        "spec_id": spec_id,
        "content_version": "ocv_" + "3" * 32,
        "rows": 1,
        "role_keys": [],
        "stat_var_count": 1,
    }


@pytest.fixture
async def real_schemas() -> list[ToolSchema]:
    """The real servers' tool schemas. A `WorkflowStep` hashed against one of
    these passes `execute_approved_plan`'s own surface check."""
    return await McpSurface(
        sources={"data-mcp": data_mcp, "analysis-mcp": analysis_mcp}
    ).list_tools()


class _CallRecordingSurface:
    """Wraps a `ToolSurface`, recording every call -- `FixtureSurface` itself does not."""

    def __init__(self, surface: ToolSurface) -> None:
        self._surface = surface
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def list_tools(self) -> list[ToolSchema]:
        return await self._surface.list_tools()

    async def call_tool(
        self,
        *,
        tool: ToolId,
        arguments: Mapping[str, Any],
        on_progress: ProgressCallback | None = None,
    ) -> ToolCallResult:
        self.calls.append((tool.name, dict(arguments)))
        return await self._surface.call_tool(
            tool=tool, arguments=arguments, on_progress=on_progress
        )


def _fixture_surface(
    schemas: list[ToolSchema],
    *,
    analysis_tool: str = "aggregate",
    analysis_result: dict[str, Any] | None = None,
    get_data_result: ToolCallResult | None = None,
    analysis_call_result: ToolCallResult | None = None,
) -> FixtureSurface:
    """A `ToolSurface` over the real schemas, answering `get_data` and one
    analysis tool. A caller wanting a refusal from either passes its own
    `ToolCallResult`."""
    named = {schema.name: schema for schema in schemas}
    return FixtureSurface(
        schemas=[named["get_data"], named[analysis_tool]],
        responses={
            "get_data": get_data_result
            or ToolCallResult(
                is_error=False, structured_content=_get_data_manifest(), text="fetched"
            ),
            analysis_tool: analysis_call_result
            or ToolCallResult(
                is_error=False,
                structured_content=analysis_result or _analysis_result(),
                text="computed",
            ),
        },
    )


def _envelope(**overrides: Any) -> dict[str, Any]:
    envelope: dict[str, Any] = {
        "status": "definite",
        "spec_id": SPEC_A,
        "choice_points": 0,
        "parts": 1,
        "warnings": [],
        "candidates": [],
    }
    envelope.update(overrides)
    return envelope


class ScriptedResolver:
    """Returns the scripted envelopes in order, then repeats the last.

    `derive_reentry_parts`, off by default, mirrors `McpResolver.__call__`'s
    `spec_id`-set shortcut: on a call where `spec_id` is set, it replaces
    the scripted envelope's `parts` with `len(parts) if parts else 1`
    instead of the value the envelope was built with, so a test using it
    proves the graph actually passed the search parts through on the
    clarify re-entry call rather than asserting on a hardcoded envelope
    field."""

    def __init__(
        self, envelopes: list[dict[str, Any]], *, derive_reentry_parts: bool = False
    ) -> None:
        self.envelopes = envelopes
        self.calls: list[dict[str, Any]] = []
        self._derive_reentry_parts = derive_reentry_parts

    async def __call__(
        self,
        *,
        question: str,
        spec_id: str | None = None,
        wide: bool = False,
        parts: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        self.calls.append({"question": question, "spec_id": spec_id, "wide": wide, "parts": parts})
        index = min(len(self.calls) - 1, len(self.envelopes) - 1)
        envelope = self.envelopes[index]
        if self._derive_reentry_parts and spec_id is not None:
            envelope = {**envelope, "parts": len(parts) if parts else 1}
        return envelope


class ScriptedExecutor:
    """Returns the scripted step results in order, then keeps returning ok."""

    def __init__(self, results: list[dict[str, Any]] | None = None) -> None:
        self.results = results or []
        self.calls: list[dict[str, Any]] = []

    async def __call__(
        self, *, tool: str, args: dict[str, Any], on_progress: ProgressCallback | None = None
    ) -> dict[str, Any]:
        self.calls.append({"tool": tool, "args": args})
        if len(self.calls) <= len(self.results):
            return self.results[len(self.calls) - 1]
        return {"outcome": "ok", "frame_refs": [REF_A]}


def _node_name(event: Event) -> str:
    path = event.node_info.path if event.node_info else ""
    return path.split("/")[-1].split("@")[0]


async def _start(workflow: Any, question: str) -> tuple[InMemoryRunner, str, list[Event]]:
    runner = InMemoryRunner(node=workflow)
    session = await runner.session_service.create_session(app_name=runner.app_name, user_id="u")
    events = [
        event
        async for event in runner.run_async(
            user_id="u",
            session_id=session.id,
            new_message=types.Content(role="user", parts=[types.Part.from_text(text=question)]),
        )
    ]
    return runner, session.id, events


async def _resume(
    runner: InMemoryRunner, session_id: str, interrupt_id: str, response: dict[str, Any]
) -> list[Event]:
    part = create_request_input_response(interrupt_id, {"result": response})
    return [
        event
        async for event in runner.run_async(
            user_id="u",
            session_id=session_id,
            new_message=types.Content(role="user", parts=[part]),
        )
    ]


def _pending_interrupt(events: list[Event]) -> str | None:
    ids = get_request_input_interrupt_ids(events[-1])
    return ids[0] if ids else None


def _output_of(events: list[Event], node: str) -> Any:
    outputs = [e.output for e in events if _node_name(e) == node and e.output is not None]
    return outputs[-1] if outputs else None


def _visited(events: list[Event]) -> list[str]:
    return [_node_name(e) for e in events]


# --- 1. the pure depth policy ------------------------------------------------


@pytest.mark.parametrize(
    ("status", "choice_points", "degraded", "part_count", "surface", "expected"),
    [
        ("definite", 0, False, 1, "studio", "none"),
        ("definite", 5, True, 3, "download", "none"),
        ("definite", 0, False, 2, "studio", "standard"),
        ("definite", 1, False, 1, "studio", "standard"),
        ("definite", 2, False, 1, "studio", "standard"),
        ("definite", 3, False, 1, "studio", "deep"),
        ("definite", 0, True, 1, "studio", "deep"),
        ("candidates", 0, False, 1, "studio", "shallow"),
    ],
)
def test_decide_depth_table(
    status: str,
    choice_points: int,
    degraded: bool,
    part_count: int,
    surface: str,
    expected: str,
) -> None:
    signal = decide_depth(
        status=status,
        choice_points=choice_points,
        degraded=degraded,
        part_count=part_count,
        surface=surface,
    )
    assert signal.depth == expected
    assert signal.reason


def test_decide_depth_none_grants_no_questions() -> None:
    signal = decide_depth(
        status="definite", choice_points=0, degraded=False, part_count=1, surface="studio"
    )
    assert signal.max_clarifying_questions == 0
    assert signal.may_use_defaults is True


def test_decide_depth_deep_forbids_defaults() -> None:
    signal = decide_depth(
        status="definite", choice_points=3, degraded=False, part_count=1, surface="studio"
    )
    assert signal.may_use_defaults is False


@pytest.mark.parametrize(
    ("hint", "status", "choice_points", "degraded", "part_count", "surface", "expected"),
    [
        # thorough raises one level, floored at standard
        ("thorough", "definite", 0, False, 1, "studio", "standard"),
        ("thorough", "candidates", 0, False, 1, "studio", "standard"),
        ("thorough", "definite", 1, False, 1, "studio", "deep"),
        ("thorough", "definite", 3, False, 1, "studio", "deep"),
        # quick lowers one level...
        ("quick", "candidates", 0, False, 1, "studio", "none"),
        ("quick", "definite", 0, False, 1, "studio", "none"),
        # ...but choice points keep the planner (floor shallow)
        ("quick", "definite", 1, False, 1, "studio", "shallow"),
        ("quick", "definite", 3, False, 1, "studio", "standard"),
        # ...and degraded floors at standard
        ("quick", "definite", 0, True, 1, "studio", "standard"),
        # the download surface ignores hints entirely
        ("thorough", "definite", 0, False, 1, "download", "none"),
    ],
)
def test_decide_depth_hint_table(
    hint: str,
    status: str,
    choice_points: int,
    degraded: bool,
    part_count: int,
    surface: str,
    expected: str,
) -> None:
    signal = decide_depth(
        status=status,
        choice_points=choice_points,
        degraded=degraded,
        part_count=part_count,
        surface=surface,
        depth_hint=hint,  # type: ignore[arg-type]
    )
    assert signal.depth == expected


def test_hint_reason_records_structural_verdict_and_hint_effect() -> None:
    signal = decide_depth(
        status="definite",
        choice_points=0,
        degraded=False,
        part_count=1,
        surface="studio",
        depth_hint="thorough",
    )
    assert "definite single-part resolution" in signal.reason
    assert "user asked for thorough" in signal.reason
    assert "none->standard" in signal.reason


def test_quick_keeps_clarifying_questions_available_when_forks_exist() -> None:
    """Quick depth shrinks the budget; clarifying questions stay available."""
    signal = decide_depth(
        status="definite",
        choice_points=2,
        degraded=False,
        part_count=1,
        surface="studio",
        depth_hint="quick",
    )
    assert signal.depth == "shallow"
    assert signal.max_clarifying_questions == 1
    assert signal.may_use_defaults is True


def test_a_wide_identity_never_gets_none_even_with_no_other_choice_points() -> None:
    """A described-arm family with every slot unbound: no facet was
    defaulted (`choice_points == 0`), but the identity itself names many
    members, so the verdict floors at `shallow` rather than `depth="none"`,
    which would skip the planner and the approval gate both."""
    signal = decide_depth(
        status="definite",
        choice_points=0,
        degraded=False,
        part_count=1,
        surface="studio",
        wide=True,
    )
    assert signal.depth == "shallow"
    assert signal.max_clarifying_questions >= 1


def test_quick_does_not_undo_the_wide_floor() -> None:
    """The same guardrail `quick` already gives choice points applies to a
    wide identity too: `quick` keeps its `shallow` verdict rather than
    pushing it down to `none`."""
    signal = decide_depth(
        status="definite",
        choice_points=0,
        degraded=False,
        part_count=1,
        surface="studio",
        depth_hint="quick",
        wide=True,
    )
    assert signal.depth == "shallow"


def test_a_narrow_identity_with_no_choice_points_still_gets_none() -> None:
    """`wide=False` (the default) leaves the fast path untouched."""
    signal = decide_depth(
        status="definite",
        choice_points=0,
        degraded=False,
        part_count=1,
        surface="studio",
    )
    assert signal.depth == "none"


def test_download_surface_is_exempt_from_the_wide_floor() -> None:
    """The wide floor is a `studio`-and-friends guardrail: `download` asks
    to fetch what matches, so a wide identity there still gets
    `depth="none"`, while the same wide identity on a non-download surface
    floors at `shallow`."""
    downloaded = decide_depth(
        status="definite",
        choice_points=0,
        degraded=False,
        part_count=1,
        surface="download",
        wide=True,
    )
    assert downloaded.depth == "none"

    studio = decide_depth(
        status="definite",
        choice_points=0,
        degraded=False,
        part_count=1,
        surface="studio",
        wide=True,
    )
    assert studio.depth == "shallow"


def test_decide_depth_notice_is_none_when_no_plan_is_shown() -> None:
    signal = decide_depth(
        status="definite", choice_points=0, degraded=False, part_count=1, surface="studio"
    )
    assert signal.depth == "none"
    assert signal.notice is None


def test_decide_depth_compound_notice_names_the_multiple_asks() -> None:
    signal = decide_depth(
        status="definite", choice_points=0, degraded=False, part_count=2, surface="studio"
    )
    assert signal.depth == "standard"
    assert signal.notice == (
        "This asks for more than one thing, so I am proposing a plan before fetching anything."
    )


_TWO_COUNTRY_ENVELOPE = (
    Path(__file__).parent / "fixtures" / "resolve_life_expectancy_two_countries.json"
)


def test_one_part_two_entities_routes_through_graph_envelope_to_depth_none() -> None:
    """The captured one-part, two-entity search envelope keeps the fast
    path. `graph_envelope` reads one part and a narrow `observationAbout`
    identity off it, and `decide_depth` takes the resulting part count and
    width to `depth="none"`: no plan card, one `get_data`, one spec behind
    both countries."""
    envelope = json.loads(_TWO_COUNTRY_ENVELOPE.read_text())
    assert envelope["parts"][0]["spec"]["definition"]["identity"]["entities"] == {
        "observationAbout": ["country/GTM", "country/NGA"]
    }
    digest = graph_envelope(envelope)
    assert digest["status"] == "definite"
    assert digest["parts"] == 1
    assert digest["wide"] is False
    signal = decide_depth(
        status=digest["status"],
        choice_points=digest["choice_points"],
        degraded=False,
        part_count=digest["parts"],
        surface="studio",
        wide=digest["wide"],
    )
    assert signal.depth == "none"
    assert signal.notice is None


def test_decide_depth_choice_point_notice_names_the_real_choice() -> None:
    signal = decide_depth(
        status="definite", choice_points=1, degraded=False, part_count=1, surface="studio"
    )
    assert signal.depth == "standard"
    assert signal.notice == (
        "This question has a real choice to settle, so here is the plan first."
    )


def test_decide_depth_degraded_notice_names_the_data_quality_issue() -> None:
    signal = decide_depth(
        status="definite", choice_points=0, degraded=True, part_count=1, surface="studio"
    )
    assert signal.depth == "deep"
    assert "quality" in signal.notice


def test_decide_depth_wide_identity_notice_names_the_family() -> None:
    signal = decide_depth(
        status="definite",
        choice_points=0,
        degraded=False,
        part_count=1,
        surface="studio",
        wide=True,
    )
    assert signal.depth == "shallow"
    assert "family" in signal.notice


def test_decide_depth_thorough_hint_notice_names_the_users_request() -> None:
    # A clean, definite lookup needs no plan on structural grounds alone: the
    # hint is the entire reason one appears, so its own notice is due.
    signal = decide_depth(
        status="definite",
        choice_points=0,
        degraded=False,
        part_count=1,
        surface="studio",
        depth_hint="thorough",
    )
    assert signal.depth == "standard"
    assert "thorough" in signal.notice


def test_decide_depth_thorough_hint_on_a_wide_resolution_keeps_the_wide_notice() -> None:
    # A wide identity already needs a plan on structural grounds (`shallow`)
    # before `thorough` ever bumps it to `standard`, so the notice stays the
    # structural one: the hint only changes how much dialogue the plan gets.
    signal = decide_depth(
        status="definite",
        choice_points=0,
        degraded=False,
        part_count=1,
        surface="studio",
        wide=True,
        depth_hint="thorough",
    )
    assert signal.depth == "standard"
    assert "family" in signal.notice
    assert "thorough" not in signal.notice


# --- 2. the fetch path: no planner, refs verbatim ----------------------------


async def test_definite_no_choice_points_skips_planner() -> None:
    resolver = ScriptedResolver([_envelope()])
    executor = ScriptedExecutor()
    _, _, events = await _start(
        create_root_workflow(resolver=resolver, executor=executor, tool_surface=None),
        "kenya oda 2023",
    )

    visited = _visited(events)
    assert "fetch" in visited
    assert "planner" not in visited
    assert "approve" not in visited
    assert _pending_interrupt(events) is None
    assert _output_of(events, "compose")["frame_refs"] == [REF_A]
    assert _output_of(events, "narrate")["frame_refs"] == [REF_A]


async def test_a_wide_definite_resolution_reaches_the_planner_not_fetch() -> None:
    """Zero facet-basis choice points, but the envelope marks the identity
    `wide` (a described-arm family): `fetch`'s no-planner, no-approval path
    must not be reached."""
    resolver = ScriptedResolver([_envelope(wide=True)])
    executor = ScriptedExecutor()
    _, _, events = await _start(
        create_root_workflow(
            resolver=resolver,
            executor=executor,
            generative=GenerativeNodes(planner=ScriptedPlannerModel()),
            tool_surface=None,
        ),
        "population of a country",
    )

    visited = _visited(events)
    assert "fetch" not in visited
    assert "planner" in visited
    assert _pending_interrupt(events) == APPROVE_INTERRUPT


# --- 3/4. the clarify loop ----------------------------------------------------


async def test_candidates_interrupt_then_resume_reprobes_with_choice() -> None:
    resolver = ScriptedResolver(
        [
            _envelope(
                status="candidates",
                spec_id=None,
                candidates=[
                    {"spec_id": SPEC_A, "label": "reading A"},
                    {"spec_id": SPEC_B, "label": "reading B"},
                ],
            ),
            _envelope(spec_id=SPEC_B),
        ]
    )
    executor = ScriptedExecutor()
    runner, session_id, events = await _start(
        create_root_workflow(resolver=resolver, executor=executor, tool_surface=None),
        "ambiguous q",
    )

    assert _pending_interrupt(events) == CLARIFY_INTERRUPT
    events = await _resume(runner, session_id, CLARIFY_INTERRUPT, {"spec_id": SPEC_B})

    assert len(resolver.calls) == 2
    assert resolver.calls[1]["spec_id"] == SPEC_B
    assert _output_of(events, "narrate") is not None


async def test_clarify_carries_the_chosen_candidate_wide_verdict_into_the_re_entry_probe() -> None:
    """`SPEC_B` is offered as `wide: True` (role-bearing, unpinnable -- the
    CRS `FinancialFlow` shape). Choosing it must re-enter `probe` with that
    width, not the resolver's own `spec_id`-set shortcut default: a real
    resolver (`McpResolver`) cannot recompute it without a second `search`,
    so `clarify`/`probe` must supply it."""
    resolver = ScriptedResolver(
        [
            _envelope(
                status="candidates",
                spec_id=None,
                candidates=[
                    {"spec_id": SPEC_A, "label": "reading A", "wide": False},
                    {"spec_id": SPEC_B, "label": "reading B", "wide": True},
                ],
            ),
            _envelope(spec_id=SPEC_B, wide=True),
        ]
    )
    executor = ScriptedExecutor()
    runner, session_id, events = await _start(
        create_root_workflow(
            resolver=resolver,
            executor=executor,
            generative=GenerativeNodes(planner=ScriptedPlannerModel()),
            tool_surface=None,
        ),
        "ambiguous q",
    )

    assert _pending_interrupt(events) == CLARIFY_INTERRUPT
    await _resume(runner, session_id, CLARIFY_INTERRUPT, {"spec_id": SPEC_B})

    assert len(resolver.calls) == 2
    assert resolver.calls[1] == {
        "question": "ambiguous q",
        "spec_id": SPEC_B,
        "wide": True,
        "parts": None,
    }


async def test_clarify_bound_takes_top_candidate_without_third_interrupt() -> None:
    candidates = [
        {"spec_id": SPEC_A, "label": "top-ranked"},
        {"spec_id": SPEC_B, "label": "runner-up"},
    ]
    unresolved = _envelope(status="candidates", spec_id=None, candidates=candidates)
    resolver = ScriptedResolver([unresolved, unresolved, unresolved, _envelope()])
    executor = ScriptedExecutor()
    runner, session_id, events = await _start(
        create_root_workflow(resolver=resolver, executor=executor, tool_surface=None),
        "stubborn q",
    )

    assert _pending_interrupt(events) == CLARIFY_INTERRUPT
    events = await _resume(runner, session_id, CLARIFY_INTERRUPT, {"spec_id": SPEC_B})
    assert _pending_interrupt(events) == CLARIFY_INTERRUPT
    events = await _resume(runner, session_id, CLARIFY_INTERRUPT, {"spec_id": SPEC_B})

    # Two consumed rounds hit the bound: the third pass through clarify takes
    # the top-ranked candidate and re-probes.
    assert _pending_interrupt(events) is None
    assert resolver.calls[-1]["spec_id"] == SPEC_A
    assert _output_of(events, "narrate") is not None


async def test_a_candidate_saved_without_statvars_is_fetched_by_its_spec_id_alone() -> None:
    """A session saved at the clarify interrupt by a build whose candidates
    carried no `stat_vars` resumes into a `get_data` call that posts none."""
    candidates = [
        {"spec_id": SPEC_A, "label": "reading A"},
        {"spec_id": SPEC_B, "label": "reading B"},
    ]
    resolver = ScriptedResolver(
        [_envelope(status="candidates", spec_id=None, candidates=candidates), _envelope()]
    )
    executor = ScriptedExecutor()
    runner, session_id, _ = await _start(
        create_root_workflow(resolver=resolver, executor=executor, tool_surface=None),
        "ambiguous q",
    )

    await _resume(runner, session_id, CLARIFY_INTERRUPT, {"spec_id": SPEC_A})

    assert executor.calls == [{"tool": "get_data", "args": {"spec_id": SPEC_A}}]


# --- 4b. the user's clarify election reaches the planner seam ---------------


_TWO_PART_DRAFT = {
    "text": "on it",
    "search_parts": [{"variable": "gdp"}, {"variable": "population"}],
    "about": "data",
}


async def test_clarify_election_reaches_the_planner_seam(
    real_schemas: list[ToolSchema],
) -> None:
    """Naming a candidate is the one genuine election: `probe`'s re-entry
    keeps `spec_id_elected` true while the envelope still names that same
    id, and the planner node passes both it and the candidate's own label
    through to the `PlannerModel` seam. The re-entry envelope's `parts` is
    derived from the search parts `probe` carries forward rather than
    hardcoded, so this also pins that the graph passes them through on
    re-entry (see `ScriptedResolver.derive_reentry_parts`)."""
    candidates = [
        {"spec_id": SPEC_A, "label": "reading A"},
        {"spec_id": SPEC_B, "label": "reading B"},
    ]
    resolver = ScriptedResolver(
        [
            _envelope(status="candidates", spec_id=None, parts=2, candidates=candidates),
            _envelope(spec_id=SPEC_B),
        ],
        derive_reentry_parts=True,
    )
    executor = ScriptedExecutor()
    converse = ScriptedConverse([_TWO_PART_DRAFT])
    seam = _Seam(
        [{"plan": _seam_plan(schemas=real_schemas).model_dump(mode="json"), "summary": {}}]
    )
    runner, session_id, events = await _start(
        create_root_workflow(
            resolver=resolver,
            executor=executor,
            generative=GenerativeNodes(planner=seam, converse=converse),
            tool_surface=None,
        ),
        "gdp and population, ambiguous",
    )

    assert _pending_interrupt(events) == CLARIFY_INTERRUPT
    await _resume(runner, session_id, CLARIFY_INTERRUPT, {"spec_id": SPEC_B})

    assert len(seam.calls) == 1
    assert seam.calls[0]["spec_id_elected"] is True
    assert seam.calls[0]["spec_id_elected_label"] == "reading B"
    assert seam.calls[0]["envelope"]["parts"] == 2


async def test_clarify_reconfirm_mismatch_does_not_elect(
    real_schemas: list[ToolSchema],
) -> None:
    """A genuine pick only survives to the planner while the re-resolve still
    names that same id: here the resolver's post-clarify call comes back
    naming a different spec_id than the one chosen, so `probe`'s
    reconfirmation check drops the election before the planner seam ever
    sees it."""
    candidates = [
        {"spec_id": SPEC_A, "label": "reading A"},
        {"spec_id": SPEC_B, "label": "reading B"},
    ]
    resolver = ScriptedResolver(
        [
            _envelope(status="candidates", spec_id=None, parts=2, candidates=candidates),
            _envelope(spec_id=SPEC_A),
        ],
        derive_reentry_parts=True,
    )
    executor = ScriptedExecutor()
    converse = ScriptedConverse([_TWO_PART_DRAFT])
    seam = _Seam(
        [{"plan": _seam_plan(schemas=real_schemas).model_dump(mode="json"), "summary": {}}]
    )
    runner, session_id, events = await _start(
        create_root_workflow(
            resolver=resolver,
            executor=executor,
            generative=GenerativeNodes(planner=seam, converse=converse),
            tool_surface=None,
        ),
        "gdp and population, ambiguous",
    )

    assert _pending_interrupt(events) == CLARIFY_INTERRUPT
    await _resume(runner, session_id, CLARIFY_INTERRUPT, {"spec_id": SPEC_B})

    assert len(seam.calls) == 1
    assert seam.calls[0]["spec_id_elected"] is False


async def test_clarify_skip_does_not_elect() -> None:
    """Declining to choose takes the default reading, a system default, so
    it must not reach the planner seam as an election."""
    candidates = [
        {"spec_id": SPEC_A, "label": "reading A"},
        {"spec_id": SPEC_B, "label": "reading B"},
    ]
    resolver = ScriptedResolver(
        [
            _envelope(status="candidates", spec_id=None, parts=2, candidates=candidates),
            _envelope(spec_id=SPEC_A, parts=2),
        ]
    )
    executor = ScriptedExecutor()
    seam = _Seam([{"plan": {}, "summary": {}}])
    runner, session_id, events = await _start(
        create_root_workflow(
            resolver=resolver,
            executor=executor,
            generative=GenerativeNodes(planner=seam),
            tool_surface=None,
        ),
        "gdp and population, ambiguous",
    )

    assert _pending_interrupt(events) == CLARIFY_INTERRUPT
    with pytest.raises(ValidationError):
        await _resume(runner, session_id, CLARIFY_INTERRUPT, {"skip": True})

    assert len(seam.calls) == 1
    assert seam.calls[0]["spec_id_elected"] is False
    assert seam.calls[0]["spec_id_elected_label"] is None


async def test_clarify_round_bound_does_not_elect() -> None:
    """The round bound takes the top-ranked candidate on no one's behalf --
    a system default, exactly like the skip branch -- so it too must not
    reach the planner seam as an election, even though an earlier round in
    the same turn was itself a genuine, later-superseded pick."""
    candidates = [
        {"spec_id": SPEC_A, "label": "top-ranked"},
        {"spec_id": SPEC_B, "label": "runner-up"},
    ]
    unresolved = _envelope(status="candidates", spec_id=None, parts=2, candidates=candidates)
    resolver = ScriptedResolver([unresolved, unresolved, unresolved, _envelope(parts=2)])
    executor = ScriptedExecutor()
    seam = _Seam([{"plan": {}, "summary": {}}])
    runner, session_id, events = await _start(
        create_root_workflow(
            resolver=resolver,
            executor=executor,
            generative=GenerativeNodes(planner=seam),
            tool_surface=None,
        ),
        "stubborn compound q",
    )

    assert _pending_interrupt(events) == CLARIFY_INTERRUPT
    events = await _resume(runner, session_id, CLARIFY_INTERRUPT, {"spec_id": SPEC_B})
    assert _pending_interrupt(events) == CLARIFY_INTERRUPT
    with pytest.raises(ValidationError):
        await _resume(runner, session_id, CLARIFY_INTERRUPT, {"spec_id": SPEC_B})

    # The bound's own probe reached the planner with no election on record,
    # even though the two consumed rounds above each named a candidate.
    assert len(seam.calls) == 1
    assert seam.calls[0]["spec_id_elected"] is False
    assert seam.calls[0]["spec_id_elected_label"] is None


# --- 5. unsupported is terminal ----------------------------------------------


async def test_unsupported_dead_ends_without_retry() -> None:
    resolver = ScriptedResolver(
        [
            _envelope(
                status="unsupported",
                spec_id=None,
                warnings=[
                    {"severity": "fatal", "message": "not expressible as published"},
                ],
            )
        ]
    )
    executor = ScriptedExecutor()
    _, _, events = await _start(
        create_root_workflow(resolver=resolver, executor=executor, tool_surface=None),
        "impossible q",
    )

    assert len(resolver.calls) == 1
    assert executor.calls == []
    assert _pending_interrupt(events) is None
    text = _output_of(events, "deadend")["text"]
    assert "cannot be expressed" in text
    # Resolver diagnostics stay OFF the user-facing close (they ride the
    # resolve step's detail on the wire instead).
    assert "not expressible as published" not in text
    assert "narrate" not in _visited(events)


# --- 5b. converse's own decomposition failure, and the emptiness guard ------


class ZeroPartsConverse:
    """Stands in for a real decomposition failure: drafts opening text but no
    search_parts at all."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    async def __call__(
        self,
        *,
        question: str,
        canvas_digest: dict[str, Any] | None = None,
        conversation_digest: dict[str, Any] | None = None,
        recall_result: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self.calls.append(question)
        return {"text": "looking into that", "search_parts": [], "about": "data", "recall": None}


async def test_converse_zero_parts_dead_ends_without_probing() -> None:
    resolver = ScriptedResolver([_envelope()])
    executor = ScriptedExecutor()
    converse = ZeroPartsConverse()
    workflow = create_root_workflow(
        resolver=resolver,
        executor=executor,
        generative=GenerativeNodes(converse=converse),
        tool_surface=None,
    )
    _, _, events = await _start(workflow, "vague noise with nothing to search for")

    assert converse.calls == ["vague noise with nothing to search for"]
    assert resolver.calls == []
    assert executor.calls == []
    assert "probe" not in _visited(events)
    assert "deadend" in _visited(events)
    assert _output_of(events, "deadend")["text"]


def _part(variable: str, status: str, spec_id: str | None = None) -> dict[str, Any]:
    candidates = [] if spec_id else [{"spec_id": SPEC_B, "label": "reading B", "wide": True}]
    return {
        "variable": variable,
        "status": status,
        "spec_id": spec_id,
        "resolved": None,
        "warnings": [],
        "candidates": candidates,
    }


async def test_every_part_is_kept_apart_from_the_planners_envelope() -> None:
    """`part_digests` never reaches `resolve_envelope`, which the planner
    reads, and a clarify election fills in the first part alone."""
    first_search = _envelope(
        status="candidates",
        spec_id=None,
        candidates=[{"spec_id": SPEC_B, "label": "reading B"}],
        part_digests=[
            _part("aid", "candidates"),
            {**_part("population", "definite", SPEC_A), "spec_block": {"spec_id": SPEC_A}},
        ],
    )
    resolver = ScriptedResolver([first_search, _envelope(spec_id=SPEC_B, wide=True)])
    workflow = create_root_workflow(
        resolver=resolver,
        executor=ScriptedExecutor(),
        generative=GenerativeNodes(planner=ScriptedPlannerModel()),
        tool_surface=None,
    )
    runner, session_id, _ = await _start(workflow, "aid and population")
    session = runner.session_service.sessions[runner.app_name]["u"][session_id]
    assert "part_digests" not in json.loads(session.state["resolve_envelope"])
    assert [p["status"] for p in json.loads(session.state["part_digests"])] == [
        "candidates",
        "definite",
    ]
    # A part's spec block (for a caller that fetches every part) stays off state.
    assert not any("spec_block" in p for p in json.loads(session.state["part_digests"]))

    await _resume(runner, session_id, CLARIFY_INTERRUPT, {"spec_id": SPEC_B})
    session = runner.session_service.sessions[runner.app_name]["u"][session_id]
    first, second = json.loads(session.state["part_digests"])
    assert (first["status"], first["spec_id"], first["resolved"]) == (
        "definite",
        SPEC_B,
        "reading B",
    )
    assert (first["wide"], first["candidates"]) == (True, [])
    assert second == _part("population", "definite", SPEC_A)


class ReplyConverse:
    """Answers the message itself: `about: "reply"` with no search parts."""

    async def __call__(
        self,
        *,
        question: str,
        canvas_digest: dict[str, Any] | None = None,
        conversation_digest: dict[str, Any] | None = None,
        recall_result: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return {"text": "Hello. Ask me about aid.", "search_parts": [], "about": "reply"}


async def test_a_reply_closes_the_turn_with_converses_answer_and_no_search() -> None:
    resolver = ScriptedResolver([_envelope()])
    executor = ScriptedExecutor()
    workflow = create_root_workflow(
        resolver=resolver,
        executor=executor,
        generative=GenerativeNodes(converse=ReplyConverse()),
        tool_surface=None,
    )
    _, _, events = await _start(workflow, "hi")

    assert resolver.calls == []
    assert executor.calls == []
    assert _visited(events) == ["converse", "reply"]
    assert "text" not in _output_of(events, "converse")
    assert _output_of(events, "reply") == {"text": "Hello. Ask me about aid."}


class ScriptedConverse:
    """Returns one scripted draft per call, in call order; records each
    question it was invoked with."""

    def __init__(self, drafts: list[dict[str, Any]]) -> None:
        self.drafts = drafts
        self.calls: list[str] = []

    async def __call__(
        self,
        *,
        question: str,
        canvas_digest: dict[str, Any] | None = None,
        conversation_digest: dict[str, Any] | None = None,
        recall_result: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self.calls.append(question)
        index = min(len(self.calls) - 1, len(self.drafts) - 1)
        return self.drafts[index]


async def test_clarify_open_text_reply_refines_through_converse_not_straight_to_probe() -> None:
    resolver = ScriptedResolver(
        [
            _envelope(
                status="candidates",
                spec_id=None,
                candidates=[
                    {"spec_id": SPEC_A, "label": "reading A"},
                    {"spec_id": SPEC_B, "label": "reading B"},
                ],
            ),
            _envelope(spec_id=SPEC_B),
        ]
    )
    executor = ScriptedExecutor()
    converse = ScriptedConverse(
        [
            {"text": "on it", "search_parts": [{"variable": "aid"}], "about": "data"},
            {
                "text": "on it",
                "search_parts": [{"variable": "aid", "entities": [{"text": "Kenya"}]}],
                "about": "data",
            },
        ]
    )
    workflow = create_root_workflow(
        resolver=resolver,
        executor=executor,
        generative=GenerativeNodes(converse=converse),
        tool_surface=None,
    )
    runner, session_id, events = await _start(workflow, "aid to somewhere ambiguous")

    assert _pending_interrupt(events) == CLARIFY_INTERRUPT
    events = await _resume(runner, session_id, CLARIFY_INTERRUPT, {"text": "I meant Kenya"})

    # The refinement re-entered converse with the combined question instead
    # of reaching probe directly with the stale (now-cleared) parts.
    assert converse.calls == [
        "aid to somewhere ambiguous",
        "aid to somewhere ambiguous -- I meant Kenya",
    ]
    assert "converse" in _visited(events)
    assert len(resolver.calls) == 2
    assert resolver.calls[1]["question"] == "aid to somewhere ambiguous -- I meant Kenya"
    assert resolver.calls[1]["spec_id"] is None
    assert resolver.calls[1]["parts"] == [{"variable": "aid", "entities": [{"text": "Kenya"}]}]
    assert _output_of(events, "narrate") is not None


async def test_clarify_skip_still_takes_the_top_candidate_via_again() -> None:
    """The other two clarify branches (skip, spec_id choice) are unaffected
    by the new "refine" route -- they keep going straight back to probe."""
    resolver = ScriptedResolver(
        [
            _envelope(
                status="candidates",
                spec_id=None,
                candidates=[
                    {"spec_id": SPEC_A, "label": "reading A"},
                    {"spec_id": SPEC_B, "label": "reading B"},
                ],
            ),
            _envelope(spec_id=SPEC_A),
        ]
    )
    executor = ScriptedExecutor()
    runner, session_id, events = await _start(
        create_root_workflow(resolver=resolver, executor=executor, tool_surface=None), "ambiguous q"
    )
    assert _pending_interrupt(events) == CLARIFY_INTERRUPT

    events = await _resume(runner, session_id, CLARIFY_INTERRUPT, {"skip": True})

    # The resumed leg never re-enters converse -- "skip" still routes
    # straight back to probe via "again".
    assert "converse" not in _visited(events)
    assert len(resolver.calls) == 2
    assert resolver.calls[1]["spec_id"] == SPEC_A
    assert _output_of(events, "narrate") is not None


async def test_role_text_warning_with_live_candidates_still_reaches_clarify() -> None:
    """A candidate to offer decides the route: dfo-apis treats
    `role_text_not_recognised` as non-fatal and still returns real
    candidates, so the turn belongs on `clarify`, which has something to
    offer."""
    resolver = ScriptedResolver(
        [
            _envelope(
                status="candidates",
                spec_id=None,
                candidates=[{"spec_id": SPEC_A, "label": "reading A"}],
                warnings=[
                    {
                        "code": "role_text_not_recognised",
                        "severity": "warn",
                        "message": "role text 'debtor' is not in the accepted grammar",
                    }
                ],
            )
        ]
    )
    executor = ScriptedExecutor()
    _, _, events = await _start(
        create_root_workflow(resolver=resolver, executor=executor, tool_surface=None),
        "debtor countries aid",
    )

    assert _pending_interrupt(events) == CLARIFY_INTERRUPT
    assert "clarify" in _visited(events)
    assert "deadend" not in _visited(events)
    assert len(resolver.calls) == 1


async def test_time_text_warning_with_live_candidates_still_reaches_clarify() -> None:
    resolver = ScriptedResolver(
        [
            _envelope(
                status="no_data",
                spec_id=None,
                candidates=[{"spec_id": SPEC_A, "label": "nearest"}],
                warnings=[
                    {
                        "code": "time_text_not_parsed",
                        "severity": "warn",
                        "message": "time text 'a while back' is not in the accepted grammar",
                    }
                ],
            )
        ]
    )
    executor = ScriptedExecutor()
    _, _, events = await _start(
        create_root_workflow(resolver=resolver, executor=executor, tool_surface=None),
        "aid a while back",
    )

    assert _pending_interrupt(events) == CLARIFY_INTERRUPT
    assert "clarify" in _visited(events)
    assert "deadend" not in _visited(events)


async def test_nonsense_bitcoin_aid_still_dead_ends_not_answered() -> None:
    """Regression guard: a no_data resolution with no nearest-real
    alternatives maps to `unsupported` (`graph_envelope`'s own mapping)
    and must stay a refusal."""
    resolver = ScriptedResolver([_envelope(status="unsupported", spec_id=None, warnings=[])])
    executor = ScriptedExecutor()
    _, _, events = await _start(
        create_root_workflow(resolver=resolver, executor=executor, tool_surface=None),
        "Does Bitcoin count as foreign aid?",
    )

    assert "deadend" in _visited(events)
    assert "narrate" not in _visited(events)
    assert executor.calls == []


async def test_candidates_status_with_empty_candidates_dead_ends_not_loops() -> None:
    """The general invariant this guard exists for: an empty `candidates`
    list decides `clarify` versus `deadend`, regardless of which warning
    code rode along. A `candidates` status with nothing to choose from
    would otherwise loop `clarify -> probe -> clarify` past
    `MAX_CLARIFY_ROUNDS` forever (module docstring), so it dead-ends
    instead, same as the `no_data` case below -- and an info-severity
    warning that has nothing to do with emptiness rides along without
    changing that."""
    statuses_seen: list[str] = []

    class RecordingRefusal:
        async def __call__(
            self, *, question: str, status: str, warnings: list[str]
        ) -> dict[str, Any]:
            statuses_seen.append(status)
            return {"text": ""}

    resolver = ScriptedResolver(
        [
            _envelope(
                status="candidates",
                spec_id=None,
                candidates=[],
                warnings=[
                    {
                        "code": "retrieval_score_weak",
                        "severity": "info",
                        "message": "weak match confidence",
                    }
                ],
            )
        ]
    )
    executor = ScriptedExecutor()
    _, _, events = await _start(
        create_root_workflow(
            resolver=resolver,
            executor=executor,
            generative=GenerativeNodes(refusal=RecordingRefusal()),
            tool_surface=None,
        ),
        "no candidates at all",
    )

    assert _pending_interrupt(events) is None
    assert "clarify" not in _visited(events)
    assert "deadend" in _visited(events)
    # The forced status reaches the refusal seam as "unsupported": deadend's
    # event output, along with its route, carries the forced status onward.
    assert statuses_seen == ["unsupported"]


async def test_no_data_status_with_empty_candidates_dead_ends() -> None:
    """The `no_data` case with nothing to choose, covered by the same guard
    as the `candidates` case above. The `no_data` finding is already an
    honest reason and reaches the refusal seam unchanged, while
    `candidates` gets forced to "unsupported"."""
    statuses_seen: list[str] = []

    class RecordingRefusal:
        async def __call__(
            self, *, question: str, status: str, warnings: list[str]
        ) -> dict[str, Any]:
            statuses_seen.append(status)
            return {"text": ""}

    resolver = ScriptedResolver([_envelope(status="no_data", spec_id=None, candidates=[])])
    executor = ScriptedExecutor()
    _, _, events = await _start(
        create_root_workflow(
            resolver=resolver,
            executor=executor,
            generative=GenerativeNodes(refusal=RecordingRefusal()),
            tool_surface=None,
        ),
        "does this exist",
    )

    assert _pending_interrupt(events) is None
    assert "clarify" not in _visited(events)
    assert "deadend" in _visited(events)
    assert statuses_seen == ["no_data"]


# --- 5c. converse's `revises`: the canvas card a question asks to change ------


class RevisesConverse:
    """Returns one draft naming `revises`, whatever the question. The node's
    validation is what the tests below exercise, so the digest it is handed
    is ignored, except by `digest_seen`, which records it."""

    def __init__(self, revises: Any) -> None:
        self.revises = revises
        self.digest_seen: dict[str, Any] | None = None

    async def __call__(
        self,
        *,
        question: str,
        canvas_digest: dict[str, Any] | None = None,
        conversation_digest: dict[str, Any] | None = None,
        recall_result: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        del question, conversation_digest, recall_result
        self.digest_seen = canvas_digest
        return {
            "text": "on it",
            "search_parts": [{"variable": "life expectancy", "entities": [{"text": "Guatemala"}]}],
            "about": "data",
            "revises": self.revises,
            "recall": None,
        }


def _digest_with(component_id: str) -> dict[str, Any]:
    return {
        "blocks": [
            {
                "component_id": component_id,
                "type": "line_chart",
                "title": "Life expectancy in Nigeria",
                "revision": 1,
                "entities": ["Nigeria"],
                "variable": ["Life Expectancy"],
                "time": "1960-2023",
            }
        ]
    }


def _valid_component(
    component_id: str = "cmp_scripted0001",
    *,
    revision: int = 1,
    fact_summary: dict[str, Any] | None = None,
    **overrides: Any,
) -> dict[str, Any]:
    """A `write_component` result's `component` member that satisfies the
    full `component.created` wire contract (`harness.wire.
    usable_component`), the same check `harness.graph`'s `compose` uses to
    decide a write is usable. A test that wants a usable write needs this,
    not a bare `{component_id, revision, fact_summary}` stub -- compose
    requires the full contract, not a present `fact_summary` alone."""
    base: dict[str, Any] = {
        "component_id": component_id,
        "revision": revision,
        "type": "stat_card",
        "catalog_version": "1",
        "min_catalog_version": "1",
        "payload": {},
        "fact_summary": fact_summary
        if fact_summary is not None
        else {
            "component_id": component_id,
            "revision": revision,
            "series": [],
            "indicators": [],
            "attribution": {"label": "Source", "source_count": 1, "indicator_count": 1},
        },
        "provenance": {
            "inputs": [REF_A],
            "sources": [{"id": "src_test", "name": "Test source", "licence": "CC BY 4.0"}],
            "freshness": {
                "data_through": "2024",
                "last_refreshed": "2026-06-30",
                "partial_period": False,
            },
            "warnings": [],
            "generated_at": "2026-01-01T00:00:00Z",
            "citation": "Test source (2026).",
        },
    }
    base.update(overrides)
    return base


async def _start_with_digest(
    workflow: Any,
    question: str,
    digest: dict[str, Any],
    *,
    composed_ids: list[str] | None = None,
    auto_add: bool = True,
    conversation_digest: dict[str, Any] | None = None,
    anaphor_window: str | None = None,
) -> Any:
    """`_start`, plus the seeded `canvas_digest`, `composed_ids` and
    `auto_add` a second turn of a real conversation carries
    (`harness.studio.StudioRunner` seeds all three the same way).
    `composed_ids` defaults to every block id `digest` itself carries,
    matching the common case where the digest's blocks are exactly what is
    composed. A test exercising a focus-only card (never composed) passes
    something narrower. `conversation_digest`, left unseeded by default,
    is `ConversationState.conversation_digest_state`'s own turn-ordered
    shape, fed to `models.converse` as cross-turn context. `anaphor_window`
    is the span `ConversationState.anaphor_window` resolved for the turn
    (`studio.py` seeds it the same way); a test of the window anaphor
    passes it explicitly."""
    if composed_ids is None:
        composed_ids = [block["component_id"] for block in digest.get("blocks", [])]
    runner = InMemoryRunner(node=workflow)
    state: dict[str, Any] = {
        "canvas_digest": json.dumps(digest),
        "composed_ids": json.dumps(composed_ids),
        "auto_add": auto_add,
    }
    if conversation_digest is not None:
        state["conversation_digest"] = json.dumps(conversation_digest)
    if anaphor_window is not None:
        state["anaphor_window"] = anaphor_window
    session = await runner.session_service.create_session(
        app_name=runner.app_name,
        user_id="u",
        state=state,
    )
    events = [
        event
        async for event in runner.run_async(
            user_id="u",
            session_id=session.id,
            new_message=types.Content(role="user", parts=[types.Part.from_text(text=question)]),
        )
    ]
    return runner, session.id, events


async def test_converse_revises_naming_a_canvas_block_is_stored_in_turn_state() -> None:
    converse = RevisesConverse("cmp_scripted0001")
    workflow = create_root_workflow(
        resolver=ScriptedResolver([_envelope()]),
        executor=ScriptedExecutor(),
        generative=GenerativeNodes(converse=converse),
        tool_surface=None,
    )
    runner, session_id, events = await _start_with_digest(
        workflow, "add Guatemala to the chart", _digest_with("cmp_scripted0001")
    )
    assert "converse" in _visited(events)

    session = runner.session_service.sessions[runner.app_name]["u"][session_id]
    assert session.state["revises"] == "cmp_scripted0001"


async def test_converse_revises_naming_no_canvas_block_is_dropped(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """An id from another conversation, a card since removed, or a model
    invention all still let the turn proceed as an append."""
    converse = RevisesConverse("cmp_someone_elses")
    workflow = create_root_workflow(
        resolver=ScriptedResolver([_envelope()]),
        executor=ScriptedExecutor(),
        generative=GenerativeNodes(converse=converse),
        tool_surface=None,
    )
    with caplog.at_level(logging.INFO, logger="harness.graph"):
        runner, session_id, events = await _start_with_digest(
            workflow, "add Guatemala to the chart", _digest_with("cmp_scripted0001")
        )

    assert "narrate" in _visited(events)
    session = runner.session_service.sessions[runner.app_name]["u"][session_id]
    assert session.state.get("revises") is None
    assert any("no block on this canvas" in r.getMessage() for r in caplog.records)


async def test_converse_revises_is_absent_when_converse_names_none() -> None:
    """For a new, independent chart, converse leaves the field unset, and no
    `revises` key reaches turn state for the recorder to act on."""
    converse = RevisesConverse(None)
    workflow = create_root_workflow(
        resolver=ScriptedResolver([_envelope()]),
        executor=ScriptedExecutor(),
        generative=GenerativeNodes(converse=converse),
        tool_surface=None,
    )
    runner, session_id, _ = await _start_with_digest(
        workflow, "now show me Guatemala", _digest_with("cmp_scripted0001")
    )

    session = runner.session_service.sessions[runner.app_name]["u"][session_id]
    assert session.state.get("revises") is None


# --- Window anaphors resolved from state's anaphor_window -----------------
#
# `ConversationState.anaphor_window` (studio_state.py) decides which window
# a card-scoped follow-up or a "the same years" question resolves to, from
# the full turn history and, for a focus card, that card's own window --
# that logic is exercised directly against `ConversationState` in
# `test_studio_state.py`. Here, `converse` gets a resolved `anaphor_window`
# from state, exactly as `studio.py` seeds it, and only has to substitute
# it in (or leave the text alone when state carries none).


class _AnaphorConverse:
    """Echoes the question it received and drafts one search part whose
    `time` field names a window anaphor, so a test can see whether the
    substitution reached both the free-text question and the part before
    either leaves `converse`."""

    def __init__(self) -> None:
        self.digest_seen: dict[str, Any] | None = None

    async def __call__(
        self,
        *,
        question: str,
        canvas_digest: dict[str, Any] | None = None,
        conversation_digest: dict[str, Any] | None = None,
        recall_result: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        del conversation_digest, recall_result
        self.digest_seen = canvas_digest
        return {
            "text": "on it",
            "search_parts": [
                {
                    "variable": "life expectancy",
                    "entities": [{"text": "Guatemala"}],
                    "time": "the same years",
                }
            ],
            "about": "data",
            "recall": None,
        }


async def test_window_anaphor_resolved_in_question_and_search_parts() -> None:
    resolver = ScriptedResolver([_envelope()])
    converse = _AnaphorConverse()
    workflow = create_root_workflow(
        resolver=resolver,
        executor=ScriptedExecutor(),
        generative=GenerativeNodes(converse=converse),
        tool_surface=None,
    )
    _, _, events = await _start_with_digest(
        workflow,
        "show it for the same years for Guatemala",
        _digest_with("cmp_scripted0001"),
        anaphor_window="2015 to 2020",
    )

    assert "the same years" not in resolver.calls[0]["question"]
    assert "2015 to 2020" in resolver.calls[0]["question"]
    assert resolver.calls[0]["parts"][0]["time"] == "2015 to 2020"
    assert _output_of(events, "converse")["question"] == resolver.calls[0]["question"]


async def test_window_anaphor_left_alone_with_no_prior_window() -> None:
    """No canvas digest at all (a turn's very first question): the anaphor
    text passes through untouched, since there is no window to substitute."""
    resolver = ScriptedResolver([_envelope()])
    converse = _AnaphorConverse()
    workflow = create_root_workflow(
        resolver=resolver,
        executor=ScriptedExecutor(),
        generative=GenerativeNodes(converse=converse),
        tool_surface=None,
    )
    _, _, events = await _start(workflow, "life expectancy for the same years")
    del events

    assert resolver.calls[0]["question"] == "life expectancy for the same years"
    assert resolver.calls[0]["parts"][0]["time"] == "the same years"


# --- Depth hint and the converse draft run concurrently -------------------


async def test_depth_hint_and_converse_draft_run_concurrently() -> None:
    """Both seams take only `question`, constructed before the
    `asyncio.gather` that awaits them (`graph.py`'s `converse` node), so both
    genuinely start before either finishes. Proved here with a pair of
    `threading.Event`s each side sets on entry and waits on for the other,
    rather than a wall-clock margin a loaded test run can blow through: a
    serial call order leaves one side waiting on a signal the other, not yet
    started, never sends, which times out and fails the assertion instead of
    hanging. `threading.Event`, not `asyncio.Event`, because `slow_hint`
    runs on a real OS thread (`asyncio.to_thread`), and only a
    `threading.Event`'s `set`/`wait` are safe to call across threads;
    `_SlowConverse` offloads its own wait onto a thread too, so it never
    blocks the event loop `slow_hint`'s own thread depends on running
    concurrently."""
    hint_started = threading.Event()
    converse_started = threading.Event()

    def slow_hint(*, question: str) -> str:
        del question
        hint_started.set()
        assert converse_started.wait(timeout=5.0), (
            "converse never started while depth_hint was still running"
        )
        return "standard"

    class _SlowConverse:
        async def __call__(
            self,
            *,
            question: str,
            canvas_digest: dict[str, Any] | None = None,
            conversation_digest: dict[str, Any] | None = None,
            recall_result: dict[str, Any] | None = None,
        ) -> dict[str, Any]:
            del question, canvas_digest, conversation_digest, recall_result
            converse_started.set()
            started = await asyncio.to_thread(hint_started.wait, 5.0)
            assert started, "depth_hint never started while converse was still running"
            return {"text": "on it", "search_parts": [{"variable": "aid"}], "about": "data"}

    workflow = create_root_workflow(
        resolver=ScriptedResolver([_envelope()]),
        executor=ScriptedExecutor(),
        depth_hint_extractor=slow_hint,
        generative=GenerativeNodes(converse=_SlowConverse()),
        tool_surface=None,
    )
    _, _, events = await _start(workflow, "aid to kenya")

    assert _output_of(events, "converse")["depth_hint"] == "standard"


class RevisesOnlyOnceConverse:
    """Names a canvas card on the first call only.

    The graph re-enters converse on clarify's "refine" edge, so a second call
    happens inside one turn. A seam that only ever sets `revises` would leave
    the first draft's target standing, and the recorder would swap a card the
    refined question no longer asks about.
    """

    def __init__(self) -> None:
        self.calls = 0

    async def __call__(
        self,
        *,
        question: str,
        canvas_digest: dict[str, Any] | None = None,
        conversation_digest: dict[str, Any] | None = None,
        recall_result: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        del question, canvas_digest, conversation_digest, recall_result
        self.calls += 1
        return {
            "text": "on it",
            "search_parts": [{"variable": "life expectancy", "entities": [{"text": "Guatemala"}]}],
            "about": "data",
            "revises": "cmp_scripted0001" if self.calls == 1 else None,
            "recall": None,
        }


async def test_a_refined_re_entry_that_drops_revises_clears_it() -> None:
    resolver = ScriptedResolver(
        [
            _envelope(
                status="candidates",
                spec_id=None,
                candidates=[
                    {"spec_id": SPEC_A, "label": "reading A"},
                    {"spec_id": SPEC_B, "label": "reading B"},
                ],
            ),
            _envelope(spec_id=SPEC_B),
        ]
    )
    converse = RevisesOnlyOnceConverse()
    workflow = create_root_workflow(
        resolver=resolver,
        executor=ScriptedExecutor(),
        generative=GenerativeNodes(converse=converse),
        tool_surface=None,
    )
    runner, session_id, events = await _start_with_digest(
        workflow, "add Guatemala to the chart", _digest_with("cmp_scripted0001")
    )
    assert _pending_interrupt(events) == CLARIFY_INTERRUPT

    # An open-text reply refines the question, which re-enters converse for a
    # fresh draft naming no card.
    await _resume(runner, session_id, CLARIFY_INTERRUPT, {"text": "actually just Guatemala"})

    assert converse.calls == 2
    session = runner.session_service.sessions[runner.app_name]["u"][session_id]
    assert session.state.get("revises") is None


async def test_narrate_facts_say_revises_requested_when_the_turn_named_a_card() -> None:
    """`revises_requested` reads only the graph's own `ctx.state["revises"]`,
    set from converse via `_validated_revises`. It records that a swap was
    asked for this turn. It does not say whether the swap landed.
    `components_replaced` is the separate fact for the outcome, false here
    because `ScriptedExecutor` defines no `components_replaced` capability,
    so `narrate` falls back to False rather than trusting the turn-start
    validation.
    `test_narrate_facts_say_components_replaced_is_true_when_the_capability_confirms_it`
    covers the case where both capabilities are present and actually replace."""
    converse = RevisesConverse("cmp_scripted0001")
    narrate = _CapturingNarrate()
    workflow = create_root_workflow(
        resolver=ScriptedResolver([_envelope()]),
        executor=ScriptedExecutor(
            [
                {"outcome": "ok", "frame_refs": [REF_A]},  # get_data
                {"outcome": "ok", "view": {"dimensions": ["stat_var"]}},  # describe_frame
                {
                    "outcome": "ok",
                    "frame_refs": [],
                    "component": _valid_component("cmp_scripted0001", revision=2),
                    "summary": "created",
                },
            ]
        ),
        generative=GenerativeNodes(converse=converse, compose=RecordingCompose(), narrate=narrate),
        tool_surface=None,
    )
    await _start_with_digest(
        workflow, "add Guatemala to the chart", _digest_with("cmp_scripted0001")
    )

    assert narrate.facts[-1]["revises_requested"] is True
    assert narrate.facts[-1]["components_replaced"] is False


class _RevisesStillThereExecutor(ScriptedExecutor):
    """`ScriptedExecutor`, plus the two capabilities `_RecordingExecutor`
    exposes in production (`harness.studio_translate`): `land_component`
    (stamping only, in production) and `components_replaced` (reading the
    stamp's outcome). This stand-in collapses stamp-then-drain into one
    step, since nothing here exercises the real translator -- the ordinary
    case where nothing touched the canvas between converse's validation
    and this write, so every `land=True` write with a revise target
    replaces."""

    def __init__(self, results: list[dict[str, Any]] | None = None) -> None:
        super().__init__(results)
        self._replaced = False

    def land_component(self, component: dict[str, Any], *, land: bool, revises: str | None) -> None:
        del component
        if land and revises:
            self._replaced = True

    def components_replaced(self) -> bool:
        return self._replaced


class _RevisesGoneMidTurnExecutor(ScriptedExecutor):
    """Same two capabilities, always appends -- stands in for
    `/composition` (`harness.studio`'s route of that name) removing the
    target between converse's turn-start validation and the write
    landing, since that route runs without the turn lock."""

    def land_component(self, component: dict[str, Any], *, land: bool, revises: str | None) -> None:
        del component, land, revises

    def components_replaced(self) -> bool:
        return False


async def test_narrate_facts_say_components_replaced_is_true_when_the_capability_confirms_it() -> (
    None
):
    """This exercises the capability path: `compose` stamps the write
    `land=True` with the revise target, and `narrate` reads `True` back
    off `components_replaced`."""
    converse = RevisesConverse("cmp_scripted0001")
    narrate = _CapturingNarrate()
    workflow = create_root_workflow(
        resolver=ScriptedResolver([_envelope()]),
        executor=_RevisesStillThereExecutor(
            [
                {"outcome": "ok", "frame_refs": [REF_A]},  # get_data
                {"outcome": "ok", "view": {"dimensions": ["stat_var"]}},  # describe_frame
                {
                    "outcome": "ok",
                    "frame_refs": [],
                    "component": _valid_component("cmp_scripted0001", revision=2),
                    "summary": "created",
                },
            ]
        ),
        generative=GenerativeNodes(converse=converse, compose=RecordingCompose(), narrate=narrate),
        tool_surface=None,
    )
    await _start_with_digest(
        workflow, "add Guatemala to the chart", _digest_with("cmp_scripted0001")
    )

    assert narrate.facts[-1]["revises_requested"] is True
    assert narrate.facts[-1]["components_replaced"] is True


async def test_narrate_facts_say_components_replaced_is_false_when_the_target_vanishes_mid_turn() -> (
    None
):
    """The target block is removed (`/composition`) between converse's
    turn-start validation and the write landing. `revises_requested` still
    says a swap was asked for, since converse's own state is untouched by
    the canvas change. `components_replaced`, read off the executor's own
    capability after the write that would have landed it, says it did not
    happen, so narrate must describe the new card as an addition."""
    converse = RevisesConverse("cmp_scripted0001")
    narrate = _CapturingNarrate()
    workflow = create_root_workflow(
        resolver=ScriptedResolver([_envelope()]),
        executor=_RevisesGoneMidTurnExecutor(
            [
                {"outcome": "ok", "frame_refs": [REF_A]},  # get_data
                {"outcome": "ok", "view": {"dimensions": ["stat_var"]}},  # describe_frame
                {
                    "outcome": "ok",
                    "frame_refs": [],
                    "component": _valid_component("cmp_scripted0002"),
                    "summary": "created",
                },
            ]
        ),
        generative=GenerativeNodes(converse=converse, compose=RecordingCompose(), narrate=narrate),
        tool_surface=None,
    )
    await _start_with_digest(
        workflow, "add Guatemala to the chart", _digest_with("cmp_scripted0001")
    )

    assert narrate.facts[-1]["revises_requested"] is True
    assert narrate.facts[-1]["components_replaced"] is False


async def test_narrate_facts_say_revises_requested_is_false_for_a_fresh_card() -> None:
    converse = RevisesConverse(None)
    narrate = _CapturingNarrate()
    workflow = create_root_workflow(
        resolver=ScriptedResolver([_envelope()]),
        executor=ScriptedExecutor(
            [
                {"outcome": "ok", "frame_refs": [REF_A]},  # get_data
                {"outcome": "ok", "view": {"dimensions": ["stat_var"]}},  # describe_frame
                {
                    "outcome": "ok",
                    "frame_refs": [],
                    "component": {"component_id": "cmp_scripted0002", "revision": 1},
                    "summary": "created",
                },
            ]
        ),
        generative=GenerativeNodes(converse=converse, compose=RecordingCompose(), narrate=narrate),
        tool_surface=None,
    )
    await _start_with_digest(workflow, "now show me Guatemala", _digest_with("cmp_scripted0001"))

    assert narrate.facts[-1]["revises_requested"] is False
    assert narrate.facts[-1]["components_replaced"] is False


async def test_converse_revises_is_dropped_when_auto_add_is_off() -> None:
    """`auto_add=False` means the recorder never composes a block for
    anything this turn, so a named card has nothing to replace.
    `_validated_revises` must drop it even though the digest still shows a
    real, composed block."""
    converse = RevisesConverse("cmp_scripted0001")
    workflow = create_root_workflow(
        resolver=ScriptedResolver([_envelope()]),
        executor=ScriptedExecutor(),
        generative=GenerativeNodes(converse=converse),
        tool_surface=None,
    )
    runner, session_id, _ = await _start_with_digest(
        workflow,
        "add Guatemala to the chart",
        _digest_with("cmp_scripted0001"),
        auto_add=False,
    )

    session = runner.session_service.sessions[runner.app_name]["u"][session_id]
    assert session.state.get("revises") is None


async def test_converse_revises_is_dropped_when_the_named_block_was_never_composed() -> None:
    """A card the digest carries only because a card-scoped follow-up
    focused it, never composed, must not validate as a replace target, or
    the recorder would be asked to swap a block that does not exist."""
    converse = RevisesConverse("cmp_scripted0001")
    workflow = create_root_workflow(
        resolver=ScriptedResolver([_envelope()]),
        executor=ScriptedExecutor(),
        generative=GenerativeNodes(converse=converse),
        tool_surface=None,
    )
    runner, session_id, _ = await _start_with_digest(
        workflow,
        "add Guatemala to the chart",
        _digest_with("cmp_scripted0001"),
        composed_ids=[],
    )

    session = runner.session_service.sessions[runner.app_name]["u"][session_id]
    assert session.state.get("revises") is None


async def test_narrate_facts_say_revises_requested_is_false_when_auto_add_is_off() -> None:
    """`auto_add=False` this turn means narrate's own fact must say the
    replace was not requested, even though converse named a real card. The
    recorder only creates this turn. It never replaces."""
    converse = RevisesConverse("cmp_scripted0001")
    narrate = _CapturingNarrate()
    workflow = create_root_workflow(
        resolver=ScriptedResolver([_envelope()]),
        executor=ScriptedExecutor(
            [
                {"outcome": "ok", "frame_refs": [REF_A]},  # get_data
                {"outcome": "ok", "view": {"dimensions": ["stat_var"]}},  # describe_frame
                {
                    "outcome": "ok",
                    "frame_refs": [],
                    "component": {"component_id": "cmp_scripted0002", "revision": 1},
                    "summary": "created",
                },
            ]
        ),
        generative=GenerativeNodes(converse=converse, compose=RecordingCompose(), narrate=narrate),
        tool_surface=None,
    )
    await _start_with_digest(
        workflow,
        "add Guatemala to the chart",
        _digest_with("cmp_scripted0001"),
        auto_add=False,
    )

    assert narrate.facts[-1]["revises_requested"] is False
    assert narrate.facts[-1]["components_replaced"] is False


async def test_narrate_facts_say_revises_requested_is_false_for_a_focus_only_card() -> None:
    """Same end-to-end shape, for the never-composed-focus-card case."""
    converse = RevisesConverse("cmp_scripted0001")
    narrate = _CapturingNarrate()
    workflow = create_root_workflow(
        resolver=ScriptedResolver([_envelope()]),
        executor=ScriptedExecutor(
            [
                {"outcome": "ok", "frame_refs": [REF_A]},  # get_data
                {"outcome": "ok", "view": {"dimensions": ["stat_var"]}},  # describe_frame
                {
                    "outcome": "ok",
                    "frame_refs": [],
                    "component": {"component_id": "cmp_scripted0002", "revision": 1},
                    "summary": "created",
                },
            ]
        ),
        generative=GenerativeNodes(converse=converse, compose=RecordingCompose(), narrate=narrate),
        tool_surface=None,
    )
    await _start_with_digest(
        workflow,
        "add Guatemala to the chart",
        _digest_with("cmp_scripted0001"),
        composed_ids=[],
    )

    assert narrate.facts[-1]["revises_requested"] is False
    assert narrate.facts[-1]["components_replaced"] is False


async def test_no_data_status_with_empty_candidates_uses_no_data_fallback_text() -> None:
    """Without a refusal seam, the `no_data` dead end falls back to its own
    fixed sentence about missing data. This is the case `deadend`'s
    `no_data` fallback text exists for."""
    resolver = ScriptedResolver([_envelope(status="no_data", spec_id=None, candidates=[])])
    executor = ScriptedExecutor()
    _, _, events = await _start(
        create_root_workflow(resolver=resolver, executor=executor, tool_surface=None),
        "does this exist",
    )

    assert "deadend" in _visited(events)
    text = _output_of(events, "deadend")["text"]
    assert text == "I couldn't find data matching that question in the published datasets."


# --- 6/7. planner path and version-bound approval -----------------------------


async def test_choice_points_route_to_planner_and_execute_after_approval() -> None:
    """The scripted seams themselves: `ScriptedPlannerModel` drafts the
    canned plan and `ScriptedSurface` runs it, the same pair `server.py`
    wires for `STUDIO_SCRIPTED=true`."""
    resolver = ScriptedResolver([_envelope(choice_points=2)])
    executor = ScriptedExecutor()
    surface = _CallRecordingSurface(ScriptedSurface())
    runner, session_id, events = await _start(
        create_root_workflow(
            resolver=resolver,
            executor=executor,
            generative=GenerativeNodes(planner=ScriptedPlannerModel()),
            fetch_series_stats=scripted_fetch_series_stats,
            tool_surface=surface,
        ),
        "q with forks",
    )

    assert _pending_interrupt(events) == APPROVE_INTERRUPT
    assert executor.calls == []  # nothing computes before approval
    plan_payload = _output_of(events, "planner")
    assert plan_payload["version"] == 1
    assert plan_payload["status"] == "pending_approval"

    events = await _resume(
        runner, session_id, APPROVE_INTERRUPT, {"approved": plan_payload["revision"]}
    )

    assert [call[0] for call in surface.calls] == ["get_data", "aggregate"]
    assert executor.calls == []  # the no-planner executor never sees a plan step
    assert _output_of(events, "narrate")["plan_version"] == 1


_PREPARING = "Preparing the dataset (in progress, 3s)"
_RECEIVED = "Result received; checking it"


def _step_progress(event: Event) -> dict[str, Any] | None:
    return (event.custom_metadata or {}).get(STEP_PROGRESS_KEY)


class _ReportsThenWaits(ScriptedSurface):
    """`ScriptedSurface`, whose `get_data` sends one progress report and then
    waits for `release` before answering, standing in for data-mcp polling
    a background dfo-apis operation."""

    def __init__(self) -> None:
        super().__init__()
        self.release = asyncio.Event()
        self.waiting: asyncio.Task[Any] | None = None
        self.cancelled = False

    async def call_tool(
        self,
        *,
        tool: ToolId,
        arguments: Mapping[str, Any],
        on_progress: ProgressCallback | None = None,
    ) -> ToolCallResult:
        if tool.name == "get_data":
            assert on_progress is not None
            on_progress(ProgressReport(progress=3, message=_PREPARING))
            self.waiting = asyncio.current_task()
            try:
                await self.release.wait()
            except asyncio.CancelledError:
                self.cancelled = True
                raise
        return await super().call_tool(tool=tool, arguments=arguments, on_progress=on_progress)


async def _approval_stream(surface: ToolSurface) -> Any:
    """The event stream of an approved scripted plan's run against `surface`."""
    runner, session_id, events = await _start(
        create_root_workflow(
            resolver=ScriptedResolver([_envelope(choice_points=2)]),
            executor=ScriptedExecutor(),
            generative=GenerativeNodes(planner=ScriptedPlannerModel()),
            fetch_series_stats=scripted_fetch_series_stats,
            tool_surface=surface,
        ),
        "q with forks",
    )
    part = create_request_input_response(
        APPROVE_INTERRUPT, {"result": {"approved": _output_of(events, "planner")["revision"]}}
    )
    return runner.run_async(
        user_id="u", session_id=session_id, new_message=types.Content(role="user", parts=[part])
    )


async def test_execute_relays_a_report_while_its_step_is_still_running() -> None:
    """The surface answers only once the test has seen the report, so the
    report must reach the event stream while the call is still waiting. Once
    a call answers, its step reports that, so the step's line stops showing
    the server's wait while the run goes on."""
    surface = _ReportsThenWaits()
    stream = await _approval_stream(surface)
    seen: list[Event] = []

    async with asyncio.timeout(10):
        async for event in stream:
            seen.append(event)
            if _step_progress(event) is not None and not surface.release.is_set():
                assert surface.waiting is not None and not surface.waiting.done()
                surface.release.set()

    progress = [_step_progress(event) for event in seen if _step_progress(event)]
    assert progress == [
        {"step_id": "retrieve", "tool": "get_data", "message": _PREPARING},
        {"step_id": "retrieve", "tool": "get_data", "message": _RECEIVED},
        {"step_id": "totals", "tool": "aggregate", "message": _RECEIVED},
    ]
    assert _node_name(next(e for e in seen if _step_progress(e))) == "execute"
    assert "narrate" in _visited(seen)


async def test_stopping_a_turn_mid_step_cancels_the_run_and_leaves_no_task_behind() -> None:
    """Studio stops a turn by cancelling the pending `anext` on its event
    stream (`harness.studio._unwind`). The plan's run is a task of its own,
    so the `execute` node must cancel it rather than leave it running."""
    surface = _ReportsThenWaits()
    stream = await _approval_stream(surface)

    async with asyncio.timeout(10):
        while _step_progress(await anext(stream)) is None:
            pass
        pending = asyncio.ensure_future(anext(stream))
        await asyncio.sleep(0)
        pending.cancel()
        with suppress(asyncio.CancelledError):
            await pending
        await stream.aclose()

    assert surface.waiting is not None and surface.waiting.done()
    assert surface.cancelled


async def test_relay_collapses_many_reports_from_one_step_while_blocked() -> None:
    """`_start_relayed` coalesces at enqueue time, not after: a step that
    reports many times before the reader gets a chance to drain leaves only
    its latest report pending, so a fast producer never grows the relay
    past one entry per distinct step."""

    async def call(on_progress: StepProgressCallback) -> str:
        for i in range(50):
            on_progress("only-step", ProgressReport(progress=i, message=f"tick {i}"))
        return "done"

    task, relay = _start_relayed(call)
    result = await task  # every report above queued before anything drained the relay

    reports = [report async for report in relay.latest()]

    assert len(reports) == 1
    assert reports[0].step_id == "only-step"
    assert reports[0].report == ProgressReport(progress=49, message="tick 49")
    assert result == "done"
    assert {task for task in asyncio.all_tasks() if not task.done()} == {asyncio.current_task()}


class _ReportingExecutor(ScriptedExecutor):
    """Reports one progress report on `get_data`, noting the step it ran under."""

    def __init__(self) -> None:
        super().__init__()
        self.step_ids: list[str | None] = []

    async def __call__(
        self, *, tool: str, args: dict[str, Any], on_progress: ProgressCallback | None = None
    ) -> dict[str, Any]:
        if tool == "get_data":
            assert on_progress is not None
            self.step_ids.append(current_step_id.get())
            on_progress(ProgressReport(progress=3, message=_PREPARING))
        return await super().__call__(tool=tool, args=args, on_progress=on_progress)


async def test_fetch_relays_its_get_data_report_under_the_fetch_step_id() -> None:
    executor = _ReportingExecutor()
    _, _, events = await _start(
        create_root_workflow(
            resolver=ScriptedResolver([_envelope()]), executor=executor, tool_surface=None
        ),
        "kenya oda 2023",
    )

    (event,) = [event for event in events if _step_progress(event)]
    assert _node_name(event) == "fetch"
    assert _step_progress(event) == {
        "step_id": FETCH_STEP_ID,
        "tool": "get_data",
        "message": _PREPARING,
    }
    # The call ran under the same step id, so its record names it too
    # (`harness.studio_translate._RecordingExecutor`), and the node's own
    # context never saw it.
    assert executor.step_ids == [FETCH_STEP_ID]
    assert current_step_id.get() is None
    assert _output_of(events, "fetch")["frame_refs"] == [REF_A]


def _searched_block(spec_id: str, *refs: object) -> dict[str, Any]:
    """The `definite` spec block a resolver carries beside its digest, electing `refs`."""
    return {
        "spec_id": spec_id,
        "resolved": {
            "rendered_sentence": "Total population in Kenya",
            "coverage": {"kind": "exact", "has_data": True},
            "stat_vars": {
                "items": [{"ref": ref, "slot_values": []} for ref in refs],
                "member_count": len(refs),
                "truncated": False,
            },
        },
    }


async def _fetch_args(envelope: dict[str, Any]) -> dict[str, Any]:
    """The arguments `fetch` calls `get_data` with on a turn `envelope` resolves."""
    executor = ScriptedExecutor()
    await _start(
        create_root_workflow(
            resolver=ScriptedResolver([envelope]), executor=executor, tool_surface=None
        ),
        "population of kenya",
    )
    (call,) = executor.calls
    assert call["tool"] == "get_data"
    return call["args"]


@pytest.mark.parametrize(
    ("member_digest", "posted"),
    [
        (None, {"resolved_stat_var_dcids": ["Count_Person", "Count_Person_Female"]}),
        # A family Search states a member digest for is elected again by the
        # mint, so its listed StatVars are exemplars and none is posted.
        ("d", {}),
    ],
    ids=["listed-membership", "digest-family"],
)
async def test_fetch_posts_the_statvars_the_searched_block_elected(
    member_digest: str | None, posted: dict[str, Any]
) -> None:
    block = _searched_block(SPEC_A, "Count_Person", "Count_Person_Female")
    if member_digest is not None:
        block["resolved"]["shape"] = {"member_count": 2, "member_digest": member_digest}

    args = await _fetch_args(_envelope(spec_block=block))

    assert args == {"spec_id": SPEC_A, **posted}


@pytest.mark.parametrize(
    "block",
    [
        None,
        _searched_block(SPEC_A),
        _searched_block(SPEC_B, "Count_Person"),
        _searched_block(SPEC_A, "dcid:Count_Person"),
        _searched_block(SPEC_A, 7),
    ],
    ids=["no-block", "none-elected", "another-spec", "a-prefixed-ref", "a-ref-that-is-no-dcid"],
)
async def test_fetch_posts_no_statvars_it_cannot_read_off_the_block_for_its_spec(
    block: dict[str, Any] | None,
) -> None:
    """`get_data` then resolves the family with nothing posted. A block that
    cannot be read as a source posts nothing: a list cut to the readable
    entries would differ from the one a plan's retrieval posts."""
    assert await _fetch_args(_envelope(spec_block=block)) == {"spec_id": SPEC_A}


async def test_a_refused_fetch_names_the_refusal_and_keeps_no_frame() -> None:
    """A searched turn with no plan: `get_data` refuses, so `fetch` hands `narrate`
    the failed step with the tool's own remedy and no frame to draw."""
    refusal = {
        "code": "over_ceiling",
        "retry_mode": "do_not_retry",
        "fix": {"remedy": "Ask for fewer years."},
        "message": "the frame is past the row ceiling",
    }
    executor = ScriptedExecutor([{"outcome": "refusal", "refusal": refusal}])

    _, _, events = await _start(
        create_root_workflow(
            resolver=ScriptedResolver([_envelope()]), executor=executor, tool_surface=None
        ),
        "population of kenya",
    )

    assert _output_of(events, "fetch") == {
        "frame_refs": [],
        "frame_ref_steps": [],
        "step_summaries": [],
        "failed_step": {
            "step_id": FETCH_STEP_ID,
            "tool": "get_data",
            "code": "over_ceiling",
            "message": "the frame is past the row ceiling",
            "remedy": "Ask for fewer years.",
        },
    }


def test_every_graph_state_field_has_a_default() -> None:
    """A session saved at an interrupt resumes under the current schema, so a
    field it does not hold has to validate from its default."""
    required = [name for name, field in GraphState.model_fields.items() if field.is_required()]

    assert required == []
    assert GraphState().model_dump()["spec_id_stat_vars"] is None
    assert GraphState().model_dump()["seeds"] is None


def _two_candidates() -> dict[str, Any]:
    """A `candidates` answer offering two narrow readings, each with the StatVars `search` elected."""
    return _envelope(
        status="candidates",
        spec_id=None,
        candidates=[
            {"spec_id": SPEC_A, "label": "reading A", "stat_vars": ["Count_Person"]},
            {"spec_id": SPEC_B, "label": "reading B", "stat_vars": ["Count_Person_Female"]},
        ],
    )


@pytest.mark.parametrize(
    ("answer", "posted"),
    [
        (
            {"spec_id": SPEC_B},
            {"spec_id": SPEC_B, "resolved_stat_var_dcids": ["Count_Person_Female"]},
        ),
        ({"skip": True}, {"spec_id": SPEC_A, "resolved_stat_var_dcids": ["Count_Person"]}),
    ],
    ids=["the-chosen-candidate", "the-default-candidate-on-a-skip"],
)
async def test_fetch_posts_the_statvars_search_elected_for_the_candidate_clarify_settled(
    answer: dict[str, Any], posted: dict[str, Any]
) -> None:
    """The re-entry resolve searches nothing and carries no block, so the
    candidate's own elected StatVars are what `fetch` posts."""
    resolver = ScriptedResolver([_two_candidates(), _envelope(spec_id=posted["spec_id"])])
    executor = ScriptedExecutor()
    runner, session_id, events = await _start(
        create_root_workflow(resolver=resolver, executor=executor, tool_surface=None),
        "population of guatemala",
    )
    assert _pending_interrupt(events) == CLARIFY_INTERRUPT
    # The interrupt reaches models, so it names no candidate's StatVars.
    (asked,) = [call.args for call in events[-1].get_function_calls()]
    assert "Count_Person" not in json.dumps(asked)

    await _resume(runner, session_id, CLARIFY_INTERRUPT, answer)

    assert executor.calls == [{"tool": "get_data", "args": posted}]


async def test_fetch_posts_the_statvars_of_the_candidate_the_round_bound_took() -> None:
    """Past `MAX_CLARIFY_ROUNDS` the top candidate is taken, with its own StatVars."""
    resolver = ScriptedResolver(
        [_two_candidates(), _two_candidates(), _two_candidates(), _envelope(spec_id=SPEC_A)]
    )
    executor = ScriptedExecutor()
    runner, session_id, _ = await _start(
        create_root_workflow(resolver=resolver, executor=executor, tool_surface=None),
        "population of guatemala",
    )

    for _ in range(MAX_CLARIFY_ROUNDS):
        await _resume(runner, session_id, CLARIFY_INTERRUPT, {"spec_id": SPEC_B})

    assert executor.calls == [
        {
            "tool": "get_data",
            "args": {"spec_id": SPEC_A, "resolved_stat_var_dcids": ["Count_Person"]},
        }
    ]


async def test_fetch_posts_no_candidate_statvars_for_a_spec_the_re_entry_resolve_renamed() -> None:
    """The list `clarify` stored is the chosen candidate's alone."""
    resolver = ScriptedResolver([_two_candidates(), _envelope(spec_id=SPEC_A)])
    executor = ScriptedExecutor()
    runner, session_id, _ = await _start(
        create_root_workflow(resolver=resolver, executor=executor, tool_surface=None),
        "population of guatemala",
    )

    await _resume(runner, session_id, CLARIFY_INTERRUPT, {"spec_id": SPEC_B})

    assert executor.calls == [{"tool": "get_data", "args": {"spec_id": SPEC_A}}]


async def test_stale_version_approval_routes_to_amend() -> None:
    """A wrong answer routes to amend; `ScriptedPlannerModel` redrafts the
    same canned steps as the next version, bumping `version` and
    `previous_version` the way any `PlannerModel` seam must."""
    resolver = ScriptedResolver([_envelope(choice_points=2)])
    executor = ScriptedExecutor()
    surface = _CallRecordingSurface(ScriptedSurface())
    runner, session_id, events = await _start(
        create_root_workflow(
            resolver=resolver,
            executor=executor,
            generative=GenerativeNodes(planner=ScriptedPlannerModel()),
            fetch_series_stats=scripted_fetch_series_stats,
            tool_surface=surface,
        ),
        "q with forks",
    )
    assert _pending_interrupt(events) == APPROVE_INTERRUPT
    first = _output_of(events, "planner")

    events = await _resume(
        runner, session_id, APPROVE_INTERRUPT, {"approved": "not-a-real-revision"}
    )

    # Wrong answer: nothing executed; the planner redrafted to version 2 and
    # a fresh approval interrupt is pending.
    assert surface.calls == []
    assert _pending_interrupt(events) == APPROVE_INTERRUPT
    amended = _output_of(events, "planner")
    assert amended["version"] == 2
    assert amended["previous_version"] == 1
    assert amended["status"] == "pending_approval"
    assert amended["revision"] != first["revision"]

    events = await _resume(runner, session_id, APPROVE_INTERRUPT, {"approved": amended["revision"]})
    assert [call[0] for call in surface.calls] == ["get_data", "aggregate"]
    assert _output_of(events, "narrate")["plan_version"] == 2


async def test_compound_plan_approved_unchanged_opens_a_span_and_logs_the_facts(
    exported_spans: InMemorySpanExporter, caplog: pytest.LogCaptureFixture
) -> None:
    """A two-part definite envelope with no choice points floors at
    `"standard"` on part count alone (`_COMPOUND_PART_FLOOR`), reaches the
    planner, and gets version 1 approved unchanged -- exactly the case
    `approve`'s `plan.approved` span and log line exist to make queryable."""
    resolver = ScriptedResolver([_envelope(parts=2)])
    executor = ScriptedExecutor()
    surface = _CallRecordingSurface(ScriptedSurface())
    runner, session_id, events = await _start(
        create_root_workflow(
            resolver=resolver,
            executor=executor,
            generative=GenerativeNodes(planner=ScriptedPlannerModel()),
            fetch_series_stats=scripted_fetch_series_stats,
            tool_surface=surface,
        ),
        "population and gdp",
    )

    assert _pending_interrupt(events) == APPROVE_INTERRUPT
    plan_payload = _output_of(events, "planner")
    assert plan_payload["version"] == 1

    with caplog.at_level("INFO", logger="harness.graph"):
        events = await _resume(
            runner, session_id, APPROVE_INTERRUPT, {"approved": plan_payload["revision"]}
        )

    assert _output_of(events, "narrate")["plan_version"] == 1

    spans = [s for s in exported_spans.get_finished_spans() if s.name == "plan.approved"]
    assert len(spans) == 1
    attributes = spans[0].attributes
    assert attributes is not None
    assert attributes["dfo.depth"] == "standard"
    assert attributes["dfo.parts"] == 2
    assert attributes["dfo.choice_points"] == 0
    assert attributes["dfo.degraded"] is False
    assert attributes["dfo.plan_version"] == 1
    assert attributes["dfo.amended"] is False

    record = next(r for r in caplog.records if "plan approved" in r.getMessage())
    assert "parts=2" in record.getMessage()
    assert "choice_points=0" in record.getMessage()
    assert "plan_version=1" in record.getMessage()
    assert "amended=False" in record.getMessage()


# --- 11. the diagram-completion path -------------------------------------------


async def test_compose_output_reaches_narrate_with_refs_verbatim() -> None:
    resolver = ScriptedResolver([_envelope()])
    executor = ScriptedExecutor([{"outcome": "ok", "frame_refs": [REF_A, REF_B]}])
    _, _, events = await _start(
        create_root_workflow(resolver=resolver, executor=executor, tool_surface=None),
        "kenya oda 2023",
    )

    composed = _output_of(events, "compose")
    narrated = _output_of(events, "narrate")
    assert composed["frame_refs"] == [REF_A, REF_B]
    assert narrated["frame_refs"] == [REF_A, REF_B]
    assert narrated["text"] == "Prepared 2 frame(s) via direct fetch."
    assert narrated["depth"] == "none"


class RecordingCompose:
    """Stands in for the real prompt: records what the compose seam was
    given, and only ever drafts a component for a frame whose view is
    present -- it never falls back to a different ref's view when the one
    it wants is missing."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def __call__(
        self,
        *,
        question: str,
        frame_refs: list[dict[str, Any]],
        frame_views: list[dict[str, Any]],
        frame_provenance: list[dict[str, Any]],
    ) -> dict[str, Any]:
        self.calls.append(
            {
                "question": question,
                "frame_refs": frame_refs,
                "frame_views": frame_views,
                "frame_provenance": frame_provenance,
            }
        )
        intents = [
            {
                "type": "stat_card",
                "frame_ref": frame_refs[index],
                "select": {"stat_var": "dcid:x"},
                "title": "final",
            }
            for index, (view, provenance) in enumerate(
                zip(frame_views, frame_provenance, strict=True)
            )
            if provenance.get("is_final_step") and view.get("available")
        ]
        return {"intents": intents}


_FRAME_DO_NOT_RETRY = "this step's data view could not be prepared"
_LEAKY_DETAIL = (
    "spec_id 'ops_9f2c' content_version 'ocv_a83d1e' names an Analysis Toolkit "
    "op-spec frame that was minted but not published to the platform"
)


async def test_refused_describe_frame_on_the_final_step_names_the_gap_instead_of_substituting(
    real_schemas: list[ToolSchema],
) -> None:
    """A refused describe_frame on the compiled path's one produced frame --
    the plan's terminal ("totals") step's, since `execute`'s Accepted branch
    never hands compose an intermediate step's frame -- must reach the
    compose seam as a named, provenance-tagged gap, never a bare `{}`. With
    that gap visible, a seam that only drafts from an available final-step
    frame writes nothing rather than presenting a hole as an answer. The
    gap's `reason` is a closed-vocabulary phrase, never the platform's raw
    refusal text -- see
    test_frame_unavailable_reason_never_carries_the_platform_refusal_text."""
    plan = _seam_plan(schemas=real_schemas)
    seam = _Seam([{"plan": plan.model_dump(mode="json"), "summary": {}}])
    resolver = ScriptedResolver([_envelope(choice_points=2)])
    compose = RecordingCompose()
    executor = ScriptedExecutor(
        [
            {
                "outcome": "refusal",
                "refusal": {
                    "code": "frame_unavailable",
                    "retry_mode": "do_not_retry",
                    "fix": None,
                    "message": _LEAKY_DETAIL,
                },
            },  # describe_frame on the terminal ("totals") step's frame refuses
        ]
    )
    workflow = create_root_workflow(
        resolver=resolver,
        executor=executor,
        generative=GenerativeNodes(planner=seam, compose=compose),
        tool_surface=_fixture_surface(real_schemas),
    )
    runner, session_id, events = await _start(workflow, "aggregate q")
    revision = plan_revision(plan)
    events = await _resume(runner, session_id, APPROVE_INTERRUPT, {"approved": revision})

    assert [c["tool"] for c in executor.calls] == ["describe_frame"]

    (call,) = compose.calls
    assert call["frame_refs"] == [
        {"spec_id": "ops_" + "2" * 32, "content_version": "ocv_" + "3" * 32}
    ]
    assert call["frame_views"][0] == {
        "available": False,
        "reason": "this step's data view could not be prepared",
    }
    assert call["frame_provenance"][0] == {"step": 1, "tool": "aggregate", "is_final_step": True}

    # The gap reached compose, but nothing was drafted to cover for it.
    composed = _output_of(events, "compose")
    assert composed["fact_summaries"] == []
    assert composed["frame_views"][0]["available"] is False

    # narrate sees the same gap, named in plain words, so the closing text
    # can say a step's frame was unavailable without repeating the
    # platform's raw refusal text.
    narrated = _output_of(events, "narrate")
    assert narrated["frame_views"][0]["available"] is False
    assert narrated["frame_views"][0]["reason"] == "this step's data view could not be prepared"


async def test_frame_unavailable_reason_never_carries_the_platform_refusal_text(
    real_schemas: list[ToolSchema],
) -> None:
    """A platform refusal's raw text -- spec_id, content_version, refusal
    code, whatever internal jargon it carries -- must never reach the
    narrate seam's inputs. Only a closed-vocabulary phrase may."""
    plan = _seam_plan(schemas=real_schemas)
    seam = _Seam([{"plan": plan.model_dump(mode="json"), "summary": {}}])
    resolver = ScriptedResolver([_envelope(choice_points=2)])
    compose = RecordingCompose()
    executor = ScriptedExecutor(
        [
            {
                "outcome": "refusal",
                "refusal": {
                    "code": "ops_unsupported",
                    "retry_mode": "do_not_retry",
                    "fix": None,
                    "message": _LEAKY_DETAIL,
                },
            },  # describe_frame refuses with the platform's raw problem detail
        ]
    )
    workflow = create_root_workflow(
        resolver=resolver,
        executor=executor,
        generative=GenerativeNodes(planner=seam, compose=compose),
        tool_surface=_fixture_surface(real_schemas),
    )
    runner, session_id, events = await _start(workflow, "aggregate q")
    revision = plan_revision(plan)
    events = await _resume(runner, session_id, APPROVE_INTERRUPT, {"approved": revision})

    narrated = _output_of(events, "narrate")
    narrate_input = json.dumps(narrated)
    assert "ops_9f2c" not in narrate_input
    assert "ocv_a83d1e" not in narrate_input
    assert "ops_unsupported" not in narrate_input
    assert _LEAKY_DETAIL not in narrate_input
    assert narrated["frame_views"][0]["reason"] == "this step's data view could not be prepared"


@pytest.mark.parametrize(
    ("retry_mode", "reason"),
    [
        pytest.param(["retry_after_delay"], _FRAME_DO_NOT_RETRY, id="list"),
        pytest.param({"mode": 1}, _FRAME_DO_NOT_RETRY, id="dict"),
        pytest.param(None, _FRAME_DO_NOT_RETRY, id="none"),
        pytest.param(
            "retry_after_delay", "this step's data view is temporarily unavailable", id="string"
        ),
    ],
)
def test_a_frame_unavailable_reason_reads_a_retry_mode_of_any_type(
    retry_mode: Any, reason: str
) -> None:
    assert _frame_unavailable_reason({"retry_mode": retry_mode}) == reason


class RawFallbackCompose:
    """A deliberately NONCOMPLIANT compose seam. Ignores `frame_provenance`
    and `frame_views` entirely and always drafts from the first frame it is
    given -- exactly the defect `_admissible_intents` exists to fix: given an
    intermediate step's frame and the plan's real final-step frame side by
    side, a model that ignores the instruction picks the earlier, raw one.
    This fake stands in for that model to prove the guard rejects it in
    code, not just in the prompt."""

    async def __call__(
        self,
        *,
        question: str,
        frame_refs: list[dict[str, Any]],
        frame_views: list[dict[str, Any]],
        frame_provenance: list[dict[str, Any]],
    ) -> dict[str, Any]:
        return {
            "intents": [
                {
                    "type": "stat_card",
                    "frame_ref": frame_refs[0],
                    "select": {"stat_var": "dcid:x"},
                    "title": "raw fallback",
                }
            ]
        }


def _seam_plan_with_intermediate_step(*, schemas: list[ToolSchema]) -> PlannerPlan:
    """A real, three-step plan: get_data -> totals (aggregate, now
    intermediate) -> change (growth, the one terminal step).

    `_seam_plan`'s two-step plan cannot exercise the finality guard on the
    replan-exhausted fallback: when its one non-terminal step (get_data) is
    the only one to complete before a failure, it is trivially "the last one
    that ran" whether or not `frame_ref_steps` carries real per-step
    provenance. A third step lets two steps -- one of them genuinely
    intermediate -- complete before the terminal one fails, giving the guard
    an actual earlier step to reject.
    """
    decision = Decision(
        slot="window",
        description="Which years?",
        options=["2015-2024", "2018-2024"],
        chosen="2015-2024",
        resolved_by=ResolvedBy.PLANNER_DEFAULT,
        rationale="a full decade covers the series",
    )
    dataset = DatasetRequirement(
        id="spending",
        decision_ids=["window"],
        measure="health spending",
        entities=[Entity(name="Kenya", role="recipient")],
        period=Period(start=2015, end=2024),
    )
    totals = AggregateRequirement(
        id="totals",
        decision_ids=["window"],
        input_id="spending",
        reduction=Reduction.SUM,
        group_by=(GroupBy.PERIOD,),
    )
    change = GrowthRequirement(
        id="change",
        decision_ids=["window"],
        input_id="totals",
        window_start=2015,
        window_end=2024,
    )
    draft = PlannerPlan(
        question="kenya debt growth",
        decisions=[decision],
        evidence_requirements=[dataset, totals, change],
        version=1,
    )
    translation = Translation(
        operations=[
            RetrieveOperation(id="get_data", requirement_id=dataset.id, spec_id=SPEC_A),
            AggregateOperation(
                id="totals",
                requirement_id=totals.id,
                input="get_data",
                reduction=Reduction.SUM,
                group_by=(GroupBy.PERIOD,),
            ),
            GrowthOperation(
                id="change",
                requirement_id=change.id,
                input="totals",
                window_start=2015,
                window_end=2024,
            ),
        ],
        terminal=["change"],
    )
    named = {schema.name: schema for schema in schemas}
    get_data_schema = named["get_data"]
    aggregate_schema = named["aggregate"]
    growth_schema = named["growth"]
    workflow = Workflow(
        steps=[
            WorkflowStep(
                id="get_data",
                operation_id="get_data",
                tool=ToolId(server="data-mcp", name="get_data"),
                args={"spec_id": SPEC_A},
                input_schema_hash=get_data_schema.input_schema_hash,
                output_schema_hash=get_data_schema.output_schema_hash,
                implementation_version=get_data_schema.version,
            ),
            WorkflowStep(
                id="totals",
                operation_id="totals",
                tool=ToolId(server="analysis-mcp", name="aggregate"),
                args={"aggregation": "sum", "group_by": ["time_period"]},
                bindings={"frame_ref": {"from_step": "get_data", "output": "frame_ref"}},
                input_schema_hash=aggregate_schema.input_schema_hash,
                output_schema_hash=aggregate_schema.output_schema_hash,
                implementation_version=aggregate_schema.version,
            ),
            WorkflowStep(
                id="change",
                operation_id="change",
                tool=ToolId(server="analysis-mcp", name="growth"),
                args={"window_start": "2015", "window_end": "2024", "entity_by": []},
                bindings={"frame_ref": {"from_step": "totals", "output": "frame_ref"}},
                input_schema_hash=growth_schema.input_schema_hash,
                output_schema_hash=growth_schema.output_schema_hash,
                implementation_version=growth_schema.version,
            ),
        ],
        sources=[
            SelectedSource(spec_id=SPEC_A, sentence="health spending in Kenya", has_data=True)
        ],
        evidence_bindings=[],
        terminal_steps=["change"],
    )
    return draft.with_translation(translation=translation, workflow=workflow)


_CHANGE_REFUSAL = ToolCallResult(
    is_error=True,
    structured_content={
        "code": "entity_ambiguous",
        "retry_mode": "correct_and_retry",
        "message": "more than one entity matches 'Kenya'",
        "fix": {"candidates": ["country/KEN", "country/KEN-old"]},
    },
    text="refused",
)


def _intermediate_step_surface(real_schemas: list[ToolSchema]) -> FixtureSurface:
    """get_data -> totals succeed; change (growth), the plan's one terminal
    step, always refuses -- so a run against this surface completes two
    steps, one of them genuinely intermediate, before exhausting the replan
    budget."""
    named = {schema.name: schema for schema in real_schemas}
    return FixtureSurface(
        schemas=[named["get_data"], named["aggregate"], named["growth"]],
        responses={
            "get_data": ToolCallResult(
                is_error=False, structured_content=_get_data_manifest(), text="fetched"
            ),
            "aggregate": ToolCallResult(
                is_error=False, structured_content=_analysis_result(), text="computed"
            ),
            "growth": _CHANGE_REFUSAL,
        },
    )


async def test_a_noncompliant_compose_seam_never_gets_its_raw_fallback_written(
    real_schemas: list[ToolSchema], caplog: pytest.LogCaptureFixture
) -> None:
    """The guard, not the prompt, is what stops the bug: given an
    intermediate step's frame (get_data's) and the plan's terminal step's
    frame side by side -- reached here once growth fails past the replan
    budget, leaving both earlier steps' frames in `completed` -- a compose
    seam that picks the earlier, non-final one must never reach
    `write_component`. Without a deterministic check in `harness.graph`, a
    noncompliant seam's output would be written verbatim, letting an
    intermediate frame stand in for the plan's answer."""
    plan = _seam_plan_with_intermediate_step(schemas=real_schemas)
    seam = _Seam([{"plan": plan.model_dump(mode="json"), "summary": {}}])
    resolver = ScriptedResolver([_envelope(choice_points=2)])
    executor = ScriptedExecutor(
        [
            {"outcome": "ok", "view": {"dimensions": ["time_period"]}},  # describe get_data's frame
            {"outcome": "ok", "view": {"dimensions": ["stat_var"]}},  # describe totals' frame
        ]
    )
    workflow = create_root_workflow(
        resolver=resolver,
        executor=executor,
        generative=GenerativeNodes(planner=seam, compose=RawFallbackCompose()),
        tool_surface=_intermediate_step_surface(real_schemas),
    )
    runner, session_id, events = await _start(workflow, "kenya debt growth")
    revision = plan_revision(plan)

    events = await _resume(runner, session_id, APPROVE_INTERRUPT, {"approved": revision})
    assert _pending_interrupt(events) == APPROVE_INTERRUPT
    replanned = _output_of(events, "planner")
    with caplog.at_level("WARNING", logger="harness.graph"):
        events = await _resume(
            runner, session_id, APPROVE_INTERRUPT, {"approved": replanned["revision"]}
        )

    executed = _output_of(events, "execute")
    assert executed["failed_step"]["step_id"] == "change"
    # get_data and totals both ran and completed before growth's refusal;
    # both frames -- one genuinely intermediate -- reach compose.
    assert len(executed["frame_refs"]) == 2

    # Only the two describe_frame calls follow execution -- no
    # write_component call was ever made for the rejected intent.
    assert [c["tool"] for c in executor.calls] == ["describe_frame", "describe_frame"]
    composed = _output_of(events, "compose")
    assert composed["fact_summaries"] == []

    # The rejection is visible, not silently dropped.
    assert any("rejected an intent" in record.getMessage() for record in caplog.records)


async def test_a_compliant_compose_seam_writes_no_component_once_the_replan_budget_is_spent(
    real_schemas: list[ToolSchema],
) -> None:
    """The replan-exhausted fallback never reaches a terminal step, so no
    entry in `frame_ref_steps` is final -- not even `totals`, the last step
    to complete before `change` (the plan's real terminal step) kept
    refusing. A seam that (correctly) drafts only from frames marked final
    gets nothing past `_admissible_intents`: the turn reports the failure
    through `failed_step` rather than a card drawn from an intermediate
    frame."""
    plan = _seam_plan_with_intermediate_step(schemas=real_schemas)
    seam = _Seam([{"plan": plan.model_dump(mode="json"), "summary": {}}])
    resolver = ScriptedResolver([_envelope(choice_points=2)])
    executor = ScriptedExecutor(
        [
            {"outcome": "ok", "view": {"dimensions": ["time_period"]}},  # describe get_data's frame
            {"outcome": "ok", "view": {"dimensions": ["stat_var"]}},  # describe totals' frame
        ]
    )
    workflow = create_root_workflow(
        resolver=resolver,
        executor=executor,
        generative=GenerativeNodes(planner=seam, compose=RecordingCompose()),
        tool_surface=_intermediate_step_surface(real_schemas),
    )
    runner, session_id, events = await _start(workflow, "kenya debt growth")
    revision = plan_revision(plan)

    events = await _resume(runner, session_id, APPROVE_INTERRUPT, {"approved": revision})
    replanned = _output_of(events, "planner")
    events = await _resume(
        runner, session_id, APPROVE_INTERRUPT, {"approved": replanned["revision"]}
    )

    executed = _output_of(events, "execute")
    assert executed["failed_step"]["step_id"] == "change"
    assert all(not step["is_final_step"] for step in executed["frame_ref_steps"])

    # No card drawn: only the two describe_frame calls happen, write_component never does.
    assert [c["tool"] for c in executor.calls] == ["describe_frame", "describe_frame"]
    composed = _output_of(events, "compose")
    assert composed["fact_summaries"] == []


def _seam_plan_with_two_terminal_steps(*, schemas: list[ToolSchema]) -> PlannerPlan:
    """A real plan with two independent terminal steps reading the same
    retrieval directly: `totals` (aggregate) and `change` (growth), at
    different positions in `outcome.steps` (`workflow.steps` is
    `[get_data, totals, change]`, so `totals` sits at index 1 and `change`
    at index 2). `Translation.terminal` is a tuple with `min_length=1`, not
    `max_length=1` (`planner.operations.py`), so a plan answering through
    more than one terminal step is a real shape, not a hypothetical one --
    and `is_final_step` must mark both `totals` and `change` final, since
    neither is any more the question's answer than the other."""
    decision = Decision(
        slot="window",
        description="Which years?",
        options=["2015-2024", "2018-2024"],
        chosen="2015-2024",
        resolved_by=ResolvedBy.PLANNER_DEFAULT,
        rationale="a full decade covers the series",
    )
    dataset = DatasetRequirement(
        id="spending",
        decision_ids=["window"],
        measure="health spending",
        entities=[Entity(name="Kenya", role="recipient")],
        period=Period(start=2015, end=2024),
    )
    totals = AggregateRequirement(
        id="totals",
        decision_ids=["window"],
        input_id="spending",
        reduction=Reduction.SUM,
        group_by=(GroupBy.PERIOD,),
    )
    change = GrowthRequirement(
        id="change",
        decision_ids=["window"],
        input_id="spending",
        window_start=2015,
        window_end=2024,
    )
    draft = PlannerPlan(
        question="kenya debt totals and growth",
        decisions=[decision],
        evidence_requirements=[dataset, totals, change],
        version=1,
    )
    translation = Translation(
        operations=[
            RetrieveOperation(id="get_data", requirement_id=dataset.id, spec_id=SPEC_A),
            AggregateOperation(
                id="totals",
                requirement_id=totals.id,
                input="get_data",
                reduction=Reduction.SUM,
                group_by=(GroupBy.PERIOD,),
            ),
            GrowthOperation(
                id="change",
                requirement_id=change.id,
                input="get_data",
                window_start=2015,
                window_end=2024,
            ),
        ],
        terminal=["totals", "change"],
    )
    named = {schema.name: schema for schema in schemas}
    get_data_schema = named["get_data"]
    aggregate_schema = named["aggregate"]
    growth_schema = named["growth"]
    workflow = Workflow(
        steps=[
            WorkflowStep(
                id="get_data",
                operation_id="get_data",
                tool=ToolId(server="data-mcp", name="get_data"),
                args={"spec_id": SPEC_A},
                input_schema_hash=get_data_schema.input_schema_hash,
                output_schema_hash=get_data_schema.output_schema_hash,
                implementation_version=get_data_schema.version,
            ),
            WorkflowStep(
                id="totals",
                operation_id="totals",
                tool=ToolId(server="analysis-mcp", name="aggregate"),
                args={"aggregation": "sum", "group_by": ["time_period"]},
                bindings={"frame_ref": {"from_step": "get_data", "output": "frame_ref"}},
                input_schema_hash=aggregate_schema.input_schema_hash,
                output_schema_hash=aggregate_schema.output_schema_hash,
                implementation_version=aggregate_schema.version,
            ),
            WorkflowStep(
                id="change",
                operation_id="change",
                tool=ToolId(server="analysis-mcp", name="growth"),
                args={"window_start": "2015", "window_end": "2024", "entity_by": []},
                bindings={"frame_ref": {"from_step": "get_data", "output": "frame_ref"}},
                input_schema_hash=growth_schema.input_schema_hash,
                output_schema_hash=growth_schema.output_schema_hash,
                implementation_version=growth_schema.version,
            ),
        ],
        sources=[
            SelectedSource(spec_id=SPEC_A, sentence="health spending in Kenya", has_data=True)
        ],
        evidence_bindings=[],
        terminal_steps=["totals", "change"],
    )
    return draft.with_translation(translation=translation, workflow=workflow)


async def test_two_terminal_steps_both_survive_the_finality_guard(
    real_schemas: list[ToolSchema],
) -> None:
    """`totals` (step index 1) and `change` (step index 2) are both
    terminal. A compose that inferred finality from `max(step)` would mark
    only `change` final and wrongly suppress `totals`'s card; `is_final_step`
    is instead stated by `execute`'s Accepted branch on each entry, so a
    card survives `_admissible_intents` for both."""
    plan = _seam_plan_with_two_terminal_steps(schemas=real_schemas)
    seam = _Seam([{"plan": plan.model_dump(mode="json"), "summary": {}}])
    resolver = ScriptedResolver([_envelope(choice_points=2)])
    named = {schema.name: schema for schema in real_schemas}
    surface = FixtureSurface(
        schemas=[named["get_data"], named["aggregate"], named["growth"]],
        responses={
            "get_data": ToolCallResult(
                is_error=False, structured_content=_get_data_manifest(), text="fetched"
            ),
            "aggregate": ToolCallResult(
                is_error=False, structured_content=_analysis_result(), text="computed"
            ),
            "growth": ToolCallResult(
                is_error=False,
                # A distinct spec_id: totals and change must be genuinely
                # different frames, or `_admissible_intents`'s frame_ref
                # lookup collapses both intents onto the same index.
                structured_content=_analysis_result(spec_id="ops_" + "4" * 32),
                text="computed",
            ),
        },
    )
    executor = ScriptedExecutor(
        [
            {"outcome": "ok", "view": {"dimensions": ["stat_var"]}},  # describe totals' frame
            {"outcome": "ok", "view": {"dimensions": ["stat_var"]}},  # describe change's frame
            {"outcome": "ok", "component": _valid_component("cmp_totals")},
            {"outcome": "ok", "component": _valid_component("cmp_change")},
        ]
    )
    workflow = create_root_workflow(
        resolver=resolver,
        executor=executor,
        generative=GenerativeNodes(planner=seam, compose=RecordingCompose()),
        tool_surface=surface,
    )
    runner, session_id, events = await _start(workflow, "kenya debt totals and growth")
    revision = plan_revision(plan)
    events = await _resume(runner, session_id, APPROVE_INTERRUPT, {"approved": revision})

    executed = _output_of(events, "execute")
    assert len(executed["frame_refs"]) == 2

    # Both frames were described, and both survived the guard: two
    # write_component calls, not one.
    assert [c["tool"] for c in executor.calls] == [
        "describe_frame",
        "describe_frame",
        "write_component",
        "write_component",
    ]
    composed = _output_of(events, "compose")
    assert [fs["component_id"] for fs in composed["fact_summaries"]] == ["cmp_totals", "cmp_change"]


_WRITE_REFUSAL_RESULT: dict[str, Any] = {
    "outcome": "refusal",
    "refusal": {
        "code": "component_rejected",
        "retry_mode": "do_not_retry",
        "origin": "service",
        "fix": None,
        "message": "the platform rejected this component",
    },
}


async def test_a_refused_write_component_lands_in_refused_writes_not_a_card() -> None:
    """A platform refusal on `write_component` -- the real
    `POST /api/presentation/components -> 400` failure mode -- must not be
    dropped the way an `_admissible_intents` rejection is not: it is logged
    and carried forward in `refused_writes` rather than the turn quietly
    drawing a smaller canvas, and no card is written for it."""
    resolver = ScriptedResolver([_envelope()])
    executor = ScriptedExecutor(
        [
            {"outcome": "ok", "frame_refs": [REF_A]},  # get_data
            {"outcome": "ok", "view": {"dimensions": ["stat_var"]}},  # describe_frame
            _WRITE_REFUSAL_RESULT,  # write_component
        ]
    )
    workflow = create_root_workflow(
        resolver=resolver,
        executor=executor,
        generative=GenerativeNodes(compose=RecordingCompose()),
        tool_surface=None,
    )
    _, _, events = await _start(workflow, "kenya oda 2023")

    assert [c["tool"] for c in executor.calls] == ["get_data", "describe_frame", "write_component"]
    composed = _output_of(events, "compose")
    assert composed["fact_summaries"] == []
    assert composed["refused_writes"] == [
        {
            "type": "stat_card",
            "title": "final",
            "reason": "the card could not be added",
        }
    ]


async def test_refused_write_reaches_narrates_facts() -> None:
    """`narrate`'s facts must carry `refused_writes` through unchanged, the
    same as `failed_step`, so the closing text can name a refused write
    instead of a chart that simply is not there."""
    captured: list[dict[str, Any]] = []

    class RecordingNarrate:
        async def __call__(self, *, question: str, facts: dict[str, Any]) -> dict[str, Any]:
            captured.append(facts)
            return {"text": ""}

    resolver = ScriptedResolver([_envelope()])
    executor = ScriptedExecutor(
        [
            {"outcome": "ok", "frame_refs": [REF_A]},  # get_data
            {"outcome": "ok", "view": {"dimensions": ["stat_var"]}},  # describe_frame
            _WRITE_REFUSAL_RESULT,  # write_component
        ]
    )
    workflow = create_root_workflow(
        resolver=resolver,
        executor=executor,
        generative=GenerativeNodes(compose=RecordingCompose(), narrate=RecordingNarrate()),
        tool_surface=None,
    )
    await _start(workflow, "kenya oda 2023")

    (facts,) = captured
    assert facts["refused_writes"] == [
        {
            "type": "stat_card",
            "title": "final",
            "reason": "the card could not be added",
        }
    ]


_LEAKY_WRITE_REFUSAL_RESULT: dict[str, Any] = {
    "outcome": "refusal",
    "refusal": {
        "code": "binding_invalid",
        "retry_mode": "do_not_retry",
        "origin": "service",
        "fix": "pin observationAbout to country/KEN before retrying",
        "message": _LEAKY_DETAIL,
    },
}


async def test_refused_write_reason_never_carries_the_platform_refusal_text_or_fix() -> None:
    """The record that reaches narrate's facts carries only the bucketed,
    closed-vocabulary `reason` -- never the refusal's raw `message` (which
    can hold a spec_id or content_version, as here) and never its `fix`,
    the same boundary `_frame_unavailable_reason` already holds for a
    `describe_frame` refusal."""
    captured: list[dict[str, Any]] = []

    class RecordingNarrate:
        async def __call__(self, *, question: str, facts: dict[str, Any]) -> dict[str, Any]:
            captured.append(facts)
            return {"text": ""}

    resolver = ScriptedResolver([_envelope()])
    executor = ScriptedExecutor(
        [
            {"outcome": "ok", "frame_refs": [REF_A]},  # get_data
            {"outcome": "ok", "view": {"dimensions": ["stat_var"]}},  # describe_frame
            _LEAKY_WRITE_REFUSAL_RESULT,  # write_component
        ]
    )
    workflow = create_root_workflow(
        resolver=resolver,
        executor=executor,
        generative=GenerativeNodes(compose=RecordingCompose(), narrate=RecordingNarrate()),
        tool_surface=None,
    )
    await _start(workflow, "kenya oda 2023")

    (facts,) = captured
    assert facts["refused_writes"] == [
        {
            "type": "stat_card",
            "title": "final",
            "reason": "the selection did not identify a single series",
        }
    ]
    facts_text = json.dumps(facts)
    assert _LEAKY_DETAIL not in facts_text
    assert "pin observationAbout" not in facts_text
    assert "binding_invalid" not in facts_text


# --- who declined: the refused-write reason names the right party -------------

_RELEASE_JSON = (
    Path(__file__).parents[2] / "presentation-mcp" / "tests" / "fixtures" / "release.json"
)
_RELEASE_CODES = sorted(json.loads(_RELEASE_JSON.read_text())["retry_modes"])
# Codes a `write_component` refusal can carry that `release.json` does not
# list: the platform's publish-door and transport codes, an unmapped code, and
# the ones the harness mints itself.
_OTHER_SERVICE_CODES = [
    "unsupported_schema_version",
    "spec_not_found",
    "snapshot_not_found",
    "snapshot_revoked",
    "artifact_expired",
    "artifact_not_found",
    "publish_not_permitted",
    "publish_not_configured",
    "transient_unavailable",
    "upstream_error",
    "unspecified",
    "component_rejected",
]
_HARNESS_CODES = [
    "provenance_unavailable",
    "resolved_revision_unavailable",
    "unknown_tool",
    "component_missing",
    "component_invalid",
    "select_invalid",
    "series_limit_exceeded",
]
_RETRY_MODES = [
    None,
    "do_not_retry",
    "correct_and_retry",
    "refresh_then_retry",
    "retry_after_delay",
]
_ORIGINS = [None, "service", "harness", "unrecognised"]
_INTERNAL_TOKEN = re.compile(r"spec_|cv_|ocv_|ops_")

_DEFAULT_REASON = "the card could not be added"
_WITHHELD_REASON = "the card was withheld because no citable source could be tied to this data view"
_TOO_MANY_SERIES_REASON = (
    "the card was withheld because its selection covers more than 24 series, "
    "the most one card draws"
)


def _refusal(code: str, *, origin: str | None, retry_mode: str | None) -> dict[str, Any]:
    refusal: dict[str, Any] = {"code": code, "message": _LEAKY_DETAIL, "fix": None}
    if origin is not None:
        refusal["origin"] = origin
    if retry_mode is not None:
        refusal["retry_mode"] = retry_mode
    return refusal


def test_the_refused_write_vocabulary_holds_fifteen_strings_with_no_internal_token() -> None:
    assert len(REFUSED_WRITE_VOCABULARY) == 15
    for reason in REFUSED_WRITE_VOCABULARY:
        assert not _INTERNAL_TOKEN.search(reason), reason


@pytest.mark.parametrize("origin", _ORIGINS)
@pytest.mark.parametrize("retry_mode", _RETRY_MODES)
@pytest.mark.parametrize("code", [*_RELEASE_CODES, *_OTHER_SERVICE_CODES, *_HARNESS_CODES])
def test_every_refusal_reads_as_a_member_of_the_closed_vocabulary_and_leaks_nothing(
    code: str, retry_mode: str | None, origin: str | None
) -> None:
    refusal = _refusal(code, origin=origin, retry_mode=retry_mode)

    reason = _refused_write_reason(refusal)

    assert reason in REFUSED_WRITE_VOCABULARY
    assert code not in reason
    assert not _INTERNAL_TOKEN.search(reason)
    assert _LEAKY_DETAIL not in reason


@pytest.mark.parametrize("retry_mode", _RETRY_MODES)
@pytest.mark.parametrize("code", [*_RELEASE_CODES, *_OTHER_SERVICE_CODES, *_HARNESS_CODES])
def test_a_refusal_the_harness_minted_is_never_blamed_on_the_platform_or_the_canvas(
    code: str, retry_mode: str | None
) -> None:
    reason = _refused_write_reason(_refusal(code, origin="harness", retry_mode=retry_mode))

    assert reason in {
        _WITHHELD_REASON,
        _TOO_MANY_SERIES_REASON,
        "the assistant could not build the card",
    }
    for named in ("platform", "canvas", "declined"):
        assert named not in reason


@pytest.mark.parametrize("retry_mode", _RETRY_MODES)
@pytest.mark.parametrize(
    "code",
    [*_RELEASE_CODES, *_OTHER_SERVICE_CODES, *_HARNESS_CODES, "an_unseen_code"],
)
@pytest.mark.parametrize("origin", [None, "unrecognised"])
def test_a_refusal_with_no_recognised_origin_blames_nobody_for_any_code_and_retry_mode(
    code: str, origin: str | None, retry_mode: str | None
) -> None:
    """A producer that forgets to stamp `origin`, or stamps one no executor
    uses, lands on the neutral default for a table code, a harness code, an
    unseen code and every retry mode alike."""
    assert (
        _refused_write_reason(_refusal(code, origin=origin, retry_mode=retry_mode))
        == _DEFAULT_REASON
    )


@pytest.mark.parametrize(
    "refusal",
    [
        pytest.param(
            {"code": "constraint_violation", "retry_mode": "correct_and_retry"},
            id="unstamped-table-code",
        ),
        pytest.param(
            {
                "code": "constraint_violation",
                "origin": "unrecognised",
                "retry_mode": "do_not_retry",
            },
            id="unrecognised-origin-table-code",
        ),
        pytest.param(
            {"code": "request_invalid", "retry_mode": "retry_after_delay"},
            id="unstamped-retry-after-delay",
        ),
        pytest.param(
            {
                "code": "request_invalid",
                "origin": "unrecognised",
                "retry_mode": "retry_after_delay",
            },
            id="unrecognised-origin-retry-after-delay",
        ),
        pytest.param(
            {"code": "frame_unavailable", "retry_mode": "refresh_then_retry"},
            id="unstamped-frame-unavailable",
        ),
        pytest.param(
            {
                "code": "frame_unavailable",
                "origin": "unrecognised",
                "retry_mode": "refresh_then_retry",
            },
            id="unrecognised-origin-frame-unavailable",
        ),
        pytest.param(
            {"code": "provenance_unavailable", "retry_mode": "do_not_retry"},
            id="unstamped-harness-code",
        ),
    ],
)
def test_a_refusal_with_no_recognised_origin_never_reaches_a_class_that_names_a_party(
    refusal: dict[str, Any],
) -> None:
    reason = _refused_write_reason(refusal)

    assert reason == _DEFAULT_REASON
    for named in ("platform", "canvas", "assistant", "withheld", "declined"):
        assert named not in reason


@pytest.mark.parametrize(
    "refusal",
    [
        pytest.param({"code": 7, "origin": ["service"], "retry_mode": {}}, id="non-string-all"),
        pytest.param({"code": None, "origin": 3, "retry_mode": None}, id="none-and-int"),
        pytest.param({"code": {"a": 1}, "origin": "service", "retry_mode": []}, id="service"),
        pytest.param({"code": ["x"], "origin": "harness", "retry_mode": 5}, id="harness"),
    ],
)
def test_a_refusal_with_non_string_fields_reads_without_raising(refusal: dict[str, Any]) -> None:
    assert _refused_write_reason(refusal) in REFUSED_WRITE_VOCABULARY


@pytest.mark.parametrize(
    ("refusal", "reason"),
    [
        pytest.param(
            {"code": "binding_invalid", "origin": "service", "retry_mode": "correct_and_retry"},
            "the selection did not identify a single series",
            id="selection",
        ),
        pytest.param(
            {"code": "schema_invalid", "origin": "service", "retry_mode": "do_not_retry"},
            "the component request was not a shape the canvas accepts",
            id="shape",
        ),
        pytest.param(
            {
                "code": "constraint_violation",
                "origin": "service",
                "retry_mode": "correct_and_retry",
            },
            "the platform's chart rules do not allow that chart for this data",
            id="rules",
        ),
        pytest.param(
            {"code": "resource_exhausted", "origin": "service", "retry_mode": "do_not_retry"},
            "the chart asked for more than the platform allows on one card",
            id="too-large",
        ),
        pytest.param(
            {"code": "request_invalid", "origin": "service", "retry_mode": "do_not_retry"},
            "the platform could not accept the component request",
            id="rejected",
        ),
        pytest.param(
            {"code": "frame_unavailable", "origin": "service", "retry_mode": "refresh_then_retry"},
            "the platform could not load this data view; asking again re-runs the query",
            id="data-view-refresh",
        ),
        pytest.param(
            {"code": "frame_unavailable", "origin": "service", "retry_mode": "do_not_retry"},
            "the platform could not load this data view",
            id="data-view-do-not-retry",
        ),
        pytest.param(
            {"code": "snapshot_revoked", "origin": "service", "retry_mode": "do_not_retry"},
            "the platform could not load this data view",
            id="data-view-snapshot-revoked",
        ),
        pytest.param(
            {"code": "artifact_expired", "origin": "service"},
            "the platform could not load this data view",
            id="data-view-no-retry-mode",
        ),
        pytest.param(
            {"code": "frame_unavailable", "origin": "service", "retry_mode": "retry_after_delay"},
            "the platform was temporarily unavailable; asking again may work",
            id="temporary-beats-the-code",
        ),
        pytest.param(
            {
                "code": "service_unavailable",
                "origin": "service",
                "retry_mode": "retry_after_delay",
                "fix": {
                    "service": "presentation-mcp",
                    "rows_requested": 250_000,
                    "rows_free": 200_000,
                    "rows_in_flight_max": 1_200_000,
                },
            },
            "the assistant was busy with other large requests; asking again shortly may work",
            id="row-budget-full",
        ),
        pytest.param(
            {
                "code": "transient_unavailable",
                "origin": "service",
                "retry_mode": "retry_after_delay",
                "fix": None,
            },
            "the platform was temporarily unavailable; asking again may work",
            id="platform-unreachable",
        ),
        pytest.param(
            {
                "code": "service_unavailable",
                "origin": "service",
                "retry_mode": "retry_after_delay",
                "fix": {"service": "dfo-apis", "spec_id": "s", "elapsed_s": 180.0},
            },
            "the platform was temporarily unavailable; asking again may work",
            id="upstream-unavailable-names-no-rows",
        ),
        pytest.param(
            {"code": "transient_unavailable", "origin": "service", "retry_mode": "do_not_retry"},
            "the platform did not confirm that the card was added",
            id="unconfirmed",
        ),
        pytest.param(
            {"code": "publish_not_configured", "origin": "service", "retry_mode": "do_not_retry"},
            "this deployment is not set up to present that kind of result",
            id="not-configured",
        ),
        pytest.param(
            {"code": "provenance_unavailable", "origin": "harness", "retry_mode": "do_not_retry"},
            _WITHHELD_REASON,
            id="unsourced",
        ),
        pytest.param(
            {
                "code": "provenance_unavailable",
                "origin": "harness",
                "retry_mode": "retry_after_delay",
            },
            _WITHHELD_REASON,
            id="harness-origin-beats-the-retry-mode",
        ),
        pytest.param(
            {"code": "unknown_tool", "origin": "harness", "retry_mode": "do_not_retry"},
            "the assistant could not build the card",
            id="internal",
        ),
        pytest.param(
            {"code": "internal_error", "origin": "service", "retry_mode": "do_not_retry"},
            _DEFAULT_REASON,
            id="unmapped-service-code",
        ),
        pytest.param({}, _DEFAULT_REASON, id="empty-refusal"),
    ],
)
def test_the_refused_write_reason_by_class(refusal: dict[str, Any], reason: str) -> None:
    assert _refused_write_reason(refusal) == reason


class _CapturedEvents:
    """Replaces `harness.graph.record_event` and keeps what it was given."""

    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, Any]]] = []

    def __call__(
        self, name: str, *, level: str = "WARNING", metadata: Mapping[str, Any] | None = None
    ) -> None:
        self.events.append((name, dict(metadata or {})))


class _CapturingNarrate:
    def __init__(self) -> None:
        self.facts: list[dict[str, Any]] = []

    async def __call__(self, *, question: str, facts: dict[str, Any]) -> dict[str, Any]:
        self.facts.append(facts)
        return {"text": ""}


async def _run_refused_write(
    monkeypatch: pytest.MonkeyPatch, *, write_result: dict[str, Any]
) -> tuple[_CapturedEvents, _CapturingNarrate, list[Event]]:
    captured = _CapturedEvents()
    monkeypatch.setattr(graph_module, "record_event", captured)
    narrate = _CapturingNarrate()
    workflow = create_root_workflow(
        resolver=ScriptedResolver([_envelope()]),
        executor=ScriptedExecutor(
            [
                {"outcome": "ok", "frame_refs": [REF_A]},  # get_data
                {"outcome": "ok", "view": {"dimensions": ["stat_var"]}},  # describe_frame
                write_result,  # write_component
            ]
        ),
        generative=GenerativeNodes(compose=RecordingCompose(), narrate=narrate),
        tool_surface=None,
    )
    _, _, events = await _start(workflow, "kenya oda 2023")
    return captured, narrate, events


async def test_a_refused_write_is_recorded_on_the_trace_with_its_raw_code_and_origin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The trace records the raw code, origin, retry mode, advice and the
    platform's own trace id, and omits the platform `message`. Narrate's facts
    and compose's output carry only the closed-vocabulary reason."""
    long_message = "the chart breaks a display rule " + "x" * 700
    long_advice = "violates constraint 'comparable_units_required' " + "y" * 400
    write_result = {
        "outcome": "refusal",
        "refusal": {
            "code": "constraint_violation",
            "retry_mode": "correct_and_retry",
            "origin": "service",
            "fix": None,
            "advice": long_advice,
            "message": long_message,
            "platform_trace_id": "41d94ed2",
        },
    }

    captured, narrate, events = await _run_refused_write(monkeypatch, write_result=write_result)

    assert captured.events == [
        (
            "compose.write_refused",
            {
                "code": "constraint_violation",
                "origin": "service",
                "retry_mode": "correct_and_retry",
                "reason": "the platform's chart rules do not allow that chart for this data",
                "component_type": "stat_card",
                "title": "final",
                "spec_id": REF_A["spec_id"],
                "content_version": REF_A["content_version"],
                "platform_trace_id": "41d94ed2",
                "advice": long_advice[:300],
            },
        )
    ]
    (facts,) = narrate.facts
    assert facts["refused_writes"] == [
        {
            "type": "stat_card",
            "title": "final",
            "reason": "the platform's chart rules do not allow that chart for this data",
        }
    ]
    for text in (json.dumps(facts), json.dumps(_output_of(events, "compose"))):
        assert "constraint_violation" not in text
        assert "comparable_units_required" not in text
        assert "display rule" not in text
        assert "correct_and_retry" not in text
        assert '"origin"' not in text
        assert "41d94ed2" not in text


@pytest.mark.parametrize(
    "platform_trace_id",
    [
        pytest.param(None, id="absent"),
        pytest.param("", id="empty"),
        pytest.param(41, id="not-a-string"),
        pytest.param(["41d94ed2"], id="a-list"),
    ],
)
async def test_a_refusal_without_a_usable_platform_trace_id_records_none(
    monkeypatch: pytest.MonkeyPatch, platform_trace_id: Any
) -> None:
    write_result = {
        "outcome": "refusal",
        "refusal": {
            "code": "internal_error",
            "retry_mode": "do_not_retry",
            "origin": "service",
            "fix": None,
            "advice": None,
            "message": "Internal error",
            "platform_trace_id": platform_trace_id,
        },
    }

    captured, _, _ = await _run_refused_write(monkeypatch, write_result=write_result)

    ((_, metadata),) = captured.events
    assert "platform_trace_id" not in metadata
    assert "message" not in metadata


async def test_a_card_the_harness_withheld_is_recorded_as_a_harness_refusal_and_read_as_withheld(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    write_result = {
        "outcome": "refusal",
        "refusal": {
            "code": "provenance_unavailable",
            "retry_mode": "do_not_retry",
            "origin": "harness",
            "fix": None,
            "message": "no source manifest is linked to this frame",
        },
    }

    captured, narrate, events = await _run_refused_write(monkeypatch, write_result=write_result)

    ((name, metadata),) = captured.events
    assert name == "compose.write_refused"
    assert metadata["code"] == "provenance_unavailable"
    assert metadata["origin"] == "harness"
    assert metadata["retry_mode"] == "do_not_retry"
    assert metadata["reason"] == _WITHHELD_REASON
    assert "advice" not in metadata
    (facts,) = narrate.facts
    assert facts["refused_writes"] == [
        {"type": "stat_card", "title": "final", "reason": _WITHHELD_REASON}
    ]
    for text in (json.dumps(facts), json.dumps(_output_of(events, "compose"))):
        assert "provenance_unavailable" not in text
        assert "no source manifest" not in text


@pytest.mark.parametrize(
    ("write_result", "code"),
    [
        pytest.param({"outcome": "ok", "frame_refs": []}, "component_missing", id="no-component"),
        pytest.param(
            {"outcome": "ok", "frame_refs": [], "component": {"component_id": "cmp_1"}},
            "component_invalid",
            id="fails-wire-validation",
        ),
    ],
)
async def test_an_ok_write_with_nothing_to_draw_from_is_recorded_as_a_harness_defect(
    monkeypatch: pytest.MonkeyPatch, write_result: dict[str, Any], code: str
) -> None:
    captured, narrate, _ = await _run_refused_write(monkeypatch, write_result=write_result)

    ((name, metadata),) = captured.events
    assert name == "compose.write_refused"
    assert (metadata["code"], metadata["origin"]) == (code, "harness")
    assert metadata["reason"] == "the assistant could not build the card"
    (facts,) = narrate.facts
    assert facts["refused_writes"] == [
        {"type": "stat_card", "title": "final", "reason": "the assistant could not build the card"}
    ]
    assert code not in json.dumps(facts)


async def test_a_refused_describe_frame_is_logged_and_recorded_with_its_raw_code(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    captured = _CapturedEvents()
    monkeypatch.setattr(graph_module, "record_event", captured)
    narrate = _CapturingNarrate()
    workflow = create_root_workflow(
        resolver=ScriptedResolver([_envelope()]),
        executor=ScriptedExecutor(
            [
                {"outcome": "ok", "frame_refs": [REF_A]},  # get_data
                {
                    "outcome": "refusal",
                    "refusal": {
                        "code": "frame_unavailable",
                        "retry_mode": "retry_after_delay",
                        "origin": "service",
                        "fix": None,
                        "message": _LEAKY_DETAIL,
                    },
                },  # describe_frame
            ]
        ),
        generative=GenerativeNodes(compose=RecordingCompose(), narrate=narrate),
        tool_surface=None,
    )

    with caplog.at_level(logging.WARNING, logger="harness.graph"):
        _, _, events = await _start(workflow, "kenya oda 2023")

    assert captured.events == [
        (
            "compose.describe_frame_refused",
            {
                "code": "frame_unavailable",
                "origin": "service",
                "retry_mode": "retry_after_delay",
                "reason": "this step's data view is temporarily unavailable",
                "spec_id": REF_A["spec_id"],
                "content_version": REF_A["content_version"],
            },
        )
    ]
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert any(
        "describe_frame refused" in r.getMessage() and "frame_unavailable" in r.getMessage()
        for r in warnings
    )
    (facts,) = narrate.facts
    assert facts["frame_views"] == [
        {"available": False, "reason": "this step's data view is temporarily unavailable"}
    ]
    for text in (json.dumps(facts), json.dumps(_output_of(events, "compose"))):
        assert "frame_unavailable" not in text
        assert _LEAKY_DETAIL not in text


_LOGGED_TRACE_ID = "41d94ed2"


def _refusal_with_trace_id(platform_trace_id: Any) -> dict[str, Any]:
    return {
        "code": "internal_error",
        "retry_mode": "do_not_retry",
        "origin": "service",
        "fix": None,
        "message": "Internal error",
        "platform_trace_id": platform_trace_id,
    }


@pytest.mark.parametrize(
    ("platform_trace_id", "logged"),
    [
        pytest.param(_LOGGED_TRACE_ID, True, id="present"),
        pytest.param(None, False, id="absent"),
        pytest.param("", False, id="empty"),
        pytest.param(41, False, id="not-a-string"),
    ],
)
async def test_a_refused_describe_frame_log_line_names_the_platform_trace_id_when_there_is_one(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    platform_trace_id: Any,
    logged: bool,
) -> None:
    monkeypatch.setattr(graph_module, "record_event", _CapturedEvents())
    workflow = create_root_workflow(
        resolver=ScriptedResolver([_envelope()]),
        executor=ScriptedExecutor(
            [
                {"outcome": "ok", "frame_refs": [REF_A]},  # get_data
                {
                    "outcome": "refusal",
                    "refusal": _refusal_with_trace_id(platform_trace_id),
                },  # describe_frame
            ]
        ),
        generative=GenerativeNodes(compose=RecordingCompose(), narrate=_CapturingNarrate()),
        tool_surface=None,
    )

    with caplog.at_level(logging.WARNING, logger="harness.graph"):
        await _start(workflow, "kenya oda 2023")

    (line,) = [
        r.getMessage()
        for r in caplog.records
        if r.getMessage().startswith("describe_frame refused")
    ]
    assert ("platform_trace_id" in line) is logged
    assert (f"platform_trace_id={_LOGGED_TRACE_ID!r}" in line) is logged


@pytest.mark.parametrize(
    ("platform_trace_id", "logged"),
    [
        pytest.param(_LOGGED_TRACE_ID, True, id="present"),
        pytest.param(None, False, id="absent"),
        pytest.param("", False, id="empty"),
        pytest.param(41, False, id="not-a-string"),
    ],
)
async def test_a_refused_write_log_line_names_the_platform_trace_id_when_there_is_one(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    platform_trace_id: Any,
    logged: bool,
) -> None:
    write_result = {"outcome": "refusal", "refusal": _refusal_with_trace_id(platform_trace_id)}

    with caplog.at_level(logging.WARNING, logger="harness.graph"):
        await _run_refused_write(monkeypatch, write_result=write_result)

    (line,) = [
        r.getMessage()
        for r in caplog.records
        if r.getMessage().startswith("write_component did not produce a usable component")
    ]
    assert ("platform_trace_id" in line) is logged
    assert (f"platform_trace_id={_LOGGED_TRACE_ID!r}" in line) is logged


# --- the stat_card binding guard ---------------------------------------------

_MULTISERIES_VIEW: dict[str, Any] = {
    "dimensions": [
        {"name": "stat_var", "plural": False, "values": ["dcid:Count_Person"]},
        {
            "name": "observationAbout",
            "plural": True,
            "values": [
                f"country/{code}" for code in ("BLZ", "CRI", "GTM", "HND", "NIC", "PAN", "SLV")
            ],
        },
    ],
    "series_count": 7,
}


class UnpinnedPluralCompose:
    """Drafts a `stat_card` intent that leaves the frame's one plural
    dimension unpinned -- the exact select the platform refuses with
    `binding_invalid` on a real multi-series frame (a `stat_var`-only
    select against a 7-country frame)."""

    async def __call__(
        self,
        *,
        question: str,
        frame_refs: list[dict[str, Any]],
        frame_views: list[dict[str, Any]],
        frame_provenance: list[dict[str, Any]],
    ) -> dict[str, Any]:
        return {
            "intents": [
                {
                    "type": "stat_card",
                    "frame_ref": frame_refs[0],
                    "select": {"stat_var": "dcid:Count_Person"},
                    "title": "population",
                }
            ]
        }


async def test_stat_card_with_unpinned_plural_dimension_never_reaches_write_component(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A `stat_card` select that leaves a plural dimension unpinned resolves
    to more than one series -- the platform's own `binding_invalid` denial.
    `_admissible_intents` catches it locally, before a `write_component`
    round trip, the same discipline it already applies to a hallucinated
    frame_ref or a non-final frame."""
    resolver = ScriptedResolver([_envelope()])
    executor = ScriptedExecutor(
        [
            {"outcome": "ok", "frame_refs": [REF_A]},  # get_data
            {"outcome": "ok", "view": _MULTISERIES_VIEW},  # describe_frame
        ]
    )
    workflow = create_root_workflow(
        resolver=resolver,
        executor=executor,
        generative=GenerativeNodes(compose=UnpinnedPluralCompose()),
        tool_surface=None,
    )
    with caplog.at_level("WARNING", logger="harness.graph"):
        _, _, events = await _start(workflow, "population by country")

    assert [c["tool"] for c in executor.calls] == ["get_data", "describe_frame"]
    composed = _output_of(events, "compose")
    assert composed["fact_summaries"] == []
    assert composed["refused_writes"] == [
        {
            "type": "stat_card",
            "title": "population",
            "reason": "the assistant could not build the card",
        }
    ]
    assert any("rejected an intent" in record.getMessage() for record in caplog.records)


class OverTheSeriesCapCompose:
    """Drafts a `bar_chart` intent that binds the measure and leaves the
    frame's 30 donors unbound, so its select covers 30 series."""

    async def __call__(
        self,
        *,
        question: str,
        frame_refs: list[dict[str, Any]],
        frame_views: list[dict[str, Any]],
        frame_provenance: list[dict[str, Any]],
    ) -> dict[str, Any]:
        return {
            "intents": [
                {
                    "type": "bar_chart",
                    "frame_ref": frame_refs[0],
                    "select": {"stat_var": "dcid:Aid"},
                    "title": "aid by donor",
                }
            ]
        }


class _RecordsNarrateFacts:
    def __init__(self) -> None:
        self.facts: list[dict[str, Any]] = []

    async def __call__(self, *, question: str, facts: dict[str, Any]) -> dict[str, Any]:
        del question
        self.facts.append(facts)
        return {"text": "drafted"}


@pytest.mark.parametrize("drafts", [True, False], ids=["model", "scripted"])
async def test_a_select_over_the_series_cap_tells_the_close_why_the_chart_is_missing(
    drafts: bool,
) -> None:
    """A select covering more than 24 series is refused before any
    `write_component` call, and its reason reaches `refused_writes`: the
    scripted close states it, and a narrate model reads it in its facts."""
    view = {
        "dimensions": [
            {"name": "stat_var", "plural": False, "values": ["dcid:Aid"], "total_count": 1},
            {"name": "donor", "plural": True, "values": [], "total_count": 30},
        ],
        "series_count": 30,
    }
    executor = ScriptedExecutor(
        [
            {"outcome": "ok", "frame_refs": [REF_A]},  # get_data
            {"outcome": "ok", "view": view},  # describe_frame
        ]
    )
    narrate = _RecordsNarrateFacts()
    workflow = create_root_workflow(
        resolver=ScriptedResolver([_envelope()]),
        executor=executor,
        generative=GenerativeNodes(
            compose=OverTheSeriesCapCompose(), narrate=narrate if drafts else None
        ),
        tool_surface=None,
    )

    _, _, events = await _start(workflow, "aid by donor")

    refused = {"type": "bar_chart", "title": "aid by donor", "reason": _TOO_MANY_SERIES_REASON}
    assert [c["tool"] for c in executor.calls] == ["get_data", "describe_frame"]
    assert _output_of(events, "compose")["refused_writes"] == [refused]
    if drafts:
        assert narrate.facts[0]["refused_writes"] == [refused]
    else:
        assert (
            f"A component 'aid by donor' was not added: {_TOO_MANY_SERIES_REASON}."
            in _output_of(events, "narrate")["text"]
        )


_REPAIRABLE_VIEW: dict[str, Any] = {
    "dimensions": [
        {"name": "stat_var", "plural": False, "values": ["dcid:Count_Person"]},
        {"name": "observationAbout", "plural": True, "values": ["country/BLZ", "country/CRI"]},
    ],
    "series_count": 2,
}

_BINDING_INVALID_REFUSAL: dict[str, Any] = {
    "outcome": "refusal",
    "refusal": {
        "code": "binding_invalid",
        "retry_mode": "correct_and_retry",
        "origin": "service",
        "fix": None,
        "message": "a stat_card names exactly one series; this selection resolves to 2",
    },
}


class StaleValueCompose:
    """Drafts a `stat_card` intent that pins the frame's plural dimension to
    a deliberately chosen value the platform does not hold -- e.g. the
    model pinned Kenya, but Kenya is stale or was itself refused for an
    unrelated reason. The presence guard passes it (a value IS given), but
    the platform still refuses `binding_invalid`."""

    async def __call__(
        self,
        *,
        question: str,
        frame_refs: list[dict[str, Any]],
        frame_views: list[dict[str, Any]],
        frame_provenance: list[dict[str, Any]],
    ) -> dict[str, Any]:
        return {
            "intents": [
                {
                    "type": "stat_card",
                    "frame_ref": frame_refs[0],
                    "select": {"stat_var": "dcid:Count_Person", "observationAbout": "country/ZZZ"},
                    "title": "population",
                }
            ]
        }


async def test_a_correct_and_retry_refusal_is_surfaced_rather_than_retried() -> None:
    """A refused stat_card is never re-sent with a substituted selection.
    Retrying by picking the view's first value for a dimension the model
    deliberately pinned can succeed as a DIFFERENT entity, showing a
    confident card about the wrong one, so a `correct_and_retry` refusal
    buys no second attempt here: `_select_binding_ok` already rejects the
    shape this refusal describes before any call is made, and a refusal
    that still arrives is reported rather than guessed at. Exactly one
    `write_component` call is made, and the refusal reaches
    `refused_writes`."""
    resolver = ScriptedResolver([_envelope()])
    executor = ScriptedExecutor(
        [
            {"outcome": "ok", "frame_refs": [REF_A]},  # get_data
            {"outcome": "ok", "view": _REPAIRABLE_VIEW},  # describe_frame
            _BINDING_INVALID_REFUSAL,  # write_component: refused, never retried
        ]
    )
    workflow = create_root_workflow(
        resolver=resolver,
        executor=executor,
        generative=GenerativeNodes(compose=StaleValueCompose()),
        tool_surface=None,
    )
    _, _, events = await _start(workflow, "population by country")

    assert [c["tool"] for c in executor.calls] == ["get_data", "describe_frame", "write_component"]
    composed = _output_of(events, "compose")
    assert composed["fact_summaries"] == []
    assert composed["refused_writes"] == [
        {
            "type": "stat_card",
            "title": "population",
            "reason": "the selection did not identify a single series",
        }
    ]


def _population_view(*, plural: str) -> dict[str, Any]:
    """The ONEcampaign/dfo-apis#953 population frame's view: one `stat_var`, seven countries, and
    a `unit` column holding `USDollar` on some rows and null on others, which
    dfo-apis reports as plural with the one non-null value listed. `plural`
    names a further dimension marked plural with two listed values."""
    return {
        "dimensions": [
            {"name": "stat_var", "plural": False, "values": ["dcid:Count_Person"]},
            {
                "name": "observationAbout",
                "plural": True,
                "values": ["country/GTM", "country/HND"],
                "total_count": 7,
            },
            {"name": "unit", "plural": True, "values": ["USDollar"], "total_count": 1},
            {"name": plural, "plural": True, "values": ["a", "b"], "total_count": 2},
        ]
    }


@pytest.mark.parametrize("facet", ["measurementMethod", "provenance"])
def test_stat_card_may_leave_plural_facets_unpinned(facet: str) -> None:
    """dfo-apis refuses an unbound facet only when it splits a series, so a
    population card pinning the country passes with `unit` = `{USDollar,
    null}` and a second plural facet column left unbound."""
    select = {"stat_var": "dcid:Count_Person", "observationAbout": "country/GTM"}
    view = _population_view(plural=facet)
    assert _select_binding_ok(type_="stat_card", select=select, view=view) is None


def test_stat_card_still_refuses_an_unpinned_non_facet_plural_dimension() -> None:
    select = {"stat_var": "dcid:Count_Person", "observationAbout": "country/GTM"}
    reason = _select_binding_ok(
        type_="stat_card", select=select, view=_population_view(plural="sex")
    )
    assert reason is not None
    assert reason["code"] == "select_invalid"


# --- the select_binding_ok array-dimension cap, per component type -----------


def test_select_binding_ok_permits_a_table_with_two_array_dimensions() -> None:
    """The platform's catalogue grammar allows `table` to carry arrays on
    more than one dimension (their cross-product, at most 24 series); only
    `line_chart` and `bar_chart` are limited to at most one array-valued
    dimension."""
    select = {"observationAbout": ["country/KEN", "country/UGA"], "stat_var": ["a", "b"]}
    assert _select_binding_ok(type_="table", select=select, view={}) is None


def test_select_binding_ok_still_rejects_a_line_chart_with_two_array_dimensions() -> None:
    select = {"observationAbout": ["country/KEN", "country/UGA"], "stat_var": ["a", "b"]}
    assert _select_binding_ok(type_="line_chart", select=select, view={}) is not None


def _countries(count: int) -> list[str]:
    return [f"country/C{index:02d}" for index in range(count)]


# The platform's release document as presentation-mcp's tests record it.
_RELEASE = Path(__file__).parents[2] / "presentation-mcp/tests/fixtures/release.json"


def test_a_line_chart_lists_no_more_values_than_the_catalogue_allows() -> None:
    select_schema = json.loads(_RELEASE.read_text())["intent_schemas_tolerant"]["line_chart"][
        "properties"
    ]["select"]
    (array,) = [
        branch
        for branch in select_schema["additionalProperties"]["anyOf"]
        if branch["type"] == "array"
    ]
    cap = array["maxItems"]

    allowed = _select_binding_ok(
        type_="line_chart", select={"observationAbout": _countries(cap)}, view={}
    )
    refused = _select_binding_ok(
        type_="line_chart", select={"observationAbout": _countries(cap + 1)}, view={}
    )

    assert cap == LINE_CHART_MAX_VALUES
    assert allowed is None
    assert refused is not None
    assert refused["code"] == "series_limit_exceeded"


@pytest.mark.parametrize("type_", ["bar_chart", "table"])
def test_select_binding_ok_permits_24_series(type_: str) -> None:
    select = {"observationAbout": _countries(24), "stat_var": "a"}
    assert _select_binding_ok(type_=type_, select=select, view={}) is None


@pytest.mark.parametrize("type_", ["bar_chart", "table"])
def test_select_binding_ok_rejects_a_25th_series(type_: str) -> None:
    """dfo-apis refuses a component past 24 series even though the catalogue
    grammar's `maxItems` allows an array of 30."""
    select = {"observationAbout": _countries(25), "stat_var": "a"}
    reason = _select_binding_ok(type_=type_, select=select, view={})
    assert reason is not None
    assert reason["message"] == (
        f"{type_} select expands to an estimated 25 series, over the platform's maximum of 24"
    )


def test_select_binding_ok_counts_a_tables_cross_product_against_the_cap() -> None:
    select = {"observationAbout": _countries(5), "stat_var": ["a", "b", "c", "d", "e"]}
    reason = _select_binding_ok(type_="table", select=select, view={})
    assert reason is not None
    assert reason["message"] == (
        "table select expands to an estimated 25 series, over the platform's maximum of 24"
    )


def _counted_view(**total_counts: int) -> dict[str, Any]:
    """A `describe_frame` view whose dimensions carry dfo-apis's `total_count`."""
    return {
        "dimensions": [
            {"name": name, "plural": count > 1, "values": [], "total_count": count}
            for name, count in total_counts.items()
        ]
    }


@pytest.mark.parametrize(
    ("type_", "select", "view", "estimate"),
    [
        ("bar_chart", {"stat_var": "a"}, _counted_view(stat_var=1, donor=30), 30),
        ("table", {"stat_var": "a"}, _counted_view(stat_var=1, donor=30), 30),
        (
            "line_chart",
            {"donor": [f"donor/{index}" for index in range(5)]},
            _counted_view(stat_var=5, donor=30),
            25,
        ),
        (
            "table",
            {"donor": [f"donor/{index}" for index in range(20)]},
            _counted_view(stat_var=2, donor=30),
            40,
        ),
        ("table", {"stat_var": "a"}, _counted_view(stat_var=1, donor=30, unit=30), 30),
        (
            "table",
            {"stat_var": "a", "unit": ["USDollar", "EUR"]},
            _counted_view(stat_var=1, donor=30, unit=3, measurementMethod=5),
            60,
        ),
    ],
)
def test_select_binding_ok_matches_dfo_apis_series_estimate(
    type_: str, select: dict[str, Any], view: dict[str, Any], estimate: int
) -> None:
    """The harness computes dfo-apis's own estimate (`_check_series_estimate`
    in dfo-data-tools' `assembler/resolve.py`): over every coordinate
    dimension, the bound array's length, 1 for a single bound value, and
    `max(total_count, 1)` for an unbound dimension, on every component type
    the assembler resolves. Each case here is one dfo-apis refuses with
    `MAX_SERIES_PER_COMPONENT`: an unbound plural dimension, a `line_chart`
    array past 24, and an unbound dimension multiplying a table's array.
    An unbound facet column counts 1 and a bound one counts its values."""
    reason = _select_binding_ok(type_=type_, select=select, view=view)
    assert reason is not None
    assert reason["message"] == (
        f"{type_} select expands to an estimated {estimate} series, "
        "over the platform's maximum of 24"
    )


def test_select_binding_ok_counts_an_unbound_dimension_at_its_total_count() -> None:
    """An unbound dimension holding 24 values is within the cap, and binding
    it to a single value brings 30 down to 1."""
    assert (
        _select_binding_ok(
            type_="bar_chart", select={"stat_var": "a"}, view=_counted_view(stat_var=1, donor=24)
        )
        is None
    )
    assert (
        _select_binding_ok(
            type_="bar_chart",
            select={"stat_var": "a", "donor": "donor/1"},
            view=_counted_view(stat_var=1, donor=30),
        )
        is None
    )
    assert (
        _select_binding_ok(
            type_="bar_chart",
            select={"stat_var": "a"},
            view=_counted_view(stat_var=1, donor=24, unit=30, provenance=30),
        )
        is None
    )


def test_select_binding_ok_counts_a_dimension_with_no_total_count_as_one() -> None:
    """Scripted fixtures list bare dimension names, or entries without
    `total_count`; neither refuses for want of the figure."""
    bare = {"dimensions": ["stat_var", "observationAbout"]}
    assert _select_binding_ok(type_="bar_chart", select={"stat_var": "a"}, view=bare) is None
    assert (
        _select_binding_ok(type_="table", select={"stat_var": "a"}, view=_MULTISERIES_VIEW) is None
    )


def test_select_binding_ok_refuses_a_dimension_the_view_does_not_declare() -> None:
    """dfo-apis estimates series over its own dimensions only, and refuses a
    select binding any other name as an invalid binding; the harness gives
    the same reason."""
    select = {"stat_var": "a", "donor": [f"donor/{index}" for index in range(30)]}
    reason = _select_binding_ok(type_="bar_chart", select=select, view=_counted_view(stat_var=1))
    assert reason is not None
    assert reason["code"] == "select_invalid"
    assert reason["message"] == "bar_chart select binds ['donor'], which the frame does not declare"
    bare = {"dimensions": ["stat_var", "observationAbout"]}
    reason = _select_binding_ok(type_="table", select={"sector": "x"}, view=bare)
    assert reason is not None
    assert reason["message"] == "table select binds ['sector'], which the frame does not declare"


def test_select_binding_ok_rejects_a_non_mapping_select() -> None:
    """A `select_json` that parsed to a JSON array or scalar is a rejection
    reason here, not an `AttributeError` on `.items()` -- defense in depth
    behind the parse-site skip in `harness.generative.GeminiCompose`."""
    assert _select_binding_ok(type_="table", select=["not", "a", "mapping"], view={}) is not None


_TWO_ENTITY_LINE_VIEW: dict[str, Any] = {
    "dimensions": [
        {"name": "stat_var", "plural": False, "values": ["dcid:LifeExpectancy_Person"]},
        {
            "name": "observationAbout",
            "plural": True,
            "values": ["country/GTM", "country/NGA"],
        },
    ],
    "period_grain": "year",
    "series_count": 2,
}


class TwoEntityLineCompose:
    """Drafts the line chart a two-country frame calls for, with the entity
    dimension selected as a two-value array, which draws one line per
    country."""

    def __init__(self) -> None:
        self.selects: list[dict[str, Any]] = []

    async def __call__(
        self,
        *,
        question: str,
        frame_refs: list[dict[str, Any]],
        frame_views: list[dict[str, Any]],
        frame_provenance: list[dict[str, Any]],
    ) -> dict[str, Any]:
        select = {"observationAbout": ["country/GTM", "country/NGA"]}
        self.selects.append(select)
        return {
            "intents": [
                {
                    "type": "line_chart",
                    "frame_ref": frame_refs[0],
                    "select": select,
                    "title": "Life expectancy in Nigeria and Guatemala",
                }
            ]
        }


async def test_a_two_entity_frame_drafts_a_two_series_line_chart() -> None:
    """A frame holding two entities of one measure reaches `write_component`
    as a line_chart with the entity dimension selected as a two-value
    array. The select passes `_select_binding_ok` and survives
    `_admissible_intents`, so the card the turn writes carries both
    countries: one `get_data`, one component."""
    resolver = ScriptedResolver([_envelope()])
    executor = ScriptedExecutor(
        [
            {"outcome": "ok", "frame_refs": [REF_A]},  # get_data
            {"outcome": "ok", "view": _TWO_ENTITY_LINE_VIEW},  # describe_frame
            {
                "outcome": "ok",
                "frame_refs": [],
                "component": _valid_component(
                    "cmp_two_entities",
                    fact_summary={
                        "component_id": "cmp_two_entities",
                        "revision": 1,
                        "series": [
                            {"id": "gtm", "label": "Guatemala", "point_count": 1},
                            {"id": "nga", "label": "Nigeria", "point_count": 1},
                        ],
                        "indicators": [],
                        "attribution": {"label": "Source", "source_count": 1, "indicator_count": 1},
                    },
                ),
                "summary": "created",
            },
        ]
    )
    compose = TwoEntityLineCompose()
    workflow = create_root_workflow(
        resolver=resolver,
        executor=executor,
        generative=GenerativeNodes(compose=compose),
        tool_surface=None,
    )
    _, _, events = await _start(workflow, "life expectancy in Nigeria and Guatemala")

    assert [c["tool"] for c in executor.calls] == ["get_data", "describe_frame", "write_component"]
    written = next(c["args"] for c in executor.calls if c["tool"] == "write_component")
    intent = written["intent"]
    assert intent["type"] == "line_chart"
    assert intent["select"]["observationAbout"] == ["country/GTM", "country/NGA"]
    assert (
        _select_binding_ok(type_="line_chart", select=intent["select"], view=_TWO_ENTITY_LINE_VIEW)
        is None
    )
    composed = _output_of(events, "compose")
    assert [series["label"] for series in composed["fact_summaries"][0]["series"]] == [
        "Guatemala",
        "Nigeria",
    ]
    assert composed["refused_writes"] == []


# --- the Plan objects stay real ------------------------------------------------


async def test_plan_amendment_lineage_is_real() -> None:
    """The plan carried through amend cycles is a valid `planner.plan.Plan`."""
    resolver = ScriptedResolver([_envelope(choice_points=2)])
    executor = ScriptedExecutor()
    runner, session_id, events = await _start(
        create_root_workflow(
            resolver=resolver,
            executor=executor,
            generative=GenerativeNodes(planner=ScriptedPlannerModel()),
            tool_surface=None,
        ),
        "q with forks",
    )
    events = await _resume(runner, session_id, APPROVE_INTERRUPT, {"approved": 99})

    output = _output_of(events, "planner")
    plan_fields = {
        k: v for k, v in output.items() if k not in ("summary", "revision", "depth", "depth_notice")
    }
    plan = PlannerPlan.model_validate(plan_fields)
    assert plan.version == 2
    assert plan.previous_version == 1
    assert plan.decisions[0].chosen in plan.decisions[0].options


# --- the planner node against a model seam --------------------------------


def _seam_plan(
    *,
    question: str = "kenya debt",
    version: int = 1,
    schemas: list[ToolSchema],
    bind_evidence: bool = False,
    stat_vars: str | None = None,
) -> PlannerPlan:
    """A real, translated `planner.plan.Plan` -- the shape a `PlannerModel` seam hands back.

    `stat_vars` is the reading its `get_data` step sends, for a plan whose
    source is a family of StatVars; `None` sends none.
    """
    decision = Decision(
        slot="window",
        description="Which years?",
        options=["2015-2024", "2018-2024"],
        chosen="2015-2024",
        resolved_by=ResolvedBy.PLANNER_DEFAULT,
        rationale="a full decade covers the series",
    )
    dataset = DatasetRequirement(
        id="spending",
        decision_ids=["window"],
        measure="health spending",
        entities=[Entity(name="Kenya", role="recipient")],
        period=Period(start=2015, end=2024),
    )
    totals = AggregateRequirement(
        id="totals",
        decision_ids=["window"],
        input_id="spending",
        reduction=Reduction.SUM,
        group_by=(GroupBy.PERIOD,),
    )
    draft = PlannerPlan(
        question=question,
        decisions=[decision],
        evidence_requirements=[dataset, totals],
        version=version,
    )
    translation = Translation(
        operations=[
            RetrieveOperation(id="get_data", requirement_id=dataset.id, spec_id=SPEC_A),
            AggregateOperation(
                id="totals",
                requirement_id=totals.id,
                input="get_data",
                reduction=Reduction.SUM,
                group_by=(GroupBy.PERIOD,),
            ),
        ],
        terminal=["totals"],
    )
    named = {schema.name: schema for schema in schemas}
    get_data_schema = named["get_data"]
    aggregate_schema = named["aggregate"]
    workflow = Workflow(
        steps=[
            WorkflowStep(
                id="get_data",
                operation_id="get_data",
                tool=ToolId(server="data-mcp", name="get_data"),
                args={"spec_id": SPEC_A} | ({"stat_vars": stat_vars} if stat_vars else {}),
                input_schema_hash=get_data_schema.input_schema_hash,
                output_schema_hash=get_data_schema.output_schema_hash,
                implementation_version=get_data_schema.version,
            ),
            WorkflowStep(
                id="totals",
                operation_id="totals",
                tool=ToolId(server="analysis-mcp", name="aggregate"),
                args={"aggregation": "sum", "group_by": ["time_period"]},
                bindings={"frame_ref": {"from_step": "get_data", "output": "frame_ref"}},
                input_schema_hash=aggregate_schema.input_schema_hash,
                output_schema_hash=aggregate_schema.output_schema_hash,
                implementation_version=aggregate_schema.version,
            ),
        ],
        sources=[
            SelectedSource(spec_id=SPEC_A, sentence="health spending in Kenya", has_data=True)
        ],
        # No evidence bindings by default: most of these tests are about
        # approval/execution wiring, not the evidence check, and carry no
        # real source discovery to check a manifest against. A caller
        # testing a refusal on the bound step (which never reaches the
        # check, since it only runs on success) opts in.
        evidence_bindings=(
            [EvidenceBinding(requirement_id=dataset.id, step_id="get_data", spec_id=SPEC_A)]
            if bind_evidence
            else []
        ),
        terminal_steps=["totals"],
    )
    return draft.with_translation(translation=translation, workflow=workflow)


class _Seam:
    """A `PlannerModel` returning `{"plan": ..., "summary": ...}` verbatim."""

    def __init__(self, plans: list[dict[str, Any]]) -> None:
        self._plans = plans
        self.calls: list[dict[str, Any]] = []

    async def __call__(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        index = min(len(self.calls) - 1, len(self._plans) - 1)
        return self._plans[index]


async def test_the_planner_node_validates_the_seams_plan_and_writes_it_to_state(
    real_schemas: list[ToolSchema],
) -> None:
    """The seam returns `{"plan": <planner.plan.Plan JSON>, "summary": ...}`;
    the node validates it as a `planner.plan.Plan`, and emits `summary` and
    `revision` beside it -- both the record and the wire keys, not one
    constructed from parts the way the frozen adapter's protocol was.
    """
    plan = _seam_plan(schemas=real_schemas)
    seam = _Seam([{"plan": plan.model_dump(mode="json"), "summary": {"version": plan.version}}])
    resolver = ScriptedResolver([_envelope(choice_points=2)])
    workflow = create_root_workflow(
        resolver=resolver,
        executor=ScriptedExecutor(),
        generative=GenerativeNodes(planner=seam),
        tool_surface=None,
    )

    _, _, events = await _start(workflow, "kenya debt")

    output = _output_of(events, "planner")
    assert output["summary"] == {"version": 1}
    assert output["revision"] == plan_revision(plan)
    plan_fields = {
        k: v for k, v in output.items() if k not in ("summary", "revision", "depth", "depth_notice")
    }
    validated = PlannerPlan.model_validate(plan_fields)
    assert validated == plan


async def test_the_planner_node_raises_on_a_malformed_plan_rather_than_constructing_one() -> None:
    """A draft the seam could not itself validate must not reach the approve gate."""
    seam = _Seam([{"plan": {"decisions": "not-a-list"}, "summary": {}}])
    resolver = ScriptedResolver([_envelope(choice_points=2)])
    workflow = create_root_workflow(
        resolver=resolver,
        executor=ScriptedExecutor(),
        generative=GenerativeNodes(planner=seam),
        tool_surface=None,
    )

    with pytest.raises(ValidationError):
        await _start(workflow, "kenya debt")


class _StallingSeam:
    """A `PlannerModel` that blocks mid-draft until released, recording
    whether its own task was cancelled and when it wrote to the graph's
    session state -- the two things a turn closed mid-draft must leave
    true: the draft is gone, and it wrote nothing after that close."""

    def __init__(self, *, ready: asyncio.Event, release: asyncio.Event) -> None:
        self._ready = ready
        self._release = release
        self.task: asyncio.Task[Any] | None = None
        self.state: dict[str, Any] | None = None

    async def __call__(self, *, ctx: Any, **_kwargs: Any) -> dict[str, Any]:
        self.task = asyncio.current_task()
        self.state = ctx.session.state
        self.state["seam_writes"] = [*self.state.get("seam_writes", []), "before"]
        self._ready.set()
        await self._release.wait()
        self.state["seam_writes"] = [*self.state.get("seam_writes", []), "after"]
        return {"plan": {}, "summary": {}}


async def test_a_turn_closed_mid_draft_cancels_the_seam_and_writes_no_state_after() -> None:
    """`studio._sse_stream`'s stop-and-disconnect path closes the ADK event
    stream through `_unwind`: cancel the in-flight next-frame task and
    await that cancellation, which throws into whatever node is suspended.
    The `planner` node is suspended in exactly that way whenever a draft is
    in flight: it awaits `models.planner(...)` directly (`graph.py`'s
    `planner`), so a cancellation there throws straight into the seam's own
    suspended `ctx.run_node` call. The draft must not survive: it is a live
    model and tool round trip that would otherwise keep running, and keep
    writing session state, after the turn has closed."""
    ready = asyncio.Event()
    release = asyncio.Event()
    seam = _StallingSeam(ready=ready, release=release)
    resolver = ScriptedResolver([_envelope(choice_points=2)])
    workflow = create_root_workflow(
        resolver=resolver,
        executor=ScriptedExecutor(),
        generative=GenerativeNodes(planner=seam),
        tool_surface=None,
    )
    runner = InMemoryRunner(node=workflow)
    session = await runner.session_service.create_session(app_name=runner.app_name, user_id="u")
    stream = runner.run_async(
        user_id="u",
        session_id=session.id,
        new_message=types.Content(role="user", parts=[types.Part.from_text(text="kenya debt")]),
    )

    # Drive the stream one frame at a time, the way `studio._sse_stream`
    # does, until the seam signals a draft is in flight. `next_frame` is
    # then the in-flight `anext()` task, suspended inside the planner
    # node's own `await models.planner(...)` call -- exactly what
    # `_unwind`, below, closes over in the real stop-and-disconnect path.
    next_frame = asyncio.ensure_future(stream.__anext__())
    ready_wait = asyncio.ensure_future(ready.wait())
    async with asyncio.timeout(5):
        while not ready_wait.done():
            done, _ = await asyncio.wait(
                {next_frame, ready_wait}, return_when=asyncio.FIRST_COMPLETED
            )
            if next_frame in done:
                next_frame.result()
                next_frame = asyncio.ensure_future(stream.__anext__())
    assert seam.task is not None and not seam.task.done(), "the draft must still be in flight"

    # Cancel the in-flight next-frame task and await the cancellation,
    # rather than firing it and moving on, so whatever the closed generator
    # does to clean up actually runs before this assertion. Unlike
    # `studio._unwind`, this wait is unshielded: nothing here cancels the
    # awaiting task, so a shield would change nothing.
    next_frame.cancel()
    with suppress(asyncio.CancelledError):
        await next_frame

    # If the draft survived the close it is still blocked on `release`;
    # release it and let it finish, so a late write shows up if one lands.
    release.set()
    if not seam.task.done():
        with suppress(asyncio.CancelledError):
            await asyncio.wait_for(seam.task, timeout=5)

    assert seam.task.cancelled(), "a turn closed mid-draft must cancel the draft, not orphan it"
    assert seam.state is not None
    assert seam.state.get("seam_writes") == ["before"], "no draft write may land after the close"


class _AskingSeam:
    """A `PlannerModel` that reports two progress rounds, then pauses the
    same way a real `PlannerSeam` does once `planner.agent`'s
    `adk_request_input` tool call is what it's awaiting: `ctx.run_node`
    raises `NodeInterruptedError` straight out of the awaited call
    (`google.adk.workflow._errors`, a `BaseException` used exclusively for
    this). Raised directly here rather than through a real nested
    `ctx.run_node` call -- so the interrupt this synthesises carries none of
    ADK's own bookkeeping for resuming it later (`test_planner_seam.py`'s
    `test_a_planner_ask_pauses_the_run_and_elects_nothing_until_answered` proves
    that half, through a real one) -- this is only for the one property
    this node itself owns: drop the progress queued before the pause, then
    let it propagate."""

    async def __call__(self, *, on_progress: Any = None, **_kwargs: Any) -> dict[str, Any]:
        if on_progress is not None:
            on_progress({"agent": "dfo_planner", "tools": ["propose_plan"]})
            on_progress({"agent": "dfo_planner", "tools": ["adk_request_input"]})
        raise NodeInterruptedError()


async def test_a_planner_side_pause_relays_no_progress_after_it() -> None:
    """The `planner` node awaits the seam directly, so a pause's
    `NodeInterruptedError` passes straight through it. A real pause has
    already streamed the ask by then, so the node relays none of the
    progress queued before it (a frame after the ask breaks the stream's
    pause), and the pause reaches the workflow's own handling untouched. No
    `plan.proposed` or `refused` output follows: the turn is suspended,
    not settled."""
    seam = _AskingSeam()
    resolver = ScriptedResolver([_envelope(choice_points=2)])
    workflow = create_root_workflow(
        resolver=resolver,
        executor=ScriptedExecutor(),
        generative=GenerativeNodes(planner=seam),
        tool_surface=None,
    )

    _, _, events = await _start(workflow, "kenya debt")

    progress = [
        event.custom_metadata["progress"]
        for event in events
        if event.custom_metadata and "progress" in event.custom_metadata
    ]
    assert progress == []
    assert _output_of(events, "planner") is None
    assert _output_of(events, "approve") is None
    assert _output_of(events, "deadend") is None


class _GatedSeam:
    """A `PlannerModel` that drafts only once its caller has seen the
    planner node's start event, then pauses like `_AskingSeam`."""

    def __init__(self) -> None:
        self.seen = asyncio.Event()
        self.drafted = False

    async def __call__(self, **_kwargs: Any) -> dict[str, Any]:
        await asyncio.wait_for(self.seen.wait(), timeout=5)
        self.drafted = True
        raise NodeInterruptedError()


async def test_the_planner_node_announces_a_draft_before_the_seam_returns() -> None:
    """The node's first event reaches the runner stream while the seam is
    still drafting, so Studio can say a plan is under way."""
    seam = _GatedSeam()
    workflow = create_root_workflow(
        resolver=ScriptedResolver([_envelope(choice_points=2)]),
        executor=ScriptedExecutor(),
        generative=GenerativeNodes(planner=seam),
        tool_surface=None,
    )
    runner = InMemoryRunner(node=workflow)
    session = await runner.session_service.create_session(app_name=runner.app_name, user_id="u")
    started = []
    async for event in runner.run_async(
        user_id="u",
        session_id=session.id,
        new_message=types.Content(role="user", parts=[types.Part.from_text(text="kenya debt")]),
    ):
        if event.custom_metadata and PLANNER_STARTED_KEY in event.custom_metadata:
            started.append(event.custom_metadata[PLANNER_STARTED_KEY])
            seam.seen.set()

    assert started == [{"revising": False}]
    assert seam.drafted


async def test_the_approve_node_refuses_an_old_shaped_plan_record(
    real_schemas: list[ToolSchema],
) -> None:
    """`ctx.state["plan"]` must hold a `planner.plan.Plan`. A record with a
    `tool_calls` sequence and an integer version, instead of a compiled
    workflow and a revision hash, is a strict-schema mismatch (`tool_calls`
    is not a `planner.plan.Plan` field), and the `approve` node must raise
    on it rather than quietly parsing it as a stand-in and keying approval
    on its version integer.
    """
    plan = _seam_plan(schemas=real_schemas)
    seam = _Seam([{"plan": plan.model_dump(mode="json"), "summary": {}}])
    resolver = ScriptedResolver([_envelope(choice_points=2)])
    workflow = create_root_workflow(
        resolver=resolver,
        executor=ScriptedExecutor(),
        generative=GenerativeNodes(planner=seam),
        tool_surface=_fixture_surface(real_schemas),
    )
    runner, session_id, events = await _start(workflow, "kenya debt")
    assert _pending_interrupt(events) == APPROVE_INTERRUPT

    # Overwrite the pending plan with the old shape directly in session
    # state -- the only way to reach `approve` with one, since every real
    # `PlannerModel` seam (the graph's own `planner` node included) drafts a
    # `planner.plan.Plan`, and nothing on the public path writes this shape.
    old_shape = {
        "question": "kenya debt",
        "decisions": [],
        "tool_calls": [{"tool": "get_data", "args": {}}],
        "version": 1,
        "previous_version": None,
        "status": "pending_approval",
    }
    session = runner.session_service.sessions[runner.app_name]["u"][session_id]
    session.state["plan"] = json.dumps(old_shape)

    with pytest.raises(ValidationError) as excinfo:
        await _resume(runner, session_id, APPROVE_INTERRUPT, {"approved": old_shape["version"]})

    # Pinned to the `approve` node itself, not to some other strict parse
    # downstream: a fallback silently accepting the old shape here would
    # instead route "approved" or "amend" and raise nothing at all, or
    # raise from `execute` instead.
    assert any(frame.name == "approve" for frame in excinfo.traceback)


async def test_an_approval_for_a_superseded_revision_routes_to_amend(
    real_schemas: list[ToolSchema],
) -> None:
    """The `approve` node keys a model-drafted plan on
    `plan_revision(plan)`, not the version integer. An answer naming a
    revision that does not match `ctx.state["plan"]` -- because an
    amendment already replaced it -- routes to "amend" the same way a stale
    version does, even when the superseding plan happens to share the
    same version number (`propose_plan` can start a fresh plan at version 1
    for what it reads as a different question, so two unrelated plans can
    both be version 1 in one conversation).
    """
    plan_v1 = _seam_plan(question="kenya debt", schemas=real_schemas)
    plan_v2 = _seam_plan(question="kenya debt, revised", schemas=real_schemas)
    seam = _Seam(
        [
            {"plan": plan_v1.model_dump(mode="json"), "summary": {}},
            {"plan": plan_v2.model_dump(mode="json"), "summary": {}},
        ]
    )
    resolver = ScriptedResolver([_envelope(choice_points=2)])
    executor = ScriptedExecutor()
    workflow = create_root_workflow(
        resolver=resolver,
        executor=executor,
        generative=GenerativeNodes(planner=seam),
        tool_surface=_fixture_surface(real_schemas),
    )
    runner, session_id, events = await _start(workflow, "kenya debt")
    assert _pending_interrupt(events) == APPROVE_INTERRUPT
    revision_1 = plan_revision(plan_v1)
    assert _output_of(events, "planner")["revision"] == revision_1

    # A wrong answer routes to amend, which redrafts -- an unrelated plan
    # that also happens to be version 1.
    events = await _resume(
        runner, session_id, APPROVE_INTERRUPT, {"approved": "not-a-real-revision"}
    )
    assert _pending_interrupt(events) == APPROVE_INTERRUPT
    amended = _output_of(events, "planner")
    assert amended["version"] == 1
    revision_2 = amended["revision"]
    assert revision_2 != revision_1

    # The first card's revision no longer names what's pending: superseded,
    # not approved -- nothing executes and a fresh approval is still pending.
    events = await _resume(runner, session_id, APPROVE_INTERRUPT, {"approved": revision_1})
    assert executor.calls == []
    assert _pending_interrupt(events) == APPROVE_INTERRUPT
    assert _output_of(events, "planner")["revision"] == revision_2

    # The correct, current revision approves cleanly: `approve()` succeeds,
    # the gate is passed (no fresh approval interrupt), and the approved
    # workflow's steps ran through `execute_approved_plan` against the
    # tool surface -- the no-planner `executor` never sees a plan step.
    events = await _resume(runner, session_id, APPROVE_INTERRUPT, {"approved": revision_2})
    assert _pending_interrupt(events) is None
    assert executor.calls == []
    assert _output_of(events, "narrate")["plan_version"] == 1


# The scripted plan as the code at commit 2563638 serialised it, beside the
# revision that code computed (`tests/planner/test_revision_format.py`). Its
# workflow is pinned to the schemas `ScriptedSurface` served at that commit,
# which `_BaseFormatSurface` serves.
_BASE_FORMAT_PLAN: dict[str, Any] = json.loads(
    (Path(__file__).parent / "fixtures" / "revision_base_plans.json").read_text()
)["aggregate"]


class _BaseFormatSurface(ScriptedSurface):
    """`ScriptedSurface` declaring `get_data` as it did at commit 2563638, when the
    tool took no `resolved_stat_var_dcids`."""

    async def list_tools(self) -> list[ToolSchema]:
        return [
            schema.model_copy(
                update={
                    "parameters": {
                        "type": "object",
                        "properties": {"spec_id": {"type": "string"}, "scope": {"type": "object"}},
                        "required": ["spec_id"],
                        "additionalProperties": False,
                    }
                }
            )
            if schema.name == "get_data"
            else schema
            for schema in await super().list_tools()
        ]


async def test_the_base_format_surface_declares_the_schemas_the_saved_plan_names() -> None:
    declared = {schema.name: schema for schema in await _BaseFormatSurface().list_tools()}

    for step in _BASE_FORMAT_PLAN["plan"]["workflow"]["steps"]:
        assert declared[step["tool"]["name"]].input_schema_hash == step["input_schema_hash"]


async def test_a_pending_plan_saved_at_the_base_revision_format_is_approved_unamended() -> None:
    """The card a user holds names the revision computed when the plan was
    saved. A plan saved without the fields its models have since gained
    still hashes to that revision, so the answer approves it and it runs."""
    seam = _Seam([{"plan": _BASE_FORMAT_PLAN["plan"], "summary": {}}])
    surface = _CallRecordingSurface(_BaseFormatSurface())
    workflow = create_root_workflow(
        resolver=ScriptedResolver([_envelope(choice_points=2)]),
        executor=ScriptedExecutor(),
        generative=GenerativeNodes(planner=seam),
        fetch_series_stats=scripted_fetch_series_stats,
        tool_surface=surface,
    )
    runner, session_id, events = await _start(workflow, "comprehensive life expectancy")
    assert _pending_interrupt(events) == APPROVE_INTERRUPT
    assert _output_of(events, "planner")["revision"] == _BASE_FORMAT_PLAN["revision"]
    # The session holds the plan exactly as it was saved.
    session = runner.session_service.sessions[runner.app_name]["u"][session_id]
    session.state["plan"] = json.dumps(_BASE_FORMAT_PLAN["plan"])

    events = await _resume(
        runner, session_id, APPROVE_INTERRUPT, {"approved": _BASE_FORMAT_PLAN["revision"]}
    )

    assert _pending_interrupt(events) is None
    assert len(seam.calls) == 1
    assert [call[0] for call in surface.calls] == ["get_data", "aggregate"]
    assert _output_of(events, "narrate")["plan_version"] == 1
    finished = PlannerPlan.model_validate_json(session.state["plan"])
    assert finished.status is PlanStatus.COMPLETED
    assert finished.approved_revision == _BASE_FORMAT_PLAN["revision"]


async def test_an_approved_plan_saved_at_the_base_revision_format_executes() -> None:
    """`approved_revision` was computed when the plan was approved;
    `Plan.start_execution` compares it with the revision computed now."""
    approved = PlannerPlan.model_validate(
        _BASE_FORMAT_PLAN["plan"]
        | {"status": "approved", "approved_revision": _BASE_FORMAT_PLAN["revision"]}
    )

    finished, outcome = await execute_approved_plan(
        approved, surface=_BaseFormatSurface(), fetch_series_stats=scripted_fetch_series_stats
    )

    assert isinstance(outcome, Accepted), outcome
    assert finished.status is PlanStatus.COMPLETED


async def test_a_saved_plan_whose_retrieval_posts_no_statvars_runs_while_get_data_is_unchanged() -> (
    None
):
    """The saved retrieval step carries `spec_id` alone, and `get_data` takes
    the elected StatVars as an optional argument, so the call is made as
    saved."""
    approved = PlannerPlan.model_validate(
        _BASE_FORMAT_PLAN["plan"]
        | {"status": "approved", "approved_revision": _BASE_FORMAT_PLAN["revision"]}
    )
    surface = _CallRecordingSurface(_BaseFormatSurface())

    _, outcome = await execute_approved_plan(
        approved, surface=surface, fetch_series_stats=scripted_fetch_series_stats
    )

    assert isinstance(outcome, Accepted), outcome
    assert surface.calls[0] == ("get_data", {"spec_id": "spec_" + "f" * 32})


async def test_a_saved_plan_is_refused_for_reapproval_once_get_data_takes_statvars() -> None:
    """The surface a plan runs against now declares `resolved_stat_var_dcids`,
    so the saved step's argument schema is no longer the tool's. The run
    stops before any call and says the plan needs approving again."""
    approved = PlannerPlan.model_validate(
        _BASE_FORMAT_PLAN["plan"]
        | {"status": "approved", "approved_revision": _BASE_FORMAT_PLAN["revision"]}
    )
    surface = _CallRecordingSurface(ScriptedSurface())

    _, outcome = await execute_approved_plan(
        approved, surface=surface, fetch_series_stats=scripted_fetch_series_stats
    )

    assert isinstance(outcome, Refused), outcome
    assert outcome.refusal is RefusalKind.SURFACE_CHANGED
    assert outcome.step_id == "retrieve"
    assert outcome.detail == (
        "'get_data' takes different arguments than the plan was compiled against; "
        "the plan needs compiling and approving again"
    )
    assert surface.calls == []


# A population rank as the code at commit 2563638 compiled it against the tool
# schemas the servers declared then, when `get_data` took no
# `resolved_stat_var_dcids`.
_BASE_FORMAT_LIVE_PLAN: dict[str, Any] = json.loads(
    (Path(__file__).parent / "fixtures" / "revision_base_plans.json").read_text()
)["rank"]


async def test_an_approved_plan_compiled_before_get_data_took_statvars_is_refused_for_reapproval(
    real_schemas: list[ToolSchema],
) -> None:
    """Its retrieval step posts no `resolved_stat_var_dcids` and names the
    argument schema `get_data` declared when it was compiled. The run stops
    before any call, with the step and the remedy named, and the plan keeps
    its revision."""
    saved = _BASE_FORMAT_LIVE_PLAN["plan"]
    (retrieval,) = (s for s in saved["workflow"]["steps"] if s["tool"]["name"] == "get_data")
    assert "resolved_stat_var_dcids" not in retrieval["args"]
    approved = PlannerPlan.model_validate(
        saved | {"status": "approved", "approved_revision": _BASE_FORMAT_LIVE_PLAN["revision"]}
    )
    surface = _CallRecordingSurface(_fixture_surface(real_schemas, analysis_tool="rank"))

    _, outcome = await execute_approved_plan(
        approved, surface=surface, fetch_series_stats=scripted_fetch_series_stats
    )

    assert isinstance(outcome, Refused), outcome
    assert outcome.refusal is RefusalKind.SURFACE_CHANGED
    assert outcome.step_id == retrieval["id"]
    assert outcome.detail == (
        "'get_data' takes different arguments than the plan was compiled against; "
        "the plan needs compiling and approving again"
    )
    assert outcome.revision == _BASE_FORMAT_LIVE_PLAN["revision"]
    assert surface.calls == []


async def test_a_refused_amendment_still_offers_the_pending_version_for_approval(
    real_schemas: list[ToolSchema],
) -> None:
    """A `PlannerModel` seam can refuse to draft at all
    (`harness.planner_seam.PlannerSeam`'s own `MAX_PLAN_ROUNDS` cap,
    surfaced as `{"refused": {"text": ...}}` rather than a raised
    `PlannerSeamError`). The planner node never writes `ctx.state["plan"]`
    on that path, so the plan the user already had -- still
    `PENDING_APPROVAL` -- stays exactly as it was. Since that plan is
    still real and still pending, the refusal is not a dead end: the node
    routes back to `approve`, which raises a fresh interrupt for the same
    plan, and answering it with the plan's own original revision approves
    it -- proving the refusal's claim that the version is "still there to
    approve" by actually approving it, not only by reading the sentence
    that says so.
    """
    plan_v1 = _seam_plan(question="kenya debt", schemas=real_schemas)
    seam = _Seam(
        [
            {"plan": plan_v1.model_dump(mode="json"), "summary": {}},
            {
                "refused": {
                    "text": "This request has used its budget of 4 plan drafts, so nothing "
                    "was recorded. Your next message starts with a fresh budget."
                }
            },
        ]
    )
    resolver = ScriptedResolver([_envelope(choice_points=2)])
    executor = ScriptedExecutor()
    workflow = create_root_workflow(
        resolver=resolver,
        executor=executor,
        generative=GenerativeNodes(planner=seam),
        tool_surface=_fixture_surface(real_schemas),
    )
    runner, session_id, events = await _start(workflow, "kenya debt")
    assert _pending_interrupt(events) == APPROVE_INTERRUPT
    revision_1 = plan_revision(plan_v1)
    assert _output_of(events, "planner")["revision"] == revision_1

    events = await _resume(
        runner, session_id, APPROVE_INTERRUPT, {"approved": "not-a-real-revision"}
    )

    # Not a dead end: a fresh approval interrupt for the same, unmodified
    # plan comes back instead, carrying why the amendment did not land.
    assert "deadend" not in _visited(events)
    assert _pending_interrupt(events) == APPROVE_INTERRUPT
    refused_output = _output_of(events, "planner")
    assert refused_output["revision"] == revision_1
    assert "budget of 4 plan drafts" in refused_output["refusal_note"]
    assert "still there to approve" in refused_output["refusal_note"]

    # `ctx.state["plan"]` was only ever written by the first, successful
    # planner call -- the refused second call left it untouched, so the
    # plan the user already saw is still exactly what's on record.
    session = runner.session_service.sessions[runner.app_name]["u"][session_id]
    assert PlannerPlan.model_validate_json(session.state["plan"]) == plan_v1

    # Answering the fresh interrupt with the version's own original
    # revision approves it -- the round-cap refusal cost the user their
    # amendment, not their already-drafted plan. The no-planner `executor`
    # never sees a plan step; the plan's own compiled workflow runs the
    # approved plan through to completion.
    events = await _resume(runner, session_id, APPROVE_INTERRUPT, {"approved": revision_1})
    assert _pending_interrupt(events) is None
    assert executor.calls == []
    session = runner.session_service.sessions[runner.app_name]["u"][session_id]
    approved = PlannerPlan.model_validate_json(session.state["plan"])
    assert approved.status is PlanStatus.COMPLETED


async def test_a_refusal_with_no_pending_plan_ends_the_turn_at_deadend(
    real_schemas: list[ToolSchema],
) -> None:
    """A seam refusal on a first draft -- nothing was ever submitted this
    turn, so `ctx.state["plan"]` holds nothing `PENDING_APPROVAL` to
    re-offer -- takes the `"refused"` route to `deadend_n` rather than the
    `"pending_refusal"` route `approve_n` gets when a plan is still
    pending (see `test_a_refused_amendment_still_offers_the_pending_
    version_for_approval`, just above, for that side). The seam refusal
    shape is the same one `PlannerSeam` uses for both a spent
    `MAX_PLAN_ROUNDS` cap and a `QUESTION_FIXED` restart; the planner node
    only ever branches on whether a pending plan exists, never on which
    code produced the refusal.
    """
    seam = _Seam([{"refused": {"text": "nothing to draft from"}}])
    resolver = ScriptedResolver([_envelope(choice_points=2)])
    workflow = create_root_workflow(
        resolver=resolver,
        executor=ScriptedExecutor(),
        generative=GenerativeNodes(planner=seam),
        tool_surface=None,
    )

    _, _, events = await _start(workflow, "kenya debt")

    assert _pending_interrupt(events) is None
    assert "deadend" in _visited(events)
    assert "approve" not in _visited(events)
    assert _output_of(events, "deadend")["text"] == "nothing to draft from"


async def test_a_fresh_turn_on_one_session_gets_no_prior_plan_even_with_one_on_record(
    real_schemas: list[ToolSchema],
) -> None:
    """`evals/agents/dfo_analyst_graph` (`just adk-web`) keeps one ADK
    session across many turns, unlike studio's own turn-scoped sessions
    (`ConversationState.session_id = f"{conversation_id}#t{turn_index}"`).
    A second turn's own first-draft entry -- reached fresh from `START`,
    through `converse`/`probe`/`choose_depth`, for a brand-new message on
    the *same* session -- is still the `choose_depth` edge, the one edge
    the planner node never passes an existing record on
    (`payload` carries neither `refusal` nor `answer`). It withholds
    `prior_plan` here even though `ctx.state["plan"]` still holds the first
    turn's own, by-then-`COMPLETED` plan: passing it would read as an
    amendment to a `PlannerSeam` built with `question_fixed=True`
    (`harness.planner_seam.PlannerSeam.__init__`), forcing the second
    turn's own question to either amend the first turn's plan or have
    `starts_new_plan` refused as answering a different one, rather than
    draft a plan of its own.
    """
    plan_v1 = _seam_plan(question="kenya debt", schemas=real_schemas)
    plan_v2 = _seam_plan(question="kenya oda 2023", schemas=real_schemas)
    seam = _Seam(
        [
            {"plan": plan_v1.model_dump(mode="json"), "summary": {}},
            {"plan": plan_v2.model_dump(mode="json"), "summary": {}},
        ]
    )
    resolver = ScriptedResolver([_envelope(choice_points=2)])
    executor = ScriptedExecutor()
    workflow = create_root_workflow(
        resolver=resolver,
        executor=executor,
        generative=GenerativeNodes(planner=seam),
        tool_surface=_fixture_surface(real_schemas),
    )
    runner, session_id, events = await _start(workflow, "kenya debt")
    assert _pending_interrupt(events) == APPROVE_INTERRUPT
    assert seam.calls[0]["prior_plan"] is None

    revision_1 = plan_revision(plan_v1)
    events = await _resume(runner, session_id, APPROVE_INTERRUPT, {"approved": revision_1})
    assert _pending_interrupt(events) is None
    session = runner.session_service.sessions[runner.app_name]["u"][session_id]
    assert PlannerPlan.model_validate_json(session.state["plan"]).status is PlanStatus.COMPLETED

    # A brand-new top-level run against the same session, the way `adk web`
    # (not `/turn`, which mints a fresh session every time) starts a second
    # turn: a plain new message, not an interrupt resume.
    events = [
        event
        async for event in runner.run_async(
            user_id="u",
            session_id=session_id,
            new_message=types.Content(
                role="user", parts=[types.Part.from_text(text="kenya oda 2023")]
            ),
        )
    ]

    assert _pending_interrupt(events) == APPROVE_INTERRUPT
    assert seam.calls[1]["prior_plan"] is None
    assert _output_of(events, "planner")["revision"] == plan_revision(plan_v2)


async def test_a_second_turn_on_one_session_carries_no_prior_spec_id_or_counters(
    real_schemas: list[ToolSchema],
) -> None:
    """The same one-session-many-turns host as the test above
    (`evals/agents/dfo_analyst_graph`, `just adk-web`) also keeps every
    `PER_TURN_STATE_KEYS` entry `converse` writes across turns unless
    something clears them: `spec_id`, `spec_id_elected*`, `clarify_rounds`,
    `replans`. Turn 1 below spends a mid-turn refine round (open text,
    proving that re-entry keeps its own counters, since it is `converse`'s
    "refine" edge, not `START`), then a real election, then its one
    replan. Turn 2 -- a brand-new top-level message on the same session --
    must reach `probe` with none of it: no leaked `spec_id` (which
    `McpResolver.__call__` would otherwise read as a clarify re-entry and
    never search for), and a `clarify_rounds` of `0`, proven
    behaviourally: turn 2's own `candidates` envelope asks the user again
    rather than silently auto-picking the way a leaked `clarify_rounds ==
    MAX_CLARIFY_ROUNDS` (2) would.
    """
    turn_1_candidates = [
        {"spec_id": SPEC_A, "label": "reading A"},
        {"spec_id": SPEC_B, "label": "reading B", "wide": True},
    ]
    turn_2_candidates = [
        {"spec_id": SPEC_A, "label": "reading C"},
        {"spec_id": SPEC_B, "label": "reading D"},
    ]
    resolver = ScriptedResolver(
        [
            _envelope(status="candidates", spec_id=None, candidates=turn_1_candidates),
            _envelope(status="candidates", spec_id=None, candidates=turn_1_candidates),
            _envelope(spec_id=SPEC_B, wide=True, choice_points=2),
            _envelope(status="candidates", spec_id=None, candidates=turn_2_candidates),
            _envelope(spec_id=SPEC_A, choice_points=2),
        ]
    )
    plan_v1 = _seam_plan(question="turn 1", schemas=real_schemas)
    plan_v2 = _seam_plan(question="turn 1 replanned", schemas=real_schemas)
    seam = _Seam(
        [
            {"plan": plan_v1.model_dump(mode="json"), "summary": {}},
            {"plan": plan_v2.model_dump(mode="json"), "summary": {}},
        ]
    )
    refusal = ToolCallResult(
        is_error=True,
        structured_content={
            "code": "entity_ambiguous",
            "retry_mode": "correct_and_retry",
            "message": "more than one entity matches 'Kenya'",
            "fix": {"candidates": ["country/KEN", "country/KEN-old"]},
        },
        text="refused",
    )
    surface = _fixture_surface(real_schemas, analysis_call_result=refusal)
    workflow = create_root_workflow(
        resolver=resolver,
        executor=ScriptedExecutor(),
        generative=GenerativeNodes(planner=seam),
        tool_surface=surface,
    )

    # Turn 1, round 1: an open-ended refine, re-entering `converse` on its
    # "refine" edge -- never `START` -- so this must not reset the round it
    # just spent.
    runner, session_id, events = await _start(workflow, "ambiguous q")
    assert _pending_interrupt(events) == CLARIFY_INTERRUPT
    events = await _resume(runner, session_id, CLARIFY_INTERRUPT, {"text": "broaden it"})
    assert _pending_interrupt(events) == CLARIFY_INTERRUPT
    session = runner.session_service.sessions[runner.app_name]["u"][session_id]
    assert session.state["clarify_rounds"] == 1

    # Turn 1, round 2: a real election.
    events = await _resume(runner, session_id, CLARIFY_INTERRUPT, {"spec_id": SPEC_B})
    assert _pending_interrupt(events) == APPROVE_INTERRUPT
    session = runner.session_service.sessions[runner.app_name]["u"][session_id]
    assert session.state["clarify_rounds"] == 2
    assert session.state["spec_id"] == SPEC_B
    assert session.state["spec_id_elected"] is True

    # The analysis refusal spends the turn's one replan.
    revision_1 = plan_revision(plan_v1)
    events = await _resume(runner, session_id, APPROVE_INTERRUPT, {"approved": revision_1})
    assert _pending_interrupt(events) == APPROVE_INTERRUPT
    session = runner.session_service.sessions[runner.app_name]["u"][session_id]
    assert session.state["replans"] == 1

    # The replan budget is spent; this second refusal closes the turn.
    revision_2 = plan_revision(plan_v2)
    events = await _resume(runner, session_id, APPROVE_INTERRUPT, {"approved": revision_2})
    assert _pending_interrupt(events) is None

    # Turn 2: a brand-new top-level message on the SAME session -- the way
    # `adk web` (not `/turn`, which mints a fresh session every time)
    # starts a second turn (see the test above).
    events = [
        event
        async for event in runner.run_async(
            user_id="u",
            session_id=session_id,
            new_message=types.Content(
                role="user", parts=[types.Part.from_text(text="unrelated q")]
            ),
        )
    ]

    # No leaked `spec_id`: `probe`'s own resolver call carries none.
    assert resolver.calls[3]["spec_id"] is None
    # No leaked `clarify_rounds`: turn 1 left `2` (== MAX_CLARIFY_ROUNDS) on
    # record, which would have silently auto-picked the top candidate
    # instead of asking.
    assert _pending_interrupt(events) == CLARIFY_INTERRUPT

    events = await _resume(runner, session_id, CLARIFY_INTERRUPT, {"spec_id": SPEC_A})
    assert _pending_interrupt(events) == APPROVE_INTERRUPT
    assert seam.calls[-1]["prior_plan"] is None

    session = runner.session_service.sessions[runner.app_name]["u"][session_id]
    # One round spent this turn, not the two turn 1 left behind; no leaked
    # replan either.
    assert session.state["clarify_rounds"] == 1
    assert session.state["replans"] == 0
    assert session.state["spec_id"] == SPEC_A
    assert session.state["spec_id_elected_label"] == "reading C"


# --- the execute node: the compiled workflow, the tool surface, the outcome map --


async def test_the_execute_node_runs_the_approved_workflow_through_the_surface(
    real_schemas: list[ToolSchema],
) -> None:
    """`execute` calls `execute_approved_plan` against the injected
    `ToolSurface`; the no-planner `executor` never sees a plan step. Before
    this change the node looped `plan.tool_calls`, which `planner.plan.Plan`
    does not have (`AttributeError`)."""
    plan = _seam_plan(schemas=real_schemas)
    seam = _Seam([{"plan": plan.model_dump(mode="json"), "summary": {}}])
    resolver = ScriptedResolver([_envelope(choice_points=2)])
    executor = ScriptedExecutor()
    surface = _CallRecordingSurface(_fixture_surface(real_schemas))
    workflow = create_root_workflow(
        resolver=resolver,
        executor=executor,
        generative=GenerativeNodes(planner=seam),
        tool_surface=surface,
    )
    runner, session_id, events = await _start(workflow, "kenya debt")
    revision = plan_revision(plan)

    events = await _resume(runner, session_id, APPROVE_INTERRUPT, {"approved": revision})

    assert [call[0] for call in surface.calls] == ["get_data", "aggregate"]
    assert executor.calls == []
    executed = _output_of(events, "execute")
    # `Accepted.results` is terminal-steps-only: one frame ref
    # for "totals", not one per step.
    assert executed["frame_refs"] == [
        {"spec_id": "ops_" + "2" * 32, "content_version": "ocv_" + "3" * 32}
    ]
    assert executed["step_summaries"] == ["fetched", "computed"]
    assert executed["plan_version"] == 1
    assert _output_of(events, "narrate")["plan_version"] == 1


@pytest.mark.parametrize("drafts", [True, False], ids=["model", "scripted"])
async def test_a_summed_familys_total_reaches_the_close_even_when_the_model_drops_it(
    real_schemas: list[ToolSchema], monkeypatch: pytest.MonkeyPatch, drafts: bool
) -> None:
    """`execute` hands `narrate` the sentences `_stat_var_totals` writes, and
    `narrate` appends each one its draft leaves out, on the model path and
    the scripted one alike."""
    sentence = (
        'By default, the figures for "OECD Creditor Reporting System (CRS)" sum the '
        "family's 74 indicators."
    )
    monkeypatch.setattr(graph_module, "_stat_var_totals", lambda *a, **k: [sentence])
    plan = _seam_plan(schemas=real_schemas)
    seam = _Seam([{"plan": plan.model_dump(mode="json"), "summary": {}}])
    seen: list[dict[str, Any]] = []

    class _DropsTheTotal:
        async def __call__(self, *, question: str, facts: dict[str, Any]) -> dict[str, Any]:
            del question
            seen.append(facts)
            return {"text": "Spending rose."}

    workflow = create_root_workflow(
        resolver=ScriptedResolver([_envelope(choice_points=2)]),
        executor=ScriptedExecutor(),
        generative=GenerativeNodes(planner=seam, narrate=_DropsTheTotal() if drafts else None),
        tool_surface=_fixture_surface(real_schemas),
    )
    runner, session_id, _ = await _start(workflow, "kenya debt")

    events = await _resume(runner, session_id, APPROVE_INTERRUPT, {"approved": plan_revision(plan)})

    text = _output_of(events, "narrate")["text"]
    assert text.endswith(f" {sentence}")
    assert text.count(sentence) == 1
    if drafts:
        assert text == f"Spending rose. {sentence}"
        assert seen[0]["stat_var_totals"] == [sentence]


async def test_a_none_tool_surface_refuses_loudly_rather_than_executing_nothing(
    real_schemas: list[ToolSchema],
) -> None:
    """`create_root_workflow(tool_surface=None)`, stated explicitly, must not
    let an approved plan reach execute silently: reading a plan the node
    cannot run has to fail loudly."""
    plan = _seam_plan(schemas=real_schemas)
    seam = _Seam([{"plan": plan.model_dump(mode="json"), "summary": {}}])
    resolver = ScriptedResolver([_envelope(choice_points=2)])
    workflow = create_root_workflow(
        resolver=resolver,
        executor=ScriptedExecutor(),
        generative=GenerativeNodes(planner=seam),
        tool_surface=None,
    )
    runner, session_id, _ = await _start(workflow, "kenya debt")
    revision = plan_revision(plan)

    with pytest.raises(RuntimeError, match="no tool surface"):
        await _resume(runner, session_id, APPROVE_INTERRUPT, {"approved": revision})


async def test_a_rediscovery_outcome_replans_naming_the_requirement(
    real_schemas: list[ToolSchema],
) -> None:
    """A retrieval step's own evidence binding names the requirement and the
    source a `RediscoveryNeeded` refusal reports -- a bare retry would only
    reproduce it, so `execute` routes back to the planner naming what to
    rediscover, never retrying the call itself."""
    plan = _seam_plan(schemas=real_schemas, bind_evidence=True)
    seam = _Seam([{"plan": plan.model_dump(mode="json"), "summary": {}}])
    resolver = ScriptedResolver([_envelope(choice_points=2)])
    refusal = ToolCallResult(
        is_error=True,
        structured_content={
            "code": "upstream_version_changed",
            "retry_mode": "refresh_then_retry",
            "message": "the source's upstream version changed since this plan was approved",
        },
        text="refused",
    )
    surface = _fixture_surface(real_schemas, get_data_result=refusal)
    workflow = create_root_workflow(
        resolver=resolver,
        executor=ScriptedExecutor(),
        generative=GenerativeNodes(planner=seam),
        tool_surface=surface,
    )
    runner, session_id, events = await _start(workflow, "kenya debt")
    revision = plan_revision(plan)

    events = await _resume(runner, session_id, APPROVE_INTERRUPT, {"approved": revision})

    # Replanned (MAX_REPLANS == 1 covers one round): a fresh approval is
    # pending rather than the turn closing on the first refusal.
    assert _pending_interrupt(events) == APPROVE_INTERRUPT
    assert seam.calls[-1]["refusal"] == {
        "kind": "rediscovery_needed",
        "step_id": "get_data",
        "tool": "get_data",
        "requirement_id": "spending",
        "spec_id": SPEC_A,
        "candidates": None,
        "refusal": {
            "code": "upstream_version_changed",
            "retry_mode": "refresh_then_retry",
            "message": "the source's upstream version changed since this plan was approved",
            "fix": {},
        },
        "also_stopped": [],
    }


async def test_a_method_choice_refusal_replans_once_then_closes_with_failed_step(
    real_schemas: list[ToolSchema],
) -> None:
    """`MAX_REPLANS == 1`: a second identical refusal closes the turn with
    what ran and the failed step named, rather than a third plan."""
    plan = _seam_plan(schemas=real_schemas)
    seam = _Seam([{"plan": plan.model_dump(mode="json"), "summary": {}}])
    resolver = ScriptedResolver([_envelope(choice_points=2)])
    refusal = ToolCallResult(
        is_error=True,
        structured_content={
            "code": "entity_ambiguous",
            "retry_mode": "correct_and_retry",
            "message": "more than one entity matches 'Kenya'",
            "fix": {"candidates": ["country/KEN", "country/KEN-old"]},
        },
        text="refused",
    )
    surface = _fixture_surface(real_schemas, analysis_call_result=refusal)
    executor = ScriptedExecutor()
    workflow = create_root_workflow(
        resolver=resolver,
        executor=executor,
        generative=GenerativeNodes(planner=seam),
        tool_surface=surface,
    )
    runner, session_id, events = await _start(workflow, "kenya debt")
    revision = plan_revision(plan)

    events = await _resume(runner, session_id, APPROVE_INTERRUPT, {"approved": revision})
    assert _pending_interrupt(events) == APPROVE_INTERRUPT
    replanned = _output_of(events, "planner")
    events = await _resume(
        runner, session_id, APPROVE_INTERRUPT, {"approved": replanned["revision"]}
    )

    assert _pending_interrupt(events) is None
    executed = _output_of(events, "execute")
    assert executed["failed_step"] == {
        "step_id": "totals",
        "tool": "aggregate",
        "code": "entity_ambiguous",
        "message": "more than one entity matches 'Kenya'",
    }
    # get_data ran and completed before the refusal; its frame is kept.
    assert executed["frame_refs"] == [REF_A]
    assert executed["step_summaries"] == ["fetched"]
    narrated = _output_of(events, "narrate")
    assert "Step 'totals' (aggregate) could not run" in narrated["text"]
    assert "entity_ambiguous" in narrated["text"]


async def test_a_stop_classified_refusal_closes_the_turn_without_spending_a_replan(
    real_schemas: list[ToolSchema],
) -> None:
    """An `over_ceiling` refusal naming no `ceiling_kind` at all -- an older
    upstream, or a raw payload assembled by hand -- is one
    `executor.classify.classify_refusal` cannot place as narrowing with scope
    or as naming a different spec, so it stays a `STOP`: `execute` closes the
    turn on this first refusal instead of spending the turn's one replan on a
    guess. Unlike `test_a_method_choice_refusal_
    replans_once_then_closes_with_failed_step`'s `entity_ambiguous`, no
    second approval is ever pending."""
    plan = _seam_plan(schemas=real_schemas)
    seam = _Seam([{"plan": plan.model_dump(mode="json"), "summary": {}}])
    resolver = ScriptedResolver([_envelope(choice_points=2)])
    refusal = ToolCallResult(
        is_error=True,
        structured_content={
            "code": "over_ceiling",
            "retry_mode": "correct_and_retry",
            "message": "over the coverage ceiling: too many series.",
            "fix": {"detail": "too many series"},
        },
        text="refused",
    )
    surface = _fixture_surface(real_schemas, analysis_call_result=refusal)
    workflow = create_root_workflow(
        resolver=resolver,
        executor=ScriptedExecutor(),
        generative=GenerativeNodes(planner=seam),
        tool_surface=surface,
    )
    runner, session_id, events = await _start(workflow, "kenya debt")
    revision = plan_revision(plan)

    events = await _resume(runner, session_id, APPROVE_INTERRUPT, {"approved": revision})

    # No replan, no second approval: the turn closes on the first refusal.
    assert _pending_interrupt(events) is None
    executed = _output_of(events, "execute")
    # A plain `Refused` (the `_stopped_by_refusal` catch-all `STEP_REFUSED`,
    # not a `MethodChoiceNeeded`/`RediscoveryNeeded`) carries `tool=None`,
    # but `_failed_step` reports the tool's own `code`/`message` off
    # `outcome.payload` -- the `ToolFailure` the raw call actually returned
    # -- rather than the internal `RefusalKind` that closed the turn.
    assert executed["failed_step"] == {
        "step_id": "totals",
        "tool": None,
        "code": "over_ceiling",
        "message": "over the coverage ceiling: too many series.",
    }
    assert executed["frame_refs"] == [REF_A]
    narrated = _output_of(events, "narrate")
    assert "Step 'totals' could not run" in narrated["text"]
    assert "over_ceiling" in narrated["text"]


async def test_a_member_cardinality_ceiling_closes_the_turn_without_replanning(
    real_schemas: list[ToolSchema],
) -> None:
    """A `member_cardinality` `over_ceiling` refusal is fixed by the resolved
    spec_id's own five-tuple and the index version behind it: no scope edit
    narrows it, and rediscovering the same spec by paraphrase search
    reproduces the identical refusal. `classify_refusal` classifies it
    `STOP`, so `execute` closes the turn on this first refusal instead of
    spending the turn's one replan on a rediscovery that cannot help."""
    plan = _seam_plan(schemas=real_schemas, bind_evidence=True)
    seam = _Seam([{"plan": plan.model_dump(mode="json"), "summary": {}}])
    resolver = ScriptedResolver([_envelope(choice_points=2)])
    refusal = ToolCallResult(
        is_error=True,
        structured_content={
            "code": "over_ceiling",
            "retry_mode": "correct_and_retry",
            "message": "over the coverage ceiling: 2933 members. A narrower question "
            "resolves to a smaller definition.",
            "fix": {
                "detail": "2933 members",
                "ceiling_kind": "member_cardinality",
                "count": 2933,
                "cap": 700,
            },
        },
        text="refused",
    )
    surface = _fixture_surface(real_schemas, get_data_result=refusal)
    workflow = create_root_workflow(
        resolver=resolver,
        executor=ScriptedExecutor(),
        generative=GenerativeNodes(planner=seam),
        tool_surface=surface,
    )
    runner, session_id, events = await _start(workflow, "kenya debt")
    revision = plan_revision(plan)

    events = await _resume(runner, session_id, APPROVE_INTERRUPT, {"approved": revision})

    # No replan, no second approval: the turn closes on the first refusal.
    assert _pending_interrupt(events) is None
    executed = _output_of(events, "execute")
    assert executed["failed_step"] == {
        "step_id": "get_data",
        "tool": None,
        "code": "over_ceiling",
        "message": "over the coverage ceiling: 2933 members. A narrower question "
        "resolves to a smaller definition.",
    }
    narrated = _output_of(events, "narrate")
    assert "Step 'get_data' could not run" in narrated["text"]
    assert "over_ceiling" in narrated["text"]


async def test_a_member_cardinality_ceilings_remedy_reaches_the_close_even_when_the_model_drops_it(
    real_schemas: list[ToolSchema],
) -> None:
    """`fix.remedy` -- the one user-facing sentence naming what a
    `member_cardinality` ceiling's dead end leaves the user to do -- must
    reach the turn's closing text by code, not by a narrator model's own
    cooperation. The stub narrator here drafts text that omits it entirely
    (the exact failure mode a prompt-only fix cannot guard against); `narrate`
    still appends it verbatim, unrelated to whatever the model drafted."""
    plan = _seam_plan(schemas=real_schemas, bind_evidence=True)
    seam = _Seam([{"plan": plan.model_dump(mode="json"), "summary": {}}])
    resolver = ScriptedResolver([_envelope(choice_points=2)])
    remedy = (
        "This spec_id's definition has more member series than one request can "
        "serve. scope.entities and scope.window do not change that. A narrower "
        "question, such as one naming a sector for aid flows, resolves to a "
        "smaller definition."
    )
    refusal = ToolCallResult(
        is_error=True,
        structured_content={
            "code": "over_ceiling",
            "retry_mode": "correct_and_retry",
            "message": f"over the coverage ceiling: 2933 members. {remedy}",
            "fix": {
                "detail": "2933 members",
                "ceiling_kind": "member_cardinality",
                "count": 2933,
                "cap": 700,
                "remedy": remedy,
            },
        },
        text="refused",
    )
    surface = _fixture_surface(real_schemas, get_data_result=refusal)

    class _DropsTheRemedy:
        async def __call__(self, *, question: str, facts: dict[str, Any]) -> dict[str, Any]:
            del question, facts
            return {"text": "The data could not be presented."}

    workflow = create_root_workflow(
        resolver=resolver,
        executor=ScriptedExecutor(),
        generative=GenerativeNodes(planner=seam, narrate=_DropsTheRemedy()),
        tool_surface=surface,
    )
    runner, session_id, events = await _start(workflow, "kenya debt")
    revision = plan_revision(plan)

    events = await _resume(runner, session_id, APPROVE_INTERRUPT, {"approved": revision})

    assert _pending_interrupt(events) is None
    executed = _output_of(events, "execute")
    assert executed["failed_step"]["remedy"] == remedy
    narrated = _output_of(events, "narrate")
    assert narrated["text"] == f"The data could not be presented. {remedy}"


async def test_a_family_returned_unsummed_closes_the_turn_on_a_message_the_user_can_act_on(
    real_schemas: list[ToolSchema],
) -> None:
    """A `get_data` asked to sum its family declares three StatVars, so the
    sum did not happen. The same plan draws the same frame, so `execute`
    closes the turn on this first refusal: no replan, no second approval of
    an identical plan. The closing text carries the executor's own message
    and remedy, both written for the user, in place of the count of
    `stat_vars` the refusal's `detail` states."""
    plan = _seam_plan(schemas=real_schemas, stat_vars="sum")
    seam = _Seam([{"plan": plan.model_dump(mode="json"), "summary": {}}])
    unsummed = ToolCallResult(
        is_error=False,
        structured_content=_get_data_manifest() | {"stat_var_count": 3},
        text="fetched",
    )
    surface = _fixture_surface(real_schemas, get_data_result=unsummed)
    workflow = create_root_workflow(
        resolver=ScriptedResolver([_envelope(choice_points=2)]),
        executor=ScriptedExecutor(),
        generative=GenerativeNodes(planner=seam),
        tool_surface=surface,
    )
    runner, session_id, events = await _start(workflow, "kenya debt")

    events = await _resume(runner, session_id, APPROVE_INTERRUPT, {"approved": plan_revision(plan)})

    assert _pending_interrupt(events) is None
    assert len(seam.calls) == 1
    message = (
        "The data service returned this measure's component series side by side and did "
        "not add them up, so there is no total to report."
    )
    remedy = "Ask for the component series separately, or ask for the total again later."
    executed = _output_of(events, "execute")
    assert executed["failed_step"] == {
        "step_id": "get_data",
        "tool": None,
        "code": "family_not_summed",
        "message": message,
        "remedy": remedy,
    }
    assert executed["frame_refs"] == []
    narrated = _output_of(events, "narrate")
    assert narrated["text"].endswith(f"family_not_summed -- {message} {remedy}")
    assert "stat_vars" not in narrated["text"]


async def test_an_over_ceiling_rows_refusal_still_replans(
    real_schemas: list[ToolSchema],
) -> None:
    """A `rows`/`bytes` ceiling narrows with `scope.entities`/`scope.window`,
    so -- unlike `member_cardinality` above -- it is a method choice, not a
    dead end, and replans once like any other correctable refusal."""
    plan = _seam_plan(schemas=real_schemas)
    seam = _Seam([{"plan": plan.model_dump(mode="json"), "summary": {}}])
    resolver = ScriptedResolver([_envelope(choice_points=2)])
    refusal = ToolCallResult(
        is_error=True,
        structured_content={
            "code": "over_ceiling",
            "retry_mode": "correct_and_retry",
            "message": "over the coverage ceiling: too many rows. Narrow the scope and retry.",
            "fix": {"detail": "too many rows", "ceiling_kind": "rows"},
        },
        text="refused",
    )
    surface = _fixture_surface(real_schemas, analysis_call_result=refusal)
    workflow = create_root_workflow(
        resolver=resolver,
        executor=ScriptedExecutor(),
        generative=GenerativeNodes(planner=seam),
        tool_surface=surface,
    )
    runner, session_id, events = await _start(workflow, "kenya debt")
    revision = plan_revision(plan)

    events = await _resume(runner, session_id, APPROVE_INTERRUPT, {"approved": revision})

    # A method choice, not a dead end: the planner is asked again (the
    # scripted seam redrafts the same plan, so the version does not bump),
    # and a fresh approval is pending rather than the turn closing.
    assert _pending_interrupt(events) == APPROVE_INTERRUPT
    assert len(seam.calls) == 2
    assert seam.calls[-1]["refusal"]["code"] == "over_ceiling"


async def test_a_not_approved_outcome_raises_instead_of_replanning(
    real_schemas: list[ToolSchema], monkeypatch: pytest.MonkeyPatch
) -> None:
    """`Refused(NOT_APPROVED)` means the graph's own `approve` node let an
    unapproved plan through -- replanning would loop on the same defect, so
    `execute` raises instead. The graph's `approve` node cannot itself
    produce this in normal operation (it calls `plan.approve()` immediately
    before routing here), so the outcome is stubbed to prove this node's own
    branch rather than trying to coax `approve` into a defect it structurally
    prevents."""
    plan = _seam_plan(schemas=real_schemas)
    seam = _Seam([{"plan": plan.model_dump(mode="json"), "summary": {}}])
    resolver = ScriptedResolver([_envelope(choice_points=2)])

    async def _stub_not_approved(
        drafted_plan: Any, *, surface: Any, max_concurrent_steps: int = 4, **_kwargs: Any
    ) -> Any:
        del surface, max_concurrent_steps
        return drafted_plan, Refused(
            revision=plan_revision(drafted_plan),
            refusal=RefusalKind.NOT_APPROVED,
            detail="this plan is approved, not approved; nothing runs",
        )

    monkeypatch.setattr("harness.graph.execute_approved_plan", _stub_not_approved)
    workflow = create_root_workflow(
        resolver=resolver,
        executor=ScriptedExecutor(),
        generative=GenerativeNodes(planner=seam),
        tool_surface=_fixture_surface(real_schemas),
    )
    runner, session_id, _ = await _start(workflow, "kenya debt")
    revision = plan_revision(plan)

    with pytest.raises(RuntimeError, match="not_approved"):
        await _resume(runner, session_id, APPROVE_INTERRUPT, {"approved": revision})


def test_the_replan_refusal_and_failed_step_maps_cover_every_outcome_kind() -> None:
    """`_replan_refusal`'s payload shapes, pinned directly against each outcome
    kind -- not merely reachable through one graph run apiece."""
    method_choice = MethodChoiceNeeded(
        revision="rev1",
        step_id="totals",
        tool=ToolId(server="analysis-mcp", name="aggregate"),
        code="entity_ambiguous",
        candidates=["country/KEN", "country/KEN-old"],
        refusal=ToolFailure(
            code="entity_ambiguous", message="ambiguous", retry_mode="correct_and_retry"
        ),
        completed=(),
    )
    assert _replan_refusal(method_choice) == {
        "kind": "method_choice_needed",
        "step_id": "totals",
        "tool": "aggregate",
        "code": "entity_ambiguous",
        "candidates": ["country/KEN", "country/KEN-old"],
        "refusal": {
            "code": "entity_ambiguous",
            "message": "ambiguous",
            "retry_mode": "correct_and_retry",
            "fix": {},
        },
        "also_stopped": [],
    }
    assert _failed_step(method_choice) == {
        "step_id": "totals",
        "tool": "aggregate",
        "code": "entity_ambiguous",
        "message": "ambiguous",
    }

    rediscovery = RediscoveryNeeded(
        revision="rev1",
        step_id="get_data",
        tool=ToolId(server="data-mcp", name="get_data"),
        requirement_id="spending",
        spec_id=SPEC_A,
        refusal=ToolFailure(
            code="upstream_version_changed",
            message="the source moved",
            retry_mode="refresh_then_retry",
        ),
        completed=(),
    )
    assert _replan_refusal(rediscovery) == {
        "kind": "rediscovery_needed",
        "step_id": "get_data",
        "tool": "get_data",
        "requirement_id": "spending",
        "spec_id": SPEC_A,
        # An empty `fix` carries nothing to choose from -- `candidates_of`
        # reads `None`, not an empty dict standing in for "no candidates".
        "candidates": None,
        "refusal": {
            "code": "upstream_version_changed",
            "message": "the source moved",
            "retry_mode": "refresh_then_retry",
            "fix": {},
        },
        "also_stopped": [],
    }
    assert _failed_step(rediscovery) == {
        "step_id": "get_data",
        "tool": "get_data",
        "code": "rediscovery_needed",
        "message": "the source moved",
    }

    refused = Refused(
        revision="rev1",
        refusal=RefusalKind.EMPTY_ANSWER,
        detail="no rows matched",
        step_id="totals",
    )
    assert _replan_refusal(refused) == {
        "kind": "refused",
        "refusal": "empty_answer",
        "detail": "no rows matched",
        "step_id": "totals",
        "requirement_id": None,
        "issues": [],
        "also_stopped": [],
    }
    assert _failed_step(refused) == {
        "step_id": "totals",
        "tool": None,
        "code": "empty_answer",
        "message": "no rows matched",
    }

    # A `STEP_REFUSED` stop carries the tool's own `ToolFailure` as
    # `payload` -- `_failed_step` reports its `code`/`message`, not
    # `refusal.value`/`detail`, so the narration model is told what the
    # tool actually said ("over_ceiling") rather than the internal kind
    # that closed the turn ("step_refused").
    refused_with_payload = Refused(
        revision="rev1",
        refusal=RefusalKind.STEP_REFUSED,
        detail="'aggregate' refused: refused",
        step_id="totals",
        payload=ToolFailure(
            code="over_ceiling",
            message="over the coverage ceiling: too many series",
            retry_mode="correct_and_retry",
        ),
    )
    assert _failed_step(refused_with_payload) == {
        "step_id": "totals",
        "tool": None,
        "code": "over_ceiling",
        "message": "over the coverage ceiling: too many series",
    }

    # A `REQUIREMENT_UNMET` stop carries `check_evidence`'s own findings as
    # `issues` -- `_failed_step` reports the specific `EvidenceIssueCode`
    # and detail of whichever ERROR(s) fired, not `refusal.value`/`detail`'s
    # generic "requirement X is not met by what Y returned", which names
    # neither which of the ten sub-checks failed nor what it found. A
    # `WARNING`-severity issue on the same outcome (a real shape:
    # `check_evidence` can return caveats alongside the errors that stopped
    # the run) is left out of the reported code/message; only errors decide
    # what the run reports as its reason.
    refused_with_issues = Refused(
        revision="rev1",
        refusal=RefusalKind.REQUIREMENT_UNMET,
        detail="requirement 'req_gdp' is not met by what 'get_data' returned",
        step_id="fetch_gdp",
        requirement_id="req_gdp",
        issues=(
            EvidenceIssue(
                code=EvidenceIssueCode.SOURCE_UNVERIFIED,
                severity=Severity.WARNING,
                requirement_id="req_gdp",
                detail="the source states no provenance",
            ),
            EvidenceIssue(
                code=EvidenceIssueCode.SOURCE_PIN_MISMATCH,
                severity=Severity.ERROR,
                requirement_id="req_gdp",
                detail="the requirement pins 'spec_a', the source found is 'spec_b'",
                found="spec_b",
                required="spec_a",
            ),
        ),
    )
    assert _failed_step(refused_with_issues) == {
        "step_id": "fetch_gdp",
        "tool": None,
        "code": "source_pin_mismatch",
        "message": "the requirement pins 'spec_a', the source found is 'spec_b'",
    }

    # No ERROR-severity issue at all (every finding a caveat) falls back to
    # the first issue given rather than reporting nothing -- a defensive
    # case `check_evidence` should never actually produce (it only raises
    # when `errors(findings)` is non-empty), but `_failed_step` still owes
    # a code/message pair for whatever `issues` it is handed.
    refused_with_only_warnings = Refused(
        revision="rev1",
        refusal=RefusalKind.REQUIREMENT_UNMET,
        detail="requirement 'req_gdp' is not met by what 'get_data' returned",
        step_id="fetch_gdp",
        requirement_id="req_gdp",
        issues=(
            EvidenceIssue(
                code=EvidenceIssueCode.SOURCE_UNVERIFIED,
                severity=Severity.WARNING,
                requirement_id="req_gdp",
                detail="the source states no provenance",
            ),
        ),
    )
    assert _failed_step(refused_with_only_warnings) == {
        "step_id": "fetch_gdp",
        "tool": None,
        "code": "source_unverified",
        "message": "the source states no provenance",
    }


def test_replan_refusal_names_every_sibling_a_wave_surfaced() -> None:
    """`also_stopped` carries the rest of a wave's stops in the primary's own
    per-kind shape, empty when the wave surfaced only the one outcome
    (`executor.run._Run.execute`'s siblings, folded into the outcome
    `_run_workflow` returns)."""
    sibling = RediscoveryNeeded(
        revision="rev1",
        step_id="fetch_population",
        tool=ToolId(server="data-mcp", name="get_data"),
        requirement_id="population",
        spec_id=SPEC_A,
        refusal=ToolFailure(
            code="upstream_version_changed",
            message="the population source moved too",
            retry_mode="refresh_then_retry",
        ),
        completed=(),
    )
    primary = RediscoveryNeeded(
        revision="rev1",
        step_id="get_data",
        tool=ToolId(server="data-mcp", name="get_data"),
        requirement_id="spending",
        spec_id=SPEC_A,
        refusal=ToolFailure(
            code="upstream_version_changed",
            message="the source moved",
            retry_mode="refresh_then_retry",
        ),
        completed=(),
        also_stopped=(sibling,),
    )

    payload = _replan_refusal(primary)

    assert payload["also_stopped"] == [
        {
            "kind": "rediscovery_needed",
            "step_id": "fetch_population",
            "tool": "get_data",
            "requirement_id": "population",
            "spec_id": SPEC_A,
            "candidates": None,
            "refusal": {
                "code": "upstream_version_changed",
                "message": "the population source moved too",
                "retry_mode": "refresh_then_retry",
                "fix": {},
            },
        }
    ]
    assert _replan_refusal(sibling)["also_stopped"] == []


def test_mark_masked_step_failed_corrects_every_sibling_a_wave_surfaced() -> None:
    """A wave whose primary and one `also_stopped` sibling both passed their
    raw call but were refused by a post-call check (`_check_evidence`) each
    carry a distinct `step_id` -- `_mark_masked_step_failed` corrects both
    Activity feed records, not just the primary's."""

    class _RecorderStub:
        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []

        def mark_step_failed(self, *, step_id: str, tool: str, code: str, detail: str) -> None:
            self.calls.append({"step_id": step_id, "code": code, "detail": detail})

    sibling = Refused(
        revision="rev1",
        refusal=RefusalKind.EMPTY_ANSWER,
        detail="no rows matched",
        step_id="fetch_population",
    )
    primary = Refused(
        revision="rev1",
        refusal=RefusalKind.REQUIREMENT_UNMET,
        detail="requirement 'req_gdp' is not met by what 'get_data' returned",
        step_id="fetch_gdp",
        requirement_id="req_gdp",
        issues=(
            EvidenceIssue(
                code=EvidenceIssueCode.SOURCE_PIN_MISMATCH,
                severity=Severity.ERROR,
                requirement_id="req_gdp",
                detail="the requirement pins 'spec_a', the source found is 'spec_b'",
                found="spec_b",
                required="spec_a",
            ),
        ),
        also_stopped=(sibling,),
    )

    recorder = _RecorderStub()
    _mark_masked_step_failed(recorder, primary, tools={})

    assert recorder.calls == [
        {
            "step_id": "fetch_gdp",
            "code": "source_pin_mismatch",
            "detail": "the requirement pins 'spec_a', the source found is 'spec_b'",
        },
        {
            "step_id": "fetch_population",
            "code": "empty_answer",
            "detail": "no rows matched",
        },
    ]


# --- depth hints through the graph (scripted extractor; tests stay model-free) --


async def test_thorough_hint_invokes_planner_on_a_clean_lookup() -> None:
    resolver = ScriptedResolver([_envelope()])  # definite, 0 forks: structurally "none"
    executor = ScriptedExecutor()
    workflow = create_root_workflow(
        resolver=resolver,
        executor=executor,
        depth_hint_extractor=lambda *, question: "thorough",
        generative=GenerativeNodes(planner=ScriptedPlannerModel()),
        tool_surface=None,
    )
    _, _, events = await _start(workflow, "give me a comprehensive overview of kenya oda")

    assert _output_of(events, "converse")["depth_hint"] == "thorough"
    depth = _output_of(events, "choose_depth")
    assert depth["depth"] == "standard"
    assert "user asked for thorough" in depth["reason"]
    assert "planner" in _visited(events)
    assert _pending_interrupt(events) == APPROVE_INTERRUPT


async def test_quick_hint_lowers_but_keeps_planner_when_forks_exist() -> None:
    resolver = ScriptedResolver([_envelope(choice_points=2)])  # structurally "standard"
    executor = ScriptedExecutor()
    workflow = create_root_workflow(
        resolver=resolver,
        executor=executor,
        depth_hint_extractor=lambda *, question: "quick",
        generative=GenerativeNodes(planner=ScriptedPlannerModel()),
        fetch_series_stats=scripted_fetch_series_stats,
        tool_surface=ScriptedSurface(),
    )
    runner, session_id, events = await _start(workflow, "quickly, kenya debt vs climate")

    depth = _output_of(events, "choose_depth")
    assert depth["depth"] == "shallow"
    assert depth["max_clarifying_questions"] == 1
    assert "planner" in _visited(events)
    assert _pending_interrupt(events) == APPROVE_INTERRUPT

    plan_payload = _output_of(events, "planner")
    events = await _resume(
        runner, session_id, APPROVE_INTERRUPT, {"approved": plan_payload["revision"]}
    )
    assert _output_of(events, "narrate") is not None


async def test_planner_output_carries_the_depth_verdict_and_its_notice() -> None:
    """`choose_depth`'s verdict rides on the planner node's own output, so
    the approve interrupt's payload (and, through it, `plan.proposed`) can
    say why a plan is being proposed."""
    resolver = ScriptedResolver([_envelope(parts=2)])  # compound, 0 choice points: "standard"
    workflow = create_root_workflow(
        resolver=resolver,
        executor=ScriptedExecutor(),
        generative=GenerativeNodes(planner=ScriptedPlannerModel()),
        fetch_series_stats=scripted_fetch_series_stats,
        tool_surface=ScriptedSurface(),
    )
    _, _, events = await _start(workflow, "population and gdp of guatemala")

    depth = _output_of(events, "choose_depth")
    assert depth["depth"] == "standard"
    assert depth["notice"] == (
        "This asks for more than one thing, so I am proposing a plan before fetching anything."
    )

    plan_payload = _output_of(events, "planner")
    assert plan_payload["depth"] == "standard"
    assert plan_payload["depth_notice"] == depth["notice"]


async def test_extractor_failure_surfaces_instead_of_flattening_to_none() -> None:
    def broken_extractor(*, question: str) -> str:
        raise RuntimeError("hint classifier unavailable")

    resolver = ScriptedResolver([_envelope()])
    workflow = create_root_workflow(
        resolver=resolver,
        executor=ScriptedExecutor(),
        depth_hint_extractor=broken_extractor,  # type: ignore[arg-type]
        tool_surface=None,
    )
    with pytest.raises(RuntimeError, match="hint classifier unavailable"):
        await _start(workflow, "any question")


# --- seeded turns: Search results handed to Studio --------------------------


@pytest.fixture
def definition_store() -> Any:
    """The definition store the process holds, fresh per test: data-mcp's, when it runs in the
    harness's process (`DataMcpDefinitions()`)."""
    store = InMemoryDefinitionStore()
    set_definition_store(store)
    yield store
    set_definition_store(None)


class _UnreachableStore:
    """A definition store whose backend cannot be reached."""

    async def put_many(self, definitions: Any) -> None:
        raise DefinitionStoreUnavailable("backend unreachable")

    async def get(self, spec_id: str, *, schema_version: int) -> None:
        raise DefinitionStoreUnavailable("backend unreachable")

    async def aclose(self) -> None:
        pass


def _seed(
    name: str,
    *,
    operations: list[dict[str, Any]] | None = None,
    resolved: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """One `harness.wire.Seed`, as `studio.py` puts it in session state, whose
    identity hashes to its spec_id."""
    identity = {"identity_version": "spec1", "seed": name, "requested_window": {"kind": "all"}}
    spec_id = spec_id_for(canonicalize(identity))
    return {
        "spec": {
            "spec_id": spec_id,
            "definition": {"schema_version": 1, "spec_id": spec_id, "identity": identity},
            "resolved": {
                "rendered_sentence": f"the reading {name}",
                "coverage": {
                    "kind": "breadth",
                    "has_data": True,
                    "dimensions": [{"label": "recipients", "count": 199}],
                },
                "stat_vars": {"items": [{"ref": f"Var_{name}", "slot_values": []}]},
                **(resolved or {}),
            },
        },
        "operations": operations or [],
    }


def _spec_id(seed: dict[str, Any]) -> str:
    return seed["spec"]["spec_id"]


_RANK_DONORS = {"verb": "rank", "phrase": "top 5 donors", "over": "origin"}
_HEADLINE = {"year": 2024, "year_named": False, "observations": 13356, "series": 13356}


class _SeedExecutor:
    """One `get_data` per call, minted under the spec_id and scope it was
    given; `steps` collects what `record_step` was handed."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.steps: list[dict[str, Any]] = []
        self.messages: dict[str, str] = {}

    def record_step(
        self, *, tool: str, args: dict[str, Any], result: dict[str, Any], step_id: str
    ) -> None:
        refusal = result.get("refusal") or {}
        if refusal.get("message"):
            self.messages[step_id] = refusal["message"]
        self.steps.append(
            {
                "tool": tool,
                "step_id": step_id,
                "outcome": result["outcome"],
                "code": refusal.get("code"),
            }
        )

    async def __call__(
        self, *, tool: str, args: dict[str, Any], on_progress: ProgressCallback | None = None
    ) -> dict[str, Any]:
        del on_progress
        self.calls.append({"tool": tool, "args": args, "step_id": current_step_id.get()})
        ref = {"spec_id": args.get("spec_id"), "content_version": "cv_" + "9" * 32}
        return {"outcome": "ok", "frame_refs": [ref], "summary": "minted"}


async def _start_seeded(
    workflow: Any, question: str, seeds: list[dict[str, Any]]
) -> tuple[InMemoryRunner, str, list[Event]]:
    runner = InMemoryRunner(node=workflow)
    session = await runner.session_service.create_session(
        app_name=runner.app_name, user_id="u", state={SEEDS_STATE_KEY: json.dumps(seeds)}
    )
    events = [
        event
        async for event in runner.run_async(
            user_id="u",
            session_id=session.id,
            new_message=types.Content(role="user", parts=[types.Part.from_text(text=question)]),
        )
    ]
    return runner, session.id, events


async def _state(runner: InMemoryRunner, session_id: str) -> dict[str, Any]:
    session = await runner.session_service.get_session(
        app_name=runner.app_name, user_id="u", session_id=session_id
    )
    assert session is not None
    return dict(session.state)


def _seeded_workflow(
    executor: _SeedExecutor,
    resolver: ScriptedResolver,
    served: Sequence[Mapping[str, Any]] = (),
) -> Any:
    """A seeded turn's graph, whose Search serves `served`, the seeds or Spec blocks it
    answers each identity with."""
    return create_root_workflow(
        resolver=resolver,
        executor=executor,
        generative=GenerativeNodes(planner=ScriptedPlannerModel()),
        tool_surface=None,
        definition_surface=DataMcpDefinitions(served=served),
    )


async def test_a_seeded_turn_stores_each_seed_through_data_mcp_and_mints_nothing_before_planning(
    definition_store: InMemoryDefinitionStore,
) -> None:
    resolver = ScriptedResolver([_envelope(spec_id=SPEC_B)])
    executor = _SeedExecutor()
    seeds = [
        _seed("education", operations=[_RANK_DONORS], resolved={"headline": _HEADLINE}),
        _seed("kenya"),
    ]
    runner, session_id, events = await _start_seeded(
        _seeded_workflow(executor, resolver, seeds),
        "top 5 donors for education aid, and aid to Kenya",
        seeds,
    )

    visited = _visited(events)
    assert "seed" in visited
    assert "probe" not in visited
    assert resolver.calls == []
    assert executor.calls == []
    assert executor.steps == [
        {"tool": SEED_STEP_TOOL, "step_id": SEEDS_STEP_ID, "outcome": "ok", "code": None}
    ]
    for seed in seeds:
        held = await definition_store.get(_spec_id(seed), schema_version=1)
        assert held is not None
        assert json.loads(held.canonical_bytes) == seed["spec"]["definition"]["identity"]

    state = await _state(runner, session_id)
    assert state[SEEDS_STATE_KEY] is None
    assert state["spec_id"] == _spec_id(seeds[0])
    assert state["spec_id_elected"] is True
    assert state["spec_id_elected_label"] == "the reading education"
    assert state["parts"] == 2
    seeded = seeded_sources(state)
    assert [item.source.spec_id for item in seeded] == [_spec_id(seed) for seed in seeds]
    assert [item.seen_year for item in seeded] == [2024, None]
    envelope = json.loads(state["resolve_envelope"])
    assert envelope["status"] == "definite"
    assert envelope["seeds"] == [
        {
            "spec_id": _spec_id(seeds[0]),
            "sentence": "the reading education",
            "window": "every year with data",
            "showed": "2024, 13,356 rows",
            "coverage": "199 recipients",
            "wide": False,
            "stat_var_shape_kind": None,
            "stat_var_item_count": 1,
            "stat_var_member_count_exact": True,
            "entity_roles": [],
            "stat_var_truncated": False,
            "open_entity_axes": [],
            "choice_points": 0,
            "warnings": [],
            "operations": [_RANK_DONORS],
        },
        {
            "spec_id": _spec_id(seeds[1]),
            "sentence": "the reading kenya",
            "window": "every year with data",
            "showed": None,
            "coverage": "199 recipients",
            "wide": False,
            "stat_var_shape_kind": None,
            "stat_var_item_count": 1,
            "stat_var_member_count_exact": True,
            "entity_roles": [],
            "stat_var_truncated": False,
            "open_entity_axes": [],
            "choice_points": 0,
            "warnings": [],
            "operations": [],
        },
    ]
    assert "planner" in visited
    assert _pending_interrupt(events) == APPROVE_INTERRUPT


async def test_a_seeded_turn_with_an_operation_reaches_the_planner_not_fetch(
    definition_store: InMemoryDefinitionStore,
) -> None:
    """One definite seed with no choice points would take `fetch`; its rank
    is work `fetch` cannot do, so depth floors at shallow."""
    del definition_store
    seed = _seed("education", operations=[_RANK_DONORS])
    _, _, events = await _start_seeded(
        _seeded_workflow(_SeedExecutor(), ScriptedResolver([_envelope()]), [seed]),
        "top 5 donors for education aid",
        [seed],
    )

    visited = _visited(events)
    assert "fetch" not in visited
    assert _output_of(events, "choose_depth")["depth"] == "shallow"
    assert "planner" in visited


async def test_a_seed_that_does_not_hash_to_its_spec_id_fails_its_own_step_and_the_other_is_pinned(
    definition_store: InMemoryDefinitionStore,
) -> None:
    executor = _SeedExecutor()
    good = _seed("education", operations=[_RANK_DONORS])
    forged = _seed("kenya")
    forged["spec"]["definition"]["identity"]["seed"] = "forged"
    runner, session_id, events = await _start_seeded(
        _seeded_workflow(executor, ScriptedResolver([_envelope()]), [good, forged]),
        "top 5 donors for education aid, and aid to Kenya",
        [good, forged],
    )

    assert executor.steps == [
        {
            "tool": SEED_STEP_TOOL,
            "step_id": "seed-1",
            "outcome": "refusal",
            "code": "identity_mismatch",
        },
        {"tool": SEED_STEP_TOOL, "step_id": SEEDS_STEP_ID, "outcome": "ok", "code": None},
    ]
    assert await definition_store.get(_spec_id(forged), schema_version=1) is None
    assert "probe" not in _visited(events)
    state = await _state(runner, session_id)
    assert [item.source.spec_id for item in seeded_sources(state)] == [_spec_id(good)]
    assert state["parts"] == 1


async def test_a_seed_that_does_not_parse_fails_its_own_step_and_the_others_go_on(
    definition_store: InMemoryDefinitionStore,
) -> None:
    del definition_store
    executor = _SeedExecutor()
    good = _seed("education", operations=[_RANK_DONORS])
    bad = _seed("kenya")
    bad["spec"]["definition"]["spec_id"] = _spec_id(good)
    runner, session_id, events = await _start_seeded(
        _seeded_workflow(executor, ScriptedResolver([_envelope()]), [good]),
        "top 5 donors for education aid, and aid to Kenya",
        [bad, good],
    )

    assert executor.steps[0] == {
        "tool": SEED_STEP_TOOL,
        "step_id": "seed-0",
        "outcome": "refusal",
        "code": "seed_invalid",
    }
    assert "probe" not in _visited(events)
    state = await _state(runner, session_id)
    assert [item.source.spec_id for item in seeded_sources(state)] == [_spec_id(good)]


async def test_a_turn_with_no_usable_seed_falls_back_without_storing(
    definition_store: InMemoryDefinitionStore,
) -> None:
    resolver = ScriptedResolver([_envelope()])
    executor = _SeedExecutor()
    bad = _seed("education")
    del bad["spec"]["definition"]
    runner, session_id, events = await _start_seeded(
        _seeded_workflow(executor, resolver), "top 5 donors for education aid", [bad]
    )

    assert [(step["step_id"], step["outcome"]) for step in executor.steps] == [
        ("seed-0", "refusal"),
        ("seeds", "ok"),
    ]
    assert "probe" in _visited(events)
    assert len(resolver.calls) == 1
    assert seeded_sources(await _state(runner, session_id)) == []
    assert await definition_store.get(_spec_id(bad), schema_version=1) is None


async def test_an_unreachable_definition_store_fails_every_seed_and_the_receive_step() -> None:
    """Every write failing fails each seed's own step, then the receive step, and the
    turn falls back to searching the question."""
    set_definition_store(_UnreachableStore())
    try:
        resolver = ScriptedResolver([_envelope(spec_id=SPEC_B)])
        executor = _SeedExecutor()
        seeds = [_seed("education", operations=[_RANK_DONORS]), _seed("kenya")]
        runner, session_id, events = await _start_seeded(
            _seeded_workflow(executor, resolver, seeds),
            "top 5 donors for education aid, and aid to Kenya",
            seeds,
        )
    finally:
        set_definition_store(None)

    assert executor.steps == [
        {
            "tool": SEED_STEP_TOOL,
            "step_id": step_id,
            "outcome": "refusal",
            "code": "service_unavailable",
        }
        for step_id in ("seed-0", "seed-1", SEEDS_STEP_ID)
    ]
    assert executor.messages["seed-1"] == (
        "data-mcp could not resolve seed 1: the definition store is unreachable right now. "
        "Retry after 5.0s."
    )
    visited = _visited(events)
    assert visited.index("seed") < visited.index("probe")
    assert resolver.calls[0]["spec_id"] is None
    state = await _state(runner, session_id)
    assert state["spec_id"] == SPEC_B
    assert state["spec_id_elected"] is False
    assert seeded_sources(state) == []
    assert "seeds" not in json.loads(state["resolve_envelope"])


class _RefusesOneSpec(InMemoryDefinitionStore):
    """An in-memory store whose backend cannot take one spec_id's write."""

    def __init__(self, refused: str) -> None:
        super().__init__()
        self._refused = refused

    async def put_many(self, definitions: Any) -> None:
        if any(definition.spec_id == self._refused for definition in definitions):
            raise DefinitionStoreUnavailable("1 of 1 definition write(s) failed")
        await super().put_many(definitions)


async def test_a_seed_the_store_cannot_take_fails_its_own_step_and_the_other_is_planned() -> None:
    education = _seed("education", operations=[_RANK_DONORS])
    kenya = _seed("kenya")
    store = _RefusesOneSpec(_spec_id(education))
    set_definition_store(store)
    try:
        resolver = ScriptedResolver([_envelope()])
        executor = _SeedExecutor()
        runner, session_id, events = await _start_seeded(
            _seeded_workflow(executor, resolver, [education, kenya]),
            "top 5 donors for education aid, and aid to Kenya",
            [education, kenya],
        )
    finally:
        set_definition_store(None)

    assert executor.steps == [
        {
            "tool": SEED_STEP_TOOL,
            "step_id": "seed-0",
            "outcome": "refusal",
            "code": "service_unavailable",
        },
        {"tool": SEED_STEP_TOOL, "step_id": SEEDS_STEP_ID, "outcome": "ok", "code": None},
    ]
    assert await store.get(_spec_id(education), schema_version=1) is None
    assert await store.get(_spec_id(kenya), schema_version=1) is not None
    assert "probe" not in _visited(events)
    assert resolver.calls == []
    state = await _state(runner, session_id)
    assert [item.source.spec_id for item in seeded_sources(state)] == [_spec_id(kenya)]
    assert state["spec_id"] == _spec_id(kenya)
    assert state["spec_id_elected_label"] == "the reading kenya"
    assert state["parts"] == 1
    envelope = json.loads(state["resolve_envelope"])
    assert envelope["spec_id"] == _spec_id(kenya)
    assert [seed["spec_id"] for seed in envelope["seeds"]] == [_spec_id(kenya)]


async def test_a_no_year_seed_on_the_fetch_path_fetches_the_year_search_showed(
    definition_store: InMemoryDefinitionStore,
) -> None:
    del definition_store
    executor = _SeedExecutor()
    seed = _seed("education", resolved={"headline": _HEADLINE})
    _, _, events = await _start_seeded(
        _seeded_workflow(executor, ScriptedResolver([_envelope()]), [seed]),
        "education aid",
        [seed],
    )

    assert _pending_interrupt(events) is None
    assert executor.calls == [
        {
            "tool": "get_data",
            "args": {
                "spec_id": _spec_id(seed),
                "resolved_stat_var_dcids": ["Var_education"],
                "scope": {"window": {"kind": "range", "start_year": 2024, "end_year": 2024}},
            },
            "step_id": FETCH_STEP_ID,
        }
    ]


# A Spec block the dev Search API served: a CRS flow to Kenya from every
# donor, no year named, with the 2024 headline Search showed.
_KENYA_SEED = {
    "spec": json.loads(
        (Path(__file__).parent / "fixtures" / "seed_education_aid_kenya.json").read_text()
    )["spec"],
    "operations": [],
}


@pytest.mark.parametrize("digested", [True, False], ids=["digest-family", "recall-sample"])
async def test_a_seed_resubmit_posts_its_stat_vars_only_for_a_family_without_a_member_digest(
    definition_store: InMemoryDefinitionStore, digested: bool
) -> None:
    """Search elects a family with a member digest again from its code index and
    ignores a posted list for it; a recall sample is re-read from the list."""
    seed = copy.deepcopy(_KENYA_SEED)
    listed = [item["ref"] for item in seed["spec"]["resolved"]["stat_vars"]["items"]]
    assert seed["spec"]["resolved"]["shape"]["member_digest"] and listed
    if not digested:
        del seed["spec"]["resolved"]["shape"]["member_digest"]
    surface = DataMcpDefinitions(served=[seed["spec"]])

    await resolve_seed(surface, TakenSeed(index=0, seed=Seed.model_validate(seed)))

    (call,) = surface.calls
    assert call.get("stat_vars", []) == ([] if digested else listed)


async def test_a_served_flow_seed_with_no_operation_plans_on_studio(
    definition_store: InMemoryDefinitionStore,
) -> None:
    """Search serves only described identities, and one that binds a flow
    party is `wide`, so a Studio turn plans it."""
    del definition_store
    executor = _SeedExecutor()
    runner, session_id, events = await _start_seeded(
        _seeded_workflow(executor, ScriptedResolver([_envelope()]), [_KENYA_SEED]),
        "education aid to Kenya",
        [_KENYA_SEED],
    )

    visited = _visited(events)
    assert "fetch" not in visited
    assert "planner" in visited
    assert executor.calls == []
    envelope = json.loads((await _state(runner, session_id))["resolve_envelope"])
    assert envelope["wide"] is True
    (seed,) = envelope["seeds"]
    assert (seed["window"], seed["showed"], seed["coverage"]) == (
        "every year with data",
        "2024, 242 rows",
        "1 sources, 282 observations, 113 variables",
    )
    # Every donor is open, the population an operation over `origin` runs across.
    assert seed["open_entity_axes"] == [{"column": "origin", "members": 55}]


async def test_a_served_flow_seed_on_the_download_surface_fetches_the_year_search_showed(
    definition_store: InMemoryDefinitionStore,
) -> None:
    """The download surface skips the wide floor, so a served flow seed takes
    `fetch`, over the year whose figures Search showed."""
    del definition_store
    executor = _SeedExecutor()
    workflow = create_root_workflow(
        resolver=ScriptedResolver([_envelope()]),
        executor=executor,
        generative=GenerativeNodes(planner=ScriptedPlannerModel()),
        tool_surface=None,
        definition_surface=DataMcpDefinitions(served=[_KENYA_SEED]),
        surface="download",
    )
    _, _, events = await _start_seeded(workflow, "education aid to Kenya", [_KENYA_SEED])

    assert _pending_interrupt(events) is None
    (call,) = executor.calls
    # The flow family states a member digest, which the mint elects again,
    # so the call posts none of its StatVars.
    assert _KENYA_SEED["spec"]["resolved"]["shape"]["member_digest"]
    assert call["args"] == {
        "spec_id": _KENYA_SEED["spec"]["spec_id"],
        "scope": {"window": {"kind": "range", "start_year": 2024, "end_year": 2024}},
    }


async def test_a_seeded_envelope_states_the_turn_at_its_top_and_each_identity_on_its_seed(
    definition_store: InMemoryDefinitionStore,
) -> None:
    """The first seed is one narrow series and the second a flow from every
    donor. The top level says a seed is wide, as `choose_depth` reads it, and
    carries no member that describes one identity."""
    del definition_store
    seeds = [_seed("population"), _KENYA_SEED]
    runner, session_id, _ = await _start_seeded(
        _seeded_workflow(_SeedExecutor(), ScriptedResolver([_envelope()]), seeds),
        "population, and education aid to Kenya",
        seeds,
    )

    envelope = json.loads((await _state(runner, session_id))["resolve_envelope"])

    assert set(envelope) == {
        "status",
        "spec_id",
        "choice_points",
        "parts",
        "wide",
        "warnings",
        "candidates",
        "seeds",
    }
    assert envelope["wide"] is True
    narrow, flow = envelope["seeds"]
    assert envelope["spec_id"] == narrow["spec_id"]
    assert (narrow["wide"], narrow["entity_roles"], narrow["open_entity_axes"]) == (False, [], [])
    assert (flow["wide"], flow["entity_roles"], flow["stat_var_shape_kind"]) == (
        True,
        ["destination"],
        "statvar_schema",
    )
    assert flow["open_entity_axes"] == [{"column": "origin", "members": 55}]


def test_a_seeded_envelope_adds_up_what_choose_depth_reads_of_every_seed() -> None:
    warning = {"code": "stale", "severity": "degraded", "message": "m"}
    seeds = [
        {"spec_id": SPEC_A, "choice_points": 0, "wide": False, "warnings": []},
        {"spec_id": SPEC_B, "choice_points": 2, "wide": True, "warnings": [warning]},
    ]

    envelope = seeded_envelope(seeds)

    assert (envelope["spec_id"], envelope["parts"]) == (SPEC_A, 2)
    assert (envelope["choice_points"], envelope["wide"]) == (2, True)
    assert envelope["warnings"] == [warning]


@pytest.mark.parametrize("contradicts", [False, True], ids=["a-copy", "another-block"])
async def test_a_seed_naming_an_earlier_seed_s_spec_id_fails_its_own_step_and_the_first_stands(
    definition_store: InMemoryDefinitionStore, contradicts: bool
) -> None:
    """One spec_id is one seed: a repeat is refused whether it copies the first
    or carries another `resolved` block and other operations under the same id."""
    del definition_store
    executor = _SeedExecutor()
    first = _seed("education")
    repeat = copy.deepcopy(first)
    if contradicts:
        repeat["spec"]["resolved"]["rendered_sentence"] = "another reading"
        repeat["spec"]["resolved"]["stat_vars"] = {"items": [{"ref": "Var_OTHER"}]}
        repeat["operations"] = [_RANK_DONORS]

    runner, session_id, _ = await _start_seeded(
        _seeded_workflow(executor, ScriptedResolver([_envelope()]), [first]),
        "education aid",
        [first, repeat],
    )

    assert [(step["step_id"], step["code"]) for step in executor.steps] == [
        ("seed-1", SEED_DUPLICATE),
        (SEEDS_STEP_ID, None),
    ]
    assert executor.messages["seed-1"] == (
        f"seed 1 names the dataset seed 0 names ({_spec_id(first)}); a dataset is handed "
        "over once, and the first stands"
    )
    state = await _state(runner, session_id)
    (seeded,) = seeded_sources(state)
    assert seeded.source.stat_vars == ("Var_education",)
    assert state["parts"] == 1
    (planned,) = json.loads(state["resolve_envelope"])["seeds"]
    assert (planned["sentence"], planned["operations"]) == ("the reading education", [])


async def test_a_single_reference_seeds_nothing_and_says_so() -> None:
    """`seed_spec` as one `{spec_id, content_version}` carries no block, so the
    turn runs from the question and a failed step reports the unused reference."""
    executor = _SeedExecutor()
    resolver = ScriptedResolver([_envelope()])
    workflow = _seeded_workflow(executor, resolver)
    runner = InMemoryRunner(node=workflow)
    reference = {"spec_id": SPEC_A, "content_version": "cv_" + "0" * 32}
    session = await runner.session_service.create_session(
        app_name=runner.app_name, user_id="u", state={SEEDS_STATE_KEY: json.dumps(reference)}
    )

    events = [
        event
        async for event in runner.run_async(
            user_id="u",
            session_id=session.id,
            new_message=types.Content(
                role="user", parts=[types.Part.from_text(text="population of kenya")]
            ),
        )
    ]

    assert [(step["step_id"], step["code"]) for step in executor.steps] == [
        (SEEDS_STEP_ID, SEED_REFERENCE_UNUSED)
    ]
    assert "probe" in _visited(events)
    assert len(resolver.calls) == 1
    assert seeded_sources(await _state(runner, session.id)) == []


def _download_workflow(executor: _SeedExecutor, served: Sequence[Mapping[str, Any]]) -> Any:
    return create_root_workflow(
        resolver=ScriptedResolver([_envelope()]),
        executor=executor,
        generative=GenerativeNodes(planner=ScriptedPlannerModel()),
        tool_surface=None,
        definition_surface=DataMcpDefinitions(served=served),
        surface="download",
    )


async def test_a_download_retrieves_its_first_seed_and_fails_a_step_for_each_other(
    definition_store: InMemoryDefinitionStore,
) -> None:
    """A download plans nothing and fetches one spec, so a second and third
    seed each fail their own step where the user can see it."""
    del definition_store
    executor = _SeedExecutor()
    seeds = [_seed("education"), _seed("kenya"), _seed("health", operations=[_RANK_DONORS])]

    runner, session_id, events = await _start_seeded(
        _download_workflow(executor, seeds), "education aid, aid to Kenya and health aid", seeds
    )

    assert _pending_interrupt(events) is None
    assert executor.steps == [
        {
            "tool": SEED_STEP_TOOL,
            "step_id": "seed-1",
            "outcome": "refusal",
            "code": ONE_DATASET_PER_DOWNLOAD,
        },
        {
            "tool": SEED_STEP_TOOL,
            "step_id": "seed-2",
            "outcome": "refusal",
            "code": ONE_DATASET_PER_DOWNLOAD,
        },
        {"tool": SEED_STEP_TOOL, "step_id": SEEDS_STEP_ID, "outcome": "ok", "code": None},
    ]
    assert executor.messages["seed-1"] == (
        "a download retrieves one dataset, so seed 1 is left out; download it on its own"
    )
    assert [call["args"]["spec_id"] for call in executor.calls] == [_spec_id(seeds[0])]
    state = await _state(runner, session_id)
    assert [item.source.spec_id for item in seeded_sources(state)] == [_spec_id(seeds[0])]
    assert state["parts"] == 1
    assert [seed["spec_id"] for seed in json.loads(state["resolve_envelope"])["seeds"]] == [
        _spec_id(seeds[0])
    ]


async def test_a_download_reports_the_operations_it_does_not_run_and_still_retrieves_the_seed(
    definition_store: InMemoryDefinitionStore,
) -> None:
    del definition_store
    executor = _SeedExecutor()
    seed = _seed("education", operations=[_RANK_DONORS])

    _, _, events = await _start_seeded(
        _download_workflow(executor, [seed]), "top 5 donors for education aid", [seed]
    )

    assert _pending_interrupt(events) is None
    assert executor.steps == [
        {
            "tool": SEED_STEP_TOOL,
            "step_id": "seed-0-operations",
            "outcome": "refusal",
            "code": NO_OPERATIONS_ON_DOWNLOAD,
        },
        {"tool": SEED_STEP_TOOL, "step_id": SEEDS_STEP_ID, "outcome": "ok", "code": None},
    ]
    assert executor.messages["seed-0-operations"] == (
        "a download retrieves the data and computes nothing, so seed 0 is retrieved "
        'without its operations: rank ("top 5 donors"); ask for them in Studio'
    )
    assert [call["args"]["spec_id"] for call in executor.calls] == [_spec_id(seed)]


async def test_a_download_of_a_seed_with_no_operation_reports_none(
    definition_store: InMemoryDefinitionStore,
) -> None:
    del definition_store
    executor = _SeedExecutor()

    seed = _seed("education")
    await _start_seeded(_download_workflow(executor, [seed]), "education aid", [seed])

    assert [step["step_id"] for step in executor.steps] == [SEEDS_STEP_ID]


async def test_a_download_retrieves_the_first_seed_it_can_use(
    definition_store: InMemoryDefinitionStore,
) -> None:
    del definition_store
    executor = _SeedExecutor()
    seeds = [{"spec": "not a block"}, _seed("kenya"), _seed("health")]

    await _start_seeded(
        _download_workflow(executor, seeds[1:]), "aid to Kenya and health aid", seeds
    )

    assert [(step["step_id"], step["code"]) for step in executor.steps] == [
        ("seed-0", "seed_invalid"),
        ("seed-2", ONE_DATASET_PER_DOWNLOAD),
        (SEEDS_STEP_ID, None),
    ]
    assert [call["args"]["spec_id"] for call in executor.calls] == [_spec_id(seeds[1])]


async def test_a_studio_turn_keeps_every_seed(definition_store: InMemoryDefinitionStore) -> None:
    """The control for the download rule: another surface plans from all of them."""
    del definition_store
    executor = _SeedExecutor()
    seeds = [_seed("education"), _seed("kenya")]

    runner, session_id, _ = await _start_seeded(
        _seeded_workflow(executor, ScriptedResolver([_envelope()]), seeds),
        "education aid and aid to Kenya",
        seeds,
    )

    assert [step["step_id"] for step in executor.steps] == [SEEDS_STEP_ID]
    state = await _state(runner, session_id)
    assert [item.source.spec_id for item in seeded_sources(state)] == [
        _spec_id(seed) for seed in seeds
    ]


async def test_a_seed_whose_question_named_its_year_fetches_without_asking(
    definition_store: InMemoryDefinitionStore,
) -> None:
    del definition_store
    executor = _SeedExecutor()
    seed = _seed("education", resolved={"headline": {**_HEADLINE, "year_named": True}})
    _, _, events = await _start_seeded(
        _seeded_workflow(executor, ScriptedResolver([_envelope()]), [seed]),
        "education aid in 2024",
        [seed],
    )

    assert _pending_interrupt(events) is None
    assert executor.calls == [
        {
            "tool": "get_data",
            "args": {"spec_id": _spec_id(seed), "resolved_stat_var_dcids": ["Var_education"]},
            "step_id": FETCH_STEP_ID,
        }
    ]


_MINT_REFUSED_SELECTION = json.loads(
    (
        Path(__file__).parents[2]
        / "data-mcp"
        / "tests"
        / "fixtures"
        / "mint_refused_selection.json"
    ).read_text()
)


def _recorded_stale_seed(*, operations: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """The seed of the recorded mint refusal: its identity, and the StatVars it posted,
    one of which has left the identity's family."""
    definition = _MINT_REFUSED_SELECTION["request"]["definition"]
    stale = _MINT_REFUSED_SELECTION["request"]["resolved_stat_var_dcids"]
    return {
        "spec": {
            "spec_id": definition["spec_id"],
            "definition": definition,
            "resolved": {
                "rendered_sentence": "the population of Kenya",
                "stat_vars": {"items": [{"ref": ref, "slot_values": []} for ref in stale]},
            },
        },
        "operations": operations or [],
    }


def _recorded_refusal(request: httpx2.Request) -> httpx2.Response:
    """dfo-apis's recorded 422 `member_not_resolvable`, answering a `spec_id` mint."""
    assert request.url.path == "/api/datasets"
    response = _MINT_REFUSED_SELECTION["response"]
    return httpx2.Response(response["status"], json=response["body"])


class _DropsTheRemedy:
    """A narrator whose closing text leaves out the failed step's remedy."""

    async def __call__(self, *, question: str, facts: dict[str, Any]) -> dict[str, Any]:
        del question, facts
        return {"text": "The data could not be fetched."}


class _DataMcpExecutor(_SeedExecutor):
    """`get_data` through data-mcp in process, its upstream mint answered by
    `respond`, and the tool's answer read as `harness.mcp_client.McpExecutor`
    reads a refusal."""

    def __init__(
        self, store: InMemoryDefinitionStore, respond: Callable[[httpx2.Request], httpx2.Response]
    ) -> None:
        super().__init__()
        self._store = store
        self._respond = respond
        self.refusals: list[dict[str, Any]] = []

    async def __call__(
        self, *, tool: str, args: dict[str, Any], on_progress: ProgressCallback | None = None
    ) -> dict[str, Any]:
        del on_progress
        self.calls.append({"tool": tool, "args": args, "step_id": current_step_id.get()})
        upstream = UpstreamClient(
            settings=DataMcpSettings(), transport=httpx2.MockTransport(self._respond)
        )
        client_holder.set_client(upstream)
        try:
            async with Client(data_mcp) as client:
                # data-mcp's lifespan installs a store of its own on entry.
                set_definition_store(self._store)
                result = await client.call_tool(tool, args, raise_on_error=False)
        finally:
            client_holder.set_client(None)
        assert result.is_error, result
        assert result.structured_content is not None
        self.refusals.append(result.structured_content)
        return _refusal_outcome(result.structured_content)


async def test_a_seed_is_stored_in_the_data_mcp_whose_get_data_retrieves_it(
    definition_store: InMemoryDefinitionStore,
) -> None:
    """data-mcp holds a definition store apart from the one the harness process
    holds, as when it runs as its own service. The seed is stored in data-mcp's,
    and the download's `get_data` there posts its definition to the mint."""
    service = InMemoryDefinitionStore()
    seed = _recorded_stale_seed()
    definition = seed["spec"]["definition"]
    posted: list[dict[str, Any]] = []

    def respond(request: httpx2.Request) -> httpx2.Response:
        posted.append(json.loads(request.content))
        return _recorded_refusal(request)

    surface = DataMcpDefinitions(service, served=[seed])
    workflow = create_root_workflow(
        resolver=ScriptedResolver([_envelope()]),
        executor=_DataMcpExecutor(service, respond),
        tool_surface=None,
        definition_surface=surface,
        surface="download",
    )
    await _start_seeded(workflow, "population of Kenya", [seed])

    assert await service.get(definition["spec_id"], schema_version=1) is not None
    assert await definition_store.get(definition["spec_id"], schema_version=1) is None
    (mint,) = posted
    assert mint["input"]["definition"]["identity"] == definition["identity"]
    assert [call["spec_id"] for call in surface.calls] == [definition["spec_id"]]


async def test_a_seed_whose_statvar_left_its_dataset_closes_the_fetch_with_the_remedy(
    definition_store: InMemoryDefinitionStore,
) -> None:
    """The recorded dfo-apis 422 for a posted StatVar outside the identity's
    family, met on the no-plan path. Nothing replans there, so the remedy is
    the turn's answer: it reaches the closing text by code even when the
    narrator drops it. The download surface takes `fetch` whatever the
    block's width."""
    recorded = _MINT_REFUSED_SELECTION
    definition = recorded["request"]["definition"]
    stale = recorded["request"]["resolved_stat_var_dcids"]
    seed = _recorded_stale_seed()
    posted: list[dict[str, Any]] = []

    def respond(request: httpx2.Request) -> httpx2.Response:
        posted.append(json.loads(request.content))
        return _recorded_refusal(request)

    executor = _DataMcpExecutor(definition_store, respond)
    workflow = create_root_workflow(
        resolver=ScriptedResolver([_envelope()]),
        executor=executor,
        generative=GenerativeNodes(narrate=_DropsTheRemedy()),
        tool_surface=None,
        definition_surface=DataMcpDefinitions(served=[seed]),
        surface="download",
    )
    _, _, events = await _start_seeded(workflow, "population of Kenya", [seed])

    assert spec_id_for(canonicalize(definition["identity"])) == definition["spec_id"]
    assert [call["args"] for call in executor.calls] == [
        {"spec_id": definition["spec_id"], "resolved_stat_var_dcids": stale}
    ]
    (mint,) = posted
    assert mint["input"]["resolved_stat_var_dcids"] == stale
    (refusal,) = executor.refusals
    assert refusal["code"] == "spec_not_exportable"
    assert refusal["fix"]["not_members"]
    remedy = refusal["fix"]["remedy"]
    assert isinstance(remedy, str) and remedy
    fetched = _output_of(events, "fetch")
    assert fetched["frame_refs"] == []
    assert fetched["failed_step"] == {
        "step_id": FETCH_STEP_ID,
        "tool": "get_data",
        "code": refusal["code"],
        "message": refusal["message"],
        "remedy": remedy,
    }
    narrated = _output_of(events, "narrate")
    assert narrated["text"] == f"The data could not be fetched. {remedy}"


class _SeedRetrievalSurface:
    """`get_data` over the real schema, answered per spec_id: a stale seed's through
    data-mcp in process, its mint answered by the recorded 422, any other's with a
    manifest. `calls` records each call's arguments."""

    def __init__(
        self, schemas: list[ToolSchema], store: InMemoryDefinitionStore, *, stale: set[str]
    ) -> None:
        self._schemas = [schema for schema in schemas if schema.name == "get_data"]
        self._store = store
        self._stale = stale
        # data-mcp's lifespan installs a store of its own on each client's entry, so
        # the wave's concurrent stale calls go through it one at a time.
        self._lock = asyncio.Lock()
        self.calls: list[dict[str, Any]] = []

    async def list_tools(self) -> list[ToolSchema]:
        return self._schemas

    async def call_tool(
        self,
        *,
        tool: ToolId,
        arguments: Mapping[str, Any],
        on_progress: ProgressCallback | None = None,
    ) -> ToolCallResult:
        del on_progress
        assert tool.name == "get_data"
        self.calls.append(dict(arguments))
        spec_id = arguments["spec_id"]
        if spec_id not in self._stale:
            return ToolCallResult(
                is_error=False, structured_content=_get_data_manifest(spec_id=spec_id), text="ok"
            )
        async with self._lock:
            upstream = UpstreamClient(
                settings=DataMcpSettings(), transport=httpx2.MockTransport(_recorded_refusal)
            )
            client_holder.set_client(upstream)
            try:
                async with Client(data_mcp) as client:
                    set_definition_store(self._store)
                    result = await client.call_tool(
                        "get_data", dict(arguments), raise_on_error=False
                    )
            finally:
                client_holder.set_client(None)
                set_definition_store(self._store)
        assert result.is_error, result
        return ToolCallResult(
            is_error=True, structured_content=result.structured_content, text="refused"
        )


def _seeded_plan(
    *,
    schemas: list[ToolSchema],
    seeds: list[dict[str, Any]],
    bound: set[str],
    version: int = 1,
) -> PlannerPlan:
    """A translated plan retrieving each of `seeds` under its own step, posting the
    StatVars its block elected. A step retrieving a spec_id in `bound` is bound as its
    requirement's evidence; a succeeding step is left unbound, since its evidence check
    is not what these tests are about."""
    decision = Decision(
        slot="window",
        description="Which years?",
        options=["2015-2024", "2018-2024"],
        chosen="2015-2024",
        resolved_by=ResolvedBy.PLANNER_DEFAULT,
        rationale="a full decade covers the series",
    )
    get_data_schema = next(schema for schema in schemas if schema.name == "get_data")
    requirements: list[DatasetRequirement] = []
    operations: list[RetrieveOperation] = []
    steps: list[WorkflowStep] = []
    bindings: list[EvidenceBinding] = []
    for index, seed in enumerate(seeds):
        spec_id = _spec_id(seed)
        requirement = DatasetRequirement(
            id=f"part_{index}",
            decision_ids=["window"],
            measure=seed["spec"]["resolved"]["rendered_sentence"],
            entities=[Entity(name="Kenya", role="recipient")],
            period=Period(start=2015, end=2024),
        )
        step_id = f"get_{index}"
        requirements.append(requirement)
        operations.append(
            RetrieveOperation(id=step_id, requirement_id=requirement.id, spec_id=spec_id)
        )
        steps.append(
            WorkflowStep(
                id=step_id,
                operation_id=step_id,
                tool=ToolId(server="data-mcp", name="get_data"),
                args={
                    "spec_id": spec_id,
                    "resolved_stat_var_dcids": [
                        item["ref"] for item in seed["spec"]["resolved"]["stat_vars"]["items"]
                    ],
                },
                input_schema_hash=get_data_schema.input_schema_hash,
                output_schema_hash=get_data_schema.output_schema_hash,
                implementation_version=get_data_schema.version,
            )
        )
        if spec_id in bound:
            bindings.append(
                EvidenceBinding(requirement_id=requirement.id, step_id=step_id, spec_id=spec_id)
            )
    draft = PlannerPlan(
        question="population and education aid in Kenya",
        decisions=[decision],
        evidence_requirements=requirements,
        version=version,
    )
    translation = Translation(
        operations=operations, terminal=[operation.id for operation in operations]
    )
    workflow = Workflow(
        steps=steps,
        sources=[
            SelectedSource(
                spec_id=_spec_id(seed),
                sentence=seed["spec"]["resolved"]["rendered_sentence"],
                has_data=True,
            )
            for seed in seeds
        ],
        evidence_bindings=bindings,
        terminal_steps=[step.id for step in steps],
    )
    return draft.with_translation(translation=translation, workflow=workflow)


def _seed_plan_turn(
    seam: _Seam,
    surface: _SeedRetrievalSurface,
    served: Sequence[Mapping[str, Any]],
    *,
    narrate: Any | None = None,
) -> Any:
    return create_root_workflow(
        resolver=ScriptedResolver([_envelope()]),
        executor=_SeedExecutor(),
        generative=GenerativeNodes(planner=seam, narrate=narrate),
        tool_surface=surface,
        definition_surface=DataMcpDefinitions(served=served),
    )


@pytest.mark.parametrize("stale_first", [False, True], ids=["stale-last", "stale-first"])
async def test_a_stale_seed_among_others_is_dropped_and_the_turn_replans_from_the_rest(
    definition_store: InMemoryDefinitionStore,
    real_schemas: list[ToolSchema],
    stale_first: bool,
) -> None:
    """One seed's retrieval meets the recorded 422: its StatVars left its dataset. The
    turn drops that seed and offers a plan built from the other, whose card names the
    dropped seed. The approved replan retrieves the other seed alone."""
    good = _seed("education", operations=[_RANK_DONORS])
    stale = _recorded_stale_seed()
    seeds = [stale, good] if stale_first else [good, stale]
    first = _seeded_plan(schemas=real_schemas, seeds=seeds, bound={_spec_id(stale)})
    second = _seeded_plan(schemas=real_schemas, seeds=[good], bound=set(), version=2)
    seam = _Seam(
        [
            {"plan": first.model_dump(mode="json"), "summary": {}},
            {"plan": second.model_dump(mode="json"), "summary": {}},
        ]
    )
    surface = _SeedRetrievalSurface(real_schemas, definition_store, stale={_spec_id(stale)})
    runner, session_id, events = await _start_seeded(
        _seed_plan_turn(seam, surface, seeds), "education aid and the population of Kenya", seeds
    )
    assert _pending_interrupt(events) == APPROVE_INTERRUPT

    events = await _resume(
        runner, session_id, APPROVE_INTERRUPT, {"approved": plan_revision(first)}
    )

    assert sorted(call["spec_id"] for call in surface.calls) == sorted(
        [_spec_id(good), _spec_id(stale)]
    )
    assert _pending_interrupt(events) == APPROVE_INTERRUPT
    refusal = seam.calls[-1]["refusal"]
    assert {key: refusal[key] for key in ("kind", "step_id", "spec_id", "also_stopped")} == {
        "kind": "seed_dropped",
        "step_id": "get_0" if stale_first else "get_1",
        "spec_id": _spec_id(stale),
        "also_stopped": [],
    }
    assert refusal["refusal"]["code"] == "spec_not_exportable"
    assert "Do not plan that dataset again in this turn" in refusal["refusal"]["message"]
    posted = _MINT_REFUSED_SELECTION["request"]["resolved_stat_var_dcids"]
    assert not [dcid for dcid in posted if dcid in refusal["refusal"]["message"]]
    assert [seed["spec_id"] for seed in seam.calls[-1]["envelope"]["seeds"]] == [_spec_id(good)]
    state = await _state(runner, session_id)
    assert [item.source.spec_id for item in seeded_sources(state)] == [_spec_id(good)]
    assert [item.source.spec_id for item in dropped_seeds(state)] == [_spec_id(stale)]
    assert state["parts"] == 1
    assert state["replans"] == 1
    assert state["spec_id"] == _spec_id(good)
    assert json.loads(state["resolve_envelope"])["spec_id"] == _spec_id(good)
    replanned = _output_of(events, "planner")
    assert replanned["revision"] == plan_revision(second)
    assert replanned["refusal_note"] == (
        'This plan leaves out "the population of Kenya": the variables Search listed for it '
        "are no longer part of its dataset. Search for it again to add it."
    )

    events = await _resume(
        runner, session_id, APPROVE_INTERRUPT, {"approved": plan_revision(second)}
    )

    assert _pending_interrupt(events) is None
    assert [call["spec_id"] for call in surface.calls[2:]] == [_spec_id(good)]
    executed = _output_of(events, "execute")
    assert "failed_step" not in executed
    assert [ref["spec_id"] for ref in executed["frame_refs"]] == [_spec_id(good)]


@pytest.mark.parametrize(
    "seeds",
    [
        [_recorded_stale_seed(operations=[_RANK_DONORS])],
        [_seed("education", operations=[_RANK_DONORS]), _recorded_stale_seed()],
    ],
    ids=["one-seed", "every-seed-stale"],
)
async def test_a_turn_with_no_seed_left_to_plan_from_stops_with_the_remedy(
    definition_store: InMemoryDefinitionStore,
    real_schemas: list[ToolSchema],
    seeds: list[dict[str, Any]],
) -> None:
    """With no other seed to plan from, a stale seed's refusal stops the turn without a
    replan, and its remedy to search again closes it."""
    plan = _seeded_plan(schemas=real_schemas, seeds=seeds, bound={_spec_id(seed) for seed in seeds})
    seam = _Seam([{"plan": plan.model_dump(mode="json"), "summary": {}}])
    surface = _SeedRetrievalSurface(
        real_schemas, definition_store, stale={_spec_id(seed) for seed in seeds}
    )
    runner, session_id, _ = await _start_seeded(
        _seed_plan_turn(seam, surface, seeds, narrate=_DropsTheRemedy()), "aid to Kenya", seeds
    )

    events = await _resume(runner, session_id, APPROVE_INTERRUPT, {"approved": plan_revision(plan)})

    assert _pending_interrupt(events) is None
    assert len(seam.calls) == 1
    failed = _output_of(events, "execute")["failed_step"]
    assert failed["code"] == "spec_not_exportable"
    assert failed["remedy"].startswith("A variable search listed for this dataset")
    assert _output_of(events, "narrate")["text"] == (
        f"The data could not be fetched. {failed['remedy']}"
    )
    state = await _state(runner, session_id)
    assert [item.source.spec_id for item in seeded_sources(state)] == [
        _spec_id(seed) for seed in seeds
    ]
    assert state["replans"] == 0


async def test_a_stale_seed_met_after_the_replan_budget_is_spent_stops_the_turn(
    definition_store: InMemoryDefinitionStore, real_schemas: list[ToolSchema]
) -> None:
    """`MAX_REPLANS` bounds the drops: a second stale seed, met by the replanned plan,
    closes the turn with its remedy rather than a third plan."""
    good = _seed("education", operations=[_RANK_DONORS])
    stale = _recorded_stale_seed()
    later = _seed("water")
    first = _seeded_plan(schemas=real_schemas, seeds=[good, stale], bound={_spec_id(stale)})
    second = _seeded_plan(
        schemas=real_schemas, seeds=[good, later], bound={_spec_id(later)}, version=2
    )
    seam = _Seam(
        [
            {"plan": first.model_dump(mode="json"), "summary": {}},
            {"plan": second.model_dump(mode="json"), "summary": {}},
        ]
    )
    surface = _SeedRetrievalSurface(
        real_schemas, definition_store, stale={_spec_id(stale), _spec_id(later)}
    )
    runner, session_id, _ = await _start_seeded(
        _seed_plan_turn(seam, surface, [good, stale, later], narrate=_DropsTheRemedy()),
        "education aid, the population of Kenya, and water aid",
        [good, stale, later],
    )
    await _resume(runner, session_id, APPROVE_INTERRUPT, {"approved": plan_revision(first)})

    events = await _resume(
        runner, session_id, APPROVE_INTERRUPT, {"approved": plan_revision(second)}
    )

    assert _pending_interrupt(events) is None
    assert len(seam.calls) == 2
    failed = _output_of(events, "execute")["failed_step"]
    assert (failed["step_id"], failed["code"]) == ("get_1", "spec_not_exportable")
    assert _output_of(events, "narrate")["text"] == (
        f"The data could not be fetched. {failed['remedy']}"
    )
    state = await _state(runner, session_id)
    assert [item.source.spec_id for item in seeded_sources(state)] == [
        _spec_id(good),
        _spec_id(later),
    ]


async def test_a_replan_that_drafts_no_plan_after_a_drop_closes_naming_every_seed(
    definition_store: InMemoryDefinitionStore, real_schemas: list[ToolSchema]
) -> None:
    """The planner drafts nothing from the seed left after a drop. The closing text names
    the dropped seed and its remedy, says the other seed was not planned, and gives the
    planner's reason. It claims no plan, and the turn draws no card."""
    good = _seed("education", operations=[_RANK_DONORS])
    stale = _recorded_stale_seed()
    first = _seeded_plan(schemas=real_schemas, seeds=[good, stale], bound={_spec_id(stale)})
    seam = _Seam(
        [
            {"plan": first.model_dump(mode="json"), "summary": {}},
            {"refused": {"text": "I could not draft a plan within the budget."}},
        ]
    )
    surface = _SeedRetrievalSurface(real_schemas, definition_store, stale={_spec_id(stale)})
    runner, session_id, _ = await _start_seeded(
        _seed_plan_turn(seam, surface, [good, stale]),
        "education aid and the population of Kenya",
        [good, stale],
    )

    events = await _resume(
        runner, session_id, APPROVE_INTERRUPT, {"approved": plan_revision(first)}
    )

    assert _pending_interrupt(events) is None
    good_name = json.loads((await _state(runner, session_id))["resolve_envelope"])["seeds"][0][
        "sentence"
    ]
    assert _output_of(events, "deadend")["text"] == (
        '"the population of Kenya" could not be retrieved: the variables Search listed for '
        "it are no longer part of its dataset. Search for it again. No plan was drafted for "
        f'"{good_name}", so nothing is shown for it. I could not draft a plan within the budget.'
    )
    assert "compose" not in _visited(events)


def test_the_unplanned_seeds_text_names_several_seeds() -> None:
    stale = [DroppedSeed(name=name, reason=None) for name in ("a", "b")]
    assert unplanned_seeds_text(stale, ["c", "d"], "") == (
        '"a" and "b" could not be retrieved: the variables Search listed for them are no '
        "longer part of their datasets. Search for each again. No plan was drafted for "
        '"c" and "d", so nothing is shown for them.'
    )


async def test_a_stale_step_on_a_seeds_copy_over_a_period_names_the_seed(
    real_schemas: list[ToolSchema],
) -> None:
    """A plan retrieves a seed over its own period as a copy of the seed
    (`SeededSource.windowed_from`); the copy's stale refusal names the seed."""
    seed = SeededSource(source=SelectedSource(spec_id=SPEC_A, sentence="seed", has_data=True))
    copy = SeededSource(
        source=SelectedSource(spec_id=SPEC_B, sentence="seed", has_data=True),
        windowed_from=SPEC_A,
    )
    workflow = _seam_plan(schemas=real_schemas).workflow
    failure = ToolFailure(
        code="spec_not_exportable",
        message="no longer members",
        retry_mode="do_not_retry",
        fix={"not_members": ["Count_Person_Female"]},
    )
    stop = Refused(
        revision="r",
        refusal=RefusalKind.STEP_REFUSED,
        detail="refused",
        step_id="get_data",
        payload=failure,
    )
    assert workflow is not None
    bound = workflow.model_copy(
        update={
            "evidence_bindings": (
                EvidenceBinding(requirement_id="spending", step_id="get_data", spec_id=SPEC_B),
            )
        }
    )

    assert stale_seeds(stop, workflow=bound, seeded=[seed, copy]) == {"get_data": SPEC_A}
    assert stale_seeds(stop, workflow=bound, seeded=[]) == {}
    unnamed = stop.model_copy(update={"payload": failure.model_copy(update={"fix": {}})})
    assert stale_seeds(unnamed, workflow=bound, seeded=[seed, copy]) == {}


class _StateOnly:
    """The part of a node's context `_drop_seeds_from_turn` reads: its state."""

    def __init__(self, state: dict[str, Any]) -> None:
        self.state = state


@pytest.mark.parametrize(
    ("dropped", "left", "wide", "choice_points"),
    [(SPEC_A, SPEC_B, True, 2), (SPEC_B, SPEC_A, False, 0)],
    ids=["the-first-seed", "the-wide-seed"],
)
def test_a_drop_builds_the_envelope_again_from_the_seeds_left(
    dropped: str, left: str, wide: bool, choice_points: int
) -> None:
    """The top level states the turn, so what the dropped seed added to it goes with it."""
    warning = {"code": "stale", "severity": "degraded", "message": "m"}
    seeds = [
        {
            "spec_id": SPEC_A,
            "sentence": "a",
            "choice_points": 0,
            "wide": False,
            "warnings": [],
            "operations": [],
        },
        {
            "spec_id": SPEC_B,
            "sentence": "b",
            "choice_points": 2,
            "wide": True,
            "warnings": [warning],
            "operations": [],
        },
    ]
    state: dict[str, Any] = {"resolve_envelope": json.dumps(seeded_envelope(seeds))}

    _drop_seeds_from_turn(cast(Any, _StateOnly(state)), {dropped}, surface="studio")

    envelope = json.loads(state["resolve_envelope"])
    assert json.loads(state["dropped_untold"]) == [
        {"name": "a" if dropped == SPEC_A else "b", "reason": None}
    ]
    assert [seed["spec_id"] for seed in envelope["seeds"]] == [left]
    assert (envelope["spec_id"], envelope["parts"]) == (left, 1)
    assert (envelope["wide"], envelope["choice_points"]) == (wide, choice_points)
    assert envelope["warnings"] == ([warning] if left == SPEC_B else [])
    assert (state["spec_id"], state["parts"], state["spec_id_wide"]) == (left, 1, wide)
    assert state["spec_id_elected_label"] == ("b" if left == SPEC_B else "a")
    # The seed left is one part: wide and degraded when it is seed b, a plain
    # single series when it is seed a. Neither is "more than one thing".
    assert state["depth"] == ("deep" if left == SPEC_B else "none")
    assert "2 part(s)" not in state["depth_reason"]
    assert (state["depth_notice"] or "").startswith("There is a data quality issue") is (
        left == SPEC_B
    )


class _FailsOnTheReplan:
    """A `PlannerModel` that drafts the first plan and raises on the next call, as the
    seam does when the pair never reaches `submit_plan`."""

    def __init__(self, plan: dict[str, Any]) -> None:
        self._plan = plan
        self.calls = 0

    async def __call__(self, **kwargs: Any) -> dict[str, Any]:
        del kwargs
        self.calls += 1
        if self.calls == 1:
            return self._plan
        raise RuntimeError("submit_plan was never called")


async def test_a_replan_that_raises_after_a_drop_closes_naming_the_seed_and_the_remedy(
    definition_store: InMemoryDefinitionStore,
    real_schemas: list[ToolSchema],
) -> None:
    """The drop is already decided when the replan fails, so the turn closes with what the
    user is owed: which dataset was left out, and to search for it again."""
    good = _seed("education", operations=[_RANK_DONORS])
    stale = _recorded_stale_seed()
    first = _seeded_plan(schemas=real_schemas, seeds=[good, stale], bound={_spec_id(stale)})
    seam = _FailsOnTheReplan({"plan": first.model_dump(mode="json"), "summary": {}})
    surface = _SeedRetrievalSurface(real_schemas, definition_store, stale={_spec_id(stale)})
    runner, session_id, _ = await _start_seeded(
        _seed_plan_turn(cast(Any, seam), surface, [good, stale]),
        "education aid and the population of Kenya",
        [good, stale],
    )

    events = await _resume(
        runner, session_id, APPROVE_INTERRUPT, {"approved": plan_revision(first)}
    )

    assert seam.calls == 2
    assert _pending_interrupt(events) is None
    assert _output_of(events, "deadend") == {
        "text": (
            '"the population of Kenya" could not be retrieved: the variables Search listed '
            "for it are no longer part of its dataset. Search for it again. No plan was "
            'drafted for "the reading education", so nothing is shown for it.'
        )
    }


async def test_a_planner_that_raises_with_no_drop_still_raises(
    definition_store: InMemoryDefinitionStore,
) -> None:
    """The control: only a replan after a drop is closed for the seam."""
    del definition_store

    class _Raises:
        async def __call__(self, **kwargs: Any) -> dict[str, Any]:
            raise RuntimeError("submit_plan was never called")

    seed = _seed("education", operations=[_RANK_DONORS])
    workflow = create_root_workflow(
        resolver=ScriptedResolver([_envelope()]),
        executor=_SeedExecutor(),
        generative=GenerativeNodes(planner=cast(Any, _Raises())),
        tool_surface=None,
        definition_surface=DataMcpDefinitions(served=[seed]),
    )

    with pytest.raises(Exception, match="submit_plan was never called"):
        await _start_seeded(workflow, "top 5 donors", [seed])


async def test_the_replanned_card_states_the_depth_of_the_seeds_left(
    definition_store: InMemoryDefinitionStore,
    real_schemas: list[ToolSchema],
) -> None:
    """Two seeds with no operation plan because they are two parts. With one dropped,
    the card for the one left does not say the question asks for more than one thing."""
    good = _seed("education")
    stale = _recorded_stale_seed()
    first = _seeded_plan(schemas=real_schemas, seeds=[good, stale], bound={_spec_id(stale)})
    second = _seeded_plan(schemas=real_schemas, seeds=[good], bound=set(), version=2)
    seam = _Seam(
        [
            {"plan": first.model_dump(mode="json"), "summary": {}},
            {"plan": second.model_dump(mode="json"), "summary": {}},
        ]
    )
    surface = _SeedRetrievalSurface(real_schemas, definition_store, stale={_spec_id(stale)})
    runner, session_id, _ = await _start_seeded(
        _seed_plan_turn(seam, surface, [good, stale]),
        "education aid and the population of Kenya",
        [good, stale],
    )
    before = await _state(runner, session_id)
    assert "2 part(s)" in before["depth_reason"]
    assert "more than one thing" in before["depth_notice"]

    events = await _resume(
        runner, session_id, APPROVE_INTERRUPT, {"approved": plan_revision(first)}
    )

    state = await _state(runner, session_id)
    replanned = _output_of(events, "planner")
    assert state["parts"] == 1
    assert "2 part(s)" not in state["depth_reason"]
    assert (replanned["depth"], replanned["depth_notice"]) == (
        state["depth"],
        state["depth_notice"],
    )
    assert "more than one thing" not in (replanned["depth_notice"] or "")


def test_a_stale_reason_for_another_dataset_omits_a_spec_id_search_did_not_name() -> None:
    part = {"status": "definite"}
    unnamed = PartNotResolvedError(PartFault.ANOTHER_DATASET, part=part)
    named = PartNotResolvedError(PartFault.ANOTHER_DATASET, part=part, answered="spec_x")
    assert stale_reason(unnamed) == "Search now resolves it to another dataset"
    assert stale_reason(named) == "Search now resolves it to another dataset (spec_x)"


def test_the_dropped_seeds_note_names_each_seed() -> None:
    stale = [DroppedSeed(name=name, reason=None) for name in ("a", "b", "c")]
    assert dropped_seeds_note(stale) == (
        'This plan leaves out "a", "b", and "c": the variables Search listed for them are no '
        "longer part of their datasets. Search for each again to add it."
    )
    assert dropped_seeds_note(stale[:2]).startswith('This plan leaves out "a" and "b":')


def test_the_dropped_seeds_note_gives_a_seed_dropped_as_it_was_taken_its_own_reason() -> None:
    drops = [
        DroppedSeed(name="a", reason=None),
        DroppedSeed(name="dataset 2 from Search", reason="Search finds no data for it (x)"),
    ]
    assert dropped_seeds_note(drops) == (
        'This plan leaves out "a": the variables Search listed for it are no longer part of '
        'its dataset; "dataset 2 from Search": Search finds no data for it (x). Search for '
        "each again to add it."
    )


async def test_a_seed_whose_statvar_carries_the_dcid_prefix_fails_its_own_step(
    definition_store: InMemoryDefinitionStore,
) -> None:
    """An answer from Search with a selection `get_data` would refuse drops its seed,
    with the reason, and the other seed plans."""
    executor = _SeedExecutor()
    good = _seed("education", operations=[_RANK_DONORS])
    prefixed = _seed("kenya")
    prefixed["spec"]["resolved"]["stat_vars"]["items"][0]["ref"] = "dcid:Var_kenya"
    runner, session_id, events = await _start_seeded(
        _seeded_workflow(executor, ScriptedResolver([_envelope()]), [prefixed, good]),
        "top 5 donors for education aid, and aid to Kenya",
        [prefixed, good],
    )

    assert executor.steps[0] == {
        "tool": SEED_STEP_TOOL,
        "step_id": "seed-0",
        "outcome": "refusal",
        "code": SEED_DROPPED,
    }
    assert executor.messages["seed-0"] == (
        "seed 0 is left out: Search's answer for it cannot be read (ValueError: the spec "
        "block is not usable as a source: resolved.stat_vars.items[0].ref 'dcid:Var_kenya' "
        "is not a bare dcid: it carries the dcid: prefix). Search for it again to add it."
    )
    assert "probe" not in _visited(events)
    state = await _state(runner, session_id)
    assert [item.source.spec_id for item in seeded_sources(state)] == [_spec_id(good)]


@pytest.mark.parametrize(
    ("about", "reached"),
    [("canvas", "canvas_answer"), ("data", "deadend")],
    ids=["canvas", "deadend"],
)
async def test_a_seeded_turn_converse_routes_away_from_seed_clears_its_seeds(
    definition_store: InMemoryDefinitionStore,
    about: str,
    reached: str,
) -> None:
    """`seed` clears the raw seeds on its own route; every other route clears them in `converse`."""
    converse = ScriptedConverse([{"text": "on it", "search_parts": [], "about": about}])
    workflow = create_root_workflow(
        resolver=ScriptedResolver([_envelope()]),
        executor=_SeedExecutor(),
        generative=GenerativeNodes(converse=converse),
        tool_surface=None,
        definition_surface=DataMcpDefinitions(served=[_seed("education")]),
    )
    runner, session_id, events = await _start_seeded(
        workflow, "what is on my canvas", [_seed("education", operations=[_RANK_DONORS])]
    )

    visited = _visited(events)
    assert reached in visited
    assert "seed" not in visited
    assert (await _state(runner, session_id))[SEEDS_STATE_KEY] is None
    assert await definition_store.get(_spec_id(_seed("education")), schema_version=1) is None


async def test_a_turn_without_seeds_never_visits_the_seed_node() -> None:
    executor = _SeedExecutor()
    _, _, events = await _start(
        create_root_workflow(
            resolver=ScriptedResolver([_envelope()]), executor=executor, tool_surface=None
        ),
        "kenya oda 2023",
    )

    assert "seed" not in _visited(events)
    assert executor.steps == []


def test_operations_refuse_depth_none_and_quick_keeps_the_floor() -> None:
    plain = decide_depth(
        status="definite", choice_points=0, degraded=False, part_count=1, surface="studio"
    )
    seeded = decide_depth(
        status="definite",
        choice_points=0,
        degraded=False,
        part_count=1,
        surface="studio",
        operations=1,
    )
    quick = decide_depth(
        status="definite",
        choice_points=0,
        degraded=False,
        part_count=1,
        surface="studio",
        depth_hint="quick",
        operations=1,
    )

    assert plain.depth == "none"
    assert seeded.depth == "shallow"
    assert seeded.notice is not None
    assert quick.depth == "shallow"
