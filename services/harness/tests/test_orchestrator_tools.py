"""The orchestrator's tools against a fake resolver, executor and tool surface."""

from __future__ import annotations

import asyncio
import copy
import dataclasses
import json
import logging
import math
import re
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx2
import pyarrow as pa
import pytest
from analysis_mcp import verbs as analysis_verbs
from common.materialisation import (
    FrameRef,
    FrameTooLarge,
    InMemoryMaterialisationStore,
    MaterialisationTrace,
    set_store,
)
from common.place_type import (
    PARTY_ROLES,
    class_column,
    kind_column,
    type_column,
    whole_column,
)
from common.row_budget import RowBudget
from compiler.evidence import EvidenceIssueCode, check_evidence, evidence_from_source
from compiler.sources import source_from_spec_block
from compiler.surface import ToolCallResult, ToolSurfaceUnavailableError
from data_mcp.catalogue_counts import COUNTS_UNAVAILABLE
from data_mcp.frame_materialisation import KIND_COLUMNS, frame_ref_for
from data_mcp.search_tool import MAX_PARTS, MAX_RAW_QUERY_CHARS, SearchContext
from data_mcp.tool_errors import ToolErrorCode
from dfo_frame_grammar import Frame
from google.adk.sessions.state import State
from google.adk.tools import FunctionTool
from harness import chart_binding, orchestrator_tools, planner_adapter
from harness.chart_binding import (
    LINE_CHART_MAX_VALUES,
    MAX_SERIES_PER_COMPONENT,
    empty_select_refusal,
)
from harness.conversation import current_user_message
from harness.dfo_faq import CANNOT_DESCRIBE_DFO, DFO_FAQ_NOT_READ
from harness.graph.plan_done import HandBack
from harness.graph.state import (
    BUDGETS,
    ORCHESTRATOR_CALLS_KEY,
    ORCHESTRATOR_FRAMES_KEY,
    ORCHESTRATOR_PENDING_KEY,
    ORCHESTRATOR_UNAVAILABLE_KEY,
    PendingAsk,
    PendingPlan,
    frame_id_for,
    pending_action,
    register_frames,
    store_search,
)
from harness.mcp_client import _presentation_refusal_outcome
from harness.orchestrator_tools import (
    HANDED_OFF,
    Chart,
    OrchestratorTool,
    build_orchestrator_tools,
)
from harness.reply_figures import card_result, drawn_series
from harness.rolling_cards import CardWithheld, RollingCards
from harness.scope_choices import served_facts
from harness.studio_edit import _revised_frames
from harness.studio_state import (
    ConversationState,
    _apply_operations,
    _next_manifest,
    land_component,
    remove_cards,
)
from harness.tool_contracts import (
    HarnessCode,
    ResolverError,
    graph_envelope,
    harness_refusal,
    reading_facts,
    resolver_digest,
)
from harness.turn_records import SearchReading, basis_line, loose_match_notes, render_readings
from harness.wire import AddOp, CompositionActor, MutationOp, component_revised
from mcp import MCPError
from mcp.types import CONNECTION_CLOSED, REQUEST_TIMEOUT, ErrorData
from orchestrator.agent import INSTRUCTION
from planner.parties import PartyRole
from planner.plan import (
    MAX_ASK_CHARS,
    MAX_PLAN_ASKS,
    SCOPE_UNREAD_BLOCK_KEY,
    AskBreakdown,
    DatasetRequirement,
    current_earlier_block,
    current_recorded_identity,
)
from planner.records import Period
from planner.workflow import ToolId
from presentation_mcp import server as presentation_server
from presentation_mcp.platform_contract import Refusal, RetryMode
from submission import service as submission_service

from . import drawn_cards
from . import test_rolling_cards as rolling_scene
from .resolved_revision import resolved_revision
from .runner_frames import hold_root
from .search_results import search_result_parts
from .test_turn_records import (
    _example_part,
    _manifest,
    _scoped_part,
    _served_identity,
    _top_donors,
)
from .unreachable_surface import unreachable_surface

SPEC_AID = "spec_" + "a" * 32
SPEC_LIFE = "spec_" + "b" * 32
REF = {"spec_id": SPEC_LIFE, "content_version": "cv_1"}
VIEW = {"dimensions": [{"name": "observationAbout", "values": ["country/NGA"], "plural": False}]}
# The one series every chart of `FakeExecutor` draws.
NIGERIA = {"country/NGA": {"2000": 47.2, "2010": 50.9, "2024": 54.5}}

# A spec block `compiler.sources.source_from_spec_block` reads as a family
# of two StatVars that sums over one open, summable axis.
FAMILY_BLOCK = {
    "spec_id": SPEC_AID,
    "resolved": {
        "rendered_sentence": "Health aid to Kenya",
        "stat_vars": {"member_count": 2, "served_count": 2},
        "slots": [
            {
                "key": {"property": {"ref": "crsPurposeCode"}},
                "binding": {"kind": "unbound"},
                "summable": True,
            }
        ],
    },
}
SINGLE_BLOCK = {
    "spec_id": SPEC_LIFE,
    "resolved": {
        "rendered_sentence": "Life expectancy in Nigeria",
        "stat_vars": {"member_count": 1, "served_count": 1},
    },
}


class FakeExecutor:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.landed: list[dict[str, Any]] = []
        # The component every write answers with: a line chart of `NIGERIA`.
        self.written = drawn_cards.write_reply(
            drawn_cards.chart("line_chart", NIGERIA), drawn_cards.stored(NIGERIA)
        )

    async def __call__(self, *, tool: str, args: dict[str, Any], on_progress: Any = None):
        self.calls.append((tool, args))
        if tool == "get_data":
            if args["spec_id"] == "spec_refused":
                return {
                    "outcome": "refusal",
                    "refusal": {"code": "over_ceiling", "message": "too big"},
                }
            if args["spec_id"] == "spec_unexplained":
                return {"outcome": "refusal", "refusal": {"code": "over_ceiling"}}
            if "breakdown" in args:
                return {
                    "outcome": "refusal",
                    "refusal": {
                        "code": ToolErrorCode.CAPABILITY_UNAVAILABLE.value,
                        "message": "The 'breakdown' option isn't live.",
                    },
                }
            return {"outcome": "ok", "frame_refs": [REF], "summary": "64 observations"}
        if tool == "describe_frame":
            return {"outcome": "ok", "view": VIEW}
        return (
            self.described()
            if tool == "describe_component"
            else {"outcome": "ok", "component": self.written}
        )

    def described(self) -> dict[str, Any]:
        """The written card read back, as presentation-mcp's
        `describe_component` gives its head revision."""
        return {
            "outcome": "ok",
            "frame_refs": [],
            "component_body": {
                "component_id": self.written["component_id"],
                "revision": self.written["revision"],
                "fact_summary": self.written["fact_summary"],
            },
            "resolved_payload": self.written["payload"],
        }

    def land_component(self, component: dict[str, Any]) -> None:
        self.landed.append(component)


def _resolver(digest: dict[str, Any] | Exception):
    async def resolve(**kwargs: Any) -> dict[str, Any]:
        resolve.calls.append(kwargs)
        if isinstance(digest, Exception):
            raise digest
        return digest

    resolve.calls = []
    return resolve


def _digest(*parts: dict[str, Any], **extra: Any) -> dict[str, Any]:
    return {"status": "mixed", "warnings": [], "part_digests": list(parts), **extra}


def _tools(
    resolver=None,
    executor=None,
    surface=None,
    *,
    surface_name: str = "studio",
    dfo_faq_surface=None,
    recall_part=None,
    canvas_series=None,
    canvas_cards=None,
    remove_cards=None,
):
    tools = build_orchestrator_tools(
        resolver=resolver or _resolver(_digest()),
        executor=executor or FakeExecutor(),
        surface=surface,
        dfo_faq_surface=dfo_faq_surface,
        recall_part=recall_part,
        canvas_series=canvas_series,
        canvas_cards=canvas_cards,
        remove_cards=remove_cards,
    )
    by_name = {fn.__name__: fn for fn in tools}
    return by_name, SimpleNamespace(
        state={"surface": surface_name},
        invocation_id="inv_1",
        actions=SimpleNamespace(skip_summarization=None),
    )


# --- declarations ---------------------------------------------------------


def _declarations() -> dict[str, str]:
    """Each tool's function declaration as the model is sent it."""
    tools, _ = _tools()
    return {
        name: json.dumps(FunctionTool(fn)._get_declaration().model_dump(exclude_none=True))
        for name, fn in tools.items()
    }


def _word_runs(text: str, length: int = 7) -> set[str]:
    words = re.findall(r"[a-z0-9_']+", text.lower().replace("\\n", " "))
    return {" ".join(words[i : i + length]) for i in range(len(words) - length + 1)}


def test_no_rule_is_stated_in_both_the_instruction_and_a_tool_declaration():
    instruction = _word_runs(INSTRUCTION)

    shared = {name: instruction & _word_runs(text) for name, text in _declarations().items()}

    assert not any(shared.values()), shared


def test_a_search_for_names_an_earlier_turn_listed_states_their_side():
    """Search refuses a part whose names carry no role (`role_not_resolved`). The search
    declaration has every name an earlier turn listed sent with its side, and the
    instruction has a refused part sent again with roles before any other retry."""
    assert "`role_not_resolved`" in _declarations()["search"]
    assert "role_not_resolved" in INSTRUCTION


def test_no_tool_declaration_names_the_implementation_behind_it():
    named = {
        name: found
        for name, text in _declarations().items()
        if (found := re.findall(r"get_data|\bmint|data-mcp|data_mcp|dfo-apis|harness\.", text))
    }

    assert not named


def test_the_search_and_ask_user_declarations_carry_prices_beside_counts():
    declarations = _declarations()

    assert "`other_prices`, each with its spec_id and label" in declarations["search"]
    ask_user = declarations["ask_user"]
    assert "An option for a count or prices that a search result names" in ask_user


def test_every_chart_field_is_described_and_select_json_shows_a_list_of_values():
    """A select is written in the dimension names its own frame's view lists,
    so the examples key on more than one name."""
    properties = Chart.model_json_schema()["properties"]
    examples = [
        json.loads(example)
        for example in re.findall(r'\{"[^{}]*\}', properties["select_json"]["description"])
    ]

    assert all(field.get("description") for field in properties.values())
    assert any(isinstance(value, list) for example in examples for value in example.values())
    assert len({name for example in examples for name in example}) > 1


def test_the_chart_descriptions_state_both_series_caps():
    properties = Chart.model_json_schema()["properties"]

    assert f"up to {LINE_CHART_MAX_VALUES} places" in properties["type"]["description"]
    select = properties["select_json"]["description"]
    assert f"line_chart draws at most {LINE_CHART_MAX_VALUES} series" in select
    assert f"a bar_chart or a table at most {MAX_SERIES_PER_COMPONENT}" in select
    assert f"{chart_binding.MAX_TABLE_ROWS:,}" in select


def test_the_figure_fields_name_what_a_card_returns_for_them():
    """The marks an unnamed chart of several series returns, and the key a
    series with no value at the asked period is listed under, are named
    where the model chooses the fields."""
    properties = Chart.model_json_schema()["properties"]
    bars = drawn_cards.chart(
        "bar_chart", {"country/KEN": {"2024": 2.0}, "country/NGA": {"2024": 1.0}}
    )
    line = drawn_cards.chart("line_chart", NIGERIA)["component"]
    compared = card_result(drawn_series(bars["component"], {}))
    unheld = card_result(drawn_series(line, {}), state_period="1990")

    marks = {figure["mark"] for figure in compared["reply_figures"]}
    assert marks == {"largest", "smallest"}
    assert marks <= _word_runs(properties["state_values_for"]["description"], 1)
    (listed,) = set(unheld) - {"holds", "reply_figures"}
    assert listed in _word_runs(properties["state_period"]["description"], 1)


def test_a_chart_title_names_its_subject_and_leaves_the_period_to_the_tool():
    properties = Chart.model_json_schema()["properties"]
    title = properties["title"]["description"]

    assert "with no period or year" in title
    assert "make_charts adds the period the card shows" in title
    assert "counted back from the latest data" in properties["latest_periods"]["description"]


async def test_make_charts_writes_latest_periods_as_a_relative_window():
    executor = FakeExecutor()
    tools, ctx = _tools(executor=executor)
    await tools["fetch"](SPEC_LIFE, ctx)
    chart = {
        "frame": frame_id_for(REF),
        "type": "line_chart",
        "title": "Over the last three years",
        "select_json": json.dumps({"observationAbout": "country/NGA"}),
    }

    await tools["make_charts"]([{**chart, "latest_periods": 3}, chart], ctx)

    moving, whole = [args["intent"] for tool, args in executor.calls if tool == "write_component"]
    assert moving["window"] == {
        "start": {"anchor": "latest", "offset": -2},
        "end": {"anchor": "latest", "offset": 0},
    }
    assert "window" not in whole


def test_validating_arguments_leaves_every_declaration_as_the_model_sees_it():
    tools, _ = _tools()
    wrapped = {name: fn for name, fn in tools.items() if hasattr(fn, "__wrapped__")}

    assert set(wrapped) == set(tools) - {"explain_dfo"}
    for fn in wrapped.values():
        declaration = FunctionTool(fn.__wrapped__)._get_declaration()
        assert FunctionTool(fn)._get_declaration() == declaration
        assert OrchestratorTool(fn)._get_declaration() == declaration


# --- argument validation ----------------------------------------------------

_NGA = json.dumps({"observationAbout": "country/NGA"})


async def _search_role_on_the_part(tools: dict[str, Any], ctx: Any) -> dict[str, Any]:
    part = {"variable": "aid", "role": "to", "entities": [{"text": "Kenya"}]}
    return await tools["search"](ctx, [part])


async def _search_string_parts(tools: dict[str, Any], ctx: Any) -> dict[str, Any]:
    return await tools["search"](ctx, ["France's total gross ODA"])


async def _search_too_many_entities(tools: dict[str, Any], ctx: Any) -> dict[str, Any]:
    part = {"variable": "population", "entities": [{"text": f"place {i}"} for i in range(17)]}
    return await tools["search"](ctx, [part])


async def _search_entity_text_too_long(tools: dict[str, Any], ctx: Any) -> dict[str, Any]:
    part = {"variable": "population", "entities": [{"text": "k" * 129}]}
    return await tools["search"](ctx, [part])


async def _fetch_unknown_scope_key(tools: dict[str, Any], ctx: Any) -> dict[str, Any]:
    return await tools["fetch"](SPEC_LIFE, ctx, scope={"places": ["Kenya"]})


async def _charts_one_malformed(tools: dict[str, Any], ctx: Any) -> dict[str, Any]:
    await tools["fetch"](SPEC_LIFE, ctx)
    good = {"frame": frame_id_for(REF), "type": "bar_chart", "title": "t", "select_json": _NGA}
    return await tools["make_charts"]([good, {**good, "type": "pie_chart"}], ctx)


async def _plan_non_text_ask(tools: dict[str, Any], ctx: Any) -> dict[str, Any]:
    return tools["plan"]("Kenya's aid", ["the total", 5], ctx)


async def _recall_unknown_part(tools: dict[str, Any], ctx: Any) -> dict[str, Any]:
    return await tools["recall"]("plans", ctx)


@pytest.mark.parametrize(
    ("call", "tool", "budget", "loc", "error_type"),
    [
        (_search_role_on_the_part, "search", "search", "parts.0.role", "extra_forbidden"),
        (_search_string_parts, "search", "search", "parts.0", "model_type"),
        (_search_too_many_entities, "search", "search", "parts.0.entities", "too_long"),
        (
            _search_entity_text_too_long,
            "search",
            "search",
            "parts.0.entities.0.text",
            "string_too_long",
        ),
        (_fetch_unknown_scope_key, "fetch", "fetch", "scope.places", "extra_forbidden"),
        (
            _charts_one_malformed,
            "make_charts",
            "make_charts_failed",
            "charts.1.type",
            "literal_error",
        ),
        (_plan_non_text_ask, "plan", "plan", "asks.1", "model_type"),
        (_recall_unknown_part, "recall", "recall", "part", "literal_error"),
    ],
)
async def test_a_malformed_argument_is_an_error_for_the_model_and_runs_nothing(
    call, tool: str, budget: str, loc: str, error_type: str, caplog: pytest.LogCaptureFixture
):
    resolver, executor = _resolver(_digest()), FakeExecutor()
    recalled: list[dict[str, Any]] = []

    async def recall_part(**target: Any) -> dict[str, Any]:
        recalled.append(target)
        return {}

    tools, ctx = _tools(resolver, executor, recall_part=recall_part)

    found = await call(tools, ctx)

    assert set(found) == {"error", "invalid", "code", "retry_mode"}
    assert _stated(found) == (HarnessCode.TOOL_ARGUMENTS_INVALID, "correct_and_retry")
    assert found["error"].startswith(f"{tool}: ")
    assert found["invalid"] == [{"loc": loc, "type": error_type}]
    # Only a role on the part carries the hint to move it onto an entity.
    assert ('{"text": "Kenya", "role": "to"}' in found["error"]) == (
        call is _search_role_on_the_part
    )
    assert "input_value" not in found["error"]
    assert not resolver.calls
    assert [name for name, _ in executor.calls if name != "describe_frame"] in ([], ["get_data"])
    assert not executor.landed
    assert not recalled
    assert pending_action(ctx.state) is None
    assert ctx.state[ORCHESTRATOR_CALLS_KEY]["invalid_arguments"] == 1
    assert budget not in ctx.state[ORCHESTRATOR_CALLS_KEY]
    assert any("input_value" in record.getMessage() for record in caplog.records)


async def _fetch_without_spec_id(tools: dict[str, Any], ctx: Any) -> dict[str, Any]:
    return await OrchestratorTool(tools["fetch"]).run_async(args={}, tool_context=ctx)


async def test_a_call_missing_a_required_argument_is_an_error_to_correct_and_runs_nothing():
    executor = FakeExecutor()
    tools, ctx = _tools(executor=executor)

    found = await _fetch_without_spec_id(tools, ctx)

    assert set(found) == {"error", "invalid", "code", "retry_mode"}
    assert _stated(found) == (HarnessCode.TOOL_ARGUMENTS_INVALID, "correct_and_retry")
    assert found["error"].startswith("fetch: spec_id: ")
    assert found["invalid"] == [{"loc": "spec_id", "type": "missing"}]
    assert not executor.calls
    assert ctx.state[ORCHESTRATOR_CALLS_KEY] == {"invalid_arguments": 1}


_PART = {"variable": "aid", "entities": [{"text": "Kenya"}]}
_DEBT_PART = {"variable": "external debt", "entities": [{"text": "Ghana"}]}


@pytest.mark.parametrize(
    ("message", "args"),
    [
        ("aid to Kenya", {"parts": [_PART]}),
        (None, {"parts": [_PART, _DEBT_PART]}),
        ("aid to Kenya", {"text": "  ", "parts": [_PART]}),
    ],
    ids=["user-message", "no-message", "blank-text-beside-one-part"],
)
async def test_a_parsed_search_sends_its_parts_and_no_raw_query(message, args):
    resolver = _resolver(_digest())
    tools, ctx = _tools(resolver)
    token = current_user_message.set(message)
    try:
        found = await OrchestratorTool(tools["search"]).run_async(args=args, tool_context=ctx)
    finally:
        current_user_message.reset(token)

    assert "invalid" not in found
    [call] = resolver.calls
    assert call["question"] is None
    assert call["parts"] == args["parts"]
    assert "raw_text" not in call
    assert "invalid_arguments" not in ctx.state.get(ORCHESTRATOR_CALLS_KEY, {})


@pytest.mark.parametrize(
    ("args", "invalid"),
    [
        ({}, {"loc": "parts", "type": "missing"}),
        ({"parts": []}, {"loc": "parts", "type": "missing"}),
        ({"text": "  "}, {"loc": "parts", "type": "missing"}),
        (
            {"text": "aid to Kenya and Ghana", "parts": [_PART, _DEBT_PART]},
            {"loc": "parts", "type": "too_long"},
        ),
        ({"text": "k" * (MAX_RAW_QUERY_CHARS + 1)}, {"loc": "text", "type": "string_too_long"}),
    ],
    ids=["nothing", "no-parts", "blank-text", "text-with-two-parts", "text-over-the-limit"],
)
async def test_a_search_with_nothing_to_search_or_text_beside_several_parts_is_refused(
    args, invalid
):
    resolver = _resolver(_digest())
    tools, ctx = _tools(resolver)

    found = await OrchestratorTool(tools["search"]).run_async(args=args, tool_context=ctx)

    assert _stated(found) == (HarnessCode.TOOL_ARGUMENTS_INVALID, "correct_and_retry")
    assert found["invalid"] == [invalid]
    assert not resolver.calls
    assert ctx.state[ORCHESTRATOR_CALLS_KEY] == {"invalid_arguments": 1}


def test_search_declares_parts_and_text_and_requires_neither():
    tools, _ = _tools()

    declaration = OrchestratorTool(tools["search"])._get_declaration()

    assert declaration is not None
    declaration = declaration.model_dump(exclude_none=True)
    schema = declaration.get("parameters") or declaration["parameters_json_schema"]
    assert set(schema["properties"]) == {"parts", "text"}
    assert not schema.get("required")


@pytest.mark.parametrize("call", [_search_role_on_the_part, _fetch_without_spec_id])
async def test_calls_with_invalid_arguments_stop_at_their_budget(call):
    tools, ctx = _tools()

    for _ in range(BUDGETS["invalid_arguments"]):
        rejected = await call(tools, ctx)
        assert "invalid" in rejected
    spent = await call(tools, ctx)

    assert "invalid" not in spent
    assert _stated(spent) == (HarnessCode.TOOL_BUDGET_SPENT, "do_not_retry")
    assert ctx.state[ORCHESTRATOR_CALLS_KEY] == {"invalid_arguments": BUDGETS["invalid_arguments"]}


async def test_adk_function_tool_returns_the_argument_error_for_a_part_level_role():
    resolver = _resolver(_digest())
    tools, ctx = _tools(resolver)
    part = {"variable": "aid", "role": "to", "entities": [{"text": "Kenya"}]}

    found = await FunctionTool(tools["search"]).run_async(args={"parts": [part]}, tool_context=ctx)

    assert found["invalid"] == [{"loc": "parts.0.role", "type": "extra_forbidden"}]
    assert '{"text": "Kenya", "role": "to"}' in found["error"]
    assert not resolver.calls
    assert ctx.state[ORCHESTRATOR_CALLS_KEY] == {"invalid_arguments": 1}


_RETRY_PART = {"variable": "climate finance", "entities": [{"text": "Kenya", "role": "to"}]}


@pytest.mark.parametrize(
    ("status", "reason", "offered"),
    [
        ("no_data", None, True),
        ("unsupported", None, True),
        ("no_data", "qualifier_not_bound", True),
        ("unsupported", "no_observations", True),
        ("definite", None, False),
        ("candidates", None, False),
        ("definite", "qualifier_not_bound", False),
        ("candidates", "variable_not_resolved", False),
        ("no_data", "role_not_resolved", False),
        ("no_data", "entity_not_resolved", False),
    ],
)
async def test_a_parsed_search_offers_a_text_search_on_each_part_it_did_not_answer(
    status: str, reason: str | None, offered: bool
):
    reading: dict[str, Any] = {"variable": "climate finance", "status": status}
    if reason is not None:
        reading["reason"] = reason
    tools, ctx = _tools(_resolver(_digest(reading, kind="resolution")))

    found = await tools["search"](ctx, [_RETRY_PART])

    assert ("retry_with_text" in found["parts"][0]) is offered
    if offered:
        assert found["parts"][0]["retry_with_text"] == orchestrator_tools.RETRY_WITH_TEXT
    # The offer is the model's to read; the stored reading does not hold it.
    assert "retry_with_text" not in ctx.state["orchestrator:searches"][0]


async def test_a_text_search_is_offered_once_per_part_and_never_on_its_own_result():
    async def resolver(**call: Any) -> dict[str, Any]:
        # Search reads each part sent as unbound, under the variable it was sent with.
        sent = call["parts"] or [{"variable": "climate finance"}]
        return _digest(
            *(
                {"variable": p["variable"], "status": "no_data", "reason": "qualifier_not_bound"}
                for p in sent
            )
        )

    tools, ctx = _tools(resolver)
    ghana = {**_RETRY_PART, "entities": [{"text": "Ghana", "role": "to"}]}
    reworded = {"variable": "Climate  Finance", "entities": [{"text": " kenya", "role": "to"}]}

    first = await tools["search"](ctx, [_RETRY_PART])
    retried = await tools["search"](ctx, [_RETRY_PART], text="climate finance to Kenya")
    again = await tools["search"](ctx, [reworded])
    both = await tools["search"](ctx, [_RETRY_PART, ghana])
    named = {**_RETRY_PART, "entities": [{"text": "Kenya", "role": "to", "name": "Kenya Colony"}]}
    told_apart = await tools["search"](ctx, [named])
    sourced = await tools["search"](ctx, [{**_RETRY_PART, "sources": [{"text": "OECD"}]}])

    assert "retry_with_text" in first["parts"][0]
    assert "retry_with_text" not in retried["parts"][0]
    assert "retry_with_text" not in again["parts"][0]
    # The same measure for another place is another part.
    assert ["retry_with_text" in part for part in both["parts"]] == [False, True]
    # So is the same text told apart by its `name`, and the part from a named source.
    assert "retry_with_text" in told_apart["parts"][0]
    assert "retry_with_text" in sourced["parts"][0]


async def test_a_part_with_its_entities_in_another_order_is_the_part_already_searched():
    unbound = {"variable": "aid", "status": "no_data", "reason": "qualifier_not_bound"}
    tools, ctx = _tools(_resolver(_digest(unbound)))
    france, kenya = {"text": "France", "role": "from"}, {"text": "Kenya", "role": "to"}
    await tools["search"](
        ctx, [{"variable": "aid", "entities": [france, kenya]}], text="aid from France to Kenya"
    )

    found = await tools["search"](ctx, [{"variable": "aid", "entities": [kenya, france]}])

    assert "retry_with_text" not in found["parts"][0]


_UNBOUND = {"variable": "climate finance", "status": "no_data", "reason": "qualifier_not_bound"}


@pytest.mark.parametrize("reason", ["entity_not_resolved", "role_not_resolved"])
async def test_a_part_refused_for_a_place_or_a_role_is_offered_a_text_search_second(reason: str):
    refused = {"variable": "climate finance", "status": "no_data", "reason": reason}
    tools, ctx = _tools(_resolver(_digest(refused, kind="resolution")))
    repaired = {**_RETRY_PART, "entities": [{"text": "Kenya", "role": "from"}]}

    first = await tools["search"](ctx, [_RETRY_PART])
    second = await tools["search"](ctx, [repaired])

    assert "retry_with_text" not in first["parts"][0]
    assert "retry_with_text" in second["parts"][0]


@pytest.mark.parametrize(
    ("earlier", "offered"),
    [
        (None, [True, True]),
        ({"parts": [_RETRY_PART], "text": "climate finance to Kenya"}, [False, False]),
        ({"text": "climate finance to Kenya"}, [False, False]),
    ],
    ids=["no-text-search-yet", "after-a-text-search-of-a-part", "after-a-text-search-alone"],
)
async def test_readings_not_tied_to_the_parts_sent_are_offered_a_text_search_until_one_runs(
    earlier: dict[str, Any] | None, offered: list[bool]
):
    """Two readings for one part sent tie no reading to a part, so none can
    be told apart from a part a `text` search already ran for."""
    tools, ctx = _tools(_resolver(_digest(_UNBOUND, _UNBOUND)))
    if earlier is not None:
        await tools["search"](ctx, **earlier)

    found = await tools["search"](ctx, [_RETRY_PART])

    assert ["retry_with_text" in part for part in found["parts"]] == offered


# --- search ---------------------------------------------------------------


async def test_search_sends_every_part_in_one_call_and_returns_every_reading():
    resolver = _resolver(
        _digest(
            {
                "variable": "life expectancy",
                "status": "definite",
                "spec_id": SPEC_LIFE,
                "spec_block": SINGLE_BLOCK,
            },
            {
                "variable": "GDP",
                "status": "candidates",
                "spec_id": None,
                "spec_block": None,
                "candidates": [{"spec_id": "spec_a", "label": "GDP (current US$)"}],
            },
            kind="resolution",
            skills=[
                {"id": f"skill-{index}", "description": f"Skill {index}"} for index in range(7)
            ],
        )
    )
    tools, ctx = _tools(resolver)

    found = await tools["search"](
        ctx,
        [
            {"variable": "life expectancy", "entities": [{"text": "Nigeria"}]},
            {"variable": "GDP", "entities": [{"text": "Africa", "members": True}]},
        ],
    )

    assert len(resolver.calls) == 1
    assert resolver.calls[0]["question"] is None
    assert resolver.calls[0]["parts"] == [
        {"variable": "life expectancy", "entities": [{"text": "Nigeria"}]},
        {"variable": "GDP", "entities": [{"text": "Africa", "members": True}]},
    ]
    assert found["kind"] == "resolution"
    assert [part["status"] for part in found["parts"]] == ["definite", "candidates"]
    assert not any("spec_block" in part for part in found["parts"])
    assert found["skills"] == [
        {"id": f"skill-{index}", "description": f"Skill {index}"} for index in range(5)
    ]
    assert found["more_skills"] == 2
    assert ctx.state["orchestrator:specs"] == {SPEC_LIFE: SINGLE_BLOCK}


async def test_search_with_text_alone_asks_search_to_read_the_text():
    resolver = _resolver(_digest())
    tools, ctx = _tools(resolver)

    await tools["search"](ctx, text="aid to Kenya from all official donors")

    assert resolver.calls == [
        {"question": "aid to Kenya from all official donors", "parts": [], "raw_text": True}
    ]


async def test_a_parsed_search_sends_the_users_message_and_the_choices_the_conversation_settled():
    """Every explicit or carried choice of an earlier search is carried, the
    newest per axis and concept; a declared default is not. A raw-text search
    is sent no context."""
    earlier = SearchReading.from_part(_scoped_part()).model_dump(mode="json")
    newer = copy.deepcopy(earlier)
    sector = newer["scope"]["choices"][0]
    sector["active"] = "sector:120+130"
    newer["scope"]["choices"] = [sector]
    newer["scope"]["entries"] = [
        {
            "axis": "sector",
            "option_id": "sector:120+130",
            "provenance": "explicit",
            "concept": "health",
            "origin": {"kind": "choice", "choice_id": sector["id"]},
        }
    ]
    resolver = _resolver(_digest())
    tools, ctx = _tools(resolver)
    ctx.state["orchestrator:prior_ledger"] = {"readings": [earlier], "warnings": {}, "spec_ids": []}
    ctx.state["orchestrator:searches"] = [newer]
    token = current_user_message.set("health aid to Africa\n(then: now Ghana)")
    try:
        await tools["search"](ctx, [{"variable": "health aid"}])
        await tools["search"](ctx, [{"variable": "health aid"}], text="health aid to Ghana")
    finally:
        current_user_message.reset(token)

    base = earlier["scope"]["id"]
    parsed, raw = resolver.calls
    assert parsed["context"] == SearchContext(
        user_message="health aid to Africa\n(then: now Ghana)",
        carried_entries=[
            {
                "axis": "region",
                "option_id": "region:members_only",
                "provenance": "carried",
                "origin": {"kind": "choice", "choice_id": f"{base}:region"},
            },
            {
                "axis": "sector",
                "option_id": "sector:120+130",
                "provenance": "carried",
                "concept": "health",
                "origin": {"kind": "choice", "choice_id": f"{base}:sector"},
            },
        ],
    )
    assert "context" not in raw


@pytest.mark.parametrize(
    ("aligned", "axes"),
    [
        ("measure_basis", ["sector", "region"]),
        (None, ["sector", "region", "measure_basis"]),
        ("provider", ["sector", "region", "measure_basis"]),
    ],
    ids=["compiled-by-the-planner", "switched-by-the-user", "provider-compiled-by-the-planner"],
)
async def test_a_basis_the_planner_compiled_is_never_carried_and_one_the_user_switched_is(
    aligned: str | None, axes: list[str]
):
    """A reading whose gross basis came from an explicit entry on the plan card:
    the planner's, compiled to put a share's inputs on one basis, carries no basis
    to a later "US total ODA" search, while its user-stated sector and region
    still carry; the same entry from the user's own switch carries, and so does
    one on a reading the planner compiled on another axis."""
    reading = SearchReading.from_part(_scoped_part()).model_dump(mode="json")
    basis = reading["scope"]["choices"][2]
    basis["provenance"] = "explicit"
    reading["scope"]["entries"].append(
        {
            "axis": "measure_basis",
            "option_id": "measure_basis:gross",
            "provenance": "explicit",
            "origin": {"kind": "choice", "choice_id": basis["id"]},
        }
    )
    if aligned is not None:
        reading["planner_axes"] = [aligned]
    resolver = _resolver(_digest())
    tools, ctx = _tools(resolver)
    ctx.state["orchestrator:searches"] = [
        SearchReading.model_validate(reading).model_dump(mode="json")
    ]

    await tools["search"](ctx, [{"variable": "US total ODA"}])

    (parsed,) = resolver.calls
    assert [entry.axis for entry in parsed["context"].carried_entries] == axes


@pytest.mark.parametrize("parts", [None, []])
async def test_text_without_parts_reads_the_text_without_attaching_asked_roles(parts):
    resolver = _resolver(_digest({"variable": "aid", "status": "no_data"}))
    tools, ctx = _tools(resolver)

    found = await tools["search"](ctx, parts, text="aid to Kenya")

    assert "error" not in found
    assert resolver.calls == [{"question": "aid to Kenya", "parts": [], "raw_text": True}]
    assert not ctx.state["orchestrator:searches"][0]["asked_roles"]


async def test_text_refuses_multiple_parts_without_searching_or_storing_readings():
    resolver = _resolver(_digest())
    tools, ctx = _tools(resolver)
    parts = [
        {"variable": "aid", "entities": [{"text": "Kenya", "role": "to"}], "time": "2020"},
        {"variable": "aid", "entities": [{"text": "France", "role": "from"}], "time": "2024"},
    ]

    for _ in range(BUDGETS["invalid_arguments"]):
        refused = await tools["search"](ctx, parts, text="aid to Kenya and from France")
        assert refused["invalid"] == [{"loc": "parts", "type": "too_long"}]
    spent = await tools["search"](ctx, parts, text="aid to Kenya and from France")

    assert "limit" in spent["error"]
    assert not resolver.calls
    assert ctx.state == {
        "surface": "studio",
        ORCHESTRATOR_CALLS_KEY: {"invalid_arguments": BUDGETS["invalid_arguments"]},
    }


async def test_a_raw_text_search_keeps_its_reading_and_relays_its_warnings():
    warning = {
        "code": "role_scope_defaulted",
        "severity": "warn",
        "message": "no provider was named; the answer was resolved across every provider with data.",
    }
    resolver = _resolver(
        _digest(
            {
                "variable": "ODA",
                "status": "definite",
                "spec_id": SPEC_LIFE,
                "spec_block": SINGLE_BLOCK,
                "warnings": [warning],
            },
            kind="resolution",
        )
    )
    tools, ctx = _tools(resolver)

    found = await tools["search"](ctx, text="How much ODA did Kenya receive in 2024?")

    assert resolver.calls[0]["raw_text"] is True
    assert found["parts"][0]["status"] == "definite"
    assert found["parts"][0]["warnings"] == [warning]
    assert ctx.state["orchestrator:specs"] == {SPEC_LIFE: SINGLE_BLOCK}


async def test_a_text_search_keeps_the_parties_its_part_asked_for():
    resolver = _resolver(
        _digest(
            {"variable": "climate finance", "status": "definite", "spec_id": SPEC_LIFE},
            kind="resolution",
        )
    )
    tools, ctx = _tools(resolver)
    part = {
        "variable": "climate finance",
        "entities": [{"text": "France", "role": "from"}, {"text": "Kenya", "role": "to"}],
    }

    await tools["search"](ctx, [part], text="France's climate finance to Kenya")

    assert resolver.calls == [
        {"question": "France's climate finance to Kenya", "parts": [], "raw_text": True}
    ]
    assert ctx.state["orchestrator:searches"][0]["asked_roles"] == ["donor", "recipient"]


async def test_search_reports_a_result_with_no_parts_by_its_status():
    tools, ctx = _tools(
        _resolver(_digest(status="unsupported", warnings=[{"message": "no measure"}]))
    )

    found = await tools["search"](ctx, [{"variable": "anything"}])

    assert found == {
        "kind": "resolution",
        "parts": [],
        "status": "unsupported",
        "warnings": ["no measure"],
    }


def test_search_tells_the_model_when_to_set_a_role():
    tools, _ = _tools()

    doc = " ".join(tools["search"].__doc__.split())

    assert "`role` on each entity in `entities`" in doc
    assert '{"text": "Kenya", "role": "to"}' in doc


async def test_search_sends_a_source_code_and_tells_the_model_where_it_goes():
    resolver = _resolver(_digest())
    tools, ctx = _tools(resolver)
    part = {"variable": "debt service", "source_code": "DT.TDS.DEGG.CD"}

    await tools["search"](ctx, [part])

    assert resolver.calls[0]["parts"] == [part]
    declared = json.loads(_declarations()["search"])["parameters_json_schema"]["$defs"]
    assert "source_code" in declared["SearchPart"]["properties"]
    doc = " ".join(tools["search"].__doc__.split())
    assert "goes in the part's `source_code`" in doc
    assert "or the code itself when the user gives none" in doc
    assert "`source_code_ambiguous`), retry with a `sources` entry" in doc
    assert "`source_codes`" in doc


@pytest.mark.parametrize("ref", [None, "provenance/WorldBankInternationalDebtStatistics"])
async def test_search_sends_each_source_a_part_names_and_offers_no_dataset_list(ref):
    resolver = _resolver(_digest())
    tools, ctx = _tools(resolver)
    named_source = {"text": "IDS", "exclude": True}
    if ref is not None:
        named_source["ref"] = ref
    part = {"variable": "debt service", "sources": [named_source]}

    await tools["search"](ctx, [part])

    assert resolver.calls[0]["parts"] == [part]
    declared = json.loads(_declarations()["search"])["parameters_json_schema"]["$defs"]
    assert set(declared["SearchPart"]["properties"]) >= {"sources"}
    assert not {"dataset", "source"} & set(declared["SearchPart"]["properties"])
    assert set(declared["SearchSource"]["properties"]) == {"text", "exclude", "ref"}


async def test_search_returns_a_refusal_as_an_error_and_stops_at_its_budget():
    refusal = {
        "code": "rate_limited",
        "retry_mode": "retry_after_delay",
        "message": "search is down",
    }
    refused = ResolverError("search refused", refusal=refusal)
    tools, ctx = _tools(_resolver(refused))

    first = await tools["search"](ctx, [{"variable": "x"}])
    # The service's code and retry mode come with its message.
    assert first == {
        "error": "search is down",
        "code": "rate_limited",
        "retry_mode": "retry_after_delay",
    }

    # A failure that states no code is given the harness's own.
    bare, bare_ctx = _tools(_resolver(ResolverError("search carried no envelope")))
    uncoded = await bare["search"](bare_ctx, [{"variable": "x"}])
    assert _stated(uncoded) == ("unspecified", "do_not_retry")

    for _ in range(BUDGETS["search"] - 1):
        await tools["search"](ctx, [{"variable": "x"}])
    spent = await tools["search"](ctx, [{"variable": "x"}])

    assert "limit" in spent["error"]
    assert _stated(spent) == (HarnessCode.TOOL_BUDGET_SPENT, "do_not_retry")


@pytest.mark.parametrize("surface_name", ["studio", "download"])
async def test_every_part_can_retry_with_text_after_a_corrected_place(surface_name):
    """Search cannot place any part, even with the place corrected. The first
    result offers no `text` search, each corrected part's result offers one,
    and the budget runs to one for every part."""
    parts = [
        {"variable": f"measure {index}", "entities": [{"text": "Keyna"}]}
        for index in range(MAX_PARTS)
    ]
    unplaced = [
        {"variable": part["variable"], "status": "no_data", "reason": "entity_not_resolved"}
        for part in parts
    ]
    resolver = _resolver(_digest(*unplaced))
    tools, ctx = _tools(resolver, surface_name=surface_name)

    results = [await tools["search"](ctx, parts)]
    corrected = [{**part, "entities": [{"text": "Kenya"}]} for part in parts]
    results.append(await tools["search"](ctx, corrected))
    results.extend(
        [
            await tools["search"](ctx, [part], text=f"{part['variable']} in Kenya")
            for part in corrected
        ]
    )

    assert all("error" not in result for result in results)
    # The first refusal leaves the place to correct; the second offers the user's words.
    assert not any("retry_with_text" in reading for reading in results[0]["parts"])
    assert all("retry_with_text" in reading for reading in results[1]["parts"])
    assert len(resolver.calls) == MAX_PARTS + 2
    assert all(call["raw_text"] for call in resolver.calls[2:])
    assert all(not call["parts"] for call in resolver.calls[2:])

    spent = await tools["search"](ctx, [parts[0]], text="another retry")
    assert "limit" in spent["error"]
    assert len(resolver.calls) == MAX_PARTS + 2


# A recorded definite part: education aid to Kenya, 2015 to 2023, every donor open.
ROLE_OPEN_SPEC = json.loads(
    (Path(__file__).parent / "fixtures" / "search_spec_role_open.json").read_text()
)["spec"]
ROLE_OPEN_ID = ROLE_OPEN_SPEC["spec_id"]
GUESSED = {"code": "time_text_not_parsed", "severity": "degraded", "message": "time guessed"}
CLOSEST = {"code": "closest_match", "text": "These readings are the closest measures."}


def _recorded_envelope() -> dict[str, Any]:
    """A definite, a no_data and a candidates part, the definite one from a recorded block."""
    spec = copy.deepcopy(ROLE_OPEN_SPEC)
    spec["resolved"]["headline"]["recent_years"] = [
        {"year": 2024, "observations": 239, "series": 239, "role_counts": {}},
        {"year": 2023, "observations": 1807, "series": 300, "role_counts": {}},
    ]
    return {
        "status": "mixed",
        "parts": [
            {"status": "definite", "variable_text": "education aid", "spec": spec},
            {
                "status": "no_data",
                "variable_text": "aid to Atlantis",
                "no_data": {"category": "unresolved", "reason": "role_not_resolved"},
            },
            {
                "status": "candidates",
                "variable_text": "health spending",
                "warnings": [GUESSED],
                "departures": [CLOSEST],
                "candidates": {"specs": [SINGLE_BLOCK]},
            },
        ],
    }


async def test_search_stores_a_reading_for_every_part_and_warnings_for_every_spec():
    envelope = _recorded_envelope()
    tools, ctx = _tools(_resolver(resolver_digest(envelope)))

    found = await tools["search"](ctx, [{"variable": "education aid"}])

    # The model sees the part summaries, which carry none of the stored facts.
    assert found["parts"] == graph_envelope(envelope)["part_digests"]
    definite, no_data, candidates = (
        SearchReading.model_validate(stored) for stored in ctx.state["orchestrator:searches"]
    )
    assert (definite.status, definite.spec_id) == ("definite", ROLE_OPEN_ID)
    assert (definite.coverage_kind, definite.requested_window_kind) == ("exact", "range")
    assert definite.roles == ("destination",)
    assert [(axis.column, axis.members) for axis in definite.open_axes] == [("origin", 55)]
    assert definite.recent_years == ()
    assert definite.observed_at is not None
    assert (no_data.status, no_data.spec_id, no_data.refusal_code) == (
        "no_data",
        None,
        "role_not_resolved",
    )
    assert candidates.status == "candidates"
    assert [c.spec_id for c in candidates.candidates] == [SPEC_LIFE]
    assert ctx.state["orchestrator:spec_warnings"] == {ROLE_OPEN_ID: [], SPEC_LIFE: [GUESSED]}
    assert set(ctx.state["orchestrator:specs"]) == {ROLE_OPEN_ID, SPEC_LIFE}


DROPPED_ROLE = {"code": "role_text_not_recognised", "severity": "degraded", "message": "role"}


@pytest.mark.parametrize(
    ("settled_with", "offered_with"),
    [([], [DROPPED_ROLE]), ([DROPPED_ROLE], [])],
    ids=["settled-clean", "settled-narrowed"],
)
def test_a_search_offering_a_settled_spec_as_a_candidate_keeps_its_block_and_warnings(
    settled_with: list[dict[str, str]], offered_with: list[dict[str, str]]
) -> None:
    """A search settles on education aid to Kenya, and a later one offers the same
    spec_id among its candidates under other warnings. The block and warnings of the
    search that settled on it stand for the spec_id."""
    state: dict[str, Any] = {}
    settled = {
        "status": "definite",
        "variable_text": "education aid",
        "warnings": settled_with,
        "spec": copy.deepcopy(ROLE_OPEN_SPEC),
    }
    store_search(state, resolver_digest({"status": "definite", "parts": [settled]}))
    block = copy.deepcopy(state["orchestrator:specs"][ROLE_OPEN_ID])
    offered = copy.deepcopy(ROLE_OPEN_SPEC)
    offered["resolved"]["rendered_sentence"] = "aid offered in its place"
    candidates = {
        "status": "candidates",
        "variable_text": "aid",
        "warnings": offered_with,
        "candidates": {"specs": [offered, SINGLE_BLOCK]},
    }

    store_search(state, resolver_digest({"status": "candidates", "parts": [candidates]}))

    assert state["orchestrator:specs"][ROLE_OPEN_ID] == block
    assert state["orchestrator:spec_warnings"][ROLE_OPEN_ID] == settled_with
    assert state["orchestrator:spec_warnings"][SPEC_LIFE] == offered_with


async def test_search_shows_no_year_counts_of_a_part_searched_for_the_latest_period():
    envelope = _recorded_envelope()
    spec = envelope["parts"][0]["spec"]
    spec["definition"]["identity"]["requested_window"] = {"kind": "latest"}
    tools, ctx = _tools(_resolver(resolver_digest(envelope)))

    found = await tools["search"](ctx, [{"variable": "education aid"}])

    # The block's headline counts its recent years, and the model is shown none of them.
    assert spec["resolved"]["headline"]["recent_years"]
    assert reading_facts(spec)["requested_window_kind"] == "latest"
    assert "recent_years" not in reading_facts(spec)
    assert all("recent_years" not in part for part in found["parts"])


def test_comparison_intent_reaches_the_stored_search_ledger() -> None:
    state: dict[str, Any] = {}
    envelope = {
        "parts": [
            {"status": "no_data", "variable_text": "health aid", "separate": True},
            {"status": "no_data", "variable_text": "education aid"},
        ]
    }

    store_search(state, graph_envelope(envelope))
    readings = [SearchReading.model_validate(row) for row in state["orchestrator:searches"]]

    assert [reading.separate for reading in readings] == [True, False]
    assert state["orchestrator:searches"][0]["separate"] is True
    assert "separate" not in state["orchestrator:searches"][1]


def test_a_qualifier_refusal_reaches_the_stored_ledger_without_offering_a_source() -> None:
    envelope = {
        "parts": [
            {
                "status": "no_data",
                "variable_text": "aid to Kenya for HIV",
                "no_data": {
                    "category": "absent",
                    "reason": "qualifier_not_bound",
                    "unbound_qualifiers": [{"text": "HIV"}],
                    "flagged_match": {
                        "spec_id": "spec_health",
                        "resolved": {
                            "rendered_sentence": "Gross ODA disbursements to Kenya for health."
                        },
                    },
                },
            }
        ]
    }
    state: dict[str, Any] = {}

    parts = store_search(state, graph_envelope(envelope))
    (stored,) = (SearchReading.model_validate(row) for row in state["orchestrator:searches"])

    assert parts[0]["unbound_qualifiers"] == ["HIV"]
    assert stored.unbound_qualifiers == ("HIV",)
    assert stored.flagged_match == "Gross ODA disbursements to Kenya for health."
    assert "spec_health" not in state.get("orchestrator:specs", {})
    assert "spec_health" not in stored.model_dump_json()
    rendered = render_readings([stored], budget=10_000)
    assert "refused phrases: HIV" in rendered.text
    assert "broader unused match: Gross ODA disbursements to Kenya for health." in rendered.text
    assert rendered.spec_ids == ()


async def test_search_keeps_the_readings_of_earlier_searches_in_the_turn():
    tools, ctx = _tools(_resolver(_digest({"variable": "x", "status": "no_data"})))

    await tools["search"](ctx, [{"variable": "x"}])
    await tools["search"](ctx, [{"variable": "x"}])

    assert len(ctx.state["orchestrator:searches"]) == 2


# --- fetch ----------------------------------------------------------------


async def test_fetch_adds_its_summary_or_refusal_to_the_reading_of_its_spec_id():
    tools, ctx = _tools()
    ctx.state["orchestrator:searches"] = [
        SearchReading(text=text, status="definite", spec_id=spec_id).model_dump(mode="json")
        for text, spec_id in (("life", SPEC_LIFE), ("refused", "spec_refused"), ("aid", SPEC_AID))
    ]

    await tools["fetch"](SPEC_LIFE, ctx)
    await tools["fetch"]("spec_refused", ctx)

    life, refused, aid = (
        SearchReading.model_validate(stored) for stored in ctx.state["orchestrator:searches"]
    )
    assert life.fetch is not None and life.fetch.summary == "64 observations"
    assert refused.fetch is not None and refused.fetch.refusal_code == "over_ceiling"
    assert aid.fetch is None


async def test_fetch_of_a_nearest_series_adds_a_settled_reading_for_it():
    """A `no_data` part's nearest real series is fetched as a candidate is:
    its block is stored, so the fetch adds a `chosen` reading the planner
    may pin with no second search. The stored block is marked for a scope
    read, since Search states no scope choices on a nearest series."""
    envelope = {
        "status": "no_data",
        "parts": [
            {
                "status": "no_data",
                "variable_text": "net flows",
                "no_data": {"reason": "no_observations", "nearest_real": [SINGLE_BLOCK]},
            }
        ],
    }
    tools, ctx = _tools(_resolver(resolver_digest(envelope)))
    await tools["search"](ctx, [{"variable": "net flows"}])

    await tools["fetch"](SPEC_LIFE, ctx)

    readings = [SearchReading.model_validate(r) for r in ctx.state["orchestrator:searches"]]
    assert [(r.status, r.spec_id, r.text) for r in readings] == [
        ("no_data", None, "net flows"),
        ("chosen", SPEC_LIFE, "net flows"),
    ]
    assert ctx.state["orchestrator:specs"][SPEC_LIFE] == {
        **SINGLE_BLOCK,
        SCOPE_UNREAD_BLOCK_KEY: True,
    }


async def test_fetch_of_a_chosen_candidate_adds_a_reading_for_it():
    envelope = _recorded_envelope()
    envelope["parts"][2]["separate"] = True
    tools, ctx = _tools(_resolver(resolver_digest(envelope)))
    await tools["search"](ctx, [{"variable": "education aid"}])

    await tools["fetch"](SPEC_LIFE, ctx)
    await tools["fetch"](ROLE_OPEN_ID, ctx)
    await tools["fetch"]("spec_other", ctx)

    readings = [SearchReading.model_validate(r) for r in ctx.state["orchestrator:searches"]]
    assert [(r.status, r.spec_id) for r in readings] == [
        ("definite", ROLE_OPEN_ID),
        ("no_data", None),
        ("candidates", None),
        ("chosen", SPEC_LIFE),
    ]
    chosen = readings[-1]
    assert chosen.text == "health spending"
    assert chosen.sentence == "Life expectancy in Nigeria"
    assert [w.code for w in chosen.warnings] == [GUESSED["code"]]
    assert [d.model_dump() for d in chosen.departures] == [CLOSEST]
    assert chosen.separate is True
    assert ctx.state["orchestrator:searches"][-1]["separate"] is True
    assert chosen.fetch is not None and chosen.fetch.summary == "64 observations"
    assert readings[0].fetch is not None


# A recorded search for "climate finance": three SDG worldwide series, offered
# as candidates under `retrieval_score_weak`.
CLIMATE_ENVELOPE = json.loads(
    (Path(__file__).parent / "fixtures" / "search_climate_finance_weak.json").read_text()
)
CLIMATE_TOTAL = "spec_8380f22015251b410d800a2774893073"


@pytest.mark.parametrize(
    ("part", "lacks"),
    [
        pytest.param(
            {"variable": "climate finance"},
            "a worldwide total with no breakdown by country",
            id="no-party-asked",
        ),
        pytest.param(
            {
                "variable": "climate finance",
                "entities": [{"text": "France", "role": "from"}, {"text": "Kenya", "role": "to"}],
            },
            "a worldwide total with no breakdown by country and no donor or recipient detail",
            id="a-donor-and-a-recipient-asked",
        ),
    ],
)
async def test_a_fetched_weak_candidate_is_noted_with_what_it_covers_and_lacks(
    part: dict[str, Any], lacks: str
):
    tools, ctx = _tools(_resolver(resolver_digest(CLIMATE_ENVELOPE)))
    await tools["search"](ctx, [part])
    specs = ctx.state["orchestrator:specs"]
    searched = loose_match_notes(ctx.state["orchestrator:searches"], specs=specs, plans=[])

    await tools["fetch"](CLIMATE_TOTAL, ctx)

    # Nothing is noted until the turn uses the series.
    assert searched == []
    assert loose_match_notes(ctx.state["orchestrator:searches"], specs=specs, plans=[]) == [
        'The closest series to "climate finance" is a loose match: Total climate-specific '
        f"financial support provided worldwide, 2011 to 2022 (US dollars, SDG Global "
        f"Database), {lacks}."
    ]


@pytest.mark.parametrize(
    ("stored", "earlier"),
    [
        ({"orchestrator:specs": {SPEC_AID: FAMILY_BLOCK, SPEC_LIFE: SINGLE_BLOCK}}, {}),
        ({}, {SPEC_AID: FAMILY_BLOCK, SPEC_LIFE: SINGLE_BLOCK}),
        (
            {"orchestrator:specs": {SPEC_AID: FAMILY_BLOCK}},
            {SPEC_AID: SINGLE_BLOCK, SPEC_LIFE: SINGLE_BLOCK},
        ),
    ],
    ids=["this_turns_blocks", "an_earlier_turns_blocks", "this_turns_block_wins"],
)
async def test_fetch_reads_a_family_as_its_total_and_a_single_indicator_as_is(
    stored: dict[str, Any], earlier: dict[str, Any]
):
    """An earlier turn's block is read through the conversation
    (`current_earlier_block`), not from the turn's session state."""
    executor = FakeExecutor()
    tools, ctx = _tools(executor=executor)
    ctx.state.update(stored)
    current_earlier_block.set(earlier.get)

    aid = await tools["fetch"](SPEC_AID, ctx)
    life = await tools["fetch"](SPEC_LIFE, ctx)
    unknown = await tools["fetch"](
        "spec_other", ctx, scope={"window": {"kind": "range", "start_year": 2015, "end_year": 2020}}
    )

    assert [args for tool, args in executor.calls if tool == "get_data"] == [
        {"spec_id": SPEC_AID, "stat_vars": "sum"},
        {"spec_id": SPEC_LIFE},
        {
            "spec_id": "spec_other",
            "scope": {"window": {"kind": "range", "start_year": 2015, "end_year": 2020}},
        },
    ]
    assert aid["frames"] == [{"frame": frame_id_for(REF), "view": VIEW}]
    assert life["summary"] == "64 observations"
    assert unknown["frames"][0]["frame"] == frame_id_for(REF)
    # The frame keeps the arguments of the fetch that last registered it.
    assert ctx.state[ORCHESTRATOR_FRAMES_KEY] == {
        frame_id_for(REF): {
            "ref": REF,
            "view": VIEW,
            "fetch_args": {
                "spec_id": "spec_other",
                "scope": {"window": {"kind": "range", "start_year": 2015, "end_year": 2020}},
            },
        }
    }


async def test_fetch_keeps_the_reading_a_plan_step_used_for_its_spec():
    executor = FakeExecutor()
    tools, ctx = _tools(executor=executor)
    ctx.state["orchestrator:readings"] = {SPEC_LIFE: "sum"}

    await tools["fetch"](SPEC_LIFE, ctx)

    assert [args for tool, args in executor.calls if tool == "get_data"] == [
        {"spec_id": SPEC_LIFE, "stat_vars": "sum"}
    ]


@pytest.mark.parametrize(
    ("spec_id", "refusal"),
    [
        ("spec_refused", "too big"),
        # A refusal data-mcp sent without a message still reads as one.
        ("spec_unexplained", "get_data refused this fetch and gave no reason."),
    ],
)
async def test_fetch_returns_the_refusal_when_the_data_cannot_be_retrieved(spec_id, refusal):
    tools, ctx = _tools()

    found = await tools["fetch"](spec_id, ctx)

    assert found == {"spec_id": spec_id, "refusal": refusal, "code": "over_ceiling"}


async def test_fetch_of_data_refused_for_a_missing_capability_is_refused_again_without_a_call():
    executor = FakeExecutor()
    tools, ctx = _tools(executor=executor)
    breakdown = {"code_system": "crsPurpose", "level": 0}

    first = await tools["fetch"](SPEC_LIFE, ctx, breakdown=breakdown)
    again = await tools["fetch"](SPEC_LIFE, ctx)
    other = await tools["fetch"](SPEC_AID, ctx)

    assert first["code"] == again["code"] == ToolErrorCode.CAPABILITY_UNAVAILABLE.value
    assert "frames" not in again
    assert [args["spec_id"] for tool, args in executor.calls if tool == "get_data"] == [
        SPEC_LIFE,
        SPEC_AID,
    ]
    assert other["frames"]


async def test_fetch_of_a_spec_a_plan_step_was_refused_for_a_missing_capability_makes_no_call():
    executor = FakeExecutor()
    tools, ctx = _tools(executor=executor)
    ctx.state[ORCHESTRATOR_UNAVAILABLE_KEY] = {SPEC_LIFE: "The 'breakdown' option isn't live."}

    found = await tools["fetch"](SPEC_LIFE, ctx)

    assert found["code"] == ToolErrorCode.CAPABILITY_UNAVAILABLE.value
    assert executor.calls == []
    assert not ctx.state.get(ORCHESTRATOR_CALLS_KEY)


class FailingGetDataExecutor(FakeExecutor):
    """An executor whose `get_data` call raises `error`, as a dropped stream does."""

    def __init__(self, error: Exception) -> None:
        super().__init__()
        self.error = error

    async def __call__(self, *, tool: str, args: dict[str, Any], on_progress: Any = None):
        if tool == "get_data":
            self.calls.append((tool, args))
            raise self.error
        return await super().__call__(tool=tool, args=args, on_progress=on_progress)


@pytest.mark.parametrize(
    ("error", "code"),
    [
        (
            MCPError.from_error_data(
                ErrorData(code=CONNECTION_CLOSED, message="SSE stream ended without a response")
            ),
            "transient_unavailable",
        ),
        (httpx2.ConnectError("connection refused"), "transient_unavailable"),
        # A deadline overrun is no transport failure: the same call runs as long again.
        (
            MCPError.from_error_data(
                ErrorData(code=REQUEST_TIMEOUT, message="get_data ran past its deadline")
            ),
            "deadline_exceeded",
        ),
    ],
    ids=["connection-closed", "connect-error", "request-timeout"],
)
async def test_fetch_returns_a_refusal_when_get_data_does_not_complete(error: Exception, code: str):
    tools, ctx = _tools(executor=FailingGetDataExecutor(error))

    found = await tools["fetch"](SPEC_LIFE, ctx)

    assert found["spec_id"] == SPEC_LIFE
    assert found["code"] == code
    assert "frames" not in found


class SurfaceExecutor(FakeExecutor):
    """An executor whose `get_data` call goes to `surface`, as `McpExecutor`'s does."""

    def __init__(self, surface: Any) -> None:
        super().__init__()
        self.surface = surface

    async def __call__(self, *, tool: str, args: dict[str, Any], on_progress: Any = None):
        if tool == "get_data":
            await self.surface.call_tool(tool=ToolId(server="data-mcp", name=tool), arguments=args)
        return await super().__call__(tool=tool, args=args, on_progress=on_progress)


class UncoveredServerSurface:
    """A surface that covers no server, whose failure no retry changes."""

    async def call_tool(self, *, tool: ToolId, arguments: Any, on_progress: Any = None):
        raise ToolSurfaceUnavailableError("no server named 'data-mcp' is covered", retryable=False)


@pytest.mark.parametrize(
    ("surface", "code"),
    [
        (unreachable_surface(), "transient_unavailable"),
        (UncoveredServerSurface(), "tool_not_configured"),
    ],
    ids=["server-down", "server-not-covered"],
)
async def test_fetch_answers_a_surface_failure_by_whether_a_retry_can_help(surface: Any, code: str):
    tools, ctx = _tools(executor=SurfaceExecutor(surface))

    found = await tools["fetch"](SPEC_LIFE, ctx)

    assert (found["spec_id"], found["code"]) == (SPEC_LIFE, code)
    assert "frames" not in found


@pytest.mark.parametrize(
    "surface",
    [unreachable_surface(), UncoveredServerSurface()],
    ids=["server-down", "server-not-covered"],
)
async def test_the_kind_read_of_a_frame_names_nothing_when_the_surface_fails(surface: Any):
    cells = await orchestrator_tools._read_member_cells(
        {"ref": {"spec_id": SPEC_LIFE}}, SurfaceExecutor(surface)
    )

    assert cells is None


def _role_open_block(spec_id: str, **identity: Any) -> dict[str, Any]:
    """The recorded education-aid block under `spec_id`, its identity members replaced."""
    block = copy.deepcopy(ROLE_OPEN_SPEC)
    block["spec_id"] = spec_id
    block["definition"]["identity"].update(identity)
    return block


GHANA = {"entities": {"destination": ["country/GHA"]}}
WINDOW_2023 = {"kind": "range", "start_year": 2023, "end_year": 2023}


@pytest.mark.parametrize(
    ("second", "scope", "calls"),
    [
        # The identities differ only in the window the scope replaces.
        ({"requested_window": {"kind": "latest"}}, {"window": WINDOW_2023}, 1),
        # The identities differ in the entities no scope replaces.
        (GHANA, {"window": WINDOW_2023}, 2),
        # The scope replaces the entities the identities differ in.
        (GHANA, {"entities": {"destination": ["Kenya"]}}, 1),
        # A scope on the other side leaves the second's open origin open.
        ({"open_roles": ["origin"]}, {"entities": {"destination": ["Kenya"]}}, 2),
        # A scope on the open side, in a caller's word for it, is not read as
        # closing the second's open origin, so the two requests key apart.
        ({"open_roles": ["origin"]}, {"entities": {"donor": ["France"]}}, 2),
    ],
)
async def test_fetch_of_the_same_data_twice_in_a_turn_retrieves_it_once(
    second: dict[str, Any], scope: dict[str, Any], calls: int
):
    executor = FakeExecutor()
    tools, ctx = _tools(executor=executor)
    ctx.state["orchestrator:specs"] = {
        SPEC_AID: _role_open_block(SPEC_AID),
        SPEC_LIFE: _role_open_block(SPEC_LIFE, **second),
    }

    first = await tools["fetch"](SPEC_AID, ctx, scope=scope)
    again = await tools["fetch"](SPEC_LIFE, ctx, scope=scope)

    assert len([tool for tool, _ in executor.calls if tool == "get_data"]) == calls
    assert again["frames"] == first["frames"]
    assert again.get("same_data_as") == (SPEC_AID if calls == 1 else None)


class _ServedExecutor(FakeExecutor):
    """Answers every `get_data` with how its frame was served: the example
    part's identity with its regional bucket left out."""

    def __init__(self) -> None:
        super().__init__()
        self.served = served_facts(_manifest(_served_identity(residuals=False)))

    async def __call__(self, *, tool: str, args: dict[str, Any], on_progress: Any = None):
        outcome = await super().__call__(tool=tool, args=args, on_progress=on_progress)
        if tool == "get_data":
            outcome["served"] = self.served
        return outcome


async def test_fetch_keeps_how_its_data_was_served_on_the_readings_and_from_the_turns_cache():
    """A fetch keeps the served record on its spec_id's reading, and a fetch
    of the same data again keeps the cached one, though `get_data` runs once
    and the model reads none of it."""
    executor = _ServedExecutor()
    tools, ctx = _tools(executor=executor)
    ctx.state["orchestrator:specs"] = {
        SPEC_AID: _role_open_block(SPEC_AID),
        SPEC_LIFE: _role_open_block(SPEC_LIFE, requested_window={"kind": "latest"}),
    }
    ctx.state["orchestrator:searches"] = [
        SearchReading.from_part({**_example_part(), "spec_id": spec_id}).model_dump(mode="json")
        for spec_id in (SPEC_AID, SPEC_LIFE)
    ]

    first = await tools["fetch"](SPEC_AID, ctx, scope={"window": WINDOW_2023})
    again = await tools["fetch"](SPEC_LIFE, ctx, scope={"window": WINDOW_2023})

    readings = [SearchReading.model_validate(r) for r in ctx.state["orchestrator:searches"]]
    assert len([tool for tool, _ in executor.calls if tool == "get_data"]) == 1
    assert [
        r.fetch.served.residuals("destination") for r in readings if r.fetch and r.fetch.served
    ] == [
        frozenset(),
        frozenset(),
    ]
    assert "served" not in first and "served" not in again
    assert basis_line(ctx.state["orchestrator:searches"], specs={}, plans=[]) == (
        "Counted as: health (CRS 120) and population and reproductive health (CRS 130) · "
        "Africa's countries only · "
        "gross disbursements, current USD."
    )


class _NotedExecutor(FakeExecutor):
    """Answers every `get_data` with `note` as its period note, or with none."""

    def __init__(self, note: str | None) -> None:
        super().__init__()
        self.note = note

    async def __call__(self, *, tool: str, args: dict[str, Any], on_progress: Any = None):
        outcome = await super().__call__(tool=tool, args=args, on_progress=on_progress)
        if tool == "get_data" and self.note:
            outcome["period_note"] = self.note
        return outcome


@pytest.mark.parametrize("searched", [True, False], ids=["after_a_search", "no_search_reading"])
async def test_fetch_states_the_period_note_its_data_carries(searched: bool):
    """The note travels with the retrieved data, so a fetch of a spec_id no
    search read this turn states it as a fetch after a search does, and so
    does the same fetch again from the turn's cache."""
    tools, ctx = _tools(
        _resolver(resolver_digest(_recorded_envelope())), executor=_NotedExecutor("2025 has 9")
    )
    if searched:
        await tools["search"](ctx, [{"variable": "education aid"}])
    assert bool(ctx.state.get("orchestrator:searches")) is searched

    found = await tools["fetch"](ROLE_OPEN_ID if searched else SPEC_LIFE, ctx)
    again = await tools["fetch"](ROLE_OPEN_ID if searched else SPEC_LIFE, ctx)

    assert found["period_note"] == "2025 has 9"
    assert found["summary"] == "64 observations"
    assert again["same_data_as"] == found["spec_id"]
    assert again["period_note"] == "2025 has 9"


async def test_fetch_of_data_with_nothing_to_note_has_no_period_note():
    tools, ctx = _tools(executor=_NotedExecutor(None))

    found = await tools["fetch"](SPEC_LIFE, ctx)
    again = await tools["fetch"](SPEC_LIFE, ctx)

    assert "period_note" not in found
    assert "period_note" not in again


async def test_fetch_of_one_spec_with_other_arguments_retrieves_again():
    executor = FakeExecutor()
    tools, ctx = _tools(executor=executor)

    await tools["fetch"](SPEC_LIFE, ctx)
    again = await tools["fetch"](SPEC_LIFE, ctx)
    await tools["fetch"](SPEC_LIFE, ctx, scope={"window": WINDOW_2023})
    await tools["fetch"]("spec_other", ctx)

    assert [args["spec_id"] for tool, args in executor.calls if tool == "get_data"] == [
        SPEC_LIFE,
        SPEC_LIFE,
        "spec_other",
    ]
    assert again["same_data_as"] == SPEC_LIFE


async def test_fetch_leaves_a_breakdown_beside_a_sum_for_get_data_to_judge():
    executor = FakeExecutor()
    tools, ctx = _tools(executor=executor)
    breakdown = {"code_system": "crsPurpose", "level": 0}

    await tools["fetch"](SPEC_AID, ctx, stat_vars="sum", breakdown=breakdown)

    (sent,) = [args for tool, args in executor.calls if tool == "get_data"]
    assert (sent["stat_vars"], sent["breakdown"]) == ("sum", breakdown)


@pytest.mark.parametrize(
    ("stated", "sent"),
    [
        # A stated reading wins over the family rule's total.
        ({"stat_vars": "separate"}, {"stat_vars": "separate"}),
        # A breakdown reads the family separately.
        (
            {"breakdown": {"code_system": "crsPurpose", "level": 0}},
            {"stat_vars": "separate", "breakdown": {"code_system": "crsPurpose", "level": 0}},
        ),
    ],
)
async def test_fetch_forwards_a_stated_reading_and_a_breakdown(
    stated: dict[str, Any], sent: dict[str, Any]
):
    executor = FakeExecutor()
    tools, ctx = _tools(executor=executor)
    ctx.state["orchestrator:specs"] = {SPEC_AID: FAMILY_BLOCK}

    await tools["fetch"](SPEC_AID, ctx, **stated)

    assert [args for tool, args in executor.calls if tool == "get_data"] == [
        {"spec_id": SPEC_AID, **sent}
    ]


async def test_the_download_surface_fetches_once_and_never_plans():
    tools, ctx = _tools(surface_name="download")

    assert "frames" in await tools["fetch"](SPEC_LIFE, ctx)
    assert "limit" in (await tools["fetch"](SPEC_LIFE, ctx))["error"]
    assert (
        "limit"
        in tools["plan"]("Kenya's aid as a share of GDP", ["Kenya's aid as a share of GDP"], ctx)[
            "error"
        ]
    )
    assert pending_action(ctx.state) is None


# --- make_charts ----------------------------------------------------------


async def test_make_charts_draws_several_charts_in_one_call():
    executor = FakeExecutor()
    tools, ctx = _tools(executor=executor)
    await tools["fetch"](SPEC_LIFE, ctx)
    frame = frame_id_for(REF)
    select = json.dumps({"observationAbout": "country/NGA"})

    drawn = await tools["make_charts"](
        [
            {"frame": frame, "type": "line_chart", "title": "Over time", "select_json": select},
            {"frame": frame, "type": "stat_card", "title": "Latest", "select_json": select},
        ],
        ctx,
    )

    writes = [args["intent"] for tool, args in executor.calls if tool == "write_component"]
    assert [w["type"] for w in writes] == ["line_chart", "stat_card"]
    assert all(w["frame_ref"] == REF for w in writes)
    assert len(executor.landed) == 2
    assert all("reply_figures" in chart for chart in drawn["charts"])


def _figures(card: dict[str, Any]) -> list[tuple[Any, ...]]:
    """Each of a drawn card's reply figures as (series, value, period), and its mark."""
    return [
        (one["series"], one["value"], one["period"], *filter(None, [one.get("mark")]))
        for one in card["reply_figures"]
    ]


@pytest.mark.parametrize(
    ("asked", "figures"),
    [
        ({}, [("NGA", 54.5, "2024")]),
        ({"state_span": True}, [("NGA", 47.2, "2000"), ("NGA", 54.5, "2024")]),
        (
            {"state_values_for": ["country/NGA"], "state_period": "2010"},
            [("NGA", 50.9, "2010"), ("NGA", 54.5, "2024")],
        ),
    ],
)
async def test_make_charts_returns_what_a_card_holds_and_the_figures_a_reply_may_state(
    asked: dict[str, Any], figures: list[tuple[Any, ...]]
) -> None:
    """A drawn card's entry holds its cover and its figures, read from the
    drawn payload, and none of the platform's per-series summary."""
    tools, ctx = _tools()
    await tools["fetch"](SPEC_LIFE, ctx)

    drawn = await tools["make_charts"]([_one_chart(frame_id_for(REF))[0] | asked], ctx)

    (card,) = drawn["charts"]
    assert set(card) == {"title", "holds", "reply_figures"}
    assert (card["holds"]["series_count"], card["holds"]["series"]) == (1, ["NGA"])
    assert (card["holds"]["first_period"], card["holds"]["last_period"]) == ("2000", "2024")
    assert _figures(card) == figures


async def test_a_card_whose_figures_cannot_be_read_keeps_its_entry_and_the_charts_after_it(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Building a card's figures runs after the card is on the canvas, so a
    failure there costs that card's figures alone."""
    read = orchestrator_tools.card_result

    def unreadable_once(series: Any, **asked: Any) -> dict[str, Any]:
        monkeypatch.setattr(orchestrator_tools, "card_result", read)
        raise TypeError("'<' not supported between instances of 'NoneType' and 'int'")

    monkeypatch.setattr(orchestrator_tools, "card_result", unreadable_once)
    executor = FakeExecutor()
    tools, ctx = _tools(executor=executor)
    await tools["fetch"](SPEC_LIFE, ctx)
    chart = _one_chart(frame_id_for(REF))[0]

    with caplog.at_level(logging.ERROR, logger=orchestrator_tools.__name__):
        drawn = await tools["make_charts"](
            [chart | {"title": "Unread"}, chart | {"type": "table"}], ctx
        )

    unread, after = drawn["charts"]
    assert (unread["title"], unread["reply_figures"]) == ("Unread", [])
    assert _stated(unread) == (HarnessCode.FIGURES_UNREAD, "do_not_retry")
    assert "error" not in unread
    assert _figures(after) == [("NGA", 54.5, "2024")]
    assert len(executor.landed) == 2
    assert ctx.state["orchestrator:calls"]["make_charts"] == 1
    assert ["Unread" in record.getMessage() for record in caplog.records] == [True]


@pytest.mark.parametrize("removed_first", [True, False])
async def test_a_redraw_of_a_series_on_the_canvas_as_the_turn_began_states_no_figure(
    removed_first: bool,
) -> None:
    """The canvas is read as the turn begins, so taking the old card off
    before the redraw still leaves its series a redrawn one. Naming the
    series gives its value all the same, and the next turn reads the canvas
    afresh."""
    executor = FakeExecutor()
    state = ConversationState(conversation_id="c1")
    land_component(state, executor.written)
    tools, ctx = _tools(
        executor=executor,
        canvas_series=state.canvas_series_ids,
        remove_cards=_remove_callable(state),
    )
    await tools["fetch"](SPEC_LIFE, ctx)
    chart = _one_chart(frame_id_for(REF))[0]

    if removed_first:
        assert (await tools["remove_cards"](ctx, all_cards=True))["removed"]
    redrawn = await tools["make_charts"]([chart], ctx)
    named = await tools["make_charts"]([chart | {"state_values_for": ["country/NGA"]}], ctx)
    ctx.invocation_id = "inv_2"
    next_turn = await tools["make_charts"]([chart], ctx)

    assert _figures(redrawn["charts"][0]) == []
    assert _figures(named["charts"][0]) == [("NGA", 54.5, "2024")]
    assert _figures(next_turn["charts"][0]) == ([("NGA", 54.5, "2024")] if removed_first else [])


def _bars(codes: list[str]) -> dict[str, Any]:
    """The write reply of a bar chart of `codes` at 2024, each valued by its place in the list."""
    series = {code: {"2024": float(number)} for number, code in enumerate(codes)}
    return drawn_cards.write_reply(
        drawn_cards.chart("bar_chart", series), drawn_cards.stored(codes)
    )


TWENTY_FOUR = [f"country/P{number:02d}" for number in range(24)]


@pytest.mark.parametrize(
    ("on_canvas", "stored_ids", "asked", "figures"),
    [
        # Every one of the 24 series was on the canvas, so it is a redraw.
        (TWENTY_FOUR, True, {}, []),
        (TWENTY_FOUR, True, {"state_values_for": ["P17"]}, [("P17", 17.0, "2024")]),
        # One series the canvas did not draw makes it a new card.
        (
            [*TWENTY_FOUR[:23], "country/Q00"],
            True,
            {},
            [("P23", 23.0, "2024", "largest"), ("P00", 0.0, "2024", "smallest")],
        ),
        # An entry stored without its series ids gives those its fact summary
        # lists, twelve of a card. The 24 read as new, and twelve as a redraw.
        (
            TWENTY_FOUR,
            False,
            {},
            [("P23", 23.0, "2024", "largest"), ("P00", 0.0, "2024", "smallest")],
        ),
        (TWENTY_FOUR[:12], False, {}, []),
    ],
)
async def test_a_redraw_is_judged_against_every_series_of_every_card_on_the_canvas(
    on_canvas: list[str], stored_ids: bool, asked: dict[str, Any], figures: list[tuple[Any, ...]]
) -> None:
    executor = FakeExecutor()
    state = ConversationState(conversation_id="c1")
    land_component(state, _bars(on_canvas))
    if not stored_ids:
        (entry,) = state.components.values()
        del entry["series_ids"]
    drawn = TWENTY_FOUR if stored_ids else on_canvas
    executor.written = _bars(drawn)
    tools, ctx = _tools(executor=executor, canvas_series=state.canvas_series_ids)
    await tools["fetch"](SPEC_LIFE, ctx)

    (card,) = (await tools["make_charts"]([_one_chart(frame_id_for(REF))[0] | asked], ctx))[
        "charts"
    ]

    assert card["holds"]["series_count"] == len(drawn)
    assert _figures(card) == figures


FOURTEEN_DONORS = {f"donor/{number:02d}": {"2024": float(number + 76)} for number in range(14)}
NAMED_STORE_REF = {"spec_id": "ops_" + "a" * 32, "content_version": "ocv_" + "a" * 32}
REPLAYED_REF = {"spec_id": "ops_" + "b" * 32, "content_version": "ocv_" + "b" * 32}


class ReplayingPlatform(FakeExecutor):
    """Draws every card on `REPLAYED_REF`, as a write the rolling-card rules
    moved to a replayed frame comes back: the revision's input frame is the
    replayed one, not the frame the intent named. The card is a bar chart of
    `FOURTEEN_DONORS`, whose platform summary states the coordinates of
    twelve."""

    def __init__(self) -> None:
        super().__init__()
        envelope = drawn_cards.chart("bar_chart", FOURTEEN_DONORS)
        envelope["provenance"]["inputs"] = [REPLAYED_REF]
        self.written = drawn_cards.write_reply(envelope, drawn_cards.stored(FOURTEEN_DONORS))

    async def __call__(self, *, tool: str, args: dict[str, Any], on_progress: Any = None):
        if tool == "get_data":
            fetched = await super().__call__(tool=tool, args=args, on_progress=on_progress)
            return {**fetched, "store_refs": [NAMED_STORE_REF]}
        return await super().__call__(tool=tool, args=args, on_progress=on_progress)


class TwoFrames:
    """The store: the replayed frame holds `FOURTEEN_DONORS`, and the frame
    the intent named holds the same donors one value lower."""

    async def open(self, ref: Any) -> SimpleNamespace | None:
        shift = {REPLAYED_REF["content_version"]: 0.0, NAMED_STORE_REF["content_version"]: 1.0}
        if ref.content_version not in shift:
            return None
        rows = [
            {
                **drawn_cards.coordinates(code),
                "time_period": period,
                "value": value - shift[ref.content_version],
            }
            for code, points in FOURTEEN_DONORS.items()
            for period, value in points.items()
        ]
        frame = Frame.build(role_keys=["observationAbout"], rows=rows)

        async def read(*, max_rows: int) -> tuple[Frame, Any]:
            return frame, None

        return SimpleNamespace(num_rows=frame.num_rows, read=read)


async def test_a_named_series_is_placed_by_the_rows_of_the_frame_the_card_draws(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A card the rolling-card rules moved to a replayed frame draws that
    frame's values, so a series the platform's summary states no coordinates
    for is placed by its rows, not by those of the frame the intent named."""
    monkeypatch.setattr(orchestrator_tools, "get_frame_store", TwoFrames)
    tools, ctx = _tools(executor=ReplayingPlatform())
    await tools["fetch"](SPEC_LIFE, ctx)
    chart = _one_chart(frame_id_for(REF))[0] | {"state_values_for": ["donor/12"]}

    (card,) = (await tools["make_charts"]([chart], ctx))["charts"]

    assert "not_matched" not in card
    assert ("12", 88.0, "2024") in _figures(card)


async def _held_per_capita_sides(store: InMemoryMaterialisationStore) -> tuple[FrameRef, FrameRef]:
    """Aid to Kenya in US dollars and Kenya's population, 2020 to 2022, each held as
    `get_data` holds a fetched frame."""
    sides = []
    for spec, stat_var, unit, values in (
        (
            "d",
            "dcid:OfficialDevelopmentAssistanceNet_Disbursement",
            "dcid:USDollar",
            (100.0, 120.0, 150.0),
        ),
        ("e", "dcid:Count_Person", None, (50.0, 50.0, 60.0)),
    ):
        frame = Frame.build(
            role_keys=["observationAbout"],
            rows=[
                {
                    "stat_var": stat_var,
                    "observationAbout": "country/KEN",
                    "time_period": period,
                    "value": value,
                    "unit": unit,
                }
                for period, value in zip(("2020", "2021", "2022"), values, strict=True)
            ],
        )
        ref = FrameRef(spec_id="spec_" + spec * 32, content_version="cv_" + spec * 32)
        await hold_root(
            store,
            ref,
            frame=frame,
            trace=MaterialisationTrace(
                upstream_trace={"graph_revision": "rev_test", "facets": []},
                spec_id=ref.spec_id,
                content_version=ref.content_version,
            ),
        )
        sides.append(ref)
    return sides[0], sides[1]


async def test_make_charts_names_every_series_of_a_quotient_with_a_composed_unit() -> None:
    """A quotient's rows carry a composed unit, which dfo-apis serves as the JCS
    text of its object. make_charts reads those rows to write an empty select as
    one naming every series, so a quotient it could not decode would be refused
    as unread."""
    store = InMemoryMaterialisationStore()
    set_store(store)
    try:
        aid, population = await _held_per_capita_sides(store)
        result = await analysis_verbs.quotient_of(
            frame_ref=aid.model_dump(), denominator_frame_ref=population.model_dump()
        )
        assert result.is_error is False, result.content[0].text
        quotient = {
            "spec_id": result.structured_content["spec_id"],
            "content_version": result.structured_content["content_version"],
        }
        # The quotient sums each side per period, so its one series is its own stat_var.
        series = f"dcid:{quotient['spec_id']}"
        view = {"dimensions": [{"name": "stat_var", "values": [series], "plural": False}]}
        executor = FakeExecutor()
        tools, ctx = _tools(executor=executor)
        register_frames(ctx.state, [(quotient, view, None)])

        drawn = await tools["make_charts"](
            [
                {
                    "frame": frame_id_for(quotient),
                    "type": "line_chart",
                    "title": "Aid per person",
                    "select_json": "{}",
                }
            ],
            ctx,
        )
    finally:
        set_store(None)

    (chart,) = drawn["charts"]
    assert "error" not in chart, chart
    (write,) = [args["intent"] for tool, args in executor.calls if tool == "write_component"]
    assert write["select"] == {"stat_var": series}


async def test_make_charts_reads_a_derived_frame_once_a_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two cards on one quotient each read its rows for a covering select and a
    drawn period, and dfo-apis serves the file once for the whole call."""
    reads: list[Any] = []
    read_file = orchestrator_tools.derived_file

    async def counting(ref: Any) -> Any:
        reads.append(ref)
        return await read_file(ref)

    monkeypatch.setattr(orchestrator_tools, "derived_file", counting)
    store = InMemoryMaterialisationStore()
    set_store(store)
    try:
        aid, population = await _held_per_capita_sides(store)
        result = await analysis_verbs.quotient_of(
            frame_ref=aid.model_dump(), denominator_frame_ref=population.model_dump()
        )
        assert result.is_error is False, result.content[0].text
        quotient = {
            "spec_id": result.structured_content["spec_id"],
            "content_version": result.structured_content["content_version"],
        }
        series = f"dcid:{quotient['spec_id']}"
        view = {"dimensions": [{"name": "stat_var", "values": [series], "plural": False}]}
        executor = FakeExecutor()
        tools, ctx = _tools(executor=executor)
        register_frames(ctx.state, [(quotient, view, None)])

        drawn = await tools["make_charts"](
            [
                {
                    "frame": frame_id_for(quotient),
                    "type": chart_type,
                    "title": "Aid per person",
                    "select_json": "{}",
                }
                for chart_type in ("line_chart", "table")
            ],
            ctx,
        )
    finally:
        set_store(None)

    assert all("error" not in chart for chart in drawn["charts"]), drawn
    assert len(reads) == 1


async def test_make_charts_refuses_an_empty_list_and_says_it_cannot_clear_the_canvas():
    """An empty call drew nothing and removed nothing; a bare `{"charts": []}`
    read as success, and the model told the user the canvas was cleared."""
    executor = FakeExecutor()
    tools, ctx = _tools(executor=executor)

    refused = await tools["make_charts"]([], ctx)

    assert _stated(refused) == (HarnessCode.TOOL_ARGUMENTS_INVALID, "correct_and_retry")
    assert "cannot remove or clear" in refused["error"]
    assert executor.calls == []
    assert ctx.state["orchestrator:calls"] == {"make_charts_failed": 1}


async def test_make_charts_refuses_a_frame_nothing_returned_and_a_select_the_platform_would_refuse():
    executor = FakeExecutor()
    tools, ctx = _tools(executor=executor)
    await tools["fetch"](SPEC_LIFE, ctx)
    two_arrays = json.dumps({"observationAbout": ["a", "b"], "unit": ["x", "y"]})

    drawn = await tools["make_charts"](
        [
            {"frame": "f999999", "type": "line_chart", "title": "x", "select_json": "{}"},
            {
                "frame": frame_id_for(REF),
                "type": "line_chart",
                "title": "y",
                "select_json": two_arrays,
            },
            {
                "frame": frame_id_for(REF),
                "type": "line_chart",
                "title": "z",
                "select_json": "{observationAbout: KEN}",
            },
        ],
        ctx,
    )

    assert "unknown frame" in drawn["charts"][0]["error"]
    assert "which the frame does not declare" in drawn["charts"][1]["error"]
    # Each is the model's to correct, by naming another frame or writing the select again.
    assert [_stated(chart) for chart in drawn["charts"]] == [
        (HarnessCode.FRAME_UNKNOWN, "correct_and_retry"),
        (HarnessCode.SELECT_INVALID, "correct_and_retry"),
        (HarnessCode.SELECT_INVALID, "correct_and_retry"),
    ]
    assert drawn["charts"][1]["fix"] == {"dimensions": ["observationAbout"]}
    assert not any(tool == "write_component" for tool, _ in executor.calls)


class CappedViewExecutor(FakeExecutor):
    """Describes a frame whose view lists two of its three places."""

    async def __call__(self, *, tool: str, args: dict[str, Any], on_progress: Any = None):
        if tool != "describe_frame":
            return await super().__call__(tool=tool, args=args, on_progress=on_progress)
        place = {"name": "observationAbout", "values": ["country/KEN", "country/NGA"]}
        return {
            "outcome": "ok",
            "view": {"dimensions": [{**place, "plural": True, "total_count": 3}, "unit"]},
        }


class RefusingExecutor(FakeExecutor):
    """Refuses every write as the executor relays presentation-mcp's
    binding_invalid refusal, its `fix` held as `advice`."""

    async def __call__(self, *, tool: str, args: dict[str, Any], on_progress: Any = None):
        if tool != "write_component":
            return await super().__call__(tool=tool, args=args, on_progress=on_progress)
        self.calls.append((tool, args))
        return {
            "outcome": "refusal",
            "refusal": {
                "code": "binding_invalid",
                "retry_mode": "correct_and_retry",
                "origin": "service",
                "fix": None,
                "advice": "legal values: Germany (country/DEU), France (country/FRA)",
                "message": "the select leaves donor unbound",
            },
        }


class RaisingExecutor(FakeExecutor):
    """Raises on write number `fails_on` (1-based), as a dropped stream does."""

    def __init__(self, fails_on: int = 1) -> None:
        super().__init__()
        self.fails_on = fails_on

    async def __call__(self, *, tool: str, args: dict[str, Any], on_progress: Any = None):
        if tool == "write_component" and _writes(self) + 1 == self.fails_on:
            self.calls.append((tool, args))
            raise RuntimeError("presentation-mcp is down")
        return await super().__call__(tool=tool, args=args, on_progress=on_progress)


class UnreadableCardExecutor(FakeExecutor):
    """Answers every write as `McpExecutor` does for a card the platform stored
    and the harness cannot read (`harness.mcp_client._withheld_card`)."""

    async def __call__(self, *, tool: str, args: dict[str, Any], on_progress: Any = None):
        if tool != "write_component":
            return await super().__call__(tool=tool, args=args, on_progress=on_progress)
        self.calls.append((tool, args))
        return {
            "outcome": "refusal",
            "refusal": harness_refusal(
                code=HarnessCode.STORED_CARD_UNREADABLE,
                message="the platform stored card cmp_1 but it fails the component contract",
            )
            | {"component_id": "cmp_1"},
        }


async def test_a_stored_card_the_harness_cannot_read_is_a_dfo_fault_not_a_refusal() -> None:
    """The platform stored the chart and the harness could not read it back.
    The model is told not to redraw it or swap in other data, the call draws
    nothing after it, and the same chart in a later call this turn is answered
    from the record rather than written again."""
    executor = UnreadableCardExecutor()
    tools, ctx = _tools(executor=executor)
    await tools["fetch"](SPEC_LIFE, ctx)

    frame = frame_id_for(REF)
    drawn = await tools["make_charts"]([*_one_chart(frame), *_one_chart(frame)], ctx)

    # The call stops at the first unreadable card: a second chart would meet it too.
    (entry,) = drawn["charts"]
    assert [tool for tool, _ in executor.calls].count("write_component") == 1
    assert _stated(entry) == (HarnessCode.STORED_CARD_UNREADABLE, "do_not_retry")
    assert "refused" not in entry["error"]
    assert executor.landed == []

    again = await tools["make_charts"](_one_chart(frame), ctx)
    (repeat,) = again["charts"]
    assert [tool for tool, _ in executor.calls].count("write_component") == 1
    assert _stated(repeat) == (HarnessCode.STORED_CARD_UNREADABLE, "do_not_retry")


def _one_chart(frame: str) -> list[dict[str, str]]:
    select = json.dumps({"observationAbout": "country/NGA"})
    return [{"frame": frame, "type": "bar_chart", "title": "t", "select_json": select}]


def _distinct_charts(frame: str, count: int) -> list[dict[str, Any]]:
    """`count` charts of one frame that differ in the window they show, so none
    repeats a card another draws."""
    return [
        chart | {"latest_periods": number}
        for number, chart in enumerate(_one_chart(frame) * count, 1)
    ]


def _stated(failed: dict[str, Any]) -> tuple[Any, Any]:
    """The code and retry mode a failed answer or chart entry states."""
    return failed.get("code"), failed.get("retry_mode")


def _writes(executor: FakeExecutor) -> int:
    return sum(tool == "write_component" for tool, _ in executor.calls)


async def test_make_charts_spends_its_drawn_budget_only_on_calls_that_drew():
    executor = FakeExecutor()
    tools, ctx = _tools(executor=executor)
    await tools["fetch"](SPEC_LIFE, ctx)
    frame = frame_id_for(REF)

    for _ in range(BUDGETS["make_charts_failed"] - 1):
        failed = await tools["make_charts"](_one_chart("f999999"), ctx)
        assert "error" in failed["charts"][0]
    *charts, last = _distinct_charts(frame, BUDGETS["make_charts"] + 1)
    for chart in charts:
        drawn = await tools["make_charts"]([chart], ctx)
        assert "reply_figures" in drawn["charts"][0]
    writes = _writes(executor)
    spent = await tools["make_charts"]([last], ctx)

    assert _stated(spent) == (HarnessCode.TOOL_BUDGET_SPENT, "do_not_retry")
    assert "charts" not in spent
    assert _writes(executor) == writes


async def test_make_charts_stops_after_its_cap_of_calls_that_drew_nothing():
    executor = RefusingExecutor()
    tools, ctx = _tools(executor=executor)
    await tools["fetch"](SPEC_LIFE, ctx)
    frame = frame_id_for(REF)
    ctx.state["orchestrator:calls"] = {"make_charts": 1}  # one call drew

    for _ in range(BUDGETS["make_charts_failed"]):
        refused = await tools["make_charts"](_one_chart(frame), ctx)
        assert "error" in refused["charts"][0]
    writes = _writes(executor)
    spent = await tools["make_charts"](_one_chart(frame), ctx)

    assert _stated(spent) == (HarnessCode.TOOL_BUDGET_SPENT, "do_not_retry")
    assert "charts" not in spent
    assert _writes(executor) == writes


class ScopedFrameExecutor(FakeExecutor):
    """Serves `scoped_ref` for a fetch with a scope, as a narrower retrieval
    states its own frame."""

    def __init__(self, scoped_ref: dict[str, Any]) -> None:
        super().__init__()
        self.scoped_ref = scoped_ref

    async def __call__(self, *, tool: str, args: dict[str, Any], on_progress: Any = None):
        if tool == "get_data" and "scope" in args:
            self.calls.append((tool, args))
            return {"outcome": "ok", "frame_refs": [self.scoped_ref], "summary": "8 observations"}
        return await super().__call__(tool=tool, args=args, on_progress=on_progress)


REF_NARROW = {"spec_id": SPEC_LIFE, "content_version": "cv_2"}


async def _fetched(tools: dict[str, Any], ctx: Any, executor: FakeExecutor, ref: dict) -> None:
    """A fetch with a different scope, whose `get_data` states `ref`."""
    window = {"kind": "range", "start_year": 2015, "end_year": 2020}
    await tools["fetch"](SPEC_LIFE, ctx, scope={"window": window})


async def _handed_back(tools: dict[str, Any], ctx: Any, executor: FakeExecutor, ref: dict) -> None:
    """A plan's hand-back of `ref`, listed as `plan_done` lists it."""
    hand_back = HandBack(executor)
    await hand_back.describe(ref, tool="aggregate", store_ref=None)
    hand_back.listing(ctx.state, None)


@pytest.mark.parametrize("arrive", [_fetched, _handed_back], ids=["fetch", "hand_back"])
@pytest.mark.parametrize(
    ("arriving", "draws"), [(REF_NARROW, True), (REF, False)], ids=["new_frame", "held_frame"]
)
async def test_make_charts_draws_a_frame_that_arrived_after_its_cap_of_calls_that_drew_nothing(
    arrive, arriving, draws
):
    """A frame registered after the calls that drew nothing clears their
    count; a registration of a frame already held leaves it spent."""
    executor = ScopedFrameExecutor(scoped_ref=arriving)
    tools, ctx = _tools(executor=executor)
    await tools["fetch"](SPEC_LIFE, ctx)
    for _ in range(BUDGETS["make_charts_failed"]):
        failed = await tools["make_charts"](_one_chart("f999999"), ctx)
        assert "error" in failed["charts"][0]
    spent = await tools["make_charts"](_one_chart(frame_id_for(REF)), ctx)
    assert _stated(spent) == (HarnessCode.TOOL_BUDGET_SPENT, "do_not_retry")

    await arrive(tools, ctx, executor, arriving)
    charted = await tools["make_charts"](_one_chart(frame_id_for(arriving)), ctx)

    if draws:
        assert "holds" in charted["charts"][0]
    else:
        assert _stated(charted) == (HarnessCode.TOOL_BUDGET_SPENT, "do_not_retry")
    assert _writes(executor) == int(draws)


@pytest.mark.parametrize(
    ("fails_on", "results", "counted"),
    [(1, ["error"], "make_charts_failed"), (2, ["holds", "error"], "make_charts")],
)
async def test_make_charts_reports_a_write_that_raised_and_draws_nothing_after_it(
    fails_on, results, counted
):
    executor = RaisingExecutor(fails_on=fails_on)
    tools, ctx = _tools(executor=executor)
    await tools["fetch"](SPEC_LIFE, ctx)

    drawn = await tools["make_charts"](_distinct_charts(frame_id_for(REF), 3), ctx)

    assert ["error" if "error" in chart else "holds" for chart in drawn["charts"]] == results
    assert "RuntimeError" in drawn["charts"][-1]["error"]
    # The write was sent, so the card may be on the canvas.
    assert _stated(drawn["charts"][-1]) == (HarnessCode.WRITE_UNCONFIRMED, "do_not_retry")
    assert _writes(executor) == fails_on
    assert len(executor.landed) == fails_on - 1
    assert ctx.state["orchestrator:calls"] == {"fetch": 1, counted: 1}


class YieldingExecutor(FakeExecutor):
    """Gives way to the event loop before each call, as a network call does,
    so concurrent tool calls interleave."""

    async def __call__(self, *, tool: str, args: dict[str, Any], on_progress: Any = None):
        await asyncio.sleep(0)
        return await super().__call__(tool=tool, args=args, on_progress=on_progress)


def _sibling_calls(session_state: dict[str, Any], count: int) -> list[SimpleNamespace]:
    """Contexts for `count` function calls of one model response, built the way
    ADK builds them: each call's own state delta over the one session state."""
    return [
        SimpleNamespace(state=State(value=session_state, delta={}), invocation_id="inv_1")
        for _ in range(count)
    ]


async def _session_with_frame(executor: FakeExecutor) -> tuple[dict[str, Any], Any]:
    """The tools, and a session state holding one fetched frame."""
    tools, ctx = _tools(executor=executor)
    await tools["fetch"](SPEC_LIFE, ctx)
    return tools, {"surface": "studio", ORCHESTRATOR_FRAMES_KEY: ctx.state[ORCHESTRATOR_FRAMES_KEY]}


async def test_concurrent_make_charts_calls_draw_no_more_than_the_budget_allows():
    executor = YieldingExecutor()
    tools, session_state = await _session_with_frame(executor)
    session_state[ORCHESTRATOR_CALLS_KEY] = {"make_charts": BUDGETS["make_charts"] - 1}
    chart = _one_chart(frame_id_for(REF))

    results = await asyncio.gather(
        *(tools["make_charts"](chart, call) for call in _sibling_calls(session_state, 3))
    )

    assert sum("charts" in result for result in results) == 1
    assert _writes(executor) == 1
    assert session_state[ORCHESTRATOR_CALLS_KEY]["make_charts"] == BUDGETS["make_charts"]


async def test_make_charts_relays_the_values_a_platform_refusal_would_accept():
    tools, ctx = _tools(executor=RefusingExecutor())
    await tools["fetch"](SPEC_LIFE, ctx)

    refused = await tools["make_charts"](_one_chart(frame_id_for(REF)), ctx)

    assert refused["charts"][0]["advice"] == (
        "legal values: Germany (country/DEU), France (country/FRA)"
    )
    # The platform's own code and retry mode word the user's line.
    assert _stated(refused["charts"][0]) == ("binding_invalid", "correct_and_retry")
    assert refused["charts"][0]["error"] == "the select leaves donor unbound"


WIDE_VIEW: dict[str, Any] = {
    "dimensions": [
        {"name": "stat_var", "values": [], "plural": False, "total_count": 1},
        {"name": "financingProvider", "values": [], "plural": True, "total_count": 30},
        {"name": "financingRecipient", "values": [], "plural": False, "total_count": 1},
    ],
    "period_grain": "year",
    "coverage": ["2023", "2024"],
}


# The pair apis serves the retrieval under, whatever its rendering.
WIDE_REF = {"spec_id": SPEC_AID, "content_version": "cv_" + "c" * 32}
SUMMED = {"by": ["time_period"], "op": "sum"}
# The pair the store holds the summed retrieval under, as data-mcp names a
# `groupby` rendering.
WIDE_STORE_REF = frame_ref_for(**WIDE_REF, groupby=SUMMED).model_dump()
# The pair the store holds the same retrieval under when it is asked with
# `place_types`, with each party's kind.
KINDS_STORE_REF = frame_ref_for(**WIDE_REF, columns=KIND_COLUMNS, groupby=SUMMED).model_dump()


class WideFrameExecutor(FakeExecutor):
    """Fetches aid to Kenya from `providers` providers, described by `view`.
    A write dfo-apis refuses for the values a line chart lists or the series
    a select is estimated at (`_apis_refusal`) is refused as presentation-mcp
    answers it, and the refusal is kept in `refusals`."""

    def __init__(self, *, providers: int = 30, view: dict[str, Any] | None = None) -> None:
        super().__init__()
        self.view = view or WIDE_VIEW
        self.view["dimensions"][1]["total_count"] = providers
        self.refusals: list[Refusal] = []

    def refuse(self, args: dict[str, Any], refusal: Refusal) -> dict[str, Any]:
        """The write `args` refused with `refusal`, as `harness.mcp_client` reads it."""
        self.calls.append(("write_component", args))
        self.refusals.append(refusal)
        result = presentation_server._refusal_result(refusal, tool="write_component")
        return _presentation_refusal_outcome(
            result.structured_content, result.meta, fallback_text=""
        )

    async def __call__(self, *, tool: str, args: dict[str, Any], on_progress: Any = None):
        if tool == "write_component" and (refusal := _apis_refusal(args["intent"], view=self.view)):
            return self.refuse(args, refusal)
        if tool == "get_data":
            self.calls.append((tool, args))
            store_ref = KINDS_STORE_REF if args.get("place_types") else WIDE_STORE_REF
            return {
                "outcome": "ok",
                "frame_refs": [WIDE_REF],
                "store_refs": [store_ref],
                "summary": "60 observations",
            }
        if tool == "describe_frame":
            self.calls.append((tool, args))
            return {"outcome": "ok", "view": self.view}
        return await super().__call__(tool=tool, args=args, on_progress=on_progress)


class HeldFrames:
    """The frame store, holding one frame of `providers` members of `role`
    whose 2024 value is their number and 2023 value its reverse under `wide`,
    or nothing. Under `KINDS_STORE_REF` it holds the same members with their
    enrichment column, and when `carried`, as a retrieval asked with
    `place_types` is stored, under `wide` too. On a party role those in
    `totals` are group totals and the rest parties; on a place role those in
    `totals` are typed `Place` and the rest countries, typed `Country` or
    `Country,Place`. It holds nothing under any other `content_version`."""

    def __init__(
        self,
        *,
        held: bool = True,
        providers: int = 30,
        trace: Any = None,
        totals: frozenset[str] = frozenset(),
        wide: dict[str, Any] = WIDE_STORE_REF,
        role: str = "financingProvider",
        carried: bool = False,
    ) -> None:
        self.held = held
        self.wide_cv = wide["content_version"]
        self.providers = providers
        self.trace = trace
        self.totals = totals
        self.role = role
        self.carried = carried

    def cell(self, member: str) -> str:
        if self.role in PARTY_ROLES:
            return "total" if member in self.totals else "party"
        if member in self.totals:
            return "Place"
        return "Country,Place" if int(member[-2:]) % 2 else "Country"

    def rows(self) -> list[dict[str, Any]]:
        return [
            {
                "stat_var": "dcid:aid",
                self.role: f"donor/{number:02d}",
                "financingRecipient": "country/KEN",
                "time_period": period,
                "value": float(number if period == "2024" else 100 - number),
                "unit": "USDollar",
            }
            for number in range(self.providers)
            for period in ("2023", "2024")
        ]

    async def open(self, ref: Any) -> SimpleNamespace | None:
        kinds_cv = KINDS_STORE_REF["content_version"]
        if not self.held or ref.content_version not in (self.wide_cv, kinds_cv):
            return None
        rows = self.rows()
        roles = [self.role, "financingRecipient"]
        if self.carried or ref.content_version == kinds_cv:
            column = (kind_column if self.role in PARTY_ROLES else type_column)(self.role)
            rows = [{**row, column: self.cell(row[self.role])} for row in rows]
            roles.append(column)
            if self.role in PARTY_ROLES:
                # apis serves a party's class and whole cells beside its kind.
                classes, whole = class_column(self.role), whole_column(self.role)
                rows = [{**row, classes: "official", whole: "false"} for row in rows]
                roles += [classes, whole]
        frame = Frame.build(role_keys=roles, rows=rows)

        async def read(*, max_rows: int) -> tuple[Frame, Any]:
            return frame, self.trace

        return SimpleNamespace(num_rows=frame.num_rows, read=read)


async def _draw_wide(
    monkeypatch: pytest.MonkeyPatch,
    store: Any,
    *,
    providers: int = 30,
    view: dict[str, Any] | None = None,
    **chart: Any,
) -> tuple[WideFrameExecutor, dict[str, Any]]:
    """Fetch the wide frame and draw one chart of it over `store`."""
    monkeypatch.setattr(orchestrator_tools, "get_frame_store", lambda: store)
    executor = WideFrameExecutor(providers=providers, view=copy.deepcopy(view or WIDE_VIEW))
    tools, ctx = _tools(executor=executor)
    await tools["fetch"](SPEC_LIFE, ctx)
    return executor, await tools["make_charts"]([_wide_chart(**chart)], ctx)


def _wide_chart(**chart: Any) -> dict[str, Any]:
    """One chart of the wide frame: a table of every provider unless `chart` says otherwise."""
    return {
        "frame": frame_id_for(WIDE_REF),
        "type": "table",
        "title": "Aid to Kenya",
        "select_json": json.dumps({"financingRecipient": "country/KEN"}),
        **chart,
    }


def _intents(executor: FakeExecutor) -> list[dict[str, Any]]:
    return [args["intent"] for tool, args in executor.calls if tool == "write_component"]


# dfo-apis' refusals of a write over a chart type's limit, copied from its
# assembler's wording (`_check_line_chart_series` and the table row check in
# dfo-data-tools' `assembler/pipeline/candidate.py`), which this package
# cannot import. dfo-apis answers them with status 400. `_apis_refusal` builds
# the refusals the fake platform makes before these two checks.
_LINE_SERIES_REFUSAL = Refusal(
    error="resource_exhausted",
    retry=RetryMode.CORRECT_AND_RETRY,
    detail=(
        "the selection resolves to 20 series, and a line chart draws at most 8; "
        "financingProvider (20 values) is the series dimension; bind it to at most 8 values, "
        "or draw the same select as a table, which draws all 20 series"
    ),
    pointer="/select",
    limit_name="MAX_LINE_CHART_SERIES",
)
_TABLE_ROWS_REFUSAL = Refusal(
    error="resource_exhausted",
    retry=RetryMode.CORRECT_AND_RETRY,
    detail=(
        "selection resolves to 20 series x 240 periods = 4800 rows, over the maximum of "
        "2000; bind the select to at most 8 series over these periods, or narrow the window "
        "to at most 100 periods"
    ),
    pointer="/select",
    limit_name="MAX_TABLE_ROWS",
)


_FACETS = frozenset({"unit", "measurementMethod", "provenance"})


def _distinct_members(values: list[Any]) -> list[str]:
    """`values` without repeats, compared as sorted JSON."""
    return sorted({json.dumps(value, sort_keys=True) for value in values})


def _apis_refusal(intent: dict[str, Any], *, view: dict[str, Any]) -> Refusal | None:
    """dfo-apis' refusal of a write of `intent` on a frame with `view`, in the
    order it checks them: a line chart's list of more than 8 distinct
    values, an empty select (`check_select_nonempty`), then a select
    whose estimate passes 24 series, which multiplies each dimension's bound
    values, 1 for an unbound facet, and the view's count for any other unbound
    dimension (`_check_series_estimate` and `multiplying_advice`). None for any
    other write."""
    select, line = intent.get("select") or {}, intent["type"] == "line_chart"
    for name, value in select.items():
        distinct = _distinct_members(value) if isinstance(value, list) else []
        if line and len(distinct) > LINE_CHART_MAX_VALUES:
            return Refusal(
                error="resource_exhausted",
                retry=RetryMode.CORRECT_AND_RETRY,
                detail=(
                    f"select lists {len(distinct)} values on {name!r}, one series each, "
                    f"over the maximum of {LINE_CHART_MAX_VALUES}; bind it to at most "
                    f"{LINE_CHART_MAX_VALUES} values"
                ),
                pointer=f"/select/{name}",
                limit_name="MAX_LINE_CHART_SERIES",
            )
    if not select:
        held = ", ".join(
            f"{dim['name']} ({dim.get('total_count', 1)} "
            f"{'value' if dim.get('total_count', 1) == 1 else 'values'})"
            for dim in view["dimensions"]
        )
        return Refusal(
            error="binding_invalid",
            retry=RetryMode.CORRECT_AND_RETRY,
            detail=f"select must name at least one dimension; this frame's dimensions are {held}",
            pointer="/select",
        )
    factors = {
        str(dim["name"]): (
            (
                len(_distinct_members(select[dim["name"]]))
                if isinstance(select[dim["name"]], list)
                else 1
            )
            if dim["name"] in select
            else 1
            if dim["name"] in _FACETS
            else max(dim.get("total_count", 1), 1)
        )
        for dim in view["dimensions"]
    }
    estimate = math.prod(factors.values())
    if estimate <= MAX_SERIES_PER_COMPONENT:
        return None
    listed = LINE_CHART_MAX_VALUES if line else MAX_SERIES_PER_COMPONENT
    named = {
        name: f"{name} ({count} values)"
        for name, count in sorted(factors.items(), key=lambda factor: (factor[1], factor[0]))
        if count > 1
    }
    kept = [
        name
        for name, value in select.items()
        if isinstance(value, list) and len(_distinct_members(value)) > 1
    ]
    others = [text for name, text in named.items() if kept != [name]]

    def joined(items: list[str]) -> str:
        return (
            " and ".join(items) if len(items) <= 2 else f"{', '.join(items[:-1])}, and {items[-1]}"
        )

    if len(kept) == 1 and others and len(others) < len(named):
        trim = f", and list at most {listed} of its values" if factors[kept[0]] > listed else ""
        advice = f"bind {joined(others)} to one value, keeping {named[kept[0]]} as the series{trim}"
    elif len(named) == 1:
        advice = f"{next(iter(named.values()))} is the series dimension; bind it to at most {listed} values"
    else:
        advice = (
            f"{joined(list(named.values()))} multiply; bind all but one of them to one value, "
            f"and the other to at most {listed} values"
        )
    over = (
        f"and a line chart draws at most {LINE_CHART_MAX_VALUES}; "
        if line
        else f"over the maximum of {MAX_SERIES_PER_COMPONENT}: "
    )
    return Refusal(
        error="resource_exhausted",
        retry=RetryMode.CORRECT_AND_RETRY,
        detail=f"selectors expand to an estimated {estimate} series, {over}{advice}",
        pointer="/select",
        limit_name="MAX_LINE_CHART_SERIES" if line else "MAX_SERIES_PER_COMPONENT",
    )


def _relayed(refusal: Refusal, *, title: str = "Aid to Kenya") -> dict[str, Any]:
    """The `make_charts` entry of a chart titled `title` that ends with the
    platform's `refusal`: its detail, code, retry mode and limit, unchanged."""
    entry = {
        "title": title,
        "error": refusal.detail,
        "code": refusal.error,
        "retry_mode": refusal.retry.value,
    }
    return entry | ({"limit_name": refusal.limit_name} if refusal.limit_name else {})


class CapExecutor(WideFrameExecutor):
    """`WideFrameExecutor`, whose writes of a `refused` chart type that list
    no dimension's values, or more than eight, and that dfo-apis refuses for
    nothing else, are refused with `refusal`, as dfo-apis refuses the series
    or rows such a select resolves to. A write listing two to eight values of
    each dimension it lists is accepted."""

    def __init__(self, *, refused: str, refusal: Refusal, **wide: Any) -> None:
        super().__init__(**wide)
        self.refused = refused
        self.refusal = refusal

    async def __call__(self, *, tool: str, args: dict[str, Any], on_progress: Any = None):
        intent = args.get("intent") or {}
        # A one-value list reads as its one value.
        listed = [
            _distinct_members(value)
            for value in (intent.get("select") or {}).values()
            if isinstance(value, list) and len(_distinct_members(value)) > 1
        ]
        if (
            tool == "write_component"
            and intent["type"] == self.refused
            and (not listed or any(len(value) > LINE_CHART_MAX_VALUES for value in listed))
            and _apis_refusal(intent, view=self.view) is None
        ):
            return self.refuse(args, self.refusal)
        return await super().__call__(tool=tool, args=args, on_progress=on_progress)


async def _draw_capped(
    monkeypatch: pytest.MonkeyPatch,
    *,
    refused: str,
    refusal: Refusal,
    view: dict[str, Any] | None = None,
    store: Any = None,
    providers: int = 20,
    **chart: Any,
) -> tuple[CapExecutor, dict[str, Any]]:
    """Draw one chart of `providers` providers' aid to Kenya, held in `store`,
    the platform refusing `refused` charts that list no providers with
    `refusal`."""
    held = store or HeldFrames(providers=providers)
    monkeypatch.setattr(orchestrator_tools, "get_frame_store", lambda: held)
    executor = CapExecutor(
        refused=refused,
        refusal=refusal,
        providers=providers,
        view=copy.deepcopy(view or WIDE_VIEW),
    )
    tools, ctx = _tools(executor=executor)
    await tools["fetch"](SPEC_LIFE, ctx)
    return executor, await tools["make_charts"]([_wide_chart(type=refused, **chart)], ctx)


# Twenty providers' aid by month over twenty years: a table of them draws
# 2,000 rows // 240 periods = 8 providers.
_MONTHLY_VIEW = {**WIDE_VIEW, "period_grain": "month", "coverage": ["2005-01", "2024-12"]}


@pytest.mark.parametrize(
    ("type_", "refusal", "view", "draws", "recipient"),
    [
        pytest.param(
            "line_chart", _LINE_SERIES_REFUSAL, None, 8, "country/KEN", id="line-chart-series"
        ),
        pytest.param(
            "line_chart",
            _LINE_SERIES_REFUSAL,
            None,
            8,
            ["country/KEN"],
            id="line-chart-one-value-list",
        ),
        pytest.param(
            "table", _TABLE_ROWS_REFUSAL, _MONTHLY_VIEW, 8, "country/KEN", id="table-rows"
        ),
    ],
)
async def test_make_charts_cuts_a_chart_the_platform_refuses_over_its_limit(
    monkeypatch: pytest.MonkeyPatch,
    type_: str,
    refusal: Refusal,
    view: dict[str, Any] | None,
    draws: int,
    recipient: str | list[str],
) -> None:
    """Twenty providers pass the harness's 24-series check. dfo-apis counts
    the series or rows the select resolves to and refuses the write, naming
    the limit, so the chart is cut to the largest members it draws, highest
    first, and written once more. The cut lists only the providers, a
    one-value list beside them written as its value."""
    executor, result = await _draw_capped(
        monkeypatch,
        refused=type_,
        refusal=refusal,
        view=view,
        select_json=json.dumps({"financingRecipient": recipient}),
    )

    refused, cut = _intents(executor)
    assert refused["select"] == {"financingRecipient": recipient}
    assert cut["select"] == {
        "financingRecipient": "country/KEN",
        "financingProvider": [f"donor/{number:02d}" for number in range(19, 19 - draws, -1)],
    }
    assert cut["title"].startswith("Aid to Kenya, ")
    assert cut["title"].endswith(f" ({draws} largest of 20 in 2024)")
    (entry,) = result["charts"]
    assert "error" not in entry
    assert f"draws the {draws} largest of the 20 members" in entry["note"]


@pytest.mark.parametrize(
    ("type_", "refusal", "view"),
    [
        pytest.param("line_chart", _LINE_SERIES_REFUSAL, None, id="line-chart-series"),
        pytest.param("table", _TABLE_ROWS_REFUSAL, _MONTHLY_VIEW, id="table-rows"),
    ],
)
async def test_make_charts_relays_the_limit_a_refused_chart_with_no_cut_crossed(
    monkeypatch: pytest.MonkeyPatch, type_: str, refusal: Refusal, view: Any
) -> None:
    """A chart whose period moves with the latest data is cut to no largest
    members, so dfo-apis' refusal reaches the model as the platform gave it,
    and the select is written once."""
    executor, result = await _draw_capped(
        monkeypatch, refused=type_, refusal=refusal, view=view, latest_periods=2
    )

    (intent,) = _intents(executor)
    assert "financingProvider" not in intent["select"]
    (entry,) = result["charts"]
    assert entry == _relayed(refusal)


class RowFrames(HeldFrames):
    """The store, holding each provider's aid to Kenya by period as `values` gives it."""

    def __init__(self, values: dict[str, dict[str, float]], **held: Any) -> None:
        super().__init__(providers=len(values), **held)
        self.values = values

    def rows(self) -> list[dict[str, Any]]:
        return [
            {
                "stat_var": "dcid:aid",
                self.role: provider,
                "financingRecipient": "country/KEN",
                "time_period": period,
                "value": value,
                "unit": "USDollar",
            }
            for provider, by_period in self.values.items()
            for period, value in by_period.items()
        ]


_CUT_TITLE = re.compile(r"^Aid to Kenya(, [^(]+)? \((\d+) largest of (\d+) in (\d{4})\)$")


@pytest.mark.parametrize(
    ("type_", "kept"),
    [("bar_chart", 20), ("line_chart", LINE_CHART_MAX_VALUES), ("table", 16)],
)
async def test_make_charts_cuts_a_select_over_the_cap_at_one_period(
    monkeypatch: pytest.MonkeyPatch, type_: str, kept: int
) -> None:
    """Thirty providers are over the cap. All hold 2023 and twenty hold 2024,
    the newest year half of them hold, so every type is cut at 2024 to
    providers holding a 2024 value, as many as the type draws: every one of
    the twenty as bars, eight as lines, and sixteen as a table, whose rows
    over the frame's 125 years must fit 2,000. The title counts the members
    kept, and the model is told what was left out."""
    values = {
        f"donor/{number:02d}": {"2023": 100.0 - number}
        | ({"2024": float(number)} if number % 3 else {})
        for number in range(30)
    }
    view = {**WIDE_VIEW, "coverage": ["1900", "2024"]}

    executor, result = await _draw_wide(monkeypatch, RowFrames(values), view=view, type=type_)

    (intent,) = _intents(executor)
    covered, drawn, holding, period = _CUT_TITLE.match(intent["title"]).groups()
    # A cut line chart or table draws every year its frame covers, and says so.
    assert covered == (None if type_ == "bar_chart" else ", 1900 to 2024")
    members = intent["select"]["financingProvider"]
    assert (period, int(holding)) == ("2024", 20)
    assert len(members) == int(drawn) == kept
    assert all(period in values[member] for member in members)
    assert members == sorted(members, key=lambda member: -values[member][period])
    assert result["charts"][0]["title"] == intent["title"]
    assert result["charts"][0]["note"] == (
        f"This card draws the {kept} largest of the 20 members with a value in 2024. "
        "Left out as group totals, unallocated amounts or members of another level: 0. "
        "Left out with no "
        "value in 2024: 10. Say in the reply that it shows the largest of them in 2024."
    )
    # A bar chart is drawn at the period it was cut at. A line chart or table keeps its window.
    assert intent.get("window") == (
        {"start": "2024", "end": "2024"} if type_ == "bar_chart" else None
    )


# Ten providers. A and B are the largest in 2023, the one year at least half
# of them hold. C and D are the largest in 2024, and E is above D in 2025.
TEN_PROVIDERS: dict[str, dict[str, float]] = {
    "A": {"2023": 100.0, "2024": 1.0},
    "B": {"2023": 90.0, "2024": 2.0},
    "C": {"2023": 30.0, "2024": 80.0},
    "D": {"2023": 20.0, "2024": 70.0, "2025": 50.0},
    "E": {"2023": 10.0, "2025": 60.0},
    **{member: {"2023": float(value)} for value, member in enumerate("FGHIJ", start=1)},
}


class BarPlatform(WideFrameExecutor):
    """Draws a bar chart of `values` as apis binds one. Each selected
    provider's points inside the intent's window, at the latest period every
    one of them holds, written into the catalogue's `bar_chart` payload, whose
    input frame is the one the intent names."""

    def __init__(self, values: dict[str, dict[str, float]]) -> None:
        super().__init__(providers=len(values))
        self.values = values
        self.payloads: list[dict[str, Any]] = []

    async def __call__(self, *, tool: str, args: dict[str, Any], on_progress: Any = None):
        if tool == "write_component":
            intent = args["intent"]
            window = intent.get("window") or {}
            bound = {
                member: {
                    period: value
                    for period, value in self.values[member].items()
                    if window.get("start", period) <= period <= window.get("end", period)
                }
                for member in intent["select"]["financingProvider"]
            }
            period = max(set.intersection(*(set(points) for points in bound.values())))
            payload = resolved_revision(
                "bar_chart", title=intent["title"], frame_ref=intent["frame_ref"]
            )
            template = payload["component"]["series"][0]
            payload["component"]["series"] = [
                {
                    **template,
                    "id": member,
                    "label": f"Provider {member}",
                    "points": [{"period": period, "value": points[period]}],
                }
                for member, points in sorted(bound.items(), key=lambda held: -held[1][period])
            ]
            payload["component"]["category_axis"]["period"] = period
            self.payloads.append(payload)
            self.written = drawn_cards.write_reply(
                payload,
                {
                    member: {
                        "financingProvider": member,
                        "financingRecipient": "country/KEN",
                        "stat_var": "dcid:aid",
                        "unit": "USDollar",
                    }
                    for member in bound
                },
            )
        return await super().__call__(tool=tool, args=args, on_progress=on_progress)


async def _draw_ten(monkeypatch: pytest.MonkeyPatch) -> BarPlatform:
    """Draw `TEN_PROVIDERS` as one bar chart under a series cap of two."""
    for module in (orchestrator_tools, chart_binding):
        monkeypatch.setattr(module, "MAX_SERIES_PER_COMPONENT", 2)
    monkeypatch.setattr(orchestrator_tools, "get_frame_store", lambda: RowFrames(TEN_PROVIDERS))
    executor = BarPlatform(TEN_PROVIDERS)
    tools, ctx = _tools(executor=executor)
    await tools["fetch"](SPEC_LIFE, ctx)
    await tools["make_charts"]([_wide_chart(type="bar_chart")], ctx)
    return executor


async def test_a_cut_cards_figures_cover_every_bar_it_draws(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Thirty providers are cut to 24 bars, of which the platform's summary
    lists twelve. A bar past the twelfth, named as the frame spells it, gives
    its value. The smallest is the smallest of the 24, said to be drawn from
    the 30 with a value."""
    values = {f"donor/{number:02d}": {"2023": 100.0 - number} for number in range(30)}
    monkeypatch.setattr(orchestrator_tools, "get_frame_store", lambda: RowFrames(values))
    executor = BarPlatform(values)
    tools, ctx = _tools(executor=executor)
    await tools["fetch"](SPEC_LIFE, ctx)

    result = await tools["make_charts"](
        [_wide_chart(type="bar_chart", state_values_for=["donor/19", "donor/29"])], ctx
    )

    (card,) = result["charts"]
    assert len(executor.written["fact_summary"]["series"]) == drawn_cards.FACT_SUMMARY_SERIES_CAP
    assert card["holds"]["series_count"] == 24
    assert card["holds"]["cut"] == {
        "drawn": 24,
        "with_a_value": 30,
        "period": "2023",
        "other_kinds": 0,
        "unvalued": 0,
    }
    assert card["holds"]["series_order"] == "highest_first"
    assert _figures(card) == [
        ("Provider donor/19", 81.0, "2023"),
        ("Provider donor/00", 100.0, "2023", "largest"),
        ("Provider donor/23", 77.0, "2023", "smallest"),
    ]
    assert card["reply_figures"][-1]["among"] == {"drawn": 24, "with_a_value": 30}
    assert card["not_in_card"] == ["donor/29"]


async def test_make_charts_pins_a_cut_bar_chart_to_the_period_it_ranked_at(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The two largest of ten are chosen at 2023, the year half of them hold,
    and the card is written with a 2023 window."""
    executor = await _draw_ten(monkeypatch)

    (intent,) = _intents(executor)
    assert intent["select"]["financingProvider"] == ["A", "B"]
    assert intent["window"] == {"start": "2023", "end": "2023"}
    assert intent["title"] == "Aid to Kenya (2 largest of 10 in 2023)"


# `TEN_PROVIDERS` described as the frame apis serves them, over the years they hold.
TEN_VIEW: dict[str, Any] = {**WIDE_VIEW, "coverage": ["2023", "2025"]}


def _ten_select(provider: str | list[str]) -> str:
    return json.dumps({"financingRecipient": "country/KEN", "financingProvider": provider})


@pytest.mark.parametrize(
    ("chart", "view", "title", "window"),
    [
        ({"type": "table"}, TEN_VIEW, "Aid to Kenya, 2023 to 2025", None),
        # C and D share 2024 at the latest, so their bars are drawn there.
        (
            {"type": "bar_chart", "select_json": _ten_select(["C", "D"])},
            TEN_VIEW,
            "Aid to Kenya, 2024",
            {"start": "2024", "end": "2024"},
        ),
        # A select listing no dimension draws one series, a bar for each period.
        (
            {"type": "bar_chart", "select_json": _ten_select("D")},
            TEN_VIEW,
            "Aid to Kenya, 2023 to 2025",
            None,
        ),
        # C's latest value is of 2024, though the frame covers 2025.
        (
            {"type": "stat_card", "select_json": _ten_select("C")},
            TEN_VIEW,
            "Aid to Kenya, 2024",
            None,
        ),
        (
            {"type": "table", "latest_periods": 3},
            TEN_VIEW,
            "Aid to Kenya, last 3 years",
            {"start": {"anchor": "latest", "offset": -2}, "end": {"anchor": "latest", "offset": 0}},
        ),
        # A rolling frame's years move on, so its card names none.
        ({"type": "table"}, {**TEN_VIEW, "rolling": True}, "Aid to Kenya", None),
    ],
    ids=["range", "bar", "period_bars", "stat", "latest_periods", "rolling"],
)
async def test_make_charts_writes_the_period_a_card_shows_into_its_title(
    monkeypatch: pytest.MonkeyPatch,
    chart: dict[str, Any],
    view: dict[str, Any],
    title: str,
    window: dict[str, Any] | None,
) -> None:
    """The model names the subject and the tool writes the period once: a
    table's or a period bar chart's first and last years, the year a bar
    chart of listed providers is drawn and pinned at, the year of a stat
    card's value, and periods counted back."""
    executor, result = await _draw_wide(
        monkeypatch, RowFrames(TEN_PROVIDERS), providers=10, view=view, **chart
    )

    (intent,) = _intents(executor)
    assert (intent["title"], intent.get("window")) == (title, window)
    assert result["charts"][0]["title"] == title


# A whole naming more donors than fit beside the subject in a title of 120 characters.
_LONG_WHOLE = (
    "from the United States, Saint Vincent and the Grenadines, the Democratic Republic of the "
    "Congo and the Central African Republic"
)


@pytest.mark.parametrize(
    ("title", "whole", "drawn"),
    [
        (
            "US share of aid to Kenya",
            "from official donors (DAC)",
            "US share of aid to Kenya, of the total from official donors (DAC), 2023 to 2025",
        ),
        (
            "US share of aid to Kenya",
            "from every donor",
            "US share of aid to Kenya, of the total from every donor, 2023 to 2025",
        ),
        # A title copied from a card of shares states the whole and the period once.
        (
            "US share of aid to Kenya, of the total from every donor, 2023 to 2025",
            "from every donor",
            "US share of aid to Kenya, of the total from every donor, 2023 to 2025",
        ),
        # A whole too long for the title is stated in the card's note.
        ("US share of aid to Kenya", _LONG_WHOLE, "US share of aid to Kenya, 2023 to 2025"),
    ],
    ids=["dac_donors", "all_donors", "copied", "long_whole"],
)
async def test_make_charts_writes_the_whole_a_card_of_shares_divides_by_into_its_title(
    monkeypatch: pytest.MonkeyPatch, title: str, whole: str, drawn: str
) -> None:
    """A frame a plan handed back with the whole its shares divide by
    (`plan_display.share_wholes`) is drawn with that whole in the card's
    title, before the period, or in its note when the title of 120
    characters cannot hold it beside the subject."""
    monkeypatch.setattr(orchestrator_tools, "get_frame_store", lambda: RowFrames(TEN_PROVIDERS))
    executor = WideFrameExecutor(providers=10, view=copy.deepcopy(TEN_VIEW))
    tools, ctx = _tools(executor=executor)
    hand_back = HandBack(executor)
    await hand_back.describe(WIDE_REF, tool="share_of", store_ref=None, whole=whole)
    hand_back.listing(ctx.state, None)

    result = await tools["make_charts"]([_wide_chart(title=title)], ctx)

    (intent,) = _intents(executor)
    (entry,) = result["charts"]
    assert intent["title"] == entry["title"] == drawn
    assert (whole in drawn) != (whole in entry.get("note", ""))


async def test_a_card_whose_frame_is_unread_names_no_period(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stat card's period is its value's, read from the frame's rows, so a
    frame the store does not hold gives none, and no period is guessed."""
    executor, _ = await _draw_wide(
        monkeypatch,
        HeldFrames(held=False),
        providers=10,
        view=TEN_VIEW,
        type="stat_card",
        select_json=_ten_select("C"),
    )

    (intent,) = _intents(executor)
    assert (intent["title"], intent.get("window")) == ("Aid to Kenya", None)


@pytest.mark.parametrize(
    ("title", "named"),
    [
        ("Aid to Kenya (2025)", "2025"),
        ("Aid to Kenya in 2024", "2024"),
        ("Aid to Kenya, 2023 to 2025", "2023"),
    ],
)
async def test_make_charts_refuses_a_title_naming_a_year_of_its_frame(
    monkeypatch: pytest.MonkeyPatch, title: str, named: str
) -> None:
    """Under a cap of two, the two largest of ten are ranked at 2023, the year
    half of them hold. A title naming any year of the frame, 2025 the latest
    included, is refused before a write, so no card states two periods; the
    subject alone is drawn with the cut's period once."""
    for module in (orchestrator_tools, chart_binding):
        monkeypatch.setattr(module, "MAX_SERIES_PER_COMPONENT", 2)
    monkeypatch.setattr(orchestrator_tools, "get_frame_store", lambda: RowFrames(TEN_PROVIDERS))
    executor = BarPlatform(TEN_PROVIDERS)
    executor.view = copy.deepcopy(TEN_VIEW)
    tools, ctx = _tools(executor=executor)
    await tools["fetch"](SPEC_LIFE, ctx)

    refused = await tools["make_charts"]([_wide_chart(type="bar_chart", title=title)], ctx)
    drawn = await tools["make_charts"]([_wide_chart(type="bar_chart")], ctx)

    (entry,) = refused["charts"]
    assert _stated(entry) == ("title_names_period", "correct_and_retry")
    assert entry["fix"] == {"period": named}
    (intent,) = _intents(executor)
    assert (
        intent["title"] == drawn["charts"][0]["title"] == "Aid to Kenya (2 largest of 10 in 2023)"
    )


async def test_make_charts_draws_a_title_naming_a_year_outside_its_frame(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A year the frame does not cover is no period of the card."""
    executor, _ = await _draw_wide(
        monkeypatch, RowFrames(TEN_PROVIDERS), providers=10, view=TEN_VIEW, title="Agenda 2030"
    )

    (intent,) = _intents(executor)
    assert intent["title"] == "Agenda 2030, 2023 to 2025"


async def test_a_cut_bar_chart_draws_every_bar_at_the_period_in_its_title(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A and B share 2024 as well, the period apis draws two bars at when
    no window names another. The payload holds their 2023 values."""
    executor = await _draw_ten(monkeypatch)

    (payload,) = executor.payloads
    bars = {series["id"]: series["points"] for series in payload["component"]["series"]}
    assert bars == {
        "A": [{"period": "2023", "value": 100.0}],
        "B": [{"period": "2023", "value": 90.0}],
    }
    assert payload["component"]["title"]["text"].endswith("in 2023)")


def _kind_reads(executor: FakeExecutor) -> int:
    return sum(
        tool == "get_data" and bool(args.get("place_types")) for tool, args in executor.calls
    )


async def test_make_charts_reads_a_frames_kinds_once_a_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Three over-cap charts of one frame, over two calls, ask the retrieval
    for its members' kinds once."""
    monkeypatch.setattr(orchestrator_tools, "get_frame_store", HeldFrames)
    executor = WideFrameExecutor()
    tools, ctx = _tools(executor=executor)
    await tools["fetch"](SPEC_LIFE, ctx)

    await tools["make_charts"]([_wide_chart(), _wide_chart(type="bar_chart")], ctx)
    await tools["make_charts"]([_wide_chart(type="line_chart")], ctx)

    assert _writes(executor) == 3
    assert _kind_reads(executor) == 1


class KindReadFails(WideFrameExecutor):
    """The retrieval asked again with `place_types` fails as `failure` says,
    until `failure` is None."""

    def __init__(self, failure: str | None) -> None:
        super().__init__()
        self.failure = failure

    async def __call__(self, *, tool: str, args: dict[str, Any], on_progress: Any = None):
        if self.failure is not None and tool == "get_data" and args.get("place_types"):
            self.calls.append((tool, args))
            if self.failure == "refused":
                return {"outcome": "refusal", "refusal": {"code": "service_unavailable"}}
            code = REQUEST_TIMEOUT if self.failure == "timed_out" else CONNECTION_CLOSED
            raise MCPError(code=code, message="get_data did not answer")
        return await super().__call__(tool=tool, args=args, on_progress=on_progress)


@pytest.mark.parametrize("failure", ["timed_out", "closed", "refused"])
async def test_make_charts_refuses_a_cut_whose_kind_read_failed(
    monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    """A select over the cap whose cut cannot learn which members are group
    totals is written as it is and ends with dfo-apis' refusal. A failed read
    is not kept, so the same chart drawn again reads again and is cut."""
    monkeypatch.setattr(orchestrator_tools, "get_frame_store", HeldFrames)
    executor = KindReadFails(failure)
    tools, ctx = _tools(executor=executor)
    await tools["fetch"](SPEC_LIFE, ctx)

    result = await tools["make_charts"]([_wide_chart(), _wide_chart(type="bar_chart")], ctx)

    assert [entry["error"] for entry in result["charts"]] == [
        refusal.detail for refusal in executor.refusals
    ]
    assert [_stated(entry) for entry in result["charts"]] == [
        ("resource_exhausted", "correct_and_retry")
    ] * 2
    assert _writes(executor) == 2
    assert _kind_reads(executor) == 2

    executor.failure = None
    retried = await tools["make_charts"]([_wide_chart()], ctx)

    assert "error" not in retried["charts"][0]
    assert _writes(executor) == 3


@pytest.mark.parametrize(
    "periods",
    [[str(1990 + number) for number in range(30)], ["2015/2019", "2020/2024"] * 15],
    ids=["a_year_each", "two_spans"],
)
async def test_make_charts_refuses_a_cut_whose_members_share_no_period(
    monkeypatch: pytest.MonkeyPatch, periods: list[str]
) -> None:
    """Thirty providers each report a year of their own, so no year is held
    by half of them, or they hold two spans, which have no latest. No period
    is chosen in its place: the select is written as it is and ends with
    dfo-apis' refusal."""
    values = {f"donor/{number:02d}": {periods[number]: 1.0} for number in range(30)}

    executor, result = await _draw_wide(monkeypatch, RowFrames(values), type="bar_chart")

    assert result["charts"][0] == _relayed(executor.refusals[-1])
    assert _writes(executor) == 1


@pytest.mark.parametrize(
    ("type_", "providers", "window"),
    [
        ("bar_chart", 30, {"start": "2020/2024", "end": "2020/2024"}),
        ("table", 30, None),
        # Twenty bars are under the cap: nothing is cut and no period is pinned.
        ("bar_chart", 20, None),
    ],
)
async def test_make_charts_cuts_a_frame_of_one_span_at_that_span(
    monkeypatch: pytest.MonkeyPatch, type_: str, providers: int, window: dict[str, str] | None
) -> None:
    """Thirty providers each hold the one period 2020/2024, so the largest
    are chosen at it and the title names it. A cut bar chart is written with
    a window of that period, whose title and members are of it, though the
    frame holds no other. A cut table and a bar chart under the cap carry
    none."""
    values = {f"donor/{number:02d}": {"2020/2024": float(number)} for number in range(providers)}
    view = {**WIDE_VIEW, "coverage": ["2020/2024", "2020/2024"]}

    executor, _ = await _draw_wide(
        monkeypatch, RowFrames(values), providers=providers, view=view, type=type_
    )

    (intent,) = _intents(executor)
    if providers > MAX_SERIES_PER_COMPONENT:
        assert intent["title"] == "Aid to Kenya (24 largest of 30 in 2020/2024)"
        assert intent["select"]["financingProvider"] == [f"donor/{n:02d}" for n in range(29, 5, -1)]
    else:
        assert (intent["title"], intent["select"]) == (
            "Aid to Kenya, 2020 to 2024",
            {"financingRecipient": "country/KEN"},
        )
    assert intent.get("window") == window


async def test_make_charts_holds_a_frames_rows_until_its_ranking_thread_returns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The ranking runs in a worker thread. Cancelling the turn while it runs
    leaves the frame's rows claimed, and they are given back once the thread
    returns."""
    budget = RowBudget(max_rows=1_000)
    monkeypatch.setattr(orchestrator_tools, "row_budget", lambda: budget)
    monkeypatch.setattr(orchestrator_tools, "get_frame_store", lambda: HeldFrames(carried=True))
    ranking, finish = threading.Event(), threading.Event()
    threads: list[threading.Thread] = []

    def held_ranking(**_: Any) -> None:
        threads.append(threading.current_thread())
        ranking.set()
        finish.wait(timeout=5)

    monkeypatch.setattr(orchestrator_tools, "top_members", held_ranking)
    tools, ctx = _tools(executor=WideFrameExecutor())
    await tools["fetch"](SPEC_LIFE, ctx)
    try:
        drawing = asyncio.ensure_future(tools["make_charts"]([_wide_chart()], ctx))
        await asyncio.to_thread(ranking.wait, 5)
        assert threads and threads[0] is not threading.main_thread()
        assert budget.in_flight == 60

        drawing.cancel()
        with pytest.raises(asyncio.CancelledError):
            await drawing
        assert budget.in_flight == 60
    finally:
        finish.set()
    async with asyncio.timeout(5):
        while budget.in_flight:
            await asyncio.sleep(0.01)


@pytest.mark.parametrize(
    ("role", "carried"),
    [("financingProvider", False), ("financingProvider", True), ("observationAbout", True)],
    ids=["kinds_asked_again", "kinds_carried", "place_types_carried"],
)
async def test_make_charts_ranks_parties_only_over_the_cap(
    monkeypatch: pytest.MonkeyPatch, role: str, carried: bool
) -> None:
    """Group totals (the largest values here) are left out of the top N, and
    out of its count, by the kind apis serves for the same retrieval; the
    chart is drawn on the frame the model chose. A frame whose rows carry the
    kind or type column is ranked from those rows, with no second retrieval.
    That means parties among totals, or countries typed `Country` and
    `Country,Place` among places that are not countries."""
    totals = frozenset(f"donor/{number:02d}" for number in range(25, 30))
    view = copy.deepcopy(WIDE_VIEW)
    view["dimensions"][1]["name"] = role
    store = HeldFrames(totals=totals, role=role, carried=carried)

    executor, result = await _draw_wide(monkeypatch, store, view=view, type="table")

    (intent,) = _intents(executor)
    assert intent["frame_ref"] == WIDE_REF
    assert intent["select"][role] == [f"donor/{number:02d}" for number in range(24, 0, -1)]
    assert intent["title"] == "Aid to Kenya, 2023 to 2024 (24 largest of 25 in 2024)"
    assert "another level: 5." in result["charts"][0]["note"]
    assert _kind_reads(executor) == (0 if carried else 1)


@pytest.mark.parametrize(
    ("spec_id", "stored", "asked"),
    [
        (SPEC_AID, {}, {"spec_id": SPEC_AID, "place_types": True}),
        (
            SPEC_AID,
            {"store_ref": {"spec_id": SPEC_AID, "content_version": None}},
            {"spec_id": SPEC_AID, "place_types": True},
        ),
        ("ops_" + "e" * 32, {}, None),
    ],
    ids=["plan_step_frame", "store_pair_unstated", "analysis_result"],
)
async def test_make_charts_asks_a_frame_with_no_fetch_for_its_kinds_by_its_spec(
    monkeypatch: pytest.MonkeyPatch,
    spec_id: str,
    stored: dict[str, Any],
    asked: dict[str, Any] | None,
) -> None:
    """A frame a plan step registered carries no fetch arguments: its kinds are
    read from its spec alone. An analysis result, which no retrieval serves, is
    not narrowed. An entry that states no store pair, or one with a member
    missing, is read from the store by its own pair."""
    monkeypatch.setattr(orchestrator_tools, "get_frame_store", lambda: HeldFrames(wide=WIDE_REF))
    executor = WideFrameExecutor()
    tools, ctx = _tools(executor=executor)
    ref = {**WIDE_REF, "spec_id": spec_id}
    ctx.state[ORCHESTRATOR_FRAMES_KEY] = {
        "fplan": {"ref": ref, "view": copy.deepcopy(WIDE_VIEW), **stored}
    }
    select = json.dumps({"financingRecipient": "country/KEN"})

    result = await tools["make_charts"](
        [{"frame": "fplan", "type": "table", "title": "t", "select_json": select}], ctx
    )

    calls = [args for tool, args in executor.calls if tool == "get_data"]
    if asked is None:
        assert calls == []
        assert "error" in result["charts"][0]
    else:
        assert calls == [asked]
        assert result["charts"][0]["title"] == "t, 2023 to 2024 (24 largest of 30 in 2024)"


async def test_make_charts_cut_title_carries_one_suffix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A chart titled with the title of a card already cut, as the model copies
    it from the first card's result, ends in one suffix."""
    cut = "Aid to Kenya, 2023 to 2024 (24 largest of 30 in 2024)"

    executor, _ = await _draw_wide(monkeypatch, HeldFrames(), type="table", title=cut)

    (intent,) = _intents(executor)
    assert intent["title"] == cut


class Landing:
    """`executor`, each card it writes given an id of its own and landed on
    `state`'s canvas with the intent it was written from, as the Studio
    translator lands it. `describe_component` reads a card back as
    presentation-mcp gives its head revision, or answers `described` when
    set."""

    def __init__(self, executor: FakeExecutor, state: ConversationState) -> None:
        self.executor = executor
        self.state = state
        self.calls = executor.calls
        self.cards: dict[str, dict[str, Any]] = {}
        self.described: dict[str, Any] | Exception | None = None

    async def __call__(self, *, tool: str, args: dict[str, Any], on_progress: Any = None):
        if tool == "describe_component":
            self.calls.append((tool, args))
            if isinstance(self.described, Exception):
                raise self.described
            card = self.cards[args["component_id"]]
            return self.described or {
                "outcome": "ok",
                "frame_refs": [],
                "component_body": {
                    "component_id": card["component_id"],
                    "revision": card["revision"],
                    "fact_summary": card["fact_summary"],
                },
                "resolved_payload": card["payload"],
            }
        written = await self.executor(tool=tool, args=args, on_progress=on_progress)
        if tool != "write_component":
            return written
        component_id = f"cmp_{len(_intents(self.executor)):02d}"
        component = written["component"] | {"component_id": component_id}
        component["payload"] = component["payload"] | {"component_id": component_id}
        self.cards[component_id] = component
        return written | {"component": component}

    def land_component(self, component: dict[str, Any]) -> None:
        land_component(self.state, component, intent=_intents(self.executor)[-1])


def _landing_tools(executor: FakeExecutor) -> tuple[Landing, dict[str, Any], SimpleNamespace]:
    """The tools over `executor`, its cards landing on one conversation's
    canvas, which the tools read through the Studio interfaces."""
    state = ConversationState(conversation_id="c1")
    landing = Landing(executor, state)
    tools, ctx = _tools(
        executor=landing,
        canvas_series=state.canvas_series_ids,
        canvas_cards=state.canvas_cards,
        remove_cards=_remove_callable(state),
    )
    return landing, tools, ctx


async def _draw_wide_twice(
    monkeypatch: pytest.MonkeyPatch,
    first: dict[str, Any],
    second: dict[str, Any],
    *,
    later_turn: bool,
) -> tuple[Landing, list[dict[str, Any]]]:
    """Fetch the wide frame and draw `first` then `second`, in two calls of
    one turn or, when `later_turn`, the second in the next turn."""
    monkeypatch.setattr(orchestrator_tools, "get_frame_store", HeldFrames)
    executor, tools, ctx = _landing_tools(WideFrameExecutor())
    await tools["fetch"](SPEC_LIFE, ctx)
    results = [await tools["make_charts"]([_wide_chart(**first)], ctx)]
    if later_turn:
        ctx.invocation_id = "inv_2"
    results.append(await tools["make_charts"]([_wide_chart(**second)], ctx))
    return executor, results


_TURNS = pytest.mark.parametrize("later_turn", [False, True], ids=["same_turn", "later_turn"])


@_TURNS
@pytest.mark.parametrize("asked", [{}, {"state_values_for": ["donor/03"], "state_span": True}])
async def test_make_charts_does_not_draw_a_card_the_canvas_holds(
    monkeypatch: pytest.MonkeyPatch, later_turn: bool, asked: dict[str, Any]
) -> None:
    """The same title, kind, frame, select and cut drawn again, in the same
    turn or a later one and whatever figures it asks for, writes no second
    component and returns the first card with a notice."""
    executor, (first, again) = await _draw_wide_twice(
        monkeypatch, {"type": "table"}, {"type": "table", **asked}, later_turn=later_turn
    )

    assert len(_intents(executor)) == 1
    (card,) = again["charts"]
    assert card["title"] == first["charts"][0]["title"]
    assert "already_on_canvas" in card


@_TURNS
async def test_make_charts_reads_the_figures_asked_of_a_card_on_the_canvas(
    monkeypatch: pytest.MonkeyPatch, later_turn: bool
) -> None:
    """A bar chart drawn again with a bar named reads that bar's value from
    the series the card drew, past the twelve the platform's summary lists,
    and writes nothing. In the same turn the largest and smallest come too;
    a later turn began with the card's series on the canvas, so it states
    only the bar named."""
    values = {f"donor/{number:02d}": {"2023": 100.0 - number} for number in range(30)}
    monkeypatch.setattr(orchestrator_tools, "get_frame_store", lambda: RowFrames(values))
    executor, tools, ctx = _landing_tools(BarPlatform(values))
    await tools["fetch"](SPEC_LIFE, ctx)
    await tools["make_charts"]([_wide_chart(type="bar_chart")], ctx)
    if later_turn:
        ctx.invocation_id = "inv_2"

    again = await tools["make_charts"](
        [_wide_chart(type="bar_chart", state_values_for=["donor/19"])], ctx
    )

    (card,) = again["charts"]
    assert len(_intents(executor)) == 1
    assert "already_on_canvas" in card
    assert [args for tool, args in executor.calls if tool == "describe_component"] == [
        {"component_id": "cmp_01"}
    ]
    # The turn keeps the card's id alone, none of its points.
    assert [set(card) for card in ctx.state["orchestrator:drawn_cards"]["drawn"].values()] == [
        {"component_id", "revision"}
    ]
    assert card["holds"]["series_count"] == 24
    assert _figures(card) == [("Provider donor/19", 81.0, "2023")] + (
        []
        if later_turn
        else [
            ("Provider donor/00", 100.0, "2023", "largest"),
            ("Provider donor/23", 77.0, "2023", "smallest"),
        ]
    )


@_TURNS
@pytest.mark.parametrize(
    "described",
    [
        {"outcome": "refusal", "refusal": {"code": "not_found", "message": "gone"}},
        {"outcome": "ok", "frame_refs": [], "component_body": {}, "resolved_payload": None},
        TimeoutError("describe_component timed out"),
    ],
    ids=["refused", "no_resolved_payload", "raised"],
)
async def test_make_charts_refuses_a_card_on_the_canvas_it_cannot_read_back(
    monkeypatch: pytest.MonkeyPatch, later_turn: bool, described: dict[str, Any] | Exception
) -> None:
    """A card on the canvas whose head revision cannot be read is refused
    under `figures_unread`, with no figure and no second copy written, and
    the call spends the failed budget."""
    monkeypatch.setattr(orchestrator_tools, "get_frame_store", HeldFrames)
    executor, tools, ctx = _landing_tools(WideFrameExecutor())
    await tools["fetch"](SPEC_LIFE, ctx)
    await tools["make_charts"]([_wide_chart()], ctx)
    if later_turn:
        ctx.invocation_id = "inv_2"
    executor.described = described

    again = await tools["make_charts"]([_wide_chart()], ctx)

    (card,) = again["charts"]
    assert len(_intents(executor)) == 1
    assert "reply_figures" not in card
    assert _stated(card) == (HarnessCode.FIGURES_UNREAD, "do_not_retry")
    calls = ctx.state["orchestrator:calls"]
    assert (calls["make_charts"], calls["make_charts_failed"]) == (1, 1)


async def test_make_charts_counts_a_call_that_read_a_card_as_one_that_drew(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Rereads give figures, so they spend the drawn budget, not the failed
    one: after three of them a fifth call still draws."""
    monkeypatch.setattr(orchestrator_tools, "get_frame_store", HeldFrames)
    executor, tools, ctx = _landing_tools(WideFrameExecutor())
    await tools["fetch"](SPEC_LIFE, ctx)
    await tools["make_charts"]([_wide_chart()], ctx)

    for _ in range(3):
        assert (
            "already_on_canvas" in (await tools["make_charts"]([_wide_chart()], ctx))["charts"][0]
        )
    drawn = await tools["make_charts"]([_wide_chart(type="bar_chart")], ctx)

    assert "error" not in drawn["charts"][0]
    assert [intent["type"] for intent in _intents(executor)] == ["table", "bar_chart"]
    calls = ctx.state["orchestrator:calls"]
    assert (calls["make_charts"], calls.get("make_charts_failed", 0)) == (5, 0)


@_TURNS
@pytest.mark.parametrize("all_cards", [False, True], ids=["named", "all_cards"])
async def test_make_charts_draws_again_a_card_remove_cards_took_off(
    monkeypatch: pytest.MonkeyPatch, later_turn: bool, all_cards: bool
) -> None:
    """A card taken off the canvas, by its id or with every card, is drawn
    again by a later call of the turn or of a later turn, and the card left
    on is not."""
    monkeypatch.setattr(orchestrator_tools, "get_frame_store", HeldFrames)
    executor, tools, ctx = _landing_tools(WideFrameExecutor())
    await tools["fetch"](SPEC_LIFE, ctx)
    table, bars = _wide_chart(), _wide_chart(type="bar_chart")
    await tools["make_charts"]([table, bars], ctx)

    removed = await tools["remove_cards"](ctx, ["cmp_01"], all_cards=all_cards)
    if later_turn:
        ctx.invocation_id = "inv_2"
    again = await tools["make_charts"]([table, bars], ctx)

    assert removed["removed"] == (["cmp_01", "cmp_02"] if all_cards else ["cmp_01"])
    assert [intent["type"] for intent in _intents(executor)] == (
        ["table", "bar_chart", "table", "bar_chart"]
        if all_cards
        else ["table", "bar_chart", "table"]
    )
    assert ["already_on_canvas" in card for card in again["charts"]] == [False, not all_cards]
    assert executor.state.composition is not None
    assert len(executor.state.composition.blocks) == 2


async def test_make_charts_reads_a_card_the_user_added_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A card the model took off and the user added back unchanged is the
    same card, so a later turn's ask for it reads it and writes nothing."""
    monkeypatch.setattr(orchestrator_tools, "get_frame_store", HeldFrames)
    executor, tools, ctx = _landing_tools(WideFrameExecutor())
    await tools["fetch"](SPEC_LIFE, ctx)
    await tools["make_charts"]([_wide_chart()], ctx)
    await tools["remove_cards"](ctx, ["cmp_01"])
    state = executor.state
    applied = _apply_operations(
        state, state.composition, [AddOp(op=MutationOp.ADD, component_id="cmp_01")]
    )
    assert isinstance(applied, tuple)
    state.composition = _next_manifest(state.composition, applied[0], actor=CompositionActor.USER)
    ctx.invocation_id = "inv_2"

    again = await tools["make_charts"]([_wide_chart()], ctx)

    assert len(_intents(executor)) == 1
    assert "already_on_canvas" in again["charts"][0]


@_TURNS
async def test_make_charts_draws_again_a_card_revised_since_it_was_drawn(
    monkeypatch: pytest.MonkeyPatch, later_turn: bool
) -> None:
    """A card the user revised through Edit Card no longer draws the intent
    it was written from, so asking for that intent draws it anew."""
    monkeypatch.setattr(orchestrator_tools, "get_frame_store", HeldFrames)
    executor, tools, ctx = _landing_tools(WideFrameExecutor())
    await tools["fetch"](SPEC_LIFE, ctx)
    await tools["make_charts"]([_wide_chart()], ctx)
    revised = executor.cards["cmp_01"] | {"revision": executor.cards["cmp_01"]["revision"] + 1}
    patch = {"title": "Renamed"}
    _revised_frames(
        executor.state,
        component_id="cmp_01",
        component=revised,
        patch=patch,
        revised=component_revised(revised, "cmp_01", patch),
    )
    executor.cards["cmp_01"] = revised
    if later_turn:
        ctx.invocation_id = "inv_2"

    again = await tools["make_charts"]([_wide_chart()], ctx)

    assert len(_intents(executor)) == 2
    assert "already_on_canvas" not in again["charts"][0]


@_TURNS
async def test_make_charts_draws_a_card_of_another_kind_again(
    monkeypatch: pytest.MonkeyPatch, later_turn: bool
) -> None:
    """A second chart of the same frame and select in another kind is drawn."""
    executor, (_, second) = await _draw_wide_twice(
        monkeypatch, {"type": "table"}, {"type": "bar_chart"}, later_turn=later_turn
    )

    assert len(_intents(executor)) == 2
    assert "already_on_canvas" not in second["charts"][0]


async def test_make_charts_counts_only_the_members_the_chart_lists(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A select listing ten providers draws the largest of those ten that are
    parties, not of every provider the retrieval holds."""
    listed = [f"donor/{number:02d}" for number in range(10)]
    select = json.dumps({"financingRecipient": "country/KEN", "financingProvider": listed})
    totals = frozenset({"donor/09"})

    executor, _ = await _draw_wide(
        monkeypatch, HeldFrames(totals=totals), type="line_chart", select_json=select
    )

    (intent,) = _intents(executor)
    assert intent["select"]["financingProvider"] == [f"donor/{n:02d}" for n in range(8, 0, -1)]
    assert intent["title"] == "Aid to Kenya, 2023 to 2024 (8 largest of 9 in 2024)"


async def test_make_charts_keeps_the_cap_refusal_when_no_member_is_a_party(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    every = frozenset(f"donor/{number:02d}" for number in range(30))

    executor, result = await _draw_wide(monkeypatch, HeldFrames(totals=every))

    assert result["charts"][0] == _relayed(executor.refusals[-1])
    assert _writes(executor) == 1


@pytest.mark.parametrize("type_", ["table", "line_chart", "bar_chart"])
async def test_make_charts_writes_no_order_for_order_by_value(
    monkeypatch: pytest.MonkeyPatch, type_: str
) -> None:
    """The platform draws every chart's members in its own order, so
    `order_by_value` writes the select as the model listed it and reads no
    frame to rank it."""
    select = {
        "financingRecipient": "country/KEN",
        "financingProvider": ["donor/00", "donor/03", "donor/01"],
    }
    store = CountedFrames(providers=5, carried=True)

    executor, _ = await _draw_wide(
        monkeypatch,
        store,
        providers=5,
        type=type_,
        order_by_value=True,
        select_json=json.dumps(select),
    )

    (intent,) = _intents(executor)
    assert intent["select"] == select
    # Only a bar chart reads the frame, to pin the period its bars draw.
    assert store.opens == (1 if type_ == "bar_chart" else 0)


async def test_make_charts_keeps_a_table_under_the_platforms_row_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A table draws series times periods rows, at most 2,000, counting every
    period of the frame's window, gaps included: over 1925-2024 that is 20
    series, though the rows hold two years."""

    class LongFrames(HeldFrames):
        def rows(self) -> list[dict[str, Any]]:
            return [
                {
                    "stat_var": "dcid:aid",
                    "financingProvider": f"donor/{number:02d}",
                    "financingRecipient": "country/KEN",
                    "time_period": year,
                    "value": float(number),
                    "unit": "USDollar",
                }
                for number in range(30)
                for year in ("1925", "2024")
            ]

    view = {**WIDE_VIEW, "coverage": ["1925", "2024"]}
    executor, _ = await _draw_wide(monkeypatch, LongFrames(), view=view)

    (intent,) = _intents(executor)
    assert len(intent["select"]["financingProvider"]) == 20
    assert intent["title"] == "Aid to Kenya, 1925 to 2024 (20 largest of 30 in 2024)"


@pytest.mark.parametrize(
    ("coverage", "periods"),
    [
        (["1925", "2024"], 100),
        (["2015-01", "2024-12"], 120),
        (["2024-01-01", "2024-03-31"], 93),
        (["2020/2024", "2020/2024"], 1),
    ],
)
def test_periods_covered_counts_every_period_of_the_window_never_too_few(
    coverage: list[str], periods: int
) -> None:
    """Days count up to 31 a month: January to March is 91 days, counted 93."""
    assert chart_binding.periods_covered({"coverage": coverage}) == periods


class CountedFrames(HeldFrames):
    """`HeldFrames`, counting the frames it `opens`."""

    def __init__(self, **held: Any) -> None:
        super().__init__(**held)
        self.opens = 0

    async def open(self, ref: Any) -> SimpleNamespace | None:
        self.opens += 1
        return await super().open(ref)


@pytest.mark.parametrize(
    "recipient", ["country/KEN", ["country/KEN"], ["country/KEN", "country/KEN"]]
)
async def test_make_charts_cuts_a_line_chart_to_the_lines_it_lists(
    monkeypatch: pytest.MonkeyPatch, recipient: str | list[str]
) -> None:
    """Thirty providers are cut to the largest eight, which the one write
    lists, a one-value or repeated list beside them written as its value.
    The rows, which carry each provider's kind, are read once."""
    store = CountedFrames(providers=30, carried=True)
    executor, result = await _draw_wide(
        monkeypatch,
        store,
        providers=30,
        type="line_chart",
        select_json=json.dumps({"financingRecipient": recipient}),
    )

    (intent,) = _intents(executor)
    assert "error" not in result["charts"][0]
    assert intent["select"]["financingRecipient"] == "country/KEN"
    assert store.opens == 1
    assert intent["select"]["financingProvider"] == [
        f"donor/{number:02d}" for number in range(29, 21, -1)
    ]
    assert "(8 largest of 30 in " in intent["title"]


class KindlessFrames(HeldFrames):
    """`HeldFrames`, whose kind read serves a frame the store cannot open."""

    async def open(self, ref: Any) -> SimpleNamespace | None:
        if ref.content_version == KINDS_STORE_REF["content_version"]:
            return None
        return await super().open(ref)


class ScatteredFrames(HeldFrames):
    """`HeldFrames`, each provider holding one year of its own, so no year is
    held by half of them."""

    def rows(self) -> list[dict[str, Any]]:
        return [
            {**row, "time_period": str(2000 + number)}
            for number, row in enumerate(super().rows()[1::2])
        ]


async def test_make_charts_cuts_a_refused_table_once_from_one_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A table of twenty providers is written as given and dfo-apis refuses
    its 4,800 rows. The cut to the largest eight is written once more, and
    the frame is read once."""
    store = CountedFrames(providers=20, carried=True)
    executor, result = await _draw_capped(
        monkeypatch,
        refused="table",
        refusal=_TABLE_ROWS_REFUSAL,
        view=_MONTHLY_VIEW,
        store=store,
    )

    whole, cut = _intents(executor)
    assert "financingProvider" not in whole["select"]
    assert cut["select"]["financingProvider"] == [
        f"donor/{number:02d}" for number in range(19, 11, -1)
    ]
    assert "error" not in result["charts"][0]
    assert store.opens == 1


class UnreadableFrames:
    async def open(self, ref: Any) -> None:
        raise FrameTooLarge(ref, reason="over the read ceiling")


class PairFrames(HeldFrames):
    """The store, holding aid from each provider to each recipient `pairs`
    names, in 2023 and 2024."""

    def __init__(self, pairs: list[tuple[int, str]], **held: Any) -> None:
        super().__init__(**held)
        self.pairs = pairs

    def rows(self) -> list[dict[str, Any]]:
        return [
            {
                "stat_var": "dcid:aid",
                "financingProvider": f"donor/{number:02d}",
                "financingRecipient": recipient,
                "time_period": period,
                "value": float(number + 1),
                "unit": "USDollar",
            }
            for number, recipient in self.pairs
            for period in ("2023", "2024")
        ]


# Six providers giving to Kenya, and twelve to four other recipients.
_TO_KENYA = [(number, "country/KEN") for number in range(6)] + [
    (number, ["country/ETH", "country/GHA", "country/MOZ", "country/UGA"][number % 4])
    for number in range(12)
]
# Three providers and three recipients holding five of their nine pairs.
_THREE_BY_THREE = [
    (0, "country/KEN"),
    (0, "country/ETH"),
    (1, "country/KEN"),
    (2, "country/ETH"),
    (2, "country/GHA"),
]


def _pair_view(*, providers: int, recipients: int) -> dict[str, Any]:
    """`WIDE_VIEW` counting `providers` and `recipients`, listing its one measure."""
    view = copy.deepcopy(WIDE_VIEW)
    view["dimensions"][0]["values"] = ["dcid:aid"]
    view["dimensions"][1]["total_count"] = providers
    view["dimensions"][2]["total_count"] = recipients
    return view


@pytest.mark.parametrize(
    ("pairs", "providers", "recipients", "select", "written"),
    [
        pytest.param(
            _TO_KENYA,
            12,
            5,
            {"financingRecipient": "country/KEN"},
            {"financingRecipient": "country/KEN"},
            id="kenya-has-six-donors",
        ),
        pytest.param(
            _THREE_BY_THREE,
            3,
            3,
            {},
            {"stat_var": "dcid:aid"},
            id="empty-select-on-three-by-three",
        ),
    ],
)
async def test_make_charts_writes_a_sparse_line_chart_as_it_is(
    monkeypatch: pytest.MonkeyPatch,
    pairs: list[tuple[int, str]],
    providers: int,
    recipients: int,
    select: dict[str, Any],
    written: dict[str, Any],
) -> None:
    """The view's counts put each select at 12 or 9 series, and its rows
    resolve to six or five. dfo-apis counts the series, so the select is
    written once as it is, an empty one bound to the frame's one measure, and
    drawn whole when the platform accepts it."""
    executor, result = await _draw_wide(
        monkeypatch,
        PairFrames(pairs),
        providers=providers,
        view=_pair_view(providers=providers, recipients=recipients),
        type="line_chart",
        select_json=json.dumps(select),
    )

    (intent,) = _intents(executor)
    assert intent["select"] == written
    assert "largest of" not in intent["title"]
    assert "error" not in result["charts"][0]


class UncountedFrames:
    async def open(self, ref: Any) -> SimpleNamespace:
        return SimpleNamespace(num_rows=None)


class FailingStore:
    async def open(self, ref: Any) -> None:
        raise TimeoutError("the bucket did not answer")


SIX_RECIPIENTS = [f"country/R{number}" for number in range(6)]


class SixSeriesExecutor(FakeExecutor):
    """Describes a frame of six series, one for each recipient, as apis
    lists a dimension, its values with the empty cells dropped. Five series
    are the United States' and the sixth, a total, has no provider."""

    async def __call__(self, *, tool: str, args: dict[str, Any], on_progress: Any = None):
        if tool == "get_data":
            fetched = await super().__call__(tool=tool, args=args, on_progress=on_progress)
            return {**fetched, "store_refs": [WIDE_STORE_REF]}
        if tool != "describe_frame":
            return await super().__call__(tool=tool, args=args, on_progress=on_progress)
        dimensions = [
            {"name": "financingRecipient", "values": SIX_RECIPIENTS, "total_count": 6},
            {"name": "financingProvider", "values": ["country/USA"], "total_count": 1},
        ]
        return {"outcome": "ok", "view": {"dimensions": dimensions}}


class SixSeriesFrames:
    """The store's frame of `SixSeriesExecutor`, as `Frame.from_table` reads a
    stored table (`Frame.build` refuses an empty role cell). The last
    recipient's row has no provider, and with `unplaced` the first has no
    recipient either."""

    def __init__(self, *, unplaced: bool = False) -> None:
        self.empty = {"financingProvider": [5], "financingRecipient": [0] if unplaced else []}

    async def open(self, ref: Any) -> SimpleNamespace:
        rows = [
            {
                "stat_var": "dcid:aid",
                "financingProvider": "country/USA",
                "financingRecipient": recipient,
                "time_period": "2024",
                "value": float(number),
                "unit": "USDollar",
            }
            for number, recipient in enumerate(SIX_RECIPIENTS)
        ]
        built = Frame.build(role_keys=list(self.empty), rows=rows)
        table = built.table
        for role, numbers in self.empty.items():
            cells = [
                None if number in numbers else cell
                for number, cell in enumerate(table.column(role).to_pylist())
            ]
            index = table.schema.get_field_index(role)
            field = table.schema.field(index)
            table = table.set_column(index, field, pa.array(cells, type=field.type))
        frame = Frame.from_table(columns=built.columns, table=table)

        async def read(*, max_rows: int) -> tuple[Frame, Any]:
            return frame, None

        return SimpleNamespace(num_rows=frame.num_rows, read=read)


@pytest.mark.parametrize(
    ("executor", "store", "select", "unread"),
    [
        # The one-valued provider leaves the total out, so the recipients are bound.
        (SixSeriesExecutor, SixSeriesFrames(), {"financingRecipient": SIX_RECIPIENTS}, False),
        # No dimension holds a value on every series.
        (SixSeriesExecutor, SixSeriesFrames(unplaced=True), None, False),
        # Rows that cannot be read confirm no dimension, and the refusal says so.
        (SixSeriesExecutor, UnreadableFrames(), None, True),
        (SixSeriesExecutor, FailingStore(), None, True),
        # The view lists two of three places.
        (CappedViewExecutor, SixSeriesFrames(), None, False),
    ],
)
async def test_make_charts_binds_an_empty_select_only_to_a_dimension_naming_every_series(
    monkeypatch: pytest.MonkeyPatch,
    executor: Callable[[], FakeExecutor],
    store: Any,
    select: dict[str, Any] | None,
    unread: bool,
) -> None:
    """The platform refuses an empty select. One on a frame under the series
    cap is written as the first dimension the view lists every value of and
    every row of the frame holds a value on, and refused before any write
    where there is none. The refusal of a frame that could not be read gives
    that as its reason."""
    monkeypatch.setattr(orchestrator_tools, "get_frame_store", lambda: store)
    platform = executor()
    tools, ctx = _tools(executor=platform)
    await tools["fetch"](SPEC_LIFE, ctx)
    chart = {"frame": frame_id_for(REF), "type": "table", "title": "t", "select_json": "{}"}

    (drawn,) = (await tools["make_charts"]([chart], ctx))["charts"]

    assert [intent["select"] for intent in _intents(platform)] == ([select] if select else [])
    if select is None:
        assert _stated(drawn) == (HarnessCode.SELECT_INVALID, "correct_and_retry")
        view = ctx.state["orchestrator:frames"][chart["frame"]]["view"]
        reasons = {
            cause: empty_select_refusal(type_="table", view=view, unread=cause)
            for cause in (True, False)
        }
        assert reasons[True]["message"] != reasons[False]["message"]
        assert (drawn["error"], drawn["fix"]) == (
            reasons[unread]["message"],
            reasons[unread]["fix"],
        )
        assert drawn["fix"]["dimensions"]
    else:
        assert "error" not in drawn


class RollingPlatform(FakeExecutor):
    """Answers around a frame as `harness.mcp_client.McpExecutor` does: a view
    of thirty reporters marked `rolling` when `cards` says the frame rolls,
    and every write passed through `RollingCards.admit`."""

    def __init__(self, cards: RollingCards, ref: dict[str, Any]) -> None:
        super().__init__()
        self.cards = cards
        self.ref = ref
        self.admitted: list[dict[str, Any]] = []

    async def __call__(self, *, tool: str, args: dict[str, Any], on_progress: Any = None):
        self.calls.append((tool, args))
        if tool == "get_data":
            return {"outcome": "ok", "frame_refs": [self.ref], "summary": "30 observations"}
        if tool == "describe_frame":
            reporters = [f"country/R{number:02d}" for number in range(30)]
            view: dict[str, Any] = {
                "dimensions": [{"name": "reporter", "values": reporters[:20], "total_count": 30}]
            }
            if await self.cards.frame(args["frame_ref"]):
                view["rolling"] = True
            return {"outcome": "ok", "view": view}
        platform = rolling_scene.Platform()
        try:
            admitted = await self.cards.admit(
                args,
                read_view=platform.view,
                read_stored=platform.stored,
                read_window=platform.window,
            )
        except CardWithheld as withheld:
            return {"outcome": "refusal", "refusal": withheld.refusal}
        self.admitted.append(admitted.args["intent"])
        return {"outcome": "ok", "component": self.written}


async def test_a_chart_over_the_cap_on_a_rolling_frame_is_refused_and_nothing_is_cut() -> None:
    """A frame the rolling-card rules call rolling is described as rolling,
    so a select over the cap on it is not cut: its empty select names no
    series the view lists in full, and no write reaches those rules. They would refuse the title a cut writes, for
    naming the frame's latest period, and would admit a window of that period
    under a title that leaves it out."""
    async with rolling_scene.one_reporter_ahead() as scene:
        ref, recipe = await scene.rolling()
        cards = RollingCards(replay=rolling_scene.Replays(scene))
        platform = RollingPlatform(cards, ref.model_dump())
        tools, ctx = _tools(executor=platform)
        await tools["fetch"](SPEC_LIFE, ctx)
        chart = {"frame": frame_id_for(ref.model_dump()), "type": "bar_chart", "title": "Growth"}

        (refused,) = (await tools["make_charts"]([chart | {"select_json": "{}"}], ctx))["charts"]

        period = recipe["anchor"]["period"]
        every = {"reporter": list(rolling_scene.REPORTERS)}
        cut = {"type": "bar_chart", "frame_ref": ref.model_dump(), "select": every}
        titled = await platform(
            tool="write_component",
            args={"intent": cut | {"title": f"Growth (3 largest of 3 in {period})"}},
        )
        pinned = await platform(
            tool="write_component",
            args={"intent": cut | {"title": "Growth", "window": {"start": period, "end": period}}},
        )

    assert _stated(refused) == (HarnessCode.SELECT_INVALID, "correct_and_retry")
    assert _writes(platform) == 2
    assert titled["refusal"]["code"] == HarnessCode.TITLE_QUOTES_ROLLING_PERIOD
    assert pinned["outcome"] == "ok"
    assert platform.admitted == [
        cut | {"title": "Growth", "window": {"start": period, "end": period}}
    ]


def _rank_positions() -> HeldFrames:
    """The store, holding a plan's per-period rank: its values are positions."""
    trace = SimpleNamespace(applied_policy=SimpleNamespace(detail={"derived": {"ordinal": True}}))
    return HeldFrames(trace=trace)


class TwoUnitFrames(HeldFrames):
    """The store, holding the frame with every other provider in euros."""

    def rows(self) -> list[dict[str, Any]]:
        return [
            {**row, "unit": "EUR"} if int(row[self.role][-2:]) % 2 else row
            for row in super().rows()
        ]


_UNIT_VIEW = copy.deepcopy(WIDE_VIEW)
_UNIT_VIEW["dimensions"].append({"name": "unit", "values": [], "plural": True, "total_count": 2})


@pytest.mark.parametrize(
    ("store", "chart", "view", "warned", "kind_reads"),
    [
        (UnreadableFrames(), {}, None, True, 0),
        (FailingStore(), {}, None, True, 0),
        (HeldFrames(held=False), {}, None, True, 0),
        (UncountedFrames(), {}, None, True, 0),
        (_rank_positions(), {}, None, False, 0),
        (TwoUnitFrames(), {}, _UNIT_VIEW, False, 1),
        (HeldFrames(), {}, {**WIDE_VIEW, "rolling": True}, False, 0),
        (HeldFrames(), {"type": "line_chart", "latest_periods": 2}, None, False, 0),
    ],
    ids=[
        "unreadable",
        "store_failed",
        "not_held",
        "no_row_count",
        "rank_positions",
        "two_units",
        "rolling",
        "latest_periods",
    ],
)
async def test_make_charts_keeps_the_cap_refusal_when_the_frame_cannot_be_ranked(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    store: Any,
    chart: dict[str, Any],
    view: Any,
    warned: bool,
    kind_reads: int,
) -> None:
    """A frame the store cannot read, a plan's rank positions (where the
    highest number is the worst), rows in two units, and a card whose period
    moves with the data (a list picked today would not follow it) are never
    narrowed: the select is written as it is, and dfo-apis' refusal reaches
    the model as the platform gave it, naming the limit the chart type
    crosses. A frame the store cannot read, does not hold or states no row
    count for is logged as a warning. Only rows in two units, which are ranked
    after the read of the members' kinds, start that read."""
    monkeypatch.setattr(orchestrator_tools, "read_derived_marker", _marker_of)

    with caplog.at_level(logging.WARNING, logger=orchestrator_tools.__name__):
        executor, result = await _draw_wide(monkeypatch, store, view=view, **chart)

    (refused,) = result["charts"]
    assert _writes(executor) == 1
    assert (orchestrator_tools.__name__ in {record.name for record in caplog.records}) is warned
    assert refused == _relayed(executor.refusals[-1])
    assert refused["limit_name"] == (
        "MAX_LINE_CHART_SERIES" if chart else "MAX_SERIES_PER_COMPONENT"
    )
    assert _kind_reads(executor) == kind_reads


def _marker_of(trace: Any) -> Any:
    """`read_derived_marker` over the test's stand-in trace."""
    derived = getattr(getattr(trace, "applied_policy", None), "detail", {}).get("derived")
    return SimpleNamespace(**derived) if derived else None


# --- ask_user and plan: one pending action per turn ----------------------


def test_ask_user_records_the_question_and_refuses_a_second_hand_off():
    """A recorded hand-off ends the turn without the model call that would only
    end it; a refused one goes back to the model."""
    tools, ctx = _tools()
    _, later = _tools()

    questions = [
        {
            "question": "Which Congo?",
            "options": [{"label": "Republic of the Congo"}, {"label": "DR Congo"}],
        }
    ]

    assert "refusal" in tools["ask_user"]([], ctx)
    assert not ctx.actions.skip_summarization
    asked = tools["ask_user"](questions, ctx)
    later.state.update(ctx.state)
    second = tools["plan"]("Congo's GDP", ["Congo's GDP"], later)

    assert asked[HANDED_OFF] == "ask"
    assert ctx.actions.skip_summarization is True
    assert HANDED_OFF not in second
    assert not later.actions.skip_summarization
    assert pending_action(ctx.state) == PendingAsk(questions=questions)
    assert "already waiting on ask" in second["error"]
    assert _stated(second) == ("hand_off_pending", "do_not_retry")


@pytest.mark.parametrize("constant", [False, True], ids=["counts", "counts_and_prices"])
async def test_search_names_the_count_it_chose_and_the_others_by_spec_id(constant: bool):
    # Search's answer to "Bilateral plus imputed multilateral aid from France to
    # Kenya", and the same answer offering the gross count in constant prices
    # as dfo-apis lists it: axis "unit", labelled by its prices.
    envelope = json.loads(
        (
            Path(__file__).parents[2]
            / "data-mcp"
            / "tests"
            / "fixtures"
            / "resolve_aid_france_kenya.json"
        ).read_text()
    )
    if constant:
        envelope["parts"][0]["spec"]["resolved"]["alternative_measures"].append(
            {"label": "constant prices", "axis": "unit", "definition": {"spec_id": SPEC_CONSTANT}}
        )
    tools, ctx = _tools(resolver=_resolver(resolver_digest(envelope)))

    found = await tools["search"](ctx, [{"variable": "aid"}])

    (part,) = found["parts"]
    assert part["counted_as"] == "gross disbursements"
    assert part["alternatives"] == [
        {"spec_id": "spec_c67524ce5da3bfd4d94994a76c560b28", "label": "net disbursements"}
    ]
    assert part.get("other_prices") == (
        [{"spec_id": SPEC_CONSTANT, "label": "constant prices"}] if constant else None
    )
    # The answer spans 1978 to 2024 in current US dollars, and the question
    # named no prices, so the prices are open with the offered sibling or,
    # when search offers none, constant prices.
    assert part["open_prices"] == [
        {"label": "current prices"},
        {"spec_id": SPEC_CONSTANT, "label": "constant prices"}
        if constant
        else {"label": "constant prices"},
    ]


SPEC_NET = "spec_" + "c" * 32
SPEC_CONSTANT = "spec_" + "e" * 32
# `FAMILY_BLOCK` counted as gross disbursements, offering net disbursements
# and constant prices, each under its own spec_id, as search's
# `resolved.alternative_measures` states them.
COUNTED_BLOCK = {
    **FAMILY_BLOCK,
    "resolved": {
        **FAMILY_BLOCK["resolved"],
        "measure_label": "gross disbursements",
        "alternative_measures": [
            {"label": "net disbursements", "axis": "what", "definition": {"spec_id": SPEC_NET}},
            {"label": "constant prices", "axis": "unit", "definition": {"spec_id": SPEC_CONSTANT}},
        ],
    },
}


def _count_question(net: str) -> dict[str, Any]:
    """`ask_user`'s question on how to count the aid: gross under `SPEC_AID`, net under `net`."""
    return {
        "question": "How should aid be counted?",
        "options": [
            {"label": "gross disbursements", "spec_id": SPEC_AID},
            {"label": "net disbursements", "spec_id": net},
        ],
    }


_PRICES_QUESTION = {
    "question": "In which prices?",
    "options": [{"label": "Current US dollars"}, {"label": "Constant 2023 US dollars"}],
}


@pytest.mark.parametrize(
    ("questions", "refused"),
    [
        ([_count_question(SPEC_NET), _PRICES_QUESTION], None),
        (
            [
                _count_question(SPEC_NET),
                {
                    "question": "In which prices?",
                    "options": [
                        {"label": "Current US dollars"},
                        {"label": "constant prices", "spec_id": SPEC_CONSTANT},
                    ],
                },
            ],
            None,
        ),
        ([_count_question("spec_" + "d" * 32)], "which no search result named this turn"),
        (
            [
                _count_question(SPEC_NET),
                {
                    "question": "In which prices?",
                    "options": [
                        {"label": "current prices", "spec_id": SPEC_AID},
                        {"label": "constant prices", "spec_id": SPEC_CONSTANT},
                    ],
                },
            ],
            f"{SPEC_AID} is on more than one option",
        ),
        ([_count_question(SPEC_AID)], "is on more than one option"),
        (
            [
                _count_question(SPEC_NET),
                {
                    "question": "Which count first?",
                    "options": [{"label": "net disbursements", "spec_id": SPEC_NET}],
                },
            ],
            f"{SPEC_NET} is on more than one option",
        ),
        ([], "it asks 0 questions, and ask_user takes 1 to 3"),
        ([_PRICES_QUESTION] * 4, "it asks 4 questions, and ask_user takes 1 to 3"),
        (
            [_count_question(SPEC_NET), {"question": "In which prices?", "options": []}],
            "question 'In which prices?' offers no option",
        ),
        (
            [{"question": "In which prices?", "prices_of": SPEC_NET}],
            f"prices {SPEC_NET}, which lists no `open_prices` this turn",
        ),
    ],
    ids=[
        "offered",
        "prices_offered",
        "unnamed",
        "elected_on_both_first_options",
        "repeated",
        "repeated_across_questions",
        "no_question",
        "four_questions",
        "no_option",
        "prices_of_no_open_part",
    ],
)
async def test_ask_user_offers_a_count_only_under_a_spec_id_this_turns_search_named(
    questions: list[dict[str, Any]], refused: str | None
):
    part = {"variable": "aid", "status": "definite", "spec_id": SPEC_AID}
    tools, ctx = _tools(resolver=_resolver(_digest({**part, "spec_block": COUNTED_BLOCK})))
    await tools["search"](ctx, [{"variable": "aid"}])

    asked = tools["ask_user"](questions, ctx)

    if refused is None:
        assert asked[HANDED_OFF] == "ask"
        pending = pending_action(ctx.state)
        assert [question.question for question in pending.questions] == [
            "How should aid be counted?",
            "In which prices?",
        ]
        assert pending.questions[0].options[1].spec_id == SPEC_NET
    else:
        assert refused in asked["refusal"]
        assert asked["code"] == ToolErrorCode.REQUEST_INVALID.value
        assert ctx.state.get(ORCHESTRATOR_PENDING_KEY) is None
        assert ctx.state.get(ORCHESTRATOR_CALLS_KEY, {}).get("ask_user") is None


# Search's answer to "Bilateral plus imputed multilateral aid from France to
# Kenya": gross disbursements, one indicator, offering net disbursements.
FRANCE_KENYA = json.loads(
    (
        Path(__file__).parents[2]
        / "data-mcp"
        / "tests"
        / "fixtures"
        / "resolve_aid_france_kenya.json"
    ).read_text()
)
FRANCE_KENYA_NET = "spec_c67524ce5da3bfd4d94994a76c560b28"


def _net_family_envelope() -> dict[str, Any]:
    """Search's answer for the net count's definition: a family of two
    indicators summable over the type of finance, unlike the elected gross
    count's one, offering the gross count back."""
    spec = copy.deepcopy(FRANCE_KENYA["parts"][0]["spec"])
    (net,) = spec["resolved"]["alternative_measures"]
    spec["resolved"]["alternative_measures"] = [
        {**net, "definition": spec["definition"], "label": "gross disbursements"}
    ]
    spec["spec_id"] = FRANCE_KENYA_NET
    spec["definition"] = net["definition"]
    spec["resolved"]["measure_label"] = "net disbursements"
    spec["resolved"]["stat_vars"].update(member_count=2, served_count=2)
    spec["resolved"]["inapplicable_slots"] = []
    for slot in spec["resolved"]["slots"]:
        if slot["binding"]["kind"] == "unbound":
            slot["summable"] = True
    return {
        "status": "resolved",
        "parts": [{"status": "definite", "variable_text": "aid", "spec": spec}],
    }


# The net count in constant prices, which the constant-prices sibling of the
# elected gross count offers as its other count.
FRANCE_KENYA_NET_CONSTANT = "spec_" + "d" * 32


def _prices_offered() -> dict[str, Any]:
    """Search's answer for aid from France to Kenya with its one sibling, under
    `FRANCE_KENYA_NET`'s spec_id, offered as the gross count in constant
    prices (axis `unit`)."""
    envelope = copy.deepcopy(FRANCE_KENYA)
    (sibling,) = envelope["parts"][0]["spec"]["resolved"]["alternative_measures"]
    sibling.update(axis="unit", label="constant prices")
    return envelope


def _constant_prices_envelope() -> dict[str, Any]:
    """Search's answer for the constant-prices sibling's definition: the gross
    count search elected, offering current prices back and the net count in
    constant prices."""
    envelope = _net_family_envelope()
    resolved = envelope["parts"][0]["spec"]["resolved"]
    (current,) = resolved["alternative_measures"]
    net = copy.deepcopy(current)
    net["definition"]["spec_id"] = FRANCE_KENYA_NET_CONSTANT
    resolved["alternative_measures"] = [
        {**net, "label": "net disbursements"},
        {**current, "axis": "unit", "label": "current prices"},
    ]
    resolved["measure_label"] = "gross disbursements"
    return envelope


# How search offers `FRANCE_KENYA_NET` beside the gross count it elected: the
# search answer, the sibling's own definition, and the lines the offering
# reading and the sibling's reading render, with `{gross}` and `{net}` for
# the two spec_ids.
_SIBLINGS = {
    "another-count": (
        FRANCE_KENYA,
        _net_family_envelope,
        (
            "   counted as gross disbursements; other counts: net disbursements [{net}]",
            "   counted as net disbursements, another count of {gross}",
        ),
    ),
    "other-prices": (
        _prices_offered(),
        _constant_prices_envelope,
        (
            "   counted as gross disbursements; other prices: constant prices [{net}]",
            "   counted as gross disbursements in constant prices, other prices of {gross}",
        ),
    ),
}


async def _searched_france_kenya(
    definition_surface, envelope: dict[str, Any] = FRANCE_KENYA
) -> tuple[dict[str, Any], Any, FakeExecutor]:
    """The tools after one search for aid from France to Kenya answered by
    `envelope`, with the conversation's identities on record as
    `harness.studio` keeps them."""
    executor = FakeExecutor()
    tools = build_orchestrator_tools(
        resolver=_resolver(resolver_digest(envelope)),
        executor=executor,
        definition_surface=definition_surface,
    )
    by_name = {fn.__name__: fn for fn in tools}
    ctx = SimpleNamespace(state={"surface": "studio"}, invocation_id="inv_1")
    conversation = ConversationState(conversation_id="c1")
    current_recorded_identity.set(conversation.recorded_identity)
    await by_name["search"](ctx, [{"variable": "aid"}])
    conversation.turn_specs = dict(ctx.state["orchestrator:specs"])
    return by_name, ctx, executor


@pytest.mark.parametrize("offered", _SIBLINGS)
async def test_a_fetch_of_a_sibling_reads_its_own_block_and_gains_a_reading(offered: str):
    """A fetched sibling gains a reading stating the count it reads, linked
    to the reading that offered it, which a later turn renders."""
    searched, definition, lines = _SIBLINGS[offered]
    surface = FakeSurface(ToolCallResult(is_error=False, **search_result_parts(definition())))
    tools, ctx, executor = await _searched_france_kenya(surface, searched)

    found = await tools["fetch"](FRANCE_KENYA_NET, ctx)

    ((tool, arguments),) = surface.calls
    assert (tool.name, arguments["spec_id"]) == ("resolve_definition", FRANCE_KENYA_NET)
    assert [args for tool, args in executor.calls if tool == "get_data"] == [
        {"spec_id": FRANCE_KENYA_NET, "stat_vars": "sum"}
    ]
    assert found["summary"] == "64 observations"
    assert ctx.state["orchestrator:specs"][FRANCE_KENYA_NET]["spec_id"] == FRANCE_KENYA_NET
    elected, chosen = (SearchReading.model_validate(r) for r in ctx.state["orchestrator:searches"])
    assert (chosen.status, chosen.spec_id, chosen.text) == (
        "chosen",
        FRANCE_KENYA_NET,
        elected.text,
    )
    assert chosen.fetch is not None and chosen.fetch.summary == "64 observations"
    assert elected.fetch is None
    assert (elected.sibling_of, chosen.sibling_of and chosen.sibling_of.spec_id) == (
        None,
        elected.spec_id,
    )
    rendered = render_readings([elected, chosen], budget=10_000).text.splitlines()
    for line in lines:
        assert line.format(gross=elected.spec_id, net=FRANCE_KENYA_NET) in rendered


@pytest.mark.parametrize(
    ("offered", "count_of", "options", "chosen", "rationale"),
    [
        (
            "another-count",
            True,
            ("gross disbursements", "net disbursements"),
            "net disbursements",
            "Chosen in place of gross disbursements, the measure search elected.",
        ),
        (
            "other-prices",
            False,
            ("gross disbursements", "net disbursements"),
            "gross disbursements",
            "Search elected gross disbursements for this source.",
        ),
    ],
)
@pytest.mark.parametrize("gross_first", [True, False], ids=["gross-first", "net-first"])
async def test_a_fetched_sibling_states_the_measure_search_elected(
    gross_first: bool,
    offered: str,
    count_of: bool,
    options: tuple[str, ...],
    chosen: str,
    rationale: str,
):
    """The net count's own block names net as its measure and offers gross
    back; a plan pinning net after a fetch of it still says search elected
    gross, whichever source the planner is shown first. The gross count in
    constant prices settles as its own source: its block elects the gross
    count search elected, and offers the net count in constant prices."""
    searched, definition, _ = _SIBLINGS[offered]
    surface = FakeSurface(ToolCallResult(is_error=False, **search_result_parts(definition())))
    tools, ctx, _ = await _searched_france_kenya(surface, searched)
    await tools["fetch"](FRANCE_KENYA_NET, ctx)
    gross = FRANCE_KENYA["parts"][0]["spec"]["spec_id"]
    shown = [gross, FRANCE_KENYA_NET] if gross_first else [FRANCE_KENYA_NET, gross]

    pins = planner_adapter._reading_pin_sources(
        ctx.state["orchestrator:searches"],
        shown,
        blocks=ctx.state["orchestrator:specs"],
        earlier=(),
        warnings={},
    )
    choice = submission_service.measure_choices(pins)[FRANCE_KENYA_NET]
    decision = submission_service._measure_decision(
        "counted_as", FRANCE_KENYA_NET, choice, user=None
    )

    assert pins[FRANCE_KENYA_NET].settled
    assert pins[FRANCE_KENYA_NET].other_count_of == (gross if count_of else None)
    assert (decision.options, decision.chosen, decision.rationale) == (
        options,
        chosen,
        rationale,
    )


@pytest.mark.parametrize(
    ("answer", "code"),
    [
        (
            ToolCallResult(is_error=True, structured_content={"code": "spec_not_resolvable"}),
            ToolErrorCode.DEFINITION_NOT_HELD.value,
        ),
        (
            ToolCallResult(is_error=False, **search_result_parts(FRANCE_KENYA)),
            ToolErrorCode.DEFINITION_NOT_HELD.value,
        ),
        (
            ToolCallResult(
                is_error=True,
                structured_content={
                    "code": "service_unavailable",
                    "retry_mode": "retry_after_delay",
                },
            ),
            HarnessCode.TRANSIENT_UNAVAILABLE,
        ),
    ],
    ids=["refused", "another-dataset", "unreachable"],
)
async def test_a_fetch_of_another_count_search_cannot_read_is_refused_without_get_data(
    answer: ToolCallResult, code: str
):
    tools, ctx, executor = await _searched_france_kenya(FakeSurface(answer))

    refused = await tools["fetch"](FRANCE_KENYA_NET, ctx)

    assert refused["code"] == code
    assert FRANCE_KENYA_NET in refused["refusal"]
    assert not [call for call in executor.calls if call[0] == "get_data"]
    assert FRANCE_KENYA_NET not in ctx.state["orchestrator:specs"]


def test_plan_records_the_brief_once_per_turn():
    tools, ctx = _tools()

    first = tools["plan"]("Kenya's aid as a share of GDP since 2010", ["the share since 2010"], ctx)
    ctx.state["orchestrator:pending"] = None  # the workflow took it
    again = tools["plan"]("Kenya's aid as a share of GDP since 2010", ["the share since 2010"], ctx)

    assert first["status"].startswith("Recorded")
    assert pending_action({"orchestrator:pending": {"kind": "plan", "brief": "b"}}) == PendingPlan(
        brief="b"
    )
    assert "limit" in again["error"]


@pytest.mark.parametrize(
    "asks",
    [
        [],
        [f"output {number}" for number in range(MAX_PLAN_ASKS + 1)],
        ["the total since 2010", "  "],
        ["x" * (MAX_ASK_CHARS + 1)],
    ],
)
def test_plan_refuses_asks_past_the_planner_limits_without_spending_the_call(asks):
    tools, ctx = _tools()

    refused = tools["plan"]("France's aid to Guatemala", asks, ctx)
    recorded = tools["plan"]("France's aid to Guatemala", [" the total since 2010 "], ctx)

    assert refused["code"] == ToolErrorCode.REQUEST_INVALID.value
    assert refused["legal"] == {"asks": [1, MAX_PLAN_ASKS], "ask_chars": [1, MAX_ASK_CHARS]}
    assert HANDED_OFF not in refused
    assert recorded[HANDED_OFF] == "plan"
    assert pending_action(ctx.state) == PendingPlan(
        brief="France's aid to Guatemala", asks=["the total since 2010"]
    )


_BY_CREDITOR = {
    "variable": "general government debt service by creditor",
    "time": "2031",
    "open_roles": ["creditor"],
}


async def _searched(part: dict[str, Any], status: str = "definite"):
    """The tools and context after one search of `part` that read as `status`."""
    reading = {"variable": part["variable"], "status": status}
    if status == "definite":
        reading["spec_id"] = SPEC_AID
    tools, ctx = _tools(_resolver(_digest(reading, kind="resolution")))
    await tools["search"](ctx, [part])
    return tools, ctx


async def test_plan_refuses_asks_that_leave_out_a_party_the_search_held_open():
    tools, ctx = await _searched(_BY_CREDITOR)

    refused = tools["plan"]("Debt service by creditor", ["Debt service by creditor"], ctx)
    recorded = tools["plan"](
        "Debt service by creditor",
        [{"text": "Debt service by creditor", "by": ["creditor", "creditor"]}],
        ctx,
    )

    assert refused["invalid"] == [{"loc": "asks", "type": "breakdown_missing", "party": "creditor"}]
    assert "`by`" in refused["error"] and "`over`" in refused["error"]
    assert recorded[HANDED_OFF] == "plan"
    assert pending_action(ctx.state) == PendingPlan(
        brief="Debt service by creditor",
        asks=["Debt service by creditor"],
        ask_breakdowns=[AskBreakdown(ask=0, by=("creditor",))],
    )


@pytest.mark.parametrize(
    ("part", "status"),
    [
        (_BY_CREDITOR, "no_data"),
        ({"variable": "aid", "entities": [{"text": "France", "role": "from"}]}, "definite"),
    ],
    ids=["no_data", "named_party"],
)
async def test_plan_takes_asks_without_parties_when_no_settled_search_held_one_open(
    part: dict[str, Any], status: str
):
    tools, ctx = await _searched(part, status)

    recorded = tools["plan"]("the brief", ["the total"], ctx)

    assert recorded[HANDED_OFF] == "plan"
    assert pending_action(ctx.state) == PendingPlan(brief="the brief", asks=["the total"])


_FROM_ANY_ORIGIN = {**_BY_CREDITOR, "open_roles": ["origin"]}


@pytest.mark.parametrize(
    "parts", [[_FROM_ANY_ORIGIN], [_FROM_ANY_ORIGIN, _BY_CREDITOR]], ids=["origin", "then-creditor"]
)
async def test_plan_takes_an_ask_naming_the_held_side_in_another_word(parts: list[dict[str, Any]]):
    tools, ctx = await _searched(parts[0])
    for part in parts[1:]:
        await tools["search"](ctx, [part])

    recorded = tools["plan"](
        "Debt service by creditor", [{"text": "Debt service by creditor", "by": ["creditor"]}], ctx
    )

    assert recorded[HANDED_OFF] == "plan"


async def test_plan_refuses_asks_naming_one_of_two_parties_two_flows_held_open():
    tools, ctx = await _searched({"variable": "aid by donor", "open_roles": ["donor"]})
    await tools["search"](ctx, [_BY_CREDITOR])

    refused = tools["plan"](
        "brief", [{"text": "aid by donor", "by": ["donor"]}, "total debt service"], ctx
    )

    assert refused["invalid"] == [{"loc": "asks", "type": "breakdown_missing", "party": "creditor"}]


async def test_plan_refuses_an_ask_listing_two_parties_of_one_side_and_takes_an_ask_for_each():
    tools, ctx = await _searched({"variable": "aid by donor", "open_roles": ["donor"]})
    await tools["search"](ctx, [_BY_CREDITOR])

    refused = tools["plan"](
        "brief", [{"text": "aid by donor, debt by creditor", "by": ["donor", "creditor"]}], ctx
    )
    recorded = tools["plan"](
        "brief",
        [{"text": "aid by donor", "by": ["donor"]}, {"text": "debt", "by": ["creditor"]}],
        ctx,
    )

    assert refused["invalid"] == [{"loc": "asks.0", "type": "value_error"}]
    assert recorded[HANDED_OFF] == "plan"
    assert pending_action(ctx.state) == PendingPlan(
        brief="brief",
        asks=["aid by donor", "debt"],
        ask_breakdowns=[
            AskBreakdown(ask=0, by=("donor",)),
            AskBreakdown(ask=1, by=("creditor",)),
        ],
    )


async def test_plan_names_the_parties_of_a_side_the_search_held_open_by_its_column():
    tools, ctx = await _searched(_FROM_ANY_ORIGIN)

    refused = tools["plan"](
        "Debt service by debtor", [{"text": "Debt service by debtor", "by": ["debtor"]}], ctx
    )

    assert refused["invalid"] == [
        {"loc": "asks", "type": "breakdown_missing", "party": "donor/creditor/exporter"}
    ]
    assert "held donor/creditor/exporter open" in refused["error"]
    assert not re.search(r'(?<!one of )"donor"', refused["error"])
    assert pending_action(ctx.state) is None


async def test_plan_carries_the_asks_put_across_years_by_number():
    tools, ctx = await _searched(_BY_CREDITOR)

    recorded = tools["plan"](
        "Total debt service between 2015 and 2022",
        [
            {"text": "Debt service by creditor", "by": ["creditor"]},
            {"text": "The total between 2015 and 2022", "over": ["creditor"], "across_years": True},
        ],
        ctx,
    )

    assert recorded[HANDED_OFF] == "plan"
    assert pending_action(ctx.state) == PendingPlan(
        brief="Total debt service between 2015 and 2022",
        asks=["Debt service by creditor", "The total between 2015 and 2022"],
        ask_breakdowns=[
            AskBreakdown(ask=0, by=("creditor",)),
            AskBreakdown(ask=1, over=("creditor",)),
        ],
        asks_across_years=[1],
    )


@pytest.mark.parametrize(
    "over",
    [("creditor",), ("donor", "creditor")],
    ids=["one-word", "two-words-for-one-side"],
)
async def test_plan_takes_a_party_the_search_held_open_that_an_ask_totals_across(
    over: tuple[PartyRole, ...],
):
    tools, ctx = await _searched(_BY_CREDITOR)

    recorded = tools["plan"](
        "Total debt service", [{"text": "Total debt service", "over": list(over)}], ctx
    )

    assert recorded[HANDED_OFF] == "plan"
    assert pending_action(ctx.state) == PendingPlan(
        brief="Total debt service",
        asks=["Total debt service"],
        ask_breakdowns=[AskBreakdown(ask=0, over=over)],
    )


_US_SHARE_OF_KENYA = {
    "variable": "ODA",
    "entities": [{"text": "Kenya", "role": "to"}, {"text": "United States", "role": "from"}],
    "operations": [
        {"verb": "share_of", "phrase": "share from the United States", "over": "origin"}
    ],
}


_EACH_DONORS_SHARE_OF_KENYA = {
    **_US_SHARE_OF_KENYA,
    "entities": [{"text": "Kenya", "role": "to"}],
    "open_roles": ["donor"],
}


@pytest.mark.parametrize(
    ("part", "ask", "taken"),
    [
        pytest.param(_US_SHARE_OF_KENYA, "US share of ODA to Kenya", False, id="no-party"),
        pytest.param(
            _US_SHARE_OF_KENYA,
            {"text": "US share of ODA to Kenya", "by": ["donor"]},
            False,
            id="named-member-by",
        ),
        pytest.param(
            _US_SHARE_OF_KENYA,
            {"text": "US share of ODA to Kenya", "over": ["donor"]},
            True,
            id="named-member-over",
        ),
        pytest.param(
            _EACH_DONORS_SHARE_OF_KENYA,
            {"text": "each donor's share", "by": ["donor"]},
            True,
            id="held-open-by",
        ),
    ],
)
async def test_plan_holds_a_share_searchs_side_to_an_ask_totalling_across_it(
    part: dict[str, Any], ask: str | dict[str, Any], taken: bool
):
    # The whole of a share over origin spans every donor, so the ask the
    # share answers totals across them, which the plan then holds it to. A
    # search holding donors open lists each, so `by` answers it too.
    tools, ctx = await _searched(part)

    recorded = tools["plan"]("brief", [ask], ctx)

    if taken:
        assert recorded[HANDED_OFF] == "plan"
        return
    assert recorded["invalid"] == [
        {"loc": "asks", "type": "share_whole_unstated", "party": "donor/creditor/exporter"}
    ]


@pytest.mark.parametrize("by", ["creditor", "donor"], ids=["one-word", "two-words-for-one-side"])
async def test_plan_refuses_an_ask_that_lists_and_totals_one_party(by: str):
    tools, ctx = await _searched(_BY_CREDITOR)

    refused = tools["plan"](
        "brief", [{"text": "Debt service", "by": [by], "over": ["creditor"]}], ctx
    )

    assert refused["invalid"] == [{"loc": "asks.0", "type": "value_error"}]
    assert pending_action(ctx.state) is None


async def test_the_breakdown_refusal_spends_the_invalid_argument_budget_not_the_plan():
    tools, ctx = await _searched(_BY_CREDITOR)

    for _ in range(BUDGETS["invalid_arguments"]):
        assert "invalid" in tools["plan"]("brief", ["Debt service"], ctx)
    spent = tools["plan"]("brief", ["Debt service"], ctx)

    assert "limit" in spent["error"]
    assert "plan" not in ctx.state[ORCHESTRATOR_CALLS_KEY]
    assert pending_action(ctx.state) is None


# --- the skill tools and explain_dfo ----------------------------------------


class FakeSurface:
    def __init__(self, result: ToolCallResult) -> None:
        self.result, self.calls = result, []

    async def list_tools(self):
        return []

    async def call_tool(self, *, tool, arguments, on_progress=None) -> ToolCallResult:
        self.calls.append((tool, arguments))
        return self.result


async def test_get_skill_loads_a_skill_through_the_tool_surface():
    surface = FakeSurface(ToolCallResult(is_error=False, text="---\nname: ODA\n---\nAid is..."))
    tools, ctx = _tools(surface=surface)

    loaded = await tools["get_skill"]("oda-measures", ctx)

    assert (surface.calls[0][0].server, surface.calls[0][0].name) == ("data-mcp", "get_skill")
    assert surface.calls[0][1] == {"skill_id": "oda-measures"}
    assert loaded == {"skill": "---\nname: ODA\n---\nAid is..."}


async def test_get_skill_reads_a_file_and_on_with_the_offset_and_digest_a_page_named():
    surface = FakeSurface(ToolCallResult(is_error=False, text="the rest"))
    tools, ctx = _tools(surface=surface)

    read = await tools["get_skill"](
        "oda-measures", ctx, path="SKILL.md", offset=32000, digest="sha256:" + "0" * 64
    )
    first = await tools["get_skill"]("oda-measures", ctx, path="references/grant.md")

    assert {(tool.server, tool.name) for tool, _ in surface.calls} == {("data-mcp", "get_skill")}
    assert [arguments for _, arguments in surface.calls] == [
        {
            "skill_id": "oda-measures",
            "path": "SKILL.md",
            "offset": 32000,
            "digest": "sha256:" + "0" * 64,
        },
        {"skill_id": "oda-measures", "path": "references/grant.md"},
    ]
    assert read == first == {"skill": "the rest"}


@pytest.mark.parametrize(
    ("path", "budget", "kind"),
    [(None, "get_skill", "first page"), ("SKILL.md", "get_skill_file", "file")],
    ids=["first-pages", "files"],
)
async def test_first_page_reads_and_file_reads_each_spend_their_own_limit(
    path: str | None, budget: str, kind: str
):
    surface = FakeSurface(ToolCallResult(is_error=False, text="page"))
    tools, ctx = _tools(surface=surface)
    other = None if path else "SKILL.md"

    for _ in range(BUDGETS[budget]):
        assert await tools["get_skill"]("oda-measures", ctx, path=path) == {"skill": "page"}
    spent = await tools["get_skill"]("oda-measures", ctx, path=path)
    still_open = await tools["get_skill"]("oda-measures", ctx, path=other)

    assert _stated(spent) == (HarnessCode.TOOL_BUDGET_SPENT, "do_not_retry")
    assert spent["error"].startswith("get_skill reading a ") and kind in spent["error"]
    assert "get_skill_file" not in spent["error"]
    assert still_open == {"skill": "page"}
    assert len(surface.calls) == BUDGETS[budget] + 1


async def test_get_skill_reports_a_refusal_and_its_absence():
    refused = ToolCallResult(
        is_error=True, structured_content={"message": "no such skill"}, text=""
    )
    tools, ctx = _tools(surface=FakeSurface(refused))
    assert await tools["get_skill"]("nope", ctx) == {
        "error": "no such skill",
        "code": "unspecified",
        "retry_mode": "do_not_retry",
    }

    # The service's own code and retry mode come with its message.
    body = {"message": "no such skill", "code": "skill_not_found", "retry_mode": "do_not_retry"}
    tools, ctx = _tools(
        surface=FakeSurface(refused.model_copy(update={"structured_content": body}))
    )
    assert _stated(await tools["get_skill"]("nope", ctx)) == ("skill_not_found", "do_not_retry")

    tools, ctx = _tools()
    unavailable = await tools["get_skill"]("nope", ctx)
    assert "not available" in unavailable["error"]
    assert _stated(unavailable) == ("tool_not_configured", "do_not_retry")


class UnreachableSurface(FakeSurface):
    """A surface whose call times out, as `compiler.surface`'s per-tool timeout raises.

    `retryable` is the mark the raise site gives the error."""

    def __init__(self, result: ToolCallResult, *, retryable: bool = True) -> None:
        super().__init__(result)
        self.retryable = retryable

    async def call_tool(self, *, tool, arguments, on_progress=None) -> ToolCallResult:
        self.calls.append((tool, arguments))
        raise ToolSurfaceUnavailableError(
            f"{tool.name} timed out after 30s", retryable=self.retryable
        )


async def test_a_skill_read_that_does_not_complete_is_a_retryable_error_for_the_model():
    surface = UnreachableSurface(ToolCallResult(is_error=False, text=""))
    tools, ctx = _tools(surface=surface)

    answered = await tools["get_skill"]("oda-measures", ctx)

    assert answered == {
        "error": "get_skill could not reach data-mcp just now. Call it again, or answer "
        "without this skill.",
        "code": "transient_unavailable",
        "retry_mode": "retry_after_delay",
    }
    assert len(surface.calls) == 1


async def test_a_skill_read_against_a_server_that_is_down_is_a_retryable_error_for_the_model():
    tools, ctx = _tools(surface=unreachable_surface())

    answered = await tools["get_skill"]("oda-measures", ctx)

    assert _stated(answered) == ("transient_unavailable", "retry_after_delay")


async def test_a_skill_read_that_cannot_complete_on_a_retry_is_not_offered_one():
    surface = UnreachableSurface(ToolCallResult(is_error=False, text=""), retryable=False)
    tools, ctx = _tools(surface=surface)

    answered = await tools["get_skill"]("oda-measures", ctx)

    assert _stated(answered) == ("tool_not_configured", "do_not_retry")
    assert "Call it again" not in answered["error"]


class ByToolSurface(FakeSurface):
    """A surface answering each tool with its own result; a tool mapped to
    `None` raises as a call that does not complete."""

    def __init__(self, results: dict[str, ToolCallResult | None]) -> None:
        super().__init__(ToolCallResult(is_error=False, text=""))
        self.results = results

    async def call_tool(self, *, tool, arguments, on_progress=None) -> ToolCallResult:
        self.calls.append((tool, arguments))
        result = self.results[tool.name]
        if result is None:
            raise ToolSurfaceUnavailableError(f"{tool.name} timed out after 30s", retryable=True)
        return result


_COUNTS = "DFO holds 108,905 indicators in 92 datasets from 42 publishers."


@pytest.mark.parametrize(
    ("counts_result", "counts"),
    [
        (ToolCallResult(is_error=False, text=_COUNTS), _COUNTS),
        # apis answers 503 while no search catalogue is published.
        (
            ToolCallResult(is_error=True, structured_content={"code": "service_unavailable"}),
            COUNTS_UNAVAILABLE,
        ),
        (None, COUNTS_UNAVAILABLE),
    ],
)
async def test_explain_dfo_reads_dfo_faq_and_the_counts_off_the_activity_feed(
    counts_result: ToolCallResult | None, counts: str
):
    """dfo-faq and the counts come through the bare surface: the recording one
    would put more steps beside `explain_dfo`'s own. Counts that could not be
    read are named unavailable, never quoted."""
    recorded = FakeSurface(ToolCallResult(is_error=False, text="unused"))
    bare = ByToolSurface(
        {
            "get_skill": ToolCallResult(is_error=False, text="# The DFO\nBuilt by ONE Data."),
            "catalogue_counts": counts_result,
        }
    )
    tools, _ = _tools(surface=recorded, dfo_faq_surface=bare)

    explained = await tools["explain_dfo"]()

    assert sorted((tool.name, arguments) for tool, arguments in bare.calls) == [
        ("catalogue_counts", {}),
        ("get_skill", {"skill_id": "dfo-faq"}),
    ]
    assert recorded.calls == []
    assert explained == {"explainer": "# The DFO\nBuilt by ONE Data.", "counts": counts}


@pytest.mark.parametrize(
    ("surface", "said"),
    [
        (
            FakeSurface(
                ToolCallResult(is_error=True, structured_content={"code": "capability_unavailable"})
            ),
            CANNOT_DESCRIBE_DFO,
        ),
        (
            FakeSurface(
                ToolCallResult(is_error=True, structured_content={"code": "skill_not_found"})
            ),
            CANNOT_DESCRIBE_DFO,
        ),
        (
            FakeSurface(
                ToolCallResult(is_error=True, structured_content={"code": "service_unavailable"})
            ),
            DFO_FAQ_NOT_READ,
        ),
        (UnreachableSurface(ToolCallResult(is_error=False, text="")), DFO_FAQ_NOT_READ),
    ],
)
async def test_explain_dfo_says_plainly_when_dfo_faq_cannot_be_read(
    surface: FakeSurface, said: str
):
    tools, _ = _tools(dfo_faq_surface=surface)

    answered = await tools["explain_dfo"]()

    assert answered["error"] == said
    assert _stated(answered) == (
        ("tool_not_configured", "do_not_retry")
        if said == CANNOT_DESCRIBE_DFO
        else ("transient_unavailable", "retry_after_delay")
    )


# --- recall -----------------------------------------------------------------


def _recall_callable(state: ConversationState):
    async def recall_part(**target: Any) -> dict[str, Any]:
        return state.recall_part(**target)

    return recall_part


# --- remove_cards -------------------------------------------------------------


def _remove_callable(state: ConversationState):
    async def remove(*, component_ids: list[str], all_cards: bool) -> tuple[list[str], list[str]]:
        return remove_cards(state, component_ids, all_cards=all_cards)

    return remove


def _two_card_canvas() -> ConversationState:
    state = ConversationState(conversation_id="c1")
    for cid in ("cmp_01", "cmp_02"):
        land_component(
            state,
            {"component_id": cid, "revision": 1, "type": "line_chart", "title": "Aid"},
        )
    return state


async def test_remove_cards_takes_named_cards_off_and_says_which_it_never_held():
    state = _two_card_canvas()
    tools, ctx = _tools(remove_cards=_remove_callable(state))

    answer = await tools["remove_cards"](ctx, ["cmp_01", "cmp_99"])

    assert answer == {"removed": ["cmp_01"], "not_in_conversation": ["cmp_99"]}
    assert state.composition is not None
    assert [b.component_id for b in state.composition.blocks] == ["cmp_02"]


async def test_remove_cards_clears_the_canvas_and_refuses_a_call_that_names_nothing():
    state = _two_card_canvas()
    tools, ctx = _tools(remove_cards=_remove_callable(state))

    nothing = await tools["remove_cards"](ctx)
    cleared = await tools["remove_cards"](ctx, all_cards=True)
    again = await tools["remove_cards"](ctx, all_cards=True)

    assert _stated(nothing) == ("tool_arguments_invalid", "correct_and_retry")
    assert cleared == {"removed": ["cmp_01", "cmp_02"]}
    assert again == {"removed": [], "note": "The canvas was already empty."}


async def test_removing_a_card_already_off_the_canvas_is_done_not_refused():
    """A hidden or removed card is already where the user asked; refusing it
    showed raw ids in Activity (UAT A6)."""
    state = _two_card_canvas()
    tools, ctx = _tools(remove_cards=_remove_callable(state))
    await tools["remove_cards"](ctx, ["cmp_01"])

    again = await tools["remove_cards"](ctx, ["cmp_01"])
    unknown = await tools["remove_cards"](ctx, ["cmp_99"])

    assert again == {"removed": [], "note": "Those cards were already off the canvas."}
    assert _stated(unknown) == ("card_not_found", "correct_and_retry")
    assert "cmp_99" not in unknown["error"]


async def test_remove_cards_is_unavailable_without_a_conversation():
    tools, ctx = _tools()

    unavailable = ("tool_not_configured", "do_not_retry")
    assert _stated(await tools["remove_cards"](ctx, ["cmp_01"])) == unavailable
    assert _stated(await tools["recall"]("plan", ctx)) == unavailable


async def test_recall_returns_an_earlier_turns_plan_record_and_refuses_a_turn_without_one():
    state = ConversationState(conversation_id="c1")
    plan = _top_donors()[1]
    state.turns = [
        {"user": "q1", "assistant": "a1", "plans": [plan.model_dump(mode="json")]},
        {"user": "q2", "assistant": "a2"},
    ]
    tools, ctx = _tools(recall_part=_recall_callable(state))

    found = await tools["recall"]("plan", ctx, turn=1)
    refused = await tools["recall"]("plan", ctx, turn=2)

    assert found["turn"] == 1
    assert found["text"] == [plan.render(budget=4_000)]
    assert refused["code"] == "request_invalid"
    assert refused["legal"] == {"turns": [1]}


async def test_a_canvas_listing_does_not_spend_the_detail_reads():
    """Looking at the canvas has its own small allowance, so a listing
    followed by three card reads all go through."""
    state = ConversationState(conversation_id="c1")
    land_component(
        state,
        {"component_id": "cmp_01", "revision": 1, "type": "line_chart", "title": "Aid"},
    )
    tools, ctx = _tools(recall_part=_recall_callable(state))

    listing = await tools["recall"]("canvas", ctx)
    reads = [await tools["recall"]("canvas", ctx, component_id="cmp_01") for _ in range(3)]

    assert [card["component_id"] for card in listing["on_canvas"]] == ["cmp_01"]
    assert all("error" not in read for read in reads)
    for _ in range(BUDGETS["recall_canvas"] - 1):
        await tools["recall"]("canvas", ctx)
    assert "limit" in (await tools["recall"]("canvas", ctx))["error"]


async def test_recall_is_limited_per_turn_and_unavailable_without_a_conversation():
    tools, ctx = _tools()
    unavailable = await tools["recall"]("plan", ctx)

    assert "not available" in unavailable["error"]

    ctx.state["orchestrator:calls"] = {"recall": BUDGETS["recall"]}
    assert "limit" in (await tools["recall"]("plan", ctx))["error"]


class EmptyExecutor(FakeExecutor):
    """get_data's outcome for apis' real zero-row mint (Kenya health spending,
    1960-1970, on dev 2026-10-07): 200, `coverage.rows` 0 and an `empty_result` warning."""

    async def __call__(self, *, tool: str, args: dict[str, Any], on_progress: Any = None):
        if tool == "get_data":
            self.calls.append((tool, args))
            return {
                "outcome": "ok",
                "frame_refs": [REF],
                "summary": None,
                "rows": 0,
                "warnings": [
                    {
                        "code": "empty_result",
                        "severity": "warn",
                        "message": "The materialised frame holds zero observations.",
                    }
                ],
            }
        return await super().__call__(tool=tool, args=args, on_progress=on_progress)


async def test_fetch_of_data_with_no_observations_is_refused_and_registers_no_frame():
    executor = EmptyExecutor()
    tools, ctx = _tools(executor=executor)

    result = await tools["fetch"](SPEC_LIFE, ctx)

    assert result["code"] == "empty_result"
    assert "no observations" in result["refusal"]
    assert "frames" not in result
    assert [tool for tool, _ in executor.calls] == ["get_data"]
    assert not ctx.state.get("orchestrator:frames")


async def test_fetch_of_empty_data_again_unchanged_answers_without_a_call():
    executor = EmptyExecutor()
    tools, ctx = _tools(executor=executor)

    await tools["fetch"](SPEC_LIFE, ctx)
    again = await tools["fetch"](SPEC_LIFE, ctx)

    assert [tool for tool, _ in executor.calls] == ["get_data"]
    assert again["code"] == "empty_result"
    assert again["same_data_as"] == SPEC_LIFE


async def test_fetch_of_empty_data_through_an_equivalent_spec_records_it_under_its_own_id():
    executor = EmptyExecutor()
    tools, ctx = _tools(executor=executor)
    ctx.state["orchestrator:specs"] = {
        SPEC_AID: _role_open_block(SPEC_AID),
        SPEC_LIFE: _role_open_block(SPEC_LIFE),
    }

    await tools["fetch"](SPEC_AID, ctx)
    again = await tools["fetch"](SPEC_LIFE, ctx)

    assert [tool for tool, _ in executor.calls] == ["get_data"]
    assert again["spec_id"] == SPEC_LIFE
    assert again["same_data_as"] == SPEC_AID


class FacetRefusingExecutor(FakeExecutor):
    """get_data's outcome for apis' real refusal of a Euro unit on Kenya and Ghana
    health spending (dev, 2026-10-07)."""

    async def __call__(self, *, tool: str, args: dict[str, Any], on_progress: Any = None):
        self.calls.append((tool, args))
        return {
            "outcome": "refusal",
            "refusal": {
                "code": "default_facet_undefined",
                "retry_mode": "correct_and_retry",
                "message": "The platform refused the requested data for Kenya, Ghana.",
                "fix": {
                    "issues": [
                        {
                            "axis": "unit",
                            "alternatives": [{"ref": "USDollar", "label": "USDollar"}],
                            "truncated": False,
                            "code": None,
                            "detail": None,
                        }
                    ],
                    "detail": "one or more facet axes could not be resolved",
                    "retryable": False,
                },
            },
        }


async def test_fetch_refusal_carries_the_platforms_details_and_retry_mode():
    tools, ctx = _tools(executor=FacetRefusingExecutor())

    result = await tools["fetch"](SPEC_LIFE, ctx)

    assert result["code"] == "default_facet_undefined"
    assert result["retry_mode"] == "correct_and_retry"
    assert result["fix"]["issues"][0]["alternatives"] == [{"ref": "USDollar", "label": "USDollar"}]


async def test_remove_cards_stops_at_its_turn_budget():
    state = _two_card_canvas()
    tools, ctx = _tools(remove_cards=_remove_callable(state))
    for _ in range(BUDGETS["remove_cards"]):
        await tools["remove_cards"](ctx, all_cards=True)
    land_component(state, {"component_id": "cmp_new", "revision": 1, "type": "table"})

    refused = await tools["remove_cards"](ctx, all_cards=True)

    assert "error" in refused
    assert [b.component_id for b in state.composition.blocks] == ["cmp_new"]


async def test_source_name_choices_reach_the_agent_and_saved_ledger():

    groups = [
        {
            "text": "shared",
            "candidates": [
                {"ref": "provenance/a", "label": "Publisher A", "kind": "provenance"},
                {"ref": "provenance/b", "label": "Publisher B", "kind": "provenance"},
            ],
        }
    ]
    envelope = {
        "status": "no_data",
        "parts": [
            {
                "status": "no_data",
                "variable_text": "aid",
                "no_data": {
                    "category": "unresolved",
                    "reason": "source_ambiguous",
                    "ambiguous_sources": groups,
                },
            }
        ],
    }
    tools, ctx = _tools(_resolver(resolver_digest(envelope)))
    found = await tools["search"](ctx, [{"variable": "aid", "sources": [{"text": "shared"}]}])
    assert found["parts"][0]["ambiguous_sources"] == groups
    reading = SearchReading.model_validate(ctx.state["orchestrator:searches"][0])
    rendered = render_readings([reading], budget=10000).text
    assert "Publisher A [provenance/a]" in rendered
    assert "Publisher B [provenance/b]" in rendered


async def test_source_alias_confirmation_survives_an_orchestrator_pin():

    envelope = {
        "status": "definite",
        "parts": [
            {
                "status": "definite",
                "variable_text": "aid",
                "restatement": {"sources": [{"text": "DAC", "polarity": "include", "ref": None}]},
                "spec": {
                    **SINGLE_BLOCK,
                    "resolved": {
                        **SINGLE_BLOCK["resolved"],
                        "facets": {
                            "provenance": {"ref": "provenance/OECDDAC2A", "label": "OECD DAC2a"}
                        },
                    },
                },
            }
        ],
    }
    tools, ctx = _tools(_resolver(resolver_digest(envelope)))
    await tools["search"](ctx, [{"variable": "aid", "sources": [{"text": "DAC"}]}])
    source = source_from_spec_block(ctx.state["orchestrator:specs"][SPEC_LIFE])
    requirement = DatasetRequirement(
        id="aid",
        measure="aid",
        across=("recipient",),
        period=Period(start=2020, end=2023),
        source="DAC",
    )
    assert EvidenceIssueCode.SOURCE_MISMATCH not in [
        f.code for f in check_evidence(requirement, evidence_from_source(source))
    ]


async def test_fetching_an_alternative_preserves_recorded_source_confirmation():
    envelope = copy.deepcopy(FRANCE_KENYA)
    mention = {"text": "DAC", "ref": None, "polarity": "include"}
    envelope["parts"][0]["restatement"] = {"sources": [mention]}
    surface = FakeSurface(
        ToolCallResult(is_error=False, **search_result_parts(_net_family_envelope()))
    )
    tools = build_orchestrator_tools(
        resolver=_resolver(resolver_digest(envelope)),
        executor=FakeExecutor(),
        definition_surface=surface,
    )
    by_name = {fn.__name__: fn for fn in tools}
    ctx = SimpleNamespace(state={"surface": "studio"}, invocation_id="inv_1")
    conversation = ConversationState(conversation_id="c1")
    token = current_recorded_identity.set(conversation.recorded_identity)
    try:
        await by_name["search"](ctx, [{"variable": "aid", "sources": [{"text": "DAC"}]}])
        conversation.turn_specs = dict(ctx.state["orchestrator:specs"])
        before = conversation.recorded_identity(FRANCE_KENYA_NET)
        assert before is not None
        assert [m.model_dump() for m in before.resolved_sources] == [mention]

        await by_name["fetch"](FRANCE_KENYA_NET, ctx)

        block = ctx.state["orchestrator:specs"][FRANCE_KENYA_NET]
        assert [m.model_dump() for m in source_from_spec_block(block).resolved_sources] == [mention]
        conversation.turn_specs = dict(ctx.state["orchestrator:specs"])
        after = conversation.recorded_identity(FRANCE_KENYA_NET)
        assert after is not None
        assert after.resolved_sources == before.resolved_sources
    finally:
        current_recorded_identity.reset(token)


class _Counted:
    """A frame store that counts the frames it is asked to open."""

    def __init__(self, store: Any) -> None:
        self.store = store
        self.opens = 0

    async def open(self, ref: Any) -> Any:
        self.opens += 1
        return await self.store.open(ref)


class MixedFrames(HeldFrames):
    """`HeldFrames` of nine parties holding 2023 and 2024, and twelve group
    totals each holding one year of its own, so no year is held by half of
    all twenty-one members while every party holds 2024."""

    def __init__(self) -> None:
        totals = frozenset(f"donor/{number:02d}" for number in range(9, 21))
        super().__init__(providers=21, totals=totals, carried=True)

    def rows(self) -> list[dict[str, Any]]:
        rows = super().rows()
        parties = [row for row in rows if row[self.role] not in self.totals]
        totals = [row for row in rows[1::2] if row[self.role] in self.totals]
        return parties + [
            {**row, "time_period": str(2000 + number)} for number, row in enumerate(totals)
        ]


class MethodFrames(HeldFrames):
    """`HeldFrames` with each provider's aid under two measurement methods,
    which dfo-apis keys as two series."""

    def rows(self) -> list[dict[str, Any]]:
        return [
            {**row, "measurementMethod": method}
            for row in super().rows()
            for method in ("OECD", "Imputed")
        ]


class DerivedFrames(HeldFrames):
    """`HeldFrames` standing for a frame the analysis runner computed, whose
    rows carry no kind cells and whose members' kinds are not read."""


# A view declaring the two measurement methods `MethodFrames` holds.
_METHOD_VIEW = copy.deepcopy(WIDE_VIEW)
_METHOD_VIEW["dimensions"].append(
    {"name": "measurementMethod", "values": ["OECD", "Imputed"], "plural": True, "total_count": 2}
)
_ROLLING_VIEW = {**WIDE_VIEW, "rolling": True}


class MethodPairFrames(PairFrames):
    """`PairFrames` with each pair's aid under two measurement methods."""

    def rows(self) -> list[dict[str, Any]]:
        return [
            {**row, "measurementMethod": method}
            for row in super().rows()
            for method in ("OECD", "Imputed")
        ]


_FIFTEEN_PAIRS = [
    (number, recipient)
    for number in range(3)
    for recipient in ("country/KEN", "country/ETH", "country/GHA", "country/UGA", "country/TZA")
]


class MeasureFrames(HeldFrames):
    """`HeldFrames` with each provider's aid under five measures."""

    def rows(self) -> list[dict[str, Any]]:
        return [
            {**row, "variableMeasured": f"dcid:aid{measure}"}
            for row in super().rows()
            for measure in range(5)
        ]


# A view listing twelve measures, of which a frame holds `HeldFrames`'
# members as measures.
_MEASURE_VIEW = copy.deepcopy(WIDE_VIEW)
_MEASURE_VIEW["dimensions"][1] = {
    "name": "variableMeasured",
    "values": [],
    "plural": True,
    "total_count": 12,
}
_TWELVE_MEASURES = {
    "financingRecipient": "country/KEN",
    "variableMeasured": [f"donor/{number:02d}" for number in range(12)],
}
# `WIDE_VIEW` with the five measures `MeasureFrames` holds.
_FIVE_MEASURE_VIEW = copy.deepcopy(WIDE_VIEW)
_FIVE_MEASURE_VIEW["dimensions"].append(
    {"name": "variableMeasured", "values": [], "plural": True, "total_count": 5}
)

_NINE_PAIRS = [
    (number, recipient)
    for number in range(3)
    for recipient in ("country/KEN", "country/ETH", "country/GHA")
]
_LIMIT = "MAX_LINE_CHART_SERIES"
_TOP_EIGHT = {
    "financingRecipient": "country/KEN",
    "financingProvider": [f"donor/{number:02d}" for number in range(19, 11, -1)],
}

_TWENTY = [f"donor/{number:02d}" for number in range(20)]
# A view listing all ten of its providers and nothing else, so the one select
# naming every series of an empty select lists ten.
_TEN_LISTED_VIEW = copy.deepcopy(WIDE_VIEW)
_TEN_LISTED_VIEW["dimensions"][1]["values"] = [f"donor/{number:02d}" for number in range(10)]
_LISTED = {"financingRecipient": "country/KEN", "financingProvider": _TWENTY}


@dataclass(frozen=True)
class _Platform:
    """The end of a chart dfo-apis refused that no cut drew, by the code and
    limit of the refusal, which reaches the model as the platform gave it."""

    code: str
    limit: str | None


_PLATFORM = _Platform("resource_exhausted", "MAX_LINE_CHART_SERIES")


@dataclass(frozen=True)
class _Cut:
    """One way a chart over its limit can end: the store, the providers the
    view counts, the view, the select, then the select drawn or the
    platform's refusal, the writes made, and the frames opened."""

    make: Callable[[], Any]
    providers: int
    view: dict[str, Any] | None
    select: dict[str, Any]
    outcome: dict[str, Any] | _Platform
    writes: int
    opens: int
    type_: str = "line_chart"
    # The select of the write the platform refused, where the cell pins it.
    refused_select: dict[str, Any] | None = None


_SIX_OF_TWELVE = {
    "financingRecipient": "country/KEN",
    "financingProvider": [f"donor/{number:02d}" for number in range(12)],
}
# `_UNIT_VIEW` listing each of thirty providers, so an empty select's one
# covering select lists them.
_THIRTY_LISTED_UNIT_VIEW = copy.deepcopy(_UNIT_VIEW)
_THIRTY_LISTED_UNIT_VIEW["dimensions"][1]["values"] = [
    f"donor/{number:02d}" for number in range(30)
]
_LINE_LIMIT = _Platform("resource_exhausted", _LIMIT)
_EMPTY = _Platform("binding_invalid", None)


class ContinuedFrames(HeldFrames):
    """`HeldFrames` whose 2024 rows are estimates continuing each provider's
    observed 2023 series, under another measurement method."""

    def rows(self) -> list[dict[str, Any]]:
        return [
            {**row, "measurementMethod": "est" if row["time_period"] == "2024" else "obs"}
            for row in super().rows()
        ]


# A view of `ContinuedFrames`: apis collapses each estimate into the series it
# continues, so the method splits no series and is not plural.
_CONTINUED_VIEW = copy.deepcopy(WIDE_VIEW)
_CONTINUED_VIEW["dimensions"].append(
    {"name": "measurementMethod", "values": ["obs", "est"], "plural": False, "total_count": 2}
)


def _holds_rows(value: Any) -> bool:
    """Whether `value` carries a frame row: a mapping with a period and a
    value, in a container or a dataclass field."""
    if isinstance(value, dict):
        return {"time_period", "value"} <= value.keys() or any(
            _holds_rows(item) for item in value.values()
        )
    if isinstance(value, list | tuple):
        return any(_holds_rows(item) for item in value)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return any(_holds_rows(getattr(value, field.name)) for field in dataclasses.fields(value))
    return False


# The select resolves past the chart's limit when written, names no dimension
# (`check_select_nonempty`, which the fake platform refuses and the harness
# does not see), lists more than eight distinct values on a line chart (the
# catalogue's `maxItems`), or is estimated past 24 series (the harness cuts
# those two before any write).
_CUT_CASES = {
    "cut-succeeds": _Cut(
        lambda: HeldFrames(providers=20, carried=True),
        20,
        None,
        {"financingRecipient": "country/KEN"},
        _TOP_EIGHT,
        2,
        1,
    ),
    "unrankable": _Cut(
        lambda: TwoUnitFrames(providers=20, carried=True),
        20,
        _UNIT_VIEW,
        {"financingRecipient": "country/KEN"},
        _PLATFORM,
        1,
        1,
    ),
    "kinds-unread": _Cut(
        lambda: KindlessFrames(providers=20),
        20,
        None,
        {"financingRecipient": "country/KEN"},
        _PLATFORM,
        1,
        2,
    ),
    "table-kinds-unread": _Cut(
        lambda: KindlessFrames(providers=20),
        20,
        _MONTHLY_VIEW,
        {"financingRecipient": "country/KEN"},
        _Platform("resource_exhausted", "MAX_TABLE_ROWS"),
        1,
        2,
        type_="table",
    ),
    "no-shared-period": _Cut(
        lambda: ScatteredFrames(providers=20),
        20,
        None,
        {"financingRecipient": "country/KEN"},
        _PLATFORM,
        1,
        3,
    ),
    "two-multiplying-dimensions": _Cut(
        lambda: PairFrames(_NINE_PAIRS),
        3,
        _pair_view(providers=3, recipients=3),
        {},
        _PLATFORM,
        1,
        1,
    ),
    "listed-over-limit": _Cut(
        lambda: HeldFrames(providers=20, carried=True),
        20,
        None,
        _LISTED,
        _TOP_EIGHT,
        1,
        1,
    ),
    "listed-over-limit-unrankable": _Cut(
        lambda: TwoUnitFrames(providers=20, carried=True),
        20,
        _UNIT_VIEW,
        _LISTED,
        _LINE_LIMIT,
        1,
        1,
    ),
    "listed-mostly-unheld": _Cut(
        lambda: HeldFrames(providers=6, carried=True),
        12,
        None,
        _SIX_OF_TWELVE,
        {
            "financingRecipient": "country/KEN",
            "financingProvider": [f"donor/{number:02d}" for number in range(5, -1, -1)],
        },
        1,
        1,
    ),
    "measures-listed-mostly-unheld": _Cut(
        lambda: HeldFrames(providers=6, role="variableMeasured"),
        12,
        _MEASURE_VIEW,
        _TWELVE_MEASURES,
        {
            "financingRecipient": "country/KEN",
            "variableMeasured": [f"donor/{number:02d}" for number in range(5, -1, -1)],
        },
        1,
        1,
    ),
    "measures-listed-ten-held": _Cut(
        lambda: HeldFrames(providers=10, role="variableMeasured"),
        12,
        _MEASURE_VIEW,
        _TWELVE_MEASURES,
        {
            "financingRecipient": "country/KEN",
            "variableMeasured": [f"donor/{number:02d}" for number in range(9, 1, -1)],
        },
        1,
        1,
    ),
    "measures-and-providers-unbound": _Cut(
        lambda: MeasureFrames(providers=6, carried=True),
        6,
        _FIVE_MEASURE_VIEW,
        {"financingRecipient": "country/KEN"},
        _LINE_LIMIT,
        1,
        0,
    ),
    "estimate-over-24": _Cut(
        lambda: HeldFrames(providers=30, carried=True),
        30,
        None,
        {"financingRecipient": "country/KEN"},
        {
            "financingRecipient": "country/KEN",
            "financingProvider": [f"donor/{number:02d}" for number in range(29, 21, -1)],
        },
        1,
        1,
    ),
    "estimate-over-24-unrankable": _Cut(
        lambda: TwoUnitFrames(providers=30, carried=True),
        30,
        _UNIT_VIEW,
        {"financingRecipient": "country/KEN"},
        _LINE_LIMIT,
        1,
        1,
    ),
    "two-dimensions-over-24": _Cut(
        lambda: PairFrames(_NINE_PAIRS),
        5,
        _pair_view(providers=5, recipients=6),
        {"stat_var": "dcid:aid"},
        _LINE_LIMIT,
        1,
        0,
    ),
    "empty-select-over-limit": _Cut(
        lambda: HeldFrames(providers=10, carried=True),
        10,
        _TEN_LISTED_VIEW,
        {},
        {"financingProvider": [f"donor/{number:02d}" for number in range(9, 1, -1)]},
        1,
        2,
    ),
    # The covering select of ten providers is no cut, so the model's empty
    # select is written and dfo-apis' answer to it relayed.
    "empty-select-over-limit-unrankable": _Cut(
        lambda: TwoUnitFrames(providers=10, carried=True),
        10,
        _TEN_LISTED_VIEW,
        {},
        _EMPTY,
        1,
        2,
        refused_select={},
    ),
    "continued-estimates": _Cut(
        lambda: ContinuedFrames(providers=20, carried=True),
        20,
        _CONTINUED_VIEW,
        {"financingRecipient": "country/KEN"},
        _TOP_EIGHT,
        2,
        1,
    ),
    "rank-positions": _Cut(
        lambda: HeldFrames(
            providers=20,
            carried=True,
            trace=SimpleNamespace(
                applied_policy=SimpleNamespace(detail={"derived": {"ordinal": True}})
            ),
        ),
        20,
        None,
        {"financingRecipient": "country/KEN"},
        _PLATFORM,
        1,
        1,
    ),
    "aggregate-among-members": _Cut(
        MixedFrames,
        21,
        None,
        {"financingRecipient": "country/KEN"},
        {
            "financingRecipient": "country/KEN",
            "financingProvider": [f"donor/{number:02d}" for number in range(8, 0, -1)],
        },
        2,
        1,
    ),
    "methods-split-series": _Cut(
        lambda: MethodFrames(providers=5, carried=True),
        5,
        _METHOD_VIEW,
        {"financingRecipient": "country/KEN"},
        _PLATFORM,
        1,
        0,
    ),
    "undeclared-split": _Cut(
        lambda: MethodFrames(providers=5, carried=True),
        5,
        None,
        {"financingRecipient": "country/KEN"},
        _PLATFORM,
        1,
        0,
    ),
    "derived-frame": _Cut(
        lambda: DerivedFrames(providers=20),
        20,
        None,
        {"financingRecipient": "country/KEN"},
        _PLATFORM,
        1,
        1,
    ),
    "period-moves": _Cut(
        lambda: HeldFrames(providers=20, carried=True),
        20,
        _ROLLING_VIEW,
        {"financingRecipient": "country/KEN"},
        _PLATFORM,
        1,
        0,
    ),
    "frame-unreadable": _Cut(
        UnreadableFrames,
        20,
        None,
        {"financingRecipient": "country/KEN"},
        _PLATFORM,
        1,
        1,
    ),
    "pairs-split-by-method": _Cut(
        lambda: MethodPairFrames(_FIFTEEN_PAIRS),
        3,
        _pair_view(providers=3, recipients=5),
        {},
        _PLATFORM,
        1,
        1,
    ),
    "table-empty-select-over-24": _Cut(
        lambda: TwoUnitFrames(providers=30, carried=True),
        30,
        _THIRTY_LISTED_UNIT_VIEW,
        {},
        _EMPTY,
        1,
        2,
        type_="table",
        refused_select={},
    ),
}


@pytest.mark.parametrize("case", _CUT_CASES)
async def test_a_chart_over_its_limit_ends_cut_or_refused(
    monkeypatch: pytest.MonkeyPatch, case: str
) -> None:
    """A chart whose select resolves past its limit is drawn cut to its
    largest members, or is written as it is and ends with dfo-apis' refusal,
    relayed exactly as presentation-mcp gave it. The harness refuses nothing
    for its series before a write; a cut is the one thing it does first. No
    read of a frame hands its rows back out of the row budget's claim (`_frame_work`)."""
    cell = _CUT_CASES[case]
    store = _Counted(cell.make())
    read = orchestrator_tools._frame_work
    returned: list[Any] = []

    async def reading(ref: Any, work: Any) -> Any:
        returned.append(await read(ref, work))
        return returned[-1]

    monkeypatch.setattr(orchestrator_tools, "_frame_work", reading)
    refusal = _LINE_SERIES_REFUSAL if cell.type_ == "line_chart" else _TABLE_ROWS_REFUSAL
    monkeypatch.setattr(orchestrator_tools, "read_derived_marker", _marker_of)
    if isinstance(store.store, DerivedFrames):
        monkeypatch.setattr(orchestrator_tools, "OP_SPEC_HASH_PREFIX", WIDE_REF["spec_id"])

    executor, result = await _draw_capped(
        monkeypatch,
        refused=cell.type_,
        refusal=refusal,
        view=cell.view,
        store=store,
        providers=cell.providers,
        select_json=json.dumps(cell.select),
    )

    (entry,) = result["charts"]
    assert _writes(executor) == cell.writes
    if isinstance(cell.outcome, dict):
        assert "error" not in entry
        assert _intents(executor)[-1]["select"] == cell.outcome
    else:
        refused = _intents(executor)[-1]["select"]
        # A refused select the harness wrote in place of the model's own, a
        # cut, rides beside the platform's body.
        harness_written = {} if refused == cell.select else {"written_select": refused}
        assert entry == _relayed(executor.refusals[-1]) | harness_written
        assert (entry["code"], entry.get("limit_name")) == (cell.outcome.code, cell.outcome.limit)
        if cell.refused_select is not None:
            assert refused == cell.refused_select
    assert store.opens == cell.opens
    assert not any(_holds_rows(value) for value in returned)


class ScriptedWrites(WideFrameExecutor):
    """`WideFrameExecutor` whose writes answer in turn as `answers` gives
    them: a presentation-mcp refusal as `harness.mcp_client` reads it, an
    exception raised, or None to accept the write."""

    def __init__(self, answers: list[Refusal | Exception | None], **wide: Any) -> None:
        super().__init__(**wide)
        self.answers = answers

    async def __call__(self, *, tool: str, args: dict[str, Any], on_progress: Any = None):
        if tool != "write_component":
            return await super().__call__(tool=tool, args=args, on_progress=on_progress)
        answer = self.answers.pop(0) if self.answers else None
        if answer is None:
            return await super().__call__(tool=tool, args=args, on_progress=on_progress)
        self.calls.append((tool, args))
        if isinstance(answer, Exception):
            raise answer
        result = presentation_server._refusal_result(answer, tool="write_component")
        return _presentation_refusal_outcome(
            result.structured_content, result.meta, fallback_text=""
        )


# dfo-apis' refusal of a write whose select binds a value the frame lacks.
_BINDING_REFUSAL = Refusal(
    error="binding_invalid",
    retry=RetryMode.CORRECT_AND_RETRY,
    detail="financingProvider binds a value the frame does not hold",
    pointer="/select/financingProvider",
)


async def _draw_scripted(
    monkeypatch: pytest.MonkeyPatch,
    answers: list[Refusal | Exception | None],
    *charts: Any,
    store: Any = None,
    view: dict[str, Any] | None = None,
    providers: int = 20,
) -> tuple[ScriptedWrites, dict[str, Any]]:
    """Draw `charts` of `providers` providers' aid to Kenya, held in `store`,
    the platform answering each write as `answers` says."""
    held = store or HeldFrames(providers=providers, carried=True)
    monkeypatch.setattr(orchestrator_tools, "get_frame_store", lambda: held)
    executor = ScriptedWrites(answers, providers=providers, view=copy.deepcopy(view or WIDE_VIEW))
    tools, ctx = _tools(executor=executor)
    await tools["fetch"](SPEC_LIFE, ctx)
    return executor, await tools["make_charts"](list(charts), ctx)


async def test_a_cut_write_the_platform_refuses_again_ends_with_that_refusal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A line chart refused over its lines is cut and written once more. The
    platform refuses the cut as well, and the chart ends with that second
    refusal: no third write."""
    executor, result = await _draw_scripted(
        monkeypatch,
        [_LINE_SERIES_REFUSAL, _BINDING_REFUSAL],
        _wide_chart(type="line_chart"),
    )

    first, cut = _intents(executor)
    assert first["select"] == {"financingRecipient": "country/KEN"}
    assert len(cut["select"]["financingProvider"]) == LINE_CHART_MAX_VALUES
    (entry,) = result["charts"]
    # The platform's body as it gave it, beside the cut it judged.
    assert entry == _relayed(_BINDING_REFUSAL) | {"written_select": cut["select"]}


@pytest.mark.parametrize("type_", ["bar_chart", "table"])
async def test_a_bar_chart_or_table_refused_on_the_series_estimate_is_written_once(
    monkeypatch: pytest.MonkeyPatch, type_: str
) -> None:
    """dfo-apis refuses a bar chart or a table over its series estimate as
    `resource_exhausted` on `MAX_SERIES_PER_COMPONENT`. The select leaves
    unbound a dimension the view counts at 20, no more than the 24 members the
    chart draws, so a cut would list the same members and meet the same
    refusal. The chart is written once, without a frame read."""
    store = CountedFrames(providers=20, carried=True)
    executor, result = await _draw_scripted(
        monkeypatch, [_COMPONENT_SERIES_REFUSAL], _wide_chart(type=type_), store=store
    )

    (written,) = _intents(executor)
    assert written["select"] == {"financingRecipient": "country/KEN"}
    (entry,) = result["charts"]
    assert entry == _relayed(_COMPONENT_SERIES_REFUSAL)
    assert store.opens == 0


async def test_a_line_chart_leaving_unbound_a_dimension_of_eight_is_written_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A line chart refused over its lines that leaves unbound a dimension the
    view counts at eight, the lines it draws, is cut to as many members, so it
    is not written again and the frame is not read."""
    store = CountedFrames(providers=8, carried=True)
    executor, result = await _draw_scripted(
        monkeypatch,
        [_LINE_SERIES_REFUSAL],
        _wide_chart(type="line_chart"),
        store=store,
        providers=8,
    )

    (written,) = _intents(executor)
    assert written["select"] == {"financingRecipient": "country/KEN"}
    (entry,) = result["charts"]
    assert entry == _relayed(_LINE_SERIES_REFUSAL)
    assert store.opens == 0


_FIVE = {
    "financingRecipient": "country/KEN",
    "financingProvider": [f"donor/{number:02d}" for number in range(5)],
}


@pytest.mark.parametrize(
    ("type_", "refusal", "select", "view"),
    [
        pytest.param("line_chart", _LINE_SERIES_REFUSAL, _TOP_EIGHT, WIDE_VIEW, id="same-order"),
        pytest.param(
            "line_chart",
            _LINE_SERIES_REFUSAL,
            {**_TOP_EIGHT, "financingProvider": _TOP_EIGHT["financingProvider"][::-1]},
            WIDE_VIEW,
            id="reordered",
        ),
        pytest.param(
            "line_chart",
            _LINE_SERIES_REFUSAL,
            {
                **_TOP_EIGHT,
                "financingProvider": [
                    *_TOP_EIGHT["financingProvider"],
                    _TOP_EIGHT["financingProvider"][0],
                ],
            },
            WIDE_VIEW,
            id="repeated",
        ),
        pytest.param(
            "line_chart",
            _LINE_SERIES_REFUSAL,
            {
                **_TOP_EIGHT,
                "financingProvider": [*_TOP_EIGHT["financingProvider"][:-1], "donor/11"],
            },
            WIDE_VIEW,
            id="same-size-swap",
        ),
        pytest.param("table", _TABLE_ROWS_REFUSAL, _FIVE, _MONTHLY_VIEW, id="table-of-five"),
    ],
)
async def test_a_refusal_the_cut_cannot_change_is_written_once(
    monkeypatch: pytest.MonkeyPatch,
    type_: str,
    refusal: Refusal,
    select: dict[str, Any],
    view: dict[str, Any],
) -> None:
    """A line chart already listing eight providers, in any order, with repeats,
    or with the weakest swapped for another, refused over its lines, or a table
    listing five, refused over its rows, is cut to as many members, so it is
    not written again and the frame is not read."""
    store = CountedFrames(providers=20, carried=True)
    executor, result = await _draw_scripted(
        monkeypatch,
        [refusal],
        _wide_chart(type=type_, select_json=json.dumps(select)),
        store=store,
        view=view,
    )

    (written,) = _intents(executor)
    assert written["select"] == select
    (entry,) = result["charts"]
    assert entry == _relayed(refusal)
    assert store.opens == 0


async def test_a_cut_written_first_and_refused_carries_the_select_written(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A line chart listing twenty providers is cut before its first write.
    The platform refuses the cut, and the model reads that refusal beside the
    select the harness wrote in place of its own."""
    executor, result = await _draw_scripted(
        monkeypatch,
        [_BINDING_REFUSAL],
        _wide_chart(type="line_chart", select_json=json.dumps(_LISTED)),
    )

    (cut,) = _intents(executor)
    assert len(cut["select"]["financingProvider"]) == LINE_CHART_MAX_VALUES
    (entry,) = result["charts"]
    assert entry == _relayed(_BINDING_REFUSAL) | {"written_select": cut["select"]}


async def test_a_cut_write_that_raises_leaves_the_charts_after_it_undrawn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A line chart refused over its lines is cut, and writing the cut raises.
    It may be on the canvas, so the model is told not to draw it again, and
    the chart after it in the call is not drawn."""
    executor, result = await _draw_scripted(
        monkeypatch,
        [_LINE_SERIES_REFUSAL, TimeoutError("write timed out")],
        _wide_chart(type="line_chart"),
        _wide_chart(type="table", title="Aid to Kenya by provider"),
    )

    assert _writes(executor) == 2
    (entry,) = result["charts"]
    assert entry["code"] == HarnessCode.WRITE_UNCONFIRMED


# dfo-apis' refusal of a component estimated past its 24 series.
_COMPONENT_SERIES_REFUSAL = Refusal(
    error="resource_exhausted",
    retry=RetryMode.CORRECT_AND_RETRY,
    detail=(
        "selectors expand to an estimated 30 series, over the maximum of 24: "
        "financingProvider (30 values) is the series dimension; bind it to at most 24 values"
    ),
    pointer="/select",
    limit_name="MAX_SERIES_PER_COMPONENT",
)


# dfo-apis' refusal of a line chart whose window passes 500 periods.
_POINTS_REFUSAL = Refusal(
    error="resource_exhausted",
    retry=RetryMode.CORRECT_AND_RETRY,
    detail="window spans 1025 periods at this frame's grain, over the maximum of 500 points per series",
    pointer="/window",
    limit_name="MAX_POINTS_PER_SERIES",
)


@pytest.mark.parametrize(
    ("type_", "refusal"),
    [
        pytest.param("line_chart", _POINTS_REFUSAL, id="line-chart-points"),
        pytest.param("bar_chart", _TABLE_ROWS_REFUSAL, id="bar-chart-table-rows"),
    ],
)
async def test_a_refusal_on_another_limit_is_relayed_as_the_platform_gave_it(
    monkeypatch: pytest.MonkeyPatch, type_: str, refusal: Refusal
) -> None:
    """A write dfo-apis refuses on a limit other than the one a cut meets for
    the chart type is not cut: the model gets the platform's code, retry mode
    and limit, and the chart is written once."""
    executor, result = await _draw_capped(monkeypatch, refused=type_, refusal=refusal)

    assert _writes(executor) == 1
    (entry,) = result["charts"]
    assert (*_stated(entry), entry["limit_name"]) == (
        "resource_exhausted",
        "correct_and_retry",
        refusal.limit_name,
    )
