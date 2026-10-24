"""The orchestrator's tools against a fake resolver, executor and tool surface.

`usable_component` is `harness.wire`'s own check, tested with the wire
contract; here it is stubbed so a fake component can stand in for a written
one.
"""

from __future__ import annotations

import asyncio
import copy
import json
import logging
import re
import threading
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx2
import pyarrow as pa
import pytest
from common.materialisation import FrameTooLarge
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
from data_mcp.search_tool import MAX_PARTS
from data_mcp.tool_errors import ToolErrorCode
from dfo_frame_grammar import Frame
from google.adk.sessions.state import State
from google.adk.tools import FunctionTool
from harness import chart_binding, orchestrator_tools, planner_seam
from harness.chart_binding import (
    LINE_CHART_MAX_VALUES,
    MAX_SERIES_PER_COMPONENT,
    empty_select_refusal,
)
from harness.dfo_faq import CANNOT_DESCRIBE_DFO, DFO_FAQ_NOT_READ
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
    store_search,
)
from harness.orchestrator_tools import (
    HANDED_OFF,
    Chart,
    build_orchestrator_tools,
)
from harness.reply_figures import card_result, drawn_series
from harness.rolling_cards import CardWithheld, RollingCards
from harness.studio_state import ConversationState, land_component, remove_cards
from harness.tool_contracts import (
    HarnessCode,
    ResolverError,
    graph_envelope,
    reading_facts,
    resolver_digest,
)
from harness.turn_records import SearchReading, loose_match_notes, render_readings
from mcp import MCPError
from mcp.types import CONNECTION_CLOSED, REQUEST_TIMEOUT, ErrorData
from orchestrator.agent import INSTRUCTION
from planner.plan import (
    MAX_ASK_CHARS,
    MAX_PLAN_ASKS,
    AskBreakdown,
    DatasetRequirement,
    current_earlier_block,
    current_recorded_identity,
)
from planner.records import Period
from submission import service as submission_service

from . import drawn_cards
from . import test_rolling_cards as rolling_scene
from .resolved_revision import resolved_revision
from .search_results import search_result_parts
from .test_turn_records import _top_donors

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
        return {"outcome": "ok", "component": self.written}

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
    remove_cards=None,
):
    tools = build_orchestrator_tools(
        resolver=resolver or _resolver(_digest()),
        executor=executor or FakeExecutor(),
        surface=surface,
        dfo_faq_surface=dfo_faq_surface,
        recall_part=recall_part,
        canvas_series=canvas_series,
        remove_cards=remove_cards,
    )
    by_name = {fn.__name__: fn for fn in tools}
    return by_name, SimpleNamespace(
        state={"surface": surface_name},
        invocation_id="inv_1",
        actions=SimpleNamespace(skip_summarization=None),
    )


@pytest.fixture(autouse=True)
def _usable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(orchestrator_tools, "usable_component", lambda component: True)


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
    assert f"at most {MAX_SERIES_PER_COMPONENT} series" in properties["select_json"]["description"]


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


def test_a_chart_title_carries_its_period_unless_the_period_moves():
    properties = Chart.model_json_schema()["properties"]
    title = properties["title"]["description"]

    assert "the period it shows" in title
    assert "a stat card's title carries the period" in title
    assert "rolling: true" in title
    assert "latest_periods" in title
    assert "its title leaves out the years it shows" in title
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
        assert (
            FunctionTool(fn)._get_declaration() == FunctionTool(fn.__wrapped__)._get_declaration()
        )


# --- argument validation ----------------------------------------------------

_NGA = json.dumps({"observationAbout": "country/NGA"})


async def _search_role_on_the_part(tools: dict[str, Any], ctx: Any) -> dict[str, Any]:
    part = {"variable": "aid", "role": "to", "entities": [{"text": "Kenya"}]}
    return await tools["search"]("aid to Kenya", ctx, [part])


async def _search_string_parts(tools: dict[str, Any], ctx: Any) -> dict[str, Any]:
    return await tools["search"]("France's aid share", ctx, ["France's total gross ODA"])


async def _search_too_many_entities(tools: dict[str, Any], ctx: Any) -> dict[str, Any]:
    part = {"variable": "population", "entities": [{"text": f"place {i}"} for i in range(17)]}
    return await tools["search"]("population", ctx, [part])


async def _search_entity_text_too_long(tools: dict[str, Any], ctx: Any) -> dict[str, Any]:
    part = {"variable": "population", "entities": [{"text": "k" * 129}]}
    return await tools["search"]("population", ctx, [part])


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


async def test_calls_with_invalid_arguments_stop_at_their_budget():
    tools, ctx = _tools()

    for _ in range(BUDGETS["invalid_arguments"]):
        rejected = await _search_role_on_the_part(tools, ctx)
        assert "invalid" in rejected
    spent = await _search_role_on_the_part(tools, ctx)

    assert "invalid" not in spent
    assert _stated(spent) == (HarnessCode.TOOL_BUDGET_SPENT, "do_not_retry")
    assert ctx.state[ORCHESTRATOR_CALLS_KEY] == {"invalid_arguments": BUDGETS["invalid_arguments"]}


async def test_adk_function_tool_returns_the_argument_error_for_a_part_level_role():
    resolver = _resolver(_digest())
    tools, ctx = _tools(resolver)
    part = {"variable": "aid", "role": "to", "entities": [{"text": "Kenya"}]}

    found = await FunctionTool(tools["search"]).run_async(
        args={"query": "aid to Kenya", "parts": [part]}, tool_context=ctx
    )

    assert found["invalid"] == [{"loc": "parts.0.role", "type": "extra_forbidden"}]
    assert '{"text": "Kenya", "role": "to"}' in found["error"]
    assert not resolver.calls
    assert ctx.state[ORCHESTRATOR_CALLS_KEY] == {"invalid_arguments": 1}


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
        "life expectancy and GDP in Nigeria",
        ctx,
        [
            {"variable": "life expectancy", "entities": [{"text": "Nigeria"}]},
            {"variable": "GDP", "entities": [{"text": "Africa", "members": True}]},
        ],
    )

    assert len(resolver.calls) == 1
    assert resolver.calls[0]["question"] == "life expectancy and GDP in Nigeria"
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


async def test_search_without_parts_asks_search_to_read_the_query():
    resolver = _resolver(_digest())
    tools, ctx = _tools(resolver)

    await tools["search"]("aid to Kenya from all official donors", ctx)

    assert resolver.calls == [
        {"question": "aid to Kenya from all official donors", "parts": [], "raw_text": True}
    ]


@pytest.mark.parametrize("parts", [None, []])
async def test_as_text_without_parts_reads_the_query_without_attaching_asked_roles(parts):
    resolver = _resolver(_digest({"variable": "aid", "status": "no_data"}))
    tools, ctx = _tools(resolver)

    found = await tools["search"]("aid to Kenya", ctx, parts, as_text=True)

    assert "error" not in found
    assert resolver.calls == [{"question": "aid to Kenya", "parts": [], "raw_text": True}]
    assert not ctx.state["orchestrator:searches"][0]["asked_roles"]


async def test_as_text_refuses_multiple_parts_without_searching_or_storing_readings():
    resolver = _resolver(_digest())
    tools, ctx = _tools(resolver)
    parts = [
        {"variable": "aid", "entities": [{"text": "Kenya", "role": "to"}], "time": "2020"},
        {"variable": "aid", "entities": [{"text": "France", "role": "from"}], "time": "2024"},
    ]

    for _ in range(BUDGETS["invalid_arguments"]):
        refused = await tools["search"]("aid to Kenya and from France", ctx, parts, as_text=True)
        assert refused["invalid"] == [{"loc": "parts", "type": "too_long"}]
        assert "retry each part separately" in refused["error"]
    spent = await tools["search"]("aid to Kenya and from France", ctx, parts, as_text=True)

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

    found = await tools["search"]("How much ODA did Kenya receive in 2024?", ctx)

    assert resolver.calls[0]["raw_text"] is True
    assert found["parts"][0]["status"] == "definite"
    assert found["parts"][0]["warnings"] == [warning]
    assert ctx.state["orchestrator:specs"] == {SPEC_LIFE: SINGLE_BLOCK}


async def test_a_search_as_text_keeps_the_parties_its_part_asked_for():
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

    await tools["search"]("France's climate finance to Kenya", ctx, [part], as_text=True)

    assert resolver.calls == [
        {"question": "France's climate finance to Kenya", "parts": [], "raw_text": True}
    ]
    assert ctx.state["orchestrator:searches"][0]["asked_roles"] == ["donor", "recipient"]


async def test_search_reports_a_result_with_no_parts_by_its_status():
    tools, ctx = _tools(
        _resolver(_digest(status="unsupported", warnings=[{"message": "no measure"}]))
    )

    found = await tools["search"]("what data do you have?", ctx, [{"variable": "anything"}])

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

    await tools["search"]("DT.TDS.DEGG.CD by creditor", ctx, [part])

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

    await tools["search"]("Ghana's debt service, not from IDS", ctx, [part])

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

    first = await tools["search"]("q", ctx, [{"variable": "x"}])
    # The service's code and retry mode come with its message.
    assert first == {
        "error": "search is down",
        "code": "rate_limited",
        "retry_mode": "retry_after_delay",
    }

    # A failure that states no code is given the harness's own.
    bare, bare_ctx = _tools(_resolver(ResolverError("search carried no envelope")))
    uncoded = await bare["search"]("q", bare_ctx, [{"variable": "x"}])
    assert _stated(uncoded) == ("unspecified", "do_not_retry")

    for _ in range(BUDGETS["search"] - 1):
        await tools["search"]("q", ctx, [{"variable": "x"}])
    spent = await tools["search"]("q", ctx, [{"variable": "x"}])

    assert "limit" in spent["error"]
    assert _stated(spent) == (HarnessCode.TOOL_BUDGET_SPENT, "do_not_retry")


@pytest.mark.parametrize("surface_name", ["studio", "download"])
async def test_every_part_can_retry_as_text_after_a_corrected_place(surface_name):
    resolver = _resolver(_digest())
    tools, ctx = _tools(resolver, surface_name=surface_name)
    parts = [
        {"variable": f"measure {index}", "entities": [{"text": "Keyna"}]}
        for index in range(MAX_PARTS)
    ]

    results = [await tools["search"]("measures in Keyna", ctx, parts)]
    corrected = [{**part, "entities": [{"text": "Kenya"}]} for part in parts]
    results.append(await tools["search"]("measures in Kenya", ctx, corrected))
    results.extend(
        [
            await tools["search"](f"{part['variable']} in Keyna", ctx, [part], as_text=True)
            for part in parts
        ]
    )

    assert all("error" not in result for result in results)
    assert len(resolver.calls) == MAX_PARTS + 2
    assert all(call["raw_text"] for call in resolver.calls[2:])
    assert all(not call["parts"] for call in resolver.calls[2:])

    spent = await tools["search"]("another retry", ctx, [parts[0]], as_text=True)
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

    found = await tools["search"]("education aid to Kenya", ctx, [{"variable": "education aid"}])

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


async def test_search_shows_no_year_counts_of_a_part_searched_for_the_latest_period():
    envelope = _recorded_envelope()
    spec = envelope["parts"][0]["spec"]
    spec["definition"]["identity"]["requested_window"] = {"kind": "latest"}
    tools, ctx = _tools(_resolver(resolver_digest(envelope)))

    found = await tools["search"]("education aid to Kenya", ctx, [{"variable": "education aid"}])

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

    await tools["search"]("q", ctx, [{"variable": "x"}])
    await tools["search"]("q", ctx, [{"variable": "x"}])

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
    may pin with no second search."""
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
    await tools["search"]("net flows to Nigeria", ctx, [{"variable": "net flows"}])

    await tools["fetch"](SPEC_LIFE, ctx)

    readings = [SearchReading.model_validate(r) for r in ctx.state["orchestrator:searches"]]
    assert [(r.status, r.spec_id, r.text) for r in readings] == [
        ("no_data", None, "net flows"),
        ("chosen", SPEC_LIFE, "net flows"),
    ]
    assert ctx.state["orchestrator:specs"][SPEC_LIFE] == SINGLE_BLOCK


async def test_fetch_of_a_chosen_candidate_adds_a_reading_for_it():
    envelope = _recorded_envelope()
    envelope["parts"][2]["separate"] = True
    tools, ctx = _tools(_resolver(resolver_digest(envelope)))
    await tools["search"]("education aid to Kenya", ctx, [{"variable": "education aid"}])

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
    ("query", "part", "lacks"),
    [
        pytest.param(
            "Do you have any climate finance data",
            {"variable": "climate finance"},
            "a worldwide total with no breakdown by country",
            id="no-party-asked",
        ),
        pytest.param(
            "France's climate finance to Kenya",
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
    query: str, part: dict[str, Any], lacks: str
):
    tools, ctx = _tools(_resolver(resolver_digest(CLIMATE_ENVELOPE)))
    await tools["search"](query, ctx, [part])
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
        await tools["search"]("education aid to Kenya", ctx, [{"variable": "education aid"}])
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
        remove_cards=_remove_seam(state),
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
    """Fetches aid to Kenya from `providers` providers, described by `view`."""

    def __init__(self, *, providers: int = 30, view: dict[str, Any] | None = None) -> None:
        super().__init__()
        self.view = view or WIDE_VIEW
        self.view["dimensions"][1]["total_count"] = providers

    async def __call__(self, *, tool: str, args: dict[str, Any], on_progress: Any = None):
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


_CUT_TITLE = re.compile(r"^Aid to Kenya \((\d+) largest of (\d+) in (\d{4})\)$")


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
    drawn, holding, period = _CUT_TITLE.match(intent["title"]).groups()
    members = intent["select"]["financingProvider"]
    assert (period, int(holding)) == ("2024", 20)
    assert len(members) == int(drawn) == kept
    assert all(period in values[member] for member in members)
    assert members == sorted(members, key=lambda member: -values[member][period])
    assert result["charts"][0]["title"] == intent["title"]
    assert result["charts"][0]["note"] == (
        f"This card draws the {kept} largest of the 20 members with a value in 2024, highest "
        "first. Left out as group totals, unallocated amounts or members of another level: 0. "
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
    """A cut that cannot learn which members are group totals draws nothing
    and states the code of a failed kind read. A failed read is not kept, so
    the retry the refusal invites reads again and draws."""
    monkeypatch.setattr(orchestrator_tools, "get_frame_store", HeldFrames)
    executor = KindReadFails(failure)
    tools, ctx = _tools(executor=executor)
    await tools["fetch"](SPEC_LIFE, ctx)

    result = await tools["make_charts"]([_wide_chart(), _wide_chart(type="bar_chart")], ctx)

    assert [_stated(entry) for entry in result["charts"]] == [
        ("member_kind_unread", "retry_after_delay")
    ] * 2
    assert _writes(executor) == 0
    assert _kind_reads(executor) == 2

    executor.failure = None
    retried = await tools["make_charts"]([_wide_chart()], ctx)

    assert "error" not in retried["charts"][0]
    assert _writes(executor) == 1


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
    is chosen in its place."""
    values = {f"donor/{number:02d}": {periods[number]: 1.0} for number in range(30)}

    executor, result = await _draw_wide(monkeypatch, RowFrames(values), type="bar_chart")

    assert result["charts"][0]["code"] == "no_shared_period"
    assert _writes(executor) == 0


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
            "Aid to Kenya",
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
    assert intent["title"] == "Aid to Kenya (24 largest of 25 in 2024)"
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
        assert result["charts"][0]["title"] == "t (24 largest of 30 in 2024)"


async def test_make_charts_cut_title_carries_one_suffix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A chart titled with the title of a card already cut, as the model copies
    it from the first card's result, ends in one suffix."""
    cut = "Aid to Kenya (24 largest of 30 in 2024)"

    executor, _ = await _draw_wide(monkeypatch, HeldFrames(), type="table", title=cut)

    (intent,) = _intents(executor)
    assert intent["title"] == cut


async def _draw_wide_twice(
    monkeypatch: pytest.MonkeyPatch, first: dict[str, Any], second: dict[str, Any]
) -> tuple[WideFrameExecutor, list[dict[str, Any]]]:
    """Fetch the wide frame and draw `first` then `second` in two calls of one turn."""
    monkeypatch.setattr(orchestrator_tools, "get_frame_store", HeldFrames)
    executor = WideFrameExecutor()
    tools, ctx = _tools(executor=executor)
    await tools["fetch"](SPEC_LIFE, ctx)
    results = [await tools["make_charts"]([_wide_chart(**chart)], ctx) for chart in (first, second)]
    return executor, results


async def test_make_charts_does_not_draw_a_card_the_canvas_holds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The same title, kind, frame, select and cut drawn again writes no
    second component and returns the first card with a notice."""
    executor, (first, again) = await _draw_wide_twice(
        monkeypatch, {"type": "table"}, {"type": "table"}
    )

    assert len(_intents(executor)) == 1
    (card,) = again["charts"]
    assert card["title"] == first["charts"][0]["title"]
    assert "already_on_canvas" in card


async def test_make_charts_draws_a_card_of_another_kind_again(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A second chart of the same frame and select in another kind is drawn."""
    executor, (_, second) = await _draw_wide_twice(
        monkeypatch, {"type": "table"}, {"type": "bar_chart"}
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
    assert intent["title"] == "Aid to Kenya (8 largest of 9 in 2024)"


async def test_make_charts_keeps_the_cap_refusal_when_no_member_is_a_party(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    every = frozenset(f"donor/{number:02d}" for number in range(30))

    executor, result = await _draw_wide(monkeypatch, HeldFrames(totals=every))

    assert "error" in result["charts"][0]
    assert _writes(executor) == 0


@pytest.mark.parametrize(
    ("moves", "chart"),
    [({}, {}), ({"rolling": True}, {}), ({}, {"latest_periods": 5})],
    ids=["fixed", "rolling", "latest_periods"],
)
async def test_make_charts_names_the_dimensions_of_a_select_over_the_cap_on_two(
    monkeypatch: pytest.MonkeyPatch, moves: dict[str, Any], chart: dict[str, Any]
) -> None:
    """A table that binds neither the six recipients nor the thirty providers
    is not narrowed: the model reads both dimensions and their counts. It is
    told a chart of the thirty alone is cut to its largest, except where the
    chart's period moves with the data: no cut is made there, so it is told
    to list fewer or to rank with a limit."""
    view = copy.deepcopy(WIDE_VIEW) | moves
    view["dimensions"][2]["total_count"] = 6

    executor, result = await _draw_wide(
        monkeypatch, HeldFrames(), view=view, select_json="{}", **chart
    )

    error = result["charts"][0]["error"]
    assert "financingRecipient (6 values)" in error
    assert "financingProvider (30 values)" in error
    moving = bool(moves or chart)
    assert ("draws its largest members" in error) is not moving
    assert ("ranking with a limit" in error) is moving
    assert _stated(result["charts"][0]) == (
        HarnessCode.SERIES_LIMIT_EXCEEDED,
        "correct_and_retry",
    )
    assert _writes(executor) == 0


@pytest.mark.parametrize(
    "store",
    [
        HeldFrames(providers=5),
        HeldFrames(providers=5, totals=frozenset({"donor/04"}), carried=True),
    ],
    ids=["plain", "kinds_carried"],
)
async def test_make_charts_orders_a_ranking_under_the_cap_highest_first(
    monkeypatch: pytest.MonkeyPatch, store: HeldFrames
) -> None:
    """A table ordered by value lists every provider, highest first, with no
    note, since nothing is left out. A kind column the rows carry leaves the
    ordering as it is, group total included."""
    executor, result = await _draw_wide(
        monkeypatch, store, providers=5, type="table", order_by_value=True
    )

    (intent,) = _intents(executor)
    assert intent["select"]["financingProvider"] == [f"donor/{n:02d}" for n in (4, 3, 2, 1, 0)]
    assert intent["title"] == "Aid to Kenya"
    assert "note" not in result["charts"][0]


@pytest.mark.parametrize(
    "chart",
    [{"type": "table"}, {"type": "bar_chart", "order_by_value": True}],
    ids=["not_asked", "bar_chart"],
)
async def test_make_charts_keeps_the_listed_order_unless_asked_to_rank(
    monkeypatch: pytest.MonkeyPatch, chart: dict[str, Any]
) -> None:
    """apis orders a bar chart's bars itself, so `order_by_value` leaves
    its select as listed."""
    listed = ["donor/00", "donor/03", "donor/01"]
    select = json.dumps({"financingRecipient": "country/KEN", "financingProvider": listed})

    executor, _ = await _draw_wide(
        monkeypatch, HeldFrames(providers=5), providers=5, select_json=select, **chart
    )

    (intent,) = _intents(executor)
    assert intent["select"]["financingProvider"] == listed


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
    assert intent["title"] == "Aid to Kenya (20 largest of 30 in 2024)"


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
    assert orchestrator_tools._periods_covered({"coverage": coverage}) == periods


async def test_make_charts_keeps_a_drawable_select_when_ordering_would_break_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Nine providers unbound draw as nine lines; listing all nine would pass
    the eight a line chart lists, so ordering leaves the select as written."""
    executor, result = await _draw_wide(
        monkeypatch,
        HeldFrames(providers=9),
        providers=9,
        type="line_chart",
        order_by_value=True,
    )

    (intent,) = _intents(executor)
    assert intent["select"] == {"financingRecipient": "country/KEN"}
    assert "error" not in result["charts"][0]


async def test_make_charts_orders_a_member_with_no_value_at_the_period_last(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ordering under the cap leaves no provider out. One with no value in
    the year the others are ordered at is listed after them, and the card
    carries no note."""
    values = {"donor/a": {"2024": 1.0}, "donor/b": {"2024": 2.0}, "donor/c": {"2023": 3.0}}

    executor, result = await _draw_wide(
        monkeypatch, RowFrames(values), providers=3, type="table", order_by_value=True
    )

    (intent,) = _intents(executor)
    assert intent["select"]["financingProvider"] == ["donor/b", "donor/a", "donor/c"]
    assert intent["title"] == "Aid to Kenya"
    assert "note" not in result["charts"][0]


class UnreadableFrames:
    async def open(self, ref: Any) -> None:
        raise FrameTooLarge(ref, reason="over the read ceiling")


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
    so a select over the cap on it is not cut: the cap refusal stands and no
    write reaches those rules. They would refuse the title a cut writes, for
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

    assert _stated(refused) == (HarnessCode.SERIES_LIMIT_EXCEEDED, "correct_and_retry")
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


@pytest.mark.parametrize(
    ("store", "chart", "view", "warned"),
    [
        (UnreadableFrames(), {}, None, True),
        (FailingStore(), {}, None, True),
        (HeldFrames(held=False), {}, None, True),
        (UncountedFrames(), {}, None, True),
        (_rank_positions(), {}, None, False),
        (HeldFrames(), {}, {**WIDE_VIEW, "rolling": True}, False),
        (HeldFrames(), {"type": "line_chart", "latest_periods": 2}, None, False),
    ],
    ids=[
        "unreadable",
        "store_failed",
        "not_held",
        "no_row_count",
        "rank_positions",
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
) -> None:
    """A frame the store cannot read, a plan's rank positions (where the
    highest number is the worst), and a card whose period moves with the
    data (a list picked today would not follow it) are never narrowed: the
    refusal stands. A frame the store cannot read, does not hold or states no
    row count for is logged as a warning."""
    monkeypatch.setattr(orchestrator_tools, "read_derived_marker", _marker_of)

    with caplog.at_level(logging.WARNING, logger=orchestrator_tools.__name__):
        executor, result = await _draw_wide(monkeypatch, store, view=view, **chart)

    assert "error" in result["charts"][0]
    assert _writes(executor) == 0
    assert (orchestrator_tools.__name__ in {record.name for record in caplog.records}) is warned
    # A chart whose period moves is told what draws in place of a cut.
    moving = bool(view and view.get("rolling")) or "latest_periods" in chart
    assert ("ranking with a limit" in result["charts"][0]["error"]) is moving
    assert _stated(result["charts"][0]) == (
        HarnessCode.SERIES_LIMIT_EXCEEDED,
        "correct_and_retry" if moving else "do_not_retry",
    )


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

    found = await tools["search"]("aid from France to Kenya", ctx, [{"variable": "aid"}])

    (part,) = found["parts"]
    assert part["counted_as"] == "gross disbursements"
    assert part["alternatives"] == [
        {"spec_id": "spec_c67524ce5da3bfd4d94994a76c560b28", "label": "net disbursements"}
    ]
    assert part.get("other_prices") == (
        [{"spec_id": SPEC_CONSTANT, "label": "constant prices"}] if constant else None
    )


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
    ],
)
async def test_ask_user_offers_a_count_only_under_a_spec_id_this_turns_search_named(
    questions: list[dict[str, Any]], refused: str | None
):
    part = {"variable": "aid", "status": "definite", "spec_id": SPEC_AID}
    tools, ctx = _tools(resolver=_resolver(_digest({**part, "spec_block": COUNTED_BLOCK})))
    await tools["search"]("aid to Kenya", ctx, [{"variable": "aid"}])

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
    await by_name["search"]("aid from France to Kenya", ctx, [{"variable": "aid"}])
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

    pins = planner_seam._reading_pin_sources(
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
    await tools["search"](part["variable"], ctx, [part])
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
        await tools["search"](part["variable"], ctx, [part])

    recorded = tools["plan"](
        "Debt service by creditor", [{"text": "Debt service by creditor", "by": ["creditor"]}], ctx
    )

    assert recorded[HANDED_OFF] == "plan"


async def test_plan_refuses_asks_naming_one_of_two_parties_two_flows_held_open():
    tools, ctx = await _searched({"variable": "aid by donor", "open_roles": ["donor"]})
    await tools["search"](_BY_CREDITOR["variable"], ctx, [_BY_CREDITOR])

    refused = tools["plan"](
        "brief", [{"text": "aid by donor", "by": ["donor"]}, "total debt service"], ctx
    )

    assert refused["invalid"] == [{"loc": "asks", "type": "breakdown_missing", "party": "creditor"}]


async def test_plan_refuses_an_ask_listing_two_parties_of_one_side_and_takes_an_ask_for_each():
    tools, ctx = await _searched({"variable": "aid by donor", "open_roles": ["donor"]})
    await tools["search"](_BY_CREDITOR["variable"], ctx, [_BY_CREDITOR])

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
        ask_breakdowns=[AskBreakdown(ask=0, by=("creditor",))],
        asks_across_years=[1],
    )


async def test_plan_takes_a_party_the_search_held_open_that_an_ask_totals_across():
    tools, ctx = await _searched(_BY_CREDITOR)

    recorded = tools["plan"](
        "Total debt service", [{"text": "Total debt service", "over": ["creditor"]}], ctx
    )

    assert recorded[HANDED_OFF] == "plan"
    assert pending_action(ctx.state) == PendingPlan(
        brief="Total debt service", asks=["Total debt service"]
    )


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


async def test_get_skill_file_reads_on_with_the_offset_and_digest_a_page_named():
    surface = FakeSurface(ToolCallResult(is_error=False, text="the rest"))
    tools, ctx = _tools(surface=surface)

    read = await tools["get_skill_file"](
        "oda-measures", "SKILL.md", ctx, offset=32000, digest="sha256:" + "0" * 64
    )
    first = await tools["get_skill_file"]("oda-measures", "references/grant.md", ctx)

    assert (surface.calls[0][0].server, surface.calls[0][0].name) == (
        "data-mcp",
        "get_skill_file",
    )
    assert [arguments for _, arguments in surface.calls] == [
        {
            "skill_id": "oda-measures",
            "path": "SKILL.md",
            "offset": 32000,
            "digest": "sha256:" + "0" * 64,
        },
        {"skill_id": "oda-measures", "path": "references/grant.md", "offset": 0},
    ]
    assert read == first == {"skill": "the rest"}


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
    """A surface whose call times out, as `compiler.surface`'s per-tool timeout raises."""

    async def call_tool(self, *, tool, arguments, on_progress=None) -> ToolCallResult:
        self.calls.append((tool, arguments))
        raise ToolSurfaceUnavailableError(f"{tool.name} timed out after 30s")


@pytest.mark.parametrize(
    ("tool", "arguments"),
    [
        ("get_skill", ("oda-measures",)),
        ("get_skill_file", ("oda-measures", "references/grant.md")),
    ],
)
async def test_a_skill_read_that_does_not_complete_is_a_retryable_error_for_the_model(
    tool: str, arguments: tuple[str, ...]
):
    surface = UnreachableSurface(ToolCallResult(is_error=False, text=""))
    tools, ctx = _tools(surface=surface)

    answered = await tools[tool](*arguments, ctx)

    assert answered == {
        "error": f"{tool} could not reach data-mcp just now. Call it again, or answer "
        "without this skill.",
        "code": "transient_unavailable",
        "retry_mode": "retry_after_delay",
    }
    assert len(surface.calls) == 1


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
            raise ToolSurfaceUnavailableError(f"{tool.name} timed out after 30s")
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


def _recall_seam(state: ConversationState):
    async def recall_part(**target: Any) -> dict[str, Any]:
        return state.recall_part(**target)

    return recall_part


# --- remove_cards -------------------------------------------------------------


def _remove_seam(state: ConversationState):
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
    tools, ctx = _tools(remove_cards=_remove_seam(state))

    answer = await tools["remove_cards"](ctx, ["cmp_01", "cmp_99"])

    assert answer == {"removed": ["cmp_01"], "not_in_conversation": ["cmp_99"]}
    assert state.composition is not None
    assert [b.component_id for b in state.composition.blocks] == ["cmp_02"]


async def test_remove_cards_clears_the_canvas_and_refuses_a_call_that_names_nothing():
    state = _two_card_canvas()
    tools, ctx = _tools(remove_cards=_remove_seam(state))

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
    tools, ctx = _tools(remove_cards=_remove_seam(state))
    await tools["remove_cards"](ctx, ["cmp_01"])

    again = await tools["remove_cards"](ctx, ["cmp_01"])
    unknown = await tools["remove_cards"](ctx, ["cmp_99"])

    assert again == {"removed": [], "note": "Those cards were already off the canvas."}
    assert _stated(unknown) == ("card_not_found", "correct_and_retry")
    assert "cmp_99" not in unknown["error"]


async def test_remove_cards_is_unavailable_without_a_conversation_seam():
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
    tools, ctx = _tools(recall_part=_recall_seam(state))

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
    tools, ctx = _tools(recall_part=_recall_seam(state))

    listing = await tools["recall"]("canvas", ctx)
    reads = [await tools["recall"]("canvas", ctx, component_id="cmp_01") for _ in range(3)]

    assert [card["component_id"] for card in listing["on_canvas"]] == ["cmp_01"]
    assert all("error" not in read for read in reads)
    for _ in range(BUDGETS["recall_canvas"] - 1):
        await tools["recall"]("canvas", ctx)
    assert "limit" in (await tools["recall"]("canvas", ctx))["error"]


async def test_recall_is_limited_per_turn_and_unavailable_without_a_conversation_seam():
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
    tools, ctx = _tools(remove_cards=_remove_seam(state))
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
    found = await tools["search"](
        "aid from shared", ctx, [{"variable": "aid", "sources": [{"text": "shared"}]}]
    )
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
    await tools["search"]("aid from DAC", ctx, [{"variable": "aid", "sources": [{"text": "DAC"}]}])
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
        await by_name["search"](
            "aid from DAC", ctx, [{"variable": "aid", "sources": [{"text": "DAC"}]}]
        )
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
