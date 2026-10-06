"""The orchestrator's tools against a fake resolver, executor and tool surface.

`usable_component` is `harness.wire`'s own check, tested with the wire
contract; here it is stubbed so a fake component can stand in for a written
one.
"""

from __future__ import annotations

import asyncio
import copy
import json
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx2
import pytest
from compiler.surface import ToolCallResult, ToolSurfaceUnavailableError
from data_mcp.tool_errors import ToolErrorCode
from google.adk.sessions.state import State
from google.adk.tools import FunctionTool
from harness import orchestrator_tools
from harness.dfo_faq import CANNOT_DESCRIBE_DFO
from harness.graph import (
    LINE_CHART_MAX_VALUES,
    MAX_SERIES_PER_COMPONENT,
    ORCHESTRATOR_CALLS_KEY,
    ORCHESTRATOR_FRAMES_KEY,
    ORCHESTRATOR_UNAVAILABLE_KEY,
    PendingAsk,
    PendingPlan,
    frame_id_for,
    pending_action,
)
from harness.orchestrator_tools import (
    BUDGETS,
    HANDED_OFF,
    Chart,
    OrchestratorSearchPart,
    build_orchestrator_tools,
)
from harness.studio_state import ConversationState
from harness.tool_contracts import ResolverError, graph_envelope, resolver_digest
from harness.turn_records import SearchReading
from mcp import MCPError
from mcp.types import CONNECTION_CLOSED, REQUEST_TIMEOUT, ErrorData
from orchestrator.agent import INSTRUCTION
from planner.plan import MAX_ASK_CHARS, MAX_PLAN_ASKS, current_earlier_block

from .test_turn_records import _top_donors

SPEC_AID = "spec_" + "a" * 32
SPEC_LIFE = "spec_" + "b" * 32
REF = {"spec_id": SPEC_LIFE, "content_version": "cv_1"}
VIEW = {"dimensions": [{"name": "observationAbout", "values": ["country/NGA"], "plural": False}]}

# A spec block `compiler.sources.source_from_spec_block` reads as a family
# of two StatVars that sums over one open, summable axis.
FAMILY_BLOCK = {
    "spec_id": SPEC_AID,
    "resolved": {
        "rendered_sentence": "Health aid to Kenya",
        "stat_vars": {"member_count": 2, "served_count": 2},
        "slots": [
            {
                "key": {"property": {"dcid": "crsPurposeCode"}},
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
        component = {"component_id": "cmp_1", "fact_summary": {"series": [{"first": 37.2}]}}
        return {"outcome": "ok", "component": component}

    def land_component(self, component: dict[str, Any], *, land: bool, revises: str | None) -> None:
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
):
    tools = build_orchestrator_tools(
        resolver=resolver or _resolver(_digest()),
        executor=executor or FakeExecutor(),
        surface=surface,
        dfo_faq_surface=dfo_faq_surface,
        recall_part=recall_part,
    )
    by_name = {fn.__name__: fn for fn in tools}
    return by_name, SimpleNamespace(state={"surface": surface_name}, invocation_id="inv_1")


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


def test_every_chart_field_is_described_and_select_json_shows_a_list_of_values():
    properties = Chart.model_json_schema()["properties"]
    examples = [
        json.loads(example)
        for example in re.findall(r'\{"[^{}]*\}', properties["select_json"]["description"])
    ]

    assert all(field.get("description") for field in properties.values())
    assert any(isinstance(value, list) for example in examples for value in example.values())


def test_the_chart_descriptions_state_both_series_caps():
    properties = Chart.model_json_schema()["properties"]

    assert f"up to {LINE_CHART_MAX_VALUES} places" in properties["type"]["description"]
    assert f"at most {MAX_SERIES_PER_COMPONENT} series" in properties["select_json"]["description"]


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
        [
            {"variable": "life expectancy", "entities": [{"text": "Nigeria"}]},
            {"variable": "GDP", "entities": [{"text": "Nigeria"}]},
        ],
        ctx,
    )

    assert len(resolver.calls) == 1
    assert resolver.calls[0]["question"] == "life expectancy and GDP in Nigeria"
    assert resolver.calls[0]["parts"] == [
        {"variable": "life expectancy", "entities": [{"text": "Nigeria"}]},
        {"variable": "GDP", "entities": [{"text": "Nigeria"}]},
    ]
    assert found["kind"] == "resolution"
    assert [part["status"] for part in found["parts"]] == ["definite", "candidates"]
    assert not any("spec_block" in part for part in found["parts"])
    assert found["skills"] == [
        {"id": f"skill-{index}", "description": f"Skill {index}"} for index in range(5)
    ]
    assert found["more_skills"] == 2
    assert ctx.state["orchestrator:specs"] == {SPEC_LIFE: SINGLE_BLOCK}


async def test_search_reports_a_result_with_no_parts_by_its_status():
    tools, ctx = _tools(
        _resolver(_digest(status="unsupported", warnings=[{"message": "no measure"}]))
    )

    found = await tools["search"]("what data do you have?", [{"variable": "anything"}], ctx)

    assert found == {
        "kind": "resolution",
        "parts": [],
        "status": "unsupported",
        "warnings": ["no measure"],
    }


def test_search_tells_the_model_when_to_set_a_role_and_to_choose_a_source_through_fetch():
    tools, _ = _tools()
    schema = OrchestratorSearchPart.model_json_schema()

    assert "role" in tools["search"].__doc__
    assert "get_data" not in schema["properties"]["source"]["description"]
    assert "fetch" in schema["properties"]["source"]["description"]


async def test_search_returns_a_refusal_as_an_error_and_stops_at_its_budget():
    refused = ResolverError("search refused", refusal={"message": "search is down"})
    tools, ctx = _tools(_resolver(refused))

    first = await tools["search"]("q", [{"variable": "x"}], ctx)
    assert first == {"error": "search is down"}

    for _ in range(BUDGETS["search"] - 1):
        await tools["search"]("q", [{"variable": "x"}], ctx)
    spent = await tools["search"]("q", [{"variable": "x"}], ctx)

    assert "limit" in spent["error"]


# A recorded definite part: education aid to Kenya, 2015 to 2023, every donor open.
ROLE_OPEN_SPEC = json.loads(
    (Path(__file__).parent / "fixtures" / "search_spec_role_open.json").read_text()
)["spec"]
ROLE_OPEN_ID = ROLE_OPEN_SPEC["spec_id"]
GUESSED = {"code": "time_text_not_parsed", "severity": "degraded", "message": "time guessed"}


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
                "candidates": {"specs": [SINGLE_BLOCK]},
            },
        ],
    }


async def test_search_stores_a_reading_for_every_part_and_warnings_for_every_spec():
    envelope = _recorded_envelope()
    tools, ctx = _tools(_resolver(resolver_digest(envelope)))

    found = await tools["search"]("education aid to Kenya", [{"variable": "education aid"}], ctx)

    # The model sees the part summaries, which carry none of the stored facts.
    assert found["parts"] == graph_envelope(envelope)["part_digests"]
    definite, no_data, candidates = (
        SearchReading.model_validate(stored) for stored in ctx.state["orchestrator:searches"]
    )
    assert (definite.status, definite.spec_id) == ("definite", ROLE_OPEN_ID)
    assert (definite.coverage_kind, definite.requested_window_kind) == ("exact", "range")
    assert definite.roles == ("destination",)
    assert [(axis.column, axis.members) for axis in definite.open_axes] == [("origin", 55)]
    assert [(y.year, y.observations, y.series) for y in definite.recent_years] == [
        (2024, 239, 239),
        (2023, 1807, 300),
    ]
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


async def test_search_keeps_the_readings_of_earlier_searches_in_the_turn():
    tools, ctx = _tools(_resolver(_digest({"variable": "x", "status": "no_data"})))

    await tools["search"]("q", [{"variable": "x"}], ctx)
    await tools["search"]("q", [{"variable": "x"}], ctx)

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


async def test_fetch_of_a_chosen_candidate_adds_a_reading_for_it():
    tools, ctx = _tools(_resolver(resolver_digest(_recorded_envelope())))
    await tools["search"]("education aid to Kenya", [{"variable": "education aid"}], ctx)

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
    assert chosen.fetch is not None and chosen.fetch.summary == "64 observations"
    assert readings[0].fetch is not None


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
    assert ctx.state[ORCHESTRATOR_FRAMES_KEY] == {frame_id_for(REF): {"ref": REF, "view": VIEW}}


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
    assert "limit" in tools["plan"]("Kenya's aid as a share of GDP", [], ctx)["error"]
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
    assert all("facts" in chart for chart in drawn["charts"])


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
        ],
        ctx,
    )

    assert "unknown frame" in drawn["charts"][0]["error"]
    assert "which the frame does not declare" in drawn["charts"][1]["error"]
    assert not any(tool == "write_component" for tool, _ in executor.calls)


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
    async def __call__(self, *, tool: str, args: dict[str, Any], on_progress: Any = None):
        if tool == "write_component":
            raise RuntimeError("presentation-mcp is down")
        return await super().__call__(tool=tool, args=args, on_progress=on_progress)


def _one_chart(frame: str) -> list[dict[str, str]]:
    select = json.dumps({"observationAbout": "country/NGA"})
    return [{"frame": frame, "type": "bar_chart", "title": "t", "select_json": select}]


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
    for _ in range(BUDGETS["make_charts"]):
        drawn = await tools["make_charts"](_one_chart(frame), ctx)
        assert "facts" in drawn["charts"][0]
    writes = _writes(executor)
    spent = await tools["make_charts"](_one_chart(frame), ctx)

    assert set(spent) == {"error"}
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

    assert set(spent) == {"error"}
    assert _writes(executor) == writes


async def test_make_charts_counts_a_call_the_executor_raised_out_of():
    tools, ctx = _tools(executor=RaisingExecutor())
    await tools["fetch"](SPEC_LIFE, ctx)

    with pytest.raises(RuntimeError):
        await tools["make_charts"](_one_chart(frame_id_for(REF)), ctx)

    assert ctx.state["orchestrator:calls"]["make_charts_failed"] == 1


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


# --- ask_user and plan: one pending action per turn ----------------------


def test_ask_user_records_the_question_and_refuses_a_second_hand_off():
    tools, ctx = _tools()

    assert "error" in tools["ask_user"]("Which Congo?", [], ctx)
    asked = tools["ask_user"]("Which Congo?", ["Republic of the Congo", "DR Congo"], ctx)
    second = tools["plan"]("Congo's GDP", [], ctx)

    assert asked[HANDED_OFF] == "ask"
    assert HANDED_OFF not in second
    assert pending_action(ctx.state) == PendingAsk(
        question="Which Congo?", options=["Republic of the Congo", "DR Congo"]
    )
    assert "already waiting on ask" in second["error"]


def test_plan_records_the_brief_once_per_turn():
    tools, ctx = _tools()

    first = tools["plan"]("Kenya's aid as a share of GDP since 2010", [], ctx)
    ctx.state["orchestrator:pending"] = None  # the workflow took it
    again = tools["plan"]("Kenya's aid as a share of GDP since 2010", [], ctx)

    assert first["status"].startswith("Recorded")
    assert pending_action({"orchestrator:pending": {"kind": "plan", "brief": "b"}}) == PendingPlan(
        brief="b"
    )
    assert "limit" in again["error"]


@pytest.mark.parametrize(
    "asks",
    [
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
    assert refused["legal"] == {"max_asks": MAX_PLAN_ASKS, "ask_chars": [1, MAX_ASK_CHARS]}
    assert HANDED_OFF not in refused
    assert recorded[HANDED_OFF] == "plan"
    assert pending_action(ctx.state) == PendingPlan(
        brief="France's aid to Guatemala", asks=["the total since 2010"]
    )


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
    assert await tools["get_skill"]("nope", ctx) == {"error": "no such skill"}

    tools, ctx = _tools()
    assert "not available" in (await tools["get_skill"]("nope", ctx))["error"]


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
        "without this skill."
    }
    assert len(surface.calls) == 1


async def test_explain_dfo_reads_dfo_faq_off_the_activity_feed():
    """dfo-faq comes through the bare surface: the recording one would put a
    second step beside `explain_dfo`'s own."""
    recorded = FakeSurface(ToolCallResult(is_error=False, text="unused"))
    bare = FakeSurface(ToolCallResult(is_error=False, text="# The DFO\nBuilt by ONE Data."))
    tools, _ = _tools(surface=recorded, dfo_faq_surface=bare)

    explained = await tools["explain_dfo"]()

    assert [(tool.name, arguments) for tool, arguments in bare.calls] == [
        ("get_skill", {"skill_id": "dfo-faq"})
    ]
    assert recorded.calls == []
    assert explained == {"explainer": "# The DFO\nBuilt by ONE Data."}


async def test_explain_dfo_says_plainly_when_the_deployment_has_no_skills_api():
    refused = ToolCallResult(is_error=True, structured_content={"code": "capability_unavailable"})
    tools, _ = _tools(dfo_faq_surface=FakeSurface(refused))

    assert await tools["explain_dfo"]() == {"error": CANNOT_DESCRIBE_DFO}


# --- recall -----------------------------------------------------------------


def _recall_seam(state: ConversationState):
    async def recall_part(**target: Any) -> dict[str, Any]:
        return state.recall_part(**target)

    return recall_part


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


async def test_recall_is_limited_per_turn_and_unavailable_without_a_conversation_seam():
    tools, ctx = _tools()
    unavailable = await tools["recall"]("plan", ctx)

    assert "not available" in unavailable["error"]

    ctx.state["orchestrator:calls"] = {"recall": BUDGETS["recall"]}
    assert "limit" in (await tools["recall"]("plan", ctx))["error"]
