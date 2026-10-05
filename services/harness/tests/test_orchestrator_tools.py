"""The orchestrator's tools against a fake resolver, executor and tool surface.

`usable_component` is `harness.wire`'s own check, tested with the wire
contract; here it is stubbed so a fake component can stand in for a written
one.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from typing import Any

import pytest
from compiler.surface import ToolCallResult, ToolSurfaceUnavailableError
from google.adk.sessions.state import State
from harness import orchestrator_tools
from harness.dfo_faq import CANNOT_DESCRIBE_DFO
from harness.graph import (
    ORCHESTRATOR_CALLS_KEY,
    ORCHESTRATOR_FRAMES_KEY,
    PendingAsk,
    PendingPlan,
    frame_id_for,
    pending_action,
)
from harness.orchestrator_tools import BUDGETS, build_orchestrator_tools
from harness.tool_contracts import ResolverError

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
        "stat_vars": {"items": [{"ref": "A"}, {"ref": "B"}], "member_count": 2},
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
        "stat_vars": {"items": [{"ref": "LE"}], "member_count": 1},
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
):
    tools = build_orchestrator_tools(
        resolver=resolver or _resolver(_digest()),
        executor=executor or FakeExecutor(),
        surface=surface,
        dfo_faq_surface=dfo_faq_surface,
    )
    by_name = {fn.__name__: fn for fn in tools}
    return by_name, SimpleNamespace(state={"surface": surface_name}, invocation_id="inv_1")


@pytest.fixture(autouse=True)
def _usable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(orchestrator_tools, "usable_component", lambda component: True)


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


async def test_search_returns_a_refusal_as_an_error_and_stops_at_its_budget():
    refused = ResolverError("search refused", refusal={"message": "search is down"})
    tools, ctx = _tools(_resolver(refused))

    first = await tools["search"]("q", [{"variable": "x"}], ctx)
    assert first == {"error": "search is down"}

    for _ in range(BUDGETS["search"] - 1):
        await tools["search"]("q", [{"variable": "x"}], ctx)
    spent = await tools["search"]("q", [{"variable": "x"}], ctx)

    assert "limit" in spent["error"]


# --- fetch ----------------------------------------------------------------


async def test_fetch_reads_a_family_as_its_total_and_a_single_indicator_as_is():
    executor = FakeExecutor()
    tools, ctx = _tools(executor=executor)
    ctx.state["orchestrator:specs"] = {SPEC_AID: FAMILY_BLOCK, SPEC_LIFE: SINGLE_BLOCK}

    aid = await tools["fetch"](SPEC_AID, ctx)
    life = await tools["fetch"](SPEC_LIFE, ctx)
    unknown = await tools["fetch"](
        "spec_other", ctx, scope={"window": {"kind": "range", "start_year": 2015, "end_year": 2020}}
    )

    assert [args for tool, args in executor.calls if tool == "get_data"] == [
        {"spec_id": SPEC_AID, "stat_vars": "sum", "resolved_stat_var_dcids": ["A", "B"]},
        {"spec_id": SPEC_LIFE, "resolved_stat_var_dcids": ["LE"]},
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


async def test_fetch_returns_the_refusal_when_the_data_cannot_be_retrieved():
    tools, ctx = _tools()

    found = await tools["fetch"]("spec_refused", ctx)

    assert found == {"spec_id": "spec_refused", "refusal": "too big", "code": "over_ceiling"}


async def test_the_download_surface_fetches_once_and_never_plans():
    tools, ctx = _tools(surface_name="download")

    assert "frames" in await tools["fetch"](SPEC_LIFE, ctx)
    assert "limit" in (await tools["fetch"](SPEC_LIFE, ctx))["error"]
    assert "limit" in tools["plan"]("Kenya's aid as a share of GDP", ctx)["error"]
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
    second = tools["plan"]("Congo's GDP", ctx)

    assert asked["status"].startswith("Recorded")
    assert pending_action(ctx.state) == PendingAsk(
        question="Which Congo?", options=["Republic of the Congo", "DR Congo"]
    )
    assert "already waiting on ask" in second["error"]


def test_plan_records_the_brief_once_per_turn():
    tools, ctx = _tools()

    first = tools["plan"]("Kenya's aid as a share of GDP since 2010", ctx)
    ctx.state["orchestrator:pending"] = None  # the workflow took it
    again = tools["plan"]("Kenya's aid as a share of GDP since 2010", ctx)

    assert first["status"].startswith("Recorded")
    assert pending_action({"orchestrator:pending": {"kind": "plan", "brief": "b"}}) == PendingPlan(
        brief="b"
    )
    assert "limit" in again["error"]


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
