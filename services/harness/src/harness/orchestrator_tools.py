"""The orchestrator's tools, over the resolver, executor and tool surface the graph uses.

`search` asks data-mcp's `search` once for every part of a question;
`fetch` retrieves one dataset and describes its frames; `make_charts`
draws on the canvas through the graph's own presentation checks;
`get_skill` loads a skill a search result named and `get_skill_file`
reads on through it; `explain_dfo` reads dfo-faq, DFO's own description of
itself (`harness.dfo_faq`). `ask_user` and `plan` record one pending action
for the workflow (`harness.orchestrator_workflow`) to act on once the model's
turn ends: a pause for the user's answer, or the graph's plan path.

The model sees digests only: part readings, frame views, one-line
summaries and chart facts, never rows. Frames are named to it by short
ids (`harness.graph.frame_id_for`) kept in session state, so `make_charts` can only draw
a frame `fetch` or an executed plan produced this turn, after
`describe_frame` answered for it. Each tool's calls per turn are capped
(`BUDGETS`); the download surface fetches one dataset and never plans.
"""

from __future__ import annotations

import asyncio
import json
import logging
import weakref
from typing import Any, Literal

from compiler.sources import default_reading, is_family, source_from_spec_block
from compiler.surface import ToolSurface, ToolSurfaceUnavailableError
from data_mcp.get_data_tool import GetDataScope
from data_mcp.search_tool import SearchPart
from google.adk.tools import ToolContext
from planner.skills import GET_SKILL, GET_SKILL_FILE
from planner.workflow import ToolId
from pydantic import BaseModel

from harness.dfo_faq import DfoFaqUnavailable, read_dfo_faq
from harness.graph import (
    ORCHESTRATOR_CALLS_KEY,
    ORCHESTRATOR_FRAMES_KEY,
    ORCHESTRATOR_PENDING_KEY,
    ORCHESTRATOR_READINGS_KEY,
    ORCHESTRATOR_SPECS_KEY,
    Executor,
    PendingAsk,
    PendingPlan,
    Resolver,
    pending_action,
    register_frames,
    select_binding_ok,
)
from harness.tool_contracts import ResolverError
from harness.wire import usable_component

logger = logging.getLogger(__name__)

# Calls one turn may make of each tool. `search` counts every resolver
# call, `fetch` every dataset read; a plan is one per turn, and a plan
# that failed gives it back (`harness.graph`'s `plan_done`) for a brief
# that avoids the failure. `make_charts` counts the calls that drew at
# least one chart, `make_charts_failed` the calls that drew none.
BUDGETS = {
    "search": 4,
    "fetch": 6,
    "make_charts": 5,
    "make_charts_failed": 3,
    "ask_user": 2,
    "plan": 1,
    "get_skill": 2,
    # A file a skill links, or the next page of a long one.
    "get_skill_file": 4,
}
# Charts one make_charts call may draw.
MAX_CHARTS_PER_CALL = 8
# The download surface fetches one dataset and never plans.
DOWNLOAD_BUDGETS = {**BUDGETS, "fetch": 1, "plan": 0}
# Skills a search result names, as shown to the model, as data-mcp's search
# digest shows them; the rest are counted.
_SKILLS_SHOWN = 5

TOOL_NAMES = (
    "search",
    "fetch",
    "make_charts",
    "ask_user",
    "plan",
    "get_skill",
    "get_skill_file",
    "explain_dfo",
)


class Chart(BaseModel):
    """One chart to draw from a frame `fetch` or a plan returned."""

    frame: str
    type: Literal["line_chart", "bar_chart", "stat_card", "table"]
    title: str
    select_json: str


_ANSWER_NOW = "Answer with what you have, and say plainly what you could not do."


def _budgets(tool_context: ToolContext) -> dict[str, int]:
    return DOWNLOAD_BUDGETS if tool_context.state.get("surface") == "download" else BUDGETS


def _used(tool_context: ToolContext, key: str) -> int:
    return int((tool_context.state.get(ORCHESTRATOR_CALLS_KEY) or {}).get(key, 0))


def _count(tool_context: ToolContext, key: str) -> None:
    """Add one to `key`'s calls this turn, reading the counts afresh."""
    calls = dict(tool_context.state.get(ORCHESTRATOR_CALLS_KEY) or {})
    calls[key] = int(calls.get(key, 0)) + 1
    tool_context.state[ORCHESTRATOR_CALLS_KEY] = calls


def _spend(tool_context: ToolContext, tool: str) -> str | None:
    """Count one call of `tool`; the refusal text when its budget is spent."""
    used = _used(tool_context, tool)
    if used >= _budgets(tool_context)[tool]:
        return f"{tool} has been called {used} time(s) this turn, its limit. {_ANSWER_NOW}"
    _count(tool_context, tool)
    return None


def _charts_spent(tool_context: ToolContext) -> str | None:
    """The refusal text when either of make_charts' budgets is spent."""
    budgets = _budgets(tool_context)
    drew = _used(tool_context, "make_charts")
    if drew >= budgets["make_charts"]:
        return f"make_charts has drawn charts in {drew} call(s) this turn, its limit. {_ANSWER_NOW}"
    failed = _used(tool_context, "make_charts_failed")
    if failed >= budgets["make_charts_failed"]:
        return (
            f"make_charts has drawn nothing in {failed} call(s) this turn, the limit for calls "
            f"that draw nothing. {_ANSWER_NOW}"
        )
    return None


def _pend(tool_context: ToolContext, action: PendingAsk | PendingPlan) -> dict[str, Any]:
    """Record `action` as the turn's pending action, unless one is pending."""
    pending = pending_action(tool_context.state)
    if pending is not None:
        return {
            "error": f"This turn is already waiting on {pending.kind}; one hand-off per turn. "
            "End your turn now."
        }
    tool_context.state[ORCHESTRATOR_PENDING_KEY] = action.model_dump()
    return {"status": "Recorded. End your turn now; the result comes back as the next message."}


def build_orchestrator_tools(
    *,
    resolver: Resolver,
    executor: Executor,
    surface: ToolSurface | None = None,
    dfo_faq_surface: ToolSurface | None = None,
) -> list[Any]:
    """The orchestrator's tools, bound to this server's resolver, executor and tool surface.

    `dfo_faq_surface` is the surface `explain_dfo` reads dfo-faq through: the
    bare one, off the Activity feed, since `explain_dfo`'s own call is the
    step the user sees. Left unset, `explain_dfo` says it cannot describe DFO.
    """
    # One lock per invocation, held by a make_charts call from its budget
    # check to its count, so concurrent calls draw one after another and each
    # is admitted on exact counts. An entry lasts while a call holds its lock.
    chart_locks: weakref.WeakValueDictionary[str, asyncio.Lock] = weakref.WeakValueDictionary()

    async def search(
        query: str, parts: list[SearchPart], tool_context: ToolContext
    ) -> dict[str, Any]:
        """Search DFO's data for every measured thing a question asks for.

        `query` is the question in the user's words; `parts` is one entry per
        measured variable in it, each with the entities, time, unit or
        source the question gives for that variable, as spoken. All parts go
        in one call.

        Returns the result's `kind`: a `resolution` reads each part
        separately; a `catalogue` means the question asked what data DFO
        holds, answered with a scope and no parts; a `declined` means
        Search would not answer, with a reason, such as a question outside
        DFO's scope or one that needs analysis. Then
        `parts`, one per part: its `status` (definite: a spec_id to fetch;
        candidates: readings to choose between, each with a spec_id and
        label; no_data or unsupported: with warnings saying why), what it
        resolved to in words, its warnings, and the window it covers. And
        `skills`: background a result calls for, each by id and
        description, loaded with get_skill, and `more_skills`, how many
        more the result names.
        """
        spent = _spend(tool_context, "search")
        if spent:
            return {"error": spent}
        wire_parts = [
            SearchPart.model_validate(part).model_dump(exclude_none=True) for part in parts
        ]
        try:
            digest = await resolver(question=query, parts=wire_parts)
        except ResolverError as exc:
            return {"error": (exc.refusal or {}).get("message") or str(exc)}
        specs = dict(tool_context.state.get(ORCHESTRATOR_SPECS_KEY) or {})
        readings = []
        for part in digest.get("part_digests") or []:
            reading = {
                key: value
                for key, value in part.items()
                if key not in ("spec_block", "candidate_blocks")
            }
            if part.get("spec_block") and part.get("spec_id"):
                specs[part["spec_id"]] = part["spec_block"]
            specs.update(part.get("candidate_blocks") or {})
            readings.append(reading)
        tool_context.state[ORCHESTRATOR_SPECS_KEY] = specs
        found: dict[str, Any] = {"kind": digest.get("kind") or "resolution", "parts": readings}
        if not readings:
            found["status"] = digest.get("status")
            found["warnings"] = [w.get("message") for w in digest.get("warnings") or []]
        # `graph_envelope`'s `skills`: every skill the result names, as
        # `{"id", "description"}`, most specific first.
        skills = digest.get("skills") or []
        if skills:
            found["skills"] = skills[:_SKILLS_SHOWN]
            if len(skills) > _SKILLS_SHOWN:
                found["more_skills"] = len(skills) - _SKILLS_SHOWN
        return found

    async def fetch(
        spec_id: str, tool_context: ToolContext, scope: GetDataScope | None = None
    ) -> dict[str, Any]:
        """Retrieve the data behind a spec_id a search returned.

        `scope` narrows or widens it: `entities` (role to place names),
        `window` and `facets` each replace that whole member of the
        definition; omit what should stay as searched. A family of
        indicators that adds up is retrieved as its total.

        Returns a one-line summary and the frames the data landed in, each
        with an id to chart and a view listing its dimensions and their
        values; or the reason it could not be retrieved.
        """
        spent = _spend(tool_context, "fetch")
        if spent:
            return {"error": spent}
        args: dict[str, Any] = {"spec_id": spec_id}
        if scope is not None:
            args["scope"] = GetDataScope.model_validate(scope).model_dump(exclude_none=True)
        # A spec a plan's step retrieved keeps the plan's reading; one a
        # search returned is read by the compiler's own family rule, and
        # posts the StatVars its block elected, as a plan's retrieval does.
        reading = (tool_context.state.get(ORCHESTRATOR_READINGS_KEY) or {}).get(spec_id)
        block = (tool_context.state.get(ORCHESTRATOR_SPECS_KEY) or {}).get(spec_id)
        source = None
        if block:
            try:
                source = source_from_spec_block(block)
            except ValueError:
                logger.warning("fetch: spec block for %s is not readable as a source", spec_id)
        if reading:
            args["stat_vars"] = reading
        elif source is not None and is_family(source):
            args["stat_vars"] = default_reading(source).value
        if source is not None and source.stat_vars:
            args["resolved_stat_var_dcids"] = list(source.stat_vars)
        data = await executor(tool="get_data", args=args)
        if data.get("outcome") != "ok":
            refusal = data.get("refusal") or {}
            return {
                "spec_id": spec_id,
                "refusal": refusal.get("message"),
                "code": refusal.get("code"),
            }
        described: list[tuple[dict[str, Any], dict[str, Any]]] = []
        unavailable: list[str] = []
        for ref in data.get("frame_refs") or []:
            view = await executor(tool="describe_frame", args={"frame_ref": ref})
            if view.get("outcome") == "ok":
                described.append((ref, view.get("view") or {}))
            else:
                unavailable.append((view.get("refusal") or {}).get("message") or "not described")
        frames: dict[str, Any] = dict(tool_context.state.get(ORCHESTRATOR_FRAMES_KEY) or {})
        found: dict[str, Any] = {
            "spec_id": spec_id,
            "summary": data.get("summary"),
            "frames": register_frames(frames, described),
        }
        tool_context.state[ORCHESTRATOR_FRAMES_KEY] = frames
        if unavailable:
            found["frames_unavailable"] = unavailable
        return found

    async def _draw(charts: list[Chart], tool_context: ToolContext) -> list[dict[str, Any]]:
        """Draw `charts`, up to `MAX_CHARTS_PER_CALL`, and count the call as one
        that drew a chart or one that drew none."""
        frames: dict[str, Any] = tool_context.state.get(ORCHESTRATOR_FRAMES_KEY) or {}
        drawn: list[dict[str, Any]] = []
        landed = False
        try:
            for raw in charts[:MAX_CHARTS_PER_CALL]:
                chart = Chart.model_validate(raw)
                frame = frames.get(chart.frame)
                if frame is None:
                    drawn.append({"title": chart.title, "error": f"unknown frame {chart.frame!r}"})
                    continue
                try:
                    select = json.loads(chart.select_json)
                except json.JSONDecodeError:
                    drawn.append({"title": chart.title, "error": "select_json is not valid JSON"})
                    continue
                issue = select_binding_ok(type_=chart.type, select=select, view=frame["view"])
                if issue is not None:
                    drawn.append({"title": chart.title, "error": issue["message"]})
                    continue
                intent = {
                    "type": chart.type,
                    "frame_ref": frame["ref"],
                    "select": select,
                    "title": chart.title[:120],
                }
                written = await executor(tool="write_component", args={"intent": intent})
                component = written.get("component") if written.get("outcome") == "ok" else None
                if component is None or not usable_component(component):
                    refusal = written.get("refusal") or {}
                    failed = {
                        "title": chart.title,
                        "error": refusal.get("message", "the platform refused it"),
                    }
                    # presentation-mcp's `fix`: the values the platform would
                    # accept, as `label (ref)`, for the retry's select.
                    if refusal.get("advice"):
                        failed["advice"] = refusal["advice"]
                    drawn.append(failed)
                    continue
                land = getattr(executor, "land_component", None)
                if land is not None:
                    land(component, land=True, revises=None)
                landed = True
                drawn.append({"title": chart.title, "facts": component["fact_summary"]})
        finally:
            # After the awaits, so the counts are read afresh; in a `finally`,
            # so a call the executor raised out of still counts.
            _count(tool_context, "make_charts" if landed else "make_charts_failed")
        return drawn

    async def make_charts(charts: list[Chart], tool_context: ToolContext) -> dict[str, Any]:
        """Draw charts on the user's canvas.

        Each chart names a frame id from fetch or a plan, a type, a title,
        and select_json: a JSON object mapping at least one dimension in that
        frame's view to the value (or, on one dimension, the list of values)
        to show, e.g. {"observationAbout": "country/KEN"}. A dimension the
        select leaves unbound becomes one series per value, so a bar chart
        comparing places, donors or categories lists the compared values (for
        a ranking, the ranked ones) on that dimension.
        Returns what each chart shows, or why it could not be drawn and, when
        the platform names them, the values it would accept.
        """
        lock = chart_locks.get(tool_context.invocation_id)
        if lock is None:
            lock = chart_locks[tool_context.invocation_id] = asyncio.Lock()
        async with lock:
            spent = _charts_spent(tool_context)
            if spent:
                return {"error": spent}
            drawn = await _draw(charts, tool_context)
        result: dict[str, Any] = {"charts": drawn}
        if len(charts) > MAX_CHARTS_PER_CALL:
            result["not_drawn"] = (
                f"{len(charts) - MAX_CHARTS_PER_CALL} charts past the "
                f"{MAX_CHARTS_PER_CALL} one call may draw"
            )
        return result

    def ask_user(question: str, options: list[str], tool_context: ToolContext) -> dict[str, Any]:
        """Ask the user a question before going on, offering the likely answers.

        `options` are the answers to choose from, at least one; the user may
        also write their own or skip. End your turn after calling it; the
        answer comes back as the next message.
        """
        if not options:
            return {"error": "ask_user needs at least one option to offer."}
        spent = _spend(tool_context, "ask_user")
        if spent:
            return {"error": spent}
        return _pend(tool_context, PendingAsk(question=question, options=list(options)))

    def plan(brief: str, tool_context: ToolContext) -> dict[str, Any]:
        """Hand a question to the planner, which constructs from the data
        what no published series states on its own: a share, ratio or
        per-person figure; growth, change or a drop over a period; a ranking
        or comparison across many places, or the places where one measure
        exceeds another; a count of places meeting a condition; a sum across
        donors, recipients or sectors; or a plan the user asked to see.
        Retrieving a measure a series already states is search and fetch.

        `brief` is the whole question in full: every measure, place and
        period. The user approves the plan before it runs. End your turn
        after calling it; the plan's result comes back as the next message.
        """
        spent = _spend(tool_context, "plan")
        if spent:
            return {"error": spent}
        return _pend(tool_context, PendingPlan(brief=brief))

    async def read_skill(
        tool: ToolId, arguments: dict[str, Any], tool_context: ToolContext
    ) -> dict[str, Any]:
        """data-mcp's skill text for one call, or the refusal's message.

        A call that does not complete, `compiler.surface`'s per-tool timeout or
        an unreachable data-mcp, is an error the model may retry, as
        `planner.skills` answers the planner, so the turn goes on.
        """
        spent = _spend(tool_context, tool.name)
        if spent:
            return {"error": spent}
        if surface is None:
            return {"error": "skills are not available on this server"}
        try:
            result = await surface.call_tool(tool=tool, arguments=arguments)
        except ToolSurfaceUnavailableError as exc:
            logger.warning(
                "orchestrator %s(%s) did not complete: %s", tool.name, dict(arguments), exc
            )
            return {
                "error": f"{tool.name} could not reach data-mcp just now. Call it again, or "
                "answer without this skill."
            }
        if result.is_error:
            body = result.structured_content or {}
            return {
                "error": str(body.get("message") or result.text or "the skill could not be loaded")
            }
        return {"skill": result.text}

    async def get_skill(skill_id: str, tool_context: ToolContext) -> dict[str, Any]:
        """Load a skill a search result named: the definitions, caveats and
        methods an answer in its domain rests on. Returns the first page of
        the skill's text, then the files it links. A page that stops short
        closes with the get_skill_file call that reads on."""
        return await read_skill(GET_SKILL, {"skill_id": skill_id}, tool_context)

    async def get_skill_file(
        skill_id: str,
        path: str,
        tool_context: ToolContext,
        offset: int = 0,
        digest: str | None = None,
    ) -> dict[str, Any]:
        """Read one file of a skill get_skill loaded, or the next page of one.

        `path` is a file exactly as get_skill lists it, or SKILL.md. A page
        that stops short of the end closes with a line naming the next call,
        with its `offset` and `digest`; make that call to read on. A file
        refused as not found is not retried under a guessed path.
        """
        arguments: dict[str, Any] = {"skill_id": skill_id, "path": path, "offset": offset}
        if digest is not None:
            arguments["digest"] = digest
        return await read_skill(GET_SKILL_FILE, arguments, tool_context)

    async def explain_dfo() -> dict[str, Any]:
        """What the Development Finance Observatory is and what this assistant
        can do, for a question about DFO itself or about what can be asked.
        Returns DFO's own description, or an error saying why it could not
        be read, which you pass on plainly."""
        faq = await read_dfo_faq(dfo_faq_surface)
        if isinstance(faq, DfoFaqUnavailable):
            return {"error": faq.reply}
        return {"explainer": faq}

    return [search, fetch, make_charts, ask_user, plan, get_skill, get_skill_file, explain_dfo]


__all__ = ["BUDGETS", "DOWNLOAD_BUDGETS", "TOOL_NAMES", "Chart", "build_orchestrator_tools"]
