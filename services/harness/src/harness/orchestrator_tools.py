"""The orchestrator's tools, over the resolver, executor and tool surface the graph uses.

`search` asks data-mcp's `search` once for every part of a question;
`fetch` retrieves one dataset and describes its frames; `make_charts`
draws on the canvas through the graph's own presentation checks;
`get_skill` loads a skill a search result named and `get_skill_file`
reads on through it; `recall` returns an earlier turn's plan record,
readings or a canvas component in full; `explain_dfo` reads dfo-faq, DFO's
own description of itself (`harness.dfo_faq`). `ask_user` and `plan` record
one pending action for the workflow (`harness.orchestrator_workflow`) to act
on once the model's turn ends: a pause for the user's answer, or the graph's
plan path.

The model sees digests only: part readings, frame views, one-line
summaries and chart facts, never rows. Frames are named to it by short
ids (`harness.graph.frame_id_for`) kept in session state, so `make_charts`
can only draw a frame `fetch` or an executed plan produced this turn, after
`describe_frame` answered for it. `search` also stores every part it
read, failed ones included, as a `harness.turn_records.SearchReading`
holding more than the model sees, and `fetch` adds its outcome to the
readings of its spec_id. A `fetch` that would retrieve what an earlier
`fetch` this turn retrieved returns that fetch's frames without a call
(`_fetch_key`). A `fetch` of a spec_id that a fetch or a plan step was
refused as `capability_unavailable` this turn is refused without a call. A
`get_data` call that does not complete, through a transport failure or an MCP
error, is a refusal the model may retry, as an unreachable skill read is.
Each tool's calls per turn are capped (`BUDGETS`). The download surface
fetches one dataset and never plans.
"""

from __future__ import annotations

import asyncio
import json
import logging
import weakref
from collections.abc import Awaitable, Callable, Mapping
from datetime import UTC, datetime
from typing import Any, Literal

import httpx2
from compiler.sources import (
    default_reading,
    is_family,
    source_from_spec_block,
)
from compiler.surface import ToolSurface, ToolSurfaceUnavailableError
from data_mcp.get_data_tool import Breakdown, GetDataScope
from data_mcp.search_tool import MAX_SOURCE_CHARS, SearchPart
from data_mcp.tool_errors import ToolErrorCode
from google.adk.tools import ToolContext
from mcp import MCPError
from mcp.types import REQUEST_TIMEOUT
from planner.plan import MAX_ASK_CHARS, MAX_PLAN_ASKS, earlier_block
from planner.skills import GET_SKILL, GET_SKILL_FILE
from planner.workflow import ToolId
from pydantic import BaseModel, Field

from harness.dfo_faq import DfoFaqUnavailable, read_dfo_faq
from harness.graph import (
    LINE_CHART_MAX_VALUES,
    MAX_SERIES_PER_COMPONENT,
    ORCHESTRATOR_CALLS_KEY,
    ORCHESTRATOR_FRAMES_KEY,
    ORCHESTRATOR_PENDING_KEY,
    ORCHESTRATOR_READINGS_KEY,
    ORCHESTRATOR_SEARCHES_KEY,
    ORCHESTRATOR_SPEC_WARNINGS_KEY,
    ORCHESTRATOR_SPECS_KEY,
    ORCHESTRATOR_UNAVAILABLE_KEY,
    Executor,
    PendingAsk,
    PendingPlan,
    Resolver,
    pending_action,
    register_frames,
    select_binding_ok,
)
from harness.studio_state import RecallPart, RecallRefusedError
from harness.tool_contracts import ResolverError, chosen_part, reading_facts
from harness.turn_records import SearchReading
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
    "recall": 3,
}
# Charts one make_charts call may draw.
MAX_CHARTS_PER_CALL = 8
# The download surface fetches one dataset and never plans.
DOWNLOAD_BUDGETS = {**BUDGETS, "fetch": 1, "plan": 0}
# The turn's completed fetches: `_fetch_key` -> the spec_id fetched, its
# summary and the ids of its frames, every one described.
_FETCHES_KEY = "orchestrator:fetches"
# The refusal code of a `fetch` whose `get_data` call did not complete.
FETCH_UNAVAILABLE = "transient_unavailable"
# The refusal code of a `fetch` whose `get_data` call ran past its deadline.
FETCH_TIMED_OUT = "deadline_exceeded"
# The key of the result of an `ask_user` or `plan` call that recorded its
# action; the Studio stream holds back later text only after one of these.
HANDED_OFF = "handed_off"
# The identity member each `get_data` scope member replaces.
_SCOPE_REPLACES = {"entities": "entities", "window": "requested_window", "facets": "facets"}
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
    "recall",
)


class Chart(BaseModel):
    """One chart to draw from a frame `fetch` or a plan returned."""

    frame: str = Field(description="the id of a frame a fetch or a plan returned this turn")
    type: Literal["line_chart", "bar_chart", "stat_card", "table"] = Field(
        description=(
            "line_chart: a series over time, or up to "
            f"{LINE_CHART_MAX_VALUES} places over time as one chart with the place "
            "dimension listing them; bar_chart: places or categories "
            "compared at one period; stat_card: one headline value; table: the detail. "
            "A chart shows every period its frame covers. A frame whose rows all cover "
            "one period (a collapsed window such as 2020/2024) has no run of periods to "
            "draw as a line: show it as a bar_chart, a table or a stat_card. A facet "
            "dimension (provenance, unit, measurementMethod) holding several values in "
            "the frame's view is bound to one of them in select_json, whatever the type."
        )
    )
    title: str = Field(
        description=(
            "what the chart shows and the period it shows; a stat card's title carries "
            "the period of the value on the card"
        )
    )
    select_json: str = Field(
        description=(
            "a JSON object mapping at least one dimension in the frame's view to the "
            "value, or on one dimension the list of values, to show, e.g. "
            '{"observationAbout": "country/KEN"} or '
            '{"observationAbout": ["country/KEN", "country/UGA", "country/TZA"]}. A '
            "dimension the select leaves unbound becomes one series per value, so a "
            "bar chart comparing places, donors or categories lists the compared values "
            "(for a ranking, the ranked ones) on that dimension. A chart draws at most "
            f"{MAX_SERIES_PER_COMPONENT} series, so bind enough dimensions to keep the "
            "series count under that, or split the data across charts."
        )
    )


class FetchScope(GetDataScope):
    """How a fetch narrows or widens the searched data. Each member given,
    `entities`, `window` or `facets`, replaces that whole member of the
    searched definition; a member left out stays as searched."""


# data-mcp's own part has a `source` description that points at its
# `get_data`. The orchestrator's retrieval tool is `fetch`.
class OrchestratorSearchPart(SearchPart):
    """One variable ask within the request, with its own entities, time, unit, and source."""

    source: str | None = Field(
        default=None,
        min_length=1,
        max_length=MAX_SOURCE_CHARS,
        description=(
            "who reported it, refused when no unit is given and ignored when one is, so "
            "choose the source through fetch's facets instead"
        ),
    )


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


def _asks_refusal(asks: list[str]) -> dict[str, Any] | None:
    """The refusal of a `plan` call whose `asks` break the planner's limits
    (`planner.plan.MAX_PLAN_ASKS`, `MAX_ASK_CHARS`), or None."""
    problems = []
    if len(asks) > MAX_PLAN_ASKS:
        problems.append(
            f"asks lists {len(asks)} outputs, more than {MAX_PLAN_ASKS}: merge outputs the "
            "same figure or chart answers"
        )
    problems += [
        f"ask {number} is empty: name the output or leave it out"
        for number, ask in enumerate(asks)
        if not ask.strip()
    ]
    problems += [
        f"ask {number} runs {len(ask.strip())} characters, more than {MAX_ASK_CHARS}: "
        "name the output in fewer words"
        for number, ask in enumerate(asks)
        if len(ask.strip()) > MAX_ASK_CHARS
    ]
    if not problems:
        return None
    return {
        "refusal": f"plan recorded nothing: {'; '.join(problems)}.",
        "code": ToolErrorCode.REQUEST_INVALID.value,
        "legal": {"max_asks": MAX_PLAN_ASKS, "ask_chars": [1, MAX_ASK_CHARS]},
    }


def _pend(tool_context: ToolContext, action: PendingAsk | PendingPlan) -> dict[str, Any]:
    """Record `action` as the turn's pending action, unless one is pending."""
    pending = pending_action(tool_context.state)
    if pending is not None:
        return {
            "error": f"This turn is already waiting on {pending.kind}; one hand-off per turn. "
            "End your turn now."
        }
    tool_context.state[ORCHESTRATOR_PENDING_KEY] = action.model_dump()
    return {
        "status": "Recorded. End your turn now; the result comes back as the next message.",
        HANDED_OFF: action.kind,
    }


def _spec_block(tool_context: ToolContext, spec_id: str) -> Mapping[str, Any] | None:
    """The spec block this turn's search returned for `spec_id`, else the one
    an earlier turn of the conversation stored (`earlier_block`); `None`
    when neither did."""
    state = tool_context.state
    block = (state.get(ORCHESTRATOR_SPECS_KEY) or {}).get(spec_id)
    return block if block is not None else earlier_block(spec_id)


def _attach_fetch(tool_context: ToolContext, result: dict[str, Any]) -> None:
    """Add a `fetch` result to every stored search reading of its spec_id.

    A spec_id that no reading holds as its own, but one offered as a
    candidate, first gains a reading of its own, read from its block
    (`chosen_part`) with the offering reading's text and skills and its
    own warnings.
    """
    spec_id = result["spec_id"]
    state = tool_context.state
    searches = [
        SearchReading.model_validate(stored)
        for stored in state.get(ORCHESTRATOR_SEARCHES_KEY) or []
    ]
    block = (state.get(ORCHESTRATOR_SPECS_KEY) or {}).get(spec_id)
    offered = next((r for r in searches if any(c.spec_id == spec_id for c in r.candidates)), None)
    if block and offered is not None and all(r.spec_id != spec_id for r in searches):
        warnings = (state.get(ORCHESTRATOR_SPEC_WARNINGS_KEY) or {}).get(spec_id) or []
        searches.append(
            SearchReading.from_part(
                {**chosen_part(block, variable=offered.text), "warnings": warnings},
                skills=offered.skills,
                observed_at=datetime.now(UTC),
            )
        )
    state[ORCHESTRATOR_SEARCHES_KEY] = [
        (reading.with_fetch(result) if reading.spec_id == spec_id else reading).model_dump(
            mode="json"
        )
        for reading in searches
    ]


def _fetch_key(args: Mapping[str, Any], block: Mapping[str, Any] | None) -> str:
    """What a `get_data` call with `args` retrieves, as a canonical JSON key.

    Two calls with one key retrieve the same data. A spec_id stands for its
    definition's identity, so the key holds the identity in its place, less
    what the call's own arguments settle: each scope member replaces its
    identity member. Two spec_ids whose remaining identity members agree
    share a key.
    A spec with no identity on record keys on its spec_id.

    The identity's `open_roles` stay in the key whatever the scope binds, so
    an open-origin spec and a plain one scoped to the same debtor key apart.
    A scope that binds an open axis yields a key of its own, so the fetch
    misses the cache and retrieves the matching data.
    """
    identity = ((block or {}).get("definition") or {}).get("identity")
    if not isinstance(identity, Mapping):
        return json.dumps(args, sort_keys=True)
    standing = dict(identity)
    scope = args.get("scope") or {}
    for member, field in _SCOPE_REPLACES.items():
        if member in scope:
            standing.pop(field, None)
    retrieved = {key: value for key, value in args.items() if key != "spec_id"}
    return json.dumps({**retrieved, "identity": standing}, sort_keys=True)


def build_orchestrator_tools(
    *,
    resolver: Resolver,
    executor: Executor,
    surface: ToolSurface | None = None,
    dfo_faq_surface: ToolSurface | None = None,
    recall_part: Callable[..., Awaitable[dict[str, Any]]] | None = None,
) -> list[Any]:
    """The orchestrator's tools, bound to this server's resolver, executor and tool surface.

    `dfo_faq_surface` is the surface `explain_dfo` reads dfo-faq through: the
    bare one, off the Activity feed, since `explain_dfo`'s own call is the
    step the user sees. Left unset, `explain_dfo` says it cannot describe DFO.
    `recall_part` is `harness.studio_state.make_recall_part`'s seam. Left
    unset, `recall` says there is nothing to recall from.
    """
    # One lock per invocation, held by a make_charts call from its budget
    # check to its count, so concurrent calls draw one after another and each
    # is admitted on exact counts. An entry lasts while a call holds its lock.
    chart_locks: weakref.WeakValueDictionary[str, asyncio.Lock] = weakref.WeakValueDictionary()

    async def search(
        query: str, parts: list[OrchestratorSearchPart], tool_context: ToolContext
    ) -> dict[str, Any]:
        """Search DFO's data for every measured thing a question asks for.

        `query` is the question in the user's words. A follow-up that leans
        on earlier turns ("the same for Ghana", "show those over time") is
        sent as one self-contained question naming the measure, places and
        period. `parts` is one entry per measured variable in it, each with
        the entities, time, unit or source the question gives for that
        variable, as spoken. All parts go in one call. A flow measure such
        as aid, donors or recipients needs its entity's role, as spoken:
        `to` for a recipient, `from` for a donor.

        Returns the result's `kind`: a `resolution` reads each part
        separately; a `catalogue` means the question asked what data DFO
        holds, answered with a scope and no parts; a `declined` means
        Search would not answer, with a reason, such as a question outside
        DFO's scope or one that needs analysis. Then
        `parts`, one per part: its `status` (definite: a spec_id to fetch;
        candidates: readings to choose between, each with a spec_id and
        label; no_data or unsupported: with warnings and, for no_data, a
        `reason` code saying why), what it resolved to in words, its
        warnings, and the window it covers. And
        `skills`: background a result calls for, each by id and
        description, loaded with get_skill, and `more_skills`, how many
        more the result names.
        """
        spent = _spend(tool_context, "search")
        if spent:
            return {"error": spent}
        wire_parts = [
            OrchestratorSearchPart.model_validate(part).model_dump(exclude_none=True)
            for part in parts
        ]
        try:
            digest = await resolver(question=query, parts=wire_parts)
        except ResolverError as exc:
            return {"error": (exc.refusal or {}).get("message") or str(exc)}
        state = tool_context.state
        specs = dict(state.get(ORCHESTRATOR_SPECS_KEY) or {})
        spec_warnings = dict(state.get(ORCHESTRATOR_SPEC_WARNINGS_KEY) or {})
        searches = list(state.get(ORCHESTRATOR_SEARCHES_KEY) or [])
        skill_ids = [skill["id"] for skill in digest.get("skills") or []]
        observed_at = datetime.now(UTC)
        readings = []
        for part in digest.get("part_digests") or []:
            reading = {
                key: value
                for key, value in part.items()
                if key not in ("spec_block", "candidate_blocks")
            }
            blocks = dict(part.get("candidate_blocks") or {})
            facts: dict[str, Any] = {}
            if part.get("spec_block") and part.get("spec_id"):
                blocks[part["spec_id"]] = part["spec_block"]
                facts = reading_facts(part["spec_block"])
            specs.update(blocks)
            spec_warnings.update(dict.fromkeys(blocks, reading.get("warnings") or []))
            stored = SearchReading.from_part(
                {**reading, **facts}, skills=skill_ids, observed_at=observed_at
            )
            searches.append(stored.model_dump(mode="json"))
            readings.append(reading)
        state[ORCHESTRATOR_SPECS_KEY] = specs
        state[ORCHESTRATOR_SPEC_WARNINGS_KEY] = spec_warnings
        state[ORCHESTRATOR_SEARCHES_KEY] = searches
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
        spec_id: str,
        tool_context: ToolContext,
        scope: FetchScope | None = None,
        stat_vars: Literal["sum", "separate"] | None = None,
        breakdown: Breakdown | None = None,
    ) -> dict[str, Any]:
        """Retrieve the data behind a spec_id a search returned.

        `scope` narrows it to the places or period the user meant, or
        widens it. A family of
        indicators that adds up is retrieved as its total unless
        `stat_vars` says otherwise: "sum" for one total, "separate" for one
        series per indicator. `breakdown` totals the family within each
        code of one code system at one level, e.g. `{"code_system":
        "crsPurpose", "level": 0}` for aid by sector group.

        Returns a one-line summary and the frames the data landed in, each
        with an id to chart and a view listing its dimensions and their
        values; or a refusal with its `code` and reason. The code
        `capability_unavailable` means this deployment cannot do what the
        call asked. A fetch of the same data as an earlier fetch this turn
        returns that fetch's frames, with `same_data_as` naming its spec_id.
        """
        unavailable = (tool_context.state.get(ORCHESTRATOR_UNAVAILABLE_KEY) or {}).get(spec_id)
        if unavailable is not None:
            return {
                "spec_id": spec_id,
                "refusal": f"This data was refused earlier this turn: {unavailable} Fetching it "
                "again, with or without that option, does not answer the part that needed it.",
                "code": ToolErrorCode.CAPABILITY_UNAVAILABLE.value,
            }
        spent = _spend(tool_context, "fetch")
        if spent:
            return {"error": spent}
        args: dict[str, Any] = {"spec_id": spec_id}
        if scope is not None:
            args["scope"] = FetchScope.model_validate(scope).model_dump(exclude_none=True)
        # A stated reading wins, and a breakdown reads the family separately.
        # A spec a plan's step retrieved keeps the plan's reading; one a
        # search returned is read by the compiler's own family rule.
        reading = stat_vars or (
            "separate"
            if breakdown is not None
            else (tool_context.state.get(ORCHESTRATOR_READINGS_KEY) or {}).get(spec_id)
        )
        block = _spec_block(tool_context, spec_id)
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
        if breakdown is not None:
            args["breakdown"] = Breakdown.model_validate(breakdown).model_dump()
        key = _fetch_key(args, block)
        earlier = (tool_context.state.get(_FETCHES_KEY) or {}).get(key)
        if earlier is not None:
            frames = tool_context.state.get(ORCHESTRATOR_FRAMES_KEY) or {}
            found = {
                "spec_id": spec_id,
                "summary": earlier["summary"],
                "frames": [{"frame": f, "view": frames[f]["view"]} for f in earlier["frames"]],
                "same_data_as": earlier["spec_id"],
            }
            _attach_fetch(tool_context, found)
            return found
        try:
            data = await executor(tool="get_data", args=args)
        except (MCPError, httpx2.TransportError) as exc:
            logger.warning(
                "fetch: get_data(%s) did not complete: %s",
                spec_id,
                exc,
                exc_info=True,
            )
            if isinstance(exc, MCPError) and exc.error.code == REQUEST_TIMEOUT:
                refused = {
                    "spec_id": spec_id,
                    "refusal": "get_data ran past its deadline. Do not repeat the same fetch "
                    "unchanged: narrow it, or answer without this data.",
                    "code": FETCH_TIMED_OUT,
                }
            else:
                refused = {
                    "spec_id": spec_id,
                    "refusal": "get_data could not reach data-mcp just now. Fetch it again, or "
                    "answer without this data.",
                    "code": FETCH_UNAVAILABLE,
                }
            _attach_fetch(tool_context, refused)
            return refused
        if data.get("outcome") != "ok":
            refusal = data.get("refusal") or {}
            refused = {
                "spec_id": spec_id,
                "refusal": refusal.get("message")
                or "get_data refused this fetch and gave no reason.",
                "code": refusal.get("code"),
            }
            if refused["code"] == ToolErrorCode.CAPABILITY_UNAVAILABLE.value:
                tool_context.state[ORCHESTRATOR_UNAVAILABLE_KEY] = {
                    **(tool_context.state.get(ORCHESTRATOR_UNAVAILABLE_KEY) or {}),
                    spec_id: refused["refusal"],
                }
            _attach_fetch(tool_context, refused)
            return refused
        described: list[tuple[dict[str, Any], dict[str, Any]]] = []
        unavailable: list[str] = []
        for ref in data.get("frame_refs") or []:
            view = await executor(tool="describe_frame", args={"frame_ref": ref})
            if view.get("outcome") == "ok":
                described.append((ref, view.get("view") or {}))
            else:
                unavailable.append((view.get("refusal") or {}).get("message") or "not described")
        frames = dict(tool_context.state.get(ORCHESTRATOR_FRAMES_KEY) or {})
        found = {
            "spec_id": spec_id,
            "summary": data.get("summary"),
            "frames": register_frames(frames, described),
        }
        tool_context.state[ORCHESTRATOR_FRAMES_KEY] = frames
        if unavailable:
            found["frames_unavailable"] = unavailable
        else:
            fetches = dict(tool_context.state.get(_FETCHES_KEY) or {})
            fetches[key] = {
                "spec_id": spec_id,
                "summary": found["summary"],
                "frames": [entry["frame"] for entry in found["frames"]],
            }
            tool_context.state[_FETCHES_KEY] = fetches
        _attach_fetch(tool_context, found)
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
                        "error": refusal.get("message") or "the platform refused it",
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

    def plan(brief: str, asks: list[str], tool_context: ToolContext) -> dict[str, Any]:
        """Hand a question to the planner, which builds a plan that
        constructs the answer from the data. The user approves the plan
        before it runs.

        `brief` is the whole question in full: every measure, place and
        period. `asks` lists each output the user asked for, one per entry,
        in the user's terms: a number, a ranking, a chart, a comparison; at
        most six, each one short line. A figure over a span is an ask of its
        own: "the total between 2010 and the latest" is one cumulative
        number beside the yearly figures. A view added to help read the
        answer is not an ask. The planner answers or declines each ask, so
        an output left off the list can drop out of the plan unnoticed.
        End your turn after calling it; the plan's result comes back as the
        next message.
        """
        refused = _asks_refusal(asks)
        if refused:
            return refused
        spent = _spend(tool_context, "plan")
        if spent:
            return {"error": spent}
        return _pend(tool_context, PendingPlan(brief=brief, asks=[ask.strip() for ask in asks]))

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
        can do. Returns DFO's own description, or an error saying why it
        could not be read, which you pass on plainly."""
        faq = await read_dfo_faq(dfo_faq_surface)
        if isinstance(faq, DfoFaqUnavailable):
            return {"error": faq.reply}
        return {"explainer": faq}

    async def recall(
        part: RecallPart,
        tool_context: ToolContext,
        turn: int | None = None,
        component_id: str | None = None,
    ) -> dict[str, Any]:
        """Read back, in full, what an earlier turn of this conversation did:
        every source and decision of a plan, what a search read, or a canvas
        component's stored entry.

        `part`: "plan" for the plan records of a turn, "readings" for the
        search readings of a turn, "canvas" for the stored entry of the
        component `component_id`. `turn` is the turn's number as the
        conversation digest counts them; leave it out for the newest turn
        that has the part. A target that does not exist is refused with the
        values that do.
        """
        spent = _spend(tool_context, "recall")
        if spent:
            return {"error": spent}
        if recall_part is None:
            return {"error": "recall is not available on this server"}
        try:
            return await recall_part(part=part, turn=turn, component_id=component_id)
        except RecallRefusedError as exc:
            return {
                "refusal": str(exc),
                "code": ToolErrorCode.REQUEST_INVALID.value,
                "legal": exc.legal,
            }

    return [
        search,
        fetch,
        make_charts,
        ask_user,
        plan,
        get_skill,
        get_skill_file,
        explain_dfo,
        recall,
    ]


__all__ = ["BUDGETS", "DOWNLOAD_BUDGETS", "TOOL_NAMES", "Chart", "build_orchestrator_tools"]
