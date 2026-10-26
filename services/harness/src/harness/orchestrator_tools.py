"""The orchestrator's tools, over the resolver, executor and tool surface the workflow uses.

`search` asks data-mcp's `search` once for every part of a question;
`fetch` retrieves one dataset and describes its frames; `make_charts`
draws on the canvas through the graph's own presentation checks;
`get_skill` loads a skill a search result named and `get_skill_file`
reads on through it; `recall` returns an earlier turn's plan record,
readings or a canvas component in full; `explain_dfo` reads dfo-faq, DFO's
own description of itself, beside the live counts of what DFO holds
(`harness.dfo_faq`). `ask_user` and `plan` record one pending action for
the workflow's `route` node (`harness.graph`) to act on once the model's
turn ends: a pause for the user's answer, or the plan path.

The model sees digests only: part readings, frame views, one-line
summaries, and of a drawn card what it holds and the figures a reply may
state (`harness.reply_figures`). It sees no rows. Frames are named to it by short
ids (`harness.graph.state.frame_id_for`) kept in session state, so `make_charts`
can only draw a frame `fetch` or an executed plan produced this turn, after
`describe_frame` answered for it. `search` also stores every part it
read, failed ones included, as a `harness.turn_records.SearchReading`
holding more than the model sees, and `fetch` adds its outcome to the
readings of its spec_id. A `fetch` of a spec_id with no stored block but a
definition on record, such as another count a search result offers, first
reads its own block from Search's answer for that definition
(`_recorded_block`), the way a plan's pin is read. A `fetch` that would retrieve what an earlier
`fetch` this turn retrieved returns that fetch's frames without a call
(`_fetch_key`). A `fetch` of a spec_id that a fetch or a plan step was
refused as `capability_unavailable` this turn is refused without a call. A
`get_data` call that does not complete, through a transport failure or an MCP
error, is a refusal the model may retry, as an unreachable skill read is.
Each tool's calls per turn are capped (`harness.graph.state.BUDGETS`), except
`make_charts`' calls that draw nothing, which are capped per frame arrival: a
fetch or plan hand-back that adds a frame clears their count. The download
surface fetches one dataset and never plans.

Every tool validates the model's arguments against its annotations before its
body runs (`_validated_arguments`), so a malformed argument comes back to the
model as an error naming the field, and the body receives model instances.
"""

from __future__ import annotations

import asyncio
import functools
import inspect
import json
import logging
import re
import weakref
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from types import FunctionType
from typing import Any, Literal, Protocol, get_args, get_type_hints

import httpx2
from analysis_mcp.frame_ref_resolution import read_derived_marker
from analysis_mcp.identity import OP_SPEC_HASH_PREFIX
from common.materialisation import FrameRef
from common.materialisation import get_store as get_frame_store
from common.place_type import (
    TYPED_ROLES,
    enrichment_role,
    kind_column,
    type_column,
)
from common.row_budget import row_budget
from compiler.sources import (
    default_reading,
    is_family,
    source_from_spec_block,
)
from compiler.surface import ToolSurface, ToolSurfaceUnavailableError
from data_mcp.get_data_tool import Breakdown, GetDataScope
from data_mcp.search_tool import SearchContext, SearchPart
from data_mcp.tool_errors import ToolErrorCode
from dfo_frame_grammar.grammar import TimeGrain
from dfo_frame_grammar.rolling import RollingError
from google.adk.tools import ToolContext
from mcp import MCPError
from mcp.types import REQUEST_TIMEOUT
from planner.parties import (
    PartyRole,
    column_party_roles,
    open_party,
    party_among,
    party_role,
    words_for_one_party,
)
from planner.plan import (
    MAX_ASK_CHARS,
    MAX_PLAN_ASKS,
    AskBreakdown,
    SessionState,
    earlier_block,
    recorded_identity,
)
from planner.skills import GET_SKILL, GET_SKILL_FILE
from planner.workflow import ToolId
from pydantic import BaseModel, Field, TypeAdapter, ValidationError, model_validator
from translator.discovery import (
    DefinitionNotResolvedError,
    PartNotResolvedError,
    resolve_definition,
    resolved_spec,
)

from harness.chart_binding import (
    LINE_CHART_MAX_VALUES,
    MAX_CHARTS_PER_CALL,
    MAX_SERIES_PER_COMPONENT,
    TopMembers,
    chart_draws,
    covers_every_row,
    differing_coordinates,
    empty_select_refusal,
    every_series_selects,
    latest_shared_period,
    over_cap_dimension,
    select_binding_ok,
    top_members,
)
from harness.conversation import current_user_message
from harness.dfo_faq import (
    CANNOT_DESCRIBE_DFO,
    DfoFaqUnavailable,
    read_catalogue_counts,
    read_dfo_faq,
)
from harness.graph.seams import Executor
from harness.graph.state import (
    ORCHESTRATOR_CALLS_KEY,
    ORCHESTRATOR_FRAMES_KEY,
    ORCHESTRATOR_PENDING_KEY,
    ORCHESTRATOR_PRIOR_LEDGER_KEY,
    ORCHESTRATOR_READINGS_KEY,
    ORCHESTRATOR_SEARCHES_KEY,
    ORCHESTRATOR_SPEC_WARNINGS_KEY,
    ORCHESTRATOR_SPECS_KEY,
    ORCHESTRATOR_UNAVAILABLE_KEY,
    AskQuestion,
    DescribedFrame,
    PendingAsk,
    PendingPlan,
    count_call,
    frame_store_ref,
    pending_action,
    register_frames,
    store_search,
    turn_budgets,
)
from harness.reply_figures import (
    DrawnSeries,
    card_result,
    drawn_series,
    unplaced_names,
    with_coordinates,
)
from harness.rolling_cards import named_year
from harness.scope_choices import PartScope, carried_entries
from harness.studio_state import RecallPart, RecallRefusedError
from harness.tool_contracts import (
    HarnessCode,
    ResolverError,
    chosen_part,
    harness_refusal,
    harness_retry_mode,
    measure_choice,
)
from harness.turn_records import SETTLED_STATUSES, SearchReading, SiblingReading
from harness.wire import RetryMode, usable_component

logger = logging.getLogger(__name__)

# The turn's completed fetches: `_fetch_key` -> the spec_id fetched, its
# summary and the ids of its frames, every one described, or the refusal of a
# fetch that served no observations.
_FETCHES_KEY = "orchestrator:fetches"
# The ids of the series the canvas drew as a turn began, with the turn's
# invocation id: {"invocation", "ids"}.
_CANVAS_SERIES_KEY = "orchestrator:canvas_series"
# The cards `make_charts` drew this turn that are still on the canvas:
# {"invocation", "drawn"}, `drawn` mapping `_card_identity` to the card's
# {"component_id", "revision"}, the revision it was drawn as. `remove_cards`
# drops the cards it takes off.
_DRAWN_CARDS_KEY = "orchestrator:drawn_cards"
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
    "remove_cards",
)


# The most periods a chart counts back from the latest data.
MAX_LATEST_PERIODS = 200


def _latest_window(periods: int) -> dict[str, dict[str, Any]]:
    """The intent window of the `periods` most recent periods: relative periods,
    which the platform resolves against the latest data each time the card is drawn."""
    return {
        "start": {"anchor": "latest", "offset": 1 - periods},
        "end": {"anchor": "latest", "offset": 0},
    }


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
            "what the chart shows, with no period or year: 'Aid to Kenya by donor'. "
            "make_charts adds the period the card shows, and refuses a title naming a "
            "year the frame covers"
        )
    )
    latest_periods: int | None = Field(
        default=None,
        ge=1,
        le=MAX_LATEST_PERIODS,
        description=(
            "set to N to show the N most recent periods counted back from the latest "
            "data, so the chart moves forward when newer data arrives: 5 for 'the last "
            "five years'. Use it when the user counts a period back from now, on a frame "
            "fetched with no window of fixed years. Leave it unset to show every period "
            "the frame covers, and on a frame whose rows all cover one period"
        ),
    )
    select_json: str = Field(
        description=(
            "a JSON object. {} draws every series of the frame when its rows confirm that one "
            "dimension covers them all, and is otherwise refused with the dimensions to "
            "name. Each key is a dimension name the "
            "frame's view lists, and its value one value the view lists for that "
            "dimension, or on one dimension a list of them: where the view lists "
            'observationAbout, {"observationAbout": ["country/KEN", "country/UGA"]}; '
            'where it lists financingRecipient, {"financingRecipient": "country/MWI"}. A '
            "dimension the select leaves unbound becomes one series per value. A chart "
            f"draws at most {MAX_SERIES_PER_COMPONENT} series. When a chart the user asked "
            "to see compares more places, donors or categories than that, leave their "
            "dimension unbound: the chart is cut for display to the largest of them at one "
            "period, and its title says how many and when. The cut is for showing them: "
            "a user asking which are largest wants a ranking calculated, and a cut chart "
            "does not replace it. Never list a subset you chose to stand in for the "
            "largest."
        )
    )
    order_by_value: bool = Field(
        default=False,
        description=(
            "true to list a table's or a line chart's compared places, donors or "
            "categories highest first by their value at the latest period most of them "
            "hold. Leave it false to keep the order select_json lists them in. A bar chart "
            "ignores it: its bars are drawn highest first when their dimension is unbound"
        ),
    )
    state_values_for: list[str] = Field(
        default_factory=list,
        max_length=MAX_SERIES_PER_COMPONENT,
        description=(
            "the series the user asked to be told a value of in words, each named by a "
            "value the frame's view lists for the compared dimension (country/KEN where "
            "the view lists it) or by its label. Leave it empty when the user asked to "
            "see a chart, a ranking, a table or a trend: the card holds those values. "
            "Left empty, a chart of one series returns its latest value in reply_figures. "
            "A chart of several series that share one period and one unit returns its "
            "largest and its smallest, each marked, and every series tying for either "
            "carries the mark; any other chart of several returns none"
        ),
    )
    state_span: bool = Field(
        default=False,
        description=(
            "true when the user asked how those series changed over a period: "
            "reply_figures gets each one's first value and its last, each with its period. "
            "false gives the latest value alone"
        ),
    )
    state_period: str | None = Field(
        default=None,
        description=(
            "a period, when the user asked for those series' value at that one earlier "
            "period: reply_figures gets the value there in place of the latest, and on a "
            "chart of one series the value there and the latest, each with its period. "
            "With state_span it gets the first value, the value there and the last. A "
            "series holding no value at that period is listed under no_value. Written "
            "as the frame's coverage writes periods (for example 2015)"
        ),
    )


class AskedOutput(BaseModel):
    """One output the user asked for, the parties it lists one by one, and
    those it totals across."""

    text: str = Field(description="the output in the user's terms, one short line")
    by: list[PartyRole] = Field(
        default=[],
        description=(
            "each party the output lists one by one, in the words search takes as "
            'open_roles: ["creditor"] for "debt service by creditor"; empty for a total '
            "or a figure about named parties. Parties listed together are shown together "
            "in one output; an output per party is an ask per party"
        ),
    )
    over: list[PartyRole] = Field(
        default=[],
        description=(
            "each party the output totals across into one figure: "
            '["donor"] for "total aid from all donors". Such a party stays out of open_roles, '
            "and the search call names the population as an entity"
        ),
    )
    across_years: bool = Field(
        default=False,
        description=(
            "true when the user asked for this output as one figure, or one ranking, taken "
            'across several years: a total, an average, a peak or a low over them ("the '
            'total between 2015 and 2022", "in total, over all years", "the year aid '
            'peaked"). It permits the plan to combine the years and requires nothing. Leave '
            "it false for a figure for each year and for one year, and for a total or a "
            "ranking when the user named no years: the years a source holds are not a span "
            "the user asked for"
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def _bare_text(cls, value: Any) -> Any:
        """Read an ask sent as its text alone as one with no parties."""
        return {"text": value} if isinstance(value, str) else value

    @model_validator(mode="after")
    def _one_party_a_side(self) -> AskedOutput:
        """Refuse a `by` holding two words for one side of a flow.

        One ask is one output, whose flow has one party on each side, so
        "donor" beside "creditor" lists the parties of two flows, each an
        ask of its own. `over` may hold both: it totals, and the plan
        checks each word on its own (`planner.plan.AskBreakdown`).
        """
        if shared := words_for_one_party(self.by):
            raise ValueError(
                f"`by` lists {', '.join(shared)}, the same side of a flow: one ask is one "
                "output and a flow has one party on each side, so give each of them an ask "
                "of its own"
            )
        return self

    @model_validator(mode="after")
    def _listed_or_totalled(self) -> AskedOutput:
        """Refuse a party the ask both lists one by one and totals across.

        One ask is one output, whose flow has one party on each side, so two
        words on one side name one party (`planner.parties.same_party`):
        `by: ["donor"]` beside `over: ["creditor"]` lists and totals the
        giving party.
        """
        listed = [role for role in dict.fromkeys(self.by) if party_among(role, self.over)]
        totalled = [role for role in dict.fromkeys(self.over) if party_among(role, self.by)]
        if listed:
            raise ValueError(
                f"`by` lists {', '.join(listed)} and `over` totals across "
                f"{', '.join(totalled)}, one party of the flow: an output lists a party one by "
                "one or totals across it, not both"
            )
        return self


class FetchScope(GetDataScope):
    """How a fetch narrows or widens the searched data. Each member given,
    `entities`, `window` or `facets`, replaces that whole member of the
    searched definition; a member left out stays as searched."""


_ANSWER_NOW = "Answer with what you have, and say plainly what you could not do."


async def _frame_work[T](ref: Mapping[str, Any], work: Callable[[Any, Any], T]) -> T | None:
    """What `work` makes of the frame the store holds under `ref` and its
    trace, or None when the store does not hold the frame or cannot read it.
    `work` runs in a worker thread, off the event loop, and the frame's rows
    stay claimed on the process's row budget until that thread returns, also
    when the task awaiting it is cancelled first."""
    try:
        held = await get_frame_store().open(
            FrameRef(spec_id=ref["spec_id"], content_version=ref["content_version"])
        )
        if held is None:
            logger.warning("make_charts: the store holds no frame under %s", ref)
            return None
        if held.num_rows is None:
            logger.warning("make_charts: the frame under %s states no row count", ref)
            return None
        claim = row_budget().claim(held.num_rows)
    except Exception:
        # A ranking leaves the select as the model wrote it when its frame is
        # unread, and an empty select is refused.
        logger.warning("make_charts: could not read %s", ref, exc_info=True)
        return None
    with claim:
        try:
            data, trace = await claim.hold_through(held.read(max_rows=held.num_rows))
        except Exception:
            logger.warning("make_charts: could not read %s", ref, exc_info=True)
            return None
        return await claim.hold_through(asyncio.to_thread(work, data, trace))


def _rows_of(data: Any) -> list[dict[str, Any]]:
    """The rows of a stored frame, each as a mapping."""
    return [dict(zip(data.columns, row, strict=True)) for row in data.rows]


def _stated_frames(data: Mapping[str, Any]) -> list[tuple[dict[str, Any], dict[str, Any] | None]]:
    """Each frame of a `get_data` outcome as its (frame ref, store ref), the
    store ref `None` where the executor states no `store_refs`."""
    refs = data.get("frame_refs") or []
    return list(zip(refs, data.get("store_refs") or [None] * len(refs), strict=True))


def _period_note_entry(held: Mapping[str, Any]) -> dict[str, str]:
    """The `period_note` of a `get_data` outcome or an earlier `fetch` as the
    entry a `fetch` result carries; empty when `held` states none."""
    note = held.get("period_note")
    return {"period_note": note} if note else {}


def _covering_selects(
    data: Any, _trace: Any, *, selects: Sequence[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Those of `selects` that bind only dimensions every row of a stored
    frame holds a value on (`covers_every_row`), in their order."""
    rows = _rows_of(data)
    return [select for select in selects if covers_every_row(select, rows)]


def _held_cells(dimension: str, rows: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    """Each `dimension` member's kind or type cell among `rows`, the kind where
    both are held; empty when the rows carry neither column. A party's class
    and whole cells take no part in a ranking."""
    return {
        str(row[dimension]): str(row[column])
        for row in rows
        for column in (type_column(dimension), kind_column(dimension))
        if row.get(column) is not None and row.get(dimension) is not None
    }


def _role_cells(data: Any, _trace: Any) -> dict[str, dict[str, str]]:
    """Each typed role's members' kind or type cells among a stored frame's
    rows, the roles it carries no enrichment column for left out."""
    rows = _rows_of(data)
    by_role = {role: _held_cells(role, rows) for role in TYPED_ROLES}
    return {role: cells for role, cells in by_role.items() if cells}


async def _read_member_cells(
    frame: Mapping[str, Any], executor: Executor
) -> dict[str, dict[str, str]] | None:
    """Each typed role's members' served kind or type cells, read from the
    same retrieval as `frame` asked again with `place_types`: the fetch's own
    arguments, or for a plan step's frame its spec alone, since a member's
    kind does not change with the period or scope asked. None when the
    retrieval fails, runs past its deadline, is refused, or serves a frame
    the store cannot read."""
    args = frame.get("fetch_args") or {"spec_id": str(frame["ref"]["spec_id"])}
    try:
        data = await executor(tool="get_data", args={**args, "place_types": True})
    except (MCPError, httpx2.TransportError):
        logger.warning("make_charts: the kind read of %s failed", frame["ref"], exc_info=True)
        return None
    if data.get("outcome") != "ok":
        logger.warning("make_charts: the kind read of %s was refused", frame["ref"])
        return None
    cells: dict[str, dict[str, str]] = {}
    for ref, store_ref in _stated_frames(data):
        read = await _frame_work(frame_store_ref({"ref": ref, "store_ref": store_ref}), _role_cells)
        if read is None:
            return None
        for role, members in read.items():
            cells.setdefault(role, {}).update(members)
    return cells


# The key of a registered frame's entry holding its kind read, each typed
# role's member cells.
_MEMBER_CELLS = "member_cells"


async def _member_cells(
    state: Any, frame_id: str, executor: Executor
) -> dict[str, dict[str, str]] | None:
    """The kind read of the frame registered as `frame_id`
    (`_read_member_cells`). A read that succeeded is kept for the turn in the
    frame's entry of `ORCHESTRATOR_FRAMES_KEY`; one that failed answers None
    and is made again on the next call."""
    frames = state.get(ORCHESTRATOR_FRAMES_KEY) or {}
    frame = frames[frame_id]
    if _MEMBER_CELLS in frame:
        return frame[_MEMBER_CELLS]
    cells = await _read_member_cells(frame, executor)
    if cells is None:
        return None
    frames = state.get(ORCHESTRATOR_FRAMES_KEY) or {}
    state[ORCHESTRATOR_FRAMES_KEY] = {
        **frames,
        frame_id: {**frames[frame_id], _MEMBER_CELLS: cells},
    }
    return cells


@dataclass(frozen=True)
class _Unranked:
    """Why a select's largest members were not picked: a cut of a typed role
    whose rows carry no kind or type cell (`kinds_needed`), a plan's rank
    positions for values (`rank_positions`), compared members with no anchor
    period (`no_period`), a chart whose period moves with the data
    (`period_moves`), matching rows that differ on the declared `coordinates`
    (`coordinates_differ`), or anything else (`not_ranked`)."""

    reason: Literal[
        "kinds_needed",
        "rank_positions",
        "no_period",
        "period_moves",
        "coordinates_differ",
        "not_ranked",
    ]
    coordinates: tuple[str, ...] = ()


def _ranked(
    data: Any,
    trace: Any,
    *,
    select: dict[str, Any],
    dimension: str,
    view: dict[str, Any],
    limit: int | None,
    cut: bool,
    kinds: Mapping[str, str] | None,
) -> TopMembers | _Unranked:
    """`top_members` over a stored frame. A cut of a place or flow party
    dimension compares only the members one ranking does, by the kind or type
    cells the frame's own rows carry, or by `kinds` when they carry none.
    Rows are ranked without their enrichment columns (`common.place_type`),
    which describe a member and split no series. Rows that give no ranking
    and differ on a dimension `view` declares are `coordinates_differ`."""
    marker = read_derived_marker(trace)
    if marker is not None and marker.ordinal:
        return _Unranked("rank_positions")
    rows = _rows_of(data)
    cells = None
    if cut and dimension in TYPED_ROLES:
        cells = _held_cells(dimension, rows) or kinds
        if cells is None:
            return _Unranked("kinds_needed")
    rows = [
        {
            name: cell
            for name, cell in row.items()
            if name in select or enrichment_role(name, roles=TYPED_ROLES) is None
        }
        for row in rows
    ]
    try:
        top = top_members(rows=rows, select=select, dimension=dimension, limit=limit, cells=cells)
    except RollingError:
        return _Unranked("no_period")
    if top is not None:
        return top
    differing = differing_coordinates(rows=rows, select=select, dimension=dimension, view=view)
    if differing:
        return _Unranked("coordinates_differ", coordinates=tuple(differing))
    return _Unranked("not_ranked")


def _over_cap_remedy(*, dimension: str, unranked: _Unranked, draws: int, plan_unspent: bool) -> str:
    """What to change in a select refused with the one `dimension` over the
    series cap, given why its largest members were not picked: bind the
    coordinates the rows differ on; or list at most `draws` of its values,
    with a plan offered beside the list when `plan_unspent` and the frame's
    values can be ranked."""
    plan = f", or plan a ranking or a total across {dimension}" if plan_unspent else ""
    if unranked.reason == "period_moves":
        return (
            "This chart's period moves with the latest data, so its largest members are not "
            f"chosen for it. List at most {draws} {dimension} values in the select{plan}."
        )
    if unranked.reason == "coordinates_differ":
        differing = " and ".join(unranked.coordinates)
        each = " each" if len(unranked.coordinates) > 1 else ""
        return (
            f"Its rows differ on {differing}, so the largest cannot be compared. "
            f"Bind {differing}{each} to one value."
        )
    if unranked.reason == "rank_positions":
        return (
            "This frame holds rank positions, which the chart draws as they are. "
            f"List the {dimension} values to show, at most {draws}."
        )
    return (
        "The largest could not be picked from this frame's rows. "
        f"List the {dimension} values the question names, at most {draws}{plan}."
    )


def _unnarrowed_refusal(
    issue: dict[str, Any],
    *,
    chart: Chart,
    select: dict[str, Any],
    view: dict[str, Any],
    unranked: _Unranked,
    plan_unspent: bool,
) -> dict[str, Any]:
    """The over-cap `issue` of a select `_top_select` left as written, with
    what to change (`_over_cap_remedy`) when one dimension is over the cap:
    `correct_and_retry`, with the dimension and its count in `fix`."""
    over = over_cap_dimension(select=select, view=view)
    if over is None:
        return issue
    dimension, count = over
    remedy = _over_cap_remedy(
        dimension=dimension,
        unranked=unranked,
        draws=chart_draws(type_=chart.type, view=view),
        plan_unspent=plan_unspent,
    )
    return harness_refusal(
        code=HarnessCode.SERIES_LIMIT_EXCEEDED,
        message=f"{issue['message']}. {remedy}",
        fix={"dimension": dimension, "values": count},
        retry_mode=RetryMode.CORRECT_AND_RETRY,
    )


@dataclass(frozen=True)
class _Narrowed:
    """A select with its one multiplying dimension listing members highest
    first, and for a select cut to what its chart draws, the cut."""

    select: dict[str, Any]
    cut: TopMembers | None


# The suffix a cut card's title ends with, as `_card_title` writes it, the
# periods a cut line chart or table draws included.
_CUT_SUFFIX = re.compile(r"(?:, [\d/-]+ to [\d/-]+)? \(\d+ largest of \d+ in [^()]+\)$")


def _card_title(subject: str, *, period: str | None, cut: TopMembers | None) -> str:
    """A card's title: the model's `subject`, then the period the card shows,
    then on a card cut to its largest members what the cut leaves out and the
    period it ranked at. The subject is shortened so the title fits 120
    characters."""
    suffix = f", {period}" if period is not None else ""
    if cut is not None:
        suffix += f" ({len(cut.members)} largest of {cut.holding} in {cut.period})"
    return f"{subject[: 120 - len(suffix)]}{suffix}"


def _covered_years(view: Mapping[str, Any]) -> list[str]:
    """Every year from the first to the last the frame's coverage names, a
    span's included; none for a view that states no coverage."""
    coverage = view.get("coverage") or []
    years = [int(year) for label in coverage for year in re.findall(r"\d{4}", str(label))]
    return [str(year) for year in range(min(years), max(years) + 1)] if years else []


def _title_refusal(subject: str, view: Mapping[str, Any]) -> dict[str, Any] | None:
    """The `title_names_period` refusal of a model's title naming a year the
    frame covers, `None` for any other title. `make_charts` writes the card's
    period, so a title naming one would state it twice or contradict it."""
    year = named_year(subject, _covered_years(view))
    if year is None:
        return None
    return harness_refusal(
        code=HarnessCode.TITLE_NAMES_PERIOD,
        message=(
            f"The title names {year}. Write what the chart shows with no period or year: "
            "make_charts adds the period the card shows."
        ),
        fix={"period": year},
    )


def _period_text(label: str) -> str:
    """A period label as a title states it, a span `2020/2024` as `2020 to 2024`."""
    return " to ".join(label.split("/"))


def _latest_text(periods: int, view: Mapping[str, Any]) -> str:
    """The period of a card counting `periods` back from the latest data, in
    the frame's grain (`TimeGrain`): `latest year`, `last 5 years`."""
    grain = view.get("period_grain")
    grain = grain if grain in get_args(TimeGrain) else "period"
    return f"latest {grain}" if periods == 1 else f"last {periods} {grain}s"


def _covered_range(view: Mapping[str, Any]) -> tuple[str, bool] | None:
    """The periods a frame's coverage names, as a title states them, and
    whether they are several: its one period, or its first and last. None for
    a rolling frame, whose years were worked out from the latest data and move
    on (`harness.rolling_cards`), and for a view that states no coverage."""
    coverage = [str(label) for label in view.get("coverage") or []]
    if view.get("rolling") or len(coverage) != 2:
        return None
    first, last = coverage
    if first == last or "/" in first:
        return _period_text(first), False
    return f"{first} to {last}", True


async def _card_period(
    chart: Chart,
    *,
    frame: Mapping[str, Any],
    select: dict[str, Any],
    cut: TopMembers | None,
) -> tuple[str | None, dict[str, Any] | None]:
    """The period a card's title states, None when nothing can or a cut bar
    chart's suffix states it, and the window its intent names, None for none.

    A card counting `latest_periods` back states them as relative periods. A
    card drawing every period its frame covers states them
    (`_covered_range`): a line chart, a table, and a bar chart whose select
    lists no dimension, which apis draws as one bar a period. A bar chart
    whose select lists one is drawn at one period, the latest every selected
    series holds inside its window, and a stat card states the period of its
    value; on a frame of several periods both are read from the frame's rows
    (`latest_shared_period`) and the bar chart is pinned there, since without
    a window it moves when the frame gains a newer period. A cut bar chart is
    pinned at the cut's period, which its suffix states (`_card_title`), and
    a cut line chart or table states its frame's periods beside it."""
    view = frame["view"]
    if chart.latest_periods is not None:
        return _latest_text(chart.latest_periods, view), _latest_window(chart.latest_periods)
    at_one = chart.type == "stat_card" or (
        chart.type == "bar_chart" and any(isinstance(value, list) for value in select.values())
    )
    if cut is not None and at_one:
        return None, {"start": cut.period, "end": cut.period}
    covered = _covered_range(view)
    if covered is None:
        return None, None
    text, several = covered
    if cut is not None:
        # The suffix states a frame of one period, the period the cut ranked at.
        return text if several else None, None
    if not (at_one and several):
        return text, None
    at = await _frame_work(frame_store_ref(frame), functools.partial(_drawn_period, select=select))
    pinned = at is not None and chart.type == "bar_chart"
    return at, {"start": at, "end": at} if pinned else None


def _drawn_period(data: Any, _trace: Any, *, select: dict[str, Any]) -> str | None:
    """`latest_shared_period` over a stored frame, read without its enrichment
    columns, which describe a member and split no series."""
    rows = [
        {
            name: cell
            for name, cell in row.items()
            if name in select or enrichment_role(name, roles=TYPED_ROLES) is None
        }
        for row in _rows_of(data)
    ]
    return latest_shared_period(rows=rows, select=select)


def _card_identity(intent: Mapping[str, Any]) -> str:
    """What a drawn card is: the intent the platform renders and checks it
    from (kind, frame, select, window, title). The figures a chart asks to be
    stated are no part of it, so asking a drawn card for other figures reads
    them from it. The title stays in, so a title the platform would refuse is
    checked afresh rather than answered from a card drawn under another."""
    return json.dumps(intent, sort_keys=True)


async def _top_select(
    chart: Chart,
    select: dict[str, Any],
    state: Any,
    executor: Executor,
    *,
    cut: bool,
) -> _Narrowed | dict[str, Any] | _Unranked:
    """`select` with its one multiplying dimension listing the frame's
    members by their value at one period (`top_members`), highest first.

    When `cut`, it lists only the members holding a value at that period, as
    many as `chart.type` draws (`chart_draws`). A cut of a place or flow
    party dimension ranks only the members one ranking compares (`_ranked`),
    by the frame's kind read (`_member_cells`) when its rows carry no kind or
    type cell. Otherwise it lists every member, those with no value at the
    period last. apis
    draws a table's and a line chart's series in the order the select lists
    them.

    Returns:
        The narrowed select; a harness refusal for a cut whose kind read
        failed (`member_kind_unread`) or whose members have no anchor period
        (`no_shared_period`); or an `_Unranked`, leaving the select as
        written: `period_moves` when the card's period moves with the data,
        so a list picked today would not follow it; `not_ranked` when no one
        dimension multiplies or the frame cannot be read; otherwise why its
        rows gave no ranking (`_ranked`).
    """
    frame = (state.get(ORCHESTRATOR_FRAMES_KEY) or {})[chart.frame]
    if frame["view"].get("rolling") or chart.latest_periods is not None:
        return _Unranked("period_moves")
    over = over_cap_dimension(select=select, view=frame["view"])
    if over is None:
        return _Unranked("not_ranked")
    dimension = over[0]
    rank = functools.partial(
        _ranked,
        select=select,
        dimension=dimension,
        view=frame["view"],
        limit=chart_draws(type_=chart.type, view=frame["view"]) if cut else None,
        cut=cut,
    )
    top = await _frame_work(frame_store_ref(frame), functools.partial(rank, kinds=None))
    analysis = str(frame["ref"]["spec_id"]).startswith(OP_SPEC_HASH_PREFIX)
    if top == _Unranked("kinds_needed") and not analysis:
        kinds = await _member_cells(state, chart.frame, executor)
        if kinds is None:
            return harness_refusal(
                code=HarnessCode.MEMBER_KIND_UNREAD,
                message=(
                    f"{chart.type} select is over the chart's series limit, and the kinds of "
                    f"its {dimension} members could not be read to choose the largest of them. "
                    "Drawing it again reads them once more; or list the members to show in select_json"
                ),
            )
        top = await _frame_work(
            frame_store_ref(frame), functools.partial(rank, kinds=kinds.get(dimension, {}))
        )
    if top == _Unranked("no_period") and cut:
        return harness_refusal(
            code=HarnessCode.NO_SHARED_PERIOD,
            message=(
                f"{chart.type} select is over the chart's series limit, and its {dimension} "
                "members hold several periods with none to choose the largest of them at: no "
                "period is held by half of them, or their periods are spans or of mixed grains"
            ),
        )
    if not isinstance(top, TopMembers) or not top.members:
        return top if isinstance(top, _Unranked) else _Unranked("not_ranked")
    members = top.members if cut else [*top.members, *top.unvalued]
    return _Narrowed(select={**select, dimension: members}, cut=top if cut else None)


def _cut_note(cut: TopMembers) -> str:
    """What a cut card leaves out, for the model, with the members of another
    kind and the members with no value at the cut's period counted apart."""
    return (
        f"This card draws the {len(cut.members)} largest of the {cut.holding} members with a "
        f"value in {cut.period}, highest first. Left out as group totals, unallocated amounts "
        f"or members of another level: {cut.other_kinds}. Left out with no value in {cut.period}: "
        f"{len(cut.unvalued)}. Say in the reply that it shows the largest of them in "
        f"{cut.period}."
    )


def _drawn_store_ref(payload: Mapping[str, Any], frame: Mapping[str, Any]) -> dict[str, Any]:
    """The pair to read the rows of the frame a written card draws: the
    registered `frame`'s store pair, unless the revision's input frame
    (`provenance.inputs[0]` of `payload`) is another. The rolling-card rules
    move a write to a frame replayed for the card's own series
    (`harness.rolling_cards`), a derived frame the store holds under its own
    pair, and the card then draws that frame's values."""
    inputs = (payload.get("provenance") or {}).get("inputs") or []
    drawn = inputs[0] if inputs and isinstance(inputs[0], Mapping) else None
    if drawn is None or dict(drawn) == dict(frame["ref"]):
        return frame_store_ref(frame)
    return dict(drawn)


def _coordinate_rows(data: Any, _trace: Any, *, series: Sequence[DrawnSeries]) -> list[DrawnSeries]:
    """`series` with the coordinates a stored frame's rows give those that
    state none (`with_coordinates`). The rows are read without their
    enrichment columns, which describe a member and split no series."""
    rows = [
        {
            name: cell
            for name, cell in row.items()
            if enrichment_role(name, roles=TYPED_ROLES) is None
        }
        for row in _rows_of(data)
    ]
    return with_coordinates(series, rows)


def _budgets(tool_context: ToolContext) -> dict[str, int]:
    return turn_budgets(tool_context.state)


def _used(tool_context: ToolContext, key: str) -> int:
    return int((tool_context.state.get(ORCHESTRATOR_CALLS_KEY) or {}).get(key, 0))


def _failed(
    code: HarnessCode, error: str, *, retry_mode: RetryMode | None = None
) -> dict[str, Any]:
    """A tool's failed answer: `error` is the model's reason, and `code` with
    its retry mode is what the user's line for the failure is worded from.
    `retry_mode` states another mode than the code's own."""
    mode = retry_mode or harness_retry_mode(code)
    return {"error": error, "code": code.value, "retry_mode": mode.value}


def _service_failed(body: Mapping[str, Any], error: str) -> dict[str, Any]:
    """A tool's failed answer for a service's refusal `body`: `error` beside
    the service's own code and retry mode, or `unspecified` when it states no
    code."""
    stated = {key: str(body[key]) for key in ("code", "retry_mode") if body.get(key)}
    if "code" not in stated:
        return _failed(HarnessCode.UNSPECIFIED, error)
    return {"error": error, **stated}


def _not_drawn(
    chart: Chart,
    error: str,
    *,
    refusal: Mapping[str, Any] | None = None,
    code: HarnessCode = HarnessCode.CHART_NOT_DRAWN,
) -> dict[str, Any]:
    """The `make_charts` entry of a chart that was not drawn: `error` is the
    model's reason. The entry states the `code` and `retry_mode` of `refusal`,
    the refusal that stopped it, with its `fix` when that is a mapping, as a
    refusal of the harness's own is. It states those of `code` when no
    refusal stopped the chart or the refusal names no code."""
    if refusal is None or not refusal.get("code"):
        return {"title": chart.title, **_failed(code, error)}
    entry = {"title": chart.title, "error": error, "code": str(refusal["code"])}
    if refusal.get("retry_mode"):
        entry["retry_mode"] = str(refusal["retry_mode"])
    if isinstance(refusal.get("fix"), Mapping):
        entry["fix"] = dict(refusal["fix"])
    return entry


def _figures_unread() -> dict[str, Any]:
    """What the `make_charts` entry of a drawn card states in place of what it
    holds and its figures when they could not be read, with no figure a reply
    may state, under the code `figures_unread`."""
    code = HarnessCode.FIGURES_UNREAD
    return {
        "reply_figures": [],
        "code": code.value,
        "retry_mode": harness_retry_mode(code).value,
        "figures_unread": (
            "This card is on the canvas, but what it holds could not be read back. State "
            "no value of it in the reply, and do not draw it again."
        ),
    }


def _spend(tool_context: ToolContext, tool: str) -> dict[str, Any] | None:
    """Count one call of `tool`; the failed answer when its budget is spent."""
    used = _used(tool_context, tool)
    if used >= _budgets(tool_context)[tool]:
        return _failed(
            HarnessCode.TOOL_BUDGET_SPENT,
            f"{tool} has been called {used} time(s) this turn, its limit. {_ANSWER_NOW}",
        )
    count_call(tool_context.state, tool)
    return None


def _charts_spent(tool_context: ToolContext) -> dict[str, Any] | None:
    """The failed answer when either of make_charts' budgets is spent."""
    budgets = _budgets(tool_context)
    drew = _used(tool_context, "make_charts")
    if drew >= budgets["make_charts"]:
        return _failed(
            HarnessCode.TOOL_BUDGET_SPENT,
            f"make_charts has drawn charts in {drew} call(s) this turn, its limit. {_ANSWER_NOW}",
        )
    failed = _used(tool_context, "make_charts_failed")
    if failed >= budgets["make_charts_failed"]:
        return _failed(
            HarnessCode.TOOL_BUDGET_SPENT,
            f"make_charts has drawn nothing in {failed} call(s) since a frame last arrived, "
            f"the limit for calls that draw nothing. {_ANSWER_NOW}",
        )
    return None


@functools.cache
def _adapter(annotation: Any) -> TypeAdapter[Any]:
    return TypeAdapter(annotation)


# A correction for a known mistake, appended to the problem it names. Keyed by
# the pydantic error type and the location with list indices as `*`.
type ArgumentHints = Mapping[tuple[str, str], str]

_SEARCH_ARGUMENT_HINTS: ArgumentHints = {
    ("extra_forbidden", "parts.*.role"): (
        'role goes on each entity in entities, e.g. {"text": "Kenya", "role": "to"}'
    ),
}


def _argument_problems(
    name: str, exc: ValidationError, hints: ArgumentHints
) -> list[tuple[dict[str, str], str]]:
    """Each of `exc`'s errors as its `{"loc", "type"}` entry and its line of
    prose (`<loc>: <msg>`, then any hint), without the input."""
    problems = []
    for error in exc.errors(include_url=False):
        steps = (name, *error["loc"])
        loc = ".".join(str(step) for step in steps)
        shape = ".".join("*" if isinstance(step, int) else str(step) for step in steps)
        line = f"{loc}: {error['msg']}"
        if hint := hints.get((error["type"], shape)):
            line += f" ({hint})"
        problems.append(({"loc": loc, "type": error["type"]}, line))
    return problems


def _invalid_arguments(
    tool_context: ToolContext, name: str, problems: Sequence[tuple[dict[str, str], str]]
) -> dict[str, Any]:
    """Return argument problems, bounded by the turn's invalid-call budget."""
    used = _used(tool_context, "invalid_arguments")
    if used >= _budgets(tool_context)["invalid_arguments"]:
        return _failed(
            HarnessCode.TOOL_BUDGET_SPENT,
            f"{name}: {used} call(s) this turn had invalid arguments, the limit. {_ANSWER_NOW}",
        )
    count_call(tool_context.state, "invalid_arguments")
    return {
        **_failed(
            HarnessCode.TOOL_ARGUMENTS_INVALID,
            f"{name}: {'; '.join(line for _, line in problems)}. "
            "Correct these arguments and call again.",
        ),
        "invalid": [entry for entry, _ in problems],
    }


def _validated_arguments(
    tool: FunctionType, *, hints: ArgumentHints | None = None
) -> Callable[..., Any]:
    """`tool` wrapped to validate every argument but `tool_context` before it runs.

    ADK hands a function tool the model's arguments as parsed JSON, whatever
    the annotations say. The wrapper validates them with pydantic and calls
    `tool` with the validated values. If any fail, it returns the problems as
    `{"error": <prose>, "invalid": [{"loc", "type"}, ...]}` under the code
    `tool_arguments_invalid`, with `hints` appended to the prose of the
    problems they match, and does not call `tool`. Rejected calls count
    against `BUDGETS["invalid_arguments"]`; past it the reply is the error
    alone, under `tool_budget_spent`. A tool with no model arguments is
    returned as is.
    """
    signature = inspect.signature(tool)
    annotations = get_type_hints(tool)
    adapters = {
        name: _adapter(annotations[name]) for name in signature.parameters if name != "tool_context"
    }
    if not adapters:
        return tool

    def bind(
        args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> inspect.BoundArguments | dict[str, Any]:
        """The call's arguments, validated, or the error the model is given."""
        bound = signature.bind(*args, **kwargs)
        problems: list[tuple[dict[str, str], str]] = []
        for name, value in bound.arguments.items():
            if name not in adapters:
                continue
            try:
                bound.arguments[name] = adapters[name].validate_python(value)
            except ValidationError as exc:
                logger.warning("orchestrator %s: invalid %s: %s", tool.__name__, name, exc)
                problems += _argument_problems(name, exc, hints or {})
        if not problems:
            return bound
        return _invalid_arguments(bound.arguments["tool_context"], tool.__name__, problems)

    if inspect.iscoroutinefunction(tool):

        @functools.wraps(tool)
        async def checked_async(*args: Any, **kwargs: Any) -> Any:
            bound = bind(args, kwargs)
            if isinstance(bound, dict):
                return bound
            return await tool(*bound.args, **bound.kwargs)

        return checked_async

    @functools.wraps(tool)
    def checked(*args: Any, **kwargs: Any) -> Any:
        bound = bind(args, kwargs)
        if isinstance(bound, dict):
            return bound
        return tool(*bound.args, **bound.kwargs)

    return checked


def _asks_refusal(asks: Sequence[str]) -> dict[str, Any] | None:
    """The refusal of a `plan` call whose `asks` list none or break the
    planner's limits (`planner.plan.MAX_PLAN_ASKS`, `MAX_ASK_CHARS`), or None.

    A plan held to no asks has no ask to license combining a step's years
    (`planner.plan.Plan.period_licences`), so a brief lists at least one."""
    problems = []
    if not asks:
        problems.append("asks lists no outputs: list each output the user asked for")
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
        "legal": {"asks": [1, MAX_PLAN_ASKS], "ask_chars": [1, MAX_ASK_CHARS]},
    }


def _breakdowns_missing(
    tool_context: ToolContext, asks: Sequence[AskedOutput]
) -> dict[str, Any] | None:
    """The refusal of a `plan` call whose asks leave out a party this turn's search held open, or None.

    A party a settled reading asked to hold open (`SearchReading.open_roles`)
    is one the user asked to see one by one or to total across, so some ask
    names it in `by` or in `over`. The side a reading's `share_of` runs over
    (`SearchReading.share_over`) is one the share's whole totals across, so
    some ask names it in `over`, which holds the plan's whole to every
    member of it (`planner.plan.WholeIsItsMemberError`); a reading that also
    held that side open, as each donor's share of every donor's total does,
    is answered by `by` too. The turn's readings are of different
    sources, so a `PARTY_ROLES` word keeps its kind of flow: only that word
    names it, and a turn that held "donor" open for aid and "creditor" open
    for debt needs an ask naming each. A role column, held open by a word
    naming a side alone (`planner.parties.open_party`), is named by any word
    on its side (`planner.parties.same_party`): a reading that held `origin`
    open is answered by `by: ["creditor"]`. The refusal names the party as
    its reading does (`_held_party_repair`), under `party` in its `invalid`
    entry. The refusal counts against the turn's
    invalid-call budget (`_invalid_arguments`), so a model that disputes it
    cannot call again without end.
    """
    named = [role for ask in asks for role in (*ask.by, *ask.over)]
    totalled = [role for ask in asks for role in ask.over]
    held: dict[str, str] = {}
    shared: dict[str, str] = {}
    for stored in tool_context.state.get(ORCHESTRATOR_SEARCHES_KEY) or []:
        reading = SearchReading.model_validate(stored)
        if reading.status in SETTLED_STATUSES and reading.spec_id:
            text = reading.text or reading.spec_id
            for role in reading.open_roles:
                held.setdefault(role, text)
            for column in reading.share_over:
                if not party_among(column, reading.open_roles):
                    shared.setdefault(column, text)
    problems = [
        (
            {"loc": "asks", "type": "breakdown_missing", "party": column_party_roles(role)},
            f"asks: this turn's search held {column_party_roles(role)} open for {text!r}, and "
            f"no ask names it; {_held_party_repair(role)}",
        )
        for role, text in held.items()
        if not (role in named if party_role(role) == role else party_among(role, named))
    ]
    problems += [
        (
            {"loc": "asks", "type": "share_whole_unstated", "party": column_party_roles(column)},
            f"asks: the whole of this turn's share for {text!r} spans every "
            f"{column_party_roles(column)}, and no ask totals across it; {_share_repair(column)}",
        )
        for column, text in shared.items()
        if not party_among(column, totalled)
    ]
    return _invalid_arguments(tool_context, "plan", problems) if problems else None


def _share_repair(column: str) -> str:
    """What a `plan` call adds to its asks for a share whose whole spans the side `column` names."""
    parties = ", ".join(f'"{party}"' for party in column_party_roles(column).split("/"))
    return (
        f"add the one of {parties} this flow has to the `over` of the ask the share answers, "
        "since its whole totals across them; a share of named members lists them as "
        "entities, not in `by`"
    )


def _held_party_repair(role: str) -> str:
    """What a `plan` call adds to its asks to name `role`, a party a search reading held open.

    A `PARTY_ROLES` word is the party to add. A role column
    (`planner.parties.open_party`) holds a different party in each kind of
    flow, so the repair lists the column's parties
    (`planner.parties.column_party_roles`) for the one this flow has.
    """
    parties = column_party_roles(role).split("/")
    if len(parties) == 1:
        return (
            f'add "{role}" to the `by` of the ask whose answer lists each {role}, or to the '
            f"`over` of the ask whose answer totals across every {role}"
        )
    return (
        f"add the one of {', '.join(f'"{party}"' for party in parties)} this flow has to the "
        "`by` of the ask whose answer lists each of them, or to the `over` of the ask whose "
        "answer totals across them"
    )


def _ask_breakdowns(asks: Sequence[AskedOutput]) -> list[AskBreakdown]:
    """The parties each ask lists in `by` and totals across in `over`, by number, for the asks that name any."""
    return [
        AskBreakdown(
            ask=number, by=tuple(dict.fromkeys(ask.by)), over=tuple(dict.fromkeys(ask.over))
        )
        for number, ask in enumerate(asks)
        if ask.by or ask.over
    ]


def _asks_across_years(asks: Sequence[AskedOutput]) -> list[int]:
    """The numbers of the asks that set `across_years`."""
    return [number for number, ask in enumerate(asks) if ask.across_years]


def _named_spec_ids(tool_context: ToolContext) -> set[str]:
    """Every spec_id this turn's search results named: each part's own and each
    candidate's (`ORCHESTRATOR_SPECS_KEY`), and each other count and other
    prices a part's result offers (`measure_choice`)."""
    specs = tool_context.state.get(ORCHESTRATOR_SPECS_KEY) or {}
    named = set(specs)
    for block in specs.values():
        choice = measure_choice(block or {})
        named.update(
            alternative["spec_id"]
            for key in ("alternatives", "other_prices")
            for alternative in choice.get(key) or ()
        )
    return named


# The questions one `ask_user` call may put to the user in one pause.
MAX_ASK_QUESTIONS = 3


def _questions_refusal(questions: list[AskQuestion], named: set[str]) -> dict[str, Any] | None:
    """The refusal of an `ask_user` call that asks no question or more than
    `MAX_ASK_QUESTIONS`, asks a question with no option, offers a spec_id no
    search result this turn named (`_named_spec_ids`), or offers one spec_id
    on two options of the call; None when it asks well."""
    options = [option for question in questions for option in question.options]
    spec_ids = [option.spec_id for option in options if option.spec_id is not None]
    problems = []
    if not 1 <= len(questions) <= MAX_ASK_QUESTIONS:
        problems.append(
            f"it asks {len(questions)} questions, and ask_user takes 1 to {MAX_ASK_QUESTIONS}"
        )
    problems += [
        f"question {question.question!r} offers no option"
        for question in questions
        if not question.options
    ]
    problems += [
        f"option {option.label!r} carries {option.spec_id}, which no search result named this turn"
        for option in options
        if option.spec_id is not None and option.spec_id not in named
    ]
    problems += [
        f"{spec_id} is on more than one option"
        for spec_id in sorted(set(spec_ids))
        if spec_ids.count(spec_id) > 1
    ]
    if not problems:
        return None
    return {
        "refusal": f"ask_user asked nothing: {'; '.join(problems)}.",
        "code": ToolErrorCode.REQUEST_INVALID.value,
        "legal": {"spec_ids": sorted(named)},
    }


def _pend(tool_context: ToolContext, action: PendingAsk | PendingPlan) -> dict[str, Any]:
    """Record `action` as the turn's pending action, unless one is pending.

    A recorded action skips the model call that would only end the turn: the
    workflow's `route` node acts on it once the turn ends. The refusal of a
    second hand-off goes back to the model.
    """
    pending = pending_action(tool_context.state)
    if pending is not None:
        return _failed(
            HarnessCode.HAND_OFF_PENDING,
            f"This turn is already waiting on {pending.kind}; one hand-off per turn. "
            "End your turn now.",
        )
    tool_context.state[ORCHESTRATOR_PENDING_KEY] = action.model_dump(mode="json")
    tool_context.actions.skip_summarization = True
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


async def _recorded_block(surface: ToolSurface | None, spec_id: str) -> Mapping[str, Any] | None:
    """The spec block Search answers now for the definition on record for
    `spec_id` (`planner.plan.recorded_identity`); `None` when none is on record.

    The definition is resolved through data-mcp (`resolve_definition`),
    which stores it for `get_data`, and the block is Search's own part's
    (`resolved_spec`), as `translator.discovery._recorded_source` reads a pin.

    Raises:
        DefinitionNotResolvedError: If data-mcp or Search could not be
            reached, Search refused the definition, or there is no surface
            to ask it through.
        PartNotResolvedError: If Search answers it with anything but that
            one dataset.
    """
    recorded = recorded_identity(spec_id)
    if recorded is None:
        return None
    if surface is None:
        raise DefinitionNotResolvedError(
            "this server has no data-mcp surface to resolve it through", retryable=False
        )
    part = await resolve_definition(
        surface, spec_id=spec_id, identity=recorded.identity, groups=recorded.groups
    )
    block = resolved_spec(part, spec_id=spec_id)
    mentions = getattr(recorded, "resolved_sources", ())
    return (
        {**block, "_resolved_sources": [mention.model_dump(mode="json") for mention in mentions]}
        if mentions
        else block
    )


def _unresolved_refusal(
    spec_id: str, exc: DefinitionNotResolvedError | PartNotResolvedError
) -> dict[str, Any]:
    """The refusal of a `fetch` whose definition on record `_recorded_block`
    could not read a block for."""
    if isinstance(exc, DefinitionNotResolvedError) and exc.retryable:
        return {
            "spec_id": spec_id,
            "refusal": f"Search could not be reached to read {spec_id} just now ({exc}). Fetch "
            "it again, or answer without this data.",
            "code": HarnessCode.TRANSIENT_UNAVAILABLE.value,
        }
    reason = str(exc) if isinstance(exc, DefinitionNotResolvedError) else exc.fault.value
    return {
        "spec_id": spec_id,
        "refusal": f"Search no longer reads {spec_id} as one dataset ({reason}), so it cannot "
        "be fetched. Fetch a spec_id this turn's search returned, or search again.",
        "code": ToolErrorCode.DEFINITION_NOT_HELD.value,
    }


def _empty_outcome(data: dict[str, Any]) -> dict[str, Any]:
    """`data`, a `get_data` outcome that served no observations, as the refusal `fetch`
    gives for it, with apis' own `empty_result` warning when it sent one."""
    warning = next(
        (w for w in data.get("warnings") or [] if w.get("code") == HarnessCode.EMPTY_RESULT), {}
    )
    said = f" ({warning['message']})" if warning.get("message") else ""
    message = (
        f"This data holds no observations for the places and period asked{said}. Fetching it "
        "again unchanged returns nothing: change the places or period, or answer without it."
    )
    return {
        "outcome": "refusal",
        "refusal": {"code": HarnessCode.EMPTY_RESULT.value, "message": message},
    }


def _attach_fetch(tool_context: ToolContext, result: dict[str, Any]) -> None:
    """Add a `fetch` result to every stored search reading of its spec_id.

    A spec_id that no reading holds as its own, but one offered as a
    candidate, as another count (`SearchReading.alternatives`) or as the
    same count in other prices (`SearchReading.other_prices`), first gains
    a reading of its own, read from its block (`chosen_part`) with the
    offering reading's text, skills and departures and asked parties and
    its own warnings, else the offering reading's. A spec_id no reading
    offers as a candidate is a sibling (`_offered_sibling`), and its reading
    names the reading that offered it (`SearchReading.sibling_of`) and the
    count it reads.
    """
    spec_id = result["spec_id"]
    state = tool_context.state
    searches = [
        SearchReading.model_validate(stored)
        for stored in state.get(ORCHESTRATOR_SEARCHES_KEY) or []
    ]
    block = (state.get(ORCHESTRATOR_SPECS_KEY) or {}).get(spec_id)
    candidate_of = next(
        (r for r in searches if any(c.spec_id == spec_id for c in r.candidates)), None
    )
    sibling = None if candidate_of else _offered_sibling(searches, spec_id)
    offered = candidate_of or (sibling[0] if sibling else None)
    if block and offered is not None and all(r.spec_id != spec_id for r in searches):
        warnings = (state.get(ORCHESTRATOR_SPEC_WARNINGS_KEY) or {}).get(spec_id)
        if warnings is None:
            warnings = [w.model_dump() for w in offered.warnings]
        chosen = SearchReading.from_part(
            {**chosen_part(block, variable=offered.text), "warnings": warnings},
            skills=offered.skills,
            observed_at=datetime.now(UTC),
        )
        update: dict[str, Any] = {
            "separate": offered.separate,
            "departures": offered.departures,
            "asked_roles": offered.asked_roles,
            "open_roles": offered.open_roles,
            "share_over": offered.share_over,
        }
        if sibling is not None:
            link = sibling[1]
            counted_as = link.label if link.axis == "what" else offered.counted_as
            update |= {"sibling_of": link, "counted_as": counted_as}
        searches.append(chosen.model_copy(update=update))
    state[ORCHESTRATOR_SEARCHES_KEY] = [
        (reading.with_fetch(result) if reading.spec_id == spec_id else reading).model_dump(
            mode="json"
        )
        for reading in searches
    ]


def _open_roles(part: SearchPart) -> list[str]:
    """The flow parties `part` asks to hold open, each as `planner.parties.open_party` reads its word.

    A word for one kind of flow gives its `PARTY_ROLES` word, and a word
    naming a side alone its role column: "lenders" gives "creditor" and
    "from" gives "origin".
    """
    return list(dict.fromkeys(role for word in part.open_roles or () if (role := open_party(word))))


def _share_sides(part: SearchPart) -> list[str]:
    """The role column of each side of the flow a `share_of` operation of `part` runs over.

    A share over `origin` divides by a whole spanning every giving party:
    "the United States' share of ODA to Kenya" is one.
    """
    return list(
        dict.fromkeys(
            operation.over
            for operation in part.operations or ()
            if operation.verb == "share_of" and operation.over in ("origin", "destination")
        )
    )


def _offered_sibling(
    searches: Sequence[SearchReading], spec_id: str
) -> tuple[SearchReading, SiblingReading] | None:
    """The first reading of `searches` offering `spec_id` as another count or
    in other prices, with the link a reading of `spec_id` keeps to it; `None`
    when none does."""
    for reading in searches:
        if reading.spec_id is None:
            continue
        for axis, offers in (("what", reading.alternatives), ("unit", reading.other_prices)):
            label = next((offer.label for offer in offers if offer.spec_id == spec_id), None)
            if label is not None:
                return reading, SiblingReading(spec_id=reading.spec_id, axis=axis, label=label)
    return None


def _asked_roles(part: SearchPart) -> list[str]:
    """The flow parties `part` asks for, as `planner.parties.PARTY_ROLES` words.

    Each entity's `role` and each of `open_roles` that names one party
    (`planner.parties.party_role`): "to" and "recipients" give "recipient".
    """
    words = [entity.role for entity in part.entities or () if entity.role]
    words += part.open_roles or []
    return list(dict.fromkeys(role for word in words if (role := party_role(word))))


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


class Resolver(Protocol):
    """Stands in for data-mcp `search`: the question and its search parts ->
    the search digest (`harness.tool_contracts.resolver_digest`).

    `context` is the conversation a parsed search is made in
    (`_search_context`); a raw-text search is sent none.
    """

    async def __call__(
        self,
        *,
        question: str,
        parts: list[dict[str, Any]],
        raw_text: bool = False,
        context: SearchContext | None = None,
    ) -> dict[str, Any]: ...


def _search_context(state: SessionState) -> SearchContext:
    """The conversation a parsed search is made in: the user's message for the
    turn (`current_user_message`) and the entries carried from the scope
    choices of every earlier search, the earlier turns' readings
    (`ORCHESTRATOR_PRIOR_LEDGER_KEY`) and then this turn's
    (`ORCHESTRATOR_SEARCHES_KEY`), oldest first (`carried_entries`).
    """
    readings = [
        *(state.get(ORCHESTRATOR_PRIOR_LEDGER_KEY) or {}).get("readings", []),
        *(state.get(ORCHESTRATOR_SEARCHES_KEY) or []),
    ]
    scopes = [
        PartScope.model_validate(reading["scope"]) for reading in readings if reading.get("scope")
    ]
    return SearchContext(
        user_message=current_user_message.get() or None,
        carried_entries=[
            entry.model_dump(mode="json", exclude_none=True) for entry in carried_entries(scopes)
        ],
    )


def build_orchestrator_tools(
    *,
    resolver: Resolver,
    executor: Executor,
    surface: ToolSurface | None = None,
    dfo_faq_surface: ToolSurface | None = None,
    definition_surface: ToolSurface | None = None,
    recall_part: Callable[..., Awaitable[dict[str, Any]]] | None = None,
    canvas_series: Callable[[], list[str]] | None = None,
    canvas_cards: Callable[[], list[dict[str, Any]]] | None = None,
    remove_cards: Callable[..., Awaitable[tuple[list[str], list[str]]]] | None = None,
) -> list[Any]:
    """The orchestrator's tools, bound to this server's resolver, executor and tool surface.

    `dfo_faq_surface` is the surface `explain_dfo` reads dfo-faq and the
    counts of what DFO holds through: the bare one, off the Activity feed,
    since `explain_dfo`'s own call is the step the user sees. Left unset,
    `explain_dfo` says it cannot describe DFO.
    `definition_surface` is the surface `fetch` resolves a definition on
    record through (`_recorded_block`), the bare one, off the Activity feed,
    as the graph's `seed` node stores definitions. Left unset, such a fetch
    is refused.
    `recall_part` is `harness.studio_state.make_recall_part`'s seam. Left
    unset, `recall` says there is nothing to recall from. `canvas_series` is
    `harness.studio_state.make_canvas_series`'s seam; left unset, no series
    counts as on the canvas. `canvas_cards` is
    `harness.studio_state.make_canvas_cards`' seam; left unset, only the
    cards drawn this turn are known to be on the canvas. `remove_cards` is
    `harness.studio_state.make_remove_cards`'s seam; left unset, the tool
    says it is not available.
    """
    # One lock per invocation, held by a make_charts call from its budget
    # check to its count, so concurrent calls draw one after another and each
    # is admitted on exact counts, and by remove_cards while it takes cards
    # off and drops them from the turn's drawn cards. An entry lasts while a
    # call holds its lock.
    chart_locks: weakref.WeakValueDictionary[str, asyncio.Lock] = weakref.WeakValueDictionary()

    def _canvas_lock(tool_context: ToolContext) -> asyncio.Lock:
        lock = chart_locks.get(tool_context.invocation_id)
        if lock is None:
            lock = chart_locks[tool_context.invocation_id] = asyncio.Lock()
        return lock

    async def search(
        query: str,
        tool_context: ToolContext,
        parts: list[SearchPart] | None = None,
        as_text: bool = False,
    ) -> dict[str, Any]:
        """Search DFO's data for every measured thing a question asks for.

        `query` is the question in the user's words. A follow-up that leans
        on earlier turns ("the same for Ghana", "show those over time") is
        sent as one self-contained question naming the measure, places and
        period. `parts` is one entry per measured variable in it, each with
        the entities, time, unit or sources the question gives for that
        variable, as spoken. If Search reports an ambiguous source, ask the
        user which candidate they mean, then set that source entry's `ref`
        to the chosen candidate's reference, keeping its text and exclusion.
        A publisher's indicator code that the user or a skill gives
        ("DT.TDS.DEGG.CD") goes in the part's `source_code`, and `variable`
        holds the user's words for the measure, or the code itself when the
        user gives none. When search reports the code ambiguous
        (`source_code_ambiguous`), retry with a `sources` entry naming one of
        the sources it lists. An entity that names a set of places keeps
        the set's name as its text, with `members: true` when the question
        asks for each member on its own: "African countries" is
        `{"text": "Africa", "members": true}`, and "Africa" alone is the
        region as one place. All parts go in one call. A flow measure such
        as aid, donors or recipients needs `role` on each entity in
        `entities`, as spoken. Use `to` for a recipient and `from` for a
        donor. Aid to Kenya is `{"text": "Kenya", "role": "to"}`. A "by
        creditor" or "which donors" question leaves that flow side open in
        `open_roles`. A total across a side leaves it out of `open_roles` and
        names the population the user asked for: "aid to Ethiopia from all
        donors" names "all donors" as a `from` entity, which covers official
        and private donors, and a donor side left unnamed on DAC2a covers
        official donors only. For a named population, search binds the
        source's published total. Where the source publishes none, search
        departs `whole_total_unpublished` and returns the members, whose sum
        is the total. Where the source holds no series of that total, such as
        "official donors" on the CRS, search finds no data. A published total
        stands in only where no operation runs over its side: any operation
        over `origin` or `destination`, `aggregate` included, opens that side and keeps its members. Search
        lists "official donors" member by member only where the dataset holds
        no private donors, and otherwise refuses `entity_role_not_admitted`:
        open the donor side and keep the parties whose
        `financingProvider_party_kind` is `party` and
        `financingProvider_party_class` is `official`. Keep `open_roles` when the
        user explicitly asks to break down that side: "Who are Kenya's top 5 creditors?" is `{"variable": "external debt",
        "entities": [{"text": "Kenya", "role": "debtor"}], "open_roles":
        ["creditor"], "operations": [{"verb": "rank",
        "phrase": "top 5 creditors", "over": "origin"}]}`. A rank over place,
        category or year has no flow side. A worldwide country
        ranking is a rank `over` "place" with no entity. A donor or creditor
        ranking uses `origin`; a recipient or debtor ranking uses
        `destination`. A flow role may contain countries and organisations, so
        opening it does not restrict the entity type. A scoped country ranking
        keeps its scope: "top 5 African countries" keeps `{"text": "Africa",
        "members": true}`, and an explicitly named country stays an entity,
        independently of the ranking axis. Do not add "World" as a stand-in for
        an open population; retain it when the user asks for the World total.
        Two entities written alike stay two rows when each carries its `name`
        or `dcid`, never both: "aid from Congo to Congo" sends `{"text":
        "Congo", "role": "from", "name": "Democratic Republic of the Congo"}`
        and `{"text": "Congo", "role": "to", "name": "Republic of the Congo"}`,
        and an `entity_ambiguous` row is resent with the chosen candidate's
        `dcid`, keeping its `text` and `role`.

        A comparison of two categories of one measure ("health versus
        education aid") is one part per category, each with `separate:
        true`, and so is any other comparison sent as several parts;
        search may otherwise merge parts that differ in one category or
        place into one answer. A total over several categories ("health
        and education aid") stays one part.

        Set `as_text` to have Search read `query` itself, as written, with
        the one part it asks about in `parts`, so the answer can be held
        against what that part asked. Send one part's question at a time
        this way: Search gives every part of the text the places and years
        it reads, and the sources it names or excludes, so `query` keeps
        the user's words for the part's sources.

        Returns the result's `kind`: a `resolution` reads each part
        separately; a `catalogue` means the question asked what data DFO
        holds, answered with a scope and no parts; a `declined` means
        Search would not answer, with a reason, such as a question outside
        DFO's scope or one that needs analysis. Then
        `parts`, one per part: its `status` (definite: a spec_id to fetch;
        candidates: readings to choose between, each with a spec_id and
        label; no_data or unsupported: with warnings and, for no_data, a
        `reason` code saying why; a part refused because search could not
        apply a phrase of it adds those phrases as `unbound_qualifiers` and,
        as `flagged_match`, the sentence of the match it found without
        them, which answers a broader question), what
        it resolved to in words, its
        warnings, its `departures` from what was asked when search states
        any, its `group_readings` when it reads a named group as the
        group's published total, and the window it covers. A definite part
        whose count search chose among others states it as `counted_as`,
        with other counts under `alternatives` and the same count in other
        prices under `other_prices`, each with its spec_id and label, and
        lists the `source_codes` of the indicators it found, each with the
        publisher that issues it when search names one. And
        `skills`: background a result calls for, each by id and
        description, loaded with get_skill, and `more_skills`, how many
        more the result names.
        """
        if as_text and parts and len(parts) > 1:
            return _invalid_arguments(
                tool_context,
                "search",
                [
                    (
                        {"loc": "parts", "type": "too_long"},
                        "parts: with as_text, send at most one part; retry each part separately",
                    )
                ],
            )
        spent = _spend(tool_context, "search")
        if spent:
            return spent
        parts = parts or []
        try:
            if as_text or not parts:
                digest = await resolver(question=query, parts=[], raw_text=True)
            else:
                digest = await resolver(
                    question=query,
                    parts=[part.model_dump(exclude_none=True) for part in parts],
                    context=_search_context(tool_context.state),
                )
        except ResolverError as exc:
            # The service's own code and retry mode word the user's line for
            # the failed search.
            return _service_failed(exc.refusal, exc.refusal.get("message") or str(exc))
        # Search answers parts in the order asked, so a digest names the part
        # at its own position when the counts agree. When they do not, no
        # digest is tied to a part, and none records parties asked.
        digests = digest.get("part_digests") or []
        tied = len(parts) == len(digests)
        readings = store_search(
            tool_context.state,
            digest,
            asked=[_asked_roles(part) for part in parts] if tied else [],
            held_open=[_open_roles(part) for part in parts] if tied else [],
            share_sides=[_share_sides(part) for part in parts] if tied else [],
        )
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

    async def fetch(  # noqa: PLR0911 -- one distinct refusal per way a fetch fails
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
        values, and a `period_note` when a newer year holds data, the year
        served is not the elected year, or a fuller earlier year is named; or
        a refusal with its `code` and reason, and the platform's
        `retry_mode` and `fix` (its details and the values to change to) when
        it gave them. The code `capability_unavailable` means this deployment
        cannot do what the call asked; `empty_result` means the data holds no
        observations for the places and period asked. A fetch of the same data
        as an earlier fetch this turn returns that fetch's result, with
        `same_data_as` naming its spec_id.
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
            return spent
        args: dict[str, Any] = {"spec_id": spec_id}
        if scope is not None:
            args["scope"] = scope.model_dump(exclude_none=True)
        # A stated reading wins, and a breakdown reads the family separately.
        # A spec a plan's step retrieved keeps the plan's reading; one a
        # search returned is read by the compiler's own family rule.
        reading = stat_vars or (
            "separate"
            if breakdown is not None
            else (tool_context.state.get(ORCHESTRATOR_READINGS_KEY) or {}).get(spec_id)
        )
        block = _spec_block(tool_context, spec_id)
        if block is None:
            try:
                block = await _recorded_block(definition_surface, spec_id)
            except (DefinitionNotResolvedError, PartNotResolvedError) as exc:
                logger.warning("fetch: no block for %s on record: %s", spec_id, exc)
                return _unresolved_refusal(spec_id, exc)
            if block is not None:
                tool_context.state[ORCHESTRATOR_SPECS_KEY] = {
                    **(tool_context.state.get(ORCHESTRATOR_SPECS_KEY) or {}),
                    spec_id: block,
                }
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
            args["breakdown"] = breakdown.model_dump()
        key = _fetch_key(args, block)
        earlier = (tool_context.state.get(_FETCHES_KEY) or {}).get(key)
        if earlier is not None:
            frames = tool_context.state.get(ORCHESTRATOR_FRAMES_KEY) or {}
            found = (
                {**earlier, "spec_id": spec_id, "same_data_as": earlier["spec_id"]}
                if earlier.get("refusal")
                else {
                    "spec_id": spec_id,
                    "summary": earlier["summary"],
                    **_period_note_entry(earlier),
                    "frames": [{"frame": f, "view": frames[f]["view"]} for f in earlier["frames"]],
                    "same_data_as": earlier["spec_id"],
                }
            )
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
                    "code": HarnessCode.DEADLINE_EXCEEDED.value,
                }
            else:
                refused = {
                    "spec_id": spec_id,
                    "refusal": "get_data could not reach data-mcp just now. Fetch it again, or "
                    "answer without this data.",
                    "code": HarnessCode.TRANSIENT_UNAVAILABLE.value,
                }
            _attach_fetch(tool_context, refused)
            return refused
        if data.get("outcome") == "ok" and data.get("rows") == 0:
            # A mint that served no observations is refused as `empty_result`.
            data = _empty_outcome(data)
        if data.get("outcome") != "ok":
            refusal = data.get("refusal") or {}
            refused = {
                "spec_id": spec_id,
                "refusal": refusal.get("message")
                or "get_data refused this fetch and gave no reason.",
                "code": refusal.get("code"),
            }
            refused.update({k: refusal[k] for k in ("retry_mode", "fix") if refusal.get(k)})
            if refused["code"] == ToolErrorCode.CAPABILITY_UNAVAILABLE.value:
                tool_context.state[ORCHESTRATOR_UNAVAILABLE_KEY] = {
                    **(tool_context.state.get(ORCHESTRATOR_UNAVAILABLE_KEY) or {}),
                    spec_id: refused["refusal"],
                }
            elif refused["code"] == HarnessCode.EMPTY_RESULT:
                # Kept for the turn, so the same fetch again answers from here.
                tool_context.state[_FETCHES_KEY] = {
                    **(tool_context.state.get(_FETCHES_KEY) or {}),
                    key: refused,
                }
            _attach_fetch(tool_context, refused)
            return refused
        described: list[DescribedFrame] = []
        unavailable: list[str] = []
        for ref, store_ref in _stated_frames(data):
            view = await executor(tool="describe_frame", args={"frame_ref": ref})
            if view.get("outcome") == "ok":
                described.append((ref, view.get("view") or {}, store_ref))
            else:
                unavailable.append((view.get("refusal") or {}).get("message") or "not described")
        found = {
            "spec_id": spec_id,
            "summary": data.get("summary"),
            **_period_note_entry(data),
            "frames": register_frames(tool_context.state, described, fetch_args=args),
        }
        if unavailable:
            found["frames_unavailable"] = unavailable
        else:
            fetches = dict(tool_context.state.get(_FETCHES_KEY) or {})
            fetches[key] = {
                "spec_id": spec_id,
                "summary": found["summary"],
                **_period_note_entry(found),
                "frames": [entry["frame"] for entry in found["frames"]],
            }
            tool_context.state[_FETCHES_KEY] = fetches
        _attach_fetch(tool_context, found)
        return found

    def _canvas_series(tool_context: ToolContext) -> list[str]:
        """The ids of the series the cards on the canvas drew as the turn
        began: every series of every card, as `canvas_series` gives them.
        `make_charts` and `remove_cards`, the tools that change the canvas,
        each read it before they act, and the first read of a turn is kept
        for the rest of it."""
        held = tool_context.state.get(_CANVAS_SERIES_KEY) or {}
        if held.get("invocation") == tool_context.invocation_id:
            return held["ids"]
        ids = list(canvas_series()) if canvas_series is not None else []
        tool_context.state[_CANVAS_SERIES_KEY] = {
            "invocation": tool_context.invocation_id,
            "ids": ids,
        }
        return ids

    async def _card_result(
        chart: Chart,
        component: Mapping[str, Any],
        *,
        frame: Mapping[str, Any],
        on_canvas: list[str],
        cut: TopMembers | None,
    ) -> dict[str, Any]:
        """What a drawn card holds and the figures `chart` asks a reply to
        state of it (`harness.reply_figures.card_result`), from the drawn
        payload of `component`, a write reply's component or `_stored_card`.
        The rows of the frame the card draws (`_drawn_store_ref`) are read
        only for a name in `chart.state_values_for` that matches no series
        while some state no coordinates: the platform's summary lists the
        coordinates of twelve."""
        payload = component.get("payload") or {}
        series = drawn_series(payload.get("component") or {}, component.get("fact_summary") or {})
        if unplaced_names(series, chart.state_values_for):
            placed = await _frame_work(
                _drawn_store_ref(payload, frame),
                functools.partial(_coordinate_rows, series=series),
            )
            series = placed or series
        return card_result(
            series,
            state_values_for=chart.state_values_for,
            state_span=chart.state_span,
            state_period=chart.state_period,
            on_canvas=on_canvas,
            cut=cut,
        )

    async def _stored_card(component_id: str) -> dict[str, Any] | None:
        """The card `component_id` as the platform holds its head revision
        (`describe_component`), shaped as a write reply's component: its
        revision, resolved payload and fact summary. None when the read
        fails or carries no resolved payload."""
        try:
            described = await executor(
                tool="describe_component", args={"component_id": component_id}
            )
        except Exception:
            logger.exception("make_charts: reading back card %r failed", component_id)
            return None
        payload = described.get("resolved_payload")
        body = described.get("component_body")
        if (
            described.get("outcome") != "ok"
            or not isinstance(payload, dict)
            or not isinstance(payload.get("component"), dict)
            or not isinstance(body, dict)
        ):
            return None
        return {
            "revision": body.get("revision"),
            "payload": payload,
            "fact_summary": body.get("fact_summary") or {},
        }

    async def _card_entry(
        chart: Chart,
        component: Mapping[str, Any],
        *,
        title: str,
        frame: Mapping[str, Any],
        on_canvas: list[str],
        cut: TopMembers | None,
    ) -> dict[str, Any]:
        """What `make_charts` returns of a card on the canvas: its title,
        `_card_result` and, on a cut card, what the cut leaves out."""
        try:
            held = await _card_result(chart, component, frame=frame, on_canvas=on_canvas, cut=cut)
        except Exception:
            # The card is on the canvas, so only what a reply may state of
            # it is lost, and the charts after it are still drawn.
            logger.exception("make_charts: reading the figures of %r failed", title)
            held = _figures_unread()
        entry = {"title": title, **held}
        if cut is not None:
            entry["note"] = _cut_note(cut)
        return entry

    async def _draw(charts: list[Chart], tool_context: ToolContext) -> list[dict[str, Any]]:
        """Draw `charts`, up to `MAX_CHARTS_PER_CALL`, and count the call as one
        that drew or read a chart or one that did neither. A chart whose card
        is on the canvas, drawn this turn or an earlier one, is not drawn
        again: its figures are read from the card's head revision
        (`_stored_card`), and a card that cannot be read is refused."""
        frames: dict[str, Any] = tool_context.state.get(ORCHESTRATOR_FRAMES_KEY) or {}
        on_canvas = _canvas_series(tool_context)
        held_cards = tool_context.state.get(_DRAWN_CARDS_KEY) or {}
        drawn_cards: dict[str, dict[str, Any]] = (
            dict(held_cards["drawn"])
            if held_cards.get("invocation") == tool_context.invocation_id
            else {}
        )
        canvas = {
            _card_identity(card["intent"]): {
                "component_id": card["component_id"],
                "revision": card["revision"],
            }
            for card in (canvas_cards() if canvas_cards is not None else [])
        }
        drawn: list[dict[str, Any]] = []
        landed = read = False
        try:
            for chart in charts[:MAX_CHARTS_PER_CALL]:
                frame = frames.get(chart.frame)
                if frame is None:
                    drawn.append(
                        _not_drawn(
                            chart,
                            f"unknown frame {chart.frame!r}: name a frame a fetch of this "
                            "turn returned",
                            code=HarnessCode.FRAME_UNKNOWN,
                        )
                    )
                    continue
                # A title the model copied from a card already cut loses that
                # suffix, so the card's own suffix is the only one.
                subject = _CUT_SUFFIX.sub("", chart.title)
                if (refusal := _title_refusal(subject, frame["view"])) is not None:
                    drawn.append(_not_drawn(chart, refusal["message"], refusal=refusal))
                    continue
                try:
                    select = json.loads(chart.select_json)
                except json.JSONDecodeError:
                    drawn.append(
                        _not_drawn(
                            chart,
                            "select_json is not valid JSON: write it as one JSON object",
                            code=HarnessCode.SELECT_INVALID,
                        )
                    )
                    continue
                plan_unspent = _used(tool_context, "plan") < _budgets(tool_context)["plan"]
                issue = select_binding_ok(
                    type_=chart.type,
                    select=select,
                    view=frame["view"],
                    latest_periods=chart.latest_periods is not None,
                    plan_unspent=plan_unspent,
                )
                if select == {} and issue is None and frame["view"].get("dimensions"):
                    # The platform refuses an empty select, so one that names
                    # every series is written for it, the first the frame's
                    # rows hold a value on throughout. A frame that cannot be
                    # read confirms none, and its refusal says so. A view that
                    # states no dimensions gives nothing to write one from.
                    selects = every_series_selects(type_=chart.type, view=frame["view"])
                    covering: list[dict[str, Any]] | None = []
                    if selects:
                        covering = await _frame_work(
                            frame_store_ref(frame),
                            functools.partial(_covering_selects, selects=selects),
                        )
                    if covering:
                        select = covering[0]
                    else:
                        issue = empty_select_refusal(
                            type_=chart.type, view=frame["view"], unread=covering is None
                        )
                cut = None
                over_cap = issue is not None and issue["code"] == HarnessCode.SERIES_LIMIT_EXCEEDED
                # apis sorts a bar chart's bars itself, at the period it draws.
                ordered = chart.order_by_value and chart.type != "bar_chart"
                if over_cap or (issue is None and ordered):
                    narrowed = await _top_select(
                        chart, select, tool_context.state, executor, cut=over_cap
                    )
                    if isinstance(narrowed, _Narrowed):
                        rewritten = select_binding_ok(
                            type_=chart.type, select=narrowed.select, view=frame["view"]
                        )
                        # Ordering alone never turns a drawable select into a refused one.
                        if over_cap or rewritten is None:
                            select, cut, issue = narrowed.select, narrowed.cut, rewritten
                    elif not isinstance(narrowed, _Unranked):
                        issue = narrowed
                    elif over_cap and issue is not None:
                        issue = _unnarrowed_refusal(
                            issue,
                            chart=chart,
                            select=select,
                            view=frame["view"],
                            unranked=narrowed,
                            plan_unspent=plan_unspent,
                        )
                if issue is not None:
                    drawn.append(_not_drawn(chart, issue["message"], refusal=issue))
                    continue
                period, window = await _card_period(chart, frame=frame, select=select, cut=cut)
                title = _card_title(subject, period=period, cut=cut)
                intent: dict[str, Any] = {
                    "type": chart.type,
                    "frame_ref": frame["ref"],
                    "select": select,
                    "title": title,
                }
                if window is not None:
                    intent["window"] = window
                identity = _card_identity(intent)
                held = drawn_cards.get(identity) or canvas.get(identity)
                stored = None if held is None else await _stored_card(held["component_id"])
                if held is not None and stored is None:
                    # The canvas holds this card, so another write would stack a copy.
                    drawn.append(
                        _not_drawn(
                            chart,
                            "This card is already on the canvas, and its figures could "
                            "not be read back. No copy was drawn; state no value of it "
                            "in the reply.",
                            code=HarnessCode.FIGURES_UNREAD,
                        )
                    )
                    continue
                # A card revised since it was drawn no longer draws this intent.
                if (
                    held is not None
                    and stored is not None
                    and stored["revision"] == held["revision"]
                ):
                    read = True
                    entry = await _card_entry(
                        chart, stored, title=title, frame=frame, on_canvas=on_canvas, cut=cut
                    )
                    drawn.append(
                        entry
                        | {
                            "already_on_canvas": (
                                "This card is already on the canvas, and no copy was "
                                "drawn. Its reply_figures are those this call asks for; "
                                "draw a chart again only with a different kind, frame, "
                                "select or period."
                            )
                        }
                    )
                    continue
                try:
                    written = await executor(tool="write_component", args={"intent": intent})
                    component = written.get("component") if written.get("outcome") == "ok" else None
                    if component is None or not usable_component(component):
                        refusal = written.get("refusal") or {}
                        failed = _not_drawn(
                            chart,
                            refusal.get("message") or "the platform refused it",
                            refusal=refusal,
                        )
                        # presentation-mcp's `fix`: the values the platform would
                        # accept, as `label (ref)`, for the retry's select.
                        if refusal.get("advice"):
                            failed["advice"] = refusal["advice"]
                        drawn.append(failed)
                        continue
                    land = getattr(executor, "land_component", None)
                    if land is not None:
                        land(component)
                except Exception as error:
                    # Charts already drawn stay on the canvas and in the result.
                    # A write that timed out may still have stored this chart,
                    # so the model is told not to draw it again.
                    logger.exception("make_charts: drawing %r failed", chart.title)
                    drawn.append(
                        _not_drawn(
                            chart,
                            f"drawing it failed inside DFO ({type(error).__name__}), and "
                            "the charts after it in this call were not drawn. Do not "
                            "draw this chart again: it may be on the canvas already.",
                            code=HarnessCode.WRITE_UNCONFIRMED,
                        )
                    )
                    break
                landed = True
                drawn.append(
                    await _card_entry(
                        chart, component, title=title, frame=frame, on_canvas=on_canvas, cut=cut
                    )
                )
                drawn_cards[identity] = {
                    "component_id": component["component_id"],
                    "revision": component["revision"],
                }
                tool_context.state[_DRAWN_CARDS_KEY] = {
                    "invocation": tool_context.invocation_id,
                    "drawn": drawn_cards,
                }
        finally:
            # After the awaits, so the counts are read afresh. The `finally`
            # covers cancellation and anything raised outside the per-chart
            # `try`. A failed write is caught above. A call that read a card
            # on the canvas gave figures, so it counts as one that drew.
            count_call(
                tool_context.state, "make_charts" if landed or read else "make_charts_failed"
            )
        return drawn

    async def make_charts(charts: list[Chart], tool_context: ToolContext) -> dict[str, Any]:
        """Draw charts on the user's canvas.

        Returns for each chart drawn what it holds and its reply_figures, or
        why it could not be drawn and, when the platform names them, the
        values it would accept. A chart drawn whose values could not be read
        back says so under the code `figures_unread`.
        """
        async with _canvas_lock(tool_context):
            spent = _charts_spent(tool_context)
            if spent:
                return spent
            if not charts:
                count_call(tool_context.state, "make_charts_failed")
                return _failed(
                    HarnessCode.TOOL_ARGUMENTS_INVALID,
                    "make_charts needs at least one chart. It only adds charts; it "
                    "cannot remove or clear cards on the canvas. Nothing changed.",
                )
            drawn = await _draw(charts, tool_context)
        result: dict[str, Any] = {"charts": drawn}
        if len(charts) > MAX_CHARTS_PER_CALL:
            result["not_drawn"] = (
                f"{len(charts) - MAX_CHARTS_PER_CALL} charts past the "
                f"{MAX_CHARTS_PER_CALL} one call may draw"
            )
        return result

    def ask_user(questions: list[AskQuestion], tool_context: ToolContext) -> dict[str, Any]:
        """Ask the user one to three questions before going on, with the likely
        answers to each. The user sees them together in one pause and answers
        each by picking an option, writing their own answer, or leaving it to
        you.

        Ask only what changes the answer. Each question offers at least one
        option. An option for a count or prices that a search result names
        carries the spec_id the result gives it, and the answer names that
        spec_id when the user picks it. Each spec_id goes on one option. End
        your turn after calling ask_user. Every answer comes back as the next
        message.
        """
        refused = _questions_refusal(questions, _named_spec_ids(tool_context))
        if refused:
            return refused
        spent = _spend(tool_context, "ask_user")
        if spent:
            return spent
        return _pend(tool_context, PendingAsk(questions=list(questions)))

    def plan(brief: str, asks: list[AskedOutput], tool_context: ToolContext) -> dict[str, Any]:
        """Hand a question to the planner, which builds a plan that
        constructs the answer from the data. The user approves the plan
        before it runs.

        `brief` is the whole question in full: every measure and place, and
        the period the user named. When the user named none, say so, with
        the year you chose if you chose one. The years a source holds are
        not the question's period. `asks` lists each output the user asked
        for, one per entry, in the user's terms: a number, a ranking, a
        chart, a comparison; at least one and at most six, each one short line. One figure
        the user asked for across several years is an ask of its own that
        sets `across_years`: "the total between 2010 and the latest" is one
        cumulative number beside the yearly figures. A view added to help
        read the answer is not an ask. The planner answers or declines each
        ask, so an output left off the list can drop out of the plan
        unnoticed.
        Each ask is `{"text", "by", "over", "across_years"}`. Every party a
        search this turn held open goes in some ask's `by` or `over`, and the
        plan keeps the parties each ask lists in `by` together in one output.
        End your turn after calling it; the plan's result comes back as the
        next message.
        """
        texts = [ask.text for ask in asks]
        refused = _asks_refusal(texts) or _breakdowns_missing(tool_context, asks)
        if refused:
            return refused
        spent = _spend(tool_context, "plan")
        if spent:
            return spent
        return _pend(
            tool_context,
            PendingPlan(
                brief=brief,
                asks=[text.strip() for text in texts],
                ask_breakdowns=_ask_breakdowns(asks),
                asks_across_years=_asks_across_years(asks),
            ),
        )

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
            return spent
        if surface is None:
            return _failed(
                HarnessCode.TOOL_NOT_CONFIGURED, "skills are not available on this server"
            )
        try:
            result = await surface.call_tool(tool=tool, arguments=arguments)
        except ToolSurfaceUnavailableError as exc:
            logger.warning(
                "orchestrator %s(%s) did not complete: %s", tool.name, dict(arguments), exc
            )
            return _failed(
                HarnessCode.TRANSIENT_UNAVAILABLE,
                f"{tool.name} could not reach data-mcp just now. Call it again, or answer "
                "without this skill.",
                retry_mode=RetryMode.RETRY_AFTER_DELAY,
            )
        if result.is_error:
            body = result.structured_content or {}
            return _service_failed(
                body, str(body.get("message") or result.text or "the skill could not be loaded")
            )
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
        can do, in general. Whether DFO has data on a topic or a measure is
        a data question, answered by searching for it. Returns DFO's own
        description and `counts`, its whole-catalogue totals; quote a count
        only from `counts`. Or returns an error saying why the description
        could not be read, which you pass on plainly."""
        faq, counts = await asyncio.gather(
            read_dfo_faq(dfo_faq_surface), read_catalogue_counts(dfo_faq_surface)
        )
        if isinstance(faq, DfoFaqUnavailable):
            if faq.reply == CANNOT_DESCRIBE_DFO:
                return _failed(HarnessCode.TOOL_NOT_CONFIGURED, faq.reply)
            return _failed(
                HarnessCode.TRANSIENT_UNAVAILABLE,
                faq.reply,
                retry_mode=RetryMode.RETRY_AFTER_DELAY,
            )
        return {"explainer": faq, "counts": counts}

    async def recall(
        part: RecallPart,
        tool_context: ToolContext,
        turn: int | None = None,
        component_id: str | None = None,
    ) -> dict[str, Any]:
        """Read back, in full, what an earlier turn of this conversation did:
        every source and decision of a plan, what a search read, or the
        user's canvas.

        `part`: "plan" for the plan records of a turn, "readings" for the
        search readings of a turn, "canvas" for every card of the conversation,
        listed as on_canvas or hidden, each with its id, title, type, facts
        and pinned state, or, with `component_id`, that one card's stored
        entry. For a plan or readings, `turn` is the turn's number as the
        conversation digest counts them; leave it out for the newest turn
        that has the part. A target that does not exist is refused with the
        values that do.
        """
        listing = part == "canvas" and component_id is None
        spent = _spend(tool_context, "recall_canvas" if listing else "recall")
        if spent:
            return spent
        if recall_part is None:
            return _failed(
                HarnessCode.TOOL_NOT_CONFIGURED, "recall is not available on this server"
            )
        try:
            return await recall_part(part=part, turn=turn, component_id=component_id)
        except RecallRefusedError as exc:
            return {
                "refusal": str(exc),
                "code": ToolErrorCode.REQUEST_INVALID.value,
                "legal": exc.legal,
            }

    def _forget_drawn(tool_context: ToolContext, removed: list[str]) -> None:
        """Drop the cards `removed` took off from the turn's drawn cards, so a
        later `make_charts` of one draws it again."""
        held = tool_context.state.get(_DRAWN_CARDS_KEY) or {}
        if not removed or held.get("invocation") != tool_context.invocation_id:
            return
        gone = set(removed)
        tool_context.state[_DRAWN_CARDS_KEY] = {
            "invocation": tool_context.invocation_id,
            "drawn": {
                identity: card
                for identity, card in held["drawn"].items()
                if card["component_id"] not in gone
            },
        }

    async def remove_cards_tool(
        tool_context: ToolContext,
        component_ids: list[str] | None = None,
        all_cards: bool = False,
    ) -> dict[str, Any]:
        """Take cards off the user's canvas.

        `component_ids` names the cards to take off, by the component ids the
        canvas lists; set `all_cards` to take off every card instead. Returns
        the ids taken off; a named card already off the canvas needs nothing,
        and an id this conversation never held is named back.
        """
        spent = _spend(tool_context, "remove_cards")
        if spent:
            return spent
        if remove_cards is None:
            return _failed(
                HarnessCode.TOOL_NOT_CONFIGURED, "removing cards is not available on this server"
            )
        if not component_ids and not all_cards:
            return _failed(
                HarnessCode.TOOL_ARGUMENTS_INVALID,
                "Name the cards to take off, or set all_cards. Nothing changed.",
            )
        _canvas_series(tool_context)
        async with _canvas_lock(tool_context):
            removed, missing = await remove_cards(
                component_ids=component_ids or [], all_cards=all_cards
            )
            _forget_drawn(tool_context, removed)
        if not removed and missing and len(missing) == len(component_ids or []):
            return _failed(
                HarnessCode.CARD_NOT_FOUND,
                "No card of this conversation has those ids. Nothing changed.",
            )
        if not removed:
            note = (
                "The canvas was already empty."
                if all_cards
                else "Those cards were already off the canvas."
            )
            return {"removed": [], "note": note}
        return {"removed": removed} | ({"not_in_conversation": missing} if missing else {})

    remove_cards_tool.__name__ = "remove_cards"

    return [
        _validated_arguments(search, hints=_SEARCH_ARGUMENT_HINTS),
        *(
            _validated_arguments(tool)
            for tool in (
                fetch,
                make_charts,
                ask_user,
                plan,
                get_skill,
                get_skill_file,
                explain_dfo,
                recall,
                remove_cards_tool,
            )
        ),
    ]


__all__ = ["TOOL_NAMES", "Chart", "Resolver", "build_orchestrator_tools"]
