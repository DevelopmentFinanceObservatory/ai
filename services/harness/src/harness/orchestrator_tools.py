"""The orchestrator's tools, over the resolver, executor and tool surface the workflow uses.

`search` asks data-mcp's `search` once for every part of a question;
`fetch` retrieves one dataset and describes its frames; `make_charts`
draws on the canvas through the graph's own presentation checks;
`get_skill` loads a skill a search result named and `get_skill_file`
reads on through it; `recall` returns an earlier turn's plan record,
readings or a canvas component in full; `explain_dfo` reads dfo-faq, DFO's
own description of itself (`harness.dfo_faq`). `ask_user` and `plan` record
one pending action for the workflow's `route` node (`harness.graph`) to act
on once the model's turn ends: a pause for the user's answer, or the plan
path.

The model sees digests only: part readings, frame views, one-line
summaries and chart facts, never rows. Frames are named to it by short
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
Each tool's calls per turn are capped (`harness.graph.state.BUDGETS`). The download surface
fetches one dataset and never plans.

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
import weakref
from collections.abc import Awaitable, Callable, Mapping, Sequence
from datetime import UTC, datetime
from types import FunctionType
from typing import Any, Literal, Protocol, get_type_hints

import httpx2
from compiler.sources import (
    default_reading,
    is_family,
    source_from_spec_block,
)
from compiler.surface import ToolSurface, ToolSurfaceUnavailableError
from data_mcp.get_data_tool import Breakdown, GetDataScope
from data_mcp.search_tool import SearchPart
from data_mcp.tool_errors import ToolErrorCode
from google.adk.tools import ToolContext
from mcp import MCPError
from mcp.types import REQUEST_TIMEOUT
from planner.parties import party_role
from planner.plan import MAX_ASK_CHARS, MAX_PLAN_ASKS, earlier_block, recorded_identity
from planner.skills import GET_SKILL, GET_SKILL_FILE
from planner.workflow import ToolId
from pydantic import BaseModel, Field, TypeAdapter, ValidationError
from translator.discovery import (
    DefinitionNotResolvedError,
    PartNotResolvedError,
    resolve_definition,
    resolved_spec,
)

from harness.chart_binding import (
    LINE_CHART_MAX_VALUES,
    MAX_SERIES_PER_COMPONENT,
    select_binding_ok,
)
from harness.dfo_faq import DfoFaqUnavailable, read_dfo_faq
from harness.graph.seams import Executor
from harness.graph.state import (
    ORCHESTRATOR_CALLS_KEY,
    ORCHESTRATOR_FRAMES_KEY,
    ORCHESTRATOR_PENDING_KEY,
    ORCHESTRATOR_READINGS_KEY,
    ORCHESTRATOR_SEARCHES_KEY,
    ORCHESTRATOR_SPEC_WARNINGS_KEY,
    ORCHESTRATOR_SPECS_KEY,
    ORCHESTRATOR_UNAVAILABLE_KEY,
    AskOption,
    PendingAsk,
    PendingPlan,
    pending_action,
    register_frames,
    store_search,
    turn_budgets,
)
from harness.studio_state import RecallPart, RecallRefusedError
from harness.tool_contracts import ResolverError, chosen_part, measure_choice
from harness.turn_records import SearchReading
from harness.wire import usable_component

logger = logging.getLogger(__name__)

# Charts one make_charts call may draw.
MAX_CHARTS_PER_CALL = 8
# The turn's completed fetches: `_fetch_key` -> the spec_id fetched, its
# summary and the ids of its frames, every one described, or the refusal of a
# fetch that served no observations.
_FETCHES_KEY = "orchestrator:fetches"
# The code dfo-apis warns with, and `fetch` refuses with, when a mint serves
# no observations.
EMPTY_RESULT = "empty_result"
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
            "what the chart shows and the period it shows; a stat card's title carries "
            "the period of the value on the card. The exception is a chart whose period "
            "moves with the data, which is one on a frame whose view says rolling: true "
            "and one given latest_periods: its title leaves out the years it shows "
            "('over the last five years', never '2019-2024'), and a title naming one of "
            "them is refused"
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


_ANSWER_NOW = "Answer with what you have, and say plainly what you could not do."


def _budgets(tool_context: ToolContext) -> dict[str, int]:
    return turn_budgets(tool_context.state)


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
        return {
            "error": f"{name}: {used} call(s) this turn had invalid arguments, "
            f"the limit. {_ANSWER_NOW}"
        }
    _count(tool_context, "invalid_arguments")
    return {
        "error": f"{name}: {'; '.join(line for _, line in problems)}. "
        "Correct these arguments and call again.",
        "invalid": [entry for entry, _ in problems],
    }


def _validated_arguments(
    tool: FunctionType, *, hints: ArgumentHints | None = None
) -> Callable[..., Any]:
    """`tool` wrapped to validate every argument but `tool_context` before it runs.

    ADK hands a function tool the model's arguments as parsed JSON, whatever
    the annotations say. The wrapper validates them with pydantic and calls
    `tool` with the validated values. If any fail, it returns the problems as
    `{"error": <prose>, "invalid": [{"loc", "type"}, ...]}`, with `hints`
    appended to the prose of the problems they match, and does not call
    `tool`. Rejected calls count against `BUDGETS["invalid_arguments"]`; past
    it the reply is `{"error": ...}` alone. A tool with no model arguments is
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


def _named_spec_ids(tool_context: ToolContext) -> set[str]:
    """Every spec_id this turn's search results named: each part's own and each
    candidate's (`ORCHESTRATOR_SPECS_KEY`), and each other count a part's
    result offers (`measure_choice`)."""
    specs = tool_context.state.get(ORCHESTRATOR_SPECS_KEY) or {}
    named = set(specs)
    for block in specs.values():
        named.update(
            alternative["spec_id"]
            for alternative in measure_choice(block or {}).get("alternatives") or ()
        )
    return named


def _options_refusal(options: list[AskOption], named: set[str]) -> dict[str, Any] | None:
    """The refusal of an `ask_user` call whose options carry a spec_id no search
    result this turn named (`_named_spec_ids`), or one spec_id twice, or None."""
    spec_ids = [option.spec_id for option in options if option.spec_id is not None]
    problems = [
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
            "code": FETCH_UNAVAILABLE,
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
    gives for it, with dfo-apis' own `empty_result` warning when it sent one."""
    warning = next((w for w in data.get("warnings") or [] if w.get("code") == EMPTY_RESULT), {})
    said = f" ({warning['message']})" if warning.get("message") else ""
    message = (
        f"This data holds no observations for the places and period asked{said}. Fetching it "
        "again unchanged returns nothing: change the places or period, or answer without it."
    )
    return {"outcome": "refusal", "refusal": {"code": EMPTY_RESULT, "message": message}}


def _attach_fetch(tool_context: ToolContext, result: dict[str, Any]) -> None:
    """Add a `fetch` result to every stored search reading of its spec_id.

    A spec_id that no reading holds as its own, but one offered as a
    candidate or as another count (`SearchReading.alternatives`), first
    gains a reading of its own, read from its block (`chosen_part`) with the
    offering reading's text, skills and departures and asked parties and
    its own warnings, else the offering reading's. A spec_id no reading
    offers as a candidate is another count, and its reading names the
    reading that offered it (`SearchReading.other_count_of`).
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
    count_of = next(
        (r for r in searches if any(a.spec_id == spec_id for a in r.alternatives)), None
    )
    offered = candidate_of or count_of
    if block and offered is not None and all(r.spec_id != spec_id for r in searches):
        warnings = (state.get(ORCHESTRATOR_SPEC_WARNINGS_KEY) or {}).get(spec_id)
        if warnings is None:
            warnings = [w.model_dump() for w in offered.warnings]
        chosen = SearchReading.from_part(
            {**chosen_part(block, variable=offered.text), "warnings": warnings},
            skills=offered.skills,
            observed_at=datetime.now(UTC),
        )
        searches.append(
            chosen.model_copy(
                update={
                    "separate": offered.separate,
                    "departures": offered.departures,
                    "asked_roles": offered.asked_roles,
                    "other_count_of": None if candidate_of else offered.spec_id,
                }
            )
        )
    state[ORCHESTRATOR_SEARCHES_KEY] = [
        (reading.with_fetch(result) if reading.spec_id == spec_id else reading).model_dump(
            mode="json"
        )
        for reading in searches
    ]


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
    the search digest (`harness.tool_contracts.resolver_digest`)."""

    async def __call__(
        self, *, question: str, parts: list[dict[str, Any]], raw_text: bool = False
    ) -> dict[str, Any]: ...


def build_orchestrator_tools(
    *,
    resolver: Resolver,
    executor: Executor,
    surface: ToolSurface | None = None,
    dfo_faq_surface: ToolSurface | None = None,
    definition_surface: ToolSurface | None = None,
    recall_part: Callable[..., Awaitable[dict[str, Any]]] | None = None,
    remove_cards: Callable[..., Awaitable[tuple[list[str], list[str]]]] | None = None,
) -> list[Any]:
    """The orchestrator's tools, bound to this server's resolver, executor and tool surface.

    `dfo_faq_surface` is the surface `explain_dfo` reads dfo-faq through: the
    bare one, off the Activity feed, since `explain_dfo`'s own call is the
    step the user sees. Left unset, `explain_dfo` says it cannot describe DFO.
    `definition_surface` is the surface `fetch` resolves a definition on
    record through (`_recorded_block`), the bare one, off the Activity feed,
    as the graph's `seed` node stores definitions. Left unset, such a fetch
    is refused.
    `recall_part` is `harness.studio_state.make_recall_part`'s seam. Left
    unset, `recall` says there is nothing to recall from. `remove_cards` is
    `harness.studio_state.make_remove_cards`'s seam; left unset, the tool
    says it is not available.
    """
    # One lock per invocation, held by a make_charts call from its budget
    # check to its count, so concurrent calls draw one after another and each
    # is admitted on exact counts. An entry lasts while a call holds its lock.
    chart_locks: weakref.WeakValueDictionary[str, asyncio.Lock] = weakref.WeakValueDictionary()

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
        An entity that names a set of places keeps
        the set's name as its text, with `members: true` when the question
        asks for each member on its own: "African countries" is
        `{"text": "Africa", "members": true}`, and "Africa" alone is the
        region as one place. All parts go in one call. A flow measure such
        as aid, donors or recipients needs `role` on each entity in
        `entities`, as spoken. Use `to` for a recipient and `from` for a
        donor. Aid to Kenya is `{"text": "Kenya", "role": "to"}`. A
        comparison of two categories of one measure ("health versus
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
        with the others under `alternatives`, each with its spec_id and
        label. And
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
            return {"error": spent}
        parts = parts or []
        raw_text = as_text or not parts
        wire_parts = [] if raw_text else [part.model_dump(exclude_none=True) for part in parts]
        try:
            digest = await resolver(question=query, parts=wire_parts, raw_text=raw_text)
        except ResolverError as exc:
            return {"error": (exc.refusal or {}).get("message") or str(exc)}
        # Search answers parts in the order asked, so a digest names the part
        # at its own position when the counts agree. When they do not, no
        # digest is tied to a part, and none records parties asked.
        digests = digest.get("part_digests") or []
        asked = [_asked_roles(part) for part in parts] if len(parts) == len(digests) else []
        readings = store_search(tool_context.state, digest, asked=asked)
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
        values; or a refusal with its `code` and reason, and the platform's
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
            return {"error": spent}
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
            elif refused["code"] == EMPTY_RESULT:
                # Kept for the turn, so the same fetch again answers from here.
                tool_context.state[_FETCHES_KEY] = {
                    **(tool_context.state.get(_FETCHES_KEY) or {}),
                    key: refused,
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
            for chart in charts[:MAX_CHARTS_PER_CALL]:
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
                intent: dict[str, Any] = {
                    "type": chart.type,
                    "frame_ref": frame["ref"],
                    "select": select,
                    "title": chart.title[:120],
                }
                if chart.latest_periods is not None:
                    intent["window"] = _latest_window(chart.latest_periods)
                try:
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
                        land(component)
                except Exception as error:
                    # Charts already drawn stay on the canvas and in the result.
                    # A write that timed out may still have stored this chart,
                    # so the model is told not to draw it again.
                    logger.exception("make_charts: drawing %r failed", chart.title)
                    drawn.append(
                        {
                            "title": chart.title,
                            "error": (
                                f"drawing it failed inside DFO ({type(error).__name__}), and "
                                "the charts after it in this call were not drawn. Do not "
                                "draw this chart again: it may be on the canvas already."
                            ),
                        }
                    )
                    break
                landed = True
                drawn.append({"title": chart.title, "facts": component["fact_summary"]})
        finally:
            # After the awaits, so the counts are read afresh. The `finally`
            # covers cancellation and anything raised outside the per-chart
            # `try`. A failed write is caught above.
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
            if not charts:
                _count(tool_context, "make_charts_failed")
                return {
                    "error": "make_charts needs at least one chart. It only adds charts; it "
                    "cannot remove or clear cards on the canvas. Nothing changed."
                }
            drawn = await _draw(charts, tool_context)
        result: dict[str, Any] = {"charts": drawn}
        if len(charts) > MAX_CHARTS_PER_CALL:
            result["not_drawn"] = (
                f"{len(charts) - MAX_CHARTS_PER_CALL} charts past the "
                f"{MAX_CHARTS_PER_CALL} one call may draw"
            )
        return result

    def ask_user(
        question: str, options: list[AskOption], tool_context: ToolContext
    ) -> dict[str, Any]:
        """Ask the user a question before going on, offering the likely answers.

        `options` are the answers to choose from, at least one; the user may
        also write their own or skip. An option for a way of counting that a
        search result names carries that count's spec_id, and the answer
        names it when the user picks it. End your turn after calling it; the
        answer comes back as the next message.
        """
        if not options:
            return {"error": "ask_user needs at least one option to offer."}
        refused = _options_refusal(options, _named_spec_ids(tool_context))
        if refused:
            return refused
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
            return {"error": spent}
        if remove_cards is None:
            return {"error": "removing cards is not available on this server"}
        if not component_ids and not all_cards:
            return {"error": "Name the cards to take off, or set all_cards. Nothing changed."}
        removed, missing = await remove_cards(
            component_ids=component_ids or [], all_cards=all_cards
        )
        if not removed and missing and len(missing) == len(component_ids or []):
            return {"error": "No card of this conversation has those ids. Nothing changed."}
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
