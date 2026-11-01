"""One planner entry, run as the served harness runs it, from a task and its recording.

A task (`tasks/<id>.json`) states a user's question, the searches the orchestrator ran
for it, and the brief and asks its `plan` tool handed over. Its recording
(`recordings/<id>.json`) holds what data-mcp answered: each search's envelope, and the
answers to each skill a planner may load, which `recordings/answers/` stores once
however many tasks read them, and under `translation` every call the translation of
each of the task's payloads makes. `run_entry` replays them through production code:

- the orchestrator's own `search` tool (`harness.orchestrator_tools`) reads each
  recorded envelope through `resolver_digest` and stores the readings, spec blocks and
  warnings as a served turn does (`harness.graph.state.store_search`);
- its `plan` tool checks the asks and records the hand-off;
- the graph's `route`, `plan_start` and `planner` nodes, wired as the served workflow
  wires them (`harness.graph.workflow.planner_entry`), run inside a conversation that
  holds those spec blocks as `harness.studio` holds a turn's
  (`planner.plan.current_recorded_identity`), and hand the brief to the real
  `PlannerSeam`, which renders the planner's first message from that state and runs
  the planner with production's instruction, tools and callbacks, each request
  under the `TurnBudget` a served request gets (`harness.turn_budget`).

Two edges are replaced. The planner's `get_skill` and `get_skill_file` read the
recording through `RecordedSurface`. The seam's translate step is the eval's own,
which ends the entry once a plan is recorded (`PlanRecorded`), so a run measures the
draft up to its first accepted plan. Given `TranslationCalls`, it first runs
production's translation on that plan (`translator.tool.build_translate`), its calls
to data-mcp and the graph's resolve door answered from the recording; that is how
`validate.py` checks each plan compiles.
Every entry runs at the planner's served thinking level and model-time budget.
The planner streams, as it does in a studio turn.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import hashlib
import json
import logging
import time
import traceback
from collections.abc import Awaitable, Callable, Collection, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

from compiler.surface import ToolCallResult, ToolId, ToolSchema, ToolSurfaceUnavailableError
from data_mcp import resolution
from data_mcp.errors import UpstreamError
from data_mcp.graph_wire import NodesResponse
from data_mcp.node_lookup import NodeOutcome
from data_mcp.search_tool import SearchContext
from google.adk import Event
from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.run_config import RunConfig, StreamingMode
from google.adk.events import EventActions
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService, Session
from google.adk.workflow import START, Workflow
from google.adk.workflow.utils._workflow_hitl_utils import create_request_input_response
from google.genai import types
from harness.graph.seams import PlannerModel
from harness.graph.state import ORCHESTRATOR_SPECS_KEY, GraphState
from harness.graph.workflow import planner_entry
from harness.orchestrator_tools import (
    HANDED_OFF,
    AskedOutput,
    Resolver,
    build_orchestrator_tools,
)
from harness.planner_seam import PlannerSeam, PlannerSeamError
from harness.prefix_cache import PrefixCache
from harness.settings import Settings
from harness.studio_state import ConversationState
from harness.tool_contracts import resolver_digest
from harness.turn_budget import CLOSED_BY_KEY, TurnBudget, current_turn_budget
from harness.turn_events import REQUEST_INPUT_FUNCTION_CALL_NAME
from harness.wiring import handing_back_planner_failures
from planner.agent import AGENT_NAME as PLANNER_AGENT
from planner.agent import read_full_history
from planner.call import PlannerCall
from planner.plan import (
    Plan,
    SessionState,
    current_recorded_identity,
    load_plan,
    unstated_period_reductions,
)
from planner.tool_results import ProblemCode
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from submission.service import first_year_reads
from translator.discovery import DefinitionNotResolvedError, graph_describe, graph_resolve
from translator.tool import Translate, build_translate

EVAL_DIR = Path(__file__).resolve().parent
TASKS_DIR = EVAL_DIR / "tasks"
RECORDINGS_DIR = EVAL_DIR / "recordings"
RECORDING_SCHEMA = 2

# Every requirement kind `propose_plan` declares, in the order reports list them.
KINDS = (
    "dataset",
    "aggregate",
    "growth",
    "comparison",
    "sum_of",
    "rank",
    "share_of",
    "quotient_of",
    "join",
    "filter",
)

# The refusal `propose_plan` answers a plan reducing across years for an ask not marked
# `across_years`.
PERIOD_REFUSAL = ProblemCode.PERIOD_REDUCTION_UNSTATED.value

_APP = "planner_entry_eval"
_USER = "eval"


# --- Tasks ---------------------------------------------------------------------------


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Search(_Strict):
    """One `search` call the orchestrator made, with the arguments it sent."""

    query: str
    parts: list[dict[str, Any]] = []
    as_text: bool = False


class Neighbour(_Strict):
    """The kind a task is most easily mistaken for, and what tells them apart."""

    kind: str
    distinction: str


class PeriodReduction(StrEnum):
    """What a plan does with the years of its data (`plan_period_reduction`).

    `NONE`: no step turns several years into one figure, so the plan keeps the years
    or reads one. `ACROSS_YEARS`: a step does, and every ask its results answer is
    marked `across_years`. `UNSTATED`: a step does and `propose_plan` refuses it, as
    it does where a result answers an unmarked ask or no ask.
    """

    NONE = "none"
    ACROSS_YEARS = "across_years"
    UNSTATED = "unstated"


class Alternative(_Strict):
    """Another plan a model could correctly write for a task. `period_reduction` is set
    when the plan does something else with its years than the task's reference."""

    description: str
    payload: dict[str, Any]
    first_proposal_refusal: str | None = None
    period_reduction: PeriodReduction | None = None


class Task(_Strict):
    """One question, as the orchestrator hands it to the planner.

    `kind` is the requirement kind the task is filed under, one a correct plan for it
    uses. `origin` says where the question came from: the planner's first hand-built
    task set (`pilot`), a traced conversation on the dev deployment, or written for
    this set. A plan passes kind selection under `selects_kinds`. `unit_asked` says
    whether the question or the brief names a unit; a dataset requirement stating one
    when neither does is an unasked unit. A task with `first_proposal_refusal` ends
    correctly in that refusal followed by an ask of the user. An ask is its text
    alone, or the `plan` tool's object naming the parties it lists one by one (`by`)
    or totals across (`over`), and whether the user asked for it as one figure over
    several years taken together (`across_years`). A task whose search holds a party
    open names it in one. `period_reduction` is what the reference plan does with the
    years of its data, `none` or `across_years`, and what a run's accepted plan is
    held to.
    """

    id: str
    kind: str
    origin: str
    question: str
    searches: list[Search] = Field(min_length=1)
    brief: str
    asks: list[str | AskedOutput]
    required_kinds: list[str]
    required_one_of: list[str] = []
    acceptable_kinds: list[str]
    neighbour: Neighbour | None = None
    unit_asked: bool = False
    first_proposal_refusal: str | None = None
    period_reduction: PeriodReduction
    reference_payload: dict[str, Any]
    alternative_payloads: list[Alternative] = []

    @property
    def across_years_numbers(self) -> tuple[int, ...]:
        """The numbers of the asks marked `across_years`, as `Plan.asks_across_years`
        holds them."""
        return tuple(
            number
            for number, ask in enumerate(self.asks)
            if isinstance(ask, AskedOutput) and ask.across_years
        )

    @property
    def asks_across_years(self) -> bool:
        """Whether any ask is marked `across_years`."""
        return bool(self.across_years_numbers)

    def selects_kinds(self, kinds: Collection[str]) -> bool:
        """Whether a plan using `kinds` selects this task's kinds: every `required_kinds`
        kind, at least one `required_one_of` kind when the task lists any, and nothing
        outside `acceptable_kinds`."""
        used = set(kinds)
        return (
            bool(used)
            and set(self.required_kinds) <= used <= set(self.acceptable_kinds)
            and (not self.required_one_of or bool(used & set(self.required_one_of)))
        )


def load_tasks(directory: Path = TASKS_DIR) -> list[Task]:
    """Every task in `directory`, by kind in `KINDS` order and then by id."""
    tasks = [
        Task.model_validate_json(path.read_text(encoding="utf-8"))
        for path in sorted(directory.glob("*.json"))
    ]
    order = {kind: index for index, kind in enumerate(KINDS)}
    return sorted(tasks, key=lambda task: (order.get(task.kind, len(KINDS)), task.id))


# --- Recordings ----------------------------------------------------------------------


def recording_path(task_id: str, directory: Path = RECORDINGS_DIR) -> Path:
    return directory / f"{task_id}.json"


def _answers_dir(directory: Path) -> Path:
    return directory / "answers"


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _write_json(path: Path, value: Any) -> None:
    # The layout the repository's pretty-format-json hook writes, so a commit keeps it.
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def answer_id(call: str, answer: Mapping[str, Any]) -> str:
    """The content address of one recorded tool answer, so the same call answered the
    same way is stored once whichever tasks recorded it."""
    return hashlib.sha256(_canonical({"call": call, "answer": answer}).encode()).hexdigest()[:16]


def _read_answers(task_id: str, names: Sequence[str], directory: Path) -> dict[str, Any]:
    """The stored answers `names` address, by the call each answers."""
    answers: dict[str, Any] = {}
    for name in names:
        path = _answers_dir(directory) / f"{name}.json"
        if not path.exists():
            raise RecordingMismatchError(f"{task_id}: answer {name} is not in answers/")
        stored = json.loads(path.read_text(encoding="utf-8"))
        if answer_id(stored["call"], stored["answer"]) != name:
            raise RecordingMismatchError(f"{task_id}: answer {name} does not match its content")
        answers[stored["call"]] = stored["answer"]
    return answers


def _store_answers(answers: Mapping[str, Any], directory: Path) -> list[str]:
    """Write each of `answers`, by the call it answers, to the shared store under its
    `answer_id`; the names, sorted."""
    store = _answers_dir(directory)
    store.mkdir(parents=True, exist_ok=True)
    names = []
    for call, answer in answers.items():
        name = answer_id(call, answer)
        names.append(name)
        _write_json(store / f"{name}.json", {"call": call, "answer": answer})
    return sorted(names)


def _named_answers(body: Mapping[str, Any]) -> list[str]:
    """Every stored answer a recording file names: its tool answers, then its
    translation's."""
    return [*body["answers"], *(body.get("translation") or {}).get("answers", [])]


def load_recording(task_id: str, directory: Path = RECORDINGS_DIR) -> dict[str, Any]:
    """The recording of `task_id`, its tool answers read back in under `tools`, and its
    translation's under `translation["calls"]`. A translation recorded before the node
    door was recorded holds no `nodes`, so a replay asking for one misses it.

    Refuses a recording written under another schema, one naming an answer the store
    lacks, and an answer whose content no longer matches its address.
    """
    recording = json.loads(recording_path(task_id, directory).read_text(encoding="utf-8"))
    if recording.get("schema") != RECORDING_SCHEMA:
        raise RecordingMismatchError(f"{task_id}: recording schema {recording.get('schema')}")
    tools = _read_answers(task_id, recording.pop("answers"), directory)
    if (translation := recording.get("translation")) is not None:
        recording["translation"] = {
            "calls": _read_answers(task_id, translation["answers"], directory),
            "resolved": translation["resolved"],
            "nodes": translation.get("nodes") or {},
        }
    return {**recording, "tools": tools}


def save_recording(
    task_id: str, recording: Mapping[str, Any], directory: Path = RECORDINGS_DIR
) -> None:
    """Write `recording`: each tool answer, and each answer its translation read, to the
    shared store under its `answer_id`, and the task's file naming the answers it
    holds. A rerun that records the same answers writes the same files."""
    body = {key: value for key, value in recording.items() if key != "tools"}
    if (translation := recording.get("translation")) is not None:
        # A translation that read no node keeps the form it had before nodes were
        # recorded.
        nodes = translation.get("nodes") or {}
        body["translation"] = {
            "answers": _store_answers(translation["calls"], directory),
            "resolved": translation["resolved"],
            **({"nodes": nodes} if nodes else {}),
        }
    names = _store_answers(recording["tools"], directory)
    _write_json(recording_path(task_id, directory), {**body, "answers": names})


def stray_recordings(task_ids: Collection[str], directory: Path = RECORDINGS_DIR) -> list[Path]:
    """Recording files of no task in `task_ids`, then stored answers none of those tasks'
    recordings name, which a removed task or a re-record leaves behind."""
    recordings = sorted(directory.glob("*.json"))
    named = {
        name
        for path in recordings
        if path.stem in task_ids
        for name in _named_answers(json.loads(path.read_text(encoding="utf-8")))
    }
    return [path for path in recordings if path.stem not in task_ids] + sorted(
        path for path in _answers_dir(directory).glob("*.json") if path.stem not in named
    )


# --- Translation replay --------------------------------------------------------------

type LiveCall = Callable[[ToolId, Mapping[str, object]], Awaitable[dict[str, Any]]]


class TranslationCalls:
    """The `ToolSurface`, resolve door and node door production's translation reads,
    answered from `held`, a recording's `translation`: tool answers under `calls`, by
    `surface_key`, each name the resolve door was asked under `resolved`, the graph
    entity it resolved to or `None`, and each dcid the node door was asked under
    `nodes`, the envelope it answered.

    A call `held` lacks is answered as an unreachable server is and listed in `misses`,
    so a payload whose translation now asks anything its recording does not hold
    fails. With `live` (`record.py`), it is made instead: a tool call through `live`, a
    name through the graph's own door (`translator.discovery.graph_resolve`), a dcid
    through its node door (`translator.discovery.graph_describe`), and kept.
    A live call that raised, or answered a refusal to retry later
    (`retry_mode: retry_after_delay`), is kept out and listed in `failures`. `kept` is
    what a recording of the calls made so far holds. The tools listed are `schemas`,
    the real servers' own.
    """

    def __init__(
        self,
        held: Mapping[str, Any],
        *,
        schemas: Sequence[ToolSchema],
        live: LiveCall | None = None,
    ) -> None:
        self._calls: dict[str, Any] = dict(held.get("calls") or {})
        self._resolved: dict[str, Any] = dict(held.get("resolved") or {})
        self._nodes: dict[str, Any] = dict(held.get("nodes") or {})
        self._dcids: set[str] = set()
        self._schemas = list(schemas)
        self._live = live
        self._used: set[str] = set()
        self._names: set[str] = set()
        self.misses: list[str] = []
        self.failures: list[str] = []

    async def list_tools(self) -> list[ToolSchema]:
        return self._schemas

    async def call_tool(
        self, *, tool: ToolId, arguments: Mapping[str, object], on_progress: Any = None
    ) -> ToolCallResult:
        del on_progress
        key = surface_key(tool.name, arguments)
        if key in self._calls:
            self._used.add(key)
            return ToolCallResult.model_validate(self._calls[key])
        if self._live is None:
            self.misses.append(key)
            raise ToolSurfaceUnavailableError(f"the recording holds no answer for {key}")
        try:
            answer = await self._live(tool, arguments)
        except BaseException as exc:
            self.failures.append(f"{key}: {type(exc).__name__}: {exc}")
            raise
        if (
            answer["is_error"]
            and (answer["structured_content"] or {}).get("retry_mode") == "retry_after_delay"
        ):
            self.failures.append(f"{key}: refused for now: {answer['structured_content']}")
        else:
            self._calls[key] = answer
            self._used.add(key)
        return ToolCallResult.model_validate(answer)

    async def resolve_entities(
        self, scope: Mapping[str, Sequence[str]]
    ) -> dict[str, dict[str, resolution.ResolutionOutcome]]:
        """The graph's resolve door over the recorded names. The translation reads only
        whether a name resolved to one entity, and to which."""
        names = list(dict.fromkeys(name.strip() for names in scope.values() for name in names))
        missing = [name for name in names if name not in self._resolved]
        if missing and self._live is None:
            self.misses += [f"resolve {name}" for name in missing]
            raise UpstreamError(status=None, code=None, detail="the recording holds no resolve")
        if missing:
            try:
                found = (await graph_resolve({"names": missing}))["names"]
            except BaseException as exc:
                self.failures.append(f"resolve {missing}: {type(exc).__name__}: {exc}")
                raise
            self._resolved |= {
                name: dataclasses.asdict(outcome)
                if isinstance(outcome, resolution.ResolvedEntity)
                else None
                for name, outcome in found.items()
            }
        self._names |= set(names)
        return {
            role: {name: _outcome(self._resolved[name.strip()]) for name in role_names}
            for role, role_names in scope.items()
        }

    async def describe_nodes(self, dcids: Sequence[str]) -> dict[str, NodeOutcome]:
        """The graph's node door over the recorded envelopes, as
        `data_mcp.node_lookup.describe_nodes` answers them."""
        asked = list(dict.fromkeys(dcids))
        missing = [dcid for dcid in asked if dcid not in self._nodes]
        if missing and self._live is None:
            self.misses += [f"node {dcid}" for dcid in missing]
            raise UpstreamError(status=None, code=None, detail="the recording holds no node")
        if missing:
            try:
                found = await graph_describe(missing)
            except BaseException as exc:
                self.failures.append(f"node {missing}: {type(exc).__name__}: {exc}")
                raise
            self._nodes |= {
                dcid: found[dcid].model_dump(mode="json", by_alias=True) for dcid in missing
            }
        self._dcids |= set(asked)
        entries = NodesResponse.model_validate({"nodes": [self._nodes[dcid] for dcid in asked]})
        return dict(zip(asked, entries.nodes, strict=True))

    def kept(self) -> dict[str, Any]:
        """The calls, names and dcids asked so far, as a recording's `translation` holds
        them."""
        return {
            "calls": {key: self._calls[key] for key in sorted(self._used)},
            "resolved": {name: self._resolved[name] for name in sorted(self._names)},
            "nodes": {dcid: self._nodes[dcid] for dcid in sorted(self._dcids)},
        }

    def unasked(self) -> list[str]:
        """What `held` records that no call asked for so far."""
        return (
            sorted(set(self._calls) - self._used)
            + [f"resolve {name}" for name in sorted(set(self._resolved) - self._names)]
            + [f"node {dcid}" for dcid in sorted(set(self._nodes) - self._dcids)]
        )


def _outcome(recorded: Mapping[str, Any] | None) -> resolution.ResolutionOutcome:
    return (
        resolution.NotFound(near_matches=[])
        if recorded is None
        else resolution.ResolvedEntity(**recorded)
    )


async def _no_anchor_read(workflow: Any, plan: Plan, *, surface: Any, held: Any) -> None:
    """The newest-year read a replayed translation makes: none. That read reads the
    data, which no recording holds, so it is skipped, as production skips it when the
    entry has no time left for it."""
    del workflow, plan, surface, held


class RecordingMismatchError(RuntimeError):
    """A replay asked for something its recording does not hold: re-record the task."""


class TaskStateError(RuntimeError):
    """The orchestrator's own tools refused a task's search or hand-off."""


def search_context_meta(context: SearchContext | None) -> dict[str, Any] | None:
    """The `_meta` a search carries for `context`, as `harness.mcp_client.McpResolver`
    sends it, or `None` when `context` states nothing Search reads
    (`data_mcp.search_tool.to_wire_request`): no user message and no carried
    entries send the same request as no context."""
    if context is None or (context.user_message is None and not context.carried_entries):
        return None
    return context.model_dump(mode="json", exclude_none=True)


class RecordedResolver:
    """The orchestrator's `Resolver`, answering each search from a recording, in order.

    A call whose arguments differ from the recorded call's is refused
    (`RecordingMismatchError`), so a task edited after recording cannot replay stale
    answers.
    """

    def __init__(self, calls: Sequence[Mapping[str, Any]]) -> None:
        self._calls = list(calls)

    async def __call__(
        self,
        *,
        question: str,
        parts: list[dict[str, Any]],
        raw_text: bool = False,
        context: SearchContext | None = None,
    ) -> dict[str, Any]:
        if not self._calls:
            raise RecordingMismatchError(f"no recorded search left for {question!r}")
        call = self._calls.pop(0)
        asked = {
            "question": question,
            "parts": parts,
            "raw_text": raw_text,
            "context": search_context_meta(context),
        }
        recorded = {key: call.get(key) for key in asked}
        if asked != recorded:
            raise RecordingMismatchError(f"search {asked} was recorded as {recorded}")
        return resolver_digest(call["envelope"])

    def left(self) -> int:
        return len(self._calls)


def surface_key(tool: str, arguments: Mapping[str, object]) -> str:
    """The key a recording files one tool call's answer under."""
    return f"{tool} {json.dumps(arguments, sort_keys=True, separators=(',', ':'))}"


class RecordedSurface:
    """The planner's `ToolSurface`, answering calls from a recording.

    The planner reaches data-mcp for `get_skill` and `get_skill_file`, and for a pinned
    dataset's first year. A call the recording lacks is answered as an unreachable
    server is (`ToolSurfaceUnavailableError`), which the planner's tools turn into a
    sentence saying the call did not complete, and is listed in `misses`.
    """

    def __init__(self, results: Mapping[str, Mapping[str, Any]]) -> None:
        self._results = dict(results)
        self.misses: list[str] = []

    async def list_tools(self) -> list[ToolSchema]:
        return []

    async def call_tool(
        self,
        *,
        tool: ToolId,
        arguments: Mapping[str, object],
        on_progress: Any = None,
    ) -> ToolCallResult:
        del on_progress
        key = surface_key(tool.name, arguments)
        if key not in self._results:
            self.misses.append(key)
            raise ToolSurfaceUnavailableError(f"the recording holds no answer for {key}")
        return ToolCallResult.model_validate(self._results[key])


# --- The orchestrator's hand-off -----------------------------------------------------


class _NoExecutor:
    """The executor `build_orchestrator_tools` requires; `search` and `plan` never call it."""

    async def __call__(self, **kwargs: Any) -> dict[str, Any]:
        raise AssertionError(f"the eval runs no plan step: {kwargs.get('tool')}")


async def orchestrator_state(task: Task, resolver: Resolver) -> dict[str, Any]:
    """The session state the orchestrator's `search` and `plan` calls leave for `task`.

    Raises:
        TaskStateError: If a search answers an error or `plan` hands nothing off.
    """
    tools = {
        tool.__name__: tool
        for tool in build_orchestrator_tools(resolver=resolver, executor=_NoExecutor())
    }
    context = SimpleNamespace(state={}, actions=EventActions())
    for search in task.searches:
        found = await tools["search"](
            query=search.query,
            tool_context=context,
            parts=search.parts or None,
            as_text=search.as_text,
        )
        if "error" in found:
            raise TaskStateError(f"{task.id}: search {search.query!r}: {found['error']}")
    handed = tools["plan"](brief=task.brief, asks=list(task.asks), tool_context=context)
    if handed.get(HANDED_OFF) != "plan":
        raise TaskStateError(f"{task.id}: plan answered {handed}")
    state: dict[str, Any] = context.state
    return state


# --- The planner entry ---------------------------------------------------------------


class PlanRecorded(Exception):  # noqa: N818 -- a stop signal, not a fault
    """Raised by the eval's translate step once `propose_plan` has recorded a plan.

    The seam answers a planner tool that raises with a final refusal
    (`planner_tool_failed`) and makes no further model call, so the entry ends on the
    first accepted proposal.
    """


class _DropPlanRecorded(logging.Filter):
    """Drops the seam's error log for the eval's own stop, which it reads as a tool failure."""

    def filter(self, record: logging.LogRecord) -> bool:
        return not (record.exc_info and isinstance(record.exc_info[1], PlanRecorded))


logging.getLogger("harness.planner_seam").addFilter(_DropPlanRecorded())


def quiet_logs() -> None:
    """Silence what a run's record already holds: ADK's traceback for a failed node and
    the wiring's line for an entry the seam gave up on."""
    for name in ("google_adk", "harness.wiring"):
        logging.getLogger(name).setLevel(logging.CRITICAL)


# The stop reason of an entry the seam gave up on (`PlannerSeamError`): the planner
# ended its turns with no accepted plan and no ask, and corrections did not change that.
GAVE_UP = "gave_up"


RequestEdit = Callable[[LlmRequest], None]


# A model call's usage before its whole answer arrives: unknown, and left so when the
# entry ends first, as a timeout does.
_UNFINISHED_CALL: dict[str, Any] = {
    "prompt_tokens": None,
    "cached_tokens": None,
    "output_tokens": None,
    "thinking_tokens": None,
    "latency_s": None,
}


@dataclass
class Probe:
    """What one entry did that its session's events do not carry: the plan the first
    accepted proposal recorded and the id of the `propose_plan` call that recorded it,
    the code of the refusal an entry with no plan ended on, each model call it
    started, with its tokens and latency once the whole answer arrived, and what
    `translator` answered for the plan when one is set."""

    plan: Plan | None = None
    plan_call: str | None = None
    refused_code: str | None = None
    calls: list[dict[str, Any]] = field(default_factory=list)
    # Production's translation, run on the recorded plan before the entry ends, and
    # what it answered.
    translator: Translate | None = None
    translation: dict[str, object] | None = None
    _started: float = 0.0
    # The requests the current turn budget had sent before the call in flight.
    _requests_before: int | None = None

    async def translate(self, call: PlannerCall) -> dict[str, object]:
        self.plan = load_plan(call.state)
        self.plan_call = call.call_id
        if self.translator is not None:
            self.translation = await self.translator(call)
        raise PlanRecorded

    def watch(self, seam: PlannerModel) -> PlannerModel:
        """`seam`, noting the code of each refusal it answers, and `GAVE_UP` when it
        raises `PlannerSeamError`."""
        return _Watched(seam, self)

    async def note(self, answering: Awaitable[dict[str, Any]]) -> dict[str, Any]:
        """`answering`'s answer, its refusal code noted as `watch` describes."""
        try:
            answer = await answering
        except PlannerSeamError:
            self.refused_code = GAVE_UP
            raise
        if "refused" in answer:
            code = answer["refused"]["code"]
            self.refused_code = None if code is None else str(code)
        return answer

    def before_model(
        self, callback_context: CallbackContext, llm_request: LlmRequest
    ) -> LlmResponse | None:
        """Note a call about to go to the model. It runs after the seam's own callbacks,
        so a request the seam answers itself is no call."""
        del callback_context, llm_request
        budget = current_turn_budget.get()
        self._requests_before = None if budget is None else budget.model_calls
        self._started = time.perf_counter()
        self.calls.append(dict(_UNFINISHED_CALL))
        return None

    def after_model(
        self, callback_context: CallbackContext, llm_response: LlmResponse
    ) -> LlmResponse | None:
        del callback_context
        budget = current_turn_budget.get()
        if (
            budget is not None
            and budget.model_calls == self._requests_before
            and CLOSED_BY_KEY in (llm_response.custom_metadata or {})
        ):
            # The budget answered this call itself, with no request to the model.
            self.calls.pop()
            self._requests_before = None
            return None
        usage = llm_response.usage_metadata
        # A streamed call answers in chunks and then once whole; count the whole answer.
        if usage is not None and not llm_response.partial and self.calls:
            self.calls[-1] = {
                "prompt_tokens": usage.prompt_token_count or 0,
                "cached_tokens": usage.cached_content_token_count or 0,
                "output_tokens": usage.candidates_token_count or 0,
                "thinking_tokens": usage.thoughts_token_count or 0,
                "latency_s": round(time.perf_counter() - self._started, 3),
            }
        return None


class _Watched:
    """The `PlannerModel` `Probe.watch` returns."""

    def __init__(self, seam: PlannerModel, probe: Probe) -> None:
        self._seam = seam
        self._probe = probe

    async def __call__(self, **kwargs: Any) -> dict[str, Any]:
        return await self._probe.note(self._seam(**kwargs))

    async def revise(self, **kwargs: Any) -> dict[str, Any]:
        return await self._probe.note(self._seam.revise(**kwargs))


def build_seam(
    *,
    probe: Probe,
    surface: RecordedSurface,
    prefix_cache: PrefixCache | None,
    edits: Sequence[RequestEdit] = (),
    model: BaseLlm | None = None,
) -> PlannerSeam:
    """The seam `harness.wiring.build_planner` builds, with the probe's translate step.

    The planner, its instruction, its tools and the seam's callbacks are production's
    own. `edits` change each request the planner sends its model (`arms.Arm`), and the
    probe notes each one and its answer. `model` stands in for the planner's model, as
    `validate.py`'s scripted model does.
    """

    def edit_request(callback_context: CallbackContext, llm_request: LlmRequest) -> None:
        del callback_context
        for edit in edits:
            edit(llm_request)

    return PlannerSeam(
        settings=Settings(),
        surface=surface,
        prefix_cache=prefix_cache,
        translate=probe.translate,
        model=model,
        before_model_callbacks=[edit_request, probe.before_model],
        after_model_callbacks=[probe.after_model],
    )


def entry_workflow(planner: PlannerModel) -> Workflow:
    """The served graph from the orchestrator's `plan` call to the end of the planner's
    entry (`harness.graph.workflow.planner_entry`), drafting with `planner`."""
    entry = planner_entry(planner)
    return Workflow(
        name="planner_entry", state_schema=GraphState, edges=[(START, entry.route), *entry.edges]
    )


@dataclass(frozen=True)
class EntryRun:
    """One finished entry: the planner's events and the user's answers to its asks, what
    the probe saw, and how it ended."""

    events: list[Event]
    state: dict[str, Any]
    probe: Probe
    inputs: str
    # The code of the bound that closed one of the entry's requests, if one did.
    budget_closed: str | None
    surface_misses: list[str]
    error: str | None
    timed_out: bool
    wall_s: float


async def run_entry(
    task: Task,
    recording: Mapping[str, Any],
    *,
    prefix_cache: PrefixCache | None,
    edits: Sequence[RequestEdit] = (),
    model: BaseLlm | None = None,
    timeout_s: float = 300.0,
    answers: Sequence[Mapping[str, Any]] = (),
    translation: TranslationCalls | None = None,
) -> EntryRun:
    """Run `task`'s planner entry once on a fresh session.

    The entry ends at the first plan `propose_plan` records, at an ask of the user, or
    wherever the seam ends it (a bound of its request's budget, its round budget, a
    correction it gave up on), or after `timeout_s`. The planner streams, as a studio
    turn runs it. `answers` are the user's replies to the entry's asks, in order; each
    resumes the paused entry, as the studio's interrupt route does, in a request of its
    own with a fresh budget. A live run passes none, so an ask ends it. With
    `translation`, production's translation runs on the recorded plan over those calls,
    and `probe.translation` holds its answer.
    """
    probe = Probe()
    if translation is not None:
        probe.translator = build_translate(
            surface=translation,
            resolve_entities=translation.resolve_entities,
            anchor_read=_no_anchor_read,
            describe_nodes=translation.describe_nodes,
        )
    surface = RecordedSurface(recording.get("tools") or {})
    resolver = RecordedResolver(recording["searches"])
    state = await orchestrator_state(task, resolver)
    if resolver.left():
        raise RecordingMismatchError(f"{task.id}: {resolver.left()} recorded searches went unasked")
    seam = build_seam(
        probe=probe, surface=surface, prefix_cache=prefix_cache, edits=edits, model=model
    )
    sessions = InMemorySessionService()
    session = await sessions.create_session(app_name=_APP, user_id=_USER, state=state)
    # The seam wrapped as `harness.server.create_app` wraps it.
    planner = handing_back_planner_failures(probe.watch(seam))
    runner = Runner(node=entry_workflow(planner), app_name=_APP, session_service=sessions)
    message = types.Content(role="user", parts=[types.Part.from_text(text=task.question)])
    # The conversation a studio turn runs in, holding the spec blocks the orchestrator's
    # searches returned, as `harness.studio` keeps them, so `propose_plan` and the
    # translation read a pin's identity on record (`planner.plan.recorded_identity`).
    conversation = ConversationState(conversation_id=session.id)
    conversation.turn_specs = dict(state.get(ORCHESTRATOR_SPECS_KEY) or {})
    pending = list(answers)
    error: str | None = None
    timed_out = False
    started = time.perf_counter()

    settings = Settings()
    budgets: list[TurnBudget] = []

    async def drive() -> None:
        content: types.Content | None = message
        while content is not None:
            budgets.append(TurnBudget.from_settings(settings))
            token = current_turn_budget.set(budgets[-1])
            identities = current_recorded_identity.set(conversation.recorded_identity)
            try:
                async for _event in runner.run_async(
                    user_id=_USER,
                    session_id=session.id,
                    new_message=content,
                    run_config=RunConfig(streaming_mode=StreamingMode.SSE),
                ):
                    pass
            finally:
                current_recorded_identity.reset(identities)
                current_turn_budget.reset(token)
            content = None
            if probe.plan is None and pending:
                events = _entry_events(await _session(sessions, session.id))
                ask = _last_call(events, REQUEST_INPUT_FUNCTION_CALL_NAME)
                if ask is not None and ask.id is not None and not _answered(events, ask.id):
                    content = types.Content(
                        role="user",
                        parts=[create_request_input_response(ask.id, dict(pending.pop(0)))],
                    )

    # Only this scope's expiry is the eval's timeout: a `TimeoutError` raised inside the
    # entry, such as the model client's own, is an error and is retried.
    limit = asyncio.timeout(timeout_s)
    try:
        async with limit:
            await drive()
    except Exception as exc:
        timed_out = limit.expired()
        if not timed_out:
            error = f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=8)}"[:3000]
    wall_s = time.perf_counter() - started
    ended = await _session(sessions, session.id)
    return EntryRun(
        events=_entry_events(ended),
        state=dict(ended.state),
        probe=probe,
        inputs=inputs_digest(task, recording),
        budget_closed=next(
            (str(budget.closed["code"]) for budget in reversed(budgets) if budget.closed), None
        ),
        surface_misses=list(surface.misses),
        error=error,
        timed_out=timed_out,
        wall_s=wall_s,
    )


def inputs_digest(task: Task, recording: Mapping[str, Any]) -> str:
    """A digest of what a run of `task` shows the planner: the question, the searches,
    the brief and asks, and the recorded search envelopes and tool answers. Runs pair
    only where their digests agree. An ask not marked `across_years` digests without
    that field, so a baseline stored before asks carried the mark still pairs on it."""
    asks = [
        ask
        if isinstance(ask, str)
        else ask.model_dump(mode="json", exclude=None if ask.across_years else {"across_years"})
        for ask in task.asks
    ]
    shown = {
        "question": task.question,
        "searches": [search.model_dump(mode="json") for search in task.searches],
        "brief": task.brief,
        "asks": asks,
        "recorded_searches": recording["searches"],
        "recorded_answers": recording["tools"],
    }
    canonical = json.dumps(shown, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


async def first_year_gaps(state: Mapping[str, Any], recording: Mapping[str, Any]) -> list[str]:
    """The first-year reads an entry that left `state` can make
    (`submission.service.first_year_reads`) that `recording` holds no answer for, by the key
    `RecordedSurface` looks each up under."""
    surface = RecordedSurface(recording["tools"])
    for identity in first_year_reads(cast(SessionState, state)):
        with contextlib.suppress(DefinitionNotResolvedError):
            await read_full_history(surface, identity)
    return surface.misses


async def _session(sessions: InMemorySessionService, session_id: str) -> Session:
    session = await sessions.get_session(app_name=_APP, user_id=_USER, session_id=session_id)
    assert session is not None
    return session


def _entry_events(session: Session) -> list[Event]:
    """The planner's events, and the user's answers to its asks, which the user authors."""
    return [
        event
        for event in session.events
        if event.author == PLANNER_AGENT
        or (event.author == "user" and event.get_function_responses())
    ]


def _last_call(events: Sequence[Event], name: str) -> types.FunctionCall | None:
    calls = [call for event in events for call in event.get_function_calls() if call.name == name]
    return calls[-1] if calls else None


def _answered(events: Sequence[Event], call_id: str) -> bool:
    return any(
        response.id == call_id for event in events for response in event.get_function_responses()
    )


# --- The record of one run -----------------------------------------------------------

RECORD_SCHEMA = 4


def tool_calls(events: Sequence[Event]) -> list[dict[str, Any]]:
    """Every tool call the planner made, in order, with the response it got: the tool's,
    or the user's answer to an ask. A call with no response is one the entry ended
    before it answered, or an ask still awaiting the user."""
    calls: list[dict[str, Any]] = []
    by_id: dict[str, dict[str, Any]] = {}
    for event in events:
        for call in event.get_function_calls():
            entry = {
                "id": call.id,
                "name": call.name,
                "args": dict(call.args or {}),
                "response": None,
            }
            calls.append(entry)
            if call.id is not None:
                by_id[call.id] = entry
        for response in event.get_function_responses():
            if response.id in by_id and by_id[response.id]["response"] is None:
                by_id[response.id]["response"] = response.response
    return calls


def _requirement_kinds(arguments: Mapping[str, Any]) -> list[str]:
    requirements = arguments.get("evidence_requirements")
    if not isinstance(requirements, list):
        return []
    return [
        item["kind"]
        for item in requirements
        if isinstance(item, dict) and isinstance(item.get("kind"), str)
    ]


def proposals(calls: Sequence[Mapping[str, Any]], *, recorded: str | None) -> list[dict[str, Any]]:
    """Each `propose_plan` call: what it sent and what the tool answered.

    The call `recorded`, which recorded the entry's plan, answers with the seam's
    tool-error text, since the eval's translate step stops inside it, so it is marked
    `ok`. A model response can hold several proposals, so the call is found by its id.
    """
    sent = [call for call in calls if call["name"] == "propose_plan"]
    if recorded is not None and recorded not in {call["id"] for call in sent}:
        raise ValueError(f"the proposal that recorded the plan, {recorded}, is not in the entry")
    found = []
    for call in sent:
        response = call["response"] if isinstance(call["response"], dict) else {}
        stopped = recorded is not None and call["id"] == recorded
        message = response.get("message")
        found.append(
            {
                "outcome": "ok" if stopped else response.get("outcome"),
                "code": None if stopped else response.get("code"),
                "message": message[:600] if isinstance(message, str) else None,
                "kinds": _requirement_kinds(call["args"]),
                "args": call["args"],
            }
        )
    return found


def period_reductions(plan: Plan) -> list[dict[str, Any]]:
    """Each step of `plan` that turns several years into one figure, with those years
    and whether `propose_plan` refuses it.

    Both readings are `planner.plan.unstated_period_reductions`, the rule `propose_plan`
    enforces at a first entry: once on the plan with no ask marked across years and no
    licence, which reports every reduction, and once on the plan as it stands, which
    reports those it refuses.
    """
    refused = {gap.step_id for gap in unstated_period_reductions(plan)}
    unmarked = plan.model_copy(update={"asks_across_years": (), "period_licences": ()})
    return [
        {
            "step": gap.step_id,
            "start": gap.held.years.start,
            "end": gap.held.years.end,
            "refused": gap.step_id in refused,
        }
        for gap in unstated_period_reductions(unmarked)
    ]


def plan_period_reduction(reductions: Sequence[Mapping[str, Any]]) -> PeriodReduction:
    """What a plan holding `reductions` (`period_reductions`) does with its years:
    unstated when the rule refuses any, else across years when it holds one."""
    if any(reduction["refused"] for reduction in reductions):
        return PeriodReduction.UNSTATED
    return PeriodReduction.ACROSS_YEARS if reductions else PeriodReduction.NONE


def record_period_reduction(record: Mapping[str, Any], task: Task) -> PeriodReduction | None:
    """What the accepted plan of `record`, a run of `task`, does with its years, or
    `None` when it accepted none or its plan fails to validate.

    A record without `period_reduction` holds a plan with no ask
    marked across years, since the revision that wrote it had no mark. Its plan is read with the
    asks `task` marks, the marks the orchestrator hands a run of the current revision,
    so a baseline stored from that revision is scored on the same asks as the arm.
    A stored plan that `Plan` refuses to validate is not read; `summarise` counts it
    as an accepted plan that misses its task's years.
    """
    if (kept := record.get("period_reduction")) is not None:
        return PeriodReduction(kept)
    if record.get("plan") is None:
        return None
    try:
        plan = Plan.model_validate(
            {**record["plan"], "asks_across_years": task.across_years_numbers}
        )
    except ValidationError:
        return None
    return plan_period_reduction(period_reductions(plan))


def stop_reason(run: EntryRun, calls: Sequence[Mapping[str, Any]]) -> str:
    """How the entry ended: `accepted`, `asked_user` (an ask the tool took, still awaiting
    the user's answer), `timeout` (the eval's own limit), `error`, or the seam's refusal
    code. An ask the tool rejected, or one the user answered, carries a response."""
    if run.probe.plan is not None:
        return "accepted"
    if run.timed_out:
        return "timeout"
    if run.error is not None:
        return "error"
    if any(
        call["name"] == REQUEST_INPUT_FUNCTION_CALL_NAME and call["response"] is None
        for call in calls
    ):
        return "asked_user"
    return run.probe.refused_code or "no_plan"


def entry_record(
    *,
    task: Task,
    run: EntryRun,
    arm: str,
    rep: int,
    model: str,
    revision: str,
    experiment: str,
) -> dict[str, Any]:
    """One run's JSON record: what the planner did, what it cost, and how it ended.

    `experiment` identifies the treatment the run had (`run.experiment`), so a resumed
    run directory never mixes two. `period_reductions` and `period_reduction` say
    what the accepted plan does with its years.
    """
    calls = tool_calls(run.events)
    proposed = proposals(calls, recorded=run.probe.plan_call)
    plan = run.probe.plan
    reductions = period_reductions(plan) if plan else None
    return {
        "schema": RECORD_SCHEMA,
        "arm": arm,
        "task": task.id,
        "rep": rep,
        "model": model,
        "revision": revision,
        "experiment": experiment,
        "inputs": run.inputs,
        "status": "error" if run.error else "ok",
        "error": run.error,
        "stop_reason": stop_reason(run, calls),
        "accepted": plan is not None,
        "plan_kinds": sorted({r.kind for r in plan.evidence_requirements}) if plan else None,
        "plan": plan.model_dump(mode="json", exclude={"translation", "workflow"}) if plan else None,
        "period_reductions": reductions,
        "period_reduction": (
            None if reductions is None else plan_period_reduction(reductions).value
        ),
        "tool_calls": [call["name"] for call in calls],
        "proposals": proposed,
        "first_proposal": (
            {key: proposed[0][key] for key in ("outcome", "code")} if proposed else None
        ),
        "calls": run.probe.calls,
        "budget_closed": run.budget_closed,
        "surface_misses": run.surface_misses,
        "wall_s": round(run.wall_s, 3),
    }


__all__ = [
    "KINDS",
    "PERIOD_REFUSAL",
    "EntryRun",
    "PeriodReduction",
    "Probe",
    "RecordedResolver",
    "RecordedSurface",
    "RecordingMismatchError",
    "Task",
    "TaskStateError",
    "answer_id",
    "build_seam",
    "entry_record",
    "first_year_gaps",
    "inputs_digest",
    "load_recording",
    "load_tasks",
    "orchestrator_state",
    "period_reductions",
    "plan_period_reduction",
    "quiet_logs",
    "record_period_reduction",
    "run_entry",
    "save_recording",
    "search_context_meta",
    "stray_recordings",
    "surface_key",
]
