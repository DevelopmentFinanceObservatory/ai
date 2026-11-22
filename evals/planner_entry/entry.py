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
- the graph's `plan_start` and `planner` nodes, with the turn's spec blocks in their
  state, take the action the `plan` call recorded, as the `orchestrate` node hands it
  on. They run as the turn loop runs them (`harness.turn.loop`), each node's state and
  input passed on as the turn store hands them back. The planner's ask pauses the
  entry, and the user's answer, recorded in the turn's transcript, resumes it. They
  run inside a conversation that holds those spec blocks as `harness.studio` holds a
  turn's (`planner.plan.current_recorded_identity`), and hand the brief to the real
  `PlannerAdapter`, which renders the planner's first message from that state and runs
  the planner, with production's instruction and tools, each request under the
  `TurnBudget` a served request gets (`harness.turn_budget`).

Two edges are replaced. The planner's `get_skill` reads the recording through
`RecordedSurface`. The adapter's translate step is the eval's own,
which ends the entry once a plan is recorded (`PlanRecorded`), so a run measures the
draft up to its first accepted plan. Given `TranslationCalls`, it first runs
production's translation on that plan (`translator.tool.build_translate`), its calls
to data-mcp and the graph's resolve door answered from the recording; that is how
`validate.py` checks each plan compiles. The planner's model calls go through a
google-genai client the eval wraps (`Probe.client`), where it notes each request and
answer and applies an arm's edits.
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
from collections.abc import (
    AsyncGenerator,
    AsyncIterator,
    Awaitable,
    Callable,
    Collection,
    Mapping,
    Sequence,
)
from contextlib import aclosing
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, Self, cast

from compiler.surface import ToolCallResult, ToolId, ToolSchema, ToolSurfaceUnavailableError
from data_mcp import resolution
from data_mcp.errors import UpstreamError
from data_mcp.graph_wire import NodesResponse
from data_mcp.node_lookup import NodeOutcome
from data_mcp.search_tool import SearchContext
from data_mcp.skills_client import name_get_skill_in_call_lines
from google.genai import Client, types
from google.genai.client import AsyncClient
from harness.gemini_agent import RecordingState
from harness.gemini_wire import model_retry_options
from harness.graph.interfaces import PlannerModel
from harness.graph.planning import plan_start, planner_node
from harness.graph.state import (
    ASK_FOR_QUESTION_KEY,
    NEW_QUESTION_KEY,
    ORCHESTRATOR_PENDING_KEY,
    ORCHESTRATOR_SPECS_KEY,
    pending_action,
)
from harness.orchestrator_tools import (
    AskedOutput,
    Resolver,
    build_orchestrator_tools,
)
from harness.planner_adapter import PlannerAdapter, PlannerAdapterError
from harness.prefix_cache import PrefixCache
from harness.scripted import instruction_text
from harness.settings import Settings
from harness.studio_state import ConversationState
from harness.tool_contracts import resolver_digest
from harness.turn import transcript
from harness.turn.node import Next, NodeCtx, Pause, StopScope, TurnKey
from harness.turn.store import AgentRecord
from harness.turn.transcript import Transcript
from harness.turn_budget import TurnBudget, current_turn_budget
from harness.wiring import handing_back_planner_failures
from orchestrator.call import HANDED_OFF, OrchestratorCall
from planner.call import PlannerCall
from planner.plan import (
    Plan,
    SessionState,
    current_recorded_identity,
    load_plan,
    unstated_period_reductions,
)
from planner.session import function_calls, function_responses
from planner.source_reads import read_full_history
from planner.tool_results import ProblemCode
from planner.tools import load_pending_ask
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator
from submission.service import first_year_reads
from translator.discovery import DefinitionNotResolvedError, graph_describe, graph_resolve
from translator.tool import Translate, build_translate

EVAL_DIR = Path(__file__).resolve().parent
TASKS_DIR = EVAL_DIR / "tasks"
AMENDMENTS_DIR = EVAL_DIR / "amendments"
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

# The principal an entry's turn runs as.
_PRINCIPAL = "eval"


# --- Tasks ---------------------------------------------------------------------------


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Search(_Strict):
    """One `search` call the orchestrator made, with the arguments it sent."""

    parts: list[dict[str, Any]] = []
    text: str | None = None

    @model_validator(mode="after")
    def _says_what_it_searched(self) -> Self:
        if not self.parts and not self.text:
            raise ValueError("a search sends `parts`, `text`, or both")
        return self


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


class Amendment(_Strict):
    """What makes a task an amendment of a plan card: the user's `text` typed on the
    card of `prior_plan`, the plan the planner drafted for task `base` (a `Plan` dump,
    as a run's record keeps it), and `answer`, the user's own words in reply to a
    question the planner asks about it, when the task scripts one. A task asking a
    new question states in `new_question_in` the value, or the list of values, a
    `starts_new_plan` call may name: `amendment` or `answer`, whose text the turn
    hands back (`handed_back`), or `none`, which hands back nothing and asks the user
    for their question. A task that amends the plan states none."""

    base: str
    prior_plan: dict[str, Any]
    text: str
    answer: str | None = None
    new_question_in: str | list[str] | None = None

    @property
    def accepted(self) -> frozenset[str]:
        """The values of `new_question_in` a run may name, empty on a task that
        amends the plan."""
        stated = self.new_question_in
        return frozenset([stated] if isinstance(stated, str) else stated or ())

    def handed_back(self, source: str) -> str | None:
        """The words a turn hands back when the planner names `source`: the text of
        the message it names, `None` for `none`."""
        return {"amendment": self.text, "answer": self.answer}.get(source)

    @model_validator(mode="after")
    def _each_named_message_exists(self) -> Self:
        Plan.model_validate(self.prior_plan)
        if isinstance(self.new_question_in, list) and (
            not self.new_question_in or len(self.accepted) < len(self.new_question_in)
        ):
            raise ValueError("a `new_question_in` list names each value once, and one at least")
        if unknown := self.accepted - {"amendment", "answer", "none"}:
            raise ValueError(f"`new_question_in` names {sorted(unknown)}, not a message")
        if "answer" in self.accepted and self.answer is None:
            raise ValueError("`new_question_in` names `answer` and the task scripts none")
        return self


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
    held to. A task with an `amendment` opens the planner on the user's amendment of
    a plan card instead of a first draft (`load_amendment_tasks`).
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
    amendment: Amendment | None = None

    @property
    def recording_id(self) -> str:
        """The task whose recording a run replays: the base task of an amendment
        task, which shares its searches and skills, else this one."""
        return self.amendment.base if self.amendment is not None else self.id

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


def load_amendment_tasks(
    directory: Path = AMENDMENTS_DIR, tasks: Sequence[Task] | None = None
) -> list[Task]:
    """Every amendment task in `directory`, by id. A file holds `id` and the
    `amendment`; the rest of the task is its base task's (`Amendment.base`), whose
    searches, brief and asks the turn ran before the user amended the card."""
    by_id = {task.id: task for task in (tasks if tasks is not None else load_tasks())}
    loaded = []
    for path in sorted(directory.glob("*.json")):
        raw = json.loads(path.read_text(encoding="utf-8"))
        amendment = Amendment.model_validate(raw["amendment"])
        base = by_id[amendment.base]
        loaded.append(
            base.model_copy(update={"id": raw["id"], "kind": "amendment", "amendment": amendment})
        )
    return loaded


def load_every_task() -> list[Task]:
    """Every first-turn task (`load_tasks`), then every amendment task."""
    first_turns = load_tasks()
    return [*first_turns, *load_amendment_tasks(tasks=first_turns)]


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


async def _live_answer(
    live: LiveCall,
    tool: ToolId,
    arguments: Mapping[str, object],
    *,
    key: str,
    failures: list[str],
) -> dict[str, Any]:
    """`live`'s answer to one call filed under `key`. A call that raised, whose raise
    propagates, and an answer `_keepable` keeps out are listed in `failures`."""
    try:
        answer = await live(tool, arguments)
    except BaseException as exc:
        failures.append(f"{key}: {type(exc).__name__}: {exc}")
        raise
    if not _keepable(answer):
        failures.append(f"{key}: refused for now: {answer['structured_content']}")
    return answer


def _keepable(answer: Mapping[str, Any]) -> bool:
    """Whether a recording keeps `answer`: anything but a refusal to retry later."""
    return not (
        answer["is_error"]
        and (answer["structured_content"] or {}).get("retry_mode") == "retry_after_delay"
    )


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
            raise ToolSurfaceUnavailableError(
                f"the recording holds no answer for {key}", retryable=False
            )
        answer = await _live_answer(self._live, tool, arguments, key=key, failures=self.failures)
        if _keepable(answer):
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
    answers. The question is compared only for a raw-text search, as a parsed
    search is read from its parts and context alone.
    """

    def __init__(self, calls: Sequence[Mapping[str, Any]]) -> None:
        self._calls = list(calls)

    async def __call__(
        self,
        *,
        question: str | None,
        parts: list[dict[str, Any]],
        raw_text: bool = False,
        context: SearchContext | None = None,
    ) -> dict[str, Any]:
        if not self._calls:
            raise RecordingMismatchError(f"no recorded search left for {question or parts!r}")
        call = self._calls.pop(0)
        asked = {
            "parts": parts,
            "raw_text": raw_text,
            "context": search_context_meta(context),
        }
        if raw_text:
            asked["question"] = question
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

    The planner reaches data-mcp for `get_skill` and for a pinned dataset's first year.
    A recording files a skill's first page under the key `get_skill` and each file's
    pages under the key `get_skill_file`, with the call's arguments after it
    (`surface_key`). A `get_skill` call is looked up under whichever of the two its
    `path` selects, and the call lines a recorded page prints, which name
    `get_skill_file`, are rewritten to name `get_skill`. A call the recording lacks is answered as an
    unreachable server is (`ToolSurfaceUnavailableError`), which the planner's tool
    turns into a sentence saying the call did not complete, and is listed in `misses`.

    With `live` (`record.py`), a call the recording lacks is made through `live` instead,
    such as the `resolve_definition` call `propose_plan` makes to admit a `whole_of`,
    and kept in `added` under its key. A live call that raised, or answered a refusal
    to retry later, is kept out and listed in `failures`.
    """

    def __init__(
        self, results: Mapping[str, Mapping[str, Any]], *, live: LiveCall | None = None
    ) -> None:
        self._results = dict(results)
        self._live = live
        self.misses: list[str] = []
        self.added: dict[str, dict[str, Any]] = {}
        self.failures: list[str] = []

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
        name, arguments = tool.name, dict(arguments)
        if name == "get_skill" and "path" in arguments:
            name = "get_skill_file"
            arguments.setdefault("offset", 0)
        key = surface_key(name, arguments)
        if key not in self._results and self._live is not None:
            answer = await _live_answer(
                self._live, tool, arguments, key=key, failures=self.failures
            )
            if not _keepable(answer):
                return ToolCallResult.model_validate(answer)
            self._results[key] = self.added[key] = answer
        if key not in self._results:
            self.misses.append(key)
            raise ToolSurfaceUnavailableError(
                f"the recording holds no answer for {key}", retryable=False
            )
        result = ToolCallResult.model_validate(self._results[key])
        if tool.name == "get_skill":
            result = result.model_copy(update={"text": name_get_skill_in_call_lines(result.text)})
        return result


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
    state: dict[str, Any] = {}
    call = OrchestratorCall(state=RecordingState(state), turn_id=task.id)
    for search in task.searches:
        found = await tools["search"](call=call, parts=search.parts or None, text=search.text)
        if "error" in found:
            raise TaskStateError(
                f"{task.id}: search {search.parts or search.text!r}: {found['error']}"
            )
    handed = tools["plan"](brief=task.brief, asks=list(task.asks), call=call)
    if handed.get(HANDED_OFF) != "plan":
        raise TaskStateError(f"{task.id}: plan answered {handed}")
    return state


# --- The planner entry ---------------------------------------------------------------


class PlanRecorded(Exception):  # noqa: N818 -- a stop signal, not a fault
    """Raised by the eval's translate step once `propose_plan` has recorded a plan.

    The adapter answers a planner tool that raises with a final refusal
    (`planner_tool_failed`) and makes no further model call, so the entry ends on the
    first accepted proposal.
    """


class _DropPlanRecorded(logging.Filter):
    """Drops the planner's error log for the eval's own stop, which it reads as a tool
    failure. The adapter logs it."""

    def filter(self, record: logging.LogRecord) -> bool:
        return not (record.exc_info and isinstance(record.exc_info[1], PlanRecorded))


logging.getLogger("harness.planner_adapter").addFilter(_DropPlanRecorded())


def quiet_logs() -> None:
    """Silence what a run's record already holds: the wiring's line for an entry the
    adapter gave up on."""
    logging.getLogger("harness.wiring").setLevel(logging.CRITICAL)


# The stop reason of an entry the adapter gave up on (`PlannerAdapterError`): the planner
# ended its turns with no accepted plan and no ask, and corrections did not change that.
GAVE_UP = "gave_up"


@dataclass
class ModelRequest:
    """One request the planner's model client is sent, in google-genai's types.

    The planner sends the system instruction as a string and every declaration in
    one `Tool` (`harness.gemini_wire.in_canonical_form`). `system_text` and
    `declarations` read any shape google-genai takes, so an edit reads a request
    the same way whichever form it arrives in.
    """

    model: str
    contents: list[types.Content]
    config: types.GenerateContentConfig

    @property
    def system_text(self) -> str:
        """The system instruction's text, whichever shape it was sent in."""
        return instruction_text(self.config.system_instruction)

    @property
    def declarations(self) -> dict[str, types.FunctionDeclaration]:
        """Every function declaration the request sends, by name, whichever `Tool`
        holds it. Each is the request's own object, so an edit may change it in place."""
        return {
            declaration.name: declaration
            for tool in self.config.tools or []
            if isinstance(tool, types.Tool)
            for declaration in tool.function_declarations or []
            if declaration.name is not None
        }


# An arm's change to each request the planner's model client is sent, made in place.
# The system instruction, tools and tool config travel where the request sends them:
# on the request itself, or, on a request sent against the prefix cache
# (`harness.prefix_cache`), on that cache's create, which the edit is handed as a
# request with no contents. The planner sends its requests in one form
# (`ModelRequest`). An edit reads the instruction through `ModelRequest.system_text`
# and finds a declaration through `ModelRequest.declarations`.
RequestEdit = Callable[[ModelRequest], None]


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
    """What one entry did that its transcript does not carry: the plan the first
    accepted proposal recorded and the id of the `propose_plan` call that recorded it,
    the code of the refusal an entry with no plan ended on, each request the planner's
    model client was sent, each model call, with its tokens and latency once the whole
    answer arrived, and what `translator` answered for the plan when one is set.

    The planner reaches the model only through the client `client` returns, so a
    turn the adapter or the turn budget answers in the model's place is no call. A
    call stays open from its first send until a whole answer arrives or the
    entry's next request starts (`next_request`). A send while one is open is that call
    sent again (a first-chunk resend, or a resend after the prefix cache was found
    gone, at open or while its stream was read), so it is one call, timed from its
    first send. A call whose answer never arrived keeps its usage unknown."""

    plan: Plan | None = None
    plan_call: str | None = None
    refused_code: str | None = None
    requests: list[ModelRequest] = field(default_factory=list)
    calls: list[dict[str, Any]] = field(default_factory=list)
    # Production's translation, run on the recorded plan before the entry ends, and
    # what it answered.
    translator: Translate | None = None
    translation: dict[str, object] | None = None
    _started: float = 0.0
    _answering: bool = False

    async def translate(self, call: PlannerCall) -> dict[str, object]:
        self.plan = load_plan(call.state)
        self.plan_call = call.call_id
        if self.translator is not None:
            self.translation = await self.translator(call)
        raise PlanRecorded

    def watch(self, planner: PlannerModel) -> PlannerModel:
        """`planner`, noting the code of each refusal it answers, and `GAVE_UP` when it
        raises `PlannerAdapterError`."""
        return _Watched(planner, self)

    async def note(self, answering: Awaitable[dict[str, Any]]) -> dict[str, Any]:
        """`answering`'s answer, its refusal code noted as `watch` describes."""
        try:
            answer = await answering
        except PlannerAdapterError:
            self.refused_code = GAVE_UP
            raise
        if "refused" in answer:
            code = answer["refused"]["code"]
            self.refused_code = None if code is None else str(code)
        return answer

    def client(self, client: Client, *, edits: Sequence[RequestEdit] = ()) -> Client:
        """`client`, each streamed request it is sent edited by `edits` and noted, and
        each answer's usage and latency noted. `client` itself is left unchanged."""
        return _ProbedClient(client, _ProbedModels(client.aio.models, self, edits), edits)

    def next_request(self) -> None:
        """Start a request of the entry. No call stays open across two requests."""
        self._answering = False

    def _sent(self, request: ModelRequest) -> None:
        self.requests.append(request)
        if self._answering:
            return
        self._answering = True
        self._started = time.perf_counter()
        self.calls.append(dict(_UNFINISHED_CALL))

    def _answered(self, usage: types.GenerateContentResponseUsageMetadata | None) -> None:
        self._answering = False
        if usage is None:
            return
        self.calls[-1] = {
            "prompt_tokens": usage.prompt_token_count or 0,
            "cached_tokens": usage.cached_content_token_count or 0,
            "output_tokens": usage.candidates_token_count or 0,
            "thinking_tokens": usage.thoughts_token_count or 0,
            "latency_s": round(time.perf_counter() - self._started, 3),
        }

    async def _watched(
        self, stream: AsyncIterator[types.GenerateContentResponse]
    ) -> AsyncIterator[types.GenerateContentResponse]:
        """`stream`'s chunks, the last usage any carried noted once it ends. A stream
        that raised or that its reader cut leaves the call open for a resend."""
        usage = None
        async with aclosing(cast(AsyncGenerator[types.GenerateContentResponse], stream)) as chunks:
            async for chunk in chunks:
                usage = chunk.usage_metadata or usage
                yield chunk
        self._answered(usage)


def _edited(
    model: str,
    contents: list[types.Content],
    config: types.GenerateContentConfig,
    edits: Sequence[RequestEdit],
) -> ModelRequest:
    """A copy of the request, with `edits` made to it."""
    request = ModelRequest(
        model=model,
        contents=[content.model_copy(deep=True) for content in contents],
        config=config.model_copy(deep=True),
    )
    for edit in edits:
        edit(request)
    return request


class _ProbedModels:
    """`client.aio.models` with its streamed call edited and noted by a `Probe`."""

    def __init__(self, models: Any, probe: Probe, edits: Sequence[RequestEdit]) -> None:
        self._models = models
        self._probe = probe
        self._edits = tuple(edits)

    async def generate_content_stream(
        self, *, model: str, contents: Any, config: Any = None
    ) -> AsyncIterator[types.GenerateContentResponse]:
        request = _edited(
            model,
            [types.Content.model_validate(content) for content in contents],
            types.GenerateContentConfig.model_validate(config or {}),
            self._edits,
        )
        self._probe._sent(request)
        if self._edits:
            model, contents, config = request.model, request.contents, request.config
        stream = await self._models.generate_content_stream(
            model=model, contents=contents, config=config
        )
        return self._probe._watched(stream)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._models, name)


class _ProbedCaches:
    """`client.aio.caches` with each create's system instruction, tools and tool config
    edited by `edits`."""

    def __init__(self, caches: Any, edits: Sequence[RequestEdit]) -> None:
        self._caches = caches
        self._edits = tuple(edits)

    async def create(self, *, model: str, config: Any = None) -> types.CachedContent:
        if self._edits:
            created = types.CreateCachedContentConfig.model_validate(config or {})
            prefix = types.GenerateContentConfig(
                system_instruction=created.system_instruction,
                tools=created.tools,
                tool_config=created.tool_config,
            )
            request = _edited(model, [], prefix, self._edits)
            model = request.model
            config = created.model_copy(
                update={
                    "system_instruction": request.config.system_instruction,
                    "tools": request.config.tools,
                    "tool_config": request.config.tool_config,
                }
            )
        return await self._caches.create(model=model, config=config)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._caches, name)


class _ProbedAsyncClient:
    """`client.aio` with `_ProbedModels` and `_ProbedCaches` in place of its own."""

    def __init__(
        self, aio: AsyncClient, models: _ProbedModels, edits: Sequence[RequestEdit]
    ) -> None:
        self._aio = aio
        self.models = models
        self.caches = _ProbedCaches(aio.caches, edits)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._aio, name)


class _ProbedClient(Client):
    """`client` with `_ProbedAsyncClient` as its `aio`.

    A `Client`, as `PlannerAdapter` types its `client`, built without
    `Client.__init__`. Every other attribute is `client`'s own, and the wrapped client
    stays open when this one is collected.
    """

    def __init__(self, client: Client, models: _ProbedModels, edits: Sequence[RequestEdit]) -> None:
        self._client = client
        self._probed_aio = _ProbedAsyncClient(client.aio, models, edits)

    @property
    def aio(self) -> AsyncClient:
        return cast(AsyncClient, self._probed_aio)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)

    def __del__(self) -> None:
        pass


class _Watched:
    """The `PlannerModel` `Probe.watch` returns."""

    def __init__(self, planner: PlannerModel, probe: Probe) -> None:
        self._planner = planner
        self._probe = probe

    async def __call__(self, **kwargs: Any) -> dict[str, Any]:
        return await self._probe.note(self._planner(**kwargs))

    async def revise(self, **kwargs: Any) -> dict[str, Any]:
        return await self._probe.note(self._planner.revise(**kwargs))


def live_client() -> Client:
    """The google-genai client a live entry's planner calls go through, built from the
    environment with the HTTP retries the served planner's model makes
    (`harness.gemini_wire.model_retry_options`)."""
    return Client(http_options=types.HttpOptions(retry_options=model_retry_options()))


def build_adapter(
    *, probe: Probe, surface: RecordedSurface, prefix_cache: PrefixCache | None, client: Client
) -> PlannerAdapter:
    """The adapter `harness.wiring.build_planner` builds, with the probe's translate
    step and the planner's model calls going through `client`.

    The planner, its instruction and its tools are production's own.
    """
    return PlannerAdapter(
        settings=Settings(),
        surface=surface,
        prefix_cache=prefix_cache,
        translate=probe.translate,
        client=client,
    )


@dataclass(frozen=True)
class EntryRun:
    """One finished entry: its turn's transcript (the user's message, the planner's
    records, each pause and the user's answer to it), what the probe saw, and how it
    ended."""

    records: list[AgentRecord]
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
    client: Client | None = None,
    edits: Sequence[RequestEdit] = (),
    timeout_s: float = 300.0,
    answers: Sequence[Mapping[str, Any]] = (),
    translation: TranslationCalls | None = None,
    surface: RecordedSurface | None = None,
) -> EntryRun:
    """Run `task`'s planner entry once on a fresh turn.

    The planner's model calls go through `client`, a live client (`live_client`) when
    None. `edits` change each request it is sent (`RequestEdit`), and the probe notes
    each one and its answer. The entry ends at the first plan `propose_plan` records,
    at an ask of the user, or wherever the adapter ends it (a bound of its request's
    budget, its round budget, a correction it gave up on), or after `timeout_s`. The
    planner streams, as a studio turn runs it. `answers` are the user's replies to
    the entry's asks, in order. Each resumes the paused entry, as the studio's
    interrupt route does, in a request of its own with a fresh budget. A live run
    passes none, so an ask ends it. Each answer is the reply the studio's interrupt
    route reads, the user's answer under `result`. With `translation`,
    production's translation runs on the recorded plan over those calls, and
    `probe.translation` holds its answer. The planner's tool calls, `propose_plan`'s
    own included, go to `surface`, by default one over the recording's `tools`.

    A task with an `amendment` opens the planner as the `approve` node's `amend` edge
    does, on the task's prior plan and the user's text, after `plan_start` took the
    hand-off; its scripted `answer`, when it has one and `answers` is empty, replies to
    the planner's first question. The entry also ends when the planner reads the text
    as a new question, the `new_question` edge, which leaves the words it hands back
    under `NEW_QUESTION_KEY`, or sets `ASK_FOR_QUESTION_KEY` when the planner names
    no message.
    """
    probe = Probe()
    if translation is not None:
        probe.translator = build_translate(
            surface=translation,
            resolve_entities=translation.resolve_entities,
            anchor_read=_no_anchor_read,
            describe_nodes=translation.describe_nodes,
        )
    if surface is None:
        surface = RecordedSurface(recording.get("tools") or {})
    resolver = RecordedResolver(recording["searches"])
    state = await orchestrator_state(task, resolver)
    if resolver.left():
        raise RecordingMismatchError(f"{task.id}: {resolver.left()} recorded searches went unasked")
    adapter = build_adapter(
        probe=probe,
        surface=surface,
        prefix_cache=prefix_cache,
        client=probe.client(client if client is not None else live_client(), edits=edits),
    )
    # The action `plan` recorded, which `orchestrate` takes from the state as the
    # `plan_start` node's input.
    action = pending_action(state)
    state.pop(ORCHESTRATOR_PENDING_KEY, None)
    if action is None or action.kind != "plan":
        raise TaskStateError(f"{task.id}: plan recorded {action}")
    # The adapter wrapped as `harness.server.create_app` wraps it.
    planner = handing_back_planner_failures(probe.watch(adapter))
    ctx = NodeCtx[Any](
        key=TurnKey(_PRINCIPAL, f"eval-{task.id}", 1),
        inputs={"message": task.question},
        state={},
        transcript=Transcript([transcript.user_message(task.question)]),
        emit=lambda event: None,
        stop=StopScope(),
    )
    ctx.state.update(state)
    # The conversation a studio turn runs in, holding the spec blocks the orchestrator's
    # searches returned, as `harness.studio` keeps them, so `propose_plan` and the
    # translation read a pin's identity on record (`planner.plan.recorded_identity`).
    conversation = ConversationState(conversation_id=ctx.key.conversation_id)
    conversation.turn_specs = dict(ctx.state.get(ORCHESTRATOR_SPECS_KEY) or {})
    pending = list(answers)
    amendment = task.amendment
    if amendment is not None and amendment.answer is not None and not pending:
        pending = [{"result": {"answers": {"q1": {"kind": "text", "text": amendment.answer}}}}]
    error: str | None = None
    timed_out = False
    started = time.perf_counter()

    settings = Settings()
    budgets: list[TurnBudget] = []

    async def segment(step: Awaitable[Next | Pause]) -> Next | Pause:
        """One request of the entry: `step` awaited under a fresh budget, then the state
        stored as the turn loop stores it at the node's boundary (`_stored`). A step that
        raises leaves the state as it was."""
        budgets.append(TurnBudget.from_settings(settings))
        probe.next_request()
        token = current_turn_budget.set(budgets[-1])
        identities = current_recorded_identity.set(conversation.recorded_identity)
        try:
            result = await step
        finally:
            current_recorded_identity.reset(identities)
            current_turn_budget.reset(token)
        ctx.state = _stored(ctx.state)
        return result

    async def drive() -> None:
        drafting = planner_node(planner=planner)
        opened = await plan_start.run(ctx, action.model_dump(mode="json"))
        assert isinstance(opened, Next)
        # The planner node runs on its input as the loop stored it with the cursor.
        ctx.state = _stored(ctx.state)
        node_input = _stored(opened.output)
        if amendment is not None:
            # The card the user amended holds the prior plan, which the `approve`
            # node hands the planner with the answer on its `amend` edge.
            prior = Plan.model_validate(amendment.prior_plan)
            ctx.state["plan"] = prior.model_dump_json()
            node_input = _stored(
                {
                    "plan": prior.model_dump(mode="json"),
                    "answer": {"approved": None, "amendment": amendment.text},
                }
            )
        result = await segment(drafting.run(ctx, node_input))
        rounds = 0
        while isinstance(result, Pause) and probe.plan is None and pending:
            rounds += 1
            interrupt_id = f"int_{rounds}"
            answer = dict(pending.pop(0)["result"])
            ctx.transcript.append(
                transcript.pause(
                    kind=result.kind,
                    interrupt_id=interrupt_id,
                    round=rounds,
                    payload=result.payload,
                    message=result.message,
                )
            )
            ctx.transcript.append(
                transcript.answer(interrupt_id=interrupt_id, round=rounds, value=answer)
            )
            assert drafting.resume is not None
            result = await segment(drafting.resume(ctx, node_input, answer))

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
    return EntryRun(
        records=list(ctx.transcript.records),
        state=dict(ctx.state),
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


def _stored(value: Any) -> Any:
    """`value` as the turn store hands it back: through JSON that holds no NaN or
    infinity (`harness.turn.store`), raising `ValueError` or `TypeError` where the
    store's write would."""
    return json.loads(json.dumps(value, allow_nan=False))


def inputs_digest(task: Task, recording: Mapping[str, Any]) -> str:
    """A digest of what a run of `task` shows the planner: the question, the searches,
    the brief and asks, and the recorded search envelopes and tool answers. Runs pair
    only where their digests agree. An ask not marked `across_years` digests without
    that field, so a baseline stored before asks carried the mark still pairs on it. The
    searches are digested as `Search` models, so a baseline stored before they took `parts`
    and `text` pairs with no task."""
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
        **(
            {"amendment": task.amendment.model_dump(mode="json")}
            if task.amendment is not None
            else {}
        ),
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


# --- The record of one run -----------------------------------------------------------

# The value of the record field and fingerprint key `planner_runner`, which stored
# baselines carry.
PLANNER_RUNNER = "pydantic_ai"
RECORD_SCHEMA = 4


def tool_calls(records: Sequence[AgentRecord]) -> list[dict[str, Any]]:
    """Every tool call the planner made, in order, with the response its tool gave, read
    off the planner's records in the turn's transcript by `planner.session`'s own
    readers. A call with no response is one the entry ended before it answered."""
    calls: list[dict[str, Any]] = []
    by_id: dict[str, dict[str, Any]] = {}
    for record in records:
        for call in function_calls(record):
            entry = {
                "id": call.get("id"),
                "name": call["name"],
                "args": dict(call.get("args") or {}),
                "response": None,
            }
            calls.append(entry)
            if entry["id"] is not None:
                by_id[entry["id"]] = entry
        for response in function_responses(record):
            answered = by_id.get(response.get("id"))
            if answered is not None and answered["response"] is None:
                answered["response"] = response.get("response")
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

    The call `recorded`, which recorded the entry's plan, answers with the adapter's
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


def stop_reason(run: EntryRun) -> str:
    """How the entry ended: `error`, `accepted`, `timeout` (the eval's own limit), the
    adapter's refusal code, `new_question` (the planner read an amendment as a new
    question, which the turn hands back), or `asked_user` (an ask the tool took, which
    the entry paused on, awaiting the user's answer). An error wins over a recorded
    plan: a step that raised, or a state the turn store refused at the node's
    boundary, fails the served turn, so the plan never reaches the user. A refusal
    wins over an ask, since the entry it ends pauses on nothing."""
    if run.error is not None:
        return "error"
    if run.probe.plan is not None:
        return "accepted"
    if run.timed_out:
        return "timeout"
    return _unplanned_ending(run)


def _unplanned_ending(run: EntryRun) -> str:
    """How an entry that recorded no plan, raised nothing and did not time out ended
    (`stop_reason`)."""
    if run.state.get(NEW_QUESTION_KEY) or run.state.get(ASK_FOR_QUESTION_KEY):
        return "new_question"
    if run.probe.refused_code is not None:
        return run.probe.refused_code
    if load_pending_ask(run.state) is not None:
        return "asked_user"
    return "no_plan"


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
    run directory never mixes two, and `planner_runner` holds `PLANNER_RUNNER`.
    `period_reductions` and `period_reduction` say what the accepted plan does with
    its years.
    """
    calls = tool_calls(run.records)
    proposed = proposals(calls, recorded=run.probe.plan_call)
    plan = run.probe.plan
    reductions = period_reductions(plan) if plan else None
    return {
        "schema": RECORD_SCHEMA,
        "arm": arm,
        "task": task.id,
        "rep": rep,
        "model": model,
        "planner_runner": PLANNER_RUNNER,
        "revision": revision,
        "experiment": experiment,
        "inputs": run.inputs,
        "status": "error" if run.error else "ok",
        "error": run.error,
        "stop_reason": stop_reason(run),
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
        "new_question_in": named_new_question(calls),
        "new_question": run.state.get(NEW_QUESTION_KEY),
        "asked_for_question": run.state.get(ASK_FOR_QUESTION_KEY) is True,
    }


def named_new_question(calls: Sequence[Mapping[str, Any]]) -> str | None:
    """The message the run's last `starts_new_plan` call named in `new_question_in`,
    `None` when none named one."""
    named = [
        call["args"].get("new_question_in")
        for call in calls
        if call["name"] == "propose_plan" and call["args"].get("starts_new_plan")
    ]
    return next((value for value in reversed(named) if value), None)


__all__ = [
    "AMENDMENTS_DIR",
    "KINDS",
    "PERIOD_REFUSAL",
    "EntryRun",
    "ModelRequest",
    "PeriodReduction",
    "Probe",
    "RecordedResolver",
    "RecordedSurface",
    "RecordingMismatchError",
    "RequestEdit",
    "Task",
    "TaskStateError",
    "answer_id",
    "build_adapter",
    "entry_record",
    "first_year_gaps",
    "inputs_digest",
    "live_client",
    "load_amendment_tasks",
    "load_every_task",
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
