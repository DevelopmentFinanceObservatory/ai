"""One planner entry with the planner on Pydantic AI, run as `evals/planner_entry/entry.py`
runs the ADK planner, and its record in that eval's own shape.

The entry is the eval's, step for step: the orchestrator's recorded `search` calls and
`plan` hand-off seed the state (`entry.orchestrator_state`), the graph's own `route`,
`plan_start` and `planner` node functions run over it, and the `planner` node calls the
real `PlannerSeam`, wrapped as the eval wraps it (`entry.Probe.watch`,
`harness.wiring.handing_back_planner_failures`). The seam's one ADK call,
`ctx.run_node`, runs the Pydantic AI agent instead (`PaiRunner`). Each request runs
under its own `TurnBudget`, an ask the tool takes pauses the entry, and the eval's
translate step (`entry.Probe.translate`) ends it at the first recorded plan.

    pai_record(task, recording, arm="pai-like-adk", mode="like-adk", prefix_cache=cache)
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import sys
import time
import traceback
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
EVAL_DIR = HERE.parents[2] / "evals" / "planner_entry"
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

os.environ.setdefault("PYDANTIC_AI_NO_BANNER", "1")

import pydantic_ai  # noqa: E402
import run as eval_run  # noqa: E402
from entry import (  # noqa: E402
    EntryRun,
    Probe,
    RecordedResolver,
    RecordedSurface,
    RecordingMismatchError,
    Task,
    entry_record,
    inputs_digest,
    orchestrator_state,
)
from google.adk import Event  # noqa: E402
from google.genai import Client, types  # noqa: E402
from harness.graph.orchestrate import route  # noqa: E402
from harness.graph.planning import plan_start, planner_node  # noqa: E402
from harness.planner_seam import PlannerSeam  # noqa: E402
from harness.prefix_cache import PrefixCache  # noqa: E402
from harness.settings import Settings  # noqa: E402
from harness.turn_budget import TurnBudget, current_turn_budget  # noqa: E402
from harness.wiring import handing_back_planner_failures  # noqa: E402
from pai_planner import (  # noqa: E402
    CallHooks,
    Mode,
    PlannerClient,
    PlannerDeps,
    PlannerModel,
    PlannerParts,
    adk_events,
    genai_client,
    google_model,
    planner_agent,
)
from planner.agent import AGENT_NAME  # noqa: E402
from planner.settings import AgentSettings  # noqa: E402
from pydantic_ai import Agent  # noqa: E402
from pydantic_ai.messages import ModelMessage, ModelRequest, UserPromptPart  # noqa: E402
from pydantic_ai.tools import DeferredToolRequests, DeferredToolResults  # noqa: E402
from pydantic_ai.usage import UsageLimits  # noqa: E402

# Pydantic AI merges consecutive requests into one before sending them
# (`pydantic_ai.messages._merge_consecutive_messages`), which would send the user's
# message and the seam's brief as one content where ADK sends two. It leaves apart two
# requests whose `instructions` both are set and differ, and the agent's own
# instruction is the one sent, so the user's message carries this note as its own.
QUESTION_INSTRUCTIONS = "The user's message, sent to the orchestrator before the planner ran."

# Pydantic AI caps a run at 50 requests by default; the turn budget's 30-call cap is
# the bound here, as in production.
_NO_REQUEST_LIMIT = UsageLimits(request_limit=None)


class PausedForUser(BaseException):
    """The planner's ask was taken and the entry waits on the user, as ADK's
    `NodeInterruptedError` pauses the planner node. A `BaseException`, as that is, so
    nothing between the run and the entry's driver mistakes it for a failure."""


@dataclass
class PaiRunner:
    """The Pydantic AI runs of one entry, sharing one message history: the user's
    message first, as ADK's session holds it, then each message the seam sends."""

    agent: Agent[PlannerDeps, Any]
    deps: PlannerDeps
    messages: list[ModelMessage]
    # The ask the entry paused on, and the user's answer to resume it with.
    pending_ask: str | None = None
    answer: Mapping[str, Any] | None = None

    async def run(self, content: types.Content) -> None:
        """Run the planner on the seam's message `content`, as `ctx.run_node` runs it.

        Every model call streams. A tool that sets `skip_summarization` ends the run
        before the next model call, as ADK ends it. An ask the tool takes ends the run
        on `PausedForUser`.
        """
        text = "".join(part.text or "" for part in content.parts or [])
        results = None
        if self.pending_ask is not None and self.answer is not None:
            results = DeferredToolResults(calls={self.pending_ask: dict(self.answer)})
            self.pending_ask = self.answer = None
        self.deps.skip_summarization = False
        async with self.agent.iter(
            text or None,
            message_history=self.messages,
            deferred_tool_results=results,
            deps=self.deps,
            usage_limits=_NO_REQUEST_LIMIT,
        ) as run:
            async for node in run:
                if not Agent.is_model_request_node(node):
                    continue
                if self.deps.skip_summarization:
                    self.messages = [*run.all_messages(), node.request]
                    return
                async with node.stream(run.ctx) as stream:
                    async for _event in stream:
                        pass
            self.messages = run.all_messages()
            output = run.result.output if run.result is not None else None
        if isinstance(output, DeferredToolRequests) and output.calls:
            self.pending_ask = output.calls[-1].tool_call_id
            raise PausedForUser


@dataclass
class _Session:
    """The session `PlannerSeam` reads through `ctx.session`: the live state and the
    events the runs so far have left."""

    runner: PaiRunner
    id: str = "pai-planner-entry"

    @property
    def state(self) -> dict[str, Any]:
        return self.runner.deps.state

    @property
    def events(self) -> list[Event]:
        return adk_events(self.runner.messages)


@dataclass
class PaiNodeContext:
    """The ADK `Context` the graph's planner nodes and `PlannerSeam` read, over one
    plain state dict. `run_node` runs the Pydantic AI planner. ADK's `ctx.state`
    records a delta that later lands on the same session state; here both are one dict,
    so every write is visible at once, as the seam's own writes through
    `ctx.session.state` are."""

    runner: PaiRunner
    resume_inputs: dict[str, Any] | None = None
    session: _Session = field(init=False)

    def __post_init__(self) -> None:
        self.session = _Session(self.runner)

    @property
    def state(self) -> dict[str, Any]:
        return self.runner.deps.state

    async def run_node(
        self, agent: Any, node_input: types.Content, *, override_branch: str | None = None
    ) -> None:
        del agent, override_branch
        await self.runner.run(node_input)


async def _drain(events: AsyncIterator[Any] | Any) -> list[Any]:
    if hasattr(events, "__aiter__"):
        return [event async for event in events]
    return list(events)


@dataclass
class PaiEntry:
    """What one entry built, kept for checks that read past its record."""

    run: EntryRun
    messages: list[ModelMessage]
    model: PlannerModel


async def run_pai_entry(
    task: Task,
    recording: Mapping[str, Any],
    *,
    mode: Mode,
    prefix_cache: PrefixCache | None,
    timeout_s: float = 300.0,
    client: Client | None = None,
    answers: Sequence[Mapping[str, Any]] = (),
    keep_thoughts: bool = False,
) -> PaiEntry:
    """Run `task`'s planner entry once with the planner on Pydantic AI, as
    `entry.run_entry` runs it with the planner on ADK.

    `client` stands in for the google-genai client, as `validate.py`'s scripted model
    stands in for the planner's. `answers` are the user's replies to the entry's asks, in
    order, each resuming it in a request of its own with a fresh budget.
    `keep_thoughts` replays thought summaries as Pydantic AI does by default.
    """
    probe = Probe()
    surface = RecordedSurface(recording.get("tools") or {})
    resolver = RecordedResolver(recording["searches"])
    state = await orchestrator_state(task, resolver)
    if resolver.left():
        raise RecordingMismatchError(f"{task.id}: {resolver.left()} recorded searches went unasked")

    settings = Settings()
    model_name = AgentSettings().planner_model_name
    seam = PlannerSeam(settings=settings, surface=surface, translate=probe.translate)
    parts = PlannerParts.of(seam)
    hooks = CallHooks(seam=seam, before=probe.before_model, after=probe.after_model)
    planner_client = PlannerClient(client or genai_client(model_name), hooks)
    model = PlannerModel(
        google_model(model_name, planner_client.client, mode=mode),
        client=planner_client,
        prefix_cache=prefix_cache,
        mode=mode,
    )
    agent = planner_agent(seam=seam, model=model, parts=parts, keep_thoughts=keep_thoughts)
    runner = PaiRunner(
        agent=agent,
        deps=PlannerDeps(state=state, session_id="pai-planner-entry"),
        messages=[
            ModelRequest(
                parts=[UserPromptPart(content=task.question)], instructions=QUESTION_INSTRUCTIONS
            )
        ],
    )
    ctx = PaiNodeContext(runner)
    draft = planner_node(planner=handing_back_planner_failures(probe.watch(seam)))
    pending = list(answers)
    budgets: list[TurnBudget] = []
    error: str | None = None
    timed_out = False

    async def request() -> bool:
        """One request's nodes; whether it paused on an ask."""
        budgets.append(TurnBudget.from_settings(settings))
        token = current_turn_budget.set(budgets[-1])
        try:
            if ctx.resume_inputs is None:
                routed = await _drain(route(ctx, None))
                await _drain(plan_start(ctx, routed[-1].output))
            await _drain(draft(ctx, {}))
        except PausedForUser:
            return True
        finally:
            current_turn_budget.reset(token)
        return False

    async def drive() -> None:
        while await request() and probe.plan is None and pending:
            runner.answer = pending.pop(0)
            ctx.resume_inputs = {"adk_request_input": runner.answer}

    started = time.perf_counter()
    limit = asyncio.timeout(timeout_s)
    try:
        async with limit:
            await drive()
    except Exception as exc:
        timed_out = limit.expired()
        if not timed_out:
            error = f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=8)}"[:3000]
    wall_s = time.perf_counter() - started
    run = EntryRun(
        events=[
            event
            for event in adk_events(runner.messages)
            if event.author == AGENT_NAME
            or (event.author == "user" and event.get_function_responses())
        ],
        state=dict(state),
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
    return PaiEntry(run=run, messages=runner.messages, model=model)


# The files whose code a PAI record runs on besides the eval's, folded into its
# experiment fingerprint.
_CODE = ("pai_planner.py", "pai_entry.py")


def experiment(arm: str, *, mode: Mode, model: str, revision: str, run_timeout: float) -> str:
    """`run.experiment` for `arm`, with the mode, the Pydantic AI version and this
    directory's code folded in, so one arm's records never mix two treatments."""
    digest = hashlib.sha256(
        eval_run.experiment(arm, model=model, revision=revision, run_timeout=run_timeout).encode()
    )
    digest.update(f"{mode} pydantic-ai {pydantic_ai.__version__}".encode())
    for name in _CODE:
        digest.update((HERE / name).read_bytes())
    return digest.hexdigest()[:16]


async def pai_record(
    task: Task,
    recording: Mapping[str, Any],
    *,
    arm: str,
    mode: Mode,
    prefix_cache: PrefixCache,
    run_timeout_s: float = 300.0,
    rep: int = 1,
) -> dict[str, Any]:
    """One live run's record, with the keys and meaning `entry.entry_record` gives an
    ADK run, so `summarise.py` reads it unchanged. `rep` is the repetition it records."""
    model = AgentSettings().planner_model_name
    revision = eval_run.revision()
    entry = await run_pai_entry(
        task, recording, mode=mode, prefix_cache=prefix_cache, timeout_s=run_timeout_s
    )
    return entry_record(
        task=task,
        run=entry.run,
        arm=arm,
        rep=rep,
        model=model,
        revision=revision,
        experiment=experiment(
            arm, mode=mode, model=model, revision=revision, run_timeout=run_timeout_s
        ),
    )


__all__ = ["PaiEntry", "PausedForUser", "experiment", "pai_record", "run_pai_entry"]
