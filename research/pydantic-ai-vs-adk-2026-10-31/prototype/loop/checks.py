"""The five checks. Each raises when the property fails and takes the switches that
build its deliberately broken variant, so the tests can show it fails."""

import asyncio
import json
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
from agents import Deps, orchestrator, planner
from loop import Cursor, Frame, Loop, Outcome, RefusedError, State
from procs import run_driver
from pydantic_ai import Conversation, ModelMessagesTypeAdapter
from reference import run_reference
from store import Store

QUESTION = "ODA to Ghana"
SCOPE = "Ghana only"
APPROVE = {"kind": "approve", "revision": 1}


def make_loop(
    db: Path,
    *,
    deps: Deps | None = None,
    store: Store | None = None,
    **switches: Any,
) -> Loop:
    return Loop(store or Store(db), orchestrator, planner, deps or Deps(), **switches)


async def collect(frames: AsyncIterator[Frame]) -> list[Frame]:
    return [frame async for frame in frames]


def last_of(frames: list[Frame], kind: str) -> Frame:
    return next(frame for frame in reversed(frames) if frame["type"] == kind)


# History comparison ------------------------------------------------------------


def _strip(value: Any, drop: set[str]) -> Any:
    if isinstance(value, dict):
        return {k: _strip(v, drop) for k, v in value.items() if k not in drop}
    if isinstance(value, list):
        return [_strip(v, drop) for v in value]
    return value


def history(conversation: Conversation, *, ordinal_run_ids: bool = True) -> str:
    """The messages as canonical JSON without timestamps.

    `run_id` is a fresh UUID per run, so it is replaced by the order in which each id
    first appears. That keeps which messages share a run and hides only the value.
    """
    messages = json.loads(ModelMessagesTypeAdapter.dump_json(conversation.messages))
    messages = _strip(messages, {"timestamp"})
    if ordinal_run_ids:
        seen: dict[str, int] = {}
        for message in messages:
            message["run_id"] = seen.setdefault(message["run_id"], len(seen))
    return json.dumps(messages, sort_keys=True)


def differing_keys(a: Any, b: Any) -> set[str]:
    """Names of the fields whose values differ between two parsed histories."""
    if isinstance(a, dict) and isinstance(b, dict):
        keys: set[str] = set()
        for key in a.keys() | b.keys():
            if a.get(key) == b.get(key):
                continue
            inner = differing_keys(a.get(key), b.get(key))
            keys |= inner or {key}
        return keys
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return {"<length>"}
        return set().union(*(differing_keys(x, y) for x, y in zip(a, b, strict=True)))
    return set()


# L1 ---------------------------------------------------------------------------


async def check_live_progress(db: Path, *, buffer_planner_events: bool = False) -> None:
    """The first progress frame reaches the client while `propose_plan` still runs."""
    gate = asyncio.Event()
    loop = make_loop(
        db,
        deps=Deps(gate=gate, gate_call=1),
        buffer_planner_events=buffer_planner_events,
    )
    frames = loop.start_turn("l1", QUESTION)

    async def until_first_progress() -> None:
        async for frame in frames:
            if frame["type"] == "activity.step":
                return

    try:
        # propose_plan is parked on the gate, so a frame that arrives now was sent live
        await asyncio.wait_for(until_first_progress(), timeout=2)
    finally:
        gate.set()
        await frames.aclose()


# L2 ---------------------------------------------------------------------------


async def check_pause_loses_nothing(db: Path, *, drain_before_pause: bool = True) -> None:
    """All K progress frames precede the pause frame, and the table matches the client.

    The client reads slower than the planner emits, so frames queue up behind it.
    """
    progress = 40
    loop = make_loop(
        db, deps=Deps(progress=progress, delay=0), drain_before_pause=drain_before_pause
    )
    client = []
    async for frame in loop.start_turn("l2", QUESTION):
        client.append(frame)
        await asyncio.sleep(0.005)
    steps = [f["done"] for f in client if f["type"] == "activity.step"]
    assert steps == list(range(1, progress + 1)), f"got progress frames {steps}"
    assert client[-1]["type"] == "elicitation.request"
    assert loop.store.frames("l2") == client


# L3 ---------------------------------------------------------------------------


@dataclass
class KillResult:
    plan_hash: str
    differing_keys: set[str] = field(default_factory=set)


async def check_survives_sigkill(
    tmp_path: Path, *, drop_planner_in_killed_process: bool = False
) -> KillResult:
    """A process killed after `plan.proposed` leaves a turn that finishes identically."""
    db = tmp_path / "killed.db"
    started = run_driver(db, "start", "--cid", "l3", "--text", QUESTION)
    assert started.returncode == 0, started.stderr
    asked = last_of(started.frames, "elicitation.request")

    killed = run_driver(
        db,
        "resume",
        "--cid",
        "l3",
        "--interrupt",
        asked["interrupt_id"],
        "--answer",
        json.dumps(SCOPE),
        global_args=(
            ("--linger", "30")
            + (("--drop-agent", "planner") if drop_planner_in_killed_process else ())
        ),
        kill_when=lambda frame: frame["type"] == "plan.proposed",
    )
    assert killed.frames and killed.frames[-1]["type"] == "plan.proposed", killed.stderr
    assert killed.returncode == -9, "the process was not killed while alive"
    proposed = killed.frames[-1]

    finished = run_driver(
        db,
        "resume",
        "--cid",
        "l3",
        "--interrupt",
        proposed["interrupt_id"],
        "--answer",
        json.dumps(APPROVE),
    )
    assert finished.returncode == 0, finished.stderr
    done = finished.frames[-1]
    assert done["type"] == "done"

    # The same turn with the answers resolved inline, once as bare agents and once
    # through the loop in one process.
    ref_orchestrator, ref_planner, ref_hash = await run_reference("l3", QUESTION, SCOPE, Deps())
    inline = make_loop(tmp_path / "inline.db")
    first = await collect(inline.start_turn("l3", QUESTION))
    second = await collect(
        inline.resume("l3", last_of(first, "elicitation.request")["interrupt_id"], SCOPE)
    )
    third = await collect(
        inline.resume("l3", last_of(second, "plan.proposed")["interrupt_id"], APPROVE)
    )

    killed_store = Store(db)
    stored = {
        name: killed_store.require_conversation("l3", name) for name in ("orchestrator", "planner")
    }
    inline_stored = {
        name: inline.store.require_conversation("l3", name) for name in ("orchestrator", "planner")
    }
    reference = {"orchestrator": ref_orchestrator, "planner": ref_planner}

    keys: set[str] = set()
    for name in ("orchestrator", "planner"):
        raw = [
            json.loads(history(c, ordinal_run_ids=False)) for c in (stored[name], reference[name])
        ]
        keys |= differing_keys(*raw)
        assert history(stored[name]) == history(reference[name]), f"{name} differs"
        assert history(stored[name]) == history(inline_stored[name]), f"{name} differs"

    assert done["plan_hash"] == ref_hash == third[-1]["plan_hash"]
    return KillResult(plan_hash=done["plan_hash"], differing_keys=keys)


# L4 ---------------------------------------------------------------------------


async def check_one_resumer_wins(db: Path, *, unconditional_store: bool = False) -> None:
    """Two resumers that both read the pause before either claims it: one runs."""
    first = make_loop(db, store=Store(db, unconditional=unconditional_store))
    asked = await collect(first.start_turn("l4", QUESTION))
    proposed = await collect(
        first.resume("l4", last_of(asked, "elicitation.request")["interrupt_id"], SCOPE)
    )
    interrupt_id = last_of(proposed, "plan.proposed")["interrupt_id"]

    barrier = asyncio.Barrier(2)
    resumers = [
        make_loop(
            db,
            store=Store(db, unconditional=unconditional_store),
            after_load=barrier.wait,
        )
        for _ in range(2)
    ]
    outcomes = await asyncio.gather(
        *(collect(loop.resume("l4", interrupt_id, APPROVE)) for loop in resumers),
        return_exceptions=True,
    )
    ran = [o for o in outcomes if isinstance(o, list)]
    refused = [o for o in outcomes if isinstance(o, RefusedError)]
    assert len(ran) == 1 and len(refused) == 1, f"outcomes: {outcomes}"

    sent = first.store.frames("l4")
    assert [f["type"] for f in sent].count("done") == 1
    # a resume after the winner has finished is refused too
    try:
        await collect(first.resume("l4", interrupt_id, APPROVE))
    except RefusedError:
        return
    raise AssertionError("a late resume of a finished pause was not refused")


# L5 ---------------------------------------------------------------------------


@dataclass
class SeededLoop(Loop):
    """A later deploy that added a state to the opening path, before ORCHESTRATE."""

    seed_runs: int = 0

    def __post_init__(self) -> None:
        super().__post_init__()
        self.states["SEED"] = self._seed

    def opening_state(self) -> str:
        return "SEED"

    async def _seed(self, conversation_id: str, cursor: Cursor, emit: Any) -> Outcome:
        self.seed_runs += 1
        return Outcome(cursor.model_copy(update={"state": State.ORCHESTRATE}))


@dataclass
class ReplayingLoop(SeededLoop):
    """A resume that re-enters the opening path first, as ADK replays from START."""

    async def on_resume(self, conversation_id: str, cursor: Cursor, emit: Any) -> None:
        await self._seed(conversation_id, cursor, emit)


async def check_deploy_safe_resume(
    db: Path, *, resume_loop_cls: type[SeededLoop] = SeededLoop
) -> None:
    """A pause written by the old loop resumes under a loop with a new opening state."""
    old = make_loop(db)
    asked = await collect(old.start_turn("l5", QUESTION))
    proposed = await collect(
        old.resume("l5", last_of(asked, "elicitation.request")["interrupt_id"], SCOPE)
    )

    # Control: the new state does run on the opening path of a fresh turn.
    control = SeededLoop(Store(db.with_name("control.db")), orchestrator, planner, Deps())
    await collect(control.start_turn("l5", QUESTION))
    assert control.seed_runs == 1

    new = resume_loop_cls(Store(db), orchestrator, planner, Deps())
    resumed = await collect(
        new.resume("l5", last_of(proposed, "plan.proposed")["interrupt_id"], APPROVE)
    )
    assert new.seed_runs == 0, "the new opening state ran on resume"
    assert resumed[-1]["type"] == "done"


# Amend and finished approval ----------------------------------------------------------


async def check_amend_and_stale_approval(db: Path) -> None:
    """An amendment gives revision 2 with a new hash; approving revision 1 is refused."""
    loop = make_loop(db)
    asked = await collect(loop.start_turn("am", QUESTION))
    first = last_of(
        await collect(
            loop.resume("am", last_of(asked, "elicitation.request")["interrupt_id"], SCOPE)
        ),
        "plan.proposed",
    )
    amended = last_of(
        await collect(
            loop.resume("am", first["interrupt_id"], {"kind": "amend", "text": "Add a chart"})
        ),
        "plan.proposed",
    )
    assert (first["revision"], amended["revision"]) == (1, 2)
    assert amended["plan_hash"] != first["plan_hash"]
    assert amended["interrupt_id"] != first["interrupt_id"]

    with pytest.raises(RefusedError):
        await collect(loop.resume("am", amended["interrupt_id"], APPROVE))
    finished = await collect(
        loop.resume("am", amended["interrupt_id"], {"kind": "approve", "revision": 2})
    )
    assert finished[-1]["plan_hash"] == amended["plan_hash"]
