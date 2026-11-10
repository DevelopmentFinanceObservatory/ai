"""The planner node on the production session service: ADK's `DatabaseSessionService` over a
real Postgres, through the Studio routes, on dev's ADK planner (`adk`) and on the Pydantic AI
planner (`pai`).

A throwaway Postgres runs in Docker for the test session (`pg.py`). Each test makes a database
of its own. A pass line that needs a second process runs its segments as pytest children
(`segment_child.py`) that share only the database. Each has a control built to fail it.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pai_node
import pg
import pytest
import scripted as sc
from entry import quiet_logs
from google.adk.events import Event
from google.adk.sessions import DatabaseSessionService, InMemorySessionService
from harness.graph.state import AGENT_NAME as GRAPH_NAME
from harness.graph.state import ORCHESTRATOR_PRIOR_LEDGER_KEY
from harness.studio_state import SEGMENT_TTL_S
from parity import WIRE, shape
from planner.agent import AGENT_NAME as PLANNER
from planner.agent import _session_steps
from planner.call import CallSession, ResponseCall, Step, prices_chosen
from rig import (
    CONVERSATION,
    SCRIPTS,
    WriteLog,
    build,
    inspect,
    planner_trace,
    planner_window,
    run_actions,
)
from scenes import differences, events, stable

quiet_logs()

HERE = Path(__file__).resolve().parent
ARMS = ["adk", "pai"]
TURN_1 = f"{GRAPH_NAME}/{CONVERSATION}#t1"
TURN_2 = f"{GRAPH_NAME}/{CONVERSATION}#t2"
LEDGER = f"{GRAPH_NAME}.ledger/{CONVERSATION}"
TOLERANCE = 0.05


def report(title: str, data: Any) -> None:
    print(f"\nREPORT {title}: {json.dumps(data, sort_keys=True, default=str)}")


# --- running a share of a conversation in a process of its own ------------------------------


@dataclass
class Child:
    process: subprocess.Popen[str]
    out: Path
    block: Path | None

    def finish(self, timeout: float = 300) -> dict[str, Any]:
        output, _ = self.process.communicate(timeout=timeout)
        assert self.process.returncode == 0, output[-4000:]
        return json.loads(self.out.read_text())


_counter = iter(range(10_000))


def start_child(url: str, tmp_path: Path, spec: dict[str, Any]) -> Child:
    index = next(_counter)
    out = tmp_path / f"out-{index}.json"
    block = tmp_path / f"blocked-{index}" if spec.pop("block", False) else None
    spec_file = tmp_path / f"spec-{index}.json"
    spec_file.write_text(json.dumps({**spec, "out": str(out), "block": str(block or "") or None}))
    env = {
        **os.environ,
        "SEGMENT_SPEC": str(spec_file),
        "SESSION_DB_URL": url,
        "PYDANTIC_AI_NO_BANNER": "1",
    }
    process = subprocess.Popen(
        [sys.executable, "-m", "pytest", "segment_child.py", "-q", "-p", "no:cacheprovider"],
        cwd=HERE,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return Child(process, out, block)


def run_child(url: str, tmp_path: Path, **spec: Any) -> dict[str, Any]:
    """One process: `spec` names the arm, the script, the planner answers it gets
    (`first`, `last`), and the actions to post. Returns what they answered."""
    return start_child(url, tmp_path, spec).finish()


def wait_blocked(child: Child, timeout: float = 120) -> int:
    """The pid of a child once it is inside the translation of a `propose_plan`."""
    assert child.block is not None
    deadline = time.monotonic() + timeout
    while not child.block.exists():
        assert child.process.poll() is None, "the child ended before it reached the translation"
        assert time.monotonic() < deadline, "the child never reached the translation"
        time.sleep(0.1)
    return int(child.block.read_text())


def kill(child: Child) -> None:
    child.process.send_signal(signal.SIGKILL)
    child.process.communicate(timeout=30)
    assert child.process.returncode == -signal.SIGKILL


TURN = {"kind": "turn"}


def answer(name: str, interrupt_id: str | None = None) -> dict[str, Any]:
    return {"kind": "answer", "answers": name, "interrupt_id": interrupt_id}


def pause_id(result: dict[str, Any], segment: int = 0) -> str:
    (frame,) = [
        f for f in result["actions"][segment]["frames"] if f["event"] == "elicitation.request"
    ]
    return frame["interrupt_id"]


# --- what a session holds --------------------------------------------------------------------


async def stored_turn(service: Any, number: int) -> Any:
    listed = await service.list_sessions(app_name=GRAPH_NAME)
    (found,) = [s for s in listed.sessions if s.id == f"{CONVERSATION}#t{number}"]
    return await service.get_session(
        app_name=GRAPH_NAME, user_id=found.user_id, session_id=found.id
    )


async def in_memory(arm: str, script: str, actions: list[dict[str, Any]], **options: Any) -> Any:
    """The same turn on the in-memory session service, in one process: frames and the turn's
    session."""
    task, steps = SCRIPTS[script]
    service = InMemorySessionService()
    rig = await build(arm, task, steps, service=service, **options)  # type: ignore[arg-type]
    results = await run_actions(rig, task, actions)
    return results, await stored_turn(service, 1)


def same_frames(left: list[dict[str, Any]], right: list[dict[str, Any]]) -> bool:
    """Whether two lists of segments' frames are the same turn: names and fields, apart from
    the ids, times and sequence numbers a run mints."""
    return [events(s["frames"]) for s in left] == [events(s["frames"]) for s in right] and (
        differences(stable([s["frames"] for s in left]), stable([s["frames"] for s in right])) == []
    )


def plain(value: Any) -> Any:
    """`value` as JSON stores it, so a state read back from Postgres compares with one that
    never left memory."""
    return json.loads(json.dumps(value, default=str))


# --- 1. A planned turn costs the same rows, bytes and writes on both arms ----------------------

METRICS = (
    "turn_event_rows",
    "turn_event_json_bytes",
    "turn_event_stored_bytes",
    "turn_state_json_bytes",
    "turn_state_stored_bytes",
    "ledger_event_rows",
    "turn_event_inserts",
    "turn_session_updates",
    "ledger_event_inserts",
    "ledger_session_updates",
)


async def measured(
    arm: str, server: pg.Server, script: str = "skill-ask", answers: bool = True
) -> dict[str, int]:
    """A planned turn on `arm` (by default one skill read, one ask, its answer and a proposal),
    then what the database holds and the write statements the engine ran."""
    url = server.url(await server.create_database())
    service = DatabaseSessionService(db_url=url)
    await service.prepare_tables()
    log = WriteLog()
    log.attach(service)
    task, steps = SCRIPTS[script]
    try:
        with planner_window(log):
            rig = await build(arm, task, steps, service=service)  # type: ignore[arg-type]
            results = await run_actions(
                rig, task, [TURN, {"kind": "answer", "answers": "text"}] if answers else [TURN]
            )
        assert events(results[-1]["frames"])[-1] == "plan.proposed"
        held = await inspect(url)
    finally:
        await service.close()
    turn, ledger = held["events"][TURN_1], held["events"][LEDGER]
    state = held["states"][TURN_1]
    return {
        "turn_event_rows": turn["rows"],
        "progress_rows": held["progress"].get(f"{CONVERSATION}#t1", 0),
        "planner_authored_rows": held["authors"][f"{CONVERSATION}#t1/{PLANNER}"],
        "turn_event_json_bytes": turn["json_bytes"],
        "turn_event_stored_bytes": turn["stored_bytes"],
        "turn_state_json_bytes": state["json_bytes"],
        "turn_state_stored_bytes": state["stored_bytes"],
        "ledger_event_rows": ledger["rows"],
        "window_event_inserts": log.count(table="events", scope="turn", planner=True),
        "window_session_updates": log.count(table="sessions", scope="turn", planner=True),
        "turn_event_inserts": log.count(table="events", scope="turn"),
        "turn_session_updates": log.count(table="sessions", scope="turn"),
        "ledger_event_inserts": log.count(table="events", scope="ledger"),
        "ledger_session_updates": log.count(table="sessions", scope="ledger"),
    }


def drift(adk: dict[str, int], pai: dict[str, int]) -> dict[str, float]:
    """Each metric's difference between the arms as a share of dev's."""
    return {k: (pai[k] - adk[k]) / adk[k] for k in METRICS if adk[k]}


def beyond(drifted: dict[str, float]) -> dict[str, float]:
    return {k: v for k, v in drifted.items() if abs(v) > TOLERANCE}


async def test_a_planned_turn_costs_the_same_rows_bytes_and_writes_on_both_arms(
    server: pg.Server,
) -> None:
    adk, pai = await measured("adk", server), await measured("pai", server)

    report("line 1 adk", adk)
    report("line 1 pai", pai)
    report("line 1 drift", drift(adk, pai))
    assert beyond(drift(adk, pai)) == {}


async def test_a_planner_run_of_four_responses_costs_the_same_rows_on_both_arms(
    server: pg.Server,
) -> None:
    adk = await measured("adk", server, "four-responses", answers=False)
    pai = await measured("pai", server, "four-responses", answers=False)

    report("rows of four responses adk", adk)
    report("rows of four responses pai", pai)
    report("rows of four responses drift", drift(adk, pai))
    assert beyond(drift(adk, pai)) == {}


async def test_control_a_node_that_writes_each_call_twice_is_caught_by_the_comparison(
    server: pg.Server, monkeypatch: pytest.MonkeyPatch
) -> None:
    adk = await measured("adk", server)
    original = pai_node.call_events

    def doubled(*args: Any) -> list[Event]:
        return [*original(*args), *original(*args)]

    monkeypatch.setattr(pai_node, "call_events", doubled)
    pai = await measured("pai", server)

    caught = beyond(drift(adk, pai))
    assert {"turn_event_rows", "turn_event_inserts"} <= set(caught)


# --- 2. Resume after an ask from a fresh process ------------------------------------------------


async def landing(
    url: str,
    segments: list[dict[str, Any]],
    reference: tuple[list[dict[str, Any]], Any],
    *,
    number: int = 1,
) -> list[str]:
    """What differs between a turn run in pieces over the database and the same turn run
    uninterrupted in memory: the frames, the keys of the turn's state and its plan."""
    results, session = reference
    problems = []
    if not same_frames(segments, results):
        problems.append("frames")
    service = DatabaseSessionService(db_url=url)
    try:
        stored = await stored_turn(service, number)
    finally:
        await service.close()
    if set(stored.state) != set(session.state):
        problems.append("state keys")
    if differences(
        stable(plain(stored.state.get("plan"))), stable(plain(session.state.get("plan")))
    ):
        problems.append("plan")
    return problems


@pytest.mark.parametrize("arm", ARMS)
async def test_an_ask_answered_in_a_new_process_lands_the_plan_an_uninterrupted_run_lands(
    arm: str, url: str, tmp_path: Path
) -> None:
    reference = await in_memory(arm, "skill-ask", [TURN, {"kind": "answer", "answers": "text"}])
    first = run_child(url, tmp_path, arm=arm, script="skill-ask", last=2, actions=[TURN])
    second = run_child(
        url,
        tmp_path,
        arm=arm,
        script="skill-ask",
        first=2,
        actions=[answer("text", pause_id(first))],
    )

    assert first["pid"] != second["pid"] != os.getpid()
    assert events(first["actions"][0]["frames"])[-1] == "elicitation.request"
    assert events(second["actions"][0]["frames"])[-1] == "plan.proposed"
    assert second["planner_requests"] == 1
    assert await landing(url, [first["actions"][0], second["actions"][0]], reference) == []


async def test_control_a_second_process_on_another_database_cannot_answer_the_ask(
    server: pg.Server, url: str, tmp_path: Path
) -> None:
    reference = await in_memory("pai", "skill-ask", [TURN, {"kind": "answer", "answers": "text"}])
    first = run_child(url, tmp_path, arm="pai", script="skill-ask", last=2, actions=[TURN])
    elsewhere = server.url(await server.create_database())
    second = run_child(
        elsewhere,
        tmp_path,
        arm="pai",
        script="skill-ask",
        first=2,
        actions=[answer("text", pause_id(first))],
    )

    answered = second["actions"][0]
    assert answered["status"] == 404
    assert "frames" in await landing(url, [first["actions"][0], answered], reference)


# --- 3. A kill mid-planning ---------------------------------------------------------------------

T1 = f"{CONVERSATION}#t1"
T2 = f"{CONVERSATION}#t2"


def summary(done: dict[str, Any]) -> dict[str, Any]:
    """What a segment answered, in the fields a comparison needs."""
    answered = done["actions"][0]
    error = answered["error"] or {}
    frames = answered["frames"]
    return {
        "status": answered["status"],
        # The wait the 409 names is the rest of the lease, which moves with the clock.
        "retry_after_within_lease": (
            error.get("retry_after_seconds") in range(1, SEGMENT_TTL_S + 1) if error else None
        ),
        "error": error.get("error"),
        "frames": events(frames),
        "versions": [f["version"] for f in frames if f["event"] == "plan.proposed"],
        "planner_requests": done["planner_requests"],
        "translations": done["translations"],
    }


def turn_records(held: dict[str, Any]) -> dict[str, Any]:
    """The conversation ledger's record of each closed turn: what the user asked and the error
    it ended with."""
    ledger = held["ledger"]
    return {
        key.removeprefix("studio:turn:"): {"user": item.get("user"), "error": item.get("error")}
        for key, item in sorted(ledger.items())
        if key.startswith("studio:turn:")
    }


async def crashed_turn(
    arm: str, url: str, tmp_path: Path, *, kill_it: bool = True
) -> dict[str, Any]:
    """A first turn killed while its first `propose_plan` translates, after the planner's
    events of its skill read are stored; then the turn sent again at once, and again after the
    segment's lease has run out. With `kill_it` false the first process finishes instead."""
    common = {"arm": arm, "script": "skill-propose", "actions": [TURN]}
    first = start_child(url, tmp_path, {**common, "block": kill_it})
    if kill_it:
        wait_blocked(first)
        at_kill = await planner_trace(url, T1)
        kill(first)
        before = None
    else:
        before = summary(first.finish())
        at_kill = await planner_trace(url, T1)
    at_once = summary(run_child(url, tmp_path, **common))
    after_lease = summary(run_child(url, tmp_path, **common, clock_skew=60))
    held = await inspect(url)
    return {
        "at_kill": at_kill,
        "first": before,
        "at_once": at_once,
        "after_lease": after_lease,
        "t1_at_end": await planner_trace(url, T1),
        "t2": await planner_trace(url, T2),
        "turn_records": turn_records(held),
        "sessions": sorted(k for k in held["events"] if "#t" in k),
    }


def the_crash_shows(outcome: dict[str, Any]) -> bool:
    """Whether a turn left a dead process's marks: the next request refused while the lease
    lives, the dead turn's planner events kept in a session of their own with the proposal's
    call unanswered, and the turn taken again from the start once the lease has run out."""
    orphan = outcome["t1_at_end"][-1]
    return (
        outcome["at_once"]["status"] == 409
        and outcome["at_once"]["retry_after_within_lease"]
        and orphan == ["dfo_planner", "call", "propose_plan"]
        and outcome["after_lease"]["status"] == 200
        and outcome["after_lease"]["frames"][-1] == "plan.proposed"
        and outcome["t2"][-2:]
        == [
            ["dfo_planner", "response", "propose_plan"],
            ["dfo_analyst_graph", "call", "adk_request_input"],
        ]
    )


async def test_a_kill_mid_planning_is_survived_as_on_dev_by_taking_the_turn_again(
    server: pg.Server, tmp_path: Path
) -> None:
    outcomes = {}
    for arm in ARMS:
        outcomes[arm] = await crashed_turn(
            arm, server.url(await server.create_database()), tmp_path
        )
        report(f"line 3 {arm}", outcomes[arm])

    assert all(the_crash_shows(o) for o in outcomes.values())
    # The skill read ran again in the second turn and the unfinished proposal ran once in
    # each turn, with the same frames, requests and records on both arms.
    assert outcomes["pai"] == outcomes["adk"]
    assert outcomes["adk"]["at_kill"][-3:] == [
        ["dfo_planner", "call", "get_skill"],
        ["dfo_planner", "response", "get_skill"],
        ["dfo_planner", "call", "propose_plan"],
    ]
    # The dead turn is closed as abandoned when the next one begins.
    assert [r["error"] for r in outcomes["adk"]["turn_records"].values()] == ["abandoned"]


async def test_control_a_turn_that_finishes_leaves_nothing_a_kill_leaves(
    server: pg.Server, tmp_path: Path
) -> None:
    outcome = await crashed_turn(
        "pai", server.url(await server.create_database()), tmp_path, kill_it=False
    )

    assert outcome["at_once"]["status"] == 200
    assert not the_crash_shows(outcome)


async def crashed_resume(
    arm: str, url: str, tmp_path: Path, *, kill_it: bool = True
) -> dict[str, Any]:
    """A turn paused on an ask, answered, and the process killed while the resumed run's
    proposal translates; then the same answer sent again at once, and again after the lease
    has run out, as the client does when its stream died. With `kill_it` false the resumed run
    finishes first."""
    script = {"arm": arm, "script": "skill-ask"}
    asked = run_child(url, tmp_path, **script, last=2, actions=[TURN])
    resume = {**script, "first": 2, "actions": [answer("text", pause_id(asked))]}
    second = start_child(url, tmp_path, {**resume, "block": kill_it})
    if kill_it:
        wait_blocked(second)
        at_kill = await planner_trace(url, T1)
        kill(second)
        before = None
    else:
        before = summary(second.finish())
        at_kill = await planner_trace(url, T1)
    at_once = summary(run_child(url, tmp_path, **resume))
    after_lease = summary(run_child(url, tmp_path, **resume, clock_skew=60))
    held = await inspect(url)
    return {
        "at_kill": at_kill,
        "first": before,
        "at_once": at_once,
        "after_lease": after_lease,
        "t1_at_end": await planner_trace(url, T1),
        "turn_records": turn_records(held),
        "sessions": sorted(k for k in held["events"] if "#t" in k),
    }


def the_resume_crash_shows(outcome: dict[str, Any]) -> bool:
    """Whether a resumed run left a dead process's marks and was resumed again by the same
    answer: refused while the lease lives, one skill read and one ask in the session, the dead
    run's proposal call and the retry's, and the plan landing."""
    trace = [tuple(t[1:]) for t in outcome["t1_at_end"] if t[0] == PLANNER]
    return (
        outcome["at_once"]["status"] == 409
        and outcome["after_lease"]["status"] == 200
        and outcome["after_lease"]["frames"][-1] == "plan.proposed"
        and trace.count(("response", "get_skill")) == 1
        and trace.count(("call", "ask_user_question")) == 1
        and trace.count(("call", "propose_plan")) == 2
        and trace.count(("response", "propose_plan")) == 1
    )


async def test_a_kill_during_a_resumed_run_is_survived_as_on_dev_by_sending_the_answer_again(
    server: pg.Server, tmp_path: Path
) -> None:
    outcomes = {}
    for arm in ARMS:
        outcomes[arm] = await crashed_resume(
            arm, server.url(await server.create_database()), tmp_path
        )
        report(f"line 3 resume {arm}", outcomes[arm])

    assert all(the_resume_crash_shows(o) for o in outcomes.values())
    assert outcomes["pai"] == outcomes["adk"]
    # One session holds the whole turn: the answer resumed it in place, so nothing is read or
    # asked again, and the retry adds the answer's response and a second proposal call.
    assert len(outcomes["adk"]["sessions"]) == 1


async def test_control_a_resumed_run_that_finishes_leaves_nothing_a_kill_leaves(
    server: pg.Server, tmp_path: Path
) -> None:
    outcome = await crashed_resume(
        "pai", server.url(await server.create_database()), tmp_path, kill_it=False
    )

    assert outcome["at_once"]["status"] == 404
    assert not the_resume_crash_shows(outcome)


# --- 4. A second turn of the same conversation --------------------------------------------------


@dataclass(frozen=True)
class SessionCall:
    """The `PlannerCall` `revise` reads, over a finished session's events."""

    steps_: tuple[Step, ...]
    state: dict[str, Any]
    call_id: str | None = None

    def steps(self) -> tuple[Step, ...]:
        return self.steps_

    def response_calls(self) -> tuple[ResponseCall, ...]:
        return ()


def reads(session: Any) -> dict[str, Any]:
    """What a revision of this turn's plan reads of its session: the steps without the ids a
    framework mints for a call, and each reader's answer over them."""
    steps = tuple(_session_steps(session.events))
    reading = CallSession(SessionCall(steps, dict(session.state)))
    return {
        "steps": [(s.kind, s.name, s.args, dict(s.body)) for s in steps],
        "elected": reading.elected_spec_ids(),
        "combinations": reading.combination_picks(),
        "since_referral": reading.answers_since_referral(),
        "prices": prices_chosen(SessionCall(steps, dict(session.state))),
    }


def starts_clean(read: dict[str, Any]) -> bool:
    """Whether a turn's planner record holds neither the question turn one's planner asked nor
    the answer it was given: nothing of an earlier turn's reaches the readers."""
    text = json.dumps(read["steps"], default=str)
    return "Which years do you mean?" not in text and "Since 2020" not in text


def request_text(call: dict[str, Any]) -> str:
    return json.dumps(sc.contents_list(call["contents"]), default=str)


@dataclass
class TwoTurns:
    results: list[dict[str, Any]]
    first: Any
    second: Any
    calls: list[dict[str, Any]]
    ledger: dict[str, Any]


async def two_turns(arm: str, server: pg.Server) -> TwoTurns:
    """Turn one asks, is answered, proposes and is approved; turn two proposes and is amended.
    Both run on one app over one database."""
    url = server.url(await server.create_database())
    service = DatabaseSessionService(db_url=url)
    task, steps = SCRIPTS["two-turns"]
    try:
        rig = await build(arm, task, steps, service=service, turns=2)  # type: ignore[arg-type]
        results = await run_actions(
            rig,
            task,
            [
                TURN,
                {"kind": "answer", "answers": "text"},
                {"kind": "approve"},
                TURN,
                {"kind": "amend", "words": "only 2020"},
            ],
        )
        held = await inspect(url)
        return TwoTurns(
            results,
            await stored_turn(service, 1),
            await stored_turn(service, 2),
            rig.client.calls,
            held["ledger"],
        )
    finally:
        await service.close()


def versions(results: list[dict[str, Any]]) -> list[list[int]]:
    return [[f["version"] for f in r["frames"] if f["event"] == "plan.proposed"] for r in results]


async def test_a_second_turn_starts_with_a_clean_planner_record_and_amends_its_own_plan(
    server: pg.Server,
) -> None:
    adk, pai = await two_turns("adk", server), await two_turns("pai", server)

    for turns in (adk, pai):
        assert [events(r["frames"])[-1] for r in turns.results] == [
            "elicitation.request",
            "plan.proposed",
            "error",
            "plan.proposed",
            "plan.proposed",
        ]
        assert events(turns.results[2]["frames"])[0] == "plan.approved"
        # Each turn has a session of its own; the second's planner record holds its own
        # steps only, and its plan counts from version one.
        assert turns.first.id != turns.second.id
        assert versions(turns.results) == [[], [1], [], [1], [2]]
        assert not starts_clean(reads(turns.first))
        assert starts_clean(reads(turns.second))
        # Turn one's answer reached turn one's resumed request and, as part of the
        # conversation digest the Studio renders, turn two's first request, which the
        # planner's own record does not hold; the amendment's words reached the amendment's.
        assert "Since 2020" in request_text(turns.calls[1])
        assert "Since 2020" in request_text(turns.calls[2])
        assert "only 2020" in request_text(turns.calls[3])

    assert same_frames(adk.results, pai.results)
    assert differences(stable(plain(reads(adk.second))), stable(plain(reads(pai.second)))) == []
    assert differences(stable(plain(reads(adk.first))), stable(plain(reads(pai.first)))) == []
    assert len(adk.calls) == len(pai.calls) == 4
    for index, (a, b) in enumerate(zip(adk.calls, pai.calls, strict=True)):
        assert shape(sc.request_differences(a, b)) == WIRE, f"request {index}"
    carried = {
        arm: {
            "turn 1 state keys": len(turns.first.state),
            "turn 2 state keys": len(turns.second.state),
            "kept in turn 2 only": sorted(set(turns.second.state) - set(turns.first.state)),
            "earlier readings in turn 2": len(
                (turns.second.state.get(ORCHESTRATOR_PRIOR_LEDGER_KEY) or {}).get("readings", [])
            ),
            "ledger keys": sorted(k for k in turns.ledger if not k.startswith("studio:turn_")),
        }
        for arm, turns in (("adk", adk), ("pai", pai))
    }
    report("line 4 carried", carried)
    assert carried["adk"] == carried["pai"]


async def test_control_a_record_over_both_turns_events_does_not_start_clean(
    server: pg.Server,
) -> None:
    turns = await two_turns("pai", server)
    both = SimpleNamespace(
        events=[*turns.first.events, *turns.second.events], state=turns.second.state
    )

    # A planner record built from both turns' events, as one session over the conversation
    # would hold them, finds the ask and the answer.
    assert starts_clean(reads(turns.second))
    assert not starts_clean(reads(both))


# --- 5. An ask paused under one planner, answered under the other -------------------------------

ORDERS = {"dev's then Pydantic AI": ("adk", "pai"), "Pydantic AI then dev's": ("pai", "adk")}


async def paused_and_answered(
    url: str,
    tmp_path: Path,
    arms: tuple[str, str],
    *,
    paused_options: dict[str, Any] | None = None,
    answered_patch: str | None = None,
) -> list[dict[str, Any]]:
    """The first half of the combine-ask turn on `arms[0]` in one process and its second half on
    `arms[1]` in another."""
    script = "combine-ask"
    paused = run_child(
        url, tmp_path, arm=arms[0], script=script, last=2, actions=[TURN], pai=paused_options
    )
    answered = run_child(
        url,
        tmp_path,
        arm=arms[1],
        script=script,
        first=2,
        actions=[answer("combine", pause_id(paused))],
        patch=answered_patch,
    )
    return [paused["actions"][0], answered["actions"][0]]


@pytest.mark.parametrize("arms", ORDERS.values(), ids=ORDERS.keys())
async def test_an_ask_paused_under_one_planner_is_answered_under_the_other_in_a_new_process(
    arms: tuple[str, str], url: str, tmp_path: Path
) -> None:
    reference = await in_memory(
        "adk", "combine-ask", [TURN, {"kind": "answer", "answers": "combine"}]
    )
    segments = await paused_and_answered(url, tmp_path, arms)

    assert events(segments[1]["frames"])[-1] == "plan.proposed"
    assert await landing(url, segments, reference) == []


async def test_control_a_planner_that_reads_none_of_the_earlier_steps_loses_the_answer(
    url: str, tmp_path: Path
) -> None:
    reference = await in_memory(
        "adk", "combine-ask", [TURN, {"kind": "answer", "answers": "combine"}]
    )
    segments = await paused_and_answered(
        url, tmp_path, ("adk", "pai"), answered_patch="no-earlier-steps"
    )

    # The user's pick of the combination is in the session and the node does not read it, so
    # the resumed proposal is refused.
    assert events(segments[1]["frames"])[-1] != "plan.proposed"
    assert "plan" in await landing(url, segments, reference)


async def test_control_a_planner_that_stores_no_events_leaves_dev_nothing_to_resume_from(
    url: str, tmp_path: Path
) -> None:
    reference = await in_memory(
        "adk", "combine-ask", [TURN, {"kind": "answer", "answers": "combine"}]
    )
    segments = await paused_and_answered(
        url, tmp_path, ("pai", "adk"), paused_options={"events": False}
    )

    assert events(segments[1]["frames"])[-1] != "plan.proposed"
    assert "plan" in await landing(url, segments, reference)
