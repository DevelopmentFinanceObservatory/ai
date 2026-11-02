"""Five cases that have hurt the harness, each run against the own loop and the
ADK workflow, every resume in a fresh process.

Each `check_*` raises `AssertionError` when its pass line fails. Own-loop
tests call it and must pass; each control test runs the loop with one
property removed and asserts the check fails. ADK tests assert what ADK does,
with and without dev's guards, and print it under `-s`.
"""

import time
import uuid
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pytest
import runlog
from procs import Child, run, spawn

RACE_REPEATS = 20


@dataclass
class Runtime:
    name: str
    script: str
    db_flag: str
    db: str
    log: Path

    def args(self, command: str, turn: str, *flags: str) -> list[str]:
        return [command, self.db_flag, self.db, "--turn", turn, "--log", str(self.log), *flags]

    def run(self, command: str, turn: str, *flags: str) -> list[dict]:
        return run(self.script, *self.args(command, turn, *flags))

    def spawn(self, command: str, turn: str, *flags: str) -> Child:
        return spawn(self.script, *self.args(command, turn, *flags))


@pytest.fixture
def own(pg: dict[str, str], tmp_path: Path) -> Runtime:
    return Runtime("own", "own_driver.py", "--dsn", pg["own"], tmp_path / "runs.jsonl")


@pytest.fixture
def adk(pg: dict[str, str], tmp_path: Path) -> Runtime:
    return Runtime("adk", "adk_driver.py", "--db", pg["adk"], tmp_path / "runs.jsonl")


def new_turn() -> tuple[str, str]:
    conversation = f"c-{uuid.uuid4().hex[:8]}"
    return f"{conversation}#t1", f"{conversation}#t2"


def runs_by(log: Path, *, turn: str, pid: int | None) -> list[str]:
    return [
        line["node"]
        for line in runlog.read(log)
        if line["turn"] == turn and line["pid"] == pid and line["what"] == "run"
    ]


def pid_of(rt: Runtime, *, turn: str, after: float) -> int | None:
    """The pid of the first process that logged a line for `turn` after
    `after`, or `None` when none did."""
    lines = runlog.read(rt.log)
    return next((line["pid"] for line in lines if line["turn"] == turn and line["t"] > after), None)


# --- T1: a deploy between pause and resume ---------------------------------------------


@dataclass
class T1:
    resumed: list[str]
    resumed_outcome: str
    next_turn: list[str]


def t1(rt: Runtime, *resume_flags: str) -> T1:
    """Pause at `approve` under v1, resume under v2 (which adds `vet` before
    `approve`), then run the next turn under v2."""
    turn, following = new_turn()
    assert rt.run("start", turn, "--version", "v1", *resume_flags)[-1]["type"] == "paused"
    before = time.time()
    resumed = rt.run("answer", turn, "--version", "v2", *resume_flags)
    resumed_runs = runs_by(rt.log, turn=turn, pid=pid_of(rt, turn=turn, after=before))
    assert rt.run("start", following, "--version", "v2", *resume_flags)[-1]["type"] == "paused"
    return T1(resumed_runs, resumed[-1]["type"], runlog.runs(rt.log, turn=following))


def check_t1(result: T1) -> None:
    assert result.resumed_outcome == "done", result
    assert "vet" not in result.resumed, f"vet ran in the resumed turn: {result.resumed}"
    assert result.resumed[0] == "approve", f"resumed turn reran {result.resumed}"
    assert "vet" in result.next_turn, f"next turn skipped vet: {result.next_turn}"


def test_t1_own(own: Runtime) -> None:
    result = t1(own)
    print(f"\nT1 own: resumed {result.resumed}; next turn {result.next_turn}")
    check_t1(result)


def test_t1_own_control_from_start(own: Runtime) -> None:
    result = t1(own, "--from-start")
    print(f"\nT1 own control: resumed {result.resumed}")
    with pytest.raises(AssertionError, match="vet ran in the resumed turn"):
        check_t1(result)


@pytest.mark.parametrize("guards", [False, True], ids=["no-guards", "guards"])
def test_t1_adk(adk: Runtime, guards: bool) -> None:
    result = t1(adk, *(["--guards"] if guards else []))
    print(f"\nT1 adk guards={guards}: resumed {result.resumed}; next turn {result.next_turn}")
    assert result.resumed_outcome == "done"
    assert "vet" in result.next_turn
    if guards:
        # `seed` skips by `_turn_ran_before`; `vet` needs a guard of its own.
        assert "vet" not in result.resumed
        assert "seed" not in result.resumed
    else:
        assert result.resumed[:2] == ["seed", "vet"]


# --- T2: a repeated interrupt -----------------------------------------------------------


@dataclass
class T2:
    paused: list[tuple[str, int]]  # (interrupt id, round) of each pause
    late: str
    final: str
    answers: list[tuple[int, int, str]]  # (round on offer, round answered, decision)


def t2(rt: Runtime, *flags: str) -> T2:
    """Round 1 amended, round 2 opened under the same interrupt id, a late
    answer to round 1, then round 2 approved."""
    turn, _ = new_turn()
    first = rt.run("start", turn, *flags)[-1]
    second = rt.run("answer", turn, "--round", "1", "--decision", "amend", *flags)[-1]
    late = rt.run("answer", turn, "--round", "1", "--decision", "approve", *flags)[-1]
    final = rt.run("answer", turn, "--round", "2", "--decision", "approve", *flags)[-1]
    answers = [
        (line["round"], line["answer_round"], line["decision"])
        for line in runlog.read(rt.log)
        if line["node"] == "approve" and line.get("phase") == "answer"
    ]
    return T2(
        [(first["interrupt"], first["round"]), (second["interrupt"], second["round"])],
        late["type"],
        final["type"],
        answers,
    )


def check_t2(result: T2) -> None:
    assert result.paused == [("approve", 1), ("approve", 2)], result
    assert result.late == "conflict", f"late round-1 answer accepted: {result}"
    assert result.answers == [(1, 1, "amend"), (2, 2, "approve")], result
    assert result.final == "done", result


def test_t2_own(own: Runtime) -> None:
    result = t2(own)
    print(f"\nT2 own: {result}")
    check_t2(result)


def test_t2_own_control_id_only(own: Runtime) -> None:
    result = t2(own, "--id-only")
    print(f"\nT2 own control: {result}")
    with pytest.raises(AssertionError, match="late round-1 answer accepted"):
        check_t2(result)


@pytest.mark.parametrize("guards", [False, True], ids=["no-guards", "guards"])
def test_t2_adk(adk: Runtime, guards: bool) -> None:
    result = t2(adk, *(["--guards"] if guards else []))
    print(f"\nT2 adk guards={guards}: {result}")
    if guards:
        check_t2(result)
    else:
        # The late answer resumes round 2's pause and approves round 2's plan.
        assert result.late == "done"
        assert result.answers == [(1, 1, "amend"), (2, 1, "approve")]


# --- T3: a race on one pause ------------------------------------------------------------


def race(rt: Runtime, tmp_path: Path, *flags: str) -> tuple[list[str], Counter]:
    """Two processes answer the same pause, both holding the same read of it
    until both have read. Their outcomes, and how often each node ran after
    the pause."""
    turn, _ = new_turn()
    assert rt.run("start", turn, *flags)[-1]["type"] == "paused"
    before = time.time()
    barrier = tmp_path / uuid.uuid4().hex
    barrier.mkdir()
    children = [rt.spawn("answer", turn, "--barrier", str(barrier), *flags) for _ in range(2)]
    try:
        for child in children:
            child.wait_for(lambda line: line["type"] == "read")
        (barrier / "go").touch()
        outcomes = sorted(child.finish()[-1]["type"] for child in children)
    finally:
        for child in children:
            child.kill()
    after = Counter(
        (line["node"], line["what"], line.get("n"))
        for line in runlog.read(rt.log)
        if line["turn"] == turn and line["t"] > before and line["what"] != "skip"
    )
    return outcomes, after


def check_t3(outcomes: list[str], after: Counter) -> None:
    twice = sorted(
        {key[0] if key[2] is None else f"step {key[2]}" for key, n in after.items() if n > 1}
    )
    assert not twice, f"ran twice: {twice}; outcomes {outcomes}"
    assert outcomes == ["conflict", "done"], f"outcomes {outcomes}"


def test_t3_own(own: Runtime, tmp_path: Path) -> None:
    for _ in range(RACE_REPEATS):
        check_t3(*race(own, tmp_path))


def test_t3_own_control_no_cas(own: Runtime, tmp_path: Path) -> None:
    failures: list[str] = []
    for _ in range(RACE_REPEATS):
        try:
            check_t3(*race(own, tmp_path, "--no-cas"))
        except AssertionError as error:
            failures.append(str(error).splitlines()[0])
    print(f"\nT3 own control: failed {len(failures)}/{RACE_REPEATS}; first: {failures[:1]}")
    assert len(failures) == RACE_REPEATS


@pytest.mark.parametrize("guards", [False, True], ids=["no-guards", "guards"])
def test_t3_adk(adk: Runtime, tmp_path: Path, guards: bool) -> None:
    passed = 0
    seen: Counter = Counter()
    for _ in range(RACE_REPEATS):
        outcomes, after = race(adk, tmp_path, *(["--guards"] if guards else []))
        seen[tuple(outcomes)] += 1
        try:
            check_t3(outcomes, after)
            passed += 1
        except AssertionError:
            pass
    print(f"\nT3 adk guards={guards}: passed {passed}/{RACE_REPEATS}; outcomes {dict(seen)}")
    # Both pass: the loser's first append fails ADK's stale-session check.
    assert passed == RACE_REPEATS


# --- T4: a crash in execute, then the same answer twice ---------------------------------


def lines_of(log: Path, *, turn: str, pid: int | None, what: str) -> list:
    return [
        line.get("n", line["node"])
        for line in runlog.read(log)
        if line["turn"] == turn and line["pid"] == pid and line["what"] == what
    ]


@dataclass
class T4:
    killed_done: list[int]  # steps the killed process finished
    killed_started: list[int]
    resumed_runs: list[str]  # nodes the first repeated answer ran
    resumed_started: list[int]  # steps it started
    first: str
    second: str
    second_runs: list[str]


def t4(rt: Runtime, *flags: str) -> T4:
    """Kill the answering process once step 1 of `execute` has finished, wait
    out its lease, then post the same answer twice."""
    turn, _ = new_turn()
    timing = ["--steps", "5", "--step-s", "0.3"]
    assert rt.run("start", turn, *timing, *flags)[-1]["type"] == "paused"
    child = rt.spawn("answer", turn, *timing, *flags)
    child.wait_for(lambda line: line["type"] == "step" and line["n"] == 1)
    child.kill()
    killed = child.proc.pid
    time.sleep(2.5)  # the default lease TTL, 2 s, lapses
    before = time.time()
    first = rt.run("answer", turn, *timing, *flags)[-1]
    first_pid = pid_of(rt, turn=turn, after=before)
    before = time.time()
    second = rt.run("answer", turn, *timing, *flags)[-1]
    second_lines = [
        line for line in runlog.read(rt.log) if line["turn"] == turn and line["t"] > before
    ]
    return T4(
        killed_done=lines_of(rt.log, turn=turn, pid=killed, what="step"),
        killed_started=lines_of(rt.log, turn=turn, pid=killed, what="step_start"),
        resumed_runs=runs_by(rt.log, turn=turn, pid=first_pid),
        resumed_started=lines_of(rt.log, turn=turn, pid=first_pid, what="step_start"),
        first=first["type"],
        second=second["type"],
        second_runs=[line["node"] for line in second_lines if line["what"] == "run"],
    )


def check_t4(result: T4) -> None:
    assert result.first == "done", result
    redone = sorted(set(result.killed_done) & set(result.resumed_started))
    assert not redone, f"finished steps reran: {redone}; {result}"
    assert result.resumed_runs[0] == "execute", f"resume reran {result.resumed_runs}"
    in_flight = sorted(set(result.killed_started) - set(result.killed_done))
    assert result.resumed_started == [
        *in_flight,
        *range(max(result.killed_done) + 1 + len(in_flight), 5),
    ]
    assert result.second == "conflict", f"second answer accepted: {result}"
    assert result.second_runs == [], result


def test_t4_own(own: Runtime) -> None:
    result = t4(own)
    print(f"\nT4 own: {result}")
    check_t4(result)


def test_t4_own_control_pause_checkpoints_only(own: Runtime) -> None:
    result = t4(own, "--pause-checkpoints-only")
    print(f"\nT4 own control: {result}")
    with pytest.raises(AssertionError, match="finished steps reran"):
        check_t4(result)


@pytest.mark.parametrize("guards", [False, True], ids=["no-guards", "guards"])
def test_t4_adk(adk: Runtime, guards: bool) -> None:
    result = t4(adk, *(["--guards"] if guards else []))
    print(f"\nT4 adk guards={guards}: {result}")
    if guards:
        # The ledger claimed the answer before the kill, so both repeats are
        # refused and nothing in this turn continues the run.
        assert (result.first, result.second) == ("conflict", "conflict")
        assert result.resumed_runs == result.resumed_started == []
    else:
        # Both repeats are accepted. Replay finds `execute` with events and no
        # output, ends its branch, and reports the turn done: steps 2 to 4 and
        # `plan_done` never run.
        assert (result.first, result.second) == ("done", "done")
        assert result.resumed_runs == ["seed"]
        assert result.resumed_started == []


# --- T5: cancel from another instance ---------------------------------------------------


@dataclass
class T5:
    outcome: str
    latency: float | None  # seconds from the cancel's commit to the stop
    started_after_cancel: list[int]
    done_before: list[int]
    resume: str
    resumed_started: list[int]


def t5(rt: Runtime, resume_command: str, *flags: str) -> T5:
    """Process A runs `execute`; once step 1 is done process B writes a cancel;
    then process C resumes the turn."""
    turn, _ = new_turn()
    timing = ["--steps", "40", "--step-s", "0.2"]
    assert rt.run("start", turn, *timing, *flags)[-1]["type"] == "paused"
    a = rt.spawn("answer", turn, *timing, *flags)
    try:
        a.wait_for(lambda line: line["type"] == "step" and line["n"] == 1)
        cancelled_at = rt.run("cancel", turn, *flags)[-1]["at"]
        lines = a.finish(timeout=30)
    finally:
        a.kill()
    outcome = lines[-1]
    stopped_at = outcome.get("stopped_at")
    started_after = [
        line["n"]
        for line in runlog.read(rt.log)
        if line["turn"] == turn and line["what"] == "step_start" and line["t"] > cancelled_at + 0.05
    ]
    done_before = lines_of(rt.log, turn=turn, pid=a.proc.pid, what="step")
    before = time.time()
    resume = rt.run(resume_command, turn, *timing, *flags)[-1]
    resumed_started = [
        line["n"]
        for line in runlog.read(rt.log)
        if line["turn"] == turn and line["what"] == "step_start" and line["t"] > before
    ]
    return T5(
        outcome["type"],
        None if stopped_at is None else stopped_at - cancelled_at,
        started_after,
        done_before,
        resume["type"],
        resumed_started,
    )


def check_t5(result: T5) -> None:
    assert result.outcome == "stopped", f"did not stop: {result.outcome}"
    assert result.latency is not None and result.latency < 1.0, result
    assert result.started_after_cancel == [], result
    assert result.resume == "done", result
    redone = sorted(set(result.done_before) & set(result.resumed_started))
    assert not redone, f"finished steps reran: {redone}"


def test_t5_own(own: Runtime) -> None:
    result = t5(own, "resume")
    print(f"\nT5 own: latency {result.latency:.3f}s; {result}")
    check_t5(result)


def test_t5_own_control_in_process_only(own: Runtime) -> None:
    result = t5(own, "resume", "--in-process-cancel-only")
    print(f"\nT5 own control: {result}")
    with pytest.raises(AssertionError, match="did not stop"):
        check_t5(result)


@pytest.mark.parametrize("guards", [False, True], ids=["no-guards", "guards"])
def test_t5_adk(adk: Runtime, guards: bool) -> None:
    result = t5(adk, "answer", *(["--guards"] if guards else []))
    print(f"\nT5 adk guards={guards}: {result}")
    if guards:
        # Dev's 5 s heartbeat sees the flag; the answer is claimed, so the
        # stopped turn cannot be continued by answering again.
        assert result.outcome == "stopped"
        assert result.latency is not None and result.latency < 5.5
        assert result.resume == "conflict"
    else:
        assert result.outcome == "done"
