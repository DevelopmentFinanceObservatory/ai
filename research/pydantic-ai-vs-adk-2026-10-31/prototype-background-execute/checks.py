"""The in-process checks X1, X3, X4 and X5. Each raises `AssertionError` when its
pass line fails, so the design's test calls it plainly and a control's test
expects the failure.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import time
from collections.abc import Callable
from typing import Any

from dbos import DBOS
from durable_execute import OPENED, BMode, launch
from executor.ledger import TERMINAL_STATUSES, RunRecord, stored_run
from fakes import SlowSurface, ledger_snapshot
from google.adk.sessions import InMemorySessionService
from harness.graph.state import AGENT_NAME
from harness.run_ledger import StudioRunLedger
from harness.studio_state import StudioStore
from studio_app import BRIEF, CANCEL, Arm, StudioApp, build_app, interrupt_body
from tests.orchestrator_fakes import (
    AUTH,
    CONVERSATION,
    handed_back,
    request_text,
    stream,
    studio_client,
    turn_body,
)
from wiring import Timed, read_timed, streaming_client

RESUMED_LEAD = "Calculations an earlier turn left unfinished were carried on"


def runs(built: StudioApp) -> list[RunRecord]:
    return [run for book in built.ledger._books.values() for run in book.runs.values()]


async def wait_for(predicate: Callable[[], bool], seconds: float) -> None:
    async def poll() -> None:
        while not predicate():
            await asyncio.sleep(0.01)

    await asyncio.wait_for(poll(), seconds)


async def first_card(built: StudioApp) -> dict[str, Any]:
    async with studio_client(built.app) as client:
        first = await stream(client, "/api/studio/turn", turn_body(BRIEF))
    return next(f for f in first if f["event"] == "plan.proposed")


# ---------------------------------------------------------------------------
# X1: the client leaves during `execute`
# ---------------------------------------------------------------------------


async def check_client_leaves(patch: Any, arm: Arm) -> dict[str, Any]:
    """Approve, read up to the aggregate step's first progress frame, close the
    stream, release the step, wait for the run, then send the next turn."""
    hold = asyncio.Event()
    built = build_app(patch, arm=arm, surface=SlowSurface(hold={"aggregate": hold}), plans=1)
    card = await first_card(built)
    async with (
        streaming_client(built.app) as client,
        client.stream(
            "POST",
            "/api/studio/interrupt",
            json=interrupt_body(card["interrupt_id"], "approve"),
            headers=AUTH,
        ) as response,
    ):

        async def read_to_aggregate() -> None:
            async for line in response.aiter_lines():
                if line.startswith("data: ") and '"detail":"aggregate' in line.replace(" ", ""):
                    return

        await asyncio.wait_for(read_to_aggregate(), 10)
    left_at = list(built.surface.lines())
    hold.set()
    with contextlib.suppress(TimeoutError):
        await wait_for(lambda: bool(runs(built)) and runs(built)[0].status in TERMINAL_STATUSES, 5)
    (run,) = runs(built)
    finished = (run.status.value, list(built.surface.lines()))
    async with studio_client(built.app) as client:
        nxt = await stream(client, "/api/studio/turn", turn_body("Is it ready?"))
    text = request_text(built.model.requests[-1])
    (after,) = runs(built)
    result = {
        "lines_when_client_left": left_at,
        "run_when_next_turn_came": finished,
        "lines": built.surface.lines(),
        "run_after": (after.status.value, after.delivered, after.resumes),
        "next_turn": [(f["event"], f.get("status"), f.get("tool")) for f in nxt],
        "orchestrator_read": text[text.find(RESUMED_LEAD) :][:600] if RESUMED_LEAD in text else "",
    }
    print(json.dumps(result))
    assert finished[0] == "completed", f"the run did not finish with no client: {finished}"
    assert built.surface.lines() == [
        "get_data start",
        "get_data end",
        "aggregate start",
        "aggregate end",
    ], f"a step ran twice: {built.surface.lines()}"
    assert RESUMED_LEAD in text and '"status": "completed"' in text, (
        "the orchestrator was not handed the run"
    )
    assert after.delivered, "the run's hand-back was not recorded delivered"
    assert nxt[-1]["event"] == "done"
    return result


# ---------------------------------------------------------------------------
# X3: live progress while the client stays
# ---------------------------------------------------------------------------

STEP_SECONDS = 2.0


def _activity(frames: list[dict[str, Any]]) -> list[tuple[Any, ...]]:
    return [
        (f["status"], f["label"], f.get("tool"), f.get("detail"))
        for f in frames
        if f["event"] == "activity.step"
    ]


async def check_live_progress(patch: Any, arm: Arm, mode: BMode = BMode.DURABLE) -> dict[str, Any]:
    """Approve over a streaming client, each step sending two reports over
    `STEP_SECONDS`; every report must reach the client within 1 s. The lag of
    each step's `complete` frame is measured too."""
    surface = SlowSurface(delay={"get_data": STEP_SECONDS, "aggregate": STEP_SECONDS})
    built = build_app(patch, arm=arm, mode=mode, surface=surface)
    card = await first_card(built)
    timed = Timed()
    started = time.monotonic()
    async with (
        streaming_client(built.app) as client,
        client.stream(
            "POST",
            "/api/studio/interrupt",
            json=interrupt_body(card["interrupt_id"], "approve"),
            headers=AUTH,
        ) as response,
    ):
        await asyncio.wait_for(read_timed(response, timed, started), 30)
    arrived = [started + at for at in timed.at]
    lags: dict[str, float] = {}
    for entry in surface.log:
        if entry["event"] == "progress":
            matches = [
                t
                for f, t in zip(timed.frames, arrived, strict=True)
                if f["event"] == "activity.step" and f.get("detail") == entry["message"]
            ]
            lags[entry["message"]] = round(matches[0] - entry["t"], 3) if matches else float("inf")
        elif entry["event"] == "end":
            matches = [
                t
                for f, t in zip(timed.frames, arrived, strict=True)
                if f["event"] == "activity.step"
                and f["status"] == "complete"
                and f.get("tool") == entry["tool"]
            ]
            lags[f"{entry['tool']} complete"] = (
                round(matches[0] - entry["t"], 3) if matches else float("inf")
            )
    result = {"lags": lags, "activity": _activity(timed.frames), "last": timed.frames[-1]["event"]}
    print(json.dumps({"arm": arm, "mode": mode, "lags": lags}))
    assert timed.frames[-1]["event"] == "done"
    # A step's `complete` frame waits for the node's next non-progress event in
    # dev's translator too (`_Translator._plan_path_frames` drains the recorder
    # there, not on a progress event), so only the reports are held to 1 s.
    late = {k: v for k, v in lags.items() if v >= 1.0 and not k.endswith(" complete")}
    assert not late, f"progress reached the client 1 s or more after the step reported it: {late}"
    return result


# ---------------------------------------------------------------------------
# X4: Stop cancels the background run
# ---------------------------------------------------------------------------


async def check_stop(patch: Any, mode: BMode) -> dict[str, Any]:
    """Approve, Stop while the first step (`get_data`) is held, then run the next
    turn. The aggregate step must never start."""
    hold = asyncio.Event()
    built = build_app(patch, arm="durable", mode=mode, surface=SlowSurface(hold={"get_data": hold}))
    card = await first_card(built)
    async with studio_client(built.app) as client:
        ran = asyncio.ensure_future(
            stream(client, "/api/studio/interrupt", interrupt_body(card["interrupt_id"], "approve"))
        )
        await wait_for(lambda: "get_data start" in built.surface.lines(), 5)
        stopped_at = time.monotonic()
        cancel = await client.post("/api/studio/cancel", json=CANCEL, headers=AUTH)
        stopped = await asyncio.wait_for(ran, 5)
        await asyncio.sleep(0.2)
        workflow = OPENED[-1][0]
        status = await DBOS.get_workflow_status_async(workflow)
        lines_at_stop = built.surface.lines()
        nxt = await stream(client, "/api/studio/turn", turn_body("And for Ghana?"))
    started_after = [
        f"{e['tool']} start"
        for e in built.surface.log
        if e["event"] == "start" and e["t"] > stopped_at
    ]
    (run, *_) = runs(built)
    result = {
        "cancel": cancel.status_code,
        "stream_end": stopped[-1].get("error", {}).get("error"),
        "workflow_status": status.status if status else None,
        "lines_at_stop": lines_at_stop,
        "steps_started_after_stop": started_after,
        "run": (run.status.value, run.failure.code if run.failure else None),
        "next_turn_ends": [f["event"] for f in nxt][-1],
    }
    print(json.dumps(result))
    _LAST.update(built=built, workflow=workflow, hold=hold)
    assert cancel.status_code == 204
    assert stopped[-1]["event"] == "error"
    assert lines_at_stop == ["get_data start", "get_data cancelled"], lines_at_stop
    assert not started_after, f"a step started after the Stop: {started_after}"
    assert result["workflow_status"] == "CANCELLED", f"workflow left {result['workflow_status']}"
    assert nxt[-1]["event"] == "plan.proposed"
    return result


async def observe_relaunch_after_local_cancel(patch: Any, dbos_db: Any) -> dict[str, Any]:
    """The `LOCAL_CANCEL` node's Stop, then DBOS relaunched as the same executor:
    what the workflow it left behind does with no one listening."""
    with contextlib.suppress(AssertionError):
        await check_stop(patch, BMode.LOCAL_CANCEL)
    built: StudioApp = _LAST["built"]
    workflow: str = _LAST["workflow"]
    _LAST["hold"].set()
    before = len(built.surface.log)
    DBOS.destroy()
    launch(dbos_db, "test")
    handle = await DBOS.retrieve_workflow_async(workflow)
    try:
        await asyncio.wait_for(handle.get_result(polling_interval_sec=0.05), 10)
    except Exception as exc:
        error = repr(exc)
    else:
        error = None
    status = await DBOS.get_workflow_status_async(workflow)
    return {
        "status_after_relaunch": status.status if status else None,
        "lines_after_relaunch": [
            f"{e['tool']} {e['event']}"
            for e in built.surface.log[before:]
            if e["event"] != "progress"
        ],
        "error": error,
        "ledger_runs": [(run.status.value, len(run.attempts)) for run in runs(built)],
    }


_LAST: dict[str, Any] = {}


# ---------------------------------------------------------------------------
# X5: the records equal dev's
# ---------------------------------------------------------------------------

# Times, and ids minted per run.
VOLATILE = {
    "run_id",
    "admitted_at",
    "at",
    "sealed_at",
    "expires_at",
    "approved_at",
    "observed_at",
    "seq",
    "interrupt_id",
    "trace_id",
    "duration_ms",
}


def stable(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: stable(v) for k, v in value.items() if k not in VOLATILE}
    if isinstance(value, list | tuple):
        return [stable(v) for v in value]
    if isinstance(value, str) and value.startswith("{"):
        try:
            return stable(json.loads(value))
        except json.JSONDecodeError:
            return value
    return value


def differences(a: Any, b: Any, path: str = "") -> list[str]:
    """The paths at which `a` and `b` differ."""
    if isinstance(a, dict) and isinstance(b, dict):
        return [
            d
            for key in sorted(set(a) | set(b), key=str)
            for d in differences(a.get(key), b.get(key), f"{path}.{key}")
        ]
    if isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        return [
            d
            for i, (x, y) in enumerate(zip(a, b, strict=True))
            for d in differences(x, y, f"{path}[{i}]")
        ]
    return [] if a == b else [f"{path}: {a!r} != {b!r}"]


async def records(patch: Any, arm: Arm) -> dict[str, Any]:
    """Run a turn, its approval and the plan's run with the client attached;
    the ledger, what was published and the hand-back, times and ids removed."""
    with patch.context() as scoped:
        built = build_app(scoped, arm=arm, plans=1)
        card = await first_card(built)
        async with studio_client(built.app) as client:
            ran = await stream(
                client, "/api/studio/interrupt", interrupt_body(card["interrupt_id"], "approve")
            )
    assert ran[-1]["event"] == "done"
    _, hand_back = handed_back(built.model)
    return stable(
        {
            "ledger": ledger_snapshot(built.ledger),  # type: ignore[arg-type]
            "published": built.publisher.published,
            "hand_back": hand_back,
        }
    )


# ---------------------------------------------------------------------------
# Probe: X1 on Studio's own ledger
# ---------------------------------------------------------------------------


async def observe_client_leaves_on_studio_ledger(patch: Any) -> dict[str, Any]:
    """X1's disconnect with `StudioRunLedger`, the ledger `harness.server` wires,
    whose writes are fenced on the conversation's live segment."""
    session_service = InMemorySessionService()
    store = StudioStore(session_service, app_name=AGENT_NAME)
    hold = asyncio.Event()
    built = build_app(
        patch,
        arm="durable",
        surface=SlowSurface(hold={"aggregate": hold}),
        plans=1,
        session_service=session_service,
        store=store,
        ledger=StudioRunLedger(store),
    )
    card = await first_card(built)
    async with (
        streaming_client(built.app) as client,
        client.stream(
            "POST",
            "/api/studio/interrupt",
            json=interrupt_body(card["interrupt_id"], "approve"),
            headers=AUTH,
        ) as response,
    ):

        async def read_to_aggregate() -> None:
            async for line in response.aiter_lines():
                if line.startswith("data: ") and '"detail":"aggregate' in line.replace(" ", ""):
                    return

        await asyncio.wait_for(read_to_aggregate(), 10)
    hold.set()
    workflow = OPENED[-1][0]
    handle = await DBOS.retrieve_workflow_async(workflow)
    result = await asyncio.wait_for(handle.get_result(polling_interval_sec=0.05), 10)
    (principal,) = session_service.sessions[f"{AGENT_NAME}.ledger"]  # type: ignore[attr-defined]
    items = await store.run_items(principal, CONVERSATION)
    (run,) = [stored_run(v) for k, v in items.items() if k.startswith("studio:run:")]
    lines_before_next = built.surface.lines()
    async with studio_client(built.app) as client:
        await stream(client, "/api/studio/turn", turn_body("Is it ready?"))
    items = await store.run_items(principal, CONVERSATION)
    (after,) = [stored_run(v) for k, v in items.items() if k.startswith("studio:run:")]
    return {
        "workflow_result": {k: v for k, v in result.items() if k != "progress"},
        "run_when_workflow_ended": (run.status.value, [a.step_id for a in run.attempts]),
        "lines_when_workflow_ended": lines_before_next,
        "lines_after_next_turn": built.surface.lines(),
        "run_after_next_turn": (after.status.value, after.resumes, after.delivered),
    }
