"""X1, X3, X4 and X5 in process; X2 in child processes (`test_kill.py`).

Each pass line runs on the background node and, as its control, on the node or
mode the line names, which fails it.
"""

from collections.abc import Iterator
from pathlib import Path

import pytest
from checks import (
    check_client_leaves,
    check_live_progress,
    check_stop,
    differences,
    observe_client_leaves_on_studio_ledger,
    observe_relaunch_after_local_cancel,
    records,
)
from dbos import DBOS
from durable_execute import BMode, launch


@pytest.fixture
def dbos_db(tmp_path: Path) -> Iterator[Path]:
    path = tmp_path / "dbos.sqlite"
    launch(path, "test")
    yield path
    DBOS.destroy()


# X1 ------------------------------------------------------------------------


async def test_x1_the_run_finishes_with_no_client_and_the_next_turn_gets_it(
    monkeypatch: pytest.MonkeyPatch, dbos_db: Path
) -> None:
    await check_client_leaves(monkeypatch, "durable")


async def test_x1_control_dev_cancels_the_run_on_disconnect(
    monkeypatch: pytest.MonkeyPatch, dbos_db: Path
) -> None:
    with pytest.raises(AssertionError, match="did not finish with no client") as caught:
        await check_client_leaves(monkeypatch, "dev")
    print(repr(caught.value))


async def test_x1_probe_studio_ledger_refuses_writes_once_the_stream_ends(
    monkeypatch: pytest.MonkeyPatch, dbos_db: Path
) -> None:
    """Probe: with the ledger `harness.server` wires, the steps finish after
    the disconnect but nothing past them is recorded, and the next turn's
    carry-on (`harness.graph.resume`) runs every step again outside DBOS."""
    observed = await observe_client_leaves_on_studio_ledger(monkeypatch)
    print(observed)
    assert observed["workflow_result"]["ledger_unavailable"] is True
    assert observed["run_when_workflow_ended"][0] == "running"
    assert observed["lines_after_next_turn"] == observed["lines_when_workflow_ended"] * 2


# X3 ------------------------------------------------------------------------


async def test_x3_progress_reaches_studio_within_1s_as_dev_frames(
    monkeypatch: pytest.MonkeyPatch, dbos_db: Path
) -> None:
    durable = await check_live_progress(monkeypatch, "durable")
    monkeypatch.undo()
    dev = await check_live_progress(monkeypatch, "dev")
    assert durable["activity"] == dev["activity"]


async def test_x3_control_progress_read_from_the_result(
    monkeypatch: pytest.MonkeyPatch, dbos_db: Path
) -> None:
    with pytest.raises(AssertionError, match="1 s or more") as caught:
        await check_live_progress(monkeypatch, "durable", BMode.BUFFERED)
    print(repr(caught.value))


# X4 ------------------------------------------------------------------------


async def test_x4_stop_cancels_the_workflow(monkeypatch: pytest.MonkeyPatch, dbos_db: Path) -> None:
    await check_stop(monkeypatch, BMode.DURABLE)


async def test_x4_control_local_cancel_leaves_the_workflow_pending(
    monkeypatch: pytest.MonkeyPatch, dbos_db: Path
) -> None:
    with pytest.raises(AssertionError, match="workflow left PENDING") as caught:
        await check_stop(monkeypatch, BMode.LOCAL_CANCEL)
    print(repr(caught.value))


async def test_x4_control_the_pending_workflow_reruns_on_relaunch(
    monkeypatch: pytest.MonkeyPatch, dbos_db: Path
) -> None:
    observed = await observe_relaunch_after_local_cancel(monkeypatch, dbos_db)
    print(observed)
    assert observed["status_after_relaunch"] == "SUCCESS"
    assert observed["lines_after_relaunch"] == [
        "get_data start",
        "get_data end",
        "aggregate start",
        "aggregate end",
    ]
    # The stopped run is failed, so the recovered workflow admits a second run.
    assert observed["ledger_runs"] == [("failed", 1), ("completed", 2)]


# X5 ------------------------------------------------------------------------


async def test_x5_records_equal_dev(monkeypatch: pytest.MonkeyPatch, dbos_db: Path) -> None:
    durable = await records(monkeypatch, "durable")
    dev = await records(monkeypatch, "dev")
    ((run,),) = [book["runs"] for book in durable["ledger"].values()]
    print(
        f"compared: run {run['status']} with {len(run['attempts'])} attempts, "
        f"{len(durable['published'])} published frame(s), hand-back keys {sorted(durable['hand_back'])}"
    )
    assert (
        run["status"] == "completed" and durable["published"] and "result" in durable["hand_back"]
    )
    assert differences(durable, dev) == []


async def test_x5_control_the_comparison_catches_a_changed_record(
    monkeypatch: pytest.MonkeyPatch, dbos_db: Path
) -> None:
    durable = await records(monkeypatch, "durable")
    dev = await records(monkeypatch, "dev")
    ((run,),) = [book["runs"] for book in durable["ledger"].values()]
    run["attempts"][0]["attempt"] = 2
    found = differences(durable, dev)
    print(found)
    assert found == [f"{path}: 2 != 1" for path in [_attempt_path(durable)]]


def _attempt_path(snapshot: dict) -> str:
    (scope,) = snapshot["ledger"]
    return f".ledger.{scope}.runs[0].attempts[0].attempt"
