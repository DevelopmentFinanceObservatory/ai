"""Each pass line passes on the durable node and fails on its control."""

from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from dbos import DBOS
from durable_checks import (
    STAGE_A,
    check_kill,
    check_live_progress,
    check_pause_loses_nothing,
    check_stop,
    check_tool_failure,
    measure_cancel_during_model_call,
    measure_external_cancel,
    observe_abandoned_workflow,
)
from durable_graph import DMode
from durable_planner import launch


@pytest.fixture
async def dbos_db(tmp_path: Path) -> AsyncIterator[Path]:
    path = tmp_path / "dbos.sqlite"
    launch(path, "test")
    yield path
    DBOS.destroy()


# D1 and D3 run in child processes and launch DBOS there.


async def test_d1_d3_kill_repeats_no_model_call(tmp_path: Path) -> None:
    result = await check_kill(tmp_path)
    print(result)


async def test_d1_fails_on_the_stage_a_node(tmp_path: Path) -> None:
    with pytest.raises(AssertionError, match="model called again") as caught:
        await check_kill(tmp_path, mode=STAGE_A)
    print(repr(caught.value))


async def test_d3_launch_recovery_beside_the_node_runs_once(tmp_path: Path) -> None:
    result = await check_kill(tmp_path, executors=("local", "local", "local"))
    print(result)


async def test_d3_fails_with_a_random_workflow_id(tmp_path: Path) -> None:
    with pytest.raises(AssertionError, match="model called again") as caught:
        await check_kill(tmp_path, mode=DMode.RANDOM_ID)
    print(repr(caught.value))


async def test_d2_live_progress(tmp_path: Path, dbos_db: Path) -> None:
    seconds = await check_live_progress(tmp_path / "s.db")
    print(f"first progress after {seconds:.3f} s")


async def test_d2_fails_without_a_live_handler(tmp_path: Path, dbos_db: Path) -> None:
    with pytest.raises(TimeoutError) as caught:
        await check_live_progress(tmp_path / "s.db", mode=DMode.NO_HANDLER)
    print(repr(caught.value))


async def test_d2_pause_loses_nothing(tmp_path: Path, dbos_db: Path) -> None:
    assert await check_pause_loses_nothing(tmp_path / "s.db") == 40


async def test_d4_stop(tmp_path: Path, dbos_db: Path) -> None:
    print(await check_stop(tmp_path / "s.db"))


async def test_d4_fails_when_only_the_local_task_is_cancelled(
    tmp_path: Path, dbos_db: Path
) -> None:
    with pytest.raises(AssertionError, match="workflow left PENDING") as caught:
        await check_stop(tmp_path / "s.db", mode=DMode.LOCAL_CANCEL)
    print(repr(caught.value))


async def test_d4_abandoned_workflow_runs_on_the_next_launch(tmp_path: Path, dbos_db: Path) -> None:
    result = await observe_abandoned_workflow(tmp_path / "s.db", dbos_db)
    print(result)
    assert result.status_after_stop == "PENDING"
    assert result.status_after_relaunch == "SUCCESS"
    assert "model call after 3 messages" in result.lines_after_relaunch


async def test_d4_model_call_in_flight_at_stop(tmp_path: Path, dbos_db: Path) -> None:
    result = await measure_cancel_during_model_call(tmp_path / "s.db")
    print(result)
    assert not result.model_call_completed


@pytest.mark.parametrize("where", ["model", "tool"])
async def test_d4_cancel_from_another_process(tmp_path: Path, dbos_db: Path, where: str) -> None:
    result = await measure_external_cancel(tmp_path / "s.db", dbos_db, where=where)
    print(result)
    assert result.status == "CANCELLED"


async def test_d5_tool_failure(tmp_path: Path, dbos_db: Path) -> None:
    print(await check_tool_failure(tmp_path / "s.db", dbos_db))


async def test_d5_fails_when_the_node_waits_on_the_queue_alone(
    tmp_path: Path, dbos_db: Path
) -> None:
    with pytest.raises(TimeoutError):
        await check_tool_failure(tmp_path / "s.db", dbos_db, mode=DMode.QUEUE_ONLY)
