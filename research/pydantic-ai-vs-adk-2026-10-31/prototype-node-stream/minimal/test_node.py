"""Each pass line passes on the live node and fails on its control."""

from pathlib import Path

import pytest
from checks import (
    check_live_progress,
    check_pause_loses_nothing,
    check_resume_continues,
    check_stop,
    check_survives_sigkill,
    check_tool_failure,
)
from graph import Mode


async def test_n1_live_progress(tmp_path: Path) -> None:
    seconds = await check_live_progress(tmp_path / "s.db")
    print(f"first progress after {seconds:.3f} s")


async def test_n1_fails_when_reports_relay_after_the_await(tmp_path: Path) -> None:
    with pytest.raises(TimeoutError):
        await check_live_progress(tmp_path / "s.db", mode=Mode.AFTER_AWAIT)


async def test_n2_pause_loses_nothing(tmp_path: Path) -> None:
    assert await check_pause_loses_nothing(tmp_path / "s.db") == 40


async def test_n2_fails_when_a_pause_drops_queued_reports(tmp_path: Path) -> None:
    with pytest.raises(AssertionError, match="got progress"):
        await check_pause_loses_nothing(tmp_path / "s.db", mode=Mode.AFTER_AWAIT)


async def test_n3_resume_continues(tmp_path: Path) -> None:
    result = await check_resume_continues(tmp_path / "s.db")
    print(
        f"model calls before the pause {result.model_calls_before}, after {result.model_calls_after}"
    )


async def test_n3_fails_when_the_planner_restarts(tmp_path: Path) -> None:
    with pytest.raises(AssertionError, match="model called again"):
        await check_resume_continues(tmp_path / "s.db", mode=Mode.RESTART)


async def test_n4_survives_sigkill(tmp_path: Path) -> None:
    result = await check_survives_sigkill(tmp_path)
    print(result)
    assert result.repeated_model_calls == ["model call after 5 messages"]
    assert result.repeated_tool_lines == ["propose_plan call=2 start"]
    assert result.repeated_progress == [(2, 1)]


async def test_n4_fails_when_the_conversation_lives_in_process_memory(tmp_path: Path) -> None:
    with pytest.raises(AssertionError, match="repeated from before the pause"):
        await check_survives_sigkill(tmp_path, mode=Mode.PROCESS_MEMORY)


async def test_n5_stop(tmp_path: Path) -> None:
    await check_stop(tmp_path / "s.db")


async def test_n5_fails_when_the_planner_task_is_orphaned(tmp_path: Path) -> None:
    with pytest.raises(AssertionError, match="tasks left running"):
        await check_stop(tmp_path / "s.db", mode=Mode.ORPHAN)


async def test_n6_tool_failure(tmp_path: Path) -> None:
    print(await check_tool_failure(tmp_path))


async def test_n6_fails_when_the_node_waits_on_the_queue_alone(tmp_path: Path) -> None:
    with pytest.raises(TimeoutError):
        await check_tool_failure(tmp_path, mode=Mode.QUEUE_ONLY)
