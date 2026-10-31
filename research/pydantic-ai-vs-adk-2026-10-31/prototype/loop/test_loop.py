"""Each check passes as built and fails on its deliberately broken variant."""

from pathlib import Path

import pytest
from checks import (
    ReplayingLoop,
    check_amend_and_stale_approval,
    check_deploy_safe_resume,
    check_live_progress,
    check_one_resumer_wins,
    check_pause_loses_nothing,
    check_survives_sigkill,
)


async def test_l1_live_progress(tmp_path: Path) -> None:
    await check_live_progress(tmp_path / "t.db")


async def test_l1_fails_when_planner_events_are_buffered(tmp_path: Path) -> None:
    with pytest.raises(TimeoutError):
        await check_live_progress(tmp_path / "t.db", buffer_planner_events=True)


async def test_l2_pause_loses_nothing(tmp_path: Path) -> None:
    await check_pause_loses_nothing(tmp_path / "t.db")


async def test_l2_fails_when_the_queue_is_not_drained(tmp_path: Path) -> None:
    with pytest.raises(AssertionError, match="progress frames"):
        await check_pause_loses_nothing(tmp_path / "t.db", drain_before_pause=False)


async def test_l3_survives_sigkill(tmp_path: Path) -> None:
    result = await check_survives_sigkill(tmp_path)
    print(f"plan hash {result.plan_hash}; fields that differ raw: {result.differing_keys}")
    assert result.differing_keys <= {"timestamp", "run_id"}


async def test_l3_fails_when_the_planner_conversation_write_is_lost(tmp_path: Path) -> None:
    with pytest.raises(AssertionError, match="planner differs"):
        await check_survives_sigkill(tmp_path, drop_planner_in_killed_process=True)


async def test_l4_one_resumer_wins(tmp_path: Path) -> None:
    await check_one_resumer_wins(tmp_path / "t.db")


async def test_l4_fails_without_compare_and_swap(tmp_path: Path) -> None:
    with pytest.raises(AssertionError, match="outcomes"):
        await check_one_resumer_wins(tmp_path / "t.db", unconditional_store=True)


async def test_l5_deploy_safe_resume(tmp_path: Path) -> None:
    await check_deploy_safe_resume(tmp_path / "t.db")


async def test_l5_fails_when_resume_replays_the_opening_path(tmp_path: Path) -> None:
    with pytest.raises(AssertionError, match="ran on resume"):
        await check_deploy_safe_resume(tmp_path / "t.db", resume_loop_cls=ReplayingLoop)


async def test_amend_and_stale_approval(tmp_path: Path) -> None:
    await check_amend_and_stale_approval(tmp_path / "t.db")
