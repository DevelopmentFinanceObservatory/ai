"""Tests for `common.row_budget`: claims, release, refusal, work that outlives its
caller, and consistency under threads and coroutines."""

from __future__ import annotations

import asyncio
import logging
import threading

import pytest
from common.row_budget import RowBudget, RowBudgetExhausted, row_budget, set_row_budget


def test_a_claim_holds_its_rows_until_its_block_exits() -> None:
    budget = RowBudget(max_rows=10)

    with budget.claim(4) as claim:
        assert claim.rows == 4
        assert budget.in_flight == 4

    assert budget.in_flight == 0
    assert claim.rows == 0


def test_a_claim_releases_its_rows_when_its_block_raises() -> None:
    budget = RowBudget(max_rows=10)

    with pytest.raises(RuntimeError, match="boom"), budget.claim(4):
        raise RuntimeError("boom")

    assert budget.in_flight == 0


def test_a_claim_that_does_not_fit_is_refused_with_the_rows_requested_and_free() -> None:
    budget = RowBudget(max_rows=10)

    with budget.claim(7), pytest.raises(RowBudgetExhausted) as refused:
        budget.claim(4)

    assert (refused.value.requested, refused.value.free, refused.value.max_rows) == (4, 3, 10)
    assert budget.in_flight == 0


def test_a_claim_that_exactly_fills_the_budget_fits() -> None:
    budget = RowBudget(max_rows=10)

    with budget.claim(7), budget.claim(3):
        assert budget.in_flight == 10


def test_a_released_claim_makes_room_for_the_next() -> None:
    budget = RowBudget(max_rows=10)

    with budget.claim(7), pytest.raises(RowBudgetExhausted):
        budget.claim(4)
    with budget.claim(4):
        assert budget.in_flight == 4


def test_a_claim_larger_than_the_budget_is_refused_when_nothing_else_is_in_flight() -> None:
    budget = RowBudget(max_rows=10)

    with pytest.raises(RowBudgetExhausted) as refused:
        budget.claim(11)

    assert (refused.value.requested, refused.value.free) == (11, 10)
    assert budget.in_flight == 0


def test_a_zero_row_claim_fits_a_full_budget() -> None:
    budget = RowBudget(max_rows=10)

    with budget.claim(10), budget.claim(0) as empty:
        assert empty.rows == 0


def test_a_claim_cannot_grow_past_the_budget_even_alone() -> None:
    budget = RowBudget(max_rows=10)

    with budget.claim(8) as claim, pytest.raises(RowBudgetExhausted):
        claim.extend(8)

    assert budget.in_flight == 0


def test_a_refused_extension_keeps_what_the_claim_already_held() -> None:
    budget = RowBudget(max_rows=10)

    with budget.claim(2), budget.claim(5) as claim:
        with pytest.raises(RowBudgetExhausted) as refused:
            claim.extend(4)
        assert refused.value.free == 3
        assert claim.rows == 5
        assert budget.in_flight == 7

    assert budget.in_flight == 0


def test_a_second_release_gives_back_nothing() -> None:
    budget = RowBudget(max_rows=10)
    claim = budget.claim(4)
    other = budget.claim(3)

    claim.release()
    claim.release()

    assert budget.in_flight == 3
    other.release()


async def test_work_held_through_a_claim_keeps_its_rows_after_its_deadline_passes() -> None:
    """A store write in a worker thread keeps running after the deadline stops the
    coroutine awaiting it; the rows stay claimed until the thread returns."""
    budget = RowBudget(max_rows=10)
    finish = threading.Event()
    try:
        with budget.claim(4) as claim:
            write = asyncio.to_thread(finish.wait)
            with pytest.raises(TimeoutError):
                async with asyncio.timeout(0.01):
                    await claim.hold_through(write)

        assert budget.in_flight == 4
    finally:
        finish.set()
    async with asyncio.timeout(5):
        while budget.in_flight:
            await asyncio.sleep(0.01)
    assert claim.rows == 0


async def test_work_held_through_a_claim_keeps_its_rows_after_its_caller_is_cancelled() -> None:
    budget = RowBudget(max_rows=10)
    finish = asyncio.Event()

    async def call() -> None:
        with budget.claim(4) as claim:
            await claim.hold_through(finish.wait())

    task = asyncio.create_task(call())
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert budget.in_flight == 4
    finish.set()
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert budget.in_flight == 0


async def test_work_that_fails_after_its_caller_is_cancelled_logs_its_exception(
    caplog: pytest.LogCaptureFixture,
) -> None:
    budget = RowBudget(max_rows=10)
    finish = asyncio.Event()

    async def upload() -> None:
        await finish.wait()
        raise RuntimeError("upload failed")

    async def call() -> None:
        with budget.claim(4) as claim:
            await claim.hold_through(upload())

    task = asyncio.create_task(call())
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    with caplog.at_level(logging.WARNING, logger="common.row_budget"):
        finish.set()
        async with asyncio.timeout(5):
            while budget.in_flight:
                await asyncio.sleep(0)

    # One record, this module's: asyncio's "exception was never retrieved" is not it.
    assert [record.name for record in caplog.records] == ["common.row_budget"]
    [record] = caplog.records
    assert "upload" in record.getMessage()
    assert record.exc_info is not None
    assert str(record.exc_info[1]) == "upload failed"


async def test_work_that_fails_while_its_caller_awaits_raises_to_it_and_logs_nothing(
    caplog: pytest.LogCaptureFixture,
) -> None:
    budget = RowBudget(max_rows=10)

    async def upload() -> None:
        raise RuntimeError("upload failed")

    with (
        caplog.at_level(logging.WARNING, logger="common.row_budget"),
        pytest.raises(RuntimeError, match="upload failed"),
        budget.claim(4) as claim,
    ):
        await claim.hold_through(upload())

    assert caplog.records == []
    assert budget.in_flight == 0


async def test_work_that_ends_inside_its_claim_releases_with_the_claim() -> None:
    budget = RowBudget(max_rows=10)

    async def write() -> str:
        return "written"

    with budget.claim(4) as claim:
        assert await claim.hold_through(write()) == "written"
        assert budget.in_flight == 4

    assert budget.in_flight == 0


def test_high_water_is_the_most_rows_held_at_once() -> None:
    budget = RowBudget(max_rows=10)

    with budget.claim(3), budget.claim(4):
        pass
    with budget.claim(5):
        pass

    assert budget.high_water == 7


@pytest.mark.parametrize("max_rows", [0, -1])
def test_a_budget_needs_at_least_one_row(max_rows: int) -> None:
    with pytest.raises(ValueError, match="at least 1"):
        RowBudget(max_rows=max_rows)


def test_a_negative_claim_is_rejected() -> None:
    with pytest.raises(ValueError, match="negative"):
        RowBudget(max_rows=10).claim(-1)


def test_claims_from_many_threads_never_exceed_the_budget_and_all_release() -> None:
    budget = RowBudget(max_rows=50)
    admitted: list[int] = []
    refused: list[int] = []
    start = threading.Barrier(8)

    def work() -> None:
        start.wait()
        for _ in range(2_000):
            try:
                with budget.claim(7):
                    admitted.append(1)
            except RowBudgetExhausted:
                refused.append(1)

    threads = [threading.Thread(target=work) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(admitted) + len(refused) == 16_000
    assert admitted
    assert budget.in_flight == 0
    assert budget.high_water <= 50


async def test_concurrent_coroutines_are_admitted_until_the_budget_is_full() -> None:
    budget = RowBudget(max_rows=10)
    release = asyncio.Event()

    async def call() -> bool:
        try:
            with budget.claim(3):
                await release.wait()
        except RowBudgetExhausted:
            return False
        return True

    calls = [asyncio.create_task(call()) for _ in range(5)]
    await asyncio.sleep(0)
    assert budget.in_flight == 9
    release.set()

    assert sorted(await asyncio.gather(*calls)) == [False, False, True, True, True]
    assert budget.in_flight == 0


def test_the_process_budget_is_built_from_settings_once(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FRAME_ROWS_IN_FLIGHT_MAX", "123")
    set_row_budget(None)

    budget = row_budget()

    assert budget.max_rows == 123
    assert row_budget() is budget


def test_an_installed_budget_replaces_the_process_budget() -> None:
    installed = RowBudget(max_rows=5)
    set_row_budget(installed)

    assert row_budget() is installed
