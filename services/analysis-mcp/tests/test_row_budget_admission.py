"""The frame row budget at analysis-mcp's acquire site: `resolve_input_frames` opens
every input, adds all their rows to the claim `holds_input_rows` opens for the verb call
in one step, and only then reads them; the claim lasts until the result's store write
ends."""

from __future__ import annotations

import asyncio
import threading
from typing import Any

import pytest
from analysis_mcp.aggregate_tool import aggregate
from analysis_mcp.fixtures import seed_fixture_store
from analysis_mcp.frame_ref_resolution import resolve_input_frame
from analysis_mcp.join_tool import join
from analysis_mcp.settings import Settings
from common.materialisation import (
    FrameExpired,
    FrameIntegrityError,
    FrameRef,
    FrameTooLarge,
    HeldFrame,
    InMemoryMaterialisationStore,
    set_store,
)
from common.row_budget import ROW_BUDGET_RETRY_AFTER_S, RowBudget, row_budget, set_row_budget


@pytest.fixture
async def seeded():
    store = InMemoryMaterialisationStore()
    handles = await seed_fixture_store(store)
    set_store(store)
    yield store, handles
    set_store(None)


async def _rows(store: InMemoryMaterialisationStore, handle: dict[str, str]) -> int:
    held = await store.get(FrameRef(**handle))
    assert held is not None
    return held[0].num_rows


async def test_join_is_refused_while_the_budget_is_full_and_holds_both_sides_once_admitted(
    seeded, monkeypatch: pytest.MonkeyPatch
) -> None:
    store, handles = seeded
    left, right = handles.join_pair
    left_rows, right_rows = await _rows(store, left), await _rows(store, right)
    in_flight_at_put: list[int] = []
    put = store.put

    async def recording_put(ref: FrameRef, **kwargs: Any) -> None:
        in_flight_at_put.append(row_budget().in_flight)
        await put(ref, **kwargs)

    monkeypatch.setattr(store, "put", recording_put)
    set_row_budget(RowBudget(max_rows=left_rows + right_rows))

    with row_budget().claim(1):
        refused = await join(frame_ref=left, right_frame_ref=right)
    admitted = await join(frame_ref=left, right_frame_ref=right)

    assert refused.is_error is True
    assert refused.structured_content == {
        "code": "service_unavailable",
        "retry_mode": "retry_after_delay",
        "message": refused.structured_content["message"],
        "fix": {
            "service": "analysis-mcp",
            "rows_requested": left_rows + right_rows,
            "rows_free": left_rows + right_rows - 1,
            "rows_in_flight_max": left_rows + right_rows,
        },
        "retry_after_s": ROW_BUDGET_RETRY_AFTER_S,
    }
    assert admitted.is_error is False
    assert in_flight_at_put == [left_rows + right_rows]
    assert row_budget().in_flight == 0


async def test_a_refused_join_reads_neither_side(seeded, monkeypatch: pytest.MonkeyPatch) -> None:
    store, handles = seeded
    left, right = handles.join_pair
    read_refs: list[FrameRef] = []
    open_frame = store.open

    async def recording_open(ref: FrameRef) -> HeldFrame | None:
        held = await open_frame(ref)
        assert held is not None

        async def read(*, max_rows: int) -> Any:
            read_refs.append(ref)
            return await held.read(max_rows=max_rows)

        return HeldFrame(num_rows=held.num_rows, read=read)

    monkeypatch.setattr(store, "open", recording_open)
    set_row_budget(RowBudget(max_rows=1))

    refused = await join(frame_ref=left, right_frame_ref=right)

    assert refused.structured_content["code"] == "service_unavailable"
    assert read_refs == []


async def test_a_join_holds_no_rows_while_it_waits_to_open_its_second_side(
    seeded, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A join that held its first side's rows while it waited for its second could
    hold them while every other such join did the same, and every second side would
    be refused."""
    store, handles = seeded
    left, right = handles.join_pair
    left_rows, right_rows = await _rows(store, left), await _rows(store, right)
    opening_right = asyncio.Event()
    open_right = asyncio.Event()
    open_frame = store.open

    async def slow_right_open(ref: FrameRef) -> HeldFrame | None:
        if ref == FrameRef(**right):
            opening_right.set()
            await open_right.wait()
        return await open_frame(ref)

    monkeypatch.setattr(store, "open", slow_right_open)
    call = asyncio.create_task(join(frame_ref=left, right_frame_ref=right))
    await opening_right.wait()

    assert row_budget().in_flight == 0
    open_right.set()
    assert (await call).is_error is False
    assert row_budget().high_water == left_rows + right_rows


async def test_a_cancelled_verb_keeps_its_rows_claimed_until_its_store_write_ends(
    seeded, monkeypatch: pytest.MonkeyPatch
) -> None:
    store, handles = seeded
    left, right = handles.join_pair
    left_rows, right_rows = await _rows(store, left), await _rows(store, right)
    writing = asyncio.Event()
    finish = threading.Event()
    put = store.put

    async def slow_put(ref: FrameRef, **kwargs: Any) -> None:
        writing.set()
        await asyncio.to_thread(finish.wait)
        await put(ref, **kwargs)

    monkeypatch.setattr(store, "put", slow_put)
    call = asyncio.create_task(join(frame_ref=left, right_frame_ref=right))
    try:
        await writing.wait()
        call.cancel()
        with pytest.raises(asyncio.CancelledError):
            await call

        assert row_budget().in_flight == left_rows + right_rows
    finally:
        finish.set()
    async with asyncio.timeout(5):
        while row_budget().in_flight:
            await asyncio.sleep(0.01)


async def test_a_verb_cancelled_mid_read_keeps_its_rows_claimed_until_the_read_ends(
    seeded, monkeypatch: pytest.MonkeyPatch
) -> None:
    store, handles = seeded
    left, right = handles.join_pair
    left_rows, right_rows = await _rows(store, left), await _rows(store, right)
    reading = asyncio.Event()
    finish = threading.Event()
    open_frame = store.open

    async def slow_open(ref: FrameRef) -> HeldFrame | None:
        held = await open_frame(ref)
        assert held is not None

        async def read(*, max_rows: int) -> Any:
            reading.set()
            await asyncio.to_thread(finish.wait)
            return await held.read(max_rows=max_rows)

        return HeldFrame(num_rows=held.num_rows, read=read)

    monkeypatch.setattr(store, "open", slow_open)
    call = asyncio.create_task(join(frame_ref=left, right_frame_ref=right))
    try:
        await reading.wait()
        call.cancel()
        with pytest.raises(asyncio.CancelledError):
            await call

        assert row_budget().in_flight == left_rows + right_rows
    finally:
        finish.set()
    async with asyncio.timeout(5):
        while row_budget().in_flight:
            await asyncio.sleep(0.01)


@pytest.mark.parametrize(
    ("error", "code"),
    [
        (FrameIntegrityError, "frame_integrity_failed"),
        (FrameTooLarge, "frame_too_large"),
        (FrameExpired, "frame_expired"),
    ],
)
async def test_a_store_failure_on_read_refuses_and_gives_its_rows_back(
    seeded, monkeypatch: pytest.MonkeyPatch, error: type[Exception], code: str
) -> None:
    """`HeldFrame.read` runs the store's checks past `open`, so its failures refuse
    exactly as the same failure on `open` does."""
    store, handles = seeded
    open_frame = store.open

    async def failing_open(ref: FrameRef) -> HeldFrame | None:
        held = await open_frame(ref)
        assert held is not None

        async def read(*, max_rows: int) -> Any:
            if error is FrameExpired:
                raise FrameExpired(ref)
            raise error(ref, reason="the stored object fails a check")

        return HeldFrame(num_rows=held.num_rows, read=read)

    monkeypatch.setattr(store, "open", failing_open)

    result = await aggregate(frame_ref=handles.plain_single_unit, aggregation="sum")

    assert result.structured_content["code"] == code
    assert row_budget().in_flight == 0


async def test_resolving_outside_a_verb_call_raises(seeded) -> None:
    _store, handles = seeded

    with pytest.raises(RuntimeError, match="holds_input_rows"):
        await resolve_input_frame(FrameRef(**handles.plain_single_unit), settings=Settings())
