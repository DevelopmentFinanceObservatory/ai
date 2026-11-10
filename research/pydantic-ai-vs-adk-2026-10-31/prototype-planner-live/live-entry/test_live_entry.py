# ruff: noqa: ANN001, ANN002, ANN003, ANN201, ANN202 -- fakes in a throwaway test file
"""Tests of the driver's pairing and retry rules, the tap, and the analysis."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import analyse
import driver
import pytest
import tap


def test_arm_order_alternates_by_pair_and_each_pair_has_both_arms() -> None:
    orders = [driver.pair_order(i, 1) for i in range(4)]
    assert orders[0] != orders[1] and orders[0] == orders[2]
    assert all(sorted(order) == ["adk", "pai"] for order in orders)
    assert driver.pair_order(0, 1) != driver.pair_order(0, 2)


def test_pending_keys_skip_done_entries() -> None:
    tasks = [SimpleNamespace(id="a"), SimpleNamespace(id="b")]
    every = driver.pending_keys(tasks, 2, done=set())
    assert len(every) == 8
    rest = driver.pending_keys(tasks, 2, done={every[0], every[5]})
    assert rest == [k for k in every if k not in (every[0], every[5])]


def test_only_a_429_error_is_a_rate_limit() -> None:
    assert driver.is_rate_limited({"error": "ClientError: 429 RESOURCE_EXHAUSTED"})
    assert not driver.is_rate_limited({"error": "ClientError: 500 INTERNAL"})
    assert not driver.is_rate_limited({"error": None})


async def test_backoff_reruns_a_rate_limited_entry_and_counts_it(monkeypatch) -> None:
    attempts = [
        {"error": "ClientError: 429", "wall_s": 1.0},
        {"error": None, "wall_s": 2.0},
    ]

    async def fake_run_arm(*_args, **_kwargs):
        return attempts.pop(0)

    async def no_sleep(_seconds: float) -> None:
        return None

    monkeypatch.setattr(driver, "run_arm", fake_run_arm)
    monkeypatch.setattr(driver.asyncio, "sleep", no_sleep)
    record = await driver.run_with_backoff(
        "pai", None, {}, prefix_cache=None, timeout_s=1, retries=4
    )
    assert record["retries"] == 1 and record["error"] is None
    assert record["discarded"][0]["wall_s"] == 1.0


class _Models:
    async def generate_content_stream(self, *, model, contents, config=None):
        async def chunks():
            yield SimpleNamespace(usage_metadata=None)
            yield SimpleNamespace(
                usage_metadata=SimpleNamespace(
                    prompt_token_count=10,
                    cached_content_token_count=4,
                    candidates_token_count=2,
                    thoughts_token_count=None,
                )
            )

        return chunks()

    async def generate_content(self, *, model, contents, config=None):
        raise RuntimeError("boom")


async def test_tap_records_usage_and_errors_and_restores_the_class() -> None:
    original = _Models.__dict__["generate_content_stream"]
    with tap.installed(models_cls=_Models), tap.collecting() as calls:
        models = _Models()
        stream = await models.generate_content_stream(
            model="m", contents=[], config={"cached_content": "caches/1"}
        )
        assert [c async for c in stream] != []
        with pytest.raises(RuntimeError):
            await models.generate_content(model="m", contents=[])
    assert _Models.__dict__["generate_content_stream"] is original
    ok, failed = calls
    assert (ok["prompt_tokens"], ok["cached_tokens"], ok["output_tokens"]) == (10, 4, 2)
    assert ok["cached_content"] and ok["first_chunk_s"] is not None
    assert failed["error"]["class"] == "RuntimeError" and failed["prompt_tokens"] is None


async def test_tap_gives_each_concurrent_entry_its_own_calls() -> None:
    async def one() -> int:
        with tap.collecting() as calls:
            with pytest.raises(RuntimeError):
                await _Models().generate_content(model="m", contents=[])
            await asyncio.sleep(0)
            return len(calls)

    with tap.installed(models_cls=_Models):
        assert await asyncio.gather(one(), one()) == [1, 1]


def record(stop: str = "accepted", **over):
    base = {
        "task": "t",
        "rep": 1,
        "stop_reason": stop,
        "error": None,
        "surface_misses": [],
        "tap": [],
    }
    return base | over


def test_failure_classes_leave_out_ordinary_endings_and_429s() -> None:
    assert analyse.failure_classes(record("asked_user")) == {}
    limited = {"class": "ClientError", "code": 429, "message": "x"}
    tapped = [{"error": limited, "first_chunk_s": None}]
    assert analyse.failure_classes(record(tap=tapped)) == {}
    other = {"class": "ServerError", "code": 503, "message": "down"}
    found = analyse.failure_classes(
        record("error", error="ValueError: bad\ntrace", tap=[{"error": other, "first_chunk_s": 1}])
    )
    assert set(found) == {"ended: error", "error: ValueError", "call failed: ServerError 503"}


def test_sign_test_and_interval_are_exact_and_deterministic() -> None:
    assert analyse.exact_sign_p(0, 0) == 1.0
    assert analyse.exact_sign_p(5, 0) == pytest.approx(0.0625)
    entries = {
        "a": [analyse.Entry("a", 1, True, True, 2, 2, 100, 50, 5, 10.0, (1.0,), False)],
        "b": [analyse.Entry("b", 1, False, True, 3, 3, 200, 100, 7, 20.0, (2.0,), False)],
    }
    stat = analyse.STATS[0]
    first = analyse.paired(stat, entries, entries)
    assert first.diff[0] == 0 and first == analyse.paired(stat, entries, entries)
