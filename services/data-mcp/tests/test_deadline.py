import dataclasses
import math

import pytest
from data_mcp.deadline import (
    DIRECT_CALL_BUDGET,
    EXTERNAL_CALL_BUDGET,
    EXTERNAL_CALL_LIMIT_S,
    EXTERNAL_CALL_MARGIN_S,
    CallBudget,
    CallDeadline,
    current_call_budget,
    use_call_budget,
)
from data_mcp.settings import Settings


class _FakeClock:
    def __init__(self, now: float = 100.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


def test_remaining_counts_down_on_the_injected_clock() -> None:
    clock = _FakeClock()
    deadline = CallDeadline.after(10.0, clock=clock)

    assert deadline.expires_at == 110.0
    assert deadline.remaining() == 10.0

    clock.now += 4.0
    assert deadline.remaining() == 6.0


def test_remaining_goes_negative_once_the_deadline_passes() -> None:
    clock = _FakeClock()
    deadline = CallDeadline.after(1.0, clock=clock)

    clock.now += 3.0

    assert deadline.remaining() == -2.0


def test_default_clock_is_monotonic_real_time() -> None:
    deadline = CallDeadline.after(60.0)

    assert 0.0 < deadline.remaining() <= 60.0


def test_reserving_brings_the_deadline_forward_by_the_given_seconds() -> None:
    clock = _FakeClock()
    deadline = CallDeadline.after(30.0, clock=clock)

    reserved = deadline.reserving(10.0)

    assert reserved.expires_at == 120.0
    assert reserved.remaining() == 20.0
    assert reserved.clock is clock
    # The original is untouched -- `reserving` returns a new deadline.
    assert deadline.remaining() == 30.0


def test_the_direct_budget_keeps_the_harness_mounts_reserves() -> None:
    assert dataclasses.asdict(DIRECT_CALL_BUDGET) == {
        "deadline_s": None,
        "pre_mint_reserve_s": 50.0,
        "mint_timeout_s": 125.0,
        "min_mint_timeout_s": 20.0,
        "post_mint_reserve_s": 30.0,
    }


def test_the_direct_deadline_is_the_settings_deadline() -> None:
    """The direct budget sets no deadline of its own, so `Settings.get_data_deadline_s`
    is its one source, and its default keeps the 165s the reserves are sized for."""
    assert DIRECT_CALL_BUDGET.deadline_s is None
    assert Settings.model_fields["get_data_deadline_s"].default == 165.0


def test_the_external_deadline_ends_inside_the_combined_servers_call_limit() -> None:
    """`get_data`'s own refusal at its deadline must reach the caller before the
    combined server cancels the call at `EXTERNAL_CALL_LIMIT_S`."""
    budget = EXTERNAL_CALL_BUDGET

    assert EXTERNAL_CALL_LIMIT_S == 45.0
    assert EXTERNAL_CALL_MARGIN_S > 0
    assert budget.deadline_s == EXTERNAL_CALL_LIMIT_S - EXTERNAL_CALL_MARGIN_S == 42.0
    # The mint POST and the poll stop with the post-mint reserve in hand.
    assert budget.deadline_s - budget.post_mint_reserve_s == 32.0
    assert budget.mint_timeout_s == budget.deadline_s - budget.post_mint_reserve_s


def test_only_the_direct_budget_waits_out_a_synchronous_mint() -> None:
    assert DIRECT_CALL_BUDGET.waits_out_sync_mint is True
    assert EXTERNAL_CALL_BUDGET.waits_out_sync_mint is False


def test_the_current_budget_is_direct_until_an_entry_point_sets_another() -> None:
    assert current_call_budget() is DIRECT_CALL_BUDGET

    with use_call_budget(EXTERNAL_CALL_BUDGET):
        assert current_call_budget() is EXTERNAL_CALL_BUDGET

    assert current_call_budget() is DIRECT_CALL_BUDGET


def test_use_call_budget_resets_when_the_block_raises() -> None:
    with pytest.raises(RuntimeError), use_call_budget(EXTERNAL_CALL_BUDGET):
        raise RuntimeError

    assert current_call_budget() is DIRECT_CALL_BUDGET


@pytest.mark.parametrize("value", [0.0, -1.0, math.nan, math.inf])
@pytest.mark.parametrize("field", [f.name for f in dataclasses.fields(CallBudget)])
def test_a_budget_field_that_is_not_finite_and_positive_is_refused(
    field: str, value: float
) -> None:
    with pytest.raises(ValueError, match=field):
        dataclasses.replace(DIRECT_CALL_BUDGET, **{field: value})


@pytest.mark.parametrize(
    "changes",
    [
        # Pre-mint work would leave the mint less than its minimum.
        {"pre_mint_reserve_s": 49.0},
        # The cap would refuse every mint it allows.
        {"mint_timeout_s": 19.0},
        # Pre-mint work would get none of the deadline.
        {"deadline_s": 50.0},
    ],
)
def test_a_budget_whose_reserves_leave_a_phase_no_time_is_refused(
    changes: dict[str, float],
) -> None:
    with pytest.raises(ValueError):
        dataclasses.replace(DIRECT_CALL_BUDGET, **changes)
