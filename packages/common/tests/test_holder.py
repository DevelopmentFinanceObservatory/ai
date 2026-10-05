"""`Holder[T]`: lazy default build, explicit set, and rebuild-after-close."""

from __future__ import annotations

from common.holder import Holder


class _FakeResource:
    def __init__(self) -> None:
        self.closed = False

    async def aclose(self) -> None:
        self.closed = True


def test_get_with_nothing_set_lazily_builds_via_default_factory() -> None:
    holder = Holder(default_factory=_FakeResource)

    value = holder.get()

    assert isinstance(value, _FakeResource)
    assert holder.get() is value


def test_set_then_get_returns_the_set_value() -> None:
    holder = Holder(default_factory=_FakeResource)
    fake = _FakeResource()

    holder.set(fake)

    assert holder.get() is fake


async def test_aclose_closes_the_set_value_and_clears_the_holder() -> None:
    holder = Holder(default_factory=_FakeResource)
    fake = _FakeResource()
    holder.set(fake)

    await holder.aclose()

    assert fake.closed is True
    # Cleared, not just closed -- a later `get()` rebuilds rather than
    # handing back the closed instance.
    rebuilt = holder.get()
    assert rebuilt is not fake
    assert rebuilt.closed is False


async def test_aclose_twice_in_a_row_is_a_no_op() -> None:
    holder = Holder(default_factory=_FakeResource)

    await holder.aclose()
    await holder.aclose()
