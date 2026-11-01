"""`InMemoryMaterialisationStore`: idempotent put, no session gate on get,
retention absolute from first write, and protocol conformance.

Every test below stands a plain sentinel object in for a `Frame` --
`packages/common` has no runtime frame-grammar dependency and must not
acquire one just to test a store that only ever holds and returns whatever
reference it's given, never inspects it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from common.materialisation import (
    DEFAULT_MINT_OPTIONS,
    FrameExpired,
    FrameIntegrityError,
    FrameListing,
    FrameRef,
    FrameRendering,
    FrameTooLarge,
    HeldFrame,
    InMemoryMaterialisationStore,
    MintRecord,
    RecordUnread,
    RenderingUnknown,
    dataset_pair,
    decode_mint_record,
    encode_mint_record,
    frame_rendering,
    record_bare_read,
    root_rendering,
    set_store,
    store_is_set,
)
from common.materialisation.memory_store import FRAME_TTL
from common.materialisation.store import MaterialisationStore
from pydantic import ValidationError


@dataclass(frozen=True)
class _FakeTrace:
    upstream_trace: dict[str, Any]


class _FakeStore:
    """The minimal `MaterialisationStore` implementation this test needs."""

    def __init__(self) -> None:
        self._entries: dict[tuple[str, str], tuple[object, _FakeTrace]] = {}

    async def put(self, ref: FrameRef, *, frame: object, trace: _FakeTrace, writer: str) -> None:
        self._entries[(ref.spec_id, ref.content_version)] = (frame, trace)

    async def get(self, ref: FrameRef) -> tuple[object, _FakeTrace] | None:
        return self._entries.get((ref.spec_id, ref.content_version))

    async def open(self, ref: FrameRef) -> HeldFrame | None:
        del ref
        return None

    async def aclose(self) -> None:
        pass


def _ref() -> FrameRef:
    return FrameRef(spec_id="spec_" + "a" * 32, content_version="cv_" + "b" * 32)


def test_fake_store_satisfies_the_protocol() -> None:
    store: MaterialisationStore = _FakeStore()
    assert store is not None


async def test_put_then_get_round_trips() -> None:
    store = InMemoryMaterialisationStore()
    ref = _ref()
    frame = object()
    trace = _FakeTrace(upstream_trace={})

    await store.put(ref, frame=frame, trace=trace, writer="writer-a")
    result = await store.get(ref)

    assert result is not None
    got_frame, got_trace = result
    assert got_frame is frame
    assert got_trace is trace


async def test_get_returns_none_for_an_unheld_ref() -> None:
    store = InMemoryMaterialisationStore()
    assert await store.get(_ref()) is None


async def test_put_is_idempotent_a_second_writer_does_not_overwrite_frame_or_trace() -> None:
    store = InMemoryMaterialisationStore()
    ref = _ref()
    first_frame, first_trace = object(), _FakeTrace(upstream_trace={"n": 1})
    second_frame, second_trace = object(), _FakeTrace(upstream_trace={"n": 2})

    await store.put(ref, frame=first_frame, trace=first_trace, writer="writer-a")
    await store.put(ref, frame=second_frame, trace=second_trace, writer="writer-b")

    result = await store.get(ref)
    assert result is not None
    got_frame, got_trace = result
    assert got_frame is first_frame
    assert got_trace is first_trace


async def test_put_of_an_already_held_key_does_not_refresh_created_at() -> None:
    """Retention is absolute from first write: a second `put` of a live key keeps the
    original `created_at`, so the pair still ages out on the schedule the *first*
    write set, not one extended by later re-puts."""
    ref = _ref()
    base = datetime(2026, 1, 1, tzinfo=UTC)
    second_put = base + timedelta(days=15)
    past_original_ttl = base + FRAME_TTL + timedelta(seconds=1)
    clock = iter(
        [
            base,  # first put: created_at
            second_put,  # second put: live-entry check only, created_at untouched
            past_original_ttl,  # get(): past the boundary the first write set
        ]
    )
    store = InMemoryMaterialisationStore(now=lambda: next(clock))

    await store.put(ref, frame=object(), trace=_FakeTrace(upstream_trace={}), writer="writer-a")
    await store.put(ref, frame=object(), trace=_FakeTrace(upstream_trace={}), writer="writer-a")

    with pytest.raises(FrameExpired):
        await store.get(ref)


async def test_get_does_not_refresh_created_at_so_a_pair_still_ages_out_on_schedule() -> None:
    ref = _ref()
    base = datetime(2026, 1, 1, tzinfo=UTC)
    first_read = base + FRAME_TTL - timedelta(days=1)
    second_read = base + FRAME_TTL + timedelta(days=1)
    clock = iter(
        [
            base,  # put()
            first_read,  # get() #1: just before the original TTL, succeeds
            second_read,  # get() #2: past the original TTL -- the read above did not
            # extend it, so this one still raises
        ]
    )
    store = InMemoryMaterialisationStore(now=lambda: next(clock))
    await store.put(ref, frame=object(), trace=_FakeTrace(upstream_trace={}), writer="writer-a")

    result = await store.get(ref)
    assert result is not None

    with pytest.raises(FrameExpired):
        await store.get(ref)


async def test_lazy_expiry_raises_frame_expired_with_no_intervening_touch() -> None:
    ref = _ref()
    clock = iter(
        [
            datetime(2026, 1, 1, tzinfo=UTC),  # put()
            datetime(2026, 1, 1, tzinfo=UTC) + FRAME_TTL + timedelta(seconds=1),  # get()
        ]
    )
    store = InMemoryMaterialisationStore(now=lambda: next(clock))
    await store.put(ref, frame=object(), trace=_FakeTrace(upstream_trace={}), writer="writer-a")

    with pytest.raises(FrameExpired) as exc_info:
        await store.get(ref)

    assert exc_info.value.ref == ref


async def test_two_consecutive_reads_of_an_expired_pair_both_raise_frame_expired() -> None:
    ref = _ref()
    base = datetime(2026, 1, 1, tzinfo=UTC)
    past_ttl = base + FRAME_TTL + timedelta(seconds=1)
    clock = iter([base, past_ttl, past_ttl])
    store = InMemoryMaterialisationStore(now=lambda: next(clock))
    await store.put(ref, frame=object(), trace=_FakeTrace(upstream_trace={}), writer="writer-a")

    with pytest.raises(FrameExpired):
        await store.get(ref)
    with pytest.raises(FrameExpired):
        await store.get(ref)


async def test_a_put_on_an_expired_key_heals_it() -> None:
    ref = _ref()
    base = datetime(2026, 1, 1, tzinfo=UTC)
    past_ttl = base + FRAME_TTL + timedelta(seconds=1)
    clock = iter([base, past_ttl, past_ttl, past_ttl, past_ttl])
    store = InMemoryMaterialisationStore(now=lambda: next(clock))
    await store.put(ref, frame=object(), trace=_FakeTrace(upstream_trace={}), writer="writer-a")

    with pytest.raises(FrameExpired):
        await store.get(ref)

    await store.put(ref, frame=object(), trace=_FakeTrace(upstream_trace={}), writer="writer-a")
    result = await store.get(ref)

    assert result is not None


async def test_a_read_just_inside_the_ttl_still_succeeds() -> None:
    ref = _ref()
    just_inside = datetime(2026, 1, 1, tzinfo=UTC) + FRAME_TTL - timedelta(seconds=1)
    clock = iter(
        [
            datetime(2026, 1, 1, tzinfo=UTC),  # put()
            just_inside,  # get(): the TTL comparison
        ]
    )
    store = InMemoryMaterialisationStore(now=lambda: next(clock))
    frame, trace = object(), _FakeTrace(upstream_trace={})
    await store.put(ref, frame=frame, trace=trace, writer="writer-a")

    result = await store.get(ref)

    assert result is not None
    got_frame, got_trace = result
    assert got_frame is frame
    assert got_trace is trace


async def test_aclose_is_a_no_op() -> None:
    store = InMemoryMaterialisationStore()
    await store.aclose()


# max_entries: bounded growth


def _ref_n(n: int) -> FrameRef:
    return FrameRef(spec_id=f"spec_{'a' * 31}{n}", content_version=_ref().content_version)


async def test_a_put_of_a_new_key_at_capacity_evicts_the_least_recently_touched() -> None:
    store = InMemoryMaterialisationStore(max_entries=2)
    ref_a, ref_b, ref_c = _ref_n(0), _ref_n(1), _ref_n(2)

    await store.put(ref_a, frame=object(), trace=_FakeTrace(upstream_trace={}), writer="w")
    await store.put(ref_b, frame=object(), trace=_FakeTrace(upstream_trace={}), writer="w")
    # Touching A leaves B as the least-recently-touched entry.
    await store.get(ref_a)
    await store.put(ref_c, frame=object(), trace=_FakeTrace(upstream_trace={}), writer="w")

    assert len(store._entries) == 2
    assert await store.get(ref_a) is not None
    assert await store.get(ref_b) is None
    assert await store.get(ref_c) is not None


async def test_an_evicted_key_reads_back_as_a_plain_miss_not_frame_expired() -> None:
    store = InMemoryMaterialisationStore(max_entries=1)
    ref_a, ref_b = _ref_n(0), _ref_n(1)

    await store.put(ref_a, frame=object(), trace=_FakeTrace(upstream_trace={}), writer="w")
    await store.put(ref_b, frame=object(), trace=_FakeTrace(upstream_trace={}), writer="w")

    assert await store.get(ref_a) is None


async def test_a_re_put_of_an_already_held_key_never_counts_as_new_even_at_capacity() -> None:
    store = InMemoryMaterialisationStore(max_entries=1)
    ref_a = _ref_n(0)
    frame = object()
    trace = _FakeTrace(upstream_trace={})

    await store.put(ref_a, frame=frame, trace=trace, writer="w1")
    await store.put(ref_a, frame=object(), trace=_FakeTrace(upstream_trace={}), writer="w2")

    assert len(store._entries) == 1
    result = await store.get(ref_a)
    assert result is not None
    got_frame, got_trace = result
    assert got_frame is frame
    assert got_trace is trace


async def test_expired_entries_are_dropped_before_a_live_one_is_evicted() -> None:
    ref_a, ref_b = _ref_n(0), _ref_n(1)
    base = datetime(2026, 1, 1, tzinfo=UTC)
    past_ttl = base + FRAME_TTL + timedelta(seconds=1)
    clock = iter(
        [
            base,  # put(a)
            past_ttl,  # put(b): _drop_expired's own read
            past_ttl,  # put(b): the new entry's created_at
        ]
    )
    store = InMemoryMaterialisationStore(now=lambda: next(clock), max_entries=1)
    await store.put(ref_a, frame=object(), trace=_FakeTrace(upstream_trace={}), writer="w")

    await store.put(ref_b, frame=object(), trace=_FakeTrace(upstream_trace={}), writer="w")

    # `a` aged out and was dropped proactively -- `b` is the only entry, not evicted.
    assert len(store._entries) == 1
    assert (ref_b.spec_id, ref_b.content_version) in store._entries


async def test_drop_expired_finds_an_expired_entry_moved_out_of_age_order_by_a_touch() -> None:
    """`created_at` is fixed at an entry's first write; its position in `_entries`
    moves on every touch. A touch can therefore put an *expired* entry at the end
    (most-recently-touched) while a *live* entry that was never touched again sits at
    the front -- `_drop_expired` has to find the expired one by scanning, not by
    stopping at the first live entry it meets front-to-back, or it would leave the
    expired entry unswept and evict the live one in its place instead.
    """
    ref_a, ref_b, ref_c = _ref_n(0), _ref_n(1), _ref_n(2)
    t0 = datetime(2026, 1, 1, tzinfo=UTC)
    t_b_created = t0 + timedelta(minutes=1)
    t_touch_a = t0 + timedelta(minutes=2)
    # Past a's own TTL boundary (t0 + FRAME_TTL) but short of b's (t_b_created + FRAME_TTL).
    t_put_c = t0 + FRAME_TTL + timedelta(seconds=1)
    clock = iter(
        [
            t0,  # put(a): created_at
            t_b_created,  # put(b): created_at
            t_touch_a,  # get(a): TTL comparison -- not expired yet, so this touches a,
            # moving it to the end of `_entries` ahead of the still-untouched b
            t_put_c,  # put(c): _drop_expired's own `now`
            t_put_c,  # put(c): the new entry's created_at
        ]
    )
    store = InMemoryMaterialisationStore(now=lambda: next(clock), max_entries=2)
    await store.put(ref_a, frame=object(), trace=_FakeTrace(upstream_trace={}), writer="w")
    await store.put(ref_b, frame=object(), trace=_FakeTrace(upstream_trace={}), writer="w")
    await store.get(ref_a)

    await store.put(ref_c, frame=object(), trace=_FakeTrace(upstream_trace={}), writer="w")

    # `a` is expired and dropped despite sitting at the back; `b`, still live, survives
    # rather than being evicted in `a`'s place.
    assert set(store._entries) == {
        (ref_b.spec_id, ref_b.content_version),
        (ref_c.spec_id, ref_c.content_version),
    }


async def test_max_entries_is_respected_across_a_mixed_put_and_get_sequence() -> None:
    store = InMemoryMaterialisationStore(max_entries=3)
    refs = [_ref_n(n) for n in range(10)]

    for i, ref in enumerate(refs):
        await store.put(ref, frame=object(), trace=_FakeTrace(upstream_trace={}), writer="w")
        if i > 0:
            await store.get(refs[i - 1])
        assert len(store._entries) <= 3


@dataclass(frozen=True)
class _SizedFrame:
    num_rows: int


async def test_open_reports_the_held_frames_rows_and_reads_to_it_and_its_trace() -> None:
    store = InMemoryMaterialisationStore()
    ref = _ref()
    frame = _SizedFrame(num_rows=3)
    trace = _FakeTrace(upstream_trace={})
    await store.put(ref, frame=frame, trace=trace, writer="writer-a")

    held = await store.open(ref)

    assert held is not None
    assert held.num_rows == 3
    read_frame, read_trace = await held.read(max_rows=3)
    assert read_frame is frame
    assert read_trace == trace
    with pytest.raises(FrameTooLarge, match="has 3 rows, over this read's 2-row limit"):
        await held.read(max_rows=2)


async def test_open_returns_none_for_a_miss_and_raises_for_an_expired_pair() -> None:
    now = datetime(2026, 1, 1, tzinfo=UTC)
    store = InMemoryMaterialisationStore(now=lambda: now)
    ref = _ref()

    assert await store.open(ref) is None

    await store.put(
        ref, frame=_SizedFrame(num_rows=1), trace=_FakeTrace(upstream_trace={}), writer="w"
    )
    now += FRAME_TTL + timedelta(seconds=1)
    with pytest.raises(FrameExpired):
        await store.open(ref)


def _mint_record(ref: FrameRef, *, scope: dict[str, Any] | None = None) -> MintRecord:
    return MintRecord(
        frame=ref,
        dataset=FrameRef(spec_id=ref.spec_id, content_version="cv_" + "d" * 32),
        options={"layout": "frame", "columns": ["observationAbout_type"]},
        scope=scope or {},
    )


async def test_get_trace_returns_the_stored_trace_and_none_for_a_miss() -> None:
    store = InMemoryMaterialisationStore()
    ref = _ref()
    trace = _FakeTrace(upstream_trace={"n": 1})

    assert await store.get_trace(ref) is None

    await store.put(ref, frame=object(), trace=trace, writer="writer-a")
    assert await store.get_trace(ref) is trace


async def test_a_mint_record_is_kept_without_its_frame_and_the_first_writer_wins() -> None:
    store = InMemoryMaterialisationStore()
    ref = _ref()
    first = _mint_record(ref)

    assert await store.get_mint_record(ref) is None

    await store.put_mint_record(first)
    await store.put_mint_record(_mint_record(ref, scope={"window": {"kind": "all"}}))

    assert await store.get_mint_record(ref) == first
    assert await store.get(ref) is None


async def test_a_mint_record_outlives_its_frames_eviction() -> None:
    store = InMemoryMaterialisationStore(max_entries=1)
    ref = _ref()
    other = FrameRef(spec_id="spec_" + "c" * 32, content_version="cv_" + "b" * 32)
    await store.put(ref, frame=object(), trace=_FakeTrace(upstream_trace={}), writer="w")
    await store.put_mint_record(_mint_record(ref))

    await store.put(other, frame=object(), trace=_FakeTrace(upstream_trace={}), writer="w")

    assert await store.get(ref) is None
    assert await store.get_mint_record(ref) == _mint_record(ref)


def test_a_mint_record_encodes_to_the_same_bytes_whatever_order_its_members_were_given() -> None:
    ref = _ref()
    dataset = FrameRef(spec_id=ref.spec_id, content_version="cv_" + "d" * 32)
    one = MintRecord(
        frame=ref, dataset=dataset, options={"layout": "frame", "bundle": False}, scope={}
    )
    other = MintRecord(
        scope={}, options={"bundle": False, "layout": "frame"}, dataset=dataset, frame=ref
    )

    assert encode_mint_record(one) == encode_mint_record(other)
    assert decode_mint_record(encode_mint_record(one)) == one


def test_a_stored_mint_record_keeps_its_groupby_in_the_order_written() -> None:
    """A stored recipe root decodes as written, so the normal form it was published
    under does not change under it, whatever order its `by` holds."""
    ref = _ref()
    dataset = FrameRef(spec_id=ref.spec_id, content_version="cv_" + "d" * 32)
    groupby = {"by": ["time_period", "crsPurpose"], "op": "sum", "levels": {"crsPurpose": 0}}
    record = MintRecord(frame=ref, dataset=dataset, options={"groupby": groupby})

    assert decode_mint_record(encode_mint_record(record)).options["groupby"] == groupby


@pytest.mark.parametrize(
    "groupby", [{"op": "sum"}, {"by": "time_period"}, {"by": [1, "time_period"]}]
)
def test_a_mint_record_whose_groupby_names_no_list_of_axes_is_refused(
    groupby: dict[str, object],
) -> None:
    ref = _ref()
    with pytest.raises(ValidationError):
        MintRecord(frame=ref, dataset=ref, options={"groupby": groupby})


def test_a_stored_mint_record_with_a_member_this_process_never_declared_still_decodes() -> None:
    record = _mint_record(_ref())
    stored = encode_mint_record(record).replace(b'{"dataset"', b'{"added_later":1,"dataset"', 1)

    assert decode_mint_record(stored) == record


def test_a_mint_record_stored_without_a_stat_var_count_reads_as_unknown() -> None:
    record = _mint_record(_ref())
    counted = record.model_copy(update={"stat_var_count": 3})
    stored = encode_mint_record(counted).replace(b',"stat_var_count":3', b"", 1)

    assert decode_mint_record(encode_mint_record(counted)) == counted
    assert decode_mint_record(stored) == record
    assert record.stat_var_count is None


def test_a_mint_records_enrichment_stamp_round_trips_and_stays_out_of_an_unstamped_record() -> None:
    record = _mint_record(_ref())
    stamped = record.model_copy(update={"enrichment_stamp": "stamp-a"})

    assert b"enrichment_stamp" not in encode_mint_record(record)
    assert decode_mint_record(encode_mint_record(stamped)) == stamped
    assert decode_mint_record(encode_mint_record(record)).enrichment_stamp is None


async def test_list_frames_is_oldest_first_by_first_write_whatever_was_read_since() -> None:
    """A read moves an entry in the eviction order and leaves its place in the listing,
    and an expired entry is not listed."""
    clock = [datetime(2026, 9, 1, tzinfo=UTC)]
    store = InMemoryMaterialisationStore(now=lambda: clock[0])
    refs = [
        FrameRef(spec_id="spec_" + "a" * 32, content_version=f"cv_{digit * 32}") for digit in "321"
    ]
    written = []
    for days, ref in zip((0, 20, 25), refs, strict=True):
        clock[0] = datetime(2026, 9, 1, tzinfo=UTC) + timedelta(days=days)
        written.append(clock[0])
        await store.put(ref, frame=object(), trace=_FakeTrace(upstream_trace={}), writer="w")
    await store.get(refs[1])

    assert await store.list_frames() == [
        FrameListing(ref=ref, created_at=at) for ref, at in zip(refs, written, strict=True)
    ]

    clock[0] = written[0] + FRAME_TTL + timedelta(seconds=1)
    assert [listing.ref for listing in await store.list_frames()] == refs[1:]


_FRAME = FrameRef(spec_id="spec_" + "a" * 32, content_version="cv_" + "f" * 32)
_DATASET = FrameRef(spec_id="spec_" + "b" * 32, content_version="cv_" + "d" * 32)


@pytest.mark.parametrize(
    ("record", "pair", "expected"),
    [
        (MintRecord(frame=_FRAME, dataset=_DATASET, options={}), _FRAME, _DATASET),
        (MintRecord(frame=_FRAME, dataset=_FRAME, options={}), _FRAME, _FRAME),
        (None, _FRAME, _FRAME),
        (None, FrameRef(spec_id="ops_" + "a" * 32, content_version="ocv_" + "f" * 32), None),
        (None, FrameRef.model_construct(spec_id="spec_1", content_version="cv_1"), None),
    ],
    ids=["record-names-another-pair", "record-names-its-own", "no-record", "ops", "not-a-pair"],
)
async def test_dataset_pair_maps_a_frame_to_the_dataset_its_record_names(
    record: MintRecord | None, pair: FrameRef, expected: FrameRef | None
) -> None:
    store = InMemoryMaterialisationStore()
    if record is not None:
        await store.put_mint_record(record)
    set_store(store)

    try:
        mapped = await dataset_pair(
            pair.spec_id, pair.content_version, logger=logging.getLogger("caller")
        )
    finally:
        set_store(None)

    assert mapped == ((expected or pair).spec_id, (expected or pair).content_version)


_READ_AS: dict[str, Any] = {"format": "parquet", "bundle": False, "layout": "frame"}


@pytest.mark.parametrize(
    ("options", "rendering", "summed", "default_rows"),
    [
        (_READ_AS, {"without_party_totals": False, "columns": []}, False, True),
        (
            {**_READ_AS, "columns": ["b", "a", "a"]},
            {"without_party_totals": False, "columns": ["a", "b"]},
            False,
            True,
        ),
        (
            {**_READ_AS, "without_party_totals": True},
            {"without_party_totals": True, "columns": []},
            False,
            False,
        ),
        (
            {**_READ_AS, "groupby": {"by": ["sector"]}},
            {"without_party_totals": False, "columns": [], "groupby": {"by": ["sector"]}},
            True,
            False,
        ),
    ],
    ids=["default", "columns", "party-total-cut", "groupby"],
)
async def test_frame_rendering_answers_what_the_record_states(
    options: dict[str, Any], rendering: dict[str, Any], summed: bool, default_rows: bool
) -> None:
    store = InMemoryMaterialisationStore()
    await store.put_mint_record(MintRecord(frame=_FRAME, dataset=_DATASET, options=options))
    set_store(store)

    try:
        answer = await frame_rendering(_FRAME, logger=logging.getLogger("caller"))
    finally:
        set_store(None)

    assert isinstance(answer, FrameRendering)
    assert answer.record.dataset == _DATASET
    assert (answer.rendering, answer.summed, answer.default_rows) == (
        rendering,
        summed,
        default_rows,
    )


class _FailingMintRecords(InMemoryMaterialisationStore):
    def __init__(self, error: Exception) -> None:
        super().__init__()
        self.error = error

    async def get_mint_record(self, ref: FrameRef) -> MintRecord | None:
        raise self.error


@pytest.mark.parametrize(
    ("store", "answer"),
    [
        (InMemoryMaterialisationStore(), RenderingUnknown()),
        (
            _FailingMintRecords(FrameIntegrityError(_FRAME, reason="does not decode")),
            RenderingUnknown(),
        ),
        (_FailingMintRecords(OSError("store unreachable")), RecordUnread()),
    ],
    ids=["absent", "undecodable", "read-failed"],
)
async def test_frame_rendering_tells_a_failed_read_from_an_unknown_rendering(
    store: InMemoryMaterialisationStore, answer: RenderingUnknown | RecordUnread
) -> None:
    set_store(store)

    try:
        assert await frame_rendering(_FRAME, logger=logging.getLogger("caller")) == answer
    finally:
        set_store(None)


async def test_frame_rendering_names_the_options_no_root_reads() -> None:
    store = InMemoryMaterialisationStore()
    options = {**_READ_AS, "pivot": "x", "currency": "EUR"}
    await store.put_mint_record(MintRecord(frame=_FRAME, dataset=_DATASET, options=options))
    set_store(store)

    try:
        answer = await frame_rendering(_FRAME, logger=logging.getLogger("caller"))
    finally:
        set_store(None)

    assert answer == RenderingUnknown(options=("currency", "pivot"))


@pytest.mark.parametrize(
    ("options", "columns_left_out"),
    [
        (_READ_AS, True),
        ({**_READ_AS, "columns": ["a"]}, False),
        ({**_READ_AS, "without_party_totals": True}, False),
        ({**_READ_AS, "groupby": {"by": ["sector"]}}, False),
    ],
    ids=["plain", "columns", "party-total-cut", "groupby"],
)
def test_only_a_plain_record_leaves_its_columns_to_the_default_enrichment(
    options: dict[str, Any], columns_left_out: bool
) -> None:
    record = MintRecord(frame=_FRAME, dataset=_DATASET, options=options)

    assert "columns" in root_rendering(record)
    assert ("columns" not in root_rendering(record, default_enrichment=True)) is columns_left_out


async def test_a_bare_read_records_the_default_rendering_and_keeps_a_record_already_held() -> None:
    store = InMemoryMaterialisationStore()
    summed = MintRecord(frame=_DATASET, dataset=_DATASET, options={"groupby": {"by": ["sector"]}})
    await store.put_mint_record(summed)
    set_store(store)
    logger = logging.getLogger("caller")

    try:
        await record_bare_read(_FRAME.spec_id, _FRAME.content_version, logger=logger)
        await record_bare_read(_DATASET.spec_id, _DATASET.content_version, logger=logger)
        await record_bare_read(_FRAME.spec_id, "cv_1", logger=logger)
        answer = await frame_rendering(_FRAME, logger=logger)
    finally:
        set_store(None)

    assert isinstance(answer, FrameRendering)
    assert answer.record.dataset == _FRAME
    assert answer.record.options == DEFAULT_MINT_OPTIONS
    assert answer.rendering == {"without_party_totals": False, "columns": []}
    assert answer.default_rows
    assert await store.get_mint_record(_DATASET) == summed


class _UnwritableMintRecords(InMemoryMaterialisationStore):
    async def put_mint_record(self, record: MintRecord) -> None:
        raise OSError("store unreachable")


async def test_a_failed_bare_read_record_is_logged_and_raises_nothing(
    caplog: pytest.LogCaptureFixture,
) -> None:
    set_store(_UnwritableMintRecords())

    try:
        with caplog.at_level(logging.WARNING, logger="caller"):
            await record_bare_read(
                _FRAME.spec_id, _FRAME.content_version, logger=logging.getLogger("caller")
            )
    finally:
        set_store(None)

    assert any(_FRAME.content_version in record.getMessage() for record in caplog.records)


async def test_dataset_pair_builds_no_store_for_a_pair_that_is_not_a_spec_pair() -> None:
    set_store(None)

    await dataset_pair("ops_" + "a" * 32, "ocv_" + "f" * 32, logger=logging.getLogger("caller"))

    assert not store_is_set()


async def test_dataset_pair_logs_a_failed_lookup_on_the_callers_logger(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = InMemoryMaterialisationStore()
    await store.put_mint_record(MintRecord(frame=_FRAME, dataset=_DATASET, options={}))

    async def failing(ref: FrameRef) -> MintRecord | None:
        raise OSError("storage unreachable")

    monkeypatch.setattr(store, "get_mint_record", failing)
    caplog.set_level(logging.WARNING, logger="caller")
    set_store(store)

    try:
        mapped = await dataset_pair(
            _FRAME.spec_id, _FRAME.content_version, logger=logging.getLogger("caller")
        )
    finally:
        set_store(None)

    assert mapped == (_FRAME.spec_id, _FRAME.content_version)
    [logged] = [r for r in caplog.records if r.name == "caller"]
    assert logged.levelno == logging.WARNING
    assert isinstance(logged.exc_info and logged.exc_info[1], OSError)
