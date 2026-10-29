"""Tests for `data_mcp.mint_recovery`: rebuilding a root frame's mint record from
its pair, the frame the store holds, and the dataset's stored versions."""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from dataclasses import replace
from typing import Any

import pytest
from common.live_apis import DatasetVersion
from common.materialisation import (
    FrameRef,
    InMemoryMaterialisationStore,
    MaterialisationTrace,
    MintRecord,
    MintRecordUnrecoverableError,
)
from data_mcp.frame_materialisation import KIND_COLUMNS, frame_ref_for
from data_mcp.mint_recovery import recover_mint_record
from dfo_frame_grammar import Frame

_SPEC_ID = "spec_" + "1" * 32
_OLDER = "cv_" + "2" * 32
_NEWER = "cv_" + "3" * 32
_SUMMED = "cv_" + "4" * 32
_UNRECORDED = "cv_" + "5" * 32
_CUT = "cv_" + "6" * 32
_CUT_SUMMED = "cv_" + "7" * 32
_DEFAULT_OPTIONS = {"format": "parquet", "bundle": False, "layout": "frame"}
_GROUPBY = {"by": ["recipient"], "op": "sum", "levels": {"crsPurpose": 0}}
_VERSIONS = (
    DatasetVersion(_CUT_SUMMED, _NEWER, "groupby", _GROUPBY, without_party_totals=True),
    DatasetVersion(_CUT, _NEWER, "without_party_totals", None, without_party_totals=True),
    DatasetVersion(_UNRECORDED, _NEWER, "not_recorded", None),
    DatasetVersion(_SUMMED, _NEWER, "groupby", _GROUPBY),
    DatasetVersion(_NEWER, None, "plain", None),
    DatasetVersion(_OLDER, None, "plain", None),
)


class _Versions:
    """A fake of `LiveApisClient.list_versions`, recording the datasets asked for. Each
    version lists `stamp`, as apis lists its digest on every version whose frame holds a
    party role, or only the versions in `stamped` when that is given."""

    def __init__(
        self,
        versions: Sequence[DatasetVersion] = _VERSIONS,
        *,
        stamp: str | None = None,
        stamped: Collection[str] | None = None,
    ) -> None:
        self._versions = tuple(
            replace(
                v,
                enrichment_stamp=stamp if stamped is None or v.content_version in stamped else None,
            )
            for v in versions
        )
        self.asked: list[str] = []

    async def __call__(self, *, spec_id: str) -> Sequence[DatasetVersion]:
        self.asked.append(spec_id)
        return self._versions


# What `get_data(place_types=True)` sent for a spec naming a flow axis before it asked for
# kinds.
_FLOW_TYPES = ("financingProvider_type", "financingRecipient_type")


def _typed(content_version: str, columns: tuple[str, ...] = ("observationAbout_type",)) -> FrameRef:
    return frame_ref_for(spec_id=_SPEC_ID, content_version=content_version, columns=columns)


async def _holding(
    ref: FrameRef,
    *stat_vars: str,
    options: dict[str, object] | None = None,
    stamp: str | None = None,
) -> InMemoryMaterialisationStore:
    """A store holding, under `ref`, a frame with one row for each of `stat_vars`,
    whose trace keeps `options` and `stamp`."""
    store = InMemoryMaterialisationStore()
    if stat_vars:
        frame = Frame.build(
            role_keys=["reporter"],
            rows=[
                {"stat_var": stat_var, "reporter": "dcid:FRA", "time_period": "2023", "value": 1.0}
                for stat_var in stat_vars
            ],
        )
        trace = MaterialisationTrace(
            upstream_trace={},
            spec_id=ref.spec_id,
            content_version=ref.content_version,
            options=options,
            enrichment_stamp=stamp,
        )
        await store.put(ref, frame=frame, trace=trace, writer="test")
    return store


@pytest.mark.parametrize(
    ("ref", "dataset_version", "options"),
    [
        (FrameRef(spec_id=_SPEC_ID, content_version=_OLDER), _OLDER, _DEFAULT_OPTIONS),
        (
            _typed(_OLDER),
            _OLDER,
            _DEFAULT_OPTIONS | {"columns": ["observationAbout_type"]},
        ),
        (
            _typed(_NEWER),
            _NEWER,
            _DEFAULT_OPTIONS | {"columns": ["observationAbout_type"]},
        ),
        (
            FrameRef(spec_id=_SPEC_ID, content_version=_SUMMED),
            _SUMMED,
            _DEFAULT_OPTIONS | {"groupby": _GROUPBY},
        ),
        (
            _typed(_SUMMED),
            _SUMMED,
            _DEFAULT_OPTIONS | {"groupby": _GROUPBY, "columns": ["observationAbout_type"]},
        ),
        (
            frame_ref_for(spec_id=_SPEC_ID, content_version=_SUMMED, groupby=_GROUPBY),
            _SUMMED,
            _DEFAULT_OPTIONS | {"groupby": _GROUPBY},
        ),
        (
            frame_ref_for(
                spec_id=_SPEC_ID,
                content_version=_SUMMED,
                columns=("observationAbout_type",),
                groupby=_GROUPBY,
            ),
            _SUMMED,
            _DEFAULT_OPTIONS | {"groupby": _GROUPBY, "columns": ["observationAbout_type"]},
        ),
        (
            _typed(_OLDER, _FLOW_TYPES),
            _OLDER,
            _DEFAULT_OPTIONS | {"columns": list(_FLOW_TYPES)},
        ),
        (
            _typed(_OLDER, ("financingProvider_type",)),
            _OLDER,
            _DEFAULT_OPTIONS | {"columns": ["financingProvider_type"]},
        ),
        (
            _typed(_OLDER, KIND_COLUMNS),
            _OLDER,
            _DEFAULT_OPTIONS | {"columns": list(KIND_COLUMNS)},
        ),
        (
            _typed(_OLDER, ("party_kind",)),
            _OLDER,
            _DEFAULT_OPTIONS | {"columns": ["party_kind"]},
        ),
        (
            _typed(_OLDER, ("kind",)),
            _OLDER,
            _DEFAULT_OPTIONS | {"columns": ["kind"]},
        ),
        (
            _typed(_OLDER, (*_FLOW_TYPES, *KIND_COLUMNS)),
            _OLDER,
            _DEFAULT_OPTIONS | {"columns": [*_FLOW_TYPES, *KIND_COLUMNS]},
        ),
        (
            _typed(_OLDER, ("financingProvider_type", "party_kind")),
            _OLDER,
            _DEFAULT_OPTIONS | {"columns": ["financingProvider_type", "party_kind"]},
        ),
        (
            frame_ref_for(spec_id=_SPEC_ID, content_version=_CUT, columns=("party_kind",)),
            _CUT,
            _DEFAULT_OPTIONS | {"columns": ["party_kind"], "without_party_totals": True},
        ),
        (
            frame_ref_for(
                spec_id=_SPEC_ID,
                content_version=_CUT_SUMMED,
                columns=("party_kind",),
                groupby=_GROUPBY,
            ),
            _CUT_SUMMED,
            _DEFAULT_OPTIONS
            | {"columns": ["party_kind"], "groupby": _GROUPBY, "without_party_totals": True},
        ),
        (
            frame_ref_for(
                spec_id=_SPEC_ID,
                content_version=_CUT,
                columns=("party_kind",),
                without_party_totals=True,
            ),
            _CUT,
            _DEFAULT_OPTIONS | {"columns": ["party_kind"], "without_party_totals": True},
        ),
        (
            frame_ref_for(
                spec_id=_SPEC_ID,
                content_version=_CUT_SUMMED,
                columns=("party_kind",),
                groupby=_GROUPBY,
                without_party_totals=True,
            ),
            _CUT_SUMMED,
            _DEFAULT_OPTIONS
            | {"columns": ["party_kind"], "groupby": _GROUPBY, "without_party_totals": True},
        ),
        (
            frame_ref_for(
                spec_id=_SPEC_ID,
                content_version=_OLDER,
                columns=("party_kind",),
                without_party_totals=True,
            ),
            _OLDER,
            _DEFAULT_OPTIONS | {"columns": ["party_kind"], "without_party_totals": True},
        ),
    ],
    ids=[
        "default-columns",
        "place-types-on-an-older-version",
        "place-types-on-a-newer-version",
        "recorded-groupby",
        "recorded-groupby-with-place-types",
        "recorded-groupby-under-its-named-pair",
        "recorded-groupby-with-place-types-under-its-named-pair",
        "both-flow-party-types",
        "the-one-flow-party-type-a-table-serves",
        "flow-party-kinds",
        "flow-party-kind-without-class-and-whole",
        "flow-party-kinds-under-the-earlier-token",
        "flow-party-types-and-kinds",
        "the-one-flow-party-type-a-table-serves-and-the-kind-without-class-and-whole",
        "flow-party-kinds-of-a-cut-version-before-its-pair-named-the-cut",
        "flow-party-kinds-of-a-sum-of-a-cut-version-before-its-pair-named-the-cut",
        "flow-party-kinds-of-a-cut-version",
        "flow-party-kinds-of-a-sum-of-a-cut-version",
        "flow-party-kinds-of-a-cut-that-left-nothing-out",
    ],
)
async def test_a_root_is_recovered_from_the_version_that_reproduces_its_pair(
    ref: FrameRef, dataset_version: str, options: dict[str, object]
) -> None:
    versions = _Versions()
    store = await _holding(ref, "dcid:A", "dcid:B")

    record = await recover_mint_record(ref, dataset_versions=versions, store=store)

    assert record == MintRecord(
        frame=ref,
        dataset=FrameRef(spec_id=_SPEC_ID, content_version=dataset_version),
        options=options,
        scope={},
        stat_var_count=2,
    )
    assert versions.asked == [_SPEC_ID]


@pytest.mark.parametrize(
    ("ref", "stat_vars", "stat_var_count"),
    [
        # Minted with or without a `groupby`, it names the same rows while one StatVar
        # contributes, and the count of 1 lets a replay refuse it once a second does.
        (FrameRef(spec_id=_SPEC_ID, content_version=_OLDER), ("dcid:A",), 1),
        # apis recorded the sum's `groupby`, so the options are the mint's.
        (FrameRef(spec_id=_SPEC_ID, content_version=_SUMMED), (), None),
    ],
    ids=["one-stat-var-under-a-raw-version", "recorded-groupby-no-longer-held"],
)
async def test_a_recovered_record_counts_the_stat_vars_of_the_held_frame(
    ref: FrameRef, stat_vars: tuple[str, ...], stat_var_count: int | None
) -> None:
    store = await _holding(ref, *stat_vars)

    record = await recover_mint_record(ref, dataset_versions=_Versions(), store=store)

    assert record.stat_var_count == stat_var_count
    assert (
        "groupby" in record.options if stat_var_count is None else "groupby" not in record.options
    )


@pytest.mark.parametrize(
    ("ref", "reason"),
    [
        (FrameRef(spec_id=_SPEC_ID, content_version="cv_" + "9" * 32), "no_version_matches"),
        (_typed(_OLDER, columns=("recipient_type",)), "no_version_matches"),
        (FrameRef(spec_id=_SPEC_ID, content_version=_UNRECORDED), "groupby_unrecorded"),
        (_typed(_UNRECORDED), "groupby_unrecorded"),
        # Its StatVar count cannot be read, so whether it was a sum cannot be bounded.
        (FrameRef(spec_id=_SPEC_ID, content_version=_OLDER), "frame_not_held"),
        # apis records no groupby on a version a sum kept, and no frame is held.
        (
            frame_ref_for(spec_id=_SPEC_ID, content_version=_OLDER, groupby=_GROUPBY),
            "no_version_matches",
        ),
    ],
    ids=[
        "version-not-listed",
        "unknown-column-selection",
        "summed-version",
        "summed-and-typed",
        "raw-version-no-longer-held",
        "sum-that-kept-its-version",
    ],
)
async def test_a_root_nothing_reproduces_fails_with_its_pair_and_reason(
    ref: FrameRef, reason: str
) -> None:
    with pytest.raises(MintRecordUnrecoverableError) as raised:
        await recover_mint_record(
            ref, dataset_versions=_Versions(), store=InMemoryMaterialisationStore()
        )

    assert raised.value.ref == ref
    assert raised.value.reason == reason


_KEPT_SUM = _DEFAULT_OPTIONS | {"groupby": _GROUPBY}


@pytest.mark.parametrize(
    ("without_party_totals", "named_cut"),
    [(False, False), (True, True), (True, False)],
    ids=["kept", "totals-left-out", "totals-left-out-before-its-pair-named-the-cut"],
)
async def test_a_sum_that_kept_its_version_is_recovered_from_its_traces_options(
    without_party_totals: bool, named_cut: bool
) -> None:
    """apis lists the version as unsummed, so only the options the held frame's
    trace keeps reproduce the pair that names the groupby."""
    kept_sum = _KEPT_SUM | (
        {"columns": ["party_kind"], "without_party_totals": True} if without_party_totals else {}
    )
    ref = frame_ref_for(
        spec_id=_SPEC_ID,
        content_version=_OLDER,
        columns=("party_kind",) if without_party_totals else (),
        groupby=_GROUPBY,
        without_party_totals=named_cut,
    )
    store = await _holding(ref, "dcid:A", options=kept_sum)

    record = await recover_mint_record(ref, dataset_versions=_Versions(), store=store)

    assert record == MintRecord(
        frame=ref,
        dataset=FrameRef(spec_id=_SPEC_ID, content_version=_OLDER),
        options=kept_sum,
        scope={},
        stat_var_count=1,
    )


@pytest.mark.parametrize(
    "options",
    [None, _DEFAULT_OPTIONS | {"groupby": {"by": ["donor"], "op": "sum"}}],
    ids=["trace-without-options", "trace-naming-another-groupby"],
)
async def test_a_sum_that_kept_its_version_is_unrecoverable_without_its_options(
    options: dict[str, object] | None,
) -> None:
    ref = frame_ref_for(spec_id=_SPEC_ID, content_version=_OLDER, groupby=_GROUPBY)
    store = await _holding(ref, "dcid:A", options=options)

    with pytest.raises(MintRecordUnrecoverableError) as raised:
        await recover_mint_record(ref, dataset_versions=_Versions(), store=store)

    assert raised.value.reason == "no_version_matches"


_KINDS = ("party_kind",)
_KIND_OPTIONS: dict[str, object] = _DEFAULT_OPTIONS | {"columns": list(_KINDS)}
_KIND_SUM: dict[str, object] = _KIND_OPTIONS | {"groupby": _GROUPBY}


def _salted(stamp: str, *, groupby: Mapping[str, Any] | None = None) -> FrameRef:
    return frame_ref_for(
        spec_id=_SPEC_ID,
        content_version=_OLDER,
        columns=_KINDS,
        groupby=groupby,
        enrichment_stamp=stamp,
    )


@pytest.mark.parametrize("kept_sum", [False, True], ids=["from-a-version", "from-the-trace"])
@pytest.mark.parametrize(
    "stamped",
    [None, {_OLDER}],
    ids=["every-version-lists-it", "only-the-roots-version-lists-it"],
)
async def test_a_kind_root_named_under_the_current_stamp_is_recovered_with_it(
    kept_sum: bool, stamped: set[str] | None
) -> None:
    """A held frame's pair is reproduced from the stamp its own version lists, whatever
    the other versions list, and the record keeps it, so a replay names the same pair."""
    ref = _salted("stamp-now", groupby=_GROUPBY if kept_sum else None)
    options = _KIND_SUM if kept_sum else _KIND_OPTIONS
    # A version reproduces the pair of an unsummed root without the trace's options.
    store = await _holding(ref, "dcid:A", options=options if kept_sum else None)

    record = await recover_mint_record(
        ref, dataset_versions=_Versions(stamp="stamp-now", stamped=stamped), store=store
    )

    assert record == MintRecord(
        frame=ref,
        dataset=FrameRef(spec_id=_SPEC_ID, content_version=_OLDER),
        options=options,
        stat_var_count=1,
        enrichment_stamp="stamp-now",
    )


@pytest.mark.parametrize("kept_sum", [False, True], ids=["from-a-version", "from-the-trace"])
@pytest.mark.parametrize("current", [None, "stamp-now"], ids=["no-current-stamp", "a-newer-stamp"])
async def test_a_kind_root_named_under_an_older_stamp_its_trace_does_not_keep_is_not_recoverable(
    kept_sum: bool, current: str | None
) -> None:
    """The held trace keeps no stamp, so nothing names the stamp its pair was derived
    from."""
    ref = _salted("stamp-before", groupby=_GROUPBY if kept_sum else None)
    store = await _holding(ref, "dcid:A", options=_KIND_SUM if kept_sum else _KIND_OPTIONS)

    with pytest.raises(MintRecordUnrecoverableError) as raised:
        await recover_mint_record(ref, dataset_versions=_Versions(stamp=current), store=store)

    assert raised.value.reason == "no_version_matches"


@pytest.mark.parametrize("kept_sum", [False, True], ids=["from-a-version", "from-the-trace"])
async def test_a_kind_root_named_under_an_older_stamp_is_recovered_with_its_traces_stamp(
    kept_sum: bool,
) -> None:
    """The record keeps the stamp the pair was named under, so a refresh sees it differ
    from the one its version lists and mints the root again."""
    ref = _salted("stamp-before", groupby=_GROUPBY if kept_sum else None)
    options = _KIND_SUM if kept_sum else _KIND_OPTIONS
    store = await _holding(ref, "dcid:A", options=options, stamp="stamp-before")

    record = await recover_mint_record(
        ref, dataset_versions=_Versions(stamp="stamp-now"), store=store
    )

    assert record == MintRecord(
        frame=ref,
        dataset=FrameRef(spec_id=_SPEC_ID, content_version=_OLDER),
        options=options,
        stat_var_count=1,
        enrichment_stamp="stamp-before",
    )


@pytest.mark.parametrize("current", [None, "stamp-now"], ids=["no-stamp-served", "a-stamp-served"])
async def test_an_unsalted_kind_root_is_recovered_without_a_stamp(current: str | None) -> None:
    """Named before apis sent a stamp. The record keeps none, so a refresh sees it
    differ from a stamp its version lists and mints the root again."""
    ref = _typed(_OLDER, _KINDS)
    store = await _holding(ref, "dcid:A")

    record = await recover_mint_record(ref, dataset_versions=_Versions(stamp=current), store=store)

    assert record.options == _KIND_OPTIONS
    assert record.enrichment_stamp is None


async def test_a_root_without_kind_columns_is_recovered_whatever_stamp_is_served() -> None:
    """Types come from the graph, so the stamp names no such frame and its record keeps none."""
    ref = _typed(_OLDER)
    store = await _holding(ref, "dcid:A")

    record = await recover_mint_record(
        ref, dataset_versions=_Versions(stamp="stamp-now"), store=store
    )

    assert record.enrichment_stamp is None
