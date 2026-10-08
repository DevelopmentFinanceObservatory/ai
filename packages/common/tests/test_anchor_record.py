"""Tests for the anchor record and the in-memory store's handling of it.
`test_gcs_store.py` covers the bucket-backed store."""

from __future__ import annotations

from typing import Any

import pytest
from common.materialisation import (
    AnchorRecord,
    InMemoryMaterialisationStore,
    decode_anchor_record,
    encode_anchor_record,
)
from common.rolling import anchor_key
from pydantic import ValidationError

_SPEC = "spec_" + "a" * 32
_ROOTS = [
    {"spec_id": _SPEC, "content_version": "cv_" + "2" * 32},
    {"spec_id": _SPEC, "content_version": "cv_" + "1" * 32},
]
_POPULATIONS: list[dict[str, Any]] = [
    {"spec_id": _SPEC, "bindings": {"reporter": ["dcid:USA", "dcid:FRA"]}}
]


def _record(*, period: str = "2023", roots: list[dict[str, str]] | None = None) -> AnchorRecord:
    return AnchorRecord(
        populations=_POPULATIONS,
        roots=roots or _ROOTS,
        rule="half1",
        period=period,
        left_out=[
            {"root": 0, "coordinates": {"reporter": "dcid:USA"}},
            {"root": 0, "coordinates": {"reporter": "dcid:FRA"}},
        ],
    )


def test_a_record_is_keyed_by_its_populations_roots_and_rule_alone() -> None:
    record = _record()

    assert record.anchor_key == anchor_key(populations=_POPULATIONS, roots=_ROOTS, rule="half1")
    assert _record(period="2022").anchor_key == record.anchor_key
    assert _record(roots=_ROOTS[:1]).anchor_key != record.anchor_key


def test_a_record_encodes_to_one_set_of_bytes_however_its_members_were_listed() -> None:
    listed_otherwise = AnchorRecord(
        left_out=list(reversed(_record().left_out)),
        period="2023",
        rule="half1",
        roots=list(reversed(_ROOTS)),
        populations=[{"spec_id": _SPEC, "bindings": {"reporter": ["dcid:FRA", "dcid:USA"]}}],
    )

    assert encode_anchor_record(listed_otherwise) == encode_anchor_record(_record())
    assert encode_anchor_record(_record()) == (
        b'{"left_out":[{"coordinates":{"reporter":"dcid:FRA"},"root":0},'
        b'{"coordinates":{"reporter":"dcid:USA"},"root":0}],"period":"2023",'
        b'"populations":[{"bindings":{"reporter":["dcid:FRA","dcid:USA"]},"spec_id":"'
        + _SPEC.encode()
        + b'"}],"roots":[{"content_version":"cv_'
        + b"1" * 32
        + b'","spec_id":"'
        + _SPEC.encode()
        + b'"},{"content_version":"cv_'
        + b"2" * 32
        + b'","spec_id":"'
        + _SPEC.encode()
        + b'"}],"rule":"half1"}'
    )
    assert decode_anchor_record(encode_anchor_record(_record())) == _record()


def test_a_stored_record_with_a_member_this_process_never_declared_still_decodes() -> None:
    stored = encode_anchor_record(_record()).replace(b'{"left_out"', b'{"added":1,"left_out"', 1)

    assert decode_anchor_record(stored) == _record()


def test_a_left_out_series_carries_no_label() -> None:
    with pytest.raises(ValidationError):
        AnchorRecord(
            populations=_POPULATIONS,
            roots=_ROOTS,
            rule="half1",
            period="2023",
            left_out=[{"root": 0, "coordinates": {}, "label": "United States"}],
        )


async def test_the_first_anchor_record_written_under_a_key_stands() -> None:
    store = InMemoryMaterialisationStore()
    first = _record()

    assert await store.get_anchor_record(first.anchor_key) is None

    await store.put_anchor_record(first)
    await store.put_anchor_record(_record(period="2022"))

    assert await store.get_anchor_record(first.anchor_key) == first
    assert await store.list_frames() == []


def test_a_populations_level_is_written_only_when_set() -> None:
    levelled = AnchorRecord(
        populations=[{**_POPULATIONS[0], "levels": {"observationAbout_type": "Country"}}],
        roots=_ROOTS,
        rule="half1",
        period="2023",
    )
    stated_empty = AnchorRecord(
        populations=[{**_POPULATIONS[0], "levels": {}}], roots=_ROOTS, rule="half1", period="2023"
    )

    assert b'"levels":{"observationAbout_type":"Country"}' in encode_anchor_record(levelled)
    assert decode_anchor_record(encode_anchor_record(levelled)) == levelled
    assert levelled.anchor_key != _record().anchor_key
    assert b"levels" not in encode_anchor_record(_record())
    assert stated_empty.anchor_key == _record().anchor_key
