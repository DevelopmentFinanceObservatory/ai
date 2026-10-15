"""Tests for `definition_store.identity`: canonicalisation and hash verification.

`test_a_real_captured_envelope_round_trips_byte_exactly` is the one test in
this file that isn't self-contained synthetic data -- see its own docstring
for where the fixture came from and why it's trustworthy.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from data_mcp.definition_store.identity import (
    IdentityMismatch,
    canonicalize,
    spec_id_for,
    verify_read,
    verify_write,
)

_FIXTURES = Path(__file__).parent / "fixtures"

_IDENTITY: dict[str, object] = {
    "identity_version": "spec1",
    "shape": {"kind": "statvar_schema", "population_type": "Person"},
    "slots": [],
    "entities": {},
    "requested_window": {"kind": "all"},
}


def test_canonicalize_is_stable_under_key_reordering() -> None:
    reordered = {
        "requested_window": _IDENTITY["requested_window"],
        "entities": _IDENTITY["entities"],
        "slots": _IDENTITY["slots"],
        "shape": _IDENTITY["shape"],
        "identity_version": _IDENTITY["identity_version"],
    }
    assert canonicalize(_IDENTITY) == canonicalize(reordered)


def test_spec_id_for_has_the_spec_underscore_plus_32_hex_shape() -> None:
    spec_id = spec_id_for(canonicalize(_IDENTITY))
    assert spec_id.startswith("spec_")
    assert len(spec_id) == len("spec_") + 32
    int(spec_id.removeprefix("spec_"), 16)  # raises ValueError if not hex


def test_verify_write_succeeds_when_identity_reproduces_the_claimed_spec_id() -> None:
    expected = spec_id_for(canonicalize(_IDENTITY))
    canonical_bytes = verify_write(_IDENTITY, expected)
    assert canonical_bytes == canonicalize(_IDENTITY)


def test_verify_write_refuses_when_identity_does_not_reproduce_the_claimed_spec_id() -> None:
    with pytest.raises(IdentityMismatch):
        verify_write(_IDENTITY, "spec_" + "0" * 32)


def test_verify_read_succeeds_on_untampered_bytes() -> None:
    canonical_bytes = canonicalize(_IDENTITY)
    verify_read(canonical_bytes, spec_id_for(canonical_bytes))  # does not raise


def test_verify_read_refuses_a_tampered_stored_value() -> None:
    canonical_bytes = canonicalize(_IDENTITY)
    expected = spec_id_for(canonical_bytes)
    tampered = canonical_bytes.replace(b"Person", b"Entity")

    with pytest.raises(IdentityMismatch):
        verify_read(tampered, expected)


def test_a_real_captured_envelope_round_trips_byte_exactly() -> None:
    """`captured_definition.json` is a real `Spec.definition` object, captured from
    apis' own offline `TestClient` fixture at
    `tests/engine/test_parsed_door.py::make_client` (fake graph/LLM/SDMX, no
    network), posting its `_fertility_kenya_payload()` to `/api/search/resolve`
    on a checkout of apis pinned to `dfo-spec-kit` revision
    `d4b1438e1d9a2aad91cdbf9f933652a952bf7c18` (`uv.lock`'s own pin at capture
    time) -- the same revision `search.models.BaseIdentity` documents as
    carrying the two-arm `shape: {kind: "statvar_schema"|"stat_var", ...}`
    union. This is real production serialisation (`Spec.definition.identity`,
    exactly as `search.engine.app` returns it) minted through the real
    engine and the real `dfo_spec_kit.identity`, not a hand-built document.

    This test is the one that would have caught a canonicalisation mismatch
    a synthetic fixture cannot: it verifies `rfc8785.dumps` plus
    `hashlib.sha256` over apis' actual wire bytes reproduces the actual
    `spec_id` apis minted, which is exactly the guarantee `search_tool`'s
    write path leans on.
    """
    captured = json.loads((_FIXTURES / "captured_definition.json").read_text())

    canonical_bytes = verify_write(captured["identity"], captured["spec_id"])

    assert canonical_bytes == canonicalize(captured["identity"])
    verify_read(canonical_bytes, captured["spec_id"])  # does not raise
