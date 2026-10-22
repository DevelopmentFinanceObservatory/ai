"""Canonicalisation and verification of the spec1 identity core.

`spec_id = "spec_" + sha256(canonical_bytes).hexdigest()[:32]`, where
`canonical_bytes` is `rfc8785.dumps(identity)` -- RFC 8785 (JSON
Canonicalization Scheme) applied to the identity document apis serialises
under `Spec.definition.identity`. This module owns no knowledge of that
document's shape beyond treating it as an opaque JSON-compatible mapping:
`dfo-spec-kit` mints and verifies spec_ids the same way internally, but is
not a dependency here -- `rfc8785` and `hashlib` are enough to check that a
document reproduces the id it claims.
"""

from __future__ import annotations

import hashlib
from typing import Any

import rfc8785

_SPEC_ID_PREFIX = "spec_"
_DIGEST_HEX_LENGTH = 32


class IdentityMismatch(Exception):  # noqa: N818 -- named for the refusal it maps to, not a generic "Error"
    """Raised when an identity document does not hash to its claimed spec_id.

    Deterministic: the same identity document reproduces the same mismatch
    every time, so a caller must refuse rather than retry unmodified.
    """

    def __init__(self, *, expected_spec_id: str, actual_spec_id: str) -> None:
        self.expected_spec_id = expected_spec_id
        self.actual_spec_id = actual_spec_id
        super().__init__(
            f"identity does not reproduce spec_id: expected {expected_spec_id!r}, "
            f"got {actual_spec_id!r}"
        )


def canonicalize(identity: dict[str, Any]) -> bytes:
    """Return the RFC 8785 canonical JSON bytes of `identity`."""
    return rfc8785.dumps(identity)


def spec_id_for(canonical_bytes: bytes) -> str:
    """Derive the spec_id from already-canonicalised bytes."""
    return _SPEC_ID_PREFIX + hashlib.sha256(canonical_bytes).hexdigest()[:_DIGEST_HEX_LENGTH]


def verify_write(identity: dict[str, Any], expected_spec_id: str) -> bytes:
    """Canonicalise `identity` and verify it reproduces `expected_spec_id`.

    Returns the canonical bytes on success, ready to store. Raises
    `IdentityMismatch` when the canonicalised identity does not reproduce the
    envelope's own spec_id.
    """
    canonical_bytes = canonicalize(identity)
    actual_spec_id = spec_id_for(canonical_bytes)
    if actual_spec_id != expected_spec_id:
        raise IdentityMismatch(expected_spec_id=expected_spec_id, actual_spec_id=actual_spec_id)
    return canonical_bytes


def verify_read(canonical_bytes: bytes, expected_spec_id: str) -> None:
    """Verify bytes read back from storage still hash to `expected_spec_id`.

    A bare hash, no re-canonicalisation: the bytes were already canonical
    when `verify_write` stored them. Raises `IdentityMismatch` on a mismatch;
    the caller must refuse rather than serve the tampered value.
    """
    actual_spec_id = spec_id_for(canonical_bytes)
    if actual_spec_id != expected_spec_id:
        raise IdentityMismatch(expected_spec_id=expected_spec_id, actual_spec_id=actual_spec_id)
