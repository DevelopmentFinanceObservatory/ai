"""Two identity families minted for the toolkit, both built on dfo-spec-kit's
JCS (RFC 8785) + SHA-256 discipline via `dfo_spec_kit.identity.canonicalize`
-- never reimplemented here.

`op_spec_hash` (domain separator `ops1`, prefix `ops_`): minted over the
fully expanded op-spec -- verb, frame_ref, and that verb's own parameters.
`frame_ref` here is always the `{spec_id, content_version}` pair
(`common.materialisation.FrameRef`), content-addressed by construction: two
callers naming the identical pair mint the identical hash, whether that pair
names a Datasets snapshot or a prior toolkit result. The hash guarantees
this: replaying the identical op-spec against the identical `frame_ref`
always reproduces the identical hash -- identical question, identical
hash, with no indirection to restore. This is what makes
`recompute_op_spec_hash` an actual integrity check rather than a
coincidence. This is a *new* family, distinct from dfo-spec-kit's own
`spec1` (dataset definitions) and `cv2` (frame-grammar content
versions): op-spec is neither a dataset-spec nor a component-spec, so it
gets its own separator rather than being squeezed into one of theirs.

Toolkit result `content_version` (domain separator `opcv1`, prefix `ocv_`):
minted over `{op_spec_hash, input content_versions, execution identity}`.
Distinct from frame-grammar's own `cv2` family (a *frame's* content_version,
the hash of its canonical bytes alone) -- a toolkit result's content_version
names a *computation's* lineage back to the op-spec and engine that produced
it, a different thing that happens to share the "content_version" name.
`ocv_` makes the two families visually distinguishable at a glance, the same
way `spec_`/`cv_` already are.

Both preimages are a closed, explicit keyword allowlist, as frame-grammar's
own `cv2` preimage is (never a wholesale `model_dump` of some richer
object). JCS handles object-member ordering on its own, and every array
member gets an explicit sort order plus a refusal -- never a silent
tie-break -- on any duplicate that sort exposes.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from itertools import pairwise
from typing import Any

from common.hashed_fields import revision_content
from dfo_spec_kit.identity import canonicalize

from analysis_mcp.execution_identity import ExecutionIdentity
from analysis_mcp.op_spec import (
    AggregateOpSpec,
    CompareOpSpec,
    FilterOpSpec,
    GrowthOpSpec,
    JoinOpSpec,
    QuotientOfOpSpec,
    RankOpSpec,
    ShareOfOpSpec,
    SumOfOpSpec,
)

OpSpecVariant = (
    AggregateOpSpec
    | CompareOpSpec
    | GrowthOpSpec
    | RankOpSpec
    | ShareOfOpSpec
    | JoinOpSpec
    | FilterOpSpec
    | SumOfOpSpec
    | QuotientOfOpSpec
)

OP_SPEC_IDENTITY_VERSION = "ops1"
OP_SPEC_HASH_PREFIX = "ops_"
OP_SPEC_HASH_PATTERN = re.compile(rf"^{OP_SPEC_HASH_PREFIX}[0-9a-f]{{32}}$")

TOOLKIT_CONTENT_VERSION_IDENTITY_VERSION = "opcv1"
TOOLKIT_CONTENT_VERSION_PREFIX = "ocv_"
TOOLKIT_CONTENT_VERSION_PATTERN = re.compile(rf"^{TOOLKIT_CONTENT_VERSION_PREFIX}[0-9a-f]{{32}}$")

# The format an *input* content_version must already carry, whether it came
# from a Datasets snapshot's `frame_ref` or from resolving a chained verb
# result's own `frame_ref` to its stored frame -- both name a
# `dfo_frame_grammar` frame, so both mint under frame-grammar's own `cv2`
# family.
_INPUT_CONTENT_VERSION_PATTERN = re.compile(r"^cv_[0-9a-f]{32}$")

_DIGEST_HEX_LENGTH = 32


class OpSpecIdentityError(ValueError):
    """A refusal while building or minting an op-spec/toolkit identity.

    `category` mirrors `dfo_spec_kit.identity.core.IdentityError`'s own
    discipline: a short, stable string a caller can branch on, never a
    message a caller would have to parse.
    """

    def __init__(self, category: str, message: str) -> None:
        self.category = category
        super().__init__(message)


def _sorted_unique_strings(items: Iterable[str], *, description: str) -> list[str]:
    """NFC-normalise, sort by Unicode code point, and refuse a duplicate.

    A duplicate is refused rather than collapsed: two entries that only
    differ because a caller repeated one aren't "the same array element
    twice," they're contradictory input the same way two `source_revisions`
    entries sharing a `source` are in frame-grammar's own preimage builder.
    """
    normalised = [unicodedata.normalize("NFC", item) for item in items]
    ordered = sorted(normalised)
    for previous, current in pairwise(ordered):
        if previous == current:
            raise OpSpecIdentityError(
                "duplicate_member", f"duplicate entry in {description}: {current!r}"
            )
    return ordered


def _sorted_unique_predicates(predicates: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Normalise each predicate's own arrays, then sort/dedup the predicate list.

    Sorted and deduplicated by each predicate's own canonical JCS bytes --
    two predicates that hash identically are the same predicate stated
    twice, a contradictory/redundant input refused the same as any other
    duplicate array member. Two different predicates over the same column
    (e.g. two value thresholds bracketing a range) are unaffected: their
    canonical bytes differ, so they sort as distinct entries.
    """
    normalised: list[dict[str, Any]] = []
    for predicate in predicates:
        entry = dict(predicate)
        if entry.get("kind") == "category_match":
            entry["values"] = _sorted_unique_strings(
                entry["values"], description="predicate values"
            )
        normalised.append(entry)

    keyed = sorted(((canonicalize(entry), entry) for entry in normalised), key=lambda kv: kv[0])
    result: list[dict[str, Any]] = []
    previous_key: bytes | None = None
    for key, entry in keyed:
        if key == previous_key:
            raise OpSpecIdentityError("duplicate_member", f"duplicate predicate: {entry!r}")
        previous_key = key
        result.append(entry)
    return result


def _normalise_frame_ref(frame_ref: object) -> object:
    if isinstance(frame_ref, Mapping):
        return {
            key: unicodedata.normalize("NFC", value) if isinstance(value, str) else value
            for key, value in frame_ref.items()
        }
    raise OpSpecIdentityError("schema_violation", f"unexpected frame_ref shape: {frame_ref!r}")


def _normalise_parameters(parameters: Mapping[str, Any]) -> dict[str, Any]:
    """Sort/dedup the array-shaped parameter fields known today.

    `AggregateOpSpec.group_by`, `FilterOpSpec.predicates`,
    `GrowthOpSpec.entity_by`, `RankOpSpec.rank_by`, `JoinOpSpec.join_keys`,
    and `ShareOfOpSpec`'s and `QuotientOfOpSpec`'s `numerator_by` are the only
    array members any verb's parameters carry -- each named explicitly here rather than walked
    generically, the same named-fields-only discipline
    `dfo_frame_grammar`'s preimage builder uses for
    `source_revisions`/`facets`. `numerator_by` names a *set* of grouping
    columns (like `group_by`/`rank_by`), so sorting it is right; its own
    pairing with `denominator_by` is carried by column *name*, not by
    either field's position, so sorting `numerator_by` here changes no
    pairing (see `ShareOfOpSpec`'s own docstring). `CompareOpSpec`'s and
    `SumOfOpSpec`'s `left_entity`/`right_entity` and `ShareOfOpSpec`'s and
    `QuotientOfOpSpec`'s `numerator_entity`/`denominator_entity`/
    `denominator_by` need no entry here: all five are JSON objects, not
    arrays, and JCS already sorts object members on its own.
    """
    normalised = dict(parameters)
    if "group_by" in normalised:
        normalised["group_by"] = _sorted_unique_strings(
            normalised["group_by"], description="group_by"
        )
    if "entity_by" in normalised:
        normalised["entity_by"] = _sorted_unique_strings(
            normalised["entity_by"], description="entity_by"
        )
    if "rank_by" in normalised:
        normalised["rank_by"] = _sorted_unique_strings(normalised["rank_by"], description="rank_by")
    if "join_keys" in normalised:
        normalised["join_keys"] = _sorted_unique_strings(
            normalised["join_keys"], description="join_keys"
        )
    if "numerator_by" in normalised:
        normalised["numerator_by"] = _sorted_unique_strings(
            normalised["numerator_by"], description="numerator_by"
        )
    if "predicates" in normalised:
        normalised["predicates"] = _sorted_unique_predicates(normalised["predicates"])
    return normalised


def build_op_spec_preimage(op_spec: OpSpecVariant) -> dict[str, Any]:
    """The closed `ops1` preimage document for `op_spec`.

    Built from `revision_content(op_spec)`: the model's JSON dump without
    the `HASHED_WHEN_SET` fields still at their default, so a parameter a
    verb gains leaves the hash of every call that does not set it as it
    was. The dump is safe to take wholesale (unlike the "never serialise a
    richer object" warning that applies to a trace/lineage object) because
    `op_spec`'s own pydantic schema is already closed (`extra="forbid"`, a
    closed verb `Literal`, a closed predicate vocabulary): every key
    reaching this preimage is one the schema itself already allowed.
    """
    data = revision_content(op_spec)
    verb = data.pop("verb")
    frame_ref = _normalise_frame_ref(data.pop("frame_ref"))
    parameters = _normalise_parameters(data)
    return {
        "identity_version": OP_SPEC_IDENTITY_VERSION,
        "verb": verb,
        "frame_ref": frame_ref,
        "parameters": parameters,
    }


def canonicalize_op_spec(op_spec: OpSpecVariant) -> bytes:
    """The canonical JCS bytes of `op_spec`'s `ops1` preimage.

    This is exactly the byte string `analysis_mcp.trace.ExecutionRecord.content`
    carries -- the trace's canonical serialisation of the fully expanded
    op-spec is this function's output, so a replayer holding only the trace
    can recompute `op_spec_hash` via `recompute_op_spec_hash` without ever
    reconstructing an `OpSpec` object.
    """
    return canonicalize(build_op_spec_preimage(op_spec))


def mint_op_spec_hash(canonical_bytes: bytes) -> str:
    """`ops_` plus the first 32 lowercase hex characters of SHA-256 of `canonical_bytes`."""
    digest = hashlib.sha256(canonical_bytes).hexdigest()
    return f"{OP_SPEC_HASH_PREFIX}{digest[:_DIGEST_HEX_LENGTH]}"


def op_spec_hash_for(op_spec: OpSpecVariant) -> str:
    """Run the full pipeline from an `OpSpec` to its minted `op_spec_hash`."""
    return mint_op_spec_hash(canonicalize_op_spec(op_spec))


def recompute_op_spec_hash(canonical_bytes: bytes) -> str:
    """Recompute `op_spec_hash` from already-canonicalised bytes, as an integrity check.

    Hashes `canonical_bytes` exactly as given -- never re-parsed, never
    re-derived from a reconstructed `OpSpec` -- so a trace's stored
    `execution_record.content` is what a replayer checks against, the same
    "hash the bytes you were actually given" discipline
    `dfo_frame_grammar.content_version.recompute_content_version` uses for a
    stored frame's canonical bytes.
    """
    return mint_op_spec_hash(canonical_bytes)


def mint_toolkit_content_version(
    *,
    op_spec_hash: str,
    input_content_versions: Iterable[str],
    execution_identity: ExecutionIdentity,
) -> str:
    """Mint a toolkit result's `content_version`: `ocv_` plus 32 hex chars of
    SHA-256 over the canonical `opcv1` preimage built from exactly these
    three named arguments.

    `input_content_versions` names every input frame's own `cv_` content
    version (one for a single-frame verb, more than one for `join`/`compare`)
    -- sorted and deduplicated the same way `group_by`/`predicates` are,
    since JCS doesn't sort arrays and two inputs listed in a different order
    must still mint the same token.
    """
    if not OP_SPEC_HASH_PATTERN.fullmatch(op_spec_hash):
        raise OpSpecIdentityError(
            "invalid_op_spec_hash", f"not a valid op_spec_hash: {op_spec_hash!r}"
        )
    versions = list(input_content_versions)
    for version in versions:
        if not _INPUT_CONTENT_VERSION_PATTERN.fullmatch(version):
            raise OpSpecIdentityError(
                "invalid_content_version", f"not a valid input content_version: {version!r}"
            )
    ordered_versions = _sorted_unique_strings(versions, description="input_content_versions")

    preimage = {
        "identity_version": TOOLKIT_CONTENT_VERSION_IDENTITY_VERSION,
        "op_spec_hash": op_spec_hash,
        "input_content_versions": ordered_versions,
        "execution_identity": execution_identity.canonical_string(),
    }
    digest = hashlib.sha256(canonicalize(preimage)).hexdigest()
    return f"{TOOLKIT_CONTENT_VERSION_PREFIX}{digest[:_DIGEST_HEX_LENGTH]}"


def recompute_toolkit_content_version(
    *,
    op_spec_hash: str,
    input_content_versions: Iterable[str],
    execution_identity: ExecutionIdentity,
) -> str:
    """Recompute a toolkit result's `content_version` from its stored identity
    fields, as an integrity check -- an alias for `mint_toolkit_content_version`
    with the recompute-facing name a replayer calls by.
    """
    return mint_toolkit_content_version(
        op_spec_hash=op_spec_hash,
        input_content_versions=input_content_versions,
        execution_identity=execution_identity,
    )
