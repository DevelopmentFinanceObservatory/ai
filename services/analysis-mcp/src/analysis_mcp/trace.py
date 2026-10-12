"""The toolkit's trace schema.

Wraps the upstream `materialise` pipeline's `ManifestTrace` as an opaque JSON
mapping -- `upstream_trace` carries it verbatim, with no `dfo-data-tools`
import and no redefinition of any field `ManifestTrace` already owns (graph
revision, source revisions, resolved stat vars, facets, frame grammar
version, materialiser build, capability version all stay exactly as
`materialise` produced them). `data_mcp.frame_materialisation` populates
this field from a real `ManifestTrace.to_dict()` on every `get_data` mint;
`analysis_mcp.fixtures` populates it with a fixture shaped the same way.

Everything past `upstream_trace` is what only the toolkit knows:

- `frame_content_version`: the stored output frame's own frame-grammar `cv_`,
  derived once when this result is written
  (`frame_ref_resolution.derive_frame_content_version`). The next hop that
  resolves this result as its own input reads this field directly
  (`frame_ref_resolution.input_content_version`).
- `op_spec_hash`: which op-spec produced this result.
- `execution_record`: the replay witness. `kind="op_spec_canonical"` and
  `content` is the canonical JCS serialisation of the fully expanded
  op-spec -- `analysis_mcp.identity.canonicalize_op_spec`'s own output,
  byte for byte. The pure-Python engine renders no SQL, so the field that
  makes a result replayable carries the canonical op-spec instead,
  kind-tagged so a future SQL-rendering engine could populate the same
  field under a different `kind` without a schema change.
- `applied_policy`: the null/coverage policy actually applied, populated
  per verb -- every field is optional here since each verb's own policy
  decides what it fills in (see `AppliedPolicy`'s own docstring).
- `exclusions`: every row this result's verb narrowed away, and why
  (`analysis_mcp.artifact.Exclusion`). `()` when nothing was dropped, or on
  a `Trace` predating this field.
- `composed_stat_vars`: what each StatVar an operation minted is made of
  (`common.materialisation.ComposedStatVar`), for every such StatVar the
  stored frame holds and every one those name.
  `tool_pipeline.store_result_and_render` sets it; no identity field reads
  it. `()` when the frame holds none, or on a `Trace` stored without the
  field.
- `time`: whether this result's `time_period` names a genuine single
  observation (`analysis_mcp.artifact.Observation`, the default) or stands
  for a collapsed span of more than one real period
  (`analysis_mcp.artifact.Interval`).
- The identity fields: `op_spec_hash` (also above), `content_version` (the
  toolkit's own `ocv_` family, not frame-grammar's `cv_`),
  `input_content_versions` (the `cv_` content_version(s) `content_version`
  was minted from -- so a stored result traces back to its inputs without
  re-deriving them from `upstream_trace`), and `execution_identity`.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from common.materialisation import ComposedStatVar, register_trace_codec
from pydantic import BaseModel, ConfigDict, Field

from analysis_mcp.artifact import Exclusion, Observation, Time
from analysis_mcp.execution_identity import ExecutionIdentity
from analysis_mcp.identity import OP_SPEC_HASH_PATTERN, TOOLKIT_CONTENT_VERSION_PATTERN

# Matches `identity._INPUT_CONTENT_VERSION_PATTERN` -- duplicated here rather
# than imported, since that name is private to `identity.py`. Both patterns
# describe the identical shape: the `cv_` family `dfo_frame_grammar`'s own
# content_version mints, whether a caller's input came from a Datasets
# snapshot's `frame_ref` or from resolving a chained verb result's own
# `frame_ref` to its stored frame.
_INPUT_CONTENT_VERSION_PATTERN = r"^cv_[0-9a-f]{32}$"


class ExecutionRecord(BaseModel):
    """The replay witness: what was actually run, in a form a replayer can rehash."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["op_spec_canonical"] = "op_spec_canonical"
    content: str = Field(
        description=(
            "The canonical JCS serialisation of the fully expanded op-spec, "
            "decoded as text -- `analysis_mcp.identity.canonicalize_op_spec(op_spec)` "
            "output. Feeding this back through `recompute_op_spec_hash` "
            "(after re-encoding to bytes) must reproduce `Trace.op_spec_hash`."
        )
    )


class AppliedPolicy(BaseModel):
    """The null/coverage policy actually applied, populated per verb.

    Every field is optional here: the shape lets every verb's trace carry a
    policy record without naming a rule yet -- `aggregate`'s skip-nulls
    policy is the first to fill it in.

    `realised_coverage_fraction` and `skipped_row_count` name grain
    coverage, not null coverage: frame-grammar cannot represent an
    in-frame null observation (`Frame.build` refuses a non-finite `value`
    and `status` is reserved, always `None` "at this grammar version" --
    see `analysis_mcp.fixtures`'s module docstring), so "skip nulls" can
    only mean "skip absent grain cells" until `status` populates. The two
    fields name cells present versus cells expected for the grouped grain,
    and the expected-but-absent count, not a per-cell null count. If
    frame-grammar's `status` column ever populates, the policy extends to
    real null cells without this shape changing.

    `resolved_unit` and `resolved_facets` carry what
    `analysis_mcp.engine.common.resolve_unit_and_facets` decided before
    computing -- `resolved_unit` is the single unit the result was computed
    over (`None` when the input frame carried no `unit` at all; a converted
    unit as its request object, `units.unit_json`), and
    `resolved_facets` is `axis -> ref` for every facet axis auto-selected
    from a frame that left it unbound on some rows (see that function's own
    docstring for when a selection is made versus refused).

    `detail` is a closed-at-the-key-level, otherwise free-form JSON mapping
    for whatever structured extra a verb's own policy needs beyond the one
    scalar `realised_coverage_fraction` -- `compare`'s per-side coverages,
    `growth`'s per-entity coverage breakdown and window bounds, `rank`'s
    coverage of each entity it keeps and the entity count and coverage range
    of its whole ranked population, and every one of the nine verbs' own
    `detail["derived"]` marker (see `frame_ref_resolution.DerivedMarker`).
    Every verb documents its own `detail` shape in its own tool module
    rather than this schema growing a field per verb.
    """

    model_config = ConfigDict(extra="forbid")

    policy_name: str | None = None
    realised_coverage_fraction: float | None = Field(default=None, ge=0.0, le=1.0)
    skipped_row_count: int | None = Field(default=None, ge=0)
    resolved_unit: str | dict[str, Any] | None = None
    resolved_facets: dict[str, str] = Field(default_factory=dict)
    detail: dict[str, Any] = Field(default_factory=dict)


class Trace(BaseModel):
    """The toolkit's own trace: upstream lineage plus the toolkit's replay witness."""

    model_config = ConfigDict(extra="forbid")

    upstream_trace: dict[str, Any] = Field(
        description="The upstream `ManifestTrace.to_dict()` output, carried verbatim."
    )
    frame_content_version: str = Field(
        pattern=_INPUT_CONTENT_VERSION_PATTERN,
        description=(
            "This result's own stored frame's frame-grammar `cv_`, derived once at write "
            "time from the frame's own bytes alone. A later hop that resolves this "
            "result as its own input reads this field directly."
        ),
    )
    op_spec_hash: str = Field(pattern=OP_SPEC_HASH_PATTERN)
    execution_record: ExecutionRecord
    applied_policy: AppliedPolicy
    exclusions: tuple[Exclusion, ...] = ()
    time: Time = Field(default_factory=Observation)
    composed_stat_vars: tuple[ComposedStatVar, ...] = Field(
        default=(),
        description=(
            "The description of every operation-minted StatVar this result's stored "
            "frame holds, and of every one those descriptions name, each after the "
            "entries its sides name."
        ),
    )
    content_version: str = Field(pattern=TOOLKIT_CONTENT_VERSION_PATTERN)
    input_content_versions: tuple[
        Annotated[str, Field(pattern=_INPUT_CONTENT_VERSION_PATTERN)], ...
    ] = Field(
        description=(
            "Every input frame's own `cv_` content_version this result was minted from -- "
            "exactly the values `identity.mint_toolkit_content_version`'s own "
            "`input_content_versions` argument was called with (one for a single-frame "
            "verb, more than one for `join`/`compare`/`share_of`/`sum_of`/`quotient_of` when "
            "their two sides differ). `content_version` above names the mint's own output; "
            "this names its "
            "inputs, so a stored result can be traced back to what it was computed from "
            "without re-deriving `input_content_version` from `upstream_trace` by hand. "
            "Every one of this toolkit's nine verbs populates it; `()` only ever appears "
            "on a `Trace` predating this field (e.g. a frozen test fixture)."
        ),
    )
    execution_identity: ExecutionIdentity


class _LenientTrace(Trace):
    """The decode-side shape for a durable `Trace` record: unknown members
    are ignored rather than refused, so a record a future deploy wrote
    with an added field decodes cleanly on an older one. Used only to
    parse; `_decode_trace` reconstructs a genuine `Trace` from the parsed
    fields rather than returning this subclass, so a value round-tripped
    through the store keeps `Trace`'s own runtime type.
    """

    model_config = ConfigDict(extra="ignore")


def _encode_trace(trace: Trace) -> dict[str, Any]:
    return trace.model_dump(mode="json")


def _decode_trace(record: dict[str, Any]) -> Trace:
    lenient = _LenientTrace.model_validate(record)
    return Trace(**lenient.model_dump())


register_trace_codec(
    Trace,
    kind="analysis_mcp_trace",
    version=1,
    encode=_encode_trace,
    decode=_decode_trace,
)
