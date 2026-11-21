"""The frame `content_version` a stored frame is verified against, and the
derived-ness marker of a frame an in-process verb stored.

`derive_frame_content_version` hashes a frame's canonical bytes into its
frame-grammar `cv_`, and is registered as the store's deriver
(`common.materialisation.register_content_version_deriver`).

`read_derived_marker` reads `applied_policy.detail["derived"]`, the marker a
producer verb wrote onto its output's trace before analysis-mcp's verbs computed
on dfo-apis' analysis runner, and `DerivedMarker` names its flags. The harness
reads it from frames the store still holds (`harness.orchestrator_tools`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from common.materialisation import register_content_version_deriver
from dfo_frame_grammar.content_version import frame_sha256, mint_content_version

from analysis_mcp.trace import Trace

if TYPE_CHECKING:
    from common.materialisation import StoredFrameTrace
    from dfo_frame_grammar import Frame


def derive_frame_content_version(*, frame: Frame) -> str:
    """`frame`'s frame-grammar `cv_` content version: a hash of its canonical
    bytes alone, so one frame has one token whatever lineage produced it.

    The store calls it to verify a frame it reads
    (`common.materialisation.register_content_version_deriver`), and an
    in-process verb called it when it wrote its output
    (`Trace.frame_content_version`).

    A `frame_ref` pair names either an original materialisation (a
    `spec_`/`cv_` pair, `get_data`'s own fixtures included) or a prior toolkit result (a chained verb call, an
    `ops_`/`ocv_` pair); this function returns the frame-grammar `cv_`
    family in both cases -- a chained result's own `Trace.content_version`
    carries the toolkit's `ocv_` family instead, which
    `mint_toolkit_content_version`'s `input_content_versions` parameter
    requires the `cv_` family for.
    """
    return mint_content_version(frame_sha256=frame_sha256(frame))


def _derive_content_version_for_store(frame: Frame) -> str:
    """Adapts `derive_frame_content_version`'s keyword-only signature to the
    positional `(frame)` shape `common.materialisation.content_version`'s
    registry calls a deriver with."""
    return derive_frame_content_version(frame=frame)


register_content_version_deriver(_derive_content_version_for_store)


@dataclass(frozen=True, kw_only=True)
class DerivedMarker:
    """What a producer verb recorded in `applied_policy.detail["derived"]`,
    naming which of a derived result's inherited fields (`unit`, `value`,
    `stat_var`) no longer mean what they meant on the input.

    `dimensionless` marks a value that is a ratio already, either a relative
    `growth` rate, whose rows carry `unit=None`, or a `share_of` result, whose
    rows carry the per cent unit (`common.unit_refs.PERCENT_UNIT`) and a
    StatVar of its own. `quotient_of` refuses a `dimensionless` side
    (`quotient_of_side_unit_unsupported`), since a ratio is no
    amount to be a part of the composed unit; no other verb refuses on it,
    and downstream unit resolution already refuses a genuine cross-unit
    combination without its help.

    `inherited_stat_var` is set when `growth`'s output inherits its input's
    `stat_var` (always). It is carried for a human reading the trace, and no
    verb refuses on it.

    `ordinal` -- a per-period `rank` with no `limit` carries the rank
    position as its output row's `value`, not the measured value the
    collapsing and limited modes emit. A ratio, a relative growth rate or a
    ranking over a position is not a meaningful number, so `share_of`,
    `quotient_of`, a relative `growth` and `rank` all refuse on this flag
    (`derived_input_ordinal`),
    independently of whatever the harness itself checks. Merged by OR like
    every other flag here -- a growth-over-a-rank output also carries
    `ordinal` onward, which over-claims (a difference of positions is a
    count of places, not itself an ordinal) but is correct in effect:
    nothing downstream of a rank position is a ratio-scale quantity either
    way, and a rank over a change in places refuses with the positions.
    """

    dimensionless: bool = False
    inherited_stat_var: bool = False
    ordinal: bool = False
    reason: str = ""


def read_derived_marker(trace: StoredFrameTrace) -> DerivedMarker | None:
    """`trace.applied_policy.detail["derived"]`, parsed into a `DerivedMarker`,
    or `None` if this input carries no such marker -- either because it
    wasn't itself a derived result, or because it's a bare `MaterialisationTrace`
    (a raw `get_data` frame nobody's verb has touched yet), which carries no
    `applied_policy` at all.

    Keys outside `DerivedMarker`'s fields are ignored, so a stored trace
    carrying any other flag still resolves.
    """
    if not isinstance(trace, Trace):
        return None
    raw = trace.applied_policy.detail.get("derived")
    if raw is None:
        return None
    return DerivedMarker(
        dimensionless=bool(raw.get("dimensionless", False)),
        inherited_stat_var=bool(raw.get("inherited_stat_var", False)),
        ordinal=bool(raw.get("ordinal", False)),
        reason=str(raw.get("reason", "")),
    )
