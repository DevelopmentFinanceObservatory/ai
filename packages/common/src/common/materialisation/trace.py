"""The trace shapes `MaterialisationStore` reads and writes.

`common` never imports `analysis_mcp`. Structural typing is what lets
`analysis_mcp.trace.Trace` satisfy `StoredFrameTrace` without either module
depending on the other.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


class StoredFrameTrace(Protocol):
    """Structural: anything with `upstream_trace` and `frame_content_version`
    satisfies this, whether or not it carries op_spec_hash/execution_identity/
    applied_policy."""

    @property
    def upstream_trace(self) -> dict[str, Any]: ...

    @property
    def frame_content_version(self) -> str:
        """The stored frame's own frame-grammar `cv_` content version.

        For a direct materialisation (`MaterialisationTrace`), this is the
        manifest's own minted token, already known at registration and
        equal to the `FrameRef.content_version` it's stored under. For a
        toolkit verb result (`analysis_mcp.trace.Trace`), this is the value
        derived once when that result was written.
        """
        ...


@dataclass(frozen=True, kw_only=True)
class MaterialisationTrace:
    """The trace for a frame nobody computed -- a direct materialisation,
    not a toolkit verb result. Carries only what's true of that: the
    upstream lineage, and the pair it's named by. No op_spec_hash, no
    execution_identity, no applied_policy, and no `kind`/discriminator
    field -- nothing reads one. The store types traces structurally via
    `StoredFrameTrace`, and the one place that tells a toolkit trace apart
    from a bare materialisation (`read_derived_marker`) does it with
    `isinstance(trace, Trace)`, not by inspecting a tag.

    `options` is the dfo-apis mint request's `options` the frame was registered
    from, or `None` for a frame registered before the trace kept them. A frame
    registered under a pair `data_mcp.frame_materialisation.frame_ref_for` derives
    from its columns and groupby carries what that pair was derived from, which
    `data_mcp.mint_recovery` reads when the frame has no mint record.

    `enrichment_stamp` is the stamp the pair is named under
    (`data_mcp.frame_materialisation.recorded_stamp`): set on a frame minted with
    party kinds whose artifact carried one, and `None` on any other frame and on one
    registered with a trace that lacks the member.
    """

    upstream_trace: dict[str, Any]
    spec_id: str
    content_version: str
    options: dict[str, Any] | None = None
    enrichment_stamp: str | None = None

    @property
    def frame_content_version(self) -> str:
        """`content_version` under this trace's own name. A direct
        materialisation's `content_version` is the stored frame's own
        `cv_`, minted upstream."""
        return self.content_version
