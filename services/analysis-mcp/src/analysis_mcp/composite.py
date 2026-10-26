"""What a verb says when it reads a composite frame's StatVars one by one.

dfo-apis mints a composite definition, such as bilateral plus imputed multilateral aid,
as one frame whose rows each belong to one component, and lists the components under
its manifest's `trace.components`, each with its member StatVars. Read with one series
per StatVar (`get_data(stat_vars="separate")`), the frame holds the members of every
component, and a verb that needs one StatVar refuses it (`stat_var_unresolved`), as it
refuses any family. Summing the members here could count a code twice where the family
nests one code inside another, so the refusal points to `get_data(stat_vars="sum")`,
where dfo-apis adds every member of every component into one series per party and year
(`composite_refusal`).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import TYPE_CHECKING

from analysis_mcp.tool_errors import ToolErrorCode, ToolRefusal

if TYPE_CHECKING:
    from common.materialisation import StoredFrameTrace

# The upstream trace member dfo-apis lists a composite frame's components under.
_COMPONENTS_KEY = "components"


def frame_components(trace: StoredFrameTrace) -> list[dict[str, object]]:
    """The components `trace`'s frame was composed from, each as its role and how many
    member StatVars it holds, in dfo-apis' order; empty for a frame of one table."""
    listed = trace.upstream_trace.get(_COMPONENTS_KEY)
    if not isinstance(listed, Sequence):
        return []
    components: list[dict[str, object]] = []
    for component in listed:
        if not isinstance(component, Mapping):
            continue
        stat_vars = component.get("stat_vars")
        components.append(
            {
                "role": component.get("role"),
                "stat_vars": len(stat_vars) if isinstance(stat_vars, Sequence) else 0,
            }
        )
    return components


def composite_refusal(refusal: ToolRefusal, traces: Iterable[StoredFrameTrace]) -> ToolRefusal:
    """`refusal`, or, when it refuses an input frame's several StatVars and one of
    `traces` is a composite frame's, the refusal naming that frame's components and the
    `get_data` read that sums them."""
    if refusal.code is not ToolErrorCode.STAT_VAR_UNRESOLVED:
        return refusal
    distinct = refusal.fix.get("distinct_stat_vars") or []
    if len(distinct) < 2:
        return refusal
    for trace in traces:
        if components := frame_components(trace):
            return ToolRefusal.composite_stat_vars_unsummed(
                distinct_stat_vars=list(distinct), components=components
            )
    return refusal
