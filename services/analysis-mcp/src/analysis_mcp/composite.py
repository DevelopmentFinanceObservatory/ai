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

from collections.abc import Iterable
from typing import TYPE_CHECKING

from common.composite import manifest_components

from analysis_mcp.tool_errors import ToolErrorCode, ToolRefusal

if TYPE_CHECKING:
    from common.materialisation import StoredFrameTrace


def composite_refusal(refusal: ToolRefusal, traces: Iterable[StoredFrameTrace]) -> ToolRefusal:
    """`refusal`, or, when it refuses an input frame's several StatVars and those are
    members of one of `traces`' composite components, the refusal naming that frame's
    components and the `get_data` read that sums them.

    A verb reading two frames refuses one side's StatVars, so the composite whose
    components hold them is the side refused; a refusal of a plain frame beside a
    composite stays the plain one.
    """
    if refusal.code is not ToolErrorCode.STAT_VAR_UNRESOLVED:
        return refusal
    distinct = refusal.fix.get("distinct_stat_vars") or []
    if len(distinct) < 2:
        return refusal
    for trace in traces:
        components = manifest_components(trace.upstream_trace)
        members = {stat_var for component in components for stat_var in component.stat_vars}
        if members.intersection(distinct):
            return ToolRefusal.composite_stat_vars_unsummed(
                distinct_stat_vars=list(distinct),
                components=[
                    {"role": component.role, "stat_vars": len(component.stat_vars)}
                    for component in components
                ],
            )
    return refusal
