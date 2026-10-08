"""The description of a StatVar an operation minted, and the one way to read
the descriptions a stored trace carries.

`sum_of`, `quotient_of` and `share_of` write their result rows under a
StatVar of their own, which no graph names. A `ComposedStatVar` says what
such a StatVar is made of, so a reader can name it from its parts.
analysis-mcp stores the descriptions on its trace and presentation-mcp sends
them with a published frame. Neither service imports the other, so the model
lives here.
"""

from __future__ import annotations

from typing import Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict

from common.materialisation.trace import MaterialisationTrace, StoredFrameTrace


class ComposedStatVar(BaseModel):
    """One StatVar an operation minted, described by the StatVars of its two sides.

    `relation="quotient"` reads `first` divided by `second`: the numerator's
    StatVar, then the denominator's. `relation="sum"` reads `first` plus
    `second`: the left side's StatVars, then the right side's.
    `relation="share"` reads `first` as a per cent share of `second`, the
    numerator's StatVar then the denominator's. A share whose two sides
    hold one StatVar, a part of a total, reads "Share of <side>". A side ref
    may name a graph StatVar or another composed StatVar.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    stat_var: str
    relation: Literal["quotient", "share", "sum"]
    first: tuple[str, ...]
    second: tuple[str, ...]


@runtime_checkable
class _CarriesComposedStatVars(Protocol):
    @property
    def composed_stat_vars(self) -> tuple[ComposedStatVar, ...]: ...


def composed_stat_vars_of(trace: StoredFrameTrace) -> tuple[ComposedStatVar, ...]:
    """The composed StatVars `trace` describes, in dependency order.

    A `MaterialisationTrace` describes a frame no operation computed, so it
    carries none. Every other stored trace must declare
    `composed_stat_vars`.

    Raises:
        TypeError: `trace` is not a `MaterialisationTrace` and has no
            `composed_stat_vars` member.
    """
    if isinstance(trace, MaterialisationTrace):
        return ()
    if not isinstance(trace, _CarriesComposedStatVars):
        raise TypeError(
            f"{type(trace).__name__} is not a MaterialisationTrace and declares no "
            "composed_stat_vars"
        )
    return tuple(trace.composed_stat_vars)
