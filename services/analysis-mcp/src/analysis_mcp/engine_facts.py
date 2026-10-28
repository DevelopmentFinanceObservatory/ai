"""An in-process engine result as the facts a digest renders.

The in-process verb tools (`*_tool.py`), which the registered tools no longer
call, still render their digests. `facts_of` turns their engine result into the
JSON dfo-apis' runner sends as a summary's `result` (`_json`, the form
`dfo_analysis_engine.calls._facts` writes) and reads it as `summary`'s facts, so
both paths render through one parse.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Mapping
from typing import Any

from dfo_frame_grammar import Frame
from dfo_frame_grammar.grammar import ComposedUnit, UnitRequest

from analysis_mcp.engine.aggregate import AggregateResult
from analysis_mcp.engine.compare import CompareResult
from analysis_mcp.engine.filter import FilterResult
from analysis_mcp.engine.growth import GrowthResult
from analysis_mcp.engine.join import JoinResult, MatchedKeys
from analysis_mcp.engine.quotient_of import QuotientOfResult
from analysis_mcp.engine.rank import RankResult
from analysis_mcp.engine.share_of import ShareOfResult
from analysis_mcp.engine.sum_of import SumOfResult
from analysis_mcp.summary import (
    AggregateFacts,
    CompareFacts,
    FilterFacts,
    GrowthFacts,
    JoinFacts,
    QuotientOfFacts,
    RankFacts,
    ResultFacts,
    ShareOfFacts,
    SumOfFacts,
)

_FACTS: Mapping[type, type[ResultFacts]] = {
    AggregateResult: AggregateFacts,
    GrowthResult: GrowthFacts,
    RankResult: RankFacts,
    CompareResult: CompareFacts,
    SumOfResult: SumOfFacts,
    ShareOfResult: ShareOfFacts,
    QuotientOfResult: QuotientOfFacts,
    JoinResult: JoinFacts,
    FilterResult: FilterFacts,
}


def _json(value: object, *, top: bool = False) -> object:
    """`value`, an engine result or a part of one, as JSON: the top-level frame
    left out, a unit in its pin form, join's matched keys as their rows."""
    if isinstance(value, MatchedKeys):
        value = value.rows()
    if isinstance(value, UnitRequest | ComposedUnit):
        return value.as_json()
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            item.name: _json(getattr(value, item.name))
            for item in dataclasses.fields(value)
            if not (top and isinstance(getattr(value, item.name), Frame))
        }
    if isinstance(value, Mapping):
        return {str(key): _json(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json(item) for item in value]
    if value is None or isinstance(value, bool | int | float | str):
        return value
    raise TypeError(f"no JSON form for {type(value)!r}")


def facts_of(result: Any) -> Any:
    """The digest facts of an in-process engine `result`, read from its JSON
    after a round trip through text, as a runner summary's `result` is."""
    facts = _FACTS[type(result)]
    return facts.model_validate(json.loads(json.dumps(_json(result, top=True))))
