"""The tools `server` registers: the nine verbs and `resolve_anchor`, computed by
dfo-apis' analysis runner.

Each verb tool takes the arguments, signature and docstring of its in-process
twin (`aggregate_tool.aggregate` and the rest), builds the same op-spec from
them, and runs it on the runner (`tool_pipeline.run_on_runner`). An argument
that forms no op-spec refuses `op_spec_invalid` before any call, as it did. A
rolling `growth`, `rank` or `filter` still needs an `anchor_key`, checked as
before (`op_spec.with_anchor_key`), and the runner checks each rolling literal
against the latest period it finds, so the key reaches no request.

`resolve_anchor` asks the runner's anchor route for the latest period of the
root frames over the populations, then stores the anchor record and returns its
`anchor_key`, as the in-process tool did, so a plan that passes the key on
keeps working. Its argument checks that read no frame run here first
(`rolling_tool._too_many`, `rolling_tool._unmatched`).

The in-process tool bodies, the engine and the store reads behind them are no
longer called by any registered tool.
"""

from __future__ import annotations

import functools
import inspect
from collections.abc import Awaitable, Callable, Mapping
from typing import Any

from common.materialisation import AnchorRecord, FrameRef, get_store
from common.rolling import canonical_roots
from fastmcp.tools import ToolResult

from analysis_mcp import (
    aggregate_tool,
    compare_tool,
    filter_tool,
    growth_tool,
    join_tool,
    quotient_of_tool,
    rank_tool,
    rolling_tool,
    share_of_tool,
    sum_of_tool,
)
from analysis_mcp.identity import OpSpecVariant
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
    with_anchor_key,
)
from analysis_mcp.runner_client import find_anchor
from analysis_mcp.runner_tree import Pair, root_node
from analysis_mcp.tool_errors import ToolRefusal, to_tool_result
from analysis_mcp.tool_pipeline import build_op_spec_or_refusal, run_on_runner

type Arguments = Mapping[str, Any]
type Tool = Callable[..., Awaitable[ToolResult]]


def _on_runner(in_process: Tool, build: Callable[[Arguments], OpSpecVariant]) -> Tool:
    """The registered tool of `in_process`: its signature and docstring, its
    op-spec built by `build` from the bound arguments, computed by the runner."""
    signature = inspect.signature(in_process)

    @functools.wraps(in_process)
    async def tool(*args: Any, **kwargs: Any) -> ToolResult:
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        op_spec = build_op_spec_or_refusal(lambda: build(bound.arguments))
        if isinstance(op_spec, ToolResult):
            return op_spec
        return await run_on_runner(op_spec)

    return tool


def _or_empty[T](value: T | None, empty: T) -> T:
    return empty if value is None else value


def _aggregate(a: Arguments) -> AggregateOpSpec:
    return AggregateOpSpec(
        frame_ref=a["frame_ref"],
        aggregation=a["aggregation"],
        group_by=tuple(a["group_by"]),
        unit=a["unit"],
    )


def _slice_pair[T: CompareOpSpec | SumOfOpSpec](op_spec: type[T]) -> Callable[[Arguments], T]:
    def build(a: Arguments) -> T:
        return op_spec(
            frame_ref=a["frame_ref"],
            right_frame_ref=a["right_frame_ref"],
            left_entity=_or_empty(a["left_entity"], {}),
            right_entity=_or_empty(a["right_entity"], {}),
            left_unit=a["left_unit"],
            right_unit=a["right_unit"],
        )

    return build


def _growth(a: Arguments) -> GrowthOpSpec:
    return with_anchor_key(
        GrowthOpSpec(
            frame_ref=a["frame_ref"],
            window_start=a["window_start"],
            window_end=a["window_end"],
            entity_by=tuple(a["entity_by"]),
            unit=a["unit"],
            change=a["change"],
            rolling=a["rolling"],
        ),
        a["anchor_key"],
    )


def _rank(a: Arguments) -> RankOpSpec:
    return with_anchor_key(
        RankOpSpec(
            frame_ref=a["frame_ref"],
            rank_by=tuple(a["rank_by"]),
            aggregation=a["aggregation"],
            window_start=a["window_start"],
            window_end=a["window_end"],
            unit=a["unit"],
            ascending=a["ascending"],
            per_period=a["per_period"],
            limit=a["limit"],
            rolling=a["rolling"],
        ),
        a["anchor_key"],
    )


def _filter(a: Arguments) -> FilterOpSpec:
    return with_anchor_key(
        FilterOpSpec(
            frame_ref=a["frame_ref"], predicates=tuple(a["predicates"]), rolling=a["rolling"]
        ),
        a["anchor_key"],
    )


def _two_sided[T: ShareOfOpSpec | QuotientOfOpSpec](op_spec: type[T]) -> Callable[[Arguments], T]:
    def build(a: Arguments) -> T:
        return op_spec(
            frame_ref=a["frame_ref"],
            denominator_frame_ref=a["denominator_frame_ref"],
            numerator_entity=_or_empty(a["numerator_entity"], {}),
            denominator_entity=_or_empty(a["denominator_entity"], {}),
            numerator_unit=a["numerator_unit"],
            denominator_unit=a["denominator_unit"],
            numerator_by=tuple(_or_empty(a["numerator_by"], ())),
            denominator_by=_or_empty(a["denominator_by"], {}),
        )

    return build


def _join(a: Arguments) -> JoinOpSpec:
    return JoinOpSpec(
        frame_ref=a["frame_ref"],
        right_frame_ref=a["right_frame_ref"],
        join_keys=tuple(a["join_keys"]),
        right_keys=_or_empty(a["right_keys"], {}),
        left_unit=a["left_unit"],
        right_unit=a["right_unit"],
    )


aggregate = _on_runner(aggregate_tool.aggregate, _aggregate)
compare = _on_runner(compare_tool.compare, _slice_pair(CompareOpSpec))
growth = _on_runner(growth_tool.growth, _growth)
rank = _on_runner(rank_tool.rank, _rank)
share_of = _on_runner(share_of_tool.share_of, _two_sided(ShareOfOpSpec))
join = _on_runner(join_tool.join, _join)
filter_verb = _on_runner(filter_tool.filter_verb, _filter)
sum_of = _on_runner(sum_of_tool.sum_of, _slice_pair(SumOfOpSpec))
quotient_of = _on_runner(quotient_of_tool.quotient_of, _two_sided(QuotientOfOpSpec))


@functools.wraps(rolling_tool.resolve_anchor)
async def resolve_anchor(populations: Any, roots: Any) -> ToolResult:
    request = build_op_spec_or_refusal(
        lambda: rolling_tool._AnchorRequest(populations=populations, roots=tuple(roots))
    )
    if isinstance(request, ToolResult):
        return request
    if (refusal := rolling_tool._too_many(request) or rolling_tool._unmatched(request)) is not None:
        return refusal

    distinct = [
        FrameRef(**root) for root in canonical_roots(root.model_dump() for root in request.roots)
    ]
    nodes: list[dict[str, Any]] = []
    refs_by_root: dict[Pair, FrameRef] = {}
    for ref in distinct:
        root = await root_node(ref)
        if isinstance(root, ToolRefusal):
            return to_tool_result(root)
        node, pair = root
        nodes.append(node)
        refs_by_root[pair] = ref

    populations_json = [population.model_dump(mode="json") for population in request.populations]
    answer = await find_anchor(nodes, populations_json, refs_by_root=refs_by_root)
    if isinstance(answer, ToolRefusal):
        return to_tool_result(answer)

    record = AnchorRecord(
        populations=populations_json,
        roots=request.roots,
        rule=answer.rule,
        period=answer.period,
        left_out=answer.left_out,
    )
    store = get_store()
    await store.put_anchor_record(record)
    # The record the store keeps is the one the key names.
    kept = await store.get_anchor_record(record.anchor_key) or record
    result = rolling_tool.ResolvedAnchor(
        anchor_key=record.anchor_key, period=kept.period, left_out=kept.left_out
    )
    return ToolResult(
        content=(
            f"The latest period is {result.period} (anchor_key {result.anchor_key}). "
            f"{len(result.left_out)} of the series counted hold no value at it."
        ),
        structured_content=result.model_dump(mode="json"),
        is_error=False,
    )
