"""The tools `server` registers: the nine verbs and `resolve_anchor`, computed by
dfo-apis' analysis runner.

Each verb tool takes the arguments, signature and docstring of its in-process
twin (`aggregate_tool.aggregate` and the rest), builds the same op-spec from
them, and runs it on the runner (`tool_pipeline.run_on_runner`). An argument
that forms no op-spec refuses `op_spec_invalid` before any call, and a frame
argument with no `content_version` refuses `frame_ref_incomplete`. `compare`
takes `sum_of`'s form under `operation="sum"`, and `share_of` takes
`quotient_of`'s under `operation="quotient"`: the form picks the op-spec, so each
keeps its own runner call, result and refusals. A share whose sides hold two
units refuses with `fix` naming the quotient form. A rolling
`growth`, `rank` or `filter` needs an `anchor_key`, checked by
`op_spec.with_anchor_key`. The key reaches no request. The call sends the roots
of the anchor record it names as `anchor_roots`, and the runner checks each
rolling literal against the latest period it finds over them
(`runner_tree.verb_tree`). The call refuses when its `rolling.population` is not
the record's, or when the runner read a root the record was not resolved over
(`runner_client.run_tree`).

`resolve_anchor` asks the runner's anchor route for the latest period of the
root frames over the populations, then stores the anchor record and returns its
`anchor_key`. Its argument checks that read no frame run here first
(`rolling_tool._too_many`, `rolling_tool._unmatched`).

No registered tool calls the in-process tool bodies, the engine or the store
reads behind them.
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
from analysis_mcp.tool_errors import ToolErrorCode, ToolRefusal, to_tool_result
from analysis_mcp.tool_pipeline import build_op_spec_or_refusal, run_on_runner

type Arguments = Mapping[str, Any]
type Tool = Callable[..., Awaitable[ToolResult]]
type Refine = Callable[[Arguments, ToolRefusal], ToolRefusal]

_FRAME_ARGUMENTS = ("frame_ref", "right_frame_ref", "denominator_frame_ref")


def frame_ref_incomplete(arguments: Arguments) -> ToolRefusal | None:
    """The refusal for the first frame argument in `arguments` that names a
    `spec_id` without its `content_version`, or `None` when none does. Any other
    malformed argument is left to the op-spec to refuse."""
    for name in _FRAME_ARGUMENTS:
        ref = arguments.get(name)
        if isinstance(ref, Mapping) and "spec_id" in ref and "content_version" not in ref:
            return ToolRefusal.frame_ref_incomplete(argument=name, spec_id=str(ref["spec_id"]))
    return None


def _on_runner(
    in_process: Tool, build: Callable[[Arguments], OpSpecVariant], *, refine: Refine | None = None
) -> Tool:
    """The registered tool for `in_process`. It keeps that tool's signature and
    docstring, builds its op-spec with `build` from the bound arguments, and
    computes it on the runner. `refine` rewrites a runner refusal given the
    call's arguments."""
    signature = inspect.signature(in_process)

    @functools.wraps(in_process)
    async def tool(*args: Any, **kwargs: Any) -> ToolResult:
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        arguments = bound.arguments
        if (incomplete := frame_ref_incomplete(arguments)) is not None:
            return to_tool_result(incomplete)
        op_spec = build_op_spec_or_refusal(lambda: build(arguments))
        if isinstance(op_spec, ToolResult):
            return op_spec
        return await run_on_runner(
            op_spec,
            anchor_key=arguments.get("anchor_key"),
            refine=None if refine is None else functools.partial(refine, arguments),
        )

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


def _compare(a: Arguments) -> CompareOpSpec | SumOfOpSpec:
    """`compare`'s op-spec, or `sum_of`'s under `operation="sum"`."""
    return _slice_pair(SumOfOpSpec if a["operation"] == "sum" else CompareOpSpec)(a)


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


def _share_of(a: Arguments) -> ShareOfOpSpec | QuotientOfOpSpec:
    """`share_of`'s op-spec, or `quotient_of`'s under `operation="quotient"`."""
    return _two_sided(QuotientOfOpSpec if a["operation"] == "quotient" else ShareOfOpSpec)(a)


def _quotient_for_two_units(a: Arguments, refusal: ToolRefusal) -> ToolRefusal:
    """A share's `cross_frame_unit_mismatch`, rewritten to name the quotient form.
    The runner's `fix` names each side's unit under its label."""
    if a["operation"] != "share" or refusal.code != ToolErrorCode.CROSS_FRAME_UNIT_MISMATCH:
        return refusal
    return ToolRefusal.share_units_differ(
        numerator_unit=refusal.fix.get("numerator"),
        denominator_unit=refusal.fix.get("denominator"),
    )


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
compare = _on_runner(compare_tool.compare, _compare)
growth = _on_runner(growth_tool.growth, _growth)
rank = _on_runner(rank_tool.rank, _rank)
share_of = _on_runner(share_of_tool.share_of, _share_of, refine=_quotient_for_two_units)
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
    refs_by_root: dict[Pair, tuple[FrameRef, ...]] = {}
    for ref in distinct:
        root = await root_node(ref)
        if isinstance(root, ToolRefusal):
            return to_tool_result(root)
        node, pair = root
        nodes.append(node)
        refs_by_root[pair] = (*refs_by_root.get(pair, ()), ref)

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
