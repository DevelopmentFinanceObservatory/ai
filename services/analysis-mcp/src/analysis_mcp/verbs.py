"""The tools `server` registers: the nine verbs and `resolve_anchor`, computed by
dfo-apis' analysis runner.

Each verb tool takes the arguments, signature and docstring of its in-process
twin (`aggregate_tool.aggregate` and the rest), builds the same op-spec from
them, and runs it on the runner (`tool_pipeline.run_on_runner`). An argument
that forms no op-spec refuses `op_spec_invalid` before any call. A frame
argument naming a `spec_` id alone is read at the Spec's newest version
(`runner_tree.newest_ref`), and one naming any other id alone refuses
`frame_ref_incomplete`. `rank`, `aggregate` and `share_of` also take
`place_level`, which no in-process twin has: it filters each input to one level
of place on the runner (`runner_tree.verb_tree`). `compare`
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
from typing import Annotated, Any

from common.materialisation import AnchorRecord, FrameRef, get_store
from common.place_type import PlaceLevel
from common.rolling import canonical_roots
from fastmcp.tools import ToolResult
from pydantic import Field

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
from analysis_mcp.runner_tree import Pair, newest_ref, root_node
from analysis_mcp.tool_errors import ToolErrorCode, ToolRefusal, to_tool_result
from analysis_mcp.tool_pipeline import build_op_spec_or_refusal, run_on_runner

type Arguments = Mapping[str, Any]
type Tool = Callable[..., Awaitable[ToolResult]]
type Refine = Callable[[Arguments, ToolRefusal], ToolRefusal]

_FRAME_ARGUMENTS = ("frame_ref", "right_frame_ref", "denominator_frame_ref")

_PLACE_LEVEL_DESCRIPTION = (
    "Keep only the places at this level before the call, such as Country. Set it when the "
    "frame holds places of several levels and the call refuses place_levels_mixed."
)
# `share_of` filters its denominator too.
_SHARE_PLACE_LEVEL_DESCRIPTION = (
    f"{_PLACE_LEVEL_DESCRIPTION} The denominator is filtered too, so a denominator bound to a "
    "group entity such as the world is left out."
)


def _place_level(description: str) -> inspect.Parameter:
    """The `place_level` argument `rank`, `aggregate` and `share_of` take beside their
    twins' own."""
    return inspect.Parameter(
        "place_level",
        inspect.Parameter.KEYWORD_ONLY,
        default=None,
        annotation=Annotated[PlaceLevel | None, Field(default=None, description=description)],
    )


def _bare(ref: Any) -> str | None:
    """The `spec_id` of a frame argument that names one with no `content_version`."""
    if isinstance(ref, Mapping) and "spec_id" in ref and "content_version" not in ref:
        return str(ref["spec_id"])
    return None


def frame_ref_incomplete(arguments: Arguments) -> ToolRefusal | None:
    """The refusal for the first frame argument in `arguments` that names an id other
    than a `spec_` id without its `content_version`, or `None` when none does. Any
    other malformed argument is left to the op-spec to refuse."""
    for name in _FRAME_ARGUMENTS:
        spec_id = _bare(arguments.get(name))
        if spec_id is not None and not spec_id.startswith("spec_"):
            return ToolRefusal.frame_ref_incomplete(argument=name, spec_id=spec_id)
    return None


def _held_newest(arguments: Arguments) -> tuple[dict[str, Any], tuple[str, ...]]:
    """`arguments` with each bare `spec_` frame argument held as `newest_ref`, and
    the Specs those name, each once."""
    held = dict(arguments)
    newest: dict[str, None] = {}
    for name in _FRAME_ARGUMENTS:
        spec_id = _bare(arguments.get(name))
        if spec_id is not None:
            held[name] = newest_ref(spec_id).model_dump()
            newest[spec_id] = None
    return held, tuple(newest)


def _on_runner(
    in_process: Tool,
    build: Callable[[Arguments], OpSpecVariant],
    *,
    refine: Refine | None = None,
    place_level_description: str | None = None,
) -> Tool:
    """The registered tool for `in_process`. It keeps that tool's signature and
    docstring, builds its op-spec with `build` from the bound arguments, and
    computes it on the runner. `refine` rewrites a runner refusal given the
    call's arguments. `place_level_description`, when given, adds `place_level` to the
    signature with that description."""
    signature = inspect.signature(in_process)

    @functools.wraps(in_process)
    async def tool(*args: Any, **kwargs: Any) -> ToolResult:
        place_level: PlaceLevel | None = (
            kwargs.pop("place_level", None) if place_level_description is not None else None
        )
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        arguments = bound.arguments
        if (incomplete := frame_ref_incomplete(arguments)) is not None:
            return to_tool_result(incomplete)
        held, newest = _held_newest(arguments)
        op_spec = build_op_spec_or_refusal(lambda: build(held))
        if isinstance(op_spec, ToolResult):
            return op_spec
        return await run_on_runner(
            op_spec,
            anchor_key=arguments.get("anchor_key"),
            place_level=place_level,
            newest=newest,
            refine=None if refine is None else functools.partial(refine, arguments),
        )

    if place_level_description is not None:
        place_level = _place_level(place_level_description)
        tool.__signature__ = signature.replace(  # ty: ignore[unresolved-attribute]  # inspect reads it
            parameters=[*signature.parameters.values(), place_level]
        )
        tool.__annotations__ = {**in_process.__annotations__, "place_level": place_level.annotation}
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


aggregate = _on_runner(
    aggregate_tool.aggregate, _aggregate, place_level_description=_PLACE_LEVEL_DESCRIPTION
)
compare = _on_runner(compare_tool.compare, _compare)
growth = _on_runner(growth_tool.growth, _growth)
rank = _on_runner(rank_tool.rank, _rank, place_level_description=_PLACE_LEVEL_DESCRIPTION)
share_of = _on_runner(
    share_of_tool.share_of,
    _share_of,
    refine=_quotient_for_two_units,
    place_level_description=_SHARE_PLACE_LEVEL_DESCRIPTION,
)
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
