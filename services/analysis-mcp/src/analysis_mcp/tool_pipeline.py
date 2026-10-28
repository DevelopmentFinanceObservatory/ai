"""The shared tail of every verb tool's own body: turning a verb's own
already-parsed parameters into a validated op-spec, minting a toolkit
result's identity, and storing the computed frame under its own pair before
rendering its digest.

`frame_ref_resolution.py` already owns the frame_ref/array-parameter/
derived-marker half of the nine verbs' shared shape -- the half with a
`frame_ref` to resolve. This module owns the rest: constructing an op-spec,
and the mint-identity/store/render/return sequence every verb's own body
ends with, once its own engine call (and any coverage-floor check) has
already succeeded. What sits between those two halves -- the engine call
itself, and mapping its own typed exceptions to a refusal -- stays inline in
each verb tool: that mapping is genuinely each verb's own.

The registered tools (`verbs`) compute in dfo-apis' analysis runner instead:
`run_on_runner` sends a validated op-spec as a one-node tree
(`runner_tree.verb_tree`) and turns the runner's summary into the reply, its
`VerbResult` and its digest (`digest.render_summary_digest`). The rest of this
module serves the in-process tools, which no registered tool calls.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from common.caller import current_writer
from common.materialisation import FrameRef, FrameTooLarge, get_store
from fastmcp.tools import ToolResult
from pydantic import BaseModel, ValidationError

from analysis_mcp.artifact import Exclusion
from analysis_mcp.digest import render_summary_digest
from analysis_mcp.execution_identity import ExecutionIdentity, current_execution_identity
from analysis_mcp.frame_ref_resolution import call_row_claim, op_spec_hash_or_refusal
from analysis_mcp.identity import mint_toolkit_content_version
from analysis_mcp.result import VerbResult, observed_shape
from analysis_mcp.runner_client import run_tree
from analysis_mcp.runner_tree import verb_tree
from analysis_mcp.tool_errors import ToolRefusal, to_tool_result

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from dfo_frame_grammar import Frame

    from analysis_mcp.engine.common import Exclusions
    from analysis_mcp.identity import OpSpecVariant
    from analysis_mcp.steps import Step
    from analysis_mcp.trace import Trace


def build_op_spec_or_refusal[T: BaseModel](build: Callable[[], T]) -> T | ToolResult:
    """Call `build` -- a verb's own op-spec constructor, bound to its
    already-parsed parameters -- catching the `pydantic.ValidationError`
    every verb maps to the identical `op_spec_invalid` refusal. Every verb's
    own body calls this first, before anything else. `resolve_anchor` builds
    its request model through it for the same refusal.
    """
    try:
        return build()
    except ValidationError as exc:
        return to_tool_result(ToolRefusal.op_spec_invalid(detail=_render_validation_errors(exc)))


_MAX_RENDERED_INPUT_CHARS = 80


def _render_validation_errors(exc: ValidationError) -> str:
    """One line per error in `exc`, each carrying the dotted field location
    and the message, plus the offending input when its own repr is short
    enough to stay readable. Pydantic's own `ValidationError.__str__` also
    prepends a "N validation errors for `<ClassName>`" header naming the
    internal model class and appends a link to pydantic's own error-code
    docs per error -- neither belongs in a refusal a caller reads, so this
    renders the errors itself instead of calling that method.
    """
    lines = []
    for error in exc.errors():
        location = ".".join(str(part) for part in error["loc"])
        line = f"{location}: {error['msg']}"
        if "input" in error:
            rendered_input = repr(error["input"])
            if len(rendered_input) <= _MAX_RENDERED_INPUT_CHARS:
                line = f"{line} (got {rendered_input})"
        lines.append(line)
    return "\n".join(lines)


def mint_identity_or_refusal(
    op_spec: OpSpecVariant, *, input_content_versions: Sequence[str]
) -> tuple[str, ExecutionIdentity, str] | ToolResult:
    """`(op_spec_hash, execution_identity, content_version)` for `op_spec`,
    minted over `input_content_versions`, or the `op_spec_invalid` refusal
    minting `op_spec_hash` can hit (see `op_spec_hash_or_refusal`'s own
    docstring for the two failure modes it catches). Every verb calls this
    once, right after its own engine call (and any coverage-floor check) has
    already succeeded.

    Callers compute their own `input_content_versions` before calling this,
    so `op_spec_hash_or_refusal` runs after that work rather than before it.
    That ordering is only safe while both of its failure modes are already
    ruled out by the time a verb gets here: a duplicate in an array-shaped
    field is refused earlier by `canonical_columns_or_refusal`/
    `canonical_predicates_or_refusal`, which run the same normalisation
    functions, and every numeric operand an op-spec carries
    (`OrderedThresholdPredicate.value` included) refuses a non-finite value
    at its own field validator. An array-shaped field that skips those
    canonicalisers, or a numeric field admitting a non-finite value, would
    reopen a path where this refuses too late to be the refusal a caller
    sees.
    """
    op_spec_hash = op_spec_hash_or_refusal(op_spec)
    if isinstance(op_spec_hash, ToolResult):
        return op_spec_hash
    execution_identity = current_execution_identity()
    content_version = mint_toolkit_content_version(
        op_spec_hash=op_spec_hash,
        input_content_versions=input_content_versions,
        execution_identity=execution_identity,
    )
    return op_spec_hash, execution_identity, content_version


def exclusions_for_trace(exclusions: Exclusions) -> tuple[Exclusion, ...]:
    """`exclusions` (an engine result's `analysis_mcp.engine.common.
    Exclusions` accumulator), wrapped into the pydantic `Exclusion` models
    `Trace.exclusions` carries. Every verb tool calls this once when building
    its output `Trace`.
    """
    return tuple(
        Exclusion(
            reason=record.reason,
            rows=record.row_count,
            entities=record.entities,
            entities_truncated=record.entities_truncated,
            side=record.side,
        )
        for record in exclusions.records
    )


async def store_result_and_render(
    *,
    frame: Frame,
    trace: Trace,
    digest: Callable[[], str],
    step: Step | None = None,
    dropped_key_count: int = 0,
    right_dropped_key_count: int = 0,
    ranked_before_limit: int | None = None,
) -> ToolResult:
    """Store `frame`/`trace` under the pair identity `trace` already carries
    (`trace.op_spec_hash`, `trace.content_version`), and render `digest` --
    a verb's own `render_x_digest` bound to every argument it already has in
    scope, taking none from here -- into the returned `ToolResult`. The
    lines every verb's own body ends with, once its own output `Trace` is
    built.

    The stored trace is `trace` with its `steps` set to `step`, the record of
    the verb's own operation, or empty when the caller passes none.

    A durable store's `put` can raise `FrameTooLarge` for a result over its
    own row ceiling -- a two-sided verb (`join`'s own doubled output rows
    are the reproduced case) can legitimately compute a result larger than
    either of its own already-admitted inputs. That's caught here and
    returned as the `frame_too_large` refusal rather than a success: no
    object was written, so there is no reference to hand back.

    Runs inside a `holds_input_rows` verb call, whose row claim covers the
    write until its worker thread finishes, even when the call is cancelled
    first; outside one it raises `RuntimeError`.

    The returned `structured_content` is built from a real `VerbResult(...)`
    -- `role_keys`/`stat_var_count` read off `frame` itself via
    `observed_shape`, so the payload matches `result.VERB_OUTPUT_SCHEMA`,
    the schema `server._register` declares at each verb's registration site.
    `dropped_key_count` and `right_dropped_key_count` are the numbers of
    the first and second inputs' keys the verb left out of `frame`, which
    `frame` cannot state. A keyed `quotient_of` or `share_of` passes the
    first, a `join` both.
    `ranked_before_limit` is how many entries a limited `rank` ranked before
    its limit cut them, which `frame` cannot state. Only `rank` passes one.
    """
    trace = trace.model_copy(update={"steps": () if step is None else (step,)})
    ref = FrameRef(spec_id=trace.op_spec_hash, content_version=trace.content_version)
    try:
        # A durable store's `put` encodes and uploads in a worker thread that a
        # cancelled call leaves running, so the call's row claim lasts until it ends.
        await call_row_claim().hold_through(
            get_store().put(ref, frame=frame, trace=trace, writer=current_writer())
        )
    except FrameTooLarge as exc:
        return to_tool_result(ToolRefusal.frame_too_large_on_write(reason=exc.reason))
    role_keys, stat_var_count = observed_shape(frame)
    result = VerbResult(
        spec_id=ref.spec_id,
        content_version=ref.content_version,
        frame_content_version=trace.frame_content_version,
        rows=frame.num_rows,
        role_keys=role_keys,
        stat_var_count=stat_var_count,
        dropped_key_count=dropped_key_count,
        right_dropped_key_count=right_dropped_key_count,
        ranked_before_limit=ranked_before_limit,
        exclusions=trace.exclusions,
        time=trace.time,
    )
    return ToolResult(
        content=digest(),
        structured_content=result.model_dump(mode="json"),
        is_error=False,
    )


async def run_on_runner(op_spec: OpSpecVariant, *, anchor_key: str | None = None) -> ToolResult:
    """`op_spec` computed by the runner: the reply every registered verb returns.
    A rolling call's `anchor_key` sends the roots of its anchor record beside the
    tree (`runner_tree.verb_tree`).

    `structured_content` is the summary's `VerbResult` fields and `content` the
    digest rendered from its facts, as the in-process tools built both. A frame
    the tree cannot name, and every runner refusal, comes back as its refusal
    (`runner_client`).
    """
    tree = await verb_tree(op_spec, anchor_key=anchor_key)
    if isinstance(tree, ToolRefusal):
        return to_tool_result(tree)
    summary = await run_tree(
        tree.node, anchor_roots=tree.anchor_roots, refs_by_root=tree.refs_by_root
    )
    if isinstance(summary, ToolRefusal):
        return to_tool_result(summary)
    return ToolResult(
        content=render_summary_digest(summary),
        structured_content=summary.verb_result().model_dump(mode="json"),
        is_error=False,
    )


def dedupe_input_content_versions(first: str, second: str) -> list[str]:
    """`[first]` when `second` names the identical content_version,
    otherwise `[first, second]` -- the two-sided verbs' own (`compare`,
    `join`, `share_of`, `sum_of`, `quotient_of`) rule for listing each distinct input
    content_version once in `Trace.input_content_versions`, whether or not
    their own second side turned out to share the first side's
    content_version.
    """
    return [first] if second == first else [first, second]


def entity_label(entity_key: tuple[tuple[str, str], ...]) -> str:
    """A JSON-object-key-safe label for an entity key tuple, for a trace's
    own `applied_policy.detail["per_entity_coverage"]` mapping (`growth`,
    `rank`).
    """
    if not entity_key:
        return "(all rows)"
    return ", ".join(f"{column}={value}" for column, value in entity_key)
