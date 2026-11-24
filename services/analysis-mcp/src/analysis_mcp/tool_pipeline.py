"""The shared tail of every registered verb tool: turning a verb's own
already-parsed parameters into a validated op-spec (`build_op_spec_or_refusal`),
and computing it on dfo-apis' analysis runner (`run_on_runner`).

The registered tools (`verbs`) compute in dfo-apis' analysis runner.
`run_on_runner` sends a validated op-spec as a one-node tree
(`runner_tree.verb_tree`) and turns the runner's summary into the reply, its
`VerbResult` and its digest (`digest.render_summary_digest`), whose breakdown
cells read as their labels (`common.breakdown_labels`), read from the graph
nodes of the cells the digest names (`digest.digest_cells`). A digest with no
cell labelled is rendered once.

On a public call (`rebuild_lost_frames`), a call or a replay refused with a code the same
request can get past (`common.refusal.REPOST_RECOVERS_CODES`, the upstream refreshing under
the read) is sent once more when a send as long as the first still ends before the combined
server's call limit (`common.call_limit`), and a second such refusal, or one with too little
time left, answers the call as the runner wrote it (`_sent_again_once`). A frame the runner
no longer holds (`ToolRefusal.lost_frames`) is rebuilt and the call sent again: a `spec_`
frame as the Spec root that fetches it at its newest version in its recorded rendering
(`runner_tree.spec_root`), an `ops_` frame as the pair the runner computes from its stored recipe on current data
(`runner_client.replay_derived`). The runner names one lost dataset per refusal, so a
call over two lost datasets is sent again once per side, at most once per distinct frame
argument. A `spec_` frame whose rendering
cannot be established refuses as `spec_root` says. A replay the runner cannot run
yet refuses with its own delay. When any other replay fails, the call refuses with
the lost-frame refusal and the sentence that says so (`ToolRefusal.not_rebuilt`). The
call sent again keeps its own refusal, and gains that sentence only when it names a
frame still lost.
The frame each bare `spec_id` and each fetched-again `spec_` frame was read as is
recorded as its mint record (`runner_tree.record_spec_reads`), and the digest
opens with a sentence naming that frame's pair, and one naming the pair each rebuilt
frame was read as when it moved, since a replay can mint a new `ops_` id.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from common.breakdown_labels import label_cells
from common.call_limit import EXTERNAL_CALL_LIMIT_S, EXTERNAL_CALL_MARGIN_S
from common.materialisation import FrameRef
from common.refusal import REPOST_RECOVERS_CODES
from fastmcp.tools import ToolResult
from pydantic import BaseModel, ValidationError

from analysis_mcp.digest import digest_cells, render_summary_digest
from analysis_mcp.runner_client import Pair, read_graph_nodes, replay_derived, run_tree
from analysis_mcp.runner_tree import Tree, newest_ref, record_spec_reads, spec_root, verb_tree
from analysis_mcp.summary import VerbSummary
from analysis_mcp.tool_errors import ToolErrorCode, ToolRefusal, to_tool_result

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Collection, Mapping

    from common.place_type import PlaceLevel

    from analysis_mcp.identity import OpSpecVariant


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


@dataclass(frozen=True, kw_only=True)
class _Rebuilt:
    """A lost frame `ref` and the root `node` that stands for it."""

    ref: FrameRef
    node: dict[str, Any]


def _transient(refusal: ToolRefusal) -> bool:
    """Whether `refusal` is a busy, still-building or failing service's, which the same
    call outlives."""
    return refusal.code is ToolErrorCode.SERVICE_UNAVAILABLE or refusal.retry_after_s is not None


# The clock a public call's deadline is read on.
_clock = time.monotonic


async def _sent_again_once[T](
    send: Callable[[], Awaitable[T | ToolRefusal]], *, deadline: float
) -> T | ToolRefusal:
    """What `send` answers, sent once more at once when it refuses with a code the same
    request can get past (`common.refusal.REPOST_RECOVERS_CODES`) and a send as long as
    the first still ends by `deadline` on `_clock`. Otherwise, and on a second refusal,
    the refusal stands."""
    started = _clock()
    answer = await send()
    if not (isinstance(answer, ToolRefusal) and answer.code in REPOST_RECOVERS_CODES):
        return answer
    ended = _clock()
    if deadline - ended < ended - started:
        return answer
    return await send()


async def _rebuilt(
    ref: FrameRef, *, lost: ToolRefusal, default_enrichment: bool, deadline: float
) -> _Rebuilt | ToolRefusal:
    """The root that rebuilds `ref`, which the refusal `lost` names, or the refusal the
    call answers when no root can. A replay is sent once more by `deadline`
    (`_sent_again_once`)."""
    if ref.spec_id.startswith("ops_"):
        replayed = await _sent_again_once(lambda: replay_derived(ref), deadline=deadline)
        if isinstance(replayed, ToolRefusal):
            if replayed.code in REPOST_RECOVERS_CODES or _transient(replayed):
                return replayed
            return lost.not_rebuilt()
        return _Rebuilt(ref=ref, node=replayed.model_dump())
    root = await spec_root(ref, default_enrichment=default_enrichment)
    return root if isinstance(root, ToolRefusal) else _Rebuilt(ref=ref, node=root)


def _pairs(frames: tuple[FrameRef, ...]) -> str:
    """Each frame as the `spec_id` and `content_version` a later call names it by."""
    return "; ".join(f"{frame.spec_id}, {frame.content_version}" for frame in frames)


def _rebuilt_sentence(rebuilt: _Rebuilt, read: Mapping[str, tuple[FrameRef, ...]]) -> str | None:
    """The sentence naming a rebuilt frame's old and new pair, `None` when its pair
    did not move or the new one is unknown."""
    ref = rebuilt.ref
    if "content_version" in rebuilt.node:
        now = (FrameRef.model_validate(rebuilt.node),)
        how = "computed again from its recipe on current data"
    else:
        now = read.get(ref.spec_id, ())
        how = "fetched again at its newest version"
    if not now or now == (ref,):
        return None
    return (
        f"The frame {ref.spec_id}, {ref.content_version} was no longer held, so it was "
        f"{how}: {_pairs(now)} in place of {ref.content_version}."
    )


def _newest_sentence(spec_id: str, read: Mapping[str, tuple[FrameRef, ...]]) -> str:
    """The sentence naming the frame a bare `spec_id` was read as."""
    if not (frames := read.get(spec_id)):
        return f"{spec_id} was read at its newest version."
    return f"{spec_id} was read at its newest version as {_pairs(frames)}."


def _frame_count(op_spec: OpSpecVariant) -> int:
    """How many distinct frames `op_spec`'s frame arguments name."""
    return len(
        {
            (value.spec_id, value.content_version)
            for _, value in op_spec
            if isinstance(value, FrameRef)
        }
    )


async def run_on_runner(
    op_spec: OpSpecVariant,
    *,
    anchor_key: str | None = None,
    place_level: PlaceLevel | None = None,
    newest: Collection[str] = (),
    follows_tree: bool = False,
    rebuild_lost_frames: bool = False,
    refine: Callable[[ToolRefusal], ToolRefusal] | None = None,
    heading: str | None = None,
) -> ToolResult:
    """The reply every registered verb returns for `op_spec`, computed by the runner.
    A rolling call's `anchor_key` sends the roots of its anchor record beside the
    tree (`runner_tree.verb_tree`), and its answer is checked against them
    (`runner_client.run_tree`). `place_level` filters each input to that level
    first. `follows_tree` lets a derived input's rolling years follow the call's
    tree (`run_tree`). `newest` are the Specs passed as bare `spec_id`s, held in
    `op_spec` under `runner_tree.newest_ref` and read at their newest version. With
    `rebuild_lost_frames`, the public server's, a lost frame is rebuilt and a call the
    upstream refreshed under is sent once more (see the module docstring). Without it,
    the runner's refusal is returned, and the harness sends a step again itself
    (`executor.classify.repost_recovers`). `refine`, when given, rewrites a runner
    refusal before it is returned. `heading`, when given, opens the digest of a
    `sum_of` or `quotient_of` computed under another tool's name.

    `structured_content` is the summary's `VerbResult` fields and `content` the
    digest rendered from its facts, as the in-process tools built both. A frame
    the tree cannot name, and every runner refusal, comes back as its refusal
    (`runner_client`).
    """
    replaced: dict[Pair, dict[str, Any]] = {
        (spec_id, newest_ref(spec_id).content_version): {"spec_id": spec_id} for spec_id in newest
    }
    sent: Tree | None = None

    async def run() -> VerbSummary | ToolRefusal:
        nonlocal sent
        # Only the public server sets `follows_tree`.
        tree = await verb_tree(
            op_spec,
            anchor_key=anchor_key,
            place_level=place_level,
            replaced=replaced,
            default_enrichment=follows_tree,
        )
        if isinstance(tree, ToolRefusal):
            return tree
        sent = tree
        return await run_tree(
            tree.node,
            anchor=tree.anchor,
            refs_by_root=tree.refs_by_root,
            spec_roots=tree.spec_roots,
            follows_tree=follows_tree,
        )

    # A public call comes from an outside client, which the combined server stops at its
    # call limit.
    deadline = _clock() + EXTERNAL_CALL_LIMIT_S - EXTERNAL_CALL_MARGIN_S

    async def send() -> VerbSummary | ToolRefusal:
        return await (_sent_again_once(run, deadline=deadline) if rebuild_lost_frames else run())

    answer = await send()
    rebuilt: dict[Pair, _Rebuilt] = {}
    lost: ToolRefusal | None = None
    frames = _frame_count(op_spec) if rebuild_lost_frames else 0
    while isinstance(answer, ToolRefusal) and len(rebuilt) < frames:
        fresh = {
            pair: ref
            for ref in answer.lost_frames
            if (pair := (ref.spec_id, ref.content_version)) not in rebuilt
        }
        if not fresh:
            break
        lost = answer
        for pair, ref in fresh.items():
            root = await _rebuilt(
                ref, lost=lost, default_enrichment=follows_tree, deadline=deadline
            )
            if isinstance(root, ToolRefusal):
                return to_tool_result(root)
            rebuilt[pair] = root
            replaced[pair] = root.node
        answer = await send()
    if isinstance(answer, ToolRefusal):
        if lost is not None and answer.lost_frames:
            return to_tool_result(answer.not_rebuilt())
        return to_tool_result(answer if refine is None else refine(answer))
    read = (
        await record_spec_reads(
            FrameRef(spec_id=answer.spec_id, content_version=answer.content_version), sent.node
        )
        if sent is not None and sent.spec_roots
        else {}
    )
    sentences = [
        sentence
        for sentence in (
            *(_newest_sentence(spec_id, read) for spec_id in newest),
            *(_rebuilt_sentence(root, read) for root in rebuilt.values()),
        )
        if sentence is not None
    ]
    digest, cells = digest_cells(answer, heading=heading)
    labels = await label_cells(cells, read_nodes=read_graph_nodes)
    if labels.labels:
        digest = render_summary_digest(answer, heading=heading, labels=labels)
    return ToolResult(
        content="\n\n".join([" ".join(sentences), digest]) if sentences else digest,
        structured_content=answer.verb_result().model_dump(mode="json"),
        is_error=False,
    )
