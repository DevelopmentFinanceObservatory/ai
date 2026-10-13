"""A derived frame's step list: the step records along its ancestry, joined
into one list whose inputs name earlier steps.

Each derived frame's stored trace carries the step record of its own operation
(`analysis_mcp.trace.Trace.steps`), whose inputs name the frames it read by
their pairs. `assemble_steps` walks the store from a derived frame's pair as
`common.recipe.build_recipe` does: a derived input is followed to its own
trace, and a root input becomes a `retrieve` step whose input is the dfo-apis
pair of its mint record (`common.materialisation.MintRecord.dataset`) and whose
`options` are the mint's. `common` does not import `analysis_mcp`, so a trace's
steps are read by attribute and handled as their JSON form.

The list holds each frame's step once, after the steps of every frame it read,
in the order a depth-first walk of the inputs first finishes them. Steps are
numbered `s1`, `s2`, ... in that order, and every input names the step of the
frame in that role, so a frame read twice is one step named twice. A step's
other members, such as the `stat_var` a compare, sum, quotient or share
writes its rows under, or the `sides` a compare holds, are carried as its
trace stored them. The list
follows dfo-spec-kit's `steps.schema.json` and its structure rules.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

from common.materialisation import FrameRef, MaterialisationStore, StoredFrameTrace
from common.recipe import AncestryReads, RecipeError, RecoverMintRecord

__all__ = ["StepRecordInvalidError", "assemble_steps"]

_ROOT_SPEC_ID_PREFIX = "spec_"
_STEP_ID_PREFIX = "s"


@runtime_checkable
class _CarriesSteps(Protocol):
    @property
    def steps(self) -> Sequence[Any]: ...


class StepRecordInvalidError(RecipeError):
    """The step record stored on the trace of the derived frame `ref` is
    unusable: `reason` says how. A stored trace that fails a store check raises
    `common.recipe.RecipeTraceUnusableError` instead."""

    def __init__(self, ref: FrameRef, *, reason: str) -> None:
        self.reason = reason
        super().__init__(
            ref,
            f"the step stored for the derived frame {ref.spec_id}, {ref.content_version} "
            f"is unusable: {reason}",
        )


class _StepsMissingError(Exception):
    """A derived frame on the walk stores no step of its own."""


class _Walk:
    """One walk of the store from a derived frame to its roots."""

    def __init__(self, *, reads: AncestryReads) -> None:
        self._reads = reads
        self._ids: dict[tuple[str, str], str] = {}
        self.steps: list[dict[str, Any]] = []

    async def step_id(self, ref: FrameRef, *, trace: StoredFrameTrace | None = None) -> str:
        """The id of `ref`'s step, emitted once after the steps it reads."""
        key = (ref.spec_id, ref.content_version)
        if key not in self._ids:
            step = (
                await self._retrieve(ref)
                if ref.spec_id.startswith(_ROOT_SPEC_ID_PREFIX)
                else await self._derived(ref, trace=trace)
            )
            self._ids[key] = step["id"] = f"{_STEP_ID_PREFIX}{len(self.steps) + 1}"
            self.steps.append(step)
        return self._ids[key]

    async def _retrieve(self, ref: FrameRef) -> dict[str, Any]:
        record = await self._reads.mint_record(ref)
        return {
            "id": "",
            "op": "retrieve",
            "input": {
                "id": record.dataset.spec_id,
                "content_version": record.dataset.content_version,
            },
            "options": dict(record.options),
        }

    async def _derived(self, ref: FrameRef, *, trace: StoredFrameTrace | None) -> dict[str, Any]:
        if trace is None:
            trace = await self._reads.trace(ref)
        step = _own_step(ref, trace)
        if "input" in step:
            step["input"] = await self.step_id(_input_ref(ref, step["input"]))
        else:
            step["inputs"] = {
                role: await self.step_id(_input_ref(ref, named))
                for role, named in step.get("inputs", {}).items()
            }
        return step


def _own_step(ref: FrameRef, trace: StoredFrameTrace) -> dict[str, Any]:
    """The JSON form of the one step `trace` stores, or `_StepsMissingError`."""
    if not isinstance(trace, _CarriesSteps) or len(trace.steps) != 1:
        raise _StepsMissingError(ref)
    (step,) = trace.steps
    return step.model_dump(mode="json")


def _input_ref(ref: FrameRef, named: Any) -> FrameRef:
    """The frame a stored step's input names, by its pair."""
    try:
        return FrameRef(spec_id=named["id"], content_version=named["content_version"])
    except (KeyError, TypeError, ValueError) as exc:
        raise StepRecordInvalidError(ref, reason="it names an input by no frame pair") from exc


async def assemble_steps(
    ref: FrameRef,
    *,
    store: MaterialisationStore,
    trace: StoredFrameTrace | None = None,
    recover: RecoverMintRecord | None = None,
    reads: AncestryReads | None = None,
) -> list[dict[str, Any]] | None:
    """The step list of the derived frame held under `ref`. See the module
    docstring.

    Args:
        ref: A derived frame's `ops_`/`ocv_` pair.
        store: The store holding the frame's ancestors' traces and its roots'
            mint records. No frame's rows are decoded.
        trace: `ref`'s own stored trace, when the caller has already read it.
        recover: Rebuilds the mint record of a root that has none. Without it
            such a root fails the walk.
        reads: The reads of `store` to share with another walk of the same
            ancestry, such as `common.recipe.build_recipe`. Its own `recover`
            stands in for `recover`. Without it the walk reads afresh.

    Returns:
        The list, or `None` when a derived frame on the walk stores no step,
        as a trace stored before steps were does.

    Raises:
        RecipeAncestorMissingError: The store holds no trace for `ref` or for
            a derived ancestor.
        RecipeTraceUnusableError: A stored trace on the walk fails a store
            check.
        StepRecordInvalidError: A stored step names an input by no frame
            pair.
        RecipeRootUnrecoverableError: A root has no mint record and none was
            rebuilt.
        ValueError: `ref` is a root frame's pair, which has no steps.
    """
    if ref.spec_id.startswith(_ROOT_SPEC_ID_PREFIX):
        raise ValueError(f"{ref.spec_id} names a root frame, which has no steps")
    walk = _Walk(reads=reads or AncestryReads(store=store, recover=recover))
    try:
        await walk.step_id(ref, trace=trace)
    except _StepsMissingError:
        return None
    return walk.steps
