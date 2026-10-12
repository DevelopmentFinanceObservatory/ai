"""Replays a stored recipe on current data and publishes the frame it gives.

A recipe (`common.recipe`) is the tree of op-specs that computed a derived frame,
down to the mint records of its root datasets. `replay_recipe` computes that tree
again on the data dfo-apis holds now:

1. Every root is minted again with its recorded `options` and `scope`
   (`data_mcp.get_data_tool.mint_recorded_root`), which registers the current
   frame of that rendering in the materialisation store. The mint may send other
   options than the recorded ones: a flow root recorded with its parties' types
   or with no columns is minted with their kinds. Every root is minted,
   moved or not, because a frame older than the store's retention has left it.
   A mint dfo-apis queues does not wait: the replay answers `Pending`, and the
   next call sends the same mint, which dfo-apis answers from the operation it
   already holds.
2. A root recorded with one StatVar that now holds more, or that comes back as a
   groupby sum with a version of its own when it was recorded under a raw
   version, refuses `root_rendering_ambiguous`: while one StatVar contributed,
   the summed and the unsummed frame shared one pair, and the record cannot say
   which of the two the frame was (`common.materialisation.mint_record`). A
   groupby root whose frame is held under the pair that names its groupby
   (`data_mcp.frame_materialisation.frame_ref_for`) was the sum, and replays.
3. A tree with a rolling op-spec has its latest period resolved once, over the
   new frames of the roots a path through a rolling op-spec reads
   (`common.rolling.rolling_roots`), by analysis-mcp's `resolve_anchor`. Each rolling literal is
   rewritten to that period at its offset, and each rolling call carries the
   returned `anchor_key`. A tree of fixed periods makes no such call.
4. The tree is walked from its roots up. Each node's verb is called with its
   recorded arguments and its inputs' new pairs. Before a call, the pair it would
   mint is computed with analysis-mcp's own identity functions, and the call is
   skipped when the store already holds that pair, so a replay cut short is
   resumed by the next one. Inputs of one node run at once, at most
   `MAX_CONCURRENT_STEPS` calls at a time. Roots are minted one at a time.
5. The final frame's recipe is built from the store (`common.recipe.build_recipe`).
   A root whose mint record holds the options this replay minted it with is
   stated with the options the stored recipe recorded for it, so the replayed
   recipe keeps the stored `normal_form_hash`. The store's mint record keeps the
   options sent, since a `get_data` call minting the same pair shares it. The
   frame is published with the recipe through the caller's `publish`
   (`presentation_mcp.client.PlatformAssembler.publish_frame`), unless dfo-apis
   already holds that pair under the recipe's `replay_key`. dfo-apis keeps a
   pair's first recipe, so a pair another card published first under a recipe of
   another normal form refuses `recipe_normal_form_changed`: no render draws this
   card on it.

The module calls no model and holds no state between calls. It imports nothing
from `planner`, `compiler`, `executor`, `translator`, `orchestrator` or the
conversation and session modules of `harness`, so it runs in a request with no
conversation. Its dependencies arrive as arguments: the verbs through a callable
over the tool surface the executor uses, the store, the dfo-apis client, the
publisher, and a row budget.

Three things reach the process-wide holders instead of the arguments, because
data-mcp and presentation-mcp read them there: the mint posts through
`data_mcp.client_holder`, and the mint and the publish read and write
`common.materialisation.get_store()`. `store` must be that store.

Every expected outcome is a return value: `Replayed`, `Pending`, or `Refused`
with a machine-readable code. A verb's own refusal on current data is returned
under the verb's code, and nothing is published for a replay that did not finish.
"""

from __future__ import annotations

import asyncio
import copy
import logging
from collections.abc import Awaitable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol

from analysis_mcp.execution_identity import current_execution_identity
from analysis_mcp.identity import (
    OpSpecIdentityError,
    mint_toolkit_content_version,
    op_spec_hash_for,
)
from analysis_mcp.op_spec import OpSpecAdapter
from common.live_apis import FramePair, LiveApisClient, LiveApisError
from common.materialisation import (
    FrameExpired,
    FrameIntegrityError,
    FrameRef,
    FrameTooLarge,
    MaterialisationStore,
    MintRecord,
)
from common.recipe import RECIPE_SCHEMA_VERSION, RecipeError, build_recipe
from common.rolling import canonical_populations, rolling_roots
from common.row_budget import ROW_BUDGET_RETRY_AFTER_S, RowBudget, RowBudgetExhausted
from data_mcp.deadline import EXTERNAL_CALL_BUDGET, CallBudget, CallDeadline, use_call_budget
from data_mcp.frame_materialisation import frame_ref_for
from data_mcp.get_data_tool import RootMint, mint_recorded_root
from data_mcp.operations import QueuedOperation
from dfo_frame_grammar import FrameError, RollingError, shift_period
from fastmcp.tools import ToolResult
from presentation_mcp.platform_contract import NormalFormConflict, Refusal, RetryMode
from pydantic import ValidationError

__all__ = [
    "MAX_CONCURRENT_STEPS",
    "CallTool",
    "MintRoot",
    "Pending",
    "PublishFrame",
    "Refused",
    "Replayed",
    "ToolAnswer",
    "replay_recipe",
]

logger = logging.getLogger(__name__)

# Verb calls on independent inputs in flight at once.
MAX_CONCURRENT_STEPS = 2

# The wait a `Pending` names when the replay stopped at its deadline. The steps
# that finished are in the store, so the next call can start at once.
RESUME_RETRY_AFTER_S = 1.0

# The wait a `Pending` names for a refusal that says to retry and names no wait.
BUSY_RETRY_AFTER_S = 2.0

RESOLVE_ANCHOR_TOOL = "resolve_anchor"

_PRIMARY_SLOT = "frame_ref"
_ROLLING = "rolling"
_RETRY_AFTER_DELAY = "retry_after_delay"

Stage = Literal["recipe", "mint", "anchor", "verb", "publish"]


class ToolAnswer(Protocol):
    """What a tool call answered. `compiler.surface.ToolCallResult` is one. A
    refusal has `is_error` set and its envelope (`code`, `retry_mode`, `message`,
    `retry_after_s`) in `structured_content`."""

    @property
    def is_error(self) -> bool: ...

    @property
    def structured_content(self) -> Mapping[str, Any] | None: ...


class CallTool(Protocol):
    """Calls the analysis-mcp tool `name` with `arguments` and returns its answer,
    refusal included. Over the executor's surface this is `surface.call_tool(
    tool=ToolId(server="analysis-mcp", name=name), arguments=arguments)`."""

    def __call__(self, *, name: str, arguments: Mapping[str, object]) -> Awaitable[ToolAnswer]: ...


class MintRoot(Protocol):
    """Mints a recorded root again. `data_mcp.get_data_tool.mint_recorded_root` is one."""

    def __call__(
        self, record: MintRecord, *, deadline: CallDeadline
    ) -> Awaitable[RootMint | QueuedOperation | ToolResult]: ...


class PublishFrame(Protocol):
    """Publishes a derived frame the store holds to dfo-apis with `recipe`, and
    says whether dfo-apis newly recorded it, or that it holds the frame under a
    recipe of another normal form.
    `presentation_mcp.client.PlatformAssembler.publish_frame` is one."""

    def __call__(
        self, *, spec_id: str, content_version: str, recipe: Mapping[str, Any]
    ) -> Awaitable[bool | Refusal | NormalFormConflict]: ...


@dataclass(frozen=True, kw_only=True)
class Replayed:
    """The recipe was computed on current data and dfo-apis holds the result.

    `frame` is the new derived pair and `recipe` its recipe. `newly_published` is
    whether this call made dfo-apis record the frame, and is false when it held it
    already. `normal_form_changed` is true only for a replay given replacement
    populations, whose recipe then has another `normal_form_hash` than the stored one.
    """

    frame: FrameRef
    recipe: dict[str, Any]
    newly_published: bool
    normal_form_changed: bool = False


@dataclass(frozen=True, kw_only=True)
class Pending:
    """The replay has not finished and the same call may be made again after
    `retry_after_s`. `reason` is `mint_queued` for a root dfo-apis is still
    building, `deadline` for a replay that ran out of time between steps,
    `row_budget` for rows the refresh budget has no room for while other replays
    hold theirs, and `busy` for a service that refused with a wait."""

    retry_after_s: float
    reason: Literal["mint_queued", "deadline", "row_budget", "busy"]


@dataclass(frozen=True, kw_only=True)
class Refused:
    """The recipe cannot be replayed on current data. `code` is machine-readable:
    one of this module's own, or the code of the mint, verb or publish refusal that
    stopped it. `stage` says where it stopped."""

    code: str
    detail: str
    stage: Stage


@dataclass(frozen=True, kw_only=True)
class _Frame:
    """A frame of the replayed tree the store holds: its new pair, the rows it
    holds, and its frame-grammar content version when already known."""

    ref: FrameRef
    rows: int
    frame_content_version: str | None


def _pair(ref: FrameRef) -> dict[str, str]:
    return {"spec_id": ref.spec_id, "content_version": ref.content_version}


def _is_rolling_ref(value: object) -> bool:
    return isinstance(value, Mapping) and "anchor" in value


def _derived_nodes(tree: Mapping[str, Any]) -> Iterator[Mapping[str, Any]]:
    if "root" in tree:
        return
    yield tree
    for node in tree["inputs"].values():
        yield from _derived_nodes(node)


class _Stopped(Exception):  # noqa: N818 -- a control-flow signal carrying an outcome, not an error
    """The replay ended without a frame. `outcome` is what `replay_recipe` returns."""

    def __init__(self, outcome: Pending | Refused) -> None:
        self.outcome = outcome
        super().__init__(type(outcome).__name__)


def _refusal_stop(structured: Mapping[str, Any] | None, *, stage: Stage) -> _Stopped:
    """What a data-mcp or analysis-mcp refusal envelope makes of the replay: a wait
    when it says to retry after a delay, else a refusal under its own code."""
    body = structured or {}
    if body.get("retry_mode") == _RETRY_AFTER_DELAY:
        wait = body.get("retry_after_s")
        return _Stopped(
            Pending(
                retry_after_s=float(wait) if isinstance(wait, int | float) else BUSY_RETRY_AFTER_S,
                reason="busy",
            )
        )
    return _Stopped(
        Refused(
            code=str(body.get("code") or "refusal_unreadable"),
            detail=str(body.get("message") or "the refusal carried no message"),
            stage=stage,
        )
    )


def _rows_stop(exc: RowBudgetExhausted, *, stage: Stage) -> _Stopped:
    """What a claim the refresh row budget refused makes of the replay: a wait
    when the rows fit an idle budget, and `rows_over_refresh_budget` when they
    exceed the whole budget, which no later call could claim."""
    if exc.requested > exc.max_rows:
        return _Stopped(
            Refused(
                code="rows_over_refresh_budget",
                detail=(
                    f"the step holds {exc.requested} frame rows at once, and the refresh "
                    f"row budget is {exc.max_rows}"
                ),
                stage=stage,
            )
        )
    return _Stopped(Pending(retry_after_s=ROW_BUDGET_RETRY_AFTER_S, reason="row_budget"))


def _apis_stop(exc: LiveApisError, *, stage: Stage) -> _Stopped:
    if exc.retryable:
        return _Stopped(Pending(retry_after_s=BUSY_RETRY_AFTER_S, reason="busy"))
    return _Stopped(Refused(code=exc.code or "dfo_apis_refused", detail=str(exc), stage=stage))


def _worst(stops: Sequence[Pending | Refused]) -> _Stopped:
    """The one of `stops` that ends the replay: a refusal before a wait, and of two
    waits the longer."""
    refused = next((stop for stop in stops if isinstance(stop, Refused)), None)
    waits = [stop for stop in stops if isinstance(stop, Pending)]
    return _Stopped(refused or max(waits, key=lambda wait: wait.retry_after_s))


async def _all_or_stop[T](steps: Sequence[Awaitable[T]]) -> list[T]:
    """Run `steps` together to the end of each, and return their results.

    Every step finishes before one that stopped ends the replay, so what the others
    computed is in the store for the next call.

    Raises:
        _Stopped: A step stopped (`_worst` of those that did).
        Exception: A step failed with anything else, which is raised first.
    """
    results = await asyncio.gather(*steps, return_exceptions=True)
    stops: list[Pending | Refused] = []
    done: list[T] = []
    for result in results:
        if isinstance(result, _Stopped):
            stops.append(result.outcome)
        elif isinstance(result, BaseException):
            raise result
        else:
            done.append(result)
    if stops:
        raise _worst(stops)
    return done


def _rewritten(
    parameters: Mapping[str, Any], *, period: str, populations: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """`parameters` of a rolling op-spec on a new latest `period`: each literal a
    rolling reference governs becomes `period` at the reference's offset, and the
    overlay carries `populations`.

    Raises:
        _Stopped: `rolling_period_invalid` when an offset moves `period` to no period.
    """
    rewritten = copy.deepcopy(dict(parameters))
    overlay = rewritten[_ROLLING]
    overlay["population"] = [dict(population) for population in populations]
    try:
        for name, reference in overlay.items():
            if _is_rolling_ref(reference):
                rewritten[name] = shift_period(period, int(reference["offset"]))
        for predicate in rewritten.get("predicates", ()):
            reference = predicate.get(_ROLLING)
            if _is_rolling_ref(reference):
                predicate["value"] = shift_period(period, int(reference["offset"]))
    except (RollingError, FrameError) as exc:
        raise _Stopped(
            Refused(code="rolling_period_invalid", detail=str(exc), stage="anchor")
        ) from exc
    return rewritten


_AMBIGUOUS_DETAIL = (
    "the root {spec_id} was recorded while one StatVar contributed to it, when its "
    "summed and unsummed frames were the same; they now differ, and the record does "
    "not say which the frame was"
)


class _Replay:
    """One replay: the arguments of `replay_recipe`, and the frames found so far.
    Every method that cannot go on raises `_Stopped`."""

    def __init__(
        self,
        *,
        tree: Mapping[str, Any],
        roots: Sequence[MintRecord],
        deadline: CallDeadline,
        call_tool: CallTool,
        store: MaterialisationStore,
        apis: LiveApisClient,
        row_budget: RowBudget,
    ) -> None:
        self.tree = tree
        self.roots = roots
        self.deadline = deadline
        self.call_tool = call_tool
        self.store = store
        self.apis = apis
        self.row_budget = row_budget
        self.root_frames: list[_Frame] = []
        # By each root's new pair: the options its mint sent, and the options the
        # stored recipe recorded for it.
        self._root_options: dict[tuple[str, str], tuple[dict[str, Any], dict[str, Any]]] = {}
        # Set for a rolling tree once its latest period is resolved.
        self.period: str | None = None
        self.anchor_key: str | None = None
        self.populations: Sequence[Mapping[str, Any]] = ()
        self._steps = asyncio.Semaphore(MAX_CONCURRENT_STEPS)
        # Each node's replay, by the pair the recipe recorded it under, so a frame
        # two slots reach is computed once.
        self._nodes: dict[tuple[str, str], asyncio.Task[_Frame]] = {}

    def check_deadline(self) -> None:
        if self.deadline.remaining() <= 0:
            raise _Stopped(Pending(retry_after_s=RESUME_RETRY_AFTER_S, reason="deadline"))

    # Roots

    async def mint_roots(self, mint_root: MintRoot, *, mint_budget: CallBudget) -> None:
        """Mint every root and keep its new frame in `root_frames`, in `roots` order.

        The roots are minted one after another: data-mcp admits one mint at a time
        under a budget that does not wait out dfo-apis' synchronous mint
        (`data_mcp.mint_ledger`), and would refuse a second sent beside the first.
        A root that stops the replay does not stop the mints after it, so every
        queued root is submitted by the first call.
        """
        frames: list[_Frame] = []
        stops: list[Pending | Refused] = []
        with use_call_budget(mint_budget):
            for record in self.roots:
                try:
                    frames.append(await self._mint(record, mint_root))
                except _Stopped as stopped:
                    stops.append(stopped.outcome)
        if stops:
            raise _worst(stops)
        self.root_frames = frames

    async def _mint(self, record: MintRecord, mint_root: MintRoot) -> _Frame:
        self.check_deadline()
        minted = await mint_root(record, deadline=self.deadline)
        if isinstance(minted, QueuedOperation):
            raise _Stopped(Pending(retry_after_s=minted.retry_after_s, reason="mint_queued"))
        if isinstance(minted, ToolResult):
            raise _refusal_stop(minted.structured_content, stage="mint")
        if minted.dataset.spec_id != record.dataset.spec_id:
            raise _Stopped(
                Refused(
                    code="root_dataset_changed",
                    detail=(
                        f"minting {record.dataset.spec_id} with its recorded scope now "
                        f"names the dataset {minted.dataset.spec_id}"
                    ),
                    stage="mint",
                )
            )
        if await self._ambiguous(record, minted):
            raise _Stopped(
                Refused(
                    code="root_rendering_ambiguous",
                    detail=_AMBIGUOUS_DETAIL.format(spec_id=record.dataset.spec_id),
                    stage="mint",
                )
            )
        rows = await self._held_rows(minted.frame)
        if rows is None:
            # The mint confirmed the frame a moment ago, so the store has evicted
            # it since, and the next call mints it again.
            raise _Stopped(Pending(retry_after_s=RESUME_RETRY_AFTER_S, reason="busy"))
        self._root_options.setdefault(
            (minted.frame.spec_id, minted.frame.content_version), (minted.options, record.options)
        )
        return _Frame(
            ref=minted.frame, rows=rows, frame_content_version=minted.frame.content_version
        )

    async def _ambiguous(self, record: MintRecord, minted: RootMint) -> bool:
        """Whether `record` cannot say which rendering its frame was, now that the
        two differ. See the module docstring, step 2.

        A record with no `stat_var_count` cannot be tested for the first case and is
        replayed as recorded: refusing it would refuse every older root that has
        always held several StatVars. Such a record either holds the `options` the
        mint sent, written before the count was kept, or was rebuilt by
        `data_mcp.mint_recovery` for a `groupby` sum with a version of its own,
        whose `groupby` dfo-apis recorded. Recovery counts or refuses every root
        under an unsummed version, where the first case arises. The second case
        reads dfo-apis' versions and needs no count, and dfo-apis' own frame refresh
        refuses a version that links both renderings.

        A record whose frame is held under the pair that names its `groupby`
        (`frame_ref_for`) is unambiguous: no unsummed frame was ever held there.
        """
        if minted.dataset == record.dataset or _names_its_groupby(record):
            return False
        if record.stat_var_count == 1 and minted.stat_var_count > 1:
            return True
        if "groupby" not in record.options:
            return False
        try:
            versions = await self.apis.list_versions(spec_id=minted.dataset.spec_id)
        except LiveApisError as exc:
            raise _apis_stop(exc, stage="mint") from exc
        collapses = {
            version.content_version
            for version in versions
            if version.base_content_version is not None
        }
        return (
            minted.dataset.content_version in collapses
            and record.dataset.content_version not in collapses
        )

    async def _held_rows(self, ref: FrameRef) -> int | None:
        """The rows of the frame the store holds under `ref`, 0 for one stored
        without a count, and `None` when it holds none."""
        try:
            held = await self.store.open(ref)
        except FrameExpired:
            return None
        except (FrameIntegrityError, FrameTooLarge) as exc:
            raise _Stopped(
                Refused(code="stored_frame_unusable", detail=str(exc), stage="verb")
            ) from exc
        return None if held is None else held.num_rows or 0

    def stated_options(self, record: MintRecord) -> Mapping[str, Any]:
        """The options the replayed recipe states for the root `record` describes:
        the stored recipe's when `record` holds the options this replay minted the
        root with, else `record`'s own. Two stored roots that now mint one pair
        share the first one's, and the normal form then differs."""
        sent, stored = self._root_options.get(
            (record.frame.spec_id, record.frame.content_version), (None, None)
        )
        return stored if stored is not None and record.options == sent else record.options

    # The latest period

    def _populations(
        self, replacement: Sequence[Mapping[str, Any]] | None
    ) -> list[dict[str, Any]] | None:
        """The populations the latest period is taken over: `replacement` when
        given, else the one list every rolling op-spec of the tree carries. `None`
        for a tree with no rolling op-spec."""
        recorded: list[Any] = []
        for node in _derived_nodes(self.tree):
            overlay = node["op_spec"]["parameters"].get(_ROLLING)
            if isinstance(overlay, Mapping) and overlay.get("population") not in recorded:
                recorded.append(overlay.get("population"))
        if not recorded and replacement is None:
            return None
        if not recorded:
            raise _Stopped(
                Refused(
                    code="populations_not_applicable",
                    detail="replacement populations were given for a recipe with no rolling period",
                    stage="anchor",
                )
            )
        if replacement is not None:
            return canonical_populations(replacement)
        if len(recorded) > 1:
            raise _Stopped(
                Refused(
                    code="rolling_populations_differ",
                    detail="the recipe's rolling op-specs do not carry one population list",
                    stage="anchor",
                )
            )
        return canonical_populations(recorded[0])

    async def resolve_rolling(self, replacement: Sequence[Mapping[str, Any]] | None) -> None:
        """Resolve the latest period of a rolling tree over the new roots. A tree
        with no rolling op-spec calls nothing."""
        populations = self._populations(replacement)
        if populations is None:
            return
        self.check_deadline()
        # The period is taken over only these roots, since a dataset joined beside
        # a rolling step does not move it (`common.rolling.rolling_roots`).
        reached = [self.root_frames[node["root"]] for node in rolling_roots(self.tree)]
        roots = {(frame.ref.spec_id, frame.ref.content_version): frame for frame in reached}
        answer = await self._claimed(
            sum(frame.rows for frame in roots.values()),
            stage="anchor",
            name=RESOLVE_ANCHOR_TOOL,
            arguments={
                "populations": populations,
                "roots": [_pair(frame.ref) for frame in roots.values()],
            },
        )
        resolved = answer.structured_content or {}
        if answer.is_error:
            raise _refusal_stop(resolved, stage="anchor")
        self.populations = populations
        self.period = resolved["period"]
        self.anchor_key = resolved["anchor_key"]

    async def _claimed(
        self, rows: int, *, stage: Stage, name: str, arguments: Mapping[str, object]
    ) -> ToolAnswer:
        """Call `name` while `rows` are held from the refresh row budget."""
        try:
            with self.row_budget.claim(rows):
                return await self.call_tool(name=name, arguments=arguments)
        except RowBudgetExhausted as exc:
            raise _rows_stop(exc, stage=stage) from exc

    # The tree

    async def frame_of(self, node: Mapping[str, Any]) -> _Frame:
        """The new frame of `node`, replayed once however many slots reach it."""
        if "root" in node:
            return self.root_frames[node["root"]]
        key = (node["frame"]["spec_id"], node["frame"]["content_version"])
        if key not in self._nodes:
            self._nodes[key] = asyncio.create_task(self._derived(node))
        return await self._nodes[key]

    async def _derived(self, node: Mapping[str, Any]) -> _Frame:
        slots = list(node["inputs"])
        frames = dict(
            zip(
                slots,
                await _all_or_stop([self.frame_of(node["inputs"][slot]) for slot in slots]),
                strict=True,
            )
        )
        op_spec = node["op_spec"]
        parameters: Mapping[str, Any] = op_spec["parameters"]
        rolling = isinstance(parameters.get(_ROLLING), Mapping)
        if rolling:
            assert self.period is not None  # resolved before the walk for a rolling tree
            parameters = _rewritten(parameters, period=self.period, populations=self.populations)
        arguments: dict[str, Any] = {
            **parameters,
            **{slot: _pair(frame.ref) for slot, frame in frames.items()},
        }

        async with self._steps:
            self.check_deadline()
            expected = await self._expected_pair(op_spec["verb"], arguments, frames)
            if expected is not None and (rows := await self._held_rows(expected)) is not None:
                return _Frame(ref=expected, rows=rows, frame_content_version=None)
            if rolling:
                arguments["anchor_key"] = self.anchor_key
            answer = await self._claimed(
                sum(frame.rows for frame in frames.values()),
                stage="verb",
                name=op_spec["verb"],
                arguments=arguments,
            )
        result = answer.structured_content or {}
        if answer.is_error:
            raise _refusal_stop(result, stage="verb")
        ref = FrameRef(spec_id=result["spec_id"], content_version=result["content_version"])
        if expected is not None and ref != expected:
            logger.warning(
                "replaying %s expected %s/%s and the verb minted %s/%s",
                op_spec["verb"],
                expected.spec_id,
                expected.content_version,
                ref.spec_id,
                ref.content_version,
            )
        return _Frame(
            ref=ref, rows=result["rows"], frame_content_version=result["frame_content_version"]
        )

    async def _expected_pair(
        self, verb: str, arguments: Mapping[str, Any], inputs: Mapping[str, _Frame]
    ) -> FrameRef | None:
        """The pair calling `verb` with `arguments` would mint: the `ops_` id of the
        op-spec the arguments state, and the `ocv_` of that id, the inputs' content
        versions and this build's execution identity. `None` when the arguments
        state no valid op-spec, or an input's trace has left the store; the verb is
        then called, and answers for itself."""
        versions: set[str] = set()
        for frame in inputs.values():
            version = frame.frame_content_version
            if version is None:
                try:
                    trace = await self.store.get_trace(frame.ref)
                except (FrameExpired, FrameIntegrityError, FrameTooLarge):
                    return None
                if trace is None:
                    return None
                version = trace.frame_content_version
            versions.add(version)
        try:
            spec_id = op_spec_hash_for(OpSpecAdapter.validate_python({"verb": verb, **arguments}))
            content_version = mint_toolkit_content_version(
                op_spec_hash=spec_id,
                input_content_versions=versions,
                execution_identity=current_execution_identity(),
            )
        except (ValidationError, OpSpecIdentityError):
            return None
        return FrameRef(spec_id=spec_id, content_version=content_version)

    # Publishing

    async def publish(self, frame: _Frame, recipe: dict[str, Any], publish: PublishFrame) -> bool:
        """Whether this call made dfo-apis record `frame`, publishing it with
        `recipe` unless dfo-apis already holds the pair under the recipe's replay key."""
        self.check_deadline()
        try:
            held = await self.apis.find_replay(replay_key=recipe["replay_key"])
        except LiveApisError as exc:
            raise _apis_stop(exc, stage="publish") from exc
        if held == FramePair(frame.ref.spec_id, frame.ref.content_version):
            return False
        try:
            with self.row_budget.claim(frame.rows):
                published = await publish(
                    spec_id=frame.ref.spec_id,
                    content_version=frame.ref.content_version,
                    recipe=recipe,
                )
        except RowBudgetExhausted as exc:
            raise _rows_stop(exc, stage="publish") from exc
        if isinstance(published, NormalFormConflict):
            raise await self._held_under_other_normal_form(frame, recipe)
        if not isinstance(published, Refusal):
            return published
        if published.retry is RetryMode.RETRY_AFTER_DELAY:
            raise _Stopped(
                Pending(retry_after_s=published.retry_after_s or BUSY_RETRY_AFTER_S, reason="busy")
            )
        raise _Stopped(Refused(code=published.error, detail=published.detail, stage="publish"))

    async def _held_under_other_normal_form(
        self, frame: _Frame, recipe: Mapping[str, Any]
    ) -> _Stopped:
        """`recipe_normal_form_changed` for `frame`, which dfo-apis holds under a
        recipe of another normal form than `recipe`'s: no card of `recipe` is drawn
        on it. The detail names the normal form dfo-apis holds."""
        try:
            held = await self.apis.read_recipe(
                spec_id=frame.ref.spec_id, content_version=frame.ref.content_version
            )
        except LiveApisError as exc:
            return _apis_stop(exc, stage="publish")
        held_hash = (held or {}).get("normal_form_hash")
        return _Stopped(
            Refused(
                code="recipe_normal_form_changed",
                detail=(
                    f"dfo-apis holds {frame.ref.spec_id}/{frame.ref.content_version} under a "
                    f"recipe with the normal form {held_hash}, and the replayed recipe has "
                    f"{recipe['normal_form_hash']}"
                ),
                stage="publish",
            )
        )


def _parsed(recipe: Mapping[str, Any]) -> tuple[Mapping[str, Any], list[MintRecord]]:
    """The recipe's tree and its roots' mint records.

    Raises:
        _Stopped: `recipe_invalid` for a recipe this build cannot read.
    """
    tree, roots = recipe.get("tree"), recipe.get("roots")
    if (
        recipe.get("schema_version") != RECIPE_SCHEMA_VERSION
        or not isinstance(tree, Mapping)
        or not isinstance(roots, list)
        or "op_spec" not in tree
    ):
        raise _Stopped(
            Refused(
                code="recipe_invalid",
                detail="the recipe is not a schema version 1 recipe of a derived frame",
                stage="recipe",
            )
        )
    try:
        return tree, [MintRecord.model_validate(root) for root in roots]
    except ValidationError as exc:
        raise _Stopped(
            Refused(
                code="recipe_invalid",
                detail=f"a root of the recipe is no mint record: {exc.error_count()} error(s)",
                stage="recipe",
            )
        ) from exc


async def _replayed(
    recipe: Mapping[str, Any],
    *,
    replay: _Replay,
    publish: PublishFrame,
    populations: Sequence[Mapping[str, Any]] | None,
    mint_root: MintRoot,
    mint_budget: CallBudget,
) -> Replayed:
    """`replay_recipe` for a parsed recipe, raising `_Stopped` where it returns
    `Pending` or `Refused`."""
    await replay.mint_roots(mint_root, mint_budget=mint_budget)
    await replay.resolve_rolling(populations)
    frame = await replay.frame_of(replay.tree)

    try:
        replayed = await build_recipe(
            frame.ref, store=replay.store, root_options=replay.stated_options
        )
    except RecipeError as exc:
        raise _Stopped(
            Refused(code="recipe_unbuildable", detail=str(exc), stage="publish")
        ) from exc
    changed = replayed["normal_form_hash"] != recipe.get("normal_form_hash")
    if changed and populations is None:
        # dfo-apis renders a card on a frame only when the two recipes share a
        # normal form, so this frame could be drawn by no card of the recipe.
        raise _Stopped(
            Refused(
                code="recipe_normal_form_changed",
                detail=(
                    f"the replayed recipe has the normal form {replayed['normal_form_hash']}, "
                    f"and the stored recipe {recipe.get('normal_form_hash')}"
                ),
                stage="publish",
            )
        )
    return Replayed(
        frame=frame.ref,
        recipe=replayed,
        newly_published=await replay.publish(frame, replayed, publish),
        normal_form_changed=changed,
    )


async def replay_recipe(
    recipe: Mapping[str, Any],
    *,
    deadline: CallDeadline,
    call_tool: CallTool,
    store: MaterialisationStore,
    apis: LiveApisClient,
    publish: PublishFrame,
    row_budget: RowBudget,
    populations: Sequence[Mapping[str, Any]] | None = None,
    mint_root: MintRoot = mint_recorded_root,
    mint_budget: CallBudget = EXTERNAL_CALL_BUDGET,
) -> Replayed | Pending | Refused:
    """Compute `recipe` again on current data and publish the frame with its recipe.
    See the module docstring for the steps.

    Args:
        recipe: A recipe as dfo-apis returns it from
            `GET /api/datasets/frames/{spec_id}/{content_version}/recipe`.
        deadline: When the replay must stop. It is read before each mint, each
            verb call, the anchor call and the publish: a step in flight runs to
            its own end, and a replay past the deadline returns `Pending` with its
            finished steps in the store.
        call_tool: Calls an analysis-mcp tool by name: the eight verbs and
            `resolve_anchor`.
        store: The process's materialisation store, `common.materialisation.get_store()`.
        apis: The dfo-apis client. `list_versions`, `find_replay` and
            `read_recipe` are called, all under the datasets principal.
        publish: Publishes the final frame with its recipe.
        row_budget: The rows replays may hold at once. Each verb call claims its
            inputs' rows, `resolve_anchor` the roots', and the publish the final
            frame's. A claim the rows in flight leave no room for returns
            `Pending`, and one larger than the whole budget is refused
            `rows_over_refresh_budget`. The tools claim from the process-wide
            budget as well.
        populations: Replaces the populations every rolling op-spec carries, for a
            card whose selection counts other series. The replayed recipe then has
            another `normal_form_hash`. Refused for a recipe with no rolling period.
        mint_root: Mints one recorded root again.
        mint_budget: Sizes each mint's steps inside `deadline`
            (`data_mcp.deadline.CallBudget`); its own `deadline_s` is not read.

    Returns:
        `Replayed`, `Pending`, or `Refused`. Own refusal codes: `recipe_invalid`,
        `root_dataset_changed`, `root_rendering_ambiguous`, `stored_frame_unusable`,
        `populations_not_applicable`, `rolling_populations_differ`,
        `rolling_period_invalid`, `rows_over_refresh_budget`, `recipe_unbuildable`,
        and `recipe_normal_form_changed`. Any other code is the mint's, the verb's, the
        publish's or dfo-apis' own.

    Raises:
        LiveApisNotConfigured: `apis` has no datasets principal.
        Exception: Whatever `call_tool` raises when a call does not complete.
    """
    try:
        tree, roots = _parsed(recipe)
        replay = _Replay(
            tree=tree,
            roots=roots,
            deadline=deadline,
            call_tool=call_tool,
            store=store,
            apis=apis,
            row_budget=row_budget,
        )
        return await _replayed(
            recipe,
            replay=replay,
            publish=publish,
            populations=populations,
            mint_root=mint_root,
            mint_budget=mint_budget,
        )
    except _Stopped as stopped:
        return stopped.outcome


def _names_its_groupby(record: MintRecord) -> bool:
    """Whether `record` is a groupby root whose frame is held under the pair
    `frame_ref_for` derives from its dataset, columns and groupby."""
    groupby = record.options.get("groupby")
    return groupby is not None and record.frame == frame_ref_for(
        spec_id=record.dataset.spec_id,
        content_version=record.dataset.content_version,
        columns=tuple(record.options.get("columns") or ()),
        groupby=groupby,
    )
