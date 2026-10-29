"""The seams the root workflow is built from: the `Executor` its nodes' own
calls run through, and the `PlannerModel` the `planner` node drafts with.
"""

from __future__ import annotations

from collections.abc import (
    Callable,
    Mapping,
    Sequence,
)
from typing import Any, Protocol

from compiler.surface import ProgressCallback
from google.adk import Context

from harness.decision_pick import DecisionPick


class Executor(Protocol):
    """Runs one plan step against the data layer (data-mcp / analysis-mcp).

    Returns `{"outcome": "ok", "frame_refs": [...], "store_refs": [...]?,
    "summary": str?}` (the optional summary is a one-line human result for
    the Activity feed) or
    `{"outcome": "refusal", "refusal": {code, retry_mode, fix, message,
    origin?, advice?, retry_after_s?}}` -- the refusal grammar's shape, minus
    transport. `harness.mcp_client.McpExecutor` retries a `retry_after_delay`
    refusal itself before returning it.
    `origin` is `"service"` when an MCP service refused and `"harness"` when
    the executor itself withheld or misrouted the call; an executor that
    states none is read as unattributed.

    `frame_refs` names each frame by the pair it is described and drawn by.
    A `get_data` outcome's `store_refs`, parallel to it, names each by the
    pair the materialisation store holds it under, which differs for a
    retrieval minted with a `groupby` or extra columns. An executor that
    states none has its frames read from the store by `frame_refs`
    (`harness.graph.state.frame_store_ref`).

    A concrete executor may also define `land_component(self, component:
    dict[str, Any]) -> None`, an optional, synchronous capability `make_charts`
    (`harness.orchestrator_tools`) probes with `getattr`. It is called right
    after the write it belongs to returns, with no `await` in between, to
    stamp (never decide or mutate) that write's own record for landing on the
    canvas. The landing itself happens at drain time (`harness.
    studio_translate`'s `_component_frames`). Its absence means no write this
    turn ever lands.
    """

    async def __call__(
        self, *, tool: str, args: dict[str, Any], on_progress: ProgressCallback | None = None
    ) -> dict[str, Any]:
        """Run `tool`; `on_progress`, when given, hears the call's progress reports."""
        ...


class PlannerModel(Protocol):
    """Drafts (or redrafts) a compiled plan from the conversation's search readings.

    Returns `{"plan": <planner.plan.Plan as JSON>, "summary": <approval summary>}`
    on a draft, or `{"refused": {"text": <plain sentence>, "final": <bool>,
    "code": <refusal code>}}` when the seam ends the entry without one
    (`harness.planner_seam.PlannerSeam._refused`); `final` says no other plan
    for the same ask can succeed, and `code` names the refusal for a caller
    that tells refusals apart. The node validates a drafted
    plan itself, so a malformed one fails here rather than at the approve
    gate; a refusal never writes `ctx.state["plan"]`, so whatever plan was
    already there is exactly what this call leaves behind. Where that
    record is a plan still `PENDING_APPROVAL` -- an amendment that hit the
    cap trying to redraft it -- the node routes back to `approve`, offering
    that same, unmodified plan again rather than ending the turn on a
    version the user could no longer act on; only a refusal with nothing
    `PENDING_APPROVAL` on record (the cap hit on a first draft, or on a
    replan against an already-approved plan) goes to `plan_done`, which
    hands the refusal back to the orchestrator.
    `prior_plan` is the record the graph holds, which the seam seeds its
    own session from before drafting. `refusal` / `amendment` are set on
    the replan and amend re-entries. `pick` is set on the amend re-entry a
    switch of a planner decision on the plan card opens: the option the user
    picked, which the drafted plan has to state. A refused pick entry's
    `refused` also carries `reason`, one sentence for the user, as `revise`'s
    does.

    `on_progress`, when given, may be called any number of times before this
    returns, each call naming one round trip the underlying model pair made
    (a shape this protocol treats as opaque -- the node only forwards it as
    an activity update). A seam with nothing to report simply never calls
    it.

    `ctx` is the planner node's own `Context`, passed through so a seam
    that hosts its models as a dynamically-scheduled child of this node
    (`ctx.run_node`) can do so in the graph's own session, rather than
    building a second `Runner`/session service of its own
    (`harness.planner_seam.PlannerSeam`). A seam with nothing to run
    through it simply never touches it.

    `readings` are the search readings of the conversation's earlier turns
    (`ORCHESTRATOR_PRIOR_LEDGER_KEY`) and then this turn's
    (`ORCHESTRATOR_SEARCHES_KEY`), oldest first, which a seam shows its
    planner.
    """

    async def __call__(
        self,
        *,
        ctx: Context,
        question: str,
        readings: Sequence[Mapping[str, Any]],
        prior_plan: dict[str, Any] | None = None,
        refusal: dict[str, Any] | None = None,
        amendment: str | None = None,
        canvas_digest: dict[str, Any] | None = None,
        conversation_digest: dict[str, Any] | None = None,
        on_progress: Callable[[dict[str, Any]], None] | None = None,
        pick: DecisionPick | None = None,
    ) -> dict[str, Any]: ...

    async def revise(
        self,
        *,
        ctx: Context,
        prior_plan: dict[str, Any],
        evidence_requirements: Sequence[Mapping[str, Any]],
        inherits: Mapping[str, str],
    ) -> dict[str, Any]:
        """Amend `prior_plan` with `evidence_requirements` and compile it again, calling no model.

        The graph's `switch` node revises the plan this way once a switched
        scope choice has its own spec_id: each requirement re-pinned to it is
        restated, and every decision the pins state follows its pin.
        `inherits` maps each new spec_id to the one it replaces. Answers as
        `__call__` does: the revised plan, `PENDING_APPROVAL`, with its
        summary, or a refusal. A refusal may leave the seam's own drafting
        state changed, which the caller restores.
        """
        ...
