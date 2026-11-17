"""The `seed` node: opens the turn with the runs an earlier turn left
unfinished, carried on, and the datasets the user brought from Search or
the chip the user clicked."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import Any

from compiler.surface import ToolSurface
from orchestrator.call import OrchestratorCall
from pydantic import ValidationError
from translator.discovery import (
    DefinitionNotResolvedError,
)

from harness.graph.events import GraphEvent
from harness.graph.interfaces import (
    Executor,
)
from harness.graph.orchestrate import AgentHandBack
from harness.graph.resume import ResumeRuns
from harness.graph.state import (
    APPLY_CHOICE_STATE_KEY,
    TurnInput,
    store_search,
    turn_budgets,
)
from harness.graph.switch import apply_stored_choice
from harness.orchestrator_tools import raised_answer
from harness.seeds import (
    SEED_DROPPED,
    DroppedSeed,
    StaleSeedError,
    answered_block,
    refuse_seed,
    refuse_seed_reference,
    resolve_seed,
    seeds_hand_back,
    seeds_received,
    take_seeds,
)
from harness.tool_contracts import (
    resolver_digest,
)
from harness.turn.node import Next, Node, NodeCtx

logger = logging.getLogger(__name__)


def _seed_hand_back(hand_back: dict[str, str]) -> dict[str, Any]:
    """`hand_back`, `{"status", "text"}`, as the `AgentHandBack` that answers
    none of the agent's calls."""
    return AgentHandBack(
        answers=None, status=hand_back["status"], text=hand_back["text"]
    ).model_dump(mode="json")


# The orchestrator's `fetch` tool, called with its arguments and the call
# (`harness.orchestrator_tools.build_orchestrator_tools`).
type Fetch = Callable[..., Awaitable[dict[str, Any]]]


def chip_fetch(tools: list[Any]) -> Fetch | None:
    """The `fetch` tool among the orchestrator's `tools`, answering a raise
    as the agent's own call is answered (`raised_answer`); `None` without
    one."""
    fetch = next((tool for tool in tools if tool.__name__ == "fetch"), None)
    if fetch is None:
        return None

    async def answered(*, call: OrchestratorCall, **args: Any) -> dict[str, Any]:
        try:
            return await fetch(call=call, **args)
        except Exception as error:
            logger.error(
                "the chip's fetch raised %s turn_id=%s",
                type(error).__name__,
                call.turn_id,
                exc_info=error,
            )
            return raised_answer("fetch", error)

    return answered


def seed_node(
    *,
    executor: Executor,
    definition_surface: ToolSurface,
    resume: ResumeRuns | None = None,
    fetch: Fetch | None = None,
) -> Node[GraphEvent]:
    """The `seed` node, carrying on unfinished runs through `resume`,
    resolving seeds and applying a chip through `definition_surface`,
    fetching a chip's data with `fetch`, the orchestrator's `fetch` tool
    (`chip_fetch`), and recording its steps through `executor`."""

    async def receive_seeds(ctx: NodeCtx[GraphEvent], node_input: Any) -> Next:
        """Carry on the runs an earlier turn left unfinished (`resume`,
        `harness.graph.resume`), then hand the agent the datasets the user
        brought from Search, or the data of the chip the user clicked
        (`APPLY_CHOICE_STATE_KEY`, `harness.graph.switch.apply_stored_choice`).
        The hand-back names the runs first, then the seeds or the chip, each
        as its own paragraph. It answers none of the agent's calls
        (`AgentHandBack`), so the agent reads it after the user's message
        (`orchestrator.run.Opening.seeded`). Once the turn's cursor holds
        it, the runs it delivers are acknowledged
        (`harness.graph.resume.Resumed.acknowledge`, run after the commit).

        The node's output is that hand-back, or None when there is none of
        these. The node runs once per turn, as its first; a rerun after a
        crash before its boundary does it all again.

        Raises:
            RuntimeError: If the turn applies a chip and the node has no `fetch`.
        """
        del node_input
        raw = TurnInput.model_validate(ctx.inputs).seeds
        resumed = await resume(ctx) if resume is not None else None
        seeded = await take(ctx, [] if raw is None else raw)
        pick = ctx.state.get(APPLY_CHOICE_STATE_KEY)
        if seeded is None and pick is not None:
            if fetch is None:
                raise RuntimeError("the seed node applies a chip only with a fetch tool")
            seeded = await apply_stored_choice(
                ctx,
                choice_id=str(pick["choice_id"]),
                option_id=str(pick["option_id"]),
                spec_id=str(pick["spec_id"]),
                surface=definition_surface,
                fetch=fetch,
                record_step=getattr(executor, "record_step", None),
            )
        if resumed is None:
            return Next("next", _seed_hand_back(seeded) if seeded is not None else None)
        ctx.after_commit(resumed.acknowledge)
        runs = resumed.hand_back
        return Next(
            "next",
            _seed_hand_back(
                runs
                if seeded is None
                else {"status": seeded["status"], "text": f"{runs['text']}\n\n{seeded['text']}"}
            ),
        )

    async def take(ctx: NodeCtx[GraphEvent], raw: Any) -> dict[str, str] | None:
        """Hand the agent the datasets the user brought from Search, as
        definite search results.

        The seeds slot (`TurnInput.seeds`) holds them as `seeds_slot` kept
        them. Each seed is taken once
        (`take_seeds`), and each one taken is resolved again through
        Search's `dataset_resubmit` door, all at once, through
        `definition_surface` (`resolve_seed`), so every fact the turn reads
        of it is Search's current answer, never the browser's block.

        The seeds kept are stored as `search` stores a definite part
        (`store_search`, with origin `seed`), so `fetch`, `plan` and later
        turns' ledgers read them as they read a search. The output is the hand-back
        `orchestrate` gives the agent (`seeds_hand_back`): the kept seeds as
        `search` returns its parts, with their operations, and each seed
        left out with why.

        A seed data-mcp or Search could not be reached for fails its own
        step as `service_unavailable`; one Search does not resolve to the
        dataset the user brought fails as `SEED_DROPPED`. Either is left
        out, and the turn goes on from the other seeds, or from the question
        when none is kept. The `SEEDS_STEP_ID` step counts the seeds kept, or
        fails `service_unavailable` when none was kept and one could not be
        reached. A slot holding one `{spec_id, content_version}` reference
        fails that step as `SEED_REFERENCE_UNUSED`.

        `None` when no seed was taken: the turn starts from the question.
        """
        record_step = getattr(executor, "record_step", None)
        if isinstance(raw, dict):
            refuse_seed_reference(record_step)
        taken = take_seeds(raw, record_step=record_step)
        if not taken:
            if isinstance(raw, list) and raw:
                seeds_received(record_step, posted=len(raw), taken=0, kept=0)
            return None
        answers = await asyncio.gather(
            *(resolve_seed(definition_surface, item) for item in taken),
            return_exceptions=True,
        )
        parts: list[dict[str, Any]] = []
        operations: list[list[dict[str, Any]]] = []
        unreachable: list[DefinitionNotResolvedError] = []
        drops: list[DroppedSeed] = []
        for item, answer in zip(taken, answers, strict=True):
            # Named by the seed's position: Search gave no name the turn can trust.
            name = f"dataset {item.index + 1} from Search"
            if isinstance(answer, DefinitionNotResolvedError) and answer.retryable:
                logger.warning("data-mcp could not resolve seed %d", item.index, exc_info=answer)
                unreachable.append(answer)
                refuse_seed(
                    record_step,
                    item.index,
                    code="service_unavailable",
                    retry_mode="retry_after_delay",
                    detail=f"data-mcp could not resolve seed {item.index}: {answer}",
                )
                drops.append(DroppedSeed(name=name, reason="it could not be read just now"))
                continue
            if isinstance(answer, DefinitionNotResolvedError):
                reason = str(answer)
            elif isinstance(answer, BaseException):
                raise answer
            else:
                try:
                    block = answered_block(item, answer)
                except StaleSeedError as exc:
                    reason = str(exc)
                except ValidationError as exc:
                    logger.warning(
                        "Search's answer for seed %d cannot be read", item.index, exc_info=True
                    )
                    reason = f"Search's answer for it cannot be read ({exc.error_count()} errors)"
                else:
                    parts.append({"status": "definite", "spec": block.model_dump(mode="json")})
                    operations.append(
                        [
                            operation.model_dump(mode="json", exclude_none=True)
                            for operation in item.seed.operations
                        ]
                    )
                    continue
            reason = reason.rstrip(".")
            refuse_seed(
                record_step,
                item.index,
                code=SEED_DROPPED,
                detail=f"seed {item.index} is left out: {reason}. Search for it again to add it.",
            )
            drops.append(DroppedSeed(name=name, reason=reason))
        seeds_received(
            record_step,
            posted=len(raw),
            taken=len(taken),
            kept=len(parts),
            unreachable=unreachable,
        )
        readings = (
            store_search(ctx.state, resolver_digest({"parts": parts}), origin="seed")
            if parts
            else []
        )
        return {
            "status": "Reading the datasets you brought from Search",
            "text": seeds_hand_back(
                # No question named a seed, so its part has no variable.
                [
                    {
                        **{key: value for key, value in reading.items() if key != "variable"},
                        "operations": asked,
                    }
                    for reading, asked in zip(readings, operations, strict=True)
                ],
                drops,
                budgets=turn_budgets(ctx.state),
            ),
        }

    return Node(run=receive_seeds)
