"""The `seed` node: opens the turn with the runs an earlier turn left
unfinished, carried on, and the datasets the user brought from Search or
the chip the user clicked."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import (
    AsyncIterator,
    Awaitable,
    Callable,
)
from typing import Any

from compiler.surface import ToolSurface
from google.adk import Context
from planner.plan import (
    SEEDS_STATE_KEY,
)
from pydantic import ValidationError
from translator.discovery import (
    DefinitionNotResolvedError,
)

from harness.graph.resume import ResumeRuns
from harness.graph.seams import (
    Executor,
)
from harness.graph.state import (
    APPLY_CHOICE_STATE_KEY,
    state_json,
    store_search,
    turn_budgets,
)
from harness.graph.switch import apply_stored_choice
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

logger = logging.getLogger(__name__)


def _turn_ran_before(ctx: Context) -> bool:
    """Whether another workflow node already ran in `ctx`'s invocation.

    ADK resumes a paused turn by replaying the workflow from START, and runs
    live any node the turn's events hold no run of
    (`google.adk.workflow._workflow.Workflow._start_node_task`). A turn that
    paused before the workflow had `seed` resumes with `seed` running after
    `orchestrate` already has, its seeds slot still holding what the host
    wrote at the turn's start.
    """
    own = ctx.node_path
    return any(
        event.invocation_id == ctx.invocation_id
        and event.node_info is not None
        and bool(event.node_info.path)
        and event.node_info.path != own
        and not event.node_info.path.startswith(f"{own}/")
        for event in ctx.session.events
    )


def seed_node(
    *,
    executor: Executor,
    definition_surface: ToolSurface,
    resume: ResumeRuns | None = None,
    fetch: Callable[..., Awaitable[dict[str, Any]]] | None = None,
) -> Callable[[Context, Any], AsyncIterator[dict[str, str]]]:
    """The `seed` node, carrying on unfinished runs through `resume`,
    resolving seeds and applying a chip through `definition_surface`,
    fetching a chip's data with `fetch`, the orchestrator's `fetch` tool,
    and recording its steps through `executor`."""

    async def receive_seeds(ctx: Context, node_input: Any) -> AsyncIterator[dict[str, str]]:
        """Carry on the runs an earlier turn left unfinished (`resume`,
        `harness.graph.resume`), then hand the agent the datasets the user
        brought from Search, or the data of the chip the user clicked
        (`APPLY_CHOICE_STATE_KEY`, `harness.graph.switch.apply_stored_choice`).
        The hand-back names the runs first, then the seeds or the chip, each
        as its own paragraph. Once the session holds it, which it does when
        the yield returns, the runs it delivers are acknowledged
        (`harness.graph.resume.Resumed.acknowledge`).

        No output when there is none of these. A run that resumes a turn
        already under way (`_turn_ran_before`) clears the seeds slot and
        does none of them. The chip's slot stays for the turn, since
        `harness.graph.state.turn_budgets` reads it.

        Raises:
            RuntimeError: If the turn applies a chip and the node has no `fetch`.
        """
        del node_input
        raw = state_json(ctx, SEEDS_STATE_KEY, default=[])
        ctx.state[SEEDS_STATE_KEY] = None
        if _turn_ran_before(ctx):
            return
        resumed = await resume(ctx) if resume is not None else None
        seeded = await take(ctx, raw)
        pick = state_json(ctx, APPLY_CHOICE_STATE_KEY)
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
            if seeded is not None:
                yield seeded
            return
        runs = resumed.hand_back
        yield (
            runs
            if seeded is None
            else {"status": seeded["status"], "text": f"{runs['text']}\n\n{seeded['text']}"}
        )
        await resumed.acknowledge()

    async def take(ctx: Context, raw: Any) -> dict[str, str] | None:
        """Hand the agent the datasets the user brought from Search, as
        definite search results.

        The seeds slot (`GraphState.seeds`) holds them as `seeds_slot` kept
        them; the node reads it once and clears it. Each seed is taken once
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

    return receive_seeds
