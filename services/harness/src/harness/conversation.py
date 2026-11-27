"""The conversation-identity contextvar shared by the studio layer and tools.

`harness.studio` sets it for the duration of each streamed segment. Every
MCP client registry the harness holds reads it through `current_caller_turn`
to pick the client of the conversation's current turn, which names the
conversation to the MCP servers in its `x-dfo-caller` header
(`common.caller`), which data-mcp and analysis-mcp record as a stored frame's
`writer`. It is a contextvar
because the value has to cross the graph: studio -> turn engine -> graph
node -> executor, and none of those layers carries wire identity, since the
graph must stay runnable without any studio layer at all.

`current_turn_id` holds the Studio turn's wire trace id beside it, which the
same clients send in their `x-dfo-turn` header (`current_caller_turn`), so a
dfo-apis call names the turn that caused it. A header is fixed when a client
is built, so each turn gets clients of its own
(`compiler.caller_clients.CallerClients`). `shared_work_context` clears both
for a task several requests join, names the Studio surface in their place,
and detaches it from the turn's trace.

`current_history` holds the turn's history arm and how it was chosen, fixed
as the turn began (`harness.studio_state.TurnUsage.history`), which every
model span of the turn names (`harness.gemini_wire.surface_attributes`).

`current_model_usage` is the sink each model call attempt of the turn reports
its model, role and token usage to (`note_model_usage`), which writes it as
one `harness.studio_state.MODEL_CALL_LOG_EVENT` line
(`harness.studio_store.write_model_call_line`).

`current_user_message` holds the user's own message for the turn, which the
orchestrator's `search` sends Search to read intent and explicit scope from.
`harness.studio` sets it at the start of a turn and again on each resume,
when an amendment or an answer to a question has been added to it.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from contextvars import Context, ContextVar
from typing import TYPE_CHECKING, Literal, NamedTuple

from common import caller as common_caller
from common.caller import conversation_caller
from compiler.caller_clients import CallerTurn
from compiler.surface import current_spec_uses
from google.genai import types
from opentelemetry import context as otel_context

if TYPE_CHECKING:
    from harness.settings import StudioHistory

logger = logging.getLogger(__name__)

NO_CONVERSATION = "no-conversation"


def ledger_conversation_id(principal: str, conversation_id: str) -> str:
    """The id the studio names `principal`'s conversation `conversation_id`
    by: what `current_conversation_id` holds for its segments, what the
    recording surface keys its step order on, and what its runs are admitted
    under (`executor.ledger.idempotency_key`), by `execute` and by the
    interrupt route. Every turn of the conversation names the same one, so a
    later turn executing the same revision carries the run on."""
    return f"{principal}/{conversation_id}"


current_conversation_id: ContextVar[str] = ContextVar(
    "studio_conversation_id", default=NO_CONVERSATION
)

# The same identity as a (principal, conversation_id) PAIR, for lookups that
# need the store key itself (the recall function). Kept
# separate because principal and conversation_id are both client-influenced
# strings -- splitting the joined form back apart would be ambiguous.
current_conversation_key: ContextVar[tuple[str, str] | None] = ContextVar(
    "studio_conversation_key", default=None
)


# The current Studio turn's wire trace id (`ConversationState.trace_id`), set
# beside `current_conversation_id`; `None` outside a Studio turn.
current_turn_id: ContextVar[str | None] = ContextVar("studio_turn_id", default=None)


class TurnHistory(NamedTuple):
    """A Studio turn's history arm, `digest` or `full`, and its `mode`: `split`
    for the arm stored with the conversation, `forced` for the one
    `STUDIO_HISTORY` named, `None` for a turn that began before turns kept it."""

    arm: StudioHistory
    mode: Literal["split", "forced"] | None


# The current Studio turn's history arm, `None` outside a Studio turn.
current_history: ContextVar[TurnHistory | None] = ContextVar("studio_history", default=None)


# The agent a model call serves: the orchestrator, the planner, or the Edit
# Card (`harness.generative.GeminiEdit`).
type ModelRole = Literal["orchestrator", "planner", "edit"]


class ModelCall(NamedTuple):
    """One model call attempt that reported usage: the agent it served, the
    model it was sent to, and the usage Gemini reported for it."""

    role: ModelRole
    model: str
    usage: types.GenerateContentResponseUsageMetadata


# Where the current Studio turn hears each model call it makes, set by
# `harness.studio` for each request that runs a turn. It is `None` outside one,
# where nothing is heard.
current_model_usage: ContextVar[Callable[[ModelCall], None] | None] = ContextVar(
    "studio_model_usage", default=None
)


def note_model_usage(
    usage: types.GenerateContentResponseUsageMetadata | None, *, model: str, role: ModelRole
) -> None:
    """Tell the current turn a model call attempt to `model`, serving `role`,
    spent `usage` (`current_model_usage`). A no-op outside a turn and for an
    attempt that reported none. Never raises, so a turn's cost record cannot
    fail the call it records."""
    sink = current_model_usage.get()
    if sink is None or usage is None:
        return
    try:
        sink(ModelCall(role=role, model=model, usage=usage))
    except Exception:
        logger.warning("recording a model call's usage failed", exc_info=True)


# The user's words for the current turn as `ConversationState.turn_user_words`
# holds them, amendments and answers included and plan-card switches left
# out; `None` outside a Studio turn.
current_user_message: ContextVar[str | None] = ContextVar("studio_user_message", default=None)


def current_conversation_caller() -> str | None:
    """The caller the harness names in its MCP calls: the current
    conversation's, or `None` outside any conversation, which sends no
    caller rather than one every conversation-less call would share.

    `None` too inside work several requests join
    (`common.caller.in_shared_work`), so a task run in
    `common.caller.shared_work_context` names no conversation even where it
    keeps the conversation's ids, as `harness.recipe_replay`'s node tasks do.
    """
    conversation = current_conversation_id.get()
    if conversation == NO_CONVERSATION or common_caller.in_shared_work():
        return None
    return conversation_caller(conversation)


def current_caller_turn() -> CallerTurn:
    """The caller and turn the harness names in its MCP calls: the current
    conversation's caller (`current_conversation_caller`) and, inside one, the
    current Studio turn. Outside any conversation it names neither, and inside
    shared work it names the surface the work serves
    (`common.caller.current_shared_surface`)."""
    caller = current_conversation_caller()
    if caller is None:
        return CallerTurn(None, surface=common_caller.current_shared_surface())
    return CallerTurn(caller, current_turn_id.get())


def shared_work_context() -> Context:
    """A copy of the current context that names no conversation and no turn,
    and serves the Studio surface.

    For a task several requests join, such as a card's replay two conversations
    both wait on (`harness.rolling_cards`). The task's MCP clients name no
    caller and no turn, only the Studio surface (`current_caller_turn`), and its
    in-process dfo-apis calls none either (`common.caller.shared_work_context`),
    so no conversation's identity is attached to the shared work, and its
    dfo-apis calls still count as Studio's. The copy holds an
    empty OpenTelemetry context too, so the task records on no turn's trace
    and its MCP calls carry no turn's Langfuse session, user or tags
    (`common.telemetry.turn_call_meta`), as work outside any turn does. It
    reports its fetches to no turn's spec-use sink (`current_spec_uses`) and its
    model calls to no turn's usage sink (`current_model_usage`).
    """
    context = common_caller.shared_work_context(surface=common_caller.STUDIO_SURFACE)
    context.run(current_conversation_id.set, NO_CONVERSATION)
    context.run(current_conversation_key.set, None)
    context.run(current_turn_id.set, None)
    context.run(current_history.set, None)
    context.run(current_model_usage.set, None)
    context.run(current_spec_uses.set, None)
    # The attach lives in the copy and dies with it.
    context.run(otel_context.attach, otel_context.Context())
    return context


__all__ = [
    "NO_CONVERSATION",
    "ModelCall",
    "ModelRole",
    "TurnHistory",
    "current_caller_turn",
    "current_conversation_caller",
    "current_conversation_id",
    "current_conversation_key",
    "current_history",
    "current_model_usage",
    "current_turn_id",
    "current_user_message",
    "note_model_usage",
    "shared_work_context",
]
