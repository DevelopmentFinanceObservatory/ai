"""The conversation-identity contextvar shared by the studio layer and tools.

`harness.studio` sets it for the duration of each streamed segment. Every
MCP client registry the harness holds reads it through `current_caller_turn`
to pick the client of the conversation's current turn, which names the
conversation to the MCP servers in its `x-dfo-caller` header
(`common.caller`), which data-mcp and analysis-mcp record as a stored frame's
`writer`. It is a contextvar
because the value has to cross the graph: studio -> ADK runner -> graph
node -> executor, and none of those layers carries wire identity, since the
graph must stay runnable without any studio layer at all.

`current_turn_id` holds the Studio turn's wire trace id beside it, which the
same clients send in their `x-dfo-turn` header (`current_caller_turn`), so a
dfo-apis call names the turn that caused it. A header is fixed when a client
is built, so each turn gets clients of its own
(`compiler.caller_clients.CallerClients`). `shared_work_context` clears both
for a task several requests join, and detaches it from the turn's trace.

`current_user_message` holds the user's own message for the turn, which the
orchestrator's `search` sends Search to read intent and explicit scope from.
`harness.studio` sets it at the start of a turn and again on each resume,
when an amendment or an answer to a question has been added to it.
"""

from __future__ import annotations

from contextvars import Context, ContextVar

from common import caller as common_caller
from common.caller import conversation_caller
from compiler.caller_clients import CallerTurn
from compiler.surface import current_spec_uses
from opentelemetry import context as otel_context

NO_CONVERSATION = "no-conversation"

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
    current Studio turn. Outside any conversation it names neither."""
    caller = current_conversation_caller()
    return CallerTurn(caller, current_turn_id.get() if caller is not None else None)


def shared_work_context() -> Context:
    """A copy of the current context that names no conversation and no turn.

    For a task several requests join, such as a card's replay two conversations
    both wait on (`harness.rolling_cards`). The task's MCP clients name no
    caller and no turn (`current_caller_turn`), and its in-process dfo-apis
    calls none either (`common.caller.shared_work_context`), so no
    conversation's identity is attached to the shared work. The copy holds an
    empty OpenTelemetry context too, so the task records on no turn's trace
    and its MCP calls carry no turn's Langfuse session, user or tags
    (`common.telemetry.turn_call_meta`), as work outside any turn does. It
    reports its fetches to no turn's spec-use sink (`current_spec_uses`).
    """
    context = common_caller.shared_work_context()
    context.run(current_conversation_id.set, NO_CONVERSATION)
    context.run(current_conversation_key.set, None)
    context.run(current_turn_id.set, None)
    context.run(current_spec_uses.set, None)
    # The attach lives in the copy and dies with it.
    context.run(otel_context.attach, otel_context.Context())
    return context


def conversation_of(session_id: str) -> str:
    """The conversation a turn whose ADK session is `session_id` belongs to:
    the studio's, which spans its turns, or outside the studio the session's."""
    conversation = current_conversation_id.get()
    return session_id if conversation == NO_CONVERSATION else conversation


__all__ = [
    "NO_CONVERSATION",
    "conversation_of",
    "current_caller_turn",
    "current_conversation_caller",
    "current_conversation_id",
    "current_conversation_key",
    "current_turn_id",
    "current_user_message",
    "shared_work_context",
]
