"""The conversation-identity contextvar shared by the studio layer and tools.

`harness.studio` sets it for the duration of each streamed segment. Every
MCP client the harness holds reads it through `current_conversation_caller`
to pick the conversation's client, which names the conversation to the MCP
servers in its `x-dfo-caller` header (`common.caller`), which data-mcp and
analysis-mcp record as a stored frame's `writer`. It is a contextvar
because the value has to cross the graph: studio -> ADK runner -> graph
node -> executor, and none of those seams carries wire identity, since the
graph must stay runnable without any studio layer at all.

`current_user_message` holds the user's own message for the turn, which the
orchestrator's `search` sends Search to read intent and explicit scope from.
`harness.studio` sets it at the start of a turn and again on each resume,
when an amendment or an answer to a question has been added to it.
"""

from __future__ import annotations

from contextvars import ContextVar

from common.caller import conversation_caller

NO_CONVERSATION = "no-conversation"

current_conversation_id: ContextVar[str] = ContextVar(
    "studio_conversation_id", default=NO_CONVERSATION
)

# The same identity as a (principal, conversation_id) PAIR, for lookups that
# need the store key itself (the recall seam). Kept
# separate because principal and conversation_id are both client-influenced
# strings -- splitting the joined form back apart would be ambiguous.
current_conversation_key: ContextVar[tuple[str, str] | None] = ContextVar(
    "studio_conversation_key", default=None
)


# The user's words for the current turn as `ConversationState.turn_user_words`
# holds them, amendments and answers included and plan-card switches left
# out; `None` outside a Studio turn.
current_user_message: ContextVar[str | None] = ContextVar("studio_user_message", default=None)


def current_conversation_caller() -> str | None:
    """The caller the harness names in its MCP calls: the current
    conversation's, or `None` outside any conversation, which sends no
    caller rather than one every conversation-less call would share."""
    conversation = current_conversation_id.get()
    return None if conversation == NO_CONVERSATION else conversation_caller(conversation)


def conversation_of(session_id: str) -> str:
    """The conversation a turn whose ADK session is `session_id` belongs to:
    the studio's, which spans its turns, or outside the studio the session's."""
    conversation = current_conversation_id.get()
    return session_id if conversation == NO_CONVERSATION else conversation


__all__ = [
    "NO_CONVERSATION",
    "conversation_of",
    "current_conversation_caller",
    "current_conversation_id",
    "current_conversation_key",
    "current_user_message",
]
