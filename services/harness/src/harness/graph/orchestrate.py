"""The `orchestrate`, `route` and `ask` nodes: the agent's turn, the one action
it asked for, and its question to the user.
"""

from __future__ import annotations

from collections.abc import (
    AsyncIterator,
    Callable,
    Iterator,
)
from typing import Any

from google.adk import Context, Event
from google.adk.agents import LlmAgent
from google.adk.events import RequestInput
from google.genai import types

from harness.graph.state import (
    CLARIFY_INTERRUPT,
    ORCHESTRATOR_ASKED_KEY,
    ORCHESTRATOR_PENDING_KEY,
    ORCHESTRATOR_STATUS_KEY,
    ask_candidates,
    ask_options,
    interrupt_event,
    pending_action,
)


def orchestrate_node(*, orchestrator: LlmAgent) -> Callable[[Context, Any], AsyncIterator[Event]]:
    """The `orchestrate` node, running `orchestrator`."""

    async def orchestrate(ctx: Context, node_input: Any) -> AsyncIterator[Event]:
        """The orchestrator's turn: the agent runs with its tools on its own
        branch, so it sees the whole turn so far -- the user's message, its
        own calls, every hand-back -- and none of the planner's internal
        events. Reached from START with the user's message (already in the
        session), and again after `ask` and `plan_done` with text to hand
        back, which goes in as a user message of its own so it is saved
        with the session and still in view after a later pause.

        The hand-back is authored `user` on the agent's branch: ADK quotes an
        event another author wrote as that agent's data, and an unbranched
        event reaches every agent, the planner on a later `plan` call
        included. The author is restored before the agent runs, since a
        child context inherits it. An event's explicit branch also becomes
        its node's branch, which ADK hands to every node downstream; the
        status event that follows the hand-back carries the empty branch,
        which ADK reads as "clear it", so this node's output and every later
        node of the turn stay unbranched, as they are before any hand-back."""
        if isinstance(node_input, dict):
            author, ctx.event_author = ctx.event_author, "user"
            try:
                yield Event(
                    content=types.Content(
                        role="user", parts=[types.Part.from_text(text=node_input["text"])]
                    ),
                    branch=orchestrator.name,
                )
            finally:
                ctx.event_author = author
            yield Event(custom_metadata={ORCHESTRATOR_STATUS_KEY: node_input["status"]}, branch="")
        else:
            yield Event(custom_metadata={ORCHESTRATOR_STATUS_KEY: "Reading your question"})
        answer = await ctx.run_node(orchestrator, None, override_branch=orchestrator.name)
        yield Event(output=answer)

    return orchestrate


def route(ctx: Context, node_input: Any) -> Iterator[Event]:
    """The one action the agent asked for, or none: it answered."""
    del node_input
    pending = pending_action(ctx.state)
    ctx.state[ORCHESTRATOR_PENDING_KEY] = None
    if pending is None:
        yield Event(output=None, route="answered")
    else:
        yield Event(output=pending.model_dump(), route=pending.kind)


def ask(ctx: Context, node_input: dict[str, Any]) -> Iterator[Any]:
    """Put the agent's question to the user through the clarify
    interrupt, its options as candidates (`ask_candidates`), and hand
    the answer back in the option's own words, with the spec_id of a
    count picked. The question waits under `ORCHESTRATOR_ASKED_KEY`
    while the interrupt is open, so a resume is told from a fresh entry."""
    answer = (ctx.resume_inputs or {}).get(CLARIFY_INTERRUPT)
    asked = ctx.state.get(ORCHESTRATOR_ASKED_KEY)
    if not (asked and answer is not None):
        ctx.state[ORCHESTRATOR_ASKED_KEY] = node_input
        yield interrupt_event(
            RequestInput(
                interrupt_id=CLARIFY_INTERRUPT,
                message=node_input["question"],
                payload={"candidates": ask_candidates(node_input["options"])},
            )
        )
        return
    ctx.state[ORCHESTRATOR_ASKED_KEY] = None
    asked_options = ask_options(asked["options"])
    options = {
        candidate["spec_id"]: option
        for candidate, option in zip(ask_candidates(asked_options), asked_options, strict=True)
    }
    if isinstance(answer, dict) and answer.get("skip"):
        reply = (
            "The user skipped the question. Do not answer it or search for it: close "
            "the turn in a sentence and ask what else you can help with."
        )
    elif isinstance(answer, dict) and answer.get("text"):
        reply = f"The user answered in their own words: {answer['text']}"
    else:
        chosen = answer.get("spec_id") if isinstance(answer, dict) else str(answer)
        option = options.get(chosen)
        if option is None:
            reply = f"The user chose: {chosen}"
        elif option.get("spec_id"):
            reply = f"The user chose: {option['label']} ({option['spec_id']})"
        else:
            reply = f"The user chose: {option['label']}"
    yield Event(
        output={
            "status": "Reading your answer",
            "text": f"You asked the user: {asked['question']}\n{reply}",
        }
    )
