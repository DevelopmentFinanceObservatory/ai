"""Stage A's planner node, fitted to the real root workflow.

The relay is stage A's `_relay_live` unchanged: each `PlanProgress` becomes an
`Event(custom_metadata={"progress": ...})` the moment it arrives, the key dev's
`planner_node` relays its queued reports under. Around it this node does what
dev's node does on a first draft (`harness.graph.planning.planner_node`): the
`PLANNER_STARTED_KEY` event, the plan validated and written to `ctx.state["plan"]`,
and an `output` with `summary` and `revision` on the `"ok"` route. The ask becomes
an interrupt on `harness.graph.state.interrupt_event`'s branch, as `approve`'s is.

`dev_stub_seam` is the control: a `PlannerModel` seam for dev's own node that
reports the same progress through `on_progress` and returns the same plan.
"""

from collections.abc import AsyncIterator, Callable
from enum import StrEnum
from typing import Any

from google.adk import Context, Event
from google.adk.events import EventActions, RequestInput
from google.genai import types
from graph import CONVERSATION_KEY, PLANNER_CONVERSATION_ID, _Entry, _relay_live
from harness.graph.state import ORCHESTRATOR_BRIEF_KEY, PLANNER_STARTED_KEY, interrupt_event
from harness.scripted_plan import scripted_plan
from orchestrator.agent import AGENT_NAME as ORCHESTRATOR_AGENT_NAME
from planner.plan import Plan as PlannerPlan
from planner.plan import plan_revision
from planner_agent import PlanProgress, plan_of
from pydantic_ai import Conversation, ConversationTypeAdapter, DeferredToolRequests
from pydantic_ai.run import AgentRunResult
from pydantic_ai.tools import DeferredToolResults
from studio_planner import StudioDeps, build_studio_planner, report_progress

ASK_PREFIX = "planner_ask:"


def _answer_text(answer: dict[str, Any]) -> str:
    """The user's answer to the planner's one question as the planner reads it:
    the option or text Studio's `/interrupt` route sends for `q1`, or `skipped`."""
    if answer.get("skip"):
        return "skipped"
    given = answer["answers"]["q1"]
    if given["kind"] == "text":
        return str(given["text"])
    if given["kind"] == "option":
        return str(given["spec_id"])
    return "skipped"


def _answer_for(ctx: Context, requests: DeferredToolRequests) -> DeferredToolResults | None:
    (call,) = requests.calls
    answer = ctx.resume_inputs.get(ASK_PREFIX + call.tool_call_id)
    if answer is None:
        return None
    return requests.build_results(calls={call.tool_call_id: _answer_text(answer)})


def _entry(ctx: Context) -> _Entry:
    """Continue the planner this invocation paused when its ask is answered;
    otherwise start from the orchestrator's brief."""
    stored = ctx.state.get(CONVERSATION_KEY)
    if stored and stored["invocation_id"] == ctx.invocation_id:
        conversation = ConversationTypeAdapter.validate_json(stored["conversation"])
        requests = conversation.deferred_tool_requests
        assert requests is not None
        results = _answer_for(ctx, requests)
        if results is not None:
            return _Entry(None, conversation, results)
    return _Entry(
        str(ctx.state.get(ORCHESTRATOR_BRIEF_KEY, "")),
        Conversation(conversation_id=PLANNER_CONVERSATION_ID),
        None,
    )


class Leak(StrEnum):
    """Where S3's probe writes the planner's text into the session."""

    # A model-role event on the empty branch, which ADK's branch rule shows
    # every agent (`_is_event_belongs_to_branch`).
    MODEL_EMPTY_BRANCH = "model_empty_branch"
    # A model-role event on the orchestrator's own branch.
    MODEL_ORCHESTRATOR_BRANCH = "model_orchestrator_branch"
    # Authored `user` on the orchestrator's branch, as `harness.graph.orchestrate`
    # writes a hand-back.
    USER_ORCHESTRATOR_BRANCH = "user_orchestrator_branch"


def _leaked(ctx: Context, leak: Leak, text: str) -> Event:
    role = "user" if leak is Leak.USER_ORCHESTRATOR_BRANCH else "model"
    event = Event(
        content=types.Content(role=role, parts=[types.Part(text=text)]),
        branch="" if leak is Leak.MODEL_EMPTY_BRANCH else ORCHESTRATOR_AGENT_NAME,
    )
    if leak is Leak.USER_ORCHESTRATOR_BRANCH:
        event.author = "user"
    return event


def pai_planner_node(
    *, deps: StudioDeps, leak: Leak | None = None
) -> Callable[[Context, Any], AsyncIterator[Any]]:
    """The `planner` node on the Pydantic AI planner. `leak`, S3's probe, also
    yields the planner's final text as session content in the shape it names."""
    agent = build_studio_planner(deps)

    async def planner(ctx: Context, node_input: Any) -> AsyncIterator[Any]:
        yield Event(custom_metadata={PLANNER_STARTED_KEY: {"revising": False}})
        entry = _entry(ctx)
        while True:
            result: AgentRunResult | None = None
            # Stage A types the relay for its own `Deps`; it only passes deps through.
            async for item in _relay_live(agent, deps, entry):  # ty: ignore[invalid-argument-type]
                if isinstance(item, AgentRunResult):
                    result = item
                else:
                    yield item
            assert result is not None
            output = result.output
            if not isinstance(output, DeferredToolRequests):
                break
            answered = _answer_for(ctx, output)
            if answered is None:
                ctx.state[CONVERSATION_KEY] = {
                    "invocation_id": ctx.invocation_id,
                    "conversation": ConversationTypeAdapter.dump_json(result.conversation).decode(),
                }
                (call,) = output.calls
                args = call.args_as_dict()
                yield interrupt_event(
                    RequestInput(
                        interrupt_id=ASK_PREFIX + call.tool_call_id,
                        message=args["message"],
                        payload={"options": args["options"]},
                    )
                )
                return
            entry = _Entry(None, result.conversation, answered)
        ctx.state[CONVERSATION_KEY] = None
        plan = PlannerPlan.model_validate(plan_of(result.conversation.messages))
        ctx.state["plan"] = plan.model_dump_json()
        if leak is not None:
            yield _leaked(ctx, leak, output)
        yield Event(
            output={
                **plan.model_dump(mode="json"),
                "summary": {"version": plan.version, "revision": plan_revision(plan)},
                "revision": plan_revision(plan),
            },
            actions=EventActions(route="ok"),
        )

    return planner


class DevStubSeam:
    """A `PlannerModel` seam for dev's `planner_node`: reports progress through
    `on_progress` (same payload, same gate) and drafts the same plan."""

    def __init__(self, deps: StudioDeps) -> None:
        self.deps = deps

    async def __call__(
        self,
        *,
        question: str,
        on_progress: Callable[[dict[str, Any]], None] | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        async def emit(event: PlanProgress) -> None:
            if on_progress is not None:
                on_progress({"call": event.call, "done": event.done, "total": event.total})

        await report_progress(self.deps, call=1, emit=emit)
        plan = scripted_plan(question, spec_id=None)
        return {
            "plan": plan.model_dump(mode="json"),
            "summary": {"version": plan.version, "revision": plan_revision(plan)},
        }

    async def revise(self, **_: Any) -> dict[str, Any]:
        return {"refused": {"text": "no revision", "code": None}}
