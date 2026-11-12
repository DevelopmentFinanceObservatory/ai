"""The planner as a workflow node (`planner_node`).

`harness.planner_adapter.PlannerAdapter` runs the planner through `ctx.run_node`.
That node is a function node named `planner.settings.AGENT_NAME` that drives
`planner.pydantic_agent.run_planner` on `harness.planner_model.planner_model`
and yields events holding what an ADK `LlmAgent` stores for the same
exchange, as far as the session's readers read it (`harness.planner_events`).
ADK's node runner stamps each with the author the node sets, the branch of the
adapter's `override_branch`, the invocation and the node's path, which
`planner.session.session_steps`, `harness.studio_translate` and the adapter read.

The run's agent is named `AGENT_NAME`, its spans name the session as their
conversation, and they carry message content when ADK's would
(`google.adk.telemetry.TelemetryConfig`, which reads
`OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT`).

The node reports to the adapter's `harness.planner_run.PlannerRun`: each tool's
answer, each tool that raised, a call the turn budget closed
(`PlannerRun.model_answered`), and before each request whether the run asks
or stops instead (`PlannerRun.next_request`). A directive is answered with
the model turn `directive_turn` builds.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Sequence
from contextlib import aclosing
from typing import Any

from common.telemetry import current_trace_id
from executor.classify import internal_fault
from google.adk import Context, Event
from google.adk.flows.llm_flows import functions as adk_functions
from google.adk.telemetry import TelemetryConfig
from google.adk.workflow import BaseNode, node
from google.genai import Client, types
from planner.pydantic_agent import (
    PlannerAnswers,
    PlannerDeps,
    PydanticPlanner,
    UnsentQuestionPartError,
    planner_agent,
    run_planner,
)
from planner.session import SessionCall
from planner.settings import AGENT_NAME
from planner.tool_spec import ToolSpec

from harness.planner_events import model_response_event, tool_answers_event
from harness.planner_model import (
    StreamResumedError,
    Usage,
    planner_model,
    resumed_over_output,
)
from harness.planner_run import Ask, Directive, PlannerRun, Stop
from harness.prefix_cache import PrefixCache
from harness.sentences import StopCode, refusal_sentence
from harness.turn_budget import TurnBudget
from harness.turn_events import PLANNER_ASK, turn_question
from harness.usage import log_usage

logger = logging.getLogger(__name__)

# Gemini requires a `thought_signature` on a generation's first `functionCall`
# part and answers a request replaying one without it with a 400. This is
# the documented placeholder that makes the API skip validation for a call
# injected outside a model turn.
SKIP_THOUGHT_SIGNATURE = b"skip_thought_signature_validator"


def pydantic_planner(specs: Sequence[ToolSpec], *, instruction: str) -> PydanticPlanner:
    """The planner on `specs` and `instruction`, named
    `planner.settings.AGENT_NAME`. It has no model of its own, since each run
    names one (`planner_node`). Built once, it serves every run. Its spans
    carry message content when ADK's telemetry settings say an ADK agent's do
    (`google.adk.telemetry.TelemetryConfig`), read once, here."""
    return planner_agent(
        specs,
        instruction=instruction,
        name=AGENT_NAME,
        include_content=TelemetryConfig().should_add_content_to_experimental_spans,
    )


def planner_node(
    run: PlannerRun,
    *,
    planner: PydanticPlanner,
    model_name: str,
    client: Client,
    budget: TurnBudget | None,
    prefix_cache: PrefixCache | None = None,
    brief: str,
) -> BaseNode:
    """One run of the planner, as the node `ctx.run_node` runs.

    Args:
        run: The run's bookkeeping, told what the run does and asked what
            replaces each request.
        planner: The planner (`pydantic_planner`).
        model_name: The Gemini model, as google-genai names it.
        client: The google-genai client the model's calls go through.
        budget: The turn's budget, or None to run every call unbounded.
        prefix_cache: The process's shared cache of static prompt prefixes,
            or None to send every request whole.
        brief: The brief the planner is shown after the turn's question.

    Returns:
        A node named `planner.settings.AGENT_NAME` that runs the planner once on
        `brief`, after the turn's question, and yields the run's events
        (`harness.planner_events`). Each model call the run makes is counted on
        the invocation's counter (`InvocationContext.increment_llm_call_count`),
        with the orchestrator's and every other planner run's. A turn put in
        the model's place is not counted, as ADK counts no response a callback
        gives. ADK raises `LlmCallsLimitExceededError` on the call past
        `RunConfig.max_llm_calls`.
        A question holding a part this planner does not send, or no part
        (`planner.pydantic_agent.UnsentQuestionPartError`), ends the run before
        any request on an `internal_error` refusal (`run.refused`). The adapter
        answers the entry with that refusal, marked not final
        (`PlannerAdapter._FINAL_CODES`), so the orchestrator may ask for a plan
        again and meet the same refusal. Studio sends only text.
        A stream google-genai resumed over output no continuation token
        covers (`harness.planner_model.resumed_over_output`) raises
        `StreamResumedError`, a dropped connection, from Pydantic AI's
        error.
    """

    async def dfo_planner(ctx: Context, node_input: Any) -> AsyncIterator[Event]:
        del node_input
        ctx.event_author = AGENT_NAME
        invocation = ctx.get_invocation_context()

        def before_request() -> types.Content | None:
            turn = directive_turn(run.next_request())
            if turn is None:
                invocation.increment_llm_call_count()
            return turn

        def after_response(closed_by: StopCode | None, usage: Usage | None) -> types.Content | None:
            # A call a bound closed is logged with the usage its chunks
            # reported before the cut.
            if usage is not None:
                log_usage(AGENT_NAME, usage)
            if closed_by is not None:
                return directive_turn(run.model_answered(closed_by))
            return None

        def tool_failed(name: str, error: BaseException) -> dict[str, Any]:
            logger.error(
                "planner tool %s raised %s session=%s invocation=%s trace_id=%s",
                name,
                type(error).__name__,
                ctx.session.id,
                ctx.invocation_id,
                current_trace_id(),
                exc_info=error,
            )
            return run.tool_failed(name, error)

        model = planner_model(
            model_name,
            client=client,
            budget=budget,
            prefix_cache=prefix_cache,
            before_request=before_request,
            after_response=after_response,
        )
        deps = PlannerDeps(
            state=ctx.session.state,
            call_for=lambda state, call_id: SessionCall(ctx.session.events, state, call_id),
            tool_answered=run.tool_answered,
            tool_failed=tool_failed,
        )
        items = run_planner(
            planner,
            deps,
            model=model,
            question=turn_question(ctx),
            brief=brief,
            conversation_id=ctx.session.id,
        )
        # Closed here, in the node's task, when ADK's node runner closes the
        # node at a yield on a Stop, so the run's spans and context variables
        # end in the task that opened them.
        try:
            async with aclosing(items):
                async for item in items:
                    if isinstance(item, PlannerAnswers):
                        yield tool_answers_event(item)
                    # A response with no parts, an empty or a blocked one,
                    # leaves nothing a reader reads.
                    elif item.parts:
                        yield model_response_event(item)
        except UnsentQuestionPartError as exc:
            # Refused before any request. The run ends on the refusal, which
            # the adapter answers the entry with. An exception here would fail
            # the whole workflow, since ADK fails the root on a child node's
            # error.
            logger.error(
                "planner question refused session=%s invocation=%s: %s",
                ctx.session.id,
                ctx.invocation_id,
                exc,
            )
            run.refused = internal_fault(str(exc))
            return
        except Exception as exc:
            if resumed_over_output(exc):
                # A dropped connection the SDK resumed. Raised as one, so the
                # turn ends as a dropped Gemini call ends it
                # (`compiler.surface.transient`). google-genai keeps the
                # transport error that dropped the stream to itself.
                logger.warning(
                    "planner stream resumed over output session=%s invocation=%s trace_id=%s: %s",
                    ctx.session.id,
                    ctx.invocation_id,
                    current_trace_id(),
                    exc,
                )
                raise StreamResumedError(str(exc)) from exc
            raise

    return node(dfo_planner, name=AGENT_NAME)


def directive_turn(directive: Directive | None) -> types.Content | None:
    """The model turn `directive` stands for, or `None` to keep the model's own.

    An ask is one `PLANNER_ASK` call with the ask's arguments, under a call id
    of the form ADK gives a call the model sent without one
    (`google.adk.flows.llm_flows.functions.generate_client_function_call_id`),
    so the resume pairs the user's answer with it the way it pairs one with a
    model-issued ask (`planner.call.elected_spec_ids`). A stop is the sentence
    of the refusal the run ends on (`harness.sentences.refusal_sentence`).
    """
    match directive:
        case Ask(args=args):
            call_id = adk_functions.generate_client_function_call_id()
            call = types.FunctionCall(id=call_id, name=PLANNER_ASK, args=args)
            part = types.Part(function_call=call, thought_signature=SKIP_THOUGHT_SIGNATURE)
        case Stop(refusal=refusal):
            part = types.Part(text=f"Stopping this turn: {refusal_sentence(refusal)}")
        case None:
            return None
    return types.Content(role="model", parts=[part])


__all__ = ["SKIP_THOUGHT_SIGNATURE", "directive_turn", "planner_node", "pydantic_planner"]
