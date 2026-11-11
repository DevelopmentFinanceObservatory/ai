"""The Pydantic AI planner as a workflow node (`planner_node`).

`harness.planner_seam.PlannerSeam` runs the planner through `ctx.run_node`.
On the Pydantic AI planner that node is a function node named as the ADK
planner (`planner.agent.AGENT_NAME`) that drives `planner.pydantic_agent.
run_planner` on `harness.planner_model.planner_model` and yields the events the
ADK planner's `LlmAgent` stores for the same exchange
(`harness.planner_events`). ADK's node runner stamps each with the author the
node sets, the branch of the seam's `override_branch`, the invocation and the
node's path, so the session the run leaves reads as the ADK planner's to
`planner.agent.session_steps`, `harness.studio_translate` and the seam.

The run traces as the ADK planner's does: its agent is named `AGENT_NAME`, its
spans name the session as their conversation, and they carry message content
when ADK's would (`google.adk.telemetry.TelemetryConfig`, which reads
`OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT`).

The node reports to the seam's `harness.planner_run.PlannerRun` as the ADK
planner's callbacks do: each tool's answer, each tool that raised, a call the
turn budget closed (`PlannerRun.model_answered`), and before each request
whether the run asks or stops instead (`PlannerRun.next_request`). A directive
is answered with the model turn `directive_turn` builds, which the ADK planner's
callbacks answer it with too.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Sequence
from contextlib import aclosing
from dataclasses import dataclass
from typing import Any

from common.telemetry import current_trace_id
from executor.classify import internal_fault
from google.adk import Context, Event
from google.adk.agents.invocation_context import LlmCallsLimitExceededError
from google.adk.agents.run_config import RunConfig
from google.adk.flows.llm_flows import functions as adk_functions
from google.adk.telemetry import TelemetryConfig
from google.adk.workflow import BaseNode, node
from google.genai import Client, types
from planner.agent import AGENT_NAME, BRIEF_STATE_KEY, session_steps
from planner.pydantic_agent import (
    PLANNER_PROFILE,
    PlannerDeps,
    PlannerResponse,
    PlannerToolAnswer,
    PydanticPlanner,
    UnsentQuestionPartError,
    planner_agent,
    run_planner,
)
from planner.tool_spec import ToolSpec

from harness.planner_events import answered_every_call, model_response_event, tool_answers_event
from harness.planner_model import ResponseMetadata, Usage, planner_model
from harness.planner_run import Ask, Directive, PlannerRun, Stop
from harness.prefix_cache import PrefixCache
from harness.sentences import SeamCode, refusal_sentence
from harness.turn_budget import CLOSED_BY_KEY, TurnBudget
from harness.turn_events import PLANNER_ASK, turn_question
from harness.usage import log_usage

logger = logging.getLogger(__name__)

# Gemini requires a `thought_signature` on a generation's first `functionCall`
# part and answers a request replaying one without it with a 400. This is
# the documented placeholder that makes the API skip validation for a call
# injected outside a model turn.
SKIP_THOUGHT_SIGNATURE = b"skip_thought_signature_validator"


@dataclass(frozen=True)
class _Recorded:
    """The metadata the ADK planner's event records beside a response's parts, and
    whether that response is the aggregate of a stream (`partial` False) or a turn
    put in the model's place (`partial` None)."""

    usage: Usage | None = None
    finish_reason: types.FinishReason | None = None
    model_version: str | None = None
    custom_metadata: dict[str, Any] | None = None
    partial: bool | None = None
    error_code: str | None = None
    error_message: str | None = None


# A model turn put in place of a request or of a closed call's response. ADK
# stores the callback's response, which carries only its content.
_PUT_IN_PLACE = _Recorded()


def pydantic_planner(specs: Sequence[ToolSpec], *, instruction: str) -> PydanticPlanner:
    """The Pydantic AI planner on `specs` and `instruction`, which the ADK planner
    serves too, named as the ADK planner (`planner.agent.AGENT_NAME`). It has no
    model of its own, since each run names one (`planner_node`). Built once, it
    serves every run. Its spans carry message content when ADK's telemetry
    settings say the ADK planner's do (`google.adk.telemetry.TelemetryConfig`),
    read once, here."""
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
) -> BaseNode:
    """One run of the Pydantic AI planner, as the node `ctx.run_node` runs.

    Args:
        run: The run's bookkeeping, told what the run does and asked what
            replaces each request.
        planner: The Pydantic AI planner (`pydantic_planner`).
        model_name: The Gemini model, as google-genai names it.
        client: The google-genai client the model's calls go through.
        budget: The turn's budget, or None to run every call unbounded.
        prefix_cache: The process's shared cache of static prompt prefixes,
            or None to send every request whole.

    Returns:
        A node named `planner.agent.AGENT_NAME` that runs the planner once on
        the session's brief (`planner.agent.BRIEF_STATE_KEY`), after the
        turn's question, and yields the ADK planner's events for the run.
        The run makes at most `RunConfig.max_llm_calls` model calls of its
        own, when that is above zero, and the call past it raises
        `LlmCallsLimitExceededError`, the error the ADK planner raises. ADK
        counts every model call of the invocation against the cap, the
        orchestrator's and earlier planner runs' included, so a turn can make
        more calls in total on this planner before the cap trips (`_LlmCalls`).
        A question holding a part this planner does not send
        (`planner.pydantic_agent.UnsentQuestionPartError`) ends the run before
        any request on an `internal_error` refusal (`run.refused`).
    """

    async def dfo_planner(ctx: Context, node_input: Any) -> AsyncIterator[Event]:
        del node_input
        ctx.event_author = AGENT_NAME
        recorded = _PUT_IN_PLACE
        calls = _LlmCalls(ctx.run_config)

        def before_request() -> types.Content | None:
            nonlocal recorded
            turn = directive_turn(run.next_request())
            if turn is not None:
                recorded = _PUT_IN_PLACE
            else:
                calls.count()
            return turn

        def after_response(
            closed_by: SeamCode | None, metadata: ResponseMetadata
        ) -> types.Content | None:
            nonlocal recorded
            if closed_by is None:
                if metadata.usage is not None:
                    log_usage(AGENT_NAME, metadata.usage)
                recorded = _Recorded(
                    usage=metadata.usage,
                    finish_reason=metadata.finish_reason,
                    model_version=metadata.model_version,
                    partial=False,
                    error_code=metadata.error_code,
                    error_message=metadata.error_message,
                )
                return None
            turn = directive_turn(run.model_answered(closed_by))
            recorded = (
                _PUT_IN_PLACE
                if turn is not None
                else _Recorded(custom_metadata={CLOSED_BY_KEY: closed_by.value})
            )
            return turn

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
            profile=PLANNER_PROFILE,
        )
        deps = PlannerDeps(
            state=ctx.session.state,
            earlier=tuple(session_steps(ctx.session.events)),
            tool_answered=run.tool_answered,
            tool_failed=tool_failed,
        )
        response: PlannerResponse | None = None
        held: list[PlannerToolAnswer] = []
        items = run_planner(
            planner,
            deps,
            model=model,
            question=turn_question(ctx),
            brief=ctx.session.state[BRIEF_STATE_KEY],
            conversation_id=ctx.session.id,
        )
        # Closed here, in the node's task, when ADK's node runner closes the
        # node at a yield on a Stop, so the run's spans and context variables
        # end in the task that opened them.
        try:
            async with aclosing(items):
                async for item in items:
                    if isinstance(item, PlannerResponse):
                        response, held = item, []
                        # ADK stores no event for a response with no parts,
                        # unless the response is an error, a blocked prompt's.
                        if item.parts or recorded.error_code is not None:
                            yield model_response_event(
                                item,
                                usage=recorded.usage,
                                finish_reason=recorded.finish_reason,
                                model_version=recorded.model_version,
                                custom_metadata=recorded.custom_metadata,
                                partial=recorded.partial,
                                error_code=recorded.error_code,
                                error_message=recorded.error_message,
                            )
                        continue
                    assert response is not None, "a call's answer follows its response"
                    held.append(item)
                    if answered_every_call(response, held):
                        yield tool_answers_event(response, held, ended=_ended(held))
                        held = []
        except UnsentQuestionPartError as exc:
            # Refused before any request, so nothing is held. The run ends on
            # the refusal, which the seam answers the entry with; an exception
            # here would fail the whole workflow, since ADK fails the root on
            # a child node's error. The ADK planner sends the part.
            logger.error(
                "planner question refused session=%s invocation=%s: %s",
                ctx.session.id,
                ctx.invocation_id,
                exc,
            )
            run.refused = internal_fault(str(exc))
            return
        except Exception:
            # The ADK planner keeps the answers of a response's finished calls
            # when another of its calls fails. A cancellation stores nothing of
            # the response, so it is not caught and the held answers go.
            if response is not None and held:
                yield tool_answers_event(response, held, ended=_ended(held))
            raise
        # A run `run_planner` ends without raising on a response Pydantic AI
        # refuses may leave calls of the last response unanswered. The answers
        # of its finished calls are kept, as an exception keeps them.
        if response is not None and held:
            yield tool_answers_event(response, held, ended=_ended(held))

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


class _LlmCalls:
    """The model calls one planner run has made, against `RunConfig.max_llm_calls`.

    ADK counts every `LlmAgent` call of the invocation on one counter
    (`InvocationContext.increment_llm_call_count`), the orchestrator's and
    every planner run's, and raises `LlmCallsLimitExceededError` on the call
    past the cap. This counts one run's calls alone, because ADK keeps the
    invocation's count where no public API reads or adds to it. A turn on the
    Pydantic AI planner can therefore make more model calls in total before
    the cap trips: up to the cap in each planner run, beside the
    orchestrator's own. A turn put in the model's place is not a call, as in
    ADK.
    """

    def __init__(self, config: RunConfig | None) -> None:
        self._cap = config.max_llm_calls if config is not None else 0
        self._made = 0

    def count(self) -> None:
        """Count one call, raising when it passes the cap."""
        self._made += 1
        if 0 < self._cap < self._made:
            raise LlmCallsLimitExceededError(
                f"Max number of llm calls limit of `{self._cap}` exceeded"
            )


def _ended(answers: Sequence[PlannerToolAnswer]) -> bool:
    """Whether a call of `answers` ended the run."""
    return any(answer.ended for answer in answers)


__all__ = ["SKIP_THOUGHT_SIGNATURE", "directive_turn", "planner_node", "pydantic_planner"]
