"""The Gemini model the orchestrator runs on, kept to the turn's budget, streaming.

`orchestrator_model` builds Pydantic AI's `GoogleModel` over a google-genai
client whose streamed call is `harness.gemini_wire.GeminiStream`, as the
planner's is (`planner.model`): the request put in one fixed form, sent
against the shared prefix cache under the turn's budget, its usage and how it
ended recorded on the call's span, and a call Gemini ended on a function call
it could not parse sent once more. Unlike the planner's, the stream passes
answer text on as it arrives, since the user reads it as it streams.
"""

from __future__ import annotations

from google.genai import Client
from harness.gemini_agent import GEMINI_PROFILE
from harness.gemini_wire import GeminiStream, wrapped_client
from harness.prefix_cache import PrefixCache
from harness.turn_budget import TurnBudget
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.providers.google import GoogleProvider


def orchestrator_model(
    model_name: str,
    *,
    client: Client,
    budget: TurnBudget | None,
    prefix_cache: PrefixCache | None = None,
) -> GoogleModel:
    """The `GoogleModel` one orchestrator run calls.

    Args:
        model_name: The Gemini model, as google-genai names it.
        client: The google-genai client the calls go through. It is left
            unchanged. The model calls a wrapper over its streamed call.
        budget: The turn's budget, or None to run every call unbounded.
        prefix_cache: The process's shared cache of static prompt prefixes,
            or None to send every request whole.

    Returns:
        A `GoogleModel` on `harness.gemini_agent.GEMINI_PROFILE`, whose
        streamed calls go through a `GeminiStream` that passes text on.
    """
    stream = GeminiStream(
        client=client,
        budget=budget,
        prefix_cache=prefix_cache,
        before_request=None,
        after_response=None,
        passes_text=True,
    )
    provider = GoogleProvider(client=wrapped_client(client, stream))
    return GoogleModel(model_name, provider=provider, profile=GEMINI_PROFILE)


__all__ = ["orchestrator_model"]
