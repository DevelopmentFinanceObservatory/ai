"""The Gemini model the planner runs on, kept to the turn's budget.

`planner_model` builds Pydantic AI's `GoogleModel` over a google-genai client
whose streamed call is `harness.gemini_wire.GeminiStream`. That call puts the
request in one fixed form, sends it against the shared prefix cache under the
turn's budget, sends a malformed function call again once, and records its
usage on the call's span. The planner adds its `before_request` and
`after_response` hooks.
`refuse_vertex_ai` refuses a model the planner's declarations cannot run on.
"""

from __future__ import annotations

import os

from google.genai import Client
from harness.gemini_agent import GEMINI_PROFILE
from harness.gemini_wire import AfterResponse, BeforeRequest, GeminiStream, wrapped_client
from harness.prefix_cache import PrefixCache
from harness.turn_budget import TurnBudget
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.providers.google import GoogleProvider


def planner_model(
    model_name: str,
    *,
    client: Client,
    budget: TurnBudget | None,
    prefix_cache: PrefixCache | None = None,
    before_request: BeforeRequest | None = None,
    after_response: AfterResponse | None = None,
) -> GoogleModel:
    """The `GoogleModel` the planner runs on.

    Args:
        model_name: The Gemini model, as google-genai names it.
        client: The google-genai client the model's calls go through. It is
            left unchanged. The model calls a wrapper over its streamed call.
        budget: The turn's budget, or None to run every call unbounded.
        prefix_cache: The process's shared cache of static prompt prefixes,
            or None to send every request whole.
        before_request: Run before each request (`BeforeRequest`).
        after_response: Run on each response the model gave or a bound
            closed (`AfterResponse`).

    Returns:
        A `GoogleModel` on `harness.gemini_agent.GEMINI_PROFILE`, the profile
        `harness.gemini_wire.in_canonical_form` expects, whose streamed calls
        go through `GeminiStream`.
    """
    stream = GeminiStream(
        client=client,
        budget=budget,
        prefix_cache=prefix_cache,
        before_request=before_request,
        after_response=after_response,
    )
    provider = GoogleProvider(client=wrapped_client(client, stream))
    return GoogleModel(model_name, provider=provider, profile=GEMINI_PROFILE)


def refuse_vertex_ai(model_name: str, client: Client | None = None) -> None:
    """Refuse to build the planner when its model would call Vertex AI.

    The planner's declarations are the Gemini developer API's form
    (`planner.tool_spec.function_declaration`), and Vertex AI rejects their
    optional `anyOf` schemas. `harness.planner_adapter.PlannerAdapter` calls this
    with the planner's model name and the client it calls through.

    A `projects/...` name is refused whatever the client, since it is a Vertex
    AI resource path and the planner runs on the developer API, which serves
    no model by that name. A given client decides by its `vertexai`. With none,
    the planner builds a plain `google.genai.Client()`, whose backend the
    environment decides: `GOOGLE_GENAI_USE_ENTERPRISE` when set, else
    `GOOGLE_GENAI_USE_VERTEXAI`, each true when `true` or `1`. The rule is read
    here without building a client, since building one without an API key or
    credentials raises.

    Args:
        model_name: The Gemini model's name, as the client is sent it.
        client: The client the model calls through, or None for the one the
            planner builds.

    Raises:
        RuntimeError: The client uses Vertex AI, or the name is a Vertex AI
            resource path.
    """
    if model_name.startswith("projects/"):
        raise RuntimeError(
            f"{model_name!r} is a Vertex AI resource path, and the planner runs on the "
            "Gemini developer API, which serves no model by that name. Name the model "
            "as the developer API does (models/... or its bare name)."
        )
    if client is not None:
        vertexai = client.vertexai
    else:
        enterprise = _env_flag("GOOGLE_GENAI_USE_ENTERPRISE")
        vertexai = enterprise if enterprise is not None else _env_flag("GOOGLE_GENAI_USE_VERTEXAI")
    if vertexai:
        raise RuntimeError(
            "The planner's google-genai client uses Vertex AI, which rejects the "
            "planner's tool declarations. Run the planner on a Gemini developer API "
            "key (GOOGLE_API_KEY): give a client built without vertexai, or name the "
            "model without a projects/ path and unset GOOGLE_GENAI_USE_ENTERPRISE and "
            "GOOGLE_GENAI_USE_VERTEXAI."
        )


def _env_flag(name: str) -> bool | None:
    """The environment variable `name` read as google-genai reads a flag.

    `true` or `1`, in any case, is true. Any other value is false, and an unset
    variable gives `None`.
    """
    value = os.environ.get(name)
    return None if value is None else value.lower() in ("true", "1")


__all__ = ["planner_model", "refuse_vertex_ai"]
