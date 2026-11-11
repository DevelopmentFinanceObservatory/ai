"""The planner agent (`create_planner_agent`) and the ADK tool that serves a planner tool.

Each planner tool is a `planner.tool_spec.ToolSpec` (`planner.specs`), whose
body runs `planner.tools`' or `planner.skills`' code over a
`planner.call.PlannerCall` and no framework. Here `SpecTool` serves a spec to
ADK, and `planner.session.SessionCall` is the `PlannerCall` an ADK tool call
is. A `propose_plan` whose translation compiles declares the plan ready for
the user's answer in code (`planner.tools.declare_ready`), and the run ends on it
(`skip_summarization`) with no model call after. `end_run_when_ready` does the
same for an answer the harness builds itself. Declaring a plan ready approves
nothing. Whatever runs this agent withholds execution until that answer
arrives, which for the served path is the harness's own graph, keyed on the
plan's revision, so no ADK tool confirmation has to cross this agent's own
nested `Runner`.

Built with a tool surface, the planner also carries `planner.skills`'
`get_skill`, and its instruction carries `_SKILLS`, the paragraph naming it.
Built without one, it carries neither, so the
standalone eval agents run without a live server.

The planner settles methodology and states what evidence an answer rests on,
then stops. A proposed plan is a `DRAFT` carrying `decisions` and
`evidence_requirements`; finding the data, proposing the operations and
compiling the workflow belong to `translator.tool`'s `build_translate`, the
translation `propose_plan` runs on every plan it records before the user ever
sees it -- `create_planner_agent` takes that function, and `harness.agents`
builds it. So no tool name, column or operation appears here,
`PENDING_APPROVAL` is a status only translation writes, and `declare_ready`
requires a plan that has been through that step. The checks a proposal passes
before it is recorded are `submission.service`'s, which `propose_plan` runs
over its call's session (`planner.call.CallSession` over `SessionCall`).

Every planner tool answers with `planner.tool_results`' typed shape, `ok` or
`problem` with a stable `code` and, where the next step is known, an
`action`. When translation cannot carry a plan out, `propose_plan`'s
`translation` is the `Referral` itself: this agent's instruction reads it
straight off that answer, with nothing here to fetch it back a second time.
"""

from __future__ import annotations

import functools
import logging
import os
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Literal, override

from compiler.surface import ToolSurface
from google.adk.agents import LlmAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.models import BaseLlm, Gemini
from google.adk.models.llm_request import LlmRequest
from google.adk.planners import BuiltInPlanner
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext
from google.genai import Client, types
from harness.turn_events import after_turn_question
from orchestrator.agent import _retrying
from translator.tool import Translate

from planner.instruction import planner_instruction
from planner.session import SessionCall
from planner.settings import AGENT_NAME, AGENT_THINKING_CONFIG, drop_thought_parts
from planner.specs import planner_specs
from planner.tool_spec import ToolSpec, function_declaration, run_tool, serve_tool
from planner.tools import EndRun, ready_and_end

if TYPE_CHECKING:
    from harness.prefix_cache import PrefixCache

logger = logging.getLogger(__name__)


def _end_run(tool_context: ToolContext) -> EndRun:
    """The `EndRun` for the tool `tool_context` serves: ADK ends the run on its
    response (`skip_summarization`), with no model call after."""

    def end_run() -> None:
        tool_context.actions.skip_summarization = True

    return end_run


def end_run_when_ready(answer: dict[str, object], context: ToolContext) -> dict[str, object]:
    """`declare_ready` over the call `context` serves (`SessionCall.of`), ending
    the run once it is ready (`ready_and_end`)."""
    return ready_and_end(answer, SessionCall.of(context), _end_run(context))


def _env_flag(name: str) -> bool | None:
    """The environment variable `name` read as google-genai reads a flag.

    `true` or `1`, in any case, is true. Any other value is false, and an unset
    variable gives `None`.
    """
    value = os.environ.get(name)
    return None if value is None else value.lower() in ("true", "1")


def refuse_vertex_ai(
    model_name: str,
    client: Client | None = None,
    *,
    runner: Literal["adk", "pydantic_ai"] = "adk",
) -> None:
    """Refuse to build the planner when its model would call Vertex AI.

    The planner's declarations are the Gemini developer API's form
    (`planner.tool_spec.function_declaration`). Vertex AI rejects their
    optional `anyOf` schemas, which ADK rewrites only for its own function
    tools. `create_planner_agent` calls this once, when the planner's model
    is ADK's `Gemini`, with that model's name and the client it calls through.
    Any other model calls through no google-genai client.
    `harness.planner_seam.PlannerSeam` calls it for the Pydantic AI planner's
    model and client.

    A given client decides by its `vertexai`. With none, each runner builds
    its own, and the decision is the one that client makes:

    - The ADK planner's `Gemini` builds its client for `model_name` with no
      client options (`Gemini.api_client`). A name of the form `projects/...`
      selects Vertex AI. Otherwise google-genai takes the backend from the
      environment: `GOOGLE_GENAI_USE_ENTERPRISE` when set, else
      `GOOGLE_GENAI_USE_VERTEXAI`, each true when `true` or `1`.
    - The Pydantic AI planner builds a plain `google.genai.Client()`, whose
      backend the environment alone decides, by the same rule.

    The rule is read here without building a client, since building one
    without an API key or credentials raises. On the Pydantic AI planner a
    `projects/...` name is refused whatever the client: it is a Vertex AI
    resource path, and that planner runs on the developer API, which serves
    no model by that name.

    Args:
        model_name: The Gemini model's name, as the client is sent it.
        client: The client the model calls through, or None for the one the
            runner builds.
        runner: The planner the model runs on.

    Raises:
        RuntimeError: The client uses Vertex AI, or the Pydantic AI planner is
            given a Vertex AI resource path.
    """
    if runner == "pydantic_ai" and model_name.startswith("projects/"):
        raise RuntimeError(
            f"{model_name!r} is a Vertex AI resource path, and the Pydantic AI planner "
            "runs on the Gemini developer API, which serves no model by that name. "
            "Name the model as the developer API does (models/... or its bare name), "
            "or set PLANNER_RUNNER=adk."
        )
    if client is not None:
        vertexai = client.vertexai
    elif runner == "adk" and model_name.startswith("projects/"):
        vertexai = True
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


class SpecTool(BaseTool):
    """A `planner.tool_spec.ToolSpec` served to an ADK `LlmAgent`.

    The model is shown `function_declaration(spec)`, and a call runs the
    spec's body over the call's `SessionCall` (`run_tool`), ending the run through
    `skip_summarization`. `_get_declaration` is the hook `BaseTool` reads the
    declaration through. Its `process_llm_request` adds the declaration to the
    request's one `types.Tool` of function declarations, creating it when the
    request has none, and registers the tool under its name.

    The declaration is the Gemini developer API's form, which Vertex AI
    rejects (`refuse_vertex_ai`).
    """

    def __init__(self, spec: ToolSpec) -> None:
        super().__init__(name=spec.name, description=spec.description)
        self.spec = spec

    @functools.cached_property
    def declaration(self) -> types.FunctionDeclaration:
        """The declaration the model is shown for the spec, built once."""
        return function_declaration(self.spec)

    @override
    def _get_declaration(self) -> types.FunctionDeclaration:
        # Return a copy, since the request holds what this returns and later
        # request processing may edit it in place.
        return self.declaration.model_copy(deep=True)

    def invoke(self, args: dict[str, Any], tool_context: ToolContext) -> object:
        """The spec's answer to a call with `args`, awaitable when its body is async."""
        return run_tool(
            self.spec, args, call=SessionCall.of(tool_context), end_run=_end_run(tool_context)
        )

    @override
    async def run_async(self, *, args: dict[str, Any], tool_context: ToolContext) -> Any:
        """The spec's answer to a call with `args` (`planner.tool_spec.serve_tool`)."""
        return await serve_tool(
            self.spec, args, call=SessionCall.of(tool_context), end_run=_end_run(tool_context)
        )


# The brief for the planner's current run, which `prepend_brief` shows the model.
# `harness.planner_seam.PlannerSeam` renders it before every run, a resume
# included, so a `temp:` key that is never persisted is enough.
BRIEF_STATE_KEY = "temp:planner_brief"


def prepend_brief(callback_context: CallbackContext, llm_request: LlmRequest) -> None:
    """Show the model the run's brief (`BRIEF_STATE_KEY`) right after the turn's question.

    Every request of the run carries it, the first request after a resume
    included, since ADK hands a node its input only on a live run. The brief
    sits ahead of the run's own calls and responses, so each request extends
    the one before it. Adds nothing when state holds no brief.
    """
    brief = callback_context.state.get(BRIEF_STATE_KEY)
    if not brief:
        return
    llm_request.contents.insert(
        after_turn_question(callback_context, llm_request.contents),
        types.Content(role="user", parts=[types.Part(text=brief)]),
    )


def create_planner_agent(
    *,
    model_name: str,
    translate: Translate | None = None,
    surface: ToolSurface | None = None,
    question_fixed: bool = False,
    reads_user_words: bool = True,
    prefix_cache: PrefixCache | None = None,
    specs: Sequence[ToolSpec] | None = None,
    client: Client | None = None,
    model: BaseLlm | None = None,
) -> LlmAgent:
    """Build the planner `LlmAgent`.

    Args:
        model_name: The model this agent runs on.
        translate: The translation (`translator.tool.build_translate`)
            `propose_plan` runs on every plan it records, which is how a
            draft plan gets translated and compiled. It runs inside
            `propose_plan`, so the model spends no generation on it. Omitted, the
            planner is the standalone agent its own eval set scores --
            which is also what keeps that set from needing a live tool
            surface.
        surface: The tool surface `get_skill` calls
            data-mcp through, and `propose_plan` reads a pinned dataset's
            first year through. Omitted, the planner has no skill tool,
            its instruction does not name it, and `propose_plan` reads
            nothing.
        question_fixed: Whether the question this planner drafts for is fixed
            for the entry it is called in. `harness.planner_seam` sets
            this when it builds the planner, since the orchestrator states
            the plan's question in its brief before the planner ever runs,
            and a restart would silently swap the question out from under
            that brief (see `submission.service._resolve_intent`). False, the default, is the standalone
            planner's and `harness_pair_v1`'s own behaviour: a restart begins
            a fresh plan.
        reads_user_words: Whether the planner reads the user's own words in
            every entry, so `propose_plan`'s `across_years` may name a step
            in any of them. True, the default, is the standalone planner's
            and `harness_pair_v1`'s, which plan the user's question
            themselves. `harness.planner_seam` sets it false, since there the
            orchestrator reads the user's words and marks its asks, and the
            planner reads them only in an entry the user's amendment opens
            (`planner.plan.amendment_entry`).
        prefix_cache: The process's shared cache of static prompt prefixes.
            Given, and `model_name` naming a Gemini model, every call reads
            the instruction and tool declarations from it. Either way a
            Gemini model retries a transient error (408, 429, 5xx) up to
            three attempts, as the orchestrator's does
            (`orchestrator.agent._retrying`).
        specs: The tools the agent serves. None gives `planner_specs` over the
            arguments above.
        client: The google-genai client a Gemini model calls through, used as
            it is, HTTP options included. None gives one ADK builds from the
            environment.
        model: A model that replaces the one `model_name` names, used as given:
            it calls through no client built here, so it is not checked for
            Vertex AI, and takes no `client`.

    Returns:
        The configured `LlmAgent`.

    Raises:
        RuntimeError: The model `model_name` names is ADK's `Gemini` and its
            google-genai client would use Vertex AI (`refuse_vertex_ai`).
        ValueError: A client is given with `model`, or for a model other than
            ADK's `Gemini`, which alone takes one.
    """
    if model is not None:
        if client is not None:
            raise ValueError("a model and a client replace the same thing; give one")
        llm: str | BaseLlm = model
    else:
        llm = _retrying(model_name, prefix_cache=prefix_cache)
        if isinstance(llm, Gemini):
            refuse_vertex_ai(llm.model, client)
            if client is not None:
                llm = llm.model_copy(update={"client": client})
        elif client is not None:
            raise ValueError(f"a client runs a Gemini planner model only, not {model_name!r}")
    return LlmAgent(
        name=AGENT_NAME,
        model=llm,
        instruction=planner_instruction(
            has_translator=translate is not None,
            has_skills=surface is not None,
            question_fixed=question_fixed,
        ),
        tools=[
            SpecTool(spec)
            for spec in (
                specs
                if specs is not None
                else planner_specs(
                    translate=translate,
                    surface=surface,
                    question_fixed=question_fixed,
                    reads_user_words=reads_user_words,
                )
            )
        ],
        # Run as a workflow node (`harness.planner_seam.PlannerSeam._propose`'s
        # `ctx.run_node`), ADK runs an agent with no explicit `mode` as
        # `single_turn`. `'none'` would build every request from
        # `_get_current_turn_contents`, which slices the session from the
        # latest turn-starting event. `'default'` uses `_get_contents`, which
        # reads the turn's question and every event on this run's node path,
        # so `prepend_brief` finds the question to follow.
        include_contents="default",
        # A stream cut off mid-response resumes from a continuation token, one
        # extra hop instead of a full retry. google-genai defaults this to
        # true; it is set so a change of that default cannot turn the resume
        # off.
        generate_content_config=types.GenerateContentConfig(automatic_continuation=True),
        # `AGENT_THINKING_CONFIG` (see `planner.settings`) pins every
        # generation to LOW thinking.
        planner=BuiltInPlanner(thinking_config=AGENT_THINKING_CONFIG),
        # Drops this generation's thought-summary parts once the trace has
        # them, so `include_contents="default"` above never replays one into
        # a later request of the same turn. See `drop_thought_parts`.
        after_model_callback=drop_thought_parts,
        before_model_callback=prepend_brief,
    )
