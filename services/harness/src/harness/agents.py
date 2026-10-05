"""Builds the planner with its translator sub-agent attached.

The two agents live in their own modules and know nothing about each other's
construction: `planner.agent` takes a translator to attach and the tool surface
it reads skills through, `translator.agent` takes the same surface to discover
on and compile against. This is where those two halves meet, which keeps the
dependency arrow pointing one way: `harness` may import both agent packages,
and neither imports `harness`.

Attaching the translator with `mode="single_turn"` is what makes it a tool on
the planner rather than a peer the harness would have to schedule: ADK wraps a
`single_turn` sub-agent as one tool declaration on its parent, so the planner
calls it inside its own turn and the plan travels between them through session
state alone: the planner names what it wants translated, the translator reads
the recorded plan itself, and no copy of the plan crosses the seam as an
argument. The translator's own turns stay out of the planner's context, and
what returns is a single function response.

What keeps them out is the branch. `harness.planner_seam.
PlannerSeam._propose` stamps the planner's own run with `_PLANNER_BRANCH`, so
every planner event carries that branch and the translator, running as its
sub-agent, has its own events published on a sub-branch of it
(`google.adk.tools.agent_tool._SingleTurnAgentTool.run_async`).
`google.adk.flows.llm_flows.contents._is_event_belongs_to_branch` keeps an
event when its branch equals the planner's or is an ancestor of it, so the
planner's own events pass on equality and the translator's sub-branch events
fail both tests and drop. The two agents share the same `isolation_scope`,
since `prepare_llm_agent_context` (`google.adk.workflow._llm_agent_wrapper`)
hands a `single_turn` agent its parent's scope, so that value cannot separate
their events. Without the branch stamp the planner runs unbranched, an
unbranched event belongs to every branch, and
`google.adk.flows.llm_flows._fencing._present_other_agent_message`
relays the translator's `translate` calls and results into the planner's
request as fenced user text: 24,683 chars of a 56,407-char history on the
reference trace, against 417 chars for the typed function responses the
`dfo_translator` route returns.
"""

from __future__ import annotations

from compiler.surface import McpSurface, ToolSurface
from google.adk.agents import LlmAgent
from planner.agent import create_planner_agent
from planner.settings import AgentSettings
from translator.agent import create_translator_agent

from harness.conversation import current_conversation_caller
from harness.mcp_deadlines import McpTimeoutPolicy
from harness.settings import Settings


def build_tool_surface(*, settings: Settings) -> McpSurface:
    """Build the surface the translator discovers on, the compiler checks against,
    and the planner reads skills through.

    Covers data-mcp and analysis-mcp, over the same URLs `harness.mcp_client`
    calls tools on: a plan retrieves before it computes, so it needs both, and
    reading the surface the way the executor reaches it stops the two
    compiling and running against different schemas. Constructing this opens
    no connection -- `McpSurface` reads each server when asked.

    Deadlines come from `harness.mcp_deadlines.McpTimeoutPolicy` -- the same
    policy `harness.mcp_client`'s `McpResolver` and `McpExecutor` build their
    own clients from, so the two paths can't drift apart. A surface built any
    other way -- `evals/agents/harness_translator_v1/discovery_stub.py`,
    `tests/test_executor_integration.py` -- gets the 60 s default for every
    tool, `get_data` included, which only matters against a transport slower
    than an in-memory one.

    Calls carry the current conversation's caller, as `harness.mcp_client`'s
    do, so a verb the executor runs names the same conversation as the
    `get_data` before it.
    """
    policy = McpTimeoutPolicy(settings=settings)
    return McpSurface(
        sources={"data-mcp": settings.data_mcp_url, "analysis-mcp": settings.analysis_mcp_url},
        call_timeouts=policy.call_timeouts,
        default_call_timeout_s=policy.default_call_timeout_s,
        list_timeout_s=policy.list_timeout_s,
        init_timeout_s=policy.init_timeout_s,
        caller=current_conversation_caller,
    )


def create_planner_with_translator(
    *,
    settings: Settings,
    agent_settings: AgentSettings | None = None,
    surface: ToolSurface | None = None,
    question_fixed: bool = False,
) -> LlmAgent:
    """Build the planner agent with the translator attached as a sub-agent.

    Args:
        settings: The harness service's settings, read for the MCP URLs the
            default surface covers.
        agent_settings: The models both agents run on. Defaults to reading the
            environment, the same way the deployed service does.
        surface: The tool surface the translator discovers on and compiles
            against, and the planner's `get_skill` and `get_skill_file` call
            data-mcp through. Tests pin one here so the pair builds without a
            reachable MCP server.
        question_fixed: Passed straight through to `create_planner_agent`.
            `harness.planner_seam.PlannerSeam` builds its pair with this set,
            since the graph derives and narrates a turn's question before the
            planner ever runs. `evals/agents/harness_pair_v1` builds its own
            pair through `create_planner_agent` directly, not through this
            function, so the default here only has to keep this function's
            other callers byte-identical.

    Returns:
        The planner `LlmAgent`, with the translator as its only sub-agent. It
        is a `chat` agent, so it can serve as a `Runner`'s root.
    """
    resolved_agent_settings = agent_settings if agent_settings is not None else AgentSettings()
    resolved_surface = surface if surface is not None else build_tool_surface(settings=settings)
    translator = create_translator_agent(
        model_name=resolved_agent_settings.translator_model_name,
        surface=resolved_surface,
        mode="single_turn",
    )
    return create_planner_agent(
        model_name=resolved_agent_settings.planner_model_name,
        translator=translator,
        surface=resolved_surface,
        question_fixed=question_fixed,
    )
