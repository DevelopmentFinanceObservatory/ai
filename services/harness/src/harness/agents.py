"""Builds the planner with translation attached.

The planner and translation live in their own modules and know nothing about
each other's construction: `planner.agent` takes a translation to run and the
tool surface it reads skills through, `translator.tool` takes the same surface
to discover on and compile against. This is where those two halves meet,
which keeps the dependency arrow pointing one way: `harness` may import both
packages, and neither imports `harness`.

`propose_plan` runs translation on every plan it records, so the plan travels
through session state alone: `propose_plan` records it, and translation reads
it back and answers with one typed verdict, which `propose_plan` returns under
`translation`. No model runs inside it.
"""

from __future__ import annotations

from compiler.surface import McpSurface, ToolSurface
from google.adk.agents import LlmAgent
from planner.agent import create_planner_agent
from planner.settings import AgentSettings
from translator.tool import build_translate

from harness.conversation import current_conversation_caller
from harness.mcp_deadlines import McpTimeoutPolicy
from harness.prefix_cache import PrefixCache
from harness.settings import Settings


def build_tool_surface(*, settings: Settings) -> McpSurface:
    """Build the surface translation discovers on, the compiler checks against,
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
    prefix_cache: PrefixCache | None = None,
) -> LlmAgent:
    """Build the planner agent with translation attached to `propose_plan`.

    Args:
        settings: The harness service's settings, read for the MCP URLs the
            default surface covers.
        agent_settings: The model the planner runs on. Defaults to reading the
            environment, the same way the deployed service does.
        surface: The tool surface translation discovers on and compiles
            against, and the planner's `get_skill` and `get_skill_file` call
            data-mcp through. Tests pin one here so the planner builds without
            a reachable MCP server.
        question_fixed: Passed straight through to `create_planner_agent`.
            `harness.planner_seam.PlannerSeam` builds its planner with this
            set, since the graph derives and narrates a turn's question before
            the planner ever runs. `evals/agents/harness_pair_v1` builds its
            own planner through `create_planner_agent` directly, not through this
            function, so the default here only has to keep this function's
            other callers byte-identical.
        prefix_cache: Passed straight through to `create_planner_agent`.

    Returns:
        The planner `LlmAgent`, whose `propose_plan` translates each plan it
        records. It is a `chat` agent, so it can serve as a `Runner`'s root.
    """
    resolved_agent_settings = agent_settings if agent_settings is not None else AgentSettings()
    resolved_surface = surface if surface is not None else build_tool_surface(settings=settings)
    return create_planner_agent(
        model_name=resolved_agent_settings.planner_model_name,
        translate=build_translate(surface=resolved_surface),
        surface=resolved_surface,
        question_fixed=question_fixed,
        prefix_cache=prefix_cache,
    )
