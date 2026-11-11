"""The tool surface the planner's translation discovers on and compiles
against, and the planner reads skills through (`build_tool_surface`).

`harness.planner_seam.PlannerSeam` builds the planner on it with translation
attached to `propose_plan`, so the planner and translation meet there.
`planner.specs.planner_specs` takes a translation to run and the surface it
reads skills through, `translator.tool` takes the same surface to discover on
and compile against, and neither imports `harness`.
"""

from __future__ import annotations

from compiler.surface import McpSurface

from harness.conversation import current_conversation_caller
from harness.mcp_deadlines import McpTimeoutPolicy
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
