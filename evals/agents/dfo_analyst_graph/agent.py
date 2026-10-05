"""The whole dfo-ai system as one ADK app, for `just adk-web`.

`harness.studio.StudioRunner` builds the real turn workflow with the same
wiring `harness.server.create_app` uses; this module exports the `App` it
built, so the ADK web UI shows a real turn's events: the converse, probe,
planner, execute, compose and narrate nodes, the planner's and translator's
model calls, and every MCP tool call under them. The MCP mounts belong to the
harness server (`just dev`), reached over the loopback URLs `Settings`
derives from `PORT`.

The directory is named after the graph (`harness.graph.AGENT_NAME`): `adk web`
registers an app under its directory name, and a `Runner` whose root agent
came from a directory of another name logs an app-name mismatch warning.

`adk web` builds its own `Runner` from the exported `app`, so nothing set on
the `StudioRunner`'s runner reaches it. The conversation identity therefore
travels on the `App` as a plugin (`ConversationIdentity`): it sets
`harness.conversation.current_conversation_id` and `current_conversation_key`
from the invocation's user and session before every run, including the resume
after the plan-approval pause. `McpExecutor` keys its per-conversation
presentation-mcp client on the first and the recall seam keys on the second,
so each web session gets its own client and recall history.

The recipe passes `--session_service_uri memory://`, which is the session
service `adk web` uses; the `StudioRunner` is given an in-memory one only
because its constructor requires one.
"""

from __future__ import annotations

from typing import override

from common.auth import apply_gate
from data_mcp.server import mcp as data_mcp
from google.adk.agents.invocation_context import InvocationContext
from google.adk.plugins.base_plugin import BasePlugin
from google.adk.sessions import InMemorySessionService
from google.genai import types
from harness.agents import build_tool_surface
from harness.conversation import current_conversation_id, current_conversation_key
from harness.graph import create_root_workflow
from harness.mcp_client import McpExecutor, McpResolver
from harness.prefix_cache import PrefixCache
from harness.provenance import ProvenanceStore
from harness.settings import Settings
from harness.studio import StudioRunner
from harness.studio_state import StudioStore, make_recall
from harness.wiring import build_generative_nodes

# The store behind the recall seam: one module-level store, as one server
# process has one. `adk web` serves no composition route, so nothing else
# reads it.
_store = StudioStore()


class ConversationIdentity(BasePlugin):
    """Sets the studio's conversation contextvars for each `adk web` run.

    The identity has the shape `harness.studio` sets per request: the id is
    `"{principal}/{conversation_id}"` and the key is the
    `(principal, conversation_id)` pair, with the ADK user as the principal
    and the ADK session as the conversation. `before_run_callback` runs in the
    invocation's own task before the root node's task is created, so every
    node, dynamically scheduled child and MCP call in the run inherits the
    values, and a resume after an `adk_request_input` pause sets them again.
    """

    def __init__(self) -> None:
        super().__init__(name="conversation_identity")

    @override
    async def before_run_callback(
        self, *, invocation_context: InvocationContext
    ) -> types.Content | None:
        principal = invocation_context.user_id
        conversation_id = invocation_context.session.id
        current_conversation_id.set(f"{principal}/{conversation_id}")
        current_conversation_key.set((principal, conversation_id))
        return None


def _resolve_mcp_auth(settings: Settings) -> None:
    """Resolve the process-wide MCP auth policy, as `create_app` does.

    The mounts belong to the harness server, which resolved this policy when
    it built them. `common.auth.bearer_token()` returns the token those
    mounts require and is the only credential `McpResolver` and `McpExecutor`
    can present; until `apply_gate` runs in this process it returns `None`,
    every tool call goes out with no `Authorization` header, and the mounts
    answer 401.

    With `MCP_API_TOKEN` set the same value reaches both processes and either
    one can resolve it, which is the case the recipe requires. Unset, the
    harness server minted a token only it knows and nothing here can reach
    those mounts.

    `data_mcp.server.mcp` is one of the three mounts `create_app` gates. The
    policy is process-wide and first-caller-wins, so gating one resolves the
    token for all three, and this process builds no mount app whose route
    table the `.auth` assignment would matter to.
    """
    apply_gate(data_mcp, token=settings.mcp_api_token, disabled=settings.mcp_auth_disabled)


def build_runner(settings: Settings | None = None) -> StudioRunner:
    """The studio's `StudioRunner`, built the way `create_app` builds it."""
    resolved = settings or Settings()
    _resolve_mcp_auth(resolved)
    provenance = ProvenanceStore()
    executor = McpExecutor(settings=resolved, provenance=provenance)
    # Built once and threaded into both the `execute` node and
    # `build_generative_nodes`' planner, so the two compile against one set
    # of schemas read once, exactly as `create_app` does.
    tool_surface = build_tool_surface(settings=resolved)
    prefix_cache = PrefixCache() if resolved.studio_prefix_cache else None
    return StudioRunner(
        create_workflow=lambda recorder, surface_recorder: create_root_workflow(
            resolver=McpResolver(settings=resolved),
            executor=recorder,
            generative=build_generative_nodes(
                settings=resolved,
                model_name=resolved.model_name,
                surface=tool_surface,
                prefix_cache=prefix_cache,
            ),
            recall=make_recall(_store),
            tool_surface=surface_recorder,
        ),
        executor=executor,
        session_service=InMemorySessionService(),
        store=_store,
        tool_surface=tool_surface,
        provenance=provenance,
        settings=resolved,
    )


# Built once at import: the graph and the MCP tool surface are process-wide
# singletons on the served path too, and the app loader imports this module
# once.
_studio = build_runner()

# The `App` `adk web` serves: the graph, plus the identity plugin. A loader
# that finds `app` uses it as-is; one that finds only `root_agent` wraps the
# graph in an `App` of its own and loses the plugin.
app = _studio.runner.app.model_copy(
    update={"plugins": [ConversationIdentity(), *_studio.runner.app.plugins]}
)

# The graph alone, for `adk eval` and loaders that wrap it themselves.
root_agent = app.root_agent

__all__ = ["ConversationIdentity", "app", "build_runner", "root_agent"]
