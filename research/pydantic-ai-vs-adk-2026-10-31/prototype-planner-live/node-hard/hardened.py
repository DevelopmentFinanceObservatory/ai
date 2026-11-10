"""The planner node of `node/` with one fix of `agent/` switched off, for a control.

`node/` and `agent/` carry every fix the hard cases called for. `hard_planner` builds the
same planner (`pai_node.pai_planner`) and, for a control, leaves one of three out:

- `hold=False`: the client passes each chunk to Pydantic AI as it arrives, so a call the
  budget cuts after a function call was streamed still runs the tool.
- `unknown_tools=False`: Pydantic AI's own retry text answers a call to a tool the planner
  lacks, and its one retry ends the run on the second such call.
- `instrument=False`: no generation, tool or agent span.

With no switch it is `pai_node.pai_planner`; `live-turns/driver.py` builds the planner
through it.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import aclosing
from typing import Any

from google.adk.models.llm_response import LlmResponse
from google.genai import types
from harness.planner_seam import PlannerSeam
from harness.turn_budget import CLOSED_BY_KEY, bounded
from pai_node import PaiPlannerSeam, pai_planner
from pai_planner import (
    _RAW_CHUNK,
    OUTPUT_TYPE,
    RETRIES,
    THINKING_SETTINGS,
    CallHooks,
    PlannerClient,
    PlannerDeps,
    PlannerModel,
    PlannerParts,
    ToolWiring,
    _as_chunk,
    adk_tool_not_found,
    drop_unsigned_thoughts,
    genai_client,
    google_model,
    planner_tools,
)
from planner.agent import AGENT_NAME
from planner.settings import AgentSettings
from pydantic_ai import Agent
from pydantic_ai.capabilities.instrumentation import Instrumentation
from pydantic_ai.capabilities.process_history import ProcessHistory


class StreamingClient(PlannerClient):
    """`PlannerClient` that hands Pydantic AI each chunk as `bounded` yields it."""

    async def _stream(
        self, *, model: str, contents: Any, config: Any
    ) -> AsyncIterator[types.GenerateContentResponse]:
        synthetic, self.pending.response = self.pending.response, None
        if synthetic is not None:
            yield _as_chunk(synthetic)
            return
        usage: types.GenerateContentResponseUsageMetadata | None = None
        closed = False
        responses = bounded(
            lambda: self._raw(model=model, contents=contents, config=config), stream=True
        )
        async with aclosing(responses) as stream:
            async for response in stream:
                metadata = response.custom_metadata or {}
                if _RAW_CHUNK in metadata:
                    chunk: types.GenerateContentResponse = metadata[_RAW_CHUNK]
                    usage = chunk.usage_metadata or usage
                    yield chunk
                elif CLOSED_BY_KEY in metadata:
                    closed = True
                    self.hooks.after(None, response)
                    replaced = self.hooks.seam._after_planner_model(None, response)
                    yield _as_chunk(replaced or response)
        if not closed:
            whole = LlmResponse(usage_metadata=usage, partial=False)
            self.hooks.after(None, whole)
            self.hooks.seam._after_planner_model(None, whole)


def hard_planner(
    seam: PlannerSeam,
    *,
    client: Any,
    surface: Any,
    translate: Any,
    prefix_cache: Any = None,
    hold: bool = True,
    unknown_tools: bool = True,
    instrument: bool = True,
    **options: Any,
) -> PaiPlannerSeam:
    """`pai_node.pai_planner`, with the fixes `hold`, `unknown_tools` and `instrument`
    each turned off by `False`."""
    if hold and unknown_tools and instrument:
        return pai_planner(
            seam,
            client=client,
            surface=surface,
            translate=translate,
            prefix_cache=prefix_cache,
            **options,
        )
    model_name = AgentSettings().planner_model_name
    client_class = PlannerClient if hold else StreamingClient
    planner_client = client_class(client or genai_client(model_name), CallHooks(seam=seam))
    model = PlannerModel(
        google_model(model_name, planner_client.client),
        client=planner_client,
        prefix_cache=prefix_cache,
    )
    parts = PlannerParts.of(seam)
    capabilities: list[Any] = [ProcessHistory(drop_unsigned_thoughts)]
    if instrument:
        capabilities.append(Instrumentation())
    if unknown_tools:
        capabilities.append(ProcessHistory(adk_tool_not_found([t.name for t in parts.tools])))
    agent = Agent(
        model,
        instructions=parts.instruction,
        tools=planner_tools(seam, parts, ToolWiring(surface=surface, translate=translate)),
        deps_type=PlannerDeps,
        output_type=OUTPUT_TYPE,
        model_settings=THINKING_SETTINGS,
        capabilities=capabilities,
        name=AGENT_NAME,
        retries=RETRIES if unknown_tools else 1,
    )
    return PaiPlannerSeam(seam, agent, **options)
