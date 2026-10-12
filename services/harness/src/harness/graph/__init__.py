"""Root workflow for the DFO harness.

The workflow is the deterministic spine around the orchestrator agent:
routing, approval, retries and execution are code; only the orchestrator and
the planner (`PlannerModel`, implemented by `harness.planner_seam`) generate.
A turn runs `seed` (any datasets the user brought from Search) ->
`orchestrate` (the agent and its tools, `harness.orchestrator_tools`) ->
`route`, which takes the one action the agent asked for: `ask` puts its
question to the user, and `plan_start` hands its brief to `planner` ->
`approve` -> `execute`, whose result `plan_done` hands back to the agent.

Everything external is injected (`executor`, `tool_surface`,
`definition_surface`). The seams are async: ADK runs a sync function node
inline on the event loop, and the real implementations (`harness.mcp_client`)
call the loopback MCP mounts over HTTP on that same loop -- a sync seam would
block the loop and deadlock the loopback request.

Behavioral guarantees the topology encodes:

* Approval is a workflow interrupt with a stable id. Nothing downstream of
  `approve` exists on a path that skips it, and approval is version-bound:
  a plan amended since the reviewed version waits for a fresh approval.
* The `execute` node runs the approved plan's compiled workflow through
  `executor.run.execute_approved_plan`, which owns per-step retry, the
  per-capability transport policy, and the same `executor.classify.
  classify_refusal` policy `executor.run` uses for its own engine; the node
  itself only maps what comes back onto two edges. A refusal that carries a
  choice -- for the user or for a fresh discovery -- routes back to the
  planner with the diagnosis attached, so choosing the correction is a plan
  decision that goes back through the approval gate, never a retry the
  executor makes for itself. A refusal `classify_refusal` calls `STOP` is a
  dead end no replan reaches: the executor may not correct the call itself,
  and the user could not pick a narrower one from offered candidates
  either, so `execute` closes the plan immediately with the refusal's own
  remedy rather than spending a replan reproducing the identical failing
  call.
* Execution-triggered replanning is bounded: after `MAX_REPLANS` rounds the
  plan closes with what ran, naming the step that could not.

One approval gate: the workflow's own `approve` interrupt. The planner's
`submit_plan` tool (`planner.agent`) declares the plan ready for the user's
answer and gates nothing itself -- only this workflow's `execute` node,
reached downstream of `approve`, runs a step against a real server.
"""
