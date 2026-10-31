# Planner tools off ADK's `ToolContext`: scope

The planner's readers and the translator can stop reading ADK types in one change that preserves behaviour, made while the harness still runs on ADK. After it, the Pydantic AI port needs only an adapter that builds one record. Line references are to dev at 190c7a9c, after the upgrade to ADK 2.13.

## What the code reads

| Use                                                        | Sites | Where                                                                                    |
| ---------------------------------------------------------- | ----- | ---------------------------------------------------------------------------------------- |
| `tool_context.state`, the session's state slots            | 62    | 19 in `planner/agent.py`, 43 in `translator/tool.py`, `mechanical.py` and `discovery.py` |
| `tool_context.session.events`, the session's event history | 4     | `_session_steps` and three readers in `planner/agent.py`                                 |
| `tool_context.actions.skip_summarization`                  | 1     | `declare_ready` (`planner/agent.py:703`)                                                 |
| `tool_context.function_call_id`                            | 1     | `evals/planner_entry/entry.py:710`, the eval's translate stub                            |

State already sits behind our own protocol, `planner.plan.SessionState`, and `submission.service.SessionView` hides the readers from `submit`. The translator reads nothing but state.

Five readers answer what the user was asked and answered:

- `_elected_spec_ids`
- `_prices_chosen`
- `_answers_since_referral`
- `_combination_picks`
- `_user_spoke_since_recorded`

Dev has already started the neutral record this scope proposed. `_Step` (`planner/agent.py:959`) is one step of the session as the readers walk it: a user message, an ask, an answer paired with the nearest preceding ask sharing its id, or another tool's response. `_session_steps` builds the steps from ADK events. `_elected_spec_ids` and `_prices_chosen` walk steps. The other three still walk `tool_context.session.events` directly.

The readers read the whole ADK session, not only the planner's own history. The graph's `ask` node raises `adk_request_input` under the fixed id `"clarify"`. The approval card's amendment arrives as a response to `harness.graph.APPROVE_INTERRUPT`. The prototype's stand-in rebuilt events from the planner's own Pydantic AI messages (`adk_events` in `prototype/gemini/pai_planner.py`). That held for one planner entry but misses both of these in a full turn. Under our own loop, the loop must append them to the record.

## Options for the readers

1. Finish the neutral record (recommended, and now under way).

   - Make `_Step` public in an ADK-free module.
   - Port the three remaining walkers onto it.
   - Leave `_session_steps` as the one function that reads ADK events.

   It costs a port of three walkers whose pairing rules must survive exactly. It buys readers with no ADK types and one record the loop will own.

1. Keep ADK `Event` as the record, with the Pydantic AI loop building ADK events. This costs nothing now. It keeps `google.adk` as a dependency of the planner for its types, and the loop still has to produce events for the graph's asks.

1. Answers in state. Each answer is written to state when it arrives, and the readers become state reads. This is the smallest end state. It cannot land under ADK: a pausing long-running tool has no event to carry its state writes (the `_elected_spec_ids` docstring), and ADK builds the answer event, not our code. It is also a change of meaning, since tests pin the pairing and reset rules. It fits after the loop owns every write, on top of option 1.

## The change

The work is two implementer slices in the worktree `dfo-ai-worktrees/planner-call-record`, branch `refactor/planner-call-record`. The second slice runs after the first.

1. Planner:
   - `planner/call.py` holds `Step`, a `PlannerCall` protocol (`state`, `call_id`, `steps()`), the five readers, and `CallSession` (today's `ToolContextSession`). It imports nothing from ADK.
   - `planner/agent.py` keeps the ADK binding: `AdkCall`, and `_session_steps` as the only reader of ADK events.
   - `declare_ready` stops setting `skip_summarization`. The ADK binding sets it when the answer carries `summary`.
   - Tool declarations stay byte for byte as they are.
   - `harness/planner_seam.py`'s `_RevisionSession` follows the rename.
1. Translator:
   - `tool_context: ToolContext` becomes `state: SessionState` throughout.
   - `Translate` takes a `PlannerCall`, so the eval's stub keeps its call id.

A guard test parses the imports of `planner/call.py`, `submission/` and `translator/` and fails on any `google.adk` import.

Out of scope:

- `harness/planner_seam.py`, which is ADK workflow-node code the turn loop replaces;
- the orchestrator's tools (`harness/orchestrator_tools.py`), the same job on the other agent.

## How to prove it changes nothing

- **Existing tests.** The tests still build fake ADK events and go through `AdkCall`, so they test the ADK path end to end with their assertions unchanged. An implementer who finds an assertion that must change stops and reports it.
- **Scripted plans.** `evals/planner_entry/validate.py`, with every scripted plan ending as it did on dev.
- **The import guard.** It must fail on a planted ADK import.

Then the Pydantic AI port builds `Step`s in the loop and drops the prototype's stand-in `ToolContext`.
