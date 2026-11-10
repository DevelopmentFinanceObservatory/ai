# Parity of the planner on Pydantic AI with the planner on ADK

## Verdict

Every `evals/planner_entry` task ends the same way through both entries and every request each arm sends matches. 70 tasks, 79 plans, 149 cases, 230 requests per arm. Apart from the six wire differences `GoogleModel` makes (below), the arms send equal system instructions, contents, declarations and tool configs, 74 of the requests after a user's answer to an ask included. The check exits 0 on dev's tree, twice, with byte-identical `parity.json`. Each of the eleven `--break` controls exits 1.

The check found three differences, all closed in `agent/`: ADK's conversion of `BaseModel`-typed arguments, its answer to a missing mandatory argument, and the response with no parts.

## What runs

`parity.py` runs each case through `entry.run_entry` (ADK) and `pai_entry.run_pai_entry` (Pydantic AI), both under the real `PlannerSeam`, each against its own scripted google-genai client (`agent/scripted.py`) that records the request as it would go on the wire. No model call and no network.

Per task:

- A case for each plan the task states (79 in all): the plan is proposed and translated over the recording as production translates it, so the plan is declared ready and the run ends on `end_run`. Four plans are refused first by their task's recording; for those the script is the refused proposal, an ask, the user's answer and the proposal again (4 cases of 3 requests).
- An `ask-then-propose` case (70), which runs without the translation and ends where the eval's probe stops, at the first recorded plan: the planner asks, the user answers, and the plan is proposed in the request that answer resumes. Three tasks refuse the plan at this point on both arms (`plan_not_ready`), which is also compared.
- The cached prefix: both arms create one cache, with the same system instruction and declarations, and send the request against it with no prefix of its own.

Each case compares:

- how the entry ended: `scripted.outcome` (stop reason, plan, tool calls, each proposal, each tool response, model call count, budget, misses, error);
- the final session state, apart from what a run mints or stamps (below);
- the number of requests, and for each request the system instruction, every declaration, how the declarations are grouped, the tool config, the contents without call ids, and every other config field;
- the type of each item of each list the `propose_plan` body receives (`LIST_ARGUMENTS`), which shows whether ADK's `BaseModel` conversion ran.

Once, on the first task: the gap cases (`gap_cases`) and a thought summary (an unsigned thought and a signed call; the next request carries no summary and keeps the signature).

## Differences and how each closed

Requests differ only by `WIRE`, compared exactly, so a difference that disappears fails as a new one does. Sources are in `agent/NOTES.md`.

- Each declaration is a `Tool` of its own (ADK sends one `Tool` holding five).
- `reject_plan` carries `parameters_json_schema` with an empty object.
- A tool config `AUTO` is sent; ADK sends none.
- `response_modalities: ["TEXT"]`.
- `automatic_continuation` is absent on the Pydantic AI side.
- `Content-Type`, `User-Agent` and tracking headers of the library's own.

The comparison normalises, and reports anything else:

- Tool call ids Pydantic AI replays are left out of the contents; ADK sends none.
- `observed_at`, the time the replayed search stamps on each reading, and the values of `translator:resolved_at:*`.
- `translator:discovery_lock_token`, minted per run.
- `translator:discovery` records are keyed by an id minted per run; they are compared as a sorted list of their content without the `id`.
- `temp:` keys (`temp:planner_brief`): ADK keeps them out of its stored session, and Pydantic AI's state dict holds them. The brief reaches the model through the contents, which are compared.

Differences found by this check and closed in `agent/pai_planner.py`:

1. A missing mandatory argument. ADK answers `{"error": "Invoking `name()` failed as the following mandatory input parameters are not present: ..."}` and the run goes on; Pydantic AI called the body, which raised `KeyError`, and the run ended with no plan. `missing_arguments` makes ADK's answer, read off the ADK tool's `_get_mandatory_args`, before the body runs. Cases `missing-ask-message`, `missing-skill-id`, `missing-skill-path`.
1. A response with no parts. ADK ends the run; Pydantic AI retried with a second request (2 requests against 1). The agent's output type is now `str | None` (`OUTPUT_TYPE`), which Pydantic AI takes as a valid empty result, so the run ends. The same holds for a thought-only response. Cases `no-parts`, `no-parts-then-plan`, `thought-only`, and `text-only` as the control that a text answer already agreed.
1. `BaseModel`-typed arguments. ADK converts each item of `decisions`, `evidence_requirements`, `declined` and `answered` to its model and leaves an item that does not validate as the dict the model wrote. Pydantic AI passed raw dicts. No outcome differed: the 79 plans, a coercible plan, an invalid item and a bad `answered` all end the same way with raw values, because `submission.service._Patch` validates the lists. The difference is in what the body receives, and `submission.service` reads a dict where a refusal names the pins a proposal states (`isinstance(requirement, dict)` before `base_year_refusal`), so it can matter on a refusal this recording set does not reach. `convert_arguments` runs ADK's conversion (`FunctionTool._preprocess_args`) and the type probe shows the same classes on both arms, for example `[dict, EvidenceRequirement-model, ...]` on a payload with one invalid item. Cases `models-valid`, `models-coercible`, `models-invalid-item`, `models-bad-answered`.

Both conversions run through the ADK tool object `PlannerParts` already holds for its declaration. A production port without ADK tools needs its own: `model_validate` on each item of a `BaseModel`-list argument with the dict kept on failure, and the same missing-argument answer.

## Controls

Each control runs the full check with the Pydantic AI arm broken in one place. Run over all 70 tasks:

| `--break`     | Break                                                                            | How it fails                                                                                    |
| ------------- | -------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------- |
| `instruction` | instruction ends in `" X"`                                                       | `requests` differs in every case of 70 tasks, and in the prefix                                 |
| `declaration` | `propose_plan`'s description ends in a space                                     | same fields                                                                                     |
| `contents`    | the user's message loses its `instructions` note, so the two user contents merge | `requests` differs in `contents` in every case; the thought check differs                       |
| `tool-result` | `propose_plan`'s result gains a key                                              | `outcome` differs in every task; `requests` where a request follows                             |
| `end-run`     | `PlannerDeps.end_run` does nothing                                               | `outcome` and `request_count` differ in every task (a further request after the end)            |
| `state`       | the first key of the seeded state is removed                                     | `state` differs in every case                                                                   |
| `thoughts`    | thought summaries replayed (`keep_thoughts`)                                     | the thought check differs: the summary is in the second request                                 |
| `prefix`      | the cached prefix carries an `AUTO` tool config                                  | `prefix` differs in `tool_config` in every task                                                 |
| `arguments`   | `convert_arguments` passes raw values                                            | `argument_types` differs in every case and gap that proposes a plan, with no outcome difference |
| `mandatory`   | `missing_arguments` answers nothing                                              | the three `missing-*` gap cases: outcome, state, request count                                  |
| `no-parts`    | the output type is `str`                                                         | `no-parts`, `no-parts-then-plan`, `thought-only` differ in outcome and request count            |

## Private names and monkeypatches

Harness: those listed in `agent/NOTES.md`. ADK: `BaseTool._get_declaration`, `FunctionTool._preprocess_args` and `_get_mandatory_args` (`agent/pai_planner.py`). Pydantic AI: those in `agent/NOTES.md`.

`parity.py` replaces, and restores in a `finally`:

- `entry.build_seam` (through `scripted.adk_client`), pointing ADK's model at the scripted client;
- `entry.Probe.translate` (`translating`), so a recorded plan is translated and declared ready as production's is;
- `planner.agent.propose` and `pai_planner.propose` (`spying`), wrappers that note the types of the lists and call the original;
- the broken variants (`broken`).

It runs in an empty temporary directory with `STUDIO_API_TOKEN` blank so a developer's `.env` does not reach `Settings`. It loads none of the harness suite's fixtures, since translation reads only the recordings and the tool schemas.

## Not covered

- A real Gemini response. Both arms read the same scripted chunk; a streamed answer split over chunks, a signature on a text part and usage metadata beyond the scripted figures are not sent.
- Wire differences below the google-genai client: how each framework's HTTP layer retries, and the headers' values.
- Gap inputs are the first task's. A refusal that reads a dict item of `evidence_requirements` (`base_year_refusal`, which needs a part the user priced) is not reached by any recording; conversion is shown by the types the body receives, not by that refusal.
- A tool name the planner lacks, a budget cut mid-stream, the first-chunk resend and the cache 403/404 resend (`agent/NOTES.md`).
- Amendments, a pick's replan, `revise`, Stop and the progress relay: the entry runs a first draft and its resume only.
- The live smoke and the `validated` tool-config mode.

## Commands

From the worktree root:

```
PYDANTIC_AI_NO_BANNER=1 uv run --with 'pydantic-ai-slim[google]==2.66.0' --package harness \
    python research/pai-planner-live/parity/parity.py [--only ID,...] [--break NAME] [--out PATH]
```

It takes about 35 s, writes `parity.json` beside the script and exits 1 unless every difference is explained. `--break` takes `instruction`, `declaration`, `contents`, `tool-result`, `end-run`, `state`, `thoughts`, `prefix`, `arguments`, `mandatory`, `no-parts`.

```
cd research/pai-planner-live/agent
PYDANTIC_AI_NO_BANNER=1 uv run --with 'pydantic-ai-slim[google]==2.66.0' --package harness pytest . -q -p no:cacheprovider
uv run --package harness ruff format research/pai-planner-live/agent research/pai-planner-live/parity
uv run --package harness ruff check research/pai-planner-live/agent research/pai-planner-live/parity
```
