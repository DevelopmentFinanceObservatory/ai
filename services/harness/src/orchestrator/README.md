# orchestrator

The Orchestrator Agent answers a studio turn by calling tools itself. It
decides what a question needs -- search and fetch data, draw charts, ask the
user, load a skill, or hand a question that needs construction or synthesis
to the planner. `harness.graph.create_root_workflow` wires it in when given one
(`STUDIO_ENGINE=orchestrator`).

`orchestrator` ships inside `harness` as a sibling top-level module
(`services/harness/src/orchestrator/`), on harness's dependencies, tests, and
CI. Its tests live in `services/harness/tests/orchestrator/`.

## What it does

`create_orchestrator_agent()` builds the `LlmAgent` on the tools it is given.
The harness gives it nine (`harness.orchestrator_tools`):

| Tool             | Behaviour                                                                                                                        |
| ---------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| `search`         | Sends the question and every part it names to Search in one call; returns each part's reading                                    |
| `fetch`          | Retrieves one dataset, summing a family of indicators, and returns frame ids with their views                                    |
| `make_charts`    | Draws charts from those frames on the canvas and returns each chart's facts, or why it failed                                    |
| `ask_user`       | Records a question for the user; the workflow pauses for the answer and hands it back                                            |
| `plan`           | Records a brief and the outputs asked for; the plan path runs and hands back its frames and each ask's status                    |
| `get_skill`      | Loads a skill a search result named, through data-mcp                                                                            |
| `get_skill_file` | Reads a file a loaded skill links, or the next page of a long one, through data-mcp                                              |
| `explain_dfo`    | What DFO is and what can be asked, from the `dfo-faq` skill read through data-mcp (`harness.dfo_faq`)                            |
| `recall`         | An earlier turn's plan records or readings, or a canvas component's entry, in full; a typed refusal names the legal turns or ids |

The model sees digests only: part readings, frame views, one-line summaries
and chart facts, never rows. Each request opens with the canvas digest and
the conversation digest from session state (`prepend_digests`), so the
instruction stays the same on every call and its prefix is served from the
shared cache (`harness.prefix_cache`). Within a turn it sees its own calls
and every hand-back (see "The orchestrator engine" in
`services/harness/README.md`).
