# orchestrator

The Orchestrator Agent answers a studio turn by calling tools itself. It
decides what a question needs -- search and fetch data, draw charts, ask the
user, load a skill, or hand a question that needs construction or synthesis
to the planner. It is a Pydantic AI agent on Gemini, run once per turn and
again after each hand-back.

`orchestrator` ships inside `harness` as a sibling top-level module
(`services/harness/src/orchestrator/`), on harness's dependencies, tests, and
CI. Its tests live in `services/harness/tests/orchestrator/`.

## What it does

`orchestrator_agent()` (`agent.py`) builds the agent on the tools it is given,
each declared from its signature and docstring (`declarations.py`) and run over
an `OrchestratorCall` (`call.py`) holding the turn's state and id. The harness
gives it these (`harness.orchestrator_tools`):

| Tool          | Behaviour                                                                                                                        |
| ------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| `search`      | Sends every part of the question to Search in one call, or `text` for Search to read itself; returns each part's reading         |
| `fetch`       | Retrieves one dataset, summing a family of indicators, and returns frame ids with their views                                    |
| `make_charts` | Draws charts from those frames on the canvas and returns what each holds and its reply figures, or why it failed                 |
| `ask_user`    | Records one to three questions for the user; the workflow pauses once for all of them and hands every answer back                |
| `plan`        | Records a brief and the outputs asked for; the plan path runs and hands back its frames and each ask's status                    |
| `get_skill`   | Loads a skill a search result named, or with `path` one file it links, through data-mcp                                          |
| `explain_dfo` | What DFO is and what can be asked, from the `dfo-faq` skill read through data-mcp (`harness.dfo_faq`)                            |
| `recall`      | An earlier turn's plan records or readings, or a canvas component's entry, in full; a typed refusal names the legal turns or ids |
| `remove_cards` | Takes the cards the user names off the canvas, or all of them for a request to clear it |

The model sees digests only: part readings, frame views, one-line summaries
and a drawn card's reply figures, with no rows. Each turn's first request
opens with the date, the canvas digest and the conversation digest, then the
user's message (`run.Opening`), so the instruction stays the same on every
call and its prefix is served from the shared cache (`harness.prefix_cache`).
Within a turn it sees its own calls and every hand-back.

## How a run goes

`run_orchestrator()` (`run.py`) drives one run on the model
`orchestrator_model()` (`model.py`) builds, and yields events as they happen:
answer text as it streams, each model response, each response's tool answers
with the state they wrote, and how the run ended. A `plan` or `ask_user` call
ends the run deferred (`HandedOff`), with the conversation to store
(`StoredHistory`) until the planner's result or the user's answer comes back
and the next run resumes it. A reply ends the run (`Answered`). The model
streams answer text as it arrives and holds a function call to its end, so a
call the turn's budget cuts runs no tool and ends on the bound's sentence
(`harness.gemini_wire`).
