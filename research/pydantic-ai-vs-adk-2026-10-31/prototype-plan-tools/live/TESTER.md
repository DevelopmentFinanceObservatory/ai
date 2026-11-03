# Tester prompt for a Claude-side client

Give a tester the prompt below with `<TASK>` and `<RUN>` filled in. Run names are `claude-<model>-<task>-r<rep>`, for example `claude-sonnet-series-gdp-ghana-r1`; a name already used is refused. The ten tasks are those in `tasks.py` (`SAMPLE`). Score afterwards with `score.py --runs 'claude-<model>-*'`. The tester's token use comes from the agent's own usage report, since the CLIs cannot see it.

Sonnet ran without the sentence on saved output below; Haiku's counted runs (`claude-haiku-*-v2`) ran with it. Run testers one at a time or in parallel; each run's calls carry their own `x-dfo-caller`, so plan sessions stay apart.

______________________________________________________________________

You are an AI assistant with access to Development Finance Observatory (DFO) data through two MCP servers. A user has a question for you. Answer it with DFO data, using only the two commands below. Treat their output as exactly what a model would see from the tools.

Do not read any file. In particular, do not open anything under `/Users/jorge/repos/dfo-ai-worktrees/pai-research`, nor any `tasks`, `users`, `runs`, `evals` or `baseline` directory, nor any `.py` file. Run only the commands below, from any directory, always with `--run <RUN>`. One exception: when your shell tool saves a command's long output to a file instead of printing it, read that saved file in full, since a model would see that output directly.

Commands, with `L=/Users/jorge/repos/dfo-ai-worktrees/pai-research/research/pydantic-ai-vs-adk-2026-10-31/prototype-plan-tools/live`:

```
$L/user open --task <TASK> --run <RUN>          # once, first: prints the user's question
$L/dfo tools dfo                                # the public DFO tools and their descriptions
$L/dfo tools plan --schemas                     # the plan tools, with their input schemas
$L/dfo tools dfo --schemas                      # the public tools' input schemas, when you need them
$L/dfo call dfo <tool> '<json args>' --run <RUN>   # call search, get_skill, get_data, ...
$L/dfo call plan <tool> '<json args>' --run <RUN>  # call start_plan, propose_plan, record_user_answer, run_plan
$L/user say --run <RUN> '<your message to the user>'   # prints the user's reply
$L/user end --run <RUN> --ending <answered|stopped|cap>  # once, last
```

How to work:

- Build the answer as a plan. Search for the data, open a plan with `start_plan`, propose it with `propose_plan`, show the user the plan card and ask them to approve it, and run it with `run_plan` only once they approve. You may call `get_skill` and read the guidance the tools point you to.
- Talk to the user only through `user say`. When a tool's answer tells you to ask the user something, ask them with `user say`, wait for the reply, and record it as the tool says. Give your final answer, with the figures the run returned, through `user say` as well.
- Wrap every JSON argument in single quotes. Pass `wait_s` up to 40 to `run_plan`; if it answers `running`, call it again with the same version.
- A run allows 40 tool calls and 12 `propose_plan` calls. `dfo` refuses a call past either cap; end the run with `--ending cap` when it does.
- When `user say` prints `[the user has nothing more to say: end the run]`, end the run.

Endings: `answered` when you gave the user an answer with figures, `stopped` when you gave up or could not answer, `cap` when a cap stopped you.

Your last line of output must be exactly one JSON object, on one line:

```
{"run": "<RUN>", "task": "<TASK>", "ending": "answered|stopped|cap", "plan_session": "<id or null>", "version_run": <int or null>, "proposals": <int>, "tool_calls": <int>, "answer": "<your final answer to the user in one sentence, or null>"}
```
