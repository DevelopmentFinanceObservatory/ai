# evals/rolling_anchor

Seeded rolling plans over synthetic series, counted as admitted and as translated. Translation restates a plan from the newest year its series hold, and the run checks against that year. Each plan is admitted by `Plan` itself, recorded and compiled against the live servers' schemas, and its run counts from the newest year at least half the world's series hold (`dfo_frame_grammar.latest_anchor`). Translation's restatement, the compile of the restated plan and the run's check are the harness's own functions. It needs no model and no network.

```bash
uv run --package harness python evals/rolling_anchor/run.py
uv run --package harness python evals/rolling_anchor/run.py --count 3000 --seed 7
```

The default run admits 15,211 plans in about four minutes, most of it compiling, and prints, for each world and side, how many runs read data, ran empty, missed, were refused at translation by a plan rule or by the compiler, or were refused at run. The three worlds are the one each plan was planned over, one where newer years reach some series, and one where a lagging series joins or a series holding the newest year stops. `generator.py` says what counts as empty and what as missed.

CI runs 500 of the same plans (`services/harness/tests/test_rolling_anchor_generator.py`).
