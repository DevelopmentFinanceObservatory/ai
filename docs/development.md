# Maintainer notes

Setup, configuration, and upgrade notes for ONE maintainers. These steps need access to the `dev-fin-observatory` Google Cloud project. Contributor guidance is in [CONTRIBUTING.md](../CONTRIBUTING.md), and deploys are covered in [deploy/README.md](../deploy/README.md).

## Local setup with cloud access

You need [`just`](https://github.com/casey/just), [`uv`](https://docs.astral.sh/uv/), the `gcloud` CLI, and access to the `dev-fin-observatory` project.

```bash
gcloud auth login
gcloud auth application-default login
just setup    # env file, dependencies, pre-commit hooks
just dev      # harness on :8080
```

`just setup` is safe to re-run and skips what is already done. It checks prerequisites, builds `.env`, syncs the workspace members into the shared `.venv`, and installs pre-commit hooks. `just doctor` checks tooling, gcloud auth, which config keys resolved, and whether the workspace is synced, and prints the command that fixes anything it finds.

## Configuration

Local config lives in `.env` at the workspace root. It is generated and gitignored, and `just env` builds it from `.env.example`. Two kinds of value live there.

Secrets come from Secret Manager in `dev-fin-observatory`:

| Key                                          | Secret                                                          |
| -------------------------------------------- | --------------------------------------------------------------- |
| `GOOGLE_API_KEY`                             | `dfo-dev-gemini-api-key` (shared with `dfo-dcp` and `dfo-apis`) |
| `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY` | `dfo-dev-langfuse-public-key`, `dfo-dev-langfuse-secret-key`    |

Everything else (`MODEL_NAME`, `DFO_APIS_URL`, `LANGFUSE_BASE_URL`, `PORT`) is a committed default in `.env.example`, so git distributes it. `just env` carries new keys into an existing `.env` on the next run, and `just doctor` reports when yours has fallen behind.

Each key resolves in order: a value already in `.env`, then an exported shell variable, then Secret Manager. Existing values survive, so re-running `just env` after an edit is safe, and a one-off override needs no file edit:

```bash
MODEL_NAME=gemini-3-pro just dev
```

With any Langfuse credential missing, the run starts anyway. Traces go to `OTEL_EXPORTER_OTLP_ENDPOINT` when it is set, and tracing is off otherwise. `/healthz` reports the resolved state as `telemetry: langfuse`, `otel` or `off`.

To rotate a credential, add a new version to the secret in Secret Manager, delete that line from your `.env`, and run `just env`.

### Adding a config key

1. Add it to `.env.example`. Give it a default value if it is not a secret.
1. Add the field to `BaseServiceSettings` (`packages/common`) if every service reads it, or to that service's own `Settings` subclass if one does.
1. For a secret, create it in Secret Manager as `dfo-dev-<name>` and add a `resolve` line to the `env` recipe in the `justfile`.

## Committed baselines

`data-mcp`'s digest corpus (`services/data-mcp/tests/digest_corpus.py`) measures the model-facing character cost of every semantic status the search door can render, against a committed baseline and a per-status budget. The budgets fail only when a status grows past its budget. A deliberate digest change that moves the recorded numbers means running `just digest-baseline`, so the committed record stays current. Reviewers read the regenerated diff to `digest_baseline.json`. `just analysis-digest-baseline` does the same for `analysis-mcp`.

`services/harness/tests/context_baseline.json` records what the planner model is handed. Regenerate it with `just context-baseline` after a deliberate change to the planner's instructions, tool surface, or history handling.

## Definition store in deployment

`data-mcp`'s definition store persists the definitions `search` mints until the first `get_data` registers them permanently. With `DEFINITION_STORE_BUCKET` unset it uses an in-memory store instead of GCS. That is fine for local development and for a single Cloud Run instance. With several instances behind one service, every other instance refuses `get_data` with `definition_not_held` for an id this instance minted, so the lifespan in `server.py` logs a warning when the bucket is unset and `K_SERVICE` (Cloud Run's service marker) is set.

## Upgrading `google-adk`

The harness pins `google-adk==2.9.1` (`services/harness/pyproject.toml`) and depends on four of its private modules: `flows/llm_flows/contents` (`_is_event_belongs_to_branch`), `flows/llm_flows/_fencing` (`_present_other_agent_message`), `tools/agent_tool` (`_SingleTurnAgentTool`) and `workflow/_llm_agent_wrapper` (`prepare_llm_agent_context`). Nothing imports them, so an upgrade that changes their behaviour produces no import error. The planner's context isolation rests on all four: its run is stamped with a branch, and ADK's branch filter drops the translator's events from what the planner is handed. The branch comparison direction, the sub-branch naming and the `single_turn` scope inheritance are each one of those modules' contract.

ADK's filter is fail-open. `_is_event_belongs_to_branch` opens with `if not invocation_branch or not event.branch: return True`, so it keeps an event whose branch is empty. The planner is isolated while `override_branch=_PLANNER_BRANCH` reaches `ctx.run_node`, which happens at one call site, in `services/harness/src/harness/planner_seam.py`. After each run, and on the way out of one that suspends for a planner-side ask, `harness.planner_seam._check_branches` checks that every planner event carries `_PLANNER_BRANCH` and every translator event a branch strictly under it, and that a completed run appended a planner event at all. It raises `PlannerSeamError` otherwise. A dropped keyword, a second call site without it, or an ADK release that stops honouring it therefore fails the run. The relay ceiling below is the measured consequence.

`services/harness/tests/test_context_budget.py` is the first test file to fail after a bump, and `just check` runs it with the rest of the suite. Three of its tests are the early warning:

- `test_no_relayed_foreign_agent_text_reaches_the_planner` holds the fenced foreign-agent relay in the planner's context at exactly zero characters.
- `test_the_relay_check_measures_a_relay_that_is_there` proves that reading can go non-zero, so the zero is a measurement.
- `test_the_cached_prefix_is_stable_across_a_run` holds the planner's static prefix byte-identical across a run. Gemini's implicit caching matches on that prefix, and so does the shared explicit cache (`harness.prefix_cache.prefix_key`).

A failure after a bump means ADK changed one of those behaviours. The failing test's docstring names the mechanism it asserts, and `.venv/lib/python3.*/site-packages/google/adk` is the installed source to check that claim against. An accommodation to the new behaviour that moves a measured number means running `just context-baseline`, and reviewers read the diff of `services/harness/tests/context_baseline.json`.

## Branches and deploys

`dev` is the trunk. CI (`lint`, `test`, `docker`) runs on every ready pull request into `dev` or `main`, and again on each push to `dev`. Merging to `dev` deploys the dev Cloud Run service. A release is a pull request from `dev` into `main`, and `deploy-prod.yml` deploys production on every push to `main`.

CI builds one image and updates the existing Cloud Run service. The `infra` repository owns the service definition, including its runtime secrets. The `.env` flow above is for local development only. [deploy/README.md](../deploy/README.md) has the full division of responsibilities and the GitHub Environment setup.
